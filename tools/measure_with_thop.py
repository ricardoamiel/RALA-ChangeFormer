#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import torch

from thop import profile
from models.ChangeFormer import ChangeFormerV6, ChangeFormerV7

# -------------------- utilidades de formato --------------------
def fmt_units(x, unit=""):
    # x es float (MACs o FLOPs o params)
    if x >= 1e12:
        return f"{x/1e12:.3f} T{unit}"
    if x >= 1e9:
        return f"{x/1e9:.3f} G{unit}"
    if x >= 1e6:
        return f"{x/1e6:.3f} M{unit}"
    if x >= 1e3:
        return f"{x/1e3:.3f} k{unit}"
    return f"{x:.0f} {unit}"

def to_flops(macs):
    return 2.0 * float(macs)

# -------------------- construcción del modelo --------------------
def build_model(tag: str, device: str):
    tag = tag.lower()
    if tag in ["v6", "changeformerv6", "og"]:
        net = ChangeFormerV6()
    elif tag in ["v7", "changeformerv7", "rala"]:
        net = ChangeFormerV7()
    else:
        raise ValueError(f"Modelo no reconocido: {tag}. Usa v6 | v7")
    net.eval().to(device)
    return net

# -------------------- inputs totales (A y B) --------------------
def dummy_inputs_total(batch: int, img_size: int, device: str):
    x1 = torch.randn(batch, 3, img_size, img_size, device=device)
    x2 = torch.randn(batch, 3, img_size, img_size, device=device)
    return x1, x2

# -------------------- profiling total --------------------
def profile_total(model, x1, x2):
    with torch.no_grad():
        macs, params = profile(model, inputs=(x1, x2), verbose=False)
    flops = to_flops(macs)
    return macs, flops, params

# -------------------- breakdown por bloque del encoder --------------------
def encoder_block_shapes(img_size: int, embed_dims):
    """
    Devuelve [(stage_name, H, C), ...] usando strides típicos [4,8,16,32].
    """
    S = img_size
    return [
        ("block1", S // 4,  embed_dims[0]),
        ("block2", S // 8,  embed_dims[1]),
        ("block3", S // 16, embed_dims[2]),
        ("block4", S // 32, embed_dims[3]),
    ]

def find_blocks_encoder(model):
    """
    Devuelve referencia al encoder y su lista de ModuleList por stage.
    """
    # En ChangeFormerV6/V7 el encoder se llama Tenc_x2 con atributos block1..block4
    enc = model.Tenc_x2
    blocks = [
        ("block1", enc.block1),
        ("block2", enc.block2),
        ("block3", enc.block3),
        ("block4", enc.block4),
    ]
    return enc, blocks

def profile_encoder_blocks(model, img_size: int, device: str):
    """
    Mide MACs/params por bloque del encoder llamando a forward(x, H, W) con:
      x: (1, H*W, C)
    """
    results = []  # filas: dicts
    embed_dims = getattr(model, "embed_dims", [64, 128, 320, 512])
    stage_specs = encoder_block_shapes(img_size, embed_dims)
    _, blocks = find_blocks_encoder(model)

    with torch.no_grad():
        for (stage_name, stage_H, stage_C), (blk_name, blk_list) in zip(stage_specs, blocks):
            assert stage_name == blk_name, "Desfase en nombres de stage vs encoder"
            N = stage_H * stage_H  # suponemos cuadrado
            for idx, m in enumerate(blk_list):
                # input tokenizado (B, N, C)
                x_tok = torch.randn(1, N, stage_C, device=device)
                macs, params = profile(m, inputs=(x_tok, stage_H, stage_H), verbose=False)
                flops = to_flops(macs)

                # Intenta identificar tipo de atención (MHSA vs RALA)
                attn_type = getattr(m, "attn", None).__class__.__name__ if hasattr(m, "attn") else "NA"
                if "LinearAttention" in attn_type:
                    kind = "RALA"
                elif "Attention" in attn_type:
                    kind = "MHSA"
                else:
                    kind = attn_type

                results.append({
                    "name": f"Tenc_x2.{stage_name}.{idx}",
                    "type": kind,
                    "H": stage_H,
                    "W": stage_H,
                    "C": stage_C,
                    "MACs": float(macs),
                    "FLOPs": float(flops),
                    "params": float(params),
                })

    return results

# -------------------- CSV --------------------
def maybe_write_csv(path, rows_total, rows_blocks):
    if path is None:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Totales
        w.writerow(["section","model","batch","img_size","MACs","FLOPs","params"])
        for r in rows_total:
            w.writerow(["total", r["model"], r["batch"], r["img_size"],
                        r["MACs"], r["FLOPs"], r["params"]])
        # Bloques
        w.writerow([])
        w.writerow(["section","name","type","H","W","C","MACs","FLOPs","params"])
        for rb in rows_blocks:
            w.writerow(["encoder_block", rb["name"], rb["type"], rb["H"], rb["W"], rb["C"],
                        rb["MACs"], rb["FLOPs"], rb["params"]])

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="v6 | v7")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--csv", type=str, default=None, help="ruta CSV opcional")
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = build_model(args.model, device)

    print(f"== Measuring total {args.model.upper()} on {device} ==")
    x1, x2 = dummy_inputs_total(args.batch, args.img_size, device)
    macs, flops, params = profile_total(model, x1, x2)
    print(f"Total MACs: {fmt_units(macs)} | FLOPs: {fmt_units(flops)} | Params: {fmt_units(params)}")

    # Breakdown por bloque (encoder)
    print("\n== Per-block encoder breakdown ==")
    rows_blocks = profile_encoder_blocks(model, args.img_size, device)
    tot_macs_blocks = sum(r["MACs"] for r in rows_blocks)
    tot_flops_blocks = sum(r["FLOPs"] for r in rows_blocks)
    tot_params_blocks = sum(r["params"] for r in rows_blocks)

    print("name,type,H,W,C,MACs,FLOPs,params")
    for r in rows_blocks:
        print(f'{r["name"]},{r["type"]},{r["H"]},{r["W"]},{r["C"]},'
              f'{fmt_units(r["MACs"], "MACs")},{fmt_units(r["FLOPs"], "FLOPs")},{fmt_units(r["params"])}')

    print("-" * 60)
    print(f"Encoder blocks sum -> MACs: {fmt_units(tot_macs_blocks)} | FLOPs: {fmt_units(tot_flops_blocks)} | params: {fmt_units(tot_params_blocks)}")

    # CSV opcional
    rows_total = [{
        "model": args.model,
        "batch": args.batch,
        "img_size": args.img_size,
        "MACs": macs,
        "FLOPs": flops,
        "params": params
    }]
    maybe_write_csv(args.csv, rows_total, rows_blocks)

if __name__ == "__main__":
    main()
