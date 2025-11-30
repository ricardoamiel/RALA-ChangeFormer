#!/usr/bin/env python3
# tools/analyze_changeformer.py
import argparse
from typing import Dict, List
import os, datetime, warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count

# ==============================
#  FLOPs handlers (upsample)
# ==============================
def _upsample_flops_handle_mod(mod_or_func, inputs, outputs):
    """Para módulos/funciones Python (nn.Upsample, F.interpolate): firma (module, inputs, outputs)."""
    x_out = outputs[0]
    N, C, H, W = x_out.shape
    mode = getattr(mod_or_func, "mode", None)
    # Aproximación típica: bilinear/bicubic/trilinear ~4*N*C*H*W; nearest/area ~0
    k = 0 if mode in ("nearest", "area") else 4
    return int(N * C * H * W * k)

def _upsample_flops_handle_aten(inputs, outputs):
    """Para ATen bilinear/bicubic/trilinear: firma (inputs, outputs). Asumimos k≈4."""
    x_out = outputs[0]
    N, C, H, W = x_out.shape
    return int(N * C * H * W * 4)

def _zero_flops_handle_mod(mod_or_func, inputs, outputs):
    return 0

def _zero_flops_handle_aten(inputs, outputs):
    return 0

# ==============================
#  Carga de modelos
# ==============================
def build_model(arch: str, input_nc: int, output_nc: int, embed_dim: int):
    tried = []
    if arch.lower() == "v6":
        # Intento 1: ruta típica del repo
        try:
            from models.ChangeFormer import ChangeFormerV6
            return ChangeFormerV6(input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim)
        except Exception as e:
            tried.append(f"models.ChangeFormer.ChangeFormerV6 -> {e}")
        # Intento 2: variante de nombre de archivo
        try:
            from models.ChangeFormer import ChangeFormerV6
            return ChangeFormerV6(input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim)
        except Exception as e:
            tried.append(f"models.changeformer_v6.ChangeFormerV6 -> {e}")
        raise RuntimeError("No pude importar ChangeFormerV6:\n  - " + "\n  - ".join(tried))
    elif arch.lower() == "v7":
        try:
            from models.ChangeFormer import ChangeFormerV7
            return ChangeFormerV7(input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim)
        except Exception as e:
            tried.append(f"models.ChangeFormer.ChangeFormerV7 -> {e}")
        try:
            from models.ChangeFormer import ChangeFormerV7
            return ChangeFormerV7(input_nc=input_nc, output_nc=output_nc, embed_dim=embed_dim)
        except Exception as e:
            tried.append(f"models.changeformer_v7.ChangeFormerV7 -> {e}")
        raise RuntimeError("No pude importar ChangeFormerV7:\n  - " + "\n  - ".join(tried))
    else:
        raise ValueError(f"--arch debe ser v6 o v7, recibido: {arch}")

# ==============================
#  Árbol jerárquico del modelo
# ==============================
class Node:
    def __init__(self, name: str, module: nn.Module):
        self.name = name
        self.module = module
        self.children: List["Node"] = []
        self.self_params = 0
        self.self_flops = 0      # FLOPs exclusivos (este módulo)
        self.total_params = 0
        self.total_flops = 0     # FLOPs acumulados (módulo + hijos)

def build_tree(model: nn.Module) -> Node:
    root = Node("", model)
    name_to_node = {"": root}
    for name, module in model.named_modules():
        if name == "":
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = name_to_node[parent_name]
        node = Node(name, module)
        parent.children.append(node)
        name_to_node[name] = node
    return root

def accumulate_stats(root: Node, params_by_tensor: Dict[str, int], flops_by_module: Dict[str, int]):
    from collections import defaultdict as dd
    module_param_counts = dd(int)
    for fullkey, cnt in params_by_tensor.items():
        modname = fullkey.rsplit(".", 1)[0] if "." in fullkey else ""
        module_param_counts[modname] += cnt

    def dfs(node: Node):
        node.self_params = module_param_counts.get(node.name, 0)
        node.self_flops = flops_by_module.get(node.name, 0)
        for ch in node.children:
            dfs(ch)
        node.total_params = node.self_params + sum(c.total_params for c in node.children)
        node.total_flops = node.self_flops + sum(c.total_flops for c in node.children)

    dfs(root)

def pct(x, total):
    return 0.0 if total == 0 else 100.0 * (x / total)

def pretty_type(m: nn.Module) -> str:
    return m.__class__.__name__

def print_tree(node: Node, total_params: int, total_flops: int, indent: int = 0,
               file=None, tree_mode: str = "exclusive"):
    """tree_mode: 'exclusive' usa self_flops; 'inclusive' usa total_flops."""
    if node.name != "":
        mod = node.module
        head = "  " * indent + f"({node.name.split('.')[-1]}): {pretty_type(mod)}("
        use_flops = node.self_flops if tree_mode == "exclusive" else node.total_flops
        p_pct = pct(node.total_params, total_params)
        f_pct = pct(use_flops, total_flops)
        print(
            f"{head}\n{'  '*indent}  {node.total_params/1e6:.3f} M, {p_pct:6.3f}% Params, "
            f"{use_flops/1e9:.3f} GFLOPs, {f_pct:6.3f}% FLOPs, ",
            file=file
        )
    for ch in node.children:
        print_tree(ch, total_params, total_flops,
                   indent + (0 if node.name == "" else 1),
                   file=file, tree_mode=tree_mode)
    if node.name != "":
        print(f"{'  '*indent})", file=file)

# ==============================
#  Resumen por operador
# ==============================
OP_BUCKETS = {
    "conv": {"aten::conv2d", "aten::_convolution", "aten::convolution"},
    "bn": {"aten::batch_norm"},
    "relu": {"aten::relu_", "aten::relu"},
    "gelu": {"aten::gelu"},
    "linear": {"aten::addmm", "aten::mm", "aten::matmul", "aten::linear", "aten::bmm"},
    "pool": {"aten::max_pool2d", "aten::avg_pool2d", "aten::adaptive_avg_pool2d"},
    "norm": {"aten::layer_norm", "aten::native_layer_norm"},
    "upsample": {"aten::upsample_bilinear2d", "aten::upsample_nearest2d", "aten::upsample_bicubic2d"},
}

def summarize_by_operator(fa: FlopCountAnalysis, total_flops: int, file=None):
    byop = fa.by_operator()
    from collections import defaultdict as dd
    bucket_totals = dd(int)
    for op, fl in byop.items():
        placed = False
        for bname, opset in OP_BUCKETS.items():
            if op in opset:
                bucket_totals[bname] += fl
                placed = True
                break
        if not placed:
            bucket_totals["other"] += fl

    print("\n---- Resumen por tipo de operador ----", file=file)
    for bname, fl in sorted(bucket_totals.items(), key=lambda x: -x[1]):
        print(f"{bname:10s}: {fl/1e9:8.3f} GFLOPs  ({pct(fl, total_flops):6.2f}%)", file=file)
    print("--------------------------------------\n", file=file)

# ==============================
#  Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, choices=["v6", "v7"], default="v6")
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--input_nc", type=int, default=3)
    ap.add_argument("--output_nc", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1, help="Batch size para el análisis de FLOPs")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--include_upsample", action="store_true",
                    help="Si se pasa, contamos FLOPs de upsample/interpolate (~4*N*C*H*W para bilinear/bicubic).")
    ap.add_argument("--tree_mode", type=str, choices=["exclusive", "inclusive"], default="exclusive",
                    help="exclusive: % por FLOPs propios del módulo; inclusive: % acumulando hijos.")
    ap.add_argument("--suppress_warnings", action="store_true",
                    help="Oculta UserWarnings de PyTorch (p.ej. floor division).")
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.simplefilter("ignore", UserWarning)

    # Carpeta y nombre de reporte
    os.makedirs("reports", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/flops_{args.arch}_{args.h}x{args.w}_{'ups' if args.include_upsample else 'noup'}_{args.tree_mode}_{ts}.txt"

    # Modelo + dummy inputs
    model = build_model(args.arch, args.input_nc, args.output_nc, args.embed_dim)
    model.eval().to(args.device)
    x1 = torch.randn(args.bs, args.input_nc, args.h, args.w, device=args.device)
    x2 = torch.randn(args.bs, args.input_nc, args.h, args.w, device=args.device)

    # Análisis de FLOPs
    fa = FlopCountAnalysis(model, (x1, x2))

    # Handlers para upsample/interpolate
    if args.include_upsample:
        # Python-level
        fa.set_op_handle(nn.Upsample, _upsample_flops_handle_mod)
        fa.set_op_handle(F.interpolate, _upsample_flops_handle_mod)
        # ATen
        try: fa.set_op_handle("aten::upsample_nearest2d", _zero_flops_handle_aten)  # nearest ≈ 0
        except Exception: pass
        for aten_op in ("aten::upsample_bilinear2d", "aten::upsample_bicubic2d",
                        "aten::upsample_linear1d", "aten::upsample_trilinear3d"):
            try: fa.set_op_handle(aten_op, _upsample_flops_handle_aten)
            except Exception: pass
    else:
        fa.set_op_handle(nn.Upsample, _zero_flops_handle_mod)
        fa.set_op_handle(F.interpolate, _zero_flops_handle_mod)
        for aten_op in ("aten::upsample_bilinear2d", "aten::upsample_nearest2d",
                        "aten::upsample_bicubic2d", "aten::upsample_linear1d",
                        "aten::upsample_trilinear3d"):
            try: fa.set_op_handle(aten_op, _zero_flops_handle_aten)
            except Exception: pass

    # Cómputo
    p_counts = parameter_count(model)           # dict param_tensor_name -> count
    total_params = sum(p_counts.values())
    total_flops = fa.total()                    # total FLOPs contados
    flops_by_module = fa.by_module()            # FLOPs exclusivos por submódulo

    # Árbol y acumulados
    root = build_tree(model)
    accumulate_stats(root, p_counts, flops_by_module)

    # Guardar reporte
    with open(filename, "w") as f:
        print(f"Guardando reporte completo en: {filename}\n", file=f)

        print("\n==============================", file=f)
        print(f"Arch  : ChangeFormer{args.arch.upper()}", file=f)
        print(f"Input : (x1,x2) => ({args.input_nc}, {args.h}, {args.w}) c/u", file=f)
        print(f"Params: {total_params/1e6:.3f} M", file=f)
        print(f"FLOPs : {total_flops/1e9:.3f} GFLOPs "
              f"(include_upsample={args.include_upsample}, tree_mode={args.tree_mode})", file=f)
        print("==============================\n", file=f)

        print(f"ChangeFormer{args.arch.upper()}(", file=f)
        print_tree(root, total_params, total_flops, file=f, tree_mode=args.tree_mode)
        print(")", file=f)

        summarize_by_operator(fa, total_flops, file=f)

        print("\n---- Top 10 módulos por FLOPs (exclusivos) ----", file=f)
        top10 = sorted(flops_by_module.items(), key=lambda x: -x[1])[:10]
        for name, fl in top10:
            print(f"{name:70s} {fl/1e9:8.3f} GFLOPs ({pct(fl, total_flops):6.2f}%)", file=f)
        print("-----------------------------------------------\n", file=f)

        unsupported = fa.unsupported_ops()
        if unsupported:
            print("⚠️  Ops no soportadas por fvcore (FLOPs no contados):", file=f)
            for k, v in unsupported.items():
                print(f"  - {k}: {v}", file=f)
            print("\nNota: suelen ser ops element-wise/activaciones; "
                  "el coste dominante está en matmul/addmm y lineales.", file=f)

    # Resumen a consola
    print(f"✅ Reporte guardado en {filename}")
    print(f"Total Params: {total_params/1e6:.3f} M | Total FLOPs: {total_flops/1e9:.3f} GFLOPs "
          f"(include_upsample={args.include_upsample}, tree_mode={args.tree_mode})")

if __name__ == "__main__":
    main()