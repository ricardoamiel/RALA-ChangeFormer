#!/usr/bin/env python3
import argparse, re, os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1) Cargar modelo ChangeFormer V6 / V7
# ----------------------------------------------------------------------
def load_model(arch, ckpt_path, device):
    if arch.lower() == "v6":
        from models.ChangeFormer import ChangeFormerV6 as CF
    elif arch.lower() == "v7":
        from models.ChangeFormer import ChangeFormerV7 as CF
    else:
        raise ValueError("arch debe ser v6 o v7")

    model = CF()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

# ----------------------------------------------------------------------
# 2) Registrar hooks en capas seleccionadas por regex
# ----------------------------------------------------------------------
def register_attn_hooks(model, layer_regex):
    hooks = []
    attn_maps = {}

    pattern = re.compile(layer_regex)

    for name, module in model.named_modules():
        if pattern.match(name):

            target = module.attn if hasattr(module, "attn") else module

            def hook(m, inp, out, name=name):
                if hasattr(m, "_last_attn"):
                    attn_maps[name] = m._last_attn.detach().cpu()
                else:
                    attn_maps[name] = out.detach().cpu()

            hooks.append(target.register_forward_hook(hook))

    return hooks, attn_maps

# ----------------------------------------------------------------------
# 3) Guardar heatmap con ejes visibles y PDF recortado
# ----------------------------------------------------------------------
def save_attn_heatmap(attn, out_path, title=""):
    """
    attn: [h,Nq,Nk] o [Nq,Nk]
    Guarda el heatmap + ejes (0..Nq, 0..Nk) recortado a PDF.
    """
    if attn.ndim == 3:
        attn = attn.mean(0)

    attn = attn.numpy()
    H, W = attn.shape  # ej: 64 x 512 o lo que sea tu feature map

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attn, cmap="magma", aspect='auto')

    # Ticks numéricos
    ax.set_xticks([0, W//2, W-1])
    ax.set_xticklabels([0, W//2, W])
    ax.set_yticks([0, H//2, H-1])
    ax.set_yticklabels([0, H//2, H])

    ax.set_xlabel("Tokens (0 → {})".format(W))
    ax.set_ylabel("Dimensión / Canal (0 → {})".format(H))
    ax.set_title(title)

    fig.tight_layout(pad=0.1)

    # Forzar PDF
    if not out_path.lower().endswith(".pdf"):
        out_path = out_path.rsplit(".", 1)[0] + ".pdf"

    # Recorte exacto al eje
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(
        out_path,
        bbox_inches=extent,
        pad_inches=0,
        format="pdf",
        dpi=300
    )

    plt.close(fig)

# ----------------------------------------------------------------------
# 4) Extraer atención
# ----------------------------------------------------------------------
def extract_attention(model, imgA, imgB, img_size, attn_maps, mode, device):

    x1 = imgA.unsqueeze(0).to(device)
    x2 = imgB.unsqueeze(0).to(device)

    with torch.no_grad():
        _ = model(x1, x2)

    return attn_maps

# ----------------------------------------------------------------------
# 5) Cargar imagen
# ----------------------------------------------------------------------
from PIL import Image
import torchvision.transforms as T

def load_image(path, img_size):
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
    return tfm(img)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--imgA", type=str, required=True)
    ap.add_argument("--imgB", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--layers", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["center", "token"], default="center")
    ap.add_argument("--heads", type=str, default="all")
    ap.add_argument("--outdir", type=str, default="attn_vis")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Usando device: {device}")

    os.makedirs(args.outdir, exist_ok=True)

    model = load_model(args.arch, args.checkpoint, device)

    hooks, attn_maps = register_attn_hooks(model, args.layers)

    imgA = load_image(args.imgA, args.img_size)
    imgB = load_image(args.imgB, args.img_size)

    attn_maps = extract_attention(model, imgA, imgB,
                                  args.img_size, attn_maps,
                                  args.mode, device)

    for name, attn in attn_maps.items():
        out_path = os.path.join(args.outdir, f"{name.replace('.', '_')}.pdf")
        print(f"Guardando {out_path}")

        if attn.ndim == 4:
            attn = attn[0]

        save_attn_heatmap(attn, out_path, title=name)

    for h in hooks:
        h.remove()

    print("🎉 Listo: visualizaciones generadas en", args.outdir)


if __name__ == "__main__":
    main()

