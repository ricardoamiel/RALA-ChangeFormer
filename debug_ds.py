# debug_ds.py
import os, numpy as np, torch
from datasets.CD_dataset import CDDataset

# >>> Ajusta estas rutas/parámetros <<<
ROOT = "./DSIFN-CD256"  # carpeta raíz del dataset (la que tiene train/val/test)
IMG_SIZE = 256

def show_stats(sample, tag):
    L = sample['L']            # tensor [1,H,W] o similar
    A = sample['A']            # imagen A (normalizada)
    B = sample['B']            # imagen B (normalizada)

    # Uniques en la máscara
    u, c = torch.unique(L, return_counts=True)
    u = u.cpu().tolist(); c = c.cpu().tolist()
    h, w = L.shape[-2], L.shape[-1]

    # % de píxeles positivos
    L01 = L.clone()
    if L01.max() > 1:
        # si está en 0/255 lo vemos aquí
        pos = (L01 == 255).float().mean().item()
    else:
        pos = (L01 == 1).float().mean().item()

    print(f"[{tag}] shape={h}x{w}  uniques={list(zip(u,c))}  pos%={pos*100:.2f}")
    return u

def inspect_split(split, n=3, is_train=False, label_transform=None, to_tensor=True):
    print(f"\n===== SPLIT: {split}  (is_train={is_train}, label_transform={label_transform}) =====")
    ds = CDDataset(
        root_dir=ROOT,
        img_size=IMG_SIZE,
        split=split,
        is_train=is_train,             # True -> activa augs; False -> camino de test/val
        label_transform=label_transform,
        to_tensor=to_tensor
    )
    # Muestra algunos ejemplos
    for idx in range(min(n, len(ds))):
        sample = ds[idx]   # esto ejecuta __getitem__ y transform()
        u = show_stats(sample, f"{split}:{idx}")

if __name__ == "__main__":
    # 1) Val con camino de test (is_train=False) — sanity check
    inspect_split("val", n=4, is_train=False, label_transform=None, to_tensor=True)

    # 2) Test con camino de test
    inspect_split("test", n=4, is_train=False, label_transform=None, to_tensor=True)

    # 3) (Opcional) Fuerza la normalización previa usando tu bandera existente
    inspect_split("test", n=4, is_train=False, label_transform="norm", to_tensor=True)
