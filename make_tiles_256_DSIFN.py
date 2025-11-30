#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import os

# === CONFIG ===
SPLIT = "test"
SRC = Path(f"DSIFN-CD-BENCH/{SPLIT}")       # dataset original (512x512): A/*.jpg, B/*.jpg, label/*.png
DST = Path(f"DSIFN-CD-BENCH256/{SPLIT}")   # tiles 256x256: todo en .png
TILE = 256

def ensure_dirs(root: Path):
    for sub in ["A", "B", "label", "list"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def tile_image(img_path: Path, out_dir: Path, base_name: str):
    """Divide una imagen (p.ej. 512x512) en tiles 256x256 y guarda como base_1.png, base_2.png, ..."""
    img = Image.open(img_path)
    W, H = img.size
    assert W % TILE == 0 and H % TILE == 0, f"{img_path} no es múltiplo de {TILE}: {W}x{H}"

    tile_num = 1
    names = []
    for r in range(0, H, TILE):
        for c in range(0, W, TILE):
            patch = img.crop((c, r, c + TILE, r + TILE))
            out_name = f"{base_name}_{tile_num}.png"
            patch.save(out_dir / out_name)
            names.append(out_name)
            tile_num += 1
    return names

def main():
    ensure_dirs(DST)
    list_file = SRC / "list" / f"{SPLIT}.txt"

    # demo.txt contiene nombres base con extensión (p.ej. 0.jpg)
    with open(list_file) as f:
        names = [x.strip() for x in f if x.strip()]

    all_tiles = []

    for name in names:
        base = os.path.splitext(name)[0]  # "0" de "0.jpg"

        # A y B: ya transformados a png
        for sub in ["A", "B"]:
            img_path = SRC / sub / f"{base}.png"
            if not img_path.exists():
                print(f"⚠️ No existe {img_path}")
                continue
            tiles = tile_image(img_path, DST / sub, base)

        # label: prioritariamente .png (también aceptamos .jpg o .tif por si hay mezcla)
        lbl_path_png = SRC / "label" / f"{base}.png"
        lbl_path_jpg = SRC / "label" / f"{base}.jpg"
        lbl_path_tif = SRC / "label" / f"{base}.tif"
        if lbl_path_png.exists():
            tile_image(lbl_path_png, DST / "label", base)
        elif lbl_path_jpg.exists():
            tile_image(lbl_path_jpg, DST / "label", base)
        elif lbl_path_tif.exists():
            tile_image(lbl_path_tif, DST / "label", base)

        all_tiles.extend(tiles)

    # genera lista demo.txt con todos los tiles .png
    with open(DST / "list" / f"{SPLIT}.txt", "w") as f:
        for n in all_tiles:
            f.write(n + "\n")

    print(f"Generados {len(all_tiles)} tiles de 256x256 en {DST}")

if __name__ == "__main__":
    main()
