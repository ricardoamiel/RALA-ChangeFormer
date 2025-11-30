#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image

SPLIT = "test"
SRC = Path(f"LEVIR-CD-BENCH/{SPLIT}")      # dataset original (1024x1024)
DST = Path(f"LEVIR-CD-BENCH256/{SPLIT}")  # dataset tileado a 256x256
TILE = 256

def ensure_dirs(root: Path):
    for sub in ["A", "B", "label", "list"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def tile_one(img_path: Path, out_dir: Path, base: str):
    # abre y valida tamaño
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    assert W % TILE == 0 and H % TILE == 0, f"{img_path} no es múltiplo de {TILE}: {W}x{H}"

    tiles = []
    for r in range(0, H, TILE):
        for c in range(0, W, TILE):
            patch = img.crop((c, r, c+TILE, r+TILE))
            name = f"{base}_{r:04d}_{c:04d}.png"
            patch.save(out_dir / name)
            tiles.append(name)
    return tiles

def main():
    assert (SRC / "A").is_dir() and (SRC / "B").is_dir(), f"Estructura esperada: A/, B/, [label/], list/{SPLIT}.txt"
    ensure_dirs(DST)

    # lee la lista original (nombres base, ej: test_1.png, test_2.png)
    list_file = SRC / "list" / f"{SPLIT}.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"No existe {list_file}")

    # acumula nombres de parches para escribir nueva lista
    out_names = []

    with list_file.open("r") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            base = os.path.splitext(name)[0]

            a_path = SRC / "A" / name
            b_path = SRC / "B" / name
            lbl_path = SRC / "label" / name

            assert a_path.exists(), f"Falta {a_path}"
            assert b_path.exists(), f"Falta {b_path}"
            has_label = lbl_path.exists()

            # tilea A y B
            tiles_a = tile_one(a_path, DST / "A", base)
            tiles_b = tile_one(b_path, DST / "B", base)

            # Chequeo de consistencia
            assert tiles_a == tiles_b, f"Patch mismatch A/B para {name}"

            # tilea label si existe (manteniendo 0/255)
            if has_label:
                # mantener como binario; no convertir a RGB (mejor L)
                lbl = Image.open(lbl_path)
                W, H = lbl.size
                assert W % TILE == 0 and H % TILE == 0, f"{lbl_path} no es múltiplo de {TILE}"
                for nm in tiles_a:
                    # r,c vienen del nombre
                    parts = nm.split("_")
                    r = int(parts[-2])
                    c = int(parts[-1].split(".")[0])
                    patch = lbl.crop((c, r, c+TILE, r+TILE))
                    # Si tu máscara está en 0/1, conviértela a 0/255:
                    if patch.mode != "L":
                        patch = patch.convert("L")
                    # normaliza a 0/255 si hace falta
                    patch = patch.point(lambda v: 255 if v >= 128 else 0)
                    patch.save(DST / "label" / nm)

            out_names.extend(tiles_a)

    # escribe nueva lista demo.txt con TODOS los parches
    with (DST / "list" / f"{SPLIT}.txt").open("w") as f:
        for nm in out_names:
            f.write(nm + "\n")

    print(f"✔ Hecho. Dataset tileado en: {DST}")
    print(f"   A: {len(list((DST/'A').glob('*.png')))} parches")
    print(f"   B: {len(list((DST/'B').glob('*.png')))} parches")
    if (DST / "label").exists():
        print(f"   label: {len(list((DST/'label').glob('*.png')))} parches")
    print(f"   Lista: {(DST/'list'/f'{SPLIT}.txt')}")

if __name__ == "__main__":
    main()
