#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

# === CONFIG ===
DATASET_ROOT = Path("DSIFN-CD")  # ajusta si tu ruta base es distinta
#SPLITS = ["train", "val", "test"]
SPLITS = ["train","test"]
def extract_num(filename: str):
    """Extrae el número del nombre (ej. 'train_12.png' -> 12)."""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

def main():
    for split in SPLITS:
        split_dir = DATASET_ROOT / split / "A"
        list_dir = DATASET_ROOT / split / "list"
        list_dir.mkdir(parents=True, exist_ok=True)

        # Archivos PNG del directorio A
        img_names = sorted(
            [p.name for p in split_dir.glob("*.png")],
            key=extract_num
        )

        list_path = list_dir / f"{split}.txt"
        with open(list_path, "w") as f:
            for name in img_names:
                f.write(name + "\n")

        print(f"[✓] {split}: {len(img_names)} archivos guardados en {list_path}")

if __name__ == "__main__":
    main()

