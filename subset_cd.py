#!/usr/bin/env python3
import argparse, os, random, shutil
from pathlib import Path
from tqdm import tqdm

def read_list(list_path: Path):
    with list_path.open("r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

def write_list(names, list_path: Path):
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w") as f:
        f.write("\n".join(names) + ("\n" if names else ""))

def choose_subset(names, frac, rng):
    if frac >= 1.0:  # take all
        return names[:]
    k = max(1, int(round(len(names) * frac)))
    # reproducible sample without replacement
    return rng.sample(names, k) if k < len(names) else names[:]

def transfer_one(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "link":
        # hard link if same filesystem; fallback to copy if fails
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        # relative symlink to keep tree portable
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    else:
        raise ValueError(f"mode must be copy|link|symlink, got {mode}")

def process_split(src_root: Path, dst_root: Path, split: str, frac: float, rng: random.Random):
    src_split = src_root / split
    dst_split = dst_root / split

    list_src = src_split / "list" / f"{split}.txt"
    names = read_list(list_src)
    subset = choose_subset(names, frac, rng)

    # copy/link files A, B, label for each name
    pbar = tqdm(subset, desc=f"{split}: {len(subset)}/{len(names)} ({frac*100:.1f}%)")
    for name in pbar:
        # las imágenes A/B tienen el mismo nombre de archivo que la label
        for sub in ("A", "B", "label"):
            src = src_split / sub / name
            dst = dst_split / sub / name
            if not src.exists():
                raise FileNotFoundError(f"Falta archivo: {src}")
            transfer_one(src, dst, args.mode)

    # escribir list/{split}.txt con los mismos nombres (labels)
    write_list(subset, dst_split / "list" / f"{split}.txt")

def main(args):
    rng = random.Random(args.seed)
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    # validación mínima de estructura
    for split in ("train", "val", "test"):
        req = [src_root / split / sub for sub in ("A","B","label","list")]
        for p in req:
            if not p.exists():
                raise FileNotFoundError(f"Esperaba {p} en el dataset origen")

        lst = src_root / split / "list" / f"{split}.txt"
        if not lst.exists():
            raise FileNotFoundError(f"Falta {lst}")

    if dst_root.exists() and any(dst_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"[ERROR] {dst_root} ya existe y no está vacío. Usa --overwrite para reutilizarlo.")

    # limpiar destino si se pide overwrite
    if dst_root.exists() and args.overwrite:
        shutil.rmtree(dst_root)

    # fracciones (por split) se interpretan como porcentaje sobre *cada split original*
    # para lograr el ~10% total: 0.06, 0.02, 0.02
    fracs = {"train": args.frac_train, "val": args.frac_val, "test": args.frac_test}

    for split, frac in fracs.items():
        process_split(src_root, dst_root, split, frac, rng)

    print("\n✓ Subconjunto creado en:", dst_root)
    print("Estructura:")
    print("  {dst}/")
    print("    train/{A,B,label,list/train.txt}")
    print("    val/{A,B,label,list/val.txt}")
    print("    test/{A,B,label,list/test.txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crear subconjunto 10% (6/2/2) de LEVIR-CD o DSIFN-CD.")
    parser.add_argument("--src", required=True, help="Raíz del dataset original (p.ej. ./LEVIR-CD256 o ./DSIFN-CD256)")
    parser.add_argument("--dst", required=True, help="Raíz del dataset de salida (p.ej. ./LEVIR-CD256-10p)")
    parser.add_argument("--frac-train", type=float, default=0.06, dest="frac_train",
                        help="Fracción de train (default: 0.06)")
    parser.add_argument("--frac-val", type=float, default=0.02, dest="frac_val",
                        help="Fracción de val (default: 0.02)")
    parser.add_argument("--frac-test", type=float, default=0.02, dest="frac_test",
                        help="Fracción de test (default: 0.02)")
    parser.add_argument("--mode", choices=["copy","link","symlink"], default="link",
                        help="Cómo materializar archivos en el subset (copy|link|symlink). Default: link")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para muestreo reproducible")
    parser.add_argument("--overwrite", action="store_true", help="Borrar destino si existe")
    args = parser.parse_args()
    main(args)
