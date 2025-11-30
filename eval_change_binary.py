#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import tifffile as tiff
import csv

# -------------------- IO utils --------------------
def read_mask(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in [".tif", ".tiff"]:
        arr = tiff.imread(str(p))
    else:
        arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def binarize(arr, threshold=128):
    a = arr.astype(np.float32)
    if a.max() <= 1.0:
        a = a * 255.0
    return (a >= threshold).astype(np.uint8)

# -------------------- Metrics --------------------
def metrics_bin(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    eps = 1e-10
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    iou  = tp / (tp + fp + fn + eps)
    oa   = (tp + tn) / (tp + fp + fn + tn + eps)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=float(prec), recall=float(rec),
                f1=float(f1), iou=float(iou), oa=float(oa))

# -------------------- Naming helpers --------------------
def detect_dataset(pred_dir_str: str) -> str:
    name = pred_dir_str.lower()
    if "dsifn" in name: return "dsifn"
    if "levir" in name: return "levir"
    return "unknown"

def detect_size(pred_dir_str: str, img_size: int) -> str:
    name = pred_dir_str.lower()
    for s in ("256", "512", "1024"):
        if s in name: return s
    if img_size in (256, 512, 1024):
        return str(img_size)
    return "full"

def detect_model(pred_dir_str: str, override: str = None) -> str:
    if override:
        return override.lower()
    name = pred_dir_str.lower()
    # señales típicas
    if "changeformer" in name: return "changeformer"
    if "bit" in name:         return "bit"
    if "transunet" in name:   return "transunet"
    if "unet" in name:        return "unet"
    # busca “MY_predict_CD_XXXX” o “predict_XXXX”
    parts = [p for p in name.replace("-", "_").split("/") if p]
    for p in parts:
        if "predict" in p and "cd_" in p:
            tail = p.split("cd_")[-1]
            return tail
    return "model"

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Carpeta con predicciones (.png/.tif)")
    ap.add_argument("--gt_dir", required=True, help="Carpeta con GT binaria")
    ap.add_argument("--list_file", required=True, help="TXT con nombres (uno por línea)")
    ap.add_argument("--threshold", type=float, default=128)
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--model_name", default="", help="Forzar nombre de modelo en el folder (opcional)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)

    dataset = detect_dataset(str(pred_dir))
    size    = detect_size(str(pred_dir), args.img_size)
    model   = detect_model(str(pred_dir), override=args.model_name.strip() or None)

    results_base = Path("results_eval") / f"{dataset}_{size}_{model}"
    results_base.mkdir(parents=True, exist_ok=True)

    out_csv     = results_base / "cd_eval_per_image.csv"
    summary_txt = results_base / "cd_eval_summary.txt"
    plot_path   = results_base / "cd_metrics_bar.png"

    names = [line.strip() for line in open(args.list_file) if line.strip()]
    print(names[0:2])
    per_image = []
    TP=FP=FN=TN=0

    for name in names:
        pred_path = pred_dir / name
        gt_path   = gt_dir / name
        if not gt_path.exists():
            alt = gt_dir / (Path(name).stem + ".tif")
            if alt.exists():
                gt_path = alt
        pred = read_mask(pred_path)
        gt   = read_mask(gt_path)
        pred_b = binarize(pred, args.threshold)
        gt_b   = binarize(gt, 128)
        m = metrics_bin(pred_b, gt_b)
        per_image.append((name, m))
        TP += m["tp"]; FP += m["fp"]; FN += m["fn"]; TN += m["tn"]

    eps = 1e-10
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    iou  = TP / (TP + FP + FN + eps)
    oa   = (TP + TN) / (TP + FP + FN + TN + eps)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","precision","recall","f1","iou","oa","tp","fp","fn","tn"])
        for name, m in per_image:
            w.writerow([name, m["precision"], m["recall"], m["f1"], m["iou"], m["oa"],
                        m["tp"], m["fp"], m["fn"], m["tn"]])

    with open(summary_txt, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Size: {size}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Images: {len(per_image)}\n")
        f.write(f"TP={TP} FP={FP} FN={FN} TN={TN}\n")
        f.write(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} IoU={iou:.4f} OA={oa:.4f}\n")

    print(f"=== Global ({dataset}_{size}_{model}) ===")
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} IoU={iou:.4f} OA={oa:.4f}")
    print(f"Saved results in: {results_base}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.bar(["precision","recall","f1","iou","oa"], [prec,rec,f1,iou,oa])
            plt.ylim(0,1)
            plt.title(f"{dataset}_{size}_{model} metrics")
            plt.savefig(plot_path, bbox_inches="tight")
            print(f"Plot saved: {plot_path}")
        except Exception as e:
            print("No se pudo generar gráfico:", e)

if __name__ == "__main__":
    main()
