#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------- Ajustes mínimos --------
# Ejemplo: checkpoints_training/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_...
RUN_DIR = Path("checkpoints_training_OG_3/.")  # <-- cambia esto al folder del run si quieres

OUT_DIR = RUN_DIR / "metrics_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_npy_safe(path):
    try:
        arr = np.load(path)
        # Asegura 1D si viene como (N,) o (N,1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr
    except Exception as e:
        print(f"[WARN] No pude cargar {path}: {e}")
        return None

def try_plot_curve(y, title, ylabel, fname):
    if y is None or len(y) == 0:
        return False
    x = np.arange(1, len(y)+1)
    plt.figure(figsize=(7,4))
    plt.plot(x, y, marker='o', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()
    return True

def parse_log_for_metrics(log_path):
    """
    Intenta extraer por época:
    - train_loss, val_loss
    - acc, miou, f1, precision, recall
    Regex flexible: busca floats tras los nombres de métricas (case-insensitive).
    Devuelve DataFrame con columnas disponibles.
    """
    if not log_path.exists():
        print(f"[INFO] No existe {log_path}")
        return pd.DataFrame()

    # Patrones comunes (ajustables a tu log)
    # Ejemplos de líneas esperadas (no tienen que estar todas):
    # Epoch 12/200 ... train_loss: 0.231 val_loss: 0.245 Acc: 0.912 mIoU: 0.621 F1: 0.765 Prec: 0.78 Rec: 0.75
    epoch_pat   = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.I)
    loss_tr_pat = re.compile(r"(train[_\s-]*loss)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    loss_va_pat = re.compile(r"(val[_\s-]*loss)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    acc_pat     = re.compile(r"(acc(?:uracy)?)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    miou_pat    = re.compile(r"(miou)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    f1_pat      = re.compile(r"(f1)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    prec_pat    = re.compile(r"(prec(?:ision)?)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    rec_pat     = re.compile(r"(rec(?:all)?)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)

    records = {}
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            me = epoch_pat.search(line)
            if not me:
                continue
            ep = int(me.group(1))
            rec = records.get(ep, {"epoch": ep})

            def pick(pat, key):
                m = pat.search(line)
                if m:
                    rec[key] = float(m.group(2))

            pick(loss_tr_pat, "train_loss")
            pick(loss_va_pat, "val_loss")
            pick(acc_pat,     "acc")
            pick(miou_pat,    "miou")
            pick(f1_pat,      "f1")
            pick(prec_pat,    "precision")
            pick(rec_pat,     "recall")

            records[ep] = rec

    if not records:
        print("[INFO] No se detectaron métricas con el patrón esperado en log.txt")
        return pd.DataFrame()

    df = pd.DataFrame(sorted(records.values(), key=lambda r: r["epoch"]))
    return df

def main():
    print(f"[INFO] Run dir: {RUN_DIR}")
    # 1) Intentar cargar npy de acc
    train_acc = load_npy_safe(RUN_DIR / "train_acc.npy")
    val_acc   = load_npy_safe(RUN_DIR / "val_acc.npy")

    # 2) Plot acc si existen
    t_ok = try_plot_curve(train_acc, "Train Accuracy", "Accuracy", "train_acc.png")
    v_ok = try_plot_curve(val_acc,   "Val Accuracy",   "Accuracy", "val_acc.png")

    # 3) Mejor epoch por val_acc
    if val_acc is not None and len(val_acc):
        best_ep = int(np.argmax(val_acc) + 1)
        best_val = float(val_acc[best_ep-1])
        print(f"[OK] Mejor epoch por val_acc: {best_ep}  (val_acc={best_val:.4f})")
        with open(OUT_DIR / "best_by_val_acc.json", "w") as f:
            json.dump({"best_epoch": best_ep, "best_val_acc": best_val}, f, indent=2)

    # 4) Parsear log.txt para más métricas
    df_log = parse_log_for_metrics(RUN_DIR / "log.txt")
    if not df_log.empty:
        df_log.to_csv(OUT_DIR / "metrics_epochwise.csv", index=False)
        # graficar lo que haya
        for col, ylabel in [
            ("train_loss", "Loss"),
            ("val_loss",   "Loss"),
            ("acc",        "Accuracy"),
            ("miou",       "mIoU"),
            ("f1",         "F1"),
            ("precision",  "Precision"),
            ("recall",     "Recall"),
        ]:
            if col in df_log.columns:
                try_plot_curve(df_log[col].values, f"{col} vs epoch", ylabel, f"{col}.png")

        # curva doble Loss (si hay train y val)
        if "train_loss" in df_log.columns and "val_loss" in df_log.columns:
            plt.figure(figsize=(7,4))
            plt.plot(df_log["epoch"], df_log["train_loss"], label="train_loss", marker='o')
            plt.plot(df_log["epoch"], df_log["val_loss"],   label="val_loss",   marker='o')
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train/Val Loss")
            plt.legend(); plt.grid(True, ls="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(OUT_DIR / "loss_train_val.png", dpi=150); plt.close()

        # curva doble Acc (si hay acc en log y val_acc.npy distinto)
        if "acc" in df_log.columns and val_acc is not None:
            plt.figure(figsize=(7,4))
            plt.plot(df_log["epoch"], df_log["acc"], label="acc(log)", marker='o')
            x = np.arange(1, len(val_acc)+1)
            plt.plot(x, val_acc, label="val_acc.npy", marker='o')
            plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Acc from log vs val_acc.npy")
            plt.legend(); plt.grid(True, ls="--", alpha=0.3)
            plt.tight_layout(); plt.savefig(OUT_DIR / "acc_log_vs_npy.png", dpi=150); plt.close()
    else:
        print("[INFO] Sin métricas en log.txt; sólo se graficaron npy si existían.")

    print(f"[DONE] Gráficos y CSV en: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
