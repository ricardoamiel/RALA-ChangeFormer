# tools/attn_grid_no_thop.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.flops_counter import count_flops
from models.ChangeFormer import ChangeFormerV6, ChangeFormerV7

# Convención: usamos "GFLOPs" ≈ MACs/1e9 (1 MAC = 1 FLOP).
# Si prefieres 2*MACs, cambia multiply_by_two=True y/o la conversión.

BATCHES = [1, 4, 8, 16] #32, 64]
SIZES = [(256, 256), (512, 512), (1024, 1024)]
SIZE_LABELS = [f"{h}" for (h, w) in SIZES]  # 256, 512, 1024

def measure_attn_macs(model, x1, x2, multiply_by_two=False):
    total_macs, by_bucket, _ = count_flops(
        model, (x1, x2), multiply_by_two=multiply_by_two, verbose=False
    )
    return by_bucket.get("attn", 0.0)  # MACs del bucket de atención

def to_gflops(macs, multiply_by_two=False):
    factor = 2.0 if multiply_by_two else 1.0
    return (macs * factor) / 1e9

def main():
    torch.set_grad_enabled(False)

    # Modelos
    model_attn   = ChangeFormerV6()  # softmax
    model_linear = ChangeFormerV7()  # linear
    model_attn.eval()
    model_linear.eval()

    # Matrices [len(SIZES), len(BATCHES)]
    A = np.zeros((len(SIZES), len(BATCHES)), dtype=np.float64)  # GFLOPs attn normal
    L = np.zeros_like(A)                                        # GFLOPs attn lineal

    # Recorremos TODAS las combinaciones
    for i, (H, W) in enumerate(SIZES):
        for j, B in enumerate(BATCHES):
            x1 = torch.randn(B, 3, H, W)
            x2 = torch.randn(B, 3, H, W)

            macs_A = measure_attn_macs(model_attn,   x1, x2, multiply_by_two=False)
            macs_L = measure_attn_macs(model_linear, x1, x2, multiply_by_two=False)

            A[i, j] = to_gflops(macs_A, multiply_by_two=False)
            L[i, j] = to_gflops(macs_L, multiply_by_two=False)

            print(f"[H=W={H}, B={B}]  AttnNorm={A[i,j]:.3f} GFLOPs | AttnLinear={L[i,j]:.3f} GFLOPs")

    # Diferencia y/o ratio
    D = A - L               # ahorro absoluto (GFLOPs)
    R = np.divide(A, L, out=np.ones_like(A), where=L>0)  # ratio (>=1 si lineal ahorra)

    # ---- Guardar CSVs
    import csv, os
    os.makedirs("outputs", exist_ok=True)
    def save_csv(path, arr, header_prefix):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{header_prefix}\\Batch"] + BATCHES)
            for i, lbl in enumerate(SIZE_LABELS):
                w.writerow([lbl] + [f"{arr[i, j]:.6f}" for j in range(len(BATCHES))])

    save_csv("outputs/attn_gflops_normal.csv", A, "Size")
    save_csv("outputs/attn_gflops_linear.csv", L, "Size")
    save_csv("outputs/attn_gflops_diff_normal_minus_linear.csv", D, "Size")
    save_csv("outputs/attn_gflops_ratio_normal_over_linear.csv", R, "Size")

    # ---- Heatmaps (1 figura por gráfico; sin estilos ni colores específicos)
    def heatmap(data, title, outpath):
        plt.figure()
        im = plt.imshow(data, aspect='auto')
        plt.title(title)
        plt.xlabel("Batch size")
        plt.ylabel("Lado de imagen (H=W)")
        plt.xticks(range(len(BATCHES)), [str(b) for b in BATCHES])
        plt.yticks(range(len(SIZE_LABELS)), SIZE_LABELS)
        cbar = plt.colorbar(im)
        cbar.set_label("GFLOPs")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)

    heatmap(A, "Atención (Softmax) — GFLOPs", "outputs/heatmap_attn_softmax.png")
    heatmap(L, "Atención (Lineal) — GFLOPs",  "outputs/heatmap_attn_linear.png")
    heatmap(D, "Ahorro (Softmax − Lineal) — GFLOPs", "outputs/heatmap_attn_diff.png")
    heatmap(R, "Ratio (Softmax / Lineal)", "outputs/heatmap_attn_ratio.png")

    print("\nArchivos guardados en ./outputs/:")
    print(" - attn_gflops_normal.csv")
    print(" - attn_gflops_linear.csv")
    print(" - attn_gflops_diff_normal_minus_linear.csv")
    print(" - attn_gflops_ratio_normal_over_linear.csv")
    print(" - heatmap_attn_softmax.png")
    print(" - heatmap_attn_linear.png")
    print(" - heatmap_attn_diff.png")
    print(" - heatmap_attn_ratio.png")

if __name__ == "__main__":
    main()
