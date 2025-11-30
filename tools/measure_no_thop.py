# tools/measure_no_thop.py
import torch
# from your_pkg import build_model_attn, build_model_linear
from tools.flops_counter import count_flops

from models.ChangeFormer import ChangeFormerV6, ChangeFormerV7

# Construye tus modelos
model_attn   = ChangeFormerV6()     # usa Block (Attention)
model_linear = ChangeFormerV7()   # usa BlockLinear (LinearAttention)

# Input realista que active SR si la hay
B, C, H, W = 1, 3, 256, 256    # ajusta a tu caso
x1 = torch.randn(B, C, H, W)
x2 = torch.randn(B, C, H, W)

print("== ATTN normal ==")
count_flops(model_attn,   (x1,x2), multiply_by_two=False, verbose=True)

print("\n== ATTN lineal ==")
count_flops(model_linear, (x1,x2), multiply_by_two=False, verbose=True)
