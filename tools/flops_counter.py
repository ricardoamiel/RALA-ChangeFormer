# tools/flops_counter.py
import torch
import torch.nn as nn
from collections import defaultdict
import re

# Si tus clases están en otro módulo, importa los tipos reales:
# from your_pkg.blocks import Attention, LinearAttention
ATTN_CLASS_NAMES = {"Attention", "LinearAttention"}

def is_attn_module(m: nn.Module):
    return m.__class__.__name__ in ATTN_CLASS_NAMES

def human(n):
    # Formato amigable (K, M, G)
    for unit in ['','K','M','G','T']:
        if abs(n) < 1000.0:
            return f"{n:3.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"

def bucket_from_name(name: str):
    name_l = name.lower()
    if re.search(r'\b(attn|attention|to_q|to_k|to_v|proj)\b', name_l):
        return "attn"
    if "mlp" in name_l or "ffn" in name_l or "fc" in name_l:
        return "mlp"
    if re.search(r'\bsr\b', name_l) and "conv" in name_l:
        return "sr"
    if "norm" in name_l:
        return "norm"
    if "conv" in name_l:
        return "conv"
    return "other"

# --------- Contadores específicos por módulo ---------

def flops_linear(module: nn.Linear, input, output):
    # input: tuple(tensor)
    x = input[0]
    # x shape: (..., in_features)
    batch_elems = x.numel() // x.size(-1)
    macs = batch_elems * module.in_features * module.out_features
    return macs

def flops_conv2d(module: nn.Conv2d, input, output):
    x = input[0]  # [B, Cin, H, W]
    B = x.shape[0]
    Cin = module.in_channels
    Cout = module.out_channels
    Kh, Kw = module.kernel_size
    groups = module.groups
    # salida
    Hout, Wout = output.shape[2], output.shape[3]
    macs_per_pos = (Cin // groups) * Kh * Kw
    macs = B * Cout * Hout * Wout * macs_per_pos
    return macs

def flops_layernorm(module: nn.LayerNorm, input, output):
    # Coste aproximado: ~ 4 * elements (mean, var, norm, scale+shift)
    x = input[0]
    elems = x.numel()
    return 4 * elems

def _safe_get(m, k, default=None):
    sh = getattr(m, "_last_shapes", None)
    return sh.get(k, default) if sh else default

def flops_attention_normal(module, input, output):
    # Usa _last_shapes poblado dentro del forward
    B = _safe_get(module, 'B', 0)
    h = _safe_get(module, 'h', 0)
    Nq = _safe_get(module, 'Nq', 0)
    Nk = _safe_get(module, 'Nk', 0)
    d  = _safe_get(module, 'd',  0)
    if min(B,h,Nq,Nk,d) == 0:
        return 0
    # QK^T y AV (MACs); softmax ~ B*h*Nq*Nk (pequeño)
    macs = B*h*Nq*Nk*d + B*h*Nq*Nk + B*h*Nq*Nk*d
    return macs

def flops_attention_linear(module, input, output):
    B = _safe_get(module, 'B', 0)
    h = _safe_get(module, 'h', 0)
    Nq = _safe_get(module, 'Nq', 0)
    Nk = _safe_get(module, 'Nk', 0)
    d  = _safe_get(module, 'd',  0)
    if min(B,h,Nq,Nk,d) == 0:
        return 0
    # KV: B*h*Nk*d*d ; out: B*h*Nq*d*d ; denom: B*h*Nq*d
    macs = B*h*Nk*d*d + B*h*Nq*d*d + B*h*Nq*d
    return macs

# --------- Motor de hooks ---------

class FLOPsCounter:
    def __init__(self):
        self.total_macs = 0
        self.by_module = {}         # name -> macs
        self.by_bucket = defaultdict(float) # attn/mlp/conv/norm/sr/other
        self.hooks = []

    def _hook_for(self, name, module):
        def hook(module, inp, out):
            macs = 0
            if isinstance(module, nn.Conv2d):
                macs = flops_conv2d(module, inp, out)
            elif isinstance(module, nn.Linear):
                macs = flops_linear(module, inp, out)
            elif isinstance(module, nn.LayerNorm):
                macs = flops_layernorm(module, inp, out)
            elif is_attn_module(module):
                if module.__class__.__name__ == "Attention":
                    macs = flops_attention_normal(module, inp, out)
                else:
                    macs = flops_attention_linear(module, inp, out)
            # Acumular
            self.by_module[name] = self.by_module.get(name, 0) + macs
            self.total_macs += macs
            # Bucket por nombre
            bucket = bucket_from_name(name)
            self.by_bucket[bucket] += macs
        return hook

    def add_model(self, model: nn.Module):
        # Registra hooks en TODOS los submódulos
        for name, m in model.named_modules():
            self.hooks.append(m.register_forward_hook(self._hook_for(name, m)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

def count_flops(model: nn.Module, example_inputs: tuple, multiply_by_two=False, verbose=True, topk=10):
    """
    Ejecuta un forward con hooks y devuelve:
      - total_macs
      - dict por bucket
      - dict top módulos por costo
    """
    was_training = model.training
    model.eval()
    counter = FLOPsCounter()
    counter.add_model(model)

    # Forward para poblar _last_shapes en atención y disparar hooks
    with torch.no_grad():
        _ = model(*example_inputs)

    counter.remove()
    if was_training:
        model.train()

    total_macs = counter.total_macs
    factor = 2 if multiply_by_two else 1
    total_flops = total_macs * factor

    # Top módulos más costosos
    top = sorted(counter.by_module.items(), key=lambda x: -x[1])[:topk]

    if verbose:
        print(f"TOTAL MACs:  {human(total_macs)}")
        if multiply_by_two:
            print(f"TOTAL FLOPs (2*MACs): {human(total_flops)}")
        print("\nDesglose por bucket:")
        for k,v in sorted(counter.by_bucket.items(), key=lambda x: -x[1]):
            print(f"  - {k:6s}: {human(v)} MACs")
        print("\nTop módulos por costo:")
        for name,macs in top:
            print(f"  {name:50s} {human(macs)} MACs")

    return total_macs, counter.by_bucket, dict(top)
