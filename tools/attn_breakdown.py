# tools/attn_breakdown.py
import argparse
import torch
from collections import defaultdict

from models.ChangeFormer import ChangeFormerV6, ChangeFormerV7
from models.ChangeFormer import Block, BlockLinear
from models.ChangeFormer import Attention
from models.linear_attention import LinearAttention

def mhsa_core_macs(B,h,Nq,Nk,d):
    # qk^T + attn@v
    return B*h*Nq*Nk*d*2

def rala_core_macs(B,h,Nq,Nk,d):
    # KV (d*Nk*d) + out (Nq*d*d) + denom (Nq*d)
    return B*h*(d*Nk*d + Nq*d*d + Nq*d)

def count_attn_params(attn: torch.nn.Module):
    # Solo los linears de la atención, NO MLP
    p = 0
    for n,m in attn.named_modules():
        if isinstance(m, torch.nn.Linear):
            p += sum(p_.numel() for p_ in m.parameters() if p_.requires_grad)
    return p

def run(model_name="v6", img_size=256, device="cpu"):
    if model_name.lower() == "v6":
        net = ChangeFormerV6().to(device)
    else:
        net = ChangeFormerV7().to(device)

    net.eval()
    x1 = torch.randn(1,3,img_size,img_size, device=device)
    x2 = torch.randn(1,3,img_size,img_size, device=device)

    with torch.no_grad():
        _ = net(x1, x2)

    rows = []
    tot_macs = 0
    tot_params = 0
    counts = defaultdict(int)

    for name, m in net.named_modules():
        if isinstance(m, Attention) and hasattr(m, "_last_shapes") and m._last_shapes:
            s = m._last_shapes
            B = s['B']; h=s['h']; Nq=s['Nq']; Nk=s['Nk']; d=s['d']
            macs = mhsa_core_macs(B,h,Nq,Nk,d)
            flops = 2 * macs
            params = count_attn_params(m)
            rows.append((name, "MHSA", B,h,Nq,Nk,d, macs, flops, params))
            tot_macs += macs
            tot_flops = 2 * tot_macs
            tot_params += params
            counts["MHSA"] += 1

        if isinstance(m, LinearAttention) and hasattr(m, "_last_shapes") and m._last_shapes:
            s = m._last_shapes
            B = s['B']; h=s['h']; Nq=s['Nq']; Nk=s['Nk']; d=s['d']
            macs = rala_core_macs(B,h,Nq,Nk,d)
            flops = 2 * macs
            params = count_attn_params(m)
            rows.append((name, "RALA", B,h,Nq,Nk,d, macs, flops, params))
            tot_macs += macs
            tot_flops = 2 * tot_macs
            tot_params += params
            counts["RALA"] += 1

    def pretty(v):
        if v >= 1e12: return f"{v/1e12:.3f} T"
        if v >= 1e9:  return f"{v/1e9:.3f} G"
        if v >= 1e6:  return f"{v/1e6:.3f} M"
        if v >= 1e3:  return f"{v/1e3:.3f} K"
        return str(v)

    print(f"== Attention breakdown for {model_name.upper()} @ {img_size} ==")
    print("block_path,type,B,h,Nq,Nk,d,MACs,FLOPs,params")
    for r in rows:
        name, typ,B,h,Nq,Nk,d,macs,flops, params = r
        print(f"{name},{typ},{B},{h},{Nq},{Nk},{d},{pretty(macs)},{pretty(flops)},{pretty(params)}")
    print("-"*60)
    print(f"Totals: MACs(attn)={pretty(tot_macs)} | FLOPs(attn)={pretty(tot_flops)} "
      f"| params(attn)={pretty(tot_params)} | blocks: MHSA={counts['MHSA']}, RALA={counts['RALA']}")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v6","v7"], default="v6")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args = ap.parse_args()
    run(args.model, args.img_size, args.device)

