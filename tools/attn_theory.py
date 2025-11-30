# tools/attn_theory.py
import math

def mhsa_macs(B, H, W, d, h, sr_ratio=1):
    N  = H*W
    Nk = (H//sr_ratio) * (W//sr_ratio) if sr_ratio>1 else N
    # MHSA core (qk^T + attn@v): B*h*Nq*Nk*d + B*h*Nq*Nk*d
    macs = B * h * N * Nk * d * 2
    return macs, {'N': N, 'Nk': Nk, 'd': d, 'h': h}

def rala_macs(B, H, W, d, h, sr_ratio=1):
    N  = H*W
    Nk = (H//sr_ratio) * (W//sr_ratio) if sr_ratio>1 else N
    # RALA core:
    #  KV = phi(K)^T V     → B*h*d*Nk*d
    #  out = phi(Q)*KV     → B*h*Nq*d*d
    #  denom phi(Q)*k_sum  → B*h*Nq*d  (pequeño; se suele ignorar, aquí lo sumo)
    macs = B*h*(d*Nk*d + N*d*d + N*d)
    return macs, {'N': N, 'Nk': Nk, 'd': d, 'h': h}

def pretty(v):
    if v >= 1e12: return f"{v/1e12:.3f} T"
    if v >= 1e9:  return f"{v/1e9:.3f} G"
    if v >= 1e6:  return f"{v/1e6:.3f} M"
    return str(v)

if __name__ == "__main__":
    # ejemplo: stage con H=W=64 (img 256, stride 4), d=64, h=1, SR=4
    # Probar con distintos batch size
    BATCHES = [1,4,8,16,32,64]
    for B in BATCHES:
        H, W, d, h, sr = 64,64,64,1,4
        print(f"\nBatch size: {B}")
        #B,H,W,d,h,sr = 16,64,64,64,1,4
        m_mhsa, s1 = mhsa_macs(B,H,W,d,h,sr)
        m_rala, s2 = rala_macs(B,H,W,d,h,sr)
        print(f"MHSA: {pretty(m_mhsa)} MACs  | N={s1['N']} Nk={s1['Nk']} d={d} h={h}")
        print(f"RALA: {pretty(m_rala)} MACs  | N={s2['N']} Nk={s2['Nk']} d={d} h={h}")
        print(f"Ratio RALA/MHSA = {m_rala/m_mhsa:.3f}")

