import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature map positiva para atención lineal (Trick clásico)
class PositiveFeatureMap(nn.Module):
    """
    Feature map positiva φ(x) para atención lineal.
    Usamos ELU + 1 como en implementaciones clásicas (Trick de Performer).
    """
    def forward(self, x):
        # x: (B, H*W, D) o (B, N, D)
        return F.elu(x, alpha=1.0) + 1.0

class LinearAttention(nn.Module):
    """
    Atención lineal: softmax(QK^T)V ≈ (φ(Q)[ φ(K)^T V ]) / (φ(Q)[ φ(K)^T 1 ])
    Donde φ(.) es feature map positiva (aquí elu+1).
    Esta versión respeta 'heads' y dropout como MHSA.
    """
    def __init__(self, dim, heads=8, attn_drop=0.0, proj_drop=0.0, bias=True):
        super().__init__()
        assert dim % heads == 0, "dim debe ser múltiplo de heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim  ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)

        self.feature_map = PositiveFeatureMap()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Para herramientas de profiling (thop/tflops)
        self._last_shapes = None # se pobla en forward
        self.eps = 1e-6          # estabilidad numérica

    def forward(self, x_q, x_kv):
        """
        x_q:  [B, Nq, C]
        x_kv: [B, Nk, C]
        return: [B, Nq, C]
        """
        B, Nq, C = x_q.shape
        Nk = x_kv.shape[1]
        d = self.head_dim

        # Proyecciones
        q = self.to_q(x_q)
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        # Split por heads => [B, h, N, d]
        q = q.reshape(B, Nq, self.heads, C // self.heads).transpose(1, 2)  # [B,h,Nq,d]
        k = k.reshape(B, Nk, self.heads, C // self.heads).transpose(1, 2)  # [B,h,Nk,d]
        v = v.reshape(B, Nk, self.heads, C // self.heads).transpose(1, 2)  # [B,h,Nk,d]

        # Escala estándar por head
        q = q * self.scale

        # Feature maps positivas (con dropout como proxy de attn_drop)
        q_phi = self.feature_map(q)  # [B,h,Nq,d]
        k_phi = self.feature_map(k)  # [B,h,Nk,d]
        if self.attn_drop.p > 0:
            q_phi = self.attn_drop(q_phi)
            k_phi = self.attn_drop(k_phi)

        # KV = (φ(K)^T V)  -> [B,h,d,d]
        #   k_phi: [B,h,Nk,d], v: [B,h,Nk,d]
        kv = torch.einsum('bhnd,bhne->bhde', k_phi, v)

        # k_sum = φ(K)^T 1  (suma sobre tokens Nk) -> [B,h,d]
        k_sum = k_phi.sum(dim=2)

        # Numerador: φ(Q) * KV -> [B,h,Nq,d]
        out = torch.einsum('bhnd,bhde->bhne', q_phi, kv)

        # Denominador: φ(Q) * k_sum -> [B,h,Nq,1]
        z = torch.einsum('bhnd,bhd->bhn', q_phi, k_sum).unsqueeze(-1) + self.eps

        out = out / z                         # [B,h,Nq,d]
        out = out.transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Guardar shapes para contadores de FLOPS/MACs
        self._last_shapes = {'B': B, 'h': self.heads, 'Nq': Nq, 'Nk': Nk, 'd': d}
        return out

"""
B, Nq, Nk, C, h = 2, 64, 64, 256, 8
x_q  = torch.randn(B, Nq, C)
x_kv = torch.randn(B, Nk, C)
attn = LinearAttention(dim=C, heads=h)
y = attn(x_q, x_kv)  # debe devolver [B, Nq, C] sin errores it works
print(y.shape)
"""
