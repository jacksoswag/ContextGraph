from __future__ import annotations
from dataclasses import dataclass
import math, torch
import torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .episode import Episode

@dataclass
class Coupling:
    sym: Tensor      # [N, N] symmetric, degree-normalized, support=E(S)
    lambda_max: float
    R_max: float     # sqrt(λmax/μ) — absorbing-ball radius
    L: float         # Lipschitz bound ‖C_sym‖ + β²·R_max² + 3μ·R_max²
    eta_bound: float # 1.9/L — safe step-size ceiling

# build: construct C_sym from episode embeddings; stores λmax, R_max, L, η_bound (spec §3.3–3.4).
# edge_w ([E], optional): Sleep-learned per-edge multiplier on the cosine coupling (support stays
# E(S) — multiplies existing edges only, never creates them; §6).
def build(ep: Episode, cfg: FieldConfig, edge_w: Tensor | None = None) -> Coupling:
    N = len(ep.node_ids)
    src, dst = ep.edge_index[0], ep.edge_index[1]
    # cosine similarities for all edges (vectorized)
    A_norm = F.normalize(ep.A.float(), dim=-1)
    cos_sim = (A_norm[src] * A_norm[dst]).sum(-1)   # [E]
    if edge_w is not None: cos_sim = cos_sim * edge_w
    # build symmetric weight matrix masked to E(S)
    sym_raw = torch.zeros(N, N)
    sym_raw[src, dst] = cos_sim
    sym_raw = (sym_raw + sym_raw.T) * 0.5           # exact symmetry
    # degree-normalize: D^{-1/2} sym_raw D^{-1/2}
    deg = sym_raw.abs().sum(-1)                      # [N]
    d_inv = torch.where(deg > 0, deg.rsqrt(), torch.zeros(N))
    sym = d_inv.unsqueeze(-1) * sym_raw * d_inv.unsqueeze(0)
    # derived quantities stored on Coupling
    lam_max = float(torch.linalg.eigvalsh(sym).max().clamp(min=1e-6))
    R_max = math.sqrt(lam_max / cfg.mu)
    # L includes the anchor (σ·I) and decay (λ·I) Hessians so η stays safe with both §3.3 potentials
    L = (float(sym.norm()) + cfg.beta ** 2 * R_max ** 2 + 3.0 * cfg.mu * R_max ** 2
         + max(cfg.sigma_anchor, cfg.decay))
    eta_bound = 1.9 / max(L, 1e-8)
    assert cfg.eta <= eta_bound, f"η={cfg.eta} exceeds L-derived bound {eta_bound:.4f} (L={L:.2f})"
    return Coupling(sym=sym, lambda_max=lam_max, R_max=R_max, L=L, eta_bound=eta_bound)
