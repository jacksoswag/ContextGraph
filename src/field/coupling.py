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
    decay_vec: Tensor | None = None     # [N] per-node leak λ_i (genericity hill); None ⇒ scalar cfg.decay

# build: construct C_sym from episode embeddings; stores λmax, R_max, L, η_bound (spec §3.3–3.4).
# edge_w ([E], optional): Sleep-learned per-edge multiplier on the cosine coupling (support stays
# E(S) — multiplies existing edges only, never creates them; §6).
def build(ep: Episode, cfg: FieldConfig, edge_w: Tensor | None = None) -> Coupling:
    N = len(ep.node_ids)
    src, dst = ep.edge_index[0], ep.edge_index[1]
    # cosine similarities for all edges (vectorized)
    A_norm = F.normalize(ep.A.float(), dim=-1)
    # semantic: strength = endpoint info_vector cosine; structural: strength = 1 (pure adjacency,
    # degree-normalized below ⇒ a nonlinear PPR that follows structure, not embeddings).
    cos_sim = (torch.ones(src.shape[0]) if cfg.couple_mode == "structural"
               else (A_norm[src] * A_norm[dst]).sum(-1))   # [E]
    if edge_w is not None: cos_sim = cos_sim * edge_w
    # build symmetric weight matrix masked to E(S)
    sym_raw = torch.zeros(N, N)
    sym_raw[src, dst] = cos_sim
    sym_raw = (sym_raw + sym_raw.T) * 0.5           # exact symmetry
    # degree-normalize: D^{-1/2} sym_raw D^{-1/2}
    deg = sym_raw.abs().sum(-1)                      # [N]
    d_inv = torch.where(deg > 0, deg.rsqrt(), torch.zeros(N))
    sym = d_inv.unsqueeze(-1) * sym_raw * d_inv.unsqueeze(0)
    # hyperedge containment: clique-couple each reified edge's members so energy stays
    # within the hyperedge. Added AFTER normalization (a direct, undiluted attraction) —
    # folding it into the degree norm would inflate members' degrees and dilute the
    # parent edge's own coupling, collapsing the fact's relevance. Still a symmetric
    # potential ⇒ E stays Lyapunov; L/R_max below are computed on the final `sym`.
    if cfg.w_hyper > 0.0 and ep.hyperedges:
        for members, w_e in ep.hyperedges:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    sym[i, j] += cfg.w_hyper * w_e; sym[j, i] += cfg.w_hyper * w_e
    # per-node leak λ_i: genericity-graded when degrees are known (generic hubs leak faster ⇒ the
    # low→high climb costs more through ambiguous hubs); uniform cfg.decay otherwise. A PSD diagonal
    # potential Σ(λ_i/2)‖x_i‖² ⇒ E stays Lyapunov; L below uses its max so η stays safe.
    decay_vec = None
    if cfg.decay_gamma > 0.0 and ep.degree is not None:
        decay_vec = cfg.decay * (1.0 + cfg.decay_gamma * torch.log1p(ep.degree.float()))
    leak_max = cfg.decay if decay_vec is None else float(decay_vec.max())
    # derived quantities stored on Coupling
    lam_max = float(torch.linalg.eigvalsh(sym).max().clamp(min=1e-6))
    R_max = math.sqrt(lam_max / cfg.mu)
    # L includes the anchor (σ·I) and decay (λ·I) Hessians so η stays safe with both §3.3 potentials
    L = (float(sym.norm()) + cfg.beta ** 2 * R_max ** 2 + 3.0 * cfg.mu * R_max ** 2
         + max(cfg.sigma_anchor, leak_max))
    # eta_bound is the advisory step-size ceiling (1.9/L); build() does not read cfg.eta — safe_build
    # clamps eta under this bound, and dynamics.step's trapping guard is the hard runtime safety net.
    eta_bound = 1.9 / max(L, 1e-8)
    return Coupling(sym=sym, lambda_max=lam_max, R_max=R_max, L=L, eta_bound=eta_bound,
                    decay_vec=decay_vec)
