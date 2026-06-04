from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .episode import Episode

@dataclass
class Coupling:
    sym: Tensor                      # [N, N] symmetric, degree-normalized, support=E(S) — the PPR W
    decay_vec: Tensor | None = None  # [N] per-node genericity leak λ_i (the PPR absorption); None ⇒ uniform

# build: construct C_sym (the PPR transition weight W) from the episode structure. edge_w ([E],
# optional): Sleep-learned per-edge multiplier on the coupling (support stays E(S) — multiplies
# existing edges only, never creates them; §6).
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
    # per-node genericity leak λ_i = decay·(1+γ·ln(1+deg_i)): graded when degrees are known (generic
    # hubs leak faster ⇒ they absorb faster in gather's PPR sink ⇒ the settle localizes); uniform
    # cfg.decay otherwise (None ⇒ no sink). The only derived quantity now that the integrator's
    # step-size machinery (λmax/R_max/L/η_bound) is gone — the PPR solve needs no step size.
    decay_vec = None
    if cfg.decay_gamma > 0.0 and ep.degree is not None:
        decay_vec = cfg.decay * (1.0 + cfg.decay_gamma * torch.log1p(ep.degree.float()))
    return Coupling(sym=sym, decay_vec=decay_vec)
