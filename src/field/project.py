from __future__ import annotations
import torch
from torch import Tensor
from .config import FieldConfig
from .spaces import Episode

# Π_S projection — soft constraint pushing node latents toward graph manifold (spec §6, §10.4).
# Generalizes settle.settle_activation to ℝ^d_lat per node with φ_t-weighted aggregation
# and explicit beyond-pairwise hyperedge term.
#
# Π_S(x)_i = (1-β)·x_i + β·(Σ_{j∈N₂(i)} φ(i,j)·x_j + Σ_{e∈H(i)} w_e·x̄_{e\i})
#                        / (Σ_{j∈N₂(i)} φ(i,j)     + Σ_{e∈H(i)} w_e     + ε)

def project(x: Tensor, phi: Tensor, ep: Episode, cfg: FieldConfig, beta: float | None = None) -> Tensor:
    # x: [N, d_lat], phi: [N, N] (edge-masked), β from control bus or config default
    if beta is None: beta = cfg.beta
    N = ep.N
    # Pairwise neighbor contribution (Σ_j φ(i,j)·x_j) using edge index
    pair_num = torch.zeros_like(x)   # [N, d_lat]
    pair_den = torch.zeros(N, dtype=x.dtype)   # [N]
    if ep.edge_src.numel() > 0:
        # Undirected: edges already doubled in materialize (src↔tgt), so use as-is
        w_vals = phi[ep.edge_src, ep.edge_tgt]   # [E] kernel weights for each edge
        # Accumulate weighted neighbor latents
        pair_num.scatter_add_(0, ep.edge_tgt.unsqueeze(-1).expand(-1, x.shape[-1]),
                               w_vals.unsqueeze(-1) * x[ep.edge_src])
        pair_den.scatter_add_(0, ep.edge_tgt, w_vals)
    # Hyperedge complement-mean contribution (Σ_{e∈H(i)} w_e·x̄_{e\i})
    hyp_num = torch.zeros_like(x)   # [N, d_lat]
    hyp_den = torch.zeros(N, dtype=x.dtype)   # [N]
    for members, w_e in ep.hyperedges:
        if len(members) < 2: continue
        m_t = torch.tensor(members, dtype=torch.long)
        x_members = x[m_t]             # [|e|, d_lat]
        total = x_members.sum(dim=0)   # [d_lat]
        for mi in members:
            # x̄_{e\i} = mean_{j∈e, j≠i} x_j = (total - x[mi]) / (|e|-1)
            complement_mean = (total - x[mi]) / max(len(members) - 1, 1)
            hyp_num[mi] += w_e * complement_mean
            hyp_den[mi] += w_e
    # Combined aggregation: Π_S(x)_i = (1-β)·x_i + β·agg / (denom + ε)
    num = pair_num + hyp_num         # [N, d_lat]
    den = (pair_den + hyp_den).unsqueeze(-1).clamp(min=cfg.eps)   # [N, 1]
    agg = num / den
    return (1.0 - beta) * x + beta * agg
