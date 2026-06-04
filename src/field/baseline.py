from __future__ import annotations
import torch
from torch import Tensor

# personalized_pagerank: the diffusion baseline the gather must earn its keep against (spec §11) and,
# post-PPR-gate, the per-settle ENGINE of the recursive mesh path. Power-iterate r = (1−α)·Pᵀr + α·t on
# the active-set graph, where P is the row-stochastic normalization of the (nonneg) coupling W and t is
# the teleport/personalization. teleport=None ⇒ uniform mass on seed_idx (the diffusion baseline);
# teleport=[N] ⇒ a weighted personalization (the recursive path's ANCESTRY vector — where the lineage
# injects energy this round). Deterministic. Returns stationary scores r [N] (∑r=1).
def personalized_pagerank(W: Tensor, seed_idx: list[int] | None = None, alpha: float = 0.15,
                          iters: int = 300, tol: float = 1e-10, teleport: Tensor | None = None) -> Tensor:
    N = W.shape[0]
    A = W.clamp(min=0.0)                     # PPR needs nonneg weights (cosine coupling can be <0)
    deg = A.sum(1, keepdim=True)
    P = A / deg.clamp(min=1e-12)             # row-stochastic transition P[i,j] = p(i→j)
    if teleport is not None:
        t = teleport.clamp(min=0.0); s = float(t.sum()); t = t / s if s > 0 else t
    else:
        t = torch.zeros(N); t[seed_idx or []] = 1.0 / max(len(seed_idx or []), 1)
    r = t.clone()
    Pt = P.t().contiguous()
    for _ in range(iters):
        r_new = (1.0 - alpha) * (Pt @ r) + alpha * t
        if float((r_new - r).abs().sum()) < tol: r = r_new; break
        r = r_new
    return r
