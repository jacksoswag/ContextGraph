from __future__ import annotations
import torch
from torch import Tensor

# personalized_pagerank: the diffusion baseline the gather must earn its keep against (spec §11).
# Power-iterate r = (1−α)·Pᵀr + α·t on the same active-set graph, where P is the row-stochastic
# normalization of the (nonneg) coupling W and t teleports to the seeds. Deterministic. Returns
# stationary scores r [N] (∑r=1) — rank/prune analogously to the gather's relevance.
def personalized_pagerank(W: Tensor, seed_idx: list[int], alpha: float = 0.15,
                          iters: int = 300, tol: float = 1e-10) -> Tensor:
    N = W.shape[0]
    A = W.clamp(min=0.0)                     # PPR needs nonneg weights (cosine coupling can be <0)
    deg = A.sum(1, keepdim=True)
    P = A / deg.clamp(min=1e-12)             # row-stochastic transition P[i,j] = p(i→j)
    t = torch.zeros(N)
    t[seed_idx] = 1.0 / max(len(seed_idx), 1)
    r = t.clone()
    Pt = P.t().contiguous()
    for _ in range(iters):
        r_new = (1.0 - alpha) * (Pt @ r) + alpha * t
        if float((r_new - r).abs().sum()) < tol: r = r_new; break
        r = r_new
    return r
