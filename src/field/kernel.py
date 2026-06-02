from __future__ import annotations
import torch, math
from torch import Tensor
from .config import FieldConfig

# φ_t — history-deformed joint kernel (spec §2.4, §10.2).
# Uses P_slow (detached) so no gradient flows through the kernel geometry.
# Returns [N, N] matrix of pairwise compatibility values in [0, 1].

def compute_phi(X: Tensor, A: Tensor, m: Tensor, P_slow: Tensor, cfg: FieldConfig) -> Tensor:
    # X: [N, d_lat] (current trainable latents, unused in kernel — geometry uses P_slow(A))
    # A: [N, d_anchor], m: [d_lat], P_slow: [d_lat, d_anchor]
    # §2.1.1: φ_t uses P_slow applied to ANCHORS A (detached slow geometry), not X.
    # "Px_i" in spec §2.4 = P_slow(a_i) = slow-anchor projection, shape [d_lat].
    with torch.no_grad():
        Px = A @ P_slow.T   # [N, d_anchor] @ [d_anchor, d_lat] = [N, d_lat]
    # φ-space axis: joint / structural-only / semantic-only (§2.4, open axis)
    scale = 1.0 / math.sqrt(cfg.d_lat)
    if cfg.phi_space == "semantic":
        a_scale = 1.0 / math.sqrt(cfg.d_anchor)
        logit_a = torch.clamp((A @ A.T) * a_scale, -cfg.C_phi, cfg.C_phi)
        return torch.sigmoid(logit_a)
    # Structural: ⟨Px_i, Px_j⟩ pairwise
    PP = Px @ Px.T          # [N, N]
    # Memory cross-terms: ⟨Px_i, m⟩ + ⟨Px_j, m⟩ (§2.4)
    Pxm = Px @ m            # [N]
    PP_mem = PP + Pxm.unsqueeze(1) + Pxm.unsqueeze(0)   # [N, N]
    logit_s = torch.clamp(PP_mem * scale, -cfg.C_phi, cfg.C_phi)
    phi = torch.sigmoid(logit_s)
    if cfg.phi_space == "structural" or cfg.lam == 0.0:
        return phi
    # Joint: add λ·σ(⟨a_i, a_j⟩/√d_anchor)
    a_scale = 1.0 / math.sqrt(cfg.d_anchor)
    logit_a = torch.clamp((A @ A.T) * a_scale, -cfg.C_phi, cfg.C_phi)
    phi = phi + cfg.lam * torch.sigmoid(logit_a)
    # Normalize to [0,1] (joint form's max is 1+λ)
    return phi / (1.0 + cfg.lam)

# Weighted Laplacian L_S under φ_t edge weights (clique-expansion, symmetric).
# L_S = D - W where W[i,j] = φ_t(i,j), D = diag(row sums).
def compute_laplacian(phi: Tensor) -> Tensor:
    D = phi.sum(dim=-1)
    return torch.diag(D) - phi

# Neighbor-masked φ_t: zero out non-adjacent pairs using the episode edge index.
# Returns [N, N] sparse-style adjacency-masked kernel (non-neighbors stay at cfg.eps).
def masked_phi(phi: Tensor, edge_src: Tensor, edge_tgt: Tensor, N: int, cfg: FieldConfig) -> Tensor:
    mask = torch.zeros(N, N, dtype=torch.bool)
    if edge_src.numel() > 0:
        mask[edge_src, edge_tgt] = True
        mask.diagonal().fill_(True)          # self-loops included
    # Non-adjacent pairs get a very small value (not zero, to avoid div-by-zero in Π_S)
    phi_m = phi.clone()
    phi_m[~mask] = cfg.eps
    return phi_m
