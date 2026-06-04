from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .coupling import Coupling

@dataclass
class Anchor:
    mask: Tensor    # [N] float — 1 on anchored (seed/inherited) rows, 0 elsewhere
    target: Tensor  # [N, d] anchor targets s_i (pins seeds to hot init / child to parent state)

# grad_E: ∇_X E = couple + inhib + barrier (+ anchor) (spec §3.3)
def grad_E(X: Tensor, C: Coupling, cfg: FieldConfig, anchor: Anchor | None = None) -> Tensor:
    # couple: ∇E_couple = −C_sym X  (spec §3.3)
    g_couple = -(C.sym @ X)
    # inhib: p = softmax_i(β·½‖x_i‖²), ∇E_inhib = p_i · x_i
    e = 0.5 * (X * X).sum(-1)           # [N]
    p = F.softmax(cfg.beta * e, dim=0)  # [N]
    g_inhib = p.unsqueeze(-1) * X       # [N, d]
    # barrier: ∇E_barrier = μ‖x_i‖² x_i  (cubic, from (μ/4)‖x‖⁴)
    g_barrier = cfg.mu * (X * X).sum(-1, keepdim=True) * X  # [N, d]
    g = g_couple + g_inhib + g_barrier
    # decay: leak λ_i·x_i on non-anchored rows (📍1) ⇒ damped diffusion, localized fixed point. λ_i is
    # the genericity leak C.decay_vec (the LOCALIZE spine) when built, else scalar cfg.decay; λ_i≥0 ⇒
    # inward damping ⇒ the absorbing ball holds.
    leak = torch.ones(X.shape[0]) if anchor is None else (1.0 - anchor.mask)
    lam = cfg.decay if C.decay_vec is None else C.decay_vec
    g = g + (lam * leak).unsqueeze(-1) * X
    # anchor: ∇E_anchor = σ(x_i − s_i) on anchored rows  (spec §3.3 — a true potential)
    if anchor is not None:
        g = g + cfg.sigma_anchor * anchor.mask.unsqueeze(-1) * (X - anchor.target)
    return g

# compute_E: scalar energy E_couple + E_inhib + E_barrier (+ E_anchor) (spec §3.3)
def compute_E(X: Tensor, C: Coupling, cfg: FieldConfig, anchor: Anchor | None = None) -> Tensor:
    # couple: −½ tr(X^T C_sym X)
    E_couple = -0.5 * (X * (C.sym @ X)).sum()
    # inhib: (1/β) · logsumexp(β · ½‖x‖²)
    e = 0.5 * (X * X).sum(-1)
    E_inhib = torch.logsumexp(cfg.beta * e, dim=0) / cfg.beta
    # barrier: Σ_i (μ/4)‖x_i‖⁴
    E_barrier = (cfg.mu / 4.0) * (X * X).sum(-1).pow(2).sum()
    E = E_couple + E_inhib + E_barrier
    # decay: Σ_{non-anchored} (λ_i/2)‖x_i‖²  (λ_i = C.decay_vec genericity leak, else scalar cfg.decay)
    leak = torch.ones(X.shape[0]) if anchor is None else (1.0 - anchor.mask)
    lam = cfg.decay if C.decay_vec is None else C.decay_vec
    E = E + 0.5 * (lam * leak * (X * X).sum(-1)).sum()
    # anchor: Σ_anchors (σ/2)‖x_i − s_i‖²
    if anchor is not None:
        E = E + (cfg.sigma_anchor / 2.0) * (anchor.mask * ((X - anchor.target) ** 2).sum(-1)).sum()
    return E
