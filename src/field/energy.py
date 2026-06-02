from __future__ import annotations
import torch, torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .coupling import Coupling

# grad_E: ∇_X E = g_couple + g_inhib + g_barrier (spec §3.3)
def grad_E(X: Tensor, C: Coupling, cfg: FieldConfig) -> Tensor:
    # couple: ∇E_couple = −C_sym X  (spec §3.3)
    g_couple = -(C.sym @ X)
    # inhib: p = softmax_i(β·½‖x_i‖²), ∇E_inhib = p_i · x_i
    e = 0.5 * (X * X).sum(-1)           # [N]
    p = F.softmax(cfg.beta * e, dim=0)  # [N]
    g_inhib = p.unsqueeze(-1) * X       # [N, d]
    # barrier: ∇E_barrier = μ‖x_i‖² x_i  (cubic, from (μ/4)‖x‖⁴)
    g_barrier = cfg.mu * (X * X).sum(-1, keepdim=True) * X  # [N, d]
    return g_couple + g_inhib + g_barrier

# compute_E: scalar energy diagnostic E_couple + E_inhib + E_barrier (spec §3.3)
def compute_E(X: Tensor, C: Coupling, cfg: FieldConfig) -> Tensor:
    # couple: −½ tr(X^T C_sym X)
    E_couple = -0.5 * (X * (C.sym @ X)).sum()
    # inhib: (1/β) · logsumexp(β · ½‖x‖²)
    e = 0.5 * (X * X).sum(-1)
    E_inhib = torch.logsumexp(cfg.beta * e, dim=0) / cfg.beta
    # barrier: Σ_i (μ/4)‖x_i‖⁴
    E_barrier = (cfg.mu / 4.0) * (X * X).sum(-1).pow(2).sum()
    return E_couple + E_inhib + E_barrier
