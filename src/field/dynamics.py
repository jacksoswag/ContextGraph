from __future__ import annotations
import torch
from torch import Tensor
from .config import FieldConfig
from .coupling import Coupling
from .energy import grad_E, compute_E

# step: one explicit-Euler gradient-descent step of Ẋ=−∇E with trapping guard (spec §3.4)
def step(X: Tensor, C: Coupling, cfg: FieldConfig) -> Tensor:
    G = -grad_E(X, C, cfg)
    X_next = X + cfg.eta * G
    # barrier enforced by clamp to the absorbing ball (R_max)
    norms = X_next.norm(dim=-1, keepdim=True)
    X_next = torch.where(norms > C.R_max, X_next * (C.R_max / norms), X_next)
    assert X_next.norm(dim=-1).max().item() <= C.R_max + 1e-4, (
        f"Trapping guard: ‖x‖_max={X_next.norm(dim=-1).max():.6f} > R_max={C.R_max:.6f}")
    return X_next

# rollout: iterate step to H_max; early-stop on a SUSTAINED fixed point (‖ΔX‖<ε_x for H_hold
# consecutive steps). Pure gradient flow on coercive E ⇒ E monotone non-increasing ⇒ the gather
# always settles. Returns (X_hist [T,N,d], E_hist [T]).
def rollout(X0: Tensor, C: Coupling, cfg: FieldConfig) -> tuple[Tensor, Tensor]:
    X = X0.clone()
    X_hist, E_hist = [X.clone()], [compute_E(X, C, cfg).item()]
    settled = 0
    for _ in range(cfg.H_max):
        X_prev = X
        X = step(X, C, cfg)
        X_hist.append(X.clone()); E_hist.append(compute_E(X, C, cfg).item())
        settled = settled + 1 if (X - X_prev).norm().item() < cfg.eps_x else 0
        if settled >= cfg.H_hold: break
    return torch.stack(X_hist), torch.tensor(E_hist)
