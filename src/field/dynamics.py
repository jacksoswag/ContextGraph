from __future__ import annotations
import torch
from torch import Tensor
from .config import FieldConfig
from .coupling import Coupling
from .energy import grad_E, compute_E, Anchor

# step: one explicit-Euler gradient-descent step of Ẋ=−∇E with trapping guard (spec §3.4)
def step(X: Tensor, C: Coupling, cfg: FieldConfig, anchor: Anchor | None = None) -> Tensor:
    G = -grad_E(X, C, cfg, anchor)
    X_next = X + cfg.eta * G
    # barrier enforced by clamp to the absorbing ball (R_max)
    norms = X_next.norm(dim=-1, keepdim=True)
    X_next = torch.where(norms > C.R_max, X_next * (C.R_max / norms), X_next)
    assert X_next.norm(dim=-1).max().item() <= C.R_max + 1e-4, (
        f"Trapping guard: ‖x‖_max={X_next.norm(dim=-1).max():.6f} > R_max={C.R_max:.6f}")
    return X_next

# rollout: iterate step to H_max; early-stop on a SUSTAINED fixed point (‖ΔX‖<ε_x for H_hold
# consecutive steps). Pure gradient flow on coercive E (barrier + anchor) ⇒ E monotone non-increasing
# ⇒ the gather always settles. Returns (X_hist, E_hist, steps). lean (live path): the readout uses
# X* only, so skip the per-step trajectory clone (~520 MB/query) AND the per-step compute_E (a 2nd
# full C_sym·X matmul) — the SETTLE ITERATION IS UNCHANGED, only the logging is dropped, so X* is
# bit-identical to the tracked path; X_hist collapses to [X0,X*] and E_hist to [E0,E*]. Tracked mode
# (default) keeps the full [T,N,d] trajectory + per-step energy for the Lyapunov/diagnostic consumers.
def rollout(X0: Tensor, C: Coupling, cfg: FieldConfig, anchor: Anchor | None = None,
            *, lean: bool = False) -> tuple[Tensor, Tensor, int]:
    X = X0.clone()
    X_hist = None if lean else [X.clone()]
    E_hist = None if lean else [compute_E(X, C, cfg, anchor).item()]
    settled = steps = 0
    for _ in range(cfg.H_max):
        X_prev = X
        X = step(X, C, cfg, anchor)
        steps += 1
        if not lean: X_hist.append(X.clone()); E_hist.append(compute_E(X, C, cfg, anchor).item())
        settled = settled + 1 if (X - X_prev).norm().item() < cfg.eps_x else 0
        if settled >= cfg.H_hold: break
    if lean:
        E = torch.tensor([compute_E(X0, C, cfg, anchor).item(), compute_E(X, C, cfg, anchor).item()])
        return torch.stack([X0, X]), E, steps
    return torch.stack(X_hist), torch.tensor(E_hist), steps
