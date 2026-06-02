from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Callable
from .config import FieldConfig
from .spaces import Episode, FieldState
from .kernel import compute_phi, masked_phi
from .memory import update_memory, assert_memory_norm
from .routing import FieldRouter
from .operators import FieldOperators
from .project import project

# Per-step structured output: all state + diagnostics for objective.py.
@dataclass
class StepResult:
    x_next: Tensor       # [N, d_lat] POST-Π_S, post-norm (used for next step)
    m_next: Tensor       # [d_lat] updated memory
    phi: Tensor          # [N, N] current kernel (for C_A diagnostic)
    alpha: Tensor        # [N, K] operator routing (for diagnostics)
    g_mem: Tensor        # [N] memory mass
    y: Tensor            # [N, d_lat] post-operator pre-noise (for B_t)
    xi: Tensor           # [N, d_lat] noise injected
    beta_used: float     # actual β applied (may differ from cfg if C-controlled)
    sigma_used: float    # actual σ applied

# Actuator state from C control bus (lagged by one step, §10.6).
@dataclass
class Actuators:
    beta: float = 0.5
    sigma: float = 0.01
    step_scale: float = 1.0
    sleep: bool = False

    # Clamp to hard ranges (§8.3)
    def clamp(self, cfg: FieldConfig) -> "Actuators":
        return Actuators(
            beta=float(torch.tensor(self.beta).clamp(cfg.beta_min, cfg.beta_max).item()),
            sigma=float(torch.tensor(self.sigma).clamp(cfg.sigma_min, cfg.sigma_max).item()),
            step_scale=float(torch.tensor(self.step_scale).clamp(cfg.step_min, cfg.step_max).item()),
            sleep=self.sleep,
        )

# FieldModel bundles all learned parameters (§2, §3).
class FieldModel(torch.nn.Module):
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        # Bridge P: ℝ^d_anchor → ℝ^d_lat, trainable (§2.1)
        self.P = torch.nn.Linear(cfg.d_anchor, cfg.d_lat, bias=False)
        torch.nn.init.xavier_uniform_(self.P.weight)
        # P_slow: EMA slow copy, stored as buffer (detached from grad, §2.1.1)
        self.register_buffer("P_slow", self.P.weight.data.clone())
        # Operators and router
        self.ops = FieldOperators(cfg)
        self.router = FieldRouter(cfg)

    # Update P_slow ← τ_P·P_slow + (1−τ_P)·P after each θ update (§2.1.1)
    def update_P_slow(self) -> None:
        with torch.no_grad():
            self.P_slow.mul_(self.cfg.tau_P).add_(self.P.weight.data, alpha=1.0 - self.cfg.tau_P)

# Execute one step of the field dynamics (§6, §10 fixed ordering).
# Actuators carry lagged C-control signals (§10.6); defaults from cfg on first step.
def field_step(state: FieldState, ep: Episode, model: FieldModel,
               actuators: Actuators | None = None) -> StepResult:
    cfg = model.cfg
    if actuators is None: actuators = Actuators(beta=cfg.beta, sigma=cfg.sigma).clamp(cfg)
    # ─── Step 1: READ m_t (frozen for this entire step) ───────────────────────
    m = state.m   # [d_lat]
    X = state.X   # [N, d_lat]
    # ─── Step 2: φ_t ← kernel(x_t, m_t) using P_slow, DETACHED ───────────────
    phi = compute_phi(X, ep.A, m, model.P_slow, cfg)          # [N, N]
    phi_m = masked_phi(phi, ep.edge_src, ep.edge_tgt, ep.N, cfg)  # [N, N] neighbor-masked
    # ─── Step 3: h_S, Score, g ← routing(x_t, φ_t) ──────────────────────────
    alpha, g_mem, score = model.router(X, phi_m)              # [N,K], [N], [N,K]
    # ─── Step 4: y ← F̃(x_t) ─────────────────────────────────────────────────
    y = model.ops(X, alpha)                                    # [N, d_lat]
    # ─── Step 5: y' ← y + ξ_t  (per-node noise PRE-projection, §10.3) ────────
    xi = torch.randn_like(y) * actuators.sigma
    y_noisy = y + xi
    # ─── Step 6: x̂_{t+1} ← Π_S(y')  (incl. hyperedge term, §10.4) ──────────
    x_hat = project(y_noisy, phi_m, ep, cfg, beta=actuators.beta)
    # ─── Step 7: x_{t+1} ← per-node L2 normalization (§10.2) ─────────────────
    x_next = x_hat / (x_hat.norm(dim=-1, keepdim=True) + cfg.eps)
    # ─── Step 8: WRITE m_{t+1} from POST-Π_S, post-norm, STOP-GRAD (§10.1) ───
    x_sg = x_next.detach()   # sg(·) — no gradient through memory
    m_next = update_memory(m, g_mem.detach(), x_sg, cfg)
    # ─── Step 9: actuators for step t+1 computed in objective.py (lagged) ─────
    return StepResult(x_next=x_next, m_next=m_next, phi=phi, alpha=alpha,
                      g_mem=g_mem, y=y, xi=xi,
                      beta_used=actuators.beta, sigma_used=actuators.sigma)

# Rollout: run up to horizon steps, terminate on L1-delta < eps OR horizon K (§6).
# Returns list of (FieldState, StepResult) pairs, one per step taken.
@dataclass
class RolloutResult:
    states: list[FieldState]            # states[0] = initial, states[-1] = final
    results: list[StepResult]           # len = steps taken
    terminated_early: bool
    steps: int

def rollout(init: FieldState, ep: Episode, model: FieldModel,
            actuators_seq: list[Actuators] | None = None,
            event_cb: Callable | None = None) -> RolloutResult:
    cfg = model.cfg
    states, results = [init], []
    current = init
    for t in range(cfg.horizon):
        act = (actuators_seq[t] if actuators_seq and t < len(actuators_seq)
               else Actuators(beta=cfg.beta, sigma=cfg.sigma).clamp(cfg))
        res = field_step(current, ep, model, act)
        # Enforce §9 guards at runtime
        assert_memory_norm(res.m_next, cfg)
        # L1-delta termination: compare current X to x_next
        delta = float((res.x_next.detach() - current.X.detach()).abs().mean().item())
        # Advance state: X becomes x_next (detach so each step is independent leaf)
        next_state = FieldState(X=res.x_next.detach().requires_grad_(True), m=res.m_next)
        states.append(next_state)
        results.append(res)
        if event_cb: event_cb("field_step", {"t": t, "delta": round(delta, 6)})
        if delta < cfg.rollout_eps:
            return RolloutResult(states=states, results=results, terminated_early=True, steps=t+1)
        current = next_state
    return RolloutResult(states=states, results=results, terminated_early=False, steps=cfg.horizon)
