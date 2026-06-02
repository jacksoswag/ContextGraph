from __future__ import annotations
import torch, math
from torch import Tensor
from dataclasses import dataclass, field
from .config import FieldConfig
from .dynamics import StepResult, Actuators

# C(τ) — trajectory functional as control bus + scalar diagnostic (spec §8).
# All components are computed on POST-update state, stored DETACHED (C ≠ loss).
# Per-step components: A_t (alignment), B_t (stability), D_t (reachability), E_t (MDL).
# The control pipeline (§8.3, §10.6) ensures no instantaneous dynamics→C→dynamics loop.

@dataclass
class CComponents:
    A: float = 0.0   # structural consistency (alignment form, large=good)
    B: float = 0.0   # local stability (‖x_{t+1} - Π_S(x_t)‖², small=good)
    D: float = 0.0   # reachability gain (Δreach_S, integer, non-diff)
    E: float = 0.0   # MDL compression penalty

# Per-trajectory running state for the control bus (§8.3, §10.6).
@dataclass
class ControlBusState:
    ema: dict = field(default_factory=lambda: {"A": 0.0, "B": 0.0, "D": 0.0, "E": 0.0})
    run_mean: dict = field(default_factory=lambda: {"A": 0.0, "B": 0.0, "D": 0.0, "E": 0.0})
    run_var: dict = field(default_factory=lambda: {"A": 1.0, "B": 1.0, "D": 1.0, "E": 1.0})
    step_count: int = 0
    sleep_state: bool = False          # hysteresis-maintained Sleep trigger
    _prev_components: CComponents = field(default_factory=CComponents)

# Compute per-step C components from a StepResult (all DETACHED, §8.1).
def compute_components(res: StepResult, x_prev: Tensor, phi_prev: Tensor,
                       reach_prev: int, node_ids_visited: set,
                       cfg: FieldConfig) -> CComponents:
    with torch.no_grad():
        # A_t = x_{t+1}^T W_hat x_{t+1}  (normalized adjacency alignment form)
        # W_hat = φ / (row_sum + ε) — row-normalized adjacency
        phi = res.phi.detach()
        row_sum = phi.sum(dim=-1, keepdim=True).clamp(min=cfg.eps)
        W_hat = phi / row_sum
        x_n = res.x_next.detach()
        A_t = float((x_n * (W_hat @ x_n)).sum().item()) / max(x_n.shape[0], 1)
        # B_t = ‖x_{t+1} - Π_S(x_t)‖²  — already have x_next and can approximate
        # Use y (pre-noise pre-proj) as proxy for Π_S(x_t) when full recompute is costly
        B_t = float(((x_n - res.y.detach())**2).mean().item())
        # D_t = Δreach_S — change in number of distinct nodes visited
        reach_new = len(node_ids_visited)
        D_t = float(max(reach_new - reach_prev, 0))
        # E_t = MDL(τ_≤t) — approximate as negative entropy of routing distribution
        alpha = res.alpha.detach()
        ent = -(alpha * (alpha + cfg.eps).log()).sum(dim=-1).mean().item()
        E_t = -ent   # lower entropy = more compressed = higher MDL penalty proxy
    return CComponents(A=A_t, B=B_t, D=D_t, E=E_t)

# Update the running EMA and running stats for bounded standardization (§10.6 steps 1-3).
# Uses LAGGED components (from previous step) — §10.6 step 1 enforced by caller.
def update_control_bus(bus: ControlBusState, components: CComponents,
                        cfg: FieldConfig) -> ControlBusState:
    import copy
    new_bus = copy.deepcopy(bus)
    new_bus._prev_components = components   # store for next step's lagged read
    keys = ["A", "B", "D", "E"]
    raw = {"A": components.A, "B": components.B, "D": components.D, "E": components.E}
    for k in keys:
        r = raw[k]
        # Step 2: EMA smoothing
        new_bus.ema[k] = cfg.alpha_c * bus.ema[k] + (1.0 - cfg.alpha_c) * r
        # Step 3: running mean/var (Welford online update)
        n = bus.step_count + 1
        delta = r - bus.run_mean[k]
        new_bus.run_mean[k] = bus.run_mean[k] + delta / n
        delta2 = r - new_bus.run_mean[k]
        new_bus.run_var[k] = max(bus.run_var[k] + (delta * delta2 - bus.run_var[k]) / n, 1e-8)
    new_bus.step_count = bus.step_count + 1
    return new_bus

# Compute actuators for the NEXT step from LAGGED (previous) components (§10.6 step 4).
# bus holds the state after update_control_bus has processed the t-1 components.
def actuators_from_bus(bus: ControlBusState, cfg: FieldConfig) -> Actuators:
    # Step 3: bounded standardization c = tanh((ema - mu) / (sqrt(var) + eps)) ∈ (-1, 1)
    def _c(k: str) -> float:
        ema = bus.ema[k]
        mu = bus.run_mean[k]
        std = math.sqrt(bus.run_var.get(k, 1.0)) + cfg.eps
        return math.tanh((ema - mu) / std)
    c_A, c_B, c_D, c_E = _c("A"), _c("B"), _c("D"), _c("E")
    # Step 4: actuator map + hard clip
    # A_t controls β (high alignment → softer projection; low → stronger)
    beta_raw = cfg.beta_base + cfg.gain_A * c_A if cfg.c_wiring.get("A") else cfg.beta
    beta = max(cfg.beta_min, min(cfg.beta_max, beta_raw))
    # B_t controls step size (large instability → smaller step)
    step_raw = 1.0 - cfg.gain_B * abs(c_B) if cfg.c_wiring.get("B") else 1.0
    step = max(cfg.step_min, min(cfg.step_max, step_raw))
    # D_t controls σ (exploration: high reachability → reduce noise; low → increase)
    sigma_raw = cfg.sigma * (1.0 - cfg.gain_D * c_D) if cfg.c_wiring.get("D") else cfg.sigma
    sigma = max(cfg.sigma_min, min(cfg.sigma_max, sigma_raw))
    # E_t controls Sleep trigger (MDL penalty → on/off hysteresis)
    sleep = bus.sleep_state
    if cfg.c_wiring.get("E"):
        e_norm = max(0.0, min(1.0, 0.5 + 0.5 * c_E))
        if not sleep and e_norm >= cfg.sleep_on_thresh:  sleep = True
        elif sleep and e_norm <= cfg.sleep_off_thresh:   sleep = False
    return Actuators(beta=beta, sigma=sigma, step_scale=step, sleep=sleep)

# Scalar diagnostic C(τ) = Σ_t γ^t [A_t·D_t − λ_B·B_t − μ·E_t] (spec §8.2).
# Pure diagnostic — never used as a loss or gradient source.
def scalar_c(components_list: list[CComponents], cfg: FieldConfig) -> float:
    total, discount = 0.0, 1.0
    for c in components_list:
        total += discount * (c.A * c.D - cfg.lam_B * c.B - cfg.mu_c * c.E)
        discount *= cfg.gamma
    return total
