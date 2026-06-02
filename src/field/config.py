from __future__ import annotations
from dataclasses import dataclass, field as _f

# Open design axes and fixed scaffold constants for the Graph-Field System v5.3.
# All stability constants live here as named values — nothing buried in code bodies.

@dataclass
class FieldConfig:
    # ── Open axes (§4 guide) ──────────────────────────────────────────────────
    phi_space: str = "joint"             # "joint" | "structural" | "semantic"
    estimator: str = "straight_through"  # "reinforce" | "straight_through" | "gumbel"
    # C→dynamics wiring flags (all on by default, §8.3)
    c_wiring: dict = _f(default_factory=lambda: {"A": True, "B": True, "D": True, "E": True})
    learning_regime: str = "hybrid"      # "hybrid" | "dist_match" | "struct_inv"
    mem_cross_ep: str = "decay"          # "decay" | "reset" | "persist"
    K: int = 8                           # operator count
    d_lat: int = 128                     # structural latent dim
    top_m: int = 4                       # routing sparsity (‖g‖₀ ≤ top_m)
    knn_k: int = 10                      # anchor k-NN for topo binding (§10.5)

    # ── Fixed scaffold constants (spec §2, §6, §8, §9, §10) ──────────────────
    d_anchor: int = 384                  # frozen MiniLM anchor dim (EMBED_DIM)
    m_ref: float = 1.0                   # memory norm target (m_ref >= eps_m, §10.2)
    eps_m: float = 0.1                   # §9.3 memory persistence floor
    tau_P: float = 0.99                  # P_slow EMA rate (§2.1.1)
    rho: float = 0.9                     # memory directional mixing rate (§2.3)
    beta: float = 0.5                    # Π_S blend strength default (§6, §10.2)
    sigma: float = 0.01                  # base noise std (§10.3)
    eps: float = 1e-8                    # normalization epsilon
    eps_phi: float = 1e-6               # §9.2 kernel entropy floor: Var(φ_t) >= eps_phi
    eps_op: float = 1e-3                # §9.1 operator diversity floor: E[‖F_i-F_j‖²] >= eps_op
    C_phi: float = 10.0                 # φ_t logit clamp (§10.2)
    lam: float = 0.5                    # semantic anchor weight in φ_t (§2.4)
    gamma: float = 0.99                 # C(τ) temporal discount (§8.2)
    lam_B: float = 1.0                  # B_t weight in C scalar (§8.2)
    mu_c: float = 0.1                   # E_t weight in C scalar (§8.2)
    eta_bind: float = 0.1              # anchor binding loss weight (§10.5)
    eta_topo: float = 0.05             # topology consistency loss weight (§10.5)
    horizon: int = 12                   # max rollout steps (§6)
    rollout_eps: float = 1e-4          # L1-delta termination threshold (§6)

    # ── C control bus stability parameters (§8.3, §10.6) — not open axes ────
    # All gains/ranges are named constants; actuators are hard-clipped.
    alpha_c: float = 0.9               # EMA smoothing coefficient
    gain_A: float = 0.2                # A_t → β modulation gain
    gain_B: float = 0.1                # B_t → step size modulation gain
    gain_D: float = 0.05               # D_t → σ modulation gain
    gain_E: float = 0.1                # E_t → Sleep trigger gain
    beta_base: float = 0.5
    beta_min: float = 0.1
    beta_max: float = 0.9
    sigma_min: float = 0.001
    sigma_max: float = 0.1
    step_min: float = 0.1
    step_max: float = 1.0
    sleep_on_thresh: float = 0.7       # Sleep trigger hysteresis ON threshold
    sleep_off_thresh: float = 0.3      # Sleep trigger hysteresis OFF threshold

# Singleton default used when callers don't supply a config.
DEFAULT_CFG = FieldConfig()
