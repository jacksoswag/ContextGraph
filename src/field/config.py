from __future__ import annotations
from dataclasses import dataclass

# FieldConfig: knobs for the convergent-gather field (spec §3, §10). Pure gradient settle —
# rotation + trajectory-classification knobs were retired at G3.
@dataclass
class FieldConfig:
    # state / coupling (§3.2–3.3)
    d: int = 64                    # node state dim
    mu: float = 0.1                # barrier coefficient (‖x‖⁴ term, §3.3)
    beta: float = 2.0              # inhibition sharpness β = gather breadth (§3.3, §3.6)
    # support / mesh / stabilization (§3.5)
    tau_support: float = 0.5       # mesh threshold τ (support_τ)
    d_mesh: float = 0.3            # Jaccard distance for mesh clustering
    eps_occ: float = 0.05          # occupancy TV stabilization threshold ε_occ
    eps_x: float = 1e-4            # settle ΔX convergence threshold
    # integration (§3.4)
    eta: float = 0.01              # step size η (bounded by c/L)
    H_max: int = 500               # max rollout horizon
    H_hold: int = 20               # consecutive sub-ε_x steps required to declare settled
    W: int = 50                    # occupancy window width
    # Sleep / offline learning (§6)
    tau_C: float = 0.99            # EMA rate for C update
    delta_trust: float = 0.05      # trust-region bound ‖ΔC‖

DEFAULT_CFG = FieldConfig()
