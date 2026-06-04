from __future__ import annotations
from dataclasses import dataclass

# FieldConfig: knobs for the convergent-gather field (spec §3, §10). Pure gradient settle —
# rotation + trajectory-classification knobs retired at G3; config B/C coherence leak + active-set
# builders (_khop/_climb) cut at finetune pass (spec §2 CUT NOW).
@dataclass
class FieldConfig:
    # state / coupling (§3.2–3.3)
    d: int = 64                    # node state dim
    mu: float = 0.1                # barrier coefficient (‖x‖⁴ term, §3.3)
    beta: float = 2.0              # inhibition sharpness β = gather breadth (§3.3, §3.6)
    # coupling strength source: "structural" (default) ignores embeddings and weights every edge by
    # struct_edge_w (count·confidence), degree-normalized ⇒ a nonlinear PPR; "semantic" weights each
    # edge by its endpoints' info_vector cosine. Gate 2: structural ≥ semantic on coverage+bridge, so
    # semantics no longer earn a place in the dynamics — they stay only at the find_vec grounding seam.
    couple_mode: str = "structural"
    # gather: seed anchoring + active-set reachability (§3.1–3.3, §10)
    sigma_anchor: float = 1.0      # anchor potential strength σ (tie-to-seed/parent)
    c_seed: float = 0.7            # seed hot-init fraction of R_max (§3.2)
    k_hop: int = 2                 # active-set depth from seeds in _grow_set (§3.1)
    N_max: int = 512               # active-set node cap, by descending edge weight (§3.1)
    up_max: int = 8                # max containing edges pulled per node when climbing up
    down_decay: float = 0.5        # weight decay per containment level in _grow_set
    # hyperedge containment: extra coupling binding a reified edge (e_ id) to its child
    # endpoints, so energy stays within a hyperedge when it has children. 0 ⇒ flat dyadic.
    w_hyper: float = 0.5
    # decay: per-node leak λ‖x‖² on non-anchored rows (PPR-teleport analogue, 📍1 decision) —
    # makes the settled fixed point localize instead of over-spreading to global co-activation.
    decay: float = 1.5             # leak strength λ (locality knob; 📍3-fixed default)
    # genericity slope on the leak: λ_i = decay·(1 + decay_gamma·ln(1+deg_i)) per node, so generic
    # hubs leak faster ⇒ the low→high climb costs more through ambiguous hubs (the hill's steepness).
    # 0 ⇒ uniform decay (scalar). Stays a PSD diagonal potential ⇒ E remains Lyapunov.
    decay_gamma: float = 0.0
    # readout breadth S*: build_mesh keeps the top `target_size` relevance-ranked nodes — the single
    # "how much context" knob. CONSTANT by decision: the S* sweep showed answer coverage is
    # monotone-then-FLAT in S* (steep to ~34, plateau to 70, NO distraction drop). 34 ≫ old 15.
    target_size: int = 34
    # query_w: readout query-focus weight at build_mesh — default on at low weight (A/B benchmark flag).
    # query_w > 0 tilts structural relevance toward query-cosine at READOUT only; growth stays blind.
    query_w: float = 0.0
    # support / mesh (§3.5)
    tau_support: float = 0.5       # mesh threshold τ (support_τ)
    eps_x: float = 1e-4            # settle ΔX convergence threshold
    # integration (§3.4)
    eta: float = 0.01              # step size η (bounded by c/L)
    H_max: int = 4000              # max rollout horizon (§10)
    H_hold: int = 50               # consecutive sub-ε_x steps required to declare settled (§10)
    # child-mesh inheritance (§5)
    k_inherit: int = 4             # parent-mesh nodes inherited as child anchors
    # Sleep / offline learning (§6)
    tau_C: float = 0.99            # EMA rate for C update
    delta_trust: float = 0.05      # trust-region bound ‖ΔC‖

DEFAULT_CFG = FieldConfig()
