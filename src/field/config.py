from __future__ import annotations
from dataclasses import dataclass

# FieldConfig: knobs for the convergent-gather field (spec §3, §10). Post-PPR-gate the per-settle
# engine is a personalized-PageRank solve, so the gradient-integrator knobs (η/H_max/H_hold/ε_x/μ/β/
# σ_anchor/c_seed/d) are gone with the integrator; the surviving knobs shape the active set, the
# coupling W, the genericity leak, and the readout.
@dataclass
class FieldConfig:
    # coupling strength source: "structural" (default) ignores embeddings and weights every edge by
    # struct_edge_w (count·confidence), degree-normalized ⇒ PPR over structure; "semantic" weights each
    # edge by its endpoints' info_vector cosine. Gate 2: structural ≥ semantic on coverage+bridge, so
    # semantics no longer earn a place in the coupling — they stay only at the find_vec grounding seam.
    couple_mode: str = "structural"
    # active-set reachability (§3.1) — _grow_set materialization from the seeds
    k_hop: int = 2                 # active-set depth from seeds in _grow_set (§3.1)
    N_max: int = 512               # active-set node cap, by descending edge weight (§3.1)
    up_max: int = 8                # max containing edges pulled per node when climbing up
    down_decay: float = 0.5        # weight decay per containment level in _grow_set
    # hyperedge containment: extra coupling binding a reified edge (e_ id) to its child endpoints, so
    # PPR mass stays within a hyperedge when it has children. 0 ⇒ flat dyadic.
    w_hyper: float = 0.5
    # decay: per-node genericity leak λ_i (📍1 LOCALIZE decision) — in the PPR family it becomes the
    # per-destination absorption sink in gather (generic hubs absorb faster ⇒ the settle localizes
    # instead of over-spreading to global co-activation).
    decay: float = 1.5             # leak floor λ (locality knob; 📍3-fixed default)
    # genericity slope on the leak: λ_i = decay·(1 + decay_gamma·ln(1+deg_i)) per node, so generic
    # hubs leak faster ⇒ the low→high climb costs more through ambiguous hubs. 0 ⇒ uniform decay.
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
    # child-mesh inheritance (§5)
    k_inherit: int = 4             # parent-mesh nodes inherited as child anchors
    # Sleep / offline learning (§6)
    tau_C: float = 0.99            # EMA rate for C update
    delta_trust: float = 0.05      # trust-region bound ‖ΔC‖

DEFAULT_CFG = FieldConfig()
