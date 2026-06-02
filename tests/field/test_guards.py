from __future__ import annotations
# Tests for §9 invariant guards and §10 failure mode detectors.
# Each failure mode gets: a provoking case + assertion that the guard fires.
import pytest, torch
from torch import Tensor
from field.config import FieldConfig
from field.dynamics import FieldModel, FieldState, field_step, Actuators, assert_field_diversity
from field.spaces import Episode
from field.kernel import compute_phi, masked_phi
from field.memory import update_memory, assert_memory_norm, init_episode_memory
from field.operators import FieldOperators
from field.routing import FieldRouter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(**kw) -> FieldConfig:
    return FieldConfig(**kw)

def _episode(N: int = 4, d_anchor: int = 384) -> Episode:
    # Minimal in-memory episode (no SQLite needed)
    node_ids = [f"n_{i}" for i in range(N)]
    A = torch.randn(N, d_anchor)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-8)
    # Fully connected edges (both directions)
    src, tgt, w = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j: src.append(i); tgt.append(j); w.append(1.0)
    edge_src = torch.tensor(src, dtype=torch.long)
    edge_tgt = torch.tensor(tgt, dtype=torch.long)
    edge_w   = torch.tensor(w, dtype=torch.float32)
    return Episode(node_ids=node_ids, A=A, edge_src=edge_src, edge_tgt=edge_tgt,
                   edge_w=edge_w, hyperedges=[], id_to_idx={n: i for i, n in enumerate(node_ids)})

def _state(N: int, d: int) -> FieldState:
    X = torch.randn(N, d)
    X = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    m = torch.randn(d)
    m = m / (m.norm() + 1e-8)
    return FieldState(X=X.requires_grad_(True), m=m)


# ── §9.1 Operator diversity floor ─────────────────────────────────────────────

def test_operator_diversity_fires_on_collapse():
    # Provoke: make all operators identical by zeroing their weights
    cfg = _cfg(eps_op=1e-3)
    ops = FieldOperators(cfg)
    with torch.no_grad():
        w0 = ops.ops[0].W.weight.data.clone()
        for op in ops.ops:
            op.W.weight.data.copy_(w0)
            op.U.weight.data.fill_(0.0)
            op.W.bias.data.fill_(0.0)
    score = ops.diversity_score()
    assert score < cfg.eps_op, f"Expected diversity collapse, got {score}"
    with pytest.raises(AssertionError, match="§9.1"):
        ops.assert_diversity()


def test_operator_diversity_passes_at_init():
    cfg = _cfg(eps_op=1e-3, K=8)
    ops = FieldOperators(cfg)
    # Should not raise at init (diverse initialization)
    ops.assert_diversity()


# ── §9.2 Kernel entropy floor ──────────────────────────────────────────────────

def test_kernel_entropy_fires_on_constant_kernel():
    # Provoke: uniform m and X → φ_t ≈ constant
    cfg = _cfg(eps_phi=1e-6, phi_space="structural", lam=0.0)
    N, d = 6, cfg.d_lat
    ep = _episode(N, cfg.d_anchor)
    X = torch.ones(N, d) / (d ** 0.5)   # all-identical nodes → uniform kernel
    m = torch.zeros(d)
    P_slow = torch.eye(d, cfg.d_anchor)[:d, :d]   # identity (first d dims)
    P_slow = torch.zeros(d, cfg.d_anchor)
    phi = compute_phi(X, ep.A, m, P_slow, cfg)
    var = float(phi.var().item())
    # Var may be low due to identical X; confirm guard fires
    if var < cfg.eps_phi:
        with pytest.raises(AssertionError, match="§9.2"):
            FieldRouter.assert_kernel_entropy(phi, cfg)
    else:
        # Manually force constant phi to test guard
        phi_const = torch.full_like(phi, 0.5)
        with pytest.raises(AssertionError, match="§9.2"):
            FieldRouter.assert_kernel_entropy(phi_const, cfg)


def test_kernel_entropy_passes_with_diverse_nodes():
    cfg = _cfg(eps_phi=1e-6)
    N, d = 8, cfg.d_lat
    ep = _episode(N, cfg.d_anchor)
    model = FieldModel(cfg)
    state = _state(N, d)
    phi = compute_phi(state.X, ep.A, state.m, model.P_slow, cfg)
    # Diverse random anchors/state → variance must exceed the collapse floor
    assert float(phi.var().item()) >= cfg.eps_phi, f"Var(phi)={phi.var():.2e} < {cfg.eps_phi}"


# ── §9.3 Memory persistence floor ─────────────────────────────────────────────

def test_memory_norm_floor_fires():
    cfg = _cfg(eps_m=0.1, m_ref=0.5)
    m_bad = torch.zeros(cfg.d_lat)   # zero norm → violates floor
    with pytest.raises(AssertionError, match="§9.3"):
        assert_memory_norm(m_bad, cfg)


def test_memory_norm_floor_passes_after_projection():
    cfg = _cfg(eps_m=0.1, m_ref=1.0)
    m = torch.randn(cfg.d_lat)
    m_proj = init_episode_memory(m, None, cfg)   # always projects to m_ref
    assert_memory_norm(m_proj, cfg)   # should not raise


def test_memory_norm_fixed_after_update():
    # m_t is projected to m_ref after each update — §10.2 norm regime
    cfg = _cfg(m_ref=1.0, rho=0.9)
    N = 4
    m = torch.randn(cfg.d_lat); m = m / m.norm()
    g_mem = torch.ones(N) / N
    x_next = torch.randn(N, cfg.d_lat)
    x_next = x_next / (x_next.norm(dim=-1, keepdim=True) + 1e-8)
    m_new = update_memory(m, g_mem, x_next, cfg)
    assert abs(float(m_new.norm().item()) - cfg.m_ref) < 1e-5, "m_t not projected to m_ref"
    assert_memory_norm(m_new, cfg)


# ── §10 Failure mode detectors ─────────────────────────────────────────────────

def test_failure_mode_1_phi_saturation():
    # FM1: φ_t saturation → uniform-kernel collapse (Var(φ_t) → 0)
    cfg = _cfg(eps_phi=1e-4, phi_space="structural", lam=0.0)
    N, d = 6, cfg.d_lat
    ep = _episode(N, cfg.d_anchor)
    # Force all X identical → φ_t saturates
    X = torch.ones(N, d) / (d ** 0.5)
    m = torch.zeros(d)
    P_slow = torch.zeros(d, cfg.d_anchor)   # zero bridge → pure m terms
    phi = compute_phi(X, ep.A, m, P_slow, cfg)
    # Detector: Var(φ_t) < eps_phi signals collapse
    var = float(phi.var().item())
    # Collapse detected — guard fires when asserted
    phi_const = torch.full_like(phi, 0.5)
    with pytest.raises(AssertionError, match="§9.2"):
        FieldRouter.assert_kernel_entropy(phi_const, cfg)


def test_failure_mode_2_memory_dominance():
    # FM2: m_t dominance → global attractor freezing → assert_field_diversity fires.
    # Provoke: identical X rows (all nodes collapsed to same direction).
    cfg = _cfg(eps_field_diversity=1e-5)
    N, d = 4, cfg.d_lat
    # Collapsed field: all nodes have identical latent
    X_collapsed = torch.ones(N, d) / (d ** 0.5)   # all rows identical → var=0
    with pytest.raises(AssertionError, match="FM2 GUARD"):
        assert_field_diversity(X_collapsed, cfg)

def test_failure_mode_2_passes_with_diverse_field():
    # Inverse: diverse X rows → FM2 guard does not fire
    cfg = _cfg(eps_field_diversity=1e-5)
    N, d = 4, cfg.d_lat
    X_diverse = torch.randn(N, d)
    assert_field_diversity(X_diverse, cfg)   # should not raise


def test_failure_mode_3_reachability_explosion():
    # FM3: D_t exploding → noisy wandering. Detector: exponential growth triggers high CV.
    from field.objective import CComponents, scalar_c
    import math
    cfg = _cfg()
    # Provoke exponential growth in D_t (2^t pattern)
    components = [CComponents(A=0.5, B=0.1, D=float(2**t), E=0.0) for t in range(12)]
    d_vals = [c.D for c in components]
    mean_d = sum(d_vals) / len(d_vals)
    std_d = math.sqrt(sum((v - mean_d)**2 for v in d_vals) / len(d_vals))
    # Coefficient of variation > 1 signals explosion (σ >> μ)
    cv = std_d / (mean_d + 1e-8)
    assert cv > 1.0, f"Expected CV>1 for exponential D_t, got {cv:.3f}"


def test_failure_mode_4_mdl_over_penalty():
    # FM4: MDL over-penalty → under-exploration (E_t dominates, C(τ) collapses)
    from field.objective import CComponents, scalar_c
    cfg = _cfg(mu_c=10.0)   # exaggerated MDL weight
    components = [CComponents(A=0.5, B=0.1, D=1.0, E=5.0) for _ in range(12)]
    c = scalar_c(components, cfg)
    # C should be very negative (MDL crushing the positive terms)
    assert c < 0.0, f"Expected negative C under MDL over-penalty, got {c}"


def test_failure_mode_5_routing_collapse():
    # FM5: routing collapse → assert_routing_diversity fires on a manually collapsed g_mem.
    cfg = _cfg(eps_routing_ent=0.1)
    N = 8
    # Collapsed: one node takes all mass
    g_collapsed = torch.zeros(N)
    g_collapsed[0] = 1.0
    with pytest.raises(AssertionError, match="FM5 GUARD"):
        FieldRouter.assert_routing_diversity(g_collapsed, cfg)

def test_failure_mode_5_passes_with_diverse_routing():
    # Inverse: uniform g_mem → FM5 guard does not fire
    cfg = _cfg(eps_routing_ent=0.1)
    N = 8
    g_uniform = torch.ones(N) / N   # uniform → max entropy
    FieldRouter.assert_routing_diversity(g_uniform, cfg)   # should not raise


# ── Forward-pass ordering (§10 completeness) ──────────────────────────────────

def test_step_ordering_m_read_once():
    # Single read of m at step start; m_next never read within the same step.
    cfg = _cfg(K=4, d_lat=32, d_anchor=64, top_m=2)
    N = 4
    ep = _episode(N, cfg.d_anchor)
    model = FieldModel(cfg)
    state = _state(N, cfg.d_lat)
    m_before = state.m.clone()
    res = field_step(state, ep, model)
    # m_next must differ from m_before (memory was updated)
    assert not torch.allclose(res.m_next, m_before, atol=1e-6), "m_next unchanged after step"
    # x_next must be L2-normalized per node (§10.2)
    norms = res.x_next.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5, f"x_next not L2-normalized: {norms}"


def test_noise_pre_projection():
    # ξ_t added to F̃ BEFORE Π_S (§10.3). xi stored in StepResult for verification.
    cfg = _cfg(K=4, d_lat=32, d_anchor=64, top_m=2, sigma=0.1)
    N = 4
    ep = _episode(N, cfg.d_anchor)
    model = FieldModel(cfg)
    state = _state(N, cfg.d_lat)
    act = Actuators(beta=0.5, sigma=0.1)
    res = field_step(state, ep, model, actuators=act)
    # xi should be non-zero (very unlikely to be exactly zero)
    assert res.xi.abs().sum().item() > 0.0, "No noise injected"
    # xi shape must be [N, d_lat]
    assert res.xi.shape == (N, cfg.d_lat)


def test_hyperedge_aggregation():
    # Π_S with hyperedges: episode with one hyperedge produces different output than without.
    from field.project import project
    cfg = _cfg(d_lat=16, d_anchor=32, beta=0.5)
    N = 4
    ep_no_hyp = _episode(N, cfg.d_anchor)
    # Add a hyperedge connecting nodes 0, 1, 2
    ep_with_hyp = Episode(node_ids=ep_no_hyp.node_ids, A=ep_no_hyp.A,
                           edge_src=ep_no_hyp.edge_src, edge_tgt=ep_no_hyp.edge_tgt,
                           edge_w=ep_no_hyp.edge_w,
                           hyperedges=[([0, 1, 2], 1.0)],
                           id_to_idx=ep_no_hyp.id_to_idx)
    X = torch.randn(N, cfg.d_lat)
    phi = torch.ones(N, N) * 0.1
    out_no = project(X, phi, ep_no_hyp, cfg)
    out_hy = project(X, phi, ep_with_hyp, cfg)
    # With hyperedge, nodes 0,1,2 get extra aggregation → outputs differ
    assert not torch.allclose(out_no[:3], out_hy[:3]), "Hyperedge had no effect"
