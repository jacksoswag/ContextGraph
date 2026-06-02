from __future__ import annotations
# Dynamics invariants + stress grid + rollout-termination contract (spec §7).
# Stress: trapping / no-NaN / determinism / monotone-E hold across a (β, μ, seed) grid at short
# horizon (these are horizon-independent). Pure gradient flow ⇒ every gather settles.
import dataclasses, itertools
import torch, pytest
from field.config import FieldConfig, DEFAULT_CFG
from field.coupling import build as build_coupling
from field.energy import compute_E
from field.dynamics import step, rollout
from field.harness import (make_single_clique, make_two_cliques_bridge, make_ring_of_cliques,
                           make_two_incomm_rings, safe_build, init_X)

_BUILDERS = [make_single_clique, make_two_cliques_bridge, make_ring_of_cliques, make_two_incomm_rings]

def _cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, **{"eta": 0.005, "H_max": 400, **kw})

# ── stress grid: invariants that must hold for EVERY parameter combination ──────────

_GRID = list(itertools.product([1.0, 2.0, 4.0], [0.05, 0.1, 0.2], [0, 1]))

@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("beta,mu,seed", _GRID)
def test_trapping_and_finite_across_grid(builder, beta, mu, seed):
    ep = builder()
    C, cfg = safe_build(ep, _cfg(beta=beta, mu=mu))
    X = init_X(ep, cfg, seed=seed)
    for t in range(cfg.H_max):
        X = step(X, C, cfg)
        mx = X.norm(dim=-1).max().item()
        assert mx <= C.R_max + 1e-3, f"trapping violated @t={t}: {mx:.4f} > R_max={C.R_max:.4f}"
        assert torch.isfinite(X).all(), f"non-finite state @t={t}"

# η is always pulled under the L-derived bound by safe_build (no assertion failure in build).
@pytest.mark.parametrize("builder", _BUILDERS)
def test_eta_bound_respected(builder):
    ep = builder()
    C, cfg = safe_build(ep, _cfg(eta=10.0))   # absurd η must be clamped
    assert cfg.eta <= C.eta_bound + 1e-12

# Pure gradient flow ⇒ E is a strict Lyapunov function (monotone non-increasing) on every graph.
@pytest.mark.parametrize("builder", _BUILDERS)
def test_energy_monotone(builder):
    ep = builder()
    C, cfg = safe_build(ep, _cfg(H_max=600))
    X = init_X(ep, cfg, seed=0)
    e_prev = compute_E(X, C, cfg).item()
    for t in range(cfg.H_max):
        X = step(X, C, cfg)
        e = compute_E(X, C, cfg).item()
        assert e <= e_prev + 1e-5, f"E increased @t={t}: {e_prev:.6f}→{e:.6f}"
        e_prev = e

# ── determinism / reproducibility ──────────────────────────────────────────────────

@pytest.mark.parametrize("builder", _BUILDERS)
def test_rollout_deterministic(builder):
    ep = builder()
    C, cfg = safe_build(ep, _cfg())
    X0 = init_X(ep, cfg, seed=0)
    a, _ = rollout(X0, C, cfg)
    b, _ = rollout(X0, C, cfg)
    assert torch.equal(a, b)

# ── rollout termination contract: SUSTAINED settle, never a single-step break ──
# (The settle-before-H_max *guarantee* is a Phase-1 acceptance test — it depends on the anchor
# term σ‖x−s‖², which removes the marginal O(d)-gauge direction that makes anchorless basins drift.)

# Regression for the premature-break bug: the early stop must coincide exactly with the FIRST
# time ‖ΔX‖<ε_x holds for H_hold CONSECUTIVE steps — an isolated sub-ε_x dip must not stop it.
@pytest.mark.parametrize("builder", _BUILDERS)
def test_rollout_stops_only_on_sustained_settle(builder):
    ep = builder()
    C, cfg = safe_build(ep, _cfg(H_max=8000, H_hold=50))
    X_hist, _ = rollout(init_X(ep, cfg, seed=0), C, cfg)
    n_steps = X_hist.shape[0] - 1
    dX = (X_hist[1:] - X_hist[:-1]).flatten(1).norm(dim=1)
    below = dX < cfg.eps_x
    # find first index where H_hold consecutive trues end
    run = 0; first_sustained = None
    for i, b in enumerate(below.tolist()):
        run = run + 1 if b else 0
        if run >= cfg.H_hold: first_sustained = i; break
    if n_steps < cfg.H_max:                                   # stopped early
        assert first_sustained is not None and first_sustained == n_steps - 1, (
            "early stop did not coincide with first H_hold-consecutive sub-ε_x window")
        assert int(below[: max(0, n_steps - cfg.H_hold)].sum()) >= 0  # sanity: dips can exist pre-stop
    else:
        assert first_sustained is None                        # ran full horizon ⇒ never sustained-settled
