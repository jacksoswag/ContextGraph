from __future__ import annotations
import numpy as np, torch
import pytest
from field.observables import (support, cluster_mesh, occupancy,
    jaccard_dist, tv_distance, MeshCatalog, StabilizationMonitor)

# ── support / mesh / occupancy ───────────────────────────────────────────────

def test_support_threshold():
    X = torch.tensor([[3.0, 0.0], [0.1, 0.0], [2.0, 0.0], [0.0, 0.0]])  # ē = 9, .01, 4, 0
    assert support(X, tau=0.5) == frozenset({0})         # thr 0.5·9=4.5 -> only node0 (9)
    assert support(X, tau=0.4) == frozenset({0, 2})      # thr 0.4·9=3.6 -> node0 (9), node2 (4)

def test_support_stable_under_small_perturbation():
    torch.manual_seed(4)
    X = torch.tensor([[3.0, 0.0], [3.1, 0.0], [0.2, 0.0], [0.1, 0.0]])  # clear hi/lo gap
    base = support(X, tau=0.5)
    for _ in range(20):
        s = support(X + torch.randn_like(X) * 0.02, tau=0.5)
        assert s == base                                  # membership invariant to ε-noise

def test_support_windowed_matches_mean():
    Xw = torch.stack([torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
                      torch.tensor([[0.0, 0.0], [2.0, 0.0]])])           # mean ē = 2,2
    assert support(Xw, tau=0.5) == frozenset({0, 1})

def test_cluster_mesh_merges_jaccard_near():
    a = frozenset({0, 1, 2}); a2 = frozenset({0, 1, 2})
    b = frozenset({5, 6, 7})
    ids = cluster_mesh([a, a2, b, a], d_mesh=0.3)
    assert ids == [0, 0, 1, 0]                           # a-cluster, then distinct b

def test_cluster_mesh_permutation_stable():
    a = frozenset({0, 1}); b = frozenset({2, 3})
    assert cluster_mesh([a, b, a, b], 0.3) == [0, 1, 0, 1]

def test_occupancy_normalized():
    occ = occupancy([0, 0, 1, 1, 1, 2])
    assert occ[0] == pytest.approx(2 / 6); assert occ[1] == pytest.approx(3 / 6)
    assert sum(occ.values()) == pytest.approx(1.0)

# ── jaccard_dist ─────────────────────────────────────────────────────────────

def test_jaccard_identical():
    assert jaccard_dist(frozenset({0, 1}), frozenset({0, 1})) == pytest.approx(0.0)

def test_jaccard_disjoint():
    assert jaccard_dist(frozenset({0, 1}), frozenset({2, 3})) == pytest.approx(1.0)

def test_jaccard_partial():
    # |{0}| / |{0,1,2}| = 1/3  →  dist = 2/3
    assert jaccard_dist(frozenset({0, 1}), frozenset({0, 2})) == pytest.approx(2 / 3)

def test_jaccard_both_empty():
    assert jaccard_dist(frozenset(), frozenset()) == pytest.approx(0.0)

def test_jaccard_one_empty():
    assert jaccard_dist(frozenset(), frozenset({0})) == pytest.approx(1.0)

# ── tv_distance ──────────────────────────────────────────────────────────────

def test_tv_identical_dists():
    q = {0: 0.5, 1: 0.5}
    assert tv_distance(q, q) == pytest.approx(0.0)

def test_tv_disjoint_dists():
    assert tv_distance({0: 1.0}, {1: 1.0}) == pytest.approx(1.0)

def test_tv_partial():
    # |0.6-0.4|+|0.4-0.6| = 0.2+0.2 → ½·0.4 = 0.2
    assert tv_distance({0: 0.6, 1: 0.4}, {0: 0.4, 1: 0.6}) == pytest.approx(0.2)

def test_tv_missing_key():
    # key absent → treat as 0
    assert tv_distance({0: 1.0}, {0: 0.5, 1: 0.5}) == pytest.approx(0.5)

# ── MeshCatalog (online) ──────────────────────────────────────────────────────

def test_mesh_catalog_identical_gets_same_id():
    cat = MeshCatalog(d_mesh=0.3)
    X = np.ones((4, 4))
    s = frozenset({0, 1, 2})
    id0 = cat.assign(s, X); id1 = cat.assign(s, X)
    assert id0 == id1 == 0 and cat.mesh_count == 1

def test_mesh_catalog_disjoint_gets_different_ids():
    cat = MeshCatalog(d_mesh=0.3)
    X = np.ones((2, 4))
    id0 = cat.assign(frozenset({0, 1}), X)
    id1 = cat.assign(frozenset({2, 3}), X)
    assert id0 != id1 and cat.mesh_count == 2

def test_mesh_catalog_near_merges():
    # {0,1,2} and {0,1,3}: union=4, inter=2, J=0.5 < 0.6 → merge
    cat = MeshCatalog(d_mesh=0.6)          # loose threshold so near sets merge
    X = np.ones((4, 4))
    id0 = cat.assign(frozenset({0, 1, 2}), X)
    id1 = cat.assign(frozenset({0, 1, 3}), X)  # J=0.5 < 0.6 → merge
    assert id0 == id1

def test_mesh_catalog_far_separates():
    cat = MeshCatalog(d_mesh=0.3)
    X = np.ones((4, 4))
    id0 = cat.assign(frozenset({0, 1, 2}), X)
    id1 = cat.assign(frozenset({0, 1, 3}), X)  # J=0.5 > 0.3 → separate
    assert id0 != id1

def test_mesh_catalog_representative_is_mean():
    cat = MeshCatalog(d_mesh=0.5)
    X1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    X2 = np.array([[3.0, 0.0], [0.0, 0.0]])
    s = frozenset({0})
    cat.assign(s, X1); cat.assign(s, X2)    # both map to mesh 0
    rep = cat.representative(0)
    assert np.allclose(rep, (X1 + X2) / 2)

def test_mesh_catalog_frozenset_unordered():
    # frozenset is already unordered; two differently-constructed sets should merge
    cat = MeshCatalog(d_mesh=0.3)
    X = np.ones((3, 4))
    s1 = frozenset({0, 1, 2}); s2 = frozenset({2, 0, 1})  # same set, different literal order
    assert cat.assign(s1, X) == cat.assign(s2, X)

# ── StabilizationMonitor ──────────────────────────────────────────────────────

def test_stab_point_test_triggers_immediately():
    mon = StabilizationMonitor(W=10, H_hold=5, eps_occ=0.05, eps_x=1e-4)
    result = mon.update(0, delta_X_norm=0.0)           # delta < eps_x → immediate
    assert result is True and mon.stabilized_at == 0

def test_stab_not_before_2W():
    mon = StabilizationMonitor(W=3, H_hold=2, eps_occ=0.05, eps_x=1e-9)
    for _ in range(5):                                  # need 6 steps (2W) before TV check
        assert mon.update(0, delta_X_norm=1.0) is False

def test_stab_tv_convergence():
    # W=3, H_hold=2: need 2 consecutive steps with TV<0.05 after first 6
    mon = StabilizationMonitor(W=3, H_hold=2, eps_occ=0.05, eps_x=1e-9)
    # first 2 steps have mixed mesh, rest all mesh 0
    for t in range(20):
        mid = t if t < 2 else 0
        if mon.update(mid, delta_X_norm=1.0):
            assert mon.stabilized_at is not None
            return
    assert False, "monitor never stabilized"

def test_stab_resets_on_tv_spike():
    # stable_count reaches 1, then a new mesh disrupts TV → count resets
    mon = StabilizationMonitor(W=2, H_hold=3, eps_occ=0.05, eps_x=1e-9)
    for _ in range(4): mon.update(0, 1.0)   # reach TV=0 once (step 3: count=1)
    mon.update(1, 1.0)                       # TV spikes → count resets
    assert mon._stable_count == 0

def test_stab_accumulates_all_mesh_ids():
    mon = StabilizationMonitor(W=5, H_hold=3, eps_occ=0.05, eps_x=1e-9)
    for mid in [0, 1, 0, 2, 0]: mon.update(mid, 1.0)
    assert mon.mesh_ids == [0, 1, 0, 2, 0]
