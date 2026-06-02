from __future__ import annotations
# Phase 1 — seed-anchored convergent gather (spec §3).
# Step 1 here: the anchor energy term (§3.3) — a true potential, so E stays Lyapunov.
import dataclasses
import torch, pytest
import torch.nn.functional as F
from field.config import FieldConfig, DEFAULT_CFG
from field.coupling import build as build_coupling, Coupling
from field.energy import grad_E, compute_E, Anchor
from field.dynamics import step, rollout
from field.harness import make_single_clique, make_two_cliques_bridge, safe_build, init_X

def _cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, **{"eta": 0.005, "H_max": 4000, "H_hold": 50, **kw})

# ── anchor energy: gradient matches finite-difference of the scalar energy ───────────

def test_anchor_grad_matches_finite_difference():
    ep = make_single_clique(); cfg = _cfg(sigma_anchor=1.3)
    C = build_coupling(ep, cfg)
    Cd = Coupling(C.sym.double(), C.lambda_max, C.R_max, C.L, C.eta_bound)  # f64 for FD accuracy
    N = len(ep.node_ids)
    torch.manual_seed(1)
    X = torch.randn(N, cfg.d, dtype=torch.float64) * 0.3
    g = torch.Generator().manual_seed(0)
    mask = torch.zeros(N, dtype=torch.float64); mask[[0, 2]] = 1.0
    anc = Anchor(mask=mask, target=torch.randn(N, cfg.d, generator=g, dtype=torch.float64))
    g_analytic = grad_E(X, Cd, cfg, anc)
    eps = 1e-6; g_num = torch.zeros_like(X)
    for i in range(N):
        for k in range(cfg.d):
            Xp = X.clone(); Xp[i, k] += eps
            Xm = X.clone(); Xm[i, k] -= eps
            g_num[i, k] = (compute_E(Xp, Cd, cfg, anc) - compute_E(Xm, Cd, cfg, anc)) / (2 * eps)
    assert torch.allclose(g_analytic, g_num, atol=1e-4), \
        f"max |Δ|={float((g_analytic-g_num).abs().max()):.2e}"

# anchor=None must reproduce the un-anchored energy/gradient exactly (backward-compat).
def test_anchor_none_is_identity():
    ep = make_single_clique(); cfg = _cfg()
    C = build_coupling(ep, cfg)
    torch.manual_seed(2); X = torch.randn(len(ep.node_ids), cfg.d)
    assert torch.equal(grad_E(X, C, cfg), grad_E(X, C, cfg, None))
    assert compute_E(X, C, cfg) == compute_E(X, C, cfg, None)

# ── anchor keeps E a Lyapunov function: rollout-with-anchor is monotone non-increasing ──

def test_energy_monotone_with_anchor():
    ep = make_two_cliques_bridge(); cfg = _cfg(sigma_anchor=1.0, H_max=2000)
    C, cfg = safe_build(ep, cfg)
    N = len(ep.node_ids)
    X0 = init_X(ep, cfg, seed=0)
    anc = Anchor(mask=torch.tensor([1.0] + [0.0] * (N - 1)), target=X0.clone())
    X = X0.clone(); e_prev = compute_E(X, C, cfg, anc).item()
    for t in range(cfg.H_max):
        X_prev = X
        X = step(X, C, cfg, anc)
        e = compute_E(X, C, cfg, anc).item()
        assert e <= e_prev + 1e-5, f"E increased @t={t}: {e_prev:.6f}→{e:.6f}"
        e_prev = e
        if (X - X_prev).norm().item() < cfg.eps_x: break

# ── anchor pins the anchored row toward its target; larger σ pins harder ──────────────

def test_anchor_pins_row_and_sigma_tightens():
    ep = make_two_cliques_bridge()
    N = len(ep.node_ids)
    devs = []
    for sigma in (0.1, 1.0, 8.0):
        cfg = _cfg(sigma_anchor=sigma, H_max=4000)
        C, cfg = safe_build(ep, cfg)
        X0 = init_X(ep, cfg, seed=0)
        tgt = torch.zeros(N, cfg.d); tgt[0] = X0[0] * 3.0          # pull node0 to 3× its init
        anc = Anchor(mask=torch.tensor([1.0] + [0.0] * (N - 1)), target=tgt)
        Xh, _ = rollout(X0, C, cfg, anc)
        devs.append(float((Xh[-1][0] - tgt[0]).norm()))
    # higher σ ⇒ settled anchored row sits closer to its target
    assert devs[0] > devs[1] > devs[2], f"σ did not tighten the pin: {devs}"


# ── Step 2/3: active-set materialization (§3.1) + the gather (§3) on a controlled graph ──

import numpy as np
from field.gather import (materialize, gather, hop_distances, seed_init, GatherResult)

# DictStore: in-memory store double (neighbors/anchor) so locality is checked on a known graph.
class DictStore:
    def __init__(self, adj, anchors):
        self._adj = adj; self._anc = anchors
    def neighbors(self, nid): return list(self._adj.get(nid, []))
    def anchor(self, nid): return self._anc.get(nid)
    def text(self, nid): return nid

# a line graph s0 - a - b - c - d (distances 0,1,2,3,4 from the seed). Anchors share a base
# (high cosine ⇒ strong coupling) so the decay term, not anchor dissimilarity, drives hop-decay.
def _line_store(seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = ["s0", "a", "b", "c", "d"]
    chain = [("s0", "a"), ("a", "b"), ("b", "c"), ("c", "d")]
    adj = {n: [] for n in ids}
    for u, v in chain:
        adj[u].append((v, 1.0)); adj[v].append((u, 1.0))
    base = F.normalize(torch.randn(384, generator=g), dim=0)
    anchors = {n: F.normalize(base + 0.05 * torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
               for n in ids}
    return DictStore(adj, anchors), ids

def test_materialize_locality_respects_k_hop():
    store, ids = _line_store()
    cfg = _cfg(k_hop=2, N_max=512)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    names = set(ep.node_ids)
    assert names == {"s0", "a", "b"}          # only ≤2 hops reachable; c (3), d (4) excluded
    assert seed_idx == [0] and ep.node_ids[0] == "s0"
    # every edge endpoint is inside N
    s, d = ep.edge_index
    assert int(s.max()) < len(ep.node_ids) and int(d.max()) < len(ep.node_ids)

def test_materialize_caps_at_N_max_keeping_seeds():
    # star: seed connected to many leaves; N_max forces a cap but the seed survives
    g = torch.Generator().manual_seed(1)
    leaves = [f"L{i}" for i in range(20)]
    adj = {"hub": [(l, 1.0) for l in leaves]}
    for l in leaves: adj[l] = [("hub", 1.0)]
    anc = {n: F.normalize(torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
           for n in ["hub"] + leaves}
    store = DictStore(adj, anc)
    cfg = _cfg(k_hop=2, N_max=6)
    ep, seed_idx = materialize(store, ["hub"], cfg)
    assert len(ep.node_ids) == 6
    assert "hub" in ep.node_ids and seed_idx == [0]

def test_materialize_hop_distances_match_graph():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    dist = hop_distances(ep, seed_idx)
    by_name = {ep.node_ids[i]: dist[i] for i in range(len(ep.node_ids))}
    assert by_name == {"s0": 0, "a": 1, "b": 2, "c": 3, "d": 4}

# ── the gather: converges, pins seeds hot, relevance decays with hop, deterministic ──

def test_gather_settles_and_seed_is_hottest():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512, sigma_anchor=1.0)
    res = gather(*materialize(store, ["s0"], cfg), cfg)
    assert res.steps < cfg.H_max, "gather did not settle before H_max"
    rel = res.relevance()
    assert int(rel.argmax()) == res.seed_idx[0], "seed is not the most-relevant node"

def test_gather_relevance_decays_with_hop():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512, sigma_anchor=1.0)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    res = gather(ep, seed_idx, cfg)
    dist = hop_distances(ep, seed_idx)
    rel = res.relevance().tolist()
    by_hop = {}
    for i, h in enumerate(dist): by_hop.setdefault(h, []).append(rel[i])
    means = [float(np.mean(by_hop[h])) for h in sorted(by_hop)]
    assert all(means[k] > means[k + 1] for k in range(len(means) - 1)), \
        f"relevance not monotone-decreasing by hop: {means}"

def test_gather_anchor_lifts_seed_relevance_vs_unseeded():
    # seeding s0 makes it far more relevant than the same node gets with a different seed (d)
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    ep, _ = materialize(store, ["s0"], cfg)
    si = ep.id_to_idx["s0"]
    rel_seeded = gather(ep, [si], cfg).relevance()[si]
    rel_unseeded = gather(ep, [ep.id_to_idx["d"]], cfg).relevance()[si]
    assert float(rel_seeded) > float(rel_unseeded) + 1e-3

def test_gather_deterministic():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    a = gather(ep, seed_idx, cfg, rng_seed=0).X_star
    b = gather(ep, seed_idx, cfg, rng_seed=0).X_star
    assert torch.equal(a, b)
