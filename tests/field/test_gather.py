from __future__ import annotations
# Phase 1 — seed-anchored gather (spec §3), post-PPR-gate: the per-settle engine is a personalized
# PageRank solve on C.sym (the gradient integrator collapsed to its linear solve). Covers active-set
# materialization (§3.1), the PPR settle + genericity localization (§3), and the mesh readout (§3.5).
import dataclasses
import numpy as np
import torch, pytest
import torch.nn.functional as F
from field.config import FieldConfig, DEFAULT_CFG
from field.coupling import build as build_coupling
from field.baseline import personalized_pagerank
from field.episode import Episode
from field.gather import (materialize, gather, hop_distances, build_mesh, Mesh, GatherResult)

def _cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, **kw)

# ── active-set materialization (§3.1) on a controlled graph ──────────────────────────

# DictStore: in-memory store double (neighbors/anchor) so locality is checked on a known graph.
class DictStore:
    def __init__(self, adj, anchors):
        self._adj = adj; self._anc = anchors
    def neighbors(self, nid): return list(self._adj.get(nid, []))
    def anchor(self, nid): return self._anc.get(nid)
    def text(self, nid): return nid

# a line graph s0 - a - b - c - d (distances 0,1,2,3,4 from the seed). Anchors share a base
# (high cosine ⇒ strong coupling) so structural diffusion, not anchor dissimilarity, drives hop-decay.
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
    s, d = ep.edge_index
    assert int(s.max()) < len(ep.node_ids) and int(d.max()) < len(ep.node_ids)

def test_materialize_caps_at_N_max_keeping_seeds():
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

# ── the PPR gather: equals the baseline, pins the seed, decays with hop, localizes, deterministic ──

# the collapse contract: gather relevance IS personalized_pagerank on the same episode's C.sym.
def test_gather_equals_personalized_pagerank():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)                 # decay_gamma=0 (default) ⇒ no genericity sink
    ep, si = materialize(store, ["s0"], cfg)
    r = gather(ep, si, cfg).relevance()
    W = build_coupling(ep, cfg).sym
    assert torch.allclose(r, personalized_pagerank(W, si))

def test_gather_seed_dominates_far_field():
    # PPR is degree-sensitive (no anchor pin), so a degree-1 seed's lone neighbor can tie it — but
    # the seed always outweighs the entire hop≥2 tail (the diffusion concentrates around the seed).
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    rel = gather(ep, seed_idx, cfg).relevance()
    dist = hop_distances(ep, seed_idx)
    far = max(float(rel[i]) for i in range(len(rel)) if dist[i] >= 2)
    assert float(rel[seed_idx[0]]) > far, "seed does not dominate the hop≥2 tail"

def test_gather_relevance_decays_with_hop():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    ep, seed_idx = materialize(store, ["s0"], cfg)
    rel = gather(ep, seed_idx, cfg).relevance().tolist()
    dist = hop_distances(ep, seed_idx)
    by_hop = {}
    for i, h in enumerate(dist): by_hop.setdefault(h, []).append(rel[i])
    means = [float(np.mean(by_hop[h])) for h in sorted(by_hop)]
    # the PPR tail (hop ≥ 1) decays monotonically; the seed outweighs the hop-2+ region. The lone
    # exception is a degree-1 seed vs its single neighbor (seed/hop-1 can tie), so the strict-decay
    # check starts at hop 1 rather than hop 0.
    assert all(means[k] > means[k + 1] for k in range(1, len(means) - 1)), \
        f"PPR tail not monotone-decreasing by hop: {means}"
    assert means[0] > means[2], f"seed does not outweigh the hop-2 region: {means}"

def test_gather_seeding_lifts_relevance_vs_unseeded():
    # seeding s0 makes it far more relevant than the same node gets when d is the seed instead
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
    a = gather(ep, seed_idx, cfg).relevance()
    b = gather(ep, seed_idx, cfg).relevance()
    assert torch.equal(a, b)

# genericity localization (📍1): the per-node leak λ_i=decay·(1+γ·ln deg) becomes a per-destination
# absorption in the PPR. With it ON, a generic hub (high degree) is demoted vs the uniform-α solve.
def test_genericity_absorption_demotes_hub():
    # star s0 - hub - leaf, hub flagged generic by a high global degree
    ids = ["s0", "hub", "leaf"]
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    ei = torch.tensor(list(zip(*edges)), dtype=torch.long)
    deg = torch.tensor([2.0, 60.0, 2.0])                  # hub is a generic high-degree node
    ep = Episode(ids, torch.zeros(3, 384), ei, {n: i for i, n in enumerate(ids)}, degree=deg)
    r_on = gather(ep, [0], _cfg(decay_gamma=1.0)).relevance()
    r_off = gather(ep, [0], _cfg(decay_gamma=0.0)).relevance()
    assert not torch.allclose(r_on, r_off)                                  # genericity changes ranking
    share = lambda r, i: float(r[i] / r.sum())
    assert share(r_on, 1) < share(r_off, 1)                                 # hub demoted under genericity


# ── mesh readout (§3.5) — ranked selection, no provenance tree ──────────────────────

# Diamond hand-set relevance: s0:9 a:4 b:2.25 c:1 (descending).
def _diamond_result():
    ids = ["s0", "a", "b", "c"]
    ep = Episode(ids, torch.zeros(4, 384), torch.zeros(2, 0, dtype=torch.long),
                 {n: i for i, n in enumerate(ids)})
    r = torch.tensor([9.0, 4.0, 2.25, 1.0])
    return GatherResult(r, ep, DEFAULT_CFG, [0])

def test_mesh_nodes_relevance_ranked():
    m = build_mesh(_diamond_result(), tau_rel=0.01)
    rels = [m.relevance[i] for i in m.nodes]
    assert rels == sorted(rels, reverse=True)           # descending relevance order
    assert m.nodes[0] == 0                              # seed is most relevant
    assert set(m.nodes) == {0, 1, 2, 3} and m.seed_roots == [0]

def test_mesh_top_k_caps_but_keeps_seed():
    m = build_mesh(_diamond_result(), tau_rel=0.01, top_k=2)
    assert 0 in m.nodes and len(m.nodes) <= 2

def test_mesh_deterministic():
    a = build_mesh(_diamond_result()); b = build_mesh(_diamond_result())
    assert a.nodes == b.nodes

def test_mesh_on_real_line_gather():
    store, _ = _line_store()
    cfg = _cfg(k_hop=4, N_max=512)
    res = gather(*materialize(store, ["s0"], cfg), cfg)
    m = build_mesh(res, tau_rel=0.02)
    assert m.seed_roots == res.seed_idx
    assert all(i in m.nodes for i in m.seed_roots)    # seeds in mesh
