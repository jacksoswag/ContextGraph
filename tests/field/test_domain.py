from __future__ import annotations
# Adaptive domain controller: moving-domain load/unload around the field (physics untouched).
import dataclasses
import numpy as np, torch
import torch.nn.functional as F
import pytest
from field.config import DEFAULT_CFG
from field.domain import TreeContext, episode_from_nodes, adaptive_gather, DomainConfig

class _Store:
    def __init__(self, adj, anc): self._adj = adj; self._anc = anc
    def neighbors(self, nid): return list(self._adj.get(nid, []))
    def anchor(self, nid): return self._anc.get(nid)
    def text(self, nid): return nid

# a long chain n0-n1-…-n{N-1}; anchors share a base (cos≈1 ⇒ strong coupling) so energy can
# propagate node-to-node and the front can walk the chain.
def _chain_store(N=14, sim=0.02, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(384, generator=g), dim=0)
    ids = [f"n{i}" for i in range(N)]
    adj = {i: [] for i in ids}
    for i in range(N - 1):
        adj[ids[i]].append((ids[i + 1], 1.0)); adj[ids[i + 1]].append((ids[i], 1.0))
    anc = {i: F.normalize(base + sim * torch.randn(384, generator=g), dim=0).numpy().astype(np.float32) for i in ids}
    return _Store(adj, anc), ids

# config tuned so the front demonstrably advances (low decay; distribution-relative thresholds)
def _cfg(): return dataclasses.replace(DEFAULT_CFG, decay=0.4, H_max=4000)
def _dcfg(**kw):
    base = dict(eps_low=0.01, eps_commit=0.05, q_commit=0.70, commit_floor=0.005,
                max_phases=20, max_live=8, loads_per_phase=4, anchor_ttl=2)
    return DomainConfig(**{**base, **kw})

# ── episode_from_nodes ────────────────────────────────────────────────────────────

def test_episode_from_nodes_edges_within_set():
    store, ids = _chain_store(6)
    ctx = TreeContext(store)
    ep = episode_from_nodes(ctx, ["n1", "n2", "n3"])
    assert set(ep.node_ids) == {"n1", "n2", "n3"}
    s, d = ep.edge_index
    assert int(s.max()) < 3 and int(d.max()) < 3            # only internal edges (n1-n2, n2-n3)
    assert ep.A.shape == (3, 384)

# ── termination / bounds / determinism ──────────────────────────────────────────────

def test_terminates_and_records_trace():
    store, _ = _chain_store(14)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg())
    assert 1 <= res.phases <= 20 and len(res.trace) == res.phases

def test_loads_bounded_per_phase():
    store, _ = _chain_store(14)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(loads_per_phase=4))
    assert all(p["loaded"] <= 4 for p in res.trace)

def test_live_set_bounded():
    store, _ = _chain_store(20)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(max_live=8))
    assert res.peak_live <= 8 + 2, f"live exceeded bound: {res.peak_live}"

def test_deterministic():
    store, _ = _chain_store(14)
    a = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg())
    b = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg())
    assert a.committed == b.committed and a.trace == b.trace

def test_seed_always_committed_and_anchor():
    store, _ = _chain_store(14)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg())
    assert "n0" in res.committed                            # the seed anchor is never lost

# ── the headline property: bounded live, growing committed (front travels) ──────────

def test_committed_exceeds_live_window():
    # long chain + small live window: the front must sweep more than it ever holds live
    store, _ = _chain_store(20)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(max_live=8, max_phases=25))
    assert len(res.committed) > res.peak_live, \
        f"committed {len(res.committed)} did not exceed live window {res.peak_live}"

def test_front_travels_along_chain():
    # reach (provenance hop-distance of committed from seed) must grow beyond the seed's neighborhood
    store, _ = _chain_store(20)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(max_phases=25))
    final_reach = res.trace[-1]["max_reach"]
    assert final_reach >= 3, f"front did not travel (reach {final_reach})"
    # reach is monotone non-decreasing over phases (front only advances)
    reaches = [p["max_reach"] for p in res.trace]
    assert reaches == sorted(reaches)

def test_culling_occurs():
    store, _ = _chain_store(20)
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(max_live=8))
    assert any(p["culled"] > 0 for p in res.trace)         # cold/expired nodes are unloaded

# ── selectivity: at equal distance, an off-topic node ranks below on-topic ──────────
# (at the low decay needed for the front to travel, off-topic dead-ends can still leak in — the
#  robust property is that the field RANKS them below on-topic context, not that it excludes them;
#  hard exclusion is the high-decay regime, i.e. the reach↔selectivity tradeoff the measurement maps.)

def test_offtopic_ranks_below_ontopic():
    # n0 — n1 (similar, on-topic) and n0 — junk (dissimilar) both 1 hop from the seed
    store, ids = _chain_store(9, sim=0.02)
    g = torch.Generator().manual_seed(9)
    store._anc["junk"] = F.normalize(torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
    store._adj["n0"].append(("junk", 1.0)); store._adj["junk"] = [("n0", 1.0)]
    res = adaptive_gather(TreeContext(store), ["n0"], _cfg(), _dcfg(max_phases=25))
    junk_r = res.relevance.get("junk", 0.0)                 # absent ⇒ not committed ⇒ ranked below
    assert junk_r < res.relevance["n1"], \
        f"off-topic not ranked below on-topic: junk={junk_r} n1={res.relevance['n1']}"
