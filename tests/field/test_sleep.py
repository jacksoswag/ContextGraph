from __future__ import annotations
# Phase 6 — Sleep / StructureRecon (spec §6): learned per-edge multipliers on C_sym.
import dataclasses
import numpy as np, torch
import torch.nn.functional as F
import pytest
from field.config import DEFAULT_CFG
from field.sleep import EdgeWeights, train
from field.gather import materialize, gather

class _Store:
    def __init__(self, adj, anc): self._adj = adj; self._anc = anc
    def neighbors(self, nid): return list(self._adj.get(nid, []))
    def anchor(self, nid): return self._anc.get(nid)
    def text(self, nid): return nid

# s0 → t_close (similar, well-gathered target), t_far (dissimilar, UNDER-gathered target);
# noise (similar, 2-hop via t_close → OVER-gathered but NOT a seed neighbor).
def _store(seed=5):
    g = torch.Generator().manual_seed(seed)
    base = F.normalize(torch.randn(384, generator=g), dim=0)
    near = lambda: F.normalize(base + 0.04 * torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
    far = lambda: F.normalize(torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
    anc = {"s0": near(), "t_close": near(), "t_far": far(), "noise": near()}
    adj = {"s0": [("t_close", 1.0), ("t_far", 1.0)],
           "t_close": [("s0", 1.0), ("noise", 1.0)],
           "t_far": [("s0", 1.0)], "noise": [("t_close", 1.0)]}
    return _Store(adj, anc)

# ── EdgeWeights container ─────────────────────────────────────────────────────────────

def test_edgeweights_undirected_and_default():
    w = EdgeWeights()
    assert w.get("a", "b") == 1.0                     # default = bootstrap
    w.set("b", "a", 1.3)
    assert w.get("a", "b") == 1.3 and w.get("b", "a") == 1.3   # undirected key

def test_edgeweights_save_load(tmp_path):
    w = EdgeWeights(); w.set("x", "y", 0.7); w.set("p", "q", 1.4)
    p = tmp_path / "w.json"; w.save(p)
    w2 = EdgeWeights.load(p)
    assert w2.get("y", "x") == pytest.approx(0.7) and w2.get("q", "p") == pytest.approx(1.4)

# ── StructureRecon delta-rule ─────────────────────────────────────────────────────────

# at the tight default λ the under-gathered (dissimilar) target neighbor's edge is strengthened
def test_sleep_strengthens_undergathered_target():
    w = train(_store(), ["s0"], DEFAULT_CFG, epochs=4, lr=0.2, trust=0.5)
    assert w.get("s0", "t_far") > 1.02, "under-gathered target edge not strengthened"

# under a BROAD gather (low λ) the 2-hop noise IS over-gathered → its in-edge is weakened
def test_sleep_weakens_overgathered_noise():
    broad = dataclasses.replace(DEFAULT_CFG, decay=0.6)
    w = train(_store(), ["s0"], broad, epochs=4, lr=0.2, trust=0.5)
    assert w.get("t_close", "noise") < 0.98, "over-gathered noise edge not weakened"

def test_sleep_respects_trust_region():
    w = train(_store(), ["s0"], DEFAULT_CFG, epochs=20, lr=1.0, trust=0.3)
    assert all(0.3 - 1e-6 <= m <= 1.3 + 1e-6 for _, m in w.items())   # |m−1| ≤ trust

def test_sleep_support_is_subset_of_edges():
    store = _store()
    w = train(store, ["s0"], DEFAULT_CFG, epochs=2)
    real = {tuple(sorted((u, v))) for u, nbrs in store._adj.items() for v, _ in nbrs}
    for key, _m in w.items():                                         # Sleep never invents an edge
        assert tuple(sorted(key)) in real

def test_sleep_deterministic():
    a = train(_store(), ["s0"], DEFAULT_CFG, epochs=3)
    b = train(_store(), ["s0"], DEFAULT_CFG, epochs=3)
    assert {k: round(m, 9) for k, m in a.items()} == {k: round(m, 9) for k, m in b.items()}

def test_learned_weights_change_the_gather():
    store = _store()
    w = train(store, ["s0"], DEFAULT_CFG, epochs=4, lr=0.2, trust=0.5)
    ep, si = materialize(store, ["s0"], DEFAULT_CFG)
    r0 = gather(ep, si, DEFAULT_CFG).relevance()
    r1 = gather(ep, si, DEFAULT_CFG, weights=w).relevance()
    assert not torch.allclose(r0, r1)                # learned C changes the settled relevance
    # the under-gathered target is pulled harder after Sleep
    j = ep.id_to_idx["t_far"]
    assert float(r1[j]) > float(r0[j])
