from __future__ import annotations
# Integration: the full gather pipeline (GraphStore → materialize → settle) on the real concept
# graph S. Skipped when the local store artifact is absent (it is gitignored, not in CI).
import os, dataclasses
from pathlib import Path
import pytest
from field.config import DEFAULT_CFG
from field.gather import gather_from_store, hop_distances

_STORE_PATH = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
pytestmark = pytest.mark.skipif(not Path(_STORE_PATH).exists(),
                                reason=f"graph store {_STORE_PATH} not present (local artifact)")

@pytest.fixture(scope="module")
def store():
    from graph import GraphStore
    s = GraphStore(_STORE_PATH); yield s; s.close()

# keep runtime small: cap the active set, but use the real decay/anchor defaults
def _cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, **{"N_max": 200, "H_max": 3000, **kw})

_SEEDS = ["dog", "france", "music"]

@pytest.mark.parametrize("word", _SEEDS)
def test_real_gather_settles_and_localizes(store, word):
    sid = store.find(word, 1)
    assert sid, f"no node for {word!r}"
    res = gather_from_store(store, sid, _cfg())
    # convergence (§7.1): the damped diffusion settles before H_max
    assert res.steps < res.cfg.H_max, f"{word}: did not settle ({res.steps} steps)"
    # the seed is the most-relevant node (§3.5 — anchor + decay localize to the seed)
    rel = res.relevance()
    assert int(rel.argmax()) == res.seed_idx[0], f"{word}: seed not hottest"

@pytest.mark.parametrize("word", _SEEDS)
def test_real_gather_locality_within_k_hop(store, word):
    sid = store.find(word, 1)
    res = gather_from_store(store, sid, _cfg(k_hop=2))
    # every materialized node is within k_hop of a seed in S (invariant 4); the high-relevance
    # mesh sits at small hop distance (locality), distant nodes decay to ~0.
    dist = hop_distances(res.ep, res.seed_idx)
    rel = res.relevance()
    mx = float(rel.max())
    hot = [i for i in range(len(res.ep.node_ids)) if float(rel[i]) >= 0.05 * mx]
    assert all(0 <= dist[i] <= 2 for i in hot), f"{word}: hot node beyond k_hop reach"

def test_real_gather_deterministic(store):
    sid = store.find("france", 1)
    import torch
    a = gather_from_store(store, sid, _cfg(), rng_seed=0).X_star
    b = gather_from_store(store, sid, _cfg(), rng_seed=0).X_star
    assert torch.equal(a, b)
