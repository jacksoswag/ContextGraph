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

# keep runtime small: cap the active set, but use the real decay defaults
def _cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, **{"N_max": 200, **kw})

_SEEDS = ["dog", "france", "music"]

@pytest.mark.parametrize("word", _SEEDS)
def test_real_gather_localizes_to_seed(store, word):
    sid = store.find(word, 1)
    assert sid, f"no node for {word!r}"
    res = gather_from_store(store, sid, _cfg())
    # the PPR settle localizes to the seed: it outweighs the entire region beyond its 1-hop ring
    # (§3.5 — teleport + genericity sink concentrate mass on the seed neighborhood).
    rel = res.relevance()
    dist = hop_distances(res.ep, res.seed_idx)
    far = [float(rel[i]) for i in range(len(rel)) if dist[i] >= 2]
    assert float(rel[res.seed_idx[0]]) > (max(far) if far else 0.0), f"{word}: seed not dominant"

@pytest.mark.parametrize("word", _SEEDS)
def test_real_gather_locality_within_k_hop(store, word):
    sid = store.find(word, 1)
    res = gather_from_store(store, sid, _cfg(k_hop=2))
    # every materialized node is within k_hop of a seed in S (invariant 4); the high-relevance
    # mesh sits at small hop distance (locality), distant nodes decay to ~0.
    dist = hop_distances(res.ep, res.seed_idx)
    rel = res.relevance()
    mx = float(rel.max())
    # e_ (reified fact) nodes enter via containment, not lateral hops — hop_distances returns -1
    # for them since neighbors() has no edges from e_ ids. Only check regular entity nodes (n_).
    hot = [i for i in range(len(res.ep.node_ids))
           if float(rel[i]) >= 0.05 * mx and not res.ep.node_ids[i].startswith("e_")]
    assert all(0 <= dist[i] <= 2 for i in hot), f"{word}: hot entity beyond k_hop reach"

def test_real_gather_deterministic(store):
    sid = store.find("france", 1)
    import torch
    a = gather_from_store(store, sid, _cfg()).relevance()
    b = gather_from_store(store, sid, _cfg()).relevance()
    assert torch.equal(a, b)
