from __future__ import annotations
# Layer-2 field tests: hyperedge-native traversal + containment. A reified edge
# (e_ id) appearing as an endpoint is bound to its child endpoints so energy stays
# within the hyperedge — and the fact's content is reachable through that binding.
import dataclasses, sqlite3, sys
from pathlib import Path
import pytest

pytest.importorskip("fastembed", reason="fastembed not installed")
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import torch; torch.set_num_threads(1)
from graph.writer import GraphWriter, node_id, edge_id
from graph import GraphStore
from field.config import DEFAULT_CFG
from field.gather import materialize, gather

def _node(t, pos="NOUN"): return {"type": "node", "text": t, "pos": pos}
def _edge(s, rel, t): return {"type": "edge", "rel": rel, "source": s, "target": t,
                              "_source_text": "x", "_clause_text": "x"}

# scientist --believe--> (smoking --cause--> cancer): one reified hyperedge.
@pytest.fixture
def hyper_store(tmp_path):
    path = str(tmp_path / "g.sqlite")
    inner = _edge(_node("smoking"), "cause", _node("cancer"))
    outer = _edge(_node("scientist"), "believe", inner)
    with GraphWriter(path) as w: w.write_clauses([outer])
    return path

def test_store_resolves_edge_id_anchor_text_children(hyper_store):
    e1 = edge_id(node_id("smoking"), "cause", node_id("cancer"))
    with GraphStore(hyper_store) as s:
        assert s.anchor(e1) is not None and s.anchor(e1).shape == (384,)   # edge is anchored
        assert "smoking" in s.text(e1) and "cancer" in s.text(e1)          # readable surface
        kids = s.children(e1)
        assert kids == (node_id("smoking"), node_id("cancer"))             # its endpoints
        assert s.children(node_id("smoking")) is None                      # plain node has none

def test_materialize_pulls_hyperedge_children_and_groups(hyper_store):
    e1 = edge_id(node_id("smoking"), "cause", node_id("cancer"))
    with GraphStore(hyper_store) as s:
        ep, seed_idx = materialize(s, [node_id("scientist")], DEFAULT_CFG)
    ids = set(ep.node_ids)
    # smoking/cancer are NOT k-hop neighbors of scientist — they enter ONLY as e1's children
    assert {node_id("smoking"), node_id("cancer"), e1} <= ids
    assert ep.hyperedges, "no hyperedge group recorded"
    members = [set(ep.node_ids[i] for i in grp) for grp, _w in ep.hyperedges]
    assert {e1, node_id("smoking"), node_id("cancer")} in members

def test_containment_makes_fact_content_reachable(hyper_store):
    # Seeding 'scientist', the fact's content (smoking/cancer) is reachable only through
    # the hyperedge binding. With w_hyper>0 they gain relevance; with w_hyper=0 they don't.
    with GraphStore(hyper_store) as s:
        ep, seed_idx = materialize(s, [node_id("scientist")], DEFAULT_CFG)
    idx = ep.id_to_idx
    i_smoke, i_cancer = idx[node_id("smoking")], idx[node_id("cancer")]
    on = gather(ep, seed_idx, DEFAULT_CFG).relevance()
    off = gather(ep, seed_idx, dataclasses.replace(DEFAULT_CFG, w_hyper=0.0)).relevance()
    assert float(on[i_smoke]) > float(off[i_smoke])     # containment lifts child relevance
    assert float(on[i_cancer]) > float(off[i_cancer])
    assert float(on[i_smoke]) > 0.0 and float(on[i_cancer]) > 0.0

def test_flat_store_fact_materializes_as_hyperedge(tmp_path):
    # _grow_set via containing_edges exposes the reified edge (e_dc) + its endpoints — the fact
    # "dog chase cat" becomes a hyperedge triple (e_dc, dog, cat). No nested e_ as endpoint here;
    # all three members are present, so one hyperedge group is recorded.
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_edge(_node("dog"), "chase", _node("cat"))])
    with GraphStore(path) as s:
        ep, _ = materialize(s, [node_id("dog")], DEFAULT_CFG)
    ids = set(ep.node_ids)
    e_dc = edge_id(node_id("dog"), "chase", node_id("cat"))
    assert {node_id("dog"), node_id("cat"), e_dc} <= ids
    members = [set(ep.node_ids[i] for i in grp) for grp, _w in ep.hyperedges]
    assert {e_dc, node_id("dog"), node_id("cat")} in members
