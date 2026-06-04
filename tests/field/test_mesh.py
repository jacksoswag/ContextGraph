from __future__ import annotations
# Recursive collapse-to-mesh gather + weighted-personalization PPR. mesh_gather is the multi-hop spine:
# a single settle reaches ~1 hop, the recursion climbs a chain. Synthetic stores need fastembed (the
# writer embeds node/edge surfaces); decision-level PPR is tested on hand matrices.
import sys
from pathlib import Path
import pytest, torch

pytest.importorskip("fastembed", reason="fastembed not installed")
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from field.baseline import personalized_pagerank
from field.loop import mesh_gather, MESH_CFG
from graph.writer import GraphWriter

def _n(t): return {"type": "node", "text": t, "pos": "NOUN"}
def _e(s, r, t): return {"type": "edge", "rel": r, "source": s, "target": t,
                         "_source_text": "x", "_clause_text": "x"}

def _chain_store(path):                                    # a -loves-> b -knows-> c -met-> d -saw-> e
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("alice"), "loves", _n("bob")),
                         _e(_n("bob"), "knows", _n("carol")),
                         _e(_n("carol"), "met", _n("dave")),
                         _e(_n("dave"), "saw", _n("erin"))])


# ── weighted-personalization PPR ──────────────────────────────────────────────

def _W():
    # 0—1—2—3 path graph
    W = torch.zeros(4, 4)
    for i, j in [(0, 1), (1, 2), (2, 3)]: W[i, j] = W[j, i] = 1.0
    return W

def test_ppr_uniform_seed_backward_compat():
    # seed_idx=[0] must be identical to a one-hot teleport on node 0 (the old API preserved)
    a = personalized_pagerank(_W(), [0])
    b = personalized_pagerank(_W(), teleport=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(a, b, atol=1e-5) and a.sum().item() == pytest.approx(1.0, abs=1e-3)

def test_ppr_weighted_teleport_shifts_mass():
    # teleporting to the far end (node 3) shifts mass toward 3 vs teleporting to node 0
    r3 = personalized_pagerank(_W(), teleport=torch.tensor([0.0, 0.0, 0.0, 1.0]))
    r0 = personalized_pagerank(_W(), teleport=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert float(r3[3]) > float(r0[3]) and float(r3[3]) > float(r3[0])

def test_ppr_teleport_weights_are_relative():
    # heavier weight on 0 than 3 ⇒ 0 outranks 3 (teleport is normalized internally)
    r = personalized_pagerank(_W(), teleport=torch.tensor([3.0, 0.0, 0.0, 1.0]))
    assert float(r[0]) > float(r[3])

def test_ppr_uniform_teleport_is_background():
    # all-ones teleport ⇒ the un-personalized background PageRank (degree-symmetric on a path: ends < middle)
    r = personalized_pagerank(_W(), teleport=torch.ones(4))
    assert float(r[1]) > float(r[0]) and float(r[2]) > float(r[3])


# ── mesh_gather ───────────────────────────────────────────────────────────────

def test_mesh_gather_empty_seeds():
    assert mesh_gather(object(), []).node_ids == []

def test_mesh_gather_climbs_the_chain(tmp_path):
    # seed on `alice`; the recursion must reach a deep edge (carol/dave/erin) a 1-hop settle wouldn't.
    from graph import GraphStore
    path = str(tmp_path / "g.sqlite"); _chain_store(path)
    s = GraphStore(path)
    seed = [r[0] for r in s._con.execute("SELECT id FROM nodes WHERE text='alice'")]
    m = mesh_gather(s, seed, bg_tau=0.0, novelty_eps=0.0, max_rounds=5)
    texts = " ".join(s.text(e) or "" for e in m.node_ids)
    assert all(e.startswith("e_") for e in m.node_ids)         # mesh is reified facts
    assert len(m.node_ids) == len(set(m.node_ids))             # no edge collapsed twice
    assert "knows" in texts or "met" in texts or "saw" in texts  # climbed past the 1-hop `loves`
    s.close()

def test_mesh_gather_respects_max_rounds(tmp_path):
    from graph import GraphStore
    path = str(tmp_path / "g.sqlite"); _chain_store(path)
    s = GraphStore(path)
    seed = [r[0] for r in s._con.execute("SELECT id FROM nodes WHERE text='alice'")]
    one = mesh_gather(s, seed, bg_tau=0.0, novelty_eps=0.0, max_rounds=1)
    five = mesh_gather(s, seed, bg_tau=0.0, novelty_eps=0.0, max_rounds=5)
    # more rounds reach at least as far (monotone accumulation), and one round can't cover the whole chain
    assert set(one.node_ids) <= set(five.node_ids)
    assert len(five.node_ids) >= len(one.node_ids)
    s.close()
