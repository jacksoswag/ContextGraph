from __future__ import annotations
# Behavioral confirmation on the emergent nested corpus. Characterization tests — they pin the
# observed behavior so a later change shows up as a diff. The generator + scorer live in scripts/.
import sys, dataclasses
from pathlib import Path
import pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from nested_corpus import build_store
from climb_eval import safety
from field.config import DEFAULT_CFG
from field.gather import materialize, gather
from graph import GraphStore

def _cfg(**kw): return dataclasses.replace(DEFAULT_CFG, **{"N_max": 150, "H_max": 4000, "k_hop": 4, **kw})

@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("nested")
    store, gold, meta = build_store(d / "c.sqlite", d / "g.json", n_edges=300, n_atoms=110,
                                    p_nest=0.55, seed=0, quiet=True)
    s = GraphStore(store)
    yield s, gold, meta
    s.close()

# ── corpus is genuinely, emergently nested (order derived, not assigned) ───────────────────
def test_order_is_emergent_and_deep(corpus):
    _, gold, meta = corpus
    assert meta["max_order"] >= 4                          # deep nesting emerged
    assert len(meta["order_hist"]) >= 4                    # a spread of orders, not one stratum
    assert meta["n_seeds"] >= 10 and gold                  # specific seeds with bounded spines exist

def test_gold_ids_exist_and_hubs_emerge(corpus):
    s, gold, _ = corpus
    for g in gold.values():
        for eid, _o in g["spine"]: assert s.exists(eid)    # derived ids match stored ids exactly
    degs = sorted(s.degree(g["seed_node"]) for g in gold.values())
    assert degs[-1] >= 3 * max(degs[0], 1)                 # heavy-tailed reuse ⇒ hubs emerged

# ── climb mechanics: reification reach + containment unpack ────────────────────────────────
def test_hyperedge_children_unpack(corpus):
    s, gold, _ = corpus
    # a reified fact binds to BOTH its endpoints as a hyperedge clique when all three are kept —
    # the containment-unpack that lets a fact's content co-activate (cap may drop some, so scan).
    for lab in gold:
        ep, _ = materialize(s, [gold[lab]["seed_node"]], _cfg())
        if ep.hyperedges:
            members, _w = ep.hyperedges[0]
            assert len(members) == 3 and ep.node_ids[members[0]].startswith("e_")
            assert all(0 <= m < len(ep.node_ids) for m in members)
            return
    pytest.fail("no hyperedge clique materialized across seeds")

# ── the field settles safely on nested data (Lyapunov + bounded) ──────────────────────────
def test_field_settles_on_nested(corpus):
    s, gold, _ = corpus
    for lab in list(gold)[:5]:
        ep, si = materialize(s, [gold[lab]["seed_node"]], _cfg(decay_gamma=1.0))
        settled, bnd, _rise = safety(gather(ep, si, _cfg(decay_gamma=1.0)))
        assert settled and bnd                             # converged, net-descending, ‖x‖≤R_max
