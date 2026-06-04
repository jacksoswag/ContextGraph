from __future__ import annotations
# Behavioral confirmation on the emergent nested corpus. Characterization tests — they pin the
# observed behavior so a later change shows up as a diff. The generator + scorer live in scripts/.
import sys, dataclasses
from pathlib import Path
import pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from nested_corpus import build_store
from field.config import DEFAULT_CFG
from field.gather import materialize, gather
from graph import GraphStore

def _cfg(**kw): return dataclasses.replace(DEFAULT_CFG, **{"N_max": 150, "k_hop": 4, **kw})

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

# ── the PPR gather localizes on nested data: genericity demotes hubs ───────────────────────
def test_genericity_localizes_on_nested(corpus):
    s, gold, _ = corpus
    # with genericity ON the ranking differs from the uniform-α PPR, and the most-generic (highest
    # global-degree) non-seed node loses relevance share — the leak's localization, now a PPR sink.
    changed = demoted = 0
    for lab in list(gold)[:5]:
        ep, si = materialize(s, [gold[lab]["seed_node"]], _cfg(decay_gamma=1.0))
        if ep.degree is None: continue
        r_on = gather(ep, si, _cfg(decay_gamma=1.0)).relevance()
        r_off = gather(ep, si, _cfg(decay_gamma=0.0)).relevance()
        if not torch.allclose(r_on, r_off): changed += 1
        hub = max((i for i in range(len(ep.node_ids)) if i not in set(si)),
                  key=lambda i: float(ep.degree[i]))
        if float(r_on[hub] / r_on.sum()) < float(r_off[hub] / r_off.sum()): demoted += 1
    assert changed >= 1 and demoted >= 1                    # genericity bites on at least one seed
