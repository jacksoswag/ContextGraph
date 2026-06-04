from __future__ import annotations
# Merge/dedup tests. Decision functions are pure (unit-tested over synthetic scores).
# Apply path tested on a forced merge. Numeric normalisation + BM25 tested directly.
import sqlite3, sys
from pathlib import Path
import pytest

pytest.importorskip("fastembed", reason="fastembed not installed")
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.writer import GraphWriter, node_id
from graph.merge import (MergeConfig, should_merge, bm25_sim, lexical_sim,
                         merge_store, _specificity, _norm_text, _build_idf, _recompute_scores)
from graph import GraphStore

def _n(t): return {"type": "node", "text": t, "pos": "NOUN"}
def _e(s, r, t): return {"type": "edge", "rel": r, "source": s, "target": t,
                         "_source_text": "x", "_clause_text": "x"}
def _connect(path):
    con = sqlite3.connect(path)
    try:
        import sqlite_vec
        con.enable_load_extension(True); sqlite_vec.load(con)
    except Exception:
        pass
    return con


# ── should_merge decision logic ───────────────────────────────────────────────

def test_should_merge_high_cosine_always_merges():
    cfg = MergeConfig()
    # high cosine alone is confident (synonym, low BM25)
    assert should_merge(cosine=0.97, bm25=0.0, spec=0.0, cfg=cfg)
    assert should_merge(cosine=0.99, bm25=0.0, spec=1.0, cfg=cfg)

def test_should_merge_specificity_raises_bar():
    cfg = MergeConfig()
    # cosine 0.89 passes for generic (spec=0, tau=0.86) but not specific (spec=1, tau=0.98)
    assert should_merge(cosine=0.89, bm25=0.0, spec=0.0, cfg=cfg)
    assert not should_merge(cosine=0.89, bm25=0.0, spec=1.0, cfg=cfg)

def test_should_merge_bm25_boost_path():
    cfg = MergeConfig()
    # moderate cosine + strong BM25 → merge via boost path
    assert should_merge(cosine=0.84, bm25=0.80, spec=0.3, cfg=cfg)
    # moderate cosine + weak BM25 → no boost
    assert not should_merge(cosine=0.84, bm25=0.20, spec=0.3, cfg=cfg)

def test_should_merge_homonym_guard():
    cfg = MergeConfig()   # tau (spec=0) = 0.86; boost zone [0.82, 0.86); guard ceiling 0.88
    # in the boost zone, near-identical surface (BM25 0.97) + low cosine ⇒ homonym, block
    assert not should_merge(cosine=0.84, bm25=0.97, spec=0.0, cfg=cfg)
    # same cosine but moderate BM25 (paraphrase, not identical surface) ⇒ boost merges
    assert should_merge(cosine=0.84, bm25=0.60, spec=0.0, cfg=cfg)
    # high cosine clears the direct bar ⇒ guard never consulted (genuine duplicate)
    assert should_merge(cosine=0.90, bm25=0.97, spec=0.0, cfg=cfg)


# ── BM25 + numeric normalisation ─────────────────────────────────────────────

def test_norm_text_strips_commas_and_scales():
    assert _norm_text("1,000,000") == "1000000"
    assert _norm_text("42.3 million") == "42300000"
    assert _norm_text("4.2e9") == "4200000000"


def test_conflict_blocks_merge_at_any_cosine():
    cfg = MergeConfig()
    # an external distinctness signal blocks even at high cosine + high bm25
    assert not should_merge(cosine=0.98, bm25=0.9, spec=0.0, cfg=cfg, conflict=True)

def test_numeric_tokens_normalise_same_value():
    from graph.merge import _numeric_tokens
    # same-value figures (commas vs scale word) normalise equal → no conflict
    assert _numeric_tokens("42,300,000") == _numeric_tokens("42.3 million")
    assert _numeric_tokens("23 may 1945") != _numeric_tokens("8 may 1945")
    # number-free synonyms carry no numeric tokens → guard inert
    assert _numeric_tokens("usa") == frozenset() and _numeric_tokens("united states") == frozenset()

def test_discriminative_conflict_distinguishes_template_slots():
    from graph.merge import _discriminative_conflict as dc
    # share a template, differ in one content-word slot → distinct entities
    assert dc("north korea", "south korea")
    assert dc("united states", "united kingdom")
    assert dc("22 june 1941", "22 july 1941")     # month-name slot the numeric guard can't see
    # function-word / morphological / no-overlap differences are NOT conflicts
    assert not dc("diary of anne frank", "anne frank's diary")  # of/s slot, both function words
    assert not dc("normandy landing", "normandy landings")     # morphological variant
    assert not dc("usa", "united states")                      # no shared template
    assert not dc("black hole", "neutron star")                # differ in both slots

def test_bm25_identical_texts_score_one():
    idf = {"marie": 2.0, "curie": 2.0}
    score = bm25_sim("marie curie", "marie curie", idf, avgdl=2.0)
    assert score == pytest.approx(1.0, abs=0.01)

def test_bm25_number_normalisation():
    # "42 million" and "42000000" normalise to the same token → high BM25
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE nodes(id TEXT, text TEXT)")
    con.execute("INSERT INTO nodes VALUES ('n_1', '42 million'), ('n_2', '42000000')")
    idf, avgdl = _build_idf(con)
    score = bm25_sim("42 million", "42000000", idf, avgdl)
    assert score > 0.8

def test_bm25_disjoint_texts_score_zero():
    idf = {"dog": 2.0, "cat": 2.0}
    assert bm25_sim("dog", "cat", idf, avgdl=1.0) == pytest.approx(0.0)

def test_lexical_sim_token_jaccard():
    assert lexical_sim("united states", "united states") == 1.0
    assert lexical_sim("usa", "united states") == 0.0
    assert 0.0 < lexical_sim("java island", "java") <= 0.5


# ── specificity ───────────────────────────────────────────────────────────────

def test_specificity_inverts_with_degree():
    import numpy as np
    cfg = MergeConfig()
    assert _specificity(1, None, None, cfg) > _specificity(1000, None, None, cfg)

def test_specificity_centroid_distance_raises_score():
    import numpy as np
    cfg = MergeConfig()
    centroid = np.ones(384, dtype=np.float32) / np.sqrt(384)
    far = np.zeros(384, dtype=np.float32); far[0] = 1.0
    assert _specificity(5, far, centroid, cfg) > _specificity(5, centroid, centroid, cfg)


# ── apply path ────────────────────────────────────────────────────────────────

def test_merge_folds_text_repoints_edges_and_logs(tmp_path, monkeypatch):
    import graph.merge as m
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("usa"), "border", _n("canada")),
                         _e(_n("united states"), "border", _n("canada"))])
    stats = merge_store(path, MergeConfig(candidates=5))
    assert stats["merged"] >= 1
    con = _connect(path)
    texts = [r[0] for r in con.execute("SELECT text FROM nodes").fetchall()]
    assert any("|" in t for t in texts)
    edge_rows = con.execute("SELECT source_id, rel_type, target_id FROM edges").fetchall()
    assert len(edge_rows) == len(set(edge_rows))
    log = con.execute("SELECT victim_id, canonical_id FROM sleep_log").fetchall()
    assert log and all(v != c for v, c in log)
    con.close()

def test_edge_candidates_only_within_shared_endpoints(tmp_path, monkeypatch):
    # Edge merge groups by endpoints: [A loves B] / [A adores B] share endpoints (comparable);
    # [A knows C] is a different bucket and must NEVER be paired with them. Force should_merge=True
    # and call the candidate generator directly so the node pass can't perturb structure.
    import graph.merge as m
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("alice"), "loves", _n("bob")),
                         _e(_n("alice"), "adores", _n("bob")),
                         _e(_n("alice"), "knows", _n("carol"))])
    con = _connect(path)
    e_ids = [r[0] for r in con.execute("SELECT id FROM edges WHERE id LIKE 'e_%'").fetchall()]
    all_vecs = m._load_vectors(con); idf, avgdl = m._build_idf(con)
    centroid = m._build_centroid(con, MergeConfig())
    # low fast-reject so the forced decision reaches every same-endpoint pair
    accepted, _ = m._accept_pairs_edges(con, e_ids, all_vecs, idf, avgdl, centroid,
                                        MergeConfig(tau_bm25_boost=0.10))
    for a, b in accepted:
        ea = con.execute("SELECT source_id, target_id FROM edges WHERE id=?", (a,)).fetchone()
        eb = con.execute("SELECT source_id, target_id FROM edges WHERE id=?", (b,)).fetchone()
        assert set(ea) == set(eb)                # only same-endpoint edges ever paired
    assert len(accepted) >= 1                    # the loves/adores-bob pair is comparable
    con.close()


def test_edge_fold_sums_count_and_removes_victim(tmp_path, monkeypatch):
    # Two same-endpoint edges merged: victim edge row gone, count folded into canonical.
    # Stub the node pass so it can't collapse alice/bob into a self-loop under the forced decision.
    import graph.merge as m
    monkeypatch.setattr(m, "_accept_pairs_nodes", lambda *a, **k: ([], {}))
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("alice"), "loves", _n("bob")),
                         _e(_n("alice"), "adores", _n("bob"))])
    before = sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM edges WHERE id LIKE 'e_%'").fetchone()[0]
    stats = merge_store(path, MergeConfig(candidates=5, tau_bm25_boost=0.10))
    con = _connect(path)
    after = con.execute("SELECT COUNT(*) FROM edges WHERE id LIKE 'e_%'").fetchone()[0]
    assert before == 2 and after == 1 and stats["merged_edges"] >= 1
    # surviving edge carries the summed count (each ingested once ⇒ 2)
    cnt = con.execute("SELECT count FROM edges WHERE id LIKE 'e_%'").fetchone()[0]
    assert cnt == 2
    con.close()

def test_scores_recomputed_after_merge(tmp_path):
    import math
    from graph.merge import _recompute_scores
    con = sqlite3.connect(str(tmp_path / "g.sqlite"))
    con.execute("CREATE TABLE edges(id TEXT, source_id TEXT, rel_type TEXT, "
                "target_id TEXT, score REAL, count INTEGER, created_at INTEGER, updated_at INTEGER)")
    con.execute("INSERT INTO edges VALUES('e1','n_a','rel','n_b',0.5,1,0,0)")
    con.execute("INSERT INTO edges VALUES('e2','n_a','rel','n_c',0.5,3,0,0)")
    _recompute_scores(con)
    rows = dict(con.execute("SELECT id, score FROM edges").fetchall())
    assert rows["e1"] == pytest.approx(math.log1p(1) / math.log1p(3), abs=0.001)
    assert rows["e2"] == pytest.approx(1.0, abs=0.001)
    con.close()

def test_merge_is_idempotent(tmp_path, monkeypatch):
    import graph.merge as m
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("usa"), "border", _n("canada")),
                         _e(_n("united states"), "border", _n("canada"))])
    merge_store(path, MergeConfig(candidates=5))
    n_first = sqlite3.connect(path).execute("SELECT count(*) FROM nodes").fetchone()[0]
    second = merge_store(path, MergeConfig(candidates=5))
    n_second = sqlite3.connect(path).execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert second["merged"] == 0 and n_first == n_second
