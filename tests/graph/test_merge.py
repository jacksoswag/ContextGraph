from __future__ import annotations
# P3 merge/dedup tests. The merge DECISION is a pure function (unit-tested here over
# synthetic cosine/lexical/structural scores); real-embedding behavior is measured in the
# benchmark. The APPLY path (fold text, re-point edges, dedup parallels, log) is tested on
# a forced merge. Skips without fastembed (writer needs it).
import sqlite3, sys
from pathlib import Path
import pytest

pytest.importorskip("fastembed", reason="fastembed not installed")
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.writer import GraphWriter, node_id
from graph.merge import (MergeConfig, should_merge, should_merge_edge,
                         lexical_sim, merge_store, _specificity)
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


def test_should_merge_decision_logic():
    cfg = MergeConfig()
    # high embedding cosine alone is confident (catches usa≈united states, low lexical)
    assert should_merge(cosine=0.97, lexical=0.0, structural=0.0, degree=3, cfg=cfg)
    # moderate cosine + similar string but DISJOINT neighborhoods ⇒ homonym, do not merge
    assert not should_merge(cosine=0.86, lexical=0.5, structural=0.0, degree=3, cfg=cfg)
    # moderate cosine confirmed by strong structural overlap ⇒ merge
    assert should_merge(cosine=0.86, lexical=0.5, structural=0.7, degree=3, cfg=cfg)
    # generic (high-degree) nodes need a higher bar — same scores, more neighbors ⇒ hold
    assert not should_merge(cosine=0.905, lexical=0.0, structural=0.0, degree=5000, cfg=cfg)


def test_lexical_sim_token_jaccard():
    assert lexical_sim("united states", "united states") == 1.0
    assert lexical_sim("usa", "united states") == 0.0          # no shared tokens
    assert 0.0 < lexical_sim("java island", "java") <= 0.5     # partial overlap


def test_merge_folds_text_repoints_edges_and_logs(tmp_path, monkeypatch):
    # Force every candidate pair to merge so we exercise the APPLY path deterministically.
    import graph.merge as m
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        # two near-duplicate surfaces for one concept + shared structure
        w.write_clauses([_e(_n("usa"), "border", _n("canada")),
                         _e(_n("united states"), "border", _n("canada"))])
    stats = merge_store(path, MergeConfig(candidates=5))
    assert stats["merged"] >= 1
    con = _connect(path)
    # the two 'usa'/'united states' nodes collapsed to one; canada + 1 survivor remain
    texts = [r[0] for r in con.execute("SELECT text FROM nodes").fetchall()]
    assert any("|" in t for t in texts)                        # folded synonym set
    # the duplicate 'border canada' edges collapsed (no parallel edge to the same target)
    edge_rows = con.execute("SELECT source_id, rel_type, target_id FROM edges").fetchall()
    assert len(edge_rows) == len(set(edge_rows))               # no parallel duplicates
    # sleep_log recorded the merge with its scores
    log = con.execute("SELECT victim_id, canonical_id, cosine FROM sleep_log").fetchall()
    assert log and all(v != c for v, c, _cos in log)
    con.close()


def test_should_merge_edge_specificity_gates_threshold():
    import numpy as np
    cfg = MergeConfig()
    # generic proposition (low specificity) → bar near tau_mid_edge (0.86); merges at 0.87
    assert should_merge_edge(cosine=0.87, src_cos=0.0, tgt_cos=0.0, spec=0.0, cfg=cfg)
    # specific proposition (high specificity) → bar near tau_embed_edge (1.0); 0.87 does NOT merge
    assert not should_merge_edge(cosine=0.87, src_cos=0.0, tgt_cos=0.0, spec=1.0, cfg=cfg)
    # mid cosine but both endpoint cosines above tau_struct_edge → structural confirm merges
    assert should_merge_edge(cosine=0.88, src_cos=0.72, tgt_cos=0.75, spec=0.5, cfg=cfg)
    # mid cosine but one endpoint too different → no merge
    assert not should_merge_edge(cosine=0.88, src_cos=0.72, tgt_cos=0.40, spec=0.5, cfg=cfg)


def test_specificity_inverts_with_degree():
    import numpy as np
    cfg = MergeConfig()
    # high degree → low specificity (generic, easy to merge)
    s_low = _specificity(1000, None, None, cfg)
    # low degree → high specificity (specific, hard to merge)
    s_high = _specificity(1, None, None, cfg)
    assert s_high > s_low

def test_specificity_centroid_distance_raises_score():
    import numpy as np
    cfg = MergeConfig()
    centroid = np.ones(384, dtype=np.float32) / np.sqrt(384)
    far = np.zeros(384, dtype=np.float32); far[0] = 1.0   # orthogonal to centroid
    near = centroid.copy()
    s_far = _specificity(5, far, centroid, cfg)
    s_near = _specificity(5, near, centroid, cfg)
    assert s_far > s_near   # farther from generic centroid = more specific


def test_merge_is_idempotent(tmp_path, monkeypatch):
    import graph.merge as m
    monkeypatch.setattr(m, "should_merge", lambda **kw: True)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        w.write_clauses([_e(_n("usa"), "border", _n("canada")),
                         _e(_n("united states"), "border", _n("canada"))])
    merge_store(path, MergeConfig(candidates=5))
    n_after_first = sqlite3.connect(path).execute("SELECT count(*) FROM nodes").fetchone()[0]
    second = merge_store(path, MergeConfig(candidates=5))      # nothing left to merge
    n_after_second = sqlite3.connect(path).execute("SELECT count(*) FROM nodes").fetchone()[0]
    assert second["merged"] == 0 and n_after_first == n_after_second
