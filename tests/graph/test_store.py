from __future__ import annotations
# Unit tests for GraphStore, focused on struct_edge_weights (Phase 2 prep) and
# the neighbors/containing_edges/degree accessors used by the gather pipeline.
import math, sqlite3, sys
from pathlib import Path
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.store import GraphStore, _EdgeWeights


# ── minimal in-memory store fixture ──────────────────────────────────────────

def _make_store(tmp_path):
    path = str(tmp_path / "t.sqlite")
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY, text TEXT, pos TEXT,
            info_vector BLOB, count INTEGER DEFAULT 1,
            created_at INTEGER DEFAULT 0, updated_at INTEGER DEFAULT 0);
        CREATE TABLE edges(id TEXT PRIMARY KEY, source_id TEXT, rel_type TEXT,
            target_id TEXT, score REAL DEFAULT 0.5, count INTEGER DEFAULT 1,
            created_at INTEGER DEFAULT 0, updated_at INTEGER DEFAULT 0,
            info_vector BLOB);
        CREATE VIRTUAL TABLE nodes_fts USING fts5(id UNINDEXED, text);
    """)
    # three nodes, two edges with different count/score
    con.execute("INSERT INTO nodes VALUES('n_a','alpha','NOUN',NULL,1,0,0)")
    con.execute("INSERT INTO nodes VALUES('n_b','beta','NOUN',NULL,1,0,0)")
    con.execute("INSERT INTO nodes VALUES('n_c','gamma','NOUN',NULL,1,0,0)")
    con.execute("INSERT INTO edges VALUES('e_ab','n_a','rel','n_b',0.8,4,0,0,NULL)")
    con.execute("INSERT INTO edges VALUES('e_bc','n_b','rel','n_c',0.5,1,0,0,NULL)")
    con.commit(); con.close()
    return GraphStore(path)


def test_struct_edge_weights_ordering(tmp_path):
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    # e_ab: log1p(4)*0.8 ≈ 1.20  e_bc: log1p(1)*0.5 ≈ 0.35  → e_ab normalised higher
    w_ab = ew.get("n_a", "n_b")
    w_bc = ew.get("n_b", "n_c")
    assert w_ab > w_bc, f"high-count high-score edge should outweigh low: {w_ab} vs {w_bc}"
    s.close()


def test_struct_edge_weights_normalised_to_one(tmp_path):
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    # strongest edge normalises to 1.0
    w_ab = ew.get("n_a", "n_b")
    assert abs(w_ab - 1.0) < 1e-6, f"max weight should be 1.0, got {w_ab}"
    s.close()


def test_struct_edge_weights_symmetric(tmp_path):
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    assert ew.get("n_a", "n_b") == ew.get("n_b", "n_a")
    s.close()


def test_struct_edge_weights_unknown_pair_returns_one(tmp_path):
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    assert ew.get("n_a", "n_c") == 1.0    # no direct edge between a and c
    s.close()


def test_struct_edge_weights_cached(tmp_path):
    s = _make_store(tmp_path)
    ew1 = s.struct_edge_weights()
    ew2 = s.struct_edge_weights()
    assert ew1 is ew2    # same object returned (cached)
    s.close()


def test_struct_edge_weights_e_endpoints_included(tmp_path):
    # e_ ids (reified edges) must also appear in the weight map so the gather's
    # edge_weights() can resolve multipliers for hyperedge-native episodes.
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    assert ew.get("n_a", "e_ab") > 0.0    # reified edge is reachable from its source
    assert ew.get("e_ab", "n_b") > 0.0    # and from its target
    s.close()


def test_edge_weights_formula(tmp_path):
    s = _make_store(tmp_path)
    ew = s.struct_edge_weights()
    # e_ab (count=4, score=0.8) should have weight log1p(4)*0.8 / max_raw
    # e_bc (count=1, score=0.5) should have weight log1p(1)*0.5 / max_raw
    raw_ab = math.log1p(4) * 0.8
    raw_bc = math.log1p(1) * 0.5
    mx = raw_ab    # e_ab is the max
    assert abs(ew.get("n_b", "n_c") - raw_bc / mx) < 1e-5
    s.close()
