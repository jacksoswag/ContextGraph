from __future__ import annotations
# Behavior tests for the node/edge writer (P1): producer clause-edges → sqlite store
# (nodes/edges + info_vector + FTS + vec), idempotent, hyperedges reified as e_-id
# endpoints. Skips when spaCy/fastembed are unavailable.
import sqlite3, sys
from pathlib import Path
import pytest

pytest.importorskip("spacy", reason="spaCy not installed")
pytest.importorskip("fastembed", reason="fastembed not installed")
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.writer import GraphWriter
from graph import GraphStore
from embed import EMBED_DIM

# vec0 virtual tables are unreadable without the extension loaded on the connection.
def _connect(path):
    con = sqlite3.connect(path)
    import sqlite_vec
    con.enable_load_extension(True); sqlite_vec.load(con)
    return con


# A node clause-edge and a nested hyperedge, built directly so the test does not
# depend on the parser (parser behavior is covered by test_extraction.py).
def _node(t, pos="NOUN"): return {"type": "node", "text": t, "pos": pos}
def _edge(s, rel, t, **kw): return {"type": "edge", "rel": rel, "source": s, "target": t,
                                    "_source_text": "src", "_clause_text": "clause", **kw}


def test_writer_creates_nodes_edges_with_embeddings_and_indexes(tmp_path):
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        stats = w.write_clauses([_edge(_node("marie curie"), "discover", _node("radium"))])
    assert stats["nodes"] == 2 and stats["edges"] == 1
    con = _connect(path)
    # nodes carry a 384-d (1536-byte) anchor
    vecs = con.execute("SELECT length(info_vector) FROM nodes").fetchall()
    assert vecs and all(n[0] == EMBED_DIM * 4 for n in vecs)
    # edge carries an anchor too (reified edges are first-class, must be embeddable)
    assert con.execute("SELECT length(info_vector) FROM edges").fetchone()[0] == EMBED_DIM * 4
    # FTS + vec are populated and queryable
    assert con.execute("SELECT count(*) FROM nodes_fts").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM nodes_vec").fetchone()[0] == 2
    con.close()
    # the field's read path sees it
    with GraphStore(path) as s:
        nid = s.find("radium", 1)
        assert nid and s.anchor(nid[0]) is not None
        assert s.neighbors(s.find("marie curie", 1)[0])    # the discover edge is traversable


def test_writer_is_idempotent(tmp_path):
    path = str(tmp_path / "g.sqlite")
    clauses = [_edge(_node("dog"), "chase", _node("cat"))]
    with GraphWriter(path) as w:
        first = w.write_clauses(clauses)
        second = w.write_clauses(clauses)            # re-ingest identical content
    assert first["nodes"] == 2 and first["edges"] == 1
    assert second["nodes"] == 0 and second["edges"] == 0   # nothing new
    con = _connect(path)
    assert con.execute("SELECT count(*) FROM nodes").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM edges").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM nodes_fts").fetchone()[0] == 2   # no FTS dup
    assert con.execute("SELECT count(*) FROM nodes_vec").fetchone()[0] == 2   # no vec dup
    # re-ingest bumps observation count rather than duplicating
    assert con.execute("SELECT count FROM edges").fetchone()[0] == 2
    con.close()


def test_hyperedge_target_is_reified_as_edge_endpoint(tmp_path):
    # scientist --believe--> (smoking --cause--> cancer): the inner edge is the
    # outer edge's target. The writer must store the outer target_id as an e_ id.
    inner = _edge(_node("smoking"), "cause", _node("cancer"))
    outer = _edge(_node("scientist"), "believe", inner)
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        stats = w.write_clauses([outer])
    assert stats["hyperedges"] >= 1
    con = _connect(path)
    # there is an edge whose target_id references another edge (e_ prefix)
    rows = con.execute("SELECT source_id, rel_type, target_id FROM edges WHERE target_id LIKE 'e_%'").fetchall()
    assert rows, "no reified-edge endpoint written"
    s_id, rel, t_id = rows[0]
    assert rel == "believe" and s_id.startswith("n_") and t_id.startswith("e_")
    # the referenced inner edge exists and is itself embeddable (anchored)
    inner_row = con.execute("SELECT length(info_vector) FROM edges WHERE id=?", (t_id,)).fetchone()
    assert inner_row and inner_row[0] == EMBED_DIM * 4
    con.close()


def test_ingest_text_runs_producer_and_writes(tmp_path):
    path = str(tmp_path / "g.sqlite")
    with GraphWriter(path) as w:
        stats = w.ingest("Marie Curie discovered radium. Radium is radioactive.")
    assert stats["nodes"] > 0 and stats["edges"] > 0
    with GraphStore(path) as s:
        assert s.find("radium", 1)
