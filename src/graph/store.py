from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np

EMBED_DIM = 384

# GraphStore: read-only adapter over the concept-graph sqlite store S (spec §2). Nodes carry a
# frozen 384-d anchor (info_vector BLOB = the embed.py vector); edges carry weight + relation.
# S is immutable at runtime — this class never writes (opened mode=ro).
class GraphStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)

    # anchor: frozen unit anchor a_v ∈ ℝ^384 for a node OR a reified edge (e_ id), else None.
    # Reified edges are first-class endpoints with their own embedding (info_vector on edges).
    def anchor(self, node_id: str) -> np.ndarray | None:
        tbl = "edges" if node_id.startswith("e_") else "nodes"
        row = self._con.execute(f"SELECT info_vector FROM {tbl} WHERE id=?", (node_id,)).fetchone()
        if not row or row[0] is None: return None
        v = np.frombuffer(row[0], dtype=np.float32)
        return v.copy() if v.shape == (EMBED_DIM,) else None

    # text: node surface form (pipe-delimited synonym set), or a reified edge's readable
    # "src rel tgt" surface (recursive — endpoints may themselves be edges).
    def text(self, node_id: str) -> str | None:
        if node_id.startswith("e_"):
            row = self._con.execute(
                "SELECT source_id, rel_type, target_id FROM edges WHERE id=?", (node_id,)).fetchone()
            if not row: return None
            s, rel, t = row
            return f"{self.text(s) or s} {rel} {self.text(t) or t}"
        row = self._con.execute("SELECT text FROM nodes WHERE id=?", (node_id,)).fetchone()
        return row[0] if row else None

    # children: the (source_id, target_id) endpoints of a reified edge — its hyperedge
    # members. None for plain nodes. Drives containment binding in materialize.
    def children(self, node_id: str) -> tuple[str, str] | None:
        if not node_id.startswith("e_"): return None
        row = self._con.execute(
            "SELECT source_id, target_id FROM edges WHERE id=?", (node_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def exists(self, node_id: str) -> bool:
        tbl = "edges" if node_id.startswith("e_") else "nodes"
        return self._con.execute(f"SELECT 1 FROM {tbl} WHERE id=?", (node_id,)).fetchone() is not None

    # neighbors: undirected adjacency on E(S) — out-edges ∪ in-edges, (nbr_id, score), self excluded
    def neighbors(self, node_id: str) -> list[tuple[str, float]]:
        out = self._con.execute(
            "SELECT target_id, score FROM edges WHERE source_id=?", (node_id,)).fetchall()
        inc = self._con.execute(
            "SELECT source_id, score FROM edges WHERE target_id=?", (node_id,)).fetchall()
        return [(n, float(s if s is not None else 0.0)) for n, s in out + inc if n != node_id]

    # degree: cheap COUNT (no row materialization) — node genericity for specificity weighting
    def degree(self, node_id: str) -> int:
        o = self._con.execute("SELECT COUNT(*) FROM edges WHERE source_id=?", (node_id,)).fetchone()[0]
        i = self._con.execute("SELECT COUNT(*) FROM edges WHERE target_id=?", (node_id,)).fetchone()[0]
        return int(o + i)

    # find: text → node-ids via FTS (porter-stemmed), ranked. Seed entry points for interpret (§4).
    def find(self, query: str, k: int = 5) -> list[str]:
        q = query.strip()
        if not q: return []
        try:
            rows = self._con.execute(
                "SELECT id FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?",
                (q, k)).fetchall()
            if rows: return [r[0] for r in rows]
        except sqlite3.OperationalError:
            pass  # malformed FTS query (e.g. punctuation) → fall back to LIKE
        rows = self._con.execute(
            "SELECT id FROM nodes WHERE text LIKE ? LIMIT ?", (f"%{q.lower()}%", k)).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None: self._con.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
