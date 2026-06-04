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
            if not t: return f"{self.text(s) or s} {rel}"               # unary intransitive (empty target)
            return f"{self.text(s) or s} {rel} {self.text(t) or t}"
        row = self._con.execute("SELECT text FROM nodes WHERE id=?", (node_id,)).fetchone()
        return row[0] if row else None

    # children: the endpoint ids of a reified edge — its hyperedge members. None for plain nodes; a
    # 1-tuple for a unary intransitive edge (empty target dropped). Drives containment in materialize.
    def children(self, node_id: str) -> tuple[str, ...] | None:
        if not node_id.startswith("e_"): return None
        row = self._con.execute(
            "SELECT source_id, target_id FROM edges WHERE id=?", (node_id,)).fetchone()
        return tuple(x for x in row if x) if row else None

    # triple: a reified edge's raw (source_id, rel_type, target_id) — None for plain nodes. Lets a renderer
    # show an edge RELATIVE to a known endpoint (rel + the other side) instead of the recursively-flattened
    # full surface (which restates a parent fact when the edge is a fact→fact connection).
    def triple(self, node_id: str) -> tuple[str, str, str] | None:
        if not node_id.startswith("e_"): return None
        row = self._con.execute(
            "SELECT source_id, rel_type, target_id FROM edges WHERE id=?", (node_id,)).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def exists(self, node_id: str) -> bool:
        tbl = "edges" if node_id.startswith("e_") else "nodes"
        return self._con.execute(f"SELECT 1 FROM {tbl} WHERE id=?", (node_id,)).fetchone() is not None

    # neighbors: undirected adjacency on E(S) — out-edges ∪ in-edges, (nbr_id, score), self excluded.
    # Score-ordered LIMIT per direction (mirrors containing_edges): bounds the work a generic hub costs
    # so a million-edge node can't drag its whole star into every expansion (the "ultra-dense stall").
    # Default 512 > N_max ⇒ a no-op at corpus scales where max degree < it (the active-set cap prunes
    # below this anyway); it only bites — keeping the strongest edges — on extreme hubs. Order does not
    # affect results downstream (materialize uses set membership + max-weight, both order-free).
    def neighbors(self, node_id: str, limit: int = 512) -> list[tuple[str, float]]:
        out = self._con.execute(
            "SELECT target_id, score FROM edges WHERE source_id=? ORDER BY score DESC LIMIT ?",
            (node_id, limit)).fetchall()
        inc = self._con.execute(
            "SELECT source_id, score FROM edges WHERE target_id=? ORDER BY score DESC LIMIT ?",
            (node_id, limit)).fetchall()
        return [(n, float(s if s is not None else 0.0)) for n, s in out + inc if n and n != node_id]

    # containing_edges: the reified edges (e_ ids) this node is a MEMBER of (source or target),
    # strongest-first — the upward/membership view of incidence, where neighbors() gives the
    # downward/co-endpoint node view. Lets a low-order seed climb to the high-order facts about it.
    def containing_edges(self, node_id: str, limit: int = 8) -> list[tuple[str, float]]:
        rows = self._con.execute(
            "SELECT id, score FROM edges WHERE source_id=? OR target_id=? ORDER BY score DESC LIMIT ?",
            (node_id, node_id, limit)).fetchall()
        return [(r[0], float(r[1] if r[1] is not None else 0.0)) for r in rows]

    # degree: cheap COUNT (no row materialization) — node genericity for specificity weighting
    def degree(self, node_id: str) -> int:
        o = self._con.execute("SELECT COUNT(*) FROM edges WHERE source_id=?", (node_id,)).fetchone()[0]
        i = self._con.execute("SELECT COUNT(*) FROM edges WHERE target_id=?", (node_id,)).fetchone()[0]
        return int(o + i)

    # find_vec: text → nearest ENDPOINTS (nodes ∪ reified edges) by info_vector cosine — the unified
    # seed rule. Embedding match (not lexical) fixes disambiguation (the real entity embeds closer
    # than a same-string disambiguation page), and ranking nodes AND edges together lets a relational
    # query seed a proposition ("japan instance-of …") while a broad one seeds the bare entity. Brute
    # force over the (cached) endpoint matrix — trivial at corpus scale, no vec extension needed.
    def find_vec(self, query: str, k: int = 5) -> list[str]:
        from embed import embed, unpack
        qb = embed(query)
        M, ids = self._endpoint_matrix()
        if qb is None or M is None: return self.find(query, k)   # no embedder/vectors ⇒ lexical
        sims = M @ unpack(qb)
        return [ids[i] for i in np.argsort(-sims)[:k]]

    # _endpoint_matrix: [E,384] unit info_vectors over all nodes+edges, with parallel id list (cached).
    def _endpoint_matrix(self) -> tuple[np.ndarray | None, list[str]]:
        if getattr(self, "_emat", None) is not None: return self._emat, self._eids
        rows = self._con.execute("SELECT id, info_vector FROM nodes WHERE info_vector IS NOT NULL").fetchall()
        rows += self._con.execute("SELECT id, info_vector FROM edges WHERE info_vector IS NOT NULL").fetchall()
        if not rows: self._emat, self._eids = None, []; return None, []
        M = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        self._emat = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        self._eids = [r[0] for r in rows]
        return self._emat, self._eids

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

    # struct_edge_weights: build a {(src_id, tgt_id): multiplier} map for the gather's edge_weights
    # hook. Weight = log1p(count) * score, normalised to [0,1] by the per-store max so the coupling
    # scale is corpus-invariant. count reflects co-occurrence evidence; score reflects extraction
    # confidence. A new edge (count=1, score=0.5) gets weight log1p(1)*0.5 ≈ 0.35; a reinforced
    # edge (count=10, score=0.9) gets log1p(10)*0.9 ≈ 2.1, normalised ~6× stronger. Returns a
    # lightweight object with .get(u_id, v_id) so it satisfies the gather's weights protocol.
    def struct_edge_weights(self) -> "_EdgeWeights":
        if getattr(self, "_ew", None) is not None: return self._ew
        import math
        rows = self._con.execute(
            "SELECT id, source_id, target_id, score, count FROM edges").fetchall()
        raw: dict[tuple[str, str], float] = {}
        for eid, src, tgt, sc, cnt in rows:
            w = math.log1p(cnt or 1) * (sc if sc is not None else 0.5)
            raw[(src, tgt)] = w; raw[(tgt, src)] = w       # symmetric (undirected coupling)
            raw[(src, eid)] = w; raw[(eid, src)] = w       # e_ endpoints exposed symmetrically
            raw[(tgt, eid)] = w; raw[(eid, tgt)] = w
        mx = max(raw.values(), default=1.0)
        normed = {k: v / mx for k, v in raw.items()}
        self._ew = _EdgeWeights(normed)
        return self._ew

    def close(self) -> None: self._con.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class _EdgeWeights:
    def __init__(self, m: dict): self._m = m
    def get(self, u: str, v: str) -> float: return self._m.get((u, v), 1.0)
