from __future__ import annotations
# GraphWriter: the missing text→store link. Reifies the producer's compositional
# clause-edges into the field-consumable schema (nodes/edges + info_vector + FTS +
# vec), idempotently. Every clause-edge becomes a first-class `e_`-id entity with
# its own anchor; when an edge's source/target is itself an edge (a hyperedge), the
# outer edge stores the inner edge's `e_` id as its endpoint — the field reads
# hyperedge-ness straight from the id namespace. Same normalized text → same id, so
# re-ingesting identical content bumps `count` instead of duplicating rows.
import hashlib, sqlite3, time
from pathlib import Path
from ingest.labels import normalize_label
from embed import embed_batch, EMBED_DIM

try: import sqlite_vec
except Exception: sqlite_vec = None

_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, text TEXT, pos TEXT, info_vector BLOB,"
    " count INTEGER DEFAULT 1, created_at INTEGER, updated_at INTEGER)",
    "CREATE TABLE IF NOT EXISTS edges(id TEXT PRIMARY KEY, source_id TEXT, rel_type TEXT, target_id TEXT,"
    " score REAL DEFAULT 0.5, count INTEGER DEFAULT 1, created_at INTEGER, updated_at INTEGER, info_vector BLOB)",
    "CREATE INDEX IF NOT EXISTS edges_src ON edges(source_id)",
    "CREATE INDEX IF NOT EXISTS edges_tgt ON edges(target_id)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(id UNINDEXED, text, tokenize='porter')",
]
# vec0 virtual tables are only created when the sqlite-vec extension loads.
_VEC_SCHEMA = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS nodes_vec USING vec0(id TEXT PRIMARY KEY, info_vector float[384])",
    "CREATE VIRTUAL TABLE IF NOT EXISTS edges_vec USING vec0(id TEXT PRIMARY KEY, info_vector float[384])",
]

def node_id(text: str) -> str: return "n_" + hashlib.md5(normalize_label(text).encode()).hexdigest()
def edge_id(src_id: str, rel: str, tgt_id: str) -> str:
    return "e_" + hashlib.md5(f"{src_id}|{rel}|{tgt_id}".encode()).hexdigest()

# A node target placeholder the producer emits for intransitive clauses; never stored.
_EVENT = "[event]"


class GraphWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._con = sqlite3.connect(self.path)
        self._vec = False
        if sqlite_vec is not None:
            try:
                self._con.enable_load_extension(True); sqlite_vec.load(self._con); self._vec = True
            except Exception: self._vec = False
        self._con.execute("PRAGMA journal_mode=WAL")
        for ddl in _SCHEMA: self._con.execute(ddl)
        if self._vec:
            for ddl in _VEC_SCHEMA: self._con.execute(ddl)
        self._con.commit()

    # ── reification: flatten a clause forest into {node_id:(text,pos)} + {edge_id:(s,rel,t,surface)} ──
    # Returns the id of a committed object, or None for skip (pending/empty/event).
    def _commit(self, obj, nodes: dict, edges: dict) -> str | None:
        if not isinstance(obj, dict): return None
        if obj.get("type") == "node":
            text = (obj.get("text") or "").strip()
            if not text or text == _EVENT or not normalize_label(text): return None
            nid = node_id(text)
            nodes.setdefault(nid, (normalize_label(text), obj.get("pos") or "NOUN"))
            return nid
        if obj.get("type") != "edge" or obj.get("_pending_completion"): return None
        s = self._commit(obj.get("source"), nodes, edges)
        t = self._commit(obj.get("target"), nodes, edges)
        rel = (obj.get("rel") or "").strip().lower()
        if not s or not t or not rel: return None
        eid = edge_id(s, rel, t)
        if eid not in edges:
            edges[eid] = (s, rel, t, float(obj.get("score", 0.5)), self._surface(obj))
        # modifiers (amod 'qualifies', …) attach to THIS edge → the edge is their source
        for m in obj.get("modifiers", []) or []:
            mt = self._commit(m.get("target"), nodes, edges)
            mrel = (m.get("rel") or "").strip().lower()
            if mt and mrel:
                mid = edge_id(eid, mrel, mt)
                edges.setdefault(mid, (eid, mrel, mt, 0.5, self._surface(m.get("target"))))
        return eid

    # Readable anchor text for an object: node→its label, edge→"src rel tgt" (recursive).
    def _surface(self, obj) -> str:
        if not isinstance(obj, dict): return ""
        if obj.get("type") == "node": return normalize_label(obj.get("text") or "")
        s, t = self._surface(obj.get("source")), self._surface(obj.get("target"))
        return f"{s} {obj.get('rel','')} {t}".strip()

    # ── write: upsert nodes+edges, keep FTS+vec in sync, embed only NEW rows ──
    def write_clauses(self, clauses) -> dict:
        nodes: dict[str, tuple[str, str]] = {}
        edges: dict[str, tuple[str, str, str, float, str]] = {}
        for c in clauses: self._commit(c, nodes, edges)
        now = int(time.time())
        con = self._con
        n_new = self._upsert_nodes(nodes, now)
        e_new = self._upsert_edges(edges, now)
        con.commit()
        hyper = sum(1 for s, _r, t, _sc, _sf in edges.values() if s.startswith("e_") or t.startswith("e_"))
        return {"nodes": n_new, "edges": e_new, "hyperedges": hyper,
                "nodes_seen": len(nodes), "edges_seen": len(edges)}

    def _existing(self, table: str, ids: list[str]) -> set[str]:
        out: set[str] = set()
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            q = f"SELECT id FROM {table} WHERE id IN ({','.join('?' * len(chunk))})"
            out.update(r[0] for r in self._con.execute(q, chunk))
        return out

    def _upsert_nodes(self, nodes: dict, now: int) -> int:
        if not nodes: return 0
        ids = list(nodes)
        existing = self._existing("nodes", ids)
        new_ids = [i for i in ids if i not in existing]
        vecs = embed_batch([nodes[i][0] for i in new_ids]) if new_ids else []
        for nid, vec in zip(new_ids, vecs):
            text, pos = nodes[nid]
            self._con.execute("INSERT INTO nodes(id,text,pos,info_vector,count,created_at,updated_at)"
                              " VALUES(?,?,?,?,1,?,?)", (nid, text, pos, vec, now, now))
            self._con.execute("INSERT INTO nodes_fts(id,text) VALUES(?,?)", (nid, text))
            if self._vec and vec is not None:
                self._con.execute("INSERT INTO nodes_vec(id,info_vector) VALUES(?,?)", (nid, vec))
        if existing:
            self._con.executemany("UPDATE nodes SET count=count+1, updated_at=? WHERE id=?",
                                  [(now, i) for i in existing])
        return len(new_ids)

    def _upsert_edges(self, edges: dict, now: int) -> int:
        if not edges: return 0
        ids = list(edges)
        existing = self._existing("edges", ids)
        new_ids = [i for i in ids if i not in existing]
        vecs = embed_batch([edges[i][4] for i in new_ids]) if new_ids else []
        for eid, vec in zip(new_ids, vecs):
            s, rel, t, score, _surface = edges[eid]
            self._con.execute("INSERT INTO edges(id,source_id,rel_type,target_id,score,count,created_at,updated_at,info_vector)"
                              " VALUES(?,?,?,?,?,1,?,?,?)", (eid, s, rel, t, score, now, now, vec))
            if self._vec and vec is not None:
                self._con.execute("INSERT INTO edges_vec(id,info_vector) VALUES(?,?)", (eid, vec))
        if existing:
            self._con.executemany("UPDATE edges SET count=count+1, updated_at=? WHERE id=?",
                                  [(now, i) for i in existing])
        return len(new_ids)

    # ── convenience: run the producer over raw text, then write ──
    def ingest(self, text: str, *, deep: bool = False) -> dict:
        from ingest.editor import ingest_text
        clauses = list(ingest_text(text))
        return self.write_clauses(clauses)

    def close(self) -> None: self._con.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
