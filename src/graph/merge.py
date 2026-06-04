from __future__ import annotations
# P3 node merge/dedup — rebuilds the absent "sleep" merge (the sleep_log producer). Same
# shape recoverable from that schema: FTS/vec NN candidate generation → cosine + lexical +
# structural (shared-neighbor) scoring → density-adjusted threshold → fold the victim's
# synonyms + edges into a canonical. Distinct concepts with similar surfaces (java island
# vs java) are held apart by the STRUCTURAL signal — identical surfaces are already one node
# (the writer keys by normalized text), so merge's job is the similar-but-distinct case.
# Runs as a batch pass (merge_store) or per-node live (merge_nodes over new ids).
import hashlib, math, sqlite3, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try: import sqlite_vec
except Exception: sqlite_vec = None

@dataclass
class MergeConfig:
    candidates: int = 8         # NN candidates examined per node
    tau_embed: float = 0.93     # cosine at which embedding alone is confident (any structure)
    tau_mid: float = 0.82       # cosine floor for a structurally-confirmed merge
    tau_struct: float = 0.30    # shared-neighbor Jaccard required at mid cosine
    density_penalty: float = 0.012   # how much node genericity (log degree) raises tau_embed
    # e_ (proposition) merge thresholds — tighter than nodes because merging distinct facts is worse
    tau_embed_edge: float = 1.0     # high-specificity propositions: needs near-identical embedding
    tau_mid_edge: float = 0.86      # low-specificity: merges with endpoint-cosine structural confirm
    tau_struct_edge: float = 0.70   # endpoint agreement required at mid cosine (both src+tgt similar)
    # centroid-distance specificity: weights for combining degree + semantic distance from generic centre
    w_deg: float = 0.5          # weight for inverted-degree term in specificity score
    w_dist: float = 0.5         # weight for centroid-distance term
    centroid_scale: float = 0.8 # tanh scale for normalising centroid distance to [0,1]
    centroid_top_k: int = 30    # how many highest-degree nodes define the generic centroid

_SLEEP_LOG = (
    "CREATE TABLE IF NOT EXISTS sleep_log(id INTEGER PRIMARY KEY AUTOINCREMENT, pass_num INTEGER,"
    " victim_id TEXT, canonical_id TEXT, victim_text TEXT, canonical_text TEXT, victim_pos TEXT,"
    " canonical_pos TEXT, cosine REAL, lexical REAL, density INTEGER, threshold REAL, ts INTEGER)")


# Merge decision (pure): high cosine is confident on its own (catches synonyms with no
# shared tokens, e.g. usa≈united states); moderate cosine merges only with structural
# confirmation (rejects same-string homonyms whose neighborhoods are disjoint). Genericity
# (degree) raises the embedding-alone bar so hub nodes are not collapsed cheaply.
def should_merge(*, cosine: float, lexical: float, structural: float, degree: int, cfg: MergeConfig) -> bool:
    bar = cfg.tau_embed + cfg.density_penalty * math.log10(1.0 + max(0, degree))
    if cosine >= bar: return True
    if cosine >= cfg.tau_mid and structural >= cfg.tau_struct: return True
    return False

# specificity ∈ [0,1]: 1 = maximally specific (high bar), 0 = generic (low bar).
# Combines inverted degree (structural) + distance from generic centroid (semantic).
# centroid should be the mean unit-vector of the cfg.centroid_top_k highest-degree nodes.
def _specificity(deg: int, emb: np.ndarray | None, centroid: np.ndarray | None, cfg: MergeConfig) -> float:
    deg_term = 1.0 / (1.0 + math.log1p(max(0, deg)))   # ∈ (0,1], falls as deg rises
    if emb is not None and centroid is not None:
        dist = float(np.linalg.norm(emb - centroid))
        dist_term = math.tanh(dist / max(cfg.centroid_scale, 1e-6))
    else:
        dist_term = deg_term   # no embedder: fall back to degree-only
    return cfg.w_deg * deg_term + cfg.w_dist * dist_term

# Merge decision for reified edge (e_) endpoints. Bar floats from tau_mid_edge (generic
# proposition) to tau_embed_edge (specific) driven by the specificity of the lower-scoring
# endpoint. Structural confirmation = endpoint cosine agreement (both src+tgt similar).
def should_merge_edge(*, cosine: float, src_cos: float, tgt_cos: float,
                      spec: float, cfg: MergeConfig) -> bool:
    bar = cfg.tau_mid_edge + spec * (cfg.tau_embed_edge - cfg.tau_mid_edge)
    if cosine >= bar: return True
    if cosine >= cfg.tau_mid_edge and src_cos >= cfg.tau_struct_edge and tgt_cos >= cfg.tau_struct_edge:
        return True
    return False

# token Jaccard over whitespace tokens (surface lexical overlap; logged, not decisive)
def lexical_sim(a: str, b: str) -> float:
    ta, tb = set((a or "").split()), set((b or "").split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def _unpack(blob) -> np.ndarray | None:
    if blob is None: return None
    return np.frombuffer(blob, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # info_vectors are unit-normed at write time

# shared-neighbor Jaccard on the (undirected) adjacency — the structural distinctness signal
def _neighbors(con: sqlite3.Connection, nid: str) -> set[str]:
    out = con.execute("SELECT target_id FROM edges WHERE source_id=?", (nid,)).fetchall()
    inc = con.execute("SELECT source_id FROM edges WHERE target_id=?", (nid,)).fetchall()
    return {r[0] for r in out + inc if r[0] != nid}

def structural_sim(con: sqlite3.Connection, a: str, b: str) -> float:
    na, nb = _neighbors(con, a) - {b}, _neighbors(con, b) - {a}
    if not na or not nb: return 0.0
    return len(na & nb) / len(na | nb)

def _degree(con: sqlite3.Connection, nid: str) -> int:
    o = con.execute("SELECT COUNT(*) FROM edges WHERE source_id=?", (nid,)).fetchone()[0]
    i = con.execute("SELECT COUNT(*) FROM edges WHERE target_id=?", (nid,)).fetchone()[0]
    return int(o + i)


# Union-find over accepted pairs so transitive duplicates collapse into one cluster.
class _UF:
    def __init__(self): self.p: dict[str, str] = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


# _knn: top-k nearest neighbours for query vector va from the pool dict {id: unit_vec}.
# Uses nodes_vec ANN when available (sqlite_vec loaded on con), falls back to brute-force.
def _knn(con: sqlite3.Connection, va: np.ndarray, pool: dict[str, np.ndarray],
         k: int, prefix: str) -> list[str]:
    try:
        rows = con.execute("SELECT id, distance FROM nodes_vec WHERE info_vector MATCH ? "
                           "AND k=? ORDER BY distance", (va.tobytes(), k + 1)).fetchall()
        ids = [r[0] for r in rows if r[0] in pool]
        if ids: return ids[:k]
    except sqlite3.OperationalError:
        pass
    # brute-force fallback: cosine over pool
    if not pool: return []
    ids_list = list(pool)
    sims = np.array([float(np.dot(va, pool[b])) for b in ids_list])
    top = np.argsort(-sims)[:k]
    return [ids_list[i] for i in top if ids_list[i].startswith(prefix)]

# _build_centroid: mean unit-vector of the top-k highest-degree nodes — the "generic centre"
# used by _specificity. Returns None if not enough vectors available.
def _build_centroid(con: sqlite3.Connection, cfg: MergeConfig) -> np.ndarray | None:
    rows = con.execute("""
        SELECT n.id, n.info_vector FROM nodes n
        WHERE n.info_vector IS NOT NULL
        ORDER BY (SELECT COUNT(*) FROM edges WHERE source_id=n.id OR target_id=n.id) DESC
        LIMIT ?""", (cfg.centroid_top_k,)).fetchall()
    vecs = [_unpack(r[1]) for r in rows if r[1] is not None]
    if not vecs: return None
    c = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    norm = np.linalg.norm(c)
    return c / norm if norm > 1e-9 else None

# Generate (a,b) candidate pairs for the given node ids via vec KNN, score them, and keep
# those should_merge accepts. Returns (accepted_pairs, scores) where scores[(a,b)] = (cos,lex,struct,deg).
# Also handles e_ (proposition) pairs using should_merge_edge with centroid-based specificity.
def _accept_pairs(con: sqlite3.Connection, node_ids: list[str], cfg: MergeConfig):
    all_vecs: dict[str, np.ndarray] = {}
    for row in con.execute("SELECT id, info_vector FROM nodes WHERE info_vector IS NOT NULL").fetchall():
        v = _unpack(row[1])
        if v is not None and v.shape == (384,): all_vecs[row[0]] = v
    candidate_set = set(node_ids)
    n_vecs = {k: v for k, v in all_vecs.items() if k.startswith("n_")}
    e_vecs = {k: v for k, v in all_vecs.items() if k.startswith("e_")}
    centroid = _build_centroid(con, cfg)
    accepted: list[tuple[str, str]] = []
    scores: dict[tuple[str, str], tuple] = {}
    seen_pairs: set[tuple[str, str]] = set()

    def _score_n_pair(a, b):
        pair = (a, b) if a < b else (b, a)
        if pair in seen_pairs or pair[0] not in candidate_set and pair[1] not in candidate_set: return
        seen_pairs.add(pair)
        ta = con.execute("SELECT text FROM nodes WHERE id=?", (pair[0],)).fetchone()
        tb = con.execute("SELECT text FROM nodes WHERE id=?", (pair[1],)).fetchone()
        cos = _cosine(n_vecs[pair[0]], n_vecs[pair[1]])
        lex = lexical_sim(ta[0] if ta else "", tb[0] if tb else "")
        struct = structural_sim(con, pair[0], pair[1])
        deg = max(_degree(con, pair[0]), _degree(con, pair[1]))
        if should_merge(cosine=cos, lexical=lex, structural=struct, degree=deg, cfg=cfg):
            accepted.append(pair); scores[pair] = (cos, lex, struct, deg)

    def _score_e_pair(a, b):
        pair = (a, b) if a < b else (b, a)
        if pair in seen_pairs or pair[0] not in candidate_set and pair[1] not in candidate_set: return
        seen_pairs.add(pair)
        va, vb = e_vecs[pair[0]], e_vecs[pair[1]]
        cos = _cosine(va, vb)
        if cos < cfg.tau_mid_edge: return   # fast reject
        # endpoint agreement: fetch src/tgt vectors for both edges
        def _ep_vecs(eid):
            row = con.execute("SELECT source_id, target_id FROM edges WHERE id=?", (eid,)).fetchone()
            if not row: return None, None
            sv = all_vecs.get(row[0]); tv = all_vecs.get(row[1])
            return sv, tv
        sa, ta_v = _ep_vecs(pair[0]); sb, tb_v = _ep_vecs(pair[1])
        src_cos = _cosine(sa, sb) if sa is not None and sb is not None else 0.0
        tgt_cos = _cosine(ta_v, tb_v) if ta_v is not None and tb_v is not None else 0.0
        # specificity: use minimum of the two edges' endpoint degrees (weakest link)
        deg_a = max(_degree(con, pair[0]), 1); deg_b = max(_degree(con, pair[1]), 1)
        spec = _specificity(min(deg_a, deg_b), vb, centroid, cfg)
        if should_merge_edge(cosine=cos, src_cos=src_cos, tgt_cos=tgt_cos, spec=spec, cfg=cfg):
            ra = con.execute("SELECT text FROM nodes WHERE id=?", (pair[0],)).fetchone()
            rb = con.execute("SELECT text FROM nodes WHERE id=?", (pair[1],)).fetchone()
            lex = lexical_sim(ra[0] if ra else "", rb[0] if rb else "")
            accepted.append(pair); scores[pair] = (cos, lex, src_cos, deg_a)

    # node pairs
    for a, va in {k: v for k, v in n_vecs.items() if k in candidate_set}.items():
        for b in _knn(con, va, n_vecs, cfg.candidates, "n_"):
            if b != a: _score_n_pair(a, b)
    # edge (e_) pairs — only if candidates include e_ ids
    e_candidates = {k: v for k, v in e_vecs.items() if k in candidate_set}
    for a, va in e_candidates.items():
        for b in _knn(con, va, e_vecs, cfg.candidates, "e_"):
            if b != a: _score_e_pair(a, b)
    return accepted, scores


# Fold victim → canonical: union synonym text, re-point edges, log, delete victim + its
# fts/vec rows. Edge ids are left as-is (opaque keys); parallel/self edges are collapsed
# afterward by _dedup_edges. cos/lex/struct/deg recorded for the audit log.
def _fold(con, victim, canonical, score, pass_num):
    cos, lex, struct, deg = score
    vr = con.execute("SELECT text, pos FROM nodes WHERE id=?", (victim,)).fetchone()
    cr = con.execute("SELECT text, pos FROM nodes WHERE id=?", (canonical,)).fetchone()
    if not vr or not cr: return
    syns = list(dict.fromkeys((cr[0] or "").split("|") + (vr[0] or "").split("|")))
    con.execute("INSERT INTO sleep_log(pass_num,victim_id,canonical_id,victim_text,canonical_text,"
                "victim_pos,canonical_pos,cosine,lexical,density,threshold,ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (pass_num, victim, canonical, vr[0], cr[0], vr[1], cr[1], cos, lex, int(deg), 0.0, int(time.time())))
    con.execute("UPDATE nodes SET text=?, updated_at=? WHERE id=?", ("|".join(syns), int(time.time()), canonical))
    con.execute("UPDATE edges SET source_id=? WHERE source_id=?", (canonical, victim))
    con.execute("UPDATE edges SET target_id=? WHERE target_id=?", (canonical, victim))
    con.execute("DELETE FROM nodes WHERE id=?", (victim,))
    con.execute("DELETE FROM nodes_fts WHERE id=?", (victim,))
    try: con.execute("DELETE FROM nodes_vec WHERE id=?", (victim,))
    except sqlite3.OperationalError: pass

# Collapse self-loops and parallel edges (same source/rel/target) created by re-pointing,
# summing counts onto the kept row and deleting the rest (+ their vec rows).
def _dedup_edges(con):
    keep: dict[tuple, str] = {}
    for eid, s, rel, t, cnt in con.execute(
            "SELECT id, source_id, rel_type, target_id, count FROM edges").fetchall():
        if s == t:
            con.execute("DELETE FROM edges WHERE id=?", (eid,))
            try: con.execute("DELETE FROM edges_vec WHERE id=?", (eid,))
            except sqlite3.OperationalError: pass
            continue
        key = (s, rel, t)
        if key in keep:
            con.execute("UPDATE edges SET count=count+? WHERE id=?", (cnt or 1, keep[key]))
            con.execute("DELETE FROM edges WHERE id=?", (eid,))
            try: con.execute("DELETE FROM edges_vec WHERE id=?", (eid,))
            except sqlite3.OperationalError: pass
        else:
            keep[key] = eid


# Core: merge the given node ids (candidate gen restricted to them, but pairs may pull in
# any near node). Used by batch (all nodes) and live (new nodes). Mutates con; caller commits.
def merge_nodes(con: sqlite3.Connection, node_ids: list[str], cfg: MergeConfig, *, pass_num: int = 1) -> dict:
    con.execute(_SLEEP_LOG)
    accepted, scores = _accept_pairs(con, [n for n in node_ids if n.startswith(("n_", "e_"))], cfg)
    if not accepted: return {"merged": 0, "clusters": 0}
    uf = _UF()
    for a, b in accepted: uf.union(a, b)
    clusters: dict[str, list[str]] = {}
    members = set(x for pair in accepted for x in pair)
    for x in members: clusters.setdefault(uf.find(x), []).append(x)
    merged = 0; n_clusters = 0
    for group in clusters.values():
        if len(group) < 2: continue
        n_clusters += 1
        canonical = max(group, key=lambda n: (_degree(con, n), n))   # most-connected wins; tie → max id
        for victim in group:
            if victim == canonical: continue
            pair = (victim, canonical) if victim < canonical else (canonical, victim)
            score = scores.get(pair, (_pair_cosine(con, victim, canonical), 0.0, 0.0, _degree(con, canonical)))
            _fold(con, victim, canonical, score, pass_num); merged += 1
    _dedup_edges(con)
    return {"merged": merged, "clusters": n_clusters}

def _pair_cosine(con, a, b) -> float:
    ra = con.execute("SELECT info_vector FROM nodes WHERE id=?", (a,)).fetchone()
    rb = con.execute("SELECT info_vector FROM nodes WHERE id=?", (b,)).fetchone()
    va, vb = _unpack(ra[0]) if ra else None, _unpack(rb[0]) if rb else None
    return _cosine(va, vb) if va is not None and vb is not None else 0.0


# Batch entry point: merge near-duplicate nodes across the whole store, in place.
def merge_store(path: str | Path, cfg: MergeConfig | None = None) -> dict:
    cfg = cfg or MergeConfig()
    con = sqlite3.connect(str(path))
    if sqlite_vec is not None:
        try: con.enable_load_extension(True); sqlite_vec.load(con)
        except Exception: pass
    try:
        ids = [r[0] for r in con.execute("SELECT id FROM nodes WHERE id LIKE 'n_%'").fetchall()]
        stats = merge_nodes(con, ids, cfg)
        con.commit()
        return stats
    finally:
        con.close()
