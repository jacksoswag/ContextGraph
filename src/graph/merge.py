from __future__ import annotations
# Source-agnostic node/edge/hyperedge merge.
# Two-phase: pass 0 collapses n_ nodes, pass 1 collapses e_ edges that share canonical endpoints.
# Endpoints live in different tables — n_ vectors/text on `nodes`, e_ vectors on `edges` (text is
# the rendered triple). Both passes share one decision: BM25-boosted cosine with centroid-distance
# specificity (bar 0.86 generic → 0.98 specific) + a homonym guard (near-identical surface but low
# cosine → block). Numeric normalisation lets "42.3 million" / "42,300,000" match.
# After both passes: score = log1p(count) / log1p(max_count) over all edges (source-agnostic count).
import math, re, sqlite3, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try: import sqlite_vec
except Exception: sqlite_vec = None

@dataclass
class MergeConfig:
    candidates: int = 8           # KNN candidates examined per node
    tau_lo: float = 0.86          # cosine threshold for generic endpoints (near centroid / high degree)
    tau_hi: float = 0.98          # cosine threshold for specific endpoints (far / low degree)
    tau_bm25_boost: float = 0.82  # cosine floor for the BM25-confirmed (boost) merge path
    bm25_threshold: float = 0.55  # min BM25 to activate the boost path
    bm25_homonym_guard: float = 0.92  # BM25 ≥ this in the boost zone signals a near-identical surface
    tau_homonym_guard: float = 0.88   # …block it as a homonym unless cosine ≥ this
    w_deg: float = 0.4            # weight for the inverted-degree term in node specificity
    w_dist: float = 0.6           # weight for the centroid-distance term in node specificity
    centroid_scale: float = 0.8   # tanh scale normalising centroid distance to [0,1]
    centroid_top_k: int = 30      # highest-degree nodes that define the generic centroid

_SLEEP_LOG = (
    "CREATE TABLE IF NOT EXISTS sleep_log(id INTEGER PRIMARY KEY AUTOINCREMENT, pass_num INTEGER,"
    " victim_id TEXT, canonical_id TEXT, victim_text TEXT, canonical_text TEXT, victim_pos TEXT,"
    " canonical_pos TEXT, cosine REAL, lexical REAL, density INTEGER, threshold REAL, ts INTEGER)")

# ── numeric normalisation ─────────────────────────────────────────────────────
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
          "k": 1e3, "m": 1e6, "b": 1e9}

def _norm_text(s: str) -> str:
    s = s.lower().strip()
    while re.search(r'\d,\d{3}', s):                                 # 1,000,000 → 1000000
        s = re.sub(r'(\d),(\d{3})', r'\1\2', s)
    def _expand(m):
        try: return str(int(float(m.group(1)) * _SCALE.get(m.group(2), 1.0)))
        except: return m.group(0)
    s = re.sub(r'(\d+\.?\d*)\s*(thousand|million|billion|trillion|[kmb])\b', _expand, s)
    def _sci(m):
        try: return str(int(float(m.group(0))))
        except: return m.group(0)
    return re.sub(r'\d+\.?\d*[eE][+-]?\d+', _sci, s)

def _tokenize(s: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', _norm_text(s))

# normalized numeric tokens — "42.3 million" and "42,300,000" both → {"42300000"}; "23 may"/"8 may"
# → {"23"}/{"8"}. Lets same-value figures match (equal sets) while different figures stay distinct.
def _numeric_tokens(s: str) -> frozenset:
    return frozenset(re.findall(r'\d+', _norm_text(s)))

# function words whose presence/absence never distinguishes two entities (vs content words)
_STOP = frozenset(
    "a an the of in on at to for from by with as is are was were be been being and or but "
    "s this that these those it he she they his her their its".split())

# morphological variants ("landing"/"landings", "color"/"colour") count as the same slot
def _same_stem(a: str, b: str) -> bool:
    if a.startswith(b) or b.startswith(a): return True
    lcp = 0
    for x, y in zip(a, b):
        if x != y: break
        lcp += 1
    return lcp >= 3 and lcp >= max(len(a), len(b)) - 2

# Distinct-entity signal for NODES: two surfaces that share a template but differ in exactly one
# content-word slot ("north korea"/"south korea", "22 june"/"22 july", "united states"/"…kingdom")
# are distinct even at high cosine. Not applied to edge predicates, where one-token differences are
# usually synonyms ("galaxy begin"/"galaxy start") we DO want merged.
def _discriminative_conflict(a: str, b: str) -> bool:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    only_a, only_b = sa - sb, sb - sa
    if len(only_a) != 1 or len(only_b) != 1 or not (sa & sb): return False
    wa, wb = next(iter(only_a)), next(iter(only_b))
    if wa in _STOP and wb in _STOP: return False     # function-word slot ("of"/"s") ⇒ same entity
    if _same_stem(wa, wb): return False              # morphological variant ⇒ same entity
    return True                                      # content-word slot differs ⇒ distinct

# ── BM25 (BM25+ variant: idf floored positive so common tokens still contribute) ───────────────
def _build_idf(con: sqlite3.Connection) -> tuple[dict[str, float], float]:
    docs = [r[0] for r in con.execute("SELECT text FROM nodes WHERE text IS NOT NULL").fetchall()]
    try:
        docs += [r[0] for r in con.execute("SELECT rel_type FROM edges WHERE rel_type IS NOT NULL").fetchall()]
    except sqlite3.OperationalError:
        pass
    N = max(len(docs), 1); total_len = 0
    df: dict[str, int] = {}
    for t in docs:
        toks = _tokenize(t); total_len += len(toks)
        for tok in set(toks): df[tok] = df.get(tok, 0) + 1
    idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
    return idf, total_len / N

def bm25_sim(a: str, b: str, idf: dict[str, float], avgdl: float,
             k1: float = 1.5, b_par: float = 0.75) -> float:
    toks_a = _tokenize(a); toks_b = _tokenize(b)
    if not toks_a or not toks_b: return 0.0
    dl = len(toks_b)
    tf_b: dict[str, int] = {}
    for t in toks_b: tf_b[t] = tf_b.get(t, 0) + 1
    norm = k1 * (1 - b_par + b_par * dl / max(avgdl, 1.0))
    score = sum(idf.get(t, 0.0) * tf_b[t] * (k1 + 1) / (tf_b[t] + norm)
                for t in set(toks_a) if t in tf_b)
    max_s = sum(idf.get(t, 0.0) * (k1 + 1) / (1.0 + norm) for t in set(toks_a))
    return min(score / max_s, 1.0) if max_s > 1e-9 else 0.0

# token Jaccard — kept as a utility for callers/tests, not in the decision path
def lexical_sim(a: str, b: str) -> float:
    ta, tb = set((a or "").split()), set((b or "").split())
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

# ── vectors + graph helpers ───────────────────────────────────────────────────
def _unpack(blob) -> np.ndarray | None:
    if blob is None: return None
    return np.frombuffer(blob, dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # info_vectors are unit-normed at write time

# all endpoint vectors: n_ from nodes, e_ from edges (reified edges have no nodes row)
def _load_vectors(con: sqlite3.Connection) -> dict[str, np.ndarray]:
    vecs: dict[str, np.ndarray] = {}
    for tbl in ("nodes", "edges"):
        for nid, blob in con.execute(f"SELECT id, info_vector FROM {tbl} WHERE info_vector IS NOT NULL").fetchall():
            v = _unpack(blob)
            if v is not None and v.shape == (384,): vecs[nid] = v
    return vecs

def _degree(con: sqlite3.Connection, nid: str) -> int:
    o = con.execute("SELECT COUNT(*) FROM edges WHERE source_id=?", (nid,)).fetchone()[0]
    i = con.execute("SELECT COUNT(*) FROM edges WHERE target_id=?", (nid,)).fetchone()[0]
    return int(o + i)

def _node_text(con, nid: str) -> str:
    r = con.execute("SELECT text FROM nodes WHERE id=?", (nid,)).fetchone()
    return r[0] if r and r[0] else nid

def _rel_type(con, eid: str) -> str:
    r = con.execute("SELECT rel_type FROM edges WHERE id=?", (eid,)).fetchone()
    return r[0] if r and r[0] else ""

# shallow render of a reified edge for the audit log (e_ endpoints shown as ids, not recursed)
def _render_triple(con, eid: str) -> str:
    r = con.execute("SELECT source_id, rel_type, target_id FROM edges WHERE id=?", (eid,)).fetchone()
    if not r: return eid
    s = _node_text(con, r[0]) if not r[0].startswith("e_") else r[0]
    t = _node_text(con, r[2]) if r[2] and not r[2].startswith("e_") else (r[2] or "")
    return f"{s} {r[1]} {t}".strip()

# ── specificity + merge decision ─────────────────────────────────────────────
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

# specificity ∈ [0,1]: 1 = maximally specific (high bar), 0 = generic (low bar).
# Combines inverted degree + distance from the generic centroid (centroid optional).
def _specificity(deg: int, emb: np.ndarray | None, centroid: np.ndarray | None, cfg: MergeConfig) -> float:
    deg_term = 1.0 / (1.0 + math.log1p(max(0, deg)))
    if emb is not None and centroid is not None:
        dist = float(np.linalg.norm(emb - centroid))
        dist_term = math.tanh(dist / max(cfg.centroid_scale, 1e-6))
    else:
        dist_term = deg_term
    return cfg.w_deg * deg_term + cfg.w_dist * dist_term

# Merge decision (pure). Bar floats tau_lo (generic) → tau_hi (specific). High cosine is confident
# on its own. Moderate cosine merges via the BM25 boost — but a near-identical surface (high BM25)
# with low semantic agreement (cosine < tau_homonym_guard) is a homonym and is refused.
def should_merge(*, cosine: float, bm25: float, spec: float, cfg: MergeConfig,
                 conflict: bool = False) -> bool:
    if conflict: return False   # external distinctness signal (figures/content-slot) ⇒ never merge
    tau = cfg.tau_lo + spec * (cfg.tau_hi - cfg.tau_lo)
    if cosine >= tau: return True
    if cosine >= cfg.tau_bm25_boost and bm25 >= cfg.bm25_threshold:
        if bm25 >= cfg.bm25_homonym_guard and cosine < cfg.tau_homonym_guard:
            return False
        return True
    return False

# ── candidate generation ───────────────────────────────────────────────────────
# Batched exact KNN over unit vectors (cosine = dot): for every query id, its top-k pool neighbours.
# A single BLAS matmul per query-chunk replaces the per-node Python loop — independent of any sqlite
# extension, so it works on Pythons built without loadable-extension support (this one).
def _knn_batch(query_ids: list[str], pool: dict[str, np.ndarray], k: int,
               chunk: int = 1024) -> dict[str, list[str]]:
    qids = [q for q in query_ids if q in pool]
    if not qids or not pool: return {}
    pool_ids = list(pool)
    M = np.stack([pool[i] for i in pool_ids])                        # [Np, d]
    out: dict[str, list[str]] = {}
    kk = min(k + 1, len(pool_ids))                                   # +1 to absorb the self-match
    for i in range(0, len(qids), chunk):
        ch = qids[i:i + chunk]
        S = np.stack([pool[q] for q in ch]) @ M.T                    # [b, Np] cosine
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        for r, q in enumerate(ch):
            cols = part[r][np.argsort(-S[r, part[r]])]
            out[q] = [pool_ids[c] for c in cols if pool_ids[c] != q][:k]
    return out

def _accept_pairs_nodes(con, n_ids: list[str], all_vecs: dict,
                        idf: dict, avgdl: float, centroid, cfg: MergeConfig):
    n_vecs = {k: v for k, v in all_vecs.items() if k.startswith("n_")}
    cand_set = set(n_ids); accepted: list[tuple] = []; scores: dict = {}; seen: set = set()

    def _score(a, b):
        pair = (a, b) if a < b else (b, a)
        if pair in seen or (pair[0] not in cand_set and pair[1] not in cand_set): return
        seen.add(pair)
        va, vb = n_vecs.get(pair[0]), n_vecs.get(pair[1])
        if va is None or vb is None: return
        cos = _cosine(va, vb)
        if cos < cfg.tau_bm25_boost - 0.06: return                   # fast reject
        ta, tb = _node_text(con, pair[0]), _node_text(con, pair[1])
        bm = bm25_sim(ta, tb, idf, avgdl)
        na, nb = _numeric_tokens(ta), _numeric_tokens(tb)
        conflict = (bool(na) and bool(nb) and na != nb) or _discriminative_conflict(ta, tb)
        deg = max(_degree(con, pair[0]), _degree(con, pair[1]))
        spec = _specificity(deg, vb, centroid, cfg)
        if should_merge(cosine=cos, bm25=bm, spec=spec, cfg=cfg, conflict=conflict):
            accepted.append(pair); scores[pair] = (cos, bm, 0.0, deg)

    nbrs = _knn_batch([k for k in n_ids if k in n_vecs], n_vecs, cfg.candidates)
    for a, cands in nbrs.items():
        for b in cands:
            if b != a: _score(a, b)
    return accepted, scores

# Reified-edge merge: only edges sharing the SAME (undirected) endpoints are comparable, so they
# are bucketed by endpoint pair and compared pairwise within a bucket. Specificity is the genericity
# of the (shared) endpoints — generic endpoints ⇒ low bar; rare endpoints ⇒ near-identical required.
def _accept_pairs_edges(con, e_ids: list[str], all_vecs: dict,
                        idf: dict, avgdl: float, centroid, cfg: MergeConfig):
    e_vecs = {k: v for k, v in all_vecs.items() if k.startswith("e_")}
    groups: dict[tuple[str, str], list[str]] = {}
    for eid in e_ids:
        row = con.execute("SELECT source_id, target_id FROM edges WHERE id=?", (eid,)).fetchone()
        if not row: continue
        src, tgt = row
        key = (src, tgt) if src <= tgt else (tgt, src)
        groups.setdefault(key, []).append(eid)

    accepted: list[tuple] = []; scores: dict = {}
    for (src, tgt), group in groups.items():
        if len(group) < 2: continue
        deg = min(_degree(con, src), _degree(con, tgt))              # endpoint genericity (weakest link)
        spec = _specificity(deg, None, None, cfg)                    # degree-only: endpoints fix the meaning
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                va, vb = e_vecs.get(a), e_vecs.get(b)
                if va is None or vb is None: continue
                cos = _cosine(va, vb)
                if cos < cfg.tau_bm25_boost - 0.06: continue
                bm = bm25_sim(_rel_type(con, a), _rel_type(con, b), idf, avgdl)
                if should_merge(cosine=cos, bm25=bm, spec=spec, cfg=cfg):
                    pair = (a, b) if a < b else (b, a)
                    accepted.append(pair); scores[pair] = (cos, bm, 0.0, deg)
    return accepted, scores

# ── fold / dedup / score ──────────────────────────────────────────────────────
class _UF:
    def __init__(self): self.p: dict[str, str] = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

def _log(con, victim, canonical, vtxt, ctxt, vpos, cpos, score, pass_num):
    cos, lex, _s, deg = score
    con.execute("INSERT INTO sleep_log(pass_num,victim_id,canonical_id,victim_text,canonical_text,"
                "victim_pos,canonical_pos,cosine,lexical,density,threshold,ts) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (pass_num, victim, canonical, vtxt, ctxt, vpos, cpos, cos, lex, int(deg), 0.0, int(time.time())))

# Fold victim → canonical. n_ : union synonym text, re-point edges, delete node row. e_ : sum count
# into canonical, re-point any qualifiers/hyperedges that referenced the victim edge, delete edge row.
def _fold(con, victim, canonical, score, pass_num):
    now = int(time.time())
    if victim.startswith("e_"):
        cnt = con.execute("SELECT COALESCE(count,1) FROM edges WHERE id=?", (victim,)).fetchone()
        if cnt is None: return
        _log(con, victim, canonical, _render_triple(con, victim), _render_triple(con, canonical),
             None, None, score, pass_num)
        con.execute("UPDATE edges SET source_id=? WHERE source_id=?", (canonical, victim))
        con.execute("UPDATE edges SET target_id=? WHERE target_id=?", (canonical, victim))
        con.execute("UPDATE edges SET count=count+?, updated_at=? WHERE id=?", (cnt[0], now, canonical))
        con.execute("DELETE FROM edges WHERE id=?", (victim,))
        try: con.execute("DELETE FROM edges_vec WHERE id=?", (victim,))
        except sqlite3.OperationalError: pass
        return
    vr = con.execute("SELECT text, pos FROM nodes WHERE id=?", (victim,)).fetchone()
    cr = con.execute("SELECT text, pos FROM nodes WHERE id=?", (canonical,)).fetchone()
    if not vr or not cr: return
    syns = list(dict.fromkeys((cr[0] or "").split("|") + (vr[0] or "").split("|")))
    _log(con, victim, canonical, vr[0], cr[0], vr[1], cr[1], score, pass_num)
    con.execute("UPDATE nodes SET text=?, updated_at=? WHERE id=?", ("|".join(syns), now, canonical))
    con.execute("UPDATE edges SET source_id=? WHERE source_id=?", (canonical, victim))
    con.execute("UPDATE edges SET target_id=? WHERE target_id=?", (canonical, victim))
    con.execute("DELETE FROM nodes WHERE id=?", (victim,))
    con.execute("DELETE FROM nodes_fts WHERE id=?", (victim,))
    try: con.execute("DELETE FROM nodes_vec WHERE id=?", (victim,))
    except sqlite3.OperationalError: pass

# Collapse self-loops and parallel edges (same source/rel/target) created by re-pointing.
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

# Source-agnostic confidence: score = log1p(count) / log1p(max_count). Higher counts approach 1.
def _recompute_scores(con: sqlite3.Connection):
    rows = con.execute("SELECT id, COALESCE(count,1) FROM edges").fetchall()
    if not rows: return
    denom = math.log1p(max(r[1] for r in rows))
    if denom < 1e-9: return
    now = int(time.time())
    con.executemany("UPDATE edges SET score=?, updated_at=? WHERE id=?",
                    [(math.log1p(cnt) / denom, now, eid) for eid, cnt in rows])

def _pair_cosine(con, a, b) -> float:
    va = _load_one(con, a); vb = _load_one(con, b)
    return _cosine(va, vb) if va is not None and vb is not None else 0.0

def _load_one(con, nid):
    tbl = "edges" if nid.startswith("e_") else "nodes"
    r = con.execute(f"SELECT info_vector FROM {tbl} WHERE id=?", (nid,)).fetchone()
    return _unpack(r[0]) if r else None

def _apply_merges(con, accepted, scores, pass_num) -> dict:
    if not accepted: return {"merged": 0, "clusters": 0}
    uf = _UF()
    for a, b in accepted: uf.union(a, b)
    clusters: dict[str, list[str]] = {}
    for x in set(n for pair in accepted for n in pair):
        clusters.setdefault(uf.find(x), []).append(x)
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
    return {"merged": merged, "clusters": n_clusters}

# ── public API ────────────────────────────────────────────────────────────────
# Pass 0: n_ node merges. Pass 1: e_ edge merges (endpoints already canonicalized by pass 0).
# Caller commits. node_ids may mix n_ and e_; live writes pass only n_ (edge merge is batch-only).
def merge_nodes(con: sqlite3.Connection, node_ids: list[str], cfg: MergeConfig, *, pass_num: int = 1) -> dict:
    con.execute(_SLEEP_LOG)
    all_vecs = _load_vectors(con)
    idf, avgdl = _build_idf(con)
    centroid = _build_centroid(con, cfg)

    accepted_n, scores_n = _accept_pairs_nodes(
        con, [n for n in node_ids if n.startswith("n_")], all_vecs, idf, avgdl, centroid, cfg)
    stats_n = _apply_merges(con, accepted_n, scores_n, pass_num)

    accepted_e, scores_e = _accept_pairs_edges(
        con, [n for n in node_ids if n.startswith("e_")], all_vecs, idf, avgdl, centroid, cfg)
    stats_e = _apply_merges(con, accepted_e, scores_e, pass_num)

    _dedup_edges(con)
    _recompute_scores(con)
    return {"merged": stats_n["merged"] + stats_e["merged"],
            "clusters": stats_n["clusters"] + stats_e["clusters"],
            "merged_nodes": stats_n["merged"], "merged_edges": stats_e["merged"]}

# Batch entry point: merge near-duplicate nodes (from `nodes`) and edges (from `edges`) in place.
def merge_store(path: str | Path, cfg: MergeConfig | None = None) -> dict:
    cfg = cfg or MergeConfig()
    con = sqlite3.connect(str(path))
    if sqlite_vec is not None:
        try: con.enable_load_extension(True); sqlite_vec.load(con)
        except Exception: pass
    try:
        ids = [r[0] for r in con.execute("SELECT id FROM nodes WHERE id LIKE 'n_%'").fetchall()]
        ids += [r[0] for r in con.execute("SELECT id FROM edges WHERE id LIKE 'e_%'").fetchall()]
        stats = merge_nodes(con, ids, cfg)
        con.commit()
        return stats
    finally:
        con.close()
