#!/usr/bin/env python3
"""Scalable node+edge merge for the corpus graph.

The repo's merge_store rules are kept verbatim (should_merge: degree/centroid
specificity, BM25-boosted cosine, homonym guard, discriminative-token + numeric
guards, fold/dedup/rescore). Only the candidate generation is replaced: the repo's
brute-force O(N^2) _knn_batch -> an hnswlib HNSW index (ANN), so it scales to the
~1.7M nodes here. Edge pass uses the repo's bucket-by-endpoint approach, streamed.

Run with PYTHONPATH=<ingest-hyperedge worktree>/src so graph.merge imports.
"""
import argparse
import json
import os
import sqlite3
import time

import numpy as np
import hnswlib

from graph.merge import (
    MergeConfig, should_merge, bm25_sim, _build_idf, _numeric_tokens,
    _discriminative_conflict, _specificity, _unpack, _cosine, _apply_merges,
    _dedup_edges, _recompute_scores, _SLEEP_LOG, _rel_type)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--status", default=None)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--ef-construction", type=int, default=128)
    ap.add_argument("--no-edge-merge", action="store_true")
    ap.add_argument("--no-vec", action="store_true", help="skip building sqlite-vec ANN indexes")
    args = ap.parse_args()
    cfg = MergeConfig(candidates=args.candidates)
    st = args.status or args.store + ".merge_status.json"

    def status(**kw):
        kw["updated"] = time.time(); kw["updated_h"] = time.strftime("%H:%M:%S")
        tmp = st + ".tmp"
        with open(tmp, "w") as f:
            f.write(json.dumps(kw, indent=2))
        os.replace(tmp, st)

    con = sqlite3.connect(args.store)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(_SLEEP_LOG); con.commit()
    t0 = time.time()

    # ── load node vectors + text ──
    status(phase="load")
    log("loading node vectors + text")
    ids, vlist, node_text = [], [], {}
    for nid, txt, blob in con.execute(
            "SELECT id,text,info_vector FROM nodes WHERE id LIKE 'n_%' AND info_vector IS NOT NULL"):
        v = _unpack(blob)
        if v is not None and v.shape == (384,):
            ids.append(nid); vlist.append(v); node_text[nid] = txt or nid
    N = len(ids)
    mat = np.vstack(vlist).astype(np.float32); del vlist
    vec_of = {nid: mat[i] for i, nid in enumerate(ids)}
    log(f"{N} node vectors loaded ({time.time()-t0:.0f}s)")

    # ── degree map (in-memory; the repo's per-node SQL would be O(N*E)) ──
    status(phase="degree")
    deg = {}
    for s, t in con.execute("SELECT source_id,target_id FROM edges"):
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1
    log(f"degree map built ({time.time()-t0:.0f}s)")

    # ── idf + centroid (centroid from top-degree nodes, in-memory) ──
    idf, avgdl = _build_idf(con)
    top = sorted((n for n in deg if n in vec_of), key=lambda n: deg[n], reverse=True)[:cfg.centroid_top_k]
    cvecs = [vec_of[n] for n in top]
    centroid = None
    if cvecs:
        c = np.mean(np.stack(cvecs), axis=0).astype(np.float32)
        nrm = np.linalg.norm(c)
        centroid = c / nrm if nrm > 1e-9 else None
    log(f"idf + centroid ready ({time.time()-t0:.0f}s)")

    # ── hnswlib ANN index ──
    status(phase="ann_build")
    index = hnswlib.Index(space="cosine", dim=384)
    index.init_index(max_elements=N, ef_construction=args.ef_construction, M=16)
    index.add_items(mat, np.arange(N))
    index.set_ef(max(args.ef, args.candidates + 1))
    log(f"hnswlib index built ({time.time()-t0:.0f}s)")

    # ── candidate gen + scoring (same decision as _accept_pairs_nodes) ──
    status(phase="candidates", total=N)
    accepted, scores, seen = [], {}, set()
    k = args.candidates + 1
    floor = cfg.tau_bm25_boost - 0.06
    B = 20000
    for i in range(0, N, B):
        sub = mat[i:i + B]
        labels, dists = index.knn_query(sub, k=k)
        for r in range(sub.shape[0]):
            a = ids[i + r]
            ta = node_text[a]
            for lab, dist in zip(labels[r], dists[r]):
                b = ids[int(lab)]
                if b == a:
                    continue
                cos = 1.0 - float(dist)
                if cos < floor:
                    continue
                pair = (a, b) if a < b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                tb = node_text[b]
                bm = bm25_sim(ta, tb, idf, avgdl)
                na, nb = _numeric_tokens(ta), _numeric_tokens(tb)
                conflict = (bool(na) and bool(nb) and na != nb) or _discriminative_conflict(ta, tb)
                degmax = max(deg.get(a, 0), deg.get(b, 0))
                spec = _specificity(degmax, vec_of.get(b), centroid, cfg)
                if should_merge(cosine=cos, bm25=bm, spec=spec, cfg=cfg, conflict=conflict):
                    accepted.append(pair); scores[pair] = (cos, bm, 0.0, degmax)
        if (i // B) % 5 == 0:
            log(f"scored {min(i+B,N)}/{N}, accepted {len(accepted)}")
            status(phase="candidates", scored=min(i + B, N), total=N, accepted=len(accepted))
    log(f"accepted {len(accepted)} node pairs ({time.time()-t0:.0f}s)")
    del index, mat, vec_of

    # ── fold node clusters (repo logic: canonical=highest degree, re-point edges) ──
    status(phase="fold_nodes", accepted=len(accepted))
    stats_n = _apply_merges(con, accepted, scores, pass_num=0)
    con.commit()
    log(f"node merge: {stats_n} ({time.time()-t0:.0f}s)")

    # ── edge pass: bucket by (canonical) endpoint pair, compare vectors streamed ──
    stats_e = {"merged": 0, "clusters": 0}
    if not args.no_edge_merge:
        status(phase="edge_buckets")
        groups = {}
        for eid, s, t in con.execute("SELECT id,source_id,target_id FROM edges WHERE id LIKE 'e_%'"):
            key = (s, t) if s <= t else (t, s)
            groups.setdefault(key, []).append(eid)
        multi = [(key, g) for key, g in groups.items() if len(g) >= 2]
        log(f"{len(multi)} multi-edge endpoint buckets ({time.time()-t0:.0f}s)")
        accepted_e, scores_e = [], {}
        deg2 = {}  # recompute degree post node-merge for endpoint specificity
        for s, t in con.execute("SELECT source_id,target_id FROM edges"):
            deg2[s] = deg2.get(s, 0) + 1; deg2[t] = deg2.get(t, 0) + 1
        for key, group in multi:
            src, tgt = key
            dmin = min(deg2.get(src, 0), deg2.get(tgt, 0))
            spec = _specificity(dmin, None, None, cfg)
            ph = ",".join("?" * len(group))
            vmap = {e: _unpack(b) for e, b in con.execute(
                f"SELECT id,info_vector FROM edges WHERE id IN ({ph})", group)}
            for ii in range(len(group)):
                for jj in range(ii + 1, len(group)):
                    a, b = group[ii], group[jj]
                    va, vb = vmap.get(a), vmap.get(b)
                    if va is None or vb is None:
                        continue
                    cos = _cosine(va, vb)
                    if cos < floor:
                        continue
                    bm = bm25_sim(_rel_type(con, a), _rel_type(con, b), idf, avgdl)
                    if should_merge(cosine=cos, bm25=bm, spec=spec, cfg=cfg):
                        p = (a, b) if a < b else (b, a)
                        accepted_e.append(p); scores_e[p] = (cos, bm, 0.0, dmin)
        log(f"accepted {len(accepted_e)} edge pairs ({time.time()-t0:.0f}s)")
        status(phase="fold_edges", accepted=len(accepted_e))
        stats_e = _apply_merges(con, accepted_e, scores_e, pass_num=1)
        con.commit()
        log(f"edge merge: {stats_e} ({time.time()-t0:.0f}s)")

    # ── final dedup + rescore ──
    status(phase="dedup")
    _dedup_edges(con); _recompute_scores(con); con.commit()
    log(f"dedup + rescore done ({time.time()-t0:.0f}s)")

    # ── vectorize everything: (re)build sqlite-vec ANN indexes on merged graph ──
    if not args.no_vec:
        status(phase="build_vec")
        try:
            import sqlite_vec
            con.enable_load_extension(True); sqlite_vec.load(con)
            for tbl, vtbl in (("nodes", "nodes_vec"), ("edges", "edges_vec")):
                con.execute(f"DROP TABLE IF EXISTS {vtbl}")
                con.execute(f"CREATE VIRTUAL TABLE {vtbl} USING vec0(id TEXT PRIMARY KEY, info_vector float[384])")
                con.commit()
                rcon = sqlite3.connect(args.store)
                cur = rcon.execute(f"SELECT id,info_vector FROM {tbl} WHERE info_vector IS NOT NULL")
                done = 0
                while True:
                    rows = cur.fetchmany(8192)
                    if not rows:
                        break
                    con.executemany(f"INSERT INTO {vtbl}(id,info_vector) VALUES(?,?)", rows)
                    con.commit(); done += len(rows)
                    status(phase="build_vec", table=vtbl, done=done)
                rcon.close()
                log(f"{vtbl}: {done} vectors indexed ({time.time()-t0:.0f}s)")
        except Exception as e:                            # noqa: BLE001
            log(f"vec index build skipped: {e!r}")

    n = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    e = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    h = con.execute("SELECT COUNT(*) FROM edges WHERE source_id LIKE 'e_%' OR target_id LIKE 'e_%'").fetchone()[0]
    folds = con.execute("SELECT COUNT(*) FROM sleep_log").fetchone()[0]
    con.close()
    status(phase="DONE", nodes=n, edges=e, hyperedges=h,
           merged_nodes=stats_n.get("merged", 0), merged_edges=stats_e.get("merged", 0),
           folds=folds, elapsed=round(time.time() - t0))
    log(f"DONE nodes={n} edges={e} hyper={h} merged_nodes={stats_n.get('merged',0)} "
        f"merged_edges={stats_e.get('merged',0)} folds={folds} in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
