#!/usr/bin/env python3
"""Fast bulk info-vector pass on Apple MPS (GPU).

Backfills info_vector for every node and reified edge in the corpus graph store
using sentence-transformers all-MiniLM-L6-v2 on the M-series GPU (~3000/s vs
fastembed's ~40/s on CPU). 384-d, unit-normalized — same shape/normalization the
repo's merge expects. Skips the sqlite-vec vec0 index (rebuildable; only the field
consumer needs it, and the deferred merge reads the info_vector column directly).

Edge surface = reconstructed "src_text rel tgt_text", recursing through reified
(e_) endpoints. Writes a heartbeat status file for the overnight monitor.
"""
import argparse
import json
import os
import sqlite3
import time

import numpy as np


def write_status(path, **kw):
    kw["updated"] = time.time()
    kw["updated_h"] = time.strftime("%H:%M:%S")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(kw, indent=2))
    os.replace(tmp, path)


def edge_surfaces(con):
    ntext = {r[0]: (r[1] or "") for r in con.execute("SELECT id,text FROM nodes")}
    emap = {r[0]: (r[1], r[2], r[3])
            for r in con.execute("SELECT id,source_id,rel_type,target_id FROM edges")}
    cache = {}

    def surf(eid, depth=0):
        if eid in cache:
            return cache[eid]
        if depth > 6 or eid not in emap:
            return ""
        s, rel, t = emap[eid]
        ss = ntext.get(s) if s.startswith("n_") else surf(s, depth + 1)
        ts = ntext.get(t) if t.startswith("n_") else surf(t, depth + 1)
        out = f"{ss or s} {rel} {ts or t}".strip()
        cache[eid] = out
        return out

    return {eid: surf(eid) for eid in emap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--status", default=None)
    ap.add_argument("--enc-batch", type=int, default=512)   # MPS encode batch
    ap.add_argument("--write-batch", type=int, default=16384)  # sqlite commit chunk
    ap.add_argument("--reset", action="store_true", help="NULL all vectors first (consistency)")
    ap.add_argument("--nodes-only", action="store_true")
    args = ap.parse_args()
    stpath = args.status or (args.store + ".mps_status.json")

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    print(f"device={dev}", flush=True)

    def enc(texts):
        v = model.encode(texts, batch_size=args.enc_batch,
                         normalize_embeddings=True, show_progress_bar=False)
        return [np.asarray(x, dtype=np.float32).tobytes() for x in v]

    con = sqlite3.connect(args.store)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    if args.reset:
        con.execute("UPDATE nodes SET info_vector=NULL")
        con.execute("UPDATE edges SET info_vector=NULL")
        con.commit()

    B = args.write_batch
    # ── nodes ──
    rows = con.execute("SELECT id,text FROM nodes WHERE info_vector IS NULL").fetchall()
    tot = len(rows)
    print(f"nodes to embed: {tot}", flush=True)
    t0 = time.time()
    done = 0
    for i in range(0, tot, B):
        chunk = rows[i:i + B]
        vecs = enc([r[1] for r in chunk])
        con.executemany("UPDATE nodes SET info_vector=? WHERE id=?",
                        [(v, r[0]) for r, v in zip(chunk, vecs)])
        con.executemany("INSERT OR REPLACE INTO nodes_fts(id,text) VALUES(?,?)",
                        [(r[0], r[1]) for r in chunk])
        con.commit()
        done += len(chunk)
        rate = done / max(time.time() - t0, 1e-9)
        write_status(stpath, phase="nodes", done=done, total=tot, rate=round(rate),
                     eta_min=round((tot - done) / max(rate, 1) / 60, 1))
        print(f"nodes {done}/{tot} {rate:.0f}/s", flush=True)

    # ── edges (reified hyperedges) ──
    if not args.nodes_only:
        print("reconstructing edge surfaces ...", flush=True)
        surf = edge_surfaces(con)
        eids = [r[0] for r in con.execute("SELECT id FROM edges WHERE info_vector IS NULL")]
        tot = len(eids)
        print(f"edges to embed: {tot}", flush=True)
        t0 = time.time()
        done = 0
        for i in range(0, tot, B):
            chunk = eids[i:i + B]
            vecs = enc([surf.get(e, "") or e for e in chunk])
            con.executemany("UPDATE edges SET info_vector=? WHERE id=?",
                            [(v, e) for e, v in zip(chunk, vecs)])
            con.commit()
            done += len(chunk)
            rate = done / max(time.time() - t0, 1e-9)
            write_status(stpath, phase="edges", done=done, total=tot, rate=round(rate),
                         eta_min=round((tot - done) / max(rate, 1) / 60, 1))
            print(f"edges {done}/{tot} {rate:.0f}/s", flush=True)

    n = con.execute("SELECT COUNT(*) FROM nodes WHERE info_vector IS NOT NULL").fetchone()[0]
    e = con.execute("SELECT COUNT(*) FROM edges WHERE info_vector IS NOT NULL").fetchone()[0]
    con.close()
    write_status(stpath, phase="DONE", nodes_embedded=n, edges_embedded=e)
    print(f"MPS EMBED DONE nodes={n} edges={e}", flush=True)


if __name__ == "__main__":
    main()
