#!/usr/bin/env python3
"""Deferred-merge corpus ingest: extract hyperedge triples + info-vector them.

Three modes (orchestrated by run_ingest.sh):

  extract  stream a disjoint slice of the gzipped JSONL shards through the
           deterministic pure-spaCy hyperedge pipeline and write the reified
           nodes/edges (NO embedding — info_vector left NULL). Run one process
           per worker with OMP_NUM_THREADS=1 and --shard i --of W.

  combine  union the per-worker stores into one master by deterministic id
           (counts summed). Structural only; vectors still NULL.

  embed    bulk-embed every NULL-vector node (and reified-edge surface) in large
           fastembed batches using ALL cores, then populate nodes_fts/nodes_vec/
           edges_vec. This is the "info-vector them" stage.

Merge is intentionally NOT run here (deferred — the repo merge is O(N^2) and
needs a scalable candidate-gen pass built separately).
"""
import argparse
import glob
import gzip
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ── extract ────────────────────────────────────────────────────────────────
def iter_records(corpus_dir, limit=None, channels=None, shard_mod=None):
    shards = sorted(glob.glob(os.path.join(corpus_dir, "*", "shard-*.jsonl.gz")))
    if shard_mod:
        idx, of = shard_mod
        shards = [s for i, s in enumerate(shards) if i % of == idx]
    n = 0
    for sp in shards:
        ch = os.path.basename(os.path.dirname(sp))
        if channels and ch not in channels:
            continue
        try:
            with gzip.open(sp, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("text") or "").strip():
                        yield rec
                        n += 1
                        if limit and n >= limit:
                            return
        except (OSError, EOFError) as e:
            print(f"shard read error {sp}: {e!r}", flush=True)


def mode_extract(args):
    import graph.writer as gw
    if args.no_embed:
        gw.embed_batch = lambda texts: [None] * len(texts)   # defer all embedding
    from graph.writer import GraphWriter
    from ingest.extraction import extract_clauses

    channels = args.channels.split(",") if args.channels else None
    shard_mod = (args.shard, args.of) if (args.shard is not None and args.of) else None
    w = GraphWriter(args.store)
    t0 = time.time()
    n = nodes = edges = hyper = errs = 0
    list(extract_clauses("warm up the models.", llm_fallback=False))   # warm
    for rec in iter_records(args.corpus, args.limit, channels, shard_mod):
        try:
            r = w.write_clauses(list(extract_clauses(rec["text"], llm_fallback=False)))
            nodes += r["nodes"]; edges += r["edges"]; hyper += r["hyperedges"]
        except Exception as e:                                # noqa: BLE001
            errs += 1
            if errs <= 5:
                print(f"ERR {rec.get('id')}: {e!r}"[:200], flush=True)
        n += 1
        if n % args.progress_every == 0:
            dt = time.time() - t0
            print(f"[w{args.shard}] docs={n} {n/dt:.1f}/s nodes+{nodes} "
                  f"edges+{edges} hyper+{hyper} errs={errs}", flush=True)
    w.close()
    dt = time.time() - t0
    print(f"[w{args.shard}] DONE docs={n} {n/max(dt,1e-9):.1f}/s nodes={nodes} "
          f"edges={edges} hyper={hyper} errs={errs} in {dt:.0f}s", flush=True)


# ── combine ──────────────────────────────────────────────────────────────────
def mode_combine(args):
    from graph.writer import GraphWriter                       # ensures schema
    sources = []
    for pat in args.sources:
        sources.extend(sorted(glob.glob(pat)))
    if not sources:
        print("no source stores matched", flush=True); return
    w = GraphWriter(args.master)
    con = w._con
    t0 = time.time()
    for i, sp in enumerate(sources):
        con.execute("ATTACH DATABASE ? AS src", (sp,))
        con.execute(
            "INSERT INTO nodes(id,text,pos,info_vector,count,created_at,updated_at) "
            "SELECT id,text,pos,info_vector,count,created_at,updated_at FROM src.nodes WHERE true "
            "ON CONFLICT(id) DO UPDATE SET count=count+excluded.count")
        con.execute(
            "INSERT INTO edges(id,source_id,rel_type,target_id,score,count,created_at,updated_at,info_vector) "
            "SELECT id,source_id,rel_type,target_id,score,count,created_at,updated_at,info_vector "
            "FROM src.edges WHERE true ON CONFLICT(id) DO UPDATE SET count=count+excluded.count")
        con.commit()
        con.execute("DETACH DATABASE src")
        nn = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        ne = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        print(f"combined {i+1}/{len(sources)} {os.path.basename(sp)} -> "
              f"nodes={nn} edges={ne} ({time.time()-t0:.0f}s)", flush=True)
    w.close()


# ── embed ────────────────────────────────────────────────────────────────────
def _open_vec(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    vec = False
    try:
        import sqlite_vec
        con.enable_load_extension(True); sqlite_vec.load(con); vec = True
    except Exception:
        vec = False
    return con, vec


def _edge_surfaces(con):
    """Reconstruct an embeddable surface per edge: 'src_surface rel tgt_surface',
    recursing through reified (e_) endpoints. Built in memory to avoid per-row SQL."""
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


def mode_embed(args):
    from embed import embed_batch
    con, vec = _open_vec(args.store)
    print(f"sqlite-vec extension: {'ON' if vec else 'OFF'}", flush=True)

    # nodes
    rows = con.execute("SELECT id,text FROM nodes WHERE info_vector IS NULL").fetchall()
    print(f"embedding {len(rows)} nodes ...", flush=True)
    t0 = time.time(); done = 0
    for chunk in _chunks(rows, args.batch):
        vecs = embed_batch([r[1] for r in chunk])
        con.executemany("UPDATE nodes SET info_vector=? WHERE id=?",
                        [(v, r[0]) for r, v in zip(chunk, vecs)])
        con.executemany("INSERT OR REPLACE INTO nodes_fts(id,text) VALUES(?,?)",
                        [(r[0], r[1]) for r in chunk])
        if vec:
            con.executemany("INSERT OR REPLACE INTO nodes_vec(id,info_vector) VALUES(?,?)",
                            [(r[0], v) for r, v in zip(chunk, vecs) if v is not None])
        con.commit(); done += len(chunk)
        if done % (args.batch * 10) < args.batch:
            print(f"  nodes {done}/{len(rows)} {done/(time.time()-t0):.0f}/s", flush=True)

    if not args.nodes_only:
        surfaces = _edge_surfaces(con)
        eids = [r[0] for r in con.execute("SELECT id FROM edges WHERE info_vector IS NULL")]
        print(f"embedding {len(eids)} edges ...", flush=True)
        t0 = time.time(); done = 0
        for chunk in _chunks(eids, args.batch):
            vecs = embed_batch([surfaces.get(e, "") or e for e in chunk])
            con.executemany("UPDATE edges SET info_vector=? WHERE id=?",
                            [(v, e) for e, v in zip(chunk, vecs)])
            if vec:
                con.executemany("INSERT OR REPLACE INTO edges_vec(id,info_vector) VALUES(?,?)",
                                [(e, v) for e, v in zip(chunk, vecs) if v is not None])
            con.commit(); done += len(chunk)
            if done % (args.batch * 10) < args.batch:
                print(f"  edges {done}/{len(eids)} {done/(time.time()-t0):.0f}/s", flush=True)
    con.close()
    print("EMBED DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("extract")
    pe.add_argument("--corpus", required=True)
    pe.add_argument("--store", required=True)
    pe.add_argument("--limit", type=int, default=None)
    pe.add_argument("--channels", default=None)
    pe.add_argument("--shard", type=int, default=None)
    pe.add_argument("--of", type=int, default=None)
    pe.add_argument("--no-embed", action="store_true")
    pe.add_argument("--progress-every", type=int, default=500)
    pe.set_defaults(func=mode_extract)

    pc = sub.add_parser("combine")
    pc.add_argument("--master", required=True)
    pc.add_argument("--sources", nargs="+", required=True)
    pc.set_defaults(func=mode_combine)

    pm = sub.add_parser("embed")
    pm.add_argument("--store", required=True)
    pm.add_argument("--batch", type=int, default=4096)
    pm.add_argument("--nodes-only", action="store_true")
    pm.set_defaults(func=mode_embed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
