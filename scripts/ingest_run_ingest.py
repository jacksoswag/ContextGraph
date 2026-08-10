#!/usr/bin/env python3
"""Autonomous overnight orchestrator: wait for the scrape to finish, then extract
hyperedge triples from the full corpus and info-vector them. Merge is DEFERRED.

Phases:  wait_scrape -> extract (W workers, pure spaCy, no embed) -> combine -> embed
Writes data/graph/ingest_status.json (heartbeat) + ingest.log + per-phase logs.
"""
import glob
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Users/jacksonadams/Home/Projects/Decentralized-Intelligence")
WT = Path("/tmp/di-hyperedge")
VENV = str(ROOT / "venv/bin/python3")
CORPUS = ROOT / "data/raw/corpus"
OUT = ROOT / "data/graph"
DRIVER = str(WT / "scripts/corpus_ingest.py")
STATUS = OUT / "ingest_status.json"
LOG = OUT / "ingest.log"
MASTER = OUT / "graph.corpus.sqlite"
W = int(os.getenv("INGEST_WORKERS", "6"))
SCRAPE_TARGET = 499000


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def write_status(**kw):
    kw["updated"] = time.time()
    kw["updated_h"] = time.strftime("%H:%M:%S")
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(kw, indent=2))
    tmp.replace(STATUS)


def base_env(omp1=True):
    e = dict(os.environ)
    e["PYTHONPATH"] = str(WT / "src")
    e["DI_LLM_EXTRACTION"] = "0"
    e["TOKENIZERS_PARALLELISM"] = "false"
    if omp1:
        e["OMP_NUM_THREADS"] = "1"
    else:
        e.pop("OMP_NUM_THREADS", None)
    return e


def count(db, table, where=""):
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        n = c.execute(f"SELECT COUNT(*) FROM {table} {where}").fetchone()[0]
        c.close()
        return n
    except sqlite3.Error:
        return 0


def scrape_done():
    n = count(CORPUS / "index.sqlite3", "seen")
    alive = False
    try:
        pid = int((CORPUS / "run.pid").read_text().strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False
    return ((not alive) or n >= SCRAPE_TARGET), n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parts").mkdir(exist_ok=True)
    t_start = time.time()
    log(f"orchestrator start; workers={W}; target store={MASTER}")

    # PHASE 0 — wait for the scrape to finish (it writes the shards we read)
    t = time.time()
    while True:
        done, n = scrape_done()
        write_status(phase="wait_scrape", scrape_count=n,
                     elapsed=round(time.time() - t_start))
        if done:
            log(f"scrape complete (count={n})")
            break
        if time.time() - t > 3600:
            log(f"scrape wait timed out at count={n}; proceeding anyway")
            break
        time.sleep(30)

    # PHASE 1 — parallel extract, no embedding
    log(f"PHASE 1 extract: {W} workers (pure spaCy, no embed)")
    t1 = time.time()
    procs = []
    for i in range(W):
        store = OUT / "parts" / f"store_{i}.sqlite"
        if store.exists():
            store.unlink()
        lf = open(OUT / f"extract_{i}.log", "w")
        p = subprocess.Popen(
            [VENV, DRIVER, "extract", "--corpus", str(CORPUS), "--store", str(store),
             "--shard", str(i), "--of", str(W), "--no-embed", "--progress-every", "2000"],
            env=base_env(omp1=True), stdout=lf, stderr=subprocess.STDOUT)
        procs.append((p, lf))
    while any(p.poll() is None for p, _ in procs):
        alive = sum(1 for p, _ in procs if p.poll() is None)
        nodes = sum(count(OUT / "parts" / f"store_{i}.sqlite", "nodes") for i in range(W))
        edges = sum(count(OUT / "parts" / f"store_{i}.sqlite", "edges") for i in range(W))
        write_status(phase="extract", workers_alive=alive, nodes=nodes, edges=edges,
                     elapsed=round(time.time() - t_start),
                     phase_elapsed=round(time.time() - t1))
        time.sleep(30)
    for p, lf in procs:
        lf.close()
    codes = [p.returncode for p, _ in procs]
    log(f"PHASE 1 done in {time.time()-t1:.0f}s exit_codes={codes}")

    # PHASE 2 — combine part stores by deterministic id
    log("PHASE 2 combine")
    t2 = time.time()
    if MASTER.exists():
        MASTER.unlink()
    with open(OUT / "combine.log", "w") as cl:
        rc = subprocess.run([VENV, DRIVER, "combine", "--master", str(MASTER),
                             "--sources", str(OUT / "parts" / "store_*.sqlite")],
                            env=base_env(), stdout=cl, stderr=subprocess.STDOUT).returncode
    tn, te = count(MASTER, "nodes"), count(MASTER, "edges")
    log(f"PHASE 2 done in {time.time()-t2:.0f}s exit={rc} nodes={tn} edges={te}")
    write_status(phase="combined", nodes=tn, edges=te, exit_combine=rc,
                 extract_exit_codes=codes, elapsed=round(time.time() - t_start))

    # PHASE 3 — bulk embed (all cores)
    log(f"PHASE 3 embed (all cores): nodes={tn} edges={te}")
    t3 = time.time()
    with open(OUT / "embed.log", "w") as el:
        rc = subprocess.run([VENV, DRIVER, "embed", "--store", str(MASTER), "--batch", "4096"],
                            env=base_env(omp1=False), stdout=el, stderr=subprocess.STDOUT).returncode
    log(f"PHASE 3 done in {time.time()-t3:.0f}s exit={rc}")

    # final
    n = count(MASTER, "nodes")
    nv = count(MASTER, "nodes", "WHERE info_vector IS NOT NULL")
    e = count(MASTER, "edges")
    ev = count(MASTER, "edges", "WHERE info_vector IS NOT NULL")
    h = count(MASTER, "edges", "WHERE source_id LIKE 'e_%' OR target_id LIKE 'e_%'")
    write_status(phase="DONE", nodes=n, nodes_embedded=nv, edges=e, edges_embedded=ev,
                 hyperedges=h, store=str(MASTER), extract_exit_codes=codes,
                 exit_embed=rc, total_elapsed=round(time.time() - t_start))
    log(f"ALL DONE nodes={n} ({nv} vec) edges={e} ({ev} vec) hyper={h} "
        f"in {(time.time()-t_start)/3600:.1f}h -> {MASTER}")


if __name__ == "__main__":
    main()
