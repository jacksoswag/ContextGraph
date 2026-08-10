#!/usr/bin/env python3
"""Graph-corpus construction for the SECOND sweep (data/raw/corpus_v2).

Same deferred-merge pipeline as scripts/ingest_run_ingest.py, but every path is
parameterised so it reads the v2 corpus and writes a SEPARATE graph store
(data/graph_v2/graph.corpus_v2.sqlite) — sweep 1's graph is never touched.

Phases:  wait_scrape -> extract (W workers, pure spaCy, no embed) -> combine -> embed

Two things differ from the sweep-1 orchestrator and matter:

  * scrape-done counts only REAL articles. The v2 index is pre-seeded with
    ~500k '_prior' rows for dedup, so a naive COUNT(*) on seen would report the
    target as already met before a single new article is scraped. We count
    `seen WHERE channel != '_prior'` instead.

  * the ingest driver/code live in the ingest-hyperedge worktree at /tmp/di-
    hyperedge, which is on volatile /tmp. If it vanished (reboot/cleanup) we
    recreate it from the branch before starting.

Override any path via env: DI_CORPUS_DIR, DI_GRAPH_OUT, DI_MASTER,
DI_SCRAPE_PID, DI_SCRAPE_TARGET, INGEST_WORKERS.
"""
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WT = Path(os.getenv("DI_INGEST_WORKTREE", "/tmp/di-hyperedge"))
INGEST_BRANCH = "ingest-hyperedge"
VENV = str(ROOT / "venv/bin/python3")

CORPUS = Path(os.getenv("DI_CORPUS_DIR", str(ROOT / "data/raw/corpus_v2")))
OUT = Path(os.getenv("DI_GRAPH_OUT", str(ROOT / "data/graph_v2")))
MASTER = Path(os.getenv("DI_MASTER", str(OUT / "graph.corpus_v2.sqlite")))
DRIVER = str(WT / "scripts/corpus_ingest.py")
STATUS = OUT / "ingest_status.json"
LOG = OUT / "ingest.log"

W = int(os.getenv("INGEST_WORKERS", "6"))
SCRAPE_TARGET = int(os.getenv("DI_SCRAPE_TARGET", "499000"))
SCRAPE_PID_FILE = Path(os.getenv("DI_SCRAPE_PID", str(CORPUS / "run.pid")))
PRIOR_CHANNEL = "_prior"


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


def ensure_worktree():
    """The ingest code lives in a /tmp worktree; recreate it if /tmp was cleared."""
    if Path(DRIVER).exists():
        return
    log(f"ingest worktree missing at {WT}; recreating from {INGEST_BRANCH}")
    WT.parent.mkdir(parents=True, exist_ok=True)
    # A /tmp wipe removes the working dir but leaves the registration in
    # .git/worktrees/, so a plain `add` fails with "missing but already
    # registered". Prune the dangling entry first, then force-add.
    subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"],
                   capture_output=True, text=True)
    r = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "-f",
                        str(WT), INGEST_BRANCH],
                       capture_output=True, text=True)
    if r.returncode != 0 or not Path(DRIVER).exists():
        log(f"FATAL: could not recreate worktree: {r.stderr.strip()}")
        sys.exit(2)
    log("worktree recreated")


def scrape_done():
    """Done when the scrape process has exited or enough REAL (non-prior)
    articles have been collected. The '_prior' rows are dedup sentinels and
    must not count toward the target."""
    n = count(CORPUS / "index.sqlite3", "seen", f"WHERE channel != '{PRIOR_CHANNEL}'")
    # A missing pid file means the scrape was never launched (not that it
    # finished). Only treat the scrape as done if the target is already met;
    # otherwise keep waiting so we never extract an empty/partial corpus.
    if not SCRAPE_PID_FILE.exists():
        return n >= SCRAPE_TARGET, n
    alive = False
    try:
        pid = int(SCRAPE_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False
    return ((not alive) or n >= SCRAPE_TARGET), n


def main():
    ensure_worktree()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parts").mkdir(exist_ok=True)
    t_start = time.time()
    log(f"sweep-2 orchestrator start; workers={W}; corpus={CORPUS}; store={MASTER}")

    # PHASE 0 — wait for the scrape to finish (it writes the shards we read)
    t = time.time()
    while True:
        done, n = scrape_done()
        write_status(phase="wait_scrape", scrape_count_new=n,
                     elapsed=round(time.time() - t_start))
        if done:
            log(f"scrape complete (new articles={n})")
            break
        if time.time() - t > 6 * 3600:
            log(f"scrape wait timed out at new={n}; proceeding anyway")
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
