#!/usr/bin/env python3
"""Prepare a SECOND 500k corpus sweep that does not overlap the first.

Sweep 1 wrote data/raw/corpus/ (index.sqlite3 + channel shards). Its dedup
index already records every article it pulled, so the only thing standing
between us and a clean second pass is making the scraper (a) treat all of
sweep-1's articles as already-seen, (b) start its per-channel counters at 0
again, and (c) continue the Wikipedia category walk into territory sweep 1
never reached instead of deadlocking on an all-visited frontier.

This script builds data/raw/corpus_v2/ to satisfy exactly that:

  index.sqlite3
    seen          <- every prior id, tagged channel/tier '_prior'
                     INSERT OR IGNORE on `id` makes the live scraper skip them
                     as duplicates; '_prior' is not a real channel, so the
                     scraper's count-resume starts all four channels at 0.
    visited_cat   <- every category sweep 1 actually walked, so the walk skips
                     them (cat_seen) and only spends article-fetches on fresh
                     categories.
  seeds.txt       <- a few thousand NOT-yet-walked category titles discovered by
                     expanding the default seeds; bulk_scrape.py reads this via
                     DI_SEEDS_FILE so the taxonomy frontier is non-empty.
  sweep2_manifest.json  <- the exact launch config + baseline counts.

Correctness of "no overlap" rests entirely on the seen/_prior pre-seed: any
article whose id was pulled in sweep 1 hits INSERT OR IGNORE and is dropped
before it is counted or written. The fresh-seed + visited_cat work is purely an
efficiency measure so the run reaches 500k *new* articles without wasting hours
re-walking ground sweep 1 already covered.

Idempotent: refuses to clobber an existing v2 index unless --force.

Usage:
  venv/bin/python3 scripts/prepare_sweep2.py            # build everything
  venv/bin/python3 scripts/prepare_sweep2.py --force    # rebuild from scratch
  venv/bin/python3 scripts/prepare_sweep2.py --no-network  # skip seed discovery
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PRIOR_DIR_DEFAULT = ROOT / "data" / "raw" / "corpus"
V2_DIR_DEFAULT = ROOT / "data" / "raw" / "corpus_v2"

WIKI_API = "https://en.wikipedia.org/w/api.php"
CONTACT = "jacksoswag@proton.me"
UA = f"DI-corpus-builder/1.0 (sweep-2 prep; {CONTACT})"

PRIOR_CHANNEL = "_prior"   # sentinel: excluded from dedup, ignored by count-resume


def log(msg: str) -> None:
    print(f"[prepare_sweep2] {msg}", flush=True)


# ── index construction ──────────────────────────────────────────────────────
def create_index(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("CREATE TABLE IF NOT EXISTS seen("
                "id TEXT PRIMARY KEY, channel TEXT, tier TEXT, ts REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS visited_cat(title TEXT PRIMARY KEY)")
    con.execute("CREATE TABLE IF NOT EXISTS arxiv_cursor(key TEXT PRIMARY KEY, start INTEGER)")
    con.commit()
    return con


def seed_exclusions(con: sqlite3.Connection, prior_db: Path) -> tuple[int, int]:
    """Copy every prior id into seen (as _prior) and every walked category."""
    con.execute("ATTACH DATABASE ? AS old", (str(prior_db),))
    con.execute(
        "INSERT OR IGNORE INTO seen(id, channel, tier, ts) "
        "SELECT id, ?, ?, ts FROM old.seen", (PRIOR_CHANNEL, PRIOR_CHANNEL))
    con.execute(
        "INSERT OR IGNORE INTO visited_cat(title) SELECT title FROM old.visited_cat")
    con.commit()
    n_seen = con.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
    n_cat = con.execute("SELECT COUNT(*) FROM visited_cat").fetchone()[0]
    con.execute("DETACH DATABASE old")
    return n_seen, n_cat


# ── fresh seed discovery (BFS for not-yet-walked categories) ─────────────────
def fetch_subcats(sess: requests.Session, cat: str) -> list[str]:
    params = {"action": "query", "list": "categorymembers",
              "cmtitle": f"Category:{cat}", "cmnamespace": 14, "cmlimit": 500,
              "format": "json", "formatversion": 2, "maxlag": 5}
    for attempt in range(5):
        try:
            r = sess.get(WIKI_API, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(min(10, 1.5 * (attempt + 1)))
            continue
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After", "")
            time.sleep(float(ra) if ra.isdigit() else min(20, 2 * (attempt + 1)))
            continue
        try:
            j = r.json()
        except ValueError:
            time.sleep(1.0)
            continue
        if isinstance(j, dict) and j.get("error", {}).get("code") == "maxlag":
            time.sleep(3.0)
            continue
        return [m["title"].split(":", 1)[-1]
                for m in j.get("query", {}).get("categorymembers", [])]
    return []


def discover_fresh_seeds(visited: set[str], default_seeds: list[str],
                         target: int, call_budget: int, gap: float) -> list[str]:
    """Recover sweep 1's unwalked frontier: categories it discovered but never walked.

    Sweep 1 enqueued ~127k subcategories but only walked (persisted to visited_cat)
    ~19k of them; the ~108k it never reached are exactly the fresh frontier we want.
    They live as not-yet-walked SUBcategories of the walked ones, so we expand the
    WALKED categories directly and keep every child that isn't itself walked. A
    BFS down from the original 81 seeds would instead retrace the fully-walked top
    of the tree and exhaust the budget before reaching the fringe (observed: 0 fresh
    after 150 calls). We shuffle the walked set so a small budget still samples
    broadly across domains; the live scraper expands much further from these.

    Falls back to BFS from default_seeds only if there is no walked set to expand."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    expand_order = list(visited) if visited else list(default_seeds)
    random.Random(20260614).shuffle(expand_order)
    queue = deque(expand_order)
    fetched: set[str] = set()
    collected: list[str] = []
    collected_set: set[str] = set()
    calls = 0
    last = 0.0
    while queue and len(collected) < target and calls < call_budget:
        cat = queue.popleft()
        if cat in fetched:
            continue
        fetched.add(cat)
        wait = gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.monotonic()
        subs = fetch_subcats(sess, cat)
        calls += 1
        for sub in subs:
            if sub in visited or sub in collected_set:
                continue
            collected_set.add(sub)
            collected.append(sub)
            # also queue fresh children so we keep finding the fringe if a small
            # sample of walked roots runs dry before the budget does
            if sub not in fetched:
                queue.append(sub)
        if calls % 25 == 0:
            log(f"  seed discovery: {calls} calls, {len(collected)} fresh, "
                f"queue={len(queue)}")
    return collected


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-dir", default=str(PRIOR_DIR_DEFAULT))
    ap.add_argument("--v2-dir", default=str(V2_DIR_DEFAULT))
    ap.add_argument("--fresh-seeds", type=int, default=2000,
                    help="how many not-yet-walked categories to discover")
    ap.add_argument("--seed-call-budget", type=int, default=800,
                    help="max Wikipedia category requests during seed discovery")
    ap.add_argument("--seed-gap", type=float, default=0.34,
                    help="seconds between category requests (politeness)")
    ap.add_argument("--no-network", action="store_true",
                    help="skip fresh-seed discovery (taxonomy will lean on defaults)")
    ap.add_argument("--target", type=int, default=500_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    prior_dir = Path(args.prior_dir)
    v2_dir = Path(args.v2_dir)
    prior_db = prior_dir / "index.sqlite3"
    v2_db = v2_dir / "index.sqlite3"
    seeds_path = v2_dir / "seeds.txt"
    manifest_path = v2_dir / "sweep2_manifest.json"

    if not prior_db.exists():
        log(f"ERROR: prior index not found at {prior_db}")
        return 1
    if v2_db.exists() and not args.force:
        log(f"ERROR: {v2_db} already exists. Re-run with --force to rebuild.")
        return 1

    v2_dir.mkdir(parents=True, exist_ok=True)
    log(f"prior index : {prior_db}")
    log(f"v2 corpus   : {v2_dir}")

    # Build everything in a staging location and promote atomically only after
    # seeds + manifest exist. Seed discovery is a multi-minute network step; a
    # crash partway must NOT leave a usable-looking but seedless index that the
    # launcher would happily start (taxonomy-less corpus). The live index/seeds/
    # manifest are only ever replaced as the very last step.
    stage_db = v2_dir / ".index.staging.sqlite3"
    stage_seeds = v2_dir / ".seeds.staging.txt"
    stage_manifest = v2_dir / ".manifest.staging.json"

    def _rm(*paths):
        for p in paths:
            Path(p).unlink(missing_ok=True)

    def _promote(stage: Path, final: Path):
        os.replace(stage, final)

    # clear any leftovers from a previously-crashed run
    _rm(stage_db, str(stage_db) + "-wal", str(stage_db) + "-shm",
        stage_seeds, stage_manifest)

    # import the default seeds + per-channel targets straight from the scraper
    sys.path.insert(0, str(ROOT / "scripts"))
    import bulk_scrape as bs  # noqa: E402

    try:
        con = create_index(stage_db)
        n_seen, n_cat = seed_exclusions(con, prior_db)
        log(f"excluded {n_seen} prior ids (channel '{PRIOR_CHANNEL}') and "
            f"carried {n_cat} walked categories")

        fresh: list[str] = []
        if args.no_network:
            log("skipping fresh-seed discovery (--no-network); taxonomy will lean "
                "on wiki_random via the scraper's stall watchdog")
        else:
            visited = {r[0] for r in con.execute("SELECT title FROM visited_cat")}
            log(f"discovering up to {args.fresh_seeds} fresh seeds "
                f"(<= {args.seed_call_budget} requests) ...")
            fresh = discover_fresh_seeds(visited, list(bs.SEED_CATEGORIES),
                                         args.fresh_seeds, args.seed_call_budget,
                                         args.seed_gap)
            if not fresh:
                con.close()
                log("ERROR: discovered 0 fresh categories — refusing to build a "
                    "seedless taxonomy index. Check connectivity, raise "
                    "--seed-call-budget, or pass --no-network to proceed without "
                    "the taxonomy walk.")
                return 1
            stage_seeds.write_text("\n".join(fresh) + "\n")
            log(f"staged {len(fresh)} fresh seeds")

        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()

        # scaled per-channel targets (mirror the scraper's --target scaling)
        base_total = sum(bs.DEFAULT_TARGETS.values())
        scale = args.target / base_total
        targets = {ch: max(1, int(n * scale)) for ch, n in bs.DEFAULT_TARGETS.items()}

        manifest = {
            "created_at": time.time(),
            "created_h": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prior_dir": str(prior_dir),
            "v2_dir": str(v2_dir),
            "v2_index": str(v2_db),
            "seeds_file": str(seeds_path) if fresh else None,
            "target_total": args.target,
            "channel_targets": targets,
            "excluded_prior_ids": n_seen,
            "carried_visited_cats": n_cat,
            "fresh_seeds": len(fresh),
            "launch_env": {
                "DI_CORPUS_DIR": str(v2_dir),
                "DI_SEEDS_FILE": str(seeds_path) if fresh else None,
            },
            "launch_cmd": (
                f"DI_CORPUS_DIR={v2_dir} "
                + (f"DI_SEEDS_FILE={seeds_path} " if fresh else "")
                + f"caffeinate -is venv/bin/python3 scripts/bulk_scrape.py --run --target {args.target}"
            ),
        }
        stage_manifest.write_text(json.dumps(manifest, indent=2))

        # ── atomic promote: index, then seeds, then manifest ──
        _rm(v2_db, str(v2_db) + "-wal", str(v2_db) + "-shm")
        _promote(stage_db, v2_db)
        if fresh:
            _promote(stage_seeds, seeds_path)
        else:
            _rm(seeds_path)            # no taxonomy seeds this build
        _promote(stage_manifest, manifest_path)
    finally:
        _rm(stage_db, str(stage_db) + "-wal", str(stage_db) + "-shm",
            stage_seeds, stage_manifest)

    log("─" * 60)
    log("READY. Pre-seed complete and verified-by-construction non-overlapping.")
    log(f"  excluded prior ids : {n_seen}")
    log(f"  carried categories : {n_cat}")
    log(f"  fresh taxonomy seeds: {len(fresh)}")
    log(f"  channel targets    : {targets}")
    log(f"  manifest           : {manifest_path}")
    log("Launch the scrape with scripts/run_sweep2.sh (or the manifest launch_cmd).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
