#!/usr/bin/env python3
"""Compact health check for the bulk_scrape job(s).

Authoritative article counts come from index.sqlite3 (shared by every scraper
process). Liveness/rate/heartbeat come from each process's progress_*.json.
Two processes are expected:
  wiki   -> pid file run.pid,        progress.json
  arxiv  -> pid file run_arxiv.pid,  progress_arxiv.json

Prints a per-source summary and a machine-readable VERDICT + RECOVER line:
  VERDICT = RUNNING_HEALTHY | RUNNING_SLOW | STALLED | DIED | DONE | NO_DATA
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_scrape import DEFAULT_TARGETS, TIER, DB_PATH, OUT_DIR  # noqa: E402

CD = "cd /Users/jacksonadams/Home/Projects/Decentralized-Intelligence"
SOURCES = {
    "wiki":  {"channels": ["wiki_taxonomy", "wiki_random"],
              "pid": OUT_DIR / "run.pid", "prog": OUT_DIR / "progress.json",
              "recover": f"{CD} && nohup ./venv/bin/python3 scripts/bulk_scrape.py --run "
                         f"> data/raw/corpus/run.out 2>&1 & echo $! > data/raw/corpus/run.pid"},
    "arxiv": {"channels": ["arxiv_main", "arxiv_niche"],
              "pid": OUT_DIR / "run_arxiv.pid", "prog": OUT_DIR / "progress_arxiv.json",
              "recover": f"{CD} && nohup ./venv/bin/python3 scripts/bulk_scrape.py --run "
                         f"--only arxiv --tag arxiv > data/raw/corpus/run_arxiv.out 2>&1 "
                         f"& echo $! > data/raw/corpus/run_arxiv.pid"},
}


def alive(pidfile):
    try:
        pid = int(Path(pidfile).read_text().strip())
    except Exception:
        return None, None
    try:
        os.kill(pid, 0)
        return pid, True
    except OSError:
        return pid, False


def sql_counts():
    if not Path(DB_PATH).exists():
        return {}
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=30)
        rows = con.execute("SELECT channel, COUNT(*) FROM seen GROUP BY channel").fetchall()
        con.close()
        return {r[0]: r[1] for r in rows}
    except sqlite3.Error as e:
        print("sqlite read error:", e)
        return {}


def load_prog(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main():
    counts = sql_counts()
    if not counts:
        print("VERDICT=NO_DATA")
        print("RECOVER=none")
        return
    target_total = sum(DEFAULT_TARGETS.values())
    total = sum(counts.get(ch, 0) for ch in DEFAULT_TARGETS)
    pct = 100 * total / target_total

    # long-baseline rate: articles since the last status call >10 min ago.
    # Robust to per-tick wiki AIMD backoff noise that wrecks instantaneous ETAs.
    bpath = OUT_DIR / "status_baseline.json"
    rate_base = None
    try:
        b = json.loads(bpath.read_text())
        dt = time.time() - b["ts"]
        if 60 <= dt <= 3 * 3600 and total >= b["total"]:
            rate_base = (total - b["total"]) / dt * 60
        if dt > 600:  # refresh baseline at most every 10 min
            bpath.write_text(json.dumps({"ts": time.time(), "total": total}))
    except Exception:
        bpath.write_text(json.dumps({"ts": time.time(), "total": total}))

    rate_min = 0.0
    recover = "none"
    verdict_rank = {"DONE": 0, "RUNNING_HEALTHY": 1, "RUNNING_SLOW": 2,
                    "STALLED": 3, "DIED": 4}
    worst = "DONE"
    lines = []
    for name, s in SOURCES.items():
        chans = s["channels"]
        sub = sum(counts.get(c, 0) for c in chans)
        tgt = sum(DEFAULT_TARGETS[c] for c in chans)
        done = all(counts.get(c, 0) >= DEFAULT_TARGETS[c] for c in chans)
        pid, al = alive(s["pid"])
        prog = load_prog(s["prog"])
        hb = (time.time() - prog["updated"]) if prog and "updated" in prog else None
        r = prog.get("rate_per_min_recent", 0) if prog else 0
        if al and not done:
            rate_min += r
        # per-source verdict
        if done:
            v = "DONE"
        elif al is False:
            v = "DIED"
        elif al is None:
            v = "DIED" if sub < tgt else "DONE"
        elif hb is not None and hb > 90:
            v = "STALLED"
        else:
            v = "RUNNING_HEALTHY"
        if v in ("DIED", "STALLED") and not done:
            recover = s["recover"]
        if verdict_rank[v] > verdict_rank[worst]:
            worst = v
        cc = "  ".join(f"{c.split('_')[1]}={counts.get(c,0):,}" for c in chans)
        hb_s = f"{hb:4.0f}s" if hb is not None else "  - "
        lines.append(f"  {name:<5} pid={pid} alive={al} hb={hb_s} done={done}  "
                     f"{sub:,}/{tgt:,}  [{cc}]")

    # prefer the long-baseline rate; fall back to summed instantaneous rate
    rate_eff = rate_base if rate_base is not None else rate_min
    if total >= 0.999 * target_total:
        worst = "DONE"
        recover = "none"
    elif worst == "RUNNING_HEALTHY" and rate_eff < 150 and pct > 1:
        worst = "RUNNING_SLOW"

    remaining = max(0, target_total - total)
    eta = remaining / (rate_eff / 60) if rate_eff > 1 else None
    eta_h = "—" if eta is None else (f"{int(eta//3600)}h{int((eta%3600)//60):02d}m"
                                     if eta >= 3600 else f"{int(eta//60)}m")
    rate_src = "60m-avg" if rate_base is not None else "inst"

    main_c = sum(counts.get(c, 0) for c in DEFAULT_TARGETS if TIER[c] == "main")
    niche_c = sum(counts.get(c, 0) for c in DEFAULT_TARGETS if TIER[c] == "niche")
    msum = main_c + niche_c

    print(f"total {total:,} / {target_total:,} ({pct:.1f}%)   "
          f"rate {rate_eff:,.0f}/min ({rate_src})   eta {eta_h}")
    print("sources:")
    for ln in lines:
        print(ln)
    if msum:
        print(f"mix:  main {main_c:,} ({100*main_c/msum:.0f}%)   "
              f"niche {niche_c:,} ({100*niche_c/msum:.0f}%)   [goal 80/20]")
    print(f"VERDICT={worst}")
    print(f"RECOVER={recover}")


if __name__ == "__main__":
    main()
