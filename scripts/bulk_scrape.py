#!/usr/bin/env python3
"""Autonomous bulk corpus scraper for Wikipedia + arXiv.

Builds a broad knowledge corpus across (nearly) every domain, mixing
~80% "main" topics (broad domains walked via Wikipedia's category graph +
mainstream arXiv categories) with ~20% "niche" topics (uniform-random
Wikipedia sampling + obscure arXiv subcategories).

Free official APIs only:
  - Wikipedia Action API   (intro plaintext extracts, 20 articles / request)
  - arXiv Atom API         (title + abstract, polite 1 request / 3s)

Wikipedia rate-limits aggressively (HTTP 429). We are a good citizen:
  - one global AIMD limiter shared by all wiki workers (additive increase on
    success, halve on 429) so the rate self-tunes to what the API allows
  - honor Retry-After, send maxlag=5, retry the same batch instead of dropping
  - probabilistic 80/20 main/niche split so the niche channel never starves

Output (all under data/raw/corpus/):
  index.sqlite3            dedup index + counts + visited-category resume state
  <channel>/shard-*.jsonl.gz   gzipped JSONL, one article per line
  progress.json            live heartbeat: counts, rate, ETA  (rewritten ~5s)
  scrape.log               append-only progress log

Records:  {id, source, channel, tier, title, url, primary, text, fetched_at}

Modes:
  --calibrate SECONDS   run every channel for N seconds, print projected rate
  --run                 run until per-channel targets are met (resumable)
  --target N            override total target (scales default channel split)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import signal
import sqlite3
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
# Output dir is overridable so independent sweeps can target separate corpora
# (e.g. a second 500k pass that must not overlap the first). Everything below
# derives from OUT_DIR, so one env var redirects the whole run. Default unchanged.
OUT_DIR = Path(os.environ.get("DI_CORPUS_DIR") or (ROOT / "data" / "raw" / "corpus"))
DB_PATH = OUT_DIR / "index.sqlite3"
PROGRESS_PATH = OUT_DIR / "progress.json"
LOG_PATH = OUT_DIR / "scrape.log"

CONTACT = "jacksoswag@proton.me"
UA = f"DI-corpus-builder/1.0 (knowledge corpus; {CONTACT})"
WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "http://export.arxiv.org/api/query"

# ---- targets ---------------------------------------------------------------
# main (~80%): wiki_taxonomy + arxiv_main    niche (~20%): wiki_random + arxiv_niche
DEFAULT_TARGETS = {
    "wiki_taxonomy": 300_000,  # main  (broad domains + their sub-topics)
    "arxiv_main":    100_000,  # main  (mainstream research categories)
    "wiki_random":    75_000,  # niche (uniform-random long tail)
    "arxiv_niche":    25_000,  # niche (obscure subcategories)
}
TIER = {
    "wiki_taxonomy": "main", "arxiv_main": "main",
    "wiki_random": "niche", "arxiv_niche": "niche",
}

# ---- Wikipedia tuning ------------------------------------------------------
WIKI_WORKERS = 3                       # concurrent wiki connections (kept low)
WIKI_BATCH = 20                        # articles per request (API cap w/ extracts)
WIKI_ARTICLES_PER_CATEGORY = 60        # breadth: drop a category after this many
WIKI_NICHE_FRACTION = 0.2             # share of wiki batches spent on random/niche
WIKI_MIN_CHARS_MAIN = 60               # quality bar for main/broad articles
WIKI_MIN_CHARS_NICHE = 40              # niche stubs are legitimately niche content
FRONTIER_CAP = 250_000                 # stop enqueuing subcats beyond this
WIKI_RETRIES = 6                       # per-batch retries on 429/503/maxlag
# AIMD limiter (requests/sec) — self-tunes to the API's tolerance
WIKI_RATE_START = 4.0
WIKI_RATE_FLOOR = 0.5
WIKI_RATE_CEIL = 9.0
WIKI_RATE_INCREASE = 0.06              # additive bump per successful request
WIKI_RATE_DECREASE = 0.5              # multiplicative cut per 429/503

# ---- arXiv tuning ----------------------------------------------------------
# The legacy search API (/api/query) throttles hard (429) and can't deliver bulk;
# we harvest via OAI-PMH (/oai2) instead — arXiv's sanctioned bulk endpoint
# (~1300 records/request, graceful 503+Retry-After flow control).
ARXIV_BATCH = 200                      # (legacy search API only; unused by OAI worker)
ARXIV_MIN_INTERVAL = 3.0
ARXIV_MAX_START = 30_000
ARXIV_OAI_URL = "https://export.arxiv.org/oai2"
ARXIV_OAI_GAP = 3.0                    # seconds between OAI requests (politeness)
ARXIV_OAI_SETS = [                     # top-level arXiv archives → broad coverage
    "cs", "math", "physics", "astro-ph", "cond-mat", "hep-ph", "hep-th",
    "gr-qc", "quant-ph", "nucl-th", "nlin", "math-ph", "stat", "eess",
    "econ", "q-bio", "q-fin", "hep-ex", "hep-lat", "nucl-ex",
]

SHARD_SIZE = 25_000                    # records per gzip shard
PROGRESS_INTERVAL = 5.0                # seconds between progress.json rewrites

# Broad domain seeds — Wikipedia category titles spanning "everything".
SEED_CATEGORIES = [
    "Science", "Mathematics", "Physics", "Chemistry", "Biology", "Medicine",
    "Health", "Technology", "Engineering", "Computer science",
    "Information technology", "Earth sciences", "Astronomy", "Geology",
    "Environment", "Climate", "Ecology", "Agriculture", "Energy",
    "History", "Geography", "Countries", "Cities", "World War II",
    "Philosophy", "Religion", "Mythology", "Ethics", "Logic",
    "Psychology", "Sociology", "Anthropology", "Linguistics", "Language",
    "Economics", "Business", "Finance", "Management", "Marketing",
    "Politics", "Law", "Government", "Military", "International relations",
    "Education", "Literature", "Poetry", "Art", "Visual arts",
    "Music", "Film", "Television", "Theatre", "Dance",
    "Architecture", "Design", "Fashion", "Photography",
    "Sports", "Games", "Food and drink", "Cuisine", "Tourism",
    "Transport", "Aviation", "Automobiles", "Ships", "Railways",
    "Society", "Culture", "Communication", "Mass media", "Internet",
    "People", "Organizations", "Events", "Nature", "Animals", "Plants",
    "Materials science", "Statistics", "Electronics", "Robotics",
    "Telecommunications", "Manufacturing",
]

# A resumed/second sweep supplies its own fresh, not-yet-walked seed categories
# via a file (one bare Category title per line). This lets the taxonomy walk
# continue into unexplored territory instead of deadlocking when every default
# seed is already in the carried-over visited_cat. Unset => broad defaults above.
_seeds_file = os.environ.get("DI_SEEDS_FILE")
if _seeds_file and Path(_seeds_file).exists():
    _override = [ln.strip() for ln in Path(_seeds_file).read_text().splitlines() if ln.strip()]
    if _override:
        SEED_CATEGORIES = _override

ARXIV_MAIN_CATS = [
    "cs.LG", "cs.CV", "cs.CL", "cs.AI", "cs.CR", "cs.DS", "cs.RO", "cs.SE",
    "cs.NI", "cs.DC", "cs.IR", "cs.HC", "cs.SY",
    "math.CO", "math.PR", "math.NT", "math.AG", "math.OC", "math.DG",
    "math.AP", "math.ST", "math.RT", "math.GT",
    "physics.optics", "physics.flu-dyn", "physics.app-ph", "physics.chem-ph",
    "astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE", "astro-ph.EP",
    "cond-mat.mes-hall", "cond-mat.str-el", "cond-mat.mtrl-sci",
    "cond-mat.stat-mech", "cond-mat.soft",
    "quant-ph", "hep-ph", "hep-th", "gr-qc", "nucl-th",
    "q-bio.NC", "q-bio.PE", "stat.ML", "stat.ME", "stat.AP",
    "econ.EM", "eess.SP", "eess.IV", "eess.SY", "eess.AS",
]

ARXIV_NICHE_CATS = [
    "math.GN", "math.CT", "math.KT", "math.GM", "math.HO", "math.SG",
    "math.OA", "math.AC", "math.LO",
    "cs.OH", "cs.GL", "cs.MS", "cs.SC", "cs.AR", "cs.PF", "cs.DM",
    "cs.ET", "cs.FL", "cs.MA",
    "physics.hist-ph", "physics.pop-ph", "physics.ed-ph", "physics.geo-ph",
    "physics.space-ph", "physics.ao-ph", "physics.plasm-ph", "physics.bio-ph",
    "nlin.CG", "nlin.SI", "nlin.PS", "nlin.AO", "nlin.CD",
    "q-bio.OT", "q-bio.SC", "q-bio.TO", "q-bio.CB", "q-bio.MN",
    "q-fin.PM", "q-fin.TR", "q-fin.RM", "q-fin.CP", "econ.TH", "econ.GN",
    "astro-ph.IM", "hep-ex", "hep-lat", "nucl-ex", "math-ph",
]

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
AX_NS = "{http://arxiv.org/OAI/arXiv/}"
ARXIV_NICHE_SET = set(ARXIV_NICHE_CATS)


# ---- shared state ----------------------------------------------------------
class State:
    def __init__(self, targets):
        self.targets = targets
        self.stop = threading.Event()
        self.counts = {ch: 0 for ch in targets}
        self.errors = {"wiki": 0, "arxiv": 0}
        self.last_error = ""
        self.err_samples = {}
        self.counts_lock = threading.Lock()
        self.start = time.monotonic()
        self.start_wall = time.time()
        self.frontier = deque()
        self.frontier_lock = threading.Lock()
        self.visited_cat = set()
        self.frontier_enqueued = 0

    def add(self, channel, n=1):
        if n <= 0:
            return
        with self.counts_lock:
            self.counts[channel] += n

    def err(self, kind, msg=""):
        with self.counts_lock:
            self.errors[kind] += 1
            if msg:
                self.last_error = msg[:300]
                key = msg.split("\n")[0][:90]
                self.err_samples[key] = self.err_samples.get(key, 0) + 1
                if len(self.err_samples) > 12:
                    drop = min(self.err_samples, key=self.err_samples.get)
                    del self.err_samples[drop]

    def total(self):
        with self.counts_lock:
            return sum(self.counts.values())

    def channel_done(self, channel):
        with self.counts_lock:
            return self.counts[channel] >= self.targets[channel]

    def all_done(self):
        with self.counts_lock:
            return all(self.counts[c] >= self.targets[c] for c in self.targets)


class AdaptiveLimiter:
    """AIMD token-bucket: additive increase on success, halve on throttle.

    Self-tunes the request rate to whatever the API tolerates without us
    needing to know the exact limit in advance.
    """
    def __init__(self, start, floor, ceil, inc, dec):
        self.rate = float(start)
        self.floor = float(floor)
        self.ceil = float(ceil)
        self.inc = float(inc)
        self.dec = float(dec)
        self.tokens = float(start)
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def take(self, stop_event):
        while not stop_event.is_set():
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.ceil, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                wait = (1.0 - self.tokens) / max(self.rate, 0.05)
            stop_event.wait(min(wait, 0.5))
        return False

    def reward(self):
        with self.lock:
            self.rate = min(self.ceil, self.rate + self.inc)

    def penalize(self):
        with self.lock:
            self.rate = max(self.floor, self.rate * self.dec)
            self.tokens = 0.0

    def snapshot_rate(self):
        with self.lock:
            return round(self.rate, 2)


class Index:
    """SQLite dedup index + visited-category resume state."""
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS seen("
            "id TEXT PRIMARY KEY, channel TEXT, tier TEXT, ts REAL)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS visited_cat(title TEXT PRIMARY KEY)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS arxiv_cursor(key TEXT PRIMARY KEY, start INTEGER)")
        self.conn.commit()
        self.lock = threading.Lock()

    def add_new(self, ids_meta):
        new = set()
        now = time.time()
        with self.lock:
            cur = self.conn.cursor()
            for _id, ch, tier in ids_meta:
                cur.execute("INSERT OR IGNORE INTO seen VALUES(?,?,?,?)", (_id, ch, tier, now))
                if cur.rowcount == 1:
                    new.add(_id)
            self.conn.commit()
        return new

    def mark_cat(self, title):
        with self.lock:
            self.conn.execute("INSERT OR IGNORE INTO visited_cat VALUES(?)", (title,))
            self.conn.commit()

    def cat_seen(self, title):
        with self.lock:
            return self.conn.execute(
                "SELECT 1 FROM visited_cat WHERE title=?", (title,)).fetchone() is not None

    def load_counts(self):
        with self.lock:
            rows = self.conn.execute(
                "SELECT channel, COUNT(*) FROM seen GROUP BY channel").fetchall()
            cats = self.conn.execute("SELECT title FROM visited_cat").fetchall()
        return {r[0]: r[1] for r in rows}, set(c[0] for c in cats)

    def save_arxiv_cursors(self, cursors):
        with self.lock:
            cur = self.conn.cursor()
            for (ch, c), st in cursors.items():
                cur.execute("INSERT OR REPLACE INTO arxiv_cursor VALUES(?,?)",
                            (f"{ch}|{c}", st))
            self.conn.commit()

    def load_arxiv_cursors(self):
        out = {}
        with self.lock:
            try:
                rows = self.conn.execute("SELECT key, start FROM arxiv_cursor").fetchall()
            except sqlite3.Error:
                return out
        for k, st in rows:
            if "|" in k:
                ch, c = k.split("|", 1)
                out[(ch, c)] = st
        return out


class ShardWriter:
    """Per-channel gzip JSONL shard writer."""
    def __init__(self, channel):
        self.dir = OUT_DIR / channel
        self.dir.mkdir(parents=True, exist_ok=True)
        self.channel = channel
        self.lock = threading.Lock()
        self.count = 0
        self.idx = len(sorted(self.dir.glob("shard-*.jsonl.gz")))
        self.fh = None
        self._open()

    def _open(self):
        self.fh = gzip.open(self.dir / f"shard-{self.idx:05d}.jsonl.gz", "at", encoding="utf-8")

    def write(self, records):
        with self.lock:
            for rec in records:
                self.fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.count += 1
                if self.count % SHARD_SIZE == 0:
                    self.fh.close()
                    self.idx += 1
                    self._open()
            self.fh.flush()

    def close(self):
        with self.lock:
            if self.fh:
                self.fh.close()
                self.fh = None


_thread_local = threading.local()


def session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _thread_local.session = s
    return s


def commit_records(recs, index, writers, state, channel):
    if not recs:
        return 0
    new_ids = index.add_new([(r["id"], channel, TIER[channel]) for r in recs])
    fresh = [r for r in recs if r["id"] in new_ids]
    if fresh:
        writers[channel].write(fresh)
        state.add(channel, len(fresh))
    return len(fresh)


# ---- Wikipedia -------------------------------------------------------------
def wiki_get(params, state, limiter):
    """GET with AIMD rate control + 429/503/maxlag backoff. Returns dict or None."""
    params = {**params, "format": "json", "formatversion": 2, "maxlag": 5}
    label = params.get("gcmtitle") or params.get("cmtitle") or "random"
    for attempt in range(WIKI_RETRIES):
        if not limiter.take(state.stop):
            return None
        try:
            r = session().get(WIKI_API, params=params, timeout=30)
        except requests.RequestException as exc:
            state.err("wiki", f"net[{label}]: {exc!r}")
            if state.stop.wait(min(10, 1.5 * (attempt + 1))):
                return None
            continue
        if r.status_code in (429, 503):
            ra = r.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else min(30, 2 * (attempt + 1))
            limiter.penalize()
            state.err("wiki", f"{r.status_code}[{label}] retry-after={ra or '-'}")
            if state.stop.wait(wait):
                return None
            continue
        try:
            j = r.json()
        except ValueError:
            state.err("wiki", f"badjson[{label}] status={r.status_code}")
            if state.stop.wait(1.0):
                return None
            continue
        if isinstance(j, dict) and j.get("error", {}).get("code") == "maxlag":
            ra = r.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else 5.0
            state.err("wiki", f"maxlag[{label}]")
            if state.stop.wait(wait):
                return None
            continue
        limiter.reward()
        return j
    return None


def wiki_extract_records(pages, channel, tier, min_chars):
    recs = []
    for p in pages:
        if p.get("ns") != 0 or p.get("missing"):
            continue
        text = (p.get("extract") or "").strip()
        if len(text) < min_chars:
            continue
        pid = p.get("pageid")
        if pid is None:
            continue
        recs.append({
            "id": f"wiki:{pid}",
            "source": "wikipedia",
            "channel": channel,
            "tier": tier,
            "title": p.get("title", ""),
            "url": "https://en.wikipedia.org/?curid=%d" % pid,
            "primary": None,
            "text": text,
            "fetched_at": time.time(),
        })
    return recs


def wiki_fetch_subcats(cat, state, limiter, index):
    data = wiki_get({
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{cat}", "cmnamespace": 14, "cmlimit": 500,
    }, state, limiter)
    if not data:
        return
    subcats = [m["title"].split(":", 1)[-1]
               for m in data.get("query", {}).get("categorymembers", [])]
    with state.frontier_lock:
        if state.frontier_enqueued >= FRONTIER_CAP:
            return
        for sc in subcats:
            if sc not in state.visited_cat:
                state.visited_cat.add(sc)
                state.frontier.append(sc)
                state.frontier_enqueued += 1


def wiki_random_batch(state, limiter, index, writers):
    data = wiki_get({
        "action": "query", "prop": "extracts",
        "explaintext": 1, "exintro": 1, "exlimit": WIKI_BATCH,
        "exsectionformat": "plain",
        "generator": "random", "grnnamespace": 0, "grnlimit": WIKI_BATCH,
    }, state, limiter)
    if not data:
        return
    pages = data.get("query", {}).get("pages", [])
    recs = wiki_extract_records(pages, "wiki_random", "niche", WIKI_MIN_CHARS_NICHE)
    commit_records(recs, index, writers, state, "wiki_random")


def wiki_article_batch(cat, cont, state, limiter):
    params = {
        "action": "query", "prop": "extracts",
        "explaintext": 1, "exintro": 1, "exlimit": WIKI_BATCH,
        "exsectionformat": "plain", "redirects": 1,
        "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
        "gcmnamespace": 0, "gcmlimit": WIKI_BATCH,
    }
    if cont:
        params.update(cont)
    data = wiki_get(params, state, limiter)
    if not data:
        return [], None
    return data.get("query", {}).get("pages", []), data.get("continue")


def wiki_worker(state, limiter, index, writers):
    """Unified worker: each loop does a niche (random) or main (taxonomy) batch,
    mixed to WIKI_NICHE_FRACTION, sharing one global AIMD limiter."""
    rnd = random.Random()
    cur_cat, cur_pulled, cur_cont = None, 0, None
    while not state.stop.is_set():
        main_done = state.channel_done("wiki_taxonomy")
        niche_done = state.channel_done("wiki_random")
        if main_done and niche_done:
            return

        do_niche = (not niche_done) and (main_done or rnd.random() < WIKI_NICHE_FRACTION)
        if do_niche:
            wiki_random_batch(state, limiter, index, writers)
            continue

        # taxonomy path
        if cur_cat is None:
            with state.frontier_lock:
                cur_cat = state.frontier.popleft() if state.frontier else None
            if cur_cat is None:
                if not niche_done:                 # keep busy on niche while frontier refills
                    wiki_random_batch(state, limiter, index, writers)
                else:
                    state.stop.wait(0.5)
                continue
            if index.cat_seen(cur_cat):
                cur_cat = None
                continue
            index.mark_cat(cur_cat)
            cur_pulled, cur_cont = 0, None
            wiki_fetch_subcats(cur_cat, state, limiter, index)

        pages, cont = wiki_article_batch(cur_cat, cur_cont, state, limiter)
        recs = wiki_extract_records(pages, "wiki_taxonomy", "main", WIKI_MIN_CHARS_MAIN)
        commit_records(recs, index, writers, state, "wiki_taxonomy")
        cur_pulled += len(pages)
        cur_cont = cont
        if not cont or cur_pulled >= WIKI_ARTICLES_PER_CATEGORY:
            cur_cat = None


# ---- arXiv -----------------------------------------------------------------
def arxiv_parse(xml_text, channel, tier):
    recs = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return recs
    for e in root.findall(f"{ATOM}entry"):
        idnode = e.find(f"{ATOM}id")
        if idnode is None or not idnode.text:
            continue
        raw = idnode.text.strip().rsplit("/abs/", 1)[-1]
        tail = raw.split("/")[-1]
        aid = tail.split("v")[0] if "v" in tail else raw
        title = (e.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        summary = (e.findtext(f"{ATOM}summary") or "").strip()
        if not summary:
            continue
        prim = e.find(f"{ARXIV_NS}primary_category")
        recs.append({
            "id": f"arxiv:{aid}",
            "source": "arxiv",
            "channel": channel,
            "tier": tier,
            "title": title,
            "url": f"https://arxiv.org/abs/{aid}",
            "primary": prim.get("term") if prim is not None else None,
            "text": f"{title}\n\n{summary}",
            "fetched_at": time.time(),
        })
    return recs


def arxiv_worker(state, index, writers):
    main = [("arxiv_main", c, "main") for c in ARXIV_MAIN_CATS]
    niche = [("arxiv_niche", c, "niche") for c in ARXIV_NICHE_CATS]
    plan = []
    for i in range(max(len(main), len(niche))):
        if i < len(main):
            plan.append(main[i])
        if i < len(niche):
            plan.append(niche[i])
    cursors = {(ch, c): 0 for ch, c, _ in plan}
    cursors.update(index.load_arxiv_cursors())     # resume deep paging across restarts
    consec_err = {(ch, c): 0 for ch, c, _ in plan}
    exhausted = set()
    last = 0.0
    interval = ARXIV_MIN_INTERVAL                   # adaptive politeness gap (s)
    since_save = 0
    try:
        while not state.stop.is_set():
            if state.channel_done("arxiv_main") and state.channel_done("arxiv_niche"):
                return
            progressed = False
            for channel, cat, tier in plan:
                if state.stop.is_set():
                    return
                key = (channel, cat)
                if state.channel_done(channel) or key in exhausted:
                    continue
                start = cursors[key]
                if start >= ARXIV_MAX_START:
                    exhausted.add(key)
                    continue
                wait = interval - (time.monotonic() - last)
                if wait > 0 and state.stop.wait(wait):
                    return
                last = time.monotonic()
                status, raw_entries, recs, r = None, 0, [], None
                try:
                    r = session().get(ARXIV_API, params={
                        "search_query": f"cat:{cat}", "start": start,
                        "max_results": ARXIV_BATCH,
                        "sortBy": "submittedDate", "sortOrder": "descending",
                    }, timeout=40)
                    status = r.status_code
                    if status == 200:
                        raw_entries = r.text.count("<entry>")
                        recs = arxiv_parse(r.text, channel, tier)
                except Exception as exc:
                    state.err("arxiv", f"{cat}: {exc!r}")

                # arXiv throttles with sporadic 429 "Unknown Error" even at 3s; treat
                # as backoff-and-retry (NEVER retire a category for throttling)
                if status in (429, 503):
                    ra = r.headers.get("Retry-After", "") if r is not None else ""
                    backoff = float(ra) if ra.isdigit() else min(30.0, interval * 2)
                    interval = min(20.0, interval * 1.3)
                    state.err("arxiv", f"{cat}: {status} backoff={backoff:.0f}s")
                    if state.stop.wait(backoff):
                        return
                    continue
                if status != 200:                       # hard error: retry, retire after 6
                    consec_err[key] += 1
                    if consec_err[key] >= 6:
                        exhausted.add(key)
                    else:
                        state.stop.wait(2.0)
                    continue
                if raw_entries == 0:                    # empty 200: usually transient
                    consec_err[key] += 1
                    if consec_err[key] >= 6:
                        exhausted.add(key)              # persistently empty => past the end
                    else:
                        state.stop.wait(2.0)
                    continue
                consec_err[key] = 0
                interval = max(ARXIV_MIN_INTERVAL, interval * 0.97)   # recover toward 3s
                commit_records(recs, index, writers, state, channel)
                cursors[key] = start + ARXIV_BATCH
                progressed = True
                since_save += 1
                if since_save >= 25:
                    index.save_arxiv_cursors(cursors)
                    since_save = 0
                if raw_entries < ARXIV_BATCH:           # genuine last page
                    exhausted.add(key)
            if not progressed and len(exhausted) >= len(plan):
                return
    finally:
        index.save_arxiv_cursors(cursors)


def arxiv_oai_fetch(sess, params, state):
    """One OAI request, honoring 503 flow control. Returns (status, text) or (None, None)."""
    for attempt in range(8):
        if state.stop.is_set():
            return None, None
        try:
            r = sess.get(ARXIV_OAI_URL, params=params, timeout=90)
        except Exception as exc:
            state.err("arxiv", f"oai net: {exc!r}")
            if state.stop.wait(min(30, 5 * (attempt + 1))):
                return None, None
            continue
        if r.status_code == 503:
            ra = r.headers.get("Retry-After", "")
            wait = int(ra) if ra.isdigit() else min(30, 10 + 5 * attempt)
            state.err("arxiv", f"oai 503 retry-after={ra or '-'}")
            if state.stop.wait(wait):
                return None, None
            continue
        return r.status_code, r.text
    return None, None


def arxiv_oai_parse(text):
    """Returns (records, resumption_token, error_code). records carry raw fields."""
    out = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out, None, "parse"
    err = root.find(f"{OAI_NS}error")
    if err is not None:
        return out, None, (err.get("code") or "error")
    for rec in root.iter(f"{OAI_NS}record"):
        hdr = rec.find(f"{OAI_NS}header")
        if hdr is not None and hdr.get("status") == "deleted":
            continue
        ax = rec.find(f".//{AX_NS}arXiv")
        if ax is None:
            continue
        aid = (ax.findtext(f"{AX_NS}id") or "").strip()
        abstract = (ax.findtext(f"{AX_NS}abstract") or "").strip()
        if not aid or not abstract:
            continue
        title = (ax.findtext(f"{AX_NS}title") or "").strip().replace("\n", " ")
        cats = (ax.findtext(f"{AX_NS}categories") or "").strip()
        out.append({"aid": aid, "title": title, "abstract": abstract,
                    "primary": cats.split()[0] if cats else None})
    tnode = root.find(f".//{OAI_NS}resumptionToken")
    token = tnode.text.strip() if (tnode is not None and tnode.text and tnode.text.strip()) else None
    return out, token, None


def arxiv_worker_oai(state, index, writers):
    """Bulk-harvest arXiv via OAI-PMH, round-robin across sets for breadth.
    Tier (main/niche) is decided per-record by its primary category."""
    sess = session()
    tokens = {st: None for st in ARXIV_OAI_SETS}   # None=start, "DONE"=finished
    last = 0.0
    while not state.stop.is_set():
        if state.channel_done("arxiv_main") and state.channel_done("arxiv_niche"):
            return
        active = [st for st in ARXIV_OAI_SETS if tokens[st] != "DONE"]
        if not active:
            return
        progressed = False
        for st in active:
            if state.stop.is_set():
                return
            if state.channel_done("arxiv_main") and state.channel_done("arxiv_niche"):
                return
            wait = ARXIV_OAI_GAP - (time.monotonic() - last)
            if wait > 0 and state.stop.wait(wait):
                return
            last = time.monotonic()
            tok = tokens[st]
            q = ({"verb": "ListRecords", "resumptionToken": tok} if tok
                 else {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": st})
            status, text = arxiv_oai_fetch(sess, q, state)
            if status is None:
                return
            if status != 200:
                state.err("arxiv", f"oai {st} status={status}")
                tokens[st] = "DONE"
                continue
            parsed, newtok, err = arxiv_oai_parse(text)
            if err:
                # expired/invalid token => restart this set from the beginning
                tokens[st] = None if (tok and "esumption" in err) else "DONE"
                state.err("arxiv", f"oai {st} err={err}")
                continue
            recs_main, recs_niche = [], []
            for p in parsed:
                niche = p["primary"] in ARXIV_NICHE_SET
                channel = "arxiv_niche" if niche else "arxiv_main"
                if state.channel_done(channel):
                    continue
                rec = {
                    "id": f"arxiv:{p['aid']}", "source": "arxiv", "channel": channel,
                    "tier": "niche" if niche else "main", "title": p["title"],
                    "url": f"https://arxiv.org/abs/{p['aid']}", "primary": p["primary"],
                    "text": f"{p['title']}\n\n{p['abstract']}", "fetched_at": time.time(),
                }
                (recs_niche if niche else recs_main).append(rec)
            commit_records(recs_main, index, writers, state, "arxiv_main")
            commit_records(recs_niche, index, writers, state, "arxiv_niche")
            tokens[st] = newtok if newtok else "DONE"
            progressed = True
        if not progressed:
            return


# ---- progress --------------------------------------------------------------
def human(sec):
    if sec is None or sec == float("inf"):
        return "unknown"
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def snapshot(state, limiter, last_total, last_t, prog_path=PROGRESS_PATH, log_path=LOG_PATH):
    now = time.monotonic()
    elapsed = now - state.start
    with state.counts_lock:
        counts = dict(state.counts)
        errors = dict(state.errors)
        last_error = state.last_error
        err_samples = dict(state.err_samples)
    total = sum(counts.values())
    target_total = sum(state.targets.values())
    overall_rate = total / elapsed if elapsed > 0 else 0.0
    dt = now - last_t
    recent_rate = (total - last_total) / dt if dt > 0 else 0.0
    remaining = max(0, target_total - total)
    eta = remaining / recent_rate if recent_rate > 0.01 else float("inf")
    by_tier = {"main": 0, "niche": 0}
    for ch, n in counts.items():
        by_tier[TIER[ch]] += n
    with state.frontier_lock:
        fsize, venq = len(state.frontier), state.frontier_enqueued
    doc = {
        "updated": time.time(),
        "elapsed_sec": round(elapsed, 1),
        "elapsed_human": human(elapsed),
        "running": not state.stop.is_set(),
        "total": total,
        "target": target_total,
        "pct": round(100 * total / target_total, 2) if target_total else 0,
        "by_channel": {
            ch: {"count": counts[ch], "target": state.targets[ch], "tier": TIER[ch],
                 "pct": round(100 * counts[ch] / state.targets[ch], 1) if state.targets[ch] else 0}
            for ch in counts},
        "by_tier": {
            "main": {"count": by_tier["main"],
                     "target": state.targets["wiki_taxonomy"] + state.targets["arxiv_main"]},
            "niche": {"count": by_tier["niche"],
                      "target": state.targets["wiki_random"] + state.targets["arxiv_niche"]}},
        "rate_per_sec_overall": round(overall_rate, 2),
        "rate_per_sec_recent": round(recent_rate, 2),
        "rate_per_min_recent": round(recent_rate * 60, 1),
        "eta_sec": None if eta == float("inf") else round(eta, 0),
        "eta_human": human(eta),
        "wiki_rate_limit": limiter.snapshot_rate() if limiter else None,
        "errors": errors,
        "last_error": last_error,
        "error_samples": err_samples,
        "frontier_size": fsize,
        "frontier_enqueued": venq,
    }
    tmp = prog_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(prog_path)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "t": round(elapsed, 1), "total": total,
            "rate_min": doc["rate_per_min_recent"], "eta": doc["eta_human"],
            "wiki_rps": doc["wiki_rate_limit"],
            "by_channel": {c: counts[c] for c in counts}, "errors": errors}) + "\n")
    return total


def progress_loop(state, limiter, calibrate=None, prog_path=PROGRESS_PATH, log_path=LOG_PATH):
    # smooth the "recent rate" over a trailing ~75s window so a single bursty
    # tick (wiki AIMD backoff) can't produce a wildly wrong ETA
    hist = deque()
    while not state.stop.is_set():
        if state.stop.wait(PROGRESS_INTERVAL):
            break
        now = time.monotonic()
        hist.append((now, state.total()))
        while len(hist) > 1 and now - hist[0][0] > 75:
            hist.popleft()
        base_t, base_total = hist[0]
        snapshot(state, limiter, base_total, base_t, prog_path, log_path)
        if calibrate and (now - state.start) >= calibrate:
            state.stop.set()
            return


# ---- orchestration ---------------------------------------------------------
def seed_frontier(state, visited):
    with state.frontier_lock:
        for c in SEED_CATEGORIES:
            if c in visited or c in state.visited_cat:
                continue
            state.visited_cat.add(c)
            state.frontier.append(c)
            state.frontier_enqueued += 1


def run(targets, calibrate=None, only=None, tag=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prog_path = OUT_DIR / (f"progress_{tag}.json" if tag else "progress.json")
    log_path = OUT_DIR / (f"scrape_{tag}.log" if tag else "scrape.log")
    run_wiki = only in (None, "wiki")
    run_arxiv = only in (None, "arxiv")
    active = ([] + (["wiki_taxonomy", "wiki_random"] if run_wiki else [])
              + (["arxiv_main", "arxiv_niche"] if run_arxiv else []))

    state = State(targets)
    index = Index(DB_PATH)

    prior_counts, visited = index.load_counts()
    if not calibrate:
        with state.counts_lock:
            for ch in state.counts:
                state.counts[ch] = prior_counts.get(ch, 0)
        state.visited_cat |= visited

    writers = {ch: ShardWriter(ch) for ch in active}
    limiter = (AdaptiveLimiter(WIKI_RATE_START, WIKI_RATE_FLOOR, WIKI_RATE_CEIL,
                               WIKI_RATE_INCREASE, WIKI_RATE_DECREASE) if run_wiki else None)
    if run_wiki:
        seed_frontier(state, set() if calibrate else visited)

    def handle_sig(signum, frame):
        state.stop.set()
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    threads = []
    if run_wiki:
        threads += [threading.Thread(target=wiki_worker, args=(state, limiter, index, writers),
                                     daemon=True) for _ in range(WIKI_WORKERS)]
    if run_arxiv:
        threads.append(threading.Thread(target=arxiv_worker_oai, args=(state, index, writers),
                                        daemon=True))
    prog = threading.Thread(target=progress_loop,
                            args=(state, limiter, calibrate, prog_path, log_path), daemon=True)

    def done_active():
        with state.counts_lock:
            return all(state.counts[c] >= targets[c] for c in active)

    print(f"[start] mode={'calibrate' if calibrate else 'run'} only={only or 'all'} "
          f"active={active} resume_total={state.total()}", flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"event": "start", "calibrate": calibrate, "only": only,
                            "active": active, "resume_total": state.total(),
                            "wall": time.time()}) + "\n")

    for t in threads:
        t.start()
    prog.start()
    # Stall watchdog. A channel with a finite supply (the Wikipedia taxonomy walk,
    # whose frontier can be exhausted — especially on a resumed/second sweep) can
    # leave the run unable to reach its per-channel target, while the wiki workers
    # busy-sleep on an empty frontier and never exit. Rather than hang forever, if
    # total progress flatlines for STALL_LIMIT we redirect the unmet deficit of the
    # stalled finite channels into wiki_random (uniform-random sampling has an
    # effectively unbounded supply), so the run still converges to the SAME overall
    # total. Healthy runs never trip it (the total keeps climbing).
    STALL_LIMIT = 360.0
    last_total = state.total()
    last_progress = time.monotonic()
    reallocated = False
    try:
        while not state.stop.is_set():
            if not calibrate and done_active():
                state.stop.set()
                break
            if not any(t.is_alive() for t in threads):
                state.stop.set()
                break
            now = time.monotonic()
            cur = state.total()
            if cur > last_total:
                last_total, last_progress = cur, now
            elif not calibrate and (now - last_progress) > STALL_LIMIT:
                with state.counts_lock:
                    can_redirect = (not reallocated and run_wiki
                                    and "wiki_random" in active)
                    if can_redirect:
                        deficit = sum(max(0, targets[c] - state.counts[c])
                                      for c in active if c != "wiki_random")
                        for c in active:
                            if c != "wiki_random" and state.counts[c] < targets[c]:
                                targets[c] = state.counts[c]
                        targets["wiki_random"] += deficit
                        reallocated = True
                        msg = (f"stall >{int(STALL_LIMIT)}s; redirected {deficit} "
                               f"deficit -> wiki_random (target {targets['wiki_random']})")
                    else:
                        for c in active:        # supply truly exhausted: finish with what we have
                            if state.counts[c] < targets[c]:
                                targets[c] = state.counts[c]
                        msg = "stall; capping channels at achieved and finishing"
                print(f"[watchdog] {msg}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"event": "watchdog", "msg": msg,
                                        "wall": time.time()}) + "\n")
                last_progress, last_total = time.monotonic(), state.total()
            time.sleep(1.0)
    except KeyboardInterrupt:
        state.stop.set()

    for t in threads:
        t.join(timeout=10)
    total = snapshot(state, limiter, 0, state.start, prog_path, log_path)
    for w in writers.values():
        w.close()

    elapsed = time.monotonic() - state.start
    print(f"[done] total={total} elapsed={human(elapsed)} "
          f"rate={total / elapsed * 60:.0f}/min "
          f"wiki_rps={limiter.snapshot_rate() if limiter else 'n/a'} "
          f"errors={state.errors}", flush=True)

    if calibrate:
        with state.counts_lock:
            counts = dict(state.counts)
        proj = {}
        for ch, n in counts.items():
            rate = n / elapsed if elapsed > 0 else 0
            proj[ch] = {"measured": n, "rate_per_sec": round(rate, 2),
                        "rate_per_min": round(rate * 60, 1), "target": targets[ch],
                        "eta_sec": round(targets[ch] / rate) if rate > 0 else None,
                        "eta_human": human(targets[ch] / rate) if rate > 0 else "n/a"}
        etas = [p["eta_sec"] for p in proj.values() if p["eta_sec"]]
        wall = max(etas) if etas else None
        report = {"calibrate_sec": calibrate, "measured_total": total, "by_channel": proj,
                  "wiki_rate_limit_settled": limiter.snapshot_rate(),
                  "projected_wall_sec": wall, "projected_wall_human": human(wall),
                  "under_72h": (wall is not None and wall < 72 * 3600)}
        (OUT_DIR / "calibration.json").write_text(json.dumps(report, indent=2))
        print("CALIBRATION_REPORT=" + json.dumps(report), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--calibrate", type=float, default=None)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--only", choices=["wiki", "arxiv"], default=None,
                    help="run only one source's channels (separate process)")
    ap.add_argument("--tag", default=None,
                    help="suffix for progress_<tag>.json / scrape_<tag>.log")
    args = ap.parse_args()

    targets = dict(DEFAULT_TARGETS)
    if args.target:
        scale = args.target / sum(DEFAULT_TARGETS.values())
        targets = {ch: max(1, int(n * scale)) for ch, n in DEFAULT_TARGETS.items()}
    if not args.run and not args.calibrate:
        ap.error("specify --run or --calibrate SECONDS")
    return run(targets, calibrate=args.calibrate, only=args.only, tag=args.tag)


if __name__ == "__main__":
    sys.exit(main())
