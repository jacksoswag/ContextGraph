from __future__ import annotations
# P2 corpus fetchers: Wikipedia + Semantic Scholar via their JSON APIs — NOT HTML
# scraping. Polite (descriptive User-Agent + per-host min interval) and cached (sqlite in
# .di-ui/, gitignored) so the fixed benchmark corpus fetches once and replays byte-stable.
# Google Scholar is deliberately NOT scraped (it blocks scrapers); scholarly data comes
# from the Semantic Scholar Graph API. Serper (scrape_worker) is untouched — that is the
# interactive research path; this module is the reproducible-corpus path.
import json, logging, os, sqlite3, time, urllib.parse, urllib.request
from contextlib import closing

LOGGER = logging.getLogger(__name__)
_CACHE_PATH = os.getenv("DI_CORPUS_CACHE", ".di-ui/corpus_cache.sqlite")
_UA = os.getenv("DI_HTTP_UA", "decentralized-intelligence/0.1 (research; contact jacksoswag@proton.me)")
WIKI_API = "https://en.wikipedia.org/w/api.php"
S2_API = "https://api.semanticscholar.org/graph/v1"
# polite per-host minimum gap between requests (seconds); S2 unauthenticated ~1 req/s.
_MIN_INTERVAL = {"en.wikipedia.org": 0.2, "api.semanticscholar.org": 1.1}
_last_hit: dict[str, float] = {}


def _ensure_cache() -> sqlite3.Connection:
    con = sqlite3.connect(_CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS fetch_cache(k TEXT PRIMARY KEY, v TEXT, ts INTEGER)")
    return con

def _cache_get(key: str):
    try:
        with closing(_ensure_cache()) as con:
            row = con.execute("SELECT v FROM fetch_cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception as exc:
        LOGGER.warning("corpus cache read failed: %s", exc); return None

def _cache_put(key: str, value) -> None:
    try:
        with closing(_ensure_cache()) as con:
            con.execute("INSERT OR REPLACE INTO fetch_cache(k,v,ts) VALUES(?,?,?)",
                        (key, json.dumps(value), int(time.time()))); con.commit()
    except Exception as exc:
        LOGGER.warning("corpus cache write failed: %s", exc)

# Per-host throttle: sleep just enough to honor the configured min interval.
def _throttle(host: str) -> None:
    gap = _MIN_INTERVAL.get(host, 0.0)
    if gap <= 0: return
    wait = gap - (time.monotonic() - _last_hit.get(host, 0.0))
    if wait > 0: time.sleep(wait)
    _last_hit[host] = time.monotonic()

# The single network seam (monkeypatched in tests). Adds UA + optional S2 key header.
def _http_get_json(url: str, params: dict) -> dict:
    _throttle(urllib.parse.urlparse(url).netloc)
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    s2_key = os.getenv("S2_API_KEY", "").strip()
    if s2_key and "semanticscholar.org" in url: headers["x-api-key"] = s2_key
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Wikipedia: plaintext article extract via the action API (prop=extracts) ──
def wikipedia_article(title: str, *, use_cache: bool = True) -> str:
    key = f"wiki:{title}"
    if use_cache:
        cached = _cache_get(key)
        if cached is not None: return cached.get("text", "")
    data = _http_get_json(WIKI_API, {"action": "query", "prop": "extracts", "explaintext": "1",
                                     "format": "json", "redirects": "1", "titles": title})
    pages = (data.get("query") or {}).get("pages") or {}
    text = next((p.get("extract", "") or "" for p in pages.values()), "")
    _cache_put(key, {"title": title, "text": text})
    return text


# ── Semantic Scholar: abstracts via the Graph API (search + by-id) ──
def semantic_scholar_search(query: str, *, limit: int = 10, use_cache: bool = True) -> list[dict]:
    key = f"s2search:{query}:{limit}"
    if use_cache:
        cached = _cache_get(key)
        if cached is not None: return cached
    data = _http_get_json(f"{S2_API}/paper/search",
                          {"query": query, "limit": limit, "fields": "title,abstract,paperId,year"})
    out = [{"paperId": p.get("paperId"), "title": p.get("title", ""),
            "abstract": p.get("abstract") or "", "year": p.get("year")}
           for p in (data.get("data") or []) if p.get("abstract")]
    _cache_put(key, out)
    return out

def semantic_scholar_paper(paper_id: str, *, use_cache: bool = True) -> dict:
    key = f"s2paper:{paper_id}"
    if use_cache:
        cached = _cache_get(key)
        if cached is not None: return cached
    data = _http_get_json(f"{S2_API}/paper/{paper_id}", {"fields": "title,abstract,year"})
    out = {"paperId": paper_id, "title": data.get("title", ""),
           "abstract": data.get("abstract") or "", "year": data.get("year")}
    _cache_put(key, out)
    return out
