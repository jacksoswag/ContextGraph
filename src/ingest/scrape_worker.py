# Research ingestion worker.
# This module owns search-provider calls, page fetching, and extracted-text
# caching into its own scrape cache. It is a pure text producer — extracted
# page text is handed to the ingest pipeline; it persists nothing else.
from __future__ import annotations

import hashlib, json, os, re, sqlite3, time
from pathlib import Path
from urllib.parse import urlparse

import requests, trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from ingest.labels import clean_search_query, clean_url, collapse_whitespace

ROOT = Path(__file__).resolve().parents[2]
WIKIPEDIA_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
SERPER_SEARCH_URL = "https://google.serper.dev/search"
SERPER_SCHOLAR_URL = "https://google.serper.dev/scholar"
HEADERS = {"User-Agent": "decentralized-intelligence/1.0 (+https://localhost)"}

MIN_EXTRACTABLE_BLOCK_CHARS = int(os.getenv("MIN_EXTRACTABLE_BLOCK_CHARS", "80"))
SCRAPE_DEFAULT_LIMIT = int(os.getenv("SCRAPE_DEFAULT_LIMIT", "24"))
RESEARCH_GOOGLE_SEARCH_LIMIT = int(os.getenv("RESEARCH_GOOGLE_SEARCH_LIMIT", "16"))
RESEARCH_GOOGLE_SCHOLAR_LIMIT = int(os.getenv("RESEARCH_GOOGLE_SCHOLAR_LIMIT", "8"))
RESEARCH_CACHE_PATH = Path(os.getenv("RESEARCH_CACHE_PATH", str(ROOT / ".di-ui" / "research_cache.sqlite"))).expanduser()
RESEARCH_CACHE_TTL_SECONDS = float(os.getenv("RESEARCH_CACHE_TTL_SECONDS", "604800"))
SERPER_PAGE_TIMEOUT = float(os.getenv("SERPER_PAGE_TIMEOUT", "8"))
SERPER_RESULT_TIMEOUT = float(os.getenv("SERPER_RESULT_TIMEOUT", "8"))
RESEARCH_CACHE_SCHEMA_VERSION = "research-cache-v4"

load_dotenv(Path(os.environ.get("BRAIN_ENV_PATH") or ROOT / ".env"))


# Clean a URL enough for safe HTTP scraping.
def _clean_url(url: object) -> str:
    value = clean_url(url).strip("\"'<>[]()")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return value


# Return the single provider selected for this process.
def selected_provider() -> str:
    configured = os.getenv("SCRAPE_PROVIDER", "").strip().lower()
    if configured in {"serper", "wikipedia"}:
        return configured
    return "serper" if os.getenv("SERPER_API_KEY", "").strip() else "wikipedia"


# Opens the SQLite research cache and ensures document table exists.
def _cache_connection() -> sqlite3.Connection:
    RESEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RESEARCH_CACHE_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    schema_row = conn.execute("SELECT value FROM research_meta WHERE key='schema_version'").fetchone()
    if (schema_row is not None and str(schema_row[0]) != RESEARCH_CACHE_SCHEMA_VERSION) or _research_cache_shape_is_stale(conn):
        conn.executescript(
            """
            DROP TABLE IF EXISTS research_documents;
            DROP TABLE IF EXISTS research_blocks;
            DROP TABLE IF EXISTS research_lookup_cache;
            """
        )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS research_lookup_cache(
            namespace TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            value TEXT NOT NULL,
            created REAL NOT NULL,
            PRIMARY KEY(namespace, cache_key)
        );
        CREATE TABLE IF NOT EXISTS research_documents(
            query TEXT NOT NULL,
            provider TEXT NOT NULL,
            search_source TEXT NOT NULL,
            url TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            status TEXT NOT NULL,
            title TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            raw_html TEXT NOT NULL,
            body_text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            error TEXT NOT NULL,
            fetched_at REAL NOT NULL,
            PRIMARY KEY(query, provider, search_source, url)
        );
        CREATE INDEX IF NOT EXISTS research_documents_query_idx
            ON research_documents(query, provider, search_source, source_rank);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO research_meta(key, value) VALUES(?, ?)",
        ("schema_version", RESEARCH_CACHE_SCHEMA_VERSION),
    )
    conn.commit()
    return conn


# Detect caches built before search_source became part of the document key.
def _research_cache_shape_is_stale(conn: sqlite3.Connection) -> bool:
    document_columns = tuple(str(row[1]) for row in conn.execute("PRAGMA table_info(research_documents)").fetchall())
    if not document_columns:
        return False
    return "search_source" not in document_columns


# Serializes a request payload into a stable lookup-cache key.
def _cache_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# Reads a cached provider value by namespace and key.
def _lookup_cache_get(namespace: str, key: str) -> object | None:
    conn = None
    try:
        conn = _cache_connection()
        row = conn.execute(
            "SELECT value FROM research_lookup_cache WHERE namespace=? AND cache_key=?",
            (namespace, key),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


# Stores a provider value in the shared research/scrape cache.
def _lookup_cache_set(namespace: str, key: str, value: object) -> None:
    conn = None
    try:
        encoded = json.dumps(value)
        conn = _cache_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_lookup_cache(namespace, cache_key, value, created) VALUES(?, ?, ?, ?)",
                (namespace, key, encoded, time.time()),
            )
    except (TypeError, sqlite3.Error):
        return
    finally:
        if conn is not None:
            conn.close()


# Returns one Serper result page for a Google search surface.
def _serper_page(query: str, page: int, page_size: int, *, surface: str) -> list[dict[str, object]]:
    if surface == "google_scholar":
        endpoint = SERPER_SCHOLAR_URL
        namespace = "serper_scholar_page"
    else:
        endpoint = SERPER_SEARCH_URL
        namespace = "serper_search_page"
        surface = "google_search"
    key = _cache_key({"query": query, "page": int(page), "page_size": int(page_size), "surface": surface})
    cached = _lookup_cache_get(namespace, key)
    if isinstance(cached, list):
        return [item for item in cached if isinstance(item, dict)]
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return []
    response = requests.post(
        endpoint,
        headers={**HEADERS, "X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": int(page_size), "page": int(page)},
        timeout=SERPER_RESULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    results: list[dict[str, object]] = []
    for position, item in enumerate(payload.get("organic", []) or [], start=(int(page) - 1) * int(page_size) + 1):
        url = _clean_url(item.get("link", ""))
        if not url:
            continue
        row: dict[str, object] = {
            "title": collapse_whitespace(item.get("title", "")),
            "url": url,
            "snippet": collapse_whitespace(item.get("snippet", "")),
            "search_source": surface,
            "source_rank": int(item.get("position") or position),
        }
        if surface == "google_scholar":
            row.update(
                {
                    "publication_info": collapse_whitespace(item.get("publicationInfo", "")),
                    "year": item.get("year", ""),
                    "cited_by": item.get("citedBy", ""),
                }
            )
        results.append(row)
    _lookup_cache_set(namespace, key, results)
    return results


# Returns Serper organic Google Search results used by research.
def _serper_organic_results(query: str, limit: int) -> list[dict[str, object]]:
    return _serper_surface_results(query, limit, surface="google_search")


# Returns Serper Google Scholar results used by research.
def _serper_scholar_results(query: str, limit: int) -> list[dict[str, object]]:
    return _serper_surface_results(query, limit, surface="google_scholar")


# Returns one Serper surface with pagination.
def _serper_surface_results(query: str, limit: int, *, surface: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    page_size = 10
    page_count = (max(1, int(limit)) + page_size - 1) // page_size
    for page in range(1, page_count + 1):
        page_results = _serper_page(query, page, page_size, surface=surface)
        if not page_results:
            break
        results.extend(page_results)
        if len(results) >= int(limit):
            break
    return results[:limit]


# Returns the research mix: Google and Scholar results interleaved, Google-first.
def _serper_results(query: str, limit: int) -> list[dict[str, object]]:
    google = _serper_organic_results(query, 10)[:10]
    scholar = _serper_scholar_results(query, 6)[:6]
    results: list[dict[str, object]] = []
    for g, s in zip(google, scholar):
        results.append(g)
        results.append(s)
    n_paired = min(len(google), len(scholar))
    results.extend(google[n_paired:])
    return results[: max(0, int(limit))]


# Returns Wikipedia OpenSearch results used by research.
def _wikipedia_results(query: str, limit: int) -> list[dict[str, object]]:
    key = _cache_key({"query": query, "limit": int(limit)})
    cached = _lookup_cache_get("wikipedia_results", key)
    if isinstance(cached, list):
        return [item for item in cached if isinstance(item, dict)]
    response = requests.get(
        WIKIPEDIA_OPENSEARCH_URL,
        params={
            "action": "opensearch",
            "search": query,
            "limit": min(max(1, int(limit)), 10),
            "namespace": 0,
            "format": "json",
        },
        headers=HEADERS,
        timeout=SERPER_RESULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    titles = payload[1] if len(payload) > 1 else []
    snippets = payload[2] if len(payload) > 2 else []
    urls = payload[3] if len(payload) > 3 else []
    results = [
        {
            "title": collapse_whitespace(titles[idx] if idx < len(titles) else ""),
            "snippet": collapse_whitespace(snippets[idx] if idx < len(snippets) else ""),
            "url": _clean_url(urls[idx] if idx < len(urls) else ""),
            "search_source": "wikipedia",
            "source_rank": idx + 1,
        }
        for idx in range(min(len(urls), int(limit)))
        if idx < len(urls) and _clean_url(urls[idx])
    ]
    _lookup_cache_set("wikipedia_results", key, results)
    return results


# Returns cached or live search results for one query and offset.
def search_results(query: object, limit: int = SCRAPE_DEFAULT_LIMIT, offset: int = 0) -> list[dict[str, object]]:
    cleaned_query = clean_search_query(query)
    if not cleaned_query:
        return []
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    provider_limit = max(1, limit + offset)
    provider = selected_provider()
    if provider == "serper":
        results = _serper_results(cleaned_query, provider_limit)
    elif provider == "wikipedia":
        results = _wikipedia_results(cleaned_query, provider_limit)
    else:
        results = []
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for item in results:
        url = _clean_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append({**item, "url": url})
    return unique[offset : offset + limit]


# Fetch raw HTML for one URL.
def _fetch_url_html(url: object) -> str:
    clean_target = _clean_url(url)
    if not clean_target:
        return ""
    response = requests.get(clean_target, headers=HEADERS, timeout=SERPER_PAGE_TIMEOUT)
    response.raise_for_status()
    return response.text


# Read one cached document.
def _cached_document(conn: sqlite3.Connection, query: str, provider: str, search_source: str, url: str) -> dict[str, object] | None:
    row = conn.execute(
        """
        SELECT source_rank, status, title, metadata_json, raw_html, body_text, content_hash, error, fetched_at
        FROM research_documents
        WHERE query=? AND provider=? AND search_source=? AND url=?
        """,
        (query, provider, search_source, url),
    ).fetchone()
    if row is None:
        return None
    return {
        "query": query,
        "provider": provider,
        "search_source": search_source,
        "url": url,
        "rank": int(row[0]),
        "status": str(row[1]),
        "cached": True,
        "title": str(row[2]),
        "metadata": _json_object(row[3]),
        "raw_html": str(row[4]),
        "body_text": str(row[5]),
        "content_hash": str(row[6]),
        "error": str(row[7]),
        "timestamp": float(row[8]),
    }


# Store one fetched document in one cache transaction.
def _store_document(conn: sqlite3.Connection, document: dict[str, object]) -> None:
    query = str(document.get("query", ""))
    provider = str(document.get("provider", ""))
    search_source = str(document.get("search_source", "") or _document_search_source(document))
    url = str(document.get("url", ""))
    if not query or not provider or not search_source or not url:
        return
    metadata_json = json.dumps(document.get("metadata", {}), sort_keys=True)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO research_documents(
                query, provider, search_source, url, source_rank, status, title, metadata_json, raw_html, body_text,
                content_hash, error, fetched_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query,
                provider,
                search_source,
                url,
                int(document.get("rank", 0) or 0),
                str(document.get("status", "")),
                str(document.get("title", "")),
                metadata_json,
                str(document.get("raw_html", "")),
                str(document.get("body_text", "")),
                str(document.get("content_hash", "")),
                str(document.get("error", "")),
                float(document.get("timestamp", time.time()) or time.time()),
            ),
        )


# Run a complete research pass for the query.
def research_query(query: object, *, limit: int = SCRAPE_DEFAULT_LIMIT, offset: int = 0) -> dict[str, object]:
    cleaned_query = clean_search_query(query)
    provider = selected_provider()
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    if not cleaned_query:
        return _research_result("", provider, [], limit, offset, searched_urls=0)
    results = search_results(cleaned_query, limit=limit, offset=offset)
    documents: list[dict[str, object]] = []
    conn = _cache_connection()
    try:
        for index, result in enumerate(results, start=offset + 1):
            url = _clean_url(result.get("url", ""))
            if not url:
                continue
            search_source = str(result.get("search_source", "google_search") or "google_search")
            cached = _cached_document(conn, cleaned_query, provider, search_source, url)
            if cached is not None and not _cached_document_expired(cached):
                documents.append(_document_with_search_rank(cached, result, index))
                continue
            document = _build_research_document(cleaned_query, provider, result, index)
            if cached is not None:
                document = _revalidated_document(document, cached)
            _store_document(conn, document)
            documents.append(document)
    finally:
        conn.close()
    return _research_result(cleaned_query, provider, documents, limit, offset, searched_urls=len(results))


# Build one research document from a search row: fetch, extract metadata and body.
def _build_research_document(query: str, provider: str, result: dict[str, object], rank: int) -> dict[str, object]:
    url = _clean_url(result.get("url", ""))
    title_hint = collapse_whitespace(result.get("title", ""))
    search_metadata = _search_result_metadata(result)
    search_source = str(search_metadata.get("search_source", "google_search"))
    timestamp = time.time()
    try:
        raw_html = _fetch_url_html(url)
    except Exception as exc:
        return _document_error(query, provider, url, title_hint, rank, "fetch_error", str(exc), timestamp, search_metadata)
    metadata, body_text = _extract_page_content(raw_html, url=url, title_hint=title_hint)
    metadata = {**metadata, **search_metadata}
    if not body_text or len(body_text) < MIN_EXTRACTABLE_BLOCK_CHARS:
        return {
            "query": query,
            "provider": provider,
            "search_source": search_source,
            "url": url,
            "rank": int(rank),
            "status": "empty",
            "cached": False,
            "title": str(metadata.get("title") or title_hint),
            "metadata": metadata,
            "raw_html": raw_html,
            "body_text": body_text,
            "content_hash": _content_hash(body_text or raw_html),
            "error": "no extractable body text",
            "timestamp": timestamp,
        }
    return {
        "query": query,
        "provider": provider,
        "search_source": search_source,
        "url": url,
        "rank": int(rank),
        "status": "fetched",
        "cached": False,
        "title": str(metadata.get("title") or title_hint),
        "metadata": metadata,
        "raw_html": raw_html,
        "body_text": body_text,
        "content_hash": _content_hash(body_text),
        "error": "",
        "timestamp": timestamp,
    }


# Build a document row for recoverable page-level fetch/extraction failures.
def _document_error(query: str, provider: str, url: str, title: str, rank: int, status: str,
                    error: str, timestamp: float, metadata: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "query": query,
        "provider": provider,
        "search_source": str((metadata or {}).get("search_source", "google_search") or "google_search"),
        "url": url,
        "rank": int(rank),
        "status": status,
        "cached": False,
        "title": title,
        "metadata": {"title": title, **(metadata or {})},
        "raw_html": "",
        "body_text": "",
        "content_hash": "",
        "error": error,
        "timestamp": timestamp,
    }


# Keep current search rank/title beside cached body content.
def _document_with_search_rank(document: dict[str, object], result: dict[str, object], rank: int) -> dict[str, object]:
    title = str(document.get("title") or collapse_whitespace(result.get("title", "")))
    metadata = dict(document.get("metadata", {}) if isinstance(document.get("metadata"), dict) else {})
    metadata.setdefault("title", title)
    metadata.update(_search_result_metadata(result))
    return {**document, "rank": int(rank), "title": title, "metadata": metadata, "search_source": metadata.get("search_source", "google_search"), "cached": True}


# Return whether a terminal cached document needs TTL-based refetch/revalidation.
def _cached_document_expired(document: dict[str, object]) -> bool:
    ttl = float(RESEARCH_CACHE_TTL_SECONDS)
    if ttl <= 0.0:
        return False
    return (time.time() - float(document.get("timestamp", 0.0) or 0.0)) > ttl


# Annotate a freshly refetched document with previous-hash revalidation metadata.
def _revalidated_document(document: dict[str, object], cached: dict[str, object]) -> dict[str, object]:
    previous_hash = str(cached.get("content_hash", ""))
    current_hash = str(document.get("content_hash", ""))
    metadata = dict(document.get("metadata", {}) if isinstance(document.get("metadata"), dict) else {})
    metadata.update(
        {
            "revalidated": True,
            "previous_content_hash": previous_hash,
            "content_changed": bool(previous_hash and current_hash and previous_hash != current_hash),
        }
    )
    return {**document, "metadata": metadata, "cached": False}


# Normalize search-result metadata that should travel with every document.
def _search_result_metadata(result: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "search_source": str(result.get("search_source", "google_search") or "google_search"),
        "search_source_rank": int(result.get("source_rank", 0) or 0),
        "search_snippet": collapse_whitespace(result.get("snippet", "")),
    }
    if metadata["search_source"] == "google_scholar":
        metadata.update(
            {
                "publication_info": collapse_whitespace(result.get("publication_info", "")),
                "year": result.get("year", ""),
                "cited_by": result.get("cited_by", ""),
            }
        )
    return metadata


# Extract page metadata and body text from raw HTML.
def _extract_page_content(html: str, *, url: str, title_hint: str = "") -> tuple[dict[str, object], str]:
    soup = BeautifulSoup(html or "", "lxml")
    metadata = _metadata_from_soup(soup, url=url, title_hint=title_hint)
    _remove_junk(soup)
    extracted = trafilatura.extract(
        str(soup),
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    body_text = collapse_whitespace(_clean_body_text(extracted)) or _clean_body_text(_region_text(soup))
    return metadata, body_text


# Extract durable source metadata before script/style cleanup removes JSON-LD.
def _metadata_from_soup(soup: BeautifulSoup, *, url: str, title_hint: str = "") -> dict[str, object]:
    title = collapse_whitespace(soup.title.get_text(" ", strip=True) if soup.title else title_hint)
    description = _meta_content(soup, name="description")
    canonical = ""
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    if canonical_tag is not None:
        canonical = _clean_url(canonical_tag.get("href", ""))
    opengraph = {
        str(tag.get("property", "")).replace("og:", ""): collapse_whitespace(tag.get("content", ""))
        for tag in soup.find_all("meta")
        if str(tag.get("property", "")).startswith("og:") and collapse_whitespace(tag.get("content", ""))
    }
    headings = [
        collapse_whitespace(heading.get_text(" ", strip=True))
        for heading in soup.find_all(["h1", "h2", "h3", "h4"])
        if collapse_whitespace(heading.get_text(" ", strip=True))
    ][:24]
    json_ld = [_json_ld(script.get_text(" ", strip=True)) for script in soup.find_all("script", type="application/ld+json")]
    json_ld = [item for item in json_ld if item]
    author = _meta_content(soup, name="author") or _meta_content(soup, property_name="article:author")
    published = (
        _meta_content(soup, property_name="article:published_time")
        or _meta_content(soup, name="date")
        or _meta_content(soup, name="pubdate")
    )
    return {
        "title": title,
        "description": description,
        "canonical_url": canonical or _clean_url(url),
        "opengraph": opengraph,
        "headings": headings,
        "json_ld": json_ld,
        "author": author,
        "date": published,
    }


# Return a meta content field by name or property.
def _meta_content(soup: BeautifulSoup, *, name: str = "", property_name: str = "") -> str:
    tag = None
    if name:
        tag = soup.find("meta", attrs={"name": name})
    if tag is None and property_name:
        tag = soup.find("meta", attrs={"property": property_name})
    return collapse_whitespace(tag.get("content", "")) if tag is not None else ""


# Parse one JSON-LD script into structured metadata when it is valid JSON.
def _json_ld(text: object) -> object:
    value = collapse_whitespace(text)
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value[:1200]


_FOOTNOTE_INLINE_RE = re.compile(r"\[\d+[\]\)][^\[]*")
_BIBLIOGRAPHY_SECTION_RE = re.compile(
    r"^[ \t]*(References|Bibliography|Notes|See also|Further reading|External links|Footnotes|Citations|Sources)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# Strip inline footnote markers and truncate at bibliography sections.
def _clean_body_text(text: str) -> str:
    if not text:
        return text
    match = _BIBLIOGRAPHY_SECTION_RE.search(text)
    if match:
        text = text[: match.start()].rstrip()
    text = _FOOTNOTE_INLINE_RE.sub("", text)
    return text


# Remove page chrome before body extraction.
def _remove_junk(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "template", "svg", "canvas", "iframe", "form"]):
        tag.decompose()
    selectors = [
        "nav", "footer", "aside", "header",
        "[role='navigation']", "[role='banner']", "[role='contentinfo']",
        ".nav", ".navbar", ".footer", ".sidebar", ".menu",
        ".cookie", ".advertisement", ".ads",
    ]
    for selector in selectors:
        for tag in soup.select(selector):
            tag.decompose()


# Read visible text from the most article-like region left after cleanup.
def _region_text(soup: BeautifulSoup) -> str:
    region = soup.find("article") or soup.find("main") or soup.body or soup
    return collapse_whitespace(region.get_text(" ", strip=True))


# Build public research result counts from document status.
def _research_result(query: str, provider: str, documents: list[dict[str, object]], limit: int,
                     offset: int, *, searched_urls: int) -> dict[str, object]:
    fetched = [doc for doc in documents if doc.get("status") == "fetched"]
    return {
        "query": query,
        "provider": provider,
        "documents": documents,
        "limit": int(limit),
        "offset": int(offset),
        "counts": {
            "searched_urls": int(searched_urls),
            "google_search_urls": sum(1 for doc in documents if _document_search_source(doc) == "google_search"),
            "scholar_urls": sum(1 for doc in documents if _document_search_source(doc) == "google_scholar"),
            "fetched_pages": len(fetched),
            "cached_pages": sum(1 for doc in fetched if doc.get("cached")),
            "extracted_bodies": sum(1 for doc in fetched if str(doc.get("body_text", "")).strip()),
            "fetch_errors": sum(1 for doc in documents if str(doc.get("status", "")).endswith("_error")),
        },
    }


# Return the explicit search source for a document, defaulting to Google Search.
def _document_search_source(document: dict[str, object]) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    return str(document.get("search_source", "") or metadata.get("search_source", "google_search") or "google_search")


# Return a stable hash of the extracted page truth.
def _content_hash(text: object) -> str:
    return hashlib.blake2b(str(text or "").encode("utf-8"), digest_size=16).hexdigest()


# Parse JSON objects from cache rows without trusting old cache content.
def _json_object(value: object) -> dict[str, object]:
    try:
        decoded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
