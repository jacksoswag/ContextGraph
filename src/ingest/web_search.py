# L2 cache: lightweight web search used by the Dynamic research trigger.
# Yields (source_label, url, body) 3-tuples; url is the page URL for M1 atom granularity.
# Prefers DI_WEB_SEARCH_HOOK env override, then Serper API. Failures yield nothing.
from __future__ import annotations

import importlib
import logging
import os
import urllib.parse
import urllib.request
from typing import Callable, Iterator

LOGGER = logging.getLogger(__name__)
HOOK_SOURCE_LABEL = "hook"


def _load_hook() -> Callable[[str], object] | None:
    spec = os.getenv("DI_WEB_SEARCH_HOOK", "").strip()
    if not spec or ":" not in spec:
        return None
    mod_name, _, attr = spec.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
    except (ImportError, AttributeError) as exc:
        LOGGER.warning("DI_WEB_SEARCH_HOOK %r unusable: %s", spec, exc)
        return None
    return fn if callable(fn) else None

def _domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.replace("www.", "")


# Fetch one search result; returns (domain, url, body) or None. Thread-safe via per-call DB connection.
def _fetch_article(
    query: str, result: dict, index: int, provider: str
) -> tuple[str, str, str] | None:
    from ingest.scrape_worker import (
        _clean_url,
        _cache_connection,
        _cached_document,
        _cached_document_expired,
        _document_with_search_rank,
        _build_research_document,
        _revalidated_document,
        _store_document,
    )

    url = _clean_url(result.get("url", ""))
    if not url:
        return None
    search_source = str(result.get("search_source", "google_search") or "google_search")
    conn = _cache_connection()
    try:
        cached = _cached_document(conn, query, provider, search_source, url)
        if cached is not None and not _cached_document_expired(cached):
            doc = _document_with_search_rank(cached, result, index)
        else:
            doc = _build_research_document(query, provider, result, index)
            if cached is not None:
                doc = _revalidated_document(doc, cached)
            _store_document(conn, doc)
    finally:
        conn.close()

    body = str(doc.get("body_text", ""))
    status = str(doc.get("status", ""))
    if status == "fetched" and body and url:
        return (_domain_of(url), url, body)
    return None


def search_result_list(query: str) -> tuple[str, list[dict]]:
    """Return (provider, results) without fetching page bodies."""
    from ingest.scrape_worker import search_results as _sr, selected_provider as _sp

    query = (query or "").strip()
    provider = _sp()
    results = _sr(query, limit=16) if query else []
    return provider, results


from concurrent.futures import ThreadPoolExecutor, as_completed

_scrape_pool = ThreadPoolExecutor(max_workers=8)

def search_stream(query: str) -> Iterator[tuple[str, str, str]]:
    # Yield (source_label, url, body) concurrently for articles matching query.
    # url is the full page URL, used by the ingest pipeline as the provenance
    # atom source_url for per-page independence scoring (M1).
    query = (query or "").strip()
    if not query:
        return

    hook = _load_hook()
    if hook is not None:
        try:
            out = hook(query)
        except Exception as exc:
            LOGGER.warning("web search hook raised: %s", exc)
            return
        if not out:
            return
        if isinstance(out, str):
            yield (HOOK_SOURCE_LABEL, out)
            return
        for item in out:
            yield (HOOK_SOURCE_LABEL, str(item))
        return

    try:
        provider, results = search_result_list(query)
        # Cap to top 3 hits for speed/relevance
        pending = {
            _scrape_pool.submit(_fetch_article, query, r, i, provider): i
            for i, r in enumerate(results[:3], start=1)
        }
        for future in as_completed(pending):
            try:
                item = future.result()
                if item is not None:
                    yield item
            except Exception:
                pass
    except Exception as exc:
        LOGGER.warning("Failed to search %r: %s", query, exc)
