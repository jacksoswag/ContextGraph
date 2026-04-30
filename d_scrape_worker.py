import json
import os
import sqlite3
import time
from pathlib import Path
from queue import Empty
from urllib.parse import urlparse

import requests
import trafilatura
from dotenv import load_dotenv

from constants import (
    MAX_CHARS_PER_BLOCK,
    MIN_EXTRACTABLE_BLOCK_CHARS,
    SCRAPE_DEFAULT_LIMIT,
    SERPER_PAGE_TIMEOUT,
    SERPER_RESULT_TIMEOUT,
)
from d_noise_cleanup import clean_text

load_dotenv(Path(__file__).resolve().parent / ".env")

WIKIPEDIA_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
SERPER_URL = "https://google.serper.dev/search"
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "scrape_cache.sqlite3"
HEADERS = {
    "User-Agent": "brain-research/1.0 (+https://localhost)",
}


def _clean_query(query):
    return clean_text(query)


def _clean_url(url):
    url = str(url or "").strip().strip("\"'<>[]()")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _result_tag(title="", url=""):
    title = clean_text(title)
    url = _clean_url(url)
    return f"{title}|{url}" if title and url else url or title or "unknown"


def _cache_connection():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scrape_cache (
            namespace TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            value TEXT NOT NULL,
            created REAL NOT NULL,
            PRIMARY KEY(namespace, cache_key)
        )
        """
    )
    return conn


def _cache_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _cache_get(namespace, key):
    conn = None
    try:
        conn = _cache_connection()
        row = conn.execute(
            "SELECT value FROM scrape_cache WHERE namespace=? AND cache_key=?",
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


def _cache_set(namespace, key, value):
    conn = None
    try:
        encoded = json.dumps(value)
        conn = _cache_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scrape_cache(namespace, cache_key, value, created)
                VALUES(?, ?, ?, ?)
                """,
                (namespace, key, encoded, time.time()),
            )
    except (TypeError, sqlite3.Error):
        return
    finally:
        if conn is not None:
            conn.close()


def _serper_page(query, page, page_size):
    key = _cache_key({"query": query, "page": int(page), "page_size": int(page_size)})
    cached = _cache_get("serper_page", key)
    if cached is not None:
        return cached

    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        return []
    response = requests.post(
        SERPER_URL,
        headers={**HEADERS, "X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": int(page_size), "page": int(page)},
        timeout=SERPER_RESULT_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    organic = payload.get("organic", []) or []
    results = []
    for item in organic:
        url = _clean_url(item.get("link", ""))
        title = clean_text(item.get("title", ""))
        snippet = clean_text(item.get("snippet", ""))
        if url:
            results.append({"title": title, "url": url, "snippet": snippet})
    _cache_set("serper_page", key, results)
    return results


def _serper_results(query, limit):
    results = []
    page_size = 10
    page_count = (max(1, int(limit)) + page_size - 1) // page_size
    for page in range(1, page_count + 1):
        page_results = _serper_page(query, page, page_size)
        if not page_results:
            break
        results.extend(page_results)
        if len(results) >= int(limit):
            break
    return results[:limit]


def _wikipedia_results(query, limit):
    key = _cache_key({"query": query, "limit": int(limit)})
    cached = _cache_get("wikipedia_results", key)
    if cached is not None:
        return cached

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
            "title": titles[idx] if idx < len(titles) else "",
            "snippet": snippets[idx] if idx < len(snippets) else "",
            "url": urls[idx] if idx < len(urls) else "",
        }
        for idx in range(min(len(urls), int(limit)))
        if idx < len(urls) and urls[idx]
    ]
    _cache_set("wikipedia_results", key, results)
    return results


def search_results(query, limit=SCRAPE_DEFAULT_LIMIT, offset=0):
    query = _clean_query(query)
    if not query:
        return []
    limit = max(0, int(limit))
    offset = max(0, int(offset))
    provider_limit = max(1, limit + offset)
    results = []
    errors = []
    for provider in (_serper_results, _wikipedia_results):
        try:
            results.extend(provider(query, provider_limit))
        except Exception as exc:
            errors.append(str(exc))
        if len(results) >= provider_limit:
            break
    seen = set()
    unique = []
    for item in results:
        url = _clean_url(item.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    if not unique and errors:
        raise RuntimeError("; ".join(errors[:2]))
    return unique[offset : offset + limit]


def _fetch_url_text(url):
    url = _clean_url(url)
    if not url:
        return ""
    cached = _cache_get("page_text", url)
    if cached is not None:
        return str(cached or "")

    response = requests.get(url, headers=HEADERS, timeout=SERPER_PAGE_TIMEOUT)
    response.raise_for_status()
    downloaded = response.text
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    text = clean_text(extracted)
    _cache_set("page_text", url, text)
    return text


def _snippet_block(result):
    snippet = clean_text(result.get("snippet", ""))
    if not snippet:
        return None
    return {
        "content": snippet[:MAX_CHARS_PER_BLOCK],
        "tag": _result_tag(result.get("title", ""), result.get("url", "")),
        "url": _clean_url(result.get("url", "")),
    }


def _block_fingerprint(block):
    url = _clean_url((block or {}).get("url", ""))
    content = clean_text((block or {}).get("content", "")).lower()
    return (url, content[:512]) if url else ("content", content[:512])


def blocks_for_query(query, limit=SCRAPE_DEFAULT_LIMIT, offset=0, prefer_snippets=False):
    blocks = []
    seen_blocks = set()
    for result in search_results(query, limit=limit, offset=offset):
        url = _clean_url(result.get("url", ""))
        title = clean_text(result.get("title", ""))
        if prefer_snippets:
            block = _snippet_block(result)
            if block and len(block["content"]) >= MIN_EXTRACTABLE_BLOCK_CHARS:
                fingerprint = _block_fingerprint(block)
                if fingerprint not in seen_blocks:
                    seen_blocks.add(fingerprint)
                    blocks.append(block)
                if len(blocks) >= limit:
                    break
                continue
        try:
            text = _fetch_url_text(url)
        except Exception:
            text = ""
        if len(text) >= MIN_EXTRACTABLE_BLOCK_CHARS:
            block = {
                "content": text[:MAX_CHARS_PER_BLOCK],
                "tag": _result_tag(title, url),
                "url": url,
            }
        else:
            block = _snippet_block(result)
        if block:
            fingerprint = _block_fingerprint(block)
            if fingerprint in seen_blocks:
                continue
            seen_blocks.add(fingerprint)
            blocks.append(block)
        if len(blocks) >= limit:
            break
    return blocks


def scrape_worker_loop(scrape_queue, extract_queue, stop_event, sync_counter):
    while not stop_event.is_set():
        try:
            task = scrape_queue.get(timeout=0.2)
        except Empty:
            continue
        query = _clean_query(task.get("query", "") if isinstance(task, dict) else task)
        limit = task.get("limit", SCRAPE_DEFAULT_LIMIT) if isinstance(task, dict) else SCRAPE_DEFAULT_LIMIT
        offset = task.get("offset", 0) if isinstance(task, dict) else 0
        research_pass = task.get("research_pass", "primary") if isinstance(task, dict) else "primary"
        try:
            blocks = blocks_for_query(
                query,
                limit=limit,
                offset=offset,
                prefer_snippets=research_pass == "refine_existing",
            )
            extract_queue.put(
                {
                    "query": query,
                    "blocks": blocks,
                    "connections": [],
                    "research_pass": research_pass,
                    "limit": int(limit),
                    "offset": int(offset),
                }
            )
        except Exception as exc:
            extract_queue.put(
                {
                    "query": query,
                    "blocks": [],
                    "connections": [],
                    "error": str(exc),
                    "research_pass": research_pass,
                    "limit": int(limit),
                    "offset": int(offset),
                }
            )
        finally:
            with sync_counter.get_lock():
                sync_counter.value = max(0, sync_counter.value - 1)
            time.sleep(0.05)
