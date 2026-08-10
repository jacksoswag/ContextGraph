"""Topic-driven article fetching.

Given a list of topics and a per-source article cap, fetch articles from any of:
Wikipedia, arXiv (no key needed), and Google web search / Google Scholar (via a
serper.dev key in SERPER_API_KEY). Every article carries its source, title, link,
plain text, and a publish/updated date when the source exposes one.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "http://export.arxiv.org/api/query"
SERPER_SEARCH = "https://google.serper.dev/search"
SERPER_SCHOLAR = "https://google.serper.dev/scholar"
HEADERS = {"User-Agent": "di-extractor/1.0 (knowledge graph builder)"}
TIMEOUT = 20
MAX_CHARS = 20000


@dataclass
class Article:
    source: str
    topic: str
    title: str
    url: str
    text: str
    published: str = ""


def _clip(text: str) -> str:
    return " ".join(str(text or "").split())[:MAX_CHARS]


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
def _wikipedia(topic: str, count: int) -> list[Article]:
    search = requests.get(
        WIKI_API,
        params={
            "action": "query", "list": "search", "srsearch": topic,
            "srlimit": count, "format": "json",
        },
        headers=HEADERS, timeout=TIMEOUT,
    ).json()
    pageids = [str(hit["pageid"]) for hit in search.get("query", {}).get("search", [])]
    if not pageids:
        return []
    pages = requests.get(
        WIKI_API,
        params={
            "action": "query", "prop": "extracts|info|revisions",
            "explaintext": 1, "exlimit": "max", "inprop": "url",
            "rvprop": "timestamp", "pageids": "|".join(pageids), "format": "json",
        },
        headers=HEADERS, timeout=TIMEOUT,
    ).json().get("query", {}).get("pages", {})
    articles = []
    for page in pages.values():
        text = _clip(page.get("extract", ""))
        if not text:
            continue
        revisions = page.get("revisions") or [{}]
        articles.append(Article(
            source="wikipedia", topic=topic, title=page.get("title", ""),
            url=page.get("canonicalurl", ""), text=text,
            published=revisions[0].get("timestamp", ""),
        ))
    return articles


# --------------------------------------------------------------------------- #
# arXiv
# --------------------------------------------------------------------------- #
def _arxiv(topic: str, count: int) -> list[Article]:
    response = requests.get(
        ARXIV_API,
        params={"search_query": f"all:{topic}", "start": 0, "max_results": count},
        headers=HEADERS, timeout=TIMEOUT,
    )
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(response.text)
    articles = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        if not summary:
            continue
        articles.append(Article(
            source="arxiv", topic=topic, title=title, url=url,
            text=_clip(f"{title}. {summary}"), published=published,
        ))
    return articles


# --------------------------------------------------------------------------- #
# serper.dev (Google web search + Google Scholar)
# --------------------------------------------------------------------------- #
def _serper(endpoint: str, topic: str, count: int):
    key = os.getenv("SERPER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPER_API_KEY not set (needed for --google/--scholar)")
    response = requests.post(
        endpoint,
        headers={**HEADERS, "X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": topic, "num": count}, timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("organic", []) or []


def _fulltext(url: str, fallback: str) -> str:
    try:
        import trafilatura

        downloaded = requests.get(url, headers=HEADERS, timeout=TIMEOUT).text
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if extracted:
            return _clip(extracted)
    except Exception:
        pass
    return _clip(fallback)


def _google(topic: str, count: int) -> list[Article]:
    articles = []
    for item in _serper(SERPER_SEARCH, topic, count):
        url = item.get("link", "")
        snippet = item.get("snippet", "")
        if not url:
            continue
        articles.append(Article(
            source="google", topic=topic, title=item.get("title", ""), url=url,
            text=_fulltext(url, snippet), published=str(item.get("date", "")),
        ))
    return articles


def _scholar(topic: str, count: int) -> list[Article]:
    articles = []
    for item in _serper(SERPER_SCHOLAR, topic, count):
        url = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        if not (url or title):
            continue
        articles.append(Article(
            source="scholar", topic=topic, title=title, url=url,
            text=_clip(f"{title}. {snippet}"), published=str(item.get("year", "")),
        ))
    return articles


_FETCHERS = {
    "wikipedia": _wikipedia,
    "arxiv": _arxiv,
    "google": _google,
    "scholar": _scholar,
}


def fetch(topics, count, wikipedia=True, arxiv=True, google=False, scholar=False):
    """Return a flat list of Articles across all topics and enabled sources."""
    enabled = [
        name for name, on in (
            ("wikipedia", wikipedia), ("arxiv", arxiv),
            ("google", google), ("scholar", scholar),
        ) if on
    ]
    articles = []
    for topic in topics:
        for name in enabled:
            try:
                found = _FETCHERS[name](topic, count)
            except Exception as exc:
                print(f"[warn] {name} fetch failed for '{topic}': {exc}")
                continue
            print(f"[fetch] {name}: {len(found)} articles for '{topic}'")
            articles.extend(found)
    return articles
