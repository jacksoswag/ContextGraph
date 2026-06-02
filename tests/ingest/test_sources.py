from __future__ import annotations
# P2 corpus-fetcher tests: Wikipedia + Semantic Scholar JSON-API parsing, caching
# (fetch once, replay), and that the network seam is hit only on cache miss. No network.
import sys
from pathlib import Path
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from ingest import sources


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_CACHE_PATH", str(tmp_path / "corpus.sqlite"))
    monkeypatch.setattr(sources, "_MIN_INTERVAL", {})   # no real sleeps in tests
    return tmp_path


def test_wikipedia_parses_extract_and_caches(cache, monkeypatch):
    calls = []
    def fake(url, params):
        calls.append((url, params))
        return {"query": {"pages": {"123": {"title": "Radium", "extract": "Radium is a chemical element."}}}}
    monkeypatch.setattr(sources, "_http_get_json", fake)
    t1 = sources.wikipedia_article("Radium")
    t2 = sources.wikipedia_article("Radium")        # served from cache
    assert "chemical element" in t1 and t1 == t2
    assert len(calls) == 1                           # one network hit, then cached
    assert calls[0][1]["explaintext"] == "1" and calls[0][1]["action"] == "query"


def test_semantic_scholar_search_filters_abstractless_and_caches(cache, monkeypatch):
    calls = []
    def fake(url, params):
        calls.append(url)
        return {"data": [
            {"paperId": "p1", "title": "Attention", "abstract": "We propose the Transformer.", "year": 2017},
            {"paperId": "p2", "title": "No abstract here", "abstract": None, "year": 2020},
        ]}
    monkeypatch.setattr(sources, "_http_get_json", fake)
    out = sources.semantic_scholar_search("transformer", limit=5)
    again = sources.semantic_scholar_search("transformer", limit=5)
    assert [p["paperId"] for p in out] == ["p1"]      # abstract-less paper dropped
    assert out[0]["abstract"].startswith("We propose")
    assert out == again and len(calls) == 1           # cached on second call


def test_semantic_scholar_paper_by_id(cache, monkeypatch):
    monkeypatch.setattr(sources, "_http_get_json",
                        lambda url, params: {"paperId": "abc", "title": "T", "abstract": "An abstract."})
    p = sources.semantic_scholar_paper("abc")
    assert p["abstract"] == "An abstract." and p["paperId"] == "abc"
