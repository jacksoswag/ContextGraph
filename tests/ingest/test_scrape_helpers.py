from __future__ import annotations
# L0 offline tests for pure helper functions in ingest/labels.py and ingest/scrape_worker.py.
# All tests are network-free; no DB or LLM dependencies.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from ingest.labels import (
    clean_search_query,
    looks_like_non_natural_label,
    collapse_whitespace,
    normalize_label,
    sentence_chunks,
    clean_label_phrase,
    natural_label_phrase,
)


# ---------------------------------------------------------------------------
# clean_search_query
# ---------------------------------------------------------------------------

def test_clean_search_query_strips_list_prefix():
    assert clean_search_query("- What is photosynthesis?") == "What is photosynthesis?"


def test_clean_search_query_strips_numbered_prefix():
    assert clean_search_query("1. How do birds navigate?") == "How do birds navigate?"


def test_clean_search_query_strips_query_prefix():
    assert clean_search_query("query: deep learning") == "deep learning"


def test_clean_search_query_strips_search_prefix():
    assert clean_search_query("search: quantum physics") == "quantum physics"


def test_clean_search_query_strips_surrounding_quotes():
    assert clean_search_query('"climate change effects"') == "climate change effects"


def test_clean_search_query_preserves_normal_text():
    text = "How do vaccines work?"
    assert clean_search_query(text) == text


def test_clean_search_query_truncates_at_max_length():
    long_query = "x" * 300
    result = clean_search_query(long_query)
    assert len(result) <= 180


def test_clean_search_query_collapses_whitespace():
    result = clean_search_query("  hello   world  ")
    assert result == "hello world"


# ---------------------------------------------------------------------------
# looks_like_non_natural_label
# ---------------------------------------------------------------------------

def test_looks_like_non_natural_label_rejects_url():
    assert looks_like_non_natural_label("https://example.com")


def test_looks_like_non_natural_label_rejects_www():
    assert looks_like_non_natural_label("www.google.com")


def test_looks_like_non_natural_label_rejects_code_token():
    assert looks_like_non_natural_label("__init__")


def test_looks_like_non_natural_label_rejects_long_hex():
    assert looks_like_non_natural_label("deadbeef12345678")


def test_looks_like_non_natural_label_rejects_all_digits():
    assert looks_like_non_natural_label("12345")


def test_looks_like_non_natural_label_rejects_json_bracket():
    assert looks_like_non_natural_label("{key: value}")


def test_looks_like_non_natural_label_accepts_normal_phrase():
    assert not looks_like_non_natural_label("machine learning")


def test_looks_like_non_natural_label_accepts_hyphenated():
    assert not looks_like_non_natural_label("well-known")


def test_looks_like_non_natural_label_accepts_apostrophe():
    assert not looks_like_non_natural_label("it's")


def test_looks_like_non_natural_label_rejects_empty():
    assert looks_like_non_natural_label("")


# ---------------------------------------------------------------------------
# collapse_whitespace
# ---------------------------------------------------------------------------

def test_collapse_whitespace_collapses_tabs_and_spaces():
    assert collapse_whitespace("hello\t\t world") == "hello world"


def test_collapse_whitespace_strips_leading_trailing():
    assert collapse_whitespace("  hi  ") == "hi"


def test_collapse_whitespace_handles_none():
    assert collapse_whitespace(None) == ""


def test_collapse_whitespace_handles_newlines():
    assert collapse_whitespace("line1\nline2") == "line1 line2"


# ---------------------------------------------------------------------------
# normalize_label
# ---------------------------------------------------------------------------

def test_normalize_label_lowercases():
    assert normalize_label("Apple") == "apple"


def test_normalize_label_strips_punctuation_ends():
    assert normalize_label("apple.") == "apple"


def test_normalize_label_collapses_whitespace():
    assert normalize_label("  two  words  ") == "two words"


# ---------------------------------------------------------------------------
# sentence_chunks
# ---------------------------------------------------------------------------

def test_sentence_chunks_splits_on_period_space():
    text = "Bees pollinate flowers. Flowers produce fruit. Fruit feeds animals."
    chunks = sentence_chunks(text, limit=10)
    assert len(chunks) >= 2
    assert any("pollinate" in c for c in chunks)


def test_sentence_chunks_respects_limit():
    text = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
    chunks = sentence_chunks(text, limit=5)
    assert len(chunks) <= 5


def test_sentence_chunks_drops_short_sentences():
    text = "Hi. The quick brown fox jumps over the lazy dog."
    chunks = sentence_chunks(text, limit=10, min_length=8)
    for c in chunks:
        assert len(c) >= 8


def test_sentence_chunks_handles_empty():
    assert sentence_chunks("", limit=5) == ()


def test_sentence_chunks_truncates_long_sentence():
    long_sentence = "word " * 300
    chunks = sentence_chunks(long_sentence, limit=1, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)


def test_sentence_chunks_truncates_at_word_boundary():
    # "abcde " * 20 = 120 chars; truncating at 100 mid-word would produce a fragment.
    long_sentence = "abcde " * 20
    chunks = sentence_chunks(long_sentence, limit=1, max_chars=97)
    assert chunks
    # Result must end at a word boundary (no partial token)
    assert not chunks[0].endswith("abcd"), "should not truncate mid-word"
    assert chunks[0].endswith("abcde"), "should end at a complete word"
    assert len(chunks[0]) <= 97


# ---------------------------------------------------------------------------
# clean_label_phrase / natural_label_phrase
# ---------------------------------------------------------------------------

def test_clean_label_phrase_rejects_url():
    assert clean_label_phrase("https://example.com", max_length=80) == ""


def test_clean_label_phrase_rejects_too_long():
    assert clean_label_phrase("a" * 100, max_length=50) == ""


def test_clean_label_phrase_normalizes_and_returns():
    result = clean_label_phrase("Machine Learning", max_length=80)
    assert result == "machine learning"


def test_natural_label_phrase_accepts_short_clean():
    assert natural_label_phrase("deep learning")


def test_natural_label_phrase_rejects_long():
    assert not natural_label_phrase("this is way too many words in a row here")


def test_natural_label_phrase_rejects_url():
    assert not natural_label_phrase("http://foo.com")
