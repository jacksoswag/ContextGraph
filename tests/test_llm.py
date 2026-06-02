from __future__ import annotations
# call_json: JSON-seam caching (spec §4 — deterministic + cached given inputs).
import pytest
import llm as llm_mod
from llm import call_json

@pytest.fixture(autouse=True)
def _clear_cache():
    llm_mod._JSON_CACHE.clear(); yield; llm_mod._JSON_CACHE.clear()

def _count_backend(monkeypatch):
    n = {"calls": 0}
    def fake(prompt, tag, options, role, event_callback):
        n["calls"] += 1
        return '{"ok": true, "n": %d}' % n["calls"]
    monkeypatch.setattr(llm_mod, "_ollama_generate", fake)
    monkeypatch.setattr(llm_mod, "_available", {llm_mod._MODEL_TAGS["3B"]})
    return n

def test_call_json_caches_identical_calls(monkeypatch):
    n = _count_backend(monkeypatch)
    a = call_json("same prompt", "3B")
    b = call_json("same prompt", "3B")
    assert a == b and n["calls"] == 1                 # second call served from cache

def test_call_json_distinct_prompts_miss(monkeypatch):
    n = _count_backend(monkeypatch)
    call_json("prompt A", "3B"); call_json("prompt B", "3B")
    assert n["calls"] == 2

def test_call_json_model_tier_is_part_of_key(monkeypatch):
    n = _count_backend(monkeypatch)
    call_json("p", "3B"); call_json("p", "1B")
    assert n["calls"] == 2                             # different tier ⇒ separate cache entry

def test_call_json_use_cache_false_bypasses(monkeypatch):
    n = _count_backend(monkeypatch)
    call_json("p", "3B", use_cache=False)
    call_json("p", "3B", use_cache=False)
    assert n["calls"] == 2

def test_call_json_does_not_cache_parse_failures(monkeypatch):
    n = {"calls": 0}
    def fake(prompt, tag, options, role, event_callback):
        n["calls"] += 1
        return "not json at all"
    monkeypatch.setattr(llm_mod, "_ollama_generate", fake)
    monkeypatch.setattr(llm_mod, "_available", {llm_mod._MODEL_TAGS["3B"]})
    assert call_json("p", "3B") == {} and call_json("p", "3B") == {}
    assert n["calls"] == 2                             # empty results are re-attempted, not cached
