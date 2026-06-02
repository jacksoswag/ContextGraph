from __future__ import annotations
# Producer-level tests for the deep-ingest stitch pass. No graph store:
#   1. _validate keeps well-formed cross-clause edges and rejects malformed ones
#      (self-loops, verb-as-text, unknown clause ids, low confidence).
#   2. extraction → triples: pronoun-subject handling with/without coref.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from ingest.deep_ingest import _validate, ingest_article, complete_pending_edges, resolve_dates


# ── _validate: the cross-clause stitch core (store-free) ──────────────────────

_INDEX = {
    "art:c0": ("france", "invade", "russia"),
    "art:c1": ("napoleon", "retreat", "moscow"),
}


def _rec(**kw):
    base = {"relation": "cause", "source_clause_id": "c0", "target_clause_id": "c1",
            "source_text": "france", "target_text": "moscow", "confidence": 0.8}
    base.update(kw)
    return base


def test_validate_keeps_wellformed_cross_clause_edge():
    kept = _validate([_rec()], _INDEX, model="test", article_hash="art")
    assert len(kept) == 1
    e = kept[0]
    assert e.relation == "cause"
    assert e.source_clause_id == "art:c0" and e.target_clause_id == "art:c1"
    assert e.verified is True


def test_validate_rejects_self_loop():
    assert _validate([_rec(target_clause_id="c0", target_text="russia")],
                     _INDEX, model="test", article_hash="art") == []


def test_validate_rejects_verb_as_text():
    # source_text copies the verb ("invade") instead of a noun phrase → rejected.
    assert _validate([_rec(source_text="invade")],
                     _INDEX, model="test", article_hash="art") == []


def test_validate_rejects_unknown_clause_id():
    assert _validate([_rec(source_clause_id="c9")],
                     _INDEX, model="test", article_hash="art") == []


def test_validate_rejects_low_confidence():
    assert _validate([_rec(confidence=0.2)],
                     _INDEX, model="test", article_hash="art") == []


# ── ingest_article: empty input short-circuits before any LLM call ────────────

def test_ingest_article_empty_when_no_triples():
    spacy = pytest.importorskip("spacy", reason="spaCy not installed")
    try: spacy.load("en_core_web_sm")
    except OSError: pytest.skip("en_core_web_sm not installed")
    result = ingest_article(None, "...")
    assert result.edges == [] and result.triples_seen == 0


# ── complete_pending_edges: in-memory pending-completion transform ────────────

def _pending(subj="army", verb="retreat"):
    return {"type": "edge", "rel": verb,
            "source": {"type": "node", "text": subj, "pos": "NOUN"},
            "target": {"type": "node", "text": "[event]", "pos": "X"},
            "_source_text": f"The {subj} {verb}ed.", "_clause_text": f"The {subj} {verb}ed",
            "_pending_completion": True}


def test_complete_pending_upgrades_target(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json",
                        lambda *a, **kw: {"completions": [{"id": 0, "object": "battlefield"}]})
    out = complete_pending_edges([_pending()])
    assert len(out) == 1
    assert out[0]["target"]["text"] == "battlefield"
    assert "_pending_completion" not in out[0]


def test_complete_pending_sweeps_uncompletable(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json",
                        lambda *a, **kw: {"completions": [{"id": 0, "object": None}]})
    assert complete_pending_edges([_pending()]) == []


def test_complete_pending_keeps_sentinel_when_not_discarding(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json",
                        lambda *a, **kw: {"completions": [{"id": 0, "object": None}]})
    out = complete_pending_edges([_pending()], discard_uncompleted=False)
    assert len(out) == 1 and out[0]["target"]["text"] == "[event]"


def test_complete_pending_passes_through_complete_edges(monkeypatch):
    import ingest.deep_ingest as dd
    called = []
    monkeypatch.setattr(dd, "call_json", lambda *a, **kw: called.append(1) or {"completions": []})
    edge = {"type": "edge", "rel": "eat",
            "source": {"type": "node", "text": "dog", "pos": "NOUN"},
            "target": {"type": "node", "text": "food", "pos": "NOUN"}}
    out = complete_pending_edges([edge])
    assert out == [edge] and not called  # no LLM call when nothing is pending


# ── resolve_dates: in-memory relative-time → ISO transform ────────────────────

def _dated(phrase, **extra):
    e = {"type": "edge", "rel": "sign",
         "source": {"type": "node", "text": "country", "pos": "NOUN"},
         "target": {"type": "node", "text": "treaty", "pos": "NOUN"},
         "time_phrase": phrase}
    e.update(extra)
    return e


def test_resolve_dates_annotates_iso(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json", lambda *a, **kw: {"resolved": [{"id": 0, "iso": "1949"}]})
    out = resolve_dates([_dated("in 1949")])
    assert out[0]["resolved_date"] == "1949"


def test_resolve_dates_marks_unresolvable_empty(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json", lambda *a, **kw: {"resolved": [{"id": 0, "iso": None}]})
    out = resolve_dates([_dated("recently")])
    assert out[0]["resolved_date"] == ""


def test_resolve_dates_skips_already_resolved(monkeypatch):
    import ingest.deep_ingest as dd
    called = []
    monkeypatch.setattr(dd, "call_json", lambda *a, **kw: called.append(1) or {"resolved": []})
    out = resolve_dates([_dated("in 1949", resolved_date="1949")])
    assert out[0]["resolved_date"] == "1949" and not called  # re-runnable, no LLM call


def test_resolve_dates_recurses_into_nested_targets(monkeypatch):
    import ingest.deep_ingest as dd
    monkeypatch.setattr(dd, "call_json",
                        lambda *a, **kw: {"resolved": [{"id": 0, "iso": "2001-09-11"}]})
    nested = _dated("September 11", rel="attack")
    parent = {"type": "edge", "rel": "say",
              "source": {"type": "node", "text": "report", "pos": "NOUN"}, "target": nested}
    resolve_dates([parent])
    assert nested["resolved_date"] == "2001-09-11"


# ── Pronoun handling at the extraction boundary ───────────────────────────────

def test_pronoun_subject_dropped_without_coref():
    # With COREF_ENABLED=0 (default), pronoun-subject clauses never reach
    # _extract_triples because _sent_to_clauses filters _VALUE_POS={NOUN,PROPN,NUM}.
    spacy = pytest.importorskip("spacy", reason="spaCy not installed")
    try: spacy.load("en_core_web_sm")
    except OSError: pytest.skip("en_core_web_sm not installed")

    import ingest.deep_ingest as dd
    body = "France invaded Russia. It suffered catastrophic losses."
    triples, _ = dd._extract_triples(body, "pronoun_test")
    subjects = [t[1].lower() for t in triples]
    assert "it" not in subjects, f"pronoun 'it' must be dropped; subjects: {subjects}"
    assert [t for t in triples if t[1].lower() == "it"] == []


def test_coref_resolves_pronoun_to_antecedent():
    # resolve_basic_pronouns always runs; with coreferee installed it
    # substitutes the antecedent before spaCy parses.
    spacy = pytest.importorskip("spacy", reason="spaCy not installed")
    try: spacy.load("en_core_web_sm")
    except OSError: pytest.skip("en_core_web_sm not installed")
    try: import coreferee  # noqa: F401
    except ImportError: pytest.skip("coreferee not installed")

    from ingest.extraction import resolve_basic_pronouns
    resolved = resolve_basic_pronouns("France invaded Russia. It suffered catastrophic losses.")
    assert "it" not in resolved.lower().split() or "france" in resolved.lower()
