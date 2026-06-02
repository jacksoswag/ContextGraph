from __future__ import annotations
# Producer behavior: extraction → editor.ingest_text → typed-triple dicts.
# Verifies the decoupled producer emits the expected clause shape for
# negation, possessive, ditransitive, and coordination patterns — no graph
# store. Tests skip if the spaCy model is unavailable.
import sys, pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

spacy = pytest.importorskip("spacy", reason="spaCy not installed")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

from ingest.editor import ingest_text


# ── triple-dict walkers ───────────────────────────────────────────────────────

def _iter_edges(clauses):
    # Yield every edge dict in the clause forest (nested targets + modifiers).
    stack = list(clauses)
    while stack:
        c = stack.pop()
        if not isinstance(c, dict): continue
        if c.get("type") == "edge":
            yield c
            stack.extend([c.get("source"), c.get("target")])
            stack.extend(m.get("target") for m in c.get("modifiers", []))
        elif c.get("type") == "node":
            stack.extend(m.get("target") for m in c.get("modifiers", []))


def _node_text(x):
    return x.get("text", "") if isinstance(x, dict) and x.get("type") == "node" else ""


def _has_edge(clauses, *, subj=None, rel=None, obj=None):
    for e in _iter_edges(clauses):
        s, o, r = _node_text(e.get("source")).lower(), _node_text(e.get("target")).lower(), (e.get("rel") or "").lower()
        if subj and subj not in s: continue
        if obj and obj not in o: continue
        if rel and rel not in r: continue
        return True
    return False


def _texts_and_rels(clauses):
    texts, rels = set(), []
    for e in _iter_edges(clauses):
        rels.append(e.get("rel") or "")
        for end in (e.get("source"), e.get("target")):
            if t := _node_text(end): texts.add(t)
    return texts, rels


def _ingest(text, *, fallback=None):
    # Stub the LLM: bare-question check returns False (ingest it); the
    # zero-extraction fallback returns `fallback` triples (default: none),
    # so these tests cover the spaCy extraction path unless overridden.
    import llm as llm_mod
    original = llm_mod.call_json

    def _stub(prompt, *a, **kw):
        if "is_question" in prompt: return {"is_question": False}
        return {"triples": fallback or []}

    llm_mod.call_json = _stub
    try:
        return list(ingest_text(text))
    finally:
        llm_mod.call_json = original


# ── Negation: "Dogs do not eat cats" → relation prefixed with "not_" ──────────

def test_negation_encodes_in_relation_label():
    clauses = _ingest("Dogs do not eat cats.")
    assert clauses, "expected at least one clause"
    texts, rels = _texts_and_rels(clauses)
    assert any("not" in r for r in rels), f"expected a not_ relation, got: {rels}"
    assert any("dog" in t for t in texts) and any("cat" in t for t in texts)


# ── Possessive: "Alice's book fell off the table" ─────────────────────────────

def test_possessive_extracts_action_edge():
    clauses = _ingest("Alice's book fell off the table.")
    assert clauses
    assert _has_edge(clauses, subj="book"), "expected 'book' as a subject node"


def test_possessive_creates_ownership_edge():
    clauses = _ingest("Alice's book fell off the table.")
    assert _has_edge(clauses, subj="alice", obj="book"), "expected an alice→book possessive edge"


# ── Ditransitive: "The teacher gave students homework" ────────────────────────

def test_ditransitive_creates_objects():
    clauses = _ingest("The teacher gave students homework.")
    assert _has_edge(clauses, subj="teacher")
    targets = {_node_text(e.get("target")).lower() for e in _iter_edges(clauses)
               if "teacher" in _node_text(e.get("source")).lower()}
    assert any("homework" in t or "student" in t for t in targets), f"got: {targets}"


# ── Coordination via LLM zero-extraction fallback ─────────────────────────────

def test_coordination_recovers_subjects_via_fallback():
    # en_core_web_sm misparses this (sting tagged ADJ); the LLM fallback recovers.
    clauses = _ingest(
        "Bees and wasps both sting predators.",
        fallback=[{"subject": "bees", "relation": "sting", "object": "predators"},
                  {"subject": "wasps", "relation": "sting", "object": "predators"}],
    )
    subjs = {_node_text(e.get("source")).lower() for e in _iter_edges(clauses)}
    assert any("bee" in s or "wasp" in s for s in subjs), f"got subjects: {subjs}"


# ── Question gate + factual coverage ──────────────────────────────────────────

def test_bare_question_yields_nothing():
    import llm as llm_mod
    original = llm_mod.call_json
    llm_mod.call_json = lambda *a, **kw: {"is_question": True}
    try:
        assert list(ingest_text("What is the speed of light?")) == []
    finally:
        llm_mod.call_json = original


def test_factual_text_yields_clauses():
    assert _ingest("Water boils at one hundred degrees Celsius."), "expected at least one clause"
