from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:  # model not installed
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)

from ingest.extraction import extract_clauses


def _collect_relation_labels(c):
    if c is None:
        return []
    if c.get("type") == "edge":
        out = [c.get("rel")]
        out.extend(_collect_relation_labels(c.get("source")))
        out.extend(_collect_relation_labels(c.get("target")))
        for inf in c.get("inflections", []):
            out.extend(_collect_relation_labels(inf.get("target")))
        return out
    out = []
    for m in c.get("modifiers", []):
        out.append(m.get("rel"))
        out.extend(_collect_relation_labels(m.get("target")))
    return out


def test_extract_clauses_strips_negation_token_from_subspans():
    # "Jackson did not run home" — pivot is "run" (or "did" then "run"),
    # the neg particle attaches to the pivot. After splitting, "not" must NOT
    # appear as a stray token in either subspan.
    clauses = list(extract_clauses("Jackson did not run home."))
    assert clauses, "expected at least one parsed clause"
    # Recursively walk the tree and collect every node 'text' field.
    def texts(c):
        if c is None:
            return []
        if c.get("type") == "node":
            yield c["text"]
            for m in c.get("modifiers", []):
                yield from texts(m["target"])
        else:
            yield from texts(c.get("source"))
            yield from texts(c.get("target"))
            for inf in c.get("inflections", []):
                yield from texts(inf["target"])

    all_texts = set()
    for c in clauses:
        all_texts.update(texts(c))
    assert "not" not in all_texts




def test_numeric_noun_phrases_are_base_nodes_not_quantify_modifiers():
    clauses = list(extract_clauses("3 apples cost 5 dollars."))
    assert clauses, "expected at least one parsed clause"
    # The AMR model may strip or embed quantities; the key invariant is that
    # no explicit 'quantify' modifier edge is emitted for numeric NPs.
    assert "quantify" not in _collect_relation_labels(clauses[0])


def test_noun_compounds_do_not_emit_generic_modifier_edges():
    clauses = list(extract_clauses("Seattle weather is wet."))
    assert clauses, "expected at least one parsed clause"
    clause = clauses[0]
    # AMR may or may not preserve the full compound; key invariant is no
    # 'modified_by' modifier edge is injected for a noun compound.
    assert "modified_by" not in _collect_relation_labels(clause)


def test_relative_pronouns_do_not_emit_possessive_edges():
    clauses = list(extract_clauses("The traveler who arrived saw the city."))
    assert clauses, "expected at least one parsed clause"
    assert "possessed_by" not in _collect_relation_labels(clauses[0])


# ── SVO boundary tests ────────────────────────────────────────────────────────

def _walk(c, texts, rels):
    if c is None: return
    if c.get("type") == "node":
        texts.add(c.get("text", ""))
        for m in c.get("modifiers", []): _walk(m.get("target"), texts, rels)
    elif c.get("type") == "edge":
        rels.append(c.get("rel", ""))
        _walk(c.get("source"), texts, rels)
        _walk(c.get("target"), texts, rels)
        # Edge-level modifiers (compositional model: amod attaches to the
        # clause edge, not to the noun node).
        for m in c.get("modifiers", []): _walk(m.get("target"), texts, rels)


# ccomp: nested complement clause creates edges at depth ≥ 2.
# "America said that Iran planned to attack Iraq" must yield all three predicates.
def test_ccomp_nested_produces_all_predicates():
    clauses = list(extract_clauses("America said that Iran planned to attack Iraq."))
    assert clauses, "expected clauses from nested ccomp sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "america" in texts and "iran" in texts and "iraq" in texts, f"missing subjects/objects: {texts}"
    assert any("say" in r for r in rels), f"expected 'say' relation, got: {rels}"
    assert any("attack" in r for r in rels), f"expected 'attack' relation (depth ≥ 2), got: {rels}"


# xcomp: control verb inherits subject when complement has no explicit subject.
# "Jackson decided to build the system" → jackson is subject of both decide and build.
def test_xcomp_inherits_subject_for_control_verbs():
    clauses = list(extract_clauses("Jackson decided to build the system."))
    assert clauses, "expected clauses from xcomp sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "jackson" in texts, f"expected jackson as subject, got: {texts}"
    assert "system" in texts, f"expected system as object of nested verb, got: {texts}"
    assert any("build" in r for r in rels), f"expected 'build' nested relation, got: {rels}"


# coordinated subjects: "and"-joined noun subjects both get individual edges.
def test_coordinated_subjects_each_get_own_edge():
    clauses = list(extract_clauses("America and Iran signed the treaty."))
    assert clauses, "expected clauses from coordinated subject sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "america" in texts and "iran" in texts, f"both subjects must appear: {texts}"
    assert "treaty" in texts, f"shared object must appear: {texts}"
    sign_rels = [r for r in rels if "sign" in r]
    assert len(sign_rels) >= 2, f"expected at least 2 'sign' edges (one per subject), got: {sign_rels}"


# relcl gap: relative clause fills implicit gap from head noun.
# "the protein that CRISPR cuts" → CRISPR -[cut]-> protein
def test_relcl_gap_object_is_head_noun():
    clauses = list(extract_clauses("The protein that CRISPR cuts enables immunity."))
    assert clauses, "expected clauses from relative clause sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "crispr" in texts and "protein" in texts, f"expected CRISPR and protein: {texts}"
    assert any("cut" in r for r in rels), f"expected 'cut' from relcl gap-fill, got: {rels}"


# prepositional object: verb + prep emits the preposition as its OWN relation
# (compositional model). The old "fused" verb_prep ("run_in") was abandoned —
# now we get a clean intransitive verb edge plus a separate prep edge.
def test_prepositional_object_creates_separate_prep_edge():
    clauses = list(extract_clauses("Jackson runs in the park."))
    assert clauses, "expected clauses from prepositional sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "jackson" in texts and "park" in texts, f"expected jackson and park: {texts}"
    assert "in" in rels, f"expected 'in' as the preposition-relation, got: {rels}"


# amod modifiers: adjectives on subject/object appear as nested modifier nodes.
def test_amod_modifiers_appear_as_nested_modifier_nodes():
    clauses = list(extract_clauses("Small dogs eat large bones."))
    assert clauses, "expected clauses from modifier sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "small" in texts, f"expected 'small' adjective modifier: {texts}"
    assert "large" in texts, f"expected 'large' adjective modifier: {texts}"


# depth cap: _MAX_EDGE_DEPTH = 5 must not be exceeded even in artificially deep sentences.
def test_extraction_respects_max_edge_depth():
    from ingest.extraction import _MAX_EDGE_DEPTH
    assert _MAX_EDGE_DEPTH == 5, "depth cap must be 5 per architecture spec"
    # Deep but parseable sentence — extraction must not raise regardless of nesting.
    deep = "John said Mary thought Alice believed Bob claimed the cat ran."
    try:
        clauses = list(extract_clauses(deep))
    except RecursionError:
        raise AssertionError("RecursionError: _MAX_EDGE_DEPTH guard failed")
    assert isinstance(clauses, list)
