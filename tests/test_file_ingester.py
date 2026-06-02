from __future__ import annotations
import sys, textwrap
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm model not installed", allow_module_level=True)

from ingest.extraction import extract_clauses
from ingest.file_reformatter import reformat_file
from ingest.file_ingester import ingest_file


# ── Reformatter boundary tests ────────────────────────────────────────────────

def test_reformat_strips_markdown_syntax():
    md = textwrap.dedent("""\
        # Section Header
        ## Subsection Title
        - **Bold** item text
        - [link text](http://example.com)
        ```
        code block to skip
        ```
        Plain sentence here.
    """)
    segments = reformat_file(md)
    joined = " ".join(segments)
    assert "#" not in joined, "header markers must be stripped"
    assert "**" not in joined, "bold markers must be stripped"
    assert "```" not in joined, "code fence markers must not appear"
    assert "code block" not in joined, "code block content must be skipped"
    assert "Plain sentence here." in joined


def test_reformat_preserves_imperative_statements():
    md = "- Do not use docstrings.\n- Use hash comments only.\n"
    segments = reformat_file(md)
    assert any("Do not use docstrings" in s for s in segments)
    assert any("Use hash comments" in s for s in segments)


def test_reformat_skips_horizontal_rules():
    segments = reformat_file("Before\n---\nAfter")
    assert all("---" not in s for s in segments)
    assert any("Before" in s for s in segments)
    assert any("After" in s for s in segments)


# ── Extraction boundary tests ─────────────────────────────────────────────────

def _walk(c, texts, rels):
    # Recursively collect node texts and edge relations from a clause dict.
    if c is None: return
    if c.get("type") == "node":
        texts.add(c.get("text", ""))
        for m in c.get("modifiers", []): _walk(m.get("target"), texts, rels)
    elif c.get("type") == "edge":
        rels.append(c.get("rel", ""))
        _walk(c.get("source"), texts, rels)
        _walk(c.get("target"), texts, rels)


# Behavioral boundary: imperative + negation → [system] -[not_*]-> [target].
# This verifies the two bugs are fixed: silent drop of imperatives and negation reversal.
def test_imperative_negation_produces_not_relation():
    clauses = list(extract_clauses("Do not use hash comments."))
    assert clauses, "expected at least one clause from imperative sentence"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert any("not_" in r for r in rels), f"expected not_* relation, got: {rels}"
    assert "system" in texts, f"expected 'system' implicit subject node, got: {texts}"


def test_plain_imperative_assigns_system_subject():
    clauses = list(extract_clauses("Use hash comments only."))
    assert clauses, "expected at least one clause from plain imperative"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    assert "system" in texts, f"expected 'system' as implicit subject, got: {texts}"


def test_negation_on_explicit_subject_uses_not_prefix():
    # "Jackson did not run" → negation must be encoded in graph (relation or node text).
    # Intransitive verbs use the fallback edge [does]->[not_run], so we check both.
    clauses = list(extract_clauses("Jackson did not run."))
    assert clauses, "expected at least one clause"
    texts, rels = set(), []
    for c in clauses: _walk(c, texts, rels)
    not_encoded = any("not_" in r for r in rels) or any("not_" in t for t in texts)
    assert not_encoded, f"expected negation encoded in rels or texts, got rels:{rels} texts:{texts}"
    assert "system" not in texts, "explicit-subject sentence must not produce implicit [system] node"


# ── File ingester boundary tests ─────────────────────────────────────────────

def test_ingest_file_raises_for_unsupported_extension(tmp_path):
    bad = tmp_path / "doc.pdf"
    bad.write_text("irrelevant content")
    with pytest.raises(ValueError, match="unsupported file type"):
        list(ingest_file(bad))  # generator raises on the suffix check


def test_reformat_plain_text_passes_through():
    segments = reformat_file("Dogs eat bones.\nCats drink milk.")
    assert any("Dogs eat bones" in s for s in segments)
    assert any("Cats drink milk" in s for s in segments)


# ── Full producer pipeline: file → typed-triple dicts ─────────────────────────

def _node_text(x):
    return x.get("text", "") if isinstance(x, dict) and x.get("type") == "node" else ""


def _walk_texts_rels(clauses):
    texts, rels = set(), []
    stack = list(clauses)
    while stack:
        c = stack.pop()
        if not isinstance(c, dict): continue
        if c.get("type") == "edge":
            rels.append(c.get("rel") or "")
            stack.extend([c.get("source"), c.get("target")])
            stack.extend(m.get("target") for m in c.get("modifiers", []))
        elif c.get("type") == "node":
            if c.get("text"): texts.add(c["text"])
            stack.extend(m.get("target") for m in c.get("modifiers", []))
    return texts, rels


def test_full_pipeline_md_to_triples(tmp_path):
    md = textwrap.dedent("""\
        # Dog Behavior
        Dogs love food.
        Do not ignore small dogs.
    """)
    md_file = tmp_path / "skill.md"
    md_file.write_text(md, encoding="utf-8")
    clauses = list(ingest_file(md_file))
    assert clauses, "expected at least one extracted clause"
    texts, rels = _walk_texts_rels(clauses)
    assert any("dog" in t for t in texts), f"expected 'dog' node, got: {texts}"
    assert any("love" in r for r in rels), f"expected 'love' relation, got: {rels}"
