from __future__ import annotations
# The interpret seam (spec §4) — prompt → 3B refine → spaCy seed-hyperedge extraction → grounded,
# multifaceted seeds. Contract-tested with a deterministic mock LLM + an injected parse stub (no spaCy,
# no backend). render_outline is covered by the gather/behavior tests.
import pytest
from field.seams import interpret, _spacy_fragments, _is_meta, _pick_seeds

# minimal store double: text→node-id resolution (no anchor ⇒ _select_seeds keeps every candidate).
class FakeStore:
    def __init__(self, mapping):
        self._m = {k.lower(): v for k, v in mapping.items()}   # text → [node_ids]
        self._ids = {nid: k for k, v in mapping.items() for nid in v}
    def find(self, q, k=5): return self._m.get(q.lower(), [])[:k]
    def text(self, nid): return self._ids.get(nid, nid)

# _node(text): a spaCy node endpoint. _edge: a clause (hyperedge) over two endpoints.
def _node(t): return {"type": "node", "text": t}
def _edge(s, r, o): return {"type": "edge", "rel": r, "source": _node(s), "target": _node(o)}
def _parse(forest): return lambda text: forest   # injectable spaCy stub

# ── interpret: spaCy endpoints grounded to real node-ids, every facet kept ────────────

def test_interpret_grounds_fragments_multifaceted(mock_llm):
    store = FakeStore({"france": ["n_fr"], "paris": ["n_par"]})
    llm = mock_llm(lambda p, m: {"statements": ["france has a capital", "paris is a city"],
                                 "intent": "capital of france"})
    forest = [_edge("france", "have", "capital"), _edge("paris", "be", "city")]
    out = interpret("capital of france?", store, llm=llm, parse=_parse(forest))
    assert out["seeds"] == ["n_fr", "n_par"]          # both facets grounded (multifaceted)
    assert out["intent"] == "capital of france"
    assert llm.calls[0][1] == "3B"                    # 3B refine tier

def test_interpret_drops_unresolvable_endpoints(mock_llm):
    store = FakeStore({"france": ["n_fr"]})
    llm = mock_llm(lambda p, m: {"statements": ["france borders atlantis"], "intent": "x"})
    forest = [_edge("france", "border", "atlantis")]
    out = interpret("q", store, llm=llm, parse=_parse(forest))
    assert out["seeds"] == ["n_fr"]                   # 'atlantis' has no node → dropped

def test_interpret_falls_back_to_query_when_no_endpoints(mock_llm):
    store = FakeStore({"some query": ["n_q1", "n_q2"]})
    llm = mock_llm(lambda p, m: {"statements": [], "intent": ""})   # nothing refined
    out = interpret("some query", store, llm=llm, parse=_parse([]))  # spaCy grounds nothing
    assert out["seeds"] == ["n_q1", "n_q2"] and out["intent"] == "some query"

def test_interpret_handles_malformed_llm(mock_llm):
    store = FakeStore({"q": ["n_q"]})
    llm = mock_llm(lambda p, m: "not a dict")
    out = interpret("q", store, llm=llm, parse=_parse([]))
    assert out["seeds"] == ["n_q"]                    # fell back to query grounding; no crash

def test_interpret_nested_hyperedge_endpoints(mock_llm):
    # a clausal-complement: papen convinced [hindenburg appoint hitler] — endpoints of the inner edge count
    store = FakeStore({"papen": ["n_p"], "hindenburg": ["n_h"], "hitler": ["n_x"]})
    llm = mock_llm(lambda p, m: {"statements": ["papen convinced hindenburg to appoint hitler"], "intent": "i"})
    inner = _edge("hindenburg", "appoint", "hitler")
    forest = [{"type": "edge", "rel": "convince", "source": _node("papen"), "target": inner}]
    out = interpret("q", store, llm=llm, parse=_parse(forest))
    assert set(out["seeds"]) == {"n_p", "n_h", "n_x"}  # outer subject + nested members all seed

# ── _spacy_fragments: relation fragments FIRST, then bare entities ────────────────────

def test_spacy_fragments_relation_then_entities():
    forest = [_edge("curie", "discover", "radium"), _edge("curie", "win", "nobel")]
    # rendered "s rel o" fragments (→ reified edges) precede the bare nodes (→ entity anchors)
    assert _spacy_fragments("text", _parse(forest)) == [
        "curie discover radium", "curie win nobel", "curie", "radium", "nobel"]

def test_spacy_fragments_missing_object_renders_subject_relation():
    # an [event]/pending object → fragment is subject+relation only, and the junk object is not seeded
    forest = [_edge("einstein", "win", "[event]")]
    assert _spacy_fragments("text", _parse(forest)) == ["einstein win", "einstein"]

def test_spacy_fragments_underscore_relation_normalized():
    forest = [_edge("einstein", "award_received", "nobel prize")]
    assert _spacy_fragments("t", _parse(forest))[0] == "einstein award received nobel prize"

def test_spacy_fragments_nested_hyperedge():
    inner = _edge("hindenburg", "appoint", "hitler")
    forest = [{"type": "edge", "rel": "convince", "source": _node("papen"), "target": inner}]
    out = _spacy_fragments("t", _parse(forest))
    assert "papen convince" in out                    # outer: object is an edge → subject+relation
    assert "hindenburg appoint hitler" in out          # inner proposition rendered in full
    assert {"papen", "hindenburg", "hitler"} <= set(out)

def test_spacy_fragments_parse_failure_is_empty():
    def boom(_text): raise RuntimeError("no spaCy")
    assert _spacy_fragments("text", boom) == []

def test_spacy_fragments_blank_text():
    assert _spacy_fragments("   ", _parse([_edge("a", "r", "b")])) == []

# ── _is_meta: namespace/admin pages filtered, real entities + propositions kept ───────

def test_is_meta_filters_namespaces():
    assert _is_meta("Category:Albert Einstein") and _is_meta("wikiproject mathematics")
    assert _is_meta("Wikipedia:Vital articles/Level/4") and _is_meta("List of physicists")
    assert not _is_meta("albert einstein") and not _is_meta("nobel prize in physics")

# ── _pick_seeds: dedup aliases, keep per-facet objects, floor junk, cap ────────────────

def test_pick_seeds_dedups_surface_form_aliases():
    facets = [["n_ae", "n_e"]]                            # albert einstein / einstein (alias)
    score = {"n_ae": 0.90, "n_e": 0.85}; text = {"n_ae": "albert einstein", "n_e": "einstein"}
    assert _pick_seeds(facets, score, text, 6) == ["n_ae"]   # "einstein" ⊂ "albert einstein" → dropped

def test_pick_seeds_keeps_object_facets_under_dominant_subject():
    # the subject grounds in EVERY facet (dominant) — each facet's distinct object must still survive
    facets = [["fermi", "italy"], ["fermi", "usa"], ["fermi", "chicago"]]
    score = {"fermi": 0.75, "italy": 0.39, "usa": 0.25, "chicago": 0.30}
    out = _pick_seeds(facets, score, {k: k for k in score}, 6)
    assert out[0] == "fermi" and set(out) == {"fermi", "italy", "usa", "chicago"}

def test_pick_seeds_floor_drops_junk():
    facets = [["fermi", "person"]]
    score = {"fermi": 0.75, "person": 0.05}
    assert _pick_seeds(facets, score, {"fermi": "fermi", "person": "person"}, 6) == ["fermi"]

def test_pick_seeds_edges_never_aliased():
    facets = [["e_1", "e_2"]]                             # two propositions with overlapping text
    score = {"e_1": 0.70, "e_2": 0.60}
    out = _pick_seeds(facets, score, {"e_1": "einstein win nobel", "e_2": "einstein win prize"}, 6)
    assert set(out) == {"e_1", "e_2"}

def test_pick_seeds_caps_at_max():
    facets = [[f"n{i}"] for i in range(10)]
    score = {f"n{i}": 1.0 - 0.01 * i for i in range(10)}
    assert len(_pick_seeds(facets, score, {f"n{i}": f"entity{i}" for i in range(10)}, 4)) == 4
