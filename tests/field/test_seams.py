from __future__ import annotations
# Phase 4 — the four LLM seams (spec §4), contract-tested with a deterministic mock (no backend).
import torch, pytest
from field.seams import interpret, decompose, sufficient, synthesize, render_mesh
from field.gather import Mesh

# minimal store double: text→node-id resolution + node text (for interpret + render_mesh)
class FakeStore:
    def __init__(self, mapping):
        self._m = {k.lower(): v for k, v in mapping.items()}   # text → [node_ids]
        self._ids = {nid: k for k, v in mapping.items() for nid in v}
    def find(self, q, k=5): return self._m.get(q.lower(), [])[:k]
    def text(self, nid): return self._ids.get(nid, nid)

# ── interpret: resolves entity phrases to REAL node-ids; falls back to query FTS ──────

def test_interpret_resolves_entities_to_node_ids(mock_llm):
    store = FakeStore({"france": ["n_fr"], "paris": ["n_par"]})
    llm = mock_llm(lambda p, m: {"entities": ["france", "paris"], "intent": "capital of france"})
    out = interpret("what is the capital of france", store, llm=llm)
    assert out["seeds"] == ["n_fr", "n_par"]          # real node-ids, not free text
    assert out["intent"] == "capital of france"
    assert llm.calls[0][1] == "1B"                    # cheap tier for interpret

def test_interpret_drops_unresolvable_entities(mock_llm):
    store = FakeStore({"france": ["n_fr"]})
    llm = mock_llm(lambda p, m: {"entities": ["france", "atlantis"], "intent": "x"})
    out = interpret("q", store, llm=llm)
    assert out["seeds"] == ["n_fr"]                   # 'atlantis' has no node → dropped

def test_interpret_falls_back_to_query_fts(mock_llm):
    store = FakeStore({"some query": ["n_q1", "n_q2"]})
    llm = mock_llm(lambda p, m: {"entities": [], "intent": ""})   # no entities resolved
    out = interpret("some query", store, llm=llm)
    assert out["seeds"] == ["n_q1", "n_q2"] and out["intent"] == "some query"

def test_interpret_handles_malformed(mock_llm):
    store = FakeStore({"q": ["n_q"]})
    llm = mock_llm(lambda p, m: "not a dict")
    out = interpret("q", store, llm=llm)
    assert out["seeds"] == ["n_q"]                    # fell back to FTS; no crash

# ── decompose: {subtasks} | {done}; malformed/empty ⇒ done ───────────────────────────

def test_decompose_returns_subtasks(mock_llm):
    llm = mock_llm(lambda p, m: {"subtasks": ["a", "b", "c"]})
    assert decompose("task", "mesh", llm=llm) == {"subtasks": ["a", "b", "c"]}

def test_decompose_done(mock_llm):
    llm = mock_llm(lambda p, m: {"done": True})
    assert decompose("task", "mesh", llm=llm) == {"done": True}

def test_decompose_caps_subtasks(mock_llm):
    llm = mock_llm(lambda p, m: {"subtasks": [f"s{i}" for i in range(9)]})
    assert len(decompose("t", "m", llm=llm, max_subtasks=4)["subtasks"]) == 4

def test_decompose_malformed_is_done(mock_llm):
    llm = mock_llm(lambda p, m: {"garbage": 1})
    assert decompose("t", "m", llm=llm) == {"done": True}

def test_decompose_empty_subtasks_is_done(mock_llm):
    llm = mock_llm(lambda p, m: {"subtasks": []})
    assert decompose("t", "m", llm=llm) == {"done": True}

# ── sufficient: {stop, reason}; malformed ⇒ stop=True ────────────────────────────────

def test_sufficient_stop(mock_llm):
    llm = mock_llm(lambda p, m: {"stop": True, "reason": "covered"})
    out = sufficient("t", "m", llm=llm)
    assert out["stop"] is True and out["reason"] == "covered"

def test_sufficient_continue(mock_llm):
    llm = mock_llm(lambda p, m: {"stop": False, "reason": "need more"})
    assert sufficient("t", "m", llm=llm)["stop"] is False

def test_sufficient_malformed_defaults_stop(mock_llm):
    llm = mock_llm(lambda p, m: {"oops": 1})
    assert sufficient("t", "m", llm=llm)["stop"] is True

# ── synthesize: {prose, citations}; citations filtered to the mesh ───────────────────

def test_synthesize_filters_citations_to_mesh(mock_llm):
    llm = mock_llm(lambda p, m: {"prose": "answer", "citations": ["n_a", "n_HALLUCINATED"]})
    out = synthesize("t", "mesh", [], valid_ids=["n_a", "n_b"], llm=llm)
    assert out["prose"] == "answer"
    assert out["citations"] == ["n_a"]               # hallucinated id dropped

def test_synthesize_empty_prose_falls_back(mock_llm):
    llm = mock_llm(lambda p, m: {"citations": []})
    out = synthesize("the task", "the mesh", [{"prose": "child says X"}], valid_ids=[], llm=llm)
    assert "the task" in out["prose"] and "child says X" in out["prose"]

def test_synthesize_includes_children(mock_llm):
    seen = {}
    def resp(p, m): seen["prompt"] = p; return {"prose": "ok", "citations": []}
    synthesize("t", "m", [{"prose": "kid1"}, {"prose": "kid2"}], valid_ids=[], llm=mock_llm(resp))
    assert "kid1" in seen["prompt"] and "kid2" in seen["prompt"]

# ── render_mesh: prompt block + citable id list ──────────────────────────────────────

def test_render_mesh_lists_nodes_and_returns_ids():
    # mesh: seed n_s (idx0) → child n_c (idx1)
    mesh = Mesh(nodes=[0, 1], node_ids=["n_s", "n_c"], relevance={0: 9.0, 1: 2.0},
                parent={0: -1, 1: 0}, layer={0: 0, 1: 1}, seed_roots=[0])
    store = FakeStore({"seedword": ["n_s"], "childword": ["n_c"]})
    text, ids = render_mesh(mesh, store, max_nodes=12)
    assert ids == ["n_s", "n_c"]
    assert "n_s" in text and "n_c" in text
    assert "via" in text                             # provenance chain rendered

def test_render_mesh_caps_nodes():
    mesh = Mesh(nodes=list(range(20)), node_ids=[f"n{i}" for i in range(20)],
                relevance={i: float(20 - i) for i in range(20)},
                parent={**{0: -1}, **{i: 0 for i in range(1, 20)}},
                layer={**{0: 0}, **{i: 1 for i in range(1, 20)}}, seed_roots=[0])
    store = FakeStore({})
    _, ids = render_mesh(mesh, store, max_nodes=5)
    assert len(ids) == 5
