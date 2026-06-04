from __future__ import annotations
# The single retrieval pipeline (respond): interpret → grow+q gather → one permissive answer. No
# recursion. Control flow is tested with interpret / gather_context / call_llm stubbed (the gather
# physics is covered by test_gather); these assert the wiring: seeds → context → answer + provenance.
import field.loop as loop_mod
from field.loop import respond, Response
from field.gather import Mesh

# fake store: text(e_) → fact surface. interpret + gather_context are stubbed per-test, so the store
# only needs to render the e_ facts respond pulls into context.
class _Store:
    def __init__(self, facts): self._facts = facts
    def text(self, nid): return self._facts.get(nid)

# a mesh whose node_ids are exactly `ids` (relevance-flat; respond only reads node_ids).
def _mesh(ids):
    n = len(ids)
    return Mesh(nodes=list(range(n)), node_ids=list(ids), relevance={i: 1.0 for i in range(n)},
                parent={i: -1 for i in range(n)}, layer={i: 0 for i in range(n)}, seed_roots=[0])

def _stub(monkeypatch, *, seeds, mesh, prose="ANSWER"):
    monkeypatch.setattr(loop_mod, "interpret", lambda q, s, **k: {"seeds": seeds, "intent": "the intent"})
    monkeypatch.setattr(loop_mod, "gather_context", lambda *a, **k: mesh)
    monkeypatch.setattr(loop_mod, "call_llm", lambda prompt, model, options=None: prose)

# ── happy path: seeds → gathered facts → answer + e_-only citations ───────────────────

def test_respond_builds_context_and_answers(monkeypatch):
    store = _Store({"e_1": "curie discover radium", "e_2": "curie win nobel"})
    _stub(monkeypatch, seeds=["n_curie"], mesh=_mesh(["e_1", "n_curie", "e_2"]))
    r = respond("what did curie do", store)
    assert isinstance(r, Response)
    assert r.answer["prose"] == "ANSWER"
    assert r.answer["citations"] == ["e_1", "e_2"]     # only reified facts cite, plain nodes excluded
    assert r.seeds == ["n_curie"] and r.intent == "the intent"
    assert r.mesh_ids == ["e_1", "n_curie", "e_2"]     # full gathered region carried as provenance

def test_respond_context_passed_to_llm(monkeypatch):
    store = _Store({"e_1": "curie discover radium"})
    seen = {}
    monkeypatch.setattr(loop_mod, "interpret", lambda q, s, **k: {"seeds": ["n_c"], "intent": "i"})
    monkeypatch.setattr(loop_mod, "gather_context", lambda *a, **k: _mesh(["e_1"]))
    def cap(prompt, model, options=None): seen["p"] = prompt; seen["m"] = model; return "ok"
    monkeypatch.setattr(loop_mod, "call_llm", cap)
    respond("q", store)
    assert "curie discover radium" in seen["p"]         # the gathered fact reached the answer prompt
    assert seen["m"] == "3B"                             # 3B answer tier

def test_respond_citations_capped_at_eight(monkeypatch):
    facts = {f"e_{i}": f"fact {i}" for i in range(12)}
    store = _Store(facts)
    _stub(monkeypatch, seeds=["n_x"], mesh=_mesh([f"e_{i}" for i in range(12)]))
    r = respond("q", store)
    assert len(r.answer["citations"]) == 8

# ── graceful degradation ──────────────────────────────────────────────────────────────

def test_respond_no_seeds_graceful(monkeypatch):
    store = _Store({})
    monkeypatch.setattr(loop_mod, "interpret", lambda q, s, **k: {"seeds": [], "intent": "i"})
    r = respond("unknown topic", store)
    assert r.seeds == [] and "No matching concepts" in r.answer["prose"]
    assert r.answer["citations"] == [] and r.mesh_ids == []

def test_respond_empty_mesh_graceful(monkeypatch):
    store = _Store({})
    monkeypatch.setattr(loop_mod, "interpret", lambda q, s, **k: {"seeds": ["n_x"], "intent": "i"})
    monkeypatch.setattr(loop_mod, "gather_context", lambda *a, **k: None)
    r = respond("q", store)
    assert r.seeds == ["n_x"] and "No matching concepts" in r.answer["prose"]
