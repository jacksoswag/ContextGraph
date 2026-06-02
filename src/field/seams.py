from __future__ import annotations
# The four LLM seams (spec §4): the ONLY places an LLM is called. Each is a pure JSON-contract
# function; the LLM is injected (default call_json) so it is deterministic + mock-testable.
# interpret → entry seeds (resolved to REAL node-ids in S, not free text); decompose → sub-tasks;
# sufficient → stop gate; synthesize → prose + citations. Nothing here touches the field math.
from typing import Callable
from .gather import Mesh
from llm import call_json

LLM = Callable[..., dict]   # (prompt, model) -> dict

# ── prompts (tunable; presented at 📍4) ───────────────────────────────────────────────
INTERPRET_PROMPT = (
    "Map this query to entry concepts in a knowledge graph. Extract the key entities/concepts "
    "(short noun phrases) and a one-sentence intent.\n"
    'Return JSON: {{"entities": ["..."], "intent": "..."}}\n\nQuery: {query}')
DECOMPOSE_PROMPT = (
    "Break the task into 2-4 independent sub-questions, using the gathered context. If the context "
    "already answers the task, return {{\"done\": true}}; otherwise return {{\"subtasks\": [\"...\"]}}.\n\n"
    "Task: {task}\n\nGathered context:\n{mesh}")
SUFFICIENT_PROMPT = (
    "Is the gathered context enough to answer the task on its own? "
    'Return JSON {{"stop": true|false, "reason": "..."}}.\n\nTask: {task}\n\nContext:\n{mesh}')
SYNTHESIZE_PROMPT = (
    "Answer the task using ONLY the gathered context and child answers. Cite the concept ids you "
    'used.\nReturn JSON {{"prose": "...", "citations": ["concept_id"]}}.\n\n'
    "Task: {task}\n\nContext:\n{mesh}\n\nChild answers:\n{children}")

# render_mesh: turn a Mesh into a compact prompt block + the set of citable node-ids. Each line is a
# concept (text, relevance) with its provenance chain back to a seed (the citation the LLM may use).
def render_mesh(mesh: Mesh, store, max_nodes: int = 12) -> tuple[str, list[str]]:
    def _t(nid): return (store.text(nid) or nid).split("|")[0]
    lines: list[str] = []; ids: list[str] = []
    for pos, i in enumerate(mesh.nodes[:max_nodes]):
        nid = mesh.node_ids[pos]
        chain = " < ".join(_t(c) for c in mesh.chain_ids(i))
        lines.append(f"- [{nid}] {_t(nid)} (rel {mesh.relevance[i]:.2f}; via {chain})")
        ids.append(nid)
    return ("\n".join(lines) or "(empty)"), ids

def _is_dict(d) -> bool: return isinstance(d, dict)

# interpret: query → {seeds:[real node_id], intent}. The LLM proposes entity phrases; each is
# resolved to a real node in S via store.find (NOT free generation). Falls back to FTS on the
# whole query when nothing resolves.
def interpret(query: str, store, *, llm: LLM = call_json, model: str = "1B", max_seeds: int = 4) -> dict:
    out = llm(INTERPRET_PROMPT.format(query=query), model)
    phrases = out.get("entities") if _is_dict(out) else None
    intent = out.get("intent") if _is_dict(out) else None
    seeds: list[str] = []
    for ph in (phrases or []):
        for nid in store.find(str(ph), 1):
            if nid not in seeds: seeds.append(nid)
    if not seeds: seeds = store.find(query, max_seeds)        # fallback: ground the raw query
    return {"seeds": seeds[:max_seeds], "intent": str(intent) if intent else query}

# decompose: task + mesh → {subtasks:[str]} or {done:true}. Malformed/empty ⇒ done (stop, safe).
def decompose(task: str, mesh_text: str, *, llm: LLM = call_json, model: str = "7B",
              max_subtasks: int = 4) -> dict:
    out = llm(DECOMPOSE_PROMPT.format(task=task, mesh=mesh_text), model)
    if _is_dict(out) and out.get("done") is True: return {"done": True}
    subs = out.get("subtasks") if _is_dict(out) else None
    if isinstance(subs, list) and subs:
        return {"subtasks": [str(s) for s in subs if str(s).strip()][:max_subtasks]}
    return {"done": True}

# sufficient: task + mesh → {stop:bool, reason}. Malformed ⇒ stop=True (don't recurse on garbage).
def sufficient(task: str, mesh_text: str, *, llm: LLM = call_json, model: str = "1B") -> dict:
    out = llm(SUFFICIENT_PROMPT.format(task=task, mesh=mesh_text), model)
    if not _is_dict(out) or "stop" not in out:
        return {"stop": True, "reason": "malformed seam output → stop"}
    return {"stop": bool(out.get("stop")), "reason": str(out.get("reason", ""))}

# synthesize: task + mesh + child answers → {prose, citations}. Citations are filtered to the
# node-ids actually present in the mesh (no hallucinated citations). Empty prose ⇒ a fallback stitch.
def synthesize(task: str, mesh_text: str, children: list[dict], valid_ids: list[str], *,
               llm: LLM = call_json, model: str = "7B") -> dict:
    child_text = "\n".join(f"- {c.get('prose', '')}" for c in children) or "(none)"
    out = llm(SYNTHESIZE_PROMPT.format(task=task, mesh=mesh_text, children=child_text), model)
    prose = str(out.get("prose", "")) if _is_dict(out) else ""
    raw_c = out.get("citations") if _is_dict(out) else None
    valid = set(valid_ids)
    citations = [c for c in raw_c if c in valid] if isinstance(raw_c, list) else []
    if not prose:                                            # fallback: stitch task + context + children
        prose = f"{task}\n\n{mesh_text}\n\n{child_text}".strip()
    return {"prose": prose, "citations": citations}
