from __future__ import annotations
# THE retrieval pipeline — one linear path, no recursion:
#   prompt → interpret (3B refine + spaCy seed-hyperedge extraction → multifaceted seeds in S)
#          → grow+q gather (physics GATHERS the connected region; the query SELECTS at readout)
#          → ONE permissive 3B answer over the rendered facts.
# Physics is query-agnostic (structure); the query enters only at readout (query_w). grow_cohere
# deleted — expansion stays query-blind (RAG-creep). The recursive decompose/solve/synthesize tree
# was DELETED — it underperformed direct context-injection at every tier (~2x worse). See memory
# project_sstar_pipeline. Context-engineering not RAG: gather a structured neighborhood, not chunks.
from dataclasses import dataclass, replace
import torch
from .config import FieldConfig, DEFAULT_CFG
from .gather import materialize, gather, build_mesh, Mesh
from .seams import interpret
from embed import embed, unpack
from llm import call_json, call_llm

# answer prompt: permissive (specifics + own knowledge + follow the chain) — the prompt the grow+q
# context was benchmarked with. Plain prose over the rendered facts.
ANSWER_PROMPT = ("Reference facts (may be partial):\n{ctx}\n\nAnswer with specific facts — name names and "
                 "list what's asked, following the chain through any intermediate entities. Use BOTH "
                 "these facts AND your own knowledge.\nQuestion: {q}")

# grow+q gather config: genericity-localized uniform growth; k_hop 3 / up_max 60 cover a 2-hop
# lateral reach. eps_x 1e-3: top-34 set stabilizes long before ‖ΔX‖ hits 1e-4 — measured top-34
# overlap 1.0000 vs tight settle, cutting steps 1.5–4×. Pairs with gather(lean=True).
GATHER_CFG = replace(DEFAULT_CFG, decay_gamma=1.0, k_hop=3, up_max=60, N_max=400,
                     target_size=34, eps_x=1e-3)
QUERY_W = 4.0                                            # readout selection weight (query meets structure)

# Response: the result of one pipeline run. Flat — there is no tree. mesh_ids is the gathered region
# (relevance-ranked) the answer was grounded in; citations are the e_ facts shown (provenance).
@dataclass
class Response:
    intent: str
    seeds: list[str]
    answer: dict                 # {prose, citations}
    mesh_ids: list[str]

# gather_context: run the uniform gather from the given seeds; query focuses readout (query_w).
# Growth is query-blind (grow_cohere deleted); query enters only at build_mesh selection.
def gather_context(store, query: str, seeds: list[str], cfg: FieldConfig = GATHER_CFG) -> Mesh | None:
    ep, si = materialize(store, seeds, cfg)
    if not ep.node_ids: return None
    ew = getattr(store, "struct_edge_weights", lambda: None)()
    res = gather(ep, si, cfg, weights=ew, lean=True)
    qb = embed(query); qv = unpack(qb) if qb is not None else None
    qvt = torch.tensor(qv) if qv is not None else None
    return build_mesh(res, top_k=cfg.target_size, query_vec=qvt, query_w=QUERY_W)

# respond: the main path — interpret grounds the SUBJECT entities (3B refine + spaCy seed extraction,
# multifaceted), then ONE grow+q gather, then ONE permissive 3B answer over the rendered facts.
def respond(query, store, cfg: FieldConfig = GATHER_CFG, *, llm=call_json, model: str = "3B") -> Response:
    interp = interpret(query, store, llm=llm)
    seeds = interp["seeds"]
    no_match = {"prose": "No matching concepts found in the graph.", "citations": []}
    if not seeds: return Response(interp["intent"], [], no_match, [])
    mesh = gather_context(store, query, seeds, cfg)
    if mesh is None or not mesh.node_ids: return Response(interp["intent"], seeds, no_match, [])
    facts = [t for n in mesh.node_ids if n.startswith("e_") and (t := store.text(n))]
    ctx = "\n".join(f"- {f}" for f in facts)
    prose = call_llm(ANSWER_PROMPT.format(ctx=ctx, q=query), model,
                     options={"temperature": 0, "num_predict": 280}).strip()
    cites = [n for n in mesh.node_ids if n.startswith("e_")][:8]
    return Response(interp["intent"], seeds, {"prose": prose, "citations": cites}, list(mesh.node_ids))
