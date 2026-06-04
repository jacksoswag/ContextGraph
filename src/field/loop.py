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
from .gather import materialize, gather, build_mesh, edge_weights, Mesh
from .harness import safe_build
from .baseline import personalized_pagerank
from .seams import interpret
from embed import embed, unpack
from llm import call_json, call_llm

# answer prompt: permissive (specifics + own knowledge + follow the chain) — the prompt the grow+q
# context was benchmarked with. Plain prose over the rendered facts.
ANSWER_PROMPT = ("Reference facts (may be partial):\n{ctx}\n\nAnswer with specific facts — name names and "
                 "list what's asked, following the chain through any intermediate entities. Use BOTH "
                 "these facts AND your own knowledge.\nQuestion: {q}")

# grow+q gather config: genericity-localized uniform growth; k_hop 3 / up_max 60 cover a 2-hop
# lateral reach. The PPR settle is a direct linear solve (no step horizon to tune).
GATHER_CFG = replace(DEFAULT_CFG, decay_gamma=1.0, k_hop=3, up_max=60, N_max=400, target_size=34)
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

# structural coupling for the recursive path: pure adjacency × struct_edge_w (count·confidence) with the
# genericity leak on; semantic cosine OFF (Gate 2 — structural ≥ semantic). N_max per round stays the
# single-settle reach; the recursion supplies multi-hop, not a bigger episode.
MESH_CFG = replace(GATHER_CFG, couple_mode="structural")

# mesh_gather: the recursive collapse-to-mesh spine. Each round runs a structural personalized PageRank
# (Gate 3 — the linear solve, not the integrator) teleported by the ANCESTRY vector, collapses the top
# reified (e_) facts into the mesh, then re-personalizes on the freshly collapsed frontier (relevance-
# weighted, depth-decayed) and expands. Two stops, whichever fires first: a round adds little new
# (novelty < novelty_eps ⇒ region covered) or its relevance converges to the background PageRank
# (TV < bg_tau ⇒ ancestry collapsed to noise). seeds given directly ⇒ testable without grounding.
def mesh_gather(store, seeds, cfg: FieldConfig = MESH_CFG, *, max_rounds: int = 6, collapse_k: int = 12,
                novelty_eps: float = 0.10, bg_tau: float = 0.05, depth_decay: float = 0.6) -> Mesh:
    seeds = [s for s in dict.fromkeys(seeds)]
    if not seeds: return Mesh([], [], {}, [])
    ew = getattr(store, "struct_edge_weights", lambda: None)()
    ancestry = {s: 1.0 for s in seeds}; frontier = list(seeds)
    scored: dict[str, float] = {}                            # e_id → relevance at collapse
    seen = set(seeds)
    for depth in range(max_rounds):
        ep, si = materialize(store, frontier, cfg)
        if not ep.node_ids: break
        C, _ = safe_build(ep, cfg, edge_weights(ep, ew)); W = C.sym; N = len(ep.node_ids)
        t = torch.zeros(N)
        for nid, w in ancestry.items():
            j = ep.id_to_idx.get(nid)
            if j is not None: t[j] = w
        if float(t.sum()) <= 0: t[si] = 1.0
        r = personalized_pagerank(W, teleport=t)
        pi = personalized_pagerank(W, teleport=torch.ones(N))    # background null (uniform teleport = global PR)
        if depth > 0 and 0.5 * float((r - pi).abs().sum()) < bg_tau: break   # ancestry → background = noise
        order = sorted((i for i in range(N) if ep.node_ids[i].startswith("e_") and ep.node_ids[i] not in scored),
                       key=lambda i: -float(r[i]))
        new = order[:collapse_k]
        if not new: break
        kids, nxt = set(), {}
        decay = depth_decay ** (depth + 1)
        for i in new:
            eid = ep.node_ids[i]; scored[eid] = float(r[i])
            for kid in (store.children(eid) or ()):
                kids.add(kid); nxt[kid] = nxt.get(kid, 0.0) + float(r[i]) * decay
        novelty = len(kids - seen) / max(len(kids), 1)
        seen |= kids; ancestry = nxt or ancestry; frontier = list(nxt) or frontier
        if novelty < novelty_eps: break                          # region covered
    ranked = sorted(scored, key=lambda e: -scored[e])
    return Mesh(list(range(len(ranked))), ranked, {i: scored[e] for i, e in enumerate(ranked)}, [])

# respond: the main path — interpret grounds the SUBJECT entities (3B refine + spaCy seed extraction,
# multifaceted), then the recursive collapse-to-mesh gather builds the connected multi-hop region, then
# ONE permissive 3B answer over the rendered facts. The query enters at grounding + the answer prompt;
# the mesh expansion stays query-blind (structure). gather_context (single settle) is kept for the benches.
def respond(query, store, cfg: FieldConfig = MESH_CFG, *, llm=call_json, model: str = "3B") -> Response:
    interp = interpret(query, store, llm=llm)
    seeds = interp["seeds"]
    no_match = {"prose": "No matching concepts found in the graph.", "citations": []}
    if not seeds: return Response(interp["intent"], [], no_match, [])
    mesh = mesh_gather(store, seeds, cfg)
    if not mesh.node_ids: return Response(interp["intent"], seeds, no_match, [])
    facts = [t for n in mesh.node_ids if n.startswith("e_") and (t := store.text(n))]
    ctx = "\n".join(f"- {f}" for f in facts)
    prose = call_llm(ANSWER_PROMPT.format(ctx=ctx, q=query), model,
                     options={"temperature": 0, "num_predict": 280}).strip()
    cites = [n for n in mesh.node_ids if n.startswith("e_")][:8]
    return Response(interp["intent"], seeds, {"prose": prose, "citations": cites}, list(mesh.node_ids))
