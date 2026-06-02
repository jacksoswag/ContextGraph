from __future__ import annotations
# The recursion (spec §1, §5): answer(query) → solve tree. Down-pass = decompose + gather
# (parent-anchored); up-pass = synthesize children into the parent answer. The tree IS the
# reasoning trace. Physics gathers; the four seams reason. Terminates on depth / budget /
# sufficient. 📍4 budgets: depth 2, branch 3, K_inherit 4, ≤40 LLM calls, ≤600 gathered nodes.
from dataclasses import dataclass, field
import torch
from .config import FieldConfig, DEFAULT_CFG
from .gather import materialize, gather, build_mesh, GatherResult, Mesh
from .seams import interpret, decompose, sufficient, synthesize, render_mesh
from llm import call_json

MAX_DEPTH = 2
MAX_SUBTASKS = 3
MAX_LLM_CALLS = 40
MAX_NODES_TOTAL = 600

@dataclass
class Budget:
    max_depth: int = MAX_DEPTH
    max_llm_calls: int = MAX_LLM_CALLS
    max_nodes: int = MAX_NODES_TOTAL
    llm_calls: int = 0
    nodes: int = 0
    def exhausted(self) -> bool:
        return self.llm_calls >= self.max_llm_calls or self.nodes >= self.max_nodes

@dataclass
class TreeNode:
    task: str
    seeds: list[str]
    answer: dict                 # {prose, citations}
    depth: int
    mesh_ids: list[str]          # gathered concept ids (relevance-ranked)
    children: list = field(default_factory=list)

# _child_anchors: top-K_inherit parent-mesh nodes + their settled parent state, to tie a child
# gather toward the parent region (spec §5 parent-anchoring).
def _child_anchors(parent: tuple[Mesh, GatherResult] | None, cfg: FieldConfig) -> dict[str, torch.Tensor]:
    if parent is None: return {}
    p_mesh, p_res = parent
    out: dict[str, torch.Tensor] = {}
    for pos, nid in enumerate(p_mesh.node_ids[: cfg.k_inherit]):
        out[nid] = p_res.X_star[p_mesh.nodes[pos]]
    return out

# _gather_node: materialize the active set from (child seeds ∪ inherited anchors) and settle —
# child seeds hot, inherited rows pinned to the parent's settled state.
def _gather_node(seed_ids, inherited, store, cfg) -> GatherResult:
    all_ids = list(dict.fromkeys(list(seed_ids) + list(inherited.keys())))
    ep, _ = materialize(store, all_ids, cfg)
    seed_idx = [ep.id_to_idx[s] for s in seed_ids if s in ep.id_to_idx] or [0]
    inh_idx: list[int] = []
    inh_tgt = torch.zeros(len(ep.node_ids), cfg.d)
    for nid, vec in inherited.items():
        j = ep.id_to_idx.get(nid)
        if j is not None and nid not in set(seed_ids):
            inh_idx.append(j); inh_tgt[j] = vec
    return gather(ep, seed_idx, cfg, inherited_idx=inh_idx or None,
                  inherited_target=inh_tgt if inh_idx else None)

# solve: one node of the reasoning tree (spec §5).
def solve(task, seed_ids, parent, depth, store, cfg, llm, budget) -> TreeNode:
    res = _gather_node(seed_ids, _child_anchors(parent, cfg), store, cfg)
    mesh = build_mesh(res, provenance="flow")
    budget.nodes += len(mesh.nodes)
    mesh_text, valid_ids = render_mesh(mesh, store)
    ids_ranked = list(mesh.node_ids)

    # termination: depth / budget / sufficient stop-gate
    stop = depth >= budget.max_depth or budget.exhausted()
    if not stop:
        stop = sufficient(task, mesh_text, llm=llm)["stop"]; budget.llm_calls += 1
    if not stop:
        dec = decompose(task, mesh_text, llm=llm, max_subtasks=MAX_SUBTASKS); budget.llm_calls += 1
        subtasks = dec.get("subtasks") if not dec.get("done") else None
    else:
        subtasks = None

    children: list[TreeNode] = []
    for st in (subtasks or []):
        if budget.exhausted(): break
        sub = interpret(st, store, llm=llm); budget.llm_calls += 1
        child = solve(st, sub["seeds"] or seed_ids, (mesh, res), depth + 1, store, cfg, llm, budget)
        children.append(child)

    ans = synthesize(task, mesh_text, [c.answer for c in children], valid_ids, llm=llm)
    budget.llm_calls += 1
    return TreeNode(task, list(seed_ids), ans, depth, ids_ranked, children)

# answer: query → reasoning tree. interpret grounds the seeds in S; solve recurses (§1).
def answer(query, store, cfg: FieldConfig = DEFAULT_CFG, *, llm=call_json, budget: Budget | None = None) -> TreeNode:
    budget = budget or Budget()
    interp = interpret(query, store, llm=llm); budget.llm_calls += 1
    seeds = interp["seeds"]
    if not seeds:
        return TreeNode(query, [], {"prose": "No matching concepts found in the graph.",
                                    "citations": []}, 0, [], [])
    return solve(interp["intent"], seeds, None, 0, store, cfg, llm, budget)

# render_tree: human-readable trace of the solve tree (for inspection / 📍5).
def render_tree(node: TreeNode, store=None, indent: int = 0) -> str:
    pad = "  " * indent
    head = f"{pad}[d{node.depth}] {node.task}"
    mesh = f"{pad}   mesh({len(node.mesh_ids)}): {', '.join(node.mesh_ids[:6])}"
    prose = f"{pad}   → {node.answer.get('prose','')[:160]}"
    cites = f"{pad}   cites: {node.answer.get('citations', [])}"
    out = [head, mesh, prose, cites]
    for c in node.children: out.append(render_tree(c, store, indent + 1))
    return "\n".join(out)
