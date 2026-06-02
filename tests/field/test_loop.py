from __future__ import annotations
# Phase 5 — the recursion (solve tree) + e2e with a mock LLM (spec §1, §5).
import numpy as np, torch
import torch.nn.functional as F
import pytest
from field.config import DEFAULT_CFG
from field.gather import build_mesh
from field.loop import answer, solve, Budget, TreeNode, _gather_node, _child_anchors, render_tree

# two clustered triangles A/B bridged a1-b1; anchors clustered per group, find() maps words→nodes
class _LoopStore:
    def __init__(self, adj, anc, find_map):
        self._adj = adj; self._anc = anc; self._find = find_map
    def neighbors(self, nid): return list(self._adj.get(nid, []))
    def anchor(self, nid): return self._anc.get(nid)
    def text(self, nid): return nid
    def find(self, q, k=5): return self._find.get(q.lower().strip(), [])[:k]

def _cluster_store(seed=3):
    g = torch.Generator().manual_seed(seed)
    A, B = ["a1", "a2", "a3"], ["b1", "b2", "b3"]
    adj = {n: [] for n in A + B}
    def link(u, v, w=1.0): adj[u].append((v, w)); adj[v].append((u, w))
    for i in range(3):
        for j in range(i + 1, 3): link(A[i], A[j]); link(B[i], B[j])
    link("a1", "b1", 0.5)
    baseA = F.normalize(torch.randn(384, generator=g), dim=0)
    baseB = F.normalize(torch.randn(384, generator=g), dim=0)
    anc = {}
    for n in A: anc[n] = F.normalize(baseA + 0.05 * torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
    for n in B: anc[n] = F.normalize(baseB + 0.05 * torch.randn(384, generator=g), dim=0).numpy().astype(np.float32)
    find_map = {"alpha": ["a1"], "beta": ["b1"], "a1": ["a1"], "b1": ["b1"], "the goal": ["a1"]}
    return _LoopStore(adj, anc, find_map)

# scripted mock LLM: routes by prompt template; sufficient never stops, decompose always splits
# (forces recursion to the depth/budget limit), so termination must come from the loop, not the LLM.
def _always_recurse(prompt, model):
    if "entry concepts" in prompt: return {"entities": ["alpha"], "intent": "about alpha"}
    if "sub-questions" in prompt: return {"subtasks": ["beta", "alpha"]}
    if "enough to answer" in prompt: return {"stop": False, "reason": "more"}
    if "Answer the task" in prompt: return {"prose": "synth", "citations": []}
    return {}

def _max_depth(node): return node.depth if not node.children else max(_max_depth(c) for c in node.children)
def _count(node): return 1 + sum(_count(c) for c in node.children)

# ── termination: recursion halts even when the LLM never stops ────────────────────────

def test_recursion_halts_at_max_depth(mock_llm):
    store = _cluster_store(); llm = mock_llm(_always_recurse)
    root = answer("the goal", store, DEFAULT_CFG, llm=llm, budget=Budget(max_depth=2))
    assert _max_depth(root) <= 2                      # never recurses past MAX_DEPTH
    assert _count(root) >= 2                          # but it DID build a tree

def test_recursion_halts_on_budget(mock_llm):
    store = _cluster_store(); llm = mock_llm(_always_recurse)
    b = Budget(max_depth=5, max_llm_calls=6)
    answer("the goal", store, DEFAULT_CFG, llm=llm, budget=b)
    assert b.exhausted()                             # stopped by the LLM-call ceiling, not depth

def test_node_budget_halts(mock_llm):
    store = _cluster_store(); llm = mock_llm(_always_recurse)
    b = Budget(max_depth=5, max_nodes=5)
    answer("the goal", store, DEFAULT_CFG, llm=llm, budget=b)
    assert b.nodes >= 5 and b.exhausted()

# ── budget accounting: a finite tree, calls bounded near the ceiling ─────────────────

def test_llm_calls_bounded(mock_llm):
    store = _cluster_store(); llm = mock_llm(_always_recurse)
    b = Budget(max_depth=2, max_llm_calls=40)
    answer("the goal", store, DEFAULT_CFG, llm=llm, budget=b)
    assert b.llm_calls == len(llm.calls)             # every LLM call is counted
    assert b.llm_calls <= 40 + 4                     # bounded (small per-node slack)

# ── parent-anchoring: a child gather is measurably pulled toward the parent region ────

def test_parent_anchoring_pulls_child_toward_parent():
    store = _cluster_store(); cfg = DEFAULT_CFG
    parent_res = _gather_node(["a1"], {}, store, cfg)
    parent_mesh = build_mesh(parent_res)
    inherited = _child_anchors((parent_mesh, parent_res), cfg)
    with_inh = _gather_node(["b1"], inherited, store, cfg)
    without = _gather_node(["b1"], {}, store, cfg)
    a_top = parent_mesh.node_ids[0]                  # parent's most-relevant node (a-cluster)
    iw, io = with_inh.ep.id_to_idx.get(a_top), without.ep.id_to_idx.get(a_top)
    rel_w = float((with_inh.X_star[iw] ** 2).sum()) if iw is not None else 0.0
    rel_o = float((without.X_star[io] ** 2).sum()) if io is not None else 0.0
    assert rel_w > rel_o + 1e-3, f"inheritance did not pull child toward parent: {rel_w} vs {rel_o}"

# ── e2e shape: bounded tree with a synthesized answer carrying citations ──────────────

def test_e2e_tree_shape(mock_llm):
    store = _cluster_store()
    def resp(prompt, model):
        if "entry concepts" in prompt: return {"entities": ["alpha"], "intent": "explain alpha"}
        if "sub-questions" in prompt: return {"subtasks": ["beta"]}
        if "enough to answer" in prompt: return {"stop": False, "reason": ""}
        if "Answer the task" in prompt:
            # cite a real id present in the rendered mesh (line "[id] ...")
            import re
            ids = re.findall(r"\[(\w+)\]", prompt)
            return {"prose": "an answer", "citations": ids[:1]}
        return {}
    root = answer("explain alpha", store, DEFAULT_CFG, llm=mock_llm(resp), budget=Budget(max_depth=2))
    assert isinstance(root, TreeNode)
    assert root.answer["prose"] == "an answer"
    assert root.children and root.children[0].depth == 1       # recursed once
    assert all(c in root.mesh_ids for c in root.answer["citations"])  # citations grounded in mesh

def test_e2e_no_seeds_returns_graceful(mock_llm):
    store = _cluster_store()
    llm = mock_llm(lambda p, m: {"entities": ["nonexistent"], "intent": "x"})
    root = answer("unknown topic", store, DEFAULT_CFG, llm=llm)
    assert root.children == [] and "No matching concepts" in root.answer["prose"]

# ── determinism: same store + same scripted LLM ⇒ identical tree ──────────────────────

def test_e2e_deterministic(mock_llm):
    store = _cluster_store()
    a = answer("the goal", store, DEFAULT_CFG, llm=mock_llm(_always_recurse), budget=Budget(max_depth=2))
    b = answer("the goal", store, DEFAULT_CFG, llm=mock_llm(_always_recurse), budget=Budget(max_depth=2))
    assert render_tree(a) == render_tree(b)
