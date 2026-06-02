from __future__ import annotations
# Phase 5 — one real-LLM end-to-end run for inspection (G5 / 📍5). Runs answer(query) against the
# real graph S with the real Ollama backend, then writes the full decomposition-tree trace.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/e2e_smoke.py "your query"
import os, sys, time
from pathlib import Path
import torch
torch.set_num_threads(1)
from field.config import DEFAULT_CFG
from field.loop import answer, Budget, render_tree
from graph import GraphStore
import llm as llm_mod

STORE = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
OUT = Path("runs_experiment/e2e_trace.md")

def render_tree_md(node, store, depth=0):
    pad = "  " * depth
    def _t(nid): return (store.text(nid) or nid).split("|")[0]
    lines = [f"{pad}- **d{node.depth}** `{node.task}`  ·seeds: {[ _t(s) for s in node.seeds]}",
             f"{pad}  - mesh({len(node.mesh_ids)}): {', '.join(_t(m) for m in node.mesh_ids[:8])}",
             f"{pad}  - answer: {node.answer.get('prose','')[:280]}",
             f"{pad}  - citations: {[_t(c) for c in node.answer.get('citations', [])]}"]
    for c in node.children: lines.append(render_tree_md(c, store, depth + 1))
    return "\n".join(lines)

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is gravity and how does it relate to weight?"
    llm_mod.ensure_ollama_models()                   # populate tier fallback cascade
    store = GraphStore(STORE)
    budget = Budget()
    t0 = time.time()
    root = answer(query, store, DEFAULT_CFG, budget=budget)
    dt = time.time() - t0
    trace = render_tree_md(root, store)
    md = [f"# E2E trace — {query!r}", "",
          f"Real Ollama backend. {dt:.1f}s, {budget.llm_calls} LLM calls, "
          f"{budget.nodes} gathered nodes.", "",
          "## Final answer", "", root.answer.get("prose", ""), "",
          f"**Citations:** {[ (store.text(c) or c).split('|')[0] for c in root.answer.get('citations', []) ]}", "",
          "## Decomposition tree", "", trace, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"{dt:.1f}s | {budget.llm_calls} LLM calls | {budget.nodes} nodes")
    print(render_tree(root, store))
    print(f"\nTrace → {OUT}")
    store.close()

if __name__ == "__main__":
    main()
