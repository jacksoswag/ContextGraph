from __future__ import annotations
# One real-LLM end-to-end run for inspection: respond(query) against the real graph S + Ollama,
# then write the answer + the gathered facts. Single-shot (no decomposition tree) — the pipeline is
# prompt → 3B refine + spaCy seeds → grow+q gather → one 3B answer.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/e2e_smoke.py "your query"
import os, sys, time
from pathlib import Path
import torch; torch.set_num_threads(1)
from field.loop import respond, GATHER_CFG
from graph import GraphStore
import llm as llm_mod

STORE = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
OUT = Path("runs_experiment/e2e_trace.md")

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is gravity and how does it relate to weight?"
    llm_mod.ensure_ollama_models()                   # populate tier fallback cascade
    store = GraphStore(STORE)
    t0 = time.time()
    r = respond(query, store, GATHER_CFG)
    dt = time.time() - t0
    def _t(nid): return (store.text(nid) or nid).split("|")[0]
    facts = [m for m in r.mesh_ids if m.startswith("e_")]
    md = [f"# E2E trace — {query!r}", "",
          f"Real Ollama backend. {dt:.1f}s · seeds: {[_t(s) for s in r.seeds]} · "
          f"{len(r.mesh_ids)} gathered nodes ({len(facts)} facts).", "",
          "## Answer", "", r.answer.get("prose", ""), "",
          f"**Citations:** {[_t(c) for c in r.answer.get('citations', [])]}", "",
          "## Gathered facts (top 20)", "", *[f"- {_t(f)}" for f in facts[:20]], ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"{dt:.1f}s | seeds={[_t(s) for s in r.seeds]} | {len(r.mesh_ids)} nodes")
    print(r.answer.get("prose", ""))
    print(f"\nTrace → {OUT}")
    store.close()

if __name__ == "__main__":
    main()
