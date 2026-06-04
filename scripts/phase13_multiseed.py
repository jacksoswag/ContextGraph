from __future__ import annotations
# Phase 1.3 — multi-seed/relational retrieval: field vs recursive PPR on 2-hop gold queries.
# Metric: gold coverage = fraction of gold answers present in the retrieved fact set (no LLM needed).
# Uses wikidata store (has full embeddings). The 2-hop bucket is the discriminator — field's
# multi-seed expansion should surface bridge facts that single-hop RAG and PPR miss.
# Run: PYTHONPATH=src:scripts python scripts/phase13_multiseed.py [--store PATH] [--n N]
import argparse, dataclasses, sys, statistics as st
from collections import defaultdict
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.store import GraphStore
from field.config import DEFAULT_CFG
from field.gather import materialize, gather, build_mesh
from field.loop import GATHER_CFG, gather_context
from wd_qa import build_qa, build_multihop, rich_subjects, coverage, _name, _clean

import numpy as np, torch
from embed import embed, unpack

def _grow_seeds(store, query, k=2):
    cands = [c for c in store.find_vec(query, 6) if not c.startswith("e_")]
    return cands[:k] or store.find_vec(query, 1)

def e_facts(store, node_ids):
    return [t for nid in node_ids if nid.startswith("e_") and (t := store.text(nid))]

CFG_STRUCT = dataclasses.replace(GATHER_CFG, couple_mode="structural")

def ctx_field(store, query):
    mesh = gather_context(store, query, _grow_seeds(store, query), GATHER_CFG)
    return e_facts(store, mesh.node_ids) if mesh else []

def ctx_field_struct(store, query):                        # Gate 2: structural coupling (no semantic cos)
    mesh = gather_context(store, query, _grow_seeds(store, query), CFG_STRUCT)
    return e_facts(store, mesh.node_ids) if mesh else []

def ctx_mesh(store, query):                                # recursive collapse-to-mesh (multi-hop spine)
    from field.loop import mesh_gather
    m = mesh_gather(store, _grow_seeds(store, query))
    return [store.text(e) for e in m.node_ids if (store.text(e))]

def ctx_ppr(store, query, recursive=False):
    from field.coupling import build as build_coupling
    from field.baseline import personalized_pagerank
    seeds = _grow_seeds(store, query)
    ep, si = materialize(store, seeds, GATHER_CFG)
    if not ep.node_ids: return []
    W = build_coupling(ep, GATHER_CFG).sym
    r = personalized_pagerank(W, si)
    if recursive:
        order = sorted(range(len(ep.node_ids)), key=lambda i: -float(r[i]))
        boost = [i for i in order if not ep.node_ids[i].startswith("e_")][:5]
        r = personalized_pagerank(W, list(dict.fromkeys(si + boost)))
    eidx = sorted((i for i in range(len(ep.node_ids)) if ep.node_ids[i].startswith("e_")),
                  key=lambda i: -float(r[i]))
    return e_facts(store, [ep.node_ids[i] for i in eidx[:GATHER_CFG.target_size]])

CONDITIONS = {
    "field":        ctx_field,
    "mesh":         ctx_mesh,
    "ppr":          lambda s, q: ctx_ppr(s, q, False),
    "pprR":         lambda s, q: ctx_ppr(s, q, True),
}

def run(store_path, n_subjects=40, per_subject=3, seed=42):
    store = GraphStore(store_path)
    print(f"Store: {store_path}")

    # 1-hop fact QA
    fact_qs = []
    fact_qs.extend(build_qa(store, n_subjects=n_subjects, per_subject=2, seed=seed))
    # 2-hop multihop QA
    hop2_qs = list(build_multihop(store, n_subjects=n_subjects, per_subject=per_subject, seed=seed))

    print(f"Questions: {len(fact_qs)} 1-hop, {len(hop2_qs)} 2-hop\n")

    results = defaultdict(lambda: defaultdict(list))   # [qtype][cond] -> [cov scores]

    for qtype, qs in [("1hop", fact_qs), ("2hop", hop2_qs)]:
        for q in qs:
            query = q.get("question") or q.get("q") or ""
            gold = q.get("gold") or []
            if not query or not gold: continue
            for cname, fn in CONDITIONS.items():
                try:
                    facts = fn(store, query)
                    answer_str = " ".join(facts)       # coverage expects a string
                    hit, total, _ = coverage(answer_str, gold)
                    cov = hit / max(total, 1)
                    results[qtype][cname].append(cov)
                except Exception as e:
                    print(f"  ERROR {cname} {qtype}: {e}"); results[qtype][cname].append(0.0)
        print(f"\n── {qtype} results (n={len(qs)}) ──")
        print(f"  {'cond':<10} mean_cov  median  p25   p75")
        for cname in CONDITIONS:
            scores = results[qtype][cname]
            if not scores: continue
            print(f"  {cname:<10} {st.mean(scores):.3f}    {st.median(scores):.3f}  "
                  f"{sorted(scores)[len(scores)//4]:.3f}  {sorted(scores)[3*len(scores)//4]:.3f}")

    # The discriminator: 2-hop field vs pprR delta
    for qtype in ("1hop", "2hop"):
        f = results[qtype].get("field", []); r = results[qtype].get("pprR", [])
        if f and r and len(f) == len(r):
            delta = [a - b for a, b in zip(f, r)]
            print(f"\nfield - pprR delta ({qtype}): mean={st.mean(delta):+.3f}  "
                  f"field_wins={sum(1 for d in delta if d>0)}/{len(delta)}")
    store.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.wikidata.sqlite")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    run(a.store, n_subjects=a.n, per_subject=a.per)
