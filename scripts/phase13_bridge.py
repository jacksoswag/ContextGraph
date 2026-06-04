from __future__ import annotations
# Phase 1.3 (Gate 3, the PPR gate) — GENUINE multi-seed relational test. For each A -r1-> B -r2-> C
# path (wd_qa.build_multihop), seed on the TWO DISTINCT entities {A, C} and try to recover the BRIDGE
# B. B is the common neighbour of both seeds — the one place the field's nonlinear vector interference
# (constructive at a shared neighbourhood, §7) can beat PPR's scalar superposition of two independent
# personalized vectors. Metric = coverage of the bridge surface in the retrieved fact set (no LLM).
# Conditions: field2 (2-seed settle) · field2_struct (couple_mode=structural) · ppr2 (linear super-
# position) · pprR2 (recursive) · field1 (1-seed on A only — reference: does the 2nd seed matter?).
# Run: PYTHONPATH=src:scripts python scripts/phase13_bridge.py [--store PATH] [--n N]
import argparse, dataclasses, sys, statistics as st
from collections import defaultdict
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from graph.store import GraphStore
from field.config import DEFAULT_CFG
from field.gather import materialize, gather, build_mesh
from field.loop import GATHER_CFG
from field.coupling import build as build_coupling
from field.baseline import personalized_pagerank
from wd_qa import build_multihop, coverage

CFG = GATHER_CFG
CFG_STRUCT = dataclasses.replace(GATHER_CFG, couple_mode="structural")

def _ground(store, name):                                  # name -> a non-e_ node id, or None
    for c in (store.find_vec(name, 4) or []):
        if not c.startswith("e_"): return c
    return None

def e_facts(store, node_ids):
    return [t for nid in node_ids if nid.startswith("e_") and (t := store.text(nid))]

# one settle over the {A,C} (or {A}) episode; return the retrieved e_ fact surfaces
def field_ctx(store, seeds, cfg):
    ep, si = materialize(store, seeds, cfg)
    if not ep.node_ids or not si: return []
    res = gather(ep, si, cfg)
    mesh = build_mesh(res, top_k=cfg.target_size)
    return e_facts(store, mesh.node_ids)

def ppr_ctx(store, seeds, recursive=False):
    ep, si = materialize(store, seeds, CFG)
    if not ep.node_ids or not si: return []
    W = build_coupling(ep, CFG).sym
    r = personalized_pagerank(W, si)
    if recursive:
        order = sorted(range(len(ep.node_ids)), key=lambda i: -float(r[i]))
        boost = [i for i in order if not ep.node_ids[i].startswith("e_")][:5]
        r = personalized_pagerank(W, list(dict.fromkeys(si + boost)))
    eidx = sorted((i for i in range(len(ep.node_ids)) if ep.node_ids[i].startswith("e_")),
                  key=lambda i: -float(r[i]))
    return e_facts(store, [ep.node_ids[i] for i in eidx[:CFG.target_size]])

def run(store_path, n_subjects=60, per_subject=3, seed=42):
    store = GraphStore(store_path)
    print(f"Store: {store_path}")
    qs = list(build_multihop(store, n_subjects=n_subjects, per_subject=per_subject, seed=seed))
    res = defaultdict(list); n_used = 0
    for q in qs:
        A, gold_C, bridges = q["subj_id"], q["gold"], q.get("bridges") or []
        if not bridges: continue
        c_node = _ground(store, gold_C[0])
        if not c_node or c_node == A: continue              # need two DISTINCT grounded seeds
        seeds2 = [A, c_node]
        bgold = bridges                                     # recover the bridge B
        n_used += 1
        conds = {
            "field2":        lambda: field_ctx(store, seeds2, CFG),
            "field2_struct": lambda: field_ctx(store, seeds2, CFG_STRUCT),
            "ppr2":          lambda: ppr_ctx(store, seeds2, False),
            "pprR2":         lambda: ppr_ctx(store, seeds2, True),
            "field1":        lambda: field_ctx(store, [A], CFG),
        }
        for name, fn in conds.items():
            try:
                hit, total, _ = coverage(" ".join(fn()), bgold)
                res[name].append(hit / max(total, 1))
            except Exception as e:
                print(f"  ERR {name}: {e}"); res[name].append(0.0)
    print(f"relational bridge queries used: {n_used}\n")
    print(f"  {'cond':<16} bridge_cov  median  wins_vs_pprR2")
    base = res.get("pprR2", [])
    for name in ("field2", "field2_struct", "ppr2", "pprR2", "field1"):
        s = res[name]
        if not s: continue
        wins = sum(1 for a, b in zip(s, base) if a > b) if base and len(s) == len(base) else 0
        print(f"  {name:<16} {st.mean(s):.3f}      {st.median(s):.3f}   {wins}/{len(s)}")
    # paired field2 vs ppr2 (the linear-superposition discriminator)
    f, p = res.get("field2", []), res.get("ppr2", [])
    if f and p and len(f) == len(p):
        d = [a - b for a, b in zip(f, p)]
        print(f"\nfield2 - ppr2 (linear superposition): mean={st.mean(d):+.3f}  "
              f"field_wins={sum(1 for x in d if x>0)}/{len(d)}  ties={sum(1 for x in d if x==0)}")
    store.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.wikidata.sqlite")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    run(a.store, n_subjects=a.n, per_subject=a.per)
