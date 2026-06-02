from __future__ import annotations
# Hyperedge containment measurement (Layer-2 validation). Ingests nested-clause facts
# (subject --verb--> [inner fact]) into a fresh store, then for each outer subject
# gathers with containment ON (w_hyper>0) vs OFF (=0) and measures whether the reified
# fact's CONTENT (the inner edge's child endpoints) is recalled in the top-K mesh.
# Without containment the content is unreachable from the subject; the number is the point.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/hyperedge_eval.py
import dataclasses, os, sys, tempfile
from pathlib import Path
import torch; torch.set_num_threads(1)
from graph.writer import GraphWriter, node_id, edge_id
from graph import GraphStore
from field.config import DEFAULT_CFG
from field.gather import materialize, gather, build_mesh

OUT = Path("runs_experiment/hyperedge_eval.md")
# (outer_subject, verb, inner_subject, inner_verb, inner_object)
FACTS = [
    ("scientist", "believe", "smoking", "cause", "cancer"),
    ("government", "announce", "economy", "enter", "recession"),
    ("study", "show", "exercise", "reduce", "mortality"),
    ("court", "rule", "law", "violate", "constitution"),
    ("report", "confirm", "company", "breach", "contract"),
    ("teacher", "explain", "gravity", "bend", "spacetime"),
    ("witness", "claim", "driver", "ignore", "signal"),
    ("model", "predict", "warming", "raise", "sea level"),
]

def _node(t): return {"type": "node", "text": t, "pos": "NOUN"}
def _edge(s, r, t): return {"type": "edge", "rel": r, "source": s, "target": t,
                            "_source_text": "x", "_clause_text": "x"}

def content_recall(store, subj, inner_s, inner_o, cfg, k=12):
    ids = store.find(subj, 1)
    if not ids: return None
    ep, si = materialize(store, [ids[0]], cfg)
    res = gather(ep, si, cfg)
    mesh = build_mesh(res, top_k=k)
    got = set(mesh.node_ids)
    want = {node_id(inner_s), node_id(inner_o)}
    return len(want & got) / len(want)

def main():
    path = os.path.join(tempfile.mkdtemp(), "hyper.sqlite")
    with GraphWriter(path) as w:
        for os_, v, is_, iv, io in FACTS:
            w.write_clauses([_edge(_node(os_), v, _edge(_node(is_), iv, _node(io)))])
    cfg_on = DEFAULT_CFG
    cfg_off = dataclasses.replace(DEFAULT_CFG, w_hyper=0.0)
    rows = []
    with GraphStore(path) as store:
        for os_, v, is_, iv, io in FACTS:
            on = content_recall(store, os_, is_, io, cfg_on)
            off = content_recall(store, os_, is_, io, cfg_off)
            if on is None: continue
            rows.append((os_, v, f"{is_} {iv} {io}", off, on))
    def mean(j): return sum(r[j] for r in rows) / len(rows) if rows else 0.0
    lines = ["# Hyperedge Containment Measurement (Layer 2)", "",
             f"{len(rows)} nested facts, fresh store, top-12 mesh. Content recall = fraction of the "
             f"inner fact's {{subject, object}} present in the mesh when seeding the outer subject.", "",
             "| outer seed | relation | reified fact | recall OFF (w_hyper=0) | recall ON |",
             "|---|---|---|---|---|"]
    for os_, v, fact, off, on in rows:
        lines.append(f"| {os_} | {v} | {fact} | {off:.2f} | {on:.2f} |")
    lines += ["",
        f"**Content recall:** {mean(3):.3f} (containment OFF) → {mean(4):.3f} (ON). "
        f"Off-state recall is ~0 because the inner fact's nodes are unreachable from the outer "
        f"subject by k-hop alone; the hyperedge binding is what makes the reified fact's content "
        f"part of the gather.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(lines))
    print(f"content recall OFF→ON: {mean(3):.3f} → {mean(4):.3f}  ({len(rows)} facts)")
    print(f"Report → {OUT}")

if __name__ == "__main__":
    main()
