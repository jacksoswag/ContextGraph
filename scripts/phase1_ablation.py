from __future__ import annotations
# Phase 1.2 ablation: couple_mode=semantic vs structural, with and without struct_edge_weights.
# Requires info_vectors to be populated (run after background embed completes).
# Reports: top-5 recall of known-related concepts, seed-hottest rate, relevance concentration.
# Run: PYTHONPATH=src python scripts/phase1_ablation.py [--store PATH]
import argparse, dataclasses, sys
from pathlib import Path
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from graph.store import GraphStore
from field.config import DEFAULT_CFG
from field.gather import gather_from_store, readout

QUERIES = [
    ("black hole",       ["event horizon", "singularity", "hawking", "schwarzschild", "accretion"]),
    ("neutron star",     ["pulsar", "supernova", "quark", "magnetar", "binary"]),
    ("gravitational wave",["ligo", "merger", "binary", "interferometer", "chirp"]),
    ("dark matter",      ["halo", "wimps", "baryon", "galaxy", "rotation curve"]),
    ("cosmic ray",       ["proton", "muon", "atmosphere", "flux", "spectrum"]),
    ("dark energy",      ["cosmological constant", "expansion", "vacuum", "lambda", "acceleration"]),
    ("supernova",        ["stellar", "iron", "collapse", "remnant", "shock"]),
    ("redshift",         ["doppler", "expansion", "hubble", "spectral", "blueshift"]),
]

def cfg(**kw):
    return dataclasses.replace(DEFAULT_CFG, N_max=200, H_max=3000, eps_x=1e-3,
                                decay=1.5, decay_gamma=1.0, **kw)

CONDITIONS = [
    ("structural      ", cfg(couple_mode="structural"), False),
    ("structural+ew   ", cfg(couple_mode="structural"), True),
    ("semantic        ", cfg(couple_mode="semantic"),   False),
    ("semantic+ew     ", cfg(couple_mode="semantic"),   True),
]

def run(store_path):
    store = GraphStore(store_path)
    # check embeddings present
    import sqlite3; con = sqlite3.connect(store_path)
    with_vec = con.execute("SELECT COUNT(*) FROM nodes WHERE info_vector IS NOT NULL").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    con.close()
    if with_vec < total * 0.8:
        print(f"WARNING: only {with_vec}/{total} nodes have embeddings — semantic coupling may be weak")

    ew = store.struct_edge_weights()
    totals = {cname: {"recall": 0, "seed_hot": 0, "top3pct": 0.0, "n": 0}
              for cname, _, _ in CONDITIONS}

    print(f"\n{'query':<24} {'condition':<18} N   seed_hot  recall  top3%")
    for qname, related in QUERIES:
        hits = store.find_vec(qname, 1) or store.find(qname, 1)
        if not hits: print(f"{qname:<24} -- no seed"); continue
        for cname, c, use_ew in CONDITIONS:
            w = ew if use_ew else None
            res = gather_from_store(store, hits, c, weights=w)
            mesh = readout(res)
            rel = res.relevance()
            seed_hot = int(rel.argmax()) in res.seed_idx
            top5_texts = {store.text(nid) or "" for nid in mesh.node_ids[:8]}
            recall = sum(1 for r in related
                         if any(r in t for t in top5_texts)) / len(related)
            top3pct = sum(sorted(rel.numpy().tolist(), reverse=True)[:3]) / (float(rel.sum()) + 1e-9)
            print(f"{qname:<24} {cname} {len(res.ep.node_ids):3d}  {'Y' if seed_hot else 'N':8}  "
                  f"{recall:.2f}    {top3pct:.3f}")
            t = totals[cname]
            t["recall"] += recall; t["seed_hot"] += int(seed_hot)
            t["top3pct"] += top3pct; t["n"] += 1
        print()

    print(f"\n{'SUMMARY':<24} {'condition':<18} seed_hot%  recall  top3%")
    for cname, _, _ in CONDITIONS:
        t = totals[cname]; n = t["n"] or 1
        print(f"{'':24} {cname} {100*t['seed_hot']/n:7.0f}%  "
              f"{t['recall']/n:.3f}   {t['top3pct']/n:.3f}")
    store.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.astro.sqlite")
    run(ap.parse_args().store)
