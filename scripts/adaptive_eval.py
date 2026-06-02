from __future__ import annotations
# Adaptive-domain measurement: does the moving front pull MUCH more relevant context than the fixed
# gather, with bounded live compute, and how does reach trade off against selectivity vs decay λ?
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/adaptive_eval.py
import os, dataclasses, time, statistics as st
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
from field.config import DEFAULT_CFG
from field.domain import TreeContext, adaptive_gather, DomainConfig
from field.gather import materialize, gather, build_mesh
from graph import GraphStore

STORE = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
OUT = Path("runs_experiment/adaptive_eval.md")
SEEDS = ["gravity", "france", "water", "music", "computer", "ocean"]
LAMBDAS = [0.3, 0.5, 0.7, 1.0, 1.5]

def precision(ctx, ids, seed_id, k=None):
    s = ctx.anchor(seed_id)
    if s is None: return 0.0
    s = s / (np.linalg.norm(s) + 1e-9)
    vals = []
    for nid in (ids[:k] if k else ids):
        if nid == seed_id: continue
        a = ctx.anchor(nid)
        if a is not None: vals.append(float(a @ s / (np.linalg.norm(a) + 1e-9)))
    return st.mean(vals) if vals else 0.0

def main():
    store = GraphStore(STORE)
    seeds = [store.find(w, 1)[0] for w in SEEDS if store.find(w, 1)]
    lines = ["# Adaptive-domain measurement", "",
             f"Store `{STORE}`. {len(seeds)} seeds. Fixed gather = current default "
             f"(λ=1.5, N_max=512). Adaptive = moving front (max_live=200, ttl=3).", ""]

    # fixed-gather reference (the current system)
    print("=== fixed gather (reference) ===")
    fx_mesh, fx_prec = [], []
    for sid in seeds:
        ep, si = materialize(store, [sid], dataclasses.replace(DEFAULT_CFG, k_hop=2, N_max=512, H_max=6000))
        m = build_mesh(gather(ep, si, dataclasses.replace(DEFAULT_CFG, k_hop=2, N_max=512, H_max=6000)))
        ctx = TreeContext(store)
        fx_mesh.append(len(m.nodes)); fx_prec.append(precision(ctx, m.node_ids, sid))
        print(f"  {store.text(sid).split('|')[0][:12]:<12} mesh={len(m.nodes):3d} prec={fx_prec[-1]:.3f}")
    lines += [f"**Fixed gather:** mean mesh {st.mean(fx_mesh):.0f} nodes, precision {st.mean(fx_prec):.3f}.", "",
              "## Adaptive front vs decay λ (mean over seeds)", "",
              "| λ | committed | peak_live | reach | loaded | precision | prec@top30 | time |",
              "|---|---|---|---|---|---|---|---|"]

    print("\n=== adaptive front, λ sweep ===")
    for lam in LAMBDAS:
        cfg = dataclasses.replace(DEFAULT_CFG, decay=lam, H_max=4000)
        dcfg = DomainConfig(max_live=200, loads_per_phase=80, max_phases=14, anchor_ttl=3)
        comm, peak, reach, loaded, prec, prec30, secs = [], [], [], [], [], [], []
        for sid in seeds:
            ctx = TreeContext(store)
            t0 = time.time(); res = adaptive_gather(ctx, [sid], cfg, dcfg); dt = time.time() - t0
            comm.append(len(res.committed)); peak.append(res.peak_live)
            reach.append(res.trace[-1]["max_reach"]); loaded.append(res.total_loaded)
            prec.append(precision(ctx, res.committed, sid)); prec30.append(precision(ctx, res.committed, sid, k=30))
            secs.append(dt)
        row = (st.mean(comm), st.mean(peak), st.mean(reach), st.mean(loaded),
               st.mean(prec), st.mean(prec30), st.mean(secs))
        lines.append(f"| {lam} | {row[0]:.0f} | {row[1]:.0f} | {row[2]:.1f} | {row[3]:.0f} | "
                     f"{row[4]:.3f} | {row[5]:.3f} | {row[6]:.1f}s |")
        print(f"  λ={lam}: committed={row[0]:.0f} peak_live={row[1]:.0f} reach={row[2]:.1f} "
              f"loaded={row[3]:.0f} prec={row[4]:.3f} prec@30={row[5]:.3f} {row[6]:.1f}s")

    lines += ["", "## Per-phase trace (gravity, λ=0.7) — front advancing under a bounded live window", "",
              "| phase | live | committed | loaded | culled | reach |", "|---|---|---|---|---|---|"]
    ctx = TreeContext(store)
    res = adaptive_gather(ctx, [store.find("gravity", 1)[0]],
                          dataclasses.replace(DEFAULT_CFG, decay=0.7, H_max=4000),
                          DomainConfig(max_live=200, loads_per_phase=80, max_phases=14, anchor_ttl=3))
    for p in res.trace:
        lines.append(f"| {p['phase']} | {p['live']} | {p['committed']} | {p['loaded']} | {p['culled']} | {p['max_reach']} |")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(lines))
    print(f"\nReport → {OUT}")
    store.close()

if __name__ == "__main__":
    main()
