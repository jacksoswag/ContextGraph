from __future__ import annotations
# Phase 6 validation (G6 + the 📍3-deferred verdict). Trains Sleep (StructureRecon) on real seeds,
# then asks two questions:
#   (1) G6: does the learned-C gather beat the bootstrap-C gather (semantic precision / held-out recall)?
#   (2) 📍3: does the NONLINEAR gather still beat LINEAR PPR on the SAME learned weights? — if PPR
#       on learned C matches the gather, the field's dynamics add nothing → switch to PPR.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/sleep_eval.py
import os, dataclasses, random, statistics as st
from pathlib import Path
import torch
torch.set_num_threads(1)
from field.config import DEFAULT_CFG
from field.coupling import build as build_coupling
from field.gather import materialize, gather, edge_weights, hop_distances
from field.episode import Episode
from field.baseline import personalized_pagerank
from field.sleep import train
from graph import GraphStore

STORE = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
OUT = Path("runs_experiment/sleep_eval.md")
SEEDS = ["dog", "france", "water", "music", "gravity", "computer", "ocean", "tree", "money", "language"]
K, N_MAX, K_HOP = 10, 250, 2
random.seed(0)

def topk(score, si, k=K):
    return sorted((i for i in range(len(score)) if i not in set(si)), key=lambda i: -float(score[i]))[:k]

def precision(ep, idxs, si):
    if not idxs: return 0.0
    A = ep.A.float(); s = A[si[0]]
    return float(st.mean(float((A[i] * s).sum()) for i in idxs))

def reduce_edges(ep, drop):
    s, d = ep.edge_index.tolist()
    keep = [(a, b) for a, b in zip(s, d) if (a, b) not in drop and (b, a) not in drop]
    ei = torch.tensor(list(zip(*keep)), dtype=torch.long) if keep else torch.zeros(2, 0, dtype=torch.long)
    return Episode(ep.node_ids, ep.A, ei, ep.id_to_idx)

def heldout_recall(ep, si, cfg, kind, w=None):
    s, d = ep.edge_index.tolist()
    nbrs = [n for n in ({b for a, b in zip(s, d) if a == si[0]} | {a for a, b in zip(s, d) if b == si[0]})
            if n not in set(si)]
    if len(nbrs) < 4: return None
    hide = set(random.sample(nbrs, max(1, int(0.3 * len(nbrs)))))
    ep_r = reduce_edges(ep, {(si[0], h) for h in hide})
    gt = [h for h in hide if hop_distances(ep_r, si)[h] >= 0]
    if not gt: return None
    if kind == "gather": score = gather(ep_r, si, cfg, weights=w).relevance()
    else: score = personalized_pagerank(build_coupling(ep_r, dataclasses.replace(cfg, eta=1e-10),
                                                        edge_weights(ep_r, w)).sym, si)
    return len(set(topk(score, si)) & set(gt)) / len(gt)

def main():
    store = GraphStore(STORE)
    seeds = [store.find(s, 1)[0] for s in SEEDS if store.find(s, 1)]
    decay = float(os.environ.get("DI_DECAY", DEFAULT_CFG.decay))
    cfg = dataclasses.replace(DEFAULT_CFG, k_hop=K_HOP, N_max=N_MAX, H_max=8000, decay=decay)
    print(f"operating point: decay λ={decay}")
    print(f"training Sleep on {len(seeds)} seeds…")
    w = train(store, seeds, cfg, epochs=4, lr=0.2, trust=0.5)
    print(f"learned {len(w)} edge multipliers")

    rows = []
    for sid in seeds:
        ep, si = materialize(store, [sid], cfg)
        gb = gather(ep, si, cfg).relevance()                       # bootstrap gather
        gl = gather(ep, si, cfg, weights=w).relevance()            # learned gather
        Cb = build_coupling(ep, dataclasses.replace(cfg, eta=1e-10))
        Cl = build_coupling(ep, dataclasses.replace(cfg, eta=1e-10), edge_weights(ep, w))
        pb = personalized_pagerank(Cb.sym, si)                     # PPR bootstrap
        pl = personalized_pagerank(Cl.sym, si)                     # PPR on learned C
        rows.append({"seed": store.text(sid).split("|")[0][:14],
                     "g_boot": precision(ep, topk(gb, si), si), "g_learn": precision(ep, topk(gl, si), si),
                     "ppr_boot": precision(ep, topk(pb, si), si), "ppr_learn": precision(ep, topk(pl, si), si),
                     "rec_boot": heldout_recall(ep, si, cfg, "gather"),
                     "rec_learn": heldout_recall(ep, si, cfg, "gather", w)})
        print(f"  {rows[-1]['seed']:<14} gΔ={rows[-1]['g_learn']-rows[-1]['g_boot']:+.3f} "
              f"gather_learn={rows[-1]['g_learn']:.3f} ppr_learn={rows[-1]['ppr_learn']:.3f}")

    def mean(k): return st.mean(r[k] for r in rows if r[k] is not None)
    lines = ["# Sleep Evaluation (G6 + 📍3 verdict)", "",
             f"Store `{STORE}`. {len(seeds)} seeds, {len(w)} learned edge multipliers (trust 0.5).", "",
             "| seed | gather boot | gather learned | PPR boot | PPR learned | recall boot | recall learned |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['g_boot']:.3f} | {r['g_learn']:.3f} | {r['ppr_boot']:.3f} | "
                     f"{r['ppr_learn']:.3f} | {r['rec_boot']} | {r['rec_learn']} |")
    lines += ["",
        f"**G6 — does Sleep help the gather?** semantic precision@10 "
        f"{mean('g_boot'):.3f} (bootstrap) → {mean('g_learn'):.3f} (learned); "
        f"held-out recall {mean('rec_boot'):.3f} → {mean('rec_learn'):.3f}.",
        f"**📍3 — field vs PPR on learned weights:** gather(learned) {mean('g_learn'):.3f} "
        f"vs PPR(learned) {mean('ppr_learn'):.3f}.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(lines))
    print(f"\nG6: gather precision {mean('g_boot'):.3f}→{mean('g_learn'):.3f} | "
          f"recall {mean('rec_boot'):.3f}→{mean('rec_learn'):.3f}")
    print(f"📍3: gather(learned) {mean('g_learn'):.3f} vs PPR(learned) {mean('ppr_learn'):.3f}")
    print(f"Report → {OUT}")
    store.close()

if __name__ == "__main__":
    main()
