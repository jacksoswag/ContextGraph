from __future__ import annotations
# Phase 3 — gather precision calibration (spec §11, 📍3). Sweeps the decay/β operating point and
# benchmarks the field gather against personalized PageRank on the SAME active-set graph.
# Ground truth (no labels needed):
#   • semantic precision@k = mean cosine(anchor(node), anchor(seed)) over top-k gathered nodes —
#     are the gathered concepts semantically related to the query (frozen embeddings, independent
#     of the cosine edge weights used to gather).
#   • held-out-edge recall@k = hide 30% of the seed's direct neighbors, then check whether the
#     method recovers them (via other paths) in its top-k — structural recovery / multi-hop value.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src venv/bin/python scripts/gather_calibration.py
import os, dataclasses, random, statistics as st
from pathlib import Path
import torch
torch.set_num_threads(1)
from field.config import DEFAULT_CFG
from field.coupling import build as build_coupling
from field.gather import materialize, gather, hop_distances
from field.episode import Episode
from field.baseline import personalized_pagerank
from graph import GraphStore

STORE = os.environ.get("DI_GRAPH_STORE", ".di-ui/graph.sleep.merged.sqlite")
OUT = Path("runs_experiment/gather_calibration.md")
SEED_WORDS = ["dog", "cat", "france", "paris", "water", "fire", "music", "gravity",
              "democracy", "computer", "ocean", "tree", "car", "money", "language"]
K = 10                      # top-k for precision/recall/overlap
N_MAX, K_HOP = 300, 2
LAMBDAS = [0.5, 1.0, 1.5, 2.0, 3.0]
ALPHAS = [0.1, 0.15, 0.3, 0.5]
random.seed(0)

def topk_idx(score, seed_idx, k):
    order = sorted((i for i in range(len(score)) if i not in set(seed_idx)),
                   key=lambda i: -float(score[i]))
    return order[:k]

def sem_precision(ep, idxs, seed_idx):
    if not idxs: return 0.0
    A = ep.A.float(); s = A[seed_idx[0]]
    return float(st.mean(float((A[i] * s).sum()) for i in idxs))

def reduce_edges(ep, drop_pairs):
    s, d = ep.edge_index.tolist()
    keep = [(a, b) for a, b in zip(s, d) if (a, b) not in drop_pairs and (b, a) not in drop_pairs]
    if not keep: ei = torch.zeros(2, 0, dtype=torch.long)
    else: ei = torch.tensor(list(zip(*keep)), dtype=torch.long)
    return Episode(ep.node_ids, ep.A, ei, ep.id_to_idx)

def heldout_recall(store, ep, seed_idx, cfg, method):
    # hide 30% of the seed's direct neighbors; recover via remaining structure
    s, d = ep.edge_index.tolist()
    nbrs = sorted({b for a, b in zip(s, d) if a == seed_idx[0]} |
                  {a for a, b in zip(s, d) if b == seed_idx[0]})
    nbrs = [n for n in nbrs if n not in set(seed_idx)]
    if len(nbrs) < 4: return None
    hide = set(random.sample(nbrs, max(1, int(0.3 * len(nbrs)))))
    drop = {(seed_idx[0], h) for h in hide}
    ep_r = reduce_edges(ep, drop)
    reach = hop_distances(ep_r, seed_idx)
    gt = [h for h in hide if reach[h] >= 0]                 # still reachable without the direct edge
    if not gt: return None
    if method == "gather":
        score = gather(ep_r, seed_idx, cfg).relevance()
    else:
        Cr = build_coupling(ep_r, dataclasses.replace(cfg, eta=1e-10))
        score = personalized_pagerank(Cr.sym, seed_idx, alpha=method)
    got = set(topk_idx(score, seed_idx, K))
    return len(got & set(gt)) / len(gt)

def main():
    store = GraphStore(STORE)
    seeds = [(w, store.find(w, 1)[0]) for w in SEED_WORDS if store.find(w, 1)]
    print(f"{len(seeds)} seeds resolved; sweeping λ={LAMBDAS} (β=2), PPR α={ALPHAS}")
    lines = ["# Gather Precision Calibration (Phase 3 / 📍3)", "",
             f"Store `{STORE}`. {len(seeds)} seeds, k_hop={K_HOP}, N_max={N_MAX}, top-k={K}.",
             "Semantic precision@k = mean cosine(top-k anchor, seed anchor). "
             "Held-out recall@k = fraction of 30%-hidden direct neighbors recovered in top-k.", ""]

    # pre-materialize each seed once (shared by gather sweep + PPR)
    mats = []
    for w, sid in seeds:
        ep, si = materialize(store, [sid], dataclasses.replace(DEFAULT_CFG, k_hop=K_HOP, N_max=N_MAX))
        mats.append((w, sid, ep, si))

    # ── λ sweep (gather) ──────────────────────────────────────────────────────────────
    lines += ["## 1. Decay (λ) sweep — gather", "",
              "| λ | settle steps | mesh(top≥1%) | sem.precision@10 | heldout recall@10 |",
              "|---|---|---|---|---|"]
    sweep_rows = {}
    for lam in LAMBDAS:
        cfg = dataclasses.replace(DEFAULT_CFG, decay=lam, k_hop=K_HOP, N_max=N_MAX, H_max=8000)
        steps, meshsz, prec, rec = [], [], [], []
        for w, sid, ep, si in mats:
            res = gather(ep, si, cfg)
            rel = res.relevance(); mx = float(rel.max())
            steps.append(res.steps)
            meshsz.append(int((rel >= 0.01 * mx).sum()))
            prec.append(sem_precision(ep, topk_idx(rel, si, K), si))
            r = heldout_recall(store, ep, si, cfg, "gather")
            if r is not None: rec.append(r)
        row = (st.mean(steps), st.mean(meshsz), st.mean(prec), st.mean(rec) if rec else float("nan"))
        sweep_rows[lam] = row
        lines.append(f"| {lam} | {row[0]:.0f} | {row[1]:.1f} | {row[2]:.3f} | {row[3]:.3f} |")
        print(f"  λ={lam}: steps={row[0]:.0f} mesh={row[1]:.1f} prec={row[2]:.3f} recall={row[3]:.3f}")

    # ── PPR α sweep ─────────────────────────────────────────────────────────────────────
    lines += ["", "## 2. Teleport (α) sweep — personalized PageRank", "",
              "| α | sem.precision@10 | heldout recall@10 |", "|---|---|---|"]
    ppr_rows = {}
    for a in ALPHAS:
        prec, rec = [], []
        for w, sid, ep, si in mats:
            C = build_coupling(ep, dataclasses.replace(DEFAULT_CFG, eta=1e-10))
            score = personalized_pagerank(C.sym, si, alpha=a)
            prec.append(sem_precision(ep, topk_idx(score, si, K), si))
            r = heldout_recall(store, ep, si, dataclasses.replace(DEFAULT_CFG, k_hop=K_HOP, N_max=N_MAX), a)
            if r is not None: rec.append(r)
        ppr_rows[a] = (st.mean(prec), st.mean(rec) if rec else float("nan"))
        lines.append(f"| {a} | {ppr_rows[a][0]:.3f} | {ppr_rows[a][1]:.3f} |")
        print(f"  PPR α={a}: prec={ppr_rows[a][0]:.3f} recall={ppr_rows[a][1]:.3f}")

    # ── head-to-head at each method's best precision operating point ───────────────────
    best_lam = max(sweep_rows, key=lambda l: sweep_rows[l][2])
    best_a = max(ppr_rows, key=lambda a: ppr_rows[a][0])
    cfg = dataclasses.replace(DEFAULT_CFG, decay=best_lam, k_hop=K_HOP, N_max=N_MAX, H_max=8000)
    lines += ["", f"## 3. Head-to-head — gather(λ={best_lam}) vs PPR(α={best_a})", "",
              "| seed | gather prec | PPR prec | gather rec | PPR rec | top-10 overlap |",
              "|---|---|---|---|---|---|"]
    gp, pp, gr, pr, ov = [], [], [], [], []
    for w, sid, ep, si in mats:
        res = gather(ep, si, cfg); grel = res.relevance()
        C = build_coupling(ep, dataclasses.replace(DEFAULT_CFG, eta=1e-10))
        psc = personalized_pagerank(C.sym, si, alpha=best_a)
        gk, pk = set(topk_idx(grel, si, K)), set(topk_idx(psc, si, K))
        gpi, ppi = sem_precision(ep, list(gk), si), sem_precision(ep, list(pk), si)
        grr = heldout_recall(store, ep, si, cfg, "gather")
        prr = heldout_recall(store, ep, si, cfg, best_a)
        jac = len(gk & pk) / len(gk | pk) if (gk | pk) else 0.0
        gp.append(gpi); pp.append(ppi); ov.append(jac)
        if grr is not None: gr.append(grr)
        if prr is not None: pr.append(prr)
        lines.append(f"| {w} | {gpi:.3f} | {ppi:.3f} | "
                     f"{grr if grr is None else round(grr,2)} | {prr if prr is None else round(prr,2)} | {jac:.2f} |")
    lines += ["", f"**Means:** gather precision {st.mean(gp):.3f} vs PPR {st.mean(pp):.3f}; "
              f"gather recall {st.mean(gr):.3f} vs PPR {st.mean(pr):.3f}; "
              f"mean top-10 overlap {st.mean(ov):.2f}.", ""]
    print(f"\nHEAD-TO-HEAD gather(λ={best_lam}) vs PPR(α={best_a}):")
    print(f"  precision {st.mean(gp):.3f} vs {st.mean(pp):.3f} | recall {st.mean(gr):.3f} vs {st.mean(pr):.3f} | overlap {st.mean(ov):.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"\nReport → {OUT}")
    store.close()

if __name__ == "__main__":
    main()
