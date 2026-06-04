from __future__ import annotations
# Reasoning bench — the decisive test: does the field's gathered context make a small model reason
# better than prior-art retrieval, and how far up the parameter-count curve does it push? Same QA set
# (1-hop + 2-hop) scored by gold coverage under:
#   closed-{1B,3B,8B}  — no context (the scaling curve for the param-count claim)
#   rag-3B             — dense RAG: embed the QUESTION, top-k facts by cosine (the prior-art baseline)
#   field-3B           — my retrieval, single gather from query-grounded seeds, top-k facts
# The 2-hop bucket is the discriminator: its answer lives in a fact that never names the seed, so
# dense-on-query can't reach it but the field climbs to it.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src:scripts venv/bin/python scripts/reason_bench.py
import argparse, dataclasses, random, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np, torch; torch.set_num_threads(1)
from field.config import DEFAULT_CFG
from field.gather import materialize, gather, build_mesh
from graph import GraphStore
from llm import call_llm, warm_models
from wd_qa import build_qa, build_multihop, coverage

CFG = dataclasses.replace(DEFAULT_CFG, decay_gamma=1.0, N_max=400, k_hop=4, up_max=30,
                          target_size=34)                 # plateau knee from the S* sweep
TOPK = 34                                                 # equal context budget for rag + field
QUERY_W = 4.0                                             # field readout: tilt toward query relevance

def e_facts(store, ids):
    out = []
    for nid in ids:
        if nid.startswith("e_") and (t := store.text(nid)): out.append(t)
    return out

# dense RAG: embed the QUESTION, rank all store e_ facts by cosine, take top-k. Standard RAG baseline.
def ctx_rag(store, query, k=TOPK):
    from embed import embed, unpack
    M, ids = store._endpoint_matrix(); qb = embed(query)
    if M is None or qb is None: return []
    sims = M @ unpack(qb)
    ranked = [ids[i] for i in np.argsort(-sims) if ids[i].startswith("e_")]
    return e_facts(store, ranked[:k])

# field single-gather: ground the query to seeds (find_vec), gather the seed neighborhood, then read
# out the top-k facts with a query-conditioned tilt (structural recall × query precision). query_w=0
# ⇒ pure structural relevance (the prior, query-agnostic readout) for the A/B.
def ctx_field(store, query, k=TOPK, query_w=QUERY_W):
    seeds = []
    for c in store.find_vec(query, 5):
        if any(c == s or c in {n for n, _ in store.neighbors(s)} for s in seeds): continue
        seeds.append(c)
        if len(seeds) >= 2: break
    ep, si = materialize(store, seeds, CFG)
    if not ep.node_ids: return []
    res = gather(ep, si, CFG)
    from embed import embed, unpack
    qb = embed(query); qv = torch.tensor(unpack(qb)) if qb is not None else None
    mesh = build_mesh(res, top_k=k, query_vec=qv, query_w=query_w)
    return e_facts(store, mesh.node_ids)

# flat system (the "optimal combo"): wide query-conditioned gather + ONE bounded expansion, no
# recursive LLM. (1) ground the subject; (2) gather; (3) pull the top bridge ENTITY from the mesh and
# re-seed {subject ∪ bridge} for a second gather — this is the mass-seed that reaches 2-hop without a
# recursive chain; (4) the now-oversized active set is read out query-conditioned (selection, not a
# no-op). Returns the top-k facts for a single capable-model call.
def _top_bridge(res, seeds, n=1):
    rel = res.relevance(); seen = set(seeds)
    ents = [(float(rel[i]), res.ep.node_ids[i]) for i in range(len(res.ep.node_ids))
            if not res.ep.node_ids[i].startswith("e_") and res.ep.node_ids[i] not in seen]
    ents.sort(reverse=True)
    return [n_ for _, n_ in ents[:n]]

def ctx_flat(store, query, k=TOPK, expand=True, query_w=QUERY_W):
    cands = [c for c in store.find_vec(query, 6) if not c.startswith("e_")]
    seeds = cands[:1] or store.find_vec(query, 1)         # the subject (top query-similar entity)
    ep, si = materialize(store, seeds, CFG)
    if not ep.node_ids: return []
    res = gather(ep, si, CFG)
    if expand:                                            # one mass-seed expansion to the bridge
        seeds2 = list(dict.fromkeys(seeds + _top_bridge(res, seeds, 1)))
        if len(seeds2) > len(seeds):
            ep, si = materialize(store, seeds2, CFG); res = gather(ep, si, CFG)
    from embed import embed, unpack
    qb = embed(query); qv = torch.tensor(unpack(qb)) if qb is not None else None
    mesh = build_mesh(res, top_k=k, query_vec=qv, query_w=query_w)
    return e_facts(store, mesh.node_ids)

# grow+q context: uniform best-first growth, query-conditioned readout.
GROW_CFG = dataclasses.replace(CFG, k_hop=3, up_max=60)
def ctx_grow(store, query, k=TOPK, query_w=QUERY_W, grow_cohere=3.0):
    cfg = GROW_CFG
    cands = [c for c in store.find_vec(query, 6) if not c.startswith("e_")]
    seeds = cands[:1] or store.find_vec(query, 1)
    from embed import embed, unpack
    qb = embed(query); qv = torch.tensor(unpack(qb)) if qb is not None else None
    ep, si = materialize(store, seeds, cfg)
    if not ep.node_ids: return []
    res = gather(ep, si, cfg)
    mesh = build_mesh(res, top_k=k, query_vec=qv, query_w=query_w)
    return e_facts(store, mesh.node_ids)

# qseed: legacy condition — deleted (attach_query_node removed in Phase 0). Stub so imports don't break.
QSEED_CFG = GROW_CFG
def ctx_qseed(store, query, k=TOPK, link_top=50, coseed=True):
    return ctx_grow(store, query, k=k)

def ask(model, q, facts):
    if facts:
        ctx = "\n".join(f"- {f}" for f in facts)
        p = (f"Reference facts (may be partial):\n{ctx}\n\nAnswer in 1-3 sentences with specifics. Use "
             f"BOTH these facts AND your own knowledge.\nQuestion: {q}")
    else:
        p = f"Answer in 1-3 sentences with specific facts.\nQuestion: {q}"
    return call_llm(p, model, options={"temperature": 0, "num_predict": 200}).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.wikidata.sqlite")
    ap.add_argument("--n1", type=int, default=18)         # 1-hop questions
    ap.add_argument("--n2", type=int, default=18)         # 2-hop questions
    ap.add_argument("--conds", default="closed1B,closed3B,closed8B,rag3B,field3B")
    ap.add_argument("--out", default="runs_experiment/reason_bench.md")
    a = ap.parse_args()
    conds = a.conds.split(",")
    tier_of = lambda c: "1B" if "1B" in c else ("7B" if "7B" in c else "3B")
    warm_models(tuple(sorted({tier_of(c) for c in conds})))
    s = GraphStore(a.store)
    rng = random.Random(0)
    qa1 = build_qa(s, n_subjects=40, per_subject=6); rng.shuffle(qa1)
    qa2 = build_multihop(s, n_subjects=50, per_subject=4); rng.shuffle(qa2)
    qa = qa1[:a.n1] + qa2[:a.n2]
    rng.shuffle(qa)

    # cov[cond] = per-question coverage ; by hops too
    cov = {c: [] for c in conds}; covh = {c: {1: [], 2: []} for c in conds}
    detail = []
    for qi, q in enumerate(qa):
        h = q.get("hops", 1)
        row = {"q": q["q"], "hops": h, "breadth": q["breadth"]}
        for c in conds:
            model = tier_of(c)
            if c.startswith("closed"): ans = ask(model, q["q"], None)
            elif c.startswith("rag"):  ans = ask(model, q["q"], ctx_rag(s, q["q"]))
            elif c.startswith("field"): ans = ask(model, q["q"], ctx_field(s, q["q"]))
            else: ans = ""
            n_hit, n_gold, _ = coverage(ans, q["gold"])
            cval = n_hit / max(n_gold, 1)
            cov[c].append(cval); covh[c][h].append(cval); row[c] = cval
        detail.append(row)
        print(f"  {qi+1}/{len(qa)} h{h} {q['q'][:48]}", end="\r")
    print()

    def mean(xs): return st.mean(xs) if xs else float("nan")
    L = ["# Reasoning bench — retrieval effect on small-LLM reasoning", "",
         f"Store `{a.store}`. {len(qa)} questions ({a.n1} 1-hop + {a.n2} 2-hop), gold-coverage scored. "
         f"Equal context budget (top-{TOPK} facts) for rag/field. 2-hop = answer fact never names the "
         f"seed (the field-vs-RAG discriminator).", "",
         "## Coverage by condition", "",
         "| condition | overall | 1-hop | 2-hop |", "|---|---|---|---|"]
    for c in conds:
        L.append(f"| {c} | {mean(cov[c]):.3f} | {mean(covh[c][1]):.3f} | {mean(covh[c][2]):.3f} |")
    g = lambda d, k: mean(d.get(k, [float("nan")]))
    L += ["", "## Reading", "",
          f"- closed-book scaling: 1B={g(cov,'closed1B'):.3f} 3B={g(cov,'closed3B'):.3f} "
          f"8B(llama3:8b)={g(cov,'closed7B'):.3f}",
          f"- 3B+field vs 3B+rag (prior art): {g(cov,'field3B'):.3f} vs {g(cov,'rag3B'):.3f} "
          f"(Δ {g(cov,'field3B')-g(cov,'rag3B'):+.3f})",
          f"- 2-hop discriminator — field {mean(covh.get('field3B',{}).get(2,[0])):.3f} vs "
          f"rag {mean(covh.get('rag3B',{}).get(2,[0])):.3f}", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text("\n".join(L))
    # also dump the raw per-question detail for the param-count analysis (stage 4)
    import json; Path(a.out.replace(".md", ".json")).write_text(json.dumps(detail))
    print("  ".join(f"{c}={mean(cov[c]):.3f}" for c in conds))
    print(f"2-hop: " + "  ".join(f"{c}={mean(covh[c][2]):.3f}" for c in conds))
    print(f"→ {a.out}")
    s.close()

if __name__ == "__main__":
    main()
