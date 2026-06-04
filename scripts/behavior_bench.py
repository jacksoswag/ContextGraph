from __future__ import annotations
# BEHAVIORAL benchmark — the fair-ground test the string-coverage benches can't do (handoff item #3).
# The claim under test: grow+q gathers a CONNECTED, contextualizing region (the "Anne Frank → Nazi
# Germany → Holocaust" effect), which helps a knowledge-POOR model answer in a well-situated, grounded
# way that flat dense-RAG (query-matched chunks) cannot. String-coverage on famous-entity QA can't see
# this — the model answers famous entities from parametric memory regardless of context.
#
# So this bench is built to NOT be gameable or overfit:
#   - LONG-TAIL emphasis: low store-degree entities the answerer has no parametric knowledge of, so the
#     context is the ONLY signal (famous entities are a contamination control, expected ~tie).
#   - CROSS-FAMILY judge: granite judges llama answers (no self-preference), BLIND to which condition
#     produced an answer, scoring against the entity's TRUE spine from the store (ground truth).
#   - BALANCED rubric: `situated` rewards context, `grounded`/`n_unsupported` PENALIZE fabrication — a
#     system that floods noisy context nets out. You can't win by dumping more text.
#   - FROZEN config (GATHER_CFG from the live path; no tuning on this set), 2 query seeds for variance,
#     per-type × per-specificity breakdown, raw answers + judge rationales dumped for human inspection.
#
# Conditions (same answerer model, equal top-K budget): closed (no ctx) · rag (dense top-K) · field
# (grow+q gather_context). Query types: open ("tell me about X" — the situatedness showcase, no single
# gold) · fact (1-hop, has gold) · multihop (2-hop, has gold). Resumable: appends to a JSONL, skips done.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src:scripts venv/bin/python scripts/behavior_bench.py --tiers 3B
import argparse, json, random, statistics as st, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch; torch.set_num_threads(1)
import llm
from llm import call_llm, call_json, warm_models, ensure_ollama_models
from graph import GraphStore
from embed import embed, unpack
from field.gather import materialize, gather
from field.loop import gather_context, GATHER_CFG
from field.seams import render_props
from wd_qa import build_qa, build_multihop, rich_subjects, coverage, _name, _clean
from reason_bench import ctx_rag, e_facts

JUDGE_TAG = "granite3.1-dense:8b"          # cross-family judge (answerer = llama); registered below
TOPK = GATHER_CFG.target_size              # equal context budget for rag + field (=34)

ANSWER_PROMPT = ("Reference facts (may be partial):\n{ctx}\n\nUsing these facts together with your own "
                 "knowledge, answer in 2-5 sentences with specifics. Where the facts connect related "
                 "entities, use that context.\nQuestion: {q}")
ANSWER_CLOSED = "Answer in 2-5 sentences with specific facts.\nQuestion: {q}"

JUDGE_PROMPT = """You are grading an answer to a question about "{name}". Below are VERIFIED reference
facts about {name} from a knowledge base — treat them as ground truth (they may be incomplete).

Reference facts:
{ref}

Question: {q}
Answer to grade:
{ans}

Grade the answer as JSON with integer 1-5 scores:
- "correct": factual accuracy (consistent with the reference, not contradicting it). 5=fully accurate, 1=mostly wrong.
- "grounded": are the answer's specific claims supported by the reference facts vs invented? 5=all supported, 1=fabricated.
- "situated": does it place {name} in meaningful context (related people/works/events/era/field) rather than a bare isolated fact? 5=richly contextualized, 1=bare.
- "overall": overall helpfulness as an informative response. 5=excellent, 1=poor.
- "n_unsupported": integer count of distinct factual claims in the answer NOT supported by the reference (hallucination count).
Return ONLY JSON: {{"correct":n,"grounded":n,"situated":n,"overall":n,"n_unsupported":n,"why":"one short sentence"}}"""


# ── entity selection: famous (model KNOWS it) vs long-tail (model does NOT) ──
# This axis is the whole point. Degree alone is wrong — it conflates crawl-centrality with fame (a
# Marie-Curie crawl makes "Bronia Dłuska" degree-30, but the model has never heard of her). So FAMOUS
# is a curated, verified-present anchor list of entities a small LLM unambiguously knows (the real
# contamination control); LONG-TAIL is junk-filtered low-degree real-name subjects the model does not.
# The report ALSO plots advantage vs per-entity closed-book score, so the finding never hinges on this
# hand-bucketing alone.
_JUNK = ("category:", "wikimedia", "template:", "list of", "disambiguation", "main topic",
         "wikiproject", "wikidata", "honorary doctor", "doctor of philosophy", "fellow of the",
         "gender gap", "decoration for")
# domain-appropriate to this store (physics / Curie crawl); each is resolved+verified present below.
_FAMOUS_ANCHORS = ["albert einstein", "isaac newton", "marie curie", "pierre curie", "max planck",
                   "niels bohr", "werner heisenberg", "quantum mechanics", "university of oxford",
                   "enrico fermi", "erwin schrödinger", "wolfgang pauli", "paul dirac", "switzerland"]

def _is_junk(t: str) -> bool:
    t = (t or "").lower()
    return (not t) or t.replace(" ", "").isdigit() or len(t) < 5 or any(j in t for j in _JUNK)

# resolve a name to its best exact (non-category) node id present in the store
def _resolve(store: GraphStore, name: str) -> str | None:
    for nid in store.find(name, 5):
        t = (store.text(nid) or "").lower()
        if nid.startswith("n_") and not _is_junk(t) and (t == name or t.startswith(name)):
            return nid
    return None

def select_entities(store: GraphStore, n_famous: int, n_longtail: int, seed: int,
                    anchors: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    fam = []
    for name in (anchors or _FAMOUS_ANCHORS):
        nid = _resolve(store, name)
        if nid and store.degree(nid) >= 4: fam.append(nid)
    rng.shuffle(fam); fam = fam[:n_famous]
    # long-tail: small but non-empty spine (3..8 outgoing facts), real-name surface, junk-filtered.
    rows = store._con.execute(
        "SELECT source_id, COUNT(*) c FROM edges GROUP BY source_id HAVING c BETWEEN 3 AND 8").fetchall()
    tail = [r[0] for r in rows if r[0] and not r[0].startswith("e_") and not _is_junk(store.text(r[0]))]
    tail = [t for t in tail if t not in set(fam)]
    rng.shuffle(tail); tail = tail[:n_longtail]
    out = []
    for bucket, ids in (("famous", fam), ("longtail", tail)):
        for eid in ids:
            out.append({"id": eid, "name": _name(store.text(eid)), "bucket": bucket,
                        "degree": store.degree(eid)})
    return out

# entity_spine: the entity's reified facts (e_ surfaces) — the judge's ground-truth reference sheet.
def entity_spine(store: GraphStore, eid: str, limit: int = 40) -> list[str]:
    out = []
    for e, _sc in store.containing_edges(eid, limit):
        if (t := store.text(e)): out.append(t)
    return out

# build_queries: per entity — one open ("tell me about"), up to 1 factual (gold), up to 1 multihop (gold).
def build_queries(store: GraphStore, entities: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_id = {e["id"]: e for e in entities}
    qa = {q["subj_id"]: q for q in build_qa(store, n_subjects=120, per_subject=2, seed=seed)}
    mh = defaultdict(list)
    for q in build_multihop(store, n_subjects=120, per_subject=2, seed=seed): mh[q["subj_id"]].append(q)
    out = []
    for e in entities:
        eid, name = e["id"], e["name"]
        out.append({"qid": f"{eid}:open", "q": f"Tell me about {name}. Who or what were they, and what is "
                    f"the context around them?", "entity": eid, "name": name, "bucket": e["bucket"],
                    "qtype": "open", "gold": []})
        if eid in qa:
            q = qa[eid]
            out.append({"qid": f"{eid}:fact", "q": q["q"], "entity": eid, "name": name,
                        "bucket": e["bucket"], "qtype": "fact", "gold": q["gold"]})
        if mh.get(eid):
            q = rng.choice(mh[eid])
            out.append({"qid": f"{eid}:multi", "q": q["q"], "entity": eid, "name": name,
                        "bucket": e["bucket"], "qtype": "multihop", "gold": q["gold"]})
    return out


# CROSS-DOCUMENT synthesis questions (prose corpus): each requires INTEGRATING facts that live in
# different articles, connected only through shared entities — the textbook RAG weakness, and the
# field's native mode (the graph has no documents; an entity is one node across all sources). refs =
# the entities whose spines form the judge's ground-truth reference (union across docs); gold = soft
# coverage strings. Judged primarily on situated/grounded/correct (synthesis has no single gold).
CROSSDOC_WWII = [
    ("How did Nazi Germany's rise and policies lead to the persecution of Anne Frank and her family?",
     ["anne frank", "nazi germany"], ["jewish", "persecution", "amsterdam", "nazi"]),
    ("What happened to Anne Frank and her sister after they were deported from the Netherlands?",
     ["anne frank", "auschwitz concentration camp", "bergen-belsen concentration camp"],
     ["auschwitz", "bergen-belsen", "typhus", "margot"]),
    ("How did Otto Frank's fate differ from the rest of the Frank family, and what did he do afterward?",
     ["otto frank", "anne frank"], ["survived", "auschwitz", "diary", "published"]),
    ("What role did Miep Gies play in preserving Anne Frank's diary?",
     ["miep gies", "anne frank"], ["diary", "saved", "annex", "hiding"]),
    ("How were the events of Kristallnacht connected to the broader Holocaust?",
     ["kristallnacht", "the holocaust"], ["jewish", "synagogue", "pogrom", "persecution"]),
    ("What was the relationship between Himmler, the SS, and the concentration camps?",
     ["heinrich himmler", "schutzstaffel", "auschwitz concentration camp"], ["ss", "camps", "extermination"]),
    ("How did the German invasions of Poland and the Netherlands set the stage for the Holocaust?",
     ["invasion of poland", "nazi germany"], ["poland", "occupation", "invasion", "jews"]),
    ("What was Adolf Eichmann's role in organizing the Holocaust?",
     ["adolf eichmann", "the holocaust"], ["deportation", "jews", "final solution"]),
]

def build_crossdoc(store: GraphStore, qset) -> list[dict]:
    out = []
    for q, refs, gold in qset:
        rids = [r for r in (_resolve(store, e) for e in refs) if r]
        if len(rids) < 2: continue                         # need >=2 resolvable docs/entities
        out.append({"qid": "xdoc:" + str(abs(hash(q)) % 10**8), "q": q, "entity": rids[0],
                    "name": _name(store.text(rids[0])), "bucket": "crossdoc", "qtype": "crossdoc",
                    "gold": gold, "refs": rids})
    return out


# ── contexts (equal top-K budget) ──
# Shared seed grounding so field/ppr/pprR all read facts from the SAME comprehensive active set — the
# comparison is then purely the SCORER (the field's energy settle vs PPR diffusion vs recursive PPR),
# isolated from retrieval. rag is the prior-art control (its own dense top-K over the whole store).
def _grow_seeds(store, query):
    cands = [c for c in store.find_vec(query, 6) if not c.startswith("e_")]
    return cands[:2] or store.find_vec(query, 1)

def ctx_field(store, query):                               # atomic readout (current live cap, 34)
    mesh = gather_context(store, query, _grow_seeds(store, query), GATHER_CFG)
    return e_facts(store, mesh.node_ids) if mesh else []

# comprehensive readout: keep what the FIELD marked warm (≥1% peak relevance) up to a generous token
# budget, instead of the atomic target_size=34 cap. No new pruner, no physics change — just stop
# discarding field-validated context. Sparse entities stay warm-limited; rich ones surface their region.
import dataclasses as _dc
GATHER_CFG_CTX = _dc.replace(GATHER_CFG, target_size=80)
def ctx_fieldC(store, query):
    mesh = gather_context(store, query, _grow_seeds(store, query), GATHER_CFG_CTX)
    return e_facts(store, mesh.node_ids) if mesh else []

# fieldO: rendering-shape ablation — now identical to fieldP (render_outline deleted with Mesh.parent in Phase 0).
def ctx_fieldO(store, query):
    return ctx_fieldP(store, query)

# fieldP: SAME comprehensive gather as fieldC/fieldO, rendered as the corpus's nested PROPOSITION lines
# (render_props) — fact→fact connections shown (A) =rel=> (B) with nested facts in recursive parens and
# [event] elided. Isolates readout SHAPE: vs fieldO's indented outline and fieldC's flattened bullets.
def ctx_fieldP(store, query):
    mesh = gather_context(store, query, _grow_seeds(store, query), GATHER_CFG_CTX)
    if not mesh or not mesh.node_ids: return ""
    return render_props(mesh, store, max_nodes=GATHER_CFG_CTX.target_size)

# ctx_front: the user's "spawn → settle → repeat" wavefront, realized on the existing physics. Each
# round settles anchored on the accumulated front (prior context), harvests its warm facts, then
# ADVANCES the front to the top NEW warm entities and re-settles — energy is RE-INJECTED outward
# (redistributed), not leaked, so 2-hop context (a bridge entity's own facts) warms instead of going
# cold. Self-terminating: when a round adds no new warm entity, the front has dissipated. No 34 cap —
# the front's natural extent IS the breadth; facts ranked by best relevance seen across rounds.
def ctx_front(store, query, rounds=2, per_round=4, budget=80):
    qv = unpack(embed(query)); anchored = list(_grow_seeds(store, query)); seen = {}
    for _r in range(rounds + 1):
        ep, si = materialize(store, anchored, GATHER_CFG_CTX)
        if not ep.node_ids: break
        res = gather(ep, si, GATHER_CFG_CTX, lean=True); rel = res.relevance(); mx = float(rel.max())
        if mx <= 0: break
        warm_ent = []
        for i in range(len(ep.node_ids)):
            n = ep.node_ids[i]; r = float(rel[i]) / mx
            if r < 0.01: continue
            if n.startswith("e_"): seen[n] = max(seen.get(n, 0.0), r)
            elif n not in set(anchored): warm_ent.append((r, n))
        warm_ent.sort(reverse=True)
        new = [n for _r2, n in warm_ent[:per_round]]
        if not new: break                                  # front dissipated → natural termination
        anchored += new
    ranked = sorted(seen, key=lambda n: -seen[n])[:budget]
    return [t for n in ranked if (t := store.text(n))]

# ctx_front_hybrid: the DRIFT-CONTROLLED front (the directed-tree upgrade). Same wavefront, but each
# round selects its K branches by relevance × query-coherence × MOMENTUM — momentum(C)=cos(C−parent,
# parent−grandparent) asks "does this step continue the branch's semantic trajectory?" (the ray), with
# parent = the nearest anchored entity (which thread C continues). A K floor keeps it a TREE not a chain;
# dilution `stop` terminates when the best new branch is too weak. K/momentum/direction live in this
# between-round SELECTION (stable) — the per-round settle is the unchanged symmetric gather (no G3). It
# targets the plain front's two measured failures: drift→hallucination and dilution→lower situatedness.
def ctx_front_hybrid(store, query, rounds=3, k=6, k_min=3, mom=2.0, qw=2.0, budget=80, stop=0.02):
    qv = unpack(embed(query))
    if qv is None: return ctx_field(store, query)
    qv = qv / (np.linalg.norm(qv) + 1e-9)
    def vec(n):
        a = store.anchor(n)
        return None if a is None else a / (np.linalg.norm(a) + 1e-9)
    anchored = {}                                          # entity -> {'v': unit emb, 'dir': branch trajectory}
    for e in _grow_seeds(store, query):
        if (v := vec(e)) is not None: anchored[e] = {"v": v, "dir": qv}
    seen = {}
    for _r in range(rounds + 1):
        ep, si = materialize(store, list(anchored) or _grow_seeds(store, query), GATHER_CFG_CTX)
        if not ep.node_ids: break
        res = gather(ep, si, GATHER_CFG_CTX, lean=True); rel = res.relevance(); mx = float(rel.max())
        if mx <= 0: break
        cands = []
        for i in range(len(ep.node_ids)):
            n = ep.node_ids[i]; rr = float(rel[i]) / mx
            if rr < 0.01: continue
            if n.startswith("e_"): seen[n] = max(seen.get(n, 0.0), rr); continue
            if n in anchored or (v := vec(n)) is None: continue
            par = max(anchored, key=lambda p: float(anchored[p]["v"] @ v))   # thread this continues
            qcoh = max(float(v @ qv), 0.0)
            step = v - anchored[par]["v"]; sd = anchored[par]["dir"]
            momentum = max(float((step / (np.linalg.norm(step) + 1e-9)) @ sd), 0.0) if sd is not None else qcoh
            cands.append((rr * (1.0 + qw * qcoh) * (1.0 + mom * momentum), n, v, par))
        if not cands: break
        cands.sort(key=lambda c: -c[0])
        keep = cands[: max(k_min, k)]
        if keep[0][0] < stop: break                        # front dissipated (best branch too weak)
        for _sc, n, v, par in keep:
            step = v - anchored[par]["v"]; anchored[n] = {"v": v, "dir": step / (np.linalg.norm(step) + 1e-9)}
    return [t for n in sorted(seen, key=lambda n: -seen[n])[:budget] if (t := store.text(n))]

# PPR over the SAME grow+q active set (answers "is PPR's diffusion a better scorer than the settle?").
# recursive = guided by previous context: round-1 PPR → its top entity nodes JOIN the personalization →
# round-2 PPR re-diffuses from {seeds ∪ round-1 context}. This is the user's "recursive PPR" — PPR's
# only handle on prior context is biasing the scalar restart vector (vs the field anchoring a vector STATE).
def ctx_ppr(store, query, recursive=False):
    from field.coupling import build as build_coupling
    from field.baseline import personalized_pagerank
    qv = unpack(embed(query))
    ep, si = materialize(store, _grow_seeds(store, query), GATHER_CFG)
    if not ep.node_ids: return []
    W = build_coupling(ep, GATHER_CFG).sym
    r = personalized_pagerank(W, si)
    if recursive:
        order = sorted(range(len(ep.node_ids)), key=lambda i: -float(r[i]))
        boost = [i for i in order if not ep.node_ids[i].startswith("e_")][:5]
        r = personalized_pagerank(W, list(dict.fromkeys(list(si) + boost)))
    eidx = sorted((i for i in range(len(ep.node_ids)) if ep.node_ids[i].startswith("e_")),
                  key=lambda i: -float(r[i]))
    return [t for i in eidx[:TOPK] if (t := store.text(ep.node_ids[i]))]

CTX = {"closed": lambda s, q: None, "rag": lambda s, q: ctx_rag(s, q, TOPK), "field": ctx_field,
       "fieldC": ctx_fieldC, "fieldO": ctx_fieldO, "fieldP": ctx_fieldP, "front": ctx_front,
       "hybrid": ctx_front_hybrid, "ppr": lambda s, q: ctx_ppr(s, q, False), "pprR": lambda s, q: ctx_ppr(s, q, True)}

# ctx is the final context string (flat bullets for list-conditions, a nested outline for fieldO); None = closed.
def answer(model, q, ctx):
    p = ANSWER_CLOSED.format(q=q) if not ctx else ANSWER_PROMPT.format(ctx=ctx, q=q)
    return call_llm(p, model, options={"temperature": 0, "num_predict": 280}).strip()

def judge(name, q, ans, ref):
    out = call_json(JUDGE_PROMPT.format(name=name, ref="\n".join(f"- {r}" for r in ref) or "(none)",
                    q=q, ans=ans or "(empty)"), JUDGE_TAG, options={"num_predict": 200})
    g = lambda k: float(out.get(k, 0) or 0) if isinstance(out, dict) else 0.0
    return {"correct": g("correct"), "grounded": g("grounded"), "situated": g("situated"),
            "overall": g("overall"), "n_unsupported": g("n_unsupported"),
            "why": (out.get("why", "") if isinstance(out, dict) else "")[:160]}


# Full transparency transcript: for every (query, condition) the EXACT rendered context mesh fed to the
# answerer + its output + the judge grade, grouped by question. Lets you read query → mesh → answer
# directly instead of trusting the aggregate tables (complements write_report; same jsonl source).
def write_transcript(rows, path):
    order = {c: i for i, c in enumerate(["closed", "rag", "field", "fieldC", "front", "hybrid", "ppr", "pprR"])}
    by_q: dict = {}
    for r in rows: by_q.setdefault((r.get("bucket", ""), r["qid"]), []).append(r)
    L = ["# Behavior-bench transcript — query → context mesh (fed to LLM) → output → judge", "",
         f"{len(by_q)} questions · {len(rows)} records. The mesh is the LITERAL bullet list passed to the "
         "answerer (the reified-edge surfaces the gather/RAG selected); `ref` = the judge's verified "
         "ground-truth spine. Read query → mesh → output to see exactly what context produced each answer.", ""]
    for (bucket, qid), rs in sorted(by_q.items()):
        q0 = rs[0]
        L += ["", "---", "", f"## [{bucket}] {q0.get('name', '?')} · {q0.get('qtype', '')}", "",
              f"**Query:** {q0.get('q', '')}", ""]
        ref = q0.get("ref") or []
        if ref:
            L += [f"<details><summary>judge reference spine (ground truth, {len(ref)} facts)</summary>", ""]
            L += [f"- {x}" for x in ref] + ["", "</details>", ""]
        for r in sorted(rs, key=lambda r: (order.get(r.get("cond", ""), 9), r.get("tier", ""))):
            L += [f"### {r.get('cond', '?')} · {r.get('tier', '')} — situated {r.get('situated', '?')} / "
                  f"grounded {r.get('grounded', '?')} / correct {r.get('correct', '?')} / "
                  f"halluc {r.get('n_unsupported', '?')}  ·  {r.get('n_facts', 0)} facts, {r.get('ctx_ms', '?')}ms", ""]
            mesh = r.get("ctx_render")
            if mesh is None:                                   # back-compat: older rows stored a fact list
                mesh = "\n".join(f"- {f}" for f in (r.get("ctx_facts") or []))
            L += ["**Context mesh fed to LLM:**", "", "```", mesh, "```", ""] if mesh.strip() \
                else ["**Context mesh fed to LLM:** _(none — closed-book)_", ""]
            ans = (r.get("answer") or "(empty)").replace("\n", "\n> ")
            L += ["**LLM output:**", "", "> " + ans, "", f"_judge:_ {r.get('why', '')}", ""]
    Path(path).write_text("\n".join(L))
    print(f"→ {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.wikidata.sqlite")
    ap.add_argument("--tiers", default="3B")                 # answerer tiers, comma-sep (3B,7B)
    ap.add_argument("--conds", default="closed,rag,field")
    ap.add_argument("--famous", type=int, default=10); ap.add_argument("--longtail", type=int, default=14)
    ap.add_argument("--seeds", default="0,7")                # query-sampling seeds for variance
    ap.add_argument("--anchors", default="")                 # famous-bucket names (prose store: article subjects)
    ap.add_argument("--crossdoc", action="store_true")       # add the cross-document synthesis question set
    ap.add_argument("--out", default="runs_experiment/behavior_bench")
    a = ap.parse_args()
    tiers = a.tiers.split(","); conds = a.conds.split(","); seeds = [int(x) for x in a.seeds.split(",")]
    anchors = [x.strip() for x in a.anchors.split(",") if x.strip()] or None
    llm._MODEL_TAGS["JUDGE"] = JUDGE_TAG                     # register judge so warm/resolve see it
    ensure_ollama_models(); warm_models(tuple(set(tiers)) + ("JUDGE",))
    s = GraphStore(a.store)
    jsonl = Path(a.out + ".jsonl"); jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            try: done.add(json.loads(ln)["key"])
            except Exception: pass
    fh = jsonl.open("a")

    # entities/queries are seed-dependent (variance); spine reference is per entity (cached)
    rows = []
    ctx_cache, ref_cache = {}, {}
    for seed in seeds:
        ents = select_entities(s, a.famous, a.longtail, seed, anchors)
        queries = build_queries(s, ents, seed)
        # process long-tail (the contamination-free differentiator) first, open/multihop before fact —
        # so the field-vs-PPR-vs-RAG verdict on knowledge-poor entities lands early, not after famous.
        order = {"open": 0, "multihop": 1, "fact": 2}
        queries.sort(key=lambda q: (q["bucket"] != "longtail", order.get(q["qtype"], 3)))
        if a.crossdoc and seed == seeds[0]:                  # curated set runs once (not seed-sampled), first
            queries = build_crossdoc(s, CROSSDOC_WWII) + queries
        for qi, q in enumerate(queries):
            # cross-doc reference = union of the question's entity spines (ground truth spanning docs)
            if q.get("refs"):
                ref = [f for rid in q["refs"] for f in entity_spine(s, rid, 15)]
            else:
                ref = ref_cache.setdefault(q["entity"], entity_spine(s, q["entity"]))
            for cond in conds:
                ck = (q["qid"], cond)
                if ck not in ctx_cache:
                    t0 = time.time(); ctx_cache[ck] = (CTX[cond](s, q["q"]), time.time() - t0)
                facts, ctx_dt = ctx_cache[ck]
                # facts is a list (flat conditions) or a pre-rendered str (fieldO outline); build the EXACT
                # ctx string fed to the LLM once, and log it verbatim so the transcript is faithful.
                ctx_render = facts if isinstance(facts, str) else ("\n".join(f"- {f}" for f in facts) if facts else "")
                n_ctx = (facts.count("\n") + 1) if isinstance(facts, str) and facts else len(facts or [])
                for tier in tiers:
                    key = f"{seed}|{q['qid']}|{cond}|{tier}"
                    if key in done:
                        continue
                    ans = answer(tier, q["q"], ctx_render)
                    n_hit, n_gold, _ = coverage(ans, q["gold"]) if q["gold"] else (0, 0, [])
                    jd = judge(q["name"], q["q"], ans, ref)
                    rec = {"key": key, "seed": seed, "qid": q["qid"], "q": q["q"], "name": q["name"],
                           "bucket": q["bucket"], "qtype": q["qtype"], "cond": cond, "tier": tier,
                           "n_facts": n_ctx, "ctx_ms": round(ctx_dt * 1000),
                           "cov": (n_hit / n_gold) if n_gold else None, **jd, "answer": ans,
                           "ctx_render": ctx_render, "ref": ref}   # the EXACT context string fed to the LLM + judge ground-truth
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); rows.append(rec)
            print(f"  s{seed} {qi+1}/{len(queries)} {q['qtype'][:5]} {q['bucket'][:4]} {q['name'][:28]}", end="\r")
    fh.close(); print()
    # reload everything (incl. prior partial runs) for the report
    allrows = [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]
    write_report(allrows, a.out + ".md", a.store, tiers, conds)
    write_transcript(allrows, a.out + ".transcript.md")
    s.close()


def write_report(rows, out, store, tiers, conds):
    def mean(xs): return st.mean(xs) if xs else float("nan")
    def sub(pred): return [r for r in rows if pred(r)]
    metrics = ["situated", "grounded", "correct", "overall", "n_unsupported"]
    L = ["# Behavioral benchmark — does grow+q's connected context beat dense RAG?", "",
         f"Store `{store}`. Answerer tiers {tiers}; conditions {conds}; judge `{JUDGE_TAG}` (cross-family, "
         f"blind to condition), scoring against each entity's true spine. Equal top-{TOPK} context budget. "
         f"{len(rows)} graded answers. Long-tail = low store-degree entities (no parametric knowledge → "
         "context is the only signal); famous = contamination control.", ""]
    for tier in tiers:
        L += [f"## Answerer = {tier}", "",
              "| bucket | qtype | cond | situated | grounded | correct | overall | halluc | cov | n |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for bucket in ("longtail", "famous", "crossdoc"):
            for qtype in ("open", "fact", "multihop", "crossdoc"):
                for cond in conds:
                    rs = sub(lambda r: r["tier"] == tier and r["bucket"] == bucket
                             and r["qtype"] == qtype and r["cond"] == cond)
                    if not rs: continue
                    cov = mean([r["cov"] for r in rs if r["cov"] is not None])
                    L.append(f"| {bucket} | {qtype} | {cond} | {mean([r['situated'] for r in rs]):.2f} "
                             f"| {mean([r['grounded'] for r in rs]):.2f} | {mean([r['correct'] for r in rs]):.2f} "
                             f"| {mean([r['overall'] for r in rs]):.2f} | {mean([r['n_unsupported'] for r in rs]):.2f} "
                             f"| {cov:.2f} | {len(rs)} |")
            L.append("| | | | | | | | | | |")
        # headline deltas: field vs each retrieval rival (rag / ppr / recursive-ppr), per bucket
        L += ["", f"### {tier} — field vs the rivals (situated / grounded / correct / overall, halluc)", ""]
        for bucket in ("longtail", "famous", "crossdoc"):
            d = {}
            for cond in conds:
                rs = sub(lambda r: r["tier"] == tier and r["bucket"] == bucket and r["cond"] == cond)
                if rs: d[cond] = {m: mean([r[m] for r in rs]) for m in metrics}
            def fmt(c): return (f"{c} {d[c]['situated']:.1f}/{d[c]['grounded']:.1f}/{d[c]['correct']:.1f}/"
                                f"{d[c]['overall']:.1f} (h{d[c]['n_unsupported']:.1f})") if c in d else f"{c} —"
            L.append(f"- **{bucket}**: " + " · ".join(fmt(c) for c in ("closed", "rag", "ppr", "pprR", "field", "fieldC", "front", "hybrid")))
            # fieldC (comprehensive readout) is the protagonist — does keeping the field's warm context
            # beat dense RAG, PPR, recursive PPR, and the atomic-capped field?
            hero = "fieldC" if "fieldC" in d else "field"
            if hero in d:
                for rival in ("rag", "ppr", "pprR", "field"):
                    if rival in d and rival != hero:
                        L.append(f"    - {hero} − {rival}: situated {d[hero]['situated']-d[rival]['situated']:+.2f}, "
                                 f"correct {d[hero]['correct']-d[rival]['correct']:+.2f}, "
                                 f"overall {d[hero]['overall']-d[rival]['overall']:+.2f}, "
                                 f"halluc {d[hero]['n_unsupported']-d[rival]['n_unsupported']:+.2f}")
        # robustness: bucket entities by the model's OWN closed-book `overall` (the parametric-knowledge
        # proxy), independent of the famous/long-tail labels. The thesis predicts field's edge over rag
        # concentrates where the model knows little (low closed-book).
        closed_by_ent = defaultdict(list)
        for r in sub(lambda r: r["tier"] == tier and r["cond"] == "closed"):
            closed_by_ent[r["name"]].append(r["overall"])
        cb = {k: mean(v) for k, v in closed_by_ent.items()}
        L += ["", f"### {tier} — advantage vs the model's parametric knowledge (not label-dependent)", "",
              "| model knows entity? | field overall | rag overall | closed overall | Δ field−rag | n |",
              "|---|---|---|---|---|---|"]
        for lab, pred in (("knows little (closed<3.5)", lambda v: v < 3.5),
                          ("knows it (closed≥3.5)", lambda v: v >= 3.5)):
            ents = {k for k, v in cb.items() if pred(v)}
            f_ = sub(lambda r: r["tier"] == tier and r["cond"] == "field" and r["name"] in ents)
            g_ = sub(lambda r: r["tier"] == tier and r["cond"] == "rag" and r["name"] in ents)
            c_ = sub(lambda r: r["tier"] == tier and r["cond"] == "closed" and r["name"] in ents)
            fo, go = mean([r["overall"] for r in f_]), mean([r["overall"] for r in g_])
            L.append(f"| {lab} | {fo:.2f} | {go:.2f} | {mean([r['overall'] for r in c_]):.2f} "
                     f"| {fo-go:+.2f} | {len(ents)} |")
    Path(out).write_text("\n".join(L))
    print("\n".join(L)); print(f"\n→ {out}")

if __name__ == "__main__":
    main()
