from __future__ import annotations
# STRONG head-to-head bench — does cheap+structure beat expensive+RAG on COMPLEX astrophysics queries?
# Six conditions = {3B, 8B} answerer × {base (closed-book), RAG (dense top-K), my-system (mesh_gather)},
# over the prose ASTROPHYSICS corpus (.di-ui/graph.astro.merged.sqlite — 49.7K concept nodes / 104K
# reified facts, all embedded). The astro corpus has no relational schema (prose predicates of/in/cause),
# so wd_qa's auto-derived gold doesn't port; the queries are a small CURATED set of real 2-hop astro
# questions whose seed/bridge/answer are VERIFIED present in the store at runtime (verification recorded
# in the report). Each requires chaining a seed concept through an intermediate (bridge) to the answer —
# the multi-hop case dense, query-seeded RAG struggles with and the graph traversal should win.
#
# FAIRNESS (all enforced here):
#   - context computed ONCE per type per query (model-independent), then answered by BOTH 3B and 8B.
#   - RAG = dense kNN over the store's e_ facts (info_vector cosine), top-K where K = the mesh's fact
#     count for that query → EQUAL token budget vs my-system.
#   - my-system context = interpret(query) seeds → mesh_gather → its e_ facts (committed code as-is).
#   - SAME answer prompt (loop.ANSWER_PROMPT) for both context conditions; base = the bare question.
#
# SCORING — two DETERMINISTIC signals, no LLM judge (at n=24 a local 8B judge is the slowest call and
# the least reliable mean; the report dumps every prompt+context+answer for human grading instead):
#   - gold coverage  : wd_qa.coverage — fraction of gold answer surfaces named in the answer.
#   - bridge-hit     : did the answer name the INTERMEDIATE concept B (the 2-hop tell) — reached vs guessed.
# 8B = the repo's committed "7B" tier tag (llama3:8b); registered at runtime so call_llm("8B") resolves
# exactly instead of .get-falling-back to the 3B tag. Resumable: appends to a JSONL, skips done keys.
# Run: OMP_NUM_THREADS=1 PYTHONPATH=src:scripts venv/bin/python scripts/strong_bench.py
import argparse, hashlib, json, re, statistics as st
from pathlib import Path
import torch; torch.set_num_threads(1)
import llm
from llm import call_llm, ensure_ollama_models, warm_models
from graph import GraphStore
from field.seams import interpret
from field.loop import mesh_gather, ANSWER_PROMPT
from reason_bench import ctx_rag, e_facts
from wd_qa import coverage

M8B = "llama3:8b"                           # the committed "7B" tier tag IS the 8B model

# CURATED astrophysics 2-hop questions — each names the SEED concept and asks for a target reachable only
# by chaining through an intermediate (BRIDGE). gold = answer-concept surfaces; bridge = the intermediate
# (its presence in an answer is the 2-hop tell). All seed/bridge/gold are verified present in the store at
# runtime (verify_query); the answer concept is distinct from the seed so query-seeded dense retrieval has
# to span a hop. `chain` is a human label of the A→B→C path. Curated (not auto-derived) because the prose
# corpus has no relational schema — the report flags this and shows the verification.
CURATED = [
    {"seed": "Chandrasekhar limit", "chain": "Chandrasekhar limit → white dwarf → supernova/neutron star",
     "q": "A white dwarf that is pushed past the Chandrasekhar limit can no longer support itself against "
          "gravity. What explosive event or compact object does this collapse produce?",
     "bridge": ["white dwarf"], "gold": ["supernova", "type ia", "neutron star"]},
    {"seed": "Schwarzschild radius", "chain": "Schwarzschild radius → black hole → event horizon",
     "q": "The Schwarzschild radius marks a critical boundary around a fully collapsed mass. What is that "
          "boundary called, and what kind of object does it surround?",
     "bridge": ["black hole"], "gold": ["event horizon", "singularity"]},
    {"seed": "dark energy", "chain": "dark energy → accelerating expansion → de Sitter universe",
     "q": "Dark energy drives the accelerating expansion of the universe toward a particular asymptotic "
          "spacetime in the far future. What is that empty, exponentially expanding end-state universe called?",
     "bridge": ["cosmological constant", "accelerating expansion", "acceleration"], "gold": ["de sitter"]},
    {"seed": "Hawking radiation", "chain": "Hawking radiation → black hole thermal emission → evaporation",
     "q": "Because of Hawking radiation, an isolated black hole slowly loses mass over time. What is the "
          "ultimate long-term fate of such a black hole?",
     "bridge": ["black hole", "thermal"], "gold": ["evaporat", "disappear"]},
    {"seed": "recombination", "chain": "recombination → neutral hydrogen / last scattering → CMB",
     "q": "During the epoch of recombination, electrons and protons combined into neutral atoms and the "
          "universe became transparent, releasing radiation still observed today. What is that relic radiation called?",
     "bridge": ["last scattering", "neutral"], "gold": ["cosmic microwave background", "microwave"]},
]

# the 6 conditions as (label, context-kind, answerer-tier). context-kind is shared across the two tiers
# (context is model-independent) so it is rendered once and answered twice — that IS the fairness design.
CONDS = [("3B base", "base", "3B"), ("8B base", "base", "8B"),
         ("3B + RAG", "rag", "3B"), ("8B + RAG", "rag", "8B"),
         ("3B + my-system", "system", "3B"), ("8B + my-system", "system", "8B")]

# bridge_hit: did the answer name the intermediate entity B? (word-boundary, lowercased) — the cheap,
# objective 2-hop tell: a model that names the bridge actually traversed the first hop rather than guessing.
def bridge_hit(ans: str, bridges: list[str]) -> bool:
    a = (ans or "").lower()
    return any(re.search(r"\b" + re.escape(b.lower()) + r"\b", a) for b in bridges)

# node_has: count of node surfaces containing a term (store-presence check for verification).
def node_has(store, term: str) -> int:
    return store._con.execute("SELECT COUNT(*) FROM nodes WHERE lower(text) LIKE ?",
                              (f"%{term.lower()}%",)).fetchone()[0]

# select_queries: the curated astro set, each verified against THIS store — the seed grounds (find_vec),
# and the bridge + every gold surface are present as node text. `verify` records what resolved so the
# report can show the questions are genuinely answerable here (not invented). Shaped like wd_qa rows
# (subj/subj_id/rel/gold/bridges) so the rest of the pipeline is unchanged.
def select_queries(store, n: int) -> list[dict]:
    out = []
    for c in CURATED[:n]:
        gsv = store.find_vec(c["seed"], 1)
        grounded = (store.text(gsv[0]).split("|")[0].strip() if gsv else "—")
        verify = {"seed_grounds_to": grounded,
                  "bridge_present": {b: node_has(store, b) for b in c["bridge"]},
                  "gold_present": {g: node_has(store, g) for g in c["gold"]}}
        out.append({"q": c["q"], "subj": c["seed"], "subj_id": "c_" + hashlib.sha1(c["seed"].encode()).hexdigest()[:10],
                    "rel": c["chain"], "gold": c["gold"], "bridges": c["bridge"], "verify": verify})
    return out

# build_contexts: the model-independent context per type, computed ONCE per query. Returns
# {kind: (facts_list, prompt)}. mesh is computed first so its fact count K sets the RAG budget.
def build_contexts(store, q: dict) -> tuple[dict, list[str]]:
    interp = interpret(q["q"], store)
    seeds = interp["seeds"] or store.find_vec(q["q"], 2)     # interpret first; find_vec as a floor
    mesh = mesh_gather(store, seeds)
    sys_facts = e_facts(store, mesh.node_ids)
    K = max(len(sys_facts), 1)                               # equal budget; never request 0
    rag_facts = ctx_rag(store, q["q"], K)
    render = lambda fs: "\n".join(f"- {f}" for f in fs)
    return {
        "base": ([], q["q"]),                                # base prompt = the bare question
        "rag": (rag_facts, ANSWER_PROMPT.format(ctx=render(rag_facts), q=q["q"])),
        "system": (sys_facts, ANSWER_PROMPT.format(ctx=render(sys_facts), q=q["q"]))}, seeds

def answer(tier: str, prompt: str) -> str:
    return call_llm(prompt, tier, options={"temperature": 0, "num_predict": 280}).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=".di-ui/graph.astro.merged.sqlite")
    ap.add_argument("--n", type=int, default=5)             # number of curated astro queries (3-5)
    ap.add_argument("--out", default="runs_experiment/strong_bench")
    a = ap.parse_args()
    llm._MODEL_TAGS["8B"] = M8B                             # register so call_llm("8B") resolves exactly
    ensure_ollama_models()
    tags = {t: llm._resolve_tag(t) for t in ("3B", "8B")}
    warm_models(("3B", "8B"))
    store = GraphStore(a.store)

    queries = select_queries(store, a.n)
    jsonl = Path(a.out + ".jsonl"); jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            try: r = json.loads(ln); done[r["key"]] = r
            except Exception: pass
    fh = jsonl.open("a")

    # PHASE 1 — compute every query's context once (the graph-heavy part: interpret + mesh_gather + RAG).
    # PHASE 2 — answer grouped by TIER (all 3B, then all 8B) so the single local Ollama loads each model
    # ONCE instead of thrashing 3B↔8B on every condition (the dominant wall-time cost — see header).
    ctx_by_q = {q["subj_id"]: build_contexts(store, q) for q in queries}
    rows = []
    for tier in ("3B", "8B"):
        for qi, q in enumerate(queries):
            ctxs, seeds = ctx_by_q[q["subj_id"]]
            for label, kind, t in CONDS:
                if t != tier: continue
                key = f"{q['subj_id']}|{kind}|{tier}"
                if key in done: rows.append(done[key]); continue
                facts, prompt = ctxs[kind]
                ans = answer(tier, prompt)
                n_hit, n_gold, hits = coverage(ans, q["gold"])
                bh = bridge_hit(ans, q["bridges"])
                rec = {"key": key, "qi": qi, "q": q["q"], "subj": q["subj"], "rel": q["rel"],
                       "gold": q["gold"], "bridges": q["bridges"], "seeds": seeds, "verify": q["verify"],
                       "label": label, "kind": kind, "tier": tier, "prompt": prompt, "facts": facts,
                       "n_facts": len(facts), "answer": ans, "cov": n_hit / max(n_gold, 1), "n_hit": n_hit,
                       "n_gold": n_gold, "hits": hits, "bridge_hit": bh}
                fh.write(json.dumps(rec) + "\n"); fh.flush(); rows.append(rec)
                print(f"  q{qi+1}/{len(queries)} {label:16} cov={rec['cov']:.2f} bridge={'Y' if bh else 'n'}",
                      flush=True)
    fh.close(); store.close()
    write_report(rows, queries, a.out + ".md", a.store, tags)

# write_report: the human-readable result file — per query, the full context + exact prompt + every
# response (with coverage + bridge-hit pre-filled), then the condition × coverage × bridge-hit summary,
# a blank hand-evaluation worksheet, and an honest verdict.
def write_report(rows, queries, out, store, tags):
    by = {(r["subj"], r["rel"], r["kind"], r["tier"]): r for r in rows}
    mean = lambda xs: st.mean(xs) if xs else float("nan")
    L = ["# Strong bench — cheap+structure vs expensive+RAG on astrophysics multi-hop queries", "",
         f"Store `{store}` (prose astrophysics corpus). Answerers: 3B=`{tags['3B']}`, 8B=`{tags['8B']}`. "
         "No LLM judge — two deterministic signals (gold coverage + bridge-hit) plus the full "
         "transcript below for human grading.", "",
         f"{len(queries)} **curated** astrophysics 2-hop questions (seed concept → bridge → answer). The "
         "astro corpus has no relational schema (prose predicates of/in/cause), so wd_qa's auto-derived "
         "gold doesn't port — instead each question is hand-written and its seed/bridge/gold are VERIFIED "
         "present in this store (shown per query). The answer concept is distinct from the seed, so "
         "query-seeded dense retrieval must span a hop. RAG and my-system get an EQUAL fact budget "
         "(K = the mesh's fact count). Context is computed once per type and answered by both tiers "
         "(the prompt shown under each type is sent verbatim to 3B and 8B).", "",
         "**Signals.** *coverage* = fraction of gold answer surfaces named. *bridge* = did the answer "
         "name the intermediate concept B (the 2-hop tell: reached the chain vs guessed the endpoint). "
         "_Curation caveat: gold is hand-picked, not machine-derived; n is small — read the transcript._", ""]

    for qi, q in enumerate(queries):
        v = q.get("verify", {})
        bp = ", ".join(f"{b}({n})" for b, n in v.get("bridge_present", {}).items())
        gp = ", ".join(f"{g}({n})" for g, n in v.get("gold_present", {}).items())
        L += ["", "---", "", f"## Q{qi+1}. {q['q']}", "",
              f"- **Chain:** {q['rel']}",
              f"- **Bridge (intermediate):** {', '.join(q['bridges'])}",
              f"- **Gold answer(s):** {', '.join(q['gold'])}",
              f"- **Store-verified:** seed `{q['subj']}` grounds to `{v.get('seed_grounds_to','?')}` · "
              f"bridge node-surfaces: {bp} · gold node-surfaces: {gp}", ""]
        for kind, title in (("base", "Base (closed-book — no context)"),
                            ("rag", "RAG (dense top-K e_ facts)"), ("system", "My system (mesh_gather)")):
            r3 = by.get((q["subj"], q["rel"], kind, "3B")); r8 = by.get((q["subj"], q["rel"], kind, "8B"))
            ref = r3 or r8
            if not ref: continue
            L += [f"### {title}"]
            if kind == "base":
                L += ["", "_Context: none. Prompt = the bare question._", ""]
            else:
                L += ["", f"**Rendered context ({ref['n_facts']} facts):**", "", "```"]
                L += [f"- {f}" for f in ref["facts"]] + ["```", ""]
            L += ["**Exact prompt (sent verbatim to both 3B and 8B):**", "", "```", ref["prompt"], "```", ""]
            for tier, rr in (("3B", r3), ("8B", r8)):
                if not rr: continue
                sc = f"coverage {rr['n_hit']}/{rr['n_gold']} · bridge {'HIT' if rr['bridge_hit'] else 'miss'}"
                ans = (rr["answer"] or "(empty)").replace("\n", "\n> ")
                L += [f"**{tier} response** — {sc}", "", "> " + ans, ""]

    # ── summary: condition × mean coverage × bridge-hit rate ──
    L += ["", "---", "", "## Summary — condition × mean coverage × bridge-hit rate", "",
          "| condition | mean coverage | bridge-hit rate | n |", "|---|---|---|---|"]
    agg = {}
    for label, kind, tier in CONDS:
        rs = [r for r in rows if r["label"] == label]
        if not rs: continue
        cov = mean([r["cov"] for r in rs]); br = mean([1.0 if r["bridge_hit"] else 0.0 for r in rs])
        agg[label] = (cov, br)
        L.append(f"| {label} | {cov:.3f} | {br:.2f} | {len(rs)} |")

    # ── the headline question: 3B+my-system vs 8B+RAG ──
    a3, a8 = agg.get("3B + my-system"), agg.get("8B + RAG")
    L += ["", "## Headline — cheap+structure vs expensive+RAG", ""]
    if a3 and a8:
        L += [f"- **3B + my-system:** coverage {a3[0]:.3f}, bridge-hit {a3[1]:.2f}",
              f"- **8B + RAG:**       coverage {a8[0]:.3f}, bridge-hit {a8[1]:.2f}",
              f"- **Δ (3B+system − 8B+RAG):** coverage {a3[0]-a8[0]:+.3f}, bridge-hit {a3[1]-a8[1]:+.2f}", ""]

    # ── hand-evaluation worksheet: blank correctness/groundedness columns for the human ──
    L += ["## Hand-evaluation worksheet", "",
          "Coverage + bridge are pre-filled (deterministic). Fill correct/grounded by reading the "
          "transcript above (1=poor … 5=excellent). correct = reaches the gold via the intermediate; "
          "grounded = specific claims are supported, not fabricated.", "",
          "| Q | condition | cov | bridge | correct (1-5) | grounded (1-5) | notes |",
          "|---|---|---|---|---|---|---|"]
    for qi, q in enumerate(queries):
        for label, kind, tier in CONDS:
            rr = by.get((q["subj"], q["rel"], kind, tier))
            if not rr: continue
            L.append(f"| Q{qi+1} | {label} | {rr['n_hit']}/{rr['n_gold']} | "
                     f"{'HIT' if rr['bridge_hit'] else '—'} |  |  |  |")
    L += ["", "## Verdict", "", "_(fill from the numbers + transcript above; n is small — read per-query.)_", ""]
    Path(out).write_text("\n".join(L))
    print("\n".join(L[:34])); print(f"\n→ {out}")

if __name__ == "__main__":
    main()
