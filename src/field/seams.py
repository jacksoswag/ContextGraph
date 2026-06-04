from __future__ import annotations
# The LLM seams: the two boundary calls of the single retrieval pipeline (no recursion).
#   interpret  — prompt → 3B REFINE (declarative, multifaceted) → spaCy seed-hyperedge
#                extraction → real node-ids in S (the seeds). spaCy OWNS the endpoints; the
#                LLM only reshapes the query text it parses, it never mints a seed.
#   render_*   — turn a gathered Mesh into a prompt block (flat) or a seed-rooted outline.
# The recursive decompose/synthesize seams were deleted with the reasoning loop — the field now
# answers in ONE permissive call over the gathered region (see field.loop.respond). Nothing here
# touches the field math; the LLM is injected (default call_json) so it stays mock-testable.
from typing import Callable, Iterable
from .gather import Mesh
from llm import call_json

LLM = Callable[..., dict]   # (prompt, model) -> dict
Parse = Callable[[str], Iterable[dict]]   # text -> spaCy clause forest

# ── the refine prompt (📍4) ─────────────────────────────────────────────────────────
# Rewrite the (often interrogative) query as short DECLARATIVE statements that NAME every entity
# and concept it touches — one per facet — so spaCy (which owns endpoint extraction) gets clean
# noun-phrase subjects/objects instead of a question's "what/who" gaps. Multifaceted by design:
# each distinct facet becomes its own statement → its own seed.
REFINE_PROMPT = (
    "Rewrite this question as 1-4 short DECLARATIVE statements that together name every entity, "
    "concept, and relationship it asks about (use a plain noun for whatever is being asked for). "
    "Put each distinct facet in its own statement. Name the SUBJECT explicitly in EVERY statement — "
    "never use a pronoun (he/she/it/they/this); repeat the entity's name instead. Write each statement "
    "as a simple subject-verb-object fact ('Albert Einstein won the Nobel Prize'), NOT as a count or "
    "definition ('the number of ... was N').\n"
    'Return JSON: {{"statements": ["...", "..."], "intent": "..."}}\n\nQuery: {query}')

_EVENT = "[event]"   # objectless/intransitive endpoint — elided to just the relation (mirrors astro_bench._node)

# render_props: the gathered region as the corpus's nested-PROPOSITION lines (astro_bench._node form),
# one bullet per ROOT edge — a mesh e_ edge no OTHER mesh edge contains (children are written out by the
# recursion, so a nested fact never also lists bare). A plain fact renders (subj -rel- obj); a fact→fact
# connection renders (factA) =rel=> (factB): the OUTERMOST relation is =rel=>, deeper ones -rel-. An
# [event] endpoint elides to (subj -rel-) / (-rel- obj). Same surfaces as render_outline; only the shape
# differs (recursive parens vs indented tree). Returns the rendered block.
def render_props(mesh: Mesh, store, max_nodes: int = 80) -> str:
    ids = [n for n in mesh.node_ids[:max_nodes] if n.startswith("e_")]; idset = set(ids)
    contained = {c for e in ids for c in (store.children(e) or ()) if c in idset}
    roots = [e for e in ids if e not in contained]        # relevance order preserved
    return "\n".join(f"- {_prop(store, e, top=True)}" for e in roots) or "(empty)"

# _prop: recursive endpoint render — node → its text; edge → (src rel tgt), recursing into nested facts.
# top marks the OUTERMOST fact→fact relation with =rel=> (the corpus connection form); below it is -rel-.
def _prop(store, nid: str, top: bool = False) -> str:
    tri = store.triple(nid)
    if tri is None: return store.text(nid) or nid
    s, rel, t = tri
    if not t: return f"({_prop(store, s)} -{rel}-)"        # unary intransitive (empty target)
    sr, tr = _prop(store, s), _prop(store, t)
    if top and s.startswith("e_") and t.startswith("e_"): return f"({sr}) ={rel}=> ({tr})"
    if tr == _EVENT: return f"({sr} -{rel}-)"
    if sr == _EVENT: return f"(-{rel}- {tr})"
    return f"({sr} -{rel}- {tr})"

def _is_dict(d) -> bool: return isinstance(d, dict)

# _phrase_str: small models often emit entities as dicts ({"name": …} / schema.org JSON-LD) instead of
# bare strings — pull the surface form so grounding isn't handed a stringified dict that matches nothing.
def _phrase_str(ph) -> str:
    if isinstance(ph, dict):
        return str(ph.get("name") or ph.get("text") or ph.get("label") or ph.get("value") or "")
    return str(ph)

# Administrative / namespace nodes that are not entities (Wikipedia/Wikidata meta). A relation fragment
# or bare name sometimes embeds closer to the "Category:X" page than to X itself; these are filtered out
# of grounding so the real entity (or proposition) is seeded instead.
_META_PREFIX = ("category:", "wikipedia:", "wikiproject", "template:", "portal:", "help:", "module:",
                "draft:", "list of ", "index of ")
def _is_meta(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(t.startswith(p) for p in _META_PREFIX)

# _ground: phrase → up to k real endpoints (nodes ∪ reified edges), skipping meta nodes. find_vec is
# over-fetched and meta pages dropped so "albert einstein" lands on the entity, not "category:albert
# einstein". Edges (e_) are always real propositions. No embedder ⇒ lexical find.
def _ground(store, phrase: str, k: int = 1) -> list[str]:
    finder = getattr(store, "find_vec", store.find); txt = getattr(store, "text", lambda n: n)
    out: list[str] = []
    for nid in finder(phrase, k + 4):
        if nid.startswith("e_") or not _is_meta(txt(nid) or nid):
            if nid not in out: out.append(nid)
        if len(out) >= k: break
    return out

# _pick_seeds: choose the seed set from the per-facet grounded candidates, given a query-relevance score
# and surface text per candidate. Three rules, in order:
#   1. DEDUP surface-form aliases — a node whose text is wholly contained in (or contains) a higher-scored
#      node's is the same entity ("einstein" ⊂ "albert einstein"); keep the better one. Reified edges (e_)
#      are distinct propositions, never aliased away. (Anchor cosine can't separate aliases from related
#      entities — "nobel prize"/"nobel prize in physics" sit at 0.87 — so containment is the signal.)
#   2. PER-FACET floor — keep each facet's top-`per_facet` surviving candidates above an absolute floor, so
#      a dominant subject (it grounds in every facet) can't crowd out each facet's distinct OBJECT (the
#      Fermi→US/Italy collapse). The subject is seeded once; every facet still contributes its own entity.
#   3. CAP — the highest-scored `max_seeds` overall.
def _pick_seeds(facets: list[list[str]], score: dict[str, float], text: dict[str, str], max_seeds: int,
                *, floor: float = 0.22, per_facet: int = 2) -> list[str]:
    flat: list[str] = []
    for fc in facets:
        for nid in fc:
            if nid not in flat: flat.append(nid)
    if len(flat) <= 1: return flat[:max_seeds]
    def alias(a, b):
        if a.startswith("e_") or b.startswith("e_"): return False
        ta, tb = text.get(a, a), text.get(b, b)
        return min(len(ta), len(tb)) >= 4 and (ta in tb or tb in ta)
    kept: list[str] = []
    for nid in sorted(flat, key=lambda n: -score[n]):
        if not any(alias(nid, k) for k in kept): kept.append(nid)
    keptset = set(kept); seeds: list[str] = []
    for fc in facets:
        surv = sorted([n for n in fc if n in keptset], key=lambda n: -score[n])
        for nid in surv[:per_facet]:
            if score[nid] >= floor and nid not in seeds: seeds.append(nid)
    seeds.sort(key=lambda n: -score[n])
    return seeds[:max_seeds]

# _select_seeds: score every grounded candidate by query cosine, then hand the per-facet groups to
# _pick_seeds (dedup + per-facet floor + cap). No embedder/anchors ⇒ keep the extractor's flat order.
def _select_seeds(facets: list[list[str]], query: str, store, max_seeds: int) -> list[str]:
    flat = list(dict.fromkeys(n for fc in facets for n in fc))
    if len(flat) <= 1: return flat[:max_seeds]
    try:
        import numpy as np
        from embed import embed, unpack
        qb = embed(query); anc = getattr(store, "anchor", None)
        if qb is None or anc is None: return flat[:max_seeds]
        qv = unpack(qb); txt = getattr(store, "text", lambda n: n)
        score: dict[str, float] = {}; text: dict[str, str] = {}
        for nid in flat:
            a = anc(nid)
            if a is None: return flat[:max_seeds]
            score[nid] = float((a / (np.linalg.norm(a) + 1e-9)) @ qv)
            text[nid] = (txt(nid) or nid).split("|")[0].strip().lower()
        return _pick_seeds(facets, score, text, max_seeds)
    except Exception:
        return flat[:max_seeds]

# Endpoint tokens that carry no grounding signal as a relation target (pronouns the refine didn't
# resolve, the [event] pending marker, interrogatives) → the fragment is rendered subject+relation only,
# and they are not seeded as bare nodes.
_FRAG_SKIP = frozenset({"[event]", "they", "he", "she", "it", "them", "this", "that", "these",
                        "those", "there", "who", "which", "what", "i", "we", "you"})

# _spacy_fragments: walk the spaCy clause forest and emit, per clause, the RELATION FRAGMENT — the
# rendered "subject relation object" (or "subject relation" when the object is missing/a pronoun) — then
# the bare node endpoints. Relation fragments come FIRST because find_vec ranks reified edges (e_)
# alongside nodes, so a fragment maps onto the PROPOSITION it describes (a first-class endpoint: e.g.
# "einstein win nobel prize" → the e_ edge "albert einstein award received nobel prize in physics"),
# while the bare entities anchor the subject against find_vec drift. Edges, nodes, and hyperedges are all
# endpoints — seed all of them. The verb need not match the store's relation lexically; find_vec is
# semantic (win ≈ award received). parse is injected (default = the ingest extractor, lazily imported so
# importing this module never pulls spaCy); any failure degrades to []. De-duped, document order kept.
def _spacy_fragments(text: str, parse: Parse | None = None) -> list[str]:
    if not text.strip(): return []
    if parse is None:
        try: from ingest.extraction import extract_clauses as parse
        except Exception: return []
    seen: set[str] = set(); frags: list[str] = []; nodes: list[str] = []
    def lab(x):
        if not (isinstance(x, dict) and x.get("type") == "node"): return None
        return (x.get("text") or "").strip().lower() or None
    def add(bucket, s):
        if s and s not in seen: seen.add(s); bucket.append(s)
    def walk(c):
        if not isinstance(c, dict) or c.get("type") != "edge": return
        s, t = c.get("source"), c.get("target")
        rel = (c.get("rel") or "").strip().lower().replace("_", " ")
        sl, tl = lab(s), lab(t)
        if sl and rel:                                       # the relation fragment → a proposition seed
            obj = tl if (tl and tl not in _FRAG_SKIP) else None
            add(frags, f"{sl} {rel} {obj}" if obj else f"{sl} {rel}")
        if sl and sl not in _FRAG_SKIP: add(nodes, sl)       # bare entities anchor the subject
        if tl and tl not in _FRAG_SKIP: add(nodes, tl)
        for x in (s, t): walk(x)
    try:
        for c in parse(text): walk(c)
    except Exception:
        return frags + nodes
    return frags + nodes

# interpret: query → {seeds:[real endpoint_id], intent}. The 3B REFINE rewrites the query into
# declarative statements (multifaceted — one per facet). For each facet, spaCy emits RELATION FRAGMENTS
# (subject+relation+object) plus the bare entities; each is resolved to the nearest ENDPOINT in S via
# store.find_vec (embedding-NN over nodes ∪ reified edges). A relation fragment grounds to the reified
# e_ edge — the PROPOSITION — so different facets of the same entity seed DIFFERENT regions ("what did
# he do" → occupation edges; "how many nobel prizes" → award edges), while the bare entity anchors the
# subject against find_vec drift. spaCy's parser is tuned for article prose and whiffs on terse text, so
# when a facet yields no fragment the statement itself is grounded (find_vec embeds it near its entity).
# Candidates are filtered to those closest to the query (_select_seeds) so off-subject phrases don't
# derail the gather. When nothing resolves at all, the raw query is grounded broadly so even a one-word
# query seeds. Seeds are a MIX of edge-endpoints (propositions) and node-endpoints (entities).
def interpret(query: str, store, *, llm: LLM = call_json, model: str = "3B", max_seeds: int = 6,
              parse: Parse | None = None) -> dict:
    out = llm(REFINE_PROMPT.format(query=query), model)
    stmts = out.get("statements") if _is_dict(out) else None
    intent = out.get("intent") if _is_dict(out) else None
    statements = [s for st in (stmts or []) if (s := _phrase_str(st).strip())]
    facets: list[list[str]] = []                             # one candidate group per facet (statement)
    for st in statements:
        fc: list[str] = []
        for ph in (_spacy_fragments(st, parse) or [st]):     # relation fragments → edges; whiff ⇒ facet text
            for nid in _ground(store, ph, 1):
                if nid not in fc: fc.append(nid)
        if fc: facets.append(fc)
    if not facets:                                           # nothing resolved → ground the raw query broadly
        g = _ground(store, query, max_seeds)
        if g: facets = [g]
    return {"seeds": _select_seeds(facets, query, store, max_seeds),
            "intent": str(intent) if intent else query}
