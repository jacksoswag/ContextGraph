# DeepWebDigest: cross-paragraph stitching over an article's per-sentence
# clauses. The fast path (extract_clauses) already pulls SVO triples, runs
# coreferee for pronouns, types same-paragraph discourse rels
# (however/therefore/next), and anchors entities within paragraphs. This pass
# adds what spaCy alone can't:
#   1. cross-paragraph implicit relations (paragraph 1 sets up X, paragraph 4
#      treats Y as following from X, without saying so)
#   2. an article-level main-idea anchor
# Runs on a small LLM (3B default) — input is the already-extracted triples,
# not the raw article — so the prompt is tiny relative to a full-article pass.
# Pure producer: returns an IngestResult of cross-clause edges; no graph store.
import hashlib, json, logging, os, re, sqlite3, time
from contextlib import closing
from dataclasses import dataclass, field, asdict
from typing import Iterator

from llm import call_json

LOGGER = logging.getLogger(__name__)

_CACHE_PATH = os.getenv("RESEARCH_CACHE_PATH", ".di-ui/research_cache.sqlite")
_DEFAULT_MODEL = os.getenv("DI_DEEP_INGEST_MODEL", "7B")
_MAX_TRIPLES = int(os.getenv("DI_DEEP_INGEST_MAX_TRIPLES", "120"))

# Relation strings the LLM may emit. Open-set in spirit (matches the rest of
# the graph), but constrained by a length / token-shape filter so callers can't
# inject prose into rel labels. spaCy already produces verb-lemma rels — the
# digest just adds cross-paragraph rels in the same vocabulary.
# Allow at most one underscore so 2-token compound verbs ("leads_to") pass but
# prose ("it_clearly_demonstrates") doesn't. Length cap keeps labels terse.
_RELATION_RE = re.compile(r"^[a-z]{2,20}(?:_[a-z]{2,20})?$")


@dataclass
class IngestEdge:
    relation: str            # short open-set verb-like string
    source_clause_id: str    # references one of the extracted triples
    target_clause_id: str
    source_text: str         # short label substring of the source clause
    target_text: str
    confidence: float
    model: str = ""
    verified: bool = False


@dataclass
class IngestResult:
    article_hash: str
    edges: list[IngestEdge] = field(default_factory=list)
    cached: bool = False
    triples_seen: int = 0    # how many spaCy clauses fed the LLM
    main_idea: str = ""      # article-level anchor returned by the LLM


# Article hash for caching. SHA256 truncated; stable across runs.
def hash_article(body_text: str) -> str:
    return hashlib.sha256((body_text or "").encode("utf-8")).hexdigest()[:24]


# Cache table lives in research_cache.sqlite (shared with the rest of web research).
def _ensure_cache(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS deep_ingest (
               article_hash TEXT PRIMARY KEY,
               model TEXT,
               created_at INTEGER,
               edges_json TEXT,
               main_idea TEXT
           )"""
    )
    conn.commit()
    return conn


def _cache_get(article_hash: str) -> tuple[list[dict], str] | None:
    try:
        with closing(_ensure_cache(_CACHE_PATH)) as conn:
            row = conn.execute(
                "SELECT edges_json, main_idea FROM deep_ingest WHERE article_hash = ?",
                (article_hash,),
            ).fetchone()
        if row and row[0]:
            return json.loads(row[0]), row[1] or ""
    except Exception as exc:
        LOGGER.warning("deep_ingest cache read failed: %s", exc)
    return None


def _cache_put(article_hash: str, model: str, edges: list[dict], main_idea: str) -> None:
    try:
        with closing(_ensure_cache(_CACHE_PATH)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO deep_ingest"
                " (article_hash, model, created_at, edges_json, main_idea)"
                " VALUES (?, ?, ?, ?, ?)",
                (article_hash, model, int(time.time()), json.dumps(edges), main_idea),
            )
            conn.commit()
    except Exception as exc:
        LOGGER.warning("deep_ingest cache write failed: %s", exc)


# Reduce a spaCy clause-edge dict to a flat (subj, rel, obj) tuple of strings.
# Returns None when the edge isn't a clean source-rel-target shape (e.g. a
# discourse-only edge that points at another edge — the digest works at the
# concept-level, not at hyperedges between facts).
def _clause_to_triple(edge: dict) -> tuple[str, str, str] | None:
    if not edge or edge.get("type") != "edge":
        return None
    src = edge.get("source") or {}
    tgt = edge.get("target") or {}
    # Skip hyperedges (source/target is itself an edge) — too structural for
    # the stitch pass which reasons over concepts.
    if src.get("type") != "node" or tgt.get("type") != "node":
        return None
    rel = (edge.get("rel") or "").strip().lower()
    s_text = (src.get("text") or "").strip()
    t_text = (tgt.get("text") or "").strip()
    if not rel or not s_text or not t_text:
        return None
    return s_text, rel, t_text


# Returns (triples, focus_concepts): triples is list of (cid, subj, rel, obj),
# focus_concepts is list of unique subject/object strings for seeding retrieval.
def _extract_triples(body_text: str, article_hash: str) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    # Coref runs once inside extract_clauses (gated by COREF_ENABLED) — no pre-pass here.
    from ingest.extraction import extract_clauses

    triples: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    focus: list[str] = []
    focus_seen: set[str] = set()
    for idx, edge in enumerate(extract_clauses(body_text, llm_fallback=False)):
        triple = _clause_to_triple(edge)
        if triple is None or triple in seen: continue
        seen.add(triple)
        cid = f"{article_hash}:c{len(triples)}"
        triples.append((cid, *triple))
        for concept in (triple[0], triple[2]):
            if concept and concept not in focus_seen:
                focus_seen.add(concept)
                focus.append(concept)
        if len(triples) >= _MAX_TRIPLES: break
    return triples, focus


# Validator: each digest edge must reference clause IDs we extracted, name a
# relation string that fits the verb-lemma shape, and (substring) ground its
# source/target text in the original clauses. Short IDs (cN) get re-prefixed
# to the article-hash form before lookup so the LLM can use compact IDs in its
# output without losing provenance.
def _validate(records: list[dict], triple_index: dict[str, tuple[str, str, str]],
              model: str, *, article_hash: str = "") -> list[IngestEdge]:
    def _norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    # Build short→full ID map. Useful when the LLM drops the article-hash
    # prefix (common for 3B models — saves tokens, easier to predict).
    short_to_full = {cid.split(":")[-1]: cid for cid in triple_index}

    def _full(cid: str) -> str:
        cid = cid.strip()
        # LLM may have emitted "c8" instead of "<hash>:c8"; rehydrate.
        return cid if cid in triple_index else short_to_full.get(cid, cid)

    kept: list[IngestEdge] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        rel = str(rec.get("relation", "")).strip().lower().replace(" ", "_")
        if not _RELATION_RE.match(rel):
            continue
        sid = _full(str(rec.get("source_clause_id", "") or ""))
        tid = _full(str(rec.get("target_clause_id", "") or ""))
        if sid not in triple_index or tid not in triple_index:
            continue
        # Reject within-clause echoes: the digest's job is CROSS-clause
        # connections. A self-loop carries no new information.
        if sid == tid:
            continue
        s_text = str(rec.get("source_text", "")).strip()
        t_text = str(rec.get("target_text", "")).strip()
        if not s_text or not t_text:
            continue
        # Reject when the LLM copied the VERB (relation) instead of a noun.
        # The relation field already captures the connection; source/target
        # text must be the noun-phrase entities being connected. Two checks:
        # (1) reject exact-match to either clause's relation, (2) require
        # substring grounding in the subj or obj of the cited clause — NOT
        # the rel position.
        s_triple, t_triple = triple_index[sid], triple_index[tid]
        s_subj, s_rel, s_obj = s_triple
        t_subj, t_rel, t_obj = t_triple
        s_norm, t_norm = _norm(s_text), _norm(t_text)
        if s_norm == _norm(s_rel) or s_norm == _norm(t_rel): continue
        if t_norm == _norm(s_rel) or t_norm == _norm(t_rel): continue
        # Substring grounding restricted to subj/obj positions.
        def _matches_noun_part(text_norm: str, triple: tuple[str, str, str]) -> bool:
            return bool(text_norm and (
                text_norm in _norm(triple[0]) or text_norm in _norm(triple[2])
            ))
        if not _matches_noun_part(s_norm, s_triple): continue
        if not _matches_noun_part(t_norm, t_triple): continue
        raw_conf = rec.get("confidence")
        try:
            conf = float(raw_conf) if raw_conf is not None else 0.6
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.5:
            continue
        kept.append(IngestEdge(
            relation=rel, source_clause_id=sid, target_clause_id=tid,
            source_text=s_text, target_text=t_text, confidence=conf,
            model=model, verified=True,
        ))
    return kept


# Build the stitching prompt. The LLM sees the already-extracted triples — far
# smaller than the raw article — and is asked for cross-clause connections
# only, using the same open-set relation vocabulary the graph uses elsewhere.
def _build_prompt(triples: list[tuple[str, str, str, str]]) -> str:
    # Show short IDs (cN) in the prompt to save tokens and stay easy to predict.
    # The validator rehydrates them with the article hash on the way back.
    triple_lines = "\n".join(
        f"{cid.split(':')[-1]}: ({s}) --{r}--> ({t})" for cid, s, r, t in triples
    )
    return f"""You are finding NEW CONNECTIONS BETWEEN PAIRS of facts already extracted
from an article. Each line is a fact triple "(subj) --verb--> (obj)" with
a clause ID like c3. Your job is to identify pairs of DIFFERENT facts that
are causally, temporally, or logically related — connections the extractor
already missed at sentence level.

DO NOT list facts. DO NOT echo existing triples. ONLY emit edges where
source_clause_id ≠ target_clause_id and the connection is meaningful.

Example input:
  c0: (france) --invade--> (russia)
  c1: (winter) --kill--> (army)
  c2: (napoleon) --retreat_from--> (moscow)
Expected output: an edge like
  {{"relation": "cause", "source_clause_id": "c0", "target_clause_id": "c2",
    "source_text": "france", "target_text": "moscow", "confidence": 0.8}}
(France invading caused the retreat from Moscow. Both source_text and
target_text are nouns from their respective clauses, never the verb.)

Output JSON only:
{{
  "main_idea": "<one short sentence — the article's central claim>",
  "edges": [
    {{
      "relation": "<ONE short lowercase verb, e.g. cause, support, enable, follow, refine, replace>",
      "source_clause_id": "<exact ID from list below>",
      "target_clause_id": "<exact ID from list below>",
      "source_text": "<the SUBJECT or OBJECT noun phrase from the source clause — NEVER the verb>",
      "target_text": "<the SUBJECT or OBJECT noun phrase from the target clause — NEVER the verb>",
      "confidence": <0.0..1.0>
    }}
  ]
}}

CRITICAL — source_text / target_text rules:
- Copy a NOUN PHRASE (the subject or object) from the cited clause.
- NEVER copy the verb. The "relation" field is the verb; source_text/target_text
  must be the entities the connection is BETWEEN.
- Example: clause "c3: (article 5) --establish--> (defense)"
  ✓ valid: source_text = "article 5" OR source_text = "defense"
  ✗ invalid: source_text = "establish"  (that's the verb, never use it as text)
- Copy verbatim — do not paraphrase, do not combine fields, do not abbreviate.

Other rules:
- Use ONLY clause IDs from the list — never invent.
- relation = ONE short lowercase word (a verb lemma). No spaces, no prose, no
  modal phrases like "would be" or "to be used".
- Skip a connection if you are not at least 60% confident.
- Prefer connections spanning paragraphs over adjacent ones.

Facts:
{triple_lines}
"""


# Public entry point. Pass body_text; returns a IngestResult (cache-aware).
def ingest_article(article_hash: str | None, body_text: str, *,
                   model: str | None = None, event_callback=None) -> IngestResult:
    model = model or _DEFAULT_MODEL
    article_hash = article_hash or hash_article(body_text)

    def _emit(event: str, payload: dict):
        if event_callback:
            try: event_callback(event, payload)
            except Exception: pass

    _emit("deep_ingest_started", {"article_hash": article_hash})

    triples, focus_concepts = _extract_triples(body_text, article_hash)
    triple_index = {cid: (s, r, t) for cid, s, r, t in triples}

    if not triples:
        return IngestResult(article_hash=article_hash, edges=[], triples_seen=0)

    cached = _cache_get(article_hash)
    if cached is not None:
        records, main_idea = cached
        edges = _validate(records, triple_index, model="cache", article_hash=article_hash)
        _emit("deep_ingest_finished", {
            "article_hash": article_hash, "edges": len(edges), "cached": True,
        })
        return IngestResult(article_hash=article_hash, edges=edges, cached=True,
                            triples_seen=len(triples), main_idea=main_idea)

    prompt = _build_prompt(triples)
    data = call_json(prompt, model=model, role="deep_ingest", event_callback=event_callback)
    records = data.get("edges", []) if isinstance(data, dict) else []
    main_idea = str(data.get("main_idea", "")).strip() if isinstance(data, dict) else ""

    edges = _validate(records, triple_index, model=model, article_hash=article_hash)
    _cache_put(article_hash, model, [asdict(e) for e in edges], main_idea)

    rejected = max(0, len(records) - len(edges))
    _emit("deep_ingest_finished", {
        "article_hash": article_hash, "edges": len(edges),
        "rejected": rejected, "triples_seen": len(triples), "cached": False,
    })
    if rejected:
        _emit("deep_ingest_rejected", {
            "article_hash": article_hash, "count": rejected,
        })

    return IngestResult(article_hash=article_hash, edges=edges, cached=False,
                        triples_seen=len(triples), main_idea=main_idea)


# Convenience iterator used by tests / scripts to inspect the extracted
# triples without invoking the LLM. Public so the architectural value-add is
# visible (spaCy does step 1; the LLM only does step 2).
def extract_triples_only(body_text: str, article_hash: str | None = None) -> list[tuple[str, str, str, str]]:
    article_hash = article_hash or hash_article(body_text)
    triples, _ = _extract_triples(body_text, article_hash)
    return triples


# ── Pending-edge completion ───────────────────────────────────────────────────
# Intransitive verbs become pending edges at extract time:
#   source --verb--> {text: "[event]"}  with _pending_completion=True.
# This pass batches the (subject, verb) pairs through one LLM call, asks for a
# plausible object noun phrase, and either rewrites the target (upgrade) or drops
# the edge (sweep). A producer transform over the in-memory clause stream — no
# graph store. Compose after extraction:
#   clauses = complete_pending_edges(list(ingest_text(text)))

_PENDING_BATCH = int(os.getenv("DI_PENDING_COMPLETION_BATCH", "30"))
_PENDING_MAX_BATCHES = int(os.getenv("DI_PENDING_COMPLETION_MAX_BATCHES", "20"))


# Walk the clause forest, yielding every edge dict (nested targets + modifiers).
def _iter_edges(clauses) -> Iterator[dict]:
    stack = list(clauses)
    while stack:
        c = stack.pop()
        if not isinstance(c, dict): continue
        if c.get("type") == "edge":
            yield c
            stack.extend([c.get("source"), c.get("target")])
            stack.extend(m.get("target") for m in c.get("modifiers", []))


def _is_pending(c) -> bool:
    return bool(
        isinstance(c, dict) and c.get("type") == "edge" and c.get("_pending_completion")
        and isinstance(c.get("source"), dict) and c["source"].get("type") == "node"
    )


# Ask the LLM to complete (subject, verb) pairs with likely noun-phrase targets.
# Returns {pair_id: completion_str}. Conservative — better to discard than fabricate.
def _llm_complete_pending(items: list[dict], *, model: str = "7B", event_callback=None) -> dict[int, str]:
    if not items: return {}
    lines = "\n".join(f"  {i['id']}: ({i['subj']}) --{i['verb']}--> ???" for i in items)
    prompt = f"""Complete these incomplete facts with the most likely object NOUN PHRASE.
Each line is (subject) --verb--> ??? . Suggest the missing object as a short
noun phrase (1-3 words). Use null when no plausible object exists — better to
skip than to guess.

Examples:
  Input:  (army) --retreat--> ???
  Output: {{"completions": [{{"id": 1, "object": "battlefield"}}]}}
  Input:  (purpose) --deter--> ???
  Output: {{"completions": [{{"id": 2, "object": "aggression"}}]}}
  Input:  (twelve country) --sign--> ???
  Output: {{"completions": [{{"id": 3, "object": "treaty"}}]}}
  Input:  (cat) --is--> ???
  Output: {{"completions": [{{"id": 4, "object": null}}]}}

Rules:
- Output ONE noun phrase per id, or null. NO sentences, NO verbs.
- Skip (null) when the subject+verb doesn't strongly imply an object.
- Each id MUST appear in your output exactly once.

Incomplete facts:
{lines}

Return JSON: {{"completions": [{{"id": <int>, "object": "<noun phrase or null>"}}, ...]}}
"""
    data = call_json(prompt, model=model, role="pending_completion", event_callback=event_callback)
    out: dict[int, str] = {}
    for rec in (data.get("completions") or []):
        try: pid = int(rec.get("id"))
        except (TypeError, ValueError): continue
        obj = rec.get("object")
        if obj is None: continue
        obj = str(obj).strip().lower()
        # Reject single-word verbs (rough heuristic against the LLM emitting another verb).
        if not obj or " " not in obj and obj.endswith("ing"): continue
        if not (2 <= len(obj) <= 60): continue
        out[pid] = obj
    return out


# Upgrade or sweep pending edges in a clause list. Completed edges get their
# [event] target rewritten and the _pending_completion flag cleared; uncompleted
# edges are dropped when discard_uncompleted (default), else left untouched.
# Returns a new list; non-pending edges pass through unchanged.
def complete_pending_edges(clauses, *, model: str = "7B", discard_uncompleted: bool = True,
                           event_callback=None) -> list[dict]:
    clauses = list(clauses)
    pending = [(i, c) for i, c in enumerate(clauses) if _is_pending(c)]
    if not pending: return clauses
    swept: set[int] = set()
    upgraded = 0
    cap = _PENDING_BATCH * _PENDING_MAX_BATCHES
    for start in range(0, min(len(pending), cap), _PENDING_BATCH):
        batch = pending[start : start + _PENDING_BATCH]
        items = [{"id": j, "subj": c["source"]["text"], "verb": c.get("rel", "")}
                 for j, (_, c) in enumerate(batch)]
        completions = _llm_complete_pending(items, model=model, event_callback=event_callback)
        for j, (idx, c) in enumerate(batch):
            comp = completions.get(j)
            if comp:
                c["target"] = {"type": "node", "text": comp, "pos": "NOUN"}
                c.pop("_pending_completion", None)
                upgraded += 1
            elif discard_uncompleted:
                swept.add(idx)
    if event_callback:
        try: event_callback("pending_completion", {
            "upgraded": upgraded, "swept": len(swept), "total": len(pending), "model": model})
        except Exception: pass
    return [c for i, c in enumerate(clauses) if i not in swept]


# ── Relative time resolution ──────────────────────────────────────────────────
# Extraction captures time_phrase strings (spaCy DATE/TIME entities) on edges.
# This pass batches them through the LLM → ISO date strings of variable precision
# and annotates each edge with resolved_date (empty string = attempted, no
# resolution possible, so re-runs skip it). A producer transform over the clause
# stream — no graph store. Compose after extraction:
#   clauses = resolve_dates(list(ingest_text(text)))


def _llm_resolve_dates(items: list[dict], *, model: str = "7B", event_callback=None) -> dict[int, str]:
    if not items: return {}
    lines = "\n".join(f"  {i['id']}: {i['phrase']!r}" for i in items)
    prompt = f"""Convert these time phrases to ISO date strings with appropriate precision.

Resolution rules:
- Explicit year ("1949", "in 1949")               → "1949"
- Year+month ("February 2022", "Feb 2022")        → "2022-02"
- Full date ("April 4, 1949", "March 4 2024")     → "1949-04-04"
- Named historical events                         → use the known date:
    "September 11", "9/11", "2001 attacks"         → "2001-09-11"
    "World War II"                                  → "1939-09"
    "the Cold War"                                  → "1947"
    "the fall of the Berlin Wall"                   → "1989-11"
- Vague / relative ("last year", "recently",
  "soon", "in the future", "today")               → null
- Pure adverbials with no semantic content        → null

Return pure JSON: {{"resolved": [{{"id": <int>, "iso": "<ISO string or null>"}}, ...]}}
Each id MUST appear in your output exactly once. Use null for unresolvable.

Phrases:
{lines}
"""
    data = call_json(prompt, model=model, role="resolve_dates", event_callback=event_callback)
    out: dict[int, str] = {}
    for rec in (data.get("resolved") or []):
        try: rid = int(rec.get("id"))
        except (TypeError, ValueError): continue
        iso = rec.get("iso")
        if iso is None: continue
        s = str(iso).strip()
        # ISO 4-10 chars: "1949" / "2022-02" / "2001-09-11". Reject anything else.
        if 4 <= len(s) <= 10 and s[:4].isdigit(): out[rid] = s
    return out


# Annotate every edge carrying a time_phrase with resolved_date (in place).
# Already-annotated edges are skipped so the pass is re-runnable. Returns the
# same clause list. Recurses into nested edge targets.
def resolve_dates(clauses, *, model: str = "7B", event_callback=None) -> list[dict]:
    clauses = list(clauses)
    pending = [e for e in _iter_edges(clauses)
               if e.get("time_phrase") and "resolved_date" not in e]
    if not pending: return clauses
    resolved_count = skipped = 0
    for start in range(0, len(pending), _PENDING_BATCH):
        batch = pending[start : start + _PENDING_BATCH]
        items = [{"id": j, "phrase": e["time_phrase"]} for j, e in enumerate(batch)]
        resolved = _llm_resolve_dates(items, model=model, event_callback=event_callback)
        for j, e in enumerate(batch):
            iso = resolved.get(j)
            if iso:
                e["resolved_date"] = iso; resolved_count += 1
            else:
                e["resolved_date"] = ""; skipped += 1
    if event_callback:
        try: event_callback("resolve_dates", {
            "resolved": resolved_count, "skipped": skipped, "total": len(pending), "model": model})
        except Exception: pass
    return clauses
