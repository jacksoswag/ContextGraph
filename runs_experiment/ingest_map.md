# P0 — Ingest Pipeline Map (text → graph store)

Branch base: `adaptive-domain` (working tree). Verified live: `embed()` → 384-d
unit vector (1536 B); `extract_clauses()` → typed clause-edges with tense/time/pending.
Store artifact inspected: `.di-ui/graph.sleep.merged.sqlite` (392 396 nodes / 773 665
edges / 100 994 sleep_log rows).

## 1. Intended end-to-end flow

```
source text
  │
  ▼  resolve_basic_pronouns (coreferee)            src/ingest/extraction.py
  ▼  spaCy SVO + passive-rewrite + prep/poss/amod   → compositional clause-edge dicts
  │     {type:edge, rel, source:{node|edge}, target:{node|edge}, modifiers:[…],
  │      tense?, time_phrase?, _pending_completion?}
  ▼  (1B LLM fallback for zero-yield sentences)
  │
  ├─ editor.ingest_text / file_ingester.ingest_file  (interactive / file entry; pure producers)
  │
  ▼  deep_ingest.ingest_article                      src/ingest/deep_ingest.py
  │     • _extract_triples: re-runs extract_clauses, FLATTENS via _clause_to_triple
  │       (drops every hyperedge — source/target that is itself an edge — :109-123)
  │     • 7B LLM stitch → cross-clause IngestEdge(relation, src_cid, tgt_cid, conf)
  │     • cache: research_cache.sqlite deep_ingest table
  │     • complete_pending_edges (7B) / resolve_dates (7B): producer transforms
  │
  ▼  ┌─────────────────────────────────────────────┐
     │  ★ NODE/EDGE WRITER  — DOES NOT EXIST (GAP 1) │
     └─────────────────────────────────────────────┘
  ▼  ┌─────────────────────────────────────────────┐
     │  ★ MERGE / DEDUP     — DOES NOT EXIST (GAP 2) │   (sleep_log was its output)
     └─────────────────────────────────────────────┘
  ▼
  store: nodes / edges / nodes_fts / nodes_vec        read-only via src/graph/store.py
  │
  ▼  CONSUMER: field-gather (materialize → gather)    src/field/gather.py
        store.neighbors / store.anchor / store.text / store.find
```

## 2. Component inventory

| File | Role | Status |
|---|---|---|
| `src/ingest/extraction.py` | spaCy SVO → compositional hyperedge dicts; coref pre-pass; 1B fallback | **WORKS** (producer) |
| `src/ingest/editor.py` | `ingest_text` — interactive entry, bare-question filter | **WORKS** (producer, no store) |
| `src/ingest/file_ingester.py` / `file_reformatter.py` | `.txt/.md` → segments → clauses | **WORKS** (producer, no store) |
| `src/ingest/deep_ingest.py` | 7B cross-clause stitch + pending/date resolution; **flattens hyperedges** | **WORKS** but **PRODUCER-ONLY** (returns `IngestResult`, never writes store) |
| `src/ingest/scrape_worker.py` | Wikipedia (`w/api.php`) + Serper Google/Scholar fetch + HTML extract + cache | **WORKS** (Serper intact) |
| `src/ingest/web_search.py` | search → `(domain,url,body)` stream over scrape_worker | **WORKS** |
| `src/ingest/labels.py` | label normalization / cleanup primitives | **WORKS** |
| `src/embed.py` | all-MiniLM-L6-v2, 384-d, `pack/unpack/embed/embed_batch` | **WORKS** (the anchor embedder) |
| `src/graph/store.py` `GraphStore` | **read-only** (`mode=ro`) adapter: `neighbors/anchor/text/find` | **WORKS** (read side only) |
| `src/field/sleep.py` | **StructureRecon** — learns per-edge weight multipliers. **NOT a node merge.** | works, but ≠ dedup |
| `tests/ingest/*`, `tests/test_extraction.py`, `tests/test_embed.py` | producer-level tests | present, **UNTRACKED** |
| `tests/eval/reporting.py`, `tests/eval/generate_stress_cases.py` | import deleted retrieval modules (`eval.assertions`, `testing_manager`) | **STALE/DEAD** — rebuild |

## 3. Concrete gaps to close

**GAP 1 — No node/edge writer.** Nothing in tracked code performs
`INSERT INTO nodes/edges` with `info_vector`, `nodes_fts`, `nodes_vec`. `git log -S`
for `victim_id`, `nodes_vec`, `CREATE TABLE … nodes` returns nothing across all
history. `GraphStore` opens `mode=ro`. The `.di-ui/graph.sleep.merged.sqlite`
bootstrap (ConceptNet 5.7 — relations like `at location`, score `0.5`, source
`data/raw/conceptnet-assertions-5.7.0.csv.gz`) was built by an **untracked/external
tool that is no longer in the repo.** → Build `src/graph/writer.py`: open RW, ensure
schema, embed node text via `embed.embed_batch`, upsert deterministic-id nodes/edges,
keep `nodes_fts`+`nodes_vec` in sync. Idempotent: same normalized text → same `n_id`.

**GAP 2 — No merge/dedup.** The `sleep_log` schema (`victim_id, canonical_id,
cosine, lexical, density, threshold`) and the pipe-delimited node `text`
("fox|fox") prove the bootstrap was deduped by **embedding-cosine + lexical +
density-adjusted threshold**, picking a canonical and folding victims' edges +
synonyms. That code is **gone**; `src/field/sleep.py` today is unrelated
(edge-weight learning). → Rebuild a node-merge pass reusing the recoverable
algorithm (FTS/vec NN candidate gen → cosine+lexical score → density threshold →
fold). Reuse `nodes_vec` for NN. (Idempotency vs semantic-dedup split: §5.)

**GAP 3 — deep_ingest silently flattens hyperedges** (`_clause_to_triple`,
`deep_ingest.py:109-123` drops any edge whose source/target is itself an edge;
`extract_clauses` modifier lists are also dropped). The field cannot represent
hyperedges, so this must be a **declared** decision — flatten vs reify — not a
silent drop. → **HUMAN GATE (hyperedges).**

**GAP 4 — Scholarly source.** Scholarly retrieval today is **Serper Google
Scholar** (`scrape_worker.py:224`). P2 asks for the **Semantic Scholar API** for
scholarly (Scholar HTML scraping is forbidden; Serper Scholar is an API proxy, so
it is not strictly disallowed but is not the requested source). → Add a Semantic
Scholar source (abstracts) behind the existing provider switch; cache + rate-limit.
Wikipedia search already uses `w/api.php`, but **body text is HTML-scraped** via
BeautifulSoup — P2 prefers the REST content/summary API. → Add a REST body path.

**GAP 5 — Benchmark harness.** `tests/eval/` is stale (imports deleted modules).
No faithfulness/coverage/merge/stress measurement exists. → Build fresh
`tests/eval/` + scripts + gold sets (P4).

## 4. Consumer contract (must stay compatible)

The field reads **only** through `GraphStore`:
- `nodes(id TEXT 'n_…', text, pos, info_vector BLOB = 384×f32 unit vector)` — `anchor()`/`text()`.
- `edges(id 'e_…', source_id, rel_type, target_id, score REAL)` — **dyadic n_→n_ only** (`neighbors()`).
- `nodes_fts` (FTS5, porter) — `find()` seed entry points; falls back to `LIKE`.
- `nodes_vec` (sqlite-vec vec0) — present in schema; `GraphStore.find` does *not*
  query it today, but the contract says keep it populated (merge NN uses it).
- `info_vector` MUST be `embed.pack`-shaped (384 f32). Fallback anchor exists
  (`gather.py:15`) but un-embedded nodes settle on a degenerate centroid → embed everything.

## 5. Design notes for P1 (pre-decisions, open to override)

- **Node id = `n_` + md5(`normalize_label(text)`)** (legacy ids are *not* a plain
  md5 — they were assigned pre-merge — so the writer will not reproduce legacy ids;
  that is fine, see below). **Edge id = `e_` + md5(`src_id|rel|tgt_id`)**. This makes
  re-ingesting identical text a no-op (idempotent), bumping `count`/`updated_at` only.
- **Two-layer identity:** (a) *idempotency* = exact normalized text → same id, handled
  by the writer; (b) *semantic dedup* = same concept under different surfaces / different
  ids (incl. legacy bootstrap ids) → handled by the merge pass (GAP 2), exactly as the
  original sleep merge did. This mirrors the existing architecture; no id-rewrite of the
  bootstrap is required.
- **Threading:** `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` for any embed/field run.
- **Branching:** the entire `src/ingest/*`, `src/embed.py`, and ingest tests are
  *currently untracked* (the "Field is canonical" pivot is mid-flight in the working
  tree). A literal `git checkout main` would orphan them. Plan: work on a new branch
  cut from the current working tree, first committing the pre-existing untracked ingest
  baseline separately so my phase commits are clean diffs. No push.

## 6. Verification commands run (P0)

```
sqlite introspection of .di-ui/*.sqlite*           → schema + row counts (§intro)
git log --all -S victim_id / nodes_vec / "CREATE … nodes"  → (empty) writer absent
OMP_NUM_THREADS=1 PYTHONPATH=src DI_LLM_EXTRACTION=0 python - (embed + extract_clauses smoke) → ok
```
