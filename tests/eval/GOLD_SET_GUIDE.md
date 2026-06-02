# Gold-Set Authoring Guide (extraction faithfulness + merge accuracy)

The benchmark is only as trustworthy as this gold set. The single biggest failure
mode is **a gold triple written in a form the extractor never emits** — that scores
a true extraction as a miss and tanks recall for no real reason. So the gold author
MUST mirror the extractor's normalization contract (§2). Read it before writing a
single triple.

Files (kept in repo, under `tests/eval/gold/`):
- `corpus.jsonl` — the fixed corpus manifest (source id + fetch key + seeds). §5
- `triples.jsonl` — ≥100 hand-labeled gold triples. §3
- `merge_pairs.jsonl` — must-merge + must-NOT-merge node pairs. §4

---

## 1. What each gold file measures

- `triples.jsonl` → **precision / recall / F1** of extraction, per relation-type
  bucket, plus the **hallucination rate** (extracted edges with no grounding).
- `merge_pairs.jsonl` → **merge precision/recall**, **false-merge rate**
  (must-NOT-merge that merged), **fragmentation rate** (must-merge that stayed split).
- `corpus.jsonl` → reproducibility: the exact documents + seeds the run uses.

---

## 2. Normalization contract — write gold the way the extractor emits

A predicted edge is `(subject) --relation--> (object)`. The extractor
(`src/ingest/extraction.py`) emits these conventions; **author gold identically**:

1. **Lowercase everything.** `Marie Curie` → `marie curie`.
2. **Subjects/objects are lemmatized noun phrases.** `discovered radium` →
   object `radium`; `electrons` → `electron`; `the United States` → `united states`
   (articles `the/a/an` stripped from NER spans). Compounds keep their modifiers in
   source order: `repair shop`, `nobel prize`.
3. **Relations are verb LEMMAS**, lowercase: `discovered` → `discover`,
   `was founded` → `found`. One word, or one underscore for a 2-token verb only.
4. **Passive → active.** "Rome was founded by Romulus" → `(romulus) --found--> (rome)`.
   Agentless passive ("the bill was signed") keeps the patient as subject with a
   `receive_` prefix: `(bill) --receive_sign--> [event]`.
5. **Prepositions are their own relations**, not fused into the verb. "discovered
   radium in 1898" yields TWO edges: `(marie curie) --discover--> (radium)` AND
   `(marie curie) --in--> (1898)`. Author both if both are factual.
6. **Possessives → `have`.** "Curie's prize" → `(curie) --have--> (prize)`.
7. **Adjective modifiers** attach as `qualifies` hyperedges, not as inline text.
   "radioactive element" → node `element` + modifier `(element)--qualifies-->(radioactive)`.
   For gold, capture these only when the adjective is the *fact* being tested
   (see relation buckets); otherwise skip — they are low-value.
8. **Negation** prefixes the relation: "did not sign" → `not_sign`.
9. **Intransitive / objectless clauses** become `[event]` pending edges. **Do NOT
   author `[event]` targets as gold** — they are not factual triples; the pending
   completion pass is measured separately, not against the gold triple set.

If a fact can be written several ways, pick the form matching rules 1–8. When unsure
what the extractor emits, run `extract_clauses` on the sentence (the benchmark ships
a `--inspect "<sentence>"` helper) and align the gold to its node/rel spelling — but
**only when the extraction is correct**; never copy a wrong extraction into gold.

---

## 3. `triples.jsonl` — ≥100 gold triples

One JSON object per line:

```json
{"doc_id": "wiki:Marie_Curie", "sent": "Marie Curie discovered radium in 1898.",
 "subject": "marie curie", "relation": "discover", "object": "radium",
 "bucket": "action", "match": "exact"}
```

Fields:
- `doc_id` — must exist in `corpus.jsonl`.
- `sent` — the **exact** source sentence the triple is grounded in (verbatim copy).
  This is what the LLM-judge groundedness check uses, and what proves the gold is
  real, not invented.
- `subject` / `relation` / `object` — normalized per §2.
- `bucket` — one of: `action` (transitive verb), `prep` (preposition relation),
  `copula` (is/are identity/category, lemma `be`), `possess` (`have`), `passive`
  (active-rewritten from passive). Buckets drive the per-relation-type breakdown.
- `match` — `exact` (subject, relation, object must all match after normalization)
  or `relaxed` (subject+object match; relation may differ within the same bucket —
  use for near-synonym verbs like `create`/`establish` where the extractor's lemma
  is legitimately unpredictable). Default `exact`; use `relaxed` sparingly and
  justify in a trailing `"note"` field.

### Coverage targets (so the gold is representative, not cherry-picked)

- **≥100 triples** total, spread across **≥10 source documents** and **≥4 domains**
  (e.g. science, history, geography, technology). No single doc > 20 triples.
- Bucket spread: at least 50% `action`, the rest across `prep`/`copula`/`possess`/`passive`.
- Deliberately include **hard cases** (these expose the failure taxonomy):
  coordination/lists ("bees and wasps sting"), passive voice, negation, a nested
  clause ("scientists believe that X causes Y"), a cross-sentence coreference
  ("She won…"), and a numeric/date fact. Tag these with `"hard": "<reason>"`.
- Author triples that are **clearly entailed by `sent`**. If you have to reason
  beyond the sentence, it is not a faithfulness gold triple — drop it.

### Scoring (how the benchmark matches predicted ↔ gold)

- A predicted edge matches a gold triple when, after §2 normalization:
  `match=="exact"` → all three equal; `match=="relaxed"` → subject+object equal and
  predicted relation shares the gold `bucket`.
- Node equality is **string-equal after `normalize_label`** AND, as a fallback,
  cosine(`embed`) ≥ 0.9 (catches `usa`↔`united states` lemma drift). Both reported.
- **Recall** = matched gold / total gold. **Precision** = matched predicted / all
  predicted *that target a gold sentence's doc* (predictions on un-annotated
  sentences are not penalized — gold is sentence-anchored, not document-exhaustive).
- **Hallucination rate** is measured separately and document-wide by the LLM-judge
  (groundedness), not by the gold triples — see §6.

---

## 4. `merge_pairs.jsonl` — coreference / node-merge accuracy

One JSON object per line. Each pair names two surface forms that the writer would
create as distinct nodes; the merge pass either folds them or keeps them apart.

```json
{"a": "usa", "b": "united states", "verdict": "merge", "reason": "same country"}
{"a": "java", "b": "java island", "verdict": "no_merge",
 "reason": "programming language vs Indonesian island — homonym"}
```

- `verdict` — `merge` (must collapse to one node) or `no_merge` (must stay separate).
- **Mandatory must-NOT-merge homonyms** (the false-merge traps): `mercury` planet vs
  `mercury` element vs `mercury` (Roman god); `java` language vs island; `python`
  language vs snake; `amazon` company vs river; `apple` fruit vs company; `mercury`
  ×3 counts as 3 pairs. Author each with a `context` field giving the disambiguating
  sentence so the merge pass has the same evidence the field would.
- **Must-merge** set: morphological/lexical variants (`color`/`colour`,
  `u.s.`/`usa`/`united states`), and coreference targets that resolve to the same
  entity across documents.
- Target **≥20 merge + ≥20 no_merge** pairs, balanced across easy (clear) and hard
  (homonym / near-synonym) cases. Tag hard ones `"hard": true`.
- **False-merge rate** = no_merge pairs that merged / total no_merge.
  **Fragmentation rate** = merge pairs that stayed split / total merge.
  These two trade off against each other — report both, never one alone.

---

## 5. `corpus.jsonl` — the fixed, reproducible corpus

```json
{"doc_id": "wiki:Marie_Curie", "source": "wikipedia", "key": "Marie_Curie",
 "domain": "science", "seeds": ["marie curie", "radium"]}
{"doc_id": "s2:0a1b2c", "source": "semantic_scholar", "key": "<paperId>",
 "domain": "ml", "seeds": ["transformer", "attention"]}
```

- ≥30 Wikipedia articles (`source: wikipedia`, `key` = page title) + ≥10 Semantic
  Scholar abstracts (`source: semantic_scholar`, `key` = S2 paperId), ≥4 domains.
- `seeds` — node texts used to seed gather/merge checks for that doc.
- The benchmark fetches strictly from `key` (cached, gitignored), never live-searches
  during scoring → byte-stable corpus. Pin the embedding model + all RNG seeds.

---

## 6. LLM-as-judge (groundedness) — the larger, looser net

Over a larger sample than the 100 gold triples, each extracted edge is shown to the
judge with its source sentence and scored **entailed / not-entailed / partial**.
- Judge runs on the 7B tier via `llm.call_json` (deterministic, temp 0, cached).
- Hallucination rate = `not-entailed / total judged`. Report partial separately.
- The judge complements, never replaces, the hand gold: gold gives
  precision/recall on a verified core; the judge gives hallucination rate at scale.
- **Do not** let the judge author or edit the hand gold — keep them independent so
  one can audit the other.

---

## 7. Authoring workflow + consistency checklist

1. Freeze `corpus.jsonl` first; fetch + cache all docs; never edit a doc after.
2. Read each chosen sentence; write only triples **entailed by that one sentence**.
3. Normalize per §2 — when in doubt, inspect the extractor's output and align spelling
   on the *correct* cases only.
4. Self-audit before committing:
   - [ ] every `doc_id` exists in `corpus.jsonl`
   - [ ] every `sent` is a verbatim substring of its source doc
   - [ ] relations are lemmas, lowercase, ≤1 underscore, no prepositions-as-verbs misfiled
   - [ ] subjects/objects are lowercase lemmatized NPs, articles stripped
   - [ ] no `[event]` targets in `triples.jsonl`
   - [ ] bucket spread + ≥4 domains + hard cases present
   - [ ] every must-NOT-merge homonym has a disambiguating `context`
   - [ ] `relaxed` matches each carry a justifying `note`
5. A second author (or a different model) re-labels 10% blind; disagreements are
   reconciled and the rule that caused them is added to §2. Record inter-annotator
   agreement in the benchmark report.

Do not tune gold to flatter the extractor. If the extractor's convention is wrong,
fix the extractor or file it as a finding — never bend the gold to hide it.
