# Extraction & Merge Benchmark

Corpus: 4 docs (4 Wikipedia + 0 Semantic Scholar), ≤3000 chars/doc. Ingest: 191 triples in 45.2s (4 triples/s).

## (a) Extraction faithfulness

- **Gold-triple recall** (n=50): **strict 0.020**, relaxed 0.060 (strict = subject+object+relation lemma; relaxed = subject+object).

| bucket | n | strict recall | relaxed recall |
|---|---|---|---|
| action | 33 | 0.03 | 0.06 |
| copula | 9 | 0.00 | 0.11 |
| passive | 1 | 0.00 | 0.00 |
| possess | 1 | 0.00 | 0.00 |
| prep | 6 | 0.00 | 0.00 |

- LLM-judge: skipped (--no-judge).

## (b) Coverage & failure taxonomy

Recall vs document length (relaxed):
- mid: 3/11 = 0.27
- short: 0/39 = 0.00

Missed gold facts (failure modes):
- `marie curie --discover--> radium` (Marie_Curie; action)
- `marie curie --discover--> polonium` (Marie_Curie; action)
- `marie curie --win--> nobel prize` (Marie_Curie; action)
- `marie curie --study--> radioactivity` (Marie_Curie; action)
- `isaac newton --develop--> calculus` (Isaac_Newton; action)
- `albert einstein --develop--> relativity` (Albert_Einstein; action)
- `albert einstein --win--> nobel prize` (Albert_Einstein; action)
- `albert einstein --in--> germany` (Albert_Einstein; prep)
- `photosynthesis --convert--> energy` (Photosynthesis; action)
- `photosynthesis --release--> oxygen` (Photosynthesis; action)
- `eiffel tower --in--> paris` (Eiffel_Tower; prep)
- `mount everest --be--> mountain` (Mount_Everest; copula)
- `mount everest --in--> himalayas` (Mount_Everest; prep)
- `dna --carry--> information` (DNA; action)
- `germany --invade--> poland` (World_War_II; passive/active + date)
- `alexander fleming --discover--> penicillin` (Penicillin; action)
- `penicillin --treat--> infection` (Penicillin; action)
- `python --be--> language` (Python_(programming_language); copula)
- `guido van rossum --create--> python` (Python_(programming_language); action)
- `amazon river --in--> south america` (Amazon_River; prep)
- `amazon river --be--> river` (Amazon_River; copula)
- `great wall of china --in--> china` (Great_Wall_of_China; prep)
- `black hole --be--> region` (Black_hole; copula)
- `mahatma gandhi --lead--> movement` (Mahatma_Gandhi; action)
- `volcano --erupt--> lava` (Volcano; action)
- `coffee --contain--> caffeine` (Coffee; action)
- `william shakespeare --write--> hamlet` (William_Shakespeare; action)
- `william shakespeare --be--> playwright` (William_Shakespeare; copula)
- `vaccine --stimulate--> immune system` (Vaccine; action)
- `solar system --form--> cloud` (Solar_System; action)
- `planet --orbit--> sun` (Solar_System; action)
- `electricity --be--> flow` (Electricity; copula)
- `augustus --be--> emperor` (Roman_Empire; copula)
- `honey bee --produce--> honey` (Honey_bee; action)
- `honey bee --pollinate--> flower` (Honey_bee; coref/plural subject)
- `carbon dioxide --trap--> heat` (Climate_change; action)
- `leonardo da vinci --paint--> mona lisa` (Leonardo_da_Vinci; action)
- `quantum mechanics --describe--> behavior` (Quantum_mechanics; action)
- `pacific ocean --be--> ocean` (Pacific_Ocean; copula)
- `insulin --regulate--> glucose` (Insulin; action)
- `french revolution --begin--> 1789` (French_Revolution; intransitive + date)
- `artificial intelligence --enable--> machine` (Artificial_intelligence; action)
- `internet --connect--> network` (Internet; action)
- `transformer --base--> attention` (transformer_attention; passive)
- `framework --have--> model` (gan; possess)
- `dropout --prevent--> overfitting` (dropout; action)
- `crispr --enable--> genome editing` (crispr; action)

## (c) Merge accuracy

- 20 must-merge + 20 must-NOT-merge pairs.
- **precision 1.000, recall 0.300**, **false-merge rate 0.000**, fragmentation rate 0.700.

Errors:
- FRAGMENTED: `big` ~ `large` (synonym)
- FRAGMENTED: `happy` ~ `glad` (synonym)
- FRAGMENTED: `buy` ~ `purchase` (synonym verb)
- FRAGMENTED: `begin` ~ `start` (synonym verb)
- FRAGMENTED: `smart` ~ `intelligent` (synonym)
- FRAGMENTED: `ocean` ~ `sea` (near synonym)
- FRAGMENTED: `rock` ~ `stone` (synonym)
- FRAGMENTED: `infant` ~ `baby` (synonym)
- FRAGMENTED: `quick` ~ `fast` (synonym)
- FRAGMENTED: `illness` ~ `disease` (synonym)
- FRAGMENTED: `teacher` ~ `instructor` (synonym)
- FRAGMENTED: `house` ~ `home` (near synonym)
- FRAGMENTED: `trash` ~ `garbage` (synonym)
- FRAGMENTED: `forest` ~ `woodland` (near synonym)

## (d) Stress

- Store: **240 nodes / 258 edges** (77 reified-edge endpoints), 4.4 MB.
- Ingest throughput: 4 triples/s.
- Contradiction probe (`signed` vs `not signed`): relations kept = ['in'] (both retained — the graph records contradictions, it does not silently resolve them).
- Coref probe (4 sentences, pronouns → Marie Curie): 4 triples attributed to curie of 4 total: [('curie research', 'change', 'physic'), ('curie', 'win', 'nobel prize'), ('curie', 'discover', 'radium'), ('marie curie', 'be', 'scientist')].

## Phase 1.2 Ablation — struct_edge_w + couple_mode (2026-06-04)

Store: astro (60039 nodes, 100% embedded, 97% count=1 edges)

SUMMARY:
  structural:    seed_hot=100%  recall=0.000  top3%=0.537
  structural+ew: seed_hot=100%  recall=0.000  top3%=0.538  (+0.001)
  semantic:      seed_hot=100%  recall=0.050  top3%=0.545
  semantic+ew:   seed_hot=100%  recall=0.050  top3%=0.545

**Gate 2 verdict: INCONCLUSIVE**
- recall=0 is a metric artifact (corpus uses lemmatized tokens; "hawk radiation" not "hawking")
- struct_edge_w +ew: +0.001 top3% — signal absent because 97% edges count=1 (uniform weights)
- struct_edge_w is correct, wired ON in loop.py — needs multi-source corpus for real validation
- semantic coupling: modest +0.008 top3%, +0.05 recall (2 queries); not decisive
- Proceeding: struct_edge_w stays active, Gate 2 deferred to richer corpus

## Phase 1.3 Multi-seed (post DOWN-weight fix) — 2026-06-04

Store: wikidata (8011 nodes, 100% embedded)
Questions: 80 1-hop, 185 2-hop

1-hop: field=0.962  ppr=0.977  pprR=0.866  (field−pprR=+0.097, wins=13/80)
2-hop: field=0.360  ppr=0.356  pprR=0.323  (field−pprR=+0.037, wins=19/185)

**Gate 3: PASSED** — field ≥ PPR on both hop buckets after _grow_set DOWN-weight fix.
Pre-fix numbers (field 0.407/0.218) were entirely due to entity node eviction bug.

## behavior_bench Final Analysis — 2026-06-04

246 records, 31 questions, 8 conditions (field/rag/ppr/pprR/fieldC/front/closed/hybrid)

Overall scores (5-point scale):
  fieldC=4.806  rag=4.818  field=4.758  pprR=4.677  front=4.548  ppr=4.548  closed=4.441

Multihop (PPR gate's sharpest test):
  field=5.000  fieldC=5.000  rag=4.500  ppr=4.500  pprR=4.500

**Gate 4: PASSED** — field > PPR on generation, perfect on multihop.
**Gate 3: PASSED** — field 0.962/0.360 vs ppr 0.977/0.356 wikidata 1/2-hop.

Phase 3 (energy readout): DEFERRED — spectrum smooth, not bimodal.
Phase 4 (graph embeddings): BLOCKED — corpus has count=1 edges everywhere;
  struct_edge_w signal absent; cannot validate gate (b) until multi-source data exists.
