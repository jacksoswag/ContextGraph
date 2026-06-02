# FIELD-GATHER — IMPLEMENTATION & TESTING PLAN
## For an autonomous Opus session. Source of truth: `.agents/field_gather_spec.md`.

> Read `field_gather_spec.md` (canonical), `AGENTS.md` (operating standard), `formatting.md`
> (hash comments, no docstrings, compact, ≤2 blank lines, one-line headers). Treat every function
> as a liability — reuse/merge before adding. Background context: `.agents/G3_escalation.md`
> (why itinerancy was retired), `runs_experiment/` (the G3 evidence).

## OPERATING MODE

- **TDD always:** write the focused test first, then the code. Run `venv/bin/python -m pytest -q`
  (pythonpath=src via `pytest.ini`). Pin threads for field runs: `OMP_NUM_THREADS=1` +
  `torch.set_num_threads(1)` (avoids the oversubscription slowdown seen in G3).
- **Commit atomically per phase** (branch off `main`; do not push unless asked).
- **Build subagents** are available (`arc:build:implementer`, `unit-test-writer`,
  `integration-test-writer`, `debugger`, `test-runner`; `arc:review:senior-engineer` at gates).
- **CHECKPOINT protocol (📍):** at each marked checkpoint, STOP. Post a short report with
  *evidence* (numbers/tables, not prose) and ask the listed question(s) via `AskUserQuestion`.
  Do not proceed past a 📍 until answered. Between checkpoints, work independently.
- **Gates (G#):** after each phase, run the full field suite + the phase acceptance; self-review
  against the spec invariants (§7) before moving on.

## DEPENDENCY ORDER

```
P0 prune ─ P1 gather core ─ P2 readout ─ P3 precision[📍 big] ─┐
                                                               ├─ P5 recursion+e2e ─ P6 Sleep
P4 LLM seams (parallel with P1–P3; mock-testable) ────────────┘
```
P0→P1→P2→P3 sequential. P4 (LLM seams, mock-tested) can run in a parallel worktree. P5 needs P3+P4.
P6 (Sleep) is optional/last.

---

## PHASE 0 — PRUNE TO THE GATHER CORE

**Goal:** remove the retired itinerancy machinery; keep the convergent-gather substrate. No new
behavior.

**Remove** (confirm no kept code imports them first): from `coupling.py` the `anti` construction +
`ρ_R`/`ρ_max` rescale; from `energy.py` `rotation`; from `dynamics.py` the `R(X)` term in `step`;
from `observables.py` `regime_label`, `classify`, `finite_time_lyapunov`, `dominant_freqs`,
`_commensurate`, `_incommensurate`, `_distinct_fundamentals`, `motion_signal`, `Regime`; from
`config.py` `rho_R`, `rho_max`, `eps_osc`, regime/`T_class`/`T_burn` knobs that only served
classification. **Keep:** `support`, `cluster_mesh`/Jaccard, `MeshCatalog`, `StabilizationMonitor`,
`tracking.py`, `episode.py`, the couple/inhib/barrier energy, the sustained-settle `rollout`.

**Tests:** delete `test_m3.py` and the regime/Lyapunov/`_distinct_fundamentals`/`motion_signal`
tests in `test_observables.py`; **keep** the dynamics invariants (`test_dynamics.py`: trapping,
finiteness, monotone-`E`, determinism, sustained-settle) and support/mesh/tracking tests.

**Acceptance / Gate G0:** field package imports; reduced suite green; `grep -r "C_anti\|rho_R\|
regime\|lyapunov\|itiner" src/field` returns nothing. Archive the deleted regime code in the commit
message (so it's recoverable if itinerancy is revisited).

📍 **CHECKPOINT 0** — before deleting, post the kill-list and ask: *"Delete the regime/rotation code
outright, or move it to `src/field/_attic/` for reference?"* (one quick question).

---

## PHASE 1 — SEED-ANCHORED CONVERGENT GATHER (physics core)

**Goal:** `gather(seeds) → settled X*`, anchored to seeds, guaranteed convergent.

**Files:** `energy.py` (+ anchor term, §3.3), `dynamics.py` (gradient-only step + settle, already
there minus rotation), new `gather.py` (active-set materialization §3.1 from `src/graph`, init §3.2,
`gather()` orchestration), `config.py` (`GatherConfig`: `d, beta, mu, sigma_anchor, k_hop, N_max,
tau, eta, eps_x, H_hold, H_max`).

**TDD order:** anchor-energy → active-set materialization → `gather`.

**Tests (assert behavior):**
- **convergence guaranteed** (stress grid over `β,μ,σ,seed` × toy graphs + one real subgraph):
  every gather settles before `H_max`; `E` monotone non-increasing; bounded (trapping); finite.
- **anchor pins seeds:** seed-node relevance stays ≥ a margin above un-seeded baseline; raising `σ`
  tightens the gather toward seeds.
- **locality:** mesh ⊆ `k_hop` reachable set; relevance decays with hop-distance from seeds.
- **determinism:** identical `(seeds, targets, C, cfg, rng)` ⇒ identical `X*`.

**Acceptance / Gate G1:** all above green on toy + a real subgraph pulled from `src/graph` store.

📍 **CHECKPOINT 1** — post a small table (real-subgraph gather: seed set, mesh size, relevance-by-hop
decay, settle step) and ask: *"Does the gather locality/size look right, or should `σ`/`k_hop`
defaults change before I build the readout?"*

---

## PHASE 2 — MESH + PROVENANCE TREE READOUT

**Goal:** turn `X*` into `Mesh{nodes, relevance, tree, seed_roots}` (§3.5).

**Files:** `gather.py` (`build_mesh(X*, C, seeds, edge_index)` → support + relevance +
provenance tree), a small `Mesh` dataclass.

**Tests:** on a hand-built graph with a known structure — tree is **rooted at seeds**, **acyclic**,
**spans the mesh**, every node has a path to a seed; parent = max-`flow` lower-layer neighbor;
deterministic; citation-chain rendering for a node returns seed→…→node.

**Acceptance / Gate G2:** tree-property tests green; render a sample citation chain.

📍 **CHECKPOINT 2** — show the provenance tree for one real gather under **two** algorithms
(flow-weighted vs shortest-path) and ask: *"Which provenance definition do you want as default?"*

---

## PHASE 3 — GATHER PRECISION CALIBRATION (real graph)  ← the decisive validation

**Goal:** answer the spec §11 open question — does the physics gather pull *relevant* context
without over-spreading, and does it beat a plain diffusion baseline?

**Work:** sweep `(β, σ, k_hop, τ)` on real seed sets from `src/graph`; measure mesh size,
relevance-by-hop, and **precision/recall vs a labeled relevant-set** (use a few hand-labeled
query→relevant-nodes cases, or held-out edge neighborhoods as proxy ground truth). Implement a
**personalized-PageRank baseline** on the same seeds/graph and compare gather-mesh vs PPR-top-k.
Emit `runs_experiment/gather_calibration.md` (sweep table + PPR comparison + plots).

**Tests:** sweep harness runs + reproducible; PPR baseline deterministic; calibration report
generated.

**Acceptance / Gate G3:** calibration report exists with real numbers.

📍 **CHECKPOINT 3 (BIG — the make-or-break validation)** — present: (a) gather size/precision vs
`(β,σ)`, (b) gather-vs-PPR comparison, (c) over-spread behavior. Ask: *"(1) What operating point
(β,σ,k_hop,τ) should be the default? (2) Does the physics gather earn its keep over personalized-
PageRank, or should the gather BE PPR-with-learned-weights? (3) Proceed to recursion, or revise the
gather first?"* This decides whether the field formulation survives as the retrieval engine.

---

## PHASE 4 — LLM SEAMS (parallelizable; mock-tested)

**Goal:** the four §4 transforms with JSON contracts, model tiers, caching, deterministic mock.

**Files:** `src/llm.py` (or reuse if present post-pivot) — `call_json(prompt, model, schema)` with
`DI_MODEL_*` tiers + cache; `seams.py` — `interpret`, `decompose`, `sufficient`, `synthesize`, each
validating its JSON schema. `interpret` resolves entities to **real node-ids** via `src/embed.py`
nearest-neighbor in `S` (not free generation).

**Tests:** schema/contract validation per seam with a deterministic mock LLM (extend
`tests/conftest.py`'s `mock_llm`); `interpret` entity-resolution returns real node-ids; malformed
LLM output is rejected/retried; caching round-trips.

**Acceptance / Gate G4:** all seams contract-tested with mock; `interpret` resolves to graph nodes.

📍 **CHECKPOINT 4** — show draft prompts for `decompose` + `sufficient` and the budget defaults
(`MAX_DEPTH`, `MAX_NODES_TOTAL`, `MAX_LLM_CALLS`) and ask: *"Approve these prompts/budgets, or
adjust the decomposition style / stopping policy?"*

---

## PHASE 5 — THE RECURSION (solve tree) + E2E

**Goal:** wire `solve()` (§5): down (decompose→gather, parent-anchored), up (synthesize), with
termination + budget. End-to-end on real queries.

**Files:** `loop.py` — `answer(query)` and `solve(task, seeds, parent_mesh, depth)`; parent-anchor
inheritance (`K_inherit`); budget tracking; trace logging via `RunLogger`.

**Tests:**
- **termination:** recursion always halts (depth/budget/sufficient); no infinite trees.
- **budget enforcement:** total gathered nodes / LLM calls ≤ limits.
- **parent-anchoring effect:** a child gather seeded under a parent has its mesh measurably pulled
  toward the parent's region vs the same child gathered standalone (the "tied-to-parent" property).
- **e2e shape** (mock LLM): real graph + scripted decompose/synthesize ⇒ a bounded tree of gathers
  with a synthesized answer carrying citations (provenance roots).
- **determinism** given fixed LLM outputs.

**Acceptance / Gate G5:** e2e green with mock LLM; one real-LLM run logged for inspection.

📍 **CHECKPOINT 5** — present one full real-query trace (the decomposition tree, each node's gathered
mesh + citations, final answer) and ask: *"Is the answer quality / tree shape right? Tune depth,
breadth, or synthesis before I consider this done?"*

---

## PHASE 6 — SLEEP (StructureRecon on `C_sym`) — optional/deferred

**Goal:** learn `C_sym` so gathers pull more coherent context (§6). EMA, trust region, support=E(S).

**Acceptance / Gate G6:** gather precision (Phase-3 metric) improves on held-out after Sleep;
constraints hold every step; runs logged.

📍 **CHECKPOINT 6** — before starting: *"Do Sleep now (needs labeled query→relevant-context pairs —
how should I source them?), or ship the bootstrap-`C_sym` gather and defer Sleep?"*

---

## TESTING TAXONOMY (what lives where)

- **unit** (`tests/field/`): energy/anchor math, settle, support, provenance-tree properties,
  active-set materialization. Fast, no LLM, no network.
- **stress** (`tests/field/`): convergence + boundedness + determinism across `(β,μ,σ,seed,graph)`
  grids — the safety net (these are the §7 invariants; they must never regress).
- **integration** (`tests/integration/`): gather over a real `src/graph` subgraph; seams with mock
  LLM; PPR baseline comparison.
- **e2e** (`tests/e2e/`): `answer(query)` full tree with mock LLM (deterministic) + one real-LLM
  smoke test (skipped in CI if no key).
- **regression locks:** Phase-3 calibration numbers and Phase-5 budget/termination are
  characterization-locked so later changes surface as diffs.

## INVARIANTS TO ASSERT IN CI (spec §7) — never let these regress

convergence (every gather settles) · boundedness (trapping, no NaN) · `S` immutable at runtime ·
gather locality (mesh ⊆ k-hop) · provenance rooted-at-seeds & spanning · determinism · LLM only at
the four seams.

## REPORT-BACK SUMMARY (for the human)

You will be asked questions at checkpoints **0,1,2,3,4,5,6**. The load-bearing ones are **📍3**
(does the gather work / beat PPR — the validation that decides if the field survives as the engine)
and **📍5** (is the end-to-end answer good). Everything else is independent execution.

END PLAN
