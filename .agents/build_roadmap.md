> # ⛔ SUPERSEDED — DO NOT FOLLOW. Kept for history only.
> This roadmap orchestrates the **itinerancy model falsified at gate G3** (Phases/gates G1–G6,
> SequenceRecon, the `(β,ρ_R)` phase diagram). See `.agents/G3_escalation.md` for why it was
> abandoned. **The active build & test plan is now `.agents/field_gather_build_plan.md`**, against
> the canonical `.agents/field_gather_spec.md`. Do not execute the phases below.

# GRAPH–FIELD SYSTEM — BUILD & BENCHMARK ROADMAP  [SUPERSEDED]
## Orchestration plan: prompts, agents, effort, concurrency, assessment gates

> Source of truth: `architecture_spec.md` + `invariant_set_phase_diagram.md` +
> `implementation_spec.md`. This roadmap is *execution orchestration only* — it adds no
> architecture. Every prompt below must obey `AGENTS.md` (consult before non-trivial
> deviation) and `formatting.md` (hash comments, no docstrings, compact).
>
> Profile assumption: Claude Max → Claude-heavy, parallel sessions/worktrees are cheap. Use
> Sonnet for implementation/tests (fast, parallelizable), reserve Opus for the two math-hard
> pieces and every assessment gate.

---

## 0. ORCHESTRATION MODEL

**Agent roles**
- **Sonnet** — all implementation, unit/integration tests, mechanical refactors, cleanup,
  plotting, fixtures. Cheap, run many in parallel.
- **Opus** — only: (a) the two design-sensitive math pieces — the regime classifier
  (§Phase 2) and `SequenceRecon` (§Phase 4); (b) **every assessment gate** (review + decide
  go/revise); (c) hard debugging Sonnet bounces off twice.

**Effort levels** (set per prompt below)
- `S-std` Sonnet standard — boilerplate, fixtures, plumbing, cleanup.
- `S-think` Sonnet extended thinking — non-trivial logic with a clear spec.
- `O-deep` Opus deep reasoning — math correctness, gate assessments, ambiguity resolution.

**Tooling per session**
- Use Claude Code build subagents inside each session: `arc:build:implementer` to build,
  `arc:build:unit-test-writer`/`integration-test-writer` for tests, `arc:build:debugger`
  for failures, `arc:build:test-runner` to run, `arc:review:senior-engineer` /
  `daniel-product-engineer` at gates.
- Concurrency = **git worktrees**, one per independent track (`git worktree add`), so
  parallel Sonnet sessions don't collide. Merge at the gate.

**Assessment gates (STOP and assess)** — after every Phase, before the next:
1. run full `pytest -q` + the phase's benchmark;
2. **Opus gate prompt** (provided per phase) reviews invariants + acceptance + numbers;
3. **human checkpoint**: read the gate report, decide go / revise / escalate.
The three high-risk gates that most need a human eye: **G1 (numerics), G3 (does itinerancy
emerge?), G4 (does SequenceRecon actually work — the one definitional risk)**.

---

## 1. DEPENDENCY DAG & CONCURRENCY TRACKS

```
config ─┬─ episode ── coupling ── energy ── dynamics ─┐
        │                                             ├─ harness/experiments
        └───────────── observables (vs synthetic) ────┤
tracking/logging  (independent) ──────────────────────┤
toy-graph fixtures (independent) ─────────────────────┘
sleep  ← needs energy + dynamics + observables
```

**Parallel tracks (run as concurrent worktrees):**
- **Track A** (core chain): config→episode→coupling→energy→dynamics. Mostly sequential.
- **Track B** (observables): build against a synthetic trajectory stub; parallel with late A.
- **Track C** (instrumentation): tracking/logging + toy fixtures + CI runner. Parallel from t0.

Phases 1–2 fan out across A/B/C; Phases 3–6 are sequential (each needs the prior gate).

---

## PHASE 0 + 1 — CLEANUP & SIMULATOR CORE  (the first prompt the user asked for)

### PROMPT 0.A — clear redundant systems  [Sonnet, `S-std`, Track C, concurrent with 1.x]
```
Read .agents/architecture_spec.md (§12 migration note + §9 invariants) and
.agents/implementation_spec.md (§1 module layout). You are clearing the v5.3 field package
down to what the new architecture keeps, with NO behavior added.

Delete these (superseded — confirm each is not imported by anything you keep):
  src/field/memory.py, objective.py, expansion.py, routing.py, operators.py,
  interpreter.py, kernel.py, learn.py, project.py, spaces.py (it becomes episode.py).
Also remove their tests under tests/field/ that target deleted modules, and any dead
imports left behind. Do NOT touch src/graph/ (S storage stays).

Method: for each file, grep for imports across src/ and tests/ first; if something you must
keep imports it, STOP and list the conflict instead of deleting. Produce a short report:
what was deleted, what referenced it, what you stubbed/kept. Run `venv/bin/python -m pytest -q`
and report the new failure surface (expected: tests for deleted modules fail/are removed).
Adhere to formatting.md. Do not implement new modules in this prompt.
```

### PROMPT 1.A — simulator core (M1)  [Sonnet, `S-think`, Track A]
```
Implement Milestone M1 from .agents/implementation_spec.md (§3.1, §3.2, §3.4, §5-M1),
strictly following .agents/architecture_spec.md (§3,4,5,14). Files: config.py, episode.py,
coupling.py, energy.py, dynamics.py.

TDD: for each module write the focused test first (tests/field/), then the code. Build in
this order (dependency): config → episode → coupling → energy → dynamics.

Hard requirements (assert in code):
 - energy.grad_E exactly matches spec §4 (couple = −C_sym X, inhib = softmax_i(β·½‖x‖²)x_i,
   barrier = μ‖x_i‖²x_i). R(X)=C_anti·X.
 - dynamics.step uses generator G = −∇E + R; barrier handled by clamp to R_max (v1);
   trapping guard asserts max_i‖x_i‖ ≤ R_max each step.
 - coupling: C masked to E(S); C_sym degree-normalized; ‖C_anti‖ ≤ ρ_R‖C_sym‖; store
   λmax(C_sym), R_max=sqrt(λmax/μ), L, and the η bound η≤c/L.
Acceptance (must pass before you report done): implementation_spec.md §4 tests 1,2,3,8 on
single-clique and two-clique fixtures; ρ_R=0 ⇒ E(t) monotone non-increasing within η tol.
Use arc:build:unit-test-writer for tests, arc:build:debugger on any failure (prefer
event/root-cause fixes over loosening tolerances). Report exact pytest output.
```

### PROMPT 1.C — instrumentation foundation  [Sonnet, `S-think`, Track C, concurrent]
```
Build the run-tracking/logging foundation described in .agents/build_roadmap.md §3. New file
src/field/tracking.py + a runs/ artifact store. No dependency on dynamics — design against
the metrics schema in §3 and a synthetic trajectory.

Deliver: RunLogger that records {git_sha, config dict+hash, seed, per-step metrics
(E, ‖ΔX‖, support_size, mesh_count, trapping_violations, nan_flag), summary metrics
(steps_to_stabilize, regime_label placeholder, runtime), artifacts (trajectory .npz, plots)}
to runs/<timestamp>_<confighash>/ as JSONL + a sqlite index runs/index.db. Add a tiny
`report.py` that renders E(t)/occupancy/mesh-timeline plots from a run dir. Unit-test the
logger writes/reads round-trip and the config hash is stable. formatting.md applies.
```

> Run 0.A + 1.A + 1.C in **three parallel worktrees**. 0.A and 1.C don’t touch the same files
> as 1.A.

### GATE G1 (numerics) — STOP & ASSESS  [Opus, `O-deep`]
```
Assessment gate G1. Read architecture_spec.md §9 (invariants) and §14.8 (numerics), the
implementation_spec.md M1 acceptance, and the current src/field/{coupling,energy,dynamics}.py
+ test output. Verify, with evidence:
 1. dynamics is exactly −∇E + R; no extra terms; R=C_anti X with C_anti antisymmetric.
 2. trapping guard holds on all fixtures and any ρ_R≤ρ_max (bounded, no NaN/Inf).
 3. ρ_R=0 reproduces Option A (E monotone non-increasing) — the Lyapunov sanity check.
 4. coupling support = E(S); degree-normalization present; ‖C_anti‖≤ρ_R‖C_sym‖ enforced.
 5. η bound derived from L and actually used.
Output: PASS/REVISE per item with the failing evidence, and a go/no-go for Phase 2. Flag any
silent deviation from the spec. Do not fix — diagnose.
```

---

## PHASE 2 — OBSERVABLES & REGIME CLASSIFIER (M2)

### PROMPT 2.B — observables  [Sonnet, `S-think`, Track B, can start during Phase 1 vs synthetic data]
```
Implement observables.py per implementation_spec.md §3.3 and architecture_spec.md §14.2-14.5:
support(τ, window), online mesh clustering (Jaccard<d_mesh), itinerary (dedup), occupancy,
and the stabilization test (TV(occupancy windows)<ε_occ held H_hold; point-test ‖ΔX‖<ε_x).
TDD with synthetic trajectories (hand-built support sequences) so this needs no real dynamics.
Acceptance: implementation_spec.md §4 tests 5,6. Wire RunLogger (tracking.py) to record
support_size, mesh_count, itinerary, occupancy, stabilized_at per run. formatting.md applies.
```

### PROMPT 2.O — regime classifier  [Opus, `O-deep`]  ← math-hard, give to Opus
```
Implement the regime classifier in observables.py per architecture_spec.md §14.6 and the
analysis in invariant_set_phase_diagram.md (§6 taxonomy, §8). This is the falsifiability
core — be rigorous.
 - dominant-frequency estimation from o(t)=E(X_t) (FFT + peak-above-noise; define the noise
   floor and the incommensurability test for torus vs cycle explicitly).
 - finite-time largest Lyapunov exponent via a paired renormalized run (define δ0, renorm
   interval, averaging window).
 - classify {point, cycle, torus, chaos} per the §14.6 rules; return Regime with freqs,
   lyap_max, itinerary, occupancy.
Acceptance: implementation_spec.md §4 test 4 (single clique→point); a hand-constructed
2-frequency synthetic signal→torus; a noisy single-frequency signal→cycle (not chaos). Add
tests asserting the torus/cycle/chaos boundaries on synthetic signals. Document every
threshold and why. formatting.md applies.
```

> 2.B and 2.O can run concurrently (different functions in observables.py; coordinate the
> file or split into observables.py + regime.py to avoid merge conflict — prefer the split).

### GATE G2 — STOP & ASSESS  [Opus, `O-deep`]
```
Gate G2. Verify observables are reproducible (test: same seed ⇒ identical
itinerary/occupancy/regime) and the classifier is falsifiable (distinguishes the four
synthetic regimes with documented thresholds). Confirm RunLogger captures every §3 metric.
Output PASS/REVISE + go/no-go for Phase 3.
```

---

## PHASE 3 — ITINERANCY EXPERIMENTS (M3)

### PROMPT 3.A — toy graphs + rollout experiments  [Sonnet, `S-think`]
```
Build harness.py experiments per implementation_spec.md §4. Toy graphs (tests/field/fixtures
or a builder): single clique, two cliques+bridge, ring-of-cliques, two incommensurate rings.
For each, run rollout (dynamics) at the spec-default (β, ρ_R) and at ρ_R=0, log full runs via
RunLogger, classify regime, dump trajectory + plots.
Acceptance (M3): two-cliques+bridge at moderate ρ_R yields a measurable, reproducible
itinerary across the bridge (occupancy over ≥2 meshes; stabilization triggers). Produce a
short experiment report (markdown) with the regime per graph and the plots. Do NOT tune to
force a result — report what emerges. If no itinerancy emerges at defaults, say so.
```

### GATE G3 (does itinerancy emerge?) — STOP & ASSESS  [Opus + human]  ← high-risk
```
Gate G3. The central empirical question: does the bridge graph produce genuine itinerancy
(ordered mesh visiting), or only a point/blob? Review the run logs, occupancy timelines, and
regime labels. Decide: (a) itinerancy emerges → proceed; (b) only points → diagnose whether
it's a parameter issue (sweep a small (β,ρ_R) neighborhood) or a structural one (escalate to
spec — possibly the dynamics need the semi-implicit barrier, or ρ_R range is wrong). Output a
diagnosis + go/revise/escalate. HUMAN: read this before continuing — it gates the whole thesis.
```

---

## PHASE 4 — SLEEP / LEARNING (M4)

### PROMPT 4.O — SequenceRecon (dual target)  [Opus, `O-deep`]  ← the definitional risk
```
Implement sleep.py per architecture_spec.md §7, §14.7 and implementation_spec.md §3.5.
StructureRecon (shapes C_sym): mesh-support vs held-out sub-details — straightforward.
SequenceRecon (shapes C_anti) — implement BOTH targets and make them swappable, because the
centroid surrogate is unvalidated (spec §13 risk):
  (T1) centroid-displacement: −Σ w_ij cos(C_anti x*_i, x*_j − x*_i)   [spec §14.7 default]
  (T2) Jacobian-unstable-eigenvector: align C_anti's action with the unstable eigenvectors of
       J=−H+A at the saddle between mesh i and j  (compute H=∇²E at the saddle estimate).
Enforce constraints: C masked to E(S), ‖C_anti‖≤ρ_R‖C_sym‖, trust region ‖ΔC‖≤δ, EMA τ_C.
Validation metric (non-diff): KL(T* ‖ T̂) on mesh-transition matrices.
Acceptance: implementation_spec.md §4 test 7 (gradient lowers loss when aligned); on the
bridge graph, training reduces KL(T*‖T̂) for at least one of {T1,T2}; constraints provably
hold. Report which target works better — this feeds the spec §14.7 decision.
```

### PROMPT 4.A — Sleep loop + StructureRecon wiring  [Sonnet, `S-think`, after 4.O lands the losses]
```
Wire the offline Sleep loop: episode batch → rollout → observe meshes → compute
StructureRecon+SequenceRecon → Adam step on C → project constraints → EMA. Log loss curves,
StructureRecon recall, KL(T*‖T̂), and constraint-satisfaction to RunLogger each Sleep step.
Acceptance (M4): StructureRecon improves mesh-vs-target recall on a toy held-out set; full
constraint set holds every step; runs logged. formatting.md applies.
```

### GATE G4 (does SequenceRecon work?) — STOP & ASSESS  [Opus + human]  ← highest risk
```
Gate G4. Decide the spec §13 open risk. From the 4.O/4.A runs: does either SequenceRecon
target (T1 centroid vs T2 Jacobian) actually drive the empirical transition matrix T̂ toward
T*? Quantify (KL before/after, itinerary match). If yes → record which target and PROMOTE it
into architecture_spec.md §14.7 (replace the surrogate-only definition). If neither works →
this is a genuine spec revision: escalate with a concrete diagnosis (is it the mesh-rep
estimate? the saddle estimate? the constraint too tight?). HUMAN: this is the make-or-break
gate for "reasoning-as-itinerary." Read fully.
```

---

## PHASE 5 — PHASE DIAGRAM + BENCHMARK SUITE (M5)

### PROMPT 5.A — (β,ρ_R) sweep + benchmark harness  [Sonnet, `S-think`]
```
Implement the benchmark suite per build_roadmap.md §3 (Benchmarks). Sweep a grid over
(β, ρ_R); for each cell run N seeds, classify regime, record to runs/ + index.db; render the
empirical phase diagram (regime per cell) and the sharp-transition map. Add the four standing
benchmarks: (B1) regime-prediction accuracy vs invariant_set_phase_diagram.md predictions on
toy graphs; (B2) reproducibility (seed-stability of itinerary/occupancy); (B3) throughput
(steps/sec, scaling vs N); (B4) numerical-safety (zero trapping/NaN violations across the
grid). Emit a benchmark_report.md generator. formatting.md applies.
```

### GATE G5 — STOP & ASSESS  [Opus + human]
```
Gate G5. Compare the empirical phase diagram to invariant_set_phase_diagram.md: are
transitions sharp (Hopf-like)? Are (β,ρ_R) orthogonal in interiors and coupled at the
boundary (§Result 4)? Does B1 accuracy clear a stated bar? Document deviations between theory
and experiment — deviations are findings, not failures. Decide go/no-go for e2e (Phase 6).
```

---

## PHASE 6 — E2E GRAPH TRAINING & FORMAL BENCHMARK

### PROMPT 6.A — real-graph episodes + e2e Sleep training  [Sonnet, `S-think`]
```
Wire episodes from the real graph S (src/graph/ store): seed active set N from a query
(reuse existing seed selection ONLY to build X_0 — this is initialization, NOT runtime
retrieval, per spec §6/§0), run the field, read out support (point) or itinerary+occupancy
(recurrent). Run the e2e Sleep training loop over the corpus: alternate inference episodes
and Sleep C-updates; log everything. formatting.md applies.
```

### PROMPT 6.O — formal e2e benchmark + report  [Opus, `O-deep`]
```
Define and run the formal e2e benchmark. Choose the task→readout mapping explicitly (e.g.,
does the settled support / itinerary reconstruct held-out graph structure or answer the
stress corpus?), justify it against architecture_spec.md (readout = invariant set, §8). Run
with fixed seeds, versioned config, git sha; produce benchmark_report.md: per-task metrics,
regime distribution, training curves, reproducibility, throughput, and a regression table vs
the prior commit. State clearly what is validated and what is not. This is the formal record.
```

### GATE G6 (final) — STOP & ASSESS  [Opus + human]
```
Final gate. Is the system a coherent, reproducible, benchmarked artifact? Does e2e behavior
match the design claims (reasoning expressed as invariant-set geometry), or does it reveal a
spec-level gap? Produce the honest verdict: validated / partially / not, with the evidence.
Recommend the next research cut.
```

---

## 3. BENCHMARKING & LOGGING SUITE (what it must record)

**Per-run metrics (JSONL, one line/step + a summary):**
`git_sha, config_hash, seed; per-step: E, ‖ΔX‖, support_size, mesh_count, max_node_norm,
trapping_violation(bool), nan(bool); summary: steps_to_stabilize, regime, freqs, lyap_max,
itinerary, occupancy, runtime_s.`

**Per-Sleep-step:** `loss_total, structure_recon, sequence_recon, KL(T*‖T̂),
constraint_ok(support⊆E(S), ‖C_anti‖≤ρ_R‖C_sym‖, ‖ΔC‖≤δ), grad_norm.`

**Store:** `runs/<ts>_<confighash>/{run.jsonl, summary.json, trajectory.npz, plots/*.png}` +
`runs/index.db` (sqlite: one row/run, queryable by config/sha/regime).

**Standing benchmarks (regression-tracked across commits):**
- B1 regime-prediction accuracy (vs theory, toy graphs).
- B2 reproducibility (seed-stable itinerary/occupancy).
- B3 throughput + scaling (steps/sec vs N).
- B4 numerical safety (0 trapping/NaN across the grid).
- B5 (Phase 6) e2e task metric + regression vs prior commit.

**Stack:** keep it lightweight/local first — JSONL + sqlite + matplotlib, deterministic
seeds, config hashing. Optional W&B adapter behind a flag; do not hard-depend on it.

---

## 4. CONCURRENCY CHEAT-SHEET

| can run together | why |
|---|---|
| 0.A (cleanup) ‖ 1.A (core) ‖ 1.C (tracking) | disjoint files, separate worktrees |
| 2.B (observables) ‖ 2.O (regime) | split into observables.py / regime.py |
| 2.B can start *during* Phase 1 | builds against synthetic trajectories |
| toy fixtures (any time) ‖ everything | pure data |
| Phases 3,4,5,6 | **sequential** — each needs the prior gate |

Rule of thumb: **fan out within a phase, serialize across gates.** Never start a phase whose
gate-input doesn’t exist yet.

---

## 5. EFFORT / MODEL SUMMARY

| work | model | effort | why |
|---|---|---|---|
| cleanup, plumbing, fixtures, plotting, Sleep loop wiring | Sonnet | S-std / S-think | well-specified, parallelizable |
| simulator core, observables, experiments, benchmark harness | Sonnet | S-think | clear spec, needs care |
| regime classifier, SequenceRecon (both targets) | **Opus** | O-deep | math-hard, falsifiability/risk |
| every assessment gate (G1–G6) | **Opus** | O-deep | judgement + invariant audit |
| hard debugging after 2 Sonnet bounces | Opus | O-deep | root-cause |

Claude-heavy usage: implementation on Sonnet across parallel worktrees; Opus only at the two
risk points + the six gates. Humans read G1, G3, G4 closely.

---

END ROADMAP
