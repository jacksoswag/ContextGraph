> # ⛔ SUPERSEDED — DO NOT FOLLOW. Kept for history only.
> Build instructions for the **dissipative-Hamiltonian itinerancy simulator falsified at gate G3**
> (modules `sleep.py`/`SequenceRecon`, regime classifier, `(β,ρ_R)` sweep, `C_anti`/`ρ_R`). See
> `.agents/G3_escalation.md`. **Canonical spec: `.agents/field_gather_spec.md`; active plan:
> `.agents/field_gather_build_plan.md`.** The module layout / milestones below are obsolete.

# GRAPH–FIELD SYSTEM — IMPLEMENTATION SPEC (for Sonnet)  [SUPERSEDED]
## Build instructions for the dissipative-Hamiltonian neural-field simulator

> Source of truth: `architecture_spec.md` (canonical) + `invariant_set_phase_diagram.md`
> (regime/learning analysis). This document translates them into a buildable plan.
> Scope of this build: a **numerical simulator + Sleep trainer + regime experiments** —
> a prototype physics simulator, NOT a full research-validation system. Build toy graphs,
> trajectory tooling, and the `(β, ρ_R)` regime sweep first.
>
> Conventions (from `.agents/formatting.md`): hash comments only, **NO docstrings**;
> combined imports on one line; ≤2 consecutive blank lines; one-line function headers;
> compact layout. Treat every function as a liability — reuse/merge before adding.
> TDD: write the focused test, then the code; run `venv/bin/python -m pytest -q`.

---

## 0. WHAT YOU ARE BUILDING (one paragraph)

A closed dynamical system on a fixed graph `S`. Node states `X ∈ ℝ^{N×d}` evolve by
`Ẋ = −∇E(X) + R(X)`: a Cohen–Grossberg energy gradient (`−∇E`, the dissipative skeleton)
plus a divergence-free antisymmetric-coupling rotation (`R`). Energy is a *skeleton +
diagnostic*, not the governing function — convergence is **not** a goal; the output is the
system's **invariant set** (a settled mesh, a limit cycle, or an ordered itinerary of
meshes), selected by two knobs `(β, ρ_R)`. The coupling `C = C_sym + C_anti` is learned
offline (Sleep); everything at runtime is closed. No retrieval, no selection, no `m_t`, no
topology change. Honor every invariant in `architecture_spec.md` §9.

---

## 1. MODULE LAYOUT (`src/field/`, rewrite of the v5.3 package)

| file | responsibility | replaces |
|---|---|---|
| `config.py` | `FieldConfig` dataclass: `d, β, ρ_R, ρ_max, μ (barrier), τ_support, d_mesh, ε_occ, ε_x, η, H_max, H_hold, W, T_burn, T_class` + Sleep params | config.py |
| `episode.py` | materialize active set `N` from `S` (SQLite), build `A∈ℝ^{N×384}`, `X_0 = rownorm(P·A)` | spaces.py |
| `coupling.py` | build `C` on `E(S)`; split `C_sym/C_anti`; degree-normalize `C_sym`; enforce `‖C_anti‖≤ρ_R‖C_sym‖`; init `C_sym` from embeddings | kernel.py |
| `energy.py` | `E(X)` and `grad_E(X)` (couple + inhib + barrier); `R(X)=C_anti·X`; `lambda_max(C_sym)`, `R_max`, Lipschitz `L` | (new) |
| `dynamics.py` | generator `G`, semi-implicit/clamped integrator step, trapping guard, `rollout` with termination | dynamics.py |
| `observables.py` | `support`, online mesh clustering, itinerary/occupancy, regime classifier (FFT + Lyapunov) | (new) |
| `sleep.py` | `StructureRecon` + `SequenceRecon` losses, `C` update (Adam), trust region, EMA | learn.py |
| `harness.py` | toy-graph experiments + `(β,ρ_R)` sweep + trajectory dump | harness.py |

**Delete / do not port:** `memory.py` (`m_t` removed), `objective.py` (no control-as-loss),
`expansion.py` (no growth), `routing.py`/`operators.py` (no operator mixture), `interpreter.py`
(retrieval/selection). Keep `src/graph/` for `S` storage.

---

## 2. DATA STRUCTURES

```python
@dataclass
class Episode:
    node_ids: list[str]           # active set N (ordered)
    A: Tensor                     # [N, 384] frozen anchors
    edge_index: Tensor            # [2, E] pairs on E(S) (both directions stored once)
    id_to_idx: dict[str, int]

@dataclass
class Coupling:
    sym: Tensor                   # [N, N] symmetric, degree-normalized, support=E(S)
    anti: Tensor                  # [N, N] antisymmetric, support=E(S), ‖·‖≤ρ_R‖sym‖
    # C = sym + anti ; never materialize dense off-support

@dataclass
class Trajectory:
    X: Tensor                     # [T, N, d] full state history (or windowed)
    E: Tensor                     # [T] energy diagnostic
    supports: list[set[int]]      # per-step support sets
    mesh_ids: list[int]           # per-step mesh assignment
    stabilized_at: int | None

@dataclass
class Regime:
    kind: str                     # "point"|"cycle"|"torus"|"chaos"
    itinerary: list[int]          # mesh-id sequence (recurrent) or [single] (point)
    occupancy: dict[int, float]
    freqs: list[float]; lyap_max: float
```

---

## 3. CORE ALGORITHMS (pseudocode — match `architecture_spec.md` §4,5,14)

### 3.1 energy.py
```
grad_E(X, C, cfg):
    # couple: −C_sym X   (note sign: ∇E_couple = −C_sym X)
    g_couple = − C.sym @ X
    # inhib (LSE over nodes): softmax_i(β·½‖x_i‖²) · x_i
    e = 0.5 * (X*X).sum(-1)                      # [N]
    p = softmax(cfg.beta * e, dim=0)             # [N]
    g_inhib = p.unsqueeze(-1) * X
    # barrier: μ‖x_i‖² x_i  (cubic)
    g_barrier = cfg.mu * (X*X).sum(-1, keepdim=True) * X
    return g_couple + g_inhib + g_barrier

E(X, C, cfg): return E_couple + (1/β)·logsumexp(β·e) + Σ (μ/4)‖x_i‖⁴   # diagnostic only
R(X, C): return C.anti @ X
```

### 3.2 dynamics.py — integrator (handles §14.8 numerical dissipativity)
```
step(X, C, cfg):
    g = grad_E(X, C, cfg) - R(X, C)              # = ∇E − R  → descent uses −g... see below
    # generator G = −∇E + R = −(grad_E − R)
    G = -(grad_E(X,C,cfg)) + R(X,C)
    # explicit Euler on couple+inhib+rotation; barrier handled by clamp (v1) :
    X_next = X + cfg.eta * G
    # numerical dissipativity guard (§14.8): clamp to absorbing ball
    norms = X_next.norm(dim=-1, keepdim=True)
    X_next = where(norms > R_max, X_next * R_max / norms, X_next)
    assert X_next.norm(dim=-1).max() <= R_max + 1e-4   # trapping guard
    return X_next
# v2 optional: semi-implicit barrier (per-node radial closed form) instead of clamp.

rollout(X0, C, cfg) -> Trajectory:
    X = X0; hist=[]; occ_windows=...
    for t in range(cfg.H_max):
        X = step(X, C, cfg); record(X, E, support)
        update online mesh clustering → mesh_ids
        if t>=2W and TV(occ[t-W:t], occ[t-2W:t-W]) < ε_occ for H_hold:
            mark stabilized; break
        if point-test ‖ΔX‖<ε_x: mark stabilized(point); break
    return Trajectory(...)
```
Note: `η ≤ c/L`, `L = ‖C_sym‖ + β·B_inhib + 3μ R_max²`, `R_max = sqrt(λmax(C_sym)/μ)`.
`B_inhib` ≤ β·(max ‖x_i‖²) bound on `Ω` → use `β·R_max²`. Compute these in `coupling.py`
after building `C` and store on the config/coupling object.

### 3.3 observables.py (§14.2–14.6)
```
support(Xwin, cfg): ē = Xwin.pow(2).sum(-1).mean(0); return {i: ē_i >= τ·ē.max()}
cluster_mesh(supp, meshes, d_mesh): assign to nearest mesh by Jaccard<d_mesh else new
itinerary(mesh_ids): dedup consecutive
occupancy(mesh_ids): normalized counts
classify(traj, cfg):
    o = traj.E[T_burn:]
    freqs = dominant_freqs(fft(o))           # peaks above noise floor
    lyap = finite_time_lyapunov(rollout, X0, δ0)   # paired renormalized run
    if 1 mesh and ‖ΔX‖<ε_x: "point"
    elif broadband or lyap>0: "chaos"
    elif len(incommensurate freqs)>=2: "torus"
    else: "cycle"
```

### 3.4 coupling.py
```
build(episode, embeddings, cfg):
    sym0 = symmetric edge weights from embedding cosine on E(S)   # bootstrap teacher
    sym = degree_normalize(sym0)                                  # D^-1/2 sym D^-1/2
    anti = init_antisym(episode, cfg)                             # from edge direction or 0
    anti = rescale so ‖anti‖ <= ρ_R·‖sym‖
    return Coupling(sym, anti)
# all matrices masked to E(S); store λmax(sym), R_max, L.
```

### 3.5 sleep.py (§7, §14.7) — offline, between episodes
```
structure_recon(meshes, held_out_subdetails): MSE/recall of mesh support vs targets
sequence_recon(meshes, targets, C):
    loss = 0
    for (i,j),w in targets:           # held-out ordered relations
        d = meshes[j].rep - meshes[i].rep
        push = C.anti @ meshes[i].rep   # rotational push at mesh i
        loss -= w * cos(push, d)
    return loss
update(C, batch, cfg):
    L = structure_recon(...) + sequence_recon(...)
    backprop into C_sym (via E/StructureRecon) and C_anti (via SequenceRecon)
    project: mask to E(S); ‖C_anti‖<=ρ_R‖C_sym‖; trust region ‖ΔC‖<=δ
    EMA: C ← τ_C·C + (1−τ_C)·C_new
```
Validation only (non-diff): build target transition matrix `T*` and empirical `T̂` from
itineraries, report `KL(T*‖T̂)`.

---

## 4. TEST / VALIDATION PROTOCOL

Toy graphs (in `tests/field/`), each with a *predicted* regime so tests are falsifiable:
- **single clique** → point (one mesh).
- **two cliques + bridge** → at `ρ_R=0` two competing point basins; at moderate `ρ_R`
  expect itinerancy across the bridge (heteroclinic). Primary itinerary test.
- **ring of cliques** → at moderate `ρ_R` expect limit cycle / itinerary over the ring.
- **two incommensurate rings** → torus candidate.

Unit tests (assert behavior, not internals):
1. `⟨X,R⟩ ≈ 0` for random `X` (antisymmetry) — the dissipativity premise.
2. trapping: `max_i‖x_i‖ ≤ R_max` for all `t`, all toy graphs, any `ρ_R≤ρ_max`.
3. `ρ_R=0` ⇒ `E(X_t)` monotone non-increasing (Option-A recovery) within `η` tolerance.
4. `ρ_R=0` ⇒ regime classifier returns "point" on single clique.
5. support is τ-thresholded and stable under small perturbation of `X`.
6. mesh clustering is permutation-stable and merges Jaccard-near supports.
7. SequenceRecon: a hand-set target transition lowers loss when `C_anti` is aligned to
   `x*_j − x*_i` (gradient sign check).
8. integrator stability: no NaN/Inf over `H_max` steps at the configured `η` for all toys.

Experiments (`harness.py`, the prototype deliverable):
- **regime sweep**: grid over `(β, ρ_R)`; classify each; render the empirical phase diagram;
  check the qualitative predictions of `invariant_set_phase_diagram.md` (sharp Hopf-like
  transitions; orthogonality in interiors, coupling at boundary).
- **trajectory viz**: plot `E(t)`, occupancy, mesh-id timeline; dump for inspection.
- **reproducibility**: same `(X_0, C, cfg, seed)` ⇒ identical itinerary/occupancy.

---

## 5. MILESTONES (gate each before the next)

- **M1 — Simulator core.** `config, episode, coupling, energy, dynamics`. Acceptance:
  tests 1–3, 8 pass on single-clique + two-clique; `ρ_R=0` descends `E`; trapping holds.
- **M2 — Observables + classifier.** `observables`. Acceptance: tests 4–6; classifier labels
  single-clique "point", two-incommensurate-rings "torus" candidate; reproducibility holds.
- **M3 — Itinerancy.** Turn on `ρ_R`; two-cliques+bridge yields a measurable itinerary;
  occupancy stabilization triggers. Acceptance: itinerary readout non-trivial and reproducible.
- **M4 — Sleep / learning.** `sleep` with both losses; train `C` on a toy held-out structure.
  Acceptance: test 7; `StructureRecon` improves mesh-vs-target recall; `KL(T*‖T̂)` decreases
  after `SequenceRecon` training; constraints (`E(S)` support, `ρ_R` bound, trust region) hold.
- **M5 — Phase diagram.** `(β,ρ_R)` sweep + render. Acceptance: empirical regime map shows the
  predicted regime classes and sharp transitions; document deviations.

---

## 6. INVARIANT CHECKLIST (assert in code / CI — from spec §9)

1. `S` never written at runtime (no mutation of graph tables during rollout).
2. `C` masked to `E(S)`; `C ⊥ X_t` within an episode (built once per episode, not per step).
3. no retrieval / selection / injection beyond `X_0`; no `m_t`.
4. dynamics is exactly `−∇E + R`; `R = C_anti X`, `C_anti` antisymmetric.
5. trapping guard active every step (`R_max`).
6. learning only in `sleep.py`, offline; `ρ_R ≤ ρ_max`; trust region + EMA enforced.
7. readout from stabilized invariant set per §14.5 — never from a single arbitrary step.

If any conflict with `architecture_spec.md` arises during build, **stop and consult**
(AGENTS.md §3) — do not silently deviate.

---

END IMPLEMENTATION SPEC
