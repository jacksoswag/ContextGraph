> # ⛔ SUPERSEDED — DO NOT FOLLOW. Kept for history only.
> This describes the **dissipative-Hamiltonian *itinerancy* model**, which was **falsified at gate
> G3** (see `.agents/G3_escalation.md`): the rotation `R=C_anti·X` provably cannot drive cross-mesh
> itinerancy, and the energy has no localized competing meshes. **Canonical architecture is now
> `.agents/field_gather_spec.md`** (the field is a convergent context-*gatherer*, not a reasoner);
> the active build/test plan is `.agents/field_gather_build_plan.md`. Anything here about rotation,
> `ρ_R`, regimes, itinerancy, or `SequenceRecon` is obsolete.

# GRAPH–FIELD SYSTEM — CANONICAL ARCHITECTURE  [SUPERSEDED]
## Dissipative-Hamiltonian Neural-Field Dynamics on a Fixed Graph

> **STATUS: CANONICAL — SINGLE SOURCE OF TRUTH.**
> This document is the authoritative architecture specification referenced by
> `.agents/AGENTS.md` §3. It supersedes and voids all prior versions (the v5.3 "grounded
> specification" and the v7 strict-gradient / Option-A candidate).
> Normative companion: `invariant_set_phase_diagram.md` (the bifurcation analysis governing
> regime selection and the `SequenceRecon` learning objective).
> Last revised: 2026-06-01.
>
> Terminology guard: "**Option B**" (the dissipative non-gradient regime adopted here) is
> unrelated to the deleted "**B_t**" retrieval operator. B_t remains removed; there is no
> retrieval anywhere in the system.

---

## 0. SYSTEM CLASS & NON-GOALS

The system is a **closed dissipative dynamical system on a fixed graph** whose generator is

```
Ẋ = G(X) = −∇E(X) + R(X)
```

a dissipative **gradient skeleton** `−∇E` plus a **structured non-conservative flow** `R`.
Energy `E` is a **shaping constraint and diagnostic / dissipative skeleton — NOT the full
generator of motion.**

Reasoning is realized as the system's **invariant set**, which — depending on the regime
parameter `ρ_R` — may be:
- a **point** (a settled concept-mesh), or
- a **limit cycle** (a sustained rhythm over a concept cluster), or
- a **metastable itinerary** (an ordered sequence of concept-meshes) — *reasoning-as-process*.

It is **NOT** RAG, retrieval, graph search, message-passing-with-selection, or
attention-as-retrieval. **No retrieval (B_t), no m_t, no runtime node selection, no
external injection beyond initialization, no topology mutation.**

**Option A is the `ρ_R = 0` limit** (`R ≡ 0`): pure gradient flow, point attractors only,
Lyapunov-inherited convergence. This architecture contains it as a special case.

---

## 1. STRUCTURE S (immutable substrate)

- `S = (V, E)`, optional hyperedges. Each node `v`: frozen anchor `a_v ∈ ℝ^384`.
- Immutable during inference. Plastic only in the offline Sleep phase (§7). Never
  `X_t → S` at runtime.

---

## 2. STATE (ephemeral field)

- `X_t ∈ ℝ^{N×d}` over active set `N`, materialized once at episode start, fixed for the
  episode. Reinitialized each episode, never persisted. The **only** dynamic variable —
  no `m_t`, no `B_t`.

---

## 3. COUPLING C (geometry, fixed within an episode)

The single learned edge object is a **general (not necessarily symmetric) coupling**
`C` on `E(S)`, decomposed into its conservative and non-conservative parts:

```
C = C_sym + C_anti
    C_sym  = ½(C + Cᵀ)      symmetric    → conservative (gradient) flow  →  E
    C_anti = ½(C − Cᵀ)      antisymmetric → non-conservative (rotational) flow  →  R
```

- **Support = `E(S)` exactly** (load-bearing for closure; dense `C` = similarity-access =
  retrieval, forbidden).
- `C_sym` degree-normalized (`D^{-1/2} C_sym D^{-1/2}`) — hub-sink control.
- **Non-conservative magnitude bounded:** `‖C_anti‖ ≤ ρ_R · ‖C_sym‖`, where `ρ_R ≥ 0` is
  the regime parameter (§8). This bound governs the invariant-set regime (not boundedness).
- Initialization: `C_sym` from frozen LLM embeddings (bootstrap "dumb teacher", weighting
  existing edges); `C_anti` from a relational/directional edge signal (e.g. asymmetric
  relation type / edge orientation) or zero, then learned in Sleep.
- **Fixed during inference**; adapts only across episodes via Sleep (§7).

---

## 4. ENERGY (dissipative skeleton — shaping, not governing)

`E` uses **only** the symmetric coupling `C_sym`:

```
E(X) = E_couple + E_inhib + E_barrier

E_couple  = −½ Σ_{(i,j)∈E(S)} (C_sym)_ij ⟨x_i, x_j⟩
E_inhib   = (1/β) · log Σ_{i∈N} exp(β · ½‖x_i‖²)
E_barrier = Σ_{i∈N} r(‖x_i‖),   r super-quadratic (e.g. (μ/4)‖x_i‖⁴)
```

Gradients: `∇_i E_couple = −Σ_j (C_sym)_ij x_j`, `∇_i E_inhib = softmax_i(β·½‖x‖²)·x_i`,
`∇_i E_barrier = r'(‖x_i‖)·x̂_i`.

**Role downgrade (the heart of the design).** `E` is *not* minimized along trajectories.
Along the flow,
```
Ė = ∇E·Ẋ = −‖∇E‖² + ∇E·R     (sign-indefinite)
```
so `E` may increase. `E` is retained for two purposes only:
1. **Dissipative skeleton:** the coercive barrier + the `−∇E` part define a trapping region
   (§5). Coercivity (barrier dominates the coupling's negative curvature) is required —
   for **dissipativity/boundedness**, not for convergence-to-minimum.
2. **Diagnostic:** `E`, `Ė`, and sub-level occupancy are logged to characterize the flow.

`E` is **not** a Lyapunov function in general (it is one only in the `ρ_R = 0` / Option-A
limit, or when `C_anti` is shaped so `∇E·R ≤ 0`).

**Nature of the split (explicit — no implied orthogonality).** `G = −∇E + R` is an
*additive* decomposition of the generator, **not** an orthogonal Helmholtz projection.
`∇E·R ≠ 0` in general — it is **not assumed zero**. However `R` is restricted to the
**divergence-free** class (§3, antisymmetric coupling), so it is genuinely separable from
the dissipative part: `R = C_anti·∇(½‖X‖²)` is the **norm-conserving Hamiltonian rotation**
generated by `C_anti` — it conserves `‖X‖²`-shells (`⟨X,R⟩ ≡ 0`) but **not** `E`
(`∇E·R ≠ 0`). Both facts coexist: `R` does no *radial* work yet does work on `E`. A
*dissipative non-gradient* `R` (one with nonzero divergence) is **excluded** by construction.

---

## 5. DYNAMICS (generator with non-conservative flow)

```
continuous:  Ẋ = G(X) = −∇E(X) + R(X),     R_i(X) = Σ_j (C_anti)_ij x_j
discrete:    X_{t+1} = X_t + η · G(X_t)
```

`R` is structured (graph-local, supported on `E(S)`), non-potential (antisymmetric
coupling has no scalar potential), **divergence-free** (`trace(C_anti)=0` ⇒ phase-volume-
preserving, norm-conserving), and bounded (via `ρ_R`). It injects **rotation** into the
flow. The *dissipative non-gradient* class (nonzero-divergence `R`) is excluded by
construction — `R` is purely the symplectic/rotational part, all dissipation lives in `−∇E`.

**Stability is ENFORCED, not inherited:**

- **Dissipativity (boundedness) — guaranteed via the radial storage function `½‖X‖²`.**
  The absorbing-set argument is run against `½‖X‖²`, **not** `E`:
  `d/dt(½‖X‖²) = ⟨X,−∇E⟩ + ⟨X,R⟩`, and `⟨X,R⟩ = Σ_ij (C_anti)_ij⟨x_i,x_j⟩ ≡ 0` because
  `C_anti` is antisymmetric (it conserves `‖X‖`-shells). So `R` drops out of the radial
  balance **identically and independently of `ρ_R`**, and the super-quadratic barrier gives
  `d/dt(½‖X‖²) ≤ −μ‖X‖⁴ + O(‖X‖²) → −∞`, yielding a compact absorbing set `Ω`. This global
  guarantee is **unconditional for divergence-free `R`** (the chosen class). *Scope caveat:*
  if `R` is ever generalized beyond antisymmetric coupling (nonzero radial work), this
  argument fails and an explicit radial-dominance bound `‖R‖ = o(‖X‖³)` must be re-imposed.
  Note `Ω` being absorbing does **not** imply convergence — limit cycles/tori live *inside*
  `Ω` and are intended.
- **Invariant-set stability — certified per regime (§8), not free.** Point attractor →
  `Re(λ(J)) < 0` at the equilibrium (now possibly a spiral sink); limit cycle → transverse
  Floquet stability; metastable itinerary → transverse contraction of the heteroclinic
  channel. None is supplied by `E`.
- **Step bound** `η` chosen for dissipativity / no discretization-induced spurious cycling
  (the discrete map can manufacture cycles if `η` too large, independent of `R`).

**Noise becomes a feature.** Optional bounded stochastic forcing
`X_{t+1} = X_t + ηG + √(2ηT_k)ξ` drives **transitions between metastable meshes**
(heteroclinic hopping / itinerancy) rather than being a nuisance. Readout time-averaged.

**`X_0` (§6) is the only external entry point.** Closed thereafter.

---

## 6. INITIALIZATION (only external input)

- `X_0`: initial excitation on the active set `N`. `N` materialized once from `S` at `t=0`;
  its breadth bounds reachability (closed system redistributes only over `N`). No runtime
  re-injection or re-seeding.

---

## 7. OFFLINE ADAPTATION — SLEEP (only place C and S change)

Timescale separation: fast field dynamics at runtime; slow `C`/`S` update offline.
`C` (both parts) and `S` are learned here only.

```
maximize  J(C) =  StructureRecon( point-attractor support ≈ held-out sub-details )      # shapes C_sym / E-minima
                + SequenceRecon ( transport alignment with saddle skeleton )             # shapes C_anti / flow
subject to  ‖C_anti‖ ≤ ρ_R · ‖C_sym‖,  ρ_R ≤ ρ_max                 # regime bound (§5, §8)
            support_breadth(mesh) ∈ [L_min, L_max]                  # β / entropy-tuned
            ‖C − C_prev‖ ≤ δ                                        # trust region (keep bootstrap geometry)
            C on E(S)
update      slow EMA (τ_C → 1)
```

- `C_sym` shapes **which** concept-meshes are low-energy basins *and the saddles between
  them* (the curvature skeleton).
- `C_anti` is **transport geometry, not a structure generator.** It does not create cycles
  or itineraries; it **aligns rotational transport with the existing saddle skeleton of
  `E`**. `SequenceRecon` therefore optimizes *transport alignment* — aligning `C_anti`'s
  rotational 2-planes with the unstable manifolds of the saddles separating consecutive
  target meshes — **not** "cycle induction" (which fights curvature and overfits). See
  `invariant_set_phase_diagram.md` §7. This is intrinsically regularized: `C_anti` can only
  route along ridges `E` already has.
- Structure `S` plasticity happens only here.

---

## 8. ATTRACTOR REGIMES & READOUT (the core generalization)

**Principle (curvature × rotation).** The invariant set is an *emergent interaction* of
basin/saddle curvature (`E`, set by `β` / `C_sym` / graph spectrum) and rotation (`C_anti`,
scaled by `ρ_R`) — neither alone. Linearizing gives `J = −H + A` (`H = ∇²E` symmetric,
`A = C_anti` antisymmetric): a strict basin (`H≻0`) is a stable focus for *any* `ρ_R`
(rotation cannot destabilize curvature); recurrence requires `H` to flatten (`λ_min(H)→0`)
in a rotating plane. So `(β, ρ_R)` are orthogonal in regime interiors but **couple on the
bifurcation boundary** `λ_min(H(β))=0 ∩ ω(ρ_R)≠0`. Full derivation:
`invariant_set_phase_diagram.md`. The admissible invariant set is selected by `(β, ρ_R)`:

| `ρ_R` | invariant set | local form | readout |
|---|---|---|---|
| `0` | point (Option A) | node / sink | bump **support** |
| small | point | **spiral sink** (rotational transient) | bump support |
| moderate | **limit cycle / metastable itinerary** | Hopf / heteroclinic channel | **itinerary** (ordered sequence of supports) and/or time-averaged **occupancy** |
| moderate, balanced skew spectrum | **quasi-periodic invariant torus** | near-neutral incommensurate imaginary pairs (damping balanced, not dominant) | occupancy / spectral signature of the structured oscillation (stable, non-chaotic) |
| large | chaos / sensitive dependence (still inside `Ω`) | — | **forbidden** (§10) |

- `β` sets **support breadth** of each mesh (spike ↔ consensus); target = moderate β.
- `ρ_R` sets **how non-conservative** the flow is (point ↔ itinerant); target depends on task.
- **Readout is regime-dependent and read from the *stabilized invariant set*, not from a
  settled `X`.** Point regime → support. Recurrent regime → the itinerary (ordered concept-
  meshes visited) and/or occupancy distribution over meshes. This is where *reasoning-as-a-
  trajectory* is expressed: the answer is the path through metastable meshes, not a single
  endpoint.

---

## 9. INVARIANTS (asserted)

1. `S` immutable at runtime; plastic only in Sleep.
2. `C` support = `E(S)`; `C = C_sym + C_anti`; `‖C_anti‖ ≤ ρ_R‖C_sym‖`, `ρ_R ≤ ρ_max`;
   fixed during inference.
3. No retrieval (B_t), no selection, no external injection beyond `X_0`; no `m_t`; no
   topology mutation.
4. `C ⊥ X_t` at runtime (coupling fixed within an episode).
5. `E` is the dissipative skeleton + diagnostic, **not** the governing function; motion is
   `−∇E + R`. `E` may increase along trajectories.
6. **Dissipativity guaranteed** for divergence-free `R`: coercive barrier + `⟨X,R⟩≡0`
   (antisymmetry) ⇒ compact absorbing set `Ω` ⇒ bounded trajectories — *independent of*
   `ρ_R`. **Convergence is NOT guaranteed — by design** (recurrent sets live inside `Ω`).
   The `ρ_R` bound governs the *invariant-set regime* (§8), not boundedness.
7. Stability of the intended invariant set is **certified per regime**, not inherited.
8. Learning offline only; grounded objectives (`StructureRecon` + `SequenceRecon`); slow
   EMA; trust region; `ρ_R ≤ ρ_max`.

---

## 10. FAILURE MODES & GUARDS

The failure surface is B's "uncontrolled limit set" (not A's "over-convergence").

| Failure | Cause | Guard |
|---|---|---|
| Unintended limit cycle (point intended) | `ρ_R` too high for task | lower `ρ_R`; certify spiral-sink stability (`Re λ(J)<0`) |
| Chaos / sensitive dependence (non-reproducible readout) | `ρ_R` large, non-normal `J` | cap `ρ_R ≤ ρ_max`; monitor finite-time Lyapunov exponent / perturbation divergence; read via **occupancy** (robust to chaotic detail) |
| Divergence (dissipativity lost) | only if `R` generalized beyond antisymmetric (nonzero radial work); or discretization with `η` too large | keep `R` divergence-free (antisymmetric) ⇒ `⟨X,R⟩≡0` ⇒ boundedness holds; super-quadratic barrier; `η` bound; **trapping-region guard** (assert `X_t ∈ Ω`) |
| Path / initial-condition dependence | multi-basin + non-conservative flow | accepted as *feature* for itinerant reasoning; reproducibility guarded via occupancy readout + bounded noise |
| Bifurcation fragility | `(β, ρ_R)` near Hopf / homoclinic threshold | keep parameters off known thresholds; Sleep trust region forbids large jumps |
| Collapse to trivial mesh (A-side) | `ρ_R = 0`, `β` mis-set | `L_min/L_max` breadth bounds; degree-normalization |

---

## 11. SYSTEM-CLASS NOTE (honest scope of guarantees)

This is a **dissipative-Hamiltonian (metriplectic-type) neural field** —
`Ẋ = −∇E + C_anti·∇(½‖X‖²)`: gradient dissipation (`−∇E`, Cohen–Grossberg / log-partition
form) plus a bounded **norm-conserving symplectic transport** (`R`, antisymmetric coupling).
Equivalently: a Hopfield/Cohen–Grossberg energy basin structure under a divergence-free
Hamiltonian perturbation. Dissipation and rotation are generated by *different* functions
(`E` and `½‖X‖²`), which is why `E` is not conserved while `‖X‖²`-shells are respected by the
rotational part.

- **Guaranteed:** a compact absorbing set ⇒ bounded trajectories (§9.6), under coercivity
  and the divergence-free `R` constraint.
- **NOT guaranteed (deliberately forfeited):** convergence to a fixed point, `E`-monotonicity,
  uniqueness of the limit set. These are surrendered in exchange for recurrent / oscillatory
  / itinerant invariant sets — the formal A→B trade (strictly larger realizable
  invariant-set class, at the cost of inherited convergence/stability guarantees).
- **Stability of the desired behavior is an obligation**, discharged by the `ρ_R` bound,
  per-regime certification (§5, §8), and Sleep shaping (§7) — never inherited from `E`.
- **Option A is the `ρ_R = 0` special case** and remains available for tasks that want a
  guaranteed-convergent point answer.

---

## 12. MIGRATION NOTE (relationship to current code)

- coupling module: generalize symmetric `φ` → `C = C_sym + C_anti`; expose `ρ_R`.
- `dynamics.py`: generator `G = −∇E + R` (single combined step); add **trapping-region /
  dissipativity guard**; replace "`ΔX < ε`-only" termination with **invariant-set
  stabilization** (occupancy distribution converges) **OR** `ΔX < ε` (point regime) **OR**
  horizon `H`.
- `memory.py`: `m_t` removed (no consumer).
- `objective.py`, B_t, `expansion.py`: removed (no control-as-loss, no retrieval, no growth).
- `learn.py` / `consolidate.py` (Sleep): add `C_anti` learning as **transport alignment**
  (`SequenceRecon` = align `C_anti` eigenplanes with saddle unstable-manifolds between
  target meshes — *not* cycle induction); enforce `ρ_R ≤ ρ_max`; keep `StructureRecon` for
  `C_sym`. See `invariant_set_phase_diagram.md` §7.
- readout: **support** for the point regime; **itinerary / occupancy** for the recurrent
  regime.

---

## 13. SPECIFICATION STATUS (fixed vs open)

**Fixed (normative — do not deviate without consultation per AGENTS.md §3):**
- Closed system; immutable `S` at runtime; plastic only in Sleep (§1, §7).
- Single state field `X_t`; no `m_t`, no `B_t`, no retrieval / selection / injection (§2, §9).
- Coupling `C = C_sym + C_anti` supported on `E(S)`; `C ⊥ X_t` at runtime (§3).
- Generator `Ẋ = −∇E + R`, `R` divergence-free antisymmetric; `E` is skeleton + diagnostic,
  not governing (§4, §5).
- Dissipativity via `½‖X‖²` (`⟨X,R⟩≡0`); convergence not guaranteed by design (§5, §9.6).
- Regime selection by `(β, ρ_R)`; readout = support (point) / itinerary+occupancy
  (recurrent) (§8).
- Learning offline only; `C_anti` = transport alignment, not cycle induction (§7).

**Now specified (§14 — operational definitions, frozen for implementation):**
- Support, mesh, invariant-set stabilization, readout (§14.2–14.5).
- Regime classification protocol (§14.6).
- Concrete `SequenceRecon` loss + validation metric (§14.7).
- Integrator, absorbing radius, step bound, numerical dissipativity (§14.8).

**Genuinely open (validation-time, not build blockers):**
- Operating regime (`β`, `ρ_R`) per task class — to be swept empirically.
- Full numerical continuation of the `(β, ρ_R)` bifurcation loci and the saddle-connection
  graph of `E` (`invariant_set_phase_diagram.md` §9) — a validation artifact, not required
  to run the simulator.

None of these change the invariants in §9.

---

## 14. OPERATIONAL DEFINITIONS (frozen for implementation)

These pin the §0–§12 semantics to computable quantities. The *definitions* are normative;
the bracketed hyperparameter defaults are tunable.

### 14.1 State representation — node-only
`X ∈ ℝ^{N×d}` (default `d=64`). No global latent, no hierarchy. `X_0 = rownorm(P·A)`,
`P: ℝ^384→ℝ^d` a fixed projection (Sleep-learnable later). Anchors `a_v` frozen.

### 14.2 Support (point-regime observable)
Over readout window `W`, `ē_i = mean_{t∈W} ‖x_{i,t}‖²`.
`support_τ(X) = { i : ē_i ≥ τ·max_j ē_j }`, default `τ=0.5`, optional cap `top-k_max`.

### 14.3 Mesh (state→symbol clustering)
A **mesh** = a cluster of per-step support-sets under **Jaccard distance < d_mesh**
(default 0.3), via greedy online clustering. Mesh representative `x*` = mean field over
members. A mesh-id is assigned each step.

### 14.4 Invariant-set stabilization (when to read out)
Windowed **occupancy** `q_W` = time-fraction per mesh over `W`. **Stabilized** when
`TV(q_{[t-W,t]}, q_{[t-2W,t-W]}) < ε_occ` (default 0.05) held `H_hold` steps; else stop at
`H_max`. Point regime additionally requires `‖ΔX‖ < ε_x`.

### 14.5 Readout
- Point regime → `support_τ(X*)`.
- Recurrent regime → **itinerary** (consecutive-dedup mesh-id sequence) + **occupancy** `q`.

### 14.6 Regime classification protocol (falsifiable)
Discard transient `T_burn`. From `o(t)=E(X_t)` (and field centre-of-mass): dominant
frequencies via FFT/autocorrelation; `λ_max` = finite-time largest Lyapunov exponent from a
paired run with renormalized separation `δ_0`. Rules: **point** (occupancy on 1 mesh ∧
`‖ΔX‖<ε_x`); **limit cycle** (1 dominant freq, `λ_max≈0`); **torus** (≥2 incommensurate
freqs, `λ_max≈0`); **chaos** (broadband, `λ_max>0`). Classify only after `≥ T_class` post-burn
steps.

### 14.7 SequenceRecon loss (concrete)
Per target transition `mesh_i → mesh_j` (held-out ordered relations of `S`), reps
`x*_i,x*_j`, `d_ij = x*_j − x*_i`:
```
SequenceRecon = − Σ_{(i,j)∈targets} w_ij · cos( C_anti · x*_i , d_ij )
```
align the rotational push at mesh `i` toward the next target mesh `j`. Differentiable in
`C_anti`, graph-local. **Validation** metric (non-differentiable): `KL(T* ‖ T̂)` between
target and empirical mesh-transition matrices.

### 14.8 Integration & numerical dissipativity
- Absorbing radius `R_max ≈ sqrt(λ_max(C_sym)/μ)` (from `μR⁴ ≳ λ_max(C_sym)R²`).
- Barrier gradient is **cubic** ⇒ not globally Lipschitz ⇒ explicit Euler can diverge.
  Integrate barrier **semi-implicitly** (closed-form per-node radial solve) or clamp
  `‖x_i‖ ≤ R_max` each step; explicit Euler for couple + inhib + rotation.
- Step bound `η ≤ c/L`, `L ≈ ‖C_sym‖ + β·B_inhib + 3μR_max²` (Lipschitz of `∇E` on `Ω`),
  `c<2`. **Trapping guard:** assert `max_i‖x_i‖ ≤ R_max` each step.

---

END SPEC
