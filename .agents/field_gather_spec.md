# FIELD-GATHER ARCHITECTURE — CANONICAL SPEC
## A convergent context-collection field under recursive LLM orchestration

> Supersedes the itinerancy ambition of `architecture_spec.md` (the dissipative-Hamiltonian /
> `ρ_R>0` rotation model). That model was **falsified at gate G3** — see `.agents/G3_escalation.md`:
> `R=C_anti·X` provably cannot drive cross-mesh itinerancy (field-of-values theorem) and the
> energy has no localized competing meshes (the couple term spreads to global co-activation).
> **The property that killed itinerancy — convergent spreading aggregation — is exactly what a
> context-collection system wants.** This spec reframes the field as a *gatherer*, not a reasoner.
>
> Style: hash comments only in code, no docstrings; see `.agents/formatting.md`. AGENTS.md §3
> applies — stop and consult before deviating from anything marked **[FIXED]**.

---

## 0. WHAT THIS IS (one paragraph)

The system answers a query by alternating two operations in a tree. The **LLM is the reasoning
transform**: it breaks a task into sub-tasks, decides when enough is known, and synthesizes prose.
The **physics is a context gatherer**: given seed concepts, it runs a convergent field over the
graph `S` that *spreads activation from the seeds and settles*, returning the **mesh** (the
relevant concept set it pulled in) plus a **provenance tree** (how each gathered concept connects
back to the seeds). Reasoning = recurse: LLM decomposes → physics gathers per sub-task (seeded and
**anchored to the parent's mesh**) → recurse → LLM synthesizes the children back up. The physics
never reasons, decides, or sequences; it only gathers and converges. The LLM only ever sees graph
content the physics gathered.

**Non-goals [FIXED]:** no rotation, no itinerancy, no limit cycles/regimes, no `ρ_R`, no
`SequenceRecon`. No runtime mutation of `S`. The physics makes no decisions.

---

## 1. THE LOOP (transform × gather, recursive)

```
answer(query):
    seeds, intent = interpret(query)                 # LLM seam 1 + graph match
    return solve(intent, seeds, parent_mesh=None, depth=0)

solve(task, seeds, parent_mesh, depth):
    mesh = gather(seeds, parent_mesh)                 # PHYSICS: seed-anchored convergent settle
    if depth >= MAX_DEPTH or sufficient(task, mesh):  # LLM seam 3 (stop gate) + budget
        return synthesize(task, mesh, [])             # LLM seam 4: mesh → prose+citations
    subtasks = decompose(task, mesh)                  # LLM seam 2: split into sub-questions
    children = [ solve(st, seed_select(st, mesh), mesh, depth+1) for st in subtasks ]
    return synthesize(task, mesh, children)           # LLM seam 4: combine children up the tree
```

- **One physics op:** `gather`. **Four LLM seams:** `interpret`, `decompose`, `sufficient`,
  `synthesize`. Nothing else calls an LLM; nothing else touches `S`.
- **Down-pass** = decompose + gather (specialize, pull context). **Up-pass** = synthesize
  (compress children into the parent answer). The tree is the reasoning trace.

---

## 2. STRUCTURE S (immutable substrate)  [FIXED]

- `S` = the concept graph in `src/graph/` (sqlite store). Nodes = concepts, each with a **frozen
  anchor embedding** `a_v ∈ ℝ^384` (from `src/embed.py`). Edges `E(S)` carry weight + relation,
  produced by `src/ingest/` (text→triple producer).
- `S` is **immutable at runtime**; plastic only in Sleep (§6). The gather reads `S`, never writes.
- **HYPEREDGES [approved deviation 2026-06-02].** `S` is hyperedge-native: every edge is reified as
  a first-class `e_`-id entity with its own anchor (`edges.info_vector`); an edge endpoint may be a
  node (`n_`) or another edge (`e_`) — the field reads hyperedge-ness from the id namespace, it does
  not distinguish. A reified edge's **children** = its `(source_id, target_id)`. `materialize` pulls
  a reified edge's children into the active set (they may be unreachable by k-hop) and records
  `Episode.hyperedges` member groups; the mesh provenance tree routes through them. This re-derives
  the hyperedge structure dropped at the G3 prune (`7a8eeaa`), but re-expressed for the convergent
  Lyapunov field (NOT the retired `Π_S` projection). See §3.3 containment term.

---

## 3. THE PHYSICS GATHER (the core)

### 3.1 Active-set materialization  [FIXED semantics, tunable radius]
From seed node-ids, materialize the active set `N` = seeds ∪ their `k`-hop neighborhood in `E(S)`
(default `k_hop=2`, capped at `N_max` nodes by descending edge-weight). Reachability bounds what a
gather can pull — it cannot gather outside the seeds' neighborhood. Build `edge_index` on `E(S)∩N`.

### 3.2 State, anchors, init
`X ∈ ℝ^{N×d}` (default `d=64`). `A ∈ ℝ^{N×384}` frozen anchors for `N`. `P: ℝ^384→ℝ^d` fixed
projection. `X_0 = rownorm(P·A)`. Seed rows initialized hot: `x_i ← R_max·c_seed · rownorm(P·a_i)`
for `i∈seeds` (default `c_seed=0.7`); non-seeds start cool (`~0`).

### 3.3 Energy (couple + inhib + barrier + seed-anchor + decay) — NO rotation  [FIXED]
```
E(X) = −½ Σ_ij C_sym[i,j]⟨x_i,x_j⟩            # couple   : spread activation along edges
       + (1/β) logΣ_i exp(β·½‖x_i‖²)           # inhib    : competition → selectivity (breadth)
       + Σ_i (μ/4)‖x_i‖⁴                        # barrier  : coercive bound
       + Σ_{i∈anchors} (σ/2)‖x_i − s_i‖²        # anchor   : pin seeds/parent state (NEW)
       + Σ_{i∉anchors} (λ/2)‖x_i‖²              # decay    : leak ⇒ localized fixed point (📍1)
∇_{x_i}E = −(C_sym X)_i + p_i x_i + μ‖x_i‖²x_i + σ(x_i−s_i)·[i∈anchors] + λ x_i·[i∉anchors]
           p = softmax_i(β·½‖x_i‖²)
```
- `C_sym` = degree-normalized cosine coupling on `E(S)∩N` (reuse `coupling.build`'s sym path),
  Sleep-shapeable (§6). **`C_anti` is removed entirely.**
- **HYPEREDGE CONTAINMENT [approved deviation 2026-06-02].** `+ w_hyper·Σ_{e} Σ_{i,j∈M(e), i≠j}
  ⟨x_i,x_j⟩` — a clique attraction among each reified edge's members `M(e)={e, src(e), tgt(e)}`,
  so energy stays within a hyperedge (the fact moves as a unit). Added to `C_sym` **after** degree
  normalization (folding it in before would inflate members' degrees and dilute the parent edge's
  own coupling, collapsing the fact's relevance — measured). It is a symmetric quadratic form ⇒ `E`
  stays a Lyapunov function; `L`/`R_max`/`η` are computed on the final `sym`, so convergence/safety
  are preserved. `w_hyper=0` ⇒ flat dyadic (pre-hyperedge); default `0.5` (content co-activates,
  seed stays hottest). Validated: content recall `0.0→1.0` (`scripts/hyperedge_eval.py`).
- The **anchor term is a true potential** ⇒ `E` stays a Lyapunov function ⇒ convergence is
  preserved while the gather is pinned to the seeds (and to parent state, §5). `s_i` = the seed
  init state for seed anchors, the parent's settled `x*_i` for inherited anchors.
- **📍1 AMENDMENT (decay term `λ`):** without it the symmetric-couple fixed point is *global
  co-activation* (G3 pathology) — on real seeds the settled mesh covers 41–76% of the reachable
  set and the seed is not even top-20 by relevance. The per-node leak `λ‖x_i‖²` on non-anchored
  rows makes the gather a **damped diffusion** whose *fixed point itself* localizes (seed hottest,
  monotone relevance-by-hop decay, settles ~300–600 steps). `λ` is the locality knob (default
  1.5; precise operating point = 📍3). The readout therefore ranks by relevance `r_i=‖x*_i‖²` and
  uses a soft/top-k mesh threshold — the hard `τ=0.5` of §3.5 collapses to seed-only under the
  steep decay (the **ranking** is the signal, not a magnitude cut).

### 3.4 Dynamics — pure gradient settle (guaranteed convergent)  [FIXED]
`Ẋ = −∇E(X)`, explicit step `X_{t+1}=X_t − η∇E`, `η ≤ c/L` (reuse the `L`/`R_max`/trapping
machinery in `coupling.py`/`dynamics.py`). `E` is coercive (barrier+anchor) and monotone
non-increasing ⇒ bounded trajectory ⇒ converges to a critical point. **Settle** = sustained
`‖ΔX‖<ε_x` for `H_hold` steps (the existing `rollout` criterion), else `H_max`.

### 3.5 Mesh + provenance readout  [FIXED outputs; algorithm tunable]
- **mesh** = `support_τ(X*) = { i : ‖x*_i‖² ≥ τ·max_j ‖x*_j‖² }` (reuse `observables.support`).
- **relevance** `r_i = ‖x*_i‖²` (settled energy) — ranks/prunes gathered context.
- **provenance tree** = a spanning tree over the mesh rooted at the seeds: BFS-layer the mesh by
  graph distance from seeds; each non-seed node `j`'s parent = `argmax_i flow[i,j]` over mesh
  neighbors `i` in a lower layer, `flow[i,j] = C_sym[i,j]·⟨x*_i,x*_j⟩`. Gives each gathered
  concept a citation chain back to a seed (the "tied to parents" structure).
- `Mesh = { nodes, relevance, tree, seed_roots }`.

### 3.6 Gather parameters
`β` = **gather breadth** (high→tight/precise, low→broad/exploratory); `σ` = anchor strength
(tie-to-seed); `k_hop`, `N_max` = reachability; `τ` = mesh threshold; `μ,η,ε_x,H_hold,H_max` as
existing. These are the only gather knobs.

---

## 4. THE LLM TRANSFORMS (the four seams)  [FIXED interface; prompts tunable]

Each seam is a pure function with a JSON contract; deterministic+cached given inputs. Model tiers
via env (`DI_MODEL_*`), small tier for `interpret`/`sufficient`, larger for `decompose`/`synthesize`.

| seam | in → out | role |
|---|---|---|
| `interpret(query)` | query → `{seeds:[node_id], intent:str}` | map prompt to graph entry points |
| `decompose(task, mesh)` | task + gathered mesh → `{subtasks:[str]}` or `{done:true}` | split |
| `sufficient(task, mesh)` | task + mesh → `{stop:bool, reason}` | stop gate (cheap tier) |
| `synthesize(task, mesh, children)` | task + mesh + child answers → `{prose, citations:[node_id]}` | transform |

`interpret` resolves entities to node-ids via embedding nearest-neighbor in `S` (the `embed` +
`graph` modules), not free generation — seeds must be real nodes.

---

## 5. THE RECURSION (the tree)  [FIXED]

```
solve(task, seeds, parent_mesh, depth):
    anchors = seeds ∪ top_k(parent_mesh.nodes by relevance, K_inherit)   # tie child to parent
    s = { i: seed_state(i) for i in seeds } ∪ { i: parent_mesh.x_star[i] for i in inherited }
    mesh = gather(active_set(anchors), anchor_targets=s)
    if depth ≥ MAX_DEPTH or budget_exhausted or sufficient(task, mesh).stop:
        return synthesize(task, mesh, [])
    children = [ solve(st, seed_select(st, mesh), mesh, depth+1) for st in decompose(task, mesh) ]
    return synthesize(task, mesh, children)
```

- **Parent-anchoring [FIXED]:** a child gather inherits the parent mesh's top-`K_inherit` nodes as
  anchors with the parent's settled state as targets — so the child's settled mesh is pulled toward
  the parent's region. This is the mechanism for "children tied back to higher parents."
- **Termination [FIXED]:** any of — `depth ≥ MAX_DEPTH`, global budget (max total gathered nodes
  or max LLM calls) exhausted, or `sufficient` returns stop.
- **Up-synthesis:** `synthesize` compresses the children + own mesh into the answer; citations
  union the provenance roots used.

---

## 6. OFFLINE ADAPTATION — SLEEP (shape `C_sym` only)  [FIXED scope]

Only place `C` and `S` change. **`StructureRecon` only** — no `SequenceRecon`, no `C_anti`.
Strengthen/weaken edges so gathers pull coherent, useful context: maximize overlap between gathered
meshes and held-out target context for known query→answer pairs. Constraints: `C_sym` support =
`E(S)`; trust region `‖ΔC‖≤δ`; slow EMA `τ_C`. (Deferrable — the bootstrap cosine `C_sym` is a
usable teacher; Sleep is an optimization, not a prerequisite.)

---

## 7. INVARIANTS (asserted in code / CI)  [FIXED]

1. **Convergence guaranteed:** pure gradient flow on coercive `E` ⇒ every gather settles
   (`‖ΔX‖→0`); no rotation term exists.
2. **Bounded:** trapping guard `max_i‖x_i‖ ≤ R_max` every step; no NaN/Inf.
3. `S` immutable at runtime; plastic only in Sleep.
4. **Gather locality:** mesh ⊆ `k_hop` reachable set of the seeds (cannot pull unreachable nodes).
5. **Provenance:** the tree is rooted at seeds and spans the mesh; every gathered node has a path
   back to a seed.
6. **Determinism:** `(seeds, anchor targets, C, cfg, rng-seed)` ⇒ identical mesh + tree.
7. **LLM boundary:** LLMs are called only at the four §4 seams; `gather` contains no LLM call and
   makes no decisions.

---

## 8. READOUT & PROVENANCE

`synthesize` consumes the mesh as: ranked concepts (by relevance) each rendered with its citation
chain (provenance path to a seed root). The answer's citations = the union of provenance roots the
synthesis used. No occupancy/itinerary (there is no time-trajectory readout — the gather converges
to one settled mesh).

---

## 9. RETIRED vs KEPT (migration from `architecture_spec.md`)

**Retired [FIXED]:** `R(X)`/`C_anti`, `ρ_R`/`ρ_max`, regime classification (point/cycle/torus/
chaos), Lyapunov/FFT/`regime_label`/`finite_time_lyapunov`/`_commensurate`/`motion_signal`,
`SequenceRecon`, the `(β,ρ_R)` phase diagram + sweep, itinerancy/occupancy readout, the
dissipative-Hamiltonian framing.

**Kept:** `S` + frozen anchors; `X_0=rownorm(P·A)`; couple+inhib+barrier energy; gradient settle +
trapping/barrier bound; `support_τ` (§14.2 of old spec); sustained-settle `rollout`; `RunLogger`
(`tracking.py`); the toy-graph harness (repurposed to gather/precision experiments).

**Changed/new:** `+` seed/parent **anchor potential** (§3.3); `+` **provenance tree** (§3.5); `+`
active-set materialization from seeds (§3.1); dynamics = pure gradient (`ρ_R≡0`); readout =
mesh+tree (not itinerary); coupling = `C_sym` only.

---

## 10. OPERATIONAL DEFINITIONS (frozen for implementation)

- **active set** `N` (§3.1): `k_hop=2`, `N_max=512`, neighbor cap by edge weight.
- **anchor potential** (§3.3): `(σ/2)Σ_{anchors}‖x_i−s_i‖²`, default `σ=1.0`.
- **settle** (§3.4): `‖ΔX‖<ε_x` (`1e-4`) sustained `H_hold` (`50`) steps, else `H_max` (`4000`).
- **mesh** (§3.5): `support_τ`, `τ=0.5`, optional `top_k_max`.
- **provenance tree** (§3.5): seed-rooted BFS layering + per-node max-`flow` parent.
- **breadth** `β` default `2.0`; **inherit** `K_inherit=4`; **recursion** `MAX_DEPTH=3`, budgets
  `MAX_NODES_TOTAL`, `MAX_LLM_CALLS`.

---

## 11. OPEN QUESTIONS (validation-time — flagged for build checkpoints)

- **Gather precision vs over-spread** on real graphs: tune `(β, σ, k_hop, τ)` so multi-hop relevant
  context is gathered but distant noise isn't. (Toys over-spread because cliques are uniform.)
- **Provenance algorithm:** flow-weighted spanning tree vs shortest-path vs activation-threshold —
  which gives the most faithful citation chains.
- **Physics value-add over personalized-PageRank:** the competition (inhib) + learned `C_sym` must
  earn their keep vs a plain diffusion baseline — to be measured, not assumed.
- **`sufficient` reliability:** can a cheap LLM reliably gate recursion depth, or is a fixed budget
  safer.

END SPEC
