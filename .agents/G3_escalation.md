# G3 ESCALATION — Is `R = C_anti·X` the right itinerancy mechanism?

**Trigger:** Gate G3 found NO cross-mesh itinerancy from bootstrap dynamics — bridge → point at
every (β,ρ_R) cell, ring → in-basin cycle only (see `runs_experiment/G3_gate_report.md`). Per
AGENTS.md §3 this escalates to the spec rather than proceeding to Phase 4. The empirical result is
**consistent with the spec's own theory**, which is what makes this a design question, not a bug.

---

## 1. The spec predicts the negative result

`architecture_spec.md §8`, verbatim:

> "a strict basin (`H≻0`) is a stable focus for *any* `ρ_R` (rotation cannot destabilize
> curvature); recurrence requires `H` to flatten (`λ_min(H)→0`) in a rotating plane."

My toy graphs have strict basins (deep clique wells + barrier curvature `~3μR_max²`), so they are
stable foci for all `ρ_R` → points. The (β,ρ_R) sweep never approached `λ_min(H)=0`, so the
recurrent regime was never entered. **The architecture is internally consistent; itinerancy simply
lives only on the bifurcation boundary, and the bootstrap geometry sits deep in the strict-basin
interior.** This is the empirical answer §13 said to "sweep for."

## 2. Four structural problems this exposes

**P1 — The target regime is the bifurcation boundary, which §10 tells us to avoid.**
§8: recurrence ⇔ `λ_min(H(β))=0 ∩ ω(ρ_R)≠0` (the Hopf/heteroclinic locus). §10 ("Bifurcation
fragility"): *"keep parameters off known thresholds; Sleep trust region forbids large jumps."* A
reasoning substrate that must sit **on** a measure-zero boundary while being instructed to stay
**off** it is self-contradictory. The recurrent regime *is* the fragile threshold.

**P2 — `R = C_anti·X` is both too weak and wrongly placed to drive robust itinerancy.**
- *Too weak:* §8 concedes rotation cannot destabilize `H≻0`; with `‖C_anti‖ ≤ ρ_R‖C_sym‖`
  (`C_sym` degree-normalized, O(1)) and barrier curvature large, `A` is negligible except exactly
  at `λ_min(H)=0`.
- *Wrongly placed:* the only **robust** itinerancy mechanism in the literature (winnerless
  competition / Lotka–Volterra heteroclinic networks, Rabinovich et al.) comes from **asymmetric
  competition** — inhibition `i⊣j ≠ j⊣i`, cyclic dominance — which builds a structurally stable
  heteroclinic channel WITHOUT fine-tuning curvature to marginal. Here the asymmetry is instead an
  additive divergence-free rotation, divorced from the inhibition term and explicitly forbidden
  (§7) from creating structure. The ingredient that robustly produces itinerancy is excluded by
  construction.
- *Honest counterpoint:* this was deliberate. Divergence-free `R` (`⟨X,R⟩≡0`) buys the clean
  `½‖X‖²` absorbing-set guarantee (§5, §11). Asymmetric competition forfeits it (needs an explicit
  radial bound, §5 caveat). **Real trade: guaranteed boundedness vs robust itinerancy. The spec
  chose boundedness; G3 shows the price is that itinerancy appears only on a fragile boundary.**

**P3 — Phase 4 as scoped cannot reach itinerancy; the dependency is mis-ordered.**
§7: `C_anti` *"does not create cycles or itineraries; it aligns rotational transport with the
existing saddle skeleton of E … can only route along ridges E already has."* With bootstrap /
StructureRecon `C_sym`, `E` has strict basins, **not** near-marginal saddle channels — so
SequenceRecon (which only shapes `C_anti`, §14.7) has no near-marginal ridge to align with and
cannot move the regime off a strict basin. **The itinerancy lever is `C_sym`/`β` (flatten `H` toward
marginal), not `C_anti`.** Proceeding to Phase 4 as written would optimize a loss that provably
cannot change the regime.

**P4 — `β` has a dual, conflicting role; the mesh/support definition degenerates near the boundary.**
§8 treats `β` as setting *support breadth* AND, via `λ_min(H(β))`, the *bifurcation*. My data: the
`β` that flattens basins drives support → the whole node set (consensus), so per-step supports stop
being distinct and the mesh abstraction (§14.3 Jaccard on node-sets) collapses to one mesh. There
may be **no `β` that both flattens basins to marginal and keeps supports sparse/distinct** — the two
roles §8 calls "orthogonal" are entangled exactly where itinerancy would live. §14.2/§14.3
(τ-threshold support + Jaccard mesh) presume sparse localized supports that the recurrent regime
may not provide.

## 3. Decision: resolve it with a cheap falsification, not by opinion or by building Phase 4

The genuinely open question (§13) is: **for a real graph, does ANY `(C_sym, β, ρ_R)` admit a
stable heteroclinic channel between two meshes?** Settle it empirically before any spec rewrite:

> **Bridge bifurcation probe (no new architecture):** on the 2-clique bridge, numerically search
> `C_sym`/`β` for a near-marginal direction (`λ_min(H)→0` at the inter-clique saddle), then test
> whether `ρ_R>0` opens a *transversely stable* cross-mesh channel (occupancy on ≥2 meshes,
> reproducible, bounded). This is the §13 bifurcation-continuation work scoped to one toy — a few
> dozen lines over existing `energy`/`coupling`, ~an afternoon.

Outcomes:
- **Channel exists** → itinerancy is reachable; the fix is **re-scoping, not redesign**: make
  StructureRecon/`β` (not SequenceRecon) the primary itinerancy lever (revise §7 emphasis +
  §8 "orthogonal" claim), redefine mesh identity to survive broad supports (revise §14.3, e.g. mesh
  = leading Gram eigenvector of the support, not node-set Jaccard), then run Phase 4 jointly on
  `C_sym`+`C_anti`.
- **No channel for any parameter** → `R=C_anti·X` cannot do robust itinerancy on real geometry.
  Then choose: **(A)** adopt asymmetric-competition dynamics (move skew into the inhibition; accept
  the radial-bound cost, revise §4/§5/§11), or **(B)** narrow scope honestly — validate Option A
  (`ρ_R=0`, point attractors, support readout — which *works*) and demote "reasoning-as-itinerary"
  to an open research conjecture (revise §0/§8/§11 scope, not the invariants).

**Recommendation:** run the bridge bifurcation probe first. It converts the escalation from an
architecture debate into a one-experiment falsification, and it is the §13 work the spec already
flagged as the open validation question. Do **not** build `sleep.py` until it returns.

---

## 4. PROBE RESULT (2026-06-02) — NO CHANNEL FOR ANY PARAMETER

Ran the probe over ring(3×2) + bridge(3+3), grid `C_sym`-gain `g∈{0.5…7}` × `β∈{0.5…4}` ×
`ρ_R∈{0…0.99}`, with the Hessian/Jacobian linear diagnostic and real rollouts. Decisive:

1. **Field-of-values theorem confirmed tightly:** `maxRe(λ(J)) ≈ −λ_min(H)` in every cell, and the
   max-real-part eigenvalue has **`Im=0`** — it is the real **O(d) gauge-symmetry mode**, not a
   complex Hopf pair. **No Hopf bifurcation is ever crossed**, so no limit cycle is *born* from a
   basin; the only oscillation seen (ring, low g) is motion along the pre-existing marginal gauge
   direction (in-basin, single mesh).
2. **λ_min(H) ≈ 0⁻ everywhere** (the gauge symmetry makes every basin marginal along its orbit).
   Driving `g` up to force `λ_min(H)<0` (−0.05…−0.12) does **not** create a channel — it just
   relocates to another **whole-graph point attractor**.
3. **Support = whole node set (~6/6) in EVERY cell**, at every β/g/ρ_R. The LSE inhibition never
   produces winner-take-all localization, so there is only ever **one mesh** ⇒ cross-mesh
   itinerancy is structurally impossible regardless of the rotation.
4. **Zero cross-mesh switching in any cell** (late_meshes=1, switches=0), all trajectories bounded
   (dissipativity intact).

**Conclusion (falsified):** within the §4 energy + divergence-free `R=C_anti·X` class, robust
cross-mesh heteroclinic itinerancy does not occur for any `(C_sym, β, ρ_R)` — foreclosed by the
field-of-values bound (strict basins are stable for all ρ_R; the only non-point escape is a
fine-tuned marginal *minimum* → an **in-basin** Hopf cycle, never a saddle **network**) and
confirmed behaviorally. Two independent structural blockers: (i) the theorem, and (ii) the
inhibition never localizes support into competing meshes.

### 4a. AIRTIGHT CONFIRMATION (attacked both confounds)

- **ρ_R cap is NOT the limiter:** cranked ρ_R to **10× past ρ_max** (renormalizing ‖C_anti‖=ρ‖C_sym‖
  directly) — still no cross-mesh switching, still bounded. The rotation magnitude is not the issue.
- **Localization/WTA is NOT rescuable, and the reason is deeper:** hand-placing the state localized
  in one clique, at every ρ_R and at β=8, it **spreads to the whole graph** and never hands off. The
  ρ_R=0 control proves localized states aren't even equilibria. **The couple term `−½tr(XᵀC_sym X)`
  is a spreading force whose minimum on a connected graph is GLOBAL co-activation** — the LSE
  inhibition does not counter it. So there are **no competing localized meshes** for itinerancy to
  move between, independent of the rotation.

**Two independent structural failures, both confirmed:** (i) field-of-values theorem forecloses
rotation from destabilizing basins; (ii) the energy has no localized competing attractors (global
co-activation minimum). And note the irony: the property that gives *free* boundedness (`⟨X,R⟩≡0`)
is exactly what prevents `R` from doing the radial work needed to drive transitions — **boundedness
and itinerancy are in tension by construction.**

→ This is the memo's "**no channel anywhere**" branch. Decision now between **(A)** asymmetric-
competition dynamics (move skew into the inhibition → structurally-stable winnerless-competition
heteroclinic network; forfeits the `½‖X‖²` boundedness guarantee, revise §4/§5/§11) and **(B)**
narrow scope honestly — validate Option A (`ρ_R=0`, point attractors, support readout — which
*works* and is bounded) and demote "reasoning-as-itinerary" to an open conjecture (revise §0/§8/§11
scope; invariants §9 unchanged). The `ρ_R>0` rotation, as specified, buys nothing for itinerancy.
