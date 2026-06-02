# GATE G3 — Does itinerancy emerge?  (build_roadmap.md §Phase 3)

**Date:** 2026-06-02  ·  **Config:** bootstrap coupling, generous horizon (H_max=15000, sustained-settle),
(β,ρ_R) swept.  ·  **Verdict: REVISE/ESCALATE — itinerancy does NOT emerge from the bootstrap dynamics.**

## 1. Measurement flaws found and fixed first

The pre-G3 "torus everywhere / M3 PASS" report was an artifact of three measurement bugs, now fixed:

1. **Rollout premature break** (`dynamics.py`): stopped on a *single* sub-ε_x step → could truncate a
   slow saddle-dwell and fake "settled." Now requires ΔX<ε_x for `H_hold` *consecutive* steps.
2. **Classifier read E(X_t)** — but the rotation `R=C_anti X` is norm-preserving (⟨X,R⟩≡0) so it moves
   along **near-iso-energy orbits**. Measured: a ring limit cycle has state-projection amplitude ~6
   while `E` rel-rms ~1e-5. **E is structurally blind to rotation-driven motion.** Classifier now reads a
   state-projection (top PC), gated point-vs-moving by sustained ΔX, with an oscillation-amplitude floor
   (`eps_osc`, set at ~5× the empirical ρ_R=0 drift/round-off floor).
3. **Spectral leakage → false torus**: FFT sidebands of one oscillator were read as multiple
   incommensurate fundamentals. Now collapsed via `_distinct_fundamentals` before commensurability.

Post-fix the classifier matches ground truth on all toy graphs; 425 field tests pass (stress grid over
β×ρ_R×μ×seed: trapping, finiteness, determinism, Lyapunov-monotonicity all hold).

## 2. Actual behavior (real runs, not synthetic)

| graph | ρ_R sweep | observed | settles? | osc/R_max | recurrent cross-mesh switches |
|---|---|---|---|---|---|
| single_clique | 0, .5, .9 | **point** | yes | — | 0 |
| **two_cliques_bridge** | 0 … .99 (β∈{1,2,4}) | **point (all cells)** | yes | ≤3.9e-4 | **0** |
| ring_of_cliques | 0, .5 | point/drift | no | <1e-4 | 0 |
| ring_of_cliques | .9 | **cycle** (in-basin) | no | 7.9e-4 | **0** |
| two_incomm_rings | 0, .5, .9 | **point** | yes | <4e-4 | 0 |

- **The bridge collapses to a single fixed-point basin at every (β,ρ_R) cell** — no recurrent itinerancy.
- The only sustained motion anywhere is the ring's **in-basin limit cycle** at strong ρ_R: a sub-percent
  wobble with **constant support** (one mesh) — explicitly NOT cross-mesh itinerancy.
- No graph, at any tested parameter, shows recurrent cross-mesh switching. **M3 FAIL.**

## 3. Diagnosis: structural, not parametric

The (β,ρ_R) sweep is uniformly "point" on the bridge → not a tuning miss. Two structural causes:

- **Dissipation dominates the bootstrap rotation.** `C_anti` is initialized from raw edge direction
  (±1, rescaled to ‖C_anti‖≤ρ_R‖C_sym‖). It has no reason to align with the inter-basin saddle, so it
  cannot carve a heteroclinic channel across the bridge; the gradient skeleton just spirals into one basin.
- **Inhibition (β) does not localize support.** Settled support is near the full node set, so "meshes"
  are not sparse/distinct — cross-mesh itinerancy is ill-posed when every node is always active.

## 4. Was "Phase 4 needed before judging Phase 3" right? — partly, but mostly a symptom of the flaws

- **Phase 3 question (answerable now, answered):** does the *mechanism* with bootstrap coupling produce
  ordered itinerancy? **No.** That is a real, clean result and the necessary input to Phase 4.
- **Kernel of truth:** bootstrap `C_anti` is untrained; `SequenceRecon` (Phase 4) is specifically designed
  to align `C_anti` with `x*_j−x*_i` across the saddle — so trained coupling *might* carve the channel the
  bootstrap can't. Whether the architecture *can* itinerate is genuinely a Phase-4 question.
- **But:** building Phase 4 first would have trained against a broken `T̂` (mislabeled, non-settling
  trajectories). The flaws made Phase 3 look uninterpretable, which invited "skip ahead." The correct order
  is the roadmap's: fix the measurement, record G3, then Phase 4 — now unblocked with a trustworthy `T̂`.

## 5. Decision required (HUMAN)

- (a) **PROCEED to Phase 4** treating G3 as: "bootstrap mechanism does not itinerate; test whether trained
  `C_anti` (SequenceRecon) can induce a heteroclinic channel on the bridge." Recommended.
- (b) **REVISE dynamics/inhibition first** (e.g. sharper β / sparsity so meshes localize; semi-implicit
  barrier) and re-run G3 before Phase 4.
- (c) **ESCALATE to spec** — reconsider whether `R=C_anti X` global linear rotation is the right itinerancy
  mechanism at all.
