# Graph-Field Experiment Report

> `harness.py` — implementation_spec.md §4 (M3 itinerancy target). Honest readout:
> regime, genuine settling, oscillation amplitude (÷R_max), and recurrent cross-mesh
> switching in the steady state. `late_switches` is the real itinerancy signal.

## Regime Summary

| graph | config | predicted | observed | settled@T | osc_amp/R | late_switches | distinct_late |
|---|---|---|---|---|---|---|---|
| single_clique | rho0 | point | **point** | yes@1426 | 8.04e-04 | 0 | 1 |
| single_clique | mod | point | **point** | yes@1544 | 7.49e-04 | 0 | 1 |
| single_clique | high | point | **point** | yes@2733 | 2.76e-04 | 0 | 1 |
| two_cliques_bridge | rho0 | itinerary | **point** | yes@9485 | 3.62e-05 | 0 | 1 |
| two_cliques_bridge | mod | itinerary | **point** | yes@5524 | 9.99e-05 | 0 | 1 |
| two_cliques_bridge | high | itinerary | **point** | yes@2950 | 2.85e-04 | 0 | 1 |
| ring_of_cliques | rho0 | cycle/itinerary | **point** | no@15001 | 1.72e-05 | 0 | 1 |
| ring_of_cliques | mod | cycle/itinerary | **point** | no@15001 | 7.95e-05 | 0 | 1 |
| ring_of_cliques | high | cycle/itinerary | **cycle** | no@15001 | 7.95e-04 | 0 | 1 |
| two_incomm_rings | rho0 | torus | **point** | yes@14213 | 2.26e-05 | 0 | 1 |
| two_incomm_rings | mod | torus | **point** | no@15001 | 3.43e-05 | 0 | 1 |
| two_incomm_rings | high | torus | **point** | yes@14096 | 3.81e-05 | 0 | 1 |

## single_clique

### rho0

- Regime: **point**  (predicted: point)
- Settled to fixed point: yes (T=1426 / H_max)
- Oscillation amplitude ÷R_max: 8.04e-04  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0019
- Frequencies: []

![trajectory.png](20260602T094140_4d312dda40fa46dd_db047d/plots/trajectory.png)

![occupancy.png](20260602T094140_4d312dda40fa46dd_db047d/plots/occupancy.png)

### mod

- Regime: **point**  (predicted: point)
- Settled to fixed point: yes (T=1544 / H_max)
- Oscillation amplitude ÷R_max: 7.49e-04  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0013
- Frequencies: []

![trajectory.png](20260602T094141_de36ecc9c63090e3_cb52b9/plots/trajectory.png)

![occupancy.png](20260602T094141_de36ecc9c63090e3_cb52b9/plots/occupancy.png)

### high

- Regime: **point**  (predicted: point)
- Settled to fixed point: yes (T=2733 / H_max)
- Oscillation amplitude ÷R_max: 2.76e-04  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [1]
- λ_max (Lyapunov): 0.0000
- Frequencies: []

![trajectory.png](20260602T094142_d5b7511aedd6121b_55bb77/plots/trajectory.png)

![occupancy.png](20260602T094142_d5b7511aedd6121b_55bb77/plots/occupancy.png)

## two_cliques_bridge

### rho0

- Regime: **point**  (predicted: itinerary)
- Settled to fixed point: yes (T=9485 / H_max)
- Oscillation amplitude ÷R_max: 3.62e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0014
- Frequencies: []

![trajectory.png](20260602T094144_4d312dda40fa46dd_c1af04/plots/trajectory.png)

![occupancy.png](20260602T094144_4d312dda40fa46dd_c1af04/plots/occupancy.png)

### mod

- Regime: **point**  (predicted: itinerary)
- Settled to fixed point: yes (T=5524 / H_max)
- Oscillation amplitude ÷R_max: 9.99e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0012
- Frequencies: []

![trajectory.png](20260602T094149_de36ecc9c63090e3_1923f1/plots/trajectory.png)

![occupancy.png](20260602T094149_de36ecc9c63090e3_1923f1/plots/occupancy.png)

### high

- Regime: **point**  (predicted: itinerary)
- Settled to fixed point: yes (T=2950 / H_max)
- Oscillation amplitude ÷R_max: 2.85e-04  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [1]
- λ_max (Lyapunov): -0.0004
- Frequencies: []

![trajectory.png](20260602T094152_d5b7511aedd6121b_e696a4/plots/trajectory.png)

![occupancy.png](20260602T094152_d5b7511aedd6121b_e696a4/plots/occupancy.png)

## ring_of_cliques

### rho0

- Regime: **point**  (predicted: cycle/itinerary)
- Settled to fixed point: no (T=15001 / H_max)
- Oscillation amplitude ÷R_max: 1.72e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0011
- Frequencies: []

![trajectory.png](20260602T094154_4d312dda40fa46dd_29a7cf/plots/trajectory.png)

![occupancy.png](20260602T094154_4d312dda40fa46dd_29a7cf/plots/occupancy.png)

### mod

- Regime: **point**  (predicted: cycle/itinerary)
- Settled to fixed point: no (T=15001 / H_max)
- Oscillation amplitude ÷R_max: 7.95e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0011
- Frequencies: []

![trajectory.png](20260602T094202_de36ecc9c63090e3_2fd1cc/plots/trajectory.png)

![occupancy.png](20260602T094202_de36ecc9c63090e3_2fd1cc/plots/occupancy.png)

### high

- Regime: **cycle**  (predicted: cycle/itinerary)
- Settled to fixed point: no (T=15001 / H_max)
- Oscillation amplitude ÷R_max: 7.95e-04  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0012
- Frequencies: [34.0]

![trajectory.png](20260602T094209_d5b7511aedd6121b_1ea7c5/plots/trajectory.png)

![occupancy.png](20260602T094209_d5b7511aedd6121b_1ea7c5/plots/occupancy.png)

## two_incomm_rings

### rho0

- Regime: **point**  (predicted: torus)
- Settled to fixed point: yes (T=14213 / H_max)
- Oscillation amplitude ÷R_max: 2.26e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0015
- Frequencies: []

![trajectory.png](20260602T094217_4d312dda40fa46dd_7572f6/plots/trajectory.png)

![occupancy.png](20260602T094217_4d312dda40fa46dd_7572f6/plots/occupancy.png)

### mod

- Regime: **point**  (predicted: torus)
- Settled to fixed point: no (T=15001 / H_max)
- Oscillation amplitude ÷R_max: 3.43e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0012
- Frequencies: []

![trajectory.png](20260602T094226_de36ecc9c63090e3_2653be/plots/trajectory.png)

![occupancy.png](20260602T094226_de36ecc9c63090e3_2653be/plots/occupancy.png)

### high

- Regime: **point**  (predicted: torus)
- Settled to fixed point: yes (T=14096 / H_max)
- Oscillation amplitude ÷R_max: 3.81e-05  (floor 1e-04)
- Recurrent cross-mesh switches (late half): 0 over 1 mesh(es)
- Itinerary (incl. transient): [0]
- λ_max (Lyapunov): -0.0002
- Frequencies: []

![trajectory.png](20260602T094234_d5b7511aedd6121b_3062fa/plots/trajectory.png)

![occupancy.png](20260602T094234_d5b7511aedd6121b_3062fa/plots/occupancy.png)

## M3 Acceptance Assessment

Criterion: two_cliques_bridge shows **recurrent** cross-mesh itinerancy (late_switches>0 over ≥2 meshes) at some ρ_R.

- **M3 FAIL** — no recurrent cross-mesh itinerancy at any tested ρ_R.
  Bridge collapses to a single basin: rho0→point(settled=True), mod→point(settled=True), high→point(settled=True).
- Reported as-is — 'Do NOT tune to force a result.' (prompt 3.A)

## (β, ρ_R) Sweep — bridge

| β \ ρ_R | 0.0 | 0.3 | 0.6 | 0.9 | 0.99 |
|---|---|---|---|---|---|
| 1.0 | point | point | point | point | point |
| 2.0 | point | point | point | point | point |
| 4.0 | point | point | point | point | point |

`*` = recurrent cross-mesh itinerancy present.
