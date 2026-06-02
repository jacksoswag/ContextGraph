# Sleep Evaluation (G6 + 📍3 verdict)

Store `.di-ui/graph.sleep.merged.sqlite`. 10 seeds, 2624 learned edge multipliers (trust 0.5).

| seed | gather boot | gather learned | PPR boot | PPR learned | recall boot | recall learned |
|---|---|---|---|---|---|---|
| dog | 0.219 | 0.219 | 0.247 | 0.247 | None | None |
| france | 0.345 | 0.335 | 0.583 | 0.552 | 1.0 | None |
| water | 0.275 | 0.275 | 0.292 | 0.292 | None | None |
| music | 0.292 | 0.299 | 0.379 | 0.379 | None | None |
| gravity | 0.297 | 0.330 | 0.283 | 0.283 | None | None |
| computer | 0.306 | 0.306 | 0.255 | 0.255 | None | None |
| ocean | 0.336 | 0.335 | 0.309 | 0.309 | None | None |
| tree | 0.210 | 0.215 | 0.236 | 0.208 | None | 0.0 |
| money | 0.220 | 0.220 | 0.220 | 0.220 | None | None |
| language | 0.344 | 0.363 | 0.457 | 0.467 | 1.0 | 0.5 |

**G6 — does Sleep help the gather?** semantic precision@10 0.284 (bootstrap) → 0.290 (learned); held-out recall 1.000 → 0.250.
**📍3 — field vs PPR on learned weights:** gather(learned) 0.290 vs PPR(learned) 0.321.
