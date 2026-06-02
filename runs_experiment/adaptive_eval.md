# Adaptive-domain measurement

Store `.di-ui/graph.sleep.merged.sqlite`. 6 seeds. Fixed gather = current default (λ=1.5, N_max=512). Adaptive = moving front (max_live=200, ttl=3).

**Fixed gather:** mean mesh 7 nodes, precision 0.344.

## Adaptive front vs decay λ (mean over seeds)

| λ | committed | peak_live | reach | loaded | precision | prec@top30 | time |
|---|---|---|---|---|---|---|---|
| 0.3 | 537 | 200 | 7.0 | 1111 | 0.232 | 0.263 | 13.0s |
| 0.5 | 472 | 200 | 6.0 | 1111 | 0.239 | 0.268 | 10.0s |
| 0.7 | 310 | 136 | 4.2 | 724 | 0.286 | 0.300 | 6.8s |
| 1.0 | 14 | 60 | 0.8 | 172 | 0.247 | 0.256 | 1.6s |
| 1.5 | 1 | 60 | 0.0 | 172 | 0.000 | 0.000 | 0.6s |

## Per-phase trace (gravity, λ=0.7) — front advancing under a bounded live window

| phase | live | committed | loaded | culled | reach |
|---|---|---|---|---|---|
| 0 | 24 | 10 | 0 | 1 | 1 |
| 1 | 23 | 11 | 80 | 0 | 1 |
| 2 | 103 | 28 | 80 | 37 | 2 |
| 3 | 146 | 56 | 80 | 60 | 2 |
| 4 | 166 | 106 | 80 | 30 | 3 |
| 5 | 200 | 147 | 80 | 27 | 4 |
| 6 | 200 | 169 | 80 | 61 | 4 |
| 7 | 200 | 207 | 80 | 71 | 4 |
| 8 | 200 | 236 | 80 | 95 | 4 |
| 9 | 185 | 271 | 80 | 57 | 5 |
| 10 | 200 | 299 | 80 | 84 | 5 |
| 11 | 196 | 335 | 80 | 73 | 5 |
| 12 | 200 | 357 | 80 | 109 | 5 |
| 13 | 171 | 399 | 80 | 43 | 6 |