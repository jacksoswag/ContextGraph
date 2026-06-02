# Adaptive-domain measurement

Store `.di-ui/graph.sleep.merged.sqlite`. 6 seeds. Fixed gather = current default (λ=1.5, N_max=512). Adaptive = moving front (max_live=200, ttl=3).

**Fixed gather:** mean mesh 7 nodes, precision 0.344.

## Adaptive front vs decay λ (mean over seeds)

| λ | committed | peak_live | reach | loaded | precision | prec@top30 | time |
|---|---|---|---|---|---|---|---|
| 0.3 | 446 | 200 | 5.8 | 817 | 0.219 | 0.253 | 10.6s |
| 0.5 | 502 | 200 | 7.0 | 971 | 0.219 | 0.274 | 10.6s |
| 0.7 | 336 | 200 | 6.3 | 915 | 0.225 | 0.267 | 9.4s |
| 1.0 | 162 | 186 | 5.7 | 878 | 0.275 | 0.283 | 11.2s |
| 1.5 | 7 | 86 | 1.0 | 284 | 0.389 | 0.389 | 1.3s |

## Per-phase trace (gravity, λ=0.7) — front advancing under a bounded live window

| phase | live | committed | loaded | culled | reach |
|---|---|---|---|---|---|
| 0 | 24 | 11 | 80 | 1 | 1 |
| 1 | 103 | 50 | 80 | 2 | 2 |
| 2 | 181 | 81 | 80 | 5 | 2 |
| 3 | 200 | 92 | 80 | 16 | 2 |
| 4 | 200 | 106 | 80 | 138 | 3 |
| 5 | 142 | 119 | 80 | 88 | 3 |
| 6 | 134 | 137 | 80 | 34 | 3 |
| 7 | 180 | 145 | 79 | 116 | 4 |
| 8 | 143 | 172 | 80 | 73 | 4 |
| 9 | 150 | 201 | 80 | 46 | 5 |
| 10 | 184 | 227 | 80 | 42 | 5 |
| 11 | 200 | 241 | 28 | 102 | 5 |
| 12 | 126 | 246 | 80 | 80 | 6 |
| 13 | 126 | 284 | 80 | 31 | 7 |