# Gather Precision Calibration (Phase 3 / 📍3)

Store `.di-ui/graph.sleep.merged.sqlite`. 15 seeds, k_hop=2, N_max=300, top-k=10.
Semantic precision@k = mean cosine(top-k anchor, seed anchor). Held-out recall@k = fraction of 30%-hidden direct neighbors recovered in top-k.

## 1. Decay (λ) sweep — gather

| λ | settle steps | mesh(top≥1%) | sem.precision@10 | heldout recall@10 |
|---|---|---|---|---|
| 0.5 | 1926 | 136.1 | 0.293 | 0.644 |
| 1.0 | 5857 | 48.7 | 0.351 | 0.188 |
| 1.5 | 513 | 7.0 | 0.329 | 0.000 |
| 2.0 | 322 | 3.7 | 0.324 | 0.000 |
| 3.0 | 211 | 2.0 | 0.319 | 0.000 |

## 2. Teleport (α) sweep — personalized PageRank

| α | sem.precision@10 | heldout recall@10 |
|---|---|---|
| 0.1 | 0.338 | 0.143 |
| 0.15 | 0.338 | 0.000 |
| 0.3 | 0.334 | 0.000 |
| 0.5 | 0.324 | 0.000 |

## 3. Head-to-head — gather(λ=1.0) vs PPR(α=0.1)

| seed | gather prec | PPR prec | gather rec | PPR rec | top-10 overlap |
|---|---|---|---|---|---|
| dog | 0.193 | 0.198 | None | None | 0.25 |
| cat | 0.226 | 0.225 | 1.0 | None | 0.82 |
| france | 0.600 | 0.583 | 1.0 | None | 0.82 |
| paris | 0.368 | 0.360 | None | None | 0.43 |
| water | 0.332 | 0.316 | 0.0 | 0.0 | 0.54 |
| fire | 0.608 | 0.585 | 0.0 | 0.0 | 0.54 |
| music | 0.415 | 0.385 | None | 1.0 | 0.82 |
| gravity | 0.385 | 0.336 | 0.0 | 0.0 | 0.67 |
| democracy | 0.549 | 0.565 | 0.33 | 0.0 | 0.43 |
| computer | 0.230 | 0.234 | None | None | 0.43 |
| ocean | 0.335 | 0.329 | None | None | 0.67 |
| tree | 0.211 | 0.209 | 0.0 | None | 0.82 |
| car | 0.000 | 0.000 | None | None | 0.00 |
| money | 0.292 | 0.266 | None | None | 0.43 |
| language | 0.522 | 0.486 | 0.0 | 0.0 | 0.67 |

**Means:** gather precision 0.351 vs PPR 0.338; gather recall 0.292 vs PPR 0.167; mean top-10 overlap 0.55.
