from __future__ import annotations

# support: τ-thresholded high-energy node set; X is [N,d] (instant snapshot only — windowed
# variant retired with rotation-era StabilizationMonitor at finetune pass).
def support(X, tau: float) -> frozenset[int]:
    e = (X * X).sum(-1)
    thr = tau * float(e.max())
    return frozenset(i for i in range(e.shape[0]) if float(e[i]) >= thr)
