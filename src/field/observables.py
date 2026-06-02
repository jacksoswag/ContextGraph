from __future__ import annotations
import numpy as np

# support: τ-thresholded high-energy node set; X is [N,d] (instant) or [W,N,d] (windowed)
def support(X, tau: float) -> frozenset[int]:
    e = (X * X).sum(-1)
    if e.ndim > 1: e = e.mean(0)
    thr = tau * float(e.max())
    return frozenset(i for i in range(e.shape[0]) if float(e[i]) >= thr)

# cluster_mesh: greedy online Jaccard clustering of per-step support sets → mesh ids
def cluster_mesh(supports: list[frozenset[int]], d_mesh: float) -> list[int]:
    reps: list[frozenset[int]] = []; ids: list[int] = []
    for s in supports:
        best, best_d = -1, 1.0
        for m, r in enumerate(reps):
            uni = len(s | r)
            jd = 1.0 - (len(s & r) / uni if uni else 1.0)
            if jd < best_d: best, best_d = m, jd
        if best >= 0 and best_d < d_mesh: ids.append(best)
        else: reps.append(s); ids.append(len(reps) - 1)
    return ids

# occupancy: normalized time-fraction per mesh id
def occupancy(mesh_ids: list[int]) -> dict[int, float]:
    n = len(mesh_ids)
    if not n: return {}
    out: dict[int, float] = {}
    for m in mesh_ids: out[m] = out.get(m, 0.0) + 1.0 / n
    return out

def jaccard_dist(a: frozenset[int], b: frozenset[int]) -> float:
    """Jaccard distance 1 - |A∩B|/|A∪B|; two empty sets → 0."""
    if not a and not b: return 0.0
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0

def tv_distance(q1: dict[int, float], q2: dict[int, float]) -> float:
    """Total variation ½ Σ|q1(k)-q2(k)| between two occupancy distributions."""
    keys = set(q1) | set(q2)
    return 0.5 * sum(abs(q1.get(k, 0.0) - q2.get(k, 0.0)) for k in keys)


class MeshCatalog:
    """Online greedy Jaccard-clustering of per-step support sets; x* = running-mean X."""

    def __init__(self, d_mesh: float) -> None:
        self._d = d_mesh
        self._reps: list[frozenset[int]] = []
        self._X_sum: list[np.ndarray] = []
        self._X_cnt: list[int] = []

    def assign(self, supp: frozenset[int], X: np.ndarray) -> int:
        """Return mesh_id of nearest mesh (Jaccard < d_mesh) or open a new one."""
        best_id, best_dist = -1, 1.0
        for mid, rep in enumerate(self._reps):
            d = jaccard_dist(supp, rep)
            if d < best_dist: best_id, best_dist = mid, d
        if best_dist < self._d:
            self._X_sum[best_id] = self._X_sum[best_id] + X
            self._X_cnt[best_id] += 1
            return best_id
        self._reps.append(supp)
        self._X_sum.append(np.array(X, dtype=float))
        self._X_cnt.append(1)
        return len(self._reps) - 1

    @property
    def mesh_count(self) -> int: return len(self._reps)

    def representative(self, mesh_id: int) -> np.ndarray:
        """Running-mean X state for mesh_id (x* per §3.5)."""
        return self._X_sum[mesh_id] / self._X_cnt[mesh_id]


class StabilizationMonitor:
    """TV(occupancy-window) convergence + ‖ΔX‖ point-test; accumulates mesh_ids."""

    def __init__(self, W: int, H_hold: int, eps_occ: float, eps_x: float) -> None:
        self._W, self._H_hold = W, H_hold
        self._eps_occ, self._eps_x = eps_occ, eps_x
        self.mesh_ids: list[int] = []
        self._stable_count = 0
        self.stabilized_at: int | None = None

    def update(self, mesh_id: int, delta_X_norm: float) -> bool:
        """Record one step; return True when stabilized (point or occupancy-TV test)."""
        step_idx = len(self.mesh_ids)
        self.mesh_ids.append(mesh_id)
        if delta_X_norm < self._eps_x:
            self.stabilized_at = step_idx; return True
        n = len(self.mesh_ids)
        if n < 2 * self._W:
            self._stable_count = 0; return False
        w_cur = self.mesh_ids[-self._W:]
        w_prev = self.mesh_ids[-2 * self._W:-self._W]
        tv = tv_distance(occupancy(w_cur), occupancy(w_prev))
        if tv < self._eps_occ: self._stable_count += 1
        else: self._stable_count = 0
        if self._stable_count >= self._H_hold:
            self.stabilized_at = step_idx; return True
        return False
