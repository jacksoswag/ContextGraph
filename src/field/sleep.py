from __future__ import annotations
# SLEEP — offline StructureRecon (spec §6). The ONLY place C_sym / S change. Learns a per-edge
# multiplier on the bootstrap cosine coupling so gathers pull more coherent context: a delta-rule
# that strengthens edges into under-gathered target neighbors and weakens edges into over-gathered
# noise. Constraints: support = E(S) (multiplies existing edges only, never creates them); per-edge
# trust region |m−1| ≤ δ; slow EMA τ_C. No SequenceRecon, no C_anti.
import json, random
from pathlib import Path
import torch
from .config import FieldConfig
from .gather import materialize, gather

# EdgeWeights: persistent undirected per-edge multiplier over S's edges (default 1.0 = bootstrap).
class EdgeWeights:
    def __init__(self, default: float = 1.0) -> None:
        self._m: dict[tuple[str, str], float] = {}
        self.default = default
    @staticmethod
    def _key(u: str, v: str) -> tuple[str, str]: return (u, v) if u <= v else (v, u)
    def get(self, u: str, v: str) -> float: return self._m.get(self._key(u, v), self.default)
    def set(self, u: str, v: str, val: float) -> None: self._m[self._key(u, v)] = val
    def __len__(self) -> int: return len(self._m)
    def items(self): return self._m.items()
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({f"{u}\t{v}": m for (u, v), m in self._m.items()}))
    @classmethod
    def load(cls, path: str | Path) -> "EdgeWeights":
        w = cls()
        for k, m in json.loads(Path(path).read_text()).items():
            u, v = k.split("\t"); w._m[(u, v)] = float(m)
        return w

# _targets: the "relevant context" for a seed = its direct graph neighbors (self-supervised, no
# labels). y=1 on targets, 0 on gathered noise; seeds are excluded (always hot).
def _targets(ep, seed_idx: list[int]) -> torch.Tensor:
    s, d = ep.edge_index.tolist()
    seeds = set(seed_idx)
    nbr = {b for a, b in zip(s, d) if a in seeds} | {a for a, b in zip(s, d) if b in seeds}
    y = torch.zeros(len(ep.node_ids))
    for i in nbr - seeds: y[i] = 1.0
    return y

# train: StructureRecon over a set of seeds. For each seed, gather under the current weights, then
# nudge active-set edge multipliers by the delta-rule toward pulling the seed's true neighbors.
def train(store, seed_ids: list[str], cfg: FieldConfig, *, lr: float = 0.15, epochs: int = 3,
          trust: float = 0.5, tau_C: float = 0.0, weights: EdgeWeights | None = None,
          rng_seed: int = 0) -> EdgeWeights:
    w = weights or EdgeWeights()
    for _ep in range(epochs):
        for sid in seed_ids:
            ep, seed_idx = materialize(store, [sid], cfg)
            if ep.edge_index.numel() == 0: continue
            res = gather(ep, seed_idx, cfg, weights=w)
            rel = res.relevance(); mx = float(rel.max())
            if mx <= 0: continue
            rhat = rel / mx                              # predicted relevance ∈ [0,1]
            err = _targets(ep, seed_idx) - rhat         # delta-rule error (target − predicted)
            s, d = ep.edge_index
            upd = lr * 0.5 * (err[d] * rhat[s] + err[s] * rhat[d])   # [E] symmetric
            nid = ep.node_ids
            cur = torch.tensor([w.get(nid[a], nid[b]) for a, b in zip(s.tolist(), d.tolist())])
            new = (cur + upd).clamp(1.0 - trust, 1.0 + trust)        # trust region |m−1| ≤ trust
            if tau_C > 0.0: new = tau_C * cur + (1.0 - tau_C) * new  # slow EMA
            for k, (a, b) in enumerate(zip(s.tolist(), d.tolist())):
                w.set(nid[a], nid[b], float(new[k]))
    return w
