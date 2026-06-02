from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import math, numpy as np, torch
import torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .episode import Episode
from .coupling import Coupling
from .energy import Anchor
from .dynamics import rollout
from .harness import safe_build

EMBED_DIM = 384
_FALLBACK_ANCHOR = (np.ones(EMBED_DIM, dtype=np.float32) / math.sqrt(EMBED_DIM))

# any object exposing neighbors/anchor/text satisfies the gather (GraphStore or a test double)
class _StoreProto:
    def neighbors(self, node_id: str) -> list[tuple[str, float]]: ...
    def anchor(self, node_id: str): ...

@dataclass
class GatherResult:
    X_star: Tensor       # [N,d] settled state X*
    X_hist: Tensor       # [T,N,d] trajectory
    E_hist: Tensor       # [T] energy trace
    ep: Episode
    C: Coupling
    cfg: FieldConfig
    seed_idx: list[int]
    steps: int           # settle step count (T-1); < H_max ⇒ converged early
    def relevance(self) -> Tensor:           # r_i = ‖x*_i‖²  (spec §3.5)
        return (self.X_star * self.X_star).sum(-1)

# materialize: active set N = seeds ∪ k-hop neighborhood (§3.1), capped to N_max by descending
# connecting edge-weight. Returns the Episode (seeds ordered first) + seed indices within it.
def materialize(store: _StoreProto, seed_ids: list[str], cfg: FieldConfig) -> tuple[Episode, list[int]]:
    seeds = [s for s in dict.fromkeys(seed_ids)]            # dedup, keep order
    weight: dict[str, float] = {s: math.inf for s in seeds}  # seeds pinned (never capped out)
    frontier = deque(seeds)
    seen = set(seeds)
    for _hop in range(cfg.k_hop):
        nxt: list[str] = []
        for _ in range(len(frontier)):
            u = frontier.popleft()
            for v, sc in store.neighbors(u):
                weight[v] = max(weight.get(v, 0.0), sc)
                if v not in seen: seen.add(v); nxt.append(v)
        frontier.extend(nxt)
    # cap: seeds always kept; fill remaining slots with highest-weight non-seeds
    others = sorted((n for n in seen if n not in set(seeds)),
                    key=lambda n: (-weight[n], n))
    keep = seeds + others[: max(0, cfg.N_max - len(seeds))]
    idx = {n: i for i, n in enumerate(keep)}
    # anchors A [N,384] (fallback centroid for nodes lacking a stored vector)
    def _anc(n):
        v = store.anchor(n)
        return _FALLBACK_ANCHOR if v is None else v
    A = np.stack([_anc(n) for n in keep]).astype(np.float32)
    # edges among N (undirected, emitted both directions to match the symmetric coupling)
    und: set[tuple[int, int]] = set()
    for u in keep:
        iu = idx[u]
        for v, _sc in store.neighbors(u):
            iv = idx.get(v)
            if iv is not None and iv != iu: und.add((min(iu, iv), max(iu, iv)))
    src: list[int] = []; dst: list[int] = []
    for a, b in und: src += [a, b]; dst += [b, a]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ep = Episode(node_ids=keep, A=torch.from_numpy(A), edge_index=edge_index, id_to_idx=idx)
    return ep, list(range(len(seeds)))

# seed_init: X_0 directions = rownorm(P·A); seeds hot (R_max·c_seed), non-seeds cool ~0 (§3.2).
# P is a fixed random projection seeded by rng_seed (determinism). Returns (X0, P).
def seed_init(ep: Episode, seed_idx: list[int], C: Coupling, cfg: FieldConfig, rng_seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(rng_seed)
    P = F.normalize(torch.randn(cfg.d, EMBED_DIM, generator=g), dim=-1)   # [d,384] fixed
    dirs = F.normalize(ep.A.float() @ P.T, dim=-1)                        # [N,d] = rownorm(P·A)
    X0 = torch.zeros(len(ep.node_ids), cfg.d)
    X0[seed_idx] = (C.R_max * cfg.c_seed) * dirs[seed_idx]
    return X0, P

# build_anchor: pin seeds to their hot init; optionally pin inherited (parent) rows to parent state
# (§3.3, §5). The anchor potential keeps E a Lyapunov function while tying the gather to its seeds.
def build_anchor(ep: Episode, seed_idx: list[int], X0: Tensor, cfg: FieldConfig,
                 inherited_idx: list[int] | None = None, inherited_target: Tensor | None = None) -> Anchor:
    N, d = len(ep.node_ids), cfg.d
    mask = torch.zeros(N); target = torch.zeros(N, d)
    mask[seed_idx] = 1.0; target[seed_idx] = X0[seed_idx]
    if inherited_idx:
        mask[inherited_idx] = 1.0; target[inherited_idx] = inherited_target
    return Anchor(mask=mask, target=target)

# gather: the one physics op — seed-anchored convergent settle over a materialized Episode (§3).
def gather(ep: Episode, seed_idx: list[int], cfg: FieldConfig, rng_seed: int = 0,
           inherited_idx: list[int] | None = None, inherited_target: Tensor | None = None) -> GatherResult:
    C, cfg = safe_build(ep, cfg)
    X0, _ = seed_init(ep, seed_idx, C, cfg, rng_seed)
    anc = build_anchor(ep, seed_idx, X0, cfg, inherited_idx, inherited_target)
    Xh, Eh = rollout(X0, C, cfg, anc)
    return GatherResult(Xh[-1], Xh, Eh, ep, C, cfg, list(seed_idx), Xh.shape[0] - 1)

# gather_from_store: materialize the active set from S, then settle (§3.1 + §3).
def gather_from_store(store: _StoreProto, seed_ids: list[str], cfg: FieldConfig, rng_seed: int = 0) -> GatherResult:
    ep, seed_idx = materialize(store, seed_ids, cfg)
    return gather(ep, seed_idx, cfg, rng_seed)

# hop_distances: graph distance (in E(S)∩N) from the nearest seed for every node — for the
# locality invariant (mesh ⊆ k-hop) and the relevance-by-hop readout. Unreachable ⇒ -1.
def hop_distances(ep: Episode, seed_idx: list[int]) -> list[int]:
    N = len(ep.node_ids)
    adj: list[list[int]] = [[] for _ in range(N)]
    s, d = ep.edge_index
    for a, b in zip(s.tolist(), d.tolist()): adj[a].append(b)
    dist = [-1] * N
    q = deque()
    for i in seed_idx: dist[i] = 0; q.append(i)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0: dist[v] = dist[u] + 1; q.append(v)
    return dist
