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

# Mesh: the gather readout (spec §3.5) — relevance-ranked nodes + a seed-rooted provenance tree
# giving every gathered concept a citation chain back to a seed.
@dataclass
class Mesh:
    nodes: list[int]               # mesh node indices into ep, relevance-descending
    node_ids: list[str]            # corresponding S node-ids
    relevance: dict[int, float]    # idx → r_i = ‖x*_i‖²
    parent: dict[int, int]         # idx → provenance parent idx; seed roots → -1
    layer: dict[int, int]          # idx → hop layer in the provenance tree
    seed_roots: list[int]          # seed indices (tree roots)
    def chain(self, i: int) -> list[int]:    # seed→…→i citation chain (indices)
        out = [i]
        while self.parent[out[-1]] != -1: out.append(self.parent[out[-1]])
        return list(reversed(out))
    def chain_ids(self, i: int) -> list[str]:    # same chain as S node-ids
        m = dict(zip(self.nodes, self.node_ids))
        return [m[j] for j in self.chain(i)]

# build_mesh: select the relevance-ranked mesh (soft/top-k, per 📍1 — τ=0.5 collapses to seed-only)
# and grow a seed-rooted spanning tree over it. provenance="flow" (parent = max
# C_sym[i,j]·⟨x*_i,x*_j⟩ lower-layer neighbor, spec §3.5 default) or "shortest" (max structural
# edge weight C_sym among min-hop neighbors). Nodes unreachable through the mesh are dropped.
def build_mesh(res: GatherResult, tau_rel: float = 0.01, top_k: int | None = None,
               provenance: str = "flow") -> Mesh:
    rel = res.relevance(); mx = float(rel.max()); N = len(res.ep.node_ids)
    seeds = list(res.seed_idx)
    if mx <= 0:
        return Mesh(list(seeds), [res.ep.node_ids[i] for i in seeds],
                    {i: 0.0 for i in seeds}, {i: -1 for i in seeds}, {i: 0 for i in seeds}, seeds)
    sel = {i for i in range(N) if float(rel[i]) >= tau_rel * mx} | set(seeds)
    if top_k is not None:
        ranked = sorted(sel, key=lambda i: (-float(rel[i]), i))[: max(top_k, len(seeds))]
        sel = set(ranked) | set(seeds)
    # induced adjacency on the selected set
    adj: dict[int, list[int]] = {i: [] for i in sel}
    s, d = res.ep.edge_index
    for a, b in zip(s.tolist(), d.tolist()):
        if a in sel and b in sel: adj[a].append(b)
    Csym, Xs = res.C.sym, res.X_star
    def key(i: int, j: int) -> float:
        w = float(Csym[i, j])
        return w * float((Xs[i] * Xs[j]).sum()) if provenance == "flow" else w
    # layered BFS from seeds; each node attaches to its best-key strictly-lower-layer neighbor
    parent: dict[int, int] = {}; layer: dict[int, int] = {}; visited: set[int] = set()
    for sd in seeds:
        if sd in sel: parent[sd] = -1; layer[sd] = 0; visited.add(sd)
    cur = 0
    while True:
        nxt = []
        for j in sorted(sel - visited):
            lower = [i for i in adj[j] if i in visited]
            if lower:
                parent[j] = max(lower, key=lambda i: (key(i, j), -i)); layer[j] = cur + 1; nxt.append(j)
        if not nxt: break
        visited.update(nxt); cur += 1
    nodes = sorted(visited, key=lambda i: (-float(rel[i]), i))
    return Mesh(nodes, [res.ep.node_ids[i] for i in nodes],
                {i: float(rel[i]) for i in nodes}, parent, layer, list(seeds))

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
