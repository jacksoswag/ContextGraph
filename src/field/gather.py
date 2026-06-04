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

# _grow_set: uniform best-first growth — every reached node expands UP (its containing edges;
# works for nodes AND edges, so containment depth still accrues) and, if it is an edge, DOWN (its
# endpoints). A laterally-reached entity therefore climbs its own spine, closing 2-hop gaps
# structurally. Flood control is guidance: weight = down_decay^level, so far nodes lose the
# N_max-cap contest. Stores without containing_edges (test doubles) fall back to lateral neighbors.
def _grow_set(store: _StoreProto, seeds: list[str], cfg: FieldConfig) -> tuple[set[str], dict[str, float]]:
    import heapq, itertools
    _ce = getattr(store, "containing_edges", None)
    _ch = getattr(store, "children", None)
    weight: dict[str, float] = {s: math.inf for s in seeds}; seen = set(seeds); cnt = itertools.count()
    heap = [(-1e18, next(cnt), s, 0) for s in seeds]; heapq.heapify(heap)
    while heap and len(seen) < cfg.N_max:
        _negw, _, u, lvl = heapq.heappop(heap)
        if lvl >= cfg.k_hop: continue
        base = cfg.down_decay ** lvl
        if _ce is not None:
            up_outs = [e for e, _sc in _ce(u, cfg.up_max)]   # UP: containment — increments level
            # DOWN: unpacking an edge's endpoints is NOT a hop — push at same lvl so a reached edge
            # always exposes its content regardless of k_hop depth (mirrors _khop_set pin step).
            dn_outs = list(_ch(u) or ()) if u.startswith("e_") and _ch is not None else []
        else:
            up_outs = [v for v, _sc in store.neighbors(u)]    # fallback: lateral neighbors
            dn_outs = []
        for v in up_outs:
            w = base
            if v not in seen:
                seen.add(v); weight[v] = w; heapq.heappush(heap, (-w, next(cnt), v, lvl + 1))
            else: weight[v] = max(weight[v], w)
        for v in dn_outs:
            # DOWN children inherit the parent e_ node's weight (not the decayed base) — they are
            # structural connectors (edge endpoints), not lateral hops, so must not be evicted by
            # N_max when the e_ node itself survives (a disconnected e_ node has no coupling).
            w = -_negw
            if v not in seen:
                seen.add(v); weight[v] = w; heapq.heappush(heap, (-w, next(cnt), v, lvl))
            else: weight[v] = max(weight[v], w)
    return seen, weight

# materialize: active set N capped to N_max by descending weight (§3.1). Uniform guided growth
# (_grow_set) is the only active-set builder — _khop_set and _climb_set deleted at finetune pass.
def materialize(store: _StoreProto, seed_ids: list[str], cfg: FieldConfig) -> tuple[Episode, list[int]]:
    seeds = [s for s in dict.fromkeys(seed_ids)]            # dedup, keep order
    seen, weight = _grow_set(store, seeds, cfg)
    _children = getattr(store, "children", None)
    # cap: seeds always kept; fill remaining slots with highest-weight non-seeds
    others = sorted((n for n in seen if n not in set(seeds)),
                    key=lambda n: (-weight[n], n))
    keep = seeds + others[: max(0, cfg.N_max - len(seeds))]
    idx = {n: i for i, n in enumerate(keep)}
    # hyperedge groups over the kept set: (member_indices, weight) for each reified edge
    # whose child endpoints are both present — coupling.build clique-binds them.
    hyperedges: list[tuple[list[int], float]] = []
    if _children is not None:
        for n in keep:
            if not n.startswith("e_"): continue
            kids = _children(n)
            if kids and len(kids) == 2 and kids[0] in idx and kids[1] in idx:   # unary (len 1) ⇒ no clique
                hyperedges.append(([idx[n], idx[kids[0]], idx[kids[1]]], 1.0))
    # anchors A [N,384] (fallback centroid for nodes lacking a stored vector)
    def _anc(n):
        v = store.anchor(n)
        return _FALLBACK_ANCHOR if v is None else v
    # genericity (global store degree) per kept node — only computed when node-wise decay is active
    deg = None
    if cfg.decay_gamma > 0.0 and hasattr(store, "degree"):
        deg = torch.tensor([float(store.degree(n)) for n in keep])
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
    ep = Episode(node_ids=keep, A=torch.from_numpy(A), edge_index=edge_index, id_to_idx=idx,
                 hyperedges=hyperedges, degree=deg)
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
# (§3.3, §5). `inherited_target` is a full [N,d] tensor (the masked rows are copied). The anchor
# potential keeps E a Lyapunov function while tying the gather to its seeds (and to the parent).
def build_anchor(ep: Episode, seed_idx: list[int], X0: Tensor, cfg: FieldConfig,
                 inherited_idx: list[int] | None = None, inherited_target: Tensor | None = None) -> Anchor:
    N, d = len(ep.node_ids), cfg.d
    mask = torch.zeros(N); target = torch.zeros(N, d)
    mask[seed_idx] = 1.0; target[seed_idx] = X0[seed_idx]
    if inherited_idx:
        mask[inherited_idx] = 1.0; target[inherited_idx] = inherited_target[inherited_idx]
    return Anchor(mask=mask, target=target)

# edge_weights: resolve a Sleep-learned per-edge multiplier tensor [E] for this episode (default 1).
# `weights` exposes .get(u_id, v_id) → multiplier; None ⇒ bootstrap coupling.
def edge_weights(ep: Episode, weights) -> Tensor | None:
    if weights is None: return None
    s, d = ep.edge_index
    nid = ep.node_ids
    return torch.tensor([weights.get(nid[a], nid[b]) for a, b in zip(s.tolist(), d.tolist())],
                        dtype=torch.float32)

# gather: the one physics op — seed-anchored convergent settle over a materialized Episode (§3).
# lean (live path) drops per-step trajectory/energy logging (X* identical; see dynamics.rollout).
def gather(ep: Episode, seed_idx: list[int], cfg: FieldConfig, rng_seed: int = 0,
           inherited_idx: list[int] | None = None, inherited_target: Tensor | None = None,
           weights=None, *, lean: bool = False) -> GatherResult:
    C, cfg = safe_build(ep, cfg, edge_weights(ep, weights))
    X0, _ = seed_init(ep, seed_idx, C, cfg, rng_seed)
    anc = build_anchor(ep, seed_idx, X0, cfg, inherited_idx, inherited_target)
    Xh, Eh, steps = rollout(X0, C, cfg, anc, lean=lean)
    return GatherResult(Xh[-1], Xh, Eh, ep, C, cfg, list(seed_idx), steps)

# gather_from_store: materialize the active set from S, then settle (§3.1 + §3).
def gather_from_store(store: _StoreProto, seed_ids: list[str], cfg: FieldConfig, rng_seed: int = 0,
                      weights=None, *, lean: bool = False) -> GatherResult:
    ep, seed_idx = materialize(store, seed_ids, cfg)
    return gather(ep, seed_idx, cfg, rng_seed, weights=weights, lean=lean)

# Mesh: the gather readout (spec §3.5) — relevance-ranked nodes, query-conditioned.
@dataclass
class Mesh:
    nodes: list[int]               # mesh node indices into ep, relevance-descending
    node_ids: list[str]            # corresponding S node-ids
    relevance: dict[int, float]    # idx → r_i = ‖x*_i‖²
    seed_roots: list[int]          # seed indices (entry points)

# build_mesh: select the relevance-ranked mesh (soft/top-k). query_w > 0 tilts the ranking toward
# query-cosine at READOUT — growth stays query-blind. tau_rel drops near-zero nodes so a thin seed
# returns its few real facets rather than target_size near-empty ones.
def build_mesh(res: GatherResult, tau_rel: float = 0.01, top_k: int | None = None,
               query_vec: Tensor | None = None, query_w: float = 0.0) -> Mesh:
    rel = res.relevance(); mx = float(rel.max()); N = len(res.ep.node_ids)
    seeds = list(res.seed_idx)
    if mx <= 0:
        return Mesh(list(seeds), [res.ep.node_ids[i] for i in seeds],
                    {i: 0.0 for i in seeds}, seeds)
    # query-conditioned readout: tilt structural relevance toward query-cosine at READOUT only.
    # rank selects/orders, rel stays reported and gates the tau_rel floor.
    if query_vec is not None and query_w > 0.0:
        qv = query_vec.float() / (query_vec.float().norm() + 1e-9)
        rank = rel * (1.0 + query_w * (F.normalize(res.ep.A.float(), dim=-1) @ qv).clamp(min=0.0))
    else:
        rank = rel
    sel = {i for i in range(N) if float(rel[i]) >= tau_rel * mx} | set(seeds)
    if top_k is not None:
        ranked = sorted(sel, key=lambda i: (-float(rank[i]), i))[: max(top_k, len(seeds))]
        sel = set(ranked) | set(seeds)
    nodes = sorted(sel, key=lambda i: (-float(rank[i]), i))
    return Mesh(nodes, [res.ep.node_ids[i] for i in nodes],
                {i: float(rel[i]) for i in nodes}, list(seeds))

# readout: the single-knob mesh — keep the top cfg.target_size relevance-ranked nodes (the breadth
# dial S*, fixed at 34). tau_rel still drops near-zero nodes so a thin seed returns its few real
# facets rather than target_size near-empty ones.
def readout(res: GatherResult, tau_rel: float = 0.01) -> Mesh:
    return build_mesh(res, tau_rel=tau_rel, top_k=res.cfg.target_size)

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
