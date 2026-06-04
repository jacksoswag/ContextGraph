from __future__ import annotations
# harness.py — toy-graph builders + episode setup, reused by gather tests/experiments.
# (The retired rotation-sweep experiment driver was removed at G3.)
import torch, torch.nn.functional as F
from torch import Tensor
from .config import FieldConfig
from .episode import Episode
from .coupling import build as build_coupling, Coupling

# ── anchor builders ─────────────────────────────────────────────────────────────
# Centers Gram-Schmidt orthogonalized → distinct energy basins per group.
def _anchors(group_sizes: list[int], spread: float = 0.15, seed: int = 42) -> Tensor:
    g = torch.Generator().manual_seed(seed); d = 384; parts: list[Tensor] = []; centers: list[Tensor] = []
    for n in group_sizes:
        v = F.normalize(torch.randn(d, generator=g), dim=0)
        for c in centers: v = F.normalize(v - (v @ c) * c, dim=0)
        centers.append(v)
        parts.append(F.normalize(v.unsqueeze(0) + torch.randn(n, d, generator=g) * spread, dim=-1))
    return torch.cat(parts, dim=0)

# ── edge helpers ────────────────────────────────────────────────────────────────
def _clique_edges(nodes: list[int]) -> list[tuple[int, int]]:
    return [(i, j) for i in nodes for j in nodes if i != j]

def _ring_edges(nodes: list[int]) -> list[tuple[int, int]]:
    n = len(nodes)
    return [(nodes[k], nodes[(k+1)%n]) for k in range(n)] + [(nodes[(k+1)%n], nodes[k]) for k in range(n)]

def _ei(edges: list[tuple[int, int]]) -> Tensor:
    s = [e[0] for e in edges]; d = [e[1] for e in edges]
    return torch.stack([torch.tensor(s, dtype=torch.long), torch.tensor(d, dtype=torch.long)])

# ── toy graph builders ───────────────────────────────────────────────────────────
def make_single_clique(n: int = 6, seed: int = 42) -> Episode:
    ids = [f"c{i}" for i in range(n)]
    return Episode(ids, _anchors([n], seed=seed), _ei(_clique_edges(list(range(n)))),
                   {v: i for i, v in enumerate(ids)})

def make_two_cliques_bridge(n1: int = 4, n2: int = 4, seed: int = 42) -> Episode:
    N = n1 + n2; ids = [f"a{i}" for i in range(n1)] + [f"b{i}" for i in range(n2)]
    edges = (_clique_edges(list(range(n1))) + _clique_edges(list(range(n1, N)))
             + [(n1-1, n1), (n1, n1-1)])
    return Episode(ids, _anchors([n1, n2], seed=seed), _ei(edges), {v: i for i, v in enumerate(ids)})

def make_ring_of_cliques(k: int = 3, csize: int = 3, seed: int = 42) -> Episode:
    ids = [f"r{g}{i}" for g in range(k) for i in range(csize)]
    edges: list[tuple[int, int]] = []
    for g in range(k):
        base = g * csize; edges += _clique_edges(list(range(base, base + csize)))
    for g in range(k):
        last = g * csize + csize - 1; nxt = ((g + 1) % k) * csize
        edges += [(last, nxt), (nxt, last)]
    return Episode(ids, _anchors([csize]*k, seed=seed), _ei(edges), {v: i for i, v in enumerate(ids)})

def make_two_incomm_rings(r1: int = 5, r2: int = 7, seed: int = 42) -> Episode:
    N = r1 + r2; ids = [f"p{i}" for i in range(r1)] + [f"q{i}" for i in range(r2)]
    edges = _ring_edges(list(range(r1))) + _ring_edges(list(range(r1, N))) + [(0, r1), (r1, 0)]
    return Episode(ids, _anchors([r1, r2], seed=seed), _ei(edges), {v: i for i, v in enumerate(ids)})

# ── coupling build: thin wrapper kept for mesh_gather's call site ────────────────
# Post-PPR-gate there is no step size to clamp, so this just builds C; the cfg is returned unchanged
# to preserve the (Coupling, cfg) tuple shape mesh_gather destructures.
def safe_build(ep: Episode, cfg: FieldConfig, edge_w=None) -> tuple[Coupling, FieldConfig]:
    return build_coupling(ep, cfg, edge_w), cfg
