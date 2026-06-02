from __future__ import annotations
# ADAPTIVE DOMAIN CONTROLLER — a moving-domain ("Lagrangian/AMR") layer around the field. It does
# NOT touch the energy or the dynamics (energy.py/dynamics.py unchanged); it only loads/unloads
# which nodes+edges are in the simulation, following where energy pools. Per phase: settle the
# field on the live set (warm-started), CULL cold non-anchors (≈energy-neutral), LOAD SQL neighbors
# at hot boundaries (energy-driven frontier), and RE-ANCHOR committed (ever-hot) nodes to their
# settled state so energy carries along the relevant trail. Bounded live set ⇒ flat compute;
# accumulating committed set ⇒ reach/breadth scale with the trajectory. Output = committed set.
from dataclasses import dataclass, field
import math, numpy as np, torch
from torch import Tensor
from .config import FieldConfig, DEFAULT_CFG
from .episode import Episode
from .energy import Anchor
from .dynamics import rollout
from .gather import _FALLBACK_ANCHOR, EMBED_DIM, seed_init
from .harness import safe_build

# TreeContext: per-tree working memory — lazy node→anchor and node→neighbors caches over the store,
# shared by every solve/gather in one answer() tree (so each node/edge is read from SQL once).
class TreeContext:
    def __init__(self, store) -> None:
        self.store = store
        self._anc: dict[str, np.ndarray | None] = {}
        self._adj: dict[str, list[tuple[str, float]]] = {}
    def neighbors(self, nid: str) -> list[tuple[str, float]]:
        v = self._adj.get(nid)
        if v is None: v = self._adj[nid] = self.store.neighbors(nid)
        return v
    def anchor(self, nid: str) -> np.ndarray | None:
        if nid not in self._anc: self._anc[nid] = self.store.anchor(nid)
        return self._anc[nid]
    @property
    def loaded(self) -> int: return len(self._adj)

# episode_from_nodes: materialize an Episode over an EXPLICIT node set (edges = those among the set).
def episode_from_nodes(ctx: TreeContext, node_ids: list[str]) -> Episode:
    ids = list(dict.fromkeys(node_ids))
    idx = {n: i for i, n in enumerate(ids)}
    A = np.stack([ctx.anchor(n) if ctx.anchor(n) is not None else _FALLBACK_ANCHOR
                  for n in ids]).astype(np.float32)
    und: set[tuple[int, int]] = set()
    for u in ids:
        iu = idx[u]
        for v, _ in ctx.neighbors(u):
            iv = idx.get(v)
            if iv is not None and iv != iu: und.add((min(iu, iv), max(iu, iv)))
    src: list[int] = []; dst: list[int] = []
    for a, b in und: src += [a, b]; dst += [b, a]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    return Episode(ids, torch.from_numpy(A), ei, idx)

@dataclass
class DomainConfig:
    # thresholds are FRACTIONS of R_max² (a fixed scale, not max-relative — so breadth can't collapse)
    eps_low: float = 0.02      # cull a non-anchor below this (cold backwater)
    eps_high: float = 0.20     # load SQL neighbors of nodes above this (hot boundary)
    eps_commit: float = 0.08   # commit (ever-hot ⇒ output + re-anchor) above this
    max_phases: int = 16
    max_live: int = 400        # bound the live simulation (flat compute)
    loads_per_phase: int = 80  # frontier I/O budget per phase
    reanchor: bool = True      # pin recent committed nodes to their settled state (carry the trail)
    anchor_ttl: int = 3        # phases a committed node stays a live re-anchor before the baton passes
                               #   on (it then leaves the live sim but STAYS in the committed output)

@dataclass
class DomainResult:
    committed: list[str]                # output: ever-hot node-ids, relevance-ranked
    relevance: dict[str, float]         # best r seen per committed node
    parent: dict[str, str]             # committed node → the committed node that first loaded it (provenance)
    live: list[str]                     # final live set
    phases: int
    peak_live: int
    total_loaded: int                   # nodes read from SQL (cost proxy)
    trace: list[dict]                   # per-phase {live, committed, loaded, culled, max_reach}

# adaptive_gather: the moving-domain settle (spec extension of §3). Physics untouched.
def adaptive_gather(ctx: TreeContext, seed_ids: list[str], cfg: FieldConfig = DEFAULT_CFG,
                    dcfg: DomainConfig = DomainConfig(), rng_seed: int = 0) -> DomainResult:
    seed_set = list(dict.fromkeys(seed_ids))
    live: set[str] = set(seed_set)
    for s in seed_set:                                  # seed the front with the seeds' 1-hop
        live |= {n for n, _ in ctx.neighbors(s)}
    committed: dict[str, float] = {}                    # OUTPUT: ever-hot id → best r (accumulates)
    commit_phase: dict[str, int] = {s: -1 for s in seed_set}    # for the TTL baton
    state: dict[str, Tensor] = {}                       # settled vector per CURRENTLY-LIVE node (warm-start + re-anchor)
    parent: dict[str, str] = {s: s for s in seed_set}
    just_culled: set[str] = set()
    trace: list[dict] = []; peak = 0; total_loaded = 0

    for phase in range(dcfg.max_phases):
        # live anchors = seeds (permanent) + committed within the TTL window (carry the trail)
        live_anchor = {s for s in seed_set} | {
            n for n, ph in commit_phase.items() if ph >= 0 and phase - ph < dcfg.anchor_ttl and n in live}
        if len(live) > dcfg.max_live:                   # compute bound: drop coldest non-anchors
            extra = sorted((n for n in live if n not in live_anchor and n not in seed_set),
                           key=lambda n: float((state.get(n, torch.zeros(cfg.d)) ** 2).sum()))
            for n in extra[: len(live) - dcfg.max_live]: live.discard(n)
        ep = episode_from_nodes(ctx, sorted(live))      # sorted ⇒ deterministic indexing
        C, cfg2 = safe_build(ep, cfg)
        R2 = C.R_max ** 2
        seed_idx = [ep.id_to_idx[s] for s in seed_set if s in ep.id_to_idx] or [0]
        # init: hot seeds + cool rest, then warm-start retained nodes from their saved settled state
        X0, _ = seed_init(ep, seed_idx, C, cfg2, rng_seed)
        for nid, vec in state.items():
            if nid in ep.id_to_idx and nid not in seed_set: X0[ep.id_to_idx[nid]] = vec
        # anchors: seeds (hot init) + TTL-recent committed (re-anchored) — energy carries the trail
        mask = torch.zeros(len(ep.node_ids)); target = torch.zeros(len(ep.node_ids), cfg2.d)
        for i in seed_idx: mask[i] = 1.0; target[i] = X0[i]
        if dcfg.reanchor:
            for nid in live_anchor:
                if nid in ep.id_to_idx and nid not in seed_set and nid in state:
                    j = ep.id_to_idx[nid]; mask[j] = 1.0; target[j] = state[nid]
        Xh, _ = rollout(X0, C, cfg2, Anchor(mask=mask, target=target))
        Xs = Xh[-1]; r = (Xs * Xs).sum(-1)              # relevance per live node

        # observe → record committed (ever-hot); refresh warm-start/anchor state for live nodes
        state = {nid: Xs[ep.id_to_idx[nid]].clone() for nid in ep.node_ids}
        for nid in ep.node_ids:
            ri = float(r[ep.id_to_idx[nid]])
            if ri >= dcfg.eps_commit * R2 or nid in seed_set:
                if nid not in commit_phase: commit_phase[nid] = phase
                committed[nid] = max(committed.get(nid, 0.0), ri)

        # DOMAIN EDIT (no physics here): cull cold non-anchors, load neighbors of hot boundary
        keep = set(seed_set) | live_anchor
        cold = {nid for nid in ep.node_ids
                if float(r[ep.id_to_idx[nid]]) < dcfg.eps_low * R2 and nid not in keep}
        expired = {n for n, ph in commit_phase.items()      # baton passed: drop from live sim (stay in output)
                   if ph >= 0 and phase - ph >= dcfg.anchor_ttl and n not in seed_set and n in live}
        hot = [nid for nid in ep.node_ids if float(r[ep.id_to_idx[nid]]) >= dcfg.eps_high * R2]
        loaded: set[str] = set()
        for nid in hot:
            for nbr, _ in ctx.neighbors(nid):
                if nbr not in live and nbr not in just_culled and nbr not in loaded:
                    loaded.add(nbr)
                    if nbr not in parent: parent[nbr] = nid       # provenance: who loaded it
                    if len(loaded) >= dcfg.loads_per_phase: break
            if len(loaded) >= dcfg.loads_per_phase: break
        total_loaded += len(loaded)
        peak = max(peak, len(live))
        trace.append({"phase": phase, "live": len(live), "committed": len(committed),
                      "loaded": len(loaded), "culled": len(cold | expired),
                      "max_reach": _reach(parent, seed_set, committed)})

        new_live = (live - cold - expired) | loaded
        for n in (cold | expired): state.pop(n, None)     # forget unloaded nodes' state
        just_culled = cold                                # one-phase cooldown (hysteresis)
        if new_live == live: break                        # quiesced: domain stable ⇒ done
        live = new_live

    ordered = [s for s in seed_set] + sorted((n for n in committed if n not in seed_set),
                                             key=lambda n: -committed[n])
    rel = {n: committed.get(n, float("inf")) for n in ordered}
    return DomainResult(committed=ordered, relevance=rel, parent=parent, live=sorted(live),
                        phases=len(trace), peak_live=peak, total_loaded=total_loaded, trace=trace)

# _reach: max hop-distance of any committed node from a seed in the provenance (load) tree.
def _reach(parent: dict[str, str], seeds: list[str], committed: dict[str, float]) -> int:
    seedset = set(seeds); best = 0
    for n in committed:
        d, cur, seen = 0, n, set()
        while cur not in seedset and cur in parent and parent[cur] != cur and cur not in seen:
            seen.add(cur); cur = parent[cur]; d += 1
            if d > 1000: break
        if cur in seedset: best = max(best, d)
    return best
