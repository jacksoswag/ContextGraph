from __future__ import annotations
import torch, numpy as np
from torch import Tensor
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .config import FieldConfig

# Per-episode subgraph materialized from the persistent hypergraph S.
# Carries frozen anchors A (MiniLM, 384-d) and the graph topology.
# Trainable latents X are NOT stored here — they live in FieldState.

@dataclass
class Episode:
    node_ids: list[str]                        # ordered active-set nodes
    A: Tensor                                  # [N, d_anchor] frozen semantic anchors
    edge_src: Tensor                           # [E] long — pairwise edge sources (local idx)
    edge_tgt: Tensor                           # [E] long — pairwise edge targets (local idx)
    edge_w: Tensor                             # [E] float — pairwise edge weights
    hyperedges: list[tuple[list[int], float]]  # [(member_indices, w), ...] beyond-pairwise
    id_to_idx: dict[str, int]

    @property
    def N(self) -> int: return len(self.node_ids)

# Current differentiable state within a rollout step.
@dataclass
class FieldState:
    X: Tensor   # [N, d_lat] — node latents (leaf when requires_grad=True)
    m: Tensor   # [d_lat]   — trajectory memory

# Load 384-d anchor for a node from a numpy float32 blob. Returns zero vec on miss.
def _load_anchor(blob: bytes | None, d: int) -> np.ndarray:
    if blob is None: return np.zeros(d, dtype=np.float32)
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.shape[0] != d: return np.zeros(d, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr

# Build Episode from SQLite for a given ordered node list.
# Loads 1-hop edges between episode nodes and detects hyperedge endpoints.
# hyperedges = edges whose id appears as source/target of other edges (e_ prefix).
def materialize(conn, node_ids: list[str], cfg: FieldConfig) -> Episode:
    if not node_ids: raise ValueError("materialize: empty node_ids")
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    # Load anchors in bulk (up to 900 per query)
    anchor_map: dict[str, bytes | None] = {}
    for i in range(0, N, 900):
        chunk = node_ids[i : i+900]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT id, info_vector FROM nodes WHERE id IN ({ph})", chunk):
            anchor_map[r["id"]] = r["info_vector"]
    A_np = np.stack([_load_anchor(anchor_map.get(nid), cfg.d_anchor) for nid in node_ids])
    A = torch.tensor(A_np, dtype=torch.float32)
    # Load all edges between episode nodes (induced subgraph)
    src_l, tgt_l, w_l = [], [], []
    seen_edges: set[str] = set()
    node_set = set(node_ids)
    for i in range(0, N, 900):
        chunk = node_ids[i : i+900]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT id, source_id, target_id, score FROM edges "
            f"WHERE source_id IN ({ph}) AND target_id IN ({ph})",
            chunk + chunk
        ):
            eid, s, t, w = r["id"], r["source_id"], r["target_id"], float(r["score"])
            if eid in seen_edges or s not in id_to_idx or t not in id_to_idx: continue
            seen_edges.add(eid)
            si, ti = id_to_idx[s], id_to_idx[t]
            src_l += [si, ti]; tgt_l += [ti, si]; w_l += [w, w]   # undirected
    # Detect hyperedges: episode nodes that are edge ids (e_ prefix) appear as
    # endpoints of other edges; load their member endpoints.
    edge_node_ids = [nid for nid in node_ids if nid.startswith("e_")]
    hyperedges: list[tuple[list[int], float]] = []
    for eid in edge_node_ids:
        row = conn.execute("SELECT source_id, target_id, score FROM edges WHERE id=?", (eid,)).fetchone()
        if not row: continue
        members = [id_to_idx[ep] for ep in (row["source_id"], row["target_id"]) if ep in id_to_idx]
        members.append(id_to_idx[eid])
        w = float(row["score"])
        if len(members) >= 2: hyperedges.append((members, w))
    if src_l:
        edge_src = torch.tensor(src_l, dtype=torch.long)
        edge_tgt = torch.tensor(tgt_l, dtype=torch.long)
        edge_w   = torch.tensor(w_l, dtype=torch.float32)
    else:
        edge_src = torch.zeros(0, dtype=torch.long)
        edge_tgt = torch.zeros(0, dtype=torch.long)
        edge_w   = torch.zeros(0, dtype=torch.float32)
    return Episode(node_ids=node_ids, A=A, edge_src=edge_src, edge_tgt=edge_tgt,
                   edge_w=edge_w, hyperedges=hyperedges, id_to_idx=id_to_idx)

# Initialize FieldState from the bridge P applied to anchors.
# X is a leaf tensor so autograd tracks operator gradients through it.
def init_state(episode: Episode, P: torch.nn.Module, cfg: FieldConfig, *, seed_m: Tensor | None = None) -> FieldState:
    with torch.no_grad():
        X0 = P(episode.A)
        norm = X0.norm(dim=-1, keepdim=True).clamp(min=cfg.eps)
        X0 = X0 / norm
    X = X0.detach().requires_grad_(True)
    if seed_m is not None:
        m = seed_m.detach().clone()
    else:
        m = torch.zeros(cfg.d_lat, dtype=torch.float32)
    return FieldState(X=X, m=m)
