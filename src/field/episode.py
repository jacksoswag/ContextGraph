from __future__ import annotations
from dataclasses import dataclass, field
from torch import Tensor

@dataclass
class Episode:
    node_ids: list[str]
    A: Tensor            # [N, 384] frozen anchors
    edge_index: Tensor   # [2, E] both-direction pairs on E(S)
    id_to_idx: dict[str, int]
    # beyond-pairwise groups: each (member_indices, weight) is a reified edge bound to
    # its child endpoints — energy stays within it (containment). Empty ⇒ flat dyadic.
    hyperedges: list[tuple[list[int], float]] = field(default_factory=list)
