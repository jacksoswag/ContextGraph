from __future__ import annotations
from dataclasses import dataclass
from torch import Tensor

@dataclass
class Episode:
    node_ids: list[str]
    A: Tensor            # [N, 384] frozen anchors
    edge_index: Tensor   # [2, E] both-direction pairs on E(S)
    id_to_idx: dict[str, int]
