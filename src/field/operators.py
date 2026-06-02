from __future__ import annotations
import torch, torch.nn as nn
from torch import Tensor
from .config import FieldConfig

# K fixed operators F^(k)(x) = W_k x + b_k + ψ_k(x, S) (spec §5).
# ψ_k is a bounded nonlinear map: ψ_k(x) = tanh(U_k x) (bounded ∈ [-1,1]).
# Mixture F̃(x) = Σ_k alpha_{n,k} · F^(k)(x_n) per node (per-node routing weights).
# §9.1: operator diversity floor enforced at init and checked at runtime.

class FieldOperator(nn.Module):
    def __init__(self, d: int, idx: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)
        self.U = nn.Linear(d, d, bias=False)   # ψ_k nonlinear branch
        # Diverse init: orthogonal + index-based rotation to spread operators apart
        nn.init.orthogonal_(self.W.weight)
        nn.init.orthogonal_(self.U.weight)
        # Additive offset so operators start far from each other (§9.1)
        with torch.no_grad():
            self.W.weight.data += (idx * 0.1) * torch.eye(d)

    def forward(self, x: Tensor) -> Tensor:
        # F^(k)(x) = W_k x + b_k + ψ_k(x)  with ψ bounded via tanh
        return self.W(x) + torch.tanh(self.U(x))


class FieldOperators(nn.Module):
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.ops = nn.ModuleList([FieldOperator(cfg.d_lat, k) for k in range(cfg.K)])
        self._check_diversity()

    # Mixture: F̃(x_n) = Σ_k alpha_{n,k} · F^(k)(x_n)
    # x: [N, d_lat], alpha: [N, K] — per-node routing over operators
    # Returns [N, d_lat]
    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        N, d = x.shape
        K = len(self.ops)
        # Stack all operator outputs: [N, K, d_lat]
        stacked = torch.stack([op(x) for op in self.ops], dim=1)
        # alpha: [N, K] → [N, K, 1] for broadcasting
        return (alpha.unsqueeze(-1) * stacked).sum(dim=1)   # [N, d_lat]

    # §9.1 diversity guard: E[‖F_i - F_j‖²] >= eps_op
    def diversity_score(self) -> float:
        K = len(self.ops)
        if K < 2: return float("inf")
        probe = torch.eye(self.cfg.d_lat)   # [d_lat, d_lat] identity probe
        with torch.no_grad():
            outs = torch.stack([op(probe) for op in self.ops])  # [K, d_lat, d_lat]
        diffs = []
        for i in range(K):
            for j in range(i+1, K):
                diffs.append(((outs[i] - outs[j])**2).mean().item())
        return float(sum(diffs) / len(diffs)) if diffs else float("inf")

    def _check_diversity(self) -> None:
        score = self.diversity_score()
        assert score >= self.cfg.eps_op, f"§9.1 VIOLATION at init: diversity={score:.6f} < eps_op={self.cfg.eps_op}"

    def assert_diversity(self) -> None:
        score = self.diversity_score()
        assert score >= self.cfg.eps_op, f"§9.1 VIOLATION: diversity={score:.6f} < eps_op={self.cfg.eps_op}"
