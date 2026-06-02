from __future__ import annotations
import torch, torch.nn as nn, math
from torch import Tensor
from .config import FieldConfig

# Routing policy g (spec §4, guide §3.5 + §10.2).
# Per-node routing over K operators: alpha_{n,k} = softmax_k(u_k^T x_n + v_k^T h_S(x_n))
# Per-node memory mass: g_mem[n] = softmax_n(max_k Score_{n,k})
# The discrete top-m sparsification is the ONLY estimator-dependent step (§7.2).

class FieldRouter(nn.Module):
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        K, d = cfg.K, cfg.d_lat
        # Per-operator scoring parameters u_k, v_k ∈ ℝ^d (§4)
        U_init = torch.empty(K, d); nn.init.xavier_uniform_(U_init.unsqueeze(0)).squeeze_(0)
        V_init = torch.empty(K, d); nn.init.xavier_uniform_(V_init.unsqueeze(0)).squeeze_(0)
        self.U = nn.Parameter(U_init)   # [K, d]
        self.V = nn.Parameter(V_init)   # [K, d]

    # h_S(x)_n = Σ_{j∈N(n)} φ_t(n,j)·x_j — φ-weighted neighbor aggregation
    # phi_masked: [N, N] adjacency-masked kernel, x: [N, d_lat] → returns [N, d_lat]
    def _h_s(self, x: Tensor, phi_masked: Tensor) -> Tensor:
        # Row-normalize φ so rows sum to 1 (prevents scale blowup)
        row_sum = phi_masked.sum(dim=-1, keepdim=True).clamp(min=self.cfg.eps)
        phi_norm = phi_masked / row_sum
        return phi_norm @ x   # [N, d_lat]

    # Main routing call. Returns (alpha, g_mem, score) where:
    #   alpha: [N, K] — per-node routing over operators (softmax over K, top-m sparsified)
    #   g_mem: [N]   — per-node memory mass (softmax over N)
    #   score: [N, K] — raw scores (for diagnostics/learning)
    def forward(self, x: Tensor, phi_masked: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self._h_s(x, phi_masked)       # [N, d_lat]
        # Score_{n,k} = u_k^T x_n + v_k^T h_n
        score = x @ self.U.T + h @ self.V.T    # [N, K]
        alpha = self._sparsify(score)           # [N, K] sparse softmax over K
        # Per-node memory mass: softmax_n over max operator score per node
        g_logit = score.max(dim=-1).values     # [N]
        g_mem = torch.softmax(g_logit, dim=0)  # [N], sums to 1
        return alpha, g_mem, score

    def _sparsify(self, score: Tensor) -> Tensor:
        # top-m sparsification with chosen estimator
        m = min(self.cfg.top_m, score.shape[-1])
        if self.cfg.estimator == "gumbel":
            return self._gumbel_topk(score, m)
        if self.cfg.estimator == "reinforce":
            return self._reinforce_sample(score, m)
        return self._straight_through_topk(score, m)   # default

    # Straight-through estimator: hard top-m forward, soft backward (§7.2)
    def _straight_through_topk(self, score: Tensor, m: int) -> Tensor:
        soft = torch.softmax(score, dim=-1)         # [N, K]
        # Hard mask: 1 for top-m per row, 0 elsewhere
        _, idx = score.topk(m, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
        # Straight-through: forward uses hard, backward uses soft
        return hard + (soft - soft.detach())

    # Gumbel-Softmax with top-m hard sample (bias→0 as temperature→0)
    def _gumbel_topk(self, score: Tensor, m: int) -> Tensor:
        tau = 1.0   # temperature — could be annealed
        gumbel = -torch.log(-torch.log(torch.rand_like(score).clamp(min=1e-10)))
        perturbed = (score + gumbel) / tau
        soft = torch.softmax(perturbed, dim=-1)
        _, idx = perturbed.topk(m, dim=-1)
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
        return hard + (soft - soft.detach())

    # REINFORCE: sample m operators per node without replacement; store log_prob in buffer
    def _reinforce_sample(self, score: Tensor, m: int) -> Tensor:
        probs = torch.softmax(score, dim=-1)   # [N, K]
        # Sample top-m via multinomial (without replacement approximation via sorting)
        sampled = torch.multinomial(probs, m, replacement=False)  # [N, m]
        hard = torch.zeros_like(probs).scatter_(-1, sampled, 1.0)
        # Store log_prob as attribute for REINFORCE gradient computation in learn.py
        self._last_log_prob = (probs * hard).sum(dim=-1).log().sum()
        return hard

    # §9.2 kernel entropy guard: assert Var(φ_t) >= eps_phi
    @staticmethod
    def assert_kernel_entropy(phi: Tensor, cfg: FieldConfig) -> None:
        var = float(phi.var().item())
        assert var >= cfg.eps_phi, f"§9.2 VIOLATION: Var(φ_t)={var:.6f} < eps_phi={cfg.eps_phi}"
