from __future__ import annotations
import torch, torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from .config import FieldConfig
from .dynamics import FieldModel, RolloutResult, FieldState
from .spaces import Episode
from .objective import CComponents, scalar_c

# θ-learning: hybrid regime (spec §7, guide §5).
# Two arms: distribution-matching + structural-invariance.
# C(τ) shapes/weights updates (never the loss itself).
# The exact learning rule is pluggable (open research question, §5 guide note).

@dataclass
class LearnerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    c_shape_weight: float = 0.1   # how strongly C shapes the update weighting

class FieldLearner:
    def __init__(self, model: FieldModel, cfg: FieldConfig, lcfg: LearnerConfig | None = None):
        self.model = model
        self.cfg = cfg
        self.lcfg = lcfg or LearnerConfig()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lcfg.lr,
                                          weight_decay=self.lcfg.weight_decay)
        self._step_count = 0

    # Full θ update from one episode's rollout.
    # components_list: C components per step (for shaping).
    # Returns dict of loss scalars for diagnostics.
    def update(self, rollout: RolloutResult, ep: Episode,
               components_list: list[CComponents] | None = None) -> dict[str, float]:
        self.optimizer.zero_grad()
        loss_dist = self._dist_match_loss(rollout, ep)
        loss_struct = self._struct_inv_loss(rollout, ep)
        # C shaping: weight the arms by (1 + c_shape_weight * |C|)
        c_val = scalar_c(components_list, self.cfg) if components_list else 0.0
        shape = 1.0 + self.lcfg.c_shape_weight * abs(c_val)
        # Total loss: C shapes magnitude but is never the target itself
        loss = shape * (loss_dist + loss_struct)
        if loss.requires_grad: loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # Update P_slow EMA after each θ step (§2.1.1)
        self.model.update_P_slow()
        self._step_count += 1
        return {"loss_dist": float(loss_dist.item()),
                "loss_struct": float(loss_struct.item()),
                "loss_total": float(loss.item()),
                "c_val": c_val, "c_shape": shape}

    # Distribution-matching arm: forward-predict next-state on the anchor manifold.
    # Match the trajectory distribution by minimizing anchor-space prediction error.
    def _dist_match_loss(self, rollout: RolloutResult, ep: Episode) -> Tensor:
        if len(rollout.results) < 2: return torch.tensor(0.0)
        losses = []
        A = ep.A   # [N, d_anchor], frozen
        for t, (state_t, res) in enumerate(zip(rollout.states[:-1], rollout.results)):
            x_t = state_t.X
            x_next = res.x_next
            # Predict next-state anchor projections from current latents
            pred_next = self.model.P(x_t)   # [N, d_lat] — using trainable P, not P_slow
            # Target: anchor-space representation of x_next (detached)
            target = x_next.detach()
            losses.append(F.mse_loss(pred_next, target))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    # Structural-invariance arm: anchor↔latent coupling + topology preservation (§10.5).
    def _struct_inv_loss(self, rollout: RolloutResult, ep: Episode) -> Tensor:
        A = ep.A   # [N, d_anchor], frozen
        # §7 / §10.5 anchor binding: ‖x_v - P(a_v)‖²
        x_final = rollout.states[-1].X
        p_av = self.model.P(A)   # [N, d_lat]
        bind_loss = F.mse_loss(x_final, p_av.detach())
        # §10.5 topology preservation: trace(X^T L_a X)
        topo_loss = self._topo_loss(x_final, A)
        return self.cfg.eta_bind * bind_loss + self.cfg.eta_topo * topo_loss

    # Laplacian consistency loss: forbids reordering of semantic neighborhood.
    # Builds anchor k-NN graph (cosine), computes R_topo = (1/|E_a|) Σ s_a(i,j)·‖x_i-x_j‖²
    # Vectorized: gather all [N, k] neighbor latents in one op, no Python loop.
    def _topo_loss(self, X: Tensor, A: Tensor) -> Tensor:
        N, d = X.shape
        k = min(self.cfg.knn_k, N - 1)
        if k < 1: return torch.tensor(0.0)
        with torch.no_grad():
            A_norm = A / (A.norm(dim=-1, keepdim=True) + self.cfg.eps)
            sim = A_norm @ A_norm.T   # [N, N]
            sim.fill_diagonal_(-2.0)
            knn_vals, knn_idx = sim.topk(k, dim=-1)   # [N, k]
            mask = knn_vals > 0                        # [N, k]
        # Gather neighbor latents: X_nbr[i, j] = X[knn_idx[i, j]]
        X_nbr = X[knn_idx.view(-1)].view(N, k, d)     # [N, k, d]
        X_src = X.unsqueeze(1).expand_as(X_nbr)        # [N, k, d]
        diff_sq = (X_src - X_nbr).pow(2).sum(-1)       # [N, k]
        # Weight by anchor cosine similarity; mask non-positive entries
        weighted = (knn_vals * diff_sq * mask.float()).sum()
        count = mask.sum().clamp(min=1)
        return weighted / count
