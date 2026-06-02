from __future__ import annotations
import torch
from torch import Tensor
from .config import FieldConfig

# m_t management: update, norm-projection, cross-episode policy (spec §2.3, §10.1, §10.2).
# m_t is always kept at ‖m_t‖ = m_ref (fixed norm = directional mixing, §10.2).
# sg(·) = stop-gradient is enforced by the caller passing detached x_{t+1}.

# Project m to fixed norm m_ref. Satisfies §9.3 floor by construction when m_ref >= eps_m.
def _project_norm(m: Tensor, cfg: FieldConfig) -> Tensor:
    n = m.norm().clamp(min=cfg.eps)
    return (m / n) * cfg.m_ref

# Update m_t → m_{t+1} from POST-Π_S, post-norm node latents x_next (detached).
# g_mem: [N] per-node routing mass (softmax-normalized, sums to 1).
# Enforces single-write rule: returns new m tensor, never mutates in-place.
def update_memory(m: Tensor, g_mem: Tensor, x_next: Tensor, cfg: FieldConfig) -> Tensor:
    # x_next must already be stop-grad (caller detaches before calling)
    # §10.1: m̃ = ρ·m_t + Σ_k g_k · x_{t+1,k}  (k over active nodes)
    weighted = (g_mem.unsqueeze(-1) * x_next).sum(dim=0)   # [d_lat]
    m_raw = cfg.rho * m + weighted
    return _project_norm(m_raw, cfg)

# Cross-episode memory initialization based on the mem_cross_ep axis (§4 guide).
# "decay": partial decay toward a seed vector (default)
# "reset": full zero reset
# "persist": keep m unchanged (risks §10.2 attractor freezing — not default)
def init_episode_memory(m_prev: Tensor | None, seed_vec: Tensor | None,
                         cfg: FieldConfig) -> Tensor:
    if cfg.mem_cross_ep == "persist" and m_prev is not None:
        return _project_norm(m_prev.detach(), cfg)
    if cfg.mem_cross_ep == "reset" or m_prev is None:
        if seed_vec is not None:
            return _project_norm(seed_vec.detach(), cfg)
        return _project_norm(torch.randn(cfg.d_lat), cfg)
    # decay-to-seed (default): blend previous m toward seed
    base = seed_vec if seed_vec is not None else torch.zeros(cfg.d_lat)
    decay_rate = 0.5  # half-life toward seed
    m_decayed = decay_rate * m_prev.detach() + (1.0 - decay_rate) * base.detach()
    return _project_norm(m_decayed, cfg)

# Guard: assert ‖m_t‖ >= eps_m (§9.3). Raises AssertionError on violation.
def assert_memory_norm(m: Tensor, cfg: FieldConfig) -> None:
    norm = float(m.norm().item())
    assert norm >= cfg.eps_m, f"§9.3 VIOLATION: ‖m_t‖={norm:.6f} < eps_m={cfg.eps_m}"
