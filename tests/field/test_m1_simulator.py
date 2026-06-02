from __future__ import annotations
import torch
import torch.nn.functional as F
import pytest
from field.config import FieldConfig
from field.episode import Episode
from field.coupling import build as build_coupling
from field.energy import compute_E
from field.dynamics import step

# ── toy-graph fixtures ─────────────────────────────────────────────────────

def _clique_edges(nodes: list[int]) -> tuple[list[int], list[int]]:
    # both directions per undirected edge
    srcs, dsts = [], []
    for i in nodes:
        for j in nodes:
            if i != j: srcs.append(i); dsts.append(j)
    return srcs, dsts

def _episode(node_ids, A, srcs, dsts) -> Episode:
    ei = torch.stack([torch.tensor(srcs), torch.tensor(dsts)])
    return Episode(node_ids=node_ids, A=A, edge_index=ei,
                   id_to_idx={n: i for i, n in enumerate(node_ids)})

@pytest.fixture
def single_clique_ep():
    torch.manual_seed(0)
    N = 4
    base = F.normalize(torch.randn(1, 384), dim=-1)
    A = F.normalize(base.expand(N, -1) + torch.randn(N, 384) * 0.05, dim=-1)
    s, d = _clique_edges(list(range(N)))
    return _episode([f"n{i}" for i in range(N)], A, s, d)

@pytest.fixture
def two_clique_ep():
    torch.manual_seed(1)
    b1 = F.normalize(torch.randn(384), dim=-1)
    b2 = torch.randn(384); b2 = F.normalize(b2 - (b1 @ b2) * b1, dim=-1)
    A1 = F.normalize(b1.unsqueeze(0).expand(3, -1) + torch.randn(3, 384) * 0.05, dim=-1)
    A2 = F.normalize(b2.unsqueeze(0).expand(3, -1) + torch.randn(3, 384) * 0.05, dim=-1)
    A = torch.cat([A1, A2])
    s1, d1 = _clique_edges([0, 1, 2])
    s2, d2 = _clique_edges([3, 4, 5])
    srcs = s1 + s2 + [2, 3]; dsts = d1 + d2 + [3, 2]
    return _episode([f"n{i}" for i in range(6)], A, srcs, dsts)

def _init_X(N: int, d: int, R_max: float, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return F.normalize(torch.randn(N, d), dim=-1) * (R_max * 0.3)

# ── test 1: trapping max_i‖x_i‖ ≤ R_max for all t ────────────────────────

@pytest.mark.parametrize("ep_fix", ["single_clique_ep", "two_clique_ep"])
def test_trapping(ep_fix, request):
    ep = request.getfixturevalue(ep_fix)
    torch.manual_seed(20)
    cfg = FieldConfig(mu=0.1, eta=0.005, H_max=200)
    C = build_coupling(ep, cfg)
    X = _init_X(len(ep.node_ids), cfg.d, C.R_max)
    for t in range(cfg.H_max):
        X = step(X, C, cfg)
        mx = X.norm(dim=-1).max().item()
        assert mx <= C.R_max + 1e-4, \
            f"Trapping violated t={t}: ‖x‖_max={mx:.4f} > R_max={C.R_max:.4f}"

# ── test 2: E(t) monotone non-increasing (pure gradient flow) ─────────────

@pytest.mark.parametrize("ep_fix", ["single_clique_ep", "two_clique_ep"])
def test_energy_monotone(ep_fix, request):
    ep = request.getfixturevalue(ep_fix)
    cfg = FieldConfig(mu=0.1, eta=0.005, H_max=300)
    C = build_coupling(ep, cfg)
    X = _init_X(len(ep.node_ids), cfg.d, C.R_max)
    E_prev = compute_E(X, C, cfg).item()
    tol = 1e-5
    for t in range(cfg.H_max):
        X = step(X, C, cfg)
        E_curr = compute_E(X, C, cfg).item()
        assert E_curr <= E_prev + tol, \
            f"E increased t={t}: {E_prev:.8f}→{E_curr:.8f} Δ={E_curr-E_prev:.2e}"
        E_prev = E_curr

# ── test 3: no NaN/Inf over H_max steps (integrator stability) ────────────

@pytest.mark.parametrize("ep_fix", ["single_clique_ep", "two_clique_ep"])
def test_integrator_stability(ep_fix, request):
    ep = request.getfixturevalue(ep_fix)
    torch.manual_seed(30)
    cfg = FieldConfig(mu=0.1, eta=0.005, H_max=500)
    C = build_coupling(ep, cfg)
    X = _init_X(len(ep.node_ids), cfg.d, C.R_max)
    for t in range(cfg.H_max):
        X = step(X, C, cfg)
        assert not X.isnan().any(), f"NaN at step {t}"
        assert not X.isinf().any(), f"Inf at step {t}"
