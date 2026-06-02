from __future__ import annotations
import torch, pytest
from field.baseline import personalized_pagerank

# small weighted graph: 0-1-2-3 line + 0-2 shortcut
def _W():
    W = torch.zeros(4, 4)
    for i, j, w in [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 2, 0.5)]:
        W[i, j] = w; W[j, i] = w
    return W

def test_ppr_is_distribution():
    r = personalized_pagerank(_W(), [0])
    assert r.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert (r >= 0).all()

def test_ppr_seed_is_highest():
    r = personalized_pagerank(_W(), [0], alpha=0.15)
    assert int(r.argmax()) == 0                      # mass concentrates at the teleport seed

def test_ppr_mass_decays_with_distance():
    r = personalized_pagerank(_W(), [0], alpha=0.3)
    # node 3 (farthest from seed 0) gets less mass than nodes 1,2 (direct/2-hop)
    assert r[3] < r[1] and r[3] < r[2]

def test_ppr_deterministic():
    a = personalized_pagerank(_W(), [0]); b = personalized_pagerank(_W(), [0])
    assert torch.equal(a, b)

def test_ppr_higher_alpha_tightens():
    # more teleport ⇒ more mass stays on the seed
    lo = personalized_pagerank(_W(), [0], alpha=0.1)[0]
    hi = personalized_pagerank(_W(), [0], alpha=0.5)[0]
    assert float(hi) > float(lo)

def test_ppr_zero_degree_node_safe():
    W = torch.zeros(3, 3); W[0, 1] = W[1, 0] = 1.0   # node 2 isolated
    r = personalized_pagerank(W, [0])
    assert torch.isfinite(r).all() and r.sum().item() == pytest.approx(1.0, abs=1e-6)
