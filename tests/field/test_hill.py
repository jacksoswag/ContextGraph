from __future__ import annotations
# The hill: node-wise genericity decay. The field dynamics are unchanged — these verify (1) the
# per-node leak grades the climb by genericity and (2) the gradient stays Lyapunov-safe.
# containing_edges membership view is tested separately (it's a store API, not field physics).
import dataclasses, os, tempfile
import torch, pytest
from graph.writer import GraphWriter, node_id, edge_id
from graph import GraphStore
from field.config import DEFAULT_CFG
from field.coupling import build as build_coupling, Coupling
from field.energy import grad_E, compute_E
from field.gather import materialize, gather
from field.harness import make_single_clique

def _node(t): return {"type": "node", "text": t, "pos": "NOUN"}
def _edge(s, r, t): return {"type": "edge", "rel": r, "source": s, "target": t,
                            "_source_text": "x", "_clause_text": "x"}
def _cfg(**kw): return dataclasses.replace(DEFAULT_CFG, **{"N_max": 200, "H_max": 2000, **kw})

# depth-4 nested spine: public trust [ journal publish [ scientist believe [ smoking cause cancer ] ] ]
_CHAIN = _edge(_node("public"), "trust", _edge(_node("journal"), "publish",
          _edge(_node("scientist"), "believe", _edge(_node("smoking"), "cause", _node("cancer")))))
_E1 = edge_id(node_id("smoking"), "cause", node_id("cancer"))

def _store(clauses) -> str:
    path = os.path.join(tempfile.mkdtemp(), "h.sqlite")
    with GraphWriter(path) as w:
        for c in clauses: w.write_clauses([c])
    return path

@pytest.fixture(scope="module")
def chain_store():
    with GraphStore(_store([_CHAIN])) as s: yield s

# ── GraphStore API: containing_edges is the membership (e_) view of incidence ────────────────

def test_containing_edges_is_membership_view(chain_store):
    cont = [e for e, _ in chain_store.containing_edges(node_id("cancer"))]
    nbr = [n for n, _ in chain_store.neighbors(node_id("cancer"))]
    assert _E1 in cont                       # cancer is a member of the fact (smoking cause cancer)
    assert node_id("smoking") in nbr         # ... and dyadically adjacent to its co-endpoint
    assert _E1 not in nbr                     # the two views are distinct

# ── node-wise genericity decay ──────────────────────────────────────────────────────────

# off by default: no degrees / decay_gamma=0 ⇒ scalar leak (exact backward-compat path).
def test_decay_uniform_when_off():
    assert build_coupling(make_single_clique(n=4), _cfg(decay_gamma=1.0)).decay_vec is None  # no degree
    ep = dataclasses.replace(make_single_clique(n=4), degree=torch.tensor([1., 2., 3., 4.]))
    assert build_coupling(ep, _cfg(decay_gamma=0.0)).decay_vec is None                       # gamma 0

# generic (high-degree) nodes leak faster; the lowest λ equals the scalar floor cfg.decay.
def test_decay_hub_leaks_more():
    ep = dataclasses.replace(make_single_clique(n=4), degree=torch.tensor([0., 5., 20., 100.]))
    C = build_coupling(ep, _cfg(decay_gamma=1.0, eta=0.005))
    dv = C.decay_vec
    assert dv is not None and float(dv[0]) == pytest.approx(DEFAULT_CFG.decay)   # deg 0 ⇒ floor
    assert float(dv[3]) > float(dv[2]) > float(dv[1]) > float(dv[0])             # monotone in degree

# the per-node decay gradient is analytically correct (finite-difference of the scalar energy).
def test_decay_grad_matches_finite_difference():
    cfg = _cfg(decay_gamma=1.0, eta=0.005)
    ep = dataclasses.replace(make_single_clique(n=5), degree=torch.tensor([1., 4., 9., 30., 80.]))
    C = build_coupling(ep, cfg)
    Cd = Coupling(C.sym.double(), C.lambda_max, C.R_max, C.L, C.eta_bound, C.decay_vec.double())
    N = len(ep.node_ids)
    torch.manual_seed(1); X = torch.randn(N, cfg.d, dtype=torch.float64) * 0.3
    g_analytic = grad_E(X, Cd, cfg)
    eps = 1e-6; g_num = torch.zeros_like(X)
    for i in range(N):
        for k in range(cfg.d):
            Xp = X.clone(); Xp[i, k] += eps; Xm = X.clone(); Xm[i, k] -= eps
            g_num[i, k] = (compute_E(Xp, Cd, cfg) - compute_E(Xm, Cd, cfg)) / (2 * eps)
    assert torch.allclose(g_analytic, g_num, atol=1e-4), \
        f"max |Δ|={float((g_analytic-g_num).abs().max()):.2e}"
