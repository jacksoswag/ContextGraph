from __future__ import annotations
# Phase 0 harness: wires tests/eval stress corpus to the field system.
# Scores answer quality per case, logs C(τ) as a diagnostic.
# C↔quality correlation is reported but never a gate (guide §6, §7 mandate).
import sys, os, logging, time
from pathlib import Path
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

# Resolve src on path (for standalone execution)
_ROOT = Path(__file__).resolve().parents[3]
for _p in [str(_ROOT / "src"), str(_ROOT / "tests")]:
    if _p not in sys.path: sys.path.insert(0, _p)

from field.config import FieldConfig, DEFAULT_CFG
from field.dynamics import FieldModel, rollout
from field.spaces import materialize, init_state
from field.objective import (CComponents, ControlBusState, compute_components,
                              update_control_bus, actuators_from_bus, scalar_c)
from field.memory import init_episode_memory
from field.learn import FieldLearner, LearnerConfig


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    basins: list[str]       # node ids of settled basins
    c_val: float            # C(τ) scalar diagnostic
    components: list        # per-step CComponents
    quality: float          # answer quality score (0–1)
    elapsed_s: float


# Quality score: embedding cosine similarity between expected targets and basin nodes,
# weighted by basin node specificity. Mirrors the retrieval.py scoring pattern:
# effective_score = cosine * (0.70 + 0.30 * specificity).
# A target counts as "hit" when any basin's effective_score >= HIT_THRESHOLD.
# Specificity weighting suppresses generic hub nodes (e.g. "data", "system") from
# falsely counting as hits; targets are embedded once, basin info_vectors reused.
# Falls back to substring match when the embedder is unavailable.
_HIT_THRESHOLD = 0.55   # effective cosine floor; conservative to avoid false positives
_DEFAULT_SPEC  = 0.5    # used when node_stats row is absent

def _quality_score(case: dict, basins: list[str], conn) -> float:
    expected = case.get("expected_targets", [])
    if not basins or not expected: return 0.0
    try:
        return _quality_score_vec(expected, basins, conn)
    except Exception:
        return _quality_score_substr(expected, basins, conn)

def _quality_score_vec(expected: list[str], basins: list[str], conn) -> float:
    import numpy as np
    from graph.vector import embed, unpack
    # Embed each expected target (short words/phrases — fast)
    target_vecs: list[tuple[str, np.ndarray]] = []
    for t in expected:
        blob = embed(t)
        if blob is not None: target_vecs.append((t, unpack(blob)))
    if not target_vecs: return _quality_score_substr(expected, basins, conn)
    # Load basin info_vectors + specificities in one query per chunk
    basin_data: list[tuple[np.ndarray, float]] = []
    for i in range(0, len(basins[:32]), 900):
        chunk = basins[i : i+900]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT n.info_vector, COALESCE(ns.specificity, {_DEFAULT_SPEC}) "
            f"FROM nodes n LEFT JOIN node_stats ns ON ns.node_id = n.id "
            f"WHERE n.id IN ({ph})", chunk
        ):
            if r[0]:
                basin_data.append((unpack(r[0]), float(r[1] if r[1] is not None else _DEFAULT_SPEC)))
    if not basin_data: return _quality_score_substr(expected, basins, conn)
    # Score: max effective cosine over basins per target; count hits
    hits = 0.0
    for _, t_vec in target_vecs:
        best = max(
            float(np.dot(t_vec, b_vec)) * (0.70 + 0.30 * spec)
            for b_vec, spec in basin_data
        )
        if best >= _HIT_THRESHOLD: hits += 1.0
    return hits / len(target_vecs)

def _quality_score_substr(expected: list[str], basins: list[str], conn) -> float:
    texts: set[str] = set()
    for nid in basins[:32]:
        row = conn.execute("SELECT text FROM nodes WHERE id=?", (nid,)).fetchone()
        if row: texts.add((row[0] or "").lower())
    hits = sum(1 for t in expected if any(t.lower() in tx for tx in texts))
    return hits / max(len(expected), 1)


# Seed the graph with memory_facts from a stress case (using GraphStore).
def _seed_graph(case: dict, store) -> None:
    if not case.get("memory_facts"): return
    for triple in case["memory_facts"]:
        if len(triple) < 3: continue
        src_txt, rel, tgt_txt = str(triple[0]), str(triple[1]), str(triple[2])
        store.add_triple(src_txt, rel, tgt_txt, source="harness", confidence=0.8)


# Run the field system on one case. Returns CaseResult.
def run_case(case: dict, conn, model: FieldModel, cfg: FieldConfig,
             learner: FieldLearner | None = None) -> CaseResult:
    from runtime.retrieval import find_seed_nodes
    t0 = time.perf_counter()
    prompt = case.get("prompt", "")
    seeds = find_seed_nodes(conn, prompt, top_k=8)
    if not seeds:
        return CaseResult(case_id=case["id"], prompt=prompt, basins=[], c_val=0.0,
                          components=[], quality=0.0, elapsed_s=time.perf_counter()-t0)
    node_ids = [nid for nid, _ in seeds]
    # Augment with 1-hop neighbors (up to 32 total)
    neighbor_ids: list[str] = []
    for nid in node_ids[:4]:
        for r in conn.execute("SELECT source_id, target_id FROM edges WHERE source_id=? OR target_id=? LIMIT 8", (nid, nid)):
            for ep_id in (r[0], r[1]):
                if ep_id not in (node_ids + neighbor_ids) and not ep_id.startswith("e_"):
                    neighbor_ids.append(ep_id)
    all_ids = list(dict.fromkeys(node_ids + neighbor_ids))[:32]
    ep = materialize(conn, all_ids, cfg)
    if ep.N == 0:
        return CaseResult(case_id=case["id"], prompt=prompt, basins=[], c_val=0.0,
                          components=[], quality=0.0, elapsed_s=time.perf_counter()-t0)
    # Init state from query vector as seed for memory
    seed_m = init_episode_memory(None, None, cfg)
    state = init_state(ep, model.P, cfg, seed_m=seed_m)
    # Rollout with C control bus
    bus = ControlBusState()
    actuators_seq, components_list = [], []
    visited: set = set(all_ids)
    reach_prev = len(visited)
    # First step uses default actuators
    rr = rollout(state, ep, model, actuators_seq=None)
    # Compute C components per step
    for t, res in enumerate(rr.results):
        x_prev = rr.states[t].X
        phi_prev = res.phi
        comp = compute_components(res, x_prev, phi_prev, reach_prev, visited, cfg)
        reach_prev = len(visited)
        components_list.append(comp)
        bus = update_control_bus(bus, comp, cfg)
    c_val = scalar_c(components_list, cfg)
    # θ-update (Phase 3, skipped in pure Phase 0 run)
    if learner is not None:
        learner.update(rr, ep, components_list)
    # Extract settled basins (top-energy nodes from final state)
    final_X = rr.states[-1].X.detach()
    energies = final_X.norm(dim=-1)    # [N] — use norm as basin energy proxy
    topk = min(8, ep.N)
    _, top_idx = energies.topk(topk)
    basins = [ep.node_ids[i] for i in top_idx.tolist()]
    quality = _quality_score(case, basins, conn)
    return CaseResult(case_id=case["id"], prompt=prompt, basins=basins, c_val=c_val,
                      components=components_list, quality=quality,
                      elapsed_s=time.perf_counter()-t0)


# Run the full stress corpus. Returns list of CaseResult + aggregate diagnostics.
def run_stress_corpus(conn, store, cfg: FieldConfig | None = None,
                      model: FieldModel | None = None,
                      with_learning: bool = False) -> dict:
    from eval.generate_stress_cases import build_rows
    if cfg is None: cfg = DEFAULT_CFG
    if model is None: model = FieldModel(cfg)
    learner = FieldLearner(model, cfg) if with_learning else None
    rows = build_rows()
    results: list[CaseResult] = []
    for case in rows:
        _seed_graph(case, store)
        try:
            r = run_case(case, conn, model, cfg, learner=learner)
            results.append(r)
            LOGGER.info("case=%s quality=%.2f C=%.4f", r.case_id, r.quality, r.c_val)
        except Exception as exc:
            LOGGER.warning("case=%s FAILED: %s", case["id"], exc)
    quality_vals = [r.quality for r in results]
    c_vals = [r.c_val for r in results]
    mean_q = sum(quality_vals) / max(len(quality_vals), 1)
    mean_c = sum(c_vals) / max(len(c_vals), 1)
    # C↔quality correlation (diagnostic only, never a gate)
    corr = _pearson(c_vals, quality_vals)
    return {"cases": len(results), "mean_quality": mean_q, "mean_c": mean_c,
            "c_quality_corr": corr, "results": results}


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2: return 0.0
    import math
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs) + 1e-12)
    dy = math.sqrt(sum((y-my)**2 for y in ys) + 1e-12)
    return num / (dx * dy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from graph.store import connect, GraphStore
    conn = connect()
    store = GraphStore(conn)
    report = run_stress_corpus(conn, store)
    print(f"cases={report['cases']} mean_quality={report['mean_quality']:.3f} "
          f"mean_C={report['mean_c']:.4f} C↔quality_corr={report['c_quality_corr']:.3f}")
