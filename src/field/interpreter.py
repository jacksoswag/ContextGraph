from __future__ import annotations
# Phase 4: TaskLoop as a policy interpreter over field states (spec §11, guide §3.9).
# The LLM reads field state, seeds x_0 from the query, decides continuation,
# triggers Sleep (consolidate.py), and verbalizes settled basins.
# This is the INTERPRETER layer — not the reasoner (field dynamics are) and not deleted.
import logging
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass
class FieldInterpretation:
    query: str
    basins: list[str]           # settled basin node ids
    basin_texts: list[str]      # verbalized basin summaries
    c_val: float                # C(τ) diagnostic for this episode
    steps_taken: int
    sleep_triggered: bool


# Policy interpreter: bridges the TaskLoop to the field system.
# Seeds the field, runs rollout, verbalizes settled basins for the LLM.
class FieldInterpreter:
    def __init__(self, conn, model, cfg, *, event_cb: Callable | None = None):
        self.conn = conn
        self.model = model
        self.cfg = cfg
        self.event_cb = event_cb
        self._m_prev = None     # cross-episode memory state

    # Seed x_0 from a query: use find_seed_nodes (reuse from retrieval.py).
    # Returns (episode, field_state) ready for rollout.
    def _seed_field(self, query: str):
        from runtime.retrieval import find_seed_nodes
        from field.spaces import materialize, init_state
        from field.memory import init_episode_memory
        seeds = find_seed_nodes(self.conn, query, top_k=8)
        if not seeds: return None, None
        node_ids = [nid for nid, _ in seeds]
        # 1-hop expansion for richer context
        extra: list[str] = []
        for nid in node_ids[:4]:
            for r in self.conn.execute(
                "SELECT source_id, target_id FROM edges WHERE source_id=? OR target_id=? LIMIT 6",
                (nid, nid)
            ):
                for ep_id in (r[0], r[1]):
                    if ep_id not in node_ids and ep_id not in extra and not ep_id.startswith("e_"):
                        extra.append(ep_id)
        all_ids = list(dict.fromkeys(node_ids + extra))[:32]
        ep = materialize(self.conn, all_ids, self.cfg)
        if ep.N == 0: return None, None
        seed_m = init_episode_memory(self._m_prev, None, self.cfg)
        state = init_state(ep, self.model.P, self.cfg, seed_m=seed_m)
        return ep, state

    # Verbalize settled basins as a human-readable summary.
    def _verbalize_basins(self, basin_ids: list[str]) -> list[str]:
        texts = []
        for nid in basin_ids[:12]:
            row = self.conn.execute("SELECT text FROM nodes WHERE id=?", (nid,)).fetchone()
            if row and row[0]: texts.append(row[0])
        return texts

    # Trigger Sleep (consolidate.py) — structural-invariance arm write (guide §0).
    def _maybe_sleep(self, sleep_triggered: bool) -> None:
        if not sleep_triggered: return
        try:
            from graph.consolidate import run_sleep
            run_sleep(self.conn, event_cb=self.event_cb)
        except Exception as exc:
            LOGGER.warning("Sleep consolidation failed: %s", exc)

    # Main interpret call: seed → rollout → extract basins → verbalize → maybe Sleep.
    # Returns FieldInterpretation for the LLM to act on.
    def interpret(self, query: str, *, do_learning: bool = False) -> FieldInterpretation | None:
        from field.dynamics import rollout, Actuators
        from field.objective import (ControlBusState, compute_components,
                                      update_control_bus, actuators_from_bus, scalar_c)
        from field.learn import FieldLearner
        ep, state = self._seed_field(query)
        if ep is None or state is None:
            return None
        # Rollout with C control bus driving actuators
        bus = ControlBusState()
        components_list = []
        visited: set = set(ep.node_ids)
        reach_prev = len(visited)
        # Single rollout pass (actuators computed from lagged bus)
        rr = rollout(ep=ep, init=state, model=self.model,
                     event_cb=self.event_cb)
        for t, res in enumerate(rr.results):
            comp = compute_components(res, rr.states[t].X, res.phi,
                                      reach_prev, visited, self.cfg)
            reach_prev = len(visited)
            components_list.append(comp)
            bus = update_control_bus(bus, comp, self.cfg)
        c_val = scalar_c(components_list, self.cfg)
        act_final = actuators_from_bus(bus, self.cfg)
        # Preserve memory for next episode (cross-episode policy from cfg)
        self._m_prev = rr.states[-1].m.detach()
        # Extract basins from final field energy
        final_X = rr.states[-1].X.detach()
        energies = final_X.norm(dim=-1)
        topk = min(8, ep.N)
        _, top_idx = energies.topk(topk)
        basin_ids = [ep.node_ids[i] for i in top_idx.tolist()]
        basin_texts = self._verbalize_basins(basin_ids)
        # θ-update when learning is enabled
        if do_learning:
            FieldLearner(self.model, self.cfg).update(rr, ep, components_list)
        # Sleep trigger from C control bus
        self._maybe_sleep(act_final.sleep)
        if self.event_cb:
            self.event_cb("field_interpret", {"query": query[:80], "c_val": round(c_val, 4),
                                               "steps": rr.steps, "basins": len(basin_ids),
                                               "sleep": act_final.sleep})
        return FieldInterpretation(query=query, basins=basin_ids, basin_texts=basin_texts,
                                   c_val=c_val, steps_taken=rr.steps,
                                   sleep_triggered=act_final.sleep)

    # Integration point: called by TaskLoop to augment context_needed with field basins.
    # Returns a list of strings suitable for adding to retrieval context.
    def augment_context(self, query: str) -> list[str]:
        interp = self.interpret(query)
        if interp is None: return []
        return interp.basin_texts
