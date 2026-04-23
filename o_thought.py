import numpy as np # type: ignore
import random
from o_ASU_agent import ASU_Agent
from constants import REL_IDENTITY, REL_CONDITIONAL, REL_SUBSET


class Thought: # A Thought that randomly traverses agents based on distance, recording as it goes
    def __init__(self, current_asu: ASU_Agent):
        self.current_asu = current_asu
        self.reset(current_asu)

    def reset(self, start_asu: ASU_Agent | None = None):
        if start_asu is not None:
            self.current_asu = start_asu
        self.last_asu_idx = -1 # Track the immediate previous node to prevent backtracking
        self.last_predicate_state = None
        self.history = [{"node": self.current_asu.ASU}] # Records the path of the Thought Agent

    def hop_is_compatible(self, conn):
        if self.last_predicate_state is None:
            return True
        return (
            conn.subject_truth == self.last_predicate_state["truth"]
            and conn.subject_tense == self.last_predicate_state["tense"]
        )

    def choose_next_hop(self): # Find connection weighted by distance/truth
        connections = self.current_asu.connectors 
        if not connections: return None

        current_pos = self.current_asu.pos
        valid_candidates = []
        weights = []
        
        for conn in connections:
            source_agent = conn.source_agent
            target = conn.target
            if source_agent is None or source_agent.index != self.current_asu.index:
                continue
            if target is None:
                continue
            if target.index == self.current_asu.index:
                continue
            if target.index == self.last_asu_idx:
                continue # Never jump directly back to the node we just came from
            if not self.hop_is_compatible(conn):
                continue
                
            dist = np.linalg.norm(current_pos - target.pos)
            dist = max(dist, 1e-6)
            
            valid_candidates.append(conn)

            # --- LOGICAL EXPRESSWAY ---
            if conn.conn_type in (REL_IDENTITY, REL_CONDITIONAL, REL_SUBSET):
                weights.append(100.0 if conn.truth else 0.01)
            else:
                # Standard associative weighting based on 3D physics distance
                weights.append((1.0 / dist) if conn.truth else dist)
        
        if not valid_candidates: return None
        
        # Return the connection itself, not just the target agent
        chosen_conn = random.choices(valid_candidates, weights=weights, k=1)[0]
        return chosen_conn

    def move(self): # Moves the agent and records the connection to history
        conn = self.choose_next_hop()
        if conn:
            self.last_asu_idx = self.current_asu.index # Store where we are before moving
            self.current_asu = conn.target
            self.last_predicate_state = {
                "truth": conn.truth,
                "tense": conn.predicate_tense,
            }
            self.history.append({
                "subject": conn.source_agent.ASU if conn.source_agent is not None else "",
                "subject_truth": conn.subject_truth,
                "subject_tense": conn.subject_tense,
                "subject_quant": conn.subject_quant,
                "relation_id": conn.conn_type,
                "relation_label": conn.relation_label,
                "truth": conn.truth,
                "predicate": self.current_asu.ASU,
                "predicate_truth": conn.predicate_truth,
                "predicate_tense": conn.predicate_tense,
                "predicate_quant": conn.predicate_quant,
                "source": conn.source,
            })
            return True
        return False
