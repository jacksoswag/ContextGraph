import struct
import random
import threading
import numpy as np # type:ignore
from multiprocessing import shared_memory
from o_info_agent import ASU_Agent
from o_connection import ConnectionEndpoint
from s_synthesis import KnowledgeSynthesizer
import p_connection_graph
import p_active_subgraph
import reporting
import thought_process
from constants import (
    AGENT_POSITION_RECORD_BYTES,
    AGENT_NEAR_PARENT_WEIGHT,
    AGENT_SPAWN_JITTER,
    EXTRACTION_BLOCK_LIMIT,
    MAX_MERGE_SIMILARITY,
    MAX_AGENTS,
    MAX_CONNECTIONS_PER_QUERY,
    MIN_MERGE_SIMILARITY,
)
from d_logic_extractor import find_connections
from d_noise_cleanup import (
    clean_agent_name,
    expand_clean_connection_record,
    is_usable_agent_text,
)
from d_word_info_map import precache_text_vectors, str_to_vector, vector_specificity

from utils import (
    flush_conn_log as util_flush_conn_log,
    clear_report as util_clear_report,
)

class Brain:
    def __init__(self, shm_names):
        self.shm_pos         = shared_memory.SharedMemory(name=shm_names["pos"])
        self.shm_connections = shared_memory.SharedMemory(name=shm_names["connections"])
        self.shm_cmd         = shared_memory.SharedMemory(name=shm_names["command"])
        self.shm_report      = shared_memory.SharedMemory(name=shm_names["report"])
        self.shm_status      = shared_memory.SharedMemory(name=shm_names["status"])

        self.agents         = {}  # name -> ASU_Agent
        self.agents_by_idx  = {}  # idx  -> ASU_Agent
        self.merge_candidate_buckets = {}
        self.thoughts       = []
        self.next_idx       = 0
        self.connection_count = 0
        self.seen_connections = set()  # exact connection signature
        self.connection_offsets = {}  # connection key -> shm offset
        self.connection_buckets = {}  # endpoint/relation bucket -> set(exact keys)
        self.connection_sources = {}  # connection key -> source tag
        self.connection_states = {}  # connection key -> full subject/predicate composed state
        self.connection_specifics = {}  # connection key -> preserved specific qualifiers
        self.connection_previous_agents = {}  # connection key -> upstream prior agent ids
        self.connection_texts = {}  # connection key -> original clause/evidence text
        self.last_cmd_id      = ""
        self.current_command_id = ""
        self.current_target_a = ""
        self.current_target_b = ""
        self.current_target   = ""
        self.synthesizer    = KnowledgeSynthesizer()
        self.current_subqueries = []
        self.target_a_queries = []
        self.target_b_queries = []
        self.bridge_queries = []
        self.target_a_focus_phrases = []
        self.target_b_focus_phrases = []
        self.current_research_results = []
        self.target_seed_agents = []
        self.target_seed_specs = []
        self.target_goal_agents = []
        self.target_thought_routes = []
        self.anchor_pair_diagnostics = []
        self.thought_process = None
        self.thought_input_path = ""
        self.thought_output_path = ""
        self.thought_results_loaded = False
        self.thought_expected_count = 0
        self.thought_active_count = 0
        self.thought_generation = 0
        self.thought_phase_started = False
        self._thought_lock = threading.Lock()

        self._conn_log      = []  # batched connection log lines
        self.reset_text_reports()
        self.finalized_command_ids = set()

    def reset_text_reports(self):
        self._conn_log.clear()
        with open("connection_report.txt", "w") as f:
            f.write("--- CONNECTION DEBUG REPORT ---\n")
        with open("argument_report.txt", "w") as f:
            f.write("--- THOUGHT CHAIN ARGUMENT REPORT ---\n")

    def flush_conn_log(self):
        util_flush_conn_log(self)

    def clear_report(self):
        util_clear_report(self)

    def reset_research_state(self):
        self.stop_thought_workers()
        self.current_subqueries = []
        self.target_a_queries = []
        self.target_b_queries = []
        self.bridge_queries = []
        self.target_a_focus_phrases = []
        self.target_b_focus_phrases = []
        self.current_research_results = []
        self.target_seed_specs = []
        self.target_seed_agents = []
        self.target_goal_agents = []
        self.target_thought_routes = []
        self.anchor_pair_diagnostics = []
        self.refinement_pass_queued = False
        self.thoughts = []
        self.reset_text_reports()

    def prime_query_plan(self, query_plan):
        self.target_a_queries = thought_process.normalize_seed_queries(
            (query_plan or {}).get("target_a_queries", [])
        )
        self.target_b_queries = thought_process.normalize_seed_queries(
            (query_plan or {}).get("target_b_queries", [])
        )
        self.bridge_queries = thought_process.normalize_seed_queries(
            (query_plan or {}).get("bridge_queries", [])
        )
        self.target_a_focus_phrases = thought_process.normalize_seed_queries(
            (query_plan or {}).get("target_a_focus_phrases", [])
        )
        self.target_b_focus_phrases = thought_process.normalize_seed_queries(
            (query_plan or {}).get("target_b_focus_phrases", [])
        )
        self.current_subqueries = thought_process.normalize_seed_queries(
            self.target_a_queries + self.target_b_queries + self.bridge_queries
        )
        return list(self.current_subqueries)

    def record_research_result(self, task):
        query = task.get("query", "unknown")
        research_pass = task.get("research_pass", "primary")
        connections = task.get("connections", [])
        blocks = task.get("blocks", [])
        if blocks and not connections and not task.get("extracted"):
            extraction_blocks = blocks[:EXTRACTION_BLOCK_LIMIT]
            print(
                f"[EXTRACT:{research_pass}] Extracting {len(extraction_blocks)}/{len(blocks)} "
                f"blocks for query: '{query}'"
            )
            connections = find_connections(
                extraction_blocks,
                query=query,
                connection_limit=MAX_CONNECTIONS_PER_QUERY,
            )
            print(
                f"[EXTRACT:{research_pass}] Extracted {len(connections)} connections "
                f"(cap {MAX_CONNECTIONS_PER_QUERY}) for query: '{query}'"
            )
        self._conn_log.append(
            f"[QUERY:{research_pass}] {query} | Blocks[{len(blocks)}] | Connections[{len(connections)}]\n"
        )
        self.flush_conn_log()
        self.current_research_results.append(
            {
                "query": query,
                "research_pass": research_pass,
                "connections": connections,
                "blocks": len(blocks),
            }
        )
        return query, connections

    def _merge_bucket_keys(self, name, vector):
        keys = []
        clean = clean_agent_name(name).strip().lower()
        for token in clean.split()[:6]:
            if len(token) > 2:
                keys.append(f"token:{token}")

        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.size:
            top_count = min(8, int(vec.size))
            top_indices = np.argsort(np.abs(vec))[-top_count:]
            for idx in top_indices:
                sign = 1 if float(vec[int(idx)]) >= 0.0 else -1
                keys.append(f"vector:{int(idx)}:{sign}")
        return keys

    def _register_merge_candidate(self, agent):
        name = str(getattr(agent, "ASU", "") or "").strip().lower()
        if not name:
            return
        vector = np.asarray(getattr(agent, "info_vector", []), dtype=np.float32).reshape(-1)
        for key in self._merge_bucket_keys(name, vector):
            self.merge_candidate_buckets.setdefault(key, set()).add(name)

    def _merge_candidate_names(self, name, vector):
        names = set()
        for key in self._merge_bucket_keys(name, vector):
            names.update(self.merge_candidate_buckets.get(key, ()))
        if not names or len(self.agents) <= 256:
            return list(self.agents)
        return list(names)

    def _resolve_existing_or_mergeable_agent_name(self, name):
        clean = clean_agent_name(name).strip().lower()
        if not clean or not is_usable_agent_text(clean):
            return ""
        if clean in self.agents:
            return clean

        incoming = np.asarray(str_to_vector(clean), dtype=np.float32).reshape(-1)
        if incoming.size == 0:
            return ""

        candidate_names = []
        candidate_vectors = []
        for agent_name in self._merge_candidate_names(clean, incoming):
            agent = self.agents.get(agent_name)
            if agent is None:
                continue
            existing = np.asarray(getattr(agent, "info_vector", []), dtype=np.float32).reshape(-1)
            if existing.size == 0 or existing.shape != incoming.shape:
                continue
            candidate_names.append(agent_name)
            candidate_vectors.append(existing)

        if not candidate_vectors:
            return ""

        matrix = np.vstack(candidate_vectors).astype(np.float32, copy=False)
        incoming_norm = float(np.linalg.norm(incoming))
        candidate_norms = np.linalg.norm(matrix, axis=1)
        valid_norms = candidate_norms > 1e-12
        if incoming_norm <= 1e-12 or not np.any(valid_norms):
            return ""

        scores = np.full(len(candidate_names), -1.0, dtype=np.float32)
        scores[valid_norms] = (
            np.matmul(matrix[valid_norms], incoming)
            / (candidate_norms[valid_norms] * incoming_norm)
        )

        dimension = max(1, int(matrix.shape[1]))
        magnitudes = candidate_norms
        nonzero_ratio = np.count_nonzero(np.abs(matrix) > 1e-8, axis=1) / float(dimension)
        magnitude_signal = magnitudes / (magnitudes + 1.0)
        sparsity_signal = 1.0 - nonzero_ratio
        candidate_specificity = np.clip(
            (0.8 * magnitude_signal) + (0.2 * sparsity_signal),
            0.0,
            1.0,
        )
        incoming_specificity = vector_specificity(incoming)
        thresholds = MIN_MERGE_SIMILARITY + (
            (MAX_MERGE_SIMILARITY - MIN_MERGE_SIMILARITY)
            * np.maximum(incoming_specificity, candidate_specificity)
        )

        eligible_scores = np.where(scores >= thresholds, scores, -1.0)
        best_index = int(np.argmax(eligible_scores))
        best_score = float(eligible_scores[best_index])
        if best_score < 0.0:
            return ""

        best_name = candidate_names[best_index]
        best_threshold = float(thresholds[best_index])

        if best_name:
            self._conn_log.append(
                f"[REFINE-MERGE] {clean} -> {best_name} | "
                f"Similarity[{best_score:.3f}] Threshold[{best_threshold:.3f}]\n"
            )
        return best_name

    def ingest_research_connections(self, query, connections, restrict_to_existing=False):
        upstream_by_source = {}
        resolution_cache = {}
        starting_count = self.connection_count
        attempted = 0
        kept = 0
        skipped = 0
        cleaned_batch = []
        for raw_connection in connections:
            cleaned_connections = expand_clean_connection_record(raw_connection)
            if not cleaned_connections:
                continue
            for c in cleaned_connections:
                cleaned_batch.append(c)

        vector_warmup_names = []
        for c in cleaned_batch:
            subject_sp = c.get("subject")
            predicate_sp = c.get("predicate")
            if isinstance(subject_sp, ConnectionEndpoint):
                subject_name = subject_sp.asu_value()
                if subject_name:
                    vector_warmup_names.append(subject_name)
            if isinstance(predicate_sp, ConnectionEndpoint):
                predicate_name = predicate_sp.asu_value()
                if predicate_name:
                    vector_warmup_names.append(predicate_name)
        if vector_warmup_names:
            precache_text_vectors(vector_warmup_names)

        for c in cleaned_batch:
            attempted += 1
            subject_sp = c.get("subject")
            predicate_sp = c.get("predicate")
            relation_id = c.get("connection")
            if not isinstance(subject_sp, ConnectionEndpoint) or not isinstance(predicate_sp, ConnectionEndpoint):
                skipped += 1
                continue
            subject_name = subject_sp.asu_value()
            predicate_name = predicate_sp.asu_value()
            if not subject_name or not predicate_name or relation_id is None:
                skipped += 1
                continue

            if restrict_to_existing:
                if subject_name not in resolution_cache:
                    resolution_cache[subject_name] = self._resolve_existing_or_mergeable_agent_name(subject_name)
                if predicate_name not in resolution_cache:
                    resolution_cache[predicate_name] = self._resolve_existing_or_mergeable_agent_name(predicate_name)
                subject_name = resolution_cache[subject_name]
                predicate_name = resolution_cache[predicate_name]
                if not subject_name or not predicate_name or subject_name == predicate_name:
                    skipped += 1
                    continue

            source = c.get("source", query)
            previous_agent_name = upstream_by_source.get((source, subject_name))
            self.add_connection(
                subject_name,
                predicate_name,
                rel_type=relation_id,
                source=source,
                subject_sp=subject_sp,
                predicate_sp=predicate_sp,
                subject_specifics=c.get("subject_specifics"),
                predicate_specifics=c.get("predicate_specifics"),
                connection_specifics=c.get("connection_specifics"),
                evidence_text=c.get("text", ""),
                previous_agent_name=previous_agent_name,
                create_agents=not restrict_to_existing,
            )
            upstream_by_source[(source, predicate_name)] = subject_name
            kept += 1
        if restrict_to_existing:
            print(
                f"[INGEST:refine_existing] Kept {kept}/{attempted} cleaned connections "
                f"for query '{query}' without creating new agents; skipped {skipped}."
            )
        if self.connection_count != starting_count or self._conn_log:
            self.flush_conn_log()

    completed_thought_histories = thought_process.completed_thought_histories
    successful_thoughts = thought_process.successful_thoughts
    thought_swarm_stats = thought_process.thought_swarm_stats
    start_thought_workers = thought_process.start_thought_workers
    thought_workers_finished = thought_process.thought_workers_finished
    stop_thought_workers = thought_process.stop_thought_workers
    bootstrap_thought_histories = thought_process.bootstrap_thought_histories
    write_argument_report = reporting.write_argument_report

    def _spawn_agent(self, name, near=None):
        """Create agent from semantic seed, with optional local bias if near is given."""
        name = clean_agent_name(name).strip().lower()
        if not name:
            return None, False
        if not is_usable_agent_text(name):
            return None, False
        if name in self.agents:
            return self.agents[name], False
        if self.next_idx >= MAX_AGENTS:
            return None, False

        seed_agent = ASU_Agent(self.next_idx, ASU=name)
        semantic_pos = seed_agent.semantic_seed_position()
        jitter = np.array(
            [random.uniform(-AGENT_SPAWN_JITTER, AGENT_SPAWN_JITTER) for _ in range(3)],
            dtype=np.float32,
        )
        if near is not None:
            near_vec = np.asarray(near, dtype=np.float32).reshape(3)
            semantic_weight = 1.0 - float(AGENT_NEAR_PARENT_WEIGHT)
            pos = (
                (semantic_pos * semantic_weight)
                + (near_vec * float(AGENT_NEAR_PARENT_WEIGHT))
                + jitter
            )
        else:
            pos = semantic_pos + jitter

        agent = ASU_Agent(self.next_idx, ASU=name, pos=pos, asu_info_ref=seed_agent.asu_info_ref)
        self.agents[name] = agent
        self.agents_by_idx[self.next_idx] = agent
        self._register_merge_candidate(agent)
        off = self.next_idx * AGENT_POSITION_RECORD_BYTES
        struct.pack_into("ffffff", self.shm_pos.buf, off, pos[0], pos[1], pos[2], 0, 0, 0)
        self.next_idx += 1
        return agent, True

    add_connection = p_connection_graph.add_connection
    prepare_active_subgraph = p_active_subgraph.prepare_active_subgraph
    run_final_synthesis = reporting.run_final_synthesis
