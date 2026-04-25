import struct
import random
import threading
import os
import re
from pathlib import Path
import numpy as np # type:ignore
from multiprocessing import shared_memory
from o_info_agent import ASU_Agent
from o_thought_agent import Thought
from o_connection import Connector, ConnectionEndpoint
from s_synthesis import KnowledgeSynthesizer
import thought_process
from constants import (
    AGENT_POSITION_RECORD_BYTES,
    AGENT_NEAR_PARENT_WEIGHT,
    AGENT_SPAWN_JITTER,
    BOOTSTRAP_THOUGHT_ROUNDS,
    CONNECTION_RECORD_SIZE,
    CONNECTION_UTILITY_OFFSET,
    EXTRACTION_BLOCK_LIMIT,
    FINAL_ARGUMENT_LIMIT,
    MAX_AGENTS,
    MAX_CONNECTIONS,
    MAX_CONNECTIONS_PER_QUERY,
    SUBJECT_QUANT_EXACT_OFFSET,
    SUBJECT_TENSE_EXACT_OFFSET,
    SUBJECT_TRUTH_EXACT_OFFSET,
    PREDICATE_QUANT_EXACT_OFFSET,
    PREDICATE_TENSE_EXACT_OFFSET,
    PREDICATE_TRUTH_EXACT_OFFSET,
    STOPWORD_TOKENS,
)
from d_logic_extractor import find_connections
from d_word_info_map import literal_from_index
ARGUMENT_SELECTION_MODE = (
    os.getenv("ARGUMENT_SELECTION_MODE", "specificity").strip().lower()
    or "specificity"
)

from utils import (
    flush_conn_log as util_flush_conn_log,
    clear_report as util_clear_report,
    display_source,
    format_specifics,
    merge_specifics,
)

TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Brain:
    def __init__(self, shm_names):
        self.shm_pos         = shared_memory.SharedMemory(name=shm_names["pos"])
        self.shm_connections = shared_memory.SharedMemory(name=shm_names["connections"])
        self.shm_cmd         = shared_memory.SharedMemory(name=shm_names["command"])
        self.shm_report      = shared_memory.SharedMemory(name=shm_names["report"])
        self.shm_status      = shared_memory.SharedMemory(name=shm_names["status"])

        self.agents         = {}  # name -> ASU_Agent
        self.agents_by_idx  = {}  # idx  -> ASU_Agent
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
        self.current_tense_preference = "none"
        self.synthesizer    = KnowledgeSynthesizer()
        self.current_subqueries = []
        self.target_a_queries = []
        self.target_b_queries = []
        self.bridge_queries = []
        self.target_a_focus_phrases = []
        self.target_b_focus_phrases = []
        self.current_research_results = []
        self.target_seed_agents = []
        self.target_seed_indices = set()
        self.target_seed_specs = []
        self.thought_worker_threads = []
        self.thought_queue = None
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
        self.target_seed_indices = set()
        self.thoughts = []
        self.reset_text_reports()

    def prime_query_plan(self, query_plan):
        self.target_a_queries = self._normalize_seed_queries(
            (query_plan or {}).get("target_a_queries", [])
        )
        self.target_b_queries = self._normalize_seed_queries(
            (query_plan or {}).get("target_b_queries", [])
        )
        self.bridge_queries = self._normalize_seed_queries(
            (query_plan or {}).get("bridge_queries", [])
        )
        self.target_a_focus_phrases = self._normalize_seed_queries(
            (query_plan or {}).get("target_a_focus_phrases", [])
        )
        self.target_b_focus_phrases = self._normalize_seed_queries(
            (query_plan or {}).get("target_b_focus_phrases", [])
        )
        self.current_subqueries = self._normalize_seed_queries(
            self.target_a_queries + self.target_b_queries + self.bridge_queries
        )
        return list(self.current_subqueries)

    def record_research_result(self, task):
        query = task.get("query", "unknown")
        connections = task.get("connections", [])
        blocks = task.get("blocks", [])
        if blocks and not connections:
            print(
                f"[EXTRACT] Extracting top {min(len(blocks), EXTRACTION_BLOCK_LIMIT)} "
                f"relevant blocks for query: '{query}'"
            )
            connections = find_connections(
                blocks,
                query=query,
                connection_limit=MAX_CONNECTIONS_PER_QUERY,
            )
            print(f"[EXTRACT] Extracted {len(connections)} capped connections for query: '{query}'")
        self.current_research_results.append(
            {
                "query": query,
                "connections": connections,
                "blocks": len(blocks),
            }
        )
        return query, connections

    def ingest_research_connections(self, query, connections):
        upstream_by_source = {}
        for c in connections:
            subject_sp = c.get("subject")
            predicate_sp = c.get("predicate")
            relation_id = c.get("connection")
            if not isinstance(subject_sp, ConnectionEndpoint) or not isinstance(predicate_sp, ConnectionEndpoint):
                continue
            subject_name = subject_sp.asu_value()
            predicate_name = predicate_sp.asu_value()
            if not subject_name or not predicate_name or relation_id is None:
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
            )
            upstream_by_source[(source, predicate_name)] = subject_name

    def _merged_connection_specifics(
        self,
        existing=None,
        subject_specifics=None,
        predicate_specifics=None,
        connection_specifics=None,
    ):
        payload = dict(existing or {})
        payload["subject_specifics"] = merge_specifics(
            payload.get("subject_specifics"),
            subject_specifics,
        )
        payload["predicate_specifics"] = merge_specifics(
            payload.get("predicate_specifics"),
            predicate_specifics,
        )
        payload["connection_specifics"] = merge_specifics(
            payload.get("connection_specifics"),
            connection_specifics,
        )
        return payload

    def _codes_compatible(self, existing_code, incoming_code, unknown_code=-1):
        existing = int(existing_code if existing_code is not None else unknown_code)
        incoming = int(incoming_code if incoming_code is not None else unknown_code)
        return existing == unknown_code or incoming == unknown_code or existing == incoming

    def _prefer_specific_code(self, existing_code, incoming_code, unknown_code=-1):
        existing = int(existing_code if existing_code is not None else unknown_code)
        incoming = int(incoming_code if incoming_code is not None else unknown_code)
        if existing == unknown_code and incoming != unknown_code:
            return incoming
        return existing

    def _modifier_codes_compatible(self, existing_modifier_ids, incoming_modifier_ids):
        existing = tuple(int(value) for value in (existing_modifier_ids or ()))
        incoming = tuple(int(value) for value in (incoming_modifier_ids or ()))
        return not existing or not incoming or existing == incoming

    def _prefer_specific_modifiers(self, existing_modifier_ids, incoming_modifier_ids):
        existing = tuple(int(value) for value in (existing_modifier_ids or ()))
        incoming = tuple(int(value) for value in (incoming_modifier_ids or ()))
        if not existing and incoming:
            return incoming
        return existing

    def _state_for_key(self, key):
        state = self.connection_states.get(key, {})
        subject_sp = state.get("subject_sp")
        predicate_sp = state.get("predicate_sp")
        if isinstance(subject_sp, ConnectionEndpoint) and isinstance(predicate_sp, ConnectionEndpoint):
            return subject_sp, predicate_sp

        s_idx, o_idx, _rel_type = key[:3]
        offset = self.connection_offsets.get(key)
        if offset is not None:
            try:
                return (
                    ConnectionEndpoint(
                        quantifier=struct.unpack_from("<i", self.shm_connections.buf, offset + SUBJECT_QUANT_EXACT_OFFSET)[0],
                        tense=struct.unpack_from("<i", self.shm_connections.buf, offset + SUBJECT_TENSE_EXACT_OFFSET)[0],
                        truth=struct.unpack_from("<i", self.shm_connections.buf, offset + SUBJECT_TRUTH_EXACT_OFFSET)[0],
                        ASU_idx=s_idx,
                    ),
                    ConnectionEndpoint(
                        quantifier=struct.unpack_from("<i", self.shm_connections.buf, offset + PREDICATE_QUANT_EXACT_OFFSET)[0],
                        tense=struct.unpack_from("<i", self.shm_connections.buf, offset + PREDICATE_TENSE_EXACT_OFFSET)[0],
                        truth=struct.unpack_from("<i", self.shm_connections.buf, offset + PREDICATE_TRUTH_EXACT_OFFSET)[0],
                        ASU_idx=o_idx,
                    ),
                )
            except struct.error:
                pass
        return (
            ConnectionEndpoint(
                quantifier=-1,
                tense=-1,
                truth=-1,
                ASU_idx=s_idx,
            ),
            ConnectionEndpoint(
                quantifier=-1,
                tense=-1,
                truth=-1,
                ASU_idx=o_idx,
            ),
        )

    def _state_specificity(self, sp):
        if not isinstance(sp, ConnectionEndpoint):
            return 0
        return sum(
            1 for value in (sp.quantifier, sp.tense, sp.truth) if int(value) >= 0
        ) + (1 if tuple(sp.modifier_idx or ()) else 0)

    def _exact_connection_signature(self, s_idx, o_idx, rel_type, subject_sp, predicate_sp):
        return (
            int(s_idx),
            int(o_idx),
            int(rel_type),
            int(getattr(subject_sp, "quantifier", -1)),
            int(getattr(subject_sp, "tense", -1)),
            int(getattr(subject_sp, "truth", -1)),
            tuple(int(value) for value in getattr(subject_sp, "modifier_idx", ()) or ()),
            int(getattr(predicate_sp, "quantifier", -1)),
            int(getattr(predicate_sp, "tense", -1)),
            int(getattr(predicate_sp, "truth", -1)),
            tuple(int(value) for value in getattr(predicate_sp, "modifier_idx", ()) or ()),
        )

    def _connection_bucket_key(self, s_idx, o_idx, rel_type):
        return (int(s_idx), int(o_idx), int(rel_type))

    def _register_connection_key(self, key, offset=None):
        self.seen_connections.add(key)
        self.connection_buckets.setdefault(
            self._connection_bucket_key(*key[:3]),
            set(),
        ).add(key)
        if offset is not None:
            self.connection_offsets[key] = offset

    def _attach_connector(self, connector):
        seen = set()
        for agent in (connector.source_agent, connector.target):
            if agent is None or agent.index in seen:
                continue
            seen.add(agent.index)
            agent.connectors.append(connector)

    def _unregister_connection_key(self, key):
        self.seen_connections.discard(key)
        self.connection_offsets.pop(key, None)
        bucket_key = self._connection_bucket_key(*key[:3])
        bucket = self.connection_buckets.get(bucket_key)
        if bucket is not None:
            bucket.discard(key)
            if not bucket:
                self.connection_buckets.pop(bucket_key, None)

    def _merge_compatible_states(self, existing_subject, existing_predicate, incoming_subject, incoming_predicate):
        if not (
            self._codes_compatible(existing_subject.truth, incoming_subject.truth, -1)
            and self._codes_compatible(existing_predicate.truth, incoming_predicate.truth, -1)
            and self._codes_compatible(existing_subject.quantifier, incoming_subject.quantifier, -1)
            and self._codes_compatible(existing_predicate.quantifier, incoming_predicate.quantifier, -1)
            and self._codes_compatible(existing_subject.tense, incoming_subject.tense, -1)
            and self._codes_compatible(existing_predicate.tense, incoming_predicate.tense, -1)
            and self._modifier_codes_compatible(existing_subject.modifier_idx, incoming_subject.modifier_idx)
            and self._modifier_codes_compatible(existing_predicate.modifier_idx, incoming_predicate.modifier_idx)
        ):
            return None

        merged_subject = ConnectionEndpoint(
            quantifier=self._prefer_specific_code(existing_subject.quantifier, incoming_subject.quantifier, -1),
            tense=self._prefer_specific_code(existing_subject.tense, incoming_subject.tense, -1),
            truth=self._prefer_specific_code(existing_subject.truth, incoming_subject.truth, -1),
            ASU_idx=getattr(existing_subject, "ASU_idx", -1),
            modifier_idx=self._prefer_specific_modifiers(existing_subject.modifier_idx, incoming_subject.modifier_idx),
        )
        merged_predicate = ConnectionEndpoint(
            quantifier=self._prefer_specific_code(existing_predicate.quantifier, incoming_predicate.quantifier, -1),
            tense=self._prefer_specific_code(existing_predicate.tense, incoming_predicate.tense, -1),
            truth=self._prefer_specific_code(existing_predicate.truth, incoming_predicate.truth, -1),
            ASU_idx=getattr(existing_predicate, "ASU_idx", -1),
            modifier_idx=self._prefer_specific_modifiers(existing_predicate.modifier_idx, incoming_predicate.modifier_idx),
        )
        return merged_subject, merged_predicate

    def _find_compatible_connection_key(self, s_idx, o_idx, rel_type, subject_sp, predicate_sp):
        bucket = self.connection_buckets.get(self._connection_bucket_key(s_idx, o_idx, rel_type), set())
        best_key = None
        best_subject = None
        best_predicate = None
        best_specificity = -1
        for key in bucket:
            existing_subject, existing_predicate = self._state_for_key(key)
            merged_state = self._merge_compatible_states(
                existing_subject,
                existing_predicate,
                subject_sp,
                predicate_sp,
            )
            if merged_state is None:
                continue
            merged_subject, merged_predicate = merged_state
            specificity = self._state_specificity(existing_subject) + self._state_specificity(existing_predicate)
            if specificity > best_specificity:
                best_key = key
                best_subject = merged_subject
                best_predicate = merged_predicate
                best_specificity = specificity
        return best_key, best_subject, best_predicate

    def _rewrite_connection_record(self, key, subject_sp, predicate_sp):
        offset = self.connection_offsets.get(key)
        if offset is None:
            return key
        s_idx, o_idx, rel_type = key[:3]
        utility = self._read_connection_utility(offset)
        self._pack_connection_record(
            offset,
            s_idx,
            rel_type,
            o_idx,
            utility,
            subject_sp,
            predicate_sp,
        )
        new_key = self._exact_connection_signature(
            s_idx,
            o_idx,
            rel_type,
            subject_sp,
            predicate_sp,
        )
        if new_key == key:
            return key

        source = self.connection_sources.pop(key, None)
        state = self.connection_states.pop(key, None)
        specifics = self.connection_specifics.pop(key, None)
        previous_agents = self.connection_previous_agents.pop(key, None)
        evidence_text = self.connection_texts.pop(key, None)
        self._unregister_connection_key(key)
        self._register_connection_key(new_key, offset=offset)
        if source is not None:
            self.connection_sources[new_key] = source
        if state is not None:
            self.connection_states[new_key] = state
        if specifics is not None:
            self.connection_specifics[new_key] = specifics
        if previous_agents is not None:
            self.connection_previous_agents[new_key] = previous_agents
        if evidence_text is not None:
            self.connection_texts[new_key] = evidence_text
        return new_key

    def _read_connection_utility(self, offset):
        try:
            return float(
                struct.unpack_from(
                    "<f",
                    self.shm_connections.buf,
                    offset + CONNECTION_UTILITY_OFFSET,
                )[0]
            )
        except struct.error:
            return 0.0

    def _write_connection_utility(self, offset, utility):
        struct.pack_into(
            "<f",
            self.shm_connections.buf,
            offset + CONNECTION_UTILITY_OFFSET,
            max(0.0, min(1.0, float(utility))),
        )

    def _pack_connection_record(self, offset, s_idx, rel_type, o_idx, utility, subject_sp, predicate_sp):
        struct.pack_into(
            "<iiifiiiiii",
            self.shm_connections.buf,
            offset,
            int(s_idx),
            int(rel_type),
            int(o_idx),
            float(max(0.0, min(1.0, utility))),
            int(getattr(subject_sp, "quantifier", -1)),
            int(getattr(subject_sp, "tense", -1)),
            int(getattr(subject_sp, "truth", -1)),
            int(getattr(predicate_sp, "quantifier", -1)),
            int(getattr(predicate_sp, "tense", -1)),
            int(getattr(predicate_sp, "truth", -1)),
        )

    def _remap_connection_key(self, key, index_map):
        s_idx, o_idx, rel_type = key[:3]
        subject_quant, subject_tense, subject_truth, subject_modifiers = key[3:7]
        predicate_quant, predicate_tense, predicate_truth, predicate_modifiers = key[7:11]
        return (
            index_map.get(s_idx, s_idx),
            index_map.get(o_idx, o_idx),
            rel_type,
            subject_quant,
            subject_tense,
            subject_truth,
            tuple(subject_modifiers),
            predicate_quant,
            predicate_tense,
            predicate_truth,
            tuple(predicate_modifiers),
        )

    def _normalize_previous_agent_ids(self, previous_agent_ids=None):
        if previous_agent_ids is None:
            return ()
        if isinstance(previous_agent_ids, (int, np.integer)):
            values = [int(previous_agent_ids)]
        else:
            values = list(previous_agent_ids or [])

        normalized = []
        seen = set()
        for value in values:
            try:
                agent_id = int(value)
            except (TypeError, ValueError):
                continue
            if agent_id < 0 or agent_id in seen:
                continue
            seen.add(agent_id)
            normalized.append(agent_id)
        return tuple(sorted(normalized))

    def _merge_previous_agent_ids(self, existing=None, incoming=None):
        return self._normalize_previous_agent_ids(
            list(self._normalize_previous_agent_ids(existing))
            + list(self._normalize_previous_agent_ids(incoming))
        )

    def _merge_connection_metadata(
        self,
        key,
        source="unknown",
        specifics_payload=None,
        previous_agent_ids=None,
        evidence_text="",
        replace_source=False,
    ):
        if source and source != "unknown":
            existing_source = self.connection_sources.get(key)
            if replace_source or existing_source in (None, "", "unknown"):
                self.connection_sources[key] = source

        if specifics_payload is not None:
            self.connection_specifics[key] = self._merged_connection_specifics(
                self.connection_specifics.get(key),
                subject_specifics=specifics_payload.get("subject_specifics"),
                predicate_specifics=specifics_payload.get("predicate_specifics"),
                connection_specifics=specifics_payload.get("connection_specifics"),
            )

        merged_previous_agents = self._merge_previous_agent_ids(
            self.connection_previous_agents.get(key),
            previous_agent_ids,
        )
        if merged_previous_agents:
            self.connection_previous_agents[key] = merged_previous_agents

        clean_text = " ".join(str(evidence_text or "").strip().split())
        if clean_text:
            existing_text = " ".join(str(self.connection_texts.get(key, "") or "").strip().split())
            if replace_source or not existing_text:
                self.connection_texts[key] = clean_text

    def _store_connection_metadata(
        self,
        key,
        source="unknown",
        specifics_payload=None,
        previous_agent_ids=None,
        evidence_text="",
    ):
        self.connection_sources[key] = source
        self.connection_specifics[key] = dict(specifics_payload or {})
        self.connection_previous_agents[key] = self._normalize_previous_agent_ids(
            previous_agent_ids
        )
        clean_text = " ".join(str(evidence_text or "").strip().split())
        self.connection_texts[key] = clean_text

    def _specifics_suffix(
        self,
        subject_specifics=None,
        predicate_specifics=None,
        connection_specifics=None,
    ):
        parts = []
        subject_text = format_specifics(subject_specifics)
        predicate_text = format_specifics(predicate_specifics)
        relation_text = format_specifics(connection_specifics)
        if subject_text:
            parts.append(f"subject: {subject_text}")
        if relation_text:
            parts.append(f"relation: {relation_text}")
        if predicate_text:
            parts.append(f"predicate: {predicate_text}")
        return f" [{' ; '.join(parts)}]".replace(" ; ", "; ") if parts else ""

    def _argument_text_is_clean(self, text):
        text = str(text or "").strip().lower()
        if not text:
            return False
        if "http://" in text or "https://" in text or "\\url{" in text:
            return False
        if "= =" in text or text.startswith("=") or text.endswith("="):
            return False
        if "{" in text or "}" in text or "|" in text:
            return False
        return True

    def _is_generic_argument_text(self, text):
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return True
        return not [token for token in TARGET_TOKEN_RE.findall(normalized) if token not in STOPWORD_TOKENS]

    def _chain_richness_profile(self, chain):
        metrics = self._chain_detail_metrics(chain)
        return (
            metrics["specifics"],
            metrics["quantified"],
            metrics["modifiers"],
            metrics["specificity_steps"],
            metrics["informative_nodes"],
            metrics["detailed_nodes"],
            -metrics["generic_nodes"],
            len(self._chain_sources(chain)),
        )

    def _chain_detail_metrics(self, chain):
        metrics = {
            "specifics": 0,
            "quantified": 0,
            "modifiers": 0,
            "detailed_nodes": 0,
            "informative_nodes": 0,
            "generic_nodes": 0,
            "specificity_steps": 0,
        }

        for step in chain[1:]:
            if not isinstance(step, dict):
                continue

            step_specifics = 0
            for key in ("subject_specifics", "predicate_specifics", "connection_specifics"):
                values = step.get(key) or []
                step_specifics += len(values)
            metrics["specifics"] += step_specifics
            if step_specifics:
                metrics["specificity_steps"] += 1

            try:
                subject_quant = int(step.get("subject_quant", -1))
            except (TypeError, ValueError):
                subject_quant = -1
            if subject_quant >= 0:
                metrics["quantified"] += 1
            try:
                predicate_quant = int(step.get("predicate_quant", -1))
            except (TypeError, ValueError):
                predicate_quant = -1
            if predicate_quant >= 0:
                metrics["quantified"] += 1

            if str(step.get("subject_modifier") or "").strip():
                metrics["modifiers"] += 1
            if str(step.get("predicate_modifier") or "").strip():
                metrics["modifiers"] += 1

            for key in ("subject", "predicate"):
                text = str(step.get(key) or "").strip()
                if not text:
                    continue
                tokens = [token for token in self._target_tokens(text) if token]
                if len(tokens) >= 2:
                    metrics["detailed_nodes"] += 1
                if self._is_generic_argument_text(text):
                    metrics["generic_nodes"] += 1
                else:
                    metrics["informative_nodes"] += 1

        return metrics

    def _chain_specificity_profile(self, chain):
        metrics = self._chain_detail_metrics(chain)
        return (
            metrics["specifics"],
            metrics["specificity_steps"],
            metrics["detailed_nodes"],
            metrics["informative_nodes"],
            metrics["modifiers"],
            metrics["quantified"],
            -metrics["generic_nodes"],
            len(self._chain_sources(chain)),
        )

    def _argument_selection_profile(self, chain):
        if ARGUMENT_SELECTION_MODE == "richness":
            return self._chain_richness_profile(chain)
        return self._chain_specificity_profile(chain)

    def _chain_unique_targets(self, chain):
        targets = {
            str(step.get("predicate") or step.get("target") or "").strip().lower()
            for step in chain[1:]
            if isinstance(step, dict)
        }
        return len({target for target in targets if target})

    def _chain_sources(self, chain):
        sources = {
            str(step.get("source") or "").strip()
            for step in chain[1:]
            if isinstance(step, dict)
        }
        return tuple(sorted(source for source in sources if source and source != "unknown"))

    def _chain_signature(self, chain):
        return repr(chain)

    def _is_usable_argument_chain(self, chain):
        if not chain or len(chain) <= 1 or not isinstance(chain[0], dict):
            return False

        for entry in chain:
            if not isinstance(entry, dict):
                return False
            for key in ("node", "subject", "predicate", "target"):
                if key in entry and entry[key] and not self._argument_text_is_clean(entry[key]):
                    return False

        step_count = len(chain) - 1
        if self._chain_unique_targets(chain) < max(1, step_count // 2):
            return False
        return True

    def _target_tokens(self, text):
        return thought_process.target_tokens(text)

    def _lexical_target_score(self, agent_name, target_text):
        return thought_process.lexical_target_score(self, agent_name, target_text)

    def _normalize_seed_queries(self, queries):
        return thought_process.normalize_seed_queries(queries)

    def _target_a_seed_queries(self):
        return thought_process.target_a_seed_queries(self)

    def _target_b_completion_queries(self):
        return thought_process.target_b_completion_queries(self)

    def _agent_info_text(self, agent, limit=None):
        return thought_process.agent_info_text(self, agent, limit=limit)

    def _lexical_info_score(self, info_text, query_text):
        return thought_process.lexical_info_score(self, info_text, query_text)

    def select_target_seed_agents(self, limit=None):
        return thought_process.select_target_seed_agents(self, limit=limit)

    def refresh_target_seed_agents(self, limit=None):
        return thought_process.refresh_target_seed_agents(self, limit=limit)

    def load_local_information_payloads(self, folder="local_information"):
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Local information folder not found: {root}")

        text_paths = sorted(path for path in root.glob("*.txt") if path.is_file())
        if not text_paths:
            raise FileNotFoundError(f"No .txt files found in {root}")

        payloads = []
        for path in text_paths:
            try:
                content = path.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            blocks = [{"content": content, "tag": f"local_information|{path.name}"}]
            connections = find_connections(blocks, query=path.stem)
            payloads.append({"query": path.stem, "connections": connections})

        if not payloads:
            raise ValueError(f"No usable text content found in {root}")
        return payloads

    def select_argument_histories(self, entries, limit=FINAL_ARGUMENT_LIMIT):
        normalized = []
        for entry in entries:
            if isinstance(entry, Thought):
                normalized.append((entry.history, int(getattr(entry, "score", 0))))
            else:
                normalized.append((entry, 0))

        usable = [item for item in normalized if self._is_usable_argument_chain(item[0])]
        pool = usable if usable else normalized
        ranked = sorted(
            pool,
            key=lambda item: (
                self._argument_selection_profile(item[0]),
                len(item[0]),
                item[1],
                self._chain_unique_targets(item[0]),
                repr(item[0]),
            ),
            reverse=True,
        )

        unique_ranked = []
        seen_signatures = set()
        for chain, score in ranked:
            signature = self._chain_signature(chain)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_ranked.append((chain, score))

        selected = []
        seen_sources = set()
        remaining = list(unique_ranked)
        while remaining and len(selected) < limit:
            best_index = 0
            best_key = None
            for idx, (chain, score) in enumerate(remaining):
                sources = set(self._chain_sources(chain))
                new_sources = len(sources - seen_sources)
                priority = self._argument_selection_profile(chain)
                key = (
                    1 if new_sources > 0 else 0,
                    new_sources,
                    priority,
                    len(chain),
                    score,
                    self._chain_unique_targets(chain),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = idx
            chain, _score = remaining.pop(best_index)
            selected.append(chain)
            seen_sources.update(self._chain_sources(chain))

        return selected

    def completed_thought_histories(self):
        return thought_process.completed_thought_histories(self)

    def successful_thoughts(self):
        return thought_process.successful_thoughts(self)

    def thought_swarm_stats(self):
        return thought_process.thought_swarm_stats(self)

    def _thought_worker_loop(self, stop_event, generation, worker_idx):
        return thought_process.thought_worker_loop(self, stop_event, generation, worker_idx)

    def start_thought_workers(self, stop_event, worker_count=None):
        return thought_process.start_thought_workers(self, stop_event, worker_count=worker_count)

    def thought_workers_finished(self):
        return thought_process.thought_workers_finished(self)

    def stop_thought_workers(self):
        return thought_process.stop_thought_workers(self)

    def bootstrap_thought_histories(self, rounds=BOOTSTRAP_THOUGHT_ROUNDS, seed_limit=None):
        return thought_process.bootstrap_thought_histories(
            self,
            rounds=rounds,
            seed_limit=seed_limit,
        )

    def write_argument_report(self, active_histories, sampled_histories, is_final=False, successful_payloads=None):
        from reporting import write_argument_report as reporting_write_argument_report

        return reporting_write_argument_report(
            self,
            active_histories,
            sampled_histories,
            is_final=is_final,
            successful_payloads=successful_payloads,
        )

    def resolve_relation_id(self, rel_type=None):
        if rel_type is not None:
            return int(rel_type)
        return -1

    def _spawn_agent(self, name, near=None):
        """Create agent from semantic seed, with optional local bias if near is given."""
        name = name.strip().lower()
        if not name:
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
        off = self.next_idx * AGENT_POSITION_RECORD_BYTES
        struct.pack_into("ffffff", self.shm_pos.buf, off, pos[0], pos[1], pos[2], 0, 0, 0)
        self.next_idx += 1
        return agent, True

    def create_agent(self, name):
        agent, _ = self._spawn_agent(name)
        return agent

    def add_connection(
        self,
        s_name,
        o_name,
        rel_type=None,
        source="unknown",
        subject_sp=None,
        predicate_sp=None,
        subject_specifics=None,
        predicate_specifics=None,
        connection_specifics=None,
        evidence_text="",
        previous_agent_name=None,
        previous_agent_ids=None,
    ):
        s_agent, _ = self._spawn_agent(s_name)
        # Always spawn o near s — if s just spawned randomly, o goes near it;
        # if s already existed, o goes near its current position.
        o_agent, _ = self._spawn_agent(o_name, near=s_agent.pos if s_agent else None)
        if not s_agent or not o_agent:
            return
        if self.connection_count >= MAX_CONNECTIONS:
            return
        relation_id = self.resolve_relation_id(rel_type=rel_type)
        if relation_id < 0:
            return
        if not isinstance(subject_sp, ConnectionEndpoint):
            subject_sp = ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=s_name)
        if not isinstance(predicate_sp, ConnectionEndpoint):
            predicate_sp = ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=o_name)
        resolved_previous_agent_ids = list(
            self._normalize_previous_agent_ids(previous_agent_ids)
        )
        previous_agent_name = str(previous_agent_name or "").strip().lower()
        if previous_agent_name:
            previous_agent = self.agents.get(previous_agent_name)
            if previous_agent is not None:
                resolved_previous_agent_ids.append(previous_agent.index)
        resolved_previous_agent_ids = self._normalize_previous_agent_ids(
            resolved_previous_agent_ids
        )
        specifics_payload = self._merged_connection_specifics(
            None,
            subject_specifics=subject_specifics,
            predicate_specifics=predicate_specifics,
            connection_specifics=connection_specifics,
        )
        utility = Connector.utility_for(
            subject_sp,
            relation_id,
            predicate_sp,
            subject_specifics=specifics_payload.get("subject_specifics"),
            predicate_specifics=specifics_payload.get("predicate_specifics"),
            connection_specifics=specifics_payload.get("connection_specifics"),
            evidence_text=evidence_text,
        )

        exact_signature = self._exact_connection_signature(
            s_agent.index,
            o_agent.index,
            relation_id,
            subject_sp,
            predicate_sp,
        )
        key = exact_signature
        if key in self.connection_offsets:
            offset = self.connection_offsets.get(key)
            if offset is not None:
                merged_utility = max(self._read_connection_utility(offset), utility)
                self._write_connection_utility(offset, merged_utility)
            self.connection_states[key] = {
                "subject_sp": subject_sp,
                "predicate_sp": predicate_sp,
            }
            self._merge_connection_metadata(
                key,
                source=source,
                specifics_payload=specifics_payload,
                previous_agent_ids=resolved_previous_agent_ids,
                evidence_text=evidence_text,
            )
            ConnectionEndpoint.register_context(subject_sp.ASU_idx, evidence_text, source)
            ConnectionEndpoint.register_context(predicate_sp.ASU_idx, evidence_text, source)
            return

        compatible_key, merged_subject_sp, merged_predicate_sp = self._find_compatible_connection_key(
            s_agent.index,
            o_agent.index,
            relation_id,
            subject_sp,
            predicate_sp,
        )
        if compatible_key is not None:
            active_key = self._rewrite_connection_record(
                compatible_key,
                merged_subject_sp or subject_sp,
                merged_predicate_sp or predicate_sp,
            )
            offset = self.connection_offsets.get(active_key)
            if offset is not None:
                merged_utility = max(self._read_connection_utility(offset), utility)
                self._write_connection_utility(offset, merged_utility)
            self.connection_states[active_key] = {
                "subject_sp": merged_subject_sp or subject_sp,
                "predicate_sp": merged_predicate_sp or predicate_sp,
            }
            self._merge_connection_metadata(
                active_key,
                source=source,
                specifics_payload=specifics_payload,
                previous_agent_ids=resolved_previous_agent_ids,
                evidence_text=evidence_text,
            )
            ConnectionEndpoint.register_context(subject_sp.ASU_idx, evidence_text, source)
            ConnectionEndpoint.register_context(predicate_sp.ASU_idx, evidence_text, source)
            return

        self._store_connection_metadata(
            key,
            source=source,
            specifics_payload=specifics_payload,
            previous_agent_ids=resolved_previous_agent_ids,
            evidence_text=evidence_text,
        )
        self.connection_states[key] = {
            "subject_sp": subject_sp,
            "predicate_sp": predicate_sp,
        }
        ConnectionEndpoint.register_context(subject_sp.ASU_idx, evidence_text, source)
        ConnectionEndpoint.register_context(predicate_sp.ASU_idx, evidence_text, source)

        off = 4 + (self.connection_count * CONNECTION_RECORD_SIZE)
        self._pack_connection_record(
            off,
            s_agent.index,
            relation_id,
            o_agent.index,
            utility,
            subject_sp,
            predicate_sp,
        )

        connector = Connector(
            self.shm_connections.buf,
            off,
            self.agents_by_idx,
            subject_sp=subject_sp,
            predicate_sp=predicate_sp,
            source=source,
            evidence_text=evidence_text,
            subject_specifics=specifics_payload.get("subject_specifics"),
            predicate_specifics=specifics_payload.get("predicate_specifics"),
            connection_specifics=specifics_payload.get("connection_specifics"),
            previous_agent_ids=resolved_previous_agent_ids,
        )
        self._attach_connector(connector)

        self.connection_count += 1
        struct.pack_into("i", self.shm_connections.buf, 0, self.connection_count)
        self._register_connection_key(key, offset=off)

        relation_text = literal_from_index(relation_id) or str(relation_id)
        previous_labels = ", ".join(
            self.agents_by_idx[idx].ASU
            for idx in resolved_previous_agent_ids
            if idx in self.agents_by_idx
        ) or "-"
        subject_modifier_text = " ".join(subject_sp.modifier_value()).strip() or "-"
        predicate_modifier_text = " ".join(predicate_sp.modifier_value()).strip() or "-"
        self._conn_log.append(
            f"[CONN] {s_name} -> {o_name} | Type: {relation_id} ({relation_text}) | "
            f"S(Q{subject_sp.quantifier},T{subject_sp.tense},TR{subject_sp.truth}) | "
            f"P(Q{predicate_sp.quantifier},T{predicate_sp.tense},TR{predicate_sp.truth}) | "
            f"Utility[{utility:.3f}] | "
            f"Modifiers: S[{subject_modifier_text}] P[{predicate_modifier_text}] | "
            f"Specifics: S[{format_specifics(specifics_payload.get('subject_specifics')) or '-'}] "
            f"P[{format_specifics(specifics_payload.get('predicate_specifics')) or '-'}] "
            f"R[{format_specifics(specifics_payload.get('connection_specifics')) or '-'}] | "
            f"Prev[{previous_labels}] | "
            f"Source: {display_source(source)}\n"
        )
        if len(self._conn_log) >= 100:
            self.flush_conn_log()

    def remap_connection_sources(self, index_map):
        remapped = {}
        for key, source in self.connection_sources.items():
            key = self._remap_connection_key(key, index_map)
            existing = remapped.get(key)
            if existing in (None, "unknown") and source:
                remapped[key] = source
            elif existing is None:
                remapped[key] = source
        self.connection_sources = remapped

        remapped_states = {}
        for key, state in self.connection_states.items():
            key = self._remap_connection_key(key, index_map)
            remapped_states[key] = state
        self.connection_states = remapped_states

        remapped_specifics = {}
        for key, specifics in self.connection_specifics.items():
            key = self._remap_connection_key(key, index_map)
            remapped_specifics[key] = self._merged_connection_specifics(
                remapped_specifics.get(key),
                subject_specifics=(specifics or {}).get("subject_specifics"),
                predicate_specifics=(specifics or {}).get("predicate_specifics"),
            connection_specifics=(specifics or {}).get("connection_specifics"),
        )
        self.connection_specifics = remapped_specifics

        remapped_previous_agents = {}
        for key, previous_agent_ids in self.connection_previous_agents.items():
            key = self._remap_connection_key(key, index_map)
            mapped_ids = self._normalize_previous_agent_ids(
                index_map.get(agent_id, agent_id)
                for agent_id in (previous_agent_ids or ())
            )
            remapped_previous_agents[key] = self._merge_previous_agent_ids(
                remapped_previous_agents.get(key),
                mapped_ids,
            )
        self.connection_previous_agents = remapped_previous_agents

        remapped_texts = {}
        for key, evidence_text in self.connection_texts.items():
            key = self._remap_connection_key(key, index_map)
            if key not in remapped_texts or not remapped_texts[key]:
                remapped_texts[key] = str(evidence_text or "")
        self.connection_texts = remapped_texts

    def rebuild_connectors_from_shared_memory(self):
        for agent in self.agents.values():
            agent.connectors = []

        try:
            self.connection_count = struct.unpack_from("i", self.shm_connections.buf, 0)[0]
        except Exception:
            self.connection_count = 0

        self.seen_connections = set()
        self.connection_offsets = {}
        self.connection_buckets = {}
        for i in range(self.connection_count):
            off = 4 + (i * CONNECTION_RECORD_SIZE)
            try:
                s_idx, relation_id, o_idx, _utility, subject_quant_exact, subject_tense_exact, subject_truth_exact, predicate_quant_exact, predicate_tense_exact, predicate_truth_exact = struct.unpack_from("<iiifiiiiii", self.shm_connections.buf, off)
            except struct.error:
                break

            subject_sp = ConnectionEndpoint(subject_quant_exact, subject_tense_exact, subject_truth_exact, s_idx)
            predicate_sp = ConnectionEndpoint(predicate_quant_exact, predicate_tense_exact, predicate_truth_exact, o_idx)
            key = self._exact_connection_signature(
                s_idx,
                o_idx,
                relation_id,
                subject_sp,
                predicate_sp,
            )
            self._register_connection_key(key, offset=off)
            s_agent = self.agents_by_idx.get(s_idx)
            if s_agent is None:
                continue

            source = self.connection_sources.get(key, "unknown")
            state = self.connection_states.get(key, {})
            specifics = self.connection_specifics.get(key, {})
            previous_agent_ids = self.connection_previous_agents.get(key, ())
            evidence_text = self.connection_texts.get(key, "")
            connector = Connector(
                self.shm_connections.buf,
                off,
                self.agents_by_idx,
                subject_sp=state.get("subject_sp"),
                predicate_sp=state.get("predicate_sp"),
                source=source,
                evidence_text=evidence_text,
                subject_specifics=specifics.get("subject_specifics"),
                predicate_specifics=specifics.get("predicate_specifics"),
                connection_specifics=specifics.get("connection_specifics"),
                previous_agent_ids=previous_agent_ids,
            )
            self._attach_connector(connector)
            subject_sp = state.get("subject_sp") if isinstance(state.get("subject_sp"), ConnectionEndpoint) else subject_sp
            predicate_sp = state.get("predicate_sp") if isinstance(state.get("predicate_sp"), ConnectionEndpoint) else predicate_sp
            if key not in self.connection_states:
                self.connection_states[key] = {
                    "subject_sp": subject_sp,
                    "predicate_sp": predicate_sp,
                }

        self.next_idx = max(self.agents_by_idx.keys(), default=-1) + 1

    def run_final_synthesis(self):
        from reporting import run_final_synthesis as reporting_run_final_synthesis

        return reporting_run_final_synthesis(self)
