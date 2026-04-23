import json
import time
import struct
import random
import threading
import os
import re
from pathlib import Path
import numpy as np # type:ignore
from multiprocessing import shared_memory
from queue import Empty
from o_ASU_agent import ASU_Agent
from o_thought import Thought
from o_connection import Connector, encode_connection_flags
from o_composed_sp import composed_sp
from s_synthesis import KnowledgeSynthesizer
from constants import PHASE_IDLE, PHASE_RESEARCH, PHASE_EXPORT, PHASE_EXPORTED, PHASE_IMPORT, PHASE_IMPORTED, PHASE_STABLE, LOGICAL_CONNECTORS
import d_query_expander
from d_logic_extractor import find_connections
from utils import normalize_relation_label, parse_command_payload, remove_shm_from_resource_tracker, flush_conn_log as util_flush_conn_log, clear_report as util_clear_report

TARGET_SEED_LIMIT = 20
TARGET_SEED_SIMILARITY = 0.60
TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORD_TOKENS = {
    "a", "an", "and", "area", "as", "at", "by", "for", "from", "history", "in",
    "is", "of", "on", "or", "the", "to", "with",
}

class Brain:
    def __init__(self, shm_names):
        self.shm_pos         = shared_memory.SharedMemory(name=shm_names["sks_pos"])
        self.shm_connections = shared_memory.SharedMemory(name=shm_names["sks_connections"])
        self.shm_cmd         = shared_memory.SharedMemory(name=shm_names["sks_command"])
        self.shm_report      = shared_memory.SharedMemory(name=shm_names["sks_report"])
        self.shm_status      = shared_memory.SharedMemory(name=shm_names["sks_status"])

        self.agents         = {}  # name -> ASU_Agent
        self.agents_by_idx  = {}  # idx  -> ASU_Agent
        self.thoughts       = []
        self.next_idx       = 0
        self.connection_count = 0
        self.seen_connections = set()  # (s_idx, o_idx, rel_type, flags)
        self.connection_sources = {}  # connection key -> source tag
        self.last_cmd_id      = ""
        self.current_command_id = ""
        self.current_goal     = ""
        self.current_target   = ""
        self.local_only_mode  = False
        self.relation_index_path = "verb_index.json"
        # creates clean lookup table from verb_index.json, or creates verb_index.json if it doesn't exist
        self.relation_labels  = self.load_relation_index()
        self.relation_to_id   = {normalize_relation_label(label): idx for idx, label in enumerate(self.relation_labels)}
        self.synthesizer    = KnowledgeSynthesizer()
        self.embedding_model = None
        self.target_seed_agents = []
        self.target_seed_indices = set()

        self._conn_log      = []  # batched connection log lines
        with open("connection_report.txt", "w") as f:
            f.write("--- CONNECTION DEBUG REPORT ---\n")
        with open("argument_report.txt", "w") as f:
            f.write("--- THOUGHT CHAIN ARGUMENT REPORT ---\n")
        self.write_relation_index()
        self.finalized_command_ids = set()

    def flush_conn_log(self):
        util_flush_conn_log(self)

    def clear_report(self):
        util_clear_report(self)

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

    def _chain_unique_targets(self, chain):
        targets = {
            str(step.get("predicate") or step.get("target") or "").strip().lower()
            for step in chain[1:]
            if isinstance(step, dict)
        }
        return len({target for target in targets if target})

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
        tokens = TARGET_TOKEN_RE.findall(str(text or "").lower())
        return [token for token in tokens if len(token) > 2 and token not in STOPWORD_TOKENS]

    def _lexical_target_score(self, agent_name, target_text):
        agent_text = str(agent_name or "").strip().lower()
        target_text = str(target_text or "").strip().lower()
        if not agent_text or not target_text:
            return 0.0
        if agent_text == target_text:
            return 1.0
        if agent_text in target_text:
            return 0.98
        if target_text in agent_text:
            return 0.90

        agent_tokens = set(self._target_tokens(agent_text))
        target_tokens = set(self._target_tokens(target_text))
        if not agent_tokens or not target_tokens:
            return 0.0

        overlap = agent_tokens & target_tokens
        if not overlap:
            return 0.0

        # Reward target coverage from the node phrase, but keep exact containment strongest.
        precision = len(overlap) / len(agent_tokens)
        recall = len(overlap) / len(target_tokens)
        return max(precision, 0.8 * recall)

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return self.embedding_model
        try:
            from sentence_transformers import SentenceTransformer # type: ignore
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[THOUGHT] Warning: Could not load embedding model for target seeding: {e}")
            self.embedding_model = False
        return self.embedding_model

    def select_target_seed_agents(self, limit=TARGET_SEED_LIMIT, min_similarity=TARGET_SEED_SIMILARITY):
        agents = list(self.agents.values())
        if not agents:
            return []

        target = str(self.current_target or "").strip()
        if not target:
            return agents[:limit]

        names = [agent.ASU for agent in agents]
        lexical_scores = np.array([self._lexical_target_score(name, target) for name in names], dtype=np.float32)
        embedding_scores = np.zeros(len(agents), dtype=np.float32)

        model = self._load_embedding_model()
        if model:
            try:
                encoded = model.encode([target] + names, normalize_embeddings=True)
                target_vec = encoded[0]
                name_vecs = encoded[1:]
                embedding_scores = np.dot(name_vecs, target_vec)
            except Exception as e:
                print(f"[THOUGHT] Warning: Embedding similarity failed, using lexical targeting only: {e}")

        combined_scores = np.maximum(embedding_scores, lexical_scores)
        ranked = sorted(
            (
                (combined_scores[i], lexical_scores[i], len(names[i]), agents[i])
                for i in range(len(agents))
            ),
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        )

        selected = [agent for score, _lexical, _length, agent in ranked if score >= min_similarity][:limit]
        if selected:
            return selected

        lexical_selected = [agent for _score, lexical, _length, agent in ranked if lexical > 0][:limit]
        if lexical_selected:
            return lexical_selected

        return [agent for _score, _lexical, _length, agent in ranked[:limit]]

    def refresh_target_seed_agents(self, limit=TARGET_SEED_LIMIT, min_similarity=TARGET_SEED_SIMILARITY):
        self.target_seed_agents = self.select_target_seed_agents(limit=limit, min_similarity=min_similarity)
        self.target_seed_indices = {agent.index for agent in self.target_seed_agents}
        self.thoughts = [Thought(agent) for agent in self.target_seed_agents]
        if self.target_seed_agents:
            preview = ", ".join(agent.ASU for agent in self.target_seed_agents[:5])
            print(f"[THOUGHT] Seeded {len(self.target_seed_agents)} target-matched thought agents for '{self.current_target}'.")
            print(f"[THOUGHT] Top seeds: {preview}")
        else:
            print(f"[THOUGHT] No target-matched thought agents found for '{self.current_target}'.")

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
            connections = find_connections(blocks)
            payloads.append({"query": path.stem, "connections": connections})

        if not payloads:
            raise ValueError(f"No usable text content found in {root}")
        return payloads

    def select_argument_histories(self, active_histories, limit=20):
        usable = [chain for chain in active_histories if self._is_usable_argument_chain(chain)]
        pool = usable if usable else active_histories
        return sorted(
            pool,
            key=lambda chain: (len(chain), self._chain_unique_targets(chain)),
            reverse=True,
        )[:limit]

    def write_argument_report(self, active_histories, sampled_histories, is_final=False):
        self.synthesizer.relation_labels = list(self.relation_labels)
        with open("argument_report.txt", "w") as f:
            f.write("--- THOUGHT CHAIN ARGUMENT REPORT ---\n")
            f.write(f"Goal: {self.current_goal or '(none)'}\n")
            f.write(f"Target: {self.current_target or '(none)'}\n")
            f.write(f"Phase: {'final' if is_final else 'interim'}\n")
            f.write(f"Active histories: {len(active_histories)}\n")
            f.write(f"Sampled histories: {len(sampled_histories)}\n\n")

            if not active_histories:
                f.write("No active thought histories were available yet.\n")
                return

            for i, chain in enumerate(sampled_histories, 1):
                formatted = self.synthesizer._format_argument(chain)
                f.write(f"[ARG {i}] {formatted or '(empty chain)'}\n")
                f.write(f"[RAW {i}] {chain}\n\n")

    def load_relation_index(self):
        relation_labels = list(LOGICAL_CONNECTORS)
        if not os.path.exists(self.relation_index_path):
            return relation_labels

        try:
            with open(self.relation_index_path, "r") as f:
                payload = json.load(f)
        except Exception:
            return relation_labels

        stored_logical = [
            normalize_relation_label(label)
            for label in payload.get("logical_connectors", [])
        ]
        expected_logical = [
            normalize_relation_label(label)
            for label in LOGICAL_CONNECTORS
        ]
        if stored_logical != expected_logical:
            return relation_labels

        seen = set(expected_logical)
        for label in payload.get("verbs", []):
            normalized = normalize_relation_label(str(label))
            if normalized and normalized not in seen:
                relation_labels.append(normalized)
                seen.add(normalized)

        return relation_labels

    def write_relation_index(self):
        relations = []
        logical_count = len(LOGICAL_CONNECTORS)
        for relation_id, label in enumerate(self.relation_labels):
            relations.append({
                "id": relation_id,
                "label": label,
                "kind": "logical" if relation_id < logical_count else "verb",
            })

        with open(self.relation_index_path, "w") as f:
            json.dump({
                "logical_connector_count": logical_count,
                "logical_connectors": list(LOGICAL_CONNECTORS),
                "verbs": self.relation_labels[logical_count:],
                "relations": relations,
            }, f, indent=2)

    def resolve_relation_id(self, rel_type=None, verb=None):
        if rel_type is not None:
            relation_id = int(rel_type)
            if relation_id >= len(self.relation_labels):
                self.relation_labels = self.load_relation_index()
                self.relation_to_id = {
                    normalize_relation_label(label): idx
                    for idx, label in enumerate(self.relation_labels)
                }
            return relation_id

        label = normalize_relation_label(verb or "")
        if not label:
            return 0

        relation_id = self.relation_to_id.get(label)
        if relation_id is not None:
            return relation_id

        relation_id = len(self.relation_labels)
        self.relation_labels.append(label)
        self.relation_to_id[label] = relation_id
        self.write_relation_index()
        print(f"[MIND] Registered verb relation {relation_id}: '{label}'")
        return relation_id

    def _spawn_agent(self, name, near=None):
        """Create agent at near+jitter if near is given, otherwise random. Returns (agent, is_new)."""
        name = name.strip().lower()
        if not name:
            return None, False
        if name in self.agents:
            return self.agents[name], False
        if self.next_idx >= 60000:
            return None, False

        if near is not None:
            pos = np.array([
                near[0] + random.uniform(-10, 10),
                near[1] + random.uniform(-10, 10),
                near[2] + random.uniform(-10, 10),
            ], dtype=np.float32)
        else:
            pos = np.array([random.uniform(-100, 100) for _ in range(3)], dtype=np.float32)

        agent = ASU_Agent(self.next_idx, ASU=name, pos=pos)
        self.agents[name] = agent
        self.agents_by_idx[self.next_idx] = agent
        off = self.next_idx * 24
        struct.pack_into("ffffff", self.shm_pos.buf, off, pos[0], pos[1], pos[2], 0, 0, 0)
        self.next_idx += 1
        return agent, True

    def create_agent(self, name):
        agent, _ = self._spawn_agent(name)
        return agent

    def add_connection(self, s_name, o_name, rel_type=None, verb=None, truth=1, source="unknown", subject_sp=None, predicate_sp=None):
        s_agent, _ = self._spawn_agent(s_name)
        # Always spawn o near s — if s just spawned randomly, o goes near it;
        # if s already existed, o goes near its current position.
        o_agent, _ = self._spawn_agent(o_name, near=s_agent.pos if s_agent else None)
        if not s_agent or not o_agent:
            return
        if self.connection_count >= 80000:
            return
        relation_id = self.resolve_relation_id(rel_type=rel_type, verb=verb)
        if not isinstance(subject_sp, composed_sp):
            subject_sp = composed_sp(quantifier=0, tense=0, truth=(1 if truth else 0), ASU_idx=s_name)
        if not isinstance(predicate_sp, composed_sp):
            predicate_sp = composed_sp(quantifier=0, tense=0, truth=(1 if truth else 0), ASU_idx=o_name)
        flags = encode_connection_flags(subject_sp, predicate_sp)

        key = (s_agent.index, o_agent.index, relation_id, flags)
        if key in self.seen_connections:
            if source and source != "unknown":
                self.connection_sources[key] = source
            return
        self.seen_connections.add(key)
        self.connection_sources[key] = source

        off = 4 + (self.connection_count * 16)
        struct.pack_into("iiii", self.shm_connections.buf, off, s_agent.index, relation_id, flags, o_agent.index)

        connector = Connector(self.shm_connections.buf, off, self.agents_by_idx, self.relation_labels, source=source)
        s_agent.connectors.append(connector)

        self.connection_count += 1
        struct.pack_into("i", self.shm_connections.buf, 0, self.connection_count)

        relation_label = self.relation_labels[relation_id] if 0 <= relation_id < len(self.relation_labels) else str(relation_id)
        self._conn_log.append(
            f"[CONN] {s_name} -> {o_name} | Type: {relation_id} ({relation_label}) | "
            f"S(Q{subject_sp.quantifier},T{subject_sp.tense},TR{subject_sp.truth}) | "
            f"P(Q{predicate_sp.quantifier},T{predicate_sp.tense},TR{predicate_sp.truth}) | Source: {source}\n"
        )
        if len(self._conn_log) >= 100:
            self.flush_conn_log()

    def remap_connection_sources(self, index_map):
        remapped = {}
        for (s_idx, o_idx, rel_type, flags), source in self.connection_sources.items():
            key = (index_map.get(s_idx, s_idx), index_map.get(o_idx, o_idx), rel_type, flags)
            existing = remapped.get(key)
            if existing in (None, "unknown") and source:
                remapped[key] = source
            elif existing is None:
                remapped[key] = source
        self.connection_sources = remapped

    def rebuild_connectors_from_shared_memory(self):
        for agent in self.agents.values():
            agent.connectors = []

        try:
            self.connection_count = struct.unpack_from("i", self.shm_connections.buf, 0)[0]
        except Exception:
            self.connection_count = 0

        self.seen_connections = set()
        for i in range(self.connection_count):
            off = 4 + (i * 16)
            try:
                s_idx, relation_id, flags, o_idx = struct.unpack_from("iiii", self.shm_connections.buf, off)
            except struct.error:
                break

            key = (s_idx, o_idx, relation_id, flags)
            self.seen_connections.add(key)
            s_agent = self.agents_by_idx.get(s_idx)
            if s_agent is None:
                continue

            source = self.connection_sources.get(key, "unknown")
            connector = Connector(self.shm_connections.buf, off, self.agents_by_idx, self.relation_labels, source=source)
            if connector.target is not None:
                s_agent.connectors.append(connector)

        self.next_idx = max(self.agents_by_idx.keys(), default=-1) + 1

    def run_final_synthesis(self):
        if not self.current_target or not self.current_command_id:
            return
        if self.current_command_id in self.finalized_command_ids:
            return

        print("[SYNTHESIS] Swarm stabilized. Running final synthesis...")
        active_histories = [thought.history for thought in self.thoughts if len(thought.history) > 1]

        if active_histories:
            histories = self.select_argument_histories(active_histories, limit=20)
            print(f"[SYNTHESIS] Running for '{self.current_target}' with {len(active_histories)} chains...")
            self.write_argument_report(active_histories, histories, is_final=True)

            report = self.synthesizer.synthesize(self.current_target, self.current_goal, histories)
            report_bytes = report.encode("utf-8")[: (128 * 1024 - 1)]
            self.shm_report.buf[:len(report_bytes)] = report_bytes
            self.shm_report.buf[len(report_bytes):len(report_bytes) + 1] = b"\x00"
            print(f"[SYNTHESIS] Report updated ({len(report_bytes)} bytes).")
        else:
            self.write_argument_report(active_histories, [], is_final=True)
            print("[SYNTHESIS] No active thought chains. Returning to idle.")

        self.finalized_command_ids.add(self.current_command_id)
        self.shm_status.buf[0] = PHASE_IDLE


def run_brain(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter):
    print("[BRAIN] Brain Online. Awaiting logical input...")
    brain = Brain(shm_names)

    t_thread = threading.Thread(target=run_thought_processes, args=(brain, stop_event), daemon=True)
    t_thread.start()

    while not stop_event.is_set():
        # 1. Check for new commands
        cmd_raw = bytes(brain.shm_cmd.buf).split(b'\x00')[0].decode('utf-8').strip()
        if cmd_raw:
            cmd_id, goal, target, local_only = parse_command_payload(cmd_raw)
            current_status = brain.shm_status.buf[0]

            if cmd_id != brain.last_cmd_id and current_status == PHASE_IDLE:
                brain.last_cmd_id = cmd_id
                brain.current_command_id = cmd_id
                brain.current_goal = goal.strip()
                brain.current_target = target.strip()
                brain.local_only_mode = bool(local_only)
                brain.finalized_command_ids.discard(cmd_id)
                brain.clear_report()
                brain.shm_status.buf[0] = PHASE_RESEARCH

                print(f"[BRAIN] New command: '{cmd_raw}'")
                print(f"[BRAIN] Goal: '{brain.current_goal}' | Target: '{brain.current_target}'")
                if brain.local_only_mode:
                    print("[BRAIN] Local-only mode enabled. Skipping internet scraping.")

                # Reserve one in-flight unit while query expansion is running so the
                # supervisor does not declare research complete before work is queued.
                with sync_counter.get_lock():
                    sync_counter.value = 1

                try:
                    if brain.local_only_mode:
                        local_payloads = brain.load_local_information_payloads()
                        print(f"[LOCAL] Loaded {len(local_payloads)} local text files.")
                        with sync_counter.get_lock(), ingest_counter.get_lock():
                            sync_counter.value = 0
                            ingest_counter.value = len(local_payloads)
                        for payload in local_payloads:
                            asu_queue.put(payload)
                    else:
                        queries = d_query_expander.expand(brain.current_target)
                        print(f"[EXPANDER] Generated {len(queries)} sub-queries.")

                        if queries:
                            with sync_counter.get_lock(), ingest_counter.get_lock():
                                sync_counter.value = len(queries)
                                ingest_counter.value = len(queries)
                            for q in queries:
                                scrape_queue.put({"query": q})
                        else:
                            print("[BRAIN] No sub-queries generated. Returning to idle.")
                            brain.shm_status.buf[0] = PHASE_IDLE
                            with sync_counter.get_lock(), ingest_counter.get_lock():
                                sync_counter.value = 0
                                ingest_counter.value = 0
                except Exception as e:
                    mode = "local information loading" if brain.local_only_mode else "query expansion"
                    print(f"[BRAIN] {mode} failed: {e}")
                    brain.shm_status.buf[0] = PHASE_IDLE
                    with sync_counter.get_lock(), ingest_counter.get_lock():
                        sync_counter.value = 0
                        ingest_counter.value = 0

        # 2. Drain extracted connections from scraper workers
        while True:
            try:
                task = asu_queue.get_nowait()
            except Empty:
                break
            if task.get("cmd") == "wave_complete":
                continue
            connections = task.get("connections", [])
            query = task.get("query", "unknown")
            if task.get("error"):
                print(f"[SCRAPE] Query failed for '{query}': {task['error']}")
            print(f"[SCRAPE] {len(connections)} connections for query: '{query}'")
            for c in connections:
                subject_sp = c.get("subject")
                predicate_sp = c.get("predicate")
                relation_id = c.get("connection")
                if not isinstance(subject_sp, composed_sp) or not isinstance(predicate_sp, composed_sp):
                    continue
                subject_name = subject_sp.asu_value()
                predicate_name = predicate_sp.asu_value()
                if not subject_name or not predicate_name or relation_id is None:
                    continue
                brain.add_connection(
                    subject_name,
                    predicate_name,
                    rel_type=relation_id,
                    source=c.get("source", query),
                    subject_sp=subject_sp,
                    predicate_sp=predicate_sp,
                )
            with ingest_counter.get_lock():
                ingest_counter.value = max(0, ingest_counter.value - 1)

        # 3. Phase transitions
        current_status = brain.shm_status.buf[0]

        if current_status == PHASE_EXPORT:
            print("[BRAIN] Exporting agent list for semantic grouping...")
            export_data = [{"index": a.index, "name": a.ASU} for a in brain.agents.values()]
            with open("agents_pre_group.json", "w") as f:
                json.dump(export_data, f)
            brain.flush_conn_log()
            brain.shm_status.buf[0] = PHASE_EXPORTED

        elif current_status == PHASE_IMPORT:
            if os.path.exists("agent_mapping.json"):
                print("[BRAIN] Importing semantic mapping and merging agents...")
                with open("agent_mapping.json", "r") as f:
                    map_data = json.load(f)

                new_agents_dict   = {}
                new_agents_by_idx = {}
                for a_info in map_data["new_agents"]:
                    idx, name = a_info["index"], a_info["name"]
                    if idx in brain.agents_by_idx:
                        agent = brain.agents_by_idx[idx]
                        agent.ASU = name
                    else:
                        agent = ASU_Agent(idx, ASU=name)
                    new_agents_dict[name]  = agent
                    new_agents_by_idx[idx] = agent

                brain.agents        = new_agents_dict
                brain.agents_by_idx = new_agents_by_idx

                # Rebuild seen_connections with remapped indices so dedup stays valid
                index_map = {int(k): v for k, v in map_data["mapping"].items()}
                brain.seen_connections = {
                    (index_map.get(s, s), index_map.get(o, o), t, f)
                    for s, o, t, f in brain.seen_connections
                }
                brain.remap_connection_sources(index_map)
                brain.rebuild_connectors_from_shared_memory()
                brain.refresh_target_seed_agents()

                print(f"[BRAIN] Merge complete. Swarm: {len(brain.agents)} agents.")
                os.remove("agent_mapping.json")
                os.remove("agents_pre_group.json")
            elif brain.agents:
                brain.refresh_target_seed_agents()

            brain.shm_status.buf[0] = PHASE_IMPORTED

        elif current_status == PHASE_STABLE:
            brain.run_final_synthesis()
        time.sleep(0.1)

def run_thought_processes(brain, stop_event):
    print("[THOUGHT] Thought Swarm Thread Active.")
    while not stop_event.is_set():
        if brain.thoughts:
            batch_size = min(100, len(brain.thoughts))
            for _ in range(batch_size):
                thought = random.choice(brain.thoughts)
                moved = thought.move()
                if not moved:
                    seed_pool = brain.target_seed_agents or list(brain.agents.values())
                    thought.reset(random.choice(seed_pool))
        time.sleep(0.05)
