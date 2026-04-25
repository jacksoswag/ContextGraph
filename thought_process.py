import re
import threading
from queue import Empty, Queue

import numpy as np  # type: ignore

from constants import (
    BOOTSTRAP_THOUGHT_ROUNDS,
    GOAL_AGENT_LIMIT,
    INFO_VECTOR_CONNECTION_LIMIT,
    MIN_SUCCESS_STEPS,
    PATH_TARGET_MATCH_THRESHOLD,
    PHASE_THINKING,
    STOPWORD_TOKENS,
    SYNTHESIS_TARGET_MATCH_THRESHOLD,
    TARGET_SEED_LIMIT,
    THOUGHT_AGENTS_PER_WORKER,
    THOUGHT_MIN_LEXICAL_SEED_SCORE,
    THINK_THREADS,
)
from d_word_info_map import literal_from_index, str_to_vector
from o_connection import ConnectionEndpoint
from o_thought_agent import Thought, clear_thought_caches
from utils import apply_quantifier

TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")
SOURCE_TITLE_TOKENS = {
    "facebook", "google", "instagram", "linkedin", "reddit", "wikipedia",
    "youtube", "zillow",
}


def target_tokens(text):
    tokens = TARGET_TOKEN_RE.findall(str(text or "").lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORD_TOKENS]


def useful_agent_tokens(text):
    return [
        token
        for token in target_tokens(text)
        if token not in SOURCE_TITLE_TOKENS
    ]


def is_useful_thought_agent(agent):
    name = str(getattr(agent, "ASU", "") or "").strip()
    if not name:
        return False
    tokens = useful_agent_tokens(name)
    if not tokens:
        return False
    if len(tokens) < 2 and len(name) < 8:
        return False
    return True


def normalize_seed_queries(queries):
    ordered = []
    seen = set()
    for query in queries:
        cleaned = " ".join(str(query or "").strip().split())
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(cleaned)
    return ordered


def lexical_target_score(_brain, agent_name, target_text):
    agent_text = str(agent_name or "").strip().lower()
    target_text = str(target_text or "").strip().lower()
    if not agent_text or not target_text:
        return 0.0

    agent_tokens = set(target_tokens(agent_text))
    target_tokens_set = set(target_tokens(target_text))
    if not agent_tokens or not target_tokens_set:
        return 0.0
    if agent_tokens == target_tokens_set:
        return 1.0
    if agent_tokens <= target_tokens_set:
        return 0.98
    if target_tokens_set <= agent_tokens:
        return 0.90

    overlap = agent_tokens & target_tokens_set
    if not overlap:
        return 0.0

    precision = len(overlap) / len(agent_tokens)
    recall = len(overlap) / len(target_tokens_set)
    return max(precision, 0.8 * recall)


def target_a_seed_queries(brain):
    queries = normalize_seed_queries(
        [brain.current_target_a] + brain.target_a_queries + brain.target_a_focus_phrases
    )
    if queries:
        return queries
    fallback = " ".join(str(brain.current_target_a or "").strip().split())
    return [fallback] if fallback else []


def target_b_completion_queries(brain):
    queries = normalize_seed_queries([brain.current_target_b] + brain.target_b_queries)
    if queries:
        return queries
    fallback = " ".join(str(brain.current_target_b or "").strip().split())
    return [fallback] if fallback else []


def agent_info_text(brain, agent, limit=None):
    if limit is None:
        limit = INFO_VECTOR_CONNECTION_LIMIT

    fragments = [agent.ASU]
    seen = set()

    for conn in agent.connectors:
        source_agent = conn.source_agent
        target = conn.target
        if source_agent is None or source_agent.index != agent.index:
            continue
        if target is None:
            continue

        subject_text = agent.ASU
        predicate_text = target.ASU

        subject_modifier_idx = conn.subject.modifier_idx[0] if conn.subject.modifier_idx else -1
        predicate_modifier_idx = conn.predicate.modifier_idx[0] if conn.predicate.modifier_idx else -1

        if subject_modifier_idx != -1:
            subject_modifier = ConnectionEndpoint.modifier_from_idx(subject_modifier_idx)
            if subject_modifier:
                subject_text = f"{subject_modifier} {subject_text}".strip()
        if predicate_modifier_idx != -1:
            predicate_modifier = ConnectionEndpoint.modifier_from_idx(predicate_modifier_idx)
            if predicate_modifier:
                predicate_text = f"{predicate_modifier} {predicate_text}".strip()

        subject_text = apply_quantifier(subject_text, conn.subject.quantifier)
        predicate_text = apply_quantifier(predicate_text, conn.predicate.quantifier)

        relation = str(literal_from_index(conn.relation_index) or conn.relation_index).strip()
        phrase = f"{subject_text} {relation} {predicate_text}".strip()
        phrase = f"{phrase}{brain._specifics_suffix([], [], [])}"
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        fragments.append(phrase)
        if len(fragments) >= limit + 1:
            break

    return " ; ".join(fragment for fragment in fragments if fragment)


def lexical_info_score(_brain, info_text, query_text):
    info_tokens = set(target_tokens(info_text))
    query_tokens = set(target_tokens(query_text))
    if not info_tokens or not query_tokens:
        return 0.0

    overlap = info_tokens & query_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(query_tokens)
    recall = len(overlap) / len(info_tokens)
    return max(precision, 0.75 * recall)


def _distribute_spawn_counts(seed_specs, total_thoughts):
    if not seed_specs:
        return []

    total_thoughts = max(len(seed_specs), int(total_thoughts or len(seed_specs)))
    weights = []
    for item in seed_specs:
        degree = max(1, len(getattr(item["agent"], "connectors", ()) or ()))
        weights.append(max(0.01, float(item.get("score", 0.0))) * degree)

    weight_total = sum(weights)
    if weight_total <= 0.0:
        weights = [1.0 for _ in seed_specs]
        weight_total = float(len(weights))

    raw_counts = [(weight / weight_total) * total_thoughts for weight in weights]
    spawn_counts = [max(1, int(count)) for count in raw_counts]
    remaining = total_thoughts - sum(spawn_counts)
    ranked_remainders = sorted(
        range(len(seed_specs)),
        key=lambda idx: raw_counts[idx] - int(raw_counts[idx]),
        reverse=True,
    )

    while remaining > 0:
        for idx in ranked_remainders:
            if remaining <= 0:
                break
            spawn_counts[idx] += 1
            remaining -= 1

    while remaining < 0:
        changed = False
        for idx in reversed(ranked_remainders):
            if remaining >= 0:
                break
            if spawn_counts[idx] <= 1:
                continue
            spawn_counts[idx] -= 1
            remaining += 1
            changed = True
        if not changed:
            break

    for item, spawn_count in zip(seed_specs, spawn_counts):
        item["spawn_count"] = spawn_count
    return seed_specs


def select_target_seed_agents(brain, limit=None, total_thoughts=None):
    agents = [
        agent
        for agent in brain.agents.values()
        if agent.connectors and is_useful_thought_agent(agent)
    ]
    if not agents:
        agents = [
            agent
            for agent in brain.agents.values()
            if is_useful_thought_agent(agent)
        ]
    if not agents:
        return []

    queries = target_a_seed_queries(brain)
    if not queries:
        return []
    seed_limit = int(limit or TARGET_SEED_LIMIT)

    names = [agent.ASU for agent in agents]
    info_texts = [agent_info_text(brain, agent) for agent in agents]
    lexical_scores = np.zeros((len(queries), len(agents)), dtype=np.float32)
    for q_idx, query in enumerate(queries):
        lexical_scores[q_idx] = np.array(
            [
                max(
                    lexical_target_score(brain, name, query),
                    lexical_info_score(brain, info_text, query),
                )
                for name, info_text in zip(names, info_texts)
            ],
            dtype=np.float32,
        )

    combined_scores = lexical_scores.copy()
    try:
        query_vecs = np.asarray(
            [str_to_vector(query) for query in queries],
            dtype=np.float32,
        )
        info_vecs = np.asarray(
            [str_to_vector(info_text) for info_text in info_texts],
            dtype=np.float32,
        )
        if (
            query_vecs.ndim == 2
            and info_vecs.ndim == 2
            and query_vecs.size
            and info_vecs.size
            and query_vecs.shape[1] == info_vecs.shape[1]
        ):
            vector_scores = np.matmul(query_vecs, info_vecs.T)
            combined_scores = np.where(
                lexical_scores >= THOUGHT_MIN_LEXICAL_SEED_SCORE,
                np.maximum(vector_scores, lexical_scores),
                lexical_scores,
            )
    except Exception as e:
        print(f"[THOUGHT] Warning: Shared-vector seed matching failed, using lexical target seeding only: {e}")

    selected_specs = []
    for agent_idx, agent in enumerate(agents):
        eligible_queries = [
            q_idx
            for q_idx in range(len(queries))
            if float(lexical_scores[q_idx][agent_idx]) >= THOUGHT_MIN_LEXICAL_SEED_SCORE
        ]
        if not eligible_queries:
            continue
        best_query_idx = max(
            eligible_queries,
            key=lambda q_idx: float(combined_scores[q_idx][agent_idx]),
        )
        selected_specs.append(
            {
                "agent": agent,
                "query": queries[best_query_idx],
                "score": float(combined_scores[best_query_idx][agent_idx]),
                "spawn_count": 1,
            }
        )

    selected_specs.sort(
        key=lambda item: (
            float(item["score"]),
            len(getattr(item["agent"], "connectors", ()) or ()),
            item["agent"].ASU,
        ),
        reverse=True,
    )
    selected_specs = selected_specs[:seed_limit]
    selected_specs = _distribute_spawn_counts(selected_specs, total_thoughts)

    brain._last_seed_debug = [
        {
            "agent": item["agent"].ASU,
            "best_score": round(item["score"], 4),
            "best_query": item["query"],
            "spawn_count": item["spawn_count"],
        }
        for item in selected_specs
    ]
    return selected_specs


def refresh_target_seed_agents(brain, limit=None, total_thoughts=None):
    clear_thought_caches()
    brain.target_seed_specs = select_target_seed_agents(
        brain,
        limit=limit,
        total_thoughts=total_thoughts,
    )
    brain.target_seed_agents = [item["agent"] for item in brain.target_seed_specs]
    brain.target_seed_indices = {agent.index for agent in brain.target_seed_agents}
    completion_queries = target_b_completion_queries(brain)
    goal_agents = select_goal_agents(brain, completion_queries, limit=GOAL_AGENT_LIMIT)
    brain.thoughts = []
    for item in brain.target_seed_specs:
        agent = item["agent"]
        for _ in range(item["spawn_count"]):
            brain.thoughts.append(
                Thought(
                    agent,
                    seed_query=item["query"],
                    start_target=brain.current_target_a,
                    target_b=brain.current_target_b,
                    success_queries=completion_queries,
                    goal_agents=goal_agents,
                    tense_preference=getattr(brain, "current_tense_preference", "none"),
                )
            )
    if brain.target_seed_agents:
        preview = ", ".join(agent.ASU for agent in brain.target_seed_agents[:5])
        print(
            f"[THOUGHT] Seeded {len(brain.thoughts)} thought agents across "
            f"{len(brain.target_seed_agents)} target-matched information agents for '{brain.current_target}'."
        )
        if brain.current_subqueries:
            print(f"[THOUGHT] Seed queries: {', '.join(brain.current_subqueries[:10])}")
        if brain.target_a_focus_phrases:
            print(f"[THOUGHT] Target A focus: {', '.join(brain.target_a_focus_phrases)}")
        if brain.target_b_focus_phrases:
            print(f"[THOUGHT] Target B focus: {', '.join(brain.target_b_focus_phrases)}")
        print(f"[THOUGHT] Top seeds: {preview}")
        if goal_agents:
            goal_preview = ", ".join(agent.ASU for agent in goal_agents[:5])
            print(f"[THOUGHT] Goal agents: {goal_preview}")
    else:
        print(f"[THOUGHT] No target-matched thought agents found for '{brain.current_target}'.")


def select_goal_agents(brain, queries=None, limit=GOAL_AGENT_LIMIT):
    queries = normalize_seed_queries(queries or target_b_completion_queries(brain))
    if not queries:
        return []

    ranked = []
    for agent in brain.agents.values():
        if not is_useful_thought_agent(agent):
            continue
        info_text = agent_info_text(brain, agent)
        score = max(
            max(lexical_target_score(brain, agent.ASU, query) for query in queries),
            max(lexical_info_score(brain, info_text, query) for query in queries),
        )
        if score < THOUGHT_MIN_LEXICAL_SEED_SCORE:
            continue
        ranked.append((score, len(agent.connectors), agent.ASU, agent))

    ranked.sort(key=lambda item: item[:3], reverse=True)
    return [agent for _score, _degree, _name, agent in ranked[:limit]]


def completed_thought_histories(brain):
    completed_thoughts = [
        thought for thought in brain.thoughts if len(getattr(thought, "history", [])) > 1
    ]
    return completed_thoughts, [thought.history for thought in completed_thoughts]


def successful_thoughts(brain):
    strict_successes = [
        thought
        for thought in brain.thoughts
        if getattr(thought, "successful", False)
        and getattr(thought, "success_payload", None)
    ]
    if strict_successes:
        return strict_successes

    candidates = []
    for thought in brain.thoughts:
        if len(getattr(thought, "history", [])) - 1 < MIN_SUCCESS_STEPS:
            continue
        try:
            lexical_a, lexical_b = thought._path_target_lexical_matches()
            if lexical_a < PATH_TARGET_MATCH_THRESHOLD or lexical_b < PATH_TARGET_MATCH_THRESHOLD:
                continue
            payload = thought._build_relationship_statement()
        except Exception:
            continue
        if (
            float(payload.get("match_target_a", 0.0) or 0.0) < SYNTHESIS_TARGET_MATCH_THRESHOLD
            or float(payload.get("match_target_b", 0.0) or 0.0) < SYNTHESIS_TARGET_MATCH_THRESHOLD
            or float(payload.get("lexical_target_a", 0.0) or 0.0) < PATH_TARGET_MATCH_THRESHOLD
            or float(payload.get("lexical_target_b", 0.0) or 0.0) < PATH_TARGET_MATCH_THRESHOLD
        ):
            continue
        thought.successful = True
        thought.success_payload = payload
        candidates.append(thought)
    return candidates


def thought_swarm_stats(brain):
    total = len(brain.thoughts)
    alive = sum(1 for thought in brain.thoughts if getattr(thought, "alive", False))
    completed = sum(1 for thought in brain.thoughts if len(getattr(thought, "history", [])) > 1)
    successful = sum(1 for thought in brain.thoughts if getattr(thought, "successful", False))
    endpoint = sum(
        1
        for thought in brain.thoughts
        if not getattr(thought, "alive", False)
        and getattr(thought, "termination_reason", "") == "endpoint"
    )
    dead = sum(
        1
        for thought in brain.thoughts
        if not getattr(thought, "alive", False)
        and getattr(thought, "termination_reason", "") == "dead"
    )
    max_hops = sum(
        1
        for thought in brain.thoughts
        if not getattr(thought, "alive", False)
        and getattr(thought, "termination_reason", "") == "max_hops"
    )
    with brain._thought_lock:
        active_workers = int(brain.thought_active_count)
    return {
        "total": total,
        "alive": alive,
        "completed": completed,
        "successful": successful,
        "endpoint": endpoint,
        "dead": dead,
        "max_hops": max_hops,
        "active_workers": active_workers,
        "seeds": len(brain.target_seed_agents),
    }


def thought_worker_loop(brain, stop_event, generation, worker_idx):
    while not stop_event.is_set():
        if brain.thought_generation != generation:
            break
        if brain.shm_status.buf[0] != PHASE_THINKING:
            break
        queue = brain.thought_queue
        if queue is None:
            break
        try:
            thought = queue.get(timeout=0.1)
        except Empty:
            continue

        with brain._thought_lock:
            brain.thought_active_count += 1
        try:
            if getattr(thought, "alive", False):
                try:
                    thought.move()
                except Exception as e:
                    thought.alive = False
                    thought.termination_reason = "dead"
                    print(f"[THOUGHT] Worker {worker_idx} move failed: {e}")
        finally:
            with brain._thought_lock:
                brain.thought_active_count = max(0, brain.thought_active_count - 1)
            queue.task_done()

        if (
            getattr(thought, "alive", False)
            and brain.thought_generation == generation
            and brain.shm_status.buf[0] == PHASE_THINKING
        ):
            queue.put(thought)


def start_thought_workers(brain, stop_event, worker_count=None):
    if worker_count is None:
        worker_count = THINK_THREADS

    if brain.thought_phase_started and brain.shm_status.buf[0] == PHASE_THINKING:
        return

    stop_thought_workers(brain)
    requested_worker_count = max(1, int(worker_count))
    refresh_target_seed_agents(
        brain,
        total_thoughts=requested_worker_count * THOUGHT_AGENTS_PER_WORKER,
    )
    brain.thought_queue = Queue()
    for thought in brain.thoughts:
        brain.thought_queue.put(thought)

    with brain._thought_lock:
        brain.thought_active_count = 0
    brain.thought_generation += 1
    generation = brain.thought_generation
    worker_total = min(requested_worker_count, max(1, len(brain.thoughts))) if brain.thoughts else 0
    brain.thought_worker_threads = []
    brain.thought_phase_started = True

    if worker_total <= 0:
        print("[THOUGHT] No thought workers launched because no thought agents were seeded.")
        return

    for worker_idx in range(worker_total):
        thread = threading.Thread(
            target=thought_worker_loop,
            args=(brain, stop_event, generation, worker_idx),
            name=f"ThoughtWorker-{worker_idx}",
            daemon=True,
        )
        thread.start()
        brain.thought_worker_threads.append(thread)

    print(
        f"[THOUGHT] Launched {worker_total} thought workers for "
        f"{len(brain.thoughts)} thought agents."
    )


def thought_workers_finished(brain):
    if not brain.thought_phase_started:
        return False
    queue = brain.thought_queue
    with brain._thought_lock:
        active_workers = int(brain.thought_active_count)
    alive = sum(1 for thought in brain.thoughts if getattr(thought, "alive", False))
    if queue is None:
        return alive == 0 and active_workers == 0
    return alive == 0 and active_workers == 0 and queue.empty()


def stop_thought_workers(brain):
    brain.thought_generation += 1
    threads = list(brain.thought_worker_threads)
    brain.thought_worker_threads = []
    brain.thought_phase_started = False
    brain.thought_queue = None
    with brain._thought_lock:
        brain.thought_active_count = 0
    for thread in threads:
        thread.join(timeout=0.2)


def bootstrap_thought_histories(brain, rounds=BOOTSTRAP_THOUGHT_ROUNDS, seed_limit=None):
    if seed_limit is None:
        seed_limit = TARGET_SEED_LIMIT

    refresh_limit = None if seed_limit == TARGET_SEED_LIMIT else seed_limit
    refresh_target_seed_agents(
        brain,
        limit=refresh_limit,
        total_thoughts=THINK_THREADS * THOUGHT_AGENTS_PER_WORKER,
    )

    for _ in range(rounds):
        moved = False
        alive_thoughts = [
            thought for thought in brain.thoughts if getattr(thought, "alive", True)
        ]
        if not alive_thoughts:
            break
        for thought in alive_thoughts:
            try:
                if thought.move():
                    moved = True
            except Exception as e:
                thought.alive = False
                print(f"[THOUGHT] Bootstrap move failed: {e}")
        if not moved:
            break

    return completed_thought_histories(brain)
