import os; import struct; import subprocess; import tempfile; from pathlib import Path
import numpy as np  # type: ignore
import engine.graph.connections as p_connection_graph; from engine.common.constants import (ACTIVE_ANCHOR_PAIR_LIMIT, CONNECTION_RECORD_SIZE, GOAL_AGENT_LIMIT, MAX_CONTEXT_SAMPLES, MAX_THOUGHT_HOPS, MIN_SUCCESS_STEPS, PATH_TARGET_MATCH_THRESHOLD, PHASE_THINKING, SYNTHESIS_TARGET_MATCH_THRESHOLD, TARGET_SEED_LIMIT, THOUGHT_AGENTS_PER_WORKER, THINK_THREADS,)
from engine.extract.noise_cleanup import is_clause_like_agent_text, is_usable_agent_text; from engine.extract.word_info_map import str_to_vector; from engine.agents.connection import Connector, ConnectionEndpoint; from engine.agents.thought import Thought, clear_thought_caches; from engine.extract.target_text import distinctive_target_tokens, target_acronym_tokens, target_tokens; from engine.common.shm import display_source

# Returns target-normalized tokens used to decide whether an agent can anchor a route.
def useful_agent_tokens(text):
    return target_tokens(text)

# Returns whether an agent name is usable as a thought-routing node.
def is_useful_thought_agent(agent):
    name = str(getattr(agent, "ASU", "") or "").strip()
    if not name: return False
    if not is_usable_agent_text(name): return False
    tokens = useful_agent_tokens(name)
    if not tokens: return False
    if len(tokens) < 2 and len(name) < 3: return False
    return True

# Deduplicates target and bridge queries while preserving their display text.
def normalize_seed_queries(queries):
    ordered = []; seen = set()
    for query in queries:
        cleaned = " ".join(str(query or "").strip().split()); normalized = cleaned.lower()
        if not cleaned or normalized in seen: continue
        seen.add(normalized); ordered.append(cleaned)
    return ordered

# Scores lexical overlap between an agent label and a target phrase.
def lexical_target_score(_graph, agent_name, target_text, require_distinctive=False):
    agent_text = str(agent_name or "").strip().lower(); target_text = str(target_text or "").strip().lower()
    if not agent_text or not target_text: return 0.0
    agent_base_tokens = set(target_tokens(agent_text)); target_base_tokens = set(target_tokens(target_text)); agent_tokens = agent_base_tokens | (set(target_acronym_tokens(agent_text)) & target_base_tokens); target_tokens_set = target_base_tokens | (set(target_acronym_tokens(target_text)) & agent_base_tokens)
    if not agent_tokens or not target_tokens_set: return 0.0
    if agent_tokens == target_tokens_set: return 1.0
    overlap = agent_tokens & target_tokens_set
    if not overlap: return 0.0
    if require_distinctive:
        distinctive_tokens = distinctive_target_tokens(target_tokens_set)
        if distinctive_tokens and not (overlap & distinctive_tokens): return 0.0
    if agent_tokens <= target_tokens_set: return 0.98
    if target_tokens_set <= agent_tokens: return 0.90
    precision = len(overlap) / len(agent_tokens); recall = len(overlap) / len(target_tokens_set); distinctive_tokens = distinctive_target_tokens(target_tokens_set) if require_distinctive else set(); distinctive_coverage = (len(overlap & distinctive_tokens) / len(distinctive_tokens) if distinctive_tokens else 0.0)
    return max(precision, 0.8 * recall, distinctive_coverage)
# Returns the active dashboard target text for side A or B.
def target_text_for_side(graph, side):
    side = str(side or "").strip().lower()
    if side == "a":
        return " ".join(str(getattr(graph, "current_target_a", "") or "").strip().split())
    if side == "b":
        return " ".join(str(getattr(graph, "current_target_b", "") or "").strip().split())
    return ""
    
# Returns normalized completion queries for one target side.
def completion_queries_for_side(graph, side):
    target = target_text_for_side(graph, side)
    return normalize_seed_queries([target])

# Returns the two directed route specs used to bridge A to B and B to A.
def route_directions():
    return ({"id": "a_to_b", "start_side": "a", "goal_side": "b", "start_label": "A", "goal_label": "B"}, {"id": "b_to_a", "start_side": "b", "goal_side": "a", "start_label": "B", "goal_label": "A"},)

# Returns readable text from structured specificity payloads.
def _specific_value_texts(values):
    texts = [];
    for value in list(values or []):
        if isinstance(value, dict):
            for key in ("text", "surface", "normalized", "context", "cue"):
                clean = " ".join(str(value.get(key, "") or "").strip().split())
                if clean: texts.append(clean)
        else:
            clean = " ".join(str(value or "").strip().split())
            if clean: texts.append(clean)
    return texts

# Adds one unique grounding text value to the candidate text list.
def _append_grounding_text(texts, seen, value):
    clean = " ".join(str(value or "").strip().split()); key = clean.lower()
    if clean and key not in seen:
        seen.add(key); texts.append(clean)

# Returns the endpoint metadata and specifics for an agent on one connector.
def _agent_endpoint_context(agent, conn):
    source_agent = getattr(conn, "source_agent", None); target_agent = getattr(conn, "target", None); agent_index = getattr(agent, "index", None)
    if getattr(source_agent, "index", None) == agent_index:
        return getattr(conn, "subject", None), list(getattr(conn, "subject_specifics", []) or [])
    if getattr(target_agent, "index", None) == agent_index:
        return getattr(conn, "predicate", None), list(getattr(conn, "predicate_specifics", []) or [])
    return None, []

# Returns agent labels, modifiers, specifics, and evidence used for target matching.
def agent_grounding_texts(agent, connector_limit=18):
    texts = []; seen = set(); name = str(getattr(agent, "ASU", "") or "").strip(); _append_grounding_text(texts, seen, name); connectors = sorted(list(getattr(agent, "connectors", []) or []), key=lambda conn: float(getattr(conn, "utility", 0.0) or 0.0), reverse=True,)
    for conn in connectors[:max(0, int(connector_limit))]:
        endpoint, endpoint_specifics = _agent_endpoint_context(agent, conn)
        if endpoint is not None:
            modifier_text = " ".join(endpoint.modifier_value()).strip(); _append_grounding_text(texts, seen, modifier_text)
            if modifier_text and name:
                _append_grounding_text(texts, seen, f"{name} {modifier_text}")
        for text in _specific_value_texts(endpoint_specifics):
            _append_grounding_text(texts, seen, text)
        evidence = str(getattr(conn, "evidence_text", "") or "").strip(); _append_grounding_text(texts, seen, evidence)
    return texts

# Scores an agent against a target using all grounding text around that agent.
def _agent_lexical_target_score(graph, agent, target_text, require_distinctive=False):
    return max((lexical_target_score(graph, text, target_text, require_distinctive=require_distinctive,) for text in agent_grounding_texts(agent)), default=0.0,)

# Scores the agent name against normalized target query variants.
def _agent_name_target_score(graph, agent, queries, require_distinctive=False):
    name = str(getattr(agent, "ASU", "") or "").strip()
    if not name:
        return 0.0
    return max((lexical_target_score(graph, name, query, require_distinctive=require_distinctive,) for query in queries), default=0.0,)

# Scores whether an agent name is concise and noun-like enough to serve as an anchor.
def _anchor_name_quality(name):
    name = str(name or "").strip()
    if not name:
        return 0.0
    tokens = useful_agent_tokens(name)
    if not tokens:
        return 0.0
    quality = 1.0
    if is_clause_like_agent_text(name):
        return 0.0
    if len(tokens) > 6:
        quality *= max(0.45, 1.0 - (0.08 * (len(tokens) - 6)))
    if len(name) > 56:
        quality *= 0.75
    return quality

# Ranks agents by name similarity for thought routing.
def _rank_agents_by_name_similarity(graph, queries, require_distinctive=False):
    queries = normalize_seed_queries(queries)
    if not queries:
        return []
    query_vecs = []
    try:
        query_vecs = [np.asarray(str_to_vector(query), dtype=np.float32) for query in queries if query]
    except Exception:
        query_vecs = []
    ranked = []; seen_names = set()
    for agent in graph.agents.values():
        if not is_useful_thought_agent(agent):
            continue
        if not getattr(agent, "connectors", None):
            continue
        name = str(getattr(agent, "ASU", "") or "").strip(); name_key = name.lower()
        if not name_key or name_key in seen_names:
            continue
        anchor_quality = _anchor_name_quality(name)
        if anchor_quality <= 0.0:
            continue
        seen_names.add(name_key); lexical_score = max(_agent_lexical_target_score(graph, agent, query, require_distinctive=require_distinctive,) for query in queries); name_lexical_score = _agent_name_target_score(graph, agent, queries, require_distinctive=True,); vector_score = 0.0
        if query_vecs:
            try:
                agent_vec = np.asarray(str_to_vector(name), dtype=np.float32); vector_score = max(float(np.dot(agent_vec, query_vec)) for query_vec in query_vecs)
            except Exception:
                vector_score = 0.0
        own_grounding_score = max(float(name_lexical_score), float(vector_score)); overall_score = max(own_grounding_score, 0.65 * float(lexical_score)); overall_score *= anchor_quality
        if overall_score <= 0.0:
            continue
        ranked.append((overall_score, float(lexical_score), float(vector_score), float(name_lexical_score), len(getattr(agent, "connectors", ()) or ()), name, float(anchor_quality), agent,))
    ranked.sort(key=lambda item: (item[0], item[3], item[4], item[6], item[5]), reverse=True)
    return ranked

# Ranks anchor candidates for thought routing.
def _rank_anchor_candidates(graph, side, limit):
    queries = completion_queries_for_side(graph, side)
    if not queries:
        return []
    opposite_side = "b" if str(side or "").strip().lower() == "a" else "a"; opposite_target = target_text_for_side(graph, opposite_side); ranked = _rank_agents_by_name_similarity(graph, queries, require_distinctive=True); candidates = []
    for score, lexical_score, vector_score, name_score, degree, name, anchor_quality, agent in ranked:
        if getattr(agent, "index", None) is None:
            continue
        grounding_score = max(float(name_score), float(lexical_score))
        if float(score) < float(PATH_TARGET_MATCH_THRESHOLD):
            continue
        if grounding_score < float(PATH_TARGET_MATCH_THRESHOLD):
            continue
        if opposite_target:
            opposite_name_score = lexical_target_score(graph, name, opposite_target, require_distinctive=False,)
            if opposite_name_score >= 0.9 and float(name_score) < 0.5:
                continue
        best_query = max(queries, key=lambda query: _agent_lexical_target_score(graph, agent, query),)
        candidates.append({"agent": agent, "query": best_query, "score": float(score), "lexical_score": float(lexical_score), "vector_score": float(vector_score), "name_score": float(name_score), "grounding_score": float(grounding_score), "degree": int(degree), "name": name, "anchor_quality": float(anchor_quality),})
        if len(candidates) >= int(limit):
            break
    return candidates

# Builds directed adjacency from ContextGraph connection keys.
def _directed_adjacency_from_connections(graph):
    adjacency = {}
    for key in getattr(graph, "connection_offsets", {}) or {}:
        try:
            source = int(key[0]); target = int(key[1])
        except (TypeError, ValueError, IndexError):
            continue
        adjacency.setdefault(source, []).append((target, key))
    return adjacency

# Reverses adjacency so reachability can be computed toward goals.
def _reverse_adjacency(adjacency):
    reverse = {}
    for source, edges in adjacency.items():
        for target, key in edges:
            reverse.setdefault(int(target), []).append((int(source), key))
    return reverse

# Returns all node indices reachable from a set of start nodes.
def _reachable_indices(adjacency, starts):
    seen = set(); queue = []
    for start in starts:
        try:
            start = int(start)
        except (TypeError, ValueError):
            continue
        if start in seen:
            continue
        seen.add(start); queue.append(start)
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]; cursor += 1
        for neighbor, _key in adjacency.get(current, ()):
            neighbor = int(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor); queue.append(neighbor)
    return seen

# Counts concrete specifics attached to one ContextGraph connection key.
def _connection_specific_count(graph, key):
    specifics = getattr(graph, "connection_specifics", {}).get(key, {}) or {}
    return sum(len(list(specifics.get(field, []) or [])) for field in ("subject_specifics", "predicate_specifics", "connection_specifics"))

# Scores route support by edge count, source diversity, and concrete specifics.
def _route_metrics(graph, start_agent, goal_agent, adjacency, reverse_adjacency, forward_reach=None, backward_reach=None,):
    empty = {"edge_count": 0, "node_count": 0, "source_count": 0, "specific_count": 0, "specific_edge_count": 0, "score": 0.0,}; start_index = getattr(start_agent, "index", None); goal_index = getattr(goal_agent, "index", None)
    if start_index is None or goal_index is None:
        return empty
    start_index = int(start_index); goal_index = int(goal_index); forward_reach = forward_reach or {}; backward_reach = backward_reach or {}; from_start = forward_reach.get(start_index)
    if from_start is None:
        from_start = _reachable_indices(adjacency, [start_index])
    if goal_index not in from_start:
        return empty
    to_goal = backward_reach.get(goal_index)
    if to_goal is None:
        to_goal = _reachable_indices(reverse_adjacency, [goal_index])
    route_edges = set(); route_nodes = set(); sources = set(); specific_count = 0; specific_edge_count = 0
    for source, edges in adjacency.items():
        if int(source) not in from_start:
            continue
        for target, key in edges:
            if int(target) not in to_goal:
                continue
            route_edges.add(key); route_nodes.add(int(source)); route_nodes.add(int(target)); display = display_source(getattr(graph, "connection_sources", {}).get(key, ""))
            if display and display != "unknown":
                sources.add(display.rstrip("/").lower())
            edge_specific_count = _connection_specific_count(graph, key); specific_count += edge_specific_count
            if edge_specific_count:
                specific_edge_count += 1
    edge_count = len(route_edges); source_count = len(sources); score = (min(edge_count, 96) + (2.5 * min(source_count, 16)) + (1.25 * min(specific_edge_count, 48)) + (0.2 * min(specific_count, 160)))
    return {"edge_count": edge_count, "node_count": len(route_nodes), "source_count": source_count, "specific_count": specific_count, "specific_edge_count": specific_edge_count, "score": float(score),}

# Builds dashboard diagnostics for one candidate anchor pair.
def _pair_diagnostic(pair, selected=False):
    a_metrics = pair.get("a_to_b_metrics", {}) or {}; b_metrics = pair.get("b_to_a_metrics", {}) or {}
    return {"selected": bool(selected), "a": pair["a"].get("name", ""), "b": pair["b"].get("name", ""), "a_score": round(float(pair["a"].get("score", 0.0) or 0.0), 3), "b_score": round(float(pair["b"].get("score", 0.0) or 0.0), 3), "a_grounding": round(float(pair["a"].get("grounding_score", 0.0) or 0.0), 3), "b_grounding": round(float(pair["b"].get("grounding_score", 0.0) or 0.0), 3), "a_to_b_edges": int(a_metrics.get("edge_count", 0) or 0), "b_to_a_edges": int(b_metrics.get("edge_count", 0) or 0), "sources": int(a_metrics.get("source_count", 0) or 0) + int(b_metrics.get("source_count", 0) or 0), "specific_edges": int(a_metrics.get("specific_edge_count", 0) or 0) + int(b_metrics.get("specific_edge_count", 0) or 0), "rank": tuple(round(float(value), 3) for value in pair.get("rank", ())),}

# Finds target-anchor pairs that have usable directed routes in both directions.
def _connected_anchor_pairs(graph, candidate_limit, pair_limit=None):
    a_candidates = _rank_anchor_candidates(graph, "a", candidate_limit); b_candidates = _rank_anchor_candidates(graph, "b", candidate_limit); adjacency = _directed_adjacency_from_connections(graph); reverse = _reverse_adjacency(adjacency)
    candidate_indices = {int(item["agent"].index) for item in list(a_candidates or []) + list(b_candidates or []) if getattr(item.get("agent"), "index", None) is not None}; forward_reach = {index: _reachable_indices(adjacency, [index]) for index in candidate_indices}
    backward_reach = {index: _reachable_indices(reverse, [index]) for index in candidate_indices}; pairs = []
    for a_item in a_candidates:
        a_agent = a_item["agent"]
        for b_item in b_candidates:
            b_agent = b_item["agent"]
            if int(a_agent.index) == int(b_agent.index):
                continue
            a_to_b_metrics = _route_metrics(graph, a_agent, b_agent, adjacency, reverse, forward_reach=forward_reach, backward_reach=backward_reach,); b_to_a_metrics = _route_metrics(graph, b_agent, a_agent, adjacency, reverse, forward_reach=forward_reach, backward_reach=backward_reach,); a_to_b = a_to_b_metrics["edge_count"] > 0
            b_to_a = b_to_a_metrics["edge_count"] > 0
            if not a_to_b and not b_to_a:
                continue
            route_score = float(a_to_b_metrics["score"]) + float(b_to_a_metrics["score"]); anchor_floor = min(float(a_item["score"]), float(b_item["score"])); anchor_sum = float(a_item["score"]) + float(b_item["score"]); source_count = int(a_to_b_metrics["source_count"]) + int(b_to_a_metrics["source_count"])
            specific_edges = int(a_to_b_metrics["specific_edge_count"]) + int(b_to_a_metrics["specific_edge_count"]); rank = (route_score * max(0.25, anchor_floor), route_score, source_count, specific_edges, 1 if a_to_b and b_to_a else 0, anchor_floor, anchor_sum, a_item["grounding_score"] + b_item["grounding_score"], a_item["degree"] + b_item["degree"],)
            pairs.append({"a": a_item, "b": b_item, "a_to_b": a_to_b, "b_to_a": b_to_a, "a_to_b_metrics": a_to_b_metrics, "b_to_a_metrics": b_to_a_metrics, "rank": rank,})
    if not pairs:
        diagnostics = []
        for a_item in a_candidates[:4]:
            for b_item in b_candidates[:4]:
                a_agent = a_item.get("agent"); b_agent = b_item.get("agent")
                if (getattr(a_agent, "index", None) is None or getattr(b_agent, "index", None) is None or int(a_agent.index) == int(b_agent.index)):
                    continue
                diagnostics.append({"a": a_item, "b": b_item, "a_to_b_metrics": {}, "b_to_a_metrics": {}, "rank": (min(float(a_item.get("score", 0.0)), float(b_item.get("score", 0.0))), float(a_item.get("score", 0.0)) + float(b_item.get("score", 0.0)), float(a_item.get("grounding_score", 0.0)) + float(b_item.get("grounding_score", 0.0)),),})
        diagnostics.sort(key=lambda item: item["rank"], reverse=True); graph.anchor_pair_diagnostics = [_pair_diagnostic(pair, selected=False) for pair in diagnostics[:8]]
        return []
    pairs.sort(key=lambda item: item["rank"], reverse=True); pair_limit = max(1, int(pair_limit or ACTIVE_ANCHOR_PAIR_LIMIT)); selected = pairs[:pair_limit]; selected_keys = {(int(pair["a"]["agent"].index), int(pair["b"]["agent"].index)) for pair in selected}; diagnostic_pairs = selected + [pair for pair in pairs[pair_limit:pair_limit + 8]]
    seen_diag = set(); graph.anchor_pair_diagnostics = []
    for pair in diagnostic_pairs:
        key = (int(pair["a"]["agent"].index), int(pair["b"]["agent"].index))
        if key in seen_diag:
            continue
        seen_diag.add(key); graph.anchor_pair_diagnostics.append(_pair_diagnostic(pair, selected=key in selected_keys))
    return selected

# Deduplicates anchor seed specs by agent index.
def _unique_anchor_items(items):
    unique = []; seen = set()
    for item in list(items or []):
        agent = item.get("agent"); index = getattr(agent, "index", None)
        if index is None or int(index) in seen:
            continue
        seen.add(int(index)); unique.append(item)
    return unique

# Deduplicates goal agents by agent index.
def _unique_goal_agents(agents):
    unique = []; seen = set()
    for agent in list(agents or []):
        index = getattr(agent, "index", None)
        if index is None or int(index) in seen:
            continue
        seen.add(int(index)); unique.append(agent)
    return unique

# Splits native thought spawn counts across active routes.
def _route_spawn_counts(active_count, total_thoughts):
    active_count = max(0, int(active_count or 0))
    if active_count <= 0:
        return []
    if total_thoughts is None:
        return [1 for _ in range(active_count)]
    total = max(active_count, int(total_thoughts or active_count)); base = total // active_count; remainder = total % active_count
    return [base + (1 if idx < remainder else 0) for idx in range(active_count)]

# Builds one seed spec from an anchor candidate and spawn count.
def _route_seed_spec(anchor_item, spawn_count):
    return {"agent": anchor_item["agent"], "query": anchor_item.get("query", ""), "score": float(anchor_item.get("score", 0.0) or 0.0), "spawn_count": max(1, int(spawn_count or 1)),}

# Reassigns spawn counts across routes while preserving selected anchors.
def _with_redistributed_route_spawns(routes, total_thoughts):
    active_refs = []
    for route_index, route in enumerate(list(routes or [])):
        if not route.get("goal_agents"):
            continue
        for seed_index, _item in enumerate(list(route.get("seed_specs", []) or [])):
            active_refs.append((route_index, seed_index))
    spawn_counts = _route_spawn_counts(len(active_refs), total_thoughts); spawn_by_ref = dict(zip(active_refs, spawn_counts)); redistributed = []
    for idx, route in enumerate(list(routes or [])):
        next_route = dict(route); next_specs = []
        for seed_idx, item in enumerate(list(route.get("seed_specs", []) or [])):
            next_item = dict(item)
            if (idx, seed_idx) in spawn_by_ref:
                next_item["spawn_count"] = spawn_by_ref[(idx, seed_idx)]
            next_specs.append(next_item)
        next_route["seed_specs"] = next_specs; next_route["seed_agents"] = [item["agent"] for item in next_specs if item.get("agent") is not None]; next_route["goal_agents"] = list(route.get("goal_agents", []) or []); redistributed.append(next_route)
    return redistributed

# Chooses paired A-to-B and B-to-A anchor routes for native thought spawning.
def select_thought_routes(graph, seed_limit=None, goal_limit=None, total_thoughts=None):
    candidate_limit = max(1, int(seed_limit or TARGET_SEED_LIMIT), int(goal_limit or GOAL_AGENT_LIMIT),); pairs = _connected_anchor_pairs(graph, candidate_limit, pair_limit=ACTIVE_ANCHOR_PAIR_LIMIT,); routes = []
    for route_index, direction in enumerate(route_directions()):
        start_side = direction["start_side"]; goal_side = direction["goal_side"]; seed_queries = completion_queries_for_side(graph, start_side); goal_queries = completion_queries_for_side(graph, goal_side); seed_specs = []; goal_agents = []
        for pair in pairs:
            if direction["id"] == "a_to_b" and pair["a_to_b"]:
                seed_specs.append(_route_seed_spec(pair["a"], 1)); goal_agents.append(pair["b"]["agent"])
            elif direction["id"] == "b_to_a" and pair["b_to_a"]:
                seed_specs.append(_route_seed_spec(pair["b"], 1)); goal_agents.append(pair["a"]["agent"])
        seed_specs = [_route_seed_spec(item, 1) for item in _unique_anchor_items(seed_specs)]; goal_agents = _unique_goal_agents(goal_agents)
        routes.append({**direction, "index": route_index, "start_target": target_text_for_side(graph, start_side), "goal_target": target_text_for_side(graph, goal_side), "seed_queries": seed_queries, "goal_queries": goal_queries, "seed_specs": seed_specs, "seed_agents": [item["agent"] for item in seed_specs], "goal_agents": list(goal_agents), "anchor_locked": True,})
    return _with_redistributed_route_spawns(routes, total_thoughts)

# Deduplicates agents by index while preserving order.
def _unique_agents(agents):
    unique = []; seen = set()
    for agent in list(agents or []):
        index = getattr(agent, "index", None)
        if index is None or int(index) in seen:
            continue
        seen.add(int(index)); unique.append(agent)
    return unique

# Stores selected route metadata on ContextGraph and assigns seed/goal agents for thought workers.
def apply_thought_routes(graph, routes):
    graph.target_thought_routes = [dict(route) for route in list(routes or [])]; seed_specs = []; goal_agents = []
    for route in graph.target_thought_routes:
        route["seed_specs"] = list(route.get("seed_specs", []) or []); route["seed_agents"] = [item["agent"] for item in route["seed_specs"] if item.get("agent") is not None]; route["goal_agents"] = list(route.get("goal_agents", []) or []); seed_specs.extend(route["seed_specs"]); goal_agents.extend(route["goal_agents"])
    graph.target_seed_specs = seed_specs; graph.target_seed_agents = _unique_agents(item["agent"] for item in seed_specs if item.get("agent") is not None); graph.target_goal_agents = _unique_agents(goal_agents); graph.thoughts = []

# Refreshes route anchors and stores seed/goal agents on ContextGraph.
def refresh_target_seed_agents(graph, limit=None, total_thoughts=None):
    clear_thought_caches(); existing_routes = list(getattr(graph, "target_thought_routes", []) or [])
    if existing_routes and all(route.get("anchor_locked") for route in existing_routes):
        routes = _with_redistributed_route_spawns(existing_routes, total_thoughts)
    else:
        routes = select_thought_routes(graph, seed_limit=limit, goal_limit=GOAL_AGENT_LIMIT, total_thoughts=total_thoughts,)
    apply_thought_routes(graph, routes)
    if graph.target_seed_agents:
        thought_count = sum(int(item.get("spawn_count", 1)) for route in graph.target_thought_routes for item in route.get("seed_specs", [])); print(f"[THOUGHT] Seeded {thought_count} thought agents across " f"{len(graph.target_seed_agents)} target-matched information agents for '{graph.current_target}'.")
        if graph.current_subqueries:
            print(f"[THOUGHT] Seed queries: {', '.join(graph.current_subqueries[:10])}")
        if graph.target_a_focus_phrases:
            print(f"[THOUGHT] Target A focus (query expansion only): {', '.join(graph.target_a_focus_phrases)}")
        if graph.target_b_focus_phrases:
            print(f"[THOUGHT] Target B focus (query expansion only): {', '.join(graph.target_b_focus_phrases)}")
        for route in graph.target_thought_routes:
            seeds = route.get("seed_agents", []); goals = route.get("goal_agents", []); seed_preview = ", ".join(agent.ASU for agent in seeds[:5]) or "(none)"; goal_preview = ", ".join(agent.ASU for agent in goals[:5]) or "(none)"; print(f"[THOUGHT] Route {route.get('start_label')}->{route.get('goal_label')} " f"seeds: {seed_preview}")
            print(f"[THOUGHT] Route {route.get('start_label')}->{route.get('goal_label')} " f"goals: {goal_preview}")
    else: print(f"[THOUGHT] No target-matched thought agents found for '{graph.current_target}'.")

# Returns histories for thoughts that have finished walking.
def completed_thought_histories(graph):
    completed_thoughts = [thought for thought in graph.thoughts if len(getattr(thought, "history", [])) > 1]
    return completed_thoughts, [thought.history for thought in completed_thoughts]

# Returns whether a thought payload has enough steps and target match.
def _payload_passes_success_threshold(payload):
    if not isinstance(payload, dict): return False
    if payload.get("endpoint_reached"): return True
    return not (float(payload.get("match_target_a", 0.0) or 0.0) < SYNTHESIS_TARGET_MATCH_THRESHOLD or float(payload.get("match_target_b", 0.0) or 0.0) < SYNTHESIS_TARGET_MATCH_THRESHOLD or float(payload.get("lexical_target_a", 0.0) or 0.0) < PATH_TARGET_MATCH_THRESHOLD)

# Returns completed thoughts that satisfy endpoint and target-match thresholds.
def successful_thoughts(graph):
    strict_successes = [thought for thought in graph.thoughts if getattr(thought, "successful", False) and getattr(thought, "termination_reason", "") == "endpoint" and getattr(thought, "success_payload", None) and _payload_passes_success_threshold(getattr(thought, "success_payload", None))]
    if strict_successes: return strict_successes
    candidates = []
    for thought in graph.thoughts:
        if getattr(thought, "termination_reason", "") != "endpoint": continue
        if len(getattr(thought, "history", [])) - 1 < MIN_SUCCESS_STEPS: continue
        try:
            lexical_a, lexical_b = thought._path_target_lexical_matches()
            if lexical_a < PATH_TARGET_MATCH_THRESHOLD or lexical_b < PATH_TARGET_MATCH_THRESHOLD: continue
            payload = thought._build_relationship_statement()
        except Exception: continue
        if not _payload_passes_success_threshold(payload): continue
        thought.successful = True; thought.success_payload = payload; candidates.append(thought)
    return candidates

# Summarizes current thought progress for the dashboard and final report.
def thought_swarm_stats(graph):
    process = getattr(graph, "thought_process", None); process_alive = bool(process is not None and process.poll() is None); expected = int(getattr(graph, "thought_expected_count", 0) or 0); total = len(graph.thoughts) or expected
    alive = expected if process_alive and not graph.thoughts else sum(1 for thought in graph.thoughts if getattr(thought, "alive", False)); completed = sum(1 for thought in graph.thoughts if len(getattr(thought, "history", [])) > 1); successful = sum(1 for thought in graph.thoughts if getattr(thought, "successful", False))
    endpoint = sum(1 for thought in graph.thoughts if not getattr(thought, "alive", False) and getattr(thought, "termination_reason", "") == "endpoint"); dead = sum(1 for thought in graph.thoughts if not getattr(thought, "alive", False) and getattr(thought, "termination_reason", "") == "dead")
    max_hops = sum(1 for thought in graph.thoughts if not getattr(thought, "alive", False) and getattr(thought, "termination_reason", "") == "max_hops")
    with graph._thought_lock:
        active_workers = 1 if process_alive else int(graph.thought_active_count)
    return {"total": total, "alive": alive, "completed": completed, "successful": successful, "endpoint": endpoint, "dead": dead, "max_hops": max_hops, "active_workers": active_workers, "seeds": len(graph.target_seed_agents),}

# Returns the configured native thought-engine binary path.
def _thought_binary_path():
    return os.environ.get("CONTEXTGRAPH_THOUGHT_ENGINE") or str(Path(__file__).resolve().parents[3] / "venv" / "thought-engine")

# Returns graph neighbor indices for an agent from attached connectors.
def _neighbor_indices_for_agent(agent):
    agent_index = getattr(agent, "index", None)
    for conn in list(getattr(agent, "connectors", []) or []):
        source_agent = getattr(conn, "source_agent", None); target_agent = getattr(conn, "target", None); source_index = getattr(source_agent, "index", None); target_index = getattr(target_agent, "index", None)
        if source_index == agent_index and target_index is not None:
            yield int(target_index)

# Returns assigned goal agents reachable from a seed agent.
def _reachable_goal_agents(graph, start_agent, goal_agents):
    start_index = getattr(start_agent, "index", None)
    if start_index is None: return []
    goal_by_index = {int(goal.index): goal for goal in list(goal_agents or []) if getattr(goal, "index", None) is not None}
    if not goal_by_index: return []
    seen = {int(start_index)}; queue = [int(start_index)]; cursor = 0; found = []
    while cursor < len(queue):
        current_index = queue[cursor]; cursor += 1
        if current_index in goal_by_index:
            found.append(goal_by_index[current_index])
            if len(found) == len(goal_by_index):
                break
        agent = graph.agents_by_idx.get(current_index)
        if agent is None:
            continue
        for neighbor_index in _neighbor_indices_for_agent(agent):
            if neighbor_index in seen:
                continue
            seen.add(neighbor_index); queue.append(neighbor_index)
    found_indices = {int(goal.index) for goal in found}
    return [goal for goal in goal_agents if int(goal.index) in found_indices]

# Builds native thought-engine seed lines from selected route specs.
def _assigned_cpp_seed_lines(graph):
    lines = []; routes = list(getattr(graph, "target_thought_routes", []) or [])
    if not routes:
        routes = [{"index": 0, "seed_specs": list(getattr(graph, "target_seed_specs", []) or []), "goal_agents": list(getattr(graph, "target_goal_agents", []) or []),}]
    for route in routes:
        goal_agents = list(route.get("goal_agents", []) or [])[:GOAL_AGENT_LIMIT]
        if not goal_agents:
            continue
        route_index = int(route.get("index", 0) or 0); thought_index = 0
        for item in list(route.get("seed_specs", []) or []):
            agent = item.get("agent")
            if agent is None:
                continue
            reachable_goals = _reachable_goal_agents(graph, agent, goal_agents)
            if not reachable_goals:
                continue
            spawn_count = max(1, int(item.get("spawn_count", 1)))
            for _ in range(spawn_count):
                goal = reachable_goals[thought_index % len(reachable_goals)]; lines.append(f"seed {int(agent.index)} 1 {int(goal.index)} {route_index}"); thought_index += 1
    return lines

# Writes cpp thought input to storage or shared memory.
def _write_cpp_thought_input(graph, path):
    lines = [f"max_hops {int(MAX_THOUGHT_HOPS)}",]; lines.extend(_assigned_cpp_seed_lines(graph))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

# Returns the total native thought spawn count assigned across routes.
def _assigned_cpp_thought_count(graph):
    return len(_assigned_cpp_seed_lines(graph))

# Maps shared-memory connection offsets back to ContextGraph connection keys.
def _offset_key_map(graph):
    return {int(offset): key for key, offset in graph.connection_offsets.items()}

# Rebuilds a Connector wrapper for one native thought result record index.
def _connector_for_record_index(graph, connection_index, offset_keys):
    try:
        connection_index = int(connection_index)
    except (TypeError, ValueError):
        return None
    if connection_index < 0:
        return None
    offset = 4 + (connection_index * CONNECTION_RECORD_SIZE)
    try:
        (s_idx, relation_id, o_idx, _utility, subject_quant, subject_tense, subject_truth, predicate_quant, predicate_tense, predicate_truth,) = struct.unpack_from("<iiifiiiiii", graph.shm_connections.buf, offset)
    except struct.error:
        return None
    subject_sp = ConnectionEndpoint(subject_quant, subject_tense, subject_truth, s_idx); predicate_sp = ConnectionEndpoint(predicate_quant, predicate_tense, predicate_truth, o_idx); key = offset_keys.get(offset) or p_connection_graph._exact_connection_signature(s_idx, o_idx, relation_id, subject_sp, predicate_sp,)
    state = graph.connection_states.get(key, {}); specifics = graph.connection_specifics.get(key, {}); previous_agent_ids = graph.connection_previous_agents.get(key, ())
    return Connector(graph.shm_connections.buf, offset, graph.agents_by_idx, subject_sp=state.get("subject_sp") if isinstance(state.get("subject_sp"), ConnectionEndpoint) else subject_sp, predicate_sp=state.get("predicate_sp") if isinstance(state.get("predicate_sp"), ConnectionEndpoint) else predicate_sp, source=graph.connection_sources.get(key, "unknown"), evidence_text=graph.connection_texts.get(key, ""), subject_specifics=specifics.get("subject_specifics"), predicate_specifics=specifics.get("predicate_specifics"), connection_specifics=specifics.get("connection_specifics"), previous_agent_ids=previous_agent_ids,)

# Returns display text for an endpoint index in a native thought step.
def _endpoint_text(agent, endpoint):
    return str(getattr(agent, "ASU", "") or "").strip()

# Attaches source-backed specifics from a connection to a Thought path.
def _remember_specific_context(thought, details):
    for detail in list(details or []):
        source = detail.get("source", "")
        if thought._source_is_citable(source) and source not in thought.collected_sources:
            thought.collected_sources.append(source)
        suffix = f" [{display_source(source)}]" if thought._source_is_citable(source) else ""; thought._remember_note(f"{detail.get('text', '')}{suffix}")

# Converts one native route step into Thought history and evidence state.
def _append_cpp_step(thought, graph, step, offset_keys):
    conn = _connector_for_record_index(graph, step["connection_index"], offset_keys); from_agent = graph.agents_by_idx.get(step["from"]); to_agent = graph.agents_by_idx.get(step["to"])
    if conn is None or from_agent is None or to_agent is None:
        return False
    thought.current_asu = from_agent; specific_details = thought._specific_context_details(conn, limit=MAX_CONTEXT_SAMPLES); _remember_specific_context(thought, specific_details)
    if conn.evidence_text:
        thought._remember_note(thought._detail_from_record({"text": conn.evidence_text, "source": conn.source}, utility=getattr(conn, "utility", 0.0),))
    if thought._source_is_citable(conn.source) and conn.source not in thought.collected_sources:
        thought.collected_sources.append(conn.source)
    subject_agent = conn.source_agent; predicate_agent = conn.target; subject_text = _endpoint_text(subject_agent, conn.subject) if subject_agent else ""; predicate_text = _endpoint_text(predicate_agent, conn.predicate) if predicate_agent else ""; thought.last_asu_idx = from_agent.index; thought.current_asu = to_agent
    thought.history.append({"subject": subject_text, "subject_truth": conn.subject.truth, "subject_tense": conn.subject.tense, "subject_quant": conn.subject.quantifier, "subject_modifier": " ".join(conn.subject.modifier_value()).strip(), "subject_specifics": list(conn.subject_specifics), "relation_id": conn.relation_index, "truth": conn.truth, "predicate": predicate_text, "predicate_truth": conn.predicate.truth, "predicate_tense": conn.predicate.tense, "predicate_quant": conn.predicate.quantifier, "predicate_modifier": " ".join(conn.predicate.modifier_value()).strip(), "predicate_specifics": list(conn.predicate_specifics), "connection_specifics": list(conn.connection_specifics), "previous_agent_ids": list(conn.previous_agent_ids), "previous_agent": "", "source": conn.source, "evidence_text": conn.evidence_text, "specific_details": specific_details, "traversal_reversed": bool(step["reversed"]), "utility": round(float(getattr(conn, "utility", 0.0) or 0.0), 4), "score": round(float(thought.score), 2), "details": list(thought.collected_notes[-3:]),})
    thought.gather_stop_details()
    return True

# Reads native thought paths and converts them back into Python Thought histories.
def _load_cpp_thought_results(graph):
    if getattr(graph, "thought_results_loaded", False): return
    graph.thought_results_loaded = True; output_path = getattr(graph, "thought_output_path", "")
    if not output_path or not os.path.exists(output_path): print("[THOUGHT] C++ thought engine produced no output."); return
    records = []; current = None
    with open(output_path, "r") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            if parts[0] == "thought" and len(parts) >= 6:
                current = {"start": int(parts[1]), "end": int(parts[2]), "reason": parts[3], "success": bool(int(parts[4])), "goal": int(parts[6]) if len(parts) >= 7 else -1, "route": int(parts[7]) if len(parts) >= 8 else 0, "steps": [],}
            elif parts[0] == "step" and current is not None and len(parts) >= 5:
                current["steps"].append({"connection_index": int(parts[1]), "from": int(parts[2]), "to": int(parts[3]), "reversed": int(parts[4]),})
            elif parts[0] == "end" and current is not None:
                records.append(current); current = None
    routes = list(getattr(graph, "target_thought_routes", []) or []); route_by_index = {int(route.get("index", 0) or 0): route for route in routes}
    if not route_by_index:
        completion_queries = completion_queries_for_side(graph, "b"); route_by_index = {0: {"start_target": graph.current_target_a, "goal_target": graph.current_target_b, "goal_queries": completion_queries, "goal_agents": getattr(graph, "target_goal_agents", []), "seed_specs": getattr(graph, "target_seed_specs", []),}}
    seed_queries = {}
    for route in route_by_index.values():
        for item in list(route.get("seed_specs", []) or []):
            agent = item.get("agent")
            if agent is not None:
                seed_queries[(int(route.get("index", 0) or 0), int(agent.index))] = str(item.get("query", ""))
    offset_keys = _offset_key_map(graph); thoughts = []
    for record in records:
        start_agent = graph.agents_by_idx.get(record["start"])
        if start_agent is None:
            continue
        route = route_by_index.get(int(record.get("route", 0) or 0), route_by_index.get(0, {}))
        thought = Thought(start_agent, seed_query=seed_queries.get((int(record.get("route", 0) or 0), record["start"]), ""), start_target=route.get("start_target", ""), target_b=route.get("goal_target", ""), success_queries=route.get("goal_queries", []), goal_agents=route.get("goal_agents", []),); thought.route_id = route.get("id", "")
        thought.route_start_label = route.get("start_label", ""); thought.route_goal_label = route.get("goal_label", "")
        for step in record["steps"]:
            _append_cpp_step(thought, graph, step, offset_keys)
        thought.alive = False; thought.termination_reason = record["reason"]; thought.successful = bool(record["success"] and len(thought.history) > 1)
        if thought.successful:
            try:
                thought.success_payload = thought._build_relationship_statement()
            except Exception as exc:
                print(f"[THOUGHT] Failed to build success payload: {exc}"); thought.successful = False; thought.success_payload = None
            if isinstance(thought.success_payload, dict):
                thought.success_payload["endpoint_reached"] = True; thought.success_payload["direction"] = route.get("id", ""); thought.success_payload["route"] = (f"{route.get('start_label', '')}->{route.get('goal_label', '')}")
        thoughts.append(thought)
    graph.thoughts = thoughts; print(f"[THOUGHT] Loaded {len(thoughts)} C++ thought paths " f"{sum(1 for thought in thoughts if thought.successful)} reached assigned endpoints.")

# Writes native thought input and launches the C++ thought engine process.
def start_thought_workers(graph, stop_event, worker_count=None):
    if worker_count is None:
        worker_count = THINK_THREADS
    if graph.thought_phase_started and graph.shm_status.buf[0] == PHASE_THINKING:
        return
    stop_thought_workers(graph); requested_worker_count = max(1, int(worker_count)); refresh_target_seed_agents(graph, total_thoughts=requested_worker_count * THOUGHT_AGENTS_PER_WORKER,); graph.thought_expected_count = _assigned_cpp_thought_count(graph)
    with graph._thought_lock:
        graph.thought_active_count = 1 if graph.thought_expected_count else 0
    graph.thought_generation += 1; graph.thought_phase_started = True; graph.thought_results_loaded = False; graph.thought_process = None
    if graph.thought_expected_count <= 0:
        print("[THOUGHT] No thought workers launched because no assigned start/end nodes were found.")
        return
    binary_path = _thought_binary_path()
    if not os.path.exists(binary_path):
        print("[THOUGHT] C++ thought engine is not built. Run build.sh to compile venv/thought-engine."); graph.thought_results_loaded = True
        with graph._thought_lock:
            graph.thought_active_count = 0
        return
    unique = f"{os.getpid()}_{graph.thought_generation}"; graph.thought_input_path = os.path.join(tempfile.gettempdir(), f"graph_thought_{unique}.in"); graph.thought_output_path = os.path.join(tempfile.gettempdir(), f"graph_thought_{unique}.out"); _write_cpp_thought_input(graph, graph.thought_input_path); env = os.environ.copy()
    env["SHM_POS"] = graph.shm_pos.name; env["SHM_CONNECTIONS"] = graph.shm_connections.name; env["THOUGHT_INPUT"] = graph.thought_input_path; env["THOUGHT_OUTPUT"] = graph.thought_output_path; graph.thought_process = subprocess.Popen([binary_path], env=env)
    print(f"[THOUGHT] Launched C++ thought engine for {graph.thought_expected_count} thought agents " f"({THOUGHT_AGENTS_PER_WORKER} argument variants per worker).")

# Returns whether the native thought process exited and loads results once.
def thought_workers_finished(graph):
    if not graph.thought_phase_started: return False
    process = getattr(graph, "thought_process", None)
    if process is not None and process.poll() is None: return False
    if process is not None and process.returncode not in (0, None): print(f"[THOUGHT] C++ thought engine exited with code {process.returncode}.")
    _load_cpp_thought_results(graph)
    with graph._thought_lock:
        graph.thought_active_count = 0
    return True

# Stops thought workers and clears runtime state.
def stop_thought_workers(graph):
    graph.thought_generation += 1; process = getattr(graph, "thought_process", None)
    if process is not None and process.poll() is None:
        process.terminate()
        try: process.wait(timeout=0.5)
        except subprocess.TimeoutExpired: process.kill()
    elif process is not None: _load_cpp_thought_results(graph)
    graph.thought_process = None; graph.thought_phase_started = False
    with graph._thought_lock:
        graph.thought_active_count = 0
