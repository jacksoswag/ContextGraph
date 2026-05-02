import struct, p_connection_graph; from collections import deque; from u_constants import (ACTIVE_SUBGRAPH_CANDIDATE_MULTIPLIER, ACTIVE_SUBGRAPH_CONTEXT_HOPS, AGENT_POSITION_RECORD_BYTES, CONNECTION_RECORD_SIZE, GOAL_AGENT_LIMIT, TARGET_SEED_LIMIT, THINK_THREADS, THOUGHT_AGENTS_PER_WORKER,); from o_connection import Connector
import p_thought_process as thought_process
# Builds adjacency from connection keys for active subgraph pruning.
def _connection_adjacency(keys, directed=False):
    adjacency = {}
    for key in keys:
        left, right = int(key[0]), int(key[1]); adjacency.setdefault(left, []).append((right, key))
        if not directed: adjacency.setdefault(right, []).append((left, key))
    return adjacency
# Builds reverse adjacency for active subgraph reachability checks.
def _reverse_adjacency(adjacency):
    reverse = {}
    for source, edges in adjacency.items():
        for target, key in edges:
            reverse.setdefault(int(target), []).append((int(source), key))
    return reverse
# Returns reachable nodes for active subgraph pruning.
def _reachable_nodes(adjacency, starts):
    seen = set(); queue = deque()
    for start in starts:
        try: start = int(start)
        except (TypeError, ValueError): continue
        if start in seen: continue
        seen.add(start); queue.append(start)
    while queue:
        node = queue.popleft()
        for neighbor, _key in adjacency.get(node, ()):
            neighbor = int(neighbor)
            if neighbor in seen: continue
            seen.add(neighbor); queue.append(neighbor)
    return seen
# Returns connection keys that lie on at least one directed seed-to-goal route.
def _edges_on_any_directed_route(adjacency, seed_indices, goal_indices):
    seed_indices = {int(index) for index in seed_indices}; goal_indices = {int(index) for index in goal_indices}
    if not seed_indices or not goal_indices:
        return set(), set()
    from_seeds = _reachable_nodes(adjacency, seed_indices); to_goals = _reachable_nodes(_reverse_adjacency(adjacency), goal_indices)
    # Keep every directed connection that can sit on some start->goal route:
    # start reaches edge.source, then edge.target reaches a goal. This is not a
    # shortest-path extraction; it preserves all possible reasoning trains.
    kept_edges = {key for source, edges in adjacency.items() if int(source) in from_seeds for target, key in edges if int(target) in to_goals}; kept_nodes = {int(idx) for key in kept_edges for idx in (key[0], key[1])}
    return kept_nodes, kept_edges
# Returns whether connection has specific information for active subgraph pruning.
def _connection_has_specific_information(brain, key):
    specifics = getattr(brain, "connection_specifics", {}).get(key, {}) or {}
    for field in ("subject_specifics", "predicate_specifics", "connection_specifics"):
        if list(specifics.get(field, []) or []):
            return True
    return False
# Builds expand with specific adjacent context for active subgraph pruning.
def _expand_with_specific_adjacent_context(brain, adjacency, core_nodes, kept_nodes, kept_edges):
    specific_nodes = set(); specific_edges = set()
    for node in set(core_nodes):
        for neighbor, key in adjacency.get(int(node), ()):
            if not _connection_has_specific_information(brain, key): continue
            kept_edges.add(key); kept_nodes.add(int(neighbor)); specific_edges.add(key)
            if int(neighbor) not in core_nodes: specific_nodes.add(int(neighbor))
    return specific_nodes, specific_edges
# Returns how many seed and goal anchors to consider for pruning.
def _anchor_candidate_limit(brain):
    candidate_multiplier = max(1, int(ACTIVE_SUBGRAPH_CANDIDATE_MULTIPLIER)); base_limit = max(TARGET_SEED_LIMIT, GOAL_AGENT_LIMIT); total_agents = max(len(getattr(brain, "agents_by_idx", {}) or {}), len(getattr(brain, "agents", {}) or {}))
    if total_agents <= 0: return base_limit
    return min(total_agents, max(base_limit, base_limit * candidate_multiplier * 2))
# Selects functioning anchors for active subgraph pruning.
def _select_functioning_anchors(brain, total_thoughts):
    thought_process.clear_thought_caches(); candidate_limit = _anchor_candidate_limit(brain); selected_routes = thought_process.select_thought_routes(brain, seed_limit=candidate_limit, goal_limit=candidate_limit, total_thoughts=total_thoughts); thought_process.apply_thought_routes(brain, selected_routes)
# Expands expand with adjacent context for active subgraph pruning.
def _expand_with_adjacent_context(adjacency, frontier_nodes, kept_nodes=None, kept_edges=None):
    kept_nodes = set(frontier_nodes if kept_nodes is None else kept_nodes); kept_edges = set(() if kept_edges is None else kept_edges); frontier = set(frontier_nodes)
    for _ in range(max(0, int(ACTIVE_SUBGRAPH_CONTEXT_HOPS))):
        next_frontier = set()
        for node in frontier:
            for neighbor, key in adjacency.get(int(node), ()):
                kept_edges.add(key)
                if int(neighbor) not in kept_nodes: next_frontier.add(int(neighbor))
                kept_nodes.add(int(neighbor))
        frontier = next_frontier
        if not frontier: break
    return kept_nodes, kept_edges
# Clears agent connectors caches or state.
def _clear_agent_connectors(brain):
    for agent in brain.agents_by_idx.values():
        agent.connectors = []
# Zeros positions for agents removed from the active subgraph.
def _zero_inactive_agent_positions(brain, active_nodes):
    for idx in list(brain.agents_by_idx):
        if int(idx) in active_nodes: continue
        offset = int(idx) * AGENT_POSITION_RECORD_BYTES; struct.pack_into("ffffff", brain.shm_pos.buf, offset, 0, 0, 0, 0, 0, 0)
# Builds filter connection dict for active subgraph pruning.
def _filter_connection_dict(mapping, kept_edges):
    return {key: value for key, value in mapping.items() if key in kept_edges}
# Repacks repack connections for active subgraph pruning.
def _repack_connections(brain, kept_edges):
    old_offsets = dict(brain.connection_offsets); old_count = int(brain.connection_count); kept_keys = sorted(kept_edges, key=lambda key: old_offsets.get(key, 10**12)); _clear_agent_connectors(brain); brain.connection_offsets = {}; brain.connection_buckets = {}; brain.seen_connections = set()
    for new_index, key in enumerate(kept_keys):
        old_offset = old_offsets.get(key)
        if old_offset is None: continue
        subject_sp, predicate_sp = p_connection_graph._state_for_key(brain, key); utility = p_connection_graph._read_connection_utility(brain, old_offset); new_offset = 4 + (new_index * CONNECTION_RECORD_SIZE); s_idx, o_idx, relation_id = int(key[0]), int(key[1]), int(key[2])
        p_connection_graph._pack_connection_record(brain, new_offset, s_idx, relation_id, o_idx, utility, subject_sp, predicate_sp); p_connection_graph._register_connection_key(brain, key, offset=new_offset); specifics = brain.connection_specifics.get(key, {})
        connector = Connector(brain.shm_connections.buf, new_offset, brain.agents_by_idx, subject_sp=subject_sp, predicate_sp=predicate_sp, source=brain.connection_sources.get(key, "unknown"), evidence_text=brain.connection_texts.get(key, ""), subject_specifics=specifics.get("subject_specifics"), predicate_specifics=specifics.get("predicate_specifics"), connection_specifics=specifics.get("connection_specifics"), previous_agent_ids=brain.connection_previous_agents.get(key, ()),)
        p_connection_graph._attach_connector(connector)
    brain.connection_count = len(kept_keys); struct.pack_into("i", brain.shm_connections.buf, 0, brain.connection_count); clear_start = 4 + (brain.connection_count * CONNECTION_RECORD_SIZE); clear_end = 4 + (old_count * CONNECTION_RECORD_SIZE)
    if clear_end > clear_start:
        brain.shm_connections.buf[clear_start:clear_end] = b"\x00" * (clear_end - clear_start)
    brain.connection_sources = _filter_connection_dict(brain.connection_sources, kept_edges); brain.connection_states = _filter_connection_dict(brain.connection_states, kept_edges); brain.connection_specifics = _filter_connection_dict(brain.connection_specifics, kept_edges)
    brain.connection_previous_agents = _filter_connection_dict(brain.connection_previous_agents, kept_edges); brain.connection_texts = _filter_connection_dict(brain.connection_texts, kept_edges)
# Filters filter anchor state for active subgraph pruning.
def _filter_anchor_state(brain, active_nodes):
    filtered_routes = []
    for route in list(getattr(brain, "target_thought_routes", []) or []):
        filtered_route = dict(route); filtered_route["seed_specs"] = [item for item in list(route.get("seed_specs", []) or []) if getattr(item.get("agent"), "index", None) in active_nodes]; filtered_route["seed_agents"] = [item["agent"] for item in filtered_route["seed_specs"]]
        filtered_route["goal_agents"] = [agent for agent in list(route.get("goal_agents", []) or []) if getattr(agent, "index", None) in active_nodes]; filtered_routes.append(filtered_route)
    thought_process.apply_thought_routes(brain, filtered_routes)
# Keeps only route-relevant nodes and connections before final thinking and synthesis.
def prepare_active_subgraph(brain, total_thoughts=None):
    total_thoughts = int(total_thoughts or (THINK_THREADS * THOUGHT_AGENTS_PER_WORKER)); all_keys = set(brain.connection_offsets); directed_adjacency = _connection_adjacency(all_keys, directed=True); context_adjacency = _connection_adjacency(all_keys, directed=False); _select_functioning_anchors(brain, total_thoughts); core_nodes, core_edges = set(), set()
    route_summaries = []
    for route in list(getattr(brain, "target_thought_routes", []) or []):
        seed_indices = {int(item["agent"].index) for item in list(route.get("seed_specs", []) or []) if item.get("agent") is not None}; goal_indices = {int(agent.index) for agent in list(route.get("goal_agents", []) or []) if getattr(agent, "index", None) is not None}
        if seed_indices and goal_indices and all_keys:
            route_nodes, route_edges = _edges_on_any_directed_route(directed_adjacency, seed_indices, goal_indices)
        else:
            route_nodes, route_edges = set(), set()
        core_nodes.update(route_nodes); core_edges.update(route_edges); route_summaries.append({"id": route.get("id", ""), "route": f"{route.get('start_label', '')}->{route.get('goal_label', '')}", "seed_count": len(seed_indices), "goal_count": len(goal_indices), "core_nodes": len(route_nodes), "core_connections": len(route_edges)})
    kept_nodes = set(core_nodes); kept_edges = set(core_edges); specific_nodes, specific_edges = _expand_with_specific_adjacent_context(brain, context_adjacency, core_nodes, kept_nodes, kept_edges); kept_nodes, kept_edges = _expand_with_adjacent_context(context_adjacency, core_nodes, kept_nodes, kept_edges)
    context_node_count = max(0, len(kept_nodes) - len(core_nodes)); context_edge_count = max(0, len(kept_edges) - len(core_edges)); removed_connections = max(0, int(brain.connection_count) - len(kept_edges)); removed_nodes = max(0, len(brain.agents_by_idx) - len(kept_nodes)); _repack_connections(brain, kept_edges)
    _zero_inactive_agent_positions(brain, kept_nodes); _filter_anchor_state(brain, kept_nodes)
    summary = {"seed_count": sum(item["seed_count"] for item in route_summaries), "goal_count": sum(item["goal_count"] for item in route_summaries), "active_seed_count": len(brain.target_seed_agents), "active_goal_count": len(brain.target_goal_agents), "kept_nodes": len(kept_nodes), "removed_nodes": removed_nodes, "kept_connections": len(kept_edges), "removed_connections": removed_connections, "context_nodes": context_node_count, "context_connections": context_edge_count, "specific_context_nodes": len(specific_nodes), "specific_context_connections": len(specific_edges - core_edges), "routes": route_summaries, "anchor_pair_diagnostics": list(getattr(brain, "anchor_pair_diagnostics", []) or []),}
    route_text = "; ".join(f"{item['route']} seeds {item['seed_count']} goals {item['goal_count']} core {item['core_connections']}" for item in route_summaries) or "none"; diagnostics = list(getattr(brain, "anchor_pair_diagnostics", []) or [])
    for item in diagnostics:
        status = "selected" if item.get("selected") else "rejected"; brain._conn_log.append(
            "[ANCHOR_PAIR:" f"{status}] A[{item.get('a', '')}] B[{item.get('b', '')}] | " f"Scores A[{item.get('a_score', 0)}] B[{item.get('b_score', 0)}] | " f"Grounding A[{item.get('a_grounding', 0)}] B[{item.get('b_grounding', 0)}] | " f"RouteEdges A->B[{item.get('a_to_b_edges', 0)}] " f"B->A[{item.get('b_to_a_edges', 0)}] | "
            f"Sources[{item.get('sources', 0)}] " f"SpecificEdges[{item.get('specific_edges', 0)}] | " f"Rank{item.get('rank', ())}\n"
        )
    brain._conn_log.append(
        "[ACTIVE_SUBGRAPH] " f"Kept {summary['kept_connections']} connections / {summary['kept_nodes']} nodes; " f"removed {summary['removed_connections']} connections / {summary['removed_nodes']} nodes; " f"context ring +{summary['context_connections']} connections / +{summary['context_nodes']} nodes; "
        f"specific adjacent +{summary['specific_context_connections']} connections / " f"+{summary['specific_context_nodes']} nodes; " f"active anchors starts:{summary['active_seed_count']}/{summary['seed_count']} " f"goals:{summary['active_goal_count']}/{summary['goal_count']}; " f"routes: {route_text}.\n"
    ); brain.flush_conn_log()
    return summary
