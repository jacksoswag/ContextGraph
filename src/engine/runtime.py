import time; import engine.extract.query_expander as d_query_expander; from queue import Empty; from engine.common.constants import (PHASE_IDLE, PHASE_RESEARCH, PHASE_SYNTHESIS, SCRAPE_DEFAULT_LIMIT, SCRAPE_REFINEMENT_ENABLED, SCRAPE_REFINEMENT_LIMIT, SCRAPE_REFINEMENT_OFFSET, THINK_THREADS, THOUGHT_AGENTS_PER_WORKER,); from engine.graph.graph import ContextGraph; from engine.common.shm import parse_command_payload; _ACTIVE_ENGINE = None

# Sets one multiprocessing counter while holding its lock.
def _set_counter(counter, value): 
# Atomically sets a shared counter to a given value
    with counter.get_lock(): counter.value = value

# Sets sync and ingest counters together for scrape/extraction accounting.
def _set_counters(sync_counter, ingest_counter, sync_value, ingest_value=None): 
# Atomically sets multiple shared counters
    if ingest_value is None: ingest_value = sync_value
    with sync_counter.get_lock(), ingest_counter.get_lock():
        sync_counter.value = sync_value; ingest_counter.value = ingest_value

# Decrements one multiprocessing counter and returns the new value.
def _decrement_counter(counter): 
# Atomically decrements a shared counter
    with counter.get_lock(): counter.value = max(0, counter.value - 1)

# Pushes query tasks into the scraper queue for a named research pass.
def _queue_scrape_queries(scrape_queue, queries, research_pass="primary", limit=SCRAPE_DEFAULT_LIMIT, offset=0): 
# Puts queries into the scrape queue
    for query in queries:
        scrape_queue.put({"query": query, "research_pass": research_pass, "limit": int(limit), "offset": int(offset),})

# Runs the ContextGraph loop, ingesting extracted connections and responding to dashboard commands.
def run_engine(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter):
    global _ACTIVE_ENGINE; print("[GRAPH] ContextGraph Online. Awaiting input..."); graph = ContextGraph(shm_names); _ACTIVE_ENGINE = graph
    while not stop_event.is_set():
        cmd_raw = bytes(graph.shm_cmd.buf).split(b"\x00")[0].decode("utf-8").strip()
        if cmd_raw:
            cmd_id, target_a, target_b = parse_command_payload(cmd_raw); current_status = graph.shm_status.buf[0]
            
# If a new command is received and the engine is idle, process it.
            if cmd_id != graph.last_cmd_id and current_status == PHASE_IDLE:
                graph.last_cmd_id = cmd_id; graph.current_command_id = cmd_id; graph.current_target_a = target_a.strip(); graph.current_target_b = target_b.strip()
                if graph.current_target_a and graph.current_target_b: 
# If both targets exist, combine them
                    graph.current_target = f"{graph.current_target_a} and {graph.current_target_b}"
                else: graph.current_target = graph.current_target_a or graph.current_target_b
                graph.reset_research_state(); graph.finalized_command_ids.discard(cmd_id); graph.clear_report(); graph.shm_status.buf[0] = PHASE_RESEARCH; print(f"[GRAPH] New command: '{cmd_raw}'"); print(f"[GRAPH] Target A: '{graph.current_target_a}' | Target B: '{graph.current_target_b}'"); _set_counter(sync_counter, 1)
                try:
                    query_plan = d_query_expander.build_query_plan(graph.current_target_a, graph.current_target_b); queries = graph.prime_query_plan(query_plan); print(f"[EXPANDER] Query plan: {len(graph.target_a_queries)} target-A, {len(graph.target_b_queries)}" f"target-B, {len(graph.bridge_queries)} bridge, {len(queries)} unique total.")
                    if graph.target_a_focus_phrases: print(f"[EXPANDER] Target A focus phrases: {', '.join(graph.target_a_focus_phrases)}")
                    if graph.target_b_focus_phrases: print(f"[EXPANDER] Target B focus phrases: {', '.join(graph.target_b_focus_phrases)}")
                    if queries:
                        _set_counters(sync_counter, ingest_counter, len(queries)); _queue_scrape_queries(scrape_queue, queries)
                    else:
                        print("[GRAPH] No sub-queries generated. Returning to idle."); graph.shm_status.buf[0] = PHASE_IDLE; _set_counters(sync_counter, ingest_counter, 0)
                except Exception as e:
                    print(f"[GRAPH] Query expansion failed: {e}"); graph.shm_status.buf[0] = PHASE_IDLE; _set_counters(sync_counter, ingest_counter, 0)
        while True:
            try: task = asu_queue.get_nowait()
            except Empty: break
            query, connections = graph.record_research_result(task)
            if task.get("error"): print(f"[SCRAPE] Query failed for '{query}': {task['error']}")
            research_pass = task.get("research_pass", "primary"); print(f"[SCRAPE:{research_pass}] {len(connections)} connections from " f"{len(task.get('blocks', []) or [])} blocks for query: '{query}'"); graph.ingest_research_connections(query, connections, restrict_to_existing=research_pass == "refine_existing"); _decrement_counter(ingest_counter)
        current_status = graph.shm_status.buf[0]
        if current_status == PHASE_SYNTHESIS:
            graph.run_final_synthesis()
        time.sleep(0.1)

# Starts native thought routing for the current ContextGraph graph if it is ready.
def launch_thought_workers(stop_event, worker_count=THINK_THREADS):
    graph = _ACTIVE_ENGINE
    if graph is None: return False
    graph.start_thought_workers(stop_event, worker_count=worker_count)
    return True

# Prunes the ContextGraph graph to the route-supporting subgraph after physics settles.
def prepare_active_subgraph():
    graph = _ACTIVE_ENGINE
    if graph is None: return None
    return graph.prepare_active_subgraph(total_thoughts=THINK_THREADS * THOUGHT_AGENTS_PER_WORKER)

# Returns whether the active thought process has completed.
def thought_workers_finished():
    graph = _ACTIVE_ENGINE
    if graph is None: return False
    return bool(graph.thought_workers_finished())

# Returns the current thought-routing stats for dashboard display.
def thought_worker_stats():
    graph = _ACTIVE_ENGINE
    if graph is None: return {"total": 0, "alive": 0, "completed": 0, "successful": 0, "endpoint": 0, "dead": 0, "max_hops": 0, "active_workers": 0, "seeds": 0}
    return graph.thought_swarm_stats()

# Stops any active thought process owned by the ContextGraph graph.
def stop_thought_workers():
    graph = _ACTIVE_ENGINE
    if graph is None: return
    graph.stop_thought_workers()

# Queues the secondary scrape pass from ContextGraph refinement queries.
def queue_refinement_scrapes(scrape_queue, sync_counter, ingest_counter):
    graph = _ACTIVE_ENGINE
    if graph is None or not SCRAPE_REFINEMENT_ENABLED: return False
    if getattr(graph, "refinement_pass_queued", False): return False
    queries = list(getattr(graph, "current_subqueries", []) or [])
    if not queries: return False
    graph.refinement_pass_queued = True; print("[RESEARCH] First scrape pass complete. Launching refinement pass " f"for {len(queries)} queries at offset {SCRAPE_REFINEMENT_OFFSET}."); _set_counters(sync_counter, ingest_counter, len(queries))
    _queue_scrape_queries(scrape_queue, queries, research_pass="refine_existing", limit=SCRAPE_REFINEMENT_LIMIT, offset=SCRAPE_REFINEMENT_OFFSET)
    return True
