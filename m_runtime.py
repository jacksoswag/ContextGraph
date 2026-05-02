import time, d_query_expander; from queue import Empty; from u_constants import (PHASE_IDLE, PHASE_RESEARCH, PHASE_SYNTHESIS, SCRAPE_DEFAULT_LIMIT, SCRAPE_REFINEMENT_ENABLED, SCRAPE_REFINEMENT_LIMIT, SCRAPE_REFINEMENT_OFFSET, THINK_THREADS, THOUGHT_AGENTS_PER_WORKER,); from p_graph import DecentralizedIntelligence; from utils import parse_command_payload; _ACTIVE_ENGINE = None

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

# Runs the Decentralized Intelligence loop, ingesting extracted connections and responding to dashboard commands.
def run_engine(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter):
    global _ACTIVE_ENGINE; print("[DI] Decentralized Intelligence Online. Awaiting input..."); brain = DecentralizedIntelligence(shm_names); _ACTIVE_ENGINE = brain
    while not stop_event.is_set():
        cmd_raw = bytes(brain.shm_cmd.buf).split(b"\x00")[0].decode("utf-8").strip()
        if cmd_raw:
            cmd_id, target_a, target_b = parse_command_payload(cmd_raw); current_status = brain.shm_status.buf[0]
            
# If a new command is received and the engine is idle, process it.
            if cmd_id != brain.last_cmd_id and current_status == PHASE_IDLE:
                brain.last_cmd_id = cmd_id; brain.current_command_id = cmd_id; brain.current_target_a = target_a.strip(); brain.current_target_b = target_b.strip()
                if brain.current_target_a and brain.current_target_b: 
# If both targets exist, combine them
                    brain.current_target = f"{brain.current_target_a} and {brain.current_target_b}"
                else: brain.current_target = brain.current_target_a or brain.current_target_b
                brain.reset_research_state(); brain.finalized_command_ids.discard(cmd_id); brain.clear_report(); brain.shm_status.buf[0] = PHASE_RESEARCH; print(f"[DI] New command: '{cmd_raw}'"); print(f"[DI] Target A: '{brain.current_target_a}' | Target B: '{brain.current_target_b}'"); _set_counter(sync_counter, 1)
                try:
                    query_plan = d_query_expander.build_query_plan(brain.current_target_a, brain.current_target_b); queries = brain.prime_query_plan(query_plan); print(f"[EXPANDER] Query plan: {len(brain.target_a_queries)} target-A, {len(brain.target_b_queries)}" f"target-B, {len(brain.bridge_queries)} bridge, {len(queries)} unique total.")
                    if brain.target_a_focus_phrases: print(f"[EXPANDER] Target A focus phrases: {', '.join(brain.target_a_focus_phrases)}")
                    if brain.target_b_focus_phrases: print(f"[EXPANDER] Target B focus phrases: {', '.join(brain.target_b_focus_phrases)}")
                    if queries:
                        _set_counters(sync_counter, ingest_counter, len(queries)); _queue_scrape_queries(scrape_queue, queries)
                    else:
                        print("[DI] No sub-queries generated. Returning to idle."); brain.shm_status.buf[0] = PHASE_IDLE; _set_counters(sync_counter, ingest_counter, 0)
                except Exception as e:
                    print(f"[DI] Query expansion failed: {e}"); brain.shm_status.buf[0] = PHASE_IDLE; _set_counters(sync_counter, ingest_counter, 0)
        while True:
            try: task = asu_queue.get_nowait()
            except Empty: break
            query, connections = brain.record_research_result(task)
            if task.get("error"): print(f"[SCRAPE] Query failed for '{query}': {task['error']}")
            research_pass = task.get("research_pass", "primary"); print(f"[SCRAPE:{research_pass}] {len(connections)} connections from " f"{len(task.get('blocks', []) or [])} blocks for query: '{query}'"); brain.ingest_research_connections(query, connections, restrict_to_existing=research_pass == "refine_existing"); _decrement_counter(ingest_counter)
        current_status = brain.shm_status.buf[0]
        if current_status == PHASE_SYNTHESIS:
            brain.run_final_synthesis()
        time.sleep(0.1)

# Starts native thought routing for the current Decentralized Intelligence graph if it is ready.
def launch_thought_workers(stop_event, worker_count=THINK_THREADS):
    brain = _ACTIVE_ENGINE
    if brain is None: return False
    brain.start_thought_workers(stop_event, worker_count=worker_count)
    return True

# Prunes the Decentralized Intelligence graph to the route-supporting subgraph after physics settles.
def prepare_active_subgraph():
    brain = _ACTIVE_ENGINE
    if brain is None: return None
    return brain.prepare_active_subgraph(total_thoughts=THINK_THREADS * THOUGHT_AGENTS_PER_WORKER)

# Returns whether the active thought process has completed.
def thought_workers_finished():
    brain = _ACTIVE_ENGINE
    if brain is None: return False
    return bool(brain.thought_workers_finished())

# Returns the current thought-routing stats for dashboard display.
def thought_worker_stats():
    brain = _ACTIVE_ENGINE
    if brain is None: return {"total": 0, "alive": 0, "completed": 0, "successful": 0, "endpoint": 0, "dead": 0, "max_hops": 0, "active_workers": 0, "seeds": 0}
    return brain.thought_swarm_stats()

# Stops any active thought process owned by the Decentralized Intelligence graph.
def stop_thought_workers():
    brain = _ACTIVE_ENGINE
    if brain is None: return
    brain.stop_thought_workers()

# Queues the secondary scrape pass from Decentralized Intelligence refinement queries.
def queue_refinement_scrapes(scrape_queue, sync_counter, ingest_counter):
    brain = _ACTIVE_ENGINE
    if brain is None or not SCRAPE_REFINEMENT_ENABLED: return False
    if getattr(brain, "refinement_pass_queued", False): return False
    queries = list(getattr(brain, "current_subqueries", []) or [])
    if not queries: return False
    brain.refinement_pass_queued = True; print("[RESEARCH] First scrape pass complete. Launching refinement pass " f"for {len(queries)} queries at offset {SCRAPE_REFINEMENT_OFFSET}."); _set_counters(sync_counter, ingest_counter, len(queries))
    _queue_scrape_queries(scrape_queue, queries, research_pass="refine_existing", limit=SCRAPE_REFINEMENT_LIMIT, offset=SCRAPE_REFINEMENT_OFFSET)
    return True
