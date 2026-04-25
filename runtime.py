import time
from queue import Empty
import d_query_expander
from constants import PHASE_IDLE, PHASE_RESEARCH, PHASE_SYNTHESIS, THINK_THREADS
from graph import Brain
from utils import parse_command_payload

_ACTIVE_BRAIN = None

def _set_counter(counter, value): # Atomically sets a shared counter to a given value
    with counter.get_lock(): counter.value = value

def _set_counters(sync_counter, ingest_counter, sync_value, ingest_value=None): # Atomically sets multiple shared counters
    if ingest_value is None: ingest_value = sync_value
    with sync_counter.get_lock(), ingest_counter.get_lock():
        sync_counter.value = sync_value
        ingest_counter.value = ingest_value

def _decrement_counter(counter): # Atomically decrements a shared counter
    with counter.get_lock(): counter.value = max(0, counter.value - 1)

def _queue_scrape_queries(scrape_queue, queries): # Puts queries into the scrape queue
    for query in queries: 
        scrape_queue.put({"query": query})

def run_brain(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter):
    global _ACTIVE_BRAIN
    print("[BRAIN] Brain Online. Awaiting input...")
    brain = Brain(shm_names)
    _ACTIVE_BRAIN = brain

    while not stop_event.is_set():
        cmd_raw = bytes(brain.shm_cmd.buf).split(b"\x00")[0].decode("utf-8").strip()
        if cmd_raw:
            cmd_id, target_a, target_b, tense_preference = parse_command_payload(cmd_raw)
            current_status = brain.shm_status.buf[0]

            # If a new command is received and the brain is idle, process it
            if cmd_id != brain.last_cmd_id and current_status == PHASE_IDLE:
                brain.last_cmd_id = cmd_id
                brain.current_command_id = cmd_id
                brain.current_target_a = target_a.strip()
                brain.current_target_b = target_b.strip()
                brain.current_tense_preference = tense_preference
                if brain.current_target_a and brain.current_target_b: # If both targets exist, combine them
                    brain.current_target = f"{brain.current_target_a} and {brain.current_target_b}"
                else:
                    brain.current_target = brain.current_target_a or brain.current_target_b
                brain.reset_research_state()
                brain.finalized_command_ids.discard(cmd_id)
                brain.clear_report()
                brain.shm_status.buf[0] = PHASE_RESEARCH

                print(f"[BRAIN] New command: '{cmd_raw}'")
                print(f"[BRAIN] Target A: '{brain.current_target_a}' | Target B: '{brain.current_target_b}'")
                print(f"[BRAIN] Tense preference: '{brain.current_tense_preference}'")
                _set_counter(sync_counter, 1)

                try:
                    query_plan = d_query_expander.build_query_plan(
                        brain.current_target_a,
                        brain.current_target_b,
                    )
                    queries = brain.prime_query_plan(query_plan)
                    print(
                        f"[EXPANDER] Query plan: "
                        f"{len(brain.target_a_queries)} target-A, "
                        f"{len(brain.target_b_queries)} target-B, "
                        f"{len(brain.bridge_queries)} bridge, "
                        f"{len(queries)} unique total."
                    )
                    if brain.target_a_focus_phrases:
                        print(
                            f"[EXPANDER] Target A focus phrases: "
                            f"{', '.join(brain.target_a_focus_phrases)}")
                    if brain.target_b_focus_phrases:
                        print(
                            f"[EXPANDER] Target B focus phrases: "
                            f"{', '.join(brain.target_b_focus_phrases)}")
                    if queries:
                        _set_counters(sync_counter, ingest_counter, len(queries))
                        _queue_scrape_queries(scrape_queue, queries)
                    else:
                        print("[BRAIN] No sub-queries generated. Returning to idle.")
                        brain.shm_status.buf[0] = PHASE_IDLE
                        _set_counters(sync_counter, ingest_counter, 0)
                except Exception as e:
                    print(f"[BRAIN] Query expansion failed: {e}")
                    brain.shm_status.buf[0] = PHASE_IDLE
                    _set_counters(sync_counter, ingest_counter, 0)

        while True:
            try: task = asu_queue.get_nowait()
            except Empty: break
            query, connections = brain.record_research_result(task)
            if task.get("error"): print(f"[SCRAPE] Query failed for '{query}': {task['error']}")
            print(f"[SCRAPE] {len(connections)} connections for query: '{query}'")
            brain.ingest_research_connections(query, connections)
            _decrement_counter(ingest_counter)

        current_status = brain.shm_status.buf[0]
        if current_status == PHASE_SYNTHESIS:
            brain.run_final_synthesis()
        time.sleep(0.1)


def launch_thought_workers(stop_event, worker_count=THINK_THREADS):
    brain = _ACTIVE_BRAIN
    if brain is None: return False
    brain.start_thought_workers(stop_event, worker_count=worker_count)
    return True

def thought_workers_finished():
    brain = _ACTIVE_BRAIN
    if brain is None: return False
    return bool(brain.thought_workers_finished())

def thought_worker_stats():
    brain = _ACTIVE_BRAIN
    if brain is None:
        return {
            "total": 0, "alive": 0, "completed": 0, "successful": 0,
            "endpoint": 0, "dead": 0, "max_hops": 0, "active_workers": 0, "seeds": 0,
        }
    return brain.thought_swarm_stats()


def stop_thought_workers():
    brain = _ACTIVE_BRAIN
    if brain is None: return
    brain.stop_thought_workers()
