import multiprocessing as mp
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from dotenv import load_dotenv # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from constants import (
    AGENT_POSITION_RECORD_BYTES,
    COMMAND_SHM_BYTES,
    CONNECTION_COUNT_BYTES,
    CONNECTION_RECORD_SIZE,
    DASHBOARD_PORT,
    DASHBOARD_URL,
    EXTRACTION_THREADS,
    MAX_AGENTS,
    MAX_CONNECTIONS,
    PHASE_IDLE,
    PHASE_PHYSICS,
    PHASE_RESEARCH,
    PHASE_STABLE,
    PHASE_SYNTHESIS,
    PHASE_THINKING,
    REPORT_SHM_BYTES,
    SCRAPE_THREADS,
    STATUS_SHM_BYTES,
)
from utils import remove_shm_from_resource_tracker, ensure_port_free, create_shm, write_report   

if __name__ == "__main__": remove_shm_from_resource_tracker()

def launch_webapp(names_dict): # Launches clean web UI (localhost:8000)
    ensure_port_free(DASHBOARD_PORT)
    print(f"[SYSTEM] Launching Dashboard UI on {DASHBOARD_URL}")
    env = os.environ.copy() # sets up env containing api key
    env["SHM_POS"]         = names_dict["pos"]
    env["SHM_CONNECTIONS"] = names_dict["connections"]
    env["SHM_STATUS"]      = names_dict["status"]
    env["SHM_COMMAND"]     = names_dict["command"]
    env["SHM_REPORT"]      = names_dict["report"]
    cmd = [sys.executable, str(PROJECT_ROOT / "frontend" / "ui.py")]
    return subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT) # opens

def launch_physics(names_dict): # calls c++ to launch physics
    physics_binary = PROJECT_ROOT / "physics_engine"
    print(f"[SYSTEM] Launching C++ Physics Engine ({os.path.basename(physics_binary)})")
    env = os.environ.copy()
    env["SHM_POS"]         = names_dict["pos"]
    env["SHM_CONNECTIONS"] = names_dict["connections"]
    env["SHM_STATUS"]      = names_dict["status"]
    return subprocess.Popen([str(physics_binary)], env=env, cwd=PROJECT_ROOT) # opens

def prune_after_physics(runtime_api):
    print("[SYSTEM] Pruning active connector subgraph after physics...")
    subgraph_summary = runtime_api.prepare_active_subgraph()
    if not subgraph_summary:
        return
    print(
        "[SYSTEM] Active connector subgraph: "
        f"{subgraph_summary['kept_connections']} connections, "
        f"{subgraph_summary['kept_nodes']} nodes, "
        f"start anchors {subgraph_summary['active_seed_count']}/{subgraph_summary['seed_count']}, "
        f"goal anchors {subgraph_summary['active_goal_count']}/{subgraph_summary['goal_count']}."
    )
    for route in subgraph_summary.get("routes", []):
        print(
            "[SYSTEM] Route "
            f"{route.get('route', '?')}: "
            f"{route.get('seed_count', 0)} starts, "
            f"{route.get('goal_count', 0)} goals, "
            f"{route.get('core_connections', 0)} core connections."
        )
    for item in subgraph_summary.get("anchor_pair_diagnostics", [])[:8]:
        status = "selected" if item.get("selected") else "rejected"
        print(
            "[SYSTEM] Anchor pair "
            f"{status}: A='{item.get('a', '')}' B='{item.get('b', '')}' "
            f"A->B edges={item.get('a_to_b_edges', 0)} "
            f"B->A edges={item.get('b_to_a_edges', 0)} "
            f"sources={item.get('sources', 0)}."
        )

def manage_shm(): 
    # pos_shm: agent positions, connection_shm: connection data, command_shm: prompt payload, status_shm: status, report_shm: reports
    anchors = {}

    def _create_and_anchor(name, size): # creates and anchors SHM
        unique_name, shm = create_shm(name, size)
        anchors[name] = shm
        return unique_name, shm

    pos_name, pos_shm = _create_and_anchor(
        "pos",
        MAX_AGENTS * AGENT_POSITION_RECORD_BYTES,
    )
    report_name, report_shm = _create_and_anchor("report", REPORT_SHM_BYTES)
    command_name, command_shm = _create_and_anchor("command", COMMAND_SHM_BYTES)
    status_name, status_shm = _create_and_anchor("status", STATUS_SHM_BYTES)
    connection_name, connection_shm = _create_and_anchor(
        "connections",
        CONNECTION_COUNT_BYTES + MAX_CONNECTIONS * CONNECTION_RECORD_SIZE,
    )

    for shm in (pos_shm, report_shm, command_shm, status_shm, connection_shm):
        shm.buf[:] = b"\x00" * shm.size # zero out SHM segments to prevent garbage data

    names = { # stores it in raw bytes
        "pos":     pos_name,
        "report":  report_name,
        "command": command_name,
        "connections": connection_name,
        "status":  status_name,
    }
    return names, anchors



def main():
    runtime_api = None
    scrapers = []
    extractors = []
    brain_thread = None
    server_proc = None
    physics_proc = None

    shm_names, shm_handles = manage_shm() # initializes and returns shared memory
    status_shm = shm_handles["status"]
    report_shm = shm_handles["report"]

    scrape_queue  = mp.Queue() # distributes scrape workers to separate processes
    extract_queue = mp.Queue()
    asu_queue     = mp.Queue()
    stop_event    = mp.Event()
    sync_counter  = mp.Value('i', 0)
    ingest_counter = mp.Value('i', 0)

    try:
        server_proc        = launch_webapp(shm_names) # launch web UI
        status_shm.buf[0]  = PHASE_IDLE # updates phase
        zero_ticks         = 0  # consecutive ticks where sync_counter == 0 and queue empty

        from d_scrape_worker import scrape_worker_loop
        from d_extraction_worker import extraction_worker_loop
        import runtime as runtime_api

        # ------------ THREAD SETUP ------------
        for i in range(SCRAPE_THREADS): # launch 8 scrape workers
            p = mp.Process(target=scrape_worker_loop, # calls scraper_worker.py for each thread
                args=(scrape_queue, extract_queue, stop_event, sync_counter), name=f"ScrapeWorker_{i}", daemon=True)
            p.start()
            scrapers.append(p)

        for i in range(EXTRACTION_THREADS):
            p = mp.Process(
                target=extraction_worker_loop,
                args=(extract_queue, asu_queue, stop_event),
                name=f"ExtractWorker_{i}",
                daemon=True,
            )
            p.start()
            extractors.append(p)

        brain_thread = threading.Thread(
            target=runtime_api.run_brain, # brain now lives in the main process instead of its own child process
            args=(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter),
            name="brain",
            daemon=True,
        )
        brain_thread.start()

        print("[SYSTEM] Engine idle. Awaiting prompts.")

        # ------------ MAIN LOOP ------------
        while not stop_event.is_set(): # continues if program is still running
            time.sleep(1)
            current_status = status_shm.buf[0] # checks phase
            
            # The following blocks check for phase end conditions and start next phase.
            # Researching → Physics
            if current_status == PHASE_RESEARCH: # Require 3 consecutive quiet ticks to guard against the sync_counter briefly hitting 0 between query batches.
                with sync_counter.get_lock(), ingest_counter.get_lock():
                    quiet = sync_counter.value <= 0 and ingest_counter.value <= 0
                if quiet: # checks if agents are done processing and query list is empty
                    zero_ticks += 1
                    if zero_ticks >= 3: # ensures that it stays in research mode for at least 3 ticks to prevent race conditions
                        if runtime_api.queue_refinement_scrapes(scrape_queue, sync_counter, ingest_counter):
                            zero_ticks = 0
                            continue
                        print("[SYSTEM] Research complete. Launching physics simulation...")
                        status_shm.buf[0] = PHASE_PHYSICS
                        physics_proc = launch_physics(shm_names)
                        if physics_proc is None:
                            print("[SYSTEM] Physics skipped. Pruning graph before marking stable.")
                            prune_after_physics(runtime_api)
                            status_shm.buf[0] = PHASE_STABLE
                        zero_ticks = 0
                else: zero_ticks = 0

            # Physics → Watching for stability
            if physics_proc and physics_proc.poll() is not None: # checks if physics engine is done
                if physics_proc.returncode == 0:
                    print("[SYSTEM] Physics stabilized.")
                    prune_after_physics(runtime_api)
                    status_shm.buf[0] = PHASE_STABLE # updates phase, will be read by synthesis.py
                elif physics_proc.returncode == 2:
                    print("[SYSTEM] Physics reached its time limit without stabilizing. Proceeding with the current layout.")
                    prune_after_physics(runtime_api)
                    status_shm.buf[0] = PHASE_STABLE
                else:
                    error = f"[error] Physics engine failed with exit code {physics_proc.returncode}."
                    print(f"[SYSTEM] {error}")
                    write_report(report_shm, error)
                    status_shm.buf[0] = PHASE_IDLE
                physics_proc = None

            if server_proc and server_proc.poll() is not None:
                print("[SYSTEM] Web server terminated. Shutting down...")
                stop_event.set()
            
            # Stable -> Thinking (Starting the autonomous analysis)
            if current_status == PHASE_STABLE:
                print("[SYSTEM] Graph stabilized. Starting Thought Processes...")
                status_shm.buf[0] = PHASE_THINKING
                if runtime_api.launch_thought_workers(stop_event):
                    print("[SYSTEM] Thought workers launched.")
                else:
                    print("[SYSTEM] Brain is not ready to launch thought workers yet.")
                
            # Thinking -> Synthesis (Moving from analysis to report writing)
            if current_status == PHASE_THINKING:
                if not brain_thread.is_alive():
                    raise RuntimeError("Brain thread terminated during thought processing.")
                runtime_api.launch_thought_workers(stop_event)
                if runtime_api.thought_workers_finished():
                    stats = runtime_api.thought_worker_stats()
                    print(
                        f"[SYSTEM] Thought workers finished: "
                        f"{stats['endpoint']} at endpoints / {stats['dead']} dead / "
                        f"{stats.get('max_hops', 0)} max hops / {stats['successful']} successful."
                    )
                    status_shm.buf[0] = PHASE_SYNTHESIS
    except KeyboardInterrupt: # stops everything at keyboard interrupt (ctrl+c)
        print("[SYSTEM] Interrupt received.")
        stop_event.set()
    except Exception as e: # stops everything at any error
        print(f"[SYSTEM] Critical error: {e}")
        stop_event.set()
    finally: # stops everything at end of program, cleans up technical stuff
        print("[SYSTEM] Cleaning up...")
        stop_event.set()
        if runtime_api:
            runtime_api.stop_thought_workers()
        for p in scrapers:
            p.terminate()
        for p in extractors:
            p.terminate()
        if server_proc:
            server_proc.terminate()
        if physics_proc:
            physics_proc.terminate()
        if brain_thread and brain_thread.is_alive():
            brain_thread.join(timeout=2.0)
        for shm in shm_handles.values(): # closes and removes shared memory segments
            try:
                shm.close()
                shm.unlink()
            except (BufferError, FileNotFoundError, OSError):
                pass
        print("[SYSTEM] Engine offline.")

if __name__ == "__main__": # entry point for program, just runs main()
    main()
