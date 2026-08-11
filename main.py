import multiprocessing as mp; import os, sys, time, threading, subprocess; from pathlib import Path
from dotenv import load_dotenv # type: ignore
PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
for _p in (str(SRC), str(PROJECT_ROOT)):
    if _p not in sys.path: sys.path.insert(0, _p)
ENV_PATH = Path(os.environ.get("BRAIN_ENV_PATH") or PROJECT_ROOT / ".env"); load_dotenv(ENV_PATH)
from engine.common.constants import (AGENT_POSITION_RECORD_BYTES, COMMAND_SHM_BYTES,CONNECTION_COUNT_BYTES, CONNECTION_RECORD_SIZE,DASHBOARD_PORT, DASHBOARD_URL, EXTRACTION_THREADS, LOG_DIR, MAX_AGENTS,MAX_CONNECTIONS, PHASE_IDLE,PHASE_PHYSICS, PHASE_RESEARCH, PHASE_STABLE, PHASE_SYNTHESIS, PHASE_THINKING, REPORT_SHM_BYTES, SCRAPE_THREADS, STATUS_SHM_BYTES)
from engine.common.shm import remove_shm_from_resource_tracker, ensure_port_free, create_shm, write_report
if __name__ == "__main__": remove_shm_from_resource_tracker()
# Mirrors stdout/stderr into a startup-timestamped log while preserving terminal output.
class StartupLogTee:
    def __init__(self, log_path):
        self.log_path = Path(log_path); self.out_fd = self.err_fd = self.read_fd = None; self.log_file = None; self.thread = None; self.closed = False
    # Starts fd-level teeing so child processes and native binaries are logged too.
    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True); self.log_file = open(self.log_path, "ab", buffering=0)
        self.out_fd, self.err_fd = os.dup(1), os.dup(2); os.set_inheritable(self.out_fd, False); os.set_inheritable(self.err_fd, False)
        self.read_fd, write_fd = os.pipe(); os.set_inheritable(self.read_fd, False); os.dup2(write_fd, 1, inheritable=True); os.dup2(write_fd, 2, inheritable=True); os.close(write_fd)
        for stream in (sys.stdout, sys.stderr):
            try: stream.reconfigure(line_buffering=True, write_through=True)
            except Exception: pass
        self.thread = threading.Thread(target=self._pump, name="startup-log-tee", daemon=True); self.thread.start()
        return self.log_path
    # Copies process output bytes into the terminal and log file.
    def _pump(self):
        while True:
            try: chunk = os.read(self.read_fd, 8192)
            except OSError: break
            if not chunk: break
            try: os.write(self.out_fd, chunk)
            except OSError: pass
            try: self.log_file.write(chunk)
            except Exception: pass
    # Restores terminal fds and gives the tee thread a moment to flush.
    def close(self):
        if self.closed: return
        self.closed = True
        try: sys.stdout.flush(); sys.stderr.flush()
        except Exception: pass
        try: os.dup2(self.out_fd, 1); os.dup2(self.err_fd, 2)
        except OSError: pass
        if self.thread: self.thread.join(timeout=1.0)
        for fd in (self.out_fd, self.err_fd, self.read_fd):
            try:
                if fd is not None: os.close(fd)
            except OSError: pass
        try:
            if self.log_file: self.log_file.close()
        except Exception: pass
# Creates a timestamped log path for one engine startup.
def _startup_log_path():
    return LOG_DIR / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
# Starts the FastAPI dashboard with shared-memory names; returns its process handle.
def launch_webapp(names_dict): # Launches clean web UI (localhost:8000)
    ensure_port_free(DASHBOARD_PORT); print(f"[SYSTEM] Launching Dashboard UI on {DASHBOARD_URL}")
    env = os.environ.copy() # sets up env containing api key
    env["SHM_POS"]         = names_dict["pos"]; env["SHM_CONNECTIONS"] = names_dict["connections"]; env["SHM_STATUS"]      = names_dict["status"]; env["SHM_COMMAND"]     = names_dict["command"]; env["SHM_REPORT"]      = names_dict["report"]; cmd = [sys.executable, str(PROJECT_ROOT / "frontend" / "ui.py")]
    return subprocess.Popen(cmd, env=env, cwd=PROJECT_ROOT) # opens
# Starts the native physics engine against the active shared-memory segments.
def launch_physics(names_dict): # calls c++ to launch physics
    physics_binary = Path(os.environ.get("BRAIN_PHYSICS_ENGINE") or PROJECT_ROOT / "venv" / "physics-engine"); print(f"[SYSTEM] Launching C++ Physics Engine ({os.path.basename(physics_binary)})"); env = os.environ.copy(); env["SHM_POS"]         = names_dict["pos"]; env["SHM_CONNECTIONS"] = names_dict["connections"]; env["SHM_STATUS"]      = names_dict["status"]
    return subprocess.Popen([str(physics_binary)], env=env, cwd=PROJECT_ROOT) # opens
# Asks runtime pruning to keep the physics-resolved graph that can support thought routes.
def prune_after_physics(runtime_api):
    print("[SYSTEM] Pruning active connector subgraph after physics..."); subgraph_summary = runtime_api.prepare_active_subgraph()
    if not subgraph_summary: return
    print("[SYSTEM] Active connector subgraph: " f"{subgraph_summary['kept_connections']} connections, "
          f"{subgraph_summary['kept_nodes']} nodes, start anchors {subgraph_summary['active_seed_count']}/{subgraph_summary['seed_count']}, " f"goal anchors {subgraph_summary['active_goal_count']}/{subgraph_summary['goal_count']}.")
    for route in subgraph_summary.get("routes", []):
        print("[SYSTEM] Route " f"{route.get('route', '?')}: " f"{route.get('seed_count', 0)} starts, "f"{route.get('goal_count', 0)} goals, "f"{route.get('core_connections', 0)} core connections.")
    for item in subgraph_summary.get("anchor_pair_diagnostics", [])[:8]:
        status = "selected" if item.get("selected") else "rejected"; print("[SYSTEM] Anchor pair " f"{status}: A='{item.get('a', '')}' B='{item.get('b', '')}' grounding=({item.get('a_grounding', 0)}, {item.get('b_grounding', 0)})" f"A->B edges={item.get('a_to_b_edges', 0)} "f"B->A edges={item.get('b_to_a_edges', 0)} "f"sources={item.get('sources', 0)}.")
# Creates and zeros all shared-memory segments; returns names for child processes and live handles.
def manage_shm(): # pos_shm: agent positions, connection_shm: connection data, command_shm: prompt payload, status_shm: status, report_shm: reports
    anchors = {}
    # Creates one shared-memory segment and stores its handle so it stays alive.
    def _create_and_anchor(name, size): # creates and anchors SHM
        unique_name, shm = create_shm(name, size); anchors[name] = shm
        return unique_name, shm
    pos_name, pos_shm = _create_and_anchor("pos", MAX_AGENTS * AGENT_POSITION_RECORD_BYTES); report_name, report_shm = _create_and_anchor("report", REPORT_SHM_BYTES); command_name, command_shm = _create_and_anchor("command", COMMAND_SHM_BYTES); status_name, status_shm = _create_and_anchor("status", STATUS_SHM_BYTES)
    connection_name, connection_shm = _create_and_anchor("connections", CONNECTION_COUNT_BYTES + MAX_CONNECTIONS * CONNECTION_RECORD_SIZE)
    for shm in (pos_shm, report_shm, command_shm, status_shm, connection_shm):
        shm.buf[:] = b"\x00" * shm.size # zero out SHM segments to prevent garbage data
    names = {"pos": pos_name, "report": report_name, "command": command_name, "connections": connection_name, "status": status_name}
    return names, anchors # stores it in raw bytes
# Bootstraps workers, dashboard, physics, thinking, and synthesis phases for one engine session.
def main():
    os.environ["PYTHONUNBUFFERED"] = "1"; log_tee = StartupLogTee(_startup_log_path()); log_path = log_tee.start(); os.environ["BRAIN_LOG_PATH"] = str(log_path); print(f"[SYSTEM] Logging debug output to {log_path}")
    runtime_api, engine_thread, server_proc, physics_proc = None, None, None, None; scrapers, extractors = [], []
    shm_names, shm_handles = manage_shm() # initializes and returns shared memory
    status_shm, report_shm = shm_handles["status"],shm_handles["report"]
    scrape_queue, extract_queue, asu_queue = mp.Queue(), mp.Queue(), mp.Queue()  # distributes workers to separate processes
    stop_event, sync_counter, ingest_counter = mp.Event(), mp.Value('i', 0), mp.Value('i', 0)
    try:
        server_proc        = launch_webapp(shm_names) # launch web UI
        status_shm.buf[0]  = PHASE_IDLE # updates phase
        zero_ticks         = 0  # consecutive ticks where sync_counter == 0 and queue empty
        from engine.extract.scrape_worker import scrape_worker_loop; from engine.extract.worker import extraction_worker_loop; import engine.runtime as runtime_api
        # ------------ THREAD SETUP ------------
        for i in range(SCRAPE_THREADS): # launch 8 scrape workers
            p = mp.Process(target=scrape_worker_loop, # calls scraper_worker.py for each thread
                args=(scrape_queue, extract_queue, stop_event, sync_counter), name=f"ScrapeWorker_{i}", daemon=True)
            p.start(); scrapers.append(p)
        for i in range(EXTRACTION_THREADS):
            p = mp.Process(target=extraction_worker_loop, args=(extract_queue, asu_queue, stop_event), name=f"ExtractWorker_{i}", daemon=True); p.start(); extractors.append(p)
        engine_thread = threading.Thread(target=runtime_api.run_engine, args=(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter), name="di-engine", daemon=True); engine_thread.start(); print("[SYSTEM] Engine idle. Awaiting prompts.")
        # ------------ MAIN LOOP ------------
        while not stop_event.is_set(): # continues if program is still running
            time.sleep(1)
            current_status = status_shm.buf[0] # checks phase
            # The following blocks check for phase end conditions and start next phase.
            # Researching -> Physics
            if current_status == PHASE_RESEARCH: # Require 3 consecutive quiet ticks to guard against the sync_counter briefly hitting 0 between query batches.
                with sync_counter.get_lock(), ingest_counter.get_lock():
                    quiet = sync_counter.value <= 0 and ingest_counter.value <= 0
                if quiet: # checks if agents are done processing and query list is empty
                    zero_ticks += 1
                    if zero_ticks >= 3: # ensures that it stays in research mode for at least 3 ticks to prevent race conditions
                        if runtime_api.queue_refinement_scrapes(scrape_queue, sync_counter, ingest_counter):
                            zero_ticks = 0
                            continue
                        print("[SYSTEM] Research complete. Launching physics simulation..."); status_shm.buf[0] = PHASE_PHYSICS; physics_proc = launch_physics(shm_names)
                        if physics_proc is None:
                            print("[SYSTEM] Physics skipped. Pruning graph before marking stable."); prune_after_physics(runtime_api); status_shm.buf[0] = PHASE_STABLE
                        zero_ticks = 0
                else: zero_ticks = 0
            # Physics -> Watching for stability
            if physics_proc and physics_proc.poll() is not None: # checks if physics engine is done
                if physics_proc.returncode == 0:
                    print("[SYSTEM] Physics stabilized."); prune_after_physics(runtime_api)
                    status_shm.buf[0] = PHASE_STABLE # updates phase, will be read by synthesis.py
                elif physics_proc.returncode == 2:
                    print("[SYSTEM] Physics reached its time limit without stabilizing. Proceeding with the current layout."); prune_after_physics(runtime_api); status_shm.buf[0] = PHASE_STABLE
                else:
                    error = f"[error] Physics engine failed with exit code {physics_proc.returncode}."; print(f"[SYSTEM] {error}"); write_report(report_shm, error); status_shm.buf[0] = PHASE_IDLE
                physics_proc = None
            if server_proc and server_proc.poll() is not None:
                print("[SYSTEM] Web server terminated. Shutting down..."); stop_event.set()
            # Stable -> Thinking (Starting the autonomous analysis)
            if current_status == PHASE_STABLE:
                print("[SYSTEM] Graph stabilized. Starting Thought Processes..."); status_shm.buf[0] = PHASE_THINKING
                if runtime_api.launch_thought_workers(stop_event): print("[SYSTEM] Thought workers launched.")
                else: print("[SYSTEM] Decentralized Intelligence is not ready to launch thought workers yet.")
            # Thinking -> Synthesis (Moving from analysis to report writing)
            if current_status == PHASE_THINKING:
                if not engine_thread.is_alive(): raise RuntimeError("Decentralized Intelligence thread terminated during thought processing.")
                if runtime_api.thought_workers_finished():
                    stats = runtime_api.thought_worker_stats(); print(f"[SYSTEM] Thought workers finished: {stats['endpoint']} at endpoints / {stats['dead']} dead / "
                        f"{stats.get('max_hops', 0)} max hops / {stats['successful']} successful.")
                    status_shm.buf[0] = PHASE_SYNTHESIS
    except KeyboardInterrupt: # stops everything at keyboard interrupt (ctrl+c)
        print("[SYSTEM] Interrupt received."); stop_event.set()
    except Exception as e: # stops everything at any error
        print(f"[SYSTEM] Critical error: {e}"); stop_event.set()
    finally: # stops everything at end of program, cleans up technical stuff
        print("[SYSTEM] Cleaning up..."); stop_event.set()
        if runtime_api: runtime_api.stop_thought_workers()
        for p in scrapers: p.terminate()
        for p in extractors: p.terminate()
        for p in scrapers + extractors:
            try: p.join(timeout=1.0)
            except Exception: pass
        for proc in (server_proc, physics_proc):
            if not proc: continue
            proc.terminate()
            try: proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try: proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired: pass
        if engine_thread and engine_thread.is_alive(): engine_thread.join(timeout=2.0)
        for shm in shm_handles.values(): # closes and removes shared memory segments
            try: shm.close(); shm.unlink()
            except (BufferError, FileNotFoundError, OSError): pass # ensures no crash on cleanup
        print("[SYSTEM] Engine offline.")
        log_tee.close()
if __name__ == "__main__": main() # entry point for program, just runs main()
