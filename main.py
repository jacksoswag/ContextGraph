
import multiprocessing as mp
import os
import socket
import sys
import time
import subprocess
from multiprocessing import shared_memory
from dotenv import load_dotenv # type: ignore
from constants import PHASE_IDLE, PHASE_RESEARCH, PHASE_EXPORT, PHASE_EXPORTED, PHASE_IMPORT, PHASE_IMPORTED, PHASE_PHYSICS, PHASE_STABLE

load_dotenv()
from d_scrape_worker import scrape_worker_loop

# Silence ResourceTracker warnings for manually managed SHM segments
from multiprocessing import resource_tracker
def remove_shm_from_resource_tracker():
    def fix_register(name, rtype):
        if rtype == "shared_memory": return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register
    def fix_unregister(name, rtype):
        if rtype == "shared_memory": return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister
if __name__ == "main" or __name__ == "__main__":
    remove_shm_from_resource_tracker()

_SHM_ANCHORS = []


def write_report(shm, message):
    payload = message.encode("utf-8")[:(shm.size - 1)]
    shm.buf[:] = b"\x00" * shm.size
    shm.buf[:len(payload)] = payload
    shm.buf[len(payload):len(payload)+1] = b"\x00"


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def find_physics_binary():
    for candidate in ("./physics_engine", "./SKS_Renderer"):
        if os.path.exists(candidate):
            return candidate
    return None

def launch_webapp(names_dict): # Launches clean web UI (localhost:8000)
    if is_port_in_use(8000):
        raise RuntimeError("Port 8000 is already in use. Refusing to kill an unrelated process.")
    print("[SYSTEM] Launching Dashboard UI on http://localhost:8000")
    env = os.environ.copy() # sets up env containing api key
    env["SHM_SKS_POS"]         = names_dict["sks_pos"]
    env["SHM_SKS_CONNECTIONS"]  = names_dict["sks_connections"]
    env["SHM_SKS_STATUS"]       = names_dict["sks_status"]
    env["SHM_SKS_COMMAND"]      = names_dict["sks_command"]
    env["SHM_SKS_REPORT"]       = names_dict["sks_report"]
    cmd = [sys.executable, "ui.py"] # checks for ui.py in cwd or frontend/ui.py
    if not os.path.exists("ui.py") and os.path.exists("frontend/ui.py"):
        cmd = [sys.executable, "frontend/ui.py"]
    return subprocess.Popen(cmd, env=env) # opens

def launch_physics(names_dict): # calls c++ to launch physics
    physics_binary = find_physics_binary()
    if not physics_binary:
        print("[SYSTEM] WARNING: physics binary not found. Skipping physics.")
        return None
    print(f"[SYSTEM] Launching C++ Physics Engine ({os.path.basename(physics_binary)})")
    env = os.environ.copy()
    env["SHM_SKS_POS"]         = names_dict["sks_pos"]
    env["SHM_SKS_CONNECTIONS"]  = names_dict["sks_connections"]
    env["SHM_SKS_STATUS"]       = names_dict["sks_status"]
    return subprocess.Popen([physics_binary], env=env) # opens

def main():
    scrape_queue  = mp.Queue() # distributes scrape workers to separate threads
    asu_queue     = mp.Queue()
    stop_event    = mp.Event()
    manager       = mp.Manager()
    shared_cache  = manager.dict()
    sync_counter  = mp.Value('i', 0)
    ingest_counter = mp.Value('i', 0)

    def _create_shm(name, size): # creates shared memory segments that the UI and Physics Engine can access
        unique_name = f"{name}_{int(time.time()) % 10000}"
        shm = shared_memory.SharedMemory(name=unique_name, create=True, size=size)
        _SHM_ANCHORS.append(shm)
        print(f"[SYSTEM] Created SHM: {unique_name} ({size} bytes)")
        return unique_name, shm

    try:
        # pos_shm handles agent positions, connection_shm handles connection data, command_shm handles commands, status_shm handles status, and report_shm handles research reports
        pos_name,     pos_shm     = _create_shm("sks_pos",    60000 * 6 * 4) # agent positions (capped at 60000 for now)
        report_name,  report_shm  = _create_shm("sks_report", 128 * 1024) # research reports (arguments, evidence, synthesis)
        command_name, command_shm = _create_shm("sks_command", 2 * 1024) # research commands (goal/target pairs)
        status_name,  status_shm  = _create_shm("sks_status",  1 * 1024) # phase number
        connection_name, connection_shm = _create_shm("sks_connections", 4 + 80000 * 16) # connection data (capped at 80000 for now)

        for shm in (pos_shm, report_shm, command_shm, status_shm, connection_shm):
            shm.buf[:] = b"\x00" * shm.size # zero out SHM segments to prevent garbage data

        shm_names = { # stores it in raw bytes
            "sks_pos":     pos_name,
            "sks_report":  report_name,
            "sks_command": command_name,
            "sks_connections": connection_name,
            "sks_status":  status_name,
        }

        server_proc        = launch_webapp(shm_names) # launch web UI
        physics_proc       = None
        status_shm.buf[0]  = PHASE_IDLE # updates phase
        zero_ticks         = 0  # consecutive ticks where sync_counter == 0 and queue empty

        scrapers = []
        for i in range(8): # launch 8 scrape workers
            p = mp.Process(
                target=scrape_worker_loop, # calls scraper_worker.py for each thread
                args=(scrape_queue, asu_queue, stop_event, shared_cache, sync_counter),
                name=f"SKS_Worker_{i}",
                daemon=True
            )
            p.start()
            scrapers.append(p)

        from m_brain import run_brain # calls brain.py
        brain_proc = mp.Process(
            target=run_brain, # agents are spawned in, but physics WAITS until all queries finish
            args=(scrape_queue, asu_queue, stop_event, shm_names, sync_counter, ingest_counter),
            name="brain",
            daemon=True
        )
        brain_proc.start() # starts brain

        print("[SYSTEM] Engine idle. Awaiting prompts.")

        while not stop_event.is_set(): # continues if program is still running
            time.sleep(1)
            current_status = status_shm.buf[0] # checks phase

            # PHASE_RESEARCH → PHASE_EXPORT
            if current_status == PHASE_RESEARCH: # Require 3 consecutive quiet ticks to guard against the sync_counter briefly hitting 0 between query batches.
                with sync_counter.get_lock(), ingest_counter.get_lock():
                    quiet = sync_counter.value <= 0 and ingest_counter.value <= 0
                if quiet: # checks if query list is empty and all scrapers have finished
                    zero_ticks += 1
                    if zero_ticks >= 3: # ensures that it stays in research mode for at least 3 ticks to prevent race conditions
                        print("[SYSTEM] Research complete. Initiating semantic grouping...")
                        status_shm.buf[0] = PHASE_EXPORT
                        zero_ticks = 0
                else: zero_ticks = 0

            # PHASE_EXPORTED → run grouping script → PHASE_IMPORT
            # At this point, all connections between ASUs have been processed and exported
            elif current_status == PHASE_EXPORTED:
                print("[SYSTEM] Agents exported. Running p_info_grouping.py...")
                env = os.environ.copy()
                env["SHM_SKS_CONNECTIONS"] = connection_name
                result = subprocess.run([sys.executable, "p_info_grouping.py"], env=env) # runs information grouping script
                if result.returncode != 0:
                    error = f"[error] Grouping failed with exit code {result.returncode}."
                    print(f"[SYSTEM] {error}")
                    write_report(report_shm, error)
                    status_shm.buf[0] = PHASE_IDLE
                else:
                    status_shm.buf[0] = PHASE_IMPORT # transitions to Phase Import (end of p_info_grouping.py)

            # PHASE_IMPORTED → launch physics
            elif current_status == PHASE_IMPORTED:
                print("[SYSTEM] Grouping complete. Launching physics simulation...")
                status_shm.buf[0] = PHASE_PHYSICS
                physics_proc = launch_physics(shm_names) # launches C++ Physics Engine
                if physics_proc is None:
                    print("[SYSTEM] Physics skipped. Marking graph as stable.")
                    status_shm.buf[0] = PHASE_STABLE

            # PHASE_PHYSICS → watch for physics exit
            if physics_proc and physics_proc.poll() is not None:
                if physics_proc.returncode == 0:
                    print("[SYSTEM] Physics stabilized.")
                    status_shm.buf[0] = PHASE_STABLE # updates phase, will be read by synthesis.py
                else:
                    error = f"[error] Physics engine failed with exit code {physics_proc.returncode}."
                    print(f"[SYSTEM] {error}")
                    write_report(report_shm, error)
                    status_shm.buf[0] = PHASE_IDLE
                physics_proc = None

            if server_proc and server_proc.poll() is not None:
                print("[SYSTEM] Web server terminated. Shutting down...")
                stop_event.set()

    except KeyboardInterrupt: # stops everything at keyboard interrupt (ctrl+c)
        print("[SYSTEM] Interrupt received.")
        stop_event.set()
    except Exception as e: # stops everything at any error
        print(f"[SYSTEM] Critical error: {e}")
        stop_event.set()
    finally: # stops everything at end of program, cleans up technical stuff
        print("[SYSTEM] Cleaning up...")
        if 'brain_proc' in locals(): brain_proc.terminate()
        if 'scrapers'   in locals():
            for p in scrapers: p.terminate()
        if 'server_proc'  in locals() and server_proc:  server_proc.terminate()
        if 'physics_proc' in locals() and physics_proc: physics_proc.terminate()
        for shm in _SHM_ANCHORS: # closes and removes shared memory segments
            try:
                shm.close()
                shm.unlink()
            except: pass # ignores errors during cleanup
        print("[SYSTEM] Engine offline.")

if __name__ == "__main__": # entry point for program, just runs main()
    main()
