import json, os, struct, asyncio, time, hashlib, re, sys, uvicorn # type: ignore
from multiprocessing import shared_memory; from pathlib import Path
from pydantic import BaseModel  # type: ignore
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import HTMLResponse # type: ignore
UI_ROOT = Path(__file__).resolve().parent; PROJECT_ROOT = UI_ROOT.parent; FRONTEND_ROOT = UI_ROOT
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from u_constants import (AGENT_POSITION_RECORD_BYTES, CONNECTION_RECORD_SIZE, COMMAND_SHM_BYTES, DASHBOARD_HOST, DASHBOARD_PORT, MAX_AGENTS, MAX_UI_AGENTS, MAX_UI_BONDS, MAX_UI_CONNECTION_STATS, RESULTS_DIR, WEBSOCKET_REFRESH_SECONDS, PHASE_IDLE, PHASE_RESEARCH, PHASE_PHYSICS, PHASE_STABLE, PHASE_THINKING, PHASE_SYNTHESIS)
from utils import remove_shm_from_resource_tracker
app = FastAPI() # setup
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]); FRONTEND_ASSET_VERSION = str(int(time.time())); NO_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache", "Expires": "0"}; remove_shm_from_resource_tracker()
# Maps the first available dashboard shared-memory name; returns a SharedMemory handle or None.
def map_shm(*env_vars):
    name = next((os.environ.get(env_var) for env_var in env_vars if os.environ.get(env_var)), None)
    if not name: return None
    try: return shared_memory.SharedMemory(name=name)
    except Exception as e:
        joined = ", ".join(env_vars); print(f"[WEB] Failed to map {joined} ({name}): {e}")
        return None
# Map segments from environment variables set by main.py
shm_connections = map_shm("SHM_CONNECTIONS"); shm_cmd = map_shm("SHM_COMMAND")
shm_pos = map_shm("SHM_POS"); shm_report = map_shm("SHM_REPORT"); shm_status = map_shm("SHM_STATUS")
# Returns whether an agent position slot contains live coordinates.
def _agent_is_initialized(x, y, z):
    return not (x == 0 and y == 0 and z == 0)
# Counts initialized shared-memory agent positions for dashboard capacity display.
def _count_initialized_agents(limit=MAX_AGENTS):
    if not shm_pos: return 0
    count = 0; scan_limit = min(int(limit), MAX_AGENTS)
    for i in range(scan_limit):
        off = i * AGENT_POSITION_RECORD_BYTES
        try: x, y, z = struct.unpack_from("fff", shm_pos.buf, off)
        except struct.error: break
        if _agent_is_initialized(x, y, z): count += 1
    return count
# Returns the folder where final synthesis result directories are stored.
def _results_root():
    return PROJECT_ROOT / RESULTS_DIR
# Reads one saved result file and returns stripped text or an empty string.
def _read_result_text(result_dir, filename):
    path = result_dir / filename
    if not path.is_file(): return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()
# Extracts a saved target label from the argument report text.
def _target_from_arguments(arguments_text, label):
    patterns = (rf"^{label}\s+is\s+(.+?)\.\s*$", rf"^{label}:\s*(.+?)\s*$",)
    for pattern in patterns:
        match = re.search(pattern, arguments_text, flags=re.IGNORECASE | re.MULTILINE)
        if not match: continue
        value = " ".join(match.group(1).strip().split())
        if value and value.lower() != "(none)": return value
    return ""
# Builds sidebar metadata for one saved synthesis result directory.
def _result_summary(result_dir):
    arguments_text = _read_result_text(result_dir, "arguments.txt"); target_a = _target_from_arguments(arguments_text, "Target A"); target_b = _target_from_arguments(arguments_text, "Target B"); title = f"{target_a} and {target_b}" if target_a and target_b else result_dir.name
    return {"id": result_dir.name, "title": title, "target_a": target_a, "target_b": target_b, "created": result_dir.name}
# Validates a result id and returns its resolved directory or raises 404.
def _safe_result_dir(result_id):
    clean_id = str(result_id or "").strip()
    if not clean_id or clean_id in {".", ".."} or "/" in clean_id or "\\" in clean_id:
        raise HTTPException(status_code=404, detail="Result not found")
    root = _results_root().resolve(); result_dir = (root / clean_id).resolve()
    if root not in result_dir.parents or not result_dir.is_dir():
        raise HTTPException(status_code=404, detail="Result not found")
    return result_dir
# Serves dashboard HTML with cache-busted CSS and JS references.
@app.get("/")
async def read_index():
    with open(FRONTEND_ROOT / "index.html", "r", encoding="utf-8") as f: html = f.read()
    html = re.sub(r'href="style\.css(?:\?v=[^"]*)?"', f'href="style.css?v={FRONTEND_ASSET_VERSION}"', html); html = re.sub(r'src="main\.js(?:\?v=[^"]*)?"', f'src="main.js?v={FRONTEND_ASSET_VERSION}"', html)
    return HTMLResponse(content=html, headers=NO_CACHE_HEADERS)
# Lists saved synthesis results for the history sidebar.
@app.get("/results")
async def list_results():
    root = _results_root()
    if not root.is_dir(): return {"results": []}
    result_dirs = [path for path in root.iterdir() if path.is_dir() and (path / "synthesis.txt").is_file()]; result_dirs.sort(key=lambda path: path.name, reverse=True)
    return {"results": [_result_summary(path) for path in result_dirs]}
# Returns one saved synthesis result payload for the dashboard.
@app.get("/results/{result_id}")
async def read_result(result_id: str):
    result_dir = _safe_result_dir(result_id); summary = _result_summary(result_dir)
    return {**summary, "synthesis": _read_result_text(result_dir, "synthesis.txt")}
# Adds no-cache headers to dashboard HTML, JS, and CSS responses.
@app.middleware("http")
async def disable_cache_for_dashboard_assets(request, call_next):
    response = await call_next(request); path = request.url.path
    if path == "/" or path.endswith(".js") or path.endswith(".css") or path.endswith(".html"):
        for key, value in NO_CACHE_HEADERS.items(): response.headers[key] = value
    return response
# Streams live shared-memory graph, phase, and report state to the dashboard.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept(); print("[WEB] WebSocket client connected."); cached_agent_count, last_agent_count_check = 0, 0.0
    try:
        while True:
            now = time.time()
            if now - last_agent_count_check >= 1.0:
                cached_agent_count, last_agent_count_check = _count_initialized_agents(), now
            state = {"agents": [], "connections": [], "status": "System Online", "phase": PHASE_IDLE, "report": "", "report_version": 0, "agent_count": cached_agent_count, "agent_capacity": MAX_AGENTS, "agent_cap_reached": cached_agent_count >= MAX_AGENTS}; total_conn_counts = {}
            if shm_connections: # Read connections first so we can compute per-agent connection counts.
                try:
                    count = struct.unpack_from("i", shm_connections.buf, 0)[0] # get number of connections
                    stats_count = min(count, MAX_UI_CONNECTION_STATS)
                    for i in range(stats_count):
                        off = 4 + (i * CONNECTION_RECORD_SIZE)
                        s, f, d, utility = struct.unpack_from("<iiif", shm_connections.buf, off) # extract all connections one by one
                        total_conn_counts[s], total_conn_counts[d] = total_conn_counts.get(s, 0) + 1, total_conn_counts.get(d, 0) + 1
                        if len(state["connections"]) < MAX_UI_BONDS: state["connections"].append({"s": s, "d": d, "f": f, "utility": utility}) # store connection
                except (BufferError, struct.error, ValueError): pass
            if shm_pos:
                for i in range(MAX_UI_AGENTS): # Read Agents (capped for web performance)
                    off = i * AGENT_POSITION_RECORD_BYTES
                    try:
                        x, y, z = struct.unpack_from("fff", shm_pos.buf, off)
                        if not _agent_is_initialized(x, y, z): continue # Skips rendering agents that haven't been initialized
                        degree = total_conn_counts.get(i, 0); agent = {"id": i, "x": x, "y": y, "z": z, "degree": degree}; state["agents"].append(agent)
                    except (BufferError, struct.error, ValueError): break
            # Reads and displays status messages
            if shm_status:
                phase = shm_status.buf[0]; messages = {PHASE_IDLE: "System Idle", PHASE_RESEARCH: "Researching", PHASE_PHYSICS: "Simulating physics", PHASE_STABLE: "Graph stabilized",
                PHASE_THINKING: "Running thought processes", PHASE_SYNTHESIS: "Generating report",} # Mapping numbers to messages
                state["phase"], state["status"] = phase, messages.get(phase, f"Phase {phase}") # reads and displays status
            if shm_report: # Read report if it exists
                try:
                    report_text = bytes(shm_report.buf).split(b'\x00')[0].decode('utf-8').strip() # gets report text from shared memory
                    if report_text: state["report"], state["report_version"] = report_text, hashlib.md5(report_text.encode("utf-8")).hexdigest()
                except (BufferError, UnicodeDecodeError): pass
            await websocket.send_json(state)
            await asyncio.sleep(WEBSOCKET_REFRESH_SECONDS) # server/browser communication
    except WebSocketDisconnect: print("[WEB] WebSocket client disconnected.")
    except Exception as e: print(f"[WEB] WebSocket Error: {e}")
# Accepts a dashboard prompt and writes the command payload into shared memory.
@app.post("/command")
async def post_command(cmd: dict):
    if not shm_cmd: raise HTTPException(status_code=500, detail="Command SHM not mapped")
    target_a, target_b = cmd.get("target_a", "").strip(), cmd.get("target_b", "").strip()
    if not target_a or not target_b: raise HTTPException(status_code=400, detail="Both targets are required")
    if shm_status: # checks if engine is busy
        phase = shm_status.buf[0]
        if phase != PHASE_IDLE: raise HTTPException(status_code=409, detail="Engine busy")
    payload = {"id": str(time.time_ns()),"target_a": target_a,"target_b": target_b}; payload_text = json.dumps(payload, separators=(",", ":")); cmd_bytes = payload_text.encode("utf-8")
    if len(cmd_bytes) > COMMAND_SHM_BYTES - 1: raise HTTPException(status_code=413, detail="Command too large for shared memory buffer")
    shm_cmd.buf[:] = b"\x00" * len(shm_cmd.buf); shm_cmd.buf[:len(cmd_bytes)] = cmd_bytes; shm_cmd.buf[len(cmd_bytes):len(cmd_bytes)+1] = b"\x00"; print(f"[WEB] Dispatched Command: {payload_text}")
    return {"status": "dispatched","command_id": payload["id"], "target_a": target_a,"target_b": target_b}
app.mount("/", StaticFiles(directory=str(FRONTEND_ROOT)), name="frontend") # Mount static files so style.css and main.js are accessible
if __name__ == "__main__":  uvicorn.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT) # entry point, runs the server
