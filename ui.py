import json
import os
import struct
import asyncio
import time
import hashlib
import re
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import FileResponse, HTMLResponse # type: ignore
from pydantic import BaseModel # type: ignore
import uvicorn # type: ignore
from multiprocessing import resource_tracker
from constants import PHASE_IDLE, PHASE_RESEARCH, PHASE_EXPORT, PHASE_EXPORTED, PHASE_IMPORT, PHASE_IMPORTED, PHASE_PHYSICS, PHASE_STABLE, LOGICAL_CONNECTORS

LOGICAL_CONNECTOR_COUNT = len(LOGICAL_CONNECTORS)

app = FastAPI() # setup
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],)
FRONTEND_ASSET_VERSION = str(int(time.time()))

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

def remove_shm_from_resource_tracker():
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.register = fix_register
    resource_tracker.unregister = fix_unregister

remove_shm_from_resource_tracker()

def map_shm(env_var): # Shared Memory Mapping
    name = os.environ.get(env_var)
    if not name: return None
    try:
        from multiprocessing import shared_memory
        return shared_memory.SharedMemory(name=name)
    except Exception as e:
        print(f"[WEB] Failed to map {env_var} ({name}): {e}")
        return None

# Map segments from environment variables set by main.py
shm_pos = map_shm("SHM_SKS_POS")
shm_connections = map_shm("SHM_SKS_CONNECTIONS")
shm_status = map_shm("SHM_SKS_STATUS")
shm_report = map_shm("SHM_SKS_REPORT")
shm_cmd = map_shm("SHM_SKS_COMMAND")

@app.get("/")
async def read_index(): # opens index.html when you go to localhost:8000
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = re.sub(r'href="style\.css(?:\?v=[^"]*)?"', f'href="style.css?v={FRONTEND_ASSET_VERSION}"', html)
    html = re.sub(r'src="main\.js(?:\?v=[^"]*)?"', f'src="main.js?v={FRONTEND_ASSET_VERSION}"', html)
    return HTMLResponse(content=html, headers=NO_CACHE_HEADERS)


@app.middleware("http")
async def disable_cache_for_dashboard_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith(".js") or path.endswith(".css") or path.endswith(".html"):
        for key, value in NO_CACHE_HEADERS.items():
            response.headers[key] = value
    return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WEB] WebSocket client connected.")
    try:
        while True:
            state = {"agents": [], "connections": [], "status": "System Online", "phase": PHASE_IDLE, "report": "", "report_version": 0}
            # Read connections first so we can compute per-agent connection counts
            conn_counts = {}
            logical_agent_ids = set()
            if shm_connections:
                try:
                    count = struct.unpack_from("i", shm_connections.buf, 0)[0] # get number of connections
                    count = min(count, 2400)
                    for i in range(count):
                        off = 4 + (i * 16)
                        s, f, flags, d = struct.unpack_from("iiii", shm_connections.buf, off) # extract all connections one by one
                        if f >= LOGICAL_CONNECTOR_COUNT:
                            continue
                        logical_agent_ids.add(s)
                        logical_agent_ids.add(d)
                        conn_counts[s] = conn_counts.get(s, 0) + 1 # counts how many connections per agent
                        conn_counts[d] = conn_counts.get(d, 0) + 1 # 
                        state["connections"].append({"s": s, "d": d, "f": f, "truth": (flags & 1), "flags": flags}) # store connection
                except: pass

            max_conns = max(conn_counts.values(), default=1) # gets max connections (only used for visuals)
            nonlogical_agents = []
            if shm_pos:
                for i in range(1600): # Read Agents (capped for web performance)
                    off = i * 24
                    try:
                        x, y, z = struct.unpack_from("fff", shm_pos.buf, off)
                        if x == 0 and y == 0 and z == 0: continue # Skips rendering agents that haven't been initialized
                        c = conn_counts.get(i, 0) / max_conns # Visualizes agent connectivity 
                        agent = {"id": i, "x": x, "y": y, "z": z, "c": c, "logical": i in logical_agent_ids}
                        if agent["logical"]:
                            state["agents"].append(agent)
                        else:
                            nonlogical_agents.append(agent)
                    except: break # Caps number of agents rendered to help the renderer, doesn't affect logic
            state["agents"].extend(nonlogical_agents)

            # Reads and displays status messages
            if shm_status:
                phase = shm_status.buf[0]
                messages = { # Map numbers to messages
                    PHASE_IDLE: "System Idle",
                    PHASE_RESEARCH: "Researching",
                    PHASE_EXPORT: "Exporting agents",
                    PHASE_EXPORTED: "Grouping agents",
                    PHASE_IMPORT: "Importing optimized graph",
                    PHASE_IMPORTED: "Preparing physics",
                    PHASE_PHYSICS: "Simulating physics",
                    PHASE_STABLE: "Stabilized. Generating report",
                }
                state["phase"] = phase
                state["status"] = messages.get(phase, f"Phase {phase}")

            if shm_report: # Read report if it exists
                try:
                    report_text = bytes(shm_report.buf).split(b'\x00')[0].decode('utf-8').strip()
                    if report_text:
                        state["report"] = report_text
                        state["report_version"] = hashlib.md5(report_text.encode("utf-8")).hexdigest()
                except: pass

            await websocket.send_json(state)
            await asyncio.sleep(0.1) # server/browser communication (10Hz)
    except WebSocketDisconnect:
        print("[WEB] WebSocket client disconnected.")
    except Exception as e:
        print(f"[WEB] WebSocket Error: {e}")

class Command(BaseModel): # verifies strict command data format
    type: str
    goal: str = ""
    target: str = ""
    local_only: bool = False

@app.post("/command") # post endpoint to dispatch user prompts
async def post_command(cmd: Command):
    if not shm_cmd: raise HTTPException(status_code=500, detail="Command SHM not mapped")

    target = cmd.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="Target is required")

    if shm_status:
        phase = shm_status.buf[0]
        if phase != PHASE_IDLE:
            raise HTTPException(status_code=409, detail="Engine busy")

    payload = {
        "id": str(time.time_ns()),
        "goal": cmd.goal.strip(),
        "target": target,
        "local_only": bool(cmd.local_only),
    }
    payload_text = json.dumps(payload, separators=(",", ":"))
    cmd_bytes = payload_text.encode("utf-8")
    if len(cmd_bytes) > 2047:
        raise HTTPException(status_code=413, detail="Command too large for shared memory buffer")

    shm_cmd.buf[:] = b"\x00" * len(shm_cmd.buf)
    shm_cmd.buf[:len(cmd_bytes)] = cmd_bytes
    shm_cmd.buf[len(cmd_bytes):len(cmd_bytes)+1] = b"\x00"

    print(f"[WEB] Dispatched Command: {payload_text}")
    return {"status": "dispatched", "command_id": payload["id"], "target": target}

app.mount("/", StaticFiles(directory="frontend"), name="frontend") # Mount static files so style.css and main.js are accessible

if __name__ == "__main__": # entry point, runs the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
