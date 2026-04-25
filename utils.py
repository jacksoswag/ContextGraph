import json, os, signal, socket, subprocess, time
from multiprocessing import resource_tracker, shared_memory
from constants import DASHBOARD_PORT

# Silence ResourceTracker warnings for manually managed SHM segments
def remove_shm_from_resource_tracker():
    def fix_register(name, rtype):
        if rtype == "shared_memory": return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register
    def fix_unregister(name, rtype):
        if rtype == "shared_memory": return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

# Parses a raw command string into a structured research request
def parse_command_payload(cmd_raw):
    try:
        payload = json.loads(cmd_raw)
        if isinstance(payload, dict):
            cmd_id = str(payload.get("id", "")).strip() or cmd_raw
            target_a = str(payload.get("target_a", "")).strip()
            target_b = str(payload.get("target_b", "")).strip()
            tense_preference = normalize_tense_preference(payload.get("tense_preference", "none"))
            return cmd_id, target_a, target_b, tense_preference
    except json.JSONDecodeError: pass
    if "|" in cmd_raw: target_a, target_b = cmd_raw.split("|", 1)
    else: target_a, target_b = cmd_raw, ""
    return " ".join(cmd_raw.strip().split()), target_a.strip(), target_b.strip(), "none"

def normalize_tense_preference(value):
    clean = str(value or "").strip().lower()
    return clean if clean in {"past", "future"} else "none"

def flush_conn_log(self): # Flushes the connection log to a file
    if not self._conn_log: return
    with open("connection_report.txt", "a") as f: f.writelines(self._conn_log)
    self._conn_log.clear()

def clear_report(self): # Clears the connection report
    self.shm_report.buf[:] = b"\x00" * len(self.shm_report.buf)

def write_report(shm, message):
    payload = message.encode("utf-8")[:(shm.size - 1)]
    shm.buf[:] = b"\x00" * shm.size
    shm.buf[:len(payload)] = payload
    shm.buf[len(payload):len(payload)+1] = b"\x00"    

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0

def _port_listener_pid(port):
    try: output = subprocess.check_output(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"], text=True, stderr=subprocess.DEVNULL)
    except Exception: return None
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) < 2: return None
    parts = lines[1].split()
    if len(parts) < 2: return None
    try: return int(parts[1])
    except (TypeError, ValueError): return None

def ensure_port_free(port=DASHBOARD_PORT):
    if is_port_in_use(port):
        if not _stop_stale_dashboard(port): raise RuntimeError(f"Port {port} is already in use by a non-dashboard process. Refusing to kill an unrelated process.")
    return True

def _stop_stale_dashboard(port): # Stops the UI if its still running
    pid = _port_listener_pid(port)
    if pid is None: return False

    print(f"[SYSTEM] Stopping stale dashboard on port {port} (pid={pid})")
    try: os.kill(pid, signal.SIGTERM)
    except OSError: pass

    for _ in range(20):
        if not is_port_in_use(port): return True
        time.sleep(0.1)

    try: os.kill(pid, signal.SIGKILL)
    except OSError: pass

    for _ in range(20):
        if not is_port_in_use(port): return True
        time.sleep(0.1)
    return not is_port_in_use(port)

def create_shm(name, size): # create shared memory segment with unique timestamped name
    unique_name = f"{name}_{int(time.time()) % 10000}"
    shm = shared_memory.SharedMemory(name=unique_name, create=True, size=size)
    print(f"[SYSTEM] Created SHM: {unique_name} ({size} bytes)")
    return unique_name, shm

def _clean_text(text):
    return " ".join(str(text or "").strip().split())

def display_source(source):
    clean = _clean_text(source)
    if "|" not in clean:
        return clean
    display = clean.rsplit("|", 1)[-1].strip()
    return display or clean

def _specific_text(value):
    if isinstance(value, dict):
        for key in ("context", "surface", "text", "normalized", "cue", "kind", "scope"):
            text = _clean_text(value.get(key, ""))
            if text: return text
        return ""
    return _clean_text(value)

def _specific_key(value): # get specific key
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return _clean_text(value).lower()

def merge_specifics(existing, incoming): # merge specifics
    merged = []
    seen = set()
    for bucket in (existing or [], incoming or []):
        key = _specific_key(bucket)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(bucket)
    return merged

def format_specifics(values): # format specifics for display
    parts = []
    seen = set()
    for value in list(values or []):
        text = _specific_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return "; ".join(parts)

def apply_quantifier(text, quantifier_id): # apply quantifier to text
    from d_word_info_map import literal_from_index

    phrase = _clean_text(text)
    quantifier = _clean_text(literal_from_index(int(quantifier_id))) if quantifier_id not in (None, "", -1) else ""
    if not phrase or not quantifier:
        return phrase
    lower_phrase = phrase.lower()
    lower_quantifier = quantifier.lower()
    if lower_phrase == lower_quantifier or lower_phrase.startswith(f"{lower_quantifier} "):
        return phrase
    return f"{quantifier} {phrase}".strip()
