import json
from multiprocessing import resource_tracker

def normalize_relation_label(label: str) -> str:
    return " ".join(label.lower().split()).strip()

def parse_command_payload(cmd_raw):
    try:
        payload = json.loads(cmd_raw)
        if isinstance(payload, dict):
            cmd_id = str(payload.get("id", "")).strip() or cmd_raw
            goal = str(payload.get("goal", ""))
            target = str(payload.get("target", ""))
            local_only = bool(payload.get("local_only", False))
            return cmd_id, goal, target, local_only
    except json.JSONDecodeError:
        pass

    if "|" in cmd_raw:
        goal, target = cmd_raw.split("|", 1)
    else:
        goal, target = "", cmd_raw
    return cmd_raw, goal, target, False

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

def flush_conn_log(self):
    if not self._conn_log: return
    with open("connection_report.txt", "a") as f: f.writelines(self._conn_log)
    self._conn_log.clear()

def clear_report(self):
    self.shm_report.buf[:] = b"\x00" * len(self.shm_report.buf)
