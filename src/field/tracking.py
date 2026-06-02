from __future__ import annotations
import time, json, sqlite3, hashlib, subprocess, os
from pathlib import Path
import numpy as np

# SQLite schema — one row per run, queryable by config/sha/label.
_DB_SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, timestamp TEXT, git_sha TEXT,
    config_hash TEXT, seed INTEGER, label TEXT,
    steps_to_stabilize INTEGER, runtime_s REAL, run_dir TEXT)"""

def _git_sha() -> str:
    # retrieve HEAD SHA; return "unknown" if outside a git repo
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "unknown"

def config_hash(cfg: dict) -> str:
    """Return 16-char hex SHA256 of canonicalized config dict."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]

def _ensure_db(db_path: Path) -> None:
    # create index.db and runs table if absent
    con = sqlite3.connect(db_path); con.execute(_DB_SCHEMA); con.commit(); con.close()


class RunLogger:
    """Writes per-step JSONL, summary.json, trajectory.npz, and sqlite index for one run."""

    def __init__(self, cfg: dict, seed: int, runs_root: str | Path = "runs") -> None:
        self._cfg, self._seed = cfg, seed
        self._chash = config_hash(cfg)
        self._sha = _git_sha()
        ts = time.strftime("%Y%m%dT%H%M%S")
        suffix = os.urandom(3).hex()  # 6-char random hex — ensures uniqueness within same ts+config
        self.run_id = f"{ts}_{self._chash}_{suffix}"
        self.run_dir = Path(runs_root) / self.run_id
        self._db_path = Path(runs_root) / "index.db"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        _ensure_db(self._db_path)
        self._fh = open(self.run_dir / "run.jsonl", "w")
        # header line — metadata for the whole run
        self._fh.write(json.dumps({
            "type": "header", "git_sha": self._sha,
            "config_hash": self._chash, "seed": seed, "config": cfg
        }) + "\n")
        self._step_count = 0
        self._t0 = time.monotonic()

    def log_step(self, step: int, E: float, delta_X: float, support_size: int, mesh_count: int, max_node_norm: float, trapping_violation: bool, nan_flag: bool) -> None:
        """Append one integration-step record to run.jsonl."""
        self._fh.write(json.dumps({
            "type": "step", "step": step, "E": float(E), "delta_X": float(delta_X),
            "support_size": int(support_size), "mesh_count": int(mesh_count),
            "max_node_norm": float(max_node_norm),
            "trapping_violation": bool(trapping_violation), "nan_flag": bool(nan_flag)
        }) + "\n")
        self._step_count += 1

    def save_trajectory(self, X_history: np.ndarray) -> None:
        """Persist X_history (T, N, d) to trajectory.npz."""
        np.savez_compressed(self.run_dir / "trajectory.npz", X=X_history)

    def finalize(self, steps_to_stabilize: int | None, label: str, runtime_s: float | None = None, **extra) -> None:
        """Flush JSONL, write summary.json, and upsert into index.db."""
        if runtime_s is None: runtime_s = time.monotonic() - self._t0
        self._fh.flush(); self._fh.close()
        summary = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": self._sha, "config_hash": self._chash,
            "config": self._cfg, "seed": self._seed,
            "steps_to_stabilize": steps_to_stabilize,
            "label": label, "runtime_s": runtime_s,
            "step_count": self._step_count, **extra
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._index_run(summary)

    def _index_run(self, summary: dict) -> None:
        # upsert one row into runs/index.db
        con = sqlite3.connect(self._db_path)
        con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            (summary["run_id"], summary["timestamp"], summary["git_sha"],
             summary["config_hash"], summary["seed"], summary["label"],
             summary["steps_to_stabilize"], summary["runtime_s"], str(self.run_dir)))
        con.commit(); con.close()


def load_steps(run_dir: str | Path) -> list[dict]:
    """Return step-type records from run.jsonl in order."""
    rows = []
    for line in (Path(run_dir) / "run.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("type") == "step": rows.append(r)
    return rows

def load_summary(run_dir: str | Path) -> dict:
    """Return parsed summary.json for a run dir."""
    return json.loads((Path(run_dir) / "summary.json").read_text())
