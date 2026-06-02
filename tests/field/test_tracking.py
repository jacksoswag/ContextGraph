from __future__ import annotations
import json, sqlite3
import numpy as np
import pytest
from field.tracking import RunLogger, config_hash, load_steps, load_summary

_CFG = {"d": 4, "beta": 2.0, "mu": 0.1, "eta": 0.01, "tau_support": 0.5}

def _logger(tmp_path) -> RunLogger:
    return RunLogger(_CFG, seed=42, runs_root=tmp_path)

# --- config_hash ---

def test_config_hash_stable():
    # same dict → same hash across repeated calls
    assert config_hash(_CFG) == config_hash(_CFG)

def test_config_hash_order_invariant():
    # key insertion order must not affect the hash
    a = {"d": 4, "beta": 2.0}; b = {"beta": 2.0, "d": 4}
    assert config_hash(a) == config_hash(b)

def test_config_hash_length():
    assert len(config_hash(_CFG)) == 16

def test_config_hash_distinct_on_different_cfg():
    other = {**_CFG, "beta": 99.0}
    assert config_hash(_CFG) != config_hash(other)

# --- write / read round-trip ---

def test_step_roundtrip(tmp_path):
    log = _logger(tmp_path)
    for i in range(5):
        log.log_step(i, E=float(10 - i), delta_X=0.5 - i * 0.05,
                     support_size=2 + i, mesh_count=1,
                     max_node_norm=0.8, trapping_violation=False, nan_flag=False)
    log.finalize(steps_to_stabilize=4, label="settled", runtime_s=0.1)
    steps = load_steps(log.run_dir)
    assert len(steps) == 5
    assert steps[0]["E"] == 10.0
    assert steps[4]["step"] == 4
    assert steps[4]["support_size"] == 6
    assert not steps[0]["trapping_violation"]
    assert not steps[0]["nan_flag"]

def test_summary_roundtrip(tmp_path):
    log = _logger(tmp_path)
    log.log_step(0, E=5.0, delta_X=0.1, support_size=3, mesh_count=1,
                 max_node_norm=0.9, trapping_violation=False, nan_flag=False)
    log.finalize(steps_to_stabilize=0, label="settled", runtime_s=0.42)
    s = load_summary(log.run_dir)
    assert s["step_count"] == 1
    assert s["label"] == "settled"
    assert s["seed"] == 42
    assert s["config_hash"] == config_hash(_CFG)
    assert s["steps_to_stabilize"] == 0
    assert abs(s["runtime_s"] - 0.42) < 1e-9
    assert s["config"] == _CFG

def test_extra_fields_in_summary(tmp_path):
    # **extra kwargs should be persisted in summary.json
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=0.0,
                 mesh_size=7, relevance=[1.0, 2.0])
    s = load_summary(log.run_dir)
    assert s["mesh_size"] == 7
    assert s["relevance"] == [1.0, 2.0]

def test_trajectory_roundtrip(tmp_path):
    log = _logger(tmp_path)
    X = np.random.default_rng(7).standard_normal((20, 4, 4))
    log.save_trajectory(X)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=0.0)
    loaded = np.load(log.run_dir / "trajectory.npz")["X"]
    assert np.allclose(loaded, X)

# --- JSONL structure ---

def test_header_line_present(tmp_path):
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=0.0)
    lines = (log.run_dir / "run.jsonl").read_text().strip().splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "header"
    assert header["config_hash"] == config_hash(_CFG)
    assert header["seed"] == 42
    assert header["config"] == _CFG

def test_no_steps_produces_empty_step_list(tmp_path):
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="settled", runtime_s=0.0)
    assert load_steps(log.run_dir) == []
    assert load_summary(log.run_dir)["step_count"] == 0

# --- run dir layout ---

def test_run_dir_structure(tmp_path):
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=0.0)
    assert (log.run_dir / "run.jsonl").exists()
    assert (log.run_dir / "summary.json").exists()
    assert (log.run_dir / "plots").is_dir()

def test_run_id_contains_confighash(tmp_path):
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=0.0)
    assert config_hash(_CFG) in log.run_id

# --- sqlite index ---

def test_sqlite_row_inserted(tmp_path):
    log = _logger(tmp_path)
    log.log_step(0, E=3.0, delta_X=0.2, support_size=2, mesh_count=1,
                 max_node_norm=0.6, trapping_violation=False, nan_flag=False)
    log.finalize(steps_to_stabilize=0, label="settled", runtime_s=0.7)
    db = tmp_path / "index.db"
    assert db.exists()
    con = sqlite3.connect(db)
    rows = con.execute("SELECT run_id, label, runtime_s FROM runs").fetchall()
    con.close()
    assert len(rows) == 1
    run_id, label, rt = rows[0]
    assert run_id == log.run_id
    assert label == "settled"
    assert abs(rt - 0.7) < 1e-9

def test_sqlite_upsert_on_duplicate_run_id(tmp_path):
    # two finalize calls with same run_id should upsert, not error
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=1, label="settled", runtime_s=0.1)
    # re-open same run (simulate re-finalize) via direct _index_run
    s = load_summary(log.run_dir)
    s["label"] = "resettled"
    log._index_run(s)
    con = sqlite3.connect(tmp_path / "index.db")
    rows = con.execute("SELECT label FROM runs WHERE run_id=?", (log.run_id,)).fetchall()
    con.close()
    assert len(rows) == 1
    assert rows[0][0] == "resettled"

def test_multiple_runs_indexed(tmp_path):
    for seed in (1, 2, 3):
        log = RunLogger(_CFG, seed=seed, runs_root=tmp_path)
        log.finalize(steps_to_stabilize=seed, label="settled", runtime_s=0.0)
    con = sqlite3.connect(tmp_path / "index.db")
    count = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    con.close()
    assert count == 3

# --- runtime computed when not supplied ---

def test_runtime_auto_computed(tmp_path):
    log = _logger(tmp_path)
    log.finalize(steps_to_stabilize=None, label="", runtime_s=None)
    s = load_summary(log.run_dir)
    assert s["runtime_s"] >= 0.0
