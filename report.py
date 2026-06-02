#!/usr/bin/env python3
# report.py — render E(t)/occupancy/mesh-timeline plots from a run dir.
# Usage: python report.py <run_dir> [--show]
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _load_steps(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "run.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("type") == "step": rows.append(r)
    return rows

def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}

def render(run_dir: str | Path, show: bool = False) -> None:
    """Generate energy/occupancy/mesh-timeline PNGs into <run_dir>/plots/."""
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    steps = _load_steps(run_dir)
    if not steps:
        print(f"No step data in {run_dir}"); return
    summary = _load_summary(run_dir)
    title_sfx = f"  [{summary.get('regime_label', '')}]" if summary.get("regime_label") else ""
    xs = [r["step"] for r in steps]
    Es = [r["E"] for r in steps]
    supports = [r["support_size"] for r in steps]
    meshes = [r["mesh_count"] for r in steps]

    # E(t) — energy trajectory
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, Es, lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("E"); ax.set_title(f"Energy E(t){title_sfx}")
    fig.tight_layout(); fig.savefig(plots_dir / "energy.png", dpi=100)
    if show: plt.show()
    plt.close(fig)

    # occupancy — support_size over time
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(xs, supports, lw=1, color="tab:orange")
    ax.set_xlabel("step"); ax.set_ylabel("support_size"); ax.set_title(f"Occupancy{title_sfx}")
    fig.tight_layout(); fig.savefig(plots_dir / "occupancy.png", dpi=100)
    if show: plt.show()
    plt.close(fig)

    # mesh timeline — step-function of active mesh count
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.step(xs, meshes, where="post", lw=1, color="tab:green")
    ax.set_xlabel("step"); ax.set_ylabel("mesh_count"); ax.set_title(f"Mesh Timeline{title_sfx}")
    fig.tight_layout(); fig.savefig(plots_dir / "mesh_timeline.png", dpi=100)
    if show: plt.show()
    plt.close(fig)

    print(f"Plots written to {plots_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render run plots from a run dir.")
    p.add_argument("run_dir", help="Path to a runs/<id>/ directory")
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    args = p.parse_args()
    render(args.run_dir, show=args.show)
