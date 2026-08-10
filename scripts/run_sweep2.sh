#!/usr/bin/env bash
# Launcher for the second, non-overlapping 500k sweep + its graph corpus.
#
# Subcommands (run from the repo root, or anywhere — paths are resolved):
#   prepare        build data/raw/corpus_v2 index (prior ids excluded, fresh seeds)
#   calibrate [N]  short scrape (default 180s) to project the wall-clock, no commit to targets
#   scrape         launch the full 500k NEW scrape in the background (caffeinate)
#   graph          launch graph-corpus construction over corpus_v2 in the background
#   status         print scrape + ingest progress
#   stop           stop a running scrape and/or ingest
#
# Nothing runs until you ask for it. `scrape` and `graph` detach via nohup and
# keep the Mac awake with caffeinate; both are resumable (just re-run).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python3"
V2="$ROOT/data/raw/corpus_v2"
GRAPH="$ROOT/data/graph_v2"
SEEDS="$V2/seeds.txt"
TARGET="${DI_TARGET:-500000}"
MIN_FREE_GB="${DI_MIN_FREE_GB:-45}"

have_index() { [[ -f "$V2/index.sqlite3" ]]; }

free_gb() { df -g "$ROOT" | tail -1 | awk '{print $4}'; }

cmd_prepare() {
  "$PY" "$ROOT/scripts/prepare_sweep2.py" "$@"
}

cmd_calibrate() {
  have_index || { echo "Run '$0 prepare' first."; exit 1; }
  local secs="${1:-180}"
  echo "Calibrating for ${secs}s against corpus_v2 (dedup excludes the prior 500k)..."
  DI_CORPUS_DIR="$V2" DI_SEEDS_FILE="$SEEDS" \
    "$PY" "$ROOT/scripts/bulk_scrape.py" --calibrate "$secs" --target "$TARGET" --tag calib
  echo "Calibration report: $V2/calibration.json"
}

cmd_scrape() {
  have_index || { echo "Run '$0 prepare' first."; exit 1; }
  if [[ -f "$V2/run.pid" ]] && kill -0 "$(cat "$V2/run.pid")" 2>/dev/null; then
    echo "Scrape already running (pid $(cat "$V2/run.pid"))."; exit 1
  fi
  local envkv=(DI_CORPUS_DIR="$V2")
  [[ -f "$SEEDS" ]] && envkv+=(DI_SEEDS_FILE="$SEEDS")
  echo "Launching sweep-2 scrape (target=$TARGET NEW articles) in background..."
  nohup env "${envkv[@]}" \
    caffeinate -is "$PY" "$ROOT/scripts/bulk_scrape.py" --run --target "$TARGET" \
    >"$V2/run.out" 2>&1 &
  echo $! > "$V2/run.pid"
  echo "pid $(cat "$V2/run.pid")  log: $V2/run.out  progress: $V2/progress.json"
}

cmd_graph() {
  # never extract an empty corpus: require the scrape to have written shards
  if [[ -z "$(find "$V2" -name 'shard-*.jsonl.gz' -print -quit 2>/dev/null)" ]]; then
    echo "REFUSING: no shards under $V2 yet. Run '$0 scrape' first and let it collect articles."
    exit 1
  fi
  local free; free="$(free_gb)"
  if (( free < MIN_FREE_GB )); then
    echo "REFUSING: only ${free}GB free, graph build needs ~${MIN_FREE_GB}GB."
    echo "Free space first (e.g. remove the 26GB data/graph/graph.corpus.sql dump,"
    echo "which is regenerable from data/graph/graph.corpus.sqlite), then re-run."
    echo "Override with DI_MIN_FREE_GB=<n> if you know better."
    exit 1
  fi
  if [[ -f "$GRAPH/ingest.pid" ]] && kill -0 "$(cat "$GRAPH/ingest.pid")" 2>/dev/null; then
    echo "Graph build already running (pid $(cat "$GRAPH/ingest.pid"))."; exit 1
  fi
  mkdir -p "$GRAPH"
  echo "Launching sweep-2 graph construction in background (free=${free}GB)..."
  nohup caffeinate -is "$PY" "$ROOT/scripts/ingest_run_sweep2.py" \
    >"$GRAPH/orchestrator.out" 2>&1 &
  echo $! > "$GRAPH/ingest.pid"
  echo "pid $(cat "$GRAPH/ingest.pid")  status: $GRAPH/ingest_status.json"
}

cmd_status() {
  echo "=== free disk ==="; echo "$(free_gb)GB"
  echo "=== scrape (corpus_v2) ==="
  if [[ -f "$V2/index.sqlite3" ]]; then
    sqlite3 "$V2/index.sqlite3" \
      "SELECT channel, COUNT(*) FROM seen WHERE channel!='_prior' GROUP BY channel;" 2>/dev/null \
      || echo "(index busy)"
  else
    echo "(not prepared)"
  fi
  [[ -f "$V2/progress.json" ]] && "$PY" -c "import json;d=json.load(open('$V2/progress.json'));print('total=%s/%s pct=%s eta=%s'%(d['total'],d['target'],d['pct'],d['eta_human']))" 2>/dev/null || true
  echo "=== graph (corpus_v2) ==="
  [[ -f "$GRAPH/ingest_status.json" ]] && cat "$GRAPH/ingest_status.json" || echo "(not started)"
}

cmd_stop() {
  for f in "$V2/run.pid" "$GRAPH/ingest.pid"; do
    if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
      echo "stopping pid $(cat "$f") ($f)"; kill "$(cat "$f")"
    fi
  done
}

sub="${1:-}"; shift || true
case "$sub" in
  prepare)   cmd_prepare "$@" ;;
  calibrate) cmd_calibrate "$@" ;;
  scrape)    cmd_scrape "$@" ;;
  graph)     cmd_graph "$@" ;;
  status)    cmd_status ;;
  stop)      cmd_stop ;;
  *) echo "usage: $0 {prepare|calibrate [secs]|scrape|graph|status|stop}"; exit 1 ;;
esac
