#!/usr/bin/env bash
# Builds the Python environment and the two native engines, then starts the system.
# Safe to run repeatedly; every step checks before it acts.

set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

echo "[build] Clearing stale processes"
pkill -9 -f "$ROOT_DIR/venv/physics-engine" &>/dev/null || true
pkill -9 -f "$ROOT_DIR/venv/thought-engine" &>/dev/null || true
pkill -9 -f "$ROOT_DIR/frontend/ui.py" &>/dev/null || true

# raylib is what the physics engine draws through, so the build fails without it.
if ! command -v brew &>/dev/null; then
    echo "[build] Homebrew is required for raylib. Install it from https://brew.sh" >&2
    exit 1
fi
brew list --formula raylib &>/dev/null || brew install raylib

if [[ ! -f venv/bin/activate ]]; then
    echo "[build] Creating the Python environment"
    python3.11 -m venv venv
fi
./venv/bin/python -m pip install --quiet --upgrade pip
./venv/bin/python -m pip install --quiet -r requirements.txt

# spaCy ships the parser separately from the library.
./venv/bin/python -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null \
    || ./venv/bin/python -m spacy download en_core_web_sm

# Synthesis runs against a local Ollama server, so the model has to be present.
if command -v ollama &>/dev/null; then
    curl -sf http://localhost:11434/api/tags >/dev/null || { ollama serve &>/dev/null & sleep 3; }
    ollama list | grep -q "llama3.2:3b" || ollama pull llama3.2:3b
else
    echo "[build] Ollama not found. Synthesis will fail until it is installed."
fi

# Both binaries land in venv/ because that is where the Python side looks for them.
echo "[build] Compiling the physics engine"
g++ -O3 -std=c++17 \
    -I"$(brew --prefix raylib)/include" \
    -L"$(brew --prefix raylib)/lib" -lraylib \
    -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL \
    src/engine/thought/physics.cpp -o venv/physics-engine

echo "[build] Compiling the thought engine"
g++ -O3 -std=c++17 src/engine/thought/thought.cpp -o venv/thought-engine

echo "[build] Done. Starting the engine on http://localhost:8000"
exec ./venv/bin/python main.py
