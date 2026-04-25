#!/usr/bin/env bash
#  Bootstrap + Build
#  macOS (Apple Silicon / Intel)
#  This script is fully idempotent: safe to run repeatedly.
#  It will install any missing dependencies, start services,
#  compile the C++ components, and launch the engine.

set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

#  0. Cleanup Stale Processes
echo "[BUILD] Cleaning up any stale instances..."
killall -9 SKS_Renderer &>/dev/null || true
killall -9 physics_engine &>/dev/null || true
pkill -9 -f "python main.py" &>/dev/null || true
pkill -9 -f "python3 main.py" &>/dev/null || true

#  1. Homebrew  (required for all native deps)
echo "[DEPS] Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    echo "[DEPS] Homebrew not found — installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Ensure brew is on PATH for Apple Silicon default install
    if [[ -x /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "[DEPS] Homebrew ✓"
fi

#  2. Native packages via Brew
BREW_DEPS=(raylib python@3.11)

for pkg in "${BREW_DEPS[@]}"; do
    if brew list --formula "$pkg" &>/dev/null; then
        echo "[DEPS] $pkg ✓"
    else
        echo "[DEPS] Installing $pkg..."
        brew install "$pkg"
    fi
done

#  3. Ollama  (install if missing)
echo "[DEPS] Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "[DEPS] Ollama not found — installing via Homebrew..."
    brew install ollama
else
    echo "[DEPS] Ollama ✓"
fi

echo "[DEPS] Setting up Python environment (forcing Python 3.11)..."
# Recreate venv if it's the wrong version or doesn't exist
RECREATE_VENV=false
if [[ ! -d "venv" ]]; then
    RECREATE_VENV=true
elif ! ./venv/bin/python3 --version 2>&1 | grep -q "3.11"; then
    echo "[DEPS] Existing venv is not Python 3.11. Recreating..."
    rm -rf venv
    RECREATE_VENV=true
fi

if [[ "$RECREATE_VENV" == "true" ]]; then
    echo "[DEPS] Creating virtual environment with Python 3.11..."
    # Ensure brew's python3.11 is reachable
    if ! command -v python3.11 &>/dev/null && [[ -x /opt/homebrew/bin/python3.11 ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    python3.11 -m venv venv
fi
source venv/bin/activate

# Upgrade pip (no longer silent so you can see it's working)
echo "[DEPS] Upgrading pip..."
./venv/bin/python3 -m pip install --upgrade pip
echo "[DEPS] Installing dependencies from requirements.txt (this may take a few minutes for Torch)..."
./venv/bin/python3 -m pip install -r requirements.txt
echo "[DEPS] Python packages ✓"

#  5. spaCy model
if ! ./venv/bin/python3 -c "import spacy; spacy.load('en_core_web_sm')" &>/dev/null; then
    echo "[DEPS] Downloading spaCy en_core_web_sm (3.5.0) model directly..."
    ./venv/bin/python3 -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
else
    echo "[DEPS] spaCy model ✓"
fi
echo "  Dependencies ready. Starting services + build..."

#  6. Cleanup previous run
echo "[BUILD] Cleaning up previous session..."
killall -9 SKS_Renderer 2>/dev/null || true
killall -9 physics_engine 2>/dev/null || true
rm -f /tmp/sks_shm /tmp/sks_roles

#  7. Ollama: start if not already serving
echo "[BUILD] Checking Ollama server..."
if curl -sf http://localhost:11434/api/tags > /dev/null; then
    echo "[BUILD] Ollama already running."
else
    echo "[BUILD] Starting Ollama..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 10s for the API to respond
    for i in $(seq 1 10); do
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then break; fi
        sleep 1
    done
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[BUILD] Ollama started (pid=$OLLAMA_PID)."
    else
        echo "[WARN] Ollama API did not respond within 10s."
    fi
fi

# Ensure required models are pulled (Hybrid Mode: 1B for speed, 3B for reasoning)
echo "[BUILD] Syncing Ollama models..."
for model in "llama3.2:1b" "llama3.2:3b"; do
    if ! ollama list | grep -q "$model"; then
        echo "[BUILD] Pulling $model..."
        ollama pull "$model"
    fi
done
echo "[BUILD] Ollama models ✓"


#  8.5 Download Premium Typography (Inter Font)
if [ ! -f "Inter-Medium.ttf" ]; then
    echo "[DEPS] Downloading Inter typography..."
    curl -sfL "https://github.com/google/fonts/raw/main/ofl/inter/static/Inter-Medium.ttf" -o "Inter-Medium.ttf" || \
    curl -sfL "https://rsms.me/inter/font-files/Inter-Medium.otf" -o "Inter-Medium.ttf" || \
    echo "[WARN] Could not download Inter font. Falling back to default."
fi

#  8.6 Cache-bust frontend assets so the browser does not reuse stale UI code
echo "[BUILD] Refreshing frontend asset version..."
ASSET_VERSION="$(date +%s)"
ASSET_VERSION="$ASSET_VERSION" python3 - <<'PY'
from pathlib import Path
import os
import re

index_path = Path("frontend/index.html")
text = index_path.read_text()
asset_version = os.environ["ASSET_VERSION"]
text = re.sub(r'href="style\\.css(?:\\?v=[^"]*)?"', f'href="style.css?v={asset_version}"', text)
text = re.sub(r'src="main\\.js(?:\\?v=[^"]*)?"', f'src="main.js?v={asset_version}"', text)
index_path.write_text(text)
PY

#  9. Compile C++ components
echo "[BUILD] p_physics.cpp -> physics_engine"
g++ -O3 -std=c++17 \
    -I"$(brew --prefix raylib)/include" \
    -L"$(brew --prefix raylib)/lib" -lraylib \
    -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL \
    p_physics.cpp -o physics_engine

echo "[BUILD] Build complete."

#  10. Shutdown hook
cleanup() {
    echo ""
    echo "[SHUTDOWN] Killing renderer and python workers..."
    killall -9 SKS_Renderer &>/dev/null || true
    killall -9 physics_engine &>/dev/null || true
    pkill -9 -f "python3.11" &>/dev/null || true
    echo "[SHUTDOWN] Done."
}
trap cleanup EXIT


#  11. Launch
(sleep 3 && open http://localhost:8000) &
./venv/bin/python3 main.py
