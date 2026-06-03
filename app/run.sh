#!/usr/bin/env bash
# run.sh — Start EEG-to-Text backend and frontend
# Run from repo root:  bash app/run.sh
# Or from app folder:  bash run.sh

set -e

# Resolve repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT"

# ── Find Python ───────────────────────────────────────────────────────────────
PY=""

# 1. Virtual environment inside the repo (checked first — always preferred)
for candidate in \
    "$ROOT/.venv/Scripts/python.exe" \
    "$ROOT/.venv/Scripts/python" \
    "$ROOT/venv/Scripts/python.exe" \
    "$ROOT/venv/Scripts/python" \
    "$ROOT/.venv/bin/python" \
    "$ROOT/venv/bin/python"
do
    if [ -f "$candidate" ]; then
        PY="$candidate"
        break
    fi
done

# 2. System-level fallbacks (python3 before python, py for Windows launcher)
if [ -z "$PY" ]; then
    for cmd in python3 python py; do
        if command -v "$cmd" > /dev/null 2>&1; then
            PY="$cmd"
            break
        fi
    done
fi

# 3. Hard stop — nothing found
if [ -z "$PY" ]; then
    echo ""
    echo "[ERROR] Could not find a Python executable."
    echo "  Tried: .venv/Scripts/python.exe, venv/Scripts/python, python3, python, py"
    echo "  Activate your virtual environment first, or install Python."
    echo ""
    exit 1
fi

echo ""
echo "======================================================"
echo "  NeuroText — EEG-to-Text App"
echo "======================================================"
echo "  Root : $ROOT"
echo "  Python: $PY"
echo "======================================================"
echo ""

# ── Start FastAPI backend ─────────────────────────────────────────────────────
echo "[1/2] Starting FastAPI backend on http://localhost:8000 ..."
"$PY" -m uvicorn app.backend.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "      Backend PID: $BACKEND_PID"

# Wait until backend is accepting connections (max 15s)
echo "      Waiting for backend to be ready..."
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "      Backend is up."
        break
    fi
    sleep 1
done

echo ""

# ── Start Streamlit frontend ──────────────────────────────────────────────────
echo "[2/2] Starting Streamlit frontend on http://localhost:8501 ..."
"$PY" -m streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!
echo "      Frontend PID: $FRONTEND_PID"

echo ""
echo "======================================================"
echo "  Backend  : http://localhost:8000"
echo "  Frontend : http://localhost:8501"
echo "  API docs : http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both processes."
echo "======================================================"
echo ""

# ── Graceful shutdown on Ctrl+C ───────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$BACKEND_PID"  2>/dev/null
    kill "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID"  2>/dev/null
    wait "$FRONTEND_PID" 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Keep script alive until Ctrl+C
wait
