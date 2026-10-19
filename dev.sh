#!/usr/bin/env bash
# KeySpark local dev launcher.
#
# Starts the backend (FastAPI on :8000, owns the batch scheduler
# and a long-lived Spark session) and the frontend dev server (Vite
# on :5173). The backend is wrapped in run-with-backoff.sh so that if
# its Spark session dies (e.g. after a sleep/wake cycle) it auto-restarts
# with exponential backoff (5s up to 60s), without manual intervention.
#
# The Mac is allowed to sleep and lock normally — no `caffeinate`
# wrappers. The recording agent and streaming job (started by
# `startup-tmux.sh`) each recover from sleep/wake on their own; the
# agent EXITS on macOS wake notifications so a fresh interpreter
# respawns with a working event tap, and the backoff loop catches the
# Spark RPC death.
#
# Ctrl+C cleans both children up.

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo
  echo "stopping backend and frontend..."
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null
  wait 2>/dev/null
  echo "done"
}
trap cleanup INT TERM

cd "$ROOT"

echo "[backend]  http://localhost:8000   (FastAPI, auto-restart with backoff)"
"$ROOT/run-with-backoff.sh" uv run uvicorn keyspark.api:app &
BACKEND_PID=$!

echo "[frontend] http://localhost:5173   (Vite)"
( cd "$ROOT/frontend" && npm run dev ) &
FRONTEND_PID=$!

echo
echo "PIDs: backend=$BACKEND_PID  frontend=$FRONTEND_PID"
echo "Ctrl+C to stop both."
echo

wait
