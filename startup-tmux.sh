#!/usr/bin/env bash
# KeySpark - tmux startup.
#
# Starts the five long-running processes (agent, streaming, backend,
# frontend, watchdog) inside a single detached tmux session named
# "keyspark", each in its own window so logs stay separated.
#
# The Mac is allowed to sleep and lock normally — no `caffeinate`
# wrappers. Every process is wrapped in run-with-backoff.sh, which
# restarts it on exit with exponential backoff (5s up to 60s). On macOS
# wake the agent EXITS on NSWorkspaceDidWakeNotification (pynput's event
# tap is silently killed during sleep and cannot be rebuilt in-process),
# so the backoff loop respawns a fresh interpreter with a working tap.
#
# The backoff loop only covers processes that EXIT. The watchdog window
# (keyspark.watchdog) covers the "alive but wedged" case the backoff loop
# misses: after wake it bounces streaming + backend (whose Spark RPC dies
# on sleep), and it restarts either if their output goes stale. See
# keyspark/watchdog.py.
#
# Usage:
#   ./startup-tmux.sh                       # start everything detached
#   tmux attach -t keyspark              # look at the live logs
#   tmux kill-session -t keyspark        # stop everything
#
# Inside tmux:
#   Ctrl+B then 0-4 - cycle windows (agent, streaming, backend, frontend, watchdog)
#   Ctrl+B then D   - detach without stopping

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install with: brew install tmux"
  exit 1
fi

# Make sure Kafka is up before any consumer starts. Docker handles the
# "already running" case quietly, so it's safe to call every time.
docker compose up -d >/dev/null || {
  echo "Failed to start Kafka via docker compose. Is Docker Desktop running?"
  exit 1
}

# Wait for the broker to actually accept connections before launching
# the agent. On cold boot the container starts in milliseconds but
# the broker needs a few seconds to listen on 9092; without this gate
# the agent would silently buffer into a dead socket.
echo "waiting for Kafka on localhost:9092..."
deadline=$(( $(date +%s) + 60 ))
until nc -z localhost 9092 2>/dev/null; do
  if (( $(date +%s) > deadline )); then
    echo "Kafka did not become reachable on localhost:9092 within 60s."
    exit 1
  fi
  sleep 1
done
echo "Kafka is reachable."

# Make sure events.raw exists before the streaming job subscribes to
# it. Kafka's auto.create.topics.enable triggers on produce but not on
# the Spark KafkaSource admin lookup, so without this step a fresh
# Kafka container puts the streaming job into a crash loop on
# UnknownTopicOrPartitionException until the agent produces its first
# event. --if-not-exists makes this idempotent for the common case
# (Kafka was just restarted with the topic intact).
# This is where the Kafka partition count is defined: 1 partition for the
# events.raw topic. One partition gives global event ordering and matches the
# single-agent / single-user setup.
docker exec keyspark-kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic events.raw \
  --partitions 1 --replication-factor 1 >/dev/null

# NOTE: a stale-checkpoint wipe used to live here - on a Kafka offset
# mismatch it ran `rm -rf output/checkpoint output/{metrics,events}/_spark_metadata`.
# That deleted the Structured Streaming commit logs, which orphaned every
# event part file written before the wipe and made the batch silently drop
# weeks of history. It has been removed: streaming_job.py now sets
# failOnDataLoss=false (the Kafka source resets to available offsets instead
# of crashing on a mismatch), and batch_job.py reads output/events as plain
# parquet, independent of the commit log. Never delete _spark_metadata.

# Kill any previous session so this script is idempotent.
tmux kill-session -t keyspark 2>/dev/null

tmux new-session -d -s keyspark -n agent \
  "$ROOT/run-with-backoff.sh uv run python -m keyspark.agent --sink kafka"

tmux new-window -t keyspark -n streaming \
  "$ROOT/run-with-backoff.sh uv run python -m keyspark.streaming_job"

tmux new-window -t keyspark -n backend \
  "$ROOT/run-with-backoff.sh uv run uvicorn keyspark.api:app"

tmux new-window -t keyspark -n frontend \
  "cd $ROOT/frontend && npm run dev"

# Self-heal supervisor. Restarts streaming/backend when they wedge alive
# (the case run-with-backoff cannot see). Wrapped in the backoff loop too
# so the watchdog itself recovers if it ever crashes.
tmux new-window -t keyspark -n watchdog \
  "$ROOT/run-with-backoff.sh uv run python -m keyspark.watchdog"

echo
echo "KeySpark started in detached tmux session 'keyspark'."
echo "  Attach: tmux attach -t keyspark"
echo "  Stop:   tmux kill-session -t keyspark"
echo "  Open:   http://localhost:5173"
