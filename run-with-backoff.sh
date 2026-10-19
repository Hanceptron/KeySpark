#!/usr/bin/env bash
# Run "$@" forever, restarting on exit with exponential backoff (5s,
# doubling up to 60s) that resets to 5s after any run lasting longer
# than 60s. A one-off crash still recovers in 5s, but a crash-on-startup
# (e.g. Kafka still down right after a host wake, or a config error)
# stops tight-looping the CPU and flooding the logs.
#
# Always runs from the repo root (this script's directory) so `uv run`
# and the output/ paths resolve regardless of the caller's cwd.
set -u
cd "$(dirname "$0")" || exit 1

delay=5
while true; do
  start=$(date +%s)
  "$@"
  code=$?
  elapsed=$(( $(date +%s) - start ))
  if (( elapsed > 60 )); then
    delay=5
  fi
  echo "[$(date)] '$*' exited (code $code) after ${elapsed}s; restarting in ${delay}s..."
  sleep "$delay"
  if (( elapsed <= 60 )); then
    delay=$(( delay * 2 ))
    (( delay > 60 )) && delay=60
  fi
done
