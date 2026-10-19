"""KeySpark self-heal watchdog. The run-with-backoff.sh loops only respawn a
component when its process EXITS; that misses the "alive but wedged" failure macOS
sleep causes: Spark's local-mode RPC dies on wake, the JVM stays up, nothing
exits. Concretely streaming_job keeps its JVM but stops emitting windows, and the
api batch tick HANGS on the dead RPC (never raises), so last_run freezes and
api.py's os._exit recovery never runs - the dashboard sits offline until a human
restarts tmux.

This tiny external supervisor fixes both via two triggers:
  1. Wake (event-driven): on NSWorkspaceDidWakeNotification it bounces the
     streaming + backend windows after a short settle (a long sleep almost always
     kills the Spark RPC, so a fresh JVM is the reliable fix).
  2. Freshness poll (every CHECK_INTERVAL): backend if /api/batch_status is
     unreachable or last_run is frozen past BATCH_STALE_SECONDS; streaming only
     while the user is active AND no window has been produced for
     STREAM_STALE_SECONDS (gating on activity avoids restarting healthy idle).

Restart is `tmux respawn-pane -k`. A per-window cooldown + startup grace prevent
restart storms. On non-macOS the wake trigger and HID gate no-op; the backend
freshness check still runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------
# Settings (tmux targets match startup-tmux.sh; all overridable via env)
# --------------------------------------------------------------------------
SESSION = os.environ.get("KEYSPARK_TMUX_SESSION", "keyspark")
STREAMING_WINDOW = os.environ.get("KEYSPARK_STREAMING_WINDOW", "streaming")
BACKEND_WINDOW = os.environ.get("KEYSPARK_BACKEND_WINDOW", "backend")
API_BASE = os.environ.get("KEYSPARK_API", "http://localhost:8000")

CHECK_INTERVAL = 60.0         # tune: seconds between freshness checks
PUMP_CHUNK = 5.0              # tune: runloop pump granularity (keeps wake responsive)
HID_ACTIVE_SECONDS = 60.0     # tune: user is "active" if last input was within this
STREAM_STALE_SECONDS = 180.0  # tune: active this long with no new window => wedged
BATCH_STALE_SECONDS = 900.0   # tune: last_run frozen longer than this => dead/hung
RESTART_COOLDOWN = 240.0      # tune: do not re-restart a window within this window
STARTUP_GRACE = 180.0         # tune: skip checks this long after start / a bounce
WAKE_SETTLE_SECONDS = 20.0    # tune: let the host settle after wake before bouncing

# pyobjc is only present (and meaningful) on macOS. On Linux the wake trigger
# no-ops and only the freshness poll applies.
try:
    import objc
    from AppKit import NSWorkspace
    from Foundation import NSDate, NSObject, NSRunLoop

    _HAS_PYOBJC = True
except ImportError:
    _HAS_PYOBJC = False


# --------------------------------------------------------------------------
# macOS wake observer
# --------------------------------------------------------------------------
if _HAS_PYOBJC:

    class _WakeObserver(NSObject):
        # NSObject subclass so NSNotificationCenter can call workspaceDidWake:.
        # Mirrors agent.py's observer.
        def initWithCallback_(self, cb):
            self = objc.super(_WakeObserver, self).init()
            if self is None:
                return None
            self._cb = cb
            return self

        def workspaceDidWake_(self, _notification):
            try:
                self._cb()
            except Exception:
                pass


def _install_wake_observer(on_wake):
    # Returns (observer, pump). The observer must outlive the run loop or
    # NSNotificationCenter drops it. pump(seconds) spins the run loop so the wake
    # notification can dispatch; on non-macOS it falls back to time.sleep.
    if not _HAS_PYOBJC:
        return None, time.sleep

    observer = _WakeObserver.alloc().initWithCallback_(on_wake)
    nc = NSWorkspace.sharedWorkspace().notificationCenter()
    nc.addObserver_selector_name_object_(
        observer, "workspaceDidWake:", "NSWorkspaceDidWakeNotification", None
    )

    def pump(seconds: float) -> None:
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(seconds)
        )

    return observer, pump


# --------------------------------------------------------------------------
# Probes (HID idle, API JSON, timestamp parse, tmux restart)
# --------------------------------------------------------------------------
def _hid_idle_seconds() -> float | None:
    # Seconds since the last keyboard/mouse/trackpad event, from IOHIDSystem. None
    # on failure or non-macOS. Same probe agent.py uses.
    try:
        out = subprocess.check_output(
            ["ioreg", "-c", "IOHIDSystem", "-rd1"],
            text=True,
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        if "HIDIdleTime" not in line:
            continue
        try:
            ns = int(line.split("=", 1)[1].strip())
        except (IndexError, ValueError):
            return None
        return ns / 1_000_000_000.0
    return None


def _get_json(path: str, timeout: float = 5.0):
    # GET API_BASE+path as JSON. None on any error. None (unreachable) vs []
    # (empty) matters for the streaming check, so callers handle both.
    try:
        with urlopen(API_BASE + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, ValueError, TimeoutError):
        return None


def _parse_epoch(s) -> float | None:
    # Parse an API timestamp (Z-suffixed, offset-bearing, or naive-UTC) to epoch
    # seconds. None on missing/unparseable input.
    if not s:
        return None
    try:
        text = str(s).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _restart(window: str, reason: str, cooldown: dict, now: float) -> None:
    # Restart one tmux window via respawn-pane -k (kills the wedged process tree
    # and re-runs the pane's run-with-backoff command). Best-effort.
    if now < cooldown.get(window, 0.0):
        return
    target = f"{SESSION}:{window}"
    try:
        subprocess.run(
            ["tmux", "respawn-pane", "-k", "-t", target],
            check=True,
            timeout=10,
            capture_output=True,
            text=True,
        )
        cooldown[window] = now + RESTART_COOLDOWN
    except (subprocess.SubprocessError, OSError):
        pass


# --------------------------------------------------------------------------
# Wedged checks
# --------------------------------------------------------------------------
def _backend_wedged() -> tuple[bool, str]:
    # True if the API is unreachable or batch last_run is frozen past
    # BATCH_STALE_SECONDS. A None last_run (never run / still building Spark) is
    # not-wedged, so a freshly respawned API is not immediately bounced again.
    st = _get_json("/api/batch_status")
    if st is None:
        return True, "API unreachable"
    last_run = _parse_epoch(st.get("last_run"))
    if last_run is None:
        return False, ""
    age = time.time() - last_run
    if age > BATCH_STALE_SECONDS:
        return True, f"batch last_run frozen {int(age)}s (status={st.get('status')})"
    return False, ""


def _metrics_newest() -> tuple[bool, float | None]:
    # (reachable, newest_window_end_epoch). reachable=False means the API did not
    # answer (the backend check owns that). reachable=True with newest=None means
    # up but no recent windows.
    rows = _get_json("/api/metrics")
    if rows is None:
        return False, None
    newest = None
    for m in rows:
        end = _parse_epoch(m.get("window_end"))
        if end is not None and (newest is None or end > newest):
            newest = end
    return True, newest


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> None:
    wake = threading.Event()

    def on_wake() -> None:
        wake.set()

    _observer, pump = _install_wake_observer(on_wake)

    now = time.time()
    next_check = now + STARTUP_GRACE
    cooldown: dict[str, float] = {}
    active_since: float | None = None
    wake_at: float | None = None

    try:
        while True:
            pump(PUMP_CHUNK)
            now = time.time()

            # Wake-triggered bounce, after a settle so the host (and Kafka/network)
            # is back before respawning the Spark processes.
            if wake.is_set() and wake_at is None:
                wake_at = now
            if wake_at is not None and (now - wake_at) >= WAKE_SETTLE_SECONDS:
                wake.clear()
                wake_at = None
                _restart(BACKEND_WINDOW, "wake", cooldown, now)
                _restart(STREAMING_WINDOW, "wake", cooldown, now)
                active_since = None
                next_check = now + STARTUP_GRACE  # let them warm up first
                continue

            if now < next_check:
                continue
            next_check = now + CHECK_INTERVAL

            # Backend: timer-based, not gated on user activity.
            wedged, why = _backend_wedged()
            if wedged:
                _restart(BACKEND_WINDOW, why, cooldown, now)

            # Streaming: only meaningful while the user is active (no windows are
            # produced or expected during idle).
            idle = _hid_idle_seconds()
            if idle is not None and idle < HID_ACTIVE_SECONDS:
                if active_since is None:
                    active_since = now
            else:
                active_since = None

            if active_since is not None and (now - active_since) >= STREAM_STALE_SECONDS:
                reachable, newest = _metrics_newest()
                if reachable and (newest is None or newest < active_since):
                    _restart(
                        STREAMING_WINDOW,
                        f"no metrics window in {int(now - active_since)}s of activity",
                        cooldown,
                        now,
                    )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
