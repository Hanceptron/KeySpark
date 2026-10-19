"""KeySpark input capture agent. Captures keyboard and mouse events on macOS via
pynput and emits one JSON object per event to stdout or Kafka. Needs macOS
Accessibility + Input Monitoring and cannot run in a container (see README).

Sleep/wake recovery: macOS kills pynput's event tap during sleep, and in-process
re-creation of listeners after wake is unreliable (the new Listener threads report
is_alive() True but the CGEventTap can be silently dead, so the agent looks
healthy while producing nothing). So on NSWorkspaceDidWakeNotification the agent
flags itself for exit and the outer restart loop respawns a fresh process with a
fresh tap; NSWorkspaceWillSleepNotification triggers a Kafka flush before the
socket dies. A second guard is the HID-idle watchdog: if macOS reports the user
as actively interacting but the agent has produced nothing for
WATCHDOG_SILENT_SECONDS, the listener is presumed dead and the same exit/respawn
path runs. The main loop spins the NSRunLoop so notifications can dispatch. On
non-macOS the observer and watchdog no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from pynput import keyboard, mouse

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
# User id stamped on every event. Default "user-001"; override with the
# KEYSPARK_USER env var or --user, so a second machine / enrollment session can
# capture under its own id (p02, p03, ...) for the multi-user impostor evaluation.
USER_ID = os.environ.get("KEYSPARK_USER", "user-001")  # tune: default user id

KAFKA_BOOTSTRAP = "localhost:9092"   # tune: Kafka broker address
KAFKA_TOPIC = "events.raw"           # tune: destination topic

MOVE_MIN_INTERVAL = 0.020            # tune: min s between mouse-move events (~50/s cap)

# Watchdog: if the OS reports input within WATCHDOG_ACTIVE_SECONDS but the agent
# has emitted nothing for WATCHDOG_SILENT_SECONDS, the listener is presumed dead
# and we exit so the outer restart loop respawns a fresh process.
WATCHDOG_ACTIVE_SECONDS = 5.0        # tune: HID recency that counts as "user active"
WATCHDOG_SILENT_SECONDS = 60.0       # tune: silence-while-active before presuming dead
WATCHDOG_CHECK_INTERVAL = 5.0        # tune: watchdog poll interval

# pyobjc is only available (and only meaningful) on macOS. Importing inside
# try/except lets the agent still run on Linux for testing (wake recovery no-ops).
try:
    import objc
    from AppKit import NSScreen, NSWorkspace
    from ApplicationServices import AXIsProcessTrusted
    from Foundation import NSDate, NSObject, NSRunLoop
    _HAS_PYOBJC = True
except ImportError:
    _HAS_PYOBJC = False
    AXIsProcessTrusted = None  # type: ignore
    NSScreen = None  # type: ignore


# --------------------------------------------------------------------------
# Kafka Producer
# --------------------------------------------------------------------------
def _key_id(key) -> str:
    # Stable id only. Printable keys -> the character; special keys -> pynput's
    # "Key.space"-style repr. Used for timing/n-gram features; never persist raw text.
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return key.char
    return str(key)


def _stdout_sink(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _kafka_sink():
    # Kafka producer. Builds a confluent_kafka Producer to localhost:9092 and
    # writes one JSON event per keystroke/mouse action to topic "events.raw",
    # keyed by USER_ID. This is the write side the Spark streaming job consumes.
    from confluent_kafka import Producer

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": "keyspark-agent",
        # Reconnect with backoff so the socket recovers on its own after
        # sleep/wake or a docker restart.
        "reconnect.backoff.ms": 500,
        "reconnect.backoff.max.ms": 10_000,
        "socket.keepalive.enable": True,
        # Retry transient sends; capped low because input events are cheap to lose.
        "message.send.max.retries": 5,
        "retry.backoff.ms": 200,
    })
    user_key = USER_ID.encode()

    def send(event: dict) -> None:
        producer.poll(0)
        producer.produce(KAFKA_TOPIC, key=user_key, value=json.dumps(event).encode())

    def flush() -> None:
        producer.flush(5)

    return send, flush


# --------------------------------------------------------------------------
# pynput listeners
# --------------------------------------------------------------------------
def _build_listeners(send):
    last_move = 0.0

    def on_press(key):
        send({"type": "key_down", "key": _key_id(key), "user": USER_ID, "ts": time.time()})

    def on_release(key):
        send({"type": "key_up", "key": _key_id(key), "user": USER_ID, "ts": time.time()})

    def on_move(x, y):
        nonlocal last_move
        t = time.time()
        if t - last_move < MOVE_MIN_INTERVAL:
            return
        last_move = t
        send({"type": "move", "x": int(x), "y": int(y), "user": USER_ID, "ts": t})

    def on_click(x, y, button, pressed):
        send({
            "type": "click", "x": int(x), "y": int(y),
            "button": str(button), "pressed": bool(pressed),
            "user": USER_ID, "ts": time.time(),
        })

    def on_scroll(x, y, dx, dy):
        send({
            "type": "scroll", "x": int(x), "y": int(y),
            "dx": int(dx), "dy": int(dy),
            "user": USER_ID, "ts": time.time(),
        })

    kb = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    return kb, ms


class _ListenerSupervisor:
    # Owns the active pynput listeners for the process lifetime. No in-process
    # restart on purpose: re-creating Listeners after a macOS sleep leaves the
    # CGEventTap quietly broken, so wake recovery exits the whole process and the
    # outer bash loop respawns a clean interpreter.
    def __init__(self, send):
        self._send = send
        self._kb = None
        self._ms = None

    def start(self):
        self._kb, self._ms = _build_listeners(self._send)
        self._kb.start()
        self._ms.start()

    def stop(self):
        for listener in (self._kb, self._ms):
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass

    def is_alive(self) -> bool:
        return (
            self._kb is not None
            and self._ms is not None
            and self._kb.is_alive()
            and self._ms.is_alive()
        )


# --------------------------------------------------------------------------
# macOS wake/sleep observer + HID idle + display size
# --------------------------------------------------------------------------
if _HAS_PYOBJC:
    class _WakeObserver(NSObject):
        # NSObject subclass so NSNotificationCenter can call the workspace…:
        # selectors. pyobjc maps trailing-underscore names to trailing-colon
        # selectors (workspaceDidWake_ -> workspaceDidWake:).
        def initWithCallbacks_(self, callbacks):
            self = objc.super(_WakeObserver, self).init()
            if self is None:
                return None
            self._on_wake, self._on_sleep = callbacks
            return self

        def workspaceWillSleep_(self, _notification):
            try:
                self._on_sleep()
            except Exception:
                pass

        def workspaceDidWake_(self, _notification):
            try:
                self._on_wake()
            except Exception:
                pass


def _install_wake_observer(on_wake, on_sleep):
    # Returns (observer, pump): the observer must outlive the run loop or
    # NSNotificationCenter drops it; pump(seconds) blocks while spinning the
    # NSRunLoop so pending sleep/wake notifications can fire. Falls back to
    # time.sleep on non-macOS.
    if not _HAS_PYOBJC:
        return None, time.sleep

    observer = _WakeObserver.alloc().initWithCallbacks_((on_wake, on_sleep))
    nc = NSWorkspace.sharedWorkspace().notificationCenter()
    nc.addObserver_selector_name_object_(
        observer, "workspaceDidWake:", "NSWorkspaceDidWakeNotification", None
    )
    nc.addObserver_selector_name_object_(
        observer, "workspaceWillSleep:", "NSWorkspaceWillSleepNotification", None
    )

    def pump(seconds: float) -> None:
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(seconds)
        )

    return observer, pump


def _hid_idle_seconds() -> float | None:
    # Seconds since the last HID (keyboard/mouse/trackpad) event, read from
    # IOHIDSystem via ioreg. None on any failure. Same value macOS uses for its
    # "user is idle" state, so it tells us whether the user is actually
    # interacting regardless of whether our pynput listener noticed.
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


def _record_display_size() -> None:
    # Write the primary display size (points) to output/display.json so the
    # dashboard can frame the mouse heatmap to the real screen. The magnitude of
    # NSScreen.mainScreen().frame().size matches pynput's coordinate range.
    # Rewritten on every startup, so it follows the current monitor setup. Screen
    # queries do not need Accessibility. Best-effort: any failure is non-fatal.
    if not _HAS_PYOBJC or NSScreen is None:
        return
    try:
        size = NSScreen.mainScreen().frame().size
        Path("output").mkdir(exist_ok=True)
        Path("output/display.json").write_text(
            json.dumps({"width": int(size.width), "height": int(size.height)})
        )
    except Exception:
        pass


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="KeySpark input capture agent")
    parser.add_argument("--sink", choices=("stdout", "kafka"), default="stdout")
    parser.add_argument("--user", default=None, help="user id to stamp (overrides KEYSPARK_USER)")
    args = parser.parse_args()

    if args.user:
        global USER_ID
        USER_ID = args.user

    _record_display_size()

    flush = None
    if args.sink == "kafka":
        raw_send, flush = _kafka_sink()
    else:
        raw_send = _stdout_sink

    # Watchdog heartbeat: each event handed to the sink resets last_event_ts; the
    # watchdog uses it to tell "silent because idle" (fine) from "silent because
    # the tap is dead" (respawn). events_seen gates the watchdog so it refuses to
    # fire until at least one event has been captured (tells "pynput works" from
    # "pynput never had permission").
    last_event_ts = time.time()
    events_seen = False

    def send(event: dict) -> None:
        nonlocal last_event_ts, events_seen
        last_event_ts = time.time()
        events_seen = True
        raw_send(event)

    supervisor = _ListenerSupervisor(send)
    supervisor.start()

    # Cross-thread "exit cleanly" flag, set from the wake handler (runs on the main
    # thread inside the pump) and the watchdog below. The main loop tests it each
    # cycle and breaks so the finally block can flush.
    exit_requested = threading.Event()

    def on_sleep() -> None:
        if flush is not None:
            try:
                flush()
            except Exception:
                pass

    def on_wake() -> None:
        exit_requested.set()

    _observer, pump = _install_wake_observer(on_wake, on_sleep)

    last_watchdog_check = time.time()

    try:
        while supervisor.is_alive() and not exit_requested.is_set():
            pump(0.2)

            now = time.time()
            if now - last_watchdog_check < WATCHDOG_CHECK_INTERVAL:
                continue
            last_watchdog_check = now

            if not events_seen:
                # No event ever observed -> no proof pynput works, so staying quiet
                # avoids crash-looping on the no-permission state.
                continue
            idle = _hid_idle_seconds()
            if idle is None:
                continue
            silent_for = now - last_event_ts
            if idle < WATCHDOG_ACTIVE_SECONDS and silent_for > WATCHDOG_SILENT_SECONDS:
                exit_requested.set()
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
        if flush is not None:
            flush()


if __name__ == "__main__":
    main()
