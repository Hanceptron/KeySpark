"""Synthetic input-automation (bot) event generator: the non-human class for the
liveness classifier. Emits raw events in the same JSON shape the capture agent
produces, so keyspark.ml featurizes them exactly like real events (no
train/serve skew).

Four bot kinds, each robotic in one modality:
  - jiggler    : small mouse nudge tracing a FIXED pattern (diagonal / horizontal
                 / octagon) in place, no keys
  - typer      : keystrokes over a small key set, no mouse
  - keep_awake : one function key (F15 / Scroll Lock) on a slow cadence, no mouse
  - clicker    : repeated clicks at one spot

Grounded in how real keep-active tools behave (research/jigglers.md): verified
open-source movers (arkane-systems Mouse Jiggler) trace a fixed relative pattern
and never randomize the path; keyboard keep-awake tools (Caffeine) emit a single
non-character key. The one thing they randomize is timing. So we keep the TIMING
jittered (per-session cadence, jitter, occasional pauses) plus a little
cross-modal contamination, so the task is not a trivial if-statement; the signal
left to learn is the residual GEOMETRY regularity (low step_cv, low key_diversity).
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
KINDS = ("jiggler", "typer", "keep_awake", "clicker")
DEFAULT_USER = "user-001"

# Per-session base inter-event interval (s) per kind, so different sessions have
# different cadences. The jiggler/keep_awake bands are slow on purpose (real
# movers nudge every few seconds, not sub-second), but fast enough that a
# one-minute window still holds several events.
_BASE_RANGE = {                      # tune: cadence band (s) per bot kind
    "jiggler": (1.0, 8.0),
    "typer": (0.10, 0.40),
    "keep_awake": (6.0, 12.0),
    "clicker": (0.3, 1.5),
}
_JITTER_RANGE = (0.20, 0.50)         # tune: per-session timing jitter fraction
CONTAMINATION = 0.08                 # tune: fraction of off-kind events (blurs mouse_fraction / diversity)
_ANCHOR = (800, 500)
_TYPER_KEYS = ("a", "s", "d", "f", "j", "k", "l", "e")  # small, low-diversity set

# Deterministic relative mouse-move patterns, mirroring arkane-systems Mouse
# Jiggler's JigglePatterns: back-and-forth pairs and a closed octagon, walked in
# order in place, so step size is near-constant (low step_cv). A per-session
# multiplier scales the deltas (the tool's configurable "distance").
_MOVE_PATTERNS = {
    "normal": ((4, 4), (-4, -4)),                     # diagonal nudge
    "linear": ((4, 0), (-4, 0)),                      # horizontal nudge
    "circle": ((3, 2), (2, 3), (-2, 3), (-3, 2),
               (-3, -2), (-2, -3), (2, -3), (3, -2)),  # closed octagon
}
_JIGGLE_PATTERNS = ("normal", "linear", "circle")
# Single non-character keep-awake keys (F15 is the classic Caffeine key). One key
# per session keeps key_diversity low - the keyboard analogue of the rigid mouse pattern.
_KEEPAWAKE_KEYS = ("f15", "f13", "f14", "scroll_lock")
# Release delay (s) after each press when key_up events are requested. The
# capture agent emits a key_up for every key_down; the default stream does not.
_KEY_UP_DELAY = (0.05, 0.15)         # tune: generated key dwell band


# --------------------------------------------------------------------------
# Event generation
# --------------------------------------------------------------------------
def _next_event(kind: str, state: dict, user: str, ts: float, rng: random.Random) -> dict:
    """One event for bot ``kind`` at time ``ts``. ``state`` carries the mouse
    position plus the per-session jiggle pattern / keep-awake key, so the jiggler
    traces a fixed pattern in place. Occasionally emits an off-kind action.
    """
    kk = kind
    if rng.random() < CONTAMINATION:
        kk = rng.choice([k for k in KINDS if k != kind])
    if kk == "jiggler":
        pattern = _MOVE_PATTERNS[state.setdefault("pattern", "normal")]
        i = state.get("i", 0)
        dx, dy = pattern[i % len(pattern)]
        dist = state.setdefault("dist", 1)
        state["i"] = i + 1
        state["x"] += dx * dist
        state["y"] += dy * dist
        return {"type": "move", "x": state["x"], "y": state["y"],
                "user": user, "ts": round(ts, 3)}
    if kk == "typer":
        return {"type": "key_down", "key": rng.choice(_TYPER_KEYS),
                "user": user, "ts": round(ts, 3)}
    if kk == "keep_awake":
        return {"type": "key_down", "key": state.setdefault("awake_key", "f15"),
                "user": user, "ts": round(ts, 3)}
    return {"type": "click", "x": state["x"], "y": state["y"],
            "button": "Button.left", "pressed": True, "user": user, "ts": round(ts, 3)}


def synthetic_events(kind: str, count: int, user: str = DEFAULT_USER,
                     start_ts: float = 0.0, seed: int = 0,
                     base: float | None = None, jitter: float = 0.35) -> list[dict]:
    """``count`` event dicts for one bot ``kind`` starting at ``start_ts`` (epoch
    seconds), advancing by ``base`` seconds with +/- ``jitter``.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown bot kind {kind!r}; choose from {KINDS}")
    rng = random.Random(seed)
    if base is None:
        base = sum(_BASE_RANGE[kind]) / 2
    ts = float(start_ts)
    # Per-session choices (fixed for the whole session, varied across sessions):
    # the jiggle pattern, its distance multiplier, and the single keep-awake key.
    state = {"x": _ANCHOR[0], "y": _ANCHOR[1], "i": 0,
             "pattern": rng.choice(_JIGGLE_PATTERNS),
             "dist": rng.randint(1, 3),
             "awake_key": rng.choice(_KEEPAWAKE_KEYS)}
    out = []
    for _ in range(count):
        step = base * (1.0 + rng.uniform(-jitter, jitter))
        if rng.random() < 0.10:
            step += base * rng.uniform(3.0, 12.0)  # occasional human-like pause/gap
        ts += step
        out.append(_next_event(kind, state, user, ts, rng))
    return out


def with_key_ups(events: list[dict], seed: int = 0) -> list[dict]:
    """``events`` plus a ``key_up`` 50 to 150 ms after every ``key_down``, on the
    same key. Uses its own RNG, so the original events are left exactly as they
    were and only the releases are added.
    """
    rng = random.Random(f"key_up:{seed}")
    out = []
    for e in events:
        out.append(e)
        if e["type"] == "key_down":
            out.append({"type": "key_up", "key": e["key"], "user": e["user"],
                        "ts": round(e["ts"] + rng.uniform(*_KEY_UP_DELAY), 3)})
    return out


def synthetic_event_frame(sessions_per_kind: int = 4, minutes_per_session: int = 30,
                          user: str = DEFAULT_USER, start_ts: float = 1.0e9,
                          seed: int = 0, key_up: bool = False):
    """A DataFrame of synthetic bot events: ``sessions_per_kind`` sessions for each
    kind, each ~``minutes_per_session`` minutes, with per-session randomized
    cadence/jitter/pattern/key so the non-human class spans the feature space.
    ``key_up`` adds a release after every press (see with_key_ups); the default
    stream is unchanged.
    """
    import pandas as pd

    rng = random.Random(seed)
    rows: list[dict] = []
    ts = float(start_ts)
    s = 0
    for kind in KINDS:
        for _ in range(sessions_per_kind):
            base = rng.uniform(*_BASE_RANGE[kind])
            jitter = rng.uniform(*_JITTER_RANGE)
            count = int(minutes_per_session * 60 / base)
            evs = synthetic_events(kind, count, user=user, start_ts=ts,
                                   seed=seed + s, base=base, jitter=jitter)
            rows.extend(with_key_ups(evs, seed=seed + s) if key_up else evs)
            ts = evs[-1]["ts"] + 3600.0  # 1h gap so sessions stay distinct
            s += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Real tools at their default settings
# --------------------------------------------------------------------------
# Kept apart from KINDS, so the training class above is unchanged. Each models
# what the capture agent would record from the tool left at its defaults:
#   jiggler_default     arkane-systems Mouse Jiggler (Windows): one Normal-pattern
#                       nudge, distance 1, every 60 s.
#   jiggler_random      the same with its random timer on: each interval a whole
#                       number of seconds drawn uniformly from 1 to 60
#                       (Random.Shared.Next(1, JigglePeriod + 1) in MainForm.cs).
#   keep_awake_default  Zhorn Caffeine (Windows): one F15 key_up every 59 s.
#   mac_jiggler_default Stick Software Jiggler (macOS): after 300 s idle, 35 pointer
#                       moves scheduled 10 ms apart, each a random offset of up to
#                       14 px per axis, thinned by the agent's 20 ms move throttle.
DEFAULT_TOOLS = ("jiggler_default", "jiggler_random", "keep_awake_default",
                 "mac_jiggler_default")
_TIMER_JITTER = 0.016                # tune: OS timer granularity (s) added to each tick
_MAC_BURST = (35, 0.010, 14)         # moves per jiggle, spacing (s), max offset (px)
_MOVE_THROTTLE = 0.020               # keyspark.agent.MOVE_MIN_INTERVAL


def default_tool_events(tool: str, minutes: int = 60, user: str = DEFAULT_USER,
                        start_ts: float = 0.0, seed: int = 0) -> list[dict]:
    """``minutes`` of events from one real tool at its default settings, starting
    at a random phase after ``start_ts``.
    """
    if tool not in DEFAULT_TOOLS:
        raise ValueError(f"unknown tool {tool!r}; choose from {DEFAULT_TOOLS}")
    rng = random.Random(f"{tool}:{seed}")
    period = {"jiggler_default": 60.0, "jiggler_random": 60.0,
              "keep_awake_default": 59.0, "mac_jiggler_default": 300.0}[tool]
    end = start_ts + minutes * 60
    ts = start_ts + rng.uniform(0, period)
    x, y = _ANCHOR
    i = 0
    out: list[dict] = []
    while ts < end:
        if tool == "keep_awake_default":
            out.append({"type": "key_up", "key": "Key.f15", "user": user, "ts": round(ts, 3)})
        elif tool == "mac_jiggler_default":
            # Bounded random walk: each move is the current position plus up to
            # ``reach`` px per axis, redrawn until within 2 * reach of the anchor.
            n, spacing, reach = _MAC_BURST
            last = None
            for k in range(n):
                t = ts + k * spacing + rng.uniform(0, 0.002)
                nx = x + rng.randint(-reach, reach)
                while abs(nx - _ANCHOR[0]) > 2 * reach:
                    nx = x + rng.randint(-reach, reach)
                ny = y + rng.randint(-reach, reach)
                while abs(ny - _ANCHOR[1]) > 2 * reach:
                    ny = y + rng.randint(-reach, reach)
                x, y = nx, ny
                if last is None or t - last >= _MOVE_THROTTLE:
                    out.append({"type": "move", "x": x, "y": y, "user": user,
                                "ts": round(t, 3)})
                    last = t
        else:
            dx, dy = _MOVE_PATTERNS["normal"][i % 2]
            x, y = x + dx, y + dy
            out.append({"type": "move", "x": x, "y": y, "user": user, "ts": round(ts, 3)})
        i += 1
        step = rng.randint(1, 60) if tool == "jiggler_random" else period
        # Jiggler polls its idle condition on a 0.25 s timer; the Windows tools
        # fire on a timer with about 16 ms granularity.
        ts += step + rng.uniform(0, 0.25 if tool == "mac_jiggler_default" else _TIMER_JITTER)
    return out


def default_tool_frame(tools: tuple[str, ...] = DEFAULT_TOOLS, sessions_per_tool: int = 4,
                       minutes_per_session: int = 60, user: str = DEFAULT_USER,
                       start_ts: float = 1.1e9, seed: int = 0):
    """A DataFrame of ``sessions_per_tool`` default-setting sessions per tool, in
    ``tools`` order, separated by one-hour gaps, with a ``tool`` column.
    """
    import pandas as pd

    rows: list[dict] = []
    ts = float(start_ts)
    s = 0
    for tool in tools:
        for _ in range(sessions_per_tool):
            evs = default_tool_events(tool, minutes_per_session, user=user,
                                      start_ts=ts, seed=seed + s)
            rows.extend({**e, "tool": tool} for e in evs)
            ts += minutes_per_session * 60 + 3600.0  # 1h gap so sessions stay distinct
            s += 1
    return pd.DataFrame(rows)
