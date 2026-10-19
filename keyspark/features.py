"""Per-window verification feature set (v2). Consumes CONTENT-FREE hardened events
(key_class + key_token, never a raw key) and emits one feature row per
(user, one-minute window). This is the shared featurizer used both for reprocessing
the historical archive (via harden_and_featurize) and, later, for the live
hardened stream, so both paths pair and count over the same post-transport events
and share one missingness profile.

Features are grouped into blocks so a window can abstain per modality and so the
keyboard-only / mouse-only ablation just selects blocks:

  keyboard_structure : what kind of keys, how varied (diversity, correction rate,
                       key-class mix). NaN when the window has < MIN_KEYSTROKES.
  keyboard_timing    : rhythm (inter-key CV, dwell stats, flight-time quantiles).
                       NaN when the window has < MIN_KEYSTROKES.
  mouse              : movement geometry and cadence (step, velocity, turns,
                       pauses, click dwell, scroll). NaN when < MIN_MOVES.
  cross_modal        : input mix and overall inter-event regularity. Always
                       present for a scoreable window.

Dwell is derived here by pairing key_down to key_up on key_token within the
window (LIFO per token, so autorepeat pairs to the most recent press); unpaired
and out-of-range dwells are dropped rather than guessed. Windows with fewer than
MIN_WINDOW_EVENTS events are dropped entirely (their shape statistics are
degenerate) - we abstain rather than emit noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keyspark import hardening
from keyspark.hardening import KEY_CLASSES

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
MIN_WINDOW_EVENTS = 5     # tune: min events for a window to be scored at all
MIN_KEYSTROKES = 5        # tune: min key_down events for the keyboard blocks
MIN_MOVES = 5             # tune: min move events for the mouse block
DWELL_MAX_MS = 2000.0     # tune: dwell above this is treated as an unpaired/held key and dropped
PAUSE_THRESHOLD_S = 0.5   # tune: gap between mouse events counted as a pause
CLICK_DWELL_MAX_MS = 2000.0  # tune: press-to-release above this is dropped

# Key-class mix fractions kept as features (a compact structural profile of what
# the user types). The rest of KEY_CLASSES are folded into the diversity/counts.
MIX_CLASSES = ("letter", "digit", "punct", "space", "nav", "modifier")

# Feature blocks, in a stable order. FEATURES (below) is their concatenation.
FEATURE_BLOCKS = {
    "keyboard_structure": (
        "key_diversity", "correction_rate",
        *(f"frac_{c}" for c in MIX_CLASSES),
    ),
    "keyboard_timing": (
        "ks_iei_cv",
        "dwell_mean_ms", "dwell_std_ms", "dwell_median_ms",
        "flight_p25_ms", "flight_p50_ms", "flight_p75_ms", "flight_p90_ms",
    ),
    "mouse": (
        "move_iei_cv", "step_mean", "step_cv",
        "velocity_mean", "velocity_std", "dir_change_rate",
        "pause_count", "pause_mean_s", "click_dwell_mean_ms", "scroll_rate",
    ),
    "cross_modal": ("iei_cv", "mouse_fraction"),
}
FEATURES = [f for block in FEATURE_BLOCKS.values() for f in block]

# The 7 original liveness features, so the feature-set contrast (counts-only vs
# liveness vs full v2) in the evaluation can select exactly the old set.
LIVENESS_FEATURES = [
    "key_diversity", "ks_iei_cv", "move_iei_cv",
    "iei_cv", "step_mean", "step_cv", "mouse_fraction",
]

_KEY_TYPES = ("key_down", "key_up")
_MOUSE_TYPES = ("move", "click", "scroll")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _cv(times: np.ndarray) -> float:
    """Coefficient of variation of the gaps between sorted timestamps. 0.0 with
    fewer than 3 events (too few gaps to have variation). Low = robotic.
    """
    if times.size < 3:
        return 0.0
    gaps = np.diff(np.sort(times))
    mean = gaps.mean()
    return float(gaps.std() / mean) if mean > 0 else 0.0


def _nan_block(block: str) -> dict:
    return {f: np.nan for f in FEATURE_BLOCKS[block]}


def _dwell_ms(g: pd.DataFrame) -> np.ndarray:
    """Dwell times (ms) by pairing key_down to key_up on key_token within the
    window, LIFO per token. Unpaired and out-of-range (<0 or > DWELL_MAX_MS) are
    dropped.
    """
    keys = g[g["type"].isin(_KEY_TYPES)]
    pending: dict[str, list[float]] = {}
    dwells: list[float] = []
    for typ, tok, t in zip(keys["type"].to_numpy(),
                           keys["key_token"].to_numpy(),
                           keys["t"].to_numpy()):
        if tok is None or (isinstance(tok, float) and np.isnan(tok)):
            continue
        if typ == "key_down":
            pending.setdefault(tok, []).append(t)
        else:  # key_up
            stack = pending.get(tok)
            if stack:
                dwell = (t - stack.pop()) * 1000.0
                if 0.0 <= dwell <= DWELL_MAX_MS:
                    dwells.append(dwell)
    return np.asarray(dwells, dtype=float)


def _keyboard_features(g: pd.DataFrame) -> dict:
    kd = g[g["type"] == "key_down"]
    n = len(kd)
    if n < MIN_KEYSTROKES:
        return {**_nan_block("keyboard_structure"), **_nan_block("keyboard_timing")}

    tokens = kd["key_token"].to_numpy()
    classes = kd["key_class"].to_numpy()
    diversity = len({t for t in tokens if t is not None}) / n
    correction_rate = float(np.count_nonzero(classes == "correction")) / n
    mix = {f"frac_{c}": float(np.count_nonzero(classes == c)) / n for c in MIX_CLASSES}

    kd_t = np.sort(kd["t"].to_numpy())
    flights = np.diff(kd_t) * 1000.0 if kd_t.size >= 2 else np.array([])
    fq = np.percentile(flights, [25, 50, 75, 90]) if flights.size else [np.nan] * 4

    dwells = _dwell_ms(g)
    if dwells.size:
        d_mean, d_std, d_med = float(dwells.mean()), float(dwells.std()), float(np.median(dwells))
    else:
        d_mean = d_std = d_med = np.nan

    return {
        "key_diversity": diversity,
        "correction_rate": correction_rate,
        **mix,
        "ks_iei_cv": _cv(kd["t"].to_numpy()),
        "dwell_mean_ms": d_mean, "dwell_std_ms": d_std, "dwell_median_ms": d_med,
        "flight_p25_ms": float(fq[0]), "flight_p50_ms": float(fq[1]),
        "flight_p75_ms": float(fq[2]), "flight_p90_ms": float(fq[3]),
    }


def _mouse_features(g: pd.DataFrame) -> dict:
    mv = g[g["type"] == "move"]
    if len(mv) < MIN_MOVES:
        return _nan_block("mouse")

    mv = mv.sort_values("t")
    t = mv["t"].to_numpy()
    xs = pd.to_numeric(mv["x"], errors="coerce").to_numpy()
    ys = pd.to_numeric(mv["y"], errors="coerce").to_numpy()
    ok = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys, t = xs[ok], ys[ok], t[ok]

    dx, dy = np.diff(xs), np.diff(ys)
    steps = np.sqrt(dx ** 2 + dy ** 2)
    dt = np.diff(t)
    step_mean = float(steps.mean()) if steps.size else 0.0
    step_cv = float(steps.std() / steps.mean()) if steps.size and steps.mean() > 0 else 0.0

    valid_dt = dt > 0
    vel = steps[valid_dt] / dt[valid_dt] if valid_dt.any() else np.array([])
    velocity_mean = float(vel.mean()) if vel.size else 0.0
    velocity_std = float(vel.std()) if vel.size else 0.0

    # Direction changes: consecutive move vectors whose dot product is negative
    # are turns sharper than 90 degrees. Rate over consecutive vector pairs.
    if dx.size >= 2:
        dots = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
        dir_change_rate = float(np.count_nonzero(dots < 0)) / dots.size
    else:
        dir_change_rate = 0.0

    # Pauses over ALL mouse events (move/click/scroll), bounded by the 1-min window.
    mouse = g[g["type"].isin(_MOUSE_TYPES)].sort_values("t")
    mt = mouse["t"].to_numpy()
    gaps = np.diff(mt) if mt.size >= 2 else np.array([])
    pause_gaps = gaps[gaps > PAUSE_THRESHOLD_S]
    pause_count = float(pause_gaps.size)
    pause_mean_s = float(pause_gaps.mean()) if pause_gaps.size else 0.0

    span = float(mt[-1] - mt[0]) if mt.size >= 2 else 0.0
    scroll_rate = float((g["type"] == "scroll").sum()) / span if span > 0 else 0.0

    return {
        "move_iei_cv": _cv(mv["t"].to_numpy()),
        "step_mean": step_mean, "step_cv": step_cv,
        "velocity_mean": velocity_mean, "velocity_std": velocity_std,
        "dir_change_rate": dir_change_rate,
        "pause_count": pause_count, "pause_mean_s": pause_mean_s,
        "click_dwell_mean_ms": _click_dwell_mean_ms(g),
        "scroll_rate": scroll_rate,
    }


def _click_dwell_mean_ms(g: pd.DataFrame) -> float:
    """Mean press-to-release time (ms) over clicks in the window, paired per
    button in order. NaN when no click pair exists.
    """
    clicks = g[g["type"] == "click"]
    if clicks.empty:
        return np.nan
    pending: dict[str, list[float]] = {}
    dwells: list[float] = []
    for btn, pressed, t in zip(clicks["button"].to_numpy(),
                               clicks["pressed"].to_numpy(),
                               clicks["t"].to_numpy()):
        if bool(pressed):
            pending.setdefault(btn, []).append(t)
        else:
            stack = pending.get(btn)
            if stack:
                d = (t - stack.pop()) * 1000.0
                if 0.0 <= d <= CLICK_DWELL_MAX_MS:
                    dwells.append(d)
    return float(np.mean(dwells)) if dwells else np.nan


def _cross_modal_features(g: pd.DataFrame) -> dict:
    n = len(g)
    is_mouse = g["type"].isin(_MOUSE_TYPES).to_numpy()
    return {
        "iei_cv": _cv(g["t"].to_numpy()),
        "mouse_fraction": float(is_mouse.sum() / n) if n else 0.0,
    }


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def _one_window(user, window_start, g: pd.DataFrame) -> dict:
    row = {
        "user": user,
        "window_start": window_start,
        "n_events": len(g),
        "keystrokes": int((g["type"] == "key_down").sum()),
        "moves": int((g["type"] == "move").sum()),
    }
    row.update(_keyboard_features(g))
    row.update(_mouse_features(g))
    row.update(_cross_modal_features(g))
    return row


def window_features(df: pd.DataFrame) -> pd.DataFrame:
    """One feature row per (user, one-minute window) over hardened events. Windows
    with fewer than MIN_WINDOW_EVENTS events are dropped. Keyboard / mouse feature
    blocks are NaN when their modality is below threshold in the window.
    """
    if "key" in df.columns:
        raise ValueError(
            "window_features received a raw 'key' column. Harden events first "
            "(features.harden_and_featurize or hardening.harden_frame); the "
            "featurizer must never see raw key content."
        )
    df = df.copy()
    if "event_time" not in df.columns:
        df["event_time"] = pd.to_datetime(df["ts"], unit="s")
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.dropna(subset=["event_time"]).sort_values(["user", "event_time"])
    if df.empty:
        return pd.DataFrame(columns=["user", "window_start", "n_events", *FEATURES])

    for c in ("key_class", "key_token", "x", "y", "button", "pressed"):
        if c not in df.columns:
            df[c] = pd.NA
    df["window_start"] = df["event_time"].dt.floor("min")
    df["t"] = df["event_time"].astype("int64") / 1e9

    rows = [
        _one_window(user, ws, g)
        for (user, ws), g in df.groupby(["user", "window_start"], sort=False)
    ]
    out = pd.DataFrame(rows)
    out = out[out["n_events"] >= MIN_WINDOW_EVENTS].reset_index(drop=True)
    return out


def harden_and_featurize(raw_df: pd.DataFrame, secret: bytes | None = None) -> pd.DataFrame:
    """Reprocess a RAW event frame (from the archive) into per-window features:
    harden first (drop raw keys), then featurize. The one entry point for turning
    the historical archive into verification features without raw content
    surviving into the feature table.
    """
    return window_features(hardening.harden_frame(raw_df, secret=secret))
