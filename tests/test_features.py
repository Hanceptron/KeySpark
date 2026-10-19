"""Invariants for the v2 per-window featurizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from keyspark import features as F
from keyspark import hardening
from tests._run import approx, run


def _key(kind, token, cls, ts):
    return {"type": kind, "key_class": cls, "key_token": token,
            "x": None, "y": None, "button": None, "pressed": None,
            "user": "u", "ts": ts}


def _move(x, y, ts):
    return {"type": "move", "key_class": None, "key_token": None,
            "x": x, "y": y, "button": None, "pressed": None, "user": "u", "ts": ts}


def test_raw_key_column_is_rejected():
    df = pd.DataFrame([{"type": "key_down", "key": "e", "user": "u", "ts": 10.0}])
    try:
        F.window_features(df)
    except ValueError as exc:
        assert "raw" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on a raw key column")


def test_dwell_pairs_down_to_up_on_token():
    # 6 keystrokes in one minute, each held ~100 ms; one unpaired up should not crash.
    base = 120.0
    evs = []
    for i in range(6):
        tok = f"tok{i}"
        evs.append(_key("key_down", tok, "letter", base + i * 0.2))
        evs.append(_key("key_up", tok, "letter", base + i * 0.2 + 0.1))
    evs.append(_key("key_up", "orphan", "letter", base + 5.0))  # unpaired up
    out = F.window_features(pd.DataFrame(evs))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["keystrokes"] == 6
    assert approx(row["dwell_mean_ms"], 100.0, 15.0)
    assert 0.0 < row["key_diversity"] <= 1.0


def test_keyboard_block_abstains_below_threshold():
    # 3 key_down (< MIN_KEYSTROKES) but enough total events to be scoreable.
    base = 120.0
    evs = [_key("key_down", f"t{i}", "letter", base + i * 0.2) for i in range(3)]
    evs += [_move(10 + i, 10 + i, base + 1.0 + i * 0.05) for i in range(6)]
    out = F.window_features(pd.DataFrame(evs))
    assert len(out) == 1
    row = out.iloc[0]
    # keyboard features NaN, mouse features present
    assert np.isnan(row["key_diversity"])
    assert np.isnan(row["dwell_mean_ms"])
    assert not np.isnan(row["step_mean"])


def test_mouse_block_abstains_below_threshold():
    base = 120.0
    evs = [_key("key_down", f"t{i}", "letter", base + i * 0.1) for i in range(8)]
    evs += [_move(10, 10, base + 2.0), _move(11, 11, base + 2.1)]  # only 2 moves
    out = F.window_features(pd.DataFrame(evs))
    row = out.iloc[0]
    assert np.isnan(row["step_mean"])
    assert not np.isnan(row["key_diversity"])


def test_window_below_min_events_is_dropped():
    evs = [_key("key_down", "t0", "letter", 120.0), _key("key_down", "t1", "letter", 120.1)]
    out = F.window_features(pd.DataFrame(evs))
    assert out.empty


def test_harden_and_featurize_from_raw_produces_features():
    # raw archive-shaped events (with a real 'key' column) go through hardening.
    base = 120.0
    raw = []
    for i, ch in enumerate("hello world"):
        raw.append({"type": "key_down", "key": ch, "x": None, "y": None,
                    "button": None, "pressed": None, "dx": None, "dy": None,
                    "user": "u", "ts": base + i * 0.15})
        raw.append({"type": "key_up", "key": ch, "x": None, "y": None,
                    "button": None, "pressed": None, "dx": None, "dy": None,
                    "user": "u", "ts": base + i * 0.15 + 0.08})
    out = F.harden_and_featurize(pd.DataFrame(raw), secret=hardening.new_secret())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["keystrokes"] == 11
    # 'l' repeats and 'o' repeats, space present -> diversity < 1
    assert row["key_diversity"] < 1.0
    assert approx(row["frac_space"], 1 / 11, 1e-6)
    assert not np.isnan(row["dwell_mean_ms"])


def test_all_declared_features_present_in_output():
    base = 120.0
    evs = [_key("key_down", f"t{i}", "letter", base + i * 0.1) for i in range(8)]
    evs += [_move(10 + i, 12 + 2 * i, base + 2.0 + i * 0.05) for i in range(8)]
    out = F.window_features(pd.DataFrame(evs))
    for feat in F.FEATURES:
        assert feat in out.columns, feat


if __name__ == "__main__":
    run(globals())
