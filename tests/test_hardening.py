"""Invariants for the content-free hardening transform."""

from __future__ import annotations

import pandas as pd

from keyspark import hardening as H
from tests._run import run


def test_classify_key_covers_the_roles():
    assert H.classify_key("a") == "letter"
    assert H.classify_key("Z") == "letter"
    assert H.classify_key("7") == "digit"
    assert H.classify_key(",") == "punct"
    assert H.classify_key('"') == "punct"
    assert H.classify_key(" ") == "space"
    assert H.classify_key("Key.space") == "space"
    assert H.classify_key("Key.backspace") == "correction"
    assert H.classify_key("Key.delete") == "correction"
    assert H.classify_key("Key.enter") == "enter"
    assert H.classify_key("Key.tab") == "tab"
    assert H.classify_key("Key.esc") == "nav"
    assert H.classify_key("Key.left") == "nav"
    assert H.classify_key("Key.cmd") == "modifier"
    assert H.classify_key("Key.shift_r") == "modifier"
    assert H.classify_key("Key.f15") == "function"
    assert H.classify_key(None) == "other"
    # every returned class is in the declared vocabulary
    for raw in ("a", "7", ",", " ", "Key.backspace", "Key.f1", "Key.media_play"):
        assert H.classify_key(raw) in H.KEY_CLASSES


def test_token_is_stable_within_a_minute_and_rotates_across_minutes():
    secret = H.new_secret()
    # minute buckets are floor(ts/60), aligned to the featurizer's floor("min").
    # 120.0 and 179.9 share bucket 2; 180.0 is bucket 3.
    assert H.key_token("e", 120.0, secret) == H.key_token("e", 179.9, secret)
    assert H.key_token("e", 120.0, secret) != H.key_token("e", 180.0, secret)
    # different keys, same minute -> different tokens (distinct-key counting works)
    assert H.key_token("e", 120.0, secret) != H.key_token("r", 120.0, secret)


def test_token_case_folds_so_shifted_release_pairs():
    secret = H.new_secret()
    # pynput reports Shift-down as 'A' and release as 'a'; they must tokenize alike
    assert H.key_token("A", 10.0, secret) == H.key_token("a", 10.0, secret)


def test_token_is_secret_dependent():
    assert H.key_token("e", 10.0, H.new_secret()) != H.key_token("e", 10.0, H.new_secret())


def test_harden_event_drops_raw_key_for_key_events():
    secret = H.new_secret()
    ev = {"type": "key_down", "key": "e", "user": "u", "ts": 10.0}
    out = H.harden_event(ev, secret)
    assert "key" not in out
    assert out["key_class"] == "letter"
    assert out["key_token"] == H.key_token("e", 10.0, secret)
    assert ev["key"] == "e"  # input not mutated


def test_harden_event_passes_through_non_key_events_unchanged():
    secret = H.new_secret()
    for ev in (
        {"type": "move", "x": 1, "y": 2, "user": "u", "ts": 10.0},
        {"type": "click", "x": 1, "y": 2, "button": "Button.left", "pressed": True, "user": "u", "ts": 10.0},
        {"type": "scroll", "x": 1, "y": 2, "dx": 0, "dy": -1, "user": "u", "ts": 10.0},
    ):
        out = H.harden_event(ev, secret)
        assert out == ev
        assert "key_class" not in out


def test_harden_frame_removes_key_column_and_marks_only_key_rows():
    df = pd.DataFrame([
        {"type": "key_down", "key": "e", "x": None, "y": None, "user": "u", "ts": 10.0},
        {"type": "move", "key": None, "x": 5, "y": 6, "user": "u", "ts": 10.1},
        {"type": "key_up", "key": "e", "x": None, "y": None, "user": "u", "ts": 10.2},
    ])
    out = H.harden_frame(df, secret=H.new_secret())
    assert "key" not in out.columns
    assert set(out.columns) >= {"key_class", "key_token"}
    key_rows = out[out["type"].isin(H.KEY_EVENT_TYPES)]
    move_rows = out[out["type"] == "move"]
    assert key_rows["key_class"].notna().all()
    assert move_rows["key_class"].isna().all()


if __name__ == "__main__":
    run(globals())
