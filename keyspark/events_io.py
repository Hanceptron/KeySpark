"""Shared read side for the raw event archive: one tolerant loader and one set of
"not genuine human" exclusions, used by both the liveness model (keyspark.ml) and
the verification self-model (keyspark.verify) so they can never disagree about
what counts as real user data.

The archive is written by Spark in append mode and accumulates thousands of part
files; at least one can be zero-byte or truncated (a crash between file create and
flush), which makes a naive pd.read_parquet raise. load_events skips such files
and reports how many were dropped.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EVENTS_PATH = "output/events"

# Deliberately-fed automation that must NOT count as genuine human data. The
# bot-test user and these seeded demo days were injected to exercise the liveness
# classifier; they poison any self-model trained on "the user". Single source of
# truth for keyspark.ml and keyspark.verify.
HUMAN_EXCLUDE_USERS = frozenset({"bot-test"})   # tune: users excluded from genuine data
HUMAN_EXCLUDE_DAYS = frozenset({               # tune: seeded demo days excluded from genuine data
    "2026-05-11", "2026-05-13", "2026-05-15",
})


def local_day(ts) -> pd.Series:
    """Map UTC-naive timestamps (as Spark writes and pandas reads them) to
    host-local calendar-day keys (YYYY-MM-DD), matching the browser's localDayKey
    and the batch job's date_format.
    """
    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    return (
        pd.to_datetime(ts)
        .dt.tz_localize("UTC")
        .dt.tz_convert(local_tz)
        .dt.strftime("%Y-%m-%d")
    )


def load_events(path: str = EVENTS_PATH, *, return_skipped: bool = False):
    """Read the whole raw event archive as one pandas frame, skipping unreadable
    or zero-byte part files instead of crashing on them. With return_skipped=True
    returns (frame, skipped_count).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {path}. Start the pipeline so the streaming job archives "
            "events before loading."
        )
    files = sorted(glob.glob(f"{path}/part-*.parquet"))
    frames = []
    skipped = 0
    for f in files:
        if os.path.getsize(f) == 0:
            skipped += 1
            continue
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            skipped += 1
    if not frames:
        # Fall back to letting pyarrow read the directory (older layouts without
        # part-*.parquet names); if that also fails, surface the real error.
        frame = pd.read_parquet(path)
    else:
        frame = pd.concat(frames, ignore_index=True)
    return (frame, skipped) if return_skipped else frame


def exclude_nonhuman(
    df: pd.DataFrame,
    users: frozenset[str] = HUMAN_EXCLUDE_USERS,
    days: frozenset[str] = HUMAN_EXCLUDE_DAYS,
) -> pd.DataFrame:
    """Drop rows from excluded users and seeded demo days, leaving only genuine
    human activity. Uses event_time when present, else derives it from epoch ts.
    """
    out = df
    if users:
        out = out[~out["user"].isin(users)]
    if days:
        et = out["event_time"] if "event_time" in out.columns else pd.to_datetime(out["ts"], unit="s")
        out = out[~local_day(et).isin(days).to_numpy()]
    return out.reset_index(drop=True)
