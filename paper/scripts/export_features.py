"""Export the content-free per-window feature table released with the paper.

    uv run python paper/scripts/export_features.py

Reads the local archive (output/events/) and writes data/keyspark_windows.csv,
one row per scored one-minute window:

  set        genuine | synthetic | injected
  label      0 = human, 1 = input automation
  kind       generator kind for synthetic windows (jiggler, typer, keep_awake,
             clicker); empty otherwise
  user       U1 = the genuine user, U2 = the development test user
  day        calendar day index, 1 = first genuine day; injected days may fall before it and carry values <= 0
  session    session id (300 s idle gap), unique within the file
  window     window index within its session, 0-based
  n_events   events in the window
  + the seven features of keyspark.ml.FEATURES

No key identity, typed text, pointer coordinate, or clock time leaves the
archive: only counts, dispersion statistics, and relative indices. The genuine
and synthetic rows reproduce the classification tables (session-wise splits,
ablation, cross-kind); the injected rows are the seeded demonstration days and
the test user, used only by the day-level policy table.
"""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (sets cwd to the repo root)
from keyspark import events_io, ml  # noqa: E402

OUT_PATH = "data/keyspark_windows.csv"


def main() -> None:
    all_events = events_io.load_events()
    all_events["day"] = pd.to_datetime(all_events["event_time"]).dt.date
    archive = all_events[all_events["day"] <= mt.FREEZE].copy()
    seeded = {dt.date.fromisoformat(d) for d in events_io.HUMAN_EXCLUDE_DAYS}
    is_injected = archive["day"].isin(seeded) | archive["user"].isin(
        events_io.HUMAN_EXCLUDE_USERS
    )

    data = mt.labeled_dataset()
    kinds = mt.kind_of_sessions(data)
    genuine = data[data["label"] == 0].copy()
    genuine["set"] = "genuine"
    synthetic = data[data["label"] == 1].copy()
    synthetic["set"] = "synthetic"
    synthetic["kind"] = synthetic["session"].map(kinds)

    injected = mt.windows_with_sessions(archive[is_injected], "I")
    injected["label"] = 1
    injected["set"] = "injected"

    out = pd.concat([genuine, synthetic, injected], ignore_index=True)

    # Replace dates and names with relative indices.
    first_day = archive.loc[~is_injected, "day"].min()
    real = out["set"] != "synthetic"
    out["day"] = pd.NA
    out.loc[real, "day"] = (
        pd.to_datetime(out.loc[real, "window_start"]).dt.date
        .map(lambda d: (d - first_day).days + 1)
    )
    out["day"] = out["day"].astype("Int64")
    users = {u: f"U{i + 1}" for i, u in enumerate(
        [u for u in archive["user"].unique() if u not in events_io.HUMAN_EXCLUDE_USERS]
        + sorted(events_io.HUMAN_EXCLUDE_USERS)
    )}
    out["user"] = out["user"].map(lambda u: users.get(u, "U_synthetic"))
    out.loc[~real, "user"] = ""
    # Row order is kept as produced by make_tables, so seeded models refit on this
    # file see the same rows in the same order and reproduce the paper exactly.
    out["window"] = out.groupby("session").cumcount()
    out["session"] = out["session"].astype("category").cat.codes.map(lambda c: f"S{c + 1:04d}")

    cols = ["set", "label", "kind", "user", "day", "session", "window",
            "n_events", *ml.FEATURES]
    out = out.reindex(columns=cols)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {len(out)} rows")
    print(out.groupby("set").size().to_string())


if __name__ == "__main__":
    main()
