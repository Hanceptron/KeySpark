"""Generator-geometry control:

    .venv/bin/python paper/scripts/geometry_control.py

Every generated pointer move in the released training class repeats a step from
a fixed displacement table, so step_cv is zero in most synthetic windows. This
script regenerates the 16 synthetic sessions with the same seeds, so every event
time, kind, key, and click is unchanged, but draws each pointer move as a random
step (sign and size uniform, 1 to 10 px per axis) from a separate generator. The genuine
windows are the released ones (data/keyspark_windows.csv); only the
automation class changes. The kind-stratified evaluation and the leave-one-kind-
out operating points are rerun with the released code (reanalysis_from_csv.py).

Check first: the unmodified generator, featurised the same way, must reproduce
the released synthetic windows exactly; their session ids are then carried over
to the variant, so every partition matches the paper's.

Writes paper/tables/geometry.json and paper/tables/geometry.tex.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tables as mt  # noqa: E402  (chdirs to the repository root)

KEYSPARK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(KEYSPARK, "paper", "scripts"))
import reanalysis_from_csv as ra  # noqa: E402

from keyspark import botgen, ml  # noqa: E402
import baselines_extra as bx  # noqa: E402

CSV = os.path.join(KEYSPARK, "data", "keyspark_windows.csv")
STEP = 10  # tune: maximum random step per axis (px)
_original_next_event = botgen._next_event


def random_step_next_event(kind, state, user, ts, rng):
    """The generator's own event, with every pointer move replaced by a random step.

    The steps come from a separate generator seeded by the session's first event
    time, so the session's own random sequence, and with it every event time,
    kind, key, and click, is exactly that of the released generator.
    """
    if "_srng" not in state:
        state["_srng"] = random.Random(f"step:{ts:.3f}")
        state["_px"], state["_py"] = state["x"], state["y"]
    ev = _original_next_event(kind, state, user, ts, rng)
    if ev["type"] == "move":
        srng = state["_srng"]
        x = state["_px"] + srng.choice((-1, 1)) * srng.randint(1, STEP)
        y = state["_py"] + srng.choice((-1, 1)) * srng.randint(1, STEP)
        state["x"] = ev["x"] = x
        state["y"] = ev["y"] = y
    state["_px"], state["_py"] = state["x"], state["y"]
    return ev


def synthetic_windows() -> pd.DataFrame:
    events = botgen.synthetic_event_frame()
    w = mt.windows_with_sessions(events, "B")
    ev = ml._prepare(events.copy())
    ev["window_start"] = ev["event_time"].dt.floor("min")
    moves = ev[ev["type"] == "move"].groupby(["user", "window_start"]).size().rename("moves")
    w = w.merge(moves.reset_index(), on=["user", "window_start"], how="left")
    w["moves"] = w["moves"].fillna(0).astype(int)
    w = w[w["n_events"] >= ml.MIN_WINDOW_EVENTS] if "n_events" in w else w
    w["kind"] = w["session"].map(mt.kind_of_sessions(w.assign(label=1)))
    w["label"] = 1
    return w.sort_values(["session", "window_start"]).reset_index(drop=True)


def evaluate(data: pd.DataFrame) -> dict:
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values
    kind = data["kind"].fillna("").values
    session_rows = data.groupby("session").indices
    strat = ra.kind_stratified(X, y, g, kind, session_rows)
    _, te0 = next(ra.GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=ra.SEED).split(X, y, groups=g))
    te_mask = np.zeros(len(y), dtype=bool)
    te_mask[te0] = True
    tt = ra.threshold_transfer(X, y, g, kind, te_mask)
    syn = data[data["label"] == 1]
    zero = syn["step_cv"] == 0
    # The two untrained rules of Table tab:baselines, on the same windows and partitions.
    rules = {name: bx.rule_row(X, y, g, kind, flag.astype(int)) for name, flag in
             (("step_cv_zero", data["step_cv"].values == 0), ("no_movement", data["step_mean"].values == 0))}
    return {
        "rules": {name: {"kind_stratified_f1": float(r["strat"][0]), "loko_mean_f1": r["loko"]}
                  for name, r in rules.items()},
        "step_cv_zero": int(zero.sum()),
        "zero_no_movement": int((zero & (syn["moves"] < 2)).sum()) if "moves" in syn else None,
        "zero_single_step": int((zero & (syn["moves"] == 2)).sum()) if "moves" in syn else None,
        "zero_constant_steps": int((zero & (syn["moves"] > 2)).sum()) if "moves" in syn else None,
        "synthetic_windows": int(len(syn)),
        "step_cv_zero_share": float((syn["step_cv"] == 0).mean()),
        "step_cv_median": float(syn["step_cv"].median()),
        "pooled_f1": strat["pooled_f1"], "pooled_f1_ci95": strat["pooled_f1_ci95"],
        "pooled_auc": strat["pooled_auc"],
        "recall": {k: v["value"] for k, v in strat["recall"].items()},
        "loko_fixed_mean_f1": tt["fixed_0.5"]["mean_f1"],
        "loko_budget1_mean_f1": tt["budget_0.01"]["mean_f1"],
        "loko_fixed_f1": {k: v["f1"] for k, v in tt["fixed_0.5"]["per_kind"].items()},
        "loko_budget1_f1": {k: v["f1"] for k, v in tt["budget_0.01"]["per_kind"].items()},
        "loko_auc": tt["forest_auc"],
    }


def main() -> None:
    df = pd.read_csv(CSV)
    genuine = df[df["set"] == "genuine"].copy()
    released = df[df["set"] == "synthetic"].copy()

    # 1. The unmodified generator must reproduce the released synthetic windows.
    orig = synthetic_windows()
    def signature(frame):
        return sorted(tuple(np.round(r, 9)) for r in frame[["n_events", *ml.FEATURES]].values)
    assert len(orig) == len(released), (len(orig), len(released))
    assert signature(orig) == signature(released), "regenerated windows differ from the released ones"
    # Carry the released session ids over, matching sessions by their window sequence.
    rel_seq = {s: tuple(released[released["session"] == s].sort_values("window")["n_events"]) for s in released["session"].unique()}
    seq_to_id = {v: k for k, v in rel_seq.items()}
    id_of = {s: seq_to_id[tuple(orig[orig["session"] == s]["n_events"])] for s in orig["session"].unique()}

    # 2. The random-step variant: same seeds and settings, random pointer steps.
    botgen._next_event = random_step_next_event
    try:
        var = synthetic_windows()
    finally:
        botgen._next_event = _original_next_event
    var["session"] = var["session"].map(id_of)

    cols = ["label", "kind", "session", *ml.FEATURES]
    mv = orig.assign(session=orig["session"].map(id_of))
    mv["window"] = mv.groupby("session").cumcount()
    released = released.merge(mv[["session", "window", "n_events", "moves"]], on=["session", "window", "n_events"], how="left")
    assert released["moves"].notna().all()
    base = pd.concat([genuine[cols], released[cols + ["moves"]]], ignore_index=True)
    variant = pd.concat([genuine[cols], var[cols + ["moves"]]], ignore_index=True)
    out = {"step_px_max": STEP, "released": evaluate(base), "random_step": evaluate(variant)}
    # The released-class rules must reproduce Table tab:baselines (0.737 and 0.379).
    tex = open(os.path.join(mt.OUT_DIR, "baselines_sensors.tex")).read()
    for name, label in (("step_cv_zero", r"\texttt{step\_cv} $= 0$ & Rule"), ("no_movement", "No pointer movement & Rule")):
        row = next(line for line in tex.splitlines() if line.startswith(label))
        assert row.rstrip(" \\").endswith(f"{out['released']['rules'][name]['loko_mean_f1']:.3f}"), (name, row)
    for name in ("released", "random_step"):
        r = out[name]
        print(f"{name:12s} windows {r['synthetic_windows']}  step_cv=0 {r['step_cv_zero_share']:.3f}  "
              f"pooled F1 {r['pooled_f1']:.3f} {r['pooled_f1_ci95']}  LOKO 0.5 {r['loko_fixed_mean_f1']:.3f}  "
              f"LOKO 1% {r['loko_budget1_mean_f1']:.3f}  AUC {r['loko_auc']}")
    with open(os.path.join(mt.OUT_DIR, "geometry.json"), "w") as fh:
        json.dump(ra.tidy(out), fh, indent=1)
    write_table(out)


def write_table(out: dict) -> None:
    rows = []
    for name, label in (("released", "Released generator (fixed steps)"),
                        ("random_step", f"Random steps (1 to {out['step_px_max']}\\,px per axis)")):
        r = out[name]
        lo, hi = r["pooled_f1_ci95"]
        rows.append(f"{label} & {r['synthetic_windows']} & {100 * r['step_cv_zero_share']:.1f} & "
                    f"{r['pooled_f1']:.3f} ({lo:.3f}--{hi:.3f}) & {r['loko_fixed_mean_f1']:.3f} & "
                    f"{r['loko_budget1_mean_f1']:.3f} & {r['rules']['step_cv_zero']['loko_mean_f1']:.3f} & "
                    f"{r['rules']['no_movement']['loko_mean_f1']:.3f} & {min(r['loko_auc'].values()):.3f}--{max(r['loko_auc'].values()):.3f} \\\\")
    lines = [
        "% generated by paper/scripts/geometry_control.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Detection against a generator whose pointer moves take random steps.}",
        r"\label{tab:geometry}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \texttt{step\_cv} & Kind-stratified & \multicolumn{2}{c}{Forest LOKO mean $F_1$} & "
        r"\multicolumn{2}{c}{Rule LOKO mean $F_1$} & LOKO \\",
        r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"Automation class & Windows & $= 0$ (\%) & $F_1$ (95\% interval) & 0.5 & 1\% budget & "
        r"\texttt{step\_cv} $= 0$ & No movement & AUC \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt] \footnotesize Genuine windows are the released ones; only the automation class changes. "
        r"The random-step generator keeps every event time, kind, key, and click of the released one and "
        r"draws every pointer move as a random step from a separate seeded generator. Designs as in "
        r"Tables~\ref{tab:stratified} and~\ref{tab:transfer}. The rules are those of Table~\ref{tab:baselines}, which "
        r"learn nothing: \texttt{step\_cv} $= 0$ flags every window with zero step variation, and the no-movement "
        r"rule every window without measurable pointer movement.",
        r"\end{table}",
    ]
    with open(os.path.join(mt.OUT_DIR, "geometry.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
