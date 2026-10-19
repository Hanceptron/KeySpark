"""Same-data comparison baselines.

On the paper's one-minute windows and session grouping, compares the seven
released features against the windowed keystroke and mouse-dynamics feature set
of Wang et al. (Sensors 2022, 22, 6627; wang2022siurua), a one-class
IsolationForest trained on genuine sessions only, and two rule baselines, under
three designs: the paper's ten GroupShuffleSplit splits, the kind-stratified
folds of typematched.py, and leave-one-kind-out.

Wang et al. feature set as implemented here (47 values per window):
  mouse, 43 values, their Section 3.1.2:
    msd_1..8  mean action speed by travelled distance, 100 px bins up to 800 px
    mda_1..8  mean action speed by movement direction, eight 45-degree sectors
    ata_*     mean action speed by action type (move, drag_drop, point_click)
    tdh_1..8  share of actions per travelled-distance bin
    mdh_1..8  share of actions per direction sector
    ath_*     share of actions per action type
    mth_1..5  share of actions per elapsed-time bin, 200 ms bins up to 1000 ms
  keystroke, 4 values: mean and standard deviation of hold time (key_down to
    key_up of the same key) and of down-down latency. Wang et al. use per-key
    and per-key-pair vectors (48,510 values for a 110-key keyboard), which need
    raw key identity; the summary keeps the same timing information content-free.
  Missing values are 0, as in their paper.

Their paper does not say how pointer actions are segmented, so this rule is
ours: pointer events (moves and button presses or releases) form one action
while consecutive events are at most ACTION_GAP_S apart, inside one window; a
release ends the action. An action is a drag-drop if a move occurs while a button
is held and a release follows, a point-click if it contains a press otherwise,
and a mouse-move if it has no press. Speed is travelled path length over elapsed
time; direction is the start-to-end vector. Scroll events are not used.

    .venv/bin/python paper/scripts/baselines.py

Writes paper/tables/baselines.tex and paper/tables/baselines.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (also puts the repo root on sys.path and chdirs there)

from keyspark import botgen, ml  # noqa: E402

SEED = mt.SEED
N_REPEATS = 10
ACTION_GAP_S = 0.5      # pointer events further apart than this start a new action
GAP_SENSITIVITY_S = 5.0  # alternative gap, reported in the JSON only
ACTIONS = ("move", "drag_drop", "point_click")
OUT_TEX = os.path.join(mt.OUT_DIR, "baselines.tex")
OUT_JSON = os.path.join(mt.OUT_DIR, "baselines.json")

WANG_MOUSE = ([f"msd_{i}" for i in range(1, 9)] + [f"mda_{i}" for i in range(1, 9)]
              + [f"ata_{a}" for a in ACTIONS] + [f"tdh_{i}" for i in range(1, 9)]
              + [f"mdh_{i}" for i in range(1, 9)] + [f"ath_{a}" for a in ACTIONS]
              + [f"mth_{i}" for i in range(1, 6)])
WANG_KEYS = ["hold_mean", "hold_std", "dd_mean", "dd_std"]
WANG = WANG_MOUSE + WANG_KEYS


# ---------------------------------------------------------------------------
# Wang et al. features
# ---------------------------------------------------------------------------
def _events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_time" not in out.columns:
        out["event_time"] = pd.to_datetime(out["ts"], unit="s")
    out["event_time"] = pd.to_datetime(out["event_time"])
    for c in ("key", "x", "y", "pressed"):
        if c not in out.columns:
            out[c] = pd.NA
    out = out[["user", "event_time", "type", "key", "x", "y", "pressed"]]
    out = out.dropna(subset=["event_time"]).sort_values(["user", "event_time"], kind="mergesort")
    out = out.reset_index(drop=True)
    out["window_start"] = out["event_time"].dt.floor("min")
    out["t"] = out["event_time"].astype("int64") / 1e9
    return out


def _binned(acts: pd.DataFrame, col: str, n: int, prefix: str, value: str | None) -> pd.DataFrame:
    """Per-window share of actions (value None) or mean ``value`` per bin."""
    keys = ["user", "window_start"]
    a = acts.dropna(subset=[col]).astype({col: int})
    if value is None:
        out = a.groupby(keys + [col]).size().unstack(col)
        out = out.div(acts.groupby(keys).size(), axis=0)
    else:
        out = a.dropna(subset=[value]).groupby(keys + [col])[value].mean().unstack(col)
    out = out.reindex(columns=range(n))
    out.columns = [f"{prefix}_{i + 1}" for i in range(n)]
    return out


def wang_features(df: pd.DataFrame, gap: float = ACTION_GAP_S) -> pd.DataFrame:
    """The 47 Wang et al. values for every (user, minute) with pointer or key events.
    Isolated pointer events (a jiggler nudge every few seconds) form one-event
    actions with no path, so they count only in the action-type shares."""
    ev = _events(df)
    keys = ["user", "window_start"]

    # -- pointer actions ------------------------------------------------------
    p = ev[ev["type"].isin(["move", "click"])].copy()
    p["is_move"] = p["type"] == "move"
    p["is_press"] = (p["type"] == "click") & (p["pressed"] == True)  # noqa: E712
    p["is_release"] = (p["type"] == "click") & (p["pressed"] == False)  # noqa: E712
    p["x"] = pd.to_numeric(p["x"], errors="coerce")
    p["y"] = pd.to_numeric(p["y"], errors="coerce")
    same = (p["user"] == p["user"].shift()) & (p["window_start"] == p["window_start"].shift())
    new = ~same | (p["t"].diff() > gap) | p["is_release"].shift(fill_value=False)
    p["action"] = new.cumsum()
    step = np.hypot(p["x"].diff(), p["y"].diff()).fillna(0.0)
    p["step"] = np.where(new, 0.0, step)
    button = pd.Series(np.where(p["is_press"], 1.0, np.where(p["is_release"], 0.0, np.nan)),
                       index=p.index)
    held = button.groupby(p["action"]).ffill().fillna(0.0)
    p["t_held_move"] = np.where(p["is_move"] & (held == 1.0), p["t"], np.inf)
    p["t_release"] = np.where(p["is_release"], p["t"], -np.inf)
    acts = p.groupby("action").agg(
        user=("user", "first"), window_start=("window_start", "first"),
        t0=("t", "min"), t1=("t", "max"), path=("step", "sum"),
        x0=("x", "first"), y0=("y", "first"), x1=("x", "last"), y1=("y", "last"),
        n_move=("is_move", "sum"), n_press=("is_press", "sum"),
        t_held_move=("t_held_move", "min"), t_release=("t_release", "max"),
    )
    acts = acts[(acts["n_move"] > 0) | (acts["n_press"] > 0)].copy()  # drop release-only
    drag = acts["t_held_move"] < acts["t_release"]
    acts["atype"] = np.where(drag, 1, np.where(acts["n_press"] > 0, 2, 0))  # index in ACTIONS
    elapsed = acts["t1"] - acts["t0"]
    acts["speed"] = np.where((elapsed > 0) & (acts["path"] > 0),
                             acts["path"] / elapsed.where(elapsed > 0, 1.0), np.nan)
    dx, dy = acts["x1"] - acts["x0"], acts["y1"] - acts["y0"]
    angle = np.mod(np.arctan2(-dy, dx), 2 * np.pi)          # screen y grows downward
    acts["dir_bin"] = np.where((dx != 0) | (dy != 0),
                               np.minimum(np.floor(angle / (np.pi / 4)), 7), np.nan)
    acts["dist_bin"] = np.where((acts["path"] > 0) & (acts["path"] <= 800),
                                np.ceil(acts["path"] / 100) - 1, np.nan)
    ms = elapsed * 1000
    acts["time_bin"] = np.where((ms > 0) & (ms <= 1000), np.ceil(ms / 200) - 1, np.nan)
    parts = [
        _binned(acts, "dist_bin", 8, "msd", "speed"),
        _binned(acts, "dir_bin", 8, "mda", "speed"),
        _binned(acts, "atype", 3, "ata", "speed"),
        _binned(acts, "dist_bin", 8, "tdh", None),
        _binned(acts, "dir_bin", 8, "mdh", None),
        _binned(acts, "atype", 3, "ath", None),
        _binned(acts, "time_bin", 5, "mth", None),
    ]
    mouse = pd.concat(parts, axis=1)
    mouse = mouse.rename(columns={f"ata_{i + 1}": f"ata_{a}" for i, a in enumerate(ACTIONS)}
                         | {f"ath_{i + 1}": f"ath_{a}" for i, a in enumerate(ACTIONS)})

    # -- keystroke timing -----------------------------------------------------
    k = ev[ev["type"].isin(["key_down", "key_up"])].copy()
    k["kc"] = k["key"].map(lambda s: s.lower() if isinstance(s, str) and len(s) == 1 else s)
    k = k.dropna(subset=["kc"]).sort_values(["user", "kc", "t"], kind="mergesort")
    g = k.groupby(["user", "kc"])
    nxt_type, nxt_t = g["type"].shift(-1), g["t"].shift(-1)
    pair = (k["type"] == "key_down") & (nxt_type == "key_up")
    hold = k.loc[pair, keys].assign(hold=(nxt_t[pair] - k.loc[pair, "t"]) * 1000)
    kd = ev[ev["type"] == "key_down"]
    same_kd = (kd["user"] == kd["user"].shift()) & (kd["window_start"] == kd["window_start"].shift())
    dd = kd.loc[same_kd, keys].assign(dd=(kd["t"].diff()[same_kd]) * 1000)
    keyfeat = pd.concat([
        hold.groupby(keys)["hold"].agg(hold_mean="mean", hold_std=lambda s: s.std(ddof=0)),
        dd.groupby(keys)["dd"].agg(dd_mean="mean", dd_std=lambda s: s.std(ddof=0)),
    ], axis=1)
    out = pd.concat([mouse, keyfeat], axis=1).reindex(columns=WANG).fillna(0.0)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Detectors with a common fit / predict / predict_proba interface
# ---------------------------------------------------------------------------
class OneClass:
    """IsolationForest fitted to the genuine (label 0) training rows only;
    the score is the negated normality, the decision IsolationForest's own."""

    def __init__(self):
        self.m = IsolationForest(n_estimators=300, random_state=SEED)

    def fit(self, X, y):
        self.m.fit(X[y == 0])
        return self

    def predict(self, X):
        return (self.m.predict(X) == -1).astype(int)

    def predict_proba(self, X):
        s = -self.m.score_samples(X)
        return np.column_stack([-s, s])


class Rule:
    """A fixed rule on the first feature column; nothing is learned."""

    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.fn(X[:, 0]).astype(int)

    def predict_proba(self, X):
        p = self.predict(X).astype(float)
        return np.column_stack([1 - p, p])


def logistic():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


# (row label, feature-set label, features, detector factory)
DETECTORS = [
    ("paper_forest", "Paper (7)", ml.FEATURES, ml._make_model),
    ("paper_logistic", "Paper (7)", ml.FEATURES, logistic),
    ("wang_forest", "Wang et al. (47)", WANG, ml._make_model),
    ("wang_logistic", "Wang et al. (47)", WANG, logistic),
    ("paper_oneclass", "Paper (7)", ml.FEATURES, OneClass),
    ("wang_oneclass", "Wang et al. (47)", WANG, OneClass),
    ("rule_stepcv", r"\texttt{step\_cv} $= 0$", ["step_cv"], lambda: Rule(lambda v: v == 0)),
    ("rule_fkey", "Only F1--F24 keys", [ml.KEY_CLASS_FEATURE], lambda: Rule(lambda v: v == 1)),
]
DETECTOR_NAMES = {
    "paper_forest": "Random forest", "paper_logistic": "Logistic regression",
    "wang_forest": "Random forest", "wang_logistic": "Logistic regression",
    "paper_oneclass": "Isolation forest", "wang_oneclass": "Isolation forest",
    "rule_stepcv": "Rule", "rule_fkey": "Rule",
}


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------
def mean_std(vals) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def fit_score(factory, X, y, tr, te) -> dict:
    model = factory().fit(X[tr], y[tr])
    return mt.score(y[te], model.predict(X[te]), model.predict_proba(X[te])[:, 1])


def fold_maps(data: pd.DataFrame, kind_of: dict) -> list[dict]:
    """The kind-stratified folds of typematched.py (4 folds x 10 seeds)."""
    sessions = data.groupby("session")["label"].first()
    genuine = np.unique(sessions.index[sessions.values == 0])
    by_kind = {k: np.unique([s for s, v in kind_of.items() if v == k]) for k in botgen.KINDS}
    n_folds = min(len(v) for v in by_kind.values())
    maps = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + r)
        fold = {}
        for k in botgen.KINDS:
            for i, s in enumerate(rng.permutation(by_kind[k])):
                fold[s] = i % n_folds
        for i, part in enumerate(np.array_split(rng.permutation(genuine), n_folds)):
            for s in part:
                fold[s] = i
        maps.append(fold)
    return maps


def evaluate(data, features, factory, test_sets, maps, kind_arr) -> dict:
    X, y = data[features].values.astype(float), data["label"].values
    gss = [fit_score(factory, X, y, ~te, te)
           for te in (data["session"].isin(t).values for t in test_sets)]
    strat = []
    for fmap in maps:
        fold = data["session"].map(fmap).values
        strat += [fit_score(factory, X, y, fold != f, fold == f) for f in sorted(set(fold))]
    te0 = data["session"].isin(test_sets[0]).values
    loko = {}
    for kind in botgen.KINDS:
        kmask = kind_arr == kind
        train = (~te0 & (y == 0)) | ((y == 1) & ~kmask)
        test = (te0 & (y == 0)) | kmask
        loko[kind] = fit_score(factory, X, y, train, test)
    return {
        "gss": {m: mean_std([r[m] for r in gss]) for m in ("f1", "roc_auc", "precision", "recall")},
        "stratified": {m: mean_std([r[m] for r in strat])
                       for m in ("f1", "roc_auc", "precision", "recall")},
        "loko": loko,
        "loko_mean_f1": float(np.mean([r["f1"] for r in loko.values()])),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading corpus...")
    genuine_raw = mt.load_genuine()
    synth_raw = botgen.synthetic_event_frame()

    def windows(raw, tag):
        ev = mt.add_sessions(raw, tag)
        feats = ml._window_features(ev, key_class=True)
        ev["window_start"] = ev["event_time"].dt.floor("min")
        owner = ev.groupby(["user", "window_start"])["session"].first().reset_index()
        feats = feats.merge(owner, on=["user", "window_start"], how="left")
        out = feats.merge(wang_features(raw), on=["user", "window_start"], how="left")
        alt = wang_features(raw, GAP_SENSITIVITY_S).rename(columns={f: f"{f}@gap" for f in WANG})
        out = out.merge(alt, on=["user", "window_start"], how="left")
        cols = WANG + [f"{f}@gap" for f in WANG]
        out[cols] = out[cols].fillna(0.0)
        return out

    print("Featurising (released and Wang et al.)...")
    data = pd.concat([windows(genuine_raw, "H").assign(label=0),
                      windows(synth_raw, "B").assign(label=1)], ignore_index=True)
    kind_of = mt.kind_of_sessions(data)
    kind_arr = np.array([kind_of.get(s, "") for s in data["session"].values])
    test_sets = [set(data["session"].values[te]) for _, te in GroupShuffleSplit(
        n_splits=N_REPEATS, test_size=0.25, random_state=SEED).split(
        data[ml.FEATURES].values, data["label"].values, groups=data["session"].values)]
    maps = fold_maps(data, kind_of)

    hum, syn = data[data.label == 0], data[data.label == 1]
    coverage = {
        "windows": {"genuine": int(len(hum)), "synthetic": int(len(syn))},
        "share_with_hold_times": {"genuine": float((hum["hold_mean"] > 0).mean()),
                                  "synthetic": float((syn["hold_mean"] > 0).mean())},
        "share_with_pointer_actions": {
            "genuine": float((hum[[f"ath_{a}" for a in ACTIONS]].sum(axis=1) > 0).mean()),
            "synthetic": float((syn[[f"ath_{a}" for a in ACTIONS]].sum(axis=1) > 0).mean())},
        "share_with_drag_drop": {"genuine": float((hum["ath_drag_drop"] > 0).mean()),
                                 "synthetic": float((syn["ath_drag_drop"] > 0).mean())},
        "medians": {f: {"genuine": float(hum[f].median()), "synthetic": float(syn[f].median())}
                    for f in WANG},
    }

    results = {}
    for name, fset, feats, factory in DETECTORS:
        print(f"  {name}")
        results[name] = evaluate(data, feats, factory, test_sets, maps, kind_arr)
        results[name]["features"] = fset
        results[name]["detector"] = DETECTOR_NAMES[name]

    # Robustness: the Wang forest without hold times, which the generator cannot
    # produce (it emits no key_up), and its permutation importance.
    no_hold = [f for f in WANG if not f.startswith("hold")]
    results["wang_forest_no_hold"] = evaluate(data, no_hold, ml._make_model,
                                              test_sets, maps, kind_arr)
    results["wang_forest_gap_sensitivity"] = evaluate(
        data, [f"{f}@gap" for f in WANG], ml._make_model, test_sets, maps, kind_arr)
    te0 = data["session"].isin(test_sets[0]).values
    Xw, y = data[WANG].values, data["label"].values
    pm = ml._make_model().fit(Xw[~te0], y[~te0])
    perm = permutation_importance(pm, Xw[te0], y[te0], n_repeats=20,
                                  random_state=SEED, scoring="f1")
    wang_perm = dict(sorted(zip(WANG, map(float, perm.importances_mean)),
                            key=lambda kv: -kv[1]))

    # The paper's step_cv rule on all windows (as quoted in the manuscript), to
    # set beside the partition-scored version above.
    rule_all = (data["step_cv"] == 0).astype(int).values
    rule_all_score = mt.score(data["label"].values, rule_all, rule_all.astype(float))

    with open(os.path.join(mt.OUT_DIR, "numbers.json")) as fh:
        paper = json.load(fh)
    checks = {
        "forest_gss_f1": (results["paper_forest"]["gss"]["f1"]["mean"],
                          paper["splits"]["Grouped (session-wise)"]["f1"]["mean"]),
        "forest_loko_mean_f1": (results["paper_forest"]["loko_mean_f1"],
                                paper["crosskind"]["mean_f1"]),
        "logistic_gss_f1": (results["paper_logistic"]["gss"]["f1"]["mean"],
                            paper["models"]["Logistic regression"]["f1"]["mean"]),
        "stepcv_rule_all_windows_f1": (rule_all_score["f1"], paper["stepcv_rule"]["f1"]),
    }
    bad = {k: v for k, v in checks.items() if not np.isclose(v[0], v[1], rtol=0, atol=1e-9)}
    if bad:
        raise SystemExit(f"paper rows do not reproduce numbers.json: {bad}")
    print(f"  paper rows reproduce numbers.json ({len(checks)} values)")

    res = {
        "config": {"seed": SEED, "n_repeats": N_REPEATS, "action_gap_s": ACTION_GAP_S,
                   "gap_sensitivity_s": GAP_SENSITIVITY_S,
                   "wang_features": WANG, "stratified_folds": 4,
                   "fkey_rule": "fkey_share == 1: at least one key event, all F1-F24",
                   "stepcv_rule": "step_cv == 0"},
        "coverage": coverage,
        "results": results,
        "wang_forest_permutation_importance": wang_perm,
        "stepcv_rule_all_windows": rule_all_score,
        "checks_vs_numbers_json": {k: v[0] for k, v in checks.items()},
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  wrote {OUT_JSON}")
    write_tex(res)
    for name, r in results.items():
        print(f"  {name:<22s} GSS F1 {r['gss']['f1']['mean']:.3f} AUC {r['gss']['roc_auc']['mean']:.4f}"
              f" | strat F1 {r['stratified']['f1']['mean']:.3f} AUC {r['stratified']['roc_auc']['mean']:.4f}"
              f" | LOKO " + " ".join(f"{k[:4]} {v['f1']:.2f}/{v['roc_auc']:.3f}"
                                     for k, v in r["loko"].items())
              + f" mean {r['loko_mean_f1']:.3f}")


def write_tex(res: dict) -> None:
    def pm(agg, places=3):
        return rf"{agg['mean']:.{places}f} $\pm$ {agg['std']:.{places}f}"

    rows = []
    for name, *_ in DETECTORS:
        r = res["results"][name]
        rows.append([r["features"], r["detector"], pm(r["gss"]["f1"]),
                     f"{r['gss']['roc_auc']['mean']:.4f}", pm(r["stratified"]["f1"]),
                     f"{r['stratified']['roc_auc']['mean']:.4f}", f"{r['loko_mean_f1']:.3f}"])
    cov = res["coverage"]["share_with_hold_times"]
    nh = res["results"]["wang_forest_no_hold"]
    top = list(res["wang_forest_permutation_importance"].items())[:3]
    top_txt = ", ".join(rf"\texttt{{{f.replace('_', chr(92) + '_')}}} ({v:.2f})" for f, v in top)
    notes = (
        r"Same one-minute windows, sessions, and partitions for every row: ten "
        r"session-wise splits as in Table~\ref{tab:splits}; four folds that each hold "
        r"out one synthetic session per kind and a quarter of the genuine sessions, "
        r"repeated over ten seeds; and leave-one-kind-out (LOKO) with the genuine "
        r"test sessions of the first split. Wang et al.\ features follow their "
        r"seven mouse groups (43 values) with keystroke hold and down-down latency "
        r"summarised by mean and standard deviation; pointer actions are segmented "
        rf"at gaps above {int(ACTION_GAP_S * 1000)}\,ms, a rule their paper does not "
        rf"specify. Hold times exist in {cov['genuine'] * 100:.0f}\,\% of genuine and "
        rf"{cov['synthetic'] * 100:.0f}\,\% of synthetic windows, because the "
        r"generator emits no \texttt{key\_up}; without them the Wang forest scores "
        rf"$F_1$ {nh['gss']['f1']['mean']:.3f} and LOKO mean {nh['loko_mean_f1']:.3f}. "
        rf"Its permutation importance is led by {top_txt}. "
        r"The isolation-forest rows are one-class, trained on genuine windows only. Rules learn "
        r"nothing; their AUC is that of a binary score."
    )
    lines = [
        "% generated by paper/scripts/baselines.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\caption{Same-data comparison: the released features against a published "
        r"keystroke and mouse-dynamics feature set, a one-class detector, and rules.}",
        r"\label{tab:baselines}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Ten splits} & \multicolumn{2}{c}{Kind-stratified} & LOKO \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Features & Detector & $F_1$ & AUC & $F_1$ & AUC & mean $F_1$ \\",
        r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
        rf"\\[2pt] \footnotesize {notes}",
        r"\end{table}",
    ]
    with open(OUT_TEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
