"""Keep-active tools at their default settings.

The generator's jiggler fires every 1 to 8 s and its keep-awake every 6 to 12 s,
but the tools it mirrors default to much slower cadences, and a one-minute
window needs five events to be scored. This script asks what the released
detector does with the real tools at their defaults, and tests two content-free
ways to cover the slow ones:

  1. botgen.default_tool_frame: four sessions of 60 minutes for each of
     Mouse Jiggler (60 s), Mouse Jiggler with its random timer (1 to 60 s),
     Caffeine (F15 key_up every 59 s), and the macOS Jiggler (a burst of pointer
     moves after 300 s idle).
  2. Score them with the forest trained as in the paper (all genuine windows and
     the 16 synthetic sessions, one-minute windows).
  3. Option A: five-minute windows for every class, with the default tools as
     extra kinds, evaluated kind-stratified and leave-one-kind-out, with the
     genuine per-window false-positive rate and day-level false alarms.
  4. Option B: an eighth feature, the share of key events that are F1 to F24
     (hardening.classify_key), and a day-level rule for abstained minutes.
  5. The documented behaviour of the real tools, with sources (TOOLS below).

Run from anywhere (the script changes to the repository root):

    .venv/bin/python paper/scripts/slowtools.py

Writes paper/tables/slowtools.tex and paper/tables/slowtools.json.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (also puts the repo root on sys.path and chdirs there)

from keyspark import botgen, events_io, hardening, ml  # noqa: E402

SEED = mt.SEED
N_REPEATS = 10
TAU, MIN_W = ml.WINDOW_THRESHOLD, ml.MIN_FLAG_WINDOWS   # released day-level policy
RUN_LENGTHS = (1, 3, 5, 10)    # consecutive F-key-only abstained minutes for the day rule
TOOL_KINDS = botgen.DEFAULT_TOOLS
FEATURES8 = ml.FEATURES + [ml.KEY_CLASS_FEATURE]
OUT_TEX = os.path.join(mt.OUT_DIR, "slowtools.tex")
OUT_JSON = os.path.join(mt.OUT_DIR, "slowtools.json")

# Documented behaviour of real keep-active tools. Commit
# hashes are the HEAD each repository was read at.
TOOLS = [
    {
        "tool": "Mouse Jiggler (arkane-systems)",
        "platform": "Windows only (.NET desktop runtime)",
        "source": "https://github.com/arkane-systems/mousejiggler",
        "read_at_commit": "c6e60d47e9a9eb3fbc53ba1878bf091b40468547",
        "default_interval_s": 60,
        "randomisation": "optional random timer (-r, default off): each interval a whole "
                         "number of seconds uniform on 1..period "
                         "(Random.Shared.Next(1, JigglePeriod + 1) in MainForm.cs)",
        "event_types": ["move (relative, SendInput MOUSEEVENTF_MOVE)"],
        "geometry": "Normal (default): (+4d, +4d) then (-4d, -4d), distance d = 1 "
                    "(max 120); Linear: (+-4d, 0); Circle: 8-step octagon; Zen: (0, 0)",
        "other": "skips a tick if the user moved the mouse since the last jiggle; "
                 "also calls SetThreadExecutionState (Helpers.StayAwake)",
        "generator_variant": ["jiggler_default", "jiggler_random"],
    },
    {
        "tool": "Caffeine (Zhorn Software)",
        "platform": "Windows only (a separate Mac app of the same name is unrelated)",
        "source": "https://www.zhornsoftware.co.uk/caffeine/",
        "default_interval_s": 59,
        "randomisation": "none documented; the interval is the first command-line number",
        "event_types": ["key_up (F15) by default", "key_down + key_up with -keypress",
                        "none with -allowss or -stes (SetThreadExecutionState only)"],
        "geometry": "none (keyboard only); -key:xx changes the virtual key, "
                    "-useshift uses Shift",
        "generator_variant": ["keep_awake_default"],
    },
    {
        "tool": "Jiggler (Stick Software)",
        "platform": "macOS",
        "source": "https://github.com/bhaller/Jiggler",
        "read_at_commit": "db2f9396ea6f3f328671e66034f0a3a8c7886ce3",
        "default_interval_s": 300,
        "randomisation": "fixed period (JiggleTime default 5 minutes; JiggleSeconds "
                         "minimum 5 s); by default jiggles only after that much idle time",
        "event_types": ["move (absolute, CGEventPost at kCGHIDEventTap)",
                        "click down/up with the click style",
                        "none with the Zen style (IOPMAssertion only)"],
        "geometry": "standard style (default): 35 moves scheduled 10 ms apart, each the "
                    "current position plus a uniform integer offset of up to "
                    "JiggleDistance^2 + 10 = 14 px per axis (default distance 2), kept "
                    "within 28 px of the last user position",
        "other": "not when the screen is locked (default); declares user activity "
                 "with an IOPMAssertion on every jiggle",
        "generator_variant": ["mac_jiggler_default"],
    },
    {
        "tool": "KeepingYouAwake / caffeinate (macOS)",
        "platform": "macOS",
        "source": "https://github.com/newmarcel/KeepingYouAwake",
        "read_at_commit": "2be93dd7f390b23fbca934ff3b1ac03468a80077",
        "default_interval_s": None,
        "randomisation": "not applicable",
        "event_types": [],
        "geometry": "none: a wrapper around Apple's caffeinate power assertion, "
                    "so it emits no input events and no input-based detector can see it",
        "generator_variant": [],
    },
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def featurize(ev: pd.DataFrame, window: str = "min", key_class: bool = False) -> pd.DataFrame:
    """Windows of ``ev`` (which carries session ids) with the owning session."""
    feats = ml._window_features(ev, window=window, key_class=key_class)
    owner = (
        ev.assign(window_start=ev["event_time"].dt.floor(window))
        .groupby(["user", "window_start"])["session"].first().reset_index()
    )
    return feats.merge(owner, on=["user", "window_start"], how="left")


def tool_events() -> pd.DataFrame:
    """Default-tool events with a session per generated run (one-hour gaps) and
    the tool name, normalised like every other frame."""
    raw = botgen.default_tool_frame().sort_values("ts").reset_index(drop=True)
    ev = ml._prepare(raw)
    ev["session"] = "T:" + (raw["ts"].diff() > 1800).cumsum().astype(str)
    ev["tool"] = raw["tool"]
    return ev


def labeled(parts: list[tuple[pd.DataFrame, int]]) -> pd.DataFrame:
    return pd.concat([p.assign(label=lab) for p, lab in parts], ignore_index=True)


def load_archive() -> pd.DataFrame:
    """The frozen archive with the day-level ground truth, as make_tables.py."""
    archive = events_io.load_events()
    archive["day"] = pd.to_datetime(archive["event_time"]).dt.date
    archive = archive[archive["day"] <= mt.FREEZE].copy()
    seeded = {dt.date.fromisoformat(d) for d in events_io.HUMAN_EXCLUDE_DAYS}
    archive["is_bot"] = archive["day"].isin(seeded) | archive["user"].isin(
        events_io.HUMAN_EXCLUDE_USERS)
    return archive


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def mean_std(vals) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def fold_maps(data: pd.DataFrame, kind_of: dict, kinds: list[str]) -> tuple[int, list[dict]]:
    """Kind-stratified folds as in typematched.py and reanalysis_from_csv.py: each
    fold holds out one session per kind and a quarter of the genuine sessions,
    for seeds SEED .. SEED + 9. With the four generator kinds this reproduces
    the step-4 folds exactly."""
    genuine = np.unique(data.loc[data.label == 0, "session"])
    present = set(data["session"])
    by_kind = {k: np.unique([s for s, v in kind_of.items() if v == k and s in present])
               for k in kinds}
    n_folds = min(len(v) for v in by_kind.values())
    maps = []
    for r in range(N_REPEATS):
        rng = np.random.default_rng(SEED + r)
        fold = {}
        for k in kinds:
            for i, s in enumerate(rng.permutation(by_kind[k])):
                fold[s] = i % n_folds
        for i, part in enumerate(np.array_split(rng.permutation(genuine), n_folds)):
            for s in part:
                fold[s] = i
        maps.append(fold)
    return n_folds, maps


def stratified(data, features, kind_arr, kinds, maps) -> dict:
    """Fold mean +- std of F1 and AUC; out-of-fold per-kind recall and genuine
    per-window false-positive rate at 0.5 and at TAU, averaged over seeds."""
    X, y = data[features].values, data["label"].values
    f1s, aucs, fpr5, fpr8 = [], [], [], []
    recall = {k: [] for k in kinds}
    first_oof = None
    for fmap in maps:
        fold = data["session"].map(fmap).values
        proba, pred = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        for f in sorted(set(fold)):
            te = fold == f
            model = ml._make_model().fit(X[~te], y[~te])
            proba[te] = model.predict_proba(X[te])[:, 1]
            pred[te] = model.predict(X[te])
            r = mt.score(y[te], pred[te], proba[te])
            f1s.append(r["f1"])
            aucs.append(r["roc_auc"])
        first_oof = proba if first_oof is None else first_oof
        fpr5.append(float(pred[y == 0].mean()))
        fpr8.append(float((proba[y == 0] >= TAU).mean()))
        for k in kinds:
            recall[k].append(float(pred[kind_arr == k].mean()))
    return {
        "fold_f1": mean_std(f1s), "fold_roc_auc": mean_std(aucs),
        "recall_by_kind": {k: float(np.mean(v)) for k, v in recall.items()},
        "genuine_fpr_at_0.5": float(np.mean(fpr5)), f"genuine_fpr_at_{TAU}": float(np.mean(fpr8)),
        "n_folds_total": len(f1s), "first_seed_oof": first_oof,
    }


def gss(data, features, test_sets) -> dict:
    X, y = data[features].values, data["label"].values
    runs = []
    for test in test_sets:
        te = data["session"].isin(test).values
        model = ml._make_model().fit(X[~te], y[~te])
        runs.append(mt.score(y[te], model.predict(X[te]), model.predict_proba(X[te])[:, 1]))
    return {k: mean_std([r[k] for r in runs]) for k in ("f1", "roc_auc")}


def loko(data, features, kind_arr, kinds, first_test) -> dict:
    """Leave-one-kind-out; genuine test partition = first session-wise split."""
    X, y = data[features].values, data["label"].values
    te_mask = data["session"].isin(first_test).values
    folds = {}
    for kind in kinds:
        kmask = kind_arr == kind
        train = (~te_mask & (y == 0)) | ((y == 1) & ~kmask)
        test = (te_mask & (y == 0)) | kmask
        model = ml._make_model().fit(X[train], y[train])
        proba = model.predict_proba(X[test])[:, 1]
        pred = model.predict(X[test])
        r = mt.score(y[test], pred, proba)
        r["proba_kind"] = proba[kmask[test]]
        r["pred_kind"] = pred[kmask[test]]
        folds[kind] = r
    return folds


def session_flags(windows: pd.DataFrame, proba: np.ndarray) -> dict:
    """Per-session decision of the released day-level policy (>= MIN_W windows
    scoring >= TAU), treating each generated session as one day."""
    hits = pd.Series(proba >= TAU).groupby(windows["session"].values).sum()
    return {s: bool(v >= MIN_W) for s, v in hits.items()}


def day_level(archive, train, window, features, key_class=False) -> dict:
    """Released policy on the 22 genuine and 4 injected user-days, as in
    make_tables.py: genuine windows scored out of fold (GroupKFold(5) on
    sessions), injected days by a forest fitted to all training windows."""
    X, y, g = train[features].values, train["label"].values, train["session"].values
    proba = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        proba[te] = ml._make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof = train.loc[y == 0, ["user", "window_start"]].assign(proba=proba[y == 0])
    genuine_mask = ~archive["is_bot"]
    gen = featurize(mt.add_sessions(archive[genuine_mask], "H"), window, key_class)
    gen = gen.merge(oof, on=["user", "window_start"], how="left")
    gen["proba"] = gen["proba"].fillna(0.0)
    bot = ml._window_features(archive[~genuine_mask], window=window, key_class=key_class)
    bot["proba"] = ml._make_model().fit(X, y).predict_proba(bot[features].values)[:, 1]
    scored = pd.concat([gen, bot], ignore_index=True)
    scored["day"] = pd.to_datetime(scored["window_start"]).dt.date
    truth = archive.groupby(["user", "day"])["is_bot"].max()
    flagged = scored[scored["proba"] >= TAU].groupby(["user", "day"]).size()
    decision = flagged.reindex(truth.index).fillna(0) >= MIN_W
    return {
        "false_alarms": int((decision & ~truth).sum()), "genuine_days": int((~truth).sum()),
        "detected": int((decision & truth).sum()), "automation_days": int(truth.sum()),
        "genuine_windows": int(len(gen)),
        "genuine_window_fpr_at_0.5": float((gen["proba"] >= 0.5).mean()),
        f"genuine_window_fpr_at_{TAU}": float((gen["proba"] >= TAU).mean()),
    }


# ---------------------------------------------------------------------------
# Option B: the F-key day rule on abstained minutes
# ---------------------------------------------------------------------------
def fkey_runs(ev: pd.DataFrame, unit: pd.Series) -> pd.Series:
    """Longest run of consecutive one-minute windows, per ``unit``, that the
    forest abstains on (< MIN_WINDOW_EVENTS events) and whose key events (at
    least one) are all F1 to F24."""
    m = pd.DataFrame({
        "unit": unit.values,
        "minute": pd.to_datetime(ev["event_time"]).dt.floor("min").values,
        "is_key": ev["type"].isin(["key_down", "key_up"]).values,
        "is_f": [ml._is_function_key(k) if t in ("key_down", "key_up") else False
                 for k, t in zip(ev["key"], ev["type"])],
    })
    per = m.groupby(["unit", "minute"]).agg(n=("is_key", "size"), k=("is_key", "sum"),
                                            f=("is_f", "sum")).reset_index()
    per["fonly"] = (per["n"] < ml.MIN_WINDOW_EVENTS) & (per["k"] > 0) & (per["f"] == per["k"])
    out = {}
    for u, g in per.groupby("unit"):
        idx = (g.loc[g["fonly"], "minute"].astype("int64") // 60_000_000_000).sort_values().values
        best = run = 0
        for i, v in enumerate(idx):
            run = run + 1 if i and v == idx[i - 1] + 1 else 1
            best = max(best, run)
        out[u] = best
    return pd.Series(out, dtype=int)


def rule_counts(runs: pd.Series, units) -> dict:
    runs = runs.reindex(list(units)).fillna(0)
    return {str(n): int((runs >= n).sum()) for n in RUN_LENGTHS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    res: dict = {"config": {
        "seed": SEED, "n_repeats": N_REPEATS, "tau": TAU, "min_windows": MIN_W,
        "min_window_events": ml.MIN_WINDOW_EVENTS, "freeze": str(mt.FREEZE),
        "tool_sessions": 4, "tool_session_minutes": 60, "run_lengths": list(RUN_LENGTHS),
        "tool_session_rule": "a generated tool session counts as one day",
    }, "tools_documented": TOOLS}

    print("Loading corpus...")
    genuine_ev = mt.add_sessions(mt.load_genuine(), "H")
    synth_ev = mt.add_sessions(botgen.synthetic_event_frame(), "B")
    tools_ev = tool_events()
    archive = load_archive()

    # -- 1-minute data exactly as in the paper, plus the optional eighth feature
    data1 = labeled([(featurize(genuine_ev, key_class=True), 0),
                     (featurize(synth_ev, key_class=True), 1)])
    kind_of = mt.kind_of_sessions(data1)
    forest = ml._make_model().fit(data1[ml.FEATURES].values, data1["label"].values)
    with open(os.path.join(mt.OUT_DIR, "numbers.json")) as fh:
        paper = json.load(fh)
    mdi = dict(zip(ml.FEATURES, forest.feature_importances_))
    assert all(np.isclose(mdi[f], paper["feature_importance"][f], rtol=0, atol=1e-12)
               for f in ml.FEATURES), "1-minute training data differs from the paper"
    assert len(data1) == paper["class_balance"]["human"] + paper["class_balance"]["non_human"]
    test_sets = [set(data1["session"].values[te]) for _, te in GroupShuffleSplit(
        n_splits=N_REPEATS, test_size=0.25, random_state=SEED).split(
        data1[ml.FEATURES].values, data1["label"].values, groups=data1["session"].values)]
    tool_of = dict(zip(tools_ev["session"], tools_ev["tool"]))

    # -- 2. Default tools scored by the paper's forest, one-minute windows ------
    print("Scoring default tools with the released forest...")
    tools1 = featurize(tools_ev)
    tools1["tool"] = tools1["session"].map(tool_of)
    minute_counts = tools_ev.groupby(
        [tools_ev["tool"], tools_ev["event_time"].dt.floor("min")]).size()
    scoring = {}
    for tool in TOOL_KINDS:
        sub = tools1[tools1["tool"] == tool]
        evs = tools_ev[tools_ev["tool"] == tool]
        wall = 4 * 60
        p = forest.predict_proba(sub[ml.FEATURES].values)[:, 1] if len(sub) else np.array([])
        pred = forest.predict(sub[ml.FEATURES].values) if len(sub) else np.array([])
        scoring[tool] = {
            "events": int(len(evs)), "events_per_minute": len(evs) / wall,
            "wall_clock_minutes": wall,
            "active_minutes": int(minute_counts[tool].size),
            "scored_minutes": int(len(sub)), "scored_share": len(sub) / wall,
            "detected_at_0.5": int(pred.sum()), f"detected_at_{TAU}": int((p >= TAU).sum()),
            "detection_rate_at_0.5": float(pred.mean()) if len(p) else None,
            "sessions_flagged": int(sum(session_flags(sub, p).values())) if len(p) else 0,
        }
        print(f"  {tool:<20s} {len(evs) / wall:5.2f} ev/min, scored {len(sub)}/{wall} "
              f"minutes, detected {int(pred.sum())}")
    res["released_forest_1min"] = scoring

    # -- 3. Option A: five-minute windows ------------------------------------
    print("Option A: five-minute windows...")
    tools5 = featurize(tools_ev, window="5min")
    data5 = labeled([(featurize(genuine_ev, window="5min"), 0),
                     (featurize(synth_ev, window="5min"), 1), (tools5, 1)])
    kind5 = {**kind_of, **{s: tool_of[s] for s in tools5["session"].unique()}}
    kinds5 = [k for k in [*botgen.KINDS, *TOOL_KINDS] if k in set(kind5.values())]
    kind_arr5 = np.array([kind5.get(s, "") for s in data5["session"].values])
    nf5, maps5 = fold_maps(data5, kind5, kinds5)
    st5 = stratified(data5, ml.FEATURES, kind_arr5, kinds5, maps5)
    lk5 = loko(data5, ml.FEATURES, kind_arr5, kinds5, test_sets[0])
    flags_st = session_flags(data5.loc[data5.label == 1],
                             st5.pop("first_seed_oof")[data5.label.values == 1])
    # Five-minute windows without the default tools in training: what the paper's
    # synthetic class alone would do with the slow tools.
    data5_paper = data5[~np.isin(kind_arr5, TOOL_KINDS)].reset_index(drop=True)
    f5_paper = ml._make_model().fit(data5_paper[ml.FEATURES].values, data5_paper["label"].values)
    option_a = {"n_folds": nf5, "kinds": kinds5, "stratified": st5,
                "windows": {"genuine": int((data5.label == 0).sum()),
                            **{k: int((kind_arr5 == k).sum()) for k in kinds5}},
                "per_tool": {}}
    for tool in TOOL_KINDS:
        sub = tools5[tools5["session"].map(tool_of) == tool]
        wall = 4 * 12
        r = lk5.get(tool)
        p_paper = f5_paper.predict(sub[ml.FEATURES].values) if len(sub) else np.array([])
        sess = [s for s, t in tool_of.items() if t == tool]
        option_a["per_tool"][tool] = {
            "wall_clock_windows": wall, "scored_windows": int(len(sub)),
            "scored_share": len(sub) / wall,
            "recall_in_training_stratified": st5["recall_by_kind"].get(tool),
            "recall_held_out_loko": (float(r["pred_kind"].mean()) if r else None),
            "loko_auc": r["roc_auc"] if r else None,
            "recall_paper_synthetic_only": float(p_paper.mean()) if len(sub) else None,
            "sessions_flagged_stratified_seed0": int(sum(flags_st.get(s, False) for s in sess)),
            "sessions_flagged_loko": (int(sum(session_flags(
                sub, r["proba_kind"]).values())) if r else 0),
        }
    option_a["loko"] = {k: {kk: v for kk, v in r.items() if kk not in ("proba_kind", "pred_kind")}
                        for k, r in lk5.items()}
    option_a["loko_mean_f1_generator_kinds"] = float(np.mean(
        [lk5[k]["f1"] for k in botgen.KINDS]))
    print("  day level, five-minute windows...")
    option_a["day_level"] = day_level(archive, data5, "5min", ml.FEATURES)
    option_a["day_level_paper_synthetic_only"] = day_level(
        archive, data5_paper, "5min", ml.FEATURES)
    print("  day level, one-minute windows (paper baseline)...")
    base = day_level(archive, data1, "min", ml.FEATURES)
    assert (base["false_alarms"], base["detected"]) == (
        paper["day_policy_shipped"]["fp"], paper["day_policy_shipped"]["tp"]), \
        "one-minute day-level baseline does not reproduce numbers.json"
    res["baseline_1min_day_level"] = base
    res["option_a_5min"] = option_a

    # -- 4. Option B: key-class feature and F-key day rule --------------------
    print("Option B: key-class feature...")
    gk = genuine_ev[genuine_ev["type"].isin(["key_down", "key_up"])]
    classes = pd.Series([hardening.classify_key(k) for k in gk["key"]]).value_counts()
    g_min = genuine_ev.groupby(genuine_ev["event_time"].dt.floor("min"))
    fmask = np.array([ml._is_function_key(k) for k in genuine_ev["key"]]) & \
        genuine_ev["type"].isin(["key_down", "key_up"]).values
    hum1, syn1 = data1[data1.label == 0], data1[data1.label == 1]
    kind_arr1 = np.array([kind_of.get(s, "") for s in data1["session"].values])
    option_b = {
        "genuine_key_events": int(len(gk)),
        "genuine_key_class_counts": {k: int(v) for k, v in classes.items()},
        "genuine_function_key_events": int(fmask.sum()),
        "genuine_minutes": int(g_min.ngroups),
        "genuine_minutes_with_key_events": int(
            genuine_ev[genuine_ev["type"].isin(["key_down", "key_up"])]
            .groupby(genuine_ev["event_time"].dt.floor("min")).ngroups),
        "genuine_minutes_with_function_key": int(
            genuine_ev[fmask].groupby(genuine_ev["event_time"].dt.floor("min")).ngroups),
        "genuine_scored_windows_fkey_positive": int((hum1[ml.KEY_CLASS_FEATURE] > 0).sum()),
        "synthetic_windows_fkey_positive_by_kind": {
            k: int((syn1[ml.KEY_CLASS_FEATURE][kind_arr1[data1.label == 1] == k] > 0).sum())
            for k in botgen.KINDS},
        "synthetic_windows_by_kind": {k: int((kind_arr1 == k).sum()) for k in botgen.KINDS},
    }
    _, maps1 = fold_maps(data1, kind_of, list(botgen.KINDS))
    eight = {}
    for name, feats in (("seven", ml.FEATURES), ("eight", FEATURES8)):
        st = stratified(data1, feats, kind_arr1, list(botgen.KINDS), maps1)
        st.pop("first_seed_oof")
        lk = loko(data1, feats, kind_arr1, list(botgen.KINDS), test_sets[0])
        eight[name] = {
            "gss": gss(data1, feats, test_sets), "stratified": st,
            "loko": {k: {kk: v for kk, v in r.items() if kk not in ("proba_kind", "pred_kind")}
                     for k, r in lk.items()},
            "loko_mean_f1": float(np.mean([r["f1"] for r in lk.values()])),
        }
    te = data1["session"].isin(test_sets[0]).values
    X8, y1 = data1[FEATURES8].values, data1["label"].values
    pm = ml._make_model().fit(X8[~te], y1[~te])
    perm = permutation_importance(pm, X8[te], y1[te], n_repeats=20,
                                  random_state=SEED, scoring="f1")
    eight["eight"]["permutation_importance"] = dict(zip(FEATURES8, map(float, perm.importances_mean)))
    sev, eig = eight["seven"], eight["eight"]
    assert np.isclose(sev["gss"]["f1"]["mean"],
                      paper["splits"]["Grouped (session-wise)"]["f1"]["mean"], atol=1e-12)
    assert np.isclose(sev["loko_mean_f1"], paper["crosskind"]["mean_f1"], atol=1e-12)
    option_b["forest_1min"] = eight

    # Day rule on abstained minutes: genuine and injected user-days from the
    # archive, default-tool and generator sessions each as one day.
    arch_unit = archive["user"].astype(str) + "|" + archive["day"].astype(str)
    runs_arch = fkey_runs(archive, arch_unit)
    truth = archive.groupby(arch_unit)["is_bot"].max()
    runs_tools = fkey_runs(tools_ev, tools_ev["session"])
    runs_synth = fkey_runs(synth_ev, synth_ev["session"])
    option_b["day_rule"] = {
        "definition": "flag a day with a run of at least N consecutive one-minute "
                      "windows that have fewer than 5 events, at least one key event, "
                      "and only F1-F24 key events",
        "genuine_false_alarms": rule_counts(runs_arch, truth.index[~truth.values]),
        "genuine_days": int((~truth).sum()),
        "injected_days_detected": rule_counts(runs_arch, truth.index[truth.values]),
        "injected_days": int(truth.sum()),
        "injected_longest_run": {u: int(runs_arch.get(u, 0)) for u in truth.index[truth.values]},
        "tool_sessions_detected": {
            t: rule_counts(runs_tools, [s for s, k in tool_of.items() if k == t])
            for t in TOOL_KINDS},
        "generator_sessions_detected": {
            k: rule_counts(runs_synth, [s for s, v in kind_of.items() if v == k])
            for k in botgen.KINDS},
    }
    res["option_b_key_class"] = option_b

    write_json(res)
    write_tex(res)
    summary(res)


def write_json(res: dict) -> None:
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  wrote {OUT_JSON}")


LABELS = {
    "jiggler_default": (r"Mouse Jiggler", r"60\,s"),
    "jiggler_random": (r"Mouse Jiggler, random timer", r"1--60\,s"),
    "keep_awake_default": (r"Caffeine (F15 \texttt{key\_up})", r"59\,s"),
    "mac_jiggler_default": (r"Jiggler (macOS), move burst", r"300\,s"),
}


def write_tex(res: dict) -> None:
    s1, a, b = res["released_forest_1min"], res["option_a_5min"], res["option_b_key_class"]
    rule_n = "5"

    def frac(k, n):
        return f"{k}/{n}"

    rows = []
    for tool in TOOL_KINDS:
        name, cad = LABELS[tool]
        r1, ra = s1[tool], a["per_tool"][tool]
        det1 = frac(r1["detected_at_0.5"], r1["scored_minutes"]) if r1["scored_minutes"] else "--"
        held = ra["recall_held_out_loko"]
        dets = ra["recall_in_training_stratified"]
        det5 = (f"{held:.2f} / {dets:.2f}" if ra["scored_windows"] else "--")
        rule = frac(b["day_rule"]["tool_sessions_detected"][tool][rule_n], 4)
        rows.append([name, cad, f"{r1['events_per_minute']:.2f}",
                     frac(r1["scored_minutes"], r1["wall_clock_minutes"]), det1,
                     frac(ra["scored_windows"], ra["wall_clock_windows"]), det5, rule])
    base, d5 = res["baseline_1min_day_level"], a["day_level"]
    dr = b["day_rule"]
    def flagged(d):
        k = round(d["genuine_window_fpr_at_0.5"] * d["genuine_windows"])
        return f"{k}/{mt.fmt(d['genuine_windows'])}"

    rows.append([r"\midrule Genuine windows flagged", "", "", "", flagged(base), "",
                 flagged(d5), "--"])
    rows.append([r"Genuine days, false alarms", "", "", "",
                 frac(base["false_alarms"], base["genuine_days"]), "",
                 frac(d5["false_alarms"], d5["genuine_days"]),
                 frac(dr["genuine_false_alarms"][rule_n], dr["genuine_days"])])
    notes = (
        r"Four generated sessions of 60 minutes per tool at its documented defaults. "
        r"One-minute columns: the released forest (trained on all genuine windows and "
        r"the 16 synthetic sessions); a window is scored with at least five events; "
        r"detected means classified as automation (score above 0.5). Five-minute columns: all classes "
        r"re-windowed to five minutes; detection is the recall with the tool's sessions "
        rf"held out of training / included as a kind in the kind-stratified folds "
        rf"(one session per kind per fold, {N_REPEATS} seeds). "
        rf"F-key rule: a session or day is flagged after {rule_n} consecutive abstained "
        r"minutes whose key events are all F1--F24. Genuine row: false alarms of the "
        rf"released day policy ($\tau={TAU}$, $m={MIN_W}$) with genuine windows scored "
        r"out of fold, and of the rule, out of the genuine days; genuine windows "
        r"flagged are those classified as automation out of fold (GroupKFold on "
        r"sessions). The two Windows tools are modelled on a macOS capture host."
    )
    lines = [
        "% generated by paper/scripts/slowtools.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\caption{Keep-active tools at their default settings: what the released "
        r"detector scores, and what five-minute windows or an F-key rule recover.}",
        r"\label{tab:slowtools}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrcccc c}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{One-minute} & \multicolumn{2}{c}{Five-minute} & F-key \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Tool (default) & Period & Ev./min & Scored & Detected & Scored & "
        r"Held out / in & rule \\",
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


def summary(res: dict) -> None:
    a, b = res["option_a_5min"], res["option_b_key_class"]
    print("\nOption A (5-min) stratified F1 "
          f"{a['stratified']['fold_f1']['mean']:.3f} +- {a['stratified']['fold_f1']['std']:.3f}, "
          f"genuine window FPR@0.5 {a['stratified']['genuine_fpr_at_0.5']:.4f}; "
          f"day false alarms {a['day_level']['false_alarms']}/{a['day_level']['genuine_days']}, "
          f"detected {a['day_level']['detected']}/{a['day_level']['automation_days']}")
    for t, r in a["per_tool"].items():
        print(f"  {t:<20s} scored {r['scored_windows']}/{r['wall_clock_windows']} "
              f"held-out recall {r['recall_held_out_loko']} in-training {r['recall_in_training_stratified']}")
    e7, e8 = b["forest_1min"]["seven"], b["forest_1min"]["eight"]
    print(f"Option B: genuine F-key events {b['genuine_function_key_events']}; "
          f"7 vs 8 features GSS F1 {e7['gss']['f1']['mean']:.3f} / {e8['gss']['f1']['mean']:.3f}, "
          f"LOKO mean F1 {e7['loko_mean_f1']:.3f} / {e8['loko_mean_f1']:.3f}")
    print(f"  day rule genuine false alarms {b['day_rule']['genuine_false_alarms']}, "
          f"Caffeine sessions {b['day_rule']['tool_sessions_detected']['keep_awake_default']}, "
          f"injected days {b['day_rule']['injected_days_detected']}")


if __name__ == "__main__":
    main()
