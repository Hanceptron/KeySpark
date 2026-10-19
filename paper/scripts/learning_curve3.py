"""Learning curve read independently of the test composition:

    .venv/bin/python paper/scripts/learning_curve3.py [sizes ...]

For 1, 2, 4, 8, 16, and 32 generated sessions per kind and five generator seeds
(0 to 4), leave-one-kind-out is repeated over the 40 genuine test partitions of
the kind-stratified design of the released class, so every size and seed is
tested on the same genuine windows as Table tab:lokocontrols. For the
seven-feature forest at the fixed 0.5 threshold, the seen-kind maximum-F1
threshold, and the 1% genuine budget (thresholds as in review2.py), it records
the held-out kind's recall and the genuine false-positive rate. Neither depends
on how many windows the held-out kind contributes to the test set. It also
records F1 at the released test composition: the same recall and false-positive
rate with the held-out kind weighted to its window count in the released class.
Seed 0 with four sessions per kind is the released class (checked against the
released windows), so that row reproduces Table tab:lokocontrols.
Writes paper/tables/learningcurve3.json, learningcurve3.tex
(tab:learningcurve), and learningcurve_kinds.tex (tab:learningcurvekinds).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import learning_curve2 as lc2  # noqa: E402  (imports make_tables, which chdirs to the repository root)

mt, r2, ra = lc2.mt, lc2.r2, lc2.ra
from keyspark import botgen, ml  # noqa: E402

CSV = lc2.CSV
SIZES = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
KINDS, SEED, N_REPEATS = ra.KINDS, ra.SEED, ra.N_REPEATS
POINTS = ("fixed_0.5", "seen_max_f1", "budget_0.01")
NAMES = {"jiggler": "Jiggler", "typer": "Auto-typer", "keep_awake": "Keep-awake", "clicker": "Auto-clicker"}


def synthetic(n: int, seed: int, released: pd.DataFrame) -> pd.DataFrame:
    if n == 4 and seed == 0:
        return released.copy()
    w = mt.windows_with_sessions(botgen.synthetic_event_frame(sessions_per_kind=n, seed=seed), "B")
    w = w[w["n_events"] >= ml.MIN_WINDOW_EVENTS].copy()
    assert w["session"].nunique() == n * len(KINDS), "a generated session was split by the idle gap"
    w["kind"] = w["session"].map(mt.kind_of_sessions(w.assign(label=1)))
    w["label"] = 1
    return w


def check_released(released: pd.DataFrame) -> None:
    """Seed 0 with four sessions per kind must give the released windows."""
    w = mt.windows_with_sessions(botgen.synthetic_event_frame(), "B")
    w = w[w["n_events"] >= ml.MIN_WINDOW_EVENTS]

    def signature(frame):
        return sorted(tuple(np.round(r, 9)) for r in frame[["n_events", *ml.FEATURES]].values)
    assert signature(w) == signature(released), "regenerated class differs from the released one"


def genuine_partitions(data: pd.DataFrame) -> list:
    """(repeat, fold, genuine-session set) of the kind-stratified design of the released class."""
    y, g = data["label"].values, data["session"].values.astype(str)
    kind = data["kind"].fillna("").values
    out = []
    for r in range(N_REPEATS):
        fold = ra.stratified_folds(y, g, kind, np.random.default_rng(SEED + r), 4)
        for f in range(4):
            out.append((r, f, frozenset(np.unique(g[(y == 0) & (fold == f)]))))
    return out


def one_task(X, y, g, kind, gen_sessions, k, p_ref):
    gen_test = (y == 0) & np.isin(g, list(gen_sessions))
    km = kind == k
    train = ((y == 0) & ~gen_test) | ((y == 1) & ~km)
    test = gen_test | km
    Xt, yt, gt = X[train], y[train], g[train]
    inner, dealt = lc2.inner_folds(yt, gt)
    oof = np.zeros(len(yt))
    for a, b in inner:
        oof[b] = lc2.forest_1job().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
    p = lc2.forest_1job().fit(Xt, yt).predict_proba(X[test])[:, 1]
    th = r2.thresholds(oof, yt, yt == 1)
    yy = y[test]
    n_gen = int((yy == 0).sum())
    rec = {"kind": k, "dealt": bool(dealt), "auc": float(roc_auc_score(yy, p)), "n_kind": int(km.sum())}
    for pt in POINTS:
        pred = p >= th[pt]
        recall = float(pred[yy == 1].mean())
        fpr = float(pred[yy == 0].mean())
        tp, fp, fn = pred[yy == 1].sum(), pred[yy == 0].sum(), (~pred[yy == 1]).sum()
        rec[pt + "_recall"] = recall
        rec[pt + "_fpr"] = fpr
        rec[pt + "_f1"] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0
        tp_ref = recall * p_ref
        den = 2 * tp_ref + fpr * n_gen + (p_ref - tp_ref)
        rec[pt + "_f1_ref"] = float(2 * tp_ref / den) if den else 0.0
    return rec


def run_size(n, genuine, released, parts, p_ref, cols) -> dict:
    from joblib import Parallel, delayed
    rows, windows = [], {}
    for seed in SEEDS:
        syn = synthetic(n, seed, released)
        windows[seed] = int(len(syn))
        data = pd.concat([genuine[cols], syn[cols]], ignore_index=True)
        X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values.astype(str)
        kind = data["kind"].fillna("").values
        tasks = [(r, f, s, k) for r, f, s in parts for k in KINDS]
        res = Parallel(n_jobs=-1)(delayed(one_task)(X, y, g, kind, s, k, p_ref[k]) for r, f, s, k in tasks)
        for (r, f, _, k), rec in zip(tasks, res):
            rows.append({"gen_seed": seed, "rep": r, "fold": f, **rec})
    d = pd.DataFrame(rows)
    out = {"synthetic_windows": windows, "dealt_tasks": int(d["dealt"].sum()),
           "auc_per_kind": {k: float(d[d["kind"] == k]["auc"].mean()) for k in KINDS}}
    for pt in POINTS:
        rep = d.groupby(["gen_seed", "rep", "fold"])[[pt + "_recall", pt + "_f1_ref", pt + "_f1"]].mean()
        by_seed = d.groupby("gen_seed")[pt + "_recall"].mean()
        out[pt] = {
            "mean_recall": float(rep[pt + "_recall"].mean()),
            "recall_range95": [float(rep[pt + "_recall"].quantile(0.025)), float(rep[pt + "_recall"].quantile(0.975))],
            "recall_seed_means": [float(v) for v in by_seed.values],
            "per_kind_recall": {k: float(d[d["kind"] == k][pt + "_recall"].mean()) for k in KINDS},
            "fpr_mean": float(d[pt + "_fpr"].mean()),
            "mean_f1_ref": float(rep[pt + "_f1_ref"].mean()),
            "f1_ref_range95": [float(rep[pt + "_f1_ref"].quantile(0.025)), float(rep[pt + "_f1_ref"].quantile(0.975))],
            "mean_f1": float(rep[pt + "_f1"].mean()),
            "mean_f1_seed0": float(rep.loc[0][pt + "_f1"].mean()),
        }
    return out


def main() -> None:
    df = pd.read_csv(CSV)
    genuine = df[df["set"] == "genuine"]
    released = df[df["set"] == "synthetic"]
    cols = ["label", "kind", "session", *ml.FEATURES]
    check_released(released)
    parts = genuine_partitions(pd.concat([genuine[cols], released[cols]], ignore_index=True))
    p_ref = released.groupby("kind").size().to_dict()
    path = os.path.join(mt.OUT_DIR, "learningcurve3.json")
    out = json.load(open(path)) if os.path.exists(path) else {"sizes": {}}
    out["seeds"], out["released_windows_per_kind"] = SEEDS, {k: int(p_ref[k]) for k in KINDS}
    for n in SIZES:
        t0 = time.time()
        res = run_size(n, genuine, released, parts, p_ref, cols)
        out["sizes"][str(n)] = res
        print(f"{n:2d} sessions/kind ({time.time() - t0:.0f} s): " + "  ".join(
            f"{pt} recall {res[pt]['mean_recall']:.3f} F1ref {res[pt]['mean_f1_ref']:.3f} FPR {100 * res[pt]['fpr_mean']:.2f}%"
            for pt in POINTS), flush=True)
        with open(path, "w") as fh:
            json.dump(ra.tidy(out), fh, indent=1)
    write_tables(out)


def write_tables(out: dict) -> None:
    sizes = sorted(out["sizes"], key=int)
    rows, krows = [], []
    for n in sizes:
        s = out["sizes"][n]
        w = list(s["synthetic_windows"].values())
        wcell = f"{min(w):,}--{max(w):,}" if min(w) != max(w) else f"{w[0]:,}"
        rec = [f"{s[pt]['mean_recall']:.2f} ({s[pt]['recall_range95'][0]:.2f}--{s[pt]['recall_range95'][1]:.2f})" for pt in POINTS]
        f1 = [f"{s[pt]['mean_f1_ref']:.3f}" for pt in POINTS]
        rows.append(f"{n} & {wcell} & " + " & ".join(rec) + f" & {100 * s['budget_0.01']['fpr_mean']:.1f} & "
                    + " & ".join(f1) + " \\\\")
        kcells = [f"{s[pt]['per_kind_recall'][k]:.2f}" for pt in ("fixed_0.5", "budget_0.01") for k in KINDS]
        fprs = " / ".join(f"{100 * s[pt]['fpr_mean']:.1f}" for pt in POINTS)
        krows.append(f"{n} & " + " & ".join(kcells) + f" & {fprs} \\\\")
    dealt = {n: out["sizes"][n]["dealt_tasks"] for n in sizes if out["sizes"][n]["dealt_tasks"]}
    dealt_text = "; ".join(f"{v} of 800 cases at {n} session{'s' if n != '1' else ''} per kind" for n, v in dealt.items())
    ref = out["released_windows_per_kind"]
    ref_text = ", ".join(f"{ref[k]} {NAMES[k].lower()}" for k in KINDS)
    main = [
        "% generated by paper/scripts/learning_curve3.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Leave-one-kind-out detection against the number of generated sessions per kind, "
        r"over 40 genuine partitions and five generator seeds.}",
        r"\label{tab:learningcurve}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"Sessions & Synthetic & \multicolumn{3}{c}{Mean held-out recall (2.5--97.5\%)} & FPR (\%) & "
        r"\multicolumn{3}{c}{Mean $F_1$ at released composition} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}",
        r"per kind & windows & Fixed 0.5 & Seen kinds & 1\% budget & 1\% budget & Fixed 0.5 & Seen kinds & 1\% budget \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt] \footnotesize Seven-feature random forest. Every size and seed is tested on the 40 genuine test "
        r"partitions of Table~\ref{tab:lokocontrols}. Thresholds as there: fixed 0.5, the seen-kind maximum-$F_1$ "
        r"threshold, and the 1\% genuine budget, both from five-fold grouped out-of-fold training scores. Recall is that "
        r"of the held-out kind and FPR the share of genuine test windows flagged; neither depends on how many windows the "
        r"held-out kind contributes. $F_1$ at released composition applies the same recall and FPR with the held-out kind "
        r"weighted to its window count in the released class (" + ref_text + r" windows). Ranges are the 2.5th and "
        r"97.5th percentiles over 200 replicates (40 partitions $\times$ 5 seeds), so they include generator variability. "
        r"Seed 0 with four sessions per kind is the released class; generated classes of different sizes do not share "
        r"sessions." + (r" Where a grouped inner fold would leave no automation in training, the synthetic sessions "
                         r"are dealt over the inner folds instead (" + dealt_text + r")." if dealt_text else ""),
        r"\end{table}",
    ]
    kinds = [
        "% generated by paper/scripts/learning_curve3.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Per-kind held-out recall and genuine false-positive rate behind Table~\ref{tab:learningcurve}.}",
        r"\label{tab:learningcurvekinds}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{rrrrrrrrrl}",
        r"\toprule",
        r"Sessions & \multicolumn{4}{c}{Recall at fixed 0.5} & \multicolumn{4}{c}{Recall at 1\% budget} & FPR (\%) \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"per kind & " + " & ".join(NAMES[k] for k in KINDS) + " & " + " & ".join(NAMES[k] for k in KINDS)
        + r" & 0.5 / seen / 1\% \\",
        r"\midrule",
        *krows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt] \footnotesize Means over 40 genuine partitions and five generator seeds. FPR is the share of genuine "
        r"test windows flagged at the fixed, seen-kind maximum-$F_1$, and 1\% budget thresholds.",
        r"\end{table}",
    ]
    for name, lines in (("learningcurve3.tex", main), ("learningcurve_kinds.tex", kinds)):
        with open(os.path.join(mt.OUT_DIR, name), "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        write_tables(json.load(open(os.path.join(mt.OUT_DIR, "learningcurve3.json"))))
    else:
        main()
