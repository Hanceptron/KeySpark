"""Extended learning curve over generated sessions per kind:

    .venv/bin/python paper/scripts/learning_curve2.py

For 1, 2, 4, 8, 16, and 32 sessions per kind, leave-one-kind-out is repeated over
the 40 genuine partitions of the kind-stratified design (4 folds x 10 seeds), as
in Table tab:lokocontrols (paper/scripts/review2.py), with the seven-feature
forest at three thresholds: the fixed 0.5, the seen-kind maximum-F1 threshold,
and the 1% genuine budget (both from five-fold grouped out-of-fold training
scores). Per-kind AUC is recorded too. At four sessions per kind the released
synthetic windows are used, so that point reproduces Table tab:lokocontrols; the
other sizes regenerate the class with keyspark.botgen (its defaults otherwise).
For the released class, logistic regression's per-kind AUC is added.

Writes paper/tables/learningcurve2.json and paper/tables/learningcurve.tex.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tables as mt  # noqa: E402  (chdirs to the repository root)
import review2 as r2  # noqa: E402

ra = r2.ra
from keyspark import botgen, ml  # noqa: E402

CSV = os.path.join(r2.KEYSPARK, "data", "keyspark_windows.csv")
SIZES = [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 4, 8, 16, 32]
KINDS, SEED, N_REPEATS = ra.KINDS, ra.SEED, ra.N_REPEATS
POINTS = ("fixed_0.5", "seen_max_f1", "budget_0.01")


def synthetic(n: int, released: pd.DataFrame) -> pd.DataFrame:
    if n == 4:
        return released.copy()
    w = mt.windows_with_sessions(botgen.synthetic_event_frame(sessions_per_kind=n), "B")
    w = w[w["n_events"] >= ml.MIN_WINDOW_EVENTS].copy()
    w["kind"] = w["session"].map(mt.kind_of_sessions(w.assign(label=1)))
    w["label"] = 1
    return w


def forest_1job():
    # Same forest as ml._make_model, single-threaded so that partitions run in
    # parallel processes; the fitted trees do not depend on n_jobs.
    return ml._make_model().set_params(n_jobs=1)


def inner_folds(yt, gt):
    """Five-fold grouped inner folds on the training side; returns (folds, dealt)."""
    inner = list(GroupKFold(n_splits=5).split(yt, yt, groups=gt))
    dealt = any(len(np.unique(yt[a])) < 2 for a, _ in inner)
    if dealt:
        # With very few synthetic sessions a grouped fold can leave no automation
        # in an inner training set; deal the synthetic sessions over the folds instead.
        rng = np.random.default_rng(SEED)
        pos = rng.permutation(np.unique(gt[yt == 1]))
        k = min(5, len(pos))
        fid = np.empty(len(gt), dtype=int)
        for i, s_ in enumerate(pos):
            fid[gt == s_] = i % k
        for i, part in enumerate(np.array_split(rng.permutation(np.unique(gt[yt == 0])), k)):
            fid[np.isin(gt, part)] = i
        inner = [(np.where(fid != i)[0], np.where(fid == i)[0]) for i in range(k)]
    return inner, dealt


def one_task(X, y, g, kind, r, f, k, factories):
    fold = ra.stratified_folds(y, g, kind, np.random.default_rng(SEED + r), 4)
    gen_test = (y == 0) & (fold == f)
    km = kind == k
    train = ((y == 0) & ~gen_test) | ((y == 1) & ~km)
    test = gen_test | km
    Xt, yt, gt = X[train], y[train], g[train]
    out = []
    inner, dealt = inner_folds(yt, gt)
    for name, factory in factories.items():
        oof = np.zeros(len(yt))
        for a, b in inner:
            oof[b] = factory().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
        p = factory().fit(Xt, yt).predict_proba(X[test])[:, 1]
        th = r2.thresholds(oof, yt, yt == 1)
        rec = {"seed": SEED + r, "fold": f, "kind": k, "model": name, "dealt": bool(dealt),
               "auc": float(roc_auc_score(y[test], p))}
        for pt in POINTS:
            pred = (p >= th[pt]).astype(int)
            rec[pt] = float(f1_score(y[test], pred, zero_division=0))
            rec[pt + "_fpr"] = float(pred[y[test] == 0].mean())
        out.append(rec)
    return out


def loko(data: pd.DataFrame, factories: dict) -> dict:
    from joblib import Parallel, delayed
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values.astype(str)
    kind = data["kind"].fillna("").values
    tasks = [(r, f, k) for r in range(N_REPEATS) for f in range(4) for k in KINDS]
    res = Parallel(n_jobs=-1)(delayed(one_task)(X, y, g, kind, r, f, k, factories) for r, f, k in tasks)
    rows = [rec for part in res for rec in part]
    df = pd.DataFrame(rows)
    out = {}
    for name, d in df.groupby("model"):
        res = {"dealt_tasks": int(d["dealt"].sum()),
               "auc_per_kind": {k: float(d[d["kind"] == k]["auc"].mean()) for k in KINDS},
               "auc_min_per_kind": {k: float(d[d["kind"] == k]["auc"].min()) for k in KINDS}}
        for pt in POINTS:
            per_rep = d.groupby(["seed", "fold"])[pt].mean()
            res[pt] = {"mean_f1": float(per_rep.mean()),
                       "range95": [float(per_rep.quantile(0.025)), float(per_rep.quantile(0.975))],
                       "per_kind_f1": {k: float(d[d["kind"] == k][pt].mean()) for k in KINDS},
                       "fpr_mean": float(d[pt + "_fpr"].mean())}
        out[name] = res
    return out


def main() -> None:
    df = pd.read_csv(CSV)
    genuine = df[df["set"] == "genuine"]
    released = df[df["set"] == "synthetic"]
    cols = ["label", "kind", "session", *ml.FEATURES]
    path = os.path.join(mt.OUT_DIR, "learningcurve2.json")
    out = json.load(open(path)) if os.path.exists(path) else {"sizes": {}}
    for n in SIZES:
        t0 = time.time()
        syn = synthetic(n, released)
        data = pd.concat([genuine[cols], syn[cols]], ignore_index=True)
        factories = {"forest": forest_1job}
        if n == 4:
            factories["logistic"] = r2.logistic
        res = loko(data, factories)
        res["synthetic_windows"] = int(len(syn))
        out["sizes"][str(n)] = res
        f = res["forest"]
        print(f"{n:2d} sessions/kind ({len(syn)} windows, {time.time() - t0:.0f} s): "
              + "  ".join(f"{pt} {f[pt]['mean_f1']:.3f} ({f[pt]['range95'][0]:.3f}-{f[pt]['range95'][1]:.3f})" for pt in POINTS)
              + f"  AUC min {min(f['auc_per_kind'].values()):.3f}", flush=True)
        with open(path, "w") as fh:
            json.dump(ra.tidy(out), fh, indent=1)
    write_table(out)


def write_table(out: dict) -> None:
    rows = []
    dealt = {n: out["sizes"][n]["forest"].get("dealt_tasks", 0) for n in out["sizes"]}
    dealt_text = "; ".join(f"{v} of 160 cases at {n} session{'s' if n != '1' else ''} per kind" for n, v in sorted(dealt.items(), key=lambda t: int(t[0])) if v)
    for n in sorted(out["sizes"], key=int):
        f = out["sizes"][n]["forest"]
        cells = [f"{f[pt]['mean_f1']:.3f} ({f[pt]['range95'][0]:.3f}--{f[pt]['range95'][1]:.3f})" for pt in POINTS]
        auc = f["auc_per_kind"]
        rows.append(f"{n} & {out['sizes'][n]['synthetic_windows']:,} & " + " & ".join(cells)
                    + f" & {min(auc.values()):.3f}--{max(auc.values()):.3f} \\\\")
    lines = [
        "% generated by paper/scripts/learning_curve2.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Leave-one-kind-out mean $F_1$ against the number of generated sessions per kind, over 40 genuine partitions.}",
        r"\label{tab:learningcurve}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Sessions & Synthetic & \multicolumn{3}{c}{Mean $F_1$ (2.5--97.5\%) at threshold} & \\",
        r"\cmidrule(lr){3-5}",
        r"per kind & windows & Fixed 0.5 & Seen kinds, max.\ $F_1$ & Genuine 1\% budget & AUC \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt] \footnotesize Seven-feature random forest. Designs and thresholds as in "
        r"Table~\ref{tab:lokocontrols}; the four-session row uses the released synthetic windows and "
        r"repeats that table, the other rows regenerate the class with the same generator settings. Where a "
        r"grouped inner fold would leave no automation in training, the synthetic "
        r"sessions are dealt over the inner folds instead (" + dealt_text + r"). AUC is the range of the per-kind "
        r"mean over the 40 partitions.",
        r"\end{table}",
    ]
    with open(os.path.join(mt.OUT_DIR, "learningcurve.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        write_table(json.load(open(os.path.join(mt.OUT_DIR, "learningcurve2.json"))))
    else:
        main()
