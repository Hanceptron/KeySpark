"""Controls and checks on the released
feature table only (no raw events):

    .venv/bin/python paper/scripts/review2.py [path/to/keyspark_windows.csv]

Writes paper/tables/review2.json and paper/tables/lokocontrols.tex.

1. Leave-one-kind-out operating points, repeated over the genuine partitions of
   the kind-stratified design (4 genuine folds x 10 seeds): the fixed 0.5
   threshold, two thresholds tuned on the three seen kinds without the held-out
   kind (maximum F1, and the highest threshold that keeps recall >= 0.95, both on
   inner out-of-fold training scores), and the 1% genuine budget. Forest and
   logistic regression.
2. Why step_cv is zero: windows with no measurable pointer movement
   (step_mean = 0) against windows with a constant non-zero step, per class and
   kind; and a "no pointer movement" rule on the paper's splits.
3. Session-level detection per kind with exact (Clopper-Pearson) intervals, and
   permutation importance over the kind-stratified folds.
4. Keyboard-only genuine windows (mouse_fraction = 0) and the genuine windows the
   released policy flags (content-free description).
5. Generated session lengths (keyspark.botgen) against the window counts.

Uses the re-analysis script reanalysis_from_csv.py for loading, folds, and
the released day-level scores, so every design matches the paper's.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KEYSPARK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(KEYSPARK, "paper", "scripts"))
sys.path.append(KEYSPARK)
import reanalysis_from_csv as ra  # noqa: E402
from keyspark import botgen  # noqa: E402

ml = ra.ml
KINDS, SEED, N_REPEATS = ra.KINDS, ra.SEED, ra.N_REPEATS
TABLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")


def clopper_pearson(k: int, n: int, a: float = 0.05) -> list:
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def thresholds(oof, yt, seen):
    """Operating points from inner out-of-fold training scores: genuine 1% budget,
    and two tuned on the seen kinds (max F1; highest threshold with recall >= 0.95)."""
    cand = np.unique(oof)
    best_f1, t_f1 = -1.0, 0.5
    for t in cand:
        f = f1_score(yt, (oof >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, t_f1 = f, t
    pos = np.sort(oof[seen])
    t_rec = pos[int(np.floor(0.05 * len(pos)))]  # keeps >= 95% of seen-kind windows
    return {"fixed_0.5": 0.5, "seen_max_f1": float(t_f1), "seen_recall_0.95": float(t_rec),
            "budget_0.01": float(np.quantile(oof[yt == 0], 0.99))}


def loko_repeated(X, y, g, kind):
    """LOKO over the 40 genuine partitions of the kind-stratified design."""
    rows = []
    for r in range(N_REPEATS):
        fold = ra.stratified_folds(y, g, kind, np.random.default_rng(SEED + r), 4)
        for f in range(4):
            gen_test = (y == 0) & (fold == f)
            for k in KINDS:
                km = kind == k
                train = ((y == 0) & ~gen_test) | ((y == 1) & ~km)
                test = gen_test | km
                Xt, yt, gt = X[train], y[train], g[train]
                seen = yt == 1
                for model_name, factory in (("forest", ml._make_model), ("logistic", logistic)):
                    oof = np.zeros(len(yt))
                    for a, b in GroupKFold(n_splits=5).split(Xt, yt, groups=gt):
                        oof[b] = factory().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
                    p = factory().fit(Xt, yt).predict_proba(X[test])[:, 1]
                    for name, t in thresholds(oof, yt, seen).items():
                        pred = (p >= t).astype(int)
                        rows.append({"seed": SEED + r, "fold": f, "kind": k, "model": model_name,
                                     "point": name, "threshold": t,
                                     "f1": float(f1_score(y[test], pred, zero_division=0)),
                                     "recall": float(pred[y[test] == 1].mean()),
                                     "fpr": float(pred[y[test] == 0].mean())})
        print(f"  repeat {r + 1}/{N_REPEATS} done", flush=True)
    df = pd.DataFrame(rows)
    out = {}
    for (model_name, name), d in df.groupby(["model", "point"]):
        per_rep = d.groupby(["seed", "fold"])["f1"].mean()
        out[f"{model_name}/{name}"] = {
            "mean_f1": float(per_rep.mean()),
            "mean_f1_range95": [float(per_rep.quantile(0.025)), float(per_rep.quantile(0.975))],
            "per_kind_f1": {k: float(d[d["kind"] == k]["f1"].mean()) for k in KINDS},
            "per_kind_recall": {k: float(d[d["kind"] == k]["recall"].mean()) for k in KINDS},
            "fpr_mean": float(d["fpr"].mean()),
            "threshold_range": [float(d["threshold"].min()), float(d["threshold"].max())],
        }
    return out


def rule_scores(X, y, g, kind, flag):
    """F1 of a fixed rule on the ten session-wise test sets and the kind-stratified folds."""
    gss = [float(f1_score(y[te], flag[te])) for _, te in
           GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=SEED).split(X, y, groups=g)]
    strat = []
    for r in range(N_REPEATS):
        fold = ra.stratified_folds(y, g, kind, np.random.default_rng(SEED + r), 4)
        strat += [float(f1_score(y[fold == f], flag[fold == f])) for f in range(4)]
    return {"gss": {"mean": float(np.mean(gss)), "std": float(np.std(gss))},
            "stratified": {"mean": float(np.mean(strat)), "std": float(np.std(strat))}}


def main(path: str) -> None:
    df = pd.read_csv(path)
    data = df[df["set"] != "injected"].reset_index(drop=True)
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values
    kind = data["kind"].fillna("").values
    out = {"source": os.path.basename(path)}

    # 5. Generator session lengths against window counts.
    syn = botgen.synthetic_event_frame()
    syn = syn.sort_values("ts")
    gap = syn["ts"].diff().fillna(0) > 1800
    syn["s"] = gap.cumsum()
    dur = syn.groupby("s")["ts"].agg(lambda t: (t.max() - t.min()) / 60)
    out["generator"] = {
        "sessions": int(dur.size), "minutes_min": float(dur.min()), "minutes_max": float(dur.max()),
        "minutes_mean": float(dur.mean()),
        "minutes_by_kind": {k: [float(v) for v in dur.values[i * 4:(i + 1) * 4]] for i, k in enumerate(KINDS)},
        "windows_by_kind": {k: int((kind == k).sum()) for k in KINDS},
        "sessions_by_kind": {k: int(len(np.unique(g[kind == k]))) for k in KINDS},
        "nominal_minutes": 30,
        "pause_rule": "10% of steps add 3 to 12 base intervals",
    }
    print("generator", json.dumps(out["generator"]))

    # 2. step_cv zero breakdown and the no-movement rule.
    sc, sm, mf = data["step_cv"].values, data["step_mean"].values, data["mouse_fraction"].values
    groups = {"genuine": y == 0, "synthetic": y == 1, **{k: kind == k for k in KINDS}}
    out["step_cv_zero"] = {
        name: {"windows": int(m.sum()),
               "step_cv_zero": int((m & (sc == 0)).sum()),
               "no_movement": int((m & (sc == 0) & (sm == 0)).sum()),
               "constant_step": int((m & (sc == 0) & (sm > 0)).sum()),
               "keyboard_only": int((m & (mf == 0)).sum())}
        for name, m in groups.items()}
    out["rules"] = {"step_cv_zero": rule_scores(X, y, g, kind, (sc == 0).astype(int)),
                    "no_pointer_movement": rule_scores(X, y, g, kind, (sm == 0).astype(int))}
    print("step_cv", json.dumps(out["step_cv_zero"]), json.dumps(out["rules"]))

    # 4. Keyboard-only genuine windows and the windows the released policy flags.
    scored = ra.released_scores(df, data, X, y, g)
    gen = scored[scored["set"] == "genuine"]
    kb = gen["mouse_fraction"] == 0
    t1 = float(np.quantile(gen["proba"], 0.99))
    flagged = gen[gen["proba"] >= ra.TAU]
    out["keyboard_only"] = {
        "genuine_windows": int(len(gen)), "keyboard_only": int(kb.sum()),
        "share": float(kb.mean()),
        "median_score_keyboard_only": float(gen.loc[kb, "proba"].median()),
        "median_score_other": float(gen.loc[~kb, "proba"].median()),
        "share_of_top1pct_keyboard_only": float(kb[gen["proba"] >= t1].mean()),
        "top1pct_windows": int((gen["proba"] >= t1).sum()),
    }
    out["flagged_genuine"] = {
        "threshold": ra.TAU, "windows": int(len(flagged)),
        "keyboard_only": int((flagged["mouse_fraction"] == 0).sum()),
        "no_movement": int((flagged["step_mean"] == 0).sum()),
        "n_events_median": float(flagged["n_events"].median()) if len(flagged) else None,
        "n_events_max": int(flagged["n_events"].max()) if len(flagged) else None,
        "genuine_n_events_median": float(gen["n_events"].median()),
    }
    print("keyboard", json.dumps(out["keyboard_only"]), json.dumps(out["flagged_genuine"]))

    # 3. Session-level detection per kind (kind-stratified out-of-fold, all seeds),
    #    and permutation importance over the kind-stratified folds.
    sess_det = {k: [] for k in KINDS}
    imp = {f: [] for f in ml.FEATURES}
    for r in range(N_REPEATS):
        fold = ra.stratified_folds(y, g, kind, np.random.default_rng(SEED + r), 4)
        pred = np.zeros(len(y), dtype=int)
        for f in range(4):
            te = fold == f
            m = ml._make_model().fit(X[~te], y[~te])
            pred[te] = m.predict(X[te])
            if r == 0:
                pi = permutation_importance(m, X[te], y[te], scoring="f1", n_repeats=20,
                                            random_state=SEED, n_jobs=-1)
                for j, feat in enumerate(ml.FEATURES):
                    imp[feat].append(float(pi.importances_mean[j]))
        for k in KINDS:
            for s in np.unique(g[kind == k]):
                sess_det[k].append(float(pred[g == s].mean()))
    out["session_detection"] = {}
    for k in KINDS:
        rates = np.array(sess_det[k]).reshape(N_REPEATS, -1)
        detected = int((rates.mean(axis=0) >= 0.5).sum())
        n = rates.shape[1]
        out["session_detection"][k] = {"sessions": n, "detected": detected,
                                       "ci95": clopper_pearson(detected, n),
                                       "min_session_recall": float(rates.mean(axis=0).min())}
    out["importance_stratified"] = {f: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                        "max": float(np.max(v))} for f, v in imp.items()}
    print("sessions", json.dumps(out["session_detection"]))
    print("importance", json.dumps(out["importance_stratified"]))

    # 1. Repeated LOKO with seen-kind thresholds.
    print("LOKO over 40 genuine partitions (slow)...", flush=True)
    out["loko_repeated"] = loko_repeated(X, y, g, kind)
    for key, v in out["loko_repeated"].items():
        print(f"  {key:28s} mean F1 {v['mean_f1']:.3f} {v['mean_f1_range95']}  "
              + " ".join(f"{k}:{v['per_kind_f1'][k]:.2f}" for k in KINDS) + f"  FPR {v['fpr_mean']:.4f}")

    with open(os.path.join(TABLES, "review2.json"), "w") as fh:
        json.dump(ra.tidy(out), fh, indent=1)
    write_table(out)


def auc_note() -> str:
    """Per-kind leave-one-kind-out AUC on the same 40 partitions, from learning_curve2.py
    (its four-session point uses the released windows and repeats this design)."""
    path = os.path.join(TABLES, "learningcurve2.json")
    if not os.path.exists(path):
        return ""
    d = json.load(open(path))["sizes"]["4"]
    names = {"jiggler": "jiggler", "typer": "auto-typer", "keep_awake": "keep-awake", "clicker": "auto-clicker"}
    parts = []
    for model, label in (("forest", "forest"), ("logistic", "logistic regression")):
        auc = d[model]["auc_per_kind"]
        parts.append(f"{label} " + ", ".join(f"{auc[k]:.3f} ({names[k]})" for k in KINDS))
    return " Mean per-kind AUC over the partitions: " + "; ".join(parts) + "."


def write_table(out: dict) -> None:
    lr = out["loko_repeated"]
    names = [("fixed_0.5", "Fixed 0.5"), ("seen_max_f1", "Seen kinds, maximum $F_1$"),
             ("seen_recall_0.95", "Seen kinds, recall $\\geq$ 0.95"), ("budget_0.01", "Genuine, 1\\% budget")]
    rows = []
    for key, label in names:
        for model_name, mlabel in (("forest", "Forest"), ("logistic", "Logistic")):
            v = lr[f"{model_name}/{key}"]
            lo, hi = v["mean_f1_range95"]
            rows.append(f"{label} & {mlabel} & " + " & ".join(f"{v['per_kind_f1'][k]:.2f}" for k in KINDS)
                        + f" & {v['mean_f1']:.3f} ({lo:.3f}--{hi:.3f}) & {100 * v['fpr_mean']:.1f} \\\\")
    lines = [
        "% generated by paper/scripts/review2.py -- do not edit by hand",
        r"\begin{table}[H]",
        r"\caption{Leave-one-kind-out $F_1$ at thresholds chosen with and without the seen kinds, over 40 genuine partitions.}",
        r"\label{tab:lokocontrols}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Threshold & Model & Jiggler & Typer & Keep-awake & Clicker & Mean $F_1$ (2.5--97.5\%) & FPR \% \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt] \footnotesize Each held-out kind is tested with the genuine test sessions of every fold "
        r"of the kind-stratified design (4 folds $\times$ 10 seeds); training uses the other genuine "
        r"sessions and the three other kinds. Thresholds come from five-fold grouped out-of-fold "
        r"training scores: the maximum-$F_1$ and recall thresholds use the three seen kinds, the budget "
        r"uses genuine windows only. Mean $F_1$ is averaged over kinds; the range covers the 40 partitions. "
        r"FPR is the mean per-window false-positive rate on genuine test windows. Logistic regression is "
        r"standardised and class-balanced." + auc_note(),
        r"\end{table}",
    ]
    with open(os.path.join(TABLES, "lokocontrols.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        write_table(json.load(open(os.path.join(TABLES, "review2.json"))))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(KEYSPARK, "data", "keyspark_windows.csv"))
