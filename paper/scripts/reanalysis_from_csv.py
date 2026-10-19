"""Re-analyses of the released feature table requested in review.

    uv run python paper/scripts/reanalysis_from_csv.py [data/keyspark_windows.csv] [--out-dir DIR]

Needs no raw events. Same loading, seeds, model (keyspark.ml._make_model), and
features (keyspark.ml.FEATURES) as reproduce_from_csv.py. Prints the analyses
below and writes reanalysis.json plus four LaTeX fragments (stratified.tex,
transfer.tex, daygrid.tex, forward.tex) to the output directory, paper/tables/
by default:

1. Kind-stratified session-wise evaluation: four folds, each holding out one
   synthetic session per kind and a quarter of the genuine sessions, repeated
   over ten seeds, with session-level bootstrap 95% intervals. Also the
   composition of the ten GroupShuffleSplit test sets used in the paper.
2. Threshold transfer for leave-one-kind-out: thresholds set from out-of-fold
   genuine training scores at genuine false-positive budgets of 0.5% to 5%.
3. One-class baseline: an IsolationForest fitted to genuine training sessions
   only, scored on each held-out kind.
4. Day-level operating-point grid: the day-policy scores of the paper under
   fixed and budget thresholds, with count and consecutive-minute day rules.
5. Forward-in-time hold-out on genuine days, a forward check of the chosen
   operating point, and Wilson 95% intervals for the day-level counts.
"""
import argparse
import json
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from keyspark import ml  # noqa: E402

SEED = 42
KINDS = ["jiggler", "typer", "keep_awake", "clicker"]
N_REPEATS = 10                     # seeds for the kind-stratified design
N_BOOT = 1000                      # session-level bootstrap replicates
BUDGETS = [0.005, 0.01, 0.02, 0.05]  # genuine per-window false-positive budgets
FIXED = [0.5, 0.8]                 # fixed window thresholds in the day grid
# Day rules: ("count", m) flags a day with >= m flagged windows; ("run", r) with
# >= r flagged windows in a row (consecutive window indices) inside one session.
RULES = [("count", 2), ("count", 3), ("count", 5), ("run", 2), ("run", 3), ("run", 5)]
TAU, MIN_WINDOWS = 0.8, 2          # released configuration
CHOSEN = (0.01, ("run", 3))        # operating point checked forward in time


def fmt_ci(interval, places: int = 3) -> str:
    return f"[{interval[0]:.{places}f}, {interval[1]:.{places}f}]"


def word(n: int) -> str:
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return words[n] if n < len(words) else str(n)


def f1_np(y_true, y_pred) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn)


def wilson(k: int, n: int, z: float = 1.96) -> list:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def tidy(obj):
    """Round floats to ten places so last-bit noise from the forest's parallel
    tree averaging does not change the JSON between runs."""
    if isinstance(obj, float):
        return round(obj, 10)
    if isinstance(obj, dict):
        return {k: tidy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tidy(v) for v in obj]
    return obj


def ci(values) -> list:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def rule_name(rule) -> str:
    return f"m>={rule[1]}" if rule[0] == "count" else f"run>={rule[1]}"


def existing_design(X, y, g, kind) -> dict:
    """Composition of the paper's ten GroupShuffleSplit test sets."""
    splits = []
    for i, (tr, te) in enumerate(GroupShuffleSplit(n_splits=10, test_size=0.25,
                                                   random_state=SEED).split(X, y, groups=g)):
        pos = te[y[te] == 1]
        per_kind = {k: len(set(g[pos][kind[pos] == k])) for k in KINDS}
        m = ml._make_model().fit(X[tr], y[tr])
        splits.append({
            "split": i,
            "synthetic_sessions": len(set(g[pos])),
            "sessions_per_kind": per_kind,
            "kinds_present": sum(1 for k in KINDS if per_kind[k]),
            "positive_windows": len(pos),
            "genuine_sessions": len(set(g[te][y[te] == 0])),
            "genuine_windows": int((y[te] == 0).sum()),
            "f1": float(f1_score(y[te], m.predict(X[te]))),
        })
    syn = [s["synthetic_sessions"] for s in splits]
    pos = [s["positive_windows"] for s in splits]
    present = [s["kinds_present"] for s in splits]
    summary = {
        "synthetic_sessions_range": [min(syn), max(syn)],
        "positive_windows_range": [min(pos), max(pos)],
        "splits_with_all_four_kinds": present.count(4),
        "splits_with_one_kind": present.count(1),
        "one_kind_splits": [[k for k in KINDS if s["sessions_per_kind"][k]][0]
                            for s in splits if s["kinds_present"] == 1],
        "splits_missing_kind": {k: sum(1 for s in splits if not s["sessions_per_kind"][k])
                                for k in KINDS},
    }
    return {"splits": splits, "summary": summary}


def stratified_folds(y, g, kind, rng, n_folds: int) -> np.ndarray:
    """Fold id per window. Within each kind the sessions are shuffled and dealt
    one per fold; genuine sessions are shuffled and split into n_folds groups."""
    fold = np.empty(len(g), dtype=int)
    for k in KINDS:
        for i, s in enumerate(rng.permutation(np.unique(g[kind == k]))):
            fold[g == s] = i % n_folds
    for i, part in enumerate(np.array_split(rng.permutation(np.unique(g[y == 0])), n_folds)):
        fold[np.isin(g, part)] = i
    return fold


def kind_stratified(X, y, g, kind, session_rows) -> dict:
    n_folds = min(len(np.unique(g[kind == k])) for k in KINDS)
    folds, probas, preds = [], [], []
    for r in range(N_REPEATS):
        fold = stratified_folds(y, g, kind, np.random.default_rng(SEED + r), n_folds)
        proba, pred = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        for f in range(n_folds):
            te = fold == f
            m = ml._make_model().fit(X[~te], y[~te])
            proba[te] = m.predict_proba(X[te])[:, 1]
            pred[te] = m.predict(X[te])
            folds.append({
                "seed": SEED + r, "fold": f,
                "f1": float(f1_score(y[te], pred[te])),
                "auc": float(roc_auc_score(y[te], proba[te])),
                "recall": {k: float(pred[te & (kind == k)].mean()) for k in KINDS},
            })
        probas.append(proba)
        preds.append(pred)

    def pooled(idx):
        """Out-of-fold metrics on rows idx, averaged over the repeats."""
        f1 = np.mean([f1_np(y[idx], p[idx]) for p in preds])
        auc = np.mean([roc_auc_score(y[idx], p[idx]) for p in probas])
        rec = {k: np.mean([p[idx][kind[idx] == k].mean() for p in preds]) for k in KINDS}
        return f1, auc, rec

    # Session-level bootstrap, stratified so every replicate keeps the genuine
    # session count and four sessions of each kind.
    strata = [np.unique(g[y == 0])] + [np.unique(g[kind == k]) for k in KINDS]
    rng = np.random.default_rng(SEED)
    boot_f1, boot_auc, boot_rec = [], [], {k: [] for k in KINDS}
    for _ in range(N_BOOT):
        idx = np.concatenate([session_rows[s] for st in strata for s in rng.choice(st, len(st))])
        f1, auc, rec = pooled(idx)
        boot_f1.append(f1)
        boot_auc.append(auc)
        for k in KINDS:
            boot_rec[k].append(rec[k])

    f1, auc, rec = pooled(np.arange(len(y)))
    fold_f1 = [f["f1"] for f in folds]
    fold_auc = [f["auc"] for f in folds]
    return {
        "n_folds": n_folds, "n_repeats": N_REPEATS, "n_boot": N_BOOT,
        "genuine_sessions": len(strata[0]),
        "fold_mean_f1": float(np.mean(fold_f1)), "fold_std_f1": float(np.std(fold_f1)),
        "fold_mean_auc": float(np.mean(fold_auc)), "fold_std_auc": float(np.std(fold_auc)),
        "pooled_f1": float(f1), "pooled_f1_ci95": ci(boot_f1),
        "pooled_auc": float(auc), "pooled_auc_ci95": ci(boot_auc),
        "recall": {k: {"value": float(rec[k]), "ci95": ci(boot_rec[k])} for k in KINDS},
        "folds": folds,
    }


def operating_point(y_true, pred, threshold) -> dict:
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(pred[y_true == 0].mean()),
    }


def threshold_transfer(X, y, g, kind, te_mask) -> dict:
    """Leave-one-kind-out at 0.5 and at thresholds set from genuine data only."""
    names = ["fixed_0.5"] + [f"budget_{b:g}" for b in BUDGETS]
    out = {n: {"per_kind": {}} for n in names}
    auc = {}
    for k in KINDS:
        km = kind == k
        train = (~te_mask & (y == 0)) | ((y == 1) & ~km)
        test = (te_mask & (y == 0)) | km
        Xt, yt, gt = X[train], y[train], g[train]
        oof = np.zeros(len(yt))
        for a, b in GroupKFold(n_splits=5).split(Xt, yt, groups=gt):
            oof[b] = ml._make_model().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
        m = ml._make_model().fit(Xt, yt)
        p = m.predict_proba(X[test])[:, 1]
        auc[k] = float(roc_auc_score(y[test], p))
        out["fixed_0.5"]["per_kind"][k] = operating_point(y[test], m.predict(X[test]), 0.5)
        for b in BUDGETS:
            thr = np.quantile(oof[yt == 0], 1 - b)
            out[f"budget_{b:g}"]["per_kind"][k] = operating_point(
                y[test], (p >= thr).astype(int), thr)
    for n in names:
        out[n]["mean_f1"] = float(np.mean([v["f1"] for v in out[n]["per_kind"].values()]))
    out["forest_auc"] = auc
    return out


def one_class(X, y, kind, te_mask) -> dict:
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[~te_mask & (y == 0)])
    score = -iso.score_samples(X)
    auc = {}
    for k in KINDS:
        test = (te_mask & (y == 0)) | (kind == k)
        auc[k] = float(roc_auc_score(y[test], score[test]))
    test = te_mask | (y == 1)
    auc["all_synthetic"] = float(roc_auc_score(y[test], score[test]))
    return auc


def released_scores(df, data, X, y, g) -> pd.DataFrame:
    """Window scores behind the paper's day-level policy (as reproduce_from_csv.py):
    genuine windows out of fold, injected windows by the forest fitted to all
    genuine and synthetic windows."""
    proba = np.zeros(len(data))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        proba[te] = ml._make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    genuine = data[data["set"] == "genuine"].assign(proba=proba[data["set"] == "genuine"])
    injected = df[df["set"] == "injected"].copy()
    forest = ml._make_model().fit(X, y)
    injected["proba"] = forest.predict_proba(injected[ml.FEATURES].values)[:, 1]
    return pd.concat([genuine, injected])


def day_stats(scored: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Per (user, day, set): windows scoring >= thr, and the longest run of such
    windows with consecutive window indices inside one session."""
    d = scored.sort_values(["set", "user", "day", "session", "window"])
    hit = d["proba"].values >= thr
    same = (d["session"].eq(d["session"].shift()) & d["day"].eq(d["day"].shift())
            & d["window"].diff().eq(1)).values
    run = np.cumsum(~(hit & np.r_[False, hit[:-1]] & same))
    d = d.assign(hit=hit, run=run)
    d["run_len"] = d.groupby("run")["hit"].transform("sum")
    return d.groupby(["user", "day", "set"]).agg(
        count=("hit", "sum"), longest=("run_len", "max")).reset_index()


def day_counts(scored: pd.DataFrame, thr: float, rule) -> dict:
    """False alarms on genuine days and detections on injected days."""
    stats = day_stats(scored, thr)
    flag = stats["count" if rule[0] == "count" else "longest"] >= rule[1]
    gen, inj = stats["set"] == "genuine", stats["set"] == "injected"
    fa, ng, det, ni = int(flag[gen].sum()), int(gen.sum()), int(flag[inj].sum()), int(inj.sum())
    out = {"false_alarms": fa, "genuine_days": ng, "false_alarms_wilson95": wilson(fa, ng)}
    if ni:
        out.update({"detected": det, "automation_days": ni, "detected_wilson95": wilson(det, ni)})
    return out


def budget_threshold(scores, budget: float) -> float:
    return float(np.quantile(scores, 1 - budget))


def day_grid(scored: pd.DataFrame) -> dict:
    gen_scores = scored.loc[scored["set"] == "genuine", "proba"].values
    thresholds = {f"fixed_{t:g}": t for t in FIXED}
    thresholds.update({f"budget_{b:g}": budget_threshold(gen_scores, b) for b in BUDGETS})
    return {name: {"threshold": float(thr),
                   "rules": {rule_name(rule): day_counts(scored, thr, rule) for rule in RULES}}
            for name, thr in thresholds.items()}


def forward_holdout(df, released) -> dict:
    """Train on genuine days up to two thirds of the genuine days plus all
    synthetic windows; test on the later genuine days and the injected days.
    The chosen operating point takes its threshold from out-of-fold scores of the
    training genuine windows only. The paper's scoring of the same days is
    reported alongside."""
    gen, syn, inj = (df[df["set"] == s] for s in ("genuine", "synthetic", "injected"))
    days = np.sort(gen["day"].unique())
    cut = days[len(days) * 2 // 3 - 1]
    train = pd.concat([gen[gen["day"] <= cut], syn]).reset_index(drop=True)
    test = gen[gen["day"] > cut]
    Xt, yt, gt = train[ml.FEATURES].values, train["label"].values, train["session"].values
    m = ml._make_model().fit(Xt, yt)
    p = m.predict_proba(test[ml.FEATURES].values)[:, 1]
    oof = np.zeros(len(train))
    for a, b in GroupKFold(n_splits=5).split(Xt, yt, groups=gt):
        oof[b] = ml._make_model().fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
    scored = pd.concat([test.assign(proba=p),
                        inj.assign(proba=m.predict_proba(inj[ml.FEATURES].values)[:, 1])])

    later = released[(released["set"] == "injected") | (released["day"] > cut)]
    p_ref = later.loc[later["set"] == "genuine", "proba"].values
    budget, rule = CHOSEN
    thr = budget_threshold(oof[yt == 0], budget)
    thr_ref = budget_threshold(released.loc[released["set"] == "genuine", "proba"].values, budget)
    return {
        "last_train_day": int(cut),
        "train_days": int((days <= cut).sum()), "test_days": int((days > cut).sum()),
        "train_genuine_windows": int((gen["day"] <= cut).sum()),
        "test_windows": len(test),
        "fpr": {"forward": {"0.5": float((p >= 0.5).mean()), "0.8": float((p >= 0.8).mean())},
                "paper_scoring": {"0.5": float((p_ref >= 0.5).mean()), "0.8": float((p_ref >= 0.8).mean())}},
        "released_rule": {
            "threshold": TAU, "rule": rule_name(("count", MIN_WINDOWS)),
            "forward": day_counts(scored, TAU, ("count", MIN_WINDOWS)),
            "paper_scoring": day_counts(later, TAU, ("count", MIN_WINDOWS)),
        },
        "chosen": {
            "budget": budget, "rule": rule_name(rule),
            "forward_threshold": float(thr), "paper_threshold": float(thr_ref),
            "forward": day_counts(scored, thr, rule),
            "paper_scoring": day_counts(later, thr_ref, rule),
        },
    }


# ---------------------------------------------------------------------------
# LaTeX fragments
# ---------------------------------------------------------------------------
def esc(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(out_dir: str, name: str, caption: str, label: str, spec: str,
                header: list, rows: list, notes: str = "", pre_header: str = "",
                colsep_pt: int = 0) -> None:
    """One table fragment in the style of make_tables.py write_table."""
    lines = [
        "% generated by paper/scripts/reanalysis_from_csv.py -- do not edit by hand",
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\small",
    ]
    if colsep_pt:
        lines.append(rf"\setlength{{\tabcolsep}}{{{colsep_pt}pt}}")
    lines += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    if pre_header:
        lines.append(pre_header)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(rf"\\[2pt] \footnotesize {notes}")
    lines.append(r"\end{table}")
    path = os.path.join(out_dir, f"{name}.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def tt(kind: str) -> str:
    return rf"\texttt{{{esc(kind)}}}"


def pct_ci(interval) -> str:
    return f"{interval[0] * 100:.1f}--{interval[1] * 100:.1f}"


def budget_label(budget: float) -> str:
    return rf"{budget * 100:g}\% budget"


def table_stratified(out_dir, strat, existing) -> None:
    def cell_ci(interval, places):
        return f"{interval[0]:.{places}f}--{interval[1]:.{places}f}"

    rows = [
        ["Pooled $F_1$", f"{strat['pooled_f1']:.3f}", cell_ci(strat["pooled_f1_ci95"], 3)],
        ["Pooled AUC", f"{strat['pooled_auc']:.4f}", cell_ci(strat["pooled_auc_ci95"], 4)],
    ]
    rows += [[f"Recall, {tt(k)}", f"{strat['recall'][k]['value']:.3f}",
              cell_ci(strat["recall"][k]["ci95"], 3)] for k in KINDS]
    rows.append(["Fold-mean $F_1$", rf"{strat['fold_mean_f1']:.3f} $\pm$ {strat['fold_std_f1']:.3f}",
                 "--"])
    sm = existing["summary"]
    missing = ", ".join(f"{tt(k)} {n}" for k, n in sm["splits_missing_kind"].items())
    one_kind = ", ".join(tt(k) for k in sm["one_kind_splits"])
    n_runs = strat["n_folds"] * strat["n_repeats"]
    notes = (
        f"Each of {word(strat['n_folds'])} folds holds out one synthetic session of every kind and a "
        f"quarter of the {strat['genuine_sessions']} genuine sessions; the design is repeated over "
        f"{word(strat['n_repeats'])} seeds ({n_runs} folds) at threshold 0.5. Pooled values use the "
        "out-of-fold scores of all windows, averaged over seeds. Intervals are the 2.5th and 97.5th "
        f"percentiles of {strat['n_boot']:,} session-level bootstrap replicates that resample the "
        "genuine sessions and the four sessions of each kind. For comparison, the ten "
        "session-wise test sets of Table~\\ref{tab:splits} contain "
        f"{sm['synthetic_sessions_range'][0]} to {sm['synthetic_sessions_range'][1]} of the 16 "
        f"synthetic sessions ({sm['positive_windows_range'][0]} to "
        f"{sm['positive_windows_range'][1]} positive windows); {word(sm['splits_with_all_four_kinds'])} "
        f"contains all four kinds and {word(sm['splits_with_one_kind'])} contains only {one_kind}. "
        f"Test sets lacking each kind: {missing}."
    )
    write_table(out_dir, "stratified",
                "Kind-stratified session-wise evaluation of the forest.",
                "tab:stratified", "lcc", ["Metric", "Estimate", r"95\% interval"], rows, notes)


def table_transfer(out_dir, transfer, iso) -> None:
    names = ["fixed_0.5"] + [f"budget_{b:g}" for b in BUDGETS]
    labels = ["Fixed 0.5"] + [budget_label(b) for b in BUDGETS]
    rows = []
    for name, label in zip(names, labels):
        op = transfer[name]
        thr = [op["per_kind"][k]["threshold"] for k in KINDS]
        fpr = [op["per_kind"][k]["fpr"] * 100 for k in KINDS]
        thr_cell = f"{thr[0]:.2f}" if min(thr) == max(thr) else f"{min(thr):.2f}--{max(thr):.2f}"
        fpr_cell = f"{min(fpr):.1f}--{max(fpr):.1f}" if max(fpr) > 0 else "0.0"
        rows.append([label, thr_cell]
                    + [f"{op['per_kind'][k]['recall']:.2f}/{op['per_kind'][k]['f1']:.2f}" for k in KINDS]
                    + [fpr_cell, f"{op['mean_f1']:.3f}"])
    rows.append(["Forest AUC", "--"] + [f"{transfer['forest_auc'][k]:.3f}" for k in KINDS] + ["--", "--"])
    rows.append(["One-class AUC", "--"] + [f"{iso[k]:.3f}" for k in KINDS] + ["--", "--"])
    notes = (
        "Leave-one-kind-out folds as in Table~\\ref{tab:crosskind}. Kind cells give recall/$F_1$ on "
        "the held-out kind against the genuine test partition. A budget threshold is the "
        "$(1-b)$ quantile of out-of-fold (five-fold grouped) scores of the genuine training "
        "windows, so it uses no window of the held-out kind; $t$ and per-window false-positive "
        "rate (FPR) are ranges over the four folds. One-class AUC: an isolation forest "
        "(300 trees) fitted to genuine training sessions only, scoring each held-out kind "
        f"against the same genuine test partition (all kinds pooled: {iso['all_synthetic']:.3f})."
    )
    write_table(out_dir, "transfer",
                "Leave-one-kind-out operating points with thresholds set from genuine data only.",
                "tab:transfer", "lc" + "c" * (len(KINDS) + 2),
                ["Threshold", "$t$"] + [tt(k) for k in KINDS] + [r"FPR (\%)", "Mean $F_1$"],
                rows, notes, colsep_pt=3)


def table_daygrid(out_dir, grid) -> None:
    names = [f"fixed_{t:g}" for t in FIXED] + [f"budget_{b:g}" for b in BUDGETS]
    labels = [f"Fixed {t:.1f}" for t in FIXED] + [budget_label(b) for b in BUDGETS]
    released = (f"fixed_{TAU:g}", rule_name(("count", MIN_WINDOWS)))
    chosen = (f"budget_{CHOSEN[0]:g}", rule_name(CHOSEN[1]))
    rows, fa_seen, det_seen = [], set(), set()
    for name, label in zip(names, labels):
        row = [label, f"{grid[name]['threshold']:.3f}"]
        for rule in RULES:
            c = grid[name]["rules"][rule_name(rule)]
            fa_seen.add(c["false_alarms"])
            det_seen.add(c["detected"])
            cell = f"{c['false_alarms']}/{c['detected']}"
            if (name, rule_name(rule)) == released:
                cell = rf"\underline{{{cell}}}"
            elif (name, rule_name(rule)) == chosen:
                cell = rf"\textbf{{{cell}}}"
            row.append(cell)
        rows.append(row)
    any_cell = grid[names[0]]["rules"][rule_name(RULES[0])]
    ng, ni = any_cell["genuine_days"], any_cell["automation_days"]
    intervals = "; ".join(f"{k}/{ng}, {pct_ci(wilson(k, ng))}" for k in sorted(fa_seen))
    intervals += "; " + "; ".join(f"{k}/{ni}, {pct_ci(wilson(k, ni))}" for k in sorted(det_seen))
    count_n = [n for kind, n in RULES if kind == "count"]
    run_n = [n for kind, n in RULES if kind == "run"]
    pre = (rf" & & \multicolumn{{{len(count_n)}}}{{c}}{{Count, $m \geq$}}"
           rf" & \multicolumn{{{len(run_n)}}}{{c}}{{Consecutive, $r \geq$}} \\"
           "\n" + rf"\cmidrule(lr){{3-{2 + len(count_n)}}}"
           rf"\cmidrule(lr){{{3 + len(count_n)}-{2 + len(RULES)}}}")
    notes = (
        f"Cells give false alarms out of {ng} genuine days / detections out of {ni} automation "
        "days. Window scores as in Table~\\ref{tab:daypolicy}: genuine windows out of fold, "
        "automation days by the forest fitted to all genuine and synthetic windows. A window is "
        "flagged when its score is at least $t$; a budget threshold is the $(1-b)$ quantile of "
        f"the out-of-fold genuine scores. A day is flagged when at least $m$ of its windows are "
        "flagged (count) or when at least $r$ flagged windows follow one another within one "
        "session (consecutive). Underlined is the released configuration, bold the operating "
        f"point checked forward in time in Table~\\ref{{tab:forward}}. Wilson 95\\% intervals "
        f"(\\%): {intervals}."
    )
    write_table(out_dir, "daygrid",
                "Day-level false alarms and detections by window threshold and day rule.",
                "tab:daygrid", "lc" + "c" * len(RULES),
                ["Threshold", "$t$"] + [str(n) for n in count_n + run_n], rows, notes,
                pre_header=pre, colsep_pt=4)


def table_forward(out_dir, fwd) -> None:
    def days(c, key, n_key):
        return f"{c[key]}/{c[n_key]} ({pct_ci(c[key + '_wilson95'])})"

    rel, ch = fwd["released_rule"], fwd["chosen"]
    rel_rule = rf"$\tau = {TAU}$, $m \geq {MIN_WINDOWS}$"
    ch_rule = rf"{budget_label(ch['budget'])}, $r \geq {CHOSEN[1][1]}$"
    rows = [
        [r"Per-window FPR, $p \geq 0.5$ (\%)", f"{fwd['fpr']['forward']['0.5'] * 100:.2f}",
         f"{fwd['fpr']['paper_scoring']['0.5'] * 100:.2f}"],
        [r"Per-window FPR, $p \geq 0.8$ (\%)", f"{fwd['fpr']['forward']['0.8'] * 100:.2f}",
         f"{fwd['fpr']['paper_scoring']['0.8'] * 100:.2f}"],
        [f"Genuine days flagged, {rel_rule}", days(rel["forward"], "false_alarms", "genuine_days"),
         days(rel["paper_scoring"], "false_alarms", "genuine_days")],
        [f"Automation days detected, {rel_rule}", days(rel["forward"], "detected", "automation_days"),
         days(rel["paper_scoring"], "detected", "automation_days")],
        [f"Threshold $t$, {budget_label(ch['budget'])}", f"{ch['forward_threshold']:.3f}",
         f"{ch['paper_threshold']:.3f}"],
        [f"Genuine days flagged, {ch_rule}", days(ch["forward"], "false_alarms", "genuine_days"),
         days(ch["paper_scoring"], "false_alarms", "genuine_days")],
        [f"Automation days detected, {ch_rule}", days(ch["forward"], "detected", "automation_days"),
         days(ch["paper_scoring"], "detected", "automation_days")],
    ]
    notes = (
        f"Forward: forest trained on the genuine days with index up to {fwd['last_train_day']} "
        f"({fwd['train_days']} of {fwd['train_days'] + fwd['test_days']} days, "
        f"{fwd['train_genuine_windows']:,} windows) and all synthetic windows, tested on the "
        f"{fwd['test_days']} later genuine days ({fwd['test_windows']:,} windows) and the automation "
        "days; its budget threshold is the $(1-b)$ quantile of out-of-fold (five-fold grouped) "
        "scores of the training genuine windows only. Paper scoring: the same days scored as in "
        "Table~\\ref{tab:daygrid}, with the threshold taken from all genuine days. Parentheses: "
        "Wilson 95\\% intervals (\\%)."
    )
    write_table(out_dir, "forward",
                "Forward-in-time hold-out on later genuine days.",
                "tab:forward", "lcc", ["Quantity", "Forward", "Paper scoring"], rows, notes)


def main(path: str, out_dir: str) -> None:
    df = pd.read_csv(path)
    data = df[df["set"] != "injected"].reset_index(drop=True)
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values
    kind = data["kind"].fillna("").values
    session_rows = data.groupby("session").indices
    _, te0 = next(GroupShuffleSplit(n_splits=10, test_size=0.25,
                                    random_state=SEED).split(X, y, groups=g))
    te_mask = np.zeros(len(y), dtype=bool)
    te_mask[te0] = True

    print("1a. Composition of the paper's GroupShuffleSplit(10, 0.25) test sets")
    existing = existing_design(X, y, g, kind)
    print("  split  synth sessions  " + "  ".join(f"{k[:6]:>6s}" for k in KINDS)
          + "  pos win  gen sess  F1")
    for s in existing["splits"]:
        print(f"  {s['split']:5d}  {s['synthetic_sessions']:14d}  "
              + "  ".join(f"{s['sessions_per_kind'][k]:6d}" for k in KINDS)
              + f"  {s['positive_windows']:7d}  {s['genuine_sessions']:8d}  {s['f1']:.3f}")
    sm = existing["summary"]
    print(f"  synthetic sessions per test set {sm['synthetic_sessions_range']}, positive windows "
          f"{sm['positive_windows_range']}, all four kinds in {sm['splits_with_all_four_kinds']}"
          f"/10, one kind only in {sm['splits_with_one_kind']}/10 {sm['one_kind_splits']}")
    print(f"  test sets lacking each kind: {sm['splits_missing_kind']}")

    print(f"\n1b. Kind-stratified session-wise evaluation "
          f"(one session per kind per fold, {N_REPEATS} seeds)")
    strat = kind_stratified(X, y, g, kind, session_rows)
    print(f"  fold mean F1 {strat['fold_mean_f1']:.3f} +- {strat['fold_std_f1']:.3f}   "
          f"AUC {strat['fold_mean_auc']:.4f} +- {strat['fold_std_auc']:.4f}")
    print(f"  pooled out-of-fold F1 {strat['pooled_f1']:.3f} {fmt_ci(strat['pooled_f1_ci95'])}   "
          f"AUC {strat['pooled_auc']:.4f} {fmt_ci(strat['pooled_auc_ci95'], 4)}  (session bootstrap 95%)")
    for k in KINDS:
        print(f"  recall {k:<11s} {strat['recall'][k]['value']:.3f} {fmt_ci(strat['recall'][k]['ci95'])}")

    print("\n2. Leave-one-kind-out threshold transfer (threshold from genuine "
          "out-of-fold training scores)")
    transfer = threshold_transfer(X, y, g, kind, te_mask)
    for name in ["fixed_0.5"] + [f"budget_{b:g}" for b in BUDGETS]:
        op = transfer[name]
        print(f"  {name}: mean F1 {op['mean_f1']:.3f}")
        for k in KINDS:
            v = op["per_kind"][k]
            print(f"    {k:<11s} thr {v['threshold']:.3f}  recall {v['recall']:.3f}  "
                  f"precision {v['precision']:.3f}  F1 {v['f1']:.3f}  FPR {v['fpr']:.4f}")

    print("\n3. One-class baseline: IsolationForest on genuine training sessions only")
    iso = one_class(X, y, kind, te_mask)
    for k in KINDS:
        print(f"  held-out {k:<11s} AUC {iso[k]:.4f}   (forest LOKO AUC {transfer['forest_auc'][k]:.4f})")
    print(f"  all synthetic        AUC {iso['all_synthetic']:.4f}")

    print("\n4. Day-level grid: false alarms/genuine days, detected/automation days")
    scored = released_scores(df, data, X, y, g)
    grid = day_grid(scored)
    print(f"  {'threshold':<14s} {'t':>6s}  " + "  ".join(f"{rule_name(r):>7s}" for r in RULES))
    for name, cell in grid.items():
        print(f"  {name:<14s} {cell['threshold']:6.3f}  " + "  ".join(
            f"{c['false_alarms']:>3d}/{c['detected']:<3d}" for c in cell["rules"].values()))
    day_level = day_counts(scored, TAU, ("count", MIN_WINDOWS))
    print(f"  released policy (tau={TAU}, m>={MIN_WINDOWS}): false alarms "
          f"{day_level['false_alarms']}/{day_level['genuine_days']} Wilson 95% "
          f"{fmt_ci(day_level['false_alarms_wilson95'])}, detected {day_level['detected']}/"
          f"{day_level['automation_days']} Wilson 95% {fmt_ci(day_level['detected_wilson95'])}")

    print("\n5. Forward-in-time hold-out")
    fwd = forward_holdout(df, scored)
    print(f"  train genuine days <= {fwd['last_train_day']} ({fwd['train_days']} days, "
          f"{fwd['train_genuine_windows']} windows) + all synthetic; test {fwd['test_days']} "
          f"later days ({fwd['test_windows']} windows) and the injected days")
    print(f"  per-window FPR at 0.5 {fwd['fpr']['forward']['0.5']:.4f}, at 0.8 "
          f"{fwd['fpr']['forward']['0.8']:.4f}   (paper scoring of the same windows: "
          f"{fwd['fpr']['paper_scoring']['0.5']:.4f}, {fwd['fpr']['paper_scoring']['0.8']:.4f})")
    for key, label in (("released_rule", f"tau={TAU}, m>={MIN_WINDOWS}"),
                       ("chosen", f"{CHOSEN[0]:.1%} budget, {rule_name(CHOSEN[1])}")):
        c = fwd[key]
        if key == "chosen":
            print(f"  {label}: threshold forward {c['forward_threshold']:.3f}, "
                  f"paper scoring {c['paper_threshold']:.3f}")
        for scoring in ("forward", "paper_scoring"):
            v = c[scoring]
            print(f"  {label:<24s} {scoring:<13s} false alarms {v['false_alarms']}/"
                  f"{v['genuine_days']} {fmt_ci(v['false_alarms_wilson95'])}, detected "
                  f"{v['detected']}/{v['automation_days']} {fmt_ci(v['detected_wilson95'])}")

    result = {
        "source": os.path.relpath(os.path.abspath(path), REPO_ROOT),
        "sklearn_version": sklearn.__version__,
        "config": {"seed": SEED, "n_repeats": N_REPEATS, "n_boot": N_BOOT, "budgets": BUDGETS,
                   "fixed_thresholds": FIXED, "rules": [rule_name(r) for r in RULES],
                   "tau": TAU, "min_windows": MIN_WINDOWS,
                   "chosen": {"budget": CHOSEN[0], "rule": rule_name(CHOSEN[1])}},
        "existing_design": existing,
        "kind_stratified": strat,
        "threshold_transfer": transfer,
        "one_class_isolation_forest_auc": iso,
        "day_grid": grid,
        "day_level": day_level,
        "forward_holdout": fwd,
    }
    print()
    os.makedirs(out_dir, exist_ok=True)
    table_stratified(out_dir, strat, existing)
    table_transfer(out_dir, transfer, iso)
    table_daygrid(out_dir, grid)
    table_forward(out_dir, fwd)
    json_path = os.path.join(out_dir, "reanalysis.json")
    with open(json_path, "w") as fh:
        json.dump(tidy(result), fh, indent=2)
    print(f"  wrote {os.path.relpath(json_path, REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-analyses of the released feature table.")
    parser.add_argument("csv", nargs="?", default=os.path.join(REPO_ROOT, "data/keyspark_windows.csv"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "paper/tables"))
    args = parser.parse_args()
    main(args.csv, args.out_dir)
