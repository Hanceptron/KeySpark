"""Event-type check for the detector's top feature.

``iei_cv`` is the coefficient of variation of the gaps between ALL events in a
one-minute window. Genuine windows contain ``key_up`` events (each a dwell time
after its ``key_down``) and unthrottled ``scroll`` bursts; the synthetic
generator emits neither, and ``mouse_fraction`` counts scroll as mouse. This
script asks whether the class separation survives when that mismatch is removed,
by rerunning the detector evaluation under three variants:

  A  current features on all event types (must reproduce numbers.json);
  B  features recomputed from key_down, move and click events only, both classes;
  C  current features, generator also emitting a key_up 50 to 150 ms after every
     key_down (botgen.synthetic_event_frame(key_up=True)), nothing else changed.

Run from anywhere (the script changes to the repository root):

    .venv/bin/python paper/scripts/typematched.py

Session ids come from the full event stream in every variant, and every split
is fixed at the session level from variant A, so the three variants are scored
on identical train/test sessions; only the windows inside them differ. Writes
paper/tables/typematched.tex and paper/tables/typematched.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (also puts the repo root on sys.path and chdirs there)

from keyspark import botgen, ml  # noqa: E402

SEED = mt.SEED
N_REPEATS = 10                          # GroupShuffleSplit repeats and stratified seeds
KEPT_TYPES = ("key_down", "move", "click")  # event types both classes emit
OUT_TEX = os.path.join(mt.OUT_DIR, "typematched.tex")
OUT_JSON = os.path.join(mt.OUT_DIR, "typematched.json")

VARIANTS = {
    "A": "current features, all event types",
    "B": "features from key_down, move and click only, both classes",
    "C": "current features, generator emits a key_up 50-150 ms after every key_down",
}

MODELS = {
    "Majority class": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision tree (depth 3)": lambda: DecisionTreeClassifier(
        max_depth=3, class_weight="balanced", random_state=SEED),
    "Random forest (proposed)": lambda: ml._make_model(),
}
FOREST = "Random forest (proposed)"

# Same blocks as the ablation table of make_tables.py.
BLOCKS = {
    "Timing only": ["ks_iei_cv", "move_iei_cv", "iei_cv"],
    "Geometry only": ["step_mean", "step_cv"],
    "Keyboard only": ["key_diversity", "ks_iei_cv"],
    "Mouse only": ["move_iei_cv", "step_mean", "step_cv", "mouse_fraction"],
    "All seven": ml.FEATURES,
}


# ---------------------------------------------------------------------------
# Variant datasets
# ---------------------------------------------------------------------------
def featurize(ev: pd.DataFrame, types=None) -> pd.DataFrame:
    """Window features of ``ev`` (which already carries session ids from the full
    stream), restricted to ``types``, with each window's owning session taken
    from the full stream as in make_tables.windows_with_sessions."""
    feats = ml._window_features(ev, types=types)
    owner = (
        ev.assign(window_start=ev["event_time"].dt.floor("min"))
        .groupby(["user", "window_start"])["session"].first().reset_index()
    )
    return feats.merge(owner, on=["user", "window_start"], how="left")


def build_variants() -> dict[str, pd.DataFrame]:
    genuine = mt.add_sessions(mt.load_genuine(), "H")
    synth = mt.add_sessions(botgen.synthetic_event_frame(), "B")
    synth_ku = mt.add_sessions(botgen.synthetic_event_frame(key_up=True), "B")

    def labeled(h, s):
        return pd.concat([h.assign(label=0), s.assign(label=1)], ignore_index=True)

    human_all = featurize(genuine)
    synth_all = featurize(synth)
    return {
        "A": labeled(human_all, synth_all),
        "B": labeled(featurize(genuine, KEPT_TYPES), featurize(synth, KEPT_TYPES)),
        "C": labeled(human_all, featurize(synth_ku)),
    }


# ---------------------------------------------------------------------------
# Session-level designs, fixed from variant A
# ---------------------------------------------------------------------------
def gss_test_sessions(data: pd.DataFrame) -> list[set]:
    """Test sessions of the paper's GroupShuffleSplit(10, 0.25, seed 42)."""
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values
    splitter = GroupShuffleSplit(n_splits=N_REPEATS, test_size=0.25, random_state=SEED)
    return [set(g[te]) for _, te in splitter.split(X, y, groups=g)]


def stratified_folds(data: pd.DataFrame, kind_of: dict) -> tuple[int, list[dict]]:
    """Kind-stratified grouped folds, as in reanalysis_from_csv.py: each fold
    holds out one synthetic session per kind and a quarter of the genuine
    sessions; one session -> fold map per seed (SEED .. SEED + 9)."""
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
    return n_folds, maps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def mean_std(vals) -> dict:
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def fit_score(factory, X, y, tr, te) -> dict:
    model = factory()
    model.fit(X[tr], y[tr])
    return mt.score(y[te], model.predict(X[te]), model.predict_proba(X[te])[:, 1])


def gss_eval(data, factory, features, test_sets) -> dict:
    """Ten session-wise splits, test sessions fixed from variant A."""
    X, y = data[features].values, data["label"].values
    runs = []
    for test in test_sets:
        te = data["session"].isin(test).values
        runs.append(fit_score(factory, X, y, ~te, te))
    out = {k: mean_std([r[k] for r in runs])
           for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    out["first"] = runs[0]
    return out


def stratified_eval(data, factory, kind_arr, fold_maps) -> dict:
    """Fold-level mean +- std over 4 folds x 10 seeds, plus pooled out-of-fold
    F1 / AUC and per-kind recall, averaged over the seeds."""
    X, y = data[ml.FEATURES].values, data["label"].values
    fold_f1, fold_auc, pooled_f1, pooled_auc = [], [], [], []
    recall = {k: [] for k in botgen.KINDS}
    for fmap in fold_maps:
        fold = data["session"].map(fmap).values
        proba, pred = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        for f in sorted(set(fold)):
            te = fold == f
            model = factory()
            model.fit(X[~te], y[~te])
            proba[te] = model.predict_proba(X[te])[:, 1]
            pred[te] = model.predict(X[te])
            r = mt.score(y[te], pred[te], proba[te])
            fold_f1.append(r["f1"])
            fold_auc.append(r["roc_auc"])
        r = mt.score(y, pred, proba)
        pooled_f1.append(r["f1"])
        pooled_auc.append(r["roc_auc"])
        for k in botgen.KINDS:
            recall[k].append(pred[kind_arr == k].mean())
    return {
        "fold_f1": mean_std(fold_f1), "fold_roc_auc": mean_std(fold_auc),
        "pooled_f1": float(np.mean(pooled_f1)), "pooled_roc_auc": float(np.mean(pooled_auc)),
        "recall_by_kind": {k: float(np.mean(v)) for k, v in recall.items()},
        "n_folds_total": len(fold_f1),
    }


def loko_eval(data, kind_arr, first_test) -> dict:
    """Leave-one-kind-out with the genuine test partition of the first split."""
    X, y = data[ml.FEATURES].values, data["label"].values
    te_mask = data["session"].isin(first_test).values
    folds = {}
    for kind in botgen.KINDS:
        kmask = kind_arr == kind
        train = (~te_mask & (y == 0)) | ((y == 1) & ~kmask)
        test = (te_mask & (y == 0)) | kmask
        folds[kind] = fit_score(ml._make_model, X, y, train, test)
    return {"folds": folds, "mean_f1": float(np.mean([r["f1"] for r in folds.values()]))}


def perm_eval(data, first_test) -> dict:
    """F1-scored permutation importance on the held-out side of the first split."""
    X, y = data[ml.FEATURES].values, data["label"].values
    te = data["session"].isin(first_test).values
    model = ml._make_model()
    model.fit(X[~te], y[~te])
    perm = permutation_importance(model, X[te], y[te], n_repeats=20,
                                  random_state=SEED, scoring="f1")
    return {f: float(v) for f, v in zip(ml.FEATURES, perm.importances_mean)}


def describe(data: pd.DataFrame) -> dict:
    hum, syn = data[data.label == 0], data[data.label == 1]
    return {
        "windows": {"genuine": int(len(hum)), "synthetic": int(len(syn))},
        "sessions": {"genuine": int(hum["session"].nunique()),
                     "synthetic": int(syn["session"].nunique())},
        "medians": {
            "genuine": {f: float(hum[f].median()) for f in ml.FEATURES},
            "synthetic": {f: float(syn[f].median()) for f in ml.FEATURES},
        },
    }


def run_variant(name, data, kind_of, test_sets, fold_maps) -> dict:
    print(f"Variant {name}: {VARIANTS[name]}")
    kind_arr = np.array([kind_of.get(s, "") for s in data["session"].values])
    out = describe(data)
    out["description"] = VARIANTS[name]
    out["gss"] = {m: gss_eval(data, f, ml.FEATURES, test_sets) for m, f in MODELS.items()}
    out["stratified"] = {m: stratified_eval(data, f, kind_arr, fold_maps)
                         for m, f in MODELS.items()}
    out["blocks_gss"] = {b: gss_eval(data, ml._make_model, feats, test_sets)
                         for b, feats in BLOCKS.items()}
    out["loko"] = loko_eval(data, kind_arr, test_sets[0])
    out["permutation_importance"] = perm_eval(data, test_sets[0])
    w, md = out["windows"], out["medians"]
    gss, st = out["gss"][FOREST], out["stratified"][FOREST]
    print(f"  windows genuine {w['genuine']:,} synthetic {w['synthetic']:,}; "
          f"median iei_cv {md['genuine']['iei_cv']:.3f}/{md['synthetic']['iei_cv']:.3f}, "
          f"mouse_fraction {md['genuine']['mouse_fraction']:.3f}/"
          f"{md['synthetic']['mouse_fraction']:.3f}")
    print(f"  forest GSS F1 {gss['f1']['mean']:.3f} +- {gss['f1']['std']:.3f} "
          f"AUC {gss['roc_auc']['mean']:.4f}; stratified F1 {st['fold_f1']['mean']:.3f} "
          f"+- {st['fold_f1']['std']:.3f} AUC {st['fold_roc_auc']['mean']:.4f}; "
          f"LOKO mean F1 {out['loko']['mean_f1']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Reproduction check and outputs
# ---------------------------------------------------------------------------
def check_against_paper(a: dict) -> dict:
    """Variant A must match paper/tables/numbers.json exactly (same data,
    seeds, partitions, and models), or the comparison means nothing."""
    with open(os.path.join(mt.OUT_DIR, "numbers.json")) as fh:
        paper = json.load(fh)
    pairs = {
        "human_windows": (a["windows"]["genuine"], paper["class_balance"]["human"]),
        "synthetic_windows": (a["windows"]["synthetic"], paper["class_balance"]["non_human"]),
        "gss_f1_mean": (a["gss"][FOREST]["f1"]["mean"],
                        paper["splits"]["Grouped (session-wise)"]["f1"]["mean"]),
        "gss_f1_std": (a["gss"][FOREST]["f1"]["std"],
                       paper["splits"]["Grouped (session-wise)"]["f1"]["std"]),
        "gss_auc_mean": (a["gss"][FOREST]["roc_auc"]["mean"],
                         paper["splits"]["Grouped (session-wise)"]["roc_auc"]["mean"]),
        "loko_mean_f1": (a["loko"]["mean_f1"], paper["crosskind"]["mean_f1"]),
        "perm_iei_cv": (a["permutation_importance"]["iei_cv"],
                        paper["permutation_importance"]["iei_cv"]),
        **{f"model_f1:{m}": (a["gss"][m]["f1"]["mean"], paper["models"][m]["f1"]["mean"])
           for m in MODELS},
        **{f"block_f1:{b}": (a["blocks_gss"][b]["f1"]["mean"], paper["blocks"][b]["f1"]["mean"])
           for b in BLOCKS},
    }
    bad = {k: v for k, v in pairs.items() if not np.isclose(v[0], v[1], rtol=0, atol=1e-9)}
    if bad:
        raise SystemExit(f"variant A does not reproduce numbers.json: {bad}")
    print(f"  variant A reproduces numbers.json ({len(pairs)} values)")
    return {k: v[0] for k, v in pairs.items()}


def perm_rank(perm: dict, feature: str) -> int:
    return sorted(perm, key=lambda f: -perm[f]).index(feature) + 1


def write_tex(res: dict) -> None:
    v = res["variants"]

    def pm(agg, places=3):
        return rf"{agg['mean']:.{places}f} $\pm$ {agg['std']:.{places}f}"

    rows = [
        ["Windows scored, genuine / synthetic"]
        + [f"{mt.fmt(v[k]['windows']['genuine'])} / {mt.fmt(v[k]['windows']['synthetic'])}"
           for k in "ABC"],
        [r"Median \texttt{iei\_cv}, genuine / synthetic"]
        + [f"{v[k]['medians']['genuine']['iei_cv']:.3f} / "
           f"{v[k]['medians']['synthetic']['iei_cv']:.3f}" for k in "ABC"],
        [r"Median \texttt{mouse\_fraction}, gen.\ / syn."]
        + [f"{v[k]['medians']['genuine']['mouse_fraction']:.3f} / "
           f"{v[k]['medians']['synthetic']['mouse_fraction']:.3f}" for k in "ABC"],
        [r"$F_1$, ten session-wise splits"]
        + [pm(v[k]["gss"][FOREST]["f1"]) for k in "ABC"],
        [r"$F_1$, kind-stratified folds"]
        + [pm(v[k]["stratified"][FOREST]["fold_f1"]) for k in "ABC"],
        ["AUC, session-wise / kind-stratified"]
        + [f"{v[k]['gss'][FOREST]['roc_auc']['mean']:.4f} / "
           f"{v[k]['stratified'][FOREST]['fold_roc_auc']['mean']:.4f}" for k in "ABC"],
        [r"Leave-one-kind-out mean $F_1$"]
        + [f"{v[k]['loko']['mean_f1']:.3f}" for k in "ABC"],
        [r"\texttt{iei\_cv} permutation importance (rank)"]
        + [f"{v[k]['permutation_importance']['iei_cv']:.3f} "
           f"({perm_rank(v[k]['permutation_importance'], 'iei_cv')})" for k in "ABC"],
    ]
    st = res["config"]
    notes = (
        r"(A) the released features over every event type; "
        r"(B) all seven features recomputed from \texttt{key\_down}, \texttt{move}, "
        r"and \texttt{click} events only, for both classes, so neither "
        r"\texttt{key\_up} nor \texttt{scroll} enters any feature; "
        r"(C) the released features with a generator that also emits a "
        r"\texttt{key\_up} 50 to 150\,ms after every \texttt{key\_down}. "
        r"Windows with fewer than five kept events are not scored. "
        r"All variants use the same train and test sessions: ten session-wise "
        r"splits as in Table~\ref{tab:splits}, and "
        rf"{st['stratified_folds']} folds that each hold out one synthetic session "
        r"per kind and a quarter of the genuine sessions, repeated over "
        rf"{st['n_repeats']} seeds; figures are means $\pm$ one standard deviation "
        r"over splits or folds. Permutation importance is $F_1$-scored on the "
        r"first session-wise split (20 shuffles); rank among seven features."
    )
    lines = [
        "% generated by paper/scripts/typematched.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\caption{Event-type check of the detector: the forest's separation with "
        r"all event types (A), with features from event types both classes emit (B), "
        r"and with key releases added to the generator (C).}",
        r"\label{tab:typematched}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Measure & (A) All types & (B) Type-matched & (C) With \texttt{key\_up} \\",
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


def main() -> None:
    print("Loading corpus and building variants...")
    data = build_variants()
    kind_of = mt.kind_of_sessions(data["A"])
    for name in ("B", "C"):
        assert mt.kind_of_sessions(data[name]) == kind_of, f"kind map differs in {name}"
    test_sets = gss_test_sessions(data["A"])
    n_folds, fold_maps = stratified_folds(data["A"], kind_of)

    variants = {name: run_variant(name, df, kind_of, test_sets, fold_maps)
                for name, df in data.items()}
    check = check_against_paper(variants["A"])

    res = {
        "config": {
            "seed": SEED, "n_repeats": N_REPEATS, "stratified_folds": n_folds,
            "kept_types_B": list(KEPT_TYPES), "key_up_delay_s_C": list(botgen._KEY_UP_DELAY),
            "freeze": str(mt.FREEZE), "min_window_events": ml.MIN_WINDOW_EVENTS,
            "split_sessions_fixed_from": "A",
        },
        "variants_described": VARIANTS,
        "check_A_vs_numbers_json": check,
        "variants": variants,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"  wrote {OUT_JSON}")
    write_tex(res)


if __name__ == "__main__":
    main()
