"""Reproduce the paper's classification results from the released feature table.

    uv run python paper/scripts/reproduce_from_csv.py [data/keyspark_windows.csv]

Needs no raw events. Recomputes, with the same seeds and protocol as
make_tables.py: the session-wise hold-out of the forest (mean over ten splits),
the leave-one-kind-out folds, and the day-level policy at the released
configuration (tau = 0.8, m = 2).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from keyspark import ml  # noqa: E402

SEED = 42
KINDS = ["jiggler", "typer", "keep_awake", "clicker"]


def main(path: str) -> None:
    df = pd.read_csv(path)
    data = df[df["set"] != "injected"].reset_index(drop=True)
    X, y, g = data[ml.FEATURES].values, data["label"].values, data["session"].values

    # Session-wise hold-out, ten group shuffles.
    f1s, aucs, first_test = [], [], None
    for tr, te in GroupShuffleSplit(n_splits=10, test_size=0.25,
                                    random_state=SEED).split(X, y, groups=g):
        first_test = te if first_test is None else first_test
        m = ml._make_model().fit(X[tr], y[tr])
        f1s.append(f1_score(y[te], m.predict(X[te])))
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    print(f"session-wise F1  {np.mean(f1s):.3f} +- {np.std(f1s):.3f}   "
          f"AUC {np.mean(aucs):.4f}")

    # Leave-one-kind-out, genuine test partition = first session-wise split.
    te_mask = np.zeros(len(y), dtype=bool)
    te_mask[first_test] = True
    kind = data["kind"].fillna("").values
    fold_f1 = []
    for k in KINDS:
        km = kind == k
        train = (~te_mask & (y == 0)) | ((y == 1) & ~km)
        test = (te_mask & (y == 0)) | km
        m = ml._make_model().fit(X[train], y[train])
        f1 = f1_score(y[test], m.predict(X[test]), zero_division=0)
        auc = roc_auc_score(y[test], m.predict_proba(X[test])[:, 1])
        fold_f1.append(f1)
        print(f"held-out {k:<11s} F1 {f1:.3f}  AUC {auc:.4f}")
    print(f"leave-one-kind-out mean F1 {np.mean(fold_f1):.3f}")

    # Day-level policy: genuine windows out of fold, injected windows by the
    # forest fitted to all genuine and synthetic windows.
    proba = np.zeros(len(data))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        proba[te] = ml._make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    genuine = data[data["set"] == "genuine"].assign(proba=proba[data["set"] == "genuine"])
    injected = df[df["set"] == "injected"].copy()
    forest = ml._make_model().fit(X, y)
    injected["proba"] = forest.predict_proba(injected[ml.FEATURES].values)[:, 1]
    scored = pd.concat([genuine, injected])
    days = scored.groupby(["user", "day", "set"]).apply(
        lambda d: (d["proba"] >= 0.8).sum() >= 2, include_groups=False
    ).rename("flag").reset_index()
    gd, idays = days[days["set"] == "genuine"], days[days["set"] == "injected"]
    print(f"day policy tau=0.8 m=2: false alarms {int(gd['flag'].sum())}/{len(gd)} "
          f"genuine days, detected {int(idays['flag'].sum())}/{len(idays)} automation days")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "data/keyspark_windows.csv"))
