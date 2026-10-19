"""Evaluation through the content-free hardening transform.

The measured deployment stores raw key identity. keyspark.hardening.harden_frame
replaces it with a coarse key class and a minute-rotating keyed token, and only
key_diversity uses key identity (distinct keys per keystroke within a minute).
This script passes the genuine archive and the synthetic class through the
transform, counts distinct tokens instead of keys (ml._window_features with
key_col="key_token"), and compares every per-window feature and the
session-wise and leave-one-kind-out results with the released ones.

The token secret is fresh per run, as the transform intends; distinct counts do
not depend on it (48-bit tokens, so a collision inside one minute is negligible).

    .venv/bin/python paper/scripts/hardened.py

Writes paper/tables/hardened.tex and paper/tables/hardened.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tables as mt  # noqa: E402  (also puts the repo root on sys.path and chdirs there)

from keyspark import botgen, hardening, ml  # noqa: E402

SEED = mt.SEED
N_REPEATS = 10
OUT_TEX = os.path.join(mt.OUT_DIR, "hardened.tex")
OUT_JSON = os.path.join(mt.OUT_DIR, "hardened.json")


def windows(raw: pd.DataFrame, tag: str, secret: bytes | None) -> pd.DataFrame:
    """Released windows (secret None) or windows of the hardened frame."""
    if secret is None:
        frame, key_col = raw, "key"
    else:
        frame, key_col = hardening.harden_frame(raw, secret), "key_token"
        assert "key" not in frame.columns
    ev = mt.add_sessions(frame, tag)
    if key_col != "key":
        ev[key_col] = frame.loc[ev.index, key_col]   # add_sessions keeps the index
    feats = ml._window_features(ev, key_col=key_col)
    ev["window_start"] = ev["event_time"].dt.floor("min")
    owner = ev.groupby(["user", "window_start"])["session"].first().reset_index()
    return feats.merge(owner, on=["user", "window_start"], how="left")


def evaluate(data: pd.DataFrame, test_sets: list[set], kind_arr: np.ndarray) -> dict:
    X, y = data[ml.FEATURES].values, data["label"].values
    runs = []
    for test in test_sets:
        te = data["session"].isin(test).values
        m = ml._make_model().fit(X[~te], y[~te])
        runs.append(mt.score(y[te], m.predict(X[te]), m.predict_proba(X[te])[:, 1]))
    te0 = data["session"].isin(test_sets[0]).values
    loko = {}
    for kind in botgen.KINDS:
        kmask = kind_arr == kind
        train = (~te0 & (y == 0)) | ((y == 1) & ~kmask)
        test = (te0 & (y == 0)) | kmask
        m = ml._make_model().fit(X[train], y[train])
        loko[kind] = mt.score(y[test], m.predict(X[test]), m.predict_proba(X[test])[:, 1])
    return {
        "gss": {k: {"mean": float(np.mean([r[k] for r in runs])),
                    "std": float(np.std([r[k] for r in runs]))}
                for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        "loko": loko,
        "loko_mean_f1": float(np.mean([r["f1"] for r in loko.values()])),
    }


def main() -> None:
    print("Loading corpus...")
    genuine_raw = mt.load_genuine()
    synth_raw = botgen.synthetic_event_frame()
    secret = hardening.new_secret()

    print("Featurising released and hardened paths...")
    paths = {}
    for name, sec in (("released", None), ("hardened", secret)):
        paths[name] = pd.concat([windows(genuine_raw, "H", sec).assign(label=0),
                                 windows(synth_raw, "B", sec).assign(label=1)],
                                ignore_index=True)
    rel, hard = paths["released"], paths["hardened"]
    keys = ["user", "window_start", "label"]
    assert rel[keys + ["session"]].equals(hard[keys + ["session"]]), \
        "the transform changed the set or order of windows"

    diffs = {}
    for f in ml.FEATURES:
        d = (rel[f] - hard[f]).abs()
        diffs[f] = {"max_abs_diff": float(d.max()), "windows_differing": int((d > 0).sum()),
                    "windows_differing_genuine": int(((d > 0) & (rel.label == 0)).sum()),
                    "windows_differing_synthetic": int(((d > 0) & (rel.label == 1)).sum())}
    kd = rel["key_diversity"] - hard["key_diversity"]
    # The transform case-folds single characters before hashing; if that is the
    # whole difference, key_diversity on case-folded raw keys equals the hardened one.
    folded = pd.concat([
        windows(df.assign(key=df["key"].map(
            lambda k: k.lower() if isinstance(k, str) and len(k) == 1 else k)), tag, None)
        for df, tag in ((genuine_raw, "H"), (synth_raw, "B"))], ignore_index=True)
    fold_gap = float((folded["key_diversity"] - hard["key_diversity"]).abs().max())
    print({f: (round(v["max_abs_diff"], 6), v["windows_differing"]) for f, v in diffs.items()})

    kind_of = mt.kind_of_sessions(rel)
    kind_arr = np.array([kind_of.get(s, "") for s in rel["session"].values])
    test_sets = [set(rel["session"].values[te]) for _, te in GroupShuffleSplit(
        n_splits=N_REPEATS, test_size=0.25, random_state=SEED).split(
        rel[ml.FEATURES].values, rel["label"].values, groups=rel["session"].values)]

    print("Evaluating both paths...")
    results = {name: evaluate(df, test_sets, kind_arr) for name, df in paths.items()}
    with open(os.path.join(mt.OUT_DIR, "numbers.json")) as fh:
        paper = json.load(fh)
    r = results["released"]
    assert np.isclose(r["gss"]["f1"]["mean"],
                      paper["splits"]["Grouped (session-wise)"]["f1"]["mean"], atol=1e-12)
    assert np.isclose(r["loko_mean_f1"], paper["crosskind"]["mean_f1"], atol=1e-12)
    h = results["hardened"]
    change = {
        "gss_f1_mean": h["gss"]["f1"]["mean"] - r["gss"]["f1"]["mean"],
        "gss_f1_std": h["gss"]["f1"]["std"] - r["gss"]["f1"]["std"],
        "gss_auc_mean": h["gss"]["roc_auc"]["mean"] - r["gss"]["roc_auc"]["mean"],
        "loko_mean_f1": h["loko_mean_f1"] - r["loko_mean_f1"],
        **{f"loko_f1_{k}": h["loko"][k]["f1"] - r["loko"][k]["f1"] for k in botgen.KINDS},
        **{f"loko_auc_{k}": h["loko"][k]["roc_auc"] - r["loko"][k]["roc_auc"]
           for k in botgen.KINDS},
    }
    res = {
        "config": {"seed": SEED, "n_repeats": N_REPEATS, "token_bytes": hardening.TOKEN_BYTES,
                   "key_col_hardened": "key_token",
                   "secret": "fresh per run (hardening.new_secret), never stored"},
        "windows": {"genuine": int((rel.label == 0).sum()), "synthetic": int((rel.label == 1).sum())},
        "feature_differences": diffs,
        "key_diversity_diff_sign": {"hardened_lower": int((kd > 0).sum()),
                                    "hardened_higher": int((kd < 0).sum())},
        "max_abs_diff_vs_case_folded_raw_keys": fold_gap,
        "results": results,
        "change_hardened_minus_released": change,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  wrote {OUT_JSON}")
    write_tex(res)
    print(f"  released F1 {r['gss']['f1']['mean']:.4f} AUC {r['gss']['roc_auc']['mean']:.5f} "
          f"LOKO {r['loko_mean_f1']:.4f}")
    print(f"  hardened F1 {h['gss']['f1']['mean']:.4f} AUC {h['gss']['roc_auc']['mean']:.5f} "
          f"LOKO {h['loko_mean_f1']:.4f}")


def write_tex(res: dict) -> None:
    d = res["feature_differences"]["key_diversity"]
    others = max(v["max_abs_diff"] for f, v in res["feature_differences"].items()
                 if f != "key_diversity")
    n = res["windows"]["genuine"] + res["windows"]["synthetic"]

    def row(label, r, changed):
        return [label, changed,
                rf"{r['gss']['f1']['mean']:.3f} $\pm$ {r['gss']['f1']['std']:.3f}",
                f"{r['gss']['roc_auc']['mean']:.4f}", f"{r['loko_mean_f1']:.3f}"]

    rows = [
        row("Raw key identity (released)", res["results"]["released"], "--"),
        row("Content-free transform", res["results"]["hardened"],
            f"{mt.fmt(d['windows_differing'])} of {mt.fmt(n)}"),
    ]
    notes = (
        r"Both paths featurise the same events, sessions, and partitions; the "
        r"transformed path replaces each key by its key class and minute-rotating "
        r"token (\texttt{hardening.harden\_frame}) and counts distinct tokens for "
        rf"\texttt{{key\_diversity}}. The largest per-window difference is "
        rf"{d['max_abs_diff']:.4f} in \texttt{{key\_diversity}} "
        rf"({mt.fmt(d['windows_differing_genuine'])} genuine and "
        rf"{d['windows_differing_synthetic']} synthetic windows differ), all from case "
        rf"folding (the transform hashes \texttt{{A}} and \texttt{{a}} alike); "
        + ("every other feature is identical. " if others == 0 else
           rf"every other feature differs by at most {others:g}. ")
        + r"$F_1$ and AUC "
        r"are over the ten session-wise splits of Table~\ref{tab:splits}, LOKO over "
        r"the folds of Table~\ref{tab:crosskind}."
    )
    lines = [
        "% generated by paper/scripts/hardened.py -- do not edit by hand",
        r"\begin{table}[t]",
        r"\caption{Detection results with raw key identity and through the "
        r"content-free transform.}",
        r"\label{tab:hardened}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Key representation & Windows changed & $F_1$ & AUC & LOKO mean $F_1$ \\",
        r"\midrule",
        *[" & ".join(rw) + r" \\" for rw in rows],
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
