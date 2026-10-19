"""KeySpark continuous verification self-model. Enrolls ONE user from their own
per-window features and scores each later window by how far it sits from that
self-model. This is one-class by construction: enrollment sees only the genuine
user, never an impostor, matching real deployment where no impostor data exists on
day one.

Primary detector (pre-specified before any impostor data, so the headline is not
selected on the test set): a robust scaled-Manhattan anomaly score. The template
is the per-feature median and MAD over enrollment windows; a window's score is the
mean, over the features actually present in that window, of the winsorized robust
z-distance |x - median| / MAD. Averaging over present features is what lets a
mouse-only or keyboard-only window score at all (the absent block is NaN and
skipped) instead of being thrown away. This is the window-level cousin of
Killourhy & Maxion's best-performing fixed-text detector; it is deliberately
simple and interpretable, since the thesis contribution is the verification
decision, not a new anomaly algorithm.

Higher score = less like the enrolled user. A threshold calibrated on held-out
enrollment windows at a target false-reject rate turns the score into a per-window
accept/reject; the multi-window decision policy (keyspark.policy, later) turns a
stream of those into a verification decision.

  uv run python -m keyspark.verify enroll    # fit + persist the self-model
  uv run python -m keyspark.verify evaluate   # genuine held-out score distribution + FRR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from keyspark import events_io, features, hardening

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
MODEL_DIR = Path("output/models")
TEMPLATE_PATH = MODEL_DIR / "self_model.joblib"

Z_CAP = 10.0              # tune: winsorize robust z per feature so no single feature dominates
MAD_EPS = 1e-9            # tune: floor to keep MAD division finite
DEFAULT_TARGET_FRR = 0.10  # tune: enrollment-calibrated false-reject target

# Selectable feature sets for the E1 feature-set contrast. "counts" is added when
# the counts-only baseline is wired; today the self-model uses "v2" or "liveness".
FEATURE_SETS = {
    "liveness": features.LIVENESS_FEATURES,
    "v2": features.FEATURES,
}


# --------------------------------------------------------------------------
# The self-model
# --------------------------------------------------------------------------
class SelfModel:
    """Robust one-class template: per-feature median and floored MAD, scored by
    the mean winsorized robust z over the features present in each window.
    """

    def __init__(self, feature_set: str = "v2"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature_set {feature_set!r}; choose {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.features = list(FEATURE_SETS[feature_set])
        self.median_: pd.Series | None = None
        self.mad_: pd.Series | None = None

    def fit(self, enroll: pd.DataFrame) -> "SelfModel":
        X = enroll[self.features]
        median = X.median()                       # per feature, ignores NaN
        mad = (X - median).abs().median()
        # Floor zero/near-zero MAD (a feature constant across enrollment) to the
        # median of the non-zero MADs, so a constant feature that later deviates
        # does not explode the score; final epsilon floor keeps division finite.
        nonzero = mad[mad > 0]
        fallback = float(nonzero.median()) if len(nonzero) else 1.0
        mad = mad.where(mad > 0, other=fallback).clip(lower=MAD_EPS)
        self.median_, self.mad_ = median, mad
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("SelfModel is not fitted")
        X = df[self.features]
        z = ((X - self.median_).abs() / self.mad_).clip(upper=Z_CAP)
        return z.mean(axis=1, skipna=True).to_numpy()

    def save(self, path: Path = TEMPLATE_PATH) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"feature_set": self.feature_set, "features": self.features,
             "median": self.median_, "mad": self.mad_},
            path,
        )

    @classmethod
    def load(cls, path: Path = TEMPLATE_PATH) -> "SelfModel":
        b = joblib.load(path)
        m = cls(feature_set=b["feature_set"])
        m.features, m.median_, m.mad_ = b["features"], b["median"], b["mad"]
        return m


# --------------------------------------------------------------------------
# Threshold calibration + metrics
# --------------------------------------------------------------------------
def calibrate_threshold(genuine_scores: np.ndarray, target_frr: float = DEFAULT_TARGET_FRR) -> float:
    """Threshold at which a fraction ``target_frr`` of held-out genuine windows are
    rejected: the (1 - target_frr) quantile of their scores.
    """
    s = np.asarray(genuine_scores, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.quantile(s, 1.0 - target_frr))


def far_at(impostor_scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(impostor_scores, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.mean(s <= threshold)) if s.size else float("nan")


def frr_at(genuine_scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(genuine_scores, dtype=float)
    s = s[~np.isnan(s)]
    return float(np.mean(s > threshold)) if s.size else float("nan")


def eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold at which FAR crosses FRR, swept over the
    pooled score set. Higher score = impostor.
    """
    g = np.asarray(genuine_scores, float); g = g[~np.isnan(g)]
    im = np.asarray(impostor_scores, float); im = im[~np.isnan(im)]
    if not g.size or not im.size:
        return float("nan"), float("nan")
    grid = np.unique(np.concatenate([g, im]))
    best_gap, best = np.inf, (float("nan"), float("nan"))
    for thr in grid:
        far = np.mean(im <= thr)
        frr = np.mean(g > thr)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best = gap, ((far + frr) / 2.0, float(thr))
    return best


# --------------------------------------------------------------------------
# Genuine feature table
# --------------------------------------------------------------------------
def genuine_features(user: str | None = None) -> pd.DataFrame:
    """Featurize the genuine archive (exclusions applied) through the hardened
    featurizer, chronologically ordered. Optionally restrict to one user.
    """
    raw = events_io.exclude_nonhuman(events_io.load_events())
    if user is not None:
        raw = raw[raw["user"] == user]
    feats = features.harden_and_featurize(raw, secret=hardening.new_secret())
    return feats.sort_values("window_start").reset_index(drop=True)


def chronological_split(feats: pd.DataFrame, enroll_frac: float = 0.5,
                        enroll_n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split windows in time: earliest for enrollment, the rest held out. A fixed
    ``enroll_n`` overrides the fraction. No random split (that would leak drift).
    """
    feats = feats.sort_values("window_start").reset_index(drop=True)
    n = enroll_n if enroll_n is not None else int(len(feats) * enroll_frac)
    n = max(1, min(n, len(feats) - 1))
    return feats.iloc[:n].copy(), feats.iloc[n:].copy()


# --------------------------------------------------------------------------
# CLI actions
# --------------------------------------------------------------------------
def enroll(user: str | None = None, feature_set: str = "v2") -> dict:
    feats = genuine_features(user)
    model = SelfModel(feature_set).fit(feats)
    model.save()
    return {"template_path": str(TEMPLATE_PATH), "feature_set": feature_set,
            "enrollment_windows": int(len(feats)), "features": len(model.features)}


def evaluate(feature_set: str = "v2", enroll_frac: float = 0.5,
             target_frr: float = DEFAULT_TARGET_FRR) -> dict:
    """Genuine-only sanity evaluation: enroll on the earliest windows, score the
    held-out later windows, and report the score distribution plus the false-reject
    rate at the enrollment-calibrated threshold. This is the first figure and needs
    no impostor data. A well-behaved self-model keeps held-out FRR near target_frr.
    """
    feats = genuine_features()
    enroll_df, test_df = chronological_split(feats, enroll_frac=enroll_frac)
    model = SelfModel(feature_set).fit(enroll_df)

    enroll_scores = model.score(enroll_df)
    test_scores = model.score(test_df)
    threshold = calibrate_threshold(enroll_scores, target_frr)

    q = np.nanpercentile(test_scores, [5, 25, 50, 75, 95]).tolist()
    return {
        "feature_set": feature_set,
        "enrollment_windows": int(len(enroll_df)),
        "test_windows": int(len(test_df)),
        "target_frr": target_frr,
        "threshold": threshold,
        "enroll_score_median": float(np.nanmedian(enroll_scores)),
        "test_score_pctiles": {p: v for p, v in zip((5, 25, 50, 75, 95), q)},
        "held_out_frr": frr_at(test_scores, threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KeySpark verification self-model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_enroll = sub.add_parser("enroll", help="fit + persist the self-model on genuine windows")
    p_enroll.add_argument("--user", default=None)
    p_enroll.add_argument("--feature-set", default="v2", choices=list(FEATURE_SETS))
    p_eval = sub.add_parser("evaluate", help="genuine held-out score distribution + FRR")
    p_eval.add_argument("--feature-set", default="v2", choices=list(FEATURE_SETS))
    p_eval.add_argument("--enroll-frac", type=float, default=0.5)
    p_eval.add_argument("--target-frr", type=float, default=DEFAULT_TARGET_FRR)
    args = parser.parse_args()

    if args.cmd == "enroll":
        print(json.dumps(enroll(args.user, args.feature_set), indent=2, default=float))
    elif args.cmd == "evaluate":
        print(json.dumps(
            evaluate(args.feature_set, args.enroll_frac, args.target_frr),
            indent=2, default=float,
        ))


if __name__ == "__main__":
    main()
