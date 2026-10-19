"""Invariants for the verification self-model and its metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from keyspark import verify
from keyspark.verify import SelfModel
from tests._run import approx, run

FEATS = verify.FEATURE_SETS["v2"]


def _frame(n, center=0.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(center, scale, n) for f in FEATS}
    df = pd.DataFrame(data)
    df["window_start"] = pd.date_range("2026-06-01", periods=n, freq="min")
    return df


def test_center_scores_lower_than_outlier():
    model = SelfModel("v2").fit(_frame(200, center=0.0, scale=1.0, seed=1))
    center = _frame(1, center=0.0, scale=0.0, seed=2)       # exactly at feature means ~0
    outlier = pd.DataFrame({f: [50.0] for f in FEATS})      # far from template
    outlier["window_start"] = pd.Timestamp("2026-06-02")
    assert model.score(center)[0] < model.score(outlier)[0]


def test_nan_features_are_skipped_not_fatal():
    model = SelfModel("v2").fit(_frame(200, seed=3))
    row = _frame(1, center=5.0, scale=0.0, seed=4)
    # blank the whole mouse block; cross-modal + keyboard remain -> still a score
    for f in verify.features.FEATURE_BLOCKS["mouse"]:
        row[f] = np.nan
    s = model.score(row)[0]
    assert not np.isnan(s)


def test_threshold_gives_target_frr_on_its_own_distribution():
    model = SelfModel("v2").fit(_frame(400, seed=5))
    scores = model.score(_frame(400, seed=6))
    thr = verify.calibrate_threshold(scores, target_frr=0.10)
    assert approx(verify.frr_at(scores, thr), 0.10, 0.03)


def test_eer_low_when_separated_high_when_overlapping():
    genuine = np.random.default_rng(7).normal(1.0, 0.2, 500)
    impostor_far = np.random.default_rng(8).normal(5.0, 0.2, 500)
    impostor_same = np.random.default_rng(9).normal(1.0, 0.2, 500)
    e_far, _ = verify.eer(genuine, impostor_far)
    e_same, _ = verify.eer(genuine, impostor_same)
    assert e_far < 0.05
    assert e_same > 0.35


def test_far_counts_impostors_below_threshold():
    # impostors that all score below threshold are all falsely accepted
    assert approx(verify.far_at(np.array([1.0, 1.5, 2.0]), threshold=3.0), 1.0, 1e-9)
    assert approx(verify.far_at(np.array([4.0, 5.0]), threshold=3.0), 0.0, 1e-9)


def test_chronological_split_preserves_order():
    feats = _frame(100, seed=10)
    enroll, test = verify.chronological_split(feats, enroll_frac=0.6)
    assert len(enroll) == 60 and len(test) == 40
    assert enroll["window_start"].max() <= test["window_start"].min()


if __name__ == "__main__":
    run(globals())
