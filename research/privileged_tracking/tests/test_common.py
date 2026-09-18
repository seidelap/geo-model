"""Tests for the shared metric and split helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    calibration_table,
    log_loss,
    mae,
    paired_bootstrap_delta,
    per_sample_log_loss,
    r2,
    skill_score,
)
from research.privileged_tracking.common.splits import (
    forward_split,
    group_kfold,
    out_of_fold_predictions,
)


def test_log_loss_and_brier_perfect_and_uninformative() -> None:
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.array([1, 0, 1, 0])) < 1e-4
    assert abs(log_loss(y, np.full(4, 0.5)) - np.log(2)) < 1e-9
    assert brier(y, np.full(4, 0.5)) == 0.25
    assert np.allclose(per_sample_log_loss(y, np.full(4, 0.5)), np.log(2))


def test_skill_r2_mae() -> None:
    assert skill_score(0.5, 1.0) == 0.5
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(r2(y, y) - 1.0) < 1e-12
    assert abs(r2(y, np.full(4, y.mean()))) < 1e-12
    assert mae(y, y + 1) == 1.0


def test_paired_bootstrap_delta_sign() -> None:
    a = np.full(500, 0.7)
    b = np.full(500, 0.6)
    m, lo, hi = paired_bootstrap_delta(a, b, n_boot=200)
    assert abs(m - 0.1) < 1e-12 and lo <= m <= hi


def test_calibration_table_bins() -> None:
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(float)
    rows = calibration_table(y, p, n_bins=5)
    assert len(rows) == 5 and sum(r["n"] for r in rows) == 100
    assert rows[0]["obs"] == 0.0 and rows[-1]["obs"] == 1.0


def test_group_kfold_holds_out_whole_groups() -> None:
    groups = np.repeat(np.arange(10), 7)
    seen = set()
    for tr, te in group_kfold(groups, n_splits=5, seed=1):
        assert not set(groups[tr]) & set(groups[te])
        seen |= set(te.tolist())
    assert seen == set(range(len(groups)))


def test_forward_split_is_chronological() -> None:
    weeks = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    tr, te = forward_split(weeks, test_frac=0.4)
    assert weeks[tr].max() < weeks[te].min()
    assert set(weeks[te].tolist()) == {4, 5}


def test_out_of_fold_predictions_cover_all_rows() -> None:
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = X["x"].to_numpy() * 2
    groups = np.arange(20) // 4

    def fit_predict(xtr: pd.DataFrame, ytr: np.ndarray, xte: pd.DataFrame) -> np.ndarray:
        slope = float(np.polyfit(xtr["x"], ytr, 1)[0])
        return slope * xte["x"].to_numpy()

    oof = out_of_fold_predictions(fit_predict, X, y, groups, n_splits=5)
    assert not np.isnan(oof).any()
    assert np.allclose(oof, y)
