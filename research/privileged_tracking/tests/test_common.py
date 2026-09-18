"""Tests for the shared metric and split helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    calibration_table,
    clustered_bootstrap_delta,
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


def test_clustered_bootstrap_delta_widens_with_within_group_correlation() -> None:
    rng = np.random.default_rng(3)
    n_groups, per_group = 40, 50
    groups = np.repeat(np.arange(n_groups), per_group)
    # per-sample deltas that are constant within a group: the play-level bootstrap is far too narrow
    d = np.repeat(rng.normal(0.1, 1.0, n_groups), per_group)
    a, b = d + 1.0, np.full(len(d), 1.0)
    m_c, lo_c, hi_c = clustered_bootstrap_delta(a, b, groups, n_boot=400)
    m_p, lo_p, hi_p = paired_bootstrap_delta(a, b, n_boot=400)
    assert abs(m_c - d.mean()) < 1e-12 and abs(m_p - m_c) < 1e-12
    assert lo_c <= m_c <= hi_c
    assert (hi_c - lo_c) > 3 * (hi_p - lo_p)
    # independent samples: the two intervals agree in width to within a factor of ~1.5
    d2 = rng.normal(0.1, 1.0, len(d))
    _, lo_c2, hi_c2 = clustered_bootstrap_delta(d2, np.zeros(len(d2)), groups, n_boot=400)
    _, lo_p2, hi_p2 = paired_bootstrap_delta(d2, np.zeros(len(d2)), n_boot=400)
    assert 0.6 < (hi_c2 - lo_c2) / (hi_p2 - lo_p2) < 1.6


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


def test_md_table_formats_ints_floats_and_nan() -> None:
    from research.privileged_tracking.common.report import md_table

    df = pd.DataFrame({"model": ["a", "b"], "n": [3, 4], "plays": [10.0, 20.0], "log_loss": [0.5, np.nan]})
    out = md_table(df, floatfmt="{:.2f}")
    lines = out.split("\n")
    assert lines[0] == "| model | n | plays | log_loss |" and lines[1] == "|---|---|---|---|"
    assert lines[2] == "| a | 3 | 10 | 0.50 |" and lines[3] == "| b | 4 | 20 |  |"
