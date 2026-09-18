"""Tests for the corners stage's model helpers, lines and betting machinery."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev.corners import (
    disagreement_table,
    line_from_mean,
    poisson_deviance,
    scenario_mask,
)


def test_poisson_deviance_is_zero_at_the_truth() -> None:
    y = np.array([0.0, 3.0, 10.0])
    assert poisson_deviance(y, y) == pytest.approx(np.zeros(3), abs=1e-5)


def test_poisson_deviance_is_positive_away_from_the_truth() -> None:
    d = poisson_deviance(np.array([5.0]), np.array([3.0]))
    assert d[0] > 0
    # y = 0 reduces to 2 * mu
    assert poisson_deviance(np.array([0.0]), np.array([2.0]))[0] == pytest.approx(4.0)


def test_line_from_mean_gives_half_integers_in_range() -> None:
    lines = line_from_mean(np.array([3.0, 8.2, 9.9, 20.0]))
    assert lines.tolist() == [6.5, 8.5, 9.5, 14.5]
    assert np.all((lines * 2) % 2 == 1)


def test_scenario_mask_selects_the_right_rows() -> None:
    df = pd.DataFrame({
        "p_fav": [0.70, 0.50, 0.40], "p_over25": [0.45, 0.65, 0.50],
        "p_home": [0.70, 0.20, 0.30], "elo_gap": [200.0, -50.0, 10.0],
        "proxy_total": [9.0, 10.0, 12.0],
    })
    cuts = {"proxy_q20": 9.5, "proxy_q80": 11.5}
    assert scenario_mask(df, "all", cuts).tolist() == [True, True, True]
    assert scenario_mask(df, "fav_strong", cuts).tolist() == [True, False, False]
    assert scenario_mask(df, "high_total", cuts).tolist() == [False, True, False]
    assert scenario_mask(df, "even_match", cuts).tolist() == [False, False, True]
    assert scenario_mask(df, "proxy_low", cuts).tolist() == [True, False, False]
    assert scenario_mask(df, "proxy_high", cuts).tolist() == [False, False, True]
    with pytest.raises(KeyError):
        scenario_mask(df, "nope", cuts)


def test_disagreement_table_orders_bins_by_gap_and_scores_the_better_model() -> None:
    rng = np.random.default_rng(0)
    n = 400
    p_true = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_model = p_true
    p_proxy = np.full(n, 0.5)
    tab = disagreement_table(p_model, p_proxy, y, n_bins=4)
    assert tab["n"].sum() == n
    assert tab["mean_abs_gap"].is_monotonic_increasing
    # the model is the truth, so it should win by more where the two disagree most
    assert tab["delta_proxy_minus_model"].iloc[-1] > tab["delta_proxy_minus_model"].iloc[0]
