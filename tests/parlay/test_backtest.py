from __future__ import annotations

import numpy as np
import pytest

from geo_model.parlay.backtest import (
    breakeven_correlation,
    correlation_summary,
    gaussian_explaining_away_corr,
    kalman_slope,
    parlay_ev,
    two_sided_parlay_roi,
)
import pandas as pd


def test_breakeven_correlation_zero_ev() -> None:
    d = 1.9090909
    rho = breakeven_correlation(d)
    assert rho == pytest.approx(0.0977, abs=1e-3)
    assert parlay_ev(rho, d) == pytest.approx(0.0, abs=1e-9)
    assert parlay_ev(0.0, d) < 0
    assert parlay_ev(0.5, d) > 0


def test_gaussian_theory_is_bounded_and_monotone() -> None:
    lat0, nxt0 = gaussian_explaining_away_corr(2.0, 13.0)
    lat1, nxt1 = gaussian_explaining_away_corr(10.0, 13.0)
    assert 0 < nxt0 < lat0 < 1
    assert nxt1 > nxt0
    # Degenerate: no game noise and equal uncertainty -> latent errors fully determined.
    assert gaussian_explaining_away_corr(5.0, 1e-6)[0] == pytest.approx(1.0, abs=1e-6)


def test_two_sided_parlay_roi_perfect_and_anti_correlation() -> None:
    d = 2.0
    x = np.array([1.0, -1.0, 2.0, -3.0])
    y = np.array([2.0, -2.0, 1.0, -1.0])
    r = two_sided_parlay_roi(x, y, d, expected_sign=1)
    assert r.n_pairs == 4 and r.hit_rate == 1.0
    assert r.profit == pytest.approx(4 * 4.0 - 8)
    assert r.roi == pytest.approx(1.0)
    r2 = two_sided_parlay_roi(x, y, d, expected_sign=-1)
    assert r2.hit_rate == 0.0 and r2.roi == pytest.approx(-1.0)
    r3 = two_sided_parlay_roi(np.array([0.0, 1.0]), np.array([1.0, 1.0]), d)
    assert r3.n_pairs == 1  # push refunded


def test_correlation_summary_recovers_correlation() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000)
    x = z + rng.normal(size=4000)
    y = z + rng.normal(size=4000)
    s = correlation_summary(x, y, n_boot=200)
    assert s.pearson == pytest.approx(0.5, abs=0.05)
    assert s.pearson_ci[0] < 0.5 < s.pearson_ci[1]
    assert s.p_both_cover > s.p_both_indep
    assert s.phi > 0.2
    assert s.n == 4000


def test_kalman_slope_recovers_unit_slope() -> None:
    rng = np.random.default_rng(1)
    cov = rng.uniform(-1, 1, 5000)
    prod = cov + rng.normal(0, 0.5, 5000)
    slope, se, p = kalman_slope(pd.DataFrame({"pred_cov": cov, "resid_1": prod, "resid_2": np.ones(5000)}))
    assert slope == pytest.approx(1.0, abs=0.05)
    assert p < 1e-6
