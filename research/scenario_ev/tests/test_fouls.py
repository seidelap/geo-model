"""Unit tests for the pure scoring helpers of stage 04 (fouls)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import common as C
from research.scenario_ev import fouls as F


def test_poisson_glm1_recovers_a_known_scale_and_slope() -> None:
    rng = np.random.default_rng(0)
    mu_ref = rng.uniform(5, 20, size=20000)
    y = rng.poisson(np.exp(0.3 + 0.8 * np.log(mu_ref)))
    a, b = F.poisson_glm1(mu_ref, y)
    assert a == pytest.approx(0.3, abs=0.05)
    assert b == pytest.approx(0.8, abs=0.02)
    fitted = F.poisson_glm1_apply(mu_ref, (a, b))
    assert fitted.mean() == pytest.approx(y.mean(), rel=0.02)


def test_poisson_glm1_is_the_identity_on_a_perfect_reference() -> None:
    rng = np.random.default_rng(1)
    mu_ref = rng.uniform(5, 20, size=20000)
    y = rng.poisson(mu_ref)
    a, b = F.poisson_glm1(mu_ref, y)
    assert a == pytest.approx(0.0, abs=0.05)
    assert b == pytest.approx(1.0, abs=0.02)


def test_poisson_deviance_is_zero_for_a_perfect_fit() -> None:
    y = np.array([0.0, 3.0, 10.0])
    assert F.poisson_deviance(y, y)[1:] == pytest.approx(np.zeros(2), abs=1e-9)
    assert (F.poisson_deviance(y, y + 2.0) > 0).all()


def test_line_probabilities_decrease_with_the_line() -> None:
    mu = np.full(5, 12.0)
    p_low = F.line_probabilities(mu, 9.5, 30.0)
    p_high = F.line_probabilities(mu, 15.5, 30.0)
    assert (p_low > p_high).all()
    assert ((0 < p_low) & (p_low < 1)).all()


def test_line_probabilities_recalibration_moves_toward_the_base_rate() -> None:
    rng = np.random.default_rng(2)
    mu = rng.uniform(8, 16, size=5000)
    y = (rng.poisson(mu) > 12.5).astype(float)
    fit = np.ones(len(mu), dtype=bool)
    raw = F.line_probabilities(mu * 2.0, 12.5, 20.0)          # badly mis-scaled
    cal = F.line_probabilities(mu * 2.0, 12.5, 20.0, y, fit)
    assert abs(cal.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_count_rows_reports_a_positive_delta_for_a_better_model() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    mu = rng.uniform(8, 16, size=n)
    y = rng.poisson(mu).astype(float)
    groups = np.repeat(np.arange(n // 2), 2)
    split = np.where(np.arange(n) < n // 2, C.DISCOVERY, C.CONFIRMATION)
    ref = np.full(n, y.mean())
    rows = F.count_rows("good", y, mu, 50.0, groups, split, ref, F.FoulConfig(
        split=C.SplitConfig(n_boot=100)))
    assert len(rows) == 2
    assert all(r["delta_vs_ref"] > 0 for r in rows)
    assert all(r["ci_lo"] > 0 for r in rows)


def test_choose_threshold_prefers_a_threshold_that_places_enough_bets() -> None:
    rng = np.random.default_rng(4)
    n = 5000
    p_book = np.full(n, 0.5)
    p_model = np.clip(p_book + rng.normal(scale=0.05, size=n), 0.01, 0.99)
    y = (rng.random(n) < p_model).astype(float)
    groups = np.arange(n)
    thr = F.choose_threshold(y, p_model, p_book, groups, 0.06,
                             F.FoulConfig(split=C.SplitConfig(n_boot=50)), min_bets=50)
    assert thr in F.FoulConfig().edge_grid


def test_simulate_real_prices_prices_a_fair_book_at_zero_ev() -> None:
    rng = np.random.default_rng(5)
    n = 20000
    p = np.full(n, 0.5)
    y = (rng.random(n) < p).astype(float)
    odds = np.full(n, 2.0)  # a 0% hold, perfectly fair book
    out = F.simulate_real_prices(y, p + 1e-9, odds, odds, np.arange(n), 0.0,
                                 F.FoulConfig(split=C.SplitConfig(n_boot=200)))
    assert out["mean_hold"] == pytest.approx(0.0, abs=1e-9)
    assert abs(out["roi"]) < 0.05


def test_model_frame_coerces_categoricals() -> None:
    df = pd.DataFrame({"div_code": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
    X = F.model_frame(df, ["div_code", "x"])
    assert isinstance(X["div_code"].dtype, pd.CategoricalDtype)
    assert X["x"].dtype.kind == "f"


def _panel(n: int = 4000, seed: int = 6) -> pd.DataFrame:
    """A small chronologically split count panel with two rows per match."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "match_id": np.repeat(np.arange(n // 2), 2),
        "x": rng.normal(size=n),
    })
    df["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(df["match_id"], unit="D")
    df["y"] = rng.poisson(np.exp(0.5 + 0.4 * df["x"].to_numpy())).astype(float)
    df["split"] = C.chronological_split(df["match_id"], df["date"], 0.6)
    return df


def test_cv_and_confirm_honours_the_params_argument() -> None:
    """A capacity override must actually reach LightGBM.

    The stage-b refit loop once omitted this positional argument and silently refitted
    the grid default, so it measured the seed spread of a different model from the one
    the report quoted it against.
    """
    df = _panel()
    cfg = F.FoulConfig()
    stump = F.cv_and_confirm(df, ["x"], "y", "count", cfg,
                             dict(n_estimators=1, num_leaves=2, min_child_samples=200,
                                  learning_rate=0.05))
    default = F.cv_and_confirm(df, ["x"], "y", "count", cfg)
    assert np.isfinite(stump).all() and np.isfinite(default).all()
    assert not np.allclose(stump, default)
    # one shallow tree cannot spread predictions as widely as the grid default
    assert np.std(stump) < np.std(default)


def test_cv_and_confirm_is_deterministic_at_a_fixed_seed() -> None:
    """Same seed, same params, same predictions -- so a spread is a parameter change."""
    df = _panel()
    cfg = F.FoulConfig()
    a = F.cv_and_confirm(df, ["x"], "y", "count", cfg, seed=11)
    b = F.cv_and_confirm(df, ["x"], "y", "count", cfg, seed=11)
    assert np.allclose(a, b)
