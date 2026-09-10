from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from geo_model.parlay import common_factor as cf
from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games
from tests.parlay.conftest import make_schedule

WIDE = ParlayConfig(seasons=(2000, 2100))


def _sim(
    factor_std: float, n_seasons: int, seed: int, obs_std: float = 13.0, persistence: float = 0.9, season_persistence: float = 0.5
) -> pd.DataFrame:
    """Synthetic schedule with a stationary shared factor of the given size."""
    sched = clean_games(make_schedule(n_teams=16, n_weeks=17, n_seasons=n_seasons, seed=seed), WIDE)
    params = cf.ScalarFactorParams.stationary(factor_std, obs_std, mean=0.0, persistence=persistence, season_persistence=season_persistence)
    return cf.simulate_common_factor(sched, params, seed=seed, value_col="tresid")


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def test_kickoff_window_maps_hours() -> None:
    s = pd.Series(["13:00", "16:25", "20:20", None, "09:30"])
    out = cf.kickoff_window(s, cf.CommonFactorConfig().kickoff_windows)
    assert out.tolist()[:3] == ["early", "late", "night"]
    assert pd.isna(out.iloc[3])
    assert out.iloc[4] == "early"


def test_add_context_joins_and_labels(raw_schedule: pd.DataFrame) -> None:
    games = clean_games(raw_schedule, ParlayConfig(seasons=(2000, 2001)))
    ctx = pd.DataFrame({"game_id": games["game_id"], "gametime": "13:00", "weekday": "Sunday"})
    out = cf.add_context(games, ctx, cf.CommonFactorConfig())
    assert len(out) == len(games)
    assert (out["kickoff_window"] == "early").all()
    assert out["gameday_str"].iloc[0] == games["gameday"].iloc[0].strftime("%Y-%m-%d")
    assert "kickoff_window" not in cf.add_context(games, None, cf.CommonFactorConfig()).columns


# ---------------------------------------------------------------------------
# ICC
# ---------------------------------------------------------------------------


def test_icc_recovers_simulated_shared_factor() -> None:
    # Low persistence so the slate effects are close to independent: the
    # within-season permutation test then has power, and the realized
    # between-slate variance of the factor is close to its stationary value.
    sim = _sim(factor_std=4.0, n_seasons=30, seed=0, persistence=0.3, season_persistence=0.3)
    r = cf.icc_with_tests(sim, "tresid", ("season", "week"), n_permutations=200, n_boot=200, seed=0)
    realized_var = float(sim.groupby(["season", "week"])["true_offset"].first().var())
    expected = realized_var / (realized_var + 169.0)
    assert r.n_groups == 30 * 17
    assert abs(expected - 16.0 / 185.0) < 0.03
    assert abs(r.icc - expected) < 0.03
    assert abs(r.pairwise_corr - r.icc) < 0.02
    assert r.perm_p < 0.01
    assert r.boot_ci[0] < expected < r.boot_ci[1]
    assert r.f_p < 0.01


def test_icc_with_persistent_factor_tracks_realized_path() -> None:
    # A persistent factor (0.9) has few effective draws; the ICC estimates the
    # realized clustering, which can differ from the stationary value.
    sim = _sim(factor_std=4.0, n_seasons=30, seed=0)
    r = cf.icc_with_tests(sim, "tresid", ("season", "week"), n_permutations=50, n_boot=50, seed=0)
    realized_var = float(sim.groupby(["season", "week"])["true_offset"].first().var())
    assert abs(r.icc - realized_var / (realized_var + 169.0)) < 0.03
    assert r.f_p < 0.01


def test_icc_null_is_near_zero() -> None:
    sim = _sim(factor_std=0.0, n_seasons=40, seed=2)
    r = cf.icc_with_tests(sim, "tresid", ("season", "week"), n_permutations=200, n_boot=200, seed=0)
    assert abs(r.icc) < 0.03
    assert r.boot_ci[0] < 0.0 < r.boot_ci[1]
    assert r.perm_p > 0.05


def test_icc_oneway_drops_small_groups_and_nans() -> None:
    vals = np.array([1.0, 2.0, np.nan, 5.0, 6.0, 7.0, 9.0])
    codes = np.array([0, 0, 0, 1, 1, 2, 3])
    r = cf.icc_oneway(vals, codes, min_group_size=2)
    assert r.n == 4 and r.n_groups == 2
    assert r.n_pairs == 2


# ---------------------------------------------------------------------------
# Same-slate pairs and parlay statistics
# ---------------------------------------------------------------------------


def _slates() -> pd.DataFrame:
    rows = []
    for week, n in ((1, 3), (2, 4)):
        for i in range(n):
            rows.append({"game_id": f"g{week}{i}", "season": 2000, "week": week, "v": float(i + 1) * (1 if i % 2 else -1)})
    return pd.DataFrame(rows)


def test_same_slate_pairs_enumerates_within_slate_pairs() -> None:
    pairs = cf.same_slate_pairs(_slates(), "v")
    assert len(pairs) == 3 + 6
    assert pairs["slate"].nunique() == 2
    assert (pairs["game_id_1"] != pairs["game_id_2"]).all()


def test_pair_stats_perfect_correlation_and_two_sided_roi() -> None:
    pairs = cf.same_slate_pairs(_slates(), "v")
    pairs["y"] = pairs["x"]
    d = 1.909
    ps = cf.pair_stats(pairs, d, n_boot=50)
    assert ps.n_pairs == 9 and ps.n_slates == 2
    assert ps.pearson == pytest.approx(1.0)
    assert ps.phi == pytest.approx(1.0)
    assert ps.same_sign == pytest.approx(1.0)
    assert ps.two_sided.roi == pytest.approx(d**2 / 2 - 1)


def test_pair_stats_symmetric_orderings() -> None:
    pairs = cf.same_slate_pairs(_slates(), "v")
    swapped = pairs.rename(columns={"x": "y", "y": "x"})
    a, b = cf.pair_stats(pairs, 1.909, n_boot=0), cf.pair_stats(swapped, 1.909, n_boot=0)
    assert a.pearson == pytest.approx(b.pearson)
    assert a.phi == pytest.approx(b.phi)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_frame() -> pd.DataFrame:
    vals = {1: [2.0, 4.0], 2: [0.0, 2.0], 3: [6.0, 6.0], 4: [1.0, 1.0], 5: [0.0, 0.0]}
    rows = [{"season": 2000, "week": w, "game_id": f"a{w}{i}", "tresid": v} for w, vs in vals.items() for i, v in enumerate(vs)]
    rows += [{"season": 2001, "week": w, "game_id": f"b{w}", "tresid": 10.0} for w in (1, 2)]
    return pd.DataFrame(rows)


def test_prior_window_mean_is_causal() -> None:
    g = _persist_frame()
    k2 = cf.prior_window_mean(g, "tresid", 2)
    by_week = g.assign(p=k2).groupby(["season", "week"])["p"].first()
    assert np.isnan(by_week[(2000, 1)]) and np.isnan(by_week[(2000, 2)])
    assert by_week[(2000, 3)] == pytest.approx(2.0)
    assert by_week[(2000, 4)] == pytest.approx(3.5)
    season = cf.prior_window_mean(g, "tresid", "season")
    assert np.isnan(season[g["week"].eq(1) & g["season"].eq(2000)]).all()
    assert season[(g["season"] == 2000) & (g["week"] == 4)].iloc[0] == pytest.approx(20.0 / 6)
    prev = cf.prior_window_mean(g, "tresid", "prev_season")
    assert prev[g["season"] == 2001].iloc[0] == pytest.approx(2.2)
    assert np.isnan(prev[g["season"] == 2000]).all()
    # Perturbing the current week's residuals must not move its own prediction.
    g2 = g.copy()
    g2.loc[(g2["season"] == 2000) & (g2["week"] == 3), "tresid"] = 100.0
    assert cf.prior_window_mean(g2, "tresid", 2)[(g2["season"] == 2000) & (g2["week"] == 3)].iloc[0] == pytest.approx(2.0)


def test_ols_clustered_recovers_slope() -> None:
    rng = np.random.default_rng(0)
    x = np.repeat(rng.normal(size=60), 5)
    y = 1.0 + 2.0 * x + rng.normal(size=300)
    cl = np.repeat(np.arange(60), 5)
    res = cf.ols_clustered(x, y, cl)
    assert res.slope == pytest.approx(2.0, abs=0.15)
    assert res.n == 300 and res.n_clusters == 60
    assert res.se > 0 and res.p < 1e-6


def test_persistence_table_columns_and_null() -> None:
    sim = _sim(factor_std=0.0, n_seasons=10, seed=4)
    tab = cf.persistence_table(sim, "tresid", (1, 4, "season"))
    assert list(tab["window"]) == ["1", "4", "season"]
    assert (tab["n_weeks"] > 0).all()
    assert (tab["p"] > 0.01).all()


# ---------------------------------------------------------------------------
# Scalar Kalman filter
# ---------------------------------------------------------------------------


def test_params_roundtrip_and_stationary() -> None:
    p = cf.ScalarFactorParams(2.0, 0.7, 1.5, 9.0, 0.8, 0.3, 0.5)
    q = cf.ScalarFactorParams.from_vector(p.to_vector())
    for name in ("prior_std", "process_std", "season_std", "obs_std", "persistence", "season_persistence", "mean"):
        assert getattr(q, name) == pytest.approx(getattr(p, name))
    s = cf.ScalarFactorParams.stationary(3.0, 13.0, persistence=0.9, season_persistence=0.5)
    assert s.stationary_std == pytest.approx(3.0)
    assert s.pair_corr(9.0) == pytest.approx(9.0 / (9.0 + 169.0))


def test_filter_likelihood_matches_dense_gaussian() -> None:
    rng = np.random.default_rng(1)
    rows = [
        {"game_id": f"{s}_{w}_{g}", "season": s, "week": w, "tresid": rng.normal(0, 10)}
        for s in (2000, 2001)
        for w in (1, 2, 3)
        for g in range(4)
    ]
    df = pd.DataFrame(rows)
    p = cf.ScalarFactorParams(2.0, 0.7, 1.5, 9.0, 0.8, 0.3, 0.5)
    _, _, post_m, post_P, ll = cf._filter_core(cf.slate_stats(df, "tresid"), p)
    m, P, ll_dense, prev = p.mean, p.prior_std**2, 0.0, None
    for (season, _), sg in df.groupby(["season", "week"], sort=True):
        if prev is not None:
            rho, q2 = (p.season_persistence, p.season_std**2) if season != prev else (p.persistence, p.process_std**2)
            m, P = p.mean + rho * (m - p.mean), rho * rho * P + q2
        prev = season
        y = sg["tresid"].to_numpy()
        n = len(y)
        cov = P * np.ones((n, n)) + p.obs_std**2 * np.eye(n)
        ll_dense += multivariate_normal(mean=m * np.ones(n), cov=cov).logpdf(y)
        k = P * np.ones(n) @ np.linalg.inv(cov)
        m, P = m + k @ (y - m), P - k @ (P * np.ones(n))
    assert ll == pytest.approx(ll_dense, abs=1e-8)
    assert post_m[-1] == pytest.approx(m) and post_P[-1] == pytest.approx(P)


def test_filter_is_causal() -> None:
    sim = _sim(factor_std=3.0, n_seasons=2, seed=0)
    kf = cf.ScalarFactorKalman(cf.ScalarFactorParams.stationary(3.0, 13.0))
    a = kf.run(sim).slates
    sim2 = sim.copy()
    sim2.loc[(sim2["season"] == 2000) & (sim2["week"] == 5), "tresid"] += 50.0
    b = kf.run(sim2).slates
    # Slates are chronological; everything up to and including (2000, week 5)
    # must be untouched, everything after it must move.
    cut = int(a.index[(a["season"] == 2000) & (a["week"] == 5)][0])
    before = a.index <= cut
    assert np.allclose(a.loc[before, "pred_mean"], b.loc[before, "pred_mean"])
    assert np.allclose(a["state_var"], b["state_var"])  # variances never depend on values
    assert not np.allclose(a.loc[~before, "pred_mean"], b.loc[~before, "pred_mean"])
    assert a.loc[0, "pred_mean"] == pytest.approx(0.0) and a.loc[0, "state_var"] == pytest.approx(9.0)
    assert a.loc[0, "pred_pair_corr"] == pytest.approx(9.0 / 178.0)


def test_run_games_align_with_slates() -> None:
    sim = _sim(factor_std=2.0, n_seasons=1, seed=0)
    out = cf.ScalarFactorKalman().run(sim)
    assert len(out.games) == len(sim)
    merged = out.games.merge(out.slates[["season", "week", "state_var"]], on=["season", "week"], suffixes=("", "_s"))
    assert np.allclose(merged["state_var"], merged["state_var_s"])
    assert not out.factor_collapsed()


def test_fit_recovers_factor_size_and_beats_true_params() -> None:
    sim = _sim(factor_std=4.0, n_seasons=20, seed=7)
    truth = cf.ScalarFactorParams.stationary(4.0, 13.0, persistence=0.9, season_persistence=0.5)
    kf = cf.ScalarFactorKalman(value_col="tresid")
    fitted = kf.fit(sim, max_iter=1500, n_starts=2)
    assert 2.0 < fitted.stationary_std < 7.0
    assert 11.0 < fitted.obs_std < 15.0
    ll_fit = kf.run(sim).log_likelihood
    ll_true = cf.ScalarFactorKalman(truth).run(sim).log_likelihood
    assert ll_fit >= ll_true - 1e-6
    cov_slope, _ = cf.kalman_pair_calibration(kf.run(sim))
    assert cov_slope.slope == pytest.approx(1.0, abs=3 * cov_slope.se + 0.2)


def test_collapsed_factor_disables_calibration() -> None:
    sim = _sim(factor_std=0.0, n_seasons=1, seed=0)
    kf = cf.ScalarFactorKalman(cf.ScalarFactorParams(1e-9, 1e-9, 1e-9, 13.0, 0.5, 0.5, 0.0))
    out = kf.run(sim)
    assert out.factor_collapsed()
    cov_slope, mean_slope = cf.kalman_pair_calibration(out)
    assert np.isnan(cov_slope.slope) and mean_slope.n == len(sim)


def test_icc_by_uncertainty_bins() -> None:
    sim = _sim(factor_std=3.0, n_seasons=4, seed=1)
    out = cf.ScalarFactorKalman(cf.ScalarFactorParams.stationary(3.0, 13.0)).run(sim)
    tab = cf.icc_by_uncertainty(out, n_bins=4)
    assert len(tab) == 4
    assert tab["games"].sum() == len(sim)
    assert tab["pred_pair_corr"].is_monotonic_increasing


def test_factor_profile_and_upper_bound() -> None:
    grid = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
    with_factor = cf.factor_profile(_sim(3.0, 20, seed=3), "tresid", grid)
    assert with_factor.loc[0, "delta_ll"] == 0.0
    best = with_factor.loc[with_factor["log_likelihood"].idxmax(), "factor_std"]
    assert 1.0 <= best <= 4.0
    assert cf.profile_upper_bound(with_factor) > 2.0
    null = cf.factor_profile(_sim(0.0, 20, seed=5), "tresid", grid)
    assert null["delta_ll"].max() < 1.92
    assert cf.profile_upper_bound(null) < 2.0
    assert (with_factor["pair_corr_filtered"] <= with_factor["pair_corr_stationary"] + 1e-12).all()


# ---------------------------------------------------------------------------
# Directional strategy and strata
# ---------------------------------------------------------------------------


def test_directional_strategy_hand_computed() -> None:
    g = pd.DataFrame(
        {
            "season": 2000,
            "week": [1, 1, 1, 2, 2],
            "sig": [1.0, 1.0, 1.0, -1.0, -1.0],
            "v": [3.0, 5.0, -2.0, -4.0, 0.0],
        }
    )
    d = 2.0
    r = cf.directional_strategy(g, "sig", "v", d, n_boot=20)
    assert r.n_slates == 2
    assert r.n_singles == 4 and r.singles_win_rate == pytest.approx(0.75)
    assert r.singles_roi == pytest.approx((3 * (d - 1) - 1) / 4)
    assert r.n_parlays == 3 and r.parlays_win_rate == pytest.approx(1 / 3)
    assert r.parlays_roi == pytest.approx(((d**2 - 1) - 2) / 3)


def test_directional_strategy_skips_zero_signal() -> None:
    g = pd.DataFrame({"season": 2000, "week": [1, 1], "sig": [0.0, np.nan], "v": [1.0, 2.0]})
    r = cf.directional_strategy(g, "sig", "v", 1.909, n_boot=5)
    assert r.n_slates == 0 and r.n_singles == 0


def test_stratum_masks_labels() -> None:
    g = pd.DataFrame({"season": [2004, 2010, 2020, 2020], "week": [1, 3, 5, 10]})
    masks = cf.stratum_masks(g, cf.CommonFactorConfig())
    assert masks["all 2010-2020"].tolist() == [False, True, True, True]
    assert masks["weeks 1-4"].tolist() == [False, True, False, False]
    assert masks["weeks 5+"].tolist() == [False, False, True, True]
    assert masks["2004 illegal-contact emphasis (in-sample)"].tolist() == [True, False, False, False]
    assert masks["2020 no fans"].tolist() == [False, False, True, True]
