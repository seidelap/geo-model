from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games
from geo_model.parlay.kalman import KalmanParams, MarketErrorKalman
from geo_model.parlay.kalman_granular import (
    AuxCalibration,
    GranularKalman,
    GranularParams,
    add_point_residuals,
    attach_aux_observations,
    calibration_by_pair_type,
    decile_calibration,
    fit_aux_calibration,
    pair_type_summary,
    simulate_granular_games,
    single_shared_game_corr,
    theoretical_max_corr,
    top_fraction_by_pair_type,
)
from tests.parlay.conftest import make_schedule

SYM = GranularParams(
    prior_std_o=3.0,
    prior_std_d=3.0,
    process_std_o=0.2,
    process_std_d=0.2,
    obs_std=9.0,
    persistence=0.9,
    od_corr=0.0,
    obs_corr=0.0,
)


def _games(n_teams: int = 8, n_weeks: int = 6, n_seasons: int = 1, seed: int = 0) -> pd.DataFrame:
    """Synthetic schedule with scores made consistent with ``result`` and ``total``.

    ``make_schedule`` draws ``total`` independently of the placeholder scores, so
    the scores are re-derived here to satisfy ``home - away = result`` and
    ``home + away = total`` as in the real games file.
    """
    raw = make_schedule(n_teams, n_weeks, n_seasons, seed)
    raw["home_score"] = (raw["total"] + raw["result"]) / 2.0
    raw["away_score"] = (raw["total"] - raw["result"]) / 2.0
    return clean_games(raw, ParlayConfig(seasons=(2000, 2100)))


def test_params_roundtrip_with_and_without_aux() -> None:
    p = GranularParams(2.0, 1.5, 0.4, 0.3, 9.5, 0.8, -0.3, 0.05)
    q = GranularParams.from_vector(p.to_vector())
    for f in ("prior_std_o", "prior_std_d", "process_std_o", "process_std_d", "obs_std", "persistence", "od_corr", "obs_corr"):
        assert getattr(q, f) == pytest.approx(getattr(p, f))
    assert len(p.to_vector()) == 8
    pa = replace(p, use_aux=True, aux_loading=0.8, aux_std=7.0, aux_corr=0.4)
    assert len(pa.to_vector()) == 11
    qa = GranularParams.from_vector(pa.to_vector(), use_aux=True)
    assert qa.aux_loading == pytest.approx(0.8)
    assert qa.aux_std == pytest.approx(7.0)
    assert qa.aux_corr == pytest.approx(0.4)
    assert qa.use_aux


def test_point_residuals_split_spread_and_total() -> None:
    g = add_point_residuals(_games())
    assert np.allclose(g["home_pts_resid"] - g["away_pts_resid"], g["resid"])
    assert np.allclose(g["home_pts_resid"] + g["away_pts_resid"], g["tresid"])


def test_week1_predictions_come_from_the_prior_only() -> None:
    p = replace(SYM, obs_corr=0.2, prior_std_o=2.0, prior_std_d=3.0)
    out = GranularKalman(p).run(_games())
    w1 = out.games[out.games["week"] == 1]
    assert np.allclose(w1[["pred_mean_home", "pred_mean_away", "pred_mean_spread", "pred_mean_total"]], 0.0)
    state_var = 2 * (2.0**2 + 3.0**2)
    assert np.allclose(w1["pred_var_spread"], state_var + p.leg_noise_var("spread"))
    assert np.allclose(w1["pred_var_total"], state_var + p.leg_noise_var("total"))
    assert p.leg_noise_var("spread") == pytest.approx(2 * 81 * 0.8)
    assert p.leg_noise_var("total") == pytest.approx(2 * 81 * 1.2)
    # No shared games yet, so every cross-game covariance is zero.
    assert np.allclose(out.pairs[out.pairs["week"] == 1]["pred_cov"], 0.0)
    # After week 1 the filter has moved off the prior.
    w2 = out.games[out.games["week"] == 2]
    assert (w2["pred_mean_spread"].abs() > 0).any()
    assert set(out.pairs["pair_type"]) == {"spread-spread", "total-total", "spread-total"}


def test_symmetric_model_reduces_to_team_level_filter() -> None:
    """With o and d exchangeable, the spread block is exactly the one-state filter."""
    games = _games(n_teams=10, n_weeks=8)
    out = GranularKalman(SYM).run(games)
    team = MarketErrorKalman(
        KalmanParams(
            prior_std=np.sqrt(2) * SYM.prior_std_o,
            process_std=np.sqrt(2) * SYM.process_std_o,
            obs_std=np.sqrt(2) * SYM.obs_std,
            persistence=SYM.persistence,
        )
    ).run(games)
    ss = out.pairs[out.pairs["pair_type"] == "spread-spread"].set_index(["game_id_1", "game_id_2"])
    tl = team.pairs.set_index(["game_id_1", "game_id_2"])
    joined = ss.join(tl, rsuffix="_team", how="inner")
    assert len(joined) == len(ss) == len(tl)
    assert np.allclose(joined["pred_cov"], joined["pred_cov_team"], atol=1e-9)
    assert np.allclose(joined["pred_corr"], joined["pred_corr_team"], atol=1e-9)
    gm = out.games.set_index("game_id").join(team.games.set_index("game_id"), rsuffix="_team")
    assert np.allclose(gm["pred_mean_spread"], gm["pred_mean"], atol=1e-9)
    assert np.allclose(gm["pred_var_spread"], gm["pred_var"], atol=1e-9)
    # Spread and total observations are orthogonal here, so no cross-type covariance.
    st = out.pairs[out.pairs["pair_type"] == "spread-total"]
    assert np.allclose(st["pred_cov"], 0.0, atol=1e-9)


def test_single_shared_game_signs() -> None:
    toy = single_shared_game_corr(SYM).set_index(["leg_1", "leg_2"])["pred_corr"]
    assert toy[("spread", "spread")] > 0
    assert toy[("total", "total")] < 0
    assert toy[("spread", "spread")] == pytest.approx(-toy[("total", "total")])
    assert toy[("spread", "total")] == pytest.approx(0.0, abs=1e-12)
    # One shared game never creates spread-total covariance: the H<->A symmetry
    # cancels the two cross terms even with asymmetric o/d uncertainty.
    asym_params = replace(SYM, prior_std_o=5.0, prior_std_d=1.0)
    asym = single_shared_game_corr(asym_params).set_index(["leg_1", "leg_2"])["pred_corr"]
    assert asym[("spread", "total")] == pytest.approx(0.0, abs=1e-12)
    # Over a whole schedule (multi-step paths) o/d asymmetry does create it,
    # while a symmetric model keeps the spread and total blocks orthogonal.
    games = _games(n_teams=8, n_weeks=6, n_seasons=2)
    st_sym = GranularKalman(SYM).run(games).pairs.query("pair_type == 'spread-total'")["pred_cov"]
    st_asym = GranularKalman(asym_params).run(games).pairs.query("pair_type == 'spread-total'")["pred_cov"]
    assert np.allclose(st_sym, 0.0, atol=1e-9)
    assert st_asym.abs().max() > 1e-3
    # Every toy correlation sits below the Cauchy-Schwarz cap.
    cap = theoretical_max_corr(SYM)
    assert abs(toy[("spread", "spread")]) < cap["spread-spread"]
    assert 0 < cap["spread-spread"] < 1


def test_theoretical_max_uses_larger_of_prior_and_stationary_variance() -> None:
    small_prior = replace(SYM, prior_std_o=0.1, prior_std_d=0.1, process_std_o=2.0, process_std_d=2.0, persistence=0.5)
    # Stationary variance 4 / (1 - 0.25) = 5.33 per state dominates the prior.
    v = 4 * 4.0 / 0.75
    expected = v / (v + small_prior.leg_noise_var("spread"))
    assert theoretical_max_corr(small_prior)["spread-spread"] == pytest.approx(expected)


def test_simulation_recovers_planted_correlation() -> None:
    sched = _games(n_teams=16, n_weeks=12, n_seasons=12, seed=3)
    params = GranularParams(8.0, 8.0, 0.3, 0.3, 9.0, 0.95, 0.0, 0.0)
    sim = simulate_granular_games(sched, params, seed=1, market="efficient")
    assert sim[["home_pts_resid", "away_pts_resid", "resid", "tresid"]].notna().all().all()
    out = GranularKalman(params).run(sim)
    cal = calibration_by_pair_type(out.pairs).set_index("pair_type")
    for pt in ("spread-spread", "total-total"):
        assert cal.loc[pt, "slope"] == pytest.approx(1.0, abs=3 * cal.loc[pt, "se"] + 0.15)
    top = top_fraction_by_pair_type(out.pairs, leg_decimal=1.909, top_frac=0.1, n_boot=20).set_index("pair_type")
    assert top.loc["spread-spread", "pearson"] > 0
    assert top.loc["total-total", "pearson"] > 0  # sign-aligned, so positive when the model is right
    # Static market: no cross-game structure relative to a line that never moves.
    static = simulate_granular_games(sched, params, seed=1, market="static")
    out_s = GranularKalman(params).run(static)
    cal_s = calibration_by_pair_type(out_s.pairs).set_index("pair_type")
    assert cal_s.loc["spread-spread", "slope"] < 0.5


def test_aux_observations_sharpen_posterior_on_simulated_data() -> None:
    sched = _games(n_teams=32, n_weeks=17, n_seasons=10, seed=7)
    params = GranularParams(8.0, 8.0, 0.3, 0.3, 9.0, 0.9, 0.0, 0.0, use_aux=True, aux_loading=1.0, aux_std=2.0, aux_corr=0.2)
    sim = simulate_granular_games(sched, params, seed=2, market="efficient")
    assert sim[["home_aux_resid", "away_aux_resid"]].notna().all().all()
    # Residuals are relative to a market that saw the auxiliary data too, so the
    # matching filter is the efficient-market one (mean zero, covariance only).
    with_aux = GranularKalman(params, market_mode="efficient").run(sim, emit_pairs=False)
    without = GranularKalman(replace(params, use_aux=False), market_mode="efficient").run(sim, emit_pairs=False)
    assert with_aux.mean_leg_state_var["spread"] < 0.7 * without.mean_leg_state_var["spread"]
    assert len(with_aux.games) == len(without.games)
    # The aux-aware filter's predictive variances are calibrated (mean z^2 near 1,
    # se about 0.03 here); the aux-blind filter overstates the variance.
    z2_with = float((with_aux.games["resid"] ** 2 / with_aux.games["pred_var_spread"]).mean())
    z2_without = float((without.games["resid"] ** 2 / without.games["pred_var_spread"]).mean())
    assert abs(z2_with - 1.0) < 0.08
    assert z2_without < z2_with - 0.05


def test_efficient_mode_keeps_mean_zero_and_same_covariances() -> None:
    games = _games(n_teams=8, n_weeks=6, n_seasons=2)
    static = GranularKalman(SYM, market_mode="static").run(games)
    eff = GranularKalman(SYM, market_mode="efficient").run(games)
    assert np.allclose(eff.games[["pred_mean_home", "pred_mean_away", "pred_mean_spread", "pred_mean_total"]], 0.0)
    assert (static.games["pred_mean_spread"].abs() > 0).any()
    assert np.allclose(static.games["pred_var_spread"], eff.games["pred_var_spread"])
    assert np.allclose(static.pairs["pred_cov"], eff.pairs["pred_cov"])
    with pytest.raises(ValueError):
        GranularKalman(SYM, market_mode="bogus")


def test_fit_runs_and_does_not_worsen_likelihood() -> None:
    games = _games(n_teams=8, n_weeks=6, n_seasons=2)
    start = GranularParams(2.0, 2.0, 0.5, 0.5, 8.0, 0.8, 0.0, 0.0)
    kf = GranularKalman(start)
    before = GranularKalman(start).run(games, emit_pairs=False).log_likelihood
    fitted = kf.fit(games, max_iter=30)
    after = GranularKalman(fitted).run(games, emit_pairs=False).log_likelihood
    assert after >= before - 1e-6
    assert 0 < fitted.persistence < 1
    assert -1 < fitted.od_corr < 1


def test_attach_aux_observations_scales_and_aliases() -> None:
    games = _games(n_teams=4, n_weeks=2)
    games["home_team"] = games["home_team"].replace({"T0": "OAK"})
    games["away_team"] = games["away_team"].replace({"T0": "OAK"})
    implied_home = (games["total_line"] + games["spread_line"]) / 2
    implied_away = (games["total_line"] - games["spread_line"]) / 2
    calib = AuxCalibration(intercept=-20.0, slope=1.5, n=100)
    rows = []
    for _, r in games.iterrows():
        h = "LV" if r["home_team"] == "OAK" else r["home_team"]  # stats file uses current abbreviations
        a = "LV" if r["away_team"] == "OAK" else r["away_team"]
        rows.append({"game_id": r["game_id"], "team": h, "off_epa": -20.0 + 1.5 * (r["home_score"])})
        rows.append({"game_id": r["game_id"], "team": a, "off_epa": -20.0 + 1.5 * (r["away_score"])})
    aux = pd.DataFrame(rows)
    out = attach_aux_observations(games, aux, calib)
    assert len(out) == len(games)
    assert np.allclose(out["home_aux_resid"], games["home_score"] - implied_home)
    assert np.allclose(out["away_aux_resid"], games["away_score"] - implied_away)
    # Games with a missing auxiliary value are dropped.
    out2 = attach_aux_observations(games, aux.iloc[2:], calib)
    assert len(out2) == len(games) - 1
    cal = fit_aux_calibration(pd.Series([1.0, 3.0, 5.0]), pd.Series([0.0, 1.0, 2.0]))
    assert cal.slope == pytest.approx(2.0)
    assert cal.intercept == pytest.approx(1.0)


def test_summary_helpers_shapes() -> None:
    out = GranularKalman(replace(SYM, od_corr=0.3)).run(_games(n_teams=8, n_weeks=6, n_seasons=2))
    summ = pair_type_summary(out.pairs)
    assert list(summ["pair_type"]) == ["spread-spread", "total-total", "spread-total"]
    assert (summ["max_abs"] <= 1).all()
    top = top_fraction_by_pair_type(out.pairs, 1.909, top_frac=0.2, n_boot=10)
    n_ss = (out.pairs["pair_type"] == "spread-spread").sum()
    assert abs(top.set_index("pair_type").loc["spread-spread", "n"] - 0.2 * n_ss) <= 0.05 * n_ss + 2
    dec = decile_calibration(out.pairs[out.pairs["pair_type"] == "spread-spread"], n_quantiles=4, n_boot=10)
    assert len(dec) == 4
    assert dec["n"].sum() == n_ss
