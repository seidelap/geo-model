from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.backtest import correlation_summary, gaussian_explaining_away_corr, kalman_slope
from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games
from geo_model.parlay.kalman import KalmanParams, MarketErrorKalman, simulate_games
from geo_model.parlay.pairs import build_shared_game_pairs
from tests.parlay.conftest import make_schedule


def test_params_roundtrip() -> None:
    p = KalmanParams(3.0, 0.4, 12.0, 0.8)
    q = KalmanParams.from_vector(p.to_vector())
    assert q.prior_std == pytest.approx(3.0)
    assert q.persistence == pytest.approx(0.8)


def test_filter_is_causal_and_shared_game_pairs_have_positive_cov() -> None:
    games = clean_games(make_schedule(n_teams=8, n_weeks=6, n_seasons=1), ParlayConfig(seasons=(2000, 2001)))
    out = MarketErrorKalman(KalmanParams(3.0, 0.1, 13.0, 1.0)).run(games)
    # First-week predictions come from the prior only: mean 0, identical variance.
    w1 = out.games[out.games["week"] == 1]
    assert np.allclose(w1["pred_mean"], 0.0)
    assert np.allclose(w1["pred_var"], 2 * 9.0 + 169.0)
    # Week-1 pairs are uncorrelated a priori; after week 1, next games of two
    # teams that just met have positive predicted covariance.
    assert np.allclose(out.pairs[out.pairs["week"] == 1]["pred_cov"], 0.0)
    tg = to_team_games(games)
    sp = build_shared_game_pairs(tg)
    sp = sp[sp["week"] == 1]
    key = out.pairs.set_index(["game_id_1", "game_id_2"])["pred_cov"]
    for _, r in sp.iterrows():
        a, b = sorted([r["h_next_game_id"], r["a_next_game_id"]], key=lambda g: list(games["game_id"]).index(g))
        cov = key.get((a, b), key.get((b, a)))
        # Sign depends on which teams are home in the next games.
        nxt = games.set_index("game_id")
        s_h = 1 if nxt.loc[r["h_next_game_id"], "home_team"] == r["team_h"] else -1
        s_a = 1 if nxt.loc[r["a_next_game_id"], "home_team"] == r["team_a"] else -1
        assert s_h * s_a * cov > 0


@pytest.mark.slow
def test_efficient_market_simulation_matches_theory() -> None:
    sched = clean_games(make_schedule(n_teams=32, n_weeks=17, n_seasons=40, seed=3), ParlayConfig(seasons=(2000, 2100)))
    params = KalmanParams(12.0, 0.3, 13.0, 0.95)
    sim = simulate_games(sched, params, seed=0, market="efficient")
    sp = build_shared_game_pairs(to_team_games(sim))
    sp = sp[sp["week"] == 1]
    s = correlation_summary(sp["h_next_resid"], sp["a_next_resid"], n_boot=50)
    _, theory = gaussian_explaining_away_corr(12.0, 13.0)
    # One-shared-game formula ignores the other network paths; require same order of magnitude.
    assert 0.5 * theory < s.pearson < 1.6 * theory
    # And the filter's predicted covariances are calibrated on such data.
    out = MarketErrorKalman(params).run(sim)
    slope, se, _ = kalman_slope(out.pairs)
    assert slope == pytest.approx(1.0, abs=3 * se + 0.1)


def test_static_market_has_no_cross_game_correlation() -> None:
    sched = clean_games(make_schedule(n_teams=16, n_weeks=10, n_seasons=8, seed=5), ParlayConfig(seasons=(2000, 2100)))
    sim = simulate_games(sched, KalmanParams(10.0, 0.0, 5.0, 1.0), seed=0, market="static")
    tg = to_team_games(sim)
    # Strong within-team persistence (directional edge)...
    m = tg["next_resid"].notna()
    # corr = s² / (2s² + σ²) = 100 / 225 ≈ 0.44 for these parameters.
    assert np.corrcoef(tg.loc[m, "resid"], tg.loc[m, "next_resid"])[0, 1] > 0.3
    # ...but no shared-game cross correlation without a market update.
    sp = build_shared_game_pairs(tg)
    s = correlation_summary(sp["h_next_resid"], sp["a_next_resid"], n_boot=50)
    assert abs(s.pearson) < 0.1
