from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.backtest import kalman_slope
from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games
from geo_model.parlay.multisport_backtest import build_pitcher_pairs, moneyline_parlay
from geo_model.parlay.pairs import build_shared_game_pairs
from geo_model.parlay.pitcher_kalman import (
    DEFENSE,
    OFFENSE,
    PITCHER,
    EntityIndex,
    GranularKalman,
    GranularParams,
    RunOffsets,
    attach_rest,
    null_log_likelihood,
    orient_pair_predictions,
    pair_requests_from_pairs,
    rest_strata,
    rest_table,
    run_residuals,
    same_day_pairs_for_moneyline,
    simulate_granular,
    theory_pair_corr,
)


def make_mlb_schedule(n_teams: int = 6, n_days: int = 30, n_seasons: int = 1, rotation: int = 3, seed: int = 0) -> pd.DataFrame:
    """Daily schedule with rotating starters, in the cleaned-games layout."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    starts = {t: 0 for t in teams}
    rows = []
    for season in range(2000, 2000 + n_seasons):
        day0 = pd.Timestamp(f"{season}-04-01", tz="UTC")
        for d in range(n_days):
            order = rng.permutation(teams)
            for k in range(0, n_teams - 1, 2):
                home, away = order[k], order[k + 1]
                hs, as_ = f"{home}_p{starts[home] % rotation}", f"{away}_p{starts[away] % rotation}"
                starts[home] += 1
                starts[away] += 1
                spread = float(rng.normal(0.3, 0.8))
                total_line = float(rng.choice([7.5, 8.0, 8.5, 9.0]))
                result = float(spread + rng.normal(0, 4))
                total = float(total_line + rng.normal(0, 4))
                day = day0 + pd.Timedelta(days=d)
                rows.append(
                    {
                        "game_id": f"{season}_{d:03d}_{away}_{home}", "season": season, "game_type": "REG",
                        "week": d // 7 + 1, "gameday": day, "home_team": home, "away_team": away,
                        "home_score": (total + result) / 2, "away_score": (total - result) / 2,
                        "result": result, "total": total, "spread_line": spread, "total_line": total_line,
                        "home_moneyline": -120.0, "away_moneyline": 100.0, "home_qb_id": hs, "away_qb_id": as_,
                        "resid": result - spread, "tresid": total - total_line,
                        "p_home": 0.55, "home_decimal": 1.0 / (0.55 * 1.02), "away_decimal": 1.0 / (0.45 * 1.02),
                    }
                )
    return pd.DataFrame(rows)


def test_params_roundtrip() -> None:
    p = GranularParams(0.7, 0.995, 0.2, 0.9, 0.4, 0.95, 3.1, -0.1)
    q = GranularParams.from_vector(p.to_vector())
    for k, v in p.__dict__.items():
        assert getattr(q, k) == pytest.approx(v)


def test_run_residuals_identities() -> None:
    g = run_residuals(make_mlb_schedule())
    assert np.allclose(g["home_run_resid"] - g["away_run_resid"], g["resid"])
    assert np.allclose(g["home_run_resid"] + g["away_run_resid"], g["tresid"])


def test_entity_index_blocks() -> None:
    g = make_mlb_schedule(n_teams=4, n_days=5)
    ent = EntityIndex.from_games(g)
    assert (ent.kind == PITCHER).sum() == 4 * 3
    assert (ent.kind == OFFENSE).sum() == 4 and (ent.kind == DEFENSE).sum() == 4
    h = ent.game_rows("T0", "T1", "T0_p0", "T1_p2")
    assert h[0, ent.index["o:T0"]] == 1 and h[0, ent.index["p:T1_p2"]] == -1 and h[0, ent.index["d:T1"]] == -1
    assert h[1, ent.index["o:T1"]] == 1 and h[1, ent.index["p:T0_p0"]] == -1 and h[1, ent.index["d:T0"]] == -1
    assert np.allclose(h.sum(axis=1), -1.0)


def test_first_slate_uses_prior_only() -> None:
    g = make_mlb_schedule(n_teams=6, n_days=3)
    p = GranularParams(0.5, 0.99, 0.3, 0.98, 0.2, 0.98, 3.0, 0.1)
    out = GranularKalman(p, RunOffsets(0.1, 0.3)).run(g)
    d1 = out.games[out.games["gameday"] == g["gameday"].min()]
    assert np.allclose(d1["pred_home_mean"], 0.1) and np.allclose(d1["pred_away_mean"], 0.3)
    v = 0.5**2 + 0.3**2 + 0.2**2 + 9.0
    assert np.allclose(d1["pred_home_var"], v)
    assert np.allclose(d1["pred_margin_var"], 2 * v - 2 * 0.1 * 9.0)
    assert np.allclose(d1["pred_total_var"], 2 * v + 2 * 0.1 * 9.0)
    first_pairs = out.pairs[out.pairs["gameday"] == g["gameday"].min()]
    assert np.allclose(first_pairs["pred_cov"], 0.0)
    assert set(out.log_likelihood_by_season) == {2000}
    assert out.n_obs_by_season[2000] == 2 * len(g)


def _one_shared_game_schedule() -> pd.DataFrame:
    """Day 1: A@B (P_A vs P_B). Day 2: A@C with P_A again, B@D with a new starter."""
    rows = []
    day = pd.Timestamp("2000-04-01", tz="UTC")
    spec = [
        (day, "B", "A", "B_p0", "A_p0"),
        (day + pd.Timedelta(days=1), "C", "A", "C_p0", "A_p0"),
        (day + pd.Timedelta(days=1), "D", "B", "D_p0", "B_p1"),
    ]
    for i, (d, home, away, hs, as_) in enumerate(spec):
        rows.append(
            {
                "game_id": f"g{i}", "season": 2000, "week": 1, "gameday": d, "home_team": home, "away_team": away,
                "home_score": 5.0, "away_score": 2.0, "result": 3.0, "total": 7.0, "spread_line": 0.0, "total_line": 8.0,
                "home_qb_id": hs, "away_qb_id": as_, "resid": 3.0, "tresid": -1.0,
            }
        )
    return pd.DataFrame(rows)


def test_one_shared_game_matches_theory() -> None:
    g = _one_shared_game_schedule()
    p = GranularParams(0.8, 1.0, 0.5, 1.0, 0.4, 1.0, 3.0, 0.0)  # persistence 1: no drift between the two days
    out = GranularKalman(p).run(g)
    th = theory_pair_corr(p)
    pair = out.pairs[out.pairs["gameday"] == g["gameday"].max()].iloc[0]
    # Game 1 on day 2 is A@C (A away), game 2 is B@D (B away): home-perspective margins flip twice, so the
    # home-perspective correlation equals the team-perspective one. The closed form uses prior variances in
    # the denominator, so the filter (posterior variances) is a fraction of a percent larger.
    assert pair["pred_corr"] == pytest.approx(th.margin_pitcher, rel=1e-2)
    assert pair["pred_tcorr"] == pytest.approx(th.total_pitcher, rel=1e-2)
    assert pair["pred_corr"] > th.margin_pitcher
    assert th.margin_pitcher > 0 > th.total_pitcher
    assert th.margin_team < th.margin_pitcher < th.margin_bound
    # Cross-slate request for the same pair gives the same numbers.
    req = pd.DataFrame({"game_id_1": ["g1"], "game_id_2": ["g2"]})
    cross = GranularKalman(p).run(g, pair_requests=req).cross_pairs
    assert cross["pred_corr"].iloc[0] == pytest.approx(pair["pred_corr"])
    assert cross["gap_days"].iloc[0] == 0


def test_cross_slate_prediction_decays_with_persistence() -> None:
    g = _one_shared_game_schedule()
    g.loc[2, "gameday"] = g.loc[2, "gameday"] + pd.Timedelta(days=5)  # B@D five days later
    req = pd.DataFrame({"game_id_1": ["g1"], "game_id_2": ["g2"]})
    c1 = GranularKalman(GranularParams(0.8, 1.0, 0.5, 1.0, 0.4, 1.0, 3.0, 0.0)).run(g, pair_requests=req).cross_pairs
    c2 = GranularKalman(GranularParams(0.8, 1.0, 0.5, 0.8, 0.4, 0.8, 3.0, 0.0)).run(g, pair_requests=req).cross_pairs
    assert c1["gap_days"].iloc[0] == 5
    assert 0 < c2["pred_corr"].iloc[0] < c1["pred_corr"].iloc[0]
    with pytest.raises(ValueError):
        GranularKalman().run(g, pair_requests=pd.DataFrame({"game_id_1": ["g2"], "game_id_2": ["g1"]}))


def test_null_likelihood_matches_zero_state_filter() -> None:
    g = make_mlb_schedule(n_teams=6, n_days=10)
    off = RunOffsets(0.05, 0.2)
    p = GranularParams(1e-6, 0.5, 1e-6, 0.5, 1e-6, 0.5, 3.2, 0.05)
    out = GranularKalman(p, off).run(g, emit_pairs=False)
    null = null_log_likelihood(g, 3.2, 0.05, off)
    assert out.log_likelihood == pytest.approx(sum(null.values()), rel=1e-6)


def test_efficient_market_simulation_is_calibrated() -> None:
    sched = make_mlb_schedule(n_teams=10, n_days=60, n_seasons=3, seed=4)
    p = GranularParams(2.0, 0.999, 1.5, 0.995, 1.0, 0.995, 3.0, 0.0)
    sim = simulate_granular(sched, p, seed=1, market="efficient")
    assert np.allclose(sim["home_run_resid"] - sim["away_run_resid"], sim["resid"])
    out = GranularKalman(p).run(sim)
    slope, se, _ = kalman_slope(out.pairs)
    assert slope == pytest.approx(1.0, abs=3 * se + 0.15)
    # Same-slate pairs with the most positive predicted correlation land on the same side more often.
    top = out.pairs[out.pairs["pred_corr"] > out.pairs["pred_corr"].quantile(0.9)]
    assert np.corrcoef(top["resid_1"], top["resid_2"])[0, 1] > 0.05
    # Innovations relative to an efficient market carry no lag-1 persistence.
    tg = to_team_games(sim)
    m = tg["next_resid"].notna()
    assert abs(np.corrcoef(tg.loc[m, "resid"], tg.loc[m, "next_resid"])[0, 1]) < 0.06


def test_static_market_shows_persistence_not_pair_correlation() -> None:
    sched = make_mlb_schedule(n_teams=10, n_days=60, n_seasons=3, seed=5)
    p = GranularParams(2.0, 0.999, 1.5, 0.999, 1.0, 0.999, 2.0, 0.0)
    sim = simulate_granular(sched, p, seed=2, market="static")
    tg = to_team_games(sim)
    m = tg["next_resid"].notna()
    # Consecutive games share the team's offense and defense errors (starters rotate):
    # corr = (s_o² + s_d²) / Var(margin) = 3.25 / 22.5 for these parameters.
    expected = (1.5**2 + 1.0**2) / (2 * (2.0**2 + 1.5**2 + 1.0**2) + 2 * 2.0**2)
    assert np.corrcoef(tg.loc[m, "resid"], tg.loc[m, "next_resid"])[0, 1] == pytest.approx(expected, abs=0.05)
    with pytest.raises(ValueError):
        simulate_granular(sched, p, market="bogus")


@pytest.mark.slow
def test_fit_recovers_large_pitcher_error() -> None:
    sched = make_mlb_schedule(n_teams=10, n_days=90, n_seasons=2, seed=7)
    truth = GranularParams(2.0, 0.999, 0.2, 0.9, 0.2, 0.9, 3.0, 0.0)
    sim = simulate_granular(sched, truth, seed=3, market="static")
    kf = GranularKalman(GranularParams(0.5, 0.99, 0.5, 0.95, 0.5, 0.95, 3.0, 0.0))
    fitted = kf.fit(sim, max_iter=120)
    assert 1.0 < fitted.pitcher_std < 3.5
    assert fitted.obs_std == pytest.approx(3.0, abs=0.5)


def test_pitcher_pair_requests_and_orientation() -> None:
    games = clean_games(make_mlb_schedule(n_teams=8, n_days=40, seed=3), ParlayConfig(seasons=(2000, 2100)))
    games = games.merge(make_mlb_schedule(n_teams=8, n_days=40, seed=3)[["game_id", "p_home", "home_decimal", "away_decimal"]], on="game_id")
    pairs = build_pitcher_pairs(games)
    assert len(pairs) > 20
    req = pair_requests_from_pairs(pairs, games)
    day = games.set_index("game_id")["gameday"]
    assert (day.reindex(req["game_id_1"]).to_numpy() <= day.reindex(req["game_id_2"]).to_numpy()).all()
    out = GranularKalman(GranularParams(0.8, 0.999, 0.5, 0.99, 0.4, 0.99, 3.0, 0.0)).run(games, emit_pairs=False, pair_requests=req)
    assert len(out.cross_pairs) == len(req)
    oriented = orient_pair_predictions(pairs, out.cross_pairs, games)
    assert oriented["pred_corr"].notna().all()
    # Team-perspective correlation flips the home-perspective one once per away leg.
    home = games.set_index("game_id")["home_team"]
    key = out.cross_pairs.set_index(["game_id_1", "game_id_2"])["pred_corr"]
    r = oriented.iloc[0]
    g1, g2 = req.iloc[0]["game_id_1"], req.iloc[0]["game_id_2"]
    sign = (1 if home[r["h_next_game_id"]] == r["team_h"] else -1) * (1 if home[r["a_next_game_id"]] == r["team_a"] else -1)
    assert r["pred_corr"] == pytest.approx(sign * key[(g1, g2)])
    assert r["pred_tcorr"] == pytest.approx(out.cross_pairs.set_index(["game_id_1", "game_id_2"])["pred_tcorr"][(g1, g2)])


def test_same_day_pairs_for_moneyline_orientation() -> None:
    games = make_mlb_schedule(n_teams=8, n_days=20, seed=1)
    out = GranularKalman(GranularParams(0.8, 0.999, 0.5, 0.99, 0.4, 0.99, 3.0, 0.0)).run(games)
    fp = out.pairs.copy()
    fp["pred_corr"] = np.where(np.arange(len(fp)) % 2 == 0, 0.01, -0.01)
    ml = same_day_pairs_for_moneyline(fp, games)
    gi = games.set_index("game_id")
    assert (ml["team_h"].to_numpy() == gi.loc[ml["h_next_game_id"], "home_team"].to_numpy()).all()
    pos = ml["pred_corr"] >= 0
    assert (ml.loc[pos, "team_a"].to_numpy() == gi.loc[ml.loc[pos, "a_next_game_id"], "home_team"].to_numpy()).all()
    assert (ml.loc[~pos, "team_a"].to_numpy() == gi.loc[ml.loc[~pos, "a_next_game_id"], "away_team"].to_numpy()).all()
    tg = to_team_games(games)
    res = moneyline_parlay(ml, tg, games, n_boot=20)
    assert res.n_pairs == len(ml)


def test_rest_helpers() -> None:
    games = clean_games(make_mlb_schedule(n_teams=6, n_days=12, seed=2), ParlayConfig(seasons=(2000, 2100)))
    raw = pd.DataFrame(
        {
            "Date": games["gameday"].dt.strftime("%Y-%m-%d"),
            "Home": games["home_team"], "Away": games["away_team"],
            "Days_Rest_Home": 1, "Days_Rest_Away": 2,
        }
    )
    rest = rest_table(raw)
    pairs = attach_rest(build_shared_game_pairs(to_team_games(games)), games, rest)
    assert pairs[["h_rest", "a_rest", "anchor_rest_h", "anchor_rest_a"]].notna().all().all()
    assert (pairs["anchor_rest_h"] == 1).all() and (pairs["anchor_rest_a"] == 2).all()
    strata = rest_strata(pairs)
    n = strata["both legs back-to-back"].sum() + strata["exactly one leg back-to-back"].sum() + strata["neither leg back-to-back"].sum()
    assert n == len(pairs)
    assert strata["anchor: either team on back-to-back"].all()


def test_cluster_bootstrap_phi_matches_pooled_phi_and_widens_ci() -> None:
    from geo_model.parlay.backtest import correlation_summary
    from geo_model.parlay.pitcher_kalman import cluster_bootstrap_phi, cluster_bootstrap_ratio

    rng = np.random.default_rng(0)
    n_cl, per = 60, 40
    # Cluster-level shocks make pairs within a cluster dependent.
    shock = rng.normal(0, 1.0, n_cl)
    cl = np.repeat(np.arange(n_cl), per)
    x = shock[cl] + rng.normal(0, 1, n_cl * per)
    y = shock[cl] + rng.normal(0, 1, n_cl * per)
    phi, ci, n = cluster_bootstrap_phi(x, y, cl, n_boot=200)
    s = correlation_summary(x, y, n_boot=200)
    assert n == n_cl * per
    assert phi == pytest.approx(s.phi, abs=1e-12)
    assert (ci[1] - ci[0]) > (s.phi_ci[1] - s.phi_ci[0])
    assert ci[0] < phi < ci[1]
    ratio, rci = cluster_bootstrap_ratio(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), n_boot=100)
    assert ratio == pytest.approx(1.0)
    assert rci[0] <= ratio <= rci[1]


def test_permutation_slope_pvalue() -> None:
    from geo_model.parlay.pitcher_kalman import permutation_slope_pvalue

    rng = np.random.default_rng(1)
    n = 2000
    day = np.repeat(np.arange(100), 20)
    cov = rng.normal(0, 1, n)
    r1 = rng.normal(0, 1, n)
    null = pd.DataFrame({"gameday": day, "pred_cov": cov, "resid_1": r1, "resid_2": rng.normal(0, 1, n)})
    slope, p = permutation_slope_pvalue(null, n_perm=99, seed=0)
    assert p > 0.05
    signal = null.copy()
    signal["resid_2"] = signal["resid_1"] * signal["pred_cov"] * 2 + rng.normal(0, 0.5, n)
    slope, p = permutation_slope_pvalue(signal, n_perm=99, seed=0)
    assert slope > 1.0 and p <= 0.02
    assert permutation_slope_pvalue(null.iloc[:2], n_perm=5) == (pytest.approx(float("nan"), nan_ok=True), pytest.approx(float("nan"), nan_ok=True))
