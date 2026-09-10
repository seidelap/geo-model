from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games
from geo_model.parlay.multisport_backtest import build_pitcher_pairs, moneyline_parlay
from geo_model.parlay.pairs import build_shared_game_pairs
from tests.parlay.conftest import make_schedule


def _with_prices(games: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = games.copy()
    g["p_home"] = rng.uniform(0.3, 0.7, len(g))
    g["home_decimal"] = 1 / (g["p_home"] * 1.02)
    g["away_decimal"] = 1 / ((1 - g["p_home"]) * 1.02)
    return g


def test_moneyline_parlay_independent_roi_matches_vig() -> None:
    games = _with_prices(clean_games(make_schedule(n_teams=12, n_weeks=12, n_seasons=6), ParlayConfig(seasons=(2000, 2100))))
    tg = to_team_games(games)
    pairs = build_shared_game_pairs(tg)
    r = moneyline_parlay(pairs, tg, games, n_boot=50)
    assert r.n_pairs > 100
    # Payouts embed a 2% margin per leg: independence ROI is about -(1 - 1/1.02^2).
    assert r.independent_roi == pytest.approx(1 / 1.02**2 - 1, abs=0.01)
    assert r.edge_ci[0] < r.edge < r.edge_ci[1]
    assert abs(r.p_same_side_indep - 0.5) < 0.05


def test_moneyline_parlay_detects_planted_correlation() -> None:
    games = _with_prices(clean_games(make_schedule(n_teams=12, n_weeks=12, n_seasons=6), ParlayConfig(seasons=(2000, 2100))))
    tg = to_team_games(games)
    pairs = build_shared_game_pairs(tg)
    # Force both legs to land on the same side for every pair.
    forced = games.set_index("game_id")
    for _, r in pairs.iterrows():
        forced.loc[r["h_next_game_id"], "result"] = 5.0 if forced.loc[r["h_next_game_id"], "home_team"] == r["team_h"] else -5.0
        forced.loc[r["a_next_game_id"], "result"] = 5.0 if forced.loc[r["a_next_game_id"], "home_team"] == r["team_a"] else -5.0
    res = moneyline_parlay(pairs, tg, forced.reset_index(), n_boot=50)
    # A game can be a leg of several pairs with conflicting forced signs, so not every pair lands same-side.
    assert res.p_same_side > 0.65
    assert res.edge > 0.3


def test_build_pitcher_pairs_structure() -> None:
    raw = make_schedule(n_teams=6, n_weeks=20, n_seasons=1, seed=2)
    # Rotate 3 "pitchers" per team so a pitcher starts every third team game.
    raw = raw.sort_values("gameday").reset_index(drop=True)
    counts: dict[str, int] = {}
    hq, aq = [], []
    for _, r in raw.iterrows():
        for col, team in (("h", r["home_team"]), ("a", r["away_team"])):
            k = counts.get(team, 0)
            counts[team] = k + 1
            (hq if col == "h" else aq).append(f"{team}_p{k % 3}")
    raw["home_qb_id"], raw["away_qb_id"] = hq, aq
    games = clean_games(raw, ParlayConfig(seasons=(2000, 2100)))
    pairs = build_pitcher_pairs(games)
    assert len(pairs) > 0
    gi = games.set_index("game_id")
    for _, r in pairs.iterrows():
        leg1 = gi.loc[r["h_next_game_id"]]
        assert r["pitcher"] in (leg1["home_qb_id"], leg1["away_qb_id"])
        assert r["h_next_game_id"] != r["a_next_game_id"]
        assert r["h_next_opp"] != r["team_a"] and r["a_next_opp"] != r["team_h"]
        assert r["h_next_opp"] != r["a_next_opp"]
        assert r["gap_days_leg1"] > 0 and r["gap_days_leg2"] > 0
