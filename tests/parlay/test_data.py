from __future__ import annotations

import pandas as pd

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games


def test_clean_games_residual_signs(raw_schedule: pd.DataFrame) -> None:
    cfg = ParlayConfig(seasons=(2000, 2001))
    games = clean_games(raw_schedule, cfg)
    assert len(games) == len(raw_schedule)
    assert (games["resid"] == games["result"] - games["spread_line"]).all()
    assert (games["tresid"] == games["total"] - games["total_line"]).all()
    assert games["gameday"].dt.tz is not None
    assert games["gameday"].is_monotonic_increasing


def test_clean_games_filters_seasons_and_types(raw_schedule: pd.DataFrame) -> None:
    raw = raw_schedule.copy()
    raw.loc[0, "game_type"] = "POST"
    raw.loc[1, "result"] = None
    games = clean_games(raw, ParlayConfig(seasons=(2001, 2001)))
    assert set(games["season"]) == {2001}
    games = clean_games(raw, ParlayConfig(seasons=(2000, 2001)))
    assert len(games) == len(raw) - 2


def test_to_team_games_next_game_and_sign(raw_schedule: pd.DataFrame) -> None:
    games = clean_games(raw_schedule, ParlayConfig(seasons=(2000, 2001)))
    tg = to_team_games(games)
    assert len(tg) == 2 * len(games)
    # Home and away rows of the same game have opposite residuals.
    g0 = games.iloc[0]
    rows = tg[tg["game_id"] == g0["game_id"]].set_index("team")
    assert rows.loc[g0["home_team"], "resid"] == g0["resid"]
    assert rows.loc[g0["away_team"], "resid"] == -g0["resid"]
    # Next game is the following week within the same season, last week has none.
    t = tg[(tg["team"] == "T0") & (tg["season"] == 2000)].sort_values("gameday")
    assert (t["next_week"].iloc[:-1].to_numpy() == t["week"].iloc[1:].to_numpy()).all()
    assert pd.isna(t["next_game_id"].iloc[-1])
    assert t["next_resid"].iloc[0] == t["resid"].iloc[1]


def test_new_qb_flag_counts_prior_starts(raw_schedule: pd.DataFrame) -> None:
    games = clean_games(raw_schedule, ParlayConfig(seasons=(2000, 2001)))
    tg = to_team_games(games, ParlayConfig(new_qb_max_prior_starts=2))
    t = tg[tg["team"] == "T0"].sort_values("gameday")
    assert t["qb_prior_starts"].tolist() == list(range(len(t)))
    assert t["new_qb"].tolist() == [True, True, True] + [False] * (len(t) - 3)
