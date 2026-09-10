from __future__ import annotations

import numpy as np
import pandas as pd

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games
from geo_model.parlay.pairs import build_shared_game_pairs, cover_indicators, stratify


def test_pairs_exclude_rematches_and_last_week(raw_schedule: pd.DataFrame) -> None:
    games = clean_games(raw_schedule, ParlayConfig(seasons=(2000, 2001)))
    tg = to_team_games(games)
    pairs = build_shared_game_pairs(tg)
    assert len(pairs) > 0
    assert (pairs["h_next_game_id"] != pairs["a_next_game_id"]).all()
    assert (pairs["h_next_opp"] != pairs["team_a"]).all()
    assert (pairs["a_next_opp"] != pairs["team_h"]).all()
    assert (pairs["h_next_opp"] != pairs["a_next_opp"]).all()
    assert pairs["week"].max() < games["week"].max()
    # Leg residuals are read from the right games and team perspective.
    row = pairs.iloc[0]
    nxt = games.set_index("game_id").loc[row["h_next_game_id"]]
    expected = nxt["resid"] if nxt["home_team"] == row["team_h"] else -nxt["resid"]
    assert row["h_next_resid"] == expected


def test_stratify_labels() -> None:
    df = pd.DataFrame({"abs_surprise": [0.0, 6.9, 7.0, 20.0]})
    labels = stratify(df, "abs_surprise", (0.0, 7.0, 14.0, 1000.0))
    assert labels.tolist() == ["[0, 7)", "[0, 7)", "[7, 14)", "[14, 1000)"]


def test_cover_indicators_drop_pushes() -> None:
    cx, cy = cover_indicators(np.array([1.0, -2.0, 0.0, 3.0]), np.array([-1.0, 4.0, 2.0, np.nan]))
    assert cx.tolist() == [1, 0]
    assert cy.tolist() == [0, 1]
