"""Tests for the corners stage's strictly-prior feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev.corners_features import (
    CornerConfig,
    add_prior_features,
    book_proxy_lambda,
    merge_opponent,
    prior_by_date,
    proxy_lambda,
    shrink,
    to_team_match,
)


def _frame() -> pd.DataFrame:
    """Two teams, four dates, one NaN value, and a same-day pair."""
    return pd.DataFrame({
        "g": ["A", "A", "A", "A", "B", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03",
                                "2020-01-01", "2020-01-02"]),
        "v": [1.0, 2.0, np.nan, 4.0, 10.0, 20.0],
    })


def test_prior_by_date_uses_strictly_earlier_dates() -> None:
    out = prior_by_date(_frame(), ["g"], "date", ["v"])
    # first row of each group sees nothing
    assert out["v_psum"].tolist()[0] == 0.0
    assert out["v_pcnt"].tolist()[0] == 0.0
    # the two same-day A rows see only 2020-01-01
    assert out["v_psum"].tolist()[1] == 1.0
    assert out["v_psum"].tolist()[2] == 1.0
    assert out["v_pcnt"].tolist()[1] == 1.0
    # the 2020-01-03 row sees 1 + 2 with the NaN not counted
    assert out["v_psum"].tolist()[3] == 3.0
    assert out["v_pcnt"].tolist()[3] == 2.0
    # groups do not mix
    assert out["v_psum"].tolist()[4] == 0.0
    assert out["v_psum"].tolist()[5] == 10.0


def test_prior_by_date_never_includes_the_row_itself() -> None:
    df = _frame()
    out = prior_by_date(df, ["g"], "date", ["v"])
    for i in range(len(df)):
        row = df.iloc[i]
        earlier = df[(df["g"] == row["g"]) & (df["date"] < row["date"])]
        assert out["v_psum"].iloc[i] == pytest.approx(earlier["v"].sum(skipna=True))
        assert out["v_pcnt"].iloc[i] == pytest.approx(earlier["v"].notna().sum())


def test_shrink_endpoints() -> None:
    assert shrink(np.array([10.0]), np.array([5.0]), np.array([1.0]), 0.0)[0] == 2.0
    assert shrink(np.array([0.0]), np.array([0.0]), np.array([7.0]), 4.0)[0] == 7.0
    # halfway: 5 prior matches summing to 10, prior mean 4, k = 5 -> (10 + 20) / 10
    assert shrink(np.array([10.0]), np.array([5.0]), np.array([4.0]), 5.0)[0] == 3.0


def _matches() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": [0, 1],
        "MatchDate": pd.to_datetime(["2020-01-01", "2020-01-08"]),
        "div_season": ["E0|2019/2020"] * 2,
        "Division": ["E0"] * 2,
        "season": ["2019/2020"] * 2,
        "HomeTeam": ["X", "Y"], "AwayTeam": ["Y", "X"],
        "HomeCorners": [6.0, 3.0], "AwayCorners": [4.0, 5.0],
        "HomeShots": [12.0, 9.0], "AwayShots": [8.0, 11.0],
        "HomeTarget": [5.0, 3.0], "AwayTarget": [2.0, 4.0],
        "HomeFouls": [10.0, 12.0], "AwayFouls": [11.0, 9.0],
        "FTHome": [2.0, 0.0], "FTAway": [1.0, 1.0],
        "HomeYellow": [1.0, 2.0], "AwayYellow": [2.0, 1.0],
    })


def test_to_team_match_swaps_correctly() -> None:
    tm = to_team_match(_matches())
    assert len(tm) == 4
    h0 = tm[(tm["match_id"] == 0) & (tm["is_home"] == 1)].iloc[0]
    a0 = tm[(tm["match_id"] == 0) & (tm["is_home"] == 0)].iloc[0]
    assert h0["team"] == "X" and h0["opp"] == "Y"
    assert h0["corners_for"] == 6.0 and h0["corners_against"] == 4.0
    assert a0["corners_for"] == 4.0 and a0["corners_against"] == 6.0
    assert a0["goals_for"] == 1.0 and a0["goals_against"] == 2.0


def test_merge_opponent_pairs_rows() -> None:
    tm = to_team_match(_matches())
    tm = merge_opponent(tm, ["corners_for"])
    for _, r in tm.iterrows():
        assert r["opp_corners_for"] == pytest.approx(r["corners_against"])


def test_add_prior_features_first_match_has_no_history() -> None:
    cfg = CornerConfig()
    tm = add_prior_features(to_team_match(_matches()), cfg)
    first = tm[tm["match_id"] == 0]
    assert (first["pc_corners_for"] == 0).all()
    assert (first["n_prior"] == 0).all()
    second = tm[tm["match_id"] == 1]
    assert (second["pc_corners_for"] == 1).all()
    # X scored 6 corners in match 0, Y 4
    x = second[second["team"] == "X"].iloc[0]
    assert x["ps_corners_for"] == 6.0


def test_proxy_lambda_matches_hand_computation() -> None:
    tm = pd.DataFrame({
        "is_home": [1],
        "lg_corners_for": [5.0], "lg_home_corners_for": [6.0], "lg_away_corners_for": [4.0],
        "ps_corners_for": [60.0], "pc_corners_for": [10.0],
        "opp_ps_corners_against": [40.0], "opp_pc_corners_against": [10.0],
    })
    # k = 0 -> attack ratio 6/5, concession ratio 4/5, level 6 -> 6 * 1.2 * 0.8
    got = proxy_lambda(tm, "corners_for", "corners_against", 0.0)[0]
    assert got == pytest.approx(6.0 * 1.2 * 0.8)
    assert book_proxy_lambda(tm, 0.0)[0] == pytest.approx(got)


def test_proxy_lambda_away_uses_away_level() -> None:
    tm = pd.DataFrame({
        "is_home": [0],
        "lg_corners_for": [5.0], "lg_home_corners_for": [6.0], "lg_away_corners_for": [4.0],
        "ps_corners_for": [50.0], "pc_corners_for": [10.0],
        "opp_ps_corners_against": [50.0], "opp_pc_corners_against": [10.0],
    })
    assert proxy_lambda(tm, "corners_for", "corners_against", 0.0)[0] == pytest.approx(4.0)
