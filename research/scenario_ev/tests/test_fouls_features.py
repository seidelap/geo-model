"""Unit tests for the pure feature helpers of stage 04 (fouls)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import fouls_features as FF
from research.scenario_ev.corners_features import CornerConfig, add_prior_features, merge_opponent, to_team_match


def _matches() -> pd.DataFrame:
    """Four synthetic matches between two teams over four distinct dates."""
    rows = []
    for i, (h, a, hf, af) in enumerate([
        ("A", "B", 10, 20), ("B", "A", 12, 22), ("A", "B", 14, 24), ("B", "A", 16, 26),
    ]):
        rows.append({
            "match_id": i, "MatchDate": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "div_season": "X|2019/2020", "Division": "X", "season": "2019/2020",
            "HomeTeam": h, "AwayTeam": a, "HomeFouls": hf, "AwayFouls": af,
            "HomeCorners": 5, "AwayCorners": 4, "HomeShots": 10, "AwayShots": 9,
            "HomeTarget": 4, "AwayTarget": 3, "FTHome": 1, "FTAway": 1,
            "HomeYellow": 2, "AwayYellow": 1,
        })
    return pd.DataFrame(rows)


def _prepared() -> pd.DataFrame:
    """Team-match frame with the foul proxy inputs attached."""
    cfg = CornerConfig()
    tm = add_prior_features(to_team_match(_matches()), cfg)
    tm = FF.add_foul_proxy_inputs(tm, cfg)
    carry = [c for c in tm.columns if c.startswith(("fps_", "fpc_"))]
    return merge_opponent(tm, carry)


def test_foul_priors_are_strictly_earlier() -> None:
    tm = _prepared()
    a = tm[(tm["team"] == "A")].sort_values("date")
    # A plays home (10), away (22), home (14), away (26)
    assert list(a["fouls_for"]) == [10, 22, 14, 26]
    assert list(a["fpc_fouls_for"]) == [0, 1, 2, 3]
    assert list(a["fps_fouls_for"]) == [0, 10, 32, 46]


def test_home_away_league_levels_are_separate() -> None:
    tm = _prepared()
    last = tm.sort_values("date").iloc[-1]
    # home rows carry 10, 12, 14; away rows 20, 22, 24 before the final date
    assert last["lg_home_fouls_for"] < last["lg_away_fouls_for"]


def test_proxy_lambda_shapes_and_monotonicity() -> None:
    tm = _prepared()
    mu_small = FF.foul_proxy_lambda(tm, 1.0)
    mu_big = FF.foul_proxy_lambda(tm, 1e6)
    assert mu_small.shape == (len(tm),)
    # infinite shrinkage collapses the two ratios to one, leaving the league level
    lvl = np.where(tm["is_home"].to_numpy() == 1, tm["lg_home_fouls_for"],
                   tm["lg_away_fouls_for"])
    assert np.allclose(mu_big, lvl, rtol=1e-3, equal_nan=True)


def test_additive_and_multiplicative_agree_at_the_league_mean() -> None:
    tm = _prepared()
    m = FF.foul_proxy_lambda(tm, 1e6, multiplicative=True)
    a = FF.foul_proxy_lambda(tm, 1e6, multiplicative=False)
    assert np.allclose(m, a, rtol=1e-3, equal_nan=True)


def test_pick_line_chooses_the_nearest_rung() -> None:
    out = FF.pick_line(np.array([9.0, 12.4, 12.6, 100.0]), (9.5, 12.5, 13.5))
    assert list(out) == [9.5, 12.5, 12.5, 13.5]


def _players() -> pd.DataFrame:
    """Two teams of two starters each, in one match, on mirrored flanks."""
    return pd.DataFrame({
        "match_id": [1, 1, 1, 1],
        "team_id": [10, 10, 20, 20],
        "opp_id": [20, 20, 10, 10],
        "starter": [True, True, True, True],
        "flank": ["left", "right", "left", "right"],
        "line": ["DEF", "DEF", "FWD", "FWD"],
        "pps_minutes": [900.0, 900.0, 900.0, 900.0],
        "pl_prior_n": [10, 10, 10, 10],
        "pl_fouls_p90": [1.0, 2.0, 3.0, 4.0],
    })


def test_direct_opponent_rates_mirror_the_flank() -> None:
    p = _players()
    mirror = {"left": "right", "right": "left"}
    prio = pd.Series([("FWD",), ("FWD",), ("DEF",), ("DEF",)], index=p.index)
    out = FF.direct_opponent_rates(p, ["pl_fouls_p90"], mirror, prio, prefix="d_")
    # the left-sided defender of team 10 meets the right-sided forward of team 20 (4.0)
    assert out.loc[0, "d_pl_fouls_p90"] == pytest.approx(4.0)
    assert out.loc[1, "d_pl_fouls_p90"] == pytest.approx(3.0)
    assert out.loc[2, "d_pl_fouls_p90"] == pytest.approx(2.0)
    assert (out["d_line_used"] != "").all()


def test_direct_opponent_rates_leave_unmatched_rows_null() -> None:
    p = _players()
    p.loc[2:, "line"] = "MID"  # no FWD on the opposing side any more
    mirror = {"left": "right", "right": "left"}
    prio = pd.Series([("FWD",), ("FWD",), ("DEF",), ("DEF",)], index=p.index)
    out = FF.direct_opponent_rates(p, ["pl_fouls_p90"], mirror, prio, prefix="d_")
    assert out.loc[0, "d_pl_fouls_p90"] != out.loc[0, "d_pl_fouls_p90"]  # NaN


def test_flank_foul_priors_are_strictly_prior() -> None:
    fouls = pd.DataFrame({
        "match_id": [1, 1, 2, 2, 3],
        "team_id": [10, 10, 10, 20, 10],
        "side": ["left", "left", "right", "left", "centre"],
    })
    tmm = pd.DataFrame({
        "match_id": [1, 2, 3], "team_id": [10, 10, 10],
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
    })
    out = FF.flank_foul_priors(fouls, tmm, k=0.0)
    out = out.sort_values("match_id")
    # k=0: match 2 sees only match 1 (2 left fouls); match 3 sees matches 1 and 2
    assert out.iloc[1]["tp_fouls_left_pm"] == pytest.approx(2.0)
    assert out.iloc[2]["tp_fouls_left_pm"] == pytest.approx(1.0)
    assert out.iloc[2]["tp_fouls_right_pm"] == pytest.approx(0.5)


def test_flank_foul_prior_shrinkage_target_is_strictly_prior() -> None:
    """The global mean the side rates shrink toward uses earlier dates only."""
    fouls = pd.DataFrame({
        "match_id": [1, 1, 2, 3],
        "team_id": [10, 10, 20, 30],
        "side": ["left", "left", "left", "left"],
    })
    tmm = pd.DataFrame({
        "match_id": [1, 2, 3], "team_id": [10, 20, 30],
        "date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
    })
    # k -> infinity leaves the shrinkage target alone, so it can be read off directly
    out = FF.flank_foul_priors(fouls, tmm, k=1e9).sort_values("match_id")
    left = out["tp_fouls_left_pm"].to_numpy()
    # row 1 has nothing earlier -> the declared constant
    assert left[0] == pytest.approx(FF.FLANK_PRIOR_FALLBACK)
    # row 2 sees only match 1 (2 left fouls); row 3 sees matches 1 and 2 -> (2 + 1) / 2
    assert left[1] == pytest.approx(2.0)
    assert left[2] == pytest.approx(1.5)


def test_flank_foul_prior_target_ignores_later_matches() -> None:
    """Appending later, far larger matches cannot change an earlier row's value."""
    fouls = pd.DataFrame({
        "match_id": [1, 1, 2], "team_id": [10, 10, 20],
        "side": ["left", "left", "left"],
    })
    tmm = pd.DataFrame({
        "match_id": [1, 2], "team_id": [10, 20],
        "date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
    })
    base = FF.flank_foul_priors(fouls, tmm, k=3.0).sort_values("match_id")
    later_fouls = pd.concat([fouls, pd.DataFrame({
        "match_id": [3] * 40, "team_id": [30] * 40, "side": ["left"] * 40})],
        ignore_index=True)
    later_tmm = pd.concat([tmm, pd.DataFrame({
        "match_id": [3], "team_id": [30],
        "date": pd.to_datetime(["2020-02-01"])})], ignore_index=True)
    grown = FF.flank_foul_priors(later_fouls, later_tmm, k=3.0).sort_values("match_id")
    assert grown["tp_fouls_left_pm"].to_numpy()[:2] == pytest.approx(
        base["tp_fouls_left_pm"].to_numpy())
