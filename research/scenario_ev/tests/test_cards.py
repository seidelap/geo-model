"""Tests for the stage-01 card helpers on small synthetic inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import cards as K


@pytest.mark.parametrize(
    "position,flank,line",
    [
        ("Right Back", "right", "DEF"),
        ("Left Wing Back", "left", "DEF"),
        ("Left Center Back", "left", "DEF"),
        ("Center Back", "centre", "DEF"),
        ("Right Defensive Midfield", "right", "MID"),
        ("Center Attacking Midfield", "centre", "MID"),
        ("Left Midfield", "left", "MID"),
        ("Right Wing", "right", "FWD"),
        ("Center Forward", "centre", "FWD"),
        ("Goalkeeper", "centre", "GK"),
        (None, "unknown", "unknown"),
    ],
)
def test_position_geometry(position: str | None, flank: str, line: str) -> None:
    assert K.position_flank(position) == flank
    assert K.position_line(position) == line


def test_mirror_flank_is_an_involution() -> None:
    for f in ("left", "right", "centre"):
        assert K.mirror_flank(K.mirror_flank(f)) == f
    assert K.mirror_flank("left") == "right"
    assert K.mirror_flank("unknown") == "unknown"


def test_direct_opponent_lines_priority() -> None:
    assert K.direct_opponent_lines("Right Back") == ("FWD", "MID")
    assert K.direct_opponent_lines("Left Wing") == ("DEF", "MID")
    assert K.direct_opponent_lines("Center Defensive Midfield") == ("MID", "FWD")
    assert K.direct_opponent_lines("Goalkeeper") == ()


def test_is_fullback_and_position_group() -> None:
    assert K.is_fullback("Right Back")
    assert K.is_fullback("Left Wing Back")
    assert not K.is_fullback("Left Center Back")
    assert not K.is_fullback("Center Back")
    assert K.position_group("Right Back") == "FB"
    assert K.position_group("Left Center Back") == "CB"
    assert K.position_group("Right Defensive Midfield") == "MID_DEF"
    assert K.position_group("Center Forward") == "FWD"
    assert K.position_group("Goalkeeper") == "GK"


def test_country_mapping() -> None:
    assert K._country("E0") == "E"
    assert K._country("EC") == "E"
    assert K._country("SC2") == "SC"
    assert K._country("SP1") == "SP"


def test_poisson_glm1_recovers_slope() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=8000)
    mu = np.exp(0.4 + 0.7 * x)
    y = rng.poisson(mu).astype(float)
    a, b = K._poisson_glm1(x, y)
    assert a == pytest.approx(0.4, abs=0.05)
    assert b == pytest.approx(0.7, abs=0.05)


def test_score_state_at_uses_strictly_earlier_goals() -> None:
    goals = pd.DataFrame({
        "match_id": [1, 1, 1],
        "minute": [10.0, 20.0, 70.0],
        "is_home": [True, False, False],
    })
    mids = np.array([1, 1, 1, 1])
    minutes = np.array([5.0, 15.0, 25.0, 80.0])
    home = np.array([True, True, True, True])
    assert list(K.score_state_at(goals, mids, minutes, home)) == [0.0, 1.0, 0.0, -1.0]
    away = np.array([False, False, False, False])
    assert list(K.score_state_at(goals, mids, minutes, away)) == [0.0, -1.0, 0.0, 1.0]


def test_score_state_at_unknown_match_is_level() -> None:
    goals = pd.DataFrame({"match_id": [1], "minute": [10.0], "is_home": [True]})
    assert list(K.score_state_at(goals, np.array([99]), np.array([50.0]),
                                 np.array([True]))) == [0.0]


def test_scenario_mask_requires_all_three_conditions() -> None:
    df = pd.DataFrame({
        "is_fb": [1.0, 1.0, 1.0, 0.0],
        "opp_flank_dribbles_pm": [9.0, 9.0, 1.0, 9.0],
        "ref_card_rate": [0.3, 0.1, 0.3, 0.3],
    })
    m = K._scenario_mask(df, flank_cut=5.0, ref_cut=0.2)
    assert list(m) == [True, False, False, False]


def test_prepare_categoricals_shares_one_vocabulary() -> None:
    df = pd.DataFrame({"competition": ["A", "B", "A", "C"], "x": [1.0, 2.0, 3.0, 4.0]})
    df = K.prepare_categoricals(df)
    left = K._as_model_frame(df.iloc[:2], ["competition", "x"])
    right = K._as_model_frame(df.iloc[2:], ["competition", "x"])
    assert list(left["competition"].cat.categories) == list(right["competition"].cat.categories)


def test_foul_design_builds_the_hypothesised_interaction() -> None:
    df = pd.DataFrame({
        **{c: [1.0, 2.0, 3.0] for c in K._FOUL_NUM},
        **{c: [0.0, 1.0, 1.0] for c in K._FOUL_BIN},
        **{c: ["a", "b", "a"] for c in K._FOUL_CAT},
    })
    df["dribbling"] = [0.0, 1.0, 1.0]
    df["own_third"] = [1.0, 0.0, 1.0]
    X = K._foul_design(df)
    assert list(X["dribbling_x_own_third"]) == [0.0, 0.0, 1.0]
    assert abs(X["x"].mean()) < 1e-9
