"""Unit tests for the pure NFL participation feature functions (synthetic input)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.nfl import participation_features as pf


def _plays() -> pd.DataFrame:
    """Two teams, one season, three weeks; team A's groupings: g1=[11,11,12], g2=[12], g3=[11]."""
    rows = [
        # season, week, game, posteam, defteam, grp, down, ydstogo
        (2020, 1, "g1", "A", "B", "11", 1, 10),
        (2020, 1, "g1", "A", "B", "11", 2, 3),
        (2020, 1, "g1", "A", "B", "12", 3, 8),
        (2020, 1, "g1", "B", "A", "21", 1, 10),
        (2020, 2, "g2", "A", "C", "12", 1, 10),
        (2020, 2, "g2", "C", "A", "11", 1, 10),
        (2020, 3, "g3", "A", "B", "11", 1, 5),
        (2020, 3, "g3", "B", "A", "11", 2, 2),
        (2021, 1, "g4", "A", "B", "12", 1, 10),
    ]
    return pd.DataFrame(rows, columns=["season", "week", "game_id", "posteam", "defteam", "grp",
                                       "down", "ydstogo"])


def test_top_classes_and_collapse() -> None:
    s = pd.Series(["a"] * 60 + ["b"] * 30 + ["c"] * 6 + ["d"] * 4)
    assert pf.top_classes(s, coverage=0.95) == ["a", "b", "c"]
    assert pf.top_classes(s, coverage=0.5) == ["a", "b"]  # min_classes
    col = pf.collapse_classes(pd.Series(["a", "d", None]), ["a", "b"])
    assert list(col[:2]) == ["a", pf.OTHER] and pd.isna(col.iloc[2])
    # an unlabelled play must not contribute to the tendency counts
    counts = pf._onehot(col, ["a", "b", pf.OTHER])
    assert counts.sum() == 2 and counts[2].sum() == 0


def test_slug_is_feature_name_safe() -> None:
    assert pf.slug("1 RB, 1 TE, 3 WR") == "1RB_1TE_3WR"
    assert pf.slug("6 OL, 1 RB, 2 TE, 1 WR") == "6OL_1RB_2TE_1WR"
    assert pf.slug("COVER_3") == "COVER_3" and pf.slug("2_MAN") == "2_MAN" and pf.slug("other") == "other"
    assert all(ch.isalnum() or ch == "_" for ch in pf.slug("4 DL, 2 LB, 5 DB"))


def test_buckets() -> None:
    assert list(pf.distance_bucket(np.array([1, 3, 4, 7, 8, np.nan]))) == [
        "short", "short", "mid", "mid", "long", "na"]
    b = pf.down_distance_bucket(np.array([1, 3, np.nan]), np.array([10, 2, 5]))
    assert list(b) == ["d1_long", "d3_short", "dna_mid"]


def test_prior_game_counts_are_strictly_prior() -> None:
    p = _plays()
    classes = ["11", "12", "other"]
    counts, n = pf.prior_game_class_counts(p, ["posteam"], "grp", classes)
    # week-1 plays see nothing
    assert n[:4].sum() == 0
    # team A in game 2 sees only game 1: two 11s and one 12
    assert counts[4].tolist() == [2.0, 1.0, 0.0] and n[4] == 3
    # team A in game 3 sees games 1 and 2
    assert counts[6].tolist() == [2.0, 2.0, 0.0]
    # team B in game 3 sees its game-1 play ("21" -> not in classes -> zero counts, n=0)
    assert counts[7].tolist() == [0.0, 0.0, 0.0]
    # new season resets
    assert n[8] == 0
    # bucketed entity: A in d1_long at game 3 sees game1 (11) + game2 (12)
    p["dd"] = pf.down_distance_bucket(p["down"].to_numpy(), p["ydstogo"].to_numpy())
    c2, n2 = pf.prior_game_class_counts(p, ["posteam", "dd"], "grp", classes)
    assert c2[6].tolist() == [0.0, 0.0, 0.0]  # game 3 play is d1_mid -> no prior d1_mid plays
    assert c2[4].tolist() == [1.0, 0.0, 0.0]  # game 2 play d1_long sees game 1's d1_long 11


def test_previous_season_and_league_prior() -> None:
    p = _plays()
    classes = ["11", "12", "other"]
    counts, n = pf.previous_season_counts(p, ["posteam"], "grp", classes)
    assert n[:8].sum() == 0  # no 2019 data
    assert counts[8].tolist() == [3.0, 2.0, 0.0] and n[8] == 5  # A's full 2020: 3x11, 2x12
    lg = pf.league_prior_shares(p, "grp", classes, alpha=3.0)
    assert np.allclose(lg[0], 1 / 3)  # first week: uniform
    # week 2 sees week 1: counts 11:2, 12:1, other(21 not in classes):0 -> (2+1, 1+1, 0+1)/(3+3)
    assert np.allclose(lg[4], [3 / 6, 2 / 6, 1 / 6])
    assert np.allclose(lg.sum(axis=1), 1.0)


def test_shrinkage_and_conditional() -> None:
    counts = np.array([[8.0, 2.0], [0.0, 0.0]])
    prior = np.array([[0.5, 0.5], [0.5, 0.5]])
    s = pf.shrink_shares(counts, counts.sum(1), prior, alpha=10.0)
    assert np.allclose(s[0], [13 / 20, 7 / 20]) and np.allclose(s[1], [0.5, 0.5])
    m = pf.shrink_mean(np.array([60.0]), np.array([10.0]), np.array([4.0]), alpha=10.0)
    assert np.isclose(m[0], 5.0)
    # joint counts for 2 cond classes x 2 target classes: cond0 -> [6, 2], cond1 -> [1, 9]
    joint = np.array([[6.0, 2.0, 1.0, 9.0]] * 3)
    pr = np.full((3, 2), 0.5)
    out = pf.conditional_from_joint(joint, np.array([0, 1, -1]), 2, 2, pr, alpha=2.0)
    assert np.allclose(out[0], [7 / 10, 3 / 10])
    assert np.allclose(out[1], [2 / 12, 10 / 12])
    assert np.allclose(out[2], [0.5, 0.5])
    assert pf.joint_classes(["a", "b"], ["x", "y"]) == ["a|x", "a|y", "b|x", "b|y"]


def test_slots() -> None:
    assert pf.offense_slots("1 RB, 1 TE, 3 WR") == {"QB": 1, "OL": 5, "RB": 1, "TE": 1, "WR": 3}
    assert pf.offense_slots("6 OL, 1 RB, 2 TE, 1 WR") == {"QB": 1, "OL": 6, "RB": 1, "TE": 2, "WR": 1}
    assert pf.offense_slots("2 QB, 1 RB, 1 TE, 2 WR")["QB"] == 2
    assert sum(pf.offense_slots("1 RB, 2 TE, 1 WR, 1 DL").values()) == 11
    assert pf.defense_slots("4 DL, 2 LB, 5 DB") == {"DL": 4, "LB": 2, "DB": 5}
    assert pf.defense_slots(None) == {"DL": 4, "LB": 2, "DB": 5}
    f = pf.slots_frame(pd.Series(["1 RB, 1 TE, 3 WR", None]), "offense")
    assert list(f.columns) == ["slot_QB", "slot_OL", "slot_RB", "slot_TE", "slot_WR"]
    assert f.iloc[1]["slot_WR"] == 3


def test_usage_rank_and_decode() -> None:
    df = pd.DataFrame({
        "team": ["A"] * 4 + ["B"] * 2,
        "grp": ["WR"] * 4 + ["WR"] * 2,
        "usage": [0.9, np.nan, 0.5, 0.5, 0.1, 0.2],
        "depth": [1, 1, 2, 1, 1, 1],
    }, index=[10, 11, 12, 13, 14, 15])
    r = pf.usage_rank(df, ["team", "grp"], "usage", "depth")
    assert r.tolist() == [1, 4, 3, 2, 2, 1]
    cands = pd.DataFrame({
        "play": [1, 1, 1, 1, 1],
        "grp": ["WR", "WR", "WR", "TE", "TE"],
        "p": [0.9, 0.2, 0.6, 0.3, 0.8],
        "slots": [2, 2, 2, 1, 1],
    })
    sel = pf.decode_lineup(cands, ["play"], "grp", "p", "slots")
    assert sel.tolist() == [True, False, True, False, True]


def test_roster_group_and_subgroup() -> None:
    assert pf.roster_group("OL") == "OL" and pf.roster_group("DB", "CB") == "DB"
    assert pf.roster_group("K") == "ST" and pf.roster_group(None, "FS") == "DB"
    assert pf.report_subgroup("WR", "WR", 2) == "WR1-2" and pf.report_subgroup("WR", "WR", 3) == "WR3+"
    assert pf.report_subgroup("DB", "FS", 1) == "S"
    assert pf.report_subgroup("DB", "CB", 3).startswith("CB3+")
    assert pf.report_subgroup("OL", "T", 1) == "OL"


@pytest.mark.parametrize("side", ["offense", "defense"])
def test_slots_sum_to_eleven_for_common_strings(side: str) -> None:
    strings = (["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR", "6 OL, 1 RB, 2 TE, 1 WR"] if side == "offense"
               else ["4 DL, 2 LB, 5 DB", "3 DL, 4 LB, 4 DB", "2 DL, 3 LB, 6 DB"])
    for s in strings:
        slots = pf.offense_slots(s) if side == "offense" else pf.defense_slots(s)
        assert sum(slots.values()) == 11
