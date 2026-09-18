"""Unit tests for the pure parts of the player-level participation module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import participation_players as pp


def test_snap_history_is_strictly_prior() -> None:
    cur = pd.DataFrame({
        "gsis_id": ["a", "a", "a", "a", "b"],
        "week": [1, 2, 4, 5, 3],
        "offense_pct": [1.0, 0.5, 0.0, 0.8, 0.2],
        "defense_pct": [0.0, 0.0, 0.0, 0.0, 0.9],
    })
    prev = pd.DataFrame({"gsis_id": ["a", "c"], "week": [10, 11], "offense_pct": [0.6, 0.1], "defense_pct": [0.0, 0.7]})
    h = pp.snap_history(cur, prev, n_weeks=6).set_index(["gsis_id", "week"])
    # week 1 sees nothing this season, but the previous season
    assert np.isnan(h.loc[("a", 1), "off_std"]) and h.loc[("a", 1), "off_prev"] == 0.6 and h.loc[("a", 1), "gp_prev"] == 1
    assert h.loc[("a", 1), "off_since"] == 7  # n_weeks + 1 when no snap yet
    # week 2 sees week 1 only
    assert h.loc[("a", 2), "off_std"] == 1.0 and h.loc[("a", 2), "off_last3"] == 1.0 and h.loc[("a", 2), "off_since"] == 1
    # week 3 (bye) sees weeks 1-2
    assert np.isclose(h.loc[("a", 3), "off_std"], 0.75) and h.loc[("a", 3), "off_since"] == 1
    # week 5 sees weeks 1, 2, 4: std mean 0.5, last3 = mean(1, .5, 0) = .5, since = 5 - 2 = 3 (week 4 had 0 snaps)
    assert np.isclose(h.loc[("a", 5), "off_std"], 0.5) and np.isclose(h.loc[("a", 5), "off_last3"], 0.5)
    assert h.loc[("a", 5), "off_since"] == 3
    # week 6 sees weeks 1, 2, 4, 5: last3 = mean(.5, 0, .8)
    assert np.isclose(h.loc[("a", 6), "off_last3"], (0.5 + 0.0 + 0.8) / 3)
    # player b: defense side, week 4 sees week 3
    assert h.loc[("b", 4), "def_std"] == 0.9 and h.loc[("b", 4), "def_since"] == 1 and h.loc[("b", 4), "gp_prev"] == 0
    # player only in previous season still gets rows
    assert h.loc[("c", 1), "def_prev"] == 0.7


def test_imputed_slots_skip_other_and_expected_value() -> None:
    classes = ["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR", "other"]
    P = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    arg, hard, expected = pp._imputed_slots(P, classes, "offense")
    assert arg.tolist() == [1, 0]  # 'other' never chosen
    assert hard["slot_TE"].tolist() == [2, 1]
    # expected TE slots: other -> first specific class (1 TE): row0 = .2*1 + .3*2 + .5*1 = 1.3
    assert np.isclose(expected["slot_TE"].iloc[0], 1.3)
    assert np.isclose(expected["slot_TE"].iloc[1], 0.7 * 1 + 0.2 * 2 + 0.1 * 1)


def test_feature_lists_are_disjoint_in_slot_columns() -> None:
    t, i = set(pp.feature_list("true")), set(pp.feature_list("imp"))
    assert "slot_true" in t and "slot_true" not in i and "slot_imp" in i and "slot_imp" not in t


def test_qb_check_table_partitions_plays_and_splits_residual() -> None:
    # five offense plays: team A week 1 (2 plays), team B week 1, team A week 2, team C week 1
    plays = pd.DataFrame({"game_id": ["g1", "g1", "g2", "g3", "g4"], "play_id": [1, 2, 1, 1, 1],
                          "team": ["A", "A", "B", "A", "C"], "week": [1, 1, 1, 2, 1]})
    # candidates are ACT players only: team B's listed QB1 is INA, team A's week-2 QB1 is on reserve,
    # team C's depth-1 QB is dressed but never takes a snap
    qb_cands = pd.DataFrame([
        ("g1", 1, "qa1", 1, 1, 1.0), ("g1", 1, "qa2", 2, 2, 0.0),
        ("g1", 2, "qa1", 1, 1, 0.0), ("g1", 2, "qa2", 2, 2, 1.0),   # in-game change
        ("g2", 1, "qb2", 2, 1, 1.0),
        ("g3", 1, "qa2", 2, 1, 1.0),
        ("g4", 1, "qc1", 1, 1, 0.0), ("g4", 1, "qc2", 2, 2, 1.0),
    ], columns=["game_id", "play_id", "gsis_id", "depth_rank", "usage_rank", "y"])
    qb1_status = pd.DataFrame({"team": ["A", "B", "A", "C"], "week": [1, 1, 2, 1],
                               "gsis_id": ["qa1", "qb1", "qa1", "qc1"], "status": ["ACT", "INA", "RES", "ACT"]})
    played = pd.DataFrame({"game_id": ["g1", "g1", "g2", "g3", "g4"], "gsis_id": ["qa1", "qa2", "qb2", "qa2", "qc2"]})
    t = pp.qb_check_table(qb_cands, plays, qb1_status, played)
    dc = t[t["candidate"] == "depth-chart QB1"].set_index("condition")
    assert dc.loc[pp.ACT_COND, "plays"] == 3 and np.isclose(dc.loc[pp.ACT_COND, "share_of_plays"], 0.6)
    assert np.isclose(dc.loc[pp.ACT_COND, "on_field_rate"], 1 / 3)
    assert dc.loc[pp.SNAP_COND, "plays"] == 2 and np.isclose(dc.loc[pp.SNAP_COND, "on_field_rate"], 0.5)
    assert dc.loc[pp.NOSNAP_COND, "plays"] == 1 and dc.loc[pp.NOSNAP_COND, "on_field_rate"] == 0.0
    assert dc.loc[pp.QB1_STATUS_LABELS["INA"], "plays"] == 1 and dc.loc[pp.QB1_STATUS_LABELS["other"], "plays"] == 1
    assert pp.QB1_STATUS_LABELS["mismatch"] not in dc.index
    # the ACT row plus the 'no depth-1 QB' rows partition the plays; the two '...' rows partition the ACT row
    top = dc.drop(index=[pp.SNAP_COND, pp.NOSNAP_COND])
    assert top["plays"].sum() == 5 and np.isclose(top["share_of_plays"].sum(), 1.0)
    assert dc.loc[pp.SNAP_COND, "plays"] + dc.loc[pp.NOSNAP_COND, "plays"] == dc.loc[pp.ACT_COND, "plays"]
    pu = t[t["candidate"] == "prior-usage QB1"].set_index("condition")
    assert pu.loc["all plays", "plays"] == 5 and np.isclose(pu.loc["all plays", "on_field_rate"], 0.6)
    sentence = pp.qb_check_sentence(t)
    assert sentence.startswith("The depth-chart QB1 is on the ACT list on 60.0% of test plays")
    assert "20.0% of plays the listed QB1 was declared inactive" in sentence
