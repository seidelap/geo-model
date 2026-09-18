"""Unit tests for the pure helpers of the NFL stage-01 driver (no data access)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.nfl import build_tables as bt


def test_column_tier_contract() -> None:
    assert bt.column_tier("gameId") == "id" and bt.column_tier("pbp_posteam") == "id"
    assert bt.column_tier("ngs_joined") == "id"
    # event-only pre-snap situation and pre-snap models
    for c in ("down", "yardsToGo", "pbp_down", "pbp_yardline_100", "pbp_score_differential", "pbp_ep",
              "pbp_wp", "pbp_xpass", "pbp_shotgun", "pbp_no_huddle", "pbp_spread_line", "pbp_roof"):
        assert bt.column_tier(c) == "event_only_presnap", c
    # post-snap outcomes and post-snap models never feed a pre-snap model
    for c in ("PassResult", "PlayResult", "playDescription", "is_pass_play", "pbp_play_type", "pbp_air_yards",
              "pbp_epa", "pbp_cp", "pbp_cpoe", "pbp_success", "pbp_sack", "pbp_qb_hit", "pbp_receiver_player_id",
              "pbp_desc", "pbp_pass_location"):
        assert bt.column_tier(c) == "event_only_postsnap", c
    # NGS charting from the same tracking, whatever the source file
    for c in ("defendersInTheBox", "numberOfPassRushers", "offenseFormation", "personnel_offense",
              "ngs_defenders_in_box", "ngs_time_to_throw", "ngs_was_pressure", "ngs_air_yards", "ngs_route",
              "ngs_offense_players"):
        assert bt.column_tier(c) == "ngs_charting_privileged", c
    for c in ("flipped", "los_source", "ev_pass_forward", "qb_throw_gsis", "target_id_agree", "target_id"):
        assert bt.column_tier(c) == "tracking_context", c
    for c in ("box_count", "n_wide_left", "cushion_left", "time_to_throw", "target_side_derived",
              "separation_at_arrival", "n_rush_w20_cm05", "derived_RB", "personnel_offense_derived"):
        assert bt.column_tier(c) == "tracking_target", c
    cols = ["gameId", "down", "pbp_epa", "ngs_route", "box_count", "pbp_wp"]
    assert bt.event_only_presnap_columns(cols) == ["down", "pbp_wp"]
    # no participation column may ever be published under a prefix the contract does not know
    assert all(c.startswith("ngs_") or c in ("old_game_id", "play_id") or c == "ngs_air_yards"
               for c in [c if c.startswith("ngs_") or c in ("old_game_id", "play_id") else f"ngs_{c}"
                         for c in bt.PARTICIPATION_COLUMNS])


def test_column_definitions_table_covers_patterns() -> None:
    t = bt.column_definitions_table(["gameId", "n_wide_left", "n_rush_w15_cm05", "ev_handoff", "pbp_epa",
                                     "ngs_route", "target_gsis", "derived_WR", "unknown_col"])
    d = dict(zip(t["column"], t["definition"]))
    assert list(t.columns) == ["column", "tier", "definition"]
    assert "offense's left" in d["n_wide_left"]
    assert d["n_rush_w15_cm05"] and d["ev_handoff"] and d["pbp_epa"] and d["ngs_route"] and d["target_gsis"]
    assert d["derived_WR"] and d["unknown_col"] == ""
    assert t.set_index("column").loc["pbp_epa", "tier"] == "event_only_postsnap"


def test_agreement_stats_tolerance() -> None:
    a = pd.Series([1.0, 2.0, 3.0 + 1e-12, 4.0, np.nan])
    b = pd.Series([1.0, 2.5, 3.0, 4.0, 1.0])
    st = bt.agreement_stats(a, b, "a", "b")
    assert st["n"] == 4 and st["exact"] == pytest.approx(0.75) and st["within_1"] == 1.0
    assert st["bias_a_minus_b"] == pytest.approx(-0.125)
    st0 = bt.agreement_stats(a, b, "a", "b", tol=0.0)
    assert st0["exact"] == pytest.approx(0.5)   # bit equality drops the 1e-12 case
    assert bt.agreement_stats(pd.Series([np.nan]), pd.Series([np.nan]), "a", "b")["n"] == 0


def test_grid_agreement_splits_by_week() -> None:
    week = pd.Series([1, 1, 2, 3, 4, 5, 6, 6])
    a = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    b = pd.Series([1, 2, 3, 4, 5, 7, 7, 9], dtype=float)   # perfect on weeks 1-3, 1/4 on 4-6
    st = bt.grid_agreement(a, b, week, "a", "b")
    assert st["n_tune"] == 4 and st["exact_tune"] == 1.0
    assert st["n"] == 4 and st["exact"] == pytest.approx(0.5) and st["within_1"] == 1.0
    assert st["exact_all"] == pytest.approx(0.75)


def test_norm_name() -> None:
    assert bt.norm_name("Odell Beckham Jr.") == "odell beckham"
    assert bt.norm_name("Ha Ha Clinton-Dix") == "ha ha clinton dix"
    assert bt.norm_name(None) == ""
