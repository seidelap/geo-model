"""Tests for the NFL 07 forward props helpers (synthetic frames, no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.nfl import props_forward as pfw


def _plays() -> pd.DataFrame:
    # receiver A: weeks 1, 2, 4 (bye in 3); receiver B: week 1 only
    rows = [
        (2020, 1, "KC", "A", 1, 10.0, 5.0, 0.6, 0.5, 2.0),
        (2020, 1, "KC", "A", 0, 0.0, 15.0, 0.4, -0.5, 3.0),
        (2020, 1, "KC", "B", 1, 20.0, 8.0, 0.7, 1.0, 4.0),
        (2020, 2, "KC", "A", 1, 30.0, 12.0, 0.5, 2.0, 2.5),
        (2020, 4, "KC", "A", 0, 0.0, 20.0, 0.3, -1.0, 1.0),
        (2020, 4, "KC", "A", 1, 5.0, 3.0, 0.8, 0.2, 1.5),
        (2020, 4, "KC", "A", 1, 7.0, 4.0, 0.8, 0.3, 1.5),
    ]
    cols = [
        "season",
        "week",
        "posteam",
        "receiver_player_id",
        "complete_pass",
        "yards_gained",
        "air_yards",
        "cp",
        "epa",
        "imp_separation_at_arrival__F1T",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_receiver_game_table_aggregates_targets_receptions_and_yards() -> None:
    rg = pfw.receiver_game_table(_plays())
    a1 = rg[(rg["receiver_player_id"] == "A") & (rg["week"] == 1)].iloc[0]
    assert a1["n_targets"] == 2 and a1["receptions"] == 1 and a1["rec_yards"] == 10.0
    assert a1["team_targets"] == 3 and a1["target_share"] == pytest.approx(2 / 3)
    assert a1["imp_separation_at_arrival__F1T"] == pytest.approx(2.5)
    a4 = rg[(rg["receiver_player_id"] == "A") & (rg["week"] == 4)].iloc[0]
    assert a4["n_targets"] == 3 and a4["rec_yards"] == 12.0 and a4["receptions"] == 2
    assert len(rg) == 4


def test_history_features_use_only_past_games_and_next_game_targets() -> None:
    rg = pfw.receiver_game_table(_plays())
    h = pfw.history_features(rg, ("rec_yards", "n_targets"), last_k=2)
    a = h[h["receiver_player_id"] == "A"].sort_values("week")
    assert a["n_games"].tolist() == [1, 2, 3]
    assert a["cur_rec_yards"].tolist() == [10.0, 30.0, 12.0]
    assert a["std_rec_yards"].tolist() == pytest.approx([10.0, 20.0, 52 / 3])
    assert a["l3_rec_yards"].tolist() == pytest.approx([10.0, 20.0, 21.0])  # last two games
    assert a["next_rec_yards"].tolist()[:2] == [30.0, 12.0] and np.isnan(
        a["next_rec_yards"].iloc[2]
    )
    assert a["next_week"].tolist()[:2] == [2.0, 4.0]
    b = h[h["receiver_player_id"] == "B"].iloc[0]
    assert np.isnan(b["next_rec_yards"]) and b["n_games"] == 1


def test_forward_frame_drops_last_games_and_merges_ngs() -> None:
    rg = pfw.receiver_game_table(_plays())
    ngs = pd.DataFrame(
        {"season": [2020], "week": [1], "receiver_player_id": ["A"], "avg_separation": [2.9]}
    )
    f = pfw.forward_frame(rg, ngs)
    assert len(f) == 2 and set(f["receiver_player_id"]) == {"A"}
    assert f["next_week"].dtype.kind == "i"
    assert f.loc[f["week"] == 1, "cur_avg_separation"].iloc[0] == pytest.approx(2.9)
    assert np.isnan(f.loc[f["week"] == 2, "cur_avg_separation"].iloc[0])
    assert f.loc[f["week"] == 2, "std_avg_separation"].iloc[0] == pytest.approx(2.9)
    assert all(c in f.columns for c in pfw.FEATURE_SETS["HIST"])
    assert all(f"{a}_imp_separation_at_arrival__F1T" in f.columns for a in pfw.AGGS)


def test_population_mask_and_feature_sets_are_consistent() -> None:
    f = pd.DataFrame({"n_games": [1, 3, 5], "std_n_targets": [5.0, 2.0, 4.0]})
    assert pfw.population_mask(f, "all").tolist() == [True, True, True]
    assert pfw.population_mask(f, "regular").tolist() == [False, False, True]
    with pytest.raises(KeyError):
        pfw.population_mask(f, "nope")
    hist = pfw.FEATURE_SETS["HIST"]
    for cols in pfw.FEATURE_SETS.values():
        assert cols[: len(hist)] == hist
        assert len(cols) == len(set(cols))
        assert not any(c.startswith(("next_", "cur_rec_yards_next")) for c in cols)
    assert "week" in hist and "n_games" in hist


def test_delta_rows_sign_and_clustering() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(30, 10, n)
    ref = y + rng.normal(0, 8, n)
    better = y + rng.normal(0, 4, n)
    groups = rng.integers(0, 15, n)
    rows = pfw.delta_rows(y, {"HIST": ref, "B": better}, "HIST", groups, n_boot=200, seed=0)
    assert len(rows) == 1 and rows[0]["model"] == "B"
    r = rows[0]
    assert r["delta_se"] > 0 and r["ci_low"] > 0 and r["ci_low_week"] > 0
    assert r["delta_ae"] > 0 and r["n_weeks"] == len(np.unique(groups))
    assert r["delta_r2"] == pytest.approx(r["delta_se"] / np.var(y))
