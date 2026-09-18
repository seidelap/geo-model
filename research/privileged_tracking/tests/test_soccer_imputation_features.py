"""Tests for the soccer 02 imputation helpers on small synthetic frames (no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.soccer import imputation_features as imf


def _frame(seed: int = 0) -> pd.DataFrame:
    """A tiny 12-row events360-like frame with every column the helpers read."""
    rng = np.random.default_rng(seed)
    n = 12
    df = pd.DataFrame(
        {
            "gender": ["male"] * 6 + ["female"] * 6,
            "competition": ["La Liga"] * 3 + ["UEFA Euro"] * 3 + ["Women's World Cup"] * 6,
            "f_type": [
                "Pass",
                "Carry",
                "Pressure",
                "Pass",
                "Shot",
                "Pass",
                "Carry",
                "Pass",
                "Duel",
                "Pass",
                "Pass",
                "Pass",
            ],
            "f_type_id": [1, 3, 4, 1, 11, 1, 3, 1, 6, 1, 1, 1],
            "f_x": [10.0, 50.0, 70.0, 45.0, 110.0, 79.9, 80.0, 60.0, 30.0, 39.9, 65.0, np.nan],
            "f_y": rng.uniform(0, 80, n),
            "f_dist_goal": rng.uniform(5, 100, n),
            "f_goal_opening": rng.uniform(0, 1, n),
            "f_goal_bearing": rng.uniform(-1, 1, n),
            "f_play_pattern": ["Regular Play"] * 8
            + ["From Corner", "From Throw In", "Regular Play", "Bogus"],
            "f_position": ["Left Back"] * 11 + [None],
            "f_under_pressure": [False, True] * 6,
            "f_counterpress": False,
            "f_minute": np.arange(n, dtype=float),
            "f_t_period": np.arange(n, dtype=float) * 60,
            "f_period": 1,
            "f_home": True,
            "f_is_possession_team": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
            ],
            "f_score_for": 0,
            "f_score_against": 0,
            "f_score_diff": 0,
            "f_goals_total": 0,
            "f_pass_type": [
                "Open Play",
                None,
                None,
                "Corner",
                None,
                "Open Play",
                None,
                "Weird",
                None,
                "Open Play",
                "Open Play",
                "Open Play",
            ],
            "f_pass_body_part": [None] * n,
            "f_shot_body_part": [None] * 4 + ["Right Foot"] + [None] * 7,
            "f_shot_technique": [None] * n,
            "f_shot_type": [None] * n,
            "f_shot_first_time": [False] * n,
            "f_poss_elapsed": [0.0, 12.0, 15.0, 30.0, 40.0, 11.0, 9.0, 20.0, 25.0, 5.0, 60.0, 12.0],
            "f_poss_n_events": 1,
            "f_poss_n_passes": 0,
            "f_poss_ball_dist": 0.0,
            "f_poss_start_type": ["recovery"] * n,
            "f_poss_t_since_ft_entry": np.nan,
            "f_poss_in_final_third": False,
            "f_poss_start_x": 50.0,
            "f_poss_start_y": 40.0,
            "f_dt_prev": 1.0,
            "f_opp_def_x_60s_mean": np.nan,
            "f_opp_def_n_60s": 0,
            "f_opp_poss_last10s": False,
            "f_ball_speed_3": 1.0,
            "f_ball_dx_3": 0.0,
            "f_ball_dy_3": 0.0,
            "f_t_since_opp_def_action": np.nan,
            "f_after_duration": 1.0,
            "f_after_pass_length": np.nan,
            "f_after_pass_angle": np.nan,
            "f_after_pass_height": ["Ground Pass", None] * 6,
            "f_after_pass_switch": False,
            "f_after_pass_cross": False,
            "f_after_pass_through_ball": False,
            "f_after_pass_cut_back": False,
            "f_after_pass_end_x": np.nan,
            "f_after_pass_end_y": np.nan,
            "f_after_pass_end_dist_goal": np.nan,
            "f_after_pass_progress": np.nan,
            "f_after_carry_length": np.nan,
            "f_after_carry_end_x": np.nan,
            "f_after_carry_end_y": np.nan,
            "f_after_carry_progress": np.nan,
            "y_keeper_consistent": [1, 1, np.nan, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        "y_reliable": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "y_frame_ok": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            "y_def_line": [50.0, 30.0, 20.0, 10.0, 8.0, 25.0, 15.0, 40.0, 35.0, 60.0, 12.0, 18.0],
            "y_n_opp_ahead_of_ball": [10, 3, 5, 2, 1, 4, 3, 7, 6, 2, 3, 3],
            "y_n_opp_within_5": [0, 1, 2, 1, 3, 0, 0, 1, 2, 1, 0, 1],
            "y_n_opp_in_lane": [0, np.nan, np.nan, 1, np.nan, 2, np.nan, 0, np.nan, 1, 0, 1],
        }
    )
    for g in imf.WINDOW_GROUPS:
        df[f"f_w10_n_{g}"] = 0
    df["f_w10_n"] = 3
    for c in [
        "y_block_depth",
        "y_block_width",
        "y_block_length",
        "y_n_opp_within_10",
        "y_nearest_opp_dist",
        "y_n_opp_in_cone",
        "y_nearest_opp_dist_in_cone",
        "y_opp_keeper_dist_to_goal_line",
        "y_n_opp_within_3_of_end",
    ]:
        df[c] = rng.uniform(0, 10, n)
    seq = {}
    for k in range(1, imf.SEQ_LEN + 1):
        seq[f"seq_type_{k:02d}"] = np.zeros(n, dtype=np.int16)
        seq[f"seq_x_{k:02d}"] = np.full(n, np.nan, dtype=np.float32)
        seq[f"seq_y_{k:02d}"] = np.full(n, np.nan, dtype=np.float32)
        seq[f"seq_dt_{k:02d}"] = np.full(n, np.nan, dtype=np.float32)
        seq[f"seq_same_{k:02d}"] = np.full(n, -1, dtype=np.int8)
    return pd.concat([df, pd.DataFrame(seq, index=df.index)], axis=1)


def test_subset_masks_and_derived_targets() -> None:
    df = _frame()
    subs = imf.subset_masks(df)
    assert subs["all"].all() and subs["pass"].sum() == 7
    assert subs["poss"].tolist() == df["f_is_possession_team"].tolist()
    # settled: possession team, Regular Play, elapsed >= 10 -> rows 1, 3, 4, 5, 7, 10 (row 2 is not
    # possession team,
    # row 6 is 9 s old, rows 8/9 are set-piece phases, row 11 has a bogus pattern)
    assert np.where(subs["settled"])[0].tolist() == [1, 3, 4, 5, 7, 10]
    # middle third: 40 <= x < 80 and possession team -> rows 1, 3, 5, 7, 10 (row 6 is exactly 80,
    # row 9 is 39.9)
    assert np.where(subs["mid"])[0].tolist() == [1, 3, 5, 7, 10]
    thr = imf.deep_block_threshold(df)
    # reliable settled rows: 1, 4, 5, 7, 10 with def_line 30, 8, 25, 40, 12 -> 25th pct = 12
    assert thr == 12.0
    ys = imf.target_frame(df, thr)
    assert list(ys.columns) == [t.name for t in imf.TARGETS]
    db = ys["deep_block"]
    assert (
        db.notna().sum() == 5
        and db[4] == 1.0
        and db[10] == 1.0
        and db[1] == 0.0
        and np.isnan(db[3])
    )
    co = ys["counter_on"]
    assert (
        co.notna().sum() == 4
        and co[1] == 1.0
        and co[7] == 0.0
        and np.isnan(co[3])
        and np.isnan(co[0])
    )
    # shape targets: possession team + reliable; ball targets: frame_ok; pass targets: Pass rows
    assert np.isnan(ys["def_line"][2]) and np.isnan(ys["def_line"][3]) and ys["def_line"][0] == 50.0
    assert (
        ys["n_opp_within_5"][2] == 2
        and np.isnan(ys["n_opp_within_5"][9])
        and ys["n_opp_within_5"][3] == 1
    )
    assert ys["n_opp_in_lane"].notna().sum() == 6 and np.isnan(ys["n_opp_in_lane"][9])
    # cone targets: possession-team events only; keeper distance: possession team, x >= 60 and a
    # consistently flagged keeper (row 4 is inconsistent, row 2 is a Pressure, row 3 is at 45)
    assert np.isnan(ys["n_opp_in_cone"][2]) and not np.isnan(ys["n_opp_in_cone"][1])
    kd = ys["opp_keeper_dist_to_goal_line"]
    assert np.where(kd.notna())[0].tolist() == [5, 6, 7, 10]
    assert subs["poss_att"].tolist() == [False, False, False, False, True, True, True, True, False,
                                         False, True, False]


def test_build_design_encodes_vocabularies_and_context() -> None:
    df = _frame()
    xs = imf.build_design(df, "E2a")
    assert list(xs.columns) == imf.FEATURE_SETS["E2a"] and (xs.dtypes == np.float32).all()
    assert xs["f_gender"].tolist() == [0.0] * 6 + [1.0] * 6
    assert xs["f_comp_type"].tolist() == [0.0] * 3 + [1.0] * 9
    # play pattern: known strings map to their vocabulary index, unseen to len(vocab)
    assert xs["f_play_pattern"][0] == 0.0 and xs["f_play_pattern"][8] == 4.0
    assert xs["f_play_pattern"][11] == float(len(imf.VOCABS["f_play_pattern"]))
    # NaN strings stay NaN, unseen pass types get the unseen code, booleans are 0/1
    assert np.isnan(xs["f_position"][11]) and np.isnan(xs["f_pass_type"][1])
    assert xs["f_pass_type"][7] == float(len(imf.VOCABS["f_pass_type"]))
    assert xs["f_under_pressure"].tolist() == [0.0, 1.0] * 6
    assert xs["f_after_pass_height"][0] == 0.0 and np.isnan(xs["f_after_pass_height"][1])
    # nested sets and aliases
    assert imf.design_columns("E2r") == imf.FEATURE_SETS["E2"]
    assert set(imf.FEATURE_SETS["E1"]) < set(imf.FEATURE_SETS["E2"]) < set(imf.FEATURE_SETS["E2a"])
    assert not any(c.startswith("f_after_") for c in imf.FEATURE_SETS["E2"])
    assert all(c.startswith(("f_", "seq_")) for fs in imf.FEATURE_SETS.values() for c in fs)
    assert "f_type_id" in imf.categorical_columns(
        "E0"
    ) and "seq_type_01" in imf.categorical_columns("E3")
    assert imf.categorical_columns("loc") == []


def test_sequence_helpers() -> None:
    df = _frame().head(4).copy()
    # row 0: opponent Pressure (type 4) in slot 3 -> 2 events since; row 1: own Duel in slot 1 (not
    # opponent) and
    # opponent Pass in slot 2 (not defensive) -> none -> SEQ_LEN; row 2: opponent Clearance in
    # slot 1 -> 0
    df.loc[0, "seq_type_03"], df.loc[0, "seq_same_03"] = 4, 0
    df.loc[0, "seq_type_01"], df.loc[0, "seq_same_01"] = 1, 1
    df.loc[1, "seq_type_01"], df.loc[1, "seq_same_01"] = 6, 1
    df.loc[1, "seq_type_02"], df.loc[1, "seq_same_02"] = 1, 0
    df.loc[2, "seq_type_01"], df.loc[2, "seq_same_01"] = 8, 0
    df.loc[2, "seq_type_05"], df.loc[2, "seq_same_05"] = 12, 0
    assert imf.events_since_opp_def_action(df).tolist() == [2, imf.SEQ_LEN, 0, imf.SEQ_LEN]
    df.loc[0, "seq_x_01"], df.loc[0, "seq_y_01"], df.loc[0, "seq_dt_01"] = 60.0, 40.0, np.e - 1
    seq_feats = imf.seq_mlp_features(df)
    assert seq_feats.shape == (4, imf.SEQ_LEN * 12)
    slot1 = seq_feats[0, :12]
    # pad flag 0, group one-hot 'pass' (first group) 1, x/120, y/80, log1p(dt) = 1, same = 1
    assert (
        slot1[0] == 0.0
        and slot1[1] == 1.0
        and np.isclose(slot1[8], 0.5)
        and np.isclose(slot1[9], 0.5)
    )
    assert np.isclose(slot1[10], 1.0) and slot1[11] == 1.0
    pad = seq_feats[3, :12]
    assert pad[0] == 1.0 and pad[1:].sum() == 0.0
    # defensive group for the opponent clearance in row 2, slot 1
    assert seq_feats[2, 1 + list(imf.SEQ_TYPE_GROUPS).index("def")] == 1.0
    assert seq_feats[2, 11] == 0.0


def test_one_hot_design_and_baselines() -> None:
    xs = pd.DataFrame({"a": [1.0, np.nan, 3.0], "c": [0.0, 2.0, np.nan]}, dtype=np.float32)
    dense = imf.one_hot_design(xs, ["c"], {"c": 2})
    # a -> (value, missing), c -> 3 one-hots (codes 0, 1, unseen=2); NaN category -> all zero
    assert dense.shape == (3, 5)
    assert dense[:, 0].tolist() == [1.0, 0.0, 3.0] and dense[:, 1].tolist() == [0.0, 1.0, 0.0]
    assert (
        dense[0, 2:].tolist() == [1.0, 0.0, 0.0]
        and dense[1, 2:].tolist() == [0.0, 0.0, 1.0]
        and dense[2, 2:].sum() == 0
    )
    base = imf.bucket_mean_baseline(
        np.array(["p", "p", "c", "c", "x"]),
        np.array([1.0, 3.0, 5.0, np.nan, 9.0]),
        np.array(["p", "c", "zz"]),
        min_n=1,
    )
    assert base.tolist() == [2.0, 5.0, 4.5]
    ex, w1 = imf.rounded_agreement(np.array([1, 2, 3, 4]), np.array([1.2, 2.6, 4.9, 4.0]))
    assert ex == 0.5 and w1 == 0.75
    assert imf.pitch_third(np.array([0.0, 39.9, 40.0, 79.9, 80.0, np.nan])).tolist() == [
        "own",
        "own",
        "middle",
        "middle",
        "final",
        "unknown",
    ]
    assert imf.phase_label(np.array(["Regular Play", "From Corner", "From Keeper"])).tolist() == [
        "open_play",
        "set_piece",
        "open_play",
    ]
    assert imf.comp_type(pd.Series(["Serie A", "Copa America"])).tolist() == [0.0, 1.0]
