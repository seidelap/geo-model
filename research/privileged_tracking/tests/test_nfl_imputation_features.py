"""Unit tests for the pure feature functions of the NFL imputation stage (no data access)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import imputation_features as imf


def _season_pbp() -> pd.DataFrame:
    """Two teams, three weeks: A hosts B in week 1, B hosts A in week 2, A vs C in week 3."""
    rows = []
    # week 1: A passes 3 of 4 scrimmage plays (all shotgun), B runs 2 of 2
    for i, (pt, sg) in enumerate([("pass", 1), ("pass", 1), ("pass", 1), ("run", 1)]):
        rows.append(dict(game_id="g1", season=2017, week=1, play_id=i, posteam="A", defteam="B", play_type=pt,
                         shotgun=sg, no_huddle=0, pass_length="deep" if i == 0 else ("short" if pt == "pass" else None),
                         air_yards=10.0 if pt == "pass" else np.nan, sack=0, qb_hit=0, yards_gained=5.0))
    for i in range(2):
        rows.append(dict(game_id="g1", season=2017, week=1, play_id=10 + i, posteam="B", defteam="A", play_type="run",
                         shotgun=0, no_huddle=0, pass_length=None, air_yards=np.nan, sack=0, qb_hit=0, yards_gained=3.0))
    rows.append(dict(game_id="g1", season=2017, week=1, play_id=20, posteam="B", defteam="A", play_type="kickoff",
                     shotgun=0, no_huddle=0, pass_length=None, air_yards=np.nan, sack=0, qb_hit=0, yards_gained=0.0))
    # week 2: A vs B again, one play each
    rows.append(dict(game_id="g2", season=2017, week=2, play_id=1, posteam="A", defteam="B", play_type="run",
                     shotgun=0, no_huddle=1, pass_length=None, air_yards=np.nan, sack=0, qb_hit=0, yards_gained=1.0))
    rows.append(dict(game_id="g2", season=2017, week=2, play_id=2, posteam="B", defteam="A", play_type="pass",
                     shotgun=1, no_huddle=0, pass_length="short", air_yards=2.0, sack=1, qb_hit=1, yards_gained=-7.0))
    # week 3: A vs C
    rows.append(dict(game_id="g3", season=2017, week=3, play_id=1, posteam="A", defteam="C", play_type="pass",
                     shotgun=1, no_huddle=0, pass_length="short", air_yards=4.0, sack=0, qb_hit=0, yards_gained=4.0))
    return pd.DataFrame(rows)


def test_play_indicators_denominators() -> None:
    ind = imf.play_indicators(_season_pbp())
    # kickoff row contributes nothing
    assert ind.loc[6].isna().all()
    # run plays have NaN pass indicators and vice versa
    assert np.isnan(ind.loc[3, "ind_air_yards"]) and ind.loc[3, "ind_rush_yards"] == 5.0
    assert ind.loc[0, "ind_deep"] == 1.0 and ind.loc[1, "ind_deep"] == 0.0 and np.isnan(ind.loc[3, "ind_deep"])


def test_pbp_team_tendencies_are_strictly_prior_and_shrunk() -> None:
    pbp = _season_pbp()
    alpha = 4.0
    t = imf.pbp_team_tendencies(pbp, alpha=alpha)
    assert list(t.index) == list(pbp.index) and t.shape == (len(pbp), len(imf.TENDENCIES))
    # week 1: nothing prior -> exactly the constant priors
    assert t.loc[0, "tend_pos_pass"] == 0.58 and t.loc[4, "tend_def_pass"] == 0.58
    # week 2, A on offense: A's week-1 pass rate 3/4 shrunk towards 0.58 with alpha plays
    expect = (3.0 + alpha * 0.58) / (4.0 + alpha)
    assert abs(t.loc[7, "tend_pos_pass"] - expect) < 1e-12
    # week 2, B on defense faced A's 4 scrimmage plays, 3 passes
    assert abs(t.loc[7, "tend_def_pass"] - (3.0 + alpha * 0.58) / (4.0 + alpha)) < 1e-12
    # week 2, A on defense faced B's 2 runs (the kickoff does not count)
    assert abs(t.loc[8, "tend_def_pass"] - (0.0 + alpha * 0.58) / (2.0 + alpha)) < 1e-12
    # B on offense in week 2 ran on both week-1 plays
    assert abs(t.loc[8, "tend_pos_pass"] - (0.0 + alpha * 0.58) / (2.0 + alpha)) < 1e-12
    # week 3: A's cumulative shotgun over weeks 1-2 = 4 of 5 scrimmage plays; own game excluded
    assert abs(t.loc[9, "tend_pos_shotgun"] - (4.0 + alpha * 0.55) / (5.0 + alpha)) < 1e-12
    # a team never seen on defense before (C) gets the prior
    assert t.loc[9, "tend_def_sack"] == 0.065
    # the week-1 rows of a game never see that game's own plays (A passed on all week-1 plays)
    assert t.loc[3, "tend_pos_pass"] == 0.58


def test_team_encoder_leave_one_game_out() -> None:
    team = np.array(["A", "A", "A", "B", "B"], dtype=object)
    game = np.array(["g1", "g1", "g2", "g1", "g2"], dtype=object)
    y = np.array([1.0, 3.0, 5.0, 0.0, np.nan])
    enc = imf.TeamEncoder(alpha=2.0).fit(team, game, y)
    assert enc.global_mean == 2.25
    logo = enc.transform_train(team, game)
    # A in g1 sees only g2 (5.0, one play): (5 + 2*2.25) / (1 + 2)
    assert abs(logo[0] - (5.0 + 4.5) / 3.0) < 1e-12 and logo[0] == logo[1]
    # A in g2 sees g1 (1, 3): (4 + 4.5) / (2 + 2)
    assert abs(logo[2] - 8.5 / 4.0) < 1e-12
    # B's NaN play contributes nothing; B in g2 sees g1 (0.0)
    assert abs(logo[4] - (0.0 + 4.5) / 3.0) < 1e-12
    full = enc.transform(np.array(["A", "B", "Z"], dtype=object))
    assert abs(full[0] - (9.0 + 4.5) / 5.0) < 1e-12 and abs(full[1] - 4.5 / 3.0) < 1e-12
    assert full[2] == enc.global_mean
    assert set(enc.table()) == {"A", "B"} and abs(enc.table()["A"] - full[0]) < 1e-12


def test_personnel_features_and_receiver_group() -> None:
    pers = imf.personnel_features(pd.Series(["1 RB, 1 TE, 3 WR", "6 OL, 2 RB, 1 TE, 1 WR", None]),
                                  pd.Series(["4 DL, 2 LB, 5 DB", "3 DL, 4 LB, 4 DB", "2 DL, 3 LB, 6 DB"]))
    assert pers.loc[0, ["pers_RB", "pers_TE", "pers_WR", "pers_OL", "pers_QB"]].tolist() == [1, 1, 3, 5, 1]
    assert pers.loc[1, ["pers_RB", "pers_OL"]].tolist() == [2, 6]
    assert np.isnan(pers.loc[2, "pers_RB"]) and pers.loc[2, ["pers_DL", "pers_LB", "pers_DB"]].tolist() == [2, 3, 6]
    g = imf.receiver_group(pd.Series(["WR", "TE", "FB", "SAF", None, "G"]))
    assert g.tolist() == ["WR", "TE", "RB", "other", None, "OL"]


def test_encode_category_fixed_vocab() -> None:
    codes = imf.encode_category(pd.Series(["left", "right", "middle", "up", None]), imf.LOCATIONS)
    assert codes[:3].tolist() == [0.0, 2.0, 1.0] and np.isnan(codes[3]) and np.isnan(codes[4])


def test_build_design_columns_and_missing_inputs() -> None:
    df = pd.DataFrame({
        "down": [1, 3, np.nan], "ydstogo": [10, 2, 5], "yardline_100": [75, 20, 50], "qtr": [1, 4, 2],
        "half_seconds_remaining": [1800, 100, 900], "game_seconds_remaining": [3600, 100, 2700],
        "score_differential": [0, -7, 3], "wp": [0.5, 0.1, 0.6], "ep": [1.0, 3.0, 2.0], "xpass": [0.5, 0.9, 0.6],
        "goal_to_go": [0, 0, 0], "shotgun": [0, 1, 1], "no_huddle": [0, 1, 0], "is_home": [1, 0, 1],
        "offense_formation": ["I_FORM", "SHOTGUN", None], "offense_personnel": ["2 RB, 1 TE, 2 WR", "1 RB, 1 TE, 3 WR", None],
        "defense_personnel": ["4 DL, 3 LB, 4 DB", "4 DL, 2 LB, 5 DB", None],
        "play_type": ["run", "pass", "pass"], "pass_length": [None, "deep", "short"], "pass_location": [None, "left", "middle"],
        "air_yards": [np.nan, 25.0, 3.0], "yards_after_catch": [np.nan, 0.0, 4.0], "complete_pass": [0, 0, 1],
        "interception": [0, 0, 0], "sack": [0, 0, 0], "qb_hit": [0, 1, 0], "run_location": ["left", None, None],
        "run_gap": ["end", None, None], "yards_gained": [4, 0, 7], "epa": [0.1, -0.5, 0.8], "qb_dropback": [0, 1, 1],
        "qb_scramble": [0, 0, 0], "receiver_position": [None, "WR", "TE"], "defenders_in_box": [8, 6, 6],
        "number_of_pass_rushers": [np.nan, 4, 5], "te_off": [1.0, 1.0, 1.0],
    })
    for c in imf.TENDENCY_COLS:
        df[c] = 0.5
    X = imf.build_design(df)
    assert list(X.columns) == imf.DESIGN_COLS and len(X) == 3
    assert X.loc[0, "dd_bucket"] == imf.DD_BUCKETS.index("d1_long") and X.loc[1, "dd_bucket"] == imf.DD_BUCKETS.index("d3_short")
    assert X.loc[2, "dd_bucket"] == imf.DD_BUCKETS.index("dna_mid")
    assert X.loc[0, "formation"] == imf.FORMATIONS.index("I_FORM") and np.isnan(X.loc[2, "formation"])
    assert X.loc[0, ["pers_RB", "pers_LB"]].tolist() == [2, 3]
    assert X.loc[0, "is_run"] == 1 and X.loc[1, "is_pass"] == 1 and X.loc[1, "pass_length"] == 1
    assert X.loc[1, "receiver_group"] == imf.RECEIVER_GROUPS.index("WR") and np.isnan(X.loc[0, "receiver_group"])
    assert X["te_off"].tolist() == [1.0, 1.0, 1.0] and X["te_def"].isna().all()
    # nested feature sets
    assert set(imf.FEATURE_SETS["F0n"]) < set(imf.FEATURE_SETS["F0"]) < set(imf.FEATURE_SETS["F0P"])
    assert set(imf.FEATURE_SETS["F0P"]) < set(imf.FEATURE_SETS["F1"]) < set(imf.FEATURE_SETS["F2"])
    # F0 must not contain any post-play or charting column
    assert not set(imf.FEATURE_SETS["F0"]) & (set(imf.POSTPLAY_COLS) | set(imf.PERSONNEL_COLS) | set(imf.OFFICIAL_COLS))
    # a bare situation frame (no participation, no post-play columns) still builds
    X0 = imf.build_design(df[["down", "ydstogo", "yardline_100", "qtr", "wp", "shotgun", "play_type"]])
    assert X0.shape == (3, len(imf.DESIGN_COLS)) and X0["formation"].isna().all()


def test_target_frame_subsets_and_pressure_flag() -> None:
    plays = pd.DataFrame({
        "is_pass_play": [True, False, True, False], "pbp_play_type": ["pass", "run", "pass", "qb_kneel"],
        "min_def_dist_qb_throw": [1.2, np.nan, 3.0, np.nan], "box_count_tuned": [6, 7, 5, 8],
    })
    for t in imf.TARGETS:
        if t.source_col not in plays.columns:
            plays[t.source_col] = [1.0, 2.0, 3.0, 4.0]
    y = imf.target_frame(plays)
    assert list(y.columns) == [t.name for t in imf.TARGETS]
    assert y["pressure_derived"].tolist()[0] == 1.0 and y["pressure_derived"].tolist()[2] == 0.0
    assert np.isnan(y.loc[1, "pressure_derived"]) and np.isnan(y.loc[3, "time_to_throw"])
    # run subset: nflfastR run that is not a BDB pass play; kneel excluded
    assert y["box_count_run"].tolist()[1] == 7 and np.isnan(y.loc[0, "box_count_run"]) and np.isnan(y.loc[3, "box_count_run"])
    assert y["box_count_tuned"].tolist() == [6, 7, 5, 8]
    masks = imf.subset_masks(pd.DataFrame({"play_type": ["pass", "run", "punt", "qb_kneel", "kickoff"]}))
    assert masks["pass"].tolist() == [True, False, False, False, False]
    assert masks["run"].tolist() == [False, True, False, False, False]
    # pre-snap students apply to scrimmage plays only on a plain nflfastR frame
    assert masks["all"].tolist() == [True, True, False, True, False]
    assert y.shape[0] == 4 and imf.subset_masks(plays)["all"].all()


def test_bucket_mean_baseline_and_rounded_agreement() -> None:
    b_tr = np.array(["a", "a", "b", "b", "c"], dtype=object)
    y_tr = np.array([1.0, 3.0, 10.0, np.nan, 5.0])
    pred = imf.bucket_mean_baseline(b_tr, y_tr, np.array(["a", "b", "zz"], dtype=object))
    assert pred.tolist() == [2.0, 10.0, np.mean([1.0, 3.0, 10.0, 5.0])]
    pred2 = imf.bucket_mean_baseline(b_tr, y_tr, np.array(["c"], dtype=object), min_count=2)
    assert pred2[0] == np.mean([1.0, 3.0, 10.0, 5.0])
    exact, within = imf.rounded_agreement(np.array([1, 2, 3]), np.array([1.4, 2.6, 4.2]))
    assert abs(exact - 1 / 3) < 1e-12 and within == 1.0
    exact2, within2 = imf.rounded_agreement(np.array([1, 2, 3]), np.array([1.4, 2.4, 5.2]))
    assert abs(exact2 - 2 / 3) < 1e-12 and abs(within2 - 2 / 3) < 1e-12
