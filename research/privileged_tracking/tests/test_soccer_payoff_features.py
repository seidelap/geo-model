"""Tests for the soccer 03 payoff helpers on small synthetic frames (no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.soccer import payoff_features as pf
from research.privileged_tracking.tests.test_soccer_imputation_features import _frame

# ---------------------------------------------------------------------------
# assist link
# ---------------------------------------------------------------------------


def test_key_pass_links_reads_only_shots_and_keeps_missing_key_pass():
    events = [
        {"id": "s1", "type": {"name": "Shot"}, "shot": {"key_pass_id": "p1", "statsbomb_xg": 0.3}},
        {"id": "s2", "type": {"name": "Shot"}, "shot": {"statsbomb_xg": 0.1}},
        {"id": "p1", "type": {"name": "Pass"}, "pass": {"assisted_shot_id": "s1"}},
        {"id": "x", "type": {"name": "Carry"}},
    ]
    d = pf.key_pass_links(events)
    assert list(d.columns) == ["event_id", "key_pass_id"]
    assert d["event_id"].tolist() == ["s1", "s2"]
    assert d["key_pass_id"].tolist()[0] == "p1"
    assert d["key_pass_id"].isna().tolist() == [False, True]
    assert pf.key_pass_links([]).empty


def _passes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["p1", "p2"],
            "f_x": [80.0, 60.0],
            "f_y": [30.0, 40.0],
            "f_pass_type": ["Open Play", "Corner"],
            "f_pass_body_part": ["Right Foot", "Left Foot"],
            "f_under_pressure": [True, False],
            "f_after_pass_length": [15.0, 30.0],
            "f_after_pass_angle": [0.1, -0.5],
            "f_after_pass_height": ["Ground Pass", "High Pass"],
            "f_after_pass_switch": [False, False],
            "f_after_pass_cross": [False, True],
            "f_after_pass_through_ball": [True, False],
            "f_after_pass_cut_back": [False, False],
            "f_after_pass_end_x": [95.0, 100.0],
            "f_after_pass_end_y": [35.0, 45.0],
            "f_t_period": [100.0, 200.0],
        }
    )


def test_assist_block_joins_key_pass_and_computes_dt():
    shots = pd.DataFrame(
        {"key_pass_id": ["p1", None, "missing", "p2"], "f_t_period": [102.0, 50.0, 60.0, 203.5]}
    )
    a = pf.assist_block(shots, _passes())
    assert a["a_has_assist"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert a["a_dt"].tolist()[0] == pytest.approx(2.0)
    assert a["a_dt"].tolist()[3] == pytest.approx(3.5)
    assert np.isnan(a["a_dt"].to_numpy(dtype=float)[1:3]).all()
    assert a["a_pass_height"].tolist()[0] == "Ground Pass"
    assert a["a_pass_cross"].tolist()[3] is True or a["a_pass_cross"].tolist()[3] == 1
    assert "a_t_period" not in a.columns
    enc = pf.encode_assist(a)
    assert enc.dtypes.map(lambda t: t == np.float32).all()
    # fixed vocabularies: Ground Pass -> 0, High Pass -> 2, missing -> NaN
    assert enc["a_pass_height"].tolist()[0] == 0.0
    assert enc["a_pass_height"].tolist()[3] == 2.0
    assert np.isnan(enc["a_pass_height"].to_numpy()[1])
    assert enc["a_pass_through_ball"].tolist()[0] == 1.0
    assert np.isnan(enc["a_pass_through_ball"].to_numpy()[1])


# ---------------------------------------------------------------------------
# designs and state blocks
# ---------------------------------------------------------------------------


def _shots_frame() -> pd.DataFrame:
    df = pd.concat([_frame(seed=s) for s in range(3)], ignore_index=True)
    df = df[df["f_type"] == "Shot"].reset_index(drop=True)
    df = pd.concat([df] * 4, ignore_index=True)
    df["f_shot_body_part"] = ["Right Foot", "Left Foot", "Head"] * 4
    df["f_after_duration"] = np.arange(len(df), dtype=float)
    df["y_n_opp_in_cone"] = 1.0
    df["sff_n_opp_in_cone"] = 2.0
    df["oracle_statsbomb_xg"] = 0.2
    df["post_shot_outcome"] = "Goal"
    return df


def test_shot_event_design_reads_only_event_columns():
    df = _shots_frame()
    x = pf.shot_event_design(df)
    assert all(c.startswith("f_") for c in x.columns)
    for c in pf.SHOT_DROP:
        assert c not in x.columns
    assert not any(c.startswith(("y_", "sff_", "oracle_", "post_")) for c in x.columns)
    assert "f_shot_body_part" in x.columns  # varies -> kept
    assert x.dtypes.map(lambda t: t == np.float32).all()
    # every stage-02 E2 design column not dropped and not constant is present
    expected = set(imf.design_columns("E2")) - set(pf.SHOT_DROP)
    assert set(x.columns) <= expected


def test_pass_event_design_after_flag_controls_post_instant_block():
    df = pd.concat([_frame(seed=s) for s in range(3)], ignore_index=True)
    df = df[df["f_type"] == "Pass"].reset_index(drop=True)
    df["f_after_pass_end_x"] = np.linspace(10, 100, len(df))
    df["f_after_pass_length"] = np.linspace(5, 40, len(df))
    with_after = pf.pass_event_design(df, with_after=True)
    no_after = pf.pass_event_design(df, with_after=False)
    assert "f_after_pass_end_x" in with_after.columns
    assert not any(c.startswith("f_after_") for c in no_after.columns)
    for c in pf.PASS_DROP:
        assert c not in with_after.columns


def test_state_block_sources_and_clipping():
    t = pd.DataFrame(
        {
            "imp_n_opp_in_cone__E2": [-0.2, 1.5],
            "imp_block_depth__E2": [10.0, -1.0],
            "y_n_opp_in_cone": [0.0, 2.0],
            "y_block_depth": [np.nan, 12.0],
            "sff_n_opp_in_cone": [1.0, 3.0],
            "sff_n_opp_ahead_of_ball": [5.0, 6.0],
            "a_imp_n_opp_in_lane__E2a": [-0.5, 0.7],
            "a_y_n_opp_in_lane": [0.0, 1.0],
        }
    )
    names = ("n_opp_in_cone", "block_depth")
    imp_b = pf.state_block(t, names, "imp", "E2", prefix="s_")
    assert imp_b["s_n_opp_in_cone"].tolist() == [0.0, 1.5]  # clipped at 0
    assert imp_b["s_block_depth"].tolist() == [10.0, 0.0]
    o360 = pf.state_block(t, names, "oracle360", prefix="s_")
    assert np.isnan(o360["s_block_depth"].to_numpy()[0])
    osh = pf.state_block(t, names, "oracleshot", prefix="s_")
    assert osh["s_block_depth"].tolist() == [5.0, 6.0]  # sff proxy
    a = pf.state_block(t, ("n_opp_in_lane",), "imp", "E2a", prefix="as_", col_prefix="a_")
    assert a["as_n_opp_in_lane"].tolist() == [0.0, pytest.approx(0.7)]
    with pytest.raises(KeyError):
        pf.state_block(t, names, "bogus")


def test_drop_constant_columns():
    x = pd.DataFrame({"a": [1.0, 1.0, np.nan], "b": [1.0, 2.0, 3.0], "c": [np.nan] * 3})
    assert list(pf.drop_constant_columns(x).columns) == ["b"]


def test_categorical_in_picks_stage02_and_assist_vocab_columns():
    x = pd.DataFrame(columns=["f_x", "f_play_pattern", "a_pass_height", "s_block_depth", "a_dt"])
    assert pf.categorical_in(x) == ["f_play_pattern", "a_pass_height"]


# ---------------------------------------------------------------------------
# labels, slices, subsample, soft labels
# ---------------------------------------------------------------------------


def test_pass_completion_label_and_keep_mask():
    o = np.array(
        [None, "Complete", "Incomplete", "Out", "Unknown", "Injury Clearance", "Pass Offside"],
        dtype=object,
    )
    y, keep = pf.pass_completion(o)
    assert y.tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert keep.tolist() == [True, True, True, True, False, False, True]


def test_pattern_group_and_bands():
    p = np.array(
        ["Regular Play", "From Counter", "From Corner", "From Throw In", "Other", "From Keeper"],
        dtype=object,
    )
    assert pf.pattern_group(p).tolist() == [
        "regular",
        "counter",
        "set_piece",
        "set_piece",
        "other",
        "other",
    ]
    assert pf.distance_band(np.array([0.0, 7.9, 8.0, 17.9, 24.9, 30.0, np.nan])).tolist() == [
        "0-8",
        "0-8",
        "8-12",
        "12-18",
        "18-25",
        "25+",
        "unknown",
    ]
    assert pf.length_band(np.array([3.0, 10.0, 34.9, 35.0])).tolist() == [
        "0-10",
        "10-20",
        "20-35",
        "35+",
    ]


def test_stratified_subsample_is_seeded_and_match_wise():
    m = np.repeat(np.arange(20), 100)
    keep = pf.stratified_subsample(m, 600, seed=3)
    assert keep.sum() == pytest.approx(600, abs=80)
    assert np.array_equal(keep, pf.stratified_subsample(m, 600, seed=3))
    assert not np.array_equal(keep, pf.stratified_subsample(m, 600, seed=4))
    # a match's sample does not depend on which other matches are present
    sub = m < 5
    assert np.array_equal(pf.stratified_subsample(m[sub], 150, seed=3), keep[sub])
    assert pf.stratified_subsample(m, 10_000, seed=0).all()


def test_soft_label_blends_and_clips():
    y = np.array([0.0, 1.0])
    t = np.array([0.0, 1.0])
    assert pf.soft_label(y, t, 1.0).tolist() == pytest.approx([1e-4, 1 - 1e-4])
    assert pf.soft_label(np.array([1.0]), np.array([0.5]), 0.5).tolist() == pytest.approx([0.75])
    assert pf.soft_label(y, t, 0.0).tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# metrics and deltas
# ---------------------------------------------------------------------------


def test_binary_metrics_and_calibration_frame():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(float)
    m = pf.binary_metrics(y, p)
    assert m["n"] == 500 and m["positives"] == int(y.sum())
    assert 0 < m["log_loss"] < 1 and 0 < m["brier"] < 0.25 and m["auc"] > 0.7
    assert m["ece"] < 0.1
    m2 = pf.binary_metrics(y, np.where(np.arange(500) < 10, np.nan, p))
    assert m2["n"] == 490
    cal = pf.calibration_frame(y, {"a": p, "b": np.full(500, y.mean())}, n_bins=5)
    assert set(cal["variant"]) == {"a", "b"}
    assert cal[cal["variant"] == "a"]["n"].sum() == 500
    assert cal[cal["variant"] == "b"]["obs"].std() < cal[cal["variant"] == "a"]["obs"].std()


def test_paired_delta_sign_and_ci():
    rng = np.random.default_rng(1)
    n = 2000
    truth = rng.uniform(0.05, 0.6, n)
    y = (rng.uniform(0, 1, n) < truth).astype(float)
    match = rng.integers(0, 40, n)
    good = np.clip(truth + rng.normal(0, 0.02, n), 0.01, 0.99)
    bad = np.clip(truth + rng.normal(0, 0.25, n), 0.01, 0.99)
    d = pf.paired_delta(y, bad, good, match, n_boot=300, seed=0)
    assert d["delta_log_loss"] > 0 and d["ci_low"] > 0 and d["significant"]
    assert d["ci_low_clustered"] <= d["delta_log_loss"] <= d["ci_high_clustered"]
    assert d["delta_brier"] > 0
    d2 = pf.paired_delta(y, good, bad, match, n_boot=300, seed=0)
    assert d2["delta_log_loss"] == pytest.approx(-d["delta_log_loss"])
    same = pf.paired_delta(y, good, good, match, n_boot=100)
    assert same["delta_log_loss"] == 0.0 and not same["significant"]
    assert pf.paired_delta(y[:1], good[:1], bad[:1], match[:1]) == {"n": 1}


def test_tables_shapes():
    rng = np.random.default_rng(2)
    n = 600
    truth = rng.uniform(0.05, 0.6, n)
    y = (rng.uniform(0, 1, n) < truth).astype(float)
    match = rng.integers(0, 12, n)
    preds = {
        "EVENT": np.clip(truth + rng.normal(0, 0.1, n), 0.01, 0.99),
        "EVENT+IMP": np.clip(truth + rng.normal(0, 0.05, n), 0.01, 0.99),
        "BASE": np.full(n, y.mean()),
    }
    m = pf.metrics_table(y, preds)
    assert m["variant"].tolist() == ["EVENT", "EVENT+IMP", "BASE"]
    d = pf.deltas_table(y, preds, "EVENT", match, n_boot=100)
    assert d["variant"].tolist() == ["EVENT+IMP", "BASE"] and (d["reference"] == "EVENT").all()
    slices = {
        "band": np.where(np.arange(n) < 300, "a", "b").astype(object),
        "tiny": np.where(np.arange(n) < 10, "x", "y").astype(object),
    }
    s = pf.slice_table(y, preds, "EVENT", slices, match, n_boot=50, min_n=50)
    assert set(s["level"]) == {"a", "b", "y"}  # 'x' has fewer than min_n rows
    assert s[(s["variant"] == "EVENT") & (s["level"] == "a")]["n"].iloc[0] == 300
    assert s[s["variant"] == "EVENT"]["delta_log_loss"].isna().all()
    assert s[s["variant"] == "EVENT+IMP"]["delta_log_loss"].notna().all()


def test_variant_catalogue_is_consistent():
    names = [v.name for v in pf.XG_VARIANTS]
    assert len(names) == len(set(names))
    assert pf.variant_by_name(pf.XG_VARIANTS, "EVENT+IMP").state == "imp"
    with pytest.raises(KeyError):
        pf.variant_by_name(pf.XG_VARIANTS, "nope")
    assert set(pf.SFF_MAP) == set(pf.SHOT_STATE)
    assert all(t in imf.TARGET_BY_NAME for t in pf.SHOT_STATE + pf.ASSIST_STATE)
    assert all(t in imf.TARGET_BY_NAME or t == "nearest_opp_to_receiver" for t in pf.PASS_STATE)
