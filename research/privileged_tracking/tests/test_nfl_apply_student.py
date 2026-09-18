"""Tests for the NFL 03 student bundle / applier and the driver's pure helpers (no data access)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import apply_student as aps
from research.privileged_tracking.nfl import imputation as imp
from research.privileged_tracking.nfl import imputation_features as imf


def _tiny_bundle(tmp_path, fset: str = "F0") -> tuple[aps.StudentBundle, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(np.nan, index=range(n), columns=imf.DESIGN_COLS)
    X["down"] = rng.integers(1, 5, n).astype(float)
    X["ydstogo"] = rng.integers(1, 15, n).astype(float)
    X["te_off"] = rng.normal(size=n)
    X["te_def"] = rng.normal(size=n)
    y = 2.0 * X["ydstogo"].to_numpy() + X["te_def"].to_numpy() + rng.normal(scale=0.1, size=n)
    feats = imf.FEATURE_SETS[fset]
    booster = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 7, "learning_rate": 0.2,
                         "min_data_in_leaf": 5, "num_threads": 1, "seed": 0, "deterministic": True},
                        lgb.Dataset(X[feats], label=y), num_boost_round=60)
    st = aps.Student(target="qb_depth", kind="reg", subset="all", model_str=booster.model_to_string(), best_iter=60,
                     te_off={"NE": 1.0, "KC": -1.0}, te_def={"NE": 0.5, "KC": -0.5}, te_global=0.0, train_n=n,
                     y_mean=float(y.mean()), y_sd=float(y.std()), oof_mean=float(y.mean()), oof_sd=float(y.std()))
    st_pass = aps.Student(**{**st.__dict__, "target": "time_to_throw", "subset": "pass"})
    b = aps.StudentBundle(feature_set=fset, features=list(feats), students={"qb_depth": st, "time_to_throw": st_pass})
    return b, X


def test_bundle_round_trip_and_subset_masks(tmp_path) -> None:
    b, X = _tiny_bundle(tmp_path)
    p = b.save(root=tmp_path)
    assert p == tmp_path / "models" / "students_F0.joblib" and p.exists()
    b2 = aps.StudentBundle.load("F0", root=tmp_path)
    assert b2.features == b.features and set(b2.students) == {"qb_depth", "time_to_throw"}
    assert b2.students["qb_depth"].te_def == {"NE": 0.5, "KC": -0.5}
    Xa = X.head(6).copy()
    frame = pd.DataFrame({"play_type": ["pass", "run", "pass", "punt", "run", "pass"]})
    masks = imf.subset_masks(frame)
    pos = np.array(["NE", "KC", "ZZ", "NE", "KC", "NE"], dtype=object)
    de = np.array(["KC", "NE", "NE", "KC", "ZZ", "KC"], dtype=object)
    out = b2.predict(Xa, pos, de, masks)
    assert list(out.columns) == ["imp_qb_depth__F0", "imp_time_to_throw__F0"]
    # pre-snap student on scrimmage plays (not the punt), pass student only on pass plays
    assert out["imp_qb_depth__F0"].notna().tolist() == [True, True, True, False, True, True]
    assert out["imp_time_to_throw__F0"].notna().tolist() == [True, False, True, False, False, True]
    # the team encodings are filled from the bundle (unseen team -> global mean): the prediction must change with te_def
    Xb = Xa.copy()
    Xb["te_def"] = np.nan
    out_b = b2.predict(Xb, pos, np.array(["NE"] * 6, dtype=object), masks)
    assert not np.allclose(out["imp_qb_depth__F0"].to_numpy(), out_b["imp_qb_depth__F0"].to_numpy())
    # and the model is a function of the features: doubling ydstogo raises the (monotone) prediction
    Xc = Xa.copy()
    Xc["ydstogo"] = 14.0
    Xd = Xa.copy()
    Xd["ydstogo"] = 1.0
    hi = b2.predict(Xc, pos, de, masks)["imp_qb_depth__F0"]
    lo = b2.predict(Xd, pos, de, masks)["imp_qb_depth__F0"]
    assert (hi[masks["all"]] > lo[masks["all"]]).all() and np.isnan(hi[3]) and np.isnan(lo[3])


def test_prepare_pbp_adds_tendencies_participation_and_receiver_position() -> None:
    pbp = pd.DataFrame({
        "game_id": ["2018_01_A_B"] * 3 + ["2018_02_B_A"] * 2, "old_game_id": ["2018090900"] * 3 + ["2018091600"] * 2,
        "play_id": [1.0, 2.0, 3.0, 1.0, 2.0], "season": 2018, "week": [1, 1, 1, 2, 2], "posteam": ["A", "A", "B", "B", "A"],
        "defteam": ["B", "B", "A", "A", "B"], "posteam_type": ["home", "home", "away", "home", "away"],
        "home_team": "A", "away_team": "B", "receiver_player_id": ["00-1", None, "00-2", None, "00-9"],
    })
    for c in imf.PBP_INPUT_COLS:
        if c not in pbp.columns:
            pbp[c] = np.nan
    pbp["play_type"] = ["pass", "run", "pass", "run", "pass"]
    pbp["shotgun"] = [1, 0, 1, 0, 1]
    part = pd.DataFrame({"old_game_id": ["2018090900", "2018091600"], "play_id": [1, 2],
                         "offense_formation": ["SHOTGUN", "I_FORM"], "offense_personnel": ["1 RB, 1 TE, 3 WR", "2 RB, 1 TE, 2 WR"],
                         "defense_personnel": ["4 DL, 2 LB, 5 DB", "4 DL, 3 LB, 4 DB"], "defenders_in_box": [6, 8],
                         "number_of_pass_rushers": [4, np.nan]})
    positions = pd.Series({"00-1": "WR", "00-2": "TE"})
    out = aps.prepare_pbp(pbp, part, positions)
    assert out["is_home"].tolist() == [1, 1, 0, 1, 0]
    assert set(imf.TENDENCY_COLS) <= set(out.columns)
    # week 1 -> priors, week 2 -> A's week-1 shotgun rate (1 of 2 scrimmage plays) shrunk to 0.55
    assert out.loc[0, "tend_pos_shotgun"] == 0.55
    assert abs(out.loc[4, "tend_pos_shotgun"] - (1 + imf.TENDENCY_ALPHA * 0.55) / (2 + imf.TENDENCY_ALPHA)) < 1e-12
    assert out["offense_formation"].tolist()[0] == "SHOTGUN" and out["offense_formation"].isna().tolist() == [False, True, True, True, False]
    assert out.loc[4, "defenders_in_box"] == 8
    rp = out["receiver_position"]
    assert rp.tolist()[0] == "WR" and rp.tolist()[2] == "TE" and rp.isna().tolist() == [False, True, False, True, True]
    X = imf.build_design(out)
    assert X.loc[0, "formation"] == imf.FORMATIONS.index("SHOTGUN") and X.loc[0, "receiver_group"] == imf.RECEIVER_GROUPS.index("WR")


def test_driver_pure_helpers() -> None:
    y = np.array([0, 1, 1, 0, 1, 0], dtype=float)
    p = np.array([0.1, 0.9, 0.8, 0.3, 0.6, 0.4])
    m = imp.score("binary", y, p)
    assert m["n"] == 6 and m["auc"] == 1.0 and m["acc"] == 1.0 and 0 < m["bss"] < 1
    assert abs(imp.skill_of("binary", m) - m["bss"]) < 1e-12
    base = imp.score("binary", y, np.full(6, 0.5))
    assert abs(base["bss"]) < 1e-12 and base["auc"] == 0.5
    yc = np.array([4, 6, 7, 5], dtype=float)
    pc = np.array([4.4, 5.6, 7.0, 6.9])
    mc = imp.score("count", yc, pc)
    assert mc["exact_rounded"] == 0.75 and mc["within_1"] == 0.75 and abs(imp.skill_of("count", mc) - mc["r2"]) < 1e-12
    assert np.allclose(imp.per_sample_loss("reg", yc, pc), np.abs(yc - pc))
    assert np.allclose(imp.per_sample_loss("binary", y, p), -(y * np.log(p) + (1 - y) * np.log(1 - p)))
    assert imp._bucket_air_yards(np.array([-2, 0, 5, 12, 40, np.nan])).tolist() == ["<0", "0-5", "5-10", "10-20", "20+", "na"]
    assert imp._bucket_ttt(np.array([1.9, 2.0, 2.7, 3.5, 6.0, np.nan])).tolist() == ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0+", "na"]
    assert [imp._label(s) for s in (0.6, 0.3, 0.1, 0.0, np.nan)] == ["well", "partly", "weakly", "not", "n/a"]
    sack = np.array([0, 0, 1, np.nan, 0], dtype=float)
    hit = np.array([0, 1, 1, 0, np.nan], dtype=float)
    assert imp.pass_qb_outcome(sack, hit).tolist() == ["clean", "qb_hit", "sack", "na", "na"]
    comp = np.array([1, 0, 0, 0, 0], dtype=float)
    intc = np.array([0, 0, 0, 1, 0], dtype=float)
    assert imp.pass_ball_outcome(np.array([0, 0, 1, 0, 0.0]), comp, intc).tolist() == ["complete", "incomplete", "sack", "interception", "incomplete"]
    assert imp.run_yards_bucket(np.array([-3, 0, 2, 3, 5, 6, 10, 11, np.nan])).tolist() == ["<0", "0-2", "0-2", "3-5", "3-5", "6-10", "6-10", "10+", "na"]
    pt = np.array(["pass", "pass", "run", "qb_kneel", "pass"], dtype=object)
    ob = imp.outcome_bucket(pt, np.array([0, 1, 0, 0, 0.0]), np.array([0, 1, 0, 0, 1.0]), np.array([1, 0, 0, 0, 0.0]),
                            np.array([0, 0, 0, 0, 0.0]), np.array([7, -8, 4, -1, 0.0]))
    assert ob.tolist() == ["pass:clean:complete", "pass:sack:sack", "run:3-5", "qb_kneel", "pass:qb_hit:incomplete"]
    # skill relative to a reference prediction: equal predictions -> 0, perfect -> 1
    yv = np.array([1.0, 2.0, 4.0])
    assert abs(imp._skill_vs("reg", yv, yv, np.full(3, 2.0)) - 1.0) < 1e-12
    assert abs(imp._skill_vs("reg", yv, np.full(3, 2.0), np.full(3, 2.0))) < 1e-12
    assert set(imp.lgb_params("binary", imp.ImputationConfig())) >= {"objective", "num_threads", "seed", "deterministic"}
