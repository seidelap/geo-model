"""Tests for the soccer 02 student bundle / applier on a tiny synthetic frame (no data access)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.privileged_tracking.soccer import apply_student as aps
from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.tests.test_soccer_imputation_features import _frame


def _big_frame(n_rep: int = 40) -> pd.DataFrame:
    df = pd.concat([_frame(seed=s) for s in range(n_rep)], ignore_index=True)
    df["match_id"] = np.repeat(np.arange(n_rep), 12)
    df["event_id"] = [f"e{i}" for i in range(len(df))]
    df["season"] = np.where(df["competition"] == "La Liga", "2015/2016", "2022")
    return df


def _tiny_bundle(fset: str = "E0") -> tuple[aps.StudentBundle, pd.DataFrame]:
    df = _big_frame()
    xs = imf.build_design(df, fset)
    feats = imf.design_columns(fset)
    # a target that is a clean function of x: the student must learn a monotone map
    y = 0.5 * df["f_x"].fillna(60.0).to_numpy(dtype=float) + 1.0
    params = {
        "objective": "regression",
        "verbose": -1,
        "num_leaves": 7,
        "learning_rate": 0.2,
        "min_data_in_leaf": 5,
        "num_threads": 1,
        "seed": 0,
        "deterministic": True,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(xs[feats], label=y, categorical_feature=imf.categorical_columns(fset)),
        num_boost_round=80,
    )
    st = aps.Student(
        target="block_depth",
        kind="reg",
        subset="poss",
        model_str=booster.model_to_string(),
        n_rounds=80,
        train_n=len(df),
        y_mean=float(y.mean()),
        y_sd=float(y.std()),
        oof_mean=float(y.mean()),
        oof_sd=float(y.std()),
        cv_skill=0.9,
    )
    st_pass = aps.Student(
        **{**st.__dict__, "target": "n_opp_in_lane", "subset": "pass", "kind": "count"}
    )
    st_all = aps.Student(**{**st.__dict__, "target": "nearest_opp_dist", "subset": "all"})
    b = aps.StudentBundle(
        feature_set=fset,
        features=list(feats),
        categorical=imf.categorical_columns(fset),
        students={"block_depth": st, "n_opp_in_lane": st_pass, "nearest_opp_dist": st_all},
        deep_block_threshold=18.5,
    )
    return b, df


def test_bundle_round_trip_and_subsets(tmp_path) -> None:
    b, df = _tiny_bundle()
    p = b.save(root=tmp_path)
    assert p == tmp_path / "models" / "students_E0.joblib" and p.exists()
    b2 = aps.StudentBundle.load("E0", root=tmp_path)
    assert (
        b2.features == b.features
        and b2.deep_block_threshold == 18.5
        and set(b2.students) == set(b.students)
    )
    out = aps.impute(df.head(12), ("E0",), root=tmp_path)
    cols = [c for c in out.columns if c.startswith("imp_")]
    assert cols == ["imp_block_depth__E0", "imp_n_opp_in_lane__E0", "imp_nearest_opp_dist__E0"]
    # shape student on possession-team rows only, pass student on Pass rows, ball student everywhere
    assert (
        out["imp_block_depth__E0"].notna().tolist() == df.head(12)["f_is_possession_team"].tolist()
    )
    assert (
        out["imp_n_opp_in_lane__E0"].notna().tolist() == (df.head(12)["f_type"] == "Pass").tolist()
    )
    assert out["imp_nearest_opp_dist__E0"].notna().all()
    # the student is a function of the features: predictions increase with x
    pred = out["imp_nearest_opp_dist__E0"].to_numpy()
    x = df.head(12)["f_x"].to_numpy()
    ok = ~np.isnan(x)
    assert np.corrcoef(x[ok], pred[ok])[0, 1] > 0.9
    # unseen strings in the applied frame do not break the design (they take the unseen code)
    df2 = df.head(12).copy()
    df2["f_play_pattern"] = "Something New"
    out2 = aps.impute(df2, ("E0",), root=tmp_path)
    assert out2["imp_nearest_opp_dist__E0"].notna().all()


def test_shift_tables() -> None:
    b, df = _tiny_bundle()
    imputed = df.copy()
    imputed["imp_block_depth__E0"] = df["f_x"].to_numpy()
    oof = df.head(120).copy()
    oof["block_depth__E0"] = 1.0
    oof["y_block_depth"] = 2.0
    sh = aps.shift_table(imputed, oof, ("E0",))
    assert set(sh["source"]) == {
        "no360 leagues 2015/16 imputed",
        "no360 other competitions imputed",
        "360 oof imputed",
        "360 truth",
    }
    assert set(sh["event_type"]) == {"all", "Pass", "Carry", "Shot"}
    row = sh[(sh["source"] == "no360 leagues 2015/16 imputed") & (sh["event_type"] == "all")].iloc[
        0
    ]
    assert row["n"] == int(
        (aps.domain_label(df["competition"], df["season"]) == "leagues_2015/16").sum()
    )
    truth = sh[(sh["source"] == "360 truth") & (sh["event_type"] == "Pass")].iloc[0]
    assert truth["mean"] == 2.0 and truth["n"] == int((oof["f_type"] == "Pass").sum())
    # Pressure / Duel rows of the 360 side are excluded (only Pass / Carry / Shot are compared)
    allrow = sh[(sh["source"] == "360 oof imputed") & (sh["event_type"] == "all")].iloc[0]
    assert allrow["n"] == int(oof["f_type"].isin(aps.NO360_TYPES).sum())
    bc = aps.shift_by_competition(imputed, oof, "E0")
    assert {"no360 imputed", "360 oof imputed"} == set(
        bc["source"]
    ) and "block_depth_mean" in bc.columns
