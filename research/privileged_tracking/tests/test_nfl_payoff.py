"""Tests for the NFL 04 payoff stage's pure helpers (synthetic inputs, no data access)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.nfl import imputation_features as imf
from research.privileged_tracking.nfl import payoff as po


def test_feature_sets_nest_and_never_contain_outcomes() -> None:
    for subset in ("pass", "run"):
        base = po.payoff_feature_columns("PBP", subset)
        for s in po.SETS_A:
            cols = po.payoff_feature_columns(s, subset)
            po.leak_check(cols)                      # raises on any outcome field
            assert cols[:len(base)] == base          # PBP is a prefix of every set
            assert len(cols) == len(set(cols))       # no duplicates
        imp = po.payoff_feature_columns("PBP+IMP", subset)
        assert all(c.startswith("imp_") for c in imp[len(base):])
        assert f"imp_box_count_tuned__F0P" in imp
        within = po.WITHIN_TARGETS[subset]
        assert all(f"imp_{t}__F1T" in imp for t in within)
        assert not any(c.endswith("__F1") for c in imp)          # F1 students only in the leaky diagnostic
        leaky = po.payoff_feature_columns("PBP+IMP_F1(leaky)", subset)
        assert any(c.endswith("__F1") for c in leaky)
        oracle = po.payoff_feature_columns("PBP+ORACLE", subset)
        assert all(c.startswith("y_") for c in oracle[len(base):])
        pre = po.payoff_feature_columns("PBP+ORACLE_PRESNAP", subset)[len(base):]
        wit = po.payoff_feature_columns("PBP+ORACLE_WITHIN", subset)[len(base):]
        assert set(pre) | set(wit) == set(oracle[len(base):]) and not (set(pre) & set(wit))
        assert all(c[2:] in po.PRESNAP_TARGETS for c in pre) and len(pre) == 12
        assert all(c[2:] in po.WITHIN_TARGETS[subset] for c in wit)
        ngs = po.payoff_feature_columns("PBP+NGS", subset)
        assert "ngs_defenders_in_box" in ngs and "pers_WR" in ngs
    # the pass PBP set carries at-release fields but no completion / yards / EPA
    pas = po.payoff_feature_columns("PBP", "pass")
    assert {"air_yards", "pass_location", "qb_hit", "receiver_group"} <= set(pas)
    assert "run_gap" not in pas and "sack" not in pas
    assert "run_gap" in po.payoff_feature_columns("PBP", "run")


def test_leak_check_raises() -> None:
    with pytest.raises(ValueError):
        po.leak_check(["down", "complete_pass"])
    with pytest.raises(ValueError):
        po.leak_check(["epa"])
    po.leak_check(["down", "air_yards", "imp_separation_at_arrival__F1T"])


def test_student_sets_exclude_outcome_inputs() -> None:
    for name, feats in po.STUDENT_SETS.items():
        assert not ({"complete_pass", "interception", "yards_after_catch", "yards_gained", "epa"} & set(feats)), name
        assert set(feats) <= set(imf.DESIGN_COLS)
        assert "air_yards" in feats and "qb_hit" in feats and "run_gap" in feats
    assert set(po.STUDENT_SETS["F0T"]) < set(po.STUDENT_SETS["F1T"])   # nested: F0T lacks personnel


def test_task_mask() -> None:
    pt = np.array(["pass", "pass", "pass", "run", "qb_spike", "no_play", "pass"], dtype=object)
    sack = np.array([0, 1, 0, 0, 0, np.nan, 0.0])
    spike = np.array([0, 0, 1, 0, 1, np.nan, 0.0])
    is_pass = np.array([True, True, True, False, True, False, False])
    np.testing.assert_array_equal(po.task_mask(pt, sack, spike, "pass"), [True, False, False, False, False, False, True])
    np.testing.assert_array_equal(po.task_mask(pt, sack, spike, "pass", is_pass_play=is_pass), [True, False, False, False, False, False, False])
    np.testing.assert_array_equal(po.task_mask(pt, sack, spike, "run"), [False, False, False, True, False, False, False])
    np.testing.assert_array_equal(po.task_mask(pt, sack, spike, "run", is_pass_play=is_pass), [False, False, False, True, False, False, False])
    with pytest.raises(ValueError):
        po.task_mask(pt, sack, spike, "kick")


def test_feature_group() -> None:
    assert po.feature_group("imp_box_count_tuned__F0P") == "imputed pre-snap"
    assert po.feature_group("imp_separation_at_arrival__F1T") == "imputed within-play"
    assert po.feature_group("y_mof_open") == "oracle pre-snap"
    assert po.feature_group("y_separation_at_arrival") == "oracle within-play"
    assert po.feature_group("y_yards_to_first_contact") == "oracle within-play"
    assert po.feature_group("ngs_was_pressure") == "ngs official"
    assert po.feature_group("pers_WR") == "personnel" and po.feature_group("formation") == "personnel"
    assert po.feature_group("p_off_1RB_1TE_3WR") == "imputed personnel"
    assert po.feature_group("true_off_1RB_1TE_3WR") == "true personnel"
    assert po.feature_group("down") == "pbp" and po.feature_group("tend_pos_pass") == "pbp"


def test_per_play_losses_and_tables() -> None:
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.6).astype(float)
    good = np.clip(0.6 + 0.3 * (y - 0.5) + rng.normal(scale=0.05, size=n), 0.01, 0.99)
    bad = np.full(n, 0.6)
    l = po.per_play_losses("binary", y, good)
    assert set(l) == {"log_loss"} and l["log_loss"].shape == (n,) and np.all(l["log_loss"] >= 0)
    lr = po.per_play_losses("reg", y, bad)
    assert set(lr) == {"se", "ae"} and np.allclose(lr["se"], (y - 0.6) ** 2)
    preds = {"PBP": bad, "PBP+IMP": good}
    m = po.metrics_table("binary", y, preds, {"task": "t"})
    assert list(m["model"]) == ["PBP", "PBP+IMP"] and (m["n"] == n).all()
    assert m.set_index("model").loc["PBP+IMP", "log_loss"] < m.set_index("model").loc["PBP", "log_loss"]
    cfg = po.PayoffConfig(n_boot=200)
    groups = np.repeat(np.arange(20), n // 20)
    d = po.deltas_table("binary", y, preds, groups, "PBP", cfg, {"task": "t"})
    assert len(d) == 1 and d.iloc[0]["from"] == "PBP" and d.iloc[0]["to"] == "PBP+IMP"
    assert d.iloc[0]["delta"] > 0 and d.iloc[0]["ci_low"] > 0 and d.iloc[0]["ci_low_game"] > 0
    assert d.iloc[0]["n"] == n and d.iloc[0]["n_games"] == 20
    # NaN predictions (e.g. nflfastR cp on throwaways) restrict the paired rows
    partial = good.copy()
    partial[:50] = np.nan
    d2 = po.deltas_table("binary", y, {"PBP": bad, "cp": partial}, groups, "PBP", cfg, {}, pairs=[("PBP", "cp")])
    assert d2.iloc[0]["n"] == n - 50
    c = po.calibration_rows(y, {"PBP+IMP": good}, {"task": "t"}, n_bins=5)
    assert len(c) == 5 and c["n"].sum() == n and c["bin"].tolist() == [0, 1, 2, 3, 4]


def test_receiver_week_and_correlations() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for season in (2018, 2019):
        for week in range(1, 6):
            for team, recs in (("A", ["r1", "r2"]), ("B", ["r3"])):
                for r in recs:
                    k = 4 if r == "r1" else 2
                    for _ in range(k):
                        rows.append({"season": season, "week": week, "posteam": team, "receiver_player_id": r,
                                     "imp_separation_at_arrival__F1": rng.normal(3, 1), "imp_cb_cushion__F0P": rng.normal(6, 1),
                                     "air_yards": rng.normal(8, 4), "cp": rng.random()})
    rows.append({"season": 2018, "week": 1, "posteam": "A", "receiver_player_id": None, "imp_separation_at_arrival__F1": 1.0,
                 "imp_cb_cushion__F0P": 1.0, "air_yards": 1.0, "cp": 0.5})
    plays = pd.DataFrame(rows)
    rw = po.receiver_week_table(plays)
    assert len(rw) == 2 * 5 * 3
    a = rw[(rw["season"] == 2018) & (rw["week"] == 1)].set_index("receiver_player_id")
    assert a.loc["r1", "n_targets"] == 4 and a.loc["r2", "n_targets"] == 2 and a.loc["r3", "n_targets"] == 2
    assert a.loc["r1", "team_targets"] == 6 and a.loc["r3", "team_targets"] == 2
    assert abs(a.loc["r1", "target_share"] - 4 / 6) < 1e-12 and abs(a.loc["r3", "target_share"] - 1.0) < 1e-12
    assert "mean_imp_separation_at_arrival__F1" in rw.columns and "mean_air_yards" in rw.columns
    # a perfectly correlated NGS field recovers pearson 1 / spearman 1, an unrelated one ~0
    ngs = rw[["season", "week", "receiver_player_id", "n_targets"]].copy()
    ngs["avg_separation"] = rw["mean_imp_separation_at_arrival__F1"] * 2 + 1
    ngs["avg_cushion"] = rng.normal(size=len(rw))
    merged = rw.merge(ngs.drop(columns="n_targets"), on=["season", "week", "receiver_player_id"])
    corr = po.correlation_table(merged)
    sep = corr[(corr["ngs_field"] == "avg_separation") & (corr["proxy"] == "mean_imp_separation_at_arrival__F1")].iloc[0]
    assert abs(sep["pearson"] - 1) < 1e-9 and abs(sep["spearman"] - 1) < 1e-9 and abs(sep["within_player_pearson"] - 1) < 1e-9
    assert sep["n"] == len(merged) and sep["n_within"] == len(merged)
    assert set(corr["proxy"]) >= {"target_share", "mean_air_yards", "mean_cp"}
    strict = po.correlation_table(merged, min_targets=3)
    assert strict["n"].max() == 10   # only r1 (4 targets a week) survives


def test_personnel_one_hot_and_shift_table() -> None:
    s = pd.Series(["1 RB, 1 TE, 3 WR", "other", None, "1 RB, 1 TE, 3 WR"])
    oh = po.personnel_one_hot(s, ["1 RB, 1 TE, 3 WR", "other"])
    assert list(oh.columns) == ["true_off_1RB_1TE_3WR", "true_off_other"]
    assert oh["true_off_1RB_1TE_3WR"].tolist()[:2] == [1.0, 0.0] and np.isnan(oh["true_off_other"].iloc[2])
    frames = pd.DataFrame({"season": [2018] * 4 + [2019] * 4, "imp_qb_depth__F0P": [5, 6, 7, 8, 1, 2, 3, 4.0]})
    oof = pd.DataFrame({"imp_qb_depth__F0P": [4.0, 6.0], "y_qb_depth": [3.0, 7.0]})
    sh = po.shift_table(frames, oof)
    assert len(sh) == 2 and sh["target"].unique().tolist() == ["qb_depth"]
    r = sh.set_index("season")
    assert r.loc[2018, "mean"] == 6.5 and r.loc[2019, "mean"] == 2.5 and r.loc[2018, "oof2017_mean"] == 5.0
    assert abs(r.loc[2018, "z_shift"] - 1.5) < 1e-12 and r.loc[2018, "truth2017_mean"] == 5.0


def test_group_gain_shares() -> None:
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"down": rng.integers(1, 5, 300).astype(float), "imp_box_count_tuned__F0P": rng.normal(size=300),
                      "ngs_defenders_in_box": rng.normal(size=300)})
    y = 3 * X["imp_box_count_tuned__F0P"] + rng.normal(scale=0.1, size=300)
    b = lgb.train({"objective": "regression", "verbose": -1, "num_leaves": 4, "num_threads": 1, "seed": 0}, lgb.Dataset(X, label=y), 30)
    g = po.group_gain_shares([b, b], list(X.columns))
    assert abs(sum(g.values()) - 1) < 1e-9 and g["imputed pre-snap"] > 0.9


def test_proxy_regression_table() -> None:
    rng = np.random.default_rng(3)
    n = 600
    season = np.repeat([2018, 2019, 2020, 2021, 2022], n // 5)
    imp = rng.normal(size=n)
    d = pd.DataFrame({"season": season, "mean_air_yards": rng.normal(size=n), "mean_cp": rng.normal(size=n),
                      "target_share": rng.random(n), "n_targets": rng.integers(1, 12, n).astype(float),
                      "mean_imp_separation_at_arrival__F1": imp, "mean_imp_cb_cushion__F0P": rng.normal(size=n)})
    d["avg_separation"] = 2 * imp + 0.05 * rng.normal(size=n)
    models = {"naive": po.NAIVE_PROXIES, "naive + imputed F1": po.NAIVE_PROXIES + ["mean_imp_separation_at_arrival__F1"]}
    t = po.proxy_regression_table(d, (2018, 2019, 2020, 2021), 2022, ngs_cols=("avg_separation",), models=models)
    assert len(t) == 2 and (t["n_test"] == n // 5).all() and (t["n_train"] == 4 * n // 5).all()
    r = t.set_index("proxy_model")
    assert r.loc["naive + imputed F1", "test_r2"] > 0.95 and abs(r.loc["naive", "test_r2"]) < 0.1
    assert r.loc["naive + imputed F1", "test_pearson"] > 0.97
    assert r["ci_low"].isna().all()                                  # the default baseline is not among the models
    # paired bootstrap CI of the R2 gain over an explicit baseline: equals the R2 difference, CI excludes zero
    t2 = po.proxy_regression_table(d, (2018, 2019, 2020, 2021), 2022, ngs_cols=("avg_separation",), models=models, baseline="naive",
                                   n_boot=300)
    r2_ = t2.set_index("proxy_model")
    gain = r2_.loc["naive + imputed F1", "delta_r2_vs_baseline"]
    assert abs(gain - (r2_.loc["naive + imputed F1", "test_r2"] - r2_.loc["naive", "test_r2"])) < 1e-9
    assert 0 < r2_.loc["naive + imputed F1", "ci_low"] <= gain <= r2_.loc["naive + imputed F1", "ci_high"]
    assert np.isnan(r2_.loc["naive", "delta_r2_vs_baseline"]) and (t2["baseline"] == "naive").all()
    # a NaN in one model's proxy drops that receiver-week for every model (paired rows)
    d2 = d.copy()
    d2.loc[d2.index[-1], "mean_imp_separation_at_arrival__F1"] = np.nan
    t3 = po.proxy_regression_table(d2, (2018, 2019, 2020, 2021), 2022, ngs_cols=("avg_separation",), models=models, baseline="naive")
    assert (t3["n_test"] == n // 5 - 1).all()


def test_metrics_table_on_restricts_rows_and_suffixes_names() -> None:
    rng = np.random.default_rng(5)
    n = 300
    y = (rng.random(n) < 0.6).astype(float)
    p = np.clip(0.6 + 0.3 * (y - 0.5) + rng.normal(scale=0.05, size=n), 0.01, 0.99)
    cp = p.copy()
    cp[:40] = np.nan                                    # throwaways: cp absent
    preds = {"PBP": p, "PBP+IMP": p, "nflfastR cp": cp, "base_global": np.full(n, 0.6)}
    m = po.metrics_table_on("binary", y, preds, ~np.isnan(cp), {"task": "t"}, po.CP_ROWS_SUFFIX, ["PBP", "PBP+IMP"])
    assert list(m["model"]) == [f"PBP{po.CP_ROWS_SUFFIX}", f"PBP+IMP{po.CP_ROWS_SUFFIX}"]
    assert (m["n"] == n - 40).all()
    full = po.metrics_table("binary", y, {"nflfastR cp": cp}, {})
    assert abs(m.iloc[0]["log_loss"] - full.iloc[0]["log_loss"]) < 1e-12   # same rows -> same number


def test_refit_spread_and_attach() -> None:
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    p0 = np.array([0.7, 0.3, 0.6, 0.8, 0.4])
    p1 = np.array([0.6, 0.3, 0.6, 0.8, 0.4])
    rows = po.refit_mean_losses("t", "binary", y, {0: {"PBP": p0}, 1: {"PBP": p1}, 2: {"PBP": p0}})
    assert len(rows) == 3 and {r["seed"] for r in rows} == {0, 1, 2} and all(r["loss"] == "log_loss" for r in rows)
    sp = po.refit_spread_table(pd.DataFrame(rows))
    assert len(sp) == 1 and sp.iloc[0]["n_seeds"] == 3
    expected = abs(np.mean(po.per_play_losses("binary", y, p1)["log_loss"]) - np.mean(po.per_play_losses("binary", y, p0)["log_loss"]))
    assert abs(sp.iloc[0]["spread"] - expected) < 1e-12
    reg_rows = po.refit_mean_losses("u", "reg", y, {0: {"PBP": p0, "PBP+IMP": p1}, 1: {"PBP": p1, "PBP+IMP": p1}})
    sp2 = po.refit_spread_table(pd.DataFrame(reg_rows))
    assert set(sp2["loss"]) == {"se", "ae"} and len(sp2) == 4
    assert sp2[sp2["model"] == "PBP+IMP"]["spread"].abs().max() < 1e-12        # identical refits -> zero spread
    deltas = pd.DataFrame({"task": ["t", "u", "u"], "from": ["PBP"] * 3, "to": ["PBP+IMP"] * 3, "loss": ["log_loss", "se", "ae"],
                           "delta": [0.01, 0.1, 0.05], "ci_low_game": [0.001, -0.1, 0.01], "ci_high_game": [0.02, 0.2, 0.1]})
    d = po.attach_refit_spread(deltas, pd.concat([sp, sp2], ignore_index=True))
    assert "refit_spread" in d.columns and "refit_measured" in d.columns and len(d) == 3
    di = d.set_index(["task", "loss"])
    # task t: only PBP was refit -> the floor is PBP's spread, flagged as partially measured
    assert abs(di.loc[("t", "log_loss"), "refit_spread"] - expected) < 1e-12 and di.loc[("t", "log_loss"), "refit_measured"] == "from"
    # task u: both members refit -> the larger of the two spreads
    assert di.loc[("u", "se"), "refit_spread"] == sp2[sp2["loss"] == "se"]["spread"].max() and di.loc[("u", "se"), "refit_measured"] == "both"
    empty = po.attach_refit_spread(deltas, po.refit_spread_table(pd.DataFrame(columns=["task", "model", "seed", "loss", "mean_loss"])))
    assert empty["refit_spread"].isna().all() and (empty["refit_measured"] == "none").all()
    # a pair against a fixed reference column is fully measured by the other member; an unmeasured pair is 'none'
    more = pd.DataFrame({"task": ["t", "t", "t", "t"], "from": ["nflfastR cp", "PBP+IMP", "PBP+NGS", "PBP"],
                         "to": ["PBP", "PBP+ORACLE", "PBP", "PBP+IMP_F0"], "loss": ["log_loss"] * 4, "delta": [0.0] * 4,
                         "ci_low_game": [0.0] * 4, "ci_high_game": [0.0] * 4})
    m = po.attach_refit_spread(more, sp)                                    # sp holds task t / PBP only
    assert m["refit_measured"].tolist() == ["both", "none", "to", "from"]
    assert abs(m["refit_spread"].iloc[0] - expected) < 1e-12 and np.isnan(m["refit_spread"].iloc[1])
    assert abs(m["refit_spread"].iloc[2] - expected) < 1e-12 and abs(m["refit_spread"].iloc[3] - expected) < 1e-12


def test_sig_uses_refit_spread() -> None:
    base = {"delta": 0.0013, "ci_low_game": 0.0001, "ci_high_game": 0.0026}
    assert po._sig(pd.Series({**base, "refit_spread": 0.002})).startswith("CI excludes 0 (better) but")
    assert po._sig(pd.Series({**base, "refit_spread": 0.0005})).startswith("better (clustered CI excludes 0 and")
    assert po._sig(pd.Series({**base, "refit_spread": np.nan})).startswith("better (clustered CI excludes 0; refit noise not measured")
    assert po._sig(pd.Series(base)).startswith("better")
    # only one member refit: the floor is a lower bound and the verdict names the member
    part = pd.Series({**base, "refit_spread": 0.002, "refit_measured": "from", "from": "PBP", "to": "PBP+IMP_F0"})
    assert "of `PBP` alone" in po._sig(part) and po._sig(part).endswith("within refit noise")
    part2 = pd.Series({**base, "refit_spread": 0.0005, "refit_measured": "to", "from": "PBP+NGS", "to": "PBP+IMP"})
    assert po._sig(part2).startswith("better (clustered CI excludes 0 and |delta| > refit spread 0.0005 of `PBP+IMP` alone")
    assert po._sig(pd.Series({**base, "refit_spread": np.nan, "refit_measured": "none"})).endswith("not measured for this pair)")
    assert po._sig(pd.Series({**base, "refit_spread": 0.002, "refit_measured": "both"})).endswith("of the pair: within refit noise")
    assert po._sig(pd.Series({"delta": -0.001, "ci_low_game": -0.003, "ci_high_game": -0.0005, "refit_spread": 0.002})).startswith("CI excludes 0 (worse)")
    assert po._sig(pd.Series({"delta": 0.001, "ci_low_game": -0.001, "ci_high_game": 0.003, "refit_spread": 0.0})) == "no distinguishable change"


def test_add_after_the_fact_and_proxy_columns() -> None:
    plays = pd.DataFrame({"complete_pass": [1.0, 0.0, 1.0, np.nan, 0.0], "yards_after_catch": [5.0, np.nan, 0.0, np.nan, np.nan],
                          "yards_gained": [12.0, 0.0, 3.0, 0.0, 0.0], "interception": [0, 0, 0, 0, 1], "qb_hit": [0, 1, 0, 0, 0],
                          "epa": [1.2, -0.5, 0.1, 0.0, -2.0]})
    out = po.add_after_the_fact(plays)
    assert "yac_per_target" not in plays.columns                    # pure
    assert out["yac_per_target"].tolist()[:3] == [5.0, 0.0, 0.0] and np.isnan(out["yac_per_target"].iloc[3]) and out["yac_per_target"].iloc[4] == 0.0
    assert all(pd.api.types.is_float_dtype(out[c]) for c in po.AFTER_FACT_COLS)
    assert set(po.AFTER_FACT_COLS) <= set(po.PROXY_COLS) and set(po.AFTER_PROXIES) == {f"mean_{c}" for c in po.AFTER_FACT_COLS}
    # the fair baseline and every "+ imputed" row on top of it exist and nest
    fair = po.PROXY_MODELS[po.FAIR_BASELINE]
    assert set(po.NAIVE_PROXIES) < set(fair) and set(po.AFTER_PROXIES) < set(fair)
    for f in ("F0T", "F1T", "F1"):
        assert set(fair) < set(po.PROXY_MODELS[f"{po.FAIR_BASELINE} + imputed {f}"])
    # after-the-fact fields never reach an outcome model's feature list
    with pytest.raises(ValueError):
        po.leak_check(po.payoff_feature_columns("PBP", "pass") + ["yards_gained"])


def test_imputation_vs_outcome_table() -> None:
    rng = np.random.default_rng(7)
    n = 1000
    season = np.repeat([2018, 2019, 2020, 2021, 2022], n // 5)
    cp_flag = (rng.random(n) < 0.65).astype(float)
    air = rng.normal(8, 5, n)
    d = pd.DataFrame({"season": season, "complete_pass": cp_flag, "yards_after_catch": np.where(cp_flag == 1, rng.exponential(5, n), np.nan),
                      "air_yards": air, "interception": (rng.random(n) < 0.03).astype(float), "qb_hit": (rng.random(n) < 0.1).astype(float),
                      "cp": rng.random(n), "yards_gained": 0.0, "epa": 0.0})
    d = po.add_after_the_fact(d)
    d["imp_a"] = 2.0 * d["complete_pass"] - 0.1 * d["air_yards"] + 0.05 * rng.normal(size=n)     # outcome-driven imputation
    d["imp_b"] = rng.normal(size=n)                                                                # unrelated
    t = po.imputation_vs_outcome_table(d, ["imp_a", "imp_b", "imp_missing"], (2018, 2019, 2020, 2021), 2022)
    assert set(t["imputed_col"]) == {"imp_a", "imp_b"} and len(t) == 4
    q = t.set_index(["imputed_col", "regressors"])
    out_key = [k for k in po.OUTCOME_REGRESSORS if k.startswith("outcome")][0]
    rel_key = [k for k in po.OUTCOME_REGRESSORS if k.startswith("at-release")][0]
    assert q.loc[("imp_a", out_key), "r2_test"] > 0.95 and q.loc[("imp_a", rel_key), "r2_test"] < 0.5
    assert abs(q.loc[("imp_b", out_key), "r2_test"]) < 0.1
    assert (t["n_test"] == n // 5).all() and (t["n_train"] == 4 * n // 5).all()
