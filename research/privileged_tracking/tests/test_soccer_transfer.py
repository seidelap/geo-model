"""Tests of the pure helpers of the soccer 04 transfer stage on synthetic inputs (no data
access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.soccer import transfer as tr


def test_domain_and_population_labels():
    comp = np.array(
        ["Premier League", "La Liga", "FIFA World Cup", "Major League Soccer", "Ligue 1"]
    )
    season = np.array(["2015/2016", "2015/2016", "2018", "2023", "2022/2023"])
    dom = tr.domain_of(comp, season)
    assert list(dom) == ["PL 2015/16", "La Liga 2015/16", "WC 2018", "other", "other"]
    pop = tr.population_of(dom)
    assert list(pop) == [tr.POP_LEAGUES, tr.POP_LEAGUES, tr.POP_TOURNAMENTS, "other", "other"]


def test_fold_vector_keeps_matches_whole_and_balanced():
    match = np.repeat(np.arange(20), 7)
    fold = tr.fold_vector(match, n_splits=5, seed=1)
    assert fold.min() == 0 and fold.max() == 4
    per_match = pd.Series(fold).groupby(match).nunique()
    assert (per_match == 1).all()
    counts = np.bincount(fold) // 7
    assert counts.tolist() == [4, 4, 4, 4, 4]
    # deterministic in the seed
    assert np.array_equal(fold, tr.fold_vector(match, n_splits=5, seed=1))
    assert not np.array_equal(fold, tr.fold_vector(match, n_splits=5, seed=2))


def test_complete_design_restores_dropped_columns_in_model_order():
    design = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0]})
    full = pd.DataFrame({"a": [9.0, 9.0], "b": [9.0, 9.0], "f_gender": [0.0, 0.0]})
    out = tr.complete_design(design, full, ["a", "f_gender", "b", "missing"])
    assert list(out.columns) == ["a", "f_gender", "b", "missing"]
    assert out["a"].tolist() == [3.0, 4.0]  # existing columns are kept, not overwritten
    assert out["f_gender"].tolist() == [0.0, 0.0]
    assert out["missing"].isna().all()


def _oracle_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs = rng.integers(0, 4, n).astype(float)
    imp_v = obs + rng.normal(0, 0.3, n)
    imp_v[:5] = -0.5  # negative imputations are clipped at 0 for counts
    dist = rng.uniform(1, 8, n)
    return pd.DataFrame(
        {
            "imp_n_opp_in_cone__E2": imp_v.astype(np.float32),
            "sff_n_opp_in_cone": obs,
            "imp_nearest_opp_dist_in_cone__E2": (dist + rng.normal(0, 1, n)).astype(np.float32),
            "sff_nearest_opp_dist_in_cone": np.where(obs > 0, dist, np.nan),
        }
    )


def test_oracle_check_table_metrics_and_group_filter():
    df = _oracle_frame()
    groups = np.array(["A"] * 300 + ["B"] * 100, dtype=object)
    pairs = tr.ORACLE_PAIRS[:2]
    t = tr.oracle_check_table(df, groups, pairs)
    assert set(t["group"]) == {"A", "B"}
    a = t[(t["group"] == "A") & (t["target"] == "n_opp_in_cone")].iloc[0]
    assert a["n"] == 300 and a["kind"] == "count"
    assert a["r2"] > 0.8 and a["mae"] < 0.4
    assert 0.9 <= a["within_1"] <= 1.0 and a["exact"] > 0.5
    # the clipped rows do not go negative
    assert a["mean_imp"] >= 0
    d = t[(t["group"] == "A") & (t["target"] == "nearest_opp_dist_in_cone")].iloc[0]
    # NaN observed rows (empty cone) are excluded from n
    assert d["n"] == int((df["sff_nearest_opp_dist_in_cone"].notna().to_numpy()[:300]).sum())
    assert "within_1" not in d or pd.isna(d["within_1"])


def test_oracle_check_table_skips_small_groups_and_missing_columns():
    df = _oracle_frame(60)
    groups = np.array(["A"] * 50 + ["tiny"] * 10, dtype=object)
    t = tr.oracle_check_table(df, groups, tr.ORACLE_PAIRS)
    assert "tiny" not in set(t["group"])
    # keeper pair is absent from the frame -> silently skipped
    assert set(t["target"]) <= {"n_opp_in_cone", "nearest_opp_dist_in_cone"}


def test_oracle_calibration_bins_are_monotone_in_imputed_value():
    df = _oracle_frame(500)
    groups = np.full(500, "G", dtype=object)
    c = tr.oracle_calibration(df, groups, tr.ORACLE_PAIRS[:1], n_bins=5)
    assert len(c) == 5 and c["n"].sum() == 500
    assert np.all(np.diff(c["imputed"].to_numpy()) >= 0)
    assert c["observed"].iloc[-1] > c["observed"].iloc[0]


def _team_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    rows = []
    for mid in (1, 2, 3):
        for team, opp in ((10, 20), (20, 10)):
            depth = 40.0 if opp == 20 else 55.0  # team 20 defends deep, team 10 high
            for _ in range(30):
                rows.append(
                    {
                        "match_id": mid,
                        "team_id": team,
                        "opp_team_id": opp,
                        "f_x": rng.uniform(20, 100),
                        "imp_block_depth__E2": depth + rng.normal(0, 2),
                        "imp_def_line__E2": depth - 10 + rng.normal(0, 2),
                        "imp_block_depth__E0": depth + rng.normal(0, 4),
                        "imp_def_line__E0": depth - 10 + rng.normal(0, 4),
                        "competition": "PL",
                        "season": "2015/2016",
                    }
                )
    rows = pd.DataFrame(rows)
    tm = pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3, 3],
            "team_id": [10, 20, 10, 20, 10, 20],
            "team": ["High", "Deep"] * 3,
            "def_line_x": [52.0, 38.0, 54.0, 40.0, 50.0, 41.0],
            "ppda": [8.0, 15.0, 7.5, 14.0, 9.0, 16.0],
            "ppda_passes_allowed": [200.0, 400.0, 210.0, 380.0, 220.0, 390.0],
            "possession": [0.6, 0.4, 0.58, 0.42, 0.61, 0.39],
        }
    )
    return rows, tm


def test_team_block_table_attributes_shape_to_the_defending_team():
    rows, tm = _team_rows()
    block = tr.team_block_table(rows, tm)
    assert len(block) == 6
    assert set(block.columns) >= {
        "match_id",
        "team_id",
        "n_rows",
        "ball_x_mean",
        "imp_block_depth__E2",
        "def_line_x",
        "ppda",
    }
    deep = block[block["team_id"] == 20]
    high = block[block["team_id"] == 10]
    assert (deep["n_rows"] == 30).all()
    assert deep["imp_block_depth__E2"].mean() < high["imp_block_depth__E2"].mean()
    block = block.merge(
        rows[["match_id", "competition", "season"]].drop_duplicates("match_id"), on="match_id"
    )
    season = tr.team_season_table(block, min_matches=2)
    assert len(season) == 2 and (season["n_matches"] == 3).all()
    sp = tr.spearman_table(block, ["imp_block_depth__E2"], ["def_line_x", "ppda"], "team-match")
    assert len(sp) == 2
    assert sp.set_index("proxy").loc["def_line_x", "spearman"] > 0.7
    assert sp.set_index("proxy").loc["ppda", "spearman"] < -0.7


def _events() -> list[dict]:
    ev = []
    for minute, team, typ, x in (
        (0, 7, "Pressure", 60.0),
        (0, 7, "Pass", 10.0),  # not a defensive action
        (0, 7, "Interception", 40.0),
        (1, 7, "Clearance", 20.0),
        (1, 8, "Pressure", 80.0),
        (2, 8, "Duel", None),  # no location -> ignored
    ):
        e = {"minute": minute, "team": {"id": team}, "type": {"name": typ}}
        if x is not None:
            e["location"] = [x, 40.0]
        ev.append(e)
    return ev


def test_def_action_x_by_minute():
    d = tr.def_action_x_by_minute(_events(), 7)
    assert d["minute"].tolist() == [0, 1]
    assert d["def_x_mean"].tolist() == pytest.approx([50.0, 20.0])
    assert d["n_def"].tolist() == [2, 1]
    assert tr.def_action_x_by_minute(_events(), 99).empty


def test_minute_profile_and_bins():
    rows = pd.DataFrame(
        {
            "opp_team_id": [7, 7, 7, 8, 8],
            "f_is_possession_team": [True, True, False, True, True],
            "f_minute": [0.0, 0.0, 0.0, 1.0, 6.0],
            "imp_block_depth__E2": [30.0, 34.0, 99.0, 50.0, np.nan],
            "imp_def_line__E2": [20.0, 24.0, 99.0, 40.0, np.nan],
        }
    )
    prof = tr.minute_profile(rows, _events(), {7: "Seven", 8: "Eight"})
    seven = prof[prof["team_id"] == 7].set_index("minute")
    # defending-team rows (possession False) and NaN imputations are ignored
    assert seven.loc[0, "imp_block_depth"] == pytest.approx(32.0)
    assert seven.loc[0, "n_rows"] == 2
    assert seven.loc[0, "def_x_mean"] == pytest.approx(50.0)
    assert np.isnan(seven.loc[1, "imp_block_depth"]) and seven.loc[1, "def_x_mean"] == 20.0
    eight = prof[prof["team_id"] == 8].set_index("minute")
    assert eight.loc[1, "imp_block_depth"] == pytest.approx(50.0)
    assert eight.loc[1, "def_x_mean"] == pytest.approx(80.0)
    b = tr.bin_profile(prof, width=5)
    s7 = b[b["team_id"] == 7].set_index("minutes")
    assert s7.loc["0-5", "n_rows"] == 2 and s7.loc["0-5", "n_def"] == 3
    # row-weighted: (2 * 50 + 1 * 20) / 3 for the defensive actions of team 7 in 0-5
    assert s7.loc["0-5", "def_x_mean"] == pytest.approx(40.0)
    corr = tr.profile_correlation(prof)
    assert set(corr["team_id"]) == {7, 8}
    assert "pearson" not in corr.columns or corr["pearson"].isna().all()  # < 5 minutes overlap


def test_choose_example_match_prefers_requested_domain():
    rows = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2, 3],
            "domain": [
                "PL 2015/16",
                "PL 2015/16",
                "PL 2015/16",
                "PL 2015/16",
                "PL 2015/16",
                "La Liga 2015/16",
            ],
        }
    )
    assert tr.choose_example_match(rows) == 1
    assert tr.choose_example_match(rows, "La Liga 2015/16") == 3
    assert tr.choose_example_match(rows, "Serie A 2015/16") == 1  # falls back to all rows


def test_oracle_summary_tables_from_a_check_table():
    df = _oracle_frame(600)
    groups = np.array(["360 all (OOF)"] * 300 + [tr.POP_LEAGUES] * 300, dtype=object)
    check = pd.concat(
        [
            tr.oracle_check_table(df, groups, tr.ORACLE_PAIRS[:1]),
            tr.oracle_check_table(
                df.iloc[:300].rename(columns={"sff_n_opp_in_cone": "y_n_opp_in_cone"}),
                groups[:300],
                tr.ORACLE_PAIRS_360[:1],
                observed="360 frame",
            ),
        ],
        ignore_index=True,
    )
    wide = tr._oracle_summary(check).set_index("target")
    assert {"r2_360_oof", "r2_leagues_1516", "bias_shift_1516", "r2_360_vs_360frame"} <= set(
        wide.columns
    )
    r = wide.loc["n_opp_in_cone"]
    assert r["bias_shift_1516"] == pytest.approx(r["bias_leagues_1516"] - r["bias_360_oof"])
    long = tr.oracle_summary_long(check)
    assert list(long["metric"]) == ["r2", "mae", "bias"]
    assert long.loc[long["metric"] == "r2", tr.POP_LEAGUES].iloc[0] == pytest.approx(
        r["r2_leagues_1516"]
    )
    assert long.loc[long["metric"] == "r2", "360 all (OOF) vs 360 frame"].notna().all()
    # groups are ordered target domains first, 360 references last
    md = tr._oracle_md(check, "n_opp_in_cone")
    assert md.index(tr.POP_LEAGUES) < md.index("360 all (OOF)")
