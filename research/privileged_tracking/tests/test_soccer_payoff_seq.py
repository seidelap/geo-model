"""Tests for the soccer 06 sequence-payoff pure helpers (synthetic frames, no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.soccer import payoff_seq as ps


def _table(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "match_id": [1, 1, 2, 2, 3, 3][:n],
            "fold": [0, 0, 1, 2, 3, 4][:n],
            "imp_n_opp_in_cone__E2": np.arange(n, dtype=float),
        }
    )


def test_restrict_to_folds_keeps_only_named_folds() -> None:
    t = _table()
    out = ps.restrict_to_folds(t, (0, 1, 2))
    assert out["fold"].tolist() == [0, 0, 1, 2]
    assert out.index.tolist() == [0, 1, 2, 3]


def test_attach_oof_adds_missing_columns_and_keeps_existing() -> None:
    t = _table()
    oof = pd.DataFrame(
        {
            "event_id": ["e0", "e2", "e5"],
            "block_depth__seq20": [10.0, 20.0, 50.0],
            "n_opp_in_cone__seq20": [1.0, 2.0, 5.0],
            "y_block_depth": [11.0, 21.0, 51.0],
            "n_opp_in_cone__E2": [99.0, 99.0, 99.0],
        }
    )
    out = ps.attach_oof(t, oof, ("block_depth", "n_opp_in_cone"), "seq20")
    assert len(out) == len(t)
    assert out["imp_block_depth__seq20"].tolist()[:3] == [10.0, np.nan, 20.0] or (
        out.loc[0, "imp_block_depth__seq20"] == 10.0
        and np.isnan(out.loc[1, "imp_block_depth__seq20"])
        and out.loc[2, "imp_block_depth__seq20"] == 20.0
    )
    assert out.loc[5, "y_block_depth"] == 51.0
    # an existing column is not overwritten unless asked
    out2 = ps.attach_oof(t, oof, ("n_opp_in_cone",), "E2")
    assert out2["imp_n_opp_in_cone__E2"].tolist() == list(np.arange(6, dtype=float))
    out3 = ps.attach_oof(t, oof, ("n_opp_in_cone",), "E2", overwrite=True)
    assert out3.loc[0, "imp_n_opp_in_cone__E2"] == 99.0
    with pytest.raises(ValueError):
        ps.attach_oof(t, pd.concat([oof, oof]), ("block_depth",), "seq20")


def test_compose_designs_nests_and_skips_missing_blocks() -> None:
    base = pd.DataFrame({"f_x": [1.0, 2.0]})
    blocks = {"a": pd.DataFrame({"s_a": [0.1, 0.2]}), "b": pd.DataFrame({"q_b": [5.0, 6.0]})}
    out = ps.compose_designs(
        base, blocks, {"E": (), "E+A": ("a",), "E+AB": ("a", "b"), "E+C": ("c",)}
    )
    assert set(out) == {"E", "E+A", "E+AB"}
    assert list(out["E"].columns) == ["f_x"]
    assert list(out["E+AB"].columns) == ["f_x", "s_a", "q_b"]
    assert out["E+AB"].shape == (2, 3)


def test_state_quality_table_scores_sources_on_identical_rows() -> None:
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(10, 3, n)
    t = pd.DataFrame(
        {
            "y_block_depth": y,
            "imp_block_depth__E2": y + rng.normal(0, 2, n),
            "imp_block_depth__seq20": y + rng.normal(0, 1, n),
            "y_deep_block": (rng.uniform(0, 1, n) < 0.3).astype(float),
        }
    )
    t.loc[:9, "imp_block_depth__seq20"] = np.nan
    t["imp_deep_block__E2"] = np.clip(t["y_deep_block"] * 0.6 + 0.2, 0, 1)
    t["imp_deep_block__seq20"] = np.clip(t["y_deep_block"] * 0.8 + 0.1, 0, 1)
    q = ps.state_quality_table(t, ("block_depth", "deep_block"), ("E2", "seq20"), "shots")
    bd = q[q["target"] == "block_depth"].set_index("source")
    assert bd.loc["E2", "n"] == bd.loc["seq20", "n"] == n - 10
    assert bd.loc["seq20", "r2"] > bd.loc["E2", "r2"] > 0.5
    db = q[q["target"] == "deep_block"].set_index("source")
    assert db.loc["seq20", "log_loss"] < db.loc["E2", "log_loss"]
    assert np.isnan(db.loc["seq20", "r2"])


def test_pairs_table_reports_named_pairs_only() -> None:
    rng = np.random.default_rng(1)
    n = 400
    y = (rng.uniform(0, 1, n) < 0.3).astype(float)
    good = np.clip(y * 0.5 + 0.2 + rng.normal(0, 0.05, n), 0.01, 0.99)
    bad = np.full(n, 0.3)
    preds = {"A": bad, "B": good, "C": good}
    match = rng.integers(0, 20, n)
    out = ps.pairs_table(y, preds, (("A", "B"), ("B", "C"), ("A", "Z")), match, n_boot=200, seed=0)
    assert out[["from", "to"]].values.tolist() == [["A", "B"], ["B", "C"]]
    ab = out.iloc[0]
    assert ab["delta_log_loss"] > 0 and ab["ci_low_clustered"] > 0
    assert abs(out.iloc[1]["delta_log_loss"]) < 1e-12
