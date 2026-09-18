"""Tests for the results-index helpers on synthetic report tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.privileged_tracking.synthesis.results_index import (
    INDEX_COLUMNS,
    Pick,
    PickError,
    all_picks,
    build_index,
    match_rows,
    resolve_pick,
)


def _tables() -> dict[str, pd.DataFrame]:
    metrics = pd.DataFrame(
        {
            "task": ["pass_completion", "pass_completion", "pass_yards"],
            "model": ["PBP", "PBP+IMP", "PBP"],
            "log_loss": [0.60, 0.61, np.nan],
            "r2": [np.nan, np.nan, 0.09],
            "n": [6225, 6225, 6225],
        }
    )
    deltas = pd.DataFrame(
        {
            "task": ["pass_completion", "pass_completion"],
            "from": ["PBP", "PBP"],
            "to": ["PBP+IMP", "PBP+IMP"],
            "loss": ["log_loss", "brier"],
            "delta": [-0.0009, -0.0003],
            "ci_low_game": [-0.0042, -0.0010],
            "ci_high_game": [0.0023, 0.0004],
            "n": [6225, 6225],
        }
    )
    return {"m": metrics, "d": deltas}


def test_match_rows_equality_and_nan() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"], "c": [np.nan, 1.0, np.nan]})
    assert len(match_rows(df, (("a", 1),))) == 2
    assert len(match_rows(df, (("a", 1), ("b", "x")))) == 1
    assert len(match_rows(df, (("c", float("nan")),))) == 2
    with pytest.raises(KeyError):
        match_rows(df, (("zzz", 1),))


def test_resolve_pick_reads_value_ci_and_n() -> None:
    t = _tables()
    p = Pick(
        "d",
        (("task", "pass_completion"), ("from", "PBP"), ("to", "PBP+IMP"), ("loss", "log_loss")),
        "delta",
        "delta_log_loss",
        "nfl",
        "04_payoff",
        "exp",
        "PBP -> PBP+IMP",
        "ci_low_game",
        "ci_high_game",
        "game-clustered",
        source_report="r.md",
    )
    row = resolve_pick(t, p)
    assert isinstance(row, dict)
    assert row["value"] == pytest.approx(-0.0009)
    assert row["ci_low"] == pytest.approx(-0.0042) and row["ci_high"] == pytest.approx(0.0023)
    assert row["n"] == 6225 and row["ci_type"] == "game-clustered"
    assert row["source_table"] == "d.parquet" and "loss=log_loss" in row["row_key"]
    assert set(row) == set(INDEX_COLUMNS)


def test_resolve_pick_errors() -> None:
    t = _tables()
    base = dict(
        value="log_loss", metric="log_loss", sport="nfl", stage="s", experiment="e", feature_set="f"
    )
    # ambiguous: two rows match
    e = resolve_pick(t, Pick("m", (("task", "pass_completion"),), **base))
    assert isinstance(e, PickError) and e.n_matched == 2
    # no row
    e = resolve_pick(t, Pick("m", (("model", "nope"),), **base))
    assert isinstance(e, PickError) and e.n_matched == 0
    # missing column
    e = resolve_pick(t, Pick("m", (("model", "PBP"),), **{**base, "value": "auc"}))
    assert isinstance(e, PickError) and e.missing_columns == ("auc",)
    # NaN value is an error, not a silent row
    e = resolve_pick(t, Pick("m", (("task", "pass_yards"), ("model", "PBP")), **base))
    assert isinstance(e, PickError) and e.missing_columns == ("log_loss",)
    # unknown table
    e = resolve_pick(t, Pick("zz", (("model", "PBP"),), **base))
    assert isinstance(e, PickError) and e.missing_columns == ("zz",)


def test_build_index_collects_rows_and_errors() -> None:
    t = _tables()
    base = dict(metric="m", sport="nfl", stage="s", experiment="e", feature_set="f")
    picks = [
        Pick("m", (("model", "PBP"), ("task", "pass_completion")), "log_loss", **base),
        Pick("m", (("model", "PBP+IMP"), ("task", "pass_completion")), "log_loss", **base),
        Pick("m", (("model", "ghost"),), "log_loss", **base),
    ]
    index, errors = build_index(t, picks)
    assert list(index.columns) == INDEX_COLUMNS
    assert len(index) == 2 and len(errors) == 1
    assert index["value"].tolist() == pytest.approx([0.60, 0.61])
    assert index["ci_low"].isna().all() and (index["ci_type"] == "").all()
    assert str(index["n"].dtype) == "Int64"


def test_catalogue_is_well_formed() -> None:
    picks = all_picks()
    assert len(picks) > 500
    keys = {(p.table, p.where, p.value) for p in picks}
    assert len(keys) == len(picks), "duplicate picks in the catalogue"
    for p in picks:
        assert p.source_report.endswith(".md")
        assert (p.ci_low is None) == (p.ci_high is None)
        assert p.sport in ("nfl", "soccer")
