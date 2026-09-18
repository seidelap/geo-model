"""Unit tests for the phase results index (pure helpers on synthetic frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import results_index as RI


def _frame() -> pd.DataFrame:
    """A tiny stand-in for a stage result table."""
    return pd.DataFrame(
        {
            "where": ["discovery", "confirmation", "confirmation"],
            "model": ["P0", "P1", "P2"],
            "hold": [0.04, 0.06, 0.06],
            "line": [2.5, 3.5, np.nan],
            "delta": [0.001, 0.002, 0.003],
            "ci_lo": [0.000, 0.001, 0.002],
            "ci_hi": [0.002, 0.003, 0.004],
            "n": [100, 200, 300],
        }
    )


def _spec(**kw: object) -> RI.MetricSpec:
    """Build a spec with sensible defaults for the synthetic frame."""
    base: dict[str, object] = dict(
        candidate="cards", stage="01", experiment="vs_proxy", model="P1",
        metric="delta_log_loss_vs_proxy_nats", source_table="t", source_report="t.md",
        split="confirmation", value_col="delta",
        filters={"where": "confirmation", "model": "P1"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    )
    base.update(kw)
    return RI.MetricSpec(**base)  # type: ignore[arg-type]


def test_select_rows_matches_strings_and_floats() -> None:
    df = _frame()
    out = RI.select_rows(df, {"where": "confirmation", "hold": 0.06})
    assert len(out) == 2
    assert set(out["model"]) == {"P1", "P2"}


def test_select_rows_tolerates_float_representation() -> None:
    df = _frame()
    # 0.04 stored as a float is matched by a value that differs in the last bits
    assert len(RI.select_rows(df, {"hold": 0.04 + 1e-15})) == 1


def test_select_rows_matches_nulls_with_none() -> None:
    assert len(RI.select_rows(_frame(), {"line": None})) == 1


def test_select_rows_rejects_an_unknown_column() -> None:
    with pytest.raises(KeyError):
        RI.select_rows(_frame(), {"not_a_column": 1})


def test_resolve_spec_reads_value_ci_and_n() -> None:
    row = RI.resolve_spec(_spec(), _frame())
    assert row["value"] == pytest.approx(0.002)
    assert row["ci_low"] == pytest.approx(0.001)
    assert row["ci_high"] == pytest.approx(0.003)
    assert row["n"] == pytest.approx(200.0)
    assert row["split"] == "confirmation"
    assert set(row) == set(RI.INDEX_COLUMNS)


def test_resolve_spec_leaves_the_interval_nan_when_none_is_declared() -> None:
    row = RI.resolve_spec(_spec(ci_lo_col=None, ci_hi_col=None, ci_type=RI.NO_CI), _frame())
    assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])
    assert row["ci_type"] == "none"


@pytest.mark.parametrize("filters", [{"model": "nope"}, {"where": "confirmation"}])
def test_resolve_spec_fails_unless_exactly_one_row_matches(filters: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="expected 1"):
        RI.resolve_spec(_spec(filters=filters), _frame())


def test_resolve_spec_spread_takes_the_range_over_the_matched_rows() -> None:
    row = RI.resolve_spec(
        _spec(agg="spread", filters={"where": "confirmation"}, value_col="delta",
              ci_lo_col=None, ci_hi_col=None, ci_type=RI.NO_CI, n_col=None),
        _frame(),
    )
    assert row["value"] == pytest.approx(0.001)  # 0.003 - 0.002
    assert np.isnan(row["n"])


def test_resolve_spec_spread_needs_more_than_one_row() -> None:
    with pytest.raises(ValueError, match="spread needs"):
        RI.resolve_spec(_spec(agg="spread", filters={"model": "P1"}), _frame())


def _gate_frame() -> pd.DataFrame:
    """Two cells of a stand-in gate sweep headline table."""
    return pd.DataFrame(
        {
            "market": ["team_cards", "team_corners"],
            "scenario": ["high_foul_both", "fav_strong"],
            "delta_gen_minus_spec": [-0.005, 0.0054],
            "ci_lo": [-0.006, 0.003],
            "ci_hi": [-0.004, 0.007],
            "n_confirmation": [26015, 9272],
        }
    )


def test_gate_index_maps_markets_to_candidates_and_keeps_the_sign() -> None:
    out = RI.gate_index(_gate_frame())
    assert list(out.columns) == list(RI.INDEX_COLUMNS)
    assert list(out["candidate"]) == ["cards", "corners"]
    assert (out["split"] == "confirmation").all()
    assert (out["experiment"] == "gate").all()
    # positive means the specialist is better, i.e. targeting has room
    assert out.loc[1, "value"] == pytest.approx(0.0054)
    assert out.loc[0, "n"] == pytest.approx(26015.0)


def test_gate_index_rejects_an_unmapped_market() -> None:
    bad = _gate_frame().assign(market=["team_cards", "team_throwins"])
    with pytest.raises(KeyError, match="unmapped"):
        RI.gate_index(bad)


def test_every_declared_spec_is_uniquely_named() -> None:
    """A (candidate, experiment, model, metric, split) key identifies exactly one row.

    ``split`` is part of the key because a few metrics are carried on both halves on
    purpose -- the corner ``fav_strong`` ROI is quoted on discovery as well as
    confirmation, because the gap between the two halves is the point being made.
    """
    keys = [(s.candidate, s.experiment, s.model, s.metric, s.split) for s in RI.SPECS]
    assert len(set(keys)) == len(keys)


def test_every_declared_spec_names_a_known_split_and_ci_type() -> None:
    for s in RI.SPECS:
        assert s.split in {"discovery", "confirmation", "cv_all"}
        assert (s.ci_type.startswith("match_clustered_bootstrap")
                or s.ci_type == RI.NO_CI
                or s.ci_type.startswith("seed_range"))
        assert s.agg in {"one", "spread", "mean"}


def test_mean_agg_reports_the_seed_mean_and_range() -> None:
    """A multi-seed channel is summarised by its mean with the seed range as the interval."""
    df = pd.DataFrame({"where": ["confirmation"] * 4, "model": ["P1"] * 4,
                       "delta": [0.001, 0.002, 0.004, 0.005], "n": [10, 10, 10, 10]})
    spec = _spec(agg="mean", value_col="delta", filters={"where": "confirmation"},
                 ci_lo_col=None, ci_hi_col=None, n_col="n", ci_type="seed_range_4")
    row = RI.resolve_spec(spec, df)
    assert row["value"] == pytest.approx(0.003)
    assert row["ci_low"] == pytest.approx(0.001)
    assert row["ci_high"] == pytest.approx(0.005)
    assert row["n"] == pytest.approx(10.0)


def test_mean_agg_needs_more_than_one_row() -> None:
    df = pd.DataFrame({"where": ["confirmation"], "model": ["P1"], "delta": [0.001],
                       "n": [10]})
    spec = _spec(agg="mean", value_col="delta", filters={"where": "confirmation"},
                 ci_lo_col=None, ci_hi_col=None, n_col="n")
    with pytest.raises(ValueError, match="mean needs"):
        RI.resolve_spec(spec, df)
