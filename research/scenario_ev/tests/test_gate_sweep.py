"""Unit tests for the stage-05 sweep helpers (no model fitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import common as C
from research.scenario_ev import gate_sweep as GS


def _target(n: int = 300) -> GS.GateTarget:
    """A tiny target carrying a proxy column and a chronological split."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "match_id": np.repeat(np.arange(n // 2), 2),
        "mu": rng.uniform(0, 10, size=n),
        "y": rng.poisson(3.0, size=n).astype(float),
    })
    df["split"] = C.chronological_split(df["match_id"], df["match_id"], 0.6)
    return GS.GateTarget(name="t", owner="tests", df=df, cols=["mu"], target="y",
                         kind="count", fit_predict=lambda a, b, c, s: np.ones(len(c)),
                         proxy_col="mu")


def test_generic_scenarios_are_discovery_terciles() -> None:
    t = GS.add_generic_scenarios(_target())
    assert set(t.scenarios) == {"proxy_low", "proxy_high"}
    disc = (t.df["split"] == C.DISCOVERY).to_numpy()
    lo = t.scenarios["proxy_low"]
    hi = t.scenarios["proxy_high"]
    assert not (lo & hi).any()
    # each cut is near a third of the discovery rows, by construction
    assert 0.25 < lo[disc].mean() < 0.42
    assert 0.25 < hi[disc].mean() < 0.42
    assert "bottom discovery tercile" in t.descriptions["proxy_low"]


def test_generic_scenarios_are_skipped_without_a_proxy_column() -> None:
    t = _target()
    t.proxy_col = "not_a_column"
    assert GS.add_generic_scenarios(t).scenarios == {}


def _summary_row(**kw: object) -> dict[str, object]:
    base = dict(target_label="m", scenario="s", where="confirmation",
                region="in_scenario", n=1000, n_matches=500, n_train_scenario=800,
                mean_y=1.0, kind="count", loss_generalist=1.0, loss_specialist=0.99,
                delta_gen_minus_spec=0.01, ci_lo=0.005, ci_hi=0.015, n_seeds=1,
                refit_spread=float("nan"), verdict="room", description="d")
    base.update(kw)
    return base


def test_headline_table_applies_the_market_noise_floor() -> None:
    rows = pd.DataFrame([
        # the market's primary scenario, run under three seeds: a big measured spread
        _summary_row(scenario="primary", n_seeds=3, refit_spread=0.05,
                     delta_gen_minus_spec=0.01, ci_lo=0.005, ci_hi=0.015),
        _summary_row(scenario="primary", where="discovery_cv", n_seeds=3,
                     refit_spread=0.05),
        # a one-seed cell with a delta well inside that spread
        _summary_row(scenario="other", n_seeds=1, delta_gen_minus_spec=0.01,
                     ci_lo=0.005, ci_hi=0.015),
        _summary_row(scenario="other", where="discovery_cv", n_seeds=1),
    ])
    head = GS.headline_table(rows).set_index("scenario")
    assert head["refit_noise_floor"].tolist() == pytest.approx([0.05, 0.05])
    # both cells have intervals clear of zero but neither clears the noise floor
    assert head.loc["primary", "verdict"].startswith("inconclusive (delta inside refit")
    assert head.loc["other", "verdict"].startswith("inconclusive (delta inside refit")


def test_headline_table_calls_a_clear_win_room() -> None:
    rows = pd.DataFrame([
        _summary_row(n_seeds=3, refit_spread=0.0001, delta_gen_minus_spec=0.02,
                     ci_lo=0.01, ci_hi=0.03),
        _summary_row(where="discovery_cv", n_seeds=3, refit_spread=0.0001),
    ])
    assert GS.headline_table(rows)["verdict"].iloc[0] == "room"


def test_headline_table_calls_a_loss_no_room() -> None:
    rows = pd.DataFrame([
        _summary_row(delta_gen_minus_spec=-0.02, ci_lo=-0.03, ci_hi=-0.01),
        _summary_row(where="discovery_cv"),
    ])
    assert GS.headline_table(rows)["verdict"].iloc[0] == "no room"


def test_headline_table_carries_both_halves_and_the_reverse_check() -> None:
    rows = pd.DataFrame([
        _summary_row(n=111),
        _summary_row(where="discovery_cv", n=222),
        _summary_row(region="off_scenario", delta_gen_minus_spec=-0.5),
    ])
    h = GS.headline_table(rows).iloc[0]
    assert h["n_confirmation"] == 111
    assert h["n_discovery"] == 222
    assert h["delta_off_scenario"] == pytest.approx(-0.5)


def test_markdown_renderer_formats_floats_and_ints() -> None:
    df = pd.DataFrame({"a": [1], "b": [0.123456], "c": ["x"], "d": [np.nan]})
    out = GS._md(df, nd=3)
    assert out.splitlines()[0] == "| a | b | c | d |"
    assert "| 1 | 0.123 | x | n/a |" in out


def test_every_builder_is_named_and_unique() -> None:
    assert set(GS.BUILDERS) == {
        "player_cards", "team_cards", "match_corners", "team_corners",
        "player_passes", "player_passes_completed", "team_fouls", "player_fouls"}


def test_headline_table_leaves_an_unmeasured_floor_missing() -> None:
    """A market with no multi-seed cell has no floor, and cannot be called "room"."""
    rows = pd.DataFrame([
        _summary_row(n_seeds=1, refit_spread=float("nan"),
                     delta_gen_minus_spec=0.02, ci_lo=0.01, ci_hi=0.03),
        _summary_row(where="discovery_cv", n_seeds=1, refit_spread=float("nan")),
    ])
    head = GS.headline_table(rows).iloc[0]
    assert not np.isfinite(head["refit_noise_floor"])
    assert head["verdict"] == "inconclusive (no measured refit floor)"
    # and it prints as n/a rather than as a measured floor of zero
    assert "n/a" in GS._md(GS.headline_table(rows), ["market", "refit_noise_floor"])
