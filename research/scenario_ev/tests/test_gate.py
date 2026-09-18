"""Unit tests for the shared generalist-vs-specialist gate (``common.run_gate``)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import common as C


def _pop(n_matches: int = 40, per_match: int = 4, seed: int = 0) -> pd.DataFrame:
    """Synthetic population: two regimes, one of which the generalist under-serves."""
    rng = np.random.default_rng(seed)
    mid = np.repeat(np.arange(n_matches), per_match)
    n = len(mid)
    scen = rng.random(n) < 0.4
    x = rng.normal(size=n)
    # a count target whose rate is much higher inside the scenario, so a model fitted on
    # everything is badly wrong there and a specialist is not
    y = rng.poisson(np.where(scen, 6.0, 1.0) * np.exp(0.1 * x)).astype(float)
    df = pd.DataFrame({"match_id": mid, "x": x, "y": y, "scen": scen,
                       "order": mid.astype(float)})
    df["split"] = C.chronological_split(df["match_id"], df["order"], 0.6)
    return df


def _mean_fit(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, seed: int
              ) -> np.ndarray:
    """A deterministic 'model': predict the training mean everywhere."""
    return np.full(len(X_te), float(np.mean(y_tr)))


def test_nb_log_score_matches_poisson_in_the_limit() -> None:
    from scipy.stats import poisson

    y = np.array([0.0, 1.0, 5.0])
    mu = np.array([1.0, 2.0, 4.0])
    assert np.allclose(C.nb_log_score(y, mu, float("inf")), -poisson.logpmf(y, mu))
    # finite dispersion is a different, heavier-tailed model
    assert not np.allclose(C.nb_log_score(y, mu, 3.0), -poisson.logpmf(y, mu))


def test_gate_loss_binary_is_log_loss() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    got = C.gate_loss("binary", y, p)
    assert got[0] == pytest.approx(-np.log(0.8), rel=1e-4)
    assert got[1] == pytest.approx(-np.log(0.7), rel=1e-4)
    with pytest.raises(ValueError):
        C.gate_loss("nonsense", y, p)


def test_gate_predictions_cover_every_row_and_respect_the_split() -> None:
    df = _pop()
    cfg = C.GateConfig(n_folds=4, min_train_scenario=5, n_boot=50)
    gen = C.gate_predictions(df, ["x"], "y", _mean_fit, cfg, seed=1)
    assert np.isfinite(gen).all()
    disc = (df["split"] == C.DISCOVERY).to_numpy()
    # the confirmation half is predicted by a single fit on all discovery rows
    assert np.allclose(gen[~disc], df.loc[disc, "y"].mean())
    assert gen.shape == (len(df),)


def test_specialist_trains_only_on_scenario_rows() -> None:
    df = _pop()
    cfg = C.GateConfig(n_folds=4, min_train_scenario=5, n_boot=50)
    scen = df["scen"].to_numpy()
    spe = C.gate_predictions(df, ["x"], "y", _mean_fit, cfg, seed=1, subset=scen)
    disc = (df["split"] == C.DISCOVERY).to_numpy()
    assert np.allclose(spe[~disc], df.loc[disc & scen, "y"].mean())


def test_gate_finds_room_when_the_scenario_really_differs() -> None:
    df = _pop()
    cfg = C.GateConfig(n_folds=4, min_train_scenario=5, n_boot=100, min_eval=10)
    rows = C.run_gate(df, ["x"], "y", "count", {"scen": df["scen"].to_numpy()},
                      _mean_fit, cfg, seeds=(1,))
    inside = rows.query("region == 'in_scenario'")
    # the specialist must win inside the scenario (positive delta) ...
    assert (inside["delta_gen_minus_spec"] > 0).all()
    assert (inside["ci_lo"] > 0).all()
    # ... and lose outside it
    outside = rows.query("region == 'off_scenario'")
    assert (outside["delta_gen_minus_spec"] < 0).all()


def test_gate_finds_no_room_when_the_scenario_is_arbitrary() -> None:
    df = _pop()
    rng = np.random.default_rng(7)
    arbitrary = rng.random(len(df)) < 0.4
    cfg = C.GateConfig(n_folds=4, min_train_scenario=5, n_boot=100, min_eval=10)
    rows = C.run_gate(df, ["x"], "y", "count", {"arb": arbitrary}, _mean_fit, cfg,
                      seeds=(1,))
    inside = rows.query("region == 'in_scenario' and where == 'confirmation'")
    assert float(inside["delta_gen_minus_spec"].iloc[0]) < 0.5


def test_summarise_gate_verdicts() -> None:
    base = dict(target="y", kind="count", where="confirmation", region="in_scenario",
                n=100, n_matches=25, mean_y=1.0, loss_generalist=1.0,
                loss_specialist=0.9, n_train_scenario=50, description="")
    rows = pd.DataFrame([
        {**base, "target_label": "room", "scenario": "s", "seed": 1,
         "delta_gen_minus_spec": 0.10, "ci_lo": 0.05, "ci_hi": 0.15},
        {**base, "target_label": "no_room", "scenario": "s", "seed": 1,
         "delta_gen_minus_spec": -0.10, "ci_lo": -0.15, "ci_hi": -0.05},
        {**base, "target_label": "spans_zero", "scenario": "s", "seed": 1,
         "delta_gen_minus_spec": 0.01, "ci_lo": -0.05, "ci_hi": 0.07},
    ])
    out = C.summarise_gate(rows, headline_seed=1).set_index("target_label")
    assert out.loc["room", "verdict"] == "room"
    assert out.loc["no_room", "verdict"] == "no room"
    assert out.loc["spans_zero", "verdict"].startswith("inconclusive")


def test_summarise_gate_calls_a_delta_inside_refit_noise_inconclusive() -> None:
    base = dict(target="y", kind="count", where="confirmation", region="in_scenario",
                n=100, n_matches=25, mean_y=1.0, loss_generalist=1.0,
                loss_specialist=0.9, n_train_scenario=50, description="",
                target_label="t", scenario="s")
    rows = pd.DataFrame([
        {**base, "seed": 1, "delta_gen_minus_spec": 0.02, "ci_lo": 0.01, "ci_hi": 0.03},
        {**base, "seed": 2, "delta_gen_minus_spec": 0.30, "ci_lo": 0.2, "ci_hi": 0.4},
    ])
    out = C.summarise_gate(rows, headline_seed=1)
    assert out["verdict"].iloc[0].startswith("inconclusive (delta inside refit noise)")
    assert out["refit_spread"].iloc[0] == pytest.approx(0.28)
