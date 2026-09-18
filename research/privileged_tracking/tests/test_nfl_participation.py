"""Smoke tests for the pure helpers of the NFL 02 participation driver (no data access)."""

from __future__ import annotations

import numpy as np
import pytest

from research.privileged_tracking.nfl import participation as pa


def test_multiclass_metrics_and_row_normalisation() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.4, 0.4, 0.2]])
    y = np.array([0, 2, 1])
    m = pa.mc_metrics(y, probs)
    assert m["n"] == 3 and m["accuracy"] == pytest.approx(2 / 3) and m["top2"] == 1.0
    ll = pa.mc_per_sample_logloss(y, probs)
    assert ll == pytest.approx(-np.log([0.7, 0.6, 0.4]))
    assert m["log_loss"] == pytest.approx(float(ll.mean()))
    raw = np.array([[2.0, 2.0], [0.0, 0.0]])
    norm = pa.normalise_rows(raw)
    assert norm[0].tolist() == [0.5, 0.5] and norm[1].tolist() == [0.0, 0.0]


def test_delta_row_reports_play_and_game_intervals() -> None:
    rng = np.random.default_rng(0)
    la = rng.uniform(0.5, 1.5, 300)
    lb = la - 0.2
    groups = rng.integers(0, 10, 300)
    r = pa._delta_row("A", "B", la, lb, n_boot=100, groups=groups)
    assert r["from"] == "A" and r["to"] == "B" and r["n"] == 300 and r["n_games"] == 10
    assert r["delta_loss"] == pytest.approx(0.2) and r["ci_low_game"] > 0.19
