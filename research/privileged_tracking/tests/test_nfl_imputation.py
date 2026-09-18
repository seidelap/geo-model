"""Smoke tests for the pure helpers of the NFL 03 imputation driver (no data access)."""

from __future__ import annotations

import numpy as np
import pytest

from research.privileged_tracking.nfl import imputation as im


def test_outcome_classes_cover_every_flag_combination() -> None:
    sack = np.array([1, 0, 0, np.nan])
    hit = np.array([0, 1, 0, np.nan])
    assert im.pass_qb_outcome(sack, hit).tolist() == ["sack", "qb_hit", "clean", "na"]
    comp = np.array([0, 1, 0, 0])
    inter = np.array([0, 0, 1, 0])
    assert im.pass_ball_outcome(sack, comp, inter).tolist() == [
        "sack",
        "complete",
        "interception",
        "na",
    ]
    assert im.run_yards_bucket(np.array([-3, 0, 2, 5, 10, 11, np.nan])).tolist() == [
        "<0",
        "0-2",
        "0-2",
        "3-5",
        "6-10",
        "10+",
        "na",
    ]
    out = im.outcome_bucket(
        np.array(["pass", "run", "qb_kneel"], dtype=object),
        np.array([0, np.nan, np.nan]),
        np.array([1, np.nan, np.nan]),
        np.array([1, np.nan, np.nan]),
        np.array([0, np.nan, np.nan]),
        np.array([np.nan, 4, np.nan]),
    )
    assert out.tolist() == ["pass:qb_hit:complete", "run:3-5", "qb_kneel"]


def test_score_skill_and_per_sample_loss_agree_on_definitions() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    p = y + rng.normal(0, 0.5, 200)
    m = im.score("reg", y, p)
    assert m["n"] == 200 and 0.5 < m["r2"] < 1.0
    assert im.skill_of("reg", m) == m["r2"]
    assert im.per_sample_loss("reg", y, p) == pytest.approx(np.abs(y - p))
    yb = (rng.uniform(0, 1, 200) < 0.3).astype(float)
    pb = np.clip(yb * 0.5 + 0.2, 0, 1)
    mb = im.score("binary", yb, pb)
    assert mb["bss"] > 0 and im.skill_of("binary", mb) == mb["bss"]
    assert im.per_sample_loss("binary", yb, pb).shape == (200,)
    mc = im.score("count", np.array([1.0, 2.0, 3.0]), np.array([1.2, 2.6, 2.9]))
    assert mc["exact_rounded"] == pytest.approx(2 / 3) and mc["within_1"] == 1.0
