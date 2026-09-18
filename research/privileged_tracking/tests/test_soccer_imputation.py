"""Smoke tests for the pure helpers of the soccer 02 imputation driver (no data access)."""

from __future__ import annotations

import numpy as np
import pytest

from research.privileged_tracking.soccer import imputation as si
from research.privileged_tracking.soccer import imputation_features as imf


def test_thin_rows_is_deterministic_and_sorted() -> None:
    idx = np.arange(1000)
    a = si.thin_rows(idx, 100, seed=3)
    b = si.thin_rows(idx, 100, seed=3)
    assert len(a) == 100 and np.array_equal(a, b) and np.all(np.diff(a) > 0)
    assert si.thin_rows(idx, 2000, seed=0) is idx


def test_resolve_targets_expands_groups_and_rejects_unknown() -> None:
    assert si.resolve_targets(None) == [t.name for t in imf.TARGETS]
    group = next(iter(imf.TARGET_GROUPS))
    assert si.resolve_targets(group) == list(imf.TARGET_GROUPS[group])
    first = imf.TARGETS[0].name
    assert si.resolve_targets(f"{first}, {first}") == [first, first]
    with pytest.raises(KeyError):
        si.resolve_targets("no_such_target")


def test_skill_and_per_sample_loss_by_kind() -> None:
    assert si.skill_of("binary", {"bss": 0.3, "r2": 0.9}) == 0.3
    assert si.skill_of("reg", {"bss": 0.3, "r2": 0.9}) == 0.9
    y = np.array([0.0, 1.0])
    p = np.array([0.2, 1.4])
    assert si.per_sample_loss("reg", y, p) == pytest.approx([0.04, 0.16])
    ll = si.per_sample_loss("binary", y, p)  # clipped to [0, 1] before the log-loss
    assert ll[0] == pytest.approx(-np.log(0.8)) and ll[1] < 1e-5
