"""Tests for the shared scenario-EV machinery on small synthetic inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import common as C


def test_chronological_split_keeps_matches_whole() -> None:
    mk = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    order = np.array([10, 10, 20, 20, 30, 30, 40, 40, 50, 50])
    lab = C.chronological_split(mk, order, 0.6)
    assert list(lab) == [C.DISCOVERY] * 6 + [C.CONFIRMATION] * 4
    for m in np.unique(mk):
        assert len(set(lab[mk == m])) == 1


def test_chronological_split_never_empties_a_side() -> None:
    mk = np.array([1, 2])
    lab = C.chronological_split(mk, np.array([1, 2]), 0.999)
    assert set(lab) == {C.DISCOVERY, C.CONFIRMATION}


def test_prior_expanding_is_strictly_prior() -> None:
    df = pd.DataFrame({
        "g": ["a", "a", "a", "b", "b"],
        "v": [1.0, 2.0, 4.0, 10.0, 20.0],
        "d": [1, 2, 3, 1, 2],
        "id": [1, 2, 3, 4, 5],
    })
    out = C.prior_expanding(df, ["g"], ["v"], ["d", "id"])
    assert list(out["prior_sum_v"]) == [0.0, 1.0, 3.0, 0.0, 10.0]
    assert list(out["prior_n"]) == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_prior_expanding_treats_nan_as_zero_but_counts_the_row() -> None:
    df = pd.DataFrame({"g": ["a"] * 3, "v": [np.nan, 2.0, 3.0], "d": [1, 2, 3],
                       "id": [1, 2, 3]})
    out = C.prior_expanding(df, ["g"], ["v"], ["d", "id"])
    assert list(out["prior_sum_v"]) == [0.0, 0.0, 2.0]
    assert list(out["prior_n"]) == [0.0, 1.0, 2.0]


def test_prior_expanding_respects_unsorted_input() -> None:
    df = pd.DataFrame({"g": ["a"] * 4, "v": [4.0, 1.0, 3.0, 2.0], "d": [4, 1, 3, 2],
                       "id": [4, 1, 3, 2]})
    out = C.prior_expanding(df, ["g"], ["v"], ["d", "id"])
    assert list(out["prior_sum_v"]) == [6.0, 0.0, 3.0, 1.0]


def test_audit_strictly_prior_agrees_with_brute_force() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "g": rng.integers(0, 5, 60),
        "v": rng.normal(size=60),
        "d": rng.integers(0, 20, 60),
    })
    df["id"] = np.arange(60)
    out = C.prior_expanding(df, ["g"], ["v"], ["d", "id"])
    joined = pd.concat([df, out], axis=1)
    res = C.audit_strictly_prior(joined, ["g"], "v", ["d", "id"], "prior_sum_v", "prior_n",
                                 n_check=60)
    assert res["max_abs_sum_error"] < 1e-9
    assert res["max_abs_count_error"] == 0.0


def test_shrunk_rate_endpoints() -> None:
    assert C.shrunk_rate(np.array([0.0]), np.array([0.0]), 0.2, 5.0)[0] == pytest.approx(0.2)
    got = C.shrunk_rate(np.array([50.0]), np.array([100.0]), 0.2, 5.0)[0]
    assert got == pytest.approx((50 + 5 * 0.2) / 105)


def test_novig_two_way_sums_to_one_and_inverts_pricing() -> None:
    p = np.array([0.3, 0.5, 0.75])
    oa, ob = C.price_two_way(p, 0.06)
    assert np.allclose(1 / oa + 1 / ob, 1.06)
    pa, pb = C.novig_two_way(oa, ob)
    assert np.allclose(pa, p)
    assert np.allclose(pa + pb, 1.0)


def test_simulate_two_way_bets_only_above_threshold() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p_book = np.array([0.5, 0.5, 0.5, 0.5])
    p_model = np.array([0.9, 0.5, 0.5, 0.1])
    r = C.simulate_two_way(y, p_model, p_book, np.arange(4),
                           C.BetSimConfig(hold=0.0, edge_threshold=0.1, n_boot=50))
    # only rows 0 (over) and 3 (under) carry a 0.8 edge; both win at even money
    assert r.n_bets == 2
    assert r.n_over == 1
    assert r.roi == pytest.approx(1.0)


def test_simulate_two_way_no_bets_is_safe() -> None:
    r = C.simulate_two_way(np.array([1.0, 0.0]), np.array([0.5, 0.5]),
                           np.array([0.5, 0.5]), np.arange(2),
                           C.BetSimConfig(hold=0.06, edge_threshold=0.5, n_boot=10))
    assert r.n_bets == 0
    assert np.isnan(r.roi)


def test_clustered_bootstrap_mean_matches_sample_mean() -> None:
    v = np.array([1.0, -1.0, 2.0, 0.0])
    m, lo, hi = C.clustered_bootstrap_mean(v, np.array([1, 1, 2, 2]), n_boot=500, seed=1)
    assert m == pytest.approx(0.5)
    assert lo <= m <= hi


def test_nb_sf_reduces_to_poisson_for_large_r() -> None:
    from scipy.stats import poisson

    mu = np.array([2.0, 3.5])
    assert np.allclose(C.nb_sf(2.5, mu, np.inf), poisson.sf(2, mu))
    assert C.nb_sf(2.5, np.array([3.0]), 5.0)[0] != pytest.approx(poisson.sf(2, 3.0))


def test_nb_sf_is_monotone_in_the_line() -> None:
    mu = np.full(1, 3.0)
    vals = [C.nb_sf(L, mu, 10.0)[0] for L in (1.5, 2.5, 3.5, 4.5)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_fit_nb_dispersion_recovers_overdispersion() -> None:
    rng = np.random.default_rng(3)
    mu = np.full(4000, 4.0)
    r_true = 6.0
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    r_hat = C.fit_nb_dispersion(y.astype(float), mu)
    assert 2.0 < r_hat < 30.0


def test_platt_round_trip_is_monotone_and_calibrates() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    ab = C.platt_fit(np.clip(p * 0.5, 1e-4, 1 - 1e-4), y)
    cal = C.platt_apply(np.clip(p * 0.5, 1e-4, 1 - 1e-4), ab)
    assert abs(cal.mean() - y.mean()) < 0.03
    order = np.argsort(p)
    assert np.all(np.diff(cal[order]) >= -1e-9)


def test_disagreement_table_shape_and_direction() -> None:
    rng = np.random.default_rng(7)
    n = 500
    p_book = np.full(n, 0.5)
    p_model = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p_model).astype(float)
    t = C.disagreement_table(y, p_model, p_book, n_bins=4)
    assert len(t) == 4
    assert t["n"].sum() == n
    assert t["model_better"].mean() > 0


def test_prior_group_daily_mean_excludes_same_day_rows() -> None:
    df = pd.DataFrame({
        "div": ["A", "A", "A", "A"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-08", "2020-01-15"]),
        "v": [2.0, 4.0, 10.0, 0.0],
        "has": [1.0, 1.0, 1.0, 1.0],
    })
    out = C.prior_group_daily_mean(df, "div", "date", "v", "has", prior_mean=1.0, k=0.0)
    assert out[0] == pytest.approx(1.0)      # no history yet -> the backstop
    assert out[0] == out[1]                  # same-day rows share a (empty) prior
    assert out[2] == pytest.approx(3.0)      # only the two 2020-01-01 rows
    assert out[3] == pytest.approx(16 / 3)   # 2 + 4 + 10 over three matches


def test_per90_and_terciles() -> None:
    assert C.per90(np.array([2.0]), np.array([45.0]))[0] == pytest.approx(4.0)
    assert C.per90(np.array([2.0]), np.array([1.0]), floor=30.0)[0] == pytest.approx(6.0)
    lo, hi = C.tercile_edges(np.array([0.0, 1.0, 2.0, 3.0, np.nan]))
    assert lo < hi
