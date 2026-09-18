"""Tests for the pure helpers of scenario_ev stage 03 (player pass counts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev import pass_counts as pc


# ---------------------------------------------------------------------------
# Poisson binomial
# ---------------------------------------------------------------------------


def test_poisson_binomial_matches_brute_force() -> None:
    p = np.array([0.2, 0.5, 0.9, 0.05])
    pmf = pc.poisson_binomial_pmf(p)
    brute = np.zeros(len(p) + 1)
    for mask in range(1 << len(p)):
        bits = [(mask >> i) & 1 for i in range(len(p))]
        prob = np.prod([pi if b else 1 - pi for pi, b in zip(p, bits)])
        brute[sum(bits)] += prob
    assert np.allclose(pmf, brute)
    assert pmf.sum() == pytest.approx(1.0)


def test_poisson_binomial_reduces_to_binomial() -> None:
    from scipy.stats import binom

    n, q = 7, 0.3
    pmf = pc.poisson_binomial_pmf(np.full(n, q))
    assert np.allclose(pmf, binom.pmf(np.arange(n + 1), n, q))


def test_poisson_binomial_empty_input() -> None:
    assert np.allclose(pc.poisson_binomial_pmf(np.array([])), [1.0])


def test_batch_matches_single_and_ignores_padding() -> None:
    p = np.array([[0.2, 0.5, 0.9], [0.4, 0.7, 0.0]])
    valid = np.array([[True, True, True], [True, True, False]])
    batch = pc.poisson_binomial_pmf_batch(p, valid)
    assert np.allclose(batch[0], pc.poisson_binomial_pmf([0.2, 0.5, 0.9]))
    single = pc.poisson_binomial_pmf([0.4, 0.7])
    assert np.allclose(batch[1][: len(single)], single)
    assert batch[1][-1] == pytest.approx(0.0)


def test_pmf_sf_mean_var_and_log_score() -> None:
    pmf = pc.poisson_binomial_pmf([0.5, 0.5])[None, :]
    assert pc.pmf_sf(pmf, 0.5)[0] == pytest.approx(0.75)  # P(1) + P(2)
    assert pc.pmf_sf(pmf, 1.5)[0] == pytest.approx(0.25)
    m, v = pc.pmf_mean_var(pmf)
    assert m[0] == pytest.approx(1.0)
    assert v[0] == pytest.approx(0.5)
    assert pc.pmf_log_score(pmf, np.array([1]))[0] == pytest.approx(-np.log(0.5))


def test_pad_by_group_places_rows_in_order() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    codes = np.array([0, 0, 1, 1])
    out, valid = pc.pad_by_group(values, codes, 2, 3)
    assert np.allclose(out[0], [0.1, 0.2, 0.0])
    assert np.allclose(out[1], [0.3, 0.4, 0.0])
    assert valid.tolist() == [[True, True, False], [True, True, False]]


def test_pad_by_group_truncates_beyond_width() -> None:
    out, valid = pc.pad_by_group(np.arange(5.0), np.zeros(5, dtype=int), 1, 3)
    assert out.shape == (1, 3)
    assert valid.sum() == 3


def test_count_distributions_recovers_attempts_and_completions() -> None:
    df = pd.DataFrame({
        "match_id": [1, 1, 1, 2],
        "player_id": [10, 10, 11, 10],
        "is_complete": [1.0, 0.0, 1.0, 1.0],
        "pred_M": [0.8, 0.6, 0.9, 0.7],
        "match_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"]),
        "competition": ["X"] * 4, "gender": ["male"] * 4, "team_id": [1, 1, 1, 2],
    })
    frame, pmfs = pc.count_distributions(df, ["M"])
    assert len(frame) == 3
    row = frame[(frame.match_id == 1) & (frame.player_id == 10)].iloc[0]
    assert row["attempts"] == 2 and row["completed"] == 1
    mu, _ = pc.pmf_mean_var(pmfs["M"])
    assert mu.sum() == pytest.approx(0.8 + 0.6 + 0.9 + 0.7)


# ---------------------------------------------------------------------------
# Count distributions and lines
# ---------------------------------------------------------------------------


def test_binomial_mix_sf_degenerate_attempts() -> None:
    p_count = np.zeros((1, 11))
    p_count[0, 10] = 1.0
    from scipy.stats import binom

    got = pc.binomial_mix_sf(p_count, np.array([0.5]), 5.5)
    assert got[0] == pytest.approx(binom.sf(5, 10, 0.5))


def test_binomial_mix_logpmf_degenerate_attempts() -> None:
    from scipy.stats import binom

    p_count = np.zeros((1, 9))
    p_count[0, 8] = 1.0
    got = pc.binomial_mix_logpmf(p_count, np.array([0.7]), np.array([6]))
    assert got[0] == pytest.approx(np.log(binom.pmf(6, 8, 0.7)))


def test_binomial_mix_never_exceeds_attempts() -> None:
    p_count = np.zeros((1, 6))
    p_count[0, 3] = 1.0
    assert pc.binomial_mix_sf(p_count, np.array([0.9]), 3.5)[0] == pytest.approx(0.0)


def test_nb_pmf_grid_sums_to_one_and_matches_poisson() -> None:
    from scipy.stats import poisson

    g = pc.nb_pmf_grid(np.array([4.0, 9.0]), float("inf"), 60)
    assert np.allclose(g.sum(axis=1), 1.0)
    assert np.allclose(g[0, :20], poisson.pmf(np.arange(20), 4.0), atol=1e-12)


def test_nb_pmf_grid_overdispersed_has_more_variance() -> None:
    mu = np.array([20.0])
    v_p = pc.pmf_mean_var(pc.nb_pmf_grid(mu, float("inf"), 200))[1][0]
    v_n = pc.pmf_mean_var(pc.nb_pmf_grid(mu, 5.0, 200))[1][0]
    assert v_n > v_p


def test_line_from_mean() -> None:
    assert np.allclose(pc.line_from_mean(np.array([10.2, 10.9])), [10.5, 10.5])
    assert np.allclose(pc.line_from_mean(np.array([10.2]), 3), [13.5])
    assert np.allclose(pc.line_from_mean(np.array([0.1]), -3), [0.5])


# ---------------------------------------------------------------------------
# Rolling / prior features
# ---------------------------------------------------------------------------


def test_rolling_prior_mean_is_strictly_prior_and_windowed() -> None:
    df = pd.DataFrame({
        "g": ["a"] * 4 + ["b"] * 2,
        "v": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
        "d": pd.to_datetime(["2020-01-0" + str(i) for i in (1, 2, 3, 4, 1, 2)]),
        "m": [1, 2, 3, 4, 5, 6],
    })
    got = pc.rolling_prior_mean(df, ["g"], "v", ["d", "m"], 2)
    assert np.isnan(got[0])
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(1.5)
    assert got[3] == pytest.approx(2.5)  # last two only
    assert np.isnan(got[4]) and got[5] == pytest.approx(10.0)


def test_position_group() -> None:
    assert pc.position_group("Left Center Back") == "CB"
    assert pc.position_group("Right Wing Back") == "FB"
    assert pc.position_group(float("nan")) == "UNK"
    assert pc.position_group("Some New Role") == "UNK"


def test_usual_passer_frame_identifies_and_flags_absence() -> None:
    # player 1 passes most for team A; he misses the fifth match
    rows = []
    for i, d in enumerate(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]):
        rows += [{"match_id": i, "team_id": "A", "player_id": 1, "date": d, "minutes": 90,
                  "passes": 90},
                 {"match_id": i, "team_id": "A", "player_id": 2, "date": d, "minutes": 90,
                  "passes": 30}]
    rows += [{"match_id": 9, "team_id": "A", "player_id": 2, "date": "2020-01-29",
              "minutes": 90, "passes": 30}]
    app = pd.DataFrame(rows)
    app["date"] = pd.to_datetime(app["date"])
    out = pc.usual_passer_frame(app, min_prior=3)
    last = out[out.match_id == 9].iloc[0]
    assert last["key_player_id"] == 1
    assert last["key_absent"] == 1.0
    assert last["key_known"] == 1.0
    early = out[out.match_id == 0].iloc[0]
    assert early["key_known"] == 0.0  # no history yet


def test_usual_passer_uses_only_earlier_matches() -> None:
    # a player who explodes in the last match must not be picked for that match
    rows = [{"match_id": i, "team_id": "A", "player_id": p, "date": d, "minutes": 90,
             "passes": 100 if (p == 2 and i == 3) else (50 if p == 1 else 10)}
            for i, d in enumerate(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"])
            for p in (1, 2)]
    app = pd.DataFrame(rows)
    app["date"] = pd.to_datetime(app["date"])
    out = pc.usual_passer_frame(app, min_prior=3)
    assert out[out.match_id == 3].iloc[0]["key_player_id"] == 1


# ---------------------------------------------------------------------------
# Calibration and metrics
# ---------------------------------------------------------------------------


def test_poisson_glm_link_recovers_identity() -> None:
    rng = np.random.default_rng(0)
    mu = rng.gamma(30.0, 1.0, 4000)
    y = rng.poisson(mu)
    a, b = pc.poisson_glm_link(mu, y)
    assert a == pytest.approx(0.0, abs=0.12)
    assert b == pytest.approx(1.0, abs=0.04)


def test_poisson_glm_link_corrects_a_scale_error() -> None:
    rng = np.random.default_rng(1)
    mu = rng.gamma(30.0, 1.0, 4000)
    y = rng.poisson(mu)
    a, b = pc.poisson_glm_link(0.5 * mu, y)
    fixed = pc.poisson_glm_apply(0.5 * mu, (a, b))
    assert fixed.mean() == pytest.approx(y.mean(), rel=0.02)


def test_count_metrics_perfect_and_ordering() -> None:
    y = np.array([3.0, 5.0, 7.0])
    good = pc.count_metrics(y, y)
    bad = pc.count_metrics(y, np.full(3, 5.0))
    assert good["mae"] == 0.0
    assert good["pois_dev"] == pytest.approx(0.0)
    assert good["log_score"] < bad["log_score"]


def test_per_row_log_score_matches_scipy() -> None:
    from scipy.stats import nbinom, poisson

    y = np.array([2.0, 8.0])
    mu = np.array([3.0, 6.0])
    assert np.allclose(pc.per_row_log_score(y, mu, float("inf")), -poisson.logpmf(y, mu))
    r = 4.0
    assert np.allclose(pc.per_row_log_score(y, mu, r), -nbinom.logpmf(y, r, r / (r + mu)))


def test_variance_decomposition_pure_volume() -> None:
    # every pass completed: all the variance must be volume
    df = pd.DataFrame({"passes": [10.0, 40.0, 70.0],
                       "passes_completed": [10.0, 40.0, 70.0]})
    d = pc.variance_decomposition(df).iloc[0]
    assert d["var_within_match_bernoulli"] == pytest.approx(0.0)
    assert d["share_volume_and_rate_level"] == pytest.approx(1.0)


def test_variance_decomposition_pure_noise() -> None:
    # constant attempts and a constant realised rate: the Bernoulli term is p(1-p) * n
    df = pd.DataFrame({"passes": [100.0] * 3, "passes_completed": [50.0, 50.0, 50.0]})
    d = pc.variance_decomposition(df).iloc[0]
    assert d["var_completed"] == pytest.approx(0.0)
    assert d["var_within_match_bernoulli"] == pytest.approx(25.0)
    assert np.isnan(d["share_volume_and_rate_level"])


def test_fold_labels_cover_every_row_and_keep_groups_together() -> None:
    groups = np.repeat(np.arange(20), 3)
    f = pc.fold_labels(groups, 5, seed=0)
    assert set(np.unique(f)) == set(range(5))
    for g in np.unique(groups):
        assert len(set(f[groups == g])) == 1


def test_scenario_mask_and_cuts() -> None:
    df = pd.DataFrame({
        "op_block_depth": [40.0, 45.0, 50.0, 55.0, 60.0, 65.0],
        "op_ppda": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "p_att_per90": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "poss_edge": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        "key_absent": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "key_known": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    })
    cuts = pc.discovery_cuts(df)
    assert pc.scenario_mask(df, "metronome_out", cuts).tolist() == [
        True, False, False, False, False, False]
    assert pc.scenario_mask(df, "deep_opponent", cuts).sum() == 2
    assert pc.scenario_mask(df, "high_volume_passer", cuts).sum() == 2
    with pytest.raises(KeyError):
        pc.scenario_mask(df, "nope", cuts)


def test_choose_threshold_prefers_a_profitable_rule() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    p_true = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_book = np.full(n, 0.5)
    cfg = pc.PassConfig()
    t = pc.choose_threshold(y, p_true, p_book, np.arange(n), 0.06, cfg, min_bets=50)
    assert t in cfg.thresholds
