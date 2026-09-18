"""Tests for the StatsBomb style layer's join and offset model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.scenario_ev.corners_style import (
    add_style_priors,
    build_name_map,
    fit_poisson_offset,
    join_sb_to_odds,
    merge_style_opponent,
)


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A four-team league played over three dates, with different naming in each source."""
    dates = pd.to_datetime(["2016-01-01", "2016-01-08", "2016-01-15"])
    sb = pd.DataFrame({
        "sb_match_id": [1, 2, 3, 4, 5, 6],
        "date": list(dates) * 2,
        "home_team": ["Alpha United", "Beta City", "Alpha United",
                      "Gamma Town", "Delta Rovers", "Gamma Town"],
        "away_team": ["Beta City", "Alpha United", "Beta City",
                      "Delta Rovers", "Gamma Town", "Delta Rovers"],
        "home_score": [1, 0, 2, 3, 1, 0],
        "away_score": [0, 2, 2, 1, 1, 1],
        "division": ["E0"] * 6,
    })
    odds = pd.DataFrame({
        "match_id": [10, 11, 12, 13, 14, 15],
        "MatchDate": list(dates) * 2,
        "Division": ["E0"] * 6,
        "HomeTeam": ["Alpha", "Beta", "Alpha", "Gamma", "Delta", "Gamma"],
        "AwayTeam": ["Beta", "Alpha", "Beta", "Delta", "Gamma", "Delta"],
        "FTHome": [1, 0, 2, 3, 1, 0], "FTAway": [0, 2, 2, 1, 1, 1],
        "HomeCorners": [5, 4, 6, 7, 3, 2], "AwayCorners": [3, 6, 4, 2, 5, 8],
    })
    return sb, odds


def test_build_name_map_recovers_the_pairing_from_dates_alone() -> None:
    sb, odds = _fixtures()
    nm = build_name_map(sb, odds)
    got = dict(zip(nm["sb_team"], nm["odds_team"], strict=True))
    assert got == {"Alpha United": "Alpha", "Beta City": "Beta",
                   "Gamma Town": "Gamma", "Delta Rovers": "Delta"}
    assert (nm["n_shared_dates"] > 0).all()


def test_join_sb_to_odds_is_complete_and_scores_agree() -> None:
    sb, odds = _fixtures()
    joined, rep = join_sb_to_odds(sb, odds, build_name_map(sb, odds))
    assert rep.n_joined == 6
    assert rep.join_rate == pytest.approx(1.0)
    assert rep.score_agreement == pytest.approx(1.0)
    assert rep.n_date_shifted == 0
    assert joined["match_id"].tolist() == sorted(odds["match_id"].tolist())


def test_style_priors_are_strictly_prior_and_opponent_merge_pairs() -> None:
    df = pd.DataFrame({
        "sb_match_id": [1, 1, 2, 2],
        "team_id": [1, 2, 1, 2],
        "opp_id": [2, 1, 2, 1],
        "division": ["E0"] * 4,
        "date": pd.to_datetime(["2016-01-01"] * 2 + ["2016-01-08"] * 2),
        "crosses": [10.0, 20.0, 30.0, 40.0],
    })
    out = add_style_priors(df, ("crosses",), 6, 1)
    first = out[out["sb_match_id"] == 1]
    assert first["sp_n"].tolist() == [0.0, 0.0]
    assert first["sp_crosses"].isna().all()
    second = out[out["sb_match_id"] == 2].sort_values("team_id")
    assert second["sp_crosses"].tolist() == [10.0, 20.0]
    merged = merge_style_opponent(out, ["sp_crosses"])
    row = merged[(merged["sb_match_id"] == 2) & (merged["team_id"] == 1)].iloc[0]
    assert row["osp_crosses"] == pytest.approx(20.0)


def test_fit_poisson_offset_returns_the_offset_without_features() -> None:
    X = pd.DataFrame(index=range(5))
    mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = fit_poisson_offset(X, np.array([1.0, 2, 3, 4, 5]), mu, X, mu, 1.0)
    assert got == pytest.approx(mu)


def test_fit_poisson_offset_recovers_a_multiplicative_effect() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(size=n)
    mu0 = np.full(n, 5.0)
    y = rng.poisson(mu0 * np.exp(0.3 * z)).astype(float)
    X = pd.DataFrame({"z": z})
    pred = fit_poisson_offset(X, y, mu0, X, mu0, alpha=1e-6)
    # predicted mean should rise with z at roughly exp(0.3 * sd)
    hi, lo = pred[z > 1].mean(), pred[z < -1].mean()
    assert hi / lo == pytest.approx(np.exp(0.3 * (z[z > 1].mean() - z[z < -1].mean())),
                                    rel=0.15)
