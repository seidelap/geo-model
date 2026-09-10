from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.config import american_to_decimal
from geo_model.parlay.data_multisport import (
    _finish,
    _median_line,
    _resign_nba_spreads,
    MultiSportConfig,
    implied_margin_scale,
    implied_probs,
)


def test_implied_probs_removes_vig() -> None:
    fair, raw = implied_probs(np.array([-110.0, -200.0]), np.array([-110.0, 170.0]))
    assert fair[0] == pytest.approx(0.5)
    assert raw[0] == pytest.approx(1 / american_to_decimal(-110))
    assert 0.6 < fair[1] < 0.7


def test_implied_margin_scale_recovers_planted_scale() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, 20000)
    from scipy import stats

    margin = 4.0 * stats.norm.ppf(p) + rng.normal(0, 3, 20000)
    assert implied_margin_scale(margin, p) == pytest.approx(4.0, abs=0.15)


def test_median_line_ignores_in_play_prices() -> None:
    entries = [
        {"currentLine": {"homeOdds": -150}},
        {"currentLine": {"homeOdds": -100000}},
        {"currentLine": {"homeOdds": -160}},
        {"currentLine": {"homeOdds": 50}},
        {"openingLine": {"homeOdds": -140}, "currentLine": {"homeOdds": 900}},  # in-play, big move
        {"openingLine": {"homeOdds": -140}, "currentLine": {"homeOdds": -170}},  # normal move
    ]
    assert _median_line(entries, "homeOdds") == pytest.approx(-160)
    # Median across the +/-100 discontinuity happens in probability space.
    mixed = [{"currentLine": {"awayOdds": o}} for o in (-104, 102, -110, 100)]
    assert -105 < _median_line(mixed, "awayOdds") < 101
    assert np.isnan(_median_line(entries[:1], "homeOdds"))
    assert _median_line([{"currentLine": {"total": 8.5}}, {"currentLine": {"total": 9.0}}], "total") == pytest.approx(8.75)


def _nba_frame(signed: bool) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 400
    p = rng.uniform(0.2, 0.8, n)
    from scipy import stats

    spread = np.round(13.0 * stats.norm.ppf(p) * 2) / 2
    ml_home = np.where(p > 0.5, -100 * p / (1 - p) * 1.05, 100 * (1 - p) / p * 1.05)
    ml_away = np.where(p > 0.5, 100 * p / (1 - p) * 0.95, -100 * (1 - p) / p * 0.95)
    ml_home = np.where(np.abs(ml_home) < 100, np.sign(ml_home) * 100, ml_home)
    ml_away = np.where(np.abs(ml_away) < 100, np.sign(ml_away) * 100, ml_away)
    days = pd.to_datetime("2015-11-01") + pd.to_timedelta(np.arange(n) // 8, unit="D")
    return pd.DataFrame(
        {
            "season": 2015,
            "gameday": days,
            "home_team": [f"H{i % 15}" for i in range(n)],
            "away_team": [f"A{i % 15}" for i in range(n)],
            "home_score": 100.0,
            "away_score": 100.0 - spread,
            "spread_line": spread if signed else np.abs(spread),
            "total_line": 200.0,
            "home_moneyline": ml_home,
            "away_moneyline": ml_away,
        }
    )


def test_resign_nba_spreads_recovers_signs() -> None:
    df = _nba_frame(signed=False)
    truth = _nba_frame(signed=True)["spread_line"].to_numpy()
    out = _resign_nba_spreads(df)
    assert out["spread_resigned"].all()
    assert len(out) >= 0.95 * len(df)
    kept = df.index.isin(out.index) if len(out) == len(df) else None
    assert np.allclose(out["spread_line"].to_numpy(), truth[: len(out)] if kept is None else truth[kept])


def test_resign_nba_spreads_drops_inconsistent_rows() -> None:
    df = _nba_frame(signed=True)
    df.loc[0, "spread_line"] = -df.loc[0, "spread_line"] - 15  # contradicts its moneyline
    out = _resign_nba_spreads(df)
    assert not out["spread_resigned"].any()
    assert len(out) == len(df) - 1


def test_finish_probit_spread_and_ids() -> None:
    df = pd.DataFrame(
        {
            "season": [2020, 2020, 2020],
            "gameday": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "home_team": ["A", "A", "C"],
            "away_team": ["B", "B", "D"],
            "home_score": [3.0, 2.0, 1.0],
            "away_score": [1.0, 4.0, 1.0],
            "total_line": [5.5, 5.5, 6.0],
            "home_moneyline": [-150.0, -150.0, 120.0],
            "away_moneyline": [130.0, 130.0, -140.0],
        }
    )
    out = _finish(df, "nhl", MultiSportConfig(margin_sd={"nhl": 2.0}))
    assert out["game_id"].tolist()[:2] == ["20200101_B_A", "20200101_B_A_1"]  # doubleheader suffix
    assert (out["spread_source"] == "moneyline_probit").all()
    assert out["spread_line"].iloc[0] > 0 > out["spread_line"].iloc[2]
    assert out["resid"].iloc[0] == pytest.approx(2.0 - out["spread_line"].iloc[0])
    assert out["tresid"].iloc[2] == pytest.approx(-4.0)
    assert (out["over_odds"] == -110).all()


def test_load_mlb_archive_repaired_shifts_rows(tmp_path, monkeypatch) -> None:
    import json

    from geo_model.parlay import data_multisport as dm

    # Two true games on one date: (away X @ home Y), (away Z @ home W). The archive stores
    # row i = (home_* = away side of game i, away_* = home side of game i-1).
    rows = [
        {"season": 2015, "date": 20150405.0, "home_team": "Cubs", "away_team": "Yankees", "home_final": 3, "away_final": 9,
         "home_close_ml": 120, "away_close_ml": -150, "close_over_under": 8.0},
        {"season": 2015, "date": 20150405.0, "home_team": "Mets", "away_team": "Cardinals", "home_final": 2, "away_final": 5,
         "home_close_ml": 105, "away_close_ml": -140, "close_over_under": 7.5},
        {"season": 2015, "date": 20150405.0, "home_team": "Rays", "away_team": "Brewers", "home_final": 1, "away_final": 4,
         "home_close_ml": 130, "away_close_ml": -115, "close_over_under": 7.0},
        {"season": 2015, "date": 20150406.0, "home_team": "Reds", "away_team": "Padres", "home_final": 0, "away_final": 0,
         "home_close_ml": 100, "away_close_ml": -110, "close_over_under": 8.5},
    ]
    d = tmp_path / "multisport"
    d.mkdir()
    (d / "mlb_archive_10Y.json").write_text(json.dumps(rows))
    cfg = dm.MultiSportConfig(data_dir=d, margin_sd={"mlb": 4.0})
    out = dm.load_mlb_archive_repaired(cfg, seasons=(2015, 2015))
    # Only rows whose successor is on the same date form a game: (Cubs @ Cardinals), (Mets @ Brewers).
    assert len(out) == 2
    g = out.set_index("away_team")
    assert g.loc["CHC", "home_team"] == "STL" and g.loc["CHC", "away_score"] == 3 and g.loc["CHC", "home_score"] == 5
    assert g.loc["CHC", "away_moneyline"] == 120 and g.loc["CHC", "home_moneyline"] == -140
    assert g.loc["CHC", "total_line"] == 8.0
    assert g.loc["NYM", "home_team"] == "MIL" and g.loc["NYM", "home_score"] == 4
    assert (out["spread_source"] == "moneyline_probit").all()
