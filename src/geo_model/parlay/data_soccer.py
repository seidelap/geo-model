"""Club soccer results with pre-match 1X2, over/under 2.5 and Asian handicap odds.

Source: ``xgabora/Club-Football-Match-Data-2000-2025`` (a consolidation of
football-data.co.uk, MIT licence). Odds are the football-data pre-match
snapshot (typically Friday), not documented closing prices, so market
residuals here are slightly noisier than closing-line residuals.

The market-implied goal margin is fitted as ``a + b * (p_home - p_away)`` on
vig-free 1X2 probabilities, and the implied total as ``a + b * Phi^-1(p_over)``
where the over/under 2.5 market exists. Both fits are OLS on the loaded data
(they are static, sport-level calibrations, not team information).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data_multisport import CLEAN_COLUMNS

SOCCER_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
BIG_FIVE = ("E0", "SP1", "I1", "D1", "F1")


@dataclass(frozen=True)
class SoccerConfig:
    """Location and division filter for the soccer file.

    Attributes:
        data_dir: Directory holding the CSV (``<data_dir>/soccer``).
        url: Source URL.
        divisions: football-data division codes to keep (``E0`` = Premier League).
    """

    data_dir: Path = field(default_factory=lambda: ParlayConfig().data_dir / "soccer")
    url: str = SOCCER_URL
    divisions: tuple[str, ...] = BIG_FIVE

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "xgabora_matches_2000_2026.csv"


def fair_1x2(odd_home: np.ndarray, odd_draw: np.ndarray, odd_away: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vig-free 1X2 probabilities from decimal odds (proportional normalisation)."""
    ih, idr, ia = 1.0 / odd_home, 1.0 / odd_draw, 1.0 / odd_away
    s = ih + idr + ia
    return ih / s, idr / s, ia / s


def load_soccer(config: SoccerConfig | None = None) -> pd.DataFrame:
    """Load matches in the same cleaned schema as the other sports.

    ``season`` is the year the season started (2015 for 2015-16). ``week`` is the
    week number within the (division, season). ``spread_line`` is the implied
    goal margin, ``total_line`` the implied total (NaN where no O/U market),
    ``p_home`` the fair home-win probability, ``home_decimal`` / ``away_decimal``
    the 1X2 prices, and ``division`` / ``p_draw`` are carried as extras.

    Args:
        config: File location and divisions.

    Returns:
        Cleaned matches, chronological.
    """
    config = config or SoccerConfig()
    path = config.csv_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(config.url, timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    raw = pd.read_csv(path, low_memory=False)
    raw = raw[raw["Division"].isin(config.divisions)].copy()
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    df = pd.DataFrame(
        {
            "division": raw["Division"],
            "gameday": pd.to_datetime(raw["MatchDate"], utc=True, errors="coerce"),
            "home_team": raw["HomeTeam"].astype(str),
            "away_team": raw["AwayTeam"].astype(str),
            "home_score": num("FTHome"),
            "away_score": num("FTAway"),
            "odd_home": num("OddHome"),
            "odd_draw": num("OddDraw"),
            "odd_away": num("OddAway"),
            "odd_over": num("Over25"),
            "odd_under": num("Under25"),
            "handicap": num("HandiSize"),
        }
    )
    df = df[df["gameday"].notna() & df["home_score"].notna() & df["away_score"].notna()]
    df = df[(df["odd_home"] > 1) & (df["odd_draw"] > 1) & (df["odd_away"] > 1)]
    df["season"] = np.where(df["gameday"].dt.month >= 7, df["gameday"].dt.year, df["gameday"].dt.year - 1)
    df["result"] = df["home_score"] - df["away_score"]
    df["total"] = df["home_score"] + df["away_score"]
    ph, pdraw, pa = fair_1x2(df["odd_home"].to_numpy(), df["odd_draw"].to_numpy(), df["odd_away"].to_numpy())
    df["p_home"], df["p_draw"], df["p_away"] = ph, pdraw, pa
    x = df["p_home"] - df["p_away"]
    a, b = np.polyfit(x, df["result"], 1)[::-1]
    df["spread_line"] = a + b * x
    df["margin_fit"] = f"{a:.3f} + {b:.3f} * (p_home - p_away)"
    has_ou = (df["odd_over"] > 1) & (df["odd_under"] > 1)
    p_over = (1 / df["odd_over"]) / (1 / df["odd_over"] + 1 / df["odd_under"])
    z = stats.norm.ppf(p_over.clip(1e-4, 1 - 1e-4))
    df["total_line"] = np.nan
    if has_ou.sum() >= 30:
        a2, b2 = np.polyfit(z[has_ou], df.loc[has_ou, "total"], 1)[::-1]
        df.loc[has_ou, "total_line"] = a2 + b2 * z[has_ou]
    df["over_odds"] = np.where(has_ou, df["odd_over"], np.nan)
    df["under_odds"] = np.where(has_ou, df["odd_under"], np.nan)
    df["resid"] = df["result"] - df["spread_line"]
    df["tresid"] = df["total"] - df["total_line"]
    df["home_decimal"] = df["odd_home"]
    df["away_decimal"] = df["odd_away"]
    df["home_moneyline"] = np.nan
    df["away_moneyline"] = np.nan
    df["home_qb_id"] = np.nan
    df["away_qb_id"] = np.nan
    df["game_type"] = "REG"
    df["spread_source"] = "1x2_linear"
    df = df.sort_values(["gameday", "division", "home_team"]).reset_index(drop=True)
    df["game_id"] = df["gameday"].dt.strftime("%Y%m%d") + "_" + df["division"] + "_" + df["away_team"].str.replace(" ", "") + "_" + df["home_team"].str.replace(" ", "")
    start = df.groupby(["division", "season"])["gameday"].transform("min")
    df["week"] = ((df["gameday"] - start).dt.days // 7 + 1).astype(int)
    # Promoted / new-to-division teams: not in this division the previous season.
    prev = df[["division", "season", "home_team"]].drop_duplicates()
    prev_set = set(zip(prev["division"], prev["season"] + 1, prev["home_team"]))
    df["home_new"] = ~pd.Series(list(zip(df["division"], df["season"], df["home_team"]))).isin(prev_set).to_numpy()
    df["away_new"] = ~pd.Series(list(zip(df["division"], df["season"], df["away_team"]))).isin(prev_set).to_numpy()
    first_season = df.groupby("division")["season"].transform("min")
    df.loc[df["season"] == first_season, ["home_new", "away_new"]] = False
    cols = CLEAN_COLUMNS + ["division", "p_draw", "home_new", "away_new", "margin_fit", "handicap"]
    return df[cols]
