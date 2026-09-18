"""Shared helpers for the soccer edge research scripts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "soccer_edge"

# European leagues with an autumn-to-spring season (used for standings logic).
EURO_DIVISIONS = ["E0", "E1", "SP1", "SP2", "I1", "I2", "D1", "D2", "F1", "F2",
                  "N1", "P1", "B1", "T1", "G1", "SC0"]
TOP5 = ["E0", "SP1", "I1", "D1", "F1"]


def data_dir() -> Path:
    """Raw data directory, configurable through ``SOCCER_EDGE_DATA``."""
    return Path(os.environ.get("SOCCER_EDGE_DATA", DEFAULT_DATA_DIR))


def season_of(dates: pd.Series) -> pd.Series:
    """Map dates to a season label: the year the season started (July cut-off)."""
    y = dates.dt.year
    return y.where(dates.dt.month >= 7, y - 1)


def load_odds_matches() -> pd.DataFrame:
    """Load the football-data derived match table (odds, corners, cards, fouls)."""
    df = pd.read_parquet(data_dir() / "odds" / "matches.parquet")
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df["season"] = season_of(df["MatchDate"])
    return df


def no_vig(odds: np.ndarray) -> np.ndarray:
    """Remove bookmaker margin by multiplicative normalisation.

    Args:
        odds: decimal odds ``[n, k]`` for k mutually exclusive outcomes.

    Returns:
        Fair probabilities ``[n, k]`` summing to one per row.
    """
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean with percentile bootstrap confidence interval."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def rolling_prior_mean(df: pd.DataFrame, group: str, value: str, window: int,
                       min_periods: int = 3) -> pd.Series:
    """Strictly-prior rolling mean of ``value`` within ``group`` (rows must be time-sorted)."""
    return (df.groupby(group)[value]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()))
