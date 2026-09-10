"""Configuration for the parlay interaction-effect backtest."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

NFLVERSE_GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds (total return per 1 unit staked).

    Args:
        odds: American odds, e.g. ``-110`` or ``+150``.

    Returns:
        Decimal odds, e.g. ``1.909`` for ``-110``.
    """
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


@dataclass(frozen=True)
class ParlayConfig:
    """Settings for data location, season range, and pricing assumptions.

    Attributes:
        data_dir: Directory for the cached games parquet. Defaults to the
            ``GEO_MODEL_DATA_DIR`` environment variable, else ``data/raw``.
        games_url: Source CSV with closing lines (nflverse schedule/lines file).
        seasons: Inclusive season range to include in the backtest.
        game_types: Game types to keep (``REG`` only by default so that "next
            game" always exists inside a regular schedule).
        leg_odds_american: Price of a single spread/total leg. ``-110`` is the
            standard vig; parlay payouts are the product of per-leg decimals.
        surprise_bins: Edges (in points) used to stratify anchor-game surprise.
        new_qb_max_prior_starts: A team-game counts as a "new QB" game when the
            starter has at most this many prior starts for that team.
    """

    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("GEO_MODEL_DATA_DIR", "data/raw"))
    )
    games_url: str = NFLVERSE_GAMES_URL
    seasons: tuple[int, int] = (1999, 2025)
    game_types: tuple[str, ...] = ("REG",)
    leg_odds_american: float = -110.0
    surprise_bins: tuple[float, ...] = (0.0, 7.0, 14.0, 1000.0)
    new_qb_max_prior_starts: int = 2

    @property
    def leg_decimal(self) -> float:
        """Decimal odds for one leg."""
        return american_to_decimal(self.leg_odds_american)

    @property
    def games_parquet(self) -> Path:
        """Location of the cached games table."""
        return self.data_dir / "nfl_games_lines.parquet"
