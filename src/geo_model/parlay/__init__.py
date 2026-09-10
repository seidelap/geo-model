"""Cross-game correlated-parlay backtest.

Exploratory side experiment (not part of the C1-C6 pipeline). Tests whether
market-error correlations induced by shared games ("explaining away") are large
enough to make non-same-game parlays +EV. See
``docs/experiments/parlay-interaction-effects.md``.
"""

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import load_games, to_team_games
from geo_model.parlay.pairs import build_shared_game_pairs
from geo_model.parlay.kalman import MarketErrorKalman, KalmanParams
from geo_model.parlay.backtest import (
    breakeven_correlation,
    parlay_ev,
    correlation_summary,
    two_sided_parlay_roi,
)

__all__ = [
    "ParlayConfig",
    "load_games",
    "to_team_games",
    "build_shared_game_pairs",
    "MarketErrorKalman",
    "KalmanParams",
    "breakeven_correlation",
    "parlay_ev",
    "correlation_summary",
    "two_sided_parlay_roi",
]
