"""Construct cross-game leg pairs whose outcomes might be correlated.

The core hypothesis: after teams H and A play each other, the surprise in that
game has two competing explanations (H is better than the market thought, or A
is worse). Conditioning on the shared game makes the market's errors on H and A
positively correlated ("explaining away"), so H's next-game spread residual and
A's next-game spread residual should be positively correlated even though they
are different games priced independently by the book. For totals the shared
observation is a *sum* of the two teams' scoring tendencies, which predicts a
*negative* correlation between the two next-game total residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_shared_game_pairs(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per anchor game whose two teams both have a distinct next game.

    Args:
        team_games: Output of :func:`geo_model.parlay.data.to_team_games`.

    Returns:
        Pairs table with anchor-game context (``game_id, season, week, team_h,
        team_a, anchor_resid`` from the home side, ``abs_surprise``, ``new_qb_any``)
        and the two legs (``h_next_game_id, a_next_game_id, h_next_resid,
        a_next_resid, h_next_tresid, a_next_tresid``). Rows where either team has
        no next game, where the next games coincide, or where the two teams play
        each other next are dropped so that the legs are independent under the
        null of no interaction.
    """
    home = team_games[team_games["is_home"]].set_index("game_id")
    away = team_games[~team_games["is_home"]].set_index("game_id")
    common = home.index.intersection(away.index)
    h = home.loc[common]
    a = away.loc[common]
    pairs = pd.DataFrame(
        {
            "game_id": common,
            "season": h["season"].to_numpy(),
            "week": h["week"].to_numpy(),
            "gameday": h["gameday"].to_numpy(),
            "team_h": h["team"].to_numpy(),
            "team_a": a["team"].to_numpy(),
            "anchor_resid": h["resid"].to_numpy(),
            "anchor_tresid": h["tresid"].to_numpy(),
            "new_qb_h": h["new_qb"].to_numpy(),
            "new_qb_a": a["new_qb"].to_numpy(),
            "h_next_game_id": h["next_game_id"].to_numpy(),
            "a_next_game_id": a["next_game_id"].to_numpy(),
            "h_next_opp": h["next_opp"].to_numpy(),
            "a_next_opp": a["next_opp"].to_numpy(),
            "h_next_week": h["next_week"].to_numpy(),
            "a_next_week": a["next_week"].to_numpy(),
            "h_next_resid": h["next_resid"].to_numpy(),
            "a_next_resid": a["next_resid"].to_numpy(),
            "h_next_tresid": h["next_tresid"].to_numpy(),
            "a_next_tresid": a["next_tresid"].to_numpy(),
        }
    )
    keep = (
        pairs["h_next_game_id"].notna()
        & pairs["a_next_game_id"].notna()
        & (pairs["h_next_game_id"] != pairs["a_next_game_id"])
        & (pairs["h_next_opp"] != pairs["team_a"])
        & (pairs["a_next_opp"] != pairs["team_h"])
        & (pairs["h_next_opp"] != pairs["a_next_opp"])
    )
    pairs = pairs[keep].copy()
    pairs["abs_surprise"] = pairs["anchor_resid"].abs()
    pairs["new_qb_any"] = pairs["new_qb_h"] | pairs["new_qb_a"]
    pairs["same_next_week"] = pairs["h_next_week"] == pairs["a_next_week"]
    return pairs.reset_index(drop=True)


def stratify(pairs: pd.DataFrame, column: str, edges: tuple[float, ...]) -> pd.Series:
    """Assign each pair to a half-open bin ``[edges[i], edges[i+1])``.

    Args:
        pairs: Pairs table.
        column: Numeric column to bin (e.g. ``abs_surprise``).
        edges: Increasing bin edges.

    Returns:
        String labels like ``"[7, 14)"`` aligned with ``pairs``.
    """
    labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(pairs[column], bins=list(edges), right=False, labels=labels).astype(str)


def cover_indicators(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary cover indicators for two residual series, dropping pushes.

    Args:
        x: Leg-1 residuals (positive = cover / over).
        y: Leg-2 residuals.

    Returns:
        ``(cx, cy)`` int arrays restricted to rows where neither leg pushed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x != 0) & (y != 0) & np.isfinite(x) & np.isfinite(y)
    return (x[mask] > 0).astype(int), (y[mask] > 0).astype(int)
