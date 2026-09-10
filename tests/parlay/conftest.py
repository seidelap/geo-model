"""Shared fixtures: a tiny synthetic schedule with lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_schedule(n_teams: int = 8, n_weeks: int = 7, n_seasons: int = 2, seed: int = 0) -> pd.DataFrame:
    """Random round-robin-ish schedule with placeholder lines and results.

    Returns a raw games table in the nflverse column layout.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for season in range(2000, 2000 + n_seasons):
        for week in range(1, n_weeks + 1):
            order = rng.permutation(teams)
            for k in range(0, n_teams - 1, 2):
                home, away = order[k], order[k + 1]
                spread = float(rng.normal(0, 4))
                result = float(spread + rng.normal(0, 13))
                total_line = float(rng.normal(45, 3))
                total = float(total_line + rng.normal(0, 10))
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away}_{home}",
                        "season": season,
                        "game_type": "REG",
                        "week": week,
                        "gameday": f"{season}-09-{min(week + 5, 28):02d}",
                        "home_team": home,
                        "away_team": away,
                        "home_score": 20.0,
                        "away_score": 20.0 - result,
                        "result": result,
                        "total": total,
                        "spread_line": spread,
                        "total_line": total_line,
                        "home_moneyline": -120.0,
                        "away_moneyline": 100.0,
                        "home_qb_id": f"qb_{home}",
                        "away_qb_id": f"qb_{away}",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_schedule() -> pd.DataFrame:
    return make_schedule()
