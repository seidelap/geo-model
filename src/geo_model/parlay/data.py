"""Load NFL game results with closing lines and reshape to team-game rows."""

from __future__ import annotations

import io

import pandas as pd
import requests

from geo_model.parlay.config import ParlayConfig

GAME_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "total",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_qb_id",
    "away_qb_id",
]


def fetch_games_csv(url: str, timeout: float = 60.0) -> pd.DataFrame:
    """Download the nflverse games CSV.

    Args:
        url: Source URL.
        timeout: Request timeout in seconds.

    Returns:
        Raw games table, one row per game.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def clean_games(raw: pd.DataFrame, config: ParlayConfig) -> pd.DataFrame:
    """Filter to completed games with lines and compute market residuals.

    ``spread_line`` follows the nflverse convention: expected ``home - away``
    margin (positive = home favored). ``result`` is realized ``home - away``.

    Args:
        raw: Table from :func:`fetch_games_csv` or the parquet cache.
        config: Season range and game-type filter.

    Returns:
        Cleaned games with ``resid`` (home margin minus spread, so home covers
        when positive) and ``tresid`` (total minus total line, so the over hits
        when positive). Sorted chronologically.
    """
    lo, hi = config.seasons
    df = raw.loc[:, [c for c in GAME_COLUMNS if c in raw.columns]].copy()
    df = df[
        df["result"].notna()
        & df["spread_line"].notna()
        & df["total_line"].notna()
        & df["season"].between(lo, hi)
        & df["game_type"].isin(config.game_types)
    ].copy()
    df["gameday"] = pd.to_datetime(df["gameday"], utc=True)
    df["resid"] = df["result"] - df["spread_line"]
    df["tresid"] = df["total"] - df["total_line"]
    df["week"] = df["week"].astype(int)
    df["season"] = df["season"].astype(int)
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def load_games(config: ParlayConfig, refresh: bool = False) -> pd.DataFrame:
    """Load games, using the parquet cache when present.

    Args:
        config: Data directory and source URL.
        refresh: Re-download even if the cache exists.

    Returns:
        Cleaned games table (see :func:`clean_games`).
    """
    path = config.games_parquet
    if path.exists() and not refresh:
        raw = pd.read_parquet(path)
    else:
        raw = fetch_games_csv(config.games_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.loc[:, [c for c in GAME_COLUMNS if c in raw.columns]].to_parquet(path, index=False)
    return clean_games(raw, config)


def to_team_games(games: pd.DataFrame, config: ParlayConfig | None = None) -> pd.DataFrame:
    """Reshape to one row per (team, game), with next-game lookups.

    Args:
        games: Output of :func:`clean_games`.
        config: Used for the new-QB threshold. Defaults to ``ParlayConfig()``.

    Returns:
        Long table with columns ``team, opp, is_home, resid`` (team perspective:
        positive means the team covered), ``tresid``, ``qb_id``, ``qb_prior_starts``,
        ``new_qb``, and ``next_game_id, next_opp, next_resid, next_tresid,
        next_week`` for the team's next game in the same season (NaN if none).
    """
    config = config or ParlayConfig()
    base_cols = ["game_id", "season", "week", "gameday", "tresid"]
    home = games[base_cols + ["home_team", "away_team", "resid", "home_qb_id"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_qb_id": "qb_id"}
    )
    home["is_home"] = True
    away = games[base_cols + ["away_team", "home_team", "resid", "away_qb_id"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_qb_id": "qb_id"}
    )
    away["is_home"] = False
    away["resid"] = -away["resid"]
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)

    # Prior starts of this QB for this team (all seasons, strictly before this game).
    tg["qb_prior_starts"] = tg.groupby(["team", "qb_id"]).cumcount()
    tg.loc[tg["qb_id"].isna(), "qb_prior_starts"] = -1
    tg["new_qb"] = tg["qb_id"].notna() & (tg["qb_prior_starts"] <= config.new_qb_max_prior_starts)

    grp = tg.groupby(["team", "season"])
    tg["next_game_id"] = grp["game_id"].shift(-1)
    tg["next_opp"] = grp["opp"].shift(-1)
    tg["next_resid"] = grp["resid"].shift(-1)
    tg["next_tresid"] = grp["tresid"].shift(-1)
    tg["next_week"] = grp["week"].shift(-1)
    tg["next_is_home"] = grp["is_home"].shift(-1)
    return tg
