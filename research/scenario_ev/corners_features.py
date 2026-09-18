"""Strictly-prior feature construction for the corners stage (02).

The pure functions here turn the football-data.co.uk match table into (a) a long
team-match frame, (b) shrunk rolling means over *strictly earlier dates*, and (c) the
multiplicative rolling-mean book proxy for corner counts. They take and return plain
frames so they can be unit-tested on small synthetic fixtures.

Why a local ``prior_by_date`` instead of :func:`research.scenario_ev.common.prior_expanding`:
corner and shot counts are missing on about half the odds table, so the *count* of usable
prior matches has to be accumulated per column rather than per row; and the corner proxy
needs league levels that exclude every match of the same day, not just those with a lower
match id.

Shapes are given in bracket notation, e.g. ``[n_matches]`` or ``[2 * n_matches]``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Statistics whose raw prior sums/counts are carried so that the proxy's shrinkage can
#: be re-tuned on the discovery set without rebuilding the table.
PROXY_STATS: tuple[str, ...] = ("corners_for", "corners_against", "goals_for", "goals_against")

#: Statistics for which a separate home and away league level is computed.
HOME_AWAY_STATS: tuple[str, ...] = ("corners_for", "goals_for")

#: Per-team-match statistics accumulated over prior matches.
TEAM_STATS: tuple[str, ...] = (
    "corners_for", "corners_against", "shots_for", "shots_against",
    "target_for", "target_against", "fouls_for", "fouls_against",
    "goals_for", "goals_against", "yellow_for",
)


@dataclass(frozen=True)
class CornerConfig:
    """Configuration for the corners stage.

    Attributes:
        shrink_k: Shrinkage strength (in matches) of a team mean toward the league level,
            used for the model features.
        proxy_k: Shrinkage strength used by the book proxy itself; chosen on discovery by
            the ``proxy`` stage (minimum squared error of the total).
        league_k: Shrinkage strength (in team-matches) of the division-season league mean
            toward the division's all-history mean.
        min_prior: Minimum prior matches with corners recorded, required for both teams.
        window: Length of the short rolling window (matches) used as an extra feature.
        discovery_frac: Fraction of matches placed in the discovery set.
        match_lines: Half-lines simulated for the match total.
        team_lines: Half-lines simulated for a single team's corners.
        seed: Base RNG seed.
        n_jobs: LightGBM threads.
        n_boot: Bootstrap replicates.
    """

    shrink_k: float = 6.0
    proxy_k: float = 40.0
    league_k: float = 100.0
    min_prior: int = 5
    window: int = 6
    discovery_frac: float = 0.60
    match_lines: tuple[float, ...] = (8.5, 9.5, 10.5)
    team_lines: tuple[float, ...] = (4.5, 5.5)
    seed: int = 20260918
    n_jobs: int = 2
    n_boot: int = 2000


def prior_by_date(frame: pd.DataFrame, group_cols: Sequence[str], date_col: str,
                  value_cols: Sequence[str]) -> pd.DataFrame:
    """Sum and count of each value column over rows with a *strictly earlier* date.

    Rows sharing a date inside a group see none of each other, so a match can never
    contribute to its own features and same-day fixtures cannot leak into one another.
    Missing values lower the count instead of entering the sum.

    Args:
        frame: Rows to aggregate.
        group_cols: Grouping keys, e.g. ``["div_season", "team"]``.
        date_col: Chronological key.
        value_cols: Numeric columns to accumulate.

    Returns:
        Frame indexed like ``frame`` with columns ``<col>_psum`` and ``<col>_pcnt``.
    """
    group_cols = list(group_cols)
    value_cols = list(value_cols)
    work = frame[group_cols + [date_col]].copy()
    agg_cols: list[str] = []
    for c in value_cols:
        v = pd.to_numeric(frame[c], errors="coerce")
        work[f"{c}__v"] = v.fillna(0.0).to_numpy()
        work[f"{c}__n"] = v.notna().astype(float).to_numpy()
        agg_cols += [f"{c}__v", f"{c}__n"]
    daily = work.groupby(group_cols + [date_col], sort=True, observed=True)[agg_cols].sum()
    levels = list(range(len(group_cols)))
    cum = daily.groupby(level=levels, sort=False, observed=True).cumsum()
    prior = cum.groupby(level=levels, sort=False, observed=True).shift(1).fillna(0.0)
    keys = pd.MultiIndex.from_frame(work[group_cols + [date_col]])
    out = prior.reindex(keys)
    out.index = frame.index
    ren = {f"{c}__v": f"{c}_psum" for c in value_cols}
    ren.update({f"{c}__n": f"{c}_pcnt" for c in value_cols})
    return out.rename(columns=ren)


def shrink(psum: np.ndarray, pcnt: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    """Shrink a prior-match mean toward a reference mean.

    ``(psum + k * prior) / (pcnt + k)``; with ``pcnt == 0`` this returns ``prior``.

    Args:
        psum: Sum over strictly prior matches [n].
        pcnt: Count of strictly prior matches [n].
        prior: Reference mean to shrink toward [n] or scalar.
        k: Shrinkage strength in units of matches.

    Returns:
        Shrunk mean [n].
    """
    psum = np.asarray(psum, dtype=float)
    pcnt = np.asarray(pcnt, dtype=float)
    pr = np.asarray(prior, dtype=float)
    return (psum + k * pr) / (pcnt + k)


def to_team_match(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode a match table into two team-oriented rows per match.

    Args:
        matches: Match table with the football-data.co.uk columns plus ``match_id``,
            ``div_season``, ``season`` and ``MatchDate``.

    Returns:
        Frame [2 * n_matches] with ``match_id``, ``date``, ``div_season``, ``Division``,
        ``season``, ``team``, ``opp``, ``is_home`` and the :data:`TEAM_STATS` columns.
    """
    base = ["match_id", "MatchDate", "div_season", "Division", "season"]
    pairs = {
        "corners_for": ("HomeCorners", "AwayCorners"),
        "corners_against": ("AwayCorners", "HomeCorners"),
        "shots_for": ("HomeShots", "AwayShots"),
        "shots_against": ("AwayShots", "HomeShots"),
        "target_for": ("HomeTarget", "AwayTarget"),
        "target_against": ("AwayTarget", "HomeTarget"),
        "fouls_for": ("HomeFouls", "AwayFouls"),
        "fouls_against": ("AwayFouls", "HomeFouls"),
        "goals_for": ("FTHome", "FTAway"),
        "goals_against": ("FTAway", "FTHome"),
        "yellow_for": ("HomeYellow", "AwayYellow"),
    }
    frames = []
    for is_home, (tcol, ocol) in ((1, ("HomeTeam", "AwayTeam")), (0, ("AwayTeam", "HomeTeam"))):
        f = matches[base].copy()
        f["team"] = matches[tcol].to_numpy()
        f["opp"] = matches[ocol].to_numpy()
        f["is_home"] = is_home
        for name, (h, a) in pairs.items():
            f[name] = matches[h if is_home == 1 else a].to_numpy()
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"MatchDate": "date"})
    out = out.sort_values(["date", "match_id", "is_home"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def add_prior_features(tm: pd.DataFrame, cfg: CornerConfig) -> pd.DataFrame:
    """Attach strictly-prior league levels, shrunk team means, windows and rest days.

    Team means come from the same division-season; the league reference they shrink
    toward is the division-season mean to date, itself shrunk toward the division's
    all-history mean so that the opening rounds of a season are not priced off three
    matches.

    Args:
        tm: Team-match frame from :func:`to_team_match`.
        cfg: Stage configuration.

    Returns:
        ``tm`` with ``n_prior``, ``pm_<stat>``, ``n_<stat>``, ``lg_<stat>``,
        ``lg_home_<stat>`` / ``lg_away_<stat>`` for :data:`HOME_AWAY_STATS`,
        ``w<window>_<stat>`` and ``rest_days``.
    """
    tm = tm.sort_values(["date", "match_id", "is_home"]).reset_index(drop=True)
    stats = list(TEAM_STATS)
    team_p = prior_by_date(tm, ["div_season", "team"], "date", stats)
    div_p = prior_by_date(tm, ["Division"], "date", stats)
    ds_p = prior_by_date(tm, ["div_season"], "date", stats)

    out = tm.copy()
    out["n_prior"] = team_p["corners_for_pcnt"].to_numpy()
    for s in stats:
        cnt = div_p[f"{s}_pcnt"].to_numpy()
        div_mean = np.where(cnt > 0, div_p[f"{s}_psum"].to_numpy() / np.maximum(cnt, 1.0), np.nan)
        lg = shrink(ds_p[f"{s}_psum"].to_numpy(), ds_p[f"{s}_pcnt"].to_numpy(),
                    div_mean, cfg.league_k)
        out[f"lg_{s}"] = lg
        out[f"n_{s}"] = team_p[f"{s}_pcnt"].to_numpy()
        out[f"pm_{s}"] = shrink(team_p[f"{s}_psum"].to_numpy(),
                                team_p[f"{s}_pcnt"].to_numpy(), lg, cfg.shrink_k)
        if s in PROXY_STATS:
            out[f"ps_{s}"] = team_p[f"{s}_psum"].to_numpy()
            out[f"pc_{s}"] = team_p[f"{s}_pcnt"].to_numpy()

    ha = tm[["div_season", "Division", "date"]].copy()
    ha_cols: list[str] = []
    for base in HOME_AWAY_STATS:
        for side, flag in (("home", 1), ("away", 0)):
            col = f"{side}_{base}"
            ha[col] = np.where(tm["is_home"].to_numpy() == flag, tm[base].to_numpy(), np.nan)
            ha_cols.append(col)
    ds_ha = prior_by_date(ha, ["div_season"], "date", ha_cols)
    div_ha = prior_by_date(ha, ["Division"], "date", ha_cols)
    for col in ha_cols:
        cnt = div_ha[f"{col}_pcnt"].to_numpy()
        div_mean = np.where(cnt > 0, div_ha[f"{col}_psum"].to_numpy() / np.maximum(cnt, 1.0),
                            np.nan)
        out[f"lg_{col}"] = shrink(ds_ha[f"{col}_psum"].to_numpy(), ds_ha[f"{col}_pcnt"].to_numpy(),
                                  div_mean, cfg.league_k / 2.0)

    g = out.groupby(["div_season", "team"], sort=False, observed=True)
    for s in ("corners_for", "corners_against", "shots_for", "target_for"):
        out[f"w{cfg.window}_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(cfg.window, min_periods=2).mean()
        )
    out["rest_days"] = (
        out.groupby(["team"], sort=False, observed=True)["date"].diff().dt.days.astype(float)
    )
    return out


def merge_opponent(tm: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Attach the opposing row's prior features to every team-match row.

    Args:
        tm: Team-match frame with ``match_id`` and ``is_home``.
        cols: Columns to copy from the opposing row, prefixed ``opp_``.

    Returns:
        ``tm`` with ``opp_<col>`` for each requested column.
    """
    cols = list(cols)
    other = tm[["match_id", "is_home"] + cols].copy()
    other["is_home"] = 1 - other["is_home"]
    other = other.rename(columns={c: f"opp_{c}" for c in cols})
    return tm.merge(other, on=["match_id", "is_home"], how="left", validate="one_to_one")


def proxy_lambda(tm: pd.DataFrame, stat: str, against_stat: str, k: float) -> np.ndarray:
    """Multiplicative rolling-mean proxy for one team's count in one match [n_rows].

    ``lambda = league_level(home or away) * attack_ratio * opponent_concession_ratio``,
    where the ratios are the team's shrunk prior rate and the opponent's shrunk prior
    conceded rate, each divided by the league's prior mean per team-match. The shrinkage
    ``k`` is applied here rather than taken from the stored ``pm_`` columns so that the
    discovery set can choose it without rebuilding the table.

    Args:
        tm: Team-match frame after :func:`add_prior_features` and :func:`merge_opponent`,
            carrying ``ps_<stat>``, ``pc_<stat>``, ``opp_ps_<against_stat>``,
            ``opp_pc_<against_stat>``, ``lg_<stat>``, ``lg_home_<stat>``,
            ``lg_away_<stat>`` and ``is_home``.
        stat: The "for" statistic, e.g. ``"corners_for"``.
        against_stat: The opponent's conceding statistic, e.g. ``"corners_against"``.
        k: Shrinkage strength in matches.

    Returns:
        Expected count for the acting team [n_rows].
    """
    lg = tm[f"lg_{stat}"].to_numpy(dtype=float)
    level = np.where(tm["is_home"].to_numpy() == 1,
                     tm[f"lg_home_{stat}"].to_numpy(dtype=float),
                     tm[f"lg_away_{stat}"].to_numpy(dtype=float))
    att = shrink(tm[f"ps_{stat}"].to_numpy(), tm[f"pc_{stat}"].to_numpy(), lg, k)
    con = shrink(tm[f"opp_ps_{against_stat}"].to_numpy(),
                 tm[f"opp_pc_{against_stat}"].to_numpy(), lg, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return level * (att / lg) * (con / lg)


def book_proxy_lambda(tm: pd.DataFrame, k: float) -> np.ndarray:
    """The corner book proxy: :func:`proxy_lambda` for corners at shrinkage ``k``.

    Args:
        tm: Team-match frame (see :func:`proxy_lambda`).
        k: Shrinkage strength in matches, chosen on discovery.

    Returns:
        Expected corners for the acting team [n_rows].
    """
    return proxy_lambda(tm, "corners_for", "corners_against", k)
