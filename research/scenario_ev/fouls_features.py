"""Strictly-prior feature construction for the fouls stage (04).

The stage deliberately builds on the machinery the earlier stages already wrote rather
than inventing a parallel design:

* the odds-table layer reuses :mod:`research.scenario_ev.corners_features`
  (:func:`~research.scenario_ev.corners_features.to_team_match`,
  :func:`~research.scenario_ev.corners_features.add_prior_features`,
  :func:`~research.scenario_ev.corners_features.prior_by_date` and
  :func:`~research.scenario_ev.corners_features.shrink`), whose ``TEAM_STATS`` already
  accumulate ``fouls_for`` / ``fouls_against``. Only the two pieces the corner proxy did
  not need -- the raw prior sums/counts for fouls (so the shrinkage can be re-tuned on
  discovery) and the home/away league levels for fouls -- are added here;
* the player layer reuses the cached StatsBomb player-match table built by
  :mod:`research.scenario_ev.cards`, adding only the direct-opponent *foul* rate and the
  opponent's prior fouls by pitch side, which the card stage had no use for.

Nothing in ``research/privileged_tracking`` or in the earlier scenario-EV stages is
modified; this module only imports from them.

Shapes are given in bracket notation, e.g. ``[n_rows]`` or ``[2 * n_matches]``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from research.scenario_ev.corners_features import CornerConfig, prior_by_date, shrink

#: The two foul statistics whose raw prior sums and counts are carried, so that the
#: proxy's shrinkage can be chosen on discovery without rebuilding the table.
FOUL_STATS: tuple[str, ...] = ("fouls_for", "fouls_against")

#: Backstop for the very first dates of :func:`flank_foul_priors`, where no strictly
#: earlier team-match exists to average over. Fixed in code rather than read off the
#: table: roughly a third of the ~14.5 fouls a team commits in a match.
FLANK_PRIOR_FALLBACK: float = 4.8


def add_foul_proxy_inputs(tm: pd.DataFrame, cfg: CornerConfig) -> pd.DataFrame:
    """Add the raw prior sums/counts and home/away league levels the foul proxy needs.

    ``add_prior_features`` stores ``ps_``/``pc_`` columns only for the corner stage's
    ``PROXY_STATS`` and home/away league levels only for its ``HOME_AWAY_STATS``. Fouls
    are in neither list, so the same two quantities are computed here with the same
    primitive (:func:`corners_features.prior_by_date`), which accumulates over rows with a
    *strictly earlier date* so that same-day fixtures cannot leak into one another.

    Args:
        tm: Team-match frame after
            :func:`~research.scenario_ev.corners_features.add_prior_features`, carrying
            ``div_season``, ``Division``, ``date``, ``team``, ``is_home``, ``fouls_for``,
            ``fouls_against`` and ``lg_fouls_for``.
        cfg: Corner-stage configuration (its ``league_k`` sets the league shrinkage).

    Returns:
        ``tm`` with ``fps_<stat>`` / ``fpc_<stat>`` for each of :data:`FOUL_STATS` and
        ``lg_home_fouls_for`` / ``lg_away_fouls_for`` [n_rows].
    """
    out = tm.copy()
    team_p = prior_by_date(out, ["div_season", "team"], "date", list(FOUL_STATS))
    for s in FOUL_STATS:
        out[f"fps_{s}"] = team_p[f"{s}_psum"].to_numpy()
        out[f"fpc_{s}"] = team_p[f"{s}_pcnt"].to_numpy()

    ha = out[["div_season", "Division", "date"]].copy()
    cols: list[str] = []
    for side, flag in (("home", 1), ("away", 0)):
        col = f"{side}_fouls_for"
        ha[col] = np.where(out["is_home"].to_numpy() == flag,
                           out["fouls_for"].to_numpy(dtype=float), np.nan)
        cols.append(col)
    ds = prior_by_date(ha, ["div_season"], "date", cols)
    dv = prior_by_date(ha, ["Division"], "date", cols)
    for col in cols:
        cnt = dv[f"{col}_pcnt"].to_numpy()
        div_mean = np.where(cnt > 0, dv[f"{col}_psum"].to_numpy() / np.maximum(cnt, 1.0), np.nan)
        out[f"lg_{col}"] = shrink(ds[f"{col}_psum"].to_numpy(), ds[f"{col}_pcnt"].to_numpy(),
                                  div_mean, cfg.league_k / 2.0)
    return out


def foul_proxy_lambda(tm: pd.DataFrame, k: float, multiplicative: bool = True) -> np.ndarray:
    """The book proxy for one team's fouls in one match [n_rows].

    Multiplicative form (the corner stage's ``proxy_lambda``, re-expressed for fouls)::

        lambda = league_level(home or away) * (team foul rate / league) * (opp conceded / league)

    Additive form::

        lambda = league_level + (team foul rate - league) + (opp conceded - league)

    Both use only strictly-prior matches of the same division-season, and both shrink the
    two team rates toward the division-season league level with strength ``k`` at call
    time, so the discovery set can choose ``k`` without rebuilding the table.

    Args:
        tm: Team-match frame after :func:`add_foul_proxy_inputs` and
            :func:`~research.scenario_ev.corners_features.merge_opponent`, carrying
            ``fps_fouls_for``, ``fpc_fouls_for``, ``opp_fps_fouls_against``,
            ``opp_fpc_fouls_against``, ``lg_fouls_for``, ``lg_home_fouls_for``,
            ``lg_away_fouls_for`` and ``is_home``.
        k: Shrinkage strength in matches.
        multiplicative: Use the multiplicative form (else additive).

    Returns:
        Expected fouls committed by the acting team [n_rows].
    """
    lg = tm["lg_fouls_for"].to_numpy(dtype=float)
    level = np.where(tm["is_home"].to_numpy() == 1,
                     tm["lg_home_fouls_for"].to_numpy(dtype=float),
                     tm["lg_away_fouls_for"].to_numpy(dtype=float))
    att = shrink(tm["fps_fouls_for"].to_numpy(), tm["fpc_fouls_for"].to_numpy(), lg, k)
    con = shrink(tm["opp_fps_fouls_against"].to_numpy(),
                 tm["opp_fpc_fouls_against"].to_numpy(), lg, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        if multiplicative:
            return level * (att / lg) * (con / lg)
        return level + (att - lg) + (con - lg)


def direct_opponent_rates(
    players: pd.DataFrame,
    rate_cols: Sequence[str],
    mirror: dict[str, str],
    line_priority: pd.Series,
    prefix: str = "dopp2_",
) -> pd.DataFrame:
    """Attach a starter's most direct opposing starter's prior rates.

    This is the card stage's direct-opponent match generalised to an arbitrary list of
    prior-rate columns, so the *fouls committed* rate of a dribbler's marker can be
    attached the same way the card stage attached his dribble volume. The candidate is the
    opposing starter on the mirrored flank and the requested line, chosen by most
    strictly-prior minutes; a second line is tried when the first is empty.

    Args:
        players: Player-match frame with ``match_id``, ``team_id``, ``opp_id``,
            ``starter``, ``flank``, ``line``, ``pps_minutes`` and the ``rate_cols``.
        rate_cols: Prior-rate columns to copy from the direct opponent.
        mirror: Map from a flank to the opposing flank, e.g. ``{"left": "right"}``.
        line_priority: Series aligned to ``players`` whose values are tuples of candidate
            opposing lines in priority order, e.g. ``("FWD", "MID")``.
        prefix: Prefix for the produced columns.

    Returns:
        Frame aligned to ``players`` with ``<prefix><col>`` for each rate column plus
        ``<prefix>prior_n`` (the direct opponent's prior appearances) and
        ``<prefix>line_used`` [n_rows].
    """
    rate_cols = list(rate_cols)
    st = players[players["starter"].astype(bool)]
    st = st[st["flank"] != "unknown"]
    cand = st[["match_id", "team_id", "flank", "line", "pps_minutes", "pl_prior_n"]
              + rate_cols].copy()
    cand = cand.sort_values(["match_id", "team_id", "flank", "line", "pps_minutes"],
                            ascending=[True, True, True, True, False], kind="mergesort")
    cand = cand.drop_duplicates(["match_id", "team_id", "flank", "line"], keep="first")
    cand = cand.rename(columns={"team_id": "opp_id", "flank": "m_flank", "line": "m_line",
                                "pl_prior_n": f"{prefix}prior_n",
                                **{c: f"{prefix}{c}" for c in rate_cols}})
    cand = cand.drop(columns=["pps_minutes"])

    out = pd.DataFrame(index=players.index)
    for c in rate_cols:
        out[f"{prefix}{c}"] = np.nan
    out[f"{prefix}prior_n"] = np.nan
    out[f"{prefix}line_used"] = ""
    base = players[["match_id", "opp_id", "flank"]].copy()
    base["m_flank"] = base["flank"].map(lambda f: mirror.get(f, "unknown"))
    take = [f"{prefix}{c}" for c in rate_cols] + [f"{prefix}prior_n"]
    for depth in (0, 1):
        need = out[f"{prefix}prior_n"].isna()
        if not need.any():
            break
        tgt = line_priority.map(lambda t, d=depth: t[d] if len(t) > d else "")
        q = base.assign(m_line=tgt)[need]
        merged = q.merge(cand, on=["match_id", "opp_id", "m_flank", "m_line"], how="left")
        merged.index = q.index
        fill = merged[f"{prefix}prior_n"].notna()
        for c in take:
            out.loc[q.index[fill], c] = merged.loc[fill, c].to_numpy()
        out.loc[q.index[fill], f"{prefix}line_used"] = merged.loc[fill, "m_line"].to_numpy()
    return out


def flank_foul_priors(fouls: pd.DataFrame, team_matches: pd.DataFrame,
                      k: float = 6.0) -> pd.DataFrame:
    """Team-level shrunk prior rate of fouls committed on each pitch side.

    The card stage carried the opponent's prior *dribbles* and *fouls won* by side but not
    its fouls *committed* by side, which is the feature the fouls-won direction of the
    matchup hypothesis needs. Sides are as recorded in ``fouls.parquet``: left / centre /
    right in the fouling team's own attacking frame.

    The shrinkage target is itself strictly prior: it is the mean over *all* team-matches
    on strictly earlier dates, not the mean of the whole table. An earlier version used
    the full-table mean, which was a single scalar per side and so of negligible numerical
    effect, but it was not strictly-prior and the columns produced here sit inside the
    matchup block that carries this stage's only positive finding. Rows with no earlier
    date at all fall back to :data:`FLANK_PRIOR_FALLBACK`, a fixed constant.

    Args:
        fouls: Foul-event table with ``match_id``, ``team_id`` and ``side``.
        team_matches: One row per (match, team) with ``match_id``, ``team_id`` and
            ``date``; defines the chronological order and the universe.
        k: Shrinkage strength in matches toward the strictly-prior global per-match rate.

    Returns:
        Frame with ``match_id``, ``team_id`` and ``tp_fouls_<side>_pm`` for each of
        left / centre / right: the team's shrunk mean fouls on that side per match over
        strictly earlier matches [n_team_matches].
    """
    from research.scenario_ev import common as C

    sides = ("left", "centre", "right")
    f = fouls[fouls["side"].isin(sides)]
    counts = (f.groupby(["match_id", "team_id", "side"]).size().rename("n").reset_index()
              .pivot_table(index=["match_id", "team_id"], columns="side", values="n",
                           fill_value=0).reset_index())
    for s in sides:
        if s not in counts.columns:
            counts[s] = 0.0
    tm = team_matches[["match_id", "team_id", "date"]].merge(
        counts[["match_id", "team_id"] + list(sides)], on=["match_id", "team_id"], how="left")
    for s in sides:
        tm[s] = tm[s].fillna(0.0).astype(float)
    tm = tm.sort_values(["date", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    pri = C.prior_expanding(tm, ["team_id"], list(sides), ["date", "match_id"],
                            count_name="tp_flank_n", prefix="fs_")
    tm = pd.concat([tm, pri], axis=1)
    tm["_all"] = 0
    gl = prior_by_date(tm, ["_all"], "date", list(sides))
    out = tm[["match_id", "team_id", "tp_flank_n"]].copy()
    for s in sides:
        gn = gl[f"{s}_pcnt"].to_numpy(dtype=float)
        glob = np.where(gn > 0, gl[f"{s}_psum"].to_numpy(dtype=float) / np.maximum(gn, 1.0),
                        FLANK_PRIOR_FALLBACK)
        out[f"tp_fouls_{s}_pm"] = C.shrunk_rate(tm[f"fs_{s}"].to_numpy(),
                                                tm["tp_flank_n"].to_numpy(), glob, k)
    return out


def pick_line(mu: np.ndarray, lines: Sequence[float]) -> np.ndarray:
    """Pick, for each row, the ladder line the proxy's mean sits closest to.

    Args:
        mu: Proxy mean count [n].
        lines: Ascending half-integer ladder.

    Returns:
        Chosen line per row [n].
    """
    lines_arr = np.asarray(list(lines), dtype=float)
    mu = np.asarray(mu, dtype=float)
    idx = np.argmin(np.abs(mu[:, None] - lines_arr[None, :]), axis=1)
    return lines_arr[idx]
