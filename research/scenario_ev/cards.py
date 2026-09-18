"""Stage 01: cards -- player card props, the foul -> card mechanism, team card totals.

Run as::

    python -m research.scenario_ev.cards --stage build
    python -m research.scenario_ev.cards --stage a
    python -m research.scenario_ev.cards --stage b
    python -m research.scenario_ev.cards --stage c

Protocol (see the stage report ``reports/01_cards.md``): every scenario definition,
threshold and hyper-parameter is chosen on a chronologically earlier *discovery* split;
the later *confirmation* split is scored once. Every rolling feature is built from
strictly earlier matches and that property is audited, not asserted.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    calibration_table,
    clustered_bootstrap_delta,
    log_loss,
    per_sample_log_loss,
)
from research.scenario_ev import common as C

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardsConfig:
    """Fixed configuration for stage 01.

    Attributes:
        split: Discovery / confirmation split configuration.
        player_k: Shrinkage strength (in prior appearances) for player card rates.
        team_k: Shrinkage strength (in prior matches) for team rates.
        ref_k: Shrinkage strength (in prior appearances) for referee rates.
        min_prior_apps: Minimum strictly-prior appearances for a player row to enter
            the headline player-level analysis.
        seasons: StatsBomb (competition, season) pairs forming the player-level universe.
        n_folds: Group-CV folds (grouped by match) used inside discovery.
        holds: Two-way overrounds reported in the betting simulation.
        lgb_params: LightGBM parameters shared by every classifier.
    """

    split: C.SplitConfig = field(default_factory=C.SplitConfig)
    player_k: float = 20.0
    team_k: float = 8.0
    ref_k: float = 60.0
    min_prior_apps: int = 3
    seasons: tuple[tuple[str, str], ...] = (
        ("Premier League", "2015/2016"),
        ("La Liga", "2015/2016"),
        ("Serie A", "2015/2016"),
        ("Ligue 1", "2015/2016"),
    )
    n_folds: int = 5
    holds: tuple[float, ...] = (0.04, 0.06, 0.08)
    lgb_params: dict[str, object] = field(
        default_factory=lambda: dict(
            objective="binary",
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=80,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=5.0,
            n_estimators=400,
            n_jobs=2,
            verbose=-1,
        )
    )


CFG = CardsConfig()

# ---------------------------------------------------------------------------
# Position geometry (pure helpers, unit tested)
# ---------------------------------------------------------------------------

_LINES = (
    ("Goalkeeper", "GK"),
    ("Back", "DEF"),
    ("Wing Back", "DEF"),
    ("Defensive Midfield", "MID"),
    ("Center Midfield", "MID"),
    ("Midfield", "MID"),
    ("Attacking Midfield", "MID"),
    ("Wing", "FWD"),
    ("Forward", "FWD"),
    ("Striker", "FWD"),
)


def position_flank(position: str | float) -> str:
    """Side of the pitch a StatsBomb starting position sits on.

    Positions are named in the team's own attacking frame, so ``Right Back`` is on the
    right of its own defence.

    Args:
        position: StatsBomb ``start_position`` string.

    Returns:
        One of ``"left"``, ``"right"``, ``"centre"``, or ``"unknown"``.
    """
    if not isinstance(position, str) or not position:
        return "unknown"
    if position.startswith("Left"):
        return "left"
    if position.startswith("Right"):
        return "right"
    if position.startswith("Center") or position == "Goalkeeper" or position == "Secondary Striker":
        return "centre"
    return "unknown"


def position_line(position: str | float) -> str:
    """Vertical line (GK / DEF / MID / FWD) of a StatsBomb starting position.

    Args:
        position: StatsBomb ``start_position`` string.

    Returns:
        One of ``"GK"``, ``"DEF"``, ``"MID"``, ``"FWD"``, ``"unknown"``.
    """
    if not isinstance(position, str) or not position:
        return "unknown"
    if position == "Goalkeeper":
        return "GK"
    if "Wing Back" in position:
        return "DEF"
    if position.endswith("Back"):
        return "DEF"
    if "Attacking Midfield" in position or "Defensive Midfield" in position or "Midfield" in position:
        return "MID"
    if "Wing" in position or "Forward" in position or "Striker" in position:
        return "FWD"
    return "unknown"


def mirror_flank(flank: str) -> str:
    """Physical opposite of a flank.

    Team A attacking east with its right side north means team B, attacking west, has its
    left side north; so A's right flank faces B's left flank.

    Args:
        flank: ``"left"``, ``"right"``, ``"centre"`` or ``"unknown"``.

    Returns:
        The mirrored flank label.
    """
    return {"left": "right", "right": "left", "centre": "centre"}.get(flank, "unknown")


def direct_opponent_lines(position: str | float) -> tuple[str, ...]:
    """Opponent lines that a player of this position is most directly matched against.

    Args:
        position: StatsBomb ``start_position``.

    Returns:
        Tuple of opponent line labels in priority order (may be empty).
    """
    line = position_line(position)
    if line == "DEF":
        return ("FWD", "MID")
    if line == "MID":
        return ("MID", "FWD")
    if line == "FWD":
        return ("DEF", "MID")
    return ()


def is_fullback(position: str | float) -> bool:
    """Whether the starting position is a full-back or wing-back."""
    if not isinstance(position, str):
        return False
    return position.endswith("Back") and position != "Center Back" and "Center Back" not in position


def position_group(position: str | float) -> str:
    """Coarse position group used as the shrinkage backstop for player card rates."""
    line = position_line(position)
    if line == "GK":
        return "GK"
    if is_fullback(position):
        return "FB"
    if line == "DEF":
        return "CB"
    if line == "MID":
        return "MID_DEF" if "Defensive" in str(position) else "MID"
    if line == "FWD":
        return "FWD"
    return "OTHER"


# ---------------------------------------------------------------------------
# Build: StatsBomb player-match table with strictly-prior features
# ---------------------------------------------------------------------------

_PLAYER_ACCUM = [
    "carded", "yellow_any", "minutes", "fouls", "fouls_left", "fouls_right",
    "fouls_centre", "fouls_def_third", "fouls_won", "fouls_won_left", "fouls_won_right",
    "dribbled_past", "tackles", "pressures", "dribbles", "dribbles_left",
    "dribbles_right", "dribbles_centre", "dribbles_final_third", "starter_i",
]

_TEAM_ACCUM = [
    "yellow", "red", "fouls", "fouls_won", "dribbles", "dribbled_past", "corners",
    "shots", "possession", "goals_for", "goals_against", "points",
    "dribbles_left", "dribbles_right", "dribbles_centre", "fouls_won_left",
    "fouls_won_right", "fouls_won_centre", "crosses",
]


def _load_sb_tables() -> dict[str, pd.DataFrame]:
    """Load the StatsBomb aggregate tables with dates parsed.

    Returns:
        Dict with keys ``matches``, ``player_match``, ``team_match``, ``fouls``.
    """
    d = C.sb_processed_dir()
    m = pd.read_parquet(d / "matches.parquet")
    m["date"] = pd.to_datetime(m["date"])
    return {
        "matches": m,
        "player_match": pd.read_parquet(d / "player_match.parquet"),
        "team_match": pd.read_parquet(d / "team_match.parquet"),
        "fouls": pd.read_parquet(d / "fouls.parquet"),
    }


def _team_prior_frame(tables: dict[str, pd.DataFrame], cfg: CardsConfig) -> pd.DataFrame:
    """Team-match table with strictly-prior rolling aggregates.

    Args:
        tables: Output of :func:`_load_sb_tables`.
        cfg: Stage configuration.

    Returns:
        Frame [n_team_matches, ...] keyed by ``(match_id, team_id)`` with ``tp_*``
        prior-rate columns computed from strictly earlier matches of that team.
    """
    tm = tables["team_match"].copy()
    m = tables["matches"][["match_id", "date", "competition", "season", "match_week",
                           "home_team_id", "away_team_id", "home_score", "away_score",
                           "referee_id", "referee"]]
    tm = tm.merge(m, on="match_id", how="left")
    tm["goals_for"] = np.where(tm["home"], tm["home_score"], tm["away_score"]).astype(float)
    tm["goals_against"] = np.where(tm["home"], tm["away_score"], tm["home_score"]).astype(float)
    tm["points"] = np.where(tm["goals_for"] > tm["goals_against"], 3.0,
                            np.where(tm["goals_for"] == tm["goals_against"], 1.0, 0.0))

    pm = tables["player_match"]
    flank_cols = ["dribbles_left", "dribbles_right", "dribbles_centre",
                  "fouls_won_left", "fouls_won_right", "fouls_won_centre"]
    agg = pm.groupby(["match_id", "team_id"], as_index=False)[flank_cols].sum()
    tm = tm.merge(agg, on=["match_id", "team_id"], how="left")

    tm = tm.sort_values(["date", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    pri = C.prior_expanding(tm, ["team_id"], _TEAM_ACCUM, ["date", "match_id"],
                            count_name="tp_n", prefix="tps_")
    tm = pd.concat([tm, pri], axis=1)
    glob = {v: float(tm[v].mean()) for v in _TEAM_ACCUM}
    for v in _TEAM_ACCUM:
        tm["tp_" + v] = C.shrunk_rate(tm["tps_" + v].to_numpy(), tm["tp_n"].to_numpy(),
                                      glob[v], cfg.team_k)
    keep = ["match_id", "team_id", "opp_id", "home", "tp_n"] + ["tp_" + v for v in _TEAM_ACCUM]
    return tm[keep]


def _referee_prior_frame(tables: dict[str, pd.DataFrame], cfg: CardsConfig) -> pd.DataFrame:
    """Per-match referee card tendency from the referee's strictly earlier matches.

    Args:
        tables: Output of :func:`_load_sb_tables`.
        cfg: Stage configuration.

    Returns:
        Frame keyed by ``match_id`` with ``ref_card_rate`` (cards per player-appearance),
        ``ref_cards_pm`` (cards per match), ``ref_foul_rate`` and ``ref_prior_n``.
    """
    pm = tables["player_match"]
    per_match = pm.groupby("match_id", as_index=False).agg(
        cards=("carded_i", "sum"), apps=("carded_i", "size"))
    fouls = tables["team_match"].groupby("match_id", as_index=False)["fouls"].sum()
    fouls = fouls.rename(columns={"fouls": "match_fouls"})
    m = tables["matches"][["match_id", "date", "referee_id"]].merge(per_match, on="match_id")
    m = m.merge(fouls, on="match_id", how="left")
    m = m.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    has_ref = m["referee_id"].notna()
    pri = C.prior_expanding(m, ["referee_id"], ["cards", "apps", "match_fouls"],
                            ["date", "match_id"], count_name="ref_prior_n", prefix="rs_")
    m = pd.concat([m, pri], axis=1)
    glob_rate = float(m["cards"].sum() / m["apps"].sum())
    glob_cards_pm = float(m["cards"].mean())
    glob_fouls_pm = float(m["match_fouls"].mean())
    m["ref_card_rate"] = C.shrunk_rate(m["rs_cards"].to_numpy(), m["rs_apps"].to_numpy(),
                                       glob_rate, cfg.ref_k)
    m["ref_cards_pm"] = C.shrunk_rate(m["rs_cards"].to_numpy(), m["ref_prior_n"].to_numpy(),
                                      glob_cards_pm, cfg.ref_k / 20.0)
    m["ref_fouls_pm"] = C.shrunk_rate(m["rs_match_fouls"].to_numpy(), m["ref_prior_n"].to_numpy(),
                                      glob_fouls_pm, cfg.ref_k / 20.0)
    m.loc[~has_ref, ["ref_card_rate", "ref_cards_pm", "ref_fouls_pm"]] = [
        glob_rate, glob_cards_pm, glob_fouls_pm]
    m.loc[~has_ref, "ref_prior_n"] = 0.0
    m["ref_known"] = has_ref.astype(float)
    return m[["match_id", "ref_card_rate", "ref_cards_pm", "ref_fouls_pm", "ref_prior_n",
              "ref_known"]]


def _direct_opponent_frame(players: pd.DataFrame) -> pd.DataFrame:
    """Attach each starter's most direct opposing starter and that player's prior rates.

    A right back's direct opponent is the opposing left-sided forward (falling back to the
    opposing left-sided midfielder); a right winger's is the opposing left back. The
    chosen candidate is the one with the most strictly-prior minutes.

    Args:
        players: Player-match frame carrying ``match_id``, ``team_id``, ``opp_id``,
            ``starter``, ``flank``, ``line``, ``pp_minutes`` and prior rate columns.

    Returns:
        Frame aligned to ``players`` with ``dopp_*`` columns.
    """
    st = players[players["starter"]].copy()
    st = st[st["flank"] != "unknown"]
    cand = st[["match_id", "team_id", "flank", "line", "pps_minutes",
               "pl_dribbles_p90", "pl_dribbles_flank_p90", "pl_fouls_won_p90",
               "pl_prior_n"]].copy()
    cand = cand.sort_values(["match_id", "team_id", "flank", "line", "pps_minutes"],
                            ascending=[True, True, True, True, False], kind="mergesort")
    cand = cand.drop_duplicates(["match_id", "team_id", "flank", "line"], keep="first")
    cand = cand.rename(columns={
        "team_id": "opp_id", "flank": "m_flank", "line": "m_line",
        "pl_dribbles_p90": "dopp_dribbles_p90",
        "pl_dribbles_flank_p90": "dopp_dribbles_flank_p90",
        "pl_fouls_won_p90": "dopp_fouls_won_p90",
        "pl_prior_n": "dopp_prior_n"}).drop(columns=["pps_minutes"])

    out = pd.DataFrame(index=players.index)
    for c in ["dopp_dribbles_p90", "dopp_dribbles_flank_p90", "dopp_fouls_won_p90",
              "dopp_prior_n"]:
        out[c] = np.nan
    out["dopp_line_used"] = ""
    base = players[["match_id", "opp_id", "flank", "start_position"]].copy()
    base["m_flank"] = base["flank"].map(mirror_flank)
    prio = base["start_position"].map(lambda p: direct_opponent_lines(p))
    for depth in (0, 1):
        need = out["dopp_prior_n"].isna()
        if not need.any():
            break
        tgt = prio.map(lambda t, d=depth: t[d] if len(t) > d else "")
        q = base.assign(m_line=tgt)[need]
        merged = q.merge(cand, on=["match_id", "opp_id", "m_flank", "m_line"], how="left")
        merged.index = q.index
        fill = merged["dopp_prior_n"].notna()
        for c in ["dopp_dribbles_p90", "dopp_dribbles_flank_p90", "dopp_fouls_won_p90",
                  "dopp_prior_n"]:
            out.loc[q.index[fill], c] = merged.loc[fill, c].to_numpy()
        out.loc[q.index[fill], "dopp_line_used"] = merged.loc[fill, "m_line"].to_numpy()
    return out


def build_player_table(cfg: CardsConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Build (and cache) the StatsBomb player-match modelling table.

    Every feature is derived from strictly earlier matches of the player, team, opponent
    or referee; the only same-match inputs are the announced-lineup facts (starter,
    starting position, competition, home/away, opponent identity) that a prop market also
    has when it prices.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        Player-match frame with targets ``y_carded`` / ``y_yellow``, feature columns and
        the ``split`` label.
    """
    path = C.processed_dir() / "cards_player_match.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    tables = _load_sb_tables()
    pm = tables["player_match"].copy()
    for c in ["yellow", "second_yellow", "red", "yellow_bad_behaviour"]:
        pm[c] = pm[c].fillna(0.0)
    pm["carded_i"] = ((pm["yellow"] > 0) | (pm["second_yellow"] > 0) | (pm["red"] > 0)).astype(float)
    pm["yellow_any"] = ((pm["yellow"] > 0) | (pm["second_yellow"] > 0)).astype(float)
    pm["carded"] = pm["carded_i"]
    pm["starter_i"] = pm["starter"].astype(float)
    tables["player_match"] = pm

    m = tables["matches"]
    pl = pm.merge(m[["match_id", "date", "competition", "season", "match_week",
                     "home_team_id", "away_team_id", "referee_id"]], on="match_id", how="left")
    pl["home"] = (pl["team_id"] == pl["home_team_id"]).astype(float)
    pl["opp_id"] = np.where(pl["home"] > 0.5, pl["away_team_id"], pl["home_team_id"])
    pl["flank"] = pl["start_position"].map(position_flank)
    pl["line"] = pl["start_position"].map(position_line)
    pl["pos_group"] = pl["start_position"].map(position_group)
    pl["is_fb"] = pl["start_position"].map(is_fullback).astype(float)

    pl = pl.sort_values(["date", "match_id", "team_id", "player_id"],
                        kind="mergesort").reset_index(drop=True)
    pri = C.prior_expanding(pl, ["player_id"], _PLAYER_ACCUM, ["date", "match_id"],
                            count_name="pl_prior_n", prefix="pps_")
    pl = pd.concat([pl, pri], axis=1)

    # position-group backstop rate from strictly earlier appearances only
    glob_card = float(pl["carded"].mean())
    gm = pl.groupby(["pos_group", "date", "match_id"], as_index=False).agg(
        gcards=("carded", "sum"), gapps=("carded", "size"))
    gm = gm.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    gpri = C.prior_expanding(gm, ["pos_group"], ["gcards", "gapps"], ["date", "match_id"],
                             count_name="grp_n", prefix="grps_")
    gm = pd.concat([gm, gpri], axis=1)
    gm["grp_card_rate"] = C.shrunk_rate(gm["grps_gcards"].to_numpy(),
                                        gm["grps_gapps"].to_numpy(), glob_card, 200.0)
    pl = pl.merge(gm[["pos_group", "match_id", "grp_card_rate"]],
                  on=["pos_group", "match_id"], how="left")

    pl["pl_card_rate"] = C.shrunk_rate(pl["pps_carded"].to_numpy(), pl["pl_prior_n"].to_numpy(),
                                       pl["grp_card_rate"].to_numpy(), cfg.player_k)
    pl["pl_yellow_rate"] = C.shrunk_rate(pl["pps_yellow_any"].to_numpy(),
                                         pl["pl_prior_n"].to_numpy(), glob_card, cfg.player_k)
    prior_min = pl["pps_minutes"].to_numpy()
    pl["pl_card_p90"] = C.shrunk_rate(pl["pps_carded"].to_numpy() * 90.0, prior_min,
                                      pl["grp_card_rate"].to_numpy(), cfg.player_k * 70.0)
    mean_min = float(pl["minutes"].mean())
    pl["pl_minutes_exp"] = C.shrunk_rate(prior_min, pl["pl_prior_n"].to_numpy(),
                                         mean_min, 3.0)
    for src, name in [("fouls", "pl_fouls_p90"), ("fouls_left", "pl_fouls_left_p90"),
                      ("fouls_right", "pl_fouls_right_p90"),
                      ("fouls_centre", "pl_fouls_centre_p90"),
                      ("fouls_def_third", "pl_fouls_def3_p90"),
                      ("fouls_won", "pl_fouls_won_p90"),
                      ("dribbled_past", "pl_dribbled_past_p90"),
                      ("tackles", "pl_tackles_p90"), ("pressures", "pl_pressures_p90"),
                      ("dribbles", "pl_dribbles_p90")]:
        glob = float(pl[src].sum() * 90.0 / max(pl["minutes"].sum(), 1.0))
        pl[name] = C.shrunk_rate(pl["pps_" + src].to_numpy() * 90.0, prior_min, glob,
                                 cfg.player_k * 70.0)
    flank_src = np.where(pl["flank"].to_numpy() == "left", pl["pps_dribbles_left"],
                         np.where(pl["flank"].to_numpy() == "right", pl["pps_dribbles_right"],
                                  pl["pps_dribbles_centre"]))
    pl["pl_dribbles_flank_p90"] = C.shrunk_rate(flank_src * 90.0, prior_min,
                                                float(pl["dribbles"].mean()), cfg.player_k * 70.0)
    pl["pl_starter_rate"] = C.shrunk_rate(pl["pps_starter_i"].to_numpy(),
                                          pl["pl_prior_n"].to_numpy(), 0.7, 3.0)

    tp = _team_prior_frame(tables, cfg)
    own = tp.drop(columns=["opp_id", "home"]).rename(
        columns={c: "own_" + c for c in tp.columns if c.startswith("tp_")})
    opp = tp.drop(columns=["opp_id", "home"]).rename(
        columns={"team_id": "opp_id", **{c: "opp_" + c for c in tp.columns if c.startswith("tp_")}})
    pl = pl.merge(own, on=["match_id", "team_id"], how="left")
    pl = pl.merge(opp, on=["match_id", "opp_id"], how="left")

    rf = _referee_prior_frame(tables, cfg)
    pl = pl.merge(rf, on="match_id", how="left")

    # opponent flank volume facing this player's flank
    mflank = pl["flank"].map(mirror_flank).to_numpy()
    pl["opp_flank_dribbles_pm"] = np.where(
        mflank == "left", pl["opp_tp_dribbles_left"],
        np.where(mflank == "right", pl["opp_tp_dribbles_right"], pl["opp_tp_dribbles_centre"]))
    pl["opp_flank_fouls_won_pm"] = np.where(
        mflank == "left", pl["opp_tp_fouls_won_left"],
        np.where(mflank == "right", pl["opp_tp_fouls_won_right"], pl["opp_tp_fouls_won_centre"]))

    dopp = _direct_opponent_frame(pl)
    pl = pd.concat([pl, dopp], axis=1)

    pl["x_ref_flank"] = pl["ref_card_rate"] * pl["opp_flank_dribbles_pm"]
    pl["x_ref_playerfoul"] = pl["ref_card_rate"] * pl["pl_fouls_p90"]
    pl["x_foul_flank"] = pl["pl_fouls_p90"] * pl["opp_flank_dribbles_pm"]
    pl["x_foul_dopp"] = pl["pl_fouls_p90"] * pl["dopp_dribbles_flank_p90"]

    pl["y_carded"] = pl["carded_i"]
    pl["y_yellow"] = pl["yellow_any"]
    pl.to_parquet(path, index=False)
    return pl


# ---------------------------------------------------------------------------
# Feature sets and model plumbing
# ---------------------------------------------------------------------------

_CAT_COLS = ["competition", "start_position", "pos_group", "flank", "line"]

_P0_INPUTS = [
    "pl_card_rate", "pl_card_p90", "pl_prior_n", "pl_minutes_exp", "pl_starter_rate",
    "grp_card_rate", "own_tp_yellow", "own_tp_red", "opp_tp_yellow", "ref_card_rate",
    "ref_cards_pm", "ref_fouls_pm", "ref_prior_n", "ref_known", "p0_logit",
]

_P1_EXTRA = [
    "pl_fouls_p90", "pl_fouls_left_p90", "pl_fouls_right_p90", "pl_fouls_centre_p90",
    "pl_fouls_def3_p90", "pl_fouls_won_p90", "pl_dribbled_past_p90", "pl_tackles_p90",
    "pl_pressures_p90", "pl_dribbles_p90", "pl_dribbles_flank_p90",
    "starter_i", "home", "match_week", "is_fb",
    "own_tp_fouls", "own_tp_fouls_won", "own_tp_dribbles", "own_tp_dribbled_past",
    "own_tp_possession", "own_tp_points", "own_tp_shots", "own_tp_corners", "own_tp_n",
    "opp_tp_fouls", "opp_tp_fouls_won", "opp_tp_dribbles", "opp_tp_dribbled_past",
    "opp_tp_possession", "opp_tp_points", "opp_tp_shots", "opp_tp_corners", "opp_tp_n",
] + _CAT_COLS

_P2_EXTRA = [
    "opp_flank_dribbles_pm", "opp_flank_fouls_won_pm", "dopp_dribbles_p90",
    "dopp_dribbles_flank_p90", "dopp_fouls_won_p90", "dopp_prior_n",
    "x_ref_flank", "x_ref_playerfoul", "x_foul_flank", "x_foul_dopp",
]

_P3_EXTRA = [
    "own_st_nearest_opp_dist", "own_st_n_opp_within_5", "own_st_block_depth",
    "own_st_counter_on", "opp_st_nearest_opp_dist", "opp_st_n_opp_within_5",
    "opp_st_block_depth", "opp_st_counter_on",
]

FEATURE_SETS: dict[str, list[str]] = {
    "P0": [],
    "P1": _P0_INPUTS + _P1_EXTRA,
    "P2": _P0_INPUTS + _P1_EXTRA + _P2_EXTRA,
    "P3": _P0_INPUTS + _P1_EXTRA + _P2_EXTRA + _P3_EXTRA,
}


def _as_model_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Select model columns and coerce the categorical ones to pandas ``category``.

    Args:
        df: Source frame.
        cols: Feature columns to select.

    Returns:
        Frame [n, len(cols)] ready for LightGBM.
    """
    X = df[list(cols)].copy()
    for c in _CAT_COLS:
        if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze categorical columns to a single global vocabulary.

    LightGBM reads category *codes*, so train and score frames must share the same
    category list. Converting once on the full universe and slicing afterwards keeps
    the codes consistent across the discovery / confirmation split.

    Args:
        df: Universe frame.

    Returns:
        The same frame with ``_CAT_COLS`` cast to ``category`` in place.
    """
    for c in _CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _fit_predict_lgb(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
                     params: dict[str, object], seed: int) -> np.ndarray:
    """Fit a LightGBM binary classifier and predict probabilities.

    Args:
        X_tr: Training design matrix [n_tr, d].
        y_tr: Training targets [n_tr].
        X_te: Scoring design matrix [n_te, d].
        params: LightGBM parameters.
        seed: Random seed.

    Returns:
        Predicted probabilities [n_te].
    """
    import lightgbm as lgb

    p = dict(params)
    p.update(random_state=seed, bagging_seed=seed, feature_fraction_seed=seed)
    cat = [c for c in _CAT_COLS if c in X_tr.columns] or "auto"
    model = lgb.LGBMClassifier(**p)
    model.fit(X_tr, y_tr, categorical_feature=cat)
    out = model.predict_proba(X_te)
    return out[:, 1]


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Log-loss, Brier and AUC for a binary prediction.

    Args:
        y: Binary outcomes [n].
        p: Predicted probabilities [n].

    Returns:
        Dict of metric name to value (AUC is NaN when a class is absent).
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    auc = float("nan")
    if 0 < y.sum() < len(y):
        auc = float(roc_auc_score(y, p))
    return {"n": int(len(y)), "base_rate": float(y.mean()), "log_loss": log_loss(y, p),
            "brier": brier(y, p), "auc": auc}


def _cv_and_confirm(df: pd.DataFrame, cols: Sequence[str], target: str,
                    disc: np.ndarray, cfg: CardsConfig, seed: int,
                    iters: Sequence[int] = (30, 60, 100, 200, 350)) -> dict[str, object]:
    """Discovery-CV out-of-fold predictions, iteration selection and a confirmation fit.

    The boosting-iteration count is chosen on the discovery out-of-fold log-loss (never
    on confirmation) and the resulting probabilities are recalibrated by a logistic map
    fitted on the discovery out-of-fold predictions -- the same treatment the book proxy
    gets, so that the comparison is about information and not about calibration.

    Args:
        df: Universe frame with ``match_id`` and the target.
        cols: Feature columns.
        target: Target column name.
        disc: Boolean discovery mask [n].
        cfg: Stage configuration.
        seed: Model / fold seed.
        iters: Candidate boosting-iteration counts.

    Returns:
        Dict with ``raw`` [n], ``cal`` [n] (both filled out-of-fold on discovery and from
        the discovery-fitted model on confirmation), ``best_iter`` and ``platt``.
    """
    import lightgbm as lgb

    from research.privileged_tracking.common.splits import group_kfold

    X = _as_model_frame(df, cols)
    y = df[target].to_numpy(dtype=float)
    d_idx = np.flatnonzero(disc)
    n_max = int(max(iters))
    params = dict(cfg.lgb_params)
    params.update(n_estimators=n_max, random_state=seed, bagging_seed=seed,
                  feature_fraction_seed=seed)
    cat = [c for c in _CAT_COLS if c in X.columns] or "auto"
    oof = {k: np.full(len(df), np.nan) for k in iters}
    for tr, te in group_kfold(df["match_id"].to_numpy()[d_idx], n_splits=cfg.n_folds, seed=seed):
        tr_g, te_g = d_idx[tr], d_idx[te]
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr_g], y[tr_g], categorical_feature=cat)
        for k in iters:
            oof[k][te_g] = model.predict_proba(X.iloc[te_g], num_iteration=int(k))[:, 1]
    scores = {k: log_loss(y[disc], oof[k][disc]) for k in iters}
    best_iter = int(min(scores, key=scores.get))

    model = lgb.LGBMClassifier(**params)
    model.fit(X.iloc[d_idx], y[d_idx], categorical_feature=cat)
    conf = model.predict_proba(X.iloc[np.flatnonzero(~disc)], num_iteration=best_iter)[:, 1]
    raw = oof[best_iter].copy()
    raw[~disc] = conf
    ab = C.platt_fit(raw[disc], y[disc])
    cal = C.platt_apply(raw, ab)
    return {"raw": raw, "cal": cal, "best_iter": best_iter, "platt": ab,
            "iter_scores": scores}


def build_proxy(df: pd.DataFrame, fit_mask: np.ndarray, target: str = "y_carded"
                ) -> tuple[np.ndarray, dict[str, float]]:
    """The book-proxy card probability P0 and its discovery-fitted recalibration.

    ``lambda = card_rate_per90 * (expected_minutes / 90) * team_factor * referee_factor``
    and ``p = 1 - exp(-lambda)``, then a single logistic recalibration fitted on the
    discovery rows only, so the strawman is a *calibrated* strawman.

    Args:
        df: Player-match frame with the prior-rate columns.
        fit_mask: Boolean mask [n] of rows the recalibration may be fitted on.
        target: Target column used for the recalibration.

    Returns:
        Tuple of calibrated probabilities [n] and a dict of fitted constants.
    """
    team_base = float(df.loc[fit_mask, "own_tp_yellow"].mean())
    ref_base = float(df.loc[fit_mask, "ref_card_rate"].mean())
    lam = (df["pl_card_p90"].to_numpy()
           * np.clip(df["pl_minutes_exp"].to_numpy(), 5.0, 95.0) / 90.0
           * (df["own_tp_yellow"].to_numpy() / team_base)
           * (df["ref_card_rate"].to_numpy() / ref_base))
    p_raw = np.clip(1.0 - np.exp(-np.clip(lam, 1e-6, 5.0)), 1e-5, 1 - 1e-5)
    ab = C.platt_fit(p_raw[fit_mask], df.loc[fit_mask, target].to_numpy())
    p_cal = C.platt_apply(p_raw, ab)
    return p_cal, {"team_base": team_base, "ref_base": ref_base, "platt_a": ab[0],
                   "platt_b": ab[1], "raw_log_loss_fit": log_loss(
                       df.loc[fit_mask, target].to_numpy(), p_raw[fit_mask])}


# ---------------------------------------------------------------------------
# P3: imputed defensive-state aggregates rolled forward
# ---------------------------------------------------------------------------

_STATE_TARGETS = ("nearest_opp_dist", "n_opp_within_5", "block_depth", "counter_on")


def build_team_state_priors(cfg: CardsConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Team-match defensive-state aggregates from the programme's imputations, rolled forward.

    The programme's state targets are defined on the *opponents of the acting team*
    (``block_depth`` is ``120 - mean x`` of the acting team's outfield opponents), so a
    team's own defensive shape is the mean over events in which the other team is in
    possession. Values come from the ``E2`` event-only student (out-of-fold on the 360
    matches, applied predictions elsewhere), never from the 360 truth.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        Frame keyed by ``(match_id, team_id)`` with ``st_<target>`` (this match, used only
        to build the rolling feature) and ``stp_<target>`` (strictly-prior shrunk mean),
        plus ``st_prior_n``.
    """
    path = C.processed_dir() / "cards_team_state.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    sdir = C.soccer_processed_dir()
    parts: list[pd.DataFrame] = []

    cols = ["match_id", "team_id", "f_is_possession_team"] + [
        f"imp_{t}__E2" for t in _STATE_TARGETS]
    a = pd.read_parquet(sdir / "imputed_no360.parquet", columns=cols)
    a = a[a["f_is_possession_team"].astype(bool)]
    a = a.rename(columns={f"imp_{t}__E2": t for t in _STATE_TARGETS})
    parts.append(a[["match_id", "team_id"] + list(_STATE_TARGETS)])
    del a

    cols_b = ["match_id", "event_id", "f_is_possession_team"] + [
        f"{t}__E2" for t in _STATE_TARGETS]
    b = pd.read_parquet(sdir / "imputed_oof.parquet", columns=cols_b)
    b = b[b["f_is_possession_team"].astype(bool)]
    key = pd.read_parquet(sdir / "events360.parquet", columns=["match_id", "event_id", "team_id"])
    b = b.merge(key, on=["match_id", "event_id"], how="left")
    del key
    b = b.rename(columns={f"{t}__E2": t for t in _STATE_TARGETS})
    parts.append(b[["match_id", "team_id"] + list(_STATE_TARGETS)])
    del b

    ev = pd.concat(parts, ignore_index=True)
    del parts
    ev = ev[ev["team_id"].notna()]
    agg = ev.groupby(["match_id", "team_id"], as_index=False)[list(_STATE_TARGETS)].mean()
    agg["team_id"] = agg["team_id"].astype("int64")
    del ev

    tables = _load_sb_tables()
    tm = tables["team_match"][["match_id", "team_id", "opp_id"]].merge(
        tables["matches"][["match_id", "date"]], on="match_id", how="left")
    # the state measured while the OPPONENT attacks describes this team's own defence
    own = agg.rename(columns={"team_id": "opp_id", **{t: "st_" + t for t in _STATE_TARGETS}})
    tm = tm.merge(own, on=["match_id", "opp_id"], how="left")
    tm = tm.sort_values(["date", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    vals = ["st_" + t for t in _STATE_TARGETS]
    tm["st_ok"] = tm[vals[0]].notna().astype(float)
    filled = tm[vals].fillna(0.0)
    pri = C.prior_expanding(tm.assign(**{v: filled[v] for v in vals}), ["team_id"],
                            vals + ["st_ok"], ["date", "match_id"],
                            count_name="st_rows", prefix="sts_")
    tm = pd.concat([tm, pri], axis=1)
    tm["st_prior_n"] = tm["sts_st_ok"]
    for t in _STATE_TARGETS:
        glob = float(tm["st_" + t].mean())
        tm["stp_" + t] = C.shrunk_rate(tm["sts_st_" + t].to_numpy(),
                                       tm["st_prior_n"].to_numpy(), glob, 2.0)
    keep = ["match_id", "team_id", "st_prior_n"] + ["stp_" + t for t in _STATE_TARGETS]
    out = tm[keep]
    out.to_parquet(path, index=False)
    return out


# ---------------------------------------------------------------------------
# Stage (a): player-match card model
# ---------------------------------------------------------------------------


def audit_player_table(pl: pd.DataFrame, cfg: CardsConfig = CFG, n_check: int = 120,
                       seed: int = 0) -> pd.DataFrame:
    """Brute-force re-derivation of the headline prior features for a sample of rows.

    For each sampled player-match this recomputes the player, team and referee priors
    directly from the raw StatsBomb tables by filtering on ``(date, match_id)`` strictly
    earlier than the row, and compares with the vectorised values in ``pl``. It is the
    check behind the claim that no feature sees the match it predicts.

    Args:
        pl: Output of :func:`build_player_table`.
        cfg: Stage configuration.
        n_check: Rows to verify.
        seed: RNG seed for the sample.

    Returns:
        One row per audited feature with the maximum absolute discrepancy.
    """
    tables = _load_sb_tables()
    pm = tables["player_match"].copy()
    for c in ("yellow", "second_yellow", "red"):
        pm[c] = pm[c].fillna(0.0)
    pm["carded_i"] = ((pm["yellow"] > 0) | (pm["second_yellow"] > 0)
                      | (pm["red"] > 0)).astype(float)
    md = tables["matches"][["match_id", "date", "referee_id"]]
    pm = pm.merge(md, on="match_id", how="left")
    tm = tables["team_match"].merge(md, on="match_id", how="left")
    ref = pm.groupby(["match_id", "date", "referee_id"], dropna=False, as_index=False).agg(
        cards=("carded_i", "sum"), apps=("carded_i", "size"))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pl), size=min(n_check, len(pl)), replace=False)
    errs = {"pl_card_rate": 0.0, "own_tp_yellow": 0.0, "ref_card_rate": 0.0}
    glob_team_yellow = float(tm["yellow"].mean())
    glob_ref = float(ref["cards"].sum() / ref["apps"].sum())
    for i in idx:
        row = pl.iloc[i]
        key = (row["date"], row["match_id"])
        earlier = lambda d: (d["date"] < key[0]) | ((d["date"] == key[0])
                                                    & (d["match_id"] < key[1]))
        p_prior = pm[(pm["player_id"] == row["player_id"]) & earlier(pm)]
        exp = C.shrunk_rate(np.array([p_prior["carded_i"].sum()]),
                            np.array([float(len(p_prior))]),
                            float(row["grp_card_rate"]), cfg.player_k)[0]
        errs["pl_card_rate"] = max(errs["pl_card_rate"], abs(exp - float(row["pl_card_rate"])))
        t_prior = tm[(tm["team_id"] == row["team_id"]) & earlier(tm)]
        exp_t = C.shrunk_rate(np.array([t_prior["yellow"].sum()]),
                              np.array([float(len(t_prior))]), glob_team_yellow,
                              cfg.team_k)[0]
        errs["own_tp_yellow"] = max(errs["own_tp_yellow"],
                                    abs(exp_t - float(row["own_tp_yellow"])))
        if pd.notna(row["referee_id"]):
            r_prior = ref[(ref["referee_id"] == row["referee_id"]) & earlier(ref)]
            exp_r = C.shrunk_rate(np.array([r_prior["cards"].sum()]),
                                  np.array([r_prior["apps"].sum()]), glob_ref, cfg.ref_k)[0]
            errs["ref_card_rate"] = max(errs["ref_card_rate"],
                                        abs(exp_r - float(row["ref_card_rate"])))
    return pd.DataFrame([{"feature": k, "max_abs_error": v, "n_checked": int(len(idx))}
                         for k, v in errs.items()])


def player_universe(pl: pd.DataFrame, cfg: CardsConfig = CFG) -> pd.DataFrame:
    """Restrict the player-match table to the modelling universe and label the split.

    The universe is the four StatsBomb club seasons (2015/16 Premier League, La Liga,
    Serie A, Ligue 1), outfield players only, with at least ``cfg.min_prior_apps``
    strictly-earlier appearances -- a prop market needs a history to price off.

    Args:
        pl: Output of :func:`build_player_table`.
        cfg: Stage configuration.

    Returns:
        Filtered frame with a ``split`` column.
    """
    keys = {(c, s) for c, s in cfg.seasons}
    m = pd.Series(list(zip(pl["competition"], pl["season"])), index=pl.index).isin(keys)
    uni = pl[m & (pl["line"] != "GK") & (pl["minutes"] > 0)].copy()
    uni["split"] = C.chronological_split(uni["match_id"], uni["date"], cfg.split.discovery_frac)
    uni = uni[uni["pl_prior_n"] >= cfg.min_prior_apps].reset_index(drop=True)
    return uni


def _scenario_mask(df: pd.DataFrame, flank_cut: float, ref_cut: float) -> np.ndarray:
    """The pre-registered card scenario: foul-prone flank duel under a card-happy referee.

    Args:
        df: Player-match frame.
        flank_cut: Discovery top-tercile threshold on opponent flank dribble volume.
        ref_cut: Discovery top-tercile threshold on the referee card rate.

    Returns:
        Boolean mask [n].
    """
    return (
        (df["is_fb"].to_numpy() > 0.5)
        & (df["opp_flank_dribbles_pm"].to_numpy() >= flank_cut)
        & (df["ref_card_rate"].to_numpy() >= ref_cut)
    )


def _gate(df: pd.DataFrame, cols: Sequence[str], target: str, scen: np.ndarray,
          cfg: CardsConfig, label: str, seed: int,
          params: dict[str, object] | None = None) -> list[dict[str, object]]:
    """Generalist-vs-specialist gate, in discovery CV and once on confirmation.

    Args:
        df: Universe frame with ``split`` and ``match_id``.
        cols: Feature columns.
        target: Target column.
        scen: Boolean scenario mask [n] aligned to ``df``.
        cfg: Stage configuration.
        label: Scenario name recorded in the output rows.
        seed: Model / fold seed.
        params: LightGBM parameters (defaults to ``cfg.lgb_params``).

    Returns:
        List of result rows (discovery CV, confirmation, and the off-scenario reverse).
    """
    params = dict(cfg.lgb_params if params is None else params)
    from research.privileged_tracking.common.splits import group_kfold

    rows: list[dict[str, object]] = []
    disc = (df["split"] == C.DISCOVERY).to_numpy()
    conf = ~disc
    X = _as_model_frame(df, cols)
    y = df[target].to_numpy(dtype=float)
    mid = df["match_id"].to_numpy()

    # --- discovery CV, scored on held-out scenario rows -----------------------------
    d_idx = np.flatnonzero(disc)
    gen = np.full(len(df), np.nan)
    spe = np.full(len(df), np.nan)
    for tr, te in group_kfold(mid[d_idx], n_splits=cfg.n_folds, seed=seed):
        tr_g, te_g = d_idx[tr], d_idx[te]
        tr_s = tr_g[scen[tr_g]]
        te_s = te_g[scen[te_g]]
        if len(te_s) == 0 or len(tr_s) < 50:
            continue
        gen[te_s] = _fit_predict_lgb(X.iloc[tr_g], y[tr_g], X.iloc[te_s], params, seed)
        spe[te_s] = _fit_predict_lgb(X.iloc[tr_s], y[tr_s], X.iloc[te_s], params, seed)
    ok = np.isfinite(gen) & np.isfinite(spe)
    rows.append(_gate_row("discovery_cv", label, y[ok], gen[ok], spe[ok], mid[ok], cfg))

    # --- confirmation, fitted once on discovery --------------------------------------
    tr_g = np.flatnonzero(disc)
    tr_s = tr_g[scen[tr_g]]
    te_s = np.flatnonzero(conf & scen)
    if len(te_s) > 0 and len(tr_s) >= 50:
        g = _fit_predict_lgb(X.iloc[tr_g], y[tr_g], X.iloc[te_s], params, seed)
        s = _fit_predict_lgb(X.iloc[tr_s], y[tr_s], X.iloc[te_s], params, seed)
        rows.append(_gate_row("confirmation", label, y[te_s], g, s, mid[te_s], cfg))
        # reverse: how the specialist does off-scenario on confirmation
        te_o = np.flatnonzero(conf & ~scen)
        g2 = _fit_predict_lgb(X.iloc[tr_g], y[tr_g], X.iloc[te_o], params, seed)
        s2 = _fit_predict_lgb(X.iloc[tr_s], y[tr_s], X.iloc[te_o], params, seed)
        rows.append(_gate_row("confirmation_off_scenario", label, y[te_o], g2, s2,
                              mid[te_o], cfg))
    return rows


def _gate_row(where: str, label: str, y: np.ndarray, gen: np.ndarray, spe: np.ndarray,
              groups: np.ndarray, cfg: CardsConfig) -> dict[str, object]:
    """One row of the gate table: specialist minus generalist log-loss with a CI."""
    lg = per_sample_log_loss(y, gen)
    ls = per_sample_log_loss(y, spe)
    d, lo, hi = clustered_bootstrap_delta(lg, ls, groups, cfg.split.n_boot, cfg.split.seed)
    return {
        "where": where, "scenario": label, "n": int(len(y)), "base_rate": float(y.mean()),
        "generalist_log_loss": float(lg.mean()), "specialist_log_loss": float(ls.mean()),
        "delta_specialist_minus_generalist": d, "ci_lo": lo, "ci_hi": hi,
        "generalist_brier": brier(y, gen), "specialist_brier": brier(y, spe),
    }


def _choose_threshold(y: np.ndarray, p_model: np.ndarray, p_book: np.ndarray,
                      groups: np.ndarray, hold: float, cfg: CardsConfig,
                      grid: Sequence[float], min_bets: int = 100) -> float:
    """Pick the EV threshold on discovery: best ROI subject to a minimum bet count.

    Args:
        y: Binary outcomes [n].
        p_model: Model probabilities [n].
        p_book: Book-proxy fair probabilities [n].
        groups: Match ids [n].
        hold: Two-way overround.
        cfg: Stage configuration.
        grid: Candidate thresholds.
        min_bets: Minimum number of bets a threshold must place.

    Returns:
        The chosen threshold (the smallest candidate if none qualifies).
    """
    best_t, best_roi = float(grid[0]), -np.inf
    for t in grid:
        r = C.simulate_two_way(y, p_model, p_book,
                               groups, C.BetSimConfig(hold=hold, edge_threshold=t,
                                                      n_boot=200, seed=cfg.split.seed))
        if r.n_bets >= min_bets and np.isfinite(r.roi) and r.roi > best_roi:
            best_roi, best_t = r.roi, float(t)
    return best_t


def _book_ladder(y: np.ndarray, p_model: np.ndarray, p_proxy: np.ndarray,
                 disc: np.ndarray, groups: np.ndarray, cfg: CardsConfig,
                 grid: Sequence[float], market: str,
                 lams: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """How sharp may the book be before the modelled edge disappears?

    The synthetic book is interpolated on the logit scale between the rolling-mean proxy
    (``lam = 0``) and our own model (``lam = 1``); each rung reports how much that book
    beats the proxy in nats, which is directly comparable to the goals-market measurement
    of how much a real bookmaker beats the same proxy.

    Args:
        y: Binary outcome [n].
        p_model: Our model's probability [n].
        p_proxy: The rolling-mean proxy's probability [n].
        disc: Discovery mask [n].
        groups: Cluster ids [n].
        cfg: Stage configuration.
        grid: Candidate EV thresholds, chosen on discovery.
        market: Market label recorded in the output.
        lams: Book-strength rungs.

    Returns:
        One row per (lam, hold, split).
    """
    rows = []
    for lam in lams:
        p_book = C.logit_blend(p_proxy, p_model, float(lam))
        gain = (log_loss(y[~disc], p_proxy[~disc]) - log_loss(y[~disc], p_book[~disc]))
        for hold in cfg.holds:
            t = _choose_threshold(y[disc], p_model[disc], p_book[disc], groups[disc],
                                  hold, cfg, grid, min_bets=100)
            for where, mask in (("discovery", disc), ("confirmation", ~disc)):
                r = C.simulate_two_way(y[mask], p_model[mask], p_book[mask], groups[mask],
                                       C.BetSimConfig(hold=hold, edge_threshold=t,
                                                      n_boot=cfg.split.n_boot,
                                                      seed=cfg.split.seed))
                rows.append(r.as_row(market=market, lam=float(lam),
                                     book_gain_vs_proxy_nats=gain, hold=hold,
                                     threshold=t, where=where))
    return pd.DataFrame(rows)


def stage_a(cfg: CardsConfig = CFG) -> dict[str, pd.DataFrame]:
    """Player-match card model: P0 proxy, P1 event-only, P2 matchup, P3 imputed state.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables, also written to the reports directory.
    """
    pl = build_player_table(cfg)
    st = build_team_state_priors(cfg)
    own_st = st.rename(columns={**{"stp_" + t: "own_st_" + t for t in _STATE_TARGETS},
                                "st_prior_n": "own_st_prior_n"})
    opp_st = st.rename(columns={"team_id": "opp_id", "st_prior_n": "opp_st_prior_n",
                                **{"stp_" + t: "opp_st_" + t for t in _STATE_TARGETS}})
    pl = pl.merge(own_st, on=["match_id", "team_id"], how="left")
    pl = pl.merge(opp_st, on=["match_id", "opp_id"], how="left")

    uni = prepare_categoricals(player_universe(pl, cfg))
    disc = (uni["split"] == C.DISCOVERY).to_numpy()
    mid = uni["match_id"].to_numpy()
    out: dict[str, pd.DataFrame] = {}

    out["a_setup"] = pd.DataFrame([{
        "player_appearances_all": int(len(pl)),
        "universe_rows": int(len(uni)),
        "universe_matches": int(uni["match_id"].nunique()),
        "n_discovery": int(disc.sum()), "n_confirmation": int((~disc).sum()),
        "matches_discovery": int(uni.loc[disc, "match_id"].nunique()),
        "matches_confirmation": int(uni.loc[~disc, "match_id"].nunique()),
        "date_min": str(pd.Timestamp(uni["date"].min()).date()),
        "date_cut": str(pd.Timestamp(uni.loc[~disc, "date"].min()).date()),
        "date_max": str(pd.Timestamp(uni["date"].max()).date()),
        "carded_rate_discovery": float(uni.loc[disc, "y_carded"].mean()),
        "carded_rate_confirmation": float(uni.loc[~disc, "y_carded"].mean()),
        "median_prior_apps": float(uni["pl_prior_n"].median()),
        "n_players": int(uni["player_id"].nunique()),
        "n_referees_known": int(uni.loc[uni["ref_known"] > 0, "match_id"].nunique()),
        "frac_rows_with_known_referee": float((uni["ref_known"] > 0).mean()),
        "frac_rows_with_direct_opponent": float(uni["dopp_prior_n"].notna().mean()),
    }])

    audit_rows = []
    for gcols, vcol, pcol, ncol in [(["player_id"], "carded", "pps_carded", "pl_prior_n")]:
        a = C.audit_strictly_prior(pl, gcols, vcol, ["date", "match_id"], pcol, ncol,
                                   n_check=120, seed=cfg.split.seed)
        a.update(group="|".join(gcols), value=vcol)
        audit_rows.append(a)
    out["a_audit"] = pd.DataFrame(audit_rows)
    out["a_audit_features"] = audit_player_table(pl, cfg, n_check=120, seed=cfg.split.seed)

    metrics: list[dict[str, object]] = []
    preds: dict[str, dict[str, np.ndarray]] = {}
    for target in ("y_carded", "y_yellow"):
        p0_all, const = build_proxy(uni, disc, target)
        uni["p0_logit"] = np.log(np.clip(p0_all, 1e-6, 1 - 1e-6) /
                                 (1 - np.clip(p0_all, 1e-6, 1 - 1e-6)))
        y = uni[target].to_numpy(dtype=float)
        preds[target] = {"P0": p0_all}
        fit_info = []
        for name in ("P1", "P2", "P3"):
            res = _cv_and_confirm(uni, FEATURE_SETS[name], target, disc, cfg, cfg.split.seed)
            preds[target][name] = res["cal"]
            preds[target][name + "_raw"] = res["raw"]
            fit_info.append({"target": target, "model": name, "best_iter": res["best_iter"],
                             "platt_a": res["platt"][0], "platt_b": res["platt"][1],
                             **{f"cv_ll_{k}": v for k, v in res["iter_scores"].items()}})
        const_p = float(y[disc].mean())
        preds[target]["CLIMA"] = np.full(len(uni), const_p)
        for where, mask in (("discovery_oof", disc), ("confirmation", ~disc)):
            base = per_sample_log_loss(y[mask], preds[target]["P0"][mask])
            p0_ll = log_loss(y[mask], preds[target]["P0"][mask])
            for name in ("CLIMA", "P0", "P1", "P2", "P3", "P1_raw", "P2_raw", "P3_raw"):
                p = preds[target][name][mask]
                row = {"target": target, "where": where, "model": name}
                row.update(_binary_metrics(y[mask], p))
                ls = per_sample_log_loss(y[mask], p)
                d, lo, hi = clustered_bootstrap_delta(base, ls, mid[mask],
                                                      cfg.split.n_boot, cfg.split.seed)
                row.update(delta_vs_P0=d, ci_lo=lo, ci_hi=hi,
                           skill_vs_P0=1.0 - row["log_loss"] / p0_ll)
                metrics.append(row)
        if target == "y_carded":
            out["a_proxy_constants"] = pd.DataFrame([const])
            out["a_fit_info"] = pd.DataFrame(fit_info)
    out["a_metrics"] = pd.DataFrame(metrics)

    target = "y_carded"
    y = uni[target].to_numpy(dtype=float)
    p0_all, _ = build_proxy(uni, disc, target)
    uni["p0_logit"] = np.log(np.clip(p0_all, 1e-6, 1 - 1e-6) / (1 - np.clip(p0_all, 1e-6, 1 - 1e-6)))

    cal_rows = []
    for where, mask in (("discovery_oof", disc), ("confirmation", ~disc)):
        for name in ("P0", "P2"):
            for r in calibration_table(y[mask], preds[target][name][mask], n_bins=10):
                r.update(where=where, model=name)
                cal_rows.append(r)
    out["a_calibration"] = pd.DataFrame(cal_rows)

    # --- P3 like-for-like subset -----------------------------------------------------
    sub = (uni["own_st_prior_n"].fillna(0) > 0) & (uni["opp_st_prior_n"].fillna(0) > 0)
    sub = sub.to_numpy()
    sub_rows = []
    for where, mask in (("discovery_oof", disc & sub), ("confirmation", (~disc) & sub)):
        base = per_sample_log_loss(y[mask], preds[target]["P2"][mask])
        for name in ("P0", "P2", "P3"):
            p = preds[target][name][mask]
            row = {"where": where, "model": name, "subset": "state_available"}
            row.update(_binary_metrics(y[mask], p))
            d, lo, hi = clustered_bootstrap_delta(base, per_sample_log_loss(y[mask], p),
                                                  mid[mask], cfg.split.n_boot, cfg.split.seed)
            row.update(delta_vs_P2=d, ci_lo=lo, ci_hi=hi)
            sub_rows.append(row)
    out["a_p3_subset"] = pd.DataFrame(sub_rows)
    stt = build_team_state_priors(cfg)
    out["a_state_coverage"] = pd.DataFrame([{
        "team_matches_total": int(len(stt)),
        "team_matches_with_prior_state": int((stt["st_prior_n"] > 0).sum()),
        "frac_team_matches_with_prior_state": float((stt["st_prior_n"] > 0).mean()),
        "player_rows": int(len(uni)),
        "rows_with_own_state_prior": int((uni["own_st_prior_n"].fillna(0) > 0).sum()),
        "rows_with_both_state_priors": int(sub.sum()),
        "frac_both": float(sub.mean()),
        "median_own_state_prior_n": float(uni["own_st_prior_n"].median()),
    }])

    # --- the gate --------------------------------------------------------------------
    fb = uni["is_fb"].to_numpy() > 0.5
    d_fb = disc & fb
    flank_cut = float(np.quantile(uni.loc[d_fb, "opp_flank_dribbles_pm"], 2 / 3))
    ref_cut = float(np.quantile(uni.loc[d_fb, "ref_card_rate"], 2 / 3))
    scenarios = {
        "fullback_only": fb,
        "fullback_x_flank": fb & (uni["opp_flank_dribbles_pm"].to_numpy() >= flank_cut),
        "fullback_x_flank_x_ref": _scenario_mask(uni, flank_cut, ref_cut),
    }
    best_iter = int(out["a_fit_info"].query("model == 'P2'")["best_iter"].iloc[0])
    gate_params = {**cfg.lgb_params, "n_estimators": best_iter}
    gate_rows: list[dict[str, object]] = []
    for label, scen in scenarios.items():
        gate_rows += _gate(uni, FEATURE_SETS["P2"], target, scen, cfg, label,
                           cfg.split.seed, gate_params)
    out["a_gate"] = pd.DataFrame(gate_rows)
    out["a_scenario_defs"] = pd.DataFrame([{
        "flank_tercile_cut": flank_cut, "ref_tercile_cut": ref_cut,
        **{f"n_discovery_{k}": int((v & disc).sum()) for k, v in scenarios.items()},
        **{f"n_confirmation_{k}": int((v & ~disc).sum()) for k, v in scenarios.items()},
        **{f"base_rate_conf_{k}": float(y[(v & ~disc)].mean()) for k, v in scenarios.items()},
    }])

    # --- refit noise floor -----------------------------------------------------------
    refit = []
    for seed in (cfg.split.seed, cfg.split.seed + 1, cfg.split.seed + 2):
        for name in ("P1", "P2"):
            r = _cv_and_confirm(uni, FEATURE_SETS[name], target, disc, cfg, seed)
            refit.append({"model": name, "seed": seed, "best_iter": r["best_iter"],
                          "confirmation_log_loss": log_loss(y[~disc], r["cal"][~disc])})
    out["a_refit"] = pd.DataFrame(refit)

    # --- betting simulation (player "to be carded", a 0.5 line) ----------------------
    grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    bet_rows: list[dict[str, object]] = []
    books = {"P0_proxy": p0_all, "P1_smart_book": preds[target]["P1"]}
    for book_name, p_book in books.items():
        for name in ("P1", "P2"):
            if book_name == "P1_smart_book" and name == "P1":
                continue
            for hold in cfg.holds:
                t = _choose_threshold(y[disc], preds[target][name][disc], p_book[disc],
                                      mid[disc], hold, cfg, grid)
                for where, mask in (("discovery", disc), ("confirmation", ~disc)):
                    r = C.simulate_two_way(y[mask], preds[target][name][mask], p_book[mask],
                                           mid[mask],
                                           C.BetSimConfig(hold=hold, edge_threshold=t,
                                                          n_boot=cfg.split.n_boot,
                                                          seed=cfg.split.seed))
                    bet_rows.append(r.as_row(book=book_name, model=name, hold=hold,
                                             threshold=t, where=where,
                                             market="player_carded_0.5"))
    out["a_bets"] = pd.DataFrame(bet_rows)

    out["a_book_ladder"] = _book_ladder(y, preds[target]["P2"], p0_all, disc, mid, cfg,
                                        grid, "player_carded_0.5")

    dis = C.disagreement_table(y[~disc], preds[target]["P2"][~disc], p0_all[~disc], n_bins=5)
    dis["where"] = "confirmation"
    dis["model"] = "P2"
    out["a_disagreement"] = dis

    for k, v in out.items():
        C.write_table(v, f"01_cards_{k}")
    return out




# ---------------------------------------------------------------------------
# Stage (b): foul-level mechanism -- given a foul, what predicts a card?
# ---------------------------------------------------------------------------


def build_goal_times(force: bool = False) -> pd.DataFrame:
    """Reconstruct goal minutes per match from the programme's event tables.

    The event tables carry a running ``f_score_for`` / ``f_score_against`` per event, so a
    goal is a minute at which the running total steps up. The recorded minute is that of
    the first event carrying the new score, which is the goal itself or the restart within
    a few seconds of it.

    Args:
        force: Rebuild even if the cache exists.

    Returns:
        Frame with ``match_id``, ``minute`` and ``is_home`` (one row per goal).
    """
    path = C.processed_dir() / "cards_goal_times.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    sdir = C.soccer_processed_dir()
    cols = ["match_id", "f_minute", "f_home", "f_score_for", "f_score_against"]
    parts = [pd.read_parquet(sdir / f, columns=cols)
             for f in ("events_no360.parquet", "events360.parquet")]
    ev = pd.concat(parts, ignore_index=True)
    del parts
    home = ev["f_home"].astype(bool).to_numpy()
    ev["home_goals"] = np.where(home, ev["f_score_for"], ev["f_score_against"])
    ev["away_goals"] = np.where(home, ev["f_score_against"], ev["f_score_for"])
    ev = ev[["match_id", "f_minute", "home_goals", "away_goals"]].dropna()
    ev = ev.sort_values(["match_id", "f_minute"], kind="mergesort")
    g = ev.groupby("match_id", sort=False)
    rows = []
    for side, col in (("home", "home_goals"), ("away", "away_goals")):
        cm = g[col].cummax()
        prev = cm.groupby(ev["match_id"], sort=False).shift(1).fillna(0.0)
        step = cm - prev
        hit = step > 0
        sub = ev.loc[hit, ["match_id", "f_minute"]].copy()
        sub["n_goals"] = step[hit].to_numpy()
        sub["is_home"] = side == "home"
        rows.append(sub)
    goals = pd.concat(rows, ignore_index=True)
    goals = goals.loc[goals.index.repeat(goals["n_goals"].astype(int))]
    goals = goals.rename(columns={"f_minute": "minute"})[["match_id", "minute", "is_home"]]
    goals = goals.sort_values(["match_id", "minute"], kind="mergesort").reset_index(drop=True)
    goals.to_parquet(path, index=False)
    return goals


def score_state_at(goals: pd.DataFrame, match_ids: np.ndarray, minutes: np.ndarray,
                   is_home: np.ndarray) -> np.ndarray:
    """Goal difference for the acting team strictly before a given minute.

    Args:
        goals: Output of :func:`build_goal_times`.
        match_ids: Match id per query row [n].
        minutes: Minute per query row [n].
        is_home: Whether the acting team is the home team [n].

    Returns:
        Signed goal difference (acting team minus opponent) [n].
    """
    out = np.zeros(len(match_ids), dtype=float)
    by_match: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for mid, sub in goals.groupby("match_id", sort=False):
        h = np.sort(sub.loc[sub["is_home"], "minute"].to_numpy())
        a = np.sort(sub.loc[~sub["is_home"], "minute"].to_numpy())
        by_match[int(mid)] = (h, a)
    for i, (mid, mn, hm) in enumerate(zip(match_ids, minutes, is_home)):
        h, a = by_match.get(int(mid), (np.empty(0), np.empty(0)))
        nh = int(np.searchsorted(h, mn, side="left"))
        na = int(np.searchsorted(a, mn, side="left"))
        out[i] = (nh - na) if hm else (na - nh)
    return out


def build_foul_table(cfg: CardsConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Foul-level table: one row per foul with pre-foul context and the card outcome.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        Frame with ``y_card``, ``y_yellow``, ``y_red`` and context features.
    """
    path = C.processed_dir() / "cards_foul_level.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    tables = _load_sb_tables()
    f = tables["fouls"].copy()
    m = tables["matches"][["match_id", "date", "competition", "season", "match_week",
                           "home_team_id", "away_team_id", "referee_id"]]
    f = f.merge(m, on="match_id", how="left")
    f["is_home"] = (f["team_id"] == f["home_team_id"])
    f["y_card"] = f["card"].notna().astype(float)
    f["y_yellow"] = (f["card"] == "Yellow Card").astype(float)
    f["y_red"] = f["card"].isin(["Red Card", "Second Yellow"]).astype(float)

    f = f.sort_values(["match_id", "period", "minute"], kind="mergesort").reset_index(drop=True)
    # strictly-prior within-match running counts
    f["one"] = 1.0
    within = C.prior_expanding(f, ["match_id", "team_id"], ["one", "y_card"],
                              ["period", "minute"], count_name="team_fouls_so_far",
                              prefix="wm_")
    f["team_fouls_so_far"] = within["team_fouls_so_far"]
    f["team_cards_so_far"] = within["wm_y_card"]
    within_p = C.prior_expanding(f, ["match_id", "player_id"], ["one", "y_card"],
                                 ["period", "minute"], count_name="player_fouls_so_far",
                                 prefix="wp_")
    f["player_fouls_so_far"] = within_p["player_fouls_so_far"]
    f["player_cards_so_far"] = within_p["wp_y_card"]

    goals = build_goal_times()
    f["score_diff"] = score_state_at(goals, f["match_id"].to_numpy(), f["minute"].to_numpy(),
                                     f["is_home"].to_numpy())
    f["leading"] = (f["score_diff"] > 0).astype(float)
    f["trailing"] = (f["score_diff"] < 0).astype(float)

    # geometry, in the fouling team's attacking frame: own goal at x = 0
    f["dist_own_goal"] = np.sqrt(f["x"] ** 2 + (f["y"] - 40.0) ** 2)
    f["dist_opp_goal"] = np.sqrt((120.0 - f["x"]) ** 2 + (f["y"] - 40.0) ** 2)
    f["own_third"] = (f["x"] < 40).astype(float)
    f["final_third"] = (f["x"] > 80).astype(float)
    f["zone"] = np.where(f["x"] < 40, "own_third",
                         np.where(f["x"] > 80, "final_third", "middle_third"))
    f["near_own_box"] = ((f["x"] < 30) & (f["y"].between(14, 66))).astype(float)
    f["fouler_line"] = f["position"].map(position_line)
    f["fouler_flank"] = f["position"].map(position_flank)
    f["fouled_line"] = f["fouled_position"].map(position_line)
    f["fouled_flank"] = f["fouled_position"].map(position_flank)
    f["dribbling"] = f["fouled_dribbling"].astype(float)
    f["handball"] = (f["foul_type"] == "Handball").astype(float)
    f["dangerous_play"] = (f["foul_type"] == "Dangerous Play").astype(float)
    f["penalty_i"] = f["penalty"].astype(float)
    f["minute_c"] = f["minute"].clip(0, 100)

    # referee tendency: cards per foul from strictly earlier matches
    per_match = f.groupby("match_id", as_index=False).agg(
        cards=("y_card", "sum"), fouls=("y_card", "size"))
    per_match = per_match.merge(m[["match_id", "date", "referee_id"]], on="match_id")
    per_match = per_match.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)
    rp = C.prior_expanding(per_match, ["referee_id"], ["cards", "fouls"], ["date", "match_id"],
                           count_name="ref_n", prefix="rs_")
    per_match = pd.concat([per_match, rp], axis=1)
    glob = float(per_match["cards"].sum() / per_match["fouls"].sum())
    per_match["ref_card_per_foul"] = C.shrunk_rate(per_match["rs_cards"].to_numpy(),
                                                   per_match["rs_fouls"].to_numpy(),
                                                   glob, 300.0)
    per_match.loc[per_match["referee_id"].isna(), "ref_card_per_foul"] = glob
    f = f.merge(per_match[["match_id", "ref_card_per_foul", "ref_n"]], on="match_id", how="left")

    # fouler's season-to-date card and foul propensity
    pl = build_player_table(cfg)
    f = f.merge(pl[["match_id", "player_id", "pl_card_rate", "pl_fouls_p90", "pl_prior_n"]],
                on=["match_id", "player_id"], how="left")
    f["split"] = C.chronological_split(f["match_id"], f["date"], cfg.split.discovery_frac)
    f.to_parquet(path, index=False)
    return f


_FOUL_NUM = [
    "x", "y", "dist_own_goal", "dist_opp_goal", "minute_c", "score_diff",
    "team_fouls_so_far", "team_cards_so_far", "player_fouls_so_far",
    "player_cards_so_far", "ref_card_per_foul", "pl_card_rate", "pl_fouls_p90",
]
_FOUL_BIN = ["dribbling", "handball", "dangerous_play", "penalty_i", "near_own_box",
             "is_home_i", "leading", "trailing", "own_third", "final_third"]
_FOUL_CAT = ["side", "fouler_line", "fouled_line", "competition"]


def _foul_design(df: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the foul-level logistic model.

    Continuous columns are standardised so the reported odds ratios are per standard
    deviation; binaries and one-hot categoricals are left on their natural scale.

    Args:
        df: Foul-level frame.

    Returns:
        Numeric design matrix [n, d] with an interaction column
        ``dribbling_x_own_third``.
    """
    X = pd.DataFrame(index=df.index)
    for c in _FOUL_NUM:
        v = df[c].astype(float)
        v = v.fillna(v.median())
        sd = v.std(ddof=0)
        X[c] = (v - v.mean()) / (sd if sd > 0 else 1.0)
    for c in _FOUL_BIN:
        X[c] = df[c].astype(float).fillna(0.0)
    X["dribbling_x_own_third"] = X["dribbling"] * X["own_third"]
    X["dribbling_x_near_box"] = X["dribbling"] * X["near_own_box"]
    for c in _FOUL_CAT:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True).astype(float)
        X = pd.concat([X, d], axis=1)
    return X


def _logit_odds_ratios(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                       n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Logistic-regression odds ratios with a match-clustered bootstrap interval.

    Args:
        X: Design matrix [n, d].
        y: Binary outcome [n].
        groups: Match id per row [n].
        n_boot: Bootstrap replicates (whole matches resampled).
        seed: RNG seed.

    Returns:
        Frame with ``term``, ``odds_ratio``, ``ci_lo``, ``ci_hi``.
    """
    from sklearn.linear_model import LogisticRegression

    def fit(Xa: np.ndarray, ya: np.ndarray) -> np.ndarray:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        lr.fit(Xa, ya)
        return lr.coef_[0]

    Xv = X.to_numpy(dtype=float)
    point = fit(Xv, y)
    codes, uniq = pd.factorize(pd.Series(groups))
    idx_by_group = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, Xv.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([idx_by_group[i] for i in pick])
        boots[b] = fit(Xv[rows], y[rows])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"term": X.columns, "coef": point,
                         "odds_ratio": np.exp(point), "ci_lo": np.exp(lo),
                         "ci_hi": np.exp(hi)})


def _rate_contrast(df: pd.DataFrame, by: Sequence[str], target: str, cfg: CardsConfig
                   ) -> pd.DataFrame:
    """Observed card rate in cells of ``by`` with a match-clustered interval.

    Args:
        df: Foul-level frame.
        by: Grouping columns.
        target: Binary outcome column.
        cfg: Stage configuration.

    Returns:
        One row per cell with ``n``, ``rate``, ``ci_lo``, ``ci_hi``.
    """
    rows = []
    for key, sub in df.groupby(list(by), sort=True):
        m, lo, hi = C.clustered_bootstrap_mean(sub[target].to_numpy(dtype=float),
                                               sub["match_id"].to_numpy(),
                                               cfg.split.n_boot, cfg.split.seed)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(n=int(len(sub)), rate=m, ci_lo=lo, ci_hi=hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def stage_b(cfg: CardsConfig = CFG) -> dict[str, pd.DataFrame]:
    """Given that a foul happened, what predicts a card? (the mechanism behind stage a).

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables, also written to the reports directory.
    """
    f = build_foul_table(cfg)
    f["is_home_i"] = f["is_home"].astype(float)
    keys = {(c, s) for c, s in cfg.seasons}
    in_uni = pd.Series(list(zip(f["competition"], f["season"])), index=f.index).isin(keys)
    uni = f[in_uni].copy().reset_index(drop=True)
    uni["split"] = C.chronological_split(uni["match_id"], uni["date"], cfg.split.discovery_frac)
    ood = f[~in_uni].copy().reset_index(drop=True)
    disc = (uni["split"] == C.DISCOVERY).to_numpy()
    out: dict[str, pd.DataFrame] = {}

    out["b_counts"] = pd.DataFrame([{
        "fouls_total": int(len(f)), "fouls_universe": int(len(uni)),
        "fouls_out_of_domain": int(len(ood)),
        "card_rate_universe": float(uni["y_card"].mean()),
        "yellow_rate": float(uni["y_yellow"].mean()), "red_rate": float(uni["y_red"].mean()),
        "n_discovery": int(disc.sum()), "n_confirmation": int((~disc).sum()),
        "dribbling_share": float(uni["dribbling"].mean()),
    }])

    # --- effect sizes ---------------------------------------------------------------
    d = uni[disc]
    X = _foul_design(uni)
    ors = _logit_odds_ratios(X[disc], uni.loc[disc, "y_card"].to_numpy(dtype=float),
                             uni.loc[disc, "match_id"].to_numpy(), n_boot=200,
                             seed=cfg.split.seed)
    ors["where"] = "discovery"
    ors_c = _logit_odds_ratios(X[~disc], uni.loc[~disc, "y_card"].to_numpy(dtype=float),
                               uni.loc[~disc, "match_id"].to_numpy(), n_boot=200,
                               seed=cfg.split.seed)
    ors_c["where"] = "confirmation"
    out["b_odds_ratios"] = pd.concat([ors, ors_c], ignore_index=True)

    contrasts = []
    for where, sub in (("discovery", d), ("confirmation", uni[~disc])):
        t = _rate_contrast(sub, ["zone", "dribbling"], "y_card", cfg)
        t["where"] = where
        contrasts.append(t)
    out["b_rate_by_zone_dribbling"] = pd.concat(contrasts, ignore_index=True)

    fouln = uni.assign(fouls_bucket=pd.cut(uni["team_fouls_so_far"], [-0.5, 2.5, 5.5, 8.5, 11.5, 100],
                                           labels=["0-2", "3-5", "6-8", "9-11", "12+"]))
    t = _rate_contrast(fouln[~disc], ["fouls_bucket"], "y_card", cfg)
    t["where"] = "confirmation"
    out["b_rate_by_team_fouls"] = t

    minb = uni.assign(minute_bucket=pd.cut(uni["minute_c"], [-0.5, 15, 30, 45, 60, 75, 101],
                                           labels=["0-15", "15-30", "30-45", "45-60",
                                                   "60-75", "75+"]))
    t = _rate_contrast(minb[~disc], ["minute_bucket"], "y_card", cfg)
    t["where"] = "confirmation"
    out["b_rate_by_minute"] = t

    t = _rate_contrast(uni[~disc], ["fouled_line"], "y_card", cfg)
    t["where"] = "confirmation"
    out["b_rate_by_fouled_line"] = t

    # --- predictive model -------------------------------------------------------------
    import lightgbm as lgb
    from research.privileged_tracking.common.splits import group_kfold

    Xn = X.copy()
    y = uni["y_card"].to_numpy(dtype=float)
    mid = uni["match_id"].to_numpy()
    params = dict(cfg.lgb_params)
    params.update(n_estimators=400)
    d_idx = np.flatnonzero(disc)
    oof = np.full(len(uni), np.nan)
    for tr, te in group_kfold(mid[d_idx], n_splits=cfg.n_folds, seed=cfg.split.seed):
        m = lgb.LGBMClassifier(**{**params, "random_state": cfg.split.seed})
        m.fit(Xn.iloc[d_idx[tr]], y[d_idx[tr]])
        oof[d_idx[te]] = m.predict_proba(Xn.iloc[d_idx[te]])[:, 1]
    m = lgb.LGBMClassifier(**{**params, "random_state": cfg.split.seed})
    m.fit(Xn.iloc[d_idx], y[d_idx])
    conf = m.predict_proba(Xn.iloc[np.flatnonzero(~disc)])[:, 1]
    ab = C.platt_fit(oof[disc], y[disc])
    rows = []
    base = float(y[disc].mean())
    for where, mask, p in (("discovery_oof", disc, C.platt_apply(oof[disc], ab)),
                           ("confirmation", ~disc, C.platt_apply(conf, ab))):
        r = {"where": where, "model": "foul_lgb"}
        r.update(_binary_metrics(y[mask], p))
        rows.append(r)
        r0 = {"where": where, "model": "base_rate"}
        r0.update(_binary_metrics(y[mask], np.full(mask.sum(), base)))
        rows.append(r0)
    # out-of-domain (tournaments and other seasons)
    if len(ood) > 0:
        Xo = _foul_design(ood).reindex(columns=Xn.columns, fill_value=0.0)
        po = C.platt_apply(m.predict_proba(Xo)[:, 1], ab)
        yo = ood["y_card"].to_numpy(dtype=float)
        r = {"where": "out_of_domain", "model": "foul_lgb"}
        r.update(_binary_metrics(yo, po))
        rows.append(r)
        r0 = {"where": "out_of_domain", "model": "base_rate"}
        r0.update(_binary_metrics(yo, np.full(len(yo), base)))
        rows.append(r0)
    out["b_metrics"] = pd.DataFrame(rows)

    imp = pd.DataFrame({"feature": Xn.columns, "gain": m.booster_.feature_importance("gain")})
    out["b_importance"] = imp.sort_values("gain", ascending=False).head(25).reset_index(drop=True)

    for k, v in out.items():
        C.write_table(v, f"01_cards_{k}")
    return out


# ---------------------------------------------------------------------------
# Stage (c): team card totals on the football-data.co.uk odds table
# ---------------------------------------------------------------------------

_COUNTRY_MAP = {"EC": "E", "SC0": "SC", "SC1": "SC", "SC2": "SC", "SC3": "SC"}

_ODDS_ACCUM = [
    "yellow_for", "yellow_against", "red_for", "match_yellow", "fouls_for",
    "fouls_against", "shots_for", "shots_against", "target_for", "corners_for",
    "corners_against", "goals_for", "goals_against", "points", "match_goals",
    "has_cards", "has_fouls", "has_goals",
]


def _country(div: str) -> str:
    """Map a football-data division code to a country key."""
    if div in _COUNTRY_MAP:
        return _COUNTRY_MAP[div]
    return "".join(ch for ch in div if not ch.isdigit())


def build_odds_table(cfg: CardsConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Match-level odds table with strictly-prior rolling team, division and pair features.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        One row per match with ``h_*`` / ``a_*`` prior features, division and pair
        priors, market probabilities, card and goal targets and the ``split`` label.
    """
    path = C.processed_dir() / "cards_odds_matches.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    d = pd.read_parquet(C.odds_path())
    d = d.sort_values(["MatchDate", "Division", "HomeTeam", "AwayTeam"],
                      kind="mergesort").reset_index(drop=True)
    d["uid"] = np.arange(len(d))
    d["country"] = d["Division"].map(_country)
    d["home_key"] = d["country"] + "|" + d["HomeTeam"]
    d["away_key"] = d["country"] + "|" + d["AwayTeam"]
    d["match_yellow"] = d["HomeYellow"] + d["AwayYellow"]
    d["match_goals"] = d["FTHome"] + d["FTAway"]
    d["has_cards"] = d["match_yellow"].notna().astype(float)
    d["has_fouls"] = (d["HomeFouls"].notna() & d["AwayFouls"].notna()).astype(float)
    d["has_goals"] = d["match_goals"].notna().astype(float)

    long = []
    for side, opp in (("Home", "Away"), ("Away", "Home")):
        t = pd.DataFrame({
            "uid": d["uid"], "MatchDate": d["MatchDate"],
            "team_key": d[f"{side.lower()}_key"], "Division": d["Division"],
            "yellow_for": d[f"{side}Yellow"], "yellow_against": d[f"{opp}Yellow"],
            "red_for": d[f"{side}Red"], "match_yellow": d["match_yellow"],
            "fouls_for": d[f"{side}Fouls"], "fouls_against": d[f"{opp}Fouls"],
            "shots_for": d[f"{side}Shots"], "shots_against": d[f"{opp}Shots"],
            "target_for": d[f"{side}Target"], "corners_for": d[f"{side}Corners"],
            "corners_against": d[f"{opp}Corners"], "goals_for": d[f"FT{side}"],
            "goals_against": d[f"FT{opp}"], "match_goals": d["match_goals"],
            "has_cards": d["has_cards"], "has_fouls": d["has_fouls"],
            "has_goals": d["has_goals"], "side": side,
        })
        t["points"] = np.where(t["goals_for"] > t["goals_against"], 3.0,
                               np.where(t["goals_for"] == t["goals_against"], 1.0, 0.0))
        long.append(t)
    lt = pd.concat(long, ignore_index=True)
    lt = lt.sort_values(["MatchDate", "uid", "side"], kind="mergesort").reset_index(drop=True)
    pri = C.prior_expanding(lt, ["team_key"], _ODDS_ACCUM, ["MatchDate", "uid"],
                            count_name="tp_matches", prefix="s_")
    lt = pd.concat([lt, pri], axis=1)

    glob = {"match_yellow": 3.7, "yellow_for": 1.85, "yellow_against": 1.85,
            "fouls_for": 12.5, "fouls_against": 12.5, "shots_for": 12.5,
            "shots_against": 12.5, "target_for": 4.5, "corners_for": 5.2,
            "corners_against": 5.2, "goals_for": 1.35, "goals_against": 1.35,
            "points": 1.4, "match_goals": 2.65, "red_for": 0.08}
    denom = {"match_yellow": "s_has_cards", "yellow_for": "s_has_cards",
             "yellow_against": "s_has_cards", "red_for": "s_has_cards",
             "fouls_for": "s_has_fouls", "fouls_against": "s_has_fouls",
             "shots_for": "s_has_fouls", "shots_against": "s_has_fouls",
             "target_for": "s_has_fouls", "corners_for": "s_has_fouls",
             "corners_against": "s_has_fouls", "goals_for": "s_has_goals",
             "goals_against": "s_has_goals", "points": "s_has_goals",
             "match_goals": "s_has_goals"}
    for v, g in glob.items():
        lt["p_" + v] = C.shrunk_rate(lt["s_" + v].to_numpy(), lt[denom[v]].to_numpy(),
                                     g, cfg.team_k)
    keep = ["uid", "side", "tp_matches", "s_has_cards", "s_has_goals"] + \
        ["p_" + v for v in glob]
    wide = lt[keep].pivot(index="uid", columns="side")
    wide.columns = [("h_" if c[1] == "Home" else "a_") + c[0] for c in wide.columns]
    d = d.merge(wide.reset_index(), on="uid", how="left")

    d["div_card_prior"] = C.prior_group_daily_mean(d, "Division", "MatchDate",
                                                   "match_yellow", "has_cards", 3.7, 40.0)
    d["div_foul_prior"] = C.prior_group_daily_mean(d, "Division", "MatchDate",
                                                   "HomeFouls", "has_fouls", 12.5, 40.0)
    d["div_goal_prior"] = C.prior_group_daily_mean(d, "Division", "MatchDate",
                                                   "match_goals", "has_goals", 2.65, 40.0)

    pair = np.where(d["home_key"] < d["away_key"], d["home_key"] + "~" + d["away_key"],
                    d["away_key"] + "~" + d["home_key"])
    d["pair_key"] = pair
    ppri = C.prior_expanding(d, ["pair_key"], ["match_yellow", "has_cards", "match_goals",
                                               "has_goals"],
                             ["MatchDate", "uid"], count_name="pair_n", prefix="pp_")
    d = pd.concat([d, ppri], axis=1)
    d["pair_card_prior"] = C.shrunk_rate(d["pp_match_yellow"].to_numpy(),
                                         d["pp_has_cards"].to_numpy(), 3.7, 6.0)
    d["pair_goal_prior"] = C.shrunk_rate(d["pp_match_goals"].to_numpy(),
                                         d["pp_has_goals"].to_numpy(), 2.65, 6.0)

    ph, pd_, pa = _novig_three_way(d["OddHome"], d["OddDraw"], d["OddAway"])
    d["mkt_p_home"], d["mkt_p_draw"], d["mkt_p_away"] = ph, pd_, pa
    d["mkt_fav"] = np.maximum(ph, pa)
    d["mkt_close"] = 1.0 - d["mkt_fav"]
    over, under = C.novig_two_way(d["Over25"].to_numpy(), d["Under25"].to_numpy())
    d["mkt_p_over25"] = over
    d["elo_gap"] = d["HomeElo"] - d["AwayElo"]
    d["abs_elo_gap"] = d["elo_gap"].abs()
    d["y_total_cards"] = d["match_yellow"]
    d["y_total_goals"] = d["match_goals"]

    # proxy mean: each team's own card environment plus the opponent's, minus the
    # division level that both already contain
    d["proxy_mu_cards"] = np.clip(d["h_p_match_yellow"] + d["a_p_match_yellow"]
                                  - d["div_card_prior"], 0.3, None)
    d["proxy_mu_goals"] = np.clip(d["h_p_match_goals"] + d["a_p_match_goals"]
                                  - d["div_goal_prior"], 0.3, None)
    d["split"] = C.chronological_split(d["uid"], d["MatchDate"], cfg.split.discovery_frac)
    d.to_parquet(path, index=False)
    return d


def _novig_three_way(o_h: pd.Series, o_d: pd.Series, o_a: pd.Series
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proportional de-vig of a 1X2 price triple.

    Args:
        o_h: Home decimal odds [n].
        o_d: Draw decimal odds [n].
        o_a: Away decimal odds [n].

    Returns:
        Tuple of no-vig probabilities, each [n].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ih, idr, ia = 1.0 / o_h.to_numpy(), 1.0 / o_d.to_numpy(), 1.0 / o_a.to_numpy()
        tot = ih + idr + ia
        out = (ih / tot, idr / tot, ia / tot)
    return tuple(np.where(np.isfinite(v), v, np.nan) for v in out)


_C_PROXY = ["proxy_mu_cards", "h_p_match_yellow", "a_p_match_yellow", "div_card_prior",
            "h_tp_matches", "a_tp_matches", "h_s_has_cards", "a_s_has_cards"]
_C_EVENT = [
    "h_p_yellow_for", "h_p_yellow_against", "a_p_yellow_for", "a_p_yellow_against",
    "h_p_red_for", "a_p_red_for", "h_p_fouls_for", "h_p_fouls_against",
    "a_p_fouls_for", "a_p_fouls_against", "h_p_shots_for", "h_p_shots_against",
    "a_p_shots_for", "a_p_shots_against", "h_p_target_for", "a_p_target_for",
    "h_p_corners_for", "h_p_corners_against", "a_p_corners_for", "a_p_corners_against",
    "h_p_goals_for", "h_p_goals_against", "a_p_goals_for", "a_p_goals_against",
    "h_p_points", "a_p_points", "div_foul_prior", "div_goal_prior", "pair_card_prior",
    "pair_n", "HomeElo", "AwayElo", "elo_gap", "abs_elo_gap", "Form3Home", "Form5Home",
    "Form3Away", "Form5Away", "Division", "country",
]
_C_MARKET = ["mkt_p_home", "mkt_p_draw", "mkt_p_away", "mkt_fav", "mkt_close",
             "mkt_p_over25"]

# the goals mirror of the card feature sets: the same rolling machinery, the same access to
# its own proxy mean, and deliberately NO market features (the market is what it is scored
# against in the honesty check)
_G_PROXY = ["proxy_mu_goals", "h_p_match_goals", "a_p_match_goals", "div_goal_prior",
            "h_tp_matches", "a_tp_matches", "h_s_has_goals", "a_s_has_goals"]

_C_SETS = {
    "C0_proxy_feats": _C_PROXY,
    "C1_event": _C_PROXY + _C_EVENT,
    "C2_event_plus_market": _C_PROXY + _C_EVENT + _C_MARKET,
}
_C_CAT = ["Division", "country"]
CARD_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)


def _c_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Design frame for the odds-table models with fixed categorical dtypes."""
    X = df[list(cols)].copy()
    for c in _C_CAT:
        if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def _count_model(df: pd.DataFrame, cols: Sequence[str], target: str, disc: np.ndarray,
                 cfg: CardsConfig, seed: int, iters: Sequence[int] = (40, 80, 150, 300)
                 ) -> dict[str, object]:
    """LightGBM Poisson count model with a discovery-fitted negative-binomial dispersion.

    Args:
        df: Match-level frame.
        cols: Feature columns.
        target: Count target column.
        disc: Discovery mask [n].
        cfg: Stage configuration.
        seed: Seed.
        iters: Candidate boosting-iteration counts, selected on discovery OOF deviance.

    Returns:
        Dict with ``mu`` [n], ``r`` (dispersion) and ``best_iter``.
    """
    import lightgbm as lgb

    from research.privileged_tracking.common.splits import group_kfold

    X = _c_frame(df, cols)
    y = df[target].to_numpy(dtype=float)
    d_idx = np.flatnonzero(disc)
    params = dict(cfg.lgb_params)
    params.pop("objective", None)
    params.update(objective="poisson", n_estimators=int(max(iters)), random_state=seed,
                  bagging_seed=seed, feature_fraction_seed=seed)
    cat = [c for c in _C_CAT if c in X.columns]
    oof = {k: np.full(len(df), np.nan) for k in iters}
    for tr, te in group_kfold(df["uid"].to_numpy()[d_idx], n_splits=cfg.n_folds, seed=seed):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[d_idx[tr]], y[d_idx[tr]], categorical_feature=cat)
        for k in iters:
            oof[k][d_idx[te]] = m.predict(X.iloc[d_idx[te]], num_iteration=int(k))
    def dev(mu: np.ndarray) -> float:
        mu = np.clip(mu, 1e-6, None)
        return float(np.mean(mu - y[disc] * np.log(mu)))
    scores = {k: dev(oof[k][disc]) for k in iters}
    best = int(min(scores, key=scores.get))
    m = lgb.LGBMRegressor(**params)
    m.fit(X.iloc[d_idx], y[d_idx], categorical_feature=cat)
    mu = oof[best].copy()
    mu[~disc] = m.predict(X.iloc[np.flatnonzero(~disc)], num_iteration=best)
    mu = np.clip(mu, 0.2, None)
    r = C.fit_nb_dispersion(y[disc], mu[disc])
    return {"mu": mu, "r": r, "best_iter": best, "iter_scores": scores}


def _proxy_count_model(df: pd.DataFrame, proxy_col: str, target: str, disc: np.ndarray
                       ) -> dict[str, object]:
    """Recalibrated rolling-mean proxy as a count distribution.

    ``log mu = a + b log(proxy)`` fitted by Poisson regression on discovery only, then a
    negative-binomial dispersion fitted on the same rows. This is the strawman a book is
    assumed to use -- recalibrated, so it is not beaten on scale alone.

    Args:
        df: Match-level frame.
        proxy_col: Column holding the raw rolling-mean proxy.
        target: Count target column.
        disc: Discovery mask [n].

    Returns:
        Dict with ``mu`` [n], ``r``, ``a`` and ``b``.
    """
    z = np.log(np.clip(df[proxy_col].to_numpy(dtype=float), 0.05, None)).reshape(-1, 1)
    y = df[target].to_numpy(dtype=float)
    a, b = _poisson_glm1(z[disc, 0], y[disc])
    mu = np.clip(np.exp(a + b * z[:, 0]), 0.2, None)
    r = C.fit_nb_dispersion(y[disc], mu[disc])
    return {"mu": mu, "r": r, "a": a, "b": b}


def _poisson_glm1(x: np.ndarray, y: np.ndarray, n_iter: int = 60) -> tuple[float, float]:
    """Fit ``log mu = a + b x`` by Newton iteration on the Poisson log-likelihood.

    Args:
        x: Single covariate [n].
        y: Counts [n].
        n_iter: Maximum Newton steps.

    Returns:
        Tuple ``(a, b)``.
    """
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([np.log(max(y.mean(), 1e-3)), 0.0])
    for _ in range(n_iter):
        mu = np.exp(np.clip(X @ beta, -20, 20))
        g = X.T @ (y - mu)
        H = X.T @ (X * mu[:, None])
        step = np.linalg.solve(H + 1e-8 * np.eye(2), g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def goals_honesty_check(d: pd.DataFrame, cfg: CardsConfig) -> dict[str, pd.DataFrame]:
    """Calibrate the rolling-mean strawman against a real bookmaker on the one count
    market whose line we actually observe: total goals at 2.5.

    Two gaps are measured. The *information gap* is how much better Bet365's no-vig
    Over/Under 2.5 probability is than the identical rolling-mean proxy (log-loss and
    Brier). The *ROI gap* runs the whole betting simulation twice with the same
    event-only model -- once against a synthetic book priced off the proxy, once against
    Bet365's real prices -- and the difference is how much a strawman book inflates ROI.
    That difference is the haircut applied to every card EV number.

    Args:
        d: Output of :func:`build_odds_table`.
        cfg: Stage configuration.

    Returns:
        Dict of result tables.
    """
    u = d[d["y_total_goals"].notna() & d["Over25"].notna() & d["Under25"].notna()
          & (d["h_s_has_goals"] >= 5) & (d["a_s_has_goals"] >= 5)].copy().reset_index(drop=True)
    disc = (u["split"] == C.DISCOVERY).to_numpy()
    y = u["y_total_goals"].to_numpy(dtype=float)
    yb = (y > 2.5).astype(float)
    gid = u["uid"].to_numpy()
    out: dict[str, pd.DataFrame] = {}

    prox = _proxy_count_model(u, "proxy_mu_goals", "y_total_goals", disc)
    p_proxy = C.nb_sf(2.5, prox["mu"], prox["r"])
    ab = C.platt_fit(p_proxy[disc], yb[disc])
    p_proxy = C.platt_apply(p_proxy, ab)
    p_book = u["mkt_p_over25"].to_numpy(dtype=float)
    over_round = (1.0 / u["Over25"] + 1.0 / u["Under25"]).to_numpy()

    goal_cols = list(dict.fromkeys(_G_PROXY + list(_C_EVENT)))
    model = _count_model(u, goal_cols, "y_total_goals", disc, cfg, cfg.split.seed)
    p_model = C.nb_sf(2.5, model["mu"], model["r"])
    abm = C.platt_fit(p_model[disc], yb[disc])
    p_model = C.platt_apply(p_model, abm)

    rows = []
    for where, mask in (("discovery", disc), ("confirmation", ~disc)):
        clim = np.full(int(mask.sum()), float(yb[disc].mean()))
        for name, p in (("climatology", clim), ("rolling_mean_proxy", p_proxy[mask]),
                        ("event_only_model", p_model[mask]),
                        ("bet365_novig", p_book[mask])):
            r = {"where": where, "model": name}
            r.update(_binary_metrics(yb[mask], p))
            d_, lo, hi = clustered_bootstrap_delta(
                per_sample_log_loss(yb[mask], p_proxy[mask]),
                per_sample_log_loss(yb[mask], p), gid[mask],
                cfg.split.n_boot, cfg.split.seed)
            r.update(delta_vs_proxy=d_, ci_lo=lo, ci_hi=hi)
            rows.append(r)
    out["c_goals_gap"] = pd.DataFrame(rows)
    out["c_goals_setup"] = pd.DataFrame([{
        "n_total": int(len(u)), "n_discovery": int(disc.sum()),
        "n_confirmation": int((~disc).sum()),
        "date_min": str(u["MatchDate"].min().date()), "date_max": str(u["MatchDate"].max().date()),
        "mean_bet365_overround": float(np.nanmean(over_round)),
        "proxy_a": prox["a"], "proxy_b": prox["b"], "proxy_r": prox["r"],
        "model_best_iter": model["best_iter"], "base_rate_over25": float(yb.mean()),
    }])

    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
    bet_rows = []
    for hold in cfg.holds:
        t = _choose_threshold(yb[disc], p_model[disc], p_proxy[disc], gid[disc], hold, cfg,
                              grid, min_bets=200)
        for where, mask in (("discovery", disc), ("confirmation", ~disc)):
            r = C.simulate_two_way(yb[mask], p_model[mask], p_proxy[mask], gid[mask],
                                   C.BetSimConfig(hold=hold, edge_threshold=t,
                                                  n_boot=cfg.split.n_boot, seed=cfg.split.seed))
            bet_rows.append(r.as_row(book="rolling_mean_proxy", hold=hold, threshold=t,
                                     where=where, market="goals_over_2.5"))
            # The margin-matched leg: the same bettor, the same threshold and the same
            # hold, but the price is built from Bet365's *no-vig* probability instead of
            # the proxy's. Differencing this against the row above charges only the
            # difference in sharpness, with the margin held fixed -- unlike the
            # actual-prices leg below, whose hold is whatever Bet365 posted.
            rm = C.simulate_two_way(yb[mask], p_model[mask], p_book[mask], gid[mask],
                                    C.BetSimConfig(hold=hold, edge_threshold=t,
                                                   n_boot=cfg.split.n_boot,
                                                   seed=cfg.split.seed))
            bet_rows.append(rm.as_row(book="bet365_novig_repriced", hold=hold, threshold=t,
                                      where=where, market="goals_over_2.5"))
    # against the real Bet365 price (its own vig, no synthetic hold)
    o_over = u["Over25"].to_numpy(dtype=float)
    o_under = u["Under25"].to_numpy(dtype=float)
    ev_o = p_model * o_over - 1.0
    ev_u = (1.0 - p_model) * o_under - 1.0
    take_o = ev_o >= ev_u
    ev = np.where(take_o, ev_o, ev_u)
    profit = np.where(take_o, np.where(yb > 0.5, o_over - 1.0, -1.0),
                      np.where(yb < 0.5, o_under - 1.0, -1.0))
    best_t, best_roi = grid[0], -np.inf
    for t in grid:
        b = disc & (ev > t)
        if b.sum() >= 200 and profit[b].mean() > best_roi:
            best_roi, best_t = float(profit[b].mean()), float(t)
    for where, mask in (("discovery", disc), ("confirmation", ~disc)):
        b = mask & (ev > best_t)
        roi, lo, hi = C.clustered_bootstrap_mean(profit[b], gid[b], cfg.split.n_boot,
                                                 cfg.split.seed)
        bet_rows.append({"book": "bet365_real", "hold": float(np.nanmean(over_round) - 1.0),
                         "threshold": best_t, "where": where, "market": "goals_over_2.5",
                         "n_rows": int(mask.sum()), "n_bets": int(b.sum()),
                         "bet_rate": float(b.sum() / max(mask.sum(), 1)),
                         "mean_edge": float(ev[b].mean()) if b.sum() else float("nan"),
                         "roi": roi, "roi_lo": lo, "roi_hi": hi,
                         "n_over": int((b & take_o).sum()), "profit": float(profit[b].sum()),
                         "mean_p_model": float(p_model[b].mean()) if b.sum() else float("nan"),
                         "mean_p_book": float(p_book[b].mean()) if b.sum() else float("nan"),
                         "realized_a_rate": float(yb[b].mean()) if b.sum() else float("nan"),
                         "mean_odds": float(np.where(take_o, o_over, o_under)[b].mean())
                         if b.sum() else float("nan")})
    out["c_goals_bets"] = pd.DataFrame(bet_rows)

    bt = out["c_goals_bets"]
    gap_ll = float(out["c_goals_gap"].query(
        "where == 'confirmation' and model == 'bet365_novig'")["delta_vs_proxy"].iloc[0])
    cut_rows = []
    for where in ("discovery", "confirmation"):
        w_ = bt[bt["where"] == where]
        real_roi = float(w_[w_["book"] == "bet365_real"]["roi"].iloc[0])
        real_hold = float(w_[w_["book"] == "bet365_real"]["hold"].iloc[0])
        for hold in cfg.holds:
            a = float(w_[(w_["book"] == "rolling_mean_proxy")
                         & (w_["hold"] == hold)]["roi"].iloc[0])
            c = float(w_[(w_["book"] == "bet365_novig_repriced")
                         & (w_["hold"] == hold)]["roi"].iloc[0])
            cut_rows.append({
                "where": where, "hold": float(hold),
                "log_loss_gap_book_minus_proxy_nats": gap_ll,
                "roi_vs_proxy": a,
                "roi_vs_real_book_matched": c,
                "roi_vs_real_book_actual_prices": real_roi,
                "real_book_hold": real_hold,
                "haircut_roi_points_matched": float(a - c),
                "haircut_roi_points_actual_prices": float(a - real_roi)})
    out["c_haircut_by_hold"] = pd.DataFrame(cut_rows)

    conf = bt[bt["where"] == "confirmation"]
    proxy_roi = float(conf[conf["book"] == "rolling_mean_proxy"]["roi"].mean())
    real_roi = float(conf[conf["book"] == "bet365_real"]["roi"].iloc[0])
    m6 = [r for r in cut_rows if r["where"] == "confirmation" and r["hold"] == 0.06]
    out["c_haircut"] = pd.DataFrame([{
        "log_loss_gap_book_minus_proxy_nats": gap_ll,
        "roi_vs_proxy_mean_over_holds": proxy_roi,
        "roi_vs_real_book": real_roi,
        "roi_haircut": float(proxy_roi - real_roi),
        "roi_haircut_matched_6pct": m6[0]["haircut_roi_points_matched"] if m6 else float("nan"),
        "real_book_hold": float(conf[conf["book"] == "bet365_real"]["hold"].iloc[0]),
    }])
    return out


def stage_c(cfg: CardsConfig = CFG) -> dict[str, pd.DataFrame]:
    """Team card totals on 125k football-data matches, plus the goals honesty check.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables, also written to the reports directory.
    """
    d = build_odds_table(cfg)
    out: dict[str, pd.DataFrame] = goals_honesty_check(d, cfg)

    u = d[d["y_total_cards"].notna() & (d["h_s_has_cards"] >= 5)
          & (d["a_s_has_cards"] >= 5)].copy().reset_index(drop=True)
    u["match_id"] = u["uid"]
    for c in _C_CAT:
        u[c] = u[c].astype("category")
    disc = (u["split"] == C.DISCOVERY).to_numpy()
    y = u["y_total_cards"].to_numpy(dtype=float)
    gid = u["uid"].to_numpy()

    out["c_setup"] = pd.DataFrame([{
        "n_total": int(len(u)), "n_discovery": int(disc.sum()),
        "n_confirmation": int((~disc).sum()),
        "date_min": str(u["MatchDate"].min().date()),
        "date_cut": str(u.loc[~disc, "MatchDate"].min().date()),
        "date_max": str(u["MatchDate"].max().date()),
        "mean_total_cards": float(y.mean()), "var_total_cards": float(y.var()),
        "n_divisions": int(u["Division"].nunique()),
        **{f"p_over_{L}": float((y > L).mean()) for L in CARD_LINES},
    }])

    prox = _proxy_count_model(u, "proxy_mu_cards", "y_total_cards", disc)
    models: dict[str, dict[str, object]] = {"C0_proxy": prox}
    for name in ("C1_event", "C2_event_plus_market"):
        models[name] = _count_model(u, _C_SETS[name], "y_total_cards", disc, cfg,
                                    cfg.split.seed)
    p_line: dict[tuple[str, float], np.ndarray] = {}
    for name, mod in models.items():
        for L in CARD_LINES:
            p = C.nb_sf(L, mod["mu"], mod["r"])
            yb = (y > L).astype(float)
            ab = C.platt_fit(p[disc], yb[disc])
            p_line[(name, L)] = C.platt_apply(p, ab)
    # direct binary classifiers at the three main lines
    for L in (3.5, 4.5, 5.5):
        u["_yb"] = (y > L).astype(float)
        res = _cv_and_confirm(u, _C_SETS["C2_event_plus_market"], "_yb", disc, cfg,
                              cfg.split.seed, iters=(30, 60, 100, 200, 350))
        p_line[("C2_binary", L)] = res["cal"]

    rows = []
    for where, mask in (("discovery_oof", disc), ("confirmation", ~disc)):
        for L in CARD_LINES:
            yb = (y > L).astype(float)
            ref = p_line[("C0_proxy", L)]
            clim = np.full(int(mask.sum()), float(yb[disc].mean()))
            cands = [("climatology", clim)] + [
                (n, p_line[(n, L)][mask]) for n in ("C0_proxy", "C1_event",
                                                    "C2_event_plus_market")]
            if ("C2_binary", L) in p_line:
                cands.append(("C2_binary", p_line[("C2_binary", L)][mask]))
            for name, p in cands:
                r = {"where": where, "model": name, "line": L}
                r.update(_binary_metrics(yb[mask], p))
                dd, lo, hi = clustered_bootstrap_delta(
                    per_sample_log_loss(yb[mask], ref[mask]),
                    per_sample_log_loss(yb[mask], p), gid[mask],
                    cfg.split.n_boot, cfg.split.seed)
                r.update(delta_vs_proxy=dd, ci_lo=lo, ci_hi=hi)
                rows.append(r)
    out["c_metrics"] = pd.DataFrame(rows)
    out["c_fit_info"] = pd.DataFrame([
        {"model": n, "best_iter": m.get("best_iter"), "nb_r": m["r"],
         "glm_a": m.get("a"), "glm_b": m.get("b")} for n, m in models.items()])

    # --- the gate at the 3.5 line ------------------------------------------------------
    fh = u["h_p_fouls_for"].to_numpy() + u["a_p_fouls_for"].to_numpy()
    fh_cut = float(np.nanquantile(fh[disc], 2 / 3))
    elo_cut = float(np.nanquantile(u.loc[disc, "abs_elo_gap"].dropna(), 1 / 3))
    div_cut = float(np.nanquantile(u.loc[disc, "div_card_prior"], 2 / 3))
    tight = u["abs_elo_gap"].to_numpy()
    scen = {
        "high_foul_both": fh >= fh_cut,
        "high_foul_tight": (fh >= fh_cut) & (tight <= elo_cut),
        "high_foul_tight_cardy_div": (fh >= fh_cut) & (tight <= elo_cut)
        & (u["div_card_prior"].to_numpy() >= div_cut),
    }
    u["_yb35"] = (y > 3.5).astype(float)
    best_iter = int(out["c_fit_info"].query("model == 'C2_event_plus_market'")
                    ["best_iter"].iloc[0])
    gate_params = {**cfg.lgb_params, "n_estimators": best_iter}
    gate_rows = []
    for label, m in scen.items():
        m = np.where(np.isfinite(m), m, False).astype(bool)
        gate_rows += _gate(u, _C_SETS["C2_event_plus_market"], "_yb35", m, cfg, label,
                           cfg.split.seed, gate_params)
    out["c_gate"] = pd.DataFrame(gate_rows)
    out["c_scenario_defs"] = pd.DataFrame([{
        "foul_sum_tercile_cut": fh_cut, "abs_elo_gap_lower_tercile_cut": elo_cut,
        "division_card_tercile_cut": div_cut,
        **{f"n_discovery_{k}": int((np.nan_to_num(v, nan=0).astype(bool) & disc).sum())
           for k, v in scen.items()},
        **{f"n_confirmation_{k}": int((np.nan_to_num(v, nan=0).astype(bool) & ~disc).sum())
           for k, v in scen.items()},
    }])

    # --- betting simulation at the proxy's own line ------------------------------------
    p_over_proxy = np.column_stack([p_line[("C0_proxy", L)] for L in CARD_LINES])
    pick = np.argmin(np.abs(p_over_proxy - 0.5), axis=1)
    lines = np.array(CARD_LINES)[pick]
    yb_pick = (y > lines).astype(float)
    p_book_pick = p_over_proxy[np.arange(len(u)), pick]
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.40]
    bet_rows = []
    for name in ("C1_event", "C2_event_plus_market"):
        pm = np.column_stack([p_line[(name, L)] for L in CARD_LINES])
        p_model_pick = pm[np.arange(len(u)), pick]
        for hold in cfg.holds:
            t = _choose_threshold(yb_pick[disc], p_model_pick[disc], p_book_pick[disc],
                                  gid[disc], hold, cfg, grid, min_bets=200)
            for where, mask in (("discovery", disc), ("confirmation", ~disc)):
                r = C.simulate_two_way(yb_pick[mask], p_model_pick[mask],
                                       p_book_pick[mask], gid[mask],
                                       C.BetSimConfig(hold=hold, edge_threshold=t,
                                                      n_boot=cfg.split.n_boot,
                                                      seed=cfg.split.seed))
                bet_rows.append(r.as_row(book="C0_proxy", model=name, hold=hold,
                                         threshold=t, where=where,
                                         market="match_cards_proxy_line"))
    bets = pd.DataFrame(bet_rows)
    haircut = float(out["c_haircut"]["roi_haircut"].iloc[0])
    bets["roi_after_haircut"] = bets["roi"] - haircut
    bets["roi_lo_after_haircut"] = bets["roi_lo"] - haircut
    out["c_bets"] = bets

    pm_best = np.column_stack([p_line[("C2_event_plus_market", L)] for L in CARD_LINES])[
        np.arange(len(u)), pick]
    out["c_book_ladder"] = _book_ladder(yb_pick, pm_best, p_book_pick, disc, gid, cfg,
                                        grid, "match_cards_proxy_line")
    out["c_line_mix"] = pd.Series(lines).value_counts().rename_axis("line").reset_index(
        name="n_matches")

    dis = C.disagreement_table(yb_pick[~disc],
                               np.column_stack([p_line[("C2_event_plus_market", L)]
                                                for L in CARD_LINES])[
                                   np.arange(len(u)), pick][~disc],
                               p_book_pick[~disc], n_bins=5)
    dis["where"] = "confirmation"
    out["c_disagreement"] = dis

    refit = []
    for seed in (cfg.split.seed, cfg.split.seed + 1, cfg.split.seed + 2):
        m = _count_model(u, _C_SETS["C2_event_plus_market"], "y_total_cards", disc, cfg, seed)
        p = C.nb_sf(3.5, m["mu"], m["r"])
        yb = (y > 3.5).astype(float)
        ab = C.platt_fit(p[disc], yb[disc])
        refit.append({"model": "C2_event_plus_market", "seed": seed, "line": 3.5,
                      "confirmation_log_loss": log_loss(yb[~disc],
                                                        C.platt_apply(p, ab)[~disc])})
    out["c_refit"] = pd.DataFrame(refit)

    for k, v in out.items():
        C.write_table(v, f"01_cards_{k}")
    return out


def main() -> None:
    """CLI entry point for the cards stage."""
    ap = argparse.ArgumentParser(description="Stage 01: cards")
    ap.add_argument("--stage", default="all", choices=["build", "a", "b", "c", "all"])
    ap.add_argument("--force", action="store_true", help="rebuild cached intermediates")
    args = ap.parse_args()
    if args.stage in ("build", "all"):
        pl = build_player_table(CFG, force=args.force)
        st = build_team_state_priors(CFG, force=args.force)
        gt = build_goal_times(force=args.force)
        fl = build_foul_table(CFG, force=args.force)
        od = build_odds_table(CFG, force=args.force)
        print(f"player table {pl.shape}, team state {st.shape}, goal times {gt.shape}, "
              f"fouls {fl.shape}, odds {od.shape}")
    if args.stage in ("b", "all"):
        res = stage_b(CFG)
        print("\n== (b) counts ==")
        print(res["b_counts"].to_string(index=False))
        print("\n== (b) metrics ==")
        print(res["b_metrics"].to_string(index=False))
        print("\n== (b) rate by zone x dribbling ==")
        print(res["b_rate_by_zone_dribbling"].to_string(index=False))
        print("\n== (b) odds ratios (confirmation) ==")
        print(res["b_odds_ratios"].query("where == 'confirmation'").to_string(index=False))
    if args.stage in ("c", "all"):
        res = stage_c(CFG)
        for k in ("c_setup", "c_goals_setup", "c_goals_gap", "c_goals_bets", "c_haircut",
                  "c_metrics", "c_gate", "c_bets"):
            print(f"\n== {k} ==")
            print(res[k].to_string(index=False))
    if args.stage in ("a", "all"):
        res = stage_a(CFG)
        print("\n== (a) metrics ==")
        print(res["a_metrics"].to_string(index=False))
        print("\n== (a) gate ==")
        print(res["a_gate"].to_string(index=False))
        print("\n== (a) bets ==")
        print(res["a_bets"].to_string(index=False))


if __name__ == "__main__":
    main()
