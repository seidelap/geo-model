"""NFL stage 01: build play-level tables with tracking-derived "privileged" targets.

Reads the 2017 Big Data Bowl tracking CSVs (91 games, weeks 1-6) one game at a time,
computes per-play targets with :mod:`tracking_features`, joins nflfastR play-by-play
and participation context, cross-checks every derived quantity against the charted
value where one exists, and writes:

* ``processed_dir('nfl')/plays_tracked.parquet``        one row per tracked play
* ``processed_dir('nfl')/participants_tracked.parquet`` one row per play x player
* ``processed_dir('nfl')/player_crosswalk.parquet``     BDB nflId -> nflverse gsis_id
* ``reports_dir()/nfl_01_build.md`` plus ``nfl_01_*.parquet`` result tables

Column contract (leakage tiers, see :func:`column_tier` and
``nfl_01_column_definitions.parquet``): ``id`` keys; ``event_only_presnap`` play-by-play
situation known before the snap (the only admissible imputation features);
``event_only_postsnap`` play-by-play outcomes and post-snap model values;
``ngs_charting_privileged`` (``ngs_*`` and the BDB charting columns) NGS charting derived
from the same tracking, labels only; ``tracking_target`` the quantities computed here;
``tracking_context`` tracking bookkeeping / diagnostics. Nothing outside ``id`` and
``event_only_presnap`` may be a feature of a pre-snap state model.

Usage::

    cd /home/user/geo-model
    python -m research.privileged_tracking.nfl.build_tables [--limit-games N] [--n-jobs 2]
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import nfl_dir, processed_dir, reports_dir
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.nfl import tracking_features as tf

PASS_RESULTS = ("C", "I", "IN", "S")
TUNE_WEEKS = (1, 2, 3)          # definition grids are selected here and reported on the other weeks
EXACT_TOL = 1e-6                # |a - b| <= tol counts as "exact" (float rounding on mirrored plays)
RUSH_GRID: tuple[tuple[float, float], ...] = tuple(
    (w, c) for w in (1.0, 1.5, 2.0) for c in (0.5, 0.0, -0.5, -1.0)
)
BOX_GRID: tuple[tuple[float, float], ...] = tuple(
    (d, lat) for d in (4.0, 5.0, 5.5, 6.0, 6.5, 7.0) for lat in (5.0, 6.0, 6.5, 7.0, 7.5, 8.0)
)

TRACKING_USECOLS = ["x", "y", "s", "dir", "event", "nflId", "displayName", "jerseyNumber",
                    "team", "frame.id", "gameId", "playId"]

PBP_COLUMNS = [
    "old_game_id", "play_id", "game_id", "week", "posteam", "defteam", "home_team", "away_team",
    "qtr", "down", "ydstogo", "goal_to_go", "yardline_100", "half_seconds_remaining",
    "game_seconds_remaining", "score_differential", "play_type", "shotgun", "no_huddle",
    "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "pass_length", "pass_location",
    "air_yards", "yards_after_catch", "complete_pass", "incomplete_pass", "interception", "sack",
    "qb_hit", "run_location", "run_gap", "yards_gained", "epa", "ep", "wp", "wpa", "cp", "cpoe",
    "xpass", "pass_oe", "success", "first_down", "passer_player_id", "passer_player_name",
    "receiver_player_id", "receiver_player_name", "rusher_player_id", "rusher_player_name",
    "spread_line", "total_line", "roof", "surface", "temp", "wind", "desc", "penalty",
    "aborted_play", "touchdown", "fumble",
]
# nflverse pbp_participation columns; every one of them is NGS charting derived from the
# tracking, so they are published under the ``ngs_`` prefix (never ``part_``) so that no
# prefix filter can mistake them for event-only context.
PARTICIPATION_COLUMNS = [
    "old_game_id", "play_id", "possession_team", "offense_formation", "offense_personnel",
    "defenders_in_box", "defense_personnel", "number_of_pass_rushers", "offense_players",
    "defense_players", "n_offense", "n_defense", "ngs_air_yards", "time_to_throw",
    "was_pressure", "route",
]
PLAYS_RENAME = {"personnel.offense": "personnel_offense", "personnel.defense": "personnel_defense"}
BDB_KEEP = ["gameId", "playId", "quarter", "GameClock", "down", "yardsToGo", "possessionTeam",
            "yardlineSide", "yardlineNumber", "offenseFormation", "personnel_offense",
            "defendersInTheBox", "numberOfPassRushers", "personnel_defense", "HomeScoreBeforePlay",
            "VisitorScoreBeforePlay", "isPenalty", "PassLength", "PassResult", "YardsAfterCatch",
            "PlayResult", "playDescription"]

# ---------------------------------------------------------------------------
# Column tiers (the leakage contract of plays_tracked.parquet)
# ---------------------------------------------------------------------------

TIERS: dict[str, str] = {
    "id": "keys, teams, week and join flags; safe anywhere",
    "event_only_presnap": "play-by-play situation known before the snap (down, distance, yard line, clock, score, "
                          "market lines, venue, pre-snap models ep/wp/xpass, description-derived shotgun/no_huddle); "
                          "the only admissible features for a pre-snap state model",
    "event_only_postsnap": "play-by-play outcome or post-snap model value (play type, yards, EPA, completion, sack, "
                           "passer/receiver/rusher ids, description); payoff experiments only, never a feature of a "
                           "pre-snap state model",
    "ngs_charting_privileged": "NGS charting derived from the same tracking (formation, personnel, defenders in box, "
                               "pass rushers, who was on the field, time to throw, pressure, air yards, route) from "
                               "pbp_participation (ngs_*) and plays.csv; labels / oracle only",
    "tracking_target": "quantities computed here from the tracking frames: the imputation targets",
    "tracking_context": "tracking bookkeeping and diagnostics (orientation, LOS source, tag presence, derived "
                        "identities and their agreement with pbp); privileged, not a feature",
}
_ID_COLUMNS = {
    "gameId", "playId", "week", "home_team", "away_team", "offense_team", "defense_team", "offense_is_home",
    "pbp_joined", "ngs_joined", "possessionTeam", "pbp_game_id", "pbp_week", "pbp_posteam", "pbp_defteam",
    "pbp_home_team", "pbp_away_team", "ngs_possession_team",
}
_BDB_PRESNAP = {"quarter", "GameClock", "down", "yardsToGo", "yardlineSide", "yardlineNumber",
                "HomeScoreBeforePlay", "VisitorScoreBeforePlay"}
_BDB_POSTSNAP = {"isPenalty", "PassLength", "PassResult", "YardsAfterCatch", "PlayResult", "playDescription",
                 "is_pass_play"}
_BDB_CHARTING = {"offenseFormation", "personnel_offense", "personnel_defense", "defendersInTheBox",
                 "numberOfPassRushers"}
_PBP_PRESNAP = {"qtr", "down", "ydstogo", "goal_to_go", "yardline_100", "half_seconds_remaining",
                "game_seconds_remaining", "score_differential", "shotgun", "no_huddle", "ep", "wp", "xpass",
                "spread_line", "total_line", "roof", "surface", "temp", "wind"}
_TRACKING_CONTEXT = {
    "flipped", "snap_frame_id", "n_frames", "n_off_tracked", "n_def_tracked", "los_x", "los_x_ball",
    "los_x_official", "los_source", "los_ball_discrepancy", "ball_y_ref", "ball_y_source", "qb_id", "qb_source",
    "qb_id_geometric", "release_event", "arrival_event", "carrier_event", "qb_id_throw", "qb_same_as_presnap",
    "qb_ball_dist_at_release", "target_id", "carrier_id", "frames_before_snap", "n_ol_derived",
}


def column_tier(name: str) -> str:
    """Leakage tier of a ``plays_tracked.parquet`` column (see :data:`TIERS`).

    Rules, in order: explicit id columns; BDB plays.csv columns (pre-snap situation /
    post-snap outcome / NGS charting); ``pbp_*`` (pre-snap situation and pre-snap models,
    everything else post-snap); ``ngs_*`` charting; tracking bookkeeping; the rest are
    tracking-derived targets.
    """
    if name in _ID_COLUMNS:
        return "id"
    if name in _BDB_PRESNAP:
        return "event_only_presnap"
    if name in _BDB_POSTSNAP:
        return "event_only_postsnap"
    if name in _BDB_CHARTING:
        return "ngs_charting_privileged"
    if name.startswith("pbp_"):
        return "event_only_presnap" if name[4:] in _PBP_PRESNAP else "event_only_postsnap"
    if name.startswith("ngs_"):
        return "ngs_charting_privileged"
    if name in _TRACKING_CONTEXT or name.startswith("ev_") or name.endswith("_gsis") or name.endswith("_id_agree"):
        return "tracking_context"
    return "tracking_target"


def event_only_presnap_columns(columns: list[str]) -> list[str]:
    """Subset of ``columns`` that a pre-snap imputation model may use as features (ids excluded)."""
    return [c for c in columns if column_tier(c) == "event_only_presnap"]


@dataclass
class BuildConfig:
    """Driver configuration.

    Attributes:
        features: thresholds for the tracking targets.
        n_jobs: worker processes over games.
        limit_games: process only the first N games (smoke runs).
        rush_grid: pass-rusher definitions tried for the agreement table.
        box_grid: box-count definitions tried for the agreement table.
    """

    features: tf.FeatureConfig = field(default_factory=tf.FeatureConfig)
    n_jobs: int = 2
    limit_games: int | None = None
    rush_grid: tuple[tuple[float, float], ...] = RUSH_GRID
    box_grid: tuple[tuple[float, float], ...] = BOX_GRID


# ---------------------------------------------------------------------------
# Per-game worker
# ---------------------------------------------------------------------------

def _qb_index(pf: tf.PlayFrames, positions: np.ndarray) -> tuple[int | None, str, int | None]:
    """Pick the QB column: roster position first, geometry as fallback / tie-break."""
    geom = tf.identify_qb_geometric(pf.x[pf.snap_idx], pf.y[pf.snap_idx], pf.off_mask,
                                    pf.los_x, pf.ball_y_ref)
    qbs = np.where(pf.off_mask & (positions == "QB"))[0]
    if len(qbs) == 1:
        return int(qbs[0]), "position", geom
    if len(qbs) > 1:
        if geom is not None and geom in qbs:
            return int(geom), "position_multi_geom", geom
        si = pf.snap_idx
        d = np.abs(pf.y[si, qbs] - pf.ball_y_ref) + np.abs(pf.los_x - pf.x[si, qbs])
        return int(qbs[int(np.argmin(d))]), "position_multi_nearest", geom
    return geom, "geometric", geom


def process_game(args: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Compute play- and participant-level rows for one game.

    Args:
        args: ``(game_id, static, cfg_dict)`` where ``static`` holds the plays for this
            game (records), the position map and the home/away abbreviations.
    """
    game_id, static, cfg_dict = args
    cfg = tf.FeatureConfig(**cfg_dict["features"])
    rush_grid = tuple(tuple(x) for x in cfg_dict["rush_grid"])
    plays = {int(r["playId"]): r for r in static["plays"]}
    pos_map: dict[int, str] = static["positions"]
    home, away = static["home"], static["away"]
    path = Path(static["tracking_path"])
    t0 = time.time()
    df = pd.read_csv(path, usecols=TRACKING_USECOLS,
                     dtype={"event": "object", "team": "object", "displayName": "object"})
    play_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for play_id, g in df.groupby("playId", sort=True):
        play_id = int(play_id)
        r = plays.get(play_id)
        if r is None:
            skipped.append({"gameId": game_id, "playId": play_id, "reason": "not_in_plays_csv"})
            continue
        if bool(r["isSTPlay"]):
            continue
        offense_label = "home" if r["possessionTeam"] == home else "away"
        los_off = tf.official_los(r["yardlineSide"], r["yardlineNumber"], r["possessionTeam"])
        los_source = "official_bdb"
        if not np.isfinite(los_off):
            y100 = r.get("pbp_yardline_100", np.nan)
            if y100 is not None and np.isfinite(float(y100)):
                los_off = 110.0 - float(y100)
                los_source = "official_pbp"
            else:
                los_source = "ball"
        events_present = set(g["event"].dropna().unique())
        if "ball_snap" not in events_present:
            skipped.append({"gameId": game_id, "playId": play_id, "reason": "no_ball_snap",
                            "has_snap_direct": "snap_direct" in events_present,
                            "desc": str(r["playDescription"])[:80]})
            continue
        pf = tf.PlayFrames.from_long(
            g["frame.id"].to_numpy(), g["nflId"].to_numpy(), g["team"].to_numpy(),
            g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy(), g["dir"].to_numpy(),
            g["event"].to_numpy(dtype=object), offense_label=offense_label,
            los_official=los_off if np.isfinite(los_off) else None, fps=cfg.fps,
        )
        pids = pf.player_ids.astype(int)
        positions = np.array([pos_map.get(int(p), "UNK") for p in pids])
        qb_idx, qb_source, qb_geom = _qb_index(pf, positions)

        row: dict[str, Any] = {
            "gameId": game_id, "playId": play_id, "week": static["week"],
            "home_team": home, "away_team": away,
            "offense_team": r["possessionTeam"], "defense_team": away if offense_label == "home" else home,
            "offense_is_home": offense_label == "home", "flipped": pf.flipped,
            "snap_frame_id": int(pf.frame_ids[pf.snap_idx]), "n_frames": pf.n_frames,
            "n_off_tracked": int(pf.off_mask.sum()), "n_def_tracked": int(pf.def_mask.sum()),
            "los_x": pf.los_x, "los_x_ball": pf.los_x_ball, "los_x_official": los_off,
            "los_source": los_source,
            "los_ball_discrepancy": pf.los_x_ball - los_off if np.isfinite(los_off) else np.nan,
            "ball_y_ref": pf.ball_y_ref, "ball_y_source": pf.ball_y_source,
            "qb_id": int(pids[qb_idx]) if qb_idx is not None else None,
            "qb_source": qb_source,
            "qb_id_geometric": int(pids[qb_geom]) if qb_geom is not None else None,
        }
        row.update(tf.presnap_targets(pf, qb_idx, cfg))
        off_counts = tf.personnel_counts(positions[pf.off_mask], "offense")
        def_counts = tf.personnel_counts(positions[pf.def_mask], "defense")
        row.update({f"derived_{k}": v for k, v in off_counts.items()})
        row.update({f"derived_{k}": v for k, v in def_counts.items()})
        row["personnel_offense_derived"] = tf.personnel_string(off_counts, "offense")
        row["personnel_defense_derived"] = tf.personnel_string(def_counts, "defense")
        for ev in ("pass_forward", "pass_shovel", "pass_arrived", "pass_outcome_caught",
                   "pass_outcome_incomplete", "qb_sack", "handoff", "run", "first_contact",
                   "tackle", "man_in_motion", "line_set", "qb_kneel", "fumble", "touchdown"):
            row[f"ev_{ev}"] = ev in pf.events
        pr = r["PassResult"]
        is_pass = isinstance(pr, str) and pr in PASS_RESULTS
        row["is_pass_play"] = is_pass
        if is_pass:
            row.update(tf.pass_targets(pf, qb_idx, pr, cfg, rush_grid=rush_grid))
        else:
            row.update(tf.run_targets(pf, qb_idx, cfg))
        play_rows.append(row)

        for prow in tf.participant_rows(pf, qb_idx, cfg):
            j = int(np.where(pids == int(prow["player_id"]))[0][0])
            prow.update({
                "gameId": game_id, "playId": play_id, "nflId": int(prow.pop("player_id")),
                "position": positions[j],
                "team_abbr": r["possessionTeam"] if prow["side"] == "offense" else row["defense_team"],
            })
            part_rows.append(prow)
    names = (df.loc[df["team"] != tf.BALL_LABEL, ["nflId", "displayName", "jerseyNumber", "team"]]
             .drop_duplicates("nflId"))
    names["team_abbr"] = np.where(names["team"] == "home", home, away)
    names = names.drop(columns="team")
    return {"game_id": game_id, "plays": play_rows, "participants": part_rows, "skipped": skipped,
            "names": names.to_dict("records"), "seconds": time.time() - t0,
            "n_tracking_plays": int(df["playId"].nunique())}


# ---------------------------------------------------------------------------
# Crosswalk BDB nflId -> nflverse gsis_id
# ---------------------------------------------------------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(s: str | None) -> str:
    """Lower-case, strip punctuation and generational suffixes."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("'", "").replace(".", ""))
    toks = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(toks)


def build_crosswalk(names: pd.DataFrame, players: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Map BDB ``nflId`` to nflverse ``gsis_id`` via team + name, jersey as tie-break.

    Args:
        names: unique tracked players ``[nflId, displayName, jerseyNumber, team_abbr]``
            (a player may appear with more than one team).
        players: BDB ``players.csv`` (``nflId, FirstName, LastName, PositionAbbr``).
        roster: nflverse ``roster_weekly_<season>`` rows.

    Returns:
        ``[nflId, gsis_id, match_method, roster_name, roster_position]``.
    """
    ros = roster[["team", "gsis_id", "full_name", "first_name", "last_name", "jersey_number",
                  "position", "status"]].dropna(subset=["gsis_id"]).copy()
    ros["nfull"] = ros["full_name"].map(norm_name)
    ros["nlast"] = ros["last_name"].map(norm_name)
    ros["nfirst"] = ros["first_name"].map(norm_name)
    ros = ros.drop_duplicates(["team", "gsis_id", "jersey_number"])
    by_team_last: dict[tuple[str, str], pd.DataFrame] = {k: v for k, v in ros.groupby(["team", "nlast"])}
    by_full: dict[str, pd.DataFrame] = {k: v for k, v in ros.groupby("nfull")}
    by_team_jersey: dict[tuple[str, float], pd.DataFrame] = {k: v for k, v in ros.groupby(["team", "jersey_number"])}
    pl = players.set_index("nflId")

    out = []
    for nfl_id, grp in names.groupby("nflId"):
        nfl_id = int(nfl_id)
        teams = sorted(set(grp["team_abbr"]))
        jerseys = set(grp["jerseyNumber"].dropna().astype(float))
        cand_names = [norm_name(n) for n in grp["displayName"].dropna().unique()]
        if nfl_id in pl.index:
            cand_names.append(norm_name(f"{pl.loc[nfl_id, 'FirstName']} {pl.loc[nfl_id, 'LastName']}"))
        cand_names = [n for n in dict.fromkeys(cand_names) if n]
        match: tuple[str, pd.Series] | None = None

        def unique_gsis(df: pd.DataFrame) -> pd.Series | None:
            if df is None or len(df) == 0:
                return None
            g = df["gsis_id"].unique()
            if len(g) == 1:
                return df.iloc[0]
            if jerseys:
                dj = df[df["jersey_number"].isin(jerseys)]
                if dj["gsis_id"].nunique() == 1:
                    return dj.iloc[0]
            return None

        for nm in cand_names:  # 1. team + full name
            toks = nm.split()
            for team in teams:
                df = by_team_last.get((team, toks[-1]))
                if df is None:
                    continue
                dfull = df[df["nfull"] == nm]
                hit = unique_gsis(dfull)
                if hit is not None:
                    match = ("team_full_name", hit)
                    break
            if match:
                break
        if match is None:  # 2. team + last name + first initial
            for nm in cand_names:
                toks = nm.split()
                for team in teams:
                    df = by_team_last.get((team, toks[-1]))
                    if df is None:
                        continue
                    di = df[df["nfirst"].str[:1] == toks[0][:1]]
                    hit = unique_gsis(di)
                    if hit is not None:
                        match = ("team_last_initial", hit)
                        break
                if match:
                    break
        if match is None:  # 3. team + jersey + last name
            for nm in cand_names:
                toks = nm.split()
                for team in teams:
                    for j in jerseys:
                        df = by_team_jersey.get((team, j))
                        if df is None:
                            continue
                        dl = df[df["nlast"] == toks[-1]]
                        hit = unique_gsis(dl)
                        if hit is not None:
                            match = ("team_jersey_last", hit)
                            break
                    if match:
                        break
                if match:
                    break
        if match is None:  # 4. league-wide unique full name
            for nm in cand_names:
                df = by_full.get(nm)
                if df is not None and df["gsis_id"].nunique() == 1:
                    match = ("name_any_team", df.iloc[0])
                    break
        if match is None:  # 5. team + jersey among active players
            for team in teams:
                for j in jerseys:
                    df = by_team_jersey.get((team, j))
                    if df is None:
                        continue
                    da = df[df["status"] == "ACT"]
                    if da["gsis_id"].nunique() == 1:
                        match = ("team_jersey_active", da.iloc[0])
                        break
                if match:
                    break
        if match is None:
            out.append({"nflId": nfl_id, "gsis_id": None, "match_method": "unmatched",
                        "roster_name": None, "roster_position": None})
        else:
            out.append({"nflId": nfl_id, "gsis_id": match[1]["gsis_id"], "match_method": match[0],
                        "roster_name": match[1]["full_name"], "roster_position": match[1]["position"]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Agreement tables
# ---------------------------------------------------------------------------

def agreement_stats(a: pd.Series, b: pd.Series, label_a: str, label_b: str,
                    tol: float = EXACT_TOL) -> dict[str, Any]:
    """Exact / within-1 agreement, MAE and mean bias between two columns.

    ``exact`` is ``|a - b| <= tol`` (not bit equality: mirrored plays go through
    ``120 - x`` and pick up ~1e-14 rounding, and NGS values are stored to 0.01 yd).
    """
    m = a.notna() & b.notna()
    if m.sum() == 0:
        return {"a": label_a, "b": label_b, "n": 0}
    d = (a[m].astype(float) - b[m].astype(float))
    return {"a": label_a, "b": label_b, "n": int(m.sum()), "exact": float((d.abs() <= tol).mean()),
            "within_1": float((d.abs() <= 1).mean()), "mae": float(d.abs().mean()),
            "bias_a_minus_b": float(d.mean()), "corr": float(np.corrcoef(a[m].astype(float), b[m].astype(float))[0, 1])
            if a[m].nunique() > 1 and b[m].nunique() > 1 else np.nan}


def confusion(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    m = a.notna() & b.notna()
    return pd.crosstab(a[m], b[m], dropna=False)


def grid_agreement(a: pd.Series, b: pd.Series, week: pd.Series, label_a: str, label_b: str,
                   tune_weeks: tuple[int, ...] = TUNE_WEEKS) -> dict[str, Any]:
    """Agreement of a candidate definition on the tuning weeks and on the held-out weeks.

    Returns the tuning-week stats (``exact_tune``, ``n_tune``), the held-out stats
    (``exact``, ``within_1``, ``mae``, ``bias_a_minus_b``, ``n``) and ``exact_all``.
    """
    tune = week.isin(tune_weeks)
    st_tune = agreement_stats(a[tune], b[tune], label_a, label_b)
    st_hold = agreement_stats(a[~tune], b[~tune], label_a, label_b)
    st_all = agreement_stats(a, b, label_a, label_b)
    out = {"a": label_a, "b": label_b, "n_tune": st_tune["n"], "exact_tune": st_tune.get("exact", np.nan),
           "n": st_hold["n"]}
    for k in ("exact", "within_1", "mae", "bias_a_minus_b"):
        out[k] = st_hold.get(k, np.nan)
    out["exact_all"] = st_all.get("exact", np.nan)
    return out


def tables_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = False) -> str:
    """Markdown table with three-decimal floats (thin wrapper over ``common.report.md_table``)."""
    return md_table(df, floatfmt=floatfmt, index=index)


def summarise_targets(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({"column": c, "n": int(s.notna().sum()), "missing_rate": float(s.isna().mean()),
                     "mean": s.mean(), "sd": s.std(), "p05": s.quantile(0.05), "p25": s.quantile(0.25),
                     "p50": s.quantile(0.5), "p75": s.quantile(0.75), "p95": s.quantile(0.95)})
    return pd.DataFrame(rows)


COLUMN_DEFS: dict[str, str] = {
    "gameId / playId": "BDB keys; pbp joins on old_game_id == str(gameId), play_id == playId",
    "week / home_team / away_team / offense_team / defense_team / offense_is_home": "games.csv week (1-6; use for forward splits) and team abbreviations (BDB codes, e.g. OAK); which side had the ball",
    "flipped": "raw coordinates were mirrored so the offense attacks toward +x",
    "snap_frame_id / n_frames / frames_before_snap": "raw frame.id of the ball_snap tag, frames in the play, frames before the snap (the tracking starts ~1.4 s before the snap)",
    "n_off_tracked / n_def_tracked": "players with tracking rows per side (11 v 11 on 99.8% of plays; nothing is imputed when fewer)",
    "los_x_official": "the official-yard-line LOS in normalised coordinates (NaN only when neither BDB nor nflfastR gives a yard line)",
    "los_x / los_source": "line of scrimmage used for all depths: BDB yard line (10+own / 110-opp; official_bdb), else nflfastR 110 - yardline_100 (official_pbp), else ball x at snap (ball)",
    "los_x_ball / los_ball_discrepancy": "ball x at the snap frame (normalised) and its difference from the official line",
    "ball_y_ref / ball_y_source": "lateral reference: ball y at snap, or an estimate of the centre's y when the ball track is implausible (out of bounds / >8 yd from the offense's median y)",
    "qb_id / qb_source / qb_id_geometric": "pre-snap QB nflId from roster position (QB); geometric rule = second most-forward offensive player within 0.8 yd laterally of the ball (the first is the centre), kept as a cross-check and fallback",
    "n_deep_safeties": "defenders with depth (x - los_x) >= 10 at the snap",
    "def_depth_max / def_depth_2nd / def_depth_mean": "max, second-max and mean defender depth at the snap",
    "mof_open": "n_deep_safeties >= 2",
    "box_count": "naive box: defenders with -1 <= depth <= 5 and |lateral| <= 8 at the snap",
    "box_count_tuned": "box definition from the grid with the best exact agreement with defendersInTheBox (see agreement section)",
    "n_dl": "defenders with |depth| <= 1.5 at the snap",
    "def_y_std / def_y_range / def_lateral_mean": "spread of defender y at the snap; mean defender lateral offset from the ball",
    "n_wide_left / n_wide_right / n_wide": "offensive non-QB players with lateral >= +8 (left) / <= -8 (right), i.e. beyond the tackle box on the offense's left / right; positive lateral is the offense's left (nflfastR pass_location 'left' <-> target_lateral > 0)",
    "widest_split": "max |lateral| of any offensive non-QB player",
    "n_backfield": "offensive non-QB players >= 2.5 yd behind los_x and within 6 yd laterally (2.5 not 1 because the OL front sits ~1.1 yd behind the official line)",
    "n_ol_derived": "number of OL candidates found (five closest laterally to the ball within 2 yd of the line)",
    "off_y_std": "standard deviation of offensive y at the snap (formation width)",
    "n_te_inline": "non-QB, non-OL players on the line (<= 2.5 yd deep) within 4 yd outside the widest lineman on their side",
    "qb_depth / qb_lateral / shotgun_derived": "QB yards behind los_x and lateral offset (positive = offense's left) at the snap; shotgun_derived = qb_depth >= 4",
    "cushion_left / cushion_right / cb_cushion": "distance from the widest skill player on the offense's left (lateral > 0) / right (lateral < 0) to the nearest defender at the snap; cb_cushion = mean of available sides",
    "motion_disp_max / motion_event / motion_derived": "max lateral range of any offensive player in the 2 s before the snap; man_in_motion tag before snap; either range > 3 yd or the tag",
    "line_set_to_snap_s / shift_to_snap_s / huddle_break_offense_to_snap_s": "seconds from the line_set / shift / huddle_break_offense tag to the snap when tagged (rare tags)",
    "derived_RB/TE/WR/OL/QB, derived_DL/LB/DB": "position-group counts of tracked players from players.csv PositionAbbr",
    "personnel_offense_derived / personnel_defense_derived": "those counts rendered in the NGS string convention",
    "is_pass_play": "PassResult in C / I / IN / S",
    "release_event / time_to_throw": "seconds from ball_snap to pass_forward (or pass_shovel); sacks use qb_sack, NaN if no such tag",
    "qb_id_throw / qb_same_as_presnap / qb_ball_dist_at_release": "offensive player nearest the ball 3 frames before the release tag (the tag lands ~0.2-0.3 s after the ball leaves the hand); agreement with the pre-snap QB; his distance to the ball there",
    "qb_depth_at_throw / qb_lateral_at_throw / qb_speed_at_throw": "QB geometry at release",
    "min_def_dist_qb_throw / min_def_dist_qb_throw_m05 / min_def_dist_qb_dropback": "closest defender to the QB at release, 0.5 s before release, and at any frame between snap and release",
    "n_def_within_r_qb_throw": "defenders within 3 yd of the QB at release",
    "n_pass_rushers_derived": "defenders whose min x within 1.5 s after the snap reaches los_x (naive spec definition)",
    "n_pass_rushers_tuned": "grid definition (window, crossing depth) with the best exact agreement with numberOfPassRushers",
    "n_rush_w<ww>_c<m|p><dd>": "pass-rusher counts for the whole grid (window ww/10 s, crossing depth +-dd/10 yd)",
    "arrival_event / target_id": "frame tag used for the arrival (pass_arrived, else pass_outcome_*) and the non-QB offensive player nearest the ball there",
    "separation_at_arrival / separation_at_throw": "min distance from the targeted receiver to any defender at arrival / at release",
    "n_def_within_r_target": "defenders within 5 yd of the target at arrival",
    "target_depth / target_lateral / target_speed_at_arrival": "target x - los_x, y - ball_y_ref (positive = offense's left) and speed at arrival",
    "target_side_derived": "left / middle / right from target_lateral with a 6 yd middle half-width (offense perspective; compare pbp_pass_location)",
    "target_ball_dist_at_arrival / ball_depth_at_arrival / air_time_s": "target-to-ball distance and ball depth at arrival; seconds from release to arrival",
    "carrier_event / carrier_id": "handoff (non-QB nearest the ball), else run tag (any offensive player), else pass_shovel",
    "time_to_handoff / carrier_speed_at_handoff / carrier_depth_at_handoff / carrier_lateral_at_handoff": "carrier state at the handoff frame (lateral positive = offense's left)",
    "n_def_within_r_carrier_handoff / min_def_dist_carrier_handoff": "defenders within 3 yd of / nearest to the carrier at handoff",
    "yards_to_first_contact / time_to_first_contact": "carrier x - los_x and seconds from snap at the first_contact tag",
    "carrier_speed_at_first_contact / n_def_within_r_carrier_first_contact": "carrier speed and defenders within 3 yd at first contact",
    "ev_<event>": "whether the tag exists anywhere in the play",
    "quarter / GameClock / down / yardsToGo / yardlineSide / yardlineNumber / HomeScoreBeforePlay / VisitorScoreBeforePlay / possessionTeam": "BDB plays.csv pre-snap situation",
    "isPenalty / PassLength / PassResult / YardsAfterCatch / PlayResult / playDescription": "BDB plays.csv outcome columns (post-snap)",
    "offenseFormation / personnel_offense / personnel_defense / defendersInTheBox / numberOfPassRushers": "BDB plays.csv NGS charting (same source as ngs_*; privileged)",
    "pbp_*": "curated nflfastR play_by_play_2017 columns (see PBP_COLUMNS); tier splits pre-snap situation / pre-snap models from post-snap outcomes",
    "ngs_*": "nflverse pbp_participation_2017 columns (see PARTICIPATION_COLUMNS): NGS charting from the same tracking; ngs_was_pressure is 1 / 0 / NaN",
    "*_gsis / *_id_agree": "crosswalked nflverse ids for the derived QB / target / carrier and agreement with pbp passer / receiver / rusher ids",
}


def _definition_lookup() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    exact: dict[str, str] = {}
    patterns: list[tuple[str, str, str]] = []   # (kind, token, definition)
    for key, definition in COLUMN_DEFS.items():
        if key.startswith("derived_"):          # "derived_RB/TE/WR/OL/QB, derived_DL/LB/DB"
            patterns.append(("prefix", "derived_", definition))
            continue
        for tok in re.split(r"\s*/\s*|,\s*", key):
            tok = tok.strip()
            if not tok:
                continue
            if "<" in tok:
                patterns.append(("prefix", tok.split("<")[0], definition))
            elif tok.endswith("*"):
                patterns.append(("prefix", tok[:-1], definition))
            elif tok.startswith("*"):
                patterns.append(("suffix", tok[1:], definition))
            else:
                exact[tok] = definition
    return exact, patterns


def column_definitions_table(columns: list[str]) -> pd.DataFrame:
    """One row per actual column: ``column``, ``tier`` (:func:`column_tier`), ``definition``."""
    exact, patterns = _definition_lookup()
    rows = []
    for c in columns:
        d = exact.get(c, "")
        if not d:
            for kind, tok, definition in patterns:
                if (kind == "prefix" and c.startswith(tok)) or (kind == "suffix" and c.endswith(tok)):
                    d = definition
                    break
        rows.append({"column": c, "tier": column_tier(c), "definition": d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: BuildConfig) -> dict[str, Any]:
    t_start = time.time()
    bdb = nfl_dir() / "bdb2017"
    nv = nfl_dir() / "nflverse"
    out_dir = processed_dir("nfl")
    rep_dir = reports_dir()

    games = pd.read_csv(bdb / "games.csv")
    plays = pd.read_csv(bdb / "plays.csv").rename(columns=PLAYS_RENAME)
    players = pd.read_csv(bdb / "players.csv")
    pos_map = {int(k): str(v) for k, v in zip(players["nflId"], players["PositionAbbr"])}
    games = games.sort_values(["week", "gameId"])
    if cfg.limit_games:
        games = games.head(cfg.limit_games)
    # nflfastR yard line as LOS fallback for the ~1.4% of plays where BDB has none (midfield)
    y100 = pd.read_parquet(nv / "play_by_play_2017.parquet", columns=["old_game_id", "play_id", "yardline_100"])
    y100 = y100.dropna(subset=["play_id"])
    y100["gameId"] = y100["old_game_id"].astype(str).astype(int)
    y100["playId"] = y100["play_id"].astype(int)
    y100 = y100.drop_duplicates(["gameId", "playId"])[["gameId", "playId", "yardline_100"]]
    plays = plays.merge(y100.rename(columns={"yardline_100": "pbp_yardline_100"}), on=["gameId", "playId"], how="left")

    jobs = []
    for _, g in games.iterrows():
        gid = int(g["gameId"])
        pg = plays[plays["gameId"] == gid]
        static = {"plays": pg.to_dict("records"), "positions": pos_map, "home": g["homeTeamAbbr"],
                  "away": g["visitorTeamAbbr"], "week": int(g["week"]),
                  "tracking_path": str(bdb / "tracking" / f"tracking_gameId_{gid}.csv")}
        jobs.append((gid, static, {"features": cfg.features.as_dict(), "rush_grid": cfg.rush_grid}))

    results = []
    if cfg.n_jobs > 1:
        with Pool(cfg.n_jobs) as pool:
            for res in pool.imap_unordered(process_game, jobs):
                results.append(res)
                print(f"game {res['game_id']} plays={len(res['plays'])} {res['seconds']:.1f}s", flush=True)
    else:
        for job in jobs:
            res = process_game(job)
            results.append(res)
            print(f"game {res['game_id']} plays={len(res['plays'])} {res['seconds']:.1f}s", flush=True)

    results.sort(key=lambda r: r["game_id"])   # imap_unordered returns games in completion order
    play_df = pd.DataFrame([r for res in results for r in res["plays"]])
    part_df = pd.DataFrame([r for res in results for r in res["participants"]])
    skipped = pd.DataFrame([r for res in results for r in res["skipped"]])
    names = pd.DataFrame([r for res in results for r in res["names"]])
    n_tracking_plays = sum(res["n_tracking_plays"] for res in results)

    # --- BDB plays columns ---
    play_df = play_df.merge(plays[BDB_KEEP], on=["gameId", "playId"], how="left")

    # --- nflfastR pbp ---
    pbp = pd.read_parquet(nv / "play_by_play_2017.parquet", columns=PBP_COLUMNS)
    pbp = pbp.dropna(subset=["play_id"]).copy()
    pbp["gameId"] = pbp["old_game_id"].astype(str).astype(int)
    pbp["playId"] = pbp["play_id"].astype(int)
    pbp = pbp.drop(columns=["old_game_id", "play_id"]).drop_duplicates(["gameId", "playId"])
    pbp = pbp.rename(columns={c: f"pbp_{c}" for c in pbp.columns if c not in ("gameId", "playId")})
    play_df = play_df.merge(pbp, on=["gameId", "playId"], how="left", indicator="pbp_join")
    play_df["pbp_joined"] = play_df.pop("pbp_join") == "both"

    part = pd.read_parquet(nv / "pbp_participation_2017.parquet", columns=PARTICIPATION_COLUMNS)
    part["gameId"] = part["old_game_id"].astype(str).astype(int)
    part["playId"] = part["play_id"].astype(int)
    part = part.drop(columns=["old_game_id", "play_id"]).drop_duplicates(["gameId", "playId"])
    part["was_pressure"] = part["was_pressure"].map(
        lambda v: {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}.get(v, np.nan) if v is not None else np.nan)
    part = part.rename(columns={c: (c if c.startswith("ngs_") else f"ngs_{c}")
                                for c in part.columns if c not in ("gameId", "playId")})
    play_df = play_df.merge(part, on=["gameId", "playId"], how="left", indicator="ngs_join")
    play_df["ngs_joined"] = play_df.pop("ngs_join") == "both"

    # --- crosswalk and participants ---
    roster = pd.read_parquet(nv / "roster_weekly_2017.parquet")
    xwalk = build_crosswalk(names, players, roster)
    part_df = part_df.merge(names[["nflId", "displayName", "jerseyNumber"]].drop_duplicates("nflId"),
                            on="nflId", how="left")
    part_df = part_df.merge(xwalk[["nflId", "gsis_id", "match_method"]], on="nflId", how="left")
    plist = play_df[["gameId", "playId", "ngs_offense_players", "ngs_defense_players"]]
    part_df = part_df.merge(plist, on=["gameId", "playId"], how="left")

    def _in_list(row: pd.Series) -> float:
        lst = row["ngs_offense_players"] if row["side"] == "offense" else row["ngs_defense_players"]
        if not isinstance(lst, str) or not isinstance(row["gsis_id"], str):
            return np.nan
        return float(row["gsis_id"] in lst.split(";"))

    part_df["in_pbp_participation"] = part_df.apply(_in_list, axis=1)
    part_df = part_df.drop(columns=["ngs_offense_players", "ngs_defense_players"])
    part_cols = ["gameId", "playId", "nflId", "gsis_id", "match_method", "displayName", "jerseyNumber",
                 "team_abbr", "side", "position", "role_derived", "is_qb_used", "in_pbp_participation",
                 "x_snap", "y_snap", "s_snap", "depth", "lateral", "in_box_naive"]
    part_df = part_df[part_cols].sort_values(["gameId", "playId", "nflId"], kind="stable").reset_index(drop=True)
    play_df = play_df.sort_values(["gameId", "playId"], kind="stable").reset_index(drop=True)
    xwalk = xwalk.sort_values("nflId").reset_index(drop=True)
    if len(skipped):
        skipped = skipped.sort_values(["gameId", "playId"]).reset_index(drop=True)

    id_map = dict(zip(xwalk["nflId"], xwalk["gsis_id"]))
    for col in ("qb_id", "qb_id_throw", "target_id", "carrier_id"):
        play_df[f"{col.replace('_id', '')}_gsis"] = play_df[col].map(lambda v: id_map.get(int(v)) if pd.notna(v) else None)
    play_df["qb_id_agree"] = np.where(play_df["qb_throw_gsis"].notna() & play_df["pbp_passer_player_id"].notna(),
                                      play_df["qb_throw_gsis"] == play_df["pbp_passer_player_id"], np.nan)
    play_df["target_id_agree"] = np.where(play_df["target_gsis"].notna() & play_df["pbp_receiver_player_id"].notna(),
                                          play_df["target_gsis"] == play_df["pbp_receiver_player_id"], np.nan)
    play_df["carrier_id_agree"] = np.where(play_df["carrier_gsis"].notna() & play_df["pbp_rusher_player_id"].notna(),
                                           play_df["carrier_gsis"] == play_df["pbp_rusher_player_id"], np.nan)

    # --- box-count grid from the participants table: select on TUNE_WEEKS, report on the rest ---
    dfd = part_df[part_df["side"] == "defense"]
    box_rows = []
    best_box = None
    for d, lat in cfg.box_grid:
        cnt = ((dfd["depth"] <= d) & (dfd["depth"] >= -1.0) & (dfd["lateral"].abs() <= lat)).groupby(
            [dfd["gameId"], dfd["playId"]]).sum().rename("cnt").reset_index()
        tmp = play_df[["gameId", "playId", "week", "defendersInTheBox"]].merge(cnt, on=["gameId", "playId"], how="left")
        st = grid_agreement(tmp["cnt"], tmp["defendersInTheBox"], tmp["week"],
                            f"box(depth<={d},|lat|<={lat})", "defendersInTheBox")
        st.update({"box_depth": d, "box_lateral": lat})
        box_rows.append(st)
        if best_box is None or np.nan_to_num(st["exact_tune"], nan=-1.0) > np.nan_to_num(best_box["exact_tune"], nan=-1.0):
            best_box = st
            play_df["box_count_tuned"] = tmp["cnt"].to_numpy()
    box_grid_df = pd.DataFrame(box_rows).sort_values("exact_tune", ascending=False)

    # --- pass-rusher grid (same protocol) ---
    rush_rows = []
    best_rush = None
    for w, c in cfg.rush_grid:
        col = tf.rush_col(w, c)
        st = grid_agreement(play_df[col], play_df["numberOfPassRushers"], play_df["week"], col, "numberOfPassRushers")
        st.update({"window_s": w, "cross_depth": c})
        rush_rows.append(st)
        if best_rush is None or np.nan_to_num(st["exact_tune"], nan=-1.0) > np.nan_to_num(best_rush["exact_tune"], nan=-1.0):
            best_rush = st
            play_df["n_pass_rushers_tuned"] = play_df[col]
    rush_grid_df = pd.DataFrame(rush_rows).sort_values("exact_tune", ascending=False)
    holdout = ~play_df["week"].isin(TUNE_WEEKS)

    # --- agreement tables ---
    agree = []
    agree.append(agreement_stats(play_df["box_count"], play_df["defendersInTheBox"], "box_count (naive)", "plays.defendersInTheBox"))
    agree.append(agreement_stats(play_df["box_count"], play_df["ngs_defenders_in_box"], "box_count (naive)", "participation.defenders_in_box"))
    agree.append(agreement_stats(play_df["box_count_tuned"], play_df["defendersInTheBox"], "box_count_tuned (all weeks, in-sample for weeks 1-3)", "plays.defendersInTheBox"))
    agree.append(agreement_stats(play_df.loc[holdout, "box_count_tuned"], play_df.loc[holdout, "defendersInTheBox"], "box_count_tuned (weeks 4-6 held out)", "plays.defendersInTheBox"))
    agree.append(agreement_stats(play_df["defendersInTheBox"], play_df["ngs_defenders_in_box"], "plays.defendersInTheBox", "participation.defenders_in_box"))
    agree.append(agreement_stats(play_df["n_pass_rushers_derived"], play_df["numberOfPassRushers"], "n_pass_rushers_derived (naive)", "plays.numberOfPassRushers"))
    agree.append(agreement_stats(play_df["n_pass_rushers_derived"], play_df["ngs_number_of_pass_rushers"], "n_pass_rushers_derived (naive)", "participation.number_of_pass_rushers"))
    agree.append(agreement_stats(play_df["n_pass_rushers_tuned"], play_df["numberOfPassRushers"], "n_pass_rushers_tuned (all weeks, in-sample for weeks 1-3)", "plays.numberOfPassRushers"))
    agree.append(agreement_stats(play_df.loc[holdout, "n_pass_rushers_tuned"], play_df.loc[holdout, "numberOfPassRushers"], "n_pass_rushers_tuned (weeks 4-6 held out)", "plays.numberOfPassRushers"))
    agree.append(agreement_stats(play_df.loc[holdout, "n_pass_rushers_derived"], play_df.loc[holdout, "numberOfPassRushers"], "n_pass_rushers_derived (naive, weeks 4-6)", "plays.numberOfPassRushers"))
    agree.append(agreement_stats(play_df["n_pass_rushers_tuned"], play_df["ngs_number_of_pass_rushers"], "n_pass_rushers_tuned", "participation.number_of_pass_rushers"))
    agree.append(agreement_stats(play_df["numberOfPassRushers"], play_df["ngs_number_of_pass_rushers"], "plays.numberOfPassRushers", "participation.number_of_pass_rushers"))
    for grp in ("RB", "TE", "WR"):
        po = play_df["personnel_offense"].map(lambda s: tf.parse_personnel(s).get(grp, np.nan))
        pp = play_df["ngs_offense_personnel"].map(lambda s: tf.parse_personnel(s).get(grp, np.nan))
        agree.append(agreement_stats(play_df[f"derived_{grp}"], po, f"derived_{grp}", f"plays.personnel_offense {grp}"))
        agree.append(agreement_stats(play_df[f"derived_{grp}"], pp, f"derived_{grp}", f"participation.offense_personnel {grp}"))
    for grp in ("DL", "LB", "DB"):
        po = play_df["personnel_defense"].map(lambda s: tf.parse_personnel(s).get(grp, np.nan))
        pp = play_df["ngs_defense_personnel"].map(lambda s: tf.parse_personnel(s).get(grp, np.nan))
        agree.append(agreement_stats(play_df[f"derived_{grp}"], po, f"derived_{grp}", f"plays.personnel_defense {grp}"))
        agree.append(agreement_stats(play_df[f"derived_{grp}"], pp, f"derived_{grp}", f"participation.defense_personnel {grp}"))
    po_dl = play_df["personnel_defense"].map(lambda s: tf.parse_personnel(s).get("DL", np.nan))
    agree.append(agreement_stats(play_df["n_dl"], po_dl, "n_dl (|depth|<=1.5; counts edge LBs too)", "plays.personnel_defense DL"))
    agree.append(agreement_stats(play_df["time_to_throw"], play_df["ngs_time_to_throw"], "time_to_throw (derived, s)", "participation.time_to_throw"))
    agree.append(agreement_stats(play_df["target_depth"], play_df["pbp_air_yards"], "target_depth (derived)", "pbp.air_yards"))
    agree.append(agreement_stats(play_df["ball_depth_at_arrival"], play_df["pbp_air_yards"], "ball_depth_at_arrival", "pbp.air_yards"))
    agree.append(agreement_stats(play_df["target_depth"], play_df["ngs_air_yards"], "target_depth (derived)", "participation.ngs_air_yards"))
    tgt_ok = play_df["target_id_agree"].astype(float) == 1.0
    agree.append(agreement_stats(play_df.loc[tgt_ok, "target_depth"], play_df.loc[tgt_ok, "ngs_air_yards"], "target_depth (derived, target == pbp receiver)", "participation.ngs_air_yards"))
    agree_df = pd.DataFrame(agree)

    # --- lateral sign convention: derived side vs nflfastR pass_location / run_location ---
    side_ct = confusion(play_df["pbp_pass_location"].rename("pbp_pass_location"),
                        play_df["target_side_derived"].rename("target_side_derived"))
    sgn = np.sign(play_df["target_lateral"]).map({1.0: "lateral>0", -1.0: "lateral<0", 0.0: "0"})
    sign_ct = confusion(play_df["pbp_pass_location"].rename("pbp_pass_location"), sgn.rename("sign(target_lateral)"))
    lr = play_df["pbp_pass_location"].isin(["left", "right"]) & play_df["target_lateral"].notna() & (play_df["target_lateral"] != 0)
    left_is_positive = ((play_df.loc[lr, "pbp_pass_location"] == "left") == (play_df.loc[lr, "target_lateral"] > 0))
    per_game = left_is_positive.groupby(play_df.loc[lr, "gameId"]).mean()
    rsgn = np.sign(play_df["carrier_lateral_at_handoff"]).map({1.0: "lateral>0", -1.0: "lateral<0", 0.0: "0"})
    run_sign_ct = confusion(play_df["pbp_run_location"].rename("pbp_run_location"), rsgn.rename("sign(carrier_lateral_at_handoff)"))
    m3 = play_df["pbp_pass_location"].notna() & play_df["target_side_derived"].notna()
    side_stats = {
        "pass_left_right_sign_agreement": float(left_is_positive.mean()),
        "n_pass_left_right": int(lr.sum()),
        "per_game_min": float(per_game.min()),
        "per_game_max": float(per_game.max()),
        "games": int(len(per_game)),
        "pass_3way_agreement_6yd_middle": float((play_df.loc[m3, "pbp_pass_location"] == play_df.loc[m3, "target_side_derived"]).mean()),
        "n_pass_3way": int(m3.sum()),
    }
    def _long(ct: pd.DataFrame, table: str) -> pd.DataFrame:
        out = ct.stack().rename("n").reset_index()
        out.columns = ["pbp_value", "derived_value", "n"]
        return out.assign(table=table)[["table", "pbp_value", "derived_value", "n"]]

    side_df = pd.concat([_long(side_ct, "pass_location_vs_target_side_derived"),
                         _long(sign_ct, "pass_location_vs_sign_target_lateral"),
                         _long(run_sign_ct, "run_location_vs_sign_carrier_lateral_at_handoff"),
                         pd.DataFrame([{"table": "summary", "pbp_value": k, "derived_value": "", "n": v}
                                       for k, v in side_stats.items()])], ignore_index=True)

    # --- front depths relative to the official line (participants table) ---
    def_front = part_df[part_df["side"] == "defense"].groupby(["gameId", "playId"])["depth"].min()
    ol_rows = part_df[(part_df["side"] == "offense") & (part_df["role_derived"] == "ol")]
    ol_front = ol_rows.groupby(["gameId", "playId"])["depth"].max()
    ol_all = ol_rows["depth"]
    front_df = pd.DataFrame({"quantity": ["defensive front (min defender depth per play)",
                                          "offensive line front (max OL depth per play, usually the centre)",
                                          "offensive linemen (all OL player-plays)"],
                             "n": [int(def_front.notna().sum()), int(ol_front.notna().sum()), int(ol_all.notna().sum())],
                             "median": [float(def_front.median()), float(ol_front.median()), float(ol_all.median())],
                             "p05": [float(def_front.quantile(0.05)), float(ol_front.quantile(0.05)), float(ol_all.quantile(0.05))],
                             "p95": [float(def_front.quantile(0.95)), float(ol_front.quantile(0.95)), float(ol_all.quantile(0.95))]})
    fronts = {"def_front": float(def_front.median()), "ol_front": float(ol_front.median()), "ol_all": float(ol_all.median())}

    # whole-string personnel agreement
    def _tuple(s: str | None, groups: tuple[str, ...]) -> tuple | None:
        d = tf.parse_personnel(s)
        return tuple(d.get(g, 0) for g in groups) if d else None

    off_groups, def_groups = ("RB", "TE", "WR"), ("DL", "LB", "DB")
    pers_rows = []
    for src, col in (("plays", "personnel_offense"), ("participation", "ngs_offense_personnel")):
        ref = play_df[col].map(lambda s: _tuple(s, off_groups))
        der = play_df["personnel_offense_derived"].map(lambda s: _tuple(s, off_groups))
        m = ref.notna() & der.notna()
        pers_rows.append({"side": "offense", "reference": src, "n": int(m.sum()),
                          "exact_RB_TE_WR": float((ref[m] == der[m]).mean())})
    for src, col in (("plays", "personnel_defense"), ("participation", "ngs_defense_personnel")):
        ref = play_df[col].map(lambda s: _tuple(s, def_groups))
        der = play_df["personnel_defense_derived"].map(lambda s: _tuple(s, def_groups))
        m = ref.notna() & der.notna()
        pers_rows.append({"side": "defense", "reference": src, "n": int(m.sum()),
                          "exact_DL_LB_DB": float((ref[m] == der[m]).mean())})
    pers_df = pd.DataFrame(pers_rows)

    shotgun_ct = confusion(play_df["shotgun_derived"].rename("shotgun_derived"),
                           play_df["pbp_shotgun"].rename("pbp_shotgun"))
    form_ct = confusion(play_df["shotgun_derived"].rename("shotgun_derived"),
                        play_df["offenseFormation"].rename("offenseFormation"))
    qb_agree = {
        "qb_position_vs_geometric": float((play_df["qb_id"] == play_df["qb_id_geometric"])[play_df["qb_id"].notna() & play_df["qb_id_geometric"].notna()].mean()),
        "qb_throw_vs_pbp_passer": float(pd.to_numeric(play_df["qb_id_agree"]).mean()),
        "n_qb_throw_vs_pbp_passer": int(pd.to_numeric(play_df["qb_id_agree"]).notna().sum()),
        "target_vs_pbp_receiver": float(pd.to_numeric(play_df["target_id_agree"]).mean()),
        "n_target_vs_pbp_receiver": int(pd.to_numeric(play_df["target_id_agree"]).notna().sum()),
        "carrier_vs_pbp_rusher": float(pd.to_numeric(play_df["carrier_id_agree"]).mean()),
        "n_carrier_vs_pbp_rusher": int(pd.to_numeric(play_df["carrier_id_agree"]).notna().sum()),
        "posteam_matches_possessionTeam": float((play_df["pbp_posteam"].replace({"LV": "OAK"}) == play_df["possessionTeam"]).mean()),
    }
    pressure = None
    if play_df["ngs_was_pressure"].notna().any():
        pressure = (play_df.assign(was_pressure=play_df["ngs_was_pressure"]).dropna(subset=["was_pressure"])
                    .groupby("was_pressure")[["min_def_dist_qb_throw", "min_def_dist_qb_throw_m05",
                                              "min_def_dist_qb_dropback", "n_def_within_r_qb_throw",
                                              "time_to_throw"]].agg(["mean", "median", "count"]))
        pressure.columns = ["_".join(c) for c in pressure.columns]
        pressure = pressure.reset_index()

    # --- counts ---
    plays_all = plays[plays["gameId"].isin(games["gameId"])]
    counts = {
        "games": int(len(games)),
        "plays_csv_total": int(len(plays_all)),
        "plays_csv_special_teams": int(plays_all["isSTPlay"].sum()),
        "plays_csv_non_st": int((~plays_all["isSTPlay"]).sum()),
        "tracking_plays_total": n_tracking_plays,
        "tracked_non_st_with_snap": int(len(play_df)),
        "skipped_no_ball_snap": int((skipped["reason"] == "no_ball_snap").sum()) if len(skipped) else 0,
        "skipped_no_ball_snap_with_snap_direct": int(skipped.get("has_snap_direct", pd.Series(dtype=bool)).fillna(False).sum()) if len(skipped) else 0,
        "skipped_not_in_plays_csv": int((skipped["reason"] == "not_in_plays_csv").sum()) if len(skipped) else 0,
        "pass_plays_C_I_IN_S": int(play_df["is_pass_play"].sum()),
        "plays_11_off_11_def_tracked": int(((play_df["n_off_tracked"] == 11) & (play_df["n_def_tracked"] == 11)).sum()),
        "plays_10_off_tracked": int((play_df["n_off_tracked"] == 10).sum()),
        "plays_10_def_tracked": int((play_df["n_def_tracked"] == 10).sum()),
        "plays_12_def_tracked": int((play_df["n_def_tracked"] == 12).sum()),
        "pbp_join_rate": float(play_df["pbp_joined"].mean()),
        "participation_join_rate": float(play_df["ngs_joined"].mean()),
        "crosswalk_players": int(len(xwalk)),
        "crosswalk_matched": int(xwalk["gsis_id"].notna().sum()),
        "participants_rows": int(len(part_df)),
        "participants_in_pbp_participation_rate": float(part_df["in_pbp_participation"].mean()),
        "build_seconds": float(time.time() - t_start),
    }
    by_type = (play_df.assign(PassResult=play_df["PassResult"].fillna("NA"),
                              pbp_play_type=play_df["pbp_play_type"].fillna("NA"))
               .groupby(["PassResult", "pbp_play_type"]).size().rename("n").reset_index())
    los_summary = play_df["los_ball_discrepancy"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_frame("los_ball_minus_official").T
    los_summary["count"] = los_summary["count"].astype(int)
    los_median = float(play_df["los_ball_discrepancy"].median())
    los_extra = pd.DataFrame({"stat": ["|discrepancy| > 1 yd", "|discrepancy| > 2 yd", "|discrepancy| > 5 yd", "ball_y_source == centre_estimate", "los_source == official_pbp", "los_source == ball"],
                              "n": [int((play_df["los_ball_discrepancy"].abs() > 1).sum()), int((play_df["los_ball_discrepancy"].abs() > 2).sum()),
                                    int((play_df["los_ball_discrepancy"].abs() > 5).sum()), int((play_df["ball_y_source"] == "centre_estimate").sum()),
                                    int((play_df["los_source"] == "official_pbp").sum()), int((play_df["los_source"] == "ball").sum())]})

    presnap_cols = ["n_deep_safeties", "def_depth_max", "def_depth_2nd", "def_depth_mean", "mof_open", "box_count", "box_count_tuned",
                    "n_dl", "def_y_std", "def_y_range", "n_wide_left", "n_wide_right", "widest_split", "n_backfield", "n_te_inline",
                    "qb_depth", "shotgun_derived", "cushion_left", "cushion_right", "cb_cushion", "motion_disp_max", "motion_event",
                    "motion_derived", "line_set_to_snap_s", "frames_before_snap"]
    pass_cols = ["time_to_throw", "qb_depth_at_throw", "qb_speed_at_throw", "min_def_dist_qb_throw", "min_def_dist_qb_throw_m05",
                 "min_def_dist_qb_dropback", "n_def_within_r_qb_throw", "n_pass_rushers_derived", "n_pass_rushers_tuned",
                 "separation_at_arrival", "separation_at_throw", "n_def_within_r_target", "target_depth", "target_lateral",
                 "target_speed_at_arrival", "target_ball_dist_at_arrival", "ball_depth_at_arrival", "air_time_s"]
    run_cols = ["time_to_handoff", "carrier_speed_at_handoff", "carrier_depth_at_handoff", "n_def_within_r_carrier_handoff",
                "min_def_dist_carrier_handoff", "yards_to_first_contact", "time_to_first_contact", "carrier_speed_at_first_contact",
                "n_def_within_r_carrier_first_contact"]
    summ_presnap = summarise_targets(play_df, presnap_cols)
    summ_pass = summarise_targets(play_df[play_df["is_pass_play"]], pass_cols)
    summ_run = summarise_targets(play_df[~play_df["is_pass_play"]], run_cols)
    ev_cols = [c for c in play_df.columns if c.startswith("ev_")]
    ev_rates = (play_df.groupby(play_df["PassResult"].fillna("run/other"))[ev_cols].mean().T
                .rename(columns=str).reset_index().rename(columns={"index": "event"}))

    # --- save ---
    play_df.to_parquet(out_dir / "plays_tracked.parquet", index=False)
    part_df.to_parquet(out_dir / "participants_tracked.parquet", index=False)
    xwalk.to_parquet(out_dir / "player_crosswalk.parquet", index=False)
    if len(skipped):
        skipped.to_parquet(out_dir / "skipped_plays.parquet", index=False)
    agree_df.to_parquet(rep_dir / "nfl_01_agreement.parquet", index=False)
    box_grid_df.to_parquet(rep_dir / "nfl_01_box_grid.parquet", index=False)
    rush_grid_df.to_parquet(rep_dir / "nfl_01_rush_grid.parquet", index=False)
    pd.concat([summ_presnap.assign(block="presnap"), summ_pass.assign(block="pass"), summ_run.assign(block="run")]
              ).to_parquet(rep_dir / "nfl_01_target_summary.parquet", index=False)
    pd.DataFrame([counts]).to_parquet(rep_dir / "nfl_01_counts.parquet", index=False)
    by_type.to_parquet(rep_dir / "nfl_01_play_types.parquet", index=False)
    coldefs = column_definitions_table(list(play_df.columns))
    coldefs.to_parquet(rep_dir / "nfl_01_column_definitions.parquet", index=False)
    side_df.to_parquet(rep_dir / "nfl_01_side_agreement.parquet", index=False)

    # --- report ---
    md = []
    md.append("# NFL 01 - build: tracking-derived targets for the 91 tracked 2017 games\n")
    md.append(f"Generated by `python -m research.privileged_tracking.nfl.build_tables` "
              f"({'limit_games=' + str(cfg.limit_games) if cfg.limit_games else 'all games'}), "
              f"build time {counts['build_seconds'] / 60:.1f} min.\n")
    md.append("Outputs: `data/raw/privileged/processed/nfl/plays_tracked.parquet` (one row per play), "
              "`participants_tracked.parquet` (one row per play x player), `player_crosswalk.parquet`, "
              "`skipped_plays.parquet`. Result tables next to this file: `nfl_01_*.parquet`.\n")
    md.append("## 1. Counts\n")
    md.append(tables_table(pd.DataFrame({"quantity": list(counts), "value": [f"{v:.4f}" if isinstance(v, float) else v for v in counts.values()]})))
    md.append("\n\nPlays by BDB `PassResult` x nflfastR `play_type` (tracked, non-ST, with a `ball_snap` tag):\n")
    md.append(tables_table(by_type))
    md.append("\n\nSkipped plays: every non-special-teams play in `plays.csv` has tracking rows; the only "
              "skips are plays with no `ball_snap` tag (direct snaps carry `snap_direct` instead, kneel-downs "
              "and aborted plays sometimes carry no tag). Special-teams plays are dropped before anything else. "
              f"Tracked players per play: 11 v 11 on {counts['plays_11_off_11_def_tracked']} plays, 10 offensive players on "
              f"{counts['plays_10_off_tracked']}, 10 defenders on {counts['plays_10_def_tracked']}, 12 defenders on "
              f"{counts['plays_12_def_tracked']} (tracking dropouts / a penalty-flagged extra man; nothing is imputed for them).\n")
    md.append("\nEvent-tag availability by play type (share of plays that carry the tag anywhere):\n")
    md.append(tables_table(ev_rates))

    md.append("\n\n## 2. Joins\n")
    md.append(f"* nflfastR `play_by_play_2017` on (`old_game_id` as int == gameId, `play_id` == playId): join rate {counts['pbp_join_rate']:.4f}.\n"
              f"* nflverse `pbp_participation_2017` on the same keys: join rate {counts['participation_join_rate']:.4f}.\n"
              f"* Player crosswalk BDB nflId -> gsis_id: {counts['crosswalk_matched']} / {counts['crosswalk_players']} tracked players "
              f"({counts['crosswalk_matched'] / max(counts['crosswalk_players'], 1):.4f}). The BDB `nflId` matches no id column in any nflverse file "
              "(Brady is 2504211 in BDB, `nfl_id` 25511 in nflverse), so the map goes through `roster_weekly_2017` team + name, "
              "with jersey number as tie-break.\n")
    md.append(tables_table(xwalk["match_method"].value_counts().rename("n").reset_index().rename(columns={"index": "match_method", "match_method": "match_method"})))
    md.append(f"\n\nShare of tracked player-plays whose gsis_id appears in the participation `offense_players`/`defense_players` list: "
              f"{counts['participants_in_pbp_participation_rate']:.4f}.\n")
    md.append("\nIdentity cross-checks (rate of agreement, n):\n")
    md.append(tables_table(pd.DataFrame({"check": list(qb_agree), "value": [f"{v:.4f}" if isinstance(v, float) else v for v in qb_agree.values()]})))

    md.append("\n\n## 3. Orientation and line of scrimmage\n")
    md.append("Attack direction = sign of (median defender x - median offense x) at the snap; a ball-relative rule "
              "fails on ~4% of plays because the tracked ball often sits behind the centre. LOS used for all depths "
              "is the official yard line (10 + own yard line, 110 - opponent yard line, 60 at midfield); the ball's x is "
              f"recorded as `los_x_ball` and compared below (median ball - official = {los_median:.2f} yd, i.e. the tracked "
              "ball sits ~0.5 yd behind the official line). The measured fronts relative to the official line (participants "
              f"table, snap frame) are given in the second table: the defensive front at {fronts['def_front']:+.2f} yd, the most "
              f"forward lineman (centre) at {fronts['ol_front']:+.2f} yd and the offensive linemen as a whole at {fronts['ol_all']:+.2f} yd "
              "(median). The line therefore sits about 1 yd behind the official line, which is why the naive 'behind the ball' "
              "definitions were re-based on the official line.\n")
    md.append(tables_table(los_summary.reset_index().rename(columns={"index": "stat"})))
    md.append("\n")
    md.append(tables_table(front_df))
    md.append("\n")
    md.append(tables_table(los_extra))
    md.append(f"\n\nFlipped plays: {int(play_df['flipped'].sum())} of {len(play_df)}.\n")
    md.append("\n**Lateral sign convention.** `lateral = y - ball_y_ref` in the normalised frame; positive lateral is the "
              "offense's LEFT (a player at larger y stands to the left of a QB facing +x). All `*_left` / `*_right` columns "
              "(`n_wide_left`, `cushion_left`, `target_side_derived`) use the offense's perspective, so they line up with "
              "nflfastR `pass_location` / `run_location`. Check on the real data (section 4f): nflfastR `pass_location == 'left'` "
              f"<-> `target_lateral > 0` on {side_stats['pass_left_right_sign_agreement']:.4f} of {side_stats['n_pass_left_right']} "
              f"left/right passes (per-game min {side_stats['per_game_min']:.3f}, max {side_stats['per_game_max']:.3f} over "
              f"{side_stats['games']} games).\n")

    md.append("\n## 4. Agreement with charted values\n")
    md.append(f"Exact = share of plays with |a - b| <= {EXACT_TOL:g} (a tolerance, not bit equality: mirrored plays go through "
              "120 - x and pick up 1e-14 rounding); within_1 = |diff| <= 1; bias = mean(a - b). The `*_tuned` rows are "
              f"definitions selected on weeks {TUNE_WEEKS} (sections 4a / 4b); their 'all weeks' rows are in-sample for those weeks, "
              "the 'weeks 4-6 held out' rows are not. `participation.ngs_air_yards` is numerically the receiver's x - LOS at "
              "arrival in the same tracking, so once the derived target is the pbp receiver the agreement is essentially exact "
              "(last row); the residual disagreement is receiver misidentification (mostly incompletions without a `pass_arrived` tag).\n")
    md.append(tables_table(agree_df))
    md.append(f"\n\n### 4a. Box-count definition grid (vs plays.defendersInTheBox; selected on weeks {TUNE_WEEKS}, reported on weeks 4-6)\n")
    md.append("`exact_tune` / `n_tune` are the selection weeks (in-sample); `exact`, `within_1`, `mae`, `bias` and `n` are the held-out weeks; `exact_all` is all weeks.\n")
    md.append(tables_table(box_grid_df[["box_depth", "box_lateral", "n_tune", "exact_tune", "n", "exact", "within_1", "mae", "bias_a_minus_b", "exact_all"]]))
    md.append(f"\n\nSelected `box_count_tuned`: depth <= {best_box['box_depth']}, |lateral| <= {best_box['box_lateral']} "
              f"(tuning exact {best_box['exact_tune']:.3f}, n={best_box['n_tune']}; held-out exact {best_box['exact']:.3f}, "
              f"within 1 {best_box['within_1']:.3f}, n={best_box['n']}). All definitions also require depth >= -1 (not lined up in the offensive backfield).\n")
    md.append(f"\n### 4b. Pass-rusher definition grid (vs plays.numberOfPassRushers; selected on weeks {TUNE_WEEKS}, reported on weeks 4-6)\n")
    md.append(tables_table(rush_grid_df[["window_s", "cross_depth", "n_tune", "exact_tune", "n", "exact", "within_1", "mae", "bias_a_minus_b", "exact_all"]]))
    naive_hold = next(r for r in agree if r["a"].startswith("n_pass_rushers_derived (naive, weeks 4-6)"))
    naive_hold = {"exact": np.nan, "within_1": np.nan, **naive_hold}
    md.append(f"\n\nSelected `n_pass_rushers_tuned`: window {best_rush['window_s']} s, crossing depth {best_rush['cross_depth']} yd "
              f"(tuning exact {best_rush['exact_tune']:.3f}, n={best_rush['n_tune']}; held-out exact {best_rush['exact']:.3f}, "
              f"within 1 {best_rush['within_1']:.3f}, n={best_rush['n']}). The naive spec definition (1.5 s, crossing the line) "
              f"has held-out exact {naive_hold['exact']:.3f} and within 1 {naive_hold['within_1']:.3f}: the tuned rule is a marginal "
              "gain on exact and no better on within-1, so pass-rusher counts stay a soft target with either definition.\n")
    md.append("\n### 4c. Personnel strings (derived from players.csv positions of the tracked 22)\n")
    md.append(tables_table(pers_df))
    md.append("\n\n### 4d. Shotgun: derived (qb_depth >= 4) vs nflfastR `shotgun`\n")
    md.append(tables_table(shotgun_ct, index=True))
    md.append("\n\nDerived shotgun vs BDB `offenseFormation`:\n")
    md.append(tables_table(form_ct, index=True))
    if pressure is not None:
        md.append("\n\n### 4e. Pressure proxies vs participation `was_pressure` (pass plays with the flag)\n")
        md.append(tables_table(pressure))
    md.append("\n\n### 4f. Lateral sign convention: derived side vs nflfastR `pass_location` / `run_location`\n")
    md.append(f"Left / right (sign only) agreement {side_stats['pass_left_right_sign_agreement']:.4f} (n={side_stats['n_pass_left_right']}); "
              f"three-way agreement with a 6 yd middle half-width {side_stats['pass_3way_agreement_6yd_middle']:.4f} (n={side_stats['n_pass_3way']}; "
              "the half-width was read off the quartiles of pbp 'middle' targets, so treat it as a definition, not a validated result).\n")
    md.append(tables_table(side_ct, index=True))
    md.append("\n\n`pass_location` vs sign of `target_lateral`:\n")
    md.append(tables_table(sign_ct, index=True))
    md.append("\n\n`run_location` vs sign of `carrier_lateral_at_handoff` (weaker: run_location describes the designed gap, the carrier at the handoff is still in the backfield):\n")
    md.append(tables_table(run_sign_ct, index=True))

    md.append("\n\n## 5. Target distributions\n")
    md.append("### Pre-snap (all tracked plays)\n")
    md.append(tables_table(summ_presnap))
    md.append("\n\n### Pass plays (PassResult in C/I/IN/S)\n")
    md.append(tables_table(summ_pass))
    md.append("\n\n### Run / other plays (everything else, incl. scrambles `R`, kneels, spikes, no-plays)\n")
    md.append(tables_table(summ_run))

    md.append("\n\n## 6. Column contract and definitions (`plays_tracked.parquet`)\n")
    md.append("### 6a. Leakage tiers\n")
    md.append("Every column carries a tier in `nfl_01_column_definitions.parquet` (`build_tables.column_tier(name)`; "
              "`build_tables.event_only_presnap_columns(cols)` returns the admissible feature set). Rule: a pre-snap state "
              "model may use only `id` and `event_only_presnap` columns as features; everything else is a label, an oracle or "
              "a post-snap outcome. The nflverse participation columns are published as `ngs_*` (not `part_*`) because every "
              "one of them is NGS charting from the same tracking; the BDB charting columns keep their plays.csv names "
              "(`offenseFormation`, `personnel_offense`, `personnel_defense`, `defendersInTheBox`, `numberOfPassRushers`) and are "
              "tiered `ngs_charting_privileged` too. `pbp_*` mixes tiers: the situation / pre-snap model columns are "
              "`event_only_presnap`, outcomes and post-snap models `event_only_postsnap` (listed below).\n")
    tier_counts = coldefs.groupby("tier").size().rename("n_columns").reset_index()
    tier_counts["meaning"] = tier_counts["tier"].map(TIERS)
    md.append(tables_table(tier_counts))
    md.append("\n")
    for tier in ("id", "event_only_presnap", "event_only_postsnap", "ngs_charting_privileged", "tracking_context"):
        cols_t = coldefs.loc[coldefs["tier"] == tier, "column"].tolist()
        md.append(f"\n* `{tier}` ({len(cols_t)}): " + ", ".join(f"`{c}`" for c in cols_t))
    n_t = int((coldefs["tier"] == "tracking_target").sum())
    md.append(f"\n* `tracking_target` ({n_t}): everything else (pre-snap, pass and run targets defined below).\n")
    md.append("\n### 6b. Definitions\n")
    md.append("All distances in yards, times in seconds (frames / 10), coordinates offense-normalised. "
              "`depth = x - los_x` (defenders positive), `lateral = y - ball_y_ref` (positive = offense's left).\n")
    md.append(tables_table(pd.DataFrame({"column": list(COLUMN_DEFS), "definition": list(COLUMN_DEFS.values())})))
    md.append("\n\n`participants_tracked.parquet` columns: gameId, playId, nflId, gsis_id, match_method, displayName, "
              "jerseyNumber, team_abbr, side (offense/defense), position (players.csv), role_derived (offense: qb / ol / "
              "backfield / wide / tight_slot; defense: line / box / deep / second_level), is_qb_used, in_pbp_participation "
              "(gsis_id listed for that play in participation), x_snap, y_snap, s_snap, depth, lateral (positive = offense's left), "
              "in_box_naive. Rows are sorted by (gameId, playId, nflId); plays by (gameId, playId).\n")
    md.append("\n## 7. Gotchas discovered\n")
    md.append("\n".join([
        "* The tracking `time` column has 1-second resolution; `frame.id` is the 10 Hz clock (10 frames per second, verified). All durations use frames / 10.",
        "* Event tags are written on every row (all 22 players and the ball) of the tagged frame, but the ball row is sometimes untagged; select the frame by id rather than by the ball's own tag.",
        "* `dir` is clockwise from +y (dx = sin, dy = cos), so mirroring a play rotates it by 180 degrees.",
        f"* The tracked ball at the snap sits ~0.5 yd behind the official line (median {los_median:.2f} yd) and has occasional 5-60 yd glitches (ball_y out of bounds); the official yard line is used for depth and the ball only for lateral reference with a fallback.",
        "* The `pass_forward` tag is ~2 frames (0.2 s) late relative to the ball leaving the hand: at the tagged frame the ball is already ~3 yd from the passer moving at 13 yd/s, so 'nearest player to the ball at pass_forward' returns a lineman on ~50% of plays. The passer is identified 3 frames earlier instead (`release_lag_frames`; nearest player is the roster QB on 95% of plays at lags 3-4, 91% at lag 2, 55% at lag 0; the rest are ball-track glitches); time_to_throw still uses the tag (it matches NGS time_to_throw to 0.03 s).",
        f"* Relative to the official line the offensive linemen sit at {fronts['ol_all']:+.2f} yd (median; the centre at {fronts['ol_front']:+.2f}), so a geometric QB rule with a 1 yd minimum depth returns a guard; the rule now takes the second most-forward player within 0.8 yd of the ball laterally.",
        "* Direct snaps (`snap_direct`) have no `ball_snap` tag and are skipped per the stage spec; kneels and spikes are tracked but carry few tags.",
        "* BDB `nflId` is not the nflverse `nfl_id`; the crosswalk is team + name (jersey tie-break) via roster_weekly_2017.",
        "* nflfastR uses `LV` for the 2017 Raiders in some columns; team-name comparisons normalise LV -> OAK.",
        "* `plays.defendersInTheBox` / participation `defenders_in_box` and `numberOfPassRushers` / participation `number_of_pass_rushers` are the same NGS charting (section 4: 0.999 and 0.998 exact on the tracked plays); the two files are interchangeable as labels for this season. Pass-rusher counts are nevertheless a soft target: the best geometric rule agrees exactly on ~80% of plays.",
        "* Positive `lateral` is the offense's left (section 3 / 4f). An earlier draft of this stage had `n_wide_left`/`n_wide_right` and `cushion_left`/`cushion_right` mirrored; they now follow the offense's (and nflfastR's) perspective.",
        "* Tracked-player counts are not always 11 v 11 (see section 1); `n_off_tracked` / `n_def_tracked` record them and no player is imputed.",
        "* PassResult `R` (252 plays) are pass plays that became runs (scrambles); they have no `pass_forward` and are treated with the run targets.",
        "* `pass_arrived` is missing on ~1/3 of incompletions; the arrival fallback uses `pass_outcome_*` tags and `arrival_event` records which one was used.",
        "* Coverage columns (`defense_man_zone_type`, `defense_coverage_type`) are empty for 2017 in participation.",
    ]))
    md.append("\n")
    (rep_dir / "nfl_01_build.md").write_text("\n".join(md))
    print(f"wrote {rep_dir / 'nfl_01_build.md'}; plays={len(play_df)} participants={len(part_df)} in {counts['build_seconds'] / 60:.1f} min")
    return {"counts": counts, "agreement": agree_df, "plays": play_df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    run(BuildConfig(n_jobs=a.n_jobs, limit_games=a.limit_games))


if __name__ == "__main__":
    main()
