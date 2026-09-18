"""SOCCER 01 BUILD: per-event 360 defensive-state targets + event-only features.

Outputs (under ``processed_dir('soccer')``):

* ``events360.parquet``    one row per event that has a 360 freeze frame (417 matches with
                           usable frames): ids, ``y_*`` targets from the frame (after the
                           per-row orientation check / repair of StatsBomb's paired-event
                           frames, see ``frame_features.resolve_frame_orientation``),
                           ``f_*`` event-only features (``f_after_*`` = post-instant
                           attributes), ``sff_*`` shot-freeze-frame geometry for shots,
                           ``seq_*`` fixed-width history block.
* ``events_no360.parquet`` same columns (``y_*`` all null) for every Shot plus a fixed
                           random fraction of Pass / Carry events in matches without
                           360 (2015/16 leagues, WC 2018, Copa America 2024, AFCON).

Plus ``reports/soccer_01_build.md`` and parquet result tables next to it.

Run from the repo root::

    python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from research.privileged_tracking.common.io import data_dir, processed_dir, reports_dir, sb_dir
from research.privileged_tracking.soccer.event_features import (
    EventFeatureConfig,
    compute_event_features,
    seq_columns,
    type_vocab_rows,
)
from research.privileged_tracking.soccer.frame_features import (
    ORIENTATIONS,
    FrameOrientationConfig,
    actor_event_distances,
    frame_coord_key,
    frame_targets,
    resolve_frame_orientation,
    shot_frame_features,
)

REPORT_STEM = "soccer_01"


@dataclass(frozen=True)
class BuildConfig:
    """Driver settings.

    Attributes:
        workers: process-pool size (keep at 2 on the shared 4-CPU box).
        no360_pass_carry_frac: fraction of Pass / Carry events kept per non-360 match
            (all Shots are always kept); ``1.0`` keeps everything.
        seed: base seed for the per-match subsampling RNG.
        limit: optional cap on matches per group (360 / non-360) for smoke runs.
        flush_rows: buffered rows before a parquet row group is written.
        no360_types: event types emitted for matches without 360.
        min_matched_frames: a 360-flagged match with fewer frames joined to its events
            is treated as a non-360 match (one AFCON 2023 match ships a single frame).
    """

    workers: int = 2
    no360_pass_carry_frac: float = 0.25
    seed: int = 0
    limit: int | None = None
    flush_rows: int = 150_000
    no360_types: tuple[str, ...] = ("Shot", "Pass", "Carry")
    min_matched_frames: int = 500


@dataclass(frozen=True)
class MatchJob:
    """Everything a worker needs to process one match."""

    match_id: int
    competition: str
    competition_id: int
    season: str
    season_id: int
    gender: str
    home_team_id: int
    away_team_id: int
    match_date: str
    match_week: int
    has_360: bool
    sb_root: str
    frac: float
    seed: int
    no360_types: tuple[str, ...]
    min_matched_frames: int = 500


# ----------------------------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------------------------
ID_COLS: tuple[str, ...] = (
    "match_id", "event_id", "event_index", "period", "team_id", "opp_team_id",
    "possession_team_id", "player_id", "competition", "competition_id", "season", "season_id",
    "gender", "match_date", "match_week", "home_team_id", "away_team_id", "has_360",
)
STRING_COLS = frozenset({
    "event_id", "competition", "season", "gender", "match_date", "f_type", "f_play_pattern",
    "f_position", "f_poss_start_type", "f_after_pass_height", "f_pass_type",
    "f_pass_body_part", "post_pass_outcome", "f_shot_body_part", "f_shot_technique",
    "f_shot_type", "post_shot_outcome", "y_frame_orientation", "y_frame_method",
})
BOOL_COLS = frozenset({
    "has_360", "f_has_location", "f_ts_repaired", "f_under_pressure", "f_counterpress", "f_home",
    "f_is_possession_team", "f_poss_in_final_third", "f_after_pass_switch",
    "f_after_pass_cross", "f_after_pass_through_ball", "f_after_pass_cut_back",
    "f_shot_first_time", "f_opp_poss_last10s",
})
POST_INSTANT_PREFIX = "f_after_"
INT64_COLS = frozenset({"match_id", "team_id", "opp_team_id", "possession_team_id", "player_id",
                        "home_team_id", "away_team_id"})
INT32_COLS = frozenset({
    "event_index", "period", "competition_id", "season_id", "match_week", "f_type_id",
    "f_period", "f_score_for", "f_score_against", "f_score_diff", "f_goals_total", "f_poss_idx",
    "f_poss_n_events", "f_poss_n_passes", "f_opp_def_n_60s",
})


def _probe_feature_columns() -> list[str]:
    """Canonical ``f_/post_/oracle_`` column order from a one-event probe."""
    probe = [{
        "id": "p", "index": 1, "period": 1, "timestamp": "00:00:01.000", "minute": 0,
        "second": 1, "type": {"name": "Pass"}, "possession": 1, "possession_team": {"id": 1},
        "team": {"id": 1}, "play_pattern": {"name": "Regular Play"}, "location": [60.0, 40.0],
        "pass": {"length": 1.0, "angle": 0.0, "end_location": [61.0, 40.0]},
    }]
    rows, _ = compute_event_features(probe, home_team_id=1)
    return [k for k in rows[0] if k not in ID_COLS]


def _probe_target_columns() -> list[str]:
    return list(frame_targets([], None, np.array([60.0, 40.0]), np.array([61.0, 40.0])).keys())


def _probe_sff_columns() -> list[str]:
    return list(shot_frame_features(None, np.array([60.0, 40.0])).keys())


def output_columns(cfg: EventFeatureConfig | None = None) -> list[str]:
    """Ordered column list shared by both output tables."""
    cfg = cfg or EventFeatureConfig()
    return (list(ID_COLS) + _probe_feature_columns() + _probe_target_columns()
            + _probe_sff_columns() + seq_columns(cfg.seq_len))


def _pa_type(col: str) -> pa.DataType:
    if col in STRING_COLS:
        return pa.string()
    if col in BOOL_COLS:
        return pa.bool_()
    if col in INT64_COLS:
        return pa.int64()
    if col in INT32_COLS or col.startswith("f_w"):
        return pa.int32()
    if col.startswith("seq_type_"):
        return pa.int16()
    if col.startswith("seq_same_"):
        return pa.int8()
    return pa.float32()


def output_schema(cfg: EventFeatureConfig | None = None) -> pa.Schema:
    """pyarrow schema shared by ``events360`` and ``events_no360``."""
    return pa.schema([pa.field(c, _pa_type(c)) for c in output_columns(cfg)])


# ----------------------------------------------------------------------------------------
# per-match worker
# ----------------------------------------------------------------------------------------
def _load_json(path: Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skipping {path.name}: {exc}", file=sys.stderr)
        return None


def _ball_xy(row: dict[str, Any]) -> np.ndarray | None:
    return np.array([row["f_x"], row["f_y"]]) if row["f_has_location"] else None


def _pass_end_xy(row: dict[str, Any]) -> np.ndarray | None:
    if row["f_type"] == "Pass" and not math.isnan(row["f_after_pass_end_x"]):
        return np.array([row["f_after_pass_end_x"], row["f_after_pass_end_y"]])
    return None


def process_match(job: MatchJob) -> tuple[pa.Table | None, dict[str, Any]]:
    """Build the output rows for one match.

    Returns:
        ``(table, stats)``; ``table`` is ``None`` when the match yields no rows.
    """
    t0 = time.time()
    root = Path(job.sb_root)
    stats: dict[str, Any] = {"match_id": job.match_id, "competition": job.competition,
                             "season": job.season, "gender": job.gender, "has_360": job.has_360,
                             "has_360_meta": job.has_360,
                             "n_events_total": 0, "n_frames": 0, "n_rows": 0, "n_shots": 0,
                             "n_pass": 0, "n_carry": 0, "n_frames_unmatched": 0,
                             "n_frames_no_location": 0, "error": None, "seconds": 0.0}
    for o in ORIENTATIONS:
        stats[f"n_frame_{o}"] = 0
    stats["n_frame_method_anchor"] = 0
    stats["n_frame_method_keeper"] = 0
    events = _load_json(root / "events" / f"{job.match_id}.json")
    if events is None:
        stats["error"] = "events_missing"
        return None, stats
    events.sort(key=lambda e: e["index"])
    stats["n_events_total"] = len(events)
    by_id = {e["id"]: e for e in events}
    ecfg = EventFeatureConfig()
    rows, seq = compute_event_features(events, job.home_team_id, ecfg)
    if not rows:
        return None, stats

    keep: list[int] = []
    extra: list[dict[str, float]] = []
    use_360 = job.has_360
    if job.has_360:
        frames = _load_json(root / "three-sixty" / f"{job.match_id}.json")
        if frames is None:
            # corrupt / missing 360 file: fall back to the event-only (non-360) output
            stats["error"] = "frames_missing"
            use_360 = False
        else:
            fmap = {f["event_uuid"]: f for f in frames}
            stats["n_frames"] = len(frames)
            stats["n_frames_unmatched"] = sum(1 for u in fmap if u not in by_id)
            n_matched = len(fmap) - stats["n_frames_unmatched"]
            if n_matched == 0:
                # 360 file and events file are out of sync (no uuid matches): no usable frames
                stats["error"] = "frames_unmatched"
                use_360 = False
            elif n_matched < job.min_matched_frames:
                stats["error"] = "frames_too_few"
                use_360 = False
    stats["has_360"] = use_360
    if use_360:
        ocfg = FrameOrientationConfig()
        # Pre-pass: group frames by their (order-free) coordinates and note, per frame,
        # how far the flagged actor is from the event location.  A frame whose actor sits
        # on its own event location is an "anchor" that fixes the flag perspective of any
        # other frame with identical coordinates (StatsBomb copies the frame of one event
        # of a pair onto the other, in the first team's orientation).
        key_of: dict[int, tuple[tuple[float, float], ...]] = {}
        groups: dict[tuple[tuple[float, float], ...], list[int]] = {}
        d_raw: dict[int, float] = {}
        for i, r in enumerate(rows):
            fr = fmap.get(r["event_id"])
            if fr is None:
                continue
            ff = fr.get("freeze_frame") or []
            k = frame_coord_key(ff, ocfg.coord_ndigits)
            key_of[i] = k
            groups.setdefault(k, []).append(i)
            d_raw[i] = actor_event_distances(ff, _ball_xy(r))[0]
        for i, r in enumerate(rows):
            fr = fmap.get(r["event_id"])
            if fr is None:
                continue
            ball = _ball_xy(r)
            if ball is None:
                stats["n_frames_no_location"] += 1
            ff = fr.get("freeze_frame")
            anchors = [(fmap[rows[j]["event_id"]].get("freeze_frame") or [],
                        rows[j]["team_id"] != r["team_id"])
                       for j in groups[key_of[i]]
                       if j != i and d_raw[j] <= ocfg.tol_ok]
            res = resolve_frame_orientation(ff, ball, anchors, ocfg)
            stats[f"n_frame_{res.orientation}"] += 1
            if res.method:
                stats[f"n_frame_method_{res.method}"] += 1
            keep.append(i)
            extra.append(frame_targets(ff, fr.get("visible_area"), ball, _pass_end_xy(r),
                                       resolution=res))
    else:
        rng = np.random.default_rng(job.seed * 1_000_003 + job.match_id)
        for i, r in enumerate(rows):
            t = r["f_type"]
            if t not in job.no360_types:
                continue
            if t != "Shot" and rng.random() >= job.frac:
                continue
            keep.append(i)
            extra.append({})
    if not keep:
        return None, stats

    for i, ex in zip(keep, extra, strict=True):
        r = rows[i]
        if r["f_type"] == "Shot":
            sff = (by_id[r["event_id"]].get("shot") or {}).get("freeze_frame")
            ex.update(shot_frame_features(sff, _ball_xy(r)))

    df = pd.DataFrame([rows[i] for i in keep])
    if extra and any(extra):
        df = pd.concat([df, pd.DataFrame(extra)], axis=1)
    seq_df = pd.DataFrame(seq[keep], columns=seq_columns(ecfg.seq_len))
    df = pd.concat([df.reset_index(drop=True), seq_df], axis=1)
    for c in seq_df.columns:
        if c.startswith("seq_type_"):
            df[c] = df[c].astype(np.int16)
        elif c.startswith("seq_same_"):
            df[c] = df[c].astype(np.int8)
    df["match_id"] = job.match_id
    df["competition"] = job.competition
    df["competition_id"] = job.competition_id
    df["season"] = job.season
    df["season_id"] = job.season_id
    df["gender"] = job.gender
    df["match_date"] = job.match_date
    df["match_week"] = job.match_week
    df["home_team_id"] = job.home_team_id
    df["away_team_id"] = job.away_team_id
    df["has_360"] = use_360
    df["player_id"] = pd.array(df["player_id"], dtype="Int64")

    schema = output_schema(ecfg)
    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame({c: [None] * len(df) for c in missing})], axis=1)
    table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)

    stats["n_rows"] = len(df)
    tc = df["f_type"].value_counts()
    stats["n_shots"] = int(tc.get("Shot", 0))
    stats["n_pass"] = int(tc.get("Pass", 0))
    stats["n_carry"] = int(tc.get("Carry", 0))
    stats["seconds"] = time.time() - t0
    return table, stats


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------
def make_jobs(cfg: BuildConfig) -> list[MatchJob]:
    """Match list from ``processed/matches.parquet`` + gender from ``competitions.json``."""
    root = sb_dir()
    m = pd.read_parquet(root / "processed" / "matches.parquet")
    comps_path = data_dir() / "competitions.json"
    gender: dict[tuple[int, int], str] = {}
    if comps_path.exists():
        with open(comps_path) as f:
            comps = json.load(f)
        for c in comps:
            gender[(int(c["competition_id"]), int(c["season_id"]))] = c["competition_gender"]
    jobs: list[MatchJob] = []
    for has_360 in (True, False):
        sub = m[m.has_360 == has_360].sort_values("match_id")
        if cfg.limit is not None:
            sub = sub.head(cfg.limit)
        for r in sub.itertuples(index=False):
            jobs.append(MatchJob(
                match_id=int(r.match_id), competition=str(r.competition),
                competition_id=int(r.competition_id), season=str(r.season),
                season_id=int(r.season_id),
                gender=gender.get((int(r.competition_id), int(r.season_id)), "unknown"),
                home_team_id=int(r.home_team_id), away_team_id=int(r.away_team_id),
                match_date=str(r.date), match_week=int(r.match_week), has_360=bool(has_360),
                sb_root=str(root), frac=cfg.no360_pass_carry_frac, seed=cfg.seed,
                no360_types=cfg.no360_types, min_matched_frames=cfg.min_matched_frames))
    return jobs


def build(cfg: BuildConfig) -> pd.DataFrame:
    """Run every match through :func:`process_match` and stream the two parquet files."""
    t0 = time.time()
    jobs = make_jobs(cfg)
    schema = output_schema()
    out = processed_dir("soccer")
    paths = {True: out / "events360.parquet", False: out / "events_no360.parquet"}
    writers = {k: pq.ParquetWriter(p, schema, compression="zstd") for k, p in paths.items()}
    buffers: dict[bool, list[pa.Table]] = {True: [], False: []}
    buffered = {True: 0, False: 0}
    stats: list[dict[str, Any]] = []
    print(f"{len(jobs)} matches ({sum(j.has_360 for j in jobs)} with 360), "
          f"workers={cfg.workers}, pass/carry frac (no 360)={cfg.no360_pass_carry_frac}",
          file=sys.stderr)

    def flush(k: bool) -> None:
        if buffers[k]:
            writers[k].write_table(pa.concat_tables(buffers[k]))
            buffers[k].clear()
            buffered[k] = 0

    try:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            for i, (tbl, st) in enumerate(ex.map(process_match, jobs, chunksize=2)):
                stats.append(st)
                if tbl is not None:
                    k = bool(st["has_360"])
                    buffers[k].append(tbl)
                    buffered[k] += tbl.num_rows
                    if buffered[k] >= cfg.flush_rows:
                        flush(k)
                if (i + 1) % 100 == 0:
                    done = sum(s["n_rows"] for s in stats)
                    print(f"  {i + 1}/{len(jobs)} matches, {done:,} rows, "
                          f"{time.time() - t0:.0f}s", file=sys.stderr)
        for k in (True, False):
            flush(k)
    finally:
        for w in writers.values():
            w.close()
    sdf = pd.DataFrame(stats)
    sdf["wall_seconds_total"] = time.time() - t0
    sdf.to_parquet(reports_dir() / f"{REPORT_STEM}_match_stats.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s: {paths[True]} ({sdf[sdf.has_360].n_rows.sum():,} "
          f"rows), {paths[False]} ({sdf[~sdf.has_360].n_rows.sum():,} rows)", file=sys.stderr)
    return sdf


# ----------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------
COLUMN_DOCS: dict[str, str] = {
    "match_id": "StatsBomb match id (group key for all splits)",
    "event_id": "StatsBomb event uuid", "event_index": "event order within the match",
    "period": "1-2 regulation, 3-4 extra time (period 5 shoot-out dropped)",
    "team_id": "team performing the event", "opp_team_id": "the other team",
    "possession_team_id": "team in possession per StatsBomb", "player_id": "actor",
    "competition": "competition name", "competition_id": "StatsBomb competition id",
    "season": "season name", "season_id": "StatsBomb season id",
    "gender": "competition gender (male/female)", "match_date": "kick-off date (YYYY-MM-DD)",
    "match_week": "StatsBomb match week", "home_team_id": "home team", "away_team_id": "away team",
    "has_360": "match has usable 360 frames (True for every events360 row; False in "
               "events_no360, including 360-flagged matches whose frames were unusable)",
    "f_type": "event type name", "f_type_id": "compact integer type id (see type vocab)",
    "f_x": "event x (event team's attacking frame, goal at 120)", "f_y": "event y (0-80)",
    "f_dist_goal": "distance from event location to goal centre (120, 40)",
    "f_goal_opening": "angle subtended by the posts from the event location (rad)",
    "f_goal_bearing": "atan2(y-40, 120-x): 0 = straight at goal (rad)",
    "f_has_location": "event carries a location",
    "f_play_pattern": "StatsBomb play pattern of the possession",
    "f_position": "actor's position name", "f_under_pressure": "StatsBomb under_pressure flag",
    "f_counterpress": "StatsBomb counterpress flag",
    "f_after_duration": "POST-INSTANT: event duration (s): for a pass/carry the time until "
                        "the ball arrives / the carry ends",
    "f_minute": "match minute incl. stoppage (minute + second/60)",
    "f_t_period": "seconds since the start of the period", "f_period": "same as period",
    "f_ts_repaired": "timestamp was > 60 s before the previous located event (StatsBomb "
                     "glitch) and was replaced by that event's time",
    "f_home": "event team is the home team",
    "f_is_possession_team": "event team == possession team",
    "f_score_for": "event team's goals before this event (periods 1-4)",
    "f_score_against": "opponent goals before this event", "f_score_diff": "for - against",
    "f_goals_total": "for + against",
    "f_dt_prev": "seconds since the previous located event (same period); NaN if none",
    "f_poss_idx": "StatsBomb possession index",
    "f_poss_elapsed": "seconds since the first event of the possession",
    "f_poss_n_events": "located events earlier in this possession (both teams)",
    "f_poss_n_passes": "passes by the possession team earlier in this possession",
    "f_poss_ball_dist": "sum of pass lengths + carry lengths so far in the possession (yd)",
    "f_poss_start_type": "how the possession started (kick_off/set_piece/recovery/"
                         "interception/duel/keeper/open_play/other)",
    "f_poss_t_since_ft_entry": "seconds since the ball last crossed into the final third "
                               "(possession team's frame); NaN if it has not",
    "f_poss_in_final_third": "ball currently in the possession team's final third",
    "f_poss_start_x": "possession start x (event team's frame)",
    "f_poss_start_y": "possession start y (event team's frame)",
    "f_pass_type": "Open Play or set-piece type (known before the pass)",
    "f_pass_body_part": "body part used (at the instant of the pass)",
    "f_after_pass_length": "POST-INSTANT: realised pass length (yd)",
    "f_after_pass_angle": "POST-INSTANT: realised pass angle (rad, StatsBomb)",
    "f_after_pass_height": "POST-INSTANT: Ground/Low/High Pass (realised trajectory)",
    "f_after_pass_switch": "POST-INSTANT: switch of play",
    "f_after_pass_cross": "POST-INSTANT: cross",
    "f_after_pass_through_ball": "POST-INSTANT: through ball",
    "f_after_pass_cut_back": "POST-INSTANT: cut back",
    "f_after_pass_end_x": "POST-INSTANT: pass end x (an incomplete pass ends where the "
                          "opponent won it)",
    "f_after_pass_end_y": "POST-INSTANT: pass end y",
    "f_after_pass_end_dist_goal": "POST-INSTANT: distance from pass end to goal centre",
    "f_after_pass_progress": "POST-INSTANT: end x - start x",
    "post_pass_outcome": "POST-EVENT: Complete / Incomplete / Out / ... (not a state feature)",
    "f_after_carry_length": "POST-INSTANT: carry length (yd) - a consequence of the "
                            "nearest defender's distance",
    "f_after_carry_end_x": "POST-INSTANT: carry end x",
    "f_after_carry_end_y": "POST-INSTANT: carry end y",
    "f_after_carry_progress": "POST-INSTANT: carry end x - start x",
    "f_shot_body_part": "shot body part", "f_shot_technique": "shot technique",
    "f_shot_type": "Open Play / Free Kick / Penalty / Corner",
    "f_shot_first_time": "first-time shot",
    "post_shot_outcome": "POST-EVENT shot outcome (Goal, Saved, ...)",
    "oracle_statsbomb_xg": "ORACLE: StatsBomb xG (freeze-frame based) - never a feature",
    "oracle_one_on_one": "ORACLE: shot.one_on_one flag",
    "oracle_open_goal": "ORACLE: shot.open_goal",
    "f_w10_n_own": "of the previous 10 located events, how many by the event team",
    "f_w10_n_opp_def": "of the previous 10, opponent defensive actions "
                       "(Pressure/Duel/Interception/Block/Clearance)",
    "f_w10_n": "number of previous located events available in the window (same period)",
    "f_opp_def_x_60s_mean": "mean x (event team's frame, 120 - x_opp) of opponent defensive "
                            "actions in the last 60 s; NaN if none",
    "f_opp_def_n_60s": "count of opponent defensive actions in the last 60 s",
    "f_opp_poss_last10s": "any event in the last 10 s with the opponent as possession team",
    "f_ball_speed_3": "path length over the last 3 event transitions / elapsed time (yd/s)",
    "f_ball_dx_3": "x displacement of the ball over the last 3 previous events",
    "f_ball_dy_3": "y displacement over the last 3 previous events",
    "f_t_since_opp_def_action": "seconds since the opponent's last defensive action "
                                "(same period); NaN if none",
    "y_frame_orientation": "360: ok / mirrored / mirrored_swapped (repaired) / far / "
                           "unresolved / no_actor / no_location / empty (unusable); see "
                           "frame_features.resolve_frame_orientation",
    "y_frame_ok": "360: geometry targets were computed (orientation ok or repaired)",
    "y_frame_repaired": "360: frame was mirrored (and flag-swapped for mirrored_swapped)",
    "y_frame_method": "360: how a mirrored frame's flag perspective was decided: anchor "
                      "(identical-coordinate ok frame of the other team) or keeper",
    "y_actor_dist_to_event": "360: distance (yd) between the raw flagged actor and the event "
                             "location (0 for ok frames)",
    "y_keeper_consistent": "360: after resolution every visible keeper is at the expected "
                           "end (opp x>60, own x<60); NaN if no keeper visible",
    "y_n_teammates_visible": "360: visible teammates excluding the actor",
    "y_n_opponents_visible": "360: visible opponents incl. keeper",
    "y_opp_keeper_visible": "360: opponent keeper visible",
    "y_tm_keeper_visible": "360: own keeper visible",
    "y_actor_visible": "360: actor present in frame",
    "y_reliable": "360: frame usable (y_frame_ok) and n_opponents_visible >= 7",
    "y_visible_area_frac": "360: visible_area polygon (clipped to pitch) / 9600",
    "y_n_opp_outfield": "360: visible outfield opponents",
    "y_block_depth": "360: 120 - mean x of outfield opponents (distance from their goal)",
    "y_def_line": "360: 120 - max x of outfield opponents (deepest defender)",
    "y_block_length": "360: x range of outfield opponents", "y_block_width": "360: y range",
    "y_block_area": "360: convex-hull area of outfield opponents (NaN < 3 players)",
    "y_block_centroid_y": "360: mean y of outfield opponents",
    "y_block_std_x": "360: std of x", "y_block_std_y": "360: std of y",
    "y_n_opp_in_box": "360: outfield opponents in the penalty box (x>=102, 18<=y<=62)",
    "y_n_tm_in_box": "360: teammates in the box",
    "y_opp_keeper_dist_to_goal_line": "360: 120 - keeper x (NaN if not visible)",
    "y_opp_keeper_y": "360: keeper y",
    "y_n_opp_ahead_of_ball": "360: opponents (incl. keeper) with x > ball x",
    "y_n_tm_ahead_of_ball": "360: teammates with x > ball x",
    "y_n_opp_within_5": "360: opponents within 5 yd of the ball",
    "y_n_opp_within_10": "360: opponents within 10 yd",
    "y_nearest_opp_dist": "360: nearest opponent (yd)",
    "y_n_tm_within_10": "360: teammates within 10 yd", "y_nearest_tm_dist": "360: nearest teammate",
    "y_n_opp_in_cone": "360: outfield opponents in triangle ball-(120,36)-(120,44)",
    "y_nearest_opp_dist_in_cone": "360: nearest outfield opponent inside the cone (NaN if none)",
    "y_opp_keeper_in_cone": "360: keeper inside the cone", "y_cone_area": "area of the shot cone",
    "y_n_opp_within_3_of_end": "360 (Pass only): opponents within 3 yd of the pass end",
    "y_n_opp_in_lane": "360 (Pass only): opponents within 2 yd of the start->end segment",
    "y_nearest_opp_to_end": "360 (Pass only): nearest opponent to the pass end",
    "y_nearest_opp_to_receiver": "360 (Pass only): nearest opponent to the receiver "
                                 "(teammate nearest the end location)",
    "y_receiver_dist_to_end": "360 (Pass only): receiver distance to the pass end",
    "sff_present": "shot.freeze_frame available (Shot rows; penalties usually lack it)",
    "sff_n_opp": "shot frame: opponents listed", "sff_n_tm": "shot frame: teammates listed",
    "sff_opp_keeper_visible": "shot frame: keeper listed",
    "sff_n_opp_in_cone": "shot frame: outfield opponents in the cone",
    "sff_nearest_opp_dist_in_cone": "shot frame: nearest outfield opponent in the cone",
    "sff_opp_keeper_in_cone": "shot frame: keeper in the cone", "sff_cone_area": "shot cone area",
    "sff_n_opp_within_5": "shot frame: opponents within 5 yd",
    "sff_n_opp_within_10": "shot frame: opponents within 10 yd",
    "sff_nearest_opp_dist": "shot frame: nearest opponent",
    "sff_n_opp_in_box": "shot frame: outfield opponents in the box",
    "sff_n_opp_ahead_of_ball": "shot frame: opponents with x > ball x",
    "sff_opp_keeper_dist_to_goal_line": "shot frame: 120 - keeper x",
    "sff_opp_keeper_y": "shot frame: keeper y",
}


def _doc_for(col: str) -> str:
    if col in COLUMN_DOCS:
        return COLUMN_DOCS[col]
    if col.startswith("f_w10_n_"):
        return f"count of '{col[len('f_w10_n_'):]}' events among the previous 10 (both teams)"
    if col.startswith("seq_"):
        _, fld, k = col.split("_")
        desc = {"type": "type id (0 = pad)", "x": "x in the event team's frame (NaN pad)",
                "y": "y (NaN pad)", "dt": "seconds before the current event (NaN pad)",
                "same": "1 own team, 0 opponent, -1 pad"}[fld]
        return f"history slot {int(k)} (1 = most recent located event, same period): {desc}"
    return ""


def _group_for(col: str) -> str:
    for p, g in (("y_", "target_360"), ("sff_", "shot_freeze_frame"), ("seq_", "sequence"),
                 ("oracle_", "oracle"), ("post_", "post_event"), ("f_", "event_feature")):
        if col.startswith(p):
            return g
    return "id"


def _instant_for(col: str) -> str:
    """``pre`` = known at the instant of the event, ``post`` = describes what happened
    after it (``f_after_*``); empty for non-feature columns."""
    if col.startswith(POST_INSTANT_PREFIX):
        return "post"
    if col.startswith("f_") or col.startswith("seq_"):
        return "pre"
    return ""


def column_table() -> pd.DataFrame:
    """Name / dtype / group / instant / definition for every output column."""
    schema = output_schema()
    return pd.DataFrame([{"column": f.name, "dtype": str(f.type), "group": _group_for(f.name),
                          "instant": _instant_for(f.name), "definition": _doc_for(f.name)}
                         for f in schema])


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-markdown table (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else floatfmt.format(v)
        if isinstance(v, (np.floating,)):
            return "" if np.isnan(v) else floatfmt.format(float(v))
        if v is None:
            return ""
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _quantiles(s: pd.Series, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> dict[str, float]:
    s = s.dropna()
    d: dict[str, float] = {"n": int(len(s)), "mean": float(s.mean()) if len(s) else float("nan"),
         "std": float(s.std()) if len(s) > 1 else float("nan")}
    for q in qs:
        d[f"p{int(q * 100)}"] = float(s.quantile(q)) if len(s) else float("nan")
    return d


# y_nearest_opp_dist mean / sd / median on "reliable" frames in the build BEFORE the
# orientation check (git history: reports/soccer_01_targets_by_type.parquet), quoted in the
# report so the effect of the repair on the affected event types stays visible.
PRE_FIX_NEAREST_OPP: dict[str, tuple[float, float, float]] = {
    "Ball Receipt*": (9.99, 12.81, 6.73), "Duel": (23.81, 30.58, 3.94),
    "Dribble": (22.73, 31.23, 2.06), "Foul Won": (38.31, 24.65, 34.86),
    "Pass": (5.06, 4.46, 3.79), "Carry": (7.22, 5.08, 6.37), "Pressure": (2.85, 1.60, 2.58),
}
AFFECTED_TYPES: tuple[str, ...] = ("Ball Receipt*", "Duel", "Dribble", "Foul Won",
                                   "Dispossessed", "Dribbled Past", "50/50", "Carry")

KEY_TARGETS: tuple[str, ...] = (
    "y_block_depth", "y_def_line", "y_block_length", "y_block_width", "y_block_area",
    "y_n_opp_ahead_of_ball", "y_n_opp_within_5", "y_n_opp_within_10", "y_nearest_opp_dist",
    "y_n_opp_in_box", "y_n_opp_in_cone", "y_nearest_opp_dist_in_cone",
    "y_n_opp_within_3_of_end", "y_n_opp_in_lane", "y_nearest_opp_to_receiver",
    "y_opp_keeper_dist_to_goal_line",
)


def write_report(cfg: BuildConfig, stats: pd.DataFrame | None = None) -> Path:
    """Data-quality report + parquet result tables from the written outputs."""
    rep = reports_dir()
    out = processed_dir("soccer")
    if stats is None:
        stats = pd.read_parquet(rep / f"{REPORT_STEM}_match_stats.parquet")
    vis_cols = ["y_n_teammates_visible", "y_n_opponents_visible", "y_opp_keeper_visible",
                "y_visible_area_frac", "y_reliable", "y_actor_visible", "y_frame_orientation",
                "y_frame_ok", "y_frame_repaired", "y_frame_method", "y_actor_dist_to_event",
                "y_keeper_consistent"]
    cols360 = (["match_id", "competition", "season", "gender", "f_type", "f_is_possession_team",
                "f_ts_repaired",
                "f_x", "f_has_location", "sff_present", "sff_n_opp", "sff_n_opp_in_cone",
                "sff_nearest_opp_dist_in_cone", "sff_opp_keeper_in_cone", "y_opp_keeper_in_cone",
                "f_shot_type"] + vis_cols + list(KEY_TARGETS))
    e360 = pd.read_parquet(out / "events360.parquet", columns=sorted(set(cols360)))
    eno = pd.read_parquet(out / "events_no360.parquet",
                          columns=["match_id", "competition", "season", "gender", "f_type",
                                   "sff_present", "f_shot_type", "f_ts_repaired"])

    # ---- 1. matches / events per competition -------------------------------------------
    ok = stats[stats.n_rows > 0]
    grp = (ok.groupby(["competition", "season", "gender", "has_360"], as_index=False)
           .agg(n_matches=("match_id", "nunique"), n_events_total=("n_events_total", "sum"),
                n_frames=("n_frames", "sum"), n_rows=("n_rows", "sum"),
                n_shots=("n_shots", "sum"), n_pass=("n_pass", "sum"), n_carry=("n_carry", "sum"))
           .sort_values(["has_360", "competition", "season"], ascending=[False, True, True]))
    grp["frames_per_match"] = np.where(grp.has_360, grp.n_frames / grp.n_matches, np.nan)
    grp["frame_coverage"] = np.where(grp.has_360, grp.n_frames / grp.n_events_total, np.nan)
    grp.to_parquet(rep / f"{REPORT_STEM}_match_counts.parquet", index=False)

    # ---- 2. visibility ----------------------------------------------------------------
    vis_rows = []
    for c in ["y_n_teammates_visible", "y_n_opponents_visible", "y_visible_area_frac"]:
        vis_rows.append({"metric": c, "scope": "all",
                         **_quantiles(e360[c], (0.05, 0.25, 0.5, 0.75, 0.95))})
    for c in ["y_opp_keeper_visible", "y_actor_visible", "y_reliable"]:
        vis_rows.append({"metric": c, "scope": "all", "n": int(e360[c].notna().sum()),
                         "mean": float(e360[c].mean())})
    vis = pd.DataFrame(vis_rows)
    type_counts = e360.f_type.value_counts()
    top_types = list(type_counts.index[:14])
    by_type = (e360[e360.f_type.isin(top_types)].groupby("f_type")
               .agg(n=("y_reliable", "size"), frame_ok_share=("y_frame_ok", "mean"),
                    reliable_share=("y_reliable", "mean"),
                    opp_visible_mean=("y_n_opponents_visible", "mean"),
                    tm_visible_mean=("y_n_teammates_visible", "mean"),
                    keeper_visible_share=("y_opp_keeper_visible", "mean"),
                    visible_area_mean=("y_visible_area_frac", "mean"),
                    in_possession_share=("f_is_possession_team", "mean"))
               .reindex(top_types).reset_index())
    by_comp = (e360.groupby(["competition", "season"])
               .agg(n=("y_reliable", "size"), reliable_share=("y_reliable", "mean"),
                    opp_visible_mean=("y_n_opponents_visible", "mean"),
                    visible_area_mean=("y_visible_area_frac", "mean")).reset_index())
    # reliability vs ball location (final third vs rest) for possession-team events
    posn = e360[e360.f_is_possession_team & e360.f_has_location].copy()
    posn["zone"] = pd.cut(posn.f_x, [-1, 40, 80, 121], labels=["own_third", "middle", "final"])
    by_zone = (posn.groupby("zone", observed=True)
               .agg(n=("y_reliable", "size"), reliable_share=("y_reliable", "mean"),
                    opp_visible_mean=("y_n_opponents_visible", "mean")).reset_index())
    vis.to_parquet(rep / f"{REPORT_STEM}_visibility.parquet", index=False)
    by_type.to_parquet(rep / f"{REPORT_STEM}_visibility_by_type.parquet", index=False)
    by_comp.to_parquet(rep / f"{REPORT_STEM}_visibility_by_competition.parquet", index=False)

    # ---- 2b. frame orientation (StatsBomb paired-event defect) ---------------------------
    orient_all = (e360.groupby("y_frame_orientation")
                  .agg(n=("y_frame_ok", "size"), frame_ok=("y_frame_ok", "mean"),
                       actor_dist_median=("y_actor_dist_to_event", "median"),
                       keeper_consistent=("y_keeper_consistent", "mean"),
                       keeper_n=("y_keeper_consistent", "count")).reset_index())
    orient_all["share"] = orient_all.n / len(e360)
    orient_all = orient_all[["y_frame_orientation", "n", "share", "frame_ok",
                             "actor_dist_median", "keeper_consistent", "keeper_n"]]
    o_types = list(type_counts.index[:18])
    ocount = pd.crosstab(e360.f_type, e360.y_frame_orientation).reindex(o_types).fillna(0)
    for o in ("ok", "mirrored", "mirrored_swapped", "far", "unresolved"):
        if o not in ocount.columns:
            ocount[o] = 0
    orient_type = pd.DataFrame({
        "f_type": o_types, "n": ocount.sum(axis=1).astype(int).to_numpy(),
        "ok": ocount["ok"].astype(int).to_numpy(),
        "mirrored": ocount["mirrored"].astype(int).to_numpy(),
        "mirrored_swapped": ocount["mirrored_swapped"].astype(int).to_numpy(),
        "far": ocount["far"].astype(int).to_numpy(),
        "unresolved": ocount["unresolved"].astype(int).to_numpy(),
    })
    other_cols = [c for c in ocount.columns
                  if c not in ("ok", "mirrored", "mirrored_swapped", "far", "unresolved")]
    orient_type["other"] = (ocount[other_cols].sum(axis=1).astype(int).to_numpy()
                            if other_cols else 0)
    orient_type["mis_oriented_share"] = 1 - orient_type.ok / orient_type.n
    n_rep = orient_type.mirrored + orient_type.mirrored_swapped
    orient_type["repaired_share"] = n_rep / orient_type.n
    orient_type["usable_share"] = (orient_type.ok + orient_type.mirrored
                                   + orient_type.mirrored_swapped) / orient_type.n
    method = (e360[e360.y_frame_repaired == 1].groupby(["y_frame_orientation", "y_frame_method"])
              .agg(n=("y_frame_ok", "size"), keeper_consistent=("y_keeper_consistent", "mean"),
                   keeper_n=("y_keeper_consistent", "count")).reset_index())
    orient_all.to_parquet(rep / f"{REPORT_STEM}_orientation.parquet", index=False)
    orient_type.to_parquet(rep / f"{REPORT_STEM}_orientation_by_type.parquet", index=False)
    method.to_parquet(rep / f"{REPORT_STEM}_orientation_method.parquet", index=False)
    # before / after on the affected types (reliable frames after the fix)
    rel_now = e360[e360.y_reliable == 1]
    ba_rows = []
    for t in AFFECTED_TYPES:
        sub = rel_now[rel_now.f_type == t]
        pre = PRE_FIX_NEAREST_OPP.get(t)
        ba_rows.append({
            "f_type": t, "n_reliable_now": int(len(sub)),
            "n_rows": int((e360.f_type == t).sum()),
            "mis_oriented_share": float(
                (e360.loc[e360.f_type == t, "y_frame_orientation"] != "ok").mean()),
            "pre_fix_nearest_opp_mean": pre[0] if pre else float("nan"),
            "pre_fix_nearest_opp_sd": pre[1] if pre else float("nan"),
            "pre_fix_nearest_opp_median": pre[2] if pre else float("nan"),
            "now_nearest_opp_mean": float(sub.y_nearest_opp_dist.mean()),
            "now_nearest_opp_sd": float(sub.y_nearest_opp_dist.std()),
            "now_nearest_opp_median": float(sub.y_nearest_opp_dist.median()),
            "now_n_opp_within_5_mean": float(sub.y_n_opp_within_5.mean()),
        })
    before_after = pd.DataFrame(ba_rows)
    before_after.to_parquet(rep / f"{REPORT_STEM}_orientation_before_after.parquet", index=False)

    # ---- 3. target distributions by type (reliable frames) ------------------------------
    rel = e360[e360.y_reliable == 1]
    tgt_rows = []
    for t in top_types:
        sub = rel[rel.f_type == t]
        for c in KEY_TARGETS:
            tgt_rows.append({"f_type": t, "target": c, **_quantiles(sub[c])})
    for c in KEY_TARGETS:
        tgt_rows.append({"f_type": "<all reliable>", "target": c, **_quantiles(rel[c])})
        tgt_rows.append({"f_type": "<all frames>", "target": c, **_quantiles(e360[c])})
    tgt = pd.DataFrame(tgt_rows)
    tgt.to_parquet(rep / f"{REPORT_STEM}_targets_by_type.parquet", index=False)
    compact_targets = ["y_block_depth", "y_def_line", "y_block_width", "y_n_opp_ahead_of_ball",
                       "y_n_opp_within_5", "y_nearest_opp_dist", "y_n_opp_in_cone"]
    compact = (tgt[tgt.target.isin(compact_targets) & ~tgt.f_type.str.startswith("<")]
               .assign(cell=lambda d: d.apply(
                   lambda r: f"{r['mean']:.1f} ({r['std']:.1f})", axis=1))
               .pivot(index="f_type", columns="target", values="cell")
               .reindex(top_types)[compact_targets].reset_index())
    compact.insert(1, "n_reliable", [int(len(rel[rel.f_type == t])) for t in top_types])

    # ---- 4. cone agreement for shots ---------------------------------------------------
    sh_all = e360[(e360.f_type == "Shot") & (e360.sff_present == 1)]
    n_shot_unusable = int((sh_all.y_frame_ok == 0).sum())
    sh = sh_all[sh_all.y_frame_ok == 1].copy()
    agree_rows = []

    def _agree(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        a = sub.y_n_opp_in_cone.to_numpy(dtype=float)
        b = sub.sff_n_opp_in_cone.to_numpy(dtype=float)
        d = {"subset": label, "n": int(len(sub))}
        if len(sub) < 2:
            return d
        d.update({
            "exact_agreement": float(np.mean(a == b)),
            "within_1": float(np.mean(np.abs(a - b) <= 1)),
            "mae": float(np.mean(np.abs(a - b))),
            "mean_360": float(a.mean()), "mean_sff": float(b.mean()),
            "pearson_r": (float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0
                          else float("nan")),
            "keeper_in_cone_agreement": float(np.mean(
                sub.y_opp_keeper_in_cone.to_numpy() == sub.sff_opp_keeper_in_cone.to_numpy())),
            "mean_opp_visible_360": float(sub.y_n_opponents_visible.mean()),
            "mean_opp_listed_sff": float(sub.sff_n_opp.mean()),
        })
        m = sub.y_nearest_opp_dist_in_cone.notna() & sub.sff_nearest_opp_dist_in_cone.notna()
        if m.sum() > 2:
            x = sub.loc[m, "y_nearest_opp_dist_in_cone"].to_numpy(dtype=float)
            y = sub.loc[m, "sff_nearest_opp_dist_in_cone"].to_numpy(dtype=float)
            d["nearest_in_cone_n"] = int(m.sum())
            d["nearest_in_cone_mae"] = float(np.mean(np.abs(x - y)))
            d["nearest_in_cone_r"] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        return d

    agree_rows.append(_agree(sh, "all shots with both frames"))
    agree_rows.append(_agree(sh[sh.y_reliable == 1], "reliable 360 frames (>=7 opp visible)"))
    agree_rows.append(_agree(sh[sh.y_reliable == 0], "unreliable 360 frames"))
    agree_rows.append(_agree(sh[sh.f_shot_type == "Open Play"], "open-play shots"))
    for g in ("male", "female"):
        agree_rows.append(_agree(sh[sh.gender == g], f"{g} competitions"))
    agree = pd.DataFrame(agree_rows)
    agree.to_parquet(rep / f"{REPORT_STEM}_cone_agreement.parquet", index=False)
    ct = pd.crosstab(sh.y_n_opp_in_cone.clip(upper=4).astype(int),
                     sh.sff_n_opp_in_cone.clip(upper=4).astype(int))
    ct.index = [f"360={i}" if i < 4 else "360=4+" for i in ct.index]
    ct.columns = [f"sff={i}" if i < 4 else "sff=4+" for i in ct.columns]
    ct = ct.reset_index().rename(columns={"index": "count"})

    # ---- 5. columns -----------------------------------------------------------------
    cols = column_table()
    cols.to_parquet(rep / f"{REPORT_STEM}_columns.parquet", index=False)
    vocab = pd.DataFrame([asdict(r) for r in type_vocab_rows()])
    vocab.to_parquet(rep / f"{REPORT_STEM}_type_vocab.parquet", index=False)

    # ---- markdown ---------------------------------------------------------------------
    n360_rows, nno_rows = len(e360), len(eno)
    errs = stats[stats.error.notna()]
    ts_rep_360, ts_rep_no = e360.f_ts_repaired.sum(), eno.f_ts_repaired.sum()
    wall = (float(stats.wall_seconds_total.iloc[0]) if "wall_seconds_total" in stats
            else float("nan"))
    shots_no = int((eno.f_type == "Shot").sum())
    shots_no_ff = int(((eno.f_type == "Shot") & (eno.sff_present == 1)).sum())
    lines = [
        "# Soccer 01 - build: 360 defensive-state targets and event-only features",
        "",
        "Source: StatsBomb open data (`data/raw/privileged/sb`). Outputs under "
        "`processed_dir('soccer')`: `events360.parquet` (one row per event with a 360 freeze "
        "frame) and `events_no360.parquet` (event-only features for matches without 360). "
        "Nothing here is modelled; this is the data layer for the imputation and payoff stages.",
        "",
        "## Run summary",
        "",
        f"- matches processed: {int(ok.match_id.nunique())} of {int(stats.match_id.nunique())} "
        f"({int(ok.has_360.sum())} with usable 360, {int((~ok.has_360).sum())} without, of which "
        f"{len(errs)} are 360-flagged matches whose frames were unusable, see bottom)",
        f"- `events360.parquet`: {n360_rows:,} rows from {int(ok[ok.has_360].n_frames.sum()):,} "
        f"frames ({int(ok[ok.has_360].n_events_total.sum()):,} events in those matches; "
        f"{int(ok.n_frames_no_location.sum())} frames on events without a location, which "
        "are unusable: `y_frame_orientation = no_location`)",
        f"- `events_no360.parquet`: {nno_rows:,} rows = all {shots_no:,} Shots "
        f"({shots_no_ff:,} with a `shot.freeze_frame`) + a fixed random "
        f"{cfg.no360_pass_carry_frac:.0%} of Pass/Carry events per match "
        f"(seeded by match id; `--frac 1.0` keeps all). "
        f"{int((eno.f_type == 'Pass').sum()):,} passes and {int((eno.f_type == 'Carry').sum()):,} "
        f"carries kept.",
        f"- wall time {wall / 60:.1f} min with {cfg.workers} worker processes",
        f"- 360 frame orientation: {int((e360.y_frame_orientation == 'ok').sum()):,} frames ok "
        f"({(e360.y_frame_orientation == 'ok').mean():.1%}), "
        f"{int(e360.y_frame_repaired.sum()):,} repaired ({e360.y_frame_repaired.mean():.1%}: "
        f"mirrored {int((e360.y_frame_orientation == 'mirrored').sum()):,}, mirrored + flag "
        f"swap {int((e360.y_frame_orientation == 'mirrored_swapped').sum()):,}), "
        f"{int((e360.y_frame_ok == 0).sum()):,} unusable ({(e360.y_frame_ok == 0).mean():.1%}; "
        "all `y_*` geometry targets NaN, `y_reliable` = 0). See the orientation section.",
        "- `f_after_*` columns (realised pass trajectory, carry end, duration) describe what "
        "happened AFTER the instant of the event; they are event-only but not pre-instant "
        "state (`instant` = post in the column table).",
        "- penalty shoot-outs (period 5) are dropped; goals for the score state are counted in "
        "periods 1-4 only; all windows and the sequence block reset at each period start",
        "- coordinates in every row are in the event team's attacking frame (goal at x=120); "
        "opponent events are mirrored with (120-x, 80-y) when they enter windows/sequence",
        "- privileged fields (`shot.freeze_frame`, `statsbomb_xg`, `one_on_one`, `open_goal`) "
        "are emitted only under the `sff_` / `oracle_` prefixes; `post_` columns are outcomes. "
        "Event-only feature sets must be selected as `f_*` + `seq_*` columns only, and "
        "results reported with and without the post-instant `f_after_*` subset.",
        "",
        "## Matches and events per competition",
        "",
        md_table(grp[["competition", "season", "gender", "has_360", "n_matches", "n_events_total",
                      "n_frames", "frame_coverage", "n_rows", "n_shots", "n_pass", "n_carry"]]),
        "",
        "`n_rows` counts output rows (frames for 360 matches; shots + sampled passes/carries "
        "otherwise). `frame_coverage` = frames / events in the match.",
        "",
        "## 360 visibility",
        "",
        md_table(vis),
        "",
        f"Share of reliable frames (>= 7 opponents visible): {e360.y_reliable.mean():.3f} "
        f"(n = {n360_rows:,}).",
        "",
        "### By event type (14 most frequent)",
        "",
        md_table(by_type),
        "",
        "### By competition",
        "",
        md_table(by_comp),
        "",
        "### By ball zone (possession-team events only)",
        "",
        md_table(by_zone),
        "",
        "## 360 frame orientation (StatsBomb paired-event defect)",
        "",
        "For paired events StatsBomb stores the 360 frame of the *other* team's event: the "
        "coordinates are in the other team's attacking frame and, for some pairs, the "
        "`teammate` flags are also from the other team's perspective. Detected per row by "
        "comparing the flagged actor with the event location (`y_actor_dist_to_event`; 0 for "
        "correct frames, the mirrored actor is within 5 yd of the event location for a "
        "mirrored frame). A mirrored frame is repaired by mirroring all coordinates "
        "(`mirrored`) or by mirroring and swapping the teammate flags (`mirrored_swapped`); "
        "which one is decided by an *anchor* (a frame of the other team with identical "
        "coordinates whose own actor sits on its event location: same flags => swap, "
        "inverted flags => pure mirror), falling back to the keeper ends after the mirror. "
        "`far` frames (actor away from both the event location and its mirror: a frame from "
        "another instant, almost all incomplete-pass receipts whose frame is the opponent's "
        "next action) and `unresolved` mirrored frames get NaN geometry targets and "
        "`y_reliable = 0`. `y_keeper_consistent` checks, after the resolution, that every "
        "visible keeper is at the expected end of the pitch.",
        "",
        md_table(orient_all),
        "",
        "### By event type (18 most frequent)",
        "",
        md_table(orient_type),
        "",
        "### Repaired frames by decision method",
        "",
        md_table(method),
        "",
        "### Effect on the affected event types",
        "",
        "`y_nearest_opp_dist` on reliable frames in the previous build (no orientation "
        "check; from the pre-fix `soccer_01_targets_by_type.parquet`) versus now. "
        "`mis_oriented_share` = share of the type's rows whose raw frame was not `ok`. "
        "Dispossessed / Dribbled Past were not in the previous per-type table; their raw "
        "all-frame medians were ~50 yd (the actor's mirror image).",
        "",
        md_table(before_after),
        "",
        "## Target distributions by event type (reliable frames only)",
        "",
        "Reliable = `y_frame_ok == 1` (orientation ok or repaired) and >= 7 opponents "
        "visible. Cells are mean (sd). Full long-format table with p10/p50/p90 for all "
        f"{len(KEY_TARGETS)} key targets in `{REPORT_STEM}_targets_by_type.parquet`.",
        "",
        md_table(compact),
        "",
        "### All frames vs reliable frames",
        "",
        md_table(tgt[tgt.f_type.str.startswith("<")][["f_type", "target", "n", "mean", "std",
                                                         "p10", "p50", "p90"]]),
        "",
        "## Shots: 360-derived vs shot.freeze_frame-derived cone counts",
        "",
        "Both are the number of outfield opponents inside the triangle ball -> posts. The shot "
        "freeze frame lists every player StatsBomb coded from video (not limited to the "
        "broadcast visible area), so it is the closer-to-truth reference; disagreement mostly "
        f"reflects 360 visibility. Shots with an unusable 360 frame ({n_shot_unusable} of "
        f"{len(sh_all):,} shots with both frames) are excluded.",
        "",
        md_table(agree),
        "",
        "Cross-tab of cone counts (360 rows, shot-frame columns, counts capped at 4+):",
        "",
        md_table(ct),
        "",
        "## Column definitions",
        "",
        f"{len(cols)} columns in both files. Groups: id, event_feature (`f_`), sequence (`seq_`), "
        "target_360 (`y_`), shot_freeze_frame (`sff_`), oracle (`oracle_`), post_event (`post_`). "
        "`instant` = pre (known at the instant of the event) or post (`f_after_*`: realised "
        "trajectory / end point / duration, i.e. after the instant the 360 frame shows); "
        f"{int((cols.instant == 'post').sum())} columns are post-instant.",
        "",
        md_table(cols[cols.group != "sequence"]),
        "",
        "Sequence block: `seq_<field>_<k>` for k = 01..20 (k = 1 most recent located event of "
        "the same period), fields type (int16 id, 0 pad), x, y (event team's frame, NaN pad), "
        "dt (seconds before the event, NaN pad), same (1 own team, 0 opponent, -1 pad).",
        "",
        "### Type vocabulary (`f_type_id`, `seq_type_*`)",
        "",
        md_table(vocab),
        "",
        "## Caveats",
        "",
        "- 360 frames only contain players inside the broadcast `visible_area`; block-shape "
        "targets on unreliable frames are biased towards the players near the ball. Downstream "
        "stages should train on `y_reliable == 1` (or weight by visibility) and report both.",
        "- `y_reliable` now requires `y_frame_ok == 1`; rows with `y_frame_ok == 0` have NaN "
        "geometry targets by construction. Repaired frames (`y_frame_repaired == 1`) are "
        "validated per row (anchor flags or keeper ends); downstream stages can exclude them "
        "with `y_frame_orientation == 'ok'` as a robustness check. After a flag swap the "
        "actor is re-identified as the teammate nearest the event location (within 5 yd), so "
        "`y_actor_visible` can be 0 on `mirrored_swapped` rows. `y_visible_area_frac` is "
        "computed from the raw polygon (its area is mirror-invariant).",
        "- `f_after_*` (pass end/length/angle/height/switch/cross/through-ball/cut-back/"
        "progress, carry end/length/progress, duration) are post-instant: a carry's length "
        "depends on the nearest defender and an incomplete pass ends where the opponent won "
        "the ball. They are legitimate event-only inputs for imputation on non-360 matches, "
        "but any imputation / payoff result must be reported with and without them "
        "(`instant` column of `soccer_01_columns.parquet`).",
        "- Lint scope: `ruff check research/privileged_tracking/soccer "
        "research/privileged_tracking/tests/test_soccer_*.py` is clean; the NFL test files in "
        "the same tests directory belong to the NFL item and are not linted here.",
        "- `y_block_*` describe the *opponents of the event team*; on defending-team events "
        "(Pressure, Duel, ...) that is the attacking team's shape. Filter with "
        "`f_is_possession_team` when a defensive-block target is wanted.",
        "- Pass geometry targets exist only for Pass events with an end location; they are "
        "NaN elsewhere and cannot be built from `shot.freeze_frame`.",
        "- `f_poss_ball_dist` uses pass lengths and carry lengths only (not receipts/dribbles).",
        "- `f_opp_def_x_60s_mean` and `f_t_since_opp_def_action` look back only within the "
        "current period; extra-time periods restart the windows.",
        "- Tendency features (team/player rolling means) are deliberately absent here; they "
        "must be built from strictly earlier matches in the modelling stage.",
        f"- {int(ts_rep_360):,} events360 rows and {int(ts_rep_no):,} events_no360 rows carry "
        "`f_ts_repaired = True` (receipt stamped 00:00:00 late in a period; time replaced by "
        "the previous event's).",
    ]
    if len(errs):
        lines += ["", f"### {len(errs)} 360-flagged matches without usable frames", "",
                  "`frames_missing` = corrupt/absent 360 file (3845506 is corrupt upstream too: "
                  "a run of null bytes); `frames_unmatched` = no frame `event_uuid` matches any "
                  "event (360 and event files out of sync); `frames_too_few` = fewer than "
                  f"{cfg.min_matched_frames} frames joined to events. These matches fall back to "
                  "the event-only output (`events_no360`, `has_360 = False`).", "",
                  md_table(errs[["match_id", "competition", "season", "n_frames", "error",
                                 "n_rows"]])]
    path = rep / f"{REPORT_STEM}_build.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"report written to {path}", file=sys.stderr)
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.25,
                    help="fraction of Pass/Carry events kept per non-360 match (all shots kept)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="matches per group (smoke runs)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from existing parquet outputs")
    a = ap.parse_args(argv)
    cfg = BuildConfig(workers=a.workers, no360_pass_carry_frac=a.frac, seed=a.seed, limit=a.limit)
    stats = None if a.report_only else build(cfg)
    write_report(cfg, stats)


if __name__ == "__main__":
    main()
