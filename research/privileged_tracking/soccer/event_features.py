"""Event-only features over an ordered list of one match's StatsBomb events.

Nothing here may look at a freeze frame, ``shot.freeze_frame``, ``shot.one_on_one``,
``shot.open_goal`` or ``statsbomb_xg``.  Post-event outcomes (pass outcome, shot
outcome) and oracle fields are emitted under ``post_`` / ``oracle_`` prefixes so a
later stage can keep them out of feature sets by prefix; genuine event-only features
carry the ``f_`` prefix and the sequence block the ``seq_`` prefix.

Within ``f_`` the sub-prefix ``f_after_`` marks attributes that describe what happened
*after* the instant of the event (and therefore after the instant a 360 frame was
captured): the realised pass trajectory (``end_x/y``, ``length``, ``angle``,
``height``, ``switch``/``cross``/``through_ball``/``cut_back``, ``progress``,
``end_dist_goal``), the carry end point / length / progress and the event ``duration``.
They exist in event-only data and may be used for imputation on non-360 matches, but a
carry's length is a consequence of the nearest defender's distance and an incomplete
pass ends where the opponent won the ball, so imputation results must be reported with
and without ``f_after_*``.

All coordinates in one output row are expressed in the *event team's* attacking frame
(opponent goal at ``x = 120``).  Events by the opposing team are flipped with
``(120 - x, 80 - y)`` when they enter a window or the sequence block.

Time: StatsBomb timestamps restart at every period.  ``event_time`` adds a fixed period
offset (45 / 90 / 105 / 120 minutes) so differences are valid *within a period*; all
windows and the sequence block only look back within the current period.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PITCH_X = 120.0
PITCH_Y = 80.0
GOAL_X = 120.0
GOAL_Y = 40.0
POST_LOW = 36.0
POST_HIGH = 44.0

PERIOD_OFFSET_S = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0, 5: 120 * 60.0}

# Compact integer vocabulary for the sequence block (0 = padding).
TYPE_VOCAB: dict[str, int] = {
    "Pass": 1, "Ball Receipt*": 2, "Carry": 3, "Pressure": 4, "Ball Recovery": 5,
    "Duel": 6, "Block": 7, "Clearance": 8, "Goal Keeper": 9, "Dribble": 10, "Shot": 11,
    "Interception": 12, "Foul Committed": 13, "Miscontrol": 14, "Dispossessed": 15,
    "Foul Won": 16, "Dribbled Past": 17, "50/50": 18, "Shield": 19, "Error": 20,
    "Own Goal Against": 21, "Own Goal For": 22, "Offside": 23, "Injury Stoppage": 24,
    "Referee Ball-Drop": 25, "Bad Behaviour": 26, "Player Off": 27, "Player On": 28,
    "Substitution": 29, "Tactical Shift": 30, "Half Start": 31, "Half End": 32,
    "Starting XI": 33, "Camera On": 34, "Camera off": 35,
}
UNKNOWN_TYPE_ID = 36

DEF_ACTION_TYPES = frozenset({"Pressure", "Duel", "Interception", "Block", "Clearance"})

# Groups used for the K-event window counts.
WINDOW_GROUP: dict[str, str] = {
    "Pass": "pass", "Carry": "carry", "Ball Receipt*": "receipt", "Pressure": "pressure",
    "Duel": "duel", "Dribble": "dribble", "Dribbled Past": "dribble", "Shot": "shot",
    "Interception": "interception", "Block": "block", "Clearance": "clearance",
    "Ball Recovery": "recovery", "Foul Committed": "foul", "Foul Won": "foul",
    "Goal Keeper": "keeper", "Miscontrol": "loss", "Dispossessed": "loss",
}
WINDOW_GROUPS: tuple[str, ...] = ("pass", "carry", "receipt", "pressure", "duel", "dribble",
                                  "shot", "interception", "block", "clearance", "recovery",
                                  "foul", "keeper", "loss", "other")

SET_PIECE_PASS_TYPES = frozenset({"Throw-in", "Corner", "Free Kick", "Goal Kick"})
SEQ_FIELDS: tuple[str, ...] = ("type", "x", "y", "dt", "same")


@dataclass(frozen=True)
class EventFeatureConfig:
    """Window sizes for the event-only features.

    Attributes:
        window_k: number of previous events (both teams) in the count window.
        seq_len: length of the fixed-width sequence block.
        def_window_s: look-back for the opponent defensive-action mean x.
        opp_poss_window_s: look-back for "opponent had possession".
        speed_n_events: number of previous events for the ball-speed estimate.
        min_speed_dt_s: floor on the elapsed time in the ball-speed denominator.
        max_backwards_s: an event timestamped more than this many seconds *before* the
            previous located event of the same period is a coding glitch (StatsBomb has a
            few receipts stamped ``00:00:00``); its time is replaced by the previous
            event's time and ``f_ts_repaired`` is set.
        final_third_x: x threshold (possession team's frame) of the final third.
        drop_period_5: drop penalty-shootout events entirely.
    """

    window_k: int = 10
    seq_len: int = 20
    def_window_s: float = 60.0
    opp_poss_window_s: float = 10.0
    speed_n_events: int = 3
    min_speed_dt_s: float = 0.5
    max_backwards_s: float = 60.0
    final_third_x: float = 80.0
    drop_period_5: bool = True


@dataclass
class _Possession:
    idx: int = -1
    team_id: int = -1
    start_t: float = 0.0
    start_x: float = float("nan")
    start_y: float = float("nan")
    start_type: str = "unknown"
    n_events: int = 0
    n_passes: int = 0
    ball_dist: float = 0.0
    last_ft_entry_t: float = float("nan")
    prev_in_ft: bool = False


@dataclass
class _Hist:
    """One located event in the history deque (coordinates in its own team's frame)."""

    t: float
    team_id: int
    poss_team_id: int
    type_name: str
    x: float
    y: float


def timestamp_seconds(ts: str) -> float:
    """``'HH:MM:SS.mmm'`` -> seconds within the period."""
    return int(ts[0:2]) * 3600 + int(ts[3:5]) * 60 + float(ts[6:])


def event_time(e: dict[str, Any]) -> float:
    """Seconds since kick-off with fixed period offsets (valid for within-period deltas)."""
    return PERIOD_OFFSET_S.get(int(e.get("period", 1)), 0.0) + timestamp_seconds(e["timestamp"])


def seq_columns(seq_len: int) -> list[str]:
    """Column names of the sequence block, slot-major: ``seq_<field>_<k>``, k=1 newest."""
    return [f"seq_{f}_{k:02d}" for k in range(1, seq_len + 1) for f in SEQ_FIELDS]


def goal_geometry(x: float, y: float) -> tuple[float, float, float]:
    """Distance to goal centre, opening angle of the goal (rad) and bearing (rad).

    The opening angle is the angle subtended by the two posts as seen from ``(x, y)``;
    the bearing is ``atan2(y - 40, 120 - x)`` (0 = straight at goal, positive = right).
    """
    dx = GOAL_X - x
    dist = math.hypot(dx, GOAL_Y - y)
    if dx <= 0.0:
        opening = 0.0
    else:
        opening = abs(math.atan2(POST_HIGH - y, dx) - math.atan2(POST_LOW - y, dx))
    bearing = math.atan2(y - GOAL_Y, dx) if dist > 0 else 0.0
    return dist, opening, bearing


def flip_xy(x: float, y: float) -> tuple[float, float]:
    """Mirror a location into the other team's attacking frame."""
    return PITCH_X - x, PITCH_Y - y


def possession_start_type(e: dict[str, Any]) -> str:
    """Classify how a possession began from its first event by the possession team."""
    t = e["type"]["name"]
    if t == "Pass":
        pt = (e.get("pass", {}).get("type") or {}).get("name")
        if pt == "Kick Off":
            return "kick_off"
        if pt in SET_PIECE_PASS_TYPES:
            return "set_piece"
        if pt == "Recovery":
            return "recovery"
        if pt == "Interception":
            return "interception"
        return "open_play"
    if t == "Shot":
        st = (e.get("shot", {}).get("type") or {}).get("name")
        return "set_piece" if st in ("Penalty", "Free Kick", "Corner") else "open_play"
    if t == "Ball Recovery":
        return "recovery"
    if t == "Interception":
        return "interception"
    if t in ("Duel", "50/50"):
        return "duel"
    if t == "Goal Keeper":
        return "keeper"
    if t in ("Carry", "Dribble", "Ball Receipt*", "Clearance", "Block"):
        return "open_play"
    return "other"


def _name(d: dict[str, Any] | None, key: str = "name") -> str | None:
    return None if not d else d.get(key)


def _location_of(e: dict[str, Any]) -> tuple[float, float] | None:
    loc = e.get("location")
    if not loc or len(loc) < 2 or loc[0] is None or loc[1] is None:
        return None
    return float(loc[0]), float(loc[1])


def compute_event_features(events: list[dict[str, Any]], home_team_id: int,
                           cfg: EventFeatureConfig | None = None
                           ) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Per-event feature rows for every event of one match (in index order).

    Args:
        events: raw StatsBomb events sorted by ``index``.
        home_team_id: id of the home team (for the ``f_home`` flag).
        cfg: window configuration.

    Returns:
        ``(rows, seq)`` where ``rows[i]`` is the feature dict of the i-th *kept* event
        (period 5 dropped when configured) and ``seq`` is ``[n_kept, seq_len * 5]``
        float32 in the order of :func:`seq_columns` (``nan`` / 0 padding).
    """
    cfg = cfg or EventFeatureConfig()
    seq_len = cfg.seq_len
    rows: list[dict[str, Any]] = []
    seq_rows: list[np.ndarray] = []
    score: dict[int, int] = {}
    poss = _Possession()
    cur_period = -1
    hist: deque[_Hist] = deque(maxlen=max(seq_len, cfg.window_k, 300))
    last_def_t: dict[int, float] = {}
    prev_t: float | None = None
    team_ids = sorted({int(e["team"]["id"]) for e in events if "team" in e})

    for e in events:
        type_name = e["type"]["name"]
        period = int(e.get("period", 1))
        if cfg.drop_period_5 and period >= 5:
            continue
        t = event_time(e)
        team_id = int(e["team"]["id"])
        opp_id = next((x for x in team_ids if x != team_id), -1)
        poss_team_id = int(e["possession_team"]["id"])
        loc = _location_of(e)

        # ---- period reset ---------------------------------------------------------
        if period != cur_period:
            cur_period = period
            hist.clear()
            last_def_t = {}
            prev_t = None
        # ---- timestamp glitch repair ----------------------------------------------
        ts_repaired = prev_t is not None and t < prev_t - cfg.max_backwards_s
        if ts_repaired:
            t = prev_t

        # ---- possession bookkeeping ------------------------------------------------
        pidx = int(e["possession"])
        if pidx != poss.idx:
            poss = _Possession(idx=pidx, team_id=poss_team_id, start_t=t)
        if loc is not None:
            # location in the possession team's frame
            px, py = (loc if team_id == poss_team_id else flip_xy(*loc))
            if math.isnan(poss.start_x):
                poss.start_x, poss.start_y = px, py
            in_ft = px >= cfg.final_third_x
            if in_ft and not poss.prev_in_ft:
                poss.last_ft_entry_t = t
            poss.prev_in_ft = in_ft
        else:
            px = py = float("nan")
        # First located event by the possession team decides how it started (skips
        # unlocated admin events such as Half Start / Camera On).
        if poss.start_type == "unknown" and team_id == poss_team_id and loc is not None:
            poss.start_type = possession_start_type(e)

        # ---- score state (before this event) ---------------------------------------
        sf, sa = score.get(team_id, 0), score.get(opp_id, 0)

        # ---- geometry -----------------------------------------------------------
        if loc is not None:
            x, y = loc
            dist, opening, bearing = goal_geometry(x, y)
        else:
            x = y = dist = opening = bearing = float("nan")

        row: dict[str, Any] = {
            "event_id": e["id"], "event_index": int(e["index"]), "period": period,
            "team_id": team_id, "opp_team_id": opp_id, "possession_team_id": poss_team_id,
            "player_id": (e.get("player") or {}).get("id"),
            "f_type": type_name, "f_type_id": TYPE_VOCAB.get(type_name, UNKNOWN_TYPE_ID),
            "f_x": x, "f_y": y, "f_dist_goal": dist, "f_goal_opening": opening,
            "f_goal_bearing": bearing, "f_has_location": loc is not None,
            "f_play_pattern": _name(e.get("play_pattern")),
            "f_position": _name(e.get("position")),
            "f_under_pressure": bool(e.get("under_pressure", False)),
            "f_counterpress": bool(e.get("counterpress", False)),
            "f_after_duration": (float(e["duration"]) if e.get("duration") is not None
                                 else float("nan")),
            "f_minute": float(e["minute"]) + float(e["second"]) / 60.0,
            "f_t_period": timestamp_seconds(e["timestamp"]),
            "f_ts_repaired": bool(ts_repaired),
            "f_period": period,
            "f_home": team_id == home_team_id,
            "f_is_possession_team": team_id == poss_team_id,
            "f_score_for": sf, "f_score_against": sa, "f_score_diff": sf - sa,
            "f_goals_total": sf + sa,
            "f_dt_prev": (t - prev_t) if prev_t is not None else float("nan"),
            # possession context
            "f_poss_idx": pidx,
            "f_poss_elapsed": t - poss.start_t,
            "f_poss_n_events": poss.n_events,
            "f_poss_n_passes": poss.n_passes,
            "f_poss_ball_dist": poss.ball_dist,
            "f_poss_start_type": poss.start_type,
            "f_poss_t_since_ft_entry": (t - poss.last_ft_entry_t)
            if not math.isnan(poss.last_ft_entry_t) else float("nan"),
            "f_poss_in_final_third": bool(px >= cfg.final_third_x) if not math.isnan(px)
            else False,
        }
        # possession start location in the *event* team's frame
        if not math.isnan(poss.start_x):
            sx, sy = ((poss.start_x, poss.start_y) if team_id == poss_team_id
                      else flip_xy(poss.start_x, poss.start_y))
        else:
            sx = sy = float("nan")
        row["f_poss_start_x"], row["f_poss_start_y"] = sx, sy

        # ---- type-specific attributes ----------------------------------------------
        _fill_type_attributes(row, e, type_name, loc)

        # ---- windows over history (same period, strictly previous) -----------------
        _fill_window_features(row, hist, t, team_id, opp_id, cfg)
        ldt = last_def_t.get(opp_id)
        row["f_t_since_opp_def_action"] = (t - ldt) if ldt is not None else float("nan")

        # ---- sequence block ------------------------------------------------------
        seq_rows.append(_sequence_block(hist, t, team_id, seq_len))
        rows.append(row)

        # ---- update state with this event --------------------------------------------
        if loc is not None:
            hist.append(_Hist(t, team_id, poss_team_id, type_name, x, y))
            prev_t = t
            poss.n_events += 1
        if type_name in DEF_ACTION_TYPES:
            last_def_t[team_id] = t
        if team_id == poss_team_id:
            if type_name == "Pass":
                poss.n_passes += 1
                ln = e.get("pass", {}).get("length")
                if ln is not None:
                    poss.ball_dist += float(ln)
            elif type_name == "Carry" and loc is not None:
                end = e.get("carry", {}).get("end_location")
                if end:
                    poss.ball_dist += math.hypot(end[0] - x, end[1] - y)
        is_goal = (type_name == "Own Goal For" or (
            type_name == "Shot" and _name(e.get("shot", {}).get("outcome")) == "Goal"))
        if period <= 4 and is_goal:
            score[team_id] = score.get(team_id, 0) + 1

    seq = (np.vstack(seq_rows).astype(np.float32) if seq_rows
           else np.zeros((0, seq_len * len(SEQ_FIELDS)), dtype=np.float32))
    return rows, seq


def _fill_type_attributes(row: dict[str, Any], e: dict[str, Any], type_name: str,
                          loc: tuple[float, float] | None) -> None:
    nan = float("nan")
    row.update({
        "f_pass_type": None, "f_pass_body_part": None,
        "f_after_pass_length": nan, "f_after_pass_angle": nan, "f_after_pass_height": None,
        "f_after_pass_switch": False, "f_after_pass_cross": False,
        "f_after_pass_through_ball": False, "f_after_pass_cut_back": False,
        "f_after_pass_end_x": nan, "f_after_pass_end_y": nan,
        "f_after_pass_end_dist_goal": nan, "f_after_pass_progress": nan,
        "post_pass_outcome": None,
        "f_after_carry_length": nan, "f_after_carry_end_x": nan, "f_after_carry_end_y": nan,
        "f_after_carry_progress": nan,
        "f_shot_body_part": None, "f_shot_technique": None, "f_shot_type": None,
        "f_shot_first_time": False, "post_shot_outcome": None,
        "oracle_statsbomb_xg": nan, "oracle_one_on_one": nan, "oracle_open_goal": nan,
    })
    if type_name == "Pass":
        p = e.get("pass", {})
        end = p.get("end_location")
        row["f_after_pass_length"] = float(p["length"]) if p.get("length") is not None else nan
        row["f_after_pass_angle"] = float(p["angle"]) if p.get("angle") is not None else nan
        row["f_after_pass_height"] = _name(p.get("height"))
        row["f_pass_type"] = _name(p.get("type")) or "Open Play"
        row["f_pass_body_part"] = _name(p.get("body_part"))
        row["f_after_pass_switch"] = bool(p.get("switch", False))
        row["f_after_pass_cross"] = bool(p.get("cross", False))
        row["f_after_pass_through_ball"] = bool(p.get("through_ball", False))
        row["f_after_pass_cut_back"] = bool(p.get("cut_back", False))
        row["post_pass_outcome"] = _name(p.get("outcome")) or "Complete"
        if end and len(end) >= 2:
            row["f_after_pass_end_x"], row["f_after_pass_end_y"] = float(end[0]), float(end[1])
            row["f_after_pass_end_dist_goal"] = math.hypot(GOAL_X - end[0], GOAL_Y - end[1])
            if loc is not None:
                row["f_after_pass_progress"] = float(end[0]) - loc[0]
    elif type_name == "Carry":
        end = e.get("carry", {}).get("end_location")
        if end and len(end) >= 2:
            row["f_after_carry_end_x"], row["f_after_carry_end_y"] = float(end[0]), float(end[1])
            if loc is not None:
                row["f_after_carry_length"] = math.hypot(end[0] - loc[0], end[1] - loc[1])
                row["f_after_carry_progress"] = float(end[0]) - loc[0]
    elif type_name == "Shot":
        s = e.get("shot", {})
        row["f_shot_body_part"] = _name(s.get("body_part"))
        row["f_shot_technique"] = _name(s.get("technique"))
        row["f_shot_type"] = _name(s.get("type"))
        row["f_shot_first_time"] = bool(s.get("first_time", False))
        row["post_shot_outcome"] = _name(s.get("outcome"))
        row["oracle_statsbomb_xg"] = (float(s["statsbomb_xg"]) if s.get("statsbomb_xg")
                                      is not None else nan)
        row["oracle_one_on_one"] = float(bool(s.get("one_on_one", False)))
        row["oracle_open_goal"] = float(bool(s.get("open_goal", False)))


def _fill_window_features(row: dict[str, Any], hist: deque[_Hist], t: float, team_id: int,
                          opp_id: int, cfg: EventFeatureConfig) -> None:
    """K-event type counts, 60 s opponent defensive x, ball speed, opponent possession."""
    k_win = cfg.window_k
    counts = dict.fromkeys(WINDOW_GROUPS, 0)
    n_own = n_opp_def = 0
    lastk = list(hist)[-k_win:] if hist else []
    for h in lastk:
        counts[WINDOW_GROUP.get(h.type_name, "other")] += 1
        if h.team_id == team_id:
            n_own += 1
        elif h.type_name in DEF_ACTION_TYPES:
            n_opp_def += 1
    for g in WINDOW_GROUPS:
        row[f"f_w{k_win}_n_{g}"] = counts[g]
    row[f"f_w{k_win}_n_own"] = n_own
    row[f"f_w{k_win}_n_opp_def"] = n_opp_def
    row[f"f_w{k_win}_n"] = len(lastk)

    # 60 s opponent defensive actions, in the event team's frame (120 - x_opp).
    sx = 0.0
    n_def = 0
    opp_poss = False
    for h in reversed(hist):
        age = t - h.t
        if age > cfg.def_window_s and age > cfg.opp_poss_window_s:
            break
        if age <= cfg.def_window_s and h.team_id == opp_id and h.type_name in DEF_ACTION_TYPES:
            sx += PITCH_X - h.x
            n_def += 1
        if age <= cfg.opp_poss_window_s and h.poss_team_id == opp_id:
            opp_poss = True
    row["f_opp_def_x_60s_mean"] = sx / n_def if n_def else float("nan")
    row["f_opp_def_n_60s"] = n_def
    row["f_opp_poss_last10s"] = opp_poss

    # Ball speed over the last N transitions (path length / elapsed time, elapsed floored
    # at ``min_speed_dt_s`` so simultaneous events do not explode the estimate).
    n = cfg.speed_n_events
    pts = list(hist)[-(n + 1):]
    if len(pts) >= 2:
        path = 0.0
        prev = None
        for h in pts:
            hx, hy = (h.x, h.y) if h.team_id == team_id else flip_xy(h.x, h.y)
            if prev is not None:
                path += math.hypot(hx - prev[0], hy - prev[1])
            prev = (hx, hy)
        row["f_ball_speed_3"] = path / max(pts[-1].t - pts[0].t, cfg.min_speed_dt_s)
    else:
        row["f_ball_speed_3"] = float("nan")
    # Displacement of the ball from the event `n` steps back to the previous event.
    if len(pts) >= 2:
        h0, h1 = pts[0], pts[-1]
        x0, y0 = (h0.x, h0.y) if h0.team_id == team_id else flip_xy(h0.x, h0.y)
        x1, y1 = (h1.x, h1.y) if h1.team_id == team_id else flip_xy(h1.x, h1.y)
        row["f_ball_dx_3"] = x1 - x0
        row["f_ball_dy_3"] = y1 - y0
    else:
        row["f_ball_dx_3"] = row["f_ball_dy_3"] = float("nan")


def _sequence_block(hist: deque[_Hist], t: float, team_id: int, seq_len: int) -> np.ndarray:
    """Fixed-width ``[seq_len * 5]`` encoding of the last ``seq_len`` located events.

    Slot 1 is the most recent previous event.  Fields per slot: type id (0 = pad), x, y
    (event team's frame, nan pad), dt = seconds before the current event (nan pad),
    same-team flag (1 own, 0 opponent, -1 pad).
    """
    n_fields = len(SEQ_FIELDS)
    out = np.full(seq_len * n_fields, np.nan, dtype=np.float32)
    out[0::n_fields] = 0.0
    out[4::n_fields] = -1.0
    for k, h in enumerate(reversed(hist)):
        if k >= seq_len:
            break
        same = h.team_id == team_id
        hx, hy = (h.x, h.y) if same else flip_xy(h.x, h.y)
        base = k * n_fields
        out[base] = TYPE_VOCAB.get(h.type_name, UNKNOWN_TYPE_ID)
        out[base + 1] = hx
        out[base + 2] = hy
        out[base + 3] = t - h.t
        out[base + 4] = 1.0 if same else 0.0
    return out


@dataclass
class TypeVocabRow:
    """One row of the type vocabulary table written next to the outputs."""

    type_name: str
    type_id: int
    window_group: str = field(default="other")


def type_vocab_rows() -> list[TypeVocabRow]:
    """The sequence-block vocabulary as rows (for the report / parquet)."""
    rows = [TypeVocabRow(n, i, WINDOW_GROUP.get(n, "other")) for n, i in TYPE_VOCAB.items()]
    rows.append(TypeVocabRow("<unknown>", UNKNOWN_TYPE_ID, "other"))
    rows.append(TypeVocabRow("<pad>", 0, "pad"))
    return sorted(rows, key=lambda r: r.type_id)
