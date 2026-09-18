"""Tests for the event-only feature builder on a tiny synthetic match."""
from __future__ import annotations

import math

import numpy as np

from research.privileged_tracking.soccer import event_features as ef

A, B = 1, 2  # team ids; A is home


def _ts(t: float) -> str:
    m, s = divmod(t, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def mk(idx: int, type_name: str, team: int, t: float, loc: list[float] | None = None,
       period: int = 1, poss: int = 1, poss_team: int = A, **extra: object) -> dict:
    """Build one synthetic StatsBomb-like event."""
    minute_off = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}[period]
    e: dict = {
        "id": f"ev{idx}", "index": idx, "period": period, "timestamp": _ts(t),
        "minute": minute_off + int(t // 60), "second": int(t % 60),
        "type": {"name": type_name}, "possession": poss,
        "possession_team": {"id": poss_team}, "team": {"id": team},
        "play_pattern": {"name": "Regular Play"}, "player": {"id": 10 * team + idx},
        "position": {"name": "Center Forward"},
    }
    if loc is not None:
        e["location"] = loc
    e.update(extra)
    return e


def synthetic_events() -> list[dict]:
    return [
        mk(0, "Half Start", A, 0.0),
        mk(1, "Pass", A, 0.0, [60, 42], duration=1.5,
           **{"pass": {"length": math.hypot(10, 3), "angle": 0.29, "end_location": [70, 45],
                       "height": {"name": "Ground Pass"}, "type": {"name": "Kick Off"},
                       "body_part": {"name": "Right Foot"}}}),
        mk(2, "Ball Receipt*", A, 1.5, [70, 45]),
        mk(3, "Carry", A, 1.5, [70, 45], duration=2.0, carry={"end_location": [80, 50]}),
        # B presses: B's frame (35, 28) == A's frame (85, 52): ball enters the final third
        mk(4, "Pressure", B, 2.5, [35, 28], duration=0.6, under_pressure=True),
        mk(5, "Pass", A, 3.5, [80, 50], duration=1.0, under_pressure=True,
           **{"pass": {"length": math.hypot(20, 10), "angle": -0.46, "end_location": [100, 40],
                       "height": {"name": "Ground Pass"}, "body_part": {"name": "Left Foot"},
                       "through_ball": True, "outcome": {"name": "Incomplete"}}}),
        mk(6, "Shot", A, 6.0, [100, 40], duration=0.9,
           shot={"outcome": {"name": "Goal"}, "statsbomb_xg": 0.3, "type": {"name": "Open Play"},
                 "body_part": {"name": "Right Foot"}, "technique": {"name": "Normal"},
                 "first_time": True, "one_on_one": True,
                 "freeze_frame": [{"teammate": False, "location": [118, 40],
                                   "position": {"name": "Goalkeeper"}}]}),
        mk(7, "Pass", B, 8.0, [60, 40], poss=2, poss_team=B, duration=1.0,
           **{"pass": {"length": 5.0, "angle": 3.1, "end_location": [55, 40],
                       "type": {"name": "Kick Off"}}}),
        mk(8, "Pressure", A, 9.0, [55, 35], poss=2, poss_team=B, duration=0.5),
        mk(9, "Half End", A, 60.0, poss=2, poss_team=B),
        mk(10, "Half Start", B, 0.0, period=2, poss=3, poss_team=B),
        mk(11, "Pass", B, 0.0, [60, 40], period=2, poss=3, poss_team=B,
           **{"pass": {"length": 5.0, "angle": 3.1, "end_location": [55, 40],
                       "type": {"name": "Kick Off"}}}),
        mk(12, "Shot", A, 30.0, [110, 40], period=5, poss=4, poss_team=A,
           shot={"outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}),
    ]


def test_time_helpers() -> None:
    assert math.isclose(ef.timestamp_seconds("00:03:39.851"), 219.851)
    e = {"period": 2, "timestamp": "00:01:00.000"}
    assert math.isclose(ef.event_time(e), 45 * 60 + 60)
    d, opening, bearing = ef.goal_geometry(108.0, 40.0)
    assert math.isclose(d, 12.0) and math.isclose(opening, 2 * math.atan(4 / 12))
    assert math.isclose(bearing, 0.0)
    d0, o0, _ = ef.goal_geometry(120.0, 40.0)
    assert d0 == 0.0 and o0 == 0.0
    assert ef.flip_xy(35.0, 28.0) == (85.0, 52.0)


def test_possession_start_type() -> None:
    assert ef.possession_start_type({"type": {"name": "Pass"},
                                     "pass": {"type": {"name": "Kick Off"}}}) == "kick_off"
    assert ef.possession_start_type({"type": {"name": "Pass"},
                                     "pass": {"type": {"name": "Throw-in"}}}) == "set_piece"
    assert ef.possession_start_type({"type": {"name": "Pass"}, "pass": {}}) == "open_play"
    assert ef.possession_start_type({"type": {"name": "Ball Recovery"}}) == "recovery"
    assert ef.possession_start_type({"type": {"name": "Interception"}}) == "interception"
    assert ef.possession_start_type({"type": {"name": "Foul Won"}}) == "other"


def test_compute_event_features_end_to_end() -> None:
    cfg = ef.EventFeatureConfig()
    rows, seq = ef.compute_event_features(synthetic_events(), home_team_id=A, cfg=cfg)
    by = {r["event_index"]: r for r in rows}
    # period 5 dropped, everything else kept in order
    assert [r["event_index"] for r in rows] == list(range(12))
    assert seq.shape == (12, cfg.seq_len * len(ef.SEQ_FIELDS))
    assert len(ef.seq_columns(cfg.seq_len)) == seq.shape[1]

    e1 = by[1]
    assert e1["f_type"] == "Pass" and e1["f_type_id"] == ef.TYPE_VOCAB["Pass"]
    assert e1["f_home"] is True and e1["f_is_possession_team"] is True
    assert e1["f_score_diff"] == 0 and e1["f_poss_n_events"] == 0
    assert e1["f_poss_start_type"] == "kick_off" and e1["f_pass_type"] == "Kick Off"
    assert math.isclose(e1["f_poss_start_x"], 60) and math.isclose(e1["f_poss_start_y"], 42)
    assert e1["post_pass_outcome"] == "Complete" and math.isclose(e1["f_after_pass_progress"], 10)
    assert math.isnan(e1["f_dt_prev"]) and e1["f_w10_n"] == 0
    assert math.isnan(e1["f_ball_speed_3"])

    e3 = by[3]
    assert e3["f_poss_n_events"] == 2 and e3["f_poss_n_passes"] == 1
    assert math.isclose(e3["f_poss_ball_dist"], math.hypot(10, 3))
    assert math.isclose(e3["f_poss_elapsed"], 1.5)
    assert e3["f_w10_n_pass"] == 1 and e3["f_w10_n_receipt"] == 1 and e3["f_w10_n_own"] == 2
    assert math.isnan(e3["f_opp_def_x_60s_mean"]) and e3["f_opp_def_n_60s"] == 0
    assert math.isclose(e3["f_ball_speed_3"], math.hypot(10, 3) / 1.5)
    assert math.isclose(e3["f_after_carry_length"], math.hypot(10, 5))
    assert math.isnan(e3["f_poss_t_since_ft_entry"])

    e4 = by[4]  # opponent pressure, coordinates stay in B's own frame
    assert e4["f_is_possession_team"] is False and e4["f_home"] is False
    assert e4["f_x"] == 35 and e4["f_y"] == 28 and e4["f_under_pressure"] is True
    assert e4["f_w10_n_own"] == 0 and e4["f_w10_n"] == 3
    assert math.isclose(e4["f_poss_start_x"], 60) and math.isclose(e4["f_poss_start_y"], 38)
    assert e4["f_poss_in_final_third"] is True  # (85, 52) in A's frame
    # sequence slot 1 for e4 is the carry at (70, 45) in A's frame -> (50, 35) in B's
    s4 = seq[4]
    assert s4[0] == ef.TYPE_VOCAB["Carry"] and s4[1] == 50 and s4[2] == 35
    assert math.isclose(s4[3], 1.0) and s4[4] == 0

    e5 = by[5]
    assert math.isclose(e5["f_opp_def_x_60s_mean"], 85.0) and e5["f_opp_def_n_60s"] == 1
    assert math.isclose(e5["f_t_since_opp_def_action"], 1.0)
    assert e5["f_opp_poss_last10s"] is False
    assert e5["f_w10_n_pressure"] == 1 and e5["f_w10_n_opp_def"] == 1 and e5["f_w10_n_own"] == 3
    assert math.isclose(e5["f_poss_t_since_ft_entry"], 1.0)
    assert e5["f_after_pass_through_ball"] is True and e5["post_pass_outcome"] == "Incomplete"
    assert math.isclose(e5["f_after_pass_end_dist_goal"], 20.0)
    assert math.isclose(e5["f_poss_ball_dist"], math.hypot(10, 3) + math.hypot(10, 5))

    e6 = by[6]
    assert e6["f_score_diff"] == 0  # before the goal
    assert e6["post_shot_outcome"] == "Goal" and math.isclose(e6["oracle_statsbomb_xg"], 0.3)
    assert e6["oracle_one_on_one"] == 1.0 and e6["f_shot_first_time"] is True
    assert math.isnan(e6["f_after_pass_length"]) and e6["f_pass_type"] is None
    assert math.isclose(e6["f_dist_goal"], 20.0)

    e7 = by[7]  # B restarts after conceding
    assert e7["f_score_for"] == 0 and e7["f_score_against"] == 1 and e7["f_score_diff"] == -1
    assert e7["f_poss_elapsed"] == 0 and e7["f_poss_n_events"] == 0
    assert e7["f_poss_start_type"] == "kick_off"
    assert e7["f_opp_poss_last10s"] is True
    assert math.isnan(e7["f_t_since_opp_def_action"])

    e8 = by[8]
    assert e8["f_score_diff"] == 1 and e8["f_opp_poss_last10s"] is True
    s8 = seq[8]
    assert s8[0] == ef.TYPE_VOCAB["Pass"] and s8[4] == 0 and math.isclose(s8[3], 1.0)
    assert s8[5] == ef.TYPE_VOCAB["Shot"] and s8[6] == 100 and s8[9] == 1
    # seven located events precede e8 -> slot 8 is padding
    pad = 7 * len(ef.SEQ_FIELDS)
    assert s8[pad] == 0 and math.isnan(s8[pad + 1]) and s8[pad + 4] == -1

    e11 = by[11]  # first located event of period 2: windows reset
    assert e11["f_w10_n"] == 0 and math.isnan(e11["f_dt_prev"])
    assert seq[11][0] == 0 and seq[11][4] == -1
    assert e11["f_score_for"] == 0 and e11["f_score_against"] == 1
    assert e11["f_period"] == 2 and math.isclose(e11["f_minute"], 45.0)


def test_timestamp_glitch_is_repaired() -> None:
    events = [
        mk(0, "Pass", A, 1500.0, [60, 40], **{"pass": {"length": 5.0, "end_location": [65, 40]}}),
        # receipt stamped at 00:00:00.3 although it follows a pass at 25:00 -> repaired
        mk(1, "Ball Receipt*", A, 0.3, [65, 40]),
        mk(2, "Carry", A, 1501.0, [65, 40], carry={"end_location": [70, 40]}),
    ]
    rows, seq = ef.compute_event_features(events, home_team_id=A)
    assert rows[1]["f_ts_repaired"] is True and rows[0]["f_ts_repaired"] is False
    assert math.isclose(rows[1]["f_dt_prev"], 0.0) and math.isclose(rows[1]["f_poss_elapsed"], 0.0)
    assert math.isclose(seq[1][3], 0.0)  # dt to the pass, not -1499.7
    assert math.isclose(rows[2]["f_dt_prev"], 1.0) and rows[2]["f_w10_n"] == 2
    assert math.isclose(rows[2]["f_poss_elapsed"], 1.0)


def test_type_vocab_rows_cover_pad_and_unknown() -> None:
    rows = ef.type_vocab_rows()
    ids = [r.type_id for r in rows]
    assert ids == sorted(ids) and 0 in ids and ef.UNKNOWN_TYPE_ID in ids
    assert len(set(ids)) == len(ids)
    assert np.all(np.diff(ids) >= 1)
