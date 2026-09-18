"""Unit tests for the pure NFL tracking feature functions on a synthetic play.

The synthetic play is authored in offense-normalised coordinates (offense attacks +x,
official LOS at x=50, ball at y=26.65) and rendered to raw coordinates either as-is or
mirrored, so every target must be invariant to the raw orientation.

Lateral sign convention pinned here: positive ``lateral`` (larger y) is the offense's
LEFT, matching nflfastR ``pass_location`` on the real data. In the play below WR 10
(y = BY + 21) and WR 11 (y = BY + 9.4) are on the offense's left, WR 9 (y = BY - 21,
motions to BY - 14) is on the offense's right.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.privileged_tracking.nfl import tracking_features as tf

LOS = 50.0
BY = 26.65
N_FRAMES = 40
SNAP = 10  # frame index (0-based); frame ids are 1..40


def _keyframes(**kf: tuple[float, float]) -> dict[int, tuple[float, float]]:
    return {int(k[1:]): v for k, v in kf.items()}


def synthetic_play(flip: bool, run_play: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    """Return ``(spec, long_arrays)`` for a synthetic 11 v 11 play.

    Offense is ``home``. Positions are piecewise-constant keyframes ``{frame_idx: (x, y)}``
    in normalised coordinates.
    """
    off = {
        1: {"pos": "C", "kf": _keyframes(f0=(LOS - 0.6, BY))},
        2: {"pos": "G", "kf": _keyframes(f0=(LOS - 1.0, BY - 1.4))},
        3: {"pos": "G", "kf": _keyframes(f0=(LOS - 1.0, BY + 1.4))},
        4: {"pos": "T", "kf": _keyframes(f0=(LOS - 1.1, BY - 2.9))},
        5: {"pos": "T", "kf": _keyframes(f0=(LOS - 1.1, BY + 2.9))},
        6: {"pos": "QB", "kf": _keyframes(f0=(LOS - 5.0, BY))},                 # shotgun
        7: {"pos": "RB", "kf": _keyframes(f0=(LOS - 6.0, BY + 2.0))},           # backfield
        8: {"pos": "TE", "kf": _keyframes(f0=(LOS - 1.3, BY + 4.8))},           # inline TE
        9: {"pos": "WR", "kf": _keyframes(f0=(LOS - 1.5, BY - 21.0), f5=(LOS - 1.5, BY - 14.0),
                                          f11=(LOS + 2.0, BY - 14.0), f20=(LOS + 6.0, BY - 14.5),
                                          f30=(LOS + 10.0, BY - 14.65))},        # motion 7 yd, then target
        10: {"pos": "WR", "kf": _keyframes(f0=(LOS - 1.5, BY + 21.0))},         # wide, offense's LEFT
        11: {"pos": "WR", "kf": _keyframes(f0=(LOS - 1.5, BY + 9.4))},          # left slot (>= 8 -> wide)
    }
    de = {
        21: {"pos": "DE", "kf": _keyframes(f0=(LOS + 0.7, BY - 2.6), f12=(LOS - 2.0, BY - 2.6))},
        22: {"pos": "DT", "kf": _keyframes(f0=(LOS + 0.7, BY - 0.5), f12=(LOS - 1.5, BY - 0.5),
                                           f25=(LOS - 2.5, BY), f30=(LOS - 3.5, BY))},  # 1.5 yd from QB at throw
        23: {"pos": "DT", "kf": _keyframes(f0=(LOS + 0.7, BY + 1.6), f12=(LOS - 2.0, BY + 1.6))},
        24: {"pos": "DE", "kf": _keyframes(f0=(LOS + 0.7, BY + 4.0))},           # does not rush
        25: {"pos": "LB", "kf": _keyframes(f0=(LOS + 4.0, BY - 1.5))},
        26: {"pos": "LB", "kf": _keyframes(f0=(LOS + 4.0, BY + 2.5))},
        27: {"pos": "CB", "kf": _keyframes(f0=(LOS + 7.0, BY - 21.0), f5=(LOS + 7.0, BY - 14.0),
                                           f30=(LOS + 11.5, BY - 15.5))},        # covers the target
        28: {"pos": "CB", "kf": _keyframes(f0=(LOS + 6.0, BY + 21.0))},           # 7.5 yd off WR 10 (left)
        29: {"pos": "FS", "kf": _keyframes(f0=(LOS + 12.0, BY - 6.0))},           # deep
        30: {"pos": "SS", "kf": _keyframes(f0=(LOS + 12.0, BY + 7.0))},           # deep
        31: {"pos": "CB", "kf": _keyframes(f0=(LOS + 5.5, BY + 9.4))},            # nickel, outside naive box depth
    }
    ball_kf = _keyframes(f0=(LOS - 0.4, BY), f11=(LOS - 5.0, BY), f30=(LOS - 5.0, BY),
                         f36=(LOS + 9.8, BY - 14.5))
    events = {SNAP: "ball_snap", 30: "pass_forward", 36: "pass_arrived", 37: "pass_outcome_caught"}
    if run_play:
        # handoff at frame 14 with the ball on the RB, first contact at 25 three yards past the line
        off[7]["kf"] = _keyframes(f0=(LOS - 6.0, BY + 2.0), f14=(LOS - 4.0, BY + 1.0),
                                  f25=(LOS + 5.0, BY + 1.0))
        for pid in (21, 22, 23):  # the line holds at the LOS on the run variant
            de[pid]["kf"] = {0: de[pid]["kf"][0]}
        de[25]["kf"] = _keyframes(f0=(LOS + 4.0, BY - 1.5), f25=(LOS + 4.0, BY + 1.5))
        de[26]["kf"] = _keyframes(f0=(LOS + 4.0, BY + 2.5), f25=(LOS + 5.0, BY + 2.0))
        ball_kf = _keyframes(f0=(LOS - 0.4, BY), f11=(LOS - 5.0, BY), f14=(LOS - 4.0, BY + 1.0),
                             f25=(LOS + 5.0, BY + 1.0))
        events = {SNAP: "ball_snap", 14: "handoff", 25: "first_contact", 28: "tackle"}

    def at(kf: dict[int, tuple[float, float]], f: int) -> tuple[float, float]:
        keys = [k for k in kf if k <= f]
        return kf[max(keys)]

    rows = {k: [] for k in ("frame_id", "player_id", "team", "x", "y", "s", "dir", "event")}

    def emit(f: int, pid: float, team: str, x: float, y: float) -> None:
        if flip:
            x, y = tf.FIELD_LENGTH - x, tf.FIELD_WIDTH - y
        rows["frame_id"].append(f + 1)
        rows["player_id"].append(pid)
        rows["team"].append(team)
        rows["x"].append(x)
        rows["y"].append(y)
        rows["s"].append(1.0)
        rows["dir"].append(90.0 if not flip else 270.0)
        rows["event"].append(events.get(f))

    for f in range(N_FRAMES):
        for pid, d in off.items():
            emit(f, pid, "home", *at(d["kf"], f))
        for pid, d in de.items():
            emit(f, pid, "away", *at(d["kf"], f))
        emit(f, np.nan, "ball", *at(ball_kf, f))
    arrays = {k: np.array(v, dtype=object if k in ("team", "event") else float) for k, v in rows.items()}
    positions = {**{k: v["pos"] for k, v in off.items()}, **{k: v["pos"] for k, v in de.items()}}
    return {"positions": positions}, arrays


def build(flip: bool, run_play: bool = False) -> tuple[tf.PlayFrames, int, dict]:
    spec, a = synthetic_play(flip, run_play)
    pf = tf.PlayFrames.from_long(a["frame_id"], a["player_id"], a["team"], a["x"], a["y"], a["s"],
                                 a["dir"], a["event"], offense_label="home", los_official=LOS)
    qb_idx = int(np.where(pf.player_ids == 6)[0][0])
    return pf, qb_idx, spec


@pytest.mark.parametrize("flip", [False, True])
def test_orientation_and_references(flip: bool) -> None:
    pf, qb_idx, _ = build(flip)
    assert pf.flipped is flip
    assert pf.snap_idx == SNAP
    assert pf.los_x == LOS
    assert pf.los_x_ball == pytest.approx(LOS - 0.4)
    assert pf.ball_y_ref == pytest.approx(BY)
    assert pf.ball_y_source == "ball"
    assert pf.off_mask.sum() == 11 and pf.def_mask.sum() == 11
    # offense median x is below the defense median after normalisation
    si = pf.snap_idx
    assert np.median(pf.x[si][pf.off_mask]) < np.median(pf.x[si][pf.def_mask])
    # dir was rotated on flip so motion still points the same way in field terms
    assert np.allclose(pf.dir[0], 90.0)


def test_infer_attack_direction_and_flip() -> None:
    x = np.array([40.0, 41.0, 39.0, 60.0, 61.0, 62.0])
    off = np.array([True, True, True, False, False, False])
    assert tf.infer_attack_direction(x, off) == 1
    assert tf.infer_attack_direction(120 - x, off) == -1
    fx, fy, fd = tf.flip_coordinates(np.array([10.0]), np.array([10.0]), np.array([350.0]))
    assert fx[0] == 110.0 and fy[0] == pytest.approx(43.3) and fd[0] == pytest.approx(170.0)


def test_official_los() -> None:
    assert tf.official_los("NE", 27, "NE") == 37.0
    assert tf.official_los("KC", 9, "NE") == 101.0
    assert tf.official_los(np.nan, 50, "NE") == 60.0
    assert np.isnan(tf.official_los("NE", np.nan, "NE"))


@pytest.mark.parametrize("flip", [False, True])
def test_presnap_targets(flip: bool) -> None:
    pf, qb_idx, _ = build(flip)
    out = tf.presnap_targets(pf, qb_idx)
    assert out["n_deep_safeties"] == 2 and out["mof_open"] == 1
    assert out["def_depth_max"] == pytest.approx(12.0) and out["def_depth_2nd"] == pytest.approx(12.0)
    assert out["box_count"] == 6          # 4 linemen + 2 LBs; nickel at 5.5 deep excluded
    assert out["n_dl"] == 4
    # left = positive lateral: WR 10 (+21) and slot WR 11 (+9.4, >= 8 counts as wide) are left,
    # WR 9 (-21) is right
    assert out["n_wide_left"] == 2 and out["n_wide_right"] == 1
    assert out["widest_split"] == pytest.approx(21.0)
    assert out["n_backfield"] == 1
    assert out["n_ol_derived"] == 5
    assert out["n_te_inline"] == 1
    assert out["qb_depth"] == pytest.approx(5.0) and out["shotgun_derived"] == 1
    assert out["motion_disp_max"] == pytest.approx(7.0) and out["motion_derived"] == 1
    assert out["motion_event"] == 0
    # cushion, offense perspective: left = WR 10 at (48.5, BY+21) vs CB 28 at (56, BY+21) -> 7.5 yd;
    # right = WR 9 (after motion, at lateral -14) vs CB 27 at (57, BY-14) -> 8.5 yd
    assert out["cushion_left"] == pytest.approx(7.5)
    assert out["cushion_right"] == pytest.approx(8.5)
    assert out["cb_cushion"] == pytest.approx(8.0)
    assert out["def_y_range"] == pytest.approx(35.0)   # right CB followed the motion to BY-14


def test_identify_qb_geometric_and_box_count() -> None:
    pf, qb_idx, _ = build(False)
    si = pf.snap_idx
    assert tf.identify_qb_geometric(pf.x[si], pf.y[si], pf.off_mask, pf.los_x, pf.ball_y_ref) == qb_idx
    depth = pf.x[si] - pf.los_x
    lateral = pf.y[si] - pf.ball_y_ref
    assert tf.box_count(depth, lateral, pf.def_mask, 5.0, 8.0) == 6
    assert tf.box_count(depth, lateral, pf.def_mask, 6.0, 8.0) == 6   # nickel is 9.4 wide
    assert tf.box_count(depth, lateral, pf.def_mask, 6.0, 10.0) == 7


@pytest.mark.parametrize("flip", [False, True])
def test_pass_targets(flip: bool) -> None:
    pf, qb_idx, _ = build(flip)
    out = tf.pass_targets(pf, qb_idx, "C", rush_grid=((1.0, -0.5),))
    assert out["release_event"] == "pass_forward"
    assert out["time_to_throw"] == pytest.approx(2.0)
    assert out["qb_id_throw"] == 6 and out["qb_same_as_presnap"] == 1
    assert out["qb_ball_dist_at_release"] == pytest.approx(0.0)
    assert out["qb_depth_at_throw"] == pytest.approx(5.0)
    assert out["min_def_dist_qb_throw"] == pytest.approx(1.5)
    assert out["min_def_dist_qb_throw_m05"] == pytest.approx(2.5)
    assert out["n_def_within_r_qb_throw"] == 1
    assert out["n_pass_rushers_derived"] == 3          # three linemen crossed, one stayed
    assert out[tf.rush_col(1.0, -0.5)] == 3
    assert out["arrival_event"] == "pass_arrived"
    assert out["target_id"] == 9
    assert out["target_depth"] == pytest.approx(10.0)
    assert out["target_lateral"] == pytest.approx(-14.65)
    assert out["target_side_derived"] == "right"       # negative lateral = offense's right
    assert out["separation_at_arrival"] == pytest.approx(np.hypot(1.5, 0.85))
    assert out["n_def_within_r_target"] == 1
    assert out["air_time_s"] == pytest.approx(0.6)
    assert out["ball_depth_at_arrival"] == pytest.approx(9.8)


def test_pass_targets_sack_without_tag() -> None:
    pf, qb_idx, _ = build(False)
    out = tf.pass_targets(pf, qb_idx, "S")
    assert out["release_event"] is None and np.isnan(out["time_to_throw"])
    # the run variant carries no arrival tag at all -> no target, no side
    pf_run, qb_run, _ = build(False, run_play=True)
    assert tf.pass_targets(pf_run, qb_run, "I")["target_side_derived"] is None


def test_lateral_sign_convention() -> None:
    """Positive lateral is the offense's left (LEFT_SIGN), for both raw orientations."""
    assert tf.LEFT_SIGN == 1.0
    assert tf.side_of_lateral(10.0, 6.0) == "left"
    assert tf.side_of_lateral(-10.0, 6.0) == "right"
    assert tf.side_of_lateral(6.0, 6.0) == "middle" and tf.side_of_lateral(-6.0, 6.0) == "middle"
    assert tf.side_of_lateral(float("nan"), 6.0) is None and tf.side_of_lateral(None, 6.0) is None
    for flip in (False, True):
        pf, qb_idx, _ = build(flip)
        rows = {int(r["player_id"]): r for r in tf.participant_rows(pf, qb_idx)}
        assert rows[10]["lateral"] == pytest.approx(21.0)    # offense's left wideout
        assert rows[9]["lateral"] == pytest.approx(-14.0)    # offense's right wideout (after motion)
        assert rows[10]["y_snap"] > rows[9]["y_snap"]        # normalised y is larger on the left


@pytest.mark.parametrize("flip", [False, True])
def test_run_targets(flip: bool) -> None:
    pf, qb_idx, _ = build(flip, run_play=True)
    out = tf.run_targets(pf, qb_idx)
    assert out["carrier_event"] == "handoff" and out["carrier_id"] == 7
    assert out["time_to_handoff"] == pytest.approx(0.4)
    assert out["carrier_depth_at_handoff"] == pytest.approx(4.0)
    assert out["yards_to_first_contact"] == pytest.approx(5.0)
    assert out["time_to_first_contact"] == pytest.approx(1.5)
    assert out["n_def_within_r_carrier_first_contact"] == 2
    assert out["n_def_within_r_carrier_handoff"] == 0


def test_participant_rows_roles() -> None:
    pf, qb_idx, spec = build(False)
    rows = tf.participant_rows(pf, qb_idx)
    roles = {int(r["player_id"]): r["role_derived"] for r in rows}
    assert roles[6] == "qb" and roles[1] == "ol" and roles[7] == "backfield"
    assert roles[9] == "wide" and roles[8] == "tight_slot"
    assert roles[21] == "line" and roles[25] == "box" and roles[29] == "deep" and roles[31] == "second_level"
    assert sum(r["in_box_naive"] for r in rows) == 6


def test_personnel_helpers() -> None:
    _, _, spec = build(False)
    pos = spec["positions"]
    off = tf.personnel_counts([pos[i] for i in range(1, 12)], "offense")
    assert off == {"RB": 1, "TE": 1, "WR": 3, "OL": 5, "QB": 1}
    de = tf.personnel_counts([pos[i] for i in range(21, 32)], "defense")
    assert de == {"DL": 4, "LB": 2, "DB": 5}
    assert tf.personnel_string(off, "offense") == "1 RB, 1 TE, 3 WR"
    assert tf.personnel_string({"OL": 6, "RB": 1, "TE": 2, "WR": 1}, "offense") == "6 OL, 1 RB, 2 TE, 1 WR"
    assert tf.personnel_string(de, "defense") == "4 DL, 2 LB, 5 DB"
    assert tf.parse_personnel("6 OL, 1 RB, 2 TE, 1 WR") == {"OL": 6, "RB": 1, "TE": 2, "WR": 1}
    assert tf.parse_personnel(np.nan) == {}
    assert tf.rush_col(1.5, -0.5) == "n_rush_w15_cm05" and tf.rush_col(2.0, 0.0) == "n_rush_w20_cp00"


def test_missing_snap_raises() -> None:
    _, a = synthetic_play(False)
    a["event"] = np.array([None] * len(a["event"]), dtype=object)
    with pytest.raises(ValueError):
        tf.PlayFrames.from_long(a["frame_id"], a["player_id"], a["team"], a["x"], a["y"], a["s"],
                                a["dir"], a["event"], offense_label="home")


def test_reference_lateral_fallback() -> None:
    pf, _, _ = build(False)
    si = pf.snap_idx
    y_ref, src = tf.reference_lateral(pf.x[si], pf.y[si], pf.off_mask, pf.los_x, ball_y=99.0)
    assert src == "centre_estimate" and y_ref == pytest.approx(BY)
    y_ref, src = tf.reference_lateral(pf.x[si], pf.y[si], pf.off_mask, pf.los_x, ball_y=BY + 0.3)
    assert src == "ball" and y_ref == pytest.approx(BY + 0.3)
    assert tf.estimate_centre_y(pf.x[si], pf.y[si], pf.off_mask, pf.los_x) == pytest.approx(BY)
