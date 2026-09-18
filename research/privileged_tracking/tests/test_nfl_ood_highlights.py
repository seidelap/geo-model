"""Tests for the pure highlight-frame functions of NFL 03b (no data access)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import ood_highlights as ood


def _play(direction: str, los: float = 60.0, with_snap: bool = True, ball_y: float | None = None,
          ball_dx: float = -0.4, positions: bool = False, frame: int = 7) -> pd.DataFrame:
    """11 v 11 plus the ball at one snap frame; offense attacks +x when ``direction == 'right'``.

    ``ball_y`` overrides the ball row's y (the formation stays centred on y = 26), ``ball_dx`` is the
    ball's offset from the LOS against the attacking direction, ``positions`` adds ``position`` /
    ``positionGroup`` columns (5 OL with one C).
    """
    sign = 1.0 if direction == "right" else -1.0
    centre_y = 26.0
    by = centre_y if ball_y is None else ball_y
    rows = []
    # offense: 5 OL 1 yd behind the line (C in the middle), QB 5 yd back, 2 WR wide, TE, RB, slot
    off = [(-1.0, centre_y + d, p, "OL") for d, p in ((-4, "T"), (-2, "G"), (0, "C"), (2, "G"), (4, "T"))]
    off += [(-5.0, centre_y, "QB", "QB"), (-0.5, centre_y + 20, "WR", "WR"), (-0.5, centre_y - 18, "WR", "WR"),
            (-1.0, centre_y + 6, "TE", "TE"), (-6.0, centre_y - 1, "RB", "RB"), (-2.0, centre_y - 9, "WR", "WR")]
    # defense: 4 linemen on the line, 2 LBs at 4 yd inside the box, 1 LB at 5 yd but 8 yd wide (outside the box),
    # 2 CBs at 6 yd over the WRs (outside), 2 deep safeties at 12 / 14 yd
    de = [(0.5, centre_y + d, "DT", "DL") for d in (-3, -1, 1, 3)]
    de += [(4.0, centre_y - 2, "LB", "LB"), (4.0, centre_y + 2, "LB", "LB"), (5.0, centre_y + 8, "OLB", "LB"),
           (6.0, centre_y + 20, "CB", "DB"), (6.0, centre_y - 18, "CB", "DB"), (12.0, centre_y + 5, "SS", "DB"),
           (14.0, centre_y - 5, "FS", "DB")]
    ev = "ball_snap" if with_snap else None
    for depth, y, pos, grp in off:
        rows.append(dict(frame=frame, event=ev, x=los + sign * depth, y=y, teamAbbr="OFF", possessionFlag=1.0,
                         playDirection=direction, absoluteYardlineNumber=los, position=pos, positionGroup=grp))
    for depth, y, pos, grp in de:
        rows.append(dict(frame=frame, event=ev, x=los + sign * depth, y=y, teamAbbr="DEF", possessionFlag=0.0,
                         playDirection=direction, absoluteYardlineNumber=los, position=pos, positionGroup=grp))
    rows.append(dict(frame=frame, event=ev, x=los + sign * ball_dx, y=by, teamAbbr=np.nan, possessionFlag=np.nan,
                     playDirection=direction, absoluteYardlineNumber=los, position=np.nan, positionGroup=np.nan))
    snap = pd.DataFrame(rows)
    # a pre-snap frame with everyone shifted, which must be ignored
    pre = snap.copy()
    pre["frame"] = frame - 4
    pre["event"] = None
    pre["x"] = pre["x"] + sign * 3.0
    out = pd.concat([pre, snap], ignore_index=True)
    if not positions:
        out = out.drop(columns=["position", "positionGroup"])
    return out


def test_highlight_targets_both_directions() -> None:
    for direction in ("right", "left"):
        t = ood.highlight_targets(_play(direction))
        assert t is not None
        assert t["n_deep_safeties"] == 2 and t["mof_open"] == 1
        # box: 4 linemen + 2 inside LBs; the wide LB (|lateral| 8 > 7) and the 6-yd CBs are out
        assert t["box_count_tuned"] == 6
        assert t["n_def"] == 11 and t["n_off"] == 11 and t["play_direction"] == direction
        assert abs(t["los_ball_discrepancy"] - (-0.4)) < 1e-9 and t["snap_frame"] == 7
        assert t["ball_valid"] is True and t["lateral_ref"] == "ball" and t["n_snap_frames"] == 1


def test_highlight_targets_requires_snap_and_players() -> None:
    assert ood.highlight_targets(_play("right", with_snap=False)) is None
    p = _play("right")
    assert ood.highlight_targets(p[p["teamAbbr"] != "DEF"]) is None


def test_highlight_targets_dedups_double_tagged_snap_frame() -> None:
    """A snap frame carrying two event tags lists every row twice; each player must count once."""
    p = _play("right")
    p["nflId"] = np.where(p["teamAbbr"].isna(), np.nan, np.arange(len(p), dtype=float) % 23)
    p["displayName"] = np.where(p["teamAbbr"].isna(), "ball", [f"p{i}" for i in range(len(p))])
    dup = p[p["frame"] == 7].copy()
    dup["event"] = "man_in_motion"
    doubled = pd.concat([p, dup], ignore_index=True)
    assert (doubled["frame"] == 7).sum() == 46
    t = ood.highlight_targets(doubled)
    assert t is not None
    assert t["n_def"] == 11 and t["n_off"] == 11 and t["n_rows_snap_raw"] == 46
    assert t["box_count_tuned"] == 6 and t["n_deep_safeties"] == 2
    # without id columns the fallback de-duplicates on (team, x, y)
    t2 = ood.highlight_targets(doubled.drop(columns=["nflId", "displayName"]))
    assert t2 is not None and t2["n_def"] == 11 and t2["box_count_tuned"] == 6


def test_highlight_targets_requires_exactly_eleven_per_side() -> None:
    p = _play("right")
    ten_def = p.drop(index=p[(p["teamAbbr"] == "DEF") & (p["frame"] == 7)].index[:1])   # one defender missing at the snap
    assert ood.highlight_targets(ten_def) is None
    extra = pd.concat([p, p[(p["teamAbbr"] == "OFF") & (p["frame"] == 7)].head(1).assign(x=lambda d: d["x"] + 0.1)],
                      ignore_index=True)                                # a 12th offensive player
    assert ood.highlight_targets(extra) is None


def test_ball_row_valid() -> None:
    assert ood.ball_row_valid(60.0, 26.0, 60.0, 1.0)
    assert ood.ball_row_valid(58.0, 0.0, 60.0, 1.0) and ood.ball_row_valid(62.4, 53.3, 60.0, 1.0)
    assert not ood.ball_row_valid(70.4, 55.5, 80.0, 1.0)          # off the field (the 2018102200/3269 row)
    assert not ood.ball_row_valid(124.6, 20.0, 99.0, 1.0)         # beyond the end line and 25 yd past the LOS
    assert not ood.ball_row_valid(63.0, 26.0, 60.0, 1.0)          # in the field but 3 yd from the LOS
    assert not ood.ball_row_valid(float("nan"), 26.0, 60.0, 1.0)  # no ball row
    assert ood.ball_row_valid(62.0, 26.0, 60.0, -1.0)             # direction only flips the sign of the offset


def test_choose_snap_frame_prefers_earliest_plausible_ball_row() -> None:
    good_first = pd.concat([_play("right", frame=7), _play("right", frame=30, ball_y=55.5)], ignore_index=True)
    assert ood.choose_snap_frame(good_first) == (7, True)
    bad_first = pd.concat([_play("right", frame=7, ball_y=55.5), _play("right", frame=30)], ignore_index=True)
    assert ood.choose_snap_frame(bad_first) == (30, True)
    t = ood.highlight_targets(bad_first)
    assert t is not None and t["snap_frame"] == 30 and t["n_snap_frames"] == 2 and t["box_count_tuned"] == 6
    both_good = pd.concat([_play("right", frame=7), _play("right", frame=30)], ignore_index=True)
    assert ood.choose_snap_frame(both_good) == (7, True)          # earliest wins, not file order
    reversed_order = pd.concat([_play("right", frame=30), _play("right", frame=7)], ignore_index=True)
    assert ood.choose_snap_frame(reversed_order) == (7, True)
    none_good = pd.concat([_play("right", frame=7, ball_y=55.5), _play("right", frame=30, ball_dx=-9.6)], ignore_index=True)
    assert ood.choose_snap_frame(none_good) == (7, False)
    assert ood.choose_snap_frame(_play("right", with_snap=False)) is None
    assert ood.snap_frames(none_good) == [7, 30]


def test_lateral_fallback_when_ball_row_is_corrupt() -> None:
    """An off-field ball row must not zero the box count: the centre's y (or the line's) is used instead."""
    naive = _play("right", ball_y=55.5, positions=True)
    t = ood.highlight_targets(naive)
    assert t is not None and t["ball_valid"] is False and t["lateral_ref"] == "centre"
    assert t["box_count_tuned"] == 6 and t["n_deep_safeties"] == 2          # identical to the clean play
    assert abs(t["los_ball_discrepancy"] - (-0.4)) < 1e-9                    # the raw ball diagnostic is still reported
    assert abs(t["lateral_ref_shift"] - 29.5) < 1e-9                         # |55.5 - 26|: a glitched row, not a snap in flight
    assert ood.highlight_targets(_play("right"))["lateral_ref_shift"] == 0.0
    # without position columns: the median y of the offensive players within 1.5 yd of the line
    t2 = ood.highlight_targets(_play("right", ball_y=55.5))
    assert t2 is not None and t2["ball_valid"] is False and t2["lateral_ref"] == "line_median"
    s = ood.frame_rows(_play("right", ball_y=55.5), 7)
    off = (s["teamAbbr"] == "OFF").to_numpy()
    depth = (s["x"].to_numpy(dtype=float) - 60.0)
    y_ref, how = ood.lateral_reference(s, off, depth)
    assert how == "line_median" and abs(y_ref - 27.0) < 1e-9                # median of the 8 near-line offensive y's
    # OL median when the centre label is missing
    s2 = ood.frame_rows(_play("right", ball_y=55.5, positions=True), 7)
    s2 = s2.assign(position=s2["position"].replace({"C": "G"}))
    y_ref2, how2 = ood.lateral_reference(s2, (s2["teamAbbr"] == "OFF").to_numpy(), s2["x"].to_numpy(dtype=float) - 60.0)
    assert how2 == "ol_median" and abs(y_ref2 - 26.0) < 1e-9


def test_skill_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    n = 150
    y = rng.integers(3, 10, n).astype(float)
    p = y + rng.normal(scale=1.0, size=n)
    from research.privileged_tracking.common.metrics import r2
    lo, hi = ood.skill_bootstrap_ci("count", y, p, n_boot=500, seed=0)
    assert lo < r2(y, p) < hi and 0 < lo and hi < 1
    yb = (rng.random(n) < 0.3).astype(float)
    pb = np.clip(0.3 + 0.4 * (yb - 0.3) + rng.normal(scale=0.05, size=n), 0.01, 0.99)
    lo_b, hi_b = ood.skill_bootstrap_ci("binary", yb, pb, n_boot=500, seed=0)
    assert lo_b < hi_b and lo_b > 0
    assert np.isnan(ood.skill_bootstrap_ci("count", y[:5], p[:5])[0])


def test_skill_summary_flags_reference_inside_interval() -> None:
    rows = []
    for tgt, kind in (("n_deep_safeties", "count"), ("box_count_tuned", "count"), ("mof_open", "binary")):
        col = "bss" if kind == "binary" else "r2"
        for sample, n in ((ood.HL_SAMPLE, 100), (ood.HL_SAMPLE_VALID, 99)):
            rows.append({"target": tgt, "sample": sample, "model": "F0", "n": n, col: 0.40, "skill_ci_low": 0.25, "skill_ci_high": 0.55,
                         "auc": 0.7})
        rows.append({"target": tgt, "sample": ood.REF_ALL, "model": "F0", "n": 11518, col: 0.43 if tgt != "mof_open" else 0.60, "auc": 0.8})
        rows.append({"target": tgt, "sample": ood.REF_LIKE, "model": "F0", "n": 971, col: 0.56, "auc": 0.85})
    summ = ood.skill_summary(pd.DataFrame(rows), ("F0",)).set_index("target")
    assert bool(summ.loc["box_count_tuned", "ref_all_inside_ci"]) and not bool(summ.loc["box_count_tuned", "ref_like_inside_ci"])
    assert not bool(summ.loc["mof_open", "ref_all_inside_ci"]) and summ.loc["mof_open", "skill_metric"] == "bss"
    assert summ.loc["n_deep_safeties", "n_ball_valid"] == 99 and summ.loc["n_deep_safeties", "n_highlights"] == 100
    text = ood.conclusion_text(summ.reset_index())
    assert "box_count_tuned" in text and "no visible degradation" in text
    assert ood.reference_position(0.434, 0.221, 0.433) == "edge" and ood.reference_position(0.40, 0.25, 0.55) == "inside"
    assert ood.reference_position(0.60, 0.25, 0.55) == "above" and ood.reference_position(0.10, 0.25, 0.55) == "below"
    assert ood.reference_position(float("nan"), 0.25, 0.55) == "unknown"
