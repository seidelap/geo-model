"""Tests for the pure highlight-frame target function of NFL 03b (no data access)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import ood_highlights as ood


def _play(direction: str, los: float = 60.0, with_snap: bool = True) -> pd.DataFrame:
    """11 v 11 plus the ball at one snap frame; offense attacks +x when ``direction == 'right'``."""
    sign = 1.0 if direction == "right" else -1.0
    ball_y = 26.0
    rows = []
    # offense: 5 OL 1 yd behind the line, QB 5 yd back, 2 WR wide, TE, RB
    off = [(-1.0, ball_y + d) for d in (-4, -2, 0, 2, 4)] + [(-5.0, ball_y), (-0.5, ball_y + 20), (-0.5, ball_y - 18),
                                                            (-1.0, ball_y + 6), (-6.0, ball_y - 1), (-2.0, ball_y - 9)]
    # defense: 4 linemen on the line, 2 LBs at 4 yd inside the box, 1 LB at 5 yd but 8 yd wide (outside the box),
    # 2 CBs at 6 yd over the WRs (outside), 2 deep safeties at 12 / 14 yd
    de = [(0.5, ball_y + d) for d in (-3, -1, 1, 3)] + [(4.0, ball_y - 2), (4.0, ball_y + 2), (5.0, ball_y + 8),
                                                        (6.0, ball_y + 20), (6.0, ball_y - 18), (12.0, ball_y + 5), (14.0, ball_y - 5)]
    for depth, y in off:
        rows.append(dict(frame=7, event="ball_snap" if with_snap else None, x=los + sign * depth, y=y, teamAbbr="OFF",
                         possessionFlag=1.0, playDirection=direction, absoluteYardlineNumber=los))
    for depth, y in de:
        rows.append(dict(frame=7, event="ball_snap" if with_snap else None, x=los + sign * depth, y=y, teamAbbr="DEF",
                         possessionFlag=0.0, playDirection=direction, absoluteYardlineNumber=los))
    rows.append(dict(frame=7, event="ball_snap" if with_snap else None, x=los - sign * 0.4, y=ball_y, teamAbbr=np.nan,
                     possessionFlag=np.nan, playDirection=direction, absoluteYardlineNumber=los))
    # a pre-snap frame with everyone shifted, which must be ignored
    pre = pd.DataFrame(rows).copy()
    pre["frame"] = 3
    pre["event"] = None
    pre["x"] = pre["x"] + sign * 3.0
    return pd.concat([pre, pd.DataFrame(rows)], ignore_index=True)


def test_highlight_targets_both_directions() -> None:
    for direction in ("right", "left"):
        t = ood.highlight_targets(_play(direction))
        assert t is not None
        assert t["n_deep_safeties"] == 2 and t["mof_open"] == 1
        # box: 4 linemen + 2 inside LBs; the wide LB (|lateral| 8 > 7) and the 6-yd CBs are out
        assert t["box_count_tuned"] == 6
        assert t["n_def"] == 11 and t["n_off"] == 11 and t["play_direction"] == direction
        assert abs(t["los_ball_discrepancy"] - (-0.4)) < 1e-9 and t["snap_frame"] == 7


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
