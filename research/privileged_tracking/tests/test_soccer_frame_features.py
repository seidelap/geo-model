"""Tests for the pure freeze-frame feature functions on a synthetic frame."""
from __future__ import annotations

import math

import numpy as np

from research.privileged_tracking.soccer import frame_features as ff


def _p(x: float, y: float, teammate: bool, keeper: bool = False, actor: bool = False) -> dict:
    return {"teammate": teammate, "keeper": keeper, "actor": actor, "location": [x, y]}


def synthetic_frame() -> list[dict]:
    """Actor at (100, 40); 2 teammates; 10 outfield opponents + keeper at (118, 40)."""
    return [
        _p(100, 40, True, actor=True),
        _p(95, 30, True), _p(105, 50, True),
        _p(118, 40, False, keeper=True),
        _p(110, 38, False), _p(110, 42, False), _p(108, 30, False), _p(112, 50, False),
        _p(100, 20, False), _p(100, 60, False), _p(90, 40, False), _p(85, 35, False),
        _p(85, 45, False), _p(80, 40, False),
    ]


BALL = np.array([100.0, 40.0])


def test_split_freeze_frame_360_flavour() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    assert fa.tm_xy.shape == (2, 2)
    assert fa.opp_xy.shape == (11, 2)
    assert fa.opp_is_keeper.sum() == 1
    assert fa.actor_xy is not None and fa.actor_xy.tolist() == [100.0, 40.0]
    assert fa.opp_outfield_xy.shape == (10, 2)
    assert fa.opp_keeper_xy is not None and fa.opp_keeper_xy.tolist() == [118.0, 40.0]


def test_split_freeze_frame_shot_flavour_uses_position_for_keeper() -> None:
    frame = [
        {"teammate": False, "location": [118.0, 40.0], "position": {"name": "Goalkeeper"}},
        {"teammate": False, "location": [110.0, 40.0], "position": {"name": "Center Back"}},
        {"teammate": True, "location": [104.0, 44.0], "position": {"name": "Left Wing"}},
    ]
    fa = ff.split_freeze_frame(frame)
    assert fa.opp_is_keeper.tolist() == [True, False]
    assert fa.actor_xy is None
    assert len(fa.tm_xy) == 1


def test_polygon_and_hull_helpers() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert math.isclose(ff.polygon_area(square), 1.0)
    tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
    assert math.isclose(ff.polygon_area(tri), 6.0)
    pts = np.vstack([square, [[0.5, 0.5], [0.2, 0.7]]])
    assert math.isclose(ff.convex_hull_area(pts), 1.0)
    assert math.isnan(ff.convex_hull_area(square[:2]))
    # clipping a polygon that spills off the pitch
    poly = np.array([[-10, 0], [60, 0], [60, 80], [-10, 80], [-10, 0]], dtype=float)
    clipped = ff.clip_polygon_to_rect(poly)
    assert math.isclose(ff.polygon_area(clipped), 60 * 80)


def test_triangle_and_segment_helpers() -> None:
    a, b, c = np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([0.0, 4.0])
    pts = np.array([[1, 1], [3, 3], [2, 0], [-1, 1]], dtype=float)
    assert ff.points_in_triangle(pts, a, b, c).tolist() == [True, False, True, False]
    d = ff.dist_to_segment(np.array([[5.0, 0.0], [2.0, 1.0], [-3.0, 4.0]]), a, b)
    assert np.allclose(d, [1.0, 1.0, 5.0])
    d0 = ff.dist_to_segment(np.array([[3.0, 4.0]]), a, a)
    assert np.allclose(d0, [5.0])


def test_visibility_features() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    v = ff.visibility_features(fa, [0, 0, 120, 0, 120, 80, 0, 80, 0, 0])
    assert v["n_teammates_visible"] == 2  # actor excluded
    assert v["n_opponents_visible"] == 11
    assert v["opp_keeper_visible"] == 1 and v["tm_keeper_visible"] == 0
    assert v["actor_visible"] == 1
    assert v["reliable"] == 1
    assert math.isclose(v["visible_area_frac"], 1.0)
    half = ff.visibility_features(fa, [-10, 0, 60, 0, 60, 80, -10, 80, -10, 0])
    assert math.isclose(half["visible_area_frac"], 0.5)
    assert math.isnan(ff.visibility_features(fa, None)["visible_area_frac"])
    few = ff.split_freeze_frame(synthetic_frame()[:8])
    assert ff.visibility_features(few, None)["reliable"] == 0


def test_block_shape_features() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    b = ff.block_shape_features(fa)
    assert b["n_opp_outfield"] == 10
    assert math.isclose(b["block_depth"], 120 - 98.0)
    assert math.isclose(b["def_line"], 8.0)
    assert math.isclose(b["block_length"], 32.0)
    assert math.isclose(b["block_width"], 40.0)
    assert math.isclose(b["block_centroid_y"], 40.0)
    assert b["block_area"] > 0
    empty = ff.block_shape_features(ff.split_freeze_frame([]))
    assert empty["n_opp_outfield"] == 0 and math.isnan(empty["block_depth"])


def test_ball_relative_features() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    r = ff.ball_relative_features(fa, BALL)
    assert r["n_opp_ahead_of_ball"] == 5  # keeper + 4 outfield with x > 100
    assert r["n_opp_within_5"] == 0
    assert r["n_opp_within_10"] == 1  # (90, 40) exactly 10 yards
    assert math.isclose(r["nearest_opp_dist"], 10.0)
    assert r["n_tm_ahead_of_ball"] == 1
    assert r["n_opp_in_box"] == 4  # outfield only
    assert r["n_tm_in_box"] == 1
    assert math.isclose(r["opp_keeper_dist_to_goal_line"], 2.0)
    assert math.isclose(r["opp_keeper_y"], 40.0)
    nb = ff.ball_relative_features(fa, None)
    assert math.isnan(nb["nearest_opp_dist"]) and nb["n_opp_in_box"] == 4


def test_shot_cone_features() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    c = ff.shot_cone_features(fa.opp_xy, fa.opp_is_keeper, BALL)
    assert c["n_opp_in_cone"] == 2  # (110, 38) and (110, 42) on the cone edges
    assert c["opp_keeper_in_cone"] == 1
    assert math.isclose(c["nearest_opp_dist_in_cone"], math.hypot(10, 2))
    assert math.isclose(c["cone_area"], 0.5 * 8 * 20)
    on_line = ff.shot_cone_features(fa.opp_xy, fa.opp_is_keeper, np.array([120.0, 40.0]))
    assert on_line["n_opp_in_cone"] == 0 and math.isnan(on_line["nearest_opp_dist_in_cone"])
    none = ff.shot_cone_features(fa.opp_xy, fa.opp_is_keeper, None)
    assert math.isnan(none["n_opp_in_cone"])


def test_pass_geometry_features() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    p = ff.pass_geometry_features(fa, BALL, np.array([110.0, 40.0]))
    assert p["n_opp_within_3_of_end"] == 2
    assert p["n_opp_in_lane"] == 2
    assert math.isclose(p["nearest_opp_to_end"], 2.0)
    # receiver = teammate nearest to the end location = (105, 50); nearest opp (112, 50)
    assert math.isclose(p["receiver_dist_to_end"], math.hypot(5, 10))
    assert math.isclose(p["nearest_opp_to_receiver"], 7.0)
    assert math.isnan(ff.pass_geometry_features(fa, BALL, None)["n_opp_in_lane"])


def test_frame_targets_and_shot_frame_features_prefixes() -> None:
    t = ff.frame_targets(synthetic_frame(), [0, 0, 120, 0, 120, 80, 0, 80, 0, 0], BALL,
                         pass_end_xy=np.array([110.0, 40.0]))
    assert all(k.startswith("y_") for k in t)
    assert t["y_n_opp_in_cone"] == 2 and t["y_n_opp_in_lane"] == 2
    assert t["y_reliable"] == 1
    sff = [{"teammate": False, "location": [118.0, 40.0], "position": {"name": "Goalkeeper"}},
           {"teammate": False, "location": [110.0, 39.0], "position": {"name": "Center Back"}}]
    s = ff.shot_frame_features(sff, BALL)
    assert s["sff_present"] == 1 and s["sff_n_opp"] == 2
    assert s["sff_n_opp_in_cone"] == 1 and s["sff_opp_keeper_in_cone"] == 1
    assert math.isclose(s["sff_opp_keeper_dist_to_goal_line"], 2.0)
    missing = ff.shot_frame_features(None, BALL)
    assert missing["sff_present"] == 0 and math.isnan(missing["sff_n_opp_in_cone"])


# ----------------------------------------------------------------------------------------
# frame orientation (StatsBomb paired-event defect)
# ----------------------------------------------------------------------------------------
def _mirror(frame: list[dict]) -> list[dict]:
    return ff.mirror_freeze_frame(frame)


def _copy_with_other_perspective(frame: list[dict], new_actor_xy: tuple[float, float]
                                 ) -> list[dict]:
    """The other team's view of the same instant: flags inverted, their player at
    ``new_actor_xy`` carries the actor flag."""
    out = []
    for p in frame:
        q = dict(p)
        q["teammate"] = not p["teammate"]
        q["actor"] = False
        if not p["teammate"] and tuple(p["location"]) == tuple(new_actor_xy):
            q["actor"] = True
        out.append(q)
    return out


def test_actor_event_distances_and_coord_key() -> None:
    frame = synthetic_frame()
    d0, d1 = ff.actor_event_distances(frame, BALL)
    assert d0 == 0.0 and math.isclose(d1, math.hypot(120 - 100 - 100, 80 - 40 - 40))
    assert all(math.isnan(v) for v in ff.actor_event_distances(frame, None))
    assert all(math.isnan(v) for v in ff.actor_event_distances(frame[1:], BALL))
    key = ff.frame_coord_key(frame)
    assert key == ff.frame_coord_key(list(reversed(frame)))
    assert key != ff.frame_coord_key(_mirror(frame))
    assert ff.flags_relation(frame, frame) == "same"
    assert ff.flags_relation(frame, _copy_with_other_perspective(frame, (110, 38))) == "inverted"
    assert ff.flags_relation(frame, _mirror(frame)) == "neither"


def test_resolve_ok_frame_is_untouched() -> None:
    res = ff.resolve_frame_orientation(synthetic_frame(), BALL)
    assert res.orientation == "ok" and res.usable and not res.repaired
    assert res.actor_dist == 0.0 and res.method == ""
    assert res.freeze_frame is synthetic_frame() or res.freeze_frame == synthetic_frame()


def test_resolve_pure_mirror_with_inverted_anchor() -> None:
    # Team A's Dispossessed at BALL; its stored frame is B's Duel frame (B's orientation)
    # with the flags re-labelled for A, so it is the mirror of A's true frame.
    true_frame = synthetic_frame()
    stored = _mirror(true_frame)
    # anchor: B's Duel frame, ok for B (B's tackler at the mirrored ball, B's flags)
    b_event_xy = (120 - 110, 80 - 38)  # B's tackler stands at (110, 38) in A's frame
    anchor = _copy_with_other_perspective(stored, b_event_xy)
    res = ff.resolve_frame_orientation(stored, BALL, [(anchor, True)])
    assert res.orientation == "mirrored" and res.usable and res.repaired
    assert res.method == "anchor"
    fa = ff.split_freeze_frame(res.freeze_frame)
    assert fa.actor_xy is not None and fa.actor_xy.tolist() == [100.0, 40.0]
    assert fa.opp_keeper_xy is not None and fa.opp_keeper_xy.tolist() == [118.0, 40.0]
    assert len(fa.opp_xy) == 11 and len(fa.tm_xy) == 2


def test_resolve_full_copy_with_same_anchor_swaps_flags() -> None:
    # Team A's aerial-lost Duel at BALL; the stored frame is B's Pass frame verbatim:
    # B's orientation AND B's flags, actor flag on B's player at (110, 38) in A's frame.
    true_frame = synthetic_frame() + [_p(101, 41, False)]  # B's header winner at the ball
    stored = _copy_with_other_perspective(_mirror(true_frame), (120 - 101, 80 - 41))
    assert ff.actor_event_distances(stored, BALL)[1] < 5
    res = ff.resolve_frame_orientation(stored, BALL, [(stored, True)])
    assert res.orientation == "mirrored_swapped" and res.usable and res.repaired
    fa = ff.split_freeze_frame(res.freeze_frame)
    # A's actor re-identified (nearest post-swap teammate to the ball, exactly on it)
    assert fa.actor_xy is not None and fa.actor_xy.tolist() == [100.0, 40.0]
    assert fa.opp_keeper_xy is not None and fa.opp_keeper_xy.tolist() == [118.0, 40.0]
    assert len(fa.opp_xy) == 12 and len(fa.tm_xy) == 2
    t = ff.frame_targets(stored, None, BALL, resolution=res)
    assert t["y_frame_ok"] == 1 and t["y_frame_repaired"] == 1
    assert t["y_keeper_consistent"] == 1 and t["y_actor_visible"] == 1
    assert t["y_n_opp_in_cone"] == 2 and t["y_n_opp_ahead_of_ball"] == 6
    assert math.isclose(t["y_nearest_opp_dist"], math.hypot(1, 1))


def test_resolve_keeper_fallback_without_anchor() -> None:
    true_frame = synthetic_frame()
    # pure mirror, no anchor: opp keeper lands at x = 118 after the mirror -> mirrored
    res = ff.resolve_frame_orientation(_mirror(true_frame), BALL, [])
    assert res.orientation == "mirrored" and res.method == "keeper"
    # full copy, no anchor: after the mirror the "opp" keeper is at x = 2 -> swap
    with_tackler = true_frame + [_p(101, 41, False)]
    stored = _copy_with_other_perspective(_mirror(with_tackler), (120 - 101, 80 - 41))
    res = ff.resolve_frame_orientation(stored, BALL, None)
    assert res.orientation == "mirrored_swapped" and res.method == "keeper"
    # same-team anchors are ignored; no keeper visible -> unresolved and unusable
    no_keeper = [p for p in _mirror(true_frame) if not p["keeper"]]
    res = ff.resolve_frame_orientation(no_keeper, BALL, [(no_keeper, False)])
    assert res.orientation == "unresolved" and not res.usable and res.freeze_frame is None


def test_resolve_far_no_actor_and_empty() -> None:
    frame = synthetic_frame()
    far = ff.resolve_frame_orientation(frame, np.array([30.0, 10.0]))
    assert far.orientation == "far" and not far.usable and far.actor_dist > 5
    assert ff.resolve_frame_orientation(frame[1:], BALL).orientation == "no_actor"
    assert ff.resolve_frame_orientation(frame, None).orientation == "no_location"
    assert ff.resolve_frame_orientation([], BALL).orientation == "empty"
    t = ff.frame_targets(frame, [0, 0, 120, 0, 120, 80, 0, 80, 0, 0], np.array([30.0, 10.0]))
    assert t["y_frame_orientation"] == "far" and t["y_frame_ok"] == 0
    assert t["y_reliable"] == 0 and math.isnan(t["y_nearest_opp_dist"])
    assert math.isnan(t["y_block_depth"]) and math.isnan(t["y_n_opp_in_cone"])
    assert math.isclose(t["y_visible_area_frac"], 1.0) and t["y_actor_visible"] == 1
    assert set(t) == set(ff.frame_targets(frame, None, BALL))


def test_keeper_consistency_and_swap_helper() -> None:
    fa = ff.split_freeze_frame(synthetic_frame())
    assert ff.keeper_consistency(fa) == 1.0
    assert ff.keeper_consistency(ff.split_freeze_frame(_mirror(synthetic_frame()))) == 0.0
    assert math.isnan(ff.keeper_consistency(ff.split_freeze_frame(synthetic_frame()[:3])))
    # (95, 30) was a teammate -> now an opponent; the nearest post-swap teammate within
    # 5 yd of (95, 30) does not exist ((100, 20) is 11 yd away) -> no actor flag at all
    swapped = ff.swap_freeze_frame_flags(synthetic_frame(), np.array([95.0, 30.0]))
    assert not any(p["actor"] for p in swapped)
    assert sum(p["teammate"] for p in swapped) == 11 and sum(p["keeper"] for p in swapped) == 1
    swapped = ff.swap_freeze_frame_flags(synthetic_frame(), np.array([100.0, 20.0]))
    assert [p["location"] for p in swapped if p["actor"]] == [[100, 20]]
    assert not any(p["actor"] for p in
                   ff.swap_freeze_frame_flags(synthetic_frame(), np.array([60.0, 40.0])))
