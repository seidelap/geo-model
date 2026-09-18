"""Pure geometric features over one StatsBomb freeze frame and an event location.

Every function takes NumPy arrays (or a parsed :class:`FrameArrays`) and returns a
flat ``dict`` of floats so it can be unit-tested on synthetic frames.  Coordinates are
StatsBomb's 120 x 80 yard pitch in the *event team's* attacking frame: the opponent
goal is at ``x = 120`` with centre ``(120, 40)`` and posts ``(120, 36)`` / ``(120, 44)``.

Two freeze-frame flavours are supported by :func:`split_freeze_frame`:

* 360 frames (``three-sixty/<match>.json``): entries carry ``teammate``, ``actor`` and
  ``keeper`` booleans and only players inside ``visible_area`` are present.
* ``shot.freeze_frame`` (present on every Shot event, even without 360): entries carry
  ``teammate``, ``player`` and ``position``; the shooter is not listed and keepers are
  identified by ``position.name == "Goalkeeper"``.

StatsBomb stores the 360 frame of a *paired* event (Dispossessed / Duel, Foul Won /
Foul Committed, Dribbled Past / Dribble, aerial-lost Duels, incomplete Ball Receipt*)
as a copy of the other team's frame: the coordinates are in the OTHER team's attacking
frame and the ``teammate`` flags are sometimes also from the other team's perspective.
:func:`resolve_frame_orientation` detects this per row (flagged actor vs event location,
identical-coordinate "anchor" frames, keeper ends) and either repairs the frame (mirror,
optionally swap flags) or declares it unusable, so that :func:`frame_targets` never
computes ball-relative targets from a mis-oriented frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

PITCH_X = 120.0
PITCH_Y = 80.0
PITCH_AREA = PITCH_X * PITCH_Y
GOAL_X = 120.0
GOAL_Y = 40.0
POST_LOW = 36.0
POST_HIGH = 44.0
BOX_X = 102.0
BOX_Y0 = 18.0
BOX_Y1 = 62.0
FINAL_THIRD_X = 80.0
RELIABLE_MIN_OPPONENTS = 7

_EMPTY_XY = np.zeros((0, 2), dtype=float)


@dataclass(frozen=True)
class FrameArrays:
    """A freeze frame split into typed arrays.

    Attributes:
        tm_xy: teammate locations excluding the actor ``[n_tm, 2]``.
        tm_is_keeper: keeper flag per teammate ``[n_tm]``.
        opp_xy: opponent locations ``[n_opp, 2]``.
        opp_is_keeper: keeper flag per opponent ``[n_opp]``.
        actor_xy: actor location ``[2]`` or ``None`` when the actor is not listed.
    """

    tm_xy: np.ndarray
    tm_is_keeper: np.ndarray
    opp_xy: np.ndarray
    opp_is_keeper: np.ndarray
    actor_xy: np.ndarray | None

    @property
    def opp_outfield_xy(self) -> np.ndarray:
        """Outfield opponent locations ``[n_out, 2]``."""
        return self.opp_xy[~self.opp_is_keeper]

    @property
    def opp_keeper_xy(self) -> np.ndarray | None:
        """First visible opponent keeper ``[2]`` or ``None``."""
        k = self.opp_xy[self.opp_is_keeper]
        return k[0] if len(k) else None


def split_freeze_frame(freeze_frame: list[dict[str, Any]] | None) -> FrameArrays:
    """Split a raw freeze frame (either flavour) into :class:`FrameArrays`.

    Args:
        freeze_frame: list of entries with ``location`` and ``teammate`` plus either
            ``keeper``/``actor`` booleans (360) or ``position`` (shot freeze frame).
    """
    tm: list[list[float]] = []
    tm_k: list[bool] = []
    opp: list[list[float]] = []
    opp_k: list[bool] = []
    actor: np.ndarray | None = None
    for p in freeze_frame or []:
        loc = p.get("location")
        if loc is None or len(loc) < 2:
            continue
        if "keeper" in p:
            keeper = bool(p["keeper"])
        else:
            keeper = (p.get("position") or {}).get("name") == "Goalkeeper"
        if p.get("teammate"):
            if p.get("actor"):
                actor = np.asarray(loc[:2], dtype=float)
                continue
            tm.append(loc[:2])
            tm_k.append(keeper)
        else:
            opp.append(loc[:2])
            opp_k.append(keeper)
    return FrameArrays(
        tm_xy=np.asarray(tm, dtype=float).reshape(-1, 2) if tm else _EMPTY_XY.copy(),
        tm_is_keeper=np.asarray(tm_k, dtype=bool),
        opp_xy=np.asarray(opp, dtype=float).reshape(-1, 2) if opp else _EMPTY_XY.copy(),
        opp_is_keeper=np.asarray(opp_k, dtype=bool),
        actor_xy=actor,
    )


# ----------------------------------------------------------------------------------------
# frame orientation (StatsBomb paired-event defect)
# ----------------------------------------------------------------------------------------
ORIENTATION_OK = "ok"
ORIENTATION_MIRRORED = "mirrored"
ORIENTATION_MIRRORED_SWAPPED = "mirrored_swapped"
ORIENTATION_FAR = "far"
ORIENTATION_UNRESOLVED = "unresolved"
ORIENTATION_NO_ACTOR = "no_actor"
ORIENTATION_NO_LOCATION = "no_location"
ORIENTATION_EMPTY = "empty"
ORIENTATIONS: tuple[str, ...] = (
    ORIENTATION_OK, ORIENTATION_MIRRORED, ORIENTATION_MIRRORED_SWAPPED, ORIENTATION_FAR,
    ORIENTATION_UNRESOLVED, ORIENTATION_NO_ACTOR, ORIENTATION_NO_LOCATION, ORIENTATION_EMPTY,
)


@dataclass(frozen=True)
class FrameOrientationConfig:
    """Tolerances for :func:`resolve_frame_orientation`.

    Attributes:
        tol_ok: the flagged actor must be within this many yards of the event location
            for the frame to be accepted as correctly oriented (in practice it is exact).
        tol_mirror: after mirroring, the flagged actor must be within this many yards of
            the event location for the frame to count as mirrored (paired events are
            coded at locations that mirror each other to ~0.1 yd; the *other* team's
            player who carries the actor flag stands a few yards away).
        keeper_half_x: a keeper with ``x`` beyond this is at the far end of the pitch.
        actor_reassign_tol: after a flag swap the new actor is the teammate nearest to the
            event location if within this distance.
        coord_ndigits: rounding used when hashing frame coordinates for anchor matching.
    """

    tol_ok: float = 1.0
    tol_mirror: float = 5.0
    keeper_half_x: float = 60.0
    actor_reassign_tol: float = 5.0
    coord_ndigits: int = 2


@dataclass(frozen=True)
class FrameResolution:
    """Outcome of :func:`resolve_frame_orientation`.

    Attributes:
        freeze_frame: the frame to compute targets from (repaired when needed); ``None``
            when the frame is unusable.
        orientation: one of :data:`ORIENTATIONS`.
        actor_dist: distance (yd) between the raw flagged actor and the event location
            (``nan`` when either is missing).
        repaired: the frame was mirrored (and possibly flag-swapped).
        usable: geometry targets may be computed from ``freeze_frame``.
        method: how a mirrored frame's flag perspective was decided
            (``"anchor"``, ``"keeper"``, ``""``).
    """

    freeze_frame: list[dict[str, Any]] | None
    orientation: str
    actor_dist: float
    repaired: bool
    usable: bool
    method: str = ""


def mirror_freeze_frame(freeze_frame: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of a frame with every location mapped to ``(120 - x, 80 - y)``."""
    out = []
    for p in freeze_frame:
        q = dict(p)
        loc = p.get("location")
        if loc is not None and len(loc) >= 2:
            q["location"] = [PITCH_X - float(loc[0]), PITCH_Y - float(loc[1])]
        out.append(q)
    return out


def swap_freeze_frame_flags(freeze_frame: list[dict[str, Any]], event_xy: np.ndarray | None,
                            actor_tol: float = 5.0) -> list[dict[str, Any]]:
    """Copy of a frame with ``teammate`` inverted and the actor flag re-assigned.

    The previously flagged actor becomes an ordinary opponent; the new actor is the
    (post-swap) teammate nearest to ``event_xy`` when it lies within ``actor_tol`` yards,
    otherwise no player carries the actor flag.
    """
    out = []
    for p in freeze_frame:
        q = dict(p)
        q["teammate"] = not bool(p.get("teammate"))
        q["actor"] = False
        out.append(q)
    if event_xy is None:
        return out
    best, best_d = -1, float("inf")
    for i, q in enumerate(out):
        loc = q.get("location")
        if not q["teammate"] or loc is None or len(loc) < 2:
            continue
        d = math.hypot(float(loc[0]) - float(event_xy[0]), float(loc[1]) - float(event_xy[1]))
        if d < best_d:
            best, best_d = i, d
    if best >= 0 and best_d <= actor_tol:
        out[best]["actor"] = True
    return out


def frame_coord_key(freeze_frame: list[dict[str, Any]] | None, ndigits: int = 2
                    ) -> tuple[tuple[float, float], ...]:
    """Hashable, order-free key of a frame's locations (for identical-frame matching)."""
    pts = []
    for p in freeze_frame or []:
        loc = p.get("location")
        if loc is not None and len(loc) >= 2:
            pts.append((round(float(loc[0]), ndigits), round(float(loc[1]), ndigits)))
    return tuple(sorted(pts))


def flags_relation(freeze_frame: list[dict[str, Any]], other: list[dict[str, Any]],
                   ndigits: int = 2) -> str:
    """``"same"`` / ``"inverted"`` / ``"neither"``: teammate flags of two frames with the
    same coordinates, compared location by location."""

    def pairs(ff: list[dict[str, Any]], invert: bool) -> list[tuple[float, float, bool]]:
        out = []
        for p in ff:
            loc = p.get("location")
            if loc is not None and len(loc) >= 2:
                out.append((round(float(loc[0]), ndigits), round(float(loc[1]), ndigits),
                            bool(p.get("teammate")) != invert))
        return sorted(out)

    a = pairs(freeze_frame, False)
    if not a:
        return "neither"
    if a == pairs(other, False):
        return "same"
    if a == pairs(other, True):
        return "inverted"
    return "neither"


def actor_event_distances(freeze_frame: list[dict[str, Any]] | None,
                          event_xy: np.ndarray | None) -> tuple[float, float]:
    """``(d_raw, d_mirrored)``: flagged-actor distance to the event location as stored
    and after mirroring the actor; ``nan`` when the actor or the location is missing."""
    nan = float("nan")
    if event_xy is None:
        return nan, nan
    for p in freeze_frame or []:
        loc = p.get("location")
        if p.get("actor") and loc is not None and len(loc) >= 2:
            ex, ey = float(event_xy[0]), float(event_xy[1])
            d0 = math.hypot(float(loc[0]) - ex, float(loc[1]) - ey)
            d1 = math.hypot(PITCH_X - float(loc[0]) - ex, PITCH_Y - float(loc[1]) - ey)
            return d0, d1
    return nan, nan


def keeper_consistency(fa: FrameArrays, half_x: float = 60.0) -> float:
    """``1`` if every visible keeper is at the expected end (opponent keeper ``x >
    half_x``, own keeper ``x < half_x``), ``0`` if any is at the wrong end, ``nan`` if
    no keeper is visible."""
    checks: list[bool] = []
    for k in fa.opp_xy[fa.opp_is_keeper]:
        checks.append(bool(k[0] > half_x))
    for k in fa.tm_xy[fa.tm_is_keeper]:
        checks.append(bool(k[0] < half_x))
    if not checks:
        return float("nan")
    return float(all(checks))


def resolve_frame_orientation(freeze_frame: list[dict[str, Any]] | None,
                              event_xy: np.ndarray | None,
                              anchors: list[tuple[list[dict[str, Any]], bool]] | None = None,
                              cfg: FrameOrientationConfig | None = None) -> FrameResolution:
    """Detect and repair a mis-oriented 360 frame for one event.

    Args:
        freeze_frame: raw 360 ``freeze_frame`` list.
        event_xy: event location ``[2]`` in the event team's attacking frame.
        anchors: ``(freeze_frame, other_team)`` for events of the same match whose frame
            has *identical coordinates* and whose own orientation is ``ok``.  An anchor
            from the other team decides whether the teammate flags are ours (inverted
            relative to the anchor, so a pure mirror repairs the frame) or theirs (same
            as the anchor, so the mirror must be combined with a flag swap).
        cfg: tolerances.

    Returns:
        :class:`FrameResolution`.  Orientation classes: ``ok`` (actor on the event
        location); ``mirrored`` / ``mirrored_swapped`` (repaired); ``far`` (actor away
        from both the event location and its mirror: the frame is from another instant);
        ``unresolved`` (mirrored but the flag perspective could not be decided);
        ``no_actor`` / ``no_location`` / ``empty``.
    """
    cfg = cfg or FrameOrientationConfig()
    nan = float("nan")
    if not freeze_frame:
        return FrameResolution(None, ORIENTATION_EMPTY, nan, False, False)
    if event_xy is None:
        return FrameResolution(None, ORIENTATION_NO_LOCATION, nan, False, False)
    d0, d1 = actor_event_distances(freeze_frame, event_xy)
    if math.isnan(d0):
        return FrameResolution(None, ORIENTATION_NO_ACTOR, nan, False, False)
    if d0 <= cfg.tol_ok:
        return FrameResolution(freeze_frame, ORIENTATION_OK, d0, False, True)
    if d1 > cfg.tol_mirror:
        return FrameResolution(None, ORIENTATION_FAR, d0, False, False)
    mirrored = mirror_freeze_frame(freeze_frame)
    # 1. decisive: an identical-coordinate frame of the other team that is itself ok
    for anchor_ff, other_team in anchors or []:
        if not other_team:
            continue
        rel = flags_relation(freeze_frame, anchor_ff, cfg.coord_ndigits)
        if rel == "inverted":
            return FrameResolution(mirrored, ORIENTATION_MIRRORED, d0, True, True, "anchor")
        if rel == "same":
            swapped = swap_freeze_frame_flags(mirrored, event_xy, cfg.actor_reassign_tol)
            return FrameResolution(swapped, ORIENTATION_MIRRORED_SWAPPED, d0, True, True,
                                   "anchor")
    # 2. fallback: keeper ends after the mirror
    fa = split_freeze_frame(mirrored)
    kc = keeper_consistency(fa, cfg.keeper_half_x)
    if kc == 1.0:
        return FrameResolution(mirrored, ORIENTATION_MIRRORED, d0, True, True, "keeper")
    if kc == 0.0:
        swapped = swap_freeze_frame_flags(mirrored, event_xy, cfg.actor_reassign_tol)
        if keeper_consistency(split_freeze_frame(swapped), cfg.keeper_half_x) == 1.0:
            return FrameResolution(swapped, ORIENTATION_MIRRORED_SWAPPED, d0, True, True,
                                   "keeper")
    return FrameResolution(None, ORIENTATION_UNRESOLVED, d0, False, False)


# ----------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------
def polygon_area(xy: np.ndarray) -> float:
    """Shoelace area of a simple polygon ``[n, 2]`` (closing edge implied)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return 0.0
    x, y = xy[:, 0], xy[:, 1]
    cross = float(x[:-1] @ y[1:] - x[1:] @ y[:-1]) + float(x[-1] * y[0] - x[0] * y[-1])
    return 0.5 * abs(cross)


def clip_polygon_to_rect(xy: np.ndarray, x0: float = 0.0, y0: float = 0.0,
                         x1: float = PITCH_X, y1: float = PITCH_Y) -> np.ndarray:
    """Sutherland-Hodgman clip of polygon ``[n, 2]`` to an axis-aligned rectangle."""
    poly = [tuple(map(float, p)) for p in np.asarray(xy, dtype=float).reshape(-1, 2)]
    if len(poly) >= 2 and poly[0] == poly[-1]:
        poly = poly[:-1]

    def inside(p: tuple[float, float], edge: int) -> bool:
        if edge == 0:
            return p[0] >= x0
        if edge == 1:
            return p[0] <= x1
        if edge == 2:
            return p[1] >= y0
        return p[1] <= y1

    def intersect(a: tuple[float, float], b: tuple[float, float], edge: int
                  ) -> tuple[float, float]:
        if edge in (0, 1):
            xe = x0 if edge == 0 else x1
            t = (xe - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
            return (xe, a[1] + t * (b[1] - a[1]))
        ye = y0 if edge == 2 else y1
        t = (ye - a[1]) / (b[1] - a[1]) if b[1] != a[1] else 0.0
        return (a[0] + t * (b[0] - a[0]), ye)

    for edge in range(4):
        if not poly:
            break
        out: list[tuple[float, float]] = []
        prev = poly[-1]
        for cur in poly:
            if inside(cur, edge):
                if not inside(prev, edge):
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            elif inside(prev, edge):
                out.append(intersect(prev, cur, edge))
            prev = cur
        poly = out
    return np.asarray(poly, dtype=float).reshape(-1, 2)


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """Convex hull (Andrew's monotone chain) of points ``[n, 2]`` -> ``[h, 2]`` CCW."""
    pts = sorted(set(map(tuple, np.asarray(xy, dtype=float).reshape(-1, 2).tolist())))
    if len(pts) < 3:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def convex_hull_area(xy: np.ndarray) -> float:
    """Area of the convex hull of ``[n, 2]``; ``nan`` with fewer than three points."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return float("nan")
    return polygon_area(convex_hull(xy))


def points_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
                       ) -> np.ndarray:
    """Boolean mask ``[n]`` of points ``[n, 2]`` inside (or on) triangle ``a, b, c``."""
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    d1 = (p[:, 0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[:, 1] - b[1])
    d2 = (p[:, 0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[:, 1] - c[1])
    d3 = (p[:, 0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[:, 1] - a[1])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def dist_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance ``[n]`` from points ``[n, 2]`` to the closed segment ``a -> b``."""
    p = np.asarray(p, dtype=float).reshape(-1, 2)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.hypot(p[:, 0] - a[0], p[:, 1] - a[1])
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.hypot(p[:, 0] - proj[:, 0], p[:, 1] - proj[:, 1])


def _in_box(xy: np.ndarray) -> np.ndarray:
    return (xy[:, 0] >= BOX_X) & (xy[:, 1] >= BOX_Y0) & (xy[:, 1] <= BOX_Y1)


# ----------------------------------------------------------------------------------------
# feature groups (numpy in, dict out)
# ----------------------------------------------------------------------------------------
def visibility_features(fa: FrameArrays, visible_area: list[float] | np.ndarray | None
                        ) -> dict[str, float]:
    """Counts of visible players and the visible-area polygon size.

    ``visible_area`` is StatsBomb's flat ``[x0, y0, x1, y1, ...]`` closed polygon; its
    area is clipped to the pitch and expressed as a fraction of 9600 square yards.
    """
    n_opp = int(len(fa.opp_xy))
    out: dict[str, float] = {
        "n_teammates_visible": float(len(fa.tm_xy)),
        "n_opponents_visible": float(n_opp),
        "opp_keeper_visible": float(bool(fa.opp_is_keeper.any())),
        "tm_keeper_visible": float(bool(fa.tm_is_keeper.any())),
        "actor_visible": float(fa.actor_xy is not None),
        "reliable": float(n_opp >= RELIABLE_MIN_OPPONENTS),
    }
    if visible_area is not None and len(visible_area) >= 6:
        poly = np.asarray(visible_area, dtype=float).reshape(-1, 2)
        out["visible_area_frac"] = polygon_area(clip_polygon_to_rect(poly)) / PITCH_AREA
    else:
        out["visible_area_frac"] = float("nan")
    return out


def block_shape_features(fa: FrameArrays) -> dict[str, float]:
    """Shape of the visible outfield opponents (their defensive block).

    ``block_depth`` and ``def_line`` are distances from the opponents' own goal line
    (``120 - x``); ``block_area`` is the convex-hull area (``nan`` below three players).
    """
    xy = fa.opp_outfield_xy
    n = len(xy)
    nan = float("nan")
    if n == 0:
        return {"n_opp_outfield": 0.0, "block_depth": nan, "def_line": nan,
                "block_length": nan, "block_width": nan, "block_area": nan,
                "block_centroid_y": nan, "block_std_x": nan, "block_std_y": nan}
    xs, ys = xy[:, 0], xy[:, 1]
    return {
        "n_opp_outfield": float(n),
        "block_depth": float(GOAL_X - xs.mean()),
        "def_line": float(GOAL_X - xs.max()),
        "block_length": float(xs.max() - xs.min()) if n >= 2 else nan,
        "block_width": float(ys.max() - ys.min()) if n >= 2 else nan,
        "block_area": convex_hull_area(xy),
        "block_centroid_y": float(ys.mean()),
        "block_std_x": float(xs.std()) if n >= 2 else nan,
        "block_std_y": float(ys.std()) if n >= 2 else nan,
    }


def ball_relative_features(fa: FrameArrays, ball_xy: np.ndarray | None) -> dict[str, float]:
    """Opponent / teammate counts relative to the ball (event) location.

    Counts of opponents include the keeper; ``nearest_opp_dist`` too.  ``ahead`` means
    ``x > ball_x`` (between the ball and the goal being attacked).
    """
    nan = float("nan")
    out: dict[str, float] = {
        "n_opp_in_box": float(_in_box(fa.opp_outfield_xy).sum()) if len(fa.opp_xy) else 0.0,
        "n_tm_in_box": float(_in_box(fa.tm_xy).sum()) if len(fa.tm_xy) else 0.0,
    }
    k = fa.opp_keeper_xy
    out["opp_keeper_dist_to_goal_line"] = float(GOAL_X - k[0]) if k is not None else nan
    out["opp_keeper_y"] = float(k[1]) if k is not None else nan
    if ball_xy is None:
        out.update({"n_opp_ahead_of_ball": nan, "n_tm_ahead_of_ball": nan,
                    "n_opp_within_5": nan, "n_opp_within_10": nan, "nearest_opp_dist": nan,
                    "n_tm_within_10": nan, "nearest_tm_dist": nan})
        return out
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    if len(fa.opp_xy):
        d = np.hypot(fa.opp_xy[:, 0] - bx, fa.opp_xy[:, 1] - by)
        out["n_opp_ahead_of_ball"] = float((fa.opp_xy[:, 0] > bx).sum())
        out["n_opp_within_5"] = float((d <= 5.0).sum())
        out["n_opp_within_10"] = float((d <= 10.0).sum())
        out["nearest_opp_dist"] = float(d.min())
    else:
        out.update({"n_opp_ahead_of_ball": 0.0, "n_opp_within_5": 0.0, "n_opp_within_10": 0.0,
                    "nearest_opp_dist": nan})
    if len(fa.tm_xy):
        d = np.hypot(fa.tm_xy[:, 0] - bx, fa.tm_xy[:, 1] - by)
        out["n_tm_ahead_of_ball"] = float((fa.tm_xy[:, 0] > bx).sum())
        out["n_tm_within_10"] = float((d <= 10.0).sum())
        out["nearest_tm_dist"] = float(d.min())
    else:
        out.update({"n_tm_ahead_of_ball": 0.0, "n_tm_within_10": 0.0, "nearest_tm_dist": nan})
    return out


def shot_cone_features(opp_xy: np.ndarray, opp_is_keeper: np.ndarray,
                       ball_xy: np.ndarray | None) -> dict[str, float]:
    """Opponents inside the triangle ball -> both posts, as if a shot were taken.

    ``n_opp_in_cone`` counts outfield opponents; the keeper is reported separately.
    """
    nan = float("nan")
    if ball_xy is None:
        return {"n_opp_in_cone": nan, "nearest_opp_dist_in_cone": nan,
                "opp_keeper_in_cone": nan, "cone_area": nan}
    a = np.asarray(ball_xy, dtype=float)
    b = np.array([GOAL_X, POST_LOW])
    c = np.array([GOAL_X, POST_HIGH])
    area = polygon_area(np.vstack([a, b, c]))
    out = {"cone_area": float(area), "n_opp_in_cone": 0.0,
           "nearest_opp_dist_in_cone": nan, "opp_keeper_in_cone": 0.0}
    opp_xy = np.asarray(opp_xy, dtype=float).reshape(-1, 2)
    opp_is_keeper = np.asarray(opp_is_keeper, dtype=bool).reshape(-1)
    if len(opp_xy) == 0 or area <= 1e-9:
        return out
    inside = points_in_triangle(opp_xy, a, b, c)
    out_in = inside & ~opp_is_keeper
    out["n_opp_in_cone"] = float(out_in.sum())
    out["opp_keeper_in_cone"] = float((inside & opp_is_keeper).any())
    if out_in.any():
        d = np.hypot(opp_xy[out_in, 0] - a[0], opp_xy[out_in, 1] - a[1])
        out["nearest_opp_dist_in_cone"] = float(d.min())
    return out


def pass_geometry_features(fa: FrameArrays, ball_xy: np.ndarray | None,
                           end_xy: np.ndarray | None, lane_width: float = 2.0,
                           receiver_radius: float = 3.0) -> dict[str, float]:
    """Opponent pressure on a pass lane and on its receiver.

    The receiver is the non-actor teammate nearest to ``end_xy``; when no teammate is
    visible the end location itself is used.  Opponent counts include the keeper.
    """
    nan = float("nan")
    out = {"n_opp_within_3_of_end": nan, "n_opp_in_lane": nan, "nearest_opp_to_end": nan,
           "nearest_opp_to_receiver": nan, "receiver_dist_to_end": nan}
    if ball_xy is None or end_xy is None:
        return out
    a = np.asarray(ball_xy, dtype=float)
    e = np.asarray(end_xy, dtype=float)
    if len(fa.tm_xy):
        dt = np.hypot(fa.tm_xy[:, 0] - e[0], fa.tm_xy[:, 1] - e[1])
        r = fa.tm_xy[int(dt.argmin())]
        out["receiver_dist_to_end"] = float(dt.min())
    else:
        r = e
    if len(fa.opp_xy) == 0:
        out.update({"n_opp_within_3_of_end": 0.0, "n_opp_in_lane": 0.0})
        return out
    d_end = np.hypot(fa.opp_xy[:, 0] - e[0], fa.opp_xy[:, 1] - e[1])
    d_rec = np.hypot(fa.opp_xy[:, 0] - r[0], fa.opp_xy[:, 1] - r[1])
    d_lane = dist_to_segment(fa.opp_xy, a, e)
    out["n_opp_within_3_of_end"] = float((d_end <= receiver_radius).sum())
    out["nearest_opp_to_end"] = float(d_end.min())
    out["nearest_opp_to_receiver"] = float(d_rec.min())
    out["n_opp_in_lane"] = float((d_lane <= lane_width).sum())
    return out


def _geometry_targets(fa: FrameArrays, visible_area: list[float] | None,
                      ball_xy: np.ndarray | None, pass_end_xy: np.ndarray | None
                      ) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(visibility_features(fa, visible_area))
    out.update(block_shape_features(fa))
    out.update(ball_relative_features(fa, ball_xy))
    out.update(shot_cone_features(fa.opp_xy, fa.opp_is_keeper, ball_xy))
    out.update(pass_geometry_features(fa, ball_xy, pass_end_xy))
    return out


GEOMETRY_TARGET_KEYS: tuple[str, ...] = tuple(
    _geometry_targets(split_freeze_frame([]), None, np.zeros(2), np.zeros(2)).keys())
ORIENTATION_TARGET_KEYS: tuple[str, ...] = (
    "frame_orientation", "frame_ok", "frame_repaired", "frame_method", "actor_dist_to_event",
    "keeper_consistent",
)


def frame_targets(freeze_frame: list[dict[str, Any]] | None,
                  visible_area: list[float] | None, ball_xy: np.ndarray | None,
                  pass_end_xy: np.ndarray | None = None, prefix: str = "y_",
                  resolution: FrameResolution | None = None) -> dict[str, Any]:
    """All 360-frame targets for one event, keys prefixed with ``prefix``.

    Args:
        freeze_frame: raw 360 ``freeze_frame`` list.
        visible_area: raw flat polygon or ``None``.
        ball_xy: event location ``[2]`` in the event team's frame or ``None``.
        pass_end_xy: pass end location ``[2]`` for Pass events, else ``None`` (pass
            geometry keys are then ``nan``).
        resolution: result of :func:`resolve_frame_orientation` (computed here without
            anchor frames when omitted).  Geometry targets are computed from the
            resolved (possibly repaired) frame; when the frame is unusable every
            geometry target is ``nan`` except ``visible_area_frac``, ``actor_visible``
            and ``reliable`` (``0``), and the ``frame_*`` diagnostics say why.

    Returns:
        ``frame_orientation`` (str), ``frame_ok`` / ``frame_repaired`` (0/1),
        ``frame_method`` (``anchor`` / ``keeper`` / ``""``), ``actor_dist_to_event``
        (raw yards), ``keeper_consistent`` (1/0/nan) followed by the geometry targets.
    """
    if resolution is None:
        resolution = resolve_frame_orientation(freeze_frame, ball_xy)
    nan = float("nan")
    out: dict[str, Any] = {
        "frame_orientation": resolution.orientation,
        "frame_ok": float(resolution.usable),
        "frame_repaired": float(resolution.repaired),
        "frame_method": resolution.method,
        "actor_dist_to_event": resolution.actor_dist,
        "keeper_consistent": nan,
    }
    if resolution.usable:
        fa = split_freeze_frame(resolution.freeze_frame)
        geo = _geometry_targets(fa, visible_area, ball_xy, pass_end_xy)
        out["keeper_consistent"] = keeper_consistency(fa)
    else:
        geo = dict.fromkeys(GEOMETRY_TARGET_KEYS, nan)
        raw = visibility_features(split_freeze_frame(freeze_frame), visible_area)
        geo["visible_area_frac"] = raw["visible_area_frac"]
        geo["actor_visible"] = raw["actor_visible"]
        geo["reliable"] = 0.0
    out.update(geo)
    return {prefix + k: v for k, v in out.items()}


def shot_frame_features(shot_freeze_frame: list[dict[str, Any]] | None,
                        ball_xy: np.ndarray | None, prefix: str = "sff_") -> dict[str, float]:
    """Cone and ball-relative geometry from a Shot's own ``shot.freeze_frame``.

    Available for (almost) every Shot in every match, so it serves as the comparison
    source for the 360-derived cone counts.  Returns ``nan``s when the frame is absent.
    """
    nan = float("nan")
    if not shot_freeze_frame:
        keys = ["present", "n_opp", "n_tm", "opp_keeper_visible", "n_opp_in_cone",
                "nearest_opp_dist_in_cone", "opp_keeper_in_cone", "cone_area",
                "n_opp_within_5", "n_opp_within_10", "nearest_opp_dist", "n_opp_in_box",
                "n_opp_ahead_of_ball", "opp_keeper_dist_to_goal_line", "opp_keeper_y"]
        d = {k: nan for k in keys}
        d["present"] = 0.0
        return {prefix + k: v for k, v in d.items()}
    fa = split_freeze_frame(shot_freeze_frame)
    out: dict[str, float] = {"present": 1.0, "n_opp": float(len(fa.opp_xy)),
                             "n_tm": float(len(fa.tm_xy)),
                             "opp_keeper_visible": float(bool(fa.opp_is_keeper.any()))}
    out.update(shot_cone_features(fa.opp_xy, fa.opp_is_keeper, ball_xy))
    br = ball_relative_features(fa, ball_xy)
    for k in ("n_opp_within_5", "n_opp_within_10", "nearest_opp_dist", "n_opp_in_box",
              "n_opp_ahead_of_ball", "opp_keeper_dist_to_goal_line", "opp_keeper_y"):
        out[k] = br[k]
    return {prefix + k: v for k, v in out.items()}
