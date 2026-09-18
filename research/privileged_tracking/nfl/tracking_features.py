"""Pure per-play feature functions over NFL player-tracking frames (2017 Big Data Bowl).

Everything here operates on one play's frames as NumPy arrays and returns plain dicts,
so it can be unit-tested on synthetic plays. The driver (``build_tables.py``) handles
IO, joins and reporting.

Conventions (after :meth:`PlayFrames.from_long`):

* Coordinates are normalised so the offense attacks toward ``+x`` (field is 0-120 by
  0-53.3 yards). When a play is flipped, ``x' = 120 - x``, ``y' = 53.3 - y`` and the
  motion angle ``dir' = (dir + 180) mod 360``. ``dir`` in the raw data is measured
  clockwise from the +y axis (``dx = sin(dir)``, ``dy = cos(dir)``); verified
  empirically against frame-to-frame displacement.
* ``depth`` of a player is ``x - los_x``: positive for defenders lined up beyond the
  line of scrimmage, negative for offensive players behind it. ``depth_behind`` used
  for the offense is ``los_x - x``.
* ``lateral`` is ``y - ball_y_ref`` where ``ball_y_ref`` is the ball's y at the snap
  (falling back to an estimate of the centre's y when the ball track is implausible).
  **Sign convention: positive ``lateral`` is the offense's LEFT.** With ``+x`` the attack
  direction and ``y`` measured across the field, a player at larger ``y`` stands to the
  left of a quarterback facing ``+x``; nflfastR ``pass_location == "left"`` corresponds
  to ``target_lateral > 0`` on 98.5% of tracked passes. Every ``*_left`` / ``*_right``
  column (``n_wide_left``, ``cushion_left``, ``target_side_derived`` ...) follows the
  offense's perspective, i.e. ``left`` means ``lateral > 0`` (``LEFT_SIGN``).
* Time is measured in frames at ``fps`` (10 Hz). The raw ``time`` column only has
  one-second resolution and is not used.
* Snap frame = first frame tagged ``ball_snap``. Plays without it are not handled here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
BALL_LABEL = "ball"
LEFT_SIGN = 1.0   # sign of ``lateral`` on the offense's left (see module docstring)

OFF_POS_RB = {"RB", "FB", "HB"}
OFF_POS_TE = {"TE"}
OFF_POS_WR = {"WR"}
OFF_POS_OL = {"T", "G", "C", "OL", "OT", "OG"}
OFF_POS_QB = {"QB"}
DEF_POS_DL = {"DE", "DT", "NT", "DL"}
DEF_POS_LB = {"OLB", "ILB", "MLB", "LB"}
DEF_POS_DB = {"CB", "FS", "SS", "DB", "S"}

ARRIVAL_EVENTS = (
    "pass_arrived",
    "pass_outcome_caught",
    "pass_outcome_touchdown",
    "pass_outcome_interception",
    "pass_outcome_incomplete",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Thresholds (yards / seconds) for the tracking-derived targets.

    Attributes:
        deep_depth: defender depth beyond the LOS to count as a deep safety.
        box_depth: max defender depth for the naive box count.
        box_lateral: max |lateral| for the naive box count.
        dl_depth: max |depth| for a defender to count as on the line (``n_dl``).
        wide_lateral: |lateral| beyond which an offensive player is "wide".
        backfield_depth: min yards behind the (official) LOS for a backfield player.
            The offensive linemen sit ~1.2 yd behind the official line (median; the
            centre ~0.7), so 1 yd would count the line; 2.5 excludes it and the
            under-centre QB is excluded explicitly.
        backfield_lateral: max |lateral| for a backfield player.
        shotgun_depth: QB depth behind the LOS at the snap that counts as shotgun.
        motion_window_s: seconds before the snap scanned for lateral motion.
        motion_disp: lateral range (max - min y) that counts as motion.
        rush_window_s: seconds after the snap scanned for pass rushers.
        rush_cross_depth: a defender counts as rushing when his min x in the window is
            ``<= los_x + rush_cross_depth`` (0 = reaches the official line; negative =
            penetrates behind it).
        pressure_radius: radius around the QB for ``n_def_within_r_qb``.
        target_radius: radius around the targeted receiver for defender counts.
        carrier_radius: radius around the ball carrier for defender counts.
        ol_front_depth: max depth_behind for a player to be an offensive-line candidate.
        inline_lateral_gap: an inline TE is within this many yards outside the widest
            offensive lineman on his side.
        side_middle_halfwidth: |lateral| up to which a location counts as "middle" in
            :func:`side_of_lateral` (6 yd: the inter-quartile range of nflfastR
            ``pass_location == "middle"`` targets is about -4.5 .. +3.4 yd).
        release_lag_frames: frames before the ``pass_forward`` tag at which the ball is
            still in the passer's hand. Empirically the tag lands ~0.2-0.3 s after
            release (ball already 3 yd away at 13 yd/s), so the passer is identified as
            the offensive player nearest the ball this many frames earlier (3 frames:
            95% roster-QB agreement vs 55% at the tag itself).
        fps: frames per second of the tracking data.
    """

    deep_depth: float = 10.0
    box_depth: float = 5.0
    box_lateral: float = 8.0
    dl_depth: float = 1.5
    wide_lateral: float = 8.0
    backfield_depth: float = 2.5
    backfield_lateral: float = 6.0
    shotgun_depth: float = 4.0
    motion_window_s: float = 2.0
    motion_disp: float = 3.0
    rush_window_s: float = 1.5
    rush_cross_depth: float = 0.0
    pressure_radius: float = 3.0
    target_radius: float = 5.0
    carrier_radius: float = 3.0
    ol_front_depth: float = 2.0
    inline_lateral_gap: float = 4.0
    side_middle_halfwidth: float = 6.0
    release_lag_frames: int = 3
    fps: float = 10.0

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class PlayFrames:
    """One play's tracking data in wide form, normalised to attack toward +x.

    Attributes:
        frame_ids: raw ``frame.id`` values ``[F]`` (sorted ascending).
        player_ids: player identifiers ``[P]``.
        off_mask: ``[P]`` bool, True for offensive players.
        x, y, s, dir: ``[F, P]`` floats (NaN where a player has no row in a frame).
        ball_x, ball_y: ``[F]`` floats (NaN where no ball row).
        events: first frame index (0-based position in ``frame_ids``) per event name.
        snap_idx: frame index of ``ball_snap``.
        los_x: line of scrimmage used for depth (official yard line when given).
        los_x_ball: ball x at the snap (normalised), NaN if no ball row.
        ball_y_ref: lateral reference (ball y at the snap or OL median fallback).
        ball_y_source: ``"ball"`` or ``"centre_estimate"``.
        flipped: whether the raw coordinates were mirrored.
        fps: frames per second.
    """

    frame_ids: np.ndarray
    player_ids: np.ndarray
    off_mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    dir: np.ndarray
    ball_x: np.ndarray
    ball_y: np.ndarray
    events: dict[str, int]
    snap_idx: int
    los_x: float
    los_x_ball: float
    ball_y_ref: float
    ball_y_source: str
    flipped: bool
    fps: float = 10.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_players(self) -> int:
        return int(self.x.shape[1])

    @property
    def def_mask(self) -> np.ndarray:
        return ~self.off_mask

    @classmethod
    def from_long(
        cls,
        frame_id: np.ndarray,
        player_id: np.ndarray,
        team: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
        dir_: np.ndarray,
        event: np.ndarray,
        offense_label: str,
        los_official: float | None = None,
        fps: float = 10.0,
        ball_label: str = BALL_LABEL,
    ) -> PlayFrames:
        """Pivot long-format rows of one play into ``[F, P]`` arrays and normalise.

        Args:
            frame_id: ``[N]`` frame ids (any ints; sorted internally).
            player_id: ``[N]`` player ids; ignored (may be NaN) on ball rows.
            team: ``[N]`` strings: ``"home"``, ``"away"`` or ``ball_label``.
            x, y, s, dir_: ``[N]`` floats in raw field coordinates.
            event: ``[N]`` object array of event tags (None / NaN for no event).
            offense_label: which of ``"home"`` / ``"away"`` is on offense.
            los_official: official LOS in raw coordinates is *not* known before the
                orientation is; pass the offense-normalised value (10 + own yardline or
                110 - opponent yardline) or None to use the ball.
            fps: frames per second.

        Raises:
            ValueError: if the play has no ``ball_snap`` event.
        """
        frame_id = np.asarray(frame_id)
        team = np.asarray(team).astype(str)
        is_ball = team == ball_label
        frames, f_inv = np.unique(frame_id, return_inverse=True)
        pid_raw = np.asarray(player_id)[~is_ball]
        pids, p_inv = np.unique(pid_raw, return_inverse=True)
        n_f, n_p = len(frames), len(pids)

        def pivot(v: np.ndarray) -> np.ndarray:
            out = np.full((n_f, n_p), np.nan)
            out[f_inv[~is_ball], p_inv] = np.asarray(v, dtype=float)[~is_ball]
            return out

        xx, yy, ss, dd = pivot(x), pivot(y), pivot(s), pivot(dir_)
        bx = np.full(n_f, np.nan)
        by = np.full(n_f, np.nan)
        bx[f_inv[is_ball]] = np.asarray(x, dtype=float)[is_ball]
        by[f_inv[is_ball]] = np.asarray(y, dtype=float)[is_ball]

        team_of_pid = np.empty(n_p, dtype=object)
        team_of_pid[p_inv] = team[~is_ball]
        off_mask = team_of_pid == offense_label

        events: dict[str, int] = {}
        ev = np.asarray(event, dtype=object)
        for i in np.where([isinstance(e, str) and e != "" for e in ev])[0]:
            name = str(ev[i])
            fi = int(f_inv[i])
            if name not in events or fi < events[name]:
                events[name] = fi
        if "ball_snap" not in events:
            raise ValueError("play has no ball_snap event")
        snap_idx = events["ball_snap"]

        attack = infer_attack_direction(xx[snap_idx], off_mask)
        flipped = attack < 0
        if flipped:
            xx, yy, dd = flip_coordinates(xx, yy, dd)
            bx = FIELD_LENGTH - bx
            by = FIELD_WIDTH - by

        los_x_ball = float(bx[snap_idx])
        if los_official is not None and np.isfinite(los_official):
            los_x = float(los_official)
        elif np.isfinite(los_x_ball):
            los_x = los_x_ball
        else:
            los_x = float(np.nanmedian(xx[snap_idx][~off_mask])) - 0.7
        ball_y_ref, src = reference_lateral(xx[snap_idx], yy[snap_idx], off_mask, los_x,
                                            float(by[snap_idx]))
        return cls(
            frame_ids=frames, player_ids=pids, off_mask=off_mask, x=xx, y=yy, s=ss, dir=dd,
            ball_x=bx, ball_y=by, events=events, snap_idx=snap_idx, los_x=los_x,
            los_x_ball=los_x_ball, ball_y_ref=ball_y_ref, ball_y_source=src,
            flipped=bool(flipped), fps=fps,
        )


# ---------------------------------------------------------------------------
# Orientation and references
# ---------------------------------------------------------------------------

def infer_attack_direction(x_snap: np.ndarray, off_mask: np.ndarray) -> int:
    """Return +1 if the offense attacks toward +x, else -1.

    Uses the median x of each unit at the snap: the offense lines up on the side of
    the ball away from the defense. Robust to ball-tracking glitches (unlike a
    ball-relative rule) and to a lone offensive player far from the line.

    Args:
        x_snap: ``[P]`` x coordinates at the snap.
        off_mask: ``[P]`` bool.
    """
    x_snap = np.asarray(x_snap, dtype=float)
    off = np.nanmedian(x_snap[off_mask])
    de = np.nanmedian(x_snap[~off_mask])
    return 1 if off < de else -1


def flip_coordinates(x: np.ndarray, y: np.ndarray, dir_: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the field through its centre and rotate motion angles by 180 degrees."""
    return FIELD_LENGTH - x, FIELD_WIDTH - y, np.mod(dir_ + 180.0, 360.0)


def official_los(yardline_side: str | None, yardline_number: float | None,
                 possession_team: str | None) -> float:
    """Offense-normalised LOS from the play-by-play yard line (NaN when unknown).

    ``10 + n`` when the ball is on the offense's own ``n`` yard line, ``110 - n`` on the
    opponent's, and 60 at midfield (side missing, number 50).
    """
    if yardline_number is None or not np.isfinite(float(yardline_number)):
        return float("nan")
    n = float(yardline_number)
    if yardline_side is None or (isinstance(yardline_side, float) and np.isnan(yardline_side)):
        return 60.0 if n == 50 else float("nan")
    return 10.0 + n if str(yardline_side) == str(possession_team) else 110.0 - n


def offensive_line_idx(x_snap: np.ndarray, y_snap: np.ndarray, off_mask: np.ndarray,
                       los_x: float, qb_idx: int | None, y_ref: float,
                       front_depth: float = 2.0, n_ol: int = 5) -> np.ndarray:
    """Indices of the (up to) ``n_ol`` offensive players closest laterally to ``y_ref``
    among those within ``front_depth`` yards behind the LOS (the interior line)."""
    cand = off_mask & (los_x - x_snap <= front_depth) & (los_x - x_snap >= -1.0)
    if qb_idx is not None:
        cand[qb_idx] = False
    idx = np.where(cand)[0]
    if len(idx) == 0:
        return idx
    order = np.argsort(np.abs(y_snap[idx] - y_ref))
    return idx[order[:n_ol]]


def estimate_centre_y(x_snap: np.ndarray, y_snap: np.ndarray, off_mask: np.ndarray,
                      los_x: float, front_depth: float = 2.0) -> float:
    """Estimate the centre's y without the ball: the most forward offensive player on the
    line is over the ball; refine with the median y of the five line players nearest him."""
    front = off_mask & (los_x - x_snap <= front_depth) & (los_x - x_snap >= -1.0) & np.isfinite(x_snap)
    if not front.any():
        return float(np.nanmedian(y_snap[off_mask]))
    idx = np.where(front)[0]
    j = idx[int(np.nanargmax(x_snap[idx]))]
    order = idx[np.argsort(np.abs(y_snap[idx] - y_snap[j]))][:5]
    return float(np.nanmedian(y_snap[order]))


def reference_lateral(x_snap: np.ndarray, y_snap: np.ndarray, off_mask: np.ndarray,
                      los_x: float, ball_y: float, tol: float = 8.0) -> tuple[float, str]:
    """Lateral reference: the ball's y at the snap unless implausible.

    Implausible = NaN, outside the field, or more than ``tol`` yards from the median y
    of the offense (tracking glitches are gross, 10+ yards). The fallback is
    :func:`estimate_centre_y`.
    """
    off_med = float(np.nanmedian(y_snap[off_mask]))
    if np.isfinite(ball_y) and 0.0 <= ball_y <= FIELD_WIDTH and abs(ball_y - off_med) <= tol:
        return float(ball_y), "ball"
    return estimate_centre_y(x_snap, y_snap, off_mask, los_x), "centre_estimate"


def identify_qb_geometric(x_snap: np.ndarray, y_snap: np.ndarray, off_mask: np.ndarray,
                          los_x: float, y_ref: float, max_lateral: float = 0.8,
                          max_depth: float = 9.0, min_gap: float = 0.5) -> int | None:
    """Geometric QB: the offensive player directly behind the centre.

    Candidates are offensive players within ``max_lateral`` of the ball and at most
    ``max_depth`` behind the line; the most forward candidate is the centre (over the
    ball) and the QB is the next candidate at least ``min_gap`` yards behind him
    (under centre ~1 yd, shotgun 4-6 yd). Returns None if no such player.
    """
    depth_behind = los_x - x_snap
    cand = off_mask & (np.abs(y_snap - y_ref) <= max_lateral) & (depth_behind <= max_depth) \
        & (depth_behind >= -1.0) & np.isfinite(depth_behind)
    idx = np.where(cand)[0]
    if len(idx) < 2:
        return None
    idx = idx[np.argsort(depth_behind[idx])]
    centre_depth = depth_behind[idx[0]]
    for j in idx[1:]:
        if depth_behind[j] >= centre_depth + min_gap:
            return int(j)
    return None


# ---------------------------------------------------------------------------
# Small geometric helpers
# ---------------------------------------------------------------------------

def nearest_to_point(x: np.ndarray, y: np.ndarray, px: float, py: float,
                     mask: np.ndarray) -> int | None:
    """Index of the masked player nearest to ``(px, py)``; None if none / NaN."""
    d = np.hypot(x - px, y - py)
    d = np.where(mask, d, np.nan)
    if np.all(np.isnan(d)):
        return None
    return int(np.nanargmin(d))


def min_dist_to_group(x: np.ndarray, y: np.ndarray, idx: int, group_mask: np.ndarray) -> float:
    """Minimum distance from player ``idx`` to any player in ``group_mask``."""
    d = np.hypot(x - x[idx], y - y[idx])
    d = np.where(group_mask, d, np.nan)
    d[idx] = np.nan
    return float(np.nanmin(d)) if not np.all(np.isnan(d)) else float("nan")


def count_within(x: np.ndarray, y: np.ndarray, idx: int, group_mask: np.ndarray,
                 radius: float) -> int:
    """Number of players in ``group_mask`` within ``radius`` of player ``idx``."""
    d = np.hypot(x - x[idx], y - y[idx])
    m = group_mask.copy()
    m[idx] = False
    return int(np.sum((d <= radius) & m & np.isfinite(d)))


def box_count(depth: np.ndarray, lateral: np.ndarray, def_mask: np.ndarray,
              box_depth: float, box_lateral: float, min_depth: float = -1.0) -> int:
    """Defenders within ``box_depth`` yards beyond the LOS and ``box_lateral`` of the ball."""
    m = def_mask & (depth <= box_depth) & (depth >= min_depth) & (np.abs(lateral) <= box_lateral)
    return int(np.sum(m))


def side_of_lateral(lateral: float, middle_halfwidth: float) -> str | None:
    """Offense-perspective side (``"left"`` / ``"middle"`` / ``"right"``) of a lateral offset.

    ``left`` is ``lateral * LEFT_SIGN > middle_halfwidth`` (positive lateral, see the
    module docstring), ``right`` the mirror image, ``middle`` in between; None for NaN.
    """
    if lateral is None or not np.isfinite(lateral):
        return None
    v = float(lateral) * LEFT_SIGN
    if v > middle_halfwidth:
        return "left"
    if v < -middle_halfwidth:
        return "right"
    return "middle"


# ---------------------------------------------------------------------------
# Pre-snap targets
# ---------------------------------------------------------------------------

def presnap_targets(pf: PlayFrames, qb_idx: int | None,
                    cfg: FeatureConfig = FeatureConfig()) -> dict[str, Any]:
    """Alignment targets at the snap frame.

    Args:
        pf: normalised play frames.
        qb_idx: column index of the quarterback (position-based when known, else from
            :func:`identify_qb_geometric`), or None.

    Returns:
        Flat dict of scalar targets (see the build report for definitions).
    """
    si = pf.snap_idx
    xs, ys = pf.x[si], pf.y[si]
    off, de = pf.off_mask, pf.def_mask
    depth = xs - pf.los_x
    lateral = ys - pf.ball_y_ref
    out: dict[str, Any] = {}

    # --- defense ---
    dd = depth[de]
    dd_sorted = np.sort(dd[np.isfinite(dd)])[::-1]
    out["n_deep_safeties"] = int(np.sum(dd >= cfg.deep_depth))
    out["def_depth_max"] = float(dd_sorted[0]) if len(dd_sorted) else np.nan
    out["def_depth_2nd"] = float(dd_sorted[1]) if len(dd_sorted) > 1 else np.nan
    out["def_depth_mean"] = float(np.nanmean(dd)) if len(dd) else np.nan
    out["mof_open"] = int(out["n_deep_safeties"] >= 2)
    out["box_count"] = box_count(depth, lateral, de, cfg.box_depth, cfg.box_lateral)
    out["n_dl"] = int(np.sum(de & (np.abs(depth) <= cfg.dl_depth)))
    out["def_y_std"] = float(np.nanstd(ys[de])) if de.any() else np.nan
    out["def_y_range"] = float(np.nanmax(ys[de]) - np.nanmin(ys[de])) if de.any() else np.nan
    out["def_lateral_mean"] = float(np.nanmean(lateral[de])) if de.any() else np.nan

    # --- offense structure ---
    ol_idx = offensive_line_idx(xs, ys, off, pf.los_x, qb_idx, pf.ball_y_ref,
                                front_depth=cfg.ol_front_depth)
    is_ol = np.zeros(pf.n_players, dtype=bool)
    is_ol[ol_idx] = True
    is_qb = np.zeros(pf.n_players, dtype=bool)
    if qb_idx is not None:
        is_qb[qb_idx] = True
    skill = off & ~is_ol & ~is_qb
    depth_behind = -depth

    # left = offense's left = positive lateral (LEFT_SIGN); see module docstring
    out["n_wide_left"] = int(np.sum(off & ~is_qb & (lateral * LEFT_SIGN >= cfg.wide_lateral)))
    out["n_wide_right"] = int(np.sum(off & ~is_qb & (lateral * LEFT_SIGN <= -cfg.wide_lateral)))
    out["n_wide"] = out["n_wide_left"] + out["n_wide_right"]
    out["widest_split"] = float(np.nanmax(np.abs(lateral[off & ~is_qb]))) if (off & ~is_qb).any() else np.nan
    bf = off & ~is_qb & (depth_behind >= cfg.backfield_depth) & (np.abs(lateral) <= cfg.backfield_lateral)
    out["n_backfield"] = int(np.sum(bf))
    out["n_ol_derived"] = int(len(ol_idx))
    out["off_y_std"] = float(np.nanstd(ys[off])) if off.any() else np.nan

    # inline TE proxy: skill players on the line, just outside the widest lineman per side
    n_inline = 0
    for sign in (-1.0, 1.0):
        side_ol = lateral[ol_idx] * sign
        edge = float(np.max(side_ol)) if len(side_ol) and np.max(side_ol) > 0 else 0.0
        m = skill & (depth_behind <= cfg.backfield_depth) & (lateral * sign > edge) \
            & (lateral * sign <= edge + cfg.inline_lateral_gap)
        n_inline += int(np.sum(m))
    out["n_te_inline"] = n_inline

    # --- QB ---
    if qb_idx is not None:
        out["qb_depth"] = float(depth_behind[qb_idx])
        out["qb_lateral"] = float(lateral[qb_idx])
        out["shotgun_derived"] = int(out["qb_depth"] >= cfg.shotgun_depth)
    else:
        out["qb_depth"] = np.nan
        out["qb_lateral"] = np.nan
        out["shotgun_derived"] = np.nan

    # --- cushion: widest skill player on each side vs nearest defender ---
    cush = {}
    for name, sign in (("left", LEFT_SIGN), ("right", -LEFT_SIGN)):
        cand = np.where(skill & (lateral * sign > 0))[0]
        if len(cand) == 0:
            cush[name] = np.nan
            continue
        w = int(cand[np.argmax(np.abs(lateral[cand]))])
        cush[name] = min_dist_to_group(xs, ys, w, de)
    out["cushion_left"] = cush["left"]
    out["cushion_right"] = cush["right"]
    vals = [v for v in cush.values() if np.isfinite(v)]
    out["cb_cushion"] = float(np.mean(vals)) if vals else np.nan

    # --- motion in the window before the snap ---
    w = int(round(cfg.motion_window_s * pf.fps))
    lo = max(0, si - w)
    out["frames_before_snap"] = int(si)
    if si - lo >= 2:
        yw = pf.y[lo:si + 1][:, off]
        rng = np.nanmax(yw, axis=0) - np.nanmin(yw, axis=0)
        out["motion_disp_max"] = float(np.nanmax(rng)) if rng.size else np.nan
    else:
        out["motion_disp_max"] = np.nan
    out["motion_event"] = int("man_in_motion" in pf.events and pf.events["man_in_motion"] <= si)
    out["motion_derived"] = int(out["motion_event"] == 1 or
                                (np.isfinite(out["motion_disp_max"]) and out["motion_disp_max"] > cfg.motion_disp))
    for ev in ("line_set", "shift", "huddle_break_offense"):
        out[f"{ev}_to_snap_s"] = (si - pf.events[ev]) / pf.fps if ev in pf.events and pf.events[ev] <= si else np.nan
    return out


def participant_rows(pf: PlayFrames, qb_idx: int | None,
                     cfg: FeatureConfig = FeatureConfig()) -> list[dict[str, Any]]:
    """One dict per tracked player with snap-frame geometry (for the participants table).

    ``lateral`` follows the module convention (positive = offense's left).
    """
    si = pf.snap_idx
    xs, ys, ss = pf.x[si], pf.y[si], pf.s[si]
    depth = xs - pf.los_x
    lateral = ys - pf.ball_y_ref
    ol_idx = offensive_line_idx(xs, ys, pf.off_mask, pf.los_x, qb_idx, pf.ball_y_ref,
                                front_depth=cfg.ol_front_depth)
    rows = []
    for j in range(pf.n_players):
        off = bool(pf.off_mask[j])
        if off:
            if qb_idx is not None and j == qb_idx:
                role = "qb"
            elif j in ol_idx:
                role = "ol"
            elif -depth[j] >= cfg.backfield_depth and abs(lateral[j]) <= cfg.backfield_lateral:
                role = "backfield"
            elif abs(lateral[j]) >= cfg.wide_lateral:
                role = "wide"
            else:
                role = "tight_slot"
        else:
            if abs(depth[j]) <= cfg.dl_depth:
                role = "line"
            elif depth[j] >= cfg.deep_depth:
                role = "deep"
            elif depth[j] <= cfg.box_depth and abs(lateral[j]) <= cfg.box_lateral:
                role = "box"
            else:
                role = "second_level"
        rows.append({
            "player_id": pf.player_ids[j], "side": "offense" if off else "defense",
            "x_snap": float(xs[j]), "y_snap": float(ys[j]), "s_snap": float(ss[j]),
            "depth": float(depth[j]), "lateral": float(lateral[j]),
            "in_box_naive": bool((not off) and depth[j] <= cfg.box_depth and depth[j] >= -1.0
                                 and abs(lateral[j]) <= cfg.box_lateral),
            "role_derived": role,
            "is_qb_used": bool(qb_idx is not None and j == qb_idx),
        })
    return rows


# ---------------------------------------------------------------------------
# Within-play targets
# ---------------------------------------------------------------------------

def pass_rushers(pf: PlayFrames, window_s: float, cross_depth: float) -> int:
    """Defenders whose minimum x within ``window_s`` after the snap reaches
    ``los_x + cross_depth`` (i.e. they cross to / behind the line)."""
    si = pf.snap_idx
    hi = min(pf.n_frames, si + int(round(window_s * pf.fps)) + 1)
    xw = pf.x[si:hi][:, pf.def_mask]
    if xw.size == 0:
        return 0
    with np.errstate(all="ignore"):
        mn = np.nanmin(xw, axis=0)
    return int(np.sum(mn <= pf.los_x + cross_depth))


def _release_index(pf: PlayFrames, is_sack: bool) -> tuple[int | None, str | None]:
    if is_sack:
        if "qb_sack" in pf.events:
            return pf.events["qb_sack"], "qb_sack"
        return None, None
    for ev in ("pass_forward", "pass_shovel"):
        if ev in pf.events:
            return pf.events[ev], ev
    if "qb_sack" in pf.events:
        return pf.events["qb_sack"], "qb_sack"
    return None, None


def pass_targets(pf: PlayFrames, qb_idx: int | None, pass_result: str,
                 cfg: FeatureConfig = FeatureConfig(),
                 rush_grid: tuple[tuple[float, float], ...] = ()) -> dict[str, Any]:
    """Within-play targets for a pass play (``pass_result`` in C / I / IN / S).

    Args:
        pf: normalised frames.
        qb_idx: pre-snap QB column (used to exclude him from receiver / carrier search).
        pass_result: BDB ``PassResult`` code.
        cfg: thresholds.
        rush_grid: extra ``(window_s, cross_depth)`` settings; each adds a column
            ``n_rush_w<window*10>_c<m|p><|depth|*10>`` for definition tuning.
    """
    out: dict[str, Any] = {}
    si = pf.snap_idx
    is_sack = pass_result == "S"
    ri, rev = _release_index(pf, is_sack)
    out["release_event"] = rev
    out["time_to_throw"] = (ri - si) / pf.fps if ri is not None else np.nan

    qb_t: int | None = None
    if ri is not None:
        li = max(si, ri - int(cfg.release_lag_frames))
        bx, by = pf.ball_x[li], pf.ball_y[li]
        if np.isfinite(bx):
            qb_t = nearest_to_point(pf.x[li], pf.y[li], bx, by, pf.off_mask)
        out["qb_ball_dist_at_release"] = (float(np.hypot(pf.x[li, qb_t] - bx, pf.y[li, qb_t] - by))
                                          if qb_t is not None else np.nan)
        if qb_t is None:
            qb_t = qb_idx
    else:
        out["qb_ball_dist_at_release"] = np.nan
    out["qb_id_throw"] = pf.player_ids[qb_t] if qb_t is not None else None
    out["qb_same_as_presnap"] = int(qb_t == qb_idx) if (qb_t is not None and qb_idx is not None) else np.nan
    if ri is not None and qb_t is not None:
        out["qb_depth_at_throw"] = float(pf.los_x - pf.x[ri, qb_t])
        out["qb_lateral_at_throw"] = float(pf.y[ri, qb_t] - pf.ball_y_ref)
        out["qb_speed_at_throw"] = float(pf.s[ri, qb_t])
        out["min_def_dist_qb_throw"] = min_dist_to_group(pf.x[ri], pf.y[ri], qb_t, pf.def_mask)
        out["n_def_within_r_qb_throw"] = count_within(pf.x[ri], pf.y[ri], qb_t, pf.def_mask,
                                                      cfg.pressure_radius)
        pi = max(si, ri - int(round(0.5 * pf.fps)))
        out["min_def_dist_qb_throw_m05"] = min_dist_to_group(pf.x[pi], pf.y[pi], qb_t, pf.def_mask)
        # closest a defender got to the QB at any point between snap and release
        seg = slice(si, ri + 1)
        dx = pf.x[seg][:, pf.def_mask] - pf.x[seg][:, [qb_t]]
        dy = pf.y[seg][:, pf.def_mask] - pf.y[seg][:, [qb_t]]
        with np.errstate(all="ignore"):
            out["min_def_dist_qb_dropback"] = float(np.nanmin(np.hypot(dx, dy))) if dx.size else np.nan
    else:
        for k in ("qb_depth_at_throw", "qb_lateral_at_throw", "qb_speed_at_throw",
                  "min_def_dist_qb_throw", "n_def_within_r_qb_throw",
                  "min_def_dist_qb_throw_m05", "min_def_dist_qb_dropback"):
            out[k] = np.nan

    out["n_pass_rushers_derived"] = pass_rushers(pf, cfg.rush_window_s, cfg.rush_cross_depth)
    for w, c in rush_grid:
        out[rush_col(w, c)] = pass_rushers(pf, w, c)

    # targeted receiver at arrival
    ai, aev = None, None
    for ev in ARRIVAL_EVENTS:
        if ev in pf.events:
            ai, aev = pf.events[ev], ev
            break
    out["arrival_event"] = aev
    tgt: int | None = None
    if ai is not None and np.isfinite(pf.ball_x[ai]):
        mask = pf.off_mask.copy()
        if qb_t is not None:
            mask[qb_t] = False
        tgt = nearest_to_point(pf.x[ai], pf.y[ai], pf.ball_x[ai], pf.ball_y[ai], mask)
    out["target_id"] = pf.player_ids[tgt] if tgt is not None else None
    if tgt is not None and ai is not None:
        out["separation_at_arrival"] = min_dist_to_group(pf.x[ai], pf.y[ai], tgt, pf.def_mask)
        out["n_def_within_r_target"] = count_within(pf.x[ai], pf.y[ai], tgt, pf.def_mask,
                                                    cfg.target_radius)
        out["target_depth"] = float(pf.x[ai, tgt] - pf.los_x)
        out["target_lateral"] = float(pf.y[ai, tgt] - pf.ball_y_ref)
        out["target_side_derived"] = side_of_lateral(out["target_lateral"], cfg.side_middle_halfwidth)
        out["target_speed_at_arrival"] = float(pf.s[ai, tgt])
        out["target_ball_dist_at_arrival"] = float(np.hypot(pf.x[ai, tgt] - pf.ball_x[ai],
                                                            pf.y[ai, tgt] - pf.ball_y[ai]))
        out["ball_depth_at_arrival"] = float(pf.ball_x[ai] - pf.los_x)
        out["air_time_s"] = (ai - ri) / pf.fps if ri is not None else np.nan
        # separation at the throw (how open the target was when the ball left)
        if ri is not None:
            out["separation_at_throw"] = min_dist_to_group(pf.x[ri], pf.y[ri], tgt, pf.def_mask)
        else:
            out["separation_at_throw"] = np.nan
    else:
        for k in ("separation_at_arrival", "n_def_within_r_target", "target_depth",
                  "target_lateral", "target_speed_at_arrival", "target_ball_dist_at_arrival",
                  "ball_depth_at_arrival", "air_time_s", "separation_at_throw"):
            out[k] = np.nan
        out["target_side_derived"] = None
    return out


def run_targets(pf: PlayFrames, qb_idx: int | None,
                cfg: FeatureConfig = FeatureConfig()) -> dict[str, Any]:
    """Within-play targets for a running play (handoff / designed run / scramble)."""
    out: dict[str, Any] = {}
    si = pf.snap_idx
    hi_ev = None
    for ev in ("handoff", "run", "pass_shovel"):
        if ev in pf.events:
            hi_ev = ev
            break
    out["carrier_event"] = hi_ev
    car: int | None = None
    hi = pf.events[hi_ev] if hi_ev is not None else None
    if hi is not None and np.isfinite(pf.ball_x[hi]):
        mask = pf.off_mask.copy()
        if hi_ev != "run" and qb_idx is not None:
            mask[qb_idx] = False
        car = nearest_to_point(pf.x[hi], pf.y[hi], pf.ball_x[hi], pf.ball_y[hi], mask)
    out["carrier_id"] = pf.player_ids[car] if car is not None else None
    if car is not None and hi is not None:
        out["time_to_handoff"] = (hi - si) / pf.fps
        out["carrier_speed_at_handoff"] = float(pf.s[hi, car])
        out["carrier_depth_at_handoff"] = float(pf.los_x - pf.x[hi, car])
        out["carrier_lateral_at_handoff"] = float(pf.y[hi, car] - pf.ball_y_ref)
        out["n_def_within_r_carrier_handoff"] = count_within(pf.x[hi], pf.y[hi], car, pf.def_mask,
                                                             cfg.carrier_radius)
        out["min_def_dist_carrier_handoff"] = min_dist_to_group(pf.x[hi], pf.y[hi], car, pf.def_mask)
    else:
        for k in ("time_to_handoff", "carrier_speed_at_handoff", "carrier_depth_at_handoff",
                  "carrier_lateral_at_handoff", "n_def_within_r_carrier_handoff",
                  "min_def_dist_carrier_handoff"):
            out[k] = np.nan
    fc = pf.events.get("first_contact")
    if car is not None and fc is not None and fc >= si:
        out["yards_to_first_contact"] = float(pf.x[fc, car] - pf.los_x)
        out["time_to_first_contact"] = (fc - si) / pf.fps
        out["carrier_speed_at_first_contact"] = float(pf.s[fc, car])
        out["n_def_within_r_carrier_first_contact"] = count_within(pf.x[fc], pf.y[fc], car,
                                                                   pf.def_mask, cfg.carrier_radius)
    else:
        for k in ("yards_to_first_contact", "time_to_first_contact",
                  "carrier_speed_at_first_contact", "n_def_within_r_carrier_first_contact"):
            out[k] = np.nan
    return out


def rush_col(window_s: float, cross_depth: float) -> str:
    """Column name for a pass-rusher definition, e.g. ``n_rush_w15_cm05``."""
    sign = "m" if cross_depth < 0 else "p"
    return f"n_rush_w{int(round(window_s * 10)):02d}_c{sign}{int(round(abs(cross_depth) * 10)):02d}"


# ---------------------------------------------------------------------------
# Personnel
# ---------------------------------------------------------------------------

def personnel_counts(positions: list[str] | np.ndarray, side: str) -> dict[str, int]:
    """Count position groups from roster position abbreviations.

    Args:
        positions: position abbreviation per player on that side.
        side: ``"offense"`` -> RB/TE/WR/OL/QB counts; ``"defense"`` -> DL/LB/DB counts.
    """
    pos = [str(p) for p in positions]
    if side == "offense":
        return {
            "RB": sum(p in OFF_POS_RB for p in pos),
            "TE": sum(p in OFF_POS_TE for p in pos),
            "WR": sum(p in OFF_POS_WR for p in pos),
            "OL": sum(p in OFF_POS_OL for p in pos),
            "QB": sum(p in OFF_POS_QB for p in pos),
        }
    return {
        "DL": sum(p in DEF_POS_DL for p in pos),
        "LB": sum(p in DEF_POS_LB for p in pos),
        "DB": sum(p in DEF_POS_DB for p in pos),
    }


def parse_personnel(s: str | None) -> dict[str, int]:
    """Parse ``"6 OL, 1 RB, 2 TE, 1 WR"`` style strings into ``{group: count}``.

    Unknown / NaN input returns an empty dict. Groups not mentioned are absent; the
    NGS convention omits ``OL`` when it is 5 and ``QB`` when it is 1.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    out: dict[str, int] = {}
    for part in str(s).split(","):
        toks = part.strip().split()
        if len(toks) == 2 and toks[0].isdigit():
            out[toks[1].upper()] = int(toks[0])
    return out


def personnel_string(counts: dict[str, int], side: str) -> str:
    """Render counts in the NGS string convention (``"1 RB, 1 TE, 3 WR"``)."""
    if side == "offense":
        parts = []
        if counts.get("OL", 5) != 5:
            parts.append(f"{counts.get('OL', 5)} OL")
        parts += [f"{counts.get('RB', 0)} RB", f"{counts.get('TE', 0)} TE", f"{counts.get('WR', 0)} WR"]
        if counts.get("QB", 1) != 1:
            parts.append(f"{counts.get('QB', 1)} QB")
        return ", ".join(parts)
    return f"{counts.get('DL', 0)} DL, {counts.get('LB', 0)} LB, {counts.get('DB', 0)} DB"
