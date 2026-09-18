"""NFL 03b - out-of-distribution check of the pre-snap students on NGS highlight plays.

The `asonty/ngs_highlights` repository holds tracking for 567 Next Gen Stats highlight plays of
2017-2019 (touchdowns and long gains: a strongly biased sample of plays, the only public NFL
tracking outside the Big Data Bowl). This script

1. draws ``max_plays`` plays with a fixed seed from ``nfl_dir()/ngs_highlights/index.tsv``
   (2018 and 2019 rows), fetching each play's TSV from the raw GitHub URL into
   ``nfl_dir()/ngs_highlights/play_data/`` (skipped when cached);
2. recomputes two pre-snap targets from the highlight frames with the NFL 01 definitions
   (:func:`highlight_targets`): ``n_deep_safeties`` (defender depth >= 10 yd at the snap,
   ``mof_open`` = at least two) and ``box_count_tuned`` (-1 <= depth <= 6, |lateral| <= 7).
   The snap frame is chosen deterministically (:func:`choose_snap_frame`: the earliest
   ``ball_snap`` frame whose ball row is physically plausible, :func:`ball_row_valid`; several
   highlight files contain the play twice or tag a mid-play frame ``ball_snap`` a second time),
   its rows are de-duplicated on ``nflId`` / ``displayName`` (a frame carrying two event tags
   lists every player twice), a play is kept only when it is exactly 11 v 11, and when no snap
   frame has a plausible ball row the lateral reference of the box falls back to the centre's
   / offensive line's y (:func:`lateral_reference`) instead of a glitched ball position;
3. applies the saved F0 / F0P / F1 students (``apply_student``) to the 2018 and 2019 nflfastR
   regular seasons and scores them on the highlight plays (play-level bootstrap CIs), next to
   the 2017 out-of-fold numbers on all tracked plays and on the "highlight-like" 2017 subset
   (touchdown or >= 20 yards gained).

Highlight frames carry ``absoluteYardlineNumber`` (the line of scrimmage in field
coordinates) and ``playDirection``; depth is measured from that line towards the defense,
lateral from the ball's y at the snap. Run from the repo root::

    python -m research.privileged_tracking.nfl.ood_highlights --max-plays 150
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from research.privileged_tracking.common.io import nfl_dir, processed_dir, reports_dir
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.nfl import apply_student as aps
from research.privileged_tracking.nfl import imputation as imp
from research.privileged_tracking.nfl import imputation_features as imf
from research.privileged_tracking.nfl import tracking_features as tf

RAW_URL = "https://raw.githubusercontent.com/asonty/ngs_highlights/master/play_data/{name}"
OOD_TARGETS = ("n_deep_safeties", "mof_open", "box_count_tuned")
DEEP_DEPTH = tf.FeatureConfig().deep_depth
BOX_DEPTH, BOX_LATERAL = 6.0, 7.0   # the ``box_count_tuned`` definition selected in NFL 01
#: Field bounds (yards) and the tolerated |ball x - LOS| (attacking direction) of a plausible snap-frame ball row.
FIELD_X: tuple[float, float] = (0.0, 120.0)
FIELD_Y: tuple[float, float] = (0.0, 53.3)
BALL_LOS_TOL = 2.5
#: Offensive players within this depth of the LOS count as "on the line" for the lateral fallback.
LINE_DEPTH_TOL = 1.5
#: Columns every highlight TSV must carry, and the optional ones used when present.
HIGHLIGHT_COLS = ["frame", "event", "x", "y", "teamAbbr", "possessionFlag", "playDirection", "absoluteYardlineNumber"]
OPTIONAL_COLS = ["nflId", "displayName", "position", "positionGroup", "seasonType"]


@dataclass
class OODConfig:
    """Driver configuration.

    Attributes:
        max_plays: highlight plays sampled (seeded) from the 2018-19 index.
        seed: sampling seed (also the bootstrap seed).
        seasons: seasons of the index to draw from.
        feature_sets: student bundles applied.
        n_boot: play-level bootstrap resamples for the highlight-sample skill intervals.
        write: write the parquet / markdown outputs.
    """

    max_plays: int = 150
    seed: int = 0
    seasons: tuple[int, ...] = (2018, 2019)
    feature_sets: tuple[str, ...] = ("F0", "F0P", "F1")
    n_boot: int = 2000
    write: bool = True


# ---------------------------------------------------------------------------
# Highlight frames -> targets (pure)
# ---------------------------------------------------------------------------

PLAYERS_PER_SIDE = 11
DEDUP_KEYS = ("nflId", "displayName")


def ball_row_valid(ball_x: float, ball_y: float, los_x: float, sign: float, tol: float = BALL_LOS_TOL) -> bool:
    """Whether a snap-frame ball row is physically plausible.

    The ball must lie inside the field (:data:`FIELD_X` x :data:`FIELD_Y`) and within ``tol``
    yards of the line of scrimmage along the attacking direction (``sign`` = +1 when the offense
    attacks +x). A missing ball (NaN) is invalid.
    """
    if not (np.isfinite(ball_x) and np.isfinite(ball_y) and np.isfinite(los_x)):
        return False
    inside = FIELD_X[0] <= ball_x <= FIELD_X[1] and FIELD_Y[0] <= ball_y <= FIELD_Y[1]
    return bool(inside and abs(sign * (ball_x - los_x)) <= tol)


def snap_frames(play: pd.DataFrame) -> list[int]:
    """Sorted distinct frames tagged ``ball_snap`` (empty when the play has none)."""
    return sorted(int(f) for f in play.loc[play["event"].astype(object).eq("ball_snap"), "frame"].unique())


def frame_rows(play: pd.DataFrame, frame: int) -> pd.DataFrame:
    """Rows of one frame with duplicated player rows dropped.

    A frame that carries two event tags (e.g. ``ball_snap`` and ``man_in_motion``) lists
    every player - and the ball - twice in the highlight export; rows are de-duplicated on
    :data:`DEDUP_KEYS` (those present; ``teamAbbr, x, y`` when neither is) so each player
    counts once.
    """
    s = play[play["frame"] == frame]
    keys = [c for c in DEDUP_KEYS if c in s.columns] or ["teamAbbr", "x", "y"]
    return s.drop_duplicates(subset=keys)


def _direction_sign(s: pd.DataFrame) -> tuple[str, float]:
    direction = str(s["playDirection"].iloc[0])
    return direction, (1.0 if direction == "right" else -1.0)     # offense attacks +x when "right"


def _ball_xy(s: pd.DataFrame) -> tuple[float, float]:
    is_ball = s["teamAbbr"].isna().to_numpy()
    if not is_ball.any():
        return float("nan"), float("nan")
    return float(s["x"].to_numpy(dtype=float)[is_ball][0]), float(s["y"].to_numpy(dtype=float)[is_ball][0])


def choose_snap_frame(play: pd.DataFrame) -> tuple[int, bool] | None:
    """Deterministic snap frame: the earliest ``ball_snap`` frame with a plausible ball row.

    Returns ``(frame, ball_valid)``; when no ``ball_snap`` frame has a plausible ball row
    (:func:`ball_row_valid`) the earliest one is returned with ``ball_valid = False``, and
    ``None`` when the play has no ``ball_snap`` tag at all. Highlight files sometimes contain
    the play twice (a second copy of the frames after the first) or tag a mid-play frame of the
    first copy as the second copy's snap, so "first in file order" is not a safe choice.
    """
    frames = snap_frames(play)
    if not frames:
        return None
    for f in frames:
        s = frame_rows(play, f)
        _, sign = _direction_sign(s)
        los_x = float(pd.to_numeric(s["absoluteYardlineNumber"], errors="coerce").iloc[0])
        bx, by = _ball_xy(s)
        if ball_row_valid(bx, by, los_x, sign):
            return f, True
    return frames[0], False


def snap_frame_rows(play: pd.DataFrame) -> pd.DataFrame | None:
    """Rows of the chosen snap frame (:func:`choose_snap_frame`), de-duplicated; ``None`` without a snap tag."""
    chosen = choose_snap_frame(play)
    if chosen is None:
        return None
    return frame_rows(play, chosen[0])


def lateral_reference(s: pd.DataFrame, off: np.ndarray, depth: np.ndarray) -> tuple[float, str]:
    """Lateral (y) reference of the formation when the ball row is unusable.

    Preference order: the centre's y (``position == "C"``, exactly one offensive player), the
    median y of the offensive linemen (``positionGroup == "OL"``, at least three), the median y
    of the offensive players within :data:`LINE_DEPTH_TOL` yd of the line (at least three), the
    median y of the offense.

    Args:
        s: de-duplicated snap-frame rows.
        off: offensive-player mask ``[n_rows]``.
        depth: signed depth per row ``[n_rows]`` (defenders positive).

    Returns:
        ``(y_reference, method)`` with method in ``centre`` / ``ol_median`` / ``line_median`` /
        ``offense_median``.
    """
    y = s["y"].to_numpy(dtype=float)
    if "position" in s.columns:
        centre = off & (s["position"].astype(object).to_numpy() == "C")
        if centre.sum() == 1:
            return float(y[centre][0]), "centre"
    if "positionGroup" in s.columns:
        ol = off & (s["positionGroup"].astype(object).to_numpy() == "OL")
        if ol.sum() >= 3:
            return float(np.median(y[ol])), "ol_median"
    line = off & (np.abs(depth) <= LINE_DEPTH_TOL)
    if line.sum() >= 3:
        return float(np.median(y[line])), "line_median"
    return float(np.median(y[off])), "offense_median"


def highlight_targets(play: pd.DataFrame) -> dict[str, Any] | None:
    """Pre-snap targets from one highlight play's frames (NFL 01 definitions).

    Args:
        play: rows of one play with ``frame, event, x, y, teamAbbr, possessionFlag,
            playDirection, absoluteYardlineNumber`` (the ball has ``teamAbbr`` NaN) and,
            when available, ``nflId`` / ``displayName`` for de-duplication and ``position`` /
            ``positionGroup`` for the lateral fallback.

    Returns:
        ``None`` when the play has no ``ball_snap`` tag or is not exactly 11 v 11 at the chosen
        snap frame after de-duplication; otherwise a dict with ``n_deep_safeties``,
        ``mof_open``, ``box_count_tuned``, ``n_def``, ``n_off``, ``los_x``,
        ``los_ball_discrepancy`` (raw ball x - LOS in the attacking direction, NaN without a
        ball row), ``play_direction``, ``snap_frame``, ``n_snap_frames`` (distinct
        ``ball_snap`` frames in the file), ``ball_valid`` (:func:`ball_row_valid` at the chosen
        frame), ``lateral_ref`` (``ball`` or the :func:`lateral_reference` method used) and
        ``lateral_ref_shift`` (|ball y - reference y| when the fallback was used, 0 when the
        ball row itself is the reference, NaN without a ball row) and ``n_rows_snap_raw`` (rows
        of the frame before de-duplication).
    """
    chosen = choose_snap_frame(play)
    if chosen is None:
        return None
    frame, ball_valid = chosen
    s = frame_rows(play, frame)
    n_raw = int((play["frame"] == frame).sum())
    is_ball = s["teamAbbr"].isna().to_numpy()
    poss = pd.to_numeric(s["possessionFlag"], errors="coerce").to_numpy(dtype=float)
    off = (~is_ball) & (poss == 1)
    de = (~is_ball) & (poss == 0)
    if de.sum() != PLAYERS_PER_SIDE or off.sum() != PLAYERS_PER_SIDE:
        return None
    direction, sign = _direction_sign(s)
    los_x = float(pd.to_numeric(s["absoluteYardlineNumber"], errors="coerce").iloc[0])
    x = s["x"].to_numpy(dtype=float)
    y = s["y"].to_numpy(dtype=float)
    depth = sign * (x - los_x)                        # defenders positive
    ball_x, ball_y = _ball_xy(s)
    if ball_valid:
        y_ref, ref, shift = ball_y, "ball", 0.0
    else:
        y_ref, ref = lateral_reference(s, off, depth)
        shift = abs(ball_y - y_ref) if np.isfinite(ball_y) else float("nan")
    lateral = y - y_ref
    dd = depth[de]
    return {"n_deep_safeties": int(np.sum(dd >= DEEP_DEPTH)), "mof_open": int(np.sum(dd >= DEEP_DEPTH) >= 2),
            "box_count_tuned": tf.box_count(depth, lateral, de, BOX_DEPTH, BOX_LATERAL),
            "n_def": int(de.sum()), "n_off": int(off.sum()), "los_x": los_x,
            "los_ball_discrepancy": sign * (ball_x - los_x), "play_direction": direction,
            "snap_frame": int(frame), "n_snap_frames": len(snap_frames(play)), "ball_valid": bool(ball_valid),
            "lateral_ref": ref, "lateral_ref_shift": shift, "n_rows_snap_raw": n_raw}


def skill_bootstrap_ci(kind: str, y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Play-level bootstrap 95% interval of the skill (R2 for reg / count, Brier skill for binary).

    Args:
        kind: ``"binary"`` or ``"reg"`` / ``"count"``.
        y, p: targets and predictions ``[n]`` (NaN pairs dropped).

    Returns:
        ``(ci_low, ci_high)``; NaN when fewer than 10 pairs or a degenerate resample dominates.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb, pb = y[idx], p[idx]
    if kind == "binary":
        pb = np.clip(pb, 1e-6, 1 - 1e-6)
        base = yb.mean(axis=1)
        ref = base * (1 - base)
        res = np.mean((pb - yb) ** 2, axis=1)
    else:
        ref = np.mean((yb - yb.mean(axis=1, keepdims=True)) ** 2, axis=1)
        res = np.mean((yb - pb) ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = 1.0 - res / ref
    skill = skill[np.isfinite(skill)]
    if len(skill) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.quantile(skill, 0.025)), float(np.quantile(skill, 0.975))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sample_index(cfg: OODConfig) -> tuple[pd.DataFrame, int]:
    """Seeded draw of ``cfg.max_plays`` index rows, then de-duplicated on (gameId, playId).

    The index lists some plays twice (under both teams, or twice under one), so a draw can hold
    the same play under two file names; the first drawn row is kept and the number of dropped
    duplicates is returned with the sample. The draw itself is made on the raw index so the
    sample is the one earlier runs used.
    """
    idx = pd.read_csv(nfl_dir() / "ngs_highlights" / "index.tsv", sep="\t")
    idx = idx[idx["season"].isin(cfg.seasons)].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.max_plays, len(idx))
    pick = idx.iloc[np.sort(rng.choice(len(idx), n, replace=False))].reset_index(drop=True)
    n_dup = int(pick.duplicated(["gameId", "playId"]).sum())
    pick = pick.drop_duplicates(["gameId", "playId"], keep="first").reset_index(drop=True)
    pick["file"] = [f"{r.season}_{r.team}_{r.gameId}_{r.playId}.tsv" for r in pick.itertuples()]
    return pick, n_dup


def fetch_play(name: str, cache_dir: Path) -> Path | None:
    """Download one play file unless cached; returns the local path or ``None`` on failure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / name
    if p.exists() and p.stat().st_size > 0:
        return p
    verify: str | bool = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or True
    try:
        r = requests.get(RAW_URL.format(name=name), timeout=120, verify=verify)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001 - one failed play must not stop the check
        print(f"fetch failed for {name}: {e}")
        return None
    p.write_bytes(r.content)
    return p


def read_play(path: Path) -> pd.DataFrame:
    """One highlight TSV with :data:`HIGHLIGHT_COLS` and whichever :data:`OPTIONAL_COLS` it carries."""
    head = pd.read_csv(path, sep="\t", nrows=0).columns
    use = HIGHLIGHT_COLS + [c for c in OPTIONAL_COLS if c in head]
    return pd.read_csv(path, sep="\t", usecols=use)


def load_highlight_targets(pick: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recomputed targets per fetched play plus the counts of plays dropped or flagged and why.

    Returns:
        ``(targets, counts)`` with ``counts`` keys ``fetched``, ``no_snap_tag``, ``not_11v11``
        (after de-duplication), ``duplicated_snap_rows`` (snap frame listed players twice; kept
        after de-duplication), ``multiple_snap_frames`` (more than one ``ball_snap`` frame in the
        file; kept with the deterministic choice) and ``ball_row_fallback`` (no snap frame with a
        plausible ball row; kept with the :func:`lateral_reference` fallback).
    """
    rows = []
    counts = {"fetched": 0, "no_snap_tag": 0, "not_11v11": 0, "duplicated_snap_rows": 0, "multiple_snap_frames": 0,
              "ball_row_fallback": 0}
    for r in pick.itertuples():
        p = fetch_play(r.file, cache_dir)
        if p is None:
            continue
        counts["fetched"] += 1
        play = read_play(p)
        t = highlight_targets(play)
        if t is None:
            if snap_frame_rows(play) is None:
                counts["no_snap_tag"] += 1
            else:
                counts["not_11v11"] += 1
            continue
        if t["n_rows_snap_raw"] > t["n_def"] + t["n_off"] + 1:
            counts["duplicated_snap_rows"] += 1
        if t["n_snap_frames"] > 1:
            counts["multiple_snap_frames"] += 1
        if not t["ball_valid"]:
            counts["ball_row_fallback"] += 1
        season_type = str(play["seasonType"].iloc[0]) if "seasonType" in play.columns else "unknown"
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.team, "gameId": int(r.gameId),
                     "playId": int(r.playId), "playDesc": r.playDesc, "seasonType": season_type, **t})
    return pd.DataFrame(rows), counts


def impute_seasons(cfg: OODConfig) -> pd.DataFrame:
    parts = []
    positions = aps.load_player_positions()
    for season in cfg.seasons:
        pbp = aps.load_pbp_season(season)
        prepared = aps.prepare_pbp(pbp, aps.load_participation_season(season), positions)
        parts.append(aps.impute(prepared, cfg.feature_sets))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_rows(y: np.ndarray, p: np.ndarray, kind: str, label: dict[str, Any]) -> dict[str, Any]:
    return {**label, **imp.score(kind, y, p)}


HL_SAMPLE = "highlights 2018-19"
HL_SAMPLE_VALID = "highlights 2018-19, ball row valid"
REF_ALL = "2017 OOF, all plays"
REF_LIKE = "2017 OOF, TD or >= 20 yd"


def ood_metrics(hl: pd.DataFrame, oof: pd.DataFrame, plays: pd.DataFrame, cfg: OODConfig) -> pd.DataFrame:
    """Students vs recomputed targets on the highlight plays (with bootstrap skill CIs), next to the 2017 OOF references.

    The highlight sample is scored twice: all joined plays and the plays whose chosen snap frame
    had a plausible ball row (the lateral fallback was not needed).
    """
    like = (plays["pbp_touchdown"].fillna(0) == 1) | (plays["pbp_yards_gained"].fillna(0) >= 20)
    ref = plays[["gameId", "playId"]].merge(oof, on=["gameId", "playId"], how="left")
    valid = hl["ball_valid"].to_numpy(dtype=bool)
    samples = [(HL_SAMPLE, np.ones(len(hl), dtype=bool)), (HL_SAMPLE_VALID, valid)]
    rows = []
    for tgt in OOD_TARGETS:
        spec = imf.TARGET_BY_NAME[tgt]
        for sname, m in samples:
            y_hl = hl[tgt].to_numpy(dtype=float)[m]
            rows.append(score_rows(y_hl, np.full(len(y_hl), float(oof[f"y_{tgt}"].mean())), spec.kind,
                                   {"target": tgt, "sample": sname, "model": "2017 mean / base rate"}))
            for fset in cfg.feature_sets:
                p_hl = hl[f"imp_{tgt}__{fset}"].to_numpy(dtype=float)[m]
                lo, hi = skill_bootstrap_ci(spec.kind, y_hl, p_hl, n_boot=cfg.n_boot, seed=cfg.seed)
                rows.append({**score_rows(y_hl, p_hl, spec.kind, {"target": tgt, "sample": sname, "model": fset}),
                             "skill_ci_low": lo, "skill_ci_high": hi})
        for fset in cfg.feature_sets:
            y17 = ref[f"y_{tgt}"].to_numpy(dtype=float)
            p17 = ref[f"{tgt}__{fset}"].to_numpy(dtype=float)
            rows.append(score_rows(y17, p17, spec.kind, {"target": tgt, "sample": REF_ALL, "model": fset}))
            rows.append(score_rows(y17[like.to_numpy()], p17[like.to_numpy()], spec.kind,
                                   {"target": tgt, "sample": REF_LIKE, "model": fset}))
    return pd.DataFrame(rows)


def skill_summary(metrics: pd.DataFrame, feature_sets: tuple[str, ...]) -> pd.DataFrame:
    """One row per (target, student): highlight skill with its CI next to the 2017 OOF references.

    ``skill`` = R2 (counts) or BSS (binary); ``ref_all_inside_ci`` / ``ref_like_inside_ci`` say
    whether the 2017 out-of-fold skill (all plays / the TD-or-20-yd subset) lies inside the
    highlight-sample bootstrap interval, i.e. whether any degradation is visible at this n.
    """
    rows = []
    for tgt in OOD_TARGETS:
        kind = imf.TARGET_BY_NAME[tgt].kind
        col = "bss" if kind == "binary" else "r2"
        m = metrics[metrics["target"] == tgt].set_index(["sample", "model"])
        for fset in feature_sets:
            hl, hv = m.loc[(HL_SAMPLE, fset)], m.loc[(HL_SAMPLE_VALID, fset)]
            ref_all, ref_like = float(m.loc[(REF_ALL, fset), col]), float(m.loc[(REF_LIKE, fset), col])
            lo, hi = float(hl["skill_ci_low"]), float(hl["skill_ci_high"])
            rows.append({"target": tgt, "model": fset, "skill_metric": col, "n_highlights": int(hl["n"]),
                         "skill_highlights": float(hl[col]), "ci_low": lo, "ci_high": hi,
                         "n_ball_valid": int(hv["n"]), "skill_ball_valid": float(hv[col]),
                         "skill_2017_oof_all": ref_all, "skill_2017_oof_like": ref_like,
                         "ref_all_inside_ci": bool(lo <= ref_all <= hi), "ref_like_inside_ci": bool(lo <= ref_like <= hi),
                         "auc_highlights": float(hl["auc"]) if "auc" in hl.index else np.nan,
                         "auc_2017_oof_all": float(m.loc[(REF_ALL, fset), "auc"]) if "auc" in m.columns else np.nan})
    return pd.DataFrame(rows)


def bias_table(hl: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """How the highlight sample differs from the tracked 2017 plays."""
    rows = []
    n17 = len(plays)
    rows.append({"quantity": "plays", "highlights": float(len(hl)), "tracked_2017": float(n17)})
    rows.append({"quantity": "pass play share", "highlights": float(hl["play_type"].eq("pass").mean()),
                 "tracked_2017": float(plays["pbp_play_type"].eq("pass").mean())})
    rows.append({"quantity": "touchdown share", "highlights": float((hl["touchdown"].fillna(0) == 1).mean()),
                 "tracked_2017": float((plays["pbp_touchdown"].fillna(0) == 1).mean())})
    rows.append({"quantity": "mean yards gained", "highlights": float(hl["yards_gained"].mean()),
                 "tracked_2017": float(plays["pbp_yards_gained"].mean())})
    for tgt in OOD_TARGETS:
        rows.append({"quantity": f"mean {tgt}", "highlights": float(hl[tgt].mean()), "tracked_2017": float(plays[tgt].mean())})
    rows.append({"quantity": "median |ball x - LOS| (yd)", "highlights": float(hl["los_ball_discrepancy"].abs().median()),
                 "tracked_2017": float(plays["los_ball_discrepancy"].abs().median())})
    return pd.DataFrame(rows)


EDGE_TOL = 0.02   # a reference within this of an interval bound is "at the edge", neither inside nor clearly outside


def reference_position(ref: float, lo: float, hi: float, tol: float = EDGE_TOL) -> str:
    """Where a reference skill sits relative to a bootstrap interval: ``inside``, ``edge`` (within ``tol`` of a
    bound, on either side), ``above`` or ``below``."""
    if not (np.isfinite(ref) and np.isfinite(lo) and np.isfinite(hi)):
        return "unknown"
    if abs(ref - hi) <= tol or abs(ref - lo) <= tol:
        return "edge"
    if lo < ref < hi:
        return "inside"
    return "above" if ref > hi else "below"


def _reading_lines(summary: pd.DataFrame) -> list[str]:
    L = []
    for r in summary.itertuples():
        pos = reference_position(r.skill_2017_oof_all, r.ci_low, r.ci_high)
        verdict = {"inside": "2017 skill inside the highlight interval: no visible degradation at this n",
                   "edge": "2017 skill at the edge of the highlight interval: a mild degradation can neither be shown nor excluded",
                   "above": "2017 all-plays skill above the highlight interval: degradation visible",
                   "below": "highlight skill above the 2017 all-plays value", "unknown": "no interval"}[pos]
        L.append(f"* `{r.target}` / {r.model}: highlight {r.skill_metric.upper()} {r.skill_highlights:.3f} "
                 f"[{r.ci_low:.3f}, {r.ci_high:.3f}] (n={r.n_highlights}; ball-row-valid plays only {r.skill_ball_valid:.3f}, "
                 f"n={r.n_ball_valid}) vs 2017 OOF {r.skill_2017_oof_all:.3f} on all plays and {r.skill_2017_oof_like:.3f} on the "
                 f"TD-or-20-yd subset: {verdict}.")
    return L


def write_report(res: dict[str, Any], cfg: OODConfig, path: Path) -> None:
    c = res["counts"]
    hl = res["plays"]
    fallback = hl[~hl["ball_valid"]]
    fb_txt = ("; ".join(f"{int(r.gameId)}/{int(r.playId)} ({r.lateral_ref}, box {int(r.box_count_tuned)}, raw ball x - LOS "
                        f"{r.los_ball_discrepancy:+.1f} yd, |ball y - reference| {r.lateral_ref_shift:.1f} yd)"
                        for r in fallback.itertuples()) or "none")
    multi = hl[hl["n_snap_frames"] > 1]
    L = ["# NFL 03b - out-of-distribution check on NGS highlight plays (2018-19)\n",
         f"Generated by `python -m research.privileged_tracking.nfl.ood_highlights --max-plays {cfg.max_plays}` in "
         f"{res['seconds'] / 60:.1f} min. {res['n_sampled'] + res['n_duplicate_index']} index rows were sampled with seed {cfg.seed} from the "
         f"2018-19 rows of `ngs_highlights/index.tsv` ({res['n_duplicate_index']} dropped because the index lists the same (gameId, playId) "
         f"twice, e.g. under both teams; {res['n_sampled']} distinct plays), {res['n_fetched']} fetched, {res['n_targets']} had a `ball_snap` "
         "tag and exactly 11 v 11 "
         f"at the chosen snap frame ({c['no_snap_tag']} dropped for no tag, {c['not_11v11']} for not 11 v 11). Of those, "
         f"{c['not_regular_season']} are postseason / Pro Bowl plays that cannot join the regular-season imputation "
         f"(`apply_student.load_pbp_season` loads REG only) and {c['unjoined_regular_season']} regular-season play(s) did not join "
         f"nflfastR by (gameId, playId); {res['n_joined'] + c['not_scrimmage']} joined and {res['n_joined']} are scrimmage plays "
         f"({c['not_scrimmage']} punt / kick returns dropped because the students only apply to scrimmage plays). The sample is "
         "therefore regular-season only.\n",
         "**Snap frame and ball row.** The snap frame is the earliest `ball_snap` frame whose ball row is plausible (inside the "
         f"field and within {BALL_LOS_TOL} yd of `absoluteYardlineNumber` along the attacking direction); its rows are "
         f"de-duplicated on `nflId` / `displayName` ({c['duplicated_snap_rows']} play(s) listed every player twice on the snap frame "
         f"because it carries two event tags). {c['multiple_snap_frames']} of the {res['n_joined']} plays carry more than one "
         "`ball_snap` frame (the file holds the play twice, or a mid-play frame of the first copy is tagged as the second copy's "
         f"snap); the earliest plausible frame was used ({', '.join(f'{int(r.gameId)}/{int(r.playId)}: frame {int(r.snap_frame)} of {int(r.n_snap_frames)}' for r in multi.itertuples()) or 'none'}). "
         f"{c['ball_row_fallback']} play(s) have no snap frame with a plausible ball row; for those the lateral reference of the box "
         f"is the centre's y (or the offensive line's median y) instead of the ball position: {fb_txt} (a shift below ~0.5 yd means a "
         "shotgun snap already travelling back at the tagged frame, where the fallback is immaterial; a shift of tens of yards is a "
         "glitched ball row). Targets were "
         "recomputed from the highlight frames with the NFL 01 definitions (deep safety = defender depth >= 10 yd from "
         "`absoluteYardlineNumber`; box = -1 <= depth <= 6 and |y - lateral reference| <= 7); the students are the final F0 / F0P / F1 "
         "bundles applied through `apply_student` to the full 2018 and 2019 regular seasons (tendencies from those seasons, 2017 "
         "team encodings).\n",
         "**Bias.** Highlight plays are touchdowns and long gains, so their pre-snap state is not a random draw: the table "
         "below shows the sample against the tracked 2017 plays, and the metrics table adds the 2017 out-of-fold numbers on "
         "the comparable 'touchdown or >= 20 yards' subset so that sample bias and season shift can be told apart.\n",
         "## 1. Sample vs tracked 2017 plays\n", md_table(res["bias"], floatfmt="{:.3f}"), "",
         "## 2. Students on the highlight plays (skill = R2 / BSS with a play-level bootstrap 95% CI; counts also rounded exact / within-1)\n",
         md_table(res["metrics"][[col for col in ("target", "sample", "model", "n", "r2", "skill_ci_low", "skill_ci_high", "mae", "bias",
                                                 "exact_rounded", "within_1", "auc", "log_loss", "bss", "acc", "base_rate")
                                  if col in res["metrics"].columns]], floatfmt="{:.3f}"), "",
         "## 3. Reading\n",
         f"Highlight-sample skill (n={res['n_joined']}, {cfg.n_boot} bootstrap resamples of plays, seed {cfg.seed}) against the 2017 "
         "out-of-fold skill of the same students; `ref_all_inside_ci` in `nfl_03_ood_highlights_summary.parquet` is the test used.\n"]
    L += _reading_lines(res["summary"])
    L += ["",
          "**Conclusion.** " + res["conclusion"] + "\n",
          "## 4. Caveats\n",
          f"* n = {res['n_joined']} plays, so the intervals on an R2 of ~0.4 are about +-0.15 wide; this is a sanity check for gross "
          "failure out of distribution, not a precise estimate of season-to-season degradation.",
          "* The highlight files are a different export (10 Hz, `absoluteYardlineNumber` instead of the BDB yard-line fields, "
          "`possessionFlag` for the side); only the two pre-snap counts were recomputed. Where the ball row is implausible the box "
          "lateral reference is the centre's / line's y; the `ball row valid` rows show the result without those plays.",
          "* Postseason and Pro Bowl highlight plays are excluded (regular-season play-by-play only is imputed), and plays with two "
          "`ball_snap` frames use the earliest plausible one; the alternative frame is usually a mid-play or post-play frame of a "
          "duplicated copy and would give different counts.",
          "* The F1 students see the outcome of the play (yards gained, touchdown, air yards), which on this sample is always "
          "extreme; their numbers here say how the after-the-fact students behave on tail plays, not how they behave on average.",
          "* History: an earlier version of this check took the first `ball_snap` frame in file order and the raw ball row as the "
          "lateral reference. One play (2018102200/3269) has the ball row off the field (y = 55.5) at both snap frames, which gave "
          "a recomputed box count of 0 against an F0P prediction of 7.9 and pulled the box_count_tuned R2 down to F0 0.21 / F0P 0.31 "
          "/ F1 0.31; the 'box recovery degrades out of distribution' conclusion drawn from that was an artifact of the corrupted "
          "row and is withdrawn."]
    path.write_text("\n".join(L) + "\n")


def conclusion_text(summary: pd.DataFrame) -> str:
    """One-paragraph conclusion computed from the summary table (which references fall inside the intervals)."""
    box = summary[summary["target"] == "box_count_tuned"]
    deep = summary[summary["target"] == "n_deep_safeties"]
    mof = summary[summary["target"] == "mof_open"]

    def fmt(df: pd.DataFrame) -> str:
        return ", ".join(f"{r.model} {r.skill_highlights:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] vs 2017 {r.skill_2017_oof_all:.2f}"
                         for r in df.itertuples())

    def verdict(df: pd.DataFrame) -> str:
        pos = {r.model: reference_position(r.skill_2017_oof_all, r.ci_low, r.ci_high) for r in df.itertuples()}
        inside = [m for m, v in pos.items() if v in ("inside", "below")]
        edge = [m for m, v in pos.items() if v == "edge"]
        above = [m for m, v in pos.items() if v == "above"]
        bits = []
        if inside:
            bits.append(f"2017 skill inside the interval for {', '.join(inside)}: no visible degradation")
        if edge:
            bits.append(f"at the edge of the interval for {', '.join(edge)}: a mild degradation can neither be shown nor excluded")
        if above:
            bits.append(f"above the interval for {', '.join(above)}: degradation visible")
        return " (" + "; ".join(bits) + ")"
    parts = [f"box_count_tuned R2 on the highlight plays: {fmt(box)}" + verdict(box),
             f"n_deep_safeties R2: {fmt(deep)}" + verdict(deep),
             f"mof_open BSS: {fmt(mof)}; AUC " + ", ".join(f"{r.model} {r.auc_highlights:.2f} vs 2017 {r.auc_2017_oof_all:.2f}" for r in mof.itertuples())
             + " (a rarer event on this sample, base rate in the table; the BSS intervals are wide)"]
    counts = pd.concat([box, deep])
    pos_all = [reference_position(r.skill_2017_oof_all, r.ci_low, r.ci_high) for r in counts.itertuples()]
    n_above = sum(v == "above" for v in pos_all)
    n_edge = sum(v == "edge" for v in pos_all)
    closing = (f"At n~125 none of the {len(pos_all)} count students (box / deep safeties x F0 / F0P / F1) shows a visible degradation out of "
               f"distribution ({n_edge} sit at the edge of their interval)." if n_above == 0 else
               f"At n~125 {n_above} of the {len(pos_all)} count students show a visible degradation out of distribution.")
    return "; ".join(parts) + ". " + closing


def run(cfg: OODConfig) -> dict[str, Any]:
    t0 = time.time()
    pick, n_dup = sample_index(cfg)
    cache = nfl_dir() / "ngs_highlights" / "play_data"
    targets, counts = load_highlight_targets(pick, cache)
    n_fetched = counts["fetched"]
    imputed = impute_seasons(cfg)
    imputed["gameId"] = imputed["old_game_id"].astype(int)
    imputed["playId"] = imputed["play_id"].astype(int)
    keep = ["gameId", "playId", "play_type", "yards_gained", "posteam", "defteam"] + [c for c in imputed.columns if c.startswith("imp_")]
    td = pd.concat([pd.read_parquet(nfl_dir() / "nflverse" / f"play_by_play_{s}.parquet",
                                    columns=["old_game_id", "play_id", "touchdown"]) for s in cfg.seasons], ignore_index=True)
    td["gameId"] = td["old_game_id"].astype(int)
    td["playId"] = td["play_id"].astype(int)
    hl = (targets.merge(imputed[keep], on=["gameId", "playId"], how="inner")
          .merge(td[["gameId", "playId", "touchdown"]], on=["gameId", "playId"], how="left"))
    counts["not_regular_season"] = int((targets["seasonType"] != "REG").sum())
    counts["unjoined_regular_season"] = int(len(targets)) - counts["not_regular_season"] - int(len(hl))
    # highlight plays that are not scrimmage plays in nflfastR (punt / kick returns) get no student prediction
    scrim = hl["play_type"].astype(object).isin(imf.SCRIMMAGE_PLAY_TYPES)
    counts["not_scrimmage"] = int((~scrim).sum())
    hl = hl[scrim].reset_index(drop=True)
    oof = pd.read_parquet(processed_dir("nfl") / "imputed_oof.parquet")
    plays = pd.read_parquet(processed_dir("nfl") / "plays_tracked.parquet",
                            columns=["gameId", "playId", "pbp_play_type", "pbp_touchdown", "pbp_yards_gained", "los_ball_discrepancy"] + list(OOD_TARGETS))
    metrics = ood_metrics(hl, oof, plays, cfg)
    summary = skill_summary(metrics, cfg.feature_sets)
    bias = bias_table(hl, plays)
    res = {"plays": hl, "metrics": metrics, "summary": summary, "bias": bias, "n_sampled": int(len(pick)), "n_duplicate_index": n_dup,
           "n_fetched": n_fetched,
           "n_targets": int(len(targets)), "n_joined": int(len(hl)), "counts": counts, "seconds": time.time() - t0}
    res["conclusion"] = conclusion_text(summary)
    if cfg.write:
        hl.to_parquet(processed_dir("nfl") / "ood_highlights_plays.parquet", index=False)
        metrics.to_parquet(reports_dir() / "nfl_03_ood_highlights.parquet", index=False)
        summary.to_parquet(reports_dir() / "nfl_03_ood_highlights_summary.parquet", index=False)
        bias.to_parquet(reports_dir() / "nfl_03_ood_highlights_bias.parquet", index=False)
        write_report(res, cfg, reports_dir() / "nfl_03_ood_highlights.md")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-plays", type=int, default=150)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    res = run(OODConfig(max_plays=args.max_plays, write=not args.no_write))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 100):
        print(res["bias"])
        print(res["summary"])
    print(res["counts"])
    print(res["conclusion"])
    print(f"done in {res['seconds'] / 60:.1f} min ({res['n_joined']} highlight plays)")


if __name__ == "__main__":
    main()
