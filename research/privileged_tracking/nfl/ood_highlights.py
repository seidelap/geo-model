"""NFL 03b - out-of-distribution check of the pre-snap students on NGS highlight plays.

The `asonty/ngs_highlights` repository holds tracking for 567 Next Gen Stats highlight plays of
2017-2019 (touchdowns and long gains: a strongly biased sample of plays, the only public NFL
tracking outside the Big Data Bowl). This script

1. draws ``max_plays`` plays with a fixed seed from ``nfl_dir()/ngs_highlights/index.tsv``
   (2018 and 2019 rows), fetching each play's TSV from the raw GitHub URL into
   ``nfl_dir()/ngs_highlights/play_data/`` (skipped when cached);
2. recomputes two pre-snap targets from the highlight frames with the NFL 01 definitions
   (:func:`highlight_targets`): ``n_deep_safeties`` (defender depth >= 10 yd at the snap,
   ``mof_open`` = at least two) and ``box_count_tuned`` (-1 <= depth <= 6, |lateral| <= 7);
   the snap frame is de-duplicated on ``nflId`` / ``displayName`` first (a frame carrying two
   event tags lists every player twice) and a play is kept only when it is exactly 11 v 11;
3. applies the saved F0 / F0P / F1 students (``apply_student``) to the 2018 and 2019 nflfastR
   seasons and scores them on the highlight plays, next to the 2017 out-of-fold numbers on all
   tracked plays and on the "highlight-like" 2017 subset (touchdown or >= 20 yards gained).

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


@dataclass
class OODConfig:
    """Driver configuration.

    Attributes:
        max_plays: highlight plays sampled (seeded) from the 2018-19 index.
        seed: sampling seed.
        seasons: seasons of the index to draw from.
        feature_sets: student bundles applied.
        write: write the parquet / markdown outputs.
    """

    max_plays: int = 150
    seed: int = 0
    seasons: tuple[int, ...] = (2018, 2019)
    feature_sets: tuple[str, ...] = ("F0", "F0P", "F1")
    write: bool = True


# ---------------------------------------------------------------------------
# Highlight frames -> targets (pure)
# ---------------------------------------------------------------------------

PLAYERS_PER_SIDE = 11
DEDUP_KEYS = ("nflId", "displayName")


def snap_frame_rows(play: pd.DataFrame) -> pd.DataFrame | None:
    """Rows of the first ``ball_snap`` frame with duplicated player rows dropped.

    A frame that carries two event tags (e.g. ``ball_snap`` and ``man_in_motion``) lists
    every player - and the ball - twice in the highlight export; rows are de-duplicated on
    :data:`DEDUP_KEYS` (those present; ``teamAbbr, x, y`` when neither is) so each player
    counts once. Returns ``None`` when the play has no ``ball_snap`` tag.
    """
    snap_frames = play.loc[play["event"].astype(object).eq("ball_snap"), "frame"].unique()
    if len(snap_frames) == 0:
        return None
    s = play[play["frame"] == snap_frames[0]]
    keys = [c for c in DEDUP_KEYS if c in s.columns] or ["teamAbbr", "x", "y"]
    return s.drop_duplicates(subset=keys)


def highlight_targets(play: pd.DataFrame) -> dict[str, Any] | None:
    """Pre-snap targets from one highlight play's frames (NFL 01 definitions).

    Args:
        play: rows of one play with ``frame, event, x, y, teamAbbr, possessionFlag,
            playDirection, absoluteYardlineNumber`` (the ball has ``teamAbbr`` NaN) and,
            when available, ``nflId`` / ``displayName`` for de-duplication.

    Returns:
        ``None`` when the play has no ``ball_snap`` tag or is not exactly 11 v 11 at the snap
        after de-duplication (:func:`snap_frame_rows`); otherwise a dict with
        ``n_deep_safeties``, ``mof_open``, ``box_count_tuned``, ``n_def``, ``n_off``,
        ``los_x``, ``los_ball_discrepancy`` (ball x - LOS in the attacking direction),
        ``play_direction``, ``snap_frame`` and ``n_rows_snap_raw`` (rows before de-duplication).
    """
    s = snap_frame_rows(play)
    if s is None:
        return None
    n_raw = int((play["frame"] == s["frame"].iloc[0]).sum())
    is_ball = s["teamAbbr"].isna().to_numpy()
    poss = pd.to_numeric(s["possessionFlag"], errors="coerce").to_numpy(dtype=float)
    off = (~is_ball) & (poss == 1)
    de = (~is_ball) & (poss == 0)
    if de.sum() != PLAYERS_PER_SIDE or off.sum() != PLAYERS_PER_SIDE:
        return None
    direction = str(s["playDirection"].iloc[0])
    sign = 1.0 if direction == "right" else -1.0     # offense attacks +x when "right"
    los_x = float(pd.to_numeric(s["absoluteYardlineNumber"], errors="coerce").iloc[0])
    x = s["x"].to_numpy(dtype=float)
    y = s["y"].to_numpy(dtype=float)
    depth = sign * (x - los_x)                        # defenders positive
    if is_ball.any():
        ball_y = float(y[is_ball][0])
        ball_x = float(x[is_ball][0])
    else:
        ball_y = float(np.median(y[off]))
        ball_x = float("nan")
    lateral = y - ball_y
    dd = depth[de]
    return {"n_deep_safeties": int(np.sum(dd >= DEEP_DEPTH)), "mof_open": int(np.sum(dd >= DEEP_DEPTH) >= 2),
            "box_count_tuned": tf.box_count(depth, lateral, de, BOX_DEPTH, BOX_LATERAL),
            "n_def": int(de.sum()), "n_off": int(off.sum()), "los_x": los_x,
            "los_ball_discrepancy": sign * (ball_x - los_x), "play_direction": direction,
            "snap_frame": int(s["frame"].iloc[0]), "n_rows_snap_raw": n_raw}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sample_index(cfg: OODConfig) -> pd.DataFrame:
    idx = pd.read_csv(nfl_dir() / "ngs_highlights" / "index.tsv", sep="\t")
    idx = idx[idx["season"].isin(cfg.seasons)].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.max_plays, len(idx))
    pick = idx.iloc[np.sort(rng.choice(len(idx), n, replace=False))].reset_index(drop=True)
    pick["file"] = [f"{r.season}_{r.team}_{r.gameId}_{r.playId}.tsv" for r in pick.itertuples()]
    return pick


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


def load_highlight_targets(pick: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recomputed targets per fetched play plus the counts of plays dropped and why.

    Returns:
        ``(targets, counts)`` with ``counts`` keys ``fetched``, ``no_snap_tag``,
        ``not_11v11`` (after de-duplication) and ``duplicated_snap_rows`` (plays whose snap
        frame listed players twice; kept after de-duplication).
    """
    rows = []
    counts = {"fetched": 0, "no_snap_tag": 0, "not_11v11": 0, "duplicated_snap_rows": 0}
    for r in pick.itertuples():
        p = fetch_play(r.file, cache_dir)
        if p is None:
            continue
        counts["fetched"] += 1
        play = pd.read_csv(p, sep="\t", usecols=["frame", "event", "x", "y", "teamAbbr", "possessionFlag",
                                                 "playDirection", "absoluteYardlineNumber", "nflId", "displayName"])
        t = highlight_targets(play)
        if t is None:
            if snap_frame_rows(play) is None:
                counts["no_snap_tag"] += 1
            else:
                counts["not_11v11"] += 1
            continue
        if t["n_rows_snap_raw"] > t["n_def"] + t["n_off"] + 1:
            counts["duplicated_snap_rows"] += 1
        rows.append({"season": int(r.season), "week": int(r.week), "team": r.team, "gameId": int(r.gameId),
                     "playId": int(r.playId), "playDesc": r.playDesc, **t})
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


def ood_metrics(hl: pd.DataFrame, oof: pd.DataFrame, plays: pd.DataFrame, cfg: OODConfig) -> pd.DataFrame:
    """Students vs recomputed targets on the highlight plays, next to the 2017 OOF references."""
    like = (plays["pbp_touchdown"].fillna(0) == 1) | (plays["pbp_yards_gained"].fillna(0) >= 20)
    ref = plays[["gameId", "playId"]].merge(oof, on=["gameId", "playId"], how="left")
    rows = []
    for tgt in OOD_TARGETS:
        spec = imf.TARGET_BY_NAME[tgt]
        y_hl = hl[tgt].to_numpy(dtype=float)
        rows.append(score_rows(y_hl, np.full(len(y_hl), float(oof[f"y_{tgt}"].mean())), spec.kind,
                               {"target": tgt, "sample": "highlights 2018-19", "model": "2017 mean / base rate"}))
        for fset in cfg.feature_sets:
            rows.append(score_rows(y_hl, hl[f"imp_{tgt}__{fset}"].to_numpy(dtype=float), spec.kind,
                                   {"target": tgt, "sample": "highlights 2018-19", "model": fset}))
            y17 = ref[f"y_{tgt}"].to_numpy(dtype=float)
            p17 = ref[f"{tgt}__{fset}"].to_numpy(dtype=float)
            rows.append(score_rows(y17, p17, spec.kind, {"target": tgt, "sample": "2017 OOF, all plays", "model": fset}))
            rows.append(score_rows(y17[like.to_numpy()], p17[like.to_numpy()], spec.kind,
                                   {"target": tgt, "sample": "2017 OOF, TD or >= 20 yd", "model": fset}))
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


def write_report(res: dict[str, Any], cfg: OODConfig, path: Path) -> None:
    L = ["# NFL 03b - out-of-distribution check on NGS highlight plays (2018-19)\n",
         f"Generated by `python -m research.privileged_tracking.nfl.ood_highlights --max-plays {cfg.max_plays}` in "
         f"{res['seconds'] / 60:.1f} min. {res['n_sampled']} plays were sampled with seed {cfg.seed} from the 2018-19 rows of "
         f"`ngs_highlights/index.tsv`, {res['n_fetched']} fetched, {res['n_targets']} had a `ball_snap` tag and exactly 11 v 11 "
         f"at the snap ({res['counts']['no_snap_tag']} dropped for no tag, {res['counts']['not_11v11']} for not 11 v 11; "
         f"{res['counts']['duplicated_snap_rows']} play(s) listed every player twice on the snap frame because it carries two "
         f"event tags and were de-duplicated on `nflId` / `displayName` first), {res['n_joined']} joined to nflfastR / "
         f"participation by (gameId, playId) and are scrimmage plays ({res['counts']['not_scrimmage']} punt / kick returns "
         "dropped because the students only apply to scrimmage plays). Targets were recomputed from the "
         "highlight frames with the NFL 01 definitions (deep safety = defender depth >= 10 yd from `absoluteYardlineNumber`; "
         "box = -1 <= depth <= 6 and |y - ball y| <= 7); the students are the final F0 / F0P / F1 bundles applied through "
         "`apply_student` to the full 2018 and 2019 seasons (tendencies from those seasons, 2017 team encodings).\n",
         "**Bias.** Highlight plays are touchdowns and long gains, so their pre-snap state is not a random draw: the table "
         "below shows the sample against the tracked 2017 plays, and the metrics table adds the 2017 out-of-fold numbers on "
         "the comparable 'touchdown or >= 20 yards' subset so that sample bias and season shift can be told apart.\n",
         "## 1. Sample vs tracked 2017 plays\n", md_table(res["bias"], floatfmt="{:.3f}"), "",
         "## 2. Students on the highlight plays (skill = R2 / BSS; counts also rounded exact / within-1)\n",
         md_table(res["metrics"][[c for c in ("target", "sample", "model", "n", "r2", "mae", "bias", "exact_rounded", "within_1",
                                               "auc", "log_loss", "bss", "acc", "base_rate") if c in res["metrics"].columns]],
                  floatfmt="{:.3f}"), "",
         "## 3. Caveats\n",
         f"* n = {res['n_joined']} plays, so a 95% interval on an R2 of ~0.4 is roughly +-0.15; treat the comparison as a sanity check.\n"
         "* The highlight files are a different export (10 Hz, `absoluteYardlineNumber` instead of the BDB yard-line fields, "
         "`possessionFlag` for the side); only the two pre-snap counts were recomputed, with the ball's y as the lateral reference "
         "and no centre-based fallback.\n"
         "* The F1 students see the outcome of the play (yards gained, touchdown, air yards), which on this sample is always "
         "extreme; their numbers here say how the after-the-fact students behave on tail plays, not how they behave on average.\n"]
    path.write_text("\n".join(L))


def run(cfg: OODConfig) -> dict[str, Any]:
    t0 = time.time()
    pick = sample_index(cfg)
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
    # highlight plays that are not scrimmage plays in nflfastR (punt / kick returns) get no student prediction
    scrim = hl["play_type"].astype(object).isin(imf.SCRIMMAGE_PLAY_TYPES)
    counts["not_scrimmage"] = int((~scrim).sum())
    hl = hl[scrim].reset_index(drop=True)
    oof = pd.read_parquet(processed_dir("nfl") / "imputed_oof.parquet")
    plays = pd.read_parquet(processed_dir("nfl") / "plays_tracked.parquet",
                            columns=["gameId", "playId", "pbp_play_type", "pbp_touchdown", "pbp_yards_gained", "los_ball_discrepancy"] + list(OOD_TARGETS))
    metrics = ood_metrics(hl, oof, plays, cfg)
    bias = bias_table(hl, plays)
    res = {"plays": hl, "metrics": metrics, "bias": bias, "n_sampled": int(len(pick)), "n_fetched": n_fetched,
           "n_targets": int(len(targets)), "n_joined": int(len(hl)), "counts": counts, "seconds": time.time() - t0}
    if cfg.write:
        hl.to_parquet(processed_dir("nfl") / "ood_highlights_plays.parquet", index=False)
        metrics.to_parquet(reports_dir() / "nfl_03_ood_highlights.parquet", index=False)
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
        print(res["metrics"])
    print(f"done in {res['seconds'] / 60:.1f} min ({res['n_joined']} highlight plays)")


if __name__ == "__main__":
    main()
