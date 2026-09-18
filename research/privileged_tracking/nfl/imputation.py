"""NFL 03 - imputation students: predict tracking-derived state from play-by-play.

Reads ``processed_dir('nfl')/plays_tracked.parquet`` (NFL 01) and trains one LightGBM
student per (target, feature set) that predicts a tracking-derived quantity from
information available in ordinary nflfastR play-by-play. Feature sets are nested
(``imputation_features.FEATURE_SETS``): ``F0n`` situation + strictly-prior team tendencies,
``F0`` + team target encodings from training-fold games, ``F0P`` + personnel / formation
charting, ``F1`` + post-play fields (the after-the-fact analytics setting), ``F2`` + the
official ``defenders_in_box`` / ``number_of_pass_rushers`` (reference).

Protocol: 5-fold group k-fold by game (out-of-fold predictions for every tracked play) and
a forward split (weeks 1-4 train, 5-6 test). Rounds are chosen by early stopping on an
inner 20% game holdout of the training fold, never on the test fold. Team target encodings
are leave-one-game-out inside the training fold. Baselines: training global mean / base
rate, per-play_type mean (post-snap information, i.e. an F1-level baseline) and
per-down-distance-bucket mean. Paired bootstrap deltas are reported play-level and
game-clustered.

Outputs: ``processed_dir('nfl')/imputed_oof.parquet`` (out-of-fold predictions, keyed by
gameId / playId, columns ``<target>__<fset>`` plus ``y_<target>``),
``imputed_forward.parquet`` (weeks 5-6 predictions of the forward split), the final students
under ``processed_dir('nfl')/models/students_<fset>.joblib`` (``apply_student.StudentBundle``),
``reports/nfl_03_imputation.md`` and ``reports/nfl_03_*.parquet``.

Run from the repo root::

    python -m research.privileged_tracking.nfl.imputation            # full run (~4 min incl. the 2018 application, 2 threads)
    python -m research.privileged_tracking.nfl.imputation --quick    # 3 targets, nothing written
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (
    calibration_table,
    clustered_bootstrap_delta,
    paired_bootstrap_delta,
    per_sample_log_loss,
)
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.nfl import apply_student as aps
from research.privileged_tracking.nfl import imputation_features as imf
from research.privileged_tracking.nfl import participation_features as pf

BASELINES = ["base_global", "base_play_type", "base_dd", "base_outcome"]
DELTA_PAIRS = [("base_global", "F0n"), ("F0n", "F0"), ("F0", "F0P"), ("F0P", "F1"), ("F1", "F2"),
               ("base_global", "F1"), ("base_dd", "F0"), ("base_outcome", "F1"), ("base_outcome", "F2")]
PRESSURE_GRID = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)
TUNE_WEEKS = (1, 2, 3)

#: Within-play targets whose F1 skill is decomposed by the nflfastR outcome class the F1
#: student sees (``grouped_errors`` tables + the ``outcome_decomposition`` summary).
OUTCOME_GROUPINGS: dict[str, str] = {
    "time_to_throw": "pass_qb_outcome", "min_def_dist_qb_throw": "pass_qb_outcome",
    "pressure_derived": "pass_qb_outcome", "n_pass_rushers_derived": "pass_qb_outcome",
    "separation_at_arrival": "pass_ball_outcome", "n_def_within_r_target": "pass_ball_outcome",
    "target_depth": "pass_ball_outcome",
    "yards_to_first_contact": "run_yards_bucket", "n_def_within_r_carrier_first_contact": "run_yards_bucket",
    "min_def_dist_carrier_handoff": "run_yards_bucket", "box_count_run": "run_yards_bucket",
}
GROUP_ORDER: dict[str, list[str]] = {
    "pass_qb_outcome": ["clean", "qb_hit", "sack", "na"],
    "pass_ball_outcome": ["complete", "incomplete", "interception", "sack", "na"],
    "run_yards_bucket": ["<0", "0-2", "3-5", "6-10", "10+", "na"],
    "air_yards": ["<0", "0-5", "5-10", "10-20", "20+", "na"],
    "time_to_throw": ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-4.0", "4.0+", "na"],
    "pass_location": ["left", "middle", "right", "na"],
}


@dataclass
class ImputationConfig:
    """Driver configuration.

    Attributes:
        n_jobs: LightGBM threads.
        n_folds: game-grouped folds.
        train_weeks / test_weeks: forward split.
        learning_rate / num_leaves / min_data_in_leaf / max_rounds / early_stopping: LightGBM.
        inner_holdout_frac: share of training games held out for early stopping.
        te_alpha: shrinkage (plays) of the team target encodings.
        n_boot: bootstrap resamples.
        feature_sets: nested feature sets to fit (order matters for the delta pairs).
        targets: subset of target names (``None`` = all).
        apply_season: season the final students are applied to (``None`` = skip).
        apply_feature_sets: bundles applied to that season.
        write: write parquet / report outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    n_folds: int = 5
    train_weeks: tuple[int, ...] = (1, 2, 3, 4)
    test_weeks: tuple[int, ...] = (5, 6)
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 40
    max_rounds: int = 600
    early_stopping: int = 40
    inner_holdout_frac: float = 0.2
    te_alpha: float = 30.0
    n_boot: int = 1000
    feature_sets: tuple[str, ...] = tuple(imf.FEATURE_SET_ORDER)
    targets: tuple[str, ...] | None = None
    apply_season: int | None = 2018
    apply_feature_sets: tuple[str, ...] = ("F0", "F0P", "F1")
    write: bool = True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

BDB_TO_PART = {"offenseFormation": "offense_formation", "personnel_offense": "offense_personnel",
               "personnel_defense": "defense_personnel", "defendersInTheBox": "defenders_in_box",
               "numberOfPassRushers": "number_of_pass_rushers"}


@dataclass
class TrackedData:
    """The tracked plays with their design matrix and targets (all aligned on row position)."""

    plays: pd.DataFrame
    frame: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    masks: dict[str, pd.Series]
    game: np.ndarray
    week: np.ndarray
    posteam: np.ndarray
    defteam: np.ndarray
    play_type: np.ndarray
    dd_bucket: np.ndarray
    outcome: np.ndarray                    # nflfastR outcome class per play (``outcome_bucket``)
    groupings: dict[str, np.ndarray]       # grouping name -> label per play (``OUTCOME_GROUPINGS``)


# ---------------------------------------------------------------------------
# Outcome classes (nflfastR post-play flags, i.e. what the F1 students see)
# ---------------------------------------------------------------------------

def pass_qb_outcome(sack: np.ndarray, qb_hit: np.ndarray) -> np.ndarray:
    """``sack`` / ``qb_hit`` (hit, no sack) / ``clean`` (neither) / ``na`` per play ``[n]``."""
    s = np.asarray(sack, dtype=float)
    h = np.asarray(qb_hit, dtype=float)
    out = np.full(len(s), "na", dtype=object)
    out[(s == 0) & (h == 0)] = "clean"
    out[(s == 0) & (h == 1)] = "qb_hit"
    out[s == 1] = "sack"
    return out


def pass_ball_outcome(sack: np.ndarray, complete_pass: np.ndarray, interception: np.ndarray) -> np.ndarray:
    """``sack`` / ``complete`` / ``interception`` / ``incomplete`` / ``na`` per play ``[n]``."""
    s = np.asarray(sack, dtype=float)
    c = np.asarray(complete_pass, dtype=float)
    i = np.asarray(interception, dtype=float)
    out = np.full(len(s), "na", dtype=object)
    known = ~(np.isnan(s) | np.isnan(c) | np.isnan(i))
    out[known] = "incomplete"
    out[known & (i == 1)] = "interception"
    out[known & (c == 1)] = "complete"
    out[known & (s == 1)] = "sack"
    return out


def run_yards_bucket(yards_gained: np.ndarray) -> np.ndarray:
    """nflfastR ``yards_gained`` bucket ``<0 / 0-2 / 3-5 / 6-10 / 10+ / na`` per play ``[n]``."""
    y = np.asarray(yards_gained, dtype=float)
    out = np.full(len(y), "na", dtype=object)
    out[y < 0] = "<0"
    out[(y >= 0) & (y <= 2)] = "0-2"
    out[(y > 2) & (y <= 5)] = "3-5"
    out[(y > 5) & (y <= 10)] = "6-10"
    out[y > 10] = "10+"
    return out


def outcome_bucket(play_type: np.ndarray, sack: np.ndarray, qb_hit: np.ndarray, complete_pass: np.ndarray,
                   interception: np.ndarray, yards_gained: np.ndarray) -> np.ndarray:
    """Joint nflfastR outcome class per play ``[n]`` for the ``base_outcome`` baseline.

    Pass plays: ``pass:<qb outcome>:<ball outcome>`` (sack x QB hit x completion x
    interception); run plays: ``run:<yards bucket>``; anything else its play type. It uses
    only F1 inputs, so the per-class mean is the "re-encode the outcome flags" reference
    that the F1 within-play skill must beat.
    """
    pt = np.asarray(play_type, dtype=object)
    qb = pass_qb_outcome(sack, qb_hit)
    ball = pass_ball_outcome(sack, complete_pass, interception)
    yb = run_yards_bucket(yards_gained)
    out = np.empty(len(pt), dtype=object)
    for i in range(len(pt)):
        if pt[i] == "pass":
            out[i] = f"pass:{qb[i]}:{ball[i]}"
        elif pt[i] == "run":
            out[i] = f"run:{yb[i]}"
        else:
            out[i] = str(pt[i])
    return out


def load_tracked(cfg: ImputationConfig) -> TrackedData:
    """Build the pbp-like frame, design matrix and targets for the tracked 2017 plays.

    The pbp columns come from the ``pbp_*`` copies in ``plays_tracked`` (nflfastR names
    restored), the charting columns from plays.csv (renamed to participation names), the
    receiver position from the nflverse players crosswalk and the team tendencies from
    the full 2017 nflfastR season (strictly earlier weeks, all 32 teams).
    """
    plays = pd.read_parquet(processed_dir("nfl") / "plays_tracked.parquet").reset_index(drop=True)
    frame = pd.DataFrame(index=plays.index)
    for c in imf.PBP_INPUT_COLS:
        frame[c] = plays[f"pbp_{c}"]
    frame["is_home"] = plays["offense_is_home"].astype(float)
    frame["posteam"] = plays["pbp_posteam"]
    frame["defteam"] = plays["pbp_defteam"]
    for bdb, part in BDB_TO_PART.items():
        frame[part] = plays[bdb]
    positions = aps.load_player_positions()
    frame["receiver_position"] = plays["pbp_receiver_player_id"].map(positions).astype(object).where(plays["pbp_receiver_player_id"].notna())
    pbp17 = aps.load_pbp_season(2017)
    tend = imf.pbp_team_tendencies(pbp17)
    tend = pd.concat([pbp17[["game_id", "play_id"]].reset_index(drop=True), tend.reset_index(drop=True)], axis=1)
    keys = pd.DataFrame({"game_id": plays["pbp_game_id"].astype(str), "play_id": plays["playId"].astype(float)})
    tend["play_id"] = tend["play_id"].astype(float)
    merged = keys.merge(tend, on=["game_id", "play_id"], how="left")
    assert len(merged) == len(plays)
    for c in imf.TENDENCY_COLS:
        frame[c] = merged[c].to_numpy()
    X = imf.build_design(frame)
    Y = imf.target_frame(plays)
    masks = imf.subset_masks(plays)
    play_type = plays["pbp_play_type"].astype(object).fillna("na").to_numpy(dtype=object)
    flags = {c: pd.to_numeric(plays[f"pbp_{c}"], errors="coerce").to_numpy(dtype=float)
             for c in ("sack", "qb_hit", "complete_pass", "interception", "yards_gained")}
    groupings = {"pass_qb_outcome": pass_qb_outcome(flags["sack"], flags["qb_hit"]),
                 "pass_ball_outcome": pass_ball_outcome(flags["sack"], flags["complete_pass"], flags["interception"]),
                 "run_yards_bucket": run_yards_bucket(flags["yards_gained"])}
    outcome = outcome_bucket(play_type, flags["sack"], flags["qb_hit"], flags["complete_pass"], flags["interception"],
                             flags["yards_gained"])
    return TrackedData(plays=plays, frame=frame, X=X, Y=Y, masks=masks, game=plays["gameId"].to_numpy(),
                       week=plays["week"].to_numpy(), posteam=plays["pbp_posteam"].to_numpy(dtype=object),
                       defteam=plays["pbp_defteam"].to_numpy(dtype=object), play_type=play_type,
                       dd_bucket=pf.down_distance_bucket(plays["pbp_down"].to_numpy(), plays["pbp_ydstogo"].to_numpy()),
                       outcome=outcome, groupings=groupings)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def lgb_params(kind: str, cfg: ImputationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "binary" if kind == "binary" else "regression", "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves, "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": cfg.n_jobs, "seed": cfg.seed,
        "verbose": -1, "max_bin": 63, "deterministic": True, "force_row_wise": True,
        "metric": "binary_logloss" if kind == "binary" else "l2",
    }
    return params


def fit_with_inner_holdout(X: pd.DataFrame, y: np.ndarray, games: np.ndarray, kind: str,
                           cfg: ImputationConfig) -> tuple[lgb.Booster, int]:
    """Early-stop on a game-grouped holdout of the training rows; returns the 80% model.

    Args:
        X: training design ``[n, d]``.
        y: targets ``[n]`` (no NaN).
        games: game id per row ``[n]``.
    """
    uniq = np.unique(games)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(uniq)
    n_hold = max(1, int(round(len(uniq) * cfg.inner_holdout_frac)))
    hold = np.isin(games, uniq[:n_hold])
    cats = [c for c in imf.CATEGORICAL_COLS if c in X.columns]
    dtr = lgb.Dataset(X[~hold], label=y[~hold], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(X[hold], label=y[hold], reference=dtr, categorical_feature=cats, free_raw_data=False)
    booster = lgb.train(lgb_params(kind, cfg), dtr, num_boost_round=cfg.max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    return booster, int(booster.best_iteration or cfg.max_rounds)


def refit(X: pd.DataFrame, y: np.ndarray, kind: str, cfg: ImputationConfig, rounds: int) -> lgb.Booster:
    cats = [c for c in imf.CATEGORICAL_COLS if c in X.columns]
    d = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    return lgb.train(lgb_params(kind, cfg), d, num_boost_round=max(1, rounds))


def score(kind: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Metrics with n: r2 / mae / rmse / bias (+ exact_rounded / within_1 for counts) or auc /
    log_loss / brier / bss / acc / base_rate for binary targets."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {"n": 0}
    if kind == "binary":
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        base = float(y.mean())
        br = float(np.mean((pc - y) ** 2))
        br_base = base * (1 - base)
        return {"n": int(len(y)), "auc": float(roc_auc_score(y, pc)) if len(np.unique(y)) > 1 else np.nan,
                "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), "brier": br,
                "bss": float(1 - br / br_base) if br_base > 0 else np.nan,
                "acc": float(np.mean((pc >= 0.5) == (y >= 0.5))), "base_rate": base, "mean_pred": float(pc.mean())}
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    d: dict[str, float] = {"n": int(len(y)), "r2": float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
                           "mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                           "bias": float(np.mean(p - y))}
    if kind == "count":
        d["exact_rounded"], d["within_1"] = imf.rounded_agreement(y, p)
    return d


def per_sample_loss(kind: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Absolute error (continuous / count) or log-loss (binary) per play, for paired bootstraps."""
    if kind == "binary":
        return per_sample_log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    return np.abs(np.asarray(y, float) - np.asarray(p, float))


def skill_of(kind: str, m: dict[str, float]) -> float:
    """R2 (continuous / count) or Brier skill score (binary): both are 1 - MSE / MSE(mean)."""
    return float(m.get("bss", np.nan)) if kind == "binary" else float(m.get("r2", np.nan))


# ---------------------------------------------------------------------------
# One target
# ---------------------------------------------------------------------------

@dataclass
class TargetRun:
    """Predictions and per-sample losses of one target for one split protocol."""

    target: str
    split: str
    preds: dict[str, np.ndarray]           # name -> [n_all] (NaN outside the evaluated rows)
    eval_idx: np.ndarray                   # labelled rows with a prediction
    metrics: dict[str, dict[str, float]]   # name -> metrics
    losses: dict[str, np.ndarray]          # name -> [len(eval_idx)]
    best_iters: dict[str, list[int]] = field(default_factory=dict)


def folds_for(data: TrackedData, cfg: ImputationConfig) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Positional (train, test) index pairs for the k-fold and forward protocols."""
    kf = list(group_kfold(data.game, n_splits=cfg.n_folds, seed=cfg.seed))
    tr = np.where(np.isin(data.week, cfg.train_weeks))[0]
    te = np.where(np.isin(data.week, cfg.test_weeks))[0]
    return {"kfold": kf, "forward": [(tr, te)]}


def run_target(spec: imf.TargetSpec, data: TrackedData, folds: list[tuple[np.ndarray, np.ndarray]],
               split: str, cfg: ImputationConfig) -> TargetRun:
    """Fit every feature set and baseline on each fold of ``split`` and collect predictions."""
    n = len(data.X)
    y = data.Y[spec.name].to_numpy(dtype=float)
    in_subset = data.masks[spec.subset].to_numpy()
    labelled = in_subset & ~np.isnan(y)
    names = list(cfg.feature_sets) + BASELINES
    preds = {k: np.full(n, np.nan) for k in names}
    best_iters: dict[str, list[int]] = {k: [] for k in cfg.feature_sets}
    for tr, te in folds:
        tr_l = tr[labelled[tr]]
        te_s = te[in_subset[te]]
        if len(tr_l) < 50 or len(te_s) == 0:
            continue
        enc_off = imf.TeamEncoder(cfg.te_alpha).fit(data.posteam[tr_l], data.game[tr_l], y[tr_l])
        enc_def = imf.TeamEncoder(cfg.te_alpha).fit(data.defteam[tr_l], data.game[tr_l], y[tr_l])
        Xtr = data.X.iloc[tr_l].copy()
        Xtr["te_off"] = enc_off.transform_train(data.posteam[tr_l], data.game[tr_l])
        Xtr["te_def"] = enc_def.transform_train(data.defteam[tr_l], data.game[tr_l])
        Xte = data.X.iloc[te_s].copy()
        Xte["te_off"] = enc_off.transform(data.posteam[te_s])
        Xte["te_def"] = enc_def.transform(data.defteam[te_s])
        for fset in cfg.feature_sets:
            feats = imf.FEATURE_SETS[fset]
            booster, bi = fit_with_inner_holdout(Xtr[feats], y[tr_l], data.game[tr_l], spec.kind, cfg)
            preds[fset][te_s] = booster.predict(Xte[feats], num_iteration=bi)
            best_iters[fset].append(bi)
        preds["base_global"][te_s] = float(y[tr_l].mean())
        preds["base_play_type"][te_s] = imf.bucket_mean_baseline(data.play_type[tr_l], y[tr_l], data.play_type[te_s], min_count=20)
        preds["base_dd"][te_s] = imf.bucket_mean_baseline(data.dd_bucket[tr_l], y[tr_l], data.dd_bucket[te_s], min_count=20)
        preds["base_outcome"][te_s] = imf.bucket_mean_baseline(data.outcome[tr_l], y[tr_l], data.outcome[te_s], min_count=20)
    eval_idx = np.where(labelled & ~np.isnan(preds["base_global"]))[0]
    metrics = {k: score(spec.kind, y[eval_idx], preds[k][eval_idx]) for k in names}
    losses = {k: per_sample_loss(spec.kind, y[eval_idx], preds[k][eval_idx]) for k in names}
    return TargetRun(spec.name, split, preds, eval_idx, metrics, losses, best_iters)


def fit_final(spec: imf.TargetSpec, data: TrackedData, cfg: ImputationConfig, oof: np.ndarray | None,
              fset: str) -> tuple[aps.Student, pd.DataFrame]:
    """Fit the student on all tracked games (rounds from an inner holdout, then a refit)."""
    y = data.Y[spec.name].to_numpy(dtype=float)
    idx = np.where(data.masks[spec.subset].to_numpy() & ~np.isnan(y))[0]
    enc_off = imf.TeamEncoder(cfg.te_alpha).fit(data.posteam[idx], data.game[idx], y[idx])
    enc_def = imf.TeamEncoder(cfg.te_alpha).fit(data.defteam[idx], data.game[idx], y[idx])
    X = data.X.iloc[idx].copy()
    X["te_off"] = enc_off.transform_train(data.posteam[idx], data.game[idx])
    X["te_def"] = enc_def.transform_train(data.defteam[idx], data.game[idx])
    feats = imf.FEATURE_SETS[fset]
    _, bi = fit_with_inner_holdout(X[feats], y[idx], data.game[idx], spec.kind, cfg)
    booster = refit(X[feats], y[idx], spec.kind, cfg, bi)
    gain = booster.feature_importance("gain")
    imp = pd.DataFrame({"target": spec.name, "feature_set": fset, "feature": feats, "gain": gain})
    imp["gain_share"] = imp["gain"] / max(float(gain.sum()), 1e-12)
    o = oof[idx] if oof is not None else np.array([np.nan])
    st = aps.Student(target=spec.name, kind=spec.kind, subset=spec.subset, model_str=booster.model_to_string(),
                     best_iter=bi, te_off=enc_off.table(), te_def=enc_def.table(), te_global=enc_off.global_mean,
                     train_n=int(len(idx)), y_mean=float(y[idx].mean()), y_sd=float(y[idx].std()),
                     oof_mean=float(np.nanmean(o)), oof_sd=float(np.nanstd(o)))
    return st, imp.sort_values("gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def pressure_threshold_table(plays: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Agreement of ``min_def_dist_* <= thr`` with NGS ``was_pressure``; selected on weeks 1-3."""
    p = plays[plays["is_pass_play"] & plays["ngs_was_pressure"].notna()]
    tune = p["week"].isin(TUNE_WEEKS).to_numpy()
    rows = []
    for col in ("min_def_dist_qb_throw", "min_def_dist_qb_dropback"):
        v = p[col].to_numpy(dtype=float)
        yv = p["ngs_was_pressure"].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        auc = float(roc_auc_score(yv[ok], -v[ok]))
        for thr in PRESSURE_GRID:
            flag = (v <= thr).astype(float)
            rows.append({"distance": col, "threshold_yd": thr, "auc_distance": auc,
                         "n_tune": int((tune & ok).sum()), "agree_tune": float(np.mean(flag[tune & ok] == yv[tune & ok])),
                         "n_heldout": int((~tune & ok).sum()), "agree_heldout": float(np.mean(flag[~tune & ok] == yv[~tune & ok])),
                         "flag_rate_heldout": float(flag[~tune & ok].mean()), "charted_rate_heldout": float(yv[~tune & ok].mean())})
    t = pd.DataFrame(rows)
    throw = t[t["distance"] == "min_def_dist_qb_throw"]
    selected = float(throw.sort_values("agree_tune", ascending=False).iloc[0]["threshold_yd"])
    return t, selected


def metrics_long(runs: dict[tuple[str, str], TargetRun]) -> pd.DataFrame:
    rows = []
    for (target, split), r in runs.items():
        spec = imf.TARGET_BY_NAME[target]
        for name, m in r.metrics.items():
            rows.append({"target": target, "kind": spec.kind, "subset": spec.subset, "split": split, "model": name,
                         "skill": skill_of(spec.kind, m), **m,
                         "mean_best_iter": float(np.mean(r.best_iters[name])) if name in r.best_iters and r.best_iters[name] else np.nan})
    return pd.DataFrame(rows)


def deltas_table(runs: dict[tuple[str, str], TargetRun], data: TrackedData, cfg: ImputationConfig) -> pd.DataFrame:
    rows = []
    for (target, split), r in runs.items():
        spec = imf.TARGET_BY_NAME[target]
        groups = data.game[r.eval_idx]
        for a, b in DELTA_PAIRS:
            if a not in r.losses or b not in r.losses:
                continue
            la, lb = r.losses[a], r.losses[b]
            m, lo, hi = paired_bootstrap_delta(la, lb, n_boot=cfg.n_boot, seed=cfg.seed)
            _, lo_g, hi_g = clustered_bootstrap_delta(la, lb, groups, n_boot=cfg.n_boot, seed=cfg.seed)
            rows.append({"target": target, "kind": spec.kind, "split": split, "a": a, "b": b,
                         "loss": "log_loss" if spec.kind == "binary" else "abs_error", "n": int(len(la)),
                         "games": int(len(np.unique(groups))), "delta_a_minus_b": m, "ci_low": lo, "ci_high": hi,
                         "ci_low_game": lo_g, "ci_high_game": hi_g, "b_better_game_ci": bool(lo_g > 0)})
    return pd.DataFrame(rows)


def oracle_metrics(spec: imf.TargetSpec, data: TrackedData, eval_idx: np.ndarray) -> dict[str, float]:
    """The official NGS field used directly as the prediction (reference row)."""
    if spec.oracle is None:
        return {}
    col = {"defenders_in_box": "defendersInTheBox", "number_of_pass_rushers": "numberOfPassRushers"}.get(spec.oracle, spec.oracle)
    y = data.Y[spec.name].to_numpy(dtype=float)[eval_idx]
    o = pd.to_numeric(data.plays[col], errors="coerce").to_numpy(dtype=float)[eval_idx]
    return score(spec.kind, y, o)


def ranking_table(runs: dict[tuple[str, str], TargetRun], data: TrackedData, cfg: ImputationConfig) -> pd.DataFrame:
    """One row per target: skill (R2 / BSS) of every feature set, k-fold and forward, plus the oracle."""
    rows = []
    for spec in imf.TARGETS:
        if (spec.name, "kfold") not in runs:
            continue
        k = runs[(spec.name, "kfold")]
        f = runs.get((spec.name, "forward"))
        row: dict[str, Any] = {"target": spec.name, "kind": spec.kind, "subset": spec.subset, "n_kfold": int(len(k.eval_idx)),
                               "n_forward": int(len(f.eval_idx)) if f else 0,
                               "base_metric": "log_loss" if spec.kind == "binary" else "mae",
                               "base_global": k.metrics["base_global"].get("log_loss" if spec.kind == "binary" else "mae", np.nan),
                               "base_dd_skill": skill_of(spec.kind, k.metrics["base_dd"]),
                               "base_play_type_skill": skill_of(spec.kind, k.metrics["base_play_type"]),
                               "base_outcome_skill": skill_of(spec.kind, k.metrics["base_outcome"])}
        for fset in cfg.feature_sets:
            row[f"{fset}_skill"] = skill_of(spec.kind, k.metrics[fset])
            if spec.kind == "binary":
                row[f"{fset}_auc"] = k.metrics[fset].get("auc", np.nan)
            else:
                row[f"{fset}_mae"] = k.metrics[fset].get("mae", np.nan)
        for fset in ("F0", "F1"):
            row[f"{fset}_skill_forward"] = skill_of(spec.kind, f.metrics[fset]) if f and fset in f.metrics else np.nan
        om = oracle_metrics(spec, data, k.eval_idx)
        row["oracle_field"] = spec.oracle
        row["oracle_skill"] = skill_of(spec.kind, om) if om else np.nan
        row["recoverable_F0"] = _label(row.get("F0_skill", np.nan))
        row["recoverable_F1"] = _label(row.get("F1_skill", np.nan))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("F0_skill", ascending=False).reset_index(drop=True)


def _label(s: float) -> str:
    if np.isnan(s):
        return "n/a"
    if s >= 0.5:
        return "well"
    if s >= 0.2:
        return "partly"
    if s >= 0.05:
        return "weakly"
    return "not"


def _bucket_air_yards(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.full(len(a), "na", dtype=object)
    out[a < 0] = "<0"
    out[(a >= 0) & (a < 5)] = "0-5"
    out[(a >= 5) & (a < 10)] = "5-10"
    out[(a >= 10) & (a < 20)] = "10-20"
    out[a >= 20] = "20+"
    return out


def _bucket_ttt(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.full(len(t), "na", dtype=object)
    out[t < 2.0] = "<2.0"
    out[(t >= 2.0) & (t < 2.5)] = "2.0-2.5"
    out[(t >= 2.5) & (t < 3.0)] = "2.5-3.0"
    out[(t >= 3.0) & (t < 4.0)] = "3.0-4.0"
    out[t >= 4.0] = "4.0+"
    return out


def grouped_errors(spec: imf.TargetSpec, run: TargetRun, data: TrackedData, by: np.ndarray, by_name: str,
                   models: tuple[str, ...], min_n: int = 20) -> pd.DataFrame:
    """Per-bucket metrics of several models (k-fold OOF) for the error analysis.

    Buckets with fewer than ``min_n`` plays are dropped; ``*_r2_within`` / ``*_bss_within`` /
    ``*_auc`` are computed inside the bucket (skill relative to the bucket's own mean), so
    they say what a model recovers beyond knowing the bucket. Rows follow
    :data:`GROUP_ORDER` when ``by_name`` has an entry, else the label order.
    """
    y = data.Y[spec.name].to_numpy(dtype=float)
    rows = []
    idx = run.eval_idx
    for b in pd.unique(by[idx]):
        sel = idx[by[idx] == b]
        if len(sel) < min_n:
            continue
        row: dict[str, Any] = {by_name: b, "n": int(len(sel))}
        if spec.kind == "binary":
            row["base_rate"] = float(y[sel].mean())
        else:
            row["y_mean"] = float(y[sel].mean())
        for m in models:
            s = score(spec.kind, y[sel], run.preds[m][sel])
            if spec.kind == "binary":
                row[f"{m}_mean_pred"] = s.get("mean_pred", np.nan)
                row[f"{m}_log_loss"] = s.get("log_loss", np.nan)
                row[f"{m}_acc"] = s.get("acc", np.nan)
                row[f"{m}_auc"] = s.get("auc", np.nan)
                row[f"{m}_bss_within"] = s.get("bss", np.nan)
            else:
                row[f"{m}_mae"] = s.get("mae", np.nan)
                row[f"{m}_bias"] = s.get("bias", np.nan)
                row[f"{m}_r2_within"] = s.get("r2", np.nan)
        rows.append(row)
    order = {v: i for i, v in enumerate(GROUP_ORDER.get(by_name, []))}
    out = pd.DataFrame(rows)
    out["_o"] = out[by_name].map(lambda v: order.get(v, len(order)))
    return out.sort_values(["_o", by_name]).drop(columns="_o").reset_index(drop=True)


def error_analysis(runs: dict[tuple[str, str], TargetRun], data: TrackedData, cfg: ImputationConfig) -> dict[str, pd.DataFrame]:
    """Error tables keyed ``<target>_by_<grouping>`` (written as ``nfl_03_error_<key>.parquet``).

    Besides the pre-existing groupings (air yards, pass location, tracking time to throw,
    down x distance), every within-play target in :data:`OUTCOME_GROUPINGS` is grouped by
    the nflfastR outcome class the F1 students see, with ``base_outcome`` next to the
    students, so the after-the-fact skill can be read inside an outcome class.
    """
    out: dict[str, pd.DataFrame] = {}
    models = tuple(m for m in ("base_global", "F0", "F0P", "F1", "F2") if m in cfg.feature_sets or m == "base_global")
    if ("separation_at_arrival", "kfold") in runs:
        spec, r = imf.TARGET_BY_NAME["separation_at_arrival"], runs[("separation_at_arrival", "kfold")]
        out["separation_by_air_yards"] = grouped_errors(spec, r, data, _bucket_air_yards(data.plays["pbp_air_yards"].to_numpy()), "air_yards", models)
        loc = data.plays["pbp_pass_location"].astype(object).fillna("na").to_numpy(dtype=object)
        out["separation_by_pass_location"] = grouped_errors(spec, r, data, loc, "pass_location", models)
    if ("pressure_derived", "kfold") in runs:
        spec, r = imf.TARGET_BY_NAME["pressure_derived"], runs[("pressure_derived", "kfold")]
        out["pressure_by_time_to_throw"] = grouped_errors(spec, r, data, _bucket_ttt(data.plays["time_to_throw"].to_numpy()), "time_to_throw", models)
    if ("mof_open", "kfold") in runs:
        spec, r = imf.TARGET_BY_NAME["mof_open"], runs[("mof_open", "kfold")]
        out["mof_open_by_down_distance"] = grouped_errors(spec, r, data, data.dd_bucket, "down_distance", models)
    omodels = tuple(m for m in ("base_global", "base_outcome", "F0P", "F1", "F2") if m in cfg.feature_sets or m.startswith("base_"))
    for tgt, grouping in OUTCOME_GROUPINGS.items():
        if (tgt, "kfold") not in runs:
            continue
        out[f"{tgt}_by_{grouping}"] = grouped_errors(imf.TARGET_BY_NAME[tgt], runs[(tgt, "kfold")], data,
                                                     data.groupings[grouping], grouping, omodels)
    return out


def _skill_vs(kind: str, y: np.ndarray, p: np.ndarray, p_ref: np.ndarray) -> float:
    """``1 - MSE(p) / MSE(p_ref)``: skill of ``p`` relative to the reference prediction ``p_ref``."""
    y = np.asarray(y, dtype=float)
    if kind == "binary":
        p, p_ref = np.clip(p, 1e-6, 1 - 1e-6), np.clip(p_ref, 1e-6, 1 - 1e-6)
    mse_ref = float(np.mean((y - p_ref) ** 2))
    return float(1 - np.mean((y - p) ** 2) / mse_ref) if mse_ref > 0 else np.nan


def outcome_decomposition(runs: dict[tuple[str, str], TargetRun], data: TrackedData, cfg: ImputationConfig,
                          errors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per target: how much of the F1 / F2 skill survives conditioning on the outcome class.

    Columns: pooled k-fold skill of ``F0P`` / ``F1`` / ``F2`` and of ``base_outcome`` (per
    outcome-class training mean, an F1-level baseline); ``F1_vs_outcome`` / ``F2_vs_outcome``
    = 1 - MSE(student) / MSE(base_outcome); for the targets in :data:`OUTCOME_GROUPINGS`
    the largest outcome class, its size and the F1 skill (R2 / BSS, plus AUC for binary
    targets) inside that class.
    """
    rows = []
    for spec in imf.TARGETS:
        if (spec.name, "kfold") not in runs:
            continue
        r = runs[(spec.name, "kfold")]
        y = data.Y[spec.name].to_numpy(dtype=float)[r.eval_idx]
        row: dict[str, Any] = {"target": spec.name, "kind": spec.kind, "subset": spec.subset, "n": int(len(r.eval_idx)),
                               "base_outcome_skill": skill_of(spec.kind, r.metrics["base_outcome"]),
                               "n_outcome_classes": int(len(np.unique(data.outcome[r.eval_idx])))}
        for fset in ("F0P", "F1", "F2"):
            row[f"{fset}_skill"] = skill_of(spec.kind, r.metrics[fset]) if fset in r.metrics else np.nan
        for fset in ("F1", "F2"):
            row[f"{fset}_vs_outcome"] = (_skill_vs(spec.kind, y, r.preds[fset][r.eval_idx], r.preds["base_outcome"][r.eval_idx])
                                         if fset in r.preds else np.nan)
        grouping = OUTCOME_GROUPINGS.get(spec.name)
        row["grouping"] = grouping or ""
        row["largest_class"] = ""
        row["largest_class_n"] = np.nan
        row["F1_skill_in_largest_class"] = np.nan
        row["F1_auc_in_largest_class"] = np.nan
        tab = errors.get(f"{spec.name}_by_{grouping}") if grouping else None
        if tab is not None and len(tab):
            top = tab.sort_values("n", ascending=False).iloc[0]
            row["largest_class"] = str(top[grouping])
            row["largest_class_n"] = int(top["n"])
            key = "F1_bss_within" if spec.kind == "binary" else "F1_r2_within"
            row["F1_skill_in_largest_class"] = float(top.get(key, np.nan))
            row["F1_auc_in_largest_class"] = float(top.get("F1_auc", np.nan)) if spec.kind == "binary" else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_tables(runs: dict[tuple[str, str], TargetRun], data: TrackedData, fsets: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for (target, split), r in runs.items():
        spec = imf.TARGET_BY_NAME[target]
        if spec.kind != "binary" or split != "kfold":
            continue
        y = data.Y[target].to_numpy(dtype=float)[r.eval_idx]
        for fset in fsets:
            if fset not in r.preds:
                continue
            p = np.clip(r.preds[fset][r.eval_idx], 0, 1)
            for i, b in enumerate(calibration_table(y, p, n_bins=10)):
                rows.append({"target": target, "feature_set": fset, "bin": i, **b})
    return pd.DataFrame(rows)


def targets_table(data: TrackedData) -> pd.DataFrame:
    rows = []
    for t in imf.TARGETS:
        y = data.Y[t.name].dropna()
        rows.append({"target": t.name, "kind": t.kind, "subset": t.subset, "source_column": t.source_col,
                     "oracle_field": t.oracle or "", "n_labelled": int(len(y)), "mean": float(y.mean()), "sd": float(y.std()),
                     "definition": t.description})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _f(df: pd.DataFrame, cols: list[str] | None = None, fmt: str = "{:.3f}") -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return md_table(d, floatfmt=fmt)


def write_report(res: dict[str, Any], cfg: ImputationConfig, path: Path) -> None:
    metrics: pd.DataFrame = res["metrics"]
    rank: pd.DataFrame = res["ranking"]
    deltas: pd.DataFrame = res["deltas"]
    fs = list(cfg.feature_sets)
    L: list[str] = []
    L.append("# NFL 03 - imputation: event-only students for tracking-derived state\n")
    L.append(f"Generated by `python -m research.privileged_tracking.nfl.imputation` in {res['seconds'] / 60:.1f} min "
             f"(LightGBM, {cfg.n_jobs} threads). Inputs: `processed_dir('nfl')/plays_tracked.parquet` (NFL 01, "
             f"{len(res['data'].plays):,} tracked non-special-teams plays of the 91 tracked 2017 games, weeks 1-6) and the full "
             "nflfastR 2017 season for the team tendencies. Outputs: `imputed_oof.parquet` (out-of-fold predictions of every "
             "student for every tracked play, `<target>__<fset>`, plus `y_<target>`), `imputed_forward.parquet` (weeks 5-6 "
             "predictions of the forward split), `models/students_<fset>.joblib` (final students fitted on all 91 games; "
             "`apply_student.StudentBundle`), and the `nfl_03_*.parquet` tables next to this file.\n")
    L.append("## 0. Headline\n")
    L.append(res["headline"] + "\n")
    L.append("## 1. Protocol\n")
    L.append(f"""* **Students.** One LightGBM model per (target, feature set): `regression` (L2) for continuous and count targets,
  `binary` for flags; learning rate {cfg.learning_rate}, {cfg.num_leaves} leaves, min {cfg.min_data_in_leaf} plays per leaf, feature / bagging
  fraction 0.8, L2 1.0, max {cfg.max_rounds} rounds. Rounds are chosen by early stopping ({cfg.early_stopping}) on an inner holdout of
  {int(cfg.inner_holdout_frac * 100)}% of the *training-fold games*; the evaluated model is the one trained on the other {int((1 - cfg.inner_holdout_frac) * 100)}%
  (no refit inside the folds). The final students in `models/` use the inner-holdout round count and are refitted on all tracked games.
* **Splits.** (a) {cfg.n_folds}-fold group k-fold by `gameId` (`common.splits.group_kfold`, seed {cfg.seed}): every tracked play gets one
  out-of-fold prediction; (b) forward: weeks {list(cfg.train_weeks)} train, weeks {list(cfg.test_weeks)} test.
* **Feature sets** (nested; `imputation_features.FEATURE_SETS`):
  `F0n` = pre-snap situation (down, distance, yard line, quarter, clocks, score differential, wp, ep, xpass, goal-to-go,
  description-derived shotgun / no-huddle, home / away, down-distance bucket) + {len(imf.TENDENCIES)} team tendencies
  (`tend_*`: posteam shotgun / no-huddle / pass / deep-pass / air-yards / sack / QB-hit / rush-yards rates and the defteam
  equivalents, computed from *strictly earlier weeks* of the full nflfastR 2017 season, shrunk with {imf.TENDENCY_ALPHA:.0f} plays towards
  fixed league-typical constants; week 1 sees only the constants);
  `F0` = F0n + team target encodings `te_off` / `te_def` (posteam's / defteam's mean of the target over the *training-fold
  games only*, leave-one-game-out for training rows, shrunk with {cfg.te_alpha:.0f} plays to the training mean);
  `F0P` = F0 + formation and personnel counts (plays.csv `offenseFormation` / `personnel_offense` / `personnel_defense`, the
  same charting nflverse publishes in `pbp_participation`);
  `F1` = F0P + post-play fields (play type, dropback / scramble, pass length and location, air yards, YAC, completion,
  interception, sack, QB hit, run location / gap, yards gained, EPA, receiver position group);
  `F2` = F1 + the official NGS `defenders_in_box` and `number_of_pass_rushers`.
* **Baselines** (all from the training fold): `base_global` = mean / base rate; `base_play_type` = mean by nflfastR
  `play_type` (post-snap information, an F1-level baseline); `base_dd` = mean by down x distance bucket (short 1-3 / mid 4-7 /
  long 8+); `base_outcome` = mean by nflfastR outcome class (pass plays: sack x QB hit x completion x interception; run
  plays: yards-gained bucket <0 / 0-2 / 3-5 / 6-10 / 10+; `imputation.outcome_bucket`), i.e. the "re-encode the outcome
  flags" reference that any after-the-fact (F1 / F2) skill has to beat. Buckets with < 20 training plays fall back to the
  global mean.
* **Metrics.** Continuous / count: R2, MAE, RMSE, bias (counts also exact and within-1 after rounding). Binary: AUC,
  log-loss, Brier, Brier skill score (BSS = 1 - Brier / Brier(base rate)), accuracy at 0.5, calibration tables. `skill` in the
  ranking table is R2 for continuous / count targets and BSS for binary targets (both are 1 - MSE / MSE of the mean).
  Paired bootstrap deltas ({cfg.n_boot} resamples) of per-play absolute error / log-loss: play-level `ci_low` / `ci_high` and
  game-clustered `ci_low_game` / `ci_high_game` (quote the clustered one). All n are labelled plays with a prediction.
* **Leakage contract.** Features come only from `id` / `event_only_presnap` columns (F0), plus `ngs_charting_privileged`
  personnel / formation (F0P), `event_only_postsnap` (F1) and the two official counts (F2); every tendency uses strictly earlier
  games; the target encodings never see a held-out game; nothing is tuned on the test folds. `pbp_shotgun` / `pbp_no_huddle`
  are description-derived, so `shotgun_derived` under F0 is a near-tautology and serves as a pipeline check.
""")
    L.append("## 2. Targets\n")
    L.append("`subset`: `all` = every tracked play; `pass` = BDB PassResult in C/I/IN/S; `run` = nflfastR `play_type == run` that is "
             "not a BDB pass play (includes scrambles). `n_labelled` counts plays where the target is defined.\n")
    L.append(_f(res["targets"], ["target", "kind", "subset", "n_labelled", "mean", "sd", "oracle_field", "definition"]) + "\n")
    L.append("### 2a. Pressure flag definition\n")
    L.append(f"`pressure_derived = min_def_dist_qb_throw <= {imf.PRESSURE_DIST_YD} yd`. Threshold grid vs NGS `was_pressure` "
             f"(pass plays with the flag), selected on weeks {list(TUNE_WEEKS)} (`agree_tune`) and reported on weeks 4-6 "
             f"(`agree_heldout`). Selected on the tuning weeks: {res['pressure_selected']:.2f} yd"
             + (" (equals the module constant)." if abs(res['pressure_selected'] - imf.PRESSURE_DIST_YD) < 1e-9
                else f" - NOTE: differs from the module constant {imf.PRESSURE_DIST_YD}, which was kept.") + "\n")
    L.append(_f(res["pressure_grid"]) + "\n")

    L.append("## 3. Results: 5-fold group k-fold by game (out-of-fold)\n")
    L.append("### 3a. Continuous and count targets (R2 / MAE; `exact` = rounded exact agreement for counts)\n")
    kf = metrics[(metrics["split"] == "kfold")]
    cont = kf[kf["kind"] != "binary"]
    piv = _pivot(cont, ["r2", "mae"], fs + BASELINES)
    L.append(_f(piv) + "\n")
    cnt = kf[kf["kind"] == "count"]
    if len(cnt):
        L.append("Rounded exact / within-1 agreement for count targets (k-fold OOF):\n")
        L.append(_f(_pivot(cnt, ["exact_rounded", "within_1"], ["base_global"] + fs)) + "\n")
    L.append("### 3b. Binary targets (AUC / log-loss / BSS)\n")
    binm = kf[kf["kind"] == "binary"]
    L.append(_f(_pivot(binm, ["auc", "log_loss", "bss"], fs + BASELINES)) + "\n")

    L.append("## 4. Results: forward split (weeks 1-4 -> 5-6)\n")
    fw = metrics[(metrics["split"] == "forward")]
    L.append("Continuous / count (R2 / MAE):\n")
    L.append(_f(_pivot(fw[fw["kind"] != "binary"], ["r2", "mae"], fs + BASELINES)) + "\n")
    L.append("Binary (AUC / log-loss / BSS):\n")
    L.append(_f(_pivot(fw[fw["kind"] == "binary"], ["auc", "log_loss", "bss"], fs + BASELINES)) + "\n")

    L.append("## 5. Ranking: what play-by-play can recover\n")
    L.append("Skill = R2 (continuous / count) or BSS (binary) of the k-fold OOF predictions; `*_skill_forward` = the same on the "
             "forward split; `oracle_skill` = the official NGS field used directly as the prediction (where one exists); "
             "`recoverable_*`: well >= 0.5, partly >= 0.2, weakly >= 0.05, else not. Sorted by F0 skill.\n")
    L.append(_f(rank, ["target", "kind", "subset", "n_kfold", "base_dd_skill", "base_play_type_skill", "base_outcome_skill"]
                + [f"{f}_skill" for f in fs]
                + ["F0_skill_forward", "F1_skill_forward", "oracle_skill", "recoverable_F0", "recoverable_F1"]) + "\n")
    L.append(res["ranking_text"] + "\n")
    L.append("### 5a. After-the-fact skill decomposed by outcome class\n")
    L.append("The F1 / F2 students see the nflfastR outcome flags (sack, QB hit, completion, interception, yards gained), and "
             "the within-play targets differ strongly between outcome classes (a sack has a long derived `time_to_throw` and a "
             "defender at the QB by definition; an incompletion has a short `separation_at_arrival`). `base_outcome_skill` is the "
             "skill of the per-outcome-class training mean alone; `F1_vs_outcome` = 1 - MSE(F1) / MSE(base_outcome) is what "
             "the student adds beyond the class means; `F1_skill_in_largest_class` is the F1 skill (R2 / BSS) computed inside "
             "the largest outcome class of the target's grouping (`pass_qb_outcome`: clean / qb_hit / sack; `pass_ball_outcome`: "
             "complete / incomplete / interception / sack; `run_yards_bucket`), see section 7e for every class. "
             "Read the pooled F1 numbers of within-play targets together with these columns.\n")
    dec = res["decomposition"].copy()
    dec["largest_class_n"] = pd.Series([np.nan if pd.isna(v) else int(v) for v in dec["largest_class_n"]], index=dec.index, dtype=object)
    L.append(_f(dec[dec["subset"] != "all"], ["target", "kind", "subset", "n", "n_outcome_classes", "base_outcome_skill", "F0P_skill",
                                              "F1_skill", "F2_skill", "F1_vs_outcome", "F2_vs_outcome", "grouping", "largest_class",
                                              "largest_class_n", "F1_skill_in_largest_class", "F1_auc_in_largest_class"]) + "\n")
    L.append("Pre-snap (`all`) targets against the same baseline (here `base_outcome` is just another F1-level baseline):\n")
    L.append(_f(dec[dec["subset"] == "all"], ["target", "kind", "n", "base_outcome_skill", "F0P_skill", "F1_skill", "F2_skill",
                                              "F1_vs_outcome", "F2_vs_outcome"]) + "\n")
    L.append(res["decomposition_text"] + "\n")

    L.append("## 6. Paired bootstrap deltas (k-fold OOF; positive = `b` better than `a`)\n")
    L.append("Loss = per-play absolute error (continuous / count) or log-loss (binary); `ci_*` play-level, `ci_*_game` game-clustered "
             f"({cfg.n_boot} resamples). `b_better_game_ci` = the clustered interval excludes 0 in favour of `b`.\n")
    dk = deltas[deltas["split"] == "kfold"]
    L.append(_f(dk, ["target", "a", "b", "loss", "n", "games", "delta_a_minus_b", "ci_low_game", "ci_high_game", "b_better_game_ci"], "{:.4f}") + "\n")
    L.append("Forward split:\n")
    df_ = deltas[deltas["split"] == "forward"]
    L.append(_f(df_[df_["a"].isin(["base_global", "F0P", "base_outcome"]) | (df_["b"] == "F0")],
                ["target", "a", "b", "loss", "n", "games", "delta_a_minus_b", "ci_low_game", "ci_high_game", "b_better_game_ci"], "{:.4f}") + "\n")

    L.append("## 7. Error analysis (k-fold OOF)\n")
    ea = res["errors"]
    if "separation_by_air_yards" in ea:
        L.append("### 7a. `separation_at_arrival` error by nflfastR `air_yards` bucket (yd)\n")
        L.append(_f(ea["separation_by_air_yards"]) + "\n")
        L.append("### 7b. `separation_at_arrival` error by nflfastR `pass_location`\n")
        L.append(_f(ea["separation_by_pass_location"]) + "\n")
    if "pressure_by_time_to_throw" in ea:
        L.append("### 7c. `pressure_derived` by tracking `time_to_throw` bucket (s; diagnostic grouping, not a feature)\n")
        L.append(_f(ea["pressure_by_time_to_throw"]) + "\n")
    if "mof_open_by_down_distance" in ea:
        L.append("### 7d. `mof_open` by down x distance bucket\n")
        L.append(_f(ea["mof_open_by_down_distance"]) + "\n")
    L.append("### 7e. Within-play targets inside each nflfastR outcome class (what the F1 students see)\n")
    L.append("`*_r2_within` / `*_bss_within` / `*_auc` are computed inside the class, i.e. relative to the class's own mean; "
             "`base_outcome` is the per-class training mean (its within-class skill is ~0 by construction). Classes: "
             "`pass_qb_outcome` = `clean` (no sack, no QB hit) / `qb_hit` (hit, no sack) / `sack`; `pass_ball_outcome` = "
             "`complete` / `incomplete` / `interception` (sacks have no arrival); `run_yards_bucket` = nflfastR `yards_gained`.\n")
    for tgt, grouping in OUTCOME_GROUPINGS.items():
        key = f"{tgt}_by_{grouping}"
        if key in ea:
            L.append(f"`{tgt}` by `{grouping}`:\n")
            L.append(_f(ea[key]) + "\n")
    L.append("### 7f. Calibration of the binary students (k-fold OOF, 10 equal-count bins)\n")
    cal = res["calibration"]
    if len(cal):
        for tgt in cal["target"].unique():
            c = cal[cal["target"] == tgt].pivot(index="bin", columns="feature_set", values=["pred", "obs"])
            c.columns = [f"{a}_{b}" for a, b in c.columns]
            c = c.reset_index()
            nn = cal[(cal["target"] == tgt) & (cal["feature_set"] == cal["feature_set"].iloc[0])].set_index("bin")["n"]
            c["n"] = c["bin"].map(nn)
            L.append(f"`{tgt}`:\n")
            L.append(_f(c) + "\n")

    L.append("## 8. Feature importance of the final students (gain share, top 5)\n")
    imp = res["importance"]
    for fset in ("F0", "F1"):
        sub = imp[imp["feature_set"] == fset]
        if not len(sub):
            continue
        rows = []
        for tgt, g in sub.groupby("target", sort=False):
            top = g.sort_values("gain_share", ascending=False).head(5)
            rows.append({"target": tgt, "feature_set": fset,
                         "top_features": ", ".join(f"{f} ({s:.2f})" for f, s in zip(top["feature"], top["gain_share"]))})
        L.append(f"`{fset}`:\n")
        L.append(_f(pd.DataFrame(rows)) + "\n")

    if res.get("apply") is not None:
        ap = res["apply"]
        season = cfg.apply_season
        L.append(f"## 9. Application to the {season} season (`apply_student.py`)\n")
        L.append(f"`python -m research.privileged_tracking.nfl.apply_student --season {season} --feature-sets "
                 f"{','.join(cfg.apply_feature_sets)}` rebuilt the tendencies from the {season} nflfastR file, merged the {season} "
                 f"`pbp_participation` charting (personnel / formation for F0P, plus the official fields used only for validation) and "
                 f"applied the final students to every {season} regular-season scrimmage play ({ap['n_rows']:,} pbp rows written to "
                 f"`imputed_pbp_{season}.parquet`, {ap['seconds']:.0f} s). Team target encodings are the 2017 tracked-season values "
                 "(a season stale; teams keep their nflfastR codes).\n")
        L.append("### 9a. Input coverage\n")
        L.append(_f(ap["coverage"]) + "\n")
        L.append(f"### 9b. External validation on {season}: imputed value vs the official NGS field of the same play\n")
        L.append("The derived 2017 targets are not identical to the official fields (first row per target, the ceiling); the "
                 f"student rows compare the imputed value with the official field on {season} plays it has never seen, next to the "
                 "same comparison for the 2017 out-of-fold imputations. `share_sack` is the share of compared plays that are sacks: "
                 "the official `time_to_throw` / `was_pressure` fields are **missing on every sack** (section 9a), so the "
                 "`time_to_throw` and `pressure_derived` rows are non-sack pass plays only, in both seasons - they are the "
                 "non-sack numbers of section 7e, not the pooled k-fold numbers of section 3.\n")
        L.append(_f(ap["external"]) + "\n")
        L.append(f"### 9c. Distribution shift: {season} imputed vs 2017 out-of-fold imputed vs 2017 truth\n")
        sh = ap["shift"]
        L.append(_f(sh[sh["feature_set"].isin(["F0", "F1"])], ["target", "feature_set", "source", "n", "mean", "sd", "p05", "p50", "p95"]) + "\n")

    L.append("## 10. Caveats\n")
    L.append(res["caveats"] + "\n")
    path.write_text("\n".join(L))


def _pivot(df: pd.DataFrame, values: list[str], models: list[str]) -> pd.DataFrame:
    if not len(df):
        return pd.DataFrame()
    first = df.drop_duplicates("target")[["target", "subset", "n"]].set_index("target")
    out = first.copy()
    for v in values:
        p = df.pivot(index="target", columns="model", values=v)
        for m in models:
            if m in p.columns:
                out[f"{m}_{v}"] = p[m]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: ImputationConfig) -> dict[str, Any]:
    t0 = time.time()
    data = load_tracked(cfg)
    specs = [t for t in imf.TARGETS if cfg.targets is None or t.name in cfg.targets]
    folds = folds_for(data, cfg)
    assert set(data.week[folds["forward"][0][1]]) == set(cfg.test_weeks)
    runs: dict[tuple[str, str], TargetRun] = {}
    for i, spec in enumerate(specs):
        t1 = time.time()
        for split in ("kfold", "forward"):
            runs[(spec.name, split)] = run_target(spec, data, folds[split], split, cfg)
        k = runs[(spec.name, "kfold")].metrics
        s0, s1 = skill_of(spec.kind, k["F0"]) if "F0" in k else np.nan, skill_of(spec.kind, k["F1"]) if "F1" in k else np.nan
        print(f"[{i + 1}/{len(specs)}] {spec.name:40s} n={len(runs[(spec.name, 'kfold')].eval_idx):5d} "
              f"skill F0={s0:6.3f} F1={s1:6.3f}  ({time.time() - t1:.0f} s, total {(time.time() - t0) / 60:.1f} min)", flush=True)

    metrics = metrics_long(runs)
    deltas = deltas_table(runs, data, cfg)
    ranking = ranking_table(runs, data, cfg)
    errors = error_analysis(runs, data, cfg)
    decomposition = outcome_decomposition(runs, data, cfg, errors)
    cal = calibration_tables(runs, data, tuple(f for f in ("F0", "F1") if f in cfg.feature_sets))
    pressure_grid, pressure_selected = pressure_threshold_table(data.plays)
    tt = targets_table(data)

    # out-of-fold / forward prediction tables
    oof_cols: dict[str, np.ndarray] = {}
    fwd_cols: dict[str, np.ndarray] = {}
    for spec in specs:
        oof_cols[f"y_{spec.name}"] = data.Y[spec.name].to_numpy()
        for fset in cfg.feature_sets:
            oof_cols[f"{spec.name}__{fset}"] = runs[(spec.name, "kfold")].preds[fset]
            fwd_cols[f"{spec.name}__{fset}"] = runs[(spec.name, "forward")].preds[fset]
    keys = data.plays[["gameId", "playId", "week"]].reset_index(drop=True)
    oof = pd.concat([keys, pd.DataFrame(oof_cols)], axis=1)
    fwd = pd.concat([keys, pd.DataFrame(fwd_cols)], axis=1)
    fwd = fwd[np.isin(fwd["week"].to_numpy(), cfg.test_weeks)].reset_index(drop=True)

    # final students
    print("fitting final students ...", flush=True)
    bundles: dict[str, aps.StudentBundle] = {}
    imps = []
    for fset in cfg.feature_sets:
        b = aps.StudentBundle(feature_set=fset, features=list(imf.FEATURE_SETS[fset]), te_alpha=cfg.te_alpha)
        for spec in specs:
            st, imp = fit_final(spec, data, cfg, runs[(spec.name, "kfold")].preds[fset], fset)
            b.students[spec.name] = st
            imps.append(imp)
        bundles[fset] = b
    importance = pd.concat(imps, ignore_index=True)
    print(f"final students fitted ({(time.time() - t0) / 60:.1f} min)", flush=True)

    proc = processed_dir("nfl")
    rdir = reports_dir()
    if cfg.write:
        oof.to_parquet(proc / "imputed_oof.parquet", index=False)
        fwd.to_parquet(proc / "imputed_forward.parquet", index=False)
        for b in bundles.values():
            b.save()
        metrics.to_parquet(rdir / "nfl_03_metrics.parquet", index=False)
        deltas.to_parquet(rdir / "nfl_03_deltas.parquet", index=False)
        ranking.to_parquet(rdir / "nfl_03_ranking.parquet", index=False)
        cal.to_parquet(rdir / "nfl_03_calibration.parquet", index=False)
        pressure_grid.to_parquet(rdir / "nfl_03_pressure_threshold.parquet", index=False)
        tt.to_parquet(rdir / "nfl_03_targets.parquet", index=False)
        importance.to_parquet(rdir / "nfl_03_importance.parquet", index=False)
        decomposition.to_parquet(rdir / "nfl_03_outcome_decomposition.parquet", index=False)
        for k, v in errors.items():
            v.to_parquet(rdir / f"nfl_03_error_{k}.parquet", index=False)

    apply_res = None
    if cfg.apply_season is not None and cfg.write:
        print(f"applying students to {cfg.apply_season} ...", flush=True)
        apply_res = aps.run_season(cfg.apply_season, cfg.apply_feature_sets, write=True)
        print(f"applied ({apply_res['seconds']:.0f} s)", flush=True)

    res: dict[str, Any] = {"data": data, "runs": runs, "metrics": metrics, "deltas": deltas, "ranking": ranking,
                           "errors": errors, "decomposition": decomposition, "calibration": cal, "pressure_grid": pressure_grid,
                           "pressure_selected": pressure_selected, "targets": tt, "importance": importance,
                           "oof": oof, "forward": fwd, "bundles": bundles, "apply": apply_res,
                           "seconds": time.time() - t0}
    res["headline"] = headline(res, cfg)
    res["ranking_text"] = ranking_text(res, cfg)
    res["decomposition_text"] = decomposition_text(res, cfg)
    res["caveats"] = caveats(res, cfg)
    if cfg.write:
        write_report(res, cfg, rdir / "nfl_03_imputation.md")
        print(f"report written to {rdir / 'nfl_03_imputation.md'}")
    return res


def headline(res: dict[str, Any], cfg: ImputationConfig) -> str:
    rank: pd.DataFrame = res["ranking"]
    if not len(rank):
        return ""
    lines = []
    for tier, col in (("F0 (pre-snap play-by-play + team tendencies)", "F0_skill"),
                      ("F0P (+ personnel / formation)", "F0P_skill"), ("F1 (+ post-play fields)", "F1_skill")):
        if col not in rank.columns:
            continue
        r = rank.sort_values(col, ascending=False)
        well = r[r[col] >= 0.5]["target"].tolist()
        partly = r[(r[col] >= 0.2) & (r[col] < 0.5)]["target"].tolist()
        not_ = r[r[col] < 0.05]["target"].tolist()
        lines.append(f"* **{tier}**: skill >= 0.5 for {', '.join(f'`{t}`' for t in well) or 'none'}; 0.2-0.5 for "
                     f"{', '.join(f'`{t}`' for t in partly) or 'none'}; < 0.05 (not recoverable) for "
                     f"{', '.join(f'`{t}`' for t in not_) or 'none'}.")
    lines.append("* " + decomposition_text(res, cfg))
    d = res["deltas"]
    dk = d[(d["split"] == "kfold")]
    n_sig = {}
    for a, b in DELTA_PAIRS[:5] + [("base_outcome", "F1")]:
        s = dk[(dk["a"] == a) & (dk["b"] == b)]
        if len(s):
            n_sig[(a, b)] = (int(s["b_better_game_ci"].sum()), int(len(s)))
    lines.append("* Game-clustered bootstrap: " + "; ".join(f"`{b}` beats `{a}` on {k}/{n} targets" for (a, b), (k, n) in n_sig.items()) + ".")
    ap = res.get("apply")
    if ap is not None and len(ap["external"]):
        ex = ap["external"]
        bits = []
        for tgt in ("box_count_tuned", "time_to_throw", "n_pass_rushers_derived", "pressure_derived", "target_depth"):
            s = ex[(ex["target"] == tgt) & (ex["season"] == cfg.apply_season) & (ex["feature_set"] == "F1")]
            s0 = ex[(ex["target"] == tgt) & (ex["season"] == 2017) & (ex["feature_set"] == "F1")]
            if len(s) and len(s0):
                key = "auc" if tgt == "pressure_derived" else "r2"
                note = " (non-sack plays only: the official field is missing on every sack)" if tgt in ("time_to_throw", "pressure_derived") else ""
                bits.append(f"`{tgt}` {key} {s.iloc[0].get(key, np.nan):.2f} on {cfg.apply_season} vs {s0.iloc[0].get(key, np.nan):.2f} on 2017 OOF (n={int(s.iloc[0]['n']):,}){note}")
        if bits:
            lines.append(f"* Out-of-season check against the official NGS fields of {cfg.apply_season} (F1 students): " + "; ".join(bits) + ".")
    return "\n".join(lines)


def decomposition_text(res: dict[str, Any], cfg: ImputationConfig) -> str:
    """One paragraph: how much of the F1 within-play skill is outcome-class identification."""
    dec: pd.DataFrame = res.get("decomposition", pd.DataFrame())
    if not len(dec) or "F1_skill" not in dec.columns:
        return ""
    d = dec.set_index("target")
    parts = []
    for tgt in ("time_to_throw", "min_def_dist_qb_throw", "pressure_derived", "separation_at_arrival", "n_def_within_r_target",
                "target_depth", "yards_to_first_contact", "n_def_within_r_carrier_first_contact"):
        if tgt not in d.index:
            continue
        r = d.loc[tgt]
        s = (f"`{tgt}` F1 {r['F1_skill']:.2f} pooled, `base_outcome` alone {r['base_outcome_skill']:.2f}, F1 vs "
             f"`base_outcome` {r['F1_vs_outcome']:.2f}")
        if r["largest_class"]:
            s += f", inside `{r['largest_class']}` (n={int(r['largest_class_n']):,}) {r['F1_skill_in_largest_class']:.2f}"
            if not np.isnan(r["F1_auc_in_largest_class"]):
                s += f" / AUC {r['F1_auc_in_largest_class']:.2f}"
        parts.append(s)
    ea = res.get("errors", {})
    extra = ""
    t = ea.get("pressure_derived_by_pass_qb_outcome")
    if t is not None and "F1_auc" in t.columns:
        extra = " Pressure AUC by class: " + ", ".join(f"`{r['pass_qb_outcome']}` {r['F1_auc']:.2f} (n={int(r['n']):,})" for _, r in t.iterrows()) + "."
    d_ = res.get("deltas", pd.DataFrame())
    if len(d_):
        sub = d_[(d_["split"] == "kfold") & (d_["a"] == "base_outcome") & (d_["b"] == "F1") & (d_["target"].isin(d.index[d["subset"] != "all"]))]
        if len(sub):
            fails = sub[~sub["b_better_game_ci"]]["target"].tolist()
            extra += (f" Game-clustered bootstrap on the within-play targets: F1 beats `base_outcome` on {int(sub['b_better_game_ci'].sum())}/{len(sub)}"
                      + (f"; it does not on {', '.join(f'`{t}`' for t in fails)} (interval includes 0)." if fails else "."))
    return ("**The F1 within-play skill is mostly identification of the outcome class.** Skill = R2 / BSS; `base_outcome` = "
            "per-outcome-class training mean (sack x QB hit x completion x interception for passes, yards-gained bucket for "
            "runs), `F1 vs base_outcome` = 1 - MSE(F1) / MSE(base_outcome), `inside <class>` = F1 skill computed within the "
            "largest outcome class (section 5a / 7e): " + "; ".join(parts) + "." + extra
            + " A downstream stage that uses `imp_time_to_throw__F1` / `imp_pressure_derived__F1` / `imp_separation_at_arrival__F1` "
            "as imputed tracking state is therefore largely re-using the sack / QB-hit / completion flags; the part that goes "
            "beyond them is the `F1 vs base_outcome` column.")


def ranking_text(res: dict[str, Any], cfg: ImputationConfig) -> str:
    rank: pd.DataFrame = res["ranking"]
    if not len(rank) or "F0_skill" not in rank.columns:
        return ""
    r = rank.set_index("target")
    parts = []
    for tgt in ("box_count_tuned", "n_deep_safeties", "mof_open", "n_backfield", "n_wide_left", "cb_cushion", "motion_derived",
                "time_to_throw", "pressure_derived", "separation_at_arrival", "n_def_within_r_target", "target_depth",
                "yards_to_first_contact"):
        if tgt in r.index:
            row = r.loc[tgt]
            parts.append(f"`{tgt}` F0 {row['F0_skill']:.2f} -> F0P {row.get('F0P_skill', np.nan):.2f} -> F1 {row.get('F1_skill', np.nan):.2f}"
                         + (f" -> F2 {row['F2_skill']:.2f}" if "F2_skill" in r.columns and not np.isnan(row["F2_skill"]) else "")
                         + (f" (oracle {row['oracle_skill']:.2f})" if not np.isnan(row.get("oracle_skill", np.nan)) else ""))
    return "Skill by tier for the headline targets: " + "; ".join(parts) + "."


def caveats(res: dict[str, Any], cfg: ImputationConfig) -> str:
    pg: pd.DataFrame = res["pressure_grid"]
    sel = pg[(pg["distance"] == "min_def_dist_qb_throw") & (np.abs(pg["threshold_yd"] - imf.PRESSURE_DIST_YD) < 1e-9)]
    pressure_txt = (f"{100 * sel.iloc[0]['agree_heldout']:.1f}% of the held-out weeks 4-6 pass plays (n={int(sel.iloc[0]['n_heldout']):,}; "
                    f"{100 * sel.iloc[0]['agree_tune']:.1f}% on the tuning weeks 1-3, n={int(sel.iloc[0]['n_tune']):,})") if len(sel) else "n/a"
    cal: pd.DataFrame = res["calibration"]
    cal_txt = ""
    if len(cal):
        gap = (cal["pred"] - cal["obs"]).abs().groupby([cal["target"], cal["feature_set"]]).max().sort_values(ascending=False)
        worst = ", ".join(f"`{t}` {f} {v:.3f}" for (t, f), v in gap.head(3).items())
        cal_txt = (f"* **Calibration.** Largest 10-bin |mean prediction - observed rate| of the binary students: {worst} "
                   "(section 7f); every other (target, feature set) is below that.\n")
    ap = res.get("apply")
    cov_txt = "~98%"
    ngs_txt = "~55%"
    if ap is not None and len(ap["coverage"]):
        cv = ap["coverage"].set_index("quantity")["value"]
        cov_txt = f"{100 * cv.get('offense_formation present (share of scrimmage plays)', np.nan):.1f}%"
        ngs_txt = f"{100 * cv.get('ngs_time_to_throw present (share of scrimmage plays)', np.nan):.1f}%"
    dec_txt = res.get("decomposition_text") or decomposition_text(res, cfg)
    return f"""* **Sample.** 91 games / 6 weeks of one season. Team target encodings and tendencies are estimated from at most 5 prior
  games per team; the k-fold protocol lets a team's other 2017 games inform its encoding (legitimate for imputing the rest of a
  tracked season, optimistic for a new season), the forward split is the honest in-season number and the 2018 external check
  the honest cross-season number. `te_off` / `te_def` are target means over the team's other tracked games, so part of the F0
  skill on box / deep-safety / cushion targets is team identity (compare the `F0n` column, which has no target encoding).
* {dec_txt}
* **Early stopping** uses an inner holdout of training games, so each fold's model is trained on ~{int((1 - cfg.inner_holdout_frac) * 100)}% of the
  training fold; the final students are refitted on all labelled plays with the inner-holdout round count.
* **Targets are themselves definitions** (NFL 01): `box_count_tuned` agrees exactly with the official box count on 71% of plays,
  `n_pass_rushers_derived` on 80%, `pressure_derived` agrees with `was_pressure` on {pressure_txt}; `separation_at_arrival` depends on the
  derived target identification (93.6% agreement with the pbp receiver). Students learn the derived quantity, not the official one.
* **`base_play_type` and `base_outcome` use post-snap information** (nflfastR play type / outcome flags) and are therefore
  F1-level baselines, listed for reference next to the pre-snap F0 students.
* **F1 on pre-snap targets** is the after-the-fact setting: e.g. `air_yards` / `pass_location` reveal where the ball went, which
  is informative about the pre-snap shell; it is not a forecast.
* **`shotgun_derived` under F0** is close to a tautology because nflfastR's `shotgun` is parsed from the play description; it is
  kept as a pipeline check. `qb_depth` inherits most of that signal.
{cal_txt}* **2018 application.** Team encodings come from 2017; the 2018 `pbp_participation` file has formation on {cov_txt} of
  scrimmage plays (personnel on 100%) and the pass-play NGS fields on {ngs_txt} (missing -> NaN features, LightGBM default branch);
  the official fields used for the external check (`defenders_in_box`, `number_of_pass_rushers`, `time_to_throw`, `was_pressure`,
  `ngs_air_yards`) are never features of the F0 / F0P / F1 students. Agreement on 2018 is measured against the official field, not
  against the derived target, so the 2017 ceiling rows bound what a perfect student could reach. The official `time_to_throw` /
  `was_pressure` are absent on every sack (section 9a), so their external rows are non-sack plays only.
* **Highlight-play check.** The optional out-of-distribution check on 150 NGS highlight plays (2018-19; touchdowns and
  long gains only, a biased sample) is a separate script, `python -m research.privileged_tracking.nfl.ood_highlights`,
  reported in `nfl_03_ood_highlights.md` next to this file.
* **Compute.** {cfg.n_jobs} LightGBM threads, {res['seconds'] / 60:.1f} min end to end; no subsampling.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 targets, 200 bootstraps, nothing written")
    ap.add_argument("--targets", default=None, help="comma-separated target names")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-apply", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()
    cfg = ImputationConfig(n_jobs=args.n_jobs)
    if args.quick:
        cfg.targets = ("mof_open", "box_count_tuned", "separation_at_arrival")
        cfg.n_boot = 200
        cfg.write = False
    if args.targets:
        cfg.targets = tuple(s.strip() for s in args.targets.split(","))
    if args.no_write:
        cfg.write = False
    if args.no_apply:
        cfg.apply_season = None
    res = run(cfg)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 100):
        print(res["ranking"][["target", "kind", "n_kfold"] + [f"{f}_skill" for f in cfg.feature_sets] + ["F0_skill_forward", "F1_skill_forward", "oracle_skill"]])
    print(f"done in {res['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
