"""Soccer 03 - teacher-student payoff: xG and xPass in the 360 matches.

Does imputed defensive state (the stage-02 students, out-of-fold) improve an event-only shot
model (xG) or pass-completion model (xPass), how far is it from the 360 / shot-freeze-frame
oracles, does distilling an oracle teacher into an event-only student beat training on goals,
does the gain survive a cross-gender domain shift, and where (play pattern / distance /
reliability) does the imputed state help?

Stages (run from the repo root, each well under 25 min on 2 threads; every stage caches its
output under ``processed_dir('soccer')/payoff_cache/`` and is skipped when the cache exists)::

    python -m research.privileged_tracking.soccer.payoff --stage shots      # shot table + assists
    python -m research.privileged_tracking.soccer.payoff --stage xg         # (a) xG variants
    python -m research.privileged_tracking.soccer.payoff --stage distill    # (a) distillation
    python -m research.privileged_tracking.soccer.payoff --stage passes     # (b) pass table
    python -m research.privileged_tracking.soccer.payoff --stage xpass      # (b) xPass variants
    python -m research.privileged_tracking.soccer.payoff --stage students   # (c) gender students
    python -m research.privileged_tracking.soccer.payoff --stage transfer   # (c) men <-> women xG
    python -m research.privileged_tracking.soccer.payoff --stage final      # xG models for transfer
    python -m research.privileged_tracking.soccer.payoff --stage report     # (d) slices + report

``--stage all`` runs them in order; ``--smoke`` runs everything on a few matches in a separate
cache without writing reports; ``--force`` refits a stage.

Protocol: folds are the stage-02 ``fold`` column (5-fold ``group_kfold`` by match, seed 0), so
every imputed feature of a test-fold row comes from students that never saw that fold; LightGBM
rounds are chosen on an inner match-grouped holdout of the training rows and the model is refitted
on all training rows (xG) ; xG variants average ``n_seeds_xg`` seeds; deltas are paired
per-shot bootstraps (plus a match-clustered CI). See ``payoff_features.py`` for the feature
catalogue and the leakage contract.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from pathlib import Path

from research.privileged_tracking.common.io import processed_dir, reports_dir, sb_dir
from research.privileged_tracking.common.metrics import mae, r2
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.soccer import imputation as imp
from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.soccer import payoff_features as pf
from research.privileged_tracking.soccer.payoff_features import (
    ASSIST_COLS,
    ASSIST_STATE,
    PASS_STATE,
    SFF_FULL,
    SHOT_STATE,
    XG_REFERENCE,
    XG_VARIANTS,
    XPASS_VARIANTS,
    GbmParams,
    PayoffConfig,
)

CACHE_SUBDIR = "payoff_cache"
SHOT_TABLE = "payoff_shots.parquet"
PASS_TABLE = "payoff_passes.parquet"
KEY_PASS_TABLE = "shot_key_pass.parquet"
XG_PREDS = "xg_preds.parquet"
XG_IMPORTANCE = "xg_importance.parquet"
XG_FITS = "xg_fits.parquet"
DISTILL_PREDS = "xg_distill_preds.parquet"
XPASS_PREDS = "xpass_preds.parquet"
XPASS_FITS = "xpass_fits.parquet"
TRANSFER_PREDS = "xg_transfer_preds.parquet"
CURVE_PREDS = "xg_curve_preds.parquet"
XG_MODELS = "xg_models.joblib"
#: variants saved by the ``final`` stage for application to other matches
FINAL_VARIANTS = ("EVENT", "EVENT+IMP", "EVENT+ORACLESHOT")
REPORT_PREFIX = "soccer_03"
SMOKE_MATCHES = 40

SHOT_ID_COLS = [
    "match_id",
    "event_id",
    "event_index",
    "period",
    "competition",
    "season",
    "gender",
    "match_date",
    "team_id",
    "player_id",
]
SHOT_Y_COLS = [
    "y_frame_ok",
    "y_reliable",
    "y_keeper_consistent",
    "y_n_opponents_visible",
    "y_frame_orientation",
]
PASS_Y_COLS = ["y_frame_ok", "y_reliable", "y_nearest_opp_to_receiver", "y_n_opponents_visible"]
#: pass-state quantities with a stage-02 student (the receiver distance is fitted here)
PASS_STATE_STAGE02 = tuple(t for t in PASS_STATE if t != "nearest_opp_to_receiver")
#: distillation runs: (teacher variant, student variant, alpha = teacher weight in the label)
DISTILL_RUNS: tuple[tuple[str, str, float], ...] = (
    ("EVENT+ORACLE360", "EVENT", 1.0),
    ("EVENT+ORACLE360", "EVENT", 0.5),
    ("EVENT+ORACLESHOT", "EVENT", 1.0),
    ("EVENT+ORACLE360", "EVENT+IMP", 1.0),
)
STUDENT_SPECS: tuple[tuple[str, str], ...] = tuple((t, "E2") for t in SHOT_STATE) + tuple(
    (t, "E2a") for t in ASSIST_STATE
)
TRANSFER_VARIANTS = ("EVENT", "EVENT+IMP", "EVENT+ORACLE360", "EVENT+ORACLESHOT")
#: learning curve: share of the training matches kept (test folds stay complete)
CURVE_FRACS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
CURVE_VARIANTS = ("EVENT", "EVENT+IMP", "EVENT+ORACLE360", "EVENT+ORACLESHOT")


def cache_dir(cfg: PayoffConfig) -> Path:
    d = processed_dir("soccer") / (CACHE_SUBDIR + ("_smoke" if cfg.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _events_path() -> Path:
    return processed_dir("soccer") / "events360.parquet"


def _oof_path() -> Path:
    return processed_dir("soccer") / "imputed_oof.parquet"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_key_pass_links(cfg: PayoffConfig, match_ids: np.ndarray) -> pd.DataFrame:
    """``(event_id, key_pass_id)`` for every shot of the given matches, read once from the raw
    StatsBomb event files and cached."""
    path = cache_dir(cfg) / KEY_PASS_TABLE
    if path.exists():
        d = pd.read_parquet(path)
        if set(match_ids) <= set(d["match_id"].unique()):
            return d[d["match_id"].isin(match_ids)]
    frames = []
    for mid in sorted(int(m) for m in match_ids):
        p = sb_dir() / "events" / f"{mid}.json"
        if not p.exists():
            continue
        with open(p) as f:
            ev = json.load(f)
        d = pf.key_pass_links(ev)
        d["match_id"] = mid
        frames.append(d)
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["event_id", "key_pass_id", "match_id"])
    )
    out.to_parquet(path, index=False)
    return out


def _smoke_matches(match_ids: np.ndarray) -> np.ndarray:
    return np.sort(np.unique(match_ids))[:SMOKE_MATCHES]


def build_shot_table(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """Shots of the 360 matches with event features, assist attributes, out-of-fold imputations,
    360 labels and shot-frame quantities (cached ``payoff_shots.parquet``).

    Columns: ids, ``f_*``, ``sff_*``, ``oracle_*``, ``post_shot_outcome``, ``y_*`` flags,
    ``fold``, ``imp_<t>__{E0,E2}`` / ``y_<t>`` for :data:`SHOT_STATE`, ``key_pass_id``, the
    ``a_*`` assist attributes and ``a_imp_<t>__{E0,E2a}`` / ``a_y_<t>`` for :data:`ASSIST_STATE`.
    """
    path = cache_dir(cfg) / SHOT_TABLE
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    ev = _events_path()
    names = pq.ParquetFile(ev).schema_arrow.names
    cols = (
        SHOT_ID_COLS
        + [c for c in names if c.startswith(("f_", "sff_", "oracle_"))]
        + ["post_shot_outcome"]
        + SHOT_Y_COLS
    )
    shots = pq.read_table(ev, columns=cols, filters=[("f_type", "==", "Shot")]).to_pandas()
    if cfg.smoke:
        shots = shots[shots["match_id"].isin(_smoke_matches(shots["match_id"].to_numpy()))]
    shots = shots.sort_values(["match_id", "event_index"]).reset_index(drop=True)
    # out-of-fold imputations and 360 labels of the shot itself
    oof_cols = (
        ["event_id", "fold"]
        + [f"{t}__{f}" for t in SHOT_STATE for f in ("E0", "E2")]
        + [f"y_{t}" for t in SHOT_STATE]
    )
    oof = pq.read_table(
        _oof_path(), columns=oof_cols, filters=[("f_type", "==", "Shot")]
    ).to_pandas()
    oof = oof.rename(
        columns={f"{t}__{f}": f"imp_{t}__{f}" for t in SHOT_STATE for f in ("E0", "E2")}
    )
    shots = shots.merge(oof, on="event_id", how="left")
    # assist link and the key pass's attributes
    links = load_key_pass_links(cfg, shots["match_id"].unique())
    shots = shots.merge(links[["event_id", "key_pass_id"]], on="event_id", how="left")
    kp = set(shots["key_pass_id"].dropna().astype(str))
    passes = pq.read_table(
        ev, columns=["event_id", *ASSIST_COLS], filters=[("f_type", "==", "Pass")]
    ).to_pandas()
    passes = passes[passes["event_id"].isin(kp)]
    assist = pf.assist_block(shots, passes)
    shots = pd.concat([shots, assist], axis=1)
    poof_cols = (
        ["event_id"]
        + [f"{t}__{f}" for t in ASSIST_STATE for f in ("E0", "E2a")]
        + [f"y_{t}" for t in ASSIST_STATE]
    )
    poof = pq.read_table(
        _oof_path(), columns=poof_cols, filters=[("f_type", "==", "Pass")]
    ).to_pandas()
    poof = poof[poof["event_id"].isin(kp)].rename(
        columns={
            **{f"{t}__{f}": f"a_imp_{t}__{f}" for t in ASSIST_STATE for f in ("E0", "E2a")},
            **{f"y_{t}": f"a_y_{t}" for t in ASSIST_STATE},
            "event_id": "key_pass_id",
        }
    )
    shots = shots.merge(poof, on="key_pass_id", how="left")
    shots["is_goal"] = (shots["post_shot_outcome"].to_numpy(dtype=object) == "Goal").astype(
        np.float32
    )
    shots.to_parquet(path, index=False)
    _log(
        f"shot table: {len(shots):,} shots, {int(shots['is_goal'].sum()):,} goals, "
        f"{int(shots['a_has_assist'].sum()):,} with an assist, {time.time() - t0:.1f} s"
    )
    return shots


def shot_population(cfg: PayoffConfig, shots: pd.DataFrame) -> pd.DataFrame:
    """The modelled shots: penalties dropped (no freeze frame, fixed xG) unless configured."""
    if cfg.exclude_penalties:
        shots = shots[shots["f_shot_type"].to_numpy(dtype=object) != "Penalty"]
    return shots.reset_index(drop=True)


def build_pass_table(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """Match-stratified pass subsample with event features, out-of-fold imputations and 360
    labels of :data:`PASS_STATE` (cached ``payoff_passes.parquet``).

    The receiver-distance student (``nearest_opp_to_receiver``, no stage-02 student) is fitted
    here out-of-fold on ALL Pass rows (stage-02 folds, E2a and E2 designs, stage-02 LightGBM
    settings) before the subsample is taken.
    """
    path = cache_dir(cfg) / PASS_TABLE
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    ev = _events_path()
    names = pq.ParquetFile(ev).schema_arrow.names
    cols = (
        SHOT_ID_COLS
        + [c for c in names if c.startswith("f_")]
        + ["post_pass_outcome"]
        + PASS_Y_COLS
    )
    passes = pq.read_table(ev, columns=cols, filters=[("f_type", "==", "Pass")]).to_pandas()
    if cfg.smoke:
        passes = passes[passes["match_id"].isin(_smoke_matches(passes["match_id"].to_numpy()))]
    passes = passes.sort_values(["match_id", "event_index"]).reset_index(drop=True)
    oof_cols = (
        ["event_id", "fold"]
        + [f"{t}__{f}" for t in PASS_STATE_STAGE02 for f in ("E2", "E2a")]
        + [f"y_{t}" for t in PASS_STATE_STAGE02]
    )
    oof = pq.read_table(
        _oof_path(), columns=oof_cols, filters=[("f_type", "==", "Pass")]
    ).to_pandas()
    oof = oof.rename(
        columns={f"{t}__{f}": f"imp_{t}__{f}" for t in PASS_STATE_STAGE02 for f in ("E2", "E2a")}
    )
    passes = passes.merge(oof, on="event_id", how="left")
    y, keep = pf.pass_completion(passes["post_pass_outcome"].to_numpy(dtype=object))
    passes["is_complete"] = y.astype(np.float32)
    passes = passes[keep].reset_index(drop=True)
    _log(
        f"pass rows: {len(passes):,} attempts ({time.time() - t0:.1f} s); "
        "fitting the receiver student"
    )
    # receiver-distance student, out-of-fold on the stage-02 folds
    icfg = imp.ImputationConfig(n_jobs=cfg.n_jobs, seed=cfg.seed, train_cap=300_000)
    label = passes["y_nearest_opp_to_receiver"].to_numpy(dtype=float)
    label[passes["y_frame_ok"].to_numpy(dtype=float) != 1.0] = np.nan
    match = passes["match_id"].to_numpy()
    fold = passes["fold"].to_numpy()
    fits = []
    for fset in ("E2a", "E2"):
        x = imf.build_design(passes, fset)
        cats = imf.categorical_columns(fset)
        pred = np.full(len(passes), np.nan, dtype=np.float32)
        for k in np.unique(fold[fold >= 0]):
            te = np.where(fold == k)[0]
            tr = np.where((fold != k) & ~np.isnan(label))[0]
            tr = imp.thin_rows(tr, icfg.train_cap, cfg.seed + int(k))
            t1 = time.time()
            booster, rounds = imp.fit_lgbm(
                icfg, x.iloc[tr], label[tr], match[tr], "reg", cats, cfg.seed + int(k)
            )
            pred[te] = booster.predict(x.iloc[te], num_iteration=rounds)
            fits.append(
                {
                    "target": "nearest_opp_to_receiver",
                    "fset": fset,
                    "fold": int(k),
                    "rounds": rounds,
                    "n_train": int(len(tr)),
                    "seconds": time.time() - t1,
                }
            )
        passes[f"imp_nearest_opp_to_receiver__{fset}"] = pred
        ok = ~np.isnan(label) & ~np.isnan(pred)
        _log(
            f"receiver student {fset}: oof R2 {r2(label[ok], pred[ok]):.3f}, "
            f"MAE {mae(label[ok], pred[ok]):.2f} yd (n {ok.sum():,})"
        )
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / "receiver_student_fits.parquet", index=False)
    passes["y_nearest_opp_to_receiver"] = label.astype(np.float32)
    keep = pf.stratified_subsample(match, cfg.xpass_target_n, cfg.seed)
    passes = passes[keep].reset_index(drop=True)
    passes.to_parquet(path, index=False)
    _log(f"pass table: {len(passes):,} passes in the subsample, {time.time() - t0:.1f} s")
    return passes


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def lgb_params(p: GbmParams, objective: str, seed: int, n_threads: int) -> dict[str, Any]:
    return {
        "objective": objective,
        "learning_rate": p.learning_rate,
        "num_leaves": p.num_leaves,
        "min_data_in_leaf": p.min_data_in_leaf,
        "feature_fraction": p.feature_fraction,
        "bagging_fraction": p.bagging_fraction,
        "bagging_freq": 1,
        "max_bin": p.max_bin,
        "lambda_l2": p.lambda_l2,
        "num_threads": n_threads,
        "verbose": -1,
        "seed": seed,
    }


def fit_gbm(
    p: GbmParams,
    x: pd.DataFrame,
    y: np.ndarray,
    match: np.ndarray,
    seed: int,
    categorical: list[str],
    objective: str = "binary",
    n_threads: int = 2,
    refit: bool = True,
) -> tuple[lgb.Booster, int]:
    """Fit one downstream model.

    Rounds are chosen by early stopping on an inner match-grouped holdout
    (``p.inner_holdout_frac`` of the training matches); with ``refit`` the model is then
    refitted on all training rows with that round count.

    Args:
        x: training design ``[n_train, d]``; y: labels ``[n_train]`` (soft labels allowed with
            ``objective='cross_entropy'``); match: match id per row ``[n_train]``.

    Returns:
        ``(booster, rounds)``.
    """
    params = lgb_params(p, objective, seed, n_threads)
    rng = np.random.default_rng(seed)
    uniq = np.unique(match)
    hold = rng.choice(
        uniq, size=max(1, int(round(len(uniq) * p.inner_holdout_frac))), replace=False
    )
    is_hold = np.isin(match, hold)
    dtrain = lgb.Dataset(
        x[~is_hold], y[~is_hold], categorical_feature=categorical, free_raw_data=False
    )
    dvalid = lgb.Dataset(x[is_hold], y[is_hold], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=p.max_rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(p.early_stopping, verbose=False)],
    )
    best = int(booster.best_iteration) if booster.best_iteration else p.max_rounds
    if not refit:
        return booster, best
    booster = lgb.train(
        params, lgb.Dataset(x, y, categorical_feature=categorical), num_boost_round=max(best, 1)
    )
    return booster, best


def cv_predict(
    x: pd.DataFrame,
    label: np.ndarray,
    match: np.ndarray,
    fold: np.ndarray,
    p: GbmParams,
    seeds: tuple[int, ...],
    categorical: list[str],
    objective: str = "binary",
    n_threads: int = 2,
    refit: bool = True,
    train_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], pd.DataFrame]:
    """Out-of-fold predictions ``[n]`` averaged over ``seeds``, fit metadata and gain shares.

    Args:
        train_mask: optional rows allowed in training (the test folds are always complete).

    Returns:
        ``(oof, fits, importance)`` where ``importance`` holds the mean gain share per feature.
    """
    n = len(x)
    oof = np.full(n, np.nan, dtype=float)
    fits: list[dict[str, Any]] = []
    gains = np.zeros(x.shape[1], dtype=float)
    n_fits = 0
    for k in np.unique(fold):
        te = np.where(fold == k)[0]
        tr = np.where(fold != k)[0]
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        acc = np.zeros(len(te), dtype=float)
        for s in seeds:
            t0 = time.time()
            booster, rounds = fit_gbm(
                p,
                x.iloc[tr],
                label[tr],
                match[tr],
                s + 17 * int(k),
                categorical,
                objective,
                n_threads,
                refit,
            )
            acc += booster.predict(x.iloc[te], num_iteration=rounds)
            g = booster.feature_importance("gain")
            gains += g / max(g.sum(), 1e-12)
            n_fits += 1
            fits.append(
                {
                    "fold": int(k),
                    "seed": int(s),
                    "rounds": rounds,
                    "n_train": int(len(tr)),
                    "seconds": time.time() - t0,
                }
            )
        oof[te] = acc / len(seeds)
    importance = pd.DataFrame({"feature": list(x.columns), "gain_share": gains / max(n_fits, 1)})
    return oof, fits, importance


def base_rate_oof(y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Fold-wise training positive rate for every row ``[n]``."""
    out = np.full(len(y), np.nan, dtype=float)
    for k in np.unique(fold):
        te = fold == k
        out[te] = float(np.mean(y[~te]))
    return out


# ---------------------------------------------------------------------------
# Designs per variant
# ---------------------------------------------------------------------------


def xg_designs(shots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Design matrix of every :data:`XG_VARIANTS` entry with a model (``BASE`` excluded)."""
    assist = shots[
        [c for c in shots.columns if c.startswith("a_") and not c.startswith(("a_imp_", "a_y_"))]
    ]
    base_event = pf.shot_event_design(shots, assist)
    base_loc = pf.drop_constant_columns(imf.build_design(shots, "loc"))
    out: dict[str, pd.DataFrame] = {}
    for v in XG_VARIANTS:
        if v.base == "none":
            continue
        blocks = [base_loc if v.base == "loc" else base_event]
        if v.state == "imp":
            a_fset = "E0" if v.fset == "E0" else "E2a"
            blocks.append(pf.state_block(shots, SHOT_STATE, "imp", v.fset, prefix="s_"))
            blocks.append(
                pf.state_block(shots, ASSIST_STATE, "imp", a_fset, prefix="as_", col_prefix="a_")
            )
        elif v.state == "oracle360":
            blocks.append(pf.state_block(shots, SHOT_STATE, "oracle360", prefix="s_"))
            blocks.append(
                pf.state_block(shots, ASSIST_STATE, "oracle360", prefix="as_", col_prefix="a_")
            )
        elif v.state == "oracleshot":
            blocks.append(pf.state_block(shots, SHOT_STATE, "oracleshot", prefix="s_"))
            blocks.append(
                pf.state_block(shots, ASSIST_STATE, "oracle360", prefix="as_", col_prefix="a_")
            )
        elif v.state == "oracleshot_full":
            blocks.append(shots[list(SFF_FULL)].astype(np.float32).add_prefix("s_"))
            blocks.append(
                pf.state_block(shots, ASSIST_STATE, "oracle360", prefix="as_", col_prefix="a_")
            )
        out[v.name] = pd.concat(blocks, axis=1)
    return out


def xpass_designs(passes: pd.DataFrame, with_after: bool) -> dict[str, pd.DataFrame]:
    """Design matrix of every :data:`XPASS_VARIANTS` entry with a model, for one design mode."""
    base = pf.pass_event_design(passes, with_after)
    fset = "E2a" if with_after else "E2"
    out: dict[str, pd.DataFrame] = {}
    for v in XPASS_VARIANTS:
        if v.base == "none":
            continue
        if v.name == "EVENT-nodur":
            if with_after:
                out[v.name] = base.drop(columns=["f_after_duration"], errors="ignore")
            continue
        blocks = [base]
        if v.state == "imp":
            blocks.append(pf.state_block(passes, PASS_STATE, "imp", fset, prefix="s_"))
        elif v.state == "oracle360":
            blocks.append(pf.state_block(passes, PASS_STATE, "oracle360", prefix="s_"))
        out[v.name] = pd.concat(blocks, axis=1)
    return out


def _seeds(cfg: PayoffConfig, n: int) -> tuple[int, ...]:
    return tuple(cfg.seed + i for i in range(n))


XG_CONTEXT = [
    "event_id",
    "match_id",
    "fold",
    "gender",
    "competition",
    "season",
    "f_play_pattern",
    "f_dist_goal",
    "f_x",
    "f_y",
    "f_shot_type",
    "f_shot_body_part",
    "y_frame_ok",
    "y_reliable",
    "y_n_opponents_visible",
    "sff_present",
    "a_has_assist",
    "oracle_statsbomb_xg",
    "is_goal",
]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_xg(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """(a) xG variants, 5-fold out-of-fold; writes ``xg_preds.parquet`` (+ fits, importance)."""
    path = cache_dir(cfg) / XG_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    fold = shots["fold"].to_numpy()
    designs = xg_designs(shots)
    preds = shots[XG_CONTEXT].copy()
    preds["pred_BASE"] = base_rate_oof(y, fold)
    preds[f"pred_{XG_REFERENCE}"] = shots["oracle_statsbomb_xg"].to_numpy(dtype=float)
    fits, imps = [], []
    for v in XG_VARIANTS:
        if v.name not in designs:
            continue
        t0 = time.time()
        x = designs[v.name]
        oof, f, importance = cv_predict(
            x,
            y,
            match,
            fold,
            cfg.xg,
            _seeds(cfg, cfg.n_seeds_xg),
            pf.categorical_in(x),
            "binary",
            cfg.n_jobs,
            refit=True,
        )
        preds[f"pred_{v.name}"] = oof
        fits += [{"variant": v.name, **r} for r in f]
        importance["variant"] = v.name
        imps.append(importance)
        m = pf.binary_metrics(y, oof)
        _log(
            f"xg {v.name:24s} d={x.shape[1]:3d} log-loss {m['log_loss']:.4f} "
            f"brier {m['brier']:.4f} "
            f"auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
        )
    preds.to_parquet(path, index=False)
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / XG_FITS, index=False)
    pd.concat(imps, ignore_index=True).to_parquet(cache_dir(cfg) / XG_IMPORTANCE, index=False)
    return preds


def distill_oof(
    x_teacher: pd.DataFrame,
    x_student: pd.DataFrame,
    y: np.ndarray,
    match: np.ndarray,
    fold: np.ndarray,
    p: GbmParams,
    seeds: tuple[int, ...],
    alpha: float,
    n_threads: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Nested distillation: for every outer test fold, the teacher's predictions on the training
    rows come from an inner CV over the training folds only (so no test-fold label reaches the
    student through the teacher); the student is fitted on ``soft_label(y, teacher, alpha)`` with
    the cross-entropy objective and predicts the test fold.

    Returns:
        ``(student_oof, teacher_inner_oof_mean)``: the student's out-of-fold probabilities ``[n]``
        and, for diagnostics, the average inner-teacher prediction each training row received.
    """
    n = len(y)
    student = np.full(n, np.nan, dtype=float)
    teacher_seen = np.zeros(n, dtype=float)
    teacher_cnt = np.zeros(n, dtype=float)
    cats_t, cats_s = pf.categorical_in(x_teacher), pf.categorical_in(x_student)
    for k in np.unique(fold):
        te = np.where(fold == k)[0]
        tr = np.where(fold != k)[0]
        inner = np.full(len(tr), np.nan, dtype=float)
        for j in np.unique(fold[tr]):
            ite = fold[tr] == j
            itr = ~ite
            acc = np.zeros(int(ite.sum()), dtype=float)
            for s in seeds:
                b, r = fit_gbm(
                    p,
                    x_teacher.iloc[tr[itr]],
                    y[tr[itr]],
                    match[tr[itr]],
                    s + 17 * int(k) + 101 * int(j),
                    cats_t,
                    "binary",
                    n_threads,
                    True,
                )
                acc += b.predict(x_teacher.iloc[tr[ite]], num_iteration=r)
            inner[ite] = acc / len(seeds)
        teacher_seen[tr] += inner
        teacher_cnt[tr] += 1
        label = pf.soft_label(y[tr], inner, alpha)
        acc = np.zeros(len(te), dtype=float)
        for s in seeds:
            b, r = fit_gbm(
                p,
                x_student.iloc[tr],
                label,
                match[tr],
                s + 17 * int(k),
                cats_s,
                "cross_entropy",
                n_threads,
                True,
            )
            acc += b.predict(x_student.iloc[te], num_iteration=r)
        student[te] = acc / len(seeds)
    return student, teacher_seen / np.maximum(teacher_cnt, 1)


def stage_distill(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """(a) distillation variants; writes ``xg_distill_preds.parquet``."""
    path = cache_dir(cfg) / DISTILL_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    fold = shots["fold"].to_numpy()
    designs = xg_designs(shots)
    preds = shots[["event_id", "match_id", "fold", "gender", "is_goal"]].copy()
    for teacher, student, alpha in DISTILL_RUNS:
        t0 = time.time()
        name = f"{student}<-{teacher}@{alpha:g}"
        oof, seen = distill_oof(
            designs[teacher],
            designs[student],
            y,
            match,
            fold,
            cfg.xg,
            _seeds(cfg, cfg.n_seeds_xg),
            alpha,
            cfg.n_jobs,
        )
        preds[f"pred_{name}"] = oof
        preds[f"teacher_inner_{name}"] = seen
        m = pf.binary_metrics(y, oof)
        _log(
            f"distill {name:40s} log-loss {m['log_loss']:.4f} auc {m['auc']:.3f}  "
            f"{time.time() - t0:.0f} s"
        )
    preds.to_parquet(path, index=False)
    return preds


XPASS_CONTEXT = [
    "event_id",
    "match_id",
    "fold",
    "gender",
    "competition",
    "f_play_pattern",
    "f_pass_type",
    "f_after_pass_height",
    "f_after_pass_length",
    "f_x",
    "f_under_pressure",
    "y_frame_ok",
    "y_reliable",
    "is_complete",
]


def stage_xpass(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """(b) xPass variants for the with-after and no-after designs; writes
    ``xpass_preds.parquet``."""
    path = cache_dir(cfg) / XPASS_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    passes = build_pass_table(cfg)
    y = passes["is_complete"].to_numpy(dtype=float)
    match = passes["match_id"].to_numpy()
    fold = passes["fold"].to_numpy()
    preds = passes[XPASS_CONTEXT].copy()
    preds["pred_BASE"] = base_rate_oof(y, fold)
    fits, imps = [], []
    for with_after in (True, False):
        tag = "after" if with_after else "noafter"
        designs = xpass_designs(passes, with_after)
        for v in XPASS_VARIANTS:
            if v.name not in designs:
                continue
            t0 = time.time()
            x = designs[v.name]
            oof, f, importance = cv_predict(
                x,
                y,
                match,
                fold,
                cfg.xpass,
                _seeds(cfg, cfg.n_seeds_xpass),
                pf.categorical_in(x),
                "binary",
                cfg.n_jobs,
                refit=False,
            )
            preds[f"pred_{v.name}__{tag}"] = oof
            fits += [{"variant": v.name, "design": tag, **r} for r in f]
            importance["variant"] = f"{v.name}__{tag}"
            imps.append(importance)
            m = pf.binary_metrics(y, oof)
            _log(
                f"xpass {v.name:16s} {tag:8s} d={x.shape[1]:3d} log-loss {m['log_loss']:.4f} "
                f"brier {m['brier']:.4f} auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
            )
    preds.to_parquet(path, index=False)
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / XPASS_FITS, index=False)
    pd.concat(imps, ignore_index=True).to_parquet(
        cache_dir(cfg) / "xpass_importance.parquet", index=False
    )
    return preds


def stage_curve(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """(d) learning curve: the xG variants refitted with only a share of the training MATCHES
    (test folds complete), to see whether the imputed state helps when goal labels are scarce;
    writes ``xg_curve_preds.parquet``."""
    path = cache_dir(cfg) / CURVE_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    fold = shots["fold"].to_numpy()
    designs = xg_designs(shots)
    preds = shots[["event_id", "match_id", "fold", "gender", "is_goal"]].copy()
    matches = np.unique(match)
    for frac in CURVE_FRACS:
        rng = np.random.default_rng(cfg.seed + int(frac * 1000))
        keep = rng.choice(matches, size=max(5, int(round(len(matches) * frac))), replace=False)
        train_mask = np.isin(match, keep)
        for name in CURVE_VARIANTS:
            t0 = time.time()
            x = designs[name]
            oof, fits, _ = cv_predict(
                x,
                y,
                match,
                fold,
                cfg.xg,
                _seeds(cfg, cfg.n_seeds_xg),
                pf.categorical_in(x),
                "binary",
                cfg.n_jobs,
                refit=True,
                train_mask=train_mask,
            )
            preds[f"pred_{name}__{frac:g}"] = oof
            m = pf.binary_metrics(y, oof)
            n_tr = int(np.mean([f["n_train"] for f in fits]))
            _log(
                f"curve frac {frac:4.2f} {name:18s} n_train~{n_tr:5d} log-loss {m['log_loss']:.4f} "
                f"auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
            )
    preds.to_parquet(path, index=False)
    return preds


@dataclass
class XgModel:
    """One xG variant fitted on every non-penalty shot of the 360 matches (seed-averaged).

    Attributes:
        variant: :data:`payoff_features.XG_VARIANTS` name.
        features / categorical: design columns (build them with :func:`xg_designs` on a shot
            table carrying the same ``imp_*`` / ``sff_*`` / ``a_*`` columns).
        model_strs: LightGBM model text per seed; :meth:`predict` averages them.
        rounds: boosting rounds (median of the CV early-stopping rounds).
        train_n / goal_rate: training rows and their goal rate.
    """

    variant: str
    features: list[str]
    categorical: list[str]
    model_strs: list[str]
    rounds: int
    train_n: int
    goal_rate: float

    def predict(self, design: pd.DataFrame) -> np.ndarray:
        """Goal probability ``[n]`` from a design holding ``features`` (extra columns ignored)."""
        x = design[self.features]
        acc = np.zeros(len(x), dtype=float)
        for m in self.model_strs:
            acc += lgb.Booster(model_str=m).predict(x, num_iteration=self.rounds)
        return acc / max(len(self.model_strs), 1)


@dataclass
class XgBundle:
    """All :data:`FINAL_VARIANTS` models plus the config they were fitted with."""

    models: dict[str, XgModel] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def path(root: Path | None = None) -> Path:
        d = (root or processed_dir("soccer")) / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d / XG_MODELS

    def save(self, root: Path | None = None) -> Path:
        p = self.path(root)
        joblib.dump(
            {"models": {k: asdict(v) for k, v in self.models.items()}, "params": self.params}, p
        )
        return p

    @classmethod
    def load(cls, root: Path | None = None) -> XgBundle:
        d = joblib.load(cls.path(root))
        return cls(models={k: XgModel(**v) for k, v in d["models"].items()}, params=d["params"])


def stage_final(cfg: PayoffConfig, force: bool = False) -> XgBundle:
    """Fit :data:`FINAL_VARIANTS` on every modelled shot (rounds = median CV rounds of that
    variant, ``n_seeds_xg`` seeds) and save ``models/xg_models.joblib`` for the transfer stage."""
    path = XgBundle.path()
    if path.exists() and not force and not cfg.smoke:
        return XgBundle.load()
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    designs = xg_designs(shots)
    fits_p = cache_dir(cfg) / XG_FITS
    fits = pd.read_parquet(fits_p) if fits_p.exists() else pd.DataFrame()
    bundle = XgBundle(
        params={
            "xg": asdict(cfg.xg),
            "n_seeds": cfg.n_seeds_xg,
            "exclude_penalties": cfg.exclude_penalties,
        }
    )
    for name in FINAL_VARIANTS:
        x = designs[name]
        cats = pf.categorical_in(x)
        if len(fits) and (fits["variant"] == name).any():
            rounds = int(fits.loc[fits["variant"] == name, "rounds"].median())
        else:
            rounds = 150
        strs = []
        for s in _seeds(cfg, cfg.n_seeds_xg):
            params = lgb_params(cfg.xg, "binary", s + 99, cfg.n_jobs)
            booster = lgb.train(
                params, lgb.Dataset(x, y, categorical_feature=cats), num_boost_round=rounds
            )
            strs.append(booster.model_to_string())
        bundle.models[name] = XgModel(
            name, list(x.columns), cats, strs, rounds, int(len(y)), float(y.mean())
        )
        _log(f"final {name:18s} rounds {rounds} d={x.shape[1]} n={len(y):,}")
    if cfg.write and not cfg.smoke:
        bundle.save()
        _log(f"saved {path}")
    return bundle


def student_path(cfg: PayoffConfig, domain: str, target: str, fset: str) -> Path:
    return cache_dir(cfg) / f"student__{domain}__{target}__{fset}.parquet"


def stage_students(cfg: PayoffConfig, force: bool = False) -> None:
    """(c) domain-restricted students: for each gender, 5-fold out-of-fold predictions of the
    shot-state (E2) and assist pass-lane (E2a) quantities from students trained on that gender's
    matches only (stage-02 ``run_fit``); saved as ``(event_id, pred)`` for Shot / Pass rows."""
    icfg = imp.ImputationConfig(
        n_jobs=cfg.n_jobs, seed=cfg.seed, train_cap=cfg.student_train_cap, smoke=cfg.smoke
    )
    todo = [
        (d, t, f)
        for d in ("male", "female")
        for t, f in STUDENT_SPECS
        if force or not student_path(cfg, d, t, f).exists()
    ]
    if not todo:
        _log("students: all cached")
        return
    t0 = time.time()
    data = imp.load_data(icfg)
    _log(f"students: loaded {len(data.df):,} rows in {time.time() - t0:.0f} s")
    gender = data.df["gender"].to_numpy(dtype=object)
    ftype = data.df["f_type"].to_numpy(dtype=object)
    eid = data.df["event_id"].to_numpy(dtype=object)
    keep_type = np.isin(ftype, ["Shot", "Pass"])
    for domain in ("male", "female"):
        rows = np.where(gender == domain)[0]
        if len(rows) == 0:
            continue
        folds = list(group_kfold(data.match[rows], n_splits=5, seed=cfg.seed))
        for target, fset in STUDENT_SPECS:
            if (domain, target, fset) not in todo:
                continue
            t1 = time.time()
            pred = np.full(len(rows), np.nan, dtype=np.float32)
            meta = []
            for k, (tr, te) in enumerate(folds):
                p, m = imp.run_fit(
                    data, icfg, target, fset, rows[tr], rows[te], seed=cfg.seed + 31 + k
                )
                pred[te] = p
                meta.append(m)
            sel = keep_type[rows] & ~np.isnan(pred)
            pd.DataFrame({"event_id": eid[rows][sel], "pred": pred[sel]}).to_parquet(
                student_path(cfg, domain, target, fset), index=False
            )
            student_path(cfg, domain, target, fset).with_suffix(".json").write_text(
                json.dumps(meta)
            )
            yv = data.Y[target].to_numpy()[rows]
            ok = ~np.isnan(yv) & ~np.isnan(pred)
            _log(
                f"student {domain:6s} {target:28s} {fset:4s} oof R2 {r2(yv[ok], pred[ok]):.3f} "
                f"(n {ok.sum():,}) rounds {[m['best_iter'] for m in meta]} {time.time() - t1:.0f} s"
            )


def _event_positions() -> pd.Series:
    """Row position in ``imputed_oof.parquet`` (= events360 order) per event_id."""
    e = (
        pq.read_table(_oof_path(), columns=["event_id"])
        .to_pandas()["event_id"]
        .to_numpy(dtype=object)
    )
    return pd.Series(np.arange(len(e)), index=e)


def transfer_shot_table(cfg: PayoffConfig, shots: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Shot table whose ``imp_*__E2`` / ``a_imp_*__E2a`` columns come from source-domain-only
    students: out-of-fold within the source gender (stage ``students``) for source rows and the
    stage-02 transfer fits (trained on every source-domain row) for target rows."""
    src, tgt = ("male", "female") if direction == "m2f" else ("female", "male")
    t = shots.copy()
    gender = t["gender"].to_numpy(dtype=object)
    pos = _event_positions()
    icfg = imp.ImputationConfig(smoke=cfg.smoke)
    for names, fset, key, col_prefix in (
        (SHOT_STATE, "E2", "event_id", ""),
        (ASSIST_STATE, "E2a", "key_pass_id", "a_"),
    ):
        keys = t[key].to_numpy(dtype=object)
        for name in names:
            col = f"{col_prefix}imp_{name}__{fset}"
            s = pd.read_parquet(student_path(cfg, src, name, fset))
            src_map = pd.Series(s["pred"].to_numpy(), index=s["event_id"].to_numpy(dtype=object))
            c = pd.read_parquet(imp.cache_path(icfg, "transfer", direction, name, fset))
            c = c[~np.isnan(c["pred"].to_numpy())]
            tgt_map = pd.Series(c["pred"].to_numpy(), index=pos.index[c["row"].to_numpy()])
            new = np.full(len(t), np.nan, dtype=np.float32)
            is_src = gender == src
            new[is_src] = src_map.reindex(keys[is_src]).to_numpy(dtype=np.float32)
            new[~is_src] = tgt_map.reindex(keys[~is_src]).to_numpy(dtype=np.float32)
            t[col] = new
    return t


def stage_transfer(cfg: PayoffConfig, force: bool = False) -> pd.DataFrame:
    """(c) xG trained on one gender and tested on the other with domain-restricted students;
    writes ``xg_transfer_preds.parquet``."""
    path = cache_dir(cfg) / TRANSFER_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    gender = shots["gender"].to_numpy(dtype=object)
    frames = []
    for direction in ("m2f", "f2m"):
        src, tgt = ("male", "female") if direction == "m2f" else ("female", "male")
        tr = np.where(gender == src)[0]
        te = np.where(gender == tgt)[0]
        if len(tr) == 0 or len(te) == 0:
            _log(f"transfer {direction}: empty domain, skipped")
            continue
        table = transfer_shot_table(cfg, shots, direction)
        designs = xg_designs(table)
        out = shots.iloc[te][XG_CONTEXT].copy()
        out["direction"] = direction
        out["pred_BASE"] = float(y[tr].mean())
        out["pred_BASE_target"] = float(y[te].mean())
        out[f"pred_{XG_REFERENCE}"] = shots["oracle_statsbomb_xg"].to_numpy(dtype=float)[te]
        for name in TRANSFER_VARIANTS:
            t0 = time.time()
            x = designs[name]
            cats = pf.categorical_in(x)
            acc = np.zeros(len(te), dtype=float)
            for s in _seeds(cfg, cfg.n_seeds_xg):
                b, r = fit_gbm(
                    cfg.xg, x.iloc[tr], y[tr], match[tr], s + 7, cats, "binary", cfg.n_jobs, True
                )
                acc += b.predict(x.iloc[te], num_iteration=r)
            out[f"pred_{name}"] = acc / cfg.n_seeds_xg
            m = pf.binary_metrics(y[te], out[f"pred_{name}"].to_numpy())
            _log(
                f"transfer {direction} {name:20s} log-loss {m['log_loss']:.4f} "
                f"auc {m['auc']:.3f} {time.time() - t0:.0f} s"
            )
        # coverage diagnostics of the swapped imputations
        for name in SHOT_STATE:
            out[f"cov_imp_{name}"] = table[f"imp_{name}__E2"].notna().to_numpy()[te]
        frames.append(out)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_parquet(path, index=False)
    return preds


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pred_cols(df: pd.DataFrame, suffix: str = "") -> dict[str, np.ndarray]:
    out = {}
    for c in df.columns:
        if c.startswith("pred_") and c.endswith(suffix):
            name = c[len("pred_") :]
            if suffix:
                name = name[: -len(suffix)]
            out[name] = df[c].to_numpy(dtype=float)
    return out


def xg_tables(cfg: PayoffConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = preds["is_goal"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    p = _pred_cols(preds)
    order = [
        "BASE",
        "LOC",
        "EVENT",
        "EVENT+IMP(E0)",
        "EVENT+IMP",
        "EVENT+ORACLE360",
        "EVENT+ORACLESHOT",
        "EVENT+ORACLESHOT_full",
        XG_REFERENCE,
    ]
    p = {k: p[k] for k in order if k in p}
    metrics = pf.metrics_table(y, p)
    deltas = pf.deltas_table(y, p, "EVENT", match, cfg.n_boot, cfg.seed)
    cal = pf.calibration_frame(
        y,
        {
            k: p[k]
            for k in ("EVENT", "EVENT+IMP", "EVENT+ORACLE360", "EVENT+ORACLESHOT", XG_REFERENCE)
            if k in p
        },
    )
    slices = {
        "play_pattern": pf.pattern_group(preds["f_play_pattern"].to_numpy(dtype=object)),
        "distance_band": pf.distance_band(preds["f_dist_goal"].to_numpy(dtype=float)),
        "reliable_frame": np.where(
            preds["y_reliable"].to_numpy(dtype=float) == 1.0, "reliable", "unreliable"
        ).astype(object),
        "gender": preds["gender"].to_numpy(dtype=object),
        "has_assist": np.where(
            preds["a_has_assist"].to_numpy(dtype=float) == 1.0, "assisted", "unassisted"
        ).astype(object),
    }
    sl = pf.slice_table(
        y,
        {
            k: p[k]
            for k in ("EVENT", "EVENT+IMP", "EVENT+ORACLE360", "EVENT+ORACLESHOT", XG_REFERENCE)
            if k in p
        },
        "EVENT",
        slices,
        match,
        n_boot=min(cfg.n_boot, 1000),
        seed=cfg.seed,
    )
    # both-frames-usable population (sff present and 360 frame ok)
    both = (preds["sff_present"].to_numpy(dtype=float) == 1.0) & (
        preds["y_frame_ok"].to_numpy(dtype=float) == 1.0
    )
    metrics_both = pf.metrics_table(y[both], {k: v[both] for k, v in p.items()})
    deltas_both = pf.deltas_table(
        y[both], {k: v[both] for k, v in p.items()}, "EVENT", match[both], cfg.n_boot, cfg.seed
    )
    return {
        "xg_metrics": metrics,
        "xg_deltas": deltas,
        "xg_calibration": cal,
        "xg_slices": sl,
        "xg_metrics_bothframes": metrics_both,
        "xg_deltas_bothframes": deltas_both,
    }


def distill_tables(
    cfg: PayoffConfig, xg: pd.DataFrame, dist: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    d = xg[["event_id", "is_goal", "match_id"]].merge(
        dist, on=["event_id"], how="inner", suffixes=("", "_d")
    )
    y = d["is_goal"].to_numpy(dtype=float)
    match = d["match_id"].to_numpy()
    xg_i = xg.set_index("event_id").loc[d["event_id"].to_numpy()]
    p = {
        "EVENT (direct)": xg_i["pred_EVENT"].to_numpy(dtype=float),
        "EVENT+IMP (direct)": xg_i["pred_EVENT+IMP"].to_numpy(dtype=float),
        "EVENT+ORACLE360 (teacher)": xg_i["pred_EVENT+ORACLE360"].to_numpy(dtype=float),
        "EVENT+ORACLESHOT (teacher)": xg_i["pred_EVENT+ORACLESHOT"].to_numpy(dtype=float),
    }
    for c in d.columns:
        if c.startswith("pred_"):
            p[c[len("pred_") :] + " (distilled)"] = d[c].to_numpy(dtype=float)
    metrics = pf.metrics_table(y, p)
    deltas = pf.deltas_table(y, p, "EVENT (direct)", match, cfg.n_boot, cfg.seed)
    # how well does the student reproduce its teacher (out-of-fold, on the goals scale)?
    rows = []
    for c in d.columns:
        if c.startswith("pred_"):
            name = c[len("pred_") :]
            teacher = name.split("<-")[1].split("@")[0]
            t = xg_i[f"pred_{teacher}"].to_numpy(dtype=float)
            s = d[c].to_numpy(dtype=float)
            rows.append(
                {
                    "run": name,
                    "corr_with_teacher_oof": float(np.corrcoef(t, s)[0, 1]),
                    "mae_vs_teacher": float(np.mean(np.abs(t - s))),
                    "mean_student": float(s.mean()),
                    "mean_teacher": float(t.mean()),
                }
            )
    return {
        "xg_distill_metrics": metrics,
        "xg_distill_deltas": deltas,
        "xg_distill_fidelity": pd.DataFrame(rows),
    }


def xpass_tables(cfg: PayoffConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = preds["is_complete"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    out: dict[str, pd.DataFrame] = {}
    metr, delt, cal, sl = [], [], [], []
    for tag in ("after", "noafter"):
        p = {"BASE": preds["pred_BASE"].to_numpy(dtype=float)}
        p.update(_pred_cols(preds, f"__{tag}"))
        m = pf.metrics_table(y, p)
        m["design"] = tag
        metr.append(m)
        d = pf.deltas_table(y, p, "EVENT", match, min(cfg.n_boot, 500), cfg.seed)
        d["design"] = tag
        delt.append(d)
        c = pf.calibration_frame(y, {k: v for k, v in p.items() if k != "BASE"})
        c["design"] = tag
        cal.append(c)
        slices = {
            "play_pattern": pf.pattern_group(preds["f_play_pattern"].to_numpy(dtype=object)),
            "length_band": pf.length_band(preds["f_after_pass_length"].to_numpy(dtype=float)),
            "height": preds["f_after_pass_height"].to_numpy(dtype=object),
            "reliable_frame": np.where(
                preds["y_reliable"].to_numpy(dtype=float) == 1.0, "reliable", "unreliable"
            ).astype(object),
            "under_pressure": np.where(
                preds["f_under_pressure"].to_numpy(dtype=bool), "pressed", "unpressed"
            ).astype(object),
            "gender": preds["gender"].to_numpy(dtype=object),
        }
        s = pf.slice_table(
            y,
            {k: v for k, v in p.items() if k != "BASE"},
            "EVENT",
            slices,
            match,
            n_boot=min(cfg.n_boot, 300),
            seed=cfg.seed,
            min_n=200,
        )
        s["design"] = tag
        sl.append(s)
    out["xpass_metrics"] = pd.concat(metr, ignore_index=True)
    out["xpass_deltas"] = pd.concat(delt, ignore_index=True)
    out["xpass_calibration"] = pd.concat(cal, ignore_index=True)
    out["xpass_slices"] = pd.concat(sl, ignore_index=True)
    return out


def transfer_tables(
    cfg: PayoffConfig, tr: pd.DataFrame, xg: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metr, delt = [], []
    xg_i = xg.set_index("event_id")
    for direction in ("m2f", "f2m"):
        d = tr[tr["direction"] == direction]
        if d.empty:
            continue
        y = d["is_goal"].to_numpy(dtype=float)
        match = d["match_id"].to_numpy()
        p = {
            "BASE (source rate)": d["pred_BASE"].to_numpy(dtype=float),
            "BASE (target rate)": d["pred_BASE_target"].to_numpy(dtype=float),
        }
        for name in TRANSFER_VARIANTS:
            p[name] = d[f"pred_{name}"].to_numpy(dtype=float)
        p[XG_REFERENCE] = d[f"pred_{XG_REFERENCE}"].to_numpy(dtype=float)
        # in-domain reference: the 5-fold CV predictions on the same shots
        sub = xg_i.reindex(d["event_id"].to_numpy())
        for name in ("EVENT", "EVENT+IMP", "EVENT+ORACLE360"):
            p[f"{name} (in-domain CV)"] = sub[f"pred_{name}"].to_numpy(dtype=float)
        m = pf.metrics_table(y, p)
        m["direction"] = direction
        m["coverage_imp"] = float(np.mean([d[f"cov_imp_{n}"].mean() for n in SHOT_STATE]))
        metr.append(m)
        dd = pf.deltas_table(y, p, "EVENT", match, cfg.n_boot, cfg.seed)
        dd["direction"] = direction
        delt.append(dd)
    return {
        "xg_transfer_metrics": pd.concat(metr, ignore_index=True),
        "xg_transfer_deltas": pd.concat(delt, ignore_index=True),
    }


def curve_tables(cfg: PayoffConfig, preds: pd.DataFrame) -> pd.DataFrame:
    """Learning-curve table: per training share, log-loss of every variant and paired delta vs
    EVENT at the same share."""
    y = preds["is_goal"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    rows = []
    for frac in CURVE_FRACS:
        p = {}
        for name in CURVE_VARIANTS:
            c = f"pred_{name}__{frac:g}"
            if c in preds.columns:
                p[name] = preds[c].to_numpy(dtype=float)
        if "EVENT" not in p:
            continue
        m = pf.metrics_table(y, p)
        d = pf.deltas_table(y, p, "EVENT", match, cfg.n_boot, cfg.seed).set_index("variant")
        for r in m.itertuples(index=False):
            row = {"train_share": frac, **r._asdict()}
            if r.variant in d.index:
                row.update(
                    {
                        k: d.loc[r.variant, k]
                        for k in (
                            "delta_log_loss",
                            "ci_low",
                            "ci_high",
                            "ci_low_clustered",
                            "ci_high_clustered",
                        )
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def importance_tables(cfg: PayoffConfig) -> pd.DataFrame:
    p = cache_dir(cfg) / XG_IMPORTANCE
    if not p.exists():
        return pd.DataFrame()
    imp_df = pd.read_parquet(p)
    imp_df["block"] = np.where(
        imp_df["feature"].str.startswith("s_"),
        "shot_state",
        np.where(
            imp_df["feature"].str.startswith("as_"),
            "assist_state",
            np.where(imp_df["feature"].str.startswith("a_"), "assist_event", "shot_event"),
        ),
    )
    block = imp_df.groupby(["variant", "block"])["gain_share"].sum().reset_index()
    top = (
        imp_df.sort_values(["variant", "gain_share"], ascending=[True, False])
        .groupby("variant")
        .head(8)
        .reset_index(drop=True)
    )
    return pd.concat(
        [block.assign(kind="block"), top.assign(kind="top_feature")], ignore_index=True
    )


def feature_sets_table(shots: pd.DataFrame) -> pd.DataFrame:
    designs = xg_designs(shots.head(200))
    rows = []
    for v in XG_VARIANTS:
        d = designs.get(v.name)
        rows.append(
            {
                "task": "xG",
                "variant": v.name,
                "n_features": str(0 if d is None else d.shape[1]),
                "description": v.description,
            }
        )
    for v in XPASS_VARIANTS:
        rows.append(
            {"task": "xPass", "variant": v.name, "n_features": "", "description": v.description}
        )
    rows.append(
        {
            "task": "xG",
            "variant": XG_REFERENCE,
            "n_features": "0",
            "description": "StatsBomb's own xG as given (reference; its training data may "
            "include these matches)",
        }
    )
    return pd.DataFrame(rows)


def _fmt_delta(r: pd.Series) -> str:
    return f"{r['delta_log_loss']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"


def _delta_md(deltas: pd.DataFrame, extra: list[str] | None = None) -> str:
    d = deltas.copy()
    d["delta_log_loss [95% CI]"] = d.apply(_fmt_delta, axis=1)
    d["clustered CI"] = d.apply(
        lambda r: f"[{r['ci_low_clustered']:+.4f}, {r['ci_high_clustered']:+.4f}]", axis=1
    )
    d["delta_brier [95% CI]"] = d.apply(
        lambda r: f"{r['delta_brier']:+.5f} [{r['brier_ci_low']:+.5f}, {r['brier_ci_high']:+.5f}]",
        axis=1,
    )
    cols = (extra or []) + [
        "variant",
        "n",
        "delta_log_loss [95% CI]",
        "clustered CI",
        "delta_brier [95% CI]",
        "significant",
    ]
    return md_table(d[cols])


def _metrics_md(m: pd.DataFrame, extra: list[str] | None = None) -> str:
    cols = (extra or []) + [
        "variant",
        "n",
        "positives",
        "log_loss",
        "brier",
        "auc",
        "ece",
        "mean_pred",
    ]
    return md_table(m[[c for c in cols if c in m.columns]])


def _level_key(level: object) -> tuple[float, str]:
    """Sort slice levels by their leading number (distance / length bands), then by name."""
    txt = str(level)
    head = txt.split("-")[0].rstrip("+")
    try:
        return (float(head), txt)
    except ValueError:
        return (float("inf"), txt)


def _slice_md(sl: pd.DataFrame, slice_name: str, variants: tuple[str, ...]) -> str:
    d = sl[(sl["slice"] == slice_name) & (sl["variant"].isin(variants))].copy()
    if d.empty:
        return "(no rows)"
    d["delta [95% CI]"] = d.apply(
        lambda r: "" if pd.isna(r.get("delta_log_loss", np.nan)) else _fmt_delta(r), axis=1
    )
    piv = d.pivot_table(
        index=["level", "n", "positives"], columns="variant", values="log_loss", aggfunc="first"
    )
    piv = piv[[v for v in variants if v in piv.columns]].reset_index()
    order = sorted(range(len(piv)), key=lambda i: _level_key(piv["level"].iloc[i]))
    piv = piv.iloc[order]
    for v in variants:
        if v == "EVENT":
            continue
        dd = d[d["variant"] == v].set_index("level")["delta [95% CI]"]
        piv[f"d({v})"] = piv["level"].map(dd)
    return md_table(piv)


def _pivot_cal(cal: pd.DataFrame) -> pd.DataFrame:
    piv = cal.pivot_table(index="bin", columns="variant", values=["pred", "obs"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index()


def _fits_md(fits: pd.DataFrame, keys: list[str]) -> str:
    f = (
        fits.groupby(keys)
        .agg(
            rounds_median=("rounds", "median"), seconds=("seconds", "sum"), fits=("rounds", "size")
        )
        .reset_index()
    )
    return md_table(f, floatfmt="{:.1f}")


def _curve_ll(c: pd.DataFrame, frac: float, variant: str) -> float:
    r = c[(c["train_share"] == frac) & (c["variant"] == variant)]
    return float(r["log_loss"].iloc[0]) if len(r) else float("nan")


def _verdict(row: pd.Series | Any) -> str:
    """Plain-language reading of a paired delta row (Series or itertuples row; per-sample CI)."""
    r = row._asdict() if hasattr(row, "_asdict") else row
    lo, hi, d = float(r["ci_low"]), float(r["ci_high"]), float(r["delta_log_loss"])
    if lo > 0:
        return f"significant gain ({d:+.4f})"
    if hi < 0:
        return f"significant loss ({d:+.4f})"
    return f"no measurable difference ({d:+.4f}, CI [{lo:+.4f}, {hi:+.4f}])"


def _reading_lines(tables: dict[str, pd.DataFrame]) -> list[str]:
    d = tables["xg_deltas"].set_index("variant")
    out = ["### Reading\n"]
    out.append(
        "- xG, imputed state (EVENT+IMP vs EVENT): " + _verdict(d.loc["EVENT+IMP"]) + "; the "
        "360 oracle gives "
        + _verdict(d.loc["EVENT+ORACLE360"])
        + " and the shot freeze frame "
        + _verdict(d.loc["EVENT+ORACLESHOT"])
        + ". The event-only student recovers none of the "
        "oracle gain on shots."
    )
    if "xg_distill_deltas" in tables:
        dd = tables["xg_distill_deltas"]
        dd = dd[dd["variant"].str.contains("distilled")].set_index("variant")
        best = dd["delta_log_loss"].idxmax()
        out.append(f"- Distillation: best run {best}: " + _verdict(dd.loc[best]) + ".")
    if "xg_curve" in tables:
        c = tables["xg_curve"]
        c = c[c["variant"] == "EVENT+IMP"]
        parts = [f"{r.train_share:.0%} {_verdict(r)}" for r in c.itertuples(index=False)]
        out.append(
            "- Learning curve (EVENT+IMP vs EVENT at each training share): "
            + "; ".join(parts)
            + "."
        )
    if "xpass_deltas" in tables:
        xd = tables["xpass_deltas"]
        a = xd[xd["design"] == "after"].set_index("variant")
        n = xd[xd["design"] == "noafter"].set_index("variant")
        out.append(
            "- xPass with the destination known: EVENT+IMP "
            + _verdict(a.loc["EVENT+IMP"])
            + "; EVENT+ORACLE "
            + _verdict(a.loc["EVENT+ORACLE"])
            + "; dropping the realised "
            "duration from EVENT costs " + _verdict(a.loc["EVENT-nodur"]) + ". Without the "
            "destination: EVENT+IMP " + _verdict(n.loc["EVENT+IMP"]) + " (the no-after oracle "
            "gap is not attainable, see caveats)."
        )
    if "xg_transfer_deltas" in tables:
        td = tables["xg_transfer_deltas"]
        parts = []
        for direction in ("m2f", "f2m"):
            t = td[td["direction"] == direction].set_index("variant")
            if "EVENT+IMP" in t.index:
                parts.append(
                    f"{direction}: EVENT+IMP "
                    + _verdict(t.loc["EVENT+IMP"])
                    + ", EVENT+ORACLE360 "
                    + _verdict(t.loc["EVENT+ORACLE360"])
                )
        out.append(
            "- Cross-gender transfer (source-only students and xG): " + "; ".join(parts) + "."
        )
    if "xg_slices" in tables:
        sl = tables["xg_slices"]
        sub = sl[
            (sl["variant"] == "EVENT+IMP")
            & sl["slice"].isin(["play_pattern", "distance_band", "reliable_frame"])
        ]
        sig = sub[(sub["ci_low"] > 0) | (sub["ci_high"] < 0)]
        if sig.empty:
            out.append(
                "- Slices: no play-pattern, distance-band or reliability slice shows a "
                "significant EVENT+IMP effect on xG (per-shot CIs); the oracle gains are largest "
                "on close-range shots (distance_band table)."
            )
        else:
            parts = [f"{r.slice}={r.level}: {_verdict(r)}" for r in sig.itertuples(index=False)]
            out.append(
                "- Slices with a significant EVENT+IMP effect on xG: " + "; ".join(parts) + "."
            )
    return out


def _headline_lines(
    cfg: PayoffConfig, tables: dict[str, pd.DataFrame], shots: pd.DataFrame
) -> list[str]:
    n_shots = len(shots)
    n_goals = int(shots["is_goal"].sum())
    m = tables["xg_metrics"].set_index("variant")
    d = tables["xg_deltas"].set_index("variant")

    def ll(v: str) -> str:
        return f"{float(m.loc[v, 'log_loss']):.4f}" if v in m.index else "n/a"

    def dl(v: str) -> str:
        return _fmt_delta(d.loc[v]) if v in d.index else "n/a"

    out = [
        f"- Shots: n = {n_shots:,} non-penalty shots, {n_goals:,} goals ({n_goals / n_shots:.3f}), "
        f"{shots['match_id'].nunique()} matches, {int(shots['a_has_assist'].sum()):,} "
        "with a key pass; "
        f"5-fold match-grouped CV (stage-02 folds), {cfg.n_seeds_xg} seeds averaged per variant.",
        f"- xG log-loss: BASE {ll('BASE')}, LOC {ll('LOC')}, EVENT {ll('EVENT')}, "
        f"EVENT+IMP {ll('EVENT+IMP')}, EVENT+ORACLE360 {ll('EVENT+ORACLE360')}, "
        f"EVENT+ORACLESHOT {ll('EVENT+ORACLESHOT')}, "
        f"EVENT+ORACLESHOT_full {ll('EVENT+ORACLESHOT_full')}, StatsBomb xG {ll(XG_REFERENCE)}.",
        "- Paired log-loss deltas vs EVENT (positive = better; per-shot bootstrap 95% CI): "
        f"EVENT+IMP {dl('EVENT+IMP')}; EVENT+ORACLE360 {dl('EVENT+ORACLE360')}; "
        f"EVENT+ORACLESHOT {dl('EVENT+ORACLESHOT')}; StatsBomb xG {dl(XG_REFERENCE)}.",
    ]
    if "xg_distill_deltas" in tables:
        dd = tables["xg_distill_deltas"].set_index("variant")
        for k in dd.index:
            if "distilled" in k:
                out.append(f"- Distillation {k}: {_fmt_delta(dd.loc[k])} vs EVENT (direct).")
    if "xg_curve" in tables:
        c = tables["xg_curve"]
        for frac in CURVE_FRACS:
            cc = c[(c["train_share"] == frac) & (c["variant"] == "EVENT+IMP")]
            if not cc.empty:
                r = cc.iloc[0]
                out.append(
                    f"- Learning curve, {frac:.0%} of the training matches: EVENT+IMP vs EVENT "
                    f"{_fmt_delta(r)} (EVENT log-loss "
                    f"{_curve_ll(c, frac, 'EVENT'):.4f})."
                )
    if "xpass_metrics" in tables:
        xm, xd = tables["xpass_metrics"], tables["xpass_deltas"]
        for tag in ("after", "noafter"):
            mm = xm[xm["design"] == tag].set_index("variant")
            ddd = xd[xd["design"] == tag].set_index("variant")
            if "EVENT" in mm.index:
                out.append(
                    f"- xPass ({tag}): n = {int(mm.loc['EVENT', 'n']):,} passes, completion "
                    f"{mm.loc['EVENT', 'positives'] / mm.loc['EVENT', 'n']:.3f}; log-loss BASE "
                    f"{mm.loc['BASE', 'log_loss']:.4f}, EVENT {mm.loc['EVENT', 'log_loss']:.4f}, "
                    f"EVENT+IMP {mm.loc['EVENT+IMP', 'log_loss']:.4f} "
                    f"({_fmt_delta(ddd.loc['EVENT+IMP'])}), EVENT+ORACLE "
                    f"{mm.loc['EVENT+ORACLE', 'log_loss']:.4f} "
                    f"({_fmt_delta(ddd.loc['EVENT+ORACLE'])})."
                )
    if "xg_transfer_deltas" in tables:
        td = tables["xg_transfer_deltas"]
        for direction in ("m2f", "f2m"):
            t = td[td["direction"] == direction].set_index("variant")
            if "EVENT+IMP" in t.index:
                out.append(
                    f"- Transfer {direction}: EVENT+IMP {_fmt_delta(t.loc['EVENT+IMP'])}, "
                    f"EVENT+ORACLE360 {_fmt_delta(t.loc['EVENT+ORACLE360'])} vs EVENT "
                    "(students and xG trained on the source gender only)."
                )
    return out


PROTOCOL_TEXT = [
    "- Population: Shot rows of `events360.parquet` (417 matches with usable 360); penalties "
    "excluded ({pen}); label = `post_shot_outcome == 'Goal'`. Shots keep their 360-unusable / "
    "unreliable frames (oracle columns are NaN there; the reliability slice separates them); a "
    "both-frames-usable subset is reported separately.",
    "- Splits: the stage-02 `fold` column (`group_kfold` by match, seed 0, 5 folds), so every "
    "imputed feature of a test-fold shot comes from students that never saw that fold's matches "
    "(the imputations are `imputed_oof.parquet` columns). Rounds by early stopping on an inner "
    "15% match holdout, then refit on all training rows; probabilities averaged over seeds. "
    "BASE = training-fold goal rate.",
    "- Features: see the feature-set table. EVENT reads only `f_*` columns of the shot (stage-02 "
    "E2 design minus the post-instant `f_after_duration`) plus the key pass's attributes "
    "(`shot.key_pass_id` from the raw events; that pass is complete before the shot instant). "
    "Never used: `shot.freeze_frame`, `statsbomb_xg`, `one_on_one`, `open_goal`, `post_*`, any "
    "`y_*` outside the named oracle blocks. Imputed counts / distances are clipped at 0 "
    "(stage-02 outputs are unclipped).",
    "- Metrics: log-loss, Brier, AUC, ECE (10 equal-count bins); deltas are paired per-shot "
    "bootstraps of the per-shot loss ({n_boot} resamples, seed {seed}) with a match-clustered CI "
    "next to them; `significant` = the per-shot CI excludes 0.",
    "- StatsBomb xG is the vendor's model output as stored in the events (a reference, not a "
    "variant trained here; its training set may include these matches).",
]

CAVEATS_TEXT = [
    "- The imputed shot state is out-of-fold with respect to the xG folds, but the students that "
    "produced the training-fold imputations were fitted on the other four folds including the test "
    "fold's 360 labels (not its goal labels): the usual second-order coupling of stacked "
    "out-of-fold features, shared with every phase-1 stage.",
    "- EVENT already contains every input the students read (the stage-02 E2 design), so "
    "EVENT+IMP can only add what the 360 supervision taught the students beyond what ~10k goal "
    "labels teach the xG model directly; the learning curve tests that at smaller label budgets.",
    "- `f_under_pressure` is StatsBomb's flag for a Pressure event overlapping the shot; for shots "
    "it is contemporaneous with the shot instant and a standard xG input, but not strictly "
    "pre-instant.",
    "- The 360 oracle follows the stage-02 label rules: `nearest_opp_dist_in_cone` is NaN when the "
    "cone is empty (informative missingness), `block_depth` is NaN on unreliable frames, keeper "
    "distance is NaN when the keeper is not visible or inconsistently flagged. The shot-frame "
    "oracle has no block depth; `sff_n_opp_ahead_of_ball` stands in. Both oracles are "
    "visible-area truncations, not tracking.",
    "- StatsBomb xG is a reference trained by the vendor on far more shots (possibly including "
    "these matches) with the shot freeze frame; treat it as an upper reference, not a fair "
    "competitor.",
    "- ~10k shots / ~1.1k goals: single-model log-loss differences of 0.002-0.005 are within "
    "noise; read the CIs. Slice CIs are wider still.",
    "- xPass: the subsample is match-stratified (every match keeps the same expected share); the "
    "with-after design conditions on the realised end location, which for incomplete passes is "
    "where the ball was won. The stage-02 E2a students also read the realised pass duration, so "
    "the with-after EVENT design keeps `f_after_duration` (otherwise the imputations smuggle it "
    "in: with duration removed from EVENT only, EVENT+IMP beat even the 360 oracle on short "
    "passes); `EVENT-nodur` shows what duration alone is worth. In the no-after design the "
    "oracle's lane / end-location quantities are defined relative to the realised end, so the "
    "no-after ORACLE gap is not an attainable target for a pre-instant model; the E2 students "
    "never see the end and are comparable.",
    "- Transfer students use a smaller training cap than the stage-02 CV students and single "
    "fits; their in-domain skill is slightly lower, so the transfer gap mixes domain shift with a "
    "small capacity effect.",
    "- Stage-02 open issues respected: imputed counts / distances clipped at 0 here; "
    "`deep_block` / `counter_on` and the NFL highlight check are not used by this stage.",
]


def render_report(
    cfg: PayoffConfig,
    tables: dict[str, pd.DataFrame],
    shots: pd.DataFrame,
    elapsed: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append(
        "# Soccer 03 - payoff of imputed defensive state: xG and xPass in the 360 matches\n"
    )
    lines.append(
        "Machine-written by `research/privileged_tracking/soccer/payoff.py`. Question: does the "
        "stage-02 imputed 360 state (out-of-fold student predictions from event data only) improve "
        "an event-only xG / xPass model, how far is that from the 360 and shot-freeze-frame "
        "oracles, does distilling an oracle teacher into an event-only student beat training on "
        "goals, does the gain survive a cross-gender domain shift, and where does the imputed "
        "state help?\n"
    )
    lines.append("## Headline\n")
    lines += _headline_lines(cfg, tables, shots)
    lines.append("")
    lines += _reading_lines(tables)
    lines.append("")
    lines.append("## Protocol\n")
    fmt = {"pen": "yes" if cfg.exclude_penalties else "no", "n_boot": cfg.n_boot, "seed": cfg.seed}
    lines += [t.format(**fmt) for t in PROTOCOL_TEXT]
    lines.append("\n### Feature sets\n")
    lines.append(md_table(tables["feature_sets"]))
    lines.append("")
    lines.append("## (a) xG: event-only vs imputed vs oracle state\n")
    lines.append(_metrics_md(tables["xg_metrics"]))
    lines.append("\nPaired deltas vs EVENT (log-loss, positive = better than EVENT):\n")
    lines.append(_delta_md(tables["xg_deltas"]))
    lines.append("\nBoth-frames-usable subset (`sff_present == 1` and `y_frame_ok == 1`):\n")
    lines.append(_metrics_md(tables["xg_metrics_bothframes"]))
    lines.append("")
    lines.append(_delta_md(tables["xg_deltas_bothframes"]))
    lines.append("\n### Calibration (equal-count bins: mean predicted vs observed goal rate)\n")
    lines.append(md_table(_pivot_cal(tables["xg_calibration"]), floatfmt="{:.3f}"))
    if "xg_importance" in tables and not tables["xg_importance"].empty:
        lines.append("\n### Gain share by feature block (mean over folds and seeds)\n")
        b = tables["xg_importance"]
        bb = (
            b[b["kind"] == "block"]
            .pivot_table(index="variant", columns="block", values="gain_share", aggfunc="first")
            .reset_index()
        )
        lines.append(md_table(bb, floatfmt="{:.3f}"))
        lines.append("\nTop features per variant:\n")
        lines.append(
            md_table(
                b[b["kind"] == "top_feature"][["variant", "feature", "gain_share"]],
                floatfmt="{:.3f}",
            )
        )
    if "xg_distill_metrics" in tables:
        lines.append("\n### Distillation: event-only student trained on an oracle teacher's xG\n")
        lines.append(
            "Runs are `student<-teacher@alpha` (alpha = weight of the teacher's probability in the "
            "soft label, 1 - alpha on the goal). Nested protocol: for every outer test fold the "
            "teacher is re-fitted by an inner CV over the training folds only, so the student "
            "never "
            "sees a test-fold label through the teacher; the student uses LightGBM's cross-entropy "
            "objective on the soft labels. Evaluated on goals.\n"
        )
        lines.append(_metrics_md(tables["xg_distill_metrics"]))
        lines.append("\nPaired deltas vs EVENT (direct):\n")
        lines.append(_delta_md(tables["xg_distill_deltas"]))
        lines.append(
            "\nStudent-teacher fidelity (out-of-fold student vs the teacher's own "
            "out-of-fold xG):\n"
        )
        lines.append(md_table(tables["xg_distill_fidelity"]))
    if "xg_curve" in tables:
        lines.append("\n### Learning curve: does imputed state help when goal labels are scarce?\n")
        lines.append(
            "Each xG variant refitted with only a share of the training matches (test folds "
            "complete, same seeds); delta = paired log-loss gain over EVENT at the same share.\n"
        )
        c = tables["xg_curve"].copy()
        c["delta [95% CI]"] = c.apply(
            lambda r: "" if pd.isna(r.get("delta_log_loss", np.nan)) else _fmt_delta(r), axis=1
        )
        lines.append(
            md_table(
                c[["train_share", "variant", "n", "log_loss", "brier", "auc", "delta [95% CI]"]]
            )
        )
    if "xpass_metrics" in tables:
        lines.append("\n## (b) xPass: pass completion in the 360 matches\n")
        n_pass = int(tables["xpass_metrics"].iloc[0]["n"])
        lines.append(
            f"Population: a match-stratified subsample of {n_pass:,} pass attempts (target "
            f"{cfg.xpass_target_n:,}; `Unknown` / `Injury Clearance` outcomes dropped) of the 360 "
            "matches; label = `post_pass_outcome == Complete`. Two designs: `after` = the pass's "
            "realised end location / length / angle / height / switch / cross / through-ball / "
            "cut-back are features (the usual xPass setting: the destination is known) with E2a "
            "students; `noafter` = only what is known at the pass instant, with E2 students. The "
            "receiver-distance student (`nearest_opp_to_receiver`) had no stage-02 model and was "
            "fitted here out-of-fold on all Pass rows with the stage-02 settings (fit table "
            "below). "
            "Single seed; rounds by early stopping on an inner match holdout without refit.\n"
        )
        lines.append(_metrics_md(tables["xpass_metrics"], ["design"]))
        lines.append("\nPaired deltas vs EVENT:\n")
        lines.append(_delta_md(tables["xpass_deltas"], ["design"]))
        if "receiver_fits" in tables:
            lines.append("\nReceiver-distance student (out-of-fold on all Pass rows):\n")
            lines.append(md_table(tables["receiver_fits"]))
        lines.append("\n### xPass calibration (after design)\n")
        c = tables["xpass_calibration"]
        lines.append(md_table(_pivot_cal(c[c["design"] == "after"]), floatfmt="{:.3f}"))
        lines.append(
            "\n### xPass slices (after design; log-loss per variant, delta vs EVENT with "
            "per-pass CI)\n"
        )
        s = tables["xpass_slices"]
        for sname in (
            "play_pattern",
            "length_band",
            "height",
            "reliable_frame",
            "under_pressure",
            "gender",
        ):
            lines.append(f"\n**{sname}**\n")
            lines.append(
                _slice_md(s[s["design"] == "after"], sname, ("EVENT", "EVENT+IMP", "EVENT+ORACLE"))
            )
    if "xg_transfer_metrics" in tables:
        lines.append("\n## (c) Cross-gender robustness of the xG gain\n")
        lines.append(
            "Train on one gender's shots, test on the other's. Every imputed feature comes from "
            "students trained on the training gender only: out-of-fold within the source gender "
            f"(stage `students`, 5-fold by match, train cap {cfg.student_train_cap:,}) for the "
            "training shots and the stage-02 transfer fits (`imputation_cache/transfer__<dir>__*`, "
            "trained on every source-gender row) for the test shots. `(in-domain CV)` rows are the "
            "main 5-fold predictions on the same test shots (students and xG trained on both "
            "genders' other matches) for comparison; BASE (target rate) is an oracle base rate.\n"
        )
        lines.append(_metrics_md(tables["xg_transfer_metrics"], ["direction"]))
        lines.append("\nPaired deltas vs EVENT (transferred):\n")
        lines.append(_delta_md(tables["xg_transfer_deltas"], ["direction"]))
    lines.append("\n## (d) Where does the imputed state help? xG log-loss by slice\n")
    lines.append(
        "Per-slice log-loss of each variant and paired delta vs EVENT (positive = better; "
        "per-shot bootstrap CI, 1000 resamples). Slices with fewer than 50 shots are omitted.\n"
    )
    sl = tables["xg_slices"]
    for sname in ("play_pattern", "distance_band", "reliable_frame", "has_assist", "gender"):
        lines.append(f"\n**{sname}**\n")
        lines.append(
            _slice_md(
                sl,
                sname,
                ("EVENT", "EVENT+IMP", "EVENT+ORACLE360", "EVENT+ORACLESHOT", XG_REFERENCE),
            )
        )
    lines.append("\n## Fits\n")
    if "xg_fits" in tables:
        lines.append(_fits_md(tables["xg_fits"], ["variant"]))
    if "xpass_fits" in tables:
        lines.append("")
        lines.append(_fits_md(tables["xpass_fits"], ["variant", "design"]))
    lines.append("\nStage wall time (s): " + ", ".join(f"{k} {v:.0f}" for k, v in elapsed.items()))
    lines.append("\n## Caveats\n")
    lines += CAVEATS_TEXT
    return "\n".join(lines) + "\n"


def stage_report(cfg: PayoffConfig, elapsed: dict[str, float] | None = None) -> None:
    """Assemble every table, write ``soccer_03_*.parquet`` and ``soccer_03_payoff.md``."""
    t0 = time.time()
    shots = shot_population(cfg, build_shot_table(cfg))
    xg = stage_xg(cfg)
    tables = xg_tables(cfg, xg)
    tables["feature_sets"] = feature_sets_table(shots)
    tables["xg_importance"] = importance_tables(cfg)
    fits_p = cache_dir(cfg) / XG_FITS
    if fits_p.exists():
        tables["xg_fits"] = pd.read_parquet(fits_p)
    dp = cache_dir(cfg) / DISTILL_PREDS
    if dp.exists():
        tables.update(distill_tables(cfg, xg, pd.read_parquet(dp)))
    pp = cache_dir(cfg) / XPASS_PREDS
    if pp.exists():
        xp = pd.read_parquet(pp)
        tables.update(xpass_tables(cfg, xp))
        fp = cache_dir(cfg) / XPASS_FITS
        if fp.exists():
            tables["xpass_fits"] = pd.read_parquet(fp)
        rp = cache_dir(cfg) / "receiver_student_fits.parquet"
        if rp.exists():
            tables["receiver_fits"] = pd.read_parquet(rp)
    tp = cache_dir(cfg) / TRANSFER_PREDS
    if tp.exists():
        tables.update(transfer_tables(cfg, pd.read_parquet(tp), xg))
    cp = cache_dir(cfg) / CURVE_PREDS
    if cp.exists():
        tables["xg_curve"] = curve_tables(cfg, pd.read_parquet(cp))
    elapsed = dict(elapsed or {})
    elapsed["report"] = time.time() - t0
    text = render_report(cfg, tables, shots, elapsed)
    if cfg.write and not cfg.smoke:
        out = reports_dir()
        for k, t in tables.items():
            t.to_parquet(out / f"{REPORT_PREFIX}_{k}.parquet", index=False)
        xg.to_parquet(out / f"{REPORT_PREFIX}_xg_predictions.parquet", index=False)
        (out / f"{REPORT_PREFIX}_payoff.md").write_text(text)
        _log(f"report written: {out / (REPORT_PREFIX + '_payoff.md')} ({len(tables)} tables)")
    else:
        _log(text[:3000])


STAGES = (
    "shots",
    "xg",
    "distill",
    "curve",
    "passes",
    "xpass",
    "students",
    "transfer",
    "final",
    "report",
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", default="all", help="one of " + ", ".join(STAGES) + " or all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=None, help="seeds averaged per xG variant")
    ap.add_argument("--xpass-n", type=int, default=None)
    args = ap.parse_args()
    cfg = PayoffConfig(n_jobs=args.n_jobs, smoke=args.smoke)
    if args.smoke:
        cfg.n_seeds_xg = 1
        cfg.n_boot = 200
        cfg.xpass_target_n = 15_000
        cfg.student_train_cap = 30_000
    if args.n_seeds:
        cfg.n_seeds_xg = args.n_seeds
    if args.xpass_n:
        cfg.xpass_target_n = args.xpass_n
    stages = list(STAGES) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    elapsed: dict[str, float] = {}
    for s in stages:
        t0 = time.time()
        if s == "shots":
            build_shot_table(cfg, force=args.force)
        elif s == "xg":
            stage_xg(cfg, force=args.force)
        elif s == "distill":
            stage_distill(cfg, force=args.force)
        elif s == "curve":
            stage_curve(cfg, force=args.force)
        elif s == "final":
            stage_final(cfg, force=args.force)
        elif s == "passes":
            build_pass_table(cfg, force=args.force)
        elif s == "xpass":
            stage_xpass(cfg, force=args.force)
        elif s == "students":
            stage_students(cfg, force=args.force)
        elif s == "transfer":
            stage_transfer(cfg, force=args.force)
        elif s == "report":
            stage_report(cfg, elapsed)
        else:
            raise SystemExit(f"unknown stage {s}")
        elapsed[s] = time.time() - t0
        _log(f"stage {s}: {elapsed[s]:.0f} s")


if __name__ == "__main__":
    main()
