"""Soccer 02 - imputation students: predict 360 defensive state from event-only features.

Reads ``processed_dir('soccer')/events360.parquet`` (soccer 01) and trains one LightGBM
student per (target, feature set) that predicts a 360-derived quantity
(:data:`imputation_features.TARGETS`: block depth / defensive line / block width and length,
opponents ahead of the ball, the ``deep_block`` and ``counter_on`` binaries, opponents within
5 / 10 yd, nearest opponent, cone occupancy, keeper distance and the two pass-lane counts)
from information available in plain StatsBomb events. Feature sets are nested
(:data:`imputation_features.FEATURE_SETS`): ``loc`` (location-only baseline), ``E0`` current
event, ``E1`` + possession context, ``E2`` + 10-event window and opponent defensive-action
features, ``E2a`` + the POST-INSTANT ``f_after_*`` columns, ``E2r`` = E2 trained on reliable
rows only, ``E3`` = E2 + the raw sequence block (LightGBM reference for the MLP student).

Protocol: (i) 5-fold ``group_kfold`` by match over all 360 matches (out-of-fold predictions
for every row); (ii) cross-competition, men -> women and women -> men; (iii) forward in time,
every competition up to 2023/24 -> Euro 2024 + Women's Euro 2025. Training rows are thinned
uniformly at random to ``train_cap`` (300k; every match keeps the same expected share) and
rounds are chosen by early stopping on an inner 15% match holdout of the training rows, never
on the test rows. Team-shape targets are trained and scored on reliable frames (>= 7
opponents visible) of possession-team events; ball-relative and pass targets on every usable
frame, with the reliability sensitivity reported. Baselines: training global mean / base
rate, per-event-type mean, location-only LightGBM. Paired deltas use a match-clustered
bootstrap. The MLP student (sklearn ``MLPRegressor``, PyTorch is unavailable) is compared to
LightGBM on three headline targets on one held-out fold.

Every stage caches its predictions under ``processed_dir('soccer')/imputation_cache`` so the
work can be split over several invocations (each well under 25 minutes)::

    python -m research.privileged_tracking.soccer.imputation --stage cv --targets shape
    python -m research.privileged_tracking.soccer.imputation --stage cv --targets ball
    python -m research.privileged_tracking.soccer.imputation --stage cv --targets pass
    python -m research.privileged_tracking.soccer.imputation --stage transfer
    python -m research.privileged_tracking.soccer.imputation --stage forward
    python -m research.privileged_tracking.soccer.imputation --stage mlp
    python -m research.privileged_tracking.soccer.imputation --stage final
    python -m research.privileged_tracking.soccer.imputation --stage report
    python -m research.privileged_tracking.soccer.apply_student          # then impute events_no360

``--smoke`` runs any stage on the first parquet row group only, into a separate cache, without
writing reports or models. Outputs: ``processed_dir('soccer')/imputed_oof.parquet``
(out-of-fold predictions for every events360 row, ``<target>__<fset>`` plus ``y_<target>``),
``models/students_<fset>.joblib`` (:class:`apply_student.StudentBundle`),
``reports/soccer_02_imputation.md`` and ``reports/soccer_02_*.parquet``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (
    brier,
    calibration_table,
    clustered_bootstrap_delta,
    log_loss,
    mae,
    per_sample_log_loss,
    r2,
)
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.soccer import apply_student as aps
from research.privileged_tracking.soccer import imputation_features as imf

if TYPE_CHECKING:
    from pathlib import Path

CACHE_SUBDIR = "imputation_cache"
BASELINES = ["base_global", "base_type"]
CV_FSETS = ["loc", "E0", "E1", "E2", "E2a"]
RELIABILITY_FSET = "E2r"  # added for ball / pass targets in the CV stage
TRANSFER_FSETS = ["loc", "E0", "E2", "E2a"]
FINAL_FSETS = ["E0", "E1", "E2", "E2a"]
DELTA_PAIRS = [
    ("base_type", "loc"),
    ("loc", "E0"),
    ("E0", "E1"),
    ("E1", "E2"),
    ("E2", "E2a"),
    ("base_type", "E2"),
    ("E2", "E2r"),
]
HEADLINE = [
    "block_depth",
    "def_line",
    "n_opp_ahead_of_ball",
    "deep_block",
    "counter_on",
    "n_opp_within_5",
    "nearest_opp_dist",
    "n_opp_in_cone",
    "n_opp_in_lane",
]
ELAPSED_BINS = [0, 2, 5, 10, 20, 40, np.inf]
SINCE_DEF_S_BINS = [0, 2, 5, 10, 30, 60, np.inf]
SINCE_DEF_N_BINS = [0, 1, 2, 3, 5, 10, 20, 21]


@dataclass
class ImputationConfig:
    """Driver configuration.

    Attributes:
        n_jobs: LightGBM / BLAS threads.
        n_folds: match-grouped folds.
        train_cap: training rows kept (uniform thinning) for the CV / transfer / forward fits.
        final_cap: training rows kept for the final students.
        learning_rate / num_leaves / min_data_in_leaf / feature_fraction / bagging_fraction /
            max_bin / max_rounds / early_stopping: LightGBM.
        inner_holdout_frac: share of training matches held out for early stopping.
        n_boot: clustered-bootstrap resamples.
        mlp_targets / mlp_train_cap / mlp_hidden / mlp_max_iter: the MLP comparison.
        forward_test: (competition, season) pairs held out by the forward split.
        smoke: first parquet row group only, separate cache, nothing written to reports / models.
        write: write parquet / report / model outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    n_folds: int = 5
    train_cap: int = 300_000
    final_cap: int = 400_000
    learning_rate: float = 0.1
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    max_bin: int = 63
    max_rounds: int = 400
    early_stopping: int = 25
    inner_holdout_frac: float = 0.15
    n_boot: int = 500
    mlp_targets: tuple[str, ...] = ("block_depth", "n_opp_within_5", "nearest_opp_dist")
    mlp_train_cap: int = 200_000
    mlp_hidden: tuple[int, ...] = (256, 128)
    mlp_max_iter: int = 40
    forward_test: tuple[tuple[str, str], ...] = (
        ("UEFA Euro", "2024"),
        ("UEFA Women's Euro", "2025"),
    )
    smoke: bool = False
    write: bool = True

    @property
    def cache_dir(self) -> Path:
        d = processed_dir("soccer") / (CACHE_SUBDIR + ("_smoke" if self.smoke else ""))
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

ID_COLS = [
    "match_id",
    "event_id",
    "event_index",
    "period",
    "competition",
    "season",
    "gender",
    "match_date",
]
Y_LOAD = sorted(
    {t.source for t in imf.TARGETS if t.source != "derived"}
    | {"y_def_line", "y_n_opp_ahead_of_ball", "y_reliable", "y_frame_ok", "y_n_opponents_visible",
       "y_keeper_consistent"}
)


@dataclass
class Data:
    """The 360 rows with their targets, subsets and fold assignment (aligned on row position).

    Attributes:
        df: ids, ``f_*`` (and ``seq_*`` when loaded) columns and the ``y_*`` sources ``[n, ...]``.
        Y: :func:`imputation_features.target_frame` ``[n, n_targets]`` (NaN = not trainable).
        masks: :func:`imputation_features.subset_masks`.
        match / fold: match id and group-k-fold id per row ``[n]``.
        deep_block_thr: the ``deep_block`` threshold (25th percentile of ``def_line``).
        designs: design matrices built on demand per feature set.
    """

    df: pd.DataFrame
    Y: pd.DataFrame
    masks: dict[str, np.ndarray]
    match: np.ndarray
    fold: np.ndarray
    deep_block_thr: float
    designs: dict[str, pd.DataFrame] = field(default_factory=dict)

    def design(self, fset: str) -> pd.DataFrame:
        key = imf.FEATURE_SET_ALIAS.get(fset, fset)
        if key not in self.designs:
            self.designs[key] = imf.build_design(self.df, key)
        return self.designs[key]


def load_data(cfg: ImputationConfig, with_seq: bool = False) -> Data:
    """Load events360 (all ``f_*`` columns, targets, ids; ``seq_*`` on request) and assign folds."""
    path = processed_dir("soccer") / "events360.parquet"
    pf = pq.ParquetFile(path)
    names = [f.name for f in pf.schema_arrow]
    cols = ID_COLS + [c for c in names if c.startswith("f_")] + Y_LOAD
    if with_seq:
        cols += [c for c in names if c.startswith("seq_")]
    if cfg.smoke:
        df = pf.read_row_group(0, columns=cols).to_pandas()
    else:
        df = pd.read_parquet(path, columns=cols)
    df = df.reset_index(drop=True)
    thr = imf.deep_block_threshold(df)
    targets_df = imf.target_frame(df, thr)
    match = df["match_id"].to_numpy()
    fold = np.full(len(df), -1, dtype=np.int8)
    for k, (_, te) in enumerate(group_kfold(match, n_splits=cfg.n_folds, seed=cfg.seed)):
        fold[te] = k
    return Data(
        df=df, Y=targets_df, masks=imf.subset_masks(df), match=match, fold=fold, deep_block_thr=thr
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def thin_rows(idx: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Keep ``cap`` of ``idx`` uniformly at random (every match keeps the same expected share)."""
    if len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=cap, replace=False))


def lgb_params(cfg: ImputationConfig, kind: str, seed: int) -> dict[str, Any]:
    return {
        "objective": "binary" if kind == "binary" else "l2",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1,
        "max_bin": cfg.max_bin,
        "num_threads": cfg.n_jobs,
        "verbose": -1,
        "seed": seed,
    }


def fit_lgbm(
    cfg: ImputationConfig,
    x: pd.DataFrame,
    y: np.ndarray,
    match: np.ndarray,
    kind: str,
    categorical: list[str],
    seed: int,
    n_rounds: int | None = None,
) -> tuple[lgb.Booster, int]:
    """Fit one student; rounds by early stopping on an inner match holdout unless given.

    Args:
        x: training design ``[n_train, d]``; y: labels ``[n_train]``; match: match id per row.

    Returns:
        ``(booster, rounds)``.
    """
    params = lgb_params(cfg, kind, seed)
    if n_rounds is not None:
        booster = lgb.train(
            params, lgb.Dataset(x, y, categorical_feature=categorical), num_boost_round=n_rounds
        )
        return booster, n_rounds
    rng = np.random.default_rng(seed)
    uniq = np.unique(match)
    hold = rng.choice(
        uniq, size=max(1, int(round(len(uniq) * cfg.inner_holdout_frac))), replace=False
    )
    is_hold = np.isin(match, hold)
    dtrain = lgb.Dataset(x[~is_hold], y[~is_hold], categorical_feature=categorical)
    dvalid = lgb.Dataset(x[is_hold], y[is_hold], reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.max_rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
    )
    best = int(booster.best_iteration) if booster.best_iteration else cfg.max_rounds
    return booster, best


def train_rows(
    data: Data, target: str, fset: str, candidate: np.ndarray, cap: int, seed: int
) -> np.ndarray:
    """Label-valid training rows of ``candidate`` for a (target, feature set), thinned to
    ``cap``."""
    y = data.Y[target].to_numpy()
    tr = candidate[~np.isnan(y[candidate])]
    if fset == RELIABILITY_FSET:
        tr = tr[data.df["y_reliable"].to_numpy(dtype=float)[tr] == 1.0]
    return thin_rows(tr, cap, seed)


def run_fit(
    data: Data,
    cfg: ImputationConfig,
    target: str,
    fset: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a student on ``train_idx`` and predict the applicable rows of ``test_idx``.

    Returns:
        predictions aligned with ``test_idx`` (NaN outside the target's subset) and fit metadata.
    """
    spec = imf.TARGET_BY_NAME[target]
    t0 = time.time()
    tr = train_rows(data, target, fset, train_idx, cfg.train_cap, seed)
    x = data.design(fset)
    y = data.Y[target].to_numpy()
    if len(tr) < 2 * cfg.min_data_in_leaf:
        meta = {"target": target, "fset": fset, "best_iter": 0, "n_train": int(len(tr)),
                "seconds": 0.0}
        return np.full(len(test_idx), np.nan, dtype=np.float32), meta
    booster, rounds = fit_lgbm(
        cfg, x.iloc[tr], y[tr], data.match[tr], spec.kind, imf.categorical_columns(fset), seed
    )
    apply = data.masks[spec.subset][test_idx]
    pred = np.full(len(test_idx), np.nan, dtype=np.float32)
    if apply.any():
        pred[apply] = booster.predict(x.iloc[test_idx[apply]], num_iteration=rounds)
    meta = {
        "target": target,
        "fset": fset,
        "best_iter": rounds,
        "n_train": int(len(tr)),
        "seconds": time.time() - t0,
    }
    return pred, meta


def baseline_preds(
    data: Data, target: str, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, np.ndarray]:
    """Training global mean and per-event-type mean for the applicable test rows."""
    spec = imf.TARGET_BY_NAME[target]
    y = data.Y[target].to_numpy()
    tr = train_idx[~np.isnan(y[train_idx])]
    apply = data.masks[spec.subset][test_idx]
    ftype = data.df["f_type"].to_numpy(dtype=object)
    g = np.full(len(test_idx), np.nan, dtype=np.float32)
    t = np.full(len(test_idx), np.nan, dtype=np.float32)
    if len(tr) and apply.any():
        g[apply] = float(y[tr].mean())
        t[apply] = imf.bucket_mean_baseline(ftype[tr], y[tr], ftype[test_idx[apply]])
    return {"base_global": g, "base_type": t}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_path(cfg: ImputationConfig, stage: str, split: str, target: str, fset: str) -> Path:
    return cfg.cache_dir / f"{stage}__{split}__{target}__{fset}.parquet"


def save_pred(
    cfg: ImputationConfig,
    stage: str,
    split: str,
    target: str,
    fset: str,
    rows: np.ndarray,
    pred: np.ndarray,
    meta: list[dict[str, Any]] | None = None,
) -> None:
    p = cache_path(cfg, stage, split, target, fset)
    pd.DataFrame({"row": rows.astype(np.int32), "pred": pred.astype(np.float32)}).to_parquet(
        p, index=False
    )
    if meta is not None:
        p.with_suffix(".json").write_text(json.dumps(meta))


def load_pred(
    cfg: ImputationConfig, stage: str, split: str, target: str, fset: str, n: int
) -> np.ndarray | None:
    """Prediction vector over all ``n`` rows (NaN where absent) or ``None`` when not cached."""
    p = cache_path(cfg, stage, split, target, fset)
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    out = np.full(n, np.nan, dtype=np.float32)
    out[d["row"].to_numpy()] = d["pred"].to_numpy()
    return out


def load_meta(
    cfg: ImputationConfig, stage: str, split: str, target: str, fset: str
) -> list[dict[str, Any]]:
    p = cache_path(cfg, stage, split, target, fset).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else []


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def resolve_targets(arg: str | None) -> list[str]:
    if not arg or arg == "all":
        return [t.name for t in imf.TARGETS]
    out: list[str] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok in imf.TARGET_GROUPS:
            out += list(imf.TARGET_GROUPS[tok])
        elif tok in imf.TARGET_BY_NAME:
            out.append(tok)
        else:
            raise KeyError(f"unknown target or group: {tok}")
    return out


def cv_fsets_for(target: str, fsets: list[str] | None) -> list[str]:
    if fsets:
        return fsets
    spec = imf.TARGET_BY_NAME[target]
    return CV_FSETS + ([RELIABILITY_FSET] if spec.family in ("ball", "pass") else [])


def stage_cv(
    data: Data, cfg: ImputationConfig, targets: list[str], fsets: list[str] | None, force: bool
) -> None:
    """5-fold match-grouped CV: out-of-fold predictions for every row, per (target, feature set)."""
    n = len(data.df)
    all_rows = np.arange(n)
    for target in targets:
        for fset in BASELINES + cv_fsets_for(target, fsets):
            if not force and cache_path(cfg, "cv", "cv", target, fset).exists():
                continue
            t0 = time.time()
            pred = np.full(n, np.nan, dtype=np.float32)
            meta: list[dict[str, Any]] = []
            for k in range(cfg.n_folds):
                te = all_rows[data.fold == k]
                tr = all_rows[data.fold != k]
                if fset in BASELINES:
                    pred[te] = baseline_preds(data, target, tr, te)[fset]
                else:
                    p, m = run_fit(data, cfg, target, fset, tr, te, seed=cfg.seed + k)
                    pred[te] = p
                    meta.append({**m, "fold": k})
            save_pred(cfg, "cv", "cv", target, fset, all_rows, pred, meta)
            print(
                f"cv {target:28s} {fset:12s} {time.time() - t0:6.1f} s"
                + (f"  rounds {[m['best_iter'] for m in meta]}" if meta else ""),
                flush=True,
            )


def transfer_splits(
    data: Data, cfg: ImputationConfig, which: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``m2f`` / ``f2m`` (gender) or ``fwd`` (time) train / test row indices."""
    n = len(data.df)
    rows = np.arange(n)
    gender = data.df["gender"].to_numpy(dtype=object)
    if which == "transfer":
        male, female = rows[gender == "male"], rows[gender == "female"]
        return {"m2f": (male, female), "f2m": (female, male)}
    key = list(
        zip(
            data.df["competition"].to_numpy(dtype=object),
            data.df["season"].to_numpy(dtype=object),
            strict=True,
        )
    )
    is_test = np.array([k in set(cfg.forward_test) for k in key], dtype=bool)
    return {"fwd": (rows[~is_test], rows[is_test])}


def stage_transfer(
    data: Data,
    cfg: ImputationConfig,
    which: str,
    targets: list[str],
    fsets: list[str] | None,
    force: bool,
) -> None:
    """Cross-gender (``transfer``) or forward-in-time (``forward``) fits, one per (split, target,
    fset)."""
    splits = transfer_splits(data, cfg, which)
    for split, (tr, te) in splits.items():
        if len(tr) == 0 or len(te) == 0:
            print(f"{which} {split}: empty train or test set, skipped", flush=True)
            continue
        for target in targets:
            for fset in BASELINES + (fsets or TRANSFER_FSETS):
                if not force and cache_path(cfg, which, split, target, fset).exists():
                    continue
                t0 = time.time()
                if fset in BASELINES:
                    pred, meta = baseline_preds(data, target, tr, te)[fset], []
                else:
                    pred, m = run_fit(data, cfg, target, fset, tr, te, seed=cfg.seed + 11)
                    meta = [m]
                save_pred(cfg, which, split, target, fset, te, pred, meta)
                print(
                    f"{which} {split} {target:28s} {fset:12s} {time.time() - t0:6.1f} s", flush=True
                )


def mlp_design(data: Data, rows: np.ndarray) -> np.ndarray:
    """E2 one-hot / indicator design + the dense sequence encoding ``[len(rows), d]``."""
    x = data.design("E2").iloc[rows]
    e2 = imf.one_hot_design(x, imf.categorical_columns("E2"), imf.vocab_sizes())
    seq = imf.seq_mlp_features(data.df.iloc[rows])
    return np.concatenate([e2, seq], axis=1).astype(np.float32)


def stage_mlp(data: Data, cfg: ImputationConfig, targets: list[str] | None, force: bool) -> None:
    """MLP (E2 + sequence) vs LightGBM E2 / E3 on the same training rows, held-out fold 0."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    n = len(data.df)
    rows = np.arange(n)
    te, tr_all = rows[data.fold == 0], rows[data.fold != 0]
    for target in targets or list(cfg.mlp_targets):
        spec = imf.TARGET_BY_NAME[target]
        y = data.Y[target].to_numpy()
        tr = train_rows(data, target, "E2", tr_all, cfg.mlp_train_cap, cfg.seed)
        apply = data.masks[spec.subset][te]
        te_apply = te[apply]
        for model in ("lgbm_E2_same_rows", "lgbm_E3", "mlp_E2seq"):
            if not force and cache_path(cfg, "mlp", "fold0", target, model).exists():
                continue
            t0 = time.time()
            pred = np.full(len(te), np.nan, dtype=np.float32)
            if model.startswith("lgbm"):
                fset = "E2" if model == "lgbm_E2_same_rows" else "E3"
                x = data.design(fset)
                booster, rounds = fit_lgbm(
                    cfg,
                    x.iloc[tr],
                    y[tr],
                    data.match[tr],
                    spec.kind,
                    imf.categorical_columns(fset),
                    cfg.seed,
                )
                pred[apply] = booster.predict(x.iloc[te_apply], num_iteration=rounds)
                meta = {"n_train": int(len(tr)), "best_iter": rounds, "n_features": int(x.shape[1])}
            else:
                x_tr = mlp_design(data, tr)
                mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
                reg = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=cfg.mlp_hidden,
                        batch_size=512,
                        learning_rate_init=1e-3,
                        max_iter=cfg.mlp_max_iter,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=cfg.seed,
                    ),
                )
                with threadpool_limits(limits=cfg.n_jobs):
                    reg.fit(x_tr, ((y[tr] - mu) / sd).astype(np.float32))
                    del x_tr
                    for chunk in np.array_split(
                        np.arange(len(te_apply)), max(1, len(te_apply) // 50_000)
                    ):
                        x_te = mlp_design(data, te_apply[chunk])
                        pred[np.where(apply)[0][chunk]] = reg.predict(x_te) * sd + mu
                mlp = reg[-1]
                meta = {
                    "n_train": int(len(tr)),
                    "best_iter": int(mlp.n_iter_),
                    "n_features": int(mlp.n_features_in_),
                    "best_validation_score": float(mlp.best_validation_score_),
                }
            save_pred(
                cfg,
                "mlp",
                "fold0",
                target,
                model,
                te,
                pred,
                [{**meta, "target": target, "model": model, "seconds": time.time() - t0}],
            )
            print(f"mlp {target:28s} {model:18s} {time.time() - t0:6.1f} s  {meta}", flush=True)


def stage_final(
    data: Data, cfg: ImputationConfig, fsets: list[str] | None, targets: list[str] | None = None
) -> None:
    """Fit the final students on all 360 matches and save one bundle per feature set.

    With ``targets`` only those students are (re)fitted and merged into the existing bundle.
    """
    n = len(data.df)
    rows = np.arange(n)
    for fset in fsets or FINAL_FSETS:
        bundle = aps.StudentBundle(
            feature_set=fset,
            features=imf.design_columns(fset),
            categorical=imf.categorical_columns(fset),
            deep_block_threshold=data.deep_block_thr,
        )
        if targets and aps.StudentBundle.path(fset).exists() and not cfg.smoke:
            bundle = aps.StudentBundle.load(fset)
        x = data.design(fset)
        for spec in imf.TARGETS:
            if targets and spec.name not in targets:
                continue
            t0 = time.time()
            y = data.Y[spec.name].to_numpy()
            tr = train_rows(data, spec.name, fset, rows, cfg.final_cap, cfg.seed + 101)
            meta = load_meta(cfg, "cv", "cv", spec.name, fset)
            rounds = int(np.median([m["best_iter"] for m in meta])) if meta else 300
            booster, _ = fit_lgbm(
                cfg,
                x.iloc[tr],
                y[tr],
                data.match[tr],
                spec.kind,
                imf.categorical_columns(fset),
                cfg.seed + 101,
                n_rounds=rounds,
            )
            oof = load_pred(cfg, "cv", "cv", spec.name, fset, n)
            lab = ~np.isnan(y)
            skill = float("nan")
            if oof is not None:
                ok = lab & ~np.isnan(oof)
                if spec.kind == "binary":
                    base = load_pred(cfg, "cv", "cv", spec.name, "base_global", n)
                    skill = (
                        1.0 - brier(y[ok], oof[ok]) / brier(y[ok], base[ok])
                        if base is not None
                        else float("nan")
                    )
                else:
                    skill = r2(y[ok], oof[ok])
            bundle.students[spec.name] = aps.Student(
                target=spec.name,
                kind=spec.kind,
                subset=spec.subset,
                model_str=booster.model_to_string(),
                n_rounds=rounds,
                train_n=int(len(tr)),
                y_mean=float(y[tr].mean()),
                y_sd=float(y[tr].std()),
                oof_mean=float(np.nanmean(oof)) if oof is not None else float("nan"),
                oof_sd=float(np.nanstd(oof)) if oof is not None else float("nan"),
                cv_skill=skill,
            )
            print(
                f"final {fset:5s} {spec.name:28s} rounds {rounds:4d} n_train {len(tr):7d} "
                f"{time.time() - t0:6.1f} s",
                flush=True,
            )
        if cfg.write and not cfg.smoke:
            print("saved", bundle.save())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    kind: str, y: np.ndarray, p: np.ndarray, base: np.ndarray | None = None
) -> dict[str, float]:
    """Metrics of a prediction vector on the rows where both label and prediction exist."""
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    if len(y) < 10:
        return {"n": int(len(y))}
    if kind == "binary":
        pc = np.clip(p, 0.0, 1.0)
        b = brier(y, pc)
        ref = brier(y, base[ok]) if base is not None else brier(y, np.full(len(y), y.mean()))
        cal = calibration_table(y, pc, 10)
        ece = float(sum(abs(r["pred"] - r["obs"]) * r["n"] for r in cal) / len(y))
        return {
            "n": int(len(y)),
            "base_rate": float(y.mean()),
            "mean_pred": float(pc.mean()),
            "auc": float(roc_auc_score(y, pc)) if len(np.unique(y)) > 1 else np.nan,
            "log_loss": log_loss(y, pc),
            "brier": b,
            "bss": float(1.0 - b / ref) if ref > 0 else np.nan,
            "ece": ece,
        }
    d = {
        "n": int(len(y)),
        "r2": r2(y, p),
        "mae": mae(y, p),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
        "sd_y": float(y.std()),
    }
    if kind == "count":
        ex, w1 = imf.rounded_agreement(y, p)
        d.update({"exact_rounded": ex, "within_1": w1})
    return d


def skill_of(kind: str, s: dict[str, float]) -> float:
    return float(s.get("bss", np.nan)) if kind == "binary" else float(s.get("r2", np.nan))


def per_sample_loss(kind: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    if kind == "binary":
        return per_sample_log_loss(y, np.clip(p, 0.0, 1.0))
    return (y - p) ** 2


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Preds:
    """Cached predictions of one stage / split: ``{target: {fset: pred[n]}}``."""

    stage: str
    split: str
    by_target: dict[str, dict[str, np.ndarray]]


def load_stage(cfg: ImputationConfig, stage: str, split: str, n: int) -> Preds:
    out: dict[str, dict[str, np.ndarray]] = {}
    for p in sorted(cfg.cache_dir.glob(f"{stage}__{split}__*.parquet")):
        _, _, target, fset = p.stem.split("__")
        pred = load_pred(cfg, stage, split, target, fset, n)
        if pred is not None:
            out.setdefault(target, {})[fset] = pred
    return Preds(stage=stage, split=split, by_target=out)


def _row(target: str, fset: str, extra: dict[str, Any], s: dict[str, float]) -> dict[str, Any]:
    spec = imf.TARGET_BY_NAME[target]
    return {
        "target": target,
        "kind": spec.kind,
        "family": spec.family,
        "feature_set": fset,
        **extra,
        **s,
    }


def metrics_table(
    data: Data, cv: Preds, sel: np.ndarray | None = None, extra: dict[str, Any] | None = None
) -> pd.DataFrame:
    """One row per (target, feature set) with :func:`score` on ``sel`` rows (all rows by
    default)."""
    rows = []
    for target, preds in cv.by_target.items():
        spec = imf.TARGET_BY_NAME[target]
        y = data.Y[target].to_numpy()
        if sel is not None:
            y = np.where(sel, y, np.nan)
        base = preds.get("base_global")
        for fset in BASELINES + [
            f for f in imf.FEATURE_SET_ORDER + [RELIABILITY_FSET] if f in preds
        ]:
            if fset not in preds:
                continue
            rows.append(_row(target, fset, extra or {}, score(spec.kind, y, preds[fset], base)))
    return pd.DataFrame(rows)


def deltas_table(data: Data, cv: Preds, cfg: ImputationConfig) -> pd.DataFrame:
    """Match-clustered bootstrap deltas of the per-row loss between nested feature sets."""
    rows = []
    for target, preds in cv.by_target.items():
        spec = imf.TARGET_BY_NAME[target]
        y = data.Y[target].to_numpy()
        for a, b in DELTA_PAIRS:
            if a not in preds or b not in preds:
                continue
            ok = ~(np.isnan(y) | np.isnan(preds[a]) | np.isnan(preds[b]))
            if ok.sum() < 100:
                continue
            la = per_sample_loss(spec.kind, y[ok], preds[a][ok])
            lb = per_sample_loss(spec.kind, y[ok], preds[b][ok])
            d, lo, hi = clustered_bootstrap_delta(
                la, lb, data.match[ok], n_boot=cfg.n_boot, seed=cfg.seed
            )
            sa, sb = (
                score(spec.kind, y, preds[a], preds.get("base_global")),
                score(spec.kind, y, preds[b], preds.get("base_global")),
            )
            rows.append(
                {
                    "target": target,
                    "kind": spec.kind,
                    "from": a,
                    "to": b,
                    "n": int(ok.sum()),
                    "loss": "log_loss" if spec.kind == "binary" else "squared_error",
                    "delta_loss": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "significant": bool(lo > 0 or hi < 0),
                    "skill_from": skill_of(spec.kind, sa),
                    "skill_to": skill_of(spec.kind, sb),
                }
            )
    return pd.DataFrame(rows)


def breakdown_table(
    data: Data,
    cv: Preds,
    fset: str,
    key: np.ndarray,
    key_name: str,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    """Per-bin metrics (n, mae / rmse / within-bin r2 / bias, or binary metrics) of one feature
    set."""
    rows = []
    keys = pd.Series(np.asarray(key, dtype=object))
    for target, preds in cv.by_target.items():
        if targets and target not in targets or fset not in preds:
            continue
        spec = imf.TARGET_BY_NAME[target]
        y = data.Y[target].to_numpy()
        p = preds[fset]
        base = preds.get("base_global")
        for k in keys.dropna().unique():
            m = (keys == k).to_numpy()
            s = score(spec.kind, np.where(m, y, np.nan), p, base)
            if s["n"] >= 200:
                rows.append(
                    {"target": target, "kind": spec.kind, "feature_set": fset, key_name: k, **s}
                )
    return pd.DataFrame(rows)


def transfer_table(data: Data, cv: Preds, cfg: ImputationConfig) -> pd.DataFrame:
    """Transferred students vs the in-domain CV predictions on the same test rows."""
    rows = []
    n = len(data.df)
    for stage, split in (("transfer", "m2f"), ("transfer", "f2m"), ("forward", "fwd")):
        pr = load_stage(cfg, stage, split, n)
        for target, preds in pr.by_target.items():
            spec = imf.TARGET_BY_NAME[target]
            y = data.Y[target].to_numpy()
            test = ~np.isnan(preds.get("base_global", np.full(n, np.nan)))
            y_test = np.where(test, y, np.nan)
            cv_preds = cv.by_target.get(target, {})
            for fset in BASELINES + TRANSFER_FSETS:
                if fset not in preds:
                    continue
                s = score(spec.kind, y_test, preds[fset], preds.get("base_global"))
                row = {
                    "split": split,
                    "target": target,
                    "kind": spec.kind,
                    "feature_set": fset,
                    **s,
                }
                if fset in cv_preds:
                    s_in = score(spec.kind, y_test, cv_preds[fset], cv_preds.get("base_global"))
                    row["skill_transfer"] = skill_of(spec.kind, s)
                    row["skill_in_domain"] = skill_of(spec.kind, s_in)
                    row["mae_in_domain"] = s_in.get("mae", np.nan)
                    row["log_loss_in_domain"] = s_in.get("log_loss", np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def mlp_table(data: Data, cfg: ImputationConfig, cv: Preds) -> pd.DataFrame:
    n = len(data.df)
    pr = load_stage(cfg, "mlp", "fold0", n)
    rows = []
    for target, preds in pr.by_target.items():
        spec = imf.TARGET_BY_NAME[target]
        y = data.Y[target].to_numpy()
        fold0 = data.fold == 0
        y0 = np.where(fold0, y, np.nan)
        cvp = cv.by_target.get(target, {})
        for model in ("base_type", "loc", "E2", "E2a"):
            if model in cvp:
                rows.append(
                    {
                        "target": target,
                        "model": f"cv_{model} (train_cap rows)",
                        "n_train": np.nan,
                        "n_features": len(imf.design_columns(model))
                        if model in imf.FEATURE_SETS
                        else np.nan,
                        **score(spec.kind, y0, cvp[model]),
                    }
                )
        for model, p in preds.items():
            meta = load_meta(cfg, "mlp", "fold0", target, model)
            m = meta[0] if meta else {}
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "n_train": m.get("n_train", np.nan),
                    "n_features": m.get("n_features", np.nan),
                    "epochs_or_rounds": m.get("best_iter", np.nan),
                    "seconds": m.get("seconds", np.nan),
                    **score(spec.kind, y0, p),
                }
            )
    return pd.DataFrame(rows)


def calibration_frame(data: Data, cv: Preds, fset: str = "E2") -> pd.DataFrame:
    rows = []
    for target, preds in cv.by_target.items():
        spec = imf.TARGET_BY_NAME[target]
        if spec.kind != "binary" or fset not in preds:
            continue
        y = data.Y[target].to_numpy()
        p = preds[fset]
        ok = ~(np.isnan(y) | np.isnan(p))
        for i, r in enumerate(calibration_table(y[ok], np.clip(p[ok], 0, 1), 10)):
            rows.append({"target": target, "feature_set": fset, "bin": i, **r})
    return pd.DataFrame(rows)


def importance_table(fsets: tuple[str, ...] = ("E2", "E2a"), top: int = 10) -> pd.DataFrame:
    rows = []
    for fset in fsets:
        try:
            b = aps.StudentBundle.load(fset)
        except FileNotFoundError:
            continue
        for name, st in b.students.items():
            booster = st.booster()
            gain = np.asarray(booster.feature_importance("gain"), dtype=float)
            tot = gain.sum() or 1.0
            order = np.argsort(-gain)[:top]
            for rank, j in enumerate(order):
                rows.append(
                    {
                        "feature_set": fset,
                        "target": name,
                        "rank": rank + 1,
                        "feature": b.features[j],
                        "gain_share": float(gain[j] / tot),
                    }
                )
    return pd.DataFrame(rows)


def targets_table(data: Data) -> pd.DataFrame:
    rows = []
    for t in imf.TARGETS:
        y = data.Y[t.name].to_numpy()
        lab = ~np.isnan(y)
        row = {
            "target": t.name,
            "kind": t.kind,
            "family": t.family,
            "subset": t.subset,
            "label_rule": t.label,
            "n_subset": int(data.masks[t.subset].sum()),
            "n_labelled": int(lab.sum()),
            "mean": float(y[lab].mean()) if lab.any() else np.nan,
            "sd": float(y[lab].std()) if lab.any() else np.nan,
            "definition": t.definition,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_table(metrics: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Per target: skill of every feature set (CV), the transfer skills of E2 and a verdict."""
    rows = []
    for target, g in metrics.groupby("target", sort=False):
        kind = g["kind"].iloc[0]
        sk = {r["feature_set"]: skill_of(kind, r) for _, r in g.iterrows()}
        row: dict[str, Any] = {
            "target": target,
            "kind": kind,
            "family": g["family"].iloc[0],
            "n": int(g["n"].max()),
            "skill_metric": "bss" if kind == "binary" else "r2",
        }
        for f in ["base_type", "loc", "E0", "E1", "E2", "E2a", RELIABILITY_FSET]:
            row[f] = sk.get(f, np.nan)
        e2 = g[g["feature_set"] == "E2"]
        row["mae_E2" if kind != "binary" else "auc_E2"] = (
            float(e2["mae"].iloc[0] if kind != "binary" else e2["auc"].iloc[0])
            if len(e2)
            else np.nan
        )
        for split in ("m2f", "f2m", "fwd"):
            t = transfer[
                (transfer["split"] == split)
                & (transfer["target"] == target)
                & (transfer["feature_set"] == "E2")
            ] if "split" in transfer.columns else transfer.iloc[:0]
            row[f"E2_{split}"] = float(t["skill_transfer"].iloc[0]) if len(t) else np.nan
            row[f"in_domain_{split}"] = float(t["skill_in_domain"].iloc[0]) if len(t) else np.nan
        e2s, loc = row["E2"], row["loc"]
        row["gain_E2_over_loc"] = e2s - loc if not (np.isnan(e2s) or np.isnan(loc)) else np.nan
        row["verdict"] = (
            ("recoverable" if e2s >= 0.5 else "partly" if e2s >= 0.2 else "weak")
            if not np.isnan(e2s)
            else "n/a"
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("E2", ascending=False).reset_index(drop=True)


def write_oof(data: Data, cv: Preds, path: Path) -> None:
    """Out-of-fold predictions for every row (ids, context, labels, ``<target>__<fset>``)."""
    out = data.df[
        [
            "match_id",
            "event_id",
            "event_index",
            "period",
            "competition",
            "season",
            "gender",
            "match_date",
            "f_type",
            "f_x",
            "f_is_possession_team",
            "f_play_pattern",
            "y_reliable",
            "y_frame_ok",
        ]
    ].copy()
    out["fold"] = data.fold
    cols = {}
    for t in imf.TARGETS:
        cols[f"y_{t.name}"] = data.Y[t.name].to_numpy(dtype=np.float32)
        for fset, p in cv.by_target.get(t.name, {}).items():
            cols[f"{t.name}__{fset}"] = p
    out = pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)
    out.to_parquet(path, index=False)


def stage_report(
    data: Data, cfg: ImputationConfig, seq_df: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """Assemble every table from the caches, write the parquet tables, the OOF file and the
    markdown report."""
    n = len(data.df)
    cv = load_stage(cfg, "cv", "cv", n)
    df = data.df
    tables: dict[str, pd.DataFrame] = {}
    tables["targets"] = targets_table(data)
    tables["feature_sets"] = pd.DataFrame(
        [
            {
                "feature_set": f,
                "n_features": len(imf.design_columns(f)),
                "description": imf.FEATURE_SET_DESCRIPTION[f],
            }
            for f in ["loc", "E0", "E1", "E2", "E2a", "E2r", "E3"]
        ]
    )
    tables["metrics"] = metrics_table(data, cv)
    tables["deltas"] = deltas_table(data, cv, cfg)
    rel = df["y_reliable"].to_numpy(dtype=float) == 1.0
    tables["reliability"] = pd.concat(
        [
            metrics_table(data, cv, rel, {"eval_rows": "reliable"}),
            metrics_table(data, cv, ~rel, {"eval_rows": "unreliable"}),
        ],
        ignore_index=True,
    )
    tables["reliability"] = tables["reliability"][
        tables["reliability"]["family"].isin(["ball", "pass"])
    ]
    ftype = df["f_type"].to_numpy(dtype=object)
    tables["by_type"] = pd.concat(
        [breakdown_table(data, cv, f, ftype, "f_type") for f in ("base_type", "E2")],
        ignore_index=True,
    )
    tables["by_competition"] = breakdown_table(
        data,
        cv,
        "E2",
        (df["competition"].astype(str) + " " + df["season"].astype(str)).to_numpy(dtype=object),
        "competition",
    )
    tables["by_third"] = breakdown_table(
        data, cv, "E2", imf.pitch_third(df["f_x"].to_numpy()), "third"
    )
    tables["by_phase"] = breakdown_table(
        data, cv, "E2", imf.phase_label(df["f_play_pattern"].to_numpy(dtype=object)), "phase"
    )
    tables["by_play_pattern"] = breakdown_table(
        data, cv, "E2", df["f_play_pattern"].to_numpy(dtype=object), "play_pattern"
    )
    el = pd.cut(df["f_poss_elapsed"].to_numpy(dtype=float), ELAPSED_BINS, right=False).astype(str)
    tables["by_poss_elapsed"] = breakdown_table(
        data, cv, "E2", el.to_numpy(dtype=object), "poss_elapsed_s"
    )
    ts = df["f_t_since_opp_def_action"].to_numpy(dtype=float)
    tsb = pd.cut(ts, SINCE_DEF_S_BINS, right=False).astype(str).to_numpy(dtype=object)
    tsb[np.isnan(ts)] = "none this period"
    tables["by_since_opp_def_s"] = breakdown_table(data, cv, "E2", tsb, "since_opp_def_action_s")
    if seq_df is not None:
        ne = imf.events_since_opp_def_action(seq_df)
        neb = (
            pd.cut(ne.astype(float), SINCE_DEF_N_BINS, right=False)
            .astype(str)
            .to_numpy(dtype=object)
        )
        neb[ne >= imf.SEQ_LEN] = "20+ / none"
        tables["by_since_opp_def_n"] = breakdown_table(
            data, cv, "E2", neb, "events_since_opp_def_action"
        )
    tables["transfer"] = transfer_table(data, cv, cfg)
    tables["mlp"] = mlp_table(data, cfg, cv)
    tables["calibration"] = calibration_frame(data, cv)
    tables["importance"] = importance_table() if not cfg.smoke else pd.DataFrame()
    fits = []
    for target, preds in cv.by_target.items():
        for fset in preds:
            for m in load_meta(cfg, "cv", "cv", target, fset):
                fits.append(m)
    tables["fits"] = pd.DataFrame(fits)
    tables["ranking"] = (
        ranking_table(tables["metrics"], tables["transfer"])
        if len(tables["metrics"])
        else pd.DataFrame()
    )
    if cfg.write and not cfg.smoke:
        rdir = reports_dir()
        for name, t in tables.items():
            if len(t):
                t.to_parquet(rdir / f"soccer_02_{name}.parquet", index=False)
        write_oof(data, cv, processed_dir("soccer") / "imputed_oof.parquet")
        (rdir / "soccer_02_imputation.md").write_text(render_report(data, cfg, tables))
        print("wrote", rdir / "soccer_02_imputation.md")
    return tables


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pivot_skill(metrics: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for target, g in metrics.groupby("target", sort=False):
        kind = g["kind"].iloc[0]
        row = {"target": target, "kind": kind, "n": int(g["n"].max())}
        for f in cols:
            r = g[g["feature_set"] == f]
            row[f] = skill_of(kind, r.iloc[0]) if len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pivot_metric(
    metrics: pd.DataFrame, metric: str, cols: list[str], kinds: tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for target, g in metrics.groupby("target", sort=False):
        if g["kind"].iloc[0] not in kinds:
            continue
        row = {"target": target, "n": int(g["n"].max())}
        for f in cols:
            r = g[g["feature_set"] == f]
            row[f] = float(r[metric].iloc[0]) if len(r) and metric in r else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _breakdown_md(t: pd.DataFrame, key: str, targets: list[str], metric_reg: str = "mae") -> str:
    """Pivot a breakdown table: rows = bins, columns = targets (mae, or log_loss for binaries)."""
    if not len(t):
        return "(not available)"
    parts = []
    for target in targets:
        g = t[t["target"] == target]
        if not len(g):
            continue
        kind = g["kind"].iloc[0]
        m = metric_reg if kind != "binary" else "log_loss"
        parts.append(g.set_index(key)[m].rename(f"{target} ({m})"))
        if len(parts) == 1:
            parts.insert(0, g.set_index(key)["n"].rename(f"n ({target})"))
    if not parts:
        return "(not available)"
    piv = pd.concat(parts, axis=1)
    piv.index.name = key
    piv = piv.loc[sorted(piv.index, key=_bin_sort_key)]
    return md_table(piv.reset_index(), floatfmt="{:.3f}")


def _bin_sort_key(label: object) -> tuple[float, str]:
    """Numeric lower bound of an interval label (``[a, b)``), non-numeric labels last."""
    text = str(label)
    if text[:1] in "[(":
        try:
            return (float(text[1:].split(",")[0]), text)
        except ValueError:
            pass
    return (float("inf"), text)


def render_report(data: Data, cfg: ImputationConfig, tables: dict[str, pd.DataFrame]) -> str:
    df = data.df
    n = len(df)
    n_matches = int(df["match_id"].nunique())
    metrics = tables["metrics"]
    fsets = ["base_global", "base_type", "loc", "E0", "E1", "E2", "E2a", "E2r"]
    lines: list[str] = []
    lines.append("# Soccer 02 - imputation: event-only students of the 360 defensive state\n")
    lines.append(
        f"Source: `processed_dir('soccer')/events360.parquet` ({n:,} rows, {n_matches} matches "
        "with usable 360; "
        "see `soccer_01_build.md`). Students are LightGBM models (`num_threads=2`) that predict a "
        "360-derived "
        "target from `f_*` event-only features; the neural student is a scikit-learn "
        "`MLPRegressor` because "
        "PyTorch is not installed. Everything is scored out of sample by match; every number "
        "states its n.\n"
    )
    lines.append("## Protocol\n")
    lines.append(
        f"- Splits: (i) {cfg.n_folds}-fold `group_kfold` by `match_id` over all {n_matches} "
        "matches (out-of-fold "
        "prediction for every row); (ii) cross-competition, men -> women (`m2f`) and women -> men "
        "(`f2m`); "
        "(iii) forward in time, every competition-season up to 2023/24 -> "
        f"{' + '.join(f'{c} {s}' for c, s in cfg.forward_test)} (`fwd`)."
    )
    lines.append(
        f"- Training rows are thinned uniformly at random to {cfg.train_cap:,} per fit "
        f"({cfg.final_cap:,} for the "
        "final students); every match keeps the same expected share. Rounds are chosen by early "
        "stopping "
        f"(patience {cfg.early_stopping}, max {cfg.max_rounds}) on an inner "
        f"{int(cfg.inner_holdout_frac * 100)}% "
        "match holdout of the training rows, never on the test rows. LightGBM: lr "
        f"{cfg.learning_rate}, "
        f"{cfg.num_leaves} leaves, min_data_in_leaf {cfg.min_data_in_leaf}, feature / bagging "
        "fraction "
        f"{cfg.feature_fraction} / {cfg.bagging_fraction}, max_bin {cfg.max_bin}, seed "
        f"{cfg.seed}; L2 objective for "
        "continuous and count targets, binary log-loss for the two binaries."
    )
    lines.append(
        "- Team-shape targets (`shape` family) are trained and scored on possession-team events "
        "with a reliable "
        "frame (`y_reliable == 1`: orientation ok or repaired and >= 7 opponents visible); "
        "ball-relative and "
        "pass targets on every usable frame (`y_frame_ok == 1`), with the reliability sensitivity "
        "below. "
        f"`deep_block` = `def_line <= {data.deep_block_thr:.2f}` yd (25th percentile over "
        "reliable settled "
        "possession events: Regular Play, possession >= 10 s old); `counter_on` = "
        "`n_opp_ahead_of_ball <= 3` on "
        "possession-team events with the ball in the middle third."
    )
    lines.append(
        "- Baselines: `base_global` (training mean / base rate), `base_type` (training mean per "
        "event type), "
        "`loc` (LightGBM on x, y, distance / opening angle / bearing to goal). Skill = R2 for "
        "continuous and "
        "count targets, Brier skill score vs `base_global` for binaries. Paired deltas use a "
        "match-clustered "
        f"bootstrap ({cfg.n_boot} resamples, 95% CI)."
    )
    lines.append(
        "- Every experiment is reported twice: pre-instant features only (`E0`, `E1`, `E2`) and "
        "with the 16 "
        "POST-INSTANT `f_after_*` columns (`E2a`). `E2r` is E2 trained on reliable rows only.\n"
    )
    lines.append("### Targets\n")
    lines.append(
        md_table(
            tables["targets"][
                [
                    "target",
                    "kind",
                    "family",
                    "subset",
                    "label_rule",
                    "n_subset",
                    "n_labelled",
                    "mean",
                    "sd",
                    "definition",
                ]
            ],
            floatfmt="{:.3f}",
        )
    )
    lines.append("\n### Feature sets\n")
    lines.append(md_table(tables["feature_sets"]))
    lines.append("\n## What is recoverable (5-fold CV, ranked by E2 skill)\n")
    lines.append(
        "Skill is R2 (continuous / count) or Brier skill score vs the base rate (binary). "
        "`E2_m2f` / `E2_f2m` / "
        "`E2_fwd` are the E2 student trained on the other gender / on the earlier competitions "
        "and scored on the "
        "held-out domain; `in_domain_*` is the CV out-of-fold E2 prediction scored on the same "
        "rows. "
        "Verdict: recoverable (E2 skill >= 0.5), partly (0.2-0.5), weak (< 0.2).\n"
    )
    if len(tables["ranking"]):
        rk = tables["ranking"]
        lines.append(
            md_table(
                rk[
                    [
                        "target",
                        "kind",
                        "n",
                        "base_type",
                        "loc",
                        "E0",
                        "E1",
                        "E2",
                        "E2a",
                        "E2r",
                        "gain_E2_over_loc",
                        "E2_m2f",
                        "E2_f2m",
                        "E2_fwd",
                        "in_domain_fwd",
                        "verdict",
                    ]
                ],
                floatfmt="{:.3f}",
            )
        )
    lines.append("\n## 5-fold CV: skill by feature set\n")
    lines.append(md_table(_pivot_skill(metrics, fsets), floatfmt="{:.3f}"))
    lines.append("\n### MAE (continuous and count targets)\n")
    lines.append(
        md_table(_pivot_metric(metrics, "mae", fsets, ("reg", "count")), floatfmt="{:.3f}")
    )
    cnt = _pivot_metric(metrics, "within_1", fsets, ("count",))
    if len(cnt):
        lines.append("\n### Count targets: share of rows whose rounded prediction is within +/-1\n")
        lines.append(md_table(cnt, floatfmt="{:.3f}"))
    b = metrics[metrics["kind"] == "binary"]
    if len(b):
        lines.append("\n### Binary targets\n")
        lines.append(
            md_table(
                b[
                    [
                        "target",
                        "feature_set",
                        "n",
                        "base_rate",
                        "mean_pred",
                        "auc",
                        "log_loss",
                        "brier",
                        "bss",
                        "ece",
                    ]
                ],
                floatfmt="{:.4f}",
            )
        )
    lines.append("\n### Paired deltas (match-clustered bootstrap)\n")
    lines.append(
        "`delta_loss` = mean per-row loss of `from` minus `to` (positive = `to` is better); "
        "squared error for "
        "continuous / count targets, log-loss for binaries.\n"
    )
    if len(tables["deltas"]):
        lines.append(
            md_table(
                tables["deltas"][
                    [
                        "target",
                        "from",
                        "to",
                        "n",
                        "loss",
                        "delta_loss",
                        "ci_low",
                        "ci_high",
                        "significant",
                        "skill_from",
                        "skill_to",
                    ]
                ],
                floatfmt="{:.4f}",
            )
        )
    lines.append("\n## Reliability sensitivity (ball-relative and pass targets)\n")
    lines.append(
        "Students trained on all usable frames (`E2`) or on reliable frames only (`E2r`), scored "
        "separately on "
        "reliable (>= 7 opponents visible) and unreliable evaluation rows. Unreliable frames "
        "under-count "
        "opponents by construction, so their labels are biased low.\n"
    )
    rel = tables["reliability"]
    if len(rel):
        rows = []
        for (target, ev), g in rel.groupby(["target", "eval_rows"], sort=False):
            kind = g["kind"].iloc[0]
            row = {"target": target, "eval_rows": ev, "n": int(g["n"].max())}
            for f in ["base_type", "E2", "E2r", "E2a"]:
                r = g[g["feature_set"] == f]
                row[f"skill_{f}"] = skill_of(kind, r.iloc[0]) if len(r) else np.nan
                row[f"mae_{f}"] = float(r["mae"].iloc[0]) if len(r) and "mae" in r else np.nan
            rows.append(row)
        lines.append(md_table(pd.DataFrame(rows), floatfmt="{:.3f}"))
    lines.append("\n## Skill by event type (E2 vs per-type mean)\n")
    lines.append(
        "Within-type R2 (or BSS) of the E2 student; the per-type baseline has zero skill within a "
        "type by "
        "construction, so this isolates what the event-only features add beyond the type. Contact "
        "events "
        "(Dispossessed, Foul Won, Dribbled Past, aerial Duels) have a trivially predictable "
        "nearest opponent "
        "(the paired actor at ~0.14 yd), which inflates their all-type skill; within those "
        "types the R2 of the shared student is negative because its residual (MAE ~0.2 yd) "
        "exceeds the type's own spread (sd ~0.2 yd).\n"
    )
    bt = tables["by_type"]
    if len(bt):
        top_types = [
            "Pass",
            "Ball Receipt*",
            "Carry",
            "Pressure",
            "Ball Recovery",
            "Duel",
            "Clearance",
            "Block",
            "Dribble",
            "Shot",
            "Miscontrol",
            "Interception",
            "Foul Won",
            "Dispossessed",
        ]
        e2 = bt[(bt["feature_set"] == "E2") & bt["f_type"].isin(top_types)]
        for target in [t for t in HEADLINE if t in set(e2["target"])]:
            g = e2[e2["target"] == target]
            kind = g["kind"].iloc[0]
            cols = (
                ["f_type", "n", "r2", "mae", "sd_y", "bias"]
                if kind != "binary"
                else ["f_type", "n", "base_rate", "auc", "log_loss", "bss"]
            )
            lines.append(f"\n**{target}**\n")
            lines.append(md_table(g[cols], floatfmt="{:.3f}"))
    lines.append("\n## Cross-competition and forward transfer\n")
    lines.append(
        "`skill_transfer`: student trained on the source domain only, scored on the target domain; "
        "`skill_in_domain`: the CV out-of-fold prediction (trained on 4/5 of all matches, i.e. "
        "including the "
        "target domain's other matches) scored on the same rows. `base_*` rows are the "
        "source-domain means. "
        "Training rows of the transfer fits are thinned to the same cap as the CV fits.\n"
    )
    tr = tables["transfer"]
    if len(tr):
        for split, label in (
            ("m2f", "men -> women"),
            ("f2m", "women -> men"),
            ("fwd", "<= 2023/24 -> Euro 2024 + Women's Euro 2025"),
        ):
            g = tr[(tr["split"] == split) & (tr["feature_set"] != "base_global")]
            if not len(g):
                continue
            lines.append(f"\n**{split}: {label}**\n")
            cols = [
                "target",
                "feature_set",
                "n",
                "skill_transfer",
                "skill_in_domain",
                "mae",
                "mae_in_domain",
                "auc",
                "log_loss",
                "log_loss_in_domain",
            ]
            lines.append(md_table(g[[c for c in cols if c in g.columns]], floatfmt="{:.3f}"))
    lines.append("\n## Error analysis (E2 student, out-of-fold)\n")
    lines.append(
        "MAE per bin for continuous / count targets, log-loss for binaries (full tables incl. n, "
        "RMSE, within-bin "
        "R2 and bias in `soccer_02_by_*.parquet`).\n"
    )
    errs = [t for t in HEADLINE if t in set(metrics["target"])]
    for key, name, title in (
        ("poss_elapsed_s", "by_poss_elapsed", "Elapsed possession time (s)"),
        (
            "events_since_opp_def_action",
            "by_since_opp_def_n",
            "Located events since the opponent's last defensive action (from the sequence block; "
            "20+ = none in the last 20 or none this period)",
        ),
        (
            "since_opp_def_action_s",
            "by_since_opp_def_s",
            "Seconds since the opponent's last defensive action",
        ),
        ("phase", "by_phase", "Set-piece phase vs open play"),
        ("third", "by_third", "Pitch third of the ball (event team's frame)"),
        ("competition", "by_competition", "Competition"),
    ):
        if name in tables and len(tables[name]):
            lines.append(f"\n### {title}\n")
            lines.append(_breakdown_md(tables[name], key, errs))
    lines.append("\n## Neural student vs LightGBM (held-out fold 0)\n")
    lines.append(
        "`mlp_E2seq`: scikit-learn `MLPRegressor` on the one-hot E2 design plus a dense per-slot "
        "encoding of the "
        f"20-event sequence block (hidden {cfg.mlp_hidden}, Adam, batch 512, early stopping on a "
        "random 10% of the "
        f"training rows, max {cfg.mlp_max_iter} epochs), trained on {cfg.mlp_train_cap:,} rows of "
        "folds 1-4. "
        "`lgbm_E2_same_rows` / `lgbm_E3` are LightGBM on the same rows with E2 / E2 + raw "
        "sequence block; "
        "`cv_*` rows are the CV students (trained on the larger cap) scored on the same fold.\n"
    )
    if len(tables["mlp"]):
        mlp_t = tables["mlp"].copy()
        for c in ("n_train", "n_features", "epochs_or_rounds"):
            if c in mlp_t.columns:
                mlp_t[c] = mlp_t[c].map(lambda v: "" if pd.isna(v) else str(int(v)))
        lines.append(
            md_table(
                mlp_t[
                    [
                        c
                        for c in [
                            "target",
                            "model",
                            "n_train",
                            "n_features",
                            "epochs_or_rounds",
                            "seconds",
                            "n",
                            "r2",
                            "mae",
                            "rmse",
                        ]
                        if c in mlp_t.columns
                    ]
                ],
                floatfmt="{:.3f}",
            )
        )
    lines.append("\n## Calibration of the binaries (E2, out-of-fold, equal-count bins)\n")
    if len(tables["calibration"]):
        lines.append(md_table(tables["calibration"], floatfmt="{:.3f}"))
    lines.append("\n## Feature importance of the final students (gain share, top 10)\n")
    imp = tables["importance"]
    if len(imp):
        for target in [t for t in HEADLINE if t in set(imp["target"])]:
            g = imp[(imp["feature_set"] == "E2") & (imp["target"] == target)]
            lines.append(
                f"\n**{target} (E2)**: "
                + ", ".join(f"{r.feature} {r.gain_share:.2f}" for r in g.itertuples())
            )
    fits = tables["fits"]
    if len(fits):
        lines.append("\n\n## Fit summary (CV)\n")
        fs = (
            fits.groupby(["target", "fset"])
            .agg(
                n_train_mean=("n_train", "mean"),
                rounds_median=("best_iter", "median"),
                rounds_min=("best_iter", "min"),
                rounds_max=("best_iter", "max"),
                seconds_mean=("seconds", "mean"),
            )
            .reset_index()
        )
        lines.append(md_table(fs, floatfmt="{:.1f}"))
    lines.append("\n## Applied to the non-360 matches\n")
    lines.append(
        "`apply_student.py` writes `processed_dir('soccer')/imputed_no360.parquet` and the shift "
        "tables "
        "`soccer_02_shift_no360.parquet` / `soccer_02_shift_by_competition.parquet` (see the "
        "section appended "
        "by that script below, if it has been run).\n"
    )
    shift_p = reports_dir() / "soccer_02_shift_no360.parquet"
    if shift_p.exists():
        lines.append(
            shift_section(shift_p, reports_dir() / "soccer_02_shift_by_competition.parquet")
        )
    lines.append("\n## Caveats\n")
    lines.append(
        "- Labels are 360 freeze frames, not tracking: only players inside the broadcast visible "
        "area are "
        "present (mean 7.9 opponents visible; reliable share 73%, only 40% in the own third). "
        "Team-shape targets "
        "are therefore computed on the visible subset of the defence even on reliable frames, and "
        "the students "
        "learn that biased quantity. Ball-relative targets are less affected (the visible area is "
        "centred on the ball)."
    )
    lines.append(
        "- The keeper-distance label is filtered on `y_keeper_consistent == 1` (0.34% of visible "
        "keepers sit at the wrong end of the pitch, ~114 yd, and a student trained on them "
        "extrapolated own-third events to ~90 yd); the keeper student is restricted to "
        "possession-team events in the attacking half (x >= 60) and the two cone students to "
        "possession-team events, where the cone towards the opponent goal is meaningful."
    )
    lines.append(
        "- The `nearest_opp_dist_in_cone` and `opp_keeper_dist_to_goal_line` students are trained "
        "only where the "
        "quantity exists (a non-empty cone / a visible keeper, mostly final-third events); "
        "applied elsewhere they "
        "extrapolate."
    )
    lines.append(
        "- `E2a` uses the realised pass end / carry end / duration, which are consequences of the "
        "defensive "
        "state (a carry ends where a defender is met); it is the after-the-fact analytics "
        "setting, not a "
        "pre-instant state model."
    )
    lines.append(
        "- Out-of-fold predictions of a student trained on the cap are what feed the shift tables "
        "and the later payoff stage; the final students (fitted on all matches with the median CV "
        "round count) are only applied to the non-360 matches."
    )
    lines.append(
        "- Training rows are thinned to the caps above; skills are therefore slightly below what "
        "the full "
        "1.3M rows would give (the fit summary shows most students still gain a little at the "
        "round cap)."
    )
    lines.append(
        "- The two binaries are defined from the same 360 targets (`def_line`, "
        "`n_opp_ahead_of_ball`); their "
        "threshold / third definitions are choices of this stage, fixed once and stored in the "
        "bundle."
    )
    lines.append(
        "- The MLP early-stopping split is a random 10% of the training rows (not match-grouped); "
        "it only "
        "chooses the epoch count, the test fold is untouched."
    )
    lines.append(
        "- `f_gender` is constant inside each cross-gender training set, so the transferred "
        "students never "
        "saw the target gender's value of that feature; `f_comp_type` (league / tournament) is "
        "constant (tournament) "
        "in the women's data."
    )
    return "\n".join(lines) + "\n"


def shift_section(shift_path: Path, by_comp_path: Path) -> str:
    """Markdown for the no360 distribution-shift tables written by ``apply_student.py``."""
    sh = pd.read_parquet(shift_path)
    lines = [
        "\n### Distribution of the imputed values: 2015/16 leagues vs the 360 domain "
        "(same event types: Pass, Carry, Shot)\n",
        "`no360 leagues 2015/16 imputed` = the four full 2015/16 seasons (shots + 25% of passes / "
        "carries); `360 oof "
        "imputed` / `360 truth` = out-of-fold imputation and 360 label on Pass / Carry / Shot "
        "rows of the 360 matches. `360 truth` exists only on labelled rows (reliable frames for "
        "the team-shape targets, visible keeper / non-empty cone for the keeper and cone "
        "distances), whereas the imputed rows cover the student's whole subset, so the shift "
        "between domains is read from the two imputed rows; imputed vs truth is a selection "
        "effect unless the labelled rows are the same.\n",
    ]
    fset = "E2" if "E2" in set(sh["feature_set"]) else sh["feature_set"].iloc[0]
    g = sh[(sh["feature_set"] == fset) & (sh["event_type"] == "all")]
    lines.append(
        md_table(g[["target", "source", "n", "mean", "sd", "p05", "p50", "p95"]], floatfmt="{:.3f}")
    )
    g2 = sh[(sh["feature_set"] == fset) & (sh["event_type"] == "Shot")]
    if len(g2):
        lines.append("\n**Shots only**\n")
        lines.append(
            md_table(g2[["target", "source", "n", "mean", "sd", "p50"]], floatfmt="{:.3f}")
        )
    if by_comp_path.exists():
        bc = pd.read_parquet(by_comp_path)
        cols = ["source", "competition", "season", "gender", "n_rows", "n_matches"] + [
            f"{t}_mean"
            for t in (
                "block_depth",
                "def_line",
                "n_opp_ahead_of_ball",
                "n_opp_within_5",
                "nearest_opp_dist",
                "n_opp_in_cone",
                "n_opp_in_lane",
                "deep_block",
                "counter_on",
            )
            if f"{t}_mean" in bc.columns
        ]
        lines.append(f"\n**Mean imputed value per competition ({fset})**\n")
        lines.append(md_table(bc[cols], floatfmt="{:.2f}"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", required=True, choices=["cv", "transfer", "forward", "mlp", "final", "report"]
    )
    ap.add_argument(
        "--targets",
        default=None,
        help="comma list of targets or groups (shape, ball, pass); default all",
    )
    ap.add_argument("--fsets", default=None, help="comma list of feature sets (default per stage)")
    ap.add_argument(
        "--force", action="store_true", help="refit configurations that are already cached"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="first parquet row group only, separate cache, no outputs",
    )
    ap.add_argument("--n-jobs", type=int, default=2)
    args = ap.parse_args()
    cfg = ImputationConfig(n_jobs=args.n_jobs, smoke=args.smoke)
    if args.smoke:
        cfg.train_cap, cfg.final_cap, cfg.mlp_train_cap, cfg.max_rounds, cfg.mlp_max_iter = (
            30_000,
            30_000,
            20_000,
            60,
            5,
        )
        cfg.n_boot = 100
    targets = resolve_targets(args.targets)
    fsets = [s.strip() for s in args.fsets.split(",")] if args.fsets else None
    t0 = time.time()
    data = load_data(cfg, with_seq=args.stage in ("mlp", "report"))
    print(
        f"loaded {len(data.df):,} rows, {data.df['match_id'].nunique()} matches in "
        f"{time.time() - t0:.1f} s; "
        f"deep_block threshold {data.deep_block_thr:.2f}",
        flush=True,
    )
    if args.stage == "cv":
        stage_cv(data, cfg, targets, fsets, args.force)
    elif args.stage in ("transfer", "forward"):
        stage_transfer(data, cfg, args.stage, targets, fsets, args.force)
    elif args.stage == "mlp":
        stage_mlp(data, cfg, [t for t in targets if args.targets] or None, args.force)
    elif args.stage == "final":
        stage_final(data, cfg, fsets, targets if args.targets else None)
    else:
        seq_cols = [
            c for c in data.df.columns if c.startswith("seq_type_") or c.startswith("seq_same_")
        ]
        tables = stage_report(data, cfg, data.df[seq_cols] if seq_cols else None)
        with pd.option_context(
            "display.width", 250, "display.max_columns", 40, "display.max_rows", 200
        ):
            if len(tables["ranking"]):
                print(
                    tables["ranking"][
                        [
                            "target",
                            "kind",
                            "n",
                            "base_type",
                            "loc",
                            "E0",
                            "E1",
                            "E2",
                            "E2a",
                            "verdict",
                        ]
                    ]
                )
    print(f"stage {args.stage} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
