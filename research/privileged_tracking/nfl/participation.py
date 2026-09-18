"""NFL stage 02: "who is on the field" as a structured latent variable.

The nflverse participation files carry tracking-derived labels for every play 2016-2022:
offensive / defensive personnel groupings, box count, pass rushers, coverage (2018+) and
the 22 participant ids. This driver quantifies how well each of them can be inferred from
play-by-play situation plus team and player tendencies computed from strictly earlier
games, and saves an offense-grouping imputer that later stages can apply to any pbp play.

Stages (``--stages a,b,c``):

* ``a`` offense personnel grouping (collapsed to the top classes covering ~95% of plays)
  with nested feature sets S0 (situation) < S1 (+ shotgun, no-huddle) < S2 (+ strictly-prior
  team / opponent tendencies) < S2 + play type (post-play upper bound); out-of-sample
  imputations for every play; model bundle ``processed/nfl/participation_offense_model.joblib``.
* ``b`` defense personnel, defenders in box, pass rushers, man/zone and coverage type given
  S2, S2 + true offense personnel (+ formation), S2 + imputed offense personnel; the
  no-personnel vs true vs imputed comparison is the marginalisation check (d).
* ``c`` player-level participation and slot decoding (``participation_players.py``).

Protocol: train 2016-2020, validation 2021 (all settings chosen there), test 2022.

Usage::

    cd /home/user/geo-model
    python -m research.privileged_tracking.nfl.participation [--stages a,b,c] [--quick]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import nfl_dir, processed_dir, reports_dir
from research.privileged_tracking.common.metrics import clustered_bootstrap_delta, mae, paired_bootstrap_delta, r2
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.nfl import participation_features as pf

PBP_COLS = [
    "old_game_id", "play_id", "game_id", "season", "season_type", "week", "posteam", "defteam",
    "posteam_type", "qtr", "down", "ydstogo", "goal_to_go", "yardline_100",
    "half_seconds_remaining", "score_differential", "wp", "shotgun", "no_huddle", "play_type",
    "qb_kneel", "qb_spike", "qb_dropback", "special_teams_play", "pass", "rush", "penalty",
]
PART_COLS = [
    "old_game_id", "play_id", "offense_formation", "offense_personnel", "defenders_in_box",
    "defense_personnel", "number_of_pass_rushers", "offense_players", "defense_players",
    "n_offense", "n_defense", "defense_man_zone_type", "defense_coverage_type",
]
FORMATIONS = ["SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY", "PISTOL", "JUMBO", "WILDCAT"]
BUNDLE_NAME = "participation_offense_model.joblib"
COVERAGE_FIRST_SEASON = 2018
# Target-free constants the league running mean of each numeric tendency is shrunk towards
# (``alpha_league`` plays): 0.5 for the shotgun and man-coverage shares, and the DL / DL+LB
# counts of the default nickel front of ``participation_features.defense_slots`` (4 DL, 2 LB,
# 5 DB) for pass rushers / box. They only matter where no earlier (season, week) exists, i.e.
# week 1 of the first season in the frame; nothing here is computed from the targets.
MEAN_PRIORS: dict[str, float] = {"shotgun": 0.5, "box": 6.0, "rushers": 4.0, "man_zone": 0.5}
TEST_LOSSES_A = "participation_test_losses_a.parquet"
TEST_LOSSES_B = "participation_test_losses_b.parquet"


@dataclass
class ParticipationConfig:
    """Driver configuration.

    Attributes:
        train_seasons: seasons used to fit every model.
        val_season: season on which rounds / num_leaves are chosen.
        test_season: reported only with settings chosen on validation.
        coverage: cumulative share of plays the kept personnel classes must cover.
        n_jobs: LightGBM threads.
        learning_rate / max_rounds / early_stopping / num_leaves_grid / min_child_samples:
            LightGBM settings; ``num_leaves`` is chosen on validation for the S2 offense model
            and reused everywhere else.
        oof_folds: game-grouped folds for out-of-fold imputations on the training seasons.
        n_boot: paired-bootstrap resamples.
        max_train_plays: optional subsample of training plays (speed); ``None`` = all.
        players_train_plays / players_test_plays: play subsample sizes for stage (c).
        stages: which stages to run.
    """

    train_seasons: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020)
    val_season: int = 2021
    test_season: int = 2022
    coverage: float = 0.95
    n_jobs: int = 2
    seed: int = 0
    learning_rate: float = 0.05
    max_rounds: int = 1500
    early_stopping: int = 50
    num_leaves_grid: tuple[int, ...] = (15, 31, 63)
    min_child_samples: int = 100
    oof_folds: int = 4
    n_boot: int = 1000
    max_train_plays: int | None = None
    players_train_plays: int = 15000
    players_test_plays: int = 15000
    stages: tuple[str, ...] = ("a", "b", "c")
    tendency: pf.TendencyConfig = field(default_factory=pf.TendencyConfig)


@dataclass
class ClassSpec:
    """Collapsed class vocabularies chosen on the training seasons."""

    off_classes: list[str]
    def_classes: list[str]
    cov_classes: list[str]

    @classmethod
    def from_plays(cls, plays: pd.DataFrame, train: np.ndarray, coverage: float) -> ClassSpec:
        tr = plays[train]
        return cls(
            off_classes=pf.top_classes(tr["offense_personnel"], coverage) + [pf.OTHER],
            def_classes=pf.top_classes(tr["defense_personnel"], coverage) + [pf.OTHER],
            cov_classes=pf.top_classes(tr["defense_coverage_type"], coverage) + [pf.OTHER],
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_plays(seasons: tuple[int, ...]) -> pd.DataFrame:
    """Regular-season scrimmage plays with a participation record, all ``seasons``.

    Excludes special teams, kneels, spikes and plays without an offense_personnel label.
    """
    d = nfl_dir() / "nflverse"
    frames = []
    for s in seasons:
        pbp = pd.read_parquet(d / f"play_by_play_{s}.parquet", columns=PBP_COLS)
        pbp["play_id"] = pbp["play_id"].astype(int)
        part = pd.read_parquet(d / f"pbp_participation_{s}.parquet", columns=PART_COLS)
        part["play_id"] = part["play_id"].astype(int)
        m = pbp.merge(part, on=["old_game_id", "play_id"], how="inner")
        keep = ((m["season_type"] == "REG") & m["offense_personnel"].notna()
                & (m["special_teams_play"].fillna(0) == 0) & (m["qb_kneel"].fillna(0) == 0)
                & (m["qb_spike"].fillna(0) == 0) & m["play_type"].isin(["pass", "run", "no_play"])
                & m["posteam"].notna() & m["defteam"].notna())
        frames.append(m[keep])
    plays = pd.concat(frames, ignore_index=True)
    plays = plays.sort_values(["season", "week", "game_id", "play_id"], kind="mergesort").reset_index(drop=True)
    return plays


def prepare_plays(plays: pd.DataFrame, spec: ClassSpec) -> pd.DataFrame:
    """Add targets and derived columns (pure; returns a copy)."""
    p = plays.reset_index(drop=True)
    new: dict[str, Any] = {}
    new["is_home"] = (p["posteam_type"] == "home").astype(float)
    new["is_pass"] = p["pass"].fillna(0).astype(float)
    new["is_rush"] = p["rush"].fillna(0).astype(float)
    new["dd_bucket"] = pf.down_distance_bucket(p["down"].to_numpy(), p["ydstogo"].to_numpy())
    off_grp = pf.collapse_classes(p["offense_personnel"], spec.off_classes)
    def_grp = pf.collapse_classes(p["defense_personnel"], spec.def_classes)
    cov_grp = pf.collapse_classes(p["defense_coverage_type"], spec.cov_classes)
    new["off_grp"], new["def_grp"], new["cov_grp"] = off_grp, def_grp, cov_grp
    mz = p["defense_man_zone_type"].astype(object)
    new["man_zone"] = np.where(mz == "MAN_COVERAGE", 1.0, np.where(mz == "ZONE_COVERAGE", 0.0, np.nan))
    new["box"] = pd.to_numeric(p["defenders_in_box"], errors="coerce")
    new["rushers"] = pd.to_numeric(p["number_of_pass_rushers"], errors="coerce")
    form = pd.Categorical(p["offense_formation"].astype(object), categories=FORMATIONS).codes.astype(float)
    new["formation_code"] = np.where(form < 0, np.nan, form)
    oc = pd.Categorical(off_grp, categories=spec.off_classes).codes.astype(float)
    dc = pd.Categorical(def_grp, categories=spec.def_classes).codes.astype(float)
    new["off_grp_code"] = np.where(oc < 0, np.nan, oc)
    new["def_grp_code"] = np.where(dc < 0, np.nan, dc)
    new["off_def_joint"] = pf.joint_label(off_grp, def_grp)
    slots = pf.slots_frame(p["offense_personnel"], "offense").astype(float)
    out = pd.concat([p.drop(columns=["qb_dropback"]), pd.DataFrame(new, index=p.index), slots,
                     p[["qb_dropback"]].astype(float)], axis=1)
    return out


def _share_cols(prefix: str, classes: list[str]) -> list[str]:
    return [f"{prefix}_{pf.slug(c)}" for c in classes]


def _hier_shares(p: pd.DataFrame, entity: list[str], class_col: str, classes: list[str],
                 tcfg: pf.TendencyConfig, prefix: str, league: np.ndarray, bucket_col: str | None
                 ) -> dict[str, np.ndarray]:
    """League -> previous season -> in-season -> bucket-conditional shrunk shares."""
    prev_c, prev_n = pf.previous_season_counts(p, entity, class_col, classes)
    prev = pf.shrink_shares(prev_c, prev_n, league, tcfg.alpha_prev)
    cur_c, cur_n = pf.prior_game_class_counts(p, entity, class_col, classes)
    cur = pf.shrink_shares(cur_c, cur_n, prev, tcfg.alpha_team)
    out: dict[str, np.ndarray] = {}
    for j, c in enumerate(classes):
        out[f"{prefix}prev_{pf.slug(c)}"] = prev[:, j]
        out[f"{prefix}_{pf.slug(c)}"] = cur[:, j]
    out[f"{prefix}_n"] = cur_n
    if bucket_col is not None:
        b_c, b_n = pf.prior_game_class_counts(p, entity + [bucket_col], class_col, classes)
        bucket = pf.shrink_shares(b_c, b_n, cur, tcfg.alpha_bucket)
        for j, c in enumerate(classes):
            out[f"{prefix}dd_{pf.slug(c)}"] = bucket[:, j]
        out[f"{prefix}dd_n"] = b_n
    return out


def _hier_mean(p: pd.DataFrame, entity: list[str], value_col: str, tcfg: pf.TendencyConfig,
               prefix: str, mask: np.ndarray | None = None, prior_value: float = 0.0) -> dict[str, np.ndarray]:
    """League -> previous season -> in-season shrunk means of a numeric target.

    The league running mean over strictly earlier ``(season, week)`` slots is shrunk towards
    ``prior_value`` (a target-free constant, see ``MEAN_PRIORS``) with ``alpha_league`` plays,
    so no row ever sees a statistic that includes its own or a later week.
    """
    q = p.copy()
    if mask is not None:
        q.loc[~mask, value_col] = np.nan
    v = pd.to_numeric(q[value_col], errors="coerce")
    # league running mean over strictly earlier (season, week)
    tmp = pd.DataFrame({"season": q["season"], "week": q["week"], "_v": v.fillna(0.0), "_c": v.notna().astype(float)})
    g = tmp.groupby(["season", "week"], sort=True)[["_v", "_c"]].sum()
    cum = (g.cumsum() - g).reset_index()
    lg = tmp[["season", "week"]].merge(cum, on=["season", "week"], how="left")
    league = pf.shrink_mean(lg["_v"].to_numpy(), lg["_c"].to_numpy(), np.full(len(q), float(prior_value)),
                            tcfg.alpha_league)
    ps, pc = pf.previous_season_value(q, entity, value_col)
    prev = pf.shrink_mean(ps, pc, league, tcfg.alpha_prev)
    cs, cc = pf.prior_game_value_sums(q, entity, value_col)
    cur = pf.shrink_mean(cs, cc, prev, tcfg.alpha_team)
    return {f"{prefix}prev": prev, f"{prefix}": cur, f"{prefix}_n": cc}


@dataclass
class FeatureBundle:
    """Feature frame plus the named feature lists for the nested sets."""

    F: pd.DataFrame
    s0: list[str]
    s1: list[str]
    s2_off: list[str]
    s2_def: list[str]
    play_type: list[str]
    joint_counts: np.ndarray  # [n, C_off * C_def] strictly-prior defteam joint counts


def build_features(p: pd.DataFrame, spec: ClassSpec, tcfg: pf.TendencyConfig) -> FeatureBundle:
    """All play-level features (situation, pre-snap observables, strictly-prior tendencies).

    Args:
        p: output of :func:`prepare_plays` (rows in chronological order).

    Returns:
        A :class:`FeatureBundle` whose frame ``F`` ``[n, d]`` is aligned to ``p``.
    """
    fs = pf.FeatureSets()
    cols: dict[str, np.ndarray] = {}
    for c in fs.s0 + fs.s1_extra + fs.play_type_extra:
        cols[c] = pd.to_numeric(p[c], errors="coerce").to_numpy(dtype=float)
    oc, dc, cc = spec.off_classes, spec.def_classes, spec.cov_classes
    # --- offense grouping tendencies
    league_off = pf.league_prior_shares(p, "off_grp", oc, tcfg.alpha_league)
    for j, c in enumerate(oc):
        cols[f"t_lg_{pf.slug(c)}"] = league_off[:, j]
    cols.update(_hier_shares(p, ["posteam"], "off_grp", oc, tcfg, "t_off", league_off, "dd_bucket"))
    cols.update(_hier_shares(p, ["defteam"], "off_grp", oc, tcfg, "t_deffaced", league_off, None))
    cols.update(_hier_mean(p, ["posteam"], "shotgun", tcfg, "t_offsg", prior_value=MEAN_PRIORS["shotgun"]))
    s2_off_cols = [k for k in cols if k.startswith("t_")]
    # --- defense tendencies (defteam) and what the offense usually faces
    league_def = pf.league_prior_shares(p, "def_grp", dc, tcfg.alpha_league)
    cols.update(_hier_shares(p, ["defteam"], "def_grp", dc, tcfg, "t_dgrp", league_def, "dd_bucket"))
    cols.update(_hier_shares(p, ["posteam"], "def_grp", dc, tcfg, "t_offfaced", league_def, None))
    cols.update(_hier_mean(p, ["defteam"], "box", tcfg, "t_dbox", prior_value=MEAN_PRIORS["box"]))
    cols.update(_hier_mean(p, ["defteam"], "rushers", tcfg, "t_drush", mask=(p["is_pass"].to_numpy() == 1),
                           prior_value=MEAN_PRIORS["rushers"]))
    cols.update(_hier_mean(p, ["defteam"], "man_zone", tcfg, "t_dman", prior_value=MEAN_PRIORS["man_zone"]))
    league_cov = pf.league_prior_shares(p, "cov_grp", cc, tcfg.alpha_league)
    cols.update(_hier_shares(p, ["defteam"], "cov_grp", cc, tcfg, "t_dcov", league_cov, None))
    s2_def_cols = [k for k in cols if k.startswith("t_") and k not in s2_off_cols]
    joint_c, _ = pf.prior_game_class_counts(p, ["defteam"], "off_def_joint", pf.joint_classes(oc, dc))
    F = pd.DataFrame(cols, index=p.index)
    return FeatureBundle(F=F, s0=list(fs.s0), s1=list(fs.s1_extra), s2_off=s2_off_cols,
                         s2_def=s2_def_cols, play_type=list(fs.play_type_extra), joint_counts=joint_c)


def conditional_def_shares(fb: FeatureBundle, spec: ClassSpec, off_idx: np.ndarray,
                           tcfg: pf.TendencyConfig, prefix: str) -> pd.DataFrame:
    """Defteam's prior-game defense-grouping shares conditional on the offense grouping."""
    prior = fb.F[_share_cols("t_dgrp", spec.def_classes)].to_numpy()
    cond = pf.conditional_from_joint(fb.joint_counts, off_idx, len(spec.off_classes),
                                     len(spec.def_classes), prior, tcfg.alpha_cond)
    return pd.DataFrame(cond, columns=_share_cols(prefix, spec.def_classes), index=fb.F.index)


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------

def lgb_params(objective: str, cfg: ParticipationConfig, num_leaves: int, n_classes: int = 0) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": objective, "learning_rate": cfg.learning_rate, "num_leaves": num_leaves,
        "min_data_in_leaf": cfg.min_child_samples, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": cfg.n_jobs, "seed": cfg.seed,
        "verbose": -1, "max_bin": 63,
    }
    if objective == "multiclass":
        params["num_class"] = n_classes
        params["metric"] = "multi_logloss"
    elif objective == "binary":
        params["metric"] = "binary_logloss"
    else:
        params["metric"] = "l2"
    return params


def fit_lgb(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame | None, y_va: np.ndarray | None,
            params: dict[str, Any], cfg: ParticipationConfig, num_rounds: int | None = None,
            categorical: list[str] | None = None) -> tuple[lgb.Booster, int]:
    """Fit with early stopping on validation (or a fixed number of rounds)."""
    cat = [c for c in (categorical or []) if c in X_tr.columns]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat or "auto", free_raw_data=False)
    if num_rounds is not None:
        booster = lgb.train(params, dtr, num_boost_round=num_rounds)
        return booster, num_rounds
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat or "auto", free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=cfg.max_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    return booster, int(booster.best_iteration or cfg.max_rounds)


def mc_per_sample_logloss(y: np.ndarray, P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    P = np.clip(np.asarray(P, float), eps, 1.0)
    return -np.log(P[np.arange(len(y)), np.asarray(y, int)])


def mc_metrics(y: np.ndarray, P: np.ndarray) -> dict[str, float]:
    """Accuracy, mean log-loss and top-2 accuracy for ``[n, C]`` probabilities."""
    y = np.asarray(y, int)
    top = np.argsort(-P, axis=1)
    return {"accuracy": float((top[:, 0] == y).mean()),
            "log_loss": float(mc_per_sample_logloss(y, P).mean()),
            "top2": float(((top[:, 0] == y) | (top[:, 1] == y)).mean()), "n": int(len(y))}


def normalise_rows(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    return P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)


def _delta_row(name_a: str, name_b: str, la: np.ndarray, lb: np.ndarray, n_boot: int,
               groups: np.ndarray) -> dict[str, Any]:
    """Mean per-play loss difference with a play-level and a game-clustered bootstrap CI."""
    m, lo, hi = paired_bootstrap_delta(la, lb, n_boot=n_boot)
    _, lo_g, hi_g = clustered_bootstrap_delta(la, lb, groups, n_boot=n_boot)
    return {"from": name_a, "to": name_b, "delta_loss": m, "ci_low": lo, "ci_high": hi,
            "ci_low_game": lo_g, "ci_high_game": hi_g, "n": int(len(la)), "n_games": int(len(np.unique(groups)))}


def _test_loss_frame(p: pd.DataFrame, te: np.ndarray, losses: dict[str, np.ndarray],
                     preds: dict[str, np.ndarray] | None = None, target: str | None = None) -> pd.DataFrame:
    """Per-play test losses (one column per model) with the play keys, for re-analysis."""
    d = {"game_id": p.loc[te, "game_id"].to_numpy(), "play_id": p.loc[te, "play_id"].to_numpy()}
    if target is not None:
        d = {"target": np.full(int(te.sum()), target, dtype=object), **d}
    for name, l in losses.items():
        d[name] = np.asarray(l, float)
    for name, v in (preds or {}).items():
        d[f"pred: {name}"] = np.asarray(v, float)
    return pd.DataFrame(d)


def _game_losses(test_losses: pd.DataFrame, model_cols: list[str], extra_keys: list[str]) -> pd.DataFrame:
    """Per-game sum and count of per-play losses: enough to recompute any game-clustered CI."""
    rows = []
    for m in model_cols:
        g = test_losses.groupby(extra_keys + ["game_id"], sort=True)[m].agg(["count", "sum"]).reset_index()
        g = g[g["count"] > 0].rename(columns={"count": "n", "sum": "loss_sum"})
        g["model"] = m
        rows.append(g)
    return pd.concat(rows, ignore_index=True)[extra_keys + ["model", "game_id", "n", "loss_sum"]]


# ---------------------------------------------------------------------------
# Stage (a): offense grouping
# ---------------------------------------------------------------------------

def stage_a(p: pd.DataFrame, fb: FeatureBundle, spec: ClassSpec, cfg: ParticipationConfig,
            masks: dict[str, np.ndarray]) -> dict[str, Any]:
    t0 = time.time()
    oc = spec.off_classes
    y = p["off_grp_code"].to_numpy(int)
    F = fb.F
    sets = {"S0": fb.s0, "S1": fb.s0 + fb.s1, "S2": fb.s0 + fb.s1 + fb.s2_off,
            "S2+play_type": fb.s0 + fb.s1 + fb.s2_off + fb.play_type}
    tr, va, te = masks["train"], masks["val"], masks["test"]
    te_games = p.loc[te, "game_id"].to_numpy()
    tr_fit = _subsample_mask(tr, cfg)
    # --- choose num_leaves on validation with S2
    grid_rows = []
    best = None
    for nl in cfg.num_leaves_grid:
        params = lgb_params("multiclass", cfg, nl, len(oc))
        b, it = fit_lgb(F.loc[tr_fit, sets["S2"]], y[tr_fit], F.loc[va, sets["S2"]], y[va], params, cfg)
        m = mc_metrics(y[va], b.predict(F.loc[va, sets["S2"]], num_iteration=it))
        grid_rows.append({"num_leaves": nl, "best_iter": it, **m})
        if best is None or m["log_loss"] < best[1]:
            best = (nl, m["log_loss"], b, it)
        print(f"  [a] grid num_leaves={nl} iter={it} val_logloss={m['log_loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    num_leaves = best[0]
    # --- nested sets
    rows, losses, models = [], {}, {}
    for name, feats in sets.items():
        params = lgb_params("multiclass", cfg, num_leaves, len(oc))
        if name == "S2":
            b, it = best[2], best[3]
        else:
            b, it = fit_lgb(F.loc[tr_fit, feats], y[tr_fit], F.loc[va, feats], y[va], params, cfg)
        models[name] = (b, it, feats)
        for split, mk in (("val", va), ("test", te)):
            P = b.predict(F.loc[mk, feats], num_iteration=it)
            rows.append({"model": name, "split": split, **mc_metrics(y[mk], P)})
            losses[(name, split)] = mc_per_sample_logloss(y[mk], P)
        print(f"  [a] {name}: iter={it} test_logloss={rows[-1]['log_loss']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    # --- baselines
    freq = np.bincount(y[tr], minlength=len(oc)) / tr.sum()
    for split, mk in (("val", va), ("test", te)):
        P_major = np.tile(freq, (mk.sum(), 1))
        P_team = normalise_rows(F.loc[mk, _share_cols("t_off", oc)].to_numpy())
        P_dd = normalise_rows(F.loc[mk, _share_cols("t_offdd", oc)].to_numpy())
        P_prev = normalise_rows(F.loc[mk, _share_cols("t_offprev", oc)].to_numpy())
        for name, P in (("baseline: majority (train freq)", P_major), ("baseline: team prior-games", P_team),
                        ("baseline: team x down-distance prior-games", P_dd), ("baseline: team previous season", P_prev)):
            rows.append({"model": name, "split": split, **mc_metrics(y[mk], P)})
            losses[(name, split)] = mc_per_sample_logloss(y[mk], P)
    metrics = pd.DataFrame(rows)
    # --- paired bootstrap (test)
    pairs = [("baseline: majority (train freq)", "baseline: team prior-games"),
             ("baseline: team prior-games", "baseline: team x down-distance prior-games"),
             ("baseline: team x down-distance prior-games", "S2"),
             ("S0", "S1"), ("S1", "S2"), ("S2", "S2+play_type")]
    deltas = pd.DataFrame([_delta_row(a, b_, losses[(a, "test")], losses[(b_, "test")], cfg.n_boot, te_games)
                           for a, b_ in pairs])
    test_losses = _test_loss_frame(p, te, {k[0]: v for k, v in losses.items() if k[1] == "test"})
    model_cols = [k[0] for k in losses if k[1] == "test"]
    # --- feature importance (S2)
    b2, it2, f2 = models["S2"]
    imp = pd.DataFrame({"feature": f2, "gain": b2.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    # --- per-class test report for S2
    P_te = b2.predict(F.loc[te, f2], num_iteration=it2)
    per_class = _per_class_table(y[te], P_te, oc)
    # --- out-of-sample imputation for every play
    P_imp = np.full((len(p), len(oc)), np.nan)
    P_imp[va] = b2.predict(F.loc[va, f2], num_iteration=it2)
    P_imp[te] = P_te
    tr_idx = np.where(tr)[0]
    params = lgb_params("multiclass", cfg, num_leaves, len(oc))
    for k, (f_tr, f_te) in enumerate(group_kfold(p.loc[tr, "game_id"].to_numpy(), n_splits=cfg.oof_folds, seed=cfg.seed)):
        fit_rows = tr_idx[f_tr]
        if cfg.max_train_plays is not None and len(fit_rows) > cfg.max_train_plays:
            fit_rows = np.random.default_rng(cfg.seed + k).choice(fit_rows, cfg.max_train_plays, replace=False)
        b, _ = fit_lgb(F.loc[fit_rows, f2], y[fit_rows], None, None, params, cfg, num_rounds=it2)
        P_imp[tr_idx[f_te]] = b.predict(F.loc[tr_idx[f_te], f2])
        print(f"  [a] oof fold {k} done ({time.time()-t0:.0f}s)", flush=True)
    oof_metrics = mc_metrics(y[tr], P_imp[tr])
    # --- confusion on test
    conf = pd.crosstab(pd.Series(np.array(oc)[y[te]], name="true"), pd.Series(np.array(oc)[P_te.argmax(1)], name="pred"))
    bundle = OffenseGroupingImputer(booster=b2, best_iter=it2, features=f2, spec=spec, tendency=cfg.tendency,
                                    num_leaves=num_leaves)
    bundle.save(processed_dir("nfl") / BUNDLE_NAME)
    return {"metrics": metrics, "deltas": deltas, "grid": pd.DataFrame(grid_rows), "importance": imp,
            "per_class": per_class, "confusion": conf, "P_imp": P_imp, "oof_metrics": oof_metrics,
            "num_leaves": num_leaves, "class_freq": pd.DataFrame({"class": oc, "train_share": freq}),
            "test_losses": test_losses, "game_losses": _game_losses(test_losses, model_cols, []),
            "seconds": time.time() - t0}


def _per_class_table(y: np.ndarray, P: np.ndarray, classes: list[str]) -> pd.DataFrame:
    pred = P.argmax(1)
    rows = []
    for j, c in enumerate(classes):
        tp = int(((pred == j) & (y == j)).sum())
        fp = int(((pred == j) & (y != j)).sum())
        fn = int(((pred != j) & (y == j)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) else float("nan")
        rows.append({"class": c, "support": int((y == j).sum()), "share": float((y == j).mean()),
                     "precision": prec, "recall": rec, "f1": f1, "mean_prob": float(P[:, j].mean())})
    return pd.DataFrame(rows)


def _subsample_mask(mask: np.ndarray, cfg: ParticipationConfig) -> np.ndarray:
    if cfg.max_train_plays is None or mask.sum() <= cfg.max_train_plays:
        return mask
    idx = np.where(mask)[0]
    keep = np.random.default_rng(cfg.seed).choice(idx, cfg.max_train_plays, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out


@dataclass
class OffenseGroupingImputer:
    """Saved offense-grouping model plus everything needed to rebuild its features.

    ``predict_proba(plays)`` expects a pbp-like frame with the columns in ``PBP_COLS`` and
    ``offense_personnel`` / ``defense_personnel`` / ``defense_coverage_type`` /
    ``defenders_in_box`` / ``number_of_pass_rushers`` / ``defense_man_zone_type`` /
    ``offense_formation`` (NaN allowed) for *all* plays that should feed the strictly-prior
    tendency features, i.e. the labelled history plus the rows to impute. Rows whose labels
    are NaN simply do not contribute counts.
    """

    booster: lgb.Booster
    best_iter: int
    features: list[str]
    spec: ClassSpec
    tendency: pf.TendencyConfig
    num_leaves: int

    def save(self, path: Path) -> None:
        joblib.dump({"model_str": self.booster.model_to_string(), "best_iter": self.best_iter,
                     "features": self.features, "spec": asdict(self.spec), "tendency": asdict(self.tendency),
                     "num_leaves": self.num_leaves, "version": 1}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> OffenseGroupingImputer:
        d = joblib.load(path or processed_dir("nfl") / BUNDLE_NAME)
        return cls(booster=lgb.Booster(model_str=d["model_str"]), best_iter=d["best_iter"], features=d["features"],
                   spec=ClassSpec(**d["spec"]), tendency=pf.TendencyConfig(**d["tendency"]), num_leaves=d["num_leaves"])

    def predict_proba(self, plays: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities ``[n, C]`` (columns = ``spec.off_classes``) aligned to ``plays``.

        The tendency features are rebuilt from ``plays`` itself, so pass the longest labelled
        history available (the training run used 2016 onwards); with a shorter history the
        league-prior features differ and probabilities move by up to a few tenths on some rows.
        """
        p = prepare_plays(plays.reset_index(drop=True), self.spec)
        fb = build_features(p, self.spec, self.tendency)
        P = self.booster.predict(fb.F[self.features], num_iteration=self.best_iter)
        return pd.DataFrame(P, columns=self.spec.off_classes, index=plays.index)


LABEL_COLUMNS = ["offense_personnel", "defense_personnel", "defense_coverage_type", "defense_man_zone_type",
                 "defenders_in_box", "number_of_pass_rushers", "offense_formation"]


def label_free_check(plays: pd.DataFrame, imputer: OffenseGroupingImputer, test_mask: np.ndarray,
                     y_test: np.ndarray) -> dict[str, float]:
    """Impute the test season with *all* of its labels blanked (no in-season tendencies).

    This is the deployment scenario in which participation labels stop being published
    in-season: tendencies then come only from the previous seasons and the league prior.
    """
    q = plays.reset_index(drop=True).copy()
    q.loc[test_mask, LABEL_COLUMNS] = np.nan
    P = imputer.predict_proba(q).to_numpy()[test_mask]
    return mc_metrics(y_test, P)


# ---------------------------------------------------------------------------
# Stage (b) + (d): defense targets given no / true / imputed offense personnel
# ---------------------------------------------------------------------------

def offense_info_frames(p: pd.DataFrame, fb: FeatureBundle, spec: ClassSpec, P_imp: np.ndarray,
                        tcfg: pf.TendencyConfig) -> dict[str, pd.DataFrame]:
    """Extra feature blocks: true personnel, true personnel + formation, imputed personnel."""
    oc = spec.off_classes
    true_idx = p["off_grp_code"].to_numpy(int)
    cond_true = conditional_def_shares(fb, spec, true_idx, tcfg, "t_dgrp_giv")
    true_pers = pd.DataFrame({"off_grp_code": p["off_grp_code"], "slot_RB": p["slot_RB"], "slot_TE": p["slot_TE"],
                              "slot_WR": p["slot_WR"], "slot_OL": p["slot_OL"]}, index=p.index)
    true_pers = pd.concat([true_pers, cond_true], axis=1)
    true_form = pd.concat([true_pers, p[["formation_code"]]], axis=1)
    imp_idx = np.where(np.isnan(P_imp).any(1), -1, np.nan_to_num(P_imp).argmax(1))
    cond_imp = conditional_def_shares(fb, spec, imp_idx, tcfg, "t_dgrp_giv")
    slot_by_class = pf.slots_frame(pd.Series(oc), "offense")
    exp_slots = np.nan_to_num(P_imp) @ slot_by_class[["slot_RB", "slot_TE", "slot_WR", "slot_OL"]].to_numpy(float)
    imp = pd.DataFrame(P_imp, columns=_share_cols("p_off", oc), index=p.index)
    imp["off_grp_code"] = np.where(imp_idx < 0, np.nan, imp_idx.astype(float))
    for j, c in enumerate(["slot_RB", "slot_TE", "slot_WR", "slot_OL"]):
        imp[c] = exp_slots[:, j]
    imp = pd.concat([imp, cond_imp], axis=1)
    return {"S2+true_pers": true_pers, "S2+true_pers+form": true_form, "S2+imp_pers": imp}


def stage_b(p: pd.DataFrame, fb: FeatureBundle, spec: ClassSpec, cfg: ParticipationConfig,
            masks: dict[str, np.ndarray], P_imp: np.ndarray, num_leaves: int) -> dict[str, Any]:
    t0 = time.time()
    F = fb.F
    base_feats = fb.s0 + fb.s1 + fb.s2_off + fb.s2_def
    extras = offense_info_frames(p, fb, spec, P_imp, cfg.tendency)
    # true and imputed blocks deliberately share column names -> one frame per variant
    frames = {"S2": F[base_feats]}
    for name, block in extras.items():
        frames[name] = pd.concat([F[base_feats], block], axis=1)
    is_pass = p["is_pass"].to_numpy() == 1
    season = p["season"].to_numpy()
    targets: dict[str, dict[str, Any]] = {
        "defense_personnel": {"kind": "multiclass", "y": p["def_grp_code"].to_numpy(float),
                              "valid": p["def_grp_code"].notna().to_numpy(), "classes": spec.def_classes,
                              "team_prior": _share_cols("t_dgrp", spec.def_classes),
                              "bucket_prior": _share_cols("t_dgrpdd", spec.def_classes)},
        "defenders_in_box": {"kind": "regression", "y": p["box"].to_numpy(float), "valid": p["box"].notna().to_numpy(),
                             "team_prior": "t_dbox"},
        "number_of_pass_rushers": {"kind": "regression", "y": p["rushers"].to_numpy(float),
                                   "valid": p["rushers"].notna().to_numpy() & is_pass, "team_prior": "t_drush"},
        "man_zone": {"kind": "binary", "y": p["man_zone"].to_numpy(float),
                     "valid": p["man_zone"].notna().to_numpy() & (season >= COVERAGE_FIRST_SEASON), "team_prior": "t_dman"},
        "coverage_type": {"kind": "multiclass", "y": pd.Categorical(p["cov_grp"], categories=spec.cov_classes).codes.astype(float),
                          "valid": p["cov_grp"].notna().to_numpy() & (season >= COVERAGE_FIRST_SEASON),
                          "classes": spec.cov_classes, "team_prior": _share_cols("t_dcov", spec.cov_classes), "bucket_prior": None},
    }
    rows, delta_rows, imp_rows, loss_frames = [], [], [], []
    P_def_imp = np.full((len(p), len(spec.def_classes)), np.nan)
    P_def_true = np.full((len(p), len(spec.def_classes)), np.nan)
    for tname, t in targets.items():
        valid = t["valid"] & (t["y"] == t["y"])
        tr = _subsample_mask(masks["train"] & valid, cfg)
        va, te = masks["val"] & valid, masks["test"] & valid
        y = t["y"]
        losses: dict[str, np.ndarray] = {}
        preds: dict[str, np.ndarray] = {}
        te_games = p.loc[te, "game_id"].to_numpy()
        kind = t["kind"]
        n_classes = len(t.get("classes", [])) if kind == "multiclass" else 0
        objective = {"multiclass": "multiclass", "binary": "binary", "regression": "regression"}[kind]
        for name, Xf in frames.items():
            params = lgb_params(objective, cfg, num_leaves, n_classes)
            b, it = fit_lgb(Xf.loc[tr], y[tr].astype(int if kind != "regression" else float), Xf.loc[va],
                            y[va].astype(int if kind != "regression" else float), params, cfg,
                            categorical=["off_grp_code", "formation_code"])
            for split, mk in (("val", va), ("test", te)):
                pred = b.predict(Xf.loc[mk], num_iteration=it)
                rows.append({"target": tname, "model": name, "split": split, **_eval(kind, y[mk], pred)})
                if split == "test":
                    losses[name] = _per_sample_loss(kind, y[mk], pred)
                    preds[name] = pred.argmax(1) if kind == "multiclass" else pred
                if tname == "defense_personnel" and split in ("val", "test"):
                    if name == "S2+imp_pers":
                        P_def_imp[mk] = pred
                    elif name == "S2+true_pers":
                        P_def_true[mk] = pred
            if tname == "defense_personnel" and name in ("S2", "S2+true_pers", "S2+imp_pers"):
                imp = pd.DataFrame({"feature": list(Xf.columns), "gain": b.feature_importance("gain")})
                imp["gain_share"] = imp["gain"] / imp["gain"].sum()
                imp["model"] = name
                imp_rows.append(imp.sort_values("gain", ascending=False).head(12))
            print(f"  [b] {tname} / {name}: iter={it} ({time.time()-t0:.0f}s)", flush=True)
        # baselines
        for split, mk in (("val", va), ("test", te)):
            if kind == "multiclass":
                freq = np.bincount(y[tr].astype(int), minlength=n_classes) / tr.sum()
                bl = {"baseline: majority (train freq)": np.tile(freq, (mk.sum(), 1)),
                      "baseline: team prior-games": normalise_rows(F.loc[mk, t["team_prior"]].to_numpy())}
                if t.get("bucket_prior"):
                    bl["baseline: team x down-distance prior-games"] = normalise_rows(F.loc[mk, t["bucket_prior"]].to_numpy())
            elif kind == "binary":
                bl = {"baseline: base rate (train)": np.full(mk.sum(), y[tr].mean()),
                      "baseline: team prior-games": F.loc[mk, t["team_prior"]].to_numpy()}
            else:
                bl = {"baseline: global mean (train)": np.full(mk.sum(), y[tr].mean()),
                      "baseline: team prior-games": F.loc[mk, t["team_prior"]].to_numpy()}
            for name, pred in bl.items():
                rows.append({"target": tname, "model": name, "split": split, **_eval(kind, y[mk], pred)})
                if split == "test":
                    losses[name] = _per_sample_loss(kind, y[mk], pred)
        best_bl = [k for k in losses if k.startswith("baseline: team prior")][0]
        for a, b_ in ((best_bl, "S2"), ("S2", "S2+imp_pers"), ("S2", "S2+true_pers"), ("S2+imp_pers", "S2+true_pers"),
                      ("S2+true_pers", "S2+true_pers+form")):
            d = _delta_row(a, b_, losses[a], losses[b_], cfg.n_boot, te_games)
            d["target"] = tname
            delta_rows.append(d)
        loss_frames.append(_test_loss_frame(p, te, losses, preds, target=tname))
    metrics = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    test_losses = pd.concat(loss_frames, ignore_index=True)
    model_cols = [c for c in test_losses.columns if c not in ("target", "game_id", "play_id") and not c.startswith("pred: ")]
    return {"metrics": metrics, "deltas": deltas, "importance": pd.concat(imp_rows, ignore_index=True),
            "P_def_imp": P_def_imp, "P_def_true": P_def_true, "test_losses": test_losses,
            "game_losses": _game_losses(test_losses, model_cols, ["target"]), "seconds": time.time() - t0}


def _eval(kind: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if kind == "multiclass":
        return mc_metrics(y.astype(int), pred)
    if kind == "binary":
        p = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        return {"accuracy": float(((p > 0.5) == (y == 1)).mean()), "log_loss": ll, "top2": float("nan"), "n": int(len(y))}
    return {"mae": mae(y, pred), "r2": r2(y, pred), "rmse": float(np.sqrt(np.mean((y - pred) ** 2))), "n": int(len(y))}


def _per_sample_loss(kind: str, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if kind == "multiclass":
        return mc_per_sample_logloss(y.astype(int), pred)
    if kind == "binary":
        p = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return (np.asarray(y, float) - np.asarray(pred, float)) ** 2


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pick(df: pd.DataFrame, **where: Any) -> pd.Series | None:
    m = np.ones(len(df), bool)
    for k, v in where.items():
        m &= (df[k] == v).to_numpy()
    return df[m].iloc[0] if m.any() else None


def _headline(a: dict[str, Any] | None, b: dict[str, Any] | None, players: dict[str, Any] | None,
              cfg: ParticipationConfig) -> str:
    """Short generated summary of the main numbers (all from the tables below)."""
    L = ["## Headline results (test season " + str(cfg.test_season) + ")\n"]
    if a is not None and "metrics" in a:
        m = a["metrics"]
        s2, s0, s1 = (_pick(m, model=k, split="test") for k in ("S2", "S0", "S1"))
        maj = _pick(m, model="baseline: majority (train freq)", split="test")
        tdd = _pick(m, model="baseline: team x down-distance prior-games", split="test")
        pt = _pick(m, model="S2+play_type", split="test")
        lf = _pick(m, model="S2 (test-season labels blanked: no in-season tendencies)", split="test")
        if s2 is not None and maj is not None and tdd is not None:
            L.append(f"- **Offense personnel grouping** ({len(a['class_freq'])} classes, n={int(s2['n'])} plays): S2 "
                     f"accuracy {s2['accuracy']:.3f} / log-loss {s2['log_loss']:.3f} / top-2 {s2['top2']:.3f}, against "
                     f"majority {maj['accuracy']:.3f} / {maj['log_loss']:.3f} and the best tendency baseline "
                     f"(team x down-distance, prior games) {tdd['accuracy']:.3f} / {tdd['log_loss']:.3f}. Situation alone "
                     f"(S0) gives {s0['log_loss']:.3f}, shotgun/no-huddle (S1) {s1['log_loss']:.3f}; tendencies are the "
                     f"largest step ({s1['log_loss'] - s2['log_loss']:.3f} nats). Knowing pass/run post hoc adds only "
                     f"{s2['log_loss'] - pt['log_loss']:.4f} nats. The grouping is mostly *not* recoverable from pbp: the "
                     f"model is essentially a calibrated '11 personnel unless the team/situation says 12 or 21' prior.")
            if lf is not None:
                L.append(f"- With the test season's labels blanked entirely (tendencies from earlier seasons only, the "
                         f"'labels disappear in-season' scenario) S2 drops to accuracy {lf['accuracy']:.3f} / log-loss "
                         f"{lf['log_loss']:.3f}.")
    if b is not None and "metrics" in b:
        m, d = b["metrics"], b["deltas"]
        for tname, kind in (("defense_personnel", "cls"), ("defenders_in_box", "reg"), ("number_of_pass_rushers", "reg"),
                            ("man_zone", "cls"), ("coverage_type", "cls")):
            s2 = _pick(m, target=tname, model="S2", split="test")
            tp = _pick(m, target=tname, model="S2+true_pers", split="test")
            ip = _pick(m, target=tname, model="S2+imp_pers", split="test")
            bl = m[(m["target"] == tname) & (m["split"] == "test") & m["model"].str.startswith("baseline: team prior")]
            d_true = _pick(d, target=tname, **{"from": "S2", "to": "S2+true_pers"})
            d_imp = _pick(d, target=tname, **{"from": "S2", "to": "S2+imp_pers"})
            if s2 is None or tp is None or ip is None or len(bl) == 0:
                continue
            bl = bl.iloc[0]
            if kind == "cls":
                txt = (f"accuracy / log-loss: team-prior baseline {bl['accuracy']:.3f} / {bl['log_loss']:.3f}; S2 "
                       f"{s2['accuracy']:.3f} / {s2['log_loss']:.3f}; + imputed offense personnel {ip['accuracy']:.3f} / "
                       f"{ip['log_loss']:.3f}; + true offense personnel {tp['accuracy']:.3f} / {tp['log_loss']:.3f}")
            else:
                txt = (f"MAE / R2: team-prior baseline {bl['mae']:.3f} / {bl['r2']:.3f}; S2 {s2['mae']:.3f} / {s2['r2']:.3f}; "
                       f"+ imputed {ip['mae']:.3f} / {ip['r2']:.3f}; + true {tp['mae']:.3f} / {tp['r2']:.3f}")
            L.append(f"- **{tname}** (n={int(s2['n'])}, {int(d_true['n_games'])} games): {txt}. Paired-bootstrap gain "
                     f"over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true {d_true['delta_loss']:.4f} "
                     f"[{d_true['ci_low']:.4f}, {d_true['ci_high']:.4f}; {d_true['ci_low_game']:.4f}, "
                     f"{d_true['ci_high_game']:.4f}], imputed {d_imp['delta_loss']:.4f} [{d_imp['ci_low']:.4f}, "
                     f"{d_imp['ci_high']:.4f}; {d_imp['ci_low_game']:.4f}, {d_imp['ci_high_game']:.4f}].")
        L.append("- **Marginalisation (d)**: true offense personnel matters a lot for the defense's personnel and box "
                 "count and nothing for pass-rush count, man/zone or coverage type; the imputed grouping recovers only a "
                 "small fraction of that gain (judge the small `S2 -> S2+imp_pers` deltas by the game-clustered CI), so a "
                 "downstream team-level model should not expect the participation step to add much unless the grouping "
                 "is observed.")
    if players is not None and "tables" in players:
        s = players["tables"]["summary"]
        rows = []
        for side in ("offense", "defense"):
            mt = _pick(s, decoder="model | true grouping", side=side)
            mi = _pick(s, decoder="model | imputed grouping", side=side)
            bu = _pick(s, decoder="baseline: prior usage rank | true grouping", side=side)
            if mt is not None and mi is not None and bu is not None:
                rows.append(f"{side}: per-slot accuracy {mt['per_slot_acc']:.3f} (exact lineup {mt['exact_lineup_acc']:.3f}) "
                            f"with the true grouping, {mi['per_slot_acc']:.3f} ({mi['exact_lineup_acc']:.3f}) with the "
                            f"imputed grouping, prior-usage-rank baseline {bu['per_slot_acc']:.3f} ({bu['exact_lineup_acc']:.3f})")
        from research.privileged_tracking.nfl import participation_players as pp
        L.append("- **Player-level participation (c)**: " + "; ".join(rows) + ". Misses concentrate in DL rotation, "
                 "LB, nickel/dime DBs, WR3+, RB2+ and TE2+; RB1 / TE1 / WR1-2 / CB1-2 are almost never missed but are "
                 "not deterministic either (on-field rates well below 1 for RB1 and TE1). The candidate set is the "
                 "48-man game-day active list (weekly roster status `ACT`; declared inactives carry `INA`), so the "
                 "residual QB / OL errors are not unknown inactives. " + pp.qb_check_sentence(players["tables"]["qb_check"]))
    return "\n".join(L) + "\n"


def _fmt(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = "{:.4f}") -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return md_table(d, floatfmt=floatfmt)


def write_report(res: dict[str, Any], spec: ClassSpec, cfg: ParticipationConfig, counts: pd.DataFrame,
                 players: dict[str, Any] | None) -> Path:
    out = reports_dir()
    a, b = res.get("a"), res.get("b")
    L: list[str] = []
    L.append("# NFL 02: participation as a structured latent variable\n")
    L.append("Question: how well can the tracking-derived participation labels in the nflverse files "
             "(personnel groupings, box count, pass rushers, coverage, the 22 participant ids) be inferred from "
             "play-by-play situation plus team and player tendencies computed from strictly earlier games?\n")
    L.append("Protocol: regular-season scrimmage plays with a non-null `offense_personnel` (special teams, kneels and "
             "spikes excluded; `no_play` penalty snaps kept because personnel is pre-snap). Train seasons "
             f"{cfg.train_seasons[0]}-{cfg.train_seasons[-1]}, validation {cfg.val_season} (all settings: `num_leaves` "
             f"and early-stopping rounds), test {cfg.test_season} (reported once). LightGBM, `n_jobs={cfg.n_jobs}`, "
             f"lr={cfg.learning_rate}, `min_data_in_leaf={cfg.min_child_samples}`, feature/bagging fraction 0.8, "
             f"max {cfg.max_rounds} rounds, early stopping {cfg.early_stopping}. Paired bootstrap CIs "
             f"({cfg.n_boot} resamples) are reported twice: over plays (`ci_low` / `ci_high`) and game-clustered "
             "(`ci_low_game` / `ci_high_game`: whole games resampled with replacement). Plays within a game are "
             "correlated, so the clustered interval is the one to trust; the play-level one is kept for comparison. "
             "Seeds fixed (0).\n")
    if cfg.max_train_plays is not None:
        L.append(f"Training plays were subsampled to at most {cfg.max_train_plays} per fit for speed.\n")
    L.append(_headline(a, b, players, cfg))
    L.append("## Data\n")
    L.append(_fmt(counts, floatfmt="{:.0f}"))
    L.append("\nCollapsed classes (chosen on the training seasons to cover ~95% of plays, remainder = `other`):\n")
    L.append(f"- offense personnel: {spec.off_classes}\n- defense personnel: {spec.def_classes}\n"
             f"- coverage type (2018+): {spec.cov_classes}\n")
    L.append("## Feature sets\n")
    L.append("- **S0** situation: down, ydstogo, yardline_100, qtr, half_seconds_remaining, score_differential, "
             "wp (nflfastR pre-play), goal_to_go, season, week, home/away.\n"
             "- **S1** = S0 + pre-snap pbp observables: shotgun, no_huddle.\n"
             "- **S2** = S1 + strictly-prior tendencies: league running share (`t_lg_*`), posteam previous-season share "
             "(`t_offprev_*`), posteam in-season share over earlier games (`t_off_*`, shrunk towards the previous season "
             f"with alpha={cfg.tendency.alpha_team} plays), posteam share by down x distance bucket (`t_offdd_*`, shrunk "
             f"towards the in-season share, alpha={cfg.tendency.alpha_bucket}), defteam share of groupings faced "
             "(`t_deffaced_*`), posteam prior shotgun rate. For the defense targets S2 additionally holds the defteam's "
             "prior-game defense-grouping shares (overall, previous season, by down x distance), prior mean box / pass "
             "rushers / man share / coverage shares, and the posteam's faced-defense shares. Numeric tendencies are "
             "shrunk league -> previous season -> in-season; the league running mean uses strictly earlier "
             f"(season, week) slots and is shrunk towards target-free constants ({MEAN_PRIORS}) with "
             f"alpha={cfg.tendency.alpha_league} plays, so no feature contains its own or a later week.\n"
             "- **S2+play_type** adds post-play information (pass / rush flag, qb_dropback): an upper bound on what a "
             "pbp-observable could add, not a deployable feature.\n"
             "- **S2+true_pers** adds the true offense grouping, its RB/TE/WR/OL slot counts and the defteam's prior-game "
             "defense-grouping shares conditional on that offense grouping; **+form** adds the charted offense formation.\n"
             "- **S2+imp_pers** replaces the true grouping by the stage (a) S2 model's out-of-sample class probabilities "
             "(out-of-fold by game on the training seasons), their argmax, expected slot counts, and the conditional "
             "defense shares looked up with the imputed argmax.\n")
    if a is not None:
        L.append("## (a) Offense personnel grouping\n")
        L.append("Validation grid for `num_leaves` (S2 features):\n")
        L.append(_fmt(a["grid"]))
        L.append(f"\nChosen `num_leaves` = {a['num_leaves']}. Training-season class shares:\n")
        L.append(_fmt(a["class_freq"]))
        L.append("\nMetrics (multiclass log-loss in nats; top-2 = true class within the two most probable):\n")
        L.append(_fmt(a["metrics"], ["model", "split", "n", "accuracy", "log_loss", "top2"]))
        L.append("\nPaired bootstrap of the per-play log-loss difference on test (positive = `to` better than `from`):\n")
        L.append(_fmt(a["deltas"]))
        L.append("\nPer-class test report for S2:\n")
        L.append(_fmt(a["per_class"]))
        L.append("\nTest confusion for S2 (rows = true, columns = predicted):\n")
        L.append(md_table(a["confusion"].reset_index(), floatfmt="{:.0f}"))
        L.append(f"\nOut-of-fold S2 on the training seasons ({cfg.oof_folds} game-grouped folds): "
                 f"accuracy {a['oof_metrics']['accuracy']:.4f}, log-loss {a['oof_metrics']['log_loss']:.4f}, "
                 f"n={a['oof_metrics']['n']}.\n")
        L.append("\nTop S2 features by gain share:\n")
        L.append(_fmt(a["importance"].head(15), ["feature", "gain_share"]))
    if b is not None:
        L.append("\n## (b) Defense personnel, box, pass rush and coverage given the offense\n")
        L.append("Targets: `defense_personnel` (collapsed), `defenders_in_box` (all plays with a value), "
                 "`number_of_pass_rushers` (pass plays only), `man_zone` (MAN=1 vs ZONE=0, 2018+ pass plays), "
                 "`coverage_type` (collapsed, 2018+). Classification: accuracy / log-loss / top-2; counts: MAE / R2 / RMSE "
                 "(LightGBM L2 objective).\n")
        for tname in b["metrics"]["target"].unique():
            sub = b["metrics"][b["metrics"]["target"] == tname]
            L.append(f"\n### {tname}\n")
            cols = ["model", "split", "n"] + (["mae", "r2", "rmse"] if "mae" in sub.columns and sub["mae"].notna().any()
                                              else ["accuracy", "log_loss", "top2"])
            L.append(_fmt(sub, cols))
        L.append("\n### (d) Marginalisation check: paired bootstrap of per-play loss on test\n")
        L.append("Loss = log-loss for classification targets, squared error for counts. Positive `delta_loss` means the "
                 "`to` model is better. `S2 -> S2+imp_pers` is what the participation step buys when personnel must be "
                 "imputed; `S2+imp_pers -> S2+true_pers` is the cost of imputation.\n")
        L.append(_fmt(b["deltas"], ["target", "from", "to", "delta_loss", "ci_low", "ci_high", "ci_low_game",
                                    "ci_high_game", "n", "n_games"]))
        L.append("\nTop features by gain share for `defense_personnel` (S2, S2+true_pers, S2+imp_pers):\n")
        L.append(_fmt(b["importance"], ["model", "feature", "gain_share"]))
    if players is not None:
        L.append("\n## (c) Player-level participation\n")
        L.append(players["text"])
    L.append("\n## Caveats\n")
    L.append("- Tendency features are computed from *true* labels of earlier games. In a fully label-free deployment "
             "(in-season, college) those histories would themselves have to be imputed, so S2 is optimistic about the "
             "tendency channel; S0/S1 are the label-free floor.\n"
             "- `wp` is nflfastR's pre-play model output, itself a function of situation and pre-game spread; it is "
             "not tracking-derived.\n"
             "- Bootstrap CIs are given over plays and game-clustered; the clustered interval is the honest one and is "
             "the wider of the two for the small `S2 -> S2+imp_pers` deltas. Per-play test losses of every stage (a) / "
             f"(b) model (and baselines) are saved as `processed/nfl/{TEST_LOSSES_A}` / `{TEST_LOSSES_B}` and per-game "
             "loss sums as `nfl_02a_offense_game_losses.parquet` / `nfl_02b_defense_game_losses.parquet`, so any "
             "clustered comparison can be recomputed without refitting.\n"
             "- The 2023 participation file has a different provenance (position-level personnel strings, different "
             "coverage vocabulary) and was not used.\n"
             "- Stage (c) uses a seeded subsample of plays for speed (sizes stated in that section). Its candidate set "
             "is the weekly roster with status `ACT`, which in these files is exactly the 48-man game-day active list "
             "(the declared inactives carry status `INA`, about 6 per team-week, and none of them takes a snap). Both "
             "statuses are fixed about 90 minutes before kickoff, so the candidate set is pre-play but not mid-week "
             "information; a mid-week forecast would have to model the inactive list itself. The residual QB / OL "
             "errors therefore come from dressed non-starters, in-game substitutions and depth-chart staleness, not "
             "from unknown inactives (see the QB check in section (c)).\n")
    path = out / "nfl_02_participation.md"
    path.write_text("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: ParticipationConfig) -> dict[str, Any]:
    t0 = time.time()
    seasons = tuple(sorted(set(cfg.train_seasons) | {cfg.val_season, cfg.test_season}))
    pdir = processed_dir("nfl")
    plays_path = pdir / "participation_plays.parquet"
    if plays_path.exists():
        plays = pd.read_parquet(plays_path)
    else:
        plays = load_plays(seasons)
        plays.to_parquet(plays_path, index=False)
    print(f"plays: {len(plays)} rows, {plays['game_id'].nunique()} games ({time.time()-t0:.0f}s)", flush=True)
    season = plays["season"].to_numpy()
    masks = {"train": np.isin(season, cfg.train_seasons), "val": season == cfg.val_season, "test": season == cfg.test_season}
    spec = ClassSpec.from_plays(plays, masks["train"], cfg.coverage)
    p = prepare_plays(plays, spec)
    fb = build_features(p, spec, cfg.tendency)
    print(f"features: {fb.F.shape} ({time.time()-t0:.0f}s)", flush=True)
    counts = (p.assign(split=np.select([masks["train"], masks["val"]], ["train", "val"], "test"))
              .groupby(["split", "season"]).agg(plays=("play_id", "size"), games=("game_id", "nunique"),
                                                 pass_plays=("is_pass", "sum"),
                                                 box_labelled=("box", "count"), rushers_labelled=("rushers", "count"),
                                                 coverage_labelled=("cov_grp", "count")).reset_index())
    res: dict[str, Any] = {}
    rdir = reports_dir()
    imp_path = pdir / "participation_imputed.parquet"
    if "a" in cfg.stages:
        res["a"] = stage_a(p, fb, spec, cfg, masks)
        lf = label_free_check(plays, OffenseGroupingImputer.load(), masks["test"],
                              p.loc[masks["test"], "off_grp_code"].to_numpy(int))
        res["a"]["metrics"] = pd.concat([res["a"]["metrics"], pd.DataFrame([{
            "model": "S2 (test-season labels blanked: no in-season tendencies)", "split": "test", **lf}])],
            ignore_index=True)
        for k in ("metrics", "deltas", "grid", "importance", "per_class", "class_freq", "game_losses"):
            res["a"][k].to_parquet(rdir / f"nfl_02a_offense_{k}.parquet", index=False)
        res["a"]["confusion"].reset_index().to_parquet(rdir / "nfl_02a_offense_confusion.parquet", index=False)
        res["a"]["test_losses"].to_parquet(pdir / TEST_LOSSES_A, index=False)
        imp = p[["game_id", "old_game_id", "play_id", "season", "week", "posteam", "defteam", "off_grp", "def_grp"]].copy()
        imp = pd.concat([imp, pd.DataFrame(res["a"]["P_imp"], columns=_share_cols("p_off", spec.off_classes),
                                           index=imp.index)], axis=1)
        imp.to_parquet(imp_path, index=False)
        print(f"stage a done in {res['a']['seconds']:.0f}s", flush=True)
    else:
        imp = pd.read_parquet(imp_path)
        res["a"] = {"P_imp": imp[_share_cols("p_off", spec.off_classes)].to_numpy(),
                    "num_leaves": OffenseGroupingImputer.load().num_leaves}
    if "b" in cfg.stages:
        res["b"] = stage_b(p, fb, spec, cfg, masks, res["a"]["P_imp"], res["a"]["num_leaves"])
        for k in ("metrics", "deltas", "importance", "game_losses"):
            res["b"][k].to_parquet(rdir / f"nfl_02b_defense_{k}.parquet", index=False)
        res["b"]["test_losses"].to_parquet(pdir / TEST_LOSSES_B, index=False)
        imp = pd.read_parquet(imp_path)
        imp = imp[[c for c in imp.columns if not c.startswith("p_def_")]]
        imp = pd.concat([imp, pd.DataFrame(res["b"]["P_def_imp"], columns=_share_cols("p_def_imp", spec.def_classes), index=imp.index),
                         pd.DataFrame(res["b"]["P_def_true"], columns=_share_cols("p_def_true", spec.def_classes), index=imp.index)], axis=1)
        imp.to_parquet(imp_path, index=False)
        print(f"stage b done in {res['b']['seconds']:.0f}s", flush=True)
    players = None
    if "c" in cfg.stages:
        from research.privileged_tracking.nfl import participation_players as pp
        imp = pd.read_parquet(imp_path)
        players = pp.run_players(p, imp, spec, cfg)
        for k, v in players["tables"].items():
            v.to_parquet(rdir / f"nfl_02c_players_{k}.parquet", index=False)
    # report: reload saved tables for stages not run this time
    if "a" not in cfg.stages or "metrics" not in res["a"]:
        res["a"] = _load_stage_tables("a", rdir, spec)
    if "b" not in cfg.stages:
        res["b"] = _load_stage_tables("b", rdir, spec)
    if players is None and (rdir / "nfl_02c_players_summary.parquet").exists():
        from research.privileged_tracking.nfl import participation_players as pp
        tables = {k: pd.read_parquet(rdir / f"nfl_02c_players_{k}.parquet") for k in
                  ("coverage", "fits", "per_subgroup", "summary", "errors", "conditional_on_grouping", "qb_check")}
        players = {"tables": tables, "text": pp._players_text(tables, cfg)}
    path = write_report(res, spec, cfg, counts, players)
    counts.to_parquet(rdir / "nfl_02_counts.parquet", index=False)
    print(f"report: {path} (total {time.time()-t0:.0f}s)", flush=True)
    return res


def _load_stage_tables(stage: str, rdir: Path, spec: ClassSpec) -> dict[str, Any] | None:
    try:
        if stage == "a":
            d = {k: pd.read_parquet(rdir / f"nfl_02a_offense_{k}.parquet") for k in
                 ("metrics", "deltas", "grid", "importance", "per_class", "class_freq")}
            d["confusion"] = pd.read_parquet(rdir / "nfl_02a_offense_confusion.parquet").set_index("true")
            imp = OffenseGroupingImputer.load()
            d["num_leaves"] = imp.num_leaves
            oof = pd.read_parquet(processed_dir("nfl") / "participation_imputed.parquet")
            tr = oof["season"] <= 2020
            P = oof.loc[tr, _share_cols("p_off", spec.off_classes)].to_numpy()
            y = pd.Categorical(oof.loc[tr, "off_grp"], categories=spec.off_classes).codes
            d["oof_metrics"] = mc_metrics(y, P)
            return d
        return {k: pd.read_parquet(rdir / f"nfl_02b_defense_{k}.parquet") for k in ("metrics", "deltas", "importance")}
    except FileNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="a,b,c")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--max-train-plays", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="smoke run: small grids, subsampled training")
    args = ap.parse_args()
    cfg = ParticipationConfig(stages=tuple(args.stages.split(",")), n_jobs=args.n_jobs,
                              max_train_plays=args.max_train_plays)
    if args.quick:
        cfg = ParticipationConfig(stages=cfg.stages, n_jobs=cfg.n_jobs, max_train_plays=20000, max_rounds=200,
                                  num_leaves_grid=(31,), oof_folds=2, n_boot=200, players_train_plays=2000,
                                  players_test_plays=2000)
    run(cfg)


if __name__ == "__main__":
    main()
