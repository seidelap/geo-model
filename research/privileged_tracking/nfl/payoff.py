"""NFL 04 payoff: does imputed tracking state improve event-only outcome prediction? (stage NFL 04).

Four sub-questions, one per ``--stages`` letter:

* ``a`` Within the 91 tracked 2017 games: pass plays -> completion probability, yards
  gained and EPA; run plays -> yards gained and success. Feature sets nest an
  nflfastR-style event-only set (``PBP``) with imputed tracking state (``PBP+IMP``:
  out-of-fold student predictions), the true tracking values (``PBP+ORACLE_PRESNAP`` = the
  12 pre-snap targets, ``PBP+ORACLE_WITHIN`` = the within-play targets measured at or after
  the throw / handoff, ``PBP+ORACLE`` = both) and the official NGS participation fields
  (``PBP+NGS``). 5-fold group k-fold by game with the
  same folds the NFL 03 students used, so a test-fold play's imputed features come from
  students that never saw that game.
* ``b`` Untracked seasons: the students are applied to 2018-2022 play-by-play, the outcome
  models are trained on 2018-2021 and tested once on 2022 (``PBP``, ``PBP+IMP``,
  ``PBP+NGS``, ``PBP+IMP+NGS``); plus the distribution shift of the imputed values.
* ``c`` Receiver-week aggregation of imputed separation / cushion vs the NGS weekly
  ``avg_separation`` / ``avg_cushion`` (the receiving-props angle), next to naive event-only
  proxies (mean air yards, target share, mean ``cp``) and, because the F1 student consumes the
  outcome of every target, next to the after-the-fact event aggregates (completion rate, YAC,
  yards, interceptions, QB hits, EPA) that form the fair after-the-fact baseline.
* ``d`` Participation payoff: imputed offense-personnel grouping probabilities (NFL 02) vs the
  true grouping as features of the 2022 pass-outcome models.

Information sets (the leakage contract of this stage):

* ``PBP`` for pass plays = the pre-snap situation + strictly-prior team tendencies of
  :mod:`imputation_features` (``SITUATION_COLS`` + ``TENDENCY_COLS``) + the at-release
  play-by-play fields ``air_yards``, ``pass_length``, ``pass_location``, ``qb_hit``,
  receiver position group and ``qb_scramble``; for run plays the situation + ``run_location``
  / ``run_gap`` / ``qb_scramble``. Outcome fields of the play being predicted
  (``complete_pass``, ``interception``, ``yards_after_catch``, ``yards_gained``, ``epa``,
  ``success``, ``cp``) are never features (:func:`leak_check`).
* The NFL 03 ``F1`` students take ``complete_pass`` / ``yards_gained`` / ``epa`` as inputs, so
  their imputations would leak the label of every task here. Within-play state is
  therefore imputed by new *at-release* students ``F1T`` (= ``F0P`` inputs + the at-release
  fields above; ``F0T`` = the same on the ``F0`` base without personnel), fitted here with
  the NFL 03 machinery on the same folds. Pre-snap state uses the NFL 03 ``F0P`` (and
  ``F0``) out-of-fold columns. One diagnostic row ``PBP+IMP_F1(leaky)`` shows what the F1
  imputations would do; it is not a payoff.

Run from the repo root::

    python -m research.privileged_tracking.nfl.payoff [--stages a,b,c,d] [--n-jobs 2] [--force-apply]

Every comparison quotes a game-clustered bootstrap CI and, for the reference / imputed sets
(:data:`REFIT_SETS`), the spread of the mean loss across three seed refits
(``PayoffConfig.refit_seeds``: the inner early-stopping holdout and the LightGBM bagging seed
change, folds / splits do not). The refit floor of a pair is the larger spread of its two members
(``refit_measured`` says whether both were refit; an external column such as ``nflfastR cp`` has
zero spread by definition); a delta whose magnitude is below the floor of a fully measured pair is
flagged as within refit noise even when its CI excludes zero.
The ``nflfastR cp`` column is NaN on throwaways, so every set is also scored on the cp-present rows
(``<set> [cp rows]``) and the cp comparison is made like with like.

``--stages none`` regenerates the report from the saved tables. Outputs: report tables
``reports/nfl_04*.parquet``, ``reports/nfl_04_payoff.md``; processed (not committed)
``processed_dir('nfl')/payoff_oof_tracked.parquet`` (per-play 2017 predictions and the F1T /
F0T out-of-fold imputations), ``payoff_pbp_<season>.parquet`` (imputed 2018-2022 seasons),
``payoff_test_losses_{b,d}.parquet`` (per-play 2022 losses), ``payoff_receiver_week.parquet``
and the student bundles ``models/students_F1T.joblib`` / ``students_F0T.joblib``.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from research.privileged_tracking.common.io import nfl_dir, processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (calibration_table, clustered_bootstrap_delta,
                                                          paired_bootstrap_delta, per_sample_log_loss)
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.nfl import apply_student as aps
from research.privileged_tracking.nfl import imputation as im
from research.privileged_tracking.nfl import imputation_features as imf
from research.privileged_tracking.nfl import participation_features as pf

# ---------------------------------------------------------------------------
# Feature-set contract
# ---------------------------------------------------------------------------

PRESNAP_TARGETS: list[str] = [t.name for t in imf.TARGETS if t.subset == "all"]
WITHIN_TARGETS: dict[str, list[str]] = {"pass": [t.name for t in imf.TARGETS if t.subset == "pass"],
                                        "run": [t.name for t in imf.TARGETS if t.subset == "run"]}

#: Post-snap play-by-play fields known at the release / handoff (no completion, YAC, yards, EPA).
AT_RELEASE_COLS: list[str] = ["play_type_code", "is_pass", "is_run", "qb_dropback", "qb_scramble", "pass_length",
                              "pass_location", "air_yards", "sack", "qb_hit", "receiver_group", "run_location", "run_gap"]
#: At-release students: the NFL 03 bases plus :data:`AT_RELEASE_COLS` (nested inside F1).
STUDENT_SETS: dict[str, list[str]] = {"F1T": imf.FEATURE_SETS["F0P"] + AT_RELEASE_COLS,
                                      "F0T": imf.FEATURE_SETS["F0"] + AT_RELEASE_COLS}
#: After-the-fact event fields of a targeted pass. Never features of an outcome model; stage c
#: aggregates them per receiver-week because the NFL 03 F1 student consumes the same fields, so
#: they form the fair after-the-fact baseline of the receiver-week regressions.
AFTER_FACT_COLS: list[str] = ["complete_pass", "yac_per_target", "yards_gained", "interception", "qb_hit", "epa"]
#: Outcome fields of the predicted play: never features of any outcome model.
FORBIDDEN_COLS: tuple[str, ...] = ("complete_pass", "interception", "yards_after_catch", "yards_gained", "epa",
                                   "success", "cp", "cpoe", "passing_yards", "rushing_yards", "wpa", "first_down")

PBP_BASE: list[str] = imf.SITUATION_COLS + imf.TENDENCY_COLS
PBP_COLS: dict[str, list[str]] = {
    "pass": PBP_BASE + ["pass_length", "pass_location", "air_yards", "qb_hit", "receiver_group", "qb_scramble"],
    "run": PBP_BASE + ["run_location", "run_gap", "qb_scramble"],
}
PERS_COLS: list[str] = list(imf.PERSONNEL_COLS)
#: Official participation fields (nflverse ``pbp_participation`` names, ``ngs_`` prefixed here).
NGS_COLS: dict[str, list[str]] = {
    "pass": ["ngs_defenders_in_box", "ngs_number_of_pass_rushers", "ngs_air_yards", "ngs_was_pressure", "ngs_time_to_throw"],
    "run": ["ngs_defenders_in_box"],
}
NGS_RENAME_TRACKED = {"ngs_defenders_in_box": "ngs_defenders_in_box", "ngs_number_of_pass_rushers": "ngs_number_of_pass_rushers",
                      "ngs_air_yards": "ngs_air_yards", "ngs_was_pressure": "ngs_was_pressure", "ngs_time_to_throw": "ngs_time_to_throw"}
NGS_RENAME_PBP = {"defenders_in_box": "ngs_defenders_in_box", "number_of_pass_rushers": "ngs_number_of_pass_rushers"}

SETS_A = ["PBP", "PBP+PERS", "PBP+IMP", "PBP+IMP_F0", "PBP+ORACLE_PRESNAP", "PBP+ORACLE_WITHIN", "PBP+ORACLE", "PBP+NGS",
          "PBP+IMP+NGS", "PBP+IMP_F1(leaky)"]
SETS_B = ["PBP", "PBP+PERS", "PBP+IMP", "PBP+IMP_F0", "PBP+NGS", "PBP+IMP+NGS"]
REFERENCE_SET = "PBP"
#: Sets refit with the extra seeds of :class:`PayoffConfig` to quantify refit-to-refit noise, per stage.
REFIT_SETS: dict[str, list[str]] = {"a": ["PBP", "PBP+IMP", "PBP+IMP_F0", "PBP+ORACLE_PRESNAP"], "b": ["PBP", "PBP+IMP", "PBP+IMP_F0"],
                                    "d": ["PBP", "PBP+IMP+IMP_PERS"]}
#: Reference columns that involve no fit of ours (``nflfastR cp`` is an external column, ``base_global`` a training
#: mean): their refit spread is zero by definition, so a pair against them is fully measured by the other member.
FIXED_MODELS: frozenset[str] = frozenset({"nflfastR cp", "base_global"})
#: Sets also scored on the cp-present rows (``nflfastR cp`` is NaN on throwaways), like with like.
CP_ROW_SETS: list[str] = ["PBP", "PBP+IMP", "PBP+IMP_F0", "PBP+ORACLE_PRESNAP", "PBP+ORACLE", "PBP+NGS", "PBP+IMP+NGS"]
CP_ROWS_SUFFIX = " [cp rows]"


@dataclass(frozen=True)
class TaskSpec:
    """One observable outcome to predict.

    Attributes:
        name: task id.
        subset: ``"pass"`` (non-sack, non-spike pass plays) or ``"run"``.
        target: nflfastR column of the outcome.
        kind: ``"binary"`` or ``"reg"``.
        description: one line for the report.
    """

    name: str
    subset: str
    target: str
    kind: str
    description: str


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("pass_completion", "pass", "complete_pass", "binary", "completion on non-sack, non-spike pass plays"),
    TaskSpec("pass_yards", "pass", "yards_gained", "reg", "yards gained on non-sack, non-spike pass plays"),
    TaskSpec("pass_epa", "pass", "epa", "reg", "EPA on non-sack, non-spike pass plays"),
    TaskSpec("run_yards", "run", "yards_gained", "reg", "yards gained on run plays"),
    TaskSpec("run_success", "run", "success", "binary", "nflfastR success (EPA > 0) on run plays"),
)
TASK_BY_NAME: dict[str, TaskSpec] = {t.name: t for t in TASKS}


def payoff_feature_columns(set_name: str, subset: str) -> list[str]:
    """Feature columns of one payoff feature set for a task subset.

    Args:
        set_name: one of :data:`SETS_A`.
        subset: ``"pass"`` or ``"run"``.

    Returns:
        column names of the wide frame (``imp_<target>__<fset>`` for imputations,
        ``y_<target>`` for the oracles, ``ngs_*`` for the official fields). ``PBP+ORACLE_PRESNAP``
        holds the true values of the pre-snap targets only, ``PBP+ORACLE_WITHIN`` those of the
        within-play targets (measured at or after the throw / handoff: quasi-outcomes),
        ``PBP+ORACLE`` both.
    """
    within = WITHIN_TARGETS[subset]
    imp = [f"imp_{t}__F0P" for t in PRESNAP_TARGETS] + [f"imp_{t}__F1T" for t in within]
    imp0 = [f"imp_{t}__F0" for t in PRESNAP_TARGETS] + [f"imp_{t}__F0T" for t in within]
    oracle_pre = [f"y_{t}" for t in PRESNAP_TARGETS]
    oracle_within = [f"y_{t}" for t in within]
    leaky = [f"imp_{t}__F0P" for t in PRESNAP_TARGETS] + [f"imp_{t}__F1" for t in within]
    ngs = NGS_COLS[subset]
    table = {"PBP": [], "PBP+PERS": PERS_COLS, "PBP+IMP": imp, "PBP+IMP_F0": imp0, "PBP+ORACLE_PRESNAP": oracle_pre,
             "PBP+ORACLE_WITHIN": oracle_within, "PBP+ORACLE": oracle_pre + oracle_within,
             "PBP+NGS": PERS_COLS + ngs, "PBP+IMP+NGS": PERS_COLS + ngs + imp, "PBP+IMP_F1(leaky)": leaky}
    return PBP_COLS[subset] + table[set_name]


def leak_check(columns: list[str], forbidden: tuple[str, ...] = FORBIDDEN_COLS) -> None:
    """Raise if any outcome field of the predicted play is among the feature columns."""
    bad = [c for c in columns if c in forbidden]
    if bad:
        raise ValueError(f"outcome fields used as features: {bad}")


def task_mask(play_type: np.ndarray, sack: np.ndarray, qb_spike: np.ndarray, subset: str,
              is_pass_play: np.ndarray | None = None) -> np.ndarray:
    """Rows of one task subset ``[n]``.

    ``pass`` = pass plays (``is_pass_play`` when given, else ``play_type == "pass"``) that are
    not sacks and not spikes; ``run`` = ``play_type == "run"`` that is not a pass play.
    Throwaways are not flagged by nflfastR and stay in the pass subset.
    """
    pt = np.asarray(play_type, dtype=object)
    s = np.nan_to_num(np.asarray(sack, dtype=float), nan=0.0)
    sp = np.nan_to_num(np.asarray(qb_spike, dtype=float), nan=0.0)
    is_pass = np.asarray(is_pass_play, dtype=bool) if is_pass_play is not None else (pt == "pass")
    if subset == "pass":
        return is_pass & (s == 0) & (sp == 0)
    if subset == "run":
        return (pt == "run") & ~is_pass
    raise ValueError(subset)


def feature_group(column: str) -> str:
    """Report group of a feature column (pbp / personnel / imputed pre-snap / imputed within-play / ngs official /
    oracle pre-snap / oracle within-play / imputed personnel / true personnel)."""
    if column.startswith("imp_"):
        t = column[4:].split("__")[0]
        return "imputed pre-snap" if t in PRESNAP_TARGETS else "imputed within-play"
    if column.startswith("y_"):
        return "oracle pre-snap" if column[2:] in PRESNAP_TARGETS else "oracle within-play"
    if column.startswith("ngs_"):
        return "ngs official"
    if column.startswith("p_off_"):
        return "imputed personnel"
    if column.startswith("true_off_"):
        return "true personnel"
    if column in PERS_COLS:
        return "personnel"
    return "pbp"


# ---------------------------------------------------------------------------
# Config, fitting, scoring
# ---------------------------------------------------------------------------

@dataclass
class PayoffConfig:
    """Driver configuration.

    Attributes:
        n_jobs: LightGBM threads.
        seed: folds, inner holdouts, bootstraps.
        n_folds: game-grouped folds on the tracked games (must equal the NFL 03 setting so the folds coincide).
        n_boot: bootstrap resamples.
        small / big: LightGBM settings for the tracked (~6k rows) and season-level (~70k rows) fits.
        train_seasons / test_season: forward protocol for stages b-d.
        apply_feature_sets: student bundles applied to the untracked seasons.
        refit_seeds: extra seeds (inner early-stopping holdout + LightGBM bagging seed; folds and
            splits unchanged) under which the :data:`REFIT_SETS` models are refit to measure the
            refit-to-refit spread of their mean loss.
        stages: subset of ``a b c d``.
        force_apply: re-impute the seasons even when ``payoff_pbp_<season>.parquet`` exists.
        write: write outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    n_folds: int = 5
    n_boot: int = 1000
    small: dict[str, Any] = field(default_factory=lambda: {"learning_rate": 0.05, "max_rounds": 600, "early_stopping": 40})
    big: dict[str, Any] = field(default_factory=lambda: {"learning_rate": 0.1, "max_rounds": 400, "early_stopping": 30})
    train_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021)
    test_season: int = 2022
    apply_feature_sets: tuple[str, ...] = ("F0", "F0P", "F1", "F1T", "F0T")
    refit_seeds: tuple[int, ...] = (1, 2)
    stages: tuple[str, ...] = ("a", "b", "c", "d")
    force_apply: bool = False
    write: bool = True

    def lgb_cfg(self, big: bool) -> im.ImputationConfig:
        p = self.big if big else self.small
        return im.ImputationConfig(n_jobs=self.n_jobs, seed=self.seed, n_folds=self.n_folds, n_boot=self.n_boot, **p)


EXTRA_PBP_COLS = ["cp", "cpoe", "success", "qb_spike", "rusher_player_id"]


def fit_predict(Xtr: pd.DataFrame, ytr: np.ndarray, gtr: np.ndarray, Xte: pd.DataFrame, kind: str,
                icfg: im.ImputationConfig) -> tuple[np.ndarray, lgb.Booster]:
    """Early-stopped LightGBM (inner game holdout of the training rows) -> test predictions ``[n_test]``."""
    booster, bi = im.fit_with_inner_holdout(Xtr, ytr, gtr, kind, icfg)
    return booster.predict(Xte, num_iteration=bi), booster


def per_play_losses(kind: str, y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Per-play losses for paired bootstraps: ``log_loss`` (binary) or ``se`` / ``ae`` (regression)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if kind == "binary":
        return {"log_loss": per_sample_log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}
    return {"se": (y - p) ** 2, "ae": np.abs(y - p)}


def score_task(kind: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Metrics of one prediction vector (``imputation.score`` plus ``rmse`` / ``bss`` naming)."""
    return im.score(kind, y, p)


def metrics_table(kind: str, y: np.ndarray, preds: dict[str, np.ndarray], extra: dict[str, Any]) -> pd.DataFrame:
    """One row per model with ``n`` and the metrics of :func:`score_task`."""
    rows = []
    for name, p in preds.items():
        ok = ~np.isnan(p)
        rows.append({**extra, "model": name, **score_task(kind, y[ok], p[ok])})
    return pd.DataFrame(rows)


def metrics_table_on(kind: str, y: np.ndarray, preds: dict[str, np.ndarray], mask: np.ndarray, extra: dict[str, Any],
                     suffix: str, models: list[str] | None = None) -> pd.DataFrame:
    """:func:`metrics_table` on the ``mask`` rows only, model names suffixed (like-with-like rows).

    Used for the cp-present plays: ``nflfastR cp`` is NaN on throwaways, so its row must be
    compared with ``<set> [cp rows]`` and not with the all-plays rows.
    """
    mask = np.asarray(mask, dtype=bool)
    sel = {f"{k}{suffix}": np.asarray(v, dtype=float)[mask] for k, v in preds.items() if models is None or k in models}
    return metrics_table(kind, np.asarray(y, dtype=float)[mask], sel, extra)


def refit_mean_losses(task: str, kind: str, y: np.ndarray, preds_by_seed: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Long rows ``task, model, seed, loss, mean_loss`` of every seed refit (NaN predictions ignored)."""
    rows = []
    for seed, preds in preds_by_seed.items():
        for model, p in preds.items():
            for lname, l in per_play_losses(kind, y, p).items():
                rows.append({"task": task, "model": model, "seed": int(seed), "loss": lname, "mean_loss": float(np.nanmean(l))})
    return rows


def refit_spread_table(mean_losses: pd.DataFrame) -> pd.DataFrame:
    """Refit-to-refit spread of the mean loss per (task, model, loss).

    Args:
        mean_losses: long frame ``task, model, seed, loss, mean_loss`` (one row per seed refit).

    Returns:
        one row per (task, model, loss) with ``n_seeds``, ``min_loss``, ``max_loss`` and
        ``spread`` = max - min, in the units of ``loss`` (nats, yd^2 / EPA^2, yd / EPA).
    """
    if len(mean_losses) == 0:
        return pd.DataFrame(columns=["task", "model", "loss", "n_seeds", "min_loss", "max_loss", "spread"])
    g = mean_losses.groupby(["task", "model", "loss"], sort=False)["mean_loss"]
    out = g.agg(n_seeds="size", min_loss="min", max_loss="max").reset_index()
    out["spread"] = out["max_loss"] - out["min_loss"]
    return out


def attach_refit_spread(deltas: pd.DataFrame, spread: pd.DataFrame, fixed: frozenset[str] = FIXED_MODELS) -> pd.DataFrame:
    """Add the refit-noise floor of each pair to a deltas table.

    ``refit_spread`` is the larger refit spread of the two members of the pair (same task and
    loss); a member in ``fixed`` (an external column or a training mean, nothing of ours is
    refit) contributes zero. ``refit_measured`` says which members have a measured spread:
    ``both`` (the floor is that of the pair), ``from`` / ``to`` (only that member was refit: the
    floor is a lower bound of the pair's refit noise) or ``none`` (``refit_spread`` NaN).

    Args:
        deltas: rows with ``task, from, to, loss``.
        spread: :func:`refit_spread_table` output (``task, model, loss, spread``).
    """
    out = deltas.copy()
    table = {(r.task, r.model, r.loss): float(r.spread) for r in spread.itertuples()} if len(spread) else {}
    spreads, measured = [], []
    for task, a, b, loss in zip(out["task"], out["from"], out["to"], out["loss"]):
        vals, have = [], []
        for m in (a, b):
            if m in fixed:
                vals.append(0.0)
                have.append(True)
            elif (task, m, loss) in table:
                vals.append(table[(task, m, loss)])
                have.append(True)
            else:
                have.append(False)
        spreads.append(max(vals) if vals else np.nan)
        measured.append("both" if all(have) else ("from" if have[0] else ("to" if have[1] else "none")))
    out["refit_spread"] = spreads
    out["refit_measured"] = measured
    return out


def deltas_table(kind: str, y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, reference: str,
                 cfg: PayoffConfig, extra: dict[str, Any], pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Paired bootstrap deltas of per-play losses, ``from`` -> ``to``; positive = ``to`` better.

    Play-level (``ci_low`` / ``ci_high``) and game-clustered (``ci_low_game`` / ``ci_high_game``)
    intervals over the rows where both models predict.
    """
    pairs = pairs or [(reference, m) for m in preds if m != reference]
    rows = []
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        ok = ~(np.isnan(preds[a]) | np.isnan(preds[b]))
        la = per_play_losses(kind, y[ok], preds[a][ok])
        lb = per_play_losses(kind, y[ok], preds[b][ok])
        for loss in la:
            m, lo, hi = paired_bootstrap_delta(la[loss], lb[loss], n_boot=cfg.n_boot, seed=cfg.seed)
            _, lo_g, hi_g = clustered_bootstrap_delta(la[loss], lb[loss], groups[ok], n_boot=cfg.n_boot, seed=cfg.seed)
            rows.append({**extra, "from": a, "to": b, "loss": loss, "delta": m, "ci_low": lo, "ci_high": hi,
                         "ci_low_game": lo_g, "ci_high_game": hi_g, "n": int(ok.sum()),
                         "n_games": int(len(np.unique(groups[ok])))})
    return pd.DataFrame(rows)


def calibration_rows(y: np.ndarray, preds: dict[str, np.ndarray], extra: dict[str, Any], n_bins: int = 10) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        ok = ~np.isnan(p)
        for i, b in enumerate(calibration_table(y[ok], np.clip(p[ok], 0, 1), n_bins=n_bins)):
            rows.append({**extra, "model": name, "bin": i, **b})
    return pd.DataFrame(rows)


def group_gain_shares(boosters: list[lgb.Booster], columns: list[str]) -> dict[str, float]:
    """Mean (over boosters) share of LightGBM gain per :func:`feature_group`."""
    acc: dict[str, float] = {}
    for b in boosters:
        gain = np.asarray(b.feature_importance("gain"), dtype=float)
        tot = max(float(gain.sum()), 1e-12)
        for c, g in zip(columns, gain):
            acc[feature_group(c)] = acc.get(feature_group(c), 0.0) + g / tot / len(boosters)
    return acc


# ---------------------------------------------------------------------------
# At-release students (F1T / F0T) on the tracked plays
# ---------------------------------------------------------------------------

def _encoded(data: im.TrackedData, tr: np.ndarray, te: np.ndarray, y: np.ndarray, te_alpha: float
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    enc_off = imf.TeamEncoder(te_alpha).fit(data.posteam[tr], data.game[tr], y[tr])
    enc_def = imf.TeamEncoder(te_alpha).fit(data.defteam[tr], data.game[tr], y[tr])
    Xtr = data.X.iloc[tr].copy()
    Xtr["te_off"] = enc_off.transform_train(data.posteam[tr], data.game[tr])
    Xtr["te_def"] = enc_def.transform_train(data.defteam[tr], data.game[tr])
    Xte = data.X.iloc[te].copy()
    Xte["te_off"] = enc_off.transform(data.posteam[te])
    Xte["te_def"] = enc_def.transform(data.defteam[te])
    return Xtr, Xte


def student_oof(spec: imf.TargetSpec, data: im.TrackedData, feats: list[str], folds: list[tuple[np.ndarray, np.ndarray]],
                icfg: im.ImputationConfig) -> np.ndarray:
    """Out-of-fold student predictions ``[n_all]`` (NaN outside the target's subset)."""
    y = data.Y[spec.name].to_numpy(dtype=float)
    in_subset = data.masks[spec.subset].to_numpy()
    labelled = in_subset & ~np.isnan(y)
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        tr_l = tr[labelled[tr]]
        te_s = te[in_subset[te]]
        if len(tr_l) < 50 or len(te_s) == 0:
            continue
        Xtr, Xte = _encoded(data, tr_l, te_s, y, icfg.te_alpha)
        booster, bi = im.fit_with_inner_holdout(Xtr[feats], y[tr_l], data.game[tr_l], spec.kind, icfg)
        pred[te_s] = booster.predict(Xte[feats], num_iteration=bi)
    return pred


def student_final(spec: imf.TargetSpec, data: im.TrackedData, feats: list[str], icfg: im.ImputationConfig,
                  oof: np.ndarray) -> aps.Student:
    """Student refit on all tracked games (rounds from an inner holdout), as an :class:`apply_student.Student`."""
    y = data.Y[spec.name].to_numpy(dtype=float)
    idx = np.where(data.masks[spec.subset].to_numpy() & ~np.isnan(y))[0]
    enc_off = imf.TeamEncoder(icfg.te_alpha).fit(data.posteam[idx], data.game[idx], y[idx])
    enc_def = imf.TeamEncoder(icfg.te_alpha).fit(data.defteam[idx], data.game[idx], y[idx])
    X = data.X.iloc[idx].copy()
    X["te_off"] = enc_off.transform_train(data.posteam[idx], data.game[idx])
    X["te_def"] = enc_def.transform_train(data.defteam[idx], data.game[idx])
    _, bi = im.fit_with_inner_holdout(X[feats], y[idx], data.game[idx], spec.kind, icfg)
    booster = im.refit(X[feats], y[idx], spec.kind, icfg, bi)
    o = oof[idx]
    return aps.Student(target=spec.name, kind=spec.kind, subset=spec.subset, model_str=booster.model_to_string(),
                       best_iter=bi, te_off=enc_off.table(), te_def=enc_def.table(), te_global=enc_off.global_mean,
                       train_n=int(len(idx)), y_mean=float(y[idx].mean()), y_sd=float(y[idx].std()),
                       oof_mean=float(np.nanmean(o)), oof_sd=float(np.nanstd(o)))


def base_outcome_skill(spec: imf.TargetSpec, data: im.TrackedData, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Out-of-fold per-outcome-class mean of a target (``imputation.outcome_bucket`` classes)."""
    y = data.Y[spec.name].to_numpy(dtype=float)
    in_subset = data.masks[spec.subset].to_numpy()
    labelled = in_subset & ~np.isnan(y)
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        tr_l = tr[labelled[tr]]
        te_s = te[in_subset[te]]
        pred[te_s] = imf.bucket_mean_baseline(data.outcome[tr_l], y[tr_l], data.outcome[te_s], min_count=20)
    return pred


# ---------------------------------------------------------------------------
# Stage a: tracked games
# ---------------------------------------------------------------------------

def tracked_wide_frame(data: im.TrackedData, oof: pd.DataFrame) -> pd.DataFrame:
    """Design matrix + NFL 03 out-of-fold imputations (``imp_<t>__<fset>``), truth (``y_<t>``), official fields, keys."""
    keys = data.plays[["gameId", "playId"]].reset_index(drop=True)
    o = keys.merge(oof, on=["gameId", "playId"], how="left")
    assert len(o) == len(keys)
    p = data.plays.reset_index(drop=True)
    extra: dict[str, np.ndarray] = {}
    for t in imf.TARGETS:
        extra[f"y_{t.name}"] = o[f"y_{t.name}"].to_numpy(dtype=float)
        for fset in ("F0", "F0P", "F1"):
            extra[f"imp_{t.name}__{fset}"] = o[f"{t.name}__{fset}"].to_numpy(dtype=float)
    for c in NGS_RENAME_TRACKED:
        extra[c] = pd.to_numeric(p[c], errors="coerce").to_numpy(dtype=float)
    for c, src in (("cp", "pbp_cp"), ("success", "pbp_success"), ("qb_spike", "pbp_qb_spike")):
        extra[c] = pd.to_numeric(p[src], errors="coerce").to_numpy(dtype=float)
    for c in ("gameId", "playId", "week"):
        extra[c] = p[c].to_numpy()
    return pd.concat([data.X.reset_index(drop=True), pd.DataFrame(extra)], axis=1)


def run_task_cv(W: pd.DataFrame, task: TaskSpec, sets: list[str], rows: np.ndarray, groups: np.ndarray,
                folds: list[tuple[np.ndarray, np.ndarray]], icfg: im.ImputationConfig
                ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Out-of-fold predictions ``{set: [n_all]}`` of every feature set plus ``base_global`` and gain shares."""
    n = len(W)
    y = W[task.target].to_numpy(dtype=float)
    sel = np.zeros(n, dtype=bool)
    sel[rows] = True
    preds = {s: np.full(n, np.nan) for s in sets}
    preds["base_global"] = np.full(n, np.nan)
    boosters: dict[str, list[lgb.Booster]] = {s: [] for s in sets}
    for tr, te in folds:
        tr_t = tr[sel[tr]]
        te_t = te[sel[te]]
        if len(tr_t) < 50 or len(te_t) == 0:
            continue
        preds["base_global"][te_t] = float(y[tr_t].mean())
        for s in sets:
            cols = payoff_feature_columns(s, task.subset)
            leak_check(cols)
            p, b = fit_predict(W.iloc[tr_t][cols], y[tr_t], groups[tr_t], W.iloc[te_t][cols], task.kind, icfg)
            preds[s][te_t] = p
            boosters[s].append(b)
    gains = {s: group_gain_shares(boosters[s], payoff_feature_columns(s, task.subset)) for s in sets if boosters[s]}
    return preds, gains


def stage_a(cfg: PayoffConfig) -> dict[str, Any]:
    """Tracked-game payoff (k-fold by game): metrics, deltas, calibration, student skill, gain shares."""
    t0 = time.time()
    icfg = cfg.lgb_cfg(big=False)
    data = im.load_tracked(icfg)
    folds = list(group_kfold(data.game, n_splits=cfg.n_folds, seed=cfg.seed))
    proc = processed_dir("nfl")
    oof = pd.read_parquet(proc / "imputed_oof.parquet")
    W = tracked_wide_frame(data, oof)
    # at-release students, both bases, same folds
    student_rows = []
    for sset, feats in STUDENT_SETS.items():
        bundle = aps.StudentBundle(feature_set=sset, features=list(feats), te_alpha=icfg.te_alpha)
        for subset in ("pass", "run"):
            for t in WITHIN_TARGETS[subset]:
                spec = imf.TARGET_BY_NAME[t]
                pred = student_oof(spec, data, feats, folds, icfg)
                W[f"imp_{t}__{sset}"] = pred
                bundle.students[t] = student_final(spec, data, feats, icfg, pred)
                y = data.Y[t].to_numpy(dtype=float)
                ok = ~np.isnan(y) & ~np.isnan(pred)
                base_out = base_outcome_skill(spec, data, folds)
                row = {"target": t, "subset": subset, "kind": spec.kind, "student": sset, "n": int(ok.sum()),
                       "skill": im.skill_of(spec.kind, im.score(spec.kind, y[ok], pred[ok]))}
                for ref in ("F0P", "F1"):
                    pr = W[f"imp_{t}__{ref}"].to_numpy(dtype=float)
                    row[f"skill_{ref}"] = im.skill_of(spec.kind, im.score(spec.kind, y[ok], pr[ok]))
                row["skill_base_outcome"] = im.skill_of(spec.kind, im.score(spec.kind, y[ok], base_out[ok]))
                student_rows.append(row)
        if cfg.write:
            bundle.save()
    students = pd.DataFrame(student_rows)
    W = W.copy()   # defragment after the column inserts
    # outcome tasks
    play_type = data.play_type
    sack = pd.to_numeric(data.plays["pbp_sack"], errors="coerce").to_numpy(dtype=float)
    spike = W["qb_spike"].to_numpy(dtype=float)
    is_pass = data.masks["pass"].to_numpy()
    metrics, deltas, calib, gains_rows, refit_rows = [], [], [], [], []
    out = W[["gameId", "playId", "week"]].copy()
    out = pd.concat([out, W[[c for c in W.columns if c.endswith("__F1T") or c.endswith("__F0T")]]], axis=1)
    for task in TASKS:
        mask = task_mask(play_type, sack, spike, task.subset, is_pass_play=is_pass)
        y = W[task.target].to_numpy(dtype=float)
        rows = np.where(mask & ~np.isnan(y))[0]
        preds, gains = run_task_cv(W, task, SETS_A, rows, data.game, folds, icfg)
        extra = {"task": task.name, "subset": task.subset, "kind": task.kind}
        ev = np.zeros(len(W), dtype=bool)
        ev[rows] = True
        ev &= ~np.isnan(preds["base_global"])
        yv = y[ev]
        pv = {k: v[ev] for k, v in preds.items()}
        if task.name == "pass_completion":
            pv["nflfastR cp"] = W["cp"].to_numpy(dtype=float)[ev]
        metrics.append(metrics_table(task.kind, yv, pv, extra))
        if task.name == "pass_completion":   # like-with-like rows on the cp-present plays
            metrics.append(metrics_table_on(task.kind, yv, pv, ~np.isnan(pv["nflfastR cp"]), extra, CP_ROWS_SUFFIX, CP_ROW_SETS))
        pairs = [(REFERENCE_SET, m) for m in pv if m != REFERENCE_SET] + [("PBP+IMP", "PBP+ORACLE"), ("PBP+NGS", "PBP+IMP+NGS"),
                                                                           ("PBP+PERS", "PBP+IMP"), ("PBP+IMP_F0", "PBP+IMP"),
                                                                           ("PBP+ORACLE_PRESNAP", "PBP+ORACLE"),
                                                                           ("PBP+ORACLE_WITHIN", "PBP+ORACLE")]
        if task.name == "pass_completion":
            pairs += [("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+IMP"), ("nflfastR cp", "PBP+NGS")]
        deltas.append(deltas_table(task.kind, yv, pv, data.game[ev], REFERENCE_SET, cfg, extra, pairs))
        if task.kind == "binary":
            calib.append(calibration_rows(yv, pv, extra))
        for s, g in gains.items():
            for grp, share in g.items():
                gains_rows.append({**extra, "model": s, "group": grp, "gain_share": share})
        # refit-to-refit noise: reference / imputed sets under other inner-holdout + LightGBM seeds (folds unchanged)
        seed_preds = {cfg.seed: {s: pv[s] for s in REFIT_SETS["a"]}}
        for sd in cfg.refit_seeds:
            ps, _ = run_task_cv(W, task, REFIT_SETS["a"], rows, data.game, folds, replace(icfg, seed=sd))
            seed_preds[sd] = {s: ps[s][ev] for s in REFIT_SETS["a"]}
            for s in REFIT_SETS["a"]:
                out[f"pred_{task.name}__{s}@seed{sd}"] = ps[s]
        refit_rows += refit_mean_losses(task.name, task.kind, yv, seed_preds)
        for k, v in preds.items():
            out[f"pred_{task.name}__{k}"] = v
        out[f"y_{task.name}"] = np.where(mask, y, np.nan)
    out = out.copy()
    refit = refit_spread_table(pd.DataFrame(refit_rows))
    res = {"metrics": pd.concat(metrics, ignore_index=True), "deltas": attach_refit_spread(pd.concat(deltas, ignore_index=True), refit),
           "calibration": pd.concat(calib, ignore_index=True), "students": students,
           "gains": pd.DataFrame(gains_rows), "refit": refit, "seconds": time.time() - t0}
    if cfg.write:
        rd = reports_dir()
        res["metrics"].to_parquet(rd / "nfl_04a_metrics.parquet", index=False)
        res["deltas"].to_parquet(rd / "nfl_04a_deltas.parquet", index=False)
        res["calibration"].to_parquet(rd / "nfl_04a_calibration.parquet", index=False)
        res["students"].to_parquet(rd / "nfl_04a_students.parquet", index=False)
        res["gains"].to_parquet(rd / "nfl_04a_gain_shares.parquet", index=False)
        res["refit"].to_parquet(rd / "nfl_04a_refit_noise.parquet", index=False)
        out.to_parquet(proc / "payoff_oof_tracked.parquet", index=False)
    return res


# ---------------------------------------------------------------------------
# Stage b: untracked seasons
# ---------------------------------------------------------------------------

def season_frame(season: int, cfg: PayoffConfig, positions: pd.Series | None = None) -> pd.DataFrame:
    """Imputed nflfastR season (REG): keys, design columns, official fields, outcomes, ``imp_*`` columns.

    Cached under ``processed_dir('nfl')/payoff_pbp_<season>.parquet``.
    """
    path = processed_dir("nfl") / f"payoff_pbp_{season}.parquet"
    if path.exists() and not cfg.force_apply:
        return pd.read_parquet(path)
    cols = list(dict.fromkeys(aps.PBP_LOAD_COLS + EXTRA_PBP_COLS))
    pbp = pd.read_parquet(nfl_dir() / "nflverse" / f"play_by_play_{season}.parquet", columns=cols)
    pbp = pbp[pbp["season_type"] == "REG"].reset_index(drop=True)
    positions = positions if positions is not None else aps.load_player_positions()
    prepared = aps.prepare_pbp(pbp, aps.load_participation_season(season), positions)
    X = imf.build_design(prepared)
    masks = imf.subset_masks(prepared)
    out = prepared[["game_id", "old_game_id", "play_id", "season", "week", "posteam", "defteam", "play_type",
                    "receiver_player_id", "rusher_player_id", "sack", "qb_spike", "complete_pass", "yards_gained",
                    "epa", "success", "cp", "cpoe", "air_yards"]].copy()
    for c in X.columns:
        if c not in out.columns:
            out[c] = X[c].to_numpy()
    for src, dst in NGS_RENAME_PBP.items():
        out[dst] = pd.to_numeric(prepared[src], errors="coerce").to_numpy(dtype=float) if src in prepared.columns else np.nan
    for c in ("ngs_air_yards", "ngs_was_pressure", "ngs_time_to_throw"):
        out[c] = pd.to_numeric(prepared[c], errors="coerce").to_numpy(dtype=float) if c in prepared.columns else np.nan
    for c in ("offense_personnel", "offense_formation"):
        out[c] = prepared[c].astype(object).to_numpy() if c in prepared.columns else None
    preds = [aps.StudentBundle.load(fset).predict(X, prepared["posteam"].to_numpy(), prepared["defteam"].to_numpy(), masks)
             for fset in cfg.apply_feature_sets]
    out = pd.concat([out] + [pr.reset_index(drop=True) for pr in preds], axis=1)
    if cfg.write:
        out.to_parquet(path, index=False)
    return out


def shift_table(frames: pd.DataFrame, oof_tracked: pd.DataFrame) -> pd.DataFrame:
    """Imputed-value moments per season vs the 2017 out-of-fold imputations and the tracked truth.

    ``z_shift`` = (season mean - 2017 OOF mean) / 2017 OOF sd.
    """
    rows = []
    imp_cols = [c for c in frames.columns if c.startswith("imp_")]
    for c in imp_cols:
        t, fset = c[4:].split("__")
        ref = oof_tracked[c].to_numpy(dtype=float) if c in oof_tracked.columns else np.array([np.nan])
        ref = ref[~np.isnan(ref)]
        truth = oof_tracked[f"y_{t}"].to_numpy(dtype=float) if f"y_{t}" in oof_tracked.columns else np.array([np.nan])
        truth = truth[~np.isnan(truth)]
        base = {"target": t, "feature_set": fset, "oof2017_mean": float(ref.mean()) if len(ref) else np.nan,
                "oof2017_sd": float(ref.std()) if len(ref) else np.nan,
                "truth2017_mean": float(truth.mean()) if len(truth) else np.nan,
                "truth2017_sd": float(truth.std()) if len(truth) else np.nan}
        for season, g in frames.groupby("season"):
            v = g[c].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            rows.append({**base, "season": int(season), "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()),
                         "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)),
                         "z_shift": float((v.mean() - base["oof2017_mean"]) / base["oof2017_sd"]) if base["oof2017_sd"] else np.nan})
    return pd.DataFrame(rows)


def forward_task(W: pd.DataFrame, task: TaskSpec, sets: dict[str, list[str]], train: np.ndarray, test: np.ndarray,
                 groups: np.ndarray, icfg: im.ImputationConfig) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Train on ``train`` rows, predict ``test`` rows for every feature set (``{set: [n_test]}``)."""
    y = W[task.target].to_numpy(dtype=float)
    preds = {"base_global": np.full(len(test), float(y[train].mean()))}
    gains = {}
    for s, cols in sets.items():
        leak_check(cols)
        p, b = fit_predict(W.iloc[train][cols], y[train], groups[train], W.iloc[test][cols], task.kind, icfg)
        preds[s] = p
        gains[s] = group_gain_shares([b], cols)
    return preds, gains


def load_seasons(cfg: PayoffConfig) -> pd.DataFrame:
    """All train + test seasons stacked (imputing the missing ones); NGS fields coerced to float."""
    positions = aps.load_player_positions()
    seasons = tuple(sorted(set(cfg.train_seasons) | {cfg.test_season}))
    W = pd.concat([season_frame(s, cfg, positions) for s in seasons], ignore_index=True)
    for c in NGS_COLS["pass"]:
        W[c] = pd.to_numeric(W[c], errors="coerce").astype(float)
    return W


def stage_b(cfg: PayoffConfig) -> dict[str, Any]:
    """Untracked seasons: train 2018-2021, test 2022; shift of the imputed values."""
    t0 = time.time()
    icfg = cfg.lgb_cfg(big=True)
    W = load_seasons(cfg)
    proc = processed_dir("nfl")
    oof_tracked = pd.read_parquet(proc / "imputed_oof.parquet")
    oof_tracked = oof_tracked.rename(columns={c: f"imp_{c}" for c in oof_tracked.columns if "__" in c})
    mine = pd.read_parquet(proc / "payoff_oof_tracked.parquet")
    for c in mine.columns:
        if c.endswith("__F1T") or c.endswith("__F0T"):
            oof_tracked[c] = mine[c].to_numpy()
    shift = shift_table(W, oof_tracked)
    season = W["season"].to_numpy()
    groups = W["game_id"].to_numpy(dtype=object)
    metrics, deltas, calib, gains_rows, loss_frames, refit_rows = [], [], [], [], [], []
    for task in TASKS:
        mask = task_mask(W["play_type"].to_numpy(dtype=object), W["sack"].to_numpy(dtype=float),
                         W["qb_spike"].to_numpy(dtype=float), task.subset)
        y = W[task.target].to_numpy(dtype=float)
        ok = mask & ~np.isnan(y)
        train = np.where(ok & np.isin(season, cfg.train_seasons))[0]
        test = np.where(ok & (season == cfg.test_season))[0]
        sets = {s: payoff_feature_columns(s, task.subset) for s in SETS_B}
        preds, gains = forward_task(W, task, sets, train, test, groups, icfg)
        yv = y[test]
        extra = {"task": task.name, "subset": task.subset, "kind": task.kind, "n_train": int(len(train))}
        if task.name == "pass_completion":
            preds["nflfastR cp"] = W["cp"].to_numpy(dtype=float)[test]
        metrics.append(metrics_table(task.kind, yv, preds, extra))
        if task.name == "pass_completion":   # like-with-like rows on the cp-present plays
            metrics.append(metrics_table_on(task.kind, yv, preds, ~np.isnan(preds["nflfastR cp"]), extra, CP_ROWS_SUFFIX, CP_ROW_SETS))
        pairs = [(REFERENCE_SET, m) for m in preds if m != REFERENCE_SET] + [("PBP+NGS", "PBP+IMP+NGS"), ("PBP+PERS", "PBP+IMP"),
                                                                              ("PBP+IMP_F0", "PBP+IMP")]
        if task.name == "pass_completion":
            pairs += [("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+IMP"), ("nflfastR cp", "PBP+NGS")]
        deltas.append(deltas_table(task.kind, yv, preds, groups[test], REFERENCE_SET, cfg, extra, pairs))
        if task.kind == "binary":
            calib.append(calibration_rows(yv, preds, extra))
        for s, g in gains.items():
            for grp, share in g.items():
                gains_rows.append({**extra, "model": s, "group": grp, "gain_share": share})
        # refit-to-refit noise: reference / imputed sets under other inner-holdout + LightGBM seeds (split unchanged)
        seed_preds = {cfg.seed: {s: preds[s] for s in REFIT_SETS["b"]}}
        for sd in cfg.refit_seeds:
            ps, _ = forward_task(W, task, {s: sets[s] for s in REFIT_SETS["b"]}, train, test, groups, replace(icfg, seed=sd))
            seed_preds[sd] = {s: ps[s] for s in REFIT_SETS["b"]}
        refit_rows += refit_mean_losses(task.name, task.kind, yv, seed_preds)
        lf = W.iloc[test][["game_id", "play_id", "season", "week"]].copy()
        lf["task"] = task.name
        lf["y"] = yv
        all_preds = dict(preds)
        for sd in cfg.refit_seeds:
            all_preds.update({f"{s}@seed{sd}": v for s, v in seed_preds[sd].items()})
        for k, v in all_preds.items():
            lf[f"pred: {k}"] = v
            for lname, l in per_play_losses(task.kind, yv, v).items():
                lf[f"{lname}: {k}"] = l
        loss_frames.append(lf)
    refit = refit_spread_table(pd.DataFrame(refit_rows))
    res = {"metrics": pd.concat(metrics, ignore_index=True), "deltas": attach_refit_spread(pd.concat(deltas, ignore_index=True), refit),
           "calibration": pd.concat(calib, ignore_index=True), "gains": pd.DataFrame(gains_rows), "shift": shift,
           "refit": refit, "seconds": time.time() - t0}
    if cfg.write:
        rd = reports_dir()
        res["metrics"].to_parquet(rd / "nfl_04b_metrics.parquet", index=False)
        res["deltas"].to_parquet(rd / "nfl_04b_deltas.parquet", index=False)
        res["calibration"].to_parquet(rd / "nfl_04b_calibration.parquet", index=False)
        res["gains"].to_parquet(rd / "nfl_04b_gain_shares.parquet", index=False)
        res["shift"].to_parquet(rd / "nfl_04b_shift.parquet", index=False)
        res["refit"].to_parquet(rd / "nfl_04b_refit_noise.parquet", index=False)
        pd.concat(loss_frames, ignore_index=True).to_parquet(proc / "payoff_test_losses_b.parquet", index=False)
    return res


# ---------------------------------------------------------------------------
# Stage c: receiver-week aggregation vs NGS weekly receiving
# ---------------------------------------------------------------------------

RECEIVER_IMP_COLS = ["imp_separation_at_arrival__F1", "imp_separation_at_arrival__F1T", "imp_separation_at_arrival__F0T",
                     "imp_cb_cushion__F0P", "imp_cb_cushion__F0", "imp_target_depth__F1T"]
PROXY_COLS = ["air_yards", "cp"] + AFTER_FACT_COLS


def add_after_the_fact(plays: pd.DataFrame) -> pd.DataFrame:
    """Add ``yac_per_target`` (yards after catch, 0 on incompletions, NaN where completion is unknown)
    and coerce the :data:`AFTER_FACT_COLS` present to float. Pure; the input is not modified."""
    p = plays.copy()
    cp = pd.to_numeric(p["complete_pass"], errors="coerce")
    yac = pd.to_numeric(p["yards_after_catch"], errors="coerce") if "yards_after_catch" in p.columns else pd.Series(np.nan, index=p.index)
    p["yac_per_target"] = np.where(cp.to_numpy() == 1, yac.fillna(0.0).to_numpy(), 0.0)
    p.loc[cp.isna(), "yac_per_target"] = np.nan
    for c in AFTER_FACT_COLS:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce").astype(float)
    return p


def receiver_week_table(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, receiver) means of the imputed columns and proxies over targeted pass plays.

    Args:
        plays: pass-subset rows with ``season, week, posteam, receiver_player_id``, the
            :data:`RECEIVER_IMP_COLS` present and :data:`PROXY_COLS` (naive proxies plus the
            after-the-fact fields of :func:`add_after_the_fact`; absent columns are skipped).

    Returns:
        one row per receiver-week with ``n_targets``, ``target_share`` (targets / the team's
        targeted passes that week), ``mean_<col>`` for every imputed / proxy column.
    """
    p = plays[plays["receiver_player_id"].notna()].copy()
    keys = ["season", "week", "receiver_player_id"]
    agg: dict[str, Any] = {"n_targets": ("receiver_player_id", "size")}
    for c in RECEIVER_IMP_COLS + PROXY_COLS:
        if c in p.columns:
            agg[f"mean_{c}"] = (c, "mean")
    g = p.groupby(keys, sort=True).agg(**agg).reset_index()
    team = p.groupby(["season", "week", "posteam", "receiver_player_id"]).size().rename("n").reset_index()
    team_tot = team.groupby(["season", "week", "posteam"])["n"].sum().rename("team_targets").reset_index()
    team = team.merge(team_tot, on=["season", "week", "posteam"])
    team = team.sort_values("n", ascending=False).drop_duplicates(keys)
    g = g.merge(team[keys + ["team_targets"]], on=keys, how="left")
    g["target_share"] = g["n_targets"] / g["team_targets"]
    return g


def correlation_table(merged: pd.DataFrame, ngs_cols: tuple[str, ...] = ("avg_separation", "avg_cushion"),
                      min_targets: int = 1) -> pd.DataFrame:
    """Pearson / Spearman of every ``mean_*`` / ``target_share`` column with the NGS weekly fields.

    ``within_player_pearson`` demeans both sides by (season, player) first, i.e. the
    week-to-week signal that a props model would need.
    """
    rows = []
    cand = [c for c in merged.columns if c.startswith("mean_")] + ["target_share"]
    m = merged[merged["n_targets"] >= min_targets]
    for ngs in ngs_cols:
        for c in cand:
            d = m[[c, ngs, "season", "receiver_player_id"]].dropna()
            if len(d) < 10:
                continue
            x, yv = d[c].to_numpy(dtype=float), d[ngs].to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(yv) == 0:
                continue
            grp = d.groupby(["season", "receiver_player_id"])
            xd = x - grp[c].transform("mean").to_numpy(dtype=float)
            yd = yv - grp[ngs].transform("mean").to_numpy(dtype=float)
            ok = (grp[c].transform("size").to_numpy() >= 2)
            rows.append({"ngs_field": ngs, "proxy": c, "n": int(len(d)), "pearson": float(pearsonr(x, yv)[0]),
                         "spearman": float(spearmanr(x, yv)[0]),
                         "within_player_pearson": float(pearsonr(xd[ok], yd[ok])[0]) if ok.sum() > 10 and np.std(xd[ok]) > 0 else np.nan,
                         "n_within": int(ok.sum())})
    return pd.DataFrame(rows)


NAIVE_PROXIES = ["mean_air_yards", "mean_cp", "target_share", "n_targets"]
#: After-the-fact event aggregates (the fields the F1 student consumes): the fair after-the-fact baseline.
AFTER_PROXIES = [f"mean_{c}" for c in AFTER_FACT_COLS]
IMPUTED_PROXIES: dict[str, list[str]] = {
    "F0T": ["mean_imp_separation_at_arrival__F0T", "mean_imp_cb_cushion__F0"],
    "F1T": ["mean_imp_separation_at_arrival__F1T", "mean_imp_cb_cushion__F0P"],
    "F1": ["mean_imp_separation_at_arrival__F1", "mean_imp_cb_cushion__F0P"],
}
PROXY_MODELS: dict[str, list[str]] = {
    "naive (air yards, cp, share, n)": NAIVE_PROXIES,
    "after-the-fact aggregates only": AFTER_PROXIES,
    "naive + after-the-fact": NAIVE_PROXIES + AFTER_PROXIES,
    "imputed F1T only": IMPUTED_PROXIES["F1T"],
    "imputed F1 only": IMPUTED_PROXIES["F1"],
    "naive + imputed F0T": NAIVE_PROXIES + IMPUTED_PROXIES["F0T"],
    "naive + imputed F1T": NAIVE_PROXIES + IMPUTED_PROXIES["F1T"],
    "naive + imputed F1": NAIVE_PROXIES + IMPUTED_PROXIES["F1"],
    "naive + after-the-fact + imputed F0T": NAIVE_PROXIES + AFTER_PROXIES + IMPUTED_PROXIES["F0T"],
    "naive + after-the-fact + imputed F1T": NAIVE_PROXIES + AFTER_PROXIES + IMPUTED_PROXIES["F1T"],
    "naive + after-the-fact + imputed F1": NAIVE_PROXIES + AFTER_PROXIES + IMPUTED_PROXIES["F1"],
}
FAIR_BASELINE = "naive + after-the-fact"
#: Play-level regressors for :func:`imputation_vs_outcome_table`.
OUTCOME_REGRESSORS: dict[str, list[str]] = {
    "at-release fields (air yards, QB hit, cp)": ["air_yards", "qb_hit", "cp"],
    "outcome fields (+ completion, YAC, INT, completion x air yards)": ["air_yards", "qb_hit", "cp", "complete_pass", "yac_per_target",
                                                                       "interception", "complete_pass_x_air_yards"],
}


def _ols_r2(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], target: str) -> tuple[float, float]:
    Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(dtype=float) for c in cols])
    Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(dtype=float) for c in cols])
    ytr, yte = tr[target].to_numpy(dtype=float), te[target].to_numpy(dtype=float)
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    r2 = []
    for X, yy in ((Xtr, ytr), (Xte, yte)):
        ss = float(np.sum((yy - yy.mean()) ** 2))
        r2.append(float(1 - np.sum((yy - X @ beta) ** 2) / ss) if ss > 0 else np.nan)
    return r2[0], r2[1]


def imputation_vs_outcome_table(plays: pd.DataFrame, imp_cols: list[str], train_seasons: tuple[int, ...], test_season: int,
                                regressors: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Play-level OLS of each imputed column on event fields: how much of an imputation is a linear
    re-encoding of the at-release / outcome fields of the same play.

    Args:
        plays: pass-subset rows after :func:`add_after_the_fact` with ``season`` and the regressors.
        imp_cols: ``imp_*`` columns to explain.

    Returns:
        one row per (imputed column, regressor set): ``n_train``, ``n_test``, ``r2_train`` (in-sample
        on ``train_seasons``) and ``r2_test`` (``test_season``).
    """
    regressors = regressors or OUTCOME_REGRESSORS
    p = plays.copy()
    p["complete_pass_x_air_yards"] = pd.to_numeric(p["complete_pass"], errors="coerce") * pd.to_numeric(p["air_yards"], errors="coerce")
    rows = []
    for name, cols in regressors.items():
        for ic in imp_cols:
            if ic not in p.columns or any(c not in p.columns for c in cols):
                continue
            d = p[cols + [ic, "season"]].dropna()
            tr, te = d[d["season"].isin(train_seasons)], d[d["season"] == test_season]
            if len(tr) < 50 or len(te) < 10:
                continue
            r2_tr, r2_te = _ols_r2(tr, te, cols, ic)
            rows.append({"imputed_col": ic, "regressors": name, "n_train": int(len(tr)), "n_test": int(len(te)),
                         "r2_train": r2_tr, "r2_test": r2_te})
    return pd.DataFrame(rows)


def proxy_regression_table(merged: pd.DataFrame, train_seasons: tuple[int, ...], test_season: int,
                           ngs_cols: tuple[str, ...] = ("avg_separation", "avg_cushion"),
                           models: dict[str, list[str]] | None = None, baseline: str = FAIR_BASELINE,
                           n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """OLS of an NGS weekly field on receiver-week proxies, fitted on ``train_seasons``, scored on ``test_season``.

    One row per (NGS field, proxy model) with the test Pearson correlation, R2 and n. All models
    of one NGS field are fitted and scored on the same receiver-weeks (a row with a NaN in any
    model's proxies is dropped for every model), so the rows are paired: ``delta_r2_vs_baseline``
    = R2(model) - R2(``baseline``) with a paired bootstrap 95% CI (``ci_low`` / ``ci_high``:
    ``n_boot`` resamples of the test receiver-weeks, seed ``seed``) of the per-week squared errors
    (:func:`common.metrics.paired_bootstrap_delta`, divided by the test variance of the NGS field;
    positive = model better). NaN for the baseline row itself and when ``baseline`` is not among
    ``models``. The question is whether the imputed means add to the fair after-the-fact baseline.
    """
    models = models or PROXY_MODELS
    rows = []
    for ngs in ngs_cols:
        use_by_model = {name: [c for c in cols if c in merged.columns] for name, cols in models.items()}
        use_by_model = {k: v for k, v in use_by_model.items() if v}
        all_cols = list(dict.fromkeys(c for v in use_by_model.values() for c in v))
        d = merged[all_cols + [ngs, "season"]].dropna()
        tr = d[d["season"].isin(train_seasons)]
        te = d[d["season"] == test_season]
        if len(tr) < 20 or len(te) < 10 or not use_by_model:
            continue
        yte = te[ngs].to_numpy(dtype=float)
        ss = float(np.sum((yte - yte.mean()) ** 2))
        se: dict[str, np.ndarray] = {}
        local = []
        for name, use in use_by_model.items():
            Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy(dtype=float) for c in use])
            Xte = np.column_stack([np.ones(len(te))] + [te[c].to_numpy(dtype=float) for c in use])
            beta = np.linalg.lstsq(Xtr, tr[ngs].to_numpy(dtype=float), rcond=None)[0]
            yhat = Xte @ beta
            se[name] = (yte - yhat) ** 2
            local.append({"ngs_field": ngs, "proxy_model": name, "n_train": int(len(tr)), "n_test": int(len(te)),
                          "test_pearson": float(np.corrcoef(yhat, yte)[0, 1]) if np.std(yhat) > 0 else np.nan,
                          "test_r2": float(1 - se[name].sum() / ss) if ss > 0 else np.nan})
        var = ss / len(yte) if ss > 0 else np.nan
        for r in local:
            name = r["proxy_model"]
            if baseline in se and name != baseline and ss > 0:
                m, lo, hi = paired_bootstrap_delta(se[baseline], se[name], n_boot=n_boot, seed=seed)
                r["delta_r2_vs_baseline"], r["ci_low"], r["ci_high"] = m / var, lo / var, hi / var
            else:
                r["delta_r2_vs_baseline"], r["ci_low"], r["ci_high"] = np.nan, np.nan, np.nan
            r["baseline"] = baseline
        rows += local
    return pd.DataFrame(rows)


def stage_c(cfg: PayoffConfig) -> dict[str, Any]:
    """Receiver-week imputed separation / cushion vs NGS weekly receiving."""
    t0 = time.time()
    W = load_seasons(cfg)
    mask = task_mask(W["play_type"].to_numpy(dtype=object), W["sack"].to_numpy(dtype=float),
                     W["qb_spike"].to_numpy(dtype=float), "pass")
    plays = add_after_the_fact(W[mask])
    rw = receiver_week_table(plays)
    vs_outcome = imputation_vs_outcome_table(plays, [c for c in RECEIVER_IMP_COLS if "separation" in c or "target_depth" in c],
                                             cfg.train_seasons, cfg.test_season)
    ngs = pd.read_parquet(nfl_dir() / "nflverse" / "ngs_receiving.parquet")
    ngs = ngs[(ngs["week"] > 0) & (ngs["season_type"] == "REG") & ngs["season"].isin(sorted(set(cfg.train_seasons) | {cfg.test_season}))]
    ngs = ngs[["season", "week", "player_gsis_id", "avg_separation", "avg_cushion", "targets", "avg_intended_air_yards"]]
    ngs = ngs.rename(columns={"player_gsis_id": "receiver_player_id", "targets": "ngs_targets"})
    merged = rw.merge(ngs, on=["season", "week", "receiver_player_id"], how="inner")
    corr = correlation_table(merged)
    reg = proxy_regression_table(merged, cfg.train_seasons, cfg.test_season)
    coverage = pd.DataFrame([{"quantity": "receiver-weeks (event data, >=1 target)", "value": float(len(rw))},
                             {"quantity": "NGS weekly receiving rows 2018-2022 REG", "value": float(len(ngs))},
                             {"quantity": "joined receiver-weeks", "value": float(len(merged))},
                             {"quantity": "joined share of NGS rows", "value": float(len(merged) / max(len(ngs), 1))},
                             {"quantity": "mean |n_targets - ngs_targets| on joined rows", "value": float(np.mean(np.abs(merged["n_targets"] - merged["ngs_targets"])))}])
    res = {"correlations": corr, "regression": reg, "coverage": coverage, "vs_outcome": vs_outcome,
           "n_joined": int(len(merged)), "seconds": time.time() - t0}
    if cfg.write:
        rd = reports_dir()
        corr.to_parquet(rd / "nfl_04c_receiver_week_correlations.parquet", index=False)
        reg.to_parquet(rd / "nfl_04c_receiver_week_regression.parquet", index=False)
        coverage.to_parquet(rd / "nfl_04c_coverage.parquet", index=False)
        vs_outcome.to_parquet(rd / "nfl_04c_imputation_vs_outcome.parquet", index=False)
        merged.to_parquet(processed_dir("nfl") / "payoff_receiver_week.parquet", index=False)
    return res


# ---------------------------------------------------------------------------
# Stage d: participation payoff
# ---------------------------------------------------------------------------

def personnel_one_hot(off_grp: pd.Series, classes: list[str]) -> pd.DataFrame:
    """``true_off_<slug>`` indicator columns ``[n, len(classes)]`` (NaN rows -> all NaN)."""
    out = pd.DataFrame(index=off_grp.index)
    v = off_grp.astype(object)
    for c in classes:
        out[f"true_off_{pf.slug(c)}"] = (v == c).astype(float).where(v.notna())
    return out


def stage_d(cfg: PayoffConfig) -> dict[str, Any]:
    """Imputed vs true offense personnel grouping as features of the 2022 pass-outcome models."""
    t0 = time.time()
    icfg = cfg.lgb_cfg(big=True)
    W = load_seasons(cfg)
    proc = processed_dir("nfl")
    pi = pd.read_parquet(proc / "participation_imputed.parquet")
    p_cols = [c for c in pi.columns if c.startswith("p_off_")]
    classes = [c[len("p_off_"):] for c in p_cols]
    pi = pi[["game_id", "play_id", "off_grp"] + p_cols]
    pi["play_id"] = pi["play_id"].astype(int)
    keys = pd.DataFrame({"game_id": W["game_id"].astype(str), "play_id": W["play_id"].astype(int)})
    m = keys.merge(pi.assign(game_id=pi["game_id"].astype(str)), on=["game_id", "play_id"], how="left")
    assert len(m) == len(W)
    slug_to_class = {pf.slug(c): c for c in m["off_grp"].dropna().unique()}
    true_classes = [slug_to_class.get(s, s) for s in classes]
    oh = personnel_one_hot(m["off_grp"], true_classes)
    W = pd.concat([W, m[p_cols].astype(float).reset_index(drop=True), oh.reset_index(drop=True)], axis=1)
    true_cols = list(oh.columns)
    season = W["season"].to_numpy()
    groups = W["game_id"].to_numpy(dtype=object)
    metrics, deltas, gains_rows, loss_frames, refit_rows = [], [], [], [], []
    for task in (TASK_BY_NAME["pass_completion"], TASK_BY_NAME["pass_epa"]):
        base = PBP_COLS[task.subset]
        imp = payoff_feature_columns("PBP+IMP", task.subset)
        sets = {"PBP": base, "PBP+IMP_PERS": base + p_cols, "PBP+TRUE_PERS": base + true_cols,
                "PBP+TRUE_PERS+FORM": base + true_cols + ["formation"], "PBP+PERS": base + PERS_COLS,
                "PBP+IMP+IMP_PERS": imp + p_cols, "PBP+IMP+TRUE_PERS": imp + true_cols}
        mask = task_mask(W["play_type"].to_numpy(dtype=object), W["sack"].to_numpy(dtype=float),
                         W["qb_spike"].to_numpy(dtype=float), task.subset)
        y = W[task.target].to_numpy(dtype=float)
        ok = mask & ~np.isnan(y) & ~np.isnan(W[p_cols[0]].to_numpy(dtype=float))
        train = np.where(ok & np.isin(season, cfg.train_seasons))[0]
        test = np.where(ok & (season == cfg.test_season))[0]
        preds, gains = forward_task(W, task, sets, train, test, groups, icfg)
        yv = y[test]
        extra = {"task": task.name, "subset": task.subset, "kind": task.kind, "n_train": int(len(train))}
        metrics.append(metrics_table(task.kind, yv, preds, extra))
        pairs = [("PBP", s) for s in sets if s != "PBP"] + [("PBP+IMP_PERS", "PBP+TRUE_PERS"), ("PBP+TRUE_PERS", "PBP+TRUE_PERS+FORM"),
                                                            ("PBP+IMP+IMP_PERS", "PBP+IMP+TRUE_PERS"), ("PBP+PERS", "PBP+TRUE_PERS")]
        deltas.append(deltas_table(task.kind, yv, preds, groups[test], "PBP", cfg, extra, pairs))
        for s, g in gains.items():
            for grp, share in g.items():
                gains_rows.append({**extra, "model": s, "group": grp, "gain_share": share})
        seed_preds = {cfg.seed: {s: preds[s] for s in REFIT_SETS["d"]}}
        for sd in cfg.refit_seeds:
            ps, _ = forward_task(W, task, {s: sets[s] for s in REFIT_SETS["d"]}, train, test, groups, replace(icfg, seed=sd))
            seed_preds[sd] = {s: ps[s] for s in REFIT_SETS["d"]}
        refit_rows += refit_mean_losses(task.name, task.kind, yv, seed_preds)
        lf = W.iloc[test][["game_id", "play_id", "season", "week"]].copy()
        lf["task"] = task.name
        lf["y"] = yv
        all_preds = dict(preds)
        for sd in cfg.refit_seeds:
            all_preds.update({f"{s}@seed{sd}": v for s, v in seed_preds[sd].items()})
        for k, v in all_preds.items():
            lf[f"pred: {k}"] = v
            for lname, l in per_play_losses(task.kind, yv, v).items():
                lf[f"{lname}: {k}"] = l
        loss_frames.append(lf)
    refit = refit_spread_table(pd.DataFrame(refit_rows))
    res = {"metrics": pd.concat(metrics, ignore_index=True), "deltas": attach_refit_spread(pd.concat(deltas, ignore_index=True), refit),
           "gains": pd.DataFrame(gains_rows), "refit": refit, "seconds": time.time() - t0}
    if cfg.write:
        rd = reports_dir()
        res["metrics"].to_parquet(rd / "nfl_04d_metrics.parquet", index=False)
        res["deltas"].to_parquet(rd / "nfl_04d_deltas.parquet", index=False)
        res["gains"].to_parquet(rd / "nfl_04d_gain_shares.parquet", index=False)
        res["refit"].to_parquet(rd / "nfl_04d_refit_noise.parquet", index=False)
        pd.concat(loss_frames, ignore_index=True).to_parquet(proc / "payoff_test_losses_d.parquet", index=False)
    return res


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _load(name: str) -> pd.DataFrame | None:
    p = reports_dir() / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _metric_cols(kind: str) -> list[str]:
    return ["model", "n", "log_loss", "brier", "bss", "auc"] if kind == "binary" else ["model", "n", "r2", "mae", "rmse"]


def _metrics_block(metrics: pd.DataFrame, deltas: pd.DataFrame, task: str, ref: str = REFERENCE_SET) -> str:
    m = metrics[metrics["task"] == task].copy()
    kind = m["kind"].iloc[0]
    cols = [c for c in _metric_cols(kind) if c in m.columns]
    m = m[cols]
    d = deltas[(deltas["task"] == task) & (deltas["from"] == ref)]
    loss = "log_loss" if kind == "binary" else "se"
    d = d[d["loss"] == loss][["to", "delta", "ci_low", "ci_high", "ci_low_game", "ci_high_game"]].rename(columns={"to": "model"})
    d = d.rename(columns={"delta": f"delta_{loss}_vs_{ref}"})
    return md_table(m.merge(d, on="model", how="left"))


def _pairs_block(deltas: pd.DataFrame, task: str, pairs: list[tuple[str, str]]) -> str:
    d = deltas[deltas["task"] == task]
    rows = []
    for a, b in pairs:
        x = d[(d["from"] == a) & (d["to"] == b)]
        rows.append(x)
    x = pd.concat(rows) if rows else d.iloc[0:0]
    return md_table(x[["from", "to", "loss", "delta", "ci_low", "ci_high", "ci_low_game", "ci_high_game", "n", "n_games"]])


def _sig(row: pd.Series) -> str:
    """Verdict of one delta row: the clustered CI against zero, then |delta| against the refit floor of the pair.

    The floor is applied as measured only when both members of the pair were refit
    (``refit_measured == "both"``; a :data:`FIXED_MODELS` member counts as measured with zero
    spread). When only one member was refit the floor is a lower bound and the verdict says so;
    when neither was, the CI verdict stands with a note. Rows without ``refit_measured`` (older
    tables) are treated as fully measured when they carry a spread.
    """
    lo, hi = float(row["ci_low_game"]), float(row["ci_high_game"])
    spread = float(row["refit_spread"]) if "refit_spread" in row.index and pd.notna(row["refit_spread"]) else np.nan
    measured = str(row["refit_measured"]) if "refit_measured" in row.index and pd.notna(row["refit_measured"]) else "both"
    if not (lo > 0 or hi < 0):
        return "no distinguishable change"
    word = "better" if lo > 0 else "worse"
    if np.isnan(spread) or measured == "none":
        return f"{word} (clustered CI excludes 0; refit noise not measured for this pair)"
    inside = abs(float(row["delta"])) <= spread
    if measured != "both":
        which = row["from"] if measured == "from" else row["to"]
        if inside:
            return (f"CI excludes 0 ({word}) but |delta| <= refit spread {spread:.4f} of `{which}` alone (other member not refit): "
                    "within refit noise")
        return f"{word} (clustered CI excludes 0 and |delta| > refit spread {spread:.4f} of `{which}` alone; other member not refit)"
    if inside:
        return f"CI excludes 0 ({word}) but |delta| <= refit spread {spread:.4f} of the pair: within refit noise"
    return f"{word} (clustered CI excludes 0 and |delta| > refit spread {spread:.4f} of the pair)"


def _verdict_lines(deltas: pd.DataFrame, tasks: list[str], pairs: list[tuple[str, str]]) -> list[str]:
    L = []
    for t in tasks:
        d = deltas[deltas["task"] == t]
        kind_loss = "log_loss" if (d["loss"] == "log_loss").any() else "se"
        for a, b in pairs:
            x = d[(d["from"] == a) & (d["to"] == b) & (d["loss"] == kind_loss)]
            if len(x):
                r = x.iloc[0]
                L.append(f"  - {t}: {a} -> {b}: {r['delta']:+.4f} {kind_loss} [game CI {r['ci_low_game']:+.4f}, {r['ci_high_game']:+.4f}], {_sig(r)} (n={int(r['n'])})")
    return L


def _delta(d: pd.DataFrame | None, task: str, a: str, b: str, loss: str) -> pd.Series | None:
    if d is None:
        return None
    x = d[(d["task"] == task) & (d["from"] == a) & (d["to"] == b) & (d["loss"] == loss)]
    return x.iloc[0] if len(x) else None


def _fmt_delta(r: pd.Series | None, unit: str) -> str:
    if r is None:
        return "n/a"
    return f"{r['delta']:+.4f} {unit} [{r['ci_low_game']:+.4f}, {r['ci_high_game']:+.4f}]"


def _fmt_neg(r: pd.Series | None, unit: str) -> str:
    """The same delta read the other way round (``from`` better than ``to`` by ...)."""
    if r is None:
        return "n/a"
    return f"{-r['delta']:+.4f} {unit} [{-r['ci_high_game']:+.4f}, {-r['ci_low_game']:+.4f}]"


def _refit_block(refit: pd.DataFrame | None, intro: str) -> list[str]:
    if refit is None or len(refit) == 0:
        return []
    r = refit[refit["loss"].isin(["log_loss", "se"])]
    return [intro, md_table(r[["task", "model", "loss", "n_seeds", "min_loss", "max_loss", "spread"]], floatfmt="{:.4f}"), ""]


def _within_noise_text(d: pd.DataFrame | None, froms: list[str], tos: list[str]) -> str:
    """Count the from->to deltas whose clustered CI excludes zero and how many of those sit inside the refit spread."""
    if d is None or "refit_spread" not in d.columns:
        return ""
    x = d[d["from"].isin(froms) & d["to"].isin(tos) & d["loss"].isin(["log_loss", "se"])]
    ex = x[(x["ci_low_game"] > 0) | (x["ci_high_game"] < 0)]
    within = ex[ex["delta"].abs() <= ex["refit_spread"]]
    if len(ex) == 0:
        return f"none of the {len(x)} {' / '.join(froms)} -> {' / '.join(tos)} deltas has a clustered CI excluding zero"
    better = ex[(ex["ci_low_game"] > 0) & (ex["delta"].abs() > ex["refit_spread"])]
    items = "; ".join(f"{r['task']} {r['from']} -> {r['to']} {r['delta']:+.4f} vs refit spread {r['refit_spread']:.4f}" for _, r in ex.iterrows())
    verb = lambda k: "is" if k == 1 else "are"
    return (f"{len(ex)} of the {len(x)} {' / '.join(froms)} -> {' / '.join(tos)} deltas have a clustered CI excluding zero, "
            f"{len(within)} of those {verb(len(within))} within the refit spread of the same models and {len(better)} {verb(len(better))} "
            f"an improvement beyond it ({items})")


def _summary_lines(ma: pd.DataFrame | None, da: pd.DataFrame | None, mb: pd.DataFrame | None, db: pd.DataFrame | None,
                   sb: pd.DataFrame | None, rc: pd.DataFrame | None, io: pd.DataFrame | None, dd: pd.DataFrame | None,
                   md_: pd.DataFrame | None) -> list[str]:
    """Plain-language answers computed from the saved tables (game-clustered CIs in brackets)."""
    L = []
    pbp_cp = f"PBP{CP_ROWS_SUFFIX}"
    if ma is not None and da is not None:
        m = ma[(ma["task"] == "pass_completion")].set_index("model")
        cp_txt = ""
        if "nflfastR cp" in m.index and pbp_cp in m.index:
            cp_txt = (f" Like with like on the {int(m.loc['nflfastR cp', 'n'])} cp-present plays: nflfastR cp {m.loc['nflfastR cp', 'log_loss']:.4f} vs "
                      f"PBP {m.loc[pbp_cp, 'log_loss']:.4f}, i.e. cp is better by {_fmt_neg(_delta(da, 'pass_completion', 'nflfastR cp', 'PBP', 'log_loss'), 'nats')} "
                      f"(cp is fitted on many seasons incl. 2017: a reference, not a held-out competitor).")
        n_run = ma[(ma["task"] == "run_yards") & (ma["model"] == "PBP")]["n"]
        n_run_txt = f" and n={int(n_run.iloc[0])} run plays" if len(n_run) else ""
        pre_cp = _delta(da, "pass_completion", "PBP", "PBP+ORACLE_PRESNAP", "log_loss")
        wit_cp = _delta(da, "pass_completion", "PBP", "PBP+ORACLE_WITHIN", "log_loss")
        pre_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE_PRESNAP", "se")
        wit_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE_WITHIN", "se")
        full_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE", "se")
        L.append(f"* **(a) Tracked games (n={int(m.loc['PBP', 'n'])} non-sack, non-spike pass plays{n_run_txt}, 91 games).** Completion "
                 f"log-loss PBP {m.loc['PBP', 'log_loss']:.4f}; adding the imputed state changes it by "
                 f"{_fmt_delta(_delta(da, 'pass_completion', 'PBP', 'PBP+IMP', 'log_loss'), 'nats')} (positive = better), the strictly "
                 f"event-only variant by {_fmt_delta(_delta(da, 'pass_completion', 'PBP', 'PBP+IMP_F0', 'log_loss'), 'nats')}; the true "
                 f"tracking values of all targets (oracle) give {_fmt_delta(_delta(da, 'pass_completion', 'PBP', 'PBP+ORACLE', 'log_loss'), 'nats')}, "
                 f"of which the 12 pre-snap targets alone give {_fmt_delta(pre_cp, 'nats')} and the within-play (post-release) targets alone "
                 f"{_fmt_delta(wit_cp, 'nats')}; the official NGS participation fields "
                 f"{_fmt_delta(_delta(da, 'pass_completion', 'PBP', 'PBP+NGS', 'log_loss'), 'nats')}. Run yards: pre-snap oracle "
                 f"{_fmt_delta(pre_run, 'yd^2')}, within-play oracle {_fmt_delta(wit_run, 'yd^2')}, full oracle {_fmt_delta(full_run, 'yd^2')}. "
                 f"Yards / EPA / run tasks: {_straddle_text(da)}.{cp_txt} "
                 f"The leaky F1 imputations reach log-loss {m.loc['PBP+IMP_F1(leaky)', 'log_loss']:.3f}: that is the label, not a payoff.")
    if mb is not None and db is not None:
        m = mb[(mb["task"] == "pass_completion")].set_index("model")
        n_all, n_cp = int(m.loc["PBP", "n"]), int(m.loc["nflfastR cp", "n"])
        ll_all = float(m.loc["PBP", "log_loss"])
        ll_pbp_cp = float(m.loc[pbp_cp, "log_loss"]) if pbp_cp in m.index else np.nan   # tables from before the like-with-like rows
        ll_ngs_cp = float(m.loc[f"PBP+NGS{CP_ROWS_SUFFIX}", "log_loss"]) if f"PBP+NGS{CP_ROWS_SUFFIX}" in m.index else np.nan
        ll_tw = (n_all * ll_all - n_cp * ll_pbp_cp) / max(n_all - n_cp, 1)
        z_txt = ""
        if sb is not None and len(sb):
            used = sb[((sb["feature_set"] == "F0P") & sb["target"].isin(PRESNAP_TARGETS)) | ((sb["feature_set"] == "F1T") & ~sb["target"].isin(PRESNAP_TARGETS))]
            z_txt = (f" The imputed values themselves drift little across seasons: max |z_shift| {used['z_shift'].abs().max():.2f} over the F0P / F1T "
                     f"columns used by PBP+IMP and {sb['z_shift'].abs().max():.2f} over all feature sets and seasons (shift table).")
        rs = _delta(db, "pass_completion", "PBP", "PBP+IMP", "log_loss")
        r_cp = _delta(db, "pass_completion", "nflfastR cp", "PBP", "log_loss")
        cp_sp = float(r_cp["refit_spread"]) if r_cp is not None and pd.notna(r_cp.get("refit_spread", np.nan)) else np.nan
        if np.isnan(cp_sp):
            cp_qual = ""
        elif abs(float(r_cp["delta"])) <= cp_sp:
            cp_qual = (f" (the clustered CI excludes zero but the delta is inside the {cp_sp:.4f}-nat refit spread of the PBP model, so cp and "
                       f"the PBP set are within refit noise of each other: the PBP set is not better than cp)")
        else:
            cp_qual = f" (beyond the {cp_sp:.4f}-nat refit spread of the PBP model)"
        spread_txt = f" Refit spread of the PBP / PBP+IMP completion models across seeds: {rs['refit_spread']:.4f} nats." if rs is not None and pd.notna(rs.get("refit_spread", np.nan)) else ""
        L.append(f"* **(b) Untracked seasons (train 2018-2021, test 2022, n={n_all} pass plays / 271 games).** "
                 f"Completion log-loss PBP {ll_all:.4f}, PBP+IMP {m.loc['PBP+IMP', 'log_loss']:.4f} "
                 f"({_fmt_delta(rs, 'nats')}), PBP+NGS {m.loc['PBP+NGS', 'log_loss']:.4f} "
                 f"({_fmt_delta(_delta(db, 'pass_completion', 'PBP', 'PBP+NGS', 'log_loss'), 'nats')}). Like with like against nflfastR cp: "
                 f"cp is NaN on the {n_all - n_cp} throwaways (all incompletions, on which PBP has log-loss {ll_tw:.3f}), so on the {n_cp} "
                 f"cp-present plays cp scores {m.loc['nflfastR cp', 'log_loss']:.4f} vs PBP {ll_pbp_cp:.4f} on the same rows: **nflfastR cp beats "
                 f"the PBP set by {_fmt_neg(r_cp, 'nats')}**{cp_qual}; cp is fitted on these seasons, so it is a reference, not a held-out "
                 f"competitor. PBP+NGS {ll_ngs_cp:.4f} on the same rows beats cp by "
                 f"{_fmt_delta(_delta(db, 'pass_completion', 'nflfastR cp', 'PBP+NGS', 'log_loss'), 'nats')} (positive = PBP+NGS better). The all-plays "
                 f"PBP figure {ll_all:.4f} must not be set next to the cp row. Imputed features add nothing out of domain on any of the five "
                 f"tasks ({_within_noise_text(db, ['PBP'], ['PBP+IMP', 'PBP+IMP_F0'])}); stacked on the NGS fields they are neutral or slightly "
                 f"harmful (EPA: {_fmt_delta(_delta(db, 'pass_epa', 'PBP+NGS', 'PBP+IMP+NGS', 'se'), 'EPA^2')}).{z_txt}{spread_txt}")
    if rc is not None and len(rc):
        r = rc.set_index(["ngs_field", "proxy_model"])
        try:
            def R(model: str) -> float:
                return float(r.loc[("avg_separation", model), "test_r2"])
            naive, fair = R("naive (air yards, cp, share, n)"), R(FAIR_BASELINE)
            naive_f1, f1t, f1, f0t = R("naive + imputed F1"), R(f"{FAIR_BASELINE} + imputed F1T"), R(f"{FAIR_BASELINE} + imputed F1"), R(f"{FAIR_BASELINE} + imputed F0T")
            n_test = int(r.loc[("avg_separation", FAIR_BASELINE), "n_test"])

            def G(model: str) -> str:
                """Gain over the fair baseline with its paired-bootstrap CI (delta R2), when the table carries it."""
                if "ci_low" not in r.columns or pd.isna(r.loc[("avg_separation", model), "ci_low"]):
                    return f"{R(model) - fair:+.3f}"
                x = r.loc[("avg_separation", model)]
                return f"{x['delta_r2_vs_baseline']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]"
            io_txt = ""
            if io is not None and len(io):
                q = io.set_index(["imputed_col", "regressors"])
                out_key = [k for k in q.index.get_level_values(1).unique() if k.startswith("outcome")]
                rel_key = [k for k in q.index.get_level_values(1).unique() if k.startswith("at-release")]
                if out_key and rel_key:
                    r2_f1 = float(q.loc[("imp_separation_at_arrival__F1", out_key[0]), "r2_test"])
                    r2_f1t = float(q.loc[("imp_separation_at_arrival__F1T", rel_key[0]), "r2_test"])
                    r2_f1_rel = float(q.loc[("imp_separation_at_arrival__F1", rel_key[0]), "r2_test"])
                    io_txt = (f" At play level a linear regression on the outcome fields (air yards, QB hit, cp, completion, YAC, INT, "
                              f"completion x air yards) reproduces the F1 imputed separation with R2 {r2_f1:.2f} on 2022 plays (at-release fields "
                              f"alone {r2_f1_rel:.2f}; the F1T imputation on the at-release fields {r2_f1t:.2f}), i.e. most of the F1 imputation is a "
                              f"linear re-encoding of the outcome flags, consistent with the NFL 03 `F1_vs_outcome` finding.")
            L.append(f"* **(c) Receiver-week vs NGS (2022 test receiver-weeks, n={n_test}).** Out-of-sample R2 for NGS `avg_separation`: naive "
                     f"event proxies (air yards, cp, share, n) {naive:.3f}; the fair after-the-fact baseline (naive + completion rate, YAC, yards, "
                     f"INT rate, QB-hit rate, EPA, the same fields the F1 student consumes) {fair:.3f}; that baseline + the at-release F1T "
                     f"student {f1t:.3f} (delta R2 {G(f'{FAIR_BASELINE} + imputed F1T')}), + the after-the-fact F1 student {f1:.3f} "
                     f"({G(f'{FAIR_BASELINE} + imputed F1')}), + F0T {f0t:.3f} ({G(f'{FAIR_BASELINE} + imputed F0T')}; brackets = paired "
                     f"bootstrap 95% CI over the 2022 receiver-weeks, 2000 resamples). Against the naive-only baseline the F1 student would look "
                     f"like {naive:.3f} -> {naive_f1:.3f}, but that comparison mismatches information sets and overstates the payoff about "
                     f"twofold.{io_txt} For `avg_cushion` nothing works (best R2 {rc[rc['ngs_field'] == 'avg_cushion']['test_r2'].max():.3f}).")
        except KeyError:
            pass
    if dd is not None:
        r_stack = _delta(dd, "pass_completion", "PBP", "PBP+IMP+IMP_PERS", "log_loss")
        stack_txt = ""
        if r_stack is not None:
            sp = float(r_stack["refit_spread"]) if pd.notna(r_stack.get("refit_spread", np.nan)) else np.nan
            excl = (r_stack["ci_low_game"] > 0) or (r_stack["ci_high_game"] < 0)
            rel = "within" if (not np.isnan(sp) and abs(r_stack["delta"]) <= sp) else "above"
            pbp_txt = ""
            if mb is not None and md_ is not None:
                b_ll = float(mb[(mb["task"] == "pass_completion") & (mb["model"] == "PBP")]["log_loss"].iloc[0])
                d_ll = float(md_[(md_["task"] == "pass_completion") & (md_["model"] == "PBP")]["log_loss"].iloc[0])
                n_b = int(mb[(mb["task"] == "pass_completion") & (mb["model"] == "PBP")]["n_train"].iloc[0])
                n_d = int(md_[(md_["task"] == "pass_completion") & (md_["model"] == "PBP")]["n_train"].iloc[0])
                pbp_txt = (f"; the PBP model alone moves by {abs(b_ll - d_ll):.4f} nats between stages b and d ({b_ll:.4f} vs {d_ll:.4f}, "
                           f"same 2022 rows, {n_b - n_d} training rows without personnel probabilities dropped)")
            stack_txt = (f"; imputed tracking state + imputed personnel: {_fmt_delta(r_stack, 'nats')}, "
                         + ("the only stage-d pair whose clustered CI excludes zero, " if excl else "")
                         + f"|delta| {rel} the refit spread {sp:.4f} nats of the PBP / PBP+IMP+IMP_PERS models under other seeds{pbp_txt}")
        n_txt = ""
        if md_ is not None:
            x = md_[(md_["task"] == "pass_completion") & (md_["model"] == "PBP")]
            if len(x):
                n_txt = f", n={int(x['n'].iloc[0])} test plays, n_train={int(x['n_train'].iloc[0])} 2018-2021 plays with NFL 02 personnel probabilities"
        L.append(f"* **(d) Participation (2022 pass plays{n_txt}).** Imputed personnel probabilities: {_fmt_delta(_delta(dd, 'pass_completion', 'PBP', 'PBP+IMP_PERS', 'log_loss'), 'nats')}; "
                 f"true grouping: {_fmt_delta(_delta(dd, 'pass_completion', 'PBP', 'PBP+TRUE_PERS', 'log_loss'), 'nats')}; imputed -> true: "
                 f"{_fmt_delta(_delta(dd, 'pass_completion', 'PBP+IMP_PERS', 'PBP+TRUE_PERS', 'log_loss'), 'nats')}{stack_txt}. The offense "
                 f"personnel grouping, imputed or true (stage d tests the grouping probabilities, the true one-hot grouping, the position-group "
                 f"counts and the formation on completion and EPA only, not player identity), adds nothing measurable once the at-release "
                 f"fields are known.")
    agg_txt = ""
    if rc is not None and len(rc):
        try:
            r = rc.set_index(["ngs_field", "proxy_model"])
            fair = float(r.loc[("avg_separation", FAIR_BASELINE), "test_r2"])
            f1 = float(r.loc[("avg_separation", f"{FAIR_BASELINE} + imputed F1"), "test_r2"])
            f1t = float(r.loc[("avg_separation", f"{FAIR_BASELINE} + imputed F1T"), "test_r2"])
            agg_txt = (f" For after-the-fact player aggregates (receiver-week separation) the imputation is a modest real gain over the fair "
                       f"after-the-fact baseline ({fair:.2f} -> {f1:.2f} R2 with the F1 student, -> {f1t:.2f} with the at-release F1T student), "
                       f"not a doubling, and most of the F1 signal is a re-encoding of outcome flags.")
        except KeyError:
            pass
    L.append("* **Bottom line.** " + _bottom_line_oracle(da) + " Consistently, the event-only students add nothing on any task, in or out of "
             "domain: their imputations are functions of the same play-by-play fields the outcome model already sees, and LightGBM extracts "
             "that information itself. The only usable extra signal in untracked seasons is the official NGS charting (`was_pressure`, "
             "`time_to_throw`), which is after-the-fact. The nflfastR-style PBP set is itself no better than nflfastR's own `cp` on the "
             "plays where both exist (cp is marginally ahead, within refit noise); only the NGS charting beats `cp`." + agg_txt)
    return L


def _straddle_text(da: pd.DataFrame | None) -> str:
    """Computed statement about the imputed and oracle deltas on the yards / EPA / run tasks (stage a)."""
    if da is None:
        return "see the tables below"
    tasks = ["pass_yards", "pass_epa", "run_yards", "run_success"]
    x = da[da["task"].isin(tasks) & (da["from"] == "PBP") & da["loss"].isin(["log_loss", "se"])]
    imp = x[x["to"].isin(["PBP+IMP", "PBP+IMP_F0"])]
    imp_ex = imp[(imp["ci_low_game"] > 0) | (imp["ci_high_game"] < 0)]
    ora = x[x["to"] == "PBP+ORACLE"]
    ora_pos = ora[ora["ci_low_game"] > 0]
    pre = x[x["to"] == "PBP+ORACLE_PRESNAP"]
    pre_ex = pre[(pre["ci_low_game"] > 0) | (pre["ci_high_game"] < 0)]
    imp_txt = ("imputed deltas all straddle zero" if len(imp_ex) == 0 else
               "imputed deltas with a clustered CI excluding zero: " + "; ".join(f"{r['task']} {r['to']} {r['delta']:+.4f}" for _, r in imp_ex.iterrows()))
    ora_txt = (f"full-oracle deltas clearly positive on {len(ora_pos)} of {len(ora)}" if len(ora) else "no oracle rows")
    pre_txt = ("" if len(pre) == 0 else
               ("; the pre-snap oracle alone straddles zero on all of them" if len(pre_ex) == 0 else
                "; pre-snap-oracle deltas with a clustered CI excluding zero: " + "; ".join(f"{r['task']} {r['delta']:+.4f}" for _, r in pre_ex.iterrows())))
    return f"{imp_txt}, {ora_txt}{pre_txt} (tables below)"


def _bottom_line_oracle(da: pd.DataFrame | None) -> str:
    """Computed first sentences of the bottom line: what the pre-snap and within-play oracles are worth."""
    if da is None:
        return "The tracking state matters (oracle gains on every task), see the tables."
    pre_cp = _delta(da, "pass_completion", "PBP", "PBP+ORACLE_PRESNAP", "log_loss")
    pre_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE_PRESNAP", "se")
    wit_cp = _delta(da, "pass_completion", "PBP", "PBP+ORACLE_WITHIN", "log_loss")
    wit_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE_WITHIN", "se")
    full_cp = _delta(da, "pass_completion", "PBP", "PBP+ORACLE", "log_loss")
    full_run = _delta(da, "run_yards", "PBP", "PBP+ORACLE", "se")
    if pre_cp is None or pre_run is None or full_cp is None or full_run is None:
        return "The tracking state matters (oracle gains on every task), see the tables."

    def share(pre: pd.Series, full: pd.Series) -> str:
        return f"{int(round(100 * float(pre['delta']) / float(full['delta'])))}% of the full-oracle gain" if float(full["delta"]) else "n/a"

    def kind(r: pd.Series) -> str:
        sp = float(r["refit_spread"]) if pd.notna(r.get("refit_spread", np.nan)) else np.nan
        if not (r["ci_low_game"] > 0 or r["ci_high_game"] < 0):
            return "clustered CI straddles zero"
        if not np.isnan(sp) and abs(float(r["delta"])) <= sp:
            return f"CI excludes zero but within the {sp:.4f} refit spread"
        return "clustered CI excludes zero"
    return (f"Perfectly known pre-snap tracking structure has no measurable play-level payoff once the at-release play-by-play fields are known: "
            f"the pre-snap oracle (true values of the 12 pre-snap targets) changes completion log-loss by {_fmt_delta(pre_cp, 'nats')} "
            f"({kind(pre_cp)}; {share(pre_cp, full_cp)}) and run-yards squared error by {_fmt_delta(pre_run, 'yd^2')} ({kind(pre_run)}; "
            f"{share(pre_run, full_run)}). The full-oracle gains on every "
            f"task come from the within-play targets ({_fmt_delta(wit_cp, 'nats')} on completion, {_fmt_delta(wit_run, 'yd^2')} on run yards): "
            f"separation and defenders near the target at ball arrival, time to throw / pressure at the release, yards to first contact, all "
            f"measured after the throw or handoff and therefore quasi-outcomes. No at-release student can recover them, and the oracle row is "
            f"a ceiling for a tracking-informed post-hoc description of the play, not for any at-release imputation.")


def write_report(path: Path, timings: dict[str, float] | None = None) -> None:
    """Assemble ``nfl_04_payoff.md`` from the saved ``nfl_04*`` tables."""
    ma, da, ca, sa, ga = (_load(f"nfl_04a_{k}") for k in ("metrics", "deltas", "calibration", "students", "gain_shares"))
    mb, db, cb, gb, sb = (_load(f"nfl_04b_{k}") for k in ("metrics", "deltas", "calibration", "gain_shares", "shift"))
    cc, covc, rc = _load("nfl_04c_receiver_week_correlations"), _load("nfl_04c_coverage"), _load("nfl_04c_receiver_week_regression")
    io = _load("nfl_04c_imputation_vs_outcome")
    md_, dd, gd = (_load(f"nfl_04d_{k}") for k in ("metrics", "deltas", "gain_shares"))
    ra, rb, rdn = (_load(f"nfl_04{k}_refit_noise") for k in ("a", "b", "d"))
    timings = timings or {}
    L = ["# NFL 04 payoff: does imputed tracking state improve event-only outcome prediction?\n",
         "Machine-written by `python -m research.privileged_tracking.nfl.payoff`. Question: within the tracked games and on "
         "untracked seasons, does adding imputed tracking state (NFL 03 students) to an nflfastR-style event-only feature set "
         "improve the prediction of observable outcomes, and how does that compare with the true tracking values (oracle) and "
         "with the official NGS participation fields?\n",
         "## Protocol\n",
         "* **Tasks.** " + "; ".join(f"`{t.name}` = {t.description} ({t.kind}, target `{t.target}`)" for t in TASKS) + ". "
         "Sacks and spikes are flagged by nflfastR (`sack`, `qb_spike` / `play_type`) and excluded from the pass tasks; throwaways are not "
         "flagged and stay in (nflfastR `cp` is NaN on them, so the `nflfastR cp` rows and the `cp` deltas cover the cp-present plays only).",
         "* **Feature sets.** `PBP` = pre-snap situation (down, distance, yard line, quarter, clocks, score, `wp`, `ep`, `xpass`, "
         "shotgun / no-huddle, home) + strictly-prior-week team tendencies (`tend_*`, from full-season play-by-play) + at-release fields "
         "(pass: `air_yards`, `pass_length`, `pass_location`, `qb_hit`, receiver position group, `qb_scramble`; run: `run_location`, "
         "`run_gap`, `qb_scramble`). `PBP+PERS` adds the personnel counts and formation (participation charting). `PBP+IMP` adds the "
         "F0P out-of-fold pre-snap imputations (" + ", ".join(PRESNAP_TARGETS) + ") and the at-release F1T imputations of the "
         "within-play targets (pass: " + ", ".join(WITHIN_TARGETS["pass"]) + "; run: " + ", ".join(WITHIN_TARGETS["run"]) + "). "
         "`PBP+IMP_F0` is the strictly event-only variant (F0 / F0T students, no personnel input). `PBP+ORACLE_PRESNAP` adds the true "
         "tracking values of the 12 pre-snap targets, `PBP+ORACLE_WITHIN` the true values of the within-play targets and `PBP+ORACLE` "
         "both. The within-play targets are measured at or after the throw / handoff (separation and defenders near the target at ball "
         "arrival, time to throw and pressure at the release, yards to first contact and defenders at first contact of the carrier), so "
         "they are quasi-outcomes: the `PBP+ORACLE` / `PBP+ORACLE_WITHIN` rows bound a tracking-informed post-hoc description of the play, "
         "not what an at-release imputation could recover; `PBP+ORACLE_PRESNAP` is the relevant ceiling for pre-snap imputation. "
         "`PBP+NGS` adds personnel + the official participation fields (`defenders_in_box`, "
         "`number_of_pass_rushers`, `ngs_air_yards`, `was_pressure`, `time_to_throw`; the last two are absent on sacks, which are excluded "
         "anyway). `PBP+IMP+NGS` stacks both. `PBP+IMP_F1(leaky)` (stage a only) uses the NFL 03 F1 students, whose inputs include "
         "`complete_pass` / `yards_gained` / `epa`: it is shown to demonstrate the label leak, not as a payoff.",
         "* **Why F1T.** The NFL 03 F1 students see the outcome of the play, so their within-play imputations cannot be features of an "
         "outcome model. F1T / F0T students are fitted here with the NFL 03 machinery (same LightGBM settings, same 5 game folds, "
         "team encodings fitted inside the training fold) on the F0P / F0 inputs plus the at-release fields only "
         "(`" + "`, `".join(AT_RELEASE_COLS) + "`); their bundles are saved as `models/students_F1T.joblib` / `students_F0T.joblib`. "
         "`qb_hit` is a charted flag of a hit on the passer during the play, so 'at release' here means 'up to and including QB contact'; "
         "it sits in `PBP` and in the F1T / F0T inputs alike, so it cannot create an imputed-vs-PBP difference, but the pass_yards / EPA "
         "models do get a mild post-release hint from it.",
         "* **Splits.** Stage a: 5-fold group k-fold by `gameId` (`common.splits.group_kfold`, seed 0, the NFL 03 folds), LightGBM with "
         "early stopping on an inner 20% game holdout of the training fold; every imputed feature of a test play is out-of-fold (its "
         "students never saw that game). Stages b-d: train 2018-2021, test 2022 once; the imputed features come from students trained "
         "on the 2017 tracked games only (a disjoint domain), the NFL 02 personnel probabilities are out-of-fold by game (2016-2020) or "
         "model predictions (2021-2022).",
         "* **Statistics.** Metrics with n; paired bootstrap (1000 resamples, seed 0) of per-play log-loss (binary) or squared / absolute "
         "error (regression), play-level and game-clustered 95% CIs; deltas are `from` minus `to`, positive = `to` better. Quote the "
         "game-clustered CI. `base_global` = training-fold mean; `nflfastR cp` = the external completion-probability column.",
         "* **Refit noise.** The bootstrap CIs condition on one fit. The `PBP`, `PBP+IMP` and `PBP+IMP_F0` models (stage a also "
         "`PBP+ORACLE_PRESNAP`; stage d: `PBP` and `PBP+IMP+IMP_PERS`) are refit under seeds " + ", ".join(str(x) for x in (0,) + PayoffConfig().refit_seeds) + " (the inner "
         "early-stopping holdout and the LightGBM bagging / feature-fraction seed change; folds and the 2018-2021 / 2022 split do not) and "
         "the spread max - min of their mean loss is tabulated in `nfl_04{a,b,d}_refit_noise.parquet`. Every deltas table carries "
         "`refit_spread` = the larger spread of the two members of the pair (`nflfastR cp` and `base_global` involve no fit and count as "
         "zero) and `refit_measured` = `both` / `from` / `to` / `none` saying which members were actually refit. A delta whose |value| is "
         "at or below the spread of a fully measured pair is reported as within refit noise even when its clustered CI excludes zero; when "
         "only one member was refit the floor is a lower bound and the verdict line says so.",
         "* **Like with like against `cp`.** `nflfastR cp` is NaN on throwaways, so every completion table also scores the sets on the cp-present "
         "rows (`<set> [cp rows]`); only those rows and the `nflfastR cp -> <set>` deltas are comparable with the `cp` row.\n"]
    L.append("## Summary\n")
    L += _summary_lines(ma, da, mb, db, sb, rc, io, dd, md_)
    L.append("")
    # ---- stage a
    L.append("## (a) Tracked games 2017 (k-fold by game)\n")
    if ma is not None:
        for t in TASKS:
            L.append(f"### {t.name}: {t.description}\n")
            L.append(_metrics_block(ma, da, t.name))
            L.append("")
            pairs = [("PBP+PERS", "PBP+IMP"), ("PBP+IMP_F0", "PBP+IMP"), ("PBP+IMP", "PBP+ORACLE"), ("PBP+ORACLE_PRESNAP", "PBP+ORACLE"),
                     ("PBP+ORACLE_WITHIN", "PBP+ORACLE"), ("PBP+NGS", "PBP+IMP+NGS")]
            if t.name == "pass_completion":
                pairs += [("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+IMP")]
            L.append("Pairwise deltas (positive = `to` better):\n")
            L.append(_pairs_block(da, t.name, pairs))
            L.append("")
        if ca is not None and len(ca):
            L.append("### Calibration (10 equal-count bins, mean predicted vs observed)\n")
            for t in ("pass_completion", "run_success"):
                c = ca[ca["task"] == t]
                for mname in ("PBP", "PBP+IMP", "PBP+ORACLE_PRESNAP", "PBP+ORACLE", "PBP+NGS", "nflfastR cp"):
                    x = c[c["model"] == mname]
                    if len(x):
                        L.append(f"`{t}` / `{mname}`: " + ", ".join(f"{r.pred:.2f}->{r.obs:.2f}" for r in x.itertuples()) +
                                 f" (bins of {int(x['n'].iloc[0])})")
            L.append("")
        if sa is not None:
            L.append("### At-release students (F1T / F0T): out-of-fold skill on the tracked plays\n")
            L.append("Skill = R2 (continuous / count) or Brier skill (binary), k-fold OOF; `skill_F0P` / `skill_F1` are the NFL 03 "
                     "students on the same rows (F1 sees the outcome), `skill_base_outcome` the per-outcome-class mean (also outcome-aware).\n")
            L.append(md_table(sa[["target", "subset", "kind", "student", "n", "skill", "skill_F0P", "skill_F1", "skill_base_outcome"]], floatfmt="{:.3f}"))
            L.append("")
        if ga is not None:
            L.append("### Where the gain comes from (LightGBM gain share by feature group, mean over folds)\n")
            piv = ga.pivot_table(index=["task", "model"], columns="group", values="gain_share", aggfunc="mean").reset_index()
            L.append(md_table(piv, floatfmt="{:.3f}"))
            L.append("")
        L.append("**Verdict (a).**\n")
        L += _verdict_lines(da, [t.name for t in TASKS], [("PBP", "PBP+IMP"), ("PBP", "PBP+IMP_F0"), ("PBP", "PBP+ORACLE_PRESNAP"),
                                                          ("PBP", "PBP+ORACLE_WITHIN"), ("PBP", "PBP+ORACLE"), ("PBP+ORACLE_PRESNAP", "PBP+ORACLE"),
                                                          ("PBP", "PBP+NGS"), ("PBP+NGS", "PBP+IMP+NGS"), ("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+NGS")])
        L.append("")
        L += _refit_block(ra, "Refit noise (k-fold OOF mean loss of the same model under three seeds; `spread` = max - min):\n")
    else:
        L.append("_stage a not run_\n")
    # ---- stage b
    L.append("## (b) Untracked seasons: train 2018-2021, test 2022\n")
    if mb is not None:
        for t in TASKS:
            L.append(f"### {t.name}: {t.description}\n")
            L.append(_metrics_block(mb, db, t.name))
            L.append("")
            pairs = [("PBP+PERS", "PBP+IMP"), ("PBP+IMP_F0", "PBP+IMP"), ("PBP+NGS", "PBP+IMP+NGS")]
            if t.name == "pass_completion":
                pairs += [("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+IMP")]
            L.append("Pairwise deltas (positive = `to` better):\n")
            L.append(_pairs_block(db, t.name, pairs))
            L.append("")
        if cb is not None and len(cb):
            L.append("### Calibration on 2022 (10 equal-count bins)\n")
            for t in ("pass_completion", "run_success"):
                c = cb[cb["task"] == t]
                for mname in ("PBP", "PBP+IMP", "PBP+NGS", "PBP+IMP+NGS", "nflfastR cp"):
                    x = c[c["model"] == mname]
                    if len(x):
                        L.append(f"`{t}` / `{mname}`: " + ", ".join(f"{r.pred:.2f}->{r.obs:.2f}" for r in x.itertuples()) +
                                 f" (bins of {int(x['n'].iloc[0])})")
            L.append("")
        if gb is not None:
            L.append("### Gain share by feature group (2022 models)\n")
            piv = gb.pivot_table(index=["task", "model"], columns="group", values="gain_share", aggfunc="mean").reset_index()
            L.append(md_table(piv, floatfmt="{:.3f}"))
            L.append("")
        if sb is not None:
            L.append("### Distribution shift of the imputed values, 2017 OOF vs 2018-2022\n")
            L.append("`z_shift` = (season mean - 2017 OOF mean) / 2017 OOF sd; `truth2017` = tracked truth. One row per target and "
                     "season for the feature sets used by `PBP+IMP` (F0P pre-snap, F1T within-play); the full table (all sets) is in "
                     "`nfl_04b_shift.parquet`.\n")
            s = sb[sb["feature_set"].isin(["F0P", "F1T"])]
            s = s[(s["feature_set"] == "F0P") & s["target"].isin(PRESNAP_TARGETS) | (s["feature_set"] == "F1T") & ~s["target"].isin(PRESNAP_TARGETS)]
            piv = s.pivot_table(index=["target", "feature_set", "oof2017_mean", "oof2017_sd", "truth2017_mean"], columns="season", values="z_shift").reset_index()
            piv.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"z_{c}" for c in piv.columns]
            L.append(md_table(piv, floatfmt="{:.2f}"))
            L.append("")
        L.append("**Verdict (b).**\n")
        L += _verdict_lines(db, [t.name for t in TASKS], [("PBP", "PBP+IMP"), ("PBP", "PBP+IMP_F0"), ("PBP", "PBP+NGS"), ("PBP+NGS", "PBP+IMP+NGS"), ("PBP+PERS", "PBP+IMP"), ("nflfastR cp", "PBP"), ("nflfastR cp", "PBP+NGS")])
        L.append("")
        L += _refit_block(rb, "Refit noise (2022 mean loss of the same model under three seeds; `spread` = max - min):\n")
    else:
        L.append("_stage b not run_\n")
    # ---- stage c
    L.append("## (c) Receiver-week aggregation vs NGS weekly receiving (2018-2022)\n")
    if cc is not None:
        L.append("Imputed separation (`separation_at_arrival`, students F1 = after the fact, F1T = at release, F0T = at release without "
                 "personnel) and cushion (`cb_cushion`, F0P / F0) averaged over a receiver's targeted non-sack passes in a week, joined to "
                 "the NGS weekly `avg_separation` / `avg_cushion` of the same season / week / gsis id. Naive event-only proxies: mean "
                 "`air_yards`, mean nflfastR `cp`, target share (targets / team targets that week). After-the-fact event aggregates (the "
                 "fields the F1 student consumes, so the fair after-the-fact baseline): completion rate, mean YAC per target (0 on "
                 "incompletions), mean yards gained, interception rate, QB-hit rate, mean EPA. `within_player_pearson` demeans by "
                 "(season, player). NGS weekly rows exist only for receivers with enough targets (min 5 in the file).\n")
        if covc is not None:
            L.append(md_table(covc, floatfmt="{:.3f}"))
            L.append("")
        L.append("Correlations over all joined receiver-weeks (every NGS weekly row joins, so no extra target threshold is applied):\n")
        L.append(md_table(cc[["ngs_field", "proxy", "n", "pearson", "spearman", "within_player_pearson", "n_within"]], floatfmt="{:.3f}"))
        L.append("")
        if rc is not None and len(rc):
            L.append("Out-of-sample proxy regressions: OLS of the NGS field on the listed receiver-week proxies, fitted on 2018-2021 "
                     "receiver-weeks and scored on 2022 (`test_pearson`, `test_r2`). The naive model combines mean air yards, mean `cp`, "
                     "target share and target count; `naive + after-the-fact` adds the after-the-fact aggregates; the imputed rows add the "
                     "student means. The gain of a student is read against the row with the same event information "
                     "(`naive + after-the-fact` for F1, which consumes the outcome; the F1T / F0T at-release students are shown against both). "
                     "`delta_r2_vs_baseline` is the R2 gain over `naive + after-the-fact` on the same 2022 receiver-weeks with a paired "
                     "bootstrap 95% CI (`ci_low` / `ci_high`, 2000 resamples of receiver-weeks, per-week squared errors).\n")
            L.append(md_table(rc, floatfmt="{:.3f}"))
            L.append("")
        if io is not None and len(io):
            L.append("Play-level linear re-encoding check: OLS of each imputed column on the at-release fields and on the outcome fields of the "
                     "same play (fitted on 2018-2021 non-sack pass plays with a receiver, `r2_test` on 2022). A high R2 means the imputation "
                     "is mostly a function of those fields.\n")
            L.append(md_table(io, floatfmt="{:.3f}"))
            L.append("")
        L.append("**Verdict (c).** The student's contribution is the `naive + after-the-fact + imputed` row minus the `naive + after-the-fact` "
                 "row (same event information on both sides); the `naive` -> `naive + imputed F1` difference mixes the student with the "
                 "after-the-fact fields it consumes and overstates it. The summary above quotes the fair numbers.\n")
    else:
        L.append("_stage c not run_\n")
    # ---- stage d
    L.append("## (d) Participation payoff on 2022 pass plays\n")
    if md_ is not None:
        L.append("`PBP+IMP_PERS` adds the NFL 02 offense-grouping probabilities (`p_off_*`, 10 classes; out-of-fold by game for 2016-2020, "
                 "model predictions for 2021-2022), `PBP+TRUE_PERS` the one-hot true grouping, `+FORM` the true formation, `PBP+PERS` the "
                 "true position-group counts + formation; `PBP+IMP+*` stack the imputed tracking state on top.\n")
        for t in ("pass_completion", "pass_epa"):
            L.append(f"### {t}\n")
            L.append(_metrics_block(md_, dd, t, ref="PBP"))
            L.append("")
            L.append(_pairs_block(dd, t, [("PBP+IMP_PERS", "PBP+TRUE_PERS"), ("PBP+PERS", "PBP+TRUE_PERS"), ("PBP+TRUE_PERS", "PBP+TRUE_PERS+FORM"), ("PBP+IMP+IMP_PERS", "PBP+IMP+TRUE_PERS")]))
            L.append("")
        if gd is not None:
            piv = gd.pivot_table(index=["task", "model"], columns="group", values="gain_share", aggfunc="mean").reset_index()
            L.append(md_table(piv, floatfmt="{:.3f}"))
            L.append("")
        L.append("**Verdict (d).**\n")
        L += _verdict_lines(dd, ["pass_completion", "pass_epa"], [("PBP", "PBP+IMP_PERS"), ("PBP", "PBP+TRUE_PERS"), ("PBP", "PBP+IMP+IMP_PERS"), ("PBP+IMP_PERS", "PBP+TRUE_PERS"), ("PBP+IMP+IMP_PERS", "PBP+IMP+TRUE_PERS")])
        L.append("")
        L += _refit_block(rdn, "Refit noise (2022 mean loss of the same model under three seeds; `spread` = max - min):\n")
    else:
        L.append("_stage d not run_\n")
    L.append("## Caveats\n")
    L += ["* The NFL 03 F1 students are inadmissible as outcome-model features (their inputs contain the label); the `PBP+IMP_F1(leaky)` "
          "row in stage a quantifies the leak and must not be read as a payoff. The F1T / F0T students used instead see the play only up "
          "to the release / handoff.",
          "* Stage a uses the NFL 03 folds, so every test-fold imputation comes from students that never saw that game; the "
          "training-fold imputations were produced by students fitted on the other folds (the usual stacking protocol), a second-order "
          "optimism that affects the `PBP+IMP*` rows only and, given their null result, cannot change the conclusion.",
          "* The imputed features are deterministic functions of the event inputs already in `PBP` (+ personnel), so `PBP+IMP` can only "
          "help through the representation the students learned from tracking targets (learning with privileged information); the tables "
          "measure exactly that channel. In stage b the paired CIs are game-clustered over 271 games.",
          "* The bootstrap CIs condition on a single fit; the refit-noise tables (`nfl_04{a,b,d}_refit_noise.parquet`, three seeds) give "
          "the spread of the mean loss of the same model, and the verdict lines flag a delta as within refit noise when |delta| is at or "
          "below the larger spread of the two models of the pair (`refit_measured` = `both`); pairs with only one refit member are labelled "
          "as such and pairs with none keep the plain CI verdict. The `PBP` model also differs between stages b and d (identical 2022 rows, "
          "570 fewer training rows in d because plays without NFL 02 personnel probabilities are dropped), which is the same order of magnitude.",
          "* The oracle rows are not a ceiling for at-release imputation: `PBP+ORACLE_WITHIN` / `PBP+ORACLE` contain within-play targets "
          "measured at or after the throw / handoff (separation at arrival, defenders near the target, yards to first contact), which are "
          "quasi-outcomes of the play; `PBP+ORACLE_PRESNAP` is the ceiling that matters for pre-snap imputation. `qb_hit` (a charted "
          "post-play flag of a hit on the passer) is part of the at-release set on every side of every comparison; 'at release' therefore "
          "means 'up to and including QB contact'.",
          "* `nflfastR cp` is NaN on throwaways (about 4% of non-sack pass plays); its metrics row and the `cp` deltas cover the cp-present "
          "plays only. Compare it only with the `<set> [cp rows]` rows: the all-plays rows include the throwaways, on which every model "
          "predicts a near-zero completion probability and so scores a near-zero loss, which lowers their average.",
          "* Stage c: the after-the-fact aggregates use the outcome of the aggregated targets by design (an after-the-fact analytics "
          "setting, not prediction); the F1 student consumes the same fields, so its gain is quoted against `naive + after-the-fact`. Mean "
          "YAC per target counts incompletions as 0 yards (so every receiver-week has a value); averaging YAC over completions only and "
          "dropping receiver-weeks without a completion gives a slightly lower baseline on slightly fewer weeks. The play-level regressions "
          "in `nfl_04c_imputation_vs_outcome.parquet` show how much of each imputation those fields reproduce.",
          "* The `PBP+IMP` set includes the F0P pre-snap students, whose inputs are the participation personnel / formation strings; "
          "`PBP+IMP_F0` is the strictly event-only variant and `PBP+PERS` isolates the personnel information itself.",
          "* Team target encodings inside the students are 2017 values; tendencies in `PBP` are strictly-prior weeks of the same season "
          "(week 1 uses the constant priors). nflfastR `cp` is a model fitted on many seasons (including these), so it is a reference, not "
          "a held-out competitor; it is NaN on throwaways.",
          "* Stage b/d LightGBM uses learning rate 0.1 with early stopping on an inner 20% game holdout of 2018-2021 (the 80% model is "
          "scored on 2022, no refit); stage a uses the NFL 03 settings (0.05). No subsampling anywhere.",
          "* Stage c joins on gsis id / season / week; NGS weekly rows cover only receivers with >= 5 targets and use NGS' own target "
          "count (`ngs_targets`), which differs slightly from the nflfastR count of targeted passes.",
          "* Run times: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timings.items()) + "." if timings else "* Run times: see the console log."]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cfg: PayoffConfig) -> dict[str, Any]:
    res: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for s in cfg.stages:
        fn = {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d}[s]
        r = fn(cfg)
        res[s] = r
        timings[s] = r["seconds"]
        print(f"stage {s}: {r['seconds'] / 60:.1f} min", flush=True)
    if cfg.write:
        write_report(reports_dir() / "nfl_04_payoff.md", timings)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="a,b,c,d", help="comma-separated subset of a,b,c,d or 'none' (report only)")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--force-apply", action="store_true")
    args = ap.parse_args()
    stages = tuple() if args.stages.strip() == "none" else tuple(s.strip() for s in args.stages.split(",") if s.strip())
    cfg = PayoffConfig(n_jobs=args.n_jobs, stages=stages, force_apply=args.force_apply)
    res = run(cfg)
    for s, r in res.items():
        if "metrics" in r:
            with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
                print(f"--- stage {s} metrics\n{r['metrics']}")
        if "correlations" in r:
            with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
                print(f"--- stage {s} correlations\n{r['correlations']}")


if __name__ == "__main__":
    main()
