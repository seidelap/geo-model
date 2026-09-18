"""Soccer 03 - pure helpers for the teacher-student payoff experiments (no data access).

Everything the driver (``payoff.py``) needs that can be unit-tested on synthetic frames: the
feature-set catalogue of the xG and xPass experiments (:data:`XG_VARIANTS`, :data:`XPASS_VARIANTS`),
the event-only designs built on top of the stage-02 design matrices
(:func:`shot_event_design`, :func:`pass_event_design`), the assist link
(:func:`key_pass_links`, :func:`assist_block`), the labels (:func:`pass_completion`), the
slicing keys (:func:`pattern_group`, :func:`distance_band`, :func:`length_band`), the
match-stratified subsample, the soft (distillation) labels and the metric / paired-delta tables.

Leakage contract (see ``soccer/CLAUDE.md``): the event-only designs read only ``f_*`` columns
(through :func:`imputation_features.build_design`) plus ``gender`` / ``competition``; the shot's
own post-instant ``f_after_duration`` is dropped; ``y_*`` (360), ``sff_*`` (shot freeze frame),
``oracle_*`` and ``post_*`` columns enter only through the explicitly named oracle blocks, and the
outcome of the very shot / pass is only ever the label. Assist attributes are the *preceding*
pass's realised trajectory (``f_after_pass_*`` of the key pass), which is complete before the shot
instant and therefore event-only for the shot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.privileged_tracking.common.metrics import (
    brier,
    calibration_table,
    clustered_bootstrap_delta,
    log_loss,
    paired_bootstrap_delta,
    per_sample_log_loss,
)
from research.privileged_tracking.soccer import imputation_features as imf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GbmParams:
    """LightGBM settings of one downstream model family.

    Attributes:
        learning_rate / num_leaves / min_data_in_leaf / feature_fraction / bagging_fraction /
            max_bin / lambda_l2: LightGBM.
        max_rounds / early_stopping: rounds are chosen by early stopping on an inner
            match-grouped holdout of the training rows (``inner_holdout_frac`` of the matches).
    """

    learning_rate: float = 0.03
    num_leaves: int = 15
    min_data_in_leaf: int = 40
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    max_bin: int = 63
    lambda_l2: float = 1.0
    max_rounds: int = 2000
    early_stopping: int = 100
    inner_holdout_frac: float = 0.15


@dataclass
class PayoffConfig:
    """Driver configuration (stage 03).

    Attributes:
        n_jobs: LightGBM threads.
        seed: base seed (folds come from the stage-02 ``fold`` column, not from here).
        n_seeds_xg: seeds averaged per xG variant (bagging over the inner holdout / feature
            subsampling randomness).
        n_seeds_xpass: same for xPass (1: the pass sample is large).
        xg / xpass: :class:`GbmParams` of the two model families.
        xpass_target_n: expected size of the match-stratified pass subsample.
        student_train_cap: training rows per domain-restricted student fit (stage ``students``).
        n_boot: bootstrap resamples of every paired delta.
        exclude_penalties: drop ``f_shot_type == 'Penalty'`` (no freeze frame, fixed xG).
        smoke: tiny run (first matches only), separate cache, nothing written to reports.
        write: write report / parquet outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    n_seeds_xg: int = 3
    n_seeds_xpass: int = 1
    xg: GbmParams = field(default_factory=GbmParams)
    xpass: GbmParams = field(
        default_factory=lambda: GbmParams(
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=100,
            max_rounds=1500,
            early_stopping=50,
        )
    )
    xpass_target_n: int = 150_000
    student_train_cap: int = 150_000
    n_boot: int = 2000
    exclude_penalties: bool = True
    smoke: bool = False
    write: bool = True


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

#: imputed / 360 / shot-frame state added to the shot's event-only design
SHOT_STATE: tuple[str, ...] = (
    "n_opp_in_cone",
    "nearest_opp_dist_in_cone",
    "opp_keeper_dist_to_goal_line",
    "n_opp_within_5",
    "block_depth",
)
#: pass-geometry state of the assist (key pass), added next to the shot state
ASSIST_STATE: tuple[str, ...] = ("n_opp_in_lane", "n_opp_within_3_of_end")
#: state added to the pass's event-only design
PASS_STATE: tuple[str, ...] = (
    "n_opp_within_3_of_end",
    "n_opp_in_lane",
    "nearest_opp_to_receiver",
    "n_opp_ahead_of_ball",
    "block_depth",
    "nearest_opp_dist",
    "n_opp_within_5",
)
#: shot.freeze_frame column carrying the same quantity (block depth has no sff analogue: the
#: count of opponents ahead of the ball is the closest shot-frame proxy)
SFF_MAP: dict[str, str] = {
    "n_opp_in_cone": "sff_n_opp_in_cone",
    "nearest_opp_dist_in_cone": "sff_nearest_opp_dist_in_cone",
    "opp_keeper_dist_to_goal_line": "sff_opp_keeper_dist_to_goal_line",
    "n_opp_within_5": "sff_n_opp_within_5",
    "block_depth": "sff_n_opp_ahead_of_ball",
}
SFF_FULL: tuple[str, ...] = (
    "sff_present",
    "sff_n_opp",
    "sff_n_tm",
    "sff_opp_keeper_visible",
    "sff_n_opp_in_cone",
    "sff_nearest_opp_dist_in_cone",
    "sff_opp_keeper_in_cone",
    "sff_cone_area",
    "sff_n_opp_within_5",
    "sff_n_opp_within_10",
    "sff_nearest_opp_dist",
    "sff_n_opp_in_box",
    "sff_n_opp_ahead_of_ball",
    "sff_opp_keeper_dist_to_goal_line",
    "sff_opp_keeper_y",
)
#: non-negative quantities (counts, distances): imputations are clipped at 0 (stage-02 open issue)
NON_NEGATIVE: frozenset[str] = frozenset(
    {
        "n_opp_in_cone",
        "nearest_opp_dist_in_cone",
        "opp_keeper_dist_to_goal_line",
        "n_opp_within_5",
        "n_opp_within_10",
        "block_depth",
        "n_opp_in_lane",
        "n_opp_within_3_of_end",
        "nearest_opp_to_receiver",
        "n_opp_ahead_of_ball",
        "nearest_opp_dist",
    }
)

#: stage-02 design columns that are constant or post-instant on Shot rows
SHOT_DROP: tuple[str, ...] = ("f_type_id", "f_pass_type", "f_pass_body_part", "f_after_duration")
#: stage-02 design columns that are constant on Pass rows (``f_after_duration`` stays in the
#: with-after design because the E2a students were trained with it; ``EVENT-nodur`` ablates it)
PASS_DROP: tuple[str, ...] = (
    "f_type_id",
    "f_shot_body_part",
    "f_shot_technique",
    "f_shot_type",
    "f_shot_first_time",
    "f_after_carry_length",
    "f_after_carry_end_x",
    "f_after_carry_end_y",
    "f_after_carry_progress",
)
LOC_ONLY: tuple[str, ...] = tuple(imf.LOC_COLS)

#: assist (key pass) attributes copied onto the shot row, prefixed ``a_``
ASSIST_COLS: dict[str, str] = {
    "f_x": "a_x",
    "f_y": "a_y",
    "f_pass_type": "a_pass_type",
    "f_pass_body_part": "a_pass_body_part",
    "f_under_pressure": "a_under_pressure",
    "f_after_pass_length": "a_pass_length",
    "f_after_pass_angle": "a_pass_angle",
    "f_after_pass_height": "a_pass_height",
    "f_after_pass_switch": "a_pass_switch",
    "f_after_pass_cross": "a_pass_cross",
    "f_after_pass_through_ball": "a_pass_through_ball",
    "f_after_pass_cut_back": "a_pass_cut_back",
    "f_after_pass_end_x": "a_pass_end_x",
    "f_after_pass_end_y": "a_pass_end_y",
    "f_after_duration": "a_pass_duration",
    "f_t_period": "a_t_period",
}
ASSIST_CATEGORICAL: dict[str, list[str]] = {
    "a_pass_type": imf.VOCABS["f_pass_type"],
    "a_pass_body_part": imf.VOCABS["f_pass_body_part"],
    "a_pass_height": imf.VOCABS["f_after_pass_height"],
}
ASSIST_BOOL: tuple[str, ...] = (
    "a_under_pressure",
    "a_pass_switch",
    "a_pass_cross",
    "a_pass_through_ball",
    "a_pass_cut_back",
)


@dataclass(frozen=True)
class Variant:
    """One downstream feature set.

    Attributes:
        name: short name used in every table.
        base: ``event`` (full event-only design), ``loc`` (location only) or ``none``.
        state: which state block is appended: ``none``, ``imp`` (out-of-fold student
            predictions), ``oracle360`` (360 frame), ``oracleshot`` (shot.freeze_frame, same
            quantities), ``oracleshot_full`` (every ``sff_*`` column).
        fset: stage-02 feature set the imputations come from (``imp`` only).
        description: one line for the report.
    """

    name: str
    base: str
    state: str
    fset: str = "E2"
    description: str = ""


XG_VARIANTS: tuple[Variant, ...] = (
    Variant("BASE", "none", "none", description="fold-wise goal rate of the training matches"),
    Variant("LOC", "loc", "none", description="x, y, distance, opening angle, bearing (LightGBM)"),
    Variant(
        "EVENT",
        "event",
        "none",
        description="event-only: stage-02 E2 design on the shot (location geometry, body part, "
        "technique, shot type, first time, play pattern, position, under pressure, "
        "minute / period, "
        "score, possession context, 10-event window, opponent defensive actions, gender, "
        "competition type) + assist (key pass) attributes",
    ),
    Variant(
        "EVENT+IMP",
        "event",
        "imp",
        "E2",
        "EVENT + out-of-fold E2 student predictions of the five shot-state quantities + E2a "
        "predictions of the assist's pass-lane quantities",
    ),
    Variant(
        "EVENT+IMP(E0)",
        "event",
        "imp",
        "E0",
        "ablation: EVENT + out-of-fold E0 (current-event-only) student predictions",
    ),
    Variant(
        "EVENT+ORACLE360",
        "event",
        "oracle360",
        description="EVENT + the same quantities measured on the 360 frame (stage-02 label "
        "rules: NaN where the frame is unusable / unreliable / the cone is empty)",
    ),
    Variant(
        "EVENT+ORACLESHOT",
        "event",
        "oracleshot",
        description="EVENT + the same quantities from shot.freeze_frame (n_opp_ahead_of_ball "
        "stands in for block depth) + the assist's 360 pass-lane quantities",
    ),
    Variant(
        "EVENT+ORACLESHOT_full",
        "event",
        "oracleshot_full",
        description="EVENT + every sff_* column of the shot freeze frame",
    ),
)
XPASS_VARIANTS: tuple[Variant, ...] = (
    Variant(
        "BASE", "none", "none", description="fold-wise completion rate of the training matches"
    ),
    Variant(
        "EVENT",
        "event",
        "none",
        description="event-only: stage-02 E2a design on the pass (start / end location, length, "
        "angle, height, duration, switch / cross / through-ball / cut-back flags, pass type, "
        "body part, "
        "under pressure, play pattern, position, possession context, window, gender, competition "
        "type); the no-after design drops the realised end / length / height / flags",
    ),
    Variant(
        "EVENT-nodur",
        "event",
        "none",
        description="ablation (after design only): EVENT without the realised pass duration, "
        "the one E2a student input that the task's xPass feature list does not name",
    ),
    Variant(
        "EVENT+IMP",
        "event",
        "imp",
        "E2a",
        "EVENT + out-of-fold student predictions of the seven pass-state quantities (E2a "
        "students for the with-after design, E2 students for the no-after design)",
    ),
    Variant(
        "EVENT+ORACLE",
        "event",
        "oracle360",
        description="EVENT + the same quantities measured on the 360 frame",
    ),
)
XG_REFERENCE = "STATSBOMB_XG"


def variant_by_name(variants: tuple[Variant, ...], name: str) -> Variant:
    for v in variants:
        if v.name == name:
            return v
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Assist link and designs
# ---------------------------------------------------------------------------


def key_pass_links(events: list[dict[str, Any]]) -> pd.DataFrame:
    """``(event_id, key_pass_id)`` of every Shot in a raw StatsBomb event list ``[n_shots, 2]``.

    ``key_pass_id`` is NaN/None when the shot has no key pass.
    """
    rows = []
    for e in events:
        if (e.get("type") or {}).get("name") != "Shot":
            continue
        rows.append({"event_id": e["id"], "key_pass_id": (e.get("shot") or {}).get("key_pass_id")})
    return pd.DataFrame(rows, columns=["event_id", "key_pass_id"])


def assist_block(shots: pd.DataFrame, passes: pd.DataFrame) -> pd.DataFrame:
    """Assist attributes aligned to ``shots`` ``[n_shots, len(ASSIST_COLS) + 2]``.

    Args:
        shots: frame with ``key_pass_id`` and ``f_t_period`` (shot instant).
        passes: build-stage Pass rows with ``event_id`` and the keys of :data:`ASSIST_COLS`.

    Returns:
        ``a_has_assist`` (0/1), the renamed pass columns (NaN without assist) and ``a_dt``
        (seconds from the pass to the shot within the period).
    """
    p = passes.drop_duplicates("event_id").set_index("event_id")
    key = shots["key_pass_id"].to_numpy(dtype=object)
    has = pd.Series(key).map(lambda k: isinstance(k, str) and k in p.index).to_numpy(dtype=bool)
    out = pd.DataFrame(index=shots.index)
    out["a_has_assist"] = has.astype(np.float32)
    idx = pd.Index([k if h else None for k, h in zip(key, has, strict=True)])
    for src, dst in ASSIST_COLS.items():
        if src not in p.columns:
            out[dst] = np.nan
            continue
        vals = p[src].reindex(idx).to_numpy(dtype=object)
        out[dst] = vals
    dt = shots["f_t_period"].to_numpy(dtype=float) - pd.to_numeric(
        out["a_t_period"], errors="coerce"
    ).to_numpy(dtype=float)
    out["a_dt"] = np.where(has, dt, np.nan)
    out = out.drop(columns=["a_t_period"])
    return out


def encode_assist(block: pd.DataFrame) -> pd.DataFrame:
    """Float32 encoding of :func:`assist_block` (fixed vocabularies, bools -> 0/1)."""
    out: dict[str, np.ndarray] = {}
    for c in block.columns:
        if c in ASSIST_CATEGORICAL:
            out[c] = imf.encode_category(block[c], ASSIST_CATEGORICAL[c]).astype(np.float32)
        elif c in ASSIST_BOOL:
            v = block[c].to_numpy(dtype=object)
            out[c] = np.array(
                [
                    np.nan
                    if x is None or (isinstance(x, float) and np.isnan(x))
                    else float(bool(x))
                    for x in v
                ],
                dtype=np.float32,
            )
        else:
            out[c] = pd.to_numeric(block[c], errors="coerce").to_numpy(dtype=np.float32)
    return pd.DataFrame(out, index=block.index)


def drop_constant_columns(x: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with a single distinct non-NaN value (or all NaN)."""
    keep = [c for c in x.columns if x[c].nunique(dropna=True) > 1]
    return x[keep]


def shot_event_design(shots: pd.DataFrame, assist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Event-only design of a Shot frame ``[n, d]``: stage-02 E2 design minus :data:`SHOT_DROP`,
    plus the encoded assist block. Constant columns are dropped."""
    x = imf.build_design(shots, "E2")
    x = x.drop(columns=[c for c in SHOT_DROP if c in x.columns])
    if assist is not None:
        x = pd.concat([x, encode_assist(assist)], axis=1)
    return drop_constant_columns(x)


def pass_event_design(passes: pd.DataFrame, with_after: bool) -> pd.DataFrame:
    """Event-only design of a Pass frame ``[n, d]``: stage-02 E2a (``with_after``) or E2 design
    minus :data:`PASS_DROP`; constant columns dropped."""
    x = imf.build_design(passes, "E2a" if with_after else "E2")
    x = x.drop(columns=[c for c in PASS_DROP if c in x.columns])
    return drop_constant_columns(x)


def categorical_in(x: pd.DataFrame) -> list[str]:
    """Categorical columns (stage-02 vocabularies + assist vocabularies) present in ``x``."""
    cats = set(imf.CATEGORICAL_COLS) | set(ASSIST_CATEGORICAL)
    return [c for c in x.columns if c in cats]


def clip_state(values: np.ndarray, name: str) -> np.ndarray:
    """Clip an imputed quantity to its support (non-negative for counts / distances)."""
    v = np.asarray(values, dtype=np.float32)
    return np.clip(v, 0.0, None) if name in NON_NEGATIVE else v


def state_block(
    table: pd.DataFrame,
    names: tuple[str, ...],
    source: str,
    fset: str = "E2",
    prefix: str = "s_",
    col_prefix: str = "",
) -> pd.DataFrame:
    """State columns ``[n, len(names)]`` from a joined table.

    Args:
        table: frame holding ``<col_prefix>imp_<name>__<fset>`` (student),
            ``<col_prefix>y_<name>`` (360) and / or the ``sff_*`` columns.
        names: quantities to take.
        source: ``imp`` / ``oracle360`` / ``oracleshot``.
        fset: student feature set for ``imp``.
        prefix: output column prefix (``s_`` shot state, ``as_`` assist state).
        col_prefix: input column prefix (``a_`` for the assist pass's columns on a shot row).
    """
    out = pd.DataFrame(index=table.index)
    for n in names:
        if source == "imp":
            col = f"{col_prefix}imp_{n}__{fset}"
            vals = clip_state(table[col].to_numpy(dtype=np.float32), n)
        elif source == "oracle360":
            vals = table[f"{col_prefix}y_{n}"].to_numpy(dtype=np.float32)
        elif source == "oracleshot":
            vals = table[SFF_MAP[n]].to_numpy(dtype=np.float32)
        else:  # pragma: no cover - catalogue error
            raise KeyError(source)
        out[f"{prefix}{n}"] = vals
    return out


# ---------------------------------------------------------------------------
# Labels, slices and subsamples
# ---------------------------------------------------------------------------


def pass_completion(post_pass_outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(y, keep)``: completion label (1 = Complete) and the rows that count as attempts
    (``Unknown`` and ``Injury Clearance`` outcomes are dropped)."""
    o = pd.Series(np.asarray(post_pass_outcome, dtype=object)).fillna("Complete").to_numpy()
    y = (o == "Complete").astype(float)
    keep = ~np.isin(o, ["Unknown", "Injury Clearance"])
    return y, keep


def pattern_group(play_pattern: np.ndarray) -> np.ndarray:
    """``regular`` / ``counter`` / ``set_piece`` / ``other`` from the StatsBomb play pattern."""
    p = np.asarray(play_pattern, dtype=object)
    out = np.full(len(p), "other", dtype=object)
    out[p == "Regular Play"] = "regular"
    out[p == "From Counter"] = "counter"
    out[np.isin(p, list(imf.SET_PIECE_PATTERNS))] = "set_piece"
    return out


DISTANCE_EDGES = (0.0, 8.0, 12.0, 18.0, 25.0, np.inf)
LENGTH_EDGES = (0.0, 10.0, 20.0, 35.0, np.inf)


def _band(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), "unknown", dtype=object)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        lab = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"
        out[(v >= lo) & (v < hi)] = lab
    return out


def distance_band(dist_goal: np.ndarray) -> np.ndarray:
    """Shot distance bands (yd): 0-8, 8-12, 12-18, 18-25, 25+."""
    return _band(dist_goal, DISTANCE_EDGES)


def length_band(length: np.ndarray) -> np.ndarray:
    """Pass length bands (yd): 0-10, 10-20, 20-35, 35+."""
    return _band(length, LENGTH_EDGES)


def stratified_subsample(match_id: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    """Boolean mask keeping the same expected share of every match, ``target_n`` rows expected."""
    m = np.asarray(match_id)
    n = len(m)
    if target_n >= n:
        return np.ones(n, dtype=bool)
    frac = target_n / n
    # per-match RNG streams (seeded by the integer match id) keep a match's sample fixed
    # regardless of which other matches are present
    keep = np.zeros(n, dtype=bool)
    for mid in np.unique(m):
        idx = np.where(m == mid)[0]
        r = np.random.default_rng([int(seed), int(mid) % (2**31)]).random(len(idx))
        keep[idx] = r < frac
    return keep


def soft_label(goal: np.ndarray, teacher: np.ndarray, alpha: float) -> np.ndarray:
    """``alpha * teacher + (1 - alpha) * goal`` clipped to (0, 1) - the distillation label."""
    t = np.clip(np.asarray(teacher, dtype=float), 1e-4, 1 - 1e-4)
    return alpha * t + (1.0 - alpha) * np.asarray(goal, dtype=float)


# ---------------------------------------------------------------------------
# Metrics and deltas
# ---------------------------------------------------------------------------


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """n, positives, log-loss, Brier, AUC, ECE (10 equal-count bins), mean prediction."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    cal = calibration_table(y, p, 10) if len(y) >= 10 else []
    ece = float(sum(abs(r["pred"] - r["obs"]) * r["n"] for r in cal) / len(y)) if cal else np.nan
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "log_loss": log_loss(y, p) if len(y) else np.nan,
        "brier": brier(y, p) if len(y) else np.nan,
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "ece": ece,
        "mean_pred": float(p.mean()) if len(y) else np.nan,
    }


def paired_delta(
    y: np.ndarray,
    p_ref: np.ndarray,
    p_var: np.ndarray,
    match: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired log-loss and Brier deltas (reference minus variant; positive = variant better).

    Both a per-sample paired bootstrap and a match-clustered bootstrap CI are returned.
    """
    y = np.asarray(y, dtype=float)
    a = np.clip(np.asarray(p_ref, dtype=float), 0, 1)
    b = np.clip(np.asarray(p_var, dtype=float), 0, 1)
    ok = ~(np.isnan(a) | np.isnan(b))
    y, a, b, g = y[ok], a[ok], b[ok], np.asarray(match)[ok]
    if len(y) < 2:
        return {"n": int(len(y))}
    la, lb = per_sample_log_loss(y, a), per_sample_log_loss(y, b)
    d, lo, hi = paired_bootstrap_delta(la, lb, n_boot=n_boot, seed=seed)
    _, clo, chi = clustered_bootstrap_delta(la, lb, g, n_boot=n_boot, seed=seed)
    ba, bb = (a - y) ** 2, (b - y) ** 2
    db, blo, bhi = paired_bootstrap_delta(ba, bb, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(y)),
        "delta_log_loss": d,
        "ci_low": lo,
        "ci_high": hi,
        "ci_low_clustered": clo,
        "ci_high_clustered": chi,
        "delta_brier": db,
        "brier_ci_low": blo,
        "brier_ci_high": bhi,
        "significant": bool(lo > 0 or hi < 0),
    }


def metrics_table(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One :func:`binary_metrics` row per prediction column."""
    return pd.DataFrame([{"variant": k, **binary_metrics(y, p)} for k, p in preds.items()])


def deltas_table(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    reference: str,
    match: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired deltas of every prediction column against ``preds[reference]``."""
    rows = []
    for k, p in preds.items():
        if k == reference:
            continue
        rows.append(
            {
                "variant": k,
                "reference": reference,
                **paired_delta(y, preds[reference], p, match, n_boot=n_boot, seed=seed),
            }
        )
    return pd.DataFrame(rows)


def slice_table(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    reference: str,
    slices: dict[str, np.ndarray],
    match: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    min_n: int = 50,
) -> pd.DataFrame:
    """Per-slice log-loss of every variant and paired deltas against the reference.

    Args:
        slices: ``{slice_name: label per row}``.

    Returns:
        rows ``(slice, level, variant, n, positives, log_loss, delta_log_loss, ci_low, ci_high,
        ci_low_clustered, ci_high_clustered)``.
    """
    rows = []
    y = np.asarray(y, dtype=float)
    for sname, labels in slices.items():
        labels = np.asarray(labels, dtype=object)
        for lev in pd.unique(labels):
            m = labels == lev
            if m.sum() < min_n:
                continue
            for k, p in preds.items():
                mm = binary_metrics(y[m], p[m])
                row = {
                    "slice": sname,
                    "level": lev,
                    "variant": k,
                    "n": mm["n"],
                    "positives": mm["positives"],
                    "log_loss": mm["log_loss"],
                    "brier": mm["brier"],
                }
                if k != reference:
                    d = paired_delta(
                        y[m], preds[reference][m], p[m], np.asarray(match)[m], n_boot, seed
                    )
                    row.update(
                        {
                            "delta_log_loss": d.get("delta_log_loss", np.nan),
                            "ci_low": d.get("ci_low", np.nan),
                            "ci_high": d.get("ci_high", np.nan),
                            "ci_low_clustered": d.get("ci_low_clustered", np.nan),
                            "ci_high_clustered": d.get("ci_high_clustered", np.nan),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_frame(
    y: np.ndarray, preds: dict[str, np.ndarray], n_bins: int = 10
) -> pd.DataFrame:
    """Equal-count reliability bins of every prediction column (``variant, bin, pred, obs, n``)."""
    rows = []
    for k, p in preds.items():
        p = np.asarray(p, dtype=float)
        ok = ~np.isnan(p)
        for i, r in enumerate(
            calibration_table(np.asarray(y, dtype=float)[ok], np.clip(p[ok], 0, 1), n_bins)
        ):
            rows.append({"variant": k, "bin": i, **r})
    return pd.DataFrame(rows)
