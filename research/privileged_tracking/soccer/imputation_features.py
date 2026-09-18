"""Soccer 02 - pure helpers for the 360-state imputation students (no data access).

Everything the driver (``imputation.py``) and the applier (``apply_student.py``) share:
the target catalogue (:data:`TARGETS`), the nested event-only feature sets
(:data:`FEATURE_SETS`), the fixed categorical vocabularies (:data:`VOCABS`), the design
matrix builder (:func:`build_design`), the row subsets a student is trained on and applied to
(:func:`subset_masks`, :func:`target_frame`), the two derived binaries (``deep_block``,
``counter_on``) and the sequence-block encodings used by the MLP student and the error
analysis (:func:`mlp_design`, :func:`events_since_opp_def_action`).

Leakage contract (see ``soccer/CLAUDE.md``): only ``f_*`` and ``seq_*`` columns of the build
stage's tables are features. ``f_after_*`` columns are event-only but POST-INSTANT and live in
their own block (:data:`AFTER_COLS`) so every experiment can be run with and without them.
``y_*``, ``sff_*``, ``oracle_*`` and ``post_*`` columns are never read by :func:`build_design`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

DEEP_BLOCK_QUANTILE = 0.25
SETTLED_MIN_ELAPSED_S = 10.0
MID_THIRD_X = (40.0, 80.0)
ATTACKING_HALF_X = 60.0
COUNTER_MAX_OPP_AHEAD = 3
MIN_OPP_VISIBLE_RELIABLE = 7  # the build stage's y_reliable rule (documented, not re-applied)


@dataclass(frozen=True)
class TargetSpec:
    """One imputation target.

    Attributes:
        name: short name (column suffix of every prediction / model).
        source: ``y_*`` column of ``events360.parquet`` or ``"derived"``.
        kind: ``reg`` (continuous), ``count`` (non-negative integer, fitted as regression)
            or ``binary``.
        family: ``shape`` (team-shape quantity, needs a reliable frame), ``ball``
            (ball-relative, all usable frames) or ``pass`` (Pass rows only).
        subset: name of the row subset the student is trained on and applied to
            (:func:`subset_masks`).
        label: label-validity rule, ``reliable`` (``y_reliable == 1``) or ``frame_ok``
            (``y_frame_ok == 1``); rows failing it are never trained or scored on.
        definition: one-line description for the report.
    """

    name: str
    source: str
    kind: str
    family: str
    subset: str
    label: str
    definition: str


TARGETS: list[TargetSpec] = [
    TargetSpec(
        "block_depth",
        "y_block_depth",
        "reg",
        "shape",
        "poss",
        "reliable",
        "120 - mean x of the visible outfield opponents (yd from their goal line)",
    ),
    TargetSpec(
        "def_line",
        "y_def_line",
        "reg",
        "shape",
        "poss",
        "reliable",
        "120 - max x of the visible outfield opponents (deepest defender, yd from their goal line)",
    ),
    TargetSpec(
        "block_width",
        "y_block_width",
        "reg",
        "shape",
        "poss",
        "reliable",
        "y range of the visible outfield opponents (yd)",
    ),
    TargetSpec(
        "block_length",
        "y_block_length",
        "reg",
        "shape",
        "poss",
        "reliable",
        "x range of the visible outfield opponents (yd)",
    ),
    TargetSpec(
        "n_opp_ahead_of_ball",
        "y_n_opp_ahead_of_ball",
        "count",
        "shape",
        "poss",
        "reliable",
        "visible opponents (incl. keeper) with x > ball x",
    ),
    TargetSpec(
        "deep_block",
        "derived",
        "binary",
        "shape",
        "settled",
        "reliable",
        "def_line <= its 25th percentile over settled possession events (Regular Play, "
        "possession >= 10 s old); threshold stored in the bundle",
    ),
    TargetSpec(
        "counter_on",
        "derived",
        "binary",
        "shape",
        "mid",
        "reliable",
        "n_opp_ahead_of_ball <= 3 with the ball in the middle third (40 <= x < 80)",
    ),
    TargetSpec(
        "n_opp_within_5",
        "y_n_opp_within_5",
        "count",
        "ball",
        "all",
        "frame_ok",
        "opponents (incl. keeper) within 5 yd of the ball",
    ),
    TargetSpec(
        "n_opp_within_10",
        "y_n_opp_within_10",
        "count",
        "ball",
        "all",
        "frame_ok",
        "opponents within 10 yd of the ball",
    ),
    TargetSpec(
        "nearest_opp_dist",
        "y_nearest_opp_dist",
        "reg",
        "ball",
        "all",
        "frame_ok",
        "distance from the ball to the nearest visible opponent (yd)",
    ),
    TargetSpec(
        "n_opp_in_cone",
        "y_n_opp_in_cone",
        "count",
        "ball",
        "poss",
        "frame_ok",
        "outfield opponents inside the triangle ball -> posts (possession-team events)",
    ),
    TargetSpec(
        "nearest_opp_dist_in_cone",
        "y_nearest_opp_dist_in_cone",
        "reg",
        "ball",
        "poss",
        "frame_ok",
        "nearest outfield opponent inside the cone (yd); possession-team events, defined only "
        "when the cone is not empty",
    ),
    TargetSpec(
        "opp_keeper_dist_to_goal_line",
        "y_opp_keeper_dist_to_goal_line",
        "reg",
        "ball",
        "poss_att",
        "keeper",
        "120 - opponent keeper x (yd); possession-team events in the attacking half (x >= 60), "
        "defined only when the keeper is visible and consistently flagged (y_keeper_consistent)",
    ),
    TargetSpec(
        "n_opp_within_3_of_end",
        "y_n_opp_within_3_of_end",
        "count",
        "pass",
        "pass",
        "frame_ok",
        "Pass only: opponents within 3 yd of the pass end location",
    ),
    TargetSpec(
        "n_opp_in_lane",
        "y_n_opp_in_lane",
        "count",
        "pass",
        "pass",
        "frame_ok",
        "Pass only: opponents within 2 yd of the start -> end segment",
    ),
]
TARGET_BY_NAME: dict[str, TargetSpec] = {t.name: t for t in TARGETS}
TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "shape": tuple(t.name for t in TARGETS if t.family == "shape"),
    "ball": tuple(t.name for t in TARGETS if t.family == "ball"),
    "pass": tuple(t.name for t in TARGETS if t.family == "pass"),
}


def settled_mask(df: pd.DataFrame) -> np.ndarray:
    """Settled possession: possession-team event in Regular Play, possession >= 10 s old ``[n]``."""
    return (
        df["f_is_possession_team"].to_numpy(dtype=bool)
        & (df["f_play_pattern"].to_numpy(dtype=object) == "Regular Play")
        & (df["f_poss_elapsed"].to_numpy(dtype=float) >= SETTLED_MIN_ELAPSED_S)
    )


def mid_third_mask(df: pd.DataFrame) -> np.ndarray:
    """Possession-team event with the ball in the middle third ``[n]``."""
    x = df["f_x"].to_numpy(dtype=float)
    return (
        df["f_is_possession_team"].to_numpy(dtype=bool)
        & (x >= MID_THIRD_X[0])
        & (x < MID_THIRD_X[1])
    )


def subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row subsets a student is trained on and applied to (boolean arrays ``[n]``).

    ``poss`` possession-team events (the 360 frame then shows the defending team);
    ``poss_att`` possession-team events in the attacking half (x >= 60); ``all`` every row;
    ``pass`` Pass events; ``settled`` / ``mid`` see :func:`settled_mask` / :func:`mid_third_mask`.
    """
    n = len(df)
    poss = df["f_is_possession_team"].to_numpy(dtype=bool)
    return {
        "all": np.ones(n, dtype=bool),
        "poss": poss,
        "poss_att": poss & (df["f_x"].to_numpy(dtype=float) >= ATTACKING_HALF_X),
        "pass": df["f_type"].to_numpy(dtype=object) == "Pass",
        "settled": settled_mask(df),
        "mid": mid_third_mask(df),
    }


def label_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Label-validity rules: ``reliable`` (>= 7 opponents visible, frame usable), ``frame_ok``
    and ``keeper`` (frame usable and every visible keeper at the expected end of the pitch)."""
    frame_ok = df["y_frame_ok"].to_numpy(dtype=float) == 1.0
    return {
        "reliable": df["y_reliable"].to_numpy(dtype=float) == 1.0,
        "frame_ok": frame_ok,
        "keeper": frame_ok & (df["y_keeper_consistent"].to_numpy(dtype=float) == 1.0),
    }


def deep_block_threshold(df: pd.DataFrame) -> float:
    """25th percentile of ``y_def_line`` over reliable settled possession events."""
    m = settled_mask(df) & label_masks(df)["reliable"]
    v = df.loc[m, "y_def_line"].to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    return float(np.quantile(v, DEEP_BLOCK_QUANTILE)) if len(v) else float("nan")


def target_frame(df: pd.DataFrame, deep_block_thr: float) -> pd.DataFrame:
    """One float column per target, NaN outside the subset or where the label is invalid.

    Args:
        df: events360-like frame with the ``y_*`` sources, ``f_*`` subset columns and flags.
        deep_block_thr: :func:`deep_block_threshold` (fixed once, stored in the bundle).

    Returns:
        ``[n, len(TARGETS)]`` frame with columns ``TARGETS[i].name``.
    """
    subs = subset_masks(df)
    labs = label_masks(df)
    out = pd.DataFrame(index=df.index)
    for t in TARGETS:
        if t.source == "derived":
            if t.name == "deep_block":
                v = (df["y_def_line"].to_numpy(dtype=float) <= deep_block_thr).astype(float)
                v[np.isnan(df["y_def_line"].to_numpy(dtype=float))] = np.nan
            elif t.name == "counter_on":
                a = df["y_n_opp_ahead_of_ball"].to_numpy(dtype=float)
                v = (a <= COUNTER_MAX_OPP_AHEAD).astype(float)
                v[np.isnan(a)] = np.nan
            else:  # pragma: no cover - catalogue error
                raise KeyError(t.name)
        else:
            v = df[t.source].to_numpy(dtype=float).copy()
        v[~(subs[t.subset] & labs[t.label])] = np.nan
        out[t.name] = v
    return out


# ---------------------------------------------------------------------------
# Feature sets and vocabularies
# ---------------------------------------------------------------------------

#: fixed StatsBomb vocabularies (ontology, not learned from a fold); unseen strings -> len(vocab)
VOCABS: dict[str, list[str]] = {
    "f_play_pattern": [
        "Regular Play",
        "From Throw In",
        "From Free Kick",
        "From Goal Kick",
        "From Corner",
        "From Kick Off",
        "From Keeper",
        "From Counter",
        "Other",
    ],
    "f_position": [
        "Goalkeeper",
        "Right Back",
        "Right Center Back",
        "Center Back",
        "Left Center Back",
        "Left Back",
        "Right Wing Back",
        "Left Wing Back",
        "Right Defensive Midfield",
        "Center Defensive Midfield",
        "Left Defensive Midfield",
        "Right Midfield",
        "Right Center Midfield",
        "Center Midfield",
        "Left Center Midfield",
        "Left Midfield",
        "Right Wing",
        "Right Attacking Midfield",
        "Center Attacking Midfield",
        "Left Attacking Midfield",
        "Left Wing",
        "Right Center Forward",
        "Striker",
        "Center Forward",
        "Left Center Forward",
        "Secondary Striker",
    ],
    "f_poss_start_type": [
        "set_piece",
        "recovery",
        "duel",
        "interception",
        "kick_off",
        "keeper",
        "open_play",
        "other",
        "unknown",
    ],
    "f_pass_type": [
        "Open Play",
        "Recovery",
        "Throw-in",
        "Free Kick",
        "Interception",
        "Corner",
        "Goal Kick",
        "Kick Off",
    ],
    "f_pass_body_part": [
        "Right Foot",
        "Left Foot",
        "Head",
        "Keeper Arm",
        "Other",
        "Drop Kick",
        "No Touch",
    ],
    "f_shot_body_part": ["Right Foot", "Left Foot", "Head", "Other"],
    "f_shot_technique": [
        "Normal",
        "Half Volley",
        "Volley",
        "Lob",
        "Backheel",
        "Overhead Kick",
        "Diving Header",
    ],
    "f_shot_type": ["Open Play", "Free Kick", "Penalty", "Corner", "Kick Off"],
    "f_after_pass_height": ["Ground Pass", "Low Pass", "High Pass"],
}
N_TYPE_IDS = 37  # build stage's type vocabulary incl. <pad> = 0 and <unknown> = 36

#: club leagues (everything else is an international tournament)
LEAGUES = frozenset(
    {"1. Bundesliga", "La Liga", "Ligue 1", "Major League Soccer", "Premier League", "Serie A"}
)

LOC_COLS = ["f_x", "f_y", "f_dist_goal", "f_goal_opening", "f_goal_bearing"]
E0_COLS = [
    "f_type_id",
    *LOC_COLS,
    "f_play_pattern",
    "f_position",
    "f_under_pressure",
    "f_counterpress",
    "f_minute",
    "f_t_period",
    "f_period",
    "f_home",
    "f_is_possession_team",
    "f_score_for",
    "f_score_against",
    "f_score_diff",
    "f_goals_total",
    "f_pass_type",
    "f_pass_body_part",
    "f_shot_body_part",
    "f_shot_technique",
    "f_shot_type",
    "f_shot_first_time",
    "f_gender",
    "f_comp_type",
]
POSS_COLS = [
    "f_poss_elapsed",
    "f_poss_n_events",
    "f_poss_n_passes",
    "f_poss_ball_dist",
    "f_poss_start_type",
    "f_poss_t_since_ft_entry",
    "f_poss_in_final_third",
    "f_poss_start_x",
    "f_poss_start_y",
    "f_dt_prev",
]
WINDOW_GROUPS = (
    "pass",
    "carry",
    "receipt",
    "pressure",
    "duel",
    "dribble",
    "shot",
    "interception",
    "block",
    "clearance",
    "recovery",
    "foul",
    "keeper",
    "loss",
    "other",
    "own",
    "opp_def",
)
WINDOW_COLS = [f"f_w10_n_{g}" for g in WINDOW_GROUPS] + [
    "f_w10_n",
    "f_opp_def_x_60s_mean",
    "f_opp_def_n_60s",
    "f_opp_poss_last10s",
    "f_ball_speed_3",
    "f_ball_dx_3",
    "f_ball_dy_3",
    "f_t_since_opp_def_action",
]
AFTER_COLS = [
    "f_after_duration",
    "f_after_pass_length",
    "f_after_pass_angle",
    "f_after_pass_height",
    "f_after_pass_switch",
    "f_after_pass_cross",
    "f_after_pass_through_ball",
    "f_after_pass_cut_back",
    "f_after_pass_end_x",
    "f_after_pass_end_y",
    "f_after_pass_end_dist_goal",
    "f_after_pass_progress",
    "f_after_carry_length",
    "f_after_carry_end_x",
    "f_after_carry_end_y",
    "f_after_carry_progress",
]
SEQ_LEN = 20
SEQ_FIELDS = ("type", "x", "y", "dt", "same")
SEQ_COLS = [f"seq_{f}_{k:02d}" for k in range(1, SEQ_LEN + 1) for f in SEQ_FIELDS]

FEATURE_SETS: dict[str, list[str]] = {
    "loc": list(LOC_COLS),
    "E0": list(E0_COLS),
    "E1": E0_COLS + POSS_COLS,
    "E2": E0_COLS + POSS_COLS + WINDOW_COLS,
    "E2a": E0_COLS + POSS_COLS + WINDOW_COLS + AFTER_COLS,
    "E3": E0_COLS + POSS_COLS + WINDOW_COLS + SEQ_COLS,
}
#: ``E2r`` = E2 features, trained on reliable rows only (a training-set variant, same design)
FEATURE_SET_ALIAS: dict[str, str] = {"E2r": "E2"}
FEATURE_SET_ORDER = ["loc", "E0", "E1", "E2", "E2a"]
FEATURE_SET_DESCRIPTION: dict[str, str] = {
    "loc": "location only: x, y, distance / opening angle / bearing to goal (LightGBM baseline)",
    "E0": "current event only: type, location geometry, play pattern, actor position, "
    "under-pressure / "
    "counterpress flags, minute / period, home, possession-team flag, score state, pre-instant "
    "pass / shot "
    "attributes, gender, competition type (league / tournament)",
    "E1": "E0 + possession context: elapsed time, events and passes so far, ball distance, start "
    "type / "
    "location, time since final-third entry, time since previous event",
    "E2": "E1 + 10-event window counts by type group, opponent defensive actions (mean x and "
    "count in 60 s, "
    "seconds since the last one), opponent possession in the last 10 s, ball speed / displacement",
    "E2a": "E2 + the 16 POST-INSTANT f_after_* columns (realised pass end / length / height / "
    "flags, carry "
    "end / length, duration)",
    "E2r": "E2 features, trained on reliable rows only (>= 7 opponents visible)",
    "E3": "E2 + the raw 20-slot sequence block (type id, x, y, dt, same-team per slot) - LightGBM "
    "reference "
    "for the MLP student",
}
CATEGORICAL_COLS = ["f_type_id", *VOCABS.keys()] + [
    f"seq_type_{k:02d}" for k in range(1, SEQ_LEN + 1)
]
BOOL_COLS = [
    "f_under_pressure",
    "f_counterpress",
    "f_home",
    "f_is_possession_team",
    "f_shot_first_time",
    "f_poss_in_final_third",
    "f_opp_poss_last10s",
    "f_after_pass_switch",
    "f_after_pass_cross",
    "f_after_pass_through_ball",
    "f_after_pass_cut_back",
]


def design_columns(feature_set: str) -> list[str]:
    """Columns of a feature set (aliases such as ``E2r`` resolve to their design)."""
    return list(FEATURE_SETS[FEATURE_SET_ALIAS.get(feature_set, feature_set)])


def categorical_columns(feature_set: str) -> list[str]:
    """The categorical columns among :func:`design_columns`."""
    cols = set(design_columns(feature_set))
    return [c for c in CATEGORICAL_COLS if c in cols]


def encode_category(values: pd.Series | np.ndarray, vocab: list[str]) -> np.ndarray:
    """Integer codes over a fixed vocabulary ``[n]`` (float: NaN stays NaN, unseen ->
    ``len(vocab)``)."""
    s = pd.Series(np.asarray(values, dtype=object)).astype(object)
    lookup = {v: float(i) for i, v in enumerate(vocab)}
    out = s.map(
        lambda v: (
            np.nan
            if v is None or (isinstance(v, float) and np.isnan(v))
            else lookup.get(v, float(len(vocab)))
        )
    )
    return out.to_numpy(dtype=float)


def comp_type(competition: pd.Series | np.ndarray) -> np.ndarray:
    """0 = club league (:data:`LEAGUES`), 1 = international tournament ``[n]``."""
    return np.asarray(
        [0.0 if c in LEAGUES else 1.0 for c in np.asarray(competition, dtype=object)], dtype=float
    )


def build_design(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Float32 design matrix ``[n, len(design_columns(feature_set))]`` from a build-stage frame.

    Strings are encoded with :data:`VOCABS`, booleans become 0/1, ``f_gender`` is 1 for female
    competitions and ``f_comp_type`` comes from :func:`comp_type`; NaN is kept (LightGBM
    handles missing values). Only ``f_*`` / ``seq_*`` columns plus ``gender`` / ``competition``
    are read.
    """
    cols = design_columns(feature_set)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        if c == "f_gender":
            out[c] = (df["gender"].to_numpy(dtype=object) == "female").astype(np.float32)
        elif c == "f_comp_type":
            out[c] = comp_type(df["competition"]).astype(np.float32)
        elif c in VOCABS:
            out[c] = encode_category(df[c], VOCABS[c]).astype(np.float32)
        elif c in BOOL_COLS:
            v = df[c]
            out[c] = (
                v.astype("float32").to_numpy()
                if v.isna().any()
                else v.to_numpy(dtype=bool).astype(np.float32)
            )
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32)
    return pd.DataFrame(out, index=df.index, columns=cols)


# ---------------------------------------------------------------------------
# Sequence block helpers (MLP design, error analysis)
# ---------------------------------------------------------------------------

#: type ids (build vocabulary) of the opponent defensive actions counted by f_w10_n_opp_def
DEF_TYPE_IDS = frozenset({4, 6, 7, 8, 12})  # Pressure, Duel, Block, Clearance, Interception
#: coarse type groups for the per-slot one-hot of the MLP design
SEQ_TYPE_GROUPS: dict[str, frozenset[int]] = {
    "pass": frozenset({1}),
    "receipt": frozenset({2}),
    "carry": frozenset({3}),
    "def": DEF_TYPE_IDS,
    "dribble_shot": frozenset({10, 11, 17}),
    "recovery_keeper": frozenset({5, 9}),
    "loss_foul": frozenset({13, 14, 15, 16}),
}


def events_since_opp_def_action(df: pd.DataFrame) -> np.ndarray:
    """Located events between the current event and the opponent's last defensive action ``[n]``.

    Read from the sequence block: the first slot ``k`` whose event is by the opponent
    (``seq_same_k == 0``) and of a defensive type (:data:`DEF_TYPE_IDS`) gives ``k - 1``.
    Rows with no such slot in the 20-slot history get :data:`SEQ_LEN` (meaning ">= 20 or
    none this period").
    """
    n = len(df)
    out = np.full(n, SEQ_LEN, dtype=np.int16)
    found = np.zeros(n, dtype=bool)
    for k in range(SEQ_LEN, 0, -1):
        t = df[f"seq_type_{k:02d}"].to_numpy(dtype=np.int64)
        same = df[f"seq_same_{k:02d}"].to_numpy(dtype=np.int64)
        hit = (same == 0) & np.isin(t, list(DEF_TYPE_IDS))
        out[hit] = k - 1
        found |= hit
    out[~found] = SEQ_LEN
    return out


def seq_mlp_features(df: pd.DataFrame) -> np.ndarray:
    """Dense per-slot encoding of the sequence block for the MLP ``[n, SEQ_LEN * 12]``.

    Per slot: pad flag, 7 coarse type-group one-hots, x / 120, y / 80, log1p(dt), same-team
    flag (pads and opponent both 0). Pad slots are all-zero except the pad flag.
    """
    n = len(df)
    groups = list(SEQ_TYPE_GROUPS)
    blocks = []
    for k in range(1, SEQ_LEN + 1):
        t = df[f"seq_type_{k:02d}"].to_numpy(dtype=np.int64)
        pad = (t == 0).astype(np.float32)
        cols = [pad]
        for g in groups:
            cols.append(np.isin(t, list(SEQ_TYPE_GROUPS[g])).astype(np.float32))
        x = np.nan_to_num(df[f"seq_x_{k:02d}"].to_numpy(dtype=np.float32) / 120.0)
        y = np.nan_to_num(df[f"seq_y_{k:02d}"].to_numpy(dtype=np.float32) / 80.0)
        dt = np.log1p(
            np.nan_to_num(np.clip(df[f"seq_dt_{k:02d}"].to_numpy(dtype=np.float32), 0, None))
        )
        same = (df[f"seq_same_{k:02d}"].to_numpy(dtype=np.int64) == 1).astype(np.float32)
        cols += [x, y, dt.astype(np.float32), same]
        blocks.append(np.stack(cols, axis=1))
    return np.concatenate(blocks, axis=1) if n else np.zeros((0, SEQ_LEN * 12), dtype=np.float32)


def one_hot_design(
    x: pd.DataFrame, categorical: list[str], vocab_sizes: dict[str, int]
) -> np.ndarray:
    """Dense MLP design from a :func:`build_design` frame ``[n, d_num * 2 + sum(vocab sizes + 1)]``.

    Numeric columns become ``(value with NaN -> 0, missing indicator)``; each categorical
    column becomes ``vocab_size + 1`` one-hots (last = unseen; NaN = all zero).
    """
    parts: list[np.ndarray] = []
    for c in x.columns:
        v = x[c].to_numpy(dtype=np.float32)
        if c in categorical:
            size = vocab_sizes[c] + 1
            code = np.where(np.isnan(v), -1, v).astype(np.int64)
            code = np.clip(code, -1, size - 1)
            oh = np.zeros((len(v), size), dtype=np.float32)
            ok = code >= 0
            oh[np.arange(len(v))[ok], code[ok]] = 1.0
            parts.append(oh)
        else:
            miss = np.isnan(v).astype(np.float32)
            parts.append(np.stack([np.nan_to_num(v), miss], axis=1))
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(x), 0), dtype=np.float32)


def vocab_sizes() -> dict[str, int]:
    """Number of codes of every categorical design column (excluding the unseen code)."""
    sizes = {c: len(v) for c, v in VOCABS.items()}
    sizes["f_type_id"] = N_TYPE_IDS
    for k in range(1, SEQ_LEN + 1):
        sizes[f"seq_type_{k:02d}"] = N_TYPE_IDS
    return sizes


# ---------------------------------------------------------------------------
# Baselines and agreement metrics
# ---------------------------------------------------------------------------


def bucket_mean_baseline(
    bucket_train: np.ndarray, y_train: np.ndarray, bucket_test: np.ndarray, min_n: int = 20
) -> np.ndarray:
    """Per-bucket training mean for every test row ``[n_test]`` (global mean below ``min_n``
    rows)."""
    yt = np.asarray(y_train, dtype=float)
    ok = ~np.isnan(yt)
    g = float(yt[ok].mean()) if ok.any() else float("nan")
    s = pd.Series(yt[ok]).groupby(np.asarray(bucket_train, dtype=object)[ok]).agg(["mean", "size"])
    means = {k: float(r["mean"]) for k, r in s.iterrows() if r["size"] >= min_n}
    return np.asarray([means.get(b, g) for b in np.asarray(bucket_test, dtype=object)], dtype=float)


def rounded_agreement(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """``(exact, within_1)`` shares after rounding a count prediction."""
    y = np.asarray(y, dtype=float)
    r = np.rint(np.asarray(yhat, dtype=float))
    return float(np.mean(np.abs(r - y) <= 1e-6)), float(np.mean(np.abs(r - y) <= 1 + 1e-6))


def pitch_third(x: np.ndarray) -> np.ndarray:
    """``own`` / ``middle`` / ``final`` third of the event team's attacking frame ``[n]``."""
    x = np.asarray(x, dtype=float)
    out = np.where(x < 40.0, "own", np.where(x < 80.0, "middle", "final")).astype(object)
    out[np.isnan(x)] = "unknown"
    return out


SET_PIECE_PATTERNS = frozenset(
    {"From Corner", "From Free Kick", "From Throw In", "From Goal Kick", "From Kick Off"}
)


def phase_label(play_pattern: np.ndarray) -> np.ndarray:
    """``set_piece`` (possession started from a dead ball) or ``open_play`` ``[n]``."""
    p = np.asarray(play_pattern, dtype=object)
    return np.where(np.isin(p, list(SET_PIECE_PATTERNS)), "set_piece", "open_play").astype(object)
