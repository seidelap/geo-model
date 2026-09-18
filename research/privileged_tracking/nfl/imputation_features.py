"""Pure feature construction for the NFL imputation students (stage NFL 03).

Everything here is a deterministic function of plain DataFrames / arrays so it can be
unit-tested on synthetic input. The driver (``imputation.py``) and the applier
(``apply_student.py``) both build their design matrices through :func:`build_design`, so a
student trained on the 2017 tracked plays sees exactly the same columns when it is applied
to an arbitrary nflfastR season.

Feature sets are nested and named after the information they assume available for a play
in ordinary play-by-play data (:data:`FEATURE_SETS`):

* ``F0n``  pre-snap situation (down, distance, yard line, clock, score, win probability,
           description-derived shotgun / no-huddle flags, home / away, the nflfastR pre-snap
           models ``ep`` / ``xpass``) plus team tendencies computed from strictly earlier
           weeks of play-by-play only (:func:`pbp_team_tendencies`). No tracking-derived
           quantity enters.
* ``F0``   F0n + team "target encodings" (``te_off`` / ``te_def``): the posteam's / defteam's
           mean of the target over the *training-fold* games only, leave-one-game-out inside
           the training fold (:class:`TeamEncoder`).
* ``F0P``  F0 + personnel groupings and the formation string (NGS charting that nflverse
           publishes in ``pbp_participation``; taken from plays.csv for the tracked plays).
* ``F1``   F0P + post-play play-by-play fields (play type, pass length / location, air yards,
           completion, sack, QB hit, run location / gap, yards gained, EPA, receiver
           position): the "impute state for after-the-fact analytics" setting.
* ``F2``   F1 + the official NGS ``defenders_in_box`` / ``number_of_pass_rushers``, as a
           reference for how much the official privileged fields add.

Input column contract of :func:`build_design` (nflfastR names; NaN allowed everywhere):
``down, ydstogo, yardline_100, qtr, half_seconds_remaining, game_seconds_remaining,
score_differential, wp, ep, xpass, goal_to_go, shotgun, no_huddle, is_home, tend_*,
offense_formation, offense_personnel, defense_personnel, play_type, pass_length,
pass_location, air_yards, yards_after_catch, complete_pass, interception, sack, qb_hit,
run_location, run_gap, yards_gained, epa, qb_dropback, qb_scramble, receiver_position,
defenders_in_box, number_of_pass_rushers``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import participation_features as pf
from research.privileged_tracking.nfl.tracking_features import parse_personnel

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

#: ``min_def_dist_qb_throw <= PRESSURE_DIST_YD`` is the tracking-derived pressure flag. The
#: threshold is selected on weeks 1-3 against NGS ``was_pressure`` by the driver (grid in the
#: report); 2.0 yd agrees with the charted flag on ~96% of pass plays.
PRESSURE_DIST_YD = 2.0


@dataclass(frozen=True)
class TargetSpec:
    """One imputation target.

    Attributes:
        name: column name of the target in the design / output tables.
        kind: ``"reg"`` (continuous), ``"count"`` (integer-valued, fitted as regression and
            additionally scored on rounded exact / within-1 agreement) or ``"binary"``.
        subset: ``"all"`` (every tracked play), ``"pass"`` (``is_pass_play``) or ``"run"``
            (nflfastR ``play_type == "run"`` that is not a BDB pass play).
        source: column of ``plays_tracked.parquet`` the target is read from (``name`` when
            it is not derived).
        description: one-line definition for the report.
        oracle: official NGS charting column that measures the same quantity (reference
            row in the ranking table), or ``None``.
    """

    name: str
    kind: str
    subset: str
    description: str
    source: str | None = None
    oracle: str | None = None

    @property
    def source_col(self) -> str:
        return self.source or self.name


TARGETS: tuple[TargetSpec, ...] = (
    # pre-snap defensive structure
    TargetSpec("n_deep_safeties", "count", "all", "defenders >= 10 yd deep at the snap"),
    TargetSpec("mof_open", "binary", "all", "two or more deep safeties (middle of the field open)"),
    TargetSpec("box_count_tuned", "count", "all", "defenders in the box (depth <= 6, |lateral| <= 7)",
               oracle="defenders_in_box"),
    TargetSpec("n_dl", "count", "all", "defenders within 1.5 yd of the line (down linemen + edge)"),
    TargetSpec("def_y_std", "reg", "all", "lateral spread (sd of y) of the defense at the snap"),
    TargetSpec("cb_cushion", "reg", "all", "mean cushion of the widest receiver on each side (yd)"),
    # pre-snap offensive structure
    TargetSpec("n_wide_left", "count", "all", "offensive players split >= 8 yd to the offense's left"),
    TargetSpec("n_wide_right", "count", "all", "offensive players split >= 8 yd to the offense's right"),
    TargetSpec("n_backfield", "count", "all", "non-QB players >= 2.5 yd behind the line within 6 yd"),
    TargetSpec("qb_depth", "reg", "all", "QB depth behind the line at the snap (yd)"),
    TargetSpec("shotgun_derived", "binary", "all", "qb_depth >= 4 (tracking shotgun)"),
    TargetSpec("motion_derived", "binary", "all", "pre-snap motion (> 3 yd lateral range in the 2 s before the snap, or the tag)"),
    # within-play, pass plays
    TargetSpec("time_to_throw", "reg", "pass", "seconds from snap to release (sack for sacks)",
               oracle="ngs_time_to_throw"),
    TargetSpec("pressure_derived", "binary", "pass",
               f"closest defender to the QB at release <= {PRESSURE_DIST_YD} yd", source="min_def_dist_qb_throw",
               oracle="ngs_was_pressure"),
    TargetSpec("min_def_dist_qb_throw", "reg", "pass", "closest defender to the QB at release (yd)"),
    TargetSpec("n_pass_rushers_derived", "count", "pass", "defenders reaching the line within 1.5 s",
               oracle="number_of_pass_rushers"),
    TargetSpec("separation_at_arrival", "reg", "pass", "target's distance to the nearest defender at arrival (yd)"),
    TargetSpec("n_def_within_r_target", "count", "pass", "defenders within 5 yd of the target at arrival"),
    TargetSpec("target_depth", "reg", "pass", "target x - LOS at arrival (yd)", oracle="ngs_air_yards"),
    # within-play, run plays
    TargetSpec("box_count_run", "count", "run", "defenders in the box on run plays", source="box_count_tuned",
               oracle="defenders_in_box"),
    TargetSpec("min_def_dist_carrier_handoff", "reg", "run", "closest defender to the carrier at the handoff (yd)"),
    TargetSpec("yards_to_first_contact", "reg", "run", "carrier x - LOS at first contact (yd)"),
    TargetSpec("n_def_within_r_carrier_first_contact", "count", "run", "defenders within 3 yd of the carrier at first contact"),
)

TARGET_BY_NAME: dict[str, TargetSpec] = {t.name: t for t in TARGETS}


def target_frame(plays: pd.DataFrame) -> pd.DataFrame:
    """Target columns ``[n, len(TARGETS)]`` from a ``plays_tracked`` frame (NaN = not applicable).

    Subset masks: ``all`` every play; ``pass`` = ``is_pass_play``; ``run`` = nflfastR
    ``play_type == "run"`` and not a BDB pass play. ``pressure_derived`` is
    ``min_def_dist_qb_throw <= PRESSURE_DIST_YD``.
    """
    masks = subset_masks(plays)
    out = pd.DataFrame(index=plays.index)
    for t in TARGETS:
        v = pd.to_numeric(plays[t.source_col], errors="coerce").astype(float)
        if t.name == "pressure_derived":
            v = (v <= PRESSURE_DIST_YD).astype(float).where(v.notna())
        out[t.name] = v.where(masks[t.subset])
    return out


SCRIMMAGE_PLAY_TYPES = ("pass", "run", "qb_kneel", "qb_spike")


def subset_masks(plays: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for the ``all`` / ``pass`` / ``run`` subsets of a plays frame.

    For the tracked table (``is_pass_play`` + ``pbp_play_type``) ``all`` is every row (the
    table holds non-special-teams plays only). For a plain nflfastR frame (``play_type``
    only) ``all`` is the scrimmage plays (:data:`SCRIMMAGE_PLAY_TYPES`, the population the
    students were trained on), ``pass`` = ``play_type == "pass"``.
    """
    if "is_pass_play" in plays.columns:
        is_pass = plays["is_pass_play"].fillna(False).astype(bool)
        all_ = pd.Series(True, index=plays.index)
    else:
        is_pass = plays["play_type"].astype(object).eq("pass")
        all_ = plays["play_type"].astype(object).isin(SCRIMMAGE_PLAY_TYPES)
    pt = plays["pbp_play_type"] if "pbp_play_type" in plays.columns else plays["play_type"]
    is_run = pt.astype(object).eq("run") & ~is_pass
    return {"all": all_, "pass": is_pass, "run": is_run}


# ---------------------------------------------------------------------------
# Categorical vocabularies (fixed label sets, not learned from the targets)
# ---------------------------------------------------------------------------

FORMATIONS = ["SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY", "PISTOL", "JUMBO", "WILDCAT", "ACE"]
PLAY_TYPES = ["pass", "run", "qb_kneel", "qb_spike", "no_play"]
PASS_LENGTHS = ["short", "deep"]
LOCATIONS = ["left", "middle", "right"]
RUN_GAPS = ["end", "tackle", "guard"]
RECEIVER_GROUPS = ["WR", "TE", "RB", "QB", "OL", "other"]
DD_BUCKETS = [f"d{d}_{b}" for d in (1, 2, 3, 4) for b in ("short", "mid", "long")] + ["dna_na", "dna_long", "dna_mid", "dna_short"]

VOCABS: dict[str, list[str]] = {
    "dd_bucket": DD_BUCKETS, "formation": FORMATIONS, "play_type": PLAY_TYPES, "pass_length": PASS_LENGTHS,
    "pass_location": LOCATIONS, "run_location": LOCATIONS, "run_gap": RUN_GAPS, "receiver_group": RECEIVER_GROUPS,
}


def encode_category(values: pd.Series | np.ndarray, vocab: list[str]) -> np.ndarray:
    """Integer codes ``[n]`` (float, NaN for missing / unknown labels) for a fixed vocabulary."""
    index = {v: float(i) for i, v in enumerate(vocab)}
    arr = pd.Series(values).astype(object).to_numpy()
    out = np.full(len(arr), np.nan, dtype=float)
    for i, v in enumerate(arr):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[i] = index.get(str(v), np.nan)
    return out


# ---------------------------------------------------------------------------
# Team tendencies from strictly earlier weeks of play-by-play (event-only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TendencySpec:
    """A running team rate: mean of ``indicator`` over the entity's strictly earlier games.

    Attributes:
        name: output column ``tend_<name>``.
        entity: ``"posteam"`` or ``"defteam"``.
        indicator: per-play indicator column produced by :func:`play_indicators`.
        prior: target-free constant the running mean is shrunk towards (``alpha`` plays); it
            only matters before the entity has played (week 1).
    """

    name: str
    entity: str
    indicator: str
    prior: float


#: Priors are league-typical values written down a priori (not computed from any frame).
TENDENCIES: tuple[TendencySpec, ...] = (
    TendencySpec("pos_shotgun", "posteam", "ind_shotgun", 0.55),
    TendencySpec("pos_no_huddle", "posteam", "ind_no_huddle", 0.08),
    TendencySpec("pos_pass", "posteam", "ind_pass", 0.58),
    TendencySpec("pos_deep", "posteam", "ind_deep", 0.18),
    TendencySpec("pos_air_yards", "posteam", "ind_air_yards", 8.0),
    TendencySpec("pos_sack", "posteam", "ind_sack", 0.065),
    TendencySpec("pos_qb_hit", "posteam", "ind_qb_hit", 0.12),
    TendencySpec("pos_rush_yards", "posteam", "ind_rush_yards", 4.2),
    TendencySpec("def_pass", "defteam", "ind_pass", 0.58),
    TendencySpec("def_deep", "defteam", "ind_deep", 0.18),
    TendencySpec("def_air_yards", "defteam", "ind_air_yards", 8.0),
    TendencySpec("def_sack", "defteam", "ind_sack", 0.065),
    TendencySpec("def_qb_hit", "defteam", "ind_qb_hit", 0.12),
    TendencySpec("def_rush_yards", "defteam", "ind_rush_yards", 4.2),
)
TENDENCY_COLS = [f"tend_{t.name}" for t in TENDENCIES]
TENDENCY_ALPHA = 50.0  # plays


def play_indicators(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-play indicators (NaN where not applicable) feeding the running tendencies.

    Requires ``play_type, shotgun, no_huddle, pass_length, air_yards, sack, qb_hit,
    yards_gained``. Scrimmage plays are ``play_type in {pass, run}``; pass-play indicators
    are NaN on runs and vice versa, so each rate has its natural denominator.
    """
    pt = pbp["play_type"].astype(object)
    is_pass = pt.eq("pass").to_numpy()
    is_run = pt.eq("run").to_numpy()
    scrim = is_pass | is_run
    num = {c: pd.to_numeric(pbp[c], errors="coerce").to_numpy(dtype=float)
           for c in ("shotgun", "no_huddle", "air_yards", "sack", "qb_hit", "yards_gained")}
    out = pd.DataFrame(index=pbp.index)
    out["ind_shotgun"] = np.where(scrim, num["shotgun"], np.nan)
    out["ind_no_huddle"] = np.where(scrim, num["no_huddle"], np.nan)
    out["ind_pass"] = np.where(scrim, is_pass.astype(float), np.nan)
    deep = pbp["pass_length"].astype(object).eq("deep").to_numpy().astype(float)
    has_len = pbp["pass_length"].notna().to_numpy()
    out["ind_deep"] = np.where(is_pass & has_len, deep, np.nan)
    out["ind_air_yards"] = np.where(is_pass, num["air_yards"], np.nan)
    out["ind_sack"] = np.where(is_pass, num["sack"], np.nan)
    out["ind_qb_hit"] = np.where(is_pass, num["qb_hit"], np.nan)
    out["ind_rush_yards"] = np.where(is_run, num["yards_gained"], np.nan)
    return out


def pbp_team_tendencies(pbp: pd.DataFrame, alpha: float = TENDENCY_ALPHA) -> pd.DataFrame:
    """Strictly-prior-week team tendencies ``tend_*`` for every row of a season's pbp frame.

    Every play of a game sees the entity's cumulative indicator sums over games of a
    smaller ``week`` in the same ``season`` and nothing from its own game
    (:func:`participation_features.prior_game_value_sums`), shrunk towards the spec's
    constant prior with ``alpha`` pseudo-plays. Requires ``game_id, season, week, posteam,
    defteam`` plus the columns of :func:`play_indicators`. Returns ``[n, len(TENDENCIES)]``
    aligned to ``pbp`` (index preserved).
    """
    ind = play_indicators(pbp)
    tmp = pd.concat([pbp[["game_id", "season", "week", "posteam", "defteam"]].reset_index(drop=True),
                     ind.reset_index(drop=True)], axis=1)
    out = pd.DataFrame(index=pbp.index)
    for t in TENDENCIES:
        total, count = pf.prior_game_value_sums(tmp, [t.entity], t.indicator, game_col="game_id",
                                                season_col="season", week_col="week")
        out[f"tend_{t.name}"] = pf.shrink_mean(total, count, np.full(len(tmp), t.prior), alpha)
    return out


# ---------------------------------------------------------------------------
# Team target encodings from training-fold games only
# ---------------------------------------------------------------------------

@dataclass
class TeamEncoder:
    """Shrunk team mean of a target, fitted on training-fold plays only.

    ``transform_train`` returns leave-one-game-out means for the plays the encoder was
    fitted on (a play never sees its own game); ``transform`` returns the full training
    mean for new games. Teams unseen in training get the training-fold global mean.

    Attributes:
        alpha: pseudo-plays of shrinkage towards the global training mean.
    """

    alpha: float = 30.0
    global_mean: float = float("nan")
    team_sum: dict[str, float] = field(default_factory=dict)
    team_count: dict[str, float] = field(default_factory=dict)
    game_sum: dict[tuple[str, object], float] = field(default_factory=dict)
    game_count: dict[tuple[str, object], float] = field(default_factory=dict)

    def fit(self, team: np.ndarray, game: np.ndarray, y: np.ndarray) -> TeamEncoder:
        """Accumulate sums / counts over rows with a non-NaN target.

        Args:
            team: entity id per play ``[n]``.
            game: game id per play ``[n]``.
            y: target ``[n]`` (NaN skipped).
        """
        y = np.asarray(y, dtype=float)
        ok = ~np.isnan(y)
        self.global_mean = float(y[ok].mean()) if ok.any() else float("nan")
        df = pd.DataFrame({"team": np.asarray(team, dtype=object)[ok], "game": np.asarray(game, dtype=object)[ok], "y": y[ok]})
        g = df.groupby(["team", "game"], sort=False)["y"].agg(["sum", "count"])
        self.game_sum = {k: float(v) for k, v in g["sum"].items()}
        self.game_count = {k: float(v) for k, v in g["count"].items()}
        t = df.groupby("team", sort=False)["y"].agg(["sum", "count"])
        self.team_sum = {k: float(v) for k, v in t["sum"].items()}
        self.team_count = {k: float(v) for k, v in t["count"].items()}
        return self

    def _encode(self, team: np.ndarray, game: np.ndarray | None) -> np.ndarray:
        out = np.empty(len(team), dtype=float)
        for i, tm in enumerate(np.asarray(team, dtype=object)):
            s = self.team_sum.get(tm, 0.0)
            c = self.team_count.get(tm, 0.0)
            if game is not None:
                key = (tm, game[i])
                s -= self.game_sum.get(key, 0.0)
                c -= self.game_count.get(key, 0.0)
            out[i] = (s + self.alpha * self.global_mean) / (c + self.alpha)
        return out

    def transform_train(self, team: np.ndarray, game: np.ndarray) -> np.ndarray:
        """Leave-one-game-out encodings ``[n]`` for training rows."""
        return self._encode(team, np.asarray(game, dtype=object))

    def transform(self, team: np.ndarray) -> np.ndarray:
        """Full training-fold encodings ``[n]`` for held-out / new rows."""
        return self._encode(team, None)

    def table(self) -> dict[str, float]:
        """Team -> shrunk mean over everything the encoder saw (for saved students)."""
        teams = sorted(self.team_sum)
        return dict(zip(teams, self.transform(np.array(teams, dtype=object)).tolist()))


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

SITUATION_COLS = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "game_seconds_remaining",
                  "score_differential", "wp", "ep", "xpass", "goal_to_go", "shotgun", "no_huddle", "is_home", "dd_bucket"]
TE_COLS = ["te_off", "te_def"]
PERSONNEL_COLS = ["formation", "pers_RB", "pers_TE", "pers_WR", "pers_OL", "pers_QB", "pers_DL", "pers_LB", "pers_DB"]
POSTPLAY_COLS = ["play_type_code", "is_pass", "is_run", "qb_dropback", "qb_scramble", "pass_length", "pass_location",
                 "air_yards", "yards_after_catch", "complete_pass", "interception", "sack", "qb_hit", "run_location",
                 "run_gap", "yards_gained", "epa", "receiver_group"]
OFFICIAL_COLS = ["defenders_in_box", "number_of_pass_rushers"]
CATEGORICAL_COLS = ["dd_bucket", "formation", "play_type_code", "pass_length", "pass_location", "run_location",
                    "run_gap", "receiver_group"]

FEATURE_SETS: dict[str, list[str]] = {
    "F0n": SITUATION_COLS + TENDENCY_COLS,
    "F0": SITUATION_COLS + TENDENCY_COLS + TE_COLS,
    "F0P": SITUATION_COLS + TENDENCY_COLS + TE_COLS + PERSONNEL_COLS,
    "F1": SITUATION_COLS + TENDENCY_COLS + TE_COLS + PERSONNEL_COLS + POSTPLAY_COLS,
    "F2": SITUATION_COLS + TENDENCY_COLS + TE_COLS + PERSONNEL_COLS + POSTPLAY_COLS + OFFICIAL_COLS,
}
FEATURE_SET_ORDER = ["F0n", "F0", "F0P", "F1", "F2"]
DESIGN_COLS = FEATURE_SETS["F2"]

#: nflfastR play-by-play columns a frame must carry before :func:`build_design`.
PBP_INPUT_COLS = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "game_seconds_remaining",
                  "score_differential", "wp", "ep", "xpass", "goal_to_go", "shotgun", "no_huddle", "play_type",
                  "pass_length", "pass_location", "air_yards", "yards_after_catch", "complete_pass", "interception",
                  "sack", "qb_hit", "run_location", "run_gap", "yards_gained", "epa", "qb_dropback", "qb_scramble"]


def personnel_features(offense_personnel: pd.Series, defense_personnel: pd.Series) -> pd.DataFrame:
    """Position-group counts ``pers_*`` parsed from NGS personnel strings (NaN when missing).

    Offense defaults follow the NGS convention (``OL`` 5 and ``QB`` 1 when omitted); a
    defensive group not mentioned counts 0.
    """
    off = [parse_personnel(s) for s in offense_personnel.astype(object).to_numpy()]
    de = [parse_personnel(s) for s in defense_personnel.astype(object).to_numpy()]
    out = pd.DataFrame(index=offense_personnel.index)
    for g in ("RB", "TE", "WR"):
        out[f"pers_{g}"] = [float(d.get(g, 0)) if d else np.nan for d in off]
    out["pers_OL"] = [float(d.get("OL", 5)) if d else np.nan for d in off]
    out["pers_QB"] = [float(d.get("QB", 1)) if d else np.nan for d in off]
    for g in ("DL", "LB", "DB"):
        out[f"pers_{g}"] = [float(d.get(g, 0)) if d else np.nan for d in de]
    return out


def receiver_group(position: pd.Series) -> np.ndarray:
    """Receiver position group label (``WR/TE/RB/QB/OL/other``, ``None`` when no receiver)."""
    out = np.full(len(position), None, dtype=object)
    for i, p in enumerate(position.astype(object).to_numpy()):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        g = pf.roster_group(str(p))
        out[i] = g if g in ("WR", "TE", "RB", "QB", "OL") else "other"
    return out


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    """Full design matrix ``[n, len(DESIGN_COLS)]`` (all feature sets) from a pbp-like frame.

    Missing input columns are allowed and give NaN features (so an F0-only frame still
    builds). ``te_off`` / ``te_def`` are copied through when present (the driver fills them
    per target and fold) and NaN otherwise. Categorical columns hold integer codes (float,
    NaN for missing) with the fixed vocabularies in :data:`VOCABS`.
    """
    n = len(df)
    idx = df.index

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=idx, dtype=float)

    def num(name: str) -> np.ndarray:
        return pd.to_numeric(col(name), errors="coerce").to_numpy(dtype=float)

    X = pd.DataFrame(index=idx)
    for c in ("down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "game_seconds_remaining",
              "score_differential", "wp", "ep", "xpass", "goal_to_go", "shotgun", "no_huddle", "is_home"):
        X[c] = num(c)
    X["dd_bucket"] = encode_category(pf.down_distance_bucket(X["down"].to_numpy(), X["ydstogo"].to_numpy()), DD_BUCKETS)
    for c in TENDENCY_COLS:
        X[c] = num(c)
    for c in TE_COLS:
        X[c] = num(c)
    X["formation"] = encode_category(col("offense_formation"), FORMATIONS)
    pers = personnel_features(col("offense_personnel"), col("defense_personnel"))
    for c in pers.columns:
        X[c] = pers[c].to_numpy()
    pt = col("play_type").astype(object)
    X["play_type_code"] = encode_category(pt, PLAY_TYPES)
    X["is_pass"] = pt.eq("pass").astype(float).where(pt.notna()).to_numpy(dtype=float)
    X["is_run"] = pt.eq("run").astype(float).where(pt.notna()).to_numpy(dtype=float)
    for c in ("qb_dropback", "qb_scramble", "air_yards", "yards_after_catch", "complete_pass", "interception",
              "sack", "qb_hit", "yards_gained", "epa"):
        X[c] = num(c)
    X["pass_length"] = encode_category(col("pass_length"), PASS_LENGTHS)
    X["pass_location"] = encode_category(col("pass_location"), LOCATIONS)
    X["run_location"] = encode_category(col("run_location"), LOCATIONS)
    X["run_gap"] = encode_category(col("run_gap"), RUN_GAPS)
    X["receiver_group"] = encode_category(pd.Series(receiver_group(col("receiver_position")), index=idx), RECEIVER_GROUPS)
    for c in OFFICIAL_COLS:
        X[c] = num(c)
    assert len(X) == n
    return X[DESIGN_COLS]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def bucket_mean_baseline(bucket_train: np.ndarray, y_train: np.ndarray, bucket_test: np.ndarray,
                         min_count: int = 1) -> np.ndarray:
    """Per-bucket training mean applied to test rows (global training mean for unseen buckets).

    Args:
        bucket_train: bucket label per training row ``[n_train]`` (NaN target rows are skipped).
        y_train: targets ``[n_train]``.
        bucket_test: bucket label per test row ``[n_test]``.
        min_count: buckets with fewer labelled training rows fall back to the global mean.
    """
    y_train = np.asarray(y_train, dtype=float)
    ok = ~np.isnan(y_train)
    g = pd.Series(y_train[ok]).groupby(np.asarray(bucket_train, dtype=object)[ok]).agg(["mean", "count"])
    global_mean = float(y_train[ok].mean()) if ok.any() else float("nan")
    lookup = {k: float(r["mean"]) for k, r in g.iterrows() if r["count"] >= min_count}
    return np.array([lookup.get(b, global_mean) for b in np.asarray(bucket_test, dtype=object)], dtype=float)


def rounded_agreement(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """``(exact, within_1)`` shares after rounding a regression prediction to an integer."""
    y = np.asarray(y, dtype=float)
    r = np.rint(np.asarray(yhat, dtype=float))
    return float(np.mean(np.abs(r - y) <= 1e-6)), float(np.mean(np.abs(r - y) <= 1 + 1e-6))
