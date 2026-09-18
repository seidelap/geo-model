"""Apply the saved NFL imputation students to any nflfastR play-by-play frame (stage NFL 03).

The students are LightGBM models trained by ``imputation.py`` on the 91 tracked 2017 games
and saved per feature set under ``processed_dir('nfl')/models/students_<fset>.joblib``.
``impute()`` rebuilds the design matrix with
:func:`research.privileged_tracking.nfl.imputation_features.build_design` and adds one
``imp_<target>__<fset>`` column per student. Pre-snap students are applied to nflfastR
scrimmage plays (``pass``, ``run``, ``qb_kneel``, ``qb_spike``), pass-play students to
``play_type == "pass"`` and run-play students to ``play_type == "run"``; everything else is NaN.

Team tendencies are rebuilt from the frame passed in (strictly earlier weeks of the same
season), so pass a whole season, not a slice. The team target encodings come from the
tracked 2017 games (stored in the bundle) and are therefore a season stale when applied to
another season; teams unseen in 2017 get the 2017 global mean.

Command line::

    python -m research.privileged_tracking.nfl.apply_student --season 2018 --feature-sets F0,F0P,F1

writes ``processed_dir('nfl')/imputed_pbp_<season>.parquet`` and two report tables
(``nfl_03_shift_<season>.parquet``: imputed distribution vs the 2017 out-of-fold imputations
and the 2017 truth; ``nfl_03_external_<season>.parquet``: agreement with the official NGS
fields that exist in that season's ``pbp_participation``).
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
from sklearn.metrics import roc_auc_score

from research.privileged_tracking.common.io import nfl_dir, processed_dir, reports_dir
from research.privileged_tracking.common.metrics import brier, log_loss, mae, r2
from research.privileged_tracking.nfl import imputation_features as imf

MODELS_SUBDIR = "models"
BUNDLE_VERSION = 1

#: nflfastR columns read for an application season (situation, tendencies, post-play fields, keys).
PBP_LOAD_COLS = ["game_id", "old_game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
                 "posteam_type", "home_team", "away_team", "receiver_player_id"] + imf.PBP_INPUT_COLS
PART_LOAD_COLS = ["old_game_id", "play_id", "offense_formation", "offense_personnel", "defense_personnel",
                  "defenders_in_box", "number_of_pass_rushers", "time_to_throw", "was_pressure", "ngs_air_yards"]
PART_RENAME = {"time_to_throw": "ngs_time_to_throw", "was_pressure": "ngs_was_pressure"}


# ---------------------------------------------------------------------------
# Saved students
# ---------------------------------------------------------------------------

@dataclass
class Student:
    """One saved LightGBM student.

    Attributes:
        target: target name (:data:`imputation_features.TARGETS`).
        kind / subset: copied from the :class:`TargetSpec`.
        model_str: LightGBM model text.
        best_iter: rounds used.
        te_off / te_def: team -> shrunk target mean over the tracked games (posteam / defteam).
        te_global: global mean the encodings shrink towards (used for unseen teams).
        train_n: labelled plays the final model was fitted on.
        y_mean / y_sd: target moments on the training plays (reference for shift tables).
        oof_mean / oof_sd: moments of the out-of-fold predictions on the tracked plays.
    """

    target: str
    kind: str
    subset: str
    model_str: str
    best_iter: int
    te_off: dict[str, float]
    te_def: dict[str, float]
    te_global: float
    train_n: int
    y_mean: float
    y_sd: float
    oof_mean: float
    oof_sd: float

    def booster(self) -> lgb.Booster:
        return lgb.Booster(model_str=self.model_str)


@dataclass
class StudentBundle:
    """All students of one feature set plus what is needed to rebuild their features."""

    feature_set: str
    features: list[str]
    students: dict[str, Student] = field(default_factory=dict)
    tendency_alpha: float = imf.TENDENCY_ALPHA
    te_alpha: float = 30.0
    version: int = BUNDLE_VERSION

    @staticmethod
    def path(feature_set: str, root: Path | None = None) -> Path:
        d = (root or processed_dir("nfl")) / MODELS_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d / f"students_{feature_set}.joblib"

    def save(self, root: Path | None = None) -> Path:
        p = self.path(self.feature_set, root)
        joblib.dump({"feature_set": self.feature_set, "features": list(self.features),
                     "students": {k: asdict(v) for k, v in self.students.items()},
                     "tendency_alpha": self.tendency_alpha, "te_alpha": self.te_alpha, "version": self.version}, p)
        return p

    @classmethod
    def load(cls, feature_set: str, root: Path | None = None) -> StudentBundle:
        d = joblib.load(cls.path(feature_set, root))
        return cls(feature_set=d["feature_set"], features=list(d["features"]),
                   students={k: Student(**v) for k, v in d["students"].items()},
                   tendency_alpha=float(d["tendency_alpha"]), te_alpha=float(d["te_alpha"]), version=int(d["version"]))

    def predict(self, X: pd.DataFrame, posteam: np.ndarray, defteam: np.ndarray,
                masks: dict[str, pd.Series]) -> pd.DataFrame:
        """Imputed values ``[n, n_students]`` (``imp_<target>__<fset>``), NaN outside each subset.

        Args:
            X: design matrix from :func:`imputation_features.build_design` ``[n, len(DESIGN_COLS)]``.
            posteam / defteam: nflfastR team codes per row ``[n]`` (for the team encodings).
            masks: :func:`imputation_features.subset_masks` of the same rows.
        """
        out = pd.DataFrame(index=X.index)
        pos = np.asarray(posteam, dtype=object)
        de = np.asarray(defteam, dtype=object)
        for name, st in self.students.items():
            m = masks[st.subset].to_numpy()
            col = np.full(len(X), np.nan)
            if m.any():
                Xs = X.loc[m, self.features].copy()
                if "te_off" in self.features:
                    Xs["te_off"] = [st.te_off.get(t, st.te_global) for t in pos[m]]
                    Xs["te_def"] = [st.te_def.get(t, st.te_global) for t in de[m]]
                col[m] = st.booster().predict(Xs, num_iteration=st.best_iter)
            out[f"imp_{name}__{self.feature_set}"] = col
        return out


# ---------------------------------------------------------------------------
# Preparing an nflfastR frame
# ---------------------------------------------------------------------------

def load_pbp_season(season: int, regular_only: bool = True) -> pd.DataFrame:
    """nflfastR play-by-play of one season with the columns the students need."""
    pbp = pd.read_parquet(nfl_dir() / "nflverse" / f"play_by_play_{season}.parquet", columns=PBP_LOAD_COLS)
    if regular_only:
        pbp = pbp[pbp["season_type"] == "REG"]
    return pbp.reset_index(drop=True)


def load_participation_season(season: int) -> pd.DataFrame | None:
    """nflverse participation charting of one season (``None`` when the file is missing)."""
    p = nfl_dir() / "nflverse" / f"pbp_participation_{season}.parquet"
    if not p.exists():
        return None
    part = pd.read_parquet(p)
    cols = [c for c in PART_LOAD_COLS if c in part.columns]
    part = part[cols].rename(columns=PART_RENAME)
    part["old_game_id"] = part["old_game_id"].astype(str)
    part["play_id"] = part["play_id"].astype(int)
    return part.drop_duplicates(["old_game_id", "play_id"])


def load_player_positions() -> pd.Series:
    """``gsis_id -> position`` from the nflverse players crosswalk."""
    pl = pd.read_parquet(nfl_dir() / "nflverse" / "players.parquet", columns=["gsis_id", "position"])
    pl = pl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    return pd.Series(pl["position"].to_numpy(), index=pl["gsis_id"].to_numpy())


def prepare_pbp(pbp: pd.DataFrame, participation: pd.DataFrame | None = None,
                positions: pd.Series | None = None, tendency_alpha: float = imf.TENDENCY_ALPHA) -> pd.DataFrame:
    """Add ``is_home``, the strictly-prior tendencies, participation charting and receiver position.

    Args:
        pbp: nflfastR frame (a whole season) with :data:`PBP_LOAD_COLS`.
        participation: optional ``pbp_participation`` frame (merged on ``old_game_id``, ``play_id``);
            without it the F0P / F1 / F2 personnel and official columns are NaN.
        positions: optional ``gsis_id -> position`` map for ``receiver_position``.

    Returns:
        a copy of ``pbp`` with the extra columns, same row order.
    """
    df = pbp.copy().reset_index(drop=True)
    if "is_home" not in df.columns:
        if "posteam_type" in df.columns:
            df["is_home"] = df["posteam_type"].astype(object).eq("home").astype(float).where(df["posteam_type"].notna())
        else:
            df["is_home"] = (df["posteam"] == df["home_team"]).astype(float)
    tend = imf.pbp_team_tendencies(df, alpha=tendency_alpha)
    for c in tend.columns:
        df[c] = tend[c].to_numpy()
    if participation is not None:
        keys = pd.DataFrame({"old_game_id": df["old_game_id"].astype(str), "play_id": df["play_id"].astype(int)})
        merged = keys.merge(participation, on=["old_game_id", "play_id"], how="left")
        for c in participation.columns:
            if c not in ("old_game_id", "play_id"):
                df[c] = merged[c].to_numpy()
    if positions is not None and "receiver_player_id" in df.columns:
        df["receiver_position"] = df["receiver_player_id"].map(positions).astype(object).where(df["receiver_player_id"].notna())
    return df


def impute(prepared: pd.DataFrame, feature_sets: tuple[str, ...] = ("F0", "F0P", "F1"),
           root: Path | None = None) -> pd.DataFrame:
    """Return ``prepared`` plus ``imp_<target>__<fset>`` columns for every saved student.

    Args:
        prepared: output of :func:`prepare_pbp`.
        feature_sets: bundles to apply (each must exist under ``root``/models).
    """
    X = imf.build_design(prepared)
    masks = imf.subset_masks(prepared)
    out = prepared.copy()
    for fset in feature_sets:
        bundle = StudentBundle.load(fset, root)
        pred = bundle.predict(X, prepared["posteam"].to_numpy(), prepared["defteam"].to_numpy(), masks)
        for c in pred.columns:
            out[c] = pred[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Season-level diagnostics
# ---------------------------------------------------------------------------

def _moments(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan}
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()), "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.5)), "p95": float(np.quantile(v, 0.95))}


def shift_table(imputed: pd.DataFrame, oof: pd.DataFrame, feature_sets: tuple[str, ...], season: int) -> pd.DataFrame:
    """Distribution of the imputed values in ``season`` vs the 2017 out-of-fold imputations and truth.

    One row per (target, feature set, source) with source in ``{"<season> imputed",
    "2017 oof imputed", "2017 tracked truth"}``.
    """
    rows = []
    for fset in feature_sets:
        for t in imf.TARGETS:
            c = f"imp_{t.name}__{fset}"
            if c not in imputed.columns:
                continue
            rows.append({"target": t.name, "feature_set": fset, "source": f"{season} imputed", **_moments(imputed[c].to_numpy())})
            oc = f"{t.name}__{fset}"
            if oc in oof.columns:
                rows.append({"target": t.name, "feature_set": fset, "source": "2017 oof imputed", **_moments(oof[oc].to_numpy())})
            if f"y_{t.name}" in oof.columns:
                rows.append({"target": t.name, "feature_set": fset, "source": "2017 tracked truth", **_moments(oof[f"y_{t.name}"].to_numpy())})
    return pd.DataFrame(rows)


def _agree(kind: str, y: np.ndarray, p: np.ndarray, sack: np.ndarray | None = None) -> dict[str, float]:
    """Agreement metrics on the rows where both ``y`` and ``p`` are present.

    ``share_sack`` is the share of those rows that are sacks (``sack`` = nflfastR flag per
    row, optional): it makes explicit which outcome classes a comparison covers, because
    the official ``time_to_throw`` / ``was_pressure`` fields are missing on every sack.
    """
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    extra: dict[str, float] = {}
    if sack is not None:
        s = np.asarray(sack, dtype=float)[ok]
        extra["share_sack"] = float(np.nanmean(s)) if len(s) and not np.all(np.isnan(s)) else np.nan
    if len(y) < 10:
        return {"n": int(len(y)), **extra}
    if kind == "binary":
        pc = np.clip(p, 0.0, 1.0)
        return {"n": int(len(y)), **extra, "auc": float(roc_auc_score(y, pc)) if len(np.unique(y)) > 1 else np.nan,
                "log_loss": log_loss(y, pc), "brier": brier(y, pc), "acc": float(np.mean((pc >= 0.5) == (y >= 0.5))),
                "base_rate": float(y.mean()), "mean_pred": float(pc.mean())}
    d: dict[str, float] = {"n": int(len(y)), **extra, "r2": r2(y, p), "mae": mae(y, p), "bias": float(np.mean(p - y)),
                           "corr": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan}
    if kind == "count":
        ex, w1 = imf.rounded_agreement(y, p)
        d.update({"exact_rounded": ex, "within_1": w1})
    return d


def external_table(imputed: pd.DataFrame, oof: pd.DataFrame, plays_tracked: pd.DataFrame,
                   feature_sets: tuple[str, ...], season: int) -> pd.DataFrame:
    """Agreement of imputed values with the official NGS fields of ``season``'s participation file.

    For every target with an ``oracle`` column present in ``imputed`` (defenders_in_box,
    number_of_pass_rushers, ngs_time_to_throw, ngs_was_pressure, ngs_air_yards) and every
    feature set: the imputed value vs the official field on the new season, next to the
    same comparison for the 2017 out-of-fold imputations and for the tracking-derived
    target itself (the ceiling: how well the derived target matches the official field).
    ``share_sack`` (when ``sack`` / ``pbp_sack`` are available) records which plays the
    comparison covers; it is 0 for the ``time_to_throw`` / ``pressure_derived`` rows because
    the official fields are absent on sacks.
    """
    oracle_2017 = {"defenders_in_box": "defendersInTheBox", "number_of_pass_rushers": "numberOfPassRushers",
                   "ngs_time_to_throw": "ngs_time_to_throw", "ngs_was_pressure": "ngs_was_pressure",
                   "ngs_air_yards": "ngs_air_yards"}
    tracked = plays_tracked.merge(oof, on=["gameId", "playId"], how="left")
    sack_new = pd.to_numeric(imputed["sack"], errors="coerce").to_numpy(dtype=float) if "sack" in imputed.columns else None
    sack_old = pd.to_numeric(tracked["pbp_sack"], errors="coerce").to_numpy(dtype=float) if "pbp_sack" in tracked.columns else None
    rows = []
    for t in imf.TARGETS:
        if t.oracle is None or t.oracle not in imputed.columns:
            continue
        y_new = pd.to_numeric(imputed[t.oracle], errors="coerce").to_numpy(dtype=float)
        y_old = pd.to_numeric(tracked[oracle_2017[t.oracle]], errors="coerce").to_numpy(dtype=float)
        rows.append({"target": t.name, "official_field": t.oracle, "feature_set": "derived target (2017 ceiling)",
                     "season": 2017, **_agree(t.kind, y_old, tracked[f"y_{t.name}"].to_numpy(dtype=float), sack_old)})
        for fset in feature_sets:
            c = f"imp_{t.name}__{fset}"
            if c not in imputed.columns:
                continue
            rows.append({"target": t.name, "official_field": t.oracle, "feature_set": fset, "season": season,
                         **_agree(t.kind, y_new, imputed[c].to_numpy(dtype=float), sack_new)})
            oc = f"{t.name}__{fset}"
            if oc in tracked.columns:
                rows.append({"target": t.name, "official_field": t.oracle, "feature_set": fset, "season": 2017,
                             **_agree(t.kind, y_old, tracked[oc].to_numpy(dtype=float), sack_old)})
    return pd.DataFrame(rows)


def coverage_table(prepared: pd.DataFrame, season: int) -> pd.DataFrame:
    """Share of scrimmage plays with each feature-set input available in ``season``."""
    masks = imf.subset_masks(prepared)
    scrim = masks["all"].to_numpy()
    rows = [{"season": season, "quantity": "scrimmage plays (pass, run, kneel, spike)", "value": float(scrim.sum())}]
    for c in ("offense_formation", "offense_personnel", "defense_personnel", "defenders_in_box",
              "number_of_pass_rushers", "ngs_time_to_throw", "ngs_was_pressure", "ngs_air_yards", "receiver_position"):
        if c in prepared.columns:
            rows.append({"season": season, "quantity": f"{c} present (share of scrimmage plays)",
                         "value": float(prepared.loc[scrim, c].notna().mean())})
    # the pass-play NGS fields by sack: they are absent on sacks, which bounds the external check
    is_pass = masks["pass"].to_numpy()
    sack = pd.to_numeric(prepared["sack"], errors="coerce").to_numpy(dtype=float) if "sack" in prepared.columns else None
    if sack is not None:
        for c in ("ngs_time_to_throw", "ngs_was_pressure"):
            if c not in prepared.columns:
                continue
            for flag, label in ((0.0, "non-sack"), (1.0, "sack")):
                sel = is_pass & (sack == flag)
                rows.append({"season": season, "quantity": f"{c} present (share of {label} pass plays)",
                             "value": float(prepared.loc[sel, c].notna().mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows)


def run_season(season: int, feature_sets: tuple[str, ...], root: Path | None = None,
               write: bool = True) -> dict[str, Any]:
    """Impute one season and write the parquet outputs plus the report tables."""
    t0 = time.time()
    pbp = load_pbp_season(season)
    part = load_participation_season(season)
    prepared = prepare_pbp(pbp, part, load_player_positions())
    imputed = impute(prepared, feature_sets, root)
    proc = root or processed_dir("nfl")
    oof = pd.read_parquet(proc / "imputed_oof.parquet")
    plays = pd.read_parquet(proc / "plays_tracked.parquet",
                            columns=["gameId", "playId", "defendersInTheBox", "numberOfPassRushers", "ngs_time_to_throw",
                                     "ngs_was_pressure", "ngs_air_yards", "pbp_sack"])
    shift = shift_table(imputed, oof, feature_sets, season)
    external = external_table(imputed, oof, plays, feature_sets, season)
    coverage = coverage_table(prepared, season)
    keep = ["game_id", "old_game_id", "play_id", "season", "week", "posteam", "defteam", "play_type"]
    out = imputed[keep + [c for c in imputed.columns if c.startswith("imp_")]]
    if write:
        out.to_parquet(proc / f"imputed_pbp_{season}.parquet", index=False)
        rdir = reports_dir()
        shift.to_parquet(rdir / f"nfl_03_shift_{season}.parquet", index=False)
        external.to_parquet(rdir / f"nfl_03_external_{season}.parquet", index=False)
        coverage.to_parquet(rdir / f"nfl_03_coverage_{season}.parquet", index=False)
    return {"imputed": out, "shift": shift, "external": external, "coverage": coverage,
            "seconds": time.time() - t0, "n_rows": int(len(out))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2018)
    ap.add_argument("--feature-sets", default="F0,F0P,F1")
    args = ap.parse_args()
    fsets = tuple(s.strip() for s in args.feature_sets.split(",") if s.strip())
    res = run_season(args.season, fsets)
    print(f"imputed {res['n_rows']} plays of {args.season} with {fsets} in {res['seconds']:.1f} s")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
        print(res["coverage"])
        print(res["external"])


if __name__ == "__main__":
    main()
