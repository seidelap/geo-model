"""Apply the saved soccer 02 students to any build-stage event-feature frame.

The students are LightGBM models trained by ``imputation.py`` on the 360 matches of
``events360.parquet`` and saved per feature set under
``processed_dir('soccer')/models/students_<fset>.joblib`` (:class:`StudentBundle`).
:func:`impute` rebuilds the design matrix with
:func:`research.privileged_tracking.soccer.imputation_features.build_design` and adds one
``imp_<target>__<fset>`` column per student, NaN outside the student's row subset
(possession-team events for the team-shape targets, settled possession / middle third for the
two binaries, Pass rows for the pass-lane targets, every row for the ball-relative ones).

Command line (repo root)::

    python -m research.privileged_tracking.soccer.apply_student --feature-sets E0,E2,E2a

reads ``processed_dir('soccer')/events_no360.parquet`` (shots + a fixed 25% of passes / carries
of the non-360 matches), writes ``imputed_no360.parquet`` (ids, a few context columns and the
``imp_*`` columns) and two report tables: ``soccer_02_shift_no360.parquet`` (distribution of
every imputed quantity in the 2015/16 leagues and the other non-360 competitions versus the
out-of-fold imputations and the 360 truth on the same event types) and
``soccer_02_shift_by_competition.parquet``. ``--input`` / ``--output`` accept any other
build-stage frame.
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

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.soccer import imputation_features as imf

MODELS_SUBDIR = "models"
BUNDLE_VERSION = 1
LEAGUES_1516 = frozenset({"La Liga", "Ligue 1", "Premier League", "Serie A"})
#: event types present in events_no360 (the 360 side of every shift table is restricted to them)
NO360_TYPES = ("Pass", "Carry", "Shot")
ID_COLS = [
    "match_id",
    "event_id",
    "event_index",
    "period",
    "team_id",
    "competition",
    "season",
    "gender",
    "match_date",
    "has_360",
    "f_type",
    "f_x",
    "f_y",
    "f_is_possession_team",
    "f_play_pattern",
]


@dataclass
class Student:
    """One saved LightGBM student.

    Attributes:
        target: target name (:data:`imputation_features.TARGETS`).
        kind / subset: copied from the :class:`imputation_features.TargetSpec`.
        model_str: LightGBM model text.
        n_rounds: boosting rounds of the final fit (median of the CV early-stopping rounds).
        train_n: labelled rows the final model was fitted on (after the match-stratified cap).
        y_mean / y_sd: target moments on the training rows (reference for shift tables).
        oof_mean / oof_sd: moments of the out-of-fold predictions on the 360 rows.
        cv_skill: out-of-fold skill (R2, or Brier skill score for binaries) of this feature set.
    """

    target: str
    kind: str
    subset: str
    model_str: str
    n_rounds: int
    train_n: int
    y_mean: float
    y_sd: float
    oof_mean: float
    oof_sd: float
    cv_skill: float

    def booster(self) -> lgb.Booster:
        return lgb.Booster(model_str=self.model_str)


@dataclass
class StudentBundle:
    """All students of one feature set plus what is needed to rebuild their design matrix."""

    feature_set: str
    features: list[str]
    categorical: list[str]
    students: dict[str, Student] = field(default_factory=dict)
    deep_block_threshold: float = float("nan")
    version: int = BUNDLE_VERSION

    @staticmethod
    def path(feature_set: str, root: Path | None = None) -> Path:
        d = (root or processed_dir("soccer")) / MODELS_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        return d / f"students_{feature_set}.joblib"

    def save(self, root: Path | None = None) -> Path:
        p = self.path(self.feature_set, root)
        joblib.dump(
            {
                "feature_set": self.feature_set,
                "features": list(self.features),
                "categorical": list(self.categorical),
                "students": {k: asdict(v) for k, v in self.students.items()},
                "deep_block_threshold": self.deep_block_threshold,
                "version": self.version,
            },
            p,
        )
        return p

    @classmethod
    def load(cls, feature_set: str, root: Path | None = None) -> StudentBundle:
        d = joblib.load(cls.path(feature_set, root))
        return cls(
            feature_set=d["feature_set"],
            features=list(d["features"]),
            categorical=list(d["categorical"]),
            students={k: Student(**v) for k, v in d["students"].items()},
            deep_block_threshold=float(d["deep_block_threshold"]),
            version=int(d["version"]),
        )

    def predict(self, design: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
        """Imputed values ``[n, n_students]`` (``imp_<target>__<fset>``), NaN outside each subset.

        Args:
            design: design matrix from :func:`imputation_features.build_design` for this feature set
                (extra columns are ignored, the bundle's ``features`` must all be present).
            masks: :func:`imputation_features.subset_masks` of the same rows.
        """
        out = pd.DataFrame(index=design.index)
        xf = design[self.features]
        for name, st in self.students.items():
            m = np.asarray(masks[st.subset], dtype=bool)
            col = np.full(len(design), np.nan, dtype=np.float32)
            if m.any():
                col[m] = st.booster().predict(xf.loc[m], num_iteration=st.n_rounds)
            out[f"imp_{name}__{self.feature_set}"] = col
        return out


def impute(
    df: pd.DataFrame, feature_sets: tuple[str, ...] = ("E0", "E2", "E2a"), root: Path | None = None
) -> pd.DataFrame:
    """Return ``df`` plus ``imp_<target>__<fset>`` columns for every saved student.

    Args:
        df: any build-stage event-feature frame (``events360.parquet`` / ``events_no360.parquet``
            columns; only ``f_*``, ``gender`` and ``competition`` are read).
        feature_sets: bundles to apply (each must exist under ``root``/models).
    """
    masks = imf.subset_masks(df)
    out = df.copy()
    for fset in feature_sets:
        bundle = StudentBundle.load(fset, root)
        design = imf.build_design(df, fset)
        pred = bundle.predict(design, masks)
        for c in pred.columns:
            out[c] = pred[c].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Distribution shift diagnostics
# ---------------------------------------------------------------------------


def _moments(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "sd": float(v.std()),
        "p05": float(np.quantile(v, 0.05)),
        "p50": float(np.quantile(v, 0.5)),
        "p95": float(np.quantile(v, 0.95)),
    }


def domain_label(competition: pd.Series, season: pd.Series) -> np.ndarray:
    """``leagues_2015/16`` for the four full 2015/16 seasons, ``other_no360`` otherwise ``[n]``."""
    c = competition.to_numpy(dtype=object)
    s = season.to_numpy(dtype=object)
    return np.where(
        np.isin(c, list(LEAGUES_1516)) & (s == "2015/2016"), "leagues_2015/16", "other_no360"
    ).astype(object)


def shift_table(
    imputed: pd.DataFrame, oof: pd.DataFrame | None, feature_sets: tuple[str, ...]
) -> pd.DataFrame:
    """Distribution of every imputed quantity in the new domain vs the 360 domain.

    One row per (target, feature set, source, event type) with source in
    ``{"no360 leagues 2015/16 imputed", "no360 other competitions imputed", "360 oof imputed",
    "360 truth"}``; the 360 rows are restricted to :data:`NO360_TYPES` so that both sides
    contain the same event types, and ``event_type`` is ``all`` or one of those types.
    """
    rows: list[dict[str, Any]] = []
    dom = domain_label(imputed["competition"], imputed["season"])
    ftype_new = imputed["f_type"].to_numpy(dtype=object)
    if oof is not None:
        keep = np.isin(oof["f_type"].to_numpy(dtype=object), list(NO360_TYPES))
        oof = oof.loc[keep]
        ftype_old = oof["f_type"].to_numpy(dtype=object)
    for fset in feature_sets:
        for t in imf.TARGETS:
            c = f"imp_{t.name}__{fset}"
            if c not in imputed.columns:
                continue
            for et in ("all", *NO360_TYPES):
                sel_new = np.ones(len(imputed), dtype=bool) if et == "all" else ftype_new == et
                for d, label in (
                    ("leagues_2015/16", "no360 leagues 2015/16 imputed"),
                    ("other_no360", "no360 other competitions imputed"),
                ):
                    rows.append(
                        {
                            "target": t.name,
                            "feature_set": fset,
                            "source": label,
                            "event_type": et,
                            **_moments(imputed.loc[sel_new & (dom == d), c].to_numpy()),
                        }
                    )
                if oof is not None:
                    sel_old = np.ones(len(oof), dtype=bool) if et == "all" else ftype_old == et
                    oc = f"{t.name}__{fset}"
                    if oc in oof.columns:
                        rows.append(
                            {
                                "target": t.name,
                                "feature_set": fset,
                                "source": "360 oof imputed",
                                "event_type": et,
                                **_moments(oof.loc[sel_old, oc].to_numpy()),
                            }
                        )
                    if f"y_{t.name}" in oof.columns:
                        rows.append(
                            {
                                "target": t.name,
                                "feature_set": fset,
                                "source": "360 truth",
                                "event_type": et,
                                **_moments(oof.loc[sel_old, f"y_{t.name}"].to_numpy()),
                            }
                        )
    return pd.DataFrame(rows)


def shift_by_competition(
    imputed: pd.DataFrame, oof: pd.DataFrame | None, feature_set: str
) -> pd.DataFrame:
    """Mean / sd of every imputed quantity per competition-season (new domain and 360
    out-of-fold)."""
    rows: list[dict[str, Any]] = []
    frames = [("no360 imputed", imputed, "imp_{}__" + feature_set)]
    if oof is not None:
        frames.append(
            (
                "360 oof imputed",
                oof.loc[np.isin(oof["f_type"].to_numpy(dtype=object), list(NO360_TYPES))],
                "{}__" + feature_set,
            )
        )
    for label, fr, pattern in frames:
        for (comp, season, gender), g in fr.groupby(["competition", "season", "gender"], sort=True):
            row: dict[str, Any] = {
                "source": label,
                "competition": comp,
                "season": season,
                "gender": gender,
                "n_rows": int(len(g)),
                "n_matches": int(g["match_id"].nunique()),
            }
            for t in imf.TARGETS:
                c = pattern.format(t.name)
                if c in g.columns:
                    v = g[c].to_numpy(dtype=float)
                    v = v[~np.isnan(v)]
                    row[f"{t.name}_mean"] = float(v.mean()) if len(v) else np.nan
                    row[f"{t.name}_n"] = int(len(v))
            rows.append(row)
    return pd.DataFrame(rows)


def run(
    input_path: Path | None = None,
    output_path: Path | None = None,
    feature_sets: tuple[str, ...] = ("E0", "E2", "E2a"),
    root: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Impute one build-stage frame and write the parquet output plus the shift tables."""
    t0 = time.time()
    proc = root or processed_dir("soccer")
    src = input_path or proc / "events_no360.parquet"
    dst = output_path or proc / "imputed_no360.parquet"
    need = set(ID_COLS) | {
        c
        for fs in feature_sets
        for c in imf.design_columns(fs)
        if c not in ("f_gender", "f_comp_type")
    }
    need |= {"f_poss_elapsed"}  # settled mask
    df = pd.read_parquet(src, columns=sorted(need))
    imputed = impute(df, feature_sets, root)
    oof_path = proc / "imputed_oof.parquet"
    oof = pd.read_parquet(oof_path) if oof_path.exists() else None
    shift = shift_table(imputed, oof, feature_sets)
    by_comp = shift_by_competition(imputed, oof, "E2" if "E2" in feature_sets else feature_sets[-1])
    out = imputed[ID_COLS + [c for c in imputed.columns if c.startswith("imp_")]]
    if write:
        out.to_parquet(dst, index=False)
        rdir = reports_dir()
        shift.to_parquet(rdir / "soccer_02_shift_no360.parquet", index=False)
        by_comp.to_parquet(rdir / "soccer_02_shift_by_competition.parquet", index=False)
    return {
        "imputed": out,
        "shift": shift,
        "by_competition": by_comp,
        "seconds": time.time() - t0,
        "n_rows": int(len(out)),
        "had_oof": oof is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="build-stage parquet (default events_no360.parquet)",
    )
    ap.add_argument(
        "--output", type=Path, default=None, help="default processed/soccer/imputed_no360.parquet"
    )
    ap.add_argument("--feature-sets", default="E0,E2,E2a")
    args = ap.parse_args()
    fsets = tuple(s.strip() for s in args.feature_sets.split(",") if s.strip())
    res = run(args.input, args.output, fsets)
    print(
        f"imputed {res['n_rows']} rows with {fsets} in {res['seconds']:.1f} s (360 oof available: "
        f"{res['had_oof']})"
    )
    with pd.option_context(
        "display.width", 220, "display.max_columns", 30, "display.max_rows", 300
    ):
        sh = res["shift"]
        print(sh[(sh["event_type"] == "all") & (sh["feature_set"] == fsets[-1])])


if __name__ == "__main__":
    main()
