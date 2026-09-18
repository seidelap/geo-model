"""NFL 06 - the participation -> alignment chain: imputed personnel as a student input.

The NFL 02 handoff recommended that the alignment students "condition on the true grouping
where it exists and treat the imputed grouping as a weak prior". NFL 03 never did that: its
``F0`` students use no personnel and ``F0P`` uses the charted (privileged) personnel /
formation strings. This stage closes the gap on the 12 pre-snap targets of the tracked 2017
games with two extra feature sets built from the NFL 02 out-of-fold offense-grouping
probabilities (``p_off_*``, out-of-fold by game over 2016-2020):

* ``F0``    situation + prior-week tendencies + team target encodings (NFL 03, event-only),
* ``F0I``   F0 + the 10 imputed grouping probabilities (event-only, the chained variant),
* ``F0P``   F0 + charted personnel / formation (NFL 03, privileged),
* ``F0PI``  F0P + the imputed probabilities (does the prior add anything to the truth?).

Same protocol as NFL 03 (5-fold group k-fold by game, out-of-fold for every play, LightGBM
with early stopping on an inner game holdout, team encodings fitted inside the training fold)
plus the forward split (weeks 1-4 -> 5-6); every pair carries a play-level and a game-clustered
paired bootstrap interval on per-play losses.

Run from the repo root (~3 min on 2 threads)::

    python -m research.privileged_tracking.nfl.imputation_chain

Outputs: ``reports/nfl_06_participation_chain.md``, ``nfl_06_chain_*.parquet``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (
    clustered_bootstrap_delta,
    paired_bootstrap_delta,
)
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.nfl import imputation as im
from research.privileged_tracking.nfl import imputation_features as imf
from research.privileged_tracking.nfl import participation_features as pf

PRESNAP_TARGETS: tuple[str, ...] = tuple(t.name for t in imf.TARGETS if t.subset == "all")
P_PREFIX = "p_off_"
FEATURE_SET_ORDER: tuple[str, ...] = ("F0", "F0I", "F0P", "F0PI")
DELTA_PAIRS: tuple[tuple[str, str], ...] = (
    ("F0", "F0I"),
    ("F0I", "F0P"),
    ("F0", "F0P"),
    ("F0P", "F0PI"),
)
REPORT_PREFIX = "nfl_06_chain"


@dataclass
class ChainConfig:
    """Driver configuration.

    Attributes:
        n_jobs: LightGBM threads.
        seed: fold / LightGBM seed (the NFL 03 folds are reproduced with the same seed).
        n_boot: bootstrap resamples.
        targets: pre-snap targets to fit (``None`` = all 12).
        write: write parquet / report outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    n_boot: int = 1000
    targets: tuple[str, ...] | None = None
    write: bool = True

    def imputation_cfg(self) -> im.ImputationConfig:
        return im.ImputationConfig(
            n_jobs=self.n_jobs, seed=self.seed, n_boot=self.n_boot, apply_season=None, write=False
        )


def _log(msg: str) -> None:
    print(f"[imputation_chain] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def attach_participation(
    game_id: pd.Series | np.ndarray,
    play_id: pd.Series | np.ndarray,
    imputed: pd.DataFrame,
) -> pd.DataFrame:
    """The NFL 02 ``p_off_*`` probabilities (and ``off_grp``) aligned on the tracked plays.

    Args:
        game_id: nflfastR ``game_id`` per tracked play ``[n]``.
        play_id: nflfastR ``play_id`` per tracked play ``[n]``.
        imputed: ``participation_imputed.parquet`` rows (``game_id, play_id, off_grp, p_off_*``).

    Returns:
        ``[n, 1 + n_classes]`` frame (``off_grp`` + the probability columns) in the order of the
        inputs; NaN where the play has no participation record.
    """
    keys = pd.DataFrame(
        {"game_id": np.asarray(game_id).astype(str), "play_id": np.asarray(play_id).astype(int)}
    )
    p_cols = [c for c in imputed.columns if c.startswith(P_PREFIX)]
    sub = imputed[["game_id", "play_id", "off_grp", *p_cols]].copy()
    sub["game_id"] = sub["game_id"].astype(str)
    sub["play_id"] = sub["play_id"].astype(int)
    sub = sub.drop_duplicates(["game_id", "play_id"])
    m = keys.merge(sub, on=["game_id", "play_id"], how="left")
    assert len(m) == len(keys)
    out = m[["off_grp", *p_cols]].reset_index(drop=True)
    out[p_cols] = out[p_cols].astype(float)
    return out


def chain_feature_sets(p_cols: list[str]) -> dict[str, list[str]]:
    """The four nested feature sets of this stage (NFL 03 ``F0`` / ``F0P`` plus the imputed
    grouping probabilities)."""
    f0, f0p = list(imf.FEATURE_SETS["F0"]), list(imf.FEATURE_SETS["F0P"])
    return {"F0": f0, "F0I": f0 + p_cols, "F0P": f0p, "F0PI": f0p + p_cols}


def grouping_agreement(
    off_grp: pd.Series, probs: pd.DataFrame, offense_personnel: pd.Series
) -> dict[str, float]:
    """How good the imputed grouping is on these plays: argmax accuracy vs the NFL 02 label
    (``off_grp``) and coverage of the join."""
    p_cols = list(probs.columns)
    ok = probs[p_cols[0]].notna().to_numpy()
    classes = [c[len(P_PREFIX) :] for c in p_cols]
    arg = np.asarray(classes, dtype=object)[
        np.nanargmax(np.where(ok[:, None], probs.to_numpy(dtype=float), -1.0), axis=1)
    ]
    truth = (
        off_grp.astype(object)
        .map(lambda v: pf.slug(v) if isinstance(v, str) else v)
        .to_numpy(dtype=object)
    )
    lab = ok & pd.notna(truth)
    acc = float(np.mean(arg[lab] == truth[lab])) if lab.any() else np.nan
    maj = pd.Series(truth[lab]).value_counts(normalize=True).iloc[0] if lab.any() else np.nan
    return {
        "n_plays": int(len(ok)),
        "share_with_probabilities": float(ok.mean()),
        "share_with_label": float(lab.mean()),
        "argmax_accuracy": acc,
        "majority_share": float(maj),
        "share_charted_personnel_present": float(offense_personnel.notna().mean()),
    }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class ChainRun:
    target: str
    split: str
    preds: dict[str, np.ndarray]
    eval_idx: np.ndarray
    metrics: dict[str, dict[str, float]]
    losses: dict[str, np.ndarray]


def run_target_sets(
    spec: imf.TargetSpec,
    data: im.TrackedData,
    design: pd.DataFrame,
    sets: dict[str, list[str]],
    folds: list[tuple[np.ndarray, np.ndarray]],
    split: str,
    cfg: im.ImputationConfig,
) -> ChainRun:
    """Fit every feature set of ``sets`` on each fold (the NFL 03 ``run_target`` protocol with an
    explicit feature-set table and ``design``) and collect out-of-fold predictions."""
    n = len(design)
    y = data.Y[spec.name].to_numpy(dtype=float)
    in_subset = data.masks[spec.subset].to_numpy()
    labelled = in_subset & ~np.isnan(y)
    names = [*sets, "base_global"]
    preds = {k: np.full(n, np.nan) for k in names}
    for tr, te in folds:
        tr_l = tr[labelled[tr]]
        te_s = te[in_subset[te]]
        if len(tr_l) < 50 or len(te_s) == 0:
            continue
        enc_off = imf.TeamEncoder(cfg.te_alpha).fit(data.posteam[tr_l], data.game[tr_l], y[tr_l])
        enc_def = imf.TeamEncoder(cfg.te_alpha).fit(data.defteam[tr_l], data.game[tr_l], y[tr_l])
        xtr = design.iloc[tr_l].copy()
        xtr["te_off"] = enc_off.transform_train(data.posteam[tr_l], data.game[tr_l])
        xtr["te_def"] = enc_def.transform_train(data.defteam[tr_l], data.game[tr_l])
        xte = design.iloc[te_s].copy()
        xte["te_off"] = enc_off.transform(data.posteam[te_s])
        xte["te_def"] = enc_def.transform(data.defteam[te_s])
        for fset, feats in sets.items():
            booster, bi = im.fit_with_inner_holdout(
                xtr[feats], y[tr_l], data.game[tr_l], spec.kind, cfg
            )
            preds[fset][te_s] = booster.predict(xte[feats], num_iteration=bi)
        preds["base_global"][te_s] = float(y[tr_l].mean())
    eval_idx = np.where(labelled & ~np.isnan(preds["base_global"]))[0]
    metrics = {k: im.score(spec.kind, y[eval_idx], preds[k][eval_idx]) for k in names}
    losses = {k: im.per_sample_loss(spec.kind, y[eval_idx], preds[k][eval_idx]) for k in names}
    return ChainRun(spec.name, split, preds, eval_idx, metrics, losses)


def metrics_table(runs: list[ChainRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        spec = imf.TARGET_BY_NAME[r.target]
        for k, m in r.metrics.items():
            rows.append(
                {
                    "target": r.target,
                    "kind": spec.kind,
                    "split": r.split,
                    "feature_set": k,
                    "skill": im.skill_of(spec.kind, m),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def deltas_table(runs: list[ChainRun], data: im.TrackedData, cfg: ChainConfig) -> pd.DataFrame:
    rows = []
    for r in runs:
        spec = imf.TARGET_BY_NAME[r.target]
        groups = data.game[r.eval_idx]
        for a, b in DELTA_PAIRS:
            la, lb = r.losses[a], r.losses[b]
            m, lo, hi = paired_bootstrap_delta(la, lb, n_boot=cfg.n_boot, seed=cfg.seed)
            _, lo_g, hi_g = clustered_bootstrap_delta(
                la, lb, groups, n_boot=cfg.n_boot, seed=cfg.seed
            )
            rows.append(
                {
                    "target": r.target,
                    "kind": spec.kind,
                    "split": r.split,
                    "a": a,
                    "b": b,
                    "loss": "log_loss" if spec.kind == "binary" else "abs_error",
                    "n": int(len(la)),
                    "games": int(len(np.unique(groups))),
                    "skill_a": im.skill_of(spec.kind, r.metrics[a]),
                    "skill_b": im.skill_of(spec.kind, r.metrics[b]),
                    "delta_a_minus_b": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_low_game": lo_g,
                    "ci_high_game": hi_g,
                    "b_better_game_ci": bool(lo_g > 0),
                    "a_better_game_ci": bool(hi_g < 0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver and report
# ---------------------------------------------------------------------------


def run(cfg: ChainConfig) -> dict[str, Any]:
    t0 = time.time()
    icfg = cfg.imputation_cfg()
    data = im.load_tracked(icfg)
    pi = pd.read_parquet(processed_dir("nfl") / "participation_imputed.parquet")
    part = attach_participation(data.plays["pbp_game_id"], data.plays["playId"], pi)
    p_cols = [c for c in part.columns if c.startswith(P_PREFIX)]
    design = pd.concat([data.X.reset_index(drop=True), part[p_cols].reset_index(drop=True)], axis=1)
    sets = chain_feature_sets(p_cols)
    agreement = grouping_agreement(part["off_grp"], part[p_cols], data.frame["offense_personnel"])
    _log(
        f"{len(design):,} tracked plays, {agreement['share_with_probabilities']:.1%} with NFL 02 "
        "probabilities; "
        f"argmax accuracy vs the participation label {agreement['argmax_accuracy']:.3f} "
        f"(majority {agreement['majority_share']:.3f})"
    )
    folds = im.folds_for(data, icfg)
    targets = cfg.targets or PRESNAP_TARGETS
    runs: list[ChainRun] = []
    for t in targets:
        spec = imf.TARGET_BY_NAME[t]
        for split, fl in folds.items():
            t1 = time.time()
            r = run_target_sets(spec, data, design, sets, fl, split, icfg)
            runs.append(r)
            sk = {k: im.skill_of(spec.kind, r.metrics[k]) for k in sets}
            _log(
                f"{t:36s} {split:8s} "
                + "  ".join(f"{k} {v:.3f}" for k, v in sk.items())
                + f"  {time.time() - t1:.0f} s"
            )
    tables = {
        "metrics": metrics_table(runs),
        "deltas": deltas_table(runs, data, cfg),
        "coverage": pd.DataFrame([agreement]),
        "feature_sets": pd.DataFrame(
            [
                {"feature_set": k, "n_features": len(v), "features": ", ".join(v)}
                for k, v in sets.items()
            ]
        ),
    }
    elapsed = time.time() - t0
    text = render_report(cfg, tables, elapsed)
    if cfg.write:
        rd = reports_dir()
        for k, t in tables.items():
            t.to_parquet(rd / f"{REPORT_PREFIX}_{k}.parquet", index=False)
        (rd / "nfl_06_participation_chain.md").write_text(text)
        _log(f"report written: {rd / 'nfl_06_participation_chain.md'} ({elapsed:.0f} s)")
    else:
        _log(text[:2000])
    return tables


def _fmt(r: pd.Series) -> str:
    return (
        f"{r['delta_a_minus_b']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
        f"(game-clustered [{r['ci_low_game']:+.4f}, {r['ci_high_game']:+.4f}])"
    )


def render_report(cfg: ChainConfig, tables: dict[str, pd.DataFrame], elapsed: float) -> str:
    m, d, cov = tables["metrics"], tables["deltas"], tables["coverage"].iloc[0]
    kf = m[m["split"] == "kfold"].pivot(index="target", columns="feature_set", values="skill")
    kf = kf.reindex(columns=["base_global", *FEATURE_SET_ORDER]).reset_index()
    n_by_target = m[(m["split"] == "kfold") & (m["feature_set"] == "F0")].set_index("target")["n"]
    kf.insert(1, "n", kf["target"].map(n_by_target).astype(int))
    fw = m[m["split"] == "forward"].pivot(index="target", columns="feature_set", values="skill")
    fw = fw.reindex(columns=["base_global", *FEATURE_SET_ORDER]).reset_index()
    dk = d[d["split"] == "kfold"].copy()
    dk["delta [play CI] (game-clustered CI)"] = dk.apply(_fmt, axis=1)
    dk["verdict"] = np.where(
        dk["b_better_game_ci"], "b better", np.where(dk["a_better_game_ci"], "a better", "n.s.")
    )
    delta_cols = [
        "target",
        "a",
        "b",
        "loss",
        "n",
        "skill_a",
        "skill_b",
        "delta [play CI] (game-clustered CI)",
        "verdict",
    ]
    f0i = dk[(dk["a"] == "F0") & (dk["b"] == "F0I")]
    f0p = dk[(dk["a"] == "F0I") & (dk["b"] == "F0P")]
    f0pi = dk[(dk["a"] == "F0P") & (dk["b"] == "F0PI")]
    n_better = int(f0i["b_better_game_ci"].sum())
    n_worse = int(f0i["a_better_game_ci"].sum())
    lines = [
        "# NFL 06 - participation -> alignment chain: imputed personnel as a student input",
        "",
        "Machine-written by `python -m research.privileged_tracking.nfl.imputation_chain`. "
        "Question: the",
        "NFL 02 handoff recommended feeding the imputed offense grouping to the alignment "
        "students as a",
        "weak prior; NFL 03 used either no personnel (`F0`) or the charted personnel (`F0P`). "
        "Does the",
        "imputed grouping (`F0I`) recover any of the `F0 -> F0P` step, and does it add anything "
        "on top of",
        "the charted truth (`F0PI`)?",
        "",
        "## Headline",
        "",
        f"- Out of {len(f0i)} pre-snap targets (5-fold by game, out-of-fold), `F0 -> F0I` is "
        "better by game-clustered",
        f"  CI on {n_better} and worse on {n_worse}; the largest skill change is "
        + (
            f"{f0i.loc[(f0i['skill_b'] - f0i['skill_a']).abs().idxmax(), 'target']} "
            f"({f0i.loc[(f0i['skill_b'] - f0i['skill_a']).abs().idxmax(), 'skill_a']:.3f} -> "
            f"{f0i.loc[(f0i['skill_b'] - f0i['skill_a']).abs().idxmax(), 'skill_b']:.3f})."
            if len(f0i)
            else "n/a."
        ),
        "- `F0I -> F0P` (imputed -> charted personnel) is better by clustered CI on "
        f"{int(f0p['b_better_game_ci'].sum())}"
        f" of {len(f0p)} targets; `F0P -> F0PI` on {int(f0pi['b_better_game_ci'].sum())} of "
        f"{len(f0pi)}.",
        "- Imputed grouping quality on the tracked plays: argmax accuracy "
        f"{cov['argmax_accuracy']:.3f} vs the NFL 02",
        f"  label (majority class {cov['majority_share']:.3f}); "
        f"{cov['share_with_probabilities']:.1%} of the",
        f"  {int(cov['n_plays']):,} tracked plays have a participation record (the rest are "
        "kneels / spikes).",
        "",
        "## Protocol",
        "",
        "* **Rows.** The 11,518 tracked 2017 plays (`plays_tracked.parquet`), design and targets "
        "as in NFL 03",
        "  (`imputation.load_tracked`). The NFL 02 `p_off_*` columns "
        "(`participation_imputed.parquet`) are joined on",
        "  (`game_id`, `play_id`); for 2016-2020 they are out-of-fold by game, i.e. the 2017 "
        "probabilities come from",
        "  models that never saw the play's game (but did see other 2017 games and 2018-2020 "
        "games: a within-",
        "  season / future-season coupling that NFL 03's own protocol shares through its k-fold "
        "team encodings).",
        "* **Feature sets.** "
        + "; ".join(
            f"`{r.feature_set}` ({r.n_features} columns)"
            for r in tables["feature_sets"].itertuples()
        )
        + ".",
        "  `F0` and `F0P` are exactly `imputation_features.FEATURE_SETS['F0' / 'F0P']`; `F0I` / "
        "`F0PI` append the 10",
        "  probabilities. Team target encodings are fitted inside the training fold as in NFL 03.",
        "* **Splits.** 5-fold `group_kfold` by `gameId` (seed 0, the NFL 03 folds) and the "
        "forward split weeks 1-4 ->",
        "  5-6; LightGBM with the NFL 03 settings, early stopping on an inner 20% game holdout "
        "of the training fold.",
        "* **Statistics.** Skill = R2 (continuous / count) or BSS (binary) vs the training-fold "
        "mean; paired bootstrap",
        f"  ({cfg.n_boot:,} resamples, seed {cfg.seed}) of per-play absolute error / log-loss, "
        "play-level and",
        "  game-clustered 95% CIs; delta = loss(a) - loss(b), positive = `b` better. Quote the "
        "clustered CI.",
        "",
        "## Skill, 5-fold by game (out-of-fold)",
        "",
        md_table(kf, floatfmt="{:.3f}"),
        "",
        "## Skill, forward split (weeks 1-4 -> 5-6)",
        "",
        md_table(fw, floatfmt="{:.3f}"),
        "",
        "## Paired deltas (k-fold; positive = `b` better)",
        "",
        md_table(dk[delta_cols], floatfmt="{:.3f}"),
        "",
        "## Imputed grouping on the tracked plays",
        "",
        md_table(tables["coverage"], floatfmt="{:.3f}"),
        "",
        "## Caveats",
        "",
        "* The `p_off_*` probabilities are NFL 02's S2 model (situation + shotgun / no-huddle + "
        "team tendencies from",
        "  strictly earlier games): a calibrated '11 personnel unless the team and situation say "
        "otherwise' prior with",
        "  accuracy 0.647 on 2022; the tracked 2017 games are inside its 2016-2020 out-of-fold "
        "training window.",
        "* Only the 12 pre-snap targets are fitted (the within-play students never had a "
        "personnel-dependent gain in",
        "  NFL 03); the payoff stage (NFL 04) already tested the same probabilities as direct "
        "outcome-model features",
        "  (`PBP+IMP_PERS`, stage d) with a null result, so this stage measures the alignment "
        "step of the chain only.",
        "* The stage does not re-run the NFL 04 payoff with `F0I` students; `PBP+IMP_F0` (no "
        "personnel at all) is the",
        "  strictly event-only payoff variant and `PBP+IMP` (F0P students) reads charted "
        "personnel.",
        "",
        f"Timing: {elapsed:.0f} s on {cfg.n_jobs} threads.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument(
        "--targets", default=None, help="comma-separated pre-snap targets (default all 12)"
    )
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    targets = tuple(t.strip() for t in args.targets.split(",")) if args.targets else None
    run(ChainConfig(n_jobs=args.n_jobs, targets=targets, write=not args.no_write))


if __name__ == "__main__":
    main()
