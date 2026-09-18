"""Soccer 06 - payoff of the sequence student: xG and xPass with ``seq20`` imputations.

Stage 03 tested the payoff of imputed defensive state with the stage-02 LightGBM E2 students
only. Stage 05 showed that a GRU over the raw 20-event history (``seq20``) imputes 7 of the
state quantities better than LightGBM on 7/7 targets. This stage asks whether the better
imputations change the payoff null: on the shots and passes of the stage-05 held-out folds
(0-2, 251 matches) the stage-03 xG / xPass models are refitted with

* ``EVENT``                   the stage-03 event-only design (reference),
* ``EVENT+IMP``               the stage-03 imputed block (LightGBM E2 shot state + E2a assist lanes
                              for xG; the seven E2 / E2a pass-state students for xPass),
* ``EVENT+SEQ7(lgbm)``        EVENT + the LightGBM E2 out-of-fold predictions of the seven
                              stage-05 targets (like-for-like input list for the next row),
* ``EVENT+SEQ7(seq20)``       EVENT + the ``seq20`` out-of-fold predictions of the same seven,
* ``EVENT+IMP+SEQ7(seq20)``   the stage-03 block plus the seven ``seq20`` columns,
* ``EVENT+ORACLE360`` (and ``EVENT+ORACLESHOT`` for xG)  the true values (ceilings).

Every imputed column is out-of-fold with respect to the stage-02 match folds; the xG / xPass
models are cross-validated over the three folds (train on two, test on one), so a test row's
imputations come from students that never saw its match. Deltas are paired bootstraps of
per-shot / per-pass log-loss with a match-clustered interval next to them.

Run from the repo root (a few minutes on 2 threads; predictions are cached under
``processed_dir('soccer')/payoff_cache/``)::

    python -m research.privileged_tracking.soccer.payoff_seq --stage all

Outputs: ``reports/soccer_06_seq_payoff.md`` and ``soccer_06_*.parquet``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import r2
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.soccer import payoff as po
from research.privileged_tracking.soccer import payoff_features as pf
from research.privileged_tracking.soccer.payoff_features import (
    ASSIST_STATE,
    PASS_STATE,
    SHOT_STATE,
    PayoffConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

#: stage-05 targets (the seven heads of the multi-task sequence student)
SEQ_TARGETS: tuple[str, ...] = (
    "block_depth",
    "def_line",
    "n_opp_ahead_of_ball",
    "nearest_opp_dist",
    "n_opp_in_cone",
    "deep_block",
    "counter_on",
)
SEQ_BINARY: frozenset[str] = frozenset({"deep_block", "counter_on"})
SEQ_FOLDS: tuple[int, ...] = (0, 1, 2)
SEQ_VARIANT = "seq20"
REPORT_PREFIX = "soccer_06"
XG_CACHE = "seq_xg_preds.parquet"
XPASS_CACHE = "seq_xpass_preds.parquet"
FITS_CACHE = "seq_fits.parquet"

#: variant name -> blocks appended to the event-only design (see :func:`compose_designs`)
XG_SEQ_VARIANTS: dict[str, tuple[str, ...]] = {
    "EVENT": (),
    "EVENT+IMP": ("imp_shot", "imp_assist"),
    "EVENT+SEQ7(lgbm)": ("seq7_lgbm",),
    "EVENT+SEQ7(seq20)": ("seq7_seq20",),
    "EVENT+IMP+SEQ7(seq20)": ("imp_shot", "imp_assist", "seq7_seq20"),
    "EVENT+ORACLE360": ("oracle_shot", "oracle_assist"),
    "EVENT+ORACLESHOT": ("oracleshot_shot", "oracle_assist"),
}
XPASS_SEQ_VARIANTS: dict[str, tuple[str, ...]] = {
    "EVENT": (),
    "EVENT+IMP": ("imp_pass",),
    "EVENT+SEQ7(lgbm)": ("seq7_lgbm",),
    "EVENT+SEQ7(seq20)": ("seq7_seq20",),
    "EVENT+IMP+SEQ7(seq20)": ("imp_pass", "seq7_seq20"),
    "EVENT+ORACLE360": ("oracle_pass",),
}
#: extra paired comparisons besides "every variant vs EVENT"
EXTRA_PAIRS: tuple[tuple[str, str], ...] = (
    ("EVENT+SEQ7(lgbm)", "EVENT+SEQ7(seq20)"),
    ("EVENT+IMP", "EVENT+IMP+SEQ7(seq20)"),
    ("EVENT+SEQ7(seq20)", "EVENT+ORACLE360"),
)


@dataclass
class SeqPayoffConfig:
    """Driver configuration.

    Attributes:
        payoff: stage-03 configuration (LightGBM settings, seeds, bootstrap size).
        folds: stage-02 folds with sequence-student predictions.
        variant: sequence-student variant whose out-of-fold columns are used.
        write: write report / parquet outputs.
    """

    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    folds: tuple[int, ...] = SEQ_FOLDS
    variant: str = SEQ_VARIANT
    write: bool = True


def _log(msg: str) -> None:
    print(f"[payoff_seq] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def restrict_to_folds(df: pd.DataFrame, folds: tuple[int, ...]) -> pd.DataFrame:
    """Rows of ``df`` whose ``fold`` is in ``folds`` (index reset)."""
    return df[df["fold"].isin(list(folds))].reset_index(drop=True)


def attach_oof(
    table: pd.DataFrame,
    oof: pd.DataFrame,
    names: tuple[str, ...],
    fset: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add ``imp_<name>__<fset>`` (and ``y_<name>`` where absent) columns from an out-of-fold
    table keyed by ``event_id``.

    Args:
        table: rows to enrich ``[n, *]`` with an ``event_id`` column.
        oof: out-of-fold predictions with ``event_id``, ``<name>__<fset>`` and optionally
            ``y_<name>`` columns; at most one row per event.
        names: quantities to attach.
        fset: feature-set / variant suffix of the prediction columns.
        overwrite: replace columns that already exist in ``table``.

    Returns:
        A copy of ``table`` with the new columns (NaN where the event has no OOF row).
    """
    if oof["event_id"].duplicated().any():
        raise ValueError("oof holds duplicate event_id rows")
    want: dict[str, str] = {}
    for n in names:
        src = f"{n}__{fset}"
        dst = f"imp_{n}__{fset}"
        if src in oof.columns and (overwrite or dst not in table.columns):
            want[src] = dst
        ysrc = f"y_{n}"
        if ysrc in oof.columns and (overwrite or ysrc not in table.columns):
            want[ysrc] = ysrc
    if not want:
        return table.copy()
    sub = oof[["event_id", *want]].rename(columns=want)
    out = table.drop(columns=[c for c in want.values() if c in table.columns])
    merged = out.merge(sub, on="event_id", how="left")
    assert len(merged) == len(table)
    return merged


def compose_designs(
    base: pd.DataFrame, blocks: dict[str, pd.DataFrame], catalogue: dict[str, tuple[str, ...]]
) -> dict[str, pd.DataFrame]:
    """Design matrix per variant: ``base`` followed by the named ``blocks`` of the catalogue.

    Args:
        base: event-only design ``[n, d]``.
        blocks: named state blocks ``[n, k]`` aligned on the same rows.
        catalogue: variant -> block names.

    Returns:
        ``{variant: design}``; a variant naming a block absent from ``blocks`` is skipped.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, needed in catalogue.items():
        if any(b not in blocks for b in needed):
            continue
        parts = [base] + [blocks[b] for b in needed]
        out[name] = pd.concat(parts, axis=1)
    return out


def state_quality_table(
    table: pd.DataFrame, names: tuple[str, ...], fsets: tuple[str, ...], population: str
) -> pd.DataFrame:
    """Skill of each imputation source against the 360 label on exactly these rows.

    R2 for continuous / count targets, log-loss and AUC for binaries (rows where the label
    and every source are present, so the sources are compared on identical rows).
    """
    rows = []
    for n in names:
        y = table.get(f"y_{n}")
        if y is None:
            continue
        y = y.to_numpy(dtype=float)
        preds = {
            f: table[f"imp_{n}__{f}"].to_numpy(dtype=float)
            for f in fsets
            if f"imp_{n}__{f}" in table
        }
        if not preds:
            continue
        ok = ~np.isnan(y)
        for p in preds.values():
            ok &= ~np.isnan(p)
        for f, p in preds.items():
            row: dict[str, Any] = {
                "population": population,
                "target": n,
                "source": f,
                "n": int(ok.sum()),
            }
            if ok.sum() < 20:
                rows.append(row)
                continue
            if n in SEQ_BINARY:
                m = pf.binary_metrics(y[ok], np.clip(p[ok], 0, 1))
                row.update({"log_loss": m["log_loss"], "auc": m["auc"], "r2": np.nan})
            else:
                row.update({"r2": r2(y[ok], p[ok]), "mae": float(np.mean(np.abs(y[ok] - p[ok])))})
            rows.append(row)
    return pd.DataFrame(rows)


def pairs_table(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    pairs: tuple[tuple[str, str], ...],
    match: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Paired deltas ``loss(from) - loss(to)`` for the named pairs present in ``preds``."""
    rows = []
    for a, b in pairs:
        if a in preds and b in preds:
            rows.append(
                {
                    "from": a,
                    "to": b,
                    **pf.paired_delta(y, preds[a], preds[b], match, n_boot=n_boot, seed=seed),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _seq_path() -> Path:
    return processed_dir("soccer") / "imputed_oof_seq.parquet"


def load_seq_oof(cfg: SeqPayoffConfig, event_ids: np.ndarray) -> pd.DataFrame:
    """``event_id`` + ``<target>__<variant>`` (+ ``y_<target>``) rows of the sequence-student
    out-of-fold table for the given events."""
    cols = (
        ["event_id", "fold"]
        + [f"{t}__{cfg.variant}" for t in SEQ_TARGETS]
        + [f"y_{t}" for t in SEQ_TARGETS]
    )
    names = pq.ParquetFile(_seq_path()).schema_arrow.names
    cols = [c for c in cols if c in names]
    df = pq.read_table(_seq_path(), columns=cols).to_pandas()
    df = df[df["event_id"].isin(set(event_ids))]
    return df.reset_index(drop=True)


def load_lgbm_oof(event_type: str, event_ids: np.ndarray) -> pd.DataFrame:
    """``event_id`` + ``<target>__E2`` (+ ``y_<target>``) rows of the stage-02 out-of-fold table."""
    path = processed_dir("soccer") / "imputed_oof.parquet"
    names = pq.ParquetFile(path).schema_arrow.names
    cols = ["event_id"] + [f"{t}__E2" for t in SEQ_TARGETS] + [f"y_{t}" for t in SEQ_TARGETS]
    cols = [c for c in cols if c in names]
    df = pq.read_table(path, columns=cols, filters=[("f_type", "==", event_type)]).to_pandas()
    df = df[df["event_id"].isin(set(event_ids))]
    return df.reset_index(drop=True)


def seq_shot_table(cfg: SeqPayoffConfig) -> pd.DataFrame:
    """Stage-03 shot table restricted to the sequence folds, with ``imp_<t>__E2`` /
    ``imp_<t>__seq20`` / ``y_<t>`` for every stage-05 target."""
    shots = po.shot_population(cfg.payoff, po.build_shot_table(cfg.payoff))
    shots = restrict_to_folds(shots, cfg.folds)
    ids = shots["event_id"].to_numpy()
    shots = attach_oof(shots, load_lgbm_oof("Shot", ids), SEQ_TARGETS, "E2")
    shots = attach_oof(shots, load_seq_oof(cfg, ids), SEQ_TARGETS, cfg.variant)
    return shots


def seq_pass_table(cfg: SeqPayoffConfig) -> pd.DataFrame:
    """Stage-03 pass subsample restricted to the sequence folds, enriched like the shot table."""
    passes = restrict_to_folds(po.build_pass_table(cfg.payoff), cfg.folds)
    ids = passes["event_id"].to_numpy()
    passes = attach_oof(passes, load_lgbm_oof("Pass", ids), SEQ_TARGETS, "E2")
    passes = attach_oof(passes, load_seq_oof(cfg, ids), SEQ_TARGETS, cfg.variant)
    return passes


def coverage_table(table: pd.DataFrame, variant: str, population: str) -> pd.DataFrame:
    """Share of rows with a prediction per target and source (they follow the stage-02 subsets)."""
    rows = []
    for t in SEQ_TARGETS:
        a = table.get(f"imp_{t}__E2")
        b = table.get(f"imp_{t}__{variant}")
        rows.append(
            {
                "population": population,
                "target": t,
                "n_rows": int(len(table)),
                "share_lgbm_E2": float(a.notna().mean()) if a is not None else np.nan,
                f"share_{variant}": float(b.notna().mean()) if b is not None else np.nan,
                "share_both": float((a.notna() & b.notna()).mean())
                if a is not None and b is not None
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------


def xg_seq_designs(shots: pd.DataFrame, variant: str) -> dict[str, pd.DataFrame]:
    """Design matrix of every :data:`XG_SEQ_VARIANTS` entry."""
    assist = shots[
        [c for c in shots.columns if c.startswith("a_") and not c.startswith(("a_imp_", "a_y_"))]
    ]
    base = pf.shot_event_design(shots, assist)
    blocks = {
        "imp_shot": pf.state_block(shots, SHOT_STATE, "imp", "E2", prefix="s_"),
        "imp_assist": pf.state_block(
            shots, ASSIST_STATE, "imp", "E2a", prefix="as_", col_prefix="a_"
        ),
        "seq7_lgbm": pf.state_block(shots, SEQ_TARGETS, "imp", "E2", prefix="q_"),
        "seq7_seq20": pf.state_block(shots, SEQ_TARGETS, "imp", variant, prefix="q_"),
        "oracle_shot": pf.state_block(shots, SHOT_STATE, "oracle360", prefix="s_"),
        "oracle_assist": pf.state_block(
            shots, ASSIST_STATE, "oracle360", prefix="as_", col_prefix="a_"
        ),
        "oracleshot_shot": pf.state_block(shots, SHOT_STATE, "oracleshot", prefix="s_"),
    }
    return compose_designs(base, blocks, XG_SEQ_VARIANTS)


def xpass_seq_designs(
    passes: pd.DataFrame, with_after: bool, variant: str
) -> dict[str, pd.DataFrame]:
    """Design matrix of every :data:`XPASS_SEQ_VARIANTS` entry for one design mode."""
    base = pf.pass_event_design(passes, with_after)
    fset = "E2a" if with_after else "E2"
    blocks = {
        "imp_pass": pf.state_block(passes, PASS_STATE, "imp", fset, prefix="s_"),
        "seq7_lgbm": pf.state_block(passes, SEQ_TARGETS, "imp", "E2", prefix="q_"),
        "seq7_seq20": pf.state_block(passes, SEQ_TARGETS, "imp", variant, prefix="q_"),
        "oracle_pass": pf.state_block(passes, PASS_STATE, "oracle360", prefix="s_"),
    }
    return compose_designs(base, blocks, XPASS_SEQ_VARIANTS)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _cache(cfg: SeqPayoffConfig) -> Path:
    return po.cache_dir(cfg.payoff)


def stage_xg(cfg: SeqPayoffConfig, force: bool = False) -> pd.DataFrame:
    """xG variants, cross-validated over the sequence folds; writes ``seq_xg_preds.parquet``."""
    path = _cache(cfg) / XG_CACHE
    if path.exists() and not force:
        return pd.read_parquet(path)
    p = cfg.payoff
    shots = seq_shot_table(cfg)
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    fold = shots["fold"].to_numpy()
    designs = xg_seq_designs(shots, cfg.variant)
    preds = shots[po.XG_CONTEXT].copy()
    preds["pred_BASE"] = po.base_rate_oof(y, fold)
    preds[f"pred_{pf.XG_REFERENCE}"] = shots["oracle_statsbomb_xg"].to_numpy(dtype=float)
    fits = []
    for name, x in designs.items():
        t0 = time.time()
        oof, f, _ = po.cv_predict(
            x,
            y,
            match,
            fold,
            p.xg,
            po._seeds(p, p.n_seeds_xg),
            pf.categorical_in(x),
            "binary",
            p.n_jobs,
            refit=True,
        )
        preds[f"pred_{name}"] = oof
        fits += [{"task": "xg", "variant": name, "d": int(x.shape[1]), **r} for r in f]
        m = pf.binary_metrics(y, oof)
        _log(
            f"xg {name:24s} d={x.shape[1]:3d} log-loss {m['log_loss']:.4f} "
            f"auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
        )
    preds.to_parquet(path, index=False)
    _append_fits(cfg, pd.DataFrame(fits), "xg")
    return preds


def stage_xpass(cfg: SeqPayoffConfig, force: bool = False) -> pd.DataFrame:
    """xPass variants (with-after and pre-instant designs) over the sequence folds."""
    path = _cache(cfg) / XPASS_CACHE
    if path.exists() and not force:
        return pd.read_parquet(path)
    p = cfg.payoff
    passes = seq_pass_table(cfg)
    y = passes["is_complete"].to_numpy(dtype=float)
    match = passes["match_id"].to_numpy()
    fold = passes["fold"].to_numpy()
    preds = passes[po.XPASS_CONTEXT].copy()
    preds["pred_BASE"] = po.base_rate_oof(y, fold)
    fits = []
    for with_after in (True, False):
        tag = "after" if with_after else "noafter"
        for name, x in xpass_seq_designs(passes, with_after, cfg.variant).items():
            t0 = time.time()
            oof, f, _ = po.cv_predict(
                x,
                y,
                match,
                fold,
                p.xpass,
                po._seeds(p, p.n_seeds_xpass),
                pf.categorical_in(x),
                "binary",
                p.n_jobs,
                refit=False,
            )
            preds[f"pred_{name}__{tag}"] = oof
            fits += [
                {"task": "xpass", "variant": name, "design": tag, "d": int(x.shape[1]), **r}
                for r in f
            ]
            m = pf.binary_metrics(y, oof)
            _log(
                f"xpass {name:24s} {tag:8s} d={x.shape[1]:3d} log-loss {m['log_loss']:.4f} "
                f"auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
            )
    preds.to_parquet(path, index=False)
    _append_fits(cfg, pd.DataFrame(fits), "xpass")
    return preds


def _append_fits(cfg: SeqPayoffConfig, fits: pd.DataFrame, task: str) -> None:
    path = _cache(cfg) / FITS_CACHE
    if path.exists():
        old = pd.read_parquet(path)
        old = old[old["task"] != task]
        fits = pd.concat([old, fits], ignore_index=True)
    fits.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Tables and report
# ---------------------------------------------------------------------------


def xg_tables(cfg: SeqPayoffConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = cfg.payoff
    y = preds["is_goal"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    cols = po._pred_cols(preds)
    order = ["BASE", *XG_SEQ_VARIANTS, pf.XG_REFERENCE]
    cols = {k: cols[k] for k in order if k in cols}
    return {
        "xg_metrics": pf.metrics_table(y, cols),
        "xg_deltas": pf.deltas_table(y, cols, "EVENT", match, p.n_boot, p.seed),
        "xg_pairs": pairs_table(y, cols, EXTRA_PAIRS, match, p.n_boot, p.seed),
    }


def xpass_tables(cfg: SeqPayoffConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = cfg.payoff
    y = preds["is_complete"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    metrics, deltas, pairs = [], [], []
    for tag in ("after", "noafter"):
        cols = po._pred_cols(preds, f"__{tag}")
        cols = {
            "BASE": preds["pred_BASE"].to_numpy(dtype=float),
            **{k: cols[k] for k in XPASS_SEQ_VARIANTS if k in cols},
        }
        m = pf.metrics_table(y, cols)
        m.insert(0, "design", tag)
        d = pf.deltas_table(y, cols, "EVENT", match, p.n_boot, p.seed)
        d.insert(0, "design", tag)
        q = pairs_table(y, cols, EXTRA_PAIRS, match, p.n_boot, p.seed)
        q.insert(0, "design", tag)
        metrics.append(m)
        deltas.append(d)
        pairs.append(q)
    return {
        "xpass_metrics": pd.concat(metrics, ignore_index=True),
        "xpass_deltas": pd.concat(deltas, ignore_index=True),
        "xpass_pairs": pd.concat(pairs, ignore_index=True),
    }


def _fmt(r: pd.Series) -> str:
    return (
        f"{r['delta_log_loss']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
        f"(clustered [{r['ci_low_clustered']:+.4f}, {r['ci_high_clustered']:+.4f}])"
    )


def _delta_md(
    d: pd.DataFrame,
    extra: list[str] | None = None,
    keys: tuple[str, str] = ("variant", "reference"),
) -> str:
    x = d.copy()
    x["delta_log_loss [per-sample CI] (match-clustered CI)"] = x.apply(_fmt, axis=1)
    x["significant (clustered)"] = (x["ci_low_clustered"] > 0) | (x["ci_high_clustered"] < 0)
    cols = (extra or []) + [
        *keys,
        "n",
        "delta_log_loss [per-sample CI] (match-clustered CI)",
        "significant (clustered)",
    ]
    return md_table(x[cols])


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


def render_report(
    cfg: SeqPayoffConfig, tables: dict[str, pd.DataFrame], elapsed: dict[str, float]
) -> str:
    p = cfg.payoff
    v = cfg.variant
    xm, xd, xq = tables["xg_metrics"], tables["xg_deltas"], tables["xg_pairs"]
    pm, pdl, pq_ = tables["xpass_metrics"], tables["xpass_deltas"], tables["xpass_pairs"]
    n_shots = int(xm.loc[xm["variant"] == "EVENT", "n"].iloc[0])
    n_goals = int(xm.loc[xm["variant"] == "EVENT", "positives"].iloc[0])
    n_pass = int(pm.loc[(pm["variant"] == "EVENT") & (pm["design"] == "after"), "n"].iloc[0])
    xdi = xd.set_index("variant")
    pdi = pdl.set_index(["design", "variant"])

    def dl(name: str) -> str:
        return _fmt(xdi.loc[name]) if name in xdi.index else "n/a"

    def pl(design: str, name: str) -> str:
        return _fmt(pdi.loc[(design, name)]) if (design, name) in pdi.index else "n/a"

    def pair(df: pd.DataFrame, a: str, b: str, design: str | None = None) -> str:
        m = (df["from"] == a) & (df["to"] == b)
        if design is not None:
            m &= df["design"] == design
        return _fmt(df[m].iloc[0]) if m.any() else "n/a"

    folds_txt = ", ".join(str(f) for f in cfg.folds)
    variants_txt = "; ".join(
        f"`{k}` = EVENT + {', '.join(b) if b else 'nothing'}" for k, b in XG_SEQ_VARIANTS.items()
    )
    fits = (
        tables["fits"]
        .groupby(["task", "variant"], as_index=False)
        .agg(
            fits=("rounds", "size"),
            rounds=("rounds", "mean"),
            seconds=("seconds", "sum"),
            d=("d", "first"),
        )
    )
    lines = [
        f"# Soccer 06 - payoff of the sequence student ({v}): xG and xPass on the stage-05 folds",
        "",
        "Machine-written by `python -m research.privileged_tracking.soccer.payoff_seq`. Question:",
        "stage 03 found no xG payoff and a small xPass payoff for imputed defensive state, using",
        f"the stage-02 LightGBM E2 students. Stage 05 showed that the `{v}` GRU over the raw",
        "20-event history imputes seven state quantities better than LightGBM on 7/7 targets.",
        "Does the better imputation change the payoff? Same protocol as stage 03, restricted to",
        f"the {len(cfg.folds)} stage-02 folds ({folds_txt}; 251 matches) on which `{v}`",
        "out-of-fold predictions exist.",
        "",
        "## Headline",
        "",
        f"- xG ({n_shots:,} non-penalty shots, {n_goals:,} goals; {p.n_seeds_xg} seeds; deltas =",
        "  log-loss(EVENT) - log-loss(variant), positive = better; per-shot paired bootstrap",
        "  95% CI, match-clustered CI in parentheses):",
        f"  EVENT+IMP (stage-03 LightGBM block) {dl('EVENT+IMP')};",
        f"  EVENT+SEQ7(lgbm) {dl('EVENT+SEQ7(lgbm)')};",
        f"  **EVENT+SEQ7({v}) {dl('EVENT+SEQ7(seq20)')}**;",
        f"  EVENT+IMP+SEQ7({v}) {dl('EVENT+IMP+SEQ7(seq20)')};",
        f"  EVENT+ORACLE360 {dl('EVENT+ORACLE360')}; EVENT+ORACLESHOT {dl('EVENT+ORACLESHOT')}.",
        "- Direct pair, same seven inputs: EVENT+SEQ7(lgbm) -> EVENT+SEQ7"
        f"({v}) {pair(xq, 'EVENT+SEQ7(lgbm)', 'EVENT+SEQ7(seq20)')}; stage-03 block -> plus the",
        f"  seven `{v}` columns {pair(xq, 'EVENT+IMP', 'EVENT+IMP+SEQ7(seq20)')}.",
        f"- xPass ({n_pass:,} pass attempts of the stage-03 match-stratified subsample, single",
        "  seed), pre-instant design:",
        f"  EVENT+IMP {pl('noafter', 'EVENT+IMP')};",
        f"  EVENT+SEQ7(lgbm) {pl('noafter', 'EVENT+SEQ7(lgbm)')};",
        f"  **EVENT+SEQ7({v}) {pl('noafter', 'EVENT+SEQ7(seq20)')}**;",
        f"  EVENT+IMP+SEQ7({v}) {pl('noafter', 'EVENT+IMP+SEQ7(seq20)')};",
        f"  EVENT+ORACLE360 {pl('noafter', 'EVENT+ORACLE360')}. Direct pair lgbm -> {v}:",
        f"  {pair(pq_, 'EVENT+SEQ7(lgbm)', 'EVENT+SEQ7(seq20)', 'noafter')}.",
        f"- xPass, destination-known design: EVENT+IMP {pl('after', 'EVENT+IMP')};",
        f"  EVENT+SEQ7({v}) {pl('after', 'EVENT+SEQ7(seq20)')};",
        f"  EVENT+IMP+SEQ7({v}) {pl('after', 'EVENT+IMP+SEQ7(seq20)')}; direct pair lgbm -> {v}:",
        f"  {pair(pq_, 'EVENT+SEQ7(lgbm)', 'EVENT+SEQ7(seq20)', 'after')}.",
        "",
        "## Protocol",
        "",
        "* **Rows.** The stage-03 shot table (`payoff_cache/payoff_shots.parquet`, penalties",
        f"  excluded) and pass subsample restricted to folds {list(cfg.folds)} of the stage-02",
        f"  5-fold match split (`group_kfold` by `match_id`, seed 0): {n_shots:,} shots and",
        f"  {n_pass:,} pass attempts in 251 matches.",
        "* **Imputed columns.** `imp_<t>__E2` = stage-02 LightGBM out-of-fold predictions;",
        f"  `imp_<t>__{v}` = stage-05 `{v}` out-of-fold predictions",
        "  (`processed/soccer/imputed_oof_seq.parquet`; for fold k the network was trained on the",
        "  other four folds with an inner match holdout). Seven targets: "
        + ", ".join(SEQ_TARGETS)
        + ".",
        "  Counts / distances are clipped at 0 (`payoff_features.clip_state`); NaN where the",
        "  stage-02 subset rule leaves the quantity undefined (e.g. cone counts outside possession",
        "  events), identically for both sources.",
        f"* **Variants.** {variants_txt}. `imp_shot` / `imp_assist` / `imp_pass` are the stage-03",
        "  blocks (SHOT_STATE E2, ASSIST_STATE E2a, PASS_STATE E2 or E2a); `seq7_lgbm` /",
        "  `seq7_seq20` are the seven stage-05 targets from LightGBM E2 or from the sequence",
        "  student; `oracle_*` the 360-frame values; `oracleshot_shot` the shot-freeze-frame",
        "  values.",
        "* **Models.** Stage-03 LightGBM settings (`PayoffConfig.xg` / `.xpass`), cross-validated",
        f"  over the {len(cfg.folds)} folds (train on the other two, early stopping on an inner",
        "  match holdout, xG refit on all training rows), so every test row's imputations come",
        "  from students that never saw its match. Training rows' imputations were produced by",
        "  students that saw the test fold's 360 labels (the stage-03 second-order stacking",
        "  coupling, unchanged).",
        f"* **Statistics.** Paired bootstrap ({p.n_boot:,} resamples, seed {p.seed}) of per-shot /",
        "  per-pass log-loss, plus a match-clustered bootstrap; deltas are",
        "  `loss(reference) - loss(variant)`, positive = the variant is better. `significant",
        "  (clustered)` uses the clustered interval.",
        "",
        "## Imputation quality on exactly these rows",
        "",
        "Skill of each source against the 360 label on the shot rows and on the pass rows",
        "(identical rows per target):",
        "",
        md_table(tables["state_quality"], floatfmt="{:.3f}"),
        "",
        "Coverage (share of rows with a prediction; both sources follow the stage-02 subset",
        "rules):",
        "",
        md_table(tables["coverage"], floatfmt="{:.3f}"),
        "",
        "## xG",
        "",
        _metrics_md(xm),
        "",
        "Deltas vs EVENT:",
        "",
        _delta_md(xd),
        "",
        "Direct pairs:",
        "",
        _delta_md(xq, keys=("from", "to")),
        "",
        "## xPass",
        "",
        _metrics_md(pm, ["design"]),
        "",
        "Deltas vs EVENT (per design):",
        "",
        _delta_md(pdl, ["design"]),
        "",
        "Direct pairs:",
        "",
        _delta_md(pq_, ["design"], keys=("from", "to")),
        "",
        "## Fits",
        "",
        md_table(fits, floatfmt="{:.1f}"),
        "",
        "## Caveats",
        "",
        f"* Only the seven stage-05 targets have `{v}` predictions; the stage-03 shot-state block",
        "  also holds `nearest_opp_dist_in_cone`, `opp_keeper_dist_to_goal_line` and",
        "  `n_opp_within_5`, and the pass-state block `n_opp_in_lane`, `n_opp_within_3_of_end` and",
        "  `nearest_opp_to_receiver`, which stay LightGBM-imputed in the `EVENT+IMP+SEQ7` variant.",
        "  `EVENT+SEQ7(lgbm)` vs `EVENT+SEQ7(seq20)` is the like-for-like comparison.",
        f"* Three folds instead of five: the xG models train on ~{2 * n_shots // 3:,} shots per",
        "  fold instead of ~8k, so the absolute log-losses are slightly worse than stage 03 and",
        "  the intervals wider; the stage-03 numbers are not comparable row for row.",
        "* The sequence student's binary heads are less calibrated than LightGBM's (stage 05); the",
        "  downstream LightGBM can re-calibrate them, so this is not a handicap for the payoff",
        "  test.",
        "* xPass uses one seed and the stage-03 40% match-stratified pass subsample; xG averages",
        f"  {p.n_seeds_xg} seeds.",
        "",
        "## Timing",
        "",
        "; ".join(f"{k}: {s:.0f} s" for k, s in elapsed.items()),
        "",
    ]
    return "\n".join(lines)


def stage_report(cfg: SeqPayoffConfig, elapsed: dict[str, float] | None = None) -> None:
    """Assemble every table, write ``soccer_06_*.parquet`` and ``soccer_06_seq_payoff.md``."""
    t0 = time.time()
    xg = stage_xg(cfg)
    xp = stage_xpass(cfg)
    tables = {**xg_tables(cfg, xg), **xpass_tables(cfg, xp)}
    shots = seq_shot_table(cfg)
    passes = seq_pass_table(cfg)
    tables["state_quality"] = pd.concat(
        [
            state_quality_table(shots, SEQ_TARGETS, ("E2", cfg.variant), "shots"),
            state_quality_table(passes, SEQ_TARGETS, ("E2", cfg.variant), "passes"),
        ],
        ignore_index=True,
    )
    tables["coverage"] = pd.concat(
        [
            coverage_table(shots, cfg.variant, "shots"),
            coverage_table(passes, cfg.variant, "passes"),
        ],
        ignore_index=True,
    )
    tables["fits"] = pd.read_parquet(_cache(cfg) / FITS_CACHE)
    elapsed = dict(elapsed or {})
    elapsed["report"] = time.time() - t0
    text = render_report(cfg, tables, elapsed)
    if cfg.write:
        out = reports_dir()
        for k, t in tables.items():
            t.to_parquet(out / f"{REPORT_PREFIX}_{k}.parquet", index=False)
        (out / f"{REPORT_PREFIX}_seq_payoff.md").write_text(text)
        _log(f"report written: {out / (REPORT_PREFIX + '_seq_payoff.md')} ({len(tables)} tables)")
    else:
        _log(text[:3000])


STAGES = ("xg", "xpass", "report")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", default="all", help="one of " + ", ".join(STAGES) + " or all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=None, help="seeds averaged per xG variant")
    ap.add_argument("--variant", default=SEQ_VARIANT)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    pcfg = PayoffConfig(n_jobs=args.n_jobs)
    if args.n_seeds:
        pcfg.n_seeds_xg = args.n_seeds
    cfg = SeqPayoffConfig(payoff=pcfg, variant=args.variant, write=not args.no_write)
    stages = list(STAGES) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    elapsed: dict[str, float] = {}
    for s in stages:
        t0 = time.time()
        if s == "xg":
            stage_xg(cfg, force=args.force)
        elif s == "xpass":
            stage_xpass(cfg, force=args.force)
        elif s == "report":
            stage_report(cfg, elapsed)
        else:
            raise SystemExit(f"unknown stage {s}")
        elapsed[s] = time.time() - t0
        _log(f"stage {s}: {elapsed[s]:.0f} s")


if __name__ == "__main__":
    main()
