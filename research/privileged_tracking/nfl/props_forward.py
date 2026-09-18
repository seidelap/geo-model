"""NFL 07 - forward receiving-props test: does imputed separation help predict the NEXT game?

Stage 04c showed that imputed separation reproduces the same week's NGS ``avg_separation``
(a descriptive, after-the-fact quantity). A player-prop price is forward-looking, so this stage
asks the question the props angle actually needs: given everything known about a receiver up
to and including week t of a season (play-by-play aggregates, with or without the imputed
tracking state of those plays), how well can the receiving yards / targets of his NEXT game be
predicted, and does the imputed block add anything to the event-only history?

Protocol (all quantities are per receiver-game; every feature is computed from games of weeks
<= t of the same season, the target is the receiver's next game with >= 1 target in that season):

* ``BASE_STD`` / ``BASE_L3``: no model, the season-to-date / last-three-game mean of the target.
* ``HIST``: LightGBM on event-only aggregates (targets, receptions, yards, air yards, ``cp``,
  EPA, target share; current game, season-to-date and last three games; games played, week).
* ``HIST+IMP_F0T`` / ``HIST+IMP_F1T`` / ``HIST+IMP_F1`` / ``HIST+IMP_ALL``: the same plus the
  imputed at-release separation (F0T: event-only student; F1T: personnel-aware student), the
  after-the-fact F1 student, the imputed cushion and target depth (same three aggregations).
* ``HIST+NGS``: the same plus the official NGS weekly separation / cushion / intended air yards
  where NGS publishes them (receivers with >= 5 targets in the week): a privileged reference.

Rolling-origin forward split: test seasons 2020, 2021 and 2022, each predicted by a model
trained on the seasons before it (rounds chosen by early stopping on the last training
season, then refit on every training season); pooled and per-season metrics; paired bootstrap
of per-row squared error (and absolute error) against ``HIST`` with a week-clustered interval;
three-seed refit spread for the key pairs. The imputed columns come from students trained on
the 2017 tracked games only (a disjoint domain).

Run from the repo root (~5 min on 2 threads)::

    python -m research.privileged_tracking.nfl.props_forward

Outputs: ``reports/nfl_07_props_forward.md``, ``nfl_07_props_*.parquet``,
``processed/nfl/props_forward_frame.parquet``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import processed_dir, reports_dir
from research.privileged_tracking.common.metrics import (
    clustered_bootstrap_delta,
    paired_bootstrap_delta,
)
from research.privileged_tracking.common.report import md_table

SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)
TEST_SEASONS: tuple[int, ...] = (2020, 2021, 2022)
#: imputed per-play columns aggregated per receiver-game (from ``payoff_pbp_<season>.parquet``)
IMP_COLS: tuple[str, ...] = (
    "imp_separation_at_arrival__F0T",
    "imp_separation_at_arrival__F1T",
    "imp_separation_at_arrival__F1",
    "imp_cb_cushion__F0",
    "imp_cb_cushion__F0P",
    "imp_target_depth__F1T",
)
#: official NGS weekly receiving fields (present only for receivers with >= 5 targets)
NGS_COLS: tuple[str, ...] = ("avg_separation", "avg_cushion", "avg_intended_air_yards")
#: event-only per-game quantities
EVENT_COLS: tuple[str, ...] = (
    "n_targets",
    "receptions",
    "rec_yards",
    "air_yards",
    "cp",
    "epa",
    "target_share",
)
TARGETS: dict[str, str] = {"next_rec_yards": "rec_yards", "next_n_targets": "n_targets"}
AGGS: tuple[str, ...] = ("cur", "std", "l3")
LAST_K = 3


def _agg_cols(cols: tuple[str, ...] | list[str]) -> list[str]:
    return [f"{a}_{c}" for c in cols for a in AGGS]


HIST_COLS: list[str] = _agg_cols(EVENT_COLS) + ["n_games", "week"]
IMP_SETS: dict[str, list[str]] = {
    "F0T": ["imp_separation_at_arrival__F0T", "imp_cb_cushion__F0"],
    "F1T": ["imp_separation_at_arrival__F1T", "imp_cb_cushion__F0P", "imp_target_depth__F1T"],
    "F1": ["imp_separation_at_arrival__F1"],
}
FEATURE_SETS: dict[str, list[str]] = {
    "HIST": HIST_COLS,
    "HIST+IMP_F0T": HIST_COLS + _agg_cols(IMP_SETS["F0T"]),
    "HIST+IMP_F1T": HIST_COLS + _agg_cols(IMP_SETS["F1T"]),
    "HIST+IMP_F1": HIST_COLS + _agg_cols(IMP_SETS["F1"]),
    "HIST+IMP_ALL": HIST_COLS + _agg_cols(list(IMP_COLS)),
    "HIST+NGS": HIST_COLS + _agg_cols(NGS_COLS),
}
BASELINES: dict[str, str] = {"BASE_STD": "std_{t}", "BASE_L3": "l3_{t}"}
REFERENCE = "HIST"
REFIT_SETS: tuple[str, ...] = ("HIST", "HIST+IMP_F1T", "HIST+IMP_ALL")
POPULATIONS: dict[str, str] = {
    "all": "every receiver-game with a later game in the season",
    "regular": ">= 3 games played and >= 3 targets per game season-to-date (props-eligible)",
}
REPORT_PREFIX = "nfl_07_props"


@dataclass
class PropsConfig:
    """Driver configuration.

    Attributes:
        n_jobs: LightGBM threads.
        seed: base seed; ``refit_seeds`` are the extra seeds of the refit-noise check.
        learning_rate / num_leaves / min_data_in_leaf / max_rounds / early_stopping: LightGBM.
        n_boot: bootstrap resamples.
        write: write parquet / report outputs.
    """

    n_jobs: int = 2
    seed: int = 0
    refit_seeds: tuple[int, ...] = (1, 2)
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 50
    max_rounds: int = 800
    early_stopping: int = 40
    n_boot: int = 2000
    write: bool = True


def _log(msg: str) -> None:
    print(f"[props_forward] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def receiver_game_table(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, receiver) receiving line and mean imputed state over targeted passes.

    Args:
        plays: pass plays with ``season, week, posteam, receiver_player_id, complete_pass,
            yards_gained, air_yards, cp, epa`` and the :data:`IMP_COLS` present.

    Returns:
        one row per receiver-game: ``n_targets``, ``receptions``, ``rec_yards`` (yards gained on
        completions), mean ``air_yards`` / ``cp`` / ``epa``, ``team_targets``, ``target_share``
        and the mean of every imputed column present; sorted by (season, receiver, week).
    """
    p = plays[plays["receiver_player_id"].notna()].copy()
    cp = pd.to_numeric(p["complete_pass"], errors="coerce").fillna(0.0)
    yg = pd.to_numeric(p["yards_gained"], errors="coerce").fillna(0.0)
    p["receptions"] = cp
    p["rec_yards"] = np.where(cp.to_numpy() == 1, yg.to_numpy(), 0.0)
    for c in ("air_yards", "cp", "epa"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    keys = ["season", "week", "receiver_player_id"]
    agg: dict[str, Any] = {
        "posteam": ("posteam", "first"),
        "n_targets": ("receiver_player_id", "size"),
        "receptions": ("receptions", "sum"),
        "rec_yards": ("rec_yards", "sum"),
        "air_yards": ("air_yards", "mean"),
        "cp": ("cp", "mean"),
        "epa": ("epa", "mean"),
    }
    for c in IMP_COLS:
        if c in p.columns:
            agg[c] = (c, "mean")
    g = p.groupby(keys, sort=True).agg(**agg).reset_index()
    team = p.groupby(["season", "week", "posteam"]).size().rename("team_targets").reset_index()
    g = g.merge(team, on=["season", "week", "posteam"], how="left")
    g["target_share"] = g["n_targets"] / g["team_targets"]
    return g.sort_values(["season", "receiver_player_id", "week"]).reset_index(drop=True)


def history_features(
    rg: pd.DataFrame, value_cols: tuple[str, ...] | list[str], last_k: int = LAST_K
) -> pd.DataFrame:
    """Season-to-date, last-``k``-game and current-game aggregates plus next-game targets.

    For every receiver-season (rows ordered by week) and every column ``c`` of ``value_cols``:
    ``cur_c`` = this game's value, ``std_c`` = mean over games <= t (NaN values skipped),
    ``l3_c`` = mean over the last ``last_k`` games <= t; ``n_games`` = games played so far;
    ``next_rec_yards`` / ``next_n_targets`` / ``next_receptions`` / ``next_week`` = the
    receiver's next game in the season (NaN on his last game). Nothing from a later week
    enters a feature.
    """
    d = rg.sort_values(["season", "receiver_player_id", "week"]).reset_index(drop=True)
    grp = d.groupby(["season", "receiver_player_id"], sort=False)
    out = d[["season", "week", "receiver_player_id", "posteam"]].copy()
    out["n_games"] = grp.cumcount() + 1
    for c in value_cols:
        s = d[c].astype(float)
        out[f"cur_{c}"] = s.to_numpy()
        out[f"std_{c}"] = grp[c].transform(lambda x: x.expanding().mean()).to_numpy()
        out[f"l3_{c}"] = (
            grp[c].transform(lambda x: x.rolling(last_k, min_periods=1).mean()).to_numpy()
        )
    for t, src in TARGETS.items():
        out[t] = grp[src].shift(-1).to_numpy()
    out["next_receptions"] = grp["receptions"].shift(-1).to_numpy()
    out["next_week"] = grp["week"].shift(-1).to_numpy()
    return out


def forward_frame(rg: pd.DataFrame, ngs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Modelling frame: :func:`history_features` over the event, imputed and (optional) NGS
    columns, restricted to receiver-games that have a next game in the season."""
    d = rg.copy()
    cols = list(EVENT_COLS) + [c for c in IMP_COLS if c in d.columns]
    if ngs is not None:
        keys = ["season", "week", "receiver_player_id"]
        n = ngs[keys + [c for c in NGS_COLS if c in ngs.columns]].drop_duplicates(keys)
        d = d.merge(n, on=keys, how="left")
        cols += [c for c in NGS_COLS if c in d.columns]
    f = history_features(d, cols)
    f = f[f["next_week"].notna()].reset_index(drop=True)
    f["next_week"] = f["next_week"].astype(int)
    return f


def population_mask(f: pd.DataFrame, name: str) -> np.ndarray:
    """Row mask of one :data:`POPULATIONS` entry."""
    if name == "all":
        return np.ones(len(f), dtype=bool)
    if name == "regular":
        return ((f["n_games"] >= 3) & (f["std_n_targets"] >= 3.0)).to_numpy()
    raise KeyError(name)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """n, R2, MAE, RMSE, bias."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "r2": float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else np.nan,
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float(np.mean(p - y)),
    }


def delta_rows(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    reference: str,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Paired deltas ``loss(reference) - loss(model)`` of squared and absolute error (positive =
    the model is better) with per-row and group-clustered 95% intervals."""
    rows = []
    ref = preds[reference]
    for name, p in preds.items():
        if name == reference:
            continue
        se_a, se_b = (y - ref) ** 2, (y - p) ** 2
        ae_a, ae_b = np.abs(y - ref), np.abs(y - p)
        d, lo, hi = paired_bootstrap_delta(se_a, se_b, n_boot=n_boot, seed=seed)
        _, glo, ghi = clustered_bootstrap_delta(se_a, se_b, groups, n_boot=n_boot, seed=seed)
        da, alo, ahi = paired_bootstrap_delta(ae_a, ae_b, n_boot=n_boot, seed=seed)
        _, aglo, aghi = clustered_bootstrap_delta(ae_a, ae_b, groups, n_boot=n_boot, seed=seed)
        var = float(np.var(y))
        rows.append(
            {
                "model": name,
                "reference": reference,
                "n": int(len(y)),
                "n_weeks": int(len(np.unique(groups))),
                "delta_se": d,
                "ci_low": lo,
                "ci_high": hi,
                "ci_low_week": glo,
                "ci_high_week": ghi,
                "delta_r2": d / var if var > 0 else np.nan,
                "delta_ae": da,
                "ae_ci_low": alo,
                "ae_ci_high": ahi,
                "ae_ci_low_week": aglo,
                "ae_ci_high_week": aghi,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _params(cfg: PropsConfig, seed: int) -> dict[str, Any]:
    return {
        "objective": "regression",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": cfg.n_jobs,
        "seed": seed,
        "verbose": -1,
        "max_bin": 63,
        "deterministic": True,
        "force_row_wise": True,
        "metric": "l2",
    }


def fit_forward(
    x: pd.DataFrame,
    y: np.ndarray,
    season: np.ndarray,
    test_season: int,
    cfg: PropsConfig,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Train on seasons < ``test_season`` (rounds by early stopping on the last of them, then a
    refit on all of them) and predict the test season's rows.

    Returns:
        ``(prediction[n_test], rounds)`` aligned with ``season == test_season``.
    """
    train = season < test_season
    val = season == (test_season - 1)
    fit = train & ~val
    if fit.sum() < 100:  # first test season with a single training season: split by week parity
        wk = x["week"].to_numpy()
        val = train & (wk % 2 == 0)
        fit = train & ~val
    params = _params(cfg, seed)
    dtr = lgb.Dataset(x[fit], label=y[fit], free_raw_data=False)
    dva = lgb.Dataset(x[val], label=y[val], reference=dtr, free_raw_data=False)
    b = lgb.train(
        params,
        dtr,
        num_boost_round=cfg.max_rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)],
    )
    rounds = int(b.best_iteration or cfg.max_rounds)
    full = lgb.train(params, lgb.Dataset(x[train], label=y[train]), num_boost_round=max(1, rounds))
    return full.predict(x[season == test_season]), rounds


def run_population(f: pd.DataFrame, target: str, pop: str, cfg: PropsConfig) -> dict[str, Any]:
    """Every model and baseline on one (target, population); pooled over the test seasons."""
    mask = population_mask(f, pop) & f[target].notna().to_numpy()
    d = f[mask].reset_index(drop=True)
    y = d[target].to_numpy(dtype=float)
    season = d["season"].to_numpy()
    src = TARGETS[target]
    preds: dict[str, np.ndarray] = {}
    seed_preds: dict[int, dict[str, np.ndarray]] = {}
    fits = []
    test = np.isin(season, list(TEST_SEASONS))
    for name, col in BASELINES.items():
        preds[name] = d[col.format(t=src)].to_numpy(dtype=float)
    for name, cols in FEATURE_SETS.items():
        x = d[cols].astype(float)
        seeds = (cfg.seed, *cfg.refit_seeds) if name in REFIT_SETS else (cfg.seed,)
        for sd in seeds:
            p = np.full(len(d), np.nan)
            for ts in TEST_SEASONS:
                t0 = time.time()
                p[season == ts], rounds = fit_forward(x, y, season, ts, cfg, sd)
                fits.append(
                    {
                        "target": target,
                        "population": pop,
                        "model": name,
                        "seed": sd,
                        "test_season": ts,
                        "n_train": int((season < ts).sum()),
                        "n_test": int((season == ts).sum()),
                        "rounds": rounds,
                        "seconds": time.time() - t0,
                    }
                )
            if sd == cfg.seed:
                preds[name] = p
            seed_preds.setdefault(sd, {})[name] = p
    # global training mean per test season (a no-skill reference)
    g = np.full(len(d), np.nan)
    for ts in TEST_SEASONS:
        g[season == ts] = float(y[season < ts].mean())
    preds = {"BASE_GLOBAL": g, **preds}
    groups = (d["season"].astype(str) + "_" + d["week"].astype(str)).to_numpy()
    metrics = []
    for name, p in preds.items():
        for scope in ("pooled", *TEST_SEASONS):
            m = test if scope == "pooled" else (season == scope)
            metrics.append(
                {
                    "target": target,
                    "population": pop,
                    "model": name,
                    "scope": str(scope),
                    **regression_metrics(y[m], p[m]),
                }
            )
    deltas = delta_rows(
        y[test],
        {k: v[test] for k, v in preds.items()},
        REFERENCE,
        groups[test],
        cfg.n_boot,
        cfg.seed,
    )
    refit = []
    for name in REFIT_SETS:
        means = [float(np.mean((y[test] - seed_preds[sd][name][test]) ** 2)) for sd in seed_preds]
        refit.append(
            {
                "target": target,
                "population": pop,
                "model": name,
                "mean_se_by_seed": ", ".join(f"{m:.3f}" for m in means),
                "refit_spread_se": max(means) - min(means),
            }
        )
    spread = {r["model"]: r["refit_spread_se"] for r in refit}
    for r in deltas:
        r.update({"target": target, "population": pop})
        r["refit_spread_se"] = max(spread.get(REFERENCE, 0.0), spread.get(r["model"], 0.0))
        r["inside_refit_noise"] = bool(abs(r["delta_se"]) <= r["refit_spread_se"])
    return {
        "metrics": pd.DataFrame(metrics),
        "deltas": pd.DataFrame(deltas),
        "refit": pd.DataFrame(refit),
        "fits": pd.DataFrame(fits),
        "n": int(test.sum()),
    }


# ---------------------------------------------------------------------------
# Data and driver
# ---------------------------------------------------------------------------

LOAD_COLS = [
    "season",
    "week",
    "posteam",
    "play_type",
    "receiver_player_id",
    "complete_pass",
    "yards_gained",
    "air_yards",
    "cp",
    "epa",
    *IMP_COLS,
]


def load_plays() -> pd.DataFrame:
    """Targeted pass plays of every season in :data:`SEASONS` from the stage-04 season frames."""
    frames = []
    for s in SEASONS:
        p = pd.read_parquet(processed_dir("nfl") / f"payoff_pbp_{s}.parquet", columns=LOAD_COLS)
        p = p[(p["play_type"] == "pass") & p["receiver_player_id"].notna()]
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def load_ngs() -> pd.DataFrame | None:
    path = processed_dir("nfl") / "payoff_receiver_week.parquet"
    if not path.exists():
        return None
    rw = pd.read_parquet(path)
    return rw[["season", "week", "receiver_player_id", *[c for c in NGS_COLS if c in rw.columns]]]


def counts_table(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pop in POPULATIONS:
        m = population_mask(f, pop)
        for s in SEASONS:
            mm = m & (f["season"].to_numpy() == s)
            rows.append(
                {
                    "population": pop,
                    "season": s,
                    "receiver_games": int(mm.sum()),
                    "receivers": int(f.loc[mm, "receiver_player_id"].nunique()),
                    "ngs_available": float(f.loc[mm, "cur_avg_separation"].notna().mean())
                    if "cur_avg_separation" in f.columns and mm.any()
                    else np.nan,
                    "mean_next_rec_yards": float(f.loc[mm, "next_rec_yards"].mean())
                    if mm.any()
                    else np.nan,
                    "mean_next_n_targets": float(f.loc[mm, "next_n_targets"].mean())
                    if mm.any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run(cfg: PropsConfig) -> dict[str, Any]:
    t0 = time.time()
    plays = load_plays()
    rg = receiver_game_table(plays)
    f = forward_frame(rg, load_ngs())
    _log(
        f"{len(plays):,} targeted passes -> {len(rg):,} receiver-games -> {len(f):,} with a next "
        "game"
    )
    res: dict[str, list[pd.DataFrame]] = {"metrics": [], "deltas": [], "refit": [], "fits": []}
    for target in TARGETS:
        for pop in POPULATIONS:
            t1 = time.time()
            r = run_population(f, target, pop, cfg)
            for k in res:
                res[k].append(r[k])
            _log(f"{target} / {pop}: n_test {r['n']:,}, {time.time() - t1:.0f} s")
    tables = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    tables["counts"] = counts_table(f)
    elapsed = time.time() - t0
    text = render_report(cfg, tables, elapsed)
    if cfg.write:
        rd = reports_dir()
        for k, t in tables.items():
            t.to_parquet(rd / f"{REPORT_PREFIX}_{k}.parquet", index=False)
        (rd / f"{REPORT_PREFIX}_forward.md").write_text(text)
        f.to_parquet(processed_dir("nfl") / "props_forward_frame.parquet", index=False)
        _log(f"report written: {rd / (REPORT_PREFIX + '_forward.md')}")
    else:
        _log(text[:2000])
    return tables


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(r: pd.Series, key: str = "se") -> str:
    if key == "se":
        return (
            f"{r['delta_se']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] "
            f"(week-clustered [{r['ci_low_week']:+.2f}, {r['ci_high_week']:+.2f}])"
        )
    return (
        f"{r['delta_ae']:+.3f} [{r['ae_ci_low']:+.3f}, {r['ae_ci_high']:+.3f}] "
        f"(week-clustered [{r['ae_ci_low_week']:+.3f}, {r['ae_ci_high_week']:+.3f}])"
    )


def _deltas_md(d: pd.DataFrame) -> str:
    x = d.copy()
    x["delta squared error [CI] (clustered)"] = x.apply(_fmt, axis=1)
    x["delta absolute error [CI] (clustered)"] = x.apply(lambda r: _fmt(r, "ae"), axis=1)
    cols = [
        "model",
        "n",
        "delta_r2",
        "delta squared error [CI] (clustered)",
        "delta absolute error [CI] (clustered)",
        "refit_spread_se",
        "inside_refit_noise",
    ]
    return md_table(x[cols], floatfmt="{:.4f}")


def render_report(cfg: PropsConfig, tables: dict[str, pd.DataFrame], elapsed: float) -> str:
    m, d, refit, counts = tables["metrics"], tables["deltas"], tables["refit"], tables["counts"]
    lines = [
        "# NFL 07 - forward receiving-props test: next-game receiving yards and targets",
        "",
        "Machine-written by `python -m research.privileged_tracking.nfl.props_forward`. "
        "Question: a",
        "player-prop price is forward-looking. Given a receiver's play-by-play history up to "
        "week t",
        "(with or without the imputed tracking state of those plays), how well is his NEXT game's",
        "receiving line predicted, and does the imputed state add anything to the event-only "
        "history?",
        "",
        "## Headline",
        "",
    ]
    for target in TARGETS:
        for pop in POPULATIONS:
            dd = d[(d["target"] == target) & (d["population"] == pop)].set_index("model")
            mm = m[(m["target"] == target) & (m["population"] == pop) & (m["scope"] == "pooled")]
            mm = mm.set_index("model")
            if dd.empty:
                continue
            n = int(mm.loc["HIST", "n"])
            lines.append(
                f"- **{target}, population `{pop}`** (n = {n:,} test receiver-games, 2020-2022): "
                "R2 "
                f"BASE_STD {mm.loc['BASE_STD', 'r2']:.3f}, BASE_L3 {mm.loc['BASE_L3', 'r2']:.3f}, "
                f"HIST {mm.loc['HIST', 'r2']:.3f} (MAE {mm.loc['HIST', 'mae']:.2f}); deltas vs "
                "HIST "
                f"(squared error, positive = better): "
                + "; ".join(
                    f"{k} {_fmt(dd.loc[k])}"
                    for k in (
                        "HIST+IMP_F0T",
                        "HIST+IMP_F1T",
                        "HIST+IMP_F1",
                        "HIST+IMP_ALL",
                        "HIST+NGS",
                    )
                    if k in dd.index
                )
                + "."
            )
    lines += [
        "",
        "## Protocol",
        "",
        "* **Rows.** Targeted pass plays of the 2018-2022 regular seasons "
        "(`payoff_pbp_<season>.parquet`,",
        "  the stage-04 season frames with every student applied) aggregated per (season, week,",
        "  receiver): `n_targets`, `receptions`, `rec_yards` (yards gained on completions), mean",
        "  `air_yards` / `cp` / `epa`, `target_share`, and the mean imputed separation / cushion /",
        "  target depth of the targeted plays. One row per receiver-game that has a later game "
        "with",
        "  >= 1 target in the same season; the target is that next game's `rec_yards` / "
        "`n_targets`.",
        "* **Features** (all from games <= t of the same season): current game (`cur_`), "
        "season-to-date",
        "  mean (`std_`), last-three-game mean (`l3_`) of each quantity, games played, week. "
        "Feature",
        "  sets: " + "; ".join(f"`{k}` ({len(v)} columns)" for k, v in FEATURE_SETS.items()) + ".",
        "  `HIST+NGS` adds the official NGS weekly `avg_separation` / `avg_cushion` /",
        "  `avg_intended_air_yards` (published for receivers with >= 5 targets; NaN otherwise) "
        "as a",
        "  privileged reference. Baselines: `BASE_STD` / `BASE_L3` predict the season-to-date /",
        "  last-three mean of the target; `BASE_GLOBAL` the training seasons' mean.",
        "* **Populations.** " + "; ".join(f"`{k}` = {v}" for k, v in POPULATIONS.items()) + ".",
        "* **Split.** Rolling origin: test seasons "
        + ", ".join(str(s) for s in TEST_SEASONS)
        + ", each",
        "  predicted by LightGBM trained on the seasons before it (rounds by early stopping on the",
        "  last training season, refit on every training season with that round count; 2020 uses",
        "  2018-2019 with 2019 as the validation season). Metrics pooled over the three test "
        "seasons",
        "  and per season.",
        f"* **Statistics.** Paired bootstrap ({cfg.n_boot:,} resamples, seed {cfg.seed}) of "
        "per-row",
        "  squared error and absolute error against `HIST` (delta = loss(HIST) - loss(model),",
        "  positive = the model is better; `delta_r2` = delta squared error / test variance), "
        "with a",
        "  (season, week)-clustered interval; `refit_spread_se` = max - min of the mean squared "
        "error",
        "  of the pair's members under seeds "
        + ", ".join(str(s) for s in (cfg.seed, *cfg.refit_seeds)),
        "  (`HIST`, `HIST+IMP_F1T`, `HIST+IMP_ALL` are refit); a delta inside it is noise "
        "whatever its",
        "  interval says.",
        "* **Leakage.** The imputed columns come from students trained on the 2017 tracked games "
        "only;",
        "  no feature uses a week later than t; NGS fields of weeks <= t are after-the-fact for "
        "those",
        "  weeks but precede the target game.",
        "",
        "## Sample",
        "",
        md_table(counts, floatfmt="{:.3f}"),
        "",
    ]
    for target in TARGETS:
        lines += [f"## {target}", ""]
        for pop in POPULATIONS:
            mm = m[(m["target"] == target) & (m["population"] == pop)]
            dd = d[(d["target"] == target) & (d["population"] == pop)]
            if mm.empty:
                continue
            pooled = mm[mm["scope"] == "pooled"][["model", "n", "r2", "mae", "rmse", "bias"]]
            lines += [
                f"### population `{pop}` ({POPULATIONS[pop]})",
                "",
                "Pooled test seasons 2020-2022:",
                "",
                md_table(pooled, floatfmt="{:.4f}"),
                "",
                "Deltas vs `HIST` (pooled test rows):",
                "",
                _deltas_md(dd),
                "",
                "R2 per test season:",
                "",
                md_table(
                    mm[mm["scope"] != "pooled"]
                    .pivot(index="model", columns="scope", values="r2")
                    .reset_index(),
                    floatfmt="{:.4f}",
                ),
                "",
            ]
    lines += [
        "## Refit noise",
        "",
        md_table(refit, floatfmt="{:.4f}"),
        "",
        "## Caveats",
        "",
        "* The target is the receiver's next game with >= 1 target, so rows are conditioned on the",
        "  player being targeted again (injuries, benchings and byes are not modelled); a props "
        "line",
        "  is only posted for players expected to play, which the `regular` population "
        "approximates.",
        "* No market lines are on disk, so this measures predictive skill against event-only "
        "history,",
        "  not against a bookmaker's number; a gain that is inside the refit spread or whose "
        "clustered",
        "  interval covers zero would not move a price.",
        "* Season-to-date aggregates restart every season (no prior-season carry-over), which "
        "limits",
        "  every model equally in the first weeks; `n_games` and `week` let the model weight them.",
        "* The F1 student reads the outcome flags of the plays it aggregates (after the fact); "
        "it is",
        "  admissible here because those plays precede the target game, but it encodes past "
        "outcomes,",
        "  not tracking.",
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
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    run(PropsConfig(n_jobs=args.n_jobs, n_boot=args.n_boot, write=not args.no_write))


if __name__ == "__main__":
    main()
