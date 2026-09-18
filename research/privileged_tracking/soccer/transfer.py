"""Soccer 04 - transfer of the 360-trained students to seasons without 360.

Does the event-only defensive-state student (stage 02, trained on 2020-2025 tournaments and a
few single-club 360 seasons) transfer to event-only club football (2015/16 Premier League, La
Liga, Serie A, Ligue 1) and to the non-360 tournaments (World Cup 2018, Copa America 2024, AFCON
2023)?

(a) Oracle check under domain shift: imputed shot state (``imp_*__E2`` from
    ``imputed_no360.parquet``) vs the ``shot.freeze_frame`` geometry (``sff_*``), which exists for
    every shot, by league / season, with the same numbers on the 360 out-of-fold shots for
    reference (and against the 360 frame itself, the students' native label).
(b) Goal prediction on 2015/16 shots: match-grouped 5-fold CV inside 2015/16 (students frozen),
    EVENT vs EVENT+IMP vs EVENT+ORACLESHOT (+ LOC, EVENT+IMP(E0), EVENT+ORACLESHOT_full,
    StatsBomb xG), pooled and per league; the stage-03 xG models applied zero-shot as a second
    kind of transfer.
(c) Pass completion in 2015/16 (match-stratified ~150k passes): EVENT vs EVENT+IMP, with and
    without the realised pass trajectory (no pass oracle exists without 360).
(d) Team-level sanity: imputed block depth / defensive line conceded per team-match and
    team-season vs event-only style proxies (``team_match.parquet``: ppda, def_line_x, passes
    allowed), Spearman correlations; distribution of the imputed quantities by league vs the 360
    domain.
(e) One worked example: imputed block depth per minute in one 2015/16 match next to the
    opponent's observed defensive-action mean x.

Stages (repo root; each caches under ``processed_dir('soccer')/transfer_cache/``)::

    python -m research.privileged_tracking.soccer.transfer --stage shots    # shot table + assists
    python -m research.privileged_tracking.soccer.transfer --stage xg       # (b)
    python -m research.privileged_tracking.soccer.transfer --stage passes   # (c) pass table
    python -m research.privileged_tracking.soccer.transfer --stage xpass    # (c)
    python -m research.privileged_tracking.soccer.transfer --stage report   # (a), (d), (e), report

``--stage all`` runs them in order; ``--smoke`` uses a few matches per domain in a separate cache
and writes nothing; ``--force`` refits a stage.

Hygiene: every student is frozen from the 360 domain (a disjoint set of matches), so the imputed
features of the 2015/16 rows are never in-sample; the downstream xG / xPass models are cross-
validated by match inside the target domain; the event-only baseline receives every ``f_*`` field
the students read (incl. ``f_after_duration`` in the with-after xPass design); paired bootstrap
CIs for every model-vs-model claim.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from pathlib import Path

from research.privileged_tracking.common.io import processed_dir, reports_dir, sb_dir
from research.privileged_tracking.common.metrics import calibration_table, mae, r2
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.soccer import apply_student as aps
from research.privileged_tracking.soccer import build_tables as bt
from research.privileged_tracking.soccer import imputation as imp
from research.privileged_tracking.soccer import imputation_features as imf
from research.privileged_tracking.soccer import payoff as po
from research.privileged_tracking.soccer import payoff_features as pf
from research.privileged_tracking.soccer.payoff_features import (
    ASSIST_COLS,
    ASSIST_STATE,
    PASS_STATE,
    SHOT_STATE,
    XG_REFERENCE,
    GbmParams,
)

CACHE_SUBDIR = "transfer_cache"
SHOT_TABLE = "transfer_shots.parquet"
PASS_TABLE = "transfer_passes.parquet"
KEY_PASS_LINKS = "shot_key_pass_no360.parquet"
KEY_PASSES = "key_passes_no360.parquet"
XG_PREDS = "xg_preds.parquet"
XG_FITS = "xg_fits.parquet"
XPASS_PREDS = "xpass_preds.parquet"
XPASS_FITS = "xpass_fits.parquet"
RECEIVER_FITS = "receiver_student_fits.parquet"
REPORT_PREFIX = "soccer_04"

#: (competition, season) -> domain label of the non-360 matches analysed here
DOMAIN_LABELS: dict[tuple[str, str], str] = {
    ("Premier League", "2015/2016"): "PL 2015/16",
    ("La Liga", "2015/2016"): "La Liga 2015/16",
    ("Serie A", "2015/2016"): "Serie A 2015/16",
    ("Ligue 1", "2015/2016"): "Ligue 1 2015/16",
    ("FIFA World Cup", "2018"): "WC 2018",
    ("Copa America", "2024"): "Copa 2024",
    ("African Cup of Nations", "2023"): "AFCON 2023",
}
LEAGUE_DOMAINS: tuple[str, ...] = (
    "PL 2015/16",
    "La Liga 2015/16",
    "Serie A 2015/16",
    "Ligue 1 2015/16",
)
TOURNAMENT_DOMAINS: tuple[str, ...] = ("WC 2018", "Copa 2024", "AFCON 2023")
POP_LEAGUES = "leagues_2015/16"
POP_TOURNAMENTS = "tournaments_no360"
OTHER = "other"

#: xG variants cross-validated inside the pooled target populations / per league
XG_VARIANTS_POOLED: tuple[str, ...] = (
    "LOC",
    "EVENT",
    "EVENT+IMP",
    "EVENT+IMP(E0)",
    "EVENT+ORACLESHOT",
    "EVENT+ORACLESHOT_full",
)
XG_VARIANTS_LEAGUE: tuple[str, ...] = ("EVENT", "EVENT+IMP", "EVENT+ORACLESHOT")
ZERO_SHOT_SUFFIX = " (zero-shot)"
XPASS_VARIANTS_HERE: tuple[str, ...] = ("EVENT", "EVENT-nodur", "EVENT+IMP")

#: (target, shot.freeze_frame column, kind) compared in the oracle check
ORACLE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("n_opp_in_cone", "sff_n_opp_in_cone", "count"),
    ("nearest_opp_dist_in_cone", "sff_nearest_opp_dist_in_cone", "reg"),
    ("opp_keeper_dist_to_goal_line", "sff_opp_keeper_dist_to_goal_line", "reg"),
    ("n_opp_within_5", "sff_n_opp_within_5", "count"),
)
#: 360 labels of the same quantities (the students' native supervision)
ORACLE_PAIRS_360: tuple[tuple[str, str, str], ...] = tuple(
    (t, f"y_{t}", k) for t, _, k in ORACLE_PAIRS
)
#: team-shape quantities aggregated per defending team in (d)
TEAM_TARGETS: tuple[str, ...] = ("block_depth", "def_line")
#: event-only style proxies of the defending team (``team_match.parquet``)
TEAM_PROXIES: tuple[str, ...] = ("def_line_x", "ppda", "ppda_passes_allowed", "possession")
#: quantities whose distribution is compared across domains (Pass / Carry possession rows)
SHIFT_TARGETS_MOVES: tuple[str, ...] = (
    "block_depth",
    "def_line",
    "n_opp_ahead_of_ball",
    "n_opp_within_5",
    "nearest_opp_dist",
)
#: the same for Shot rows
SHIFT_TARGETS_SHOTS: tuple[str, ...] = (
    "n_opp_in_cone",
    "nearest_opp_dist_in_cone",
    "opp_keeper_dist_to_goal_line",
    "n_opp_within_5",
    "block_depth",
)
#: defensive-action types of the observed per-minute series in (e) (as ``sb_parse``)
DEF_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "Pressure",
        "Duel",
        "Interception",
        "Block",
        "Clearance",
        "Foul Committed",
        "Ball Recovery",
        "50/50",
    }
)

SHOT_ID_COLS = list(po.SHOT_ID_COLS)
XG_CONTEXT = [
    "event_id",
    "match_id",
    "domain",
    "competition",
    "season",
    "f_play_pattern",
    "f_dist_goal",
    "f_x",
    "f_y",
    "f_shot_type",
    "sff_present",
    "a_has_assist",
    "oracle_statsbomb_xg",
    "is_goal",
]
XPASS_CONTEXT = [
    "event_id",
    "match_id",
    "domain",
    "f_play_pattern",
    "f_pass_type",
    "f_after_pass_height",
    "f_after_pass_length",
    "f_x",
    "f_under_pressure",
    "is_complete",
]


@dataclass
class TransferConfig:
    """Driver configuration (stage 04).

    Attributes:
        n_jobs: LightGBM threads. workers: processes for the key-pass rebuild.
        seed: base seed (folds, subsamples, LightGBM).
        n_folds: match-grouped folds inside every target population.
        n_seeds_xg / n_seeds_league / n_seeds_xpass: seeds averaged per variant (pooled xG,
            per-league xG, xPass).
        xg / xpass: :class:`GbmParams` (stage-03 defaults).
        xpass_target_n: expected size of the match-stratified 2015/16 pass subsample.
        receiver_train_cap: training rows of the receiver-distance student (360 Pass rows).
        n_boot / n_boot_pass: bootstrap resamples for shot / pass deltas.
        exclude_penalties: drop ``f_shot_type == 'Penalty'``.
        example_match: match id of the worked example (None = chosen by rule, see
            :func:`choose_example_match`).
        smoke: first ``smoke_matches`` matches per domain, separate cache, nothing written.
        write: write report / parquet outputs.
    """

    n_jobs: int = 2
    workers: int = 2
    seed: int = 0
    n_folds: int = 5
    n_seeds_xg: int = 3
    n_seeds_league: int = 2
    n_seeds_xpass: int = 1
    xg: GbmParams = field(default_factory=GbmParams)
    xpass: GbmParams = field(default_factory=lambda: po.PayoffConfig().xpass)
    xpass_target_n: int = 150_000
    receiver_train_cap: int = 300_000
    n_boot: int = 2000
    n_boot_pass: int = 500
    exclude_penalties: bool = True
    example_match: int | None = None
    smoke: bool = False
    smoke_matches: int = 30
    write: bool = True


def cache_dir(cfg: TransferConfig) -> Path:
    d = processed_dir("soccer") / (CACHE_SUBDIR + ("_smoke" if cfg.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _events_no360() -> Path:
    return processed_dir("soccer") / "events_no360.parquet"


def _events_360() -> Path:
    return processed_dir("soccer") / "events360.parquet"


def _imputed_no360() -> Path:
    return processed_dir("soccer") / "imputed_no360.parquet"


def _imputed_oof() -> Path:
    return processed_dir("soccer") / "imputed_oof.parquet"


def _team_match() -> Path:
    return sb_dir() / "processed" / "team_match.parquet"


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def domain_of(competition: np.ndarray | pd.Series, season: np.ndarray | pd.Series) -> np.ndarray:
    """Domain label per row ``[n]`` (:data:`DOMAIN_LABELS`; ``other`` for the stray non-360
    matches of other competitions)."""
    c = np.asarray(competition, dtype=object)
    s = np.asarray(season, dtype=object)
    return np.array(
        [DOMAIN_LABELS.get((str(a), str(b)), OTHER) for a, b in zip(c, s, strict=True)],
        dtype=object,
    )


def population_of(domain: np.ndarray) -> np.ndarray:
    """``leagues_2015/16`` / ``tournaments_no360`` / ``other`` per row ``[n]``."""
    d = np.asarray(domain, dtype=object)
    out = np.full(len(d), OTHER, dtype=object)
    out[np.isin(d, list(LEAGUE_DOMAINS))] = POP_LEAGUES
    out[np.isin(d, list(TOURNAMENT_DOMAINS))] = POP_TOURNAMENTS
    return out


def fold_vector(match_id: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row ``[n]`` from :func:`common.splits.group_kfold` (whole matches per fold)."""
    m = np.asarray(match_id)
    out = np.full(len(m), -1, dtype=int)
    for k, (_, te) in enumerate(group_kfold(m, n_splits=n_splits, seed=seed)):
        out[te] = k
    return out


def complete_design(design: pd.DataFrame, full: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """``design`` restricted / extended to ``features`` ``[n, len(features)]``.

    Columns missing from ``design`` (typically constants dropped by
    :func:`payoff_features.drop_constant_columns`, e.g. ``f_gender`` in a single-gender domain)
    are taken from ``full`` (an undropped design of the same rows) or NaN.
    """
    out = design.copy()
    for c in features:
        if c not in out.columns:
            out[c] = full[c].to_numpy() if c in full.columns else np.nan
    return out[features]


def _within(imp_v: np.ndarray, obs: np.ndarray, tol: float) -> float:
    return float(np.mean(np.abs(np.rint(imp_v) - obs) <= tol))


def oracle_check_table(
    df: pd.DataFrame,
    groups: np.ndarray,
    pairs: tuple[tuple[str, str, str], ...] = ORACLE_PAIRS,
    imp_fmt: str = "imp_{}__E2",
    observed: str = "shot.freeze_frame",
) -> pd.DataFrame:
    """Imputed vs observed per (group, target): n, R2, MAE, bias, correlation, means, sd of the
    observed value, and for counts the exact / within-1 agreement of the rounded imputation.

    Args:
        df: rows holding ``imp_fmt.format(target)`` and the observed column of every pair.
        groups: group label per row ``[n]`` (league / season / domain).
        pairs: ``(target, observed column, kind)``; imputations are clipped at 0 for counts /
            distances (:func:`payoff_features.clip_state`).
    """
    rows: list[dict[str, Any]] = []
    g = np.asarray(groups, dtype=object)
    for target, obs_col, kind in pairs:
        ic = imp_fmt.format(target)
        if ic not in df.columns or obs_col not in df.columns:
            continue
        p_all = pf.clip_state(df[ic].to_numpy(dtype=np.float32), target).astype(float)
        o_all = df[obs_col].to_numpy(dtype=float)
        for lev in pd.unique(g):
            m = (g == lev) & ~np.isnan(p_all) & ~np.isnan(o_all)
            n = int(m.sum())
            if n < 20:
                continue
            p, o = p_all[m], o_all[m]
            row: dict[str, Any] = {
                "group": lev,
                "target": target,
                "observed": observed,
                "kind": kind,
                "n": n,
                "r2": r2(o, p),
                "mae": mae(o, p),
                "bias": float(p.mean() - o.mean()),
                "corr": float(np.corrcoef(o, p)[0, 1]) if p.std() > 0 and o.std() > 0 else np.nan,
                "mean_imp": float(p.mean()),
                "mean_obs": float(o.mean()),
                "sd_obs": float(o.std()),
                "sd_imp": float(p.std()),
            }
            if kind == "count":
                row["exact"] = _within(p, o, 0.0)
                row["within_1"] = _within(p, o, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def oracle_calibration(
    df: pd.DataFrame,
    groups: np.ndarray,
    pairs: tuple[tuple[str, str, str], ...] = ORACLE_PAIRS,
    imp_fmt: str = "imp_{}__E2",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Equal-count bins of the imputed value: mean imputed vs mean observed and n per bin, per
    (group, target) - the regression analogue of a reliability diagram."""
    rows: list[dict[str, Any]] = []
    g = np.asarray(groups, dtype=object)
    for target, obs_col, _ in pairs:
        ic = imp_fmt.format(target)
        if ic not in df.columns or obs_col not in df.columns:
            continue
        p_all = pf.clip_state(df[ic].to_numpy(dtype=np.float32), target).astype(float)
        o_all = df[obs_col].to_numpy(dtype=float)
        for lev in pd.unique(g):
            m = (g == lev) & ~np.isnan(p_all) & ~np.isnan(o_all)
            if m.sum() < 10 * n_bins:
                continue
            for i, r in enumerate(calibration_table(o_all[m], p_all[m], n_bins)):
                rows.append(
                    {
                        "group": lev,
                        "target": target,
                        "bin": i,
                        "imputed": r["pred"],
                        "observed": r["obs"],
                        "n": int(r["n"]),
                    }
                )
    return pd.DataFrame(rows)


def team_block_table(rows: pd.DataFrame, team_match: pd.DataFrame) -> pd.DataFrame:
    """Imputed defensive shape conceded per (match, defending team) joined to that team's
    event-only style proxies.

    Args:
        rows: possession-team Pass / Carry rows with ``match_id``, ``opp_team_id`` (the defending
            team), ``f_x`` (ball x in the attacking frame) and ``imp_<t>__{E0,E2}`` for
            :data:`TEAM_TARGETS`.
        team_match: ``team_match.parquet`` rows (``match_id``, ``team_id`` and
            :data:`TEAM_PROXIES`).

    Returns:
        one row per (match_id, team_id) with ``n_rows``, ``ball_x_mean`` (mean attacking-frame x
        of the opponent's events, a location-only proxy), ``imp_<t>__<fset>`` means and the proxies.
    """
    agg: dict[str, tuple[str, str]] = {"n_rows": ("f_x", "size"), "ball_x_mean": ("f_x", "mean")}
    for t in TEAM_TARGETS:
        for fset in ("E0", "E2"):
            c = f"imp_{t}__{fset}"
            if c in rows.columns:
                agg[c] = (c, "mean")
    per = rows.groupby(["match_id", "opp_team_id"]).agg(**agg).reset_index()
    per = per.rename(columns={"opp_team_id": "team_id"})
    keep = ["match_id", "team_id", *[c for c in TEAM_PROXIES if c in team_match.columns]]
    if "team" in team_match.columns:
        keep.append("team")
    return per.merge(team_match[keep], on=["match_id", "team_id"], how="inner")


def team_season_table(block: pd.DataFrame, min_matches: int = 5) -> pd.DataFrame:
    """Team-season means of :func:`team_block_table` (``block`` must carry ``competition`` and
    ``season``); teams with fewer than ``min_matches`` matches are dropped."""
    num = [c for c in block.columns if c.startswith(("imp_", "ball_x_mean")) or c in TEAM_PROXIES]
    keys = ["competition", "season", "team_id"] + (["team"] if "team" in block.columns else [])
    g = block.groupby(keys)
    out = g[num].mean().reset_index()
    out["n_matches"] = g.size().to_numpy()
    out["n_rows"] = g["n_rows"].sum().to_numpy()
    return out[out["n_matches"] >= min_matches].reset_index(drop=True)


def spearman_table(
    df: pd.DataFrame, imputed_cols: list[str], proxy_cols: list[str], level: str
) -> pd.DataFrame:
    """Spearman rank correlation of every imputed column with every proxy column."""
    rows = []
    for ic in imputed_cols:
        if ic not in df.columns:
            continue
        for pc in proxy_cols:
            if pc not in df.columns:
                continue
            a = df[ic].to_numpy(dtype=float)
            b = df[pc].to_numpy(dtype=float)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 5:
                continue
            rho, pval = spearmanr(a[ok], b[ok])
            rows.append(
                {
                    "level": level,
                    "imputed": ic,
                    "proxy": pc,
                    "n": int(ok.sum()),
                    "spearman": float(rho),
                    "p_value": float(pval),
                }
            )
    return pd.DataFrame(rows)


def def_action_x_by_minute(events: list[dict[str, Any]], team_id: int) -> pd.DataFrame:
    """Per-minute mean x of one team's located defensive actions (:data:`DEF_ACTION_TYPES`;
    StatsBomb locations are in the acting team's attacking frame, so x is the distance from that
    team's own goal line) ``[n_minutes, 3]`` with columns ``minute, def_x_mean, n_def``."""
    rows = []
    for e in events:
        if (e.get("type") or {}).get("name") not in DEF_ACTION_TYPES:
            continue
        if (e.get("team") or {}).get("id") != team_id:
            continue
        loc = e.get("location")
        if not loc or len(loc) < 2:
            continue
        rows.append({"minute": int(e.get("minute", 0)), "x": float(loc[0])})
    if not rows:
        return pd.DataFrame(columns=["minute", "def_x_mean", "n_def"])
    d = pd.DataFrame(rows)
    out = d.groupby("minute").agg(def_x_mean=("x", "mean"), n_def=("x", "size")).reset_index()
    return out


def imputed_block_by_minute(rows: pd.DataFrame, defending_team_id: int) -> pd.DataFrame:
    """Per-minute mean imputed block depth / defensive line of ``defending_team_id`` from the
    opponent's possession rows (``opp_team_id == defending_team_id``) ``[n_minutes, 4]`` with
    columns ``minute, imp_block_depth, imp_def_line, n_rows``."""
    r = rows[
        (rows["opp_team_id"].to_numpy() == defending_team_id)
        & rows["f_is_possession_team"].to_numpy(dtype=bool)
    ]
    r = r[~np.isnan(r["imp_block_depth__E2"].to_numpy(dtype=float))]
    if r.empty:
        return pd.DataFrame(columns=["minute", "imp_block_depth", "imp_def_line", "n_rows"])
    out = (
        r.assign(minute=r["f_minute"].to_numpy(dtype=float).astype(int))
        .groupby("minute")
        .agg(
            imp_block_depth=("imp_block_depth__E2", "mean"),
            imp_def_line=("imp_def_line__E2", "mean"),
            n_rows=("imp_block_depth__E2", "size"),
        )
        .reset_index()
    )
    return out


def minute_profile(
    rows: pd.DataFrame, events: list[dict[str, Any]], teams: dict[int, str]
) -> pd.DataFrame:
    """Long per-minute table of one match: for every team, its imputed block depth (from the
    opponent's possession rows) next to its observed defensive-action mean x.

    Returns:
        columns ``team_id, team, minute, imp_block_depth, imp_def_line, n_rows, def_x_mean,
        n_def`` (NaN where a minute has no rows / actions).
    """
    frames = []
    for tid, name in teams.items():
        a = imputed_block_by_minute(rows, tid)
        b = def_action_x_by_minute(events, tid)
        minutes = sorted(set(a["minute"].tolist()) | set(b["minute"].tolist()))
        m = pd.DataFrame({"minute": minutes})
        m = m.merge(a, on="minute", how="left").merge(b, on="minute", how="left")
        m.insert(0, "team", name)
        m.insert(0, "team_id", tid)
        frames.append(m)
    if not frames:
        return pd.DataFrame(
            columns=[
                "team_id",
                "team",
                "minute",
                "imp_block_depth",
                "imp_def_line",
                "n_rows",
                "def_x_mean",
                "n_def",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def bin_profile(profile: pd.DataFrame, width: int = 5) -> pd.DataFrame:
    """Row-weighted ``width``-minute bins of :func:`minute_profile` (weights: n_rows / n_def)."""
    p = profile.copy()
    p["bin_start"] = (p["minute"] // width) * width
    rows = []
    for (tid, name, b), g in p.groupby(["team_id", "team", "bin_start"], sort=True):
        w_imp = g["n_rows"].fillna(0).to_numpy(dtype=float)
        w_def = g["n_def"].fillna(0).to_numpy(dtype=float)
        bd = g["imp_block_depth"].to_numpy(dtype=float)
        dl = g["imp_def_line"].to_numpy(dtype=float)
        dx = g["def_x_mean"].to_numpy(dtype=float)
        rows.append(
            {
                "team_id": tid,
                "team": name,
                "minutes": f"{int(b)}-{int(b) + width}",
                "imp_block_depth": float(np.nansum(bd * w_imp) / w_imp.sum())
                if w_imp.sum()
                else np.nan,
                "imp_def_line": float(np.nansum(dl * w_imp) / w_imp.sum())
                if w_imp.sum()
                else np.nan,
                "n_rows": int(w_imp.sum()),
                "def_x_mean": float(np.nansum(dx * w_def) / w_def.sum()) if w_def.sum() else np.nan,
                "n_def": int(w_def.sum()),
            }
        )
    return pd.DataFrame(rows)


def profile_correlation(profile: pd.DataFrame, value_col: str = "imp_block_depth") -> pd.DataFrame:
    """Per-team Pearson / Spearman correlation between the imputed series and the observed
    defensive-action x over the minutes where both exist."""
    rows = []
    for (tid, name), g in profile.groupby(["team_id", "team"], sort=True):
        a = g[value_col].to_numpy(dtype=float)
        b = g["def_x_mean"].to_numpy(dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        row = {"team_id": tid, "team": name, "n_minutes": int(ok.sum())}
        if ok.sum() >= 5:
            row["pearson"] = float(np.corrcoef(a[ok], b[ok])[0, 1])
            row["spearman"] = float(spearmanr(a[ok], b[ok])[0])
            row["mean_imputed"] = float(a[ok].mean())
            row["mean_observed"] = float(b[ok].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def choose_example_match(rows: pd.DataFrame, domain: str = "PL 2015/16") -> int:
    """Rule: the match of ``domain`` with the most imputed possession Pass / Carry rows."""
    r = rows[(rows["domain"].to_numpy(dtype=object) == domain)]
    if r.empty:
        r = rows
    counts = r.groupby("match_id").size()
    return int(counts.idxmax())


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def _smoke_filter(df: pd.DataFrame, cfg: TransferConfig) -> pd.DataFrame:
    """First ``cfg.smoke_matches`` matches of every domain."""
    if not cfg.smoke:
        return df
    dom = df["domain"].to_numpy(dtype=object)
    keep = np.zeros(len(df), dtype=bool)
    for d in pd.unique(dom):
        mids = np.sort(df.loc[dom == d, "match_id"].unique())[: cfg.smoke_matches]
        keep |= (dom == d) & df["match_id"].isin(mids).to_numpy()
    return df[keep].reset_index(drop=True)


def load_key_pass_links(cfg: TransferConfig, match_ids: np.ndarray) -> pd.DataFrame:
    """``(event_id, key_pass_id, match_id)`` for every shot of the given non-360 matches, read
    from the raw StatsBomb events and cached (separate from the stage-03 cache)."""
    path = cache_dir(cfg) / KEY_PASS_LINKS
    if path.exists():
        d = pd.read_parquet(path)
        if set(int(m) for m in match_ids) <= set(d["match_id"].unique()):
            return d[d["match_id"].isin(match_ids)]
    frames = []
    for mid in sorted(int(m) for m in match_ids):
        p = sb_dir() / "events" / f"{mid}.json"
        if not p.exists():
            continue
        with open(p) as f:
            ev = json.load(f)
        d = pf.key_pass_links(ev)
        d["match_id"] = mid
        frames.append(d)
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["event_id", "key_pass_id", "match_id"])
    )
    out.to_parquet(path, index=False)
    return out


def _key_pass_rows(args: tuple[bt.MatchJob, tuple[str, ...]]) -> pd.DataFrame | None:
    """Worker: rebuild every Pass row of one non-360 match (build stage, ``frac = 1``) and keep
    the key passes only."""
    job, ids = args
    table, _ = bt.process_match(job)
    if table is None:
        return None
    df = table.to_pandas()
    df = df[df["event_id"].isin(ids)]
    return df if len(df) else None


def build_key_passes(cfg: TransferConfig, links: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Build-stage rows of every key pass of the non-360 shots (cached ``key_passes_no360``).

    ``events_no360.parquet`` keeps only a fixed 25% of the passes, so the key passes are rebuilt
    with :func:`build_tables.process_match` (all Pass rows of the match, ``frac = 1``), which
    recomputes the same event-only features and sequence block as the build stage.
    """
    path = cache_dir(cfg) / KEY_PASSES
    need = links.dropna(subset=["key_pass_id"])
    if path.exists() and not force:
        d = pd.read_parquet(path)
        if set(need["match_id"].unique()) <= set(d["match_id"].unique()):
            return d
    t0 = time.time()
    by_match = need.groupby("match_id")["key_pass_id"].apply(lambda s: tuple(s.astype(str)))
    jobs = bt.make_jobs(bt.BuildConfig(no360_pass_carry_frac=1.0, no360_types=("Pass",)))
    job_of = {j.match_id: j for j in jobs}
    work = [(job_of[int(m)], ids) for m, ids in by_match.items() if int(m) in job_of]
    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for i, d in enumerate(ex.map(_key_pass_rows, work, chunksize=8)):
            if d is not None:
                frames.append(d)
            if (i + 1) % 200 == 0:
                _log(f"  key passes: {i + 1}/{len(work)} matches, {time.time() - t0:.0f} s")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(path, index=False)
    _log(
        f"key passes rebuilt: {len(out):,} of {len(need):,} links found in {len(work):,} "
        f"matches, {time.time() - t0:.0f} s"
    )
    return out


def build_shot_table(cfg: TransferConfig, force: bool = False) -> pd.DataFrame:
    """Shots of the non-360 matches with event features, shot-frame geometry, frozen-student
    imputations (``imputed_no360.parquet``), the key pass's attributes and imputations, and
    NaN 360 columns so that :func:`payoff.xg_designs` applies unchanged (cached)."""
    path = cache_dir(cfg) / SHOT_TABLE
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    ev = _events_no360()
    names = pq.ParquetFile(ev).schema_arrow.names
    cols = (
        SHOT_ID_COLS
        + [c for c in names if c.startswith(("f_", "sff_", "oracle_"))]
        + ["post_shot_outcome"]
    )
    shots = pq.read_table(ev, columns=cols, filters=[("f_type", "==", "Shot")]).to_pandas()
    shots["domain"] = domain_of(shots["competition"], shots["season"])
    shots = shots[shots["domain"] != OTHER]
    shots = _smoke_filter(
        shots.sort_values(["match_id", "event_index"]).reset_index(drop=True), cfg
    )
    imp_cols = ["event_id"] + [f"imp_{t}__{f}" for t in SHOT_STATE for f in ("E0", "E2")]
    im = pq.read_table(
        _imputed_no360(), columns=imp_cols, filters=[("f_type", "==", "Shot")]
    ).to_pandas()
    shots = shots.merge(im, on="event_id", how="left")
    for t in SHOT_STATE:
        shots[f"y_{t}"] = np.float32(np.nan)
    links = load_key_pass_links(cfg, shots["match_id"].unique())
    shots = shots.merge(links[["event_id", "key_pass_id"]], on="event_id", how="left")
    kp = build_key_passes(cfg, links[links["event_id"].isin(shots["event_id"])])
    if len(kp):
        kp = kp[kp["event_id"].isin(set(shots["key_pass_id"].dropna().astype(str)))]
        assist = pf.assist_block(shots, kp[["event_id", *ASSIST_COLS]])
    else:
        assist = pf.assist_block(shots, pd.DataFrame(columns=["event_id", *ASSIST_COLS]))
    shots = pd.concat([shots, assist], axis=1)
    if len(kp):
        kpi = aps.impute(kp, ("E0", "E2a"))
        keep = ["event_id"] + [f"imp_{t}__{f}" for t in ASSIST_STATE for f in ("E0", "E2a")]
        kpi = kpi[keep].rename(
            columns={
                **{f"imp_{t}__{f}": f"a_imp_{t}__{f}" for t in ASSIST_STATE for f in ("E0", "E2a")},
                "event_id": "key_pass_id",
            }
        )
        shots = shots.merge(kpi, on="key_pass_id", how="left")
    else:
        for t in ASSIST_STATE:
            for f in ("E0", "E2a"):
                shots[f"a_imp_{t}__{f}"] = np.float32(np.nan)
    for t in ASSIST_STATE:
        shots[f"a_y_{t}"] = np.float32(np.nan)
    shots["is_goal"] = (shots["post_shot_outcome"].to_numpy(dtype=object) == "Goal").astype(
        np.float32
    )
    shots.to_parquet(path, index=False)
    _log(
        f"shot table: {len(shots):,} shots, {int(shots['is_goal'].sum()):,} goals, "
        f"{int(shots['a_has_assist'].sum()):,} with an assist "
        f"({int(shots['key_pass_id'].notna().sum()):,} key-pass links), {time.time() - t0:.0f} s"
    )
    return shots


def shot_population(cfg: TransferConfig, shots: pd.DataFrame) -> pd.DataFrame:
    if cfg.exclude_penalties:
        shots = shots[shots["f_shot_type"].to_numpy(dtype=object) != "Penalty"]
    return shots.reset_index(drop=True)


def shot_designs(shots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """:func:`payoff.xg_designs` without the 360 oracle (all-NaN outside 360 matches)."""
    designs = po.xg_designs(shots)
    designs.pop("EVENT+ORACLE360", None)
    return designs


def full_event_design(shots: pd.DataFrame) -> pd.DataFrame:
    """Undropped event-only design (E2 + encoded assist) used to complete zero-shot designs."""
    assist = shots[
        [c for c in shots.columns if c.startswith("a_") and not c.startswith(("a_imp_", "a_y_"))]
    ]
    return pd.concat([imf.build_design(shots, "E2"), pf.encode_assist(assist)], axis=1)


def zero_shot_predictions(shots: pd.DataFrame, designs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stage-03 xG models (fitted on all 360 shots) applied to ``shots`` ``[n, 1 + k]``."""
    bundle = po.XgBundle.load()
    full = full_event_design(shots)
    out = pd.DataFrame({"event_id": shots["event_id"].to_numpy()})
    for name, model in bundle.models.items():
        if name not in designs:
            continue
        x = complete_design(designs[name], full, model.features)
        out[f"pred_{name}{ZERO_SHOT_SUFFIX}"] = model.predict(x)
    return out


def stage_xg(cfg: TransferConfig, force: bool = False) -> pd.DataFrame:
    """(b) xG cross-validated inside every target population (pooled 2015/16 leagues, pooled
    non-360 tournaments, each league) + zero-shot stage-03 models; writes ``xg_preds.parquet``."""
    path = cache_dir(cfg) / XG_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    shots = shot_population(cfg, build_shot_table(cfg))
    y = shots["is_goal"].to_numpy(dtype=float)
    match = shots["match_id"].to_numpy()
    dom = shots["domain"].to_numpy(dtype=object)
    pop = population_of(dom)
    designs = shot_designs(shots)
    zs = zero_shot_predictions(shots, designs)
    populations: list[tuple[str, np.ndarray, int, tuple[str, ...]]] = [
        (POP_LEAGUES, pop == POP_LEAGUES, cfg.n_seeds_xg, XG_VARIANTS_POOLED),
        (POP_TOURNAMENTS, pop == POP_TOURNAMENTS, cfg.n_seeds_xg, XG_VARIANTS_LEAGUE),
    ] + [(d, dom == d, cfg.n_seeds_league, XG_VARIANTS_LEAGUE) for d in LEAGUE_DOMAINS]
    frames, fits = [], []
    for name, mask, n_seeds, variants in populations:
        idx = np.where(mask)[0]
        if len(idx) < 200 or len(np.unique(match[idx])) < cfg.n_folds:
            _log(f"xg {name}: {len(idx)} shots, skipped")
            continue
        fold = fold_vector(match[idx], cfg.n_folds, cfg.seed)
        out = shots.iloc[idx][XG_CONTEXT].copy()
        out["population"] = name
        out["fold"] = fold
        out["pred_BASE"] = po.base_rate_oof(y[idx], fold)
        out[f"pred_{XG_REFERENCE}"] = shots["oracle_statsbomb_xg"].to_numpy(dtype=float)[idx]
        for v in variants:
            t0 = time.time()
            x = designs[v].iloc[idx].reset_index(drop=True)
            x = pf.drop_constant_columns(x)
            oof, f, _ = po.cv_predict(
                x,
                y[idx],
                match[idx],
                fold,
                cfg.xg,
                tuple(cfg.seed + i for i in range(n_seeds)),
                pf.categorical_in(x),
                "binary",
                cfg.n_jobs,
                refit=True,
            )
            out[f"pred_{v}"] = oof
            fits += [{"population": name, "variant": v, **r} for r in f]
            m = pf.binary_metrics(y[idx], oof)
            _log(
                f"xg {name:18s} {v:22s} d={x.shape[1]:3d} n={len(idx):6d} "
                f"log-loss {m['log_loss']:.4f} auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
            )
        out = out.merge(zs, on="event_id", how="left")
        frames.append(out)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_parquet(path, index=False)
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / XG_FITS, index=False)
    return preds


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def fit_receiver_student(
    cfg: TransferConfig, passes: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """``nearest_opp_to_receiver`` student (no stage-02 model) fitted on the 360 Pass rows (E2a
    and E2 designs, stage-02 LightGBM settings, ``receiver_train_cap`` rows) and applied to
    ``passes``; returns the two ``imp_nearest_opp_to_receiver__<fset>`` columns and fit metadata.
    The 360 matches are disjoint from every non-360 match, so nothing is in-sample."""
    need = sorted(
        {c for c in imf.design_columns("E2a") if c not in ("f_gender", "f_comp_type")}
        | {"match_id", "gender", "competition", "y_nearest_opp_to_receiver", "y_frame_ok"}
    )
    src = pq.read_table(_events_360(), columns=need, filters=[("f_type", "==", "Pass")]).to_pandas()
    label = src["y_nearest_opp_to_receiver"].to_numpy(dtype=float)
    label[src["y_frame_ok"].to_numpy(dtype=float) != 1.0] = np.nan
    ok = np.where(~np.isnan(label))[0]
    icfg = imp.ImputationConfig(n_jobs=cfg.n_jobs, seed=cfg.seed, train_cap=cfg.receiver_train_cap)
    tr = imp.thin_rows(ok, cfg.receiver_train_cap, cfg.seed)
    match = src["match_id"].to_numpy()
    out = pd.DataFrame(index=passes.index)
    fits = []
    for fset in ("E2a", "E2"):
        t0 = time.time()
        x = imf.build_design(src, fset)
        cats = imf.categorical_columns(fset)
        booster, rounds = imp.fit_lgbm(
            icfg, x.iloc[tr], label[tr], match[tr], "reg", cats, cfg.seed
        )
        xt = imf.build_design(passes, fset)
        out[f"imp_nearest_opp_to_receiver__{fset}"] = booster.predict(
            xt, num_iteration=rounds
        ).astype(np.float32)
        fits.append(
            {
                "target": "nearest_opp_to_receiver",
                "fset": fset,
                "rounds": rounds,
                "n_train": int(len(tr)),
                "seconds": time.time() - t0,
            }
        )
        _log(
            f"receiver student {fset}: {rounds} rounds on {len(tr):,} rows, "
            f"{time.time() - t0:.0f} s"
        )
    return out, fits


def build_pass_table(cfg: TransferConfig, force: bool = False) -> pd.DataFrame:
    """Match-stratified subsample of the 2015/16 pass attempts with event features and frozen
    student imputations of :data:`PASS_STATE` (cached ``transfer_passes.parquet``)."""
    path = cache_dir(cfg) / PASS_TABLE
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    ev = _events_no360()
    names = pq.ParquetFile(ev).schema_arrow.names
    cols = SHOT_ID_COLS + [c for c in names if c.startswith("f_")] + ["post_pass_outcome"]
    passes = pq.read_table(ev, columns=cols, filters=[("f_type", "==", "Pass")]).to_pandas()
    passes["domain"] = domain_of(passes["competition"], passes["season"])
    passes = passes[np.isin(passes["domain"].to_numpy(dtype=object), list(LEAGUE_DOMAINS))]
    passes = _smoke_filter(
        passes.sort_values(["match_id", "event_index"]).reset_index(drop=True), cfg
    )
    y, keep = pf.pass_completion(passes["post_pass_outcome"].to_numpy(dtype=object))
    passes["is_complete"] = y.astype(np.float32)
    passes = passes[keep].reset_index(drop=True)
    n_all = len(passes)
    keep = pf.stratified_subsample(passes["match_id"].to_numpy(), cfg.xpass_target_n, cfg.seed)
    passes = passes[keep].reset_index(drop=True)
    imp_cols = ["event_id"] + [
        f"imp_{t}__{f}" for t in po.PASS_STATE_STAGE02 for f in ("E2", "E2a")
    ]
    im = pq.read_table(
        _imputed_no360(), columns=imp_cols, filters=[("f_type", "==", "Pass")]
    ).to_pandas()
    passes = passes.merge(im, on="event_id", how="left")
    rec, fits = fit_receiver_student(cfg, passes)
    for c in rec.columns:
        passes[c] = rec[c].to_numpy()
    for t in PASS_STATE:
        passes[f"y_{t}"] = np.float32(np.nan)
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / RECEIVER_FITS, index=False)
    passes.to_parquet(path, index=False)
    _log(
        f"pass table: {len(passes):,} of {n_all:,} 2015/16 pass attempts in the subsample, "
        f"completion {passes['is_complete'].mean():.3f}, {time.time() - t0:.0f} s"
    )
    return passes


def pass_designs(passes: pd.DataFrame, with_after: bool) -> dict[str, pd.DataFrame]:
    """:func:`payoff.xpass_designs` without the 360 oracle."""
    designs = po.xpass_designs(passes, with_after)
    designs.pop("EVENT+ORACLE", None)
    return designs


def stage_xpass(cfg: TransferConfig, force: bool = False) -> pd.DataFrame:
    """(c) xPass variants in 2015/16 for the with-after and no-after designs; writes
    ``xpass_preds.parquet``."""
    path = cache_dir(cfg) / XPASS_PREDS
    if path.exists() and not force:
        return pd.read_parquet(path)
    passes = build_pass_table(cfg)
    y = passes["is_complete"].to_numpy(dtype=float)
    match = passes["match_id"].to_numpy()
    fold = fold_vector(match, cfg.n_folds, cfg.seed)
    preds = passes[XPASS_CONTEXT].copy()
    preds["fold"] = fold
    preds["pred_BASE"] = po.base_rate_oof(y, fold)
    fits = []
    for with_after in (True, False):
        tag = "after" if with_after else "noafter"
        designs = pass_designs(passes, with_after)
        for v in XPASS_VARIANTS_HERE:
            if v not in designs:
                continue
            t0 = time.time()
            x = designs[v]
            oof, f, _ = po.cv_predict(
                x,
                y,
                match,
                fold,
                cfg.xpass,
                tuple(cfg.seed + i for i in range(cfg.n_seeds_xpass)),
                pf.categorical_in(x),
                "binary",
                cfg.n_jobs,
                refit=False,
            )
            preds[f"pred_{v}__{tag}"] = oof
            fits += [{"variant": v, "design": tag, **r} for r in f]
            m = pf.binary_metrics(y, oof)
            _log(
                f"xpass {v:12s} {tag:8s} d={x.shape[1]:3d} log-loss {m['log_loss']:.4f} "
                f"auc {m['auc']:.3f}  {time.time() - t0:.0f} s"
            )
    preds.to_parquet(path, index=False)
    pd.DataFrame(fits).to_parquet(cache_dir(cfg) / XPASS_FITS, index=False)
    return preds


# ---------------------------------------------------------------------------
# (a) oracle check, (d) team level, (e) example
# ---------------------------------------------------------------------------


def oracle_tables(cfg: TransferConfig, shots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(a) imputed shot state vs shot.freeze_frame by domain, plus the 360 out-of-fold reference
    (vs shot.freeze_frame and vs the 360 frame)."""
    dom = shots["domain"].to_numpy(dtype=object)
    pop = population_of(dom)
    groups = dom.copy()
    parts = [oracle_check_table(shots, groups)]
    pooled = shots[pop == POP_LEAGUES]
    if len(pooled):
        parts.append(oracle_check_table(pooled, np.full(len(pooled), POP_LEAGUES, dtype=object)))
    cal_groups = np.where(pop == POP_LEAGUES, POP_LEAGUES, dom).astype(object)
    cal = [oracle_calibration(shots, cal_groups)]
    ref_path = processed_dir("soccer") / po.CACHE_SUBDIR / po.SHOT_TABLE
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        ref = ref[ref["f_shot_type"].to_numpy(dtype=object) != "Penalty"].reset_index(drop=True)
        comp = np.where(
            imf.comp_type(ref["competition"]) == 0.0, "club season", "tournament"
        ).astype(object)
        g360 = np.array(
            [f"360 {g} {c} (OOF)" for g, c in zip(ref["gender"], comp, strict=True)], dtype=object
        )
        g_all = np.full(len(ref), "360 all (OOF)", dtype=object)
        parts.append(oracle_check_table(ref, g_all))
        parts.append(oracle_check_table(ref, g360))
        parts.append(oracle_check_table(ref, g_all, ORACLE_PAIRS_360, observed="360 frame"))
        parts.append(oracle_check_table(ref, g360, ORACLE_PAIRS_360, observed="360 frame"))
        cal.append(oracle_calibration(ref, g_all))
    check = pd.concat(parts, ignore_index=True)
    return {"oracle_check": check, "oracle_calibration": pd.concat(cal, ignore_index=True)}


def _move_rows(cfg: TransferConfig) -> pd.DataFrame:
    """Possession-team Pass / Carry rows of the non-360 matches with the E0 / E2 shape
    imputations (ids from ``events_no360``, imputations from ``imputed_no360``)."""
    ev = pq.read_table(
        _events_no360(),
        columns=[
            "match_id",
            "event_id",
            "team_id",
            "opp_team_id",
            "competition",
            "season",
            "f_type",
            "f_is_possession_team",
            "f_x",
            "f_minute",
        ],
        filters=[("f_type", "in", ["Pass", "Carry"])],
    ).to_pandas()
    ev["domain"] = domain_of(ev["competition"], ev["season"])
    ev = ev[(ev["domain"] != OTHER) & ev["f_is_possession_team"].to_numpy(dtype=bool)]
    ev = _smoke_filter(ev.reset_index(drop=True), cfg)
    cols = ["event_id"] + [f"imp_{t}__{f}" for t in SHIFT_TARGETS_MOVES for f in ("E0", "E2")]
    im = pq.read_table(
        _imputed_no360(), columns=cols, filters=[("f_type", "in", ["Pass", "Carry"])]
    ).to_pandas()
    return ev.merge(im, on="event_id", how="left")


def team_tables(cfg: TransferConfig, moves: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(d) team-match / team-season block tables and Spearman correlations (2015/16 leagues)."""
    tm = pd.read_parquet(_team_match())
    rows = moves[np.isin(moves["domain"].to_numpy(dtype=object), list(LEAGUE_DOMAINS))]
    block = team_block_table(rows, tm)
    meta = rows[["match_id", "competition", "season", "domain"]].drop_duplicates("match_id")
    block = block.merge(meta, on="match_id", how="left")
    season = team_season_table(block)
    imputed_cols = [f"imp_{t}__{f}" for t in TEAM_TARGETS for f in ("E2", "E0")] + ["ball_x_mean"]
    sp = pd.concat(
        [
            spearman_table(block, imputed_cols, list(TEAM_PROXIES), "team-match"),
            spearman_table(season, imputed_cols, list(TEAM_PROXIES), "team-season"),
        ],
        ignore_index=True,
    )
    # per-league team-season correlations of the main pair
    per = []
    for d, g in season.groupby("competition"):
        s = spearman_table(
            g,
            ["imp_block_depth__E2", "imp_def_line__E2"],
            ["def_line_x", "ppda"],
            f"team-season {d}",
        )
        per.append(s)
    if per:
        sp = pd.concat([sp, *per], ignore_index=True)
    return {"team_match_block": block, "team_season_block": season, "team_spearman": sp}


def shift_tables(cfg: TransferConfig, moves: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """(d) distribution of the imputed quantities by domain vs the 360 out-of-fold imputations
    and the 360 truth, on possession Pass / Carry rows and on Shot rows."""
    rows: list[dict[str, Any]] = []
    dom_m = moves["domain"].to_numpy(dtype=object)
    dom_s = shots["domain"].to_numpy(dtype=object)
    for kind, targets, df, dom in (
        ("Pass/Carry (possession)", SHIFT_TARGETS_MOVES, moves, dom_m),
        ("Shot", SHIFT_TARGETS_SHOTS, shots, dom_s),
    ):
        for t in targets:
            c = f"imp_{t}__E2"
            if c not in df.columns:
                continue
            v_all = pf.clip_state(df[c].to_numpy(dtype=np.float32), t).astype(float)
            for lev in [*LEAGUE_DOMAINS, *TOURNAMENT_DOMAINS]:
                m = dom == lev
                if m.sum():
                    rows.append(
                        {"rows": kind, "target": t, "source": lev, **aps._moments(v_all[m])}
                    )
    # 360 side: out-of-fold E2 imputations and the truth on the same event types
    need = (
        ["f_type", "f_is_possession_team", "competition"]
        + [f"{t}__E2" for t in set(SHIFT_TARGETS_MOVES) | set(SHIFT_TARGETS_SHOTS)]
        + [f"y_{t}" for t in set(SHIFT_TARGETS_MOVES) | set(SHIFT_TARGETS_SHOTS)]
    )
    oof = pq.read_table(
        _imputed_oof(), columns=need, filters=[("f_type", "in", ["Pass", "Carry", "Shot"])]
    ).to_pandas()
    ctype = np.where(imf.comp_type(oof["competition"]) == 0.0, "club season", "tournament")
    ftype = oof["f_type"].to_numpy(dtype=object)
    poss = oof["f_is_possession_team"].to_numpy(dtype=bool)
    for kind, targets, m_kind in (
        ("Pass/Carry (possession)", SHIFT_TARGETS_MOVES, np.isin(ftype, ["Pass", "Carry"]) & poss),
        ("Shot", SHIFT_TARGETS_SHOTS, ftype == "Shot"),
    ):
        for t in targets:
            p = pf.clip_state(oof[f"{t}__E2"].to_numpy(dtype=np.float32), t).astype(float)
            yv = oof[f"y_{t}"].to_numpy(dtype=float)
            for lab, m in (
                ("360 all", m_kind),
                ("360 tournaments", m_kind & (ctype == "tournament")),
                ("360 club seasons", m_kind & (ctype == "club season")),
            ):
                rows.append(
                    {
                        "rows": kind,
                        "target": t,
                        "source": f"{lab} OOF imputed",
                        **aps._moments(p[m]),
                    }
                )
                rows.append(
                    {"rows": kind, "target": t, "source": f"{lab} truth", **aps._moments(yv[m])}
                )
    return pd.DataFrame(rows)


def example_tables(cfg: TransferConfig, moves: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """(e) per-minute imputed block depth vs observed defensive-action x for one match."""
    mid = cfg.example_match or choose_example_match(moves)
    rows = moves[moves["match_id"].to_numpy() == mid]
    with open(sb_dir() / "events" / f"{mid}.json") as f:
        events = json.load(f)
    matches = pd.read_parquet(_team_match())
    tm = matches[matches["match_id"] == mid]
    teams = {int(r.team_id): str(r.team) for r in tm.itertuples(index=False)}
    if not teams:
        ids = sorted(set(rows["team_id"].tolist()) | set(rows["opp_team_id"].tolist()))
        teams = {int(t): str(t) for t in ids}
    prof = minute_profile(rows, events, teams)
    prof.insert(0, "match_id", mid)
    binned = bin_profile(prof)
    binned.insert(0, "match_id", mid)
    corr = profile_correlation(prof)
    corr["level"] = "per minute"
    corr5 = profile_correlation(binned.rename(columns={"minutes": "minute"}))
    corr5["level"] = "5-minute bins"
    meta = rows[["competition", "season", "domain"]].drop_duplicates().head(1)
    info = pd.DataFrame(
        {
            "match_id": [mid],
            "competition": meta["competition"].iloc[0] if len(meta) else "",
            "season": meta["season"].iloc[0] if len(meta) else "",
            "teams": [" vs ".join(teams.values())],
            "n_imputed_rows": [int(len(rows))],
            "n_def_actions": [
                int(sum(1 for e in events if (e.get("type") or {}).get("name") in DEF_ACTION_TYPES))
            ],
        }
    )
    return {
        "example_minutes": prof,
        "example_bins": binned,
        "example_correlation": pd.concat([corr, corr5], ignore_index=True),
        "example_info": info,
    }


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------


def _pred_cols(df: pd.DataFrame, suffix: str = "") -> dict[str, np.ndarray]:
    out = {}
    for c in df.columns:
        if c.startswith("pred_") and c.endswith(suffix):
            name = c[len("pred_") :]
            if suffix:
                name = name[: -len(suffix)]
            v = df[c].to_numpy(dtype=float)
            if np.isnan(v).all():
                continue
            out[name] = v
    return out


XG_ORDER = [
    "BASE",
    "LOC",
    "EVENT",
    "EVENT+IMP(E0)",
    "EVENT+IMP",
    "EVENT+ORACLESHOT",
    "EVENT+ORACLESHOT_full",
    "EVENT" + ZERO_SHOT_SUFFIX,
    "EVENT+IMP" + ZERO_SHOT_SUFFIX,
    "EVENT+ORACLESHOT" + ZERO_SHOT_SUFFIX,
    XG_REFERENCE,
]


def xg_tables(cfg: TransferConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metr, delt, dzs, cal, sl = [], [], [], [], []
    for popn, d in preds.groupby("population", sort=False):
        y = d["is_goal"].to_numpy(dtype=float)
        match = d["match_id"].to_numpy()
        p = _pred_cols(d)
        p = {k: p[k] for k in XG_ORDER if k in p}
        m = pf.metrics_table(y, p)
        m.insert(0, "population", popn)
        metr.append(m)
        dd = pf.deltas_table(y, p, "EVENT", match, cfg.n_boot, cfg.seed)
        dd.insert(0, "population", popn)
        delt.append(dd)
        zref = "EVENT" + ZERO_SHOT_SUFFIX
        if zref in p:
            pz = {k: v for k, v in p.items() if k.endswith(ZERO_SHOT_SUFFIX) or k == XG_REFERENCE}
            dz = pf.deltas_table(y, pz, zref, match, cfg.n_boot, cfg.seed)
            dz.insert(0, "population", popn)
            dzs.append(dz)
        if popn in (POP_LEAGUES, POP_TOURNAMENTS):
            c = pf.calibration_frame(
                y,
                {
                    k: p[k]
                    for k in (
                        "EVENT",
                        "EVENT+IMP",
                        "EVENT+ORACLESHOT",
                        "EVENT" + ZERO_SHOT_SUFFIX,
                        "EVENT+IMP" + ZERO_SHOT_SUFFIX,
                        XG_REFERENCE,
                    )
                    if k in p
                },
            )
            c.insert(0, "population", popn)
            cal.append(c)
            slices = {
                "domain": d["domain"].to_numpy(dtype=object),
                "play_pattern": pf.pattern_group(d["f_play_pattern"].to_numpy(dtype=object)),
                "distance_band": pf.distance_band(d["f_dist_goal"].to_numpy(dtype=float)),
                "has_assist": np.where(
                    d["a_has_assist"].to_numpy(dtype=float) == 1.0, "assisted", "unassisted"
                ).astype(object),
            }
            s = pf.slice_table(
                y,
                {
                    k: p[k]
                    for k in (
                        "EVENT",
                        "EVENT+IMP",
                        "EVENT+ORACLESHOT",
                        "EVENT" + ZERO_SHOT_SUFFIX,
                        XG_REFERENCE,
                    )
                    if k in p
                },
                "EVENT",
                slices,
                match,
                n_boot=min(cfg.n_boot, 1000),
                seed=cfg.seed,
            )
            s.insert(0, "population", popn)
            sl.append(s)
    out = {
        "xg_metrics": pd.concat(metr, ignore_index=True),
        "xg_deltas": pd.concat(delt, ignore_index=True),
        "xg_calibration": pd.concat(cal, ignore_index=True) if cal else pd.DataFrame(),
        "xg_slices": pd.concat(sl, ignore_index=True) if sl else pd.DataFrame(),
    }
    if dzs:
        out["xg_deltas_zeroshot"] = pd.concat(dzs, ignore_index=True)
    return out


def xpass_tables(cfg: TransferConfig, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    y = preds["is_complete"].to_numpy(dtype=float)
    match = preds["match_id"].to_numpy()
    metr, delt, cal, sl = [], [], [], []
    for tag in ("after", "noafter"):
        p = {"BASE": preds["pred_BASE"].to_numpy(dtype=float)}
        p.update(_pred_cols(preds, f"__{tag}"))
        if "EVENT" not in p:
            continue
        m = pf.metrics_table(y, p)
        m.insert(0, "design", tag)
        metr.append(m)
        d = pf.deltas_table(y, p, "EVENT", match, cfg.n_boot_pass, cfg.seed)
        d.insert(0, "design", tag)
        delt.append(d)
        c = pf.calibration_frame(y, {k: v for k, v in p.items() if k != "BASE"})
        c.insert(0, "design", tag)
        cal.append(c)
        slices = {
            "domain": preds["domain"].to_numpy(dtype=object),
            "length_band": pf.length_band(preds["f_after_pass_length"].to_numpy(dtype=float)),
            "height": preds["f_after_pass_height"].to_numpy(dtype=object),
            "play_pattern": pf.pattern_group(preds["f_play_pattern"].to_numpy(dtype=object)),
            "under_pressure": np.where(
                preds["f_under_pressure"].to_numpy(dtype=bool), "pressed", "unpressed"
            ).astype(object),
        }
        s = pf.slice_table(
            y,
            {k: v for k, v in p.items() if k != "BASE"},
            "EVENT",
            slices,
            match,
            n_boot=min(cfg.n_boot_pass, 300),
            seed=cfg.seed,
            min_n=200,
        )
        s.insert(0, "design", tag)
        sl.append(s)
    return {
        "xpass_metrics": pd.concat(metr, ignore_index=True),
        "xpass_deltas": pd.concat(delt, ignore_index=True),
        "xpass_calibration": pd.concat(cal, ignore_index=True),
        "xpass_slices": pd.concat(sl, ignore_index=True),
    }


def in_domain_reference() -> pd.DataFrame:
    """Stage-03 in-domain xG / xPass numbers (log-loss and paired delta vs EVENT) for the report."""
    rows = []
    rd = reports_dir()
    for task, mfile, dfile, key in (
        ("xG (360, 5-fold)", "soccer_03_xg_metrics.parquet", "soccer_03_xg_deltas.parquet", None),
        (
            "xPass (360, 5-fold)",
            "soccer_03_xpass_metrics.parquet",
            "soccer_03_xpass_deltas.parquet",
            "design",
        ),
    ):
        mp, dp = rd / mfile, rd / dfile
        if not (mp.exists() and dp.exists()):
            continue
        m = pd.read_parquet(mp)
        d = pd.read_parquet(dp)
        for r in m.itertuples(index=False):
            row = {
                "task": task,
                "variant": r.variant,
                "n": int(r.n),
                "log_loss": float(r.log_loss),
                "auc": float(r.auc),
            }
            if key:
                row["design"] = getattr(r, key)
                sel = d[(d["variant"] == r.variant) & (d[key] == getattr(r, key))]
            else:
                sel = d[d["variant"] == r.variant]
            if len(sel):
                row.update(
                    {
                        "delta_log_loss": float(sel["delta_log_loss"].iloc[0]),
                        "ci_low": float(sel["ci_low"].iloc[0]),
                        "ci_high": float(sel["ci_high"].iloc[0]),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


#: display order of the oracle-check groups (target domains first, then the 360 references)
GROUP_ORDER: tuple[str, ...] = (
    POP_LEAGUES,
    *LEAGUE_DOMAINS,
    *TOURNAMENT_DOMAINS,
    "360 all (OOF)",
    "360 male club season (OOF)",
    "360 male tournament (OOF)",
    "360 female tournament (OOF)",
)


def _group_key(group: object) -> tuple[int, str]:
    g = str(group)
    return (GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER), g)


def _oracle_md(check: pd.DataFrame, target: str) -> str:
    d = check[check["target"] == target].copy()
    d["_k"] = [_group_key(g) for g in d["group"]]
    d["_o"] = (d["observed"] == "360 frame").astype(int)
    d = d.sort_values(["_o", "_k"]).drop(columns=["_k", "_o"])
    cols = ["group", "observed", "n", "r2", "mae", "bias", "corr", "mean_imp", "mean_obs", "sd_obs"]
    if "within_1" in d.columns and d["within_1"].notna().any():
        cols += ["exact", "within_1"]
    return md_table(d[cols], floatfmt="{:.3f}")


def oracle_summary_long(check: pd.DataFrame) -> pd.DataFrame:
    """R2 / MAE / bias vs the shot frame, one row per (target, metric), one column per group
    (:data:`GROUP_ORDER`), plus the R2 vs the 360 frame of the 360 out-of-fold rows."""
    d = check[check["observed"] == "shot.freeze_frame"]
    rows = []
    for t, _, _ in ORACLE_PAIRS:
        dt = d[d["target"] == t].set_index("group")
        if dt.empty:
            continue
        d360 = check[(check["target"] == t) & (check["observed"] == "360 frame")].set_index("group")
        for metric in ("r2", "mae", "bias"):
            row: dict[str, Any] = {"target": t, "metric": metric}
            for g in GROUP_ORDER:
                if g in dt.index:
                    row[g] = float(dt.loc[g, metric])
            if metric == "r2" and "360 all (OOF)" in d360.index:
                row["360 all (OOF) vs 360 frame"] = float(d360.loc["360 all (OOF)", "r2"])
            rows.append(row)
    return pd.DataFrame(rows)


def _cal_md(cal: pd.DataFrame, target: str, groups: list[str]) -> str:
    d = cal[(cal["target"] == target) & cal["group"].isin(groups)]
    if d.empty:
        return "(no rows)"
    piv = d.pivot_table(
        index="bin", columns="group", values=["imputed", "observed"], aggfunc="first"
    )
    piv.columns = [f"{b}: {a}" for a, b in piv.columns]
    piv = piv[sorted(piv.columns, key=lambda c: (groups.index(c.split(": ")[0]), c))]
    return md_table(piv.reset_index(), floatfmt="{:.2f}")


def _oracle_summary(check: pd.DataFrame) -> pd.DataFrame:
    """Compact degradation table: 360 OOF (vs sff) -> 2015/16 pooled -> tournaments per target."""
    rows = []
    for t, _, kind in ORACLE_PAIRS:
        d = check[(check["target"] == t) & (check["observed"] == "shot.freeze_frame")].set_index(
            "group"
        )
        d360 = check[(check["target"] == t) & (check["observed"] == "360 frame")].set_index("group")
        row: dict[str, Any] = {"target": t, "kind": kind}
        if "360 all (OOF)" in d360.index:
            row["r2_360_vs_360frame"] = d360.loc["360 all (OOF)", "r2"]
        for g, lab in (
            ("360 all (OOF)", "360_oof"),
            ("360 male club season (OOF)", "360_club"),
            (POP_LEAGUES, "leagues_1516"),
            ("WC 2018", "wc2018"),
            ("Copa 2024", "copa2024"),
            ("AFCON 2023", "afcon2023"),
        ):
            if g in d.index:
                row[f"r2_{lab}"] = d.loc[g, "r2"]
                row[f"mae_{lab}"] = d.loc[g, "mae"]
                row[f"bias_{lab}"] = d.loc[g, "bias"]
        if "bias_leagues_1516" in row and "bias_360_oof" in row:
            # transfer-specific bias: the shot-frame-vs-360-frame measurement gap is removed
            row["bias_shift_1516"] = row["bias_leagues_1516"] - row["bias_360_oof"]
        rows.append(row)
    return pd.DataFrame(rows)


def _reading(tables: dict[str, pd.DataFrame]) -> list[str]:
    out = []
    xd = tables["xg_deltas"]
    pooled = xd[xd["population"] == POP_LEAGUES].set_index("variant")
    if "EVENT+IMP" in pooled.index:
        out.append(
            "- xG in the 2015/16 leagues (within-domain CV, frozen students): EVENT+IMP vs EVENT "
            + po._verdict(pooled.loc["EVENT+IMP"])
            + "; EVENT+ORACLESHOT "
            + po._verdict(pooled.loc["EVENT+ORACLESHOT"])
            + "; StatsBomb xG "
            + po._verdict(pooled.loc[XG_REFERENCE])
            + "."
        )
    if "xg_deltas_zeroshot" in tables:
        z = tables["xg_deltas_zeroshot"]
        zp = z[z["population"] == POP_LEAGUES].set_index("variant")
        parts = []
        for v in ("EVENT+IMP" + ZERO_SHOT_SUFFIX, "EVENT+ORACLESHOT" + ZERO_SHOT_SUFFIX):
            if v in zp.index:
                parts.append(f"{v} {po._verdict(zp.loc[v])}")
        if "EVENT" + ZERO_SHOT_SUFFIX in pooled.index:
            zs_row = pooled.loc["EVENT" + ZERO_SHOT_SUFFIX]
            parts.insert(0, f"EVENT (zero-shot) vs within-2015/16 EVENT {po._verdict(zs_row)}")
        out.append(
            "- Zero-shot stage-03 xG models on 2015/16 shots: "
            + "; ".join(parts)
            + " (deltas of the zero-shot IMP / ORACLESHOT models are vs the zero-shot EVENT model; "
            "the zero-shot vs within-CV EVENT gap mixes domain shift with training-set size: "
            "10,233 360 shots vs ~30k 2015/16 shots per training fold)."
        )
    per = []
    for d in LEAGUE_DOMAINS:
        t = xd[xd["population"] == d].set_index("variant")
        if "EVENT+IMP" in t.index:
            per.append(
                f"{d}: IMP {po._fmt_delta(t.loc['EVENT+IMP'])}, "
                f"ORACLESHOT {po._fmt_delta(t.loc['EVENT+ORACLESHOT'])}"
            )
    if per:
        n_sig = sum(
            1
            for d in LEAGUE_DOMAINS
            if "EVENT+IMP" in xd[xd["population"] == d].set_index("variant").index
            and xd[(xd["population"] == d) & (xd["variant"] == "EVENT+IMP")]["ci_low"].iloc[0] > 0
        )
        out.append(
            "- Per league (5-fold CV inside the league): "
            + "; ".join(per)
            + f". {n_sig} of {len(per)} leagues show a nominally significant IMP gain; with four "
            "tests one nominal hit is expected by chance, so the pooled estimate is the primary "
            "result."
        )
    if "xpass_deltas" in tables:
        pdl = tables["xpass_deltas"]
        a = pdl[pdl["design"] == "after"].set_index("variant")
        n = pdl[pdl["design"] == "noafter"].set_index("variant")
        s = "- xPass 2015/16: with the destination known EVENT+IMP " + po._verdict(
            a.loc["EVENT+IMP"]
        )
        if "EVENT-nodur" in a.index:
            s += ", dropping the realised duration from EVENT " + po._verdict(a.loc["EVENT-nodur"])
        s += "; pre-instant EVENT+IMP " + po._verdict(n.loc["EVENT+IMP"]) + "."
        out.append(s)
    if "oracle_summary" in tables:
        o = tables["oracle_summary"].set_index("target")
        parts = []
        for t in ("n_opp_in_cone", "nearest_opp_dist_in_cone", "opp_keeper_dist_to_goal_line"):
            if t in o.index and "r2_leagues_1516" in o.columns:
                parts.append(
                    f"{t}: R2 vs shot frame {o.loc[t, 'r2_360_oof']:.3f} (360 OOF) -> "
                    f"{o.loc[t, 'r2_leagues_1516']:.3f} (2015/16), "
                    f"MAE {o.loc[t, 'mae_360_oof']:.2f} -> {o.loc[t, 'mae_leagues_1516']:.2f}, "
                    f"bias {o.loc[t, 'bias_360_oof']:+.2f} -> {o.loc[t, 'bias_leagues_1516']:+.2f} "
                    f"(transfer-specific {o.loc[t, 'bias_shift_1516']:+.2f})"
                )
        out.append(
            "- Oracle check (imputed vs shot.freeze_frame; the in-domain bias is the shot-frame vs "
            "360-frame measurement gap, the transfer-specific part is the change from the 360 OOF "
            "rows to 2015/16): " + "; ".join(parts) + "."
        )
    if "team_spearman" in tables:
        sp = tables["team_spearman"]
        ts = sp[
            (sp["level"] == "team-season") & (sp["imputed"] == "imp_block_depth__E2")
        ].set_index("proxy")
        tm = sp[(sp["level"] == "team-match") & (sp["imputed"] == "imp_block_depth__E2")].set_index(
            "proxy"
        )
        if len(ts):

            def pair(proxy: str) -> str:
                n_s, n_m = int(ts.loc[proxy, "n"]), int(tm.loc[proxy, "n"])
                return (
                    f"{ts.loc[proxy, 'spearman']:+.2f} (team-season, n {n_s}) / "
                    f"{tm.loc[proxy, 'spearman']:+.2f} (team-match, n {n_m})"
                )

            bx = sp[(sp["level"] == "team-season") & (sp["imputed"] == "ball_x_mean")].set_index(
                "proxy"
            )
            e0 = sp[
                (sp["level"] == "team-season") & (sp["imputed"] == "imp_block_depth__E0")
            ].set_index("proxy")
            comp = ""
            if "def_line_x" in bx.index and "def_line_x" in e0.index:
                comp = (
                    f" Comparators at team-season level vs def_line_x: mean ball x alone "
                    f"{bx.loc['def_line_x', 'spearman']:+.2f} (sign reversed by construction), "
                    f"E0 student {e0.loc['def_line_x', 'spearman']:+.2f} - the ranking is "
                    "mostly where the ball is, not what the window features add."
                )
            out.append(
                "- Team level (2015/16): Spearman of the imputed block depth conceded with "
                f"def_line_x {pair('def_line_x')}; with ppda {pair('ppda')}; with passes allowed "
                f"{pair('ppda_passes_allowed')}." + comp
            )
    return out


def _verdict_lines(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Plain verdict from the tables."""
    xd = tables["xg_deltas"]
    pooled = xd[xd["population"] == POP_LEAGUES].set_index("variant")
    o = (
        tables.get("oracle_summary", pd.DataFrame()).set_index("target")
        if "oracle_summary" in tables
        else None
    )
    lines = []
    imp_row = pooled.loc["EVENT+IMP"] if "EVENT+IMP" in pooled.index else None
    osh = pooled.loc["EVENT+ORACLESHOT"] if "EVENT+ORACLESHOT" in pooled.index else None
    if imp_row is not None:
        gain = float(imp_row["delta_log_loss"])
        sig = bool(imp_row["ci_low"] > 0)
        if sig and osh is not None and gain >= 0.25 * float(osh["delta_log_loss"]):
            v = "YES, partly"
        elif sig:
            v = "MARGINALLY"
        else:
            v = "NO measurable downstream value"
        osh_txt = po._fmt_delta(osh) + "." if osh is not None else "n/a."
        lines.append(
            f"- **Does the 360-trained student transfer to event-only club data?** {v}: on "
            f"2015/16 shots the frozen imputations change the xG log-loss by "
            f"{po._fmt_delta(imp_row)} vs an event-only model that already sees every student "
            f"input, while the shot freeze frame (the real oracle) gives {osh_txt}"
        )
    if o is not None and "r2_leagues_1516" in o.columns:
        parts, ratios = [], []
        for t in ("n_opp_in_cone", "nearest_opp_dist_in_cone", "opp_keeper_dist_to_goal_line"):
            if t in o.index and not np.isnan(o.loc[t, "r2_360_oof"]):
                a, b = float(o.loc[t, "r2_360_oof"]), float(o.loc[t, "r2_leagues_1516"])
                parts.append(f"{t} {a:.2f} -> {b:.2f}")
                ratios.append(b / a if a > 0 else np.nan)
        worst = float(np.nanmin(ratios)) if ratios and not np.all(np.isnan(ratios)) else np.nan
        if np.isnan(worst):
            how = "cannot be rated"
        elif worst >= 0.9:
            how = "survives essentially unchanged"
        elif worst >= 0.7:
            how = "survives with a modest loss"
        elif worst >= 0.4:
            how = "degrades materially"
        else:
            how = "largely breaks"
        lines.append(
            f"- As a *measurement* the student {how}: against the shot freeze frame its R2 goes "
            + ", ".join(parts)
            + " from the 360 out-of-fold shots to 2015/16 (MAE / bias / calibration in the oracle "
            "tables). The shot state it recovers was already weak in-domain, so what transfers "
            "is a weak signal."
        )
    if "xpass_deltas" in tables:
        pdl = tables["xpass_deltas"]
        a = pdl[pdl["design"] == "after"].set_index("variant")
        n = pdl[pdl["design"] == "noafter"].set_index("variant")
        da, dn = a.loc["EVENT+IMP"], n.loc["EVENT+IMP"]
        if da["ci_low"] > 0 and dn["ci_low"] > 0:
            qual = "both significant"
        elif dn["ci_low"] > 0:
            qual = "only the pre-instant one significant"
        elif da["ci_low"] > 0:
            qual = "only the destination-known one significant"
        else:
            qual = "neither significant"
        lines.append(
            f"- Passes: the frozen pass-state students give {po._fmt_delta(da)} with the "
            f"destination known and {po._fmt_delta(dn)} pre-instant ({qual}; in-domain stage 03: "
            "+0.0005 / +0.0016)."
        )
    if "team_spearman" in tables:
        lines.append(
            "- Team-level aggregates of the imputed block depth rank teams consistently with the "
            "event-only style proxies (Spearman table), but the E2 student reads the opponent's "
            "recent defensive-action x directly, so this is a consistency check, not independent "
            "validation."
        )
    return lines


PROTOCOL_TEXT = [
    "- Populations: Shot rows of `events_no360.parquet` in the four 2015/16 leagues (pooled and "
    "per league) and in the three non-360 tournaments (World Cup 2018, Copa America 2024, AFCON "
    "2023, pooled); penalties excluded; label `post_shot_outcome == 'Goal'`. Passes: the 2015/16 "
    "Pass rows (the build stage kept a fixed 25% of passes per match), `Unknown` / `Injury "
    "Clearance` dropped, match-stratified subsample. Stray non-360 matches of other competitions "
    "(9 360-flagged fall-backs) are excluded.",
    "- Students: frozen stage-02 bundles (`students_E0/E2/E2a.joblib`, trained on all 417 360 "
    "matches) as applied by `apply_student` (`imputed_no360.parquet`); the assist pass of every "
    "shot is rebuilt with the build-stage feature code (`process_match`, all passes of the match) "
    "because the 25% pass sample misses most key passes, then imputed with the same bundles; the "
    "receiver-distance student (no stage-02 model) is fitted once on the 360 Pass rows and applied "
    "here. No non-360 match ever enters a student's training set.",
    "- Downstream models: LightGBM (stage-03 settings), `group_kfold` by match inside the target "
    "population (seed {seed}, {folds} folds), rounds by early stopping on an inner match holdout; "
    "xG refitted on all training rows and averaged over seeds (pooled {sx}, per league {sl}); "
    "xPass single seed without refit. BASE = training-fold positive rate. Zero-shot = the "
    "stage-03 `xg_models.joblib` (EVENT / EVENT+IMP / EVENT+ORACLESHOT fitted on the 10,233 360 "
    "shots) applied as is; constant design columns of a single-gender / single-competition-type "
    "domain (`f_gender`, `f_comp_type`) are restored from the undropped design.",
    "- Features: EVENT = stage-02 E2 design of the shot minus `f_after_duration` plus the key "
    "pass's attributes (incl. its realised trajectory and duration, complete before the shot); "
    "EVENT+IMP adds the frozen E2 shot-state and E2a assist-lane imputations (clipped at 0); "
    "EVENT+ORACLESHOT the same quantities from `shot.freeze_frame`; `_full` every `sff_*` column. "
    "xPass EVENT = the E2a (after) or E2 (noafter) design incl. `f_after_duration` in the after "
    "design (every student input is in the baseline); EVENT+IMP adds the seven pass-state "
    "imputations (E2a / E2 students). Never used as features: `shot.freeze_frame` outside the "
    "oracle variants, `statsbomb_xg`, `one_on_one`, `open_goal`, `post_*`, any `y_*`.",
    "- Metrics: log-loss, Brier, AUC, ECE (10 equal-count bins); paired per-sample bootstrap of "
    "the per-sample loss ({nb} resamples for shots, {nbp} for passes, seed {seed}) with a "
    "match-clustered CI next to it; `significant` = the per-sample CI excludes 0. Positive delta "
    "= better than the reference.",
    "- Oracle check: imputed E2 shot state vs the `shot.freeze_frame` quantity on every shot where "
    "both exist (`sff_nearest_opp_dist_in_cone` is NaN when the cone is empty, ~49% of shots; the "
    "students were trained on the 360 frame, whose cone count agrees with the shot frame only to "
    "r 0.75, so the 360 out-of-fold rows scored against the same shot frame are the like-for-like "
    "reference). Calibration = equal-count bins of the imputed value, mean imputed vs mean "
    "observed.",
]

CAVEATS_TEXT = [
    "- The shot-state students were weak in-domain (OOF R2 vs the 360 frame 0.16 / 0.41 / 0.48 "
    "for cone count / cone distance / keeper distance, stage 02), so a small transfer loss on top "
    "does not change the downstream picture; the interesting oracle (the shot freeze frame) "
    "exists in 2015/16 and is not imputed by anyone here.",
    "- The 2015/16 event data is older StatsBomb collection (2015/16 was coded retrospectively); "
    "collection conventions (pressure events, freeze-frame completeness) may differ from 2020-25. "
    "The domain shift therefore mixes era, competition type (club vs tournament; only a handful "
    "of single-club seasons were in the 360 training set) and collection.",
    "- The key-pass rebuild recomputes the build-stage features with the same code and settings; "
    "9 of the non-360 matches are 360-flagged fall-backs whose passes are rebuilt through the "
    "same fall-back path. Shots whose key pass is missing from the raw events keep "
    "`a_has_assist = 0`.",
    "- The team-level check is partly circular: `f_opp_def_x_60s_mean` (the opponent's recent "
    "defensive-action x) is an E2 input, and `def_line_x` in `team_match.parquet` is the match "
    "mean of the same quantity. The E0 student (no window features) and the raw mean ball x are "
    "shown as comparators. Imputed block depth is a visible-area-truncated 360 quantity (stage-02 "
    "caveat), not tracking.",
    "- The worked example uses a 25% pass / carry sample per match, so most minutes have 0-3 "
    "imputed rows; read the 5-minute bins. The observed series (mean x of the team's defensive "
    "actions) is where the team acted, not where its block stood, and is one of the student's "
    "own inputs.",
    "- Zero-shot xG models were fitted on shots from both genders and mostly tournaments; their "
    "`f_gender` / `f_comp_type` inputs take constant values here. The within-2015/16 CV models "
    "have ~4x more shots than the 360 xG models had, so a lower within-CV log-loss is expected "
    "even without any domain shift.",
    "- Stage-02 open issues respected: imputed counts / distances clipped at 0 wherever they are "
    "used or scored here; the transfer BSS reference issue does not apply (no binary targets "
    "used); the NFL highlight check is not used.",
    "- Single machine, 2 threads; LightGBM is seeded but not deterministic across machines.",
]


def _oracle_section(tables: dict[str, pd.DataFrame]) -> list[str]:
    out = ["\n## (a) Oracle check under domain shift: imputed shot state vs shot.freeze_frame\n"]
    out.append(
        "Per group: n shots where both values exist, R2 / MAE / bias (imputed - observed) / "
        "correlation, means; for counts the exact and within-1 agreement of the rounded "
        "imputation. `360 ... (OOF)` rows are the stage-02 out-of-fold imputations of the 360 "
        "shots scored against the same shot frame (like for like) and against the 360 frame "
        "(their own label).\n"
    )
    out.append(
        "### Summary: R2 / MAE / bias (imputed - observed) vs the shot frame per group; the last "
        "column is the 360 out-of-fold R2 against the 360 frame (the students' own label)\n"
    )
    out.append(md_table(oracle_summary_long(tables["oracle_check"]), floatfmt="{:.3f}"))
    for t, _, _ in ORACLE_PAIRS:
        out.append(f"\n**{t}**\n")
        out.append(_oracle_md(tables["oracle_check"], t))
    out.append(
        "\n### Calibration (equal-count bins of the imputed value; mean imputed vs mean observed)\n"
    )
    groups = ["360 all (OOF)", POP_LEAGUES, "WC 2018", "Copa 2024", "AFCON 2023"]
    for t in ("n_opp_in_cone", "nearest_opp_dist_in_cone", "opp_keeper_dist_to_goal_line"):
        out.append(f"\n**{t}**\n")
        out.append(_cal_md(tables["oracle_calibration"], t, groups))
    return out


def _xg_section(tables: dict[str, pd.DataFrame], xm: pd.DataFrame) -> list[str]:
    out = ["\n## (b) Goal prediction on 2015/16 shots (students frozen from the 360 domain)\n"]
    for popn in (POP_LEAGUES, POP_TOURNAMENTS, *LEAGUE_DOMAINS):
        m = xm[xm["population"] == popn]
        if m.empty:
            continue
        out.append(f"\n### {popn}\n")
        out.append(po._metrics_md(m))
        d = tables["xg_deltas"][tables["xg_deltas"]["population"] == popn]
        out.append("\nPaired deltas vs EVENT (within-domain CV):\n")
        out.append(po._delta_md(d))
        if "xg_deltas_zeroshot" in tables:
            dz = tables["xg_deltas_zeroshot"]
            dz = dz[dz["population"] == popn]
            if len(dz):
                out.append("\nPaired deltas vs EVENT (zero-shot):\n")
                out.append(po._delta_md(dz))
    ref = tables.get("in_domain_reference")
    if ref is not None and len(ref):
        out.append("\n### In-domain reference (stage 03, 360 matches)\n")
        out.append(md_table(ref))
    if len(tables["xg_calibration"]):
        out.append(
            "\n### Calibration, 2015/16 pooled (equal-count bins: mean predicted vs observed goal "
            "rate)\n"
        )
        c = tables["xg_calibration"]
        out.append(md_table(po._pivot_cal(c[c["population"] == POP_LEAGUES]), floatfmt="{:.3f}"))
    if len(tables["xg_slices"]):
        out.append(
            "\n### Slices, 2015/16 pooled (log-loss per variant; delta vs EVENT with per-shot CI)\n"
        )
        s = tables["xg_slices"]
        s = s[s["population"] == POP_LEAGUES]
        variants = (
            "EVENT",
            "EVENT+IMP",
            "EVENT+ORACLESHOT",
            "EVENT" + ZERO_SHOT_SUFFIX,
            XG_REFERENCE,
        )
        for sname in ("domain", "distance_band", "play_pattern", "has_assist"):
            out.append(f"\n**{sname}**\n")
            out.append(po._slice_md(s, sname, variants))
    return out


def _xpass_section(tables: dict[str, pd.DataFrame]) -> list[str]:
    out = ["\n## (c) Pass completion in 2015/16 (no pass oracle without 360)\n"]
    out.append(
        "`after` = realised end location / length / height / flags / duration are features "
        "(E2a students); `noafter` = pre-instant only (E2 students). `EVENT-nodur` drops the "
        "realised duration from the after design (what that one field is worth).\n"
    )
    out.append(po._metrics_md(tables["xpass_metrics"], ["design"]))
    out.append("\nPaired deltas vs EVENT:\n")
    out.append(po._delta_md(tables["xpass_deltas"], ["design"]))
    if "receiver_fits" in tables:
        out.append("\nReceiver-distance student fitted on the 360 Pass rows:\n")
        out.append(md_table(tables["receiver_fits"]))
    out.append("\n### xPass calibration (after design)\n")
    c = tables["xpass_calibration"]
    out.append(md_table(po._pivot_cal(c[c["design"] == "after"]), floatfmt="{:.3f}"))
    out.append("\n### xPass slices (after design)\n")
    s = tables["xpass_slices"]
    after, noafter = s[s["design"] == "after"], s[s["design"] == "noafter"]
    for sname in ("domain", "length_band", "height", "play_pattern", "under_pressure"):
        out.append(f"\n**{sname}**\n")
        out.append(po._slice_md(after, sname, ("EVENT", "EVENT+IMP", "EVENT-nodur")))
    out.append("\n### xPass slices (noafter design)\n")
    for sname in ("domain", "length_band"):
        out.append(f"\n**{sname}**\n")
        out.append(po._slice_md(noafter, sname, ("EVENT", "EVENT+IMP")))
    return out


def _team_section(tables: dict[str, pd.DataFrame]) -> list[str]:
    out = ["\n## (d) Team-level sanity (2015/16 leagues)\n"]
    out.append(
        "Imputed block depth / defensive line conceded = mean of the E2 (and E0) imputations over "
        "the opponent's possession Pass / Carry rows (25% sample) per match, attributed to the "
        "defending team; `ball_x_mean` = mean attacking-frame x of those rows (location-only "
        "comparator). Proxies from `team_match.parquet`: `def_line_x` (mean x of the team's "
        "defensive actions in its own frame), `ppda`, `ppda_passes_allowed`, `possession`. "
        "Spearman rank correlations:\n"
    )
    out.append(md_table(tables["team_spearman"], floatfmt="{:.3f}"))
    out.append(
        "\nTeam-season means (top / bottom 5 by imputed block depth conceded, pooled leagues):\n"
    )
    ts = tables["team_season_block"].sort_values("imp_block_depth__E2", ascending=False)
    cols = [
        "competition",
        "team",
        "n_matches",
        "imp_block_depth__E2",
        "imp_def_line__E2",
        "imp_block_depth__E0",
        "ball_x_mean",
        "def_line_x",
        "ppda",
        "ppda_passes_allowed",
    ]
    cols = [c for c in cols if c in ts.columns]
    out.append(md_table(pd.concat([ts.head(5), ts.tail(5)])[cols], floatfmt="{:.2f}"))
    out.append(
        "\n### Domain shift: distribution of the imputed quantities by domain vs the 360 domain\n"
    )
    out.append(
        "Possession Pass / Carry rows and Shot rows; `360 ... OOF imputed` = stage-02 out-of-fold "
        "imputations on the same event types, `truth` = the 360 label where valid (reliable "
        "frames only for shape targets, so truth and imputed rows differ in selection).\n"
    )
    sh = tables["shift"]
    for kind in ("Pass/Carry (possession)", "Shot"):
        out.append(f"\n**{kind}**\n")
        d = sh[sh["rows"] == kind][["target", "source", "n", "mean", "sd", "p05", "p50", "p95"]]
        out.append(md_table(d, floatfmt="{:.2f}"))
    return out


def _example_section(tables: dict[str, pd.DataFrame]) -> list[str]:
    out = ["\n## (e) Worked example: imputed block depth over time in one 2015/16 match\n"]
    info = tables["example_info"].iloc[0]
    out.append(
        f"Match {int(info['match_id'])}: {info['teams']} ({info['competition']} "
        f"{info['season']}); {int(info['n_imputed_rows'])} imputed possession rows, "
        f"{int(info['n_def_actions'])} defensive actions. For every team: the imputed block depth "
        "of that team (E2 student, from the opponent's possession Pass / Carry rows, yards from "
        "the team's own goal line) next to the mean x of the team's own defensive actions in the "
        "same minutes (also yards from its own goal line). 5-minute bins (row-weighted); the "
        "per-minute table is in `soccer_04_example_minutes.parquet`.\n"
    )
    b = tables["example_bins"]
    piv = []
    for _tid, g in b.groupby("team_id", sort=True):
        name = g["team"].iloc[0]
        gg = g[["minutes", "imp_block_depth", "n_rows", "def_x_mean", "n_def"]].rename(
            columns={
                "imp_block_depth": f"{name}: imputed block depth",
                "n_rows": f"{name}: n rows",
                "def_x_mean": f"{name}: def-action x",
                "n_def": f"{name}: n actions",
            }
        )
        piv.append(gg.set_index("minutes"))
    if piv:
        tab = pd.concat(piv, axis=1).reset_index()
        tab["_k"] = tab["minutes"].str.split("-").str[0].astype(int)
        tab = tab.sort_values("_k").drop(columns="_k")
        for c in tab.columns:
            if c.endswith((": n rows", ": n actions")):
                tab[c] = tab[c].fillna(0).astype(int)
        out.append(md_table(tab, floatfmt="{:.1f}"))
    out.append("\nCorrelation between the two series (over minutes / bins where both exist):\n")
    out.append(md_table(tables["example_correlation"], floatfmt="{:.3f}"))
    return out


def render_report(
    cfg: TransferConfig,
    tables: dict[str, pd.DataFrame],
    shots: pd.DataFrame,
    elapsed: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# Soccer 04 - transfer of the 360-trained students to seasons without 360\n")
    lines.append(
        "Machine-written by `research/privileged_tracking/soccer/transfer.py`. Question: does the "
        "event-only defensive-state student trained on 2020-2025 360 data transfer to event-only "
        "club football (2015/16 Premier League, La Liga, Serie A, Ligue 1) and to non-360 "
        "tournaments (World Cup 2018, Copa America 2024, AFCON 2023) - as a measurement (vs the "
        "shot freeze frame), as a downstream feature (xG, xPass) and at team level?\n"
    )
    is_l = shots["domain"].isin(LEAGUE_DOMAINS).to_numpy()
    is_t = shots["domain"].isin(TOURNAMENT_DOMAINS).to_numpy()
    lines.append("## Headline\n")
    lines.append(
        f"- Shots: {int(is_l.sum()):,} non-penalty shots / "
        f"{int(shots.loc[is_l, 'is_goal'].sum()):,} goals in "
        f"{shots.loc[is_l, 'match_id'].nunique():,} 2015/16 matches; {int(is_t.sum()):,} / "
        f"{int(shots.loc[is_t, 'is_goal'].sum()):,} in {shots.loc[is_t, 'match_id'].nunique():,} "
        f"non-360 tournament matches; {int(shots['a_has_assist'].sum()):,} shots with a rebuilt "
        "key pass."
    )
    xm = tables["xg_metrics"]
    for popn in (POP_LEAGUES, POP_TOURNAMENTS):
        m = xm[xm["population"] == popn].set_index("variant")
        if "EVENT" not in m.index:
            continue

        def ll(v: str, m: pd.DataFrame = m) -> str:
            return f"{float(m.loc[v, 'log_loss']):.4f}" if v in m.index else "n/a"

        zs = ZERO_SHOT_SUFFIX
        lines.append(
            f"- xG log-loss, {popn} (within-domain CV): BASE {ll('BASE')}, LOC {ll('LOC')}, "
            f"EVENT {ll('EVENT')}, EVENT+IMP {ll('EVENT+IMP')}, EVENT+ORACLESHOT "
            f"{ll('EVENT+ORACLESHOT')}, _full {ll('EVENT+ORACLESHOT_full')}; zero-shot 360 "
            f"models: EVENT {ll('EVENT' + zs)}, EVENT+IMP {ll('EVENT+IMP' + zs)}, "
            f"EVENT+ORACLESHOT {ll('EVENT+ORACLESHOT' + zs)}; StatsBomb xG {ll(XG_REFERENCE)}."
        )
    if "xpass_metrics" in tables:
        pm = tables["xpass_metrics"]
        for tag in ("after", "noafter"):
            mm = pm[pm["design"] == tag].set_index("variant")
            if "EVENT" in mm.index:
                nodur = (
                    f", EVENT-nodur {mm.loc['EVENT-nodur', 'log_loss']:.4f}"
                    if "EVENT-nodur" in mm.index
                    else ""
                )
                lines.append(
                    f"- xPass 2015/16 ({tag}): n = {int(mm.loc['EVENT', 'n']):,}, completion "
                    f"{mm.loc['EVENT', 'positives'] / mm.loc['EVENT', 'n']:.3f}; log-loss BASE "
                    f"{mm.loc['BASE', 'log_loss']:.4f}, EVENT {mm.loc['EVENT', 'log_loss']:.4f}, "
                    f"EVENT+IMP {mm.loc['EVENT+IMP', 'log_loss']:.4f}{nodur}."
                )
    lines.append("")
    lines.append("### Reading\n")
    lines += _reading(tables)
    lines.append("")
    lines.append("### Verdict\n")
    lines += _verdict_lines(tables)
    lines.append("")
    lines.append("## Protocol\n")
    lines += [
        t.format(
            seed=cfg.seed,
            folds=cfg.n_folds,
            sx=cfg.n_seeds_xg,
            sl=cfg.n_seeds_league,
            nb=cfg.n_boot,
            nbp=cfg.n_boot_pass,
        )
        for t in PROTOCOL_TEXT
    ]
    lines += _oracle_section(tables)
    lines += _xg_section(tables, xm)
    if "xpass_metrics" in tables:
        lines += _xpass_section(tables)
    lines += _team_section(tables)
    lines += _example_section(tables)
    lines.append("\n## Fits\n")
    if "xg_fits" in tables:
        lines.append(po._fits_md(tables["xg_fits"], ["population", "variant"]))
    if "xpass_fits" in tables:
        lines.append("")
        lines.append(po._fits_md(tables["xpass_fits"], ["variant", "design"]))
    lines.append("\nStage wall time (s): " + ", ".join(f"{k} {v:.0f}" for k, v in elapsed.items()))
    lines.append("\n## Caveats\n")
    lines += CAVEATS_TEXT
    return "\n".join(lines) + "\n"


def stage_report(
    cfg: TransferConfig, elapsed: dict[str, float] | None = None
) -> dict[str, pd.DataFrame]:
    """Assemble (a), (d), (e) and every result table; write ``soccer_04_*.parquet`` and
    ``soccer_04_transfer.md``."""
    t0 = time.time()
    shots = shot_population(cfg, build_shot_table(cfg))
    tables: dict[str, pd.DataFrame] = {}
    tables.update(oracle_tables(cfg, shots))
    tables["oracle_summary"] = _oracle_summary(tables["oracle_check"])
    tables.update(xg_tables(cfg, stage_xg(cfg)))
    fp = cache_dir(cfg) / XG_FITS
    if fp.exists():
        tables["xg_fits"] = pd.read_parquet(fp)
    pp = cache_dir(cfg) / XPASS_PREDS
    if pp.exists():
        tables.update(xpass_tables(cfg, pd.read_parquet(pp)))
        for key, name in (("xpass_fits", XPASS_FITS), ("receiver_fits", RECEIVER_FITS)):
            p = cache_dir(cfg) / name
            if p.exists():
                tables[key] = pd.read_parquet(p)
    tables["in_domain_reference"] = in_domain_reference()
    moves = _move_rows(cfg)
    tables.update(team_tables(cfg, moves))
    tables["shift"] = shift_tables(cfg, moves, shots)
    tables.update(example_tables(cfg, moves))
    elapsed = dict(elapsed or {})
    elapsed["report"] = time.time() - t0
    text = render_report(cfg, tables, shots, elapsed)
    if cfg.write and not cfg.smoke:
        out = reports_dir()
        for k, t in tables.items():
            t.to_parquet(out / f"{REPORT_PREFIX}_{k}.parquet", index=False)
        stage_xg(cfg).to_parquet(out / f"{REPORT_PREFIX}_xg_predictions.parquet", index=False)
        (out / f"{REPORT_PREFIX}_transfer.md").write_text(text)
        _log(f"report written: {out / (REPORT_PREFIX + '_transfer.md')} ({len(tables)} tables)")
    else:
        _log(text[:4000])
    return tables


STAGES = ("shots", "xg", "passes", "xpass", "report")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", default="all", help="one of " + ", ".join(STAGES) + " or all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--xpass-n", type=int, default=None)
    ap.add_argument("--example-match", type=int, default=None)
    args = ap.parse_args()
    cfg = TransferConfig(
        n_jobs=args.n_jobs, workers=args.workers, smoke=args.smoke, example_match=args.example_match
    )
    if args.smoke:
        cfg.n_seeds_xg = 1
        cfg.n_seeds_league = 1
        cfg.n_boot = 200
        cfg.n_boot_pass = 100
        cfg.xpass_target_n = 8_000
        cfg.receiver_train_cap = 30_000
    if args.xpass_n:
        cfg.xpass_target_n = args.xpass_n
    stages = list(STAGES) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]
    elapsed: dict[str, float] = {}
    for s in stages:
        t0 = time.time()
        if s == "shots":
            build_shot_table(cfg, force=args.force)
        elif s == "xg":
            stage_xg(cfg, force=args.force)
        elif s == "passes":
            build_pass_table(cfg, force=args.force)
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
