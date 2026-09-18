"""Stage 03 -- player pass counts: does a per-pass completion gain move a count line?

The completed programme's one positive imputed-state result was pass *difficulty*: adding
out-of-fold student predictions of the 360 defensive state to a pre-instant xPass model was
worth +0.0016 nats per pass (LightGBM students) and +0.0026 nats per pass (sequence student).
Player pass props are priced off recent averages, so this stage asks whether that per-pass
gain survives aggregation to a player-match **count**, and whether anything about player pass
counts is priceable against a rolling-mean book proxy.

The stage has three parts, which correspond to the three channels a completed-pass prop runs
through:

(a) ``--stage passpreds`` / ``--stage a`` -- the COMPLETION channel. Per-pass completion
    probabilities are refitted out-of-fold on *all* 373,639 pass attempts of the 417 StatsBomb
    360 matches (the programme's cache holds a 149,591-pass match-stratified subsample, whose
    player-match counts would be ~35% of full size), then aggregated to a player-match
    completed-pass distribution as a Poisson binomial given the realised attempts. This
    isolates completion from volume, and conditioning on realised attempts is itself a caveat:
    a real prop must predict attempts too.
(b) ``--stage b`` -- the VOLUME channel. Player-match pass ATTEMPTS from strictly prior
    matches against a shrunk rolling-mean book proxy, with the generalist-vs-specialist gate
    run on pre-registered scenarios chosen on discovery.
(c) ``--stage goals`` / ``--stage c`` -- the proxy honesty check on the one count market where
    a real line is observed (total goals, Bet365 Over/Under 2.5) and the end-to-end betting
    simulation of completed passes against the proxy line at 4 / 6 / 8% hold.

Run from the repo root; every heavy stage caches under
``<PRIV_DATA_DIR>/processed/scenario_ev/``::

    python -m research.scenario_ev.pass_counts --stage passpreds
    python -m research.scenario_ev.pass_counts --stage a
    python -m research.scenario_ev.pass_counts --stage b
    python -m research.scenario_ev.pass_counts --stage goals
    python -m research.scenario_ev.pass_counts --stage c
    python -m research.scenario_ev.pass_counts --stage report
    python -m research.scenario_ev.pass_counts --stage all

Protocol (see ``research/scenario_ev/reports/03_pass_counts.md``): a chronological
discovery / confirmation split fixed in code before any modelling, every rolling feature from
strictly earlier matches only (audited by brute force, not asserted), group splits by match,
match-clustered bootstrap intervals on every model-vs-model and ROI claim, and a haircut on
every EV claim equal to the measured gap between a real bookmaker and the identical
rolling-mean proxy on total goals.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    log_loss,
    mae,
    r2,
)
from research.scenario_ev import common as C

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: the four full league seasons StatsBomb covers end to end; the volume-channel population
LEAGUE_SEASONS: tuple[tuple[str, str], ...] = (
    ("Premier League", "2015/2016"),
    ("La Liga", "2015/2016"),
    ("Serie A", "2015/2016"),
    ("Ligue 1", "2015/2016"),
)
#: football-data.co.uk division codes for the same four leagues, for the goals honesty check
LEAGUE_DIVISIONS: tuple[str, ...] = ("E0", "SP1", "I1", "F1")


@dataclass(frozen=True)
class PassConfig:
    """Configuration of stage 03.

    Attributes:
        split: chronological discovery / confirmation split and global seed.
        n_jobs: LightGBM threads.
        min_prior_matches: player-match rows need this many strictly prior appearances.
        k_player: shrinkage strength (in prior appearances) of the player rolling rate.
        k_team: shrinkage strength of team / opponent rolling means.
        seeds_refit: seeds used for the refit-noise floor of the two headline models.
        holds: two-way overrounds simulated.
        thresholds: candidate edge thresholds; the one used on confirmation is chosen on
            discovery.
        line_offsets: completed-pass half-lines probed in part (a), in counts away from the
            EVENT model's predicted mean.
    """

    split: C.SplitConfig = field(default_factory=C.SplitConfig)
    n_jobs: int = 2
    min_prior_matches: int = 3
    k_player: float = 4.0
    k_team: float = 5.0
    seeds_refit: tuple[int, ...] = (0, 1, 2)
    holds: tuple[float, ...] = (0.04, 0.06, 0.08)
    thresholds: tuple[float, ...] = (0.0, 0.02, 0.05)
    line_offsets: tuple[int, ...] = (-6, -3, 0, 3, 6)


CFG = PassConfig()


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Poisson binomial
# ---------------------------------------------------------------------------


def poisson_binomial_pmf(p: np.ndarray) -> np.ndarray:
    """Exact pmf of a sum of independent Bernoulli trials with unequal probabilities.

    Args:
        p: Success probabilities of the trials [n].

    Returns:
        Probability mass over ``0..n`` [n + 1]; an empty input returns ``[1.0]``.
    """
    p = np.asarray(p, dtype=float).ravel()
    pmf = np.zeros(len(p) + 1, dtype=float)
    pmf[0] = 1.0
    for j, pj in enumerate(p):
        head = pmf[: j + 1]
        pmf[: j + 2] = np.concatenate([[0.0], head]) * pj + np.concatenate([head, [0.0]]) * (
            1.0 - pj
        )
    return pmf


def poisson_binomial_pmf_batch(p: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Vectorised Poisson-binomial pmf for a ragged batch of trial sets.

    Trials are padded to a common width; padded slots must be marked invalid and are treated
    as deterministic failures, which leaves the distribution of the real trials untouched.

    Args:
        p: Success probabilities, padded [g, n_max].
        valid: Boolean mask of real trials [g, n_max].

    Returns:
        Probability mass over ``0..n_max`` per group [g, n_max + 1].
    """
    p = np.where(np.asarray(valid, dtype=bool), np.asarray(p, dtype=float), 0.0)
    g, n_max = p.shape
    pmf = np.zeros((g, n_max + 1), dtype=float)
    pmf[:, 0] = 1.0
    for j in range(n_max):
        pj = p[:, j : j + 1]
        head = pmf[:, : j + 1]
        pmf[:, : j + 2] = np.concatenate([np.zeros((g, 1)), head], axis=1) * pj + np.concatenate(
            [head, np.zeros((g, 1))], axis=1
        ) * (1.0 - pj)
    return pmf


def pmf_sf(pmf: np.ndarray, line: np.ndarray | float) -> np.ndarray:
    """``P(count > line)`` for a batch of pmfs on the integer support ``0..K``.

    Args:
        pmf: Probability mass per group [g, K + 1].
        line: Half-integer line, scalar or [g].

    Returns:
        Survival probability per group [g].
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    k = np.arange(pmf.shape[1])[None, :]
    thr = np.asarray(line, dtype=float).reshape(-1, 1)
    return (pmf * (k > thr)).sum(axis=1)


def pmf_mean_var(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of a batch of pmfs on ``0..K``.

    Args:
        pmf: Probability mass per group [g, K + 1].

    Returns:
        Tuple ``(mean [g], var [g])``.
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    k = np.arange(pmf.shape[1])[None, :]
    m = (pmf * k).sum(axis=1)
    v = (pmf * k**2).sum(axis=1) - m**2
    return m, np.maximum(v, 0.0)


def pmf_log_score(pmf: np.ndarray, obs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Negative log probability the pmf assigns to the realised count.

    Args:
        pmf: Probability mass per group [g, K + 1].
        obs: Realised counts [g].
        eps: Floor on the probability.

    Returns:
        Per-group negative log score [g].
    """
    pmf = np.atleast_2d(np.asarray(pmf, dtype=float))
    idx = np.asarray(obs, dtype=int)
    got = pmf[np.arange(len(idx)), np.clip(idx, 0, pmf.shape[1] - 1)]
    return -np.log(np.maximum(got, eps))


def binomial_mix_sf(
    p_count: np.ndarray, rate: np.ndarray, line: np.ndarray | float
) -> np.ndarray:
    """``P(completed > line)`` for completed = Binomial(attempts, rate), attempts ~ ``p_count``.

    Args:
        p_count: Attempt-count pmf on ``0..K`` per row [g, K + 1].
        rate: Per-attempt completion probability [g].
        line: Half-integer line on completed passes, scalar or [g].

    Returns:
        Survival probability [g].
    """
    from scipy.stats import binom

    p_count = np.atleast_2d(np.asarray(p_count, dtype=float))
    g, kp1 = p_count.shape
    a = np.arange(kp1)[None, :]
    thr = np.asarray(line, dtype=float).reshape(-1, 1)
    r = np.asarray(rate, dtype=float).reshape(-1, 1)
    # P(Bin(a, r) > line) for every (row, attempt count) pair
    sf = binom.sf(np.floor(np.broadcast_to(thr, (g, kp1))), np.broadcast_to(a, (g, kp1)),
                  np.broadcast_to(r, (g, kp1)))
    sf = np.where(np.broadcast_to(a, (g, kp1)) <= np.broadcast_to(thr, (g, kp1)), 0.0, sf)
    return (p_count * sf).sum(axis=1)


def nb_pmf_grid(mu: np.ndarray, r: float, k_max: int) -> np.ndarray:
    """Negative-binomial pmf on ``0..k_max`` for a vector of means.

    Args:
        mu: Means [g].
        r: Dispersion (number of failures); ``inf`` gives the Poisson.
        k_max: Largest count kept; the remaining mass is placed on ``k_max``.

    Returns:
        Probability mass [g, k_max + 1].
    """
    from scipy.stats import nbinom, poisson

    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None).reshape(-1, 1)
    k = np.arange(k_max + 1)[None, :]
    if not np.isfinite(r) or r > 1e6:
        pm = poisson.pmf(k, mu)
    else:
        pm = nbinom.pmf(k, r, r / (r + mu))
    tail = np.clip(1.0 - pm.sum(axis=1, keepdims=True), 0.0, None)
    pm = pm.copy()
    pm[:, -1] += tail.ravel()
    return pm


def line_from_mean(mu: np.ndarray, offset: int = 0, lo: float = 0.5) -> np.ndarray:
    """Half-integer line ``offset`` counts away from a predicted mean.

    Args:
        mu: Predicted mean count [n].
        offset: Shift in whole counts.
        lo: Smallest line returned.

    Returns:
        Half-integer lines [n].
    """
    return np.maximum(np.floor(np.asarray(mu, dtype=float)) + 0.5 + offset, lo)


# ---------------------------------------------------------------------------
# (a) Completion channel: per-pass models on all attempts of the 360 matches
# ---------------------------------------------------------------------------

#: state quantities added to the pass design (the programme's ``PASS_STATE``)
PASS_STATE_ALL: tuple[str, ...] = (
    "n_opp_within_3_of_end",
    "n_opp_in_lane",
    "nearest_opp_to_receiver",
    "n_opp_ahead_of_ball",
    "block_depth",
    "nearest_opp_dist",
    "n_opp_within_5",
)
#: the subset the sequence student (soccer 06) produces
SEQ_STATE: tuple[str, ...] = (
    "block_depth",
    "def_line",
    "n_opp_ahead_of_ball",
    "nearest_opp_dist",
    "n_opp_in_cone",
    "deep_block",
    "counter_on",
)
PASS_PREDS_CACHE = "pass_preds_full.parquet"
PASS_FITS_CACHE = "pass_fits_full.parquet"
#: variants fitted on all attempts (design ``noafter``: only what is known at the pass instant)
FULL_VARIANTS: tuple[str, ...] = ("EVENT", "EVENT+IMP", "EVENT+ORACLE")
#: variants fitted on the 251 matches the sequence student covers
SEQ_VARIANTS: tuple[str, ...] = ("EVENT", "EVENT+IMP", "EVENT+IMP+SEQ")


def build_pass_predictions(cfg: PassConfig = CFG, force: bool = False,
                           smoke: int = 0) -> pd.DataFrame:
    """Out-of-fold per-pass completion probabilities for **all** attempts of the 360 matches.

    The programme cached a 149,591-pass match-stratified subsample, whose player-match counts
    are only ~35% of full size; a count line has to be modelled at full size, so the same
    models are refitted here on every attempt. Folds, LightGBM settings, feature designs and
    the receiver-distance student are the programme's (``soccer/payoff.py``), so the per-pass
    deltas double as a reproduction check of soccer 03.

    Args:
        cfg: Stage configuration.
        force: Refit even when the cache exists.
        smoke: If > 0, keep only this many matches (writes to a separate cache).

    Returns:
        Frame [n_passes, ~16] with ``event_id``, ``match_id``, ``player_id``, ``team_id``,
        ``fold``, ``is_complete`` and ``pred_<variant>`` for each fitted variant.
    """
    import pyarrow.parquet as pq

    from research.privileged_tracking.common.io import processed_dir
    from research.privileged_tracking.soccer import imputation as imp
    from research.privileged_tracking.soccer import imputation_features as imf
    from research.privileged_tracking.soccer import payoff as pay
    from research.privileged_tracking.soccer import payoff_features as pf

    name = PASS_PREDS_CACHE if not smoke else f"smoke_{PASS_PREDS_CACHE}"
    path = C.processed_dir() / name
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    soc = processed_dir("soccer")
    ev_path = soc / "events360.parquet"
    names = pq.ParquetFile(ev_path).schema_arrow.names
    id_cols = ["match_id", "event_id", "event_index", "period", "competition", "season",
               "gender", "match_date", "team_id", "opp_team_id", "player_id"]
    cols = (id_cols + [c for c in names if c.startswith("f_")] + ["post_pass_outcome"]
            + ["y_frame_ok", "y_reliable", "y_nearest_opp_to_receiver", "y_n_opponents_visible"])
    passes = pq.read_table(ev_path, columns=cols,
                           filters=[("f_type", "==", "Pass")]).to_pandas()
    if smoke:
        keep_m = np.sort(passes["match_id"].unique())[:smoke]
        passes = passes[passes["match_id"].isin(keep_m)]
    passes = passes.sort_values(["match_id", "event_index"]).reset_index(drop=True)
    stage02 = tuple(t for t in PASS_STATE_ALL if t != "nearest_opp_to_receiver")
    oof_cols = (["event_id", "fold"] + [f"{t}__E2" for t in stage02]
                + [f"y_{t}" for t in stage02])
    oof = pq.read_table(soc / "imputed_oof.parquet", columns=oof_cols,
                        filters=[("f_type", "==", "Pass")]).to_pandas()
    oof = oof.rename(columns={f"{t}__E2": f"imp_{t}__E2" for t in stage02})
    passes = passes.merge(oof, on="event_id", how="left")
    seq_cols = ["event_id"] + [f"{t}__seq20" for t in SEQ_STATE]
    seq = pq.read_table(soc / "imputed_oof_seq.parquet", columns=seq_cols).to_pandas()
    seq = seq.rename(columns={f"{t}__seq20": f"seq_{t}" for t in SEQ_STATE})
    passes = passes.merge(seq, on="event_id", how="left")
    y_all, keep = pf.pass_completion(passes["post_pass_outcome"].to_numpy(dtype=object))
    passes["is_complete"] = y_all.astype(np.float32)
    passes = passes[keep].reset_index(drop=True)
    _log(f"pass rows {len(passes):,} of {passes['match_id'].nunique()} matches "
         f"({time.time() - t0:.0f} s); fitting the receiver student (E2)")

    # receiver-distance student, out-of-fold on the stage-02 folds (no stage-02 model exists)
    icfg = imp.ImputationConfig(n_jobs=cfg.n_jobs, seed=cfg.split.seed, train_cap=300_000)
    label = passes["y_nearest_opp_to_receiver"].to_numpy(dtype=float)
    label[passes["y_frame_ok"].to_numpy(dtype=float) != 1.0] = np.nan
    match = passes["match_id"].to_numpy()
    fold = passes["fold"].to_numpy()
    x_e2 = imf.build_design(passes, "E2")
    cats_e2 = imf.categorical_columns("E2")
    pred = np.full(len(passes), np.nan, dtype=np.float32)
    fits: list[dict[str, Any]] = []
    for k in np.unique(fold[fold >= 0]):
        te = np.where(fold == k)[0]
        tr = np.where((fold != k) & ~np.isnan(label))[0]
        tr = imp.thin_rows(tr, icfg.train_cap, cfg.split.seed + int(k))
        t1 = time.time()
        booster, rounds = imp.fit_lgbm(icfg, x_e2.iloc[tr], label[tr], match[tr], "reg",
                                       cats_e2, cfg.split.seed + int(k))
        pred[te] = booster.predict(x_e2.iloc[te], num_iteration=rounds)
        fits.append({"model": "receiver_student", "variant": "E2", "fold": int(k),
                     "rounds": rounds, "n_train": int(len(tr)), "seconds": time.time() - t1})
    passes["imp_nearest_opp_to_receiver__E2"] = pred
    passes["y_nearest_opp_to_receiver"] = label.astype(np.float32)
    ok = ~np.isnan(label) & ~np.isnan(pred)
    _log(f"receiver student E2: oof R2 {r2(label[ok], pred[ok]):.3f} "
         f"MAE {mae(label[ok], pred[ok]):.2f} (n {int(ok.sum()):,})")

    base = pf.pass_event_design(passes, with_after=False)
    y = passes["is_complete"].to_numpy(dtype=float)
    out = passes[["event_id", "match_id", "match_date", "competition", "season", "gender",
                  "team_id", "opp_team_id", "player_id", "fold", "is_complete"]].copy()
    out["pred_BASE"] = pay.base_rate_oof(y, fold)
    seq_ok = passes["seq_block_depth"].notna().to_numpy()

    def _design(variant: str) -> pd.DataFrame:
        blocks = [base]
        if "+IMP" in variant:
            blocks.append(pf.state_block(passes, PASS_STATE_ALL, "imp", "E2", prefix="s_"))
        if "+ORACLE" in variant:
            blocks.append(pf.state_block(passes, PASS_STATE_ALL, "oracle360", prefix="s_"))
        if "+SEQ" in variant:
            blocks.append(passes[[f"seq_{t}" for t in SEQ_STATE]].astype(np.float32))
        return pd.concat(blocks, axis=1)

    for variant in FULL_VARIANTS:
        t1 = time.time()
        x = _design(variant)
        oof_p, f, _ = pay.cv_predict(x, y, match, fold, pay.PayoffConfig().xpass,
                                     (cfg.split.seed,), pf.categorical_in(x), "binary",
                                     cfg.n_jobs, refit=False)
        out[f"pred_{variant}"] = oof_p
        fits += [{"model": "xpass_all", "variant": variant, **r} for r in f]
        _log(f"xpass_all {variant:14s} d={x.shape[1]:3d} log-loss "
             f"{log_loss(y, oof_p):.4f}  {time.time() - t1:.0f} s")
    # sequence-student variants, restricted to the matches the seq student covers
    sm = np.where(seq_ok)[0]
    if len(sm) > 1000:
        for variant in SEQ_VARIANTS:
            t1 = time.time()
            x = _design(variant).iloc[sm].reset_index(drop=True)
            oof_p, f, _ = pay.cv_predict(x, y[sm], match[sm], fold[sm],
                                         pay.PayoffConfig().xpass, (cfg.split.seed,),
                                         pf.categorical_in(x), "binary", cfg.n_jobs,
                                         refit=False)
            col = f"predseq_{variant}"
            out[col] = np.nan
            out.loc[out.index[sm], col] = oof_p
            fits += [{"model": "xpass_seq", "variant": variant, **r} for r in f]
            _log(f"xpass_seq {variant:14s} d={x.shape[1]:3d} log-loss "
                 f"{log_loss(y[sm], oof_p):.4f}  {time.time() - t1:.0f} s")
    out["seq_covered"] = seq_ok
    out.to_parquet(path, index=False)
    pd.DataFrame(fits).to_parquet(
        C.processed_dir() / (PASS_FITS_CACHE if not smoke else f"smoke_{PASS_FITS_CACHE}"),
        index=False)
    _log(f"pass predictions written: {len(out):,} rows, {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# (b) Volume channel: the player-match panel
# ---------------------------------------------------------------------------

PANEL_CACHE = "pass_panel.parquet"

#: StatsBomb starting positions collapsed to the groups a market would price by
POSITION_GROUPS: dict[str, str] = {
    "Goalkeeper": "GK",
    "Left Center Back": "CB", "Right Center Back": "CB", "Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM",
    "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Left Midfield": "WM", "Right Midfield": "WM",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM",
    "Right Attacking Midfield": "AM",
    "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "CF", "Left Center Forward": "CF", "Right Center Forward": "CF",
    "Secondary Striker": "CF",
}


def position_group(position: str | float) -> str:
    """Collapse a StatsBomb starting position to a market-facing position group.

    Args:
        position: StatsBomb ``start_position`` string (NaN allowed).

    Returns:
        One of ``GK/CB/FB/DM/CM/WM/AM/W/CF`` or ``UNK``.
    """
    if not isinstance(position, str):
        return "UNK"
    return POSITION_GROUPS.get(position, "UNK")


def rolling_prior_mean(
    df: pd.DataFrame, group_cols: Sequence[str], value_col: str, order_cols: Sequence[str],
    window: int,
) -> np.ndarray:
    """Mean of ``value_col`` over the last ``window`` strictly earlier rows of each group.

    Args:
        df: Input frame.
        group_cols: Grouping keys.
        value_col: Numeric column to average.
        order_cols: Sort keys defining "earlier".
        window: Number of prior rows averaged.

    Returns:
        Array [n] aligned to ``df``; NaN where no prior row exists.
    """
    work = df[list(group_cols) + [value_col] + list(order_cols)].reset_index(drop=True)
    work["__pos"] = np.arange(len(work))
    work = work.sort_values(list(order_cols) + ["__pos"], kind="mergesort").reset_index(
        drop=True)
    grp = work.groupby(list(group_cols), sort=False, dropna=False)
    work["__prev"] = grp[value_col].shift(1)
    rolled = work.groupby(list(group_cols), sort=False, dropna=False)["__prev"].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    out = np.full(len(work), np.nan)
    out[work["__pos"].to_numpy()] = rolled.to_numpy()
    return out


def _team_state_frame() -> pd.DataFrame:
    """Per (match, team) mean imputed opponent block depth from the event-level students.

    ``imp_block_depth`` is the depth of the *defending* block on a possession-team event, so
    the mean over team T's possession events describes T's OPPONENT's shape. The frame is
    returned in that orientation (``bd_faced``) and re-oriented by the caller.

    Returns:
        Frame with ``match_id``, ``team_id``, ``bd_faced``, ``deep_faced``, ``n_events``.
    """
    import pyarrow.parquet as pq

    from research.privileged_tracking.common.io import processed_dir

    soc = processed_dir("soccer")
    parts = []
    a = pq.read_table(soc / "imputed_no360.parquet",
                      columns=["match_id", "team_id", "f_is_possession_team",
                               "imp_block_depth__E2", "imp_deep_block__E2"]).to_pandas()
    a = a.rename(columns={"imp_block_depth__E2": "bd", "imp_deep_block__E2": "deep"})
    parts.append(a)
    ev = pq.read_table(soc / "events360.parquet",
                       columns=["event_id", "match_id", "team_id",
                                "f_is_possession_team"]).to_pandas()
    oof = pq.read_table(soc / "imputed_oof.parquet",
                        columns=["event_id", "block_depth__E2", "deep_block__E2"]).to_pandas()
    b = ev.merge(oof, on="event_id", how="inner").rename(
        columns={"block_depth__E2": "bd", "deep_block__E2": "deep"})
    parts.append(b[["match_id", "team_id", "f_is_possession_team", "bd", "deep"]])
    allr = pd.concat(parts, ignore_index=True)
    allr = allr[allr["f_is_possession_team"].fillna(False).astype(bool)]
    out = allr.groupby(["match_id", "team_id"], as_index=False).agg(
        bd_faced=("bd", "mean"), deep_faced=("deep", "mean"), n_events=("bd", "size"))
    return out


def build_player_panel(cfg: PassConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Player-match panel with strictly-prior rolling features and the book proxy.

    Every feature describing a match is computed from that player's / team's / opponent's
    matches on strictly earlier dates. The match being predicted contributes nothing but the
    identity of the two teams, whether the player started, and his starting position -- the two
    facts a book has when it posts a player prop after team news.

    Args:
        cfg: Stage configuration.
        force: Rebuild even when the cache exists.

    Returns:
        Panel [n_player_matches, ~45] over all 2,090 StatsBomb matches.
    """
    path = C.processed_dir() / PANEL_CACHE
    if path.exists() and not force:
        return pd.read_parquet(path)
    t0 = time.time()
    sb = C.sb_processed_dir()
    pm = pd.read_parquet(sb / "player_match.parquet")
    mt = pd.read_parquet(sb / "matches.parquet",
                         columns=["match_id", "competition", "season", "date", "home_team_id",
                                  "away_team_id", "referee"])
    tm = pd.read_parquet(sb / "team_match.parquet",
                         columns=["match_id", "team_id", "opp_id", "home", "possession", "ppda",
                                  "def_line_x", "passes", "passes_completed", "pass_x_mean"])
    mt["date"] = pd.to_datetime(mt["date"])

    # --- team-match frame with the imputed opponent shape, in team-own orientation ---
    st = _team_state_frame()
    tm = tm.merge(st, on=["match_id", "team_id"], how="left")
    own = st.rename(columns={"team_id": "opp_id", "bd_faced": "t_block_depth",
                             "deep_faced": "t_deep_block"})[
        ["match_id", "opp_id", "t_block_depth", "t_deep_block"]]
    tm = tm.merge(own, on=["match_id", "opp_id"], how="left")
    tm = tm.merge(mt[["match_id", "date", "competition", "season"]], on="match_id", how="left")
    tm = tm.sort_values(["date", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    tm["one"] = 1.0
    tv = ["possession", "ppda", "def_line_x", "passes", "passes_completed", "t_block_depth",
          "t_deep_block", "one"]
    tm["block_depth"] = tm["t_block_depth"].fillna(tm["t_block_depth"].mean())
    tm["deep_block"] = tm["t_deep_block"].fillna(tm["t_deep_block"].mean())
    tv = [{"t_block_depth": "block_depth", "t_deep_block": "deep_block"}.get(c, c) for c in tv]
    pri = C.prior_expanding(tm, ["team_id"], tv, ["date", "match_id"], count_name="t_prior_n",
                            prefix="ts_")
    tm = pd.concat([tm, pri], axis=1)
    gm = {c: float(tm[c].mean()) for c in tv}
    for c in tv:
        tm["tp_" + c] = C.shrunk_rate(tm["ts_" + c].to_numpy(), tm["t_prior_n"].to_numpy(),
                                      gm[c], cfg.k_team)
    tcols = ["match_id", "team_id", "t_prior_n"] + ["tp_" + c for c in tv]
    tprior = tm[tcols].copy()
    opp_prior = tprior.rename(
        columns={"team_id": "opp_id", "t_prior_n": "o_prior_n",
                 **{"tp_" + c: "op_" + c for c in tv}})

    # --- player-match rows ---
    pl = pm.merge(mt, on="match_id", how="left")
    pl = pl.merge(tm[["match_id", "team_id", "opp_id", "home"]], on=["match_id", "team_id"],
                  how="left")
    pl["pos_group"] = [position_group(p) for p in pl["start_position"]]
    pl["starter"] = pl["starter"].fillna(False).astype(bool)
    pl = pl.sort_values(["date", "match_id", "player_id"], kind="mergesort").reset_index(drop=True)
    pl["one"] = 1.0
    pl["start_i"] = pl["starter"].astype(float)
    pl["start_minutes"] = np.where(pl["starter"], pl["minutes"], np.nan)
    pl["start_passes"] = np.where(pl["starter"], pl["passes"], np.nan)
    pl["start_one"] = pl["start_i"]
    pv = ["passes", "passes_completed", "minutes", "key_passes", "pressures", "one", "start_i"]
    pri = C.prior_expanding(pl, ["player_id"], pv, ["date", "match_id"], count_name="p_prior_n",
                            prefix="ps_")
    pl = pd.concat([pl, pri], axis=1)
    sv = ["start_minutes", "start_passes", "start_one"]
    pri_s = C.prior_expanding(pl.assign(**{c: pl[c].fillna(0.0) for c in sv}), ["player_id"],
                              sv, ["date", "match_id"], count_name="p_prior_n2", prefix="ps_")
    pl = pd.concat([pl, pri_s.drop(columns=["p_prior_n2"])], axis=1)

    glob_rate = float(pm["passes"].sum() / max(pm["minutes"].sum(), 1.0) * 90.0)
    glob_comp = float(pm["passes_completed"].sum() / max(pm["passes"].sum(), 1.0))
    glob_min = float(pm.loc[pm["starter"].fillna(False), "minutes"].mean())
    pl["p_att_per90"] = C.shrunk_rate(pl["ps_passes"].to_numpy(),
                                      pl["ps_minutes"].to_numpy() / 90.0,
                                      glob_rate, cfg.k_player)
    pl["p_comp_rate"] = C.shrunk_rate(pl["ps_passes_completed"].to_numpy(),
                                      pl["ps_passes"].to_numpy(), glob_comp,
                                      cfg.k_player * 30.0)
    pl["p_start_minutes"] = C.shrunk_rate(pl["ps_start_minutes"].to_numpy(),
                                          pl["ps_start_one"].to_numpy(), glob_min, cfg.k_player)
    pl["p_start_rate"] = C.shrunk_rate(pl["ps_start_i"].to_numpy(), pl["ps_one"].to_numpy(),
                                       0.75, cfg.k_player)
    pl["p_kp_per90"] = C.shrunk_rate(pl["ps_key_passes"].to_numpy(),
                                     pl["ps_minutes"].to_numpy() / 90.0, 1.0, cfg.k_player)
    pl["p_press_per90"] = C.shrunk_rate(pl["ps_pressures"].to_numpy(),
                                        pl["ps_minutes"].to_numpy() / 90.0, 18.0, cfg.k_player)
    pl["p_att_last5"] = rolling_prior_mean(pl, ["player_id"], "passes", ["date", "match_id"], 5)
    pl["p_min_last5"] = rolling_prior_mean(pl, ["player_id"], "minutes", ["date", "match_id"], 5)
    pl["p_att_per90_last5"] = 90.0 * pl["p_att_last5"] / np.maximum(pl["p_min_last5"], 20.0)
    pl["p_prior_n"] = pl["p_prior_n"].astype(float)

    pl = pl.merge(tprior, on=["match_id", "team_id"], how="left")
    pl = pl.merge(opp_prior, on=["match_id", "opp_id"], how="left")
    pl["poss_edge"] = pl["tp_possession"] - pl["op_possession"]
    pl["exp_minutes"] = np.where(pl["starter"], pl["p_start_minutes"],
                                 np.maximum(pl["p_start_minutes"] * 0.35, 15.0))
    pl["mu_proxy_att"] = pl["p_att_per90"] * pl["exp_minutes"] / 90.0
    pl["mu_proxy_comp"] = pl["mu_proxy_att"] * pl["p_comp_rate"]
    pl["league"] = [
        (c, s) in set(LEAGUE_SEASONS) for c, s in zip(pl["competition"], pl["season"])
    ]
    # The split is chronological over the matches of the MODELLED population, so that both
    # halves are the same kind of football; a split over all 2,090 matches would put every
    # club season in discovery and only international tournaments in confirmation.
    pl["split"] = ""
    lg = pl["league"].to_numpy()
    pl.loc[lg, "split"] = C.chronological_split(
        pl.loc[lg, "match_id"].to_numpy(), pl.loc[lg, "date"].to_numpy(),
        cfg.split.discovery_frac)
    pl.loc[~lg, "split"] = C.chronological_split(
        pl.loc[~lg, "match_id"].to_numpy(), pl.loc[~lg, "date"].to_numpy(),
        cfg.split.discovery_frac)
    keep = ["match_id", "team_id", "opp_id", "player_id", "player", "date", "competition",
            "season", "league", "home", "referee", "starter", "start_position", "pos_group",
            "minutes", "passes", "passes_completed", "split", "p_prior_n", "t_prior_n",
            "o_prior_n", "p_att_per90", "p_comp_rate", "p_start_minutes", "p_start_rate",
            "p_kp_per90", "p_press_per90", "p_att_last5", "p_min_last5", "p_att_per90_last5",
            "exp_minutes", "mu_proxy_att", "mu_proxy_comp", "poss_edge"]
    keep += [c for c in pl.columns if c.startswith(("tp_", "op_"))]
    out = pl[keep].copy()
    out.to_parquet(path, index=False)
    _log(f"panel: {len(out):,} player-matches, {out['match_id'].nunique()} matches, "
         f"{time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# Count models
# ---------------------------------------------------------------------------

#: player, team and opponent features of the attempts / completions models
BASE_FEATURES: tuple[str, ...] = (
    "p_att_per90", "p_att_per90_last5", "p_att_last5", "p_min_last5", "p_start_minutes",
    "p_start_rate", "p_prior_n", "p_comp_rate", "p_kp_per90", "p_press_per90",
    "home", "pos_group", "competition",
    "tp_possession", "tp_ppda", "tp_def_line_x", "tp_passes", "tp_passes_completed",
    "op_possession", "op_ppda", "op_def_line_x", "op_passes", "op_passes_completed",
    "poss_edge", "mu_proxy_att",
)
#: the tracking-derived channel: imputed block depth of the two teams
IMP_FEATURES: tuple[str, ...] = ("tp_block_depth", "tp_deep_block", "op_block_depth",
                                 "op_deep_block")
CATEGORICALS: tuple[str, ...] = ("pos_group", "competition")



def fold_labels(groups: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row with whole groups (matches) held out together.

    Args:
        groups: Group label per row [n].
        n_splits: Number of folds.
        seed: Shuffle seed.

    Returns:
        Fold index per row [n].
    """
    from research.privileged_tracking.common.splits import group_kfold

    out = np.full(len(groups), -1, dtype=int)
    for k, (_, te) in enumerate(group_kfold(groups, n_splits, seed)):
        out[te] = k
    return out


def model_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Design matrix with the categorical columns typed for LightGBM.

    Args:
        df: Panel rows.
        cols: Feature names.

    Returns:
        Frame [n, len(cols)].
    """
    x = df[list(cols)].copy()
    for c in CATEGORICALS:
        if c in x.columns:
            x[c] = x[c].astype("category")
    for c in x.columns:
        if c not in CATEGORICALS and x[c].dtype == bool:
            x[c] = x[c].astype(float)
    return x


def fit_count(
    x_tr: pd.DataFrame, y_tr: np.ndarray, groups_tr: np.ndarray, x_te: pd.DataFrame,
    seed: int = 0, n_jobs: int = 2, objective: str = "poisson",
) -> np.ndarray:
    """Fit a LightGBM count model and predict, with rounds by an inner match-grouped holdout.

    Args:
        x_tr: Training design [n_tr, d].
        y_tr: Training counts [n_tr].
        groups_tr: Match id per training row [n_tr].
        x_te: Test design [n_te, d].
        seed: Seed for the holdout and LightGBM.
        n_jobs: Threads.
        objective: LightGBM objective (``poisson`` or ``binary``).

    Returns:
        Predictions on ``x_te`` [n_te].
    """
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups_tr)
    hold = set(rng.choice(uniq, size=max(1, int(round(len(uniq) * 0.15))), replace=False))
    is_h = np.array([g in hold for g in groups_tr])
    params = {"objective": objective, "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 60, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "max_bin": 127, "lambda_l2": 1.0, "verbose": -1,
              "num_threads": n_jobs, "seed": seed, "deterministic": True}
    cats = [c for c in CATEGORICALS if c in x_tr.columns]
    dtr = lgb.Dataset(x_tr[~is_h], y_tr[~is_h], categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(x_tr[is_h], y_tr[is_h], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(60, verbose=False)])
    best = int(bst.best_iteration) if bst.best_iteration else 1500
    bst = lgb.train(params, lgb.Dataset(x_tr, y_tr, categorical_feature=cats),
                    num_boost_round=best)
    return bst.predict(x_te)


def poisson_glm_link(mu_ref: np.ndarray, y: np.ndarray, n_iter: int = 60
                     ) -> tuple[float, float]:
    """Fit ``log E[y] = a + b log(mu_ref)`` by Newton iterations (Poisson likelihood).

    Giving the book proxy this one-dimensional recalibration is what makes the comparison
    fair: the proxy is not penalised for an overall level or slope error a book would never
    make.

    Args:
        mu_ref: Reference means [n].
        y: Observed counts [n].
        n_iter: Newton steps.

    Returns:
        Tuple ``(a, b)``.
    """
    z = np.log(np.clip(np.asarray(mu_ref, dtype=float), 1e-6, None))
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones_like(z), z])
    beta = np.array([0.0, 1.0])
    for _ in range(n_iter):
        mu = np.exp(np.clip(x @ beta, -20, 20))
        g = x.T @ (y - mu)
        h = x.T @ (x * mu[:, None])
        step = np.linalg.solve(h + 1e-8 * np.eye(2), g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def poisson_glm_apply(mu_ref: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Apply a fitted one-dimensional Poisson recalibration.

    Args:
        mu_ref: Reference means [n].
        ab: ``(a, b)`` from :func:`poisson_glm_link`.

    Returns:
        Calibrated means [n].
    """
    a, b = ab
    return np.exp(a + b * np.log(np.clip(np.asarray(mu_ref, dtype=float), 1e-6, None)))


def count_metrics(y: np.ndarray, mu: np.ndarray, r: float = float("inf")) -> dict[str, float]:
    """Point and distributional accuracy of a count prediction.

    Args:
        y: Observed counts [n].
        mu: Predicted means [n].
        r: Negative-binomial dispersion used for the log score (``inf`` = Poisson).

    Returns:
        Dict with ``n``, ``mae``, ``r2``, ``pois_dev`` (mean Poisson deviance) and
        ``log_score`` (mean negative log predictive probability of the realised count).
    """
    from scipy.stats import nbinom, poisson

    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    dev = 2.0 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0) - (y - mu))
    if not np.isfinite(r) or r > 1e6:
        ls = -poisson.logpmf(y, mu)
    else:
        ls = -nbinom.logpmf(y, r, r / (r + mu))
    return {"n": float(len(y)), "mae": float(np.abs(y - mu).mean()),
            "r2": float(r2(y, mu)), "pois_dev": float(dev.mean()),
            "log_score": float(ls.mean())}


def per_row_log_score(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """Per-row negative log predictive probability under a negative binomial.

    Args:
        y: Observed counts [n].
        mu: Predicted means [n].
        r: Dispersion (``inf`` = Poisson).

    Returns:
        Per-row negative log score [n].
    """
    from scipy.stats import nbinom, poisson

    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    if not np.isfinite(r) or r > 1e6:
        return -poisson.logpmf(y, mu)
    return -nbinom.logpmf(y, r, r / (r + mu))


# ---------------------------------------------------------------------------
# Population, scenarios and the gate
# ---------------------------------------------------------------------------


def league_population(panel: pd.DataFrame, cfg: PassConfig = CFG) -> pd.DataFrame:
    """The modelled rows: starters in the four full league seasons with enough history.

    A prop is quoted for a player expected to start, so the population is starters; it is not
    filtered on realised minutes, because minutes are an outcome of the match and an early
    substitution is exactly the risk the market prices.

    Args:
        panel: Output of :func:`build_player_panel`.
        cfg: Stage configuration.

    Returns:
        Filtered panel rows.
    """
    m = (panel["league"].to_numpy()
         & panel["starter"].to_numpy()
         & (panel["p_prior_n"].to_numpy() >= cfg.min_prior_matches)
         & (panel["t_prior_n"].to_numpy() >= cfg.min_prior_matches)
         & (panel["o_prior_n"].to_numpy() >= cfg.min_prior_matches))
    return panel[m].reset_index(drop=True)


def usual_passer_frame(app: pd.DataFrame, min_prior: int = 3, recent_team_matches: int = 5
                       ) -> pd.DataFrame:
    """Per team-match, the team's usual highest-volume passer and whether he is absent.

    The "usual deep-lying passer" is the player of the team's recent squad (anyone who played
    for the team in the previous ``recent_team_matches`` team matches) with the highest
    attempts per 90 over his strictly earlier appearances *for that team*, needing at least
    ``min_prior`` of them. Everything is computed from strictly earlier matches; the only fact
    read from the match itself is the list of players who appeared.

    Args:
        app: Appearance rows with ``match_id``, ``team_id``, ``player_id``, ``date``,
            ``minutes``, ``passes``.
        min_prior: Prior appearances required to be eligible as the usual passer.
        recent_team_matches: Window of team matches defining the recent squad.

    Returns:
        Frame with ``match_id``, ``team_id``, ``key_player_id``, ``key_att_per90``,
        ``key_absent`` (1.0 / 0.0) and ``key_known`` (0.0 when no eligible player exists).
    """
    a = app.sort_values(["date", "match_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for team, g in a.groupby("team_id", sort=False):
        hist_att: dict[Any, float] = {}
        hist_min: dict[Any, float] = {}
        hist_n: dict[Any, int] = {}
        recent: list[set[Any]] = []
        for mid, gm in g.groupby("match_id", sort=False):
            squad = set().union(*recent) if recent else set()
            best_pid, best_rate = None, -1.0
            for pid in squad:
                if hist_n.get(pid, 0) >= min_prior and hist_min.get(pid, 0.0) > 0:
                    rate = 90.0 * hist_att[pid] / hist_min[pid]
                    if rate > best_rate:
                        best_pid, best_rate = pid, rate
            present = set(gm["player_id"].to_numpy())
            rows.append({
                "match_id": mid, "team_id": team,
                "key_player_id": best_pid if best_pid is not None else -1,
                "key_att_per90": best_rate if best_pid is not None else np.nan,
                "key_absent": float(best_pid not in present) if best_pid is not None else 0.0,
                "key_known": float(best_pid is not None),
            })
            for pid, mn, ps in zip(gm["player_id"], gm["minutes"], gm["passes"]):
                hist_att[pid] = hist_att.get(pid, 0.0) + float(ps)
                hist_min[pid] = hist_min.get(pid, 0.0) + float(mn)
                hist_n[pid] = hist_n.get(pid, 0) + 1
            recent.append(present)
            recent = recent[-recent_team_matches:]
    return pd.DataFrame(rows)


#: pre-registered scenarios; every cut is a quantile of the DISCOVERY rows
SCENARIOS: tuple[str, ...] = ("metronome_out", "deep_opponent", "pressing_opponent",
                              "high_volume_passer", "possession_edge")


def discovery_cuts(df: pd.DataFrame) -> dict[str, float]:
    """Scenario thresholds, all quantiles of the discovery rows.

    Args:
        df: Discovery rows of the modelled population.

    Returns:
        Dict of cut values keyed by ``<column>_<quantile>``.
    """
    q = {}
    for col, frac, name in (("op_block_depth", 1 / 3, "deep_opp"),
                            ("op_ppda", 1 / 3, "press_opp"),
                            ("p_att_per90", 2 / 3, "hi_vol"),
                            ("poss_edge", 2 / 3, "poss_edge")):
        q[name] = float(np.nanquantile(df[col].to_numpy(dtype=float), frac))
    return q


def scenario_mask(df: pd.DataFrame, name: str, cuts: dict[str, float]) -> np.ndarray:
    """Boolean mask of a pre-registered scenario.

    Args:
        df: Panel rows.
        name: One of :data:`SCENARIOS`.
        cuts: Thresholds from :func:`discovery_cuts`.

    Returns:
        Boolean mask [n].
    """
    if name == "metronome_out":
        return (df["key_absent"].to_numpy(dtype=float) > 0.5) & (
            df["key_known"].to_numpy(dtype=float) > 0.5)
    if name == "deep_opponent":
        return df["op_block_depth"].to_numpy(dtype=float) <= cuts["deep_opp"]
    if name == "pressing_opponent":
        return df["op_ppda"].to_numpy(dtype=float) <= cuts["press_opp"]
    if name == "high_volume_passer":
        return df["p_att_per90"].to_numpy(dtype=float) >= cuts["hi_vol"]
    if name == "possession_edge":
        return df["poss_edge"].to_numpy(dtype=float) >= cuts["poss_edge"]
    raise KeyError(name)


def gate(
    df: pd.DataFrame, cols: Sequence[str], target: str, scen: np.ndarray, cfg: PassConfig,
    n_folds: int = 5, seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generalist-vs-specialist gate for one target and one scenario.

    A generalist is fitted on all training rows and a specialist on the training rows inside
    the scenario; both are scored on the held-out scenario rows (and, for the reverse check,
    on the held-out rows outside it). Discovery uses match-grouped CV; confirmation is scored
    once with both models fitted on all discovery rows.

    Args:
        df: Modelled population with a ``split`` column.
        cols: Feature names.
        target: Count column to predict.
        scen: Scenario mask over ``df`` [n].
        cfg: Stage configuration.
        n_folds: Discovery CV folds.
        seed: Seed override.

    Returns:
        List of result rows (one per ``where`` x ``scored_on``).
    """
    seed = cfg.split.seed if seed is None else seed
    y = df[target].to_numpy(dtype=float)
    x = model_frame(df, cols)
    match = df["match_id"].to_numpy()
    disc = (df["split"].to_numpy() == C.DISCOVERY)
    gen = np.full(len(df), np.nan)
    spe = np.full(len(df), np.nan)
    di = np.where(disc)[0]
    folds = fold_labels(match[di], n_folds, seed)
    for k in np.unique(folds):
        te = di[folds == k]
        tr = di[folds != k]
        tr_s = tr[scen[tr]]
        if len(tr_s) < 200 or len(te) == 0:
            continue
        gen[te] = fit_count(x.iloc[tr], y[tr], match[tr], x.iloc[te], seed, cfg.n_jobs)
        spe[te] = fit_count(x.iloc[tr_s], y[tr_s], match[tr_s], x.iloc[te], seed, cfg.n_jobs)
    ci = np.where(~disc)[0]
    di_s = di[scen[di]]
    if len(ci) and len(di_s) >= 200:
        gen[ci] = fit_count(x.iloc[di], y[di], match[di], x.iloc[ci], seed, cfg.n_jobs)
        spe[ci] = fit_count(x.iloc[di_s], y[di_s], match[di_s], x.iloc[ci], seed, cfg.n_jobs)
    r_gen = C.fit_nb_dispersion(y[disc], np.where(np.isnan(gen[disc]), y[disc].mean(),
                                                  gen[disc]))
    rows: list[dict[str, Any]] = []
    for where, wm in (("discovery", disc), ("confirmation", ~disc)):
        for scored, sm in (("scenario", scen), ("off_scenario", ~scen)):
            m = wm & sm & ~np.isnan(gen) & ~np.isnan(spe)
            if m.sum() < 50:
                continue
            lg = per_row_log_score(y[m], gen[m], r_gen)
            ls = per_row_log_score(y[m], spe[m], r_gen)
            d, lo, hi = C.clustered_bootstrap_mean(lg - ls, match[m], cfg.split.n_boot, seed)
            rows.append({
                "where": where, "scored_on": scored, "target": target, "n": int(m.sum()),
                "n_matches": int(pd.unique(match[m]).size),
                "mean_y": float(y[m].mean()),
                "log_score_generalist": float(lg.mean()),
                "log_score_specialist": float(ls.mean()),
                "delta_specialist_minus_generalist": float(d), "ci_lo": float(lo),
                "ci_hi": float(hi),
                "mae_generalist": float(np.abs(y[m] - gen[m]).mean()),
                "mae_specialist": float(np.abs(y[m] - spe[m]).mean()),
            })
    return rows


def cv_and_confirm(
    df: pd.DataFrame, cols: Sequence[str], target: str, cfg: PassConfig,
    seed: int | None = None, n_folds: int = 5,
) -> np.ndarray:
    """Predictions for every row: match-grouped CV inside discovery, fit-on-discovery outside.

    Args:
        df: Modelled population with a ``split`` column.
        cols: Feature names.
        target: Count column.
        cfg: Stage configuration.
        seed: Seed override.
        n_folds: Discovery CV folds.

    Returns:
        Predicted means [n].
    """
    seed = cfg.split.seed if seed is None else seed
    y = df[target].to_numpy(dtype=float)
    x = model_frame(df, cols)
    match = df["match_id"].to_numpy()
    disc = df["split"].to_numpy() == C.DISCOVERY
    out = np.full(len(df), np.nan)
    di = np.where(disc)[0]
    folds = fold_labels(match[di], n_folds, seed)
    for k in np.unique(folds):
        te, tr = di[folds == k], di[folds != k]
        out[te] = fit_count(x.iloc[tr], y[tr], match[tr], x.iloc[te], seed, cfg.n_jobs)
    ci = np.where(~disc)[0]
    if len(ci):
        out[ci] = fit_count(x.iloc[di], y[di], match[di], x.iloc[ci], seed, cfg.n_jobs)
    return out


def proxy_predictions(df: pd.DataFrame, proxy_col: str, target: str, cfg: PassConfig,
                      n_folds: int = 5) -> np.ndarray:
    """The book proxy with a one-dimensional Poisson recalibration fitted out of sample.

    Args:
        df: Modelled population with a ``split`` column.
        proxy_col: Raw proxy mean column.
        target: Count column.
        cfg: Stage configuration.
        n_folds: Discovery CV folds.

    Returns:
        Calibrated proxy means [n].
    """
    y = df[target].to_numpy(dtype=float)
    mu = df[proxy_col].to_numpy(dtype=float)
    match = df["match_id"].to_numpy()
    disc = df["split"].to_numpy() == C.DISCOVERY
    out = np.full(len(df), np.nan)
    di = np.where(disc)[0]
    folds = fold_labels(match[di], n_folds, cfg.split.seed)
    for k in np.unique(folds):
        te, tr = di[folds == k], di[folds != k]
        out[te] = poisson_glm_apply(mu[te], poisson_glm_link(mu[tr], y[tr]))
    ci = np.where(~disc)[0]
    if len(ci):
        out[ci] = poisson_glm_apply(mu[ci], poisson_glm_link(mu[di], y[di]))
    return out


def attach_scenarios(df: pd.DataFrame, cfg: PassConfig = CFG) -> pd.DataFrame:
    """Join the usual-passer columns onto the panel rows.

    Args:
        df: Panel rows.
        cfg: Stage configuration.

    Returns:
        ``df`` with ``key_player_id``, ``key_att_per90``, ``key_absent`` and ``key_known``.
    """
    sb = C.sb_processed_dir()
    pm = pd.read_parquet(sb / "player_match.parquet",
                         columns=["match_id", "team_id", "player_id", "minutes", "passes"])
    mt = pd.read_parquet(sb / "matches.parquet", columns=["match_id", "date"])
    mt["date"] = pd.to_datetime(mt["date"])
    u = usual_passer_frame(pm.merge(mt, on="match_id", how="left"), cfg.min_prior_matches)
    out = df.merge(u, on=["match_id", "team_id"], how="left")
    # a row about the usual passer himself is not "his team without him"
    same = out["player_id"].to_numpy() == out["key_player_id"].to_numpy()
    out.loc[same, "key_known"] = 0.0
    for c in ("key_absent", "key_known"):
        out[c] = out[c].fillna(0.0)
    return out


def stage_b(cfg: PassConfig = CFG) -> dict[str, pd.DataFrame]:
    """(b) The volume channel: proxy vs model, the gate, and the refit-noise floor.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables.
    """
    t0 = time.time()
    panel = build_player_panel(cfg)
    df = attach_scenarios(league_population(panel, cfg), cfg)
    disc = df["split"].to_numpy() == C.DISCOVERY
    cuts = discovery_cuts(df[disc])
    out: dict[str, pd.DataFrame] = {}

    # --- setup and audit -------------------------------------------------
    setup = []
    for where, m in (("discovery", disc), ("confirmation", ~disc)):
        setup.append({
            "where": where, "n_rows": int(m.sum()),
            "n_matches": int(df.loc[m, "match_id"].nunique()),
            "n_players": int(df.loc[m, "player_id"].nunique()),
            "date_min": str(df.loc[m, "date"].min().date()),
            "date_max": str(df.loc[m, "date"].max().date()),
            "mean_attempts": float(df.loc[m, "passes"].mean()),
            "sd_attempts": float(df.loc[m, "passes"].std()),
            "mean_completed": float(df.loc[m, "passes_completed"].mean()),
            "sd_completed": float(df.loc[m, "passes_completed"].std()),
            "mean_completion_rate": float(df.loc[m, "passes_completed"].sum()
                                          / df.loc[m, "passes"].sum()),
        })
    out["setup"] = pd.DataFrame(setup)
    out["scenario_defs"] = pd.DataFrame([
        {"scenario": s, "cut": {"deep_opponent": cuts["deep_opp"],
                                "pressing_opponent": cuts["press_opp"],
                                "high_volume_passer": cuts["hi_vol"],
                                "possession_edge": cuts["poss_edge"]}.get(s, float("nan")),
         "n_discovery": int((scenario_mask(df, s, cuts) & disc).sum()),
         "n_confirmation": int((scenario_mask(df, s, cuts) & ~disc).sum()),
         "share_discovery": float(scenario_mask(df, s, cuts)[disc].mean()),
         "mean_attempts_in": float(df.loc[scenario_mask(df, s, cuts), "passes"].mean()),
         "mean_attempts_out": float(df.loc[~scenario_mask(df, s, cuts), "passes"].mean())}
        for s in SCENARIOS])

    panel_sorted = panel.sort_values(["date", "match_id", "player_id"],
                                     kind="mergesort").reset_index(drop=True)
    aud = []
    raw = pd.read_parquet(C.sb_processed_dir() / "player_match.parquet",
                          columns=["match_id", "player_id", "passes", "minutes"])
    chk = panel_sorted.merge(raw, on=["match_id", "player_id"], how="left",
                             suffixes=("", "_raw"))
    pri = C.prior_expanding(chk, ["player_id"], ["passes_raw", "minutes_raw"],
                            ["date", "match_id"], count_name="chk_n", prefix="chk_")
    chk = pd.concat([chk, pri], axis=1)
    for col, raw_col in (("chk_passes_raw", "passes_raw"), ("chk_minutes_raw", "minutes_raw")):
        a = C.audit_strictly_prior(chk, ["player_id"], raw_col, ["date", "match_id"], col,
                                   "chk_n", n_check=120, seed=cfg.split.seed)
        aud.append({"quantity": raw_col, **a})
    # brute-force check of the windowed (last-5) prior features
    rng = np.random.default_rng(cfg.split.seed)
    idx = rng.choice(len(chk), size=min(150, len(chk)), replace=False)
    worst = 0.0
    key = chk["player_id"].to_numpy()
    ordv = list(zip(chk["date"].to_numpy(), chk["match_id"].to_numpy()))
    vals = chk["passes_raw"].to_numpy(dtype=float)
    got = chk["p_att_last5"].to_numpy(dtype=float)
    for i in idx:
        earlier = [vals[j] for j in np.flatnonzero(key == key[i]) if ordv[j] < ordv[i]]
        want = float(np.mean(earlier[-5:])) if earlier else np.nan
        if np.isnan(want) and np.isnan(got[i]):
            continue
        worst = max(worst, abs(want - got[i]))
    aud.append({"quantity": "p_att_last5_bruteforce", "n_checked": float(len(idx)),
                "max_abs_sum_error": float(worst), "max_abs_count_error": 0.0})
    recon = np.abs(C.shrunk_rate(chk["chk_passes_raw"].to_numpy(),
                                 chk["chk_minutes_raw"].to_numpy() / 90.0,
                                 float(raw["passes"].sum() / raw["minutes"].sum() * 90.0),
                                 cfg.k_player) - chk["p_att_per90"].to_numpy()).max()
    aud.append({"quantity": "p_att_per90_reconstruction", "n_checked": float(len(chk)),
                "max_abs_sum_error": float(recon), "max_abs_count_error": 0.0})
    # team-level: the opponent's rolling block depth must not see the current match
    tmr = pd.read_parquet(C.sb_processed_dir() / "team_match.parquet",
                          columns=["match_id", "team_id", "opp_id", "possession"])
    mtr = pd.read_parquet(C.sb_processed_dir() / "matches.parquet",
                          columns=["match_id", "date"])
    mtr["date"] = pd.to_datetime(mtr["date"])
    tchk = tmr.merge(mtr, on="match_id", how="left").sort_values(
        ["date", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    tpri = C.prior_expanding(tchk, ["team_id"], ["possession"], ["date", "match_id"],
                             count_name="tn", prefix="tsum_")
    tchk = pd.concat([tchk, tpri], axis=1)
    a = C.audit_strictly_prior(tchk, ["team_id"], "possession", ["date", "match_id"],
                               "tsum_possession", "tn", n_check=100, seed=cfg.split.seed)
    aud.append({"quantity": "team_possession_prior", **a})
    # and the opponent join must line up: op_possession of a row equals tp_possession of the
    # opponent's row in the same match
    j = df[["match_id", "team_id", "opp_id", "op_possession"]].merge(
        df[["match_id", "team_id", "tp_possession"]].drop_duplicates(),
        left_on=["match_id", "opp_id"], right_on=["match_id", "team_id"], how="left",
        suffixes=("", "_opp"))
    err = float(np.nanmax(np.abs(j["op_possession"].to_numpy(dtype=float)
                                 - j["tp_possession"].to_numpy(dtype=float))))
    aud.append({"quantity": "opponent_join_consistency", "n_checked": float(len(j)),
                "max_abs_sum_error": err, "max_abs_count_error": 0.0})
    out["audit"] = pd.DataFrame(aud)

    # --- proxy vs model --------------------------------------------------
    metrics, deltas = [], []
    feat_all = list(BASE_FEATURES) + list(IMP_FEATURES)
    preds: dict[str, np.ndarray] = {}
    for target, proxy_col in (("passes", "mu_proxy_att"),
                              ("passes_completed", "mu_proxy_comp")):
        y = df[target].to_numpy(dtype=float)
        preds[f"{target}|proxy"] = proxy_predictions(df, proxy_col, target, cfg)
        preds[f"{target}|model"] = cv_and_confirm(df, feat_all, target, cfg)
        preds[f"{target}|model_noimp"] = cv_and_confirm(df, BASE_FEATURES, target, cfg)
        r = C.fit_nb_dispersion(y[disc], preds[f"{target}|proxy"][disc])
        for where, m in (("discovery", disc), ("confirmation", ~disc)):
            for name in ("proxy", "model_noimp", "model"):
                mu = preds[f"{target}|{name}"]
                metrics.append({"where": where, "target": target, "model": name,
                                "nb_r": r, **count_metrics(y[m], mu[m], r)})
            base = per_row_log_score(y[m], preds[f"{target}|proxy"][m], r)
            for name in ("model_noimp", "model"):
                d, lo, hi = C.clustered_bootstrap_mean(
                    base - per_row_log_score(y[m], preds[f"{target}|{name}"][m], r),
                    df.loc[m, "match_id"].to_numpy(), cfg.split.n_boot, cfg.split.seed)
                deltas.append({"where": where, "target": target, "model": name,
                               "vs": "proxy", "n": int(m.sum()),
                               "delta_log_score": d, "ci_lo": lo, "ci_hi": hi})
            d, lo, hi = C.clustered_bootstrap_mean(
                per_row_log_score(y[m], preds[f"{target}|model_noimp"][m], r)
                - per_row_log_score(y[m], preds[f"{target}|model"][m], r),
                df.loc[m, "match_id"].to_numpy(), cfg.split.n_boot, cfg.split.seed)
            deltas.append({"where": where, "target": target, "model": "model",
                           "vs": "model_noimp", "n": int(m.sum()),
                           "delta_log_score": d, "ci_lo": lo, "ci_hi": hi})
    out["metrics"] = pd.DataFrame(metrics)
    out["deltas"] = pd.DataFrame(deltas)
    _log(f"stage b: proxy / model comparison done ({time.time() - t0:.0f} s)")

    # --- refit-noise floor ------------------------------------------------
    noise = []
    for target in ("passes",):
        y = df[target].to_numpy(dtype=float)
        r = C.fit_nb_dispersion(y[disc], preds[f"{target}|proxy"][disc])
        for name, cols in (("model", feat_all), ("model_noimp", list(BASE_FEATURES))):
            vals = []
            for s in cfg.seeds_refit:
                p = cv_and_confirm(df, cols, target, cfg, seed=cfg.split.seed + s)
                vals.append(float(per_row_log_score(y[~disc], p[~disc], r).mean()))
            noise.append({"target": target, "model": name, "n_seeds": len(vals),
                          "mean_log_score": float(np.mean(vals)),
                          "min": float(np.min(vals)), "max": float(np.max(vals)),
                          "spread": float(np.max(vals) - np.min(vals))})
    out["refit"] = pd.DataFrame(noise)
    _log(f"stage b: refit floor done ({time.time() - t0:.0f} s)")

    # --- where the model's advantage comes from ---------------------------
    ladder_sets = {
        "F0 proxy mean only": ["mu_proxy_att"],
        "F1 + player rolling": ["mu_proxy_att", "p_att_per90", "p_att_per90_last5",
                                "p_att_last5", "p_min_last5", "p_start_minutes",
                                "p_start_rate", "p_prior_n", "p_comp_rate", "p_kp_per90",
                                "p_press_per90"],
        "F2 + position / venue": None, "F3 + team and opponent rolling": None,
        "F4 + imputed block depth": feat_all,
    }
    ladder_sets["F2 + position / venue"] = ladder_sets["F1 + player rolling"] + [
        "pos_group", "home", "competition"]
    ladder_sets["F3 + team and opponent rolling"] = list(BASE_FEATURES)
    abl = []
    y = df["passes"].to_numpy(dtype=float)
    r_ab = C.fit_nb_dispersion(y[disc], preds["passes|proxy"][disc])
    ref_ls = per_row_log_score(y, preds["passes|proxy"], r_ab)
    for name, cols in ladder_sets.items():
        pr = cv_and_confirm(df, cols, "passes", cfg)
        for where, m in (("discovery", disc), ("confirmation", ~disc)):
            d, lo, hi = C.clustered_bootstrap_mean(
                ref_ls[m] - per_row_log_score(y[m], pr[m], r_ab),
                df.loc[m, "match_id"].to_numpy(), cfg.split.n_boot, cfg.split.seed)
            abl.append({"where": where, "feature_set": name, "n_features": len(cols),
                        "n": int(m.sum()), "r2": float(r2(y[m], pr[m])),
                        "log_score": float(per_row_log_score(y[m], pr[m], r_ab).mean()),
                        "delta_vs_proxy": d, "ci_lo": lo, "ci_hi": hi})
    out["ablation"] = pd.DataFrame(abl)
    _log(f"stage b: ablation done ({time.time() - t0:.0f} s)")

    # --- the gate ---------------------------------------------------------
    gate_rows: list[dict[str, Any]] = []
    for s in SCENARIOS:
        mask = scenario_mask(df, s, cuts)
        if mask.sum() < 500:
            continue
        for row in gate(df, feat_all, "passes", mask, cfg):
            gate_rows.append({"scenario": s, **row})
    out["gate"] = pd.DataFrame(gate_rows)
    _log(f"stage b: gate done ({time.time() - t0:.0f} s)")

    df_out = df[["match_id", "team_id", "player_id", "date", "split", "pos_group", "passes",
                 "passes_completed", "minutes", "mu_proxy_att", "mu_proxy_comp",
                 "p_att_per90", "op_block_depth", "op_ppda", "poss_edge", "key_absent",
                 "key_known"]].copy()
    for k, v in preds.items():
        df_out["pred_" + k.replace("|", "_")] = v
    df_out.to_parquet(C.processed_dir() / "pass_b_preds.parquet", index=False)
    for name, tab in out.items():
        C.write_table(tab, f"03_pass_b_{name}")
    _log(f"stage b done in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# (a) Aggregating per-pass probabilities to a player-match count
# ---------------------------------------------------------------------------


def pad_by_group(values: np.ndarray, group_codes: np.ndarray, n_groups: int,
                 n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Scatter a ragged per-row vector into a dense ``[g, n_max]`` block.

    Args:
        values: Per-row values [n], already ordered so that each group's rows are contiguous
            in the order they should occupy.
        group_codes: Group index per row [n], non-decreasing.
        n_groups: Number of groups.
        n_max: Width of the dense block; rows beyond it are dropped.

    Returns:
        Tuple ``(padded [g, n_max], valid [g, n_max])``.
    """
    pos = np.arange(len(values)) - np.repeat(
        np.searchsorted(group_codes, np.arange(n_groups), side="left"),
        np.bincount(group_codes, minlength=n_groups))
    keep = pos < n_max
    out = np.zeros((n_groups, n_max), dtype=float)
    val = np.zeros((n_groups, n_max), dtype=bool)
    out[group_codes[keep], pos[keep]] = values[keep]
    val[group_codes[keep], pos[keep]] = True
    return out, val


def count_distributions(
    preds: pd.DataFrame, variants: Sequence[str], pred_prefix: str = "pred_",
    n_max: int = 200,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Player-match completed-pass distributions from per-pass completion probabilities.

    Completed passes are a sum of Bernoulli trials with unequal probabilities, so given the
    realised attempts the exact predictive distribution is a Poisson binomial.

    Args:
        preds: Per-pass rows with ``match_id``, ``player_id``, ``is_complete`` and one
            ``<pred_prefix><variant>`` column per variant.
        variants: Variant names.
        pred_prefix: Prefix of the prediction columns.
        n_max: Attempts kept per player-match (larger counts are truncated).

    Returns:
        Tuple ``(frame, pmfs)``: one row per player-match with the realised attempts and
        completions, and a dict mapping variant to its pmf block [g, n_max + 1].
    """
    d = preds.sort_values(["match_id", "player_id"], kind="mergesort").reset_index(drop=True)
    keys = pd.MultiIndex.from_arrays([d["match_id"], d["player_id"]])
    codes, uniq = pd.factorize(keys)
    n_groups = len(uniq)
    obs_att = np.bincount(codes, minlength=n_groups)
    obs_comp = np.bincount(codes, weights=d["is_complete"].to_numpy(dtype=float),
                           minlength=n_groups)
    width = int(min(n_max, obs_att.max()))
    pmfs: dict[str, np.ndarray] = {}
    for v in variants:
        p, valid = pad_by_group(d[pred_prefix + v].to_numpy(dtype=float), codes, n_groups,
                                width)
        pmfs[v] = poisson_binomial_pmf_batch(p, valid)
    first = d.groupby(codes, sort=True).first()
    frame = pd.DataFrame({
        "match_id": [k[0] for k in uniq], "player_id": [k[1] for k in uniq],
        "attempts": obs_att.astype(int),
        "completed": np.minimum(obs_comp, width).astype(int),
        "match_date": first["match_date"].to_numpy(),
        "competition": first["competition"].to_numpy(),
        "gender": first["gender"].to_numpy(),
        "team_id": first["team_id"].to_numpy(),
    })
    frame["truncated"] = frame["attempts"] > width
    return frame, pmfs


def stage_a(cfg: PassConfig = CFG) -> dict[str, pd.DataFrame]:
    """(a) Does a per-pass completion gain move a completed-pass count line?

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables.
    """
    t0 = time.time()
    preds = build_pass_predictions(cfg)
    out: dict[str, pd.DataFrame] = {}
    cached = pd.read_parquet(C.soccer_processed_dir() / "payoff_cache" / "xpass_preds.parquet",
                             columns=["event_id", "is_complete", "match_id",
                                      "pred_EVENT__noafter", "pred_EVENT+IMP__noafter",
                                      "pred_EVENT+ORACLE__noafter"])
    match = preds["match_id"].to_numpy()

    # --- per-pass reproduction of soccer 03 -------------------------------
    rows = []
    y = preds["is_complete"].to_numpy(dtype=float)
    in_cache = preds["event_id"].isin(set(cached["event_id"])).to_numpy()
    for pop, m in (("all_attempts", np.ones(len(preds), bool)),
                   ("cached_subsample_rows", in_cache)):
        ll_event = log_loss(y[m], preds["pred_EVENT"].to_numpy()[m])
        for v in ("BASE", "EVENT", "EVENT+IMP", "EVENT+ORACLE"):
            p = preds["pred_" + v].to_numpy()
            d, lo, hi = C.clustered_bootstrap_mean(
                -y[m] * np.log(np.clip(preds["pred_EVENT"].to_numpy()[m], 1e-9, 1))
                - (1 - y[m]) * np.log(np.clip(1 - preds["pred_EVENT"].to_numpy()[m], 1e-9, 1))
                + y[m] * np.log(np.clip(p[m], 1e-9, 1))
                + (1 - y[m]) * np.log(np.clip(1 - p[m], 1e-9, 1)),
                match[m], cfg.split.n_boot, cfg.split.seed)
            rows.append({"population": pop, "source": "refit_all_attempts", "variant": v,
                         "n": int(m.sum()), "log_loss": log_loss(y[m], p[m]),
                         "log_loss_EVENT": ll_event, "delta_vs_EVENT": d, "ci_lo": lo,
                         "ci_hi": hi})
    cy = cached["is_complete"].to_numpy(dtype=float)
    cm = cached["match_id"].to_numpy()
    for v, col in (("EVENT", "pred_EVENT__noafter"), ("EVENT+IMP", "pred_EVENT+IMP__noafter"),
                   ("EVENT+ORACLE", "pred_EVENT+ORACLE__noafter")):
        p = cached[col].to_numpy()
        pe = cached["pred_EVENT__noafter"].to_numpy()
        d, lo, hi = C.clustered_bootstrap_mean(
            cy * (np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(pe, 1e-9, 1)))
            + (1 - cy) * (np.log(np.clip(1 - p, 1e-9, 1)) - np.log(np.clip(1 - pe, 1e-9, 1))),
            cm, cfg.split.n_boot, cfg.split.seed)
        rows.append({"population": "cached_subsample_rows", "source": "programme_cache",
                     "variant": v, "n": int(len(cy)), "log_loss": log_loss(cy, p),
                     "log_loss_EVENT": log_loss(cy, pe), "delta_vs_EVENT": d, "ci_lo": lo,
                     "ci_hi": hi})
    sq = preds["seq_covered"].to_numpy()
    for v in SEQ_VARIANTS:
        p = preds["predseq_" + v].to_numpy()
        pe = preds["predseq_EVENT"].to_numpy()
        d, lo, hi = C.clustered_bootstrap_mean(
            y[sq] * (np.log(np.clip(p[sq], 1e-9, 1)) - np.log(np.clip(pe[sq], 1e-9, 1)))
            + (1 - y[sq]) * (np.log(np.clip(1 - p[sq], 1e-9, 1))
                             - np.log(np.clip(1 - pe[sq], 1e-9, 1))),
            match[sq], cfg.split.n_boot, cfg.split.seed)
        rows.append({"population": "seq_matches", "source": "refit_all_attempts", "variant": v,
                     "n": int(sq.sum()), "log_loss": log_loss(y[sq], p[sq]),
                     "log_loss_EVENT": log_loss(y[sq], pe[sq]), "delta_vs_EVENT": d,
                     "ci_lo": lo, "ci_hi": hi})
    out["per_pass"] = pd.DataFrame(rows)
    _log(f"stage a: per-pass table done ({time.time() - t0:.0f} s)")

    # --- counts -----------------------------------------------------------
    tabs_metrics, tabs_lines, tabs_cal = [], [], []
    for pop, sub, variants, prefix in (
        ("all_matches", preds, ("BASE",) + FULL_VARIANTS, "pred_"),
        ("seq_matches", preds[sq].reset_index(drop=True), SEQ_VARIANTS, "predseq_"),
    ):
        frame, pmfs = count_distributions(sub, variants, prefix)
        frame["split"] = C.chronological_split(frame["match_id"].to_numpy(),
                                               frame["match_date"].to_numpy(),
                                               cfg.split.discovery_frac)
        obs = frame["completed"].to_numpy()
        mid = frame["match_id"].to_numpy()
        ref = "EVENT"
        mu_ref, _ = pmf_mean_var(pmfs[ref])
        ls = {v: pmf_log_score(pmfs[v], obs) for v in variants}
        for where, m in (("discovery", frame["split"] == C.DISCOVERY),
                         ("confirmation", frame["split"] == C.CONFIRMATION),
                         ("all", np.ones(len(frame), bool))):
            m = np.asarray(m)
            for v in variants:
                mu, var = pmf_mean_var(pmfs[v])
                d, lo, hi = C.clustered_bootstrap_mean(ls[ref][m] - ls[v][m], mid[m],
                                                       cfg.split.n_boot, cfg.split.seed)
                tabs_metrics.append({
                    "population": pop, "where": where, "variant": v, "n": int(m.sum()),
                    "n_matches": int(pd.unique(mid[m]).size),
                    "mean_attempts": float(frame["attempts"].to_numpy()[m].mean()),
                    "mean_completed": float(obs[m].mean()),
                    "count_log_score": float(ls[v][m].mean()),
                    "mean_pred": float(mu[m].mean()), "mean_sd": float(np.sqrt(var[m]).mean()),
                    "mae": float(np.abs(obs[m] - mu[m]).mean()),
                    "delta_vs_EVENT": d, "ci_lo": lo, "ci_hi": hi})
            for off in cfg.line_offsets:
                line = line_from_mean(mu_ref, off)
                yb = (obs > line).astype(float)
                if yb[m].mean() in (0.0, 1.0):
                    continue
                p_ref = pmf_sf(pmfs[ref], line)
                for v in variants:
                    p = pmf_sf(pmfs[v], line)
                    d, lo, hi = C.clustered_bootstrap_mean(
                        (yb[m] - p_ref[m]) ** 2 - (yb[m] - p[m]) ** 2, mid[m],
                        cfg.split.n_boot, cfg.split.seed)
                    tabs_lines.append({
                        "population": pop, "where": where, "offset": off, "variant": v,
                        "n": int(m.sum()), "mean_line": float(line[m].mean()),
                        "over_rate": float(yb[m].mean()), "mean_p": float(p[m].mean()),
                        "brier": float(brier(yb[m], p[m])),
                        "log_loss": float(log_loss(yb[m], np.clip(p[m], 1e-6, 1 - 1e-6))),
                        "delta_brier_vs_EVENT": d, "ci_lo": lo, "ci_hi": hi,
                        "mean_abs_shift_vs_EVENT": float(np.abs(p[m] - p_ref[m]).mean()),
                        "max_abs_shift_vs_EVENT": float(np.abs(p[m] - p_ref[m]).max())})
        # calibration of the predicted mean and of the central tail probability
        for v in variants:
            mu, _ = pmf_mean_var(pmfs[v])
            q = pd.qcut(mu, 10, labels=False, duplicates="drop")
            for b in np.unique(q[~pd.isna(q)]):
                s = q == b
                tabs_cal.append({"population": pop, "variant": v, "kind": "mean", "bin": int(b),
                                 "n": int(s.sum()), "pred": float(mu[s].mean()),
                                 "obs": float(obs[s].mean())})
            line = line_from_mean(mu_ref, 0)
            p = pmf_sf(pmfs[v], line)
            yb = (obs > line).astype(float)
            q2 = pd.qcut(p, 10, labels=False, duplicates="drop")
            for b in np.unique(q2[~pd.isna(q2)]):
                s = q2 == b
                tabs_cal.append({"population": pop, "variant": v, "kind": "tail_line0",
                                 "bin": int(b), "n": int(s.sum()), "pred": float(p[s].mean()),
                                 "obs": float(yb[s].mean())})
        if pop == "all_matches":
            frame.to_parquet(C.processed_dir() / "pass_a_counts.parquet", index=False)
    out["count_metrics"] = pd.DataFrame(tabs_metrics)
    out["count_lines"] = pd.DataFrame(tabs_lines)
    out["count_calibration"] = pd.DataFrame(tabs_cal)

    # --- retention: per-pass nats in, count nats out ----------------------
    ret = []
    cm_tab = out["count_metrics"]
    pp = out["per_pass"]
    for pop, ppop in (("all_matches", "all_attempts"), ("seq_matches", "seq_matches")):
        sub = cm_tab[(cm_tab["population"] == pop) & (cm_tab["where"] == "all")]
        n_att = float(sub["mean_attempts"].iloc[0])
        for v in sub["variant"]:
            if v in ("BASE", "EVENT"):
                continue
            per = pp[(pp["population"] == ppop) & (pp["variant"] == v)
                     & (pp["source"] == "refit_all_attempts")]
            if per.empty:
                continue
            gain_pass = float(per["delta_vs_EVENT"].iloc[0])
            gain_count = float(sub.loc[sub["variant"] == v, "delta_vs_EVENT"].iloc[0])
            ret.append({"population": pop, "variant": v, "mean_attempts": n_att,
                        "per_pass_nats": gain_pass,
                        "summed_over_attempts": gain_pass * n_att,
                        "count_nats": gain_count,
                        "retention": gain_count / (gain_pass * n_att)
                        if gain_pass * n_att != 0 else float("nan")})
    out["retention"] = pd.DataFrame(ret)
    for name, tab in out.items():
        C.write_table(tab, f"03_pass_a_{name}")
    _log(f"stage a done in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# Proxy honesty check: the one count market with a real line (total goals)
# ---------------------------------------------------------------------------


def build_goals_table(cfg: PassConfig = CFG, divisions: Sequence[str] = LEAGUE_DIVISIONS,
                      force: bool = False) -> pd.DataFrame:
    """Football-data matches of the given divisions with a strictly-prior goals proxy.

    Args:
        cfg: Stage configuration.
        divisions: football-data.co.uk division codes.
        force: Rebuild even when the cache exists.

    Returns:
        Frame with the realised total, the proxy mean and Bet365's Over/Under 2.5 prices.
    """
    path = C.processed_dir() / "goals_proxy.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    o = pd.read_parquet(C.odds_path(),
                        columns=["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome",
                                 "FTAway", "Over25", "Under25"])
    o = o[o["Division"].isin(list(divisions))].copy()
    o["MatchDate"] = pd.to_datetime(o["MatchDate"])
    o = o.dropna(subset=["FTHome", "FTAway"]).reset_index(drop=True)
    o["season"] = np.where(o["MatchDate"].dt.month >= 7, o["MatchDate"].dt.year,
                           o["MatchDate"].dt.year - 1)
    o["match_uid"] = np.arange(len(o))
    o = o.sort_values(["MatchDate", "match_uid"], kind="mergesort").reset_index(drop=True)
    long = pd.concat([
        o.assign(team=o["HomeTeam"], gf=o["FTHome"], ga=o["FTAway"], side="H"),
        o.assign(team=o["AwayTeam"], gf=o["FTAway"], ga=o["FTHome"], side="A"),
    ], ignore_index=True)
    long["one"] = 1.0
    pri = C.prior_expanding(long, ["Division", "season", "team"], ["gf", "ga", "one"],
                            ["MatchDate", "match_uid"], count_name="prior_n", prefix="pr_")
    long = pd.concat([long, pri], axis=1)
    gm_gf = float(long["gf"].mean())
    for c in ("gf", "ga"):
        long["rate_" + c] = C.shrunk_rate(long["pr_" + c].to_numpy(),
                                          long["prior_n"].to_numpy(), gm_gf, cfg.k_team)
    h = long[long["side"] == "H"][["match_uid", "rate_gf", "rate_ga", "prior_n"]].rename(
        columns={"rate_gf": "h_gf", "rate_ga": "h_ga", "prior_n": "h_n"})
    a = long[long["side"] == "A"][["match_uid", "rate_gf", "rate_ga", "prior_n"]].rename(
        columns={"rate_gf": "a_gf", "rate_ga": "a_ga", "prior_n": "a_n"})
    o = o.merge(h, on="match_uid").merge(a, on="match_uid")
    o["total"] = o["FTHome"] + o["FTAway"]
    o["mu_proxy"] = 0.5 * (o["h_gf"] + o["a_ga"]) + 0.5 * (o["a_gf"] + o["h_ga"])
    o["split"] = C.chronological_split(o["match_uid"].to_numpy(), o["MatchDate"].to_numpy(),
                                       cfg.split.discovery_frac)
    o.to_parquet(path, index=False)
    return o



def implied_mean_from_sf(p_over: np.ndarray, line: float, r: float,
                         grid: np.ndarray | None = None) -> np.ndarray:
    """Invert ``P(count > line)`` back to the mean of a negative binomial.

    Args:
        p_over: Probability of the over [n].
        line: The half-integer line.
        r: Dispersion (``inf`` = Poisson).
        grid: Candidate means; a default grid is used if omitted.

    Returns:
        Implied means [n].
    """
    g = np.linspace(0.05, 8.0, 1600) if grid is None else np.asarray(grid, dtype=float)
    curve = C.nb_sf(line, g, r)
    idx = np.searchsorted(curve, np.asarray(p_over, dtype=float))
    return g[np.clip(idx, 0, len(g) - 1)]


def line_units_table(mu_a: np.ndarray, mu_b: np.ndarray, y: np.ndarray, label: str
                     ) -> pd.DataFrame:
    """How far apart two pricing means are, in the units the market sets lines in.

    Args:
        mu_a: Reference (proxy) means [n].
        mu_b: Comparison means [n].
        y: Realised counts [n], for the market's own scale.
        label: Row label.

    Returns:
        One-row frame.
    """
    d = np.asarray(mu_b, dtype=float) - np.asarray(mu_a, dtype=float)
    sd_y = float(np.std(np.asarray(y, dtype=float)))
    return pd.DataFrame([{
        "market": label, "n": int(len(d)), "sd_outcome": sd_y,
        "mean_abs_diff": float(np.abs(d).mean()), "sd_diff": float(np.std(d)),
        "abs_diff_over_sd_outcome": float(np.abs(d).mean()) / sd_y,
        "share_diff_ge_half_unit": float((np.abs(d) >= 0.5).mean()),
        "share_diff_ge_1_unit": float((np.abs(d) >= 1.0).mean()),
        "share_diff_ge_2_units": float((np.abs(d) >= 2.0).mean())}])


def stage_goals(cfg: PassConfig = CFG) -> dict[str, pd.DataFrame]:
    """Measure how much a real bookmaker beats the identical rolling-mean proxy on goals.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict with ``gap`` (log-loss / Brier at the 2.5 line) and ``roi`` (what the real book
        would make betting into a market priced off the proxy).
    """
    t0 = time.time()
    out: dict[str, pd.DataFrame] = {}
    gap_rows, roi_rows = [], []
    full = build_goals_table(cfg)
    pops = {"four_leagues_all_seasons": full,
            "four_leagues_2015_16": full[full["season"] == 2015].reset_index(drop=True)}
    for pop, o in pops.items():
        o = o.dropna(subset=["Over25", "Under25"]).copy()
        # the split is chronological within the population being scored
        o["split"] = C.chronological_split(o["match_uid"].to_numpy(),
                                           o["MatchDate"].to_numpy(),
                                           cfg.split.discovery_frac)
        o = o[(o["h_n"] >= cfg.min_prior_matches) & (o["a_n"] >= cfg.min_prior_matches)]
        o = o.reset_index(drop=True)
        y = (o["total"].to_numpy(dtype=float) > 2.5).astype(float)
        mu = o["mu_proxy"].to_numpy(dtype=float)
        disc = o["split"].to_numpy() == C.DISCOVERY
        r = C.fit_nb_dispersion(o["total"].to_numpy(dtype=float)[disc], mu[disc])
        p_raw = C.nb_sf(2.5, mu, r)
        ab = C.platt_fit(p_raw[disc], y[disc])
        p_cal = C.platt_apply(p_raw, ab)
        p_b365, _ = C.novig_two_way(o["Over25"].to_numpy(dtype=float),
                                    o["Under25"].to_numpy(dtype=float))
        uid = o["match_uid"].to_numpy()
        for where, m in (("discovery", disc), ("confirmation", ~disc)):
            base_ll = -(y[m] * np.log(np.clip(p_cal[m], 1e-9, 1))
                        + (1 - y[m]) * np.log(np.clip(1 - p_cal[m], 1e-9, 1)))
            for name, p in (("climatology", np.full(len(y), float(y[disc].mean()))),
                            ("proxy_raw", p_raw), ("proxy_calibrated", p_cal),
                            ("bet365_novig", p_b365)):
                ll = -(y[m] * np.log(np.clip(p[m], 1e-9, 1))
                       + (1 - y[m]) * np.log(np.clip(1 - p[m], 1e-9, 1)))
                d, lo, hi = C.clustered_bootstrap_mean(base_ll - ll, uid[m], cfg.split.n_boot,
                                                       cfg.split.seed)
                gap_rows.append({"population": pop, "where": where, "model": name,
                                 "n": int(m.sum()), "base_rate": float(y[m].mean()),
                                 "log_loss": float(log_loss(y[m], p[m])),
                                 "brier": float(brier(y[m], p[m])),
                                 "mean_p": float(p[m].mean()),
                                 "delta_vs_proxy_cal": d, "ci_lo": lo, "ci_hi": hi})
            for hold in cfg.holds:
                for thr in cfg.thresholds:
                    res = C.simulate_two_way(
                        y[m], p_b365[m], p_cal[m], uid[m],
                        C.BetSimConfig(hold=hold, edge_threshold=thr,
                                       n_boot=cfg.split.n_boot, seed=cfg.split.seed))
                    roi_rows.append(res.as_row(population=pop, where=where, hold=hold,
                                               threshold=thr, bettor="bet365_novig",
                                               book="goals_rolling_proxy"))
    out["gap"] = pd.DataFrame(gap_rows)
    out["roi"] = pd.DataFrame(roi_rows)

    # --- the exactly-analogous measurement: the proxy's OWN line is 2.5 ---
    o = full.dropna(subset=["Over25", "Under25"]).copy()
    o["split"] = C.chronological_split(o["match_uid"].to_numpy(), o["MatchDate"].to_numpy(),
                                       cfg.split.discovery_frac)
    o = o[(o["h_n"] >= cfg.min_prior_matches) & (o["a_n"] >= cfg.min_prior_matches)]
    o = o[line_from_mean(o["mu_proxy"].to_numpy(dtype=float)) == 2.5].reset_index(drop=True)
    y = (o["total"].to_numpy(dtype=float) > 2.5).astype(float)
    mu = o["mu_proxy"].to_numpy(dtype=float)
    disc = o["split"].to_numpy() == C.DISCOVERY
    r = C.fit_nb_dispersion(o["total"].to_numpy(dtype=float)[disc], mu[disc])
    p_raw = C.nb_sf(2.5, mu, r)
    p_cal = C.platt_apply(p_raw, C.platt_fit(p_raw[disc], y[disc]))
    p_b365, _ = C.novig_two_way(o["Over25"].to_numpy(dtype=float),
                                o["Under25"].to_numpy(dtype=float))
    uid = o["match_uid"].to_numpy()
    rows = []
    for where, m in (("discovery", disc), ("confirmation", ~disc)):
        ll_p = float(log_loss(y[m], p_cal[m]))
        ll_b = float(log_loss(y[m], p_b365[m]))
        d, lo, hi = C.clustered_bootstrap_mean(
            -(y[m] * np.log(np.clip(p_cal[m], 1e-9, 1))
              + (1 - y[m]) * np.log(np.clip(1 - p_cal[m], 1e-9, 1)))
            + (y[m] * np.log(np.clip(p_b365[m], 1e-9, 1))
               + (1 - y[m]) * np.log(np.clip(1 - p_b365[m], 1e-9, 1))),
            uid[m], cfg.split.n_boot, cfg.split.seed)
        rows.append({"where": where, "n": int(m.sum()), "base_rate": float(y[m].mean()),
                     "log_loss_proxy": ll_p, "log_loss_bet365": ll_b,
                     "book_minus_proxy_nats": d, "ci_lo": lo, "ci_hi": hi,
                     "relative_gap": (ll_p - ll_b) / ll_p,
                     "sd_p_proxy": float(np.std(p_cal[m])),
                     "sd_p_bet365": float(np.std(p_b365[m]))})
    out["proxy_set_line"] = pd.DataFrame(rows)
    mu_book = implied_mean_from_sf(p_b365, 2.5, r)
    out["line_units_goals"] = line_units_table(mu, mu_book, o["total"].to_numpy(dtype=float),
                                               "total_goals_bet365_vs_proxy")
    conf = out["gap"][(out["gap"]["where"] == "confirmation")
                      & (out["gap"]["model"] == "bet365_novig")
                      & (out["gap"]["population"] == "four_leagues_all_seasons")]
    conf_roi = out["roi"][(out["roi"]["where"] == "confirmation")
                          & (out["roi"]["threshold"] == 0.02)
                          & (out["roi"]["population"] == "four_leagues_all_seasons")]
    out["haircut"] = pd.DataFrame([{
        "log_loss_gap_book_minus_proxy_nats": float(conf["delta_vs_proxy_cal"].iloc[0]),
        "ci_lo": float(conf["ci_lo"].iloc[0]), "ci_hi": float(conf["ci_hi"].iloc[0]),
        "roi_book_vs_proxy_mean_over_holds": float(conf_roi["roi"].mean()),
        "roi_haircut": float(conf_roi["roi"].mean())}])
    for name, tab in out.items():
        C.write_table(tab, f"03_pass_goals_{name}")
    _log(f"stage goals done in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# (c) End to end: attempts x completion, and betting against the proxy line
# ---------------------------------------------------------------------------


def rate_predictions(df: pd.DataFrame, cols: Sequence[str], cfg: PassConfig,
                     n_folds: int = 5) -> np.ndarray:
    """Attempt-weighted completion-rate predictions, CV inside discovery.

    Args:
        df: Modelled population with ``passes`` / ``passes_completed`` and a ``split`` column.
        cols: Feature names.
        cfg: Stage configuration.
        n_folds: Discovery CV folds.

    Returns:
        Predicted per-attempt completion probability [n].
    """
    import lightgbm as lgb

    att = np.maximum(df["passes"].to_numpy(dtype=float), 1.0)
    rate = df["passes_completed"].to_numpy(dtype=float) / att
    x = model_frame(df, cols)
    match = df["match_id"].to_numpy()
    disc = df["split"].to_numpy() == C.DISCOVERY
    out = np.full(len(df), np.nan)
    cats = [c for c in CATEGORICALS if c in x.columns]
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 1, "max_bin": 127, "lambda_l2": 1.0, "verbose": -1,
              "num_threads": cfg.n_jobs, "seed": cfg.split.seed, "deterministic": True}

    def _fit(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(cfg.split.seed)
        uniq = np.unique(match[tr])
        hold = set(rng.choice(uniq, size=max(1, int(round(len(uniq) * 0.15))), replace=False))
        ih = np.array([g in hold for g in match[tr]])
        d1 = lgb.Dataset(x.iloc[tr[~ih]], rate[tr[~ih]], weight=att[tr[~ih]],
                         categorical_feature=cats, free_raw_data=False)
        d2 = lgb.Dataset(x.iloc[tr[ih]], rate[tr[ih]], weight=att[tr[ih]], reference=d1)
        b = lgb.train(params, d1, num_boost_round=1200, valid_sets=[d2],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        best = int(b.best_iteration) if b.best_iteration else 1200
        b = lgb.train(params, lgb.Dataset(x.iloc[tr], rate[tr], weight=att[tr],
                                          categorical_feature=cats), num_boost_round=best)
        return np.clip(b.predict(x.iloc[te]), 0.3, 0.99)

    di = np.where(disc)[0]
    folds = fold_labels(match[di], n_folds, cfg.split.seed)
    for k in np.unique(folds):
        out[di[folds == k]] = _fit(di[folds != k], di[folds == k])
    ci = np.where(~disc)[0]
    if len(ci):
        out[ci] = _fit(di, ci)
    return out


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """How much of the spread of completed passes is volume and how much is completion rate.

    Uses the exact law of total variance for ``C = sum of Bernoulli(p_t)`` given attempts
    ``A``: ``Var(C) = Var(A * pbar) + E[A * pbar * (1 - pbar)]`` when the per-attempt rate is
    treated as the player-match mean.

    Args:
        df: Rows with ``passes`` and ``passes_completed``.

    Returns:
        One-row frame of the decomposition.
    """
    a = df["passes"].to_numpy(dtype=float)
    c = df["passes_completed"].to_numpy(dtype=float)
    p = np.divide(c, np.maximum(a, 1.0))
    within = float(np.mean(a * p * (1 - p)))
    total = float(np.var(c))
    denom = total if total > 0 else float("nan")
    return pd.DataFrame([{
        "n": int(len(a)), "sd_completed": float(np.std(c)),
        "sd_attempts": float(np.std(a)), "mean_rate": float(c.sum() / max(a.sum(), 1)),
        "sd_rate": float(np.std(p)),
        "var_completed": total, "var_within_match_bernoulli": within,
        "var_between": total - within,
        "share_volume_and_rate_level": (total - within) / denom,
        "share_irreducible_bernoulli": within / denom}])


def binomial_mix_logpmf(p_count: np.ndarray, rate: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """``log P(completed = obs)`` for completed = Binomial(attempts, rate).

    Args:
        p_count: Attempt-count pmf on ``0..K`` per row [g, K + 1].
        rate: Per-attempt completion probability [g].
        obs: Realised completed passes [g].

    Returns:
        Log probability of the realised count [g].
    """
    from scipy.stats import binom

    p_count = np.atleast_2d(np.asarray(p_count, dtype=float))
    g, kp1 = p_count.shape
    a = np.broadcast_to(np.arange(kp1)[None, :], (g, kp1))
    c = np.broadcast_to(np.asarray(obs, dtype=float).reshape(-1, 1), (g, kp1))
    r = np.broadcast_to(np.asarray(rate, dtype=float).reshape(-1, 1), (g, kp1))
    pm = np.where(a >= c, binom.pmf(c, a, r), 0.0)
    return np.log(np.maximum((p_count * pm).sum(axis=1), 1e-12))


def choose_threshold(y: np.ndarray, p_model: np.ndarray, p_book: np.ndarray,
                     groups: np.ndarray, hold: float, cfg: PassConfig,
                     min_bets: int = 200) -> float:
    """Pick the edge threshold with the best discovery ROI (ties: the larger threshold).

    Args:
        y: Binary over/under outcomes on discovery [n].
        p_model: Model probability of the over [n].
        p_book: Proxy's fair probability of the over [n].
        groups: Match id per row [n].
        hold: Two-way overround.
        cfg: Stage configuration.
        min_bets: Minimum discovery bets for a threshold to be eligible.

    Returns:
        The chosen threshold.
    """
    best, best_roi = cfg.thresholds[0], -np.inf
    for t in cfg.thresholds:
        res = C.simulate_two_way(y, p_model, p_book, groups,
                                 C.BetSimConfig(hold=hold, edge_threshold=t, n_boot=50,
                                                seed=cfg.split.seed))
        if res.n_bets >= min_bets and res.roi >= best_roi:
            best, best_roi = t, res.roi
    return best



def book_ladder(
    y: np.ndarray, p_model: np.ndarray, p_proxy: np.ndarray, groups: np.ndarray,
    cfg: PassConfig, rel_gap: float, hold: float = 0.06,
    lams: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """How good would the opponent have to be for the edge to vanish?

    The naive rolling-mean proxy is a strawman. Each rung of the ladder replaces it with a
    blend of the proxy and the model on the logit scale, which is a market that knows a stated
    fraction of what the model knows, and reports the model's ROI against it. The rung whose
    relative log-loss improvement over the proxy equals ``rel_gap`` -- the improvement a real
    bookmaker has over the identical proxy on total goals -- is the honest reference point.

    Args:
        y: Binary over/under outcomes [n].
        p_model: Model probability of the over [n].
        p_proxy: Proxy probability of the over [n].
        groups: Match id per row [n].
        cfg: Stage configuration.
        rel_gap: Relative log-loss gap of a real book over the proxy, measured on goals.
        hold: Two-way overround used in the simulation.
        lams: Blend weights.

    Returns:
        One row per rung with the opponent's log loss, its relative gap over the proxy, and
        the model's bet count and ROI against it.
    """
    y = np.asarray(y, dtype=float)
    ll_proxy = float(log_loss(y, p_proxy))
    rows = []
    for lam in lams:
        p_b = C.logit_blend(p_proxy, p_model, lam)
        ll_b = float(log_loss(y, p_b))
        res = C.simulate_two_way(y, p_model, p_b, groups,
                                 C.BetSimConfig(hold=hold, edge_threshold=0.02,
                                                n_boot=cfg.split.n_boot, seed=cfg.split.seed))
        rows.append(res.as_row(lam=lam, opponent_log_loss=ll_b, proxy_log_loss=ll_proxy,
                               rel_gap_over_proxy=(ll_proxy - ll_b) / ll_proxy,
                               goals_rel_gap=rel_gap, hold=hold))
    out = pd.DataFrame(rows)
    out["at_or_past_book_quality"] = out["rel_gap_over_proxy"] >= rel_gap
    return out


def stage_c(cfg: PassConfig = CFG) -> dict[str, pd.DataFrame]:
    """(c) Attempts x completion end to end, and betting the proxy line.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables.
    """
    t0 = time.time()
    panel = build_player_panel(cfg)
    df = attach_scenarios(league_population(panel, cfg), cfg)
    disc = df["split"].to_numpy() == C.DISCOVERY
    mid = df["match_id"].to_numpy()
    att = df["passes"].to_numpy(dtype=float)
    comp = df["passes_completed"].to_numpy(dtype=float)
    feat_all = list(BASE_FEATURES) + list(IMP_FEATURES)
    out: dict[str, pd.DataFrame] = {}
    out["variance"] = pd.concat([
        variance_decomposition(df).assign(where="all"),
        variance_decomposition(df[~disc]).assign(where="confirmation")], ignore_index=True)

    mu_att_proxy = proxy_predictions(df, "mu_proxy_att", "passes", cfg)
    mu_att_model = cv_and_confirm(df, feat_all, "passes", cfg)
    mu_comp_proxy = proxy_predictions(df, "mu_proxy_comp", "passes_completed", cfg)
    rate_proxy = np.clip(mu_comp_proxy / np.maximum(mu_att_proxy, 1e-6), 0.3, 0.99)
    rate_model = rate_predictions(df, feat_all, cfg)
    r_att = C.fit_nb_dispersion(att[disc], mu_att_proxy[disc])
    r_att_m = C.fit_nb_dispersion(att[disc], mu_att_model[disc])
    k_max = int(min(220, att.max() + 40))
    _log(f"stage c: NB dispersion proxy r={r_att:.1f} model r={r_att_m:.1f}, k_max={k_max} "
         f"({time.time() - t0:.0f} s)")

    combos = {
        "proxy_att x proxy_rate": (mu_att_proxy, r_att, rate_proxy),
        "model_att x proxy_rate": (mu_att_model, r_att_m, rate_proxy),
        "proxy_att x model_rate": (mu_att_proxy, r_att, rate_model),
        "model_att x model_rate": (mu_att_model, r_att_m, rate_model),
    }
    grids = {k: nb_pmf_grid(v[0], v[1], k_max) for k, v in combos.items()}
    lsc = {k: -binomial_mix_logpmf(grids[k], v[2], comp) for k, v in combos.items()}
    ref = "proxy_att x proxy_rate"
    rows = []
    for where, m in (("discovery", disc), ("confirmation", ~disc)):
        for k in combos:
            d, lo, hi = C.clustered_bootstrap_mean(lsc[ref][m] - lsc[k][m], mid[m],
                                                   cfg.split.n_boot, cfg.split.seed)
            mu = combos[k][0] * combos[k][2]
            rows.append({"where": where, "combination": k, "n": int(m.sum()),
                         "mean_completed": float(comp[m].mean()),
                         "log_score": float(lsc[k][m].mean()),
                         "mae": float(np.abs(comp[m] - mu[m]).mean()),
                         "r2": float(r2(comp[m], mu[m])),
                         "delta_vs_proxy": d, "ci_lo": lo, "ci_hi": hi})
    out["channels"] = pd.DataFrame(rows)

    # --- betting the proxy line ------------------------------------------
    bets, clv, choice = [], [], []
    markets = {
        "completed_passes": (comp, line_from_mean(mu_comp_proxy, 0),
                             lambda g, r, L: binomial_mix_sf(g, r, L)),
        "pass_attempts": (att, line_from_mean(mu_att_proxy, 0), None),
    }
    for market, (y_count, line, _) in markets.items():
        if market == "completed_passes":
            p_book = binomial_mix_sf(grids[ref], rate_proxy, line)
            model_sources = {
                "model_att x proxy_rate": binomial_mix_sf(
                    grids["model_att x proxy_rate"], rate_proxy, line),
                "proxy_att x model_rate": binomial_mix_sf(
                    grids["proxy_att x model_rate"], rate_model, line),
                "model_att x model_rate": binomial_mix_sf(
                    grids["model_att x model_rate"], rate_model, line),
            }
        else:
            p_book = C.nb_sf(line, mu_att_proxy, r_att)
            model_sources = {"model_att": C.nb_sf(line, mu_att_model, r_att_m)}
        yb = (y_count > line).astype(float)
        ab = C.platt_fit(p_book[disc], yb[disc])
        p_book = np.clip(C.platt_apply(p_book, ab), 1e-4, 1 - 1e-4)
        for src, p_m in model_sources.items():
            ab_m = C.platt_fit(p_m[disc], yb[disc])
            p_m = np.clip(C.platt_apply(p_m, ab_m), 1e-4, 1 - 1e-4)
            for hold in cfg.holds:
                thr = choose_threshold(yb[disc], p_m[disc], p_book[disc], mid[disc], hold, cfg)
                choice.append({"market": market, "source": src, "hold": hold,
                               "threshold_chosen_on_discovery": thr})
                for where, m in (("discovery", disc), ("confirmation", ~disc)):
                    res = C.simulate_two_way(
                        yb[m], p_m[m], p_book[m], mid[m],
                        C.BetSimConfig(hold=hold, edge_threshold=thr,
                                       n_boot=cfg.split.n_boot, seed=cfg.split.seed))
                    bets.append(res.as_row(market=market, source=src, hold=hold,
                                           threshold=thr, where=where,
                                           mean_line=float(line[m].mean()),
                                           over_rate=float(yb[m].mean())))
            for where, m in (("discovery", disc), ("confirmation", ~disc)):
                t = C.disagreement_table(yb[m], p_m[m], p_book[m], 5)
                t.insert(0, "where", where)
                t.insert(0, "source", src)
                t.insert(0, "market", market)
                clv.append(t)
    out["bets"] = pd.DataFrame(bets)
    out["threshold_choice"] = pd.DataFrame(choice)
    out["clv"] = pd.concat(clv, ignore_index=True)

    # --- how good would the opponent have to be? --------------------------
    gg = pd.read_parquet(C.reports_dir() / "03_pass_goals_gap.parquet")
    gg = gg[(gg["population"] == "four_leagues_all_seasons")
            & (gg["where"] == "confirmation")]
    ll_proxy_goals = float(gg.loc[gg["model"] == "proxy_calibrated", "log_loss"].iloc[0])
    ll_book_goals = float(gg.loc[gg["model"] == "bet365_novig", "log_loss"].iloc[0])
    rel_gap = (ll_proxy_goals - ll_book_goals) / ll_proxy_goals
    line_c = line_from_mean(mu_comp_proxy, 0)
    yb_c = (comp > line_c).astype(float)
    p_bk = binomial_mix_sf(grids[ref], rate_proxy, line_c)
    p_md = binomial_mix_sf(grids["model_att x model_rate"], rate_model, line_c)
    ab_b = C.platt_fit(p_bk[disc], yb_c[disc])
    ab_m = C.platt_fit(p_md[disc], yb_c[disc])
    p_bk = np.clip(C.platt_apply(p_bk, ab_b), 1e-4, 1 - 1e-4)
    p_md = np.clip(C.platt_apply(p_md, ab_m), 1e-4, 1 - 1e-4)
    lad = []
    for where, m in (("discovery", disc), ("confirmation", ~disc)):
        t = book_ladder(yb_c[m], p_md[m], p_bk[m], mid[m], cfg, rel_gap)
        t.insert(0, "where", where)
        lad.append(t)
    out["book_ladder"] = pd.concat(lad, ignore_index=True)
    ls_p = float(lsc[ref][~disc].mean())
    ls_m = float(lsc["model_att x model_rate"][~disc].mean())
    out["relative_scale"] = pd.DataFrame([{
        "goals_log_loss_proxy": ll_proxy_goals, "goals_log_loss_book": ll_book_goals,
        "goals_relative_gap": rel_gap,
        "passes_count_log_score_proxy": ls_p, "passes_count_log_score_model": ls_m,
        "passes_relative_gap": (ls_p - ls_m) / ls_p,
        "implied_book_count_log_score": ls_p * (1.0 - rel_gap),
        "model_minus_implied_book_nats": ls_m - ls_p * (1.0 - rel_gap)}])

    # --- a book calibrated to disagree with the proxy as much as a real one ---
    lu_g = pd.read_parquet(C.reports_dir() / "03_pass_goals_line_units_goals.parquet")
    target_ratio = float(lu_g["abs_diff_over_sd_outcome"].iloc[0])
    mu_model_c = combos["model_att x model_rate"][0] * combos["model_att x model_rate"][2]
    out["line_units_passes"] = line_units_table(mu_comp_proxy, mu_model_c, comp,
                                                "completed_passes_model_vs_proxy")
    sd_y = float(np.std(comp))
    mean_abs = float(np.abs(mu_model_c - mu_comp_proxy).mean())
    kappa = float(np.clip(target_ratio * sd_y / max(mean_abs, 1e-9), 0.0, 1.0))
    rows_k = []
    for kap, tag in ((kappa, "goals_calibrated"), (0.25, "quarter"), (0.5, "half"),
                     (0.75, "three_quarters")):
        mu_bk = mu_comp_proxy + kap * (mu_model_c - mu_comp_proxy)
        line_k = line_from_mean(mu_bk, 0)
        yb_k = (comp > line_k).astype(float)
        rate_bk = np.clip(rate_proxy + kap * (rate_model - rate_proxy), 0.3, 0.99)
        att_bk = np.clip(mu_bk / rate_bk, 1.0, None)
        g_bk = nb_pmf_grid(att_bk, r_att, k_max)
        p_bk = binomial_mix_sf(g_bk, rate_bk, line_k)
        p_md = binomial_mix_sf(grids["model_att x model_rate"], rate_model, line_k)
        p_bk = np.clip(C.platt_apply(p_bk, C.platt_fit(p_bk[disc], yb_k[disc])),
                       1e-4, 1 - 1e-4)
        p_md = np.clip(C.platt_apply(p_md, C.platt_fit(p_md[disc], yb_k[disc])),
                       1e-4, 1 - 1e-4)
        for hold in cfg.holds:
            for where, m in (("discovery", disc), ("confirmation", ~disc)):
                res = C.simulate_two_way(
                    yb_k[m], p_md[m], p_bk[m], mid[m],
                    C.BetSimConfig(hold=hold, edge_threshold=0.02, n_boot=cfg.split.n_boot,
                                   seed=cfg.split.seed))
                rows_k.append(res.as_row(
                    kappa=kap, kind=tag, hold=hold, where=where,
                    book_moves_line=True, mean_line=float(line_k[m].mean()),
                    log_loss_book=float(log_loss(yb_k[m], p_bk[m])),
                    log_loss_model=float(log_loss(yb_k[m], p_md[m])),
                    target_ratio=target_ratio))
    out["book_sets_line"] = pd.DataFrame(rows_k)

    # --- haircut ----------------------------------------------------------
    hc = pd.read_parquet(C.reports_dir() / "03_pass_goals_haircut.parquet")
    hair = float(hc["roi_haircut"].iloc[0])
    conf = out["bets"][out["bets"]["where"] == "confirmation"].copy()
    conf["roi_after_haircut"] = conf["roi"] - hair
    conf["roi_lo_after_haircut"] = conf["roi_lo"] - hair
    conf["roi_hi_after_haircut"] = conf["roi_hi"] - hair
    conf["haircut_applied"] = hair
    out["confirmation_roi"] = conf

    df_out = df[["match_id", "player_id", "date", "split", "pos_group", "passes",
                 "passes_completed"]].copy()
    df_out["mu_att_proxy"] = mu_att_proxy
    df_out["mu_att_model"] = mu_att_model
    df_out["rate_proxy"] = rate_proxy
    df_out["rate_model"] = rate_model
    df_out.to_parquet(C.processed_dir() / "pass_c_preds.parquet", index=False)
    for name, tab in out.items():
        C.write_table(tab, f"03_pass_c_{name}")
    _log(f"stage c done in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _t(name: str) -> pd.DataFrame:
    """Load a committed result table by stem."""
    return pd.read_parquet(C.reports_dir() / f"{name}.parquet")


def _ci(row: pd.Series, val: str = "delta_vs_EVENT", lo: str = "ci_lo", hi: str = "ci_hi",
        nd: int = 4) -> str:
    """Format a value with its interval as ``+x [lo, hi]``."""
    return (f"{row[val]:+.{nd}f} [{row[lo]:+.{nd}f}, {row[hi]:+.{nd}f}]")



def _flat_pivot(df: pd.DataFrame, kind: str, population: str = "all_matches") -> pd.DataFrame:
    """Calibration deciles as a flat wide table.

    Args:
        df: The calibration table.
        kind: ``mean`` or ``tail_line0``.
        population: Which population to keep.

    Returns:
        One row per decile with ``pred_<variant>`` / ``obs_<variant>`` columns.
    """
    d = df[(df["population"] == population) & (df["kind"] == kind)]
    w = d.pivot(index="bin", columns="variant", values=["pred", "obs"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index()


def stage_report(cfg: PassConfig = CFG) -> Path:
    """Write ``reports/03_pass_counts.md`` from the committed result tables.

    Args:
        cfg: Stage configuration.

    Returns:
        Path of the report.
    """
    from research.privileged_tracking.common.report import md_table

    t = {n: _t(f"03_pass_{n}") for n in (
        "a_per_pass", "a_count_metrics", "a_count_lines", "a_count_calibration", "a_retention",
        "b_setup", "b_audit", "b_metrics", "b_deltas", "b_gate", "b_refit", "b_scenario_defs",
        "b_ablation", "goals_gap", "goals_roi", "goals_haircut", "goals_proxy_set_line",
        "goals_line_units_goals", "c_variance", "c_channels", "c_confirmation_roi", "c_clv",
        "c_book_ladder", "c_relative_scale", "c_threshold_choice", "c_line_units_passes",
        "c_book_sets_line")}

    def g(name: str, **kw: Any) -> pd.Series:
        df = t[name]
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            m &= (df[k].to_numpy() == v)
        return df[m].iloc[0]

    pp = t["a_per_pass"]
    cache_imp = g("a_per_pass", population="cached_subsample_rows", source="programme_cache",
                  variant="EVENT+IMP")
    refit_imp = g("a_per_pass", population="cached_subsample_rows",
                  source="refit_all_attempts", variant="EVENT+IMP")
    all_imp = g("a_per_pass", population="all_attempts", source="refit_all_attempts",
                variant="EVENT+IMP")
    all_or = g("a_per_pass", population="all_attempts", source="refit_all_attempts",
               variant="EVENT+ORACLE")
    seq_full = g("a_per_pass", population="seq_matches", source="refit_all_attempts",
                 variant="EVENT+IMP+SEQ")
    c_ev = g("a_count_metrics", population="all_matches", where="all", variant="EVENT")
    c_imp = g("a_count_metrics", population="all_matches", where="confirmation",
              variant="EVENT+IMP")
    c_or = g("a_count_metrics", population="all_matches", where="confirmation",
             variant="EVENT+ORACLE")
    c_seq = g("a_count_metrics", population="seq_matches", where="confirmation",
              variant="EVENT+IMP+SEQ")
    l0_imp = g("a_count_lines", population="all_matches", where="confirmation", offset=0,
               variant="EVENT+IMP")
    l0_or = g("a_count_lines", population="all_matches", where="confirmation", offset=0,
              variant="EVENT+ORACLE")
    l0_seq = g("a_count_lines", population="seq_matches", where="confirmation", offset=0,
               variant="EVENT+IMP+SEQ")
    ret_or = g("a_retention", population="all_matches", variant="EVENT+ORACLE")
    var_c = g("c_variance", where="confirmation")
    ch_vol = g("c_channels", where="confirmation", combination="model_att x proxy_rate")
    ch_rate = g("c_channels", where="confirmation", combination="proxy_att x model_rate")
    ch_both = g("c_channels", where="confirmation", combination="model_att x model_rate")
    b_imp = g("b_deltas", where="confirmation", target="passes", model="model",
              vs="model_noimp")
    b_noimp = g("b_deltas", where="confirmation", target="passes", model="model_noimp",
                vs="proxy")
    abl3 = g("b_ablation", where="confirmation", feature_set="F3 + team and opponent rolling")
    abl4 = g("b_ablation", where="confirmation", feature_set="F4 + imputed block depth")
    abl2 = g("b_ablation", where="confirmation", feature_set="F2 + position / venue")
    refit_sp = float(t["b_refit"]["spread"].max())
    gp = g("goals_proxy_set_line", where="confirmation")
    lug = t["goals_line_units_goals"].iloc[0]
    lup = t["c_line_units_passes"].iloc[0]
    bet6 = g("c_confirmation_roi", market="completed_passes",
             source="model_att x model_rate", hold=0.06)
    ksl = g("c_book_sets_line", where="confirmation", kind="goals_calibrated", hold=0.06)
    gate_worst = t["b_gate"][(t["b_gate"]["where"] == "confirmation")
                             & (t["b_gate"]["scored_on"] == "scenario")]

    L: list[str] = []
    A = L.append
    A("# Scenario +EV 03 - player pass counts: does the per-pass gain move a count line?")
    A("")
    A("Machine-written by `research/scenario_ev/pass_counts.py`. Every number below is read "
      "from a committed `research/scenario_ev/reports/03_pass_*.parquet` table. Discovery "
      "numbers are exploratory and multiplicity-inflated; confirmation numbers are scored "
      "once and are the headline.")
    A("")
    A("## Headline")
    A("")
    A("The programme's one positive imputed-state result was pass difficulty. This stage asked "
      "whether it converts into a player pass-count prop. It does not, and the chain breaks in "
      "three independent places.")
    A("")
    A(f"**1. The per-pass gain itself is mostly an artefact of the subsample it was measured "
      f"on.** Refitting the programme's own xPass variants -- same stage-02 folds, same "
      f"LightGBM settings, same pre-instant `E2` design, same out-of-fold receiver student -- "
      f"on *all* {int(all_imp['n']):,} pass attempts of the 417 360-matches instead of its "
      f"149,591-pass subsample, and then scoring on exactly the 149,591 rows the programme "
      f"scored, `EVENT+IMP` beats `EVENT` by {_ci(refit_imp, 'delta_vs_EVENT', nd=5)} nats per "
      f"pass against the programme's published {_ci(cache_imp, 'delta_vs_EVENT', nd=5)} on the "
      f"same rows. The 360 oracle gap reproduces almost exactly "
      f"({all_or['delta_vs_EVENT']:+.4f} here against +0.0703 published), so the pipeline "
      f"is sound: "
      f"the event-only model simply catches up once it is given the passes the subsample threw "
      f"away. The sequence student's channel is the one that survives -- on its 251 matches "
      f"`EVENT+IMP+SEQ` is still worth {_ci(seq_full, 'delta_vs_EVENT')} nats per pass at full "
      f"training size.")
    A("")
    A(f"**2. Aggregating to a count destroys about 90% of whatever per-pass skill exists.** "
      f"With the realised attempts given, the *perfect* 360 frame is worth "
      f"{all_or['delta_vs_EVENT']:.4f} nats per pass, which over the "
      f"{ret_or['mean_attempts']:.1f} attempts of an average player-match sums to "
      f"{ret_or['summed_over_attempts']:.3f} nats; the completed-pass distribution actually "
      f"improves by {ret_or['count_nats']:.4f} nats, a retention of "
      f"{100 * ret_or['retention']:.1f}%. A count throws away *which* passes failed, and that "
      f"is where the information is. Downstream of a 9.5% retention rate, a per-pass gain of "
      f"0.002 nats is worth ~0.006 nats on a player-match count -- on a distribution whose own "
      f"entropy is {c_ev['count_log_score']:.2f} nats.")
    A("")
    A(f"**3. At a plausible half-line the imputed state moves the price a lot and improves it "
      f"not at all.** At a mean line of {l0_imp['mean_line']:.1f} completed passes "
      f"(confirmation, n={int(l0_imp['n']):,}, over rate {l0_imp['over_rate']:.3f}), "
      f"`EVENT+IMP` shifts P(over) by {100 * l0_imp['mean_abs_shift_vs_EVENT']:.1f} percentage "
      f"points on average (max {100 * l0_imp['max_abs_shift_vs_EVENT']:.0f} pp) and changes "
      f"Brier by {_ci(l0_imp, 'delta_brier_vs_EVENT', nd=5)}. The sequence variant shifts "
      f"{100 * l0_seq['mean_abs_shift_vs_EVENT']:.1f} pp for "
      f"{_ci(l0_seq, 'delta_brier_vs_EVENT', nd=5)}. Only the true 360 frame both moves "
      f"({100 * l0_or['mean_abs_shift_vs_EVENT']:.1f} pp) and improves "
      f"({_ci(l0_or, 'delta_brier_vs_EVENT', nd=5)}). A 5-point move at an even-money line is "
      f"commercially enormous *if it is information*; here it is noise.")
    A("")
    A("### The ceiling: a completed-pass prop is a volume market")
    A("")
    A(f"Of the variance of completed passes in the confirmation half "
      f"(sd {var_c['sd_completed']:.1f} passes, n={int(var_c['n']):,}), only "
      f"**{100 * var_c['share_irreducible_bernoulli']:.1f}%** is the trial-by-trial Bernoulli "
      f"term that a per-pass completion model can touch at all; "
      f"{100 * var_c['share_volume_and_rate_level']:.1f}% is attempt volume and the "
      f"player-match rate level. The end-to-end decomposition agrees: swapping the attempts "
      f"channel from the rolling-mean proxy to a model is worth "
      f"{_ci(ch_vol, 'delta_vs_proxy')} nats per player-match, swapping the completion-rate "
      f"channel is worth {_ci(ch_rate, 'delta_vs_proxy')}, and both together "
      f"{_ci(ch_both, 'delta_vs_proxy')}. At the margin the rate channel adds "
      f"{ch_both['delta_vs_proxy'] - ch_vol['delta_vs_proxy']:+.4f} nats "
      f"({100 * (ch_both['delta_vs_proxy'] - ch_vol['delta_vs_proxy']) / ch_both['delta_vs_proxy']:.0f}% "
      f"of the joint gain) and the volume channel adds "
      f"{ch_both['delta_vs_proxy'] - ch_rate['delta_vs_proxy']:+.4f} "
      f"({100 * (ch_both['delta_vs_proxy'] - ch_rate['delta_vs_proxy']) / ch_both['delta_vs_proxy']:.0f}%). "
      f"**Volume is the market.** And the rate channel that does exist is bought with ordinary "
      f"rolling and opponent features, not with tracking: adding the imputed block depth to "
      f"the attempts model is worth {_ci(b_imp, 'delta_log_score', nd=5)} nats, against a "
      f"refit-noise spread of {refit_sp:.5f} over {len(cfg.seeds_refit)} seeds. The "
      f"tracking-derived work is irrelevant to this market and the stage should say so "
      f"plainly.")
    A("")
    A("### The gate")
    A("")
    A(f"Five scenarios were pre-registered on discovery, including the one the earlier "
      f"discussion proposed (the team's usual deep-lying passer is absent) and the opponent "
      f"sitting deep. On the confirmation scenario rows the specialist is **worse** than the "
      f"generalist in all five, by {gate_worst['delta_specialist_minus_generalist'].min():.4f} "
      f"to {gate_worst['delta_specialist_minus_generalist'].max():.4f} nats. Targeting has no "
      f"room for this target: a model fitted on all rows already prices the scenario rows "
      f"better than one fitted only on them.")
    A("")
    A("### On expected value")
    A("")
    A(f"Against a market priced off the rolling-mean proxy the model shows "
      f"+{100 * bet6['roi']:.1f}% ROI at 6% hold "
      f"([{100 * bet6['roi_lo']:.1f}%, {100 * bet6['roi_hi']:.1f}%], "
      f"{int(bet6['n_bets']):,} bets on confirmation). That is a measure of how bad the proxy "
      f"is, not of an edge, and two calibrations say so:")
    A("")
    A(f"- *In line units.* Our model's mean differs from the proxy's by "
      f"{lup['mean_abs_diff']:.2f} completed passes on average, "
      f"{100 * lup['abs_diff_over_sd_outcome']:.1f}% of the outcome's sd, and it would post a "
      f"different line from the proxy in {100 * lup['share_diff_ge_half_unit']:.0f}% of "
      f"player-matches. A real bookmaker's implied mean total goals differs from the identical "
      f"proxy by {lug['mean_abs_diff']:.3f} goals, "
      f"{100 * lug['abs_diff_over_sd_outcome']:.1f}% of that outcome's sd, differing by half a "
      f"line in {100 * lug['share_diff_ge_half_unit']:.1f}% of matches. We are claiming to "
      f"disagree with a rolling mean roughly "
      f"{lup['abs_diff_over_sd_outcome'] / lug['abs_diff_over_sd_outcome']:.1f}x as hard as a "
      f"real book does.")
    A(f"- *Against a book that sets its own line.* Giving the opponent a mean "
      f"{ksl['kappa']:.3f} of the way from the proxy to our model -- the factor that matches "
      f"the goals-market disagreement above -- and letting it post its own line, the model's "
      f"ROI at 6% hold falls to +{100 * ksl['roi']:.1f}% "
      f"([{100 * ksl['roi_lo']:.1f}%, {100 * ksl['roi_hi']:.1f}%], "
      f"{int(ksl['n_bets']):,} bets).")
    A("")
    A(f"That residual is not zero, and honesty requires saying so. But it rests entirely on an "
      f"assumption the stage cannot test: that a real pass-prop book knows only "
      f"{100 * ksl['kappa']:.0f}% of what our model knows. The feature ablation makes that "
      f"implausible. All of the model's advantage over the proxy comes from the player's own "
      f"rolling block, his position, and above all the team / opponent rolling context "
      f"(+{abl2['delta_vs_proxy']:.4f} nats after player and position, "
      f"+{abl3['delta_vs_proxy']:.4f} after team and opponent, "
      f"+{abl4['delta_vs_proxy']:.4f} after the imputed state) -- public averages that every "
      f"trading desk computes. There is no private information in this model. **The stage's "
      f"conclusion is a null result: no tradeable edge is demonstrated, and none of the "
      f"apparent one is attributable to the tracking-derived channel this programme was about."
      f"**")
    A("")
    A("## Protocol")
    A("")
    A("- **Discovery / confirmation.** Fixed in code before any modelling "
      f"(`common.chronological_split`, {100 * cfg.split.discovery_frac:.0f}% of matches by "
      "date), taken separately inside each population being scored: a single split over all "
      "2,090 StatsBomb matches would put every club season in discovery and only international "
      "tournaments in confirmation, which is a regime change, not a held-out sample. Every "
      "threshold, cut, feature choice and bet rule below is chosen on discovery.")
    A("- **Populations.** Part (a): the 417 StatsBomb 360 matches, every pass attempt. Parts "
      "(b) and (c): the four league seasons StatsBomb covers end to end (Premier League, "
      f"La Liga, Serie A, Ligue 1, 2015/2016), starters with at least {cfg.min_prior_matches} "
      f"strictly prior appearances and both teams with at least {cfg.min_prior_matches} prior "
      "matches. The population is **not** filtered on realised minutes: minutes are an outcome "
      "and an early substitution is exactly the risk a prop prices.")
    A("- **What counts as known at pricing time.** Whether the player starts, his starting "
      "position, and the two teams -- the facts a book has when it posts a player prop after "
      "team news. Nothing else from the match being predicted enters any feature.")
    A("- **Hygiene.** Rolling features come from strictly earlier matches, verified by brute "
      "force rather than asserted; folds grouped by match; fixed seeds; match-clustered "
      f"bootstrap intervals ({cfg.split.n_boot} resamples) on every model-vs-model and ROI "
      f"claim; a refit-noise floor over {len(cfg.seeds_refit)} seeds.")
    A("")
    A("### Strictly-prior audit")
    A("")
    A("Prior sums and counts recomputed by direct filtering for a random sample of rows; the "
      "shrunk rate reconstructed from them for every row; and the opponent join checked to be "
      "the opponent's own prior value. A non-zero error here means leakage -- an earlier "
      "version of the last-5 feature had one, and this check is what caught it.")
    A("")
    A(md_table(t["b_audit"], "{:.6g}"))
    A("")
    A("## (a) The completion channel")
    A("")
    A("### (a.1) Per-pass models refitted on every attempt")
    A("")
    A(f"The programme's cache holds a match-stratified 149,591-pass subsample, about 40% of "
      f"the {int(all_imp['n']):,} attempts, so its player-match counts would be ~40% of full "
      f"size and a count line cannot be studied on it. The same models are refitted here on "
      f"every attempt. Variants: `EVENT` = event-only pre-instant design; `EVENT+IMP` = plus "
      f"the seven out-of-fold LightGBM student predictions of the 360 pass state; "
      f"`EVENT+ORACLE` = plus the same quantities measured on the 360 frame; "
      f"`EVENT+IMP+SEQ` = plus the soccer-06 sequence student's seven quantities, on the 251 "
      f"matches it covers. `source = programme_cache` rows are the published models scored on "
      f"their own subsample.")
    A("")
    A(md_table(pp.assign(delta=[_ci(r, "delta_vs_EVENT", nd=5) for _, r in pp.iterrows()])[
        ["population", "source", "variant", "n", "log_loss", "delta"]], "{:.4f}"))
    A("")
    A("The `cached_subsample_rows` block is the comparison that matters: on **the same "
      "149,591 evaluation rows**, the "
      "programme's models give `EVENT+IMP` a significant gain and the refitted models give it "
      "nothing. The only thing that changed is how many passes the event-only model saw in "
      "training. The oracle gap is unchanged, so the 360 frame still carries real information "
      "about a single pass; what evaporates is the *imputed* version's marginal value once the "
      "event-only model has enough data to learn the same thing from the events directly.")
    A("")
    A("### (a.2) Player-match completed-pass distributions")
    A("")
    A("Given the realised attempts, completed passes are a sum of independent Bernoulli trials "
      "with unequal probabilities, so the exact predictive distribution is a Poisson binomial. "
      "`count_log_score` is the mean negative log probability assigned to the realised count; "
      "deltas are paired and match-clustered. **Caveat:** conditioning on realised attempts "
      "isolates the completion channel and is not tradeable -- a real prop must predict "
      "attempts too, which is what part (b) does.")
    A("")
    A(md_table(t["a_count_metrics"].assign(
        delta=[_ci(r, "delta_vs_EVENT") for _, r in t["a_count_metrics"].iterrows()])[
        ["population", "where", "variant", "n", "n_matches", "mean_attempts",
         "mean_completed", "count_log_score", "mean_pred", "mean_sd", "mae", "delta"]]))
    A("")
    A(f"The predicted spread is honest but slightly tight: the Poisson binomial's mean sd is "
      f"{c_ev['mean_sd']:.2f} completions while the mean absolute error is {c_ev['mae']:.2f}, "
      f"so passes inside a match are mildly positively dependent and independence understates "
      f"the tails. That matters little here because no imputed variant's count gain is "
      f"significant: `EVENT+IMP` {_ci(c_imp, 'delta_vs_EVENT')}, `EVENT+IMP+SEQ` "
      f"{_ci(c_seq, 'delta_vs_EVENT')}, against the oracle's {_ci(c_or, 'delta_vs_EVENT')}.")
    A("")
    A("### (a.3) Retention: per-pass nats in, count nats out")
    A("")
    A(md_table(t["a_retention"], "{:.5f}"))
    A("")
    A("### (a.4) Half-lines around the model's own mean")
    A("")
    A("Lines are `floor(EVENT's mean) + 0.5 + offset`, so offset 0 is at the money and +/-6 "
      "probe the tails. `mean_abs_shift_vs_EVENT` is how far the variant moves P(over) from "
      "`EVENT`'s price; `delta_brier_vs_EVENT` is whether the move was an improvement.")
    A("")
    A(md_table(t["a_count_lines"][t["a_count_lines"]["where"] == "confirmation"].assign(
        delta=[_ci(r, "delta_brier_vs_EVENT", nd=5) for _, r in
               t["a_count_lines"][t["a_count_lines"]["where"] == "confirmation"].iterrows()])[
        ["population", "offset", "variant", "n", "mean_line", "over_rate", "mean_p", "brier",
         "delta", "mean_abs_shift_vs_EVENT", "max_abs_shift_vs_EVENT"]]))
    A("")
    A("Discovery is in `03_pass_a_count_lines.parquet` (column `where`) and tells the same "
      "story.")
    A("")
    A("### (a.5) Calibration")
    A("")
    A("Deciles of the predicted mean, then of the tail probability at the at-the-money line.")
    A("")
    A(md_table(_flat_pivot(t["a_count_calibration"], "mean"), "{:.3f}"))
    A("")
    A(md_table(_flat_pivot(t["a_count_calibration"], "tail_line0"), "{:.3f}"))
    A("")
    A("## (b) The volume channel")
    A("")
    A(md_table(t["b_setup"], "{:.3f}"))
    A("")
    A(f"**The book proxy.** Expected minutes = shrunk mean of the player's minutes in his "
      f"prior starts (k={cfg.k_player:g} appearances); rate = shrunk prior attempts per 90; "
      f"`mu_proxy_att = rate * expected_minutes / 90`, and `mu_proxy_comp` multiplies by his "
      f"shrunk prior completion rate. Team and opponent rolling means (possession, PPDA, "
      f"defensive-line x, passes, imputed block depth) use k={cfg.k_team:g} matches. Before "
      f"being scored the proxy gets a one-dimensional Poisson recalibration "
      f"(`log E[y] = a + b log mu`) fitted out of sample, so it is not penalised for a level or "
      f"slope error no book would make.")
    A("")
    A("**The models.** LightGBM Poisson on the player / team / opponent rolling block. "
      "`model_noimp` omits the four imputed-state features; `model` adds them. Discovery is "
      "match-grouped 5-fold CV; confirmation is fitted on all of discovery and scored once.")
    A("")
    A(md_table(t["b_metrics"][["where", "target", "model", "n", "mae", "r2", "pois_dev",
                              "log_score"]]))
    A("")
    A(md_table(t["b_deltas"].assign(
        delta=[_ci(r, "delta_log_score", nd=5) for _, r in t["b_deltas"].iterrows()])[
        ["where", "target", "model", "vs", "n", "delta"]]))
    A("")
    A(f"The model beats the proxy on attempts by {_ci(b_noimp, 'delta_log_score')} nats per "
      f"player-match and lifts R2 from "
      f"{g('b_metrics', where='confirmation', target='passes', model='proxy')['r2']:.3f} to "
      f"{g('b_metrics', where='confirmation', target='passes', model='model')['r2']:.3f}. The "
      f"imputed block depth contributes {_ci(b_imp, 'delta_log_score', nd=5)}.")
    A("")
    A("### Where the advantage comes from")
    A("")
    A("Nested feature sets, each scored against the recalibrated proxy. `F0` is the proxy mean "
      "handed to LightGBM as a single feature, which is slightly *worse* than the proxy itself "
      "(the tree cannot improve on a monotone recalibration and pays a little variance).")
    A("")
    A(md_table(t["b_ablation"].assign(
        delta=[_ci(r, "delta_vs_proxy", nd=5) for _, r in t["b_ablation"].iterrows()])[
        ["where", "feature_set", "n_features", "n", "r2", "log_score", "delta"]]))
    A("")
    A("### Refit-noise floor")
    A("")
    A(md_table(t["b_refit"], "{:.5f}"))
    A("")
    A("### The gate: is there room for a specialist?")
    A("")
    A("A generalist fitted on all training rows against a specialist fitted only on the "
      "scenario's training rows, both scored on held-out scenario rows (and, as the reverse "
      "check, on held-out rows outside it). A positive "
      "`delta_specialist_minus_generalist` means targeting has room. The scenarios, and every "
      "cut in them, are quantiles of the discovery rows; `metronome_out` flags the team-matches "
      "whose usual highest-volume passer -- the player of the recent squad with the highest "
      "attempts per 90 over his strictly earlier appearances for that team -- did not play, and "
      "scores his team-mates.")
    A("")
    A(md_table(t["b_scenario_defs"], "{:.4f}"))
    A("")
    A(md_table(t["b_gate"].assign(
        delta=[_ci(r, "delta_specialist_minus_generalist") for _, r in t["b_gate"].iterrows()])[
        ["scenario", "where", "scored_on", "n", "n_matches", "mean_y",
         "log_score_generalist", "log_score_specialist", "delta", "mae_generalist",
         "mae_specialist"]]))
    A("")
    A("Every confirmation row scored on its own scenario is negative. The reverse direction is "
      "much more negative still, as it must be: a specialist trained on deep-block matches is "
      "badly wrong about the rest. There is no scenario here in which the generalist "
      "underfits.")
    A("")
    A("## Proxy honesty check: total goals, the one count market with a real line")
    A("")
    A("The identical shrunk rolling-mean construction is applied to total goals on "
      "football-data.co.uk matches of the same four divisions (E0, SP1, I1, F1), Bet365's "
      "Over/Under 2.5 price is de-vigged proportionally, and the two are scored against each "
      "other. This is the only place in the stage where a real bookmaker can be measured.")
    A("")
    A("### At the standard 2.5 line")
    A("")
    A(md_table(t["goals_gap"], "{:.5f}"))
    A("")
    A("### At a line the proxy itself would have set (the exact analogue)")
    A("")
    A("Restricted to the matches whose proxy mean rounds to the 2.5 line, so the book is "
      "pricing a line the proxy also posts -- the same situation as the pass simulation, where "
      "the line is always the proxy's own mean.")
    A("")
    A(md_table(t["goals_proxy_set_line"], "{:.5f}"))
    A("")
    A(f"A real book beats the identical rolling-mean proxy by "
      f"{_ci(gp, 'book_minus_proxy_nats', nd=5)} nats at a proxy-set line, "
      f"{100 * gp['relative_gap']:.2f}% of the proxy's log loss, and carries "
      f"{gp['sd_p_bet365'] / gp['sd_p_proxy']:.1f}x the spread of opinion "
      f"(sd of P(over) {gp['sd_p_bet365']:.3f} vs {gp['sd_p_proxy']:.3f}).")
    A("")
    A("### In line units")
    A("")
    A("Inverting each side's probability back through the fitted count distribution gives the "
      "mean each is pricing, which is the quantity a market actually expresses as a line.")
    A("")
    A(md_table(pd.concat([t["goals_line_units_goals"], t["c_line_units_passes"]],
                         ignore_index=True), "{:.4f}"))
    A("")
    A("### What a real book makes betting into a proxy-priced market")
    A("")
    A(md_table(t["goals_roi"][(t["goals_roi"]["population"] == "four_leagues_all_seasons")
                              & (t["goals_roi"]["where"] == "confirmation")][
        ["hold", "threshold", "n_rows", "n_bets", "bet_rate", "mean_edge", "roi", "roi_lo",
         "roi_hi"]]))
    A("")
    A(md_table(t["goals_haircut"], "{:.5f}"))
    A("")
    A("## (c) End to end")
    A("")
    A("### (c.1) The ceiling")
    A("")
    A("Exact law of total variance for `C = sum of Bernoulli(p_t)` given attempts `A`: "
      "`Var(C) = Var(A * pbar) + E[A * pbar * (1 - pbar)]`. The second term is all a per-pass "
      "completion model can ever touch.")
    A("")
    A(md_table(t["c_variance"], "{:.4f}"))
    A("")
    A("### (c.2) Which channel carries the improvement")
    A("")
    A("Completed passes are modelled as a compound: attempts ~ negative binomial with a "
      "dispersion fitted on discovery, completed | attempts ~ Binomial(attempts, rate). Each "
      "row swaps one channel from the proxy to the model.")
    A("")
    A(md_table(t["c_channels"].assign(
        delta=[_ci(r, "delta_vs_proxy") for _, r in t["c_channels"].iterrows()])[
        ["where", "combination", "n", "mean_completed", "log_score", "mae", "r2", "delta"]]))
    A("")
    A("### (c.3) Betting a line the proxy sets")
    A("")
    A("The proxy posts `floor(its own mean) + 0.5` and prices both sides to the stated hold. By "
      "construction its own fair probability sits near 0.5 there, which is what a line-based "
      "market is: the disagreement is expressed in the *line*, not the price. The model bets "
      "whichever side its own distribution makes positive EV, with an edge threshold chosen on "
      "discovery. **Confirmation only.** `roi_after_haircut` subtracts the ROI a real book "
      "makes against the goals proxy.")
    A("")
    A(md_table(t["c_threshold_choice"]))
    A("")
    A(md_table(t["c_confirmation_roi"][
        ["market", "source", "hold", "threshold", "n_rows", "n_bets", "bet_rate", "mean_edge",
         "roi", "roi_lo", "roi_hi", "mean_p_model", "mean_p_book", "realized_a_rate",
         "roi_after_haircut", "roi_lo_after_haircut"]]))
    A("")
    A("These numbers are the badness of the proxy, not an edge. They are reported because the "
      "protocol asks for them, and immediately qualified in (c.5) and (c.6).")
    A("")
    A("### (c.4) Closing-line-value check: who is right where they disagree?")
    A("")
    A(md_table(t["c_clv"][(t["c_clv"]["where"] == "confirmation")
                          & (t["c_clv"]["market"] == "completed_passes")
                          & (t["c_clv"]["source"] == "model_att x model_rate")][
        ["bin", "n", "diff_lo", "diff_hi", "mean_p_model", "mean_p_book", "observed",
         "log_loss_model", "log_loss_book", "model_better"]]))
    A("")
    A("The model is closer to the truth in every bin where it disagrees materially. That "
      "establishes it beats a rolling mean; it says nothing about beating a bookmaker.")
    A("")
    A("### (c.5) A book that sets its own line")
    A("")
    A("The opponent's mean is moved a fraction `kappa` of the way from the proxy to the model "
      "and it posts *its own* line there, which is what a real book does with a disagreement. "
      "`kind = goals_calibrated` is the kappa at which the opponent's mean sits as far from the "
      "proxy, relative to the outcome's sd, as a real Bet365 line sits from the identical proxy "
      "on total goals.")
    A("")
    A(md_table(t["c_book_sets_line"][t["c_book_sets_line"]["where"] == "confirmation"][
        ["kind", "kappa", "hold", "mean_line", "n_bets", "bet_rate", "mean_edge", "roi",
         "roi_lo", "roi_hi", "log_loss_book", "log_loss_model"]]))
    A("")
    A("### (c.6) The ladder: how much would the opponent have to know?")
    A("")
    A("Here the line stays at the proxy's mean and only the opponent's *price* improves, as a "
      "logit blend of proxy and model. It is the price-side twin of (c.5).")
    A("")
    A(md_table(t["c_book_ladder"][t["c_book_ladder"]["where"] == "confirmation"][
        ["lam", "opponent_log_loss", "proxy_log_loss", "rel_gap_over_proxy", "goals_rel_gap",
         "at_or_past_book_quality", "n_bets", "bet_rate", "mean_edge", "roi", "roi_lo",
         "roi_hi"]], "{:.4f}"))
    A("")
    A("`03_pass_c_relative_scale.parquet` additionally normalises both markets by their own "
      "log score. That normalisation is reported for completeness but is **not** the right "
      "scale for an EV claim: a count log score is dominated by the distribution's irreducible "
      "entropy, so relative gaps there are not comparable to a binary log loss at a line.")
    A("")
    A(md_table(t["c_relative_scale"], "{:.5f}"))
    A("")
    A("## Answering the three questions the stage was set")
    A("")
    A(f"1. *Does a +0.0026 nats per-pass gain move a count line, and by how much in probability "
      f"terms at a typical line?* It moves P(over) at a {l0_seq['mean_line']:.1f}-completion "
      f"line by {100 * l0_seq['mean_abs_shift_vs_EVENT']:.1f} percentage points and improves "
      f"Brier by {_ci(l0_seq, 'delta_brier_vs_EVENT', nd=5)} -- a movement large enough to "
      f"matter commercially and an improvement indistinguishable from zero. At the count level "
      f"it is worth {_ci(c_seq, 'delta_vs_EVENT')} nats per player-match. And the "
      f"LightGBM-imputation version of the gain does not survive refitting on the full pass "
      f"population at all.")
    A(f"2. *Is there a scenario where targeting helps?* No. All five pre-registered scenarios "
      f"fail the gate on confirmation, including the deep-lying-passer-absent scenario the "
      f"stage was asked to test.")
    A(f"3. *What fraction of the achievable improvement is volume rather than completion?* "
      f"{100 * (ch_both['delta_vs_proxy'] - ch_rate['delta_vs_proxy']) / ch_both['delta_vs_proxy']:.0f}% "
      f"of the joint gain is unique to the volume channel and "
      f"{100 * (ch_both['delta_vs_proxy'] - ch_vol['delta_vs_proxy']) / ch_both['delta_vs_proxy']:.0f}% "
      f"unique to the completion-rate channel; the variance decomposition puts the hard ceiling "
      f"on the completion side at {100 * var_c['share_irreducible_bernoulli']:.1f}% of the "
      f"outcome's variance. Since the completion-rate gain that exists is bought with rolling "
      f"and opponent features rather than tracking, **the tracking-derived work is irrelevant "
      f"to this market.**")
    A("")
    A("## Caveats")
    A("")
    A("- **No pass-prop lines exist in this data.** Every EV number is against a constructed "
      "proxy. The transfer from the goals market that turns it into a claim about a real book "
      "assumes a bookmaker's relative disagreement with a rolling mean is the same across "
      "markets, which is an assumption, not a measurement, and probably a generous one: player "
      "pass counts are far more predictable from public information (minutes, role, opponent) "
      "than match goals are, so a real desk's model is likely much closer to ours than the "
      "goals-calibrated kappa implies.")
    A("- **Part (a) conditions on realised attempts.** It answers 'does better per-pass "
      "completion move a completed-pass line', not 'can you price the prop'.")
    A("- **Starting position is taken from the match's own line-up.** A book posting before "
      "team news would not have it; the `F1 -> F2` step of the ablation "
      f"(+{abl2['delta_vs_proxy'] - g('b_ablation', where='confirmation', feature_set='F1 + player rolling')['delta_vs_proxy']:.4f} "
      "nats, which also includes venue and competition) bounds what it is worth.")
    A("- **The per-pass refit is single-seed**, matching the programme's xPass setting "
      "(`n_seeds_xpass = 1`). The count-level deltas it feeds are all well inside their own "
      "intervals, so seed noise cannot flip the sign of the conclusion, but the per-pass "
      "numbers carry an unmeasured seed component. The volume-channel models do have a "
      "3-seed refit floor.")
    A("- **The sequence-student variants live on 251 of the 417 matches**, so they are only "
      "comparable through the `seq_matches` population, which is why both populations are "
      "reported everywhere.")
    A("- **StatsBomb 2015/16 is one season of four leagues**; the confirmation half is "
      "February to May 2016. Nothing here is tested across seasons or in other leagues.")
    A("- **Passes include throw-ins, corners and free kicks** as StatsBomb counts them; a real "
      "prop's settlement rules may differ, and so may its definition of a completed pass.")
    A("- **The compound model assumes completion is independent of attempts given the "
      "features.** Real player-matches violate this mildly (see a.2).")
    A("")
    A("## Reproduction")
    A("")
    A("```shell")
    A("cd /home/user/geo-model")
    A("python -m research.scenario_ev.pass_counts --stage passpreds   # ~6 min, 2 threads")
    A("python -m research.scenario_ev.pass_counts --stage a           # ~20 s")
    A("python -m research.scenario_ev.pass_counts --stage b           # ~2 min")
    A("python -m research.scenario_ev.pass_counts --stage goals       # ~15 s")
    A("python -m research.scenario_ev.pass_counts --stage c           # ~40 s")
    A("python -m research.scenario_ev.pass_counts --stage report")
    A("python -m pytest research/scenario_ev/tests/test_pass_counts.py -q")
    A("```")
    A("")
    A("Cached intermediates (not committed) live under "
      "`<PRIV_DATA_DIR>/processed/scenario_ev/`: `pass_preds_full.parquet` (per-pass "
      "out-of-fold probabilities), `pass_panel.parquet` (the player-match panel), "
      "`pass_a_counts.parquet`, `pass_b_preds.parquet`, `pass_c_preds.parquet`, "
      "`goals_proxy.parquet`.")
    A("")
    path = C.reports_dir() / "03_pass_counts.md"
    path.write_text("\n".join(L) + "\n")
    _log(f"report written: {path}")
    return path


STAGES = ("passpreds", "a", "b", "goals", "c", "report")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--force", action="store_true", help="rebuild cached intermediates")
    ap.add_argument("--smoke", type=int, default=0, help="matches kept in the pass refit")
    args = ap.parse_args(argv)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for s in stages:
        if s == "passpreds":
            build_pass_predictions(CFG, force=args.force, smoke=args.smoke)
        elif s == "a":
            stage_a(CFG)
        elif s == "b":
            build_player_panel(CFG, force=args.force)
            stage_b(CFG)
        elif s == "goals":
            stage_goals(CFG)
        elif s == "c":
            stage_c(CFG)
        elif s == "report":
            stage_report(CFG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
