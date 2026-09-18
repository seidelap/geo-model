"""Shared machinery for the scenario expected-value stages.

Everything here is deliberately small and pure where it can be: strictly-prior rolling
statistics, empirical-Bayes shrinkage, two-way market pricing, and a betting simulator.
The rule the whole phase lives by is that *every* feature describing a match must be
computable from strictly earlier matches, and that the discovery / confirmation split is
fixed in code before any modelling.

Array shapes are given in bracket notation, e.g. ``[n]`` for a per-row vector and
``[n, k]`` for a design matrix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import data_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def reports_dir() -> Path:
    """Directory for committed result tables and markdown reports.

    Returns:
        Path to ``research/scenario_ev/reports`` (created if missing).
    """
    p = REPO_ROOT / "research" / "scenario_ev" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    """Directory for uncommitted processed intermediates.

    Returns:
        Path to ``<PRIV_DATA_DIR>/processed/scenario_ev`` (created if missing).
    """
    p = data_dir() / "processed" / "scenario_ev"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sb_processed_dir() -> Path:
    """Directory holding the StatsBomb aggregate tables."""
    return data_dir() / "sb" / "processed"


def soccer_processed_dir() -> Path:
    """Directory holding the completed programme's soccer intermediates."""
    return data_dir() / "processed" / "soccer"


def odds_path() -> Path:
    """Path to the football-data.co.uk match table."""
    return data_dir() / "odds" / "matches.parquet"


# ---------------------------------------------------------------------------
# Discovery / confirmation split
# ---------------------------------------------------------------------------

DISCOVERY = "discovery"
CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration of the fixed chronological discovery / confirmation split.

    Attributes:
        discovery_frac: Fraction of *matches* (not rows) assigned to discovery.
        seed: Global seed used for every model fit and bootstrap in the stage.
        n_boot: Bootstrap replicates for clustered intervals.
    """

    discovery_frac: float = 0.60
    seed: int = 20260918
    n_boot: int = 2000


def chronological_split(
    match_keys: pd.Series | np.ndarray,
    order: pd.Series | np.ndarray,
    discovery_frac: float = 0.60,
) -> np.ndarray:
    """Assign rows to discovery / confirmation by a chronological cut on matches.

    The cut is taken over *distinct matches* ordered by ``order`` so that both halves
    contain whole matches and no match straddles the boundary.

    Args:
        match_keys: Match identifier per row [n].
        order: Sort key per row (a date, or a (date, id) rank) [n].
        discovery_frac: Fraction of distinct matches placed in discovery.

    Returns:
        Array [n] of ``"discovery"`` / ``"confirmation"`` labels.
    """
    mk = pd.Series(np.asarray(match_keys))
    od = pd.Series(np.asarray(order))
    per_match = pd.DataFrame({"m": mk, "o": od}).groupby("m", sort=False)["o"].min()
    per_match = per_match.sort_values(kind="mergesort")
    n_disc = int(round(len(per_match) * discovery_frac))
    n_disc = max(1, min(len(per_match) - 1, n_disc))
    disc_matches = set(per_match.index[:n_disc])
    return np.where(mk.isin(disc_matches).to_numpy(), DISCOVERY, CONFIRMATION)


# ---------------------------------------------------------------------------
# Strictly-prior rolling statistics
# ---------------------------------------------------------------------------


def prior_expanding(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    order_cols: Sequence[str],
    count_name: str = "prior_n",
    prefix: str = "prior_sum_",
) -> pd.DataFrame:
    """Expanding sums over *strictly earlier* rows within each group.

    For row ``i`` in group ``g`` the returned ``prior_sum_<v>`` is the sum of ``v`` over
    the rows of ``g`` that sort strictly before ``i`` under ``order_cols``; ``prior_n``
    is how many such rows there are. Ties in ``order_cols`` are broken by the frame's
    original order, so ``order_cols`` should include a unique tiebreaker (e.g. match id)
    whenever two rows of a group can share a date.

    Args:
        df: Input frame.
        group_cols: Grouping keys (e.g. ``["player_id"]``).
        value_cols: Numeric columns to accumulate.
        order_cols: Sort keys defining "earlier".
        count_name: Name of the prior-count column.
        prefix: Prefix for the prior-sum columns.

    Returns:
        Frame indexed like ``df`` with columns ``prefix + v`` for each value column and
        ``count_name``. NaNs in ``value_cols`` are treated as zero for the sum but the
        row still counts towards ``prior_n``.
    """
    work = df[list(group_cols) + list(value_cols) + list(order_cols)].reset_index(drop=True)
    pos = np.arange(len(work))
    work = work.assign(__pos=pos)
    work = work.sort_values(list(order_cols) + ["__pos"], kind="mergesort")
    for v in value_cols:
        work[v] = work[v].astype(float).fillna(0.0)
    g = work.groupby(list(group_cols), sort=False, dropna=False)
    cols: dict[str, np.ndarray] = {}
    inv = np.empty(len(work), dtype=np.int64)
    inv[work["__pos"].to_numpy()] = np.arange(len(work))
    for v in value_cols:
        prior = (g[v].cumsum() - work[v]).to_numpy()
        cols[prefix + v] = prior[inv]
    cols[count_name] = g.cumcount().to_numpy().astype(float)[inv]
    return pd.DataFrame(cols, index=df.index)


def audit_strictly_prior(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    order_cols: Sequence[str],
    prior_sum_col: str,
    prior_n_col: str,
    n_check: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Brute-force verification that prior statistics use strictly earlier rows only.

    Recomputes the prior sum and count for a random sample of rows by direct filtering
    and compares with the vectorised result.

    Args:
        df: Frame carrying both the raw values and the computed prior columns.
        group_cols: Grouping keys used to build the prior columns.
        value_col: Raw value column.
        order_cols: Sort keys used to build the prior columns.
        prior_sum_col: Name of the computed prior-sum column.
        prior_n_col: Name of the computed prior-count column.
        n_check: Number of rows to verify.
        seed: RNG seed for the sample.

    Returns:
        Dict with ``n_checked``, ``max_abs_sum_error`` and ``max_abs_count_error``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_check, len(df)), replace=False)
    ord_arr = df[list(order_cols)].to_numpy()
    grp_arr = df[list(group_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    vals = df[value_col].astype(float).fillna(0.0).to_numpy()
    pos = np.arange(len(df))
    max_s, max_c = 0.0, 0.0
    for i in idx:
        same = grp_arr == grp_arr[i]
        earlier = np.zeros(len(df), dtype=bool)
        for j in np.flatnonzero(same):
            a, b = ord_arr[j], ord_arr[i]
            if tuple(a) < tuple(b) or (tuple(a) == tuple(b) and pos[j] < pos[i]):
                earlier[j] = True
        max_s = max(max_s, abs(vals[earlier].sum() - float(df[prior_sum_col].to_numpy()[i])))
        max_c = max(max_c, abs(earlier.sum() - float(df[prior_n_col].to_numpy()[i])))
    return {"n_checked": float(len(idx)), "max_abs_sum_error": max_s, "max_abs_count_error": max_c}


def shrunk_rate(
    prior_sum: np.ndarray,
    prior_n: np.ndarray,
    prior_mean: np.ndarray | float,
    k: float,
) -> np.ndarray:
    """Empirical-Bayes shrinkage of a rate towards a prior mean.

    ``rate = (prior_sum + k * prior_mean) / (prior_n + k)``; with ``prior_n == 0`` this
    returns ``prior_mean`` exactly.

    Args:
        prior_sum: Sum of the quantity over strictly earlier rows [n].
        prior_n: Count of strictly earlier rows (or exposure) [n].
        prior_mean: Backstop mean, scalar or [n].
        k: Shrinkage strength in units of ``prior_n``.

    Returns:
        Shrunk rate [n].
    """
    prior_sum = np.asarray(prior_sum, dtype=float)
    prior_n = np.asarray(prior_n, dtype=float)
    pm = np.asarray(prior_mean, dtype=float)
    denom = prior_n + k
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (prior_sum + k * pm) / denom
    return np.where(denom > 0, out, np.broadcast_to(pm, np.shape(out)))


# ---------------------------------------------------------------------------
# Market pricing
# ---------------------------------------------------------------------------


def novig_two_way(odds_a: np.ndarray, odds_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a two-way decimal price pair to no-vig probabilities.

    Uses proportional (multiplicative) de-vigging: ``p_a = (1/o_a) / (1/o_a + 1/o_b)``.

    Args:
        odds_a: Decimal odds of side A [n].
        odds_b: Decimal odds of side B [n].

    Returns:
        Tuple ``(p_a, p_b)``, each [n], summing to one.
    """
    ia = 1.0 / np.asarray(odds_a, dtype=float)
    ib = 1.0 / np.asarray(odds_b, dtype=float)
    tot = ia + ib
    return ia / tot, ib / tot


def price_two_way(p: np.ndarray, hold: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn a fair probability into a two-way priced market with a stated hold.

    Both sides are marked up proportionally so the implied probabilities sum to
    ``1 + hold``.

    Args:
        p: Fair probability of the "over"/A side [n].
        hold: Two-way overround, e.g. 0.06 for a 6% book.

    Returns:
        Tuple ``(odds_a, odds_b)`` of decimal prices, each [n].
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    qa = p * (1.0 + hold)
    qb = (1.0 - p) * (1.0 + hold)
    return 1.0 / qa, 1.0 / qb


# ---------------------------------------------------------------------------
# Betting simulation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetSimConfig:
    """Configuration of a two-way betting simulation.

    Attributes:
        hold: Two-way overround applied to the book proxy's fair probability.
        edge_threshold: Minimum expected value per unit stake required to bet.
        n_boot: Bootstrap replicates for the ROI interval.
        seed: RNG seed.
    """

    hold: float = 0.06
    edge_threshold: float = 0.02
    n_boot: int = 2000
    seed: int = 20260918


@dataclass(frozen=True)
class BetSimResult:
    """Outcome of a betting simulation.

    Attributes:
        n_rows: Candidate markets offered.
        n_bets: Markets actually bet.
        bet_rate: ``n_bets / n_rows``.
        mean_edge: Mean modelled EV per unit stake among placed bets.
        roi: Realised profit per unit staked.
        roi_lo: Lower end of the match-clustered bootstrap interval.
        roi_hi: Upper end of the match-clustered bootstrap interval.
        n_over: Number of placed bets on the over/A side.
        profit: Total profit in units.
        mean_p_model: Mean model probability of the A side among placed bets.
        mean_p_book: Mean book fair probability of the A side among placed bets.
        realized_a_rate: Observed rate of the A side among placed bets.
        mean_odds: Mean decimal price actually taken.
    """

    n_rows: int
    n_bets: int
    bet_rate: float
    mean_edge: float
    roi: float
    roi_lo: float
    roi_hi: float
    n_over: int
    profit: float
    mean_p_model: float = float("nan")
    mean_p_book: float = float("nan")
    realized_a_rate: float = float("nan")
    mean_odds: float = float("nan")

    def as_row(self, **extra: object) -> dict[str, object]:
        """Flatten to a dict row for a result table."""
        row: dict[str, object] = dict(extra)
        row.update(
            n_rows=self.n_rows,
            n_bets=self.n_bets,
            bet_rate=self.bet_rate,
            mean_edge=self.mean_edge,
            roi=self.roi,
            roi_lo=self.roi_lo,
            roi_hi=self.roi_hi,
            n_over=self.n_over,
            profit=self.profit,
            mean_p_model=self.mean_p_model,
            mean_p_book=self.mean_p_book,
            realized_a_rate=self.realized_a_rate,
            mean_odds=self.mean_odds,
        )
        return row


def clustered_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a cluster (group) bootstrap interval.

    Args:
        values: Per-row quantity to average [n].
        groups: Cluster label per row (e.g. match id) [n].
        n_boot: Bootstrap replicates.
        seed: RNG seed.
        alpha: Two-sided interval level.

    Returns:
        Tuple ``(mean, lo, hi)``. Returns ``(nan, nan, nan)`` for empty input.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    codes, uniq = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="mergesort")
    sorted_codes = codes[order]
    sorted_vals = values[order]
    starts = np.searchsorted(sorted_codes, np.arange(len(uniq)), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(len(uniq)), side="right")
    csum = np.concatenate([[0.0], np.cumsum(sorted_vals)])
    group_sums = csum[ends] - csum[starts]
    group_ns = (ends - starts).astype(float)
    rng = np.random.default_rng(seed)
    n_g = len(uniq)
    draws = rng.integers(0, n_g, size=(n_boot, n_g))
    boot_sums = group_sums[draws].sum(axis=1)
    boot_ns = group_ns[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = np.where(boot_ns > 0, boot_sums / boot_ns, np.nan)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def simulate_two_way(
    y: np.ndarray,
    p_model: np.ndarray,
    p_book_fair: np.ndarray,
    groups: np.ndarray,
    cfg: BetSimConfig,
) -> BetSimResult:
    """Simulate betting a two-way market priced off a book proxy.

    The book's fair probability ``p_book_fair`` is marked up to ``1 + hold`` on both
    sides; the model bets whichever side has positive expected value, and only when that
    EV exceeds ``cfg.edge_threshold``. Stakes are flat (one unit).

    Args:
        y: Binary outcome of the A ("over") side [n].
        p_model: Model probability of the A side [n].
        p_book_fair: Book proxy's fair probability of the A side [n].
        groups: Cluster label per row for the ROI interval [n].
        cfg: Simulation configuration.

    Returns:
        A :class:`BetSimResult`.
    """
    y = np.asarray(y, dtype=float)
    p_model = np.asarray(p_model, dtype=float)
    o_a, o_b = price_two_way(p_book_fair, cfg.hold)
    ev_a = p_model * o_a - 1.0
    ev_b = (1.0 - p_model) * o_b - 1.0
    take_a = ev_a >= ev_b
    ev = np.where(take_a, ev_a, ev_b)
    bet = ev > cfg.edge_threshold
    if bet.sum() == 0:
        return BetSimResult(len(y), 0, 0.0, float("nan"), float("nan"), float("nan"),
                            float("nan"), 0, 0.0)
    win_a = y > 0.5
    profit = np.where(
        take_a,
        np.where(win_a, o_a - 1.0, -1.0),
        np.where(~win_a, o_b - 1.0, -1.0),
    )
    pr = profit[bet]
    roi, lo, hi = clustered_bootstrap_mean(pr, np.asarray(groups)[bet], cfg.n_boot, cfg.seed)
    return BetSimResult(
        n_rows=int(len(y)),
        n_bets=int(bet.sum()),
        bet_rate=float(bet.mean()),
        mean_edge=float(ev[bet].mean()),
        roi=roi,
        roi_lo=lo,
        roi_hi=hi,
        n_over=int((bet & take_a).sum()),
        profit=float(pr.sum()),
        mean_p_model=float(p_model[bet].mean()),
        mean_p_book=float(np.asarray(p_book_fair, dtype=float)[bet].mean()),
        realized_a_rate=float(y[bet].mean()),
        mean_odds=float(np.where(take_a, o_a, o_b)[bet].mean()),
    )


# ---------------------------------------------------------------------------
# Count distributions
# ---------------------------------------------------------------------------


def nb_sf(line: float | np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """P(count > line) for a negative binomial with mean ``mu`` and dispersion ``r``.

    Variance is ``mu + mu**2 / r``; ``r -> inf`` recovers the Poisson.

    Args:
        line: Half-integer betting line, scalar or [n].
        mu: Mean count [n].
        r: Dispersion (number of failures) parameter, > 0.

    Returns:
        Survival probability [n].
    """
    from scipy.stats import nbinom, poisson

    mu = np.asarray(mu, dtype=float)
    k = np.floor(np.asarray(line, dtype=float))
    if not np.isfinite(r) or r > 1e6:
        return poisson.sf(k, mu)
    p = r / (r + mu)
    return nbinom.sf(k, r, p)


def fit_nb_dispersion(y: np.ndarray, mu: np.ndarray, grid: Iterable[float] | None = None) -> float:
    """Fit the negative-binomial dispersion by profile likelihood on a grid.

    Args:
        y: Observed counts [n].
        mu: Fitted means [n].
        grid: Candidate ``r`` values; a default log-spaced grid is used if omitted.

    Returns:
        The ``r`` maximising the NB log-likelihood (``inf`` if the Poisson wins).
    """
    from scipy.stats import nbinom, poisson

    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    cand = list(grid) if grid is not None else list(np.exp(np.linspace(np.log(0.5), np.log(500), 40)))
    best_r, best_ll = float("inf"), float(poisson.logpmf(y, mu).sum())
    for r in cand:
        p = r / (r + mu)
        ll = float(nbinom.logpmf(y, r, p).sum())
        if ll > best_ll:
            best_ll, best_r = ll, float(r)
    return best_r


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def platt_fit(p_raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a one-dimensional logistic recalibration ``logit(p) -> a + b*logit(p)``.

    Args:
        p_raw: Uncalibrated probabilities [n].
        y: Binary outcomes [n].

    Returns:
        Tuple ``(a, b)``.
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, np.asarray(y, dtype=int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def platt_apply(p_raw: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Apply a fitted Platt recalibration.

    Args:
        p_raw: Uncalibrated probabilities [n].
        ab: ``(a, b)`` from :func:`platt_fit`.

    Returns:
        Calibrated probabilities [n].
    """
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a + b * _logit(p_raw))))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerically safe logit."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def per90(count: np.ndarray, minutes: np.ndarray, floor: float = 30.0) -> np.ndarray:
    """Rate per 90 minutes with a floor on the exposure denominator.

    Args:
        count: Event counts [n].
        minutes: Minutes of exposure [n].
        floor: Minimum minutes used in the denominator.

    Returns:
        Rate per 90 minutes [n].
    """
    return np.asarray(count, dtype=float) * 90.0 / np.maximum(np.asarray(minutes, dtype=float), floor)


def tercile_edges(x: np.ndarray) -> tuple[float, float]:
    """Return the 1/3 and 2/3 quantiles of ``x`` ignoring NaNs."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Write a result table to the reports directory as Parquet.

    Args:
        df: Table to write.
        name: File stem, e.g. ``"01_cards_gate"``.

    Returns:
        Path written.
    """
    p = reports_dir() / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def disagreement_table(
    y: np.ndarray,
    p_model: np.ndarray,
    p_book: np.ndarray,
    n_bins: int = 5,
    signed: bool = True,
) -> pd.DataFrame:
    """Closing-line-value style check: who is closer to the truth where they disagree.

    Rows are binned by ``p_model - p_book`` (signed) or its absolute value, and each bin
    reports the mean of both probabilities, the observed rate and each side's log-loss.

    Args:
        y: Binary outcomes [n].
        p_model: Model probabilities [n].
        p_book: Book-proxy probabilities [n].
        n_bins: Number of equal-count bins.
        signed: Bin on the signed difference (else the absolute difference).

    Returns:
        Frame with one row per bin.
    """
    from research.privileged_tracking.common.metrics import log_loss as _ll

    y = np.asarray(y, dtype=float)
    p_model = np.asarray(p_model, dtype=float)
    p_book = np.asarray(p_book, dtype=float)
    d = p_model - p_book
    key = d if signed else np.abs(d)
    order = np.argsort(key)
    rows = []
    for i, chunk in enumerate(np.array_split(order, n_bins)):
        if len(chunk) == 0:
            continue
        rows.append({
            "bin": i,
            "n": int(len(chunk)),
            "diff_lo": float(key[chunk].min()),
            "diff_hi": float(key[chunk].max()),
            "mean_p_model": float(p_model[chunk].mean()),
            "mean_p_book": float(p_book[chunk].mean()),
            "observed": float(y[chunk].mean()),
            "log_loss_model": _ll(y[chunk], p_model[chunk]),
            "log_loss_book": _ll(y[chunk], p_book[chunk]),
        })
    out = pd.DataFrame(rows)
    out["model_better"] = out["log_loss_book"] - out["log_loss_model"]
    return out


def prior_group_daily_mean(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    value_col: str,
    has_col: str,
    prior_mean: float,
    k: float,
) -> np.ndarray:
    """Shrunk group mean over rows on strictly earlier dates.

    Aggregating to one row per (group, date) before accumulating guarantees that rows
    sharing a date never enter one another's prior, which a plain row-level expanding
    sum cannot promise.

    Args:
        df: Input frame.
        group_col: Group key (e.g. a division).
        date_col: Date column.
        value_col: Quantity to average.
        has_col: 0/1 column marking rows where ``value_col`` is observed.
        prior_mean: Backstop mean used before any history exists.
        k: Shrinkage strength in observations.

    Returns:
        Array [n] of shrunk prior means aligned to ``df``.
    """
    day = df.groupby([group_col, date_col], as_index=False).agg(
        _s=(value_col, lambda s: float(np.nansum(s))), _n=(has_col, "sum"))
    day = day.sort_values([date_col, group_col], kind="mergesort").reset_index(drop=True)
    pri = prior_expanding(day, [group_col], ["_s", "_n"], [date_col], count_name="_days",
                          prefix="ps_")
    day = pd.concat([day, pri], axis=1)
    day["_rate"] = shrunk_rate(day["ps__s"].to_numpy(), day["ps__n"].to_numpy(), prior_mean, k)
    out = df[[group_col, date_col]].merge(day[[group_col, date_col, "_rate"]],
                                          on=[group_col, date_col], how="left")
    return out["_rate"].to_numpy()


def logit_blend(p_a: np.ndarray, p_b: np.ndarray, lam: float) -> np.ndarray:
    """Interpolate two probability vectors on the logit scale.

    Args:
        p_a: Probabilities at ``lam = 0`` [n].
        p_b: Probabilities at ``lam = 1`` [n].
        lam: Blend weight in [0, 1].

    Returns:
        Blended probabilities [n].
    """
    z = (1.0 - lam) * _logit(p_a) + lam * _logit(p_b)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# The gate: generalist vs specialist, one implementation for every stage
# ---------------------------------------------------------------------------
#
# Stages 01-03 each grew a local copy of this loop before the shared version existed
# (``cards._gate``, ``corners.stage_gate``, ``pass_counts.gate``). They agree on the
# design -- generalist fitted on all training rows, specialist on the scenario's training
# rows, both scored on the *same* held-out scenario rows -- but differ in loss (binary log
# loss / Poisson deviance / negative-binomial log score) and in the sign convention of the
# reported delta, which makes their tables hard to read side by side. Stage 05 runs every
# target through the implementation below instead; see ``reports/05_gate_sweep.md``.


@dataclass(frozen=True)
class GateConfig:
    """Configuration of a generalist-vs-specialist gate run.

    Attributes:
        n_folds: Match-grouped CV folds used inside the discovery half.
        n_boot: Bootstrap replicates for the match-clustered delta interval.
        min_train_scenario: Minimum scenario training rows required to fit a specialist.
        min_eval: Minimum scored rows required to emit a result row.
        seed: Default model / fold seed.
    """

    n_folds: int = 5
    n_boot: int = 2000
    min_train_scenario: int = 200
    min_eval: int = 50
    seed: int = 20260918


def nb_log_score(y: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
    """Per-row negative log-likelihood of a count under a negative binomial, in nats.

    ``r = inf`` gives the Poisson. Keeping counts on the same nat scale as the binary
    targets' log loss is what lets one table hold both.

    Args:
        y: Observed counts [n].
        mu: Predicted means [n].
        r: Dispersion (number of failures); ``inf`` for Poisson.

    Returns:
        Per-row negative log-likelihood [n].
    """
    from scipy.stats import nbinom, poisson

    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    if not np.isfinite(r) or r > 1e6:
        return -poisson.logpmf(y, mu)
    p = r / (r + mu)
    return -nbinom.logpmf(y, r, p)


def gate_loss(kind: str, y: np.ndarray, pred: np.ndarray, r: float = float("inf")
              ) -> np.ndarray:
    """Per-row loss in nats for the gate.

    Args:
        kind: ``"binary"`` (log loss) or ``"count"`` (negative-binomial log score).
        y: Outcomes [n].
        pred: Predicted probabilities or means [n].
        r: Dispersion used for ``"count"``.

    Returns:
        Per-row loss [n].
    """
    if kind == "binary":
        from research.privileged_tracking.common.metrics import per_sample_log_loss

        return per_sample_log_loss(np.asarray(y, dtype=float), np.asarray(pred, dtype=float))
    if kind == "count":
        return nb_log_score(y, pred, r)
    raise ValueError(f"unknown kind {kind!r}")


def gate_predictions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    fit_predict,
    cfg: GateConfig,
    seed: int,
    subset: np.ndarray | None = None,
    match_col: str = "match_id",
    split_col: str = "split",
    frame_fn=None,
) -> np.ndarray:
    """Out-of-sample predictions for every row, honouring the discovery/confirmation split.

    Discovery rows are predicted by match-grouped k-fold inside the discovery half;
    confirmation rows by a single fit on all discovery rows. With ``subset`` given, only
    the training rows inside the subset are used -- that is the specialist. The *scored*
    rows are never restricted here, so a specialist can be scored off its own scenario.

    Args:
        df: Population with ``split_col`` and ``match_col``.
        feature_cols: Feature names.
        target: Target column.
        fit_predict: Callable ``(X_tr, y_tr, X_te, seed) -> pred`` supplied by the owning
            stage, so each target keeps its own model design.
        cfg: Gate configuration.
        seed: Model / fold seed.
        subset: Boolean training mask [n] (the scenario), or ``None`` for the generalist.
        match_col: Cluster / grouping column.
        split_col: Column holding ``"discovery"`` / ``"confirmation"``.
        frame_fn: Optional ``(df, cols) -> DataFrame`` hook for stages whose design frame
            needs categorical dtypes; defaults to plain column selection.

    Returns:
        Predictions [n]; ``nan`` wherever no model could be fitted.
    """
    from research.privileged_tracking.common.splits import group_kfold

    X = df[list(feature_cols)] if frame_fn is None else frame_fn(df, list(feature_cols))
    y = df[target].to_numpy(dtype=float)
    match = df[match_col].to_numpy()
    disc = (df[split_col].to_numpy() == DISCOVERY)
    out = np.full(len(df), np.nan)
    d_idx = np.flatnonzero(disc)
    for tr, te in group_kfold(match[d_idx], n_splits=cfg.n_folds, seed=seed):
        tr_i, te_i = d_idx[tr], d_idx[te]
        if subset is not None:
            tr_i = tr_i[subset[tr_i]]
        if len(tr_i) < (cfg.min_train_scenario if subset is not None else 1) or len(te_i) == 0:
            continue
        out[te_i] = fit_predict(X.iloc[tr_i], y[tr_i], X.iloc[te_i], seed)
    c_idx = np.flatnonzero(~disc)
    tr_i = d_idx if subset is None else d_idx[subset[d_idx]]
    if len(c_idx) and len(tr_i) >= (cfg.min_train_scenario if subset is not None else 1):
        out[c_idx] = fit_predict(X.iloc[tr_i], y[tr_i], X.iloc[c_idx], seed)
    return out


def gate_rows(
    df: pd.DataFrame,
    target: str,
    kind: str,
    generalist: np.ndarray,
    specialist: np.ndarray,
    scenario_mask: np.ndarray,
    cfg: GateConfig,
    seed: int,
    r: float = float("inf"),
    match_col: str = "match_id",
    split_col: str = "split",
    **extra: object,
) -> list[dict[str, object]]:
    """Score one generalist / specialist pair on the scenario and off it.

    The sign convention, used everywhere in stage 05, is
    ``delta = loss(generalist) - loss(specialist)``: **positive means the specialist is
    better**, i.e. targeting has room.

    Args:
        df: Population with ``split_col`` and ``match_col``.
        target: Target column (scored from ``df``).
        kind: ``"binary"`` or ``"count"``.
        generalist: Predictions of the model fitted on all training rows [n].
        specialist: Predictions of the model fitted on the scenario's training rows [n].
        scenario_mask: Boolean scenario mask [n].
        cfg: Gate configuration.
        seed: Seed that produced the predictions (recorded in the rows).
        r: Dispersion for count targets.
        match_col: Cluster column for the bootstrap.
        split_col: Split column.
        **extra: Extra fields copied into every row (e.g. ``target_label``).

    Returns:
        Up to four rows: {discovery_cv, confirmation} x {in_scenario, off_scenario}.
    """
    y = df[target].to_numpy(dtype=float)
    match = df[match_col].to_numpy()
    disc = (df[split_col].to_numpy() == DISCOVERY)
    ok = np.isfinite(generalist) & np.isfinite(specialist)
    rows: list[dict[str, object]] = []
    for where, wm in (("discovery_cv", disc), ("confirmation", ~disc)):
        for region, rm in (("in_scenario", scenario_mask),
                           ("off_scenario", ~scenario_mask)):
            m = wm & rm & ok
            if m.sum() < cfg.min_eval:
                continue
            lg = gate_loss(kind, y[m], generalist[m], r)
            ls = gate_loss(kind, y[m], specialist[m], r)
            d, lo, hi = clustered_bootstrap_mean(lg - ls, match[m], cfg.n_boot, seed)
            row: dict[str, object] = dict(extra)
            row.update(
                target=target, kind=kind, where=where, region=region, seed=int(seed),
                n=int(m.sum()), n_matches=int(pd.unique(match[m]).size),
                mean_y=float(y[m].mean()),
                loss_generalist=float(lg.mean()), loss_specialist=float(ls.mean()),
                delta_gen_minus_spec=float(d), ci_lo=float(lo), ci_hi=float(hi),
                mae_generalist=float(np.abs(y[m] - generalist[m]).mean()),
                mae_specialist=float(np.abs(y[m] - specialist[m]).mean()),
            )
            rows.append(row)
    return rows


def run_gate(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    kind: str,
    scenarios: dict[str, np.ndarray],
    fit_predict,
    cfg: GateConfig,
    seeds: Sequence[int] = (),
    extra_seed_scenarios: Sequence[str] = (),
    match_col: str = "match_id",
    split_col: str = "split",
    frame_fn=None,
    descriptions: dict[str, str] | None = None,
    **extra: object,
) -> pd.DataFrame:
    """Run the gate for one target over several scenarios, sharing the generalist fits.

    The generalist does not depend on the scenario, so it is fitted once per seed and
    reused; only the specialist is refitted per scenario. The first seed is the headline;
    the remaining seeds are run on ``extra_seed_scenarios`` only and exist to measure the
    refit-noise floor, so that deltas smaller than the noise are not oversold.

    Args:
        df: Population with ``split_col`` and ``match_col``.
        feature_cols: Feature names.
        target: Target column.
        kind: ``"binary"`` or ``"count"``.
        scenarios: Map from scenario name to boolean mask [n].
        fit_predict: Callable ``(X_tr, y_tr, X_te, seed) -> pred``.
        cfg: Gate configuration.
        seeds: Seeds to run; defaults to ``(cfg.seed,)``.
        extra_seed_scenarios: Scenario names re-run under the non-headline seeds.
        match_col: Cluster column.
        split_col: Split column.
        frame_fn: Optional design-frame hook (see :func:`gate_predictions`).
        descriptions: Optional human-readable scenario definitions.
        **extra: Extra fields copied into every row.

    Returns:
        One row per (scenario, seed, where, region).
    """
    seeds = tuple(seeds) if len(tuple(seeds)) else (cfg.seed,)
    y = df[target].to_numpy(dtype=float)
    disc = (df[split_col].to_numpy() == DISCOVERY)
    rows: list[dict[str, object]] = []
    for si, seed in enumerate(seeds):
        gen = gate_predictions(df, feature_cols, target, fit_predict, cfg, seed,
                               None, match_col, split_col, frame_fn)
        r = float("inf")
        if kind == "count":
            fin = disc & np.isfinite(gen)
            r = fit_nb_dispersion(y[fin], gen[fin]) if fin.sum() > 50 else float("inf")
        for name, mask in scenarios.items():
            if si > 0 and name not in set(extra_seed_scenarios):
                continue
            mask = np.asarray(mask, dtype=bool)
            spe = gate_predictions(df, feature_cols, target, fit_predict, cfg, seed,
                                   mask, match_col, split_col, frame_fn)
            rows += gate_rows(
                df, target, kind, gen, spe, mask, cfg, seed, r, match_col, split_col,
                scenario=name,
                description=(descriptions or {}).get(name, ""),
                n_train_scenario=int((disc & mask).sum()),
                share_of_rows=float(mask.mean()),
                dispersion_r=r,
                **extra,
            )
    return pd.DataFrame(rows)


def summarise_gate(rows: pd.DataFrame, headline_seed: int | None = None) -> pd.DataFrame:
    """Collapse a :func:`run_gate` table to one row per (target, scenario, where, region).

    The headline delta and interval come from the headline seed; ``refit_spread`` is the
    range of the delta across all seeds that ran, and ``verdict`` compares the headline
    interval with zero and with that spread.

    Args:
        rows: Output of :func:`run_gate` (possibly concatenated over targets).
        headline_seed: Seed treated as the headline; defaults to the first seen.

    Returns:
        Summary table with ``verdict`` in {``room``, ``no room``, ``inconclusive``}.
    """
    if rows.empty:
        return rows
    keys = [c for c in ("target_label", "target", "scenario", "where", "region")
            if c in rows.columns]
    out: list[dict[str, object]] = []
    for key, g in rows.groupby(keys, sort=False):
        hs = headline_seed if headline_seed is not None else int(g["seed"].iloc[0])
        h = g[g["seed"] == hs]
        if h.empty:
            h = g.iloc[[0]]
        h0 = h.iloc[0]
        n_seeds = int(g["seed"].nunique())
        spread = (float(g["delta_gen_minus_spec"].max() - g["delta_gen_minus_spec"].min())
                  if n_seeds > 1 else float("nan"))
        d, lo, hi = (float(h0["delta_gen_minus_spec"]), float(h0["ci_lo"]), float(h0["ci_hi"]))
        floor = 0.0 if not np.isfinite(spread) else spread
        if hi < 0:
            verdict = "no room"
        elif lo > 0 and d > floor:
            verdict = "room"
        elif lo > 0:
            verdict = "inconclusive (delta inside refit noise)"
        else:
            verdict = "inconclusive (interval spans zero)"
        row = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,))}
        row.update(
            n=int(h0["n"]), n_matches=int(h0["n_matches"]),
            n_train_scenario=int(h0.get("n_train_scenario", -1)),
            mean_y=float(h0["mean_y"]), kind=str(h0["kind"]),
            loss_generalist=float(h0["loss_generalist"]),
            loss_specialist=float(h0["loss_specialist"]),
            delta_gen_minus_spec=d, ci_lo=lo, ci_hi=hi,
            n_seeds=n_seeds, refit_spread=spread,
            verdict=verdict, description=str(h0.get("description", "")),
        )
        out.append(row)
    return pd.DataFrame(out)
