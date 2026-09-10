"""Statistics and pricing for the cross-game correlation backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from geo_model.parlay.pairs import cover_indicators


def parlay_ev(rho: float, leg_decimal: float, p1: float = 0.5, p2: float = 0.5) -> float:
    """Expected profit per unit staked on a 2-leg parlay with correlated legs.

    ``P(both) = p1*p2 + rho*sqrt(p1(1-p1)p2(1-p2))`` (phi-coefficient form).

    Args:
        rho: Phi correlation between the two leg-win indicators.
        leg_decimal: Decimal odds per leg (parlay pays the product).
        p1: True win probability of leg 1.
        p2: True win probability of leg 2.

    Returns:
        Expected profit per 1 unit staked.
    """
    p_both = p1 * p2 + rho * np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    return float(p_both * leg_decimal**2 - 1.0)


def breakeven_correlation(leg_decimal: float, p1: float = 0.5, p2: float = 0.5) -> float:
    """Phi correlation at which a 2-leg parlay has zero expected profit.

    Args:
        leg_decimal: Decimal odds per leg.
        p1: True win probability of leg 1.
        p2: True win probability of leg 2.

    Returns:
        Break-even phi correlation (about ``0.097`` at ``-110`` and 50/50 legs).
    """
    return float((1.0 / leg_decimal**2 - p1 * p2) / np.sqrt(p1 * (1 - p1) * p2 * (1 - p2)))


def gaussian_explaining_away_corr(prior_std: float, obs_std: float) -> tuple[float, float]:
    """Theoretical correlations after one shared game in the Gaussian model.

    Four teams with independent market errors ``~ N(0, s²)``. H plays A and the
    residual ``e_H - e_A + n`` (``n ~ N(0, σ²)``) is observed. An efficient
    market moves both teams' next lines to the posterior means, leaving
    posterior covariance ``Cov(e_H, e_A | r) = s⁴ / (2s² + σ²)`` and variance
    ``Var(e_H | r) = s² - s⁴ / (2s² + σ²)``. H then plays C and A plays D, so
    each next-game residual is the team's posterior error minus an unseen
    opponent error plus fresh noise:

        corr = [s⁴/(2s²+σ²)] / [2s² + σ² - s⁴/(2s²+σ²)]

    Args:
        prior_std: Std of a team's market error, ``s``.
        obs_std: Std of single-game noise, ``σ``.

    Returns:
        ``(latent_posterior_corr, next_game_residual_corr)``.
    """
    s2, o2 = prior_std**2, obs_std**2
    cov = s2 * s2 / (2 * s2 + o2)
    latent = cov / (s2 - cov)
    nxt = cov / (2 * s2 + o2 - cov)
    return float(latent), float(nxt)


@dataclass
class CorrelationSummary:
    """Correlation diagnostics for one leg-pair sample.

    Attributes:
        n: Pairs used for continuous statistics.
        pearson: Pearson correlation of the two residuals.
        pearson_ci: 95% bootstrap CI.
        pearson_p: Two-sided p-value.
        spearman: Rank correlation.
        n_no_push: Pairs used for cover statistics.
        phi: Phi coefficient between the two cover indicators.
        phi_ci: 95% bootstrap CI for phi.
        p_both_cover: Empirical joint cover frequency.
        p_both_indep: Product of marginal cover frequencies.
    """

    n: int
    pearson: float
    pearson_ci: tuple[float, float]
    pearson_p: float
    spearman: float
    n_no_push: int
    phi: float
    phi_ci: tuple[float, float]
    p_both_cover: float
    p_both_indep: float

    def to_row(self) -> dict[str, float | int | str]:
        """Flatten for tabular reporting."""
        return {
            "n": self.n,
            "pearson": round(self.pearson, 4),
            "pearson_ci": f"[{self.pearson_ci[0]:+.3f}, {self.pearson_ci[1]:+.3f}]",
            "p": round(self.pearson_p, 3),
            "spearman": round(self.spearman, 4),
            "phi": round(self.phi, 4),
            "phi_ci": f"[{self.phi_ci[0]:+.3f}, {self.phi_ci[1]:+.3f}]",
            "p_both": round(self.p_both_cover, 4),
            "p_indep": round(self.p_both_indep, 4),
        }


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        xs, ys = x[ix], y[ix]
        if xs.std() == 0 or ys.std() == 0:
            vals[b] = 0.0
        else:
            vals[b] = np.corrcoef(xs, ys)[0, 1]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def correlation_summary(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> CorrelationSummary:
    """Compute :class:`CorrelationSummary` for two aligned residual series.

    Args:
        x: Leg-1 residuals.
        y: Leg-2 residuals.
        n_boot: Bootstrap resamples for confidence intervals.
        seed: RNG seed.

    Returns:
        Summary of continuous and binary correlations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        nan = float("nan")
        return CorrelationSummary(len(x), nan, (nan, nan), nan, nan, 0, nan, (nan, nan), nan, nan)
    pr = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    cx, cy = cover_indicators(x, y)
    if len(cx) >= 3 and cx.std() > 0 and cy.std() > 0:
        phi = float(np.corrcoef(cx, cy)[0, 1])
    else:
        phi = float("nan")
    return CorrelationSummary(
        n=int(len(x)),
        pearson=float(pr.statistic),
        pearson_ci=_bootstrap_corr(x, y, n_boot, seed),
        pearson_p=float(pr.pvalue),
        spearman=float(sp.statistic),
        n_no_push=int(len(cx)),
        phi=phi,
        phi_ci=_bootstrap_corr(cx.astype(float), cy.astype(float), n_boot, seed + 1),
        p_both_cover=float(np.mean((cx == 1) & (cy == 1))) if len(cx) else float("nan"),
        p_both_indep=float(cx.mean() * cy.mean()) if len(cx) else float("nan"),
    )


@dataclass
class ParlayRoi:
    """Realized result of a mechanical two-sided parlay strategy.

    Attributes:
        n_pairs: Pairs bet (each pair places two 1-unit parlays).
        staked: Units staked (excluding refunded pushes).
        profit: Net units won.
        roi: ``profit / staked``.
        hit_rate: Fraction of resolved pairs where one of the two parlays won.
    """

    n_pairs: int
    staked: float
    profit: float
    roi: float
    hit_rate: float


def two_sided_parlay_roi(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    leg_decimal: float,
    expected_sign: int = 1,
) -> ParlayRoi:
    """Bet both same-sign (or both opposite-sign) parlays on every pair.

    With ``expected_sign=+1`` each pair gets a 1-unit parlay on (leg1 cover,
    leg2 cover) and a 1-unit parlay on (leg1 fail, leg2 fail); exactly one wins
    when the legs land on the same side. With ``-1`` the mixed-sign parlays are
    bet instead. Any push refunds both parlays.

    Args:
        x: Leg-1 residuals.
        y: Leg-2 residuals.
        leg_decimal: Decimal odds per leg.
        expected_sign: ``+1`` to bet on positive correlation, ``-1`` negative.

    Returns:
        :class:`ParlayRoi` for the strategy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    x, y = x[mask], y[mask]
    same = np.sign(x) == np.sign(y)
    win = same if expected_sign > 0 else ~same
    payout = leg_decimal**2
    profit = float(win.sum() * payout - 2 * len(x))
    staked = float(2 * len(x))
    return ParlayRoi(
        n_pairs=int(len(x)),
        staked=staked,
        profit=profit,
        roi=profit / staked if staked else float("nan"),
        hit_rate=float(win.mean()) if len(x) else float("nan"),
    )


def kalman_pair_calibration(pairs: pd.DataFrame, n_quantiles: int = 10) -> pd.DataFrame:
    """Compare predicted residual covariance with realized residual products.

    Args:
        pairs: ``KalmanOutput.pairs`` (same-slate game pairs with ``pred_cov``,
            ``pred_corr``, ``resid_1``, ``resid_2``).
        n_quantiles: Number of predicted-correlation bins.

    Returns:
        One row per bin with mean predicted correlation, realized Pearson
        correlation of the two residuals, phi of the cover indicators, and n.
    """
    df = pairs.copy()
    df["bin"] = pd.qcut(df["pred_corr"], q=n_quantiles, labels=False, duplicates="drop")
    rows = []
    for b, g in df.groupby("bin", sort=True):
        s = correlation_summary(g["resid_1"], g["resid_2"], n_boot=200)
        rows.append(
            {
                "bin": int(b),
                "n": len(g),
                "pred_corr_mean": float(g["pred_corr"].mean()),
                "realized_pearson": s.pearson,
                "realized_phi": s.phi,
                "phi_ci_lo": s.phi_ci[0],
                "phi_ci_hi": s.phi_ci[1],
            }
        )
    return pd.DataFrame(rows)


def kalman_slope(pairs: pd.DataFrame) -> tuple[float, float, float]:
    """OLS slope of realized residual product on predicted covariance.

    Under a correctly specified model the slope is 1. A slope near 0 means the
    predicted covariances carry no realized signal.

    Args:
        pairs: ``KalmanOutput.pairs``.

    Returns:
        ``(slope, stderr, p_value)``.
    """
    y = (pairs["resid_1"] * pairs["resid_2"]).to_numpy(dtype=float)
    x = pairs["pred_cov"].to_numpy(dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    res = stats.linregress(x, y)
    return float(res.slope), float(res.stderr), float(res.pvalue)
