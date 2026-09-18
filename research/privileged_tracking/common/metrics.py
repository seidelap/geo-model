"""Metrics shared by the NFL and soccer experiments.

All functions accept plain NumPy arrays so they can be unit-tested without data.
"""
from __future__ import annotations

import numpy as np


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    """Mean binary log-loss with probability clipping."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between outcome and probability."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def skill_score(score: float, reference: float) -> float:
    """Fractional improvement of a lower-is-better score over a reference (1 = perfect)."""
    if reference == 0:
        return float("nan")
    return float(1.0 - score / reference)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Coefficient of determination."""
    y = np.asarray(y, dtype=float)
    ss_res = float(np.sum((y - np.asarray(yhat, dtype=float)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float))))


def paired_bootstrap_delta(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 2000,
                           seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for mean(loss_a - loss_b) over paired per-sample losses.

    Positive values mean ``b`` is better (lower loss) than ``a``.

    Returns:
        ``(mean_delta, ci_low, ci_high)`` at 95%.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def per_sample_log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Element-wise binary log-loss, for paired comparisons."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-count reliability bins: mean predicted vs observed rate and count."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        rows.append({"pred": float(p[chunk].mean()), "obs": float(y[chunk].mean()),
                     "n": float(len(chunk))})
    return rows
