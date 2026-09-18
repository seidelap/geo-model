"""Leakage-safe split helpers.

Every evaluation in this research is grouped by game / match: no play or event from a
held-out game may appear in training, and any rolling "tendency" feature must be
computed from strictly earlier games.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def group_kfold(groups: pd.Series | np.ndarray, n_splits: int = 5,
                seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` positional indices with whole groups held out.

    Groups are shuffled once with ``seed`` and dealt round-robin into folds so fold
    sizes are balanced in number of groups.
    """
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {u: i % n_splits for i, u in enumerate(uniq)}
    folds = np.array([fold_of[x] for x in g])
    for k in range(n_splits):
        test = np.where(folds == k)[0]
        train = np.where(folds != k)[0]
        yield train, test


def forward_split(order: pd.Series | np.ndarray, test_frac: float = 0.3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split: the last ``test_frac`` of distinct ``order`` values is test.

    ``order`` is any sortable key (week number, date, match_week).
    """
    o = np.asarray(order)
    uniq = np.sort(np.unique(o))
    cut = uniq[int(np.floor(len(uniq) * (1 - test_frac)))]
    test = np.where(o >= cut)[0]
    train = np.where(o < cut)[0]
    return train, test


def out_of_fold_predictions(fit_predict, X: pd.DataFrame, y: np.ndarray,
                            groups: pd.Series | np.ndarray, n_splits: int = 5,
                            seed: int = 0) -> np.ndarray:
    """Return out-of-fold predictions for every row using ``group_kfold``.

    Args:
        fit_predict: callable ``(X_train, y_train, X_test) -> predictions`` for X_test.
        X: feature frame ``[n, d]``.
        y: targets ``[n]``.
        groups: group id per row ``[n]``.
    """
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in group_kfold(groups, n_splits=n_splits, seed=seed):
        oof[te] = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
    return oof
