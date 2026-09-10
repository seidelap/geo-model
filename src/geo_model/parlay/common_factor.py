"""Shared (league-wide) latent factors in NFL closing-line residuals.

Every game on a slate shares quantities the market estimates with error: the
league scoring environment (rule changes, officiating points of emphasis,
ball/weather season effects; hits totals) and the size of home-field advantage
(hits spreads; e.g. the 2020 no-fans season). Unlike a team-level error, a
shared-factor error is not diluted by opponent noise: it induces the same
correlation across every game on the slate at once and accumulates across
weeks. Books price same-week games in different matchups as independent, so a
large enough shared-factor error would make same-week parlays +EV.

This module measures how large such factors are *relative to the closing line*:

* one-way random-effects intraclass correlation (ICC) of residuals within a
  slate, with a permutation test that shuffles slate labels within season and
  a slate-cluster bootstrap;
* strictly causal persistence of the slate-mean residual (does the mean over
  the previous ``k`` weeks predict this week's residuals?);
* a scalar-state Kalman filter on the league offset (AR(1) with process noise,
  a separate transition at season boundaries), whose prior variance at each
  slate is the model's predicted same-week pair covariance, plus a profile
  likelihood over the stationary size of the factor (an upper bound on how
  large a shared factor the data allow);
* parlay implications: same-sign rates, two-sided parlay ROI, and a directional
  slate strategy, with confidence intervals that resample whole slates because
  same-week pairs share games.

Residual conventions follow :mod:`geo_model.parlay.data`: ``tresid`` is
``total - total_line`` (positive = over), ``resid`` is ``result - spread_line``
(positive = home covered).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from geo_model.parlay.backtest import ParlayRoi, two_sided_parlay_roi
from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import fetch_games_csv

CONTEXT_COLUMNS = [
    "game_id",
    "weekday",
    "gametime",
    "roof",
    "surface",
    "temp",
    "wind",
    "referee",
    "over_odds",
    "under_odds",
    "div_game",
]

CONTEXT_PARQUET_NAME = "nfl_games_context.parquet"


@dataclass(frozen=True)
class RuleChange:
    """A season whose first slate follows a known league-wide environment change.

    Attributes:
        season: First season played under the new environment.
        label: Short name used in report tables.
        reason: Why the market might misprice the shared factor that season.
    """

    season: int
    label: str
    reason: str


DEFAULT_RULE_CHANGES: tuple[RuleChange, ...] = (
    RuleChange(2004, "2004 illegal-contact emphasis", "officiating point of emphasis; passing surge (totals)"),
    RuleChange(2011, "2011 kickoff to 35", "kickoffs moved to the 35, fewer returns; post-lockout offseason (totals)"),
    RuleChange(2015, "2015 PAT from 15", "extra point moved to the 15-yard line; missed PATs (totals)"),
    RuleChange(2018, "2018 helmet/RTP emphasis", "helmet rule and roughing-the-passer emphasis; record scoring (totals)"),
    RuleChange(2020, "2020 no fans", "COVID season with empty or reduced stadiums (home-field advantage)"),
    RuleChange(2021, "2021 17 games", "17-game schedule, first full season back with fans (both)"),
    RuleChange(2024, "2024 dynamic kickoff", "new kickoff formation and touchback rules (totals)"),
)


@dataclass(frozen=True)
class CommonFactorConfig:
    """Settings for the shared-factor analysis.

    Attributes:
        train_seasons_end: Last season used to fit Kalman hyperparameters; all
            evaluation statistics that depend on fitted parameters use seasons
            strictly after this.
        persistence_windows: Look-back windows ``k`` (in weeks) for the causal
            persistence regressions.
        directional_window: Pre-registered ``k`` for the directional strategy.
        early_weeks: Weeks ``1..early_weeks`` form the "early season" stratum.
        n_permutations: Label permutations for the ICC test.
        n_boot: Cluster-bootstrap resamples for confidence intervals.
        seed: RNG seed.
        rule_changes: Seasons following known environment changes (stratum list).
        kickoff_windows: ``(label, start_hour, end_hour)`` in Eastern time,
            half-open ``[start, end)``.
        min_group_size: Slates with fewer games than this are dropped from ICC
            computations (a slate of one game carries no pair information).
    """

    train_seasons_end: int = 2009
    persistence_windows: tuple[int, ...] = (1, 2, 4, 8)
    directional_window: int = 4
    early_weeks: int = 4
    n_permutations: int = 1000
    n_boot: int = 1000
    seed: int = 0
    rule_changes: tuple[RuleChange, ...] = DEFAULT_RULE_CHANGES
    kickoff_windows: tuple[tuple[str, int, int], ...] = (
        ("early", 0, 16),
        ("late", 16, 20),
        ("night", 20, 24),
    )
    min_group_size: int = 2


# ---------------------------------------------------------------------------
# Context columns (kickoff time, weekday, weather) from the nflverse games file
# ---------------------------------------------------------------------------


def context_parquet_path(config: ParlayConfig) -> Path:
    """Location of the cached context sidecar table."""
    return config.data_dir / CONTEXT_PARQUET_NAME


def load_game_context(config: ParlayConfig, refresh: bool = False) -> pd.DataFrame:
    """Load per-game context columns (kickoff time, weekday, roof, weather, odds).

    The main games cache in :mod:`geo_model.parlay.data` does not keep these
    columns, so they are cached in a separate parquet sidecar keyed by ``game_id``.

    Args:
        config: Data directory and source URL.
        refresh: Re-download even if the sidecar exists.

    Returns:
        Table with :data:`CONTEXT_COLUMNS` (one row per game in the source file).
    """
    path = context_parquet_path(config)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    raw = fetch_games_csv(config.games_url)
    ctx = raw.loc[:, [c for c in CONTEXT_COLUMNS if c in raw.columns]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx.to_parquet(path, index=False)
    return ctx


def kickoff_window(gametime: pd.Series, windows: tuple[tuple[str, int, int], ...]) -> pd.Series:
    """Map ``"HH:MM"`` Eastern kickoff strings to window labels.

    Args:
        gametime: Kickoff time strings (NaN allowed).
        windows: ``(label, start_hour, end_hour)`` half-open hour ranges.

    Returns:
        String labels aligned with ``gametime``; NaN where the time is missing.
    """
    hour = pd.to_numeric(gametime.astype("string").str.slice(0, 2), errors="coerce")
    out = pd.Series(pd.NA, index=gametime.index, dtype="string")
    for label, lo, hi in windows:
        out[(hour >= lo) & (hour < hi)] = label
    return out


def add_context(games: pd.DataFrame, context: pd.DataFrame | None, config: CommonFactorConfig) -> pd.DataFrame:
    """Attach context columns and a ``kickoff_window`` label to cleaned games.

    Args:
        games: Output of :func:`geo_model.parlay.data.clean_games`.
        context: Output of :func:`load_game_context`, or ``None`` to skip.
        config: Provides the kickoff-window definition.

    Returns:
        Copy of ``games`` with ``gameday_str`` (date only) and, when context is
        given, the context columns plus ``kickoff_window``.
    """
    out = games.copy()
    out["gameday_str"] = out["gameday"].dt.strftime("%Y-%m-%d")
    if context is None:
        return out
    ctx = context.drop_duplicates("game_id").set_index("game_id")
    cols = [c for c in ctx.columns if c not in out.columns]
    out = out.join(ctx[cols], on="game_id")
    if "gametime" in out.columns:
        out["kickoff_window"] = kickoff_window(out["gametime"], config.kickoff_windows)
    return out


# ---------------------------------------------------------------------------
# Intraclass correlation
# ---------------------------------------------------------------------------


@dataclass
class IccResult:
    """One-way random-effects ICC of a residual within slates.

    Attributes:
        n: Games used.
        n_groups: Slates used.
        icc: ANOVA estimator ``(MSB - MSW) / (MSB + (k0 - 1) MSW)``. Under the
            random-effects model this is the correlation between two games on
            the same slate.
        pairwise_corr: Pearson correlation over all within-slate pairs (both
            orderings); an alternative estimator of the same quantity.
        n_pairs: Within-slate pairs.
        msb: Mean square between slates.
        msw: Mean square within slates.
        f_stat: ``MSB / MSW``.
        f_p: Parametric F-test p-value (one-sided, positive clustering).
        perm_mean: Mean ICC under within-season label permutation.
        perm_sd: Std of the permutation ICC.
        perm_p: One-sided permutation p-value ``P(ICC_perm >= ICC_obs)``.
        perm_p_two: Two-sided permutation p-value.
        boot_ci: 95% slate-cluster bootstrap CI for ``icc``.
    """

    n: int
    n_groups: int
    icc: float
    pairwise_corr: float
    n_pairs: int
    msb: float
    msw: float
    f_stat: float
    f_p: float
    perm_mean: float = float("nan")
    perm_sd: float = float("nan")
    perm_p: float = float("nan")
    perm_p_two: float = float("nan")
    boot_ci: tuple[float, float] = (float("nan"), float("nan"))

    def to_row(self) -> dict[str, float | int | str]:
        """Flatten for tabular reporting."""
        return {
            "n": self.n,
            "groups": self.n_groups,
            "icc": round(float(self.icc), 4),
            "icc_ci": f"[{self.boot_ci[0]:+.4f}, {self.boot_ci[1]:+.4f}]",
            "pairwise": round(float(self.pairwise_corr), 4),
            "F": round(float(self.f_stat), 3),
            "perm_mean": round(float(self.perm_mean), 4),
            "perm_sd": round(float(self.perm_sd), 4),
            "perm_p": round(float(self.perm_p), 3),
            "perm_p_two": round(float(self.perm_p_two), 3),
        }


def group_codes(df: pd.DataFrame, group_cols: tuple[str, ...] | list[str]) -> np.ndarray:
    """Dense integer codes for the combination of ``group_cols``.

    Args:
        df: Table with the grouping columns.
        group_cols: Columns whose combination defines a slate.

    Returns:
        Int array ``[n]`` with codes ``0..G-1``.
    """
    key = pd.MultiIndex.from_frame(df.loc[:, list(group_cols)])
    codes, _ = pd.factorize(key, sort=True)
    return codes.astype(int)


def _icc_core(values: np.ndarray, codes: np.ndarray) -> tuple[float, float, int, float, float, int, int]:
    """ICC statistics from values and dense group codes.

    Returns:
        ``(icc, pairwise_corr, n_pairs, msb, msw, n, n_groups)``.
    """
    n = len(values)
    n_groups = int(codes.max()) + 1 if n else 0
    if n_groups < 2 or n <= n_groups:
        nan = float("nan")
        return nan, nan, 0, nan, nan, n, n_groups
    cnt = np.bincount(codes, minlength=n_groups).astype(float)
    s = np.bincount(codes, weights=values, minlength=n_groups)
    s2 = np.bincount(codes, weights=values * values, minlength=n_groups)
    grand = values.mean()
    gm = s / cnt
    ssb = float(np.sum(cnt * (gm - grand) ** 2))
    ssw = float(np.sum(s2 - cnt * gm * gm))
    msb = ssb / (n_groups - 1)
    msw = ssw / (n - n_groups)
    k0 = (n - np.sum(cnt * cnt) / n) / (n_groups - 1)
    denom = msb + (k0 - 1) * msw
    icc = (msb - msw) / denom if denom > 0 else float("nan")
    # Pairwise estimator: every within-group pair, both orderings.
    pair_prod = float(np.sum((s * s - s2) / 2.0))
    n_pairs = int(np.sum(cnt * (cnt - 1) / 2.0))
    w = (cnt - 1.0)[codes]
    wsum = w.sum()
    if n_pairs == 0 or wsum <= 0:
        return icc, float("nan"), 0, msb, msw, n, n_groups
    mu = float(np.sum(w * values) / wsum)
    var = float(np.sum(w * (values - mu) ** 2) / wsum)
    pairwise = (pair_prod / n_pairs - mu * mu) / var if var > 0 else float("nan")
    return icc, pairwise, n_pairs, msb, msw, n, n_groups


def _restrict_min_size(values: np.ndarray, codes: np.ndarray, min_size: int) -> tuple[np.ndarray, np.ndarray]:
    if min_size <= 1 or len(values) == 0:
        return values, codes
    cnt = np.bincount(codes)
    keep = cnt[codes] >= min_size
    values = values[keep]
    codes = pd.factorize(codes[keep], sort=True)[0]
    return values, codes


def icc_oneway(values: np.ndarray | pd.Series, codes: np.ndarray, min_group_size: int = 2) -> IccResult:
    """One-way random-effects ICC without permutation or bootstrap fields.

    Args:
        values: Residuals ``[n]``.
        codes: Dense group codes ``[n]`` (see :func:`group_codes`).
        min_group_size: Drop groups smaller than this.

    Returns:
        :class:`IccResult` with permutation fields left as NaN.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(codes, dtype=int)
    ok = np.isfinite(v)
    v, c = v[ok], pd.factorize(c[ok], sort=True)[0]
    v, c = _restrict_min_size(v, c, min_group_size)
    icc, pw, n_pairs, msb, msw, n, g = _icc_core(v, c)
    if np.isfinite(msb) and msw > 0:
        f_stat = msb / msw
        f_p = float(stats.f.sf(f_stat, g - 1, n - g))
    else:
        f_stat, f_p = float("nan"), float("nan")
    return IccResult(n, g, icc, pw, n_pairs, msb, msw, f_stat, f_p)


def icc_with_tests(
    df: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] | list[str],
    within_col: str = "season",
    n_permutations: int = 1000,
    n_boot: int = 1000,
    seed: int = 0,
    min_group_size: int = 2,
) -> IccResult:
    """ICC within slates plus a within-season permutation test and cluster bootstrap.

    The permutation shuffles slate labels among the games of the same
    ``within_col`` (season), so the null keeps any season-level shared factor
    and tests only for clustering *within* the season at the slate level. The
    bootstrap resamples whole slates.

    Args:
        df: Games with ``value_col``, ``group_cols`` and ``within_col``.
        value_col: Residual column.
        group_cols: Columns defining a slate.
        within_col: Column within which labels are exchangeable under the null.
        n_permutations: Number of label permutations.
        n_boot: Number of slate-cluster bootstrap resamples.
        seed: RNG seed.
        min_group_size: Drop slates smaller than this.

    Returns:
        :class:`IccResult` with all fields populated.
    """
    d = df.loc[df[value_col].notna(), list(dict.fromkeys([*group_cols, within_col, value_col]))].copy()
    d = d.dropna(subset=list(group_cols))
    v = d[value_col].to_numpy(dtype=float)
    c = group_codes(d, group_cols)
    v, c = _restrict_min_size(v, c, min_group_size)
    keep_mask = np.ones(len(d), dtype=bool)
    if len(v) != len(d):  # rebuild the within labels after dropping small slates
        cnt = np.bincount(group_codes(d, group_cols))
        keep_mask = cnt[group_codes(d, group_cols)] >= min_group_size
    within = pd.factorize(d.loc[keep_mask, within_col], sort=True)[0]
    base = icc_oneway(v, c, min_group_size=1)
    if not np.isfinite(base.icc):
        return base
    rng = np.random.default_rng(seed)

    # Permutation: shuffle values within each season block.
    blocks = [np.flatnonzero(within == s) for s in range(within.max() + 1)]
    perm_vals = np.empty(n_permutations)
    vp = v.copy()
    for b in range(n_permutations):
        for ix in blocks:
            vp[ix] = v[rng.permutation(ix)]
        perm_vals[b] = _icc_core(vp, c)[0]
    perm_mean = float(perm_vals.mean())
    perm_sd = float(perm_vals.std(ddof=1)) if n_permutations > 1 else float("nan")
    perm_p = float((1 + np.sum(perm_vals >= base.icc)) / (1 + n_permutations))
    perm_p_two = float((1 + np.sum(np.abs(perm_vals - perm_mean) >= abs(base.icc - perm_mean))) / (1 + n_permutations))

    # Slate-cluster bootstrap.
    n_groups = int(c.max()) + 1
    members = [np.flatnonzero(c == g) for g in range(n_groups)]
    sizes = np.array([len(m) for m in members])
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n_groups, n_groups)
        bv = np.concatenate([v[members[g]] for g in draw])
        bc = np.repeat(np.arange(n_groups), sizes[draw])
        boot_vals[b] = _icc_core(bv, bc)[0]
    ci = (float(np.nanquantile(boot_vals, 0.025)), float(np.nanquantile(boot_vals, 0.975)))
    return IccResult(
        n=base.n,
        n_groups=base.n_groups,
        icc=base.icc,
        pairwise_corr=base.pairwise_corr,
        n_pairs=base.n_pairs,
        msb=base.msb,
        msw=base.msw,
        f_stat=base.f_stat,
        f_p=base.f_p,
        perm_mean=perm_mean,
        perm_sd=perm_sd,
        perm_p=perm_p,
        perm_p_two=perm_p_two,
        boot_ci=ci,
    )


# ---------------------------------------------------------------------------
# Same-slate pairs and parlay statistics
# ---------------------------------------------------------------------------


def same_slate_pairs(
    games: pd.DataFrame,
    value_col: str,
    group_cols: tuple[str, ...] | list[str] = ("season", "week"),
) -> pd.DataFrame:
    """All unordered pairs of games on the same slate.

    Args:
        games: Games with ``game_id``, ``value_col`` and ``group_cols``.
        value_col: Residual column used for both legs.
        group_cols: Columns defining a slate.

    Returns:
        Table with the slate columns, ``game_id_1, game_id_2, x, y`` (residuals
        of the two legs) and ``slate`` (dense slate code).
    """
    d = games.loc[games[value_col].notna()].dropna(subset=list(group_cols))
    codes = group_codes(d, group_cols)
    vals = d[value_col].to_numpy(dtype=float)
    gids = d["game_id"].to_numpy()
    keys = d.loc[:, list(group_cols)].to_numpy()
    rows: list[np.ndarray] = []
    for g in range(codes.max() + 1 if len(codes) else 0):
        ix = np.flatnonzero(codes == g)
        if len(ix) < 2:
            continue
        i, j = np.triu_indices(len(ix), k=1)
        a, b = ix[i], ix[j]
        block = np.column_stack([np.full(len(a), g), a, b])
        rows.append(block)
    if not rows:
        cols = [*group_cols, "slate", "game_id_1", "game_id_2", "x", "y"]
        return pd.DataFrame(columns=cols)
    m = np.vstack(rows)
    out = pd.DataFrame(keys[m[:, 1]], columns=list(group_cols))
    out["slate"] = m[:, 0]
    out["game_id_1"] = gids[m[:, 1]]
    out["game_id_2"] = gids[m[:, 2]]
    out["x"] = vals[m[:, 1]]
    out["y"] = vals[m[:, 2]]
    return out


@dataclass
class PairStats:
    """Correlation and parlay statistics over same-slate pairs.

    Same-slate pairs share games, so all confidence intervals resample whole
    slates (cluster bootstrap) rather than pairs.

    Attributes:
        n_pairs: Pairs with both legs observed.
        n_slates: Slates contributing pairs.
        pairs_per_slate: Mean pairs per slate.
        pearson: Symmetric Pearson correlation of the two residuals.
        pearson_ci: 95% slate-bootstrap CI.
        n_no_push: Pairs where neither leg pushed.
        phi: Symmetric phi coefficient of the two cover indicators.
        phi_ci: 95% slate-bootstrap CI.
        same_sign: Fraction of no-push pairs landing on the same side.
        same_sign_ci: 95% slate-bootstrap CI.
        two_sided: Mechanical two-sided parlay result (bet both same-sign parlays).
    """

    n_pairs: int
    n_slates: int
    pairs_per_slate: float
    pearson: float
    pearson_ci: tuple[float, float]
    n_no_push: int
    phi: float
    phi_ci: tuple[float, float]
    same_sign: float
    same_sign_ci: tuple[float, float]
    two_sided: ParlayRoi

    def to_row(self) -> dict[str, float | int | str]:
        """Flatten for tabular reporting."""
        return {
            "pairs": self.n_pairs,
            "slates": self.n_slates,
            "pairs/slate": round(self.pairs_per_slate, 1),
            "pearson": round(self.pearson, 4),
            "pearson_ci": f"[{self.pearson_ci[0]:+.4f}, {self.pearson_ci[1]:+.4f}]",
            "phi": round(self.phi, 4),
            "phi_ci": f"[{self.phi_ci[0]:+.4f}, {self.phi_ci[1]:+.4f}]",
            "same_sign": round(self.same_sign, 4),
            "same_sign_ci": f"[{self.same_sign_ci[0]:.4f}, {self.same_sign_ci[1]:.4f}]",
            "two_sided_roi": round(self.two_sided.roi, 4),
        }


def _sym_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    a = np.concatenate([x, y])
    b = np.concatenate([y, x])
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pair_point_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``(pearson, phi, same_sign)`` for pair arrays (pushes dropped for phi/same-sign)."""
    pear = _sym_corr(x, y)
    m = (x != 0) & (y != 0)
    if m.sum() < 2:
        return pear, float("nan"), float("nan")
    cx, cy = (x[m] > 0).astype(float), (y[m] > 0).astype(float)
    phi = _sym_corr(cx, cy)
    same = float(np.mean(cx == cy))
    return pear, phi, same


def pair_stats(pairs: pd.DataFrame, leg_decimal: float, n_boot: int = 1000, seed: int = 0) -> PairStats:
    """Compute :class:`PairStats` for a same-slate pairs table.

    Args:
        pairs: Output of :func:`same_slate_pairs` (needs ``slate, x, y``).
        leg_decimal: Decimal odds per leg for the two-sided ROI.
        n_boot: Slate-cluster bootstrap resamples.
        seed: RNG seed.

    Returns:
        Pair statistics with slate-bootstrap CIs.
    """
    d = pairs.loc[pairs["x"].notna() & pairs["y"].notna()]
    x = d["x"].to_numpy(dtype=float)
    y = d["y"].to_numpy(dtype=float)
    slate = pd.factorize(d["slate"], sort=True)[0] if len(d) else np.array([], dtype=int)
    n_slates = int(slate.max()) + 1 if len(d) else 0
    pear, phi, same = _pair_point_stats(x, y)
    n_no_push = int(np.sum((x != 0) & (y != 0)))
    nan = float("nan")
    ci_p, ci_phi, ci_same = (nan, nan), (nan, nan), (nan, nan)
    if n_slates >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        members = [np.flatnonzero(slate == g) for g in range(n_slates)]
        vals = np.empty((n_boot, 3))
        for b in range(n_boot):
            draw = rng.integers(0, n_slates, n_slates)
            ix = np.concatenate([members[g] for g in draw])
            vals[b] = _pair_point_stats(x[ix], y[ix])
        q = np.nanquantile(vals, [0.025, 0.975], axis=0)
        ci_p, ci_phi, ci_same = (float(q[0, 0]), float(q[1, 0])), (float(q[0, 1]), float(q[1, 1])), (float(q[0, 2]), float(q[1, 2]))
    roi = two_sided_parlay_roi(x, y, leg_decimal, expected_sign=1)
    return PairStats(
        n_pairs=int(len(d)),
        n_slates=n_slates,
        pairs_per_slate=float(len(d) / n_slates) if n_slates else nan,
        pearson=pear,
        pearson_ci=ci_p,
        n_no_push=n_no_push,
        phi=phi,
        phi_ci=ci_phi,
        same_sign=same,
        same_sign_ci=ci_same,
        two_sided=roi,
    )


# ---------------------------------------------------------------------------
# Causal persistence of the slate-mean residual
# ---------------------------------------------------------------------------


def prior_window_mean(games: pd.DataFrame, value_col: str, k: int | str) -> pd.Series:
    """Mean residual over the previous ``k`` weeks of the same season (strictly causal).

    The window is ``[week - k, week - 1]``; weeks ``1..k`` get NaN so that every
    prediction uses exactly ``k`` weeks of history. Missing weeks inside the
    window (e.g. postponed slates) contribute nothing.

    Args:
        games: Games with ``season, week, value_col``.
        value_col: Residual column.
        k: Window length in weeks, ``"season"`` for all previous weeks of the
            season (week 1 is NaN), or ``"prev_season"`` for the previous
            season's full-season mean.

    Returns:
        Series aligned with ``games.index``; NaN where no history exists.
    """
    wk = games.groupby(["season", "week"])[value_col].agg(["sum", "count"])
    out = pd.Series(np.nan, index=games.index, dtype=float)
    if k == "prev_season":
        ss = games.groupby("season")[value_col].mean()
        prev = ss.reindex(games["season"] - 1).to_numpy()
        return pd.Series(prev, index=games.index, dtype=float)
    for season, g in wk.groupby(level="season"):
        weeks = g.index.get_level_values("week")
        full = pd.RangeIndex(1, int(weeks.max()) + 1)
        s = g["sum"].droplevel("season").reindex(full, fill_value=0.0)
        c = g["count"].droplevel("season").reindex(full, fill_value=0.0)
        if k == "season":
            rs, rc = s.cumsum().shift(1), c.cumsum().shift(1)
        else:
            kk = int(k)
            rs = s.rolling(kk, min_periods=kk).sum().shift(1)
            rc = c.rolling(kk, min_periods=kk).sum().shift(1)
        mean = (rs / rc.replace(0.0, np.nan)).to_dict()
        m = games["season"] == season
        out[m] = games.loc[m, "week"].map(mean).to_numpy()
    return out


@dataclass
class OlsResult:
    """Slope of ``y`` on ``x`` with cluster-robust standard errors.

    Attributes:
        slope: OLS slope.
        intercept: OLS intercept.
        se: Cluster-robust standard error of the slope.
        t: ``slope / se``.
        p: Two-sided p-value using ``t`` with ``n_clusters - 1`` degrees of freedom.
        n: Observations.
        n_clusters: Clusters.
    """

    slope: float
    intercept: float
    se: float
    t: float
    p: float
    n: int
    n_clusters: int


def ols_clustered(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, clusters: np.ndarray | pd.Series) -> OlsResult:
    """OLS of ``y`` on ``x`` with intercept and cluster-robust (Liang-Zeger) SE.

    Args:
        x: Predictor ``[n]``.
        y: Outcome ``[n]``.
        clusters: Cluster labels ``[n]``.

    Returns:
        :class:`OlsResult`. NaNs in ``x`` or ``y`` are dropped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], cl[ok]
    n = len(x)
    nan = float("nan")
    if n < 3 or x.std() == 0:
        return OlsResult(nan, nan, nan, nan, nan, n, 0)
    xc = x - x.mean()
    slope = float(np.sum(xc * y) / np.sum(xc * xc))
    intercept = float(y.mean() - slope * x.mean())
    e = y - intercept - slope * x
    codes = pd.factorize(cl)[0]
    g = int(codes.max()) + 1
    score = np.bincount(codes, weights=xc * e, minlength=g)
    sxx = float(np.sum(xc * xc))
    var = float(np.sum(score**2)) / sxx**2
    if g > 1:
        var *= g / (g - 1) * (n - 1) / (n - 2)
    se = float(np.sqrt(var))
    t = slope / se if se > 0 else nan
    p = float(2 * stats.t.sf(abs(t), max(g - 1, 1))) if np.isfinite(t) else nan
    return OlsResult(slope, intercept, se, t, p, n, g)


def persistence_table(
    games: pd.DataFrame,
    value_col: str,
    windows: tuple[int | str, ...],
) -> pd.DataFrame:
    """Does the causal prior-window mean predict the current week's residuals?

    For each window ``k`` regresses each game's residual on the mean residual
    of the previous ``k`` weeks (same season), clustering standard errors by
    (season, week) since the predictor is constant within a week.

    Args:
        games: Games with ``season, week, value_col``.
        value_col: Residual column.
        windows: Values accepted by :func:`prior_window_mean`.

    Returns:
        One row per window: ``window, n_games, n_weeks, slope, se, t, p,
        game_sign_agree, week_sign_agree, week_sign_se, pred_sd``.
    """
    rows = []
    for k in windows:
        pred = prior_window_mean(games, value_col, k)
        d = pd.DataFrame(
            {"x": pred, "y": games[value_col], "season": games["season"], "week": games["week"]}
        ).dropna()
        cl = d["season"].astype(str) + "_" + d["week"].astype(str)
        res = ols_clustered(d["x"], d["y"], cl)
        nz = (d["x"] != 0) & (d["y"] != 0)
        game_agree = float(np.mean(np.sign(d.loc[nz, "x"]) == np.sign(d.loc[nz, "y"]))) if nz.any() else float("nan")
        wk = d.groupby(["season", "week"]).agg(x=("x", "first"), y=("y", "mean"))
        wk = wk[(wk["x"] != 0) & (wk["y"] != 0)]
        week_agree = float(np.mean(np.sign(wk["x"]) == np.sign(wk["y"]))) if len(wk) else float("nan")
        rows.append(
            {
                "window": str(k),
                "n_games": res.n,
                "n_weeks": res.n_clusters,
                "slope": round(res.slope, 4),
                "se": round(res.se, 4),
                "t": round(res.t, 2),
                "p": round(res.p, 3),
                "game_sign_agree": round(game_agree, 4),
                "week_sign_agree": round(week_agree, 4),
                "week_sign_se": round(float(np.sqrt(0.25 / len(wk))), 4) if len(wk) else float("nan"),
                "pred_sd": round(float(d["x"].std()), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scalar-state Kalman filter on the league offset
# ---------------------------------------------------------------------------


def _clip_unit(x: float, eps: float = 1e-6) -> float:
    return float(min(max(x, eps), 1.0 - eps))


@dataclass(frozen=True)
class ScalarFactorParams:
    """Hyperparameters of the league-offset AR(1) model.

    Between consecutive slates of the same season the offset ``theta`` evolves
    as ``theta' = mean + persistence * (theta - mean) + N(0, process_std²)``;
    across a season boundary it evolves as
    ``theta' = mean + season_persistence * (theta - mean) + N(0, season_std²)``.
    Each game on a slate observes ``theta + N(0, obs_std²)``, so two games on
    the same slate have covariance ``Var(theta)`` and correlation
    ``Var(theta) / (Var(theta) + obs_std²)``.

    Attributes:
        prior_std: Std of the offset before the first slate of the sample.
        process_std: Week-to-week innovation std (points).
        season_std: Innovation std across a season boundary (points).
        obs_std: Single-game noise std (points).
        persistence: AR(1) coefficient between consecutive slates.
        season_persistence: AR(1) coefficient across a season boundary.
        mean: Long-run level the offset reverts to (points). This is the
            market's persistent bias: a marginal (single-bet) effect, not a
            source of cross-game correlation.
    """

    prior_std: float = 2.0
    process_std: float = 0.3
    season_std: float = 1.0
    obs_std: float = 13.0
    persistence: float = 0.95
    season_persistence: float = 0.5
    mean: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Unconstrained parameterization for the optimizer."""
        rho = _clip_unit(self.persistence)
        rho_s = _clip_unit(self.season_persistence)
        return np.array(
            [
                np.log(max(self.prior_std, 1e-9)),
                np.log(max(self.process_std, 1e-9)),
                np.log(max(self.season_std, 1e-9)),
                np.log(max(self.obs_std, 1e-9)),
                np.log(rho / (1 - rho)),
                np.log(rho_s / (1 - rho_s)),
                self.mean,
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "ScalarFactorParams":
        """Inverse of :meth:`to_vector`."""
        return cls(
            prior_std=float(np.exp(v[0])),
            process_std=float(np.exp(v[1])),
            season_std=float(np.exp(v[2])),
            obs_std=float(np.exp(v[3])),
            persistence=float(1 / (1 + np.exp(-v[4]))),
            season_persistence=float(1 / (1 + np.exp(-v[5]))),
            mean=float(v[6]),
        )

    @classmethod
    def stationary(
        cls,
        factor_std: float,
        obs_std: float,
        mean: float = 0.0,
        persistence: float = 0.9,
        season_persistence: float = 0.5,
    ) -> "ScalarFactorParams":
        """Parameters whose offset has stationary std ``factor_std`` at every slate.

        Args:
            factor_std: Stationary std of the shared offset (points).
            obs_std: Single-game noise std (points).
            mean: Long-run offset level.
            persistence: Within-season AR(1) coefficient.
            season_persistence: Across-season AR(1) coefficient.

        Returns:
            Parameters with innovations ``factor_std * sqrt(1 - rho²)`` so that
            the marginal variance of the offset is ``factor_std²`` before and
            after every transition.
        """
        return cls(
            prior_std=factor_std,
            process_std=factor_std * float(np.sqrt(1 - persistence**2)),
            season_std=factor_std * float(np.sqrt(1 - season_persistence**2)),
            obs_std=obs_std,
            persistence=persistence,
            season_persistence=season_persistence,
            mean=mean,
        )

    @property
    def stationary_std(self) -> float:
        """Stationary std of the within-season AR(1) (``inf`` when persistence is 1)."""
        if self.persistence >= 1.0:
            return float("inf")
        return float(self.process_std / np.sqrt(1 - self.persistence**2))

    def pair_corr(self, state_var: float) -> float:
        """Same-slate pair correlation implied by an offset variance."""
        return float(state_var / (state_var + self.obs_std**2))


@dataclass
class SlateStats:
    """Per-slate sufficient statistics of a residual, in chronological order.

    The Gaussian predictive density of a slate depends on its residuals only
    through ``n``, ``sum`` and ``sum of squares``, so the filter and the
    likelihood can run on these arrays without touching the games table.

    Attributes:
        season: Season of each slate ``[S]``.
        week: Week of each slate ``[S]``.
        n: Games on each slate ``[S]``.
        total: Sum of residuals per slate ``[S]``.
        total_sq: Sum of squared residuals per slate ``[S]``.
    """

    season: np.ndarray
    week: np.ndarray
    n: np.ndarray
    total: np.ndarray
    total_sq: np.ndarray

    def __len__(self) -> int:
        return int(len(self.n))


def slate_stats(games: pd.DataFrame, value_col: str) -> SlateStats:
    """Sufficient statistics of ``value_col`` per (season, week) slate.

    Args:
        games: Games with ``season, week`` and ``value_col``.
        value_col: Residual column (NaNs are dropped).

    Returns:
        :class:`SlateStats` sorted by (season, week).
    """
    d = games.loc[games[value_col].notna(), ["season", "week", value_col]]
    v = d[value_col].to_numpy(dtype=float)
    agg = (
        d.assign(_v=v, _v2=v * v)
        .groupby(["season", "week"], sort=True)
        .agg(n=("_v", "size"), total=("_v", "sum"), total_sq=("_v2", "sum"))
        .reset_index()
    )
    return SlateStats(
        season=agg["season"].to_numpy(dtype=int),
        week=agg["week"].to_numpy(dtype=int),
        n=agg["n"].to_numpy(dtype=float),
        total=agg["total"].to_numpy(dtype=float),
        total_sq=agg["total_sq"].to_numpy(dtype=float),
    )


def _filter_core(
    stats: SlateStats, p: ScalarFactorParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run the scalar Kalman filter over slate statistics.

    Args:
        stats: Slate sufficient statistics.
        p: Model parameters.

    Returns:
        ``(prior_mean, prior_var, post_mean, post_var, log_likelihood)`` where the
        arrays are ``[S]`` and the prior quantities for slate ``t`` use only
        slates ``< t``.
    """
    S = len(stats)
    r2 = p.obs_std**2
    prior_m = np.empty(S)
    prior_P = np.empty(S)
    post_m = np.empty(S)
    post_P = np.empty(S)
    m = p.mean
    P = p.prior_std**2
    ll = 0.0
    for t in range(S):
        if t > 0:
            if stats.season[t] != stats.season[t - 1]:
                rho, q2 = p.season_persistence, p.season_std**2
            else:
                rho, q2 = p.persistence, p.process_std**2
            m = p.mean + rho * (m - p.mean)
            P = rho * rho * P + q2
        n, s, s2 = stats.n[t], stats.total[t], stats.total_sq[t]
        prior_m[t], prior_P[t] = m, P
        # Predictive density of the slate: y ~ N(m 1, P 11' + r2 I).
        dev_sum = s - n * m
        dev_sq = s2 - 2 * m * s + n * m * m
        shrink = P / (r2 + n * P)
        quad = (dev_sq - shrink * dev_sum * dev_sum) / r2
        logdet = n * np.log(r2) + np.log1p(n * P / r2)
        ll += -0.5 * (n * np.log(2 * np.pi) + logdet + quad)
        # Posterior update with all n observations at once (gain form, stable
        # as P -> 0): gain on the slate sum is P / (r2 + n P).
        m_new = m + shrink * dev_sum
        P_new = P * r2 / (r2 + n * P)
        post_m[t], post_P[t] = m_new, P_new
        m, P = m_new, P_new
    return prior_m, prior_P, post_m, post_P, float(ll)


@dataclass
class ScalarFactorOutput:
    """Predictions from one pass of :class:`ScalarFactorKalman`.

    Attributes:
        games: Per-game rows ``game_id, season, week, resid, pred_mean, pred_var,
            state_var, pred_pair_corr`` where ``pred_mean`` is the filtered
            offset before the slate, ``state_var`` its prior variance (the
            predicted covariance of any two residuals on the slate) and
            ``pred_pair_corr = state_var / (state_var + obs_std²)``.
        slates: One row per slate: ``season, week, n_games, pred_mean, state_var,
            pred_pair_corr, slate_mean, post_mean, post_var``.
        log_likelihood: Sum of Gaussian predictive log densities over slates.
        params: Parameters used for the pass.
    """

    games: pd.DataFrame
    slates: pd.DataFrame
    log_likelihood: float
    params: ScalarFactorParams

    def factor_collapsed(self, tol: float = 1e-6) -> bool:
        """True when the predicted offset variance is negligible on every slate.

        A maximum-likelihood fit that finds no shared factor drives the state
        variance to (numerically) zero; calibration regressions on such
        predictions are not identified.
        """
        return bool(len(self.slates) == 0 or self.slates["state_var"].max() < tol)


class ScalarFactorKalman:
    """Causal Kalman filter for a scalar league-wide offset in a residual.

    Args:
        params: Model hyperparameters.
        value_col: Residual column (``tresid`` for the scoring environment,
            ``resid`` for home-field advantage).
    """

    def __init__(self, params: ScalarFactorParams | None = None, value_col: str = "tresid") -> None:
        self.params = params or ScalarFactorParams()
        self.value_col = value_col

    def run(self, games: pd.DataFrame) -> ScalarFactorOutput:
        """Filter all slates chronologically; every prediction precedes its slate.

        Args:
            games: Games with ``game_id, season, week`` and ``value_col``.

        Returns:
            :class:`ScalarFactorOutput`.
        """
        p = self.params
        stats = slate_stats(games, self.value_col)
        prior_m, prior_P, post_m, post_P, ll = _filter_core(stats, p)
        pair_corr = prior_P / (prior_P + p.obs_std**2)
        slates = pd.DataFrame(
            {
                "season": stats.season,
                "week": stats.week,
                "n_games": stats.n.astype(int),
                "pred_mean": prior_m,
                "state_var": prior_P,
                "pred_pair_corr": pair_corr,
                "slate_mean": stats.total / np.maximum(stats.n, 1.0),
                "post_mean": post_m,
                "post_var": post_P,
            }
        )
        d = games.loc[games[self.value_col].notna(), ["game_id", "season", "week", self.value_col]]
        d = d.sort_values(["season", "week", "game_id"]).rename(columns={self.value_col: "resid"})
        key = pd.MultiIndex.from_arrays([stats.season, stats.week])
        idx = key.get_indexer(pd.MultiIndex.from_frame(d[["season", "week"]]))
        games_out = d.assign(
            pred_mean=prior_m[idx],
            pred_var=prior_P[idx] + p.obs_std**2,
            state_var=prior_P[idx],
            pred_pair_corr=pair_corr[idx],
        ).reset_index(drop=True)
        return ScalarFactorOutput(games=games_out, slates=slates, log_likelihood=ll, params=p)

    def fit(self, games: pd.DataFrame, max_iter: int = 4000, n_starts: int = 3) -> ScalarFactorParams:
        """Maximize predictive log-likelihood on training seasons.

        Nelder-Mead from several starting points (the current parameters, a
        no-factor start and a large-factor start); the best optimum is kept.

        Args:
            games: Training games (must precede any evaluation games).
            max_iter: Nelder-Mead iteration budget per start.
            n_starts: Number of starting points to use (1-3).

        Returns:
            Fitted parameters (also stored on ``self.params``).
        """
        stats = slate_stats(games, self.value_col)

        def neg_ll(v: np.ndarray) -> float:
            return -_filter_core(stats, ScalarFactorParams.from_vector(v))[4]

        obs0 = float(np.sqrt(np.sum(stats.total_sq) / np.sum(stats.n)))
        mean0 = float(np.sum(stats.total) / np.sum(stats.n))
        starts = [
            self.params,
            ScalarFactorParams(0.1, 0.05, 0.1, obs0, 0.9, 0.5, mean0),
            ScalarFactorParams.stationary(3.0, obs0, mean0, 0.9, 0.5),
        ][: max(1, n_starts)]
        best: tuple[float, ScalarFactorParams] | None = None
        for s in starts:
            res = optimize.minimize(
                neg_ll, s.to_vector(), method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-6}
            )
            if best is None or res.fun < best[0]:
                best = (float(res.fun), ScalarFactorParams.from_vector(res.x))
        assert best is not None
        self.params = best[1]
        return self.params


def simulate_common_factor(
    schedule: pd.DataFrame,
    params: ScalarFactorParams,
    seed: int = 0,
    value_col: str = "tresid",
) -> pd.DataFrame:
    """Draw synthetic residuals from the scalar-factor generative model.

    Args:
        schedule: Games with ``season, week, game_id``.
        params: Generative parameters.
        seed: RNG seed.
        value_col: Name of the synthetic residual column.

    Returns:
        Copy of ``schedule`` with ``value_col`` and ``true_offset`` columns.
    """
    rng = np.random.default_rng(seed)
    out = schedule.sort_values(["season", "week", "game_id"]).copy()
    out[value_col] = np.nan
    out["true_offset"] = np.nan
    theta = params.mean + rng.normal(0, params.prior_std)
    prev_season: int | None = None
    for (season, _), sg in out.groupby(["season", "week"], sort=True):
        if prev_season is not None:
            if season != prev_season:
                rho, sd = params.season_persistence, params.season_std
            else:
                rho, sd = params.persistence, params.process_std
            theta = params.mean + rho * (theta - params.mean) + rng.normal(0, sd)
        prev_season = int(season)
        out.loc[sg.index, value_col] = theta + rng.normal(0, params.obs_std, len(sg))
        out.loc[sg.index, "true_offset"] = theta
    return out


def kalman_pair_calibration(output: ScalarFactorOutput, value_col: str = "resid") -> tuple[OlsResult, OlsResult]:
    """Calibration regressions for a scalar-factor filter pass.

    Args:
        output: :class:`ScalarFactorOutput` (the evaluation window).
        value_col: Residual column inside ``output.games`` (always ``resid``).

    Returns:
        ``(cov_slope, mean_slope)``: OLS of realized same-slate residual
        products on predicted covariance ``state_var`` (1 = calibrated), and
        OLS of realized residuals on ``pred_mean`` (1 = the market under-reacts
        exactly as the filter says, 0 = market efficient). Both cluster by
        slate. ``cov_slope`` is all-NaN when the fitted factor has collapsed
        to zero variance (see :meth:`ScalarFactorOutput.factor_collapsed`).
    """
    g = output.games
    nan = float("nan")
    cl = g["season"].astype(str) + "_" + g["week"].astype(str)
    mean_slope = ols_clustered(g["pred_mean"], g[value_col], cl)
    if output.factor_collapsed():
        return OlsResult(nan, nan, nan, nan, nan, 0, 0), mean_slope
    pairs = same_slate_pairs(g.rename(columns={value_col: "v"}), "v", ("season", "week"))
    cov = g.groupby(["season", "week"])["state_var"].first()
    pairs["pred_cov"] = cov.reindex(pd.MultiIndex.from_frame(pairs[["season", "week"]])).to_numpy()
    cov_slope = ols_clustered(pairs["pred_cov"], pairs["x"] * pairs["y"], pairs["slate"])
    return cov_slope, mean_slope


def icc_by_uncertainty(output: ScalarFactorOutput, n_bins: int = 4) -> pd.DataFrame:
    """Realized within-slate ICC in bins of the filter's predicted uncertainty.

    If the filter is right, slates where its prior variance is largest (after
    a season boundary, or after slates with few games) should show the largest
    realized clustering.

    Args:
        output: :class:`ScalarFactorOutput` for the evaluation window.
        n_bins: Quantile bins of ``state_var`` over slates.

    Returns:
        One row per bin: ``bin, slates, games, pred_pair_corr, icc, pairwise,
        pairs``.
    """
    sl = output.slates
    if len(sl) == 0:
        return pd.DataFrame(columns=["bin", "slates", "games", "pred_pair_corr", "icc", "pairwise", "pairs"])
    ranks = sl["state_var"].rank(method="first")
    sl = sl.assign(bin=pd.qcut(ranks, q=min(n_bins, len(sl)), labels=False))
    g = output.games.merge(sl[["season", "week", "bin"]], on=["season", "week"])
    rows = []
    for b, gb in g.groupby("bin", sort=True):
        codes = group_codes(gb, ("season", "week"))
        r = icc_oneway(gb["resid"].to_numpy(), codes)
        rows.append(
            {
                "bin": int(b),
                "slates": int(sl.loc[sl["bin"] == b].shape[0]),
                "games": int(len(gb)),
                "pred_pair_corr": float(gb["pred_pair_corr"].mean()),
                "icc": float(r.icc),
                "pairwise": float(r.pairwise_corr),
                "pairs": int(r.n_pairs),
            }
        )
    return pd.DataFrame(rows)


def factor_profile(
    games: pd.DataFrame,
    value_col: str,
    factor_stds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),
    persistence: float = 0.9,
    season_persistence: float = 0.5,
) -> pd.DataFrame:
    """Profile log-likelihood over the stationary size of the shared factor.

    For each ``factor_std`` the offset is a stationary AR(1) with that std and
    fixed persistence; ``obs_std`` and ``mean`` are re-optimized (nuisance
    parameters). ``delta_ll`` is relative to ``factor_std = 0``; the usual 95%
    profile-likelihood bound is the largest ``factor_std`` with
    ``delta_ll >= -1.92``.

    Args:
        games: Games with ``season, week, value_col``.
        value_col: Residual column.
        factor_stds: Grid of stationary factor stds (points), should include 0.
        persistence: Within-season AR(1) coefficient held fixed.
        season_persistence: Across-season AR(1) coefficient held fixed.

    Returns:
        One row per grid point: ``factor_std, obs_std, mean, log_likelihood,
        delta_ll, pair_corr_stationary`` (``s² / (s² + obs²)``, what a bettor
        who ignores history faces) and ``pair_corr_filtered`` (mean over slates
        of the filter's prior ``P / (P + obs²)``, what remains after learning).
    """
    stats = slate_stats(games, value_col)
    obs0 = float(np.sqrt(np.sum(stats.total_sq) / np.sum(stats.n)))
    mean0 = float(np.sum(stats.total) / np.sum(stats.n))
    rows = []
    for s in factor_stds:

        def neg_ll(v: np.ndarray, s: float = float(s)) -> float:
            p = ScalarFactorParams.stationary(s, float(np.exp(v[0])), float(v[1]), persistence, season_persistence)
            return -_filter_core(stats, p)[4]

        res = optimize.minimize(neg_ll, np.array([np.log(obs0), mean0]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8})
        obs_std, mean = float(np.exp(res.x[0])), float(res.x[1])
        p = ScalarFactorParams.stationary(float(s), obs_std, mean, persistence, season_persistence)
        _, prior_P, _, _, ll = _filter_core(stats, p)
        rows.append(
            {
                "factor_std": float(s),
                "obs_std": obs_std,
                "mean": mean,
                "log_likelihood": ll,
                "pair_corr_stationary": p.pair_corr(float(s) ** 2),
                "pair_corr_filtered": float(np.mean(prior_P / (prior_P + obs_std**2))),
            }
        )
    out = pd.DataFrame(rows)
    base = out.loc[out["factor_std"] == out["factor_std"].min(), "log_likelihood"].iloc[0]
    out["delta_ll"] = out["log_likelihood"] - base
    return out


def profile_upper_bound(profile: pd.DataFrame, threshold: float = 1.92) -> float:
    """Largest factor std whose profile log-likelihood is within ``threshold`` of the best.

    Linear interpolation between the last grid point inside the bound and the
    first outside it. Returns the largest grid value if no point is outside.

    Args:
        profile: Output of :func:`factor_profile`.
        threshold: Log-likelihood drop defining the bound (1.92 for 95%).

    Returns:
        Upper bound on the stationary factor std (points).
    """
    pr = profile.sort_values("factor_std").reset_index(drop=True)
    rel = (pr["log_likelihood"] - pr["log_likelihood"].max()).to_numpy()
    s = pr["factor_std"].to_numpy(dtype=float)
    i_max = int(np.argmax(rel))
    outside = np.flatnonzero(rel[i_max:] < -threshold)
    if len(outside) == 0:
        return float(s[-1])
    j = i_max + int(outside[0])  # first grid point past the maximum that is outside the bound
    r0, r1 = rel[j - 1], rel[j]
    frac = (r0 + threshold) / (r0 - r1) if r0 != r1 else 0.0
    return float(s[j - 1] + frac * (s[j] - s[j - 1]))


# ---------------------------------------------------------------------------
# Directional slate strategy
# ---------------------------------------------------------------------------


@dataclass
class DirectionalResult:
    """Realized result of betting every game on a slate in one direction.

    Attributes:
        n_slates: Slates with a non-zero signal.
        n_singles: Single bets resolved (pushes excluded).
        singles_win_rate: Fraction of resolved singles won.
        singles_roi: Profit per unit staked on singles.
        singles_roi_ci: 95% slate-bootstrap CI.
        n_parlays: 2-leg same-slate parlays resolved (a push refunds the parlay).
        parlays_win_rate: Fraction of resolved parlays won.
        parlays_roi: Profit per unit staked on parlays.
        parlays_roi_ci: 95% slate-bootstrap CI.
    """

    n_slates: int
    n_singles: int
    singles_win_rate: float
    singles_roi: float
    singles_roi_ci: tuple[float, float]
    n_parlays: int
    parlays_win_rate: float
    parlays_roi: float
    parlays_roi_ci: tuple[float, float]

    def to_row(self) -> dict[str, float | int | str]:
        """Flatten for tabular reporting."""
        return {
            "slates": self.n_slates,
            "singles": self.n_singles,
            "single_win": round(self.singles_win_rate, 4),
            "single_roi": round(self.singles_roi, 4),
            "single_roi_ci": f"[{self.singles_roi_ci[0]:+.3f}, {self.singles_roi_ci[1]:+.3f}]",
            "parlays": self.n_parlays,
            "parlay_win": round(self.parlays_win_rate, 4),
            "parlay_roi": round(self.parlays_roi, 4),
            "parlay_roi_ci": f"[{self.parlays_roi_ci[0]:+.3f}, {self.parlays_roi_ci[1]:+.3f}]",
        }


def _slate_outcomes(games: pd.DataFrame, signal_col: str, value_col: str) -> pd.DataFrame:
    d = games.loc[games[signal_col].notna() & (games[signal_col] != 0) & games[value_col].notna()]
    direction = np.sign(d[signal_col].to_numpy(dtype=float))
    outcome = np.sign(d[value_col].to_numpy(dtype=float)) * direction
    tmp = pd.DataFrame({"season": d["season"].to_numpy(), "week": d["week"].to_numpy(), "o": outcome})
    agg = tmp.groupby(["season", "week"])["o"].agg(
        wins=lambda s: float((s > 0).sum()), losses=lambda s: float((s < 0).sum()), pushes=lambda s: float((s == 0).sum())
    )
    return agg.reset_index()


def _direction_pnl(s: pd.DataFrame, leg_decimal: float) -> tuple[float, float, float, float, float, float]:
    """``(singles_n, singles_wins, singles_profit, parlays_n, parlays_wins, parlays_profit)``."""
    w, l = s["wins"].to_numpy(), s["losses"].to_numpy()
    n_res = w + l
    singles_n = float(n_res.sum())
    singles_profit = float(np.sum(w * (leg_decimal - 1.0) - l))
    par_n = n_res * (n_res - 1) / 2.0
    par_w = w * (w - 1) / 2.0
    parlays_n = float(par_n.sum())
    parlays_profit = float(np.sum(par_w * (leg_decimal**2 - 1.0) - (par_n - par_w)))
    return singles_n, float(w.sum()), singles_profit, parlays_n, float(par_w.sum()), parlays_profit


def directional_strategy(
    games: pd.DataFrame,
    signal_col: str,
    value_col: str,
    leg_decimal: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> DirectionalResult:
    """Bet every game on a slate on the side indicated by a slate-level signal.

    A positive signal bets over (or home cover) on every game of the slate, a
    negative signal bets the other side; a zero/NaN signal skips the slate. All
    2-leg parlays among the slate's games are bet in the same direction. A push
    refunds the single, and any parlay containing it.

    Args:
        games: Games with ``season, week, signal_col, value_col``.
        signal_col: Slate-level signal column (constant within a slate).
        value_col: Residual column deciding the outcome.
        leg_decimal: Decimal odds per leg.
        n_boot: Slate-cluster bootstrap resamples for ROI CIs.
        seed: RNG seed.

    Returns:
        :class:`DirectionalResult`.
    """
    s = _slate_outcomes(games, signal_col, value_col)
    nan = float("nan")
    if len(s) == 0:
        return DirectionalResult(0, 0, nan, nan, (nan, nan), 0, nan, nan, (nan, nan))
    sn, sw, sp, pn, pw, pp = _direction_pnl(s, leg_decimal)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        bs = s.iloc[rng.integers(0, len(s), len(s))]
        bsn, _, bsp, bpn, _, bpp = _direction_pnl(bs, leg_decimal)
        boots[b] = (bsp / bsn if bsn else np.nan, bpp / bpn if bpn else np.nan)
    q = np.nanquantile(boots, [0.025, 0.975], axis=0) if n_boot else np.full((2, 2), np.nan)
    return DirectionalResult(
        n_slates=int(len(s)),
        n_singles=int(sn),
        singles_win_rate=sw / sn if sn else nan,
        singles_roi=sp / sn if sn else nan,
        singles_roi_ci=(float(q[0, 0]), float(q[1, 0])),
        n_parlays=int(pn),
        parlays_win_rate=pw / pn if pn else nan,
        parlays_roi=pp / pn if pn else nan,
        parlays_roi_ci=(float(q[0, 1]), float(q[1, 1])),
    )


# ---------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------


def stratum_masks(games: pd.DataFrame, config: CommonFactorConfig, evaluation_only: bool = True) -> dict[str, pd.Series]:
    """Pre-registered subsets for the shared-factor analysis.

    Args:
        games: Games with ``season, week``.
        config: Provides the early-week cutoff, rule-change list and training end.
        evaluation_only: Restrict the generic strata to seasons after
            ``train_seasons_end``. Rule-change seasons inside the training
            window are still returned (labelled ``(in-sample)``).

    Returns:
        Ordered mapping ``label -> boolean mask``.
    """
    oos = games["season"] > config.train_seasons_end
    base = oos if evaluation_only else pd.Series(True, index=games.index)
    lo, hi = int(games.loc[base, "season"].min()), int(games.loc[base, "season"].max())
    masks: dict[str, pd.Series] = {
        f"all {lo}-{hi}": base,
        f"weeks 1-{config.early_weeks}": base & (games["week"] <= config.early_weeks),
        f"weeks {config.early_weeks + 1}+": base & (games["week"] > config.early_weeks),
    }
    for rc in config.rule_changes:
        label = rc.label if rc.season > config.train_seasons_end else f"{rc.label} (in-sample)"
        masks[label] = games["season"] == rc.season
    return masks
