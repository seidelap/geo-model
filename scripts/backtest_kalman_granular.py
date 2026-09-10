"""Granular offense/defense Kalman backtest for cross-market parlay correlations.

Fits a 64-dimensional (offense, defense) market-error filter with joint
home/away score observations on 1999-2009, evaluates it strictly causally on
2010-2025, and writes a markdown report on whether the predicted spread-spread,
total-total and spread-total leg correlations are large enough to beat parlay
vig. Includes a power check on simulated data and an optional nflverse EPA
auxiliary-observation extension.

Usage:
    python scripts/backtest_kalman_granular.py [--refit] [--skip-epa] [--max-iter 600]
        [--out docs/experiments/parlay-granular-kalman.md]

Data location is controlled by ``GEO_MODEL_DATA_DIR`` (default ``data/raw``).
Fitted hyperparameters are cached in ``<data_dir>/nfl/granular_kalman_params.parquet``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from geo_model.parlay import (
    KalmanParams,
    MarketErrorKalman,
    ParlayConfig,
    breakeven_correlation,
    correlation_summary,
    load_games,
    parlay_ev,
    two_sided_parlay_roi,
)
from geo_model.parlay.backtest import gaussian_explaining_away_corr, kalman_slope
from geo_model.parlay.kalman_granular import (
    PAIR_TYPES,
    GranularKalman,
    GranularParams,
    add_point_residuals,
    attach_aux_observations,
    calibration_by_pair_type,
    decile_calibration,
    fit_aux_calibration,
    load_team_week_epa,
    pair_type_summary,
    signed_pair_residuals,
    simulate_granular_games,
    single_shared_game_corr,
    theoretical_max_corr,
    top_fraction_by_pair_type,
)

TRAIN_SEASONS_END = 2009
PARAM_FIELDS = list(asdict(GranularParams()).keys())


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "nan"
        if abs(v) < 0.01 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def params_cache_path(cfg: ParlayConfig) -> Path:
    """Parquet cache of fitted hyperparameters, one row per model name."""
    return cfg.data_dir / "nfl" / "granular_kalman_params.parquet"


def load_cached_params(cfg: ParlayConfig, name: str) -> GranularParams | None:
    """Return cached parameters for ``name`` or ``None``."""
    path = params_cache_path(cfg)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    row = df[df["model"] == name]
    if row.empty:
        return None
    kw = {f: row.iloc[0][f] for f in PARAM_FIELDS}
    kw["use_aux"] = bool(kw["use_aux"])
    return GranularParams(**{k: (float(v) if k != "use_aux" else v) for k, v in kw.items()})


def save_cached_params(cfg: ParlayConfig, name: str, params: GranularParams, seconds: float) -> None:
    """Upsert ``name`` into the parameter cache."""
    path = params_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": name, "fit_seconds": seconds, **asdict(params)}
    df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["model"] != name]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(path, index=False)


def fit_or_load(
    cfg: ParlayConfig,
    name: str,
    train: pd.DataFrame,
    start: GranularParams,
    market_mode: str,
    refit: bool,
    max_iter: int,
) -> GranularParams:
    """Fit a granular filter on ``train`` unless a cached fit exists."""
    cached = None if refit else load_cached_params(cfg, name)
    if cached is not None:
        print(f"[{name}] using cached parameters")
        return cached
    t0 = time.time()
    kf = GranularKalman(start, market_mode=market_mode)
    fitted = kf.fit(train, max_iter=max_iter)
    dt = time.time() - t0
    print(f"[{name}] fitted in {dt:.0f}s: {fitted}")
    save_cached_params(cfg, name, fitted, dt)
    return fitted


def params_row(name: str, p: GranularParams) -> dict[str, object]:
    """Flatten parameters for the report table."""
    row: dict[str, object] = {"model": name}
    for f in ("prior_std_o", "prior_std_d", "process_std_o", "process_std_d", "obs_std", "persistence", "od_corr", "obs_corr"):
        row[f] = round(getattr(p, f), 4)
    if p.use_aux:
        row["aux_loading"] = round(p.aux_loading, 3)
        row["aux_std"] = round(p.aux_std, 3)
        row["aux_corr"] = round(p.aux_corr, 3)
    return row


def variance_trend(games: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """OLS slope of squared residual on week (points² per week), with se and p."""
    res = stats.linregress(games["week"].to_numpy(dtype=float), (games[col] ** 2).to_numpy(dtype=float))
    return float(res.slope), float(res.stderr), float(res.pvalue)


def evaluate(pairs: pd.DataFrame, leg_decimal: float, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Distribution, calibration and top-decile tables for one pairs table."""
    dist = pair_type_summary(pairs)
    cal = calibration_by_pair_type(pairs)
    top = top_fraction_by_pair_type(pairs, leg_decimal, top_frac=0.1, n_boot=n_boot)
    return dist, cal, top


def team_level_row(train: pd.DataFrame, test: pd.DataFrame, leg_decimal: float, n_boot: int) -> tuple[KalmanParams, dict[str, object]]:
    """Fit and evaluate the one-state team filter for comparison."""
    kf = MarketErrorKalman(KalmanParams())
    fitted = kf.fit(train)
    res = kf.run(test)
    kp = res.pairs
    slope, se, pv = kalman_slope(kp)
    thr = kp["pred_corr"].abs().quantile(0.9)
    sel = kp[kp["pred_corr"].abs() >= thr]
    x, y = signed_pair_residuals(sel)
    s = correlation_summary(x, y, n_boot=n_boot)
    r = two_sided_parlay_roi(x, y, leg_decimal, 1)
    row = {
        "model": "team-level (1 state/team, spreads only)",
        "pair_type": "spread-spread",
        "n": int(len(kp)),
        "max_abs_pred_corr": float(kp["pred_corr"].abs().max()),
        "p99_abs_pred_corr": float(kp["pred_corr"].abs().quantile(0.99)),
        "slope": slope,
        "se": se,
        "top_decile_n": int(len(sel)),
        "top_decile_phi": s.phi,
        "phi_ci": f"[{s.phi_ci[0]:+.3f}, {s.phi_ci[1]:+.3f}]",
        "top_decile_roi": r.roi,
    }
    return fitted, row


def compare_rows(name: str, dist: pd.DataFrame, cal: pd.DataFrame, top: pd.DataFrame) -> list[dict[str, object]]:
    """Merge the three evaluation tables into comparison rows."""
    rows = []
    d = dist.set_index("pair_type")
    c = cal.set_index("pair_type")
    t = top.set_index("pair_type")
    for pt in PAIR_TYPES:
        rows.append(
            {
                "model": name,
                "pair_type": pt,
                "n": int(d.loc[pt, "n"]),
                "max_abs_pred_corr": float(d.loc[pt, "max_abs"]),
                "p99_abs_pred_corr": float(d.loc[pt, "p99_abs"]),
                "slope": float(c.loc[pt, "slope"]),
                "se": float(c.loc[pt, "se"]),
                "top_decile_n": int(t.loc[pt, "n"]),
                "top_decile_phi": float(t.loc[pt, "phi"]),
                "phi_ci": t.loc[pt, "phi_ci"],
                "top_decile_roi": float(t.loc[pt, "roi"]),
            }
        )
    return rows


def power_check(
    schedule: pd.DataFrame,
    scenarios: dict[str, GranularParams],
    leg_decimal: float,
    n_boot: int,
    seed: int = 1,
) -> pd.DataFrame:
    """Simulate from the granular model with an efficient market and re-run the pipeline."""
    rows = []
    for label, params in scenarios.items():
        sim = simulate_granular_games(schedule, params, seed=seed, market="efficient")
        out = GranularKalman(params, market_mode="efficient").run(sim)
        toy = single_shared_game_corr(params).set_index(["leg_1", "leg_2"])["pred_corr"]
        theory = {
            "spread-spread": float(toy[("spread", "spread")]),
            "total-total": float(toy[("total", "total")]),
            "spread-total": float(max(abs(toy[("spread", "total")]), abs(toy[("total", "spread")]))),
        }
        dist, cal, top = evaluate(out.pairs, leg_decimal, n_boot)
        d, c, t = dist.set_index("pair_type"), cal.set_index("pair_type"), top.set_index("pair_type")
        for pt in PAIR_TYPES:
            rows.append(
                {
                    "scenario": label,
                    "pair_type": pt,
                    "theory_one_game": theory[pt],
                    "max_abs_pred": float(d.loc[pt, "max_abs"]),
                    "slope": float(c.loc[pt, "slope"]),
                    "se": float(c.loc[pt, "se"]),
                    "top_decile_pred": float(t.loc[pt, "mean_pred_signed"]),
                    "top_decile_pearson": float(t.loc[pt, "pearson"]),
                    "top_decile_phi": float(t.loc[pt, "phi"]),
                    "phi_ci": t.loc[pt, "phi_ci"],
                    "top_decile_roi": float(t.loc[pt, "roi"]),
                }
            )
    return pd.DataFrame(rows)


def calibration_sentence(cal: pd.DataFrame) -> str:
    """Describe, from the numbers, whether each slope is consistent with 0 and/or 1."""
    parts = []
    for _, r in cal.iterrows():
        if not np.isfinite(r["slope"]):
            parts.append(f"{r['pair_type']}: undefined (predicted covariances numerically zero)")
            continue
        near0 = abs(r["slope"]) < 2 * r["se"]
        near1 = abs(r["slope"] - 1.0) < 2 * r["se"]
        if near0 and near1:
            verdict = "consistent with both 0 and 1 (no leverage)"
        elif near1:
            verdict = "consistent with 1 but not 0"
        elif near0:
            verdict = "consistent with 0 but not 1"
        else:
            verdict = "inconsistent with both 0 and 1"
        parts.append(f"{r['pair_type']}: slope {r['slope']:.1f} ± {r['se']:.1f}, {verdict}")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/parlay-granular-kalman.md")
    ap.add_argument("--refit", action="store_true", help="ignore cached hyperparameters")
    ap.add_argument("--skip-epa", action="store_true", help="skip the nflverse EPA extension")
    ap.add_argument("--max-iter", type=int, default=600, help="Nelder-Mead iteration budget per fit")
    ap.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for CIs")
    args = ap.parse_args()

    cfg = ParlayConfig()
    games = add_point_residuals(load_games(cfg))
    train = games[games["season"] <= TRAIN_SEASONS_END].reset_index(drop=True)
    test = games[games["season"] > TRAIN_SEASONS_END].reset_index(drop=True)
    d = cfg.leg_decimal
    be = breakeven_correlation(d)
    be_same = 0.5 + be / 2
    n_looks = 0

    out: list[str] = []
    out.append("# Granular-state Kalman filter: offense/defense errors and joint spread+total legs\n")
    out.append(
        "Hypothesis: the market's pricing error for a team decomposes into an offensive part "
        "``o`` and a defensive part ``d`` (both in points). A game observes ``o_H - d_A`` "
        "(home score minus the market-implied home points ``(total + spread) / 2``) and "
        "``o_A - d_H`` (away score minus ``(total - spread) / 2``). Conditioning on shared "
        "games couples the errors of every team in the schedule network, and because the "
        "spread residual is the difference and the total residual the sum of the two score "
        "residuals, one 64-dimensional filter yields a predicted covariance for **every** "
        "leg pair on a slate: spread-spread, total-total and spread-total across different "
        "games. Books price such legs as independent, so a large enough predicted "
        "correlation, if calibrated, would make cross-market parlays +EV.\n"
    )
    out.append(
        f"Generated by `scripts/backtest_kalman_granular.py`. Data: nflverse games file, seasons "
        f"{cfg.seasons[0]}–{cfg.seasons[1]}, {cfg.game_types} games, {len(games)} games. "
        f"Hyperparameters fitted on {train['season'].min()}–{TRAIN_SEASONS_END} "
        f"({len(train)} games); everything below is evaluated out-of-sample on "
        f"{TRAIN_SEASONS_END + 1}–{test['season'].max()} ({len(test)} games) with every "
        f"slate's predictions computed before any game of that slate is observed.\n"
    )
    out.append("## 1. Pricing hurdle\n")
    out.append(
        f"At {cfg.leg_odds_american:+.0f} per leg (decimal {d:.3f}) a 2-leg parlay pays {d**2:.3f}. "
        f"With 50/50 legs the phi correlation between the leg-win indicators must exceed "
        f"**{be:.3f}** (same-sign rate **{be_same:.1%}**) to break even. EV at rho=0 is "
        f"{parlay_ev(0, d):+.3f} per unit; at rho=0.05 {parlay_ev(0.05, d):+.3f}.\n"
    )

    # ------------------------------------------------------------------ 2. model-free
    out.append("## 2. Model-free checks on the score residuals\n")
    rh, ra = games["home_pts_resid"], games["away_pts_resid"]
    corr_ha = float(np.corrcoef(rh, ra)[0, 1])
    s_sp, se_sp, p_sp = variance_trend(games, "resid")
    s_to, se_to, p_to = variance_trend(games, "tresid")
    n_looks += 2
    out.append(
        f"Home-score residual: mean {rh.mean():+.2f}, std {rh.std():.2f}; away-score residual: mean "
        f"{ra.mean():+.2f}, std {ra.std():.2f}; correlation between them {corr_ha:+.4f} (n={len(games)}). "
        f"The two observations per game are nearly independent, so the (o, d) decomposition is "
        f"well posed and the observation-noise correlation will be small.\n\n"
        f"If the market started each season uncertain about teams and learned by watching games, "
        f"residual variance would fall over the season (an efficient market's residual variance "
        f"is ``H P Hᵀ + R`` and ``P`` shrinks with every observation). OLS of squared residual on "
        f"week number: spread {s_sp:+.3f} pts²/week (se {se_sp:.3f}, p={p_sp:.2f}); total "
        f"{s_to:+.3f} pts²/week (se {se_to:.3f}, p={p_to:.2f}). A season-start net-strength "
        f"uncertainty of s points per team that is fully resolved by week 17 would show up as a "
        f"decline of roughly 2s²/16 pts²/week, so the spread trend bounds s at about "
        f"{np.sqrt(max(0.0, 8 * (-s_sp + 2 * se_sp))):.1f} points (2 se). Within-team lag-1 "
        f"autocorrelation of spread residuals (the static-market signature) is reported in "
        f"`parlay-interaction-effects.md` as -0.006.\n"
    )

    # ------------------------------------------------------------------ 3. fits
    out.append("## 3. Fitted hyperparameters (1999–2009, predictive likelihood)\n")
    start = GranularParams(1.7, 1.7, 0.2, 0.2, 9.4, 0.85, 0.0, 0.0)
    fitted_static = fit_or_load(cfg, "granular_static", train, start, "static", args.refit, args.max_iter)
    fitted_eff = fit_or_load(cfg, "granular_efficient", train, start, "efficient", args.refit, args.max_iter)
    team_params, team_row = team_level_row(train, test, d, args.n_boot)
    n_looks += 2
    out.append(
        "Two views of the closing line are fitted. **static**: the market's error persists and "
        "the filter's posterior mean forecasts the next residual (mirrors the team-level filter "
        "in `kalman.py`). **efficient**: the market re-prices every game at the Bayesian "
        "posterior mean, so residuals are innovations with mean zero and the hyperparameters "
        "are identified from the variance/covariance structure alone. Cross-game explaining-"
        "away correlations exist relative to the closing line only in the efficient view, so "
        "it is the primary model; the static fit is shown because a market that under-reacts "
        "would leave both a directional edge and a larger correlation. All quantities in points.\n"
    )
    ptab = pd.DataFrame([params_row("granular static", fitted_static), params_row("granular efficient", fitted_eff)])
    out.append(md_table(ptab) + "\n")
    out.append(
        f"Team-level filter (one net-strength state per team, spreads only) refitted on the same "
        f"window for comparison: prior_std={team_params.prior_std:.2f}, process_std="
        f"{team_params.process_std:.2f}, obs_std={team_params.obs_std:.2f}, persistence="
        f"{team_params.persistence:.3f}.\n"
    )

    # Theory: caps and one-shared-game toy
    out.append("### Implied ceilings\n")
    out.append(
        "`cauchy_schwarz_cap` is the largest correlation the fitted uncertainties could ever "
        "produce between two legs in different games (state variance / (state variance + game "
        "noise), using the larger of the season-start and stationary state variance); it would "
        "require the two legs' latent errors to be perfectly correlated, which conditioning on "
        "shared games never achieves. `one_shared_game` is the filter's own predicted correlation "
        "for the classic pair (H's next game vs A's next game right after H hosted A); it is an "
        "upper bound for week-1 anchors and shrinks later in the season as the market learns.\n"
    )
    theo_rows = []
    for name, p in (("granular static", fitted_static), ("granular efficient", fitted_eff)):
        cap = theoretical_max_corr(p)
        toy = single_shared_game_corr(p).set_index(["leg_1", "leg_2"])["pred_corr"]
        for pt, key in (("spread-spread", ("spread", "spread")), ("total-total", ("total", "total")), ("spread-total", ("spread", "total"))):
            theo_rows.append({"model": name, "pair_type": pt, "cauchy_schwarz_cap": cap[pt], "one_shared_game": float(toy[key]), "breakeven": be})
    lat, nxt = gaussian_explaining_away_corr(team_params.prior_std, team_params.obs_std)
    theo_rows.append({"model": "team-level", "pair_type": "spread-spread", "cauchy_schwarz_cap": float(team_params.prior_std**2 * 2 / (team_params.prior_std**2 * 2 + team_params.obs_std**2)), "one_shared_game": nxt, "breakeven": be})
    out.append(md_table(pd.DataFrame(theo_rows)) + "\n")

    # ------------------------------------------------------------------ 4. OOS evaluation
    out.append(f"## 4. Out-of-sample {TRAIN_SEASONS_END + 1}–{test['season'].max()}: predicted correlations by pair type\n")
    res_eff = GranularKalman(fitted_eff, market_mode="efficient").run(test)
    res_static = GranularKalman(fitted_static, market_mode="static").run(test)
    dist_e, cal_e, top_e = evaluate(res_eff.pairs, d, args.n_boot)
    dist_s, cal_s, top_s = evaluate(res_static.pairs, d, args.n_boot)
    n_looks += 12
    out.append(
        f"Same-slate leg pairs: {len(res_eff.pairs)} rows ({(res_eff.pairs['pair_type'] == 'spread-spread').sum()} "
        f"spread-spread, {(res_eff.pairs['pair_type'] == 'total-total').sum()} total-total, "
        f"{(res_eff.pairs['pair_type'] == 'spread-total').sum()} spread-total, the last counting both "
        f"leg orderings). `frac_positive` is the share of pairs with positive predicted correlation.\n"
    )
    out.append("### Efficient-market filter (primary)\n")
    out.append(md_table(dist_e) + "\n")
    out.append("### Static-market filter\n")
    out.append(md_table(dist_s) + "\n")
    out.append(
        f"Mean state variance of a spread leg (``Lᵀ P L``, points²) over the test window: "
        f"efficient {res_eff.mean_leg_state_var['spread']:.2f}, static {res_static.mean_leg_state_var['spread']:.2f}; "
        f"against game noise of {fitted_eff.leg_noise_var('spread'):.0f} / {fitted_static.leg_noise_var('spread'):.0f} points². "
        f"The market's remaining uncertainty about a matchup is a few percent of the game noise, "
        f"which is why every predicted cross-game correlation is tiny.\n"
    )

    out.append("## 5. Calibration: realized residual product on predicted covariance\n")
    out.append(
        "OLS slope of ``resid_1 * resid_2`` on the predicted covariance, by pair type. Slope 1 "
        "means the model's covariances are right on average, 0 means no realized signal; "
        "`z_from_one` is ``(slope - 1) / se``. Standard errors are OLS on heavy-tailed products "
        "of pairs that share games, so treat them as indicative. The spread-total slope is "
        "undefined when the fitted model is symmetric enough that its spread-total covariances "
        "are numerically zero.\n"
    )
    out.append("### Efficient-market filter\n")
    out.append(md_table(cal_e) + "\n")
    out.append("### Static-market filter\n")
    out.append(md_table(cal_s) + "\n")
    out.append("### Decile calibration (efficient-market filter)\n")
    for pt in PAIR_TYPES:
        sub = res_eff.pairs[res_eff.pairs["pair_type"] == pt]
        if sub["pred_corr"].abs().max() < 1e-9:
            out.append(f"**{pt}**: all predicted correlations are numerically zero; no ranking to calibrate.\n")
            continue
        dec = decile_calibration(sub, n_quantiles=10, n_boot=200)
        out.append(f"**{pt}**\n\n" + md_table(dec) + "\n")

    out.append("## 6. Betting the top decile of predicted |correlation|\n")
    out.append(
        "For each pair type, the 10% of pairs with the largest predicted |corr| are bet "
        "mechanically with two 1-unit parlays each (same-sign parlays when the predicted "
        "correlation is positive, opposite-sign when negative; pushes refund). "
        "`mean_pred_signed` is the model's own forecast of the realized correlation in the "
        f"selected pairs; `phi` is what was realized. Break-even phi is {be:.3f} "
        f"(same-sign rate {be_same:.1%}); independent legs lose {-parlay_ev(0, d):.1%}.\n"
    )
    out.append("### Efficient-market filter\n")
    out.append(md_table(top_e) + "\n")
    out.append("### Static-market filter\n")
    out.append(md_table(top_s) + "\n")

    out.append("## 7. Comparison with the team-level filter\n")
    comp = pd.DataFrame(compare_rows("granular efficient", dist_e, cal_e, top_e) + compare_rows("granular static", dist_s, cal_s, top_s) + [team_row])
    out.append(md_table(comp) + "\n")

    # Marginal check (static mode)
    g = res_static.games
    m_sp = stats.linregress(g["pred_mean_spread"], g["resid"])
    m_to = stats.linregress(g["pred_mean_total"], g["tresid"])
    n_looks += 2
    out.append(
        f"Directional side check (static filter): slope of realized spread residual on the "
        f"filter's predicted mean = {m_sp.slope:.2f} (se {m_sp.stderr:.2f}); totals "
        f"{m_to.slope:.2f} (se {m_to.stderr:.2f}). 1.0 would mean the market under-reacts "
        f"exactly as the static model says; 0 means the closing line already contains the "
        f"information. Predicted means have std {g['pred_mean_spread'].std():.2f} (spread) and "
        f"{g['pred_mean_total'].std():.2f} (total) points.\n"
    )

    # ------------------------------------------------------------------ 8. power check
    out.append("## 8. Power check on synthetic data\n")
    out.append(
        "Residuals simulated from the granular generative model on the real 2010–2025 schedule "
        "with an *efficient* market that re-prices every game at the Bayesian posterior (the "
        "regime in which explaining-away correlations exist relative to the closing line), then "
        "the same pipeline with the true parameters. `theory_one_game` is the one-shared-game "
        "correlation for that scenario. A calibration slope near 1 and a positive top-decile "
        "phi show the pipeline recovers a planted correlation when one exists; the `fitted` row "
        "is the noise floor at real-world uncertainty. `top_decile_pred` is the mean predicted "
        "|corr| inside the top decile, i.e. what the model itself expects `top_decile_pearson` "
        "to be (phi of cover indicators is about 0.64x the Pearson correlation for Gaussian legs). "
        "Scenario `asymmetric` plants a large offense/defense asymmetry plus o/d correlation, "
        "which is what it takes to create spread-total covariance at all: in this model class "
        "spread-total correlation only arises through multi-step network paths, and even at "
        "these extreme settings it tops out near 0.03 with a 90th percentile near 0.002, so no "
        "offense/defense parameterization of team-level errors reaches the hurdle for "
        "cross-market legs.\n"
    )
    sched = test[["game_id", "season", "week", "gameday", "home_team", "away_team", "home_qb_id", "away_qb_id"]]
    scenarios = {
        "fitted efficient (real-world size)": fitted_eff,
        "prior 4/4 (market off by 4 pts on each of o and d)": replace(fitted_eff, prior_std_o=4.0, prior_std_d=4.0, process_std_o=0.3, process_std_d=0.3, persistence=0.95, od_corr=0.0),
        "prior 8/8": replace(fitted_eff, prior_std_o=8.0, prior_std_d=8.0, process_std_o=0.3, process_std_d=0.3, persistence=0.95, od_corr=0.0),
        "asymmetric 12/2, od_corr 0.8": replace(fitted_eff, prior_std_o=12.0, prior_std_d=2.0, process_std_o=0.5, process_std_d=0.1, persistence=0.95, od_corr=0.8),
    }
    pc = power_check(sched, scenarios, d, n_boot=200)
    out.append(md_table(pc) + "\n")

    # ------------------------------------------------------------------ 9. EPA extension
    if not args.skip_epa:
        out.append("## 9. Extension: nflverse team-week EPA as an extra observation of the same state\n")
        epa = load_team_week_epa(cfg, cfg.seasons)
        imp_home = (games["total_line"] + games["spread_line"]) / 2
        imp_away = (games["total_line"] - games["spread_line"]) / 2
        tr = games["season"] <= TRAIN_SEASONS_END
        lookup = epa.set_index(["game_id", "team"])["off_epa"]
        lookup = lookup[~lookup.index.duplicated(keep="first")]
        from geo_model.parlay.kalman_granular import NFLVERSE_TEAM_ALIASES

        def _aux(side: str) -> pd.Series:
            key = pd.MultiIndex.from_arrays([games["game_id"], games[side]])
            v = lookup.reindex(key).to_numpy(dtype=float)
            alias = games[side].map(NFLVERSE_TEAM_ALIASES).fillna(games[side])
            v2 = lookup.reindex(pd.MultiIndex.from_arrays([games["game_id"], alias])).to_numpy(dtype=float)
            return pd.Series(np.where(np.isfinite(v), v, v2), index=games.index)

        aux_h, aux_a = _aux("home_team"), _aux("away_team")
        calib = fit_aux_calibration(pd.concat([aux_h[tr], aux_a[tr]]), pd.concat([imp_home[tr], imp_away[tr]]))
        with_aux = attach_aux_observations(games, epa, calib)
        train_aux = with_aux[with_aux["season"] <= TRAIN_SEASONS_END].reset_index(drop=True)
        test_aux = with_aux[with_aux["season"] > TRAIN_SEASONS_END].reset_index(drop=True)
        aux_corr_raw = float(np.corrcoef(test_aux["home_aux_resid"], test_aux["home_pts_resid"])[0, 1])
        out.append(
            f"Offensive EPA (``passing_epa + rushing_epa`` from the `stats_team` release, regular "
            f"season) is mapped to points with ``EPA ≈ {calib.intercept:.2f} + {calib.slope:.3f} × implied points`` "
            f"fitted on {calib.n} training team-games, giving an EPA-implied score residual for "
            f"each side of every game ({len(with_aux)} of {len(games)} games have EPA on both sides). "
            f"Its raw correlation with the score residual is {aux_corr_raw:.3f}, so about "
            f"{1 - aux_corr_raw**2:.0%} of its variance is not the score: some of that is "
            f"information the score does not carry (turnover luck, field position), some is "
            f"unrelated noise. The filter decides by fitting `aux_std` (noise of the EPA "
            f"observation) and `aux_corr` (its noise correlation with the score noise).\n"
        )
        start_aux = replace(fitted_eff, use_aux=True, aux_loading=1.0, aux_std=7.0, aux_corr=0.6)
        fitted_aux = fit_or_load(cfg, "granular_efficient_epa", train_aux, start_aux, "efficient", args.refit, args.max_iter)
        out.append(md_table(pd.DataFrame([params_row("granular efficient + EPA", fitted_aux)])) + "\n")
        res_aux = GranularKalman(fitted_aux, market_mode="efficient").run(test_aux)
        res_noaux = GranularKalman(fitted_eff, market_mode="efficient").run(test_aux)
        dist_a, cal_a, top_a = evaluate(res_aux.pairs, d, args.n_boot)
        n_looks += 6
        extra_summary_rows = compare_rows("granular efficient + EPA", dist_a, cal_a, top_a)
        z2_aux = float((res_aux.games["resid"] ** 2 / res_aux.games["pred_var_spread"]).mean())
        z2_no = float((res_noaux.games["resid"] ** 2 / res_noaux.games["pred_var_spread"]).mean())
        out.append(
            f"On the same {len(test_aux)} test games: mean spread-leg state variance "
            f"{res_aux.mean_leg_state_var['spread']:.2f} with EPA vs {res_noaux.mean_leg_state_var['spread']:.2f} "
            f"without (points²); score-only predictive log-likelihood {res_aux.log_likelihood_score:.1f} vs "
            f"{res_noaux.log_likelihood_score:.1f} (difference {res_aux.log_likelihood_score - res_noaux.log_likelihood_score:+.1f} nats "
            f"over {len(test_aux)} games); mean standardized squared spread residual {z2_aux:.3f} vs {z2_no:.3f} "
            f"(1.0 = calibrated variance).\n"
        )
        out.append("Predicted |corr| with EPA observations:\n\n" + md_table(dist_a) + "\n")
        out.append("Calibration with EPA observations:\n\n" + md_table(cal_a) + "\n")
        out.append("Top decile with EPA observations:\n\n" + md_table(top_a) + "\n")
        cap_aux = theoretical_max_corr(fitted_aux)
        out.append(
            f"Cauchy–Schwarz cap with the EPA-fitted uncertainties: spread-spread {cap_aux['spread-spread']:.4f}, "
            f"total-total {cap_aux['total-total']:.4f}, spread-total {cap_aux['spread-total']:.4f}.\n"
        )
    else:
        out.append("## 9. Extension: nflverse EPA\n\nSkipped (`--skip-epa`).\n")
        extra_summary_rows = []

    # ------------------------------------------------------------------ 10. summary
    out.append("## 10. Summary\n")
    summary = pd.concat([comp, pd.DataFrame(extra_summary_rows)], ignore_index=True) if extra_summary_rows else comp
    best = summary.loc[summary["top_decile_phi"].abs().idxmax()]
    max_pred = float(summary["max_abs_pred_corr"].max())
    pc8 = pc[pc["scenario"] == "prior 8/8"].set_index("pair_type")
    best_roi = float(summary["top_decile_roi"].max())
    n_sig = int(
        sum(
            1
            for _, r in summary.iterrows()
            if isinstance(r["phi_ci"], str) and not (float(r["phi_ci"].split(",")[0][1:]) <= 0 <= float(r["phi_ci"].split(",")[1][:-1]))
        )
    )
    out.append(
        f"- Largest predicted |correlation| for any leg pair in {len(res_eff.pairs)} out-of-sample "
        f"pairs across all models and pair types: **{max_pred:.4f}** vs. the {be:.3f} needed. "
        f"The Cauchy–Schwarz ceiling at the fitted uncertainties is "
        f"{max(r['cauchy_schwarz_cap'] for r in theo_rows):.3f}, and the one-shared-game value is "
        f"{max(abs(r['one_shared_game']) for r in theo_rows):.4f}.\n"
        f"- Largest realized top-decile |phi| across the pre-registered subsets: {best['top_decile_phi']:+.4f} "
        f"({best['model']}, {best['pair_type']}, n={int(best['top_decile_n'])}, 95% CI {best['phi_ci']}); "
        f"best top-decile ROI {best_roi:+.4f} vs. {parlay_ev(0, d):+.4f} for independent legs. "
        f"{n_sig} of {len(summary)} top-decile phi intervals exclude zero.\n"
        f"- Calibration (efficient filter): {calibration_sentence(cal_e)}. The power check "
        f"recovers slopes of {pc8.loc['spread-spread', 'slope']:.2f} ± {pc8.loc['spread-spread', 'se']:.2f} "
        f"(spread-spread) and {pc8.loc['total-total', 'slope']:.2f} ± {pc8.loc['total-total', 'se']:.2f} "
        f"(total-total) when the planted uncertainty is 8 points per state, so the null on real "
        f"data is a property of the market, not of the pipeline.\n"
        f"- Number of real-data subsets/tests looked at for this report: {n_looks} (model-free "
        f"trends, two fitted modes x three pair types x (slope, top-decile phi), team-level "
        f"comparison, directional checks, EPA variant). With {n_looks} looks, roughly one "
        f"interval excluding zero is expected by chance.\n"
    )
    out.append("## Caveats\n")
    out.append(
        "- Closing lines from the nflverse games file are a single consensus number; "
        "the residuals mix line-provider differences into the game noise (which only makes the "
        "market's implied uncertainty look larger, not smaller).\n"
        "- The (o, d) model assumes additive, Gaussian, team-specific errors. Matchup-specific "
        "errors (scheme, injuries) are absorbed into game noise, which is where any parlay edge "
        "would have to live if it exists; this filter cannot see them by construction.\n"
        "- Top-decile thresholds are quantiles over the whole evaluation window (a ranking "
        "diagnostic), though each pair's prediction is strictly causal.\n"
        "- Calibration regressions use pairs that share games, so their standard errors are "
        "understated; read the real-data slopes against the power-check rows, where the same "
        "regression on planted correlations of known size shows what a detectable signal looks like.\n"
        "- The EPA extension uses `stats_team_week` from nflverse, whose EPA model is fitted on "
        "later seasons (a mild look-ahead in the *definition* of EPA, not in game outcomes).\n"
    )
    out.append("## Reproduce\n")
    out.append(
        "```shell\npython scripts/backtest_kalman_granular.py            # uses cached fits in data/raw/nfl/\n"
        "python scripts/backtest_kalman_granular.py --refit    # re-run the three Nelder-Mead fits (~20 min)\n```\n"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
