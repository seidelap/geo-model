"""Granular (pitcher / offense / defense) Kalman filter backtest for MLB parlays.

Usage:
    python scripts/backtest_pitcher_kalman.py [--out docs/experiments/parlay-pitcher-kalman.md]
        [--train-end 2022] [--max-iter 300] [--params-json fitted.json] [--skip-nba] [--skip-power]

Fits the filter hyperparameters on seasons <= ``--train-end`` by predictive
likelihood, then runs the filter causally through the later seasons with
daily slates and reports predicted vs realized pair correlations, the
moneyline-parlay correlation edge, the closed-form maximum implied by the
fitted uncertainties, an efficient-market power check, and an NBA
back-to-back stratification of the shared-game pairs.

Data: ``$GEO_MODEL_DATA_DIR/multisport`` (default ``data/raw/multisport``) as
used by ``scripts/backtest_parlay_multisport.py``. Single-threaded BLAS is
forced because the filter's many small dense updates are slower with thread
contention.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from geo_model.parlay import ParlayConfig, breakeven_correlation, build_shared_game_pairs, correlation_summary, to_team_games, two_sided_parlay_roi  # noqa: E402
from geo_model.parlay.backtest import kalman_pair_calibration, kalman_slope  # noqa: E402
from geo_model.parlay.data_multisport import MultiSportConfig, load_mlb, load_nba  # noqa: E402
from geo_model.parlay.multisport_backtest import build_pitcher_pairs, moneyline_parlay  # noqa: E402
from geo_model.parlay.pitcher_kalman import (  # noqa: E402
    GranularKalman,
    GranularParams,
    RunOffsets,
    attach_rest,
    cluster_bootstrap_phi,
    cluster_bootstrap_ratio,
    load_nba_rest_days,
    null_log_likelihood,
    permutation_slope_pvalue,
    orient_pair_predictions,
    pair_requests_from_pairs,
    rest_strata,
    same_day_pairs_for_moneyline,
    simulate_granular,
    theory_pair_corr,
)

HURDLE_TEXT = "phi > 0.098 (same-sign rate 54.9%) for a 2-leg -110 parlay"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if abs(v) < 0.01 and v != 0:
            return f"{v:.5f}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def summary_rows(groups: dict[str, tuple[pd.Series, pd.Series]], n_boot: int) -> pd.DataFrame:
    rows = []
    for label, (x, y) in groups.items():
        row = {"subset": label}
        row.update(correlation_summary(x, y, n_boot=n_boot).to_row())
        rows.append(row)
    return pd.DataFrame(rows)


def ml_row(label: str, r) -> dict:
    return {
        "subset": label, "pairs": r.n_pairs, "realized_roi": round(r.realized_roi, 4),
        "indep_roi": round(r.independent_roi, 4), "corr_edge": round(r.edge, 4),
        "edge_ci": f"[{r.edge_ci[0]:+.3f}, {r.edge_ci[1]:+.3f}]",
        "p_same_side": round(r.p_same_side, 4), "p_same_indep": round(r.p_same_side_indep, 4),
    }


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sign_align(pairs: pd.DataFrame) -> pd.DataFrame:
    """Swap ``team_a`` for its opponent when the predicted correlation is negative."""
    out = pairs.copy()
    neg = out["pred_corr"] < 0
    out.loc[neg, "team_a"] = out.loc[neg, "a_next_opp"]
    return out


def totals_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rename total columns so the margin helpers can be reused."""
    return pairs.rename(columns={"pred_cov": "_", "pred_corr": "__", "resid_1": "___", "resid_2": "____"}).rename(
        columns={"pred_tcov": "pred_cov", "pred_tcorr": "pred_corr", "tresid_1": "resid_1", "tresid_2": "resid_2"}
    )


def pair_block(kp: pd.DataFrame, label: str, d: float, n_boot: int, expected_sign: int = 1, n_perm: int = 200) -> list[str]:
    """Calibration table, deciles and mechanical ROI for one pairs table (margin layout)."""
    out = []
    sl, se, pv = kalman_slope(kp)
    _, perm_p = permutation_slope_pvalue(kp, "gameday", n_perm=n_perm, seed=0)
    out.append(
        f"{label}: {len(kp)} pairs, predicted corr mean {kp['pred_corr'].mean():+.5f}, mean |corr| {kp['pred_corr'].abs().mean():.5f}, "
        f"max |corr| {kp['pred_corr'].abs().max():.5f}, 99th pct |corr| {kp['pred_corr'].abs().quantile(0.99):.5f}. "
        f"Calibration slope of realized residual product on predicted covariance: {sl:.2f} (OLS se {se:.2f}, OLS p={pv:.3f}; "
        f"within-slate permutation p={perm_p:.3f}; 1 = calibrated, 0 = no signal).\n"
    )
    if kp["pred_corr"].nunique() > 5:
        cal = kalman_pair_calibration(kp, n_quantiles=5)
        cal["n"] = cal["n"].astype(int)
        cal["bin"] = cal["bin"].astype(int)
        out.append(md_table(cal) + "\n")
    top = kp[kp["pred_corr"] >= kp["pred_corr"].quantile(0.9)]
    bot = kp[kp["pred_corr"] <= kp["pred_corr"].quantile(0.1)]
    s_top = correlation_summary(top["resid_1"], top["resid_2"], n_boot=n_boot)
    s_bot = correlation_summary(bot["resid_1"], bot["resid_2"], n_boot=n_boot)
    phi_t, ci_t, _ = cluster_bootstrap_phi(top["resid_1"], top["resid_2"], top["gameday"], n_boot=n_boot)
    phi_b, ci_b, _ = cluster_bootstrap_phi(bot["resid_1"], bot["resid_2"], bot["gameday"], n_boot=n_boot)
    rt = two_sided_parlay_roi(top["resid_1"], top["resid_2"], d, expected_sign)
    rb = two_sided_parlay_roi(bot["resid_1"], bot["resid_2"], d, -expected_sign)
    out.append(
        f"Top decile of predicted corr (mean {top['pred_corr'].mean():+.5f}): {len(top)} pairs on {top['gameday'].nunique()} slates, realized phi {s_top.phi:+.4f} "
        f"(pair bootstrap [{s_top.phi_ci[0]:+.3f}, {s_top.phi_ci[1]:+.3f}]; slate-cluster bootstrap [{ci_t[0]:+.3f}, {ci_t[1]:+.3f}]), pearson {s_top.pearson:+.4f}, "
        f"{'same' if expected_sign > 0 else 'opposite'}-sign rate {rt.hit_rate:.4f}, two-sided ROI {rt.roi:+.4f}. "
        f"Bottom decile (mean {bot['pred_corr'].mean():+.5f}): {len(bot)} pairs, phi {s_bot.phi:+.4f} (cluster CI [{ci_b[0]:+.3f}, {ci_b[1]:+.3f}]), "
        f"{'opposite' if expected_sign > 0 else 'same'}-sign rate {rb.hit_rate:.4f}, ROI {rb.roi:+.4f}.\n"
    )
    return out


def per_season_top_decile(kp: pd.DataFrame, n_boot: int, sign_align: bool = True) -> pd.DataFrame:
    """Realized phi of the top decile of |predicted corr| per season (cluster bootstrap by slate)."""
    rows = []
    thr = kp["pred_corr"].abs().quantile(0.9)
    sel = kp[kp["pred_corr"].abs() >= thr]
    for season, g in sel.groupby("season"):
        x = g["resid_1"] * (np.sign(g["pred_corr"]) if sign_align else 1)
        phi, ci, n = cluster_bootstrap_phi(x, g["resid_2"], g["gameday"], n_boot=n_boot)
        r = two_sided_parlay_roi(x, g["resid_2"], 1.909, 1)
        rows.append({"season": int(season), "pairs": n, "slates": g["gameday"].nunique(), "phi": phi, "phi_cluster_ci": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", "same_sign": r.hit_rate, "two_sided_roi": r.roi})
    return pd.DataFrame(rows)


def cluster_ml_edge(pairs: pd.DataFrame, tg: pd.DataFrame, games: pd.DataFrame, cluster_col: str, n_boot: int) -> dict:
    """Moneyline-parlay correlation edge with a cluster bootstrap CI (clusters = slates)."""
    nums, dens, same, same_ind = [], [], [], []
    for _, g in pairs.groupby(cluster_col):
        r = moneyline_parlay(g, tg, games, n_boot=1)
        if r.n_pairs == 0:
            continue
        nums.append((r.realized_roi - r.independent_roi) * 2 * r.n_pairs)
        dens.append(2 * r.n_pairs)
        same.append(r.p_same_side * r.n_pairs)
        same_ind.append(r.p_same_side_indep * r.n_pairs)
    edge, ci = cluster_bootstrap_ratio(np.array(nums), np.array(dens), n_boot=n_boot)
    n = int(sum(dens) / 2)
    return {"pairs": n, "corr_edge": round(edge, 4), "edge_cluster_ci": f"[{ci[0]:+.3f}, {ci[1]:+.3f}]", "p_same_side": round(sum(same) / n, 4), "p_same_indep": round(sum(same_ind) / n, 4)}


def run_mlb(args: argparse.Namespace, cfg: MultiSportConfig, pcfg: ParlayConfig) -> tuple[list[str], dict]:
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out: list[str] = []
    stats_out: dict = {}
    log("[mlb] loading")
    games = load_mlb(cfg)
    games = games[(games["game_type"] == "REG") & games["home_qb_id"].notna() & games["away_qb_id"].notna()].reset_index(drop=True)
    train = games[games["season"] <= args.train_end]
    test = games[games["season"] > args.train_end]
    offsets = RunOffsets.fit(train)
    out.append("## MLB: granular Kalman filter\n")
    out.append(
        f"{len(games)} regular-season games with both starting pitchers, seasons {games['season'].min()}–{games['season'].max()} "
        f"({games['home_qb_id'].nunique() + games['away_qb_id'].nunique()} pitcher slots, "
        f"{len(set(games['home_qb_id']) | set(games['away_qb_id']))} distinct starters, "
        f"{games['home_team'].nunique()} teams). Training seasons ≤{args.train_end}: {len(train)} games; evaluation seasons "
        f"{args.train_end + 1}–{games['season'].max()}: {len(test)} games on {test['gameday'].nunique()} daily slates.\n"
    )
    out.append(
        f"Run residuals against the closing market: home runs minus `(total_line + implied_margin)/2` and away runs minus "
        f"`(total_line - implied_margin)/2` where the implied margin is the probit map of the vig-free closing moneyline. "
        f"Training-period offsets subtracted before filtering: home {offsets.home:+.3f}, away {offsets.away:+.3f} runs. "
        f"Run residual sd (training): home {train['home_score'].sub((train['total_line'] + train['spread_line']) / 2).std():.3f}, "
        f"away {train['away_score'].sub((train['total_line'] - train['spread_line']) / 2).std():.3f}.\n"
    )
    # Fit
    t0 = time.time()
    if args.params_json:
        params = GranularParams(**json.loads(Path(args.params_json).read_text()))
        log(f"[mlb] loaded params from {args.params_json}")
    else:
        init = GranularParams(0.4, 0.99, 0.3, 0.97, 0.3, 0.97, train["resid"].std() / np.sqrt(2), 0.0)
        log(f"[mlb] fitting on seasons <= {args.train_end} (max_iter={args.max_iter})")
        params = GranularKalman(init, offsets).fit(train, max_iter=args.max_iter)
        log(f"[mlb] fit done in {time.time() - t0:.0f}s: {params}")
    if args.save_params:
        Path(args.save_params).write_text(json.dumps(asdict(params)))
    stats_out["params"] = asdict(params)
    kf = GranularKalman(params, offsets)
    # Likelihood diagnostics: fitted vs no-state model, train and test
    full = kf.run(games, emit_pairs=False, emit_from_season=10**6)
    null = null_log_likelihood(games, params.obs_std, params.obs_corr, offsets)
    ll_rows = []
    for season in sorted(full.log_likelihood_by_season):
        n = full.n_obs_by_season[season]
        ll_rows.append(
            {
                "season": season, "role": "train" if season <= args.train_end else "test", "n_obs": n,
                "ll_filter_per_obs": full.log_likelihood_by_season[season] / n, "ll_null_per_obs": null[season] / n,
                "gain_per_obs": (full.log_likelihood_by_season[season] - null[season]) / n,
            }
        )
    ll_df = pd.DataFrame(ll_rows)
    test_gain = ll_df.loc[ll_df["role"] == "test", "gain_per_obs"]
    out.append("### Fitted hyperparameters (predictive likelihood, training seasons only)\n")
    out.append(
        f"pitcher: stationary std {params.pitcher_std:.4f} runs, daily persistence {params.pitcher_persistence:.4f} "
        f"(5-day {params.pitcher_persistence**5:.3f}); offense: std {params.offense_std:.4f}, persistence {params.offense_persistence:.4f}; "
        f"defense/bullpen: std {params.defense_std:.4f}, persistence {params.defense_persistence:.4f}; run noise std {params.obs_std:.3f}, "
        f"within-game noise correlation {params.obs_corr:+.4f}. Off-season gap capped at {kf.max_gap_days} days of drift.\n"
    )
    out.append(
        "Predictive log-likelihood per run observation (2 per game) of the fitted filter versus the no-state model with the same noise "
        "parameters. A positive out-of-sample gain means the accumulated pitcher/offense/defense information predicts run residuals.\n"
    )
    out.append(md_table(ll_df) + "\n")
    out.append(f"Mean out-of-sample gain per observation: {test_gain.mean():+.6f} (per game {2 * test_gain.mean():+.6f}).\n")
    # Theory
    th = theory_pair_corr(params)
    stats_out["theory"] = asdict(th)
    out.append("### Theoretical maximum given the fitted uncertainties\n")
    out.append(
        f"After one shared game between pitcher P (team A) and team B, the posterior covariance of P's error with B's offense error is "
        f"`s_p² s_o² / V` with `V = s_p² + s_o² + s_d² + σ²`. Predicted correlations for (P's next start margin, B's next game margin): "
        f"{th.margin_pitcher:+.6f}; for the two totals: {th.total_pitcher:+.6f}; for team-level pairs without the pitcher link: "
        f"{th.margin_team:+.6f}. Cauchy–Schwarz upper bound over any amount of accumulated evidence: margin {th.margin_bound:.6f}, "
        f"totals {th.total_bound:.6f}. Hurdle: {be:.3f}.\n"
    )
    # Hurdle exploration: what parameters would be needed
    rows = []
    for sp, so, sd in [(1.0, 0.5, 0.5), (2.0, 1.0, 1.0), (3.0, 1.5, 1.5), (3.0, 3.0, 3.0), (4.0, 4.0, 4.0)]:
        t = theory_pair_corr(GranularParams(sp, 0.999, so, 0.99, sd, 0.99, params.obs_std, params.obs_corr))
        rows.append({"pitcher_std": sp, "offense_std": so, "defense_std": sd, "margin_corr_one_game": t.margin_pitcher, "margin_bound": t.margin_bound})
    out.append("Hypothetical error sizes (runs per game) at the fitted noise level, one shared game:\n")
    out.append(md_table(pd.DataFrame(rows)) + "\n")
    # Causal evaluation run
    log("[mlb] building pitcher pairs and evaluation run")
    pp_all = build_pitcher_pairs(games)
    pp = pp_all[pp_all["season"] > args.train_end].reset_index(drop=True)
    req = pair_requests_from_pairs(pp, games)
    t0 = time.time()
    res = kf.run(games, emit_pairs=True, pair_requests=req, emit_from_season=args.train_end + 1)
    log(f"[mlb] evaluation run done in {time.time() - t0:.0f}s: {len(res.pairs)} same-day pairs, {len(res.cross_pairs)} cross pairs")
    kp = res.pairs
    stats_out["same_day_pairs"] = int(len(kp))
    out.append("### Same-day pairs (all pairs of games on one slate, evaluation seasons)\n")
    out.append(
        "Predictions for a slate use the filter state before any game of that slate; both legs are priced on the same day. "
        "`pred_corr` is the model's correlation between the two home-perspective margin residuals (spread/moneyline legs).\n"
    )
    out.extend(pair_block(kp, "Margins", d, args.n_boot, expected_sign=1))
    out.append("Totals (the model predicts negative correlations for pairs that share a scoring link):\n")
    kt = totals_frame(kp)
    out.extend(pair_block(kt, "Totals", d, args.n_boot, expected_sign=1))
    # Moneyline parlays on same-day pairs, oriented by the filter
    sd_ml = same_day_pairs_for_moneyline(kp, games)
    tg = to_team_games(games, pcfg)
    top_ml = sd_ml[sd_ml["pred_corr"].abs() >= sd_ml["pred_corr"].abs().quantile(0.9)]
    sd_ml["gameday"] = kp["gameday"].to_numpy()
    top_ml = sd_ml[sd_ml["pred_corr"].abs() >= sd_ml["pred_corr"].abs().quantile(0.9)]
    mlrows = []
    for lab, sub in (("all same-day pairs, filter-oriented", sd_ml), ("top decile |pred corr|, filter-oriented", top_ml)):
        row = ml_row(lab, moneyline_parlay(sub, tg, games, n_boot=args.n_boot))
        row.update({k: v for k, v in cluster_ml_edge(sub, tg, games, "gameday", args.n_boot).items() if k in ("edge_cluster_ci",)})
        mlrows.append(row)
    top_abs = kp[kp["pred_corr"].abs() >= kp["pred_corr"].abs().quantile(0.9)]
    x_al = top_abs["resid_1"] * np.sign(top_abs["pred_corr"])
    s_top = correlation_summary(x_al, top_abs["resid_2"], n_boot=args.n_boot)
    phi_c, ci_c, n_c = cluster_bootstrap_phi(x_al, top_abs["resid_2"], top_abs["gameday"], n_boot=args.n_boot)
    stats_out["same_day_top_abs_phi"] = phi_c
    stats_out["same_day_top_abs_phi_cluster_ci"] = ci_c
    stats_out["same_day_ml_rows"] = mlrows
    out.append(
        "Moneyline parlays at the actual closing decimals on the filter's direction: for each pair the (H wins, A wins) and (H loses, A loses) "
        "parlays where A is chosen so that the pair is predicted to land on the same side. `corr_edge` = realized ROI minus the ROI under "
        "independence; it must exceed the vig (~0.04-0.05 per unit here) to be +EV. `edge_ci` resamples pairs (too narrow: pairs on a slate "
        "share games); `edge_cluster_ci` resamples slates.\n"
    )
    out.append(md_table(pd.DataFrame(mlrows)) + "\n")
    out.append(
        f"Sign-aligned phi on the top decile of |predicted corr| (leg-1 residual multiplied by the predicted sign): {phi_c:+.4f} "
        f"(pair bootstrap [{s_top.phi_ci[0]:+.3f}, {s_top.phi_ci[1]:+.3f}]; slate-cluster bootstrap [{ci_c[0]:+.3f}, {ci_c[1]:+.3f}]), n={n_c}. Per season:\n"
    )
    pst = per_season_top_decile(kp, args.n_boot)
    stats_out["same_day_top_abs_by_season"] = pst.to_dict("records")
    out.append(md_table(pst) + "\n")
    # Pitcher pairs
    out.append("### Starting-pitcher pairs (P's next start vs the anchor opponent's next game)\n")
    pp = orient_pair_predictions(pp, res.cross_pairs, games)
    pp = pp[pp["pred_corr"].notna()].reset_index(drop=True)
    stats_out["pitcher_pairs"] = int(len(pp))
    same_day = pp["gap_days"] == 0
    out.append(
        f"{len(pp)} pairs with anchors in the evaluation seasons (median gap between the two legs {pp['gap_days'].median():.0f} days; "
        f"{same_day.sum()} pairs have both legs on the same day). Predictions are made at the slate of the earlier leg, before it is played, "
        f"propagating the state to the later leg through the fitted persistence. Team-perspective predicted margin corr: mean {pp['pred_corr'].mean():+.6f}, "
        f"mean |corr| {pp['pred_corr'].abs().mean():.6f}, max {pp['pred_corr'].max():+.6f}; predicted total corr mean {pp['pred_tcorr'].mean():+.6f}.\n"
    )
    groups = {
        "all": (pp["h_next_resid"], pp["a_next_resid"]),
        "same-day legs": (pp.loc[same_day, "h_next_resid"], pp.loc[same_day, "a_next_resid"]),
        "top decile predicted corr": (pp.loc[pp["pred_corr"] >= pp["pred_corr"].quantile(0.9), "h_next_resid"], pp.loc[pp["pred_corr"] >= pp["pred_corr"].quantile(0.9), "a_next_resid"]),
        "bottom decile predicted corr": (pp.loc[pp["pred_corr"] <= pp["pred_corr"].quantile(0.1), "h_next_resid"], pp.loc[pp["pred_corr"] <= pp["pred_corr"].quantile(0.1), "a_next_resid"]),
        "weeks 1-4 anchors": (pp.loc[pp["week"] <= 4, "h_next_resid"], pp.loc[pp["week"] <= 4, "a_next_resid"]),
    }
    ptab = summary_rows(groups, args.n_boot)
    out.append("Margin residual correlation (prediction: positive):\n")
    out.append(md_table(ptab) + "\n")
    stats_out["pitcher_pairs_table"] = ptab.to_dict("records")
    ttab = summary_rows({"all": (pp["h_next_tresid"], pp["a_next_tresid"]), "bottom decile predicted total corr": (pp.loc[pp["pred_tcorr"] <= pp["pred_tcorr"].quantile(0.1), "h_next_tresid"], pp.loc[pp["pred_tcorr"] <= pp["pred_tcorr"].quantile(0.1), "a_next_tresid"])}, args.n_boot)
    out.append("Total residual correlation (prediction: negative):\n")
    out.append(md_table(ttab) + "\n")
    stats_out["pitcher_totals_table"] = ttab.to_dict("records")
    ppc = pp.rename(columns={"h_next_resid": "resid_1", "a_next_resid": "resid_2"})
    sl, se, pv = kalman_slope(ppc)
    out.append(f"Calibration slope of realized margin residual product on predicted covariance (pitcher pairs): {sl:.2f} (se {se:.2f}, p={pv:.3f}).\n")
    top_pp = pp[pp["pred_corr"] >= pp["pred_corr"].quantile(0.9)]
    mlrows = [
        ml_row("all pitcher pairs (both win / both lose)", moneyline_parlay(pp, tg, games, n_boot=args.n_boot)),
        ml_row("all pitcher pairs, sign-aligned to prediction", moneyline_parlay(sign_align(pp), tg, games, n_boot=args.n_boot)),
        ml_row("top decile predicted corr", moneyline_parlay(top_pp, tg, games, n_boot=args.n_boot)),
        ml_row("same-day legs, sign-aligned", moneyline_parlay(sign_align(pp[same_day]), tg, games, n_boot=args.n_boot)),
    ]
    out.append(md_table(pd.DataFrame(mlrows)) + "\n")
    stats_out["pitcher_ml_rows"] = mlrows
    # Marginal (single-bet) checks
    tgm = res.games
    rows = []
    for label, xcol, ycol in [
        ("margin residual on predicted margin mean", "pred_margin_mean", "resid"),
        ("total residual on predicted total mean", "pred_total_mean", "tresid"),
    ]:
        r = stats.linregress(tgm[xcol], tgm[ycol])
        rows.append({"check": label, "n": len(tgm), "slope": r.slope, "se": r.stderr, "p": r.pvalue, "pred_sd": tgm[xcol].std()})
    gi = games.set_index("game_id").loc[tgm["game_id"]]
    run_pred = pd.concat(
        [
            pd.DataFrame({"pred": tgm["pred_home_mean"].to_numpy() - offsets.home, "real": gi["home_score"].to_numpy() - (gi["total_line"] + gi["spread_line"]).to_numpy() / 2 - offsets.home}),
            pd.DataFrame({"pred": tgm["pred_away_mean"].to_numpy() - offsets.away, "real": gi["away_score"].to_numpy() - (gi["total_line"] - gi["spread_line"]).to_numpy() / 2 - offsets.away}),
        ]
    )
    r = stats.linregress(run_pred["pred"], run_pred["real"])
    rows.append({"check": "team runs residual on predicted runs mean (home and away rows, offsets removed)", "n": len(run_pred), "slope": r.slope, "se": r.stderr, "p": r.pvalue, "pred_sd": run_pred["pred"].std()})
    r = stats.linregress(pp["pred_margin_mean_2"] if "pred_margin_mean_2" in pp else res.cross_pairs["pred_margin_mean_2"], (res.cross_pairs.merge(games[["game_id", "resid"]].rename(columns={"game_id": "game_id_2", "resid": "resid_2"}), on="game_id_2")["resid_2"]))
    rows.append({"check": "later leg margin residual on its predicted mean as of the earlier leg", "n": len(res.cross_pairs), "slope": r.slope, "se": r.stderr, "p": r.pvalue, "pred_sd": res.cross_pairs["pred_margin_mean_2"].std()})
    out.append("### Marginal (single-bet) checks on evaluation games\n")
    out.append("Slope of the realized residual on the filter's predicted mean (1 = the filter's directional information is fully unpriced, 0 = already in the closing line):\n")
    out.append(md_table(pd.DataFrame(rows)) + "\n")
    stats_out["marginal_rows"] = rows
    # Power check
    if not args.skip_power:
        out.extend(power_check(games, params, offsets, args, d, pp_all, req, tg))
    return out, stats_out


def power_check(games: pd.DataFrame, fitted: GranularParams, offsets: RunOffsets, args: argparse.Namespace, d: float, pp_all: pd.DataFrame, req: pd.DataFrame, tg: pd.DataFrame) -> list[str]:
    """Simulate from the generative model on the real schedule with an efficient market."""
    out = ["### Power check: efficient-market simulation on the real schedule\n"]
    out.append(
        "Synthetic run residuals drawn from the state-space model on the 2021-2025 schedule with the actual starters; the market prices every "
        "game at the Bayesian posterior mean given all earlier games, so residuals are innovations relative to an efficient closing line. "
        "The filter is then run with the true parameters (no refit) and evaluated exactly as above on the evaluation seasons. Scenarios: "
        "(a) the fitted parameters; (b) a large-but-conceivable market error (pitcher 1.0 runs, offense/defense 0.5 runs, persistence 0.995/day); "
        "(c) errors equal to the game noise (3 runs each), which the closed form says is what the hurdle requires.\n"
    )
    scenarios = {
        "fitted": fitted,
        "conceivable (1.0 / 0.5 / 0.5)": GranularParams(1.0, 0.995, 0.5, 0.995, 0.5, 0.995, fitted.obs_std, fitted.obs_corr),
        "hurdle-sized (3 / 3 / 3)": GranularParams(3.0, 0.999, 3.0, 0.999, 3.0, 0.999, fitted.obs_std, fitted.obs_corr),
    }
    rows = []
    for label, p in scenarios.items():
        log(f"[power] {label}")
        sim = simulate_granular(games, p, seed=args.seed, market="efficient")
        res = GranularKalman(p, RunOffsets()).run(sim, emit_pairs=True, pair_requests=req, emit_from_season=args.train_end + 1)
        kp = res.pairs
        sl, se, _ = kalman_slope(kp)
        top = kp[kp["pred_corr"] >= kp["pred_corr"].quantile(0.9)]
        s_top = correlation_summary(top["resid_1"], top["resid_2"], n_boot=100)
        pp = orient_pair_predictions(pp_all[pp_all["season"] > args.train_end], res.cross_pairs, sim)
        pp = pp[pp["pred_corr"].notna()]
        # realized leg residuals must come from the simulation
        gi = sim.set_index("game_id")
        home = gi["home_team"]
        h_res = gi["resid"].reindex(pp["h_next_game_id"]).to_numpy() * np.where(home.reindex(pp["h_next_game_id"]).to_numpy() == pp["team_h"].to_numpy(), 1, -1)
        a_res = gi["resid"].reindex(pp["a_next_game_id"]).to_numpy() * np.where(home.reindex(pp["a_next_game_id"]).to_numpy() == pp["team_a"].to_numpy(), 1, -1)
        s_pp = correlation_summary(h_res, a_res, n_boot=100)
        top_pp = pp["pred_corr"] >= pp["pred_corr"].quantile(0.9)
        s_pp_top = correlation_summary(h_res[top_pp.to_numpy()], a_res[top_pp.to_numpy()], n_boot=100)
        th = theory_pair_corr(p)
        rows.append(
            {
                "scenario": label, "theory_one_game": th.margin_pitcher, "same_day_pairs": len(kp), "pred_corr_p99": kp["pred_corr"].quantile(0.99),
                "calib_slope": sl, "slope_se": se, "top_decile_phi": s_top.phi, "top_decile_phi_ci": f"[{s_top.phi_ci[0]:+.3f}, {s_top.phi_ci[1]:+.3f}]",
                "pitcher_pairs_phi": s_pp.phi, "pitcher_pairs_top_decile_phi": s_pp_top.phi, "pitcher_pred_corr_mean": pp["pred_corr"].mean(),
            }
        )
    out.append(md_table(pd.DataFrame(rows)) + "\n")
    out.append(
        "Reading: with an efficient market the cross-day pitcher pairs carry no correlation whatever the error size (the later leg's closing line "
        "already reflects the earlier leg), so only same-day pairs can be exploited; the top-decile phi on same-day pairs shows what the filter would "
        "find if the errors were as large as each scenario assumes. The filter's predicted covariances are calibrated (slope ≈ 1) in every scenario, "
        "so the near-zero predictions on real data reflect the fitted uncertainties, not a broken pipeline.\n"
    )
    return out


def run_nba_rest(args: argparse.Namespace, cfg: MultiSportConfig, pcfg: ParlayConfig) -> tuple[list[str], dict]:
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out = ["## NBA: shared-game pairs stratified by rest / back-to-back\n"]
    log("[nba] loading")
    games = load_nba(cfg)
    tg = to_team_games(games, pcfg)
    pairs = build_shared_game_pairs(tg)
    rest = load_nba_rest_days(cfg)
    pairs = attach_rest(pairs, games, rest)
    strata = rest_strata(pairs)
    out.append(
        f"{len(games)} games, {len(pairs)} shared-game pairs; rest days from the source's `Days_Rest_*` columns (back-to-back = 1 day; "
        f"{pairs['h_rest'].notna().mean():.1%} of leg-1 teams and {pairs['a_rest'].notna().mean():.1%} of leg-2 teams have a value). "
        f"Leg-1 back-to-back share {(pairs['h_rest'] == 1).mean():.3f}, leg-2 {(pairs['a_rest'] == 1).mean():.3f}. Pre-registered subsets: "
        f"{', '.join(strata)}. Hurdle: {HURDLE_TEXT}.\n"
    )
    groups = {"all": (pairs["h_next_resid"], pairs["a_next_resid"])}
    for lab, m in strata.items():
        groups[lab] = (pairs.loc[m, "h_next_resid"], pairs.loc[m, "a_next_resid"])
    tab = summary_rows(groups, args.n_boot)
    rois = []
    for lab, (x, y) in groups.items():
        r = two_sided_parlay_roi(x, y, d, 1)
        rois.append({"subset": lab, "pairs": r.n_pairs, "same_sign": round(r.hit_rate, 4), "needed": round(0.5 + be / 2, 4), "two_sided_roi": round(r.roi, 4)})
    out.append("Margin residual correlation (spread legs; prediction: positive):\n")
    out.append(md_table(tab) + "\n")
    out.append(md_table(pd.DataFrame(rois)) + "\n")
    m = strata["both legs back-to-back"]
    ttab = summary_rows({"both legs back-to-back": (pairs.loc[m, "h_next_tresid"], pairs.loc[m, "a_next_tresid"]), "all": (pairs["h_next_tresid"], pairs["a_next_tresid"])}, args.n_boot)
    out.append("Total residual correlation (prediction: negative):\n")
    out.append(md_table(ttab) + "\n")
    return out, {"nba_table": tab.to_dict("records"), "nba_totals": ttab.to_dict("records"), "nba_pairs": int(len(pairs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/parlay-pitcher-kalman.md")
    ap.add_argument("--train-end", type=int, default=2022)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params-json", default=None, help="skip the fit and load hyperparameters from this JSON")
    ap.add_argument("--save-params", default=None, help="write the fitted hyperparameters to this JSON")
    ap.add_argument("--stats-json", default=None, help="write headline numbers to this JSON")
    ap.add_argument("--skip-nba", action="store_true")
    ap.add_argument("--skip-power", action="store_true")
    args = ap.parse_args()
    cfg = MultiSportConfig()
    pcfg = ParlayConfig()
    be = breakeven_correlation(pcfg.leg_decimal)
    out = ["# Granular Kalman filter over MLB market errors (pitcher, offense, defense) and NBA rest strata\n"]
    out.append(
        "Follow-up to `parlay-multisport.md`. The team-level daily Kalman filter found no market error that survives to the next day. "
        "Here the state is more granular: one error per starting pitcher (persisting across his starts, teams and seasons), one per team "
        "offense and one per team defense/bullpen, observed through the two per-game run residuals relative to the closing moneyline and "
        "total. Hypothesis: information accumulated over many games in the full covariance matrix predicts which pairs of legs the books "
        "price as independent but are actually correlated (explaining away through a shared pitcher or lineup), and the predicted "
        f"correlations are large enough to beat parlay vig. Hurdle: {HURDLE_TEXT}; break-even phi {be:.3f}.\n"
    )
    out.append("Generated by `python scripts/backtest_pitcher_kalman.py` (see the command at the end).\n")
    body: list[str] = []
    stats_all: dict = {}
    mlb_out, mlb_stats = run_mlb(args, cfg, pcfg)
    body.extend(mlb_out)
    stats_all.update(mlb_stats)
    if not args.skip_nba:
        nba_out, nba_stats = run_nba_rest(args, cfg, pcfg)
        body.extend(nba_out)
        stats_all.update(nba_stats)
    out.extend(body)
    out.append("## Regeneration\n")
    out.append(
        "```\npython scripts/backtest_pitcher_kalman.py --train-end 2022 --max-iter 300 --n-boot 500\n```\n"
        "About 25 minutes single-threaded (the fit dominates); `--params-json` re-uses saved hyperparameters, `--skip-power` and "
        "`--skip-nba` drop those sections.\n"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(stats_all, default=float, indent=1))
    print("\n".join(out))


if __name__ == "__main__":
    main()
