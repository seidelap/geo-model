"""Cross-game correlated-parlay backtest for NBA, NHL and MLB.

Usage:
    python scripts/backtest_parlay_multisport.py [--sports nba,nhl,mlb] [--out docs/experiments/parlay-multisport.md]

Data is cached under ``$GEO_MODEL_DATA_DIR/multisport`` (default ``data/raw/multisport``).
MLB starting pitchers require ``mlb_starting_pitchers.parquet`` in that directory
(build it with ``geo_model.parlay.retrosheet.load_starting_pitchers`` from a
sparse checkout of ``github.com/chadwickbureau/retrosheet``, ``seasons/`` dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from geo_model.parlay import (
    KalmanParams,
    MarketErrorKalman,
    ParlayConfig,
    breakeven_correlation,
    build_shared_game_pairs,
    correlation_summary,
    to_team_games,
    two_sided_parlay_roi,
)
from geo_model.parlay.backtest import gaussian_explaining_away_corr, kalman_pair_calibration, kalman_slope
from geo_model.parlay.data_multisport import MultiSportConfig, load_mlb, load_nba, load_sbr_archive
from geo_model.parlay.multisport_backtest import build_pitcher_pairs, moneyline_parlay
from geo_model.parlay.pairs import stratify

SPORTS = {
    # loader, surprise bins (score units), Kalman training seasons (inclusive end), primary market
    "nba": dict(bins=(0.0, 7.0, 14.0, 1000.0), train_end=2013, market="spread"),
    "nhl": dict(bins=(0.0, 1.0, 2.0, 1000.0), train_end=2014, market="moneyline"),
    "mlb": dict(bins=(0.0, 2.0, 4.0, 1000.0), train_end=2022, market="moneyline"),
}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def summary_rows(groups: dict[str, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    rows = []
    for label, (x, y) in groups.items():
        row = {"subset": label}
        row.update(correlation_summary(x, y, n_boot=500).to_row())
        rows.append(row)
    return pd.DataFrame(rows)


def load(sport: str, cfg: MultiSportConfig) -> pd.DataFrame:
    if sport == "nba":
        return load_nba(cfg)
    if sport == "nhl":
        return load_sbr_archive("nhl", cfg)
    return load_mlb(cfg)


def ml_row(label: str, r) -> dict:
    return {
        "subset": label, "pairs": r.n_pairs, "realized_roi": round(r.realized_roi, 4),
        "indep_roi": round(r.independent_roi, 4), "corr_edge": round(r.edge, 4),
        "edge_ci": f"[{r.edge_ci[0]:+.3f}, {r.edge_ci[1]:+.3f}]",
        "p_same_side": round(r.p_same_side, 4), "p_same_indep": round(r.p_same_side_indep, 4),
    }


def run_sport(sport: str, cfg: MultiSportConfig, pcfg: ParlayConfig) -> list[str]:
    spec = SPORTS[sport]
    games = load(sport, cfg)
    tg = to_team_games(games, pcfg)
    pairs = build_shared_game_pairs(tg)
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out = [f"## {sport.upper()}\n"]
    slope = np.polyfit(games["spread_line"], games["result"], 1)[0]
    out.append(
        f"{len(games)} games, seasons {games['season'].min()}–{games['season'].max()}, "
        f"{len(pairs)} shared-game leg pairs. Market sanity: slope of realized margin on the "
        f"{'closing spread' if spec['market'] == 'spread' else 'moneyline-implied margin'} = {slope:.3f}, "
        f"mean residual {games['resid'].mean():+.3f}, residual sd {games['resid'].std():.2f}; "
        f"home win rate {(games['result'] > 0).mean():.3f} vs. implied {games['p_home'].mean():.3f}.\n"
    )
    # Shared-game pairs, margin residuals
    sb = stratify(pairs, "abs_surprise", spec["bins"])
    groups = {"all": (pairs["h_next_resid"], pairs["a_next_resid"])}
    for lab in sorted(sb.unique()):
        m = sb == lab
        groups[f"|surprise| {lab}"] = (pairs.loc[m, "h_next_resid"], pairs.loc[m, "a_next_resid"])
    for lab, m in {
        "weeks 1-3 of season": pairs["week"] <= 3,
        "both legs same week": pairs["same_next_week"],
        "out-of-sample seasons": pairs["season"] > spec["train_end"],
    }.items():
        groups[lab] = (pairs.loc[m, "h_next_resid"], pairs.loc[m, "a_next_resid"])
    out.append("### Shared-game pairs: margin residual correlation (prediction: positive)\n")
    out.append(md_table(summary_rows(groups)) + "\n")
    tgroups = {
        "all": (pairs["h_next_tresid"], pairs["a_next_tresid"]),
        "weeks 1-3 of season": (pairs.loc[pairs["week"] <= 3, "h_next_tresid"], pairs.loc[pairs["week"] <= 3, "a_next_tresid"]),
    }
    out.append("### Shared-game pairs: total residual correlation (prediction: negative)\n")
    out.append(md_table(summary_rows(tgroups)) + "\n")
    # Same-sign / ROI
    out.append("### Mechanical two-sided parlays\n")
    rows = []
    if spec["market"] == "spread":
        r = two_sided_parlay_roi(pairs["h_next_resid"], pairs["a_next_resid"], d, 1)
        rows.append({"strategy": "spread (cover,cover)+(fail,fail) @-110", "pairs": r.n_pairs, "roi": round(r.roi, 4), "same_sign": round(r.hit_rate, 4), "needed": round(0.5 + be / 2, 4)})
    r = two_sided_parlay_roi(pairs["h_next_tresid"], pairs["a_next_tresid"], d, -1)
    rows.append({"strategy": "totals (over,under)+(under,over) @-110", "pairs": r.n_pairs, "roi": round(r.roi, 4), "same_sign": round(1 - r.hit_rate, 4), "needed": round(0.5 - be / 2, 4)})
    out.append(md_table(pd.DataFrame(rows)) + "\n")
    mlrows = [ml_row("all pairs", moneyline_parlay(pairs, tg, games))]
    for lab, m in {"weeks 1-3": pairs["week"] <= 3, "|surprise| top bin": sb == sorted(sb.unique())[-1]}.items():
        mlrows.append(ml_row(lab, moneyline_parlay(pairs[m], tg, games)))
    out.append(
        "Moneyline parlays at actual closing prices: (H wins, A wins) + (H loses, A loses), 1 unit each. "
        "`corr_edge` = realized ROI minus the ROI expected under independence (vig-free probabilities, actual payouts); "
        "a positive edge larger than the vig (about 0.04–0.05 per unit here) would make the parlay +EV.\n"
    )
    out.append(md_table(pd.DataFrame(mlrows)) + "\n")
    # Kalman with daily slates
    out.append("### Whole-network Kalman filter (daily slates)\n")
    train = games[games["season"] <= spec["train_end"]]
    test = games[games["season"] > spec["train_end"]]
    kf = MarketErrorKalman(KalmanParams(prior_std=games["resid"].std() * 0.2, process_std=0.05 * games["resid"].std(), obs_std=games["resid"].std(), persistence=0.95), slate_col="gameday")
    fitted = kf.fit(train)
    lat, nxt = gaussian_explaining_away_corr(fitted.prior_std, fitted.obs_std)
    res = kf.run(test)
    kp = res.pairs
    sl, se, pv = kalman_slope(kp)
    out.append(
        f"Fitted on seasons ≤{spec['train_end']}: prior_std={fitted.prior_std:.3f}, process_std={fitted.process_std:.3f}, "
        f"obs_std={fitted.obs_std:.3f}, persistence={fitted.persistence:.3f}. One-shared-game theory: latent corr {lat:.4f}, "
        f"next-game residual corr {nxt:.5f} (break-even {be:.3f}). Out-of-sample {len(kp)} same-day pairs, predicted |corr| "
        f"max {kp['pred_corr'].abs().max():.4f}, mean {kp['pred_corr'].abs().mean():.5f}. Calibration slope of realized "
        f"residual product on predicted covariance: {sl:.2f} (se {se:.2f}, p={pv:.3f}).\n"
    )
    cal = kalman_pair_calibration(kp, n_quantiles=5)
    cal["n"] = cal["n"].astype(int)
    cal["bin"] = cal["bin"].astype(int)
    out.append(md_table(cal) + "\n")
    top = kp[kp["pred_corr"] >= kp["pred_corr"].quantile(0.9)]
    bot = kp[kp["pred_corr"] <= kp["pred_corr"].quantile(0.1)]
    rt = two_sided_parlay_roi(top["resid_1"], top["resid_2"], d, 1)
    rb = two_sided_parlay_roi(bot["resid_1"], bot["resid_2"], d, -1)
    out.append(f"Top decile predicted positive corr: {rt.n_pairs} pairs, same-sign {rt.hit_rate:.4f}, ROI {rt.roi:+.4f}. "
               f"Bottom decile: {rb.n_pairs} pairs, opposite-sign {rb.hit_rate:.4f}, ROI {rb.roi:+.4f}.\n")
    mres = stats.linregress(res.games["pred_mean"], res.games["resid"])
    tgm = tg[tg["next_resid"].notna()]
    lag1 = np.corrcoef(tgm["resid"], tgm["next_resid"])[0, 1]
    out.append(f"Marginal check: slope of realized residual on Kalman predicted mean {mres.slope:.2f} (se {mres.stderr:.2f}); "
               f"within-team lag-1 residual autocorrelation {lag1:+.4f} (n={len(tgm)}).\n")
    # MLB pitchers
    if sport == "mlb" and games["home_qb_id"].notna().any():
        pp = build_pitcher_pairs(games)
        out.append("### Starting-pitcher pairs (MLB)\n")
        out.append(
            f"Anchor = one start by pitcher P (team A) vs team B. Leg 1 = P's next start vs a team other than B "
            f"(median gap {pp['gap_days_leg1'].median():.0f} days); leg 2 = B's next game vs a team other than A "
            f"(median gap {pp['gap_days_leg2'].median():.0f} days). {len(pp)} pairs.\n"
        )
        pb = stratify(pp, "abs_surprise", spec["bins"])
        pg = {"all": (pp["h_next_resid"], pp["a_next_resid"])}
        for lab in sorted(pb.unique()):
            m = pb == lab
            pg[f"|surprise| {lab}"] = (pp.loc[m, "h_next_resid"], pp.loc[m, "a_next_resid"])
        m = pp["week"] <= 4
        pg["weeks 1-4"] = (pp.loc[m, "h_next_resid"], pp.loc[m, "a_next_resid"])
        out.append(md_table(summary_rows(pg)) + "\n")
        out.append("Totals (prediction: negative):\n")
        out.append(md_table(summary_rows({"all": (pp["h_next_tresid"], pp["a_next_tresid"])})) + "\n")
        prow = [ml_row("all pitcher pairs", moneyline_parlay(pp, tg, games)),
                ml_row("|surprise| top bin", moneyline_parlay(pp[pb == sorted(pb.unique())[-1]], tg, games)),
                ml_row("weeks 1-4", moneyline_parlay(pp[m], tg, games))]
        out.append(md_table(pd.DataFrame(prow)) + "\n")
    # Per-season
    ps = pairs.groupby("season").apply(
        lambda q: pd.Series({"n": len(q), "pearson": np.corrcoef(q["h_next_resid"], q["a_next_resid"])[0, 1]}), include_groups=False
    ).reset_index()
    ps["n"] = ps["n"].astype(int)
    ps["season"] = ps["season"].astype(int)
    out.append("### Per-season margin-residual correlation (shared-game pairs)\n")
    out.append(md_table(ps) + "\n")
    out.append(f"Seasons positive: {(ps['pearson'] > 0).mean():.0%}.\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sports", default="nba,nhl,mlb")
    ap.add_argument("--out", default="docs/experiments/parlay-multisport.md")
    args = ap.parse_args()
    cfg = MultiSportConfig()
    pcfg = ParlayConfig()
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out = ["# Cross-game correlated parlays: NBA, NHL, MLB\n"]
    out.append(
        "Same hypothesis and machinery as `parlay-interaction-effects.md` (NFL), applied to daily sports. "
        "Anchor game H vs A; legs are H's next game and A's next game against different opponents, priced "
        "independently by the book. Explaining-away predicts a positive correlation of the two margin residuals "
        "and a negative correlation of the two total residuals. NHL and MLB use the closing moneyline as the "
        "primary market: the market-implied margin is `sd * Phi^-1(p_home)` with `p_home` vig-free and `sd` "
        "fitted by OLS; parlays on those legs are priced at the actual closing decimals.\n"
    )
    out.append(f"Hurdle: 2-leg spread/total parlay at -110 needs phi > {be:.3f} (same-sign rate {0.5 + be / 2:.1%}). "
               f"Generated by `scripts/backtest_parlay_multisport.py`.\n")
    for sport in args.sports.split(","):
        out.extend(run_sport(sport.strip(), cfg, pcfg))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
