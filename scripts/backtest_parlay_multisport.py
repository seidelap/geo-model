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
import sys
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
from geo_model.parlay.data_multisport import MultiSportConfig, load_mlb, load_mlb_archive_repaired, load_nba, load_sbr_archive
from geo_model.parlay.data_soccer import SoccerConfig, load_soccer
from geo_model.parlay.multisport_backtest import build_pitcher_pairs, moneyline_parlay
from geo_model.parlay.pairs import stratify

SPORTS = {
    # loader, surprise bins (score units), Kalman training seasons (inclusive end), primary market
    "nba": dict(bins=(0.0, 7.0, 14.0, 1000.0), train_end=2013, market="spread"),
    "nhl": dict(bins=(0.0, 1.0, 2.0, 1000.0), train_end=2014, market="moneyline"),
    "mlb": dict(bins=(0.0, 2.0, 4.0, 1000.0), train_end=2022, market="moneyline"),
    "mlb_2011_2020": dict(bins=(0.0, 2.0, 4.0, 1000.0), train_end=2014, market="moneyline"),
    "epl": dict(bins=(0.0, 1.0, 2.0, 1000.0), train_end=2012, market="moneyline"),
    "big5": dict(bins=(0.0, 1.0, 2.0, 1000.0), train_end=2012, market="moneyline"),
}


SUMMARY = """## Summary of findings

All-pairs shared-game correlations are indistinguishable from zero in every sport and far below the
hurdle: NBA -0.002 (20,495 pairs), NHL -0.022 (11,498), MLB 2021-25 +0.022 (3,549), MLB 2011-20 -0.004
(7,076), Premier League -0.014 (9,413), big-five leagues -0.004 (43,634); NFL was +0.002 (6,488). Totals
correlations, mechanical two-sided parlays (ROI -8% to -10%, i.e. the vig), the daily-slate Kalman filters
(predicted |corr| never above 0.004; the fitted market-error persistence is 0.27 in NHL and collapses to
~0 in MLB and soccer, and the NBA hyperparameters are unidentified: two optima 0.13 log-likelihood units
apart, both predicting |corr| below 0.004, so closing lines carry no team-level error that survives to
the next day) and the MLB
starting-pitcher pairs (18,549 pairs, phi -0.000, moneyline-parlay edge +0.002) all agree.

What the null result does and does not show. Simulating the explaining-away model on the real NBA
schedule with a deliberately huge season-start team error (prior_std 10 points, obs_std 12) and a market
that updates every day gives an all-pairs residual correlation of 0.000 (CI -0.014 to +0.013) and about
+0.03 in weeks 1-2 (MLB 2011-20 schedule, planted 3-run error: all-pairs +0.02, weeks 1-2 phi +0.02, below
the watch item's out-of-sample +0.07): an efficiently updating market learns a team-level error within days, so the
one-shared-game formula is an upper bound that applies only to the first games of a season. The
all-pairs tests therefore cannot separate "no market error" from "large but quickly corrected error";
what they establish, with CI upper bounds of 0.01-0.04 on phi against a 0.098 hurdle, is that no
bettable cross-game correlation exists in any sport. The within-team lag-1 residual autocorrelations
(all between -0.015 and +0.007) rule out the static-market alternative, where an uncorrected error
would appear as a single-bet edge instead.

One watch item, reported for completeness and not as a finding. MLB team-level pairs in weeks 1-2 of
the season: in 2021-25 (189 pairs, where it was noticed) residual correlation +0.23 and phi +0.19; in the
pre-registered out-of-sample 2011-20 check (490 pairs) residual correlation +0.01 (CI -0.07 to +0.09),
phi +0.07 (CI -0.03 to +0.15), same-side rate 55.7% vs 49.9% expected, moneyline-parlay edge +9.5% per
unit (CI +0.7% to +17.5%). Pooled 2011-25 (679 pairs): phi 0.10 (CI 0.02 to 0.18). Reasons for
scepticism: the continuous correlation does not replicate, the per-season sign is split 5-5 out of
sample, no other sport shows an early-season effect (NFL, NBA, NHL and soccer weeks 1-2 are all within
noise of zero or negative), the Gaussian theory predicts ~0.0005 for MLB, and this is one of roughly
sixty subsets examined across sports, so one nominal p~0.03 is what chance produces. Early-season
favourite calibration (+1.9 points on the legs of those pairs) cannot explain it: shifting favourites up by
that amount lowers the independence baseline, so the "edge" would grow, not shrink. The composition also
differs between the two samples (2011-20: the same-side excess is in favourite/underdog mixed pairs;
2021-25: in both-favourite and both-underdog pairs), which is what noise looks like. At 15-60 qualifying
pairs per season it would take several more seasons (2026 onward) to confirm or kill it.

Caveats: soccer odds are pre-match snapshots, not closing; the soccer moneyline "edge" is contaminated by
the favourite-longshot bias in the proportional vig removal (the independence baseline is too
optimistic for longshot legs), so use the residual correlations and same-side rates there; NBA spreads
before 2022-23 were re-signed from an unsigned source (about 1.6% of rows dropped, see the NBA section);
the NHL archive dated the 2020 bubble playoffs and Jan-Mar 2021 a year early and those dates were
corrected; NHL goalies are unavailable; the repaired 2011-20 MLB archive cannot recover the last game
listed on each date (about 8% of games, concentrated on WAS/PHI/PIT/CHC/MIA/NYM/CIN), so for those teams
"next game" is sometimes the game after next; the moneyline-to-margin scale `sd` and the soccer margin fit
are single sport-level constants fitted on all seasons (they carry no team or date information, but
residual signs near zero depend on them); bootstrap CIs resample pairs independently, and cluster
bootstraps by anchor day, season-week and season give the same intervals to within 0.01.
"""


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
    if sport == "mlb_2011_2020":
        return load_mlb_archive_repaired(cfg)
    if sport == "epl":
        return load_soccer(SoccerConfig(divisions=("E0",)))
    if sport == "big5":
        return load_soccer(SoccerConfig())
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
    print(f"[{sport}] loading", file=sys.stderr, flush=True)
    games = load(sport, cfg)
    tg = to_team_games(games, pcfg)
    pairs = build_shared_game_pairs(tg)
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out = [f"## {sport.upper()}\n"]
    if sport == "mlb_2011_2020":
        out.append(
            "Repaired 10-year SportsBookReview archive (row misalignment fixed by shifting the away-side fields up one row "
            "within each date; validated on the 2021 overlap: 97.9% exact score agreement, moneyline correlation 0.96). "
            "No starting pitchers. Serves as the out-of-sample check for the 2021-2025 early-season subset.\n"
        )
    if sport in ("epl", "big5"):
        out.append(
            "Soccer (football-data.co.uk via the xgabora consolidation): pre-match 1X2 odds, not closing prices. "
            f"Implied goal margin = {games['margin_fit'].iloc[0]}; draws are a priced outcome, so the moneyline parlay "
            "below uses the 3-way fair probabilities. 'promoted' = a team not in the division the previous season.\n"
        )
    if sport == "nba" and "nba_rows_in" in games.attrs:
        a = games.attrs
        out.append(
            f"Source cleaning: {a['nba_rows_in']} rows; dropped {a['nba_rows_dropped_inconsistent']} whose (re)signed spread "
            f"disagrees with the moneyline by more than {6.0:g} points, {a['nba_rows_dropped_even_ml']} unsigned rows with an "
            f"even moneyline (sign unrecoverable), {a['nba_rows_dropped_no_ml']} without a moneyline; whole months dropped: "
            f"{a['nba_months_dropped'] or 'none'}. Seasons 2007-08 to 2021-22 store |spread| and were re-signed by moneyline favourite.\n"
        )
    if sport == "nhl" and "archive_rows_redated" in games.attrs:
        out.append(
            f"Source cleaning: {games.attrs['archive_rows_redated']} games the archive dated a year early (2019-20 bubble "
            "playoffs dated Aug-Sep 2019; Jan-Mar 2021 dated 2020) were moved forward one year so that week 1 and the "
            "next-game ordering are chronological. No goalie data.\n"
        )
    if sport in ("mlb", "mlb_2011_2020"):
        out.append(
            "Week 1 starts at the season's first game, so seasons with an international opening series (2014, 2019 in the "
            "archive, 2024, 2025) have a short week 1-2 window with few pairs.\n"
        )
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
    extra = {
        "weeks 1-2 of season": pairs["week"] <= 2,
        "weeks 1-3 of season": pairs["week"] <= 3,
        "both legs same week": pairs["same_next_week"],
        "out-of-sample seasons": pairs["season"] > spec["train_end"],
    }
    if "home_new" in games:
        flags = games.set_index("game_id")[["home_new", "away_new"]]
        extra["promoted team in anchor"] = (flags.loc[pairs["game_id"], "home_new"] | flags.loc[pairs["game_id"], "away_new"]).to_numpy()
        extra["no promoted team"] = ~extra["promoted team in anchor"]
    for lab, m in extra.items():
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
    for lab, m in {"weeks 1-2": pairs["week"] <= 2, "weeks 1-3": pairs["week"] <= 3, "|surprise| top bin": sb == sorted(sb.unique())[-1]}.items():
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
    print(f"[{sport}] fitting Kalman on seasons <= {spec['train_end']}", file=sys.stderr, flush=True)
    init = KalmanParams(prior_std=train["resid"].std() * 0.2, process_std=0.05 * train["resid"].std(), obs_std=train["resid"].std(), persistence=0.95)
    fitted = MarketErrorKalman(init, slate_col="week").fit(train, max_iter=150)  # weekly slates: fast, same hyperparameters
    kf = MarketErrorKalman(fitted, slate_col="gameday")
    lat, nxt = gaussian_explaining_away_corr(fitted.prior_std, fitted.obs_std)
    res = kf.run(test)
    kp = res.pairs
    sl, se, pv = kalman_slope(kp)
    degenerate = kp["pred_cov"].abs().max() < 1e-9  # team-level uncertainty collapsed: slopes are 1/0 noise
    print(f"[{sport}] Kalman done: {fitted}", file=sys.stderr, flush=True)
    slope_txt = (
        "not defined (all predicted covariances are numerically zero)" if degenerate
        else f"{sl:.2f} (se {se:.2f}, p={pv:.3f})"
    )
    out.append(
        f"Fitted on seasons ≤{spec['train_end']}: prior_std={fitted.prior_std:.3f}, process_std={fitted.process_std:.3f}, "
        f"obs_std={fitted.obs_std:.3f}, persistence={fitted.persistence:.3f}. One-shared-game theory: latent corr {lat:.4f}, "
        f"next-game residual corr {nxt:.5f} (break-even {be:.3f}). Out-of-sample {len(kp)} same-day pairs, predicted |corr| "
        f"max {kp['pred_corr'].abs().max():.4f}, mean {kp['pred_corr'].abs().mean():.5f}. Calibration slope of realized "
        f"residual product on predicted covariance: {slope_txt}.\n"
    )
    if kp["pred_corr"].nunique() > 5:
        cal = kalman_pair_calibration(kp, n_quantiles=5)
        cal["n"] = cal["n"].astype(int)
        cal["bin"] = cal["bin"].astype(int)
        out.append(md_table(cal) + "\n")
    else:
        out.append("Predicted correlations are all (numerically) zero: the fitted team-level uncertainty collapsed, so no calibration table.\n")
    top = kp[kp["pred_corr"] >= kp["pred_corr"].quantile(0.9)]
    bot = kp[kp["pred_corr"] <= kp["pred_corr"].quantile(0.1)]
    rt = two_sided_parlay_roi(top["resid_1"], top["resid_2"], d, 1)
    rb = two_sided_parlay_roi(bot["resid_1"], bot["resid_2"], d, -1)
    out.append(f"Top decile predicted positive corr: {rt.n_pairs} pairs, same-sign {rt.hit_rate:.4f}, ROI {rt.roi:+.4f}. "
               f"Bottom decile: {rb.n_pairs} pairs, opposite-sign {rb.hit_rate:.4f}, ROI {rb.roi:+.4f}."
               + (" (Deciles are ill-defined here: most predicted correlations are tied at zero.)" if degenerate else "") + "\n")
    tgm = tg[tg["next_resid"].notna()]
    lag1 = np.corrcoef(tgm["resid"], tgm["next_resid"])[0, 1]
    if res.games["pred_mean"].abs().max() < 1e-9:
        mtxt = "not defined (all predicted means are numerically zero)"
    else:
        mres = stats.linregress(res.games["pred_mean"], res.games["resid"])
        mtxt = f"{mres.slope:.2f} (se {mres.stderr:.2f})"
    out.append(f"Marginal check: slope of realized residual on Kalman predicted mean {mtxt}; "
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
    ap.add_argument("--sports", default="nba,nhl,mlb,mlb_2011_2020,epl,big5")
    ap.add_argument("--out", default="docs/experiments/parlay-multisport.md")
    args = ap.parse_args()
    cfg = MultiSportConfig()
    pcfg = ParlayConfig()
    d = pcfg.leg_decimal
    be = breakeven_correlation(d)
    out = ["# Cross-game correlated parlays: NBA, NHL, MLB, soccer\n"]
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
    out.append(SUMMARY)
    for sport in args.sports.split(","):
        out.extend(run_sport(sport.strip(), cfg, pcfg))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
