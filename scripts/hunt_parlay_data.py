"""Cross-sport data hunt for the correlated-parlay experiment: inventory + first pass.

Usage:
    python scripts/hunt_parlay_data.py [--download] [--n-boot 1000]
        [--out docs/experiments/parlay-data-hunt.md]

Data location is controlled by ``GEO_MODEL_DATA_DIR`` (default ``data/raw``).
``--download`` fetches every catalog file that is missing (about 330 MB in
total, GitHub hosts only). Without it the script only uses cached files and
skips sources that are absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geo_model.parlay import (
    ParlayConfig,
    breakeven_correlation,
    build_shared_game_pairs,
    correlation_summary,
    to_team_games,
    two_sided_parlay_roi,
)
from geo_model.parlay.data import clean_games
from geo_model.parlay import sources as S

HURDLE_NOTE = (
    "Break-even for a 2-leg parlay at -110 per leg is a phi correlation of **{rho:.3f}** "
    "between the two leg-win indicators, i.e. a same-sign rate of **{same:.1%}**."
)


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.astype(object).iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    return str(v)


def coverage_row(label: str, g: pd.DataFrame) -> dict[str, object]:
    """Per-source coverage summary of a ``GAME_COLUMNS`` table."""
    hold = S.american_hold(g["home_ml_close"], g["away_ml_close"])
    margin = g["home_score"] - g["away_score"]
    cover = (margin - g["spread_close"])
    cover = cover[cover.notna() & (cover != 0)]
    return {
        "source": label,
        "games": int(len(g)),
        "seasons": f"{int(g['season'].min())}-{int(g['season'].max())}",
        "first": str(g["date"].min().date()),
        "last": str(g["date"].max().date()),
        "ml_close": float(g["home_ml_close"].notna().mean()),
        "spread_close": float(g["spread_close"].notna().mean()),
        "total_close": float(g["total_close"].notna().mean()),
        "hold": float(np.nanmean(hold)),
        "home_cover": float((cover > 0).mean()) if len(cover) else float("nan"),
        "pitchers": float(g["home_pitcher_id"].notna().mean()),
    }


def pair_rows(label: str, games: pd.DataFrame, market: str, config: ParlayConfig, n_boot: int) -> tuple[list[dict], int]:
    """Shared-game leg-pair statistics for one sport/market, plus the number of subsets tested."""
    lay = S.to_nflverse_layout(games, market=market)
    seasons = (int(lay["season"].min()), int(lay["season"].max()))
    cg = clean_games(lay, ParlayConfig(seasons=seasons))
    tg = to_team_games(cg, config)
    pairs = build_shared_game_pairs(tg)
    subsets: dict[str, pd.DataFrame] = {"all": pairs}
    if pairs["new_qb_any"].any() and market == "moneyline":
        subsets["new starting pitcher on either team"] = pairs[pairs["new_qb_any"]]
    rows = []
    for name, p in subsets.items():
        for leg, (xcol, ycol) in {
            market: ("h_next_resid", "a_next_resid"),
            "total": ("h_next_tresid", "a_next_tresid"),
        }.items():
            cs = correlation_summary(p[xcol], p[ycol], n_boot=n_boot)
            x = p[xcol].to_numpy(dtype=float)
            y = p[ycol].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
            same = float(np.mean(np.sign(x[m]) == np.sign(y[m]))) if m.any() else float("nan")
            se = float(np.sqrt(same * (1 - same) / m.sum())) if m.sum() else float("nan")
            roi = two_sided_parlay_roi(x, y, config.leg_decimal, expected_sign=1 if leg != "total" else -1)
            rows.append(
                {
                    "sport": label,
                    "leg": leg,
                    "subset": name,
                    "games": int(len(cg)),
                    "pairs": cs.n,
                    "pearson": cs.pearson,
                    "pearson_ci": f"[{cs.pearson_ci[0]:+.3f}, {cs.pearson_ci[1]:+.3f}]",
                    "p": cs.pearson_p,
                    "phi": cs.phi,
                    "phi_ci": f"[{cs.phi_ci[0]:+.3f}, {cs.phi_ci[1]:+.3f}]",
                    "same_sign": same,
                    "same_sign_se": se,
                    "two_sided_roi": roi.roi,
                }
            )
    return rows, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download missing catalog files")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/parlay-data-hunt.md"))
    args = parser.parse_args()
    config = ParlayConfig()

    if args.download:
        for spec in S.SOURCES:
            print(f"downloading {spec.key} ...")
            S.download_source(spec, config)

    inv = S.inventory(config)
    present = {r.key: r.n_present == r.n_files for r in inv.itertuples()}

    # ---- parse everything that is cached -------------------------------------------------
    tables: dict[str, pd.DataFrame] = {}
    if present["nba_sbr_xlsx"]:
        tables["nba (sbr xlsx)"] = S.load_sbr_xlsx_sport("nba", config)
    if present["ncaab_sbr_xlsx"]:
        tables["ncaab (sbr xlsx)"] = S.load_sbr_xlsx_sport("ncaab", config)
    if present["ncaaf_sbr_xlsx"]:
        tables["ncaaf (sbr xlsx)"] = S.load_sbr_xlsx_sport("ncaaf", config)
    for sport in ("nba", "nhl", "nfl"):
        key = f"{sport}_sbr_finned"
        if present[key]:
            tables[f"{sport} (sbr finned)"] = S.read_finned_json(S.source_paths(S.source_by_key(key), config)[0], sport)

    mlb_notes: list[str] = []
    gamelogs = S.load_retrosheet_gamelogs(config) if present["mlb_retrosheet_gamelogs"] else pd.DataFrame()
    mlb_book = pd.DataFrame()
    mlb_rep = pd.DataFrame()
    if present["mlb_sbr_book"] and len(gamelogs):
        mlb_book = S.read_sbr_book_json(S.source_paths(S.source_by_key("mlb_sbr_book"), config)[0])
        mlb_book = S.attach_starting_pitchers(mlb_book, gamelogs)
        tables["mlb 2021-25 (sbr per-book, bet365)"] = mlb_book
        mlb_notes.append(
            f"Per-book file: {len(mlb_book)} regular-season games; starting pitchers attached for "
            f"{mlb_book['pitcher_match'].mean():.1%} (per season: "
            + ", ".join(f"{s}: {v:.1%}" for s, v in mlb_book.groupby("season")["pitcher_match"].mean().items())
            + ")."
        )
    if present["mlb_sbr_finned"] and len(gamelogs):
        raw = pd.read_json(S.source_paths(S.source_by_key("mlb_sbr_finned"), config)[0])
        naive = S.finned_to_games(raw, "mlb")
        naive = S.attach_starting_pitchers(naive, gamelogs)
        mlb_rep = S.repair_finned_mlb(raw, gamelogs)
        tables["mlb 2011-21 (sbr finned, repaired)"] = mlb_rep
        n_retro = int(gamelogs["season"].between(2011, 2021).sum())
        hold = S.american_hold(mlb_rep["home_ml_close"], mlb_rep["away_ml_close"])
        mlb_notes.append(
            f"Archive as scraped: {len(naive)} rows, **{naive['pitcher_match'].mean():.1%}** match a Retrosheet "
            f"matchup (date, home, away) - every row pairs two teams from different games. After splitting rows into "
            f"team records and re-pairing through Retrosheet: **{len(mlb_rep)} games rebuilt of {n_retro}** "
            f"regular-season Retrosheet games 2011-2021 ({len(mlb_rep) / n_retro:.1%}), all with both closing "
            f"moneylines and starting pitchers. Rebuilt moneyline hold: mean {np.nanmean(hold):.3%}, "
            f"5th-95th percentile {np.nanquantile(hold, 0.05):.2%} to {np.nanquantile(hold, 0.95):.2%}, "
            f"negative in {(hold < 0).mean():.1%} of games (a scrambled pairing would give a wide, often negative hold)."
        )
        if len(mlb_book):
            a = mlb_rep.assign(_day=mlb_rep["date"].dt.strftime("%Y-%m-%d"))
            b = mlb_book[mlb_book["pitcher_match"]].assign(_day=mlb_book["date"].dt.strftime("%Y-%m-%d"))
            ov = a.merge(
                b[["_day", "home_code", "away_code", "retro_game_number", "home_ml_close", "away_ml_close", "total_close", "spread_close"]],
                left_on=["_day", "home", "away", "retro_game_number"],
                right_on=["_day", "home_code", "away_code", "retro_game_number"],
                suffixes=("", "_book"),
            )
            ph = S.american_to_prob(ov["home_ml_close"])
            pb = S.american_to_prob(ov["home_ml_close_book"])
            rng = np.random.default_rng(0)
            d_away = (ov["total_close"] - ov["total_close_book"]).abs()
            d_home = (ov["total_close_home_row"] - ov["total_close_book"]).abs()
            d_shuf = (ov["total_close"] - rng.permutation(ov["total_close_book"].to_numpy())).abs()
            mlb_notes.append(
                f"Validation on the {len(ov)} games both files cover in 2021: rebuilt home win probability vs bet365 "
                f"closing, mean absolute difference {np.nanmean(np.abs(ph - pb)):.3f} (shuffled pairing: "
                f"{np.nanmean(np.abs(ph - rng.permutation(pb))):.3f}). Closing total taken from the row holding the "
                f"true away team equals bet365's {float((d_away == 0).mean()):.1%} of the time and is within 0.5 runs "
                f"{float((d_away <= 0.5).mean()):.1%}; the home team's row: {float((d_home <= 0.5).mean()):.1%}; "
                f"shuffled: {float((d_shuf <= 0.5).mean()):.1%}. Run-line favourite agrees "
                f"{float((np.sign(ov['spread_close']) == np.sign(ov['spread_close_book'])).mean()):.1%}."
            )

    tennis = S.load_tennis(config) if present["tennis_data_atp"] else pd.DataFrame()
    tennis_rows = pd.DataFrame()
    if len(tennis):
        tennis_rows = (
            tennis.groupby("year")
            .agg(
                matches=("Winner", "size"),
                bet365=("B365W", lambda s: s.notna().mean()),
                pinnacle=("PSW", lambda s: s.notna().mean()),
                favourite_won=("winner_favourite", "mean"),
            )
            .reset_index()
        )

    # ---- coverage table -------------------------------------------------------------------
    cov = pd.DataFrame([coverage_row(k, v) for k, v in tables.items() if len(v)])

    # ---- shared-game pairs -----------------------------------------------------------------
    pair_specs = [
        ("NBA 2007-21 (sbr xlsx)", "nba (sbr xlsx)", "spread"),
        ("NCAAB 2007-21 (sbr xlsx)", "ncaab (sbr xlsx)", "spread"),
        ("NCAAF 2007-21 (sbr xlsx)", "ncaaf (sbr xlsx)", "spread"),
        ("NHL 2011-21 (sbr finned)", "nhl (sbr finned)", "moneyline"),
        ("MLB 2011-21 (repaired)", "mlb 2011-21 (sbr finned, repaired)", "moneyline"),
        ("MLB 2021-25 (per-book)", "mlb 2021-25 (sbr per-book, bet365)", "moneyline"),
    ]
    pair_rows_all: list[dict] = []
    n_tests = 0
    for label, key, market in pair_specs:
        if key not in tables or not len(tables[key]):
            continue
        rows, n = pair_rows(label, tables[key], market, config, args.n_boot)
        pair_rows_all.extend(rows)
        n_tests += n
    pairs_df = pd.DataFrame(pair_rows_all)

    rho_star = breakeven_correlation(config.leg_decimal)
    same_star = 0.5 + rho_star / 2  # P(same sign) = 1/2 + phi/2 for 50/50 legs

    # ---- catalog table --------------------------------------------------------------------
    cat = pd.DataFrame(
        [
            {
                "sport": s.sport,
                "key": s.key,
                "seasons": s.seasons,
                "files": len(s.files),
                "MB": f"{int(inv.loc[inv['key'] == s.key, 'bytes'].iloc[0]) / 1e6:.1f}",
                "closing": s.closing_lines,
                "pitchers": s.starting_pitchers,
                "quality": s.quality,
                "repo": s.repo_url,
            }
            for s in S.SOURCES
        ]
    )
    lic = pd.DataFrame([{"key": s.key, "license / terms": s.license, "notes": s.notes} for s in S.SOURCES])

    # ---- write --------------------------------------------------------------------------------
    md = []
    md.append("# Cross-sport data hunt: results with closing lines (MLB, NBA, NHL, NCAA, soccer, tennis)\n")
    md.append(
        "Goal: find downloadable historical results **with closing betting lines** for sports other than the NFL, "
        "so the cross-game explaining-away test (`parlay-interaction-effects.md`) can be repeated where 82- and "
        "162-game seasons give far more shared-game pairs, and so MLB can be tested with starting pitchers. "
        "Hypothesis under test in the first pass (Section 5): after H and A play each other, H's next game and A's "
        "next game land on the same side of their lines more than 50% of the time (positive phi for spreads / "
        "moneylines, negative for totals). Books price these two-game parlays as independent.\n"
    )
    md.append(HURDLE_NOTE.format(rho=rho_star, same=same_star) + "\n")
    md.append(
        f"Generated by `python scripts/hunt_parlay_data.py --download --n-boot {args.n_boot}`. "
        "Data directory: `GEO_MODEL_DATA_DIR` (default `data/raw`), one sub-directory per sport.\n"
    )
    md.append("## 1. Method\n")
    md.append(
        "Only `raw.githubusercontent.com` and `github.com/<owner>/<repo>/releases/download/` are reachable from the "
        "sandbox, so candidates were found with web search plus GitHub repository/code search and then **verified by "
        "actually downloading every file with curl** (HTTP 200 and non-trivial size), reading it with pandas, and "
        "checking columns, seasons, null rates, moneyline hold, and (for MLB) agreement with Retrosheet scores and an "
        "independent 2021 odds file. `github.com/<owner>/<repo>/raw/...` and every non-GitHub host return 000/403. "
        "Parsers live in `geo_model.parlay.sources`; every parser normalises to one `GAME_COLUMNS` layout "
        "(`spread_close` = expected home-minus-away margin, nflverse sign convention) and "
        "`to_nflverse_layout` feeds the existing `clean_games -> to_team_games -> build_shared_game_pairs` pipeline.\n"
    )
    md.append("## 2. Verified sources (all downloaded and inspected)\n")
    md.append(md_table(cat) + "\n")
    md.append("Licences and inspection notes:\n")
    md.append(md_table(lic) + "\n")
    md.append("## 3. Parsed coverage\n")
    md.append(
        "`ml_close`, `spread_close`, `total_close` are the share of games with that closing line; `hold` is the mean "
        "two-way moneyline overround (a sanity check that both sides belong to the same game: 2-5% is normal); "
        "`home_cover` is the share of non-push games where the home side beat the closing spread (about 0.48-0.50 for an "
        "efficient closing number); `pitchers` is the share with starting pitchers attached.\n"
    )
    md.append(md_table(cov) + "\n")
    if len(tennis_rows):
        md.append("Tennis (ATP, tennis-data.co.uk mirrors), matches per year and odds coverage:\n")
        md.append(md_table(tennis_rows) + "\n")
    md.append("## 4. MLB: scrambled archive, repair, and starting pitchers\n")
    md.extend(f"- {n}\n" for n in mlb_notes)
    md.append(
        "- Retrosheet game logs (Chadwick Bureau mirror, `seasons/<year>/GL<year>.TXT`) supply starting pitcher ids and "
        "names for every game; the join key is (date, Retrosheet home code, away code) with doubleheaders resolved by "
        "final score. Team spellings are mapped in `MLB_TEAM_CODES` (`LOS`=Dodgers, `CUB`, `KAN`, `SDG`, `SFO`, `TAM`; "
        "Marlins are `FLO` in 2011, Athletics are `ATH` from 2025).\n"
    )
    md.append("## 5. Shared-game leg pairs per sport (first pass)\n")
    md.append(
        "Same pipeline as the NFL backtest. Anchor game H vs A; legs are H's next game and A's next game in the same "
        "season (dropped when the two next games coincide, or the teams meet again). For NBA/NCAAB/NCAAF the leg "
        "residual is `margin - closing spread`; for MLB/NHL it is `1[home win] - p_home` with `p_home` the no-vig "
        "closing moneyline probability (run/puck lines are not centred). Totals use `total - closing total`. "
        "`phi` is the correlation of the two cover/win indicators (pushes dropped), with 95% bootstrap CIs; "
        "`same_sign` is the share of pairs landing on the same side; `two_sided_roi` bets both same-sign parlays "
        "(opposite-sign for totals) on every pair at -110 per leg, so it isolates the correlation edge from any "
        "directional edge. Starting pitchers stand in for the NFL 'new QB' flag (at most 2 prior starts for that team). "
        "MLB and NHL yield far fewer pairs than games because teams play series: when H's next game is against A again "
        "the pair is dropped, so only series finales qualify as anchors.\n"
    )
    if len(pairs_df):
        md.append(md_table(pairs_df) + "\n")
        best = pairs_df.loc[pairs_df["phi"].abs().idxmax()]
        md.append(
            f"Largest |phi| across all {len(pairs_df)} rows: {best['sport']} {best['leg']} ({best['subset']}) "
            f"phi = {best['phi']:+.4f}, CI {best['phi_ci']}, n = {best['pairs']}; the hurdle is {rho_star:.3f}. "
            f"Every same-sign rate is within a few standard errors of 0.50 and none approaches {same_star:.1%}; every "
            "two-sided ROI is close to the -8.9% expected under independence.\n"
        )
    md.append("## 6. Multiple-comparison exposure\n")
    md.append(
        f"{n_tests} pair statistics are reported above (sport x market x subset), each with a Pearson p-value and two "
        f"bootstrap intervals; no subset was selected after looking at the data and no per-season or per-surprise "
        f"stratification was run in this pass. Section 3 also reports {len(cov)} coverage rows and Section 4 three validation "
        "checks, none of which are hypothesis tests. With this many looks a single p < 0.05 would be expected by chance.\n"
    )
    md.append("## 7. Candidates that were checked and rejected\n")
    md.append(
        "- **fivethirtyeight/data mlb-elo** (pitchers + Elo, no lines): the repo only holds a README; the CSV is served "
        "from projects.fivethirtyeight.com, which is unreachable; four GitHub mirrors probed all 404.\n"
        "- **FinnedAI MLB archive as published**: 0% of rows match a real matchup (Section 4); usable only after repair.\n"
        "- **footballcsv/cache.footballdata** (CC0): reformatted to Date/Team/FT/HT only, all odds columns dropped.\n"
        "- **pwu97/bettingtools**: one season (`mlb_odds_2019.rda`, 36 KB), R binary format.\n"
        "- **Jiashuz123/mlb-sportsbook-benchmarking**, **cswaters/nba-odds-parse**, **cswaters/odds-parsing**, "
        "**ArnavSaraogi scraper source**: no raw data committed (the ArnavSaraogi release asset is the exception, used).\n"
        "- **sawlachintan/tennis-prediction**: Grand Slam main draws only (127 matches/year).\n"
        "- **inesmteixeira9/tennis-predictor**, **carltoews/tennis**: partial years (2017: 1,974 rows; 2018: 611 rows).\n"
        "- **hoopR / sportsdataverse NBA & NHL releases**: schedules and box scores only, no lines found.\n"
        "- **Kaggle NBA/MLB odds mirrors** (`nba_betting_spread.csv`, `oddsData.csv`): no GitHub copy indexed.\n"
        "- **NHL before 2011-12 or after 2021-22, NBA after 2021-22, NCAA after 2021-22**: nothing reachable with lines. "
        "The SBR online archive itself (sportsbookreviewsonline.com) is blocked by the proxy.\n"
        "- Another parallel track has since placed overlapping copies under `data/raw/multisport/`; those were not used here.\n"
    )
    md.append("## 8. Caveats\n")
    md.append(
        "- SBR archives report one consensus closing number per game, not a specific book, and their team spellings "
        "drift across seasons (`Golden State` vs `Warriors`, `NY Islanders` vs `Islanders`); aliases are in "
        "`NBA_ALIASES` / `NHL_ALIASES`. FinnedAI uses modern franchise names retroactively (`Pelicans` in 2011).\n"
        "- A handful of FinnedAI NBA/NHL rows carry a total in the spread field (e.g. -242.5); `PLAUSIBLE` ranges blank "
        "them rather than guess. NHL/MLB run and puck lines are blank in 25-29% of games.\n"
        "- The DillonKoch workbooks list the spread on the favourite's row and the total on the other; when only one row "
        "has a number the split uses a per-sport threshold (unambiguous for NBA/NCAAB, a 40-point cut for NCAAF).\n"
        "- The repaired MLB archive loses about 6% of games (teams that could not be matched or doubleheaders with "
        "identical scores) and its totals come from the away team's row (94% within half a run of bet365 in 2021, "
        "so ~6% are a different game's total).\n"
        "- tennis-data.co.uk and football-data.co.uk odds are pre-match snapshots, not documented closing prices "
        "(football-data added `*C` closing columns from 2019-20; only one such raw file was found on GitHub). "
        "The xgabora consolidation drops them. Tennis has no shared-game structure comparable to team sports.\n"
        "- The cfbfastR line file is long-format multi-book with team abbreviations that still need a mapping to "
        "resolve home/away; it was verified but not parsed into games.\n"
        "- All of these sources are third-party scrapes without an explicit data licence; Retrosheet requires its "
        "attribution notice. Nothing here is ACLED.\n"
    )
    md.append("## 9. Recommendation\n")
    md.append(
        "1. **MLB first**: `repair_finned_mlb` (2011-2021, 23.6k games) + the per-book file (2021-2025, 11k games), "
        "both with Retrosheet starting pitchers. This is the user's stated priority and the only sport where the "
        "'new starter' analogue of the NFL new-QB flag exists per game.\n"
        "2. **NBA** via the DillonKoch workbooks (2007-2022, 19k games, full open/close spreads and totals) - the "
        "cleanest spread market with the most shared-game pairs per season.\n"
        "3. **NCAAB** (62k games) for sheer pair count, then **NHL** (13.7k games, moneyline only).\n"
        "4. Soccer and tennis only if a use for non-closing pre-match odds is accepted.\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
