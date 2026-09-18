"""H5: top-down line shopping.

Hypothesis: when one bookmaker's price implies a lower probability than the no-vig
consensus by more than a margin, betting that price is +EV.

Data limitation: Pinnacle is unreachable from this environment, so the consensus is
Bet365's no-vig price and the "outlier" price is the best price across ~17 European
books (football-data.co.uk ``Max`` columns). Both are near-closing, not closing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import TOP5, bootstrap_mean_ci, load_odds_matches, no_vig


def build_bets_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, outcome) with consensus prob, best price and result."""
    d = df.dropna(subset=["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
                          "FTResult"]).copy()
    d = d[(d[["OddHome", "OddDraw", "OddAway"]] > 1).all(axis=1)]
    p = no_vig(d[["OddHome", "OddDraw", "OddAway"]].to_numpy())
    rows = []
    for j, (oc, pcol, mcol) in enumerate([("H", "OddHome", "MaxHome"), ("D", "OddDraw", "MaxDraw"),
                                          ("A", "OddAway", "MaxAway")]):
        r = pd.DataFrame({
            "Division": d["Division"].values, "season": d["season"].values,
            "outcome": oc, "p_fair": p[:, j], "b365": d[pcol].values, "best": d[mcol].values,
            "win": (d["FTResult"].values == oc).astype(float),
        })
        rows.append(r)
    b = pd.concat(rows, ignore_index=True)
    b["p_best"] = 1.0 / b["best"]
    b["edge"] = b["p_fair"] - b["p_best"]           # positive => best price too long
    b["ev"] = b["p_fair"] * b["best"] - 1.0          # EV per unit at consensus prob
    b["pnl"] = b["win"] * b["best"] - 1.0
    return b


def build_bets_ou(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["Over25", "Under25", "MaxOver25", "MaxUnder25", "FTHome", "FTAway"]).copy()
    d = d[(d[["Over25", "Under25"]] > 1).all(axis=1)]
    p = no_vig(d[["Over25", "Under25"]].to_numpy())
    tot = (d["FTHome"] + d["FTAway"]).values
    rows = []
    for j, (oc, pcol, mcol, w) in enumerate([("O", "Over25", "MaxOver25", tot > 2.5),
                                             ("U", "Under25", "MaxUnder25", tot < 2.5)]):
        rows.append(pd.DataFrame({
            "Division": d["Division"].values, "season": d["season"].values, "outcome": oc,
            "p_fair": p[:, j], "b365": d[pcol].values, "best": d[mcol].values,
            "win": w.astype(float)}))
    b = pd.concat(rows, ignore_index=True)
    b["p_best"] = 1.0 / b["best"]
    b["edge"] = b["p_fair"] - b["p_best"]
    b["ev"] = b["p_fair"] * b["best"] - 1.0
    b["pnl"] = b["win"] * b["best"] - 1.0
    return b


def roi_table(b: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    out = []
    for th in thresholds:
        s = b[b["edge"] >= th]
        m, lo, hi = bootstrap_mean_ci(s["pnl"].to_numpy(), n_boot=500)
        out.append({"edge>=": th, "n_bets": len(s), "share_of_all": len(s) / len(b),
                    "mean_model_ev": s["ev"].mean(), "roi": m, "roi_lo": lo, "roi_hi": hi,
                    "avg_best_odds": s["best"].mean(),
                    "realised_p": s["win"].mean(), "consensus_p": s["p_fair"].mean()})
    return pd.DataFrame(out)


def main() -> None:
    df = load_odds_matches()
    df = df[df["season"] >= 2005]
    pd.set_option("display.width", 200); pd.set_option("display.precision", 4)
    ths = [-1.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07]

    b = build_bets_1x2(df)
    print(f"1X2 bets universe: {len(b):,} outcome-rows from {len(b)//3:,} matches\n")
    print("== 1X2, all divisions, best price vs Bet365 no-vig ==")
    print(roi_table(b, ths).to_string(index=False))
    print("\n== 1X2, top-5 leagues only ==")
    print(roi_table(b[b["Division"].isin(TOP5)], ths).to_string(index=False))
    print("\n== 1X2, edge>=0.03 by era ==")
    b["era"] = pd.cut(b["season"], [2004, 2011, 2018, 2030], labels=["2005-11", "2012-18", "2019-26"])
    for era, g in b.groupby("era", observed=True):
        s = g[g["edge"] >= 0.03]
        m, lo, hi = bootstrap_mean_ci(s["pnl"].to_numpy(), n_boot=500)
        print(f"  {era}: n={len(s):6d} roi={m:+.4f} [{lo:+.4f},{hi:+.4f}]  model_ev={s['ev'].mean():+.4f}")
    print("\n== 1X2, edge>=0.03 by outcome ==")
    for oc, g in b.groupby("outcome"):
        s = g[g["edge"] >= 0.03]
        m, lo, hi = bootstrap_mean_ci(s["pnl"].to_numpy(), n_boot=500)
        print(f"  {oc}: n={len(s):6d} roi={m:+.4f} [{lo:+.4f},{hi:+.4f}] realised={s['win'].mean():.3f} consensus={s['p_fair'].mean():.3f}")
    print("\n== 1X2: does the consensus stay calibrated where the outlier disagrees? ==")
    s = b[b["edge"] >= 0.03]
    print(f"  n={len(s)} consensus p={s['p_fair'].mean():.4f} realised={s['win'].mean():.4f} "
          f"best-price implied={s['p_best'].mean():.4f}")
    # Control: betting Bet365's own price at the same rows
    s = s.copy(); s["pnl_b365"] = s["win"] * s["b365"] - 1
    print(f"  same rows at Bet365 price: roi={s['pnl_b365'].mean():+.4f}")

    bo = build_bets_ou(df)
    print(f"\nO/U 2.5 bets universe: {len(bo):,} rows")
    print("== O/U 2.5, all divisions ==")
    print(roi_table(bo, ths).to_string(index=False))
    print("\n== O/U 2.5, top-5 leagues ==")
    print(roi_table(bo[bo["Division"].isin(TOP5)], ths).to_string(index=False))


if __name__ == "__main__":
    main()
