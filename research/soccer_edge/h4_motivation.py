"""H4: late-season motivation asymmetries.

For every league match we reconstruct the table as it stood before kick-off and flag
teams that are mathematically "dead" (safe from relegation and unable to reach the
European places) with few rounds left. We then ask two questions:

1. Main market: does Bet365's 1X2 price already reflect a dead team facing a
   motivated one? (implied vs realised win rate, ROI of backing the motivated side)
2. Side markets: do corners, cards and goals in these matches deviate from what a
   rolling-mean model (the assumed prop-market baseline) predicts?

Standing rules are approximated per division (relegation zone incl. playoff spot,
top-6 as "European places"). See README for the caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import EURO_DIVISIONS, bootstrap_mean_ci, load_odds_matches, no_vig

# Relegation-risk zone size (direct + playoff) and "European places" cut-off.
RELEGATION_ZONE = {"E0": 3, "E1": 3, "SP1": 3, "SP2": 4, "I1": 3, "I2": 4, "D1": 3, "D2": 3,
                   "F1": 3, "F2": 3, "N1": 3, "P1": 3, "B1": 2, "T1": 3, "G1": 3, "SC0": 2}
EURO_PLACES = 6


def standings_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match table context to every match of every (division, season).

    Adds, for home (``H``) and away (``A``): points, matches played, matches remaining,
    position, ``safe`` (cannot be relegated), ``no_europe`` (cannot reach top-6),
    ``no_title`` and ``dead`` = safe & no_europe.

    Seasons with an incomplete fixture list (fewer than 90% of expected matches) are
    dropped because "matches remaining" would be wrong.
    """
    out = []
    df = df.sort_values(["Division", "season", "MatchDate"]).copy()
    for (div, season), g in df.groupby(["Division", "season"], sort=False):
        teams = pd.unique(pd.concat([g["HomeTeam"], g["AwayTeam"]]))
        n = len(teams)
        expected = n * (n - 1)
        if len(g) < 0.9 * expected or n < 10:
            continue
        per_team_total = 2 * (n - 1)
        rz = RELEGATION_ZONE.get(div, 3)
        pts = dict.fromkeys(teams, 0)
        played = dict.fromkeys(teams, 0)
        gd = dict.fromkeys(teams, 0)
        rows = []
        for date, day in g.groupby("MatchDate", sort=True):
            # Snapshot before this date's matches.
            table = sorted(teams, key=lambda t: (-pts[t], -gd[t]))
            pos = {t: i + 1 for i, t in enumerate(table)}
            pts_at = {i + 1: pts[t] for i, t in enumerate(table)}
            rem_at = {i + 1: per_team_total - played[t] for i, t in enumerate(table)}
            # A team is safe if its points exceed the max achievable by the team currently
            # at the top of the relegation zone (conservative: that team could still climb).
            safe_line_pos = n - rz + 1
            safe_thresh = pts_at[safe_line_pos] + 3 * rem_at[safe_line_pos]
            euro_pts = pts_at[EURO_PLACES]
            top_pts = pts_at[1]
            for _, m in day.iterrows():
                r = {"idx": m.name}
                for side, team in (("H", m["HomeTeam"]), ("A", m["AwayTeam"])):
                    p, pl = pts[team], played[team]
                    rem = per_team_total - pl
                    mx = p + 3 * rem
                    r[f"{side}_pts"] = p; r[f"{side}_played"] = pl; r[f"{side}_rem"] = rem
                    r[f"{side}_pos"] = pos[team]
                    r[f"{side}_safe"] = p > safe_thresh
                    r[f"{side}_no_europe"] = mx < euro_pts
                    r[f"{side}_no_title"] = mx < top_pts
                    r[f"{side}_dead"] = bool(r[f"{side}_safe"] and r[f"{side}_no_europe"])
                    r[f"{side}_in_rz"] = pos[team] >= safe_line_pos
                    r[f"{side}_title_race"] = (not r[f"{side}_no_title"]) and pos[team] <= 3
                rows.append(r)
            for _, m in day.iterrows():
                h, a = m["HomeTeam"], m["AwayTeam"]
                fh, fa = m["FTHome"], m["FTAway"]
                if pd.isna(fh) or pd.isna(fa):
                    continue
                played[h] += 1; played[a] += 1
                gd[h] += fh - fa; gd[a] += fa - fh
                if fh > fa:
                    pts[h] += 3
                elif fh < fa:
                    pts[a] += 3
                else:
                    pts[h] += 1; pts[a] += 1
        out.append(pd.DataFrame(rows).set_index("idx"))
    feats = pd.concat(out)
    return df.join(feats, how="inner")


def add_rolling_baselines(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Strictly-prior rolling team means for corners, yellows, goals (for and against)."""
    long = pd.concat([
        pd.DataFrame({"idx": df.index, "team": df["HomeTeam"], "date": df["MatchDate"],
                      "season": df["season"], "cf": df["HomeCorners"], "ca": df["AwayCorners"],
                      "yf": df["HomeYellow"], "ya": df["AwayYellow"], "gf": df["FTHome"],
                      "ga": df["FTAway"], "side": "H"}),
        pd.DataFrame({"idx": df.index, "team": df["AwayTeam"], "date": df["MatchDate"],
                      "season": df["season"], "cf": df["AwayCorners"], "ca": df["HomeCorners"],
                      "yf": df["AwayYellow"], "ya": df["HomeYellow"], "gf": df["FTAway"],
                      "ga": df["FTHome"], "side": "A"}),
    ]).sort_values(["team", "date"])
    for c in ["cf", "ca", "yf", "ya", "gf", "ga"]:
        long[f"r_{c}"] = (long.groupby(["team", "season"])[c]
                          .transform(lambda s: s.shift(1).rolling(window, min_periods=4).mean()))
    h = long[long["side"] == "H"].set_index("idx")[[f"r_{c}" for c in ["cf", "ca", "yf", "ya", "gf", "ga"]]]
    a = long[long["side"] == "A"].set_index("idx")[[f"r_{c}" for c in ["cf", "ca", "yf", "ya", "gf", "ga"]]]
    h.columns = [f"H_{c}" for c in h.columns]; a.columns = [f"A_{c}" for c in a.columns]
    df = df.join(h).join(a)
    # Naive expected totals: average of "for" of one side and "against" of the other.
    df["exp_corners"] = (df["H_r_cf"] + df["A_r_ca"]) / 2 + (df["A_r_cf"] + df["H_r_ca"]) / 2
    df["exp_yellows"] = (df["H_r_yf"] + df["A_r_ya"]) / 2 + (df["A_r_yf"] + df["H_r_ya"]) / 2
    df["exp_goals"] = (df["H_r_gf"] + df["A_r_ga"]) / 2 + (df["A_r_gf"] + df["H_r_ga"]) / 2
    df["tot_corners"] = df["HomeCorners"] + df["AwayCorners"]
    df["tot_yellows"] = df["HomeYellow"] + df["AwayYellow"]
    df["tot_goals"] = df["FTHome"] + df["FTAway"]
    return df


def main() -> None:
    pd.set_option("display.width", 220); pd.set_option("display.precision", 4)
    df = load_odds_matches()
    df = df[df["Division"].isin(EURO_DIVISIONS) & (df["season"] >= 2005) & (df["season"] <= 2025)]
    df = df.dropna(subset=["FTHome", "FTAway", "OddHome", "OddDraw", "OddAway"])
    df = standings_features(df)
    df = add_rolling_baselines(df)
    print(f"matches with standings context: {len(df):,}")

    late = df[(df["H_rem"] <= 5) & (df["A_rem"] <= 5)].copy()
    late["type"] = np.select(
        [late["H_dead"] & ~late["A_dead"], late["A_dead"] & ~late["H_dead"],
         late["H_dead"] & late["A_dead"]],
        ["home_dead", "away_dead", "both_dead"], default="both_alive")
    print("\nlast-5-rounds matches by motivation type:")
    print(late["type"].value_counts().to_string())

    # ---- 1. main market -----------------------------------------------------------
    p = no_vig(late[["OddHome", "OddDraw", "OddAway"]].to_numpy())
    late["pH"], late["pD"], late["pA"] = p[:, 0], p[:, 1], p[:, 2]
    print("\n== Main market: motivated side vs dead side (Bet365 no-vig implied vs realised) ==")
    for t, g in late.groupby("type"):
        if t == "home_dead":
            imp, real = g["pA"], (g["FTResult"] == "A").astype(float)
            pnl = real * g["OddAway"] - 1; pnl_max = real * g["MaxAway"] - 1
            imp_d, real_d = g["pH"], (g["FTResult"] == "H").astype(float)
        elif t == "away_dead":
            imp, real = g["pH"], (g["FTResult"] == "H").astype(float)
            pnl = real * g["OddHome"] - 1; pnl_max = real * g["MaxHome"] - 1
            imp_d, real_d = g["pA"], (g["FTResult"] == "A").astype(float)
        else:
            continue
        m, lo, hi = bootstrap_mean_ci(pnl.to_numpy())
        mm, mlo, mhi = bootstrap_mean_ci(pnl_max.dropna().to_numpy())
        print(f"  {t:10s} n={len(g):5d} motivated: implied={imp.mean():.3f} realised={real.mean():.3f} "
              f"| dead: implied={imp_d.mean():.3f} realised={real_d.mean():.3f} "
              f"| ROI back motivated @B365={m:+.3f} [{lo:+.3f},{hi:+.3f}] @best={mm:+.3f} [{mlo:+.3f},{mhi:+.3f}]")
    # Split by what the motivated side is playing for.
    print("\n  by motivated side's stake (relegation zone vs Europe/title chase):")
    for t, g in late.groupby("type"):
        if t not in ("home_dead", "away_dead"):
            continue
        ms = "A" if t == "home_dead" else "H"
        for stake, mask in (("in relegation zone", g[f"{ms}_in_rz"]),
                            ("title/top-3 race", g[f"{ms}_title_race"]),
                            ("other (europe chase)", ~g[f"{ms}_in_rz"] & ~g[f"{ms}_title_race"])):
            s = g[mask]
            if len(s) < 30:
                continue
            real = (s["FTResult"] == ms).astype(float)
            imp = s[f"p{ms}"]
            pnl = real * s["OddHome" if ms == "H" else "OddAway"] - 1
            m, lo, hi = bootstrap_mean_ci(pnl.to_numpy())
            print(f"    {t:10s} {stake:22s} n={len(s):4d} implied={imp.mean():.3f} realised={real.mean():.3f} ROI={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # Calibration control: same implied-prob bucket outside the late season.
    early = df[(df["H_rem"] > 8) & (df["A_rem"] > 8)].copy()
    pe = no_vig(early[["OddHome", "OddDraw", "OddAway"]].to_numpy())
    early["pH"], early["pA"] = pe[:, 0], pe[:, 2]
    print("\n  control (rounds with >8 remaining): favourites at similar implied prob")
    for side, oc in (("pH", "H"), ("pA", "A")):
        for lo_, hi_ in ((0.35, 0.45), (0.45, 0.55), (0.55, 0.7)):
            s = early[(early[side] >= lo_) & (early[side] < hi_)]
            print(f"    {oc} implied in [{lo_},{hi_}): n={len(s):6d} implied={s[side].mean():.3f} realised={(s['FTResult']==oc).mean():.3f}")

    # ---- 2. side markets -----------------------------------------------------------
    print("\n== Side markets: residual vs rolling-mean baseline (actual - expected) ==")
    base = df.dropna(subset=["exp_corners", "exp_yellows", "exp_goals"]).copy()
    base["type"] = np.select(
        [(base["H_rem"] <= 5) & (base["A_rem"] <= 5) & base["H_dead"] & ~base["A_dead"],
         (base["H_rem"] <= 5) & (base["A_rem"] <= 5) & base["A_dead"] & ~base["H_dead"],
         (base["H_rem"] <= 5) & (base["A_rem"] <= 5) & base["H_dead"] & base["A_dead"],
         (base["H_rem"] <= 5) & (base["A_rem"] <= 5)],
        ["home_dead", "away_dead", "both_dead", "late_both_alive"], default="regular")
    for tgt, exp in (("tot_corners", "exp_corners"), ("tot_yellows", "exp_yellows"),
                     ("tot_goals", "exp_goals")):
        print(f"\n  {tgt}: (baseline MAE regular = {np.abs(base.loc[base['type']=='regular', tgt]-base.loc[base['type']=='regular', exp]).mean():.3f})")
        for t, g in base.groupby("type"):
            res = (g[tgt] - g[exp]).to_numpy()
            m, lo, hi = bootstrap_mean_ci(res)
            print(f"    {t:16s} n={len(g):6d} actual={g[tgt].mean():.3f} expected={g[exp].mean():.3f} resid={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        # per-team split for the dead side vs the motivated side
        if tgt != "tot_goals":
            col_h, col_a = ("HomeCorners", "AwayCorners") if tgt == "tot_corners" else ("HomeYellow", "AwayYellow")
            rh, ra = ("H_r_cf", "A_r_cf") if tgt == "tot_corners" else ("H_r_yf", "A_r_yf")
            for t, ms in (("home_dead", "H"), ("away_dead", "A")):
                g = base[base["type"] == t]
                dead_res = (g[col_h] - g[rh]) if ms == "H" else (g[col_a] - g[ra])
                mot_res = (g[col_a] - g[ra]) if ms == "H" else (g[col_h] - g[rh])
                dm = bootstrap_mean_ci(dead_res.to_numpy()); mm = bootstrap_mean_ci(mot_res.to_numpy())
                print(f"      {t}: dead team resid={dm[0]:+.3f} [{dm[1]:+.3f},{dm[2]:+.3f}]  motivated team resid={mm[0]:+.3f} [{mm[1]:+.3f},{mm[2]:+.3f}]")

    # Over 2.5 market in dead rubbers
    ou = base.dropna(subset=["Over25", "Under25"]).copy()
    po = no_vig(ou[["Over25", "Under25"]].to_numpy()); ou["pOver"] = po[:, 0]
    print("\n== O/U 2.5 market in dead rubbers (implied over prob vs realised) ==")
    for t, g in ou.groupby("type"):
        real = (g["tot_goals"] > 2.5).astype(float)
        pnl = real * g["Over25"] - 1
        m, lo, hi = bootstrap_mean_ci(pnl.to_numpy())
        print(f"    {t:16s} n={len(g):6d} implied_over={g['pOver'].mean():.3f} realised_over={real.mean():.3f} ROI over @B365={m:+.3f} [{lo:+.3f},{hi:+.3f}]")

    late.to_parquet(__import__("common").data_dir() / "processed_h4_late.parquet", index=False)


if __name__ == "__main__":
    main()
