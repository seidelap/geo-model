"""06 The haircut, reconciled: one definition, one bar, applied to every stage.

Protocol step 4 makes every EV claim in this phase conditional on a number measured on the
one count market whose line is observed: how much a real bookmaker beats the identical
rolling-mean proxy on total goals. Stages 01-04 each measured it, but they expressed it in
money three different ways and on overlapping universes, so "does this market survive the
haircut" depended on which stage happened to compute it. Two readers could read the same
tables and disagree about whether the corner team line clears the bar.

This module removes that freedom. It

1. measures how independent the four stages' goals universes actually are, directly from
   the odds table, so the phrase "measured four times" can be replaced by what is true;
2. collects the three haircut definitions into one table with one sign convention, per
   hold and per split;
3. names one of them the phase reference -- the **margin-matched round trip**, which
   charges both the profit that evaporates and the loss that appears while holding the
   bookmaker's margin fixed -- and
4. applies all three, uniformly, to every confirmation betting row the phase published,
   including the scenario-level rows, so the haircut-dependence of each conclusion is
   visible instead of implicit.

Nothing here refits a model. Every input is a committed result table; the only raw input
is the odds table, read for the universe overlap.

Run::

    cd /home/user/geo-model
    python -m research.scenario_ev.haircut            # ~20 s

Outputs ``reports/06_haircut_universes.parquet``, ``06_haircut_nats.parquet``,
``06_haircut_definitions.parquet``, ``06_haircut_applied.parquet`` and the report
``reports/06_haircut.md``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.scenario_ev import common as C

DISC, CONF = C.DISCOVERY, C.CONFIRMATION

#: The phase's reference definition. See :data:`DEFINITION_NOTES`.
REFERENCE = "margin_matched_round_trip"

DEFINITION_NOTES: dict[str, str] = {
    "book_earns_one_way": (
        "ROI a real book makes betting its de-vigged goals price into prices built from "
        "the rolling-mean proxy at the stated hold. One-directional: it charges the "
        "profit that evaporates but not the loss that appears, so it is a LOWER BOUND."),
    "margin_matched_round_trip": (
        "ROI(our goals model vs proxy prices at hold h) minus ROI(our goals model vs the "
        "real book's no-vig price re-priced at the same hold h). Both legs carry the same "
        "margin, so the difference is sharpness alone. This is the phase reference."),
    "actual_prices_round_trip": (
        "the same round trip, but the second leg uses Bet365's posted prices at whatever "
        "hold Bet365 actually charged. It mixes a margin difference into the sharpness "
        "difference whenever that hold differs from h."),
}


@dataclass(frozen=True)
class HaircutConfig:
    """Configuration of the reconciliation.

    Attributes:
        holds: Two-way overrounds at which the phase's simulations were run.
        reference: Key of :data:`DEFINITION_NOTES` used as the phase bar.
        headline_hold: The hold quoted in the synthesis headline.
    """

    holds: tuple[float, ...] = (0.04, 0.06, 0.08)
    reference: str = REFERENCE
    headline_hold: float = 0.06


CFG = HaircutConfig()


# ---------------------------------------------------------------------------
# (1) How independent are the four goals universes?
# ---------------------------------------------------------------------------


#: Each stage's chronological cut, read from its own setup table (``date_cut`` in
#: ``01_cards_c_setup``; the corner and foul stages cut at 2019-08-31, as their reports
#: state). The cut is what makes the card measurement a different *era* from the other
#: two, which matters more than the match pools do.
SPLIT_DATES: dict[str, str] = {
    "01 cards goals check": "2017-11-30",
    "02 corners goals check": "2019-08-31",
    "04 fouls goals check": "2019-08-31",
}


def universe_masks(odds: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks [n_matches] of each stage's goals-honesty *pool*.

    These are pools, not the stages' scored rows: every stage additionally requires both
    teams to have enough strictly-prior matches for the rolling proxy, which removes a few
    thousand more rows. The point of reconstructing them here is to check the synthesis's
    independence claim against the data, so the reconstruction deliberately uses the
    stages' stated filters rather than reading their row counts back.

    Note one filter that is easy to get wrong and that the synthesis originally got wrong:
    stage 01's goals check does **not** condition on card counts. It conditions on having
    a goals result, a Bet365 price and prior history, so its pool is the whole priced
    table. Stages 02 and 04 do condition on their own market being recorded.

    Args:
        odds: Raw football-data odds table.

    Returns:
        Mapping from stage label to a boolean mask [n_matches].
    """
    ou = odds["Over25"].notna().to_numpy() & odds["Under25"].notna().to_numpy()
    goals = odds["FTHome"].notna().to_numpy() & odds["FTAway"].notna().to_numpy()
    def rec(a: str, b: str) -> np.ndarray:
        return odds[a].notna().to_numpy() & odds[b].notna().to_numpy()
    return {
        "01 cards goals check": ou & goals,
        "02 corners goals check": ou & goals & rec("HomeCorners", "AwayCorners"),
        "04 fouls goals check": ou & goals & rec("HomeFouls", "AwayFouls"),
    }


def confirmation_overlap(odds: pd.DataFrame) -> pd.DataFrame:
    """Overlap of the stages' *confirmation* halves, which is what the headline is scored on.

    Two stages can share 99% of a match pool and still score on almost disjoint rows if
    their chronological cuts differ, which is the case for cards (cut 2017-11-30) against
    corners and fouls (cut 2019-08-31).

    Args:
        odds: Raw football-data odds table.

    Returns:
        One row per ordered pair with the confirmation-half sizes, their intersection and
        the share of the first that the intersection covers.
    """
    masks = universe_masks(odds)
    date = pd.to_datetime(odds["MatchDate"]).to_numpy()
    conf = {k: m & (date > np.datetime64(SPLIT_DATES[k]))
            for k, m in masks.items()}
    rows: list[dict[str, object]] = []
    for a, ma in conf.items():
        for b, mb in conf.items():
            if a == b:
                continue
            inter = int((ma & mb).sum())
            rows.append({"universe_a": a, "universe_b": b,
                         "split_a": SPLIT_DATES[a], "split_b": SPLIT_DATES[b],
                         "n_conf_a": int(ma.sum()), "n_conf_b": int(mb.sum()),
                         "n_intersection": inter,
                         "share_of_a": round(inter / max(int(ma.sum()), 1), 4)})
    return pd.DataFrame(rows)


def universe_overlap(odds: pd.DataFrame) -> pd.DataFrame:
    """Pairwise overlap of the stages' goals universes.

    Args:
        odds: Raw football-data odds table.

    Returns:
        One row per ordered pair with the sizes, the intersection, the Jaccard index and
        whether the first universe is a subset of the second.
    """
    masks = universe_masks(odds)
    rows: list[dict[str, object]] = []
    for a, ma in masks.items():
        for b, mb in masks.items():
            if a == b:
                continue
            inter = int((ma & mb).sum())
            union = int((ma | mb).sum())
            rows.append({"universe_a": a, "universe_b": b, "n_a": int(ma.sum()),
                         "n_b": int(mb.sum()), "n_intersection": inter,
                         "jaccard": round(inter / max(union, 1), 4),
                         "a_subset_of_b": bool(inter == int(ma.sum()))})
    return pd.DataFrame(rows)


def realised_holds(odds: pd.DataFrame) -> pd.DataFrame:
    """Bet365's realised two-way overround on the Over/Under 2.5 price, by era and universe.

    The round-trip haircuts are differenced against simulations run at a stated synthetic
    hold, so the hold Bet365 actually charged on the rows a stage used is part of reading
    that stage's number.

    Args:
        odds: Raw football-data odds table.

    Returns:
        One row per (universe, era) with the mean overround and n.
    """
    masks = universe_masks(odds)
    date = pd.to_datetime(odds["MatchDate"])
    over = 1.0 / odds["Over25"].to_numpy(dtype=float)
    under = 1.0 / odds["Under25"].to_numpy(dtype=float)
    orr = over + under - 1.0
    eras = {"all years": np.ones(len(odds), dtype=bool),
            "cards confirmation (from 2017-12)":
                (date > pd.Timestamp(SPLIT_DATES["01 cards goals check"])).to_numpy(),
            "corner/foul confirmation (from 2019-09)":
                (date > pd.Timestamp(SPLIT_DATES["02 corners goals check"])).to_numpy()}
    rows: list[dict[str, object]] = []
    for uname, m in masks.items():
        for ename, e in eras.items():
            sel = m & e & np.isfinite(orr)
            if not sel.any():
                continue
            rows.append({"universe": uname, "era": ename, "n": int(sel.sum()),
                         "mean_overround": float(np.mean(orr[sel]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (2) The nats measurement and (3) the three ROI definitions
# ---------------------------------------------------------------------------


def nats_table(rd: Path) -> pd.DataFrame:
    """The four stages' log-loss gaps between Bet365 and the rolling-mean goals proxy.

    Args:
        rd: Reports directory.

    Returns:
        One row per (stage, split) with the gap in nats and its interval.
    """
    rows: list[dict[str, object]] = []

    g = pd.read_parquet(rd / "01_cards_c_goals_gap.parquet")
    for half in ("discovery", "confirmation"):
        r = g[(g["where"] == half) & (g["model"] == "bet365_novig")].iloc[0]
        rows.append({"stage": "01 cards", "universe": "01 cards goals check",
                     "split": half, "n": int(r["n"]), "gap_nats": float(r["delta_vs_proxy"]),
                     "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                     "source_table": "01_cards_c_goals_gap"})

    for stage, uni, fn in (("02 corners", "02 corners goals check",
                            "02_corners_goals_haircut_metrics.parquet"),
                           ("04 fouls", "04 fouls goals check",
                            "04_fouls_d_goals_metrics.parquet")):
        m = pd.read_parquet(rd / fn)
        for half in (DISC, CONF):
            r = m[(m["split"] == half) & (m["model"] == "bet365_novig")].iloc[0]
            rows.append({"stage": stage, "universe": uni, "split": half, "n": int(r["n"]),
                         "gap_nats": float(r["delta_vs_proxy_cal"]),
                         "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                         "source_table": fn.replace(".parquet", "")})

    p = pd.read_parquet(rd / "03_pass_goals_gap.parquet")
    p = p[p["population"] == "four_leagues_all_seasons"]
    for half in ("discovery", "confirmation"):
        r = p[(p["where"] == half) & (p["model"] == "bet365_novig")].iloc[0]
        rows.append({"stage": "03 pass counts",
                     "universe": "03 four leagues (nested in 02 / 04)", "split": half,
                     "n": int(r["n"]), "gap_nats": float(r["delta_vs_proxy_cal"]),
                     "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                     "source_table": "03_pass_goals_gap"})
    return pd.DataFrame(rows)


def definition_table(rd: Path, cfg: HaircutConfig = CFG) -> pd.DataFrame:
    """Every haircut the phase measured, in one sign convention, per hold and split.

    Args:
        rd: Reports directory.
        cfg: Configuration.

    Returns:
        One row per (definition, stage, split, hold) with ``haircut_roi_points`` and the
        two ROI legs it is built from where they are published.
    """
    rows: list[dict[str, object]] = []

    # one-directional book-earns: stages 02 and 03
    r2 = pd.read_parquet(rd / "02_corners_goals_haircut_roi.parquet")
    for _, r in r2[(r2["threshold"] == 0.02)].iterrows():
        rows.append({"definition": "book_earns_one_way", "stage": "02 corners",
                     "universe": "02 corners goals check", "split": str(r["split"]),
                     "hold": float(r["hold"]), "haircut_roi_points": float(r["roi"]),
                     "leg_vs_proxy": float(r["roi"]), "leg_vs_book": float("nan"),
                     "n_bets": int(r["n_bets"]),
                     "source_table": "02_corners_goals_haircut_roi"})
    p3 = pd.read_parquet(rd / "03_pass_goals_haircut.parquet").iloc[0]
    for hold in cfg.holds:
        rows.append({"definition": "book_earns_one_way", "stage": "03 pass counts",
                     "universe": "03 four leagues (nested in 02 / 04)", "split": CONF,
                     "hold": float(hold),
                     "haircut_roi_points": float(p3["roi_haircut"]),
                     "leg_vs_proxy": float(p3["roi_book_vs_proxy_mean_over_holds"]),
                     "leg_vs_book": float("nan"), "n_bets": -1,
                     "source_table": "03_pass_goals_haircut (mean over holds)"})

    # round trips: stage 04 publishes both, stage 01 publishes both once the
    # margin-matched leg is fitted (see cards._goals_honesty)
    f4 = pd.read_parquet(rd / "04_fouls_d_haircut.parquet")
    for _, r in f4.iterrows():
        rows.append({"definition": "margin_matched_round_trip", "stage": "04 fouls",
                     "universe": "04 fouls goals check", "split": str(r["split"]),
                     "hold": float(r["hold"]),
                     "haircut_roi_points": float(r["haircut_roi_points"]),
                     "leg_vs_proxy": float(r["roi_vs_proxy"]),
                     "leg_vs_book": float(r["roi_vs_real_book_matched"]),
                     "n_bets": -1, "source_table": "04_fouls_d_haircut"})
        rows.append({"definition": "actual_prices_round_trip", "stage": "04 fouls",
                     "universe": "04 fouls goals check", "split": str(r["split"]),
                     "hold": float(r["hold"]),
                     "haircut_roi_points": float(r["haircut_roi_points_actual_prices"]),
                     "leg_vs_proxy": float(r["roi_vs_proxy"]),
                     "leg_vs_book": float(r["roi_vs_real_book_actual_prices"]),
                     "n_bets": -1, "source_table": "04_fouls_d_haircut"})

    c1p = rd / "01_cards_c_haircut_by_hold.parquet"
    if c1p.exists():
        c1 = pd.read_parquet(c1p)
        for _, r in c1.iterrows():
            rows.append({"definition": "margin_matched_round_trip", "stage": "01 cards",
                         "universe": "01 cards goals check", "split": str(r["where"]),
                         "hold": float(r["hold"]),
                         "haircut_roi_points": float(r["haircut_roi_points_matched"]),
                         "leg_vs_proxy": float(r["roi_vs_proxy"]),
                         "leg_vs_book": float(r["roi_vs_real_book_matched"]),
                         "n_bets": -1, "source_table": "01_cards_c_haircut_by_hold"})
            rows.append({"definition": "actual_prices_round_trip", "stage": "01 cards",
                         "universe": "01 cards goals check", "split": str(r["where"]),
                         "hold": float(r["hold"]),
                         "haircut_roi_points": float(r["haircut_roi_points_actual_prices"]),
                         "leg_vs_proxy": float(r["roi_vs_proxy"]),
                         "leg_vs_book": float(r["roi_vs_real_book_actual_prices"]),
                         "n_bets": -1, "source_table": "01_cards_c_haircut_by_hold"})
    return pd.DataFrame(rows)


def reference_haircuts(defs: pd.DataFrame, cfg: HaircutConfig = CFG) -> pd.DataFrame:
    """The phase bar per hold: the reference definition on each universe that measured it.

    Args:
        defs: Output of :func:`definition_table`.
        cfg: Configuration.

    Returns:
        One row per (hold, universe) with the confirmation reference haircut.
    """
    d = defs[(defs["definition"] == cfg.reference) & (defs["split"] == CONF)]
    out = (d.groupby(["hold", "stage", "universe"], as_index=False)["haircut_roi_points"]
           .first().sort_values(["hold", "stage"]).reset_index(drop=True))
    return out


# ---------------------------------------------------------------------------
# (4) Apply all three to every published confirmation betting row
# ---------------------------------------------------------------------------


def headline_bets(rd: Path) -> pd.DataFrame:
    """Every confirmation betting row the phase published, in one schema.

    The corners table carries a ``scenario`` column and the others do not; scenario rows
    are kept, because the phase's question is about specific scenarios and the largest
    single after-haircut number in the phase lives in one of them.

    Args:
        rd: Reports directory.

    Returns:
        One row per (candidate, market, scenario, source, hold, threshold) on the
        confirmation split.
    """
    rows: list[pd.DataFrame] = []

    a = pd.read_parquet(rd / "01_cards_a_bets.parquet")
    a = a[(a["where"] == CONF) & (a["book"] == "P0_proxy") & (a["model"] == "P2")].copy()
    a["candidate"], a["scenario"], a["source"] = "cards", "all", "binary_direct"
    a["market"] = "player_carded_0.5"
    rows.append(a)

    c = pd.read_parquet(rd / "01_cards_c_bets.parquet")
    c = c[(c["where"] == CONF) & (c["model"] == "C2_event_plus_market")].copy()
    c["candidate"], c["scenario"], c["source"] = "cards", "all", "count_nb"
    c["market"] = "match_cards_proxy_line"
    rows.append(c)

    b = pd.read_parquet(rd / "02_corners_bets.parquet")
    b = b[(b["split"] == CONF) & (b["source"] == "count_nb")
          & (b["threshold"] == 0.02)].copy()
    b["candidate"], b["where"] = "corners", CONF
    rows.append(b)

    p = pd.read_parquet(rd / "03_pass_c_confirmation_roi.parquet")
    p = p[p["where"] == CONF].copy()
    p["candidate"], p["scenario"] = "pass_counts", "all"
    rows.append(p)

    f = pd.read_parquet(rd / "04_fouls_d_bets.parquet")
    f = f[f["split"] == CONF].copy()
    f["candidate"], f["scenario"], f["source"] = "fouls", "all", "count_nb"
    f["where"] = CONF
    rows.append(f)

    keep = ["candidate", "market", "scenario", "source", "hold", "threshold", "n_rows",
            "n_bets", "bet_rate", "mean_edge", "roi", "roi_lo", "roi_hi"]
    out = pd.concat([r.reindex(columns=keep) for r in rows], ignore_index=True)
    return out.sort_values(["candidate", "market", "scenario", "hold"]).reset_index(drop=True)


#: Which goals universe each candidate's betting rows are haircut against. The corner,
#: foul and pass simulations all run on the odds table, whose goals universe stages 02 and
#: 04 measured (and of which the pass universe is a subset); the card simulations run on
#: the card universe, which is a different match set with a different split date.
CANDIDATE_UNIVERSE: dict[str, str] = {
    "cards": "01 cards goals check",
    "corners": "04 fouls goals check",
    "fouls": "04 fouls goals check",
    "pass_counts": "04 fouls goals check",
}


def apply_haircuts(bets: pd.DataFrame, defs: pd.DataFrame,
                   cfg: HaircutConfig = CFG) -> pd.DataFrame:
    """Subtract all three haircut definitions from every confirmation betting row.

    Args:
        bets: Output of :func:`headline_bets`.
        defs: Output of :func:`definition_table`.
        cfg: Configuration.

    Returns:
        ``bets`` with one ``after_<definition>`` column per definition, a
        ``haircut_<definition>`` column recording what was subtracted, and
        ``survives_reference`` / ``survives_reference_lo`` flags.
    """
    d = defs[defs["split"] == CONF]
    out = bets.copy()
    for name in DEFINITION_NOTES:
        vals, hairs = [], []
        for _, r in out.iterrows():
            uni = CANDIDATE_UNIVERSE[str(r["candidate"])]
            sel = d[(d["definition"] == name) & (d["hold"] == float(r["hold"]))]
            pick = sel[sel["universe"] == uni]
            if not len(pick):
                pick = sel
            h = float(pick["haircut_roi_points"].iloc[0]) if len(pick) else float("nan")
            hairs.append(h)
            vals.append(float(r["roi"]) - h)
        out[f"haircut_{name}"] = hairs
        out[f"after_{name}"] = vals
        out[f"after_{name}_lo"] = out["roi_lo"].to_numpy(dtype=float) - np.array(hairs)
    ref = f"after_{cfg.reference}"
    out["survives_reference"] = out[ref] > 0
    out["survives_reference_lo"] = out[f"{ref}_lo"] > 0
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def stage_report(rd: Path, cfg: HaircutConfig = CFG) -> Path:
    """Render ``reports/06_haircut.md`` from the tables this module writes.

    Args:
        rd: Reports directory.
        cfg: Configuration.

    Returns:
        Path of the written report.
    """
    from research.privileged_tracking.common.report import md_table

    uni = pd.read_parquet(rd / "06_haircut_universes.parquet")
    conf_uni = pd.read_parquet(rd / "06_haircut_confirmation_overlap.parquet")
    holds_t = pd.read_parquet(rd / "06_haircut_holds.parquet")
    nats = pd.read_parquet(rd / "06_haircut_nats.parquet")
    defs = pd.read_parquet(rd / "06_haircut_definitions.parquet")
    app = pd.read_parquet(rd / "06_haircut_applied.parquet")

    out: list[str] = []
    A = out.append
    A("# 06 The haircut, reconciled: one definition, one bar")
    A("")
    A("Code: `research/scenario_ev/haircut.py`. Inputs: the committed result tables of "
      "stages 01-04 plus the raw odds table (for the universe overlap only). Nothing "
      "here refits a model.")
    A("")
    A("```")
    A("cd /home/user/geo-model")
    A("python -m research.scenario_ev.haircut     # ~20 s")
    A("```")
    A("")
    A("## 1. Why this exists")
    A("")
    A("Every EV claim in this phase is conditional on protocol step 4: a real bookmaker "
      "is sharper than a rolling mean, and by how much is measured on total goals, the "
      "one count market with an observed line. Stages 01-04 each ran that measurement, "
      "but expressed it in money three different ways on overlapping universes. Which "
      "market 'survives the haircut' therefore depended on which stage's arithmetic a "
      "reader picked up. This module fixes one definition and applies it to everything.")
    A("")
    A("## 2. How independent are the four measurements?")
    A("")
    A("The synthesis originally called the nats gap 'measured four times "
      "independently'. The odds table says otherwise:")
    A("")
    A(md_table(uni))
    A("")
    def pair(a: str, b: str) -> pd.Series:
        return uni[uni.universe_a.str.startswith(a)
                   & uni.universe_b.str.startswith(b)].iloc[0]

    fc, cc = pair("04", "02"), pair("02", "01")
    A(f"Read the `a_subset_of_b` column. The foul pool is a **subset** of the corner pool "
      f"({int(fc['n_intersection'])} of {int(fc['n_b'])} matches, Jaccard "
      f"{float(fc['jaccard']):.4f}), which is why stages 02 and 04 agree to five decimals "
      "on both the proxy and the Bet365 log-loss: they are the same measurement, run "
      f"twice. The corner pool is in turn a subset of the card pool "
      f"({int(cc['n_intersection'])} of {int(cc['n_b'])}), because stage 01's goals check "
      "does not condition on card counts at all -- it conditions on a result, a price and "
      "prior history, so its pool is the whole priced table. Stage 03's four-league "
      "universe is nested inside all of them. **The three pools are a chain of nested "
      "sets, not three samples.**")
    A("")
    A("What separates the card measurement from the corner/foul one is therefore not the "
      "matches available but the **chronological cut**, which puts the confirmation halves "
      "in different eras:")
    A("")
    A(md_table(conf_uni))
    A("")
    cnest = conf_uni[(conf_uni.universe_a.str.startswith("02"))
                     & (conf_uni.universe_b.str.startswith("01"))].iloc[0]
    A(f"And that does not separate them either: the corner/foul confirmation half is "
      f"{float(cnest['share_of_a']):.1%} contained in the card confirmation half "
      f"({int(cnest['n_conf_a'])} of its {int(cnest['n_conf_b'])} matches). The card "
      "measurement is the corner measurement plus about "
      f"{int(cnest['n_conf_b']) - int(cnest['n_conf_a']):,} extra matches from the "
      "2017-12 to 2019-08 window, which is why its gap is larger: that window is where "
      "the rolling mean falls furthest behind.")
    A("")
    A("**The phase has one construction, on one nested match pool, scored on two nested "
      "row sets** -- not four independent replications, and not even two. All four stages "
      "also call the same de-vig helper (`common.novig_two_way`) on the same Bet365 price "
      "columns. What the agreement does establish is that the number is not an artefact "
      "of one stage's proxy tuning, capacity choice or split date, since those differ "
      "between the stages and the answer does not. That is worth something. It is not "
      "four independent estimates, and the synthesis no longer says it is.")
    A("")
    A("Bet365's realised two-way overround on the Over/Under 2.5 price, which is what the "
      "actual-prices round trip charges:")
    A("")
    A(md_table(holds_t))
    A("")
    A("## 3. The gap in nats")
    A("")
    A(md_table(nats.sort_values(["split", "stage"])))
    A("")
    A("This is the robust, comparable form of the bar and the one the synthesis leans on: "
      "a real book is roughly 0.009-0.011 nats better than a shrunk, discovery-tuned, "
      "recalibrated rolling mean on a count market. Section 2 is the caveat to read it "
      "with: these are nested row sets scored over two overlapping eras, so the four "
      "intervals are not four independent estimates and the spread between 0.0089 and "
      "0.0111 is mostly the era, not sampling.")
    A("")
    A("## 4. The three definitions in ROI, side by side")
    A("")
    for k, v in DEFINITION_NOTES.items():
        A(f"* **`{k}`** -- {v}")
    A("")
    A(md_table(defs[defs["split"] == CONF]
               .sort_values(["definition", "stage", "hold"])
               [["definition", "stage", "universe", "hold", "haircut_roi_points",
                 "leg_vs_proxy", "leg_vs_book"]]))
    A("")
    ref6 = defs[(defs["definition"] == cfg.reference) & (defs["split"] == CONF)
                & (defs["hold"] == cfg.headline_hold)]
    parts = ", ".join(f"{r['stage']} {r['haircut_roi_points']:.4f}"
                      for _, r in ref6.iterrows())
    A(f"**The phase reference is `{cfg.reference}`** at the matching universe: "
      f"{parts} at a {cfg.headline_hold:.0%} hold. It is the definition a bettor actually "
      "experiences and the only one that holds the margin fixed between its two legs. The "
      "one-directional `book_earns_one_way` number is kept everywhere as a **lower "
      "bound**, never as the bar.")
    A("")
    A("## 5. Every published confirmation betting row, after each haircut")
    A("")
    A("`haircut_*` is what was subtracted; `after_*` is the point estimate that remains; "
      "`after_*_lo` applies the same subtraction to the ROI interval's lower bound. Card "
      "rows are haircut against the card universe, everything else against the "
      "corner/foul universe on which it was simulated.")
    A("")
    cols = ["candidate", "market", "scenario", "source", "hold", "n_bets", "bet_rate", "roi",
            "roi_lo", f"haircut_{cfg.reference}", f"after_{cfg.reference}",
            f"after_{cfg.reference}_lo", "after_book_earns_one_way",
            "after_actual_prices_round_trip"]
    A(md_table(app[cols]))
    A("")
    surv = app[app["survives_reference_lo"]]
    lb = app[app["after_book_earns_one_way"] > 0]
    A(f"**Under the phase reference, {len(surv)} of {len(app)} rows keep a lower bound "
      f"above zero**" + (": " + ", ".join(
          f"`{r['candidate']}/{r['market']}/{r['scenario']}/{r['source']}` at "
          f"{r['hold']:.0%} "
          f"({r[f'after_{cfg.reference}']:+.3f} [{r[f'after_{cfg.reference}_lo']:+.3f}])"
          for _, r in surv.iterrows()) if len(surv) else ", so nothing survives it.") + ".")
    A("")
    A(f"Under the one-directional lower bound {len(lb)} of {len(app)} rows have a "
      "positive point estimate, which is exactly the freedom this module removes: the "
      "lenient definition promotes rows the reference definition kills, and the two were "
      "measured on essentially the same matches.")
    A("")
    A("## 6. What this changes in the phase's conclusions")
    A("")
    A("* The corner **team** line splits. On all rows it clears the one-directional "
      "lower bound (+0.008 at a 6% hold) and does **not** clear the reference "
      "(-0.075 [-0.084]), so the phase's previously reported '+0.008 residual' was an "
      "artefact of the lenient definition. Inside `fav_strong` it clears the reference "
      "too (+0.055 [+0.035]) -- but that rule bets 99.95% of the matches in the "
      "scenario, so it is not a selective edge, it is the proxy being wrong about every "
      "strong-favourite match because it cannot see the 1X2 price. Stage 02's ablation "
      "puts 95% of the team model's gain in exactly that market block.")
    A("* The **pass-count** residual survives the reference haircut (+0.043 [+0.020] at "
      "a 6% hold). The haircut does not argue it away and the synthesis does not pretend "
      "it does. What argues it away is stated in stage 03: every feature carrying the "
      "advantage is a public rolling average, the model claims to disagree with the "
      "rolling mean about 1.8x as hard as a real book does on goals, and against a book "
      "allowed to set its own line at the goals-matched disagreement the ROI falls to "
      "+0.090 [+0.064, +0.118] before any haircut.")
    A("* One instrument points the other way and is reported rather than suppressed. "
      "`03_pass_c_relative_scale.parquet` normalises each market by its own proxy's log "
      "score: on goals a real book beats the proxy by 1.54% of the proxy's log loss, "
      "while our pass model beats its proxy by 2.00% of its count log score, so a book "
      "that was only *goals-equivalently* sharp on pass props would still score 0.0176 "
      "nats per player-match **worse** than our model. Stage 03 reports that ratio for "
      "completeness and explicitly declines to use it, because a count log score is "
      "dominated by irreducible entropy and its relative gaps are not comparable to a "
      "binary log loss at a line. It is quoted here so that a reader is not left with "
      "the impression that every instrument points at a null.")
    A("* Card and foul rows are negative under the reference definition at the 6% "
      "headline hold, which is what the synthesis already said. Two card rows turn "
      "slightly positive at 4% and 8% (+0.054 and +0.040) on point estimate only, with "
      "lower bounds of -0.035 and -0.089; the 6% row is -0.004 [-0.118]. The card "
      "conclusion is 'indistinguishable from zero and thin' rather than 'clearly "
      "negative', and the 0.4-0.9% bet rates are why.")
    A("")
    A("## 7. Limitations")
    A("")
    A("* The reference haircut is still a **transfer**: it is measured on goals and "
      "applied to corners, cards, fouls and pass props, on the assumption that a "
      "side-market book is as sharp relative to a rolling mean as a goals book is. Goals "
      "is the market a book works hardest on, so the transfer probably overstates a side "
      "book's sharpness; side-market margins are wider than the 5.8-6.6% observed here, "
      "which pushes the bar the other way. Neither correction is measurable in this data.")
    A("* A flat ROI subtraction across markets is crude. Bet rates differ by two orders "
      "of magnitude between the card rules (0.4-0.9%) and the pass rules (70-80%), and "
      "the haircut was measured at bet rates of 25-50%. The nats form of the bar "
      "(section 3) does not have this problem and is the form to prefer.")
    A("* The card universe's margin-matched leg re-prices Bet365's no-vig probability at "
      "the synthetic hold. That removes the margin mismatch but keeps the assumption that "
      "the de-vig is proportional; stage 01 checked Shin and additive de-vigs and found "
      "they move the nats gap by ten-thousandths.")
    A("")
    p = rd / "06_haircut.md"
    p.write_text("\n".join(out) + "\n")
    return p


def main(argv: list[str] | None = None) -> int:
    """Build every haircut reconciliation table and render the report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", help="dump the tables to stdout")
    args = ap.parse_args(argv)
    cfg = CFG
    rd = C.reports_dir()

    odds = pd.read_parquet(
        C.odds_path(),
        columns=["MatchDate", "Over25", "Under25", "FTHome", "FTAway",
                 "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls"])
    uni = universe_overlap(odds)
    conf_uni = confirmation_overlap(odds)
    holds_t = realised_holds(odds)
    del odds

    nats = nats_table(rd)
    defs = definition_table(rd, cfg)
    bets = headline_bets(rd)
    app = apply_haircuts(bets, defs, cfg)

    for name, df in (("universes", uni), ("confirmation_overlap", conf_uni),
                     ("holds", holds_t), ("nats", nats),
                     ("definitions", defs), ("applied", app),
                     ("reference", reference_haircuts(defs, cfg))):
        df.to_parquet(rd / f"06_haircut_{name}.parquet", index=False)
        print(f"wrote 06_haircut_{name}: {df.shape}")
        if args.print:
            print(df.to_string(index=False))
    print(f"report: {stage_report(rd, cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
