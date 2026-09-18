"""Unit tests for the haircut reconciliation (pure helpers on synthetic frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.scenario_ev import haircut as H


def _odds() -> pd.DataFrame:
    """Six matches with a deliberate nesting: fouls subset of corners, cards disjoint-ish.

    Rows 0-3 have Over/Under prices; all six have a result. Corners are recorded on 0-3,
    fouls on 0-2 only, so the foul pool is a strict subset of the corner pool, and the
    card pool (which does not condition on any side market) contains both.
    """
    return pd.DataFrame({
        "MatchDate": pd.to_datetime(["2016-01-01", "2018-01-01", "2020-01-01",
                                     "2021-01-01", "2021-02-01", "2021-03-01"]),
        "Over25": [2.0, 2.0, 2.0, 2.0, np.nan, np.nan],
        "Under25": [2.0, 2.0, 1.9, 1.9, np.nan, np.nan],
        "HomeCorners": [5, 5, 5, 5, np.nan, np.nan],
        "AwayCorners": [4, 4, 4, 4, np.nan, np.nan],
        "HomeFouls": [11, 11, 11, np.nan, np.nan, np.nan],
        "AwayFouls": [12, 12, 12, np.nan, np.nan, np.nan],
        "FTHome": [1, 1, 1, 1, 1, 1],
        "FTAway": [0, 0, 0, 0, 0, 0],
    })


def test_universe_masks_apply_the_price_filter() -> None:
    m = H.universe_masks(_odds())
    assert m["02 corners goals check"].sum() == 4
    assert m["04 fouls goals check"].sum() == 3
    # the card stage's goals check conditions on price and result only, not on cards
    assert m["01 cards goals check"].sum() == 4


def test_universe_overlap_detects_the_subset_relation() -> None:
    out = H.universe_overlap(_odds())
    row = out[(out.universe_a == "04 fouls goals check")
              & (out.universe_b == "02 corners goals check")].iloc[0]
    assert bool(row["a_subset_of_b"]) is True
    assert int(row["n_intersection"]) == 3
    assert np.isclose(float(row["jaccard"]), 3 / 4)
    rev = out[(out.universe_a == "02 corners goals check")
              & (out.universe_b == "04 fouls goals check")].iloc[0]
    assert bool(rev["a_subset_of_b"]) is False


def test_realised_holds_are_positive_and_era_filtered() -> None:
    out = H.realised_holds(_odds())
    assert set(out["era"]) <= {"all years", "cards confirmation (from 2017-12)",
                               "corner/foul confirmation (from 2019-09)"}
    assert (out["mean_overround"] > 0).all()
    corners_all = out[(out.universe == "02 corners goals check")
                      & (out.era == "all years")].iloc[0]
    # 1/2.0 + 1/2.0 - 1 = 0 on rows 0-1 and 1/2.0 + 1/1.9 - 1 = 0.0263 on rows 2-3
    assert np.isclose(float(corners_all["mean_overround"]), (0.0 + 0.0 + 2 * (0.5 + 1 / 1.9 - 1)) / 4)
    corners_recent = out[(out.universe == "02 corners goals check")
                         & (out.era == "corner/foul confirmation (from 2019-09)")].iloc[0]
    assert int(corners_recent["n"]) == 2


def _defs() -> pd.DataFrame:
    """Two definitions at one hold on two universes."""
    rows = []
    for uni, lenient, strict in (("01 cards goals check", 0.10, 0.30),
                                 ("04 fouls goals check", 0.12, 0.20)):
        rows.append({"definition": "book_earns_one_way", "stage": uni[:2],
                     "universe": uni, "split": "confirmation", "hold": 0.06,
                     "haircut_roi_points": lenient, "leg_vs_proxy": lenient,
                     "leg_vs_book": np.nan, "n_bets": 10, "source_table": "t"})
        rows.append({"definition": "margin_matched_round_trip", "stage": uni[:2],
                     "universe": uni, "split": "confirmation", "hold": 0.06,
                     "haircut_roi_points": strict, "leg_vs_proxy": lenient,
                     "leg_vs_book": lenient - strict, "n_bets": 10, "source_table": "t"})
        rows.append({"definition": "actual_prices_round_trip", "stage": uni[:2],
                     "universe": uni, "split": "confirmation", "hold": 0.06,
                     "haircut_roi_points": strict + 0.05, "leg_vs_proxy": lenient,
                     "leg_vs_book": np.nan, "n_bets": 10, "source_table": "t"})
    return pd.DataFrame(rows)


def _bets() -> pd.DataFrame:
    """One card row and one corner row, both with the same raw ROI."""
    return pd.DataFrame({
        "candidate": ["cards", "corners"],
        "market": ["match_cards_proxy_line", "team_total"],
        "scenario": ["all", "fav_strong"],
        "source": ["count_nb", "count_nb"],
        "hold": [0.06, 0.06], "threshold": [0.4, 0.02],
        "n_rows": [1000, 1000], "n_bets": [500, 900], "bet_rate": [0.5, 0.9],
        "mean_edge": [0.4, 0.2], "roi": [0.25, 0.25],
        "roi_lo": [0.20, 0.22], "roi_hi": [0.30, 0.28],
    })


def test_apply_haircuts_uses_each_candidates_own_universe() -> None:
    out = H.apply_haircuts(_bets(), _defs())
    card = out[out.candidate == "cards"].iloc[0]
    corner = out[out.candidate == "corners"].iloc[0]
    # cards are charged the card universe's 0.30; corners the foul universe's 0.20
    assert np.isclose(float(card["haircut_margin_matched_round_trip"]), 0.30)
    assert np.isclose(float(corner["haircut_margin_matched_round_trip"]), 0.20)
    assert np.isclose(float(card["after_margin_matched_round_trip"]), -0.05)
    assert np.isclose(float(corner["after_margin_matched_round_trip"]), 0.05)


def test_apply_haircuts_lower_bound_and_survival_flags() -> None:
    out = H.apply_haircuts(_bets(), _defs())
    corner = out[out.candidate == "corners"].iloc[0]
    assert np.isclose(float(corner["after_margin_matched_round_trip_lo"]), 0.02)
    assert bool(corner["survives_reference"]) is True
    assert bool(corner["survives_reference_lo"]) is True
    card = out[out.candidate == "cards"].iloc[0]
    assert bool(card["survives_reference"]) is False


def test_lenient_definition_promotes_rows_the_reference_kills() -> None:
    """The whole reason the module exists: the choice of definition flips a verdict."""
    out = H.apply_haircuts(_bets(), _defs())
    card = out[out.candidate == "cards"].iloc[0]
    assert float(card["after_book_earns_one_way"]) > 0
    assert float(card["after_margin_matched_round_trip"]) < 0


def test_reference_haircuts_selects_the_reference_definition_only() -> None:
    out = H.reference_haircuts(_defs())
    assert len(out) == 2
    assert set(np.round(out["haircut_roi_points"], 6)) == {0.30, 0.20}


def test_definition_notes_cover_every_applied_column() -> None:
    out = H.apply_haircuts(_bets(), _defs())
    for name in H.DEFINITION_NOTES:
        assert f"after_{name}" in out.columns
        assert f"haircut_{name}" in out.columns
    assert H.CFG.reference in H.DEFINITION_NOTES


def test_confirmation_overlap_reflects_the_different_cuts() -> None:
    """Shared pools, different cuts: the confirmation halves need not overlap much."""
    out = H.confirmation_overlap(_odds())
    row = out[(out.universe_a == "01 cards goals check")
              & (out.universe_b == "02 corners goals check")].iloc[0]
    # cards cut at 2017-11-30 keeps rows 1-3; corners cut at 2019-08-31 keeps rows 2-3
    assert int(row["n_conf_a"]) == 3
    assert int(row["n_conf_b"]) == 2
    assert int(row["n_intersection"]) == 2
    assert np.isclose(float(row["share_of_a"]), round(2 / 3, 4))
