"""Tests for the NFL 06 participation-chain helpers (synthetic frames, no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl import imputation_chain as ch
from research.privileged_tracking.nfl import imputation_features as imf


def _imputed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2017_01_A_B", "2017_01_A_B", "2017_02_C_D", "2017_02_C_D"],
            "play_id": [10, 20, 10, 30],
            "off_grp": ["1 RB, 1 TE, 3 WR", "1 RB, 2 TE, 2 WR", None, "1 RB, 1 TE, 3 WR"],
            "p_off_1RB_1TE_3WR": [0.7, 0.4, 0.6, 0.9],
            "p_off_1RB_2TE_2WR": [0.3, 0.6, 0.4, 0.1],
        }
    )


def test_attach_participation_aligns_on_keys_and_leaves_nan_for_missing_plays() -> None:
    game = pd.Series(["2017_01_A_B", "2017_02_C_D", "2017_02_C_D", "2017_03_E_F"])
    play = pd.Series([20.0, 30.0, 99.0, 10.0])
    out = ch.attach_participation(game, play, _imputed())
    assert len(out) == 4
    assert out.loc[0, "off_grp"] == "1 RB, 2 TE, 2 WR" and out.loc[0, "p_off_1RB_2TE_2WR"] == 0.6
    assert out.loc[1, "p_off_1RB_1TE_3WR"] == 0.9
    assert np.isnan(out.loc[2, "p_off_1RB_1TE_3WR"]) and np.isnan(out.loc[3, "p_off_1RB_2TE_2WR"])
    assert list(out.columns) == ["off_grp", "p_off_1RB_1TE_3WR", "p_off_1RB_2TE_2WR"]


def test_chain_feature_sets_nest_the_nfl03_sets() -> None:
    p_cols = ["p_off_a", "p_off_b"]
    sets = ch.chain_feature_sets(p_cols)
    assert list(sets) == list(ch.FEATURE_SET_ORDER)
    assert sets["F0"] == list(imf.FEATURE_SETS["F0"])
    assert sets["F0P"] == list(imf.FEATURE_SETS["F0P"])
    assert sets["F0I"] == sets["F0"] + p_cols and sets["F0PI"] == sets["F0P"] + p_cols
    assert not any(c in imf.OFFICIAL_COLS or c in imf.POSTPLAY_COLS for c in sets["F0PI"])
    assert set(sets["F0I"]) & set(imf.PERSONNEL_COLS) == set()


def test_grouping_agreement_accuracy_and_coverage() -> None:
    imp = _imputed()
    part = ch.attach_participation(imp["game_id"], imp["play_id"], imp)
    probs = part[[c for c in part.columns if c.startswith("p_off_")]]
    out = ch.grouping_agreement(part["off_grp"], probs, pd.Series(["a", None, "b", "c"]))
    assert out["n_plays"] == 4 and out["share_with_probabilities"] == 1.0
    assert out["share_with_label"] == 0.75
    # argmax: row0 -> 1RB_1TE_3WR (correct), row1 -> 1RB_2TE_2WR (correct), row3 -> correct
    assert out["argmax_accuracy"] == 1.0
    assert out["majority_share"] == 2 / 3
    assert out["share_charted_personnel_present"] == 0.75
