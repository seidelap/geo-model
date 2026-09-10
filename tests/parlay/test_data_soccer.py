from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.data_soccer import SoccerConfig, fair_1x2, load_soccer


def test_fair_1x2_sums_to_one() -> None:
    ph, pdr, pa = fair_1x2(np.array([2.0, 1.5]), np.array([3.4, 4.0]), np.array([3.6, 6.5]))
    assert np.allclose(ph + pdr + pa, 1.0)
    assert ph[1] > ph[0]


def test_load_soccer_schema_and_new_team_flag(tmp_path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    teams = [f"T{i}" for i in range(6)]
    for season in (2018, 2019):
        pool = teams if season == 2018 else teams[:5] + ["Promoted"]
        for wk in range(10):
            order = rng.permutation(pool)
            for k in range(0, 6, 2):
                rows.append(
                    {
                        "Division": "E0", "MatchDate": f"{season}-08-{10 + wk:02d}", "HomeTeam": order[k], "AwayTeam": order[k + 1],
                        "FTHome": int(rng.integers(0, 4)), "FTAway": int(rng.integers(0, 4)),
                        "OddHome": 2.2, "OddDraw": 3.3, "OddAway": 3.4, "Over25": 1.9, "Under25": 1.95, "HandiSize": -0.25,
                    }
                )
    csv = tmp_path / "soccer" / "xgabora_matches_2000_2026.csv"
    csv.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = load_soccer(SoccerConfig(data_dir=tmp_path / "soccer", divisions=("E0",)))
    assert len(df) == len(rows)
    assert set(df["season"]) == {2018, 2019}
    assert df["gameday"].dt.tz is not None
    assert (df["resid"] == df["result"] - df["spread_line"]).all()
    assert df["total_line"].notna().all()
    assert df["p_home"].between(0, 1).all() and df["p_draw"].between(0, 1).all()
    assert not df.loc[df["season"] == 2018, "home_new"].any()  # first season: no flags
    promoted = df[(df["season"] == 2019) & ((df["home_team"] == "Promoted") | (df["away_team"] == "Promoted"))]
    assert (promoted["home_new"] | promoted["away_new"]).all()
    others = df[(df["season"] == 2019) & (df["home_team"] != "Promoted") & (df["away_team"] != "Promoted")]
    assert not (others["home_new"] | others["away_new"]).any()
    assert df["week"].min() == 1
