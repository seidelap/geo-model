from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_model.parlay.config import ParlayConfig
from geo_model.parlay.data import clean_games, to_team_games
from geo_model.parlay.pairs import build_shared_game_pairs
from geo_model.parlay import sources as S


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def test_catalog_keys_unique_and_urls_allowed() -> None:
    keys = [s.key for s in S.SOURCES]
    assert len(keys) == len(set(keys))
    for spec in S.SOURCES:
        assert spec.sport in {"mlb", "nba", "nhl", "nfl", "ncaab", "ncaaf", "soccer", "tennis"}
        assert spec.quality in {"excellent", "usable", "weak"}
        for f in spec.files:
            assert f.url.startswith(S.ALLOWED_URL_PREFIXES)
            assert not f.filename.startswith("/")
        assert spec.repo_url.startswith("https://github.com/")
    assert S.source_by_key("mlb_retrosheet_gamelogs").starting_pitchers
    with pytest.raises(KeyError):
        S.source_by_key("nope")


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, timeout: float, stream: bool) -> _FakeResp:
        self.calls.append(url)
        return _FakeResp(b"payload")


def test_download_source_skips_existing_files(tmp_path: Path) -> None:
    cfg = ParlayConfig(data_dir=tmp_path)
    spec = S.SourceSpec(
        key="t", sport="nba", description="", seasons="", closing_lines=True, starting_pitchers=False,
        license="", quality="usable",
        files=(S.SourceFile(S.RAW_GH + "/o/r/main/a.csv", "a.csv"), S.SourceFile(S.RAW_GH + "/o/r/main/b.csv", "sub/b.csv")),
    )
    sess = _FakeSession()
    paths = S.download_source(spec, cfg, session=sess)
    assert [p.name for p in paths] == ["a.csv", "b.csv"]
    assert all(p.read_bytes() == b"payload" for p in paths)
    assert len(sess.calls) == 2
    S.download_source(spec, cfg, session=sess)
    assert len(sess.calls) == 2  # cached
    S.download_source(spec, cfg, session=sess, refresh=True)
    assert len(sess.calls) == 4
    inv = S.inventory(cfg)
    assert set(inv.columns) >= {"key", "sport", "n_files", "n_present", "bytes"}
    assert (inv["n_present"] == 0).all()  # catalog sources are not in tmp_path


def test_download_source_rejects_foreign_hosts(tmp_path: Path) -> None:
    spec = S.SourceSpec(
        key="t", sport="nba", description="", seasons="", closing_lines=True, starting_pitchers=False,
        license="", quality="usable", files=(S.SourceFile("https://example.com/a.csv", "a.csv"),),
    )
    with pytest.raises(ValueError):
        S.download_source(spec, ParlayConfig(data_dir=tmp_path), session=_FakeSession())


# --------------------------------------------------------------------------- #
# Odds helpers
# --------------------------------------------------------------------------- #


def test_parse_line_and_moneyline_cells() -> None:
    assert S.parse_line_value("pk") == 0.0
    assert S.parse_line_value("PK") == 0.0
    assert np.isnan(S.parse_line_value("NL"))
    assert S.parse_line_value("237½") == 237.5
    assert S.parse_line_value(" 3.5 ") == 3.5
    assert S.parse_line_value(7) == 7.0
    assert np.isnan(S.parse_line_value(None))
    assert np.isnan(S.parse_line_value("garbage"))
    assert S.parse_moneyline("+145\xa0") == 145.0
    assert S.parse_moneyline("ev") == 100.0
    assert np.isnan(S.parse_moneyline(0))
    assert np.isnan(S.parse_moneyline("NL"))


def test_hold_and_no_vig_probability() -> None:
    hold = S.american_hold(np.array([-110.0]), np.array([-110.0]))
    assert hold[0] == pytest.approx(0.0476, abs=1e-3)
    p = S.no_vig_home_prob(np.array([-150.0, 100.0]), np.array([130.0, -120.0]))
    assert p[0] > 0.5 > p[1]
    assert np.isnan(S.no_vig_home_prob(np.array([np.nan]), np.array([100.0]))[0])


# --------------------------------------------------------------------------- #
# SBR workbooks
# --------------------------------------------------------------------------- #


def _sbr_rows() -> pd.DataFrame:
    # Game 1: home favourite by 4.5, total 200 -> 205 (visitor row holds totals).
    # Game 2: visitor favourite by 'pk'... open pk, close 2 ; home row holds totals.
    # Game 3: neutral site, total only offered at close ('NL' on the spread side).
    return pd.DataFrame(
        [
            {"Date": 1030, "VH": "V", "Team": "A", "Final": 97, "Open": 200, "Close": 205, "ML": 170},
            {"Date": 1030, "VH": "H", "Team": "B", "Final": 106, "Open": 4.5, "Close": "4½", "ML": -190},
            {"Date": 1231, "VH": "V", "Team": "C", "Final": 100, "Open": "pk", "Close": 2, "ML": -110},
            {"Date": 1231, "VH": "H", "Team": "D", "Final": 99, "Open": 210, "Close": 212.5, "ML": "-110"},
            {"Date": 105, "VH": "N", "Team": "E", "Final": 80, "Open": "NL", "Close": "NL", "ML": "NL"},
            {"Date": 105, "VH": "N", "Team": "F", "Final": 90, "Open": "NL", "Close": 190, "ML": "NL"},
        ]
    )


def test_sbr_rows_to_games_conventions() -> None:
    games = S.sbr_rows_to_games(_sbr_rows(), "nba", 2015)
    assert list(games.columns[: len(S.GAME_COLUMNS)]) == S.GAME_COLUMNS
    assert len(games) == 3
    g1 = games[games["home"] == "B"].iloc[0]
    assert g1["spread_open"] == 4.5 and g1["spread_close"] == 4.5
    assert g1["total_open"] == 200 and g1["total_close"] == 205
    assert g1["home_ml_close"] == -190 and g1["away_ml_close"] == 170
    assert g1["home_score"] == 106 and g1["away_score"] == 97
    assert g1["date"] == pd.Timestamp("2015-10-30", tz="UTC")
    g2 = games[games["home"] == "D"].iloc[0]
    assert g2["spread_open"] == 0.0 and g2["spread_close"] == -2.0
    assert g2["total_close"] == 212.5
    assert g2["date"] == pd.Timestamp("2015-12-31", tz="UTC")
    g3 = games[games["home"] == "F"].iloc[0]
    assert bool(g3["neutral"]) is True
    assert np.isnan(g3["spread_close"]) and g3["total_close"] == 190
    assert np.isnan(g3["home_ml_close"])
    assert g3["date"] == pd.Timestamp("2016-01-05", tz="UTC")  # rolled into the next year
    assert (games["season"] == 2015).all()


def test_sbr_season_years_handles_bubble_season() -> None:
    # Oct 2019 ... Mar 2020, then the bubble in Jul-Oct 2020 must stay in 2020.
    dates = pd.Series([1022, 1201, 115, 311, 730, 815, 1011])
    years = S.sbr_season_years(dates, 2019)
    assert years.tolist() == [2019, 2019, 2020, 2020, 2020, 2020, 2020]


def test_sbr_rows_drops_misaligned_pairs_and_odd_tail() -> None:
    rows = _sbr_rows().iloc[[0, 1, 1, 0, 2]]  # H then V is invalid; trailing single row dropped
    games = S.sbr_rows_to_games(rows, "nba", 2015)
    assert len(games) == 1


def test_sbr_start_year_from_filename() -> None:
    assert S.sbr_start_year(Path("x/nba_odds_2015-16.xlsx")) == 2015
    assert S.sbr_start_year(Path("ncaab_odds_2007-08.xlsx")) == 2007


# --------------------------------------------------------------------------- #
# FinnedAI archives
# --------------------------------------------------------------------------- #


def test_finned_to_games_signs_and_cleanup() -> None:
    rec = pd.DataFrame(
        [
            {"season": 2015, "date": 20151027.0, "home_team": "Golden State", "away_team": "NewJersey",
             "home_final": 111, "away_final": 95, "home_close_ml": -300, "away_close_ml": 250,
             "home_open_spread": -6.5, "away_open_spread": 6.5, "home_close_spread": -7.0, "away_close_spread": 7.0,
             "open_over_under": 210.5, "close_over_under": 212.0},
            {"season": 2015, "date": 20151027.0, "home_team": "0", "away_team": "Heat",
             "home_final": 0, "away_final": 1, "home_close_ml": 0, "away_close_ml": 150,
             "home_open_spread": 0, "away_open_spread": 0, "home_close_spread": 0, "away_close_spread": 0,
             "open_over_under": 0, "close_over_under": 0},
            {"season": 2015, "date": 20151028.0, "home_team": "Heat", "away_team": "Bulls",
             "home_final": 100, "away_final": 90, "home_close_ml": -120, "away_close_ml": 100,
             "home_open_spread": -1.0, "away_open_spread": 1.0, "home_close_spread": -242.5, "away_close_spread": 242.5,
             "open_over_under": 199.0, "close_over_under": 1.0},
        ]
    )
    games = S.finned_to_games(rec, "nba")
    assert len(games) == 2  # junk '0' row dropped
    g = games[games["home"] == "Warriors"].iloc[0]  # alias applied
    assert g["away"] == "Nets"
    assert g["spread_close"] == 7.0 and g["spread_open"] == 6.5  # home favoured -> positive expected margin
    assert g["total_close"] == 212.0
    assert g["date"] == pd.Timestamp("2015-10-27", tz="UTC")
    h = games[games["home"] == "Heat"].iloc[0]
    assert np.isnan(h["spread_close"]) and np.isnan(h["total_close"])  # implausible values blanked
    assert h["spread_open"] == 1.0 and h["total_open"] == 199.0


def test_finned_mlb_run_line_zero_means_missing() -> None:
    rec = pd.DataFrame(
        [
            {"season": 2019, "date": 20190401.0, "home_team": "LOS", "away_team": "SFO", "home_final": 5, "away_final": 3,
             "home_open_ml": -180, "away_open_ml": 160, "home_close_ml": -175, "away_close_ml": 155,
             "home_close_spread": -1.5, "away_close_spread": 1.5, "home_close_spread_odds": 120, "away_close_spread_odds": -140,
             "open_over_under": 8.0, "open_over_under_odds": -110, "close_over_under": 7.5, "close_over_under_odds": -105},
            {"season": 2019, "date": 20190402.0, "home_team": "LOS", "away_team": "SFO", "home_final": 2, "away_final": 3,
             "home_open_ml": -170, "away_open_ml": 150, "home_close_ml": -160, "away_close_ml": 140,
             "home_close_spread": 0.0, "away_close_spread": 0.0, "home_close_spread_odds": 0, "away_close_spread_odds": 0,
             "open_over_under": 8.0, "open_over_under_odds": -110, "close_over_under": 8.0, "close_over_under_odds": -110},
        ]
    )
    games = S.finned_to_games(rec, "mlb")
    assert games["spread_close"].tolist()[0] == 1.5
    assert np.isnan(games["spread_close"].tolist()[1])
    assert np.isnan(games["home_close_spread_odds"].tolist()[1])
    assert games["home_ml_open"].tolist() == [-180.0, -170.0]


# --------------------------------------------------------------------------- #
# Per-book JSON
# --------------------------------------------------------------------------- #


def _book_entry(book: str, h: float, a: float, hs: float = -1.5) -> dict:
    return {
        "sportsbook": book,
        "openingLine": {"homeOdds": h + 10, "awayOdds": a - 10, "homeSpread": hs, "awaySpread": -hs},
        "currentLine": {"homeOdds": h, "awayOdds": a, "homeSpread": hs, "awaySpread": -hs},
    }


def _book_game(home: str, away: str, hs: int, as_: int, gtype: str = "R", books: tuple[str, ...] = ("fanduel", "bet365"), status: str = "Final") -> dict:
    return {
        "gameView": {
            "startDate": "2021-04-01T17:05:00+00:00",
            "homeTeam": {"shortName": home}, "awayTeam": {"shortName": away},
            "homeTeamScore": hs, "awayTeamScore": as_, "gameStatusText": status, "gameType": gtype,
        },
        "odds": {
            "moneyline": [_book_entry(b, -150, 130) for b in books],
            "pointspread": [_book_entry(b, 110, -130) for b in books],
            "totals": [{"sportsbook": b, "openingLine": {"overOdds": -110, "underOdds": -110, "total": 8.0},
                        "currentLine": {"overOdds": -115, "underOdds": -105, "total": 7.5}} for b in books],
        },
    }


def test_sbr_book_records_to_games_prefers_book_and_filters() -> None:
    data = {
        "2021-04-01": [
            _book_game("NYY", "TOR", 2, 3),
            _book_game("AL", "NL", 5, 4),
            _book_game("SD", "SF", 1, 0, gtype="S"),
            _book_game("BOS", "BAL", 0, 0, status="Postponed"),
        ],
        "2021-04-02": [_book_game("LAD", "COL", 4, 1, books=("draftkings",))],
    }
    games = S.sbr_book_records_to_games(data)
    assert len(games) == 2
    g = games[games["home"] == "NYY"].iloc[0]
    assert g["book"] == "bet365" and g["n_books"] == 2
    assert g["home_ml_close"] == -150 and g["home_ml_open"] == -140
    assert g["spread_close"] == 1.5 and g["total_close"] == 7.5 and g["total_open"] == 8.0
    assert g["season"] == 2021 and g["date"] == pd.Timestamp("2021-04-01", tz="UTC")
    assert games[games["home"] == "LAD"].iloc[0]["book"] == "draftkings"
    everything = S.sbr_book_records_to_games(data, game_types=None)
    assert len(everything) == 3  # spring-training game kept, All-Star and postponed still dropped


# --------------------------------------------------------------------------- #
# Retrosheet + pitcher join
# --------------------------------------------------------------------------- #


def _gamelog_row(date: int, num: int, away: str, home: str, as_: int, hs: int, asp: str, hsp: str) -> list:
    row = [""] * 161
    row[0], row[1], row[3], row[6], row[9], row[10] = date, num, away, home, as_, hs
    row[101], row[102], row[103], row[104] = asp, asp.upper(), hsp, hsp.upper()
    return row


def test_retrosheet_gamelog_and_pitcher_join_with_doubleheader() -> None:
    raw = pd.DataFrame(
        [
            _gamelog_row(20190401, 0, "SFN", "LAN", 3, 5, "bumgm001", "kersc001"),
            _gamelog_row(20190402, 1, "SFN", "LAN", 2, 1, "samaj001", "buehw001"),
            _gamelog_row(20190402, 2, "SFN", "LAN", 4, 6, "holld001", "ryu-h001"),
            _gamelog_row(20190403, 1, "NYA", "BOS", 2, 2, "colej001", "salec001"),
            _gamelog_row(20190403, 2, "NYA", "BOS", 2, 2, "tanam001", "pricd001"),
            _gamelog_row(20110405, 0, "NYN", "FLO", 1, 0, "dickr001", "johnj009"),
        ]
    )
    gl = S.retrosheet_gamelog_to_games(raw)
    assert list(gl["game_number"]) == [0, 1, 2, 1, 2, 0]
    assert gl["season"].tolist()[-1] == 2011
    odds = pd.DataFrame(
        {
            "season": [2019, 2019, 2019, 2019, 2011],
            "date": pd.to_datetime(["2019-04-01", "2019-04-02", "2019-04-02", "2019-04-03", "2011-04-05"], utc=True),
            "home": ["LOS", "Dodgers", "LOS", "Red Sox", "Marlins"],
            "away": ["SFO", "SFO", "SFO", "Yankees", "Mets"],
            "home_score": [5, 1, 6, 2, 0],
            "away_score": [3, 2, 4, 2, 1],
        }
    )
    joined = S.attach_starting_pitchers(odds, gl)
    assert joined["home_code"].tolist() == ["LAN", "LAN", "LAN", "BOS", "FLO"]
    assert joined["pitcher_match"].tolist() == [True, True, True, False, True]
    assert joined["home_pitcher_id"].tolist()[:3] == ["kersc001", "buehw001", "ryu-h001"]
    assert joined["away_pitcher"].tolist()[0] == "BUMGM001"
    assert joined["retro_game_number"].tolist()[1:3] == [1, 2]
    assert joined["home_pitcher_id"].isna().tolist()[3]  # identical scores -> ambiguous -> no pitcher


def test_mlb_retrosheet_code_season_rules() -> None:
    assert S.mlb_retrosheet_code("Marlins", 2011) == "FLO"
    assert S.mlb_retrosheet_code("MIA", 2012) == "MIA"
    assert S.mlb_retrosheet_code("Athletics", 2024) == "OAK"
    assert S.mlb_retrosheet_code("ATH", 2025) == "ATH"
    assert S.mlb_retrosheet_code("LOS", 2015) == "LAN"
    assert S.mlb_retrosheet_code("AL", 2021) is None


def test_normalize_team_aliases() -> None:
    assert S.normalize_team("Phoenix", "nhl") == "Coyotes"
    assert S.normalize_team("St.Louis", "nhl") == "Blues"
    assert S.normalize_team("Seventysixers", "nba") == "76ers"
    assert S.normalize_team("0", "nba") is None
    assert S.normalize_team(float("nan"), "nhl") is None
    assert S.normalize_team("Cubs", "mlb") == "Cubs"


# --------------------------------------------------------------------------- #
# Tennis
# --------------------------------------------------------------------------- #


def test_tennis_rows_to_matches_parses_dates_and_favourite() -> None:
    df = pd.DataFrame(
        {
            "ATP": [1, 1], "Location": ["Doha", "Doha"], "Tournament": ["Qatar Open"] * 2,
            "Date": ["05/01/2014", "06/01/2014"], "Series": ["ATP250"] * 2, "Court": ["Outdoor"] * 2,
            "Surface": ["Hard"] * 2, "Round": ["1st Round"] * 2, "Best of": [3, 3],
            "Winner": ["A", "B"], "Loser": ["C", "D"], "WRank": [10, "NR"], "LRank": [50, 40],
            "B365W": [1.5, 2.5], "B365L": [2.5, 1.5], "PSW": [1.55, np.nan], "PSL": [2.6, np.nan],
        }
    )
    m = S.tennis_rows_to_matches(df)
    assert m["Date"].tolist()[0] == pd.Timestamp("2014-01-05", tz="UTC")
    assert m["year"].tolist() == [2014, 2014]
    assert m["winner_favourite"].tolist() == [True, False]
    assert np.isnan(m["WRank"].tolist()[1])


# --------------------------------------------------------------------------- #
# Bridge to the NFL pipeline
# --------------------------------------------------------------------------- #


def _games_frame() -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    day = pd.Timestamp("2015-10-01", tz="UTC")
    rng = np.random.default_rng(1)
    for k in range(8):
        h, a = teams[k % 4], teams[(k + 1) % 4]
        rows.append(
            {
                "season": 2015, "date": day + pd.Timedelta(days=k), "home": h, "away": a, "neutral": False,
                "home_score": 100 + int(rng.integers(0, 10)), "away_score": 100 + int(rng.integers(0, 10)),
                "home_ml_close": -130.0, "away_ml_close": 110.0, "spread_close": 2.5, "total_close": 205.0,
                "home_pitcher_id": f"p{h}", "away_pitcher_id": f"p{a}",
            }
        )
    g = pd.DataFrame(rows)
    return S._finish(g, "nba", "test")


def test_to_nflverse_layout_spread_and_moneyline_round_trip() -> None:
    g = _games_frame()
    lay = S.to_nflverse_layout(g, "spread")
    assert set(["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team", "result", "total", "spread_line", "total_line", "home_qb_id", "away_qb_id"]) <= set(lay.columns)
    assert lay["game_id"].is_unique
    assert (lay["result"] == lay["home_score"] - lay["away_score"]).all()
    assert lay["week"].tolist() == [1, 1, 1, 1, 1, 1, 1, 2]
    cleaned = clean_games(lay, ParlayConfig(seasons=(2015, 2015)))
    tg = to_team_games(cleaned)
    pairs = build_shared_game_pairs(tg)
    assert len(cleaned) == 8 and len(tg) == 16 and len(pairs) >= 1
    ml = S.to_nflverse_layout(g, "moneyline")
    p_home = S.no_vig_home_prob(np.array([-130.0]), np.array([110.0]))[0]
    home_win = (g["home_score"] > g["away_score"]).astype(float)
    ties = g["home_score"] == g["away_score"]
    assert (ml["spread_line"] == 0.0).all()
    assert np.allclose(ml.loc[~ties.to_numpy(), "result"], (home_win - p_home)[~ties])
    with pytest.raises(ValueError):
        S.to_nflverse_layout(g, "puckline")


def test_to_nflverse_layout_suffixes_duplicate_game_ids() -> None:
    g = _games_frame().iloc[[0, 0]].copy()
    lay = S.to_nflverse_layout(g, "spread")
    assert lay["game_id"].tolist()[1].endswith("_g2")


# --------------------------------------------------------------------------- #
# Scrambled FinnedAI MLB archive
# --------------------------------------------------------------------------- #


def test_repair_finned_mlb_reassembles_games_from_scrambled_rows() -> None:
    # Two real games on 2015-06-10: SEA @ CLE (9-3) and ANA @ TBA (2-4), plus a
    # doubleheader NYA @ BOS (1: 2-5, 2: 6-1). The archive pairs teams from
    # different games and lists the true away team in the 'home' column.
    raw_logs = pd.DataFrame(
        [
            _gamelog_row(20150610, 0, "SEA", "CLE", 9, 3, "walkt002", "bauet001"),
            _gamelog_row(20150610, 0, "ANA", "TBA", 2, 4, "weavj003", "ramie002"),
            _gamelog_row(20150610, 1, "NYA", "BOS", 2, 5, "tanam001", "porcr001"),
            _gamelog_row(20150610, 2, "NYA", "BOS", 6, 1, "pinem001", "milew001"),
        ]
    )
    gl = S.retrosheet_gamelog_to_games(raw_logs)
    records = pd.DataFrame(
        [
            # 'home' column = Angels (true away @TBA), 'away' column = Indians (true home vs SEA)
            {"season": 2015, "date": 20150610.0, "home_team": "Angels", "away_team": "Indians",
             "home_final": 2, "away_final": 3, "home_open_ml": 118, "away_open_ml": -130, "home_close_ml": 113, "away_close_ml": -124,
             "home_close_spread": 1.5, "away_close_spread": -1.5, "home_close_spread_odds": -180, "away_close_spread_odds": 140,
             "open_over_under": 7.0, "close_over_under": 7.5},
            # 'home' column = Mariners (true away @CLE), 'away' column = TAM (true home vs ANA)
            {"season": 2015, "date": 20150610.0, "home_team": "Mariners", "away_team": "TAM",
             "home_final": 9, "away_final": 4, "home_open_ml": 110, "away_open_ml": -128, "home_close_ml": 114, "away_close_ml": -123,
             "home_close_spread": 1.5, "away_close_spread": -1.5, "home_close_spread_odds": -150, "away_close_spread_odds": 130,
             "open_over_under": 8.0, "close_over_under": 8.5},
            # Doubleheader game 1 and game 2 records for NYA/BOS, disambiguated by score
            {"season": 2015, "date": 20150610.0, "home_team": "Yankees", "away_team": "Red Sox",
             "home_final": 2, "away_final": 5, "home_open_ml": 100, "away_open_ml": -110, "home_close_ml": -105, "away_close_ml": -105,
             "home_close_spread": 0.0, "away_close_spread": 0.0, "home_close_spread_odds": 0, "away_close_spread_odds": 0,
             "open_over_under": 9.0, "close_over_under": 9.0},
            {"season": 2015, "date": 20150610.0, "home_team": "Yankees", "away_team": "Red Sox",
             "home_final": 6, "away_final": 1, "home_open_ml": -120, "away_open_ml": 110, "home_close_ml": -125, "away_close_ml": 115,
             "home_close_spread": -1.5, "away_close_spread": 1.5, "home_close_spread_odds": 130, "away_close_spread_odds": -150,
             "open_over_under": 10.0, "close_over_under": 10.5},
            # Junk row: unknown team and a missing date
            {"season": 2015, "date": np.nan, "home_team": "0", "away_team": "Reds", "home_final": 0, "away_final": 0,
             "home_open_ml": 0, "away_open_ml": 0, "home_close_ml": 0, "away_close_ml": 0, "home_close_spread": 0,
             "away_close_spread": 0, "home_close_spread_odds": 0, "away_close_spread_odds": 0, "open_over_under": 0, "close_over_under": 0},
        ]
    )
    games = S.repair_finned_mlb(records, gl).set_index(["home", "away", "retro_game_number"])
    assert len(games) == 4
    cle = games.loc[("CLE", "SEA", 0)]
    assert cle["home_ml_close"] == -124 and cle["away_ml_close"] == 114
    assert cle["home_ml_open"] == -130 and cle["away_ml_open"] == 110
    assert cle["home_score"] == 3 and cle["away_score"] == 9
    assert cle["home_pitcher_id"] == "bauet001" and cle["away_pitcher_id"] == "walkt002"
    assert cle["total_close"] == 8.5  # from the row holding the true away team (Mariners)
    assert cle["total_close_home_row"] == 7.5
    assert cle["spread_close"] == 1.5  # CLE -1.5 run line -> home favoured by 1.5
    tba = games.loc[("TBA", "ANA", 0)]
    assert tba["home_ml_close"] == -123 and tba["away_ml_close"] == 113 and tba["total_close"] == 7.5
    g1 = games.loc[("BOS", "NYA", 1)]
    g2 = games.loc[("BOS", "NYA", 2)]
    assert g1["home_ml_close"] == -105 and np.isnan(g1["spread_close"])
    assert g2["home_ml_close"] == 115 and g2["away_ml_close"] == -125 and g2["spread_close"] == -1.5
    assert g2["home_pitcher_id"] == "milew001"
    assert (games["source"] == "sbr_finned_repaired").all()
