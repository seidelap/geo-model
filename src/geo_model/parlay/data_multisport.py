"""Loaders for NBA, NHL and MLB results with closing lines.

Sources (all public GitHub files, the only hosts reachable from this
environment):

* NBA: ``Data/OddsData.sqlite`` from ``kyleskom/NBA-Machine-Learning-Sports-Betting``
  (SportsBookReview closing spread, total and moneylines, 2007-08 onward).
* NHL / MLB / NBA 2011-2021: ``data/<sport>_archive_10Y.json`` from
  ``flancast90/sportsbookreview-scraper`` (closing moneylines and totals; NHL
  puck line from 2014).

Every loader returns the same cleaned schema as :func:`geo_model.parlay.data.clean_games`
so the shared-game pair construction and the Kalman filter can be reused. For
sports whose primary market is the moneyline, the market-implied home margin
is derived from the vig-free win probability via a probit mapping fitted per
sport (see :func:`implied_margin_scale`), and ``resid`` is realized margin
minus that implied margin (positive = home beat the market).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

from geo_model.parlay.config import ParlayConfig, american_to_decimal

MLB_ODDS_URL = "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/mlb_odds_dataset.json"
MLB_SHORTNAME_TO_CANON = {"AZ": "ARI", "ATH": "OAK"}
MLB_GAME_TYPES = {"R": "REG", "D": "POST", "L": "POST", "F": "POST", "W": "POST"}
NBA_SPREAD_TOLERANCE = 6.0

SBR_ARCHIVE_URL = "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/{sport}_archive_10Y.json"
NBA_SQLITE_URL = "https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/master/Data/OddsData.sqlite"

CLEAN_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
    "home_score", "away_score", "result", "total", "spread_line", "total_line",
    "home_moneyline", "away_moneyline", "home_qb_id", "away_qb_id", "resid", "tresid",
    "p_home", "home_decimal", "away_decimal", "spread_source", "over_odds", "under_odds",
]


@dataclass(frozen=True)
class MultiSportConfig:
    """Locations of the cached multi-sport files.

    Attributes:
        data_dir: Directory holding the downloaded files (``<data_dir>/multisport``).
        sbr_url: Template for the 10-year SportsBookReview archives.
        nba_sqlite_url: NBA odds SQLite database URL.
        margin_sd: Per-sport standard deviation used to map a win probability to
            an implied margin. Fitted from data when ``None`` (see
            :func:`implied_margin_scale`).
    """

    data_dir: Path = field(default_factory=lambda: ParlayConfig().data_dir / "multisport")
    sbr_url: str = SBR_ARCHIVE_URL
    nba_sqlite_url: str = NBA_SQLITE_URL
    mlb_odds_url: str = MLB_ODDS_URL
    margin_sd: dict[str, float] = field(default_factory=dict)

    @property
    def pitchers_parquet(self) -> Path:
        """Starting pitchers table produced from Retrosheet event files."""
        return self.data_dir / "mlb_starting_pitchers.parquet"


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def implied_probs(home_ml: np.ndarray, away_ml: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vig-free home win probability and the raw (vig-included) probabilities.

    Args:
        home_ml: American moneyline on the home side.
        away_ml: American moneyline on the away side.

    Returns:
        ``(p_home_fair, p_home_raw)`` where the fair probability normalises the
        two implied probabilities to sum to one (proportional vig removal).
    """
    home_ml = np.asarray(home_ml, dtype=float)
    away_ml = np.asarray(away_ml, dtype=float)
    dh = np.vectorize(american_to_decimal)(home_ml)
    da = np.vectorize(american_to_decimal)(away_ml)
    ph, pa = 1.0 / dh, 1.0 / da
    return ph / (ph + pa), ph


def implied_margin_scale(margin: np.ndarray, p_home: np.ndarray) -> float:
    """Fit the scale ``s`` in ``E[margin] = s * Phi^{-1}(p_home)`` by OLS through the origin.

    Args:
        margin: Realized home minus away score.
        p_home: Fair home win probability.

    Returns:
        Scale in score units.
    """
    z = stats.norm.ppf(np.clip(np.asarray(p_home, dtype=float), 1e-4, 1 - 1e-4))
    m = np.asarray(margin, dtype=float)
    ok = np.isfinite(z) & np.isfinite(m)
    return float(np.sum(z[ok] * m[ok]) / np.sum(z[ok] ** 2))


def _season_week(gameday: pd.Series, season: pd.Series) -> pd.Series:
    start = gameday.groupby(season).transform("min")
    return ((gameday - start).dt.days // 7 + 1).astype(int)


def _finish(df: pd.DataFrame, sport: str, config: MultiSportConfig) -> pd.DataFrame:
    """Common tail of every loader: residuals, ids, ordering."""
    attrs = dict(df.attrs)
    df = df.copy()
    df["gameday"] = pd.to_datetime(df["gameday"], utc=True)
    df = df[df["home_team"] != df["away_team"]]
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df["result"] = df["home_score"] - df["away_score"]
    df["total"] = df["home_score"] + df["away_score"]
    df = df.sort_values(["gameday", "home_team"]).reset_index(drop=True)
    seq = df.groupby(["gameday", "home_team", "away_team"]).cumcount()
    df["game_id"] = (
        df["gameday"].dt.strftime("%Y%m%d") + "_" + df["away_team"].str.replace(" ", "")
        + "_" + df["home_team"].str.replace(" ", "") + np.where(seq > 0, "_" + seq.astype(str), "")
    )
    df["week"] = _season_week(df["gameday"], df["season"])
    if "game_type" not in df:
        df["game_type"] = "REG"
    for c in ("home_qb_id", "away_qb_id"):
        if c not in df:
            df[c] = np.nan
    for c in ("over_odds", "under_odds"):
        if c not in df:
            df[c] = -110.0
    has_ml = df["home_moneyline"].notna() & df["away_moneyline"].notna()
    df["p_home"] = np.nan
    df.loc[has_ml, "p_home"], _ = implied_probs(df.loc[has_ml, "home_moneyline"], df.loc[has_ml, "away_moneyline"])
    df["home_decimal"] = df["home_moneyline"].map(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    df["away_decimal"] = df["away_moneyline"].map(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    if "spread_line" not in df or df["spread_line"].isna().all():
        sd = config.margin_sd.get(sport) or implied_margin_scale(df.loc[has_ml, "result"], df.loc[has_ml, "p_home"])
        df["spread_line"] = sd * stats.norm.ppf(df["p_home"].clip(1e-4, 1 - 1e-4))
        df["spread_source"] = "moneyline_probit"
    else:
        df["spread_source"] = "spread"
    df["resid"] = df["result"] - df["spread_line"]
    df["tresid"] = df["total"] - df["total_line"]
    df = df[df["spread_line"].notna() & df["total_line"].notna()]
    out = df[CLEAN_COLUMNS].reset_index(drop=True)
    out.attrs.update(attrs)
    return out


def load_nba(config: MultiSportConfig | None = None) -> pd.DataFrame:
    """NBA games 2007-08 onward with closing spread, total and moneylines.

    ``Spread`` in the source is the expected home margin (positive = home
    favored), matching the nflverse convention used throughout.

    Args:
        config: File locations.

    Returns:
        Cleaned games table (see module docstring). ``season`` is the year the
        season started (2007 for 2007-08).
    """
    config = config or MultiSportConfig()
    path = _download(config.nba_sqlite_url, config.data_dir / "nba_odds_sbr.sqlite")
    con = sqlite3.connect(path)
    tables = pd.read_sql("select name from sqlite_master where type='table'", con)["name"].tolist()
    frames = []
    for t in tables:
        name = t.replace("odds_", "").replace("_new", "")
        if not (len(name) == 7 and name[4] == "-"):
            continue
        # Prefer the *_new tables (ISO dates); plain "2023-24"-style tables are used
        # only where no *_new table exists.
        if not t.endswith("_new") and f"odds_{name}_new" in tables:
            continue
        if t.startswith("odds_") and not t.endswith("_new") and name in tables:
            continue
        df = pd.read_sql(f'select * from "{t}"', con)
        if not df["Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all():
            continue
        df["season"] = int(name[:4])
        frames.append(df)
    con.close()
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "season": raw["season"].astype(int),
            "gameday": raw["Date"],
            "home_team": raw["Home"],
            "away_team": raw["Away"],
            "home_score": (raw["Points"] + raw["Win_Margin"]) / 2.0,
            "away_score": (raw["Points"] - raw["Win_Margin"]) / 2.0,
            "spread_line": _nba_numeric(raw["Spread"].replace({"PK": "0", "pk": "0"})),
            "total_line": _nba_numeric(raw["OU"]),
            "home_moneyline": _nba_numeric(raw["ML_Home"]),
            "away_moneyline": _nba_numeric(raw["ML_Away"]),
        }
    )
    out = out.drop_duplicates(subset=["gameday", "home_team", "away_team"])
    out = _resign_nba_spreads(out)
    return _finish(out, "nba", config)


def _nba_numeric(s: pd.Series) -> pd.Series:
    """Parse the sqlite odds columns, which mix numbers with strings like ``"+145\\xa0"``."""
    return pd.to_numeric(s.astype(str).str.replace("\xa0", "", regex=False).str.strip().str.lstrip("+"), errors="coerce")


def _resign_nba_spreads(out: pd.DataFrame) -> pd.DataFrame:
    """Restore the sign of unsigned spreads and drop rows inconsistent with the moneyline.

    Seasons 2007-08 through 2021-22 of the source store ``|spread|`` only. The
    sign is recovered from the moneyline favorite; rows with an even moneyline
    (``p_home == 0.5``) and a non-zero ``|spread|`` have no recoverable sign and
    are dropped. Rows whose (re)signed spread disagrees with the
    moneyline-implied margin by more than ``NBA_SPREAD_TOLERANCE`` points are
    dropped, as are (season, month) blocks where more than 20% of rows fail the
    check. On the current file this removes about 1.6% of rows (212 spread /
    moneyline disagreements, 165 even-moneyline rows, 1 without a moneyline)
    and no whole month; the counts are stored in ``DataFrame.attrs``
    (``nba_rows_in``, ``nba_rows_dropped_inconsistent``,
    ``nba_rows_dropped_even_ml``, ``nba_rows_dropped_no_ml``, ``nba_months_dropped``).

    Args:
        out: NBA table before :func:`_finish` with ``spread_line`` and moneylines.

    Returns:
        Filtered table with signed ``spread_line`` and a ``spread_resigned`` flag.
    """
    out = out.copy()
    has_ml = out["home_moneyline"].notna() & out["away_moneyline"].notna()
    p_home = np.full(len(out), np.nan)
    p_home[has_ml.to_numpy()], _ = implied_probs(out.loc[has_ml, "home_moneyline"], out.loc[has_ml, "away_moneyline"])
    neg_frac = out.groupby("season")["spread_line"].transform(lambda s: (s < 0).mean())
    unsigned = neg_frac < 0.05
    sign = np.sign(p_home - 0.5)
    out["spread_resigned"] = unsigned & has_ml
    even_ml = out["spread_resigned"] & (sign == 0) & (out["spread_line"].abs() > 0)  # before re-signing zeroes them
    out.loc[out["spread_resigned"], "spread_line"] = out.loc[out["spread_resigned"], "spread_line"].abs() * sign[out["spread_resigned"].to_numpy()]
    z = stats.norm.ppf(np.clip(p_home, 1e-4, 1 - 1e-4))
    signed_rows = (~unsigned) & has_ml & out["spread_line"].notna()
    if signed_rows.sum() < 100:
        signed_rows = has_ml & out["spread_line"].notna()  # fall back to the re-signed rows
    scale = float(np.sum(z[signed_rows] * out.loc[signed_rows, "spread_line"]) / np.sum(z[signed_rows] ** 2))
    implied = scale * z
    bad = has_ml & ~even_ml & (np.abs(out["spread_line"] - implied) > NBA_SPREAD_TOLERANCE)
    month = pd.to_datetime(out["gameday"]).dt.strftime("%Y-%m")
    bad_month = bad.groupby(month).transform("mean") > 0.20
    keep = ~bad & ~bad_month & has_ml & ~even_ml
    res = out[keep].reset_index(drop=True)
    res.attrs.update(
        {
            "nba_rows_in": int(len(out)),
            "nba_rows_dropped_inconsistent": int((bad | (bad_month & has_ml & ~even_ml)).sum()),
            "nba_rows_dropped_even_ml": int(even_ml.sum()),
            "nba_rows_dropped_no_ml": int((~has_ml).sum()),
            "nba_months_dropped": sorted(month[bad_month & ~bad].unique().tolist()) if bad_month.any() else [],
        }
    )
    return res


def load_sbr_archive(sport: str, config: MultiSportConfig | None = None) -> pd.DataFrame:
    """NHL, MLB or NBA games 2011-2021 with closing moneylines and totals.

    Args:
        sport: ``"nhl"``, ``"mlb"`` or ``"nba"``.
        config: File locations.

    Returns:
        Cleaned games table. For NHL/MLB ``spread_line`` is the probit-implied
        margin from the moneyline (``spread_source == "moneyline_probit"``); for
        NBA the archive's closing spread is used. Games the archive dates before
        October of their season's starting year are moved forward one year (see
        the inline comment); the count is in ``attrs["archive_rows_redated"]``.
    """
    if sport == "mlb":
        raise ValueError(
            "the 10-year MLB archive has mis-paired home/away rows (verified on 2019 Opening Day); "
            "use load_mlb() for 2021+ or load_mlb_archive_repaired() for 2011-2020"
        )
    config = config or MultiSportConfig()
    path = _download(config.sbr_url.format(sport=sport), config.data_dir / f"{sport}_archive_10Y.json")
    raw = pd.DataFrame(json.loads(Path(path).read_text()))
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    out = pd.DataFrame(
        {
            "season": num("season").astype("Int64"),
            "gameday": pd.to_datetime(num("date").astype("Int64").astype(str), format="%Y%m%d", errors="coerce"),
            "home_team": raw["home_team"].astype(str),
            "away_team": raw["away_team"].astype(str),
            "home_score": num("home_final"),
            "away_score": num("away_final"),
            "total_line": num("close_over_under").replace(0.0, np.nan),
            "home_moneyline": num("home_close_ml").replace(0.0, np.nan),
            "away_moneyline": num("away_close_ml").replace(0.0, np.nan),
        }
    )
    if "home_close_spread" in raw and sport == "nba":
        out["spread_line"] = -num("home_close_spread")
    out = out[out["gameday"].notna() & out["season"].notna()]
    out["season"] = out["season"].astype(int)
    # The archive dates some games with the wrong year: NHL 2019-20 bubble playoffs
    # (Aug-Sep 2020) are dated 2019-08/09 and the Jan-Mar 2021 games of 2020-21 are
    # dated 2020-01..03. A season that starts in October cannot have games before
    # October of its own starting year, so shift those dates forward one year.
    misdated = (out["gameday"].dt.year == out["season"]) & (out["gameday"].dt.month < 10)
    out.loc[misdated, "gameday"] = out.loc[misdated, "gameday"] + pd.DateOffset(years=1)
    out.attrs["archive_rows_redated"] = int(misdated.sum())
    return _finish(out, sport, config)


MAX_ABS_ODDS = 1000.0
MAX_OPEN_CLOSE_PROB_MOVE = 0.15


def _raw_prob(odds: float) -> float:
    return 100.0 / (100.0 + odds) if odds > 0 else -odds / (-odds + 100.0)


def _valid_price(entry: dict, key: str) -> bool:
    """A book's closing price is usable if it is a pre-game price.

    Some ``currentLine`` values were captured in-play (``-100000``, or a +4000
    underdog that was already trailing). A price is rejected when it is outside
    ``[100, MAX_ABS_ODDS]`` in absolute value or when its implied probability
    moved more than ``MAX_OPEN_CLOSE_PROB_MOVE`` from the book's opening price.
    """
    cl = (entry.get("currentLine") or {}).get(key)
    if cl is None or not (100 <= abs(float(cl)) <= MAX_ABS_ODDS):
        return False
    op = (entry.get("openingLine") or {}).get(key)
    if op is not None and abs(float(op)) >= 100:
        if abs(_raw_prob(float(cl)) - _raw_prob(float(op))) > MAX_OPEN_CLOSE_PROB_MOVE:
            return False
    return True


def _prob_to_american(p: float) -> float:
    return -100.0 * p / (1 - p) if p >= 0.5 else 100.0 * (1 - p) / p


def _median_line(entries: list[dict], key: str, min_books: int = 2) -> float:
    """Median closing value of ``key`` across books with a valid pre-game price.

    American prices are averaged in implied-probability space (the arithmetic
    median of e.g. ``-104, +102, -110, +100`` is meaningless across the +/-100
    discontinuity) and converted back. For non-price keys (``total``,
    ``homeSpread``) the book is kept only if its price columns look pre-game
    (see :func:`_valid_price`).
    """
    vals = []
    is_price = key.endswith("Odds")
    for e in entries:
        cl = e.get("currentLine") or {}
        v = cl.get(key)
        if v is None:
            continue
        price_keys = [key] if is_price else [k for k in cl if k.endswith("Odds")]
        if not all(_valid_price(e, k) for k in price_keys):
            continue
        vals.append(_raw_prob(float(v)) if is_price else float(v))
    if len(vals) < min_books:
        return np.nan
    med = float(np.median(vals))
    return _prob_to_american(med) if is_price else med


def load_mlb(config: MultiSportConfig | None = None, with_pitchers: bool = True) -> pd.DataFrame:
    """MLB games 2021-2025 with closing consensus lines and starting pitchers.

    Closing lines are the median ``currentLine`` across the books in the source
    (DraftKings, FanDuel, Caesars, bet365, BetMGM, BetRivers). Spring-training
    and All-Star games are dropped. Starting pitchers come from
    ``config.pitchers_parquet`` (built with :mod:`geo_model.parlay.retrosheet`).

    Args:
        config: File locations.
        with_pitchers: Merge Retrosheet starting pitchers into ``home_qb_id`` /
            ``away_qb_id`` (the generic "key player" slots used by the pair code).

    Returns:
        Cleaned games table with ``spread_source == "moneyline_probit"`` and
        ``runline`` / ``runline_home_odds`` columns in addition to the schema.
    """
    config = config or MultiSportConfig()
    path = _download(config.mlb_odds_url, config.data_dir / "mlb_odds_dataset.json")
    data = json.loads(Path(path).read_text())
    rows = []
    for date, games in data.items():
        for order, g in enumerate(games):
            gv = g["gameView"]
            od = g.get("odds") or {}
            gtype = MLB_GAME_TYPES.get(gv.get("gameType"))
            if gtype is None or not str(gv.get("gameStatusText", "")).startswith("Final"):
                continue
            home = gv["homeTeam"]["shortName"]
            away = gv["awayTeam"]["shortName"]
            rows.append(
                {
                    "gameday": date,
                    "order": order,
                    "game_type": gtype,
                    "home_team": MLB_SHORTNAME_TO_CANON.get(home, home),
                    "away_team": MLB_SHORTNAME_TO_CANON.get(away, away),
                    "home_score": gv.get("homeTeamScore"),
                    "away_score": gv.get("awayTeamScore"),
                    "home_moneyline": _median_line(od.get("moneyline") or [], "homeOdds"),
                    "away_moneyline": _median_line(od.get("moneyline") or [], "awayOdds"),
                    "total_line": _median_line(od.get("totals") or [], "total"),
                    "over_odds": _median_line(od.get("totals") or [], "overOdds"),
                    "under_odds": _median_line(od.get("totals") or [], "underOdds"),
                    "runline": _median_line(od.get("pointspread") or [], "homeSpread"),
                    "runline_home_odds": _median_line(od.get("pointspread") or [], "homeOdds"),
                }
            )
    out = pd.DataFrame(rows)
    out = out[~out["home_team"].isin(["AL", "NL"])]
    out["gameday"] = pd.to_datetime(out["gameday"], utc=True)
    out["season"] = out["gameday"].dt.year
    out = out[out["home_moneyline"].notna() & out["away_moneyline"].notna()]
    clean = _finish(out, "mlb", config)
    extra = out[["gameday", "home_team", "away_team", "order", "runline", "runline_home_odds"]].copy()
    extra["seq"] = extra.groupby(["gameday", "home_team", "away_team"]).cumcount()
    clean["seq"] = clean.groupby(["gameday", "home_team", "away_team"]).cumcount()
    clean = clean.merge(extra.drop(columns="order"), on=["gameday", "home_team", "away_team", "seq"], how="left")
    if with_pitchers and config.pitchers_parquet.exists():
        sp = pd.read_parquet(config.pitchers_parquet)
        sp = sp.sort_values(["gameday", "home", "away", "number"])
        sp["seq"] = sp.groupby(["gameday", "home", "away"]).cumcount()
        sp = sp.rename(columns={"home": "home_team", "away": "away_team"})
        clean = clean.merge(
            sp[["gameday", "home_team", "away_team", "seq", "home_sp", "away_sp"]],
            on=["gameday", "home_team", "away_team", "seq"],
            how="left",
        )
        clean["home_qb_id"] = clean["home_sp"]
        clean["away_qb_id"] = clean["away_sp"]
        clean = clean.drop(columns=["home_sp", "away_sp"])
    return clean.drop(columns=["seq"]).reset_index(drop=True)


MLB_NICKNAME_TO_CANON = {
    "Diamondbacks": "ARI", "Braves": "ATL", "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CHW",
    "Reds": "CIN", "Indians": "CLE", "Guardians": "CLE", "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK", "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD",
    "Mariners": "SEA", "Giants": "SF", "Cardinals": "STL", "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR",
    "Nationals": "WAS", "SFO": "SF", "SDG": "SD", "TAM": "TB", "TBR": "TB", "KAN": "KC", "CUB": "CHC", "CWS": "CHW",
    "LOS": "LAD", "BRS": "BOS", "NYY": "NYY", "NYM": "NYM", "WAS": "WAS", "CHW": "CHW", "LAA": "LAA", "LAD": "LAD",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL",
    "DET": "DET", "HOU": "HOU", "KC": "KC", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "OAK": "OAK", "PHI": "PHI",
    "PIT": "PIT", "SD": "SD", "SEA": "SEA", "SF": "SF", "STL": "STL", "TB": "TB", "TEX": "TEX", "TOR": "TOR",
}


def load_mlb_archive_repaired(config: MultiSportConfig | None = None, seasons: tuple[int, int] = (2011, 2020)) -> pd.DataFrame:
    """MLB 2011-2020 from the 10-year archive after repairing its row misalignment.

    In the source, row ``i`` carries the *away* side of game ``i`` in its
    ``home_*`` fields and the *home* side of game ``i-1`` in its ``away_*``
    fields (the scraper paired consecutive table rows). Shifting the ``away_*``
    fields up by one row within each date restores the games: validated on the
    2021 overlap with :func:`load_mlb` (2046 of 2209 games matched by date and
    teams, 97.9% exact score agreement, moneyline correlation 0.96-0.97, total
    line correlation 0.86 using the row-``i`` over/under).

    Args:
        config: File locations.
        seasons: Inclusive season range to keep (2021 is available from the
            clean multi-book file instead).

    Returns:
        Cleaned games table (moneyline-probit ``spread_line``); no pitchers.
    """
    config = config or MultiSportConfig()
    path = _download(config.sbr_url.format(sport="mlb"), config.data_dir / "mlb_archive_10Y.json")
    raw = pd.DataFrame(json.loads(Path(path).read_text())).reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors="coerce")  # noqa: E731
    date = num("date").astype("Int64")
    nxt = raw.shift(-1)
    nnum = lambda c: pd.to_numeric(nxt[c], errors="coerce")  # noqa: E731
    out = pd.DataFrame(
        {
            "season": num("season").astype("Int64"),
            "gameday": pd.to_datetime(date.astype(str), format="%Y%m%d", errors="coerce"),
            "away_team": raw["home_team"].map(MLB_NICKNAME_TO_CANON),
            "away_score": num("home_final"),
            "away_moneyline": num("home_close_ml").replace(0.0, np.nan),
            "home_team": nxt["away_team"].map(MLB_NICKNAME_TO_CANON),
            "home_score": nnum("away_final"),
            "home_moneyline": nnum("away_close_ml").replace(0.0, np.nan),
            "total_line": num("close_over_under").replace(0.0, np.nan),
            "same_date": date == pd.to_numeric(nxt["date"], errors="coerce").astype("Int64"),
        }
    )
    out = out[out["same_date"].fillna(False).astype(bool)].drop(columns="same_date")
    out = out[out["gameday"].notna() & out["home_team"].notna() & out["away_team"].notna()]
    out["season"] = out["season"].astype(int)
    out = out[out["season"].between(*seasons)]
    out = out[(out["home_moneyline"].abs() >= 100) & (out["away_moneyline"].abs() >= 100)]
    out = out.drop_duplicates(subset=["gameday", "home_team", "away_team", "home_score", "away_score"])
    return _finish(out, "mlb", config)
