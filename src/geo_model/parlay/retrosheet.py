"""Parse Retrosheet event files for starting pitchers.

The Chadwick Bureau mirror (``github.com/chadwickbureau/retrosheet``,
``seasons/<year>/*.EV[AN]``) is reachable from this environment via a sparse
git checkout. Only the ``id``, ``info`` and ``start`` records are used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Retrosheet team code -> canonical key shared with the odds dataset.
RETROSHEET_TO_CANON = {
    "ANA": "LAA", "ARI": "ARI", "ATH": "OAK", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHA": "CHW", "CHN": "CHC", "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KCA": "KC", "LAN": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NYA": "NYY", "NYN": "NYM", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SDN": "SD",
    "SEA": "SEA", "SFN": "SF", "SLN": "STL", "TBA": "TB", "TEX": "TEX", "TOR": "TOR",
    "WAS": "WAS",
}


def parse_event_file(path: Path) -> list[dict]:
    """Extract one row per game from a Retrosheet event file.

    Args:
        path: ``*.EVA`` / ``*.EVN`` / ``*.EVE`` file.

    Returns:
        Rows with ``date`` (``YYYYMMDD`` string), ``home``, ``away`` (Retrosheet
        codes), ``number`` (0 single game, 1/2 doubleheader), ``gametype``,
        ``home_sp``, ``away_sp`` (Retrosheet player ids).
    """
    rows: list[dict] = []
    cur: dict | None = None
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.split(",")
        tag = parts[0]
        if tag == "id":
            if cur is not None:
                rows.append(cur)
            cur = {"game_key": parts[1], "home_sp": None, "away_sp": None, "gametype": None}
        elif cur is None:
            continue
        elif tag == "info" and len(parts) >= 3:
            k, v = parts[1], parts[2]
            if k == "visteam":
                cur["away"] = v
            elif k == "hometeam":
                cur["home"] = v
            elif k == "date":
                cur["date"] = v.replace("/", "")
            elif k == "number":
                cur["number"] = int(v) if v.isdigit() else 0
            elif k == "gametype":
                cur["gametype"] = v
        elif tag == "start" and len(parts) >= 6:
            if parts[5].strip() == "1":
                if parts[3].strip() == "1":
                    cur["home_sp"] = parts[1]
                else:
                    cur["away_sp"] = parts[1]
    if cur is not None:
        rows.append(cur)
    return rows


def load_starting_pitchers(seasons_dir: Path, years: list[int]) -> pd.DataFrame:
    """Starting pitchers for every game in the given seasons.

    Args:
        seasons_dir: The ``seasons`` directory of the Retrosheet mirror.
        years: Seasons to parse.

    Returns:
        Table with ``gameday`` (UTC midnight), ``home``, ``away`` (canonical
        codes), ``number``, ``gametype``, ``home_sp``, ``away_sp``.
    """
    rows: list[dict] = []
    for y in years:
        for f in sorted(Path(seasons_dir, str(y)).glob("*.EV*")):
            rows.extend(parse_event_file(f))
    df = pd.DataFrame(rows)
    df["gametype"] = df["gametype"].fillna("regular")
    df = df[df["gametype"].isin(["regular", "wildcard", "divisionseries", "lcs", "worldseries"])]
    df["home"] = df["home"].map(RETROSHEET_TO_CANON)
    df["away"] = df["away"].map(RETROSHEET_TO_CANON)
    df = df[df["home"].notna() & df["away"].notna()]
    df["gameday"] = pd.to_datetime(df["date"], format="%Y%m%d", utc=True)
    df["number"] = df["number"].fillna(0).astype(int)
    return df[["gameday", "home", "away", "number", "gametype", "home_sp", "away_sp"]].sort_values(
        ["gameday", "home", "number"]
    ).reset_index(drop=True)
