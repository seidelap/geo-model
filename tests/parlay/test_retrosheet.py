from __future__ import annotations

from pathlib import Path

import pandas as pd

from geo_model.parlay.retrosheet import load_starting_pitchers, parse_event_file

EVENT = """id,ATL202304060
version,2
info,visteam,SDN
info,hometeam,ATL
info,date,2023/04/06
info,number,0
info,gametype,regular
start,grist001,"Trent Grisham",0,1,8
start,snelb001,"Blake Snell",0,0,1
start,acunr001,"Ronald Acuna",1,1,9
start,strim001,"Max Fried",1,0,1
play,1,0,grist001,00,,8/F
id,ATL202304070
version,2
info,visteam,SDN
info,hometeam,ATL
info,date,2023/04/07
info,number,1
start,darvy001,"Yu Darvish",0,0,1
start,eldes001,"Bryce Elder",1,0,1
"""


def test_parse_event_file(tmp_path: Path) -> None:
    f = tmp_path / "2023ATL.EVN"
    f.write_text(EVENT)
    rows = parse_event_file(f)
    assert len(rows) == 2
    assert rows[0]["home"] == "ATL" and rows[0]["away"] == "SDN"
    assert rows[0]["home_sp"] == "strim001" and rows[0]["away_sp"] == "snelb001"
    assert rows[0]["date"] == "20230406" and rows[0]["number"] == 0
    assert rows[1]["gametype"] is None and rows[1]["number"] == 1


def test_load_starting_pitchers_maps_codes_and_defaults_gametype(tmp_path: Path) -> None:
    (tmp_path / "2023").mkdir()
    (tmp_path / "2023" / "2023ATL.EVN").write_text(EVENT)
    df = load_starting_pitchers(tmp_path, [2023])
    assert list(df["home"]) == ["ATL", "ATL"] and list(df["away"]) == ["SD", "SD"]
    assert df["gametype"].tolist() == ["regular", "regular"]
    assert df["gameday"].dt.tz is not None
    assert df["home_sp"].tolist() == ["strim001", "eldes001"]
