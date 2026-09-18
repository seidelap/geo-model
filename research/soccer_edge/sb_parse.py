"""Parse StatsBomb open-data JSON into compact Parquet tables.

Produces, under ``<data_dir>/processed/``:

* ``matches.parquet``      one row per match (teams, referee, scores, 360 flag)
* ``player_match.parquet`` one row per player appearance (position, minutes, counts)
* ``team_match.parquet``   one row per team per match (style + volume features)
* ``fouls.parquet``        one row per foul committed (fouler, fouled, location, card)
* ``shapes360.parquet``    one row per defending team per match (block depth/compactness
                           from 360 freeze frames, where available)

Coordinates: StatsBomb pitch is 120 x 80, always from the perspective of the team
performing the event (attacking left to right). ``y > 40`` is the attacker's right flank.

Usage::

    SOCCER_EDGE_DATA=/path/to/sb python research/soccer_edge/sb_parse.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "soccer_edge"

# Position family used for flank matching (lineup position names).
FULLBACK_POS = {"Right Back", "Left Back", "Right Wing Back", "Left Wing Back"}
WINGER_POS = {
    "Right Wing", "Left Wing", "Right Midfield", "Left Midfield",
    "Right Attacking Midfield", "Left Attacking Midfield",
}
DEF_ACTIONS = {"Pressure", "Duel", "Interception", "Block", "Clearance", "Foul Committed",
               "Ball Recovery", "50/50"}


def data_dir() -> Path:
    """Resolve the raw data directory from ``SOCCER_EDGE_DATA`` or the repo default."""
    return Path(os.environ.get("SOCCER_EDGE_DATA", DEFAULT_DATA_DIR))


def side_of(y: float | None) -> str | None:
    """Classify a y coordinate (attacker's view) into left / centre / right channel."""
    if y is None:
        return None
    if y < 26.67:
        return "left"
    if y > 53.33:
        return "right"
    return "centre"


def _clock_to_min(clock: str | None) -> float | None:
    if clock is None:
        return None
    mm, ss = clock.split(":")
    return int(mm) + int(ss) / 60.0


@dataclass
class PlayerAcc:
    """Mutable per-player counters for one match."""

    team_id: int
    player_id: int
    player: str
    counts: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def _match_end_minute(events: list[dict[str, Any]]) -> float:
    end = 0.0
    for e in events:
        if e["type"]["name"] == "Half End":
            m = e["minute"] + e["second"] / 60.0
            end = max(end, m)
    return end or 90.0


def parse_match(args: tuple[int, dict[str, Any], Path]) -> dict[str, Any] | None:
    """Parse one match into row lists for every output table.

    Args:
        args: ``(match_id, match_meta, data_dir)``.

    Returns:
        Dict with keys ``match``, ``players``, ``teams``, ``fouls``, ``shape`` or ``None``
        if the events file is missing.
    """
    match_id, meta, ddir = args
    ev_path = ddir / "events" / f"{match_id}.json"
    lu_path = ddir / "lineups" / f"{match_id}.json"
    if not ev_path.exists() or not lu_path.exists():
        return None
    try:
        with open(ev_path) as f:
            events = json.load(f)
        with open(lu_path) as f:
            lineups = json.load(f)
    except json.JSONDecodeError as exc:  # corrupt download - skip the match
        print(f"skipping {match_id}: {exc}", file=sys.stderr)
        return None

    home_id = meta["home_team"]["home_team_id"]
    away_id = meta["away_team"]["away_team_id"]
    team_names = {home_id: meta["home_team"]["home_team_name"],
                  away_id: meta["away_team"]["away_team_name"]}
    end_min = _match_end_minute(events)

    # ---- players from lineups -------------------------------------------------------
    players: dict[int, PlayerAcc] = {}
    prow: dict[int, dict[str, Any]] = {}
    for tl in lineups:
        tid = tl["team_id"]
        for p in tl["lineup"]:
            pid = p["player_id"]
            pos = p.get("positions", [])
            if not pos:
                continue
            start = _clock_to_min(pos[0]["from"]) or 0.0
            last_to = pos[-1].get("to")
            end = _clock_to_min(last_to) if last_to else end_min
            # Some 'to' clocks exceed the Half End clock (added time) - clip.
            minutes = max(0.0, min(end, end_min + 10) - start)
            prow[pid] = {
                "match_id": match_id, "team_id": tid, "player_id": pid,
                "player": p["player_name"], "nickname": p.get("player_nickname"),
                "starter": pos[0]["start_reason"] == "Starting XI",
                "start_position": pos[0]["position"],
                "n_positions": len({q["position"] for q in pos}),
                "positions": "|".join(q["position"] for q in pos),
                "start_min": start, "minutes": minutes,
            }
            players[pid] = PlayerAcc(tid, pid, p["player_name"])

    def pc(pid: int | None, key: str, val: float = 1.0) -> None:
        if pid is not None and pid in players:
            players[pid].counts[key] += val

    # ---- team accumulators -----------------------------------------------------------
    tc: dict[int, dict[str, float]] = {home_id: defaultdict(float), away_id: defaultdict(float)}
    by_id = {e["id"]: e for e in events}
    fouls: list[dict[str, Any]] = []
    # For possession share: seconds of possession per team from possession sequences.
    poss_seconds: dict[int, float] = defaultdict(float)
    last_poss: tuple[int, int, float] | None = None  # (possession idx, team, start min)

    for e in events:
        t = e["type"]["name"]
        tid = e["team"]["id"]
        opp = away_id if tid == home_id else home_id
        pid = e.get("player", {}).get("id")
        loc = e.get("location")
        x = loc[0] if loc else None
        y = loc[1] if loc else None
        minute = e["minute"] + e["second"] / 60.0
        pteam = e["possession_team"]["id"]
        pidx = e["possession"]
        if last_poss is None or last_poss[0] != pidx:
            if last_poss is not None:
                poss_seconds[last_poss[1]] += max(0.0, minute - last_poss[2])
            last_poss = (pidx, pteam, minute)

        if t == "Pass":
            p = e["pass"]
            ptype = p.get("type", {}).get("name")
            complete = "outcome" not in p
            pc(pid, "passes"); tc[tid]["passes"] += 1
            if complete:
                pc(pid, "passes_completed"); tc[tid]["passes_completed"] += 1
            if p.get("cross"):
                pc(pid, "crosses"); tc[tid]["crosses"] += 1
                if complete:
                    tc[tid]["crosses_completed"] += 1
            if ptype == "Corner":
                pc(pid, "corners_taken"); tc[tid]["corners"] += 1
                tc[opp]["corners_against"] += 1
            if ptype == "Throw-in":
                tc[tid]["throw_ins"] += 1
            if x is not None:
                if x >= 80:
                    tc[tid]["passes_final_third"] += 1
                if x < 40:
                    tc[tid]["passes_own_third"] += 1
                # PPDA numerator: opponent passes in their own 60% of the pitch.
                if x < 72:
                    tc[opp]["ppda_passes_allowed"] += 1
                tc[tid]["pass_x_sum"] += x
            if p.get("height", {}).get("name") == "High Pass":
                tc[tid]["high_passes"] += 1
            end_loc = p.get("end_location")
            if end_loc and end_loc[0] >= 102 and 18 <= end_loc[1] <= 62:
                tc[tid]["passes_into_box"] += 1
            if p.get("switch"):
                tc[tid]["switches"] += 1
            pc(pid, "pass_length_sum", p.get("length", 0.0))
            if p.get("shot_assist"):
                pc(pid, "key_passes")
        elif t == "Carry":
            c = e["carry"]
            if loc and c.get("end_location"):
                d = float(np.hypot(c["end_location"][0] - x, c["end_location"][1] - y))
                pc(pid, "carry_dist", d); tc[tid]["carry_dist"] += d
                if c["end_location"][0] - x >= 5:
                    pc(pid, "progressive_carries")
        elif t == "Dribble":
            outcome = e["dribble"].get("outcome", {}).get("name")
            pc(pid, "dribbles"); tc[tid]["dribbles"] += 1
            if outcome == "Complete":
                pc(pid, "dribbles_completed"); tc[tid]["dribbles_completed"] += 1
            s = side_of(y)
            if s:
                pc(pid, f"dribbles_{s}")
            if x is not None and x >= 80:
                pc(pid, "dribbles_final_third")
        elif t == "Dribbled Past":
            pc(pid, "dribbled_past"); tc[tid]["dribbled_past"] += 1
        elif t == "Foul Committed":
            fc = e.get("foul_committed", {})
            card = fc.get("card", {}).get("name")
            pc(pid, "fouls"); tc[tid]["fouls"] += 1
            if x is not None and x < 40:
                pc(pid, "fouls_def_third")
            s = side_of(y)
            if s:
                pc(pid, f"fouls_{s}")
            # Find the fouled player via related Foul Won.
            fouled_pid = fouled_pos = None
            fouled_dribbling = False
            for rid in e.get("related_events", []) or []:
                r = by_id.get(rid)
                if r and r["type"]["name"] == "Foul Won":
                    fouled_pid = r.get("player", {}).get("id")
                    fouled_pos = r.get("position", {}).get("name")
                    fw = r.get("foul_won", {})
                    fouled_dribbling = bool(fw.get("advantage")) is False and False
            # Was the fouled player dribbling / carrying just before? Check the preceding
            # events by the fouled player in the same possession within 5 seconds.
            if fouled_pid is not None:
                for k in range(e["index"] - 2, max(0, e["index"] - 12), -1):
                    q = events[k] if k < len(events) else None
                    if not q:
                        continue
                    if q.get("player", {}).get("id") == fouled_pid and q["type"]["name"] in (
                            "Dribble", "Carry") and minute - (q["minute"] + q["second"] / 60) < 0.1:
                        fouled_dribbling = True
                        break
            fouls.append({
                "match_id": match_id, "minute": minute, "period": e["period"],
                "team_id": tid, "player_id": pid, "position": e.get("position", {}).get("name"),
                "x": x, "y": y, "side": side_of(y), "card": card,
                "foul_type": fc.get("type", {}).get("name"), "penalty": bool(fc.get("penalty")),
                "fouled_player_id": fouled_pid, "fouled_position": fouled_pos,
                "fouled_dribbling": fouled_dribbling,
            })
            if card == "Yellow Card":
                pc(pid, "yellow"); tc[tid]["yellow"] += 1
            elif card == "Second Yellow":
                pc(pid, "second_yellow"); tc[tid]["yellow"] += 1; tc[tid]["red"] += 1
            elif card == "Red Card":
                pc(pid, "red"); tc[tid]["red"] += 1
        elif t == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name")
            if card == "Yellow Card":
                pc(pid, "yellow"); pc(pid, "yellow_bad_behaviour"); tc[tid]["yellow"] += 1
            elif card == "Second Yellow":
                pc(pid, "second_yellow"); tc[tid]["yellow"] += 1; tc[tid]["red"] += 1
            elif card == "Red Card":
                pc(pid, "red"); tc[tid]["red"] += 1
        elif t == "Foul Won":
            pc(pid, "fouls_won"); tc[tid]["fouls_won"] += 1
            s = side_of(y)
            if s:
                pc(pid, f"fouls_won_{s}")
            if x is not None and x >= 80:
                pc(pid, "fouls_won_final_third")
        elif t == "Shot":
            sh = e["shot"]
            out = sh.get("outcome", {}).get("name")
            pc(pid, "shots"); tc[tid]["shots"] += 1
            tc[tid]["xg"] += sh.get("statsbomb_xg", 0.0)
            if out == "Blocked":
                pc(pid, "shots_blocked"); tc[tid]["shots_blocked"] += 1
            if out in ("Goal", "Saved", "Saved To Post"):
                tc[tid]["shots_on_target"] += 1
            if x is not None and x >= 102 and 18 <= y <= 62:
                tc[tid]["shots_in_box"] += 1
            if sh.get("type", {}).get("name") == "Penalty":
                tc[tid]["penalties"] += 1
        elif t == "Block":
            pc(pid, "blocks"); tc[tid]["blocks"] += 1
        elif t == "Pressure":
            pc(pid, "pressures"); tc[tid]["pressures"] += 1
        elif t == "Duel":
            if e.get("duel", {}).get("type", {}).get("name") == "Tackle":
                pc(pid, "tackles"); tc[tid]["tackles"] += 1
        elif t == "Interception":
            pc(pid, "interceptions"); tc[tid]["interceptions"] += 1
        elif t == "Clearance":
            pc(pid, "clearances"); tc[tid]["clearances"] += 1
        elif t == "Ball Recovery":
            pc(pid, "recoveries")
        elif t == "Ball Receipt*":
            pc(pid, "receipts")

        if t in DEF_ACTIONS and x is not None:
            tc[tid]["def_actions"] += 1
            tc[tid]["def_x_sum"] += x
            if x >= 48:  # PPDA denominator zone (opponent's 60%)
                tc[tid]["ppda_def_actions"] += 1
            if x >= 80:
                tc[tid]["def_actions_high"] += 1
    if last_poss is not None:
        poss_seconds[last_poss[1]] += max(0.0, end_min - last_poss[2])

    # ---- assemble rows -----------------------------------------------------------------
    player_rows = []
    for pid, row in prow.items():
        r = dict(row)
        r.update(players[pid].counts)
        player_rows.append(r)

    total_poss = sum(poss_seconds.values()) or 1.0
    team_rows = []
    for tid in (home_id, away_id):
        opp = away_id if tid == home_id else home_id
        c = tc[tid]
        r: dict[str, Any] = {
            "match_id": match_id, "team_id": tid, "team": team_names[tid],
            "opp_id": opp, "opp": team_names[opp], "home": tid == home_id,
            "possession": poss_seconds[tid] / total_poss,
        }
        r.update(c)
        r["ppda"] = c["ppda_passes_allowed"] / max(1.0, c["ppda_def_actions"])
        r["def_line_x"] = c["def_x_sum"] / max(1.0, c["def_actions"])
        r["pass_x_mean"] = c["pass_x_sum"] / max(1.0, c["passes"])
        team_rows.append(r)

    match_row = {
        "match_id": match_id,
        "competition_id": meta["competition"]["competition_id"],
        "competition": meta["competition"]["competition_name"],
        "season_id": meta["season"]["season_id"],
        "season": meta["season"]["season_name"],
        "date": meta["match_date"], "kick_off": meta.get("kick_off"),
        "match_week": meta.get("match_week"),
        "stage": (meta.get("competition_stage") or {}).get("name"),
        "home_team_id": home_id, "home_team": team_names[home_id],
        "away_team_id": away_id, "away_team": team_names[away_id],
        "home_score": meta["home_score"], "away_score": meta["away_score"],
        "referee_id": (meta.get("referee") or {}).get("id"),
        "referee": (meta.get("referee") or {}).get("name"),
        "end_minute": end_min,
        "has_360": meta.get("match_status_360") == "available",
    }

    shape_rows = parse_360(match_id, ddir, by_id, home_id, away_id)
    return {"match": match_row, "players": player_rows, "teams": team_rows,
            "fouls": fouls, "shape": shape_rows}


def parse_360(match_id: int, ddir: Path, by_id: dict[str, dict[str, Any]],
              home_id: int, away_id: int) -> list[dict[str, Any]]:
    """Summarise defending-team shape from 360 freeze frames.

    For every frame attached to an in-possession event by team A located in B's half
    (``x >= 60`` from A's view), the non-teammate outfield players are B's block.
    Block depth is measured as distance from B's own goal line (``120 - x``); higher
    means B defends further up the pitch. Compactness is the spread of B's block.

    Returns one row per defending team with frame counts and means, or ``[]`` when the
    360 file does not exist.
    """
    p = ddir / "three-sixty" / f"{match_id}.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            frames = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"skipping 360 for {match_id}: {exc}", file=sys.stderr)
        return []
    acc: dict[int, dict[str, list[float]]] = {home_id: defaultdict(list),
                                              away_id: defaultdict(list)}
    for fr in frames:
        e = by_id.get(fr["event_uuid"])
        if e is None or e["type"]["name"] not in ("Pass", "Carry", "Shot", "Dribble",
                                                   "Ball Receipt*"):
            continue
        loc = e.get("location")
        if not loc:
            continue
        att = e["team"]["id"]
        if e["possession_team"]["id"] != att:
            continue
        dfd = away_id if att == home_id else home_id
        opp = [ff["location"] for ff in fr["freeze_frame"]
               if not ff["teammate"] and not ff["keeper"]]
        if len(opp) < 7:
            continue
        xs = np.array([o[0] for o in opp]); ys = np.array([o[1] for o in opp])
        settled = loc[0] >= 60
        key = "settled" if settled else "all"
        for k in ({"settled", "all"} if settled else {"all"}):
            a = acc[dfd]
            a[f"{k}_depth"].append(120 - xs.mean())          # mean block distance from own goal
            a[f"{k}_line"].append(120 - xs.max())            # deepest outfield defender
            a[f"{k}_len"].append(xs.max() - xs.min())        # block length (x spread)
            a[f"{k}_width"].append(ys.max() - ys.min())      # block width (y spread)
            a[f"{k}_n"].append(len(opp))
            a[f"{k}_ball_dist"].append(float(np.hypot(xs - loc[0], ys - loc[1]).min()))
    rows = []
    for tid, a in acc.items():
        if not a.get("all_depth"):
            continue
        r: dict[str, Any] = {"match_id": match_id, "team_id": tid,
                             "frames_all": len(a["all_depth"]),
                             "frames_settled": len(a.get("settled_depth", []))}
        for k, v in a.items():
            if k.endswith("_n"):
                continue
            r[f"m_{k}"] = float(np.mean(v))
        rows.append(r)
    return rows


def build(ddir: Path, workers: int = 4) -> None:
    """Parse every downloaded match and write the Parquet tables."""
    metas: list[tuple[int, dict[str, Any], Path]] = []
    for mp in sorted((ddir / "matches").glob("*.json")):
        with open(mp) as f:
            for m in json.load(f):
                metas.append((m["match_id"], m, ddir))
    print(f"{len(metas)} match records listed", file=sys.stderr)
    out = {"match": [], "players": [], "teams": [], "fouls": [], "shape": []}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(parse_match, metas, chunksize=8)):
            if res is None:
                continue
            out["match"].append(res["match"])
            for k in ("players", "teams", "fouls", "shape"):
                out[k].extend(res[k])
            if i % 200 == 0:
                print(f"  parsed {i}", file=sys.stderr)
    proc = ddir / "processed"
    proc.mkdir(exist_ok=True)
    names = {"match": "matches", "players": "player_match", "teams": "team_match",
             "fouls": "fouls", "shape": "shapes360"}
    for k, name in names.items():
        df = pd.DataFrame(out[k])
        if k in ("players", "teams"):
            num = [c for c in df.columns if df[c].dtype == object and c not in (
                "player", "nickname", "positions", "start_position", "team", "opp")]
            for c in num:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            df = df.fillna({c: 0.0 for c in df.columns if df[c].dtype.kind == "f"})
        df.to_parquet(proc / f"{name}.parquet", index=False)
        print(f"{name}: {df.shape}", file=sys.stderr)


if __name__ == "__main__":
    build(data_dir(), workers=int(os.environ.get("SOCCER_EDGE_WORKERS", "4")))
