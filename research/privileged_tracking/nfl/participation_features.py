"""Pure feature functions for the NFL participation stage (who is on the field).

Everything in this module is a deterministic function of plain DataFrames / arrays so it
can be unit-tested on synthetic input. The leakage contract for the "tendency" features
is: a play may only see counts from games that were completed strictly before the play's
game (``prior_game_class_counts``), from the previous season (``previous_season_shares``)
or from league-wide plays in strictly earlier weeks (``league_prior_shares``).

Column conventions used throughout:

* ``season`` (int), ``week`` (int), ``game_id`` (any hashable) order and identify games;
  a team plays at most one game per week.
* Personnel strings follow the NGS convention parsed by
  :func:`research.privileged_tracking.nfl.tracking_features.parse_personnel`
  (``"1 RB, 1 TE, 3 WR"``; ``OL`` omitted when 5, ``QB`` omitted when 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.privileged_tracking.nfl.tracking_features import parse_personnel

OTHER = "other"
OFF_GROUPS = ("QB", "OL", "RB", "TE", "WR")
DEF_GROUPS = ("DL", "LB", "DB")
# roster ``position`` -> candidate position group and side
ROSTER_POSITION_GROUP: dict[str, str] = {
    "QB": "QB", "OL": "OL", "T": "OL", "G": "OL", "C": "OL", "RB": "RB", "FB": "RB", "HB": "RB",
    "TE": "TE", "WR": "WR", "DL": "DL", "DE": "DL", "DT": "DL", "NT": "DL", "LB": "LB",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "DB": "DB", "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB",
}
OFFENSE_SIDE_GROUPS = frozenset(OFF_GROUPS)
DEFENSE_SIDE_GROUPS = frozenset(DEF_GROUPS)


@dataclass(frozen=True)
class TendencyConfig:
    """Shrinkage pseudo-counts (in plays) for the hierarchical tendency features.

    Attributes:
        alpha_league: pseudo-count towards uniform for the league-wide running share.
        alpha_prev: pseudo-count towards the league share for a team's previous-season share.
        alpha_team: pseudo-count towards the previous-season share for the in-season share.
        alpha_bucket: pseudo-count towards the in-season team share for bucket-conditional shares.
        alpha_cond: pseudo-count for shares conditional on the opponent's grouping.
    """

    alpha_league: float = 10.0
    alpha_prev: float = 100.0
    alpha_team: float = 100.0
    alpha_bucket: float = 50.0
    alpha_cond: float = 50.0


# ---------------------------------------------------------------------------
# Class collapsing and buckets
# ---------------------------------------------------------------------------

def top_classes(labels: pd.Series, coverage: float = 0.95, min_classes: int = 2) -> list[str]:
    """Most frequent labels whose cumulative share first reaches ``coverage``.

    Args:
        labels: categorical labels ``[n]`` (NaN ignored).
        coverage: cumulative share to reach (inclusive of the class that crosses it).
        min_classes: never return fewer classes than this.

    Returns:
        Labels in descending frequency order (``OTHER`` is not appended).
    """
    vc = labels.dropna().value_counts()
    if vc.empty:
        return []
    cum = vc.cumsum() / vc.sum()
    k = int(np.searchsorted(cum.to_numpy(), coverage, side="left")) + 1
    k = max(min_classes, min(k, len(vc)))
    return [str(x) for x in vc.index[:k]]


def slug(label: str) -> str:
    """Feature-name-safe class label: ``"1 RB, 1 TE, 3 WR"`` -> ``"1RB_1TE_3WR"``."""
    return "".join(ch if ch.isalnum() or ch == "_" else ("_" if ch == "," else "") for ch in str(label)).strip("_")


def collapse_classes(labels: pd.Series, keep: list[str]) -> pd.Series:
    """Map labels outside ``keep`` to ``OTHER``; NaN stays NaN (an unlabelled play is not a class)."""
    keep_set = set(keep)
    out = labels.astype(object).map(lambda x: x if x in keep_set else OTHER)
    return out.where(labels.notna(), np.nan).astype(object)


def distance_bucket(ydstogo: np.ndarray) -> np.ndarray:
    """Short (1-3) / mid (4-7) / long (8+) distance buckets; NaN -> ``"na"``."""
    y = np.asarray(ydstogo, dtype=float)
    out = np.full(len(y), "na", dtype=object)
    out[(y >= 0) & (y <= 3)] = "short"
    out[(y > 3) & (y <= 7)] = "mid"
    out[y > 7] = "long"
    return out


def down_distance_bucket(down: np.ndarray, ydstogo: np.ndarray) -> np.ndarray:
    """Down x distance bucket label, e.g. ``"d1_long"``; missing down -> ``"dna"``."""
    d = np.asarray(down, dtype=float)
    dist = distance_bucket(ydstogo)
    dl = np.where(np.isnan(d), "dna", np.char.add("d", np.nan_to_num(d, nan=0).astype(int).astype(str)))
    return np.array([f"{a}_{b}" for a, b in zip(dl, dist)], dtype=object)


# ---------------------------------------------------------------------------
# Strictly-prior tendency counts (the leakage-critical part)
# ---------------------------------------------------------------------------

def _onehot(labels: pd.Series, classes: list[str]) -> np.ndarray:
    """``[n, C]`` 0/1 matrix; labels outside ``classes`` (or NaN) give an all-zero row."""
    index = {c: i for i, c in enumerate(classes)}
    codes = np.array([index.get(x, -1) for x in labels.astype(object).to_numpy()], dtype=np.int64)
    eye = np.eye(len(classes), dtype=np.int64)
    out = np.zeros((len(labels), len(classes)), dtype=np.int64)
    ok = codes >= 0
    out[ok] = eye[codes[ok]]
    return out


def prior_game_class_counts(plays: pd.DataFrame, entity_cols: list[str], class_col: str,
                            classes: list[str], game_col: str = "game_id",
                            season_col: str = "season", week_col: str = "week"
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Per-play class counts over the entity's strictly earlier games in the same season.

    Games are ordered by ``week`` within ``(entity, season)``; every play of a game sees
    the cumulative counts of all games of that entity with a smaller week and nothing
    from its own game.

    Args:
        plays: one row per play with ``entity_cols``, ``class_col``, game/season/week.
        entity_cols: e.g. ``["posteam"]`` or ``["posteam", "dd_bucket"]``.
        classes: class order of the output columns.

    Returns:
        ``(counts [n, C], n_prior [n])`` aligned to ``plays`` row order.
    """
    key = list(entity_cols) + [season_col, game_col]
    oh = pd.DataFrame(_onehot(plays[class_col], classes), columns=list(classes), index=plays.index)
    tmp = pd.concat([plays[key + [week_col]].reset_index(drop=True), oh.reset_index(drop=True)], axis=1)
    grouped = tmp.groupby(key, sort=False, dropna=False)
    g = pd.concat([grouped[list(classes)].sum(), grouped[week_col].min()], axis=1).reset_index()
    ent = list(entity_cols) + [season_col]
    g = g.sort_values(ent + [week_col, game_col], kind="mergesort")
    cum = g.groupby(ent, sort=False, dropna=False)[list(classes)].cumsum() - g[list(classes)]
    cum = pd.concat([g[key].reset_index(drop=True), cum.reset_index(drop=True)], axis=1)
    out = plays[key].reset_index(drop=True).merge(cum, on=key, how="left")
    counts = out[list(classes)].to_numpy(dtype=float)
    counts = np.nan_to_num(counts, nan=0.0)
    return counts, counts.sum(axis=1)


def prior_game_value_sums(plays: pd.DataFrame, entity_cols: list[str], value_col: str,
                          game_col: str = "game_id", season_col: str = "season",
                          week_col: str = "week") -> tuple[np.ndarray, np.ndarray]:
    """Per-play ``(sum, count)`` of a numeric column over strictly earlier games (NaN skipped)."""
    v = pd.to_numeric(plays[value_col], errors="coerce").to_numpy(dtype=float)
    tmp = plays[list(entity_cols) + [season_col, game_col, week_col]].reset_index(drop=True).copy()
    tmp["_v"] = np.nan_to_num(v, nan=0.0)
    tmp["_c"] = (~np.isnan(v)).astype(float)
    key = list(entity_cols) + [season_col, game_col]
    g = tmp.groupby(key, sort=False, dropna=False).agg({"_v": "sum", "_c": "sum", week_col: "min"}).reset_index()
    ent = list(entity_cols) + [season_col]
    g = g.sort_values(ent + [week_col, game_col], kind="mergesort")
    cum = g.groupby(ent, sort=False, dropna=False)[["_v", "_c"]].cumsum() - g[["_v", "_c"]]
    cum = pd.concat([g[key].reset_index(drop=True), cum.reset_index(drop=True)], axis=1)
    out = plays[key].reset_index(drop=True).merge(cum, on=key, how="left")
    return (np.nan_to_num(out["_v"].to_numpy(float)), np.nan_to_num(out["_c"].to_numpy(float)))


def season_class_shares(plays: pd.DataFrame, entity_cols: list[str], class_col: str,
                        classes: list[str], season_col: str = "season") -> pd.DataFrame:
    """Full-season class counts per ``(entity, season)`` with an ``n`` column."""
    oh = pd.DataFrame(_onehot(plays[class_col], classes), columns=list(classes))
    tmp = pd.concat([plays[list(entity_cols) + [season_col]].reset_index(drop=True), oh], axis=1)
    g = tmp.groupby(list(entity_cols) + [season_col], sort=False, dropna=False)[list(classes)].sum()
    g["n"] = g[list(classes)].sum(axis=1)
    return g.reset_index()


def previous_season_counts(plays: pd.DataFrame, entity_cols: list[str], class_col: str,
                           classes: list[str], season_col: str = "season"
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-play class counts of the same entity over the whole previous season.

    Returns:
        ``(counts [n, C], n_prev [n])``; zeros when the entity has no previous season.
    """
    s = season_class_shares(plays, entity_cols, class_col, classes, season_col)
    s[season_col] = s[season_col] + 1
    key = list(entity_cols) + [season_col]
    out = plays[key].reset_index(drop=True).merge(s, on=key, how="left")
    counts = np.nan_to_num(out[list(classes)].to_numpy(float))
    return counts, counts.sum(axis=1)


def previous_season_value(plays: pd.DataFrame, entity_cols: list[str], value_col: str,
                          season_col: str = "season") -> tuple[np.ndarray, np.ndarray]:
    """Per-play ``(sum, count)`` of a numeric column over the entity's previous season."""
    v = pd.to_numeric(plays[value_col], errors="coerce")
    tmp = plays[list(entity_cols) + [season_col]].reset_index(drop=True).copy()
    tmp["_v"] = v.fillna(0.0).to_numpy()
    tmp["_c"] = v.notna().astype(float).to_numpy()
    g = tmp.groupby(list(entity_cols) + [season_col], sort=False, dropna=False)[["_v", "_c"]].sum().reset_index()
    g[season_col] = g[season_col] + 1
    key = list(entity_cols) + [season_col]
    out = plays[key].reset_index(drop=True).merge(g, on=key, how="left")
    return np.nan_to_num(out["_v"].to_numpy(float)), np.nan_to_num(out["_c"].to_numpy(float))


def league_prior_shares(plays: pd.DataFrame, class_col: str, classes: list[str],
                        alpha: float, season_col: str = "season", week_col: str = "week"
                        ) -> np.ndarray:
    """League-wide class share over all plays in strictly earlier ``(season, week)`` slots.

    Shrunk towards uniform with ``alpha`` pseudo-counts so week 1 of the first season is
    uniform rather than undefined. Returns ``[n, C]`` aligned to ``plays``.
    """
    oh = pd.DataFrame(_onehot(plays[class_col], classes), columns=list(classes))
    tmp = pd.concat([plays[[season_col, week_col]].reset_index(drop=True), oh], axis=1)
    g = tmp.groupby([season_col, week_col], sort=True)[list(classes)].sum()
    cum = g.cumsum() - g
    out = plays[[season_col, week_col]].reset_index(drop=True).merge(cum.reset_index(), on=[season_col, week_col], how="left")
    counts = np.nan_to_num(out[list(classes)].to_numpy(float))
    return shrink_shares(counts, counts.sum(axis=1), np.full(counts.shape, 1.0 / len(classes)), alpha)


def shrink_shares(counts: np.ndarray, n: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Dirichlet-style shrinkage: ``(counts + alpha * prior) / (n + alpha)``.

    Args:
        counts: ``[n, C]`` observed counts.
        n: ``[n]`` total observed count per row.
        prior: ``[n, C]`` prior shares (rows sum to 1).
        alpha: pseudo-count weight of the prior.
    """
    counts = np.asarray(counts, dtype=float)
    n = np.asarray(n, dtype=float)
    return (counts + alpha * np.asarray(prior, dtype=float)) / (n + alpha)[:, None]


def shrink_mean(total: np.ndarray, count: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Shrunk mean ``(total + alpha * prior) / (count + alpha)`` for numeric tendencies."""
    return (np.asarray(total, float) + alpha * np.asarray(prior, float)) / (np.asarray(count, float) + alpha)


def conditional_from_joint(joint_counts: np.ndarray, cond_idx: np.ndarray, n_cond: int,
                           n_target: int, prior: np.ndarray, alpha: float) -> np.ndarray:
    """Target-class shares conditional on a given class of the conditioning variable.

    ``joint_counts`` holds counts for joint classes ordered ``cond * n_target + target``.
    For each row the block of its conditioning class ``cond_idx[i]`` is selected and
    shrunk towards ``prior`` (e.g. the unconditional shrunk share).

    Args:
        joint_counts: ``[n, n_cond * n_target]``.
        cond_idx: ``[n]`` conditioning class index per row (``-1`` -> prior only).
        prior: ``[n, n_target]``.

    Returns:
        ``[n, n_target]`` conditional shares.
    """
    jc = np.asarray(joint_counts, dtype=float).reshape(len(cond_idx), n_cond, n_target)
    idx = np.asarray(cond_idx)
    safe = np.where(idx < 0, 0, idx)
    block = jc[np.arange(len(idx)), safe]
    block[idx < 0] = 0.0
    return shrink_shares(block, block.sum(axis=1), prior, alpha)


def joint_label(a: pd.Series, b: pd.Series) -> pd.Series:
    """``"a|b"`` joint class label for two categorical series."""
    return (a.astype(object).astype(str) + "|" + b.astype(object).astype(str)).astype(object)


def joint_classes(a_classes: list[str], b_classes: list[str]) -> list[str]:
    """Joint class order matching :func:`conditional_from_joint` (``a`` major)."""
    return [f"{a}|{b}" for a in a_classes for b in b_classes]


# ---------------------------------------------------------------------------
# Slots and lineup decoding
# ---------------------------------------------------------------------------

def offense_slots(personnel: str | None) -> dict[str, int]:
    """Slots per offensive position group implied by an NGS personnel string.

    ``"1 RB, 1 TE, 3 WR"`` -> ``{QB: 1, OL: 5, RB: 1, TE: 1, WR: 3}``. Unknown input
    returns the 11-personnel default. Extra groups (``DL`` on offense) are folded into
    ``TE`` slots so that slots always sum to 11 when the string is consistent.
    """
    c = parse_personnel(personnel)
    if not c:
        c = {"RB": 1, "TE": 1, "WR": 3}
    slots = {"QB": c.get("QB", 1), "OL": c.get("OL", 5), "RB": c.get("RB", 0),
             "TE": c.get("TE", 0), "WR": c.get("WR", 0)}
    extra = sum(v for k, v in c.items() if k not in slots)
    slots["TE"] += extra
    return slots


def defense_slots(personnel: str | None) -> dict[str, int]:
    """Slots per defensive position group; unknown input -> nickel default (4 DL, 2 LB, 5 DB)."""
    c = parse_personnel(personnel)
    if not c:
        return {"DL": 4, "LB": 2, "DB": 5}
    slots = {"DL": c.get("DL", 0), "LB": c.get("LB", 0), "DB": c.get("DB", 0)}
    extra = sum(v for k, v in c.items() if k not in slots)
    slots["DL"] += extra
    return slots


def slots_frame(personnel: pd.Series, side: str) -> pd.DataFrame:
    """Vectorised slots per row: columns ``slot_<group>`` (``[n, groups]``)."""
    fn = offense_slots if side == "offense" else defense_slots
    groups = OFF_GROUPS if side == "offense" else DEF_GROUPS
    cache: dict[str, dict[str, int]] = {}
    rows = []
    for s in personnel.astype(object):
        k = str(s)
        if k not in cache:
            cache[k] = fn(None if s is None or (isinstance(s, float) and np.isnan(s)) else s)
        rows.append([cache[k][g] for g in groups])
    return pd.DataFrame(rows, columns=[f"slot_{g}" for g in groups], index=personnel.index)


def usage_rank(df: pd.DataFrame, keys: list[str], usage_col: str, tiebreak_col: str) -> np.ndarray:
    """1-based rank within ``keys`` by ``usage_col`` descending, ties by ``tiebreak_col`` ascending.

    NaN usage ranks last. Returns ``[n]`` int array aligned to ``df``.
    """
    u = pd.to_numeric(df[usage_col], errors="coerce").fillna(-1.0)
    t = pd.to_numeric(df[tiebreak_col], errors="coerce").fillna(99.0)
    tmp = pd.DataFrame({"_u": -u.to_numpy(), "_t": t.to_numpy()}, index=df.index)
    for k in keys:
        tmp[k] = df[k].to_numpy()
    order = tmp.sort_values(keys + ["_u", "_t"], kind="mergesort")
    rank = order.groupby(keys, sort=False, dropna=False).cumcount().to_numpy() + 1
    out = np.empty(len(df), dtype=int)
    out[df.index.get_indexer(order.index)] = rank
    return out


def decode_lineup(cands: pd.DataFrame, play_cols: list[str], group_col: str, score_col: str,
                  slot_col: str) -> np.ndarray:
    """Pick the top ``slot`` candidates per (play, group) by score.

    Args:
        cands: one row per (play, candidate) with the group, score and the number of
            slots that group has on that play (``slot_col``, identical within a group).

    Returns:
        boolean ``[n]`` selection aligned to ``cands``.
    """
    keys = list(play_cols) + [group_col]
    tmp = pd.DataFrame({"_s": -pd.to_numeric(cands[score_col], errors="coerce").fillna(-1.0).to_numpy()},
                       index=cands.index)
    for k in keys:
        tmp[k] = cands[k].to_numpy()
    order = tmp.sort_values(keys + ["_s"], kind="mergesort")
    rank = order.groupby(keys, sort=False, dropna=False).cumcount().to_numpy()
    sel = np.zeros(len(cands), dtype=bool)
    pos = cands.index.get_indexer(order.index)
    slots = pd.to_numeric(cands[slot_col], errors="coerce").fillna(0).to_numpy()[pos]
    sel[pos] = rank < slots
    return sel


def roster_group(position: str | None, depth_chart_position: str | None = None) -> str:
    """Candidate position group from roster ``position`` (falls back to depth-chart position)."""
    for p in (position, depth_chart_position):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        g = ROSTER_POSITION_GROUP.get(str(p).upper())
        if g is not None:
            return g
    return "ST"


def report_subgroup(group: str, depth_chart_position: str | None, rank: int) -> str:
    """Finer reporting bucket: WR1-2 vs WR3+, TE1 vs TE2+, RB1 vs RB2+, CB1-2 vs CB3+, S."""
    dcp = str(depth_chart_position).upper() if depth_chart_position is not None else ""
    if group == "WR":
        return "WR1-2" if rank <= 2 else "WR3+"
    if group == "TE":
        return "TE1" if rank <= 1 else "TE2+"
    if group == "RB":
        return "RB1" if rank <= 1 else "RB2+"
    if group == "DB":
        if dcp in ("FS", "SS", "S"):
            return "S"
        return "CB1-2" if rank <= 2 else "CB3+ (nickel/dime)"
    return group


@dataclass
class FeatureSets:
    """Named nested feature lists used by the driver (kept here so tests can import them)."""

    s0: list[str] = field(default_factory=lambda: [
        "down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "score_differential",
        "wp", "goal_to_go", "season", "week", "is_home"])
    s1_extra: list[str] = field(default_factory=lambda: ["shotgun", "no_huddle"])
    play_type_extra: list[str] = field(default_factory=lambda: ["is_pass", "is_rush", "qb_dropback"])
