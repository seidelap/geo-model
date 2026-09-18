"""NFL stage 02 (c): player-level participation ("is this roster player on the field?").

For every play the candidate set is the team's weekly roster rows with status ``ACT``,
restricted to the side's position groups (QB/OL/RB/TE/WR for the offense, DL/LB/DB for
the defense). In the nflverse weekly roster ``ACT`` is exactly the 48-man game-day active
list and ``INA`` the declared inactives (about 6 per team-week, none of whom takes a
snap); both are fixed ~90 minutes before kickoff, so the candidate set is pre-play but
not mid-week information. Per-candidate features use only pre-kickoff information: position group, depth-chart rank
that week, snap shares from strictly earlier games (season-to-date, last three games,
previous season), weeks since the player last took a snap, a prior-usage rank within the
team-week position group, the play's personnel grouping (true, and separately imputed by
stages (a)/(b)) and how many slots that grouping gives the candidate's group, plus the
play situation. A LightGBM binary model is trained on 2021 plays (rounds chosen on a
game-held-out fifth of 2021) and tested on 2022; "slot decoding" then fills each position
group's slots with the top-ranked candidates and is scored on exact-lineup and per-slot
accuracy, with true and with imputed groupings.

Called from :mod:`participation` (``--stages c``); the public entry point is
:func:`run_players`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.privileged_tracking.common.io import nfl_dir
from research.privileged_tracking.common.report import md_table
from research.privileged_tracking.common.splits import group_kfold
from research.privileged_tracking.nfl import participation_features as pf

if TYPE_CHECKING:  # pragma: no cover
    from research.privileged_tracking.nfl.participation import ClassSpec, ParticipationConfig

SITUATION = ["down", "ydstogo", "yardline_100", "qtr", "half_seconds_remaining", "score_differential",
             "wp", "goal_to_go", "shotgun", "no_huddle"]
DCP_VOCAB = ["QB", "T", "G", "C", "OL", "RB", "FB", "TE", "WR", "DE", "DT", "NT", "DL", "OLB", "ILB", "MLB",
             "LB", "CB", "FS", "SS", "S", "DB"]
GROUP_VOCAB = list(pf.OFF_GROUPS) + list(pf.DEF_GROUPS)
SIDES = ("offense", "defense")
SUBGROUP_ORDER = ["QB", "OL", "RB1", "RB2+", "WR1-2", "WR3+", "TE1", "TE2+", "DL", "LB", "CB1-2",
                  "CB3+ (nickel/dime)", "S"]


@dataclass
class PlayerTables:
    """Pre-game per-player tables for one season."""

    roster: pd.DataFrame      # team, week, gsis_id, position, dcp, group
    depth: pd.DataFrame       # team, week, side, gsis_id, depth_rank
    snaps: pd.DataFrame       # gsis_id, week, off_std, off_last3, off_prev, off_gp_prev, off_since, def_*


# ---------------------------------------------------------------------------
# Loading pre-game tables
# ---------------------------------------------------------------------------

def load_roster_status(season: int) -> pd.DataFrame:
    """All regular-season weekly roster rows: ``team, week, gsis_id, status, position, depth_chart_position``.

    ``status == "ACT"`` is the 48-man game-day active list, ``"INA"`` the declared inactives
    (both known ~90 minutes before kickoff); ``RES`` / ``DEV`` / ``CUT`` / ... never play.
    """
    d = nfl_dir() / "nflverse"
    r = pd.read_parquet(d / f"roster_weekly_{season}.parquet",
                        columns=["team", "week", "gsis_id", "status", "position", "depth_chart_position", "game_type"])
    r = r[(r["game_type"] == "REG") & r["gsis_id"].notna()]
    return r.drop(columns="game_type").drop_duplicates(["team", "week", "gsis_id"]).reset_index(drop=True)


def load_roster(season: int) -> pd.DataFrame:
    """Candidate roster: the ``ACT`` (game-day active) rows with position group codes, K/P/LS removed."""
    r = load_roster_status(season)
    r = r[r["status"] == "ACT"]
    r = r.drop_duplicates(["team", "week", "gsis_id"]).copy()
    r["group"] = [pf.roster_group(a, b) for a, b in zip(r["position"].astype(object), r["depth_chart_position"].astype(object))]
    r = r[r["group"] != "ST"]
    r["dcp_code"] = pd.Categorical(r["depth_chart_position"].astype(object), categories=DCP_VOCAB).codes.astype(float)
    r["group_code"] = pd.Categorical(r["group"], categories=GROUP_VOCAB).codes.astype(float)
    return r[["team", "week", "gsis_id", "depth_chart_position", "group", "dcp_code", "group_code"]].reset_index(drop=True)


def load_depth(season: int) -> pd.DataFrame:
    d = nfl_dir() / "nflverse"
    dc = pd.read_parquet(d / f"depth_charts_{season}.parquet",
                         columns=["club_code", "week", "game_type", "formation", "depth_team", "gsis_id"])
    dc = dc[(dc["game_type"] == "REG") & dc["formation"].isin(["Offense", "Defense"]) & dc["gsis_id"].notna()].copy()
    dc["side"] = dc["formation"].map({"Offense": "offense", "Defense": "defense"})
    dc["depth_rank"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    dc["week"] = dc["week"].astype(int)
    g = dc.groupby(["club_code", "week", "side", "gsis_id"], as_index=False)["depth_rank"].min()
    return g.rename(columns={"club_code": "team"})


def load_snap_pcts(season: int, players: pd.DataFrame) -> pd.DataFrame:
    """Per (gsis_id, week) offense / defense snap shares for the regular season."""
    d = nfl_dir() / "nflverse"
    sc = pd.read_parquet(d / f"snap_counts_{season}.parquet",
                         columns=["game_type", "week", "pfr_player_id", "offense_pct", "defense_pct", "offense_snaps", "defense_snaps"])
    sc = sc[sc["game_type"] == "REG"].merge(players, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sc = sc.groupby(["gsis_id", "week"], as_index=False)[["offense_pct", "defense_pct"]].max()
    return sc


def snap_history(cur: pd.DataFrame, prev: pd.DataFrame, n_weeks: int = 18) -> pd.DataFrame:
    """Strictly-prior snap-share features per (gsis_id, week) for weeks 1..``n_weeks``.

    Args:
        cur: this season's ``(gsis_id, week, offense_pct, defense_pct)``.
        prev: previous season's table (same columns), may be empty.

    Returns:
        One row per player in ``cur`` or ``prev`` and week with, per side ``s`` in
        ``{off, def}``: ``s_std`` season-to-date mean share over earlier games with a snap
        record, ``s_last3`` mean over the last three such games, ``s_prev`` previous-season
        mean share, ``s_gp_prev`` previous-season games, ``s_since`` weeks since the last
        game with a positive share this season (``n_weeks + 1`` if none).
    """
    prev_agg = (prev.groupby("gsis_id").agg(off_prev=("offense_pct", "mean"), def_prev=("defense_pct", "mean"),
                                            gp_prev=("week", "size")) if len(prev) else
                pd.DataFrame(columns=["off_prev", "def_prev", "gp_prev"]))
    ids = sorted(set(cur["gsis_id"]) | set(prev_agg.index))
    idx = {g: i for i, g in enumerate(ids)}
    W = n_weeks
    mats = {}
    for side, col in (("off", "offense_pct"), ("def", "defense_pct")):
        M = np.full((len(ids), W), np.nan)
        rows = cur["gsis_id"].map(idx).to_numpy()
        wk = cur["week"].to_numpy(int) - 1
        ok = (wk >= 0) & (wk < W)
        M[rows[ok], wk[ok]] = cur[col].to_numpy(float)[ok]
        mats[side] = M
    out: dict[str, np.ndarray] = {}
    gs, ws = np.meshgrid(np.arange(len(ids)), np.arange(1, W + 1), indexing="ij")
    out["gsis_id"] = np.array(ids, dtype=object)[gs.ravel()]
    out["week"] = ws.ravel()
    for side, M in mats.items():
        has = ~np.isnan(M)
        v = np.nan_to_num(M)
        cum_v = np.concatenate([np.zeros((len(ids), 1)), np.cumsum(v, axis=1)], axis=1)[:, :W]
        cum_c = np.concatenate([np.zeros((len(ids), 1)), np.cumsum(has, axis=1)], axis=1)[:, :W]
        std = np.where(cum_c > 0, cum_v / np.maximum(cum_c, 1), np.nan)
        last3 = np.full((len(ids), W), np.nan)
        since = np.full((len(ids), W), float(W + 1))
        for w in range(1, W):
            vals = M[:, :w]
            hv = has[:, :w]
            # last three games with a record before week w+1
            cnt = np.cumsum(hv[:, ::-1], axis=1)[:, ::-1]  # games with record from column j onward
            take = hv & (cnt <= 3)
            s = (np.nan_to_num(vals) * take).sum(axis=1)
            c = take.sum(axis=1)
            last3[:, w] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            pos = (np.nan_to_num(vals) > 0)
            last_pos = np.where(pos.any(axis=1), w - np.argmax(pos[:, ::-1], axis=1) - 1, -1)
            since[:, w] = np.where(last_pos >= 0, (w + 1) - (last_pos + 1), float(W + 1))
        out[f"{side}_std"] = std.ravel()
        out[f"{side}_last3"] = last3.ravel()
        out[f"{side}_since"] = since.ravel()
    df = pd.DataFrame(out)
    df = df.merge(prev_agg.reset_index(), on="gsis_id", how="left")
    df["gp_prev"] = df["gp_prev"].fillna(0.0)
    return df


def build_player_tables(season: int) -> PlayerTables:
    players = pd.read_parquet(nfl_dir() / "nflverse" / "players.parquet", columns=["gsis_id", "pfr_id"]).dropna()
    cur = load_snap_pcts(season, players)
    prev = load_snap_pcts(season - 1, players)
    return PlayerTables(roster=load_roster(season), depth=load_depth(season), snaps=snap_history(cur, prev))


# ---------------------------------------------------------------------------
# Candidate rows
# ---------------------------------------------------------------------------

def _imputed_slots(P: np.ndarray, classes: list[str], side: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Argmax class (``other`` -> most probable specific class), its slots and expected slots."""
    P = np.nan_to_num(np.asarray(P, float))
    specific = np.array([c != pf.OTHER for c in classes])
    P_spec = np.where(specific[None, :], P, -1.0)
    arg = P_spec.argmax(axis=1)
    labels = pd.Series(np.array(classes, dtype=object)[arg])
    slots_by_class = pf.slots_frame(pd.Series(classes, dtype=object), side)
    slots_by_class.loc[[c == pf.OTHER for c in classes], :] = slots_by_class.loc[[c != pf.OTHER for c in classes], :].iloc[0].to_numpy()
    hard = pf.slots_frame(labels, side)
    expected = pd.DataFrame(P @ slots_by_class.to_numpy(float), columns=slots_by_class.columns)
    return arg, hard, expected


def candidate_rows(plays: pd.DataFrame, tabs: PlayerTables, side: str, spec: ClassSpec,
                   imp: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, roster candidate) for one side with features and label.

    Args:
        plays: play rows (one season) with situation, personnel strings and participant ids.
        imp: imputed offense (``p_off_*``) and defense (``p_def_imp_*``) probabilities aligned to ``plays``.
    """
    team_col = "posteam" if side == "offense" else "defteam"
    pers_col = "offense_personnel" if side == "offense" else "defense_personnel"
    part_col = "offense_players" if side == "offense" else "defense_players"
    groups = pf.OFF_GROUPS if side == "offense" else pf.DEF_GROUPS
    classes = spec.off_classes if side == "offense" else spec.def_classes
    pcols = [f"{'p_off' if side == 'offense' else 'p_def_imp'}_{pf.slug(c)}" for c in classes]
    P = imp[pcols].to_numpy(float)
    keep = ~np.isnan(P).any(axis=1)
    plays = plays.loc[keep].reset_index(drop=True)
    P = P[keep]
    true_slots = pf.slots_frame(plays[pers_col], side).reset_index(drop=True)
    _, hard, expected = _imputed_slots(P, classes, side)
    pl = plays[["game_id", "play_id", "season", "week", team_col, part_col] + SITUATION].copy()
    pl = pl.rename(columns={team_col: "team"})
    pl["imp_grp_ok"] = (hard.to_numpy() == true_slots.to_numpy()).all(axis=1)
    for g in groups:
        pl[f"slot_true_{g}"] = true_slots[f"slot_{g}"].to_numpy()
        pl[f"slot_imp_{g}"] = hard[f"slot_{g}"].to_numpy()
        pl[f"slot_exp_{g}"] = expected[f"slot_{g}"].to_numpy()
    ros = tabs.roster[tabs.roster["group"].isin(groups)]
    cand = pl.merge(ros, on=["team", "week"], how="inner")
    cand = cand.merge(tabs.depth[tabs.depth["side"] == side].drop(columns="side"), on=["team", "week", "gsis_id"], how="left")
    cand["depth_rank"] = cand["depth_rank"].fillna(4.0)
    s = "off" if side == "offense" else "def"
    sn = tabs.snaps[["gsis_id", "week", f"{s}_std", f"{s}_last3", f"{s}_since", f"{s}_prev", "gp_prev"]].rename(columns={
        f"{s}_std": "snap_std", f"{s}_last3": "snap_last3", f"{s}_since": "weeks_since", f"{s}_prev": "snap_prev"})
    cand = cand.merge(sn, on=["gsis_id", "week"], how="left")
    cand["weeks_since"] = cand["weeks_since"].fillna(19.0)
    cand["gp_prev"] = cand["gp_prev"].fillna(0.0)
    cand["usage"] = cand["snap_std"].where(cand["snap_std"].notna(), cand["snap_prev"])
    cand["usage_rank"] = pf.usage_rank(cand.assign(_w=cand["week"]), ["team", "_w", "group"], "usage", "depth_rank").astype(float)
    # candidate-level rank must be within the team-week, not per play: recompute on unique team-week-players
    tw = cand.drop_duplicates(["team", "week", "gsis_id"])[["team", "week", "gsis_id", "group", "usage", "depth_rank"]].copy()
    tw["usage_rank"] = pf.usage_rank(tw, ["team", "week", "group"], "usage", "depth_rank").astype(float)
    tw["group_size"] = tw.groupby(["team", "week", "group"])["gsis_id"].transform("size").astype(float)
    cand = cand.drop(columns=["usage_rank"]).merge(tw[["team", "week", "gsis_id", "usage_rank", "group_size"]],
                                                    on=["team", "week", "gsis_id"], how="left")
    # slots for the candidate's own group
    gidx = cand["group"].to_numpy(object)
    for kind in ("true", "imp", "exp"):
        cand[f"slot_{kind}"] = np.stack([cand[f"slot_{kind}_{g}"].to_numpy(float) for g in groups], axis=1)[
            np.arange(len(cand)), pd.Categorical(gidx, categories=list(groups)).codes]
    cand["slot_minus_rank_true"] = cand["slot_true"] - cand["usage_rank"]
    cand["slot_minus_rank_imp"] = cand["slot_imp"] - cand["usage_rank"]
    cand["slot_minus_rank_exp"] = cand["slot_exp"] - cand["usage_rank"]
    # label
    part = pl[["game_id", "play_id", part_col]].copy()
    part["gsis_id"] = part[part_col].astype(object).str.split(";")
    part = part.explode("gsis_id")
    part = part[part["gsis_id"].notna() & (part["gsis_id"] != "")]
    part["y"] = 1.0
    cand = cand.merge(part[["game_id", "play_id", "gsis_id", "y"]], on=["game_id", "play_id", "gsis_id"], how="left")
    cand["y"] = cand["y"].fillna(0.0)
    cand["side"] = side
    cand["subgroup"] = [pf.report_subgroup(g, d, int(r)) for g, d, r in
                        zip(cand["group"], cand["depth_chart_position"].astype(object), cand["usage_rank"])]
    n_true = part.groupby(["game_id", "play_id"]).size().rename("n_true")
    n_cov = cand[cand["y"] == 1].groupby(["game_id", "play_id"]).size().rename("n_covered")
    cov = pd.concat([n_true, n_cov], axis=1).fillna(0)
    cand = cand.merge(cov.reset_index(), on=["game_id", "play_id"], how="left")
    return cand.drop(columns=[part_col])


# ---------------------------------------------------------------------------
# Modelling and decoding
# ---------------------------------------------------------------------------

def feature_list(variant: str) -> list[str]:
    base = ["group_code", "dcp_code", "depth_rank", "usage_rank", "group_size", "snap_std", "snap_last3",
            "snap_prev", "gp_prev", "weeks_since"] + SITUATION
    if variant == "true":
        return base + ["slot_true", "slot_minus_rank_true"]
    return base + ["slot_imp", "slot_exp", "slot_minus_rank_imp", "slot_minus_rank_exp"]


def fit_binary(tr: pd.DataFrame, feats: list[str], cfg: ParticipationConfig) -> tuple[lgb.Booster, int]:
    """Fit with rounds chosen on a game-held-out fifth of the training plays."""
    folds = list(group_kfold(tr["game_id"].to_numpy(), n_splits=5, seed=cfg.seed))
    f_tr, f_va = folds[0]
    params = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "num_threads": cfg.n_jobs, "seed": cfg.seed, "verbose": -1, "max_bin": 63}
    cat = ["group_code", "dcp_code"]
    dtr = lgb.Dataset(tr.iloc[f_tr][feats], label=tr.iloc[f_tr]["y"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(tr.iloc[f_va][feats], label=tr.iloc[f_va]["y"], reference=dtr, categorical_feature=cat, free_raw_data=False)
    b = lgb.train(params, dtr, num_boost_round=cfg.max_rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])
    it = int(b.best_iteration or cfg.max_rounds)
    dall = lgb.Dataset(tr[feats], label=tr["y"], categorical_feature=cat, free_raw_data=False)
    return lgb.train(params, dall, num_boost_round=it), it


def decode_and_score(te: pd.DataFrame, score_col: str, slot_kind: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slot-decode ``score_col`` with ``slot_<slot_kind>`` slots; return per-subgroup, per-play and error tables."""
    sel = pf.decode_lineup(te, ["game_id", "play_id", "side"], "group", score_col, f"slot_{slot_kind}")
    te = te.assign(pred=sel.astype(float))
    rows = []
    for sg, d in te.groupby("subgroup", sort=False):
        tp = float(((d["pred"] == 1) & (d["y"] == 1)).sum())
        fp = float(((d["pred"] == 1) & (d["y"] == 0)).sum())
        fn = float(((d["pred"] == 0) & (d["y"] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) else float("nan")
        try:
            auc = float(roc_auc_score(d["y"], d[score_col])) if d["y"].nunique() == 2 else float("nan")
        except ValueError:
            auc = float("nan")
        rows.append({"decoder": label, "subgroup": sg, "n_candidates": int(len(d)), "n_on_field": int(d["y"].sum()),
                     "on_field_rate": float(d["y"].mean()), "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                     "false_negatives": int(fn), "false_positives": int(fp)})
    per_sub = pd.DataFrame(rows)
    per_sub["subgroup"] = pd.Categorical(per_sub["subgroup"], categories=SUBGROUP_ORDER + sorted(set(per_sub["subgroup"]) - set(SUBGROUP_ORDER)))
    per_sub = per_sub.sort_values("subgroup").reset_index(drop=True)
    per_sub["subgroup"] = per_sub["subgroup"].astype(str)
    keys = ["game_id", "play_id", "side"]
    per_play = te.groupby(keys, sort=False).agg(n_true=("n_true", "first"), imp_grp_ok=("imp_grp_ok", "first")).reset_index()
    hits = te[(te["pred"] == 1) & (te["y"] == 1)].groupby(keys).size().rename("hits").reset_index()
    per_play = per_play.merge(hits, on=keys, how="left")
    per_play["hits"] = per_play["hits"].fillna(0.0)
    per_play["per_slot_acc"] = per_play["hits"] / per_play["n_true"].clip(lower=1)
    per_play["exact"] = (per_play["hits"] == per_play["n_true"]).astype(float)
    per_play["decoder"] = label
    misses = te[(te["pred"] == 0) & (te["y"] == 1)].groupby("subgroup").size().rename("missed_players")
    errors = misses.reset_index()
    errors["share_of_misses"] = errors["missed_players"] / max(errors["missed_players"].sum(), 1)
    errors["decoder"] = label
    return per_sub, per_play, errors


def run_players(p: pd.DataFrame, imp: pd.DataFrame, spec: ClassSpec, cfg: ParticipationConfig) -> dict[str, Any]:
    """Stage (c) driver; returns ``{"tables": {...}, "text": markdown}``."""
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    imp = imp.set_index(["game_id", "play_id"])
    subsets: dict[int, pd.DataFrame] = {}
    for season, n_plays in ((cfg.val_season, cfg.players_train_plays), (cfg.test_season, cfg.players_test_plays)):
        q = p[(p["season"] == season) & (p["n_offense"] == 11) & (p["n_defense"] == 11)]
        if len(q) > n_plays:
            q = q.iloc[np.sort(rng.choice(len(q), n_plays, replace=False))]
        subsets[season] = q.reset_index(drop=True)
    cands: dict[int, pd.DataFrame] = {}
    coverage_rows = []
    for season, q in subsets.items():
        tabs = build_player_tables(season)
        qi = imp.loc[list(zip(q["game_id"], q["play_id"]))].reset_index(drop=True)
        parts = [candidate_rows(q, tabs, side, spec, qi) for side in SIDES]
        c = pd.concat(parts, ignore_index=True)
        cands[season] = c
        for side in SIDES:
            d = c[c["side"] == side].drop_duplicates(["game_id", "play_id"])
            coverage_rows.append({"season": season, "side": side, "plays": int(len(d)),
                                  "candidates_per_play": float(c[c["side"] == side].groupby(["game_id", "play_id"]).size().mean()),
                                  "true_players_in_candidates": float(d["n_covered"].sum() / d["n_true"].sum()),
                                  "imputed_grouping_slots_correct": float(d["imp_grp_ok"].mean())})
        print(f"  [c] candidates {season}: {len(c)} rows ({time.time()-t0:.0f}s)", flush=True)
    coverage = pd.DataFrame(coverage_rows)
    tr_all, te_all = cands[cfg.val_season], cands[cfg.test_season]
    per_sub_tabs, per_play_tabs, err_tabs, fit_rows = [], [], [], []
    te_all = te_all.copy()
    for side in SIDES:
        tr = tr_all[tr_all["side"] == side].reset_index(drop=True)
        te_mask = te_all["side"] == side
        for variant in ("true", "imp"):
            feats = feature_list(variant)
            b, it = fit_binary(tr, feats, cfg)
            te_all.loc[te_mask, f"score_{variant}"] = b.predict(te_all.loc[te_mask, feats])
            imp_tab = pd.DataFrame({"feature": feats, "gain": b.feature_importance("gain")})
            imp_tab["gain_share"] = imp_tab["gain"] / imp_tab["gain"].sum()
            fit_rows.append({"side": side, "variant": variant, "rounds": it, "n_train_rows": int(len(tr)),
                             "top_features": ", ".join(imp_tab.sort_values("gain", ascending=False)["feature"].head(5))})
            print(f"  [c] fit {side}/{variant}: rounds={it} ({time.time()-t0:.0f}s)", flush=True)
    te_all["score_usage"] = te_all["usage"].fillna(-1.0) - 0.001 * te_all["depth_rank"]
    te_all["score_depth"] = -te_all["depth_rank"] + 0.001 * te_all["usage"].fillna(-1.0)
    decoders = [("model | true grouping", "score_true", "true"), ("model | imputed grouping", "score_imp", "imp"),
                ("model(true feats) | imputed slots", "score_true", "imp"),
                ("baseline: prior usage rank | true grouping", "score_usage", "true"),
                ("baseline: depth chart rank | true grouping", "score_depth", "true"),
                ("baseline: prior usage rank | imputed grouping", "score_usage", "imp")]
    for label, score_col, slot_kind in decoders:
        ps, pp, er = decode_and_score(te_all, score_col, slot_kind, label)
        per_sub_tabs.append(ps)
        per_play_tabs.append(pp)
        err_tabs.append(er)
    per_sub = pd.concat(per_sub_tabs, ignore_index=True)
    per_play = pd.concat(per_play_tabs, ignore_index=True)
    errors = pd.concat(err_tabs, ignore_index=True)
    summary = (per_play.groupby(["decoder", "side"]).agg(plays=("exact", "size"), exact_lineup_acc=("exact", "mean"),
                                                         per_slot_acc=("per_slot_acc", "mean"),
                                                         mean_errors_per_play=("hits", lambda s: float((11 - s).mean())))
               .reset_index())
    cond = (per_play[per_play["decoder"] == "model | imputed grouping"]
            .groupby(["side", "imp_grp_ok"]).agg(plays=("exact", "size"), exact_lineup_acc=("exact", "mean"),
                                                 per_slot_acc=("per_slot_acc", "mean")).reset_index())
    qb_check = qb_determinism_check(te_all, cfg.test_season)
    tables = {"coverage": coverage, "fits": pd.DataFrame(fit_rows), "per_subgroup": per_sub, "summary": summary,
              "errors": errors, "conditional_on_grouping": cond, "qb_check": qb_check}
    text = _players_text(tables, cfg)
    print(f"  [c] done ({time.time()-t0:.0f}s)", flush=True)
    return {"tables": tables, "text": text}


def depth_chart_qb1(season: int) -> pd.DataFrame:
    """``team, week, gsis_id`` of every QB listed at depth 1 on the offense depth chart (REG weeks)."""
    d = nfl_dir() / "nflverse"
    dc = pd.read_parquet(d / f"depth_charts_{season}.parquet",
                         columns=["club_code", "week", "game_type", "formation", "depth_team", "position", "gsis_id"])
    dc = dc[(dc["game_type"] == "REG") & (dc["formation"] == "Offense") & (dc["position"] == "QB") & dc["gsis_id"].notna()]
    dc = dc[pd.to_numeric(dc["depth_team"], errors="coerce") == 1].copy()
    dc["week"] = dc["week"].astype(int)
    return dc.rename(columns={"club_code": "team"})[["team", "week", "gsis_id"]].drop_duplicates().reset_index(drop=True)


def players_with_a_snap(season: int) -> pd.DataFrame:
    """``(game_id, gsis_id)`` pairs of players with a snap-count row in that game (post-game information)."""
    d = nfl_dir() / "nflverse"
    sc = pd.read_parquet(d / f"snap_counts_{season}.parquet", columns=["game_id", "game_type", "pfr_player_id"])
    sc = sc[sc["game_type"] == "REG"]
    pl = pd.read_parquet(d / "players.parquet", columns=["gsis_id", "pfr_id"]).dropna()
    return sc.merge(pl, left_on="pfr_player_id", right_on="pfr_id")[["game_id", "gsis_id"]].drop_duplicates()


ACT_COND = "depth-1 QB is on the ACT list (a candidate)"
SNAP_COND = "  ... and took a snap that day (post-game split)"
NOSNAP_COND = "  ... and took no snap that day: dressed non-starter (post-game split)"
QB1_STATUS_LABELS = {
    "INA": "no depth-1 QB on the ACT list: listed QB1 declared inactive (INA)",
    "other": "no depth-1 QB on the ACT list: listed QB1 on a reserve / other list",
    "not_on_roster": "no depth-1 QB on the ACT list: listed QB1 not on that week's roster",
    "not_listed": "no depth-1 QB on the ACT list: no QB1 on the depth chart",
    "mismatch": "no depth-1 QB candidate although the listed QB1 is ACT (depth-chart / roster mismatch)",
}


def qb_check_table(qb_cands: pd.DataFrame, plays: pd.DataFrame, qb1_status: pd.DataFrame,
                   played: pd.DataFrame) -> pd.DataFrame:
    """Decompose how often the pre-kickoff QB1 is the QB on the field (pure).

    Args:
        qb_cands: QB candidate rows ``[m]`` (ACT roster only) with ``game_id, play_id, gsis_id,
            depth_rank, usage_rank, y``.
        plays: one row per offense play ``[n]`` with ``game_id, play_id, team, week``.
        qb1_status: depth-chart QB1 rows ``[k]`` with ``team, week, gsis_id, status`` (weekly
            roster status, NaN when the player is not on that week's roster).
        played: ``(game_id, gsis_id)`` pairs of players with at least one snap in that game
            (post-game information, used only to split the residual of the ACT rows).

    Returns:
        Rows ``candidate, condition, plays, share_of_plays, on_field_rate``. For the
        depth-chart candidate the ``ACT`` row plus the ``no depth-1 QB ...`` rows partition the
        plays (shares sum to 1) and the two ``...`` rows partition the ``ACT`` row; the
        prior-usage rows are a second view of the same plays.
    """
    n_plays = int(len(plays))
    keys = ["game_id", "play_id"]
    pl = played[["game_id", "gsis_id"]].drop_duplicates().assign(played_in_game=1.0)
    qb = qb_cands.merge(pl, on=["game_id", "gsis_id"], how="left")
    qb["played_in_game"] = qb["played_in_game"].fillna(0.0)
    rows: list[dict[str, Any]] = []

    def add(candidate: str, condition: str, n: int, rate: float) -> None:
        rows.append({"candidate": candidate, "condition": condition, "plays": int(n),
                     "share_of_plays": float(n / max(n_plays, 1)), "on_field_rate": rate})

    for label, mask in (("depth-chart QB1", qb["depth_rank"] == 1), ("prior-usage QB1", qb["usage_rank"] == 1)):
        s = qb[mask].drop_duplicates(keys)
        add(label, ACT_COND if label == "depth-chart QB1" else "all plays", len(s), float(s["y"].mean()) if len(s) else float("nan"))
        for cond, sub in ((SNAP_COND, s[s["played_in_game"] == 1]), (NOSNAP_COND, s[s["played_in_game"] == 0])):
            add(label, cond, len(sub), float(sub["y"].mean()) if len(sub) else float("nan"))
        if label == "depth-chart QB1":
            rest = plays.merge(s[keys].assign(_has=1), on=keys, how="left")
            rest = rest[rest["_has"].isna()][["game_id", "play_id", "team", "week"]]
            st = qb1_status.copy()
            st["_rank"] = np.select([st["status"] == "ACT", st["status"] == "INA", st["status"].isna()], [0, 1, 3], 2)
            best = st.groupby(["team", "week"], as_index=False)["_rank"].min()
            rest = rest.merge(best, on=["team", "week"], how="left")
            rest["_rank"] = rest["_rank"].fillna(4).astype(int)
            for r_, key in ((1, "INA"), (2, "other"), (3, "not_on_roster"), (4, "not_listed"), (0, "mismatch")):
                n = int((rest["_rank"] == r_).sum())
                if n:
                    add(label, QB1_STATUS_LABELS[key], n, float("nan"))
    return pd.DataFrame(rows)


def qb_determinism_check(te: pd.DataFrame, season: int) -> pd.DataFrame:
    """How often the pre-kickoff depth-1 / usage-1 QB is actually the QB on the field (test plays).

    Candidates are ACT roster players, i.e. the game-day active list, so a listed QB1 who was
    declared inactive (``INA``) is never a candidate; those plays are counted separately from
    the weekly roster status. Among plays with an ACT depth-1 QB the residual is split, with
    the post-game snap-count file, into "dressed but never took a snap" (a stale depth chart
    or a listed non-starter) and "played that day but not on this play" (in-game changes).
    """
    off = te[te["side"] == "offense"]
    plays = off.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", "team", "week"]].reset_index(drop=True)
    qb1 = depth_chart_qb1(season).merge(load_roster_status(season)[["team", "week", "gsis_id", "status"]],
                                        on=["team", "week", "gsis_id"], how="left")
    qb_cands = off[off["group"] == "QB"][["game_id", "play_id", "gsis_id", "depth_rank", "usage_rank", "y"]]
    return qb_check_table(qb_cands, plays, qb1, players_with_a_snap(season))


def qb_check_sentence(t: pd.DataFrame) -> str:
    """One sentence with the QB-check decomposition, shared by the headline and section (c)."""
    dc = t[t["candidate"] == "depth-chart QB1"].set_index("condition")
    if not {ACT_COND, SNAP_COND, NOSNAP_COND} <= set(dc.index):
        return ""
    act, snap, nosnap = dc.loc[ACT_COND], dc.loc[SNAP_COND], dc.loc[NOSNAP_COND]
    ina = float(dc.loc[QB1_STATUS_LABELS["INA"], "share_of_plays"]) if QB1_STATUS_LABELS["INA"] in dc.index else 0.0
    other = float(sum(dc.loc[QB1_STATUS_LABELS[k], "share_of_plays"] for k in ("other", "not_on_roster", "not_listed", "mismatch")
                      if QB1_STATUS_LABELS[k] in dc.index))
    return (f"The depth-chart QB1 is on the ACT list on {act['share_of_plays']:.1%} of test plays and is the QB on the "
            f"field on {act['on_field_rate']:.1%} of them; {nosnap['share_of_plays'] / act['share_of_plays']:.1%} of "
            f"those plays had him dressed but never taking a snap (stale depth chart / listed non-starter), and when he "
            f"did play that day he is off the field on {1 - snap['on_field_rate']:.1%} of plays (in-game changes). On "
            f"{ina:.1%} of plays the listed QB1 was declared inactive and on {other:.1%} he was on a reserve list or "
            f"missing from the depth chart; the weekly roster status resolves those pre-kickoff.")


def _players_text(t: dict[str, pd.DataFrame], cfg: ParticipationConfig) -> str:
    L = []
    L.append(f"Seeded subsample of {cfg.players_train_plays} plays from {cfg.val_season} (training; rounds chosen on a "
             f"game-held-out fifth) and {cfg.players_test_plays} plays from {cfg.test_season} (test), restricted to plays "
             "with exactly 11 participants per side. Candidates = the team's `ACT` weekly roster (see below), side-matched "
             "position groups (K/P/LS excluded). Per-candidate features: position group and depth-chart position, "
             "depth-chart rank that week (min over listed positions, 4 if unlisted), season-to-date / last-3 / "
             "previous-season snap share from strictly earlier games, weeks since last snap, prior-usage rank within the "
             "team-week position group (season-to-date share, previous season for week 1, ties by depth rank), group "
             "size, the slots the play's grouping gives the candidate's group (true, or imputed argmax and expected "
             "value from stages (a)/(b)), slot minus rank, and the situation. `imputed grouping` for the defense uses "
             "the stage (b) `S2+imp_pers` defense-personnel model (itself fed by imputed offense personnel). "
             "`Candidates` are the weekly roster rows with status `ACT`: in these files that is exactly the 48-man "
             "game-day active list (declared inactives carry `INA`, ~6 per team-week, none of them takes a snap), fixed "
             "about 90 minutes before kickoff, so the candidate set is pre-play but not mid-week information.\n")
    L.append("Candidate coverage:\n")
    L.append(md_table(t["coverage"], floatfmt="{:.4f}"))
    L.append("\nFits (LightGBM binary, on-field or not):\n")
    L.append(md_table(t["fits"]))
    L.append("\nSlot decoding on test: exact-lineup accuracy (all 11 correct) and per-slot accuracy (correct players / 11):\n")
    L.append(md_table(t["summary"], floatfmt="{:.4f}"))
    L.append("\nSame, split by whether the imputed grouping gave the correct slot counts:\n")
    L.append(md_table(t["conditional_on_grouping"], floatfmt="{:.4f}"))
    L.append("\nPer position subgroup on test (precision / recall / F1 after decoding; AUC of the raw score among "
             "candidates of that subgroup; WR/TE/RB/CB tiers use the prior-usage rank):\n")
    L.append(md_table(t["per_subgroup"], floatfmt="{:.4f}"))
    L.append("\nWhere the misses are (true on-field players not selected), by subgroup:\n")
    piv = t["errors"].pivot_table(index="subgroup", columns="decoder", values="share_of_misses", aggfunc="first").reset_index()
    L.append(md_table(piv, floatfmt="{:.3f}"))
    L.append("\nQB check on test. Because candidates are the `ACT` (game-day active) list, a listed QB1 who was "
             "declared inactive is never a candidate; those plays are counted from the weekly roster status. Among plays "
             "with an ACT depth-1 QB the residual is split with the post-game snap-count file (it lists only players who "
             "took a snap, so this split is diagnostic, not a feature): `dressed non-starter` = the depth chart listed a "
             "QB who never played that day, `took a snap` = the residual is an in-game change (injury, garbage time, "
             "wildcat / trick plays). " + qb_check_sentence(t["qb_check"]) + "\n")
    L.append(md_table(t["qb_check"], floatfmt="{:.4f}"))
    return "\n".join(L)
