"""Sport-agnostic pieces of the cross-game correlation backtest.

Adds two things the NFL-only code did not need:

* Moneyline parlays priced at the actual closing decimals (NHL/MLB legs are not
  50/50 at -110), with the independence-expected ROI computed from the vig-free
  probabilities so the correlation edge can be isolated.
* Starting-pitcher pairs for MLB: the anchor is one start by pitcher P for team
  A against team B; leg 1 is P's next start, leg 2 is B's next game against a
  different opponent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geo_model.parlay.data import to_team_games


@dataclass
class MoneylineParlayResult:
    """Two-sided moneyline parlay outcome on a set of leg pairs.

    Attributes:
        n_pairs: Pairs bet (2 units each: both-win and both-lose parlays).
        realized_roi: Realized profit per unit staked.
        independent_roi: Expected profit per unit under leg independence, using
            vig-free probabilities and the actual payouts.
        edge: ``realized_roi - independent_roi`` (the correlation contribution).
        edge_ci: 95% bootstrap CI for ``edge``.
        p_same_side: Realized frequency that both teams won or both lost.
        p_same_side_indep: Same under independence.
    """

    n_pairs: int
    realized_roi: float
    independent_roi: float
    edge: float
    edge_ci: tuple[float, float]
    p_same_side: float
    p_same_side_indep: float


def leg_prices(team_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach win/lose decimals and fair probabilities to each team-game row.

    Three-way markets (soccer 1X2) are supported through an optional ``p_draw``
    column on ``games``: the "lose" leg is the opponent's win price, whose fair
    probability is ``1 - p_win - p_draw``.

    Args:
        team_games: Output of :func:`to_team_games`.
        games: Cleaned games with ``home_decimal, away_decimal, p_home`` and
            optionally ``p_draw``.

    Returns:
        ``team_games`` with ``win_decimal, lose_decimal, p_win, p_lose`` columns.
    """
    cols = ["home_decimal", "away_decimal", "p_home"] + (["p_draw"] if "p_draw" in games else [])
    g = games.set_index("game_id")[cols]
    tg = team_games.join(g, on="game_id")
    if "p_draw" not in tg:
        tg["p_draw"] = 0.0
    tg["p_draw"] = tg["p_draw"].fillna(0.0)
    home = tg["is_home"].to_numpy()
    tg["win_decimal"] = np.where(home, tg["home_decimal"], tg["away_decimal"])
    tg["lose_decimal"] = np.where(home, tg["away_decimal"], tg["home_decimal"])
    tg["p_win"] = np.where(home, tg["p_home"], 1 - tg["p_home"] - tg["p_draw"])
    tg["p_lose"] = np.where(home, 1 - tg["p_home"] - tg["p_draw"], tg["p_home"])
    return tg


def _attach_next_prices(pairs: pd.DataFrame, tg: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """For each pair add each leg's win/lose decimals, fair p(win) and win flag."""
    priced = leg_prices(tg, games)
    key = priced.set_index(["game_id", "team"])
    res = games.set_index("game_id")["result"]
    out = pairs.copy()
    for side, team_col, gid_col in (("h", "team_h", "h_next_game_id"), ("a", "team_a", "a_next_game_id")):
        idx = pd.MultiIndex.from_arrays([out[gid_col], out[team_col]])
        rows = key.reindex(idx)
        out[f"{side}_win_dec"] = rows["win_decimal"].to_numpy()
        out[f"{side}_lose_dec"] = rows["lose_decimal"].to_numpy()
        out[f"{side}_p_win"] = rows["p_win"].to_numpy()
        out[f"{side}_p_lose"] = rows["p_lose"].to_numpy()
        margin = res.reindex(out[gid_col]).to_numpy() * np.where(rows["is_home"].to_numpy(), 1, -1)
        out[f"{side}_won"] = margin > 0
        out[f"{side}_lost"] = margin < 0
    return out


def moneyline_parlay(pairs: pd.DataFrame, tg: pd.DataFrame, games: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> MoneylineParlayResult:
    """Bet (H wins, A wins) and (H loses, A loses) parlays on every pair.

    Args:
        pairs: Shared-game pairs (from ``build_shared_game_pairs`` or
            :func:`build_pitcher_pairs`).
        tg: Team-game table the pairs were built from.
        games: Cleaned games with prices.
        n_boot: Bootstrap resamples for the edge CI.
        seed: RNG seed.

    Returns:
        :class:`MoneylineParlayResult`.
    """
    p = _attach_next_prices(pairs, tg, games)
    p = p.dropna(subset=["h_win_dec", "a_win_dec", "h_lose_dec", "a_lose_dec", "h_p_win", "a_p_win", "h_p_lose", "a_p_lose"])
    # Ties/draws: a tied leg loses both parlays (no push handling; in soccer the draw is a priced outcome).
    both_win = p["h_won"] & p["a_won"]
    both_lose = p["h_lost"] & p["a_lost"]
    pay_ww = p["h_win_dec"] * p["a_win_dec"]
    pay_ll = p["h_lose_dec"] * p["a_lose_dec"]
    profit = np.where(both_win, pay_ww, 0.0) + np.where(both_lose, pay_ll, 0.0) - 2.0
    ind = p["h_p_win"] * p["a_p_win"] * pay_ww + p["h_p_lose"] * p["a_p_lose"] * pay_ll - 2.0
    n = len(p)
    if n == 0:
        nan = float("nan")
        return MoneylineParlayResult(0, nan, nan, nan, (nan, nan), nan, nan)
    edge = float(profit.mean() - ind.mean()) / 2.0
    rng = np.random.default_rng(seed)
    diffs = profit - ind.to_numpy()
    boots = np.array([diffs[rng.integers(0, n, n)].mean() / 2.0 for _ in range(n_boot)])
    same = both_win | both_lose
    same_ind = p["h_p_win"] * p["a_p_win"] + p["h_p_lose"] * p["a_p_lose"]
    return MoneylineParlayResult(
        n_pairs=int(n),
        realized_roi=float(profit.mean() / 2.0),
        independent_roi=float(ind.mean() / 2.0),
        edge=edge,
        edge_ci=(float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        p_same_side=float(same.mean()),
        p_same_side_indep=float(same_ind.mean()),
    )


def build_pitcher_pairs(games: pd.DataFrame) -> pd.DataFrame:
    """Pairs anchored on one pitcher's start.

    Requires ``home_qb_id`` / ``away_qb_id`` to hold starting pitcher ids (as
    produced by :func:`geo_model.parlay.data_multisport.load_mlb`).

    Leg 1 (``h`` side in the output, to reuse the pair-analysis helpers) is the
    pitcher's next start for the same team against a team other than the anchor
    opponent. Leg 2 (``a`` side) is the anchor opponent's next game against a
    team other than the pitcher's team, with a different starting pitcher than
    the one it just faced. Both legs must be different games.

    Args:
        games: Cleaned MLB games.

    Returns:
        Pairs table in the same layout as ``build_shared_game_pairs`` with extra
        ``pitcher`` and ``anchor_pitcher_resid`` columns. Each anchor game yields
        up to two rows (one per starting pitcher).
    """
    tg = to_team_games(games)
    tg = tg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    # Next game of each team (any opponent), then the first future game whose opponent differs.
    by_team = {t: g.reset_index(drop=True) for t, g in tg.groupby("team")}
    rows = []
    for t, g in by_team.items():
        gd = g["gameday"].to_numpy()
        opp = g["opp"].to_numpy()
        qb = g["qb_id"].to_numpy()
        season = g["season"].to_numpy()
        for i in range(len(g)):
            if pd.isna(qb[i]):
                continue
            # leg 1: pitcher's next start (same team, same season) vs a different opponent
            j = i + 1
            leg1 = None
            while j < len(g) and season[j] == season[i]:
                if qb[j] == qb[i]:
                    leg1 = j if opp[j] != opp[i] else None
                    break
                j += 1
            if leg1 is None:
                continue
            # leg 2: opponent's next game vs someone else, with a different starter than the anchor
            og = by_team.get(opp[i])
            if og is None:
                continue
            k = int(np.searchsorted(og["gameday"].to_numpy(), gd[i], side="right"))
            leg2 = None
            while k < len(og) and og["season"].iat[k] == season[i]:
                if og["opp"].iat[k] != t:
                    leg2 = k
                    break
                k += 1
            if leg2 is None or og["game_id"].iat[leg2] == g["game_id"].iat[leg1]:
                continue
            if og["opp"].iat[leg2] == g["opp"].iat[leg1]:
                continue  # legs would share an opponent
            rows.append(
                {
                    "game_id": g["game_id"].iat[i],
                    "season": season[i],
                    "week": g["week"].iat[i],
                    "gameday": gd[i],
                    "team_h": t,
                    "team_a": opp[i],
                    "pitcher": qb[i],
                    "anchor_resid": g["resid"].iat[i],
                    "anchor_tresid": g["tresid"].iat[i],
                    "h_next_game_id": g["game_id"].iat[leg1],
                    "a_next_game_id": og["game_id"].iat[leg2],
                    "h_next_opp": opp[leg1],
                    "a_next_opp": og["opp"].iat[leg2],
                    "h_next_week": g["week"].iat[leg1],
                    "a_next_week": og["week"].iat[leg2],
                    "h_next_resid": g["resid"].iat[leg1],
                    "a_next_resid": og["resid"].iat[leg2],
                    "h_next_tresid": g["tresid"].iat[leg1],
                    "a_next_tresid": og["tresid"].iat[leg2],
                    "gap_days_leg1": (gd[leg1] - gd[i]) / np.timedelta64(1, "D"),
                    "gap_days_leg2": (og["gameday"].iat[leg2] - gd[i]) / np.timedelta64(1, "D"),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["abs_surprise"] = out["anchor_resid"].abs()
        out["new_qb_any"] = False
        out["same_next_week"] = out["h_next_week"] == out["a_next_week"]
    return out
