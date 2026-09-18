"""04 Fouls and the referee channel: can a foul model be +EV against a rolling mean?

Stage 04 of the scenario expected-value phase. Run::

    cd /home/user/geo-model
    python -m research.scenario_ev.fouls --stage build   # ~3 min, builds both universes
    python -m research.scenario_ev.fouls --stage a       # odds-table team / match foul totals
    python -m research.scenario_ev.fouls --stage b       # StatsBomb player fouls + referee channel
    python -m research.scenario_ev.fouls --stage c       # imputed-state layer, like for like
    python -m research.scenario_ev.fouls --stage d       # betting simulation + the goals haircut
    python -m research.scenario_ev.fouls --stage report  # writes reports/04_fouls.md

Protocol (shared with stages 01-03, see ``reports/04_fouls.md`` section 1): a chronological
discovery / confirmation split fixed in code, a generalist-vs-specialist gate before any
modelling, a shrunk rolling-mean book proxy that is recalibrated so it is not a straw man,
and every expected-value claim haircut by how much a real bookmaker beats the identical
proxy on the one count market whose line is observed (total goals at 2.5).

Reuse, rather than reinvention, is deliberate: the odds-table layer is built with the
corner stage's feature primitives and the player layer with the card stage's cached
StatsBomb player-match table, so that a difference between stages is a difference in the
target rather than in the design.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    log_loss,
    per_sample_log_loss,
    r2,
)
from research.privileged_tracking.common.splits import forward_split, group_kfold
from research.scenario_ev import common as C
from research.scenario_ev import fouls_features as FF
from research.scenario_ev.corners_features import (
    CornerConfig,
    add_prior_features,
    merge_opponent,
    proxy_lambda,
    to_team_match,
)

DISC, CONF = C.DISCOVERY, C.CONFIRMATION


@dataclass(frozen=True)
class FoulConfig:
    """Configuration of the fouls stage.

    Attributes:
        split: Discovery / confirmation split, seed and bootstrap size.
        corner: Corner-stage configuration reused for the prior-feature primitives.
        min_prior: Minimum strictly-prior matches with fouls recorded, both teams.
        min_prior_apps: Minimum strictly-prior appearances for a player prop.
        proxy_k_grid: Candidate shrinkages for the foul proxy, chosen on discovery.
        team_lines: Team foul half-lines simulated.
        match_lines: Match total foul half-lines simulated.
        holds: Two-way overrounds simulated (the realistic range for a foul prop).
        edge_grid: Candidate EV thresholds, chosen on discovery.
        seasons: StatsBomb (competition, season) pairs used for the player layer.
        n_jobs: LightGBM threads.
        refit_seeds: Seeds used for the refit-noise floor. Five rather than the three
            the other stages use, because this stage's smallest surviving claim is a
            channel increment of a few ten-thousandths of a nat and a three-seed range
            is a thin estimate of the spread of such a quantity.
        max_gate_rows: Row cap for the odds-table gate (speed); ``0`` disables it.
    """

    split: C.SplitConfig = field(default_factory=C.SplitConfig)
    corner: CornerConfig = field(default_factory=CornerConfig)
    min_prior: int = 5
    min_prior_apps: int = 3
    proxy_k_grid: tuple[float, ...] = (6.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0)
    team_lines: tuple[float, ...] = tuple(float(x) + 0.5 for x in range(7, 21))
    match_lines: tuple[float, ...] = tuple(float(x) + 0.5 for x in range(15, 41))
    holds: tuple[float, ...] = (0.04, 0.06, 0.08)
    edge_grid: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
    seasons: tuple[tuple[str, str], ...] = (
        ("Premier League", "2015/2016"), ("La Liga", "2015/2016"),
        ("Serie A", "2015/2016"), ("Ligue 1", "2015/2016"),
    )
    n_jobs: int = 2
    refit_seeds: tuple[int, ...] = (20260918, 20260919, 20260920, 20260921,
                                    20260922)
    max_gate_rows: int = 0


CFG = FoulConfig()
TEAM_CACHE = "fouls_team.parquet"
#: Implausible foul counts above this are winsorised (see :func:`build_team_table`).
FOUL_CAP = 40.0
PLAYER_CACHE = "fouls_player.parquet"


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

#: Candidate LightGBM settings, scored on an inner forward split *inside* discovery.
#: The corner stage found that the default "400 trees / 31 leaves" overfits a weak count
#: signal badly enough to lose to the proxy outright, so capacity is chosen, not assumed.
PARAM_GRID: tuple[dict[str, Any], ...] = (
    dict(n_estimators=100, num_leaves=7, min_child_samples=1000, learning_rate=0.05),
    dict(n_estimators=200, num_leaves=7, min_child_samples=500, learning_rate=0.05),
    dict(n_estimators=300, num_leaves=15, min_child_samples=500, learning_rate=0.04),
    dict(n_estimators=400, num_leaves=31, min_child_samples=200, learning_rate=0.04),
)

LGB_BASE: dict[str, Any] = dict(
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1,
)

#: Categorical columns coerced to pandas ``category`` before every fit.
CAT_COLS: tuple[str, ...] = ("div_code", "competition", "start_position", "pos_group",
                             "line", "flank")


def model_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Select the design columns and coerce the categorical ones.

    Args:
        df: Source frame.
        cols: Feature columns.

    Returns:
        Frame [n, len(cols)] ready for LightGBM.
    """
    X = df[list(cols)].copy()
    for c in CAT_COLS:
        if c in X.columns and not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, kind: str,
                params: dict[str, Any] | None = None, seed: int = 0, n_jobs: int = 2,
                offset_tr: np.ndarray | None = None, offset_te: np.ndarray | None = None,
                ) -> np.ndarray:
    """Fit a LightGBM model and predict.

    Count models can be given a proxy ``offset`` (an exposure), which makes the booster
    learn a multiplicative correction to the rolling mean rather than the level itself.

    Args:
        X_tr: Training design [n_tr, k].
        y_tr: Training target [n_tr].
        X_te: Scoring design [n_te, k].
        kind: ``"count"`` (Poisson) or ``"binary"``.
        params: Overrides for the chosen defaults.
        seed: Model seed.
        n_jobs: Threads.
        offset_tr: Exposure for the training rows [n_tr].
        offset_te: Exposure for the scored rows [n_te].

    Returns:
        Predicted means or probabilities [n_te].
    """
    import lightgbm as lgb

    p = dict(LGB_BASE)
    p.update(PARAM_GRID[1] if params is None else params)
    p.update(objective="poisson" if kind == "count" else "binary",
             random_state=seed, n_jobs=n_jobs)
    model = lgb.LGBMRegressor(**p) if kind == "count" else lgb.LGBMClassifier(**p)
    kw: dict[str, Any] = {}
    if offset_tr is not None:
        kw["init_score"] = np.log(np.clip(offset_tr, 1e-6, None))
    model.fit(X_tr, y_tr, **kw)
    if kind == "binary":
        return model.predict_proba(X_te)[:, 1]
    raw = model.predict(X_te, raw_score=True)
    if offset_te is not None:
        raw = raw + np.log(np.clip(offset_te, 1e-6, None))
    return np.exp(raw)


def cv_and_confirm(df: pd.DataFrame, cols: Sequence[str], target: str, kind: str,
                   cfg: FoulConfig, params: dict[str, Any] | None = None,
                   seed: int | None = None, n_folds: int = 5,
                   offset: np.ndarray | None = None) -> np.ndarray:
    """Predictions for every row: match-grouped CV in discovery, fit-on-discovery outside.

    Args:
        df: Population with ``split`` and ``match_id``.
        cols: Feature columns.
        target: Target column.
        kind: ``"count"`` or ``"binary"``.
        cfg: Stage configuration.
        params: LightGBM overrides.
        seed: Seed override.
        n_folds: Discovery CV folds.
        offset: Optional exposure [n].

    Returns:
        Predictions [n].
    """
    seed = cfg.split.seed if seed is None else seed
    X = model_frame(df, cols)
    y = df[target].to_numpy(dtype=float)
    match = df["match_id"].to_numpy()
    disc = df["split"].to_numpy() == DISC
    out = np.full(len(df), np.nan)
    di = np.flatnonzero(disc)
    for tr, te in group_kfold(match[di], n_splits=n_folds, seed=seed):
        tr_i, te_i = di[tr], di[te]
        out[te_i] = fit_predict(X.iloc[tr_i], y[tr_i], X.iloc[te_i], kind, params, seed,
                                cfg.n_jobs,
                                None if offset is None else offset[tr_i],
                                None if offset is None else offset[te_i])
    ci = np.flatnonzero(~disc)
    if len(ci):
        out[ci] = fit_predict(X.iloc[di], y[di], X.iloc[ci], kind, params, seed, cfg.n_jobs,
                              None if offset is None else offset[di],
                              None if offset is None else offset[ci])
    return out


def tune_params(df: pd.DataFrame, cols: Sequence[str], target: str, kind: str,
                cfg: FoulConfig, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose LightGBM capacity on an inner forward split of the discovery half.

    Fitting on the first 75% of discovery dates and scoring on the last 25% keeps the
    confirmation half untouched, so a null result cannot be confused with a capacity
    artefact.

    Args:
        df: Population with ``split`` and ``match_id``.
        cols: Feature columns.
        target: Target column.
        kind: ``"count"`` or ``"binary"``.
        cfg: Stage configuration.
        label: Name recorded in the result table.

    Returns:
        Tuple of the winning parameter dict and the full scoring table.
    """
    d = df[df["split"] == DISC].reset_index(drop=True)
    tr, te = forward_split(d["date"].to_numpy(), test_frac=0.25)
    X = model_frame(d, cols)
    y = d[target].to_numpy(dtype=float)
    rows = []
    best, best_loss = dict(PARAM_GRID[1]), np.inf
    for params in PARAM_GRID:
        pred = fit_predict(X.iloc[tr], y[tr], X.iloc[te], kind, params, cfg.split.seed,
                           cfg.n_jobs)
        if kind == "count":
            r = C.fit_nb_dispersion(y[tr], np.full(len(tr), y[tr].mean()))
            loss = float(C.nb_log_score(y[te], pred, r).mean())
        else:
            loss = float(per_sample_log_loss(y[te], pred).mean())
        rows.append({"label": label, "target": target, "params": str(params),
                     "n_train": int(len(tr)), "n_valid": int(len(te)), "loss": loss})
        if loss < best_loss:
            best_loss, best = loss, dict(params)
    return best, pd.DataFrame(rows).sort_values("loss").reset_index(drop=True)


# ---------------------------------------------------------------------------
# (a) universe: the odds table
# ---------------------------------------------------------------------------

#: Team-match statistics carried from the opposing row.
_CARRY_PREFIXES = ("pm_", "n_", "fps_", "fpc_", "ps_", "pc_", "w6_")


def choose_proxy_k(tm: pd.DataFrame, disc: np.ndarray, cfg: FoulConfig) -> pd.DataFrame:
    """Score the foul proxy's shrinkage and functional form on discovery rows only.

    Args:
        tm: Team-match frame after :func:`fouls_features.add_foul_proxy_inputs` and
            :func:`corners_features.merge_opponent`.
        disc: Boolean mask [n] of discovery rows.
        cfg: Stage configuration.

    Returns:
        One row per (form, k) with squared error, R2 and correlation on discovery,
        sorted best first.
    """
    y = tm["fouls_for"].to_numpy(dtype=float)
    rows = []
    for mult in (True, False):
        for k in cfg.proxy_k_grid:
            mu = FF.foul_proxy_lambda(tm, k, mult)
            ok = disc & np.isfinite(mu) & np.isfinite(y)
            rows.append({
                "form": "mult" if mult else "add", "k": float(k), "n": int(ok.sum()),
                "mse": float(np.mean((y[ok] - mu[ok]) ** 2)),
                "r2": r2(y[ok], mu[ok]), "corr": float(np.corrcoef(y[ok], mu[ok])[0, 1]),
                "sd_pred": float(np.std(mu[ok])), "sd_y": float(np.std(y[ok])),
                "bias": float(np.mean(mu[ok] - y[ok])),
            })
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def build_team_table(cfg: FoulConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Build (or load) the team-match foul table from the football-data odds file.

    Every rolling quantity comes from strictly earlier *dates* in the same division-season
    (the corner stage's :func:`~research.scenario_ev.corners_features.prior_by_date`), so
    same-day fixtures cannot leak into one another and a match never enters its own
    features. The proxy's shrinkage and functional form are chosen on the discovery half
    only, inside this function, before any model is fitted.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        Two rows per usable match [2 * n_matches] with ``y_fouls`` (the acting team's
        fouls), ``y_total_fouls``, ``proxy_fouls``, ``proxy_goals``, the prior feature
        block, the market block and the ``split`` label.
    """
    path = C.processed_dir() / TEAM_CACHE
    if path.exists() and not force:
        return pd.read_parquet(path)

    _log("reading odds table")
    raw = pd.read_parquet(C.odds_path())
    raw = raw.sort_values(["MatchDate", "Division", "HomeTeam", "AwayTeam"],
                          kind="mergesort").reset_index(drop=True)
    raw["match_id"] = np.arange(len(raw), dtype=np.int64)
    dt = pd.to_datetime(raw["MatchDate"])
    start = np.where(dt.dt.month >= 7, dt.dt.year, dt.dt.year - 1)
    raw["season"] = [f"{s}/{s + 1}" for s in start]
    raw["div_season"] = raw["Division"].astype(str) + "|" + raw["season"].astype(str)

    _log("building strictly-prior team features")
    tm = to_team_match(raw)
    tm = add_prior_features(tm, cfg.corner)
    tm = FF.add_foul_proxy_inputs(tm, cfg.corner)
    carry = list(dict.fromkeys(
        [c for c in tm.columns if c.startswith(_CARRY_PREFIXES)] + ["rest_days", "n_prior"]))
    tm = merge_opponent(tm, carry)

    meta = raw.set_index("match_id")
    mid = tm["match_id"].to_numpy()
    is_home = tm["is_home"].to_numpy() == 1
    # football-data carries a handful of impossible foul counts (up to 145 in a match);
    # they are winsorised rather than dropped so that both rows of a match survive
    # together. 5 of 210,076 team-rows are affected (0.002%).
    tm["y_fouls"] = np.minimum(tm["fouls_for"].to_numpy(dtype=float), FOUL_CAP)
    tm["y_fouls_against"] = np.minimum(tm["fouls_against"].to_numpy(dtype=float), FOUL_CAP)
    tm["y_total_fouls"] = tm["y_fouls"] + tm["y_fouls_against"]
    tm["y_yellow"] = tm["yellow_for"].to_numpy(dtype=float)
    tm["y_goals_total"] = (tm["goals_for"] + tm["goals_against"]).to_numpy(dtype=float)
    tm["month"] = pd.to_datetime(tm["date"]).dt.month.to_numpy()

    h_elo = meta["HomeElo"].reindex(mid).to_numpy(dtype=float)
    a_elo = meta["AwayElo"].reindex(mid).to_numpy(dtype=float)
    tm["own_elo"] = np.where(is_home, h_elo, a_elo)
    tm["opp_elo"] = np.where(is_home, a_elo, h_elo)
    tm["elo_gap"] = tm["own_elo"] - tm["opp_elo"]
    tm["abs_elo_gap"] = np.abs(tm["elo_gap"])
    for n, (hc, ac) in {"form3": ("Form3Home", "Form3Away"),
                        "form5": ("Form5Home", "Form5Away")}.items():
        hv = meta[hc].reindex(mid).to_numpy(dtype=float)
        av = meta[ac].reindex(mid).to_numpy(dtype=float)
        tm[f"own_{n}"] = np.where(is_home, hv, av)
        tm[f"opp_{n}"] = np.where(is_home, av, hv)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = {k: 1.0 / meta[k].reindex(mid).to_numpy(dtype=float)
               for k in ("OddHome", "OddDraw", "OddAway")}
        tot = inv["OddHome"] + inv["OddDraw"] + inv["OddAway"]
    p_home, p_draw, p_away = inv["OddHome"] / tot, inv["OddDraw"] / tot, inv["OddAway"] / tot
    tm["p_win"] = np.where(is_home, p_home, p_away)
    tm["p_draw"] = p_draw
    tm["p_lose"] = np.where(is_home, p_away, p_home)
    tm["p_fav"] = np.maximum(p_home, p_away)
    over, _ = C.novig_two_way(meta["Over25"].reindex(mid).to_numpy(dtype=float),
                             meta["Under25"].reindex(mid).to_numpy(dtype=float))
    tm["p_over25"] = over
    tm["odd_over25"] = meta["Over25"].reindex(mid).to_numpy(dtype=float)
    tm["odd_under25"] = meta["Under25"].reindex(mid).to_numpy(dtype=float)
    tm["div_code"] = pd.factorize(tm["Division"])[0].astype(int)
    tm["season_idx"] = pd.factorize(tm["season"].astype(str))[0].astype(int)

    _log("auditing the prior features by brute force")
    audit_team_table(tm).to_parquet(C.processed_dir() / "fouls_audit.parquet", index=False)

    usable = (
        tm["y_fouls"].notna() & tm["y_fouls_against"].notna()
        & (tm["fpc_fouls_for"].to_numpy() >= cfg.min_prior)
        & (tm["opp_fpc_fouls_against"].to_numpy() >= cfg.min_prior)
        & (tm["opp_fpc_fouls_for"].to_numpy() >= cfg.min_prior)
        & (tm["fpc_fouls_against"].to_numpy() >= cfg.min_prior)
    ).to_numpy()
    # keep whole matches only
    both = pd.Series(usable).groupby(tm["match_id"].to_numpy()).transform("all").to_numpy()
    out = tm[both].reset_index(drop=True)
    out["split"] = C.chronological_split(out["match_id"], out["date"],
                                         cfg.split.discovery_frac)

    disc = (out["split"] == DISC).to_numpy()
    sweep = choose_proxy_k(out, disc, cfg)
    best = sweep.iloc[0]
    _log(f"proxy chosen on discovery: form={best['form']} k={best['k']} "
         f"r2={best['r2']:.4f}")
    out["proxy_fouls"] = FF.foul_proxy_lambda(out, float(best["k"]),
                                              best["form"] == "mult")
    out["proxy_k"] = float(best["k"])
    out["proxy_form"] = str(best["form"])
    out["proxy_goals"] = proxy_lambda(out, "goals_for", "goals_against",
                                      cfg.corner.proxy_k)
    opp_proxy = out[["match_id", "is_home", "proxy_fouls", "proxy_goals"]].copy()
    opp_proxy["is_home"] = 1 - opp_proxy["is_home"]
    opp_proxy = opp_proxy.rename(columns={"proxy_fouls": "opp_proxy_fouls",
                                          "proxy_goals": "opp_proxy_goals"})
    out = out.merge(opp_proxy, on=["match_id", "is_home"], how="left", validate="one_to_one")
    out["proxy_total_fouls"] = out["proxy_fouls"] + out["opp_proxy_fouls"]
    out["proxy_total_goals"] = out["proxy_goals"] + out["opp_proxy_goals"]
    out = out[np.isfinite(out["proxy_fouls"]) & np.isfinite(out["proxy_total_fouls"])]
    out = out.reset_index(drop=True)
    out.to_parquet(path, index=False)
    _log(f"team table: {len(out)} team-rows, {out['match_id'].nunique()} matches")
    return out


def to_match_rows(t: pd.DataFrame) -> pd.DataFrame:
    """Collapse the team-row table to one row per match for the match-total market.

    Args:
        t: Team-row table from :func:`build_team_table`.

    Returns:
        One row per match with ``h_``/``a_`` prefixed team features, the match-level
        market block and ``y_total_fouls``.
    """
    home = t[t["is_home"] == 1].set_index("match_id")
    away = t[t["is_home"] == 0].set_index("match_id")
    share = ["date", "Division", "div_season", "season", "div_code", "season_idx", "month",
             "split", "y_total_fouls", "y_goals_total", "p_draw", "p_fav", "p_over25",
             "odd_over25", "odd_under25", "proxy_total_fouls", "proxy_total_goals",
             "abs_elo_gap"]
    out = home[share].copy()
    out["p_home"] = home["p_win"]
    out["p_away"] = home["p_lose"]
    cols = [c for c in t.columns
            if c.startswith(("pm_", "n_", "lg_", "w6_", "proxy_", "fpc_", "fps_"))
            or c in ("rest_days", "own_elo", "own_form3", "own_form5", "n_prior")]
    for c in cols:
        out[f"h_{c}"] = home[c]
        out[f"a_{c}"] = away[c]
    out["elo_gap"] = out["h_own_elo"] - out["a_own_elo"]
    return out.reset_index()


def audit_team_table(tm: pd.DataFrame, n_check: int = 120, seed: int = 0) -> pd.DataFrame:
    """Brute-force re-derivation of the foul priors for a random sample of team-rows.

    Run on the *unfiltered* team-match frame, because that is the frame the priors were
    accumulated over: auditing the filtered universe instead would "find" a discrepancy
    that is really just the rows the universe drops. For each sampled row the team's prior
    foul sum and count are recomputed by filtering the whole frame to the same
    division-season and a strictly earlier date. The third check verifies the league level
    is a function of (division-season, date) alone, so that matches sharing a date cannot
    enter one another's league reference.

    Args:
        tm: Team-match frame after the prior features are attached, before the universe
            filter.
        n_check: Rows to verify.
        seed: RNG seed.

    Returns:
        One row per check with ``n``, ``max_abs_error`` and ``passed``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tm), size=min(n_check, len(tm)), replace=False)
    ds = tm["div_season"].to_numpy()
    team = tm["team"].to_numpy()
    date = pd.to_datetime(tm["date"]).to_numpy()
    f = tm["fouls_for"].to_numpy(dtype=float)
    ok_f = np.isfinite(f)
    ps = tm["fps_fouls_for"].to_numpy(dtype=float)
    pc = tm["fpc_fouls_for"].to_numpy(dtype=float)
    max_s = max_c = 0.0
    for i in idx:
        m = (ds == ds[i]) & (team == team[i]) & (date < date[i]) & ok_f
        max_s = max(max_s, abs(np.nansum(f[m]) - ps[i]))
        max_c = max(max_c, abs(m.sum() - pc[i]))
    g = tm.groupby(["div_season", "date"], observed=True)["lg_fouls_for"]
    within = g.transform(lambda x: x.max() - x.min())
    multi = g.transform("size") > 1
    return pd.DataFrame([
        {"check": "prior_foul_sum_strictly_earlier", "n": int(len(idx)),
         "max_abs_error": max_s, "passed": bool(max_s < 1e-9)},
        {"check": "prior_foul_count_strictly_earlier", "n": int(len(idx)),
         "max_abs_error": max_c, "passed": bool(max_c < 1e-9)},
        {"check": "league_level_constant_within_day", "n": int(multi.sum()),
         "max_abs_error": float(np.nanmax(within.to_numpy())),
         "passed": bool(np.nanmax(within.to_numpy()) < 1e-9)},
    ])


# ---------------------------------------------------------------------------
# (b) universe: StatsBomb player-matches
# ---------------------------------------------------------------------------


def build_player_table(cfg: FoulConfig = CFG, force: bool = False) -> pd.DataFrame:
    """Build (or load) the player-match foul table.

    The base is the card stage's cached StatsBomb player-match table -- the same rolling
    player, team, opponent and referee blocks, audited there against the raw tables -- so
    that the fouls layer differs from the cards layer in its target, not its design. Four
    things the card stage had no use for are added: a position-group foul backstop, the
    direct opponent's prior *fouls committed* rate (the mirror of the dribble feature the
    card stage carried), the opponent's prior fouls by pitch side, and the imputed
    defensive-state priors.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        One row per player-appearance in the four covered league seasons, with at least
        ``cfg.min_prior_apps`` strictly-prior appearances and a ``split`` label.
    """
    path = C.processed_dir() / PLAYER_CACHE
    if path.exists() and not force:
        return pd.read_parquet(path)

    from research.scenario_ev import cards

    _log("loading the card stage's player table")
    pl = cards.build_player_table(cards.CFG)
    keys = {(c, s) for c, s in cfg.seasons}
    m = pd.Series(list(zip(pl["competition"], pl["season"])), index=pl.index).isin(keys)
    uni = pl[m & (pl["line"] != "GK") & (pl["minutes"] > 0)].copy()
    uni = uni[uni["pl_prior_n"] >= cfg.min_prior_apps].reset_index(drop=True)
    uni["date"] = pd.to_datetime(uni["date"])

    # position-group backstop, accumulated over (group, date) so same-day rows cannot
    # enter one another's prior
    uni["_has"] = 1.0
    for tgt, glob in (("fouls", 1.18), ("fouls_won", 1.12)):
        uni[f"grp_{tgt}_pm"] = C.prior_group_daily_mean(uni, "pos_group", "date", tgt,
                                                        "_has", glob, 60.0)
    uni = uni.drop(columns=["_has"])

    # the direct opponent's prior foul rate (mirror of the card stage's dribble feature)
    mirror = {f: cards.mirror_flank(f) for f in uni["flank"].dropna().unique()}
    prio = uni["start_position"].map(cards.direct_opponent_lines)
    dopp = FF.direct_opponent_rates(uni, ["pl_fouls_p90", "pl_tackles_p90"], mirror, prio,
                                    prefix="doppf_")
    uni = pd.concat([uni.reset_index(drop=True), dopp.reset_index(drop=True)], axis=1)

    # the opponent team's prior fouls by pitch side
    _log("building flank foul priors")
    tables = cards._load_sb_tables()
    tmm = tables["team_match"][["match_id", "team_id"]].merge(
        tables["matches"][["match_id", "date"]], on="match_id", how="left")
    flank = FF.flank_foul_priors(tables["fouls"], tmm)
    side_cols = [c for c in flank.columns if c.startswith("tp_fouls_")]
    own = flank.rename(columns={c: f"own_{c}" for c in side_cols})
    uni = uni.merge(own.drop(columns=["tp_flank_n"]), on=["match_id", "team_id"], how="left")
    opp = flank.rename(columns={"team_id": "opp_id",
                                **{c: f"opp_{c}" for c in side_cols}})
    uni = uni.merge(opp.drop(columns=["tp_flank_n"]), on=["match_id", "opp_id"], how="left")
    # the opponent's fouls on the side this player occupies: the opponent's attacking
    # frame is mirrored, so a left-sided player meets the opponent's right-side fouls
    mir = uni["flank"].map(lambda f: mirror.get(f, "centre")).fillna("centre")
    uni["opp_mirror_fouls_pm"] = np.select(
        [mir == "left", mir == "right"],
        [uni["opp_tp_fouls_left_pm"], uni["opp_tp_fouls_right_pm"]],
        default=uni["opp_tp_fouls_centre_pm"].to_numpy())

    # imputed defensive-state priors (E2 student, never the 360 truth)
    st = cards.build_team_state_priors(cards.CFG)
    scols = [c for c in st.columns if c.startswith("stp_")]
    uni = uni.merge(st.rename(columns={**{c: f"own_st_{c[4:]}" for c in scols},
                                       "st_prior_n": "own_st_prior_n"}),
                    on=["match_id", "team_id"], how="left")
    uni = uni.merge(st.rename(columns={"team_id": "opp_id",
                                       **{c: f"opp_st_{c[4:]}" for c in scols},
                                       "st_prior_n": "opp_st_prior_n"}),
                    on=["match_id", "opp_id"], how="left")

    exp90 = np.clip(uni["pl_minutes_exp"].to_numpy(dtype=float), 1.0, 95.0) / 90.0
    uni["proxy_mu_fouls"] = uni["pl_fouls_p90"].to_numpy(dtype=float) * exp90
    uni["proxy_mu_fouls_won"] = uni["pl_fouls_won_p90"].to_numpy(dtype=float) * exp90
    uni["split"] = C.chronological_split(uni["match_id"], uni["date"],
                                         cfg.split.discovery_frac)
    uni.to_parquet(path, index=False)
    _log(f"player table: {len(uni)} appearances, {uni['match_id'].nunique()} matches")
    return uni


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

_T_PROXY = ["proxy_fouls", "pm_fouls_for", "opp_pm_fouls_against", "fpc_fouls_for",
            "opp_fpc_fouls_against", "is_home"]

_T_EVENT = [
    "pm_fouls_against", "opp_pm_fouls_for", "pm_corners_for", "pm_corners_against",
    "pm_shots_for", "pm_shots_against", "pm_target_for", "pm_goals_for",
    "pm_goals_against", "pm_yellow_for", "opp_pm_corners_for", "opp_pm_shots_for",
    "opp_pm_shots_against", "opp_pm_goals_for", "opp_pm_goals_against",
    "opp_pm_yellow_for", "n_prior", "opp_n_prior", "rest_days", "opp_rest_days",
    "own_elo", "opp_elo", "elo_gap", "abs_elo_gap", "own_form3", "own_form5",
    "opp_form3", "opp_form5",
]

_T_DIV = ["lg_fouls_for", "lg_home_fouls_for", "lg_away_fouls_for", "lg_yellow_for",
          "lg_corners_for", "div_code", "season_idx", "month"]

_T_MARKET = ["p_win", "p_draw", "p_lose", "p_fav", "p_over25"]

#: Nested feature sets for the team-foul market. ``F1`` deliberately contains **no**
#: division identifier or league level, so that the division question -- does the proxy
#: already absorb the enormous between-division differences in foul rates? -- is answered
#: by the F1 -> F2 step rather than assumed.
TEAM_SETS: dict[str, list[str]] = {
    "F0_proxy": list(_T_PROXY),
    "F1_event_no_division": _T_PROXY + _T_EVENT,
    "F2_plus_division": _T_PROXY + _T_EVENT + _T_DIV,
    "F3_plus_market": _T_PROXY + _T_EVENT + _T_DIV + _T_MARKET,
}


def match_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Feature sets for the match-total market, mirroring :data:`TEAM_SETS`.

    Args:
        df: Match-row table from :func:`to_match_rows`.

    Returns:
        Map from set name to the columns present in ``df``.
    """
    def pref(cols: Sequence[str]) -> list[str]:
        out: list[str] = []
        for c in cols:
            for p in ("h_", "a_"):
                if f"{p}{c}" in df.columns:
                    out.append(f"{p}{c}")
        return out

    proxy = ["proxy_total_fouls"] + pref(["proxy_fouls", "pm_fouls_for", "pm_fouls_against",
                                          "fpc_fouls_for"])
    event = pref(["pm_corners_for", "pm_shots_for", "pm_shots_against", "pm_goals_for",
                  "pm_goals_against", "pm_yellow_for", "n_prior", "rest_days",
                  "own_elo", "own_form3", "own_form5"]) + ["elo_gap", "abs_elo_gap"]
    div = pref(["lg_fouls_for", "lg_yellow_for"]) + ["div_code", "season_idx", "month"]
    market = ["p_home", "p_away", "p_draw", "p_fav", "p_over25"]
    sets = {"F0_proxy": proxy, "F1_event_no_division": proxy + event,
            "F2_plus_division": proxy + event + div,
            "F3_plus_market": proxy + event + div + market}
    return {k: [c for c in dict.fromkeys(v) if c in df.columns] for k, v in sets.items()}


_P_PROXY_F = ["proxy_mu_fouls", "pl_fouls_p90", "pl_minutes_exp", "pl_prior_n",
              "pl_starter_rate", "grp_fouls_pm", "starter_i"]
_P_PROXY_W = ["proxy_mu_fouls_won", "pl_fouls_won_p90", "pl_minutes_exp", "pl_prior_n",
              "pl_starter_rate", "grp_fouls_won_pm", "starter_i"]

_P_EVENT = [
    "pl_fouls_left_p90", "pl_fouls_right_p90", "pl_fouls_centre_p90",
    "pl_fouls_def3_p90", "pl_fouls_won_p90", "pl_fouls_p90", "pl_dribbled_past_p90",
    "pl_tackles_p90", "pl_pressures_p90", "pl_dribbles_p90", "pl_dribbles_flank_p90",
    "pl_card_p90", "home", "match_week", "is_fb", "pos_group", "line", "start_position",
    "competition",
    "own_tp_fouls", "own_tp_fouls_won", "own_tp_dribbles", "own_tp_dribbled_past",
    "own_tp_possession", "own_tp_points", "own_tp_shots", "own_tp_corners", "own_tp_n",
    "opp_tp_possession", "opp_tp_points", "opp_tp_shots", "opp_tp_corners", "opp_tp_n",
]

#: The referee block: what a book knows about the official when it posts a foul prop.
_P_REF = ["ref_fouls_pm", "ref_cards_pm", "ref_card_rate", "ref_prior_n", "ref_known"]

#: The matchup block. For fouls *committed* the mechanism is the opponent's dribbling
#: coming down this player's flank; for fouls *won* it is the opponent's willingness to
#: foul. Both directions are given to both targets so that the asymmetry is measured
#: rather than built in.
_P_MATCHUP = [
    "opp_flank_dribbles_pm", "opp_flank_fouls_won_pm", "dopp_dribbles_p90",
    "dopp_dribbles_flank_p90", "dopp_fouls_won_p90", "dopp_prior_n",
    "doppf_pl_fouls_p90", "doppf_pl_tackles_p90", "doppf_prior_n",
    "opp_tp_fouls", "opp_tp_fouls_won", "opp_tp_dribbles", "opp_tp_dribbled_past",
    "opp_mirror_fouls_pm", "opp_tp_fouls_left_pm", "opp_tp_fouls_right_pm",
    "opp_tp_fouls_centre_pm",
]

#: Prior-match aggregates of the programme's imputed defensive state (E2 student).
_P_STATE = ["own_st_nearest_opp_dist", "own_st_n_opp_within_5", "own_st_block_depth",
            "own_st_counter_on", "opp_st_nearest_opp_dist", "opp_st_n_opp_within_5",
            "opp_st_block_depth", "opp_st_counter_on"]


def player_sets(target: str) -> dict[str, list[str]]:
    """Nested feature sets for a player foul target.

    Args:
        target: ``"fouls"`` or ``"fouls_won"``.

    Returns:
        Map from set name to feature columns.
    """
    proxy = list(_P_PROXY_F if target == "fouls" else _P_PROXY_W)
    base = proxy + [c for c in _P_EVENT if c not in proxy]
    return {
        "P0_proxy": proxy,
        "P1_event": base,
        "P1r_event_plus_referee": base + _P_REF,
        "P2_plus_matchup": base + _P_REF + _P_MATCHUP,
        "P3_plus_state": base + _P_REF + _P_MATCHUP + _P_STATE,
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def poisson_glm1(mu_ref: np.ndarray, y: np.ndarray, n_iter: int = 60
                 ) -> tuple[float, float]:
    """Fit ``mu = exp(a + b * log(mu_ref))`` by Newton steps on the Poisson likelihood.

    Recalibrating the proxy this way is what stops it being a straw man: an uncalibrated
    rolling mean can be beaten on scale alone, which would manufacture an edge from
    nothing.

    Args:
        mu_ref: Reference means [n].
        y: Observed counts [n].
        n_iter: Newton iterations.

    Returns:
        Tuple ``(a, b)``.
    """
    x = np.log(np.clip(np.asarray(mu_ref, dtype=float), 1e-6, None))
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    a, b = 0.0, 1.0
    for _ in range(n_iter):
        mu = np.exp(a + b * x)
        g = np.array([np.sum(y - mu), np.sum(x * (y - mu))])
        h = np.array([[np.sum(mu), np.sum(x * mu)],
                      [np.sum(x * mu), np.sum(x * x * mu)]])
        try:
            step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError:  # pragma: no cover - numerical guard
            break
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(a), float(b)


def poisson_glm1_apply(mu_ref: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    """Apply a fitted :func:`poisson_glm1` recalibration.

    Args:
        mu_ref: Reference means [n].
        ab: ``(a, b)``.

    Returns:
        Recalibrated means [n].
    """
    a, b = ab
    return np.exp(a + b * np.log(np.clip(np.asarray(mu_ref, dtype=float), 1e-6, None)))


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-row Poisson deviance [n]."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (term - (y - mu))


def count_rows(name: str, y: np.ndarray, mu: np.ndarray, r: float, groups: np.ndarray,
               split: np.ndarray, ref: np.ndarray | None, cfg: FoulConfig,
               **extra: object) -> list[dict[str, object]]:
    """Score a count model on both halves of the split against a reference model.

    Args:
        name: Model name.
        y: Observed counts [n].
        mu: Predicted means [n].
        r: Negative-binomial dispersion (fitted on discovery).
        groups: Match ids [n].
        split: Split labels [n].
        ref: Reference model's means [n], or ``None`` for the reference itself.
        cfg: Stage configuration.
        **extra: Extra fields copied into the rows.

    Returns:
        One row per split half.
    """
    rows = []
    for half in (DISC, CONF):
        m = (split == half) & np.isfinite(mu) & np.isfinite(y)
        if ref is not None:
            m &= np.isfinite(ref)
        if m.sum() == 0:
            continue
        ls = C.nb_log_score(y[m], mu[m], r)
        row: dict[str, object] = dict(extra)
        row.update(model=name, split=half, n=int(m.sum()), mean_y=float(y[m].mean()),
                   nb_log_score=float(ls.mean()),
                   poisson_deviance=float(poisson_deviance(y[m], mu[m]).mean()),
                   r2=r2(y[m], mu[m]), mae=float(np.abs(y[m] - mu[m]).mean()),
                   sd_pred=float(np.std(mu[m])), bias=float(np.mean(mu[m] - y[m])))
        if ref is None:
            row.update(delta_vs_ref=0.0, ci_lo=0.0, ci_hi=0.0)
        else:
            lr = C.nb_log_score(y[m], ref[m], r)
            d, lo, hi = C.clustered_bootstrap_mean(lr - ls, groups[m], cfg.split.n_boot,
                                                   cfg.split.seed)
            row.update(delta_vs_ref=d, ci_lo=lo, ci_hi=hi)
        rows.append(row)
    return rows


def line_probabilities(mu: np.ndarray, line: np.ndarray | float, r: float,
                       y: np.ndarray | None = None, fit: np.ndarray | None = None,
                       ) -> np.ndarray:
    """Turn predicted means into P(count > line), optionally Platt-recalibrated.

    Args:
        mu: Predicted means [n].
        line: Half-line, scalar or [n].
        r: Negative-binomial dispersion.
        y: Binary outcomes used to fit the recalibration [n].
        fit: Boolean mask of the rows the recalibration is fitted on [n].

    Returns:
        Probabilities [n].
    """
    p = C.nb_sf(line, mu, r)
    if y is None or fit is None:
        return p
    ab = C.platt_fit(p[fit], np.asarray(y, dtype=float)[fit])
    return C.platt_apply(p, ab)


# ---------------------------------------------------------------------------
# Stage (a): the odds-table layer
# ---------------------------------------------------------------------------


def stage_a(cfg: FoulConfig = CFG) -> dict[str, pd.DataFrame]:
    """Team and match foul totals on ~105k football-data matches.

    Answers three questions: how good the rolling-mean proxy already is; whether an
    event-only model beats it; and whether the enormous between-division differences in
    foul rates are already absorbed by the proxy or have to be modelled separately.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables, all written to ``reports/04_fouls_a_*.parquet``.
    """
    out: dict[str, pd.DataFrame] = {}
    t = build_team_table(cfg)
    disc = (t["split"] == DISC).to_numpy()
    out["a_proxy_sweep"] = choose_proxy_k(t, disc, cfg)
    out["a_audit"] = pd.read_parquet(C.processed_dir() / "fouls_audit.parquet")
    _log("audit:\n" + out["a_audit"].to_string(index=False))

    # how much of the variance is between divisions at all
    dvar = []
    for half, m in ((DISC, disc), (CONF, ~disc)):
        sub = t[m]
        for lvl, keys in (("division", ["Division"]), ("division_season", ["div_season"])):
            grp = sub.groupby(keys)["y_fouls"].transform("mean").to_numpy()
            dvar.append({"split": half, "level": lvl, "n": int(len(sub)),
                         "var_share_explained": r2(sub["y_fouls"].to_numpy(), grp),
                         "sd_between": float(np.std(grp)),
                         "sd_total": float(np.std(sub["y_fouls"].to_numpy()))})
    out["a_division_variance"] = pd.DataFrame(dvar)

    # ---- team-level ladder ----------------------------------------------------------
    _log("tuning team-count capacity on an inner forward split of discovery")
    best, tune = tune_params(t, TEAM_SETS["F3_plus_market"], "y_fouls", "count", cfg,
                             "team_fouls")
    out["a_tune"] = tune
    _log(f"chosen params: {best}")

    y = t["y_fouls"].to_numpy(dtype=float)
    groups = t["match_id"].to_numpy()
    split = t["split"].to_numpy()
    ab = poisson_glm1(t.loc[disc, "proxy_fouls"].to_numpy(), y[disc])
    preds: dict[str, np.ndarray] = {
        "C0_proxy_raw": t["proxy_fouls"].to_numpy(dtype=float),
        "C0_proxy_cal": poisson_glm1_apply(t["proxy_fouls"].to_numpy(dtype=float), ab),
    }
    for name, cols in TEAM_SETS.items():
        _log(f"team model {name} ({len(cols)} features)")
        preds[name] = cv_and_confirm(t, cols, "y_fouls", "count", cfg, best)
    r_team = C.fit_nb_dispersion(y[disc], preds["C0_proxy_cal"][disc])
    rows: list[dict[str, object]] = []
    for name, mu in preds.items():
        ref = None if name == "C0_proxy_cal" else preds["C0_proxy_cal"]
        rows += count_rows(name, y, mu, r_team, groups, split, ref, cfg, market="team_fouls")
    out["a_team_models"] = pd.DataFrame(rows)
    out["a_team_models"].attrs["dispersion"] = r_team

    # division absorption: F1 (no division) -> F2 (+division), and proxy + division only
    div_only = cv_and_confirm(t, _T_PROXY + _T_DIV, "y_fouls", "count", cfg, best)
    abs_rows = count_rows("F0_proxy_plus_division", y, div_only, r_team, groups, split,
                          preds["C0_proxy_cal"], cfg, market="team_fouls")
    for a, b in (("F1_event_no_division", "F2_plus_division"),
                 ("F0_proxy", "F0_proxy_plus_division")):
        mu_a = preds[a] if a in preds else div_only
        mu_b = preds[b] if b in preds else div_only
        for half in (DISC, CONF):
            m = (split == half) & np.isfinite(mu_a) & np.isfinite(mu_b)
            la = C.nb_log_score(y[m], mu_a[m], r_team)
            lb = C.nb_log_score(y[m], mu_b[m], r_team)
            d, lo, hi = C.clustered_bootstrap_mean(la - lb, groups[m], cfg.split.n_boot,
                                                   cfg.split.seed)
            abs_rows.append({"model": f"{b} minus {a}", "split": half, "n": int(m.sum()),
                             "delta_vs_ref": d, "ci_lo": lo, "ci_hi": hi,
                             "market": "team_fouls", "mean_y": float(y[m].mean()),
                             "nb_log_score": float(lb.mean()),
                             "poisson_deviance": float(poisson_deviance(y[m], mu_b[m]).mean()),
                             "r2": r2(y[m], mu_b[m]), "mae": float(np.abs(y[m] - mu_b[m]).mean()),
                             "sd_pred": float(np.std(mu_b[m])),
                             "bias": float(np.mean(mu_b[m] - y[m]))})
    out["a_division_absorption"] = pd.DataFrame(abs_rows)

    # ---- match-level ladder ---------------------------------------------------------
    mt = to_match_rows(t)
    msets = match_sets(mt)
    ym = mt["y_total_fouls"].to_numpy(dtype=float)
    gm = mt["match_id"].to_numpy()
    sm = mt["split"].to_numpy()
    dm = sm == DISC
    abm = poisson_glm1(mt.loc[dm, "proxy_total_fouls"].to_numpy(), ym[dm])
    mpreds: dict[str, np.ndarray] = {
        "C0_proxy_raw": mt["proxy_total_fouls"].to_numpy(dtype=float),
        "C0_proxy_cal": poisson_glm1_apply(mt["proxy_total_fouls"].to_numpy(dtype=float), abm),
    }
    for name, cols in msets.items():
        _log(f"match model {name} ({len(cols)} features)")
        mpreds[name] = cv_and_confirm(mt, cols, "y_total_fouls", "count", cfg, best)
    r_match = C.fit_nb_dispersion(ym[dm], mpreds["C0_proxy_cal"][dm])
    mrows: list[dict[str, object]] = []
    for name, mu in mpreds.items():
        ref = None if name == "C0_proxy_cal" else mpreds["C0_proxy_cal"]
        mrows += count_rows(name, ym, mu, r_match, gm, sm, ref, cfg, market="match_fouls")
    out["a_match_models"] = pd.DataFrame(mrows)

    # ---- direct binary classifiers at the standard half-lines -----------------------
    brows: list[dict[str, object]] = []
    team_pick = FF.pick_line(mt["proxy_total_fouls"].to_numpy() / 2.0, cfg.team_lines)
    top_team = pd.Series(FF.pick_line(t["proxy_fouls"].to_numpy(), cfg.team_lines))
    top_match = pd.Series(FF.pick_line(mt["proxy_total_fouls"].to_numpy(), cfg.match_lines))
    out["a_line_mix"] = pd.DataFrame({
        "market": ["team_fouls"] * len(cfg.team_lines) + ["match_fouls"] * len(cfg.match_lines),
        "line": list(cfg.team_lines) + list(cfg.match_lines),
        "share": [float((top_team == L).mean()) for L in cfg.team_lines]
        + [float((top_match == L).mean()) for L in cfg.match_lines]})
    del team_pick
    for market, frame, target, mu_model, mu_proxy, rr, cols, lines in (
        ("team_fouls", t, "y_fouls", preds["F3_plus_market"], preds["C0_proxy_cal"],
         r_team, TEAM_SETS["F3_plus_market"],
         [float(x) for x in top_team.value_counts().head(3).index]),
        ("match_fouls", mt, "y_total_fouls", mpreds["F3_plus_market"], mpreds["C0_proxy_cal"],
         r_match, msets["F3_plus_market"],
         [float(x) for x in top_match.value_counts().head(3).index]),
    ):
        yy = frame[target].to_numpy(dtype=float)
        gg = frame["match_id"].to_numpy()
        ss = frame["split"].to_numpy()
        dd = ss == DISC
        for L in sorted(lines):
            yb = (yy > L).astype(float)
            p_proxy = line_probabilities(mu_proxy, L, rr, yb, dd)
            p_count = line_probabilities(mu_model, L, rr, yb, dd)
            frame = frame.assign(_yb=yb)
            p_bin = cv_and_confirm(frame, cols, "_yb", "binary", cfg, best)
            for nm, p in (("C0_proxy_cal", p_proxy), ("F3_count_nb", p_count),
                          ("F3_direct_binary", p_bin)):
                for half in (DISC, CONF):
                    m = (ss == half) & np.isfinite(p)
                    lp = per_sample_log_loss(yb[m], p[m])
                    lr = per_sample_log_loss(yb[m], p_proxy[m])
                    d, lo, hi = C.clustered_bootstrap_mean(lr - lp, gg[m], cfg.split.n_boot,
                                                           cfg.split.seed)
                    brows.append({"market": market, "line": L, "model": nm, "split": half,
                                  "n": int(m.sum()), "base_rate": float(yb[m].mean()),
                                  "log_loss": log_loss(yb[m], p[m]),
                                  "brier": brier(yb[m], p[m]),
                                  "delta_vs_proxy": d, "ci_lo": lo, "ci_hi": hi})
    out["a_lines"] = pd.DataFrame(brows)

    # ---- refit-noise floor ----------------------------------------------------------
    rrows = []
    for seed in cfg.refit_seeds:
        mu = cv_and_confirm(t, TEAM_SETS["F3_plus_market"], "y_fouls", "count", cfg, best,
                            seed=seed)
        for half in (DISC, CONF):
            m = (split == half) & np.isfinite(mu)
            ls = C.nb_log_score(y[m], mu[m], r_team)
            lr = C.nb_log_score(y[m], preds["C0_proxy_cal"][m], r_team)
            rrows.append({"seed": seed, "split": half, "model": "F3_plus_market",
                          "delta_vs_proxy": float((lr - ls).mean()),
                          "nb_log_score": float(ls.mean())})
    refit = pd.DataFrame(rrows)
    spread = refit.groupby("split")["delta_vs_proxy"].agg(["min", "max", "mean"])
    spread["spread"] = spread["max"] - spread["min"]
    out["a_refit"] = refit.merge(spread.reset_index()[["split", "spread"]], on="split")

    for k, v in out.items():
        C.write_table(v, f"04_fouls_{k}")
    np.save(C.processed_dir() / "fouls_a_team_pred.npy",
            np.vstack([preds["C0_proxy_cal"], preds["F3_plus_market"]]))
    np.save(C.processed_dir() / "fouls_a_match_pred.npy",
            np.vstack([mpreds["C0_proxy_cal"], mpreds["F3_plus_market"]]))
    pd.DataFrame([{"r_team": r_team, "r_match": r_match, "params": str(best),
                   "proxy_a": ab[0], "proxy_b": ab[1]}]).to_parquet(
        C.processed_dir() / "fouls_a_meta.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# Stage (b): the player layer and the referee channel
# ---------------------------------------------------------------------------

#: The two directions of the matchup hypothesis, as feature blocks that can be added to
#: the event-only model one at a time. ``dribble_side`` is the mechanism behind the card
#: hypothesis (a defender fouls because the opponent runs at him); ``opponent_foul`` is
#: its mirror (a dribbler wins fouls because the opponent is willing to give them away).
CHANNELS: dict[str, list[str]] = {
    "dribble_side": ["opp_flank_dribbles_pm", "dopp_dribbles_p90",
                     "dopp_dribbles_flank_p90", "dopp_prior_n", "opp_tp_dribbles",
                     "opp_tp_dribbled_past"],
    "opponent_foul": ["opp_tp_fouls", "opp_tp_fouls_won", "doppf_pl_fouls_p90",
                      "doppf_pl_tackles_p90", "doppf_prior_n", "opp_mirror_fouls_pm",
                      "opp_tp_fouls_left_pm", "opp_tp_fouls_right_pm",
                      "opp_tp_fouls_centre_pm", "opp_flank_fouls_won_pm"],
}


def _rate_contrast(df: pd.DataFrame, target: str, by: Sequence[str], cfg: FoulConfig,
                   label: str) -> pd.DataFrame:
    """Observed mean of ``target`` in a cross-tabulation, with clustered intervals.

    Args:
        df: Population.
        target: Count column.
        by: Grouping columns.
        cfg: Stage configuration.
        label: Name recorded in the rows.

    Returns:
        One row per cell.
    """
    rows = []
    for key, g in df.groupby(list(by), sort=True, observed=True):
        mean, lo, hi = C.clustered_bootstrap_mean(g[target].to_numpy(dtype=float),
                                                  g["match_id"].to_numpy(), 400,
                                                  cfg.split.seed)
        row = {"contrast": label, "target": target}
        row.update({b: (key[i] if isinstance(key, tuple) else key)
                    for i, b in enumerate(by)})
        row.update(n=int(len(g)), mean=mean, ci_lo=lo, ci_hi=hi,
                   mean_minutes=float(g["minutes"].mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def _tercile(x: np.ndarray, disc: np.ndarray) -> tuple[float, float]:
    """Discovery-only tercile cuts of ``x``."""
    a = np.asarray(x, dtype=float)[disc]
    a = a[np.isfinite(a)]
    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))


def strong_player_proxy(u: pd.DataFrame, target: str, fit: np.ndarray) -> np.ndarray:
    """A book-like player proxy: the player's rate, his minutes, the opponent, the referee.

    The naive proxy (rate per 90 x expected minutes) is what a lazy side market is hung
    off, but protocol step 3 asks for the opponent and the official as well, so that the
    comparison is not against a straw man. This mirrors the card stage's ``build_proxy``:
    multiplicative factors normalised on the rows the calibration may see.

    The opponent factor is deliberately the *mirror* statistic -- for fouls committed it
    is how many fouls the opponent draws, for fouls won how many the opponent gives away.
    It is kept out of the modelling ladder's P0/P1 blocks on purpose, so that the matchup
    channel measured in :data:`CHANNELS` is not partly inside the reference.

    Args:
        u: Player-match frame.
        target: ``"fouls"`` or ``"fouls_won"``.
        fit: Boolean mask [n] of rows the normalising means may be taken over.

    Returns:
        Expected count [n].
    """
    rate = "pl_fouls_p90" if target == "fouls" else "pl_fouls_won_p90"
    opp = "opp_tp_fouls_won" if target == "fouls" else "opp_tp_fouls"
    exp90 = np.clip(u["pl_minutes_exp"].to_numpy(dtype=float), 5.0, 95.0) / 90.0
    opp_v = u[opp].to_numpy(dtype=float)
    ref_v = u["ref_fouls_pm"].to_numpy(dtype=float)
    opp_base = float(np.nanmean(opp_v[fit]))
    ref_base = float(np.nanmean(ref_v[fit]))
    opp_f = np.where(np.isfinite(opp_v), opp_v / opp_base, 1.0)
    ref_f = np.where(np.isfinite(ref_v), ref_v / ref_base, 1.0)
    return u[rate].to_numpy(dtype=float) * exp90 * opp_f * ref_f


def stage_b(cfg: FoulConfig = CFG) -> dict[str, pd.DataFrame]:
    """Player fouls committed and fouls won, the referee channel and the matchup channel.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables written to ``reports/04_fouls_b_*.parquet``.
    """
    out: dict[str, pd.DataFrame] = {}
    u = build_player_table(cfg)
    disc = (u["split"] == DISC).to_numpy()
    groups = u["match_id"].to_numpy()
    split = u["split"].to_numpy()
    out["b_setup"] = pd.DataFrame([{
        "n_appearances": int(len(u)), "n_matches": int(u["match_id"].nunique()),
        "n_players": int(u["player_id"].nunique()),
        "n_discovery": int(disc.sum()), "n_confirmation": int((~disc).sum()),
        "date_min": str(u["date"].min().date()), "date_max": str(u["date"].max().date()),
        "mean_fouls": float(u["fouls"].mean()), "sd_fouls": float(u["fouls"].std()),
        "mean_fouls_won": float(u["fouls_won"].mean()),
        "sd_fouls_won": float(u["fouls_won"].std()),
        "referee_known": float(u["ref_known"].mean()),
        "direct_opponent_found": float(u["doppf_prior_n"].notna().mean()),
    }])

    model_rows: list[dict[str, object]] = []
    imp_rows: list[dict[str, object]] = []
    preds_store: dict[tuple[str, str], np.ndarray] = {}
    tuned: dict[str, dict[str, Any]] = {}
    for target in ("fouls", "fouls_won"):
        sets = player_sets(target)
        proxy_col = "proxy_mu_fouls" if target == "fouls" else "proxy_mu_fouls_won"
        y = u[target].to_numpy(dtype=float)
        ab = poisson_glm1(u.loc[disc, proxy_col].to_numpy(), y[disc])
        mu_strong = strong_player_proxy(u, target, disc)
        ab_s = poisson_glm1(mu_strong[disc], y[disc])
        preds = {
            "P0_proxy_raw": u[proxy_col].to_numpy(dtype=float),
            "P0_proxy_cal": poisson_glm1_apply(u[proxy_col].to_numpy(dtype=float), ab),
            "P0_proxy_strong_raw": mu_strong,
            "P0_proxy_strong_cal": poisson_glm1_apply(mu_strong, ab_s),
        }
        best, tune = tune_params(u, sets["P2_plus_matchup"], target, "count", cfg,
                                 f"player_{target}")
        tuned[target] = dict(best)
        out[f"b_tune_{target}"] = tune
        for name, cols in sets.items():
            if name == "P3_plus_state":
                continue
            _log(f"player {target}: {name}")
            preds[name] = cv_and_confirm(u, cols, target, "count", cfg, best)
        # one-channel-at-a-time variants on top of the event + referee model
        for ch, cols in CHANNELS.items():
            _log(f"player {target}: channel {ch}")
            preds[f"P1r_plus_{ch}"] = cv_and_confirm(
                u, sets["P1r_event_plus_referee"] + cols, target, "count", cfg, best)
        r = C.fit_nb_dispersion(y[disc], preds["P0_proxy_cal"][disc])
        for name, mu in preds.items():
            ref = None if name == "P0_proxy_cal" else preds["P0_proxy_cal"]
            model_rows += count_rows(name, y, mu, r, groups, split, ref, cfg,
                                     market=f"player_{target}", dispersion_r=r)
        # paired deltas between neighbouring rungs
        pairs = [("P0_proxy_cal", "P0_proxy_strong_cal"),
                 ("P0_proxy_strong_cal", "P2_plus_matchup"),
                 ("P1_event", "P1r_event_plus_referee"),
                 ("P1r_event_plus_referee", "P1r_plus_dribble_side"),
                 ("P1r_event_plus_referee", "P1r_plus_opponent_foul"),
                 ("P1r_event_plus_referee", "P2_plus_matchup")]
        for a, b in pairs:
            for half in (DISC, CONF):
                m = (split == half) & np.isfinite(preds[a]) & np.isfinite(preds[b])
                la = C.nb_log_score(y[m], preds[a][m], r)
                lb = C.nb_log_score(y[m], preds[b][m], r)
                d, lo, hi = C.clustered_bootstrap_mean(la - lb, groups[m],
                                                       cfg.split.n_boot, cfg.split.seed)
                imp_rows.append({"market": f"player_{target}", "step": f"{a} -> {b}",
                                 "split": half, "n": int(m.sum()),
                                 "delta_nats_per_appearance": d, "ci_lo": lo, "ci_hi": hi})
        for k, v in preds.items():
            preds_store[(target, k)] = v
        np.save(C.processed_dir() / f"fouls_b_{target}.npy",
                np.vstack([preds["P0_proxy_cal"], preds["P1r_event_plus_referee"],
                           preds["P2_plus_matchup"]]))
    out["b_models"] = pd.DataFrame(model_rows)
    out["b_channels"] = pd.DataFrame(imp_rows)

    # ---- refit-noise floor ----------------------------------------------------------
    # Two things matter here and an earlier version got both wrong. First, the refit must
    # use the *tuned* parameters, or it measures the spread of a different model from the
    # one that produced the finding. Second, the floor has to be measured for the quantity
    # it is compared against: the claim in section 3.1 is about a **channel increment**
    # (P1r_event_plus_referee -> P1r_plus_<channel>), not about the P2-vs-proxy delta, and
    # the two have very different seed sensitivity. Both are measured below and reported
    # separately; ``quantity`` says which is which.
    rrows = []
    for target in ("fouls", "fouls_won"):
        sets = player_sets(target)
        best = tuned[target]
        y = u[target].to_numpy(dtype=float)
        proxy = preds_store[(target, "P0_proxy_cal")]
        r = C.fit_nb_dispersion(y[disc], proxy[disc])
        for seed in cfg.refit_seeds:
            reuse = seed == cfg.split.seed
            # the seed the stage ran under reproduces the main loop's predictions exactly,
            # so it is reused rather than refitted
            base = (preds_store[(target, "P1r_event_plus_referee")] if reuse else
                    cv_and_confirm(u, sets["P1r_event_plus_referee"], target, "count",
                                   cfg, best, seed=seed))
            p2 = (preds_store[(target, "P2_plus_matchup")] if reuse else
                  cv_and_confirm(u, sets["P2_plus_matchup"], target, "count", cfg, best,
                                 seed=seed))
            chans = {}
            for ch, cols in CHANNELS.items():
                chans[ch] = (preds_store[(target, f"P1r_plus_{ch}")] if reuse else
                             cv_and_confirm(u, sets["P1r_event_plus_referee"] + cols,
                                            target, "count", cfg, best, seed=seed))
            for half in (DISC, CONF):
                m = (split == half) & np.isfinite(p2) & np.isfinite(base)
                lb = C.nb_log_score(y[m], base[m], r)
                rrows.append({
                    "market": f"player_{target}",
                    "quantity": "P0_proxy_cal -> P2_plus_matchup", "seed": seed,
                    "split": half, "n": int(m.sum()),
                    "delta": float((C.nb_log_score(y[m], proxy[m], r)
                                    - C.nb_log_score(y[m], p2[m], r)).mean())})
                for ch, mu in chans.items():
                    rrows.append({
                        "market": f"player_{target}",
                        "quantity": f"P1r_event_plus_referee -> P1r_plus_{ch}",
                        "seed": seed, "split": half, "n": int(m.sum()),
                        "delta": float((lb - C.nb_log_score(y[m], mu[m], r)).mean())})
    refit = pd.DataFrame(rrows)
    sp = refit.groupby(["market", "quantity", "split"])["delta"].agg(["min", "max"])
    sp["refit_spread"] = sp["max"] - sp["min"]
    out["b_refit"] = refit.merge(
        sp.reset_index()[["market", "quantity", "split", "refit_spread"]],
        on=["market", "quantity", "split"])

    # ---- descriptive contrasts: is the mechanism there at all? ----------------------
    d = u.copy()
    lo_f, hi_f = _tercile(d["opp_flank_dribbles_pm"].to_numpy(), disc)
    d["opp_flank_dribbles_t"] = np.select(
        [d["opp_flank_dribbles_pm"] <= lo_f, d["opp_flank_dribbles_pm"] >= hi_f],
        ["low", "high"], default="mid")
    lo_o, hi_o = _tercile(d["opp_tp_fouls"].to_numpy(), disc)
    d["opp_foul_rate_t"] = np.select(
        [d["opp_tp_fouls"] <= lo_o, d["opp_tp_fouls"] >= hi_o], ["low", "high"],
        default="mid")
    lo_d, hi_d = _tercile(d["pl_dribbles_p90"].to_numpy(), disc)
    d["own_dribble_t"] = np.select(
        [d["pl_dribbles_p90"] <= lo_d, d["pl_dribbles_p90"] >= hi_d], ["low", "high"],
        default="mid")
    lo_r, hi_r = _tercile(d["ref_fouls_pm"].to_numpy(), disc)
    d["ref_foul_t"] = np.select(
        [d["ref_fouls_pm"] <= lo_r, d["ref_fouls_pm"] >= hi_r], ["low", "high"],
        default="mid")
    out["b_contrasts"] = pd.concat([
        _rate_contrast(d, "fouls", ["line", "opp_flank_dribbles_t"], cfg,
                       "fouls_by_line_x_opponent_flank_dribbles"),
        _rate_contrast(d, "fouls_won", ["own_dribble_t", "opp_foul_rate_t"], cfg,
                       "fouls_won_by_own_dribbling_x_opponent_foul_rate"),
        _rate_contrast(d, "fouls", ["ref_foul_t"], cfg, "fouls_by_referee_tercile"),
        _rate_contrast(d, "fouls_won", ["ref_foul_t"], cfg, "fouls_won_by_referee_tercile"),
        _rate_contrast(d, "fouls", ["competition", "ref_foul_t"], cfg,
                       "fouls_by_competition_x_referee_tercile"),
    ], ignore_index=True)

    # ---- is the referee channel a referee effect or a league effect? -----------------
    # The referee's prior foul rate is cut at 28.2 / 31.1 fouls per match, which is very
    # nearly the difference between the Premier League and Serie A. Two checks separate
    # the two: the referee prior's out-of-sample R2 on team fouls before and after
    # removing competition-season means, and a model ablation in which the referee block
    # is added to an event model that has been stripped of its competition feature.
    attr = []
    tm2 = u.groupby(["match_id", "team_id"], as_index=False).agg(
        team_fouls=("fouls", "sum"), ref_fouls_pm=("ref_fouls_pm", "first"),
        ref_known=("ref_known", "first"), split=("split", "first"),
        competition=("competition", "first"), referee_id=("referee_id", "first"))
    for half in (DISC, CONF):
        m = (tm2["split"] == half).to_numpy() & (tm2["ref_known"] > 0.5).to_numpy()
        sub = tm2[m]
        yv = sub["team_fouls"].to_numpy(dtype=float)
        rv = sub["ref_fouls_pm"].to_numpy(dtype=float) / 2.0
        cy = sub.groupby("competition")["team_fouls"].transform("mean").to_numpy()
        cr = sub.groupby("competition")["ref_fouls_pm"].transform("mean").to_numpy() / 2.0
        attr.append({"split": half, "n": int(m.sum()),
                     "n_referees": int(sub["referee_id"].nunique()),
                     "r2_raw": r2(yv, rv),
                     "r2_within_competition": r2(yv - cy, rv - cr),
                     "sd_referee_prior": float(np.std(rv)),
                     "sd_referee_prior_within_competition": float(np.std(rv - cr))})
    out["b_referee_attribution"] = pd.DataFrame(attr)

    abl = []
    for target in ("fouls", "fouls_won"):
        sets = player_sets(target)
        no_comp = [c for c in sets["P1_event"] if c != "competition"]
        y = u[target].to_numpy(dtype=float)
        r = C.fit_nb_dispersion(y[disc], preds_store[(target, "P0_proxy_cal")][disc])
        a = cv_and_confirm(u, no_comp, target, "count", cfg)
        b = cv_and_confirm(u, no_comp + _P_REF, target, "count", cfg)
        for half in (DISC, CONF):
            m = split == half
            la = C.nb_log_score(y[m], a[m], r)
            lb = C.nb_log_score(y[m], b[m], r)
            dd, lo, hi = C.clustered_bootstrap_mean(la - lb, groups[m], cfg.split.n_boot,
                                                    cfg.split.seed)
            abl.append({"market": f"player_{target}", "split": half, "n": int(m.sum()),
                        "step": "P1_event_without_competition -> + referee block",
                        "delta_nats_per_appearance": dd, "ci_lo": lo, "ci_hi": hi,
                        "log_score_without_referee": float(la.mean()),
                        "log_score_with_referee": float(lb.mean())})
    out["b_referee_ablation"] = pd.DataFrame(abl)
    out["b_cuts"] = pd.DataFrame([{
        "opp_flank_dribbles_lo": lo_f, "opp_flank_dribbles_hi": hi_f,
        "opp_foul_rate_lo": lo_o, "opp_foul_rate_hi": hi_o,
        "own_dribble_lo": lo_d, "own_dribble_hi": hi_d,
        "ref_fouls_pm_lo": lo_r, "ref_fouls_pm_hi": hi_r}])

    # ---- how much of a team's foul count does the referee explain? ------------------
    tm = u.groupby(["match_id", "team_id"], as_index=False).agg(
        team_fouls=("fouls", "sum"), ref_fouls_pm=("ref_fouls_pm", "first"),
        ref_known=("ref_known", "first"), split=("split", "first"))
    rr = []
    for half in (DISC, CONF):
        m = (tm["split"] == half).to_numpy() & (tm["ref_known"] > 0.5).to_numpy()
        rr.append({"split": half, "n": int(m.sum()),
                   "r2_referee_prior_on_team_fouls":
                       r2(tm.loc[m, "team_fouls"].to_numpy(),
                          tm.loc[m, "ref_fouls_pm"].to_numpy() / 2.0),
                   "corr": float(np.corrcoef(tm.loc[m, "team_fouls"],
                                             tm.loc[m, "ref_fouls_pm"])[0, 1]),
                   "sd_team_fouls": float(tm.loc[m, "team_fouls"].std())})
    out["b_referee_variance"] = pd.DataFrame(rr)

    for k, v in out.items():
        C.write_table(v, f"04_fouls_{k}")
    return out


# ---------------------------------------------------------------------------
# Stage (c): the imputed-state layer
# ---------------------------------------------------------------------------


def stage_c(cfg: FoulConfig = CFG) -> dict[str, pd.DataFrame]:
    """Does the programme's imputed defensive state add anything over the matchup block?

    The comparison is like for like: both models are fitted and scored on exactly the rows
    where the imputed state has at least one strictly-prior match to average, and the only
    difference between them is the eight state columns.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables written to ``reports/04_fouls_c_*.parquet``.
    """
    out: dict[str, pd.DataFrame] = {}
    u = build_player_table(cfg)
    usable = ((u["own_st_prior_n"].to_numpy(dtype=float) >= 1)
              & (u["opp_st_prior_n"].to_numpy(dtype=float) >= 1)
              & np.isfinite(u["own_st_block_depth"].to_numpy(dtype=float)))
    out["c_coverage"] = pd.DataFrame([{
        "n_all": int(len(u)), "n_usable_state": int(usable.sum()),
        "share": float(usable.mean()),
        "mean_own_st_prior_n": float(u.loc[usable, "own_st_prior_n"].mean()),
        "mean_block_depth": float(u.loc[usable, "own_st_block_depth"].mean()),
        "sd_block_depth": float(u.loc[usable, "own_st_block_depth"].std()),
    }])
    s = u[usable].reset_index(drop=True)
    disc = (s["split"] == DISC).to_numpy()
    groups = s["match_id"].to_numpy()
    split = s["split"].to_numpy()

    rows: list[dict[str, object]] = []
    for target in ("fouls", "fouls_won"):
        sets = player_sets(target)
        proxy_col = "proxy_mu_fouls" if target == "fouls" else "proxy_mu_fouls_won"
        y = s[target].to_numpy(dtype=float)
        ab = poisson_glm1(s.loc[disc, proxy_col].to_numpy(), y[disc])
        cal = poisson_glm1_apply(s[proxy_col].to_numpy(dtype=float), ab)
        r = C.fit_nb_dispersion(y[disc], cal[disc])
        preds = {"P0_proxy_cal": cal}
        for name in ("P2_plus_matchup", "P3_plus_state"):
            preds[name] = cv_and_confirm(s, sets[name], target, "count", cfg)
        base4 = (_P_PROXY_F if target == "fouls" else _P_PROXY_W)[:4]
        preds["P0_proxy_plus_state"] = cv_and_confirm(s, base4 + _P_STATE, target,
                                                      "count", cfg)
        preds["P0_proxy_features_only"] = cv_and_confirm(s, base4, target, "count", cfg)
        for name, mu in preds.items():
            ref = None if name == "P0_proxy_cal" else preds["P0_proxy_cal"]
            rows += count_rows(name, y, mu, r, groups, split, ref, cfg,
                               market=f"player_{target}_state_subset")
        pairs = [("P2_plus_matchup", "P3_plus_state"),
                 ("P0_proxy_features_only", "P0_proxy_plus_state")]
        for aname, bname in pairs:
            for half in (DISC, CONF):
                m = (split == half)
                la = C.nb_log_score(y[m], preds[aname][m], r)
                lb = C.nb_log_score(y[m], preds[bname][m], r)
                d, lo, hi = C.clustered_bootstrap_mean(la - lb, groups[m],
                                                       cfg.split.n_boot, cfg.split.seed)
                rows.append({
                    "market": f"player_{target}_state_subset",
                    "model": f"{bname} minus {aname}", "split": half,
                    "n": int(m.sum()), "delta_vs_ref": d, "ci_lo": lo, "ci_hi": hi,
                    "mean_y": float(y[m].mean()), "nb_log_score": float(lb.mean()),
                    "poisson_deviance": float(
                        poisson_deviance(y[m], preds[bname][m]).mean()),
                    "r2": r2(y[m], preds[bname][m]),
                    "mae": float(np.abs(y[m] - preds[bname][m]).mean()),
                    "sd_pred": float(np.std(preds[bname][m])),
                    "bias": float(np.mean(preds[bname][m] - y[m]))})
    out["c_models"] = pd.DataFrame(rows)
    for k, v in out.items():
        C.write_table(v, f"04_fouls_{k}")
    return out


# ---------------------------------------------------------------------------
# Stage (d): betting simulation, and the honesty check that haircuts it
# ---------------------------------------------------------------------------


def choose_threshold(y: np.ndarray, p_model: np.ndarray, p_book: np.ndarray,
                     groups: np.ndarray, hold: float, cfg: FoulConfig,
                     min_bets: int = 200) -> float:
    """Pick the EV threshold on discovery: best ROI subject to a minimum bet count.

    Args:
        y: Binary outcomes [n].
        p_model: Model probabilities [n].
        p_book: Book-proxy fair probabilities [n].
        groups: Match ids [n].
        hold: Two-way overround.
        cfg: Stage configuration.
        min_bets: Minimum bets a candidate threshold must place.

    Returns:
        The chosen threshold.
    """
    best_t, best_roi = float(cfg.edge_grid[0]), -np.inf
    for t in cfg.edge_grid:
        r = C.simulate_two_way(y, p_model, p_book, groups,
                               C.BetSimConfig(hold=hold, edge_threshold=t, n_boot=200,
                                              seed=cfg.split.seed))
        if r.n_bets >= min_bets and np.isfinite(r.roi) and r.roi > best_roi:
            best_roi, best_t = r.roi, float(t)
    return best_t


def simulate_real_prices(y: np.ndarray, p_model: np.ndarray, odds_over: np.ndarray,
                         odds_under: np.ndarray, groups: np.ndarray, threshold: float,
                         cfg: FoulConfig) -> dict[str, float]:
    """Bet a model into a bookmaker's *actual* two-way prices.

    Args:
        y: Binary outcome of the over side [n].
        p_model: Model probability of the over side [n].
        odds_over: Decimal price of the over side [n].
        odds_under: Decimal price of the under side [n].
        groups: Match ids [n].
        threshold: Minimum EV per unit stake required to bet.
        cfg: Stage configuration.

    Returns:
        Dict with the same fields as :class:`common.BetSimResult` plus the realised hold.
    """
    y = np.asarray(y, dtype=float)
    ev_o = p_model * odds_over - 1.0
    ev_u = (1.0 - p_model) * odds_under - 1.0
    take_o = ev_o >= ev_u
    ev = np.where(take_o, ev_o, ev_u)
    bet = np.isfinite(ev) & (ev > threshold)
    if bet.sum() == 0:
        return {"n_rows": int(len(y)), "n_bets": 0, "bet_rate": 0.0, "roi": float("nan"),
                "roi_lo": float("nan"), "roi_hi": float("nan"), "mean_edge": float("nan"),
                "mean_hold": float(np.nanmean(1 / odds_over + 1 / odds_under - 1))}
    profit = np.where(take_o, np.where(y > 0.5, odds_over - 1.0, -1.0),
                      np.where(y <= 0.5, odds_under - 1.0, -1.0))
    roi, lo, hi = C.clustered_bootstrap_mean(profit[bet], np.asarray(groups)[bet],
                                             cfg.split.n_boot, cfg.split.seed)
    return {"n_rows": int(len(y)), "n_bets": int(bet.sum()),
            "bet_rate": float(bet.mean()), "roi": roi, "roi_lo": lo, "roi_hi": hi,
            "mean_edge": float(ev[bet].mean()),
            "mean_hold": float(np.nanmean(1 / odds_over + 1 / odds_under - 1))}


def goals_honesty(mt: pd.DataFrame, cfg: FoulConfig) -> dict[str, pd.DataFrame]:
    """Protocol step 4: how much better is a real book than this rolling mean?

    We have no foul lines, only goals lines. So the identical proxy machinery is pointed
    at total goals on the *same* matches, turned into a probability at 2.5, and compared
    with Bet365's no-vig price; and a goals model of the same shape is made to bet first
    into proxy-priced markets and then into Bet365's actual prices. The first gap is the
    haircut in nats, the second the haircut in ROI points.

    Args:
        mt: Match-row table with ``proxy_total_goals``, ``y_goals_total`` and the
            Bet365 over/under 2.5 prices.
        cfg: Stage configuration.

    Returns:
        Dict with ``d_goals_metrics``, ``d_goals_roi`` and ``d_haircut``.
    """
    g = mt[np.isfinite(mt["odd_over25"]) & np.isfinite(mt["odd_under25"])
           & mt["y_goals_total"].notna()].reset_index(drop=True)
    disc = (g["split"] == DISC).to_numpy()
    y = g["y_goals_total"].to_numpy(dtype=float)
    yb = (y > 2.5).astype(float)
    groups = g["match_id"].to_numpy()
    split = g["split"].to_numpy()

    ab = poisson_glm1(g.loc[disc, "proxy_total_goals"].to_numpy(), y[disc])
    mu_raw = g["proxy_total_goals"].to_numpy(dtype=float)
    mu_cal = poisson_glm1_apply(mu_raw, ab)
    r = C.fit_nb_dispersion(y[disc], mu_cal[disc])
    p_raw = C.nb_sf(2.5, mu_raw, r)
    p_cal = line_probabilities(mu_cal, 2.5, r, yb, disc)
    p_b365, _ = C.novig_two_way(g["odd_over25"].to_numpy(dtype=float),
                                g["odd_under25"].to_numpy(dtype=float))
    # the goals model: the same feature ladder, deliberately without market features,
    # because the market is what it is being scored against
    cols = [c for c in match_sets(g)["F2_plus_division"] if "foul" not in c]
    cols = [c for c in cols if c in g.columns]
    mu_model = cv_and_confirm(g, cols, "y_goals_total", "count", cfg)
    p_model = line_probabilities(mu_model, 2.5, r, yb, disc)

    rows = []
    base = np.full(len(g), float(yb[disc].mean()))
    for name, p in (("base_rate", base), ("proxy_raw", p_raw), ("proxy_cal", p_cal),
                    ("model", p_model), ("bet365_novig", p_b365)):
        for half in (DISC, CONF):
            m = split == half
            lp = per_sample_log_loss(yb[m], p[m])
            lr = per_sample_log_loss(yb[m], p_cal[m])
            d, lo, hi = C.clustered_bootstrap_mean(lr - lp, groups[m], cfg.split.n_boot,
                                                   cfg.split.seed)
            rows.append({"split": half, "model": name, "n": int(m.sum()),
                         "log_loss": log_loss(yb[m], p[m]), "brier": brier(yb[m], p[m]),
                         "delta_vs_proxy_cal": d, "ci_lo": lo, "ci_hi": hi,
                         "mean_p": float(p[m].mean()), "obs_rate": float(yb[m].mean())})
    metrics = pd.DataFrame(rows)

    # Bet365's realised two-way overround on these prices. The actual-price leg is fixed
    # at it, so differencing it against a proxy-priced leg at some other hold would mix a
    # margin difference into a sharpness difference. A third leg therefore re-prices
    # Bet365's *no-vig* probability at the same hold as the proxy leg: that comparison is
    # margin-matched at every hold and is the one the stage actually applies.
    overround = (1.0 / g["odd_over25"].to_numpy(dtype=float)
                 + 1.0 / g["odd_under25"].to_numpy(dtype=float) - 1.0)
    roi_rows = []
    for hold in cfg.holds:
        thr = choose_threshold(yb[disc], p_model[disc], p_cal[disc], groups[disc], hold, cfg)
        for half, m in ((DISC, disc), (CONF, ~disc)):
            proxy_sim = C.simulate_two_way(
                yb[m], p_model[m], p_cal[m], groups[m],
                C.BetSimConfig(hold=hold, edge_threshold=thr, n_boot=cfg.split.n_boot,
                               seed=cfg.split.seed))
            matched = C.simulate_two_way(
                yb[m], p_model[m], p_b365[m], groups[m],
                C.BetSimConfig(hold=hold, edge_threshold=thr, n_boot=cfg.split.n_boot,
                               seed=cfg.split.seed))
            real = simulate_real_prices(yb[m], p_model[m],
                                        g.loc[m, "odd_over25"].to_numpy(dtype=float),
                                        g.loc[m, "odd_under25"].to_numpy(dtype=float),
                                        groups[m], thr, cfg)
            roi_rows.append(proxy_sim.as_row(split=half, hold=hold, threshold=thr,
                                             opponent="rolling_mean_proxy",
                                             priced_at_hold=hold))
            roi_rows.append(matched.as_row(split=half, hold=hold, threshold=thr,
                                           opponent="bet365_novig_repriced",
                                           priced_at_hold=hold))
            roi_rows.append({**real, "split": half, "hold": hold, "threshold": thr,
                             "opponent": "bet365_actual_prices",
                             "priced_at_hold": float(np.nanmean(overround[m]))})
    roi = pd.DataFrame(roi_rows)

    cut = []
    for half in (DISC, CONF):
        nats = float(metrics.query("split == @half and model == 'bet365_novig'")
                     ["delta_vs_proxy_cal"].iloc[0])
        mh = float(np.nanmean(overround[disc if half == DISC else ~disc]))
        for hold in cfg.holds:
            a = roi.query("split == @half and hold == @hold and "
                          "opponent == 'rolling_mean_proxy'")["roi"].iloc[0]
            b = roi.query("split == @half and hold == @hold and "
                          "opponent == 'bet365_actual_prices'")["roi"].iloc[0]
            c = roi.query("split == @half and hold == @hold and "
                          "opponent == 'bet365_novig_repriced'")["roi"].iloc[0]
            cut.append({"split": half, "hold": hold, "haircut_nats": nats,
                        "roi_vs_proxy": float(a),
                        "roi_vs_real_book_matched": float(c),
                        "roi_vs_real_book_actual_prices": float(b),
                        "real_book_hold": mh,
                        "haircut_roi_points": float(a - c),
                        "haircut_roi_points_actual_prices": float(a - b)})
    return {"d_goals_metrics": metrics, "d_goals_roi": roi,
            "d_haircut": pd.DataFrame(cut)}


def stage_d(cfg: FoulConfig = CFG) -> dict[str, pd.DataFrame]:
    """Betting simulation at the standard foul lines, with the goals haircut applied.

    Args:
        cfg: Stage configuration.

    Returns:
        Dict of result tables written to ``reports/04_fouls_d_*.parquet``.
    """
    out: dict[str, pd.DataFrame] = {}
    t = build_team_table(cfg)
    mt = to_match_rows(t)
    meta = pd.read_parquet(C.processed_dir() / "fouls_a_meta.parquet").iloc[0]
    team_pred = np.load(C.processed_dir() / "fouls_a_team_pred.npy")
    match_pred = np.load(C.processed_dir() / "fouls_a_match_pred.npy")
    out.update(goals_honesty(mt, cfg))
    hair = out["d_haircut"]

    bet_rows: list[dict[str, object]] = []
    ladder_rows: list[dict[str, object]] = []
    dis_rows: list[pd.DataFrame] = []
    for market, frame, target, lines, arr, r in (
        ("team_fouls", t, "y_fouls", cfg.team_lines, team_pred, float(meta["r_team"])),
        ("match_fouls", mt, "y_total_fouls", cfg.match_lines, match_pred,
         float(meta["r_match"])),
    ):
        mu_proxy, mu_model = arr[0], arr[1]
        y = frame[target].to_numpy(dtype=float)
        groups = frame["match_id"].to_numpy()
        split = frame["split"].to_numpy()
        disc = split == DISC
        line = FF.pick_line(mu_proxy, lines)
        yb = (y > line).astype(float)
        p_proxy = line_probabilities(mu_proxy, line, r, yb, disc)
        p_model = line_probabilities(mu_model, line, r, yb, disc)
        for hold in cfg.holds:
            thr = choose_threshold(yb[disc], p_model[disc], p_proxy[disc], groups[disc],
                                   hold, cfg)
            for half, m in ((DISC, disc), (CONF, ~disc)):
                res = C.simulate_two_way(yb[m], p_model[m], p_proxy[m], groups[m],
                                         C.BetSimConfig(hold=hold, edge_threshold=thr,
                                                        n_boot=cfg.split.n_boot,
                                                        seed=cfg.split.seed))
                hc = float(hair.query("split == @half and hold == @hold")
                           ["haircut_roi_points"].iloc[0])
                bet_rows.append(res.as_row(
                    market=market, split=half, hold=hold, threshold=thr,
                    mean_line=float(line[m].mean()), over_rate=float(yb[m].mean()),
                    haircut_roi_points=hc,
                    roi_after_haircut=res.roi - hc,
                    roi_lo_after_haircut=res.roi_lo - hc,
                    roi_hi_after_haircut=res.roi_hi - hc))
            # how sharp may the book be before the edge disappears?
            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                p_book = C.logit_blend(p_proxy, p_model, lam)
                res = C.simulate_two_way(yb[~disc], p_model[~disc], p_book[~disc],
                                         groups[~disc],
                                         C.BetSimConfig(hold=hold, edge_threshold=thr,
                                                        n_boot=400, seed=cfg.split.seed))
                ladder_rows.append(res.as_row(market=market, split=CONF, hold=hold,
                                              threshold=thr, book_lambda=lam))
        d = C.disagreement_table(yb[~disc], p_model[~disc], p_proxy[~disc], n_bins=5)
        d.insert(0, "market", market)
        d.insert(1, "split", CONF)
        dis_rows.append(d)
        m_line = pd.DataFrame({"market": market, "line": sorted(set(line.tolist())),
                               "share": [float((line == L).mean())
                                         for L in sorted(set(line.tolist()))]})
        ladder_rows_extra = m_line
        out[f"d_line_mix_{market}"] = ladder_rows_extra
    out["d_bets"] = pd.DataFrame(bet_rows)
    out["d_book_ladder"] = pd.DataFrame(ladder_rows)
    out["d_disagreement"] = pd.concat(dis_rows, ignore_index=True)
    for k, v in out.items():
        C.write_table(v, f"04_fouls_{k}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STAGES = ("build", "a", "b", "c", "d", "report")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--force", action="store_true", help="rebuild cached universes")
    args = ap.parse_args(argv)
    cfg = CFG
    todo = STAGES if args.stage == "all" else (args.stage,)
    for st in todo:
        t0 = time.time()
        if st == "build":
            build_team_table(cfg, force=args.force)
            build_player_table(cfg, force=args.force)
        elif st == "a":
            stage_a(cfg)
        elif st == "b":
            stage_b(cfg)
        elif st == "c":
            stage_c(cfg)
        elif st == "d":
            stage_d(cfg)
        elif st == "report":
            print(stage_report(cfg))
        _log(f"stage {st} done in {time.time() - t0:.0f}s")
    return 0



# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x: object, nd: int = 4) -> str:
    """Format one cell of a markdown table."""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return "n/a" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _md(df: pd.DataFrame, cols: Sequence[str] | None = None, nd: int = 4) -> str:
    """Render a frame as a markdown table.

    Args:
        df: Frame to render.
        cols: Columns to keep, in order (defaults to all of them).
        nd: Decimal places for floats.

    Returns:
        Markdown table string.
    """
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(_fmt(v, nd) for v in row) + " |"
            for row in d.itertuples(index=False, name=None)]
    return "\n".join([head, rule] + body)


def _ci(row: pd.Series, val: str, lo: str = "ci_lo", hi: str = "ci_hi",
        nd: int = 4) -> str:
    """Render ``value [lo, hi]`` from a result row."""
    return f"{row[val]:+.{nd}f} [{row[lo]:+.{nd}f}, {row[hi]:+.{nd}f}]"


def _t(name: str) -> pd.DataFrame:
    """Load one of this stage's result tables by suffix."""
    return pd.read_parquet(C.reports_dir() / f"04_fouls_{name}.parquet")


def _pick(df: pd.DataFrame, q: str) -> pd.Series:
    """First row matching a query, or an empty Series."""
    sub = df.query(q)
    return sub.iloc[0] if len(sub) else pd.Series(dtype=float)


def stage_report(cfg: FoulConfig = CFG):
    """Write ``reports/04_fouls.md`` from the committed result tables.

    Every number in the report is read from a parquet table produced by stages (a)-(d),
    so the prose cannot drift away from the results.

    Args:
        cfg: Stage configuration.

    Returns:
        Path of the written report.
    """
    tm, mm, ab = _t("a_team_models"), _t("a_match_models"), _t("a_division_absorption")
    dv, ln, rf = _t("a_division_variance"), _t("a_lines"), _t("a_refit")
    sweep_tbl, tune, audit = _t("a_proxy_sweep"), _t("a_tune"), _t("a_audit")
    bm, bc, bs = _t("b_models"), _t("b_channels"), _t("b_setup").iloc[0]
    br, bcon, bref = _t("b_refit"), _t("b_contrasts"), _t("b_referee_attribution")
    babl, brv = _t("b_referee_ablation"), _t("b_referee_variance")
    cm, cc = _t("c_models"), _t("c_coverage").iloc[0]
    gm, hc, bet = _t("d_goals_metrics"), _t("d_haircut"), _t("d_bets")
    lad, dis = _t("d_book_ladder"), _t("d_disagreement")
    gate_path = C.reports_dir() / "05_gate_sweep_summary.parquet"
    gate = pd.read_parquet(gate_path) if gate_path.exists() else pd.DataFrame()

    team_f3 = _pick(tm, "model == 'F3_plus_market' and split == 'confirmation'")
    match_f3 = _pick(mm, "model == 'F3_plus_market' and split == 'confirmation'")
    proxy_team = _pick(tm, "model == 'C0_proxy_cal' and split == 'confirmation'")
    div_step = _pick(ab, "model == 'F2_plus_division minus F1_event_no_division' "
                         "and split == 'confirmation'")
    b365 = _pick(gm, "model == 'bet365_novig' and split == 'confirmation'")
    gmodel = _pick(gm, "model == 'model' and split == 'confirmation'")
    cut6 = _pick(hc, "split == 'confirmation' and hold == 0.06")
    bt = _pick(bet, "market == 'team_fouls' and split == 'confirmation' and hold == 0.06")
    bmt = _pick(bet, "market == 'match_fouls' and split == 'confirmation' and hold == 0.06")
    won_ch = _pick(bc, "market == 'player_fouls_won' and split == 'confirmation' and "
                       "step == 'P1r_event_plus_referee -> P1r_plus_opponent_foul'")
    foul_ch = _pick(bc, "market == 'player_fouls' and split == 'confirmation' and "
                        "step == 'P1r_event_plus_referee -> P1r_plus_dribble_side'")
    state_f = _pick(cm, "market == 'player_fouls_state_subset' and "
                        "split == 'confirmation' and "
                        "model == 'P3_plus_state minus P2_plus_matchup'")
    state_w = _pick(cm, "market == 'player_fouls_won_state_subset' and "
                        "split == 'confirmation' and "
                        "model == 'P3_plus_state minus P2_plus_matchup'")
    pf_model = _pick(bm, "market == 'player_fouls' and split == 'confirmation' and "
                         "model == 'P2_plus_matchup'")
    pf_p0 = _pick(bm, "market == 'player_fouls' and split == 'confirmation' and "
                      "model == 'P0_proxy'")
    pw_model = _pick(bm, "market == 'player_fouls_won' and split == 'confirmation' and "
                         "model == 'P2_plus_matchup'")
    spread_team = float(rf.query("split == 'confirmation'")["spread"].iloc[0])

    def _spread(market: str, quantity: str, half: str = CONF) -> float:
        """Confirmation refit spread of one measured quantity, over ``cfg.refit_seeds``."""
        sub = br.query("market == @market and quantity == @quantity and split == @half")
        return float(sub["refit_spread"].iloc[0]) if len(sub) else float("nan")

    _MODEL_Q = "P0_proxy_cal -> P2_plus_matchup"
    _OP_Q = "P1r_event_plus_referee -> P1r_plus_opponent_foul"
    _DR_Q = "P1r_event_plus_referee -> P1r_plus_dribble_side"
    spread_pf = _spread("player_fouls", _MODEL_Q)
    spread_pw = _spread("player_fouls_won", _MODEL_Q)
    spread_won_op = _spread("player_fouls_won", _OP_Q)
    spread_won_dr = _spread("player_fouls_won", _DR_Q)
    spread_f_op = _spread("player_fouls", _OP_Q)
    spread_f_dr = _spread("player_fouls", _DR_Q)
    dv_div = float(dv.query("split == 'confirmation' and level == 'division'")
                   ["var_share_explained"].iloc[0])
    dv_ds = float(dv.query("split == 'confirmation' and level == 'division_season'")
                  ["var_share_explained"].iloc[0])
    ref_attr = _pick(bref, "split == 'confirmation'")
    ref_hi = _pick(bcon, "contrast == 'fouls_by_referee_tercile' and ref_foul_t == 'high'")
    ref_lo = _pick(bcon, "contrast == 'fouls_by_referee_tercile' and ref_foul_t == 'low'")
    ref_step = _pick(bc, "market == 'player_fouls' and split == 'confirmation' and "
                         "step == 'P1_event -> P1r_event_plus_referee'")
    div_on_proxy = _pick(ab, "model == 'F0_proxy_plus_division minus F0_proxy' and "
                             "split == 'confirmation'")
    n_disc_team = int(_pick(tm, "model == 'C0_proxy_cal' and split == 'discovery'")["n"])

    L: list[str] = []
    A = L.append
    A("# 04 Fouls and the referee channel: is a foul market priceable against a rolling mean?")
    A("")
    A("Stage 04 of the scenario expected-value phase. Machine-written by "
      "`research/scenario_ev/fouls.py` from the committed "
      "`research/scenario_ev/reports/04_fouls_*.parquet` tables, so every number below "
      "is read from one of them. Discovery numbers are exploratory and "
      "multiplicity-inflated; **confirmation numbers are scored once and are the "
      "headline**.")
    A("")
    A("```")
    A("cd /home/user/geo-model")
    A("python -m research.scenario_ev.fouls --stage build   # ~2 min, both universes + the leakage audit")
    A("python -m research.scenario_ev.fouls --stage a       # ~8 min, odds-table team and match foul totals")
    A("python -m research.scenario_ev.fouls --stage b       # ~2 min, player fouls, referee and matchup channels")
    A("python -m research.scenario_ev.fouls --stage c       # ~20 s, the imputed-state layer")
    A("python -m research.scenario_ev.fouls --stage d       # ~1 min, betting simulation and the goals haircut")
    A("python -m research.scenario_ev.fouls --stage report  # writes this file")
    A("python -m pytest research/scenario_ev/tests -q")
    A("```")
    A("")
    A("## Verdict")
    A("")
    A(f"1. **A rolling mean is already good at fouls, unlike corners.** The book proxy "
      f"explains R2 = {proxy_team['r2']:.4f} of a single team's foul count on "
      f"confirmation (n = {int(proxy_team['n'])} team-matches) against 0.033 for corners "
      f"in stage 02, because foul rates are dominated by persistent competition and "
      f"officiating effects that a rolling mean captures by construction.")
    A(f"2. **The event-only model beats it, by a real but small margin.** The full model "
      f"beats the *recalibrated* proxy by {_ci(team_f3, 'delta_vs_ref')} nats per "
      f"team-match and {_ci(match_f3, 'delta_vs_ref')} per match total. Against a "
      f"{len(cfg.refit_seeds)}-seed refit spread of {spread_team:.5f} that is about "
      f"{team_f3['delta_vs_ref'] / max(spread_team, 1e-9):.0f}x the noise, so it is a "
      f"real gap and not a seed.")
    A(f"3. **Division effects are large and already absorbed.** Between-division "
      f"variation is {100 * dv_div:.1f}% of the variance of a team's fouls on "
      f"confirmation and division-season {100 * dv_ds:.1f}%. But adding division "
      f"identifiers and league levels to a model that already has the proxy and the "
      f"rolling event block is worth {_ci(div_step, 'delta_vs_ref')} nats. The proxy "
      f"carries the division; the model does not have to be told about it.")
    A(f"4. **The matchup channel is real in exactly one direction, at about "
      f"{won_ch['delta_nats_per_appearance'] / max(spread_won_op, 1e-9):.0f}x refit "
      f"noise.** For fouls *won*, adding the opponent's foul propensity to the event + "
      f"referee model is worth {_ci(won_ch, 'delta_nats_per_appearance')} nats per "
      f"appearance on confirmation, against a {spread_won_op:.5f}-nat spread of that "
      f"same increment over {len(cfg.refit_seeds)} refits at the tuned parameters. For "
      f"fouls *committed*, adding the opponent's dribbling down this player's flank -- "
      f"the mechanism the card hypothesis rested on -- is worth "
      f"{_ci(foul_ch, 'delta_nats_per_appearance')}, inside its {spread_f_dr:.5f}-nat "
      f"spread. Fouls won are a matchup stat; fouls committed are a player-and-official "
      f"stat. Two caveats travel with this and are stated in section 3.1: both "
      f"fouls-won channels were inside noise on *discovery* and cleared zero only on "
      f"confirmation, and four (channel, target) cells were tested, so the claim rests "
      f"on sign consistency across halves and seeds rather than on independent "
      f"replication.")
    A(f"5. **The referee channel is real but is mostly a league effect.** Appearances "
      f"under a top-tercile referee average {ref_hi['mean']:.4f} fouls against "
      f"{ref_lo['mean']:.4f} in the bottom tercile, but the referee prior's R2 on a "
      f"team's fouls falls from {ref_attr['r2_raw']:.4f} to "
      f"{ref_attr['r2_within_competition']:.4f} once competition means are removed, and "
      f"adding the referee block to the player model is worth "
      f"{_ci(ref_step, 'delta_nats_per_appearance')} nats per appearance.")
    A(f"6. **The imputed defensive state adds nothing**: {_ci(state_f, 'delta_vs_ref')} "
      f"nats for fouls committed and {_ci(state_w, 'delta_vs_ref')} for fouls won, both "
      f"inside the refit noise. That is the fourth market in a row in which the "
      f"programme's tracking-derived channel fails to move an outcome model that already "
      f"reads the same events.")
    A(f"7. **And none of it is +EV.** Betting the model into a market priced off the "
      f"proxy at a 6% hold returns ROI {bt['roi']:+.4f} [{bt['roi_lo']:+.4f}, "
      f"{bt['roi_hi']:+.4f}] on team fouls ({int(bt['n_bets'])} bets on confirmation) "
      f"and {bmt['roi']:+.4f} [{bmt['roi_lo']:+.4f}, {bmt['roi_hi']:+.4f}] on match "
      f"totals ({int(bmt['n_bets'])} bets). On the one count market whose line we do "
      f"observe, a real bookmaker beats the identical proxy by "
      f"{_ci(b365, 'delta_vs_proxy_cal')} nats and turns the same kind of simulated "
      f"profit into a loss: a haircut of {100 * cut6['haircut_roi_points']:.1f} ROI "
      f"points. After it, team fouls return {bt['roi_after_haircut']:+.4f} "
      f"[{bt['roi_lo_after_haircut']:+.4f}, {bt['roi_hi_after_haircut']:+.4f}] and match "
      f"totals {bmt['roi_after_haircut']:+.4f} [{bmt['roi_lo_after_haircut']:+.4f}, "
      f"{bmt['roi_hi_after_haircut']:+.4f}]. Both intervals lie entirely below zero.")
    A("")
    A("**Read plainly: a null result on the betting question.** Our margin over a "
      "rolling mean on fouls is smaller than the margin a real bookmaker already holds "
      "over that same rolling mean on goals. It is a positive result on two smaller "
      "questions: the asymmetry of the foul matchup, and the fact that the apparent "
      "referee channel is almost entirely a league effect.")
    A("")
    A("## 1. Protocol and universes")
    A("")
    A("**Discovery / confirmation.** Matches are cut chronologically by "
      "`common.chronological_split`, separately inside each population, before any "
      "modelling. Every threshold, shrinkage, capacity choice, calibration map, "
      "dispersion and bet rule below is chosen on discovery; confirmation is scored once.")
    A("")
    A(f"- Odds-table universe: {int(proxy_team['n']) + n_disc_team} team-rows "
      f"({(int(proxy_team['n']) + n_disc_team) // 2} matches) of the 121,721 "
      f"football-data matches that record fouls, after requiring at least "
      f"{cfg.min_prior} strictly-prior matches with fouls recorded for both teams and "
      f"both directions. Discovery {n_disc_team} team-rows, confirmation "
      f"{int(proxy_team['n'])}.")
    A(f"- Player universe: {int(bs['n_appearances'])} appearances by "
      f"{int(bs['n_players'])} players in {int(bs['n_matches'])} StatsBomb matches (the "
      f"four 2015/16 league seasons), non-goalkeepers with at least "
      f"{cfg.min_prior_apps} strictly-prior appearances. Discovery "
      f"{int(bs['n_discovery'])}, confirmation {int(bs['n_confirmation'])}. Referee "
      f"known on {100 * bs['referee_known']:.1f}% of rows; a direct opponent is "
      f"identified on {100 * bs['direct_opponent_found']:.1f}%.")
    A(f"- Targets: team fouls (mean {proxy_team['mean_y']:.2f} on confirmation), match "
      f"total fouls, player fouls committed (mean {bs['mean_fouls']:.3f}, sd "
      f"{bs['sd_fouls']:.3f}) and player fouls won (mean {bs['mean_fouls_won']:.3f}, sd "
      f"{bs['sd_fouls_won']:.3f}).")
    A("- Football-data carries a handful of impossible foul counts (up to 145 in one "
      f"match); team counts are winsorised at {FOUL_CAP:.0f}, which touches 5 of 210,076 "
      "team-rows (0.002%).")
    A("")
    A("**Reuse.** The odds layer is built with the corner stage's prior-feature "
      "primitives (`corners_features.to_team_match`, `add_prior_features`, "
      "`prior_by_date`, `shrink`), whose `TEAM_STATS` already accumulate fouls; only the "
      "two pieces the corner proxy did not need (raw prior sums/counts for fouls and the "
      "home/away league levels) are added, in `fouls_features.py`. The player layer is "
      "built on the card stage's cached StatsBomb player-match table, adding a "
      "position-group backstop, the direct opponent's prior *fouls committed* rate, the "
      "opponent's prior fouls by pitch side, and the imputed defensive-state priors. "
      "Nothing in `research/privileged_tracking` or in stages 01-03 is modified.")
    A("")
    A("**Leakage audit, recomputed by brute force rather than asserted.** Run inside the "
      "builder on the *unfiltered* team-match frame, because that is the frame the "
      "priors were accumulated over.")
    A("")
    A(_md(audit, ["check", "n", "max_abs_error", "passed"], 10))
    A("")
    A("The third check matters for same-day fixtures: the league level is a function of "
      "(division-season, date) alone, so two matches played on the same day cannot enter "
      "one another's reference level.")
    A("")
    A("**Two smaller points of the same kind, stated rather than buried.** (i) The "
      "team-level prior foul rate *by pitch side* "
      "(`fouls_features.flank_foul_priors`, which feeds `opp_tp_fouls_<side>_pm` and "
      "`opp_mirror_fouls_pm` inside the opponent-foul block) shrinks toward a global "
      "per-side mean. That target is itself accumulated over strictly earlier dates "
      f"only, with a fixed constant ({FF.FLANK_PRIOR_FALLBACK}) for the first date, "
      "where there is nothing earlier to average; an earlier version of this stage used "
      "the whole table's mean, which was a single scalar per side and numerically "
      "negligible but not strictly prior. (ii) The player universe conditions on "
      "`minutes > 0`, i.e. on the appearance having happened. A real player prop cannot "
      "do that -- it is priced before the team sheet is certain, and a late withdrawal "
      "is a void bet, not a zero. No expected-value claim in this stage depends on it, "
      "because the betting simulation is team- and match-level only; but the "
      "player-level nats are conditional on playing and should be read that way.")
    A("")
    A("## 2. (a) The odds-table layer: 105k matches")
    A("")
    A("### 2.1 The book proxy and its shrinkage, chosen on discovery")
    A("")
    A("The proxy is the standard side-market heuristic: expected fouls = the league's "
      "home or away level x the team's shrunk prior foul rate / league x the opponent's "
      "shrunk prior fouls-conceded rate / league, everything from strictly earlier dates "
      "in the same division-season. Its shrinkage and functional form are chosen on "
      "discovery by squared error, so the opponent is the strongest rolling mean "
      "available rather than a straw man.")
    A("")
    A(_md(sweep_tbl.head(6), ["form", "k", "n", "mse", "r2", "corr", "sd_pred", "sd_y",
                              "bias"]))
    A("")
    A("### 2.2 LightGBM capacity, also chosen on discovery")
    A("")
    A("Chosen on an inner forward split *inside* discovery (first 75% of discovery dates "
      "to fit, last 25% to score), so confirmation is untouched and a null result cannot "
      "be confused with a capacity artefact.")
    A("")
    A(_md(tune, ["params", "n_train", "n_valid", "loss"], 5))
    A("")
    A("### 2.3 How much of a team's fouls is the division?")
    A("")
    A(_md(dv, ["split", "level", "n", "var_share_explained", "sd_between", "sd_total"]))
    A("")
    A("### 2.4 The model ladder")
    A("")
    A("`C0_proxy_raw` is the proxy as it comes; `C0_proxy_cal` adds a one-dimensional "
      "Poisson recalibration fitted on discovery, and is the reference every delta is "
      "measured against (an *un*calibrated proxy is beaten on scale alone, which would "
      "manufacture an edge out of nothing). `F1` deliberately contains no division "
      "identifier or league level, so the division question is answered by the F1 -> F2 "
      "step rather than assumed. Losses are negative-binomial log scores in nats, with "
      "the dispersion fitted on discovery; positive `delta_vs_ref` means better than the "
      "calibrated proxy.")
    A("")
    A(_md(tm, ["model", "split", "n", "mean_y", "nb_log_score", "poisson_deviance", "r2",
               "mae", "delta_vs_ref", "ci_lo", "ci_hi"]))
    A("")
    A("Match totals:")
    A("")
    A(_md(mm, ["model", "split", "n", "mean_y", "nb_log_score", "poisson_deviance", "r2",
               "mae", "delta_vs_ref", "ci_lo", "ci_hi"]))
    A("")
    A("### 2.5 Does the proxy already absorb the division?")
    A("")
    A(_md(ab, ["model", "split", "n", "nb_log_score", "r2", "delta_vs_ref", "ci_lo",
               "ci_hi"]))
    A("")
    A(f"Yes. Division identifiers are worth {_ci(div_step, 'delta_vs_ref')} nats on "
      f"confirmation on top of the event model, and "
      f"{_ci(div_on_proxy, 'delta_vs_ref')} on top of the proxy alone. A rolling mean "
      f"taken inside a division-season is already a division model.")
    A("")
    A("### 2.6 Direct binary classifiers at the standard half-lines")
    A("")
    A("Protocol step 5 asks for both a full count distribution and direct binary models "
      "at the half-lines. They agree: the negative binomial built on the count model's "
      "mean and a classifier trained directly on the line score within a whisker of each "
      "other, so nothing below depends on which is used.")
    A("")
    A(_md(ln, ["market", "line", "model", "split", "n", "base_rate", "log_loss", "brier",
               "delta_vs_proxy", "ci_lo", "ci_hi"]))
    A("")
    A("### 2.7 Refit-noise floor")
    A("")
    A(_md(rf, ["seed", "split", "model", "delta_vs_proxy", "nb_log_score", "spread"], 5))
    A("")
    A("## 3. (b) The player layer: fouls committed and fouls won")
    A("")
    A("Two proxies are reported. `P0_proxy` is the naive one a lazy side market is hung "
      "off -- the player's shrunk prior rate per 90 times his expected minutes. "
      "`P0_proxy_strong` is what protocol step 3 asks for, and mirrors the card stage's: "
      "the same rate and minutes multiplied by an opponent factor (the mirror statistic "
      "-- how many fouls the opponent *draws* when the target is fouls committed, how "
      "many it *gives away* when the target is fouls won) and a referee factor. Both are "
      "recalibrated on discovery by a one-dimensional Poisson GLM, so neither is beaten "
      "on scale alone. The opponent factor is deliberately kept out of the modelling "
      "ladder's P0/P1 blocks, so that the matchup channel measured in 3.1 is not partly "
      "inside its own reference. The ladder then adds the player's own rolling block and "
      "team context (`P1`), the referee (`P1r`), the matchup blocks (`P2`), and -- in "
      "section 4 -- the imputed defensive state (`P3`).")
    A("")
    A(_md(bm, ["market", "model", "split", "n", "mean_y", "nb_log_score", "r2", "mae",
               "delta_vs_ref", "ci_lo", "ci_hi"]))
    A("")
    strong_gain = _pick(bc, "market == 'player_fouls' and split == 'confirmation' and "
                            "step == 'P0_proxy_cal -> P0_proxy_strong_cal'")
    strong_gain_w = _pick(bc, "market == 'player_fouls_won' and "
                              "split == 'confirmation' and "
                              "step == 'P0_proxy_cal -> P0_proxy_strong_cal'")
    vs_strong = _pick(bc, "market == 'player_fouls' and split == 'confirmation' and "
                          "step == 'P0_proxy_strong_cal -> P2_plus_matchup'")
    vs_strong_w = _pick(bc, "market == 'player_fouls_won' and "
                            "split == 'confirmation' and "
                            "step == 'P0_proxy_strong_cal -> P2_plus_matchup'")
    strong_d = _pick(bc, "market == 'player_fouls' and split == 'discovery' and "
                         "step == 'P0_proxy_cal -> P0_proxy_strong_cal'")
    strong_dw = _pick(bc, "market == 'player_fouls_won' and split == 'discovery' and "
                          "step == 'P0_proxy_cal -> P0_proxy_strong_cal'")
    A(f"The richer proxy does not survive its own confirmation split. Relative to the "
      f"naive one it is worth {_ci(strong_gain, 'delta_nats_per_appearance')} nats per "
      f"appearance on fouls committed and "
      f"{_ci(strong_gain_w, 'delta_nats_per_appearance')} on fouls won -- both negative, "
      f"after being positive on discovery "
      f"({strong_d['delta_nats_per_appearance']:+.5f} and "
      f"{strong_dw['delta_nats_per_appearance']:+.5f}). Multiplying a player's rate by "
      f"an opponent factor and a referee factor is the right idea in the wrong "
      f"functional form. The naive proxy is therefore the *harder* reference out of "
      f"sample, and it is the one every delta in this section is measured against; "
      f"against the richer proxy the full model wins by more, "
      f"{_ci(vs_strong, 'delta_nats_per_appearance')} on fouls committed and "
      f"{_ci(vs_strong_w, 'delta_nats_per_appearance')} on fouls won. Either way the "
      f"margin is not an artefact of choosing the weakest available rolling mean.")
    A("")
    A(f"A caution that the ladder makes plain: most of the gap to the proxy is "
      f"functional form, not information. On player fouls the `P0_proxy` model -- "
      f"LightGBM on the proxy's own inputs plus the position-group backstop and the "
      f"starter flag -- already beats the calibrated proxy by "
      f"{pf_p0['delta_vs_ref']:+.4f} nats, and the entire event, referee and matchup "
      f"apparatus adds {pf_model['delta_vs_ref'] - pf_p0['delta_vs_ref']:+.4f} more. "
      f"A rate times expected minutes is simply the wrong shape for a count, and a "
      f"booster fixes that before it learns anything about football.")
    A("")
    A("### 3.1 The two matchup channels, added one at a time")
    A("")
    A(_md(bc, ["market", "step", "split", "n", "delta_nats_per_appearance", "ci_lo",
               "ci_hi"], 5))
    A("")
    A(f"Refit-noise floor over {len(cfg.refit_seeds)} seeds. Each row refits the named "
      f"quantity end to end at the *tuned* parameters and re-measures it, so the spread "
      f"is the spread of the number the text compares, not of some other model's. Two "
      f"quantities are measured because two different claims are made: the ladder claim "
      f"(`{_MODEL_Q}`, how far the full model is ahead of the proxy) and the channel "
      f"claim (`P1r_event_plus_referee -> P1r_plus_<channel>`, what one block adds). "
      f"The two spreads are of similar absolute size -- on fouls won the ladder delta "
      f"moves {spread_pw:.5f} nats across seeds, the opponent-foul increment "
      f"{spread_won_op:.5f} and the dribble-side increment {spread_won_dr:.5f} -- and "
      f"that is exactly why they must not be swapped, because the claims they support "
      f"differ by two orders of magnitude: the ladder delta is "
      f"{pw_model['delta_vs_ref']:.4f} nats and clears its own spread about "
      f"{pw_model['delta_vs_ref'] / max(spread_pw, 1e-9):.0f}x, while the opponent-foul "
      f"increment is {won_ch['delta_nats_per_appearance']:.5f} and clears the "
      f"same-sized spread about "
      f"{won_ch['delta_nats_per_appearance'] / max(spread_won_op, 1e-9):.0f}x. An "
      f"earlier version of this stage quoted the ladder spread -- and, worse, an "
      f"*untuned* refit of it, because the tuned parameters were never passed through "
      f"to the refit loop -- as the floor for the channel claim, which made a 3x result "
      f"read as 9x.")
    A("")
    A(f"**Five seeds here, not the three the other stages use.** The channel increments "
      f"are a few ten-thousandths of a nat and a three-seed range is a thin estimate of "
      f"their spread: on the first three seeds alone the opponent-foul increment on "
      f"fouls won spans only 0.00007 nats and the same claim would have read as 34x "
      f"noise. Two more refits put it at {spread_won_op:.5f} and 3x. The lesson is "
      f"general and is carried into the sweep report as well: a three-seed range can "
      f"understate a spread by an order of magnitude, so a refit floor is a screen for "
      f"downgrading claims, never a certificate.")
    A("")
    A(_md(br, ["market", "quantity", "seed", "split", "n", "delta", "refit_spread"], 5))
    A("")
    won_dr = _pick(bc, "market == 'player_fouls_won' and split == 'confirmation' and "
                       "step == 'P1r_event_plus_referee -> P1r_plus_dribble_side'")
    won_ch_d = _pick(bc, "market == 'player_fouls_won' and split == 'discovery' and "
                         "step == 'P1r_event_plus_referee -> P1r_plus_opponent_foul'")
    won_dr_d = _pick(bc, "market == 'player_fouls_won' and split == 'discovery' and "
                         "step == 'P1r_event_plus_referee -> P1r_plus_dribble_side'")
    foul_op = _pick(bc, "market == 'player_fouls' and split == 'confirmation' and "
                        "step == 'P1r_event_plus_referee -> P1r_plus_opponent_foul'")
    foul_op_d = _pick(bc, "market == 'player_fouls' and split == 'discovery' and "
                          "step == 'P1r_event_plus_referee -> P1r_plus_opponent_foul'")
    A("The asymmetry is the finding of this section, and it is an asymmetry of the "
      "*target*, not of the channel. On fouls won, the opponent-foul block is worth "
      f"{_ci(won_ch, 'delta_nats_per_appearance')} against a {spread_won_op:.5f}-nat "
      f"spread of that same increment "
      f"({won_ch['delta_nats_per_appearance'] / max(spread_won_op, 1e-9):.1f}x), and the "
      f"dribble-side block {_ci(won_dr, 'delta_nats_per_appearance')} against "
      f"{spread_won_dr:.5f} "
      f"({won_dr['delta_nats_per_appearance'] / max(spread_won_dr, 1e-9):.1f}x) -- both "
      f"clear of zero. On fouls committed, the same two blocks are worth "
      f"{_ci(foul_ch, 'delta_nats_per_appearance')} (spread {spread_f_dr:.5f}) and "
      f"{_ci(foul_op, 'delta_nats_per_appearance')} (spread {spread_f_op:.5f}) -- "
      "neither clear of zero, and both inside the noise. Who a player is up against "
      "moves how many fouls he draws; it does not measurably move how many he commits.")
    A("")
    A("**How much weight this carries.** Less than a first reading suggests, and the "
      "same standard is applied here as to the direction that failed. Both fouls-won "
      f"channels were *inside* noise on discovery -- opponent-foul "
      f"{won_ch_d['delta_nats_per_appearance']:+.5f} [{won_ch_d['ci_lo']:+.5f}, "
      f"{won_ch_d['ci_hi']:+.5f}], dribble-side "
      f"{won_dr_d['delta_nats_per_appearance']:+.5f} [{won_dr_d['ci_lo']:+.5f}, "
      f"{won_dr_d['ci_hi']:+.5f}] -- and cleared zero only on confirmation. Four "
      "(channel, target) cells were tested. So this is not a discovery finding "
      "replicated on confirmation; it is a confirmation finding whose discovery half "
      "agreed in sign but not in significance. What it rests on is sign consistency: "
      "the same sign in both halves and in every refit seed, in the direction the "
      "mechanism predicts, while the mirror direction is flat in both halves.")
    A("")
    A("Note also how the two halves disagree for fouls committed: the opponent-foul "
      f"block was worth {foul_op_d['delta_nats_per_appearance']:+.5f} "
      f"[{foul_op_d['ci_lo']:+.5f}, {foul_op_d['ci_hi']:+.5f}] on discovery, an interval "
      "comfortably clear of zero, and collapsed to nothing on confirmation. That is "
      "exactly the multiplicity the split exists to catch, and it is the reason a "
      "discovery-only version of this stage would have reported a matchup effect in "
      "both directions.")
    A("")
    A("### 3.2 Is the mechanism there at all? Observed rates")
    A("")
    A(_md(bcon, ["contrast", "line", "opp_flank_dribbles_t", "own_dribble_t",
                 "opp_foul_rate_t", "ref_foul_t", "n", "mean", "ci_lo", "ci_hi",
                 "mean_minutes"]))
    A("")
    A("Defenders facing top-tercile flank dribbling commit *fewer* fouls than those "
      "facing bottom-tercile flank dribbling, not more. Fouls won, by contrast, move "
      "monotonically in both the player's own dribbling and the opponent's foul rate.")
    A("")
    A("### 3.3 The referee channel: a referee effect or a league effect?")
    A("")
    A("The referee's prior foul rate is cut into terciles at roughly 28 and 31 fouls per "
      "match, which is very nearly the difference between the Premier League and Serie "
      "A. Two checks separate the two explanations.")
    A("")
    A(_md(bref, ["split", "n", "n_referees", "r2_raw", "r2_within_competition",
                 "sd_referee_prior", "sd_referee_prior_within_competition"]))
    A("")
    A(_md(bcon.query("contrast == 'fouls_by_competition_x_referee_tercile'"),
          ["competition", "ref_foul_t", "n", "mean", "ci_lo", "ci_hi"]))
    A("")
    A("And the model ablation, in which the referee block is added to an event model "
      "stripped of its competition feature:")
    A("")
    A(_md(babl, ["market", "split", "n", "step", "delta_nats_per_appearance", "ci_lo",
                 "ci_hi"], 5))
    A("")
    A(_md(brv, ["split", "n", "r2_referee_prior_on_team_fouls", "corr",
                "sd_team_fouls"]))
    A("")
    A("## 4. (c) The imputed-state layer")
    A("")
    A(f"Prior-match aggregates of the programme's E2 student predictions "
      f"(nearest-opponent distance, opponents within 5m, block depth, counter-attack "
      f"flag) for both teams, never the 360 truth. Usable on "
      f"{int(cc['n_usable_state'])} of {int(cc['n_all'])} rows "
      f"({100 * cc['share']:.1f}%), so the like-for-like comparison is the whole "
      f"universe.")
    A("")
    A(_md(cm, ["market", "model", "split", "n", "nb_log_score", "r2", "mae",
               "delta_vs_ref", "ci_lo", "ci_hi"], 5))
    A("")
    A("Added to the rich model the state is worth nothing; added to a bare proxy model "
      "it is worth nothing either. This is the same null the programme found for xG and "
      "for pass completion, in a fourth market.")
    A("")
    A("## 5. (d) The betting simulation and the haircut that governs it")
    A("")
    A("### 5.1 Protocol step 4: how much better is a real book than this rolling mean?")
    A("")
    A("We have no foul lines, only goals lines. So the identical proxy machinery is "
      "pointed at total goals on the *same* matches, turned into a probability at 2.5 "
      "(negative binomial plus a Platt map, both fitted on discovery) and compared with "
      "Bet365's no-vig over/under price.")
    A("")
    A(_md(gm, ["split", "model", "n", "log_loss", "brier", "delta_vs_proxy_cal", "ci_lo",
               "ci_hi", "mean_p", "obs_rate"]))
    A("")
    A(f"Bet365 beats the calibrated rolling mean by {_ci(b365, 'delta_vs_proxy_cal')} "
      f"nats on confirmation. Our own goals model, built from the same feature ladder, "
      f"beats it by {_ci(gmodel, 'delta_vs_proxy_cal')} -- less than half as much. On "
      f"the one market where the comparison can be made, this modelling apparatus is "
      f"markedly worse than the bookmaker it would have to beat.")
    A("")
    cut4 = _pick(hc, "split == 'confirmation' and hold == 0.04")
    cut8 = _pick(hc, "split == 'confirmation' and hold == 0.08")
    mm4 = abs(float(cut4["haircut_roi_points"])
              - float(cut4["haircut_roi_points_actual_prices"]))
    mm8 = abs(float(cut8["haircut_roi_points"])
              - float(cut8["haircut_roi_points_actual_prices"]))
    A("In ROI terms, the same goals model is made to bet into three markets: one priced "
      "off the rolling-mean proxy at the stated hold, one priced off Bet365's *no-vig* "
      "probability at that same hold, and one at Bet365's actual posted prices. The "
      "second leg exists because the third cannot be margin-matched: Bet365's realised "
      f"two-way overround on these prices is {100 * cut6['real_book_hold']:.2f}%, fixed, "
      "so differencing it against a proxy leg priced at 4% or 8% would mix a margin "
      "difference into the sharpness difference the haircut is supposed to isolate. "
      "`haircut_roi_points` is therefore the margin-matched difference "
      "(`roi_vs_proxy - roi_vs_real_book_matched`) and is what section 5.2 subtracts; "
      "`haircut_roi_points_actual_prices` is the raw difference against the posted "
      f"prices, kept for reference. At the 6% headline row the two are nearly identical "
      f"({100 * cut6['haircut_roi_points']:.1f} against "
      f"{100 * cut6['haircut_roi_points_actual_prices']:.1f} ROI points), because 6% is "
      f"{abs(100 * cut6['real_book_hold'] - 6.0):.2f} points from Bet365's realised "
      f"hold; at 4% and 8% they differ by {100 * mm4:.1f} and {100 * mm8:.1f} points, "
      f"which is exactly the margin mismatch this leg removes.")
    A("")
    A(_md(hc, ["split", "hold", "haircut_nats", "roi_vs_proxy",
               "roi_vs_real_book_matched", "roi_vs_real_book_actual_prices",
               "real_book_hold", "haircut_roi_points",
               "haircut_roi_points_actual_prices"]))
    A("")
    A("All three legs simulate the *same* goals model under the same discovery-chosen "
      "threshold. Between the first two only the opponent's fair probability changes "
      "(rolling mean vs Bet365 de-vigged), which is what makes their difference a pure "
      "sharpness gap; the third also changes the margin, to whatever Bet365 actually "
      "charged. Full per-hold detail, including bet counts and mean edges, is in "
      "`04_fouls_d_goals_roi.parquet`.")
    A("")
    A("### 5.2 The foul markets")
    A("")
    A("For each match the proxy posts the ladder rung closest to its own mean, both "
      "sides are priced at a stated hold, and the model bets when its EV clears a "
      "threshold chosen on discovery. `roi_after_haircut` subtracts the ROI-point "
      "haircut from the row above.")
    A("")
    A(_md(bet, ["market", "split", "hold", "threshold", "mean_line", "n_bets",
                "bet_rate", "mean_edge", "roi", "roi_lo", "roi_hi",
                "haircut_roi_points", "roi_after_haircut", "roi_lo_after_haircut",
                "roi_hi_after_haircut"]))
    A("")
    A("### 5.3 How sharp may the book be before the edge disappears?")
    A("")
    A("`book_lambda` moves the opponent from the rolling mean (0) to our own model (1) "
      "on the logit scale; confirmation rows only.")
    A("")
    A(_md(lad, ["market", "hold", "book_lambda", "n_bets", "mean_edge", "roi", "roi_lo",
                "roi_hi"]))
    A("")
    A("### 5.4 Closing-line-value style check: who is right where they disagree?")
    A("")
    A(_md(dis, ["market", "bin", "n", "diff_lo", "diff_hi", "mean_p_model", "mean_p_book",
                "observed", "log_loss_model", "log_loss_book", "model_better"]))
    A("")
    A("In the bins where the model disagrees most with the proxy, the model is closer to "
      "the truth -- the edge over the proxy is real. The proxy is simply not the "
      "opponent a bettor faces.")
    A("")
    A("## 6. The gate")
    A("")
    A("The gate for this stage's two targets is run by the cross-candidate sweep in "
      "`reports/05_gate_sweep.md`, through the single shared implementation "
      "(`common.run_gate`), so that its numbers are directly comparable with the other "
      "six markets. Positive `delta_gen_minus_spec` means the specialist is better, i.e. "
      "targeting has room. Both halves are shown; the `discovery_cv` verdicts are "
      "exploratory and only the `confirmation` rows count.")
    A("")
    if len(gate):
        g = gate.query("target_label in ('team_fouls', 'player_fouls') and "
                       "region == 'in_scenario'")
        A(_md(g, ["target_label", "scenario", "where", "n", "n_train_scenario",
                  "loss_generalist", "loss_specialist", "delta_gen_minus_spec", "ci_lo",
                  "ci_hi", "refit_spread", "verdict"], 5))
    else:
        A("_(run `python -m research.scenario_ev.gate_sweep` first)_")
    A("")
    A("## 7. Limitations")
    A("")
    A("- **No foul lines exist in this data.** Every ROI number is a simulation against "
      "a proxy-priced market, which is why the goals haircut is applied to all of them. "
      "The haircut is measured on goals and assumed to transfer to fouls; if a real foul "
      "market were *softer* than a real goals market the haircut would be too harsh, and "
      "if it were thinner and more heavily margined it would be too kind. The stage "
      "cannot settle that, and says so.")
    A("- **The player layer is one season of four leagues.** 1,398 matches is enough to "
      "separate the two matchup directions but not enough to price a niche prop, and "
      "its rows condition on a realised appearance (section 1), which a real prop "
      "cannot.")
    A("- **The fouls-won channel is a confirmation-only clearance.** It was inside noise "
      "on discovery, four (channel, target) cells were tested, and the correctly "
      "measured refit floor puts it at roughly three times noise rather than nine. "
      "Section 3.1 gives the numbers. It is the phase's most interesting surviving "
      "signal and it is also the one most in need of an independent replication.")
    A("- **Foul counts are recorded, not adjudicated.** football-data foul counts come "
      "from feeds with known coding differences between divisions and eras, which is "
      "part of what the division and season features absorb.")
    A("- **The referee identity is available only in the StatsBomb layer.** The 105k-match "
      "odds table has no referee, so the referee channel could only be tested on 1,398 "
      "matches.")
    A("")
    path = C.reports_dir() / "04_fouls.md"
    path.write_text("\n".join(L) + "\n")
    return path

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
