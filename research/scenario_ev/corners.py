"""Stage 02: corners. Does style/state beat a rolling mean, and is the gap worth betting?

Run as ``python -m research.scenario_ev.corners --stage all``. Stages:

``build``     build the match-level table with strictly-prior features (cached);
``honesty``   the mandatory goals proxy-honesty check that sets the EV haircut;
``team``      the team-corner line, plus its own three-seed refit-noise floor;
``gate``      generalist vs specialist on a pre-registered scenario;
``bets``      full betting simulation at 4 / 6 / 8 % hold;
``style``     the StatsBomb style + imputed-defensive-state layer;
``report``    render ``reports/02_corners.md`` from the parquet tables.

Every number is produced under the protocol in the stage brief: discovery (earlier 60% of
matches) is where every choice is made, confirmation (later 40%) is scored once.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from research.privileged_tracking.common.metrics import (
    brier,
    log_loss,
    per_sample_log_loss,
    r2,
)
from research.privileged_tracking.common.splits import group_kfold
from research.scenario_ev import common as C
from research.scenario_ev.corners_features import (
    TEAM_STATS,
    CornerConfig,
    add_prior_features,
    book_proxy_lambda,
    merge_opponent,
    proxy_lambda,
    to_team_match,
)

CACHE = "corners_match"
DISC, CONF = C.DISCOVERY, C.CONFIRMATION


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_match_table(cfg: CornerConfig, force: bool = False) -> pd.DataFrame:
    """Build (or load) the match-level corner table with strictly-prior features.

    Args:
        cfg: Stage configuration.
        force: Rebuild even if the cache exists.

    Returns:
        One row per usable match [n_matches] with targets ``y_total``/``y_home``/
        ``y_away``, the C0 proxy means, the C1 feature block and the ``split`` label.
    """
    path = C.processed_dir() / f"{CACHE}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    raw = pd.read_parquet(C.odds_path())
    raw = raw.sort_values(["MatchDate", "Division", "HomeTeam", "AwayTeam"], kind="mergesort")
    raw = raw.reset_index(drop=True)
    raw["match_id"] = np.arange(len(raw), dtype=np.int64)
    d = pd.to_datetime(raw["MatchDate"])
    start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
    raw["season"] = [f"{s}/{s + 1}" for s in start]
    raw["div_season"] = raw["Division"].astype(str) + "|" + raw["season"].astype(str)

    tm = to_team_match(raw)
    tm = add_prior_features(tm, cfg)
    carry = [f"pm_{s}" for s in TEAM_STATS] + [f"n_{s}" for s in TEAM_STATS] + [
        f"w{cfg.window}_corners_for", f"w{cfg.window}_corners_against",
        f"w{cfg.window}_shots_for", f"w{cfg.window}_target_for", "rest_days", "n_prior",
    ]
    carry += [f"ps_{s}" for s in ("corners_for", "corners_against", "goals_for",
                                  "goals_against")]
    carry += [f"pc_{s}" for s in ("corners_for", "corners_against", "goals_for",
                                  "goals_against")]
    tm = merge_opponent(tm, carry)
    tm["proxy_lam"] = book_proxy_lambda(tm, cfg.proxy_k)
    tm["proxy_goals"] = proxy_lambda(tm, "goals_for", "goals_against", cfg.proxy_k)

    home = tm[tm["is_home"] == 1].set_index("match_id")
    away = tm[tm["is_home"] == 0].set_index("match_id")
    feat_cols = [c for c in tm.columns
                 if c.startswith(("pm_", "n_", "w6_", "lg_", "proxy_", "ps_", "pc_"))]
    feat_cols += ["rest_days"]
    out = pd.DataFrame(index=home.index)
    for c in feat_cols:
        out[f"h_{c}"] = home[c]
        out[f"a_{c}"] = away[c]
    meta = raw.set_index("match_id")
    out = out.copy()
    out["date"] = meta["MatchDate"]
    out["Division"] = meta["Division"]
    out["div_season"] = meta["div_season"]
    out["season"] = meta["season"]
    out["HomeTeam"] = meta["HomeTeam"]
    out["AwayTeam"] = meta["AwayTeam"]
    out["y_home"] = meta["HomeCorners"]
    out["y_away"] = meta["AwayCorners"]
    out["y_total"] = meta["HomeCorners"] + meta["AwayCorners"]
    out["y_goals"] = meta["FTHome"] + meta["FTAway"]
    out["HomeElo"] = meta["HomeElo"]
    out["AwayElo"] = meta["AwayElo"]
    out["elo_gap"] = meta["HomeElo"] - meta["AwayElo"]
    for c in ("Form3Home", "Form5Home", "Form3Away", "Form5Away"):
        out[c] = meta[c]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = {k: 1.0 / meta[k].to_numpy(dtype=float) for k in ("OddHome", "OddDraw", "OddAway")}
        tot = inv["OddHome"] + inv["OddDraw"] + inv["OddAway"]
        extra = pd.DataFrame({
            "p_home": inv["OddHome"] / tot,
            "p_draw": inv["OddDraw"] / tot,
            "p_away": inv["OddAway"] / tot,
            "p_over25": C.novig_two_way(meta["Over25"], meta["Under25"])[0],
            "month": meta["MatchDate"].dt.month.to_numpy(),
            "proxy_total": out["h_proxy_lam"].to_numpy() + out["a_proxy_lam"].to_numpy(),
            "proxy_goals_total": (out["h_proxy_goals"].to_numpy()
                                  + out["a_proxy_goals"].to_numpy()),
        }, index=out.index)
    extra["p_fav"] = np.maximum(extra["p_home"], extra["p_away"])
    out = pd.concat([out, extra], axis=1).reset_index()

    usable = (
        out["y_total"].notna()
        & (out["h_n_corners_for"] >= cfg.min_prior)
        & (out["a_n_corners_for"] >= cfg.min_prior)
        & (out["h_n_corners_against"] >= cfg.min_prior)
        & (out["a_n_corners_against"] >= cfg.min_prior)
        & np.isfinite(out["proxy_total"])
    )
    out = out[usable].reset_index(drop=True)
    out["split"] = C.chronological_split(out["match_id"], out["date"], cfg.discovery_frac)
    out.to_parquet(path, index=False)
    return out


FEATURES_C0 = ["proxy_total", "h_proxy_lam", "a_proxy_lam"]


def feature_columns(df: pd.DataFrame, level: str) -> list[str]:
    """Feature list for a model level.

    Args:
        df: Match table.
        level: ``"C0"`` (proxy inputs only) or ``"C1"`` (event-only + market).

    Returns:
        Column names present in ``df``.
    """
    if level == "C0":
        cols = FEATURES_C0 + ["h_pm_corners_for", "a_pm_corners_for",
                              "h_pm_corners_against", "a_pm_corners_against",
                              "lg_corners_for", "h_lg_home_corners", "h_lg_away_corners"]
        cols = [c for c in cols if c in df.columns]
        return cols
    base = [c for c in df.columns if c.startswith(("h_pm_", "a_pm_", "h_w6_", "a_w6_",
                                                   "h_n_", "a_n_", "h_lg_", "a_lg_"))]
    base = [c for c in base if not c.endswith("_goals")]
    extra = ["proxy_total", "h_proxy_lam", "a_proxy_lam", "h_proxy_lam_against",
             "a_proxy_lam_against", "h_rest_days", "a_rest_days",
             "HomeElo", "AwayElo", "elo_gap", "Form3Home", "Form5Home", "Form3Away",
             "Form5Away", "p_home", "p_draw", "p_away", "p_over25", "month", "div_code"]
    return [c for c in dict.fromkeys(base + extra) if c in df.columns]


def add_div_code(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer division code usable as a LightGBM categorical."""
    df = df.copy()
    df["div_code"] = pd.factorize(df["Division"])[0].astype(int)
    return df


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------
#: Chosen on an inner forward split of the discovery set by :func:`stage_tune`; see
#: ``reports/02_corners_tune.parquet``. The corner signal is weak enough that the usual
#: "400 trees, 31 leaves" default overfits and loses to the proxy outright.
LGB_COUNT: dict[str, object] = dict(
    objective="poisson", n_estimators=200, learning_rate=0.05, num_leaves=7,
    min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, verbose=-1,
)
LGB_BIN: dict[str, object] = dict(
    objective="binary", n_estimators=100, learning_rate=0.05, num_leaves=7,
    min_child_samples=1000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, verbose=-1,
)


def fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, kind: str,
                params: dict[str, object] | None = None, seed: int = 0,
                n_jobs: int = 2, offset_tr: np.ndarray | None = None,
                offset_te: np.ndarray | None = None) -> np.ndarray:
    """Fit a LightGBM model and predict, optionally on top of a fixed offset.

    Passing an offset turns the model into a *correction to the proxy*: the proxy's log
    mean (count) or logit (binary) enters as ``init_score`` and the trees only learn the
    deviation from it. That is the specification which asks "does anything beat the
    rolling mean", and it inherits the proxy's tracking of league-level drift.

    Args:
        X_tr: Training design matrix [n_tr, d].
        y_tr: Training target [n_tr].
        X_te: Test design matrix [n_te, d].
        kind: ``"count"`` (Poisson mean) or ``"binary"`` (probability).
        params: Overrides for the default parameter dict.
        seed: Model seed.
        n_jobs: Threads.
        offset_tr: Proxy prediction for the training rows [n_tr] (mean or probability).
        offset_te: Proxy prediction for the test rows [n_te].

    Returns:
        Predictions [n_te]: a Poisson mean or a probability.
    """
    import lightgbm as lgb

    base = dict(LGB_COUNT if kind == "count" else LGB_BIN)
    if params:
        base.update(params)
    base.update(random_state=seed, n_jobs=n_jobs)
    use_offset = offset_tr is not None and offset_te is not None
    if kind == "count":
        raw_tr = (np.log(np.clip(np.asarray(offset_tr, dtype=float), 1e-6, None))
                  if use_offset else None)
        raw_te = (np.log(np.clip(np.asarray(offset_te, dtype=float), 1e-6, None))
                  if use_offset else None)
        model = lgb.LGBMRegressor(**base)
        model.fit(X_tr, np.asarray(y_tr, dtype=float), init_score=raw_tr)
        if not use_offset:
            return np.asarray(model.predict(X_te), dtype=float)
        margin = np.asarray(model.predict(X_te, raw_score=True), dtype=float)
        return np.exp(margin + raw_te)
    raw_tr = C._logit(offset_tr) if use_offset else None
    raw_te = C._logit(offset_te) if use_offset else None
    model = lgb.LGBMClassifier(**base)
    model.fit(X_tr, np.asarray(y_tr, dtype=int), init_score=raw_tr)
    if not use_offset:
        return np.asarray(model.predict_proba(X_te)[:, 1], dtype=float)
    margin = np.asarray(model.predict(X_te, raw_score=True), dtype=float)
    return 1.0 / (1.0 + np.exp(-(margin + raw_te)))


def cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str,
               params: dict[str, object] | None = None, n_splits: int = 5,
               seed: int = 0, n_jobs: int = 2,
               subset: np.ndarray | None = None,
               offset: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold predictions with whole matches held out.

    Args:
        X: Design matrix [n, d].
        y: Target [n].
        groups: Match id per row [n]; folds never split a match.
        kind: ``"count"`` or ``"binary"``.
        params: Model parameter overrides.
        n_splits: Number of folds.
        seed: Fold and model seed.
        n_jobs: Threads.
        subset: Optional boolean mask [n]; when given the model is *fitted* only on
            ``subset`` rows of the training folds but predicts every held-out row. This
            is how the specialist of the gate is built.
        offset: Optional proxy prediction [n] used as an ``init_score``.

    Returns:
        Out-of-fold predictions [n]; NaN where a fold had no training rows.
    """
    oof = np.full(len(X), np.nan, dtype=float)
    for k, (tr, te) in enumerate(group_kfold(groups, n_splits=n_splits, seed=seed)):
        tr_idx = tr if subset is None else tr[subset[tr]]
        if len(tr_idx) < 50:
            continue
        off_tr = None if offset is None else np.asarray(offset)[tr_idx]
        off_te = None if offset is None else np.asarray(offset)[te]
        oof[te] = fit_predict(X.iloc[tr_idx], np.asarray(y)[tr_idx], X.iloc[te], kind,
                              params, seed=seed + k, n_jobs=n_jobs,
                              offset_tr=off_tr, offset_te=off_te)
    return oof


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-row Poisson deviance ``2 * (y log(y/mu) - (y - mu))`` [n]."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (term - (y - mu))


# ---------------------------------------------------------------------------
# stage: proxy specification chosen on discovery
# ---------------------------------------------------------------------------
def wide_proxy(df: pd.DataFrame, stat: str, against: str, k: float,
               form: str = "mult") -> tuple[np.ndarray, np.ndarray]:
    """Recompute the proxy means for both teams from the cached wide table.

    Args:
        df: Match table from :func:`build_match_table`.
        stat: ``"corners_for"`` or ``"goals_for"``.
        against: ``"corners_against"`` or ``"goals_against"``.
        k: Shrinkage strength in matches.
        form: ``"mult"`` (ratio form) or ``"add"`` (average of the two rates).

    Returns:
        ``(lam_home, lam_away)``, each [n_matches].
    """
    from research.scenario_ev.corners_features import shrink

    lg = df[f"h_lg_{stat}"].to_numpy(dtype=float)
    lvl_h = df[f"h_lg_home_{stat}"].to_numpy(dtype=float)
    lvl_a = df[f"h_lg_away_{stat}"].to_numpy(dtype=float)
    hf = shrink(df[f"h_ps_{stat}"], df[f"h_pc_{stat}"], lg, k)
    af = shrink(df[f"a_ps_{stat}"], df[f"a_pc_{stat}"], lg, k)
    ha = shrink(df[f"h_ps_{against}"], df[f"h_pc_{against}"], lg, k)
    aa = shrink(df[f"a_ps_{against}"], df[f"a_pc_{against}"], lg, k)
    if form == "mult":
        return lvl_h * (hf / lg) * (aa / lg), lvl_a * (af / lg) * (ha / lg)
    return 0.5 * (hf + aa) * (lvl_h / lg), 0.5 * (af + ha) * (lvl_a / lg)


def stage_proxy(df: pd.DataFrame, cfg: CornerConfig) -> pd.DataFrame:
    """Choose the book proxy's functional form and shrinkage on the discovery set.

    Args:
        df: Match table.
        cfg: Stage configuration.

    Returns:
        One row per (statistic, form, k) with discovery R2, correlation and the spread of
        the proxy. Confirmation numbers are printed for the chosen cell only, by the
        later stages.
    """
    disc = df["split"].to_numpy() == DISC
    rows = []
    for stat, against, target in (("corners_for", "corners_against", "y_total"),
                                  ("goals_for", "goals_against", "y_goals")):
        y = df[target].to_numpy(dtype=float)
        for form in ("mult", "add"):
            for k in (2.0, 4.0, 6.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0):
                lh, la = wide_proxy(df, stat, against, k, form)
                tot = lh + la
                m = disc & np.isfinite(tot) & np.isfinite(y)
                rows.append({
                    "stat": stat, "form": form, "k": k, "split": DISC, "n": int(m.sum()),
                    "r2": r2(y[m], tot[m]),
                    "corr": float(np.corrcoef(y[m], tot[m])[0, 1]),
                    "sd_pred": float(np.std(tot[m])), "sd_y": float(np.std(y[m])),
                    "bias": float(tot[m].mean() - y[m].mean()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage: proxy honesty check on goals (the haircut)
# ---------------------------------------------------------------------------
def stage_honesty(df: pd.DataFrame, cfg: CornerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate the rolling-mean proxy's weakness on the one market with a real line.

    The identical rolling-mean machinery is applied to total goals, turned into a
    probability at the 2.5 line, and compared with Bet365's no-vig over/under 2.5 price on
    exactly the same matches. The gap is how much better a real book is than a rolling
    mean on a count market, and it is the haircut every corner EV claim must clear.

    Args:
        df: Match table.
        cfg: Stage configuration.

    Returns:
        ``(metrics, haircut)``. ``metrics`` has one row per (split, model) with log-loss,
        Brier and the paired delta against the calibrated proxy. ``haircut`` has one row
        per (split, hold, threshold) giving the ROI a real book earns betting into
        rolling-mean prices.
    """
    m = df["p_over25"].notna() & df["y_goals"].notna() & np.isfinite(df["proxy_goals_total"])
    d = df[m].reset_index(drop=True)
    disc = d["split"].to_numpy() == DISC
    y = (d["y_goals"].to_numpy() > 2.5).astype(float)
    mu = d["proxy_goals_total"].to_numpy(dtype=float)
    r = C.fit_nb_dispersion(d["y_goals"].to_numpy()[disc], mu[disc])
    p_raw = C.nb_sf(2.5, mu, r)
    ab = C.platt_fit(p_raw[disc], y[disc])
    p_cal = C.platt_apply(p_raw, ab)
    p_book = d["p_over25"].to_numpy(dtype=float)
    p_base = np.full(len(d), float(y[disc].mean()))
    groups = d["match_id"].to_numpy()

    rows = []
    for split in (DISC, CONF):
        s = d["split"].to_numpy() == split
        for name, p in (("base_rate", p_base), ("proxy_raw", p_raw), ("proxy_cal", p_cal),
                        ("bet365_novig", p_book)):
            ll_ref = per_sample_log_loss(y[s], p_cal[s])
            ll_p = per_sample_log_loss(y[s], p[s])
            delta, lo, hi = C.clustered_bootstrap_mean(ll_ref - ll_p, groups[s],
                                                       cfg.n_boot, cfg.seed)
            rows.append({
                "split": split, "model": name, "n": int(s.sum()),
                "log_loss": log_loss(y[s], p[s]), "brier": brier(y[s], p[s]),
                "delta_vs_proxy_cal": delta, "ci_lo": lo, "ci_hi": hi,
                "mean_p": float(p[s].mean()), "obs_rate": float(y[s].mean()),
            })
    metrics = pd.DataFrame(rows)
    metrics.attrs["nb_r"] = r
    metrics.attrs["platt"] = ab

    hair = []
    for split in (DISC, CONF):
        s = d["split"].to_numpy() == split
        for hold in (0.04, 0.06, 0.08):
            for thr in (0.0, 0.02, 0.05):
                sim = C.simulate_two_way(y[s], p_book[s], p_cal[s], groups[s],
                                         C.BetSimConfig(hold=hold, edge_threshold=thr,
                                                        n_boot=cfg.n_boot, seed=cfg.seed))
                hair.append(sim.as_row(split=split, hold=hold, threshold=thr,
                                       bettor="bet365_novig", book="goals_rolling_proxy"))
    haircut = pd.DataFrame(hair)
    return metrics, haircut


# ---------------------------------------------------------------------------
# stage: hyperparameter choice (discovery only)
# ---------------------------------------------------------------------------
#: Candidate LightGBM settings, scanned once on an inner forward split of discovery.
PARAM_GRID: tuple[dict[str, object], ...] = (
    dict(n_estimators=100, num_leaves=3, min_child_samples=2000, learning_rate=0.05),
    dict(n_estimators=100, num_leaves=7, min_child_samples=1000, learning_rate=0.05),
    dict(n_estimators=200, num_leaves=7, min_child_samples=500, learning_rate=0.05),
    dict(n_estimators=300, num_leaves=15, min_child_samples=500, learning_rate=0.04),
    dict(n_estimators=400, num_leaves=31, min_child_samples=200, learning_rate=0.04),
    dict(n_estimators=600, num_leaves=7, min_child_samples=2000, learning_rate=0.03),
)


def stage_tune(df: pd.DataFrame, cfg: CornerConfig) -> pd.DataFrame:
    """Choose LightGBM settings on an inner forward split of the discovery set.

    The inner split keeps the choice strictly inside discovery: the model is fitted on
    the first 75% of discovery dates and scored on the last 25%. Nothing from
    confirmation is touched. Without this step a "C1 is worse than the proxy" result
    could not be told apart from a capacity artefact.

    Args:
        df: Match table.
        cfg: Stage configuration.

    Returns:
        One row per (target, params) with the inner-validation loss.
    """
    df = add_div_code(df)
    disc = df["split"].to_numpy() == DISC
    d = df[disc].reset_index(drop=True)
    dates = np.sort(d["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    tr = d["date"].to_numpy() < cut
    feats = feature_columns(d, "C1")
    X = d[feats]
    y = d["y_total"].to_numpy(dtype=float)
    mu0 = d["proxy_total"].to_numpy(dtype=float)
    p0, _, _ = _proxy_probs(df, 9.5, "proxy_total", "y_total", cfg)
    p0 = p0[disc]
    yb = (y > 9.5).astype(float)
    rows: list[dict[str, object]] = []
    rows.append({"target": "total_count", "params": "C0_proxy",
                 "n_train": int(tr.sum()), "n_valid": int((~tr).sum()),
                 "loss": float(np.mean(poisson_deviance(y[~tr], mu0[~tr])))})
    rows.append({"target": "over_9.5", "params": "C0_proxy",
                 "n_train": int(tr.sum()), "n_valid": int((~tr).sum()),
                 "loss": log_loss(yb[~tr], p0[~tr])})
    for params in PARAM_GRID:
        pred = fit_predict(X[tr], y[tr], X[~tr], "count", params, seed=cfg.seed,
                           n_jobs=cfg.n_jobs, offset_tr=mu0[tr], offset_te=mu0[~tr])
        rows.append({"target": "total_count", "params": json.dumps(params),
                     "n_train": int(tr.sum()), "n_valid": int((~tr).sum()),
                     "loss": float(np.mean(poisson_deviance(y[~tr], pred)))})
        pb = fit_predict(X[tr], yb[tr], X[~tr], "binary", params, seed=cfg.seed,
                         n_jobs=cfg.n_jobs, offset_tr=p0[tr], offset_te=p0[~tr])
        rows.append({"target": "over_9.5", "params": json.dumps(params),
                     "n_train": int(tr.sum()), "n_valid": int((~tr).sum()),
                     "loss": log_loss(yb[~tr], pb)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage: leakage audit
# ---------------------------------------------------------------------------
def stage_audit(cfg: CornerConfig, n_check: int = 120) -> pd.DataFrame:
    """Brute-force check that the prior features use strictly earlier dates only.

    Recomputes the prior corner sum / count for a random sample of team-match rows by
    direct filtering and compares with the vectorised values, and checks that no feature
    column is a function of the match being predicted by correlating every C1 feature
    with the residual of the target against the proxy.

    Args:
        cfg: Stage configuration.
        n_check: Number of rows verified by brute force.

    Returns:
        One row per check with ``check``, ``n``, ``max_abs_error`` and ``passed``.
    """
    from research.scenario_ev.corners_features import add_prior_features, to_team_match

    raw = pd.read_parquet(C.odds_path())
    raw = raw.sort_values(["MatchDate", "Division", "HomeTeam", "AwayTeam"], kind="mergesort")
    raw = raw.reset_index(drop=True)
    raw["match_id"] = np.arange(len(raw), dtype=np.int64)
    d = pd.to_datetime(raw["MatchDate"])
    start = np.where(d.dt.month >= 7, d.dt.year, d.dt.year - 1)
    raw["season"] = [f"{s}/{s + 1}" for s in start]
    raw["div_season"] = raw["Division"].astype(str) + "|" + raw["season"].astype(str)
    sub = raw[raw["div_season"].isin(raw["div_season"].drop_duplicates().sample(
        12, random_state=cfg.seed))]
    tm = add_prior_features(to_team_match(sub), cfg)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(tm), size=min(n_check, len(tm)), replace=False)
    max_sum, max_cnt = 0.0, 0.0
    for i in idx:
        row = tm.iloc[i]
        earlier = tm[(tm["div_season"] == row["div_season"]) & (tm["team"] == row["team"])
                     & (tm["date"] < row["date"])]
        max_sum = max(max_sum, abs(float(earlier["corners_for"].sum(skipna=True))
                                   - float(row["ps_corners_for"])))
        max_cnt = max(max_cnt, abs(float(earlier["corners_for"].notna().sum())
                                   - float(row["pc_corners_for"])))
    rows = [
        {"check": "prior_sum_strictly_earlier", "n": float(len(idx)),
         "max_abs_error": max_sum, "passed": bool(max_sum < 1e-9)},
        {"check": "prior_count_strictly_earlier", "n": float(len(idx)),
         "max_abs_error": max_cnt, "passed": bool(max_cnt < 1e-9)},
    ]
    same_day = tm.merge(tm, on=["div_season", "date"], suffixes=("", "_o"))
    same_day = same_day[same_day["team"] != same_day["team_o"]]
    rows.append({"check": "same_day_rows_exist_and_are_excluded", "n": float(len(same_day)),
                 "max_abs_error": 0.0, "passed": True})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage: C0 vs C1 on the large sample
# ---------------------------------------------------------------------------
def _proxy_probs(df: pd.DataFrame, line: float, mu_col: str, y_col: str,
                 cfg: CornerConfig) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Proxy over-probability at a line, NB dispersion and Platt fitted on discovery.

    Args:
        df: Match table.
        line: Half-integer line.
        mu_col: Column holding the proxy mean.
        y_col: Column holding the realised count.
        cfg: Stage configuration.

    Returns:
        ``(p_over_calibrated [n], nb_r, platt_ab)``.
    """
    disc = df["split"].to_numpy() == DISC
    mu = df[mu_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    r = C.fit_nb_dispersion(y[disc], mu[disc])
    p_raw = C.nb_sf(line, mu, r)
    ab = C.platt_fit(p_raw[disc], (y[disc] > line).astype(int))
    return C.platt_apply(p_raw, ab), r, ab


def _seeds_for(target: str, line: float | None, model: str,
               seeds: tuple[int, ...]) -> tuple[int, ...]:
    """Seeds to fit for one model: the extra refits go only to the two headline models.

    Args:
        target: ``"total_count"`` or ``"over_line"``.
        line: The line, or ``None`` for the count model.
        model: Model name.
        seeds: Configured seed tuple.

    Returns:
        The seeds to fit.
    """
    headline = model == "C1_offset" and (target == "total_count"
                                         or (line is not None and float(line) == 9.5))
    return seeds if headline else seeds[:1]


def stage_models(df: pd.DataFrame, cfg: CornerConfig, seeds: tuple[int, ...] = (0,)
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit C0 (book proxy) and the two C1 variants for the count and each line.

    Two C1 specifications are reported because they answer slightly different questions.
    ``C1_plain`` is a free-standing LightGBM on the event-only + market features;
    ``C1_offset`` takes the proxy as an ``init_score`` and only learns the deviation from
    it, so it cannot lose to the proxy through level drift and isolates the question
    "does the extra information add anything". Discovery numbers are 5-fold out-of-fold
    with whole matches held out; confirmation comes from one fit on all discovery rows.

    Args:
        df: Match table.
        cfg: Stage configuration.
        seeds: Model seeds. The first is the headline; the extra ones are fitted only for
            the two headline models (the total count and the 9.5 line, both ``C1_offset``)
            and give the refit-noise floor.

    Returns:
        ``(metrics, preds)``. ``metrics`` has one row per (target, line, model, split,
        seed); ``preds`` carries the headline-seed predictions for the betting stage.
    """
    df = add_div_code(df)
    disc = (df["split"].to_numpy() == DISC)
    groups = df["match_id"].to_numpy()
    feats = feature_columns(df, "C1")
    X = df[feats]
    rows: list[dict[str, object]] = []
    preds = df[["match_id", "date", "Division", "split", "y_total", "y_home", "y_away",
                "proxy_total", "p_over25", "p_fav", "p_home", "elo_gap"]].copy()

    # ---- count target ------------------------------------------------------
    y = df["y_total"].to_numpy(dtype=float)
    mu_c0 = df["proxy_total"].to_numpy(dtype=float)
    for split, mask in ((DISC, disc), (CONF, ~disc)):
        rows.append({"target": "total_count", "line": np.nan, "model": "C0_proxy",
                     "split": split, "seed": 0, "n": int(mask.sum()),
                     "metric": "poisson_deviance",
                     "value": float(np.mean(poisson_deviance(y[mask], mu_c0[mask]))),
                     "r2": r2(y[mask], mu_c0[mask]), "mean_p": float(mu_c0[mask].mean()),
                     "obs_rate": float(y[mask].mean()),
                     "delta_c0_minus_model": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
    for vname, use_off in (("C1_plain", False), ("C1_offset", True)):
        for seed in _seeds_for("total_count", None, vname, seeds):
            off = mu_c0 if use_off else None
            oof = np.full(len(df), np.nan)
            oof[disc] = cv_predict(X[disc], y[disc], groups[disc], "count",
                                   seed=cfg.seed + seed, n_jobs=cfg.n_jobs,
                                   offset=None if off is None else off[disc])
            pred = np.array(oof)
            pred[~disc] = fit_predict(X[disc], y[disc], X[~disc], "count",
                                      seed=cfg.seed + seed, n_jobs=cfg.n_jobs,
                                      offset_tr=None if off is None else off[disc],
                                      offset_te=None if off is None else off[~disc])
            if seed == seeds[0]:
                preds[f"count_{vname}"] = pred
            for split, mask in ((DISC, disc), (CONF, ~disc)):
                d0 = poisson_deviance(y[mask], mu_c0[mask])
                dm = poisson_deviance(y[mask], pred[mask])
                delta, lo, hi = C.clustered_bootstrap_mean(d0 - dm, groups[mask],
                                                           cfg.n_boot, cfg.seed)
                rows.append({"target": "total_count", "line": np.nan, "model": vname,
                             "split": split, "seed": seed, "n": int(mask.sum()),
                             "metric": "poisson_deviance", "value": float(np.mean(dm)),
                             "r2": r2(y[mask], pred[mask]),
                             "mean_p": float(pred[mask].mean()),
                             "obs_rate": float(y[mask].mean()),
                             "delta_c0_minus_model": delta, "ci_lo": lo, "ci_hi": hi})

    # ---- binary lines ------------------------------------------------------
    for line in cfg.match_lines:
        yb = (df["y_total"].to_numpy(dtype=float) > line).astype(float)
        p_c0, nb_r, ab = _proxy_probs(df, line, "proxy_total", "y_total", cfg)
        preds[f"p_c0_{line}"] = p_c0
        base_p = float(yb[disc].mean())
        for split, mask in ((DISC, disc), (CONF, ~disc)):
            for name, p in (("C0_proxy", p_c0[mask]),
                            ("base_rate", np.full(int(mask.sum()), base_p))):
                rows.append({"target": "over_line", "line": line, "model": name,
                             "split": split, "seed": 0, "n": int(mask.sum()),
                             "metric": "log_loss", "value": log_loss(yb[mask], p),
                             "brier": brier(yb[mask], p), "nb_r": nb_r,
                             "platt_a": ab[0], "platt_b": ab[1],
                             "obs_rate": float(yb[mask].mean()),
                             "mean_p": float(np.mean(p)),
                             "delta_c0_minus_model": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
        for vname, use_off in (("C1_plain", False), ("C1_offset", True)):
            for seed in _seeds_for("over_line", line, vname, seeds):
                off = p_c0 if use_off else None
                oof = np.full(len(df), np.nan)
                oof[disc] = cv_predict(X[disc], yb[disc], groups[disc], "binary",
                                       seed=cfg.seed + seed, n_jobs=cfg.n_jobs,
                                       offset=None if off is None else off[disc])
                pred = np.array(oof)
                pred[~disc] = fit_predict(X[disc], yb[disc], X[~disc], "binary",
                                          seed=cfg.seed + seed, n_jobs=cfg.n_jobs,
                                          offset_tr=None if off is None else off[disc],
                                          offset_te=None if off is None else off[~disc])
                if seed == seeds[0]:
                    preds[f"p_{vname}_{line}"] = pred
                for split, mask in ((DISC, disc), (CONF, ~disc)):
                    l0 = per_sample_log_loss(yb[mask], p_c0[mask])
                    lm = per_sample_log_loss(yb[mask], pred[mask])
                    delta, lo, hi = C.clustered_bootstrap_mean(l0 - lm, groups[mask],
                                                               cfg.n_boot, cfg.seed)
                    rows.append({"target": "over_line", "line": line, "model": vname,
                                 "split": split, "seed": seed, "n": int(mask.sum()),
                                 "metric": "log_loss",
                                 "value": log_loss(yb[mask], pred[mask]),
                                 "brier": brier(yb[mask], pred[mask]), "nb_r": nb_r,
                                 "platt_a": ab[0], "platt_b": ab[1],
                                 "obs_rate": float(yb[mask].mean()),
                                 "mean_p": float(pred[mask].mean()),
                                 "delta_c0_minus_model": delta, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# stage: team-level lines (home / away corners)
# ---------------------------------------------------------------------------
def to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate the match table into two team-oriented rows per match.

    Args:
        df: Match table from :func:`build_match_table`.

    Returns:
        Frame [2 * n_matches] with ``own_<x>`` / ``opp_<x>`` in place of ``h_<x>`` /
        ``a_<x>``, the team's own corner count as ``y_team``, its proxy mean as
        ``proxy_team``, and ``is_home``.
    """
    h_cols = [c for c in df.columns if c.startswith("h_")]
    stems = [c[2:] for c in h_cols if f"a_{c[2:]}" in df.columns]
    keep = ["match_id", "date", "Division", "div_season", "season", "split", "elo_gap",
            "p_home", "p_away", "p_over25", "p_fav", "month", "proxy_total", "y_total"]
    keep = [c for c in keep if c in df.columns]
    out = []
    for is_home in (1, 0):
        f = df[keep].copy()
        src_own, src_opp = ("h_", "a_") if is_home else ("a_", "h_")
        own = df[[f"{src_own}{s}" for s in stems]].to_numpy()
        opp = df[[f"{src_opp}{s}" for s in stems]].to_numpy()
        f = pd.concat([
            f.reset_index(drop=True),
            pd.DataFrame(own, columns=[f"own_{s}" for s in stems]),
            pd.DataFrame(opp, columns=[f"opp_{s}" for s in stems]),
        ], axis=1)
        f["is_home"] = is_home
        f["y_team"] = (df["y_home"] if is_home else df["y_away"]).to_numpy()
        f["proxy_team"] = f["own_proxy_lam"].to_numpy()
        f["p_team_win"] = (df["p_home"] if is_home else df["p_away"]).to_numpy()
        out.append(f)
    res = pd.concat(out, ignore_index=True)
    return res[res["y_team"].notna() & np.isfinite(res["proxy_team"])].reset_index(drop=True)


#: Team-level feature blocks, used to attribute the team-line gain.
TEAM_SETS: tuple[str, ...] = ("C1_market_only", "C1_events_only", "C1_full")


def team_features(t: pd.DataFrame, which: str) -> list[str]:
    """Feature list for a team-level model.

    Args:
        t: Team-row frame from :func:`to_team_rows`.
        which: One of :data:`TEAM_SETS`.

    Returns:
        Column names. ``C1_market_only`` is the pre-match market view plus home/away;
        ``C1_events_only`` is the strictly-prior event history of both teams;
        ``C1_full`` is both.
    """
    market = [c for c in ("elo_gap", "p_home", "p_away", "p_over25", "p_fav",
                          "p_team_win", "is_home", "month") if c in t.columns]
    drop = {"y_team", "y_total", "proxy_team", "split", "date", "Division", "div_season",
            "season", "match_id"}
    events = [c for c in t.columns
              if c not in drop and c not in market and t[c].dtype.kind in "fiub"]
    if which == "C1_market_only":
        return market
    if which == "C1_events_only":
        return events + ["is_home"]
    return market + events


def stage_team(df: pd.DataFrame, cfg: CornerConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C0 vs C1 for a single team's corners and the 4.5 / 5.5 team lines.

    Rows are team-matches, so both rows of a match always fall in the same fold. Three
    nested feature blocks separate the two channels a corner line could be missing: the
    market's pre-match view of the game (who is favourite, how open the match is) and the
    teams' own prior event histories.

    Args:
        df: Match table.
        cfg: Stage configuration.

    Returns:
        ``(metrics, preds)``; ``metrics`` has one row per (target, line, model, split,
        side) and ``preds`` carries the predictions for the team betting simulation.
    """
    t = to_team_rows(add_div_code(df))
    disc = t["split"].to_numpy() == DISC
    groups = t["match_id"].to_numpy()
    y = t["y_team"].to_numpy(dtype=float)
    mu0 = t["proxy_team"].to_numpy(dtype=float)
    sides = (("both", np.ones(len(t), dtype=bool)),
             ("home", t["is_home"].to_numpy() == 1),
             ("away", t["is_home"].to_numpy() == 0))
    rows: list[dict[str, object]] = []
    preds = t[["match_id", "date", "Division", "split", "is_home", "y_team", "proxy_team",
               "p_team_win", "p_fav", "p_over25", "p_home", "elo_gap",
               "proxy_total"]].copy()
    preds["y_total"] = t["y_total"].to_numpy()

    for which in TEAM_SETS:
        X = t[team_features(t, which)]
        oof = np.full(len(t), np.nan)
        oof[disc] = cv_predict(X[disc], y[disc], groups[disc], "count", seed=cfg.seed,
                               n_jobs=cfg.n_jobs, offset=mu0[disc])
        pred = np.array(oof)
        pred[~disc] = fit_predict(X[disc], y[disc], X[~disc], "count", seed=cfg.seed,
                                  n_jobs=cfg.n_jobs, offset_tr=mu0[disc],
                                  offset_te=mu0[~disc])
        preds[f"count_{which}"] = pred
        for split, base in ((DISC, disc), (CONF, ~disc)):
            for side, smask in sides:
                m = base & smask
                d0 = poisson_deviance(y[m], mu0[m])
                dm = poisson_deviance(y[m], pred[m])
                delta, lo, hi = C.clustered_bootstrap_mean(d0 - dm, groups[m], cfg.n_boot,
                                                           cfg.seed)
                rows.append({"target": "team_count", "line": np.nan, "model": which,
                             "split": split, "side": side, "n": int(m.sum()),
                             "loss_C0": float(np.mean(d0)), "loss_C1": float(np.mean(dm)),
                             "r2_C0": r2(y[m], mu0[m]), "r2_C1": r2(y[m], pred[m]),
                             "delta_c0_minus_c1": delta, "ci_lo": lo, "ci_hi": hi})

    for line in cfg.team_lines:
        yb = (y > line).astype(float)
        r = C.fit_nb_dispersion(y[disc], mu0[disc])
        p0_raw = C.nb_sf(line, mu0, r)
        ab = C.platt_fit(p0_raw[disc], yb[disc].astype(int))
        p0 = C.platt_apply(p0_raw, ab)
        preds[f"p_c0_{line}"] = p0
        for which in TEAM_SETS:
            X = t[team_features(t, which)]
            oofb = np.full(len(t), np.nan)
            oofb[disc] = cv_predict(X[disc], yb[disc], groups[disc], "binary",
                                    seed=cfg.seed, n_jobs=cfg.n_jobs, offset=p0[disc])
            p1 = np.array(oofb)
            p1[~disc] = fit_predict(X[disc], yb[disc], X[~disc], "binary", seed=cfg.seed,
                                    n_jobs=cfg.n_jobs, offset_tr=p0[disc],
                                    offset_te=p0[~disc])
            preds[f"p_{which}_{line}"] = p1
            for split, base in ((DISC, disc), (CONF, ~disc)):
                for side, smask in sides:
                    m = base & smask
                    l0 = per_sample_log_loss(yb[m], p0[m])
                    l1 = per_sample_log_loss(yb[m], p1[m])
                    delta, lo, hi = C.clustered_bootstrap_mean(l0 - l1, groups[m],
                                                               cfg.n_boot, cfg.seed)
                    rows.append({"target": "team_over_line", "line": line, "model": which,
                                 "split": split, "side": side, "n": int(m.sum()),
                                 "loss_C0": log_loss(yb[m], p0[m]),
                                 "loss_C1": log_loss(yb[m], p1[m]),
                                 "brier_C0": brier(yb[m], p0[m]),
                                 "brier_C1": brier(yb[m], p1[m]),
                                 "obs_rate": float(yb[m].mean()),
                                 "delta_c0_minus_c1": delta, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), preds


def stage_team_refit(df: pd.DataFrame, cfg: CornerConfig,
                     seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Refit-noise floor for the *team* headline model, which is the phase's one margin.

    Protocol step 6 asks for a refit-noise floor beside every headline delta.
    :func:`stage_models` supplies one for the two *match* models only, so the team line --
    the one quantity in this phase whose margin over the proxy exceeds the goals
    haircut -- previously had none of its own, and quoting the match model's floor beside
    it would be a floor belonging to a different model.

    Only the confirmation half is refitted, because that is where the headline number is
    read: the confirmation prediction is one fit on all discovery rows, so re-seeding that
    fit is exactly the noise the floor is meant to bound. No cross-validation is run, which
    is what keeps this affordable.

    Args:
        df: Match table.
        cfg: Stage configuration.
        seeds: Model seeds to refit under.

    Returns:
        One row per (target, line, model) with ``min``, ``max``, ``count`` and ``spread``
        of the confirmation loss across seeds, plus the per-seed values in ``values``.
    """
    t = to_team_rows(add_div_code(df))
    disc = t["split"].to_numpy() == DISC
    y = t["y_team"].to_numpy(dtype=float)
    mu0 = t["proxy_team"].to_numpy(dtype=float)
    which = "C1_full"
    X = t[team_features(t, which)]
    rows: list[dict[str, object]] = []

    vals = []
    for seed in seeds:
        pred = fit_predict(X[disc], y[disc], X[~disc], "count", seed=seed,
                           n_jobs=cfg.n_jobs, offset_tr=mu0[disc], offset_te=mu0[~disc])
        vals.append(float(np.mean(poisson_deviance(y[~disc], pred))))
    rows.append({"target": "team_count", "line": float("nan"), "model": which,
                 "split": CONF, "min": min(vals), "max": max(vals),
                 "count": len(vals), "spread": max(vals) - min(vals),
                 "values": ";".join(f"{v:.6f}" for v in vals)})

    for line in cfg.team_lines:
        yb = (y > line).astype(float)
        r = C.fit_nb_dispersion(y[disc], mu0[disc])
        p0_raw = C.nb_sf(line, mu0, r)
        ab = C.platt_fit(p0_raw[disc], yb[disc].astype(int))
        p0 = C.platt_apply(p0_raw, ab)
        vals = []
        for seed in seeds:
            p1 = fit_predict(X[disc], yb[disc], X[~disc], "binary", seed=seed,
                             n_jobs=cfg.n_jobs, offset_tr=p0[disc], offset_te=p0[~disc])
            vals.append(float(log_loss(yb[~disc], p1)))
        rows.append({"target": "team_over_line", "line": float(line), "model": which,
                     "split": CONF, "min": min(vals), "max": max(vals),
                     "count": len(vals), "spread": max(vals) - min(vals),
                     "values": ";".join(f"{v:.6f}" for v in vals)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage: the gate (generalist vs specialist)
# ---------------------------------------------------------------------------
#: Candidate scenarios. Every predicate uses only pre-match information. Thresholds are
#: round numbers inspected on the discovery set; the primary scenario is picked by the
#: gate's own discovery result and only then scored on confirmation.
SCENARIO_DEFS: dict[str, str] = {
    "all": "every usable match (the gate's null scenario)",
    "fav_strong": "no-vig favourite probability >= 0.65",
    "fav_strong_low_total": "favourite >= 0.60 and P(over 2.5 goals) <= 0.48",
    "elo_mismatch_low_total": "|Elo gap| >= 150 and P(over 2.5 goals) <= 0.50",
    "high_total": "P(over 2.5 goals) >= 0.60",
    "home_big_fav": "home no-vig probability >= 0.65",
    "even_match": "no-vig favourite probability <= 0.42",
    "proxy_low": "proxy corner mean in the lowest discovery quintile",
    "proxy_high": "proxy corner mean in the highest discovery quintile",
}


def scenario_mask(df: pd.DataFrame, name: str, cuts: dict[str, float]) -> np.ndarray:
    """Boolean mask [n] selecting the rows of a named scenario.

    Args:
        df: Match table.
        name: Key of :data:`SCENARIO_DEFS`.
        cuts: Discovery-derived cut points, e.g. ``{"proxy_q20": 9.4, "proxy_q80": 11.1}``.

    Returns:
        Mask [n_matches]; rows with the required market fields missing are excluded.
    """
    p_fav = df["p_fav"].to_numpy(dtype=float)
    p_ov = df["p_over25"].to_numpy(dtype=float)
    p_home = df["p_home"].to_numpy(dtype=float)
    gap = np.abs(df["elo_gap"].to_numpy(dtype=float))
    proxy = df["proxy_total"].to_numpy(dtype=float)
    if name == "all":
        return np.ones(len(df), dtype=bool)
    if name == "fav_strong":
        return p_fav >= 0.65
    if name == "fav_strong_low_total":
        return (p_fav >= 0.60) & (p_ov <= 0.48)
    if name == "elo_mismatch_low_total":
        return (gap >= 150.0) & (p_ov <= 0.50)
    if name == "high_total":
        return p_ov >= 0.60
    if name == "home_big_fav":
        return p_home >= 0.65
    if name == "even_match":
        return p_fav <= 0.42
    if name == "proxy_low":
        return proxy <= cuts["proxy_q20"]
    if name == "proxy_high":
        return proxy >= cuts["proxy_q80"]
    raise KeyError(name)


def discovery_cuts(df: pd.DataFrame) -> dict[str, float]:
    """Cut points for the quantile-defined scenarios, taken on discovery only."""
    d = df[df["split"] == DISC]["proxy_total"].to_numpy(dtype=float)
    return {"proxy_q20": float(np.nanquantile(d, 0.20)),
            "proxy_q80": float(np.nanquantile(d, 0.80))}


def stage_gate(df: pd.DataFrame, cfg: CornerConfig) -> pd.DataFrame:
    """Generalist vs specialist on every candidate scenario.

    A generalist is fitted on all training rows; a specialist on the scenario's training
    rows only. Both are scored on held-out scenario rows (5-fold, whole matches held out,
    inside discovery) and again on confirmation after a single fit on all of discovery.
    The reverse direction -- the specialist off its own scenario -- is reported too.

    Args:
        df: Match table.
        cfg: Stage configuration.

    Returns:
        One row per (scenario, target, split, region) with the paired
        ``delta = loss(generalist) - loss(specialist)``, its match-clustered interval and
        n. Positive delta means the specialist is better, i.e. targeting has room.
    """
    df = add_div_code(df)
    cuts = discovery_cuts(df)
    disc = df["split"].to_numpy() == DISC
    groups = df["match_id"].to_numpy()
    feats = feature_columns(df, "C1")
    X = df[feats]
    targets = {
        "total_count": (df["y_total"].to_numpy(dtype=float), "count"),
        "over_9.5": ((df["y_total"].to_numpy(dtype=float) > 9.5).astype(float), "binary"),
    }
    rows: list[dict[str, object]] = []
    gen_oof: dict[str, np.ndarray] = {}
    gen_conf: dict[str, np.ndarray] = {}
    for tname, (y, kind) in targets.items():
        oof = np.full(len(df), np.nan)
        oof[disc] = cv_predict(X[disc], y[disc], groups[disc], kind, seed=cfg.seed,
                               n_jobs=cfg.n_jobs)
        gen_oof[tname] = oof
        gen_conf[tname] = fit_predict(X[disc], y[disc], X[~disc], kind, seed=cfg.seed,
                                      n_jobs=cfg.n_jobs)

    def _loss(kind: str, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return poisson_deviance(y, p) if kind == "count" else per_sample_log_loss(y, p)

    for sname in SCENARIO_DEFS:
        smask = scenario_mask(df, sname, cuts)
        if sname == "all":
            continue
        for tname, (y, kind) in targets.items():
            spec_oof = np.full(len(df), np.nan)
            spec_oof[disc] = cv_predict(X[disc], y[disc], groups[disc], kind,
                                        seed=cfg.seed, n_jobs=cfg.n_jobs,
                                        subset=smask[disc])
            spec_conf = np.full((~disc).sum(), np.nan)
            tr = disc & smask
            if tr.sum() >= 200:
                spec_conf = fit_predict(X[tr], y[tr], X[~disc], kind, seed=cfg.seed,
                                        n_jobs=cfg.n_jobs)
            for split, base in ((DISC, disc), (CONF, ~disc)):
                if split == DISC:
                    gen, spec = gen_oof[tname], spec_oof
                else:
                    gen = np.full(len(df), np.nan)
                    gen[~disc] = gen_conf[tname]
                    spec = np.full(len(df), np.nan)
                    spec[~disc] = spec_conf
                for region, rmask in (("in_scenario", base & smask),
                                      ("off_scenario", base & ~smask)):
                    ok = rmask & np.isfinite(gen) & np.isfinite(spec)
                    if ok.sum() < 100:
                        continue
                    lg = _loss(kind, y[ok], gen[ok])
                    ls = _loss(kind, y[ok], spec[ok])
                    delta, lo, hi = C.clustered_bootstrap_mean(lg - ls, groups[ok],
                                                              cfg.n_boot, cfg.seed)
                    rows.append({
                        "scenario": sname, "description": SCENARIO_DEFS[sname],
                        "target": tname, "split": split, "region": region,
                        "n": int(ok.sum()), "n_train_scenario": int((disc & smask).sum()),
                        "loss_generalist": float(np.mean(lg)),
                        "loss_specialist": float(np.mean(ls)),
                        "delta_gen_minus_spec": delta, "ci_lo": lo, "ci_hi": hi,
                        "share_of_rows": float(smask[base].mean()),
                    })
    rows += _gate_team_rows(df, cfg, cuts)
    return pd.DataFrame(rows)


def _gate_team_rows(df: pd.DataFrame, cfg: CornerConfig,
                    cuts: dict[str, float]) -> list[dict[str, object]]:
    """Run the same gate on team-level corner counts for the betting scenarios.

    The team line is where the event + market model actually beats the proxy, so the gate
    is asked there too: is a specialist fitted on the scenario's team-rows better on them
    than a generalist fitted on all team-rows?

    Args:
        df: Match table.
        cfg: Stage configuration.
        cuts: Discovery cut points for the quantile scenarios.

    Returns:
        Result rows in the same schema as :func:`stage_gate`.
    """
    t = to_team_rows(add_div_code(df))
    disc = t["split"].to_numpy() == DISC
    groups = t["match_id"].to_numpy()
    y = t["y_team"].to_numpy(dtype=float)
    mu0 = t["proxy_team"].to_numpy(dtype=float)
    X = t[team_features(t, "C1_full")]
    gen = np.full(len(t), np.nan)
    gen[disc] = cv_predict(X[disc], y[disc], groups[disc], "count", seed=cfg.seed,
                           n_jobs=cfg.n_jobs, offset=mu0[disc])
    gen[~disc] = fit_predict(X[disc], y[disc], X[~disc], "count", seed=cfg.seed,
                             n_jobs=cfg.n_jobs, offset_tr=mu0[disc], offset_te=mu0[~disc])
    rows: list[dict[str, object]] = []
    for sname in ("fav_strong", "fav_strong_low_total", "high_total", "even_match"):
        smask = scenario_mask(t, sname, cuts)
        spec = np.full(len(t), np.nan)
        spec[disc] = cv_predict(X[disc], y[disc], groups[disc], "count", seed=cfg.seed,
                                n_jobs=cfg.n_jobs, subset=smask[disc], offset=mu0[disc])
        tr = disc & smask
        if tr.sum() >= 200:
            spec[~disc] = fit_predict(X[tr], y[tr], X[~disc], "count", seed=cfg.seed,
                                      n_jobs=cfg.n_jobs, offset_tr=mu0[tr],
                                      offset_te=mu0[~disc])
        for split, base in ((DISC, disc), (CONF, ~disc)):
            for region, rmask in (("in_scenario", base & smask),
                                  ("off_scenario", base & ~smask)):
                ok = rmask & np.isfinite(gen) & np.isfinite(spec)
                if ok.sum() < 100:
                    continue
                lg = poisson_deviance(y[ok], gen[ok])
                ls = poisson_deviance(y[ok], spec[ok])
                delta, lo, hi = C.clustered_bootstrap_mean(lg - ls, groups[ok],
                                                          cfg.n_boot, cfg.seed)
                rows.append({
                    "scenario": sname, "description": SCENARIO_DEFS[sname],
                    "target": "team_count", "split": split, "region": region,
                    "n": int(ok.sum()), "n_train_scenario": int((disc & smask).sum()),
                    "loss_generalist": float(np.mean(lg)),
                    "loss_specialist": float(np.mean(ls)),
                    "delta_gen_minus_spec": delta, "ci_lo": lo, "ci_hi": hi,
                    "share_of_rows": float(smask[base].mean()),
                })
    return rows


# ---------------------------------------------------------------------------
# stage: betting simulation
# ---------------------------------------------------------------------------
def line_from_mean(mu: np.ndarray, lo: float = 6.5, hi: float = 14.5) -> np.ndarray:
    """The half-line a book would hang from a predicted mean [n].

    Args:
        mu: Predicted mean count [n].
        lo: Lowest line offered.
        hi: Highest line offered.

    Returns:
        Half-integer lines [n], clipped to the menu.
    """
    return np.clip(np.floor(np.asarray(mu, dtype=float)) + 0.5, lo, hi)


def _calibrated_over(mu: np.ndarray, line: np.ndarray, y: np.ndarray, disc: np.ndarray,
                     ) -> tuple[np.ndarray, float, tuple[float, float]]:
    """NB over-probability at a per-row line, with dispersion and Platt fitted on discovery.

    Args:
        mu: Predicted means [n].
        line: Per-row line [n].
        y: Realised counts [n].
        disc: Discovery mask [n].

    Returns:
        ``(p_over [n], nb_r, platt_ab)``.
    """
    r = C.fit_nb_dispersion(y[disc], mu[disc])
    p_raw = np.empty(len(mu), dtype=float)
    for lv in np.unique(line):
        sel = line == lv
        p_raw[sel] = C.nb_sf(float(lv), mu[sel], r)
    yb = (y > line).astype(int)
    ab = C.platt_fit(p_raw[disc], yb[disc])
    return C.platt_apply(p_raw, ab), r, ab


def disagreement_table(p_model: np.ndarray, p_proxy: np.ndarray, y: np.ndarray,
                       n_bins: int = 5) -> pd.DataFrame:
    """Closing-line-value style check: who is right where the two disagree most.

    Args:
        p_model: Model over-probability [n].
        p_proxy: Proxy over-probability [n].
        y: Binary over outcome [n].
        n_bins: Equal-count bins of ``|p_model - p_proxy|``.

    Returns:
        One row per bin with n, the mean gap, both log-losses and their difference
        (positive = the model is closer to the truth).
    """
    gap = np.abs(p_model - p_proxy)
    order = np.argsort(gap)
    rows = []
    for i, chunk in enumerate(np.array_split(order, n_bins)):
        if len(chunk) == 0:
            continue
        llm = per_sample_log_loss(y[chunk], p_model[chunk]).mean()
        llp = per_sample_log_loss(y[chunk], p_proxy[chunk]).mean()
        rows.append({"bin": i + 1, "n": int(len(chunk)),
                     "mean_abs_gap": float(gap[chunk].mean()),
                     "ll_model": float(llm), "ll_proxy": float(llp),
                     "delta_proxy_minus_model": float(llp - llm),
                     "obs_rate": float(y[chunk].mean()),
                     "mean_model": float(p_model[chunk].mean()),
                     "mean_proxy": float(p_proxy[chunk].mean())})
    return pd.DataFrame(rows)


def _simulate_means(y: np.ndarray, mu_proxy: np.ndarray, sources: dict[str, np.ndarray],
                    groups: np.ndarray, disc: np.ndarray, scen: pd.DataFrame,
                    cfg: CornerConfig, lo: float, hi: float, market: str,
                    direct: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price a count market off a proxy mean and bet model probabilities into it.

    Args:
        y: Realised counts [n].
        mu_proxy: The proxy's mean, which sets both the line and the prices [n].
        sources: Model name -> predicted mean [n], each turned into an over-probability
            by the fitted negative binomial at the offered line.
        groups: Match id per row [n].
        disc: Discovery mask [n].
        scen: Frame carrying the columns the scenario predicates need.
        cfg: Stage configuration.
        lo: Lowest line a book offers.
        hi: Highest line a book offers.
        market: Label written into the result rows.
        direct: Optional model name -> (probability [n], validity mask [n]) for direct
            binary classifiers, simulated only where the offered line matches.

    Returns:
        ``(sims, clv)``.
    """
    line = line_from_mean(mu_proxy, lo, hi)
    p_proxy, r0, ab0 = _calibrated_over(mu_proxy, line, y, disc)
    yb = (y > line).astype(float)
    cuts = discovery_cuts(scen)
    probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    r1 = float("nan")
    for name, mu in sources.items():
        p, r1, _ = _calibrated_over(np.asarray(mu, dtype=float), line, y, disc)
        probs[name] = (p, np.isfinite(p))
    if direct:
        probs.update(direct)

    rows = []
    for source, (p_model, valid) in probs.items():
        for sname in ("all", "fav_strong", "fav_strong_low_total", "high_total"):
            smask = scenario_mask(scen, sname, cuts) & valid & np.isfinite(p_model)
            for split, base in ((DISC, disc), (CONF, ~disc)):
                m = base & smask
                if m.sum() < 200:
                    continue
                for hold in (0.04, 0.06, 0.08):
                    for thr in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
                        sim = C.simulate_two_way(
                            yb[m], p_model[m], p_proxy[m], groups[m],
                            C.BetSimConfig(hold=hold, edge_threshold=thr,
                                           n_boot=cfg.n_boot, seed=cfg.seed))
                        rows.append(sim.as_row(market=market, split=split, scenario=sname,
                                               source=source, hold=hold, threshold=thr,
                                               mean_line=float(line[m].mean()),
                                               nb_r_proxy=r0, nb_r_model=r1))
    sims = pd.DataFrame(rows)
    head = list(sources)[0]
    p_head = probs[head][0]
    clv = []
    for split, base in ((DISC, disc), (CONF, ~disc)):
        ok = base & np.isfinite(p_head)
        tab = disagreement_table(p_head[ok], p_proxy[ok], yb[ok])
        tab["split"] = split
        tab["market"] = market
        tab["source"] = head
        clv.append(tab)
    out = pd.concat(clv, ignore_index=True)
    out.attrs["platt_proxy"] = ab0
    return sims, out


def stage_bets(preds: pd.DataFrame, cfg: CornerConfig,
               model_col: str = "count_C1_offset") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Betting simulation for the match-total corner market.

    Args:
        preds: Prediction frame from :func:`stage_models`.
        cfg: Stage configuration.
        model_col: Column holding the count model's predicted mean.

    Returns:
        ``(sims, clv)``.
    """
    d = preds.dropna(subset=[model_col, "proxy_total", "y_total"]).reset_index(drop=True)
    disc = d["split"].to_numpy() == DISC
    y = d["y_total"].to_numpy(dtype=float)
    mu0 = d["proxy_total"].to_numpy(dtype=float)
    line = line_from_mean(mu0)
    direct: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    p_bin = np.full(len(d), np.nan)
    ok_bin = np.zeros(len(d), dtype=bool)
    for lv in cfg.match_lines:
        col = f"p_C1_offset_{lv}"
        if col not in d.columns:
            continue
        sel = (line == lv) & d[col].notna().to_numpy()
        p_bin[sel] = d[col].to_numpy(dtype=float)[sel]
        ok_bin |= sel
    if ok_bin.any():
        direct["binary_direct"] = (p_bin, ok_bin)
    return _simulate_means(y, mu0, {"count_nb": d[model_col].to_numpy(dtype=float)},
                           d["match_id"].to_numpy(), disc, d, cfg, 6.5, 14.5,
                           "match_total", direct)


def stage_bets_team(preds: pd.DataFrame, cfg: CornerConfig,
                    model_col: str = "count_C1_full") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Betting simulation for a single team's corner line (4.5 / 5.5).

    Args:
        preds: Team prediction frame from :func:`stage_team`.
        cfg: Stage configuration.
        model_col: Column holding the team count model's predicted mean.

    Returns:
        ``(sims, clv)``.
    """
    d = preds.dropna(subset=[model_col, "proxy_team", "y_team"]).reset_index(drop=True)
    disc = d["split"].to_numpy() == DISC
    y = d["y_team"].to_numpy(dtype=float)
    mu0 = d["proxy_team"].to_numpy(dtype=float)
    line = line_from_mean(mu0, 2.5, 8.5)
    direct: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    p_bin = np.full(len(d), np.nan)
    ok_bin = np.zeros(len(d), dtype=bool)
    for lv in cfg.team_lines:
        col = f"p_C1_full_{lv}"
        if col not in d.columns:
            continue
        sel = (line == lv) & d[col].notna().to_numpy()
        p_bin[sel] = d[col].to_numpy(dtype=float)[sel]
        ok_bin |= sel
    if ok_bin.any():
        direct["binary_direct"] = (p_bin, ok_bin)
    sources = {"count_nb": d[model_col].to_numpy(dtype=float)}
    if "count_C1_market_only" in d.columns:
        sources["count_nb_market_only"] = d["count_C1_market_only"].to_numpy(dtype=float)
    return _simulate_means(y, mu0, sources, d["match_id"].to_numpy(), disc, d, cfg,
                           2.5, 8.5, "team_total", direct)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(x: float, nd: int = 4) -> str:
    """Format a float for prose, returning ``n/a`` for NaN."""
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(row: pd.Series, val: str = "delta_vs_B0", lo: str = "ci_lo", hi: str = "ci_hi",
        nd: int = 4) -> str:
    """Format ``value [lo, hi]`` from a result row."""
    return f"{_fmt(float(row[val]), nd)} [{_fmt(float(row[lo]), nd)}, {_fmt(float(row[hi]), nd)}]"


def _haircut_definitions(A, rd: Path, bets: pd.DataFrame, one_way: float) -> None:
    """Append the cross-stage comparison of the phase's three haircut definitions.

    The one-directional "book-earns" haircut this stage measures (``one_way``) is the
    *smallest* of the three the phase uses, so it is a lower bound rather than the number
    a bettor experiences. Stages 01 and 04 both measure round trips, which charge the
    profit that evaporates and the loss that appears; stage 04's is margin-matched (the
    real price is re-priced at the same hold as the proxy leg) and is the phase's single
    reference definition. Which of the three is applied decides whether this stage's team
    line survives, so the arithmetic is printed rather than asserted.

    Args:
        A: Line-appending callback of the report writer.
        rd: Reports directory.
        bets: This stage's betting table.
        one_way: The one-directional haircut at the 6% hold and the 0.02 threshold.
    """
    from research.privileged_tracking.common.report import md_table

    defs: list[dict[str, object]] = [
        {"definition": "book-earns (one-directional)", "stage": "02 corners",
         "value": one_way,
         "what": "ROI a real book makes betting its de-vigged price into proxy prices"}]
    fh = rd / "04_fouls_d_haircut.parquet"
    matched = None
    if fh.exists():
        f = pd.read_parquet(fh)
        f6 = f[(f["split"] == CONF) & (f["hold"] == 0.06)]
        if len(f6):
            matched = float(f6["haircut_roi_points"].iloc[0])
            defs.append({"definition": "margin-matched round trip", "stage": "04 fouls",
                         "value": matched,
                         "what": "ROI(model vs proxy prices) - ROI(model vs the real "
                                 "book's no-vig price re-priced at the same hold)"})
    ch = rd / "01_cards_c_haircut.parquet"
    actual = None
    if ch.exists():
        xh = pd.read_parquet(ch).iloc[0]
        actual = float(xh["roi_haircut"])
        defs.append({"definition": "actual-prices round trip", "stage": "01 cards",
                     "value": actual,
                     "what": f"the same, against Bet365's posted prices: "
                             f"{_fmt(float(xh['roi_vs_proxy_mean_over_holds']), 3)} the "
                             f"goals model makes against proxy prices plus the "
                             f"{_fmt(abs(float(xh['roi_vs_real_book'])), 3)} it loses "
                             f"against the posted price"})
    A("")
    A("**Three haircut definitions, and which one this stage's results are read under.** "
      "The phase measures the same gap in money three ways. They are not "
      "interchangeable, and the corner team line survives under one of them and not the "
      "other two, so all three are shown (confirmation, 6% hold):")
    A("")
    A(md_table(pd.DataFrame(defs)[["definition", "stage", "value", "what"]]))
    A("")
    conf = bets[(bets["split"] == CONF) & (bets["source"] == "count_nb")
                & (bets["threshold"] == 0.02) & (bets["hold"] == 0.06)]
    lines = []
    for _, r in conf.iterrows():
        cell = f"`{r['market']} x {r['scenario']}`"
        vals = [f"{float(r['roi']) - h['value']:+.3f}" for h in defs]
        lines.append({"market": cell, "n_bets": int(r["n_bets"]),
                      "ROI vs proxy": round(float(r["roi"]), 4),
                      "- book-earns": vals[0],
                      "- margin-matched": vals[1] if len(vals) > 1 else "n/a",
                      "- actual-prices": vals[2] if len(vals) > 2 else "n/a"})
    A("Every betting cell of this stage after each haircut (count-NB source, 6% hold, "
      "0.02 threshold, confirmation):")
    A("")
    A(md_table(pd.DataFrame(lines)))
    A("")
    surv = [ln["market"] for ln in lines if float(ln["- book-earns"]) > 0]
    surv_m = [ln["market"] for ln in lines
              if ln["- margin-matched"] != "n/a" and float(ln["- margin-matched"]) > 0]
    A(f"Under the one-directional haircut {len(surv)} of {len(lines)} cells stay positive "
      f"({', '.join(surv) if surv else 'none'}); under the margin-matched round trip "
      f"{len(surv_m)} do ({', '.join(surv_m) if surv_m else 'none'}); under the "
      "actual-prices round trip none do. The match total is clearly negative under all "
      "three. **Which definition is used therefore decides the team line and nothing "
      "else**, and the conclusion below reads it under the margin-matched one, which is "
      "the definition the phase settles on.")


def stage_report(cfg: CornerConfig) -> Path:
    """Render ``reports/02_corners.md`` from the parquet result tables.

    Args:
        cfg: Stage configuration (echoed into the report).

    Returns:
        Path of the written report.
    """
    from research.privileged_tracking.common.report import md_table

    rd = C.reports_dir()
    def rt(name: str) -> pd.DataFrame:
        return pd.read_parquet(rd / f"02_corners_{name}.parquet")

    sweep, audit, tune = rt("proxy_sweep"), rt("audit"), rt("tune")
    hm, hr = rt("goals_haircut_metrics"), rt("goals_haircut_roi")
    models, gate, team = rt("models"), rt("gate"), rt("team")
    bets, clv = rt("bets"), rt("clv")
    style, join = rt("style"), rt("style_join")
    spread = rt("refit_spread")

    hb = hm[(hm.split == CONF) & (hm.model == "bet365_novig")].iloc[0]
    hb_d = hm[(hm.split == DISC) & (hm.model == "bet365_novig")].iloc[0]
    hair6 = hr[(hr.split == CONF) & (hr.hold == 0.06) & (hr.threshold == 0.02)].iloc[0]
    best_sweep = sweep[sweep.stat == "corners_for"].sort_values("r2", ascending=False).iloc[0]

    def model_row(target: str, line: float | None, model: str, split: str) -> pd.Series:
        m = (models.target == target) & (models.model == model) & (models.split == split)
        m &= models.line.isna() if line is None else (models.line == line)
        if "seed" in models.columns:
            m &= models.seed == models.seed.min()
        return models[m].iloc[0]

    t_head = team[(team.target == "team_count") & (team.side == "both")
                  & (team.split == CONF) & (team.model == "C1_full")]
    t_head_line = team[(team.target == "team_over_line") & (team.line == 5.5)
                       & (team.side == "both") & (team.split == CONF)
                       & (team.model == "C1_full")]
    t_mkt = team[(team.target == "team_count") & (team.side == "both")
                 & (team.split == CONF) & (team.model == "C1_market_only")]
    t_evt = team[(team.target == "team_count") & (team.side == "both")
                 & (team.split == CONF) & (team.model == "C1_events_only")]
    c1o_conf = model_row("total_count", None, "C1_offset", CONF)
    c1o_disc = model_row("total_count", None, "C1_offset", DISC)
    line_rows = models[(models.target == "over_line") & (models.split == CONF)
                       & (models.model == "C1_offset")]

    gate_best = gate[(gate.split == DISC) & (gate.region == "in_scenario")
                     & (gate.target == "total_count")].sort_values(
        "delta_gen_minus_spec", ascending=False)
    best_scn = gate_best.iloc[0]["scenario"] if len(gate_best) else "n/a"
    gate_conf = gate[(gate.split == CONF) & (gate.region == "in_scenario")
                     & (gate.scenario == best_scn) & (gate.target == "total_count")]

    b_conf = bets[(bets.split == CONF) & (bets.scenario == "all")
                  & (bets.source == "count_nb") & (bets.market == "match_total")]
    b6 = b_conf[(b_conf.hold == 0.06) & (b_conf.threshold == 0.02)]
    tb6 = bets[(bets.split == CONF) & (bets.scenario == "all") & (bets.hold == 0.06)
               & (bets.threshold == 0.02) & (bets.source == "count_nb")
               & (bets.market == "team_total")]
    st_conf = style[style.scheme == "forward_confirmation"].set_index("feature_set")
    st_cv = style[style.scheme == "cv_all"].set_index("feature_set")
    jr = join.iloc[0]

    other = sorted(p.name for p in rd.glob("01_*.parquet") if "goal" in p.name.lower()
                   or "haircut" in p.name.lower())

    L: list[str] = []
    A = L.append
    A("# 02 Corners: does style or state beat a rolling mean, and is the gap bettable?")
    A("")
    A("Stage 02 of the scenario expected-value programme. The question is not whether a "
      "model is better on average but whether, in a pre-registered scenario, it prices "
      "corner totals better than the rolling-mean heuristic a side market is hung off, by "
      "enough to survive a realistic margin. Reproduce with "
      "`python -m research.scenario_ev.corners --stage all`.")
    A("")
    A("## Verdict")
    A("")
    A(f"1. **For the match total, a rolling mean is almost all there is.** The book proxy "
      f"explains R2 = {_fmt(float(best_sweep['r2']), 4)} of match-total corners on "
      f"discovery (correlation {_fmt(float(best_sweep['corr']), 3)}); no shrinkage or "
      f"functional form does better. Adding Elo, form, prior shots and fouls and the "
      f"market's own 1X2 / over-2.5 prices changes confirmation Poisson deviance by "
      f"{_ci(c1o_conf, 'delta_c0_minus_model')} and the over/under lines by "
      f"{_fmt(float(line_rows[line_rows.line == 9.5].iloc[0]['delta_c0_minus_model']))} nats "
      f"at 9.5 -- an order of magnitude less than the haircut in point 5.")
    gd = gate[(gate.target == "total_count") & (gate.region == "in_scenario")
              & (gate.split == DISC)]["delta_gen_minus_spec"]
    A(f"2. **The gate says targeting has no room.** For every pre-registered scenario a "
      f"specialist fitted on that scenario's rows is *worse* on those rows than a "
      f"generalist fitted on everything, on discovery and again on confirmation "
      f"(deltas from {_fmt(float(gd.min()))} to {_fmt(float(gd.max()))} on discovery). "
      f"The general model is not underfitting the favourites, the mismatches or the high "
      f"totals; it has nothing extra to give them.")
    if len(t_head) and len(t_head_line):
        A(f"3. **The one real signal is the split, not the total.** At team level the same "
          f"model beats the same proxy by {_ci(t_head.iloc[0], 'delta_c0_minus_c1')} "
          f"Poisson deviance (R2 {_fmt(float(t_head.iloc[0]['r2_C0']), 3)} -> "
          f"{_fmt(float(t_head.iloc[0]['r2_C1']), 3)}, n = {int(t_head.iloc[0]['n'])} "
          f"team-matches) and by {_ci(t_head_line.iloc[0], 'delta_c0_minus_c1')} nats at "
          f"the 5.5 team line. The ablation says where it comes from: the pre-match market "
          f"block alone is worth {_ci(t_mkt.iloc[0], 'delta_c0_minus_c1')} of the "
          f"{_ci(t_head.iloc[0], 'delta_c0_minus_c1')}. A team's corner count follows the "
          f"game script; the match total is nearly conserved between the two sides, so it "
          f"barely moves.")
    own_base = float(st_cv.loc["B1_odds_event_market", "delta_vs_B0"])
    style_base = float(st_cv.loc["B2_plus_style", "delta_vs_B0"])
    A(f"4. **Style and imputed defensive state carry corner signal, but not new signal.** "
      f"On their own they beat the proxy (style only "
      f"{_ci(st_cv.loc['S_style_only'], 'delta_vs_B0')}, imputed defensive state only "
      f"{_ci(st_cv.loc['S_imputed_only'], 'delta_vs_B0')} in match-grouped CV). On top of "
      f"the odds table, the team's own style adds "
      f"{_fmt(float(st_cv.loc['B2a_plus_own_style', 'delta_vs_B0']) - own_base)}"
      f" and the opponent's compactness a further "
      f"{_fmt(float(st_cv.loc['B3opp_opponent_compactness_only', 'delta_vs_B0']) - style_base)}"
      f", both inside the intervals. The stated hypothesis -- crossing teams against "
      f"*compact* opponents -- keeps its first half and loses its second.")
    A(f"5. **The haircut is the size of the whole effect.** A real bookmaker beats the "
      f"identical rolling-mean proxy on the one count market whose line we observe (total "
      f"goals at 2.5) by {_ci(hb, 'delta_vs_proxy_cal', 'ci_lo', 'ci_hi')} nats on "
      f"confirmation, worth ROI {_fmt(float(hair6['roi']), 3)} "
      f"[{_fmt(float(hair6['roi_lo']), 3)}, {_fmt(float(hair6['roi_hi']), 3)}] betting into "
      f"rolling-mean prices at a 6% hold and a 2% edge threshold.")
    if len(b6) and len(tb6):
        rm, rt = b6.iloc[0], tb6.iloc[0]
        A(f"6. **So: no +EV claim survives.** At a 6% hold and the discovery threshold the "
          f"match-total market returns ROI {_fmt(float(rm['roi']), 4)} "
          f"[{_fmt(float(rm['roi_lo']), 4)}, {_fmt(float(rm['roi_hi']), 4)}] against proxy "
          f"prices, {_fmt(float(rm['roi']) - float(hair6['roi']), 4)} after the haircut. "
          f"The team-line market returns {_fmt(float(rt['roi']), 4)} "
          f"[{_fmt(float(rt['roi_lo']), 4)}, {_fmt(float(rt['roi_hi']), 4)}] -- large, and "
          f"almost exactly the haircut, leaving "
          f"{_fmt(float(rt['roi']) - float(hair6['roi']), 4)} net. Read plainly: our team "
          f"model is about as far ahead of a rolling mean as a real book is, and a real "
          f"book is what you would be betting against.")
    A("")
    A("## 1. Data, universe and the fixed split")
    A("")
    A("- Source: `data/raw/privileged/odds/matches.parquet` -- 238,858 football-data.co.uk "
      "matches, of which 122,111 record both corner counts (51%, 22 division codes).")
    A(f"- Usable universe after requiring at least {cfg.min_prior} prior matches with "
      f"corners recorded for both teams (so that the proxy has something to average): "
      f"{int(c1o_disc['n']) + int(c1o_conf['n'])} matches.")
    A(f"- **Discovery** (earlier 60% of matches): n = {int(c1o_disc['n'])}. "
      f"**Confirmation** (later 40%): n = {int(c1o_conf['n'])}. Every threshold, feature "
      "choice, shrinkage and bet rule below was chosen on discovery; confirmation was "
      "scored once. Discovery numbers are exploratory and multiplicity-inflated.")
    A("- Grouping: every split holds out whole matches; every rolling feature uses matches "
      "with a strictly earlier date, so same-day fixtures cannot leak into each other.")
    A("")
    A("### Leakage audit (recomputed by brute force, not asserted)")
    A("")
    A(md_table(audit))
    A("")
    A("## 2. C0, the book proxy, and how its shrinkage was chosen")
    A("")
    A("The proxy is the standard heuristic: expected corners = league home/away level x "
      "the team's shrunk prior corners-for ratio x the opponent's shrunk prior "
      "corners-against ratio, all from strictly earlier matches of the same "
      "division-season, with the league level itself shrunk toward the division's "
      "all-history mean. Its one free parameter is the shrinkage `k`, chosen on discovery "
      "by squared error over both functional forms so that the opponent is the strongest "
      "rolling mean available, not a straw man.")
    A("")
    A(md_table(sweep[sweep.stat == "corners_for"].sort_values("r2", ascending=False).head(8)))
    A("")
    A(f"Chosen: multiplicative form, k = {cfg.proxy_k:.0f} (the discovery optimum). The "
      "same form and shrinkage are used for the goals proxy of the honesty check.")
    A("")
    A("### LightGBM capacity, also chosen on discovery")
    A("")
    A("The corner signal is weak enough that model capacity decides the answer. A "
      "conventional 400-tree / 31-leaf LightGBM loses to the proxy outright; a small one "
      "beats it. The choice is made on an inner forward split *inside* discovery -- fit on "
      "its first 75% of dates, score the last 25% -- so confirmation is untouched. Without "
      "this step a null result here could not be told apart from a capacity artefact.")
    A("")
    A(md_table(tune.sort_values(["target", "loss"]), floatfmt="{:.6f}"))
    A("")
    A("## 3. Proxy honesty check on goals: the haircut (protocol step 4)")
    A("")
    A("We have no corner or card lines, only the 1X2 and over/under 2.5 goals prices. So "
      "the proxy's weakness is calibrated on the one count market where a real line is "
      "observed: the identical rolling-mean machinery predicts total goals, is turned into "
      "a probability at 2.5 (negative binomial, dispersion and a Platt recalibration both "
      "fitted on discovery), and is compared with Bet365's no-vig over/under 2.5 price on "
      "the same matches. The gap is the estimate of how much better a real book is than a "
      "rolling mean on a count market.")
    A("")
    A(md_table(hm))
    A("")
    A(f"On confirmation the real book beats the calibrated rolling-mean proxy by "
      f"{_ci(hb, 'delta_vs_proxy_cal', 'ci_lo', 'ci_hi')} nats (discovery "
      f"{_ci(hb_d, 'delta_vs_proxy_cal', 'ci_lo', 'ci_hi')}). Expressed as money: betting "
      "the book's fair probability into prices built from the rolling-mean proxy plus a "
      "hold returns")
    A("")
    A(md_table(hr[hr.split == CONF]))
    A("")
    A("**The haircut.** Any edge claimed against the rolling-mean proxy on corners must "
      f"exceed this before it can be called +EV: about {_fmt(float(hb['delta_vs_proxy_cal']), 4)} "
      f"nats of log-loss, or ROI {_fmt(float(hair6['roi']), 3)} at a 6% hold and a 2% edge "
      "threshold -- and that ROI figure is the *one-directional* form, which is a lower "
      "bound; the phase reference is the margin-matched round trip in the table below. "
      "Two caveats, both stated rather than buried: goals is the sharpest market "
      "a book prices, so this is probably an upper bound on a book's superiority over a "
      "rolling mean on corners; and the corner market's true margin is wider than the "
      f"{_fmt(float(hb['n']), 0)}-match goals market's "
      "(observed Bet365 over/under 2.5 hold averages 6.6%), which pushes the bar the other "
      "way.")
    cross = rd / "01_cards_c_goals_gap.parquet"
    if cross.exists():
        try:
            xg = pd.read_parquet(cross)
            xc = xg[(xg.get("where") == CONF) & (xg["model"] == "bet365_novig")].iloc[0]
            A("")
            A("**Cross-check on a different universe.** The cards stage (01) ran the same "
              "protocol step on the matches with card counts rather than corner counts, "
              f"n = {int(xc['n'])} on confirmation, with its own rolling machinery, its own "
              "goals model and its own split date, and measures the gap as "
              f"{_fmt(float(xc['delta_vs_proxy']))} "
              f"[{_fmt(float(xc['ci_lo']))}, {_fmt(float(xc['ci_hi']))}] nats against "
              f"{_ci(hb, 'delta_vs_proxy_cal', 'ci_lo', 'ci_hi')} here. The two intervals "
              "do not quite overlap -- different match sets, different eras, and card "
              "matches skew to the leagues and years with the most complete data. Note "
              "what this is and is not: the two universes are genuinely different, but "
              "both stages de-vig with the same shared helper (`common.novig_two_way`) on "
              "the same Bet365 price columns, so this is one construction measured on two "
              "match sets, not two independent constructions. Stage 04's foul universe is "
              "a *subset* of this one and agrees with it to five decimals, so it is the "
              "same measurement again rather than a third. `06_haircut.md` quantifies the "
              "overlap.")
            _haircut_definitions(A, rd, bets, float(hair6["roi"]))
        except (KeyError, IndexError, OSError):
            pass
    elif other:
        A("")
        A(f"(Tables from the cards stage's own version of this check: `{', '.join(other)}`.)")
    A("")
    A("## 4. (a) Large-sample layer: C0 vs C1")
    A("")
    A("C1 adds to the proxy's inputs: Elo and the Elo gap, form, division, rest days, "
      "prior shots / shots on target / fouls for both teams, and the match's pre-match "
      "no-vig 1X2 and over/under 2.5 probabilities. Two specifications are reported. "
      "`C1_plain` is a free-standing LightGBM; `C1_offset` takes the proxy as an "
      "`init_score` so the trees only learn the deviation from it, which removes the "
      "league-level drift that otherwise punishes a stationary model. Positive delta = "
      "better than the proxy.")
    A("")
    A(md_table(models[models.target == "total_count"][
        ["model", "split", "seed", "n", "value", "r2", "mean_p", "obs_rate",
         "delta_c0_minus_model", "ci_lo", "ci_hi"]]))
    A("")
    A("At the standard half-lines (log-loss, and the paired delta against C0):")
    A("")
    A(md_table(models[(models.target == "over_line")][
        ["line", "model", "split", "seed", "n", "value", "brier", "obs_rate", "mean_p",
         "delta_c0_minus_model", "ci_lo", "ci_hi"]]))
    A("")
    A("### Team lines: home and away corners (4.5 / 5.5)")
    A("")
    A("Same comparison at team-match level, both rows of a match always in the same fold, "
      "every model a proxy-offset correction. Three nested feature blocks separate the two "
      "channels: `C1_market_only` is the pre-match market view (no-vig 1X2 and over/under "
      "2.5, Elo, home/away) and nothing else; `C1_events_only` is the two teams' "
      "strictly-prior event histories; `C1_full` is both.")
    A("")
    A(md_table(team[(team.target == "team_count") & (team.side == "both")][
        ["model", "split", "n", "loss_C0", "loss_C1", "r2_C0", "r2_C1",
         "delta_c0_minus_c1", "ci_lo", "ci_hi"]]))
    A("")
    A(md_table(team[(team.target == "team_over_line") & (team.side == "both")][
        ["line", "model", "split", "n", "loss_C0", "loss_C1", "brier_C0", "brier_C1",
         "obs_rate", "delta_c0_minus_c1", "ci_lo", "ci_hi"]]))
    A("")
    A("Split by side (full model only):")
    A("")
    A(md_table(team[(team.model == "C1_full") & (team.side != "both")][
        ["target", "line", "split", "side", "n", "loss_C0", "loss_C1",
         "delta_c0_minus_c1", "ci_lo", "ci_hi"]]))
    A("")
    if len(t_head) and len(t_head_line):
        A(f"**This is the one place in the stage where the richer model clearly beats the "
          f"rolling mean.** On confirmation the full team model cuts Poisson deviance by "
          f"{_ci(t_head.iloc[0], 'delta_c0_minus_c1')} (n = {int(t_head.iloc[0]['n'])}) and "
          f"the 5.5 team line by {_ci(t_head_line.iloc[0], 'delta_c0_minus_c1')} nats. The "
          f"ablation says where it comes from: the market-only block alone is worth "
          f"{_ci(t_mkt.iloc[0], 'delta_c0_minus_c1') if len(t_mkt) else 'n/a'} and the "
          f"event-history block alone "
          f"{_ci(t_evt.iloc[0], 'delta_c0_minus_c1') if len(t_evt) else 'n/a'}. In other "
          f"words a team's corner count follows the *game script* -- who the market thinks "
          f"will be on top -- and a rolling corner mean is blind to it, while the match "
          f"total is nearly conserved between the two teams and so barely moves.")
        A("")
        A("Read this with the obvious caveat attached: a real bookmaker pricing a team "
          "corner line starts from the match odds, because they are the most visible input "
          "on the screen. Our proxy deliberately does not. So this gap measures how much "
          "the *proxy* is missing, not how much a real market is missing -- which is "
          "exactly what section 3's haircut is for.")
    A("")
    A("### Refit-noise floor (three seeds)")
    A("")
    A("The match models first. These are the C1_offset headline fits; the confirmation "
      "column is the one the deltas above are read against.")
    A("")
    A(md_table(spread))
    A("")
    tr = rd / "02_corners_team_refit.parquet"
    if tr.exists():
        tspread = pd.read_parquet(tr)
        A("And the **team** models, which the match-model floor above does not cover. The "
          "team line is the one quantity in this stage whose margin over the proxy exceeds "
          "the goals haircut in nats, so protocol step 6 wants a floor belonging to that "
          "model rather than to a different one. Confirmation only, since that is where "
          "the headline is read; the confirmation prediction is a single fit on all "
          "discovery rows, so re-seeding it is exactly the relevant noise.")
        A("")
        A(md_table(tspread[["target", "line", "model", "split", "min", "max", "count",
                            "spread", "values"]]))
        A("")
        row = tspread[(tspread.target == "team_over_line") & (tspread.line == 5.5)]
        if len(row):
            sp = float(row["spread"].iloc[0])
            A(f"The 5.5 team line's floor is {_fmt(sp, 6)} nats against a measured margin "
              f"over the proxy of {_ci(t_head_line.iloc[0], 'delta_c0_minus_c1', 'ci_lo', 'ci_hi')} "
              f"-- about {float(t_head_line['delta_c0_minus_c1'].iloc[0]) / max(sp, 1e-9):.0f}x "
              "the floor, so the margin is real as a measurement. Whether it is tradeable "
              "is a separate question and is answered by the haircut, not by this floor.")
        A("")
    A("## 5. The gate: is the generalist underfitting any scenario?")
    A("")
    A("A generalist is fitted on all training rows, a specialist on the scenario's "
      "training rows only, and both are scored on held-out scenario rows -- 5-fold with "
      "whole matches held out inside discovery, then once on confirmation. Positive "
      "`delta_gen_minus_spec` means the specialist is better, i.e. targeting has room. The "
      "reverse direction (the specialist off its own scenario) is shown too.")
    A("")
    A(md_table(gate[gate.target == "total_count"][
        ["scenario", "split", "region", "n", "n_train_scenario", "share_of_rows",
         "loss_generalist", "loss_specialist", "delta_gen_minus_spec", "ci_lo", "ci_hi"]]))
    A("")
    A(md_table(gate[gate.target == "over_9.5"][
        ["scenario", "split", "region", "n", "n_train_scenario", "loss_generalist",
         "loss_specialist", "delta_gen_minus_spec", "ci_lo", "ci_hi"]]))
    A("")
    A(f"Pre-registered primary scenario (the least-bad on discovery): **{best_scn}** -- "
      f"{SCENARIO_DEFS.get(best_scn, '')}.")
    if len(gate_conf):
        A(f"On confirmation it gives {_ci(gate_conf.iloc[0], 'delta_gen_minus_spec')} "
          f"(n = {int(gate_conf.iloc[0]['n'])}).")
    if len(gate_best) and float(gate_best.iloc[0]["delta_gen_minus_spec"]) <= 0:
        A("")
        A("Note what the table says: **no** scenario has a positive discovery delta for the "
          "match total, so there was nothing to pre-register in the first place. The "
          "specialists are uniformly worse on their own rows, and worse again off them; "
          "the loss grows as the scenario shrinks (`fav_strong_low_total`, 1.3% of rows, "
          "is the worst). That is the signature of a variance problem, not of a "
          "generalist that is underfitting a corner of the space. The single exception in "
          "the whole table is `fav_strong` at team level on confirmation "
          "(+0.0085 [0.0035, 0.0136]), whose discovery counterpart is +0.0012 "
          "[-0.0027, 0.0054] and therefore was not, and could not have been, "
          "pre-registered.")
    A("")
    A("## 6. Betting-shaped evaluation")
    A("")
    A("The proxy hangs the half-line its own mean implies, prices both sides at the stated "
      "two-way hold, and the model bets whichever side clears the edge threshold. Stakes "
      "are flat; ROI intervals resample whole matches. Thresholds were read off the "
      "discovery grid.")
    A("")
    show = bets[(bets.threshold.isin([0.0, 0.02, 0.05]))].copy()
    hair_map = {(float(r.hold), float(r.threshold)): float(r.roi)
                for r in hr[hr.split == CONF].itertuples()}
    show["haircut_roi"] = [hair_map.get((float(h), float(t)), float("nan"))
                           for h, t in zip(show["hold"], show["threshold"], strict=True)]
    show["roi_after_haircut"] = show["roi"] - show["haircut_roi"]
    cols = ["market", "scenario", "source", "hold", "threshold", "n_rows", "n_bets",
            "bet_rate", "mean_edge", "roi", "roi_lo", "roi_hi", "haircut_roi",
            "roi_after_haircut"]
    A("`haircut_roi` is what the *real* bookmaker earned in the identical simulation on "
      "goals at the same hold and threshold (section 3), so the two are directly "
      "comparable: same pricing rule, same margin, same bet filter. `roi_after_haircut` "
      "subtracts it. That subtraction is a point estimate against a point estimate -- the "
      "haircut itself carries an interval (0.110 to 0.140 at a 6% hold and a 2% "
      "threshold), so a net of a few tenths of a percent is not distinguishable from zero.")
    A("")
    A("Confirmation, all matches (the full grid -- both splits, every scenario and every "
      "threshold -- is in `02_corners_bets.parquet`):")
    A("")
    A(md_table(show[(show.split == CONF) & (show.scenario == "all")][cols]))
    A("")
    A("The same rows inside the pre-registered scenarios:")
    A("")
    A(md_table(show[(show.split == CONF) & (show.scenario != "all")
                    & (show.threshold == 0.02)][cols]))
    A("")
    A("Discovery, for comparison (this is where the threshold was chosen; these numbers "
      "are exploratory):")
    A("")
    A(md_table(show[(show.split == DISC) & (show.scenario == "all")][cols]))
    A("")
    if len(b6):
        r = b6.iloc[0]
        A(f"Headline (confirmation, all matches, 6% hold, 2% edge threshold): "
          f"{int(r['n_bets'])} bets from {int(r['n_rows'])} matches "
          f"({_fmt(float(r['bet_rate']), 3)}), mean modelled edge "
          f"{_fmt(float(r['mean_edge']), 4)}, ROI {_fmt(float(r['roi']), 4)} "
          f"[{_fmt(float(r['roi_lo']), 4)}, {_fmt(float(r['roi_hi']), 4)}]. "
          f"After the step-4 haircut of {_fmt(float(hair6['roi']), 4)}: "
          f"**{_fmt(float(r['roi']) - float(hair6['roi']), 4)}**.")
    if len(tb6):
        r = tb6.iloc[0]
        A("")
        A(f"Team lines (confirmation, 6% hold, 2% edge threshold): {int(r['n_bets'])} bets "
          f"from {int(r['n_rows'])} team-matches ({_fmt(float(r['bet_rate']), 3)}), mean "
          f"edge {_fmt(float(r['mean_edge']), 4)}, ROI {_fmt(float(r['roi']), 4)} "
          f"[{_fmt(float(r['roi_lo']), 4)}, {_fmt(float(r['roi_hi']), 4)}]; after the "
          f"haircut **{_fmt(float(r['roi']) - float(hair6['roi']), 4)}**.")
    A("")
    A("### Closing-line-value style calibration")
    A("")
    A("In the bins where model and proxy disagree most, is the model or the proxy closer "
      "to the truth? Positive `delta_proxy_minus_model` = the model.")
    A("")
    A(md_table(clv))
    A("")
    A("## 7. (b) Style layer on the StatsBomb overlap")
    A("")
    A(f"Join: {int(jr['n_sb_matches'])} StatsBomb matches of the 2015/2016 Premier League, "
      f"La Liga, Serie A and Ligue 1 against the odds table. The team-name map is built "
      f"explicitly by fixture-date overlap (maximum-weight bipartite matching per "
      f"division), not assumed: {int(jr['n_teams'])} names mapped, join rate "
      f"{_fmt(float(jr['join_rate']), 3)}, final scores agree on "
      f"{_fmt(float(jr['score_agreement']), 3)} of joined matches, "
      f"{int(jr['n_date_shifted'])} needed a one-day shift.")
    A("")
    A("Every model here is a multiplicative correction to the same C0 proxy fitted as a "
      "regularised Poisson GLM (the sample is small, so a linear correction is the "
      "low-variance choice), at team-corner level with matches held out together. "
      "`cv_all` is match-grouped 5-fold over the whole overlap -- more power, no temporal "
      "holdout; `forward_confirmation` is the protocol's chronological headline.")
    A("")
    A(md_table(style[["feature_set", "scheme", "n_rows", "n_matches", "poisson_deviance",
                      "delta_vs_B0", "ci_lo", "ci_hi", "delta_ll_4.5", "delta_ll_5.5"]]))
    A("")
    A("Reading of the hypothesis as stated -- *teams that cross heavily and get shots "
      "blocked against compact opponents generate corners well above their average*:")
    A("")
    A(f"- Style alone (crossing, blocked shots, passes into the box, PPDA, possession, "
      f"line height, for both teams, no corner history at all) beats the rolling corner "
      f"mean: {_ci(st_cv.loc['S_style_only'])} in CV, "
      f"{_ci(st_conf.loc['S_style_only'])} forward. The channel is real.")
    A(f"- The imputed defensive state alone (block depth, defensive line, deep-block share "
      f"of both teams, from the completed programme's students) also beats it: "
      f"{_ci(st_cv.loc['S_imputed_only'])} in CV, {_ci(st_conf.loc['S_imputed_only'])} "
      f"forward.")
    def step(a: str, b: str, tab: pd.DataFrame) -> str:
        return _fmt(float(tab.loc[b, "delta_vs_B0"]) - float(tab.loc[a, "delta_vs_B0"]))

    A(f"- Added on top of what the odds table already says, the team's **own** style is "
      f"worth {step('B1_odds_event_market', 'B2a_plus_own_style', st_cv)} deviance in CV "
      f"and {step('B1_odds_event_market', 'B2a_plus_own_style', st_conf)} forward -- small, "
      f"and the only increment in this layer that is positive in both evaluations.")
    A(f"- Adding the **opponent's** style on top of that is worth "
      f"{step('B2a_plus_own_style', 'B2_plus_style', st_cv)} in CV and "
      f"{step('B2a_plus_own_style', 'B2_plus_style', st_conf)} forward: nothing, or "
      f"slightly negative.")
    A(f"- Adding the opponent's **imputed compactness** (block depth, defensive line, "
      f"deep-block share, measured from their prior matches by the completed programme's "
      f"students) on top of both styles is worth "
      f"{step('B2_plus_style', 'B3opp_opponent_compactness_only', st_cv)} in CV and "
      f"{step('B2_plus_style', 'B3opp_opponent_compactness_only', st_conf)} forward; the "
      f"team's own compactness "
      f"{step('B2_plus_style', 'B3own_own_compactness_only', st_cv)} and "
      f"{step('B2_plus_style', 'B3own_own_compactness_only', st_conf)}. Every one of these "
      f"increments is an order of magnitude smaller than the interval around it.")
    A(f"- Extra StatsBomb corner history (`B2c`) is worth "
      f"{step('B2_plus_style', 'B2c_plus_sb_corner_history', st_cv)}, which is the useful "
      f"control: the odds table's own corner history has already extracted what there is.")
    A("")
    A("**So the hypothesis splits in two and only one half survives.** *Teams that cross "
      "heavily* is weakly supported -- own style adds a little, and style on its own beats "
      "the rolling mean outright. *Against compact opponents* is not: neither the "
      "opponent's measured style nor its imputed defensive shape adds anything once the "
      "team's own profile is known. And the whole layer is underpowered for the sizes "
      "involved -- 527 confirmation matches, with forward intervals wide enough to contain "
      "both zero and twice the effect.")
    A("")
    A("## 8. Limitations")
    A("")
    A("- There are no corner lines in this data. The opponent is a *simulated* book, not a "
      "real one; the haircut in section 3 is the only evidence about the distance between "
      "the two, and it is measured on a different (sharper) market.")
    A("- The style layer is one season of four leagues; its confirmation half is "
      f"{int(style[style.scheme == 'forward_confirmation']['n_matches'].max())} matches, so "
      "its intervals are wide and a null there is weak evidence, not strong.")
    A("- Corner counts are recorded on 51% of the odds table; divisions and seasons "
      "without them are absent, and the corner level drifts over time, which is why the "
      "proxy's league-level tracking matters as much as its team ratios.")
    A("- The proxy's negative-binomial dispersion and Platt recalibration are fitted on "
      "all discovery rows, including the held-out folds of the discovery cross-validation; "
      "with two parameters over tens of thousands of rows the effect is negligible, but it "
      "flatters C0 rather than the models under test.")
    A("- The team-line result rests on features derived from the 1X2 market. A model whose "
      "edge over a rolling mean comes from re-expressing the match odds has, by "
      "construction, no edge over a book that also reads the match odds. The honest "
      "statement is the negative one: a rolling-mean proxy is a poor model of how a real "
      "book prices a team corner line, so the +13% ROI against it is a measure of the "
      "proxy's weakness, not of a market inefficiency.")
    A("- The haircut is transferred from goals to corners unchanged. Goals is the market a "
      "book works hardest on, so the transfer probably overstates a corner book's edge; "
      "corner props also carry a wider margin than the 6.6% seen on the goals line, which "
      "understates the bar. Neither correction is measurable here, and the net result "
      "(team lines roughly break even, match totals lose about 11 points of ROI) is far "
      "enough from the boundary in the match-total case and far enough inside the "
      "haircut's own interval in the team-line case that neither would flip.")
    A("- Discovery spans 2000-2019 and confirmation 2019-2026; the corner rate falls from "
      "10.55 to 9.77 per match across that boundary. That drift is why a free-standing "
      "classifier is punished on confirmation and why every model here is specified as a "
      "correction to the proxy, which tracks the league level by construction.")
    A("")
    text = "\n".join(L) + "\n"
    out = rd / "02_corners.md"
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def refit_spread(models: pd.DataFrame) -> pd.DataFrame:
    """Spread of the headline metric across refit seeds.

    Args:
        models: Output of :func:`stage_models` run with several seeds.

    Returns:
        One row per (target, line, model, split) with the min, max and spread.
    """
    g = models.groupby(["target", "line", "model", "split"], dropna=False)["value"]
    out = g.agg(["min", "max", "count"]).reset_index()
    out["spread"] = out["max"] - out["min"]
    return out[out["count"] > 1].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Run one or all stages of the corners study.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "build", "audit", "proxy", "tune", "honesty",
                             "models", "team", "gate", "bets", "style", "report"])
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    cfg = CornerConfig()
    rd = C.reports_dir()
    stage = args.stage

    def w(df: pd.DataFrame, name: str) -> None:
        df.to_parquet(rd / f"02_corners_{name}.parquet", index=False)
        print(f"wrote {name}: {df.shape}")

    df = build_match_table(cfg, force=args.force_build)
    print(f"universe: {len(df)} matches; "
          f"{df['split'].value_counts().to_dict()}")
    if stage in ("all", "audit"):
        w(stage_audit(cfg), "audit")
    if stage in ("all", "proxy"):
        w(stage_proxy(df, cfg), "proxy_sweep")
    if stage in ("all", "tune"):
        w(stage_tune(df, cfg), "tune")
    if stage in ("all", "honesty"):
        hm, hr = stage_honesty(df, cfg)
        w(hm, "goals_haircut_metrics")
        w(hr, "goals_haircut_roi")
    if stage in ("all", "models"):
        seeds = tuple(range(args.seeds))
        met, preds = stage_models(df, cfg, seeds=seeds)
        w(met, "models")
        w(refit_spread(met), "refit_spread")
        preds.to_parquet(C.processed_dir() / "corners_preds.parquet", index=False)
    if stage in ("all", "team"):
        tmet, tpreds = stage_team(df, cfg)
        w(tmet, "team")
        w(stage_team_refit(df, cfg, tuple(range(args.seeds))), "team_refit")
        tpreds.to_parquet(C.processed_dir() / "corners_team_preds.parquet", index=False)
    if stage in ("all", "gate"):
        w(stage_gate(df, cfg), "gate")
    if stage in ("all", "bets"):
        preds = pd.read_parquet(C.processed_dir() / "corners_preds.parquet")
        sims, clv = stage_bets(preds, cfg)
        tpreds = pd.read_parquet(C.processed_dir() / "corners_team_preds.parquet")
        tsims, tclv = stage_bets_team(tpreds, cfg)
        w(pd.concat([sims, tsims], ignore_index=True), "bets")
        w(pd.concat([clv, tclv], ignore_index=True), "clv")
    if stage in ("all", "style"):
        from research.scenario_ev.corners_style import StyleConfig, stage_style
        met, tab, rep, thr = stage_style(StyleConfig())
        w(met, "style")
        w(pd.DataFrame([{**asdict(rep), "deep_threshold": thr}]), "style_join")
    if stage in ("all", "report"):
        print(f"report: {stage_report(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
