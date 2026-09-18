"""StatsBomb style / imputed-state layer for the corners stage (02b).

The hypothesis under test is that a team's *style* -- crossing volume, blocked shots,
passes into the box, pressing intensity, defensive line height -- and the *compactness of
its opponent* explain corners above what the opponent's rolling corner mean already says.
Style is measured from StatsBomb aggregates over strictly earlier matches of the same
season; compactness is measured by the imputed defensive state produced by the completed
privileged-tracking programme.

The join between StatsBomb and football-data.co.uk is built explicitly rather than
assumed: team names are matched by the overlap of the dates on which they play, and the
result is verified against the final scores of both sources.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.scenario_ev import common as C

#: StatsBomb competition -> football-data.co.uk division for the 2015/2016 club seasons.
COMPETITION_DIVISION: dict[str, str] = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

#: Season label used by both sources for the overlapping club seasons.
STYLE_SEASON = "2015/2016"


@dataclass(frozen=True)
class JoinReport:
    """Outcome of the StatsBomb <-> odds join.

    Attributes:
        n_sb_matches: StatsBomb matches in the overlapping seasons.
        n_joined: Matches matched to an odds row on date and mapped names.
        join_rate: ``n_joined / n_sb_matches``.
        n_teams: Team names mapped.
        score_agreement: Fraction of joined matches whose final scores agree.
        n_date_shifted: Joined matches whose odds date differs by one day.
    """

    n_sb_matches: int
    n_joined: int
    join_rate: float
    n_teams: int
    score_agreement: float
    n_date_shifted: int


def build_name_map(sb_matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Map StatsBomb team names to football-data.co.uk names by fixture-date overlap.

    Two records of the same team play on the same dates, so the number of shared match
    dates identifies the pairing without any string heuristics. The assignment is solved
    per division-season as a maximum-weight bipartite matching.

    Args:
        sb_matches: StatsBomb matches with ``date``, ``home_team``, ``away_team`` and a
            ``division`` column already attached.
        odds: Odds rows for the same division-seasons with ``MatchDate``, ``HomeTeam``,
            ``AwayTeam``, ``Division``.

    Returns:
        Frame with ``division``, ``sb_team``, ``odds_team``, ``n_shared_dates``,
        ``n_sb_dates``.
    """
    from scipy.optimize import linear_sum_assignment

    rows = []
    for div in sorted(sb_matches["division"].unique()):
        sb = sb_matches[sb_matches["division"] == div]
        od = odds[odds["Division"] == div]
        sb_long = pd.concat([
            sb[["date", "home_team"]].rename(columns={"home_team": "team"}),
            sb[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ])
        od_long = pd.concat([
            od[["MatchDate", "HomeTeam"]].rename(columns={"MatchDate": "date",
                                                          "HomeTeam": "team"}),
            od[["MatchDate", "AwayTeam"]].rename(columns={"MatchDate": "date",
                                                          "AwayTeam": "team"}),
        ])
        sb_teams = sorted(sb_long["team"].unique())
        od_teams = sorted(od_long["team"].unique())
        sb_dates = {t: set(pd.to_datetime(g["date"]).dt.normalize())
                    for t, g in sb_long.groupby("team")}
        od_dates = {t: set(pd.to_datetime(g["date"]).dt.normalize())
                    for t, g in od_long.groupby("team")}
        score = np.zeros((len(sb_teams), len(od_teams)), dtype=float)
        for i, a in enumerate(sb_teams):
            for j, b in enumerate(od_teams):
                score[i, j] = len(sb_dates[a] & od_dates[b])
        ri, ci = linear_sum_assignment(-score)
        for i, j in zip(ri, ci, strict=True):
            rows.append({"division": div, "sb_team": sb_teams[i], "odds_team": od_teams[j],
                         "n_shared_dates": float(score[i, j]),
                         "n_sb_dates": float(len(sb_dates[sb_teams[i]]))})
    return pd.DataFrame(rows)


def join_sb_to_odds(sb_matches: pd.DataFrame, odds: pd.DataFrame, name_map: pd.DataFrame,
                    max_day_shift: int = 1) -> tuple[pd.DataFrame, JoinReport]:
    """Join StatsBomb matches to odds rows on mapped names and (near-)equal dates.

    Args:
        sb_matches: StatsBomb matches with ``division`` attached.
        odds: Odds rows for the same division-seasons, carrying ``match_id``.
        name_map: Output of :func:`build_name_map`.
        max_day_shift: Allowed absolute difference in calendar days.

    Returns:
        ``(joined, report)``; ``joined`` has one row per matched StatsBomb match with the
        odds ``match_id`` and both sources' scores.
    """
    m = {(r.division, r.sb_team): r.odds_team for r in name_map.itertuples()}
    sb = sb_matches.copy()
    sb["home_odds"] = [m.get((d, t)) for d, t in zip(sb["division"], sb["home_team"], strict=True)]
    sb["away_odds"] = [m.get((d, t)) for d, t in zip(sb["division"], sb["away_team"], strict=True)]
    sb["date"] = pd.to_datetime(sb["date"]).dt.normalize()
    od = odds.copy()
    od["date"] = pd.to_datetime(od["MatchDate"]).dt.normalize()
    keep = ["match_id", "date", "Division", "HomeTeam", "AwayTeam", "FTHome", "FTAway",
            "HomeCorners", "AwayCorners"]
    best: list[pd.DataFrame] = []
    for shift in range(0, max_day_shift + 1):
        for sgn in ((1,) if shift == 0 else (1, -1)):
            left = sb.copy()
            left["join_date"] = left["date"] + pd.Timedelta(days=sgn * shift)
            j = left.merge(od[keep], left_on=["division", "home_odds", "away_odds",
                                              "join_date"],
                           right_on=["Division", "HomeTeam", "AwayTeam", "date"],
                           how="inner", suffixes=("", "_odds"))
            j["day_shift"] = sgn * shift
            best.append(j)
    joined = pd.concat(best, ignore_index=True)
    joined = joined.sort_values(["match_id_x" if "match_id_x" in joined else "match_id",
                                 "day_shift"], key=lambda s: s.abs()
                                if s.name == "day_shift" else s)
    joined = joined.drop_duplicates(subset=["sb_match_id"], keep="first")
    agree = float((joined["home_score"] == joined["FTHome"]).mean()) if len(joined) else 0.0
    rep = JoinReport(
        n_sb_matches=int(len(sb)), n_joined=int(len(joined)),
        join_rate=float(len(joined) / max(len(sb), 1)), n_teams=int(len(name_map)),
        score_agreement=agree, n_date_shifted=int((joined["day_shift"] != 0).sum()),
    )
    return joined.reset_index(drop=True), rep


def load_sb_matches() -> pd.DataFrame:
    """StatsBomb matches of the overlapping club seasons with ``division`` attached."""
    mm = pd.read_parquet(C.sb_processed_dir() / "matches.parquet")
    mm = mm[(mm["season"] == STYLE_SEASON) & mm["competition"].isin(COMPETITION_DIVISION)]
    mm = mm.rename(columns={"match_id": "sb_match_id"})
    mm["division"] = mm["competition"].map(COMPETITION_DIVISION)
    return mm.reset_index(drop=True)


#: Style statistics taken from the StatsBomb team-match aggregates.
STYLE_STATS: tuple[str, ...] = (
    "crosses", "crosses_completed", "passes_into_box", "shots_blocked", "blocks",
    "high_passes", "switches", "throw_ins", "possession", "ppda", "def_line_x",
    "pass_x_mean", "passes_final_third", "dribbles", "shots", "shots_in_box",
    "corners", "corners_against", "xg",
)

#: Imputed defensive-state columns aggregated per team-match.
IMPUTED_STATS: tuple[str, ...] = ("block_depth", "def_line", "deep_share")


def load_style_team_match() -> pd.DataFrame:
    """StatsBomb team-match aggregates for the overlapping club seasons.

    Returns:
        One row per (match, team) with ``sb_match_id``, ``team_id``, ``opp_id``, ``home``,
        ``date``, ``division`` and the :data:`STYLE_STATS` columns.
    """
    tm = pd.read_parquet(C.sb_processed_dir() / "team_match.parquet")
    mm = load_sb_matches()[["sb_match_id", "date", "division", "competition"]]
    tm = tm.rename(columns={"match_id": "sb_match_id"})
    out = tm.merge(mm, on="sb_match_id", how="inner")
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.reset_index(drop=True)


def imputed_team_match(deep_quantile: float = 0.25) -> pd.DataFrame:
    """Aggregate the imputed defensive state to one row per defending team-match.

    ``imputed_no360.parquet`` stores, for each event, the imputed shape of the *opposing*
    block as seen by the acting team. Averaging over the events of team ``a`` therefore
    describes how deep and how compact ``a``'s opponent sat, so the aggregate is assigned
    to the opponent as the defending team.

    Args:
        deep_quantile: Quantile of ``imp_def_line`` defining a "deep block" event; the
            threshold is computed once over the pooled overlapping seasons.

    Returns:
        One row per (``sb_match_id``, ``team_id``) describing that team's own block:
        ``block_depth`` (mean distance of its outfield players from their own goal),
        ``def_line`` (deepest defender) and ``deep_share`` (share of opponent events
        facing a deep line), plus ``n_events``.
    """
    import pyarrow.parquet as pq

    cols = ["match_id", "team_id", "competition", "season",
            "imp_block_depth__E2", "imp_def_line__E2"]
    df = pq.read_table(C.soccer_processed_dir() / "imputed_no360.parquet",
                       columns=cols).to_pandas()
    df = df[(df["season"] == STYLE_SEASON) & df["competition"].isin(COMPETITION_DIVISION)]
    thr = float(np.nanquantile(df["imp_def_line__E2"].to_numpy(dtype=float), deep_quantile))
    df["deep"] = (df["imp_def_line__E2"] <= thr).astype(float)
    agg = df.groupby(["match_id", "team_id"], observed=True).agg(
        block_depth=("imp_block_depth__E2", "mean"),
        def_line=("imp_def_line__E2", "mean"),
        deep_share=("deep", "mean"),
        n_events=("deep", "size"),
    ).reset_index().rename(columns={"match_id": "sb_match_id", "team_id": "actor_id"})
    agg.attrs["deep_threshold"] = thr
    return agg


def attach_imputed(style: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's own imputed block shape, measured by its opponent's events.

    Args:
        style: Team-match frame with ``sb_match_id``, ``team_id``, ``opp_id``.
        imputed: Output of :func:`imputed_team_match`, keyed by the *acting* team.

    Returns:
        ``style`` with :data:`IMPUTED_STATS` columns and ``imp_n_events``.
    """
    ren = {c: c for c in IMPUTED_STATS}
    ren.update({"actor_id": "opp_id", "n_events": "imp_n_events"})
    return style.merge(imputed.rename(columns=ren), on=["sb_match_id", "opp_id"],
                       how="left")


def add_style_priors(df: pd.DataFrame, stats: tuple[str, ...], window: int,
                     min_prior: int) -> pd.DataFrame:
    """Rolling means of style statistics over strictly earlier matches of the season.

    Args:
        df: Team-match frame with ``division``, ``team_id``, ``date``.
        stats: Columns to roll.
        window: Ignored beyond documentation; the expanding season mean is used because
            the sample is a single season.
        min_prior: Minimum prior matches for a row to be usable (applied by the caller).

    Returns:
        ``df`` with ``sp_<stat>`` columns and ``sp_n``.
    """
    from research.scenario_ev.corners_features import prior_by_date

    df = df.sort_values(["date", "sb_match_id", "team_id"]).reset_index(drop=True)
    pri = prior_by_date(df, ["division", "team_id"], "date", list(stats))
    out = df.copy()
    n = pri[f"{stats[0]}_pcnt"].to_numpy()
    out["sp_n"] = n
    for s in stats:
        cnt = pri[f"{s}_pcnt"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"sp_{s}"] = np.where(cnt > 0, pri[f"{s}_psum"].to_numpy()
                                      / np.maximum(cnt, 1.0), np.nan)
    return out


def merge_style_opponent(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach the opponent's rolling style columns, prefixed ``o``.

    Args:
        df: Team-match frame with ``sb_match_id`` and ``team_id``/``opp_id``.
        cols: Columns to copy from the opponent's row.

    Returns:
        ``df`` with ``o<col>`` for each requested column.
    """
    other = df[["sb_match_id", "team_id"] + cols].rename(
        columns={"team_id": "opp_id", **{c: f"o{c}" for c in cols}})
    return df.merge(other, on=["sb_match_id", "opp_id"], how="left", validate="one_to_one")


@dataclass(frozen=True)
class StyleConfig:
    """Configuration of the style / imputed-state layer.

    Attributes:
        min_prior: Minimum prior StatsBomb matches in the season for a usable row.
        discovery_frac: Fraction of matches in the discovery half.
        n_splits: Folds for the match-grouped cross-validation.
        seed: RNG seed.
        n_boot: Bootstrap replicates.
        alpha: L2 strength of the Poisson GLM, chosen on discovery.
        team_lines: Half-lines scored for a single team's corners.
        deep_quantile: Quantile of the imputed defensive line defining a deep block.
    """

    min_prior: int = 4
    discovery_frac: float = 0.60
    n_splits: int = 5
    seed: int = 20260918
    n_boot: int = 2000
    alpha: float = 1.0
    team_lines: tuple[float, ...] = (4.5, 5.5)
    deep_quantile: float = 0.25


def build_style_table(cfg: StyleConfig) -> tuple[pd.DataFrame, JoinReport, float]:
    """Assemble the team-match table for the style layer.

    Args:
        cfg: Style-layer configuration.

    Returns:
        ``(table, join_report, deep_threshold)``. ``table`` has one row per team-match of
        the overlapping club seasons with the odds-side proxy and market columns, the
        team's and opponent's rolling style, and the imputed defensive state.
    """
    from research.scenario_ev.corners import build_match_table
    from research.scenario_ev.corners_features import CornerConfig

    sb = load_sb_matches()
    odds = pd.read_parquet(C.odds_path())
    odds = odds.sort_values(["MatchDate", "Division", "HomeTeam", "AwayTeam"],
                            kind="mergesort").reset_index(drop=True)
    odds["match_id"] = np.arange(len(odds), dtype=np.int64)
    win = odds[odds["Division"].isin(COMPETITION_DIVISION.values())
               & (odds["MatchDate"] >= "2015-07-01") & (odds["MatchDate"] <= "2016-07-01")]
    name_map = build_name_map(sb, win)
    joined, rep = join_sb_to_odds(sb, win, name_map)

    style = load_style_team_match()
    imputed = imputed_team_match(cfg.deep_quantile)
    thr = float(imputed.attrs["deep_threshold"])
    style = attach_imputed(style, imputed)
    style = add_style_priors(style, STYLE_STATS + IMPUTED_STATS, 6, cfg.min_prior)
    sp_cols = [c for c in style.columns if c.startswith("sp_") and c != "sp_n"]
    style = merge_style_opponent(style, sp_cols + ["sp_n"])

    link = joined[["sb_match_id", "match_id", "HomeTeam", "AwayTeam", "HomeCorners",
                   "AwayCorners"]]
    style = style.merge(link, on="sb_match_id", how="inner")
    style["y_corners"] = np.where(style["home"].to_numpy() == 1,
                                  style["HomeCorners"].to_numpy(),
                                  style["AwayCorners"].to_numpy())

    match_tab = build_match_table(CornerConfig())
    keep = ["match_id", "h_proxy_lam", "a_proxy_lam", "proxy_total", "HomeElo", "AwayElo",
            "elo_gap", "p_home", "p_away", "p_over25", "p_fav",
            "h_pm_corners_for", "a_pm_corners_for", "h_pm_corners_against",
            "a_pm_corners_against", "h_pm_shots_for", "a_pm_shots_for",
            "h_pm_target_for", "a_pm_target_for", "h_pm_fouls_for", "a_pm_fouls_for",
            "h_w6_corners_for", "a_w6_corners_for"]
    keep = [c for c in keep if c in match_tab.columns]
    style = style.merge(match_tab[keep], on="match_id", how="inner")
    home = style["home"].to_numpy() == 1
    style["proxy_lam"] = np.where(home, style["h_proxy_lam"], style["a_proxy_lam"])
    style["opp_proxy_lam"] = np.where(home, style["a_proxy_lam"], style["h_proxy_lam"])
    for stat in ("pm_corners_for", "pm_corners_against", "pm_shots_for", "pm_target_for",
                 "pm_fouls_for", "w6_corners_for"):
        hc, ac = f"h_{stat}", f"a_{stat}"
        if hc in style.columns:
            style[f"own_{stat}"] = np.where(home, style[hc], style[ac])
            style[f"opp_{stat}"] = np.where(home, style[ac], style[hc])
    style["is_home"] = home.astype(int)
    style["p_team_win"] = np.where(home, style["p_home"], style["p_away"])

    usable = (style["sp_n"] >= cfg.min_prior) & (style["osp_n"] >= cfg.min_prior) \
        & style["y_corners"].notna() & np.isfinite(style["proxy_lam"])
    style = style[usable].reset_index(drop=True)
    style["split"] = C.chronological_split(style["match_id"], style["date"],
                                           cfg.discovery_frac)
    return style, rep, thr


#: Nested feature sets of the style layer.
def style_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Nested feature lists, each strictly prior to the match.

    Args:
        df: Style table from :func:`build_style_table`.

    Returns:
        Mapping from feature-set name to the columns it uses. ``B0`` is empty because the
        proxy enters every model as an offset.
    """
    odds_cols = [c for c in ("own_pm_corners_for", "opp_pm_corners_against",
                             "own_pm_corners_against", "opp_pm_corners_for",
                             "own_pm_shots_for", "opp_pm_shots_for",
                             "own_pm_target_for", "opp_pm_target_for",
                             "own_pm_fouls_for", "opp_pm_fouls_for",
                             "own_w6_corners_for", "opp_proxy_lam", "is_home",
                             "HomeElo", "AwayElo", "elo_gap", "p_team_win", "p_over25")
                 if c in df.columns]
    style_own = [f"sp_{s}" for s in STYLE_STATS
                 if s not in ("corners", "corners_against") and f"sp_{s}" in df.columns]
    style_opp = [f"osp_{s}" for s in STYLE_STATS
                 if s not in ("corners", "corners_against") and f"osp_{s}" in df.columns]
    sb_corners = [c for c in ("sp_corners", "sp_corners_against", "osp_corners",
                              "osp_corners_against") if c in df.columns]
    imp_own = [f"sp_{s}" for s in IMPUTED_STATS if f"sp_{s}" in df.columns]
    imp_opp = [f"osp_{s}" for s in IMPUTED_STATS if f"osp_{s}" in df.columns]
    sets = {
        "B0_proxy": [],
        "B1_odds_event_market": odds_cols,
        "B2a_plus_own_style": odds_cols + style_own,
        "B2_plus_style": odds_cols + style_own + style_opp,
        "B2c_plus_sb_corner_history": odds_cols + style_own + style_opp + sb_corners,
        "B3_plus_imputed_state": odds_cols + style_own + style_opp + imp_own + imp_opp,
        "B3opp_opponent_compactness_only": odds_cols + style_own + style_opp + imp_opp,
        "B3own_own_compactness_only": odds_cols + style_own + style_opp + imp_own,
        "S_style_only": style_own + style_opp,
        "S_imputed_only": imp_own + imp_opp,
    }
    return sets


def fit_poisson_offset(X_tr: pd.DataFrame, y_tr: np.ndarray, mu_tr: np.ndarray,
                       X_te: pd.DataFrame, mu_te: np.ndarray, alpha: float) -> np.ndarray:
    """Poisson GLM on top of a fixed offset, fitted by the equivalent weighted form.

    Fitting ``y ~ Poisson(mu0 * exp(X b))`` is equivalent to a Poisson regression of
    ``y / mu0`` weighted by ``mu0``, which sklearn supports directly.

    Args:
        X_tr: Training features [n_tr, d]; may be empty, giving the offset itself.
        y_tr: Training counts [n_tr].
        mu_tr: Offset means for the training rows [n_tr].
        X_te: Test features [n_te, d].
        mu_te: Offset means for the test rows [n_te].
        alpha: L2 penalty.

    Returns:
        Predicted means [n_te].
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    mu_te = np.asarray(mu_te, dtype=float)
    if X_tr.shape[1] == 0:
        return mu_te
    mu_tr = np.clip(np.asarray(mu_tr, dtype=float), 1e-6, None)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("glm", PoissonRegressor(alpha=alpha, max_iter=2000)),
    ])
    pipe.fit(X_tr, np.asarray(y_tr, dtype=float) / mu_tr, glm__sample_weight=mu_tr)
    return mu_te * np.asarray(pipe.predict(X_te), dtype=float)


def stage_style(cfg: StyleConfig) -> tuple[pd.DataFrame, pd.DataFrame, JoinReport, float]:
    """Does style, or opponent compactness, beat the rolling corner mean?

    Every model is a multiplicative correction to the C0 proxy, so ``B0_proxy`` is the
    proxy itself and a positive delta means the feature set adds something. Two
    evaluations are reported: a match-grouped 5-fold cross-validation over the whole
    overlapping sample (higher power, no temporal holdout) and the protocol's
    chronological discovery / confirmation split (the headline).

    Args:
        cfg: Style-layer configuration.

    Returns:
        ``(metrics, table, join_report, deep_threshold)``.
    """
    from research.privileged_tracking.common.metrics import log_loss, per_sample_log_loss
    from research.privileged_tracking.common.splits import group_kfold
    from research.scenario_ev.corners import poisson_deviance

    df, rep, thr = build_style_table(cfg)
    sets = style_feature_sets(df)
    y = df["y_corners"].to_numpy(dtype=float)
    mu0 = df["proxy_lam"].to_numpy(dtype=float)
    groups = df["match_id"].to_numpy()
    disc = df["split"].to_numpy() == C.DISCOVERY
    preds: dict[tuple[str, str], np.ndarray] = {}

    for name, cols in sets.items():
        X = df[cols] if cols else df[[]]
        cv = np.full(len(df), np.nan)
        for tr, te in group_kfold(groups, n_splits=cfg.n_splits, seed=cfg.seed):
            cv[te] = fit_poisson_offset(X.iloc[tr], y[tr], mu0[tr], X.iloc[te], mu0[te],
                                        cfg.alpha)
        preds[(name, "cv")] = cv
        fwd = np.full(len(df), np.nan)
        fwd[~disc] = fit_poisson_offset(X[disc], y[disc], mu0[disc], X[~disc], mu0[~disc],
                                        cfg.alpha)
        preds[(name, "forward")] = fwd

    r_nb = C.fit_nb_dispersion(y[disc], mu0[disc])
    rows: list[dict[str, object]] = []
    for name in sets:
        for scheme, mask in (("cv_all", np.ones(len(df), dtype=bool)),
                             ("cv_discovery", disc),
                             ("forward_confirmation", ~disc)):
            key = "cv" if scheme.startswith("cv") else "forward"
            p = preds[(name, key)]
            b0 = preds[("B0_proxy", key)]
            ok = mask & np.isfinite(p) & np.isfinite(b0)
            if ok.sum() < 50:
                continue
            d0 = poisson_deviance(y[ok], b0[ok])
            dm = poisson_deviance(y[ok], p[ok])
            delta, lo, hi = C.clustered_bootstrap_mean(d0 - dm, groups[ok], cfg.n_boot,
                                                       cfg.seed)
            row: dict[str, object] = {
                "feature_set": name, "scheme": scheme, "n_rows": int(ok.sum()),
                "n_matches": int(pd.unique(groups[ok]).size),
                "poisson_deviance": float(np.mean(dm)),
                "delta_vs_B0": delta, "ci_lo": lo, "ci_hi": hi,
                "mean_pred": float(p[ok].mean()), "mean_y": float(y[ok].mean()),
            }
            for line in cfg.team_lines:
                yb = (y[ok] > line).astype(float)
                pm = C.nb_sf(line, p[ok], r_nb)
                pb = C.nb_sf(line, b0[ok], r_nb)
                dl, llo, lhi = C.clustered_bootstrap_mean(
                    per_sample_log_loss(yb, pb) - per_sample_log_loss(yb, pm),
                    groups[ok], cfg.n_boot, cfg.seed)
                row[f"ll_{line}"] = log_loss(yb, pm)
                row[f"delta_ll_{line}"] = dl
                row[f"delta_ll_{line}_lo"] = llo
                row[f"delta_ll_{line}_hi"] = lhi
            rows.append(row)
    return pd.DataFrame(rows), df, rep, thr
