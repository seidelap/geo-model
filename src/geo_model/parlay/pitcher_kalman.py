"""Granular Kalman filter over MLB market errors: starting pitcher, offense, defense.

Extends :mod:`geo_model.parlay.kalman` (one error per team) to a state with one
entry per starting pitcher, one per team offense and one per team
defense/bullpen, all measured in runs per game relative to the closing market.
Each game contributes two observations, the home and away run residuals
against the market-implied runs::

    implied_home_runs = (total_line + spread_line) / 2
    implied_away_runs = (total_line - spread_line) / 2
    home_run_resid    = home_score - implied_home_runs = o[home] - x[away SP] - d[away] + noise
    away_run_resid    = away_score - implied_away_runs = o[away] - x[home SP] - d[home] + noise

where ``o`` is an offense error (positive = scores more than the market
thinks), ``x`` a pitcher error and ``d`` a defense/bullpen error (positive =
allows fewer runs than the market thinks). Every state entry is a stationary
AR(1) process in calendar days with a per-block stationary standard deviation
and daily persistence, so pitcher errors carry across the pitcher's starts
(5-day cadence), across teams and across seasons, and the filter accumulates
information over every game a pitcher, offense or defense has been involved in.

For two games priced on the same slate the predicted covariance of their
margin residuals is ``c_mᵀ H1 P H2ᵀ c_m`` with ``c_m = (1, -1)`` and ``P`` the
prior covariance at the start of the slate; for totals ``c_t = (1, 1)``. For a
later game the covariance is propagated through the AR(1) dynamics. All
quantities for a slate are computed before any game of that slate is observed.

Also holds the NBA rest-day helpers used for the back-to-back stratification of
the shared-game pairs (the NBA source carries ``Days_Rest_Home/Away``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from geo_model.parlay.data_multisport import MultiSportConfig

PITCHER, OFFENSE, DEFENSE = 0, 1, 2
MARGIN = np.array([1.0, -1.0])
TOTAL = np.array([1.0, 1.0])
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GranularParams:
    """Hyperparameters of the pitcher / offense / defense error processes.

    Every block is a stationary AR(1) in calendar days: stationary std ``s``
    and daily persistence ``phi``; the innovation variance over a gap of
    ``k`` days is ``s² (1 - phi^(2k))`` so the prior for a new entity is the
    stationary distribution.

    Attributes:
        pitcher_std: Stationary std of a starting pitcher's market error (runs).
        pitcher_persistence: Daily AR(1) coefficient of the pitcher error.
        offense_std: Stationary std of a team's offense error (runs).
        offense_persistence: Daily AR(1) coefficient of the offense error.
        defense_std: Stationary std of a team's defense/bullpen error (runs).
        defense_persistence: Daily AR(1) coefficient of the defense error.
        obs_std: Std of a team's runs around its market-implied expectation.
        obs_corr: Correlation of the two noise terms of one game.
    """

    pitcher_std: float = 0.5
    pitcher_persistence: float = 0.99
    offense_std: float = 0.3
    offense_persistence: float = 0.98
    defense_std: float = 0.3
    defense_persistence: float = 0.98
    obs_std: float = 3.0
    obs_corr: float = 0.0

    def stds(self) -> np.ndarray:
        """Stationary stds indexed by block ``[PITCHER, OFFENSE, DEFENSE]``."""
        return np.array([self.pitcher_std, self.offense_std, self.defense_std])

    def persistences(self) -> np.ndarray:
        """Daily persistences indexed by block ``[PITCHER, OFFENSE, DEFENSE]``."""
        return np.array([self.pitcher_persistence, self.offense_persistence, self.defense_persistence])

    def to_vector(self) -> np.ndarray:
        """Unconstrained parameterization for the optimizer."""
        logit = lambda p: np.log(p / (1 - p))  # noqa: E731
        return np.array(
            [
                np.log(self.pitcher_std), logit(self.pitcher_persistence),
                np.log(self.offense_std), logit(self.offense_persistence),
                np.log(self.defense_std), logit(self.defense_persistence),
                np.log(self.obs_std), np.arctanh(self.obs_corr),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> GranularParams:
        """Inverse of :meth:`to_vector`."""
        sig = lambda z: float(1 / (1 + np.exp(-z)))  # noqa: E731
        return cls(
            pitcher_std=float(np.exp(v[0])), pitcher_persistence=sig(v[1]),
            offense_std=float(np.exp(v[2])), offense_persistence=sig(v[3]),
            defense_std=float(np.exp(v[4])), defense_persistence=sig(v[5]),
            obs_std=float(np.exp(v[6])), obs_corr=float(np.tanh(v[7])),
        )


@dataclass(frozen=True)
class RunOffsets:
    """Mean run residuals subtracted before filtering (fit on training seasons only).

    Attributes:
        home: Mean of ``home_run_resid`` on the training seasons.
        away: Mean of ``away_run_resid`` on the training seasons.
    """

    home: float = 0.0
    away: float = 0.0

    @classmethod
    def fit(cls, games: pd.DataFrame) -> RunOffsets:
        """Offsets from the mean run residuals of ``games``."""
        r = run_residuals(games)
        return cls(home=float(r["home_run_resid"].mean()), away=float(r["away_run_resid"].mean()))


def run_residuals(games: pd.DataFrame) -> pd.DataFrame:
    """Attach market-implied runs and the two per-game run residuals.

    Args:
        games: Cleaned games with ``spread_line`` (expected home minus away
            margin), ``total_line``, ``home_score``, ``away_score``.

    Returns:
        Copy with ``implied_home_runs, implied_away_runs, home_run_resid,
        away_run_resid``. Note ``home_run_resid - away_run_resid == resid`` and
        ``home_run_resid + away_run_resid == tresid``.
    """
    g = games.copy()
    g["implied_home_runs"] = (g["total_line"] + g["spread_line"]) / 2.0
    g["implied_away_runs"] = (g["total_line"] - g["spread_line"]) / 2.0
    g["home_run_resid"] = g["home_score"] - g["implied_home_runs"]
    g["away_run_resid"] = g["away_score"] - g["implied_away_runs"]
    return g


@dataclass
class EntityIndex:
    """State layout: one entry per pitcher, team offense and team defense.

    Attributes:
        names: Entity names (``p:<pitcher>``, ``o:<team>``, ``d:<team>``).
        kind: Block id per entity (``PITCHER``, ``OFFENSE`` or ``DEFENSE``).
        index: Name to position.
    """

    names: list[str]
    kind: np.ndarray
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {n: i for i, n in enumerate(self.names)}

    @classmethod
    def from_games(cls, games: pd.DataFrame) -> EntityIndex:
        """Entities appearing in ``games`` (pitchers from ``home_qb_id``/``away_qb_id``)."""
        teams = sorted(set(games["home_team"]) | set(games["away_team"]))
        pitchers = sorted(set(games["home_qb_id"].dropna()) | set(games["away_qb_id"].dropna()))
        names = [f"p:{p}" for p in pitchers] + [f"o:{t}" for t in teams] + [f"d:{t}" for t in teams]
        kind = np.array([PITCHER] * len(pitchers) + [OFFENSE] * len(teams) + [DEFENSE] * len(teams), dtype=int)
        return cls(names=names, kind=kind)

    def game_rows(self, home: str, away: str, home_sp: str, away_sp: str) -> np.ndarray:
        """Dense ``[2, n]`` observation rows for one game (home runs, away runs)."""
        h = np.zeros((2, len(self.names)))
        h[0, self.index[f"o:{home}"]] = 1.0
        h[0, self.index[f"p:{away_sp}"]] = -1.0
        h[0, self.index[f"d:{away}"]] = -1.0
        h[1, self.index[f"o:{away}"]] = 1.0
        h[1, self.index[f"p:{home_sp}"]] = -1.0
        h[1, self.index[f"d:{home}"]] = -1.0
        return h


@dataclass
class GranularOutput:
    """Predictions from one filter pass.

    Attributes:
        games: Per-game rows: ``game_id, season, gameday, pred_home_mean,
            pred_away_mean, pred_margin_mean, pred_margin_var, pred_total_mean,
            pred_total_var, resid, tresid`` (predicted means include the offsets;
            residuals are the raw market residuals).
        pairs: Same-slate game pairs with ``pred_cov, pred_corr, resid_1,
            resid_2`` for margins and ``pred_tcov, pred_tcorr, tresid_1,
            tresid_2`` for totals.
        cross_pairs: Requested (possibly cross-slate) pairs with the same
            columns plus ``gap_days`` and ``pred_margin_mean_2`` (game 2's
            predicted margin residual as of game 1's slate).
        log_likelihood: Sum of Gaussian predictive log densities of the run
            residual vectors, all seasons.
        log_likelihood_by_season: The same split by season.
        n_obs_by_season: Number of run observations (2 per game) per season.
    """

    games: pd.DataFrame
    pairs: pd.DataFrame
    cross_pairs: pd.DataFrame
    log_likelihood: float
    log_likelihood_by_season: dict[int, float]
    n_obs_by_season: dict[int, int]


def _prepare(games: pd.DataFrame) -> pd.DataFrame:
    g = run_residuals(games)
    g = g[g["home_qb_id"].notna() & g["away_qb_id"].notna()]
    g["gameday"] = pd.to_datetime(g["gameday"], utc=True).dt.normalize()
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


class GranularKalman:
    """Daily-slate Kalman filter over pitcher / offense / defense market errors.

    Args:
        params: Filter hyperparameters.
        offsets: Mean run residuals to subtract (fit on training data).
        max_gap_days: Cap on the calendar gap used for state propagation, so
            an off-season counts as this many days of drift.
    """

    def __init__(
        self,
        params: GranularParams | None = None,
        offsets: RunOffsets | None = None,
        max_gap_days: int = 30,
    ) -> None:
        self.params = params or GranularParams()
        self.offsets = offsets or RunOffsets()
        self.max_gap_days = max_gap_days

    def run(
        self,
        games: pd.DataFrame,
        emit_pairs: bool = True,
        pair_requests: pd.DataFrame | None = None,
        emit_from_season: int | None = None,
    ) -> GranularOutput:
        """Filter all games in chronological order, one slate per calendar day.

        Args:
            games: Cleaned games with ``game_id, season, gameday, home_team,
                away_team, home_qb_id, away_qb_id, home_score, away_score,
                spread_line, total_line, resid, tresid``. Games with a missing
                starter are skipped.
            emit_pairs: Build the (quadratic-size) same-slate pairs table.
            pair_requests: Optional rows ``game_id_1, game_id_2`` whose
                predicted covariance is wanted, evaluated at the slate of
                ``game_id_1`` (which must not be after ``game_id_2``).
            emit_from_season: Emit per-game rows and pairs only for seasons
                ``>=`` this (the likelihood is still accumulated for all).

        Returns:
            :class:`GranularOutput` with predictions made strictly before each slate.
        """
        p = self.params
        g = _prepare(games)
        ent = EntityIndex.from_games(g)
        n = len(ent.names)
        s = p.stds()[ent.kind]
        phi = p.persistences()[ent.kind]
        r_block = np.array([[1.0, p.obs_corr], [p.obs_corr, 1.0]]) * p.obs_std**2
        info = {
            gid: (day, ent.game_rows(ht, at, hs, as_))
            for gid, day, ht, at, hs, as_ in zip(
                g["game_id"], g["gameday"], g["home_team"], g["away_team"], g["home_qb_id"], g["away_qb_id"]
            )
        }
        requests: dict[str, list[str]] = {}
        if pair_requests is not None:
            for g1, g2 in zip(pair_requests["game_id_1"], pair_requests["game_id_2"]):
                if g1 in info and g2 in info:
                    requests.setdefault(g1, []).append(g2)
        m = np.zeros(n)
        P = np.diag(s**2)
        prev_day: pd.Timestamp | None = None
        ll_by_season: dict[int, float] = {}
        n_by_season: dict[int, int] = {}
        game_rows: list[dict] = []
        pair_frames: list[pd.DataFrame] = []
        cross_rows: list[dict] = []
        for day, sg in g.groupby("gameday", sort=True):
            if prev_day is not None:
                d = phi ** min((day - prev_day).days, self.max_gap_days)
                m = d * m
                P = P * np.outer(d, d)
                P[np.diag_indices(n)] += s**2 * (1 - d**2)
            prev_day = day
            G = len(sg)
            gids = sg["game_id"].to_numpy()
            H = np.vstack([info[gid][1] for gid in gids])
            y = np.empty(2 * G)
            y[0::2] = sg["home_run_resid"].to_numpy() - self.offsets.home
            y[1::2] = sg["away_run_resid"].to_numpy() - self.offsets.away
            PHt = P @ H.T
            S = H @ PHt + np.kron(np.eye(G), r_block)
            mu = H @ m
            L = np.linalg.cholesky(S)
            innov = y - mu
            z = linalg.solve_triangular(L, innov, lower=True)
            season = int(sg["season"].iloc[0])
            ll_by_season[season] = ll_by_season.get(season, 0.0) + float(-0.5 * z @ z - np.log(np.diag(L)).sum() - G * LOG_2PI)
            n_by_season[season] = n_by_season.get(season, 0) + 2 * G
            emit = emit_from_season is None or season >= emit_from_season
            if emit:
                Cm = np.kron(np.eye(G), MARGIN)
                Ct = np.kron(np.eye(G), TOTAL)
                Vm = Cm @ S @ Cm.T
                Vt = Ct @ S @ Ct.T
                mh = mu[0::2] + self.offsets.home
                ma = mu[1::2] + self.offsets.away
                for i in range(G):
                    game_rows.append(
                        {
                            "game_id": gids[i], "season": season, "gameday": day,
                            "pred_home_mean": mh[i], "pred_away_mean": ma[i],
                            "pred_home_var": S[2 * i, 2 * i], "pred_away_var": S[2 * i + 1, 2 * i + 1],
                            "pred_margin_mean": mh[i] - ma[i], "pred_margin_var": Vm[i, i],
                            "pred_total_mean": mh[i] + ma[i], "pred_total_var": Vt[i, i],
                            "resid": float(sg["resid"].iloc[i]), "tresid": float(sg["tresid"].iloc[i]),
                        }
                    )
                if emit_pairs and G > 1:
                    iu, ju = np.triu_indices(G, 1)
                    sd_m, sd_t = np.sqrt(np.diag(Vm)), np.sqrt(np.diag(Vt))
                    pair_frames.append(
                        pd.DataFrame(
                            {
                                "season": season, "gameday": day,
                                "game_id_1": gids[iu], "game_id_2": gids[ju],
                                "pred_cov": Vm[iu, ju], "pred_corr": Vm[iu, ju] / (sd_m[iu] * sd_m[ju]),
                                "pred_tcov": Vt[iu, ju], "pred_tcorr": Vt[iu, ju] / (sd_t[iu] * sd_t[ju]),
                                "resid_1": sg["resid"].to_numpy()[iu], "resid_2": sg["resid"].to_numpy()[ju],
                                "tresid_1": sg["tresid"].to_numpy()[iu], "tresid_2": sg["tresid"].to_numpy()[ju],
                            }
                        )
                    )
                for i, gid in enumerate(gids):
                    for g2 in requests.get(gid, []):
                        cross_rows.append(
                            self._cross_prediction(gid, g2, i, day, info, m, P, S, H, s, phi, r_block, sg)
                        )
            K = linalg.cho_solve((L, True), PHt.T).T
            m = m + K @ innov
            P = P - K @ PHt.T
            P = 0.5 * (P + P.T)
        pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
        return GranularOutput(
            games=pd.DataFrame(game_rows),
            pairs=pairs,
            cross_pairs=pd.DataFrame(cross_rows),
            log_likelihood=float(sum(ll_by_season.values())),
            log_likelihood_by_season=ll_by_season,
            n_obs_by_season=n_by_season,
        )

    def _cross_prediction(
        self, gid1: str, gid2: str, i: int, day: pd.Timestamp, info: dict, m: np.ndarray, P: np.ndarray,
        S: np.ndarray, H: np.ndarray, s: np.ndarray, phi: np.ndarray, r_block: np.ndarray, sg: pd.DataFrame,
    ) -> dict:
        """Predicted covariance between game 1 (on this slate) and a later game 2."""
        day2, h2 = info[gid2]
        gap = (day2 - day).days
        if gap < 0:
            raise ValueError(f"pair request {gid1}->{gid2} is not in chronological order")
        d2 = phi ** min(gap, self.max_gap_days)
        h2d = h2 * d2  # rows of H2 Φ^gap
        cov12 = H[2 * i : 2 * i + 2] @ (P @ h2d.T)  # Cov(y1, y2) as of this slate  [2, 2]
        var2 = h2d @ P @ h2d.T + (h2 * (s**2 * (1 - d2**2))) @ h2.T + r_block
        mean2 = h2 @ (d2 * m)
        s11 = S[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        vm1, vm2 = MARGIN @ s11 @ MARGIN, MARGIN @ var2 @ MARGIN
        vt1, vt2 = TOTAL @ s11 @ TOTAL, TOTAL @ var2 @ TOTAL
        return {
            "game_id_1": gid1, "game_id_2": gid2, "gap_days": gap, "season": int(sg["season"].iloc[0]),
            "pred_cov": MARGIN @ cov12 @ MARGIN, "pred_corr": (MARGIN @ cov12 @ MARGIN) / np.sqrt(vm1 * vm2),
            "pred_tcov": TOTAL @ cov12 @ TOTAL, "pred_tcorr": (TOTAL @ cov12 @ TOTAL) / np.sqrt(vt1 * vt2),
            "pred_margin_mean_2": mean2[0] - mean2[1] + self.offsets.home - self.offsets.away,
            "resid_1": float(sg["resid"].iloc[i]), "tresid_1": float(sg["tresid"].iloc[i]),
        }

    def fit(self, games: pd.DataFrame, max_iter: int = 400) -> GranularParams:
        """Maximize the predictive log-likelihood over the hyperparameters.

        Args:
            games: Training games (must precede any evaluation games).
            max_iter: Nelder-Mead iteration budget.

        Returns:
            Fitted parameters (also stored on ``self.params``).
        """

        def neg_ll(v: np.ndarray) -> float:
            out = GranularKalman(GranularParams.from_vector(v), self.offsets, self.max_gap_days).run(games, emit_pairs=False, emit_from_season=10**6)
            return -out.log_likelihood

        res = optimize.minimize(neg_ll, self.params.to_vector(), method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2})
        self.params = GranularParams.from_vector(res.x)
        return self.params


def null_log_likelihood(games: pd.DataFrame, obs_std: float, obs_corr: float, offsets: RunOffsets) -> dict[int, float]:
    """Predictive log-likelihood per season of the no-state model (pure noise).

    Args:
        games: Cleaned games (rows with a missing starter are skipped, as in
            :meth:`GranularKalman.run`).
        obs_std: Noise std.
        obs_corr: Within-game noise correlation.
        offsets: Mean run residuals to subtract.

    Returns:
        Season -> summed log density of the ``(home, away)`` run residuals.
    """
    g = _prepare(games)
    R = np.array([[1.0, obs_corr], [obs_corr, 1.0]]) * obs_std**2
    Ri = np.linalg.inv(R)
    logdet = float(np.log(np.linalg.det(R)))
    y = np.column_stack([g["home_run_resid"] - offsets.home, g["away_run_resid"] - offsets.away])
    q = np.einsum("ij,jk,ik->i", y, Ri, y)
    ll = -0.5 * q - 0.5 * logdet - LOG_2PI
    return pd.Series(ll, index=g["season"].to_numpy()).groupby(level=0).sum().to_dict()


@dataclass(frozen=True)
class TheoryCorr:
    """Closed-form pair correlations implied by a parameter set.

    One shared game between pitcher P (team A) and team B; leg 1 is P's next
    start (team A vs C), leg 2 is B's next game (vs D, a different starter for
    B). With ``V = s_p² + s_o² + s_d² + σ²`` the posterior covariances after the
    shared game are ``Cov(x_P, o_B) = s_p² s_o² / V`` and
    ``Cov(d_A, o_B) = Cov(o_A, d_B) = s_d² s_o² / V``, so

        Cov(margin_A, margin_B) = (s_p² s_o² + 2 s_d² s_o²) / V   (positive)
        Cov(total_1, total_2)   = -(s_p² s_o² + 2 s_d² s_o²) / V  (negative)

    divided by the margin (total) residual variance
    ``2 (s_p² + s_o² + s_d²) + 2 σ² (1 ∓ ρ)``. ``*_team`` drops the pitcher
    term (H's next game vs A's next game, different starters). ``*_bound`` is
    the supremum over any amount of accumulated evidence, obtained by replacing
    each posterior covariance with the Cauchy-Schwarz limit ``s_i s_j``.

    Attributes:
        margin_pitcher: Margin-residual correlation for pitcher pairs.
        total_pitcher: Total-residual correlation for pitcher pairs.
        margin_team: Margin-residual correlation for team-level pairs.
        margin_bound: Upper bound on the margin correlation.
        total_bound: Upper bound on the magnitude of the total correlation.
    """

    margin_pitcher: float
    total_pitcher: float
    margin_team: float
    margin_bound: float
    total_bound: float


def theory_pair_corr(params: GranularParams) -> TheoryCorr:
    """Evaluate :class:`TheoryCorr` for ``params``."""
    sp2, so2, sd2 = params.pitcher_std**2, params.offense_std**2, params.defense_std**2
    o2, rho = params.obs_std**2, params.obs_corr
    V = sp2 + so2 + sd2 + o2
    var_m = 2 * (sp2 + so2 + sd2) + 2 * o2 * (1 - rho)
    var_t = 2 * (sp2 + so2 + sd2) + 2 * o2 * (1 + rho)
    cov = (sp2 * so2 + 2 * sd2 * so2) / V
    bound = params.pitcher_std * params.offense_std + 2 * params.defense_std * params.offense_std
    return TheoryCorr(
        margin_pitcher=cov / var_m,
        total_pitcher=-cov / var_t,
        margin_team=(2 * sd2 * so2 / V) / var_m,
        margin_bound=bound / var_m,
        total_bound=bound / var_t,
    )


def simulate_granular(
    games: pd.DataFrame,
    params: GranularParams,
    seed: int = 0,
    market: str = "efficient",
    max_gap_days: int = 30,
) -> pd.DataFrame:
    """Draw synthetic run residuals from the generative model on a real schedule.

    Args:
        games: Schedule with ``game_id, season, gameday, home_team, away_team,
            home_qb_id, away_qb_id, spread_line, total_line`` (results are
            replaced).
        params: Generative parameters for the latent errors and noise.
        seed: RNG seed.
        market: ``"efficient"`` prices each game at the Bayesian posterior mean
            given all earlier games (a Kalman filter with the true parameters),
            so residuals are innovations relative to a market that updates on
            every game; ``"local"`` is the same market but blind to the network:
            it keeps only the diagonal of its covariance, so it updates every
            entity on its own games and never propagates explaining-away
            (the hypothesis under test); ``"static"`` never updates.
        max_gap_days: Cap on the calendar gap used for state propagation.

    Returns:
        Copy of ``games`` (rows with a missing starter dropped) with synthetic
        ``home_score, away_score, result, total, resid, tresid``.
    """
    if market not in {"efficient", "static", "local"}:
        raise ValueError(f"unknown market mode {market!r}")
    rng = np.random.default_rng(seed)
    g = _prepare(games)
    ent = EntityIndex.from_games(g)
    n = len(ent.names)
    s = params.stds()[ent.kind]
    phi = params.persistences()[ent.kind]
    r_block = np.array([[1.0, params.obs_corr], [params.obs_corr, 1.0]]) * params.obs_std**2
    Lr = np.linalg.cholesky(r_block)
    x = rng.normal(0.0, s)
    m = np.zeros(n)
    P = np.diag(s**2)
    prev_day: pd.Timestamp | None = None
    rh = np.full(len(g), np.nan)
    ra = np.full(len(g), np.nan)
    for day, sg in g.groupby("gameday", sort=True):
        if prev_day is not None:
            d = phi ** min((day - prev_day).days, max_gap_days)
            x = d * x + rng.normal(0.0, 1.0, n) * np.sqrt(s**2 * (1 - d**2))
            m = d * m
            P = P * np.outer(d, d)
            P[np.diag_indices(n)] += s**2 * (1 - d**2)
        prev_day = day
        G = len(sg)
        H = np.vstack([ent.game_rows(ht, at, hs, as_) for ht, at, hs, as_ in zip(sg["home_team"], sg["away_team"], sg["home_qb_id"], sg["away_qb_id"])])
        noise = (rng.normal(size=(G, 2)) @ Lr.T).reshape(-1)
        y = H @ x + noise
        if market in {"efficient", "local"}:
            PHt = P @ H.T
            S = H @ PHt + np.kron(np.eye(G), r_block)
            mu = H @ m
            K = PHt @ np.linalg.inv(S)
            m = m + K @ (y - mu)
            P = P - K @ PHt.T
            P = 0.5 * (P + P.T)
            if market == "local":
                P = np.diag(np.diag(P))
            y = y - mu
        rh[sg.index] = y[0::2]
        ra[sg.index] = y[1::2]
    g["home_run_resid"], g["away_run_resid"] = rh, ra
    g["resid"] = g["home_run_resid"] - g["away_run_resid"]
    g["tresid"] = g["home_run_resid"] + g["away_run_resid"]
    g["result"] = g["spread_line"] + g["resid"]
    g["total"] = g["total_line"] + g["tresid"]
    g["home_score"] = (g["total"] + g["result"]) / 2.0
    g["away_score"] = (g["total"] - g["result"]) / 2.0
    return g


def orient_pair_predictions(pairs: pd.DataFrame, cross: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach cross-slate predictions to team-perspective leg pairs.

    The filter predicts correlations of *home-perspective* margin residuals.
    Pairs from :func:`geo_model.parlay.multisport_backtest.build_pitcher_pairs`
    carry residuals from the perspective of ``team_h`` and ``team_a``, so the
    predicted margin correlation is flipped once for every leg played away.
    Total correlations are orientation-free.

    Args:
        pairs: Leg pairs with ``team_h, team_a, h_next_game_id, a_next_game_id``.
        cross: ``GranularOutput.cross_pairs`` (``game_id_1`` is the earlier leg).
        games: Cleaned games (for ``home_team`` of each leg).

    Returns:
        ``pairs`` with ``pred_corr`` (team perspective), ``pred_tcorr``,
        ``pred_cov`` and ``gap_days`` merged in; unmatched pairs get NaN.
    """
    home = games.set_index("game_id")["home_team"]
    out = pairs.copy()
    req = pair_requests_from_pairs(out, games, keep_duplicates=True)
    g1, g2 = req["game_id_1"].to_numpy(), req["game_id_2"].to_numpy()
    key = cross.set_index(["game_id_1", "game_id_2"])
    rows = key.reindex(pd.MultiIndex.from_arrays([g1, g2]))
    sign_h = np.where(home.reindex(out["h_next_game_id"]).to_numpy() == out["team_h"].to_numpy(), 1.0, -1.0)
    sign_a = np.where(home.reindex(out["a_next_game_id"]).to_numpy() == out["team_a"].to_numpy(), 1.0, -1.0)
    out["pred_corr"] = rows["pred_corr"].to_numpy() * sign_h * sign_a
    out["pred_cov"] = rows["pred_cov"].to_numpy() * sign_h * sign_a
    out["pred_tcorr"] = rows["pred_tcorr"].to_numpy()
    out["gap_days"] = rows["gap_days"].to_numpy()
    return out


def pair_requests_from_pairs(pairs: pd.DataFrame, games: pd.DataFrame, keep_duplicates: bool = False) -> pd.DataFrame:
    """Chronologically ordered ``game_id_1, game_id_2`` requests for leg pairs.

    Args:
        pairs: Leg pairs with ``h_next_game_id, a_next_game_id``.
        games: Cleaned games (for the leg dates).
        keep_duplicates: Keep one row per input pair (aligned with ``pairs``)
            instead of de-duplicating.

    Returns:
        Requests where ``game_id_1`` is the earlier leg (ties keep leg 1 first).
    """
    day = games.set_index("game_id")["gameday"]
    first = day.reindex(pairs["h_next_game_id"]).to_numpy() <= day.reindex(pairs["a_next_game_id"]).to_numpy()
    req = pd.DataFrame(
        {
            "game_id_1": np.where(first, pairs["h_next_game_id"], pairs["a_next_game_id"]),
            "game_id_2": np.where(first, pairs["a_next_game_id"], pairs["h_next_game_id"]),
        }
    )
    return req if keep_duplicates else req.drop_duplicates()


def same_day_pairs_for_moneyline(filter_pairs: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Re-shape filter pairs into the layout ``moneyline_parlay`` expects.

    ``team_h`` is the home team of game 1. ``team_a`` is the home team of game
    2 when the predicted margin correlation is non-negative and its away team
    otherwise, so that the (both win, both lose) parlays follow the filter's
    predicted direction.

    Args:
        filter_pairs: ``GranularOutput.pairs`` (or ``cross_pairs``) with
            ``game_id_1, game_id_2, pred_corr``.
        games: Cleaned games.

    Returns:
        Table with ``team_h, team_a, h_next_game_id, a_next_game_id, pred_corr``.
    """
    gi = games.set_index("game_id")
    h1 = gi["home_team"].reindex(filter_pairs["game_id_1"]).to_numpy()
    h2 = gi["home_team"].reindex(filter_pairs["game_id_2"]).to_numpy()
    a2 = gi["away_team"].reindex(filter_pairs["game_id_2"]).to_numpy()
    pos = filter_pairs["pred_corr"].to_numpy() >= 0
    return pd.DataFrame(
        {
            "team_h": h1,
            "team_a": np.where(pos, h2, a2),
            "h_next_game_id": filter_pairs["game_id_1"].to_numpy(),
            "a_next_game_id": filter_pairs["game_id_2"].to_numpy(),
            "pred_corr": filter_pairs["pred_corr"].to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# Cluster-aware inference for same-slate pairs (pairs sharing a game are dependent)
# --------------------------------------------------------------------------- #


def _phi_from_counts(n: float, sx: float, sy: float, sxy: float) -> float:
    if n == 0:
        return float("nan")
    px, py, pxy = sx / n, sy / n, sxy / n
    den = np.sqrt(px * (1 - px) * py * (1 - py))
    return float((pxy - px * py) / den) if den > 0 else float("nan")


def cluster_bootstrap_phi(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    clusters: pd.Series | np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, tuple[float, float], int]:
    """Phi of two cover indicators with a cluster bootstrap CI.

    Pairs built from all games on one slate share games, so pair-level
    resampling understates the uncertainty; here whole clusters (slates) are
    resampled with replacement.

    Args:
        x: Leg-1 residuals (positive = cover).
        y: Leg-2 residuals.
        clusters: Cluster label per pair (e.g. the slate date).
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(phi, (ci_lo, ci_hi), n_pairs)`` after dropping pushes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cl = np.asarray(clusters)
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    cx, cy, cl = (x[mask] > 0).astype(float), (y[mask] > 0).astype(float), cl[mask]
    if len(cx) < 3:
        return float("nan"), (float("nan"), float("nan")), int(len(cx))
    _, inv = np.unique(cl, return_inverse=True)
    k = inv.max() + 1
    n_c = np.bincount(inv, minlength=k).astype(float)
    sx = np.bincount(inv, weights=cx, minlength=k)
    sy = np.bincount(inv, weights=cy, minlength=k)
    sxy = np.bincount(inv, weights=cx * cy, minlength=k)
    phi = _phi_from_counts(n_c.sum(), sx.sum(), sy.sum(), sxy.sum())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        boots[b] = _phi_from_counts(w @ n_c, w @ sx, w @ sy, w @ sxy)
    return phi, (float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))), int(len(cx))


def cluster_bootstrap_ratio(
    numerators: np.ndarray, denominators: np.ndarray, n_boot: int = 500, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Ratio of sums with a cluster bootstrap CI (one entry per cluster).

    Args:
        numerators: Per-cluster sums of the quantity of interest (e.g. profit
            minus independence-expected profit).
        denominators: Per-cluster sums of the normaliser (e.g. units staked).
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(ratio, (ci_lo, ci_hi))``.
    """
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    k = len(num)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        boots[b] = (w @ num) / (w @ den)
    return float(num.sum() / den.sum()), (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))


def permutation_slope_pvalue(
    pairs: pd.DataFrame, cluster_col: str = "gameday", n_perm: int = 200, seed: int = 0
) -> tuple[float, float]:
    """Calibration slope with a p-value from permuting predictions within clusters.

    Residual products are heavy-tailed and pairs within a slate share games, so
    the OLS standard error of the slope of ``resid_1 * resid_2`` on ``pred_cov``
    is unreliable. The predicted covariances are shuffled among the pairs of
    each cluster, which preserves the slate structure and the marginal
    distributions.

    Args:
        pairs: Pairs with ``pred_cov, resid_1, resid_2`` and ``cluster_col``.
        cluster_col: Cluster label column.
        n_perm: Permutations.
        seed: RNG seed.

    Returns:
        ``(slope, two_sided_p)``.
    """
    y = (pairs["resid_1"] * pairs["resid_2"]).to_numpy(dtype=float)
    x = pairs["pred_cov"].to_numpy(dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")

    def slope(xv: np.ndarray) -> float:
        xc = xv - xv.mean()
        return float(xc @ (y - y.mean()) / (xc @ xc))

    obs = slope(x)
    rng = np.random.default_rng(seed)
    codes = pd.factorize(pairs[cluster_col])[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)
    count = 0
    for _ in range(n_perm):
        xp = x.copy()
        for gidx in groups:
            xp[gidx] = x[rng.permutation(gidx)]
        if abs(slope(xp)) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# --------------------------------------------------------------------------- #
# NBA rest days
# --------------------------------------------------------------------------- #


def load_nba_rest_days(config: MultiSportConfig | None = None) -> pd.DataFrame:
    """Days of rest for both teams of every NBA game in the SBR sqlite source.

    ``Days_Rest_* == 1`` is a back-to-back; the first game of a season carries
    a placeholder (7 or more). Agreement with rest computed from the cleaned
    schedule is 99% (the remainder are games dropped by the loader's
    consistency filter).

    Args:
        config: File locations (``nba_odds_sbr.sqlite`` under ``data_dir``).

    Returns:
        ``gameday`` (UTC), ``home_team, away_team, rest_home, rest_away``; one
        row per game.
    """
    config = config or MultiSportConfig()
    con = sqlite3.connect(config.data_dir / "nba_odds_sbr.sqlite")
    tables = pd.read_sql("select name from sqlite_master where type='table'", con)["name"].tolist()
    frames = []
    for t in tables:
        df = pd.read_sql(f'select * from "{t}"', con)
        if "Days_Rest_Home" not in df or not df["Date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all():
            continue
        frames.append(df[["Date", "Home", "Away", "Days_Rest_Home", "Days_Rest_Away"]])
    con.close()
    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "Home", "Away"])
    return rest_table(raw)


def rest_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a ``Date, Home, Away, Days_Rest_Home, Days_Rest_Away`` table.

    Args:
        raw: Rows in the sqlite source's column names.

    Returns:
        ``gameday`` (UTC), ``home_team, away_team, rest_home, rest_away``.
    """
    return pd.DataFrame(
        {
            "gameday": pd.to_datetime(raw["Date"], utc=True),
            "home_team": raw["Home"].astype(str),
            "away_team": raw["Away"].astype(str),
            "rest_home": pd.to_numeric(raw["Days_Rest_Home"], errors="coerce"),
            "rest_away": pd.to_numeric(raw["Days_Rest_Away"], errors="coerce"),
        }
    )


def attach_rest(pairs: pd.DataFrame, games: pd.DataFrame, rest: pd.DataFrame) -> pd.DataFrame:
    """Add each leg team's rest days (and the anchor teams') to shared-game pairs.

    Args:
        pairs: Output of ``build_shared_game_pairs`` (``team_h, team_a,
            game_id, h_next_game_id, a_next_game_id``).
        games: Cleaned games (for the dates of the leg games).
        rest: Output of :func:`load_nba_rest_days` / :func:`rest_table`.

    Returns:
        ``pairs`` with ``h_rest, a_rest`` (rest of ``team_h`` before leg 1 and
        of ``team_a`` before leg 2) and ``anchor_rest_h, anchor_rest_a``.
    """
    long = pd.concat(
        [
            rest[["gameday", "home_team", "rest_home"]].rename(columns={"home_team": "team", "rest_home": "rest"}),
            rest[["gameday", "away_team", "rest_away"]].rename(columns={"away_team": "team", "rest_away": "rest"}),
        ]
    ).drop_duplicates(subset=["gameday", "team"])
    key = long.set_index(["gameday", "team"])["rest"]
    day = games.set_index("game_id")["gameday"]
    out = pairs.copy()
    for col, team_col, gid_col in (
        ("h_rest", "team_h", "h_next_game_id"),
        ("a_rest", "team_a", "a_next_game_id"),
        ("anchor_rest_h", "team_h", "game_id"),
        ("anchor_rest_a", "team_a", "game_id"),
    ):
        idx = pd.MultiIndex.from_arrays([day.reindex(out[gid_col]).to_numpy(), out[team_col].to_numpy()])
        out[col] = key.reindex(idx).to_numpy()
    return out


def rest_strata(pairs: pd.DataFrame) -> dict[str, pd.Series]:
    """Pre-registered back-to-back subsets of shared-game pairs.

    Args:
        pairs: Output of :func:`attach_rest`.

    Returns:
        Label -> boolean mask aligned with ``pairs``.
    """
    h_b2b = pairs["h_rest"] == 1
    a_b2b = pairs["a_rest"] == 1
    known = pairs["h_rest"].notna() & pairs["a_rest"].notna()
    return {
        "both legs back-to-back": known & h_b2b & a_b2b,
        "exactly one leg back-to-back": known & (h_b2b ^ a_b2b),
        "neither leg back-to-back": known & ~h_b2b & ~a_b2b,
        "anchor: either team on back-to-back": (pairs["anchor_rest_h"] == 1) | (pairs["anchor_rest_a"] == 1),
    }
