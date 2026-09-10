"""Granular (offense/defense) Kalman filter on market errors with joint spread+total observations.

The team-level filter in :mod:`geo_model.parlay.kalman` keeps one state per team
(net strength error) and only sees spread residuals. This module keeps two
states per team, an offensive error ``o`` and a defensive-weakness error ``d``
(both in points, relative to the closing line), so the state has ``2 * n_teams``
dimensions (64 for 32 teams). The market implies expected points for each side
of a game::

    implied_home = (total_line + spread_line) / 2
    implied_away = (total_line - spread_line) / 2

and each game contributes two observations of the state::

    home_score - implied_home = o_H - d_A + e_H
    away_score - implied_away = o_A - d_H + e_A

Because the spread residual is the difference and the total residual is the sum
of these two observations, the filter's prior covariance at the start of a slate
yields a predicted covariance for every leg pair on the slate: spread-spread,
total-total and spread-total across different games. All predictions for a slate
are computed before any game of that slate is observed.

Optionally, an auxiliary observation of the same linear combination (for example
offensive EPA converted to point units) can be attached to every team-game to
sharpen the posterior; see :func:`attach_aux_observations`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from geo_model.parlay.backtest import correlation_summary, two_sided_parlay_roi
from geo_model.parlay.config import ParlayConfig

PAIR_TYPES: tuple[str, ...] = ("spread-spread", "total-total", "spread-total")
LEG_NAMES: tuple[str, str] = ("spread", "total")

HOME_PTS_RESID = "home_pts_resid"
AWAY_PTS_RESID = "away_pts_resid"
HOME_AUX = "home_aux_resid"
AWAY_AUX = "away_aux_resid"


def _logit(x: float) -> float:
    return float(np.log(x / (1 - x)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


@dataclass(frozen=True)
class GranularParams:
    """Hyperparameters of the offense/defense market-error random walk.

    All point-valued quantities are in NFL points. Correlations are constrained
    to ``(-1, 1)`` and ``persistence`` to ``(0, 1)`` by the vector mapping used
    by the optimizer.

    Attributes:
        prior_std_o: Std of a team's offensive market error at season start.
        prior_std_d: Std of a team's defensive-weakness market error at season start.
        process_std_o: Std of week-to-week innovation in the offensive error.
        process_std_d: Std of week-to-week innovation in the defensive error.
        obs_std: Std of single-game noise around a team's expected score.
        persistence: AR(1) coefficient of both errors between slates.
        od_corr: Prior (and innovation) correlation between a team's ``o`` and ``d``.
        obs_corr: Correlation between the home and away score noises in one game.
        use_aux: Whether to expect auxiliary observations (``home_aux_resid``,
            ``away_aux_resid``) in the games table.
        aux_loading: Scale of the auxiliary observation on ``o_T - d_opp``.
        aux_std: Std of the auxiliary observation noise.
        aux_corr: Correlation between a team-game's score noise and its
            auxiliary noise.
    """

    prior_std_o: float = 1.5
    prior_std_d: float = 1.5
    process_std_o: float = 0.3
    process_std_d: float = 0.3
    obs_std: float = 9.5
    persistence: float = 0.9
    od_corr: float = 0.0
    obs_corr: float = 0.0
    use_aux: bool = False
    aux_loading: float = 1.0
    aux_std: float = 9.5
    aux_corr: float = 0.5

    def to_vector(self) -> np.ndarray:
        """Unconstrained parameterization for the optimizer.

        Returns:
            Vector of length 8, or 11 when ``use_aux`` is set.
        """
        v = [
            np.log(self.prior_std_o),
            np.log(self.prior_std_d),
            np.log(self.process_std_o),
            np.log(self.process_std_d),
            np.log(self.obs_std),
            _logit(self.persistence),
            np.arctanh(self.od_corr),
            np.arctanh(self.obs_corr),
        ]
        if self.use_aux:
            v += [np.log(self.aux_loading), np.log(self.aux_std), np.arctanh(self.aux_corr)]
        return np.array(v, dtype=float)

    @classmethod
    def from_vector(cls, v: np.ndarray, use_aux: bool = False) -> "GranularParams":
        """Inverse of :meth:`to_vector`.

        Args:
            v: Unconstrained vector (length 8, or 11 with ``use_aux``).
            use_aux: Whether the vector carries auxiliary-observation parameters.

        Returns:
            Parameter set.
        """
        kw = dict(
            prior_std_o=float(np.exp(v[0])),
            prior_std_d=float(np.exp(v[1])),
            process_std_o=float(np.exp(v[2])),
            process_std_d=float(np.exp(v[3])),
            obs_std=float(np.exp(v[4])),
            persistence=_sigmoid(float(v[5])),
            od_corr=float(np.tanh(v[6])),
            obs_corr=float(np.tanh(v[7])),
            use_aux=use_aux,
        )
        if use_aux:
            kw.update(
                aux_loading=float(np.exp(v[8])),
                aux_std=float(np.exp(v[9])),
                aux_corr=float(np.tanh(v[10])),
            )
        return cls(**kw)

    def team_prior(self) -> np.ndarray:
        """Prior covariance of one team's ``(o, d)`` pair, shape ``[2, 2]``."""
        c = self.od_corr * self.prior_std_o * self.prior_std_d
        return np.array([[self.prior_std_o**2, c], [c, self.prior_std_d**2]])

    def team_process(self) -> np.ndarray:
        """Innovation covariance of one team's ``(o, d)`` pair, shape ``[2, 2]``."""
        c = self.od_corr * self.process_std_o * self.process_std_d
        return np.array([[self.process_std_o**2, c], [c, self.process_std_d**2]])

    def leg_noise_var(self, leg: str) -> float:
        """Game-noise variance of a spread or total residual.

        Args:
            leg: ``"spread"`` or ``"total"``.

        Returns:
            ``2 σ² (1 ∓ obs_corr)``.
        """
        sgn = -1.0 if leg == "spread" else 1.0
        return 2.0 * self.obs_std**2 * (1.0 + sgn * self.obs_corr)


@dataclass
class GranularOutput:
    """Predictions produced by one filter pass.

    Attributes:
        games: One row per game with predicted means/variances of the home and
            away point residuals and of the spread/total legs, the realized
            residuals, and ``ll_score`` (bivariate predictive log density of the
            two score residuals; comparable across models with and without
            auxiliary observations).
        pairs: One row per (game pair, leg combination) on the same slate:
            ``season, week, game_id_1, game_id_2, leg_1, leg_2, pair_type,
            pred_cov, pred_corr, resid_1, resid_2``.
        log_likelihood: Sum over slates of the joint Gaussian predictive log
            density of all observations (scores, plus auxiliary when used).
        log_likelihood_score: Same, restricted to the per-game bivariate score
            density (the sum of ``games.ll_score``).
        mean_leg_state_var: Mean over games of ``Lᵀ P L`` for the spread and
            total legs (posterior sharpness diagnostic), keyed by leg name.
    """

    games: pd.DataFrame
    pairs: pd.DataFrame
    log_likelihood: float
    log_likelihood_score: float
    mean_leg_state_var: dict[str, float]


def add_point_residuals(games: pd.DataFrame) -> pd.DataFrame:
    """Attach home/away score residuals relative to the market-implied points.

    Args:
        games: Cleaned games with ``home_score, away_score, spread_line, total_line``.

    Returns:
        Copy with ``home_pts_resid`` and ``away_pts_resid`` columns. Their
        difference equals ``resid`` and their sum equals ``tresid``.
    """
    out = games.copy()
    implied_home = (out["total_line"] + out["spread_line"]) / 2.0
    implied_away = (out["total_line"] - out["spread_line"]) / 2.0
    out[HOME_PTS_RESID] = out["home_score"] - implied_home
    out[AWAY_PTS_RESID] = out["away_score"] - implied_away
    return out


def _season_layout(sg: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    teams = sorted(set(sg["home_team"]) | set(sg["away_team"]))
    return teams, {t: i for i, t in enumerate(teams)}


def _block_diag_teams(block: np.ndarray, n: int) -> np.ndarray:
    """Expand a per-team ``[2, 2]`` block into ``[2n, 2n]`` with o first, d second."""
    P = np.zeros((2 * n, 2 * n))
    P[:n, :n] = np.eye(n) * block[0, 0]
    P[n:, n:] = np.eye(n) * block[1, 1]
    P[:n, n:] = np.eye(n) * block[0, 1]
    P[n:, :n] = np.eye(n) * block[1, 0]
    return P


def _slate_matrices(
    wg: pd.DataFrame, idx: dict[str, int], n: int, p: GranularParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Observation matrix, noise covariance, observation vector and leg matrix.

    Observation rows per game are ``[home score, away score]`` and, when
    ``p.use_aux`` is set, ``[home aux, away aux]`` appended after all score rows.

    Returns:
        ``(H, R, y, A)`` where ``H`` is ``[n_obs, 2n]``, ``R`` is ``[n_obs, n_obs]``,
        ``y`` is ``[n_obs]`` and ``A`` is ``[2G, n_obs]`` mapping observations to
        legs (row ``2g`` = spread of game ``g``, ``2g + 1`` = total).
    """
    G = len(wg)
    n_obs = 4 * G if p.use_aux else 2 * G
    H = np.zeros((n_obs, 2 * n))
    R = np.zeros((n_obs, n_obs))
    s2 = p.obs_std**2
    for g, (ht, at) in enumerate(zip(wg["home_team"], wg["away_team"])):
        h, a = idx[ht], idx[at]
        H[2 * g, h] = 1.0
        H[2 * g, n + a] = -1.0
        H[2 * g + 1, a] = 1.0
        H[2 * g + 1, n + h] = -1.0
        R[2 * g, 2 * g] = s2
        R[2 * g + 1, 2 * g + 1] = s2
        R[2 * g, 2 * g + 1] = R[2 * g + 1, 2 * g] = p.obs_corr * s2
        if p.use_aux:
            za, zb = 2 * G + 2 * g, 2 * G + 2 * g + 1
            H[za] = p.aux_loading * H[2 * g]
            H[zb] = p.aux_loading * H[2 * g + 1]
            R[za, za] = R[zb, zb] = p.aux_std**2
            c = p.aux_corr * p.obs_std * p.aux_std
            R[za, 2 * g] = R[2 * g, za] = c
            R[zb, 2 * g + 1] = R[2 * g + 1, zb] = c
    y = np.empty(n_obs)
    y[0 : 2 * G : 2] = wg[HOME_PTS_RESID].to_numpy(dtype=float)
    y[1 : 2 * G : 2] = wg[AWAY_PTS_RESID].to_numpy(dtype=float)
    if p.use_aux:
        y[2 * G :: 2] = wg[HOME_AUX].to_numpy(dtype=float)
        y[2 * G + 1 :: 2] = wg[AWAY_AUX].to_numpy(dtype=float)
    A = np.zeros((2 * G, n_obs))
    for g in range(G):
        A[2 * g, 2 * g] = 1.0
        A[2 * g, 2 * g + 1] = -1.0
        A[2 * g + 1, 2 * g] = 1.0
        A[2 * g + 1, 2 * g + 1] = 1.0
    return H, R, y, A


def _mvn_logpdf(r: np.ndarray, S: np.ndarray) -> float:
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, r)
    return float(-0.5 * (r.size * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + z @ z))


def _bivariate_logpdf(r: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-game log density of consecutive ``(home, away)`` residual pairs.

    Args:
        r: Residual vector of length ``2G`` ordered ``[home_0, away_0, home_1, ...]``.
        S: Predictive covariance ``[2G, 2G]``; only the diagonal 2x2 blocks are used.

    Returns:
        Log densities, shape ``[G]``.
    """
    G = r.size // 2
    i = np.arange(G)
    a = S[2 * i, 2 * i]
    d = S[2 * i + 1, 2 * i + 1]
    b = S[2 * i, 2 * i + 1]
    det = a * d - b * b
    x, y = r[0::2], r[1::2]
    quad = (d * x * x - 2 * b * x * y + a * y * y) / det
    return -0.5 * (2 * np.log(2 * np.pi) + np.log(det) + quad)


def _pair_arrays(G: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Leg-index arrays for all cross-game leg combinations on a slate.

    Returns:
        ``(gi, gj, li, lj)`` with game indices and leg indices (``0`` spread,
        ``1`` total) for the four combinations of every game pair ``gi < gj``.
    """
    gi, gj = np.triu_indices(G, 1)
    combos = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    gi = np.repeat(gi, 4)
    gj = np.repeat(gj, 4)
    li = np.tile(combos[:, 0], len(gi) // 4)
    lj = np.tile(combos[:, 1], len(gj) // 4)
    return gi, gj, li, lj


MARKET_MODES: tuple[str, ...] = ("static", "efficient")


class GranularKalman:
    """Season-by-season offense/defense Kalman filter on score residuals.

    Two views of what the closing line already knows are supported:

    * ``market_mode="static"`` (the convention of :class:`MarketErrorKalman`):
      the market's error persists and the filter's posterior mean is a forecast
      of the next residual (a directional, single-bet edge if it works).
    * ``market_mode="efficient"``: the market re-prices every game at the
      Bayesian posterior mean, so residuals are innovations with mean zero and
      only the covariance recursion is informative. This is the regime in which
      cross-game "explaining away" correlations exist relative to the closing
      line, and it matches the ``market="efficient"`` simulator.

    Args:
        params: Filter hyperparameters.
        slate_col: Column defining a slate (games observed together).
        market_mode: ``"static"`` or ``"efficient"``.
    """

    def __init__(
        self,
        params: GranularParams | None = None,
        slate_col: str = "week",
        market_mode: str = "static",
    ) -> None:
        if market_mode not in MARKET_MODES:
            raise ValueError(f"unknown market mode {market_mode!r}")
        self.params = params or GranularParams()
        self.slate_col = slate_col
        self.market_mode = market_mode

    def run(self, games: pd.DataFrame, emit_pairs: bool = True) -> GranularOutput:
        """Filter every season in chronological order.

        Args:
            games: Cleaned games with ``season, week, game_id, home_team,
                away_team, home_pts_resid, away_pts_resid`` (see
                :func:`add_point_residuals`) and, when ``params.use_aux`` is
                set, ``home_aux_resid, away_aux_resid``.
            emit_pairs: Whether to build the (quadratic-size) pairs table.

        Returns:
            :class:`GranularOutput` with predictions made strictly before each slate.
        """
        p = self.params
        if HOME_PTS_RESID not in games.columns:
            games = add_point_residuals(games)
        game_frames: list[pd.DataFrame] = []
        pair_frames: list[pd.DataFrame] = []
        ll = 0.0
        ll_score = 0.0
        leg_state_var = {"spread": [], "total": []}
        for season, sg in games.groupby("season", sort=True):
            teams, idx = _season_layout(sg)
            n = len(teams)
            m = np.zeros(2 * n)
            P = _block_diag_teams(p.team_prior(), n)
            Q = _block_diag_teams(p.team_process(), n)
            first = True
            for slate, wg in sg.groupby(self.slate_col, sort=True):
                if not first:
                    m = p.persistence * m
                    P = p.persistence**2 * P + Q
                first = False
                G = len(wg)
                H, R, y, A = _slate_matrices(wg, idx, n, p)
                mu = H @ m
                HPHt = H @ P @ H.T
                S = HPHt + R
                r = y - mu
                ll += _mvn_logpdf(r, S)
                # Leg-level predictive quantities (score observations only).
                A_s = A[:, : 2 * G]
                C_state = A_s @ HPHt[: 2 * G, : 2 * G] @ A_s.T
                C_leg = A_s @ S[: 2 * G, : 2 * G] @ A_s.T
                v_leg = np.diag(C_leg)
                mu_leg = A_s @ mu[: 2 * G]
                y_leg = A_s @ y[: 2 * G]
                leg_state_var["spread"].append(np.diag(C_state)[0::2])
                leg_state_var["total"].append(np.diag(C_state)[1::2])
                per_game_ll = _bivariate_logpdf(r[: 2 * G], S[: 2 * G, : 2 * G])
                ll_score += float(per_game_ll.sum())
                game_frames.append(
                    pd.DataFrame(
                        {
                            "game_id": wg["game_id"].to_numpy(),
                            "season": season,
                            self.slate_col: slate,
                            "pred_mean_home": mu[0 : 2 * G : 2],
                            "pred_mean_away": mu[1 : 2 * G : 2],
                            "home_pts_resid": y[0 : 2 * G : 2],
                            "away_pts_resid": y[1 : 2 * G : 2],
                            "pred_mean_spread": mu_leg[0::2],
                            "pred_var_spread": v_leg[0::2],
                            "resid": y_leg[0::2],
                            "pred_mean_total": mu_leg[1::2],
                            "pred_var_total": v_leg[1::2],
                            "tresid": y_leg[1::2],
                            "ll_score": per_game_ll,
                        }
                    )
                )
                if emit_pairs and G > 1:
                    gi, gj, li, lj = _pair_arrays(G)
                    ii = 2 * gi + li
                    jj = 2 * gj + lj
                    cov = C_leg[ii, jj]
                    corr = cov / np.sqrt(v_leg[ii] * v_leg[jj])
                    gids = wg["game_id"].to_numpy()
                    leg_names = np.array(LEG_NAMES)
                    ptype = np.where(
                        li == lj, np.where(li == 0, PAIR_TYPES[0], PAIR_TYPES[1]), PAIR_TYPES[2]
                    )
                    pair_frames.append(
                        pd.DataFrame(
                            {
                                "season": season,
                                self.slate_col: slate,
                                "game_id_1": gids[gi],
                                "game_id_2": gids[gj],
                                "leg_1": leg_names[li],
                                "leg_2": leg_names[lj],
                                "pair_type": ptype,
                                "pred_cov": cov,
                                "pred_corr": corr,
                                "resid_1": y_leg[ii],
                                "resid_2": y_leg[jj],
                            }
                        )
                    )
                # Joint update with all observations on the slate. Under an
                # efficient market the line absorbs the posterior mean, so the
                # residual state keeps mean zero and only P is updated.
                K = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
                if self.market_mode == "static":
                    m = m + K @ r
                P = P - K @ H @ P
                P = 0.5 * (P + P.T)
        games_df = pd.concat(game_frames, ignore_index=True) if game_frames else pd.DataFrame()
        pairs_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
        return GranularOutput(
            games=games_df,
            pairs=pairs_df,
            log_likelihood=ll,
            log_likelihood_score=ll_score,
            mean_leg_state_var={
                k: float(np.concatenate(v).mean()) if v else float("nan") for k, v in leg_state_var.items()
            },
        )

    def fit(self, games: pd.DataFrame, max_iter: int = 600) -> GranularParams:
        """Maximize the joint predictive log-likelihood over the hyperparameters.

        Args:
            games: Training games (must precede any evaluation games).
            max_iter: Nelder-Mead iteration budget.

        Returns:
            Fitted parameters (also stored on ``self.params``).
        """
        if HOME_PTS_RESID not in games.columns:
            games = add_point_residuals(games)
        use_aux = self.params.use_aux

        def neg_ll(v: np.ndarray) -> float:
            params = GranularParams.from_vector(v, use_aux=use_aux)
            out = GranularKalman(params, self.slate_col, self.market_mode).run(games, emit_pairs=False)
            return -out.log_likelihood

        res = optimize.minimize(
            neg_ll,
            self.params.to_vector(),
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-2},
        )
        self.params = GranularParams.from_vector(res.x, use_aux=use_aux)
        return self.params


def simulate_granular_games(
    schedule: pd.DataFrame,
    params: GranularParams,
    seed: int = 0,
    market: str = "efficient",
) -> pd.DataFrame:
    """Draw synthetic score residuals from the granular generative model.

    Mirrors :func:`geo_model.parlay.kalman.simulate_games`: latent ``(o, d)``
    errors per team evolve as an AR(1) between weeks; each game produces two
    noisy score observations. With ``market="efficient"`` every game is priced
    at the Bayesian posterior mean given all earlier games (a filter with the
    true parameters), so the residuals are relative to a market that updates
    on shared games. With ``"static"`` the market never updates.

    Args:
        schedule: Games with ``season, week, game_id, home_team, away_team``.
        params: Generative parameters. Auxiliary observations are simulated when
            ``params.use_aux`` is set.
        seed: RNG seed.
        market: ``"efficient"`` or ``"static"``.

    Returns:
        Copy of ``schedule`` with ``home_pts_resid, away_pts_resid, resid,
        tresid`` (and ``home_aux_resid, away_aux_resid`` when used).
    """
    if market not in {"efficient", "static"}:
        raise ValueError(f"unknown market mode {market!r}")
    rng = np.random.default_rng(seed)
    out = schedule.copy()
    for col in (HOME_PTS_RESID, AWAY_PTS_RESID, "resid", "tresid"):
        out[col] = np.nan
    if params.use_aux:
        out[HOME_AUX] = np.nan
        out[AWAY_AUX] = np.nan
    for _, sg in out.groupby("season", sort=True):
        teams, idx = _season_layout(sg)
        n = len(teams)
        P0 = _block_diag_teams(params.team_prior(), n)
        Q = _block_diag_teams(params.team_process(), n)
        x = rng.multivariate_normal(np.zeros(2 * n), P0)
        m = np.zeros(2 * n)
        P = P0.copy()
        first = True
        for _, wg in sg.groupby("week", sort=True):
            if not first:
                x = params.persistence * x + rng.multivariate_normal(np.zeros(2 * n), Q)
                m = params.persistence * m
                P = params.persistence**2 * P + Q
            first = False
            G = len(wg)
            H, R, _, A = _slate_matrices(_with_zero_obs(wg, params), idx, n, params)
            noise = rng.multivariate_normal(np.zeros(R.shape[0]), R)
            y = H @ x + noise
            if market == "efficient":
                mu = H @ m
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
                m = m + K @ (y - mu)
                P = P - K @ H @ P
                P = 0.5 * (P + P.T)
                res = y - mu
            else:
                res = y
            out.loc[wg.index, HOME_PTS_RESID] = res[0 : 2 * G : 2]
            out.loc[wg.index, AWAY_PTS_RESID] = res[1 : 2 * G : 2]
            if params.use_aux:
                out.loc[wg.index, HOME_AUX] = res[2 * G :: 2]
                out.loc[wg.index, AWAY_AUX] = res[2 * G + 1 :: 2]
            legs = A[:, : 2 * G] @ res[: 2 * G]
            out.loc[wg.index, "resid"] = legs[0::2]
            out.loc[wg.index, "tresid"] = legs[1::2]
    return out


def _with_zero_obs(wg: pd.DataFrame, params: GranularParams) -> pd.DataFrame:
    """Add zero-filled observation columns so :func:`_slate_matrices` can be reused."""
    tmp = wg.copy()
    for col in (HOME_PTS_RESID, AWAY_PTS_RESID):
        tmp[col] = 0.0
    if params.use_aux:
        tmp[HOME_AUX] = 0.0
        tmp[AWAY_AUX] = 0.0
    return tmp


# ---------------------------------------------------------------------------
# Theory helpers
# ---------------------------------------------------------------------------


def stationary_team_cov(params: GranularParams) -> np.ndarray:
    """Largest per-team ``(o, d)`` state covariance the filter can hold.

    The prior covariance is the season-start uncertainty; between slates the
    covariance evolves as ``φ² P + Q`` whose fixed point is ``Q / (1 - φ²)``.
    Because the posterior never exceeds the predicted covariance, the
    element-wise maximum of the two diagonals bounds the state uncertainty at
    any point in the season.

    Args:
        params: Filter parameters.

    Returns:
        ``[2, 2]`` covariance with the larger of prior and stationary variances
        on the diagonal (off-diagonal from ``od_corr``).
    """
    prior = params.team_prior()
    if params.persistence < 1.0:
        stat = params.team_process() / (1.0 - params.persistence**2)
    else:
        stat = np.full((2, 2), np.inf)
    var_o = max(prior[0, 0], stat[0, 0])
    var_d = max(prior[1, 1], stat[1, 1])
    c = params.od_corr * np.sqrt(var_o * var_d)
    return np.array([[var_o, c], [c, var_d]])


def theoretical_max_corr(params: GranularParams) -> dict[str, float]:
    """Cauchy–Schwarz cap on the predicted correlation between two legs.

    For legs in different games the state parts of the two residuals satisfy
    ``|L1ᵀ P L2| ≤ sqrt(L1ᵀ P L1 · L2ᵀ P L2)``, so the correlation is bounded
    by ``sqrt(v1 / (v1 + R1)) · sqrt(v2 / (v2 + R2))`` where ``v`` is the leg's
    state variance and ``R`` its game-noise variance. Using the largest state
    variance the filter can hold (:func:`stationary_team_cov`) gives a bound
    that no schedule, however informative, can exceed. It is attained only if
    the two legs' latent errors were perfectly correlated, which conditioning
    on shared games never achieves, so it is a loose cap.

    Args:
        params: Filter parameters.

    Returns:
        Bound per pair type (``spread-spread``, ``total-total``, ``spread-total``).
    """
    T = stationary_team_cov(params)
    # Spread leg on states (o_H, d_H, o_A, d_A): (1, +1, -1, -1); total: (1, -1, 1, -1).
    # Two independent teams, so v = Σ_team wᵀ T w.
    w_spread_home, w_spread_away = np.array([1.0, 1.0]), np.array([-1.0, -1.0])
    w_total_home, w_total_away = np.array([1.0, -1.0]), np.array([1.0, -1.0])
    v_spread = w_spread_home @ T @ w_spread_home + w_spread_away @ T @ w_spread_away
    v_total = w_total_home @ T @ w_total_home + w_total_away @ T @ w_total_away
    f_spread = np.sqrt(v_spread / (v_spread + params.leg_noise_var("spread")))
    f_total = np.sqrt(v_total / (v_total + params.leg_noise_var("total")))
    return {
        "spread-spread": float(f_spread * f_spread),
        "total-total": float(f_total * f_total),
        "spread-total": float(f_spread * f_total),
    }


def single_shared_game_corr(params: GranularParams) -> pd.DataFrame:
    """Predicted leg correlations after one shared game in a four-team toy.

    Week 1: H hosts A. Week 2: H hosts C and A hosts D. The filter's week-2
    predicted covariance between the two next games is the granular analogue of
    :func:`geo_model.parlay.backtest.gaussian_explaining_away_corr` and is an
    upper bound for week-1 anchors (later weeks are smaller because the market
    has learned).

    Args:
        params: Filter parameters (auxiliary observations are ignored).

    Returns:
        Rows ``leg_1, leg_2, pair_type, pred_corr`` for the (H vs C, A vs D) pair.
    """
    p = replace(params, use_aux=False)
    sched = pd.DataFrame(
        {
            "game_id": ["w1", "w2_h", "w2_a"],
            "season": [2000, 2000, 2000],
            "week": [1, 2, 2],
            "home_team": ["H", "H", "A"],
            "away_team": ["A", "C", "D"],
            HOME_PTS_RESID: [0.0, 0.0, 0.0],
            AWAY_PTS_RESID: [0.0, 0.0, 0.0],
        }
    )
    out = GranularKalman(p).run(sched)
    pr = out.pairs[out.pairs["week"] == 2]
    return pr[["leg_1", "leg_2", "pair_type", "pred_corr"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def pair_type_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    """Distribution of predicted |corr| by pair type.

    Args:
        pairs: ``GranularOutput.pairs``.

    Returns:
        One row per pair type with ``n, mean_abs, p50_abs, p90_abs, p99_abs,
        max_abs, frac_positive``.
    """
    rows = []
    for pt in PAIR_TYPES:
        g = pairs[pairs["pair_type"] == pt]
        a = g["pred_corr"].abs()
        rows.append(
            {
                "pair_type": pt,
                "n": int(len(g)),
                "mean_abs": float(a.mean()) if len(g) else float("nan"),
                "p50_abs": float(a.quantile(0.5)) if len(g) else float("nan"),
                "p90_abs": float(a.quantile(0.9)) if len(g) else float("nan"),
                "p99_abs": float(a.quantile(0.99)) if len(g) else float("nan"),
                "max_abs": float(a.max()) if len(g) else float("nan"),
                "frac_positive": float((g["pred_corr"] > 0).mean()) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def calibration_slope(pairs: pd.DataFrame) -> tuple[float, float, float, float]:
    """OLS of realized residual product on predicted covariance.

    Args:
        pairs: Pair rows with ``pred_cov, resid_1, resid_2``.

    Returns:
        ``(slope, stderr, p_value, r_squared)``; slope 1 means calibrated.
    """
    y = (pairs["resid_1"] * pairs["resid_2"]).to_numpy(dtype=float)
    x = pairs["pred_cov"].to_numpy(dtype=float)
    # A predicted covariance that is numerically zero everywhere (e.g. the
    # spread-total block of a symmetric model) carries no ranking information.
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return float("nan"), float("nan"), float("nan"), float("nan")
    res = stats.linregress(x, y)
    return float(res.slope), float(res.stderr), float(res.pvalue), float(res.rvalue**2)


def calibration_by_pair_type(pairs: pd.DataFrame) -> pd.DataFrame:
    """Calibration slope per pair type.

    Args:
        pairs: ``GranularOutput.pairs``.

    Returns:
        Rows ``pair_type, n, slope, se, p, r2, z_from_one`` where ``z_from_one``
        is ``(slope - 1) / se`` (how far the fit is from a calibrated model).
    """
    rows = []
    for pt in PAIR_TYPES:
        g = pairs[pairs["pair_type"] == pt]
        slope, se, pv, r2 = calibration_slope(g)
        rows.append(
            {
                "pair_type": pt,
                "n": int(len(g)),
                "slope": slope,
                "se": se,
                "p": pv,
                "r2": r2,
                "z_from_one": (slope - 1.0) / se if se and np.isfinite(se) and se > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def signed_pair_residuals(pairs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flip leg 2 by the sign of the predicted correlation.

    After flipping, a positive realized correlation means the model's sign was
    right, so same-sign parlays are the correct mechanical bet for every pair.

    Args:
        pairs: Pair rows with ``pred_corr, resid_1, resid_2``.

    Returns:
        ``(resid_1, resid_2 * sign(pred_corr))``.
    """
    sgn = np.sign(pairs["pred_corr"]).replace(0, 1)
    return pairs["resid_1"], pairs["resid_2"] * sgn


def top_fraction_by_pair_type(
    pairs: pd.DataFrame, leg_decimal: float, top_frac: float = 0.1, n_boot: int = 500
) -> pd.DataFrame:
    """Realized correlation and parlay ROI for the top ``top_frac`` of predicted |corr|.

    Selection thresholds are quantiles of the out-of-sample predictions, which
    is legitimate for a *ranking* diagnostic but note that the threshold itself
    uses the full evaluation window; the per-pair predictions are strictly causal.

    Args:
        pairs: ``GranularOutput.pairs``.
        leg_decimal: Decimal odds per leg.
        top_frac: Fraction of pairs (by |pred_corr|) to bet.
        n_boot: Bootstrap resamples for the phi CI.

    Returns:
        Rows ``pair_type, n, min_abs_pred, mean_pred_signed, pearson, pearson_ci,
        phi, phi_ci, same_sign_rate, roi``. ``mean_pred_signed`` is the mean
        predicted correlation after sign alignment (the model's own forecast of
        what ``pearson`` should be).
    """
    rows = []
    for pt in PAIR_TYPES:
        g = pairs[pairs["pair_type"] == pt]
        if g.empty:
            continue
        thr = g["pred_corr"].abs().quantile(1.0 - top_frac)
        sel = g[g["pred_corr"].abs() >= thr]
        x, y = signed_pair_residuals(sel)
        s = correlation_summary(x, y, n_boot=n_boot)
        r = two_sided_parlay_roi(x, y, leg_decimal, expected_sign=1)
        rows.append(
            {
                "pair_type": pt,
                "n": int(len(sel)),
                "min_abs_pred": float(thr),
                "mean_pred_signed": float(sel["pred_corr"].abs().mean()),
                "pearson": s.pearson,
                "pearson_ci": f"[{s.pearson_ci[0]:+.3f}, {s.pearson_ci[1]:+.3f}]",
                "phi": s.phi,
                "phi_ci": f"[{s.phi_ci[0]:+.3f}, {s.phi_ci[1]:+.3f}]",
                "same_sign_rate": r.hit_rate,
                "roi": r.roi,
            }
        )
    return pd.DataFrame(rows)


def decile_calibration(pairs: pd.DataFrame, n_quantiles: int = 10, n_boot: int = 200) -> pd.DataFrame:
    """Realized vs predicted correlation across predicted-correlation deciles.

    Args:
        pairs: Pair rows for one pair type.
        n_quantiles: Number of bins over ``pred_corr``.
        n_boot: Bootstrap resamples for the phi CI.

    Returns:
        Rows ``bin, n, pred_corr_mean, realized_pearson, realized_phi, phi_ci_lo, phi_ci_hi``.
    """
    df = pairs.copy()
    df["bin"] = pd.qcut(df["pred_corr"].rank(method="first"), q=n_quantiles, labels=False)
    rows = []
    for b, g in df.groupby("bin", sort=True):
        s = correlation_summary(g["resid_1"], g["resid_2"], n_boot=n_boot)
        rows.append(
            {
                "bin": int(b),
                "n": int(len(g)),
                "pred_corr_mean": float(g["pred_corr"].mean()),
                "realized_pearson": s.pearson,
                "realized_phi": s.phi,
                "phi_ci_lo": s.phi_ci[0],
                "phi_ci_hi": s.phi_ci[1],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auxiliary observations (nflverse team-week EPA)
# ---------------------------------------------------------------------------

STATS_TEAM_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.parquet"
)

# The games file keeps historical abbreviations; the stats_team release uses the
# franchise's current one for every season.
NFLVERSE_TEAM_ALIASES: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def stats_team_week_path(config: ParlayConfig, season: int) -> Path:
    """Local cache path of one season's nflverse team-week stats."""
    return config.data_dir / "nfl" / f"stats_team_week_{season}.parquet"


def load_team_week_epa(config: ParlayConfig, seasons: tuple[int, int], download: bool = True) -> pd.DataFrame:
    """Offensive EPA per team-game from the nflverse ``stats_team`` release.

    Args:
        config: Data directory.
        seasons: Inclusive season range.
        download: Fetch missing seasons from GitHub releases.

    Returns:
        Rows ``season, week, game_id, team, off_epa`` (regular season only).
        ``off_epa = passing_epa + rushing_epa`` for the team's offense.
    """
    import requests

    frames = []
    for season in range(seasons[0], seasons[1] + 1):
        path = stats_team_week_path(config, season)
        if not path.exists():
            if not download:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(STATS_TEAM_WEEK_URL.format(season=season), timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        df = pd.read_parquet(path, columns=["season", "week", "game_id", "team", "season_type", "passing_epa", "rushing_epa"])
        df = df[df["season_type"] == "REG"]
        frames.append(
            pd.DataFrame(
                {
                    "season": df["season"].astype(int),
                    "week": df["week"].astype(int),
                    "game_id": df["game_id"].astype(str),
                    "team": df["team"].astype(str),
                    "off_epa": df["passing_epa"].astype(float) + df["rushing_epa"].astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["season", "week", "game_id", "team", "off_epa"])


@dataclass(frozen=True)
class AuxCalibration:
    """Linear map from an auxiliary team-game statistic to point units.

    Attributes:
        intercept: ``alpha`` in ``aux ≈ alpha + beta * implied_points``.
        slope: ``beta`` (auxiliary units per implied point).
        n: Team-games used in the fit.
    """

    intercept: float
    slope: float
    n: int


def fit_aux_calibration(team_aux: pd.Series, implied_points: pd.Series) -> AuxCalibration:
    """Regress an auxiliary statistic on market-implied points.

    Args:
        team_aux: Auxiliary statistic per team-game (e.g. offensive EPA).
        implied_points: Market-implied points for the same team-game.

    Returns:
        :class:`AuxCalibration`.
    """
    x = np.asarray(implied_points, dtype=float)
    y = np.asarray(team_aux, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    res = stats.linregress(x[m], y[m])
    return AuxCalibration(float(res.intercept), float(res.slope), int(m.sum()))


def attach_aux_observations(
    games: pd.DataFrame, team_aux: pd.DataFrame, calib: AuxCalibration, value_col: str = "off_epa"
) -> pd.DataFrame:
    """Attach point-scaled auxiliary residuals to every game.

    ``home_aux_resid = (aux_H - alpha) / beta - implied_home`` and likewise for
    the away team, so the auxiliary residual is on the same scale as the score
    residual and observes ``o_H - d_A`` (up to the filter's ``aux_loading``).
    Games with a missing auxiliary value on either side are dropped.

    Args:
        games: Cleaned games (with or without point residuals).
        team_aux: Rows ``game_id, team, <value_col>``.
        calib: Fitted linear map (must be fitted on training seasons only).
        value_col: Column of ``team_aux`` holding the statistic.

    Returns:
        Games with ``home_pts_resid, away_pts_resid, home_aux_resid, away_aux_resid``.
    """
    g = add_point_residuals(games)
    lookup = team_aux.set_index(["game_id", "team"])[value_col]
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    def _pick(side: str) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([g["game_id"], g[side]])
        vals = lookup.reindex(key).to_numpy(dtype=float)
        alias = g[side].map(NFLVERSE_TEAM_ALIASES).fillna(g[side])
        alias_key = pd.MultiIndex.from_arrays([g["game_id"], alias])
        alias_vals = lookup.reindex(alias_key).to_numpy(dtype=float)
        return np.where(np.isfinite(vals), vals, alias_vals)

    aux_h = _pick("home_team")
    aux_a = _pick("away_team")
    implied_home = (g["total_line"] + g["spread_line"]) / 2.0
    implied_away = (g["total_line"] - g["spread_line"]) / 2.0
    g[HOME_AUX] = (aux_h - calib.intercept) / calib.slope - implied_home
    g[AWAY_AUX] = (aux_a - calib.intercept) / calib.slope - implied_away
    return g[g[HOME_AUX].notna() & g[AWAY_AUX].notna()].reset_index(drop=True)
