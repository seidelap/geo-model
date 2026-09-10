"""Causal Kalman filter over the market's team-strength errors.

State ``eps[t]`` (one entry per team) is the true strength minus the strength
implied by the closing line. Each game observes ``eps[home] - eps[away]`` plus
game noise, so the posterior covariance of ``eps`` carries exactly the
"explaining away" structure the parlay hypothesis relies on, generalized from
one shared game to the whole shared-opponent network of the season.

For two games ``g1, g2`` on the same slate, the model's predicted covariance of
their spread residuals is ``h1ᵀ P h2`` where ``h = e_home - e_away`` and ``P``
is the prior covariance at the start of the slate. All quantities for a slate
are computed before any game of that slate is observed (no leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize


@dataclass(frozen=True)
class KalmanParams:
    """Hyperparameters of the market-error random walk.

    Attributes:
        prior_std: Std of a team's market error at season start (points).
        process_std: Std of week-to-week innovation in the error (points).
        obs_std: Std of single-game noise around the true margin (points).
        persistence: AR(1) coefficient of the error between slates. Below 1
            models the market correcting itself over time.
    """

    prior_std: float = 2.0
    process_std: float = 0.5
    obs_std: float = 13.0
    persistence: float = 0.9

    def to_vector(self) -> np.ndarray:
        """Unconstrained parameterization for the optimizer."""
        return np.array(
            [
                np.log(self.prior_std),
                np.log(self.process_std),
                np.log(self.obs_std),
                np.log(self.persistence / (1 - self.persistence)),
            ]
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "KalmanParams":
        """Inverse of :meth:`to_vector`."""
        return cls(
            prior_std=float(np.exp(v[0])),
            process_std=float(np.exp(v[1])),
            obs_std=float(np.exp(v[2])),
            persistence=float(1 / (1 + np.exp(-v[3]))),
        )


@dataclass
class KalmanOutput:
    """Predictions produced by one filter pass.

    Attributes:
        games: Per-game rows ``game_id, season, week, pred_mean, pred_var, resid``
            where ``pred_mean`` is the predicted spread residual (home
            perspective) and ``pred_var`` its predictive variance.
        pairs: Per-pair rows for all game pairs on the same slate:
            ``season, week, game_id_1, game_id_2, pred_cov, pred_corr,
            resid_1, resid_2``.
        log_likelihood: Sum of Gaussian predictive log densities of residuals.
    """

    games: pd.DataFrame
    pairs: pd.DataFrame
    log_likelihood: float


class MarketErrorKalman:
    """Season-by-season Kalman filter on spread residuals.

    Args:
        params: Filter hyperparameters.
        slate_col: Column defining a slate (games observed together). ``week``
            by default.
    """

    def __init__(self, params: KalmanParams | None = None, slate_col: str = "week") -> None:
        self.params = params or KalmanParams()
        self.slate_col = slate_col

    def run(self, games: pd.DataFrame, emit_pairs: bool = True) -> KalmanOutput:
        """Filter every season in chronological order.

        Args:
            games: Cleaned games with ``season, week, home_team, away_team, resid``.
            emit_pairs: Whether to build the (quadratic-size) pairs table.

        Returns:
            :class:`KalmanOutput` with predictions made strictly before each slate.
        """
        p = self.params
        game_rows: list[dict] = []
        pair_rows: list[dict] = []
        ll = 0.0
        for season, sg in games.groupby("season", sort=True):
            teams = sorted(set(sg["home_team"]) | set(sg["away_team"]))
            idx = {t: i for i, t in enumerate(teams)}
            n = len(teams)
            m = np.zeros(n)
            P = np.eye(n) * p.prior_std**2
            first = True
            for slate, wg in sg.groupby(self.slate_col, sort=True):
                if not first:
                    m = p.persistence * m
                    P = p.persistence**2 * P + np.eye(n) * p.process_std**2
                first = False
                H = np.zeros((len(wg), n))
                for r, (ht, at) in enumerate(zip(wg["home_team"], wg["away_team"])):
                    H[r, idx[ht]] = 1.0
                    H[r, idx[at]] = -1.0
                y = wg["resid"].to_numpy(dtype=float)
                mu = H @ m
                S = H @ P @ H.T + np.eye(len(wg)) * p.obs_std**2
                var = np.diag(S)
                ll += float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var))
                gids = wg["game_id"].to_numpy()
                for r in range(len(wg)):
                    game_rows.append(
                        {
                            "game_id": gids[r],
                            "season": season,
                            self.slate_col: slate,
                            "pred_mean": mu[r],
                            "pred_var": var[r],
                            "resid": y[r],
                        }
                    )
                if emit_pairs:
                    C = H @ P @ H.T
                    for i in range(len(wg)):
                        for j in range(i + 1, len(wg)):
                            pair_rows.append(
                                {
                                    "season": season,
                                    self.slate_col: slate,
                                    "game_id_1": gids[i],
                                    "game_id_2": gids[j],
                                    "pred_cov": C[i, j],
                                    "pred_corr": C[i, j] / np.sqrt(var[i] * var[j]),
                                    "resid_1": y[i],
                                    "resid_2": y[j],
                                }
                            )
                # Joint update with all games on the slate.
                K = P @ H.T @ np.linalg.solve(S, np.eye(len(wg)))
                m = m + K @ (y - mu)
                P = P - K @ H @ P
                P = 0.5 * (P + P.T)
        return KalmanOutput(
            games=pd.DataFrame(game_rows),
            pairs=pd.DataFrame(pair_rows),
            log_likelihood=ll,
        )

    def fit(self, games: pd.DataFrame, max_iter: int = 200) -> KalmanParams:
        """Maximize the predictive log-likelihood over the hyperparameters.

        Args:
            games: Training games (must precede any evaluation games to avoid
                leaking hyperparameter information).
            max_iter: Nelder-Mead iteration budget.

        Returns:
            Fitted parameters (also stored on ``self.params``).
        """

        def neg_ll(v: np.ndarray) -> float:
            params = KalmanParams.from_vector(v)
            out = MarketErrorKalman(params, self.slate_col).run(games, emit_pairs=False)
            return -out.log_likelihood

        res = optimize.minimize(
            neg_ll, self.params.to_vector(), method="Nelder-Mead", options={"maxiter": max_iter}
        )
        self.params = KalmanParams.from_vector(res.x)
        return self.params


def simulate_games(
    schedule: pd.DataFrame,
    params: KalmanParams,
    seed: int = 0,
    market: str = "efficient",
) -> pd.DataFrame:
    """Draw synthetic spread residuals from the filter's generative model.

    Useful as a power check: on data generated with a large ``prior_std`` the
    backtest must recover a positive shared-game correlation.

    Args:
        schedule: Games with ``season, week, game_id, home_team, away_team``.
        params: Generative parameters for the latent team errors.
        seed: RNG seed.
        market: ``"efficient"`` prices each game at the Bayesian posterior mean
            given all earlier games (a Kalman filter with the true parameters),
            so residuals are relative to a market that updates on shared games.
            ``"static"`` never updates, so the latent errors show up as a
            directional edge instead of a cross-game correlation.

    Returns:
        Copy of ``schedule`` with a synthetic ``resid`` column (home perspective)
        and ``true_dev`` (margin deviation from the static prior line).
    """
    if market not in {"efficient", "static"}:
        raise ValueError(f"unknown market mode {market!r}")
    rng = np.random.default_rng(seed)
    out = schedule.copy()
    out["true_dev"] = np.nan
    out["resid"] = np.nan
    for _, sg in out.groupby("season", sort=True):
        teams = sorted(set(sg["home_team"]) | set(sg["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        eps = rng.normal(0, params.prior_std, n)
        m = np.zeros(n)
        P = np.eye(n) * params.prior_std**2
        first = True
        for _, wg in sg.groupby("week", sort=True):
            if not first:
                eps = params.persistence * eps + rng.normal(0, params.process_std, n)
                m = params.persistence * m
                P = params.persistence**2 * P + np.eye(n) * params.process_std**2
            first = False
            H = np.zeros((len(wg), n))
            for r, (ht, at) in enumerate(zip(wg["home_team"], wg["away_team"])):
                H[r, idx[ht]] = 1.0
                H[r, idx[at]] = -1.0
            y = H @ eps + rng.normal(0, params.obs_std, len(wg))
            out.loc[wg.index, "true_dev"] = y
            if market == "efficient":
                mu = H @ m
                S = H @ P @ H.T + np.eye(len(wg)) * params.obs_std**2
                K = P @ H.T @ np.linalg.solve(S, np.eye(len(wg)))
                m = m + K @ (y - mu)
                P = 0.5 * ((P - K @ H @ P) + (P - K @ H @ P).T)
                out.loc[wg.index, "resid"] = y - mu
            else:
                out.loc[wg.index, "resid"] = y
    return out
