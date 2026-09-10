# geo_model/parlay

Exploratory side experiment, not part of the C1-C6 pipeline: backtests whether
cross-game "explaining away" correlations make non-same-game parlays +EV.
Results and method: `docs/experiments/parlay-interaction-effects.md`.

## Modules
- `config.py` — `ParlayConfig` (data dir via `GEO_MODEL_DATA_DIR`, seasons, leg odds).
- `data.py` — nflverse games CSV → parquet cache → `clean_games` / `to_team_games`.
- `pairs.py` — shared-game leg pairs (H's next game vs. A's next game), strata helpers.
- `kalman.py` — causal Kalman filter on market errors; `simulate_games` for power checks.
- `backtest.py` — parlay pricing, break-even correlation, correlation summaries, ROI.

## Conventions and gotchas
- `spread_line` is the expected `home - away` margin (nflverse sign convention).
  `resid = result - spread_line`, so positive means the home side covered.
  Team-perspective rows in `to_team_games` flip the sign for the away team.
- The explaining-away correlation exists only relative to a market that updates
  on the shared game. Against a static line it appears as within-team residual
  persistence (a single-bet edge), not as cross-game correlation. Use
  `simulate_games(..., market="efficient")` for power checks.
- The one-shared-game formula in `gaussian_explaining_away_corr` is an upper
  bound for week-1 anchors; later weeks are smaller because the market has learned.
- Break-even phi for a 2-leg -110 parlay with 50/50 legs is 0.098, i.e. a
  same-sign rate of 54.9%.
- All filter predictions for a slate are computed before any game in that slate
  is observed. Fit hyperparameters on seasons strictly before the evaluation window.
- `tests/parlay/conftest.py::make_schedule` builds a synthetic schedule; tests
  never touch the network or the parquet cache.
