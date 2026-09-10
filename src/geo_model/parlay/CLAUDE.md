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
- `data_multisport.py` — NBA (2007+), NHL (2011-21), MLB (2021-25) loaders in the same
  cleaned schema; moneyline sports get a probit-implied margin as `spread_line`.
- `retrosheet.py` — starting pitchers from Retrosheet event files (Chadwick mirror).
- `multisport_backtest.py` — moneyline parlays at actual closing decimals, pitcher pairs.

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
- Only `raw.githubusercontent.com` and `github.com/<owner>/<repo>/releases/download/`
  are reachable from the sandbox. Multi-sport data lives under `data/raw/multisport/`.
- Source gotchas: the NBA sqlite stores `|spread|` for 2007-08..2021-22 (re-signed by
  moneyline favourite, rows inconsistent with the moneyline dropped); the 10-year
  MLB archive from the same scraper has mis-paired home/away rows and is rejected;
  MLB book prices must be medianed in probability space (never average American odds
  across +/-100) after dropping in-play captures.
- Regenerate MLB pitchers: sparse-clone `chadwickbureau/retrosheet` (`seasons/`), then
  `load_starting_pitchers(dir, [2021..2025]).to_parquet(MultiSportConfig().pitchers_parquet)`.
