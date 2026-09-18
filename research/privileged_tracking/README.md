# Privileged-information tracking research

Goal: learn game state from rare tracking data and apply it where only event or
play-by-play data exists. NFL first (2017 Big Data Bowl tracking, 91 games, against
nflverse play-by-play for every game), then soccer (StatsBomb 360 freeze frames as
tracking-like supervision, against plain StatsBomb events).

Layout:

- `common/`  shared IO, split and metric helpers (`PRIV_DATA_DIR` points at raw data)
- `nfl/`     play-level target construction, participation / alignment / movement
             imputation, downstream payoff experiments
- `soccer/`  per-event 360 state targets, event-window features, imputation model,
             xG / pass-difficulty payoff, transfer to non-360 seasons
- `reports/` machine-written result tables (Markdown / Parquet)
- `tests/`   unit tests for pure feature functions

Data directory (not committed): `data/raw/privileged/` contains
`nfl/bdb2017/` (games, plays, players, tracking CSVs), `nfl/nflverse/` (parquet
releases) and `sb/` (StatsBomb events, lineups, three-sixty, matches, processed).

Results are written up in `reports/README.md` once the experiments have run.
