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

Stages and entry points (run from the repo root):

| stage | command | outputs |
|---|---|---|
| NFL 01 build | `python -m research.privileged_tracking.nfl.build_tables` | `processed/nfl/plays_tracked.parquet`, `participants_tracked.parquet`, `player_crosswalk.parquet`; `reports/nfl_01_build.md` + `nfl_01_*.parquet` |
| NFL 02 participation | `python -m research.privileged_tracking.nfl.participation [--stages a,b,c]` | `processed/nfl/participation_plays.parquet` (2016-2022 scrimmage plays with participation labels), `participation_imputed.parquet` (out-of-sample offense / defense grouping probabilities per play), `participation_offense_model.joblib` (offense-grouping imputer: `participation.OffenseGroupingImputer.load().predict_proba(pbp_like_df)`), `participation_test_losses_{a,b}.parquet` (per-play test losses of every model and baseline); `reports/nfl_02_participation.md` + `nfl_02*.parquet` (incl. per-game loss sums for clustered CIs) |
| NFL 03 imputation | `python -m research.privileged_tracking.nfl.imputation` (then `python -m research.privileged_tracking.nfl.apply_student --season 2018`, `python -m research.privileged_tracking.nfl.ood_highlights`) | `processed/nfl/imputed_oof.parquet` (out-of-fold student predictions for every tracked play, `<target>__<fset>`), `imputed_forward.parquet`, `models/students_<fset>.joblib` (`apply_student.StudentBundle`), `imputed_pbp_2018.parquet`; `reports/nfl_03_imputation.md` + `nfl_03_*.parquet` (incl. `nfl_03_outcome_decomposition.parquet`: after-the-fact F1 skill vs the per-outcome-class baseline and inside each outcome class), `nfl_03_ood_highlights.md` |
| Soccer 01 build | `python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25` | `processed/soccer/events360.parquet` (one row per event with a 360 frame: `y_*` targets, `f_*`/`seq_*` event-only features), `events_no360.parquet` (shots + 25% of passes/carries in non-360 matches); `reports/soccer_01_build.md` + `soccer_01_*.parquet` |
| Soccer 02 imputation | `python -m research.privileged_tracking.soccer.imputation --stage cv --targets shape` (then `ball`, `pass`, `--stage transfer`, `forward`, `mlp`, `final`, `report`; each < 25 min, fits are cached) and `python -m research.privileged_tracking.soccer.apply_student` | `processed/soccer/imputed_oof.parquet` (out-of-fold student predictions for every events360 row, `<target>__<fset>` + `y_<target>`), `models/students_<fset>.joblib` (`apply_student.StudentBundle`), `imputed_no360.parquet` (students applied to the non-360 matches, `imp_<target>__<fset>`); `reports/soccer_02_imputation.md` + `soccer_02_*.parquet` |

NFL conventions (see `nfl/tracking_features.py` docstring and `nfl/CLAUDE.md`): plays are
normalised so the offense attacks toward +x; depth is measured from the official yard line, not
the tracked ball (which sits ~0.5 yd behind it and glitches); positive `lateral` is the offense's
LEFT and every `*_left`/`*_right` column follows the offense's perspective (matches nflfastR
`pass_location`); `frame.id` is the 10 Hz clock (the `time` column has 1 s resolution); the passer
is identified 3 frames before the `pass_forward` tag. Leakage contract of `plays_tracked.parquet`:
every column has a tier in `reports/nfl_01_column_definitions.parquet`
(`nfl.build_tables.column_tier`): only `id` and `event_only_presnap` columns may be features of a
pre-snap state model; `event_only_postsnap` (pbp outcomes / post-snap models), `ngs_*` and the BDB
charting columns (`ngs_charting_privileged`), `tracking_target` and `tracking_context` are labels,
oracles or diagnostics.

Soccer conventions (see `soccer/CLAUDE.md`): column prefixes are the leakage contract (`f_*` + `seq_*` event-only, of which `f_after_*` are post-instant attributes to be reported with and without; `y_*` 360 targets, usable only where `y_frame_ok == 1` because StatsBomb stores paired-event frames in the other team's orientation; `sff_*` shot freeze frame; `oracle_*`/`post_*` never features); all coordinates in the event team's attacking frame; windows reset each period.
Soccer 02 students (`soccer/imputation_features.py`): nested feature sets `loc` < `E0` < `E1` < `E2` < `E2a` (the last adds the post-instant `f_after_*` block), fixed StatsBomb vocabularies, per-target row subsets (team shape on possession-team reliable frames, ball-relative on all usable frames, pass lanes on Pass rows); `apply_student.impute(df, ("E0", "E2", "E2a"))` adds `imp_<target>__<fset>` to any build-stage frame.

Tests: `python -m pytest research/privileged_tracking/tests -q`.

Results are written up in `reports/README.md` once the experiments have run.
