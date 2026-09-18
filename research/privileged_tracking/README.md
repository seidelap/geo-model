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
| NFL 03 imputation | `python -m research.privileged_tracking.nfl.imputation` (then `python -m research.privileged_tracking.nfl.apply_student --season 2018`, `python -m research.privileged_tracking.nfl.ood_highlights`) | `processed/nfl/imputed_oof.parquet` (out-of-fold student predictions for every tracked play, `<target>__<fset>`), `imputed_forward.parquet`, `models/students_<fset>.joblib` (`apply_student.StudentBundle`), `imputed_pbp_2018.parquet`; `reports/nfl_03_imputation.md` + `nfl_03_*.parquet` (incl. `nfl_03_outcome_decomposition.parquet`: after-the-fact F1 skill vs the per-outcome-class baseline and inside each outcome class), `nfl_03_ood_highlights.md` + `nfl_03_ood_highlights_summary.parquet` (highlight skill with bootstrap CI vs 2017 OOF; no visible OOD degradation of the box / deep-safety students) |
| NFL 04 payoff | `python -m research.privileged_tracking.nfl.payoff [--stages a,b,c,d] [--force-apply]` | `processed/nfl/payoff_oof_tracked.parquet` (2017 per-play outcome predictions + F1T / F0T out-of-fold imputations), `payoff_pbp_<season>.parquet` (2018-2022 seasons with every student applied), `payoff_test_losses_{b,d}.parquet` (per-play 2022 predictions and losses of every set incl. the seed refits `<set>@seed<k>`), `payoff_receiver_week.parquet`, `models/students_F1T.joblib` / `students_F0T.joblib` (at-release students); `reports/nfl_04_payoff.md` + `nfl_04{a,b,c,d}_*.parquet` (incl. `nfl_04{a,b,d}_refit_noise.parquet`, `nfl_04c_imputation_vs_outcome.parquet`) |
| Soccer 01 build | `python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25` | `processed/soccer/events360.parquet` (one row per event with a 360 frame: `y_*` targets, `f_*`/`seq_*` event-only features), `events_no360.parquet` (shots + 25% of passes/carries in non-360 matches); `reports/soccer_01_build.md` + `soccer_01_*.parquet` |
| Soccer 02 imputation | `python -m research.privileged_tracking.soccer.imputation --stage cv --targets shape` (then `ball`, `pass`, `--stage transfer`, `forward`, `mlp`, `final`, `report`; each < 25 min, fits are cached) and `python -m research.privileged_tracking.soccer.apply_student` | `processed/soccer/imputed_oof.parquet` (out-of-fold student predictions for every events360 row, `<target>__<fset>` + `y_<target>`), `models/students_<fset>.joblib` (`apply_student.StudentBundle`), `imputed_no360.parquet` (students applied to the non-360 matches, `imp_<target>__<fset>`); `reports/soccer_02_imputation.md` + `soccer_02_*.parquet` |
| Soccer 03 payoff | `python -m research.privileged_tracking.soccer.payoff --stage all` (stages `shots`, `xg`, `distill`, `curve`, `passes`, `xpass`, `students`, `transfer`, `final`, `report`; cached, each < 25 min) | `processed/soccer/payoff_cache/` (shot / pass tables with out-of-fold imputations, per-shot predictions of every xG variant), `models/xg_models.joblib` (`payoff.XgBundle`: EVENT / EVENT+IMP / EVENT+ORACLESHOT xG fitted on all 360 shots); `reports/soccer_03_payoff.md` + `soccer_03_*.parquet` |
| Soccer 04 transfer | `python -m research.privileged_tracking.soccer.transfer --stage shots` (then `xg`, `passes`, `xpass`, `report`; cached under `processed/soccer/transfer_cache/`) | frozen stage-02 students applied to 2015/16 PL / La Liga / Serie A / Ligue 1 and to WC 2018 / Copa 2024 / AFCON 2023: oracle check vs `shot.freeze_frame`, within-domain xG / xPass with and without the imputations, zero-shot stage-03 xG, team-level sanity; `reports/soccer_04_transfer.md` + `soccer_04_*.parquet` |
| Soccer 05 sequence | `python -m research.privileged_tracking.soccer.sequence_student --variants seq20,seq1,seq0 --folds 0,1,2` | PyTorch GRU student over the raw 20-event history vs the stage-02 LightGBM E2 on identical held-out rows: `processed/soccer/imputed_oof_seq.parquet`, `models/seq/<variant>__fold<k>.pt` (`sequence_student.SeqBundle`); `reports/soccer_05_sequence.md` + `soccer_05_*.parquet` |
| NFL 06 chain | `python -m research.privileged_tracking.nfl.imputation_chain` | the participation -> alignment chain with *imputed* personnel: NFL 03 pre-snap students refitted with `F0I` = F0 + the NFL 02 out-of-fold `p_off_*` probabilities and `F0PI` = F0P + the same, k-fold and forward, paired game-clustered deltas; `reports/nfl_06_participation_chain.md` + `nfl_06_chain_*.parquet` |
| NFL 07 props | `python -m research.privileged_tracking.nfl.props_forward` | forward receiving-props test: next-game receiving yards / targets per receiver-game from the season-to-date history with and without the imputed separation / cushion (and the NGS history as an oracle), rolling test seasons 2020-2022, week-clustered paired deltas, refit spread; `processed/nfl/props_forward_frame.parquet`; `reports/nfl_07_props_forward.md` + `nfl_07_props_*.parquet` |
| Soccer 06 seq payoff | `python -m research.privileged_tracking.soccer.payoff_seq --stage all` | stage-03 xG / xPass refitted on the stage-05 folds with the `seq20` out-of-fold imputations next to the LightGBM ones (`EVENT+SEQ7(lgbm)` vs `EVENT+SEQ7(seq20)` like for like); cached under `processed/soccer/payoff_cache/seq_*`; `reports/soccer_06_seq_payoff.md` + `soccer_06_*.parquet` |
| Synthesis | `python -m research.privileged_tracking.synthesis.results_index` | `reports/results_index.parquet` (one row per headline metric with n, CI and the source table / row); the cross-sport write-up is `reports/README.md` |

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

Results are written up in `reports/README.md` (cross-sport synthesis; read it first) with one machine-written report per stage next to it and `reports/results_index.parquet` mapping every headline number to its source table.
