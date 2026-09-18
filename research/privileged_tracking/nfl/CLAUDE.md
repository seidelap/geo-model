# research/privileged_tracking/nfl

NFL side of the privileged-tracking research. Raw data lives under `nfl_dir()` (see
`common/io.py`, override with `PRIV_DATA_DIR`); processed parquet under `processed_dir('nfl')`;
result tables under `reports_dir()`.

Stages and entry points (run from the repo root):

- `build_tables.py` (NFL 01): 2017 Big Data Bowl tracking -> per-play tracking targets joined to nflfastR.
  Pure geometry lives in `tracking_features.py`.
- `participation.py` (NFL 02): "who is on the field" from play-by-play + tendencies.
  Pure functions in `participation_features.py`, player-level stage (c) in `participation_players.py`.
- `imputation.py` (NFL 03): event-only LightGBM "students" that predict the NFL 01 tracking targets from
  play-by-play; pure feature code in `imputation_features.py`; `apply_student.py` applies the saved students
  to any nflfastR season; `ood_highlights.py` (NFL 03b) checks the pre-snap students on NGS highlight plays
  (snap frame = the earliest `ball_snap` frame whose ball row is plausible, `ball_row_valid`: in the field and within
  2.5 yd of `absoluteYardlineNumber`; rows de-duplicated on `nflId` / `displayName` because a doubly-tagged frame lists
  every player twice; plays must be exactly 11 v 11 after that; when no snap frame has a plausible ball row the box's
  lateral reference is the centre's / OL median y, `lateral_reference`; index rows that repeat a (gameId, playId) are
  dropped; highlight skill carries a play-level bootstrap CI and `nfl_03_ood_highlights_summary.parquet` says whether
  the 2017 OOF skill lies inside it). Never trust a raw highlight ball row: one play has it 25 yd off the field.
- `payoff.py` (NFL 04): do imputed tracking features improve event-only outcome models (completion, yards, EPA,
  run success) on the tracked games (k-fold), on 2018-2022 (train 2018-2021, test 2022), per receiver-week vs NGS
  weekly receiving (against a fair after-the-fact baseline), and via imputed vs true personnel; every delta carries a
  game-clustered CI and a seed-refit spread.
- `imputation_chain.py` (NFL 06): the participation -> alignment chain with imputed personnel. Reuses `imputation.load_tracked`
  / `fit_with_inner_holdout` / `folds_for` with an explicit feature-set table (`chain_feature_sets`: `F0`, `F0I` = F0 + the
  NFL 02 out-of-fold `p_off_*`, `F0P`, `F0PI`); `attach_participation` joins `participation_imputed.parquet` on
  (`pbp_game_id`, `playId`). Result: `F0I` recovers none of the `F0 -> F0P` step (2 of 12 pre-snap targets improve by
  clustered CI). Tests: `tests/test_nfl_imputation_chain.py`.
- `props_forward.py` (NFL 07): forward receiving-props test on the stage-04 season frames. `receiver_game_table` ->
  `history_features` (`cur_` / `std_` season-to-date / `l3_` last-three aggregates from weeks <= t only, `next_*` targets =
  the receiver's next game with a target) -> `forward_frame`; LightGBM per test season 2020-2022 trained on the seasons
  before it (`fit_forward`), paired squared / absolute-error deltas vs `HIST` with (season, week)-clustered CIs and a
  three-seed refit spread. Populations `all` and `regular` (props-eligible). Tests: `tests/test_nfl_props_forward.py`.

Conventions established in NFL 01 (`build_tables.py`, `tracking_features.py`):

- **Coordinates.** Offense attacks +x; `depth = x - los_x` from the official yard line (the tracked
  ball sits ~0.5 yd behind it); `lateral = y - ball_y_ref` with **positive = offense's left**
  (`tracking_features.LEFT_SIGN`; nflfastR `pass_location == "left"` <-> `target_lateral > 0` on
  98.9% of passes). All `*_left` / `*_right` columns use the offense's perspective.
- **Column tier contract** for `plays_tracked.parquet` (`column_tier(name)`, table in
  `reports/nfl_01_column_definitions.parquet`): `id`, `event_only_presnap` (the only admissible
  features of a pre-snap state model), `event_only_postsnap`, `ngs_charting_privileged` (`ngs_*`
  from pbp_participation plus BDB `offenseFormation` / `personnel_*` / `defendersInTheBox` /
  `numberOfPassRushers`), `tracking_target`, `tracking_context`. Participation columns are never
  published as `part_*`.
- **Agreement metrics.** `exact` is `|a - b| <= 1e-6`, not bit equality. Definition grids
  (`box_count_tuned`, `n_pass_rushers_tuned`) are selected on weeks 1-3 and reported on weeks 4-6.
- Outputs are sorted by (gameId, playId[, nflId]) and byte-reproducible across runs.

Conventions established in NFL 02:

- **Tendency leakage contract.** Any team/player "tendency" feature is built with
  `participation_features.prior_game_class_counts` / `prior_game_value_sums` (strictly earlier games
  of the same season, ordered by `week`), `previous_season_counts` (whole previous season) or
  `league_prior_shares` (strictly earlier `(season, week)` slots). Shrinkage is hierarchical:
  league -> previous season -> in-season -> bucket-conditional (`TendencyConfig` alphas, in plays).
  Unlabelled plays (NaN class) contribute no counts; `collapse_classes` keeps NaN as NaN.
- **Class vocabularies** are chosen on the training seasons only (`ClassSpec.from_plays`, top classes
  covering 95% + `other`). Feature names embed `participation_features.slug(class)`
  (`"1 RB, 1 TE, 3 WR"` -> `1RB_1TE_3WR`) because LightGBM rejects commas/spaces.
- **Protocol.** Train 2016-2020, choose every setting on 2021, report 2022 once. Out-of-fold (by game)
  predictions are used wherever an imputed quantity feeds a later model on the training seasons.
- **Imputer API.** `participation.OffenseGroupingImputer.load().predict_proba(plays)` returns
  `[n, C]` class probabilities for a pbp-like frame (columns in `participation.PBP_COLS` plus the
  participation label columns, NaN allowed). Tendencies are rebuilt from the frame passed in, so pass
  the labelled history together with the rows to impute (the model was trained with 2016+ history).
- **Player candidates** are the weekly roster rows with status `ACT`, side-matched position groups
  (`participation_features.roster_group`). In `roster_weekly_<season>` `ACT` is exactly the 48-man
  game-day active list and `INA` the declared inactives (~6 per team-week, none takes a snap); both are
  fixed ~90 min before kickoff, so ACT candidates are pre-play but *not* mid-week information (a
  mid-week forecast must model the inactive list itself). The snap-count file lists only players who
  took a snap, so its row existence is post-game information (used only as a diagnostic split in the
  QB check, `participation_players.qb_check_table`).
- **Bootstrap CIs** come in pairs: play-level (`ci_low` / `ci_high`) and game-clustered
  (`common.metrics.clustered_bootstrap_delta`, `ci_low_game` / `ci_high_game`); quote the clustered
  one. Per-play test losses of every model and baseline are saved under `processed_dir('nfl')`
  (`participation_test_losses_{a,b}.parquet`) and per-game loss sums in
  `reports/nfl_02a_offense_game_losses.parquet` / `nfl_02b_defense_game_losses.parquet`.
- **Numeric tendency priors** (`participation.MEAN_PRIORS`) are target-free constants; never shrink a
  running mean towards a statistic computed over the whole frame (that leaks held-out targets).
- Personnel strings follow the NGS convention parsed by `tracking_features.parse_personnel`;
  slots per position group come from `participation_features.offense_slots` / `defense_slots`.
- Report tables go through `common/report.py::md_table` (ints without decimals, NaN blank).

Conventions established in NFL 03 (`imputation.py`, `imputation_features.py`, `apply_student.py`):

- **Feature sets are nested and named by the information assumed available** (`imputation_features.FEATURE_SETS`):
  `F0n` situation + strictly-prior `tend_*` team rates (from the *full* nflfastR season, `pbp_team_tendencies`,
  shrunk to fixed constants in `TENDENCIES`); `F0` + `te_off` / `te_def` team target encodings (`TeamEncoder`:
  fitted on training-fold plays only, leave-one-game-out for training rows); `F0P` + formation / personnel counts;
  `F1` + post-play pbp fields; `F2` + official `defenders_in_box` / `number_of_pass_rushers`. Categorical columns
  are integer codes over the fixed vocabularies in `VOCABS` (never learned from a fold).
- **Targets** are declared in `imputation_features.TARGETS` (`TargetSpec`: kind reg / count / binary, subset all /
  pass / run, source column, optional NGS `oracle`). `pressure_derived = min_def_dist_qb_throw <= 2.0` (threshold
  selected on weeks 1-3 against `was_pressure`, reported on weeks 4-6). Skill = R2 for reg / count, BSS for binary.
- **Protocol.** 5-fold `group_kfold` by game + forward split weeks 1-4 -> 5-6; rounds by early stopping on an inner
  20% game holdout of the training fold (never the test fold); final students refit on all games with that round
  count. Baselines `base_global`, `base_play_type` (post-snap, F1-level), `base_dd` and `base_outcome` (per nflfastR
  outcome class: sack x QB hit x completion x interception for passes, yards-gained bucket for runs;
  `imputation.outcome_bucket`) sit in every table.
- **After-the-fact skill must be decomposed by outcome class.** F1 / F2 students see the outcome flags, and the
  within-play targets differ by outcome (sacks have long `time_to_throw` and a defender at the QB by definition), so a
  pooled F1 R2 / AUC on `time_to_throw`, `pressure_derived`, `separation_at_arrival` mostly measures outcome
  identification. Always quote `F1_vs_outcome` (skill relative to `base_outcome`) and the within-class skill
  (`nfl_03_outcome_decomposition.parquet`, `nfl_03_error_<target>_by_<grouping>.parquet`, `imputation.OUTCOME_GROUPINGS`)
  next to the pooled number. The official `time_to_throw` / `was_pressure` fields are missing on every sack, so any
  comparison against them (external check on another season) is a non-sack comparison (`share_sack` column).
- **Student API.** `apply_student.StudentBundle.load(fset).predict(X, posteam, defteam, masks)` or, end to end,
  `apply_student.impute(apply_student.prepare_pbp(pbp, participation, positions), ("F0", "F0P", "F1"))` which adds
  `imp_<target>__<fset>` columns to a whole-season nflfastR frame (tendencies are rebuilt from the frame passed in;
  team encodings are the 2017 tracked-season values). Pre-snap students apply to scrimmage plays
  (`imputation_features.SCRIMMAGE_PLAY_TYPES`), pass / run students to their play type. Official NGS fields (`ngs_time_to_throw`, `ngs_was_pressure`, `ngs_air_yards`,
  `defenders_in_box`, `number_of_pass_rushers`) are merged only for the external check and are features of F2 alone.
- Out-of-fold predictions for every tracked play live in `processed_dir('nfl')/imputed_oof.parquet`
  (`<target>__<fset>` + `y_<target>`), forward-split predictions in `imputed_forward.parquet`, applied seasons in
  `imputed_pbp_<season>.parquet`; models under `processed_dir('nfl')/models/students_<fset>.joblib`.

Conventions established in NFL 04 (`payoff.py`):

- **Outcome models never see the outcome.** `payoff.FORBIDDEN_COLS` (`complete_pass`, `interception`, `yards_after_catch`,
  `yards_gained`, `epa`, `success`, `cp`, ...) are rejected by `payoff.leak_check` on every feature list. The NFL 03 F1 / F2
  students take those fields as inputs, so `imp_*__F1` columns are inadmissible as features of any outcome model (the
  `PBP+IMP_F1(leaky)` row exists only to show the leak). Within-play state for outcome models comes from the *at-release*
  students `F1T` (F0P inputs + `payoff.AT_RELEASE_COLS`: play type, dropback / scramble, pass length / location, air yards,
  sack, QB hit, receiver group, run location / gap) and `F0T` (same on the F0 base); their bundles live next to the NFL 03 ones
  (`models/students_F1T.joblib`) and `apply_student.impute(..., ("F1T",))` applies them like any other set.
- **Task subsets.** `payoff.task_mask`: pass tasks = pass plays that are not sacks and not spikes (both flagged by nflfastR),
  throwaways stay in (`cp` is NaN there, so `nflfastR cp` rows cover fewer plays); run tasks = `play_type == "run"`.
- **Feature-set names** `PBP`, `PBP+PERS`, `PBP+IMP` (F0P pre-snap + F1T within-play), `PBP+IMP_F0` (strictly event-only),
  `PBP+ORACLE_PRESNAP` (true values of the 12 pre-snap targets), `PBP+ORACLE_WITHIN` (true within-play targets, measured at
  or after the throw / handoff: quasi-outcomes), `PBP+ORACLE` (both), `PBP+NGS` (personnel + official `defenders_in_box` /
  `number_of_pass_rushers` / `ngs_air_yards` / `was_pressure` / `time_to_throw`), `PBP+IMP+NGS`;
  `payoff.payoff_feature_columns(set, subset)` is the single source of truth. Official fields carry the `ngs_` prefix in
  every frame of this stage. **Always split the oracle**: the full-oracle gain is entirely within-play (post-release)
  state; the pre-snap oracle adds nothing at play level, so `PBP+ORACLE` is not a ceiling for an at-release imputation.
- Stage a reuses the NFL 03 folds (`group_kfold(gameId, 5, seed=0)`) so test-fold imputations are out-of-fold; stages b-d use
  students trained on 2017 only (disjoint domain) and report 2022 once. Deltas are `from` minus `to` (positive = `to` better)
  with play-level and game-clustered CIs; regression deltas are given on squared (`se`) and absolute (`ae`) error.
- `pbp_participation` `was_pressure` is an object column (True / False / None): coerce with `pd.to_numeric(errors="coerce")`
  before LightGBM.
- **Compare like with like.** `nflfastR cp` is NaN on throwaways (all incompletions, near-zero loss for every model), so an
  all-plays log-loss must never be set next to the `cp` row; every completion table carries `<set> [cp rows]` rows
  (`payoff.metrics_table_on`, `CP_ROW_SETS`) and the `nflfastR cp -> <set>` deltas are the only admissible cp comparison.
- **Refit noise next to the bootstrap CI.** A single-fit bootstrap CI does not see refit-to-refit variation (~0.001 nats on
  70k-row completion models). `REFIT_SETS` (`PBP`, `PBP+IMP`, `PBP+IMP_F0`, stage a also `PBP+ORACLE_PRESNAP`; stage d `PBP`,
  `PBP+IMP+IMP_PERS`) are refit under
  `PayoffConfig.refit_seeds` (inner early-stopping holdout + LightGBM bagging seed change, folds / splits do not);
  `refit_spread_table` / `attach_refit_spread` put the larger spread of the two members of each pair into every deltas table
  as `refit_spread` with `refit_measured` = `both` / `from` / `to` / `none` (`FIXED_MODELS` such as `nflfastR cp` count as
  zero-spread members). The verdict (`_sig`) calls a delta "within refit noise" when |delta| <= spread of a fully measured
  pair; a half-measured pair is labelled as such. Never borrow one model's spread for a pair it is not part of. Tables:
  `reports/nfl_04{a,b,d}_refit_noise.parquet`.
- **After-the-fact aggregates need an after-the-fact baseline.** A student that consumes outcome fields (NFL 03 F1) must be
  scored against a baseline holding the same fields aggregated the same way (`payoff.AFTER_FACT_COLS`, `AFTER_PROXIES`,
  `FAIR_BASELINE = "naive + after-the-fact"`); the naive-only comparison overstated the receiver-week gain twofold.
  `imputation_vs_outcome_table` quantifies how much of an imputation is a linear function of those fields (F1 separation: 0.67).
  `proxy_regression_table` fits every proxy model on the same receiver-weeks and reports `delta_r2_vs_baseline` with a paired
  bootstrap CI (per-week squared errors / test variance), so every stage-c gain is a paired claim like the play-level ones.
- `qb_hit` is a charted post-play flag but sits in the at-release set (`AT_RELEASE_COLS`, `PBP_COLS["pass"]`) on every side of
  every comparison: "at release" means "up to and including QB contact"; say so wherever the set is described.
