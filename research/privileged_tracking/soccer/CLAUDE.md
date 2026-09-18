# research/privileged_tracking/soccer

StatsBomb 360 freeze frames as "privileged" supervision for event-only models.
Stage 01 (`build_tables.py`) writes the data layer; later stages model on top of it.

## Modules
- `frame_features.py` pure geometry over one freeze frame (numpy in, dict out). Handles
  both 360 frames (`teammate/actor/keeper` flags) and `shot.freeze_frame` (keeper via
  `position.name`). `resolve_frame_orientation` detects StatsBomb's paired-event defect
  (frame stored in the other team's orientation, sometimes with the other team's flags)
  per row and repairs (mirror / mirror + flag swap) or rejects the frame.
  Tests: `tests/test_soccer_frame_features.py`.
- `event_features.py` pure per-event features over one match's ordered events: possession
  context, score state, K=10 window, 60 s opponent defensive x, fixed-width 20-slot
  sequence block. Tests: `tests/test_soccer_event_features.py`.
- `build_tables.py` driver (2 workers, streaming `ParquetWriter`, fixed pyarrow schema
  shared by both outputs) + data-quality report `reports/soccer_01_build.md`.
- `imputation_features.py` (stage 02, pure): target catalogue `TARGETS` (`TargetSpec`: kind
  reg / count / binary, family shape / ball / pass, row `subset`, label rule), nested feature
  sets `FEATURE_SETS` (`loc`, `E0`, `E1`, `E2`, `E2a`, `E3`; alias `E2r`), fixed `VOCABS`,
  `build_design`, `subset_masks`, `target_frame` (incl. the derived `deep_block` / `counter_on`),
  MLP encodings and the sequence-block helper `events_since_opp_def_action`.
  Tests: `tests/test_soccer_imputation_features.py`.
- `imputation.py` (stage 02 driver): staged (`--stage cv|transfer|forward|mlp|final|report`),
  every fit cached under `processed_dir('soccer')/imputation_cache/` so the work is split over
  several short invocations; writes `imputed_oof.parquet`, `models/students_<fset>.joblib`,
  `reports/soccer_02_imputation.md` + `soccer_02_*.parquet`.
- `apply_student.py` (stage 02): `StudentBundle.load(fset)` / `impute(df, fsets)` add
  `imp_<target>__<fset>` columns to any build-stage frame; CLI imputes `events_no360.parquet`
  into `imputed_no360.parquet` and writes the shift tables. Tests: `tests/test_soccer_apply_student.py`.
- `payoff_features.py` (stage 03, pure): xG / xPass feature catalogue (`XG_VARIANTS`,
  `XPASS_VARIANTS`, `SHOT_STATE`, `ASSIST_STATE`, `PASS_STATE`, `SFF_MAP`), event-only designs on top
  of `build_design` (`shot_event_design`, `pass_event_design`), the assist link (`key_pass_links`,
  `assist_block`), state blocks (`state_block`, clipped at 0), labels / slices / match-stratified
  subsample, `binary_metrics`, `paired_delta`, `metrics_table`, `deltas_table`, `slice_table`,
  `calibration_frame`. Tests: `tests/test_soccer_payoff_features.py`.
- `sequence_student.py` (stage 05): PyTorch multi-task student over the raw 20-slot `seq_*`
  history (GRU default, transformer optional) + an MLP over the E2 current-event design, scored
  against the stage-02 LightGBM E2 out-of-fold predictions on identical folds / rows. Pure
  helpers `seq_arrays`, `sequence_tokens`, `split_design`, `NumericStats`, `TargetStats`,
  `target_matrix`, `decode_outputs`, `bootstrap_delta_per_sample`; `SeqBundle.load(path)
  .predict_frame(df)` applies a checkpoint to any build-stage frame. Caches per variant / fold
  under `processed_dir('soccer')/seq_cache/`; writes `imputed_oof_seq.parquet`,
  `models/seq/<variant>__fold<k>.pt`, `reports/soccer_05_sequence.md` + `soccer_05_*.parquet`.
  Tests: `tests/test_soccer_sequence_student.py`.
- `payoff.py` (stage 03 driver): staged (`--stage shots|xg|distill|curve|passes|xpass|students|
  transfer|final|report`), caches under `processed_dir('soccer')/payoff_cache/`; `fit_gbm` /
  `cv_predict` / `distill_oof`, `XgBundle` (`models/xg_models.joblib`, `XgBundle.load().models[v]
  .predict(design)`), writes `reports/soccer_03_payoff.md` + `soccer_03_*.parquet`.
  Tests: `tests/test_soccer_payoff.py`.

- `payoff_seq.py` (stage 06 driver): payoff of the stage-05 sequence student. Restricts the stage-03
  shot / pass tables to the folds with `seq20` predictions (`restrict_to_folds`), attaches
  `imp_<t>__E2` / `imp_<t>__seq20` / `y_<t>` for the seven stage-05 targets from `imputed_oof.parquet`
  and `imputed_oof_seq.parquet` (`attach_oof`, keyed by `event_id`), composes variants from named
  state blocks (`compose_designs`, catalogues `XG_SEQ_VARIANTS` / `XPASS_SEQ_VARIANTS`; the
  like-for-like pair is `EVENT+SEQ7(lgbm)` vs `EVENT+SEQ7(seq20)`), reuses `payoff.cv_predict` with
  3-fold CV over those folds, and reports per-sample and match-clustered deltas plus direct pairs
  (`pairs_table`) and imputation quality on exactly those rows (`state_quality_table`). Caches
  `payoff_cache/seq_*.parquet`; writes `reports/soccer_06_seq_payoff.md` + `soccer_06_*.parquet`.
  Tests: `tests/test_soccer_payoff_seq.py`.
- `transfer.py` (stage 04 driver): transfer to the seasons without 360 (2015/16 PL / La Liga /
  Serie A / Ligue 1, World Cup 2018, Copa America 2024, AFCON 2023). Stages `shots|xg|passes|xpass|
  report`, cached under `processed_dir('soccer')/transfer_cache/`: (a) `oracle_check_table` /
  `oracle_calibration` (imputed E2 shot state vs `sff_*`, with the 360 out-of-fold rows as the
  like-for-like reference), (b) `stage_xg` (within-2015/16 CV via `payoff.xg_designs` +
  `payoff.cv_predict`, plus the stage-03 `XgBundle` zero-shot through `complete_design`),
  (c) `stage_xpass` (frozen E2 / E2a students + a receiver student fitted on the 360 Pass rows),
  (d) `team_block_table` / `team_season_table` / `spearman_table` vs `team_match.parquet`
  and `shift_tables`, (e) `minute_profile` / `bin_profile` (imputed block depth vs observed
  defensive-action x per minute). Key passes of non-360 shots are rebuilt with
  `build_tables.process_match` (`frac = 1`, Pass rows) because the 25% pass sample misses them.
  Writes `reports/soccer_04_transfer.md` + `soccer_04_*.parquet`. Tests: `tests/test_soccer_transfer.py`.

## Commands (repo root)
- `python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25`
  (~12 min; `--limit 3` for a smoke run; `--report-only` regenerates the report)
- Stage 02 (each invocation < 25 min): `python -m research.privileged_tracking.soccer.imputation
  --stage cv --targets shape|ball|pass` (or comma lists of targets; cached configurations are
  skipped, `--force` refits), then `--stage transfer`, `--stage forward`, `--stage mlp`,
  `--stage final`, `--stage report`; then `python -m research.privileged_tracking.soccer.apply_student`.
  `--smoke` runs any stage on the first parquet row group without writing outputs.
- Stage 03: `python -m research.privileged_tracking.soccer.payoff --stage all` (~25 min total;
  stages are cached and individually re-runnable, `--smoke` runs on 40 matches, `--force` refits).
- Stage 05: `python -m research.privileged_tracking.soccer.sequence_student --stage train
  --variants seq20,seq1,seq0 --folds 0,1,2` (~10 min per seq20 fold on 2 threads, cached per
  variant / fold, `--force` refits) then `--stage report --folds 0,1,2`; `--stage all --smoke
  --max-epochs 2 --train-cap 20000` runs on the first parquet row group without writing outputs.
- Stage 06: `python -m research.privileged_tracking.soccer.payoff_seq --stage all` (~3.5 min: xg 0.5,
  xpass 1.6, report 1.2; `--force` refits, `--no-write` prints instead of writing).
- Stage 04: `python -m research.privileged_tracking.soccer.transfer --stage all` (~25 min in
  five cached invocations: `shots` ~8 min incl. the key-pass rebuild with 2 workers, `xg` ~8 min,
  `passes,xpass` ~5 min, `report` ~2 min; `--smoke` runs on 30 matches per domain in a separate
  cache without writing, `--force` refits, `--example-match <id>` picks the worked example).
- `python -m pytest research/privileged_tracking/tests -q`

## Conventions
- Column prefixes are the leakage contract: `f_*` + `seq_*` are the only event-only
  features; `y_*` are 360 targets; `sff_*` come from `shot.freeze_frame`; `oracle_*`
  (`statsbomb_xg`, `one_on_one`, `open_goal`) and `post_*` (outcomes) are never features.
- `f_after_*` (pass end / length / angle / height / switch / cross / through-ball /
  cut-back / progress, carry end / length / progress, duration) are event-only but
  POST-INSTANT (they describe the realised trajectory after the frame instant). Report
  every imputation / payoff result with and without them; `instant` column of
  `reports/soccer_01_columns.parquet` marks them.
- 360 frame orientation: `y_frame_orientation` in {ok, mirrored, mirrored_swapped}
  means usable (`y_frame_ok == 1`; the last two were repaired, `y_frame_repaired == 1`);
  {far, unresolved, no_actor, no_location, empty} rows have NaN geometry targets and
  `y_reliable == 0`. Mis-oriented raw frames are ~6% of rows and concentrate on
  Dispossessed, Foul Won, Dribbled Past, aerial Duels, Dribbles paired with a Duel,
  50/50 and incomplete Ball Receipt*; Pass / Shot / Pressure / Carry frames are fine.
- All coordinates in a row are in the event team's attacking frame (goal at x=120);
  opponent events are mirrored `(120-x, 80-y)` when entering windows / sequence.
- Timestamps restart each period: windows and the sequence block look back only within
  the period; period 5 (shoot-outs) is dropped; score state counts periods 1-4.
- `y_block_*` describe the opponents of the *event* team; filter on
  `f_is_possession_team` for a defensive-block target. Train on `y_reliable == 1`
  (frame usable and >= 7 opponents visible) and report both.
- Non-360 matches keep all Shots and a fixed 25% of Pass/Carry per match
  (RNG seeded by match id); say so in every downstream report. 360-flagged matches whose
  frames are corrupt or whose `event_uuid`s match no event fall back to this output
  (`has_360 = False`); 9 such matches, listed at the bottom of the build report.
- StatsBomb data glitches handled in `event_features.py`: receipts stamped `00:00:00`
  late in a period inherit the previous event's time (`f_ts_repaired`); ball speed uses a
  0.5 s floor on elapsed time.
- Split by `match_id` (`common.splits.group_kfold`) or forward by `match_date`.

## Stage 02 conventions (imputation students)
- Feature sets are nested and named by the information assumed available: `loc` (x, y, goal
  geometry), `E0` current event, `E1` + possession context, `E2` + 10-event window / opponent
  defensive actions, `E2a` = E2 + the post-instant `f_after_*` block; `E2r` = E2 trained on
  reliable rows only; `E3` = E2 + raw `seq_*` (LightGBM reference for the MLP). Categorical
  strings are integer codes over the fixed StatsBomb vocabularies in `VOCABS` (never learned
  from a fold; unseen -> `len(vocab)`, NaN stays NaN); `f_gender` / `f_comp_type` (league vs
  tournament) are derived from the id columns.
- Row subsets per target (`TargetSpec.subset`): team-shape targets on possession-team events
  (`poss`), `deep_block` on settled possession (`settled`: Regular Play, >= 10 s), `counter_on`
  in the middle third (`mid`), pass-lane targets on Pass rows, `n_opp_within_5/10` and
  `nearest_opp_dist` on all rows, the two cone targets on possession-team events and the keeper
  distance on possession-team events in the attacking half (`poss_att`, x >= 60). Labels are
  valid only where `y_reliable == 1` (shape), `y_frame_ok == 1` (ball / pass) or additionally
  `y_keeper_consistent == 1` (keeper distance: mis-flagged keepers sit at ~114 yd); the
  `deep_block` threshold (25th pct of `def_line` on reliable settled rows) is fixed once and
  stored in every bundle.
- Protocol: 5-fold `group_kfold` by match, plus `m2f` / `f2m` (gender) and `fwd` (<= 2023/24 ->
  Euro 2024 + Women's Euro 2025) splits; training rows thinned uniformly to `train_cap`; rounds by
  early stopping on an inner 15% match holdout of the training rows; final students refit on all
  matches with the median CV round count. Baselines `base_global`, `base_type`, `loc` sit in every
  table; skill = R2 (reg / count) or BSS vs `base_global` (binary); deltas use
  `common.metrics.clustered_bootstrap_delta` by match.
- Cache / output naming: `imputation_cache/<stage>__<split>__<target>__<fset>.parquet` (`row`,
  `pred`) + `.json` fit metadata; `imputed_oof.parquet` columns `<target>__<fset>` + `y_<target>`;
  applied frames get `imp_<target>__<fset>` (NaN outside the subset).
- The stage-02 neural student is scikit-learn `MLPRegressor` (PyTorch was not installed at the
  time; it is now, CPU build); its design is `one_hot_design(E2) + seq_mlp_features` (one-hot
  type groups, x / 120, y / 80, log1p(dt), same). The PyTorch sequence student lives in stage 05.

## Stage 03 conventions (payoff)
- Downstream models use the stage-02 `fold` column of `imputed_oof.parquet` so that every imputed
  feature of a test-fold row is out-of-fold; imputed columns are read as `imp_<target>__<fset>`
  (E2 for the shot itself, E2a for the assist pass and for the with-after xPass design) and the 360
  oracle as `y_<target>` (stage-02 label rules). Shot-frame oracle = `sff_*` (`SFF_MAP`).
- Variant names: `EVENT`, `EVENT+IMP`, `EVENT+ORACLE360`, `EVENT+ORACLESHOT`(`_full`), `LOC`,
  `BASE`, reference `STATSBOMB_XG`; xPass designs `after` / `noafter`. Deltas are paired per-sample
  bootstraps (positive = better than the reference) with a match-clustered CI next to them.
- The assist is `shot.key_pass_id` read from the raw event JSON (`payoff_cache/shot_key_pass.parquet`);
  the key pass's `f_after_*` fields are pre-shot and therefore event-only for the shot.
- Penalties are excluded from every shot experiment.

## Stage 05 conventions (sequence student)
- Same folds and labels as stage 02: the `fold` column and `y_<target>` columns of
  `imputed_oof.parquet` are read directly (fold ids are re-verified against `group_kfold(seed=0)`),
  so every comparison with `lgbm_E2` / `base_type` is on identical held-out rows; the report
  states which folds were trained (3 of 5 under the CPU budget).
- Tokens run oldest -> newest (slot 20 ... slot 1); per-slot numerics are x/120, y/80,
  log1p(dt), own / opponent / pad flags and the displacement to the current event; the learned
  position id is `slot - 1`, so `seq1` (window 1) and `seq20` share the slot-1 embedding.
- `seq0` (no sequence branch) is the neural analogue of LightGBM E2; `seq1` vs `seq20` isolates
  what the raw history adds. Predictions of counts / distances are clipped at 0 (stage-02 LightGBM
  outputs in `imputed_oof.parquet` are not).
- Deltas: `bootstrap_delta_per_sample` (chunked per-row bootstrap, same semantics as
  `common.metrics.paired_bootstrap_delta`, which would allocate `n_boot x n` indices) plus the
  match-clustered CI; `torch.set_num_threads(2)`, seeds `cfg.seed + fold`.
