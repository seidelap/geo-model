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

## Commands (repo root)
- `python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25`
  (~12 min; `--limit 3` for a smoke run; `--report-only` regenerates the report)
- Stage 02 (each invocation < 25 min): `python -m research.privileged_tracking.soccer.imputation
  --stage cv --targets shape|ball|pass` (or comma lists of targets; cached configurations are
  skipped, `--force` refits), then `--stage transfer`, `--stage forward`, `--stage mlp`,
  `--stage final`, `--stage report`; then `python -m research.privileged_tracking.soccer.apply_student`.
  `--smoke` runs any stage on the first parquet row group without writing outputs.
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
- The neural student is scikit-learn `MLPRegressor` (PyTorch is not installed); its design is
  `one_hot_design(E2) + seq_mlp_features` (one-hot type groups, x / 120, y / 80, log1p(dt), same).
