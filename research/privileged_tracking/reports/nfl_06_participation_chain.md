# NFL 06 - participation -> alignment chain: imputed personnel as a student input

Machine-written by `python -m research.privileged_tracking.nfl.imputation_chain`. Question: the
NFL 02 handoff recommended feeding the imputed offense grouping to the alignment students as a
weak prior; NFL 03 used either no personnel (`F0`) or the charted personnel (`F0P`). Does the
imputed grouping (`F0I`) recover any of the `F0 -> F0P` step, and does it add anything on top of
the charted truth (`F0PI`)?

## Headline

- Out of 12 pre-snap targets (5-fold by game, out-of-fold), `F0 -> F0I` is better by game-clustered
  CI on 2 and worse on 0; the largest skill change is cb_cushion (0.239 -> 0.230).
- `F0I -> F0P` (imputed -> charted personnel) is better by clustered CI on 10 of 12 targets; `F0P -> F0PI` on 1 of 12.
- Imputed grouping quality on the tracked plays: argmax accuracy 0.634 vs the NFL 02
  label (majority class 0.584); 98.7% of the
  11,518 tracked plays have a participation record (the rest are kneels / spikes).

## Protocol

* **Rows.** The 11,518 tracked 2017 plays (`plays_tracked.parquet`), design and targets as in NFL 03
  (`imputation.load_tracked`). The NFL 02 `p_off_*` columns (`participation_imputed.parquet`) are joined on
  (`game_id`, `play_id`); for 2016-2020 they are out-of-fold by game, i.e. the 2017 probabilities come from
  models that never saw the play's game (but did see other 2017 games and 2018-2020 games: a within-
  season / future-season coupling that NFL 03's own protocol shares through its k-fold team encodings).
* **Feature sets.** `F0` (31 columns); `F0I` (41 columns); `F0P` (40 columns); `F0PI` (50 columns).
  `F0` and `F0P` are exactly `imputation_features.FEATURE_SETS['F0' / 'F0P']`; `F0I` / `F0PI` append the 10
  probabilities. Team target encodings are fitted inside the training fold as in NFL 03.
* **Splits.** 5-fold `group_kfold` by `gameId` (seed 0, the NFL 03 folds) and the forward split weeks 1-4 ->
  5-6; LightGBM with the NFL 03 settings, early stopping on an inner 20% game holdout of the training fold.
* **Statistics.** Skill = R2 (continuous / count) or BSS (binary) vs the training-fold mean; paired bootstrap
  (1,000 resamples, seed 0) of per-play absolute error / log-loss, play-level and
  game-clustered 95% CIs; delta = loss(a) - loss(b), positive = `b` better. Quote the clustered CI.

## Skill, 5-fold by game (out-of-fold)

| target | n | base_global | F0 | F0I | F0P | F0PI |
|---|---|---|---|---|---|---|
| box_count_tuned | 11518 | -0.001 | 0.434 | 0.442 | 0.509 | 0.513 |
| cb_cushion | 11518 | -0.005 | 0.239 | 0.230 | 0.287 | 0.277 |
| def_y_std | 11518 | -0.004 | 0.443 | 0.443 | 0.538 | 0.537 |
| mof_open | 11518 | -0.001 | 0.255 | 0.257 | 0.264 | 0.259 |
| motion_derived | 11518 | -0.002 | 0.067 | 0.068 | 0.067 | 0.068 |
| n_backfield | 11518 | -0.002 | 0.144 | 0.148 | 0.487 | 0.486 |
| n_deep_safeties | 11518 | -0.000 | 0.400 | 0.393 | 0.403 | 0.396 |
| n_dl | 11518 | -0.001 | 0.117 | 0.115 | 0.124 | 0.124 |
| n_wide_left | 11518 | -0.001 | 0.159 | 0.163 | 0.209 | 0.215 |
| n_wide_right | 11518 | -0.001 | 0.179 | 0.183 | 0.240 | 0.240 |
| qb_depth | 11471 | -0.002 | 0.868 | 0.874 | 0.883 | 0.885 |
| shotgun_derived | 11518 | -0.002 | 0.939 | 0.939 | 0.957 | 0.956 |

## Skill, forward split (weeks 1-4 -> 5-6)

| target | base_global | F0 | F0I | F0P | F0PI |
|---|---|---|---|---|---|
| box_count_tuned | -0.000 | 0.430 | 0.430 | 0.495 | 0.499 |
| cb_cushion | -0.001 | 0.269 | 0.263 | 0.304 | 0.286 |
| def_y_std | -0.001 | 0.453 | 0.437 | 0.535 | 0.540 |
| mof_open | -0.000 | 0.239 | 0.241 | 0.244 | 0.242 |
| motion_derived | -0.000 | 0.050 | 0.052 | 0.048 | 0.049 |
| n_backfield | -0.000 | 0.160 | 0.159 | 0.508 | 0.501 |
| n_deep_safeties | -0.000 | 0.420 | 0.418 | 0.427 | 0.423 |
| n_dl | -0.000 | 0.119 | 0.115 | 0.132 | 0.126 |
| n_wide_left | -0.004 | 0.159 | 0.163 | 0.213 | 0.215 |
| n_wide_right | -0.001 | 0.191 | 0.188 | 0.241 | 0.244 |
| qb_depth | -0.001 | 0.938 | 0.939 | 0.955 | 0.956 |
| shotgun_derived | -0.001 | 0.946 | 0.945 | 0.964 | 0.963 |

## Paired deltas (k-fold; positive = `b` better)

| target | a | b | loss | n | skill_a | skill_b | delta [play CI] (game-clustered CI) | verdict |
|---|---|---|---|---|---|---|---|---|
| n_deep_safeties | F0 | F0I | abs_error | 11518 | 0.400 | 0.393 | -0.0013 [-0.0032, +0.0007] (game-clustered [-0.0045, +0.0016]) | n.s. |
| n_deep_safeties | F0I | F0P | abs_error | 11518 | 0.393 | 0.403 | +0.0004 [-0.0018, +0.0026] (game-clustered [-0.0036, +0.0043]) | n.s. |
| n_deep_safeties | F0 | F0P | abs_error | 11518 | 0.400 | 0.403 | -0.0009 [-0.0028, +0.0009] (game-clustered [-0.0039, +0.0021]) | n.s. |
| n_deep_safeties | F0P | F0PI | abs_error | 11518 | 0.403 | 0.396 | -0.0016 [-0.0033, +0.0002] (game-clustered [-0.0046, +0.0015]) | n.s. |
| mof_open | F0 | F0I | log_loss | 11518 | 0.255 | 0.257 | +0.0015 [-0.0009, +0.0040] (game-clustered [-0.0020, +0.0052]) | n.s. |
| mof_open | F0I | F0P | log_loss | 11518 | 0.257 | 0.264 | +0.0042 [+0.0013, +0.0068] (game-clustered [+0.0006, +0.0077]) | b better |
| mof_open | F0 | F0P | log_loss | 11518 | 0.255 | 0.264 | +0.0057 [+0.0036, +0.0076] (game-clustered [+0.0026, +0.0087]) | b better |
| mof_open | F0P | F0PI | log_loss | 11518 | 0.264 | 0.259 | -0.0029 [-0.0052, -0.0006] (game-clustered [-0.0059, +0.0004]) | n.s. |
| box_count_tuned | F0 | F0I | abs_error | 11518 | 0.434 | 0.442 | +0.0018 [-0.0009, +0.0044] (game-clustered [-0.0033, +0.0062]) | n.s. |
| box_count_tuned | F0I | F0P | abs_error | 11518 | 0.442 | 0.509 | +0.0285 [+0.0233, +0.0336] (game-clustered [+0.0223, +0.0344]) | b better |
| box_count_tuned | F0 | F0P | abs_error | 11518 | 0.434 | 0.509 | +0.0302 [+0.0252, +0.0351] (game-clustered [+0.0237, +0.0360]) | b better |
| box_count_tuned | F0P | F0PI | abs_error | 11518 | 0.509 | 0.513 | +0.0000 [-0.0023, +0.0022] (game-clustered [-0.0029, +0.0032]) | n.s. |
| n_dl | F0 | F0I | abs_error | 11518 | 0.117 | 0.115 | -0.0020 [-0.0044, +0.0005] (game-clustered [-0.0059, +0.0022]) | n.s. |
| n_dl | F0I | F0P | abs_error | 11518 | 0.115 | 0.124 | +0.0066 [+0.0038, +0.0096] (game-clustered [+0.0023, +0.0113]) | b better |
| n_dl | F0 | F0P | abs_error | 11518 | 0.117 | 0.124 | +0.0046 [+0.0020, +0.0070] (game-clustered [+0.0005, +0.0087]) | b better |
| n_dl | F0P | F0PI | abs_error | 11518 | 0.124 | 0.124 | -0.0019 [-0.0041, +0.0005] (game-clustered [-0.0057, +0.0023]) | n.s. |
| def_y_std | F0 | F0I | abs_error | 11518 | 0.443 | 0.443 | -0.0002 [-0.0031, +0.0031] (game-clustered [-0.0049, +0.0047]) | n.s. |
| def_y_std | F0I | F0P | abs_error | 11518 | 0.443 | 0.538 | +0.0734 [+0.0667, +0.0800] (game-clustered [+0.0653, +0.0817]) | b better |
| def_y_std | F0 | F0P | abs_error | 11518 | 0.443 | 0.538 | +0.0733 [+0.0667, +0.0803] (game-clustered [+0.0637, +0.0818]) | b better |
| def_y_std | F0P | F0PI | abs_error | 11518 | 0.538 | 0.537 | -0.0015 [-0.0041, +0.0010] (game-clustered [-0.0043, +0.0014]) | n.s. |
| cb_cushion | F0 | F0I | abs_error | 11518 | 0.239 | 0.230 | -0.0032 [-0.0075, +0.0012] (game-clustered [-0.0101, +0.0043]) | n.s. |
| cb_cushion | F0I | F0P | abs_error | 11518 | 0.230 | 0.287 | +0.0539 [+0.0466, +0.0617] (game-clustered [+0.0428, +0.0652]) | b better |
| cb_cushion | F0 | F0P | abs_error | 11518 | 0.239 | 0.287 | +0.0508 [+0.0439, +0.0580] (game-clustered [+0.0411, +0.0608]) | b better |
| cb_cushion | F0P | F0PI | abs_error | 11518 | 0.287 | 0.277 | -0.0075 [-0.0117, -0.0028] (game-clustered [-0.0157, +0.0001]) | n.s. |
| n_wide_left | F0 | F0I | abs_error | 11518 | 0.159 | 0.163 | +0.0017 [+0.0003, +0.0031] (game-clustered [-0.0003, +0.0035]) | n.s. |
| n_wide_left | F0I | F0P | abs_error | 11518 | 0.163 | 0.209 | +0.0168 [+0.0138, +0.0196] (game-clustered [+0.0131, +0.0207]) | b better |
| n_wide_left | F0 | F0P | abs_error | 11518 | 0.159 | 0.209 | +0.0184 [+0.0158, +0.0212] (game-clustered [+0.0147, +0.0222]) | b better |
| n_wide_left | F0P | F0PI | abs_error | 11518 | 0.209 | 0.215 | +0.0028 [+0.0016, +0.0040] (game-clustered [+0.0013, +0.0042]) | b better |
| n_wide_right | F0 | F0I | abs_error | 11518 | 0.179 | 0.183 | +0.0019 [+0.0004, +0.0034] (game-clustered [+0.0001, +0.0038]) | b better |
| n_wide_right | F0I | F0P | abs_error | 11518 | 0.183 | 0.240 | +0.0344 [+0.0308, +0.0375] (game-clustered [+0.0303, +0.0386]) | b better |
| n_wide_right | F0 | F0P | abs_error | 11518 | 0.179 | 0.240 | +0.0363 [+0.0328, +0.0395] (game-clustered [+0.0324, +0.0404]) | b better |
| n_wide_right | F0P | F0PI | abs_error | 11518 | 0.240 | 0.240 | -0.0002 [-0.0013, +0.0009] (game-clustered [-0.0015, +0.0011]) | n.s. |
| n_backfield | F0 | F0I | abs_error | 11518 | 0.144 | 0.148 | +0.0001 [-0.0012, +0.0012] (game-clustered [-0.0025, +0.0027]) | n.s. |
| n_backfield | F0I | F0P | abs_error | 11518 | 0.148 | 0.487 | +0.1128 [+0.1073, +0.1184] (game-clustered [+0.1016, +0.1228]) | b better |
| n_backfield | F0 | F0P | abs_error | 11518 | 0.144 | 0.487 | +0.1129 [+0.1074, +0.1187] (game-clustered [+0.1016, +0.1237]) | b better |
| n_backfield | F0P | F0PI | abs_error | 11518 | 0.487 | 0.486 | -0.0014 [-0.0023, -0.0004] (game-clustered [-0.0046, +0.0019]) | n.s. |
| qb_depth | F0 | F0I | abs_error | 11471 | 0.868 | 0.874 | +0.0108 [+0.0086, +0.0130] (game-clustered [+0.0026, +0.0208]) | b better |
| qb_depth | F0I | F0P | abs_error | 11471 | 0.874 | 0.883 | +0.0089 [+0.0055, +0.0123] (game-clustered [+0.0022, +0.0147]) | b better |
| qb_depth | F0 | F0P | abs_error | 11471 | 0.868 | 0.883 | +0.0197 [+0.0163, +0.0231] (game-clustered [+0.0113, +0.0289]) | b better |
| qb_depth | F0P | F0PI | abs_error | 11471 | 0.883 | 0.885 | +0.0041 [+0.0025, +0.0058] (game-clustered [-0.0006, +0.0096]) | n.s. |
| shotgun_derived | F0 | F0I | log_loss | 11518 | 0.939 | 0.939 | -0.0006 [-0.0020, +0.0008] (game-clustered [-0.0035, +0.0017]) | n.s. |
| shotgun_derived | F0I | F0P | log_loss | 11518 | 0.939 | 0.957 | +0.0219 [+0.0169, +0.0270] (game-clustered [+0.0140, +0.0312]) | b better |
| shotgun_derived | F0 | F0P | log_loss | 11518 | 0.939 | 0.957 | +0.0213 [+0.0168, +0.0262] (game-clustered [+0.0144, +0.0290]) | b better |
| shotgun_derived | F0P | F0PI | log_loss | 11518 | 0.957 | 0.956 | -0.0024 [-0.0037, -0.0011] (game-clustered [-0.0051, -0.0004]) | a better |
| motion_derived | F0 | F0I | log_loss | 11518 | 0.067 | 0.068 | +0.0002 [-0.0019, +0.0022] (game-clustered [-0.0022, +0.0026]) | n.s. |
| motion_derived | F0I | F0P | log_loss | 11518 | 0.068 | 0.067 | -0.0006 [-0.0025, +0.0015] (game-clustered [-0.0030, +0.0020]) | n.s. |
| motion_derived | F0 | F0P | log_loss | 11518 | 0.067 | 0.067 | -0.0004 [-0.0018, +0.0010] (game-clustered [-0.0022, +0.0013]) | n.s. |
| motion_derived | F0P | F0PI | log_loss | 11518 | 0.067 | 0.068 | +0.0007 [-0.0013, +0.0024] (game-clustered [-0.0017, +0.0030]) | n.s. |

## Imputed grouping on the tracked plays

| n_plays | share_with_probabilities | share_with_label | argmax_accuracy | majority_share | share_charted_personnel_present |
|---|---|---|---|---|---|
| 11518 | 0.987 | 0.987 | 0.634 | 0.584 | 1.000 |

## Caveats

* The `p_off_*` probabilities are NFL 02's S2 model (situation + shotgun / no-huddle + team tendencies from
  strictly earlier games): a calibrated '11 personnel unless the team and situation say otherwise' prior with
  accuracy 0.647 on 2022; the tracked 2017 games are inside its 2016-2020 out-of-fold training window.
* Only the 12 pre-snap targets are fitted (the within-play students never had a personnel-dependent gain in
  NFL 03); the payoff stage (NFL 04) already tested the same probabilities as direct outcome-model features
  (`PBP+IMP_PERS`, stage d) with a null result, so this stage measures the alignment step of the chain only.
* The stage does not re-run the NFL 04 payoff with `F0I` students; `PBP+IMP_F0` (no personnel at all) is the
  strictly event-only payoff variant and `PBP+IMP` (F0P students) reads charted personnel.

Timing: 67 s on 2 threads.
