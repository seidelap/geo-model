# Soccer 02 - imputation: event-only students of the 360 defensive state

Source: `processed_dir('soccer')/events360.parquet` (1,357,506 rows, 417 matches with usable 360; see `soccer_01_build.md`). Students are LightGBM models (`num_threads=2`) that predict a 360-derived target from `f_*` event-only features; the neural student is a scikit-learn `MLPRegressor` because PyTorch is not installed. Everything is scored out of sample by match; every number states its n.

## Protocol

- Splits: (i) 5-fold `group_kfold` by `match_id` over all 417 matches (out-of-fold prediction for every row); (ii) cross-competition, men -> women (`m2f`) and women -> men (`f2m`); (iii) forward in time, every competition-season up to 2023/24 -> UEFA Euro 2024 + UEFA Women's Euro 2025 (`fwd`).
- Training rows are thinned uniformly at random to 300,000 per fit (400,000 for the final students); every match keeps the same expected share. Rounds are chosen by early stopping (patience 25, max 400) on an inner 15% match holdout of the training rows, never on the test rows. LightGBM: lr 0.1, 63 leaves, min_data_in_leaf 100, feature / bagging fraction 0.8 / 0.8, max_bin 63, seed 0; L2 objective for continuous and count targets, binary log-loss for the two binaries.
- Team-shape targets (`shape` family) are trained and scored on possession-team events with a reliable frame (`y_reliable == 1`: orientation ok or repaired and >= 7 opponents visible); ball-relative and pass targets on every usable frame (`y_frame_ok == 1`), with the reliability sensitivity below. `deep_block` = `def_line <= 18.59` yd (25th percentile over reliable settled possession events: Regular Play, possession >= 10 s old); `counter_on` = `n_opp_ahead_of_ball <= 3` on possession-team events with the ball in the middle third.
- Baselines: `base_global` (training mean / base rate), `base_type` (training mean per event type), `loc` (LightGBM on x, y, distance / opening angle / bearing to goal). Skill = R2 for continuous and count targets, Brier skill score vs `base_global` for binaries. Paired deltas use a match-clustered bootstrap (500 resamples, 95% CI).
- Every experiment is reported twice: pre-instant features only (`E0`, `E1`, `E2`) and with the 16 POST-INSTANT `f_after_*` columns (`E2a`). `E2r` is E2 trained on reliable rows only.

### Targets

| target | kind | family | subset | label_rule | n_subset | n_labelled | mean | sd | definition |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | reg | shape | poss | reliable | 1129198 | 850149 | 43.826 | 20.755 | 120 - mean x of the visible outfield opponents (yd from their goal line) |
| def_line | reg | shape | poss | reliable | 1129198 | 850149 | 33.439 | 18.339 | 120 - max x of the visible outfield opponents (deepest defender, yd from their goal line) |
| block_width | reg | shape | poss | reliable | 1129198 | 850149 | 38.037 | 8.515 | y range of the visible outfield opponents (yd) |
| block_length | reg | shape | poss | reliable | 1129198 | 850149 | 24.168 | 7.099 | x range of the visible outfield opponents (yd) |
| n_opp_ahead_of_ball | count | shape | poss | reliable | 1129198 | 850149 | 6.639 | 2.553 | visible opponents (incl. keeper) with x > ball x |
| deep_block | binary | shape | settled | reliable | 299645 | 235225 | 0.250 | 0.433 | def_line <= its 25th percentile over settled possession events (Regular Play, possession >= 10 s old); threshold stored in the bundle |
| counter_on | binary | shape | mid | reliable | 554426 | 456199 | 0.083 | 0.276 | n_opp_ahead_of_ball <= 3 with the ball in the middle third (40 <= x < 80) |
| n_opp_within_5 | count | ball | all | frame_ok | 1357506 | 1337830 | 0.720 | 0.805 | opponents (incl. keeper) within 5 yd of the ball |
| n_opp_within_10 | count | ball | all | frame_ok | 1357506 | 1337830 | 1.715 | 1.391 | opponents within 10 yd of the ball |
| nearest_opp_dist | reg | ball | all | frame_ok | 1357506 | 1336461 | 6.016 | 5.148 | distance from the ball to the nearest visible opponent (yd) |
| n_opp_in_cone | count | ball | poss | frame_ok | 1129198 | 1113443 | 0.305 | 0.554 | outfield opponents inside the triangle ball -> posts (possession-team events) |
| nearest_opp_dist_in_cone | reg | ball | poss | frame_ok | 1129198 | 293797 | 16.847 | 9.875 | nearest outfield opponent inside the cone (yd); possession-team events, defined only when the cone is not empty |
| opp_keeper_dist_to_goal_line | reg | ball | poss_att | keeper | 599391 | 233791 | 3.877 | 2.877 | 120 - opponent keeper x (yd); possession-team events in the attacking half (x >= 60), defined only when the keeper is visible and consistently flagged (y_keeper_consistent) |
| n_opp_within_3_of_end | count | pass | pass | frame_ok | 373639 | 373481 | 0.086 | 0.324 | Pass only: opponents within 3 yd of the pass end location |
| n_opp_in_lane | count | pass | pass | frame_ok | 373639 | 373481 | 0.494 | 0.733 | Pass only: opponents within 2 yd of the start -> end segment |

### Feature sets

| feature_set | n_features | description |
|---|---|---|
| loc | 5 | location only: x, y, distance / opening angle / bearing to goal (LightGBM baseline) |
| E0 | 27 | current event only: type, location geometry, play pattern, actor position, under-pressure / counterpress flags, minute / period, home, possession-team flag, score state, pre-instant pass / shot attributes, gender, competition type (league / tournament) |
| E1 | 37 | E0 + possession context: elapsed time, events and passes so far, ball distance, start type / location, time since final-third entry, time since previous event |
| E2 | 62 | E1 + 10-event window counts by type group, opponent defensive actions (mean x and count in 60 s, seconds since the last one), opponent possession in the last 10 s, ball speed / displacement |
| E2a | 78 | E2 + the 16 POST-INSTANT f_after_* columns (realised pass end / length / height / flags, carry end / length, duration) |
| E2r | 62 | E2 features, trained on reliable rows only (>= 7 opponents visible) |
| E3 | 162 | E2 + the raw 20-slot sequence block (type id, x, y, dt, same-team per slot) - LightGBM reference for the MLP student |

## What is recoverable (5-fold CV, ranked by E2 skill)

Skill is R2 (continuous / count) or Brier skill score vs the base rate (binary). `E2_m2f` / `E2_f2m` / `E2_fwd` are the E2 student trained on the other gender / on the earlier competitions and scored on the held-out domain; `in_domain_*` is the CV out-of-fold E2 prediction scored on the same rows. Verdict: recoverable (E2 skill >= 0.5), partly (0.2-0.5), weak (< 0.2).

| target | kind | n | base_type | loc | E0 | E1 | E2 | E2a | E2r | gain_E2_over_loc | E2_m2f | E2_f2m | E2_fwd | in_domain_fwd | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_depth | reg | 850149 | 0.032 | 0.861 | 0.941 | 0.958 | 0.964 | 0.965 |  | 0.103 | 0.967 | 0.952 | 0.966 | 0.967 | recoverable |
| def_line | reg | 850149 | 0.031 | 0.845 | 0.929 | 0.946 | 0.952 | 0.952 |  | 0.107 | 0.954 | 0.933 | 0.955 | 0.956 | recoverable |
| deep_block | binary | 235225 | 0.050 | 0.645 | 0.700 | 0.757 | 0.772 | 0.774 |  | 0.127 | 0.783 | 0.749 | 0.789 | 0.788 | recoverable |
| n_opp_ahead_of_ball | count | 850149 | 0.054 | 0.200 | 0.527 | 0.633 | 0.671 | 0.674 |  | 0.471 | 0.589 | 0.551 | 0.671 | 0.683 | recoverable |
| nearest_opp_dist | reg | 1336461 | 0.163 | 0.123 | 0.597 | 0.639 | 0.668 | 0.698 | 0.656 | 0.545 | 0.680 | 0.629 | 0.681 | 0.684 | recoverable |
| n_opp_within_10 | count | 1337830 | 0.080 | 0.237 | 0.514 | 0.584 | 0.621 | 0.630 | 0.615 | 0.384 | 0.622 | 0.587 | 0.634 | 0.636 | recoverable |
| n_opp_within_5 | count | 1337830 | 0.142 | 0.145 | 0.450 | 0.490 | 0.523 | 0.537 | 0.520 | 0.378 | 0.518 | 0.493 | 0.545 | 0.545 | recoverable |
| opp_keeper_dist_to_goal_line | reg | 233791 | 0.010 | 0.322 | 0.399 | 0.461 | 0.476 | 0.484 | 0.470 | 0.154 | 0.429 | 0.434 | 0.499 | 0.520 | partly |
| block_width | reg | 850149 | 0.039 | 0.286 | 0.372 | 0.432 | 0.464 | 0.466 |  | 0.178 | 0.227 | 0.258 | 0.471 | 0.473 | partly |
| block_length | reg | 850149 | 0.019 | 0.320 | 0.390 | 0.424 | 0.436 | 0.438 |  | 0.116 | 0.330 | 0.315 | 0.447 | 0.449 | partly |
| nearest_opp_dist_in_cone | reg | 293797 | 0.040 | 0.244 | 0.380 | 0.405 | 0.415 | 0.419 | 0.387 | 0.171 | 0.314 | 0.325 | 0.420 | 0.423 | partly |
| counter_on | binary | 456199 | 0.037 | 0.005 | 0.233 | 0.309 | 0.341 | 0.345 |  | 0.336 | 0.312 | 0.327 | 0.366 | 0.369 | partly |
| n_opp_in_lane | count | 373481 | -0.000 | 0.125 | 0.247 | 0.263 | 0.292 | 0.484 | 0.283 | 0.166 | 0.277 | 0.247 | 0.305 | 0.308 | partly |
| n_opp_in_cone | count | 1113443 | 0.008 | 0.050 | 0.127 | 0.152 | 0.160 | 0.160 | 0.137 | 0.110 | 0.168 | 0.128 | 0.166 | 0.168 | weak |
| n_opp_within_3_of_end | count | 373481 | -0.000 | 0.124 | 0.144 | 0.148 | 0.150 | 0.436 | 0.147 | 0.026 | 0.148 | 0.129 | 0.179 | 0.179 | weak |

## 5-fold CV: skill by feature set

| target | kind | n | base_global | base_type | loc | E0 | E1 | E2 | E2a | E2r |
|---|---|---|---|---|---|---|---|---|---|---|
| block_depth | reg | 850149 | -0.000 | 0.032 | 0.861 | 0.941 | 0.958 | 0.964 | 0.965 |  |
| block_length | reg | 850149 | -0.000 | 0.019 | 0.320 | 0.390 | 0.424 | 0.436 | 0.438 |  |
| block_width | reg | 850149 | -0.001 | 0.039 | 0.286 | 0.372 | 0.432 | 0.464 | 0.466 |  |
| counter_on | binary | 456199 | 0.000 | 0.037 | 0.005 | 0.233 | 0.309 | 0.341 | 0.345 |  |
| deep_block | binary | 235225 | 0.000 | 0.050 | 0.645 | 0.700 | 0.757 | 0.772 | 0.774 |  |
| def_line | reg | 850149 | -0.001 | 0.031 | 0.845 | 0.929 | 0.946 | 0.952 | 0.952 |  |
| n_opp_ahead_of_ball | count | 850149 | -0.000 | 0.054 | 0.200 | 0.527 | 0.633 | 0.671 | 0.674 |  |
| n_opp_in_cone | count | 1113443 | -0.000 | 0.008 | 0.050 | 0.127 | 0.152 | 0.160 | 0.160 | 0.137 |
| n_opp_in_lane | count | 373481 | -0.000 | -0.000 | 0.125 | 0.247 | 0.263 | 0.292 | 0.484 | 0.283 |
| n_opp_within_10 | count | 1337830 | -0.000 | 0.080 | 0.237 | 0.514 | 0.584 | 0.621 | 0.630 | 0.615 |
| n_opp_within_3_of_end | count | 373481 | -0.000 | -0.000 | 0.124 | 0.144 | 0.148 | 0.150 | 0.436 | 0.147 |
| n_opp_within_5 | count | 1337830 | -0.000 | 0.142 | 0.145 | 0.450 | 0.490 | 0.523 | 0.537 | 0.520 |
| nearest_opp_dist | reg | 1336461 | -0.000 | 0.163 | 0.123 | 0.597 | 0.639 | 0.668 | 0.698 | 0.656 |
| nearest_opp_dist_in_cone | reg | 293797 | -0.000 | 0.040 | 0.244 | 0.380 | 0.405 | 0.415 | 0.419 | 0.387 |
| opp_keeper_dist_to_goal_line | reg | 233791 | -0.000 | 0.010 | 0.322 | 0.399 | 0.461 | 0.476 | 0.484 | 0.470 |

### MAE (continuous and count targets)

| target | n | base_global | base_type | loc | E0 | E1 | E2 | E2a | E2r |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | 850149 | 16.940 | 16.592 | 6.058 | 3.893 | 3.264 | 3.006 | 2.980 |  |
| block_length | 850149 | 5.598 | 5.541 | 4.639 | 4.387 | 4.259 | 4.217 | 4.207 |  |
| block_width | 850149 | 6.694 | 6.558 | 5.582 | 5.240 | 5.001 | 4.871 | 4.859 |  |
| def_line | 850149 | 14.836 | 14.554 | 5.500 | 3.703 | 3.222 | 3.023 | 3.006 |  |
| n_opp_ahead_of_ball | 850149 | 2.120 | 2.038 | 1.868 | 1.398 | 1.231 | 1.167 | 1.162 |  |
| n_opp_in_cone | 1113443 | 0.449 | 0.446 | 0.427 | 0.399 | 0.391 | 0.388 | 0.387 | 0.411 |
| n_opp_in_lane | 373481 | 0.608 | 0.608 | 0.559 | 0.470 | 0.461 | 0.441 | 0.359 | 0.453 |
| n_opp_within_10 | 1337830 | 1.105 | 1.057 | 0.987 | 0.751 | 0.692 | 0.660 | 0.649 | 0.671 |
| n_opp_within_3_of_end | 373481 | 0.159 | 0.159 | 0.143 | 0.140 | 0.138 | 0.138 | 0.097 | 0.144 |
| n_opp_within_5 | 1337830 | 0.657 | 0.593 | 0.622 | 0.430 | 0.411 | 0.398 | 0.389 | 0.404 |
| nearest_opp_dist | 1336461 | 3.964 | 3.410 | 3.721 | 2.302 | 2.176 | 2.073 | 1.975 | 2.086 |
| nearest_opp_dist_in_cone | 293797 | 8.039 | 7.830 | 6.864 | 6.256 | 6.118 | 6.064 | 6.052 | 6.160 |
| opp_keeper_dist_to_goal_line | 233791 | 2.153 | 2.131 | 1.715 | 1.628 | 1.559 | 1.540 | 1.529 | 1.541 |

### Count targets: share of rows whose rounded prediction is within +/-1

| target | n | base_global | base_type | loc | E0 | E1 | E2 | E2a | E2r |
|---|---|---|---|---|---|---|---|---|---|
| n_opp_ahead_of_ball | 850149 | 0.405 | 0.415 | 0.448 | 0.614 | 0.675 | 0.700 | 0.702 |  |
| n_opp_in_cone | 1113443 | 0.964 | 0.964 | 0.970 | 0.978 | 0.979 | 0.980 | 0.980 | 0.983 |
| n_opp_in_lane | 373481 | 0.915 | 0.915 | 0.947 | 0.963 | 0.965 | 0.967 | 0.983 | 0.968 |
| n_opp_within_10 | 1337830 | 0.707 | 0.749 | 0.793 | 0.884 | 0.907 | 0.919 | 0.922 | 0.917 |
| n_opp_within_3_of_end | 373481 | 0.992 | 0.992 | 0.994 | 0.994 | 0.994 | 0.994 | 0.998 | 0.994 |
| n_opp_within_5 | 1337830 | 0.973 | 0.959 | 0.972 | 0.974 | 0.978 | 0.981 | 0.982 | 0.981 |

### Binary targets

| target | feature_set | n | base_rate | mean_pred | auc | log_loss | brier | bss | ece |
|---|---|---|---|---|---|---|---|---|---|
| counter_on | base_global | 456199 | 0.0832 | 0.0832 | 0.4848 | 0.2865 | 0.0763 | 0.0000 | 0.0087 |
| counter_on | base_type | 456199 | 0.0832 | 0.0832 | 0.6464 | 0.2719 | 0.0734 | 0.0369 | 0.0033 |
| counter_on | loc | 456199 | 0.0832 | 0.0837 | 0.5696 | 0.2840 | 0.0759 | 0.0051 | 0.0024 |
| counter_on | E0 | 456199 | 0.0832 | 0.0836 | 0.8850 | 0.1984 | 0.0585 | 0.2330 | 0.0023 |
| counter_on | E1 | 456199 | 0.0832 | 0.0827 | 0.9140 | 0.1772 | 0.0527 | 0.3091 | 0.0028 |
| counter_on | E2 | 456199 | 0.0832 | 0.0817 | 0.9267 | 0.1672 | 0.0503 | 0.3408 | 0.0047 |
| counter_on | E2a | 456199 | 0.0832 | 0.0815 | 0.9282 | 0.1660 | 0.0500 | 0.3448 | 0.0049 |
| deep_block | base_global | 235225 | 0.2500 | 0.2501 | 0.4821 | 0.5626 | 0.1876 | 0.0000 | 0.0156 |
| deep_block | base_type | 235225 | 0.2500 | 0.2499 | 0.5517 | 0.5405 | 0.1782 | 0.0499 | 0.0100 |
| deep_block | loc | 235225 | 0.2500 | 0.2502 | 0.9617 | 0.2100 | 0.0666 | 0.6451 | 0.0016 |
| deep_block | E0 | 235225 | 0.2500 | 0.2491 | 0.9730 | 0.1788 | 0.0564 | 0.6996 | 0.0023 |
| deep_block | E1 | 235225 | 0.2500 | 0.2492 | 0.9819 | 0.1478 | 0.0456 | 0.7571 | 0.0027 |
| deep_block | E2 | 235225 | 0.2500 | 0.2491 | 0.9842 | 0.1384 | 0.0428 | 0.7721 | 0.0027 |
| deep_block | E2a | 235225 | 0.2500 | 0.2491 | 0.9845 | 0.1373 | 0.0425 | 0.7737 | 0.0029 |

### Paired deltas (match-clustered bootstrap)

`delta_loss` = mean per-row loss of `from` minus `to` (positive = `to` is better); squared error for continuous / count targets, log-loss for binaries.

| target | from | to | n | loss | delta_loss | ci_low | ci_high | significant | skill_from | skill_to |
|---|---|---|---|---|---|---|---|---|---|---|
| block_depth | base_type | loc | 850149 | squared_error | 356.8893 | 351.0983 | 363.2144 | True | 0.0324 | 0.8609 |
| block_depth | loc | E0 | 850149 | squared_error | 34.4039 | 33.6355 | 35.3317 | True | 0.8609 | 0.9407 |
| block_depth | E0 | E1 | 850149 | squared_error | 7.3448 | 7.1639 | 7.5375 | True | 0.9407 | 0.9578 |
| block_depth | E1 | E2 | 850149 | squared_error | 2.8091 | 2.7045 | 2.9109 | True | 0.9578 | 0.9643 |
| block_depth | E2 | E2a | 850149 | squared_error | 0.2620 | 0.2349 | 0.2924 | True | 0.9643 | 0.9649 |
| block_depth | base_type | E2 | 850149 | squared_error | 401.4472 | 395.6080 | 407.3832 | True | 0.0324 | 0.9643 |
| block_length | base_type | loc | 850149 | squared_error | 15.1545 | 14.5491 | 15.7435 | True | 0.0193 | 0.3200 |
| block_length | loc | E0 | 850149 | squared_error | 3.5109 | 3.3147 | 3.7421 | True | 0.3200 | 0.3897 |
| block_length | E0 | E1 | 850149 | squared_error | 1.7519 | 1.6531 | 1.8485 | True | 0.3897 | 0.4244 |
| block_length | E1 | E2 | 850149 | squared_error | 0.5818 | 0.5265 | 0.6386 | True | 0.4244 | 0.4360 |
| block_length | E2 | E2a | 850149 | squared_error | 0.1174 | 0.0876 | 0.1462 | True | 0.4360 | 0.4383 |
| block_length | base_type | E2 | 850149 | squared_error | 20.9992 | 20.2345 | 21.8082 | True | 0.0193 | 0.4360 |
| block_width | base_type | loc | 850149 | squared_error | 17.8984 | 17.2741 | 18.5080 | True | 0.0392 | 0.2860 |
| block_width | loc | E0 | 850149 | squared_error | 6.2484 | 5.7234 | 6.8258 | True | 0.2860 | 0.3722 |
| block_width | E0 | E1 | 850149 | squared_error | 4.3438 | 4.1702 | 4.5071 | True | 0.3722 | 0.4321 |
| block_width | E1 | E2 | 850149 | squared_error | 2.2808 | 2.1599 | 2.4084 | True | 0.4321 | 0.4636 |
| block_width | E2 | E2a | 850149 | squared_error | 0.1548 | 0.1149 | 0.1983 | True | 0.4636 | 0.4657 |
| block_width | base_type | E2 | 850149 | squared_error | 30.7713 | 29.9125 | 31.6584 | True | 0.0392 | 0.4636 |
| counter_on | base_type | loc | 456199 | log_loss | -0.0122 | -0.0129 | -0.0115 | True | 0.0369 | 0.0051 |
| counter_on | loc | E0 | 456199 | log_loss | 0.0856 | 0.0828 | 0.0885 | True | 0.0051 | 0.2330 |
| counter_on | E0 | E1 | 456199 | log_loss | 0.0212 | 0.0202 | 0.0223 | True | 0.2330 | 0.3091 |
| counter_on | E1 | E2 | 456199 | log_loss | 0.0100 | 0.0093 | 0.0106 | True | 0.3091 | 0.3408 |
| counter_on | E2 | E2a | 456199 | log_loss | 0.0012 | 0.0009 | 0.0016 | True | 0.3408 | 0.3448 |
| counter_on | base_type | E2 | 456199 | log_loss | 0.1047 | 0.1014 | 0.1076 | True | 0.0369 | 0.3408 |
| deep_block | base_type | loc | 235225 | log_loss | 0.3305 | 0.3243 | 0.3382 | True | 0.0499 | 0.6451 |
| deep_block | loc | E0 | 235225 | log_loss | 0.0312 | 0.0292 | 0.0334 | True | 0.6451 | 0.6996 |
| deep_block | E0 | E1 | 235225 | log_loss | 0.0310 | 0.0290 | 0.0331 | True | 0.6996 | 0.7571 |
| deep_block | E1 | E2 | 235225 | log_loss | 0.0094 | 0.0085 | 0.0102 | True | 0.7571 | 0.7721 |
| deep_block | E2 | E2a | 235225 | log_loss | 0.0011 | 0.0006 | 0.0016 | True | 0.7721 | 0.7737 |
| deep_block | base_type | E2 | 235225 | log_loss | 0.4021 | 0.3945 | 0.4109 | True | 0.0499 | 0.7721 |
| def_line | base_type | loc | 850149 | squared_error | 273.8731 | 268.9739 | 279.0721 | True | 0.0310 | 0.8453 |
| def_line | loc | E0 | 850149 | squared_error | 28.2806 | 27.6037 | 29.0468 | True | 0.8453 | 0.9294 |
| def_line | E0 | E1 | 850149 | squared_error | 5.4964 | 5.3302 | 5.6514 | True | 0.9294 | 0.9457 |
| def_line | E1 | E2 | 850149 | squared_error | 2.0777 | 1.9997 | 2.1548 | True | 0.9457 | 0.9519 |
| def_line | E2 | E2a | 850149 | squared_error | 0.1653 | 0.1380 | 0.1900 | True | 0.9519 | 0.9524 |
| def_line | base_type | E2 | 850149 | squared_error | 309.7278 | 304.9493 | 314.9064 | True | 0.0310 | 0.9519 |
| n_opp_ahead_of_ball | base_type | loc | 850149 | squared_error | 0.9523 | 0.9177 | 0.9869 | True | 0.0543 | 0.2004 |
| n_opp_ahead_of_ball | loc | E0 | 850149 | squared_error | 2.1293 | 2.0783 | 2.1896 | True | 0.2004 | 0.5269 |
| n_opp_ahead_of_ball | E0 | E1 | 850149 | squared_error | 0.6945 | 0.6760 | 0.7119 | True | 0.5269 | 0.6335 |
| n_opp_ahead_of_ball | E1 | E2 | 850149 | squared_error | 0.2449 | 0.2355 | 0.2533 | True | 0.6335 | 0.6710 |
| n_opp_ahead_of_ball | E2 | E2a | 850149 | squared_error | 0.0187 | 0.0158 | 0.0217 | True | 0.6710 | 0.6739 |
| n_opp_ahead_of_ball | base_type | E2 | 850149 | squared_error | 4.0210 | 3.9568 | 4.0922 | True | 0.0543 | 0.6710 |
| n_opp_in_cone | base_type | loc | 1113443 | squared_error | 0.0129 | 0.0122 | 0.0136 | True | 0.0084 | 0.0504 |
| n_opp_in_cone | loc | E0 | 1113443 | squared_error | 0.0234 | 0.0224 | 0.0245 | True | 0.0504 | 0.1266 |
| n_opp_in_cone | E0 | E1 | 1113443 | squared_error | 0.0078 | 0.0074 | 0.0083 | True | 0.1266 | 0.1521 |
| n_opp_in_cone | E1 | E2 | 1113443 | squared_error | 0.0025 | 0.0023 | 0.0028 | True | 0.1521 | 0.1602 |
| n_opp_in_cone | E2 | E2a | 1113443 | squared_error | -0.0002 | -0.0003 | -0.0000 | True | 0.1602 | 0.1596 |
| n_opp_in_cone | base_type | E2 | 1113443 | squared_error | 0.0466 | 0.0450 | 0.0485 | True | 0.0084 | 0.1602 |
| n_opp_in_cone | E2 | E2r | 1113443 | squared_error | -0.0071 | -0.0081 | -0.0061 | True | 0.1602 | 0.1371 |
| n_opp_in_lane | base_type | loc | 373481 | squared_error | 0.0675 | 0.0651 | 0.0702 | True | -0.0001 | 0.1253 |
| n_opp_in_lane | loc | E0 | 373481 | squared_error | 0.0652 | 0.0632 | 0.0674 | True | 0.1253 | 0.2465 |
| n_opp_in_lane | E0 | E1 | 373481 | squared_error | 0.0091 | 0.0085 | 0.0098 | True | 0.2465 | 0.2635 |
| n_opp_in_lane | E1 | E2 | 373481 | squared_error | 0.0152 | 0.0144 | 0.0161 | True | 0.2635 | 0.2918 |
| n_opp_in_lane | E2 | E2a | 373481 | squared_error | 0.1032 | 0.1003 | 0.1065 | True | 0.2918 | 0.4835 |
| n_opp_in_lane | base_type | E2 | 373481 | squared_error | 0.1570 | 0.1532 | 0.1610 | True | -0.0001 | 0.2918 |
| n_opp_in_lane | E2 | E2r | 373481 | squared_error | -0.0047 | -0.0053 | -0.0041 | True | 0.2918 | 0.2830 |
| n_opp_within_10 | base_type | loc | 1337830 | squared_error | 0.3031 | 0.2931 | 0.3148 | True | 0.0803 | 0.2370 |
| n_opp_within_10 | loc | E0 | 1337830 | squared_error | 0.5362 | 0.5260 | 0.5471 | True | 0.2370 | 0.5142 |
| n_opp_within_10 | E0 | E1 | 1337830 | squared_error | 0.1359 | 0.1325 | 0.1391 | True | 0.5142 | 0.5844 |
| n_opp_within_10 | E1 | E2 | 1337830 | squared_error | 0.0710 | 0.0691 | 0.0728 | True | 0.5844 | 0.6211 |
| n_opp_within_10 | E2 | E2a | 1337830 | squared_error | 0.0175 | 0.0166 | 0.0184 | True | 0.6211 | 0.6302 |
| n_opp_within_10 | base_type | E2 | 1337830 | squared_error | 1.0462 | 1.0279 | 1.0643 | True | 0.0803 | 0.6211 |
| n_opp_within_10 | E2 | E2r | 1337830 | squared_error | -0.0121 | -0.0135 | -0.0105 | True | 0.6211 | 0.6149 |
| n_opp_within_3_of_end | base_type | loc | 373481 | squared_error | 0.0130 | 0.0121 | 0.0141 | True | -0.0000 | 0.1242 |
| n_opp_within_3_of_end | loc | E0 | 373481 | squared_error | 0.0021 | 0.0018 | 0.0024 | True | 0.1242 | 0.1439 |
| n_opp_within_3_of_end | E0 | E1 | 373481 | squared_error | 0.0005 | 0.0003 | 0.0006 | True | 0.1439 | 0.1484 |
| n_opp_within_3_of_end | E1 | E2 | 373481 | squared_error | 0.0002 | -0.0000 | 0.0003 | False | 0.1484 | 0.1499 |
| n_opp_within_3_of_end | E2 | E2a | 373481 | squared_error | 0.0301 | 0.0289 | 0.0313 | True | 0.1499 | 0.4364 |
| n_opp_within_3_of_end | base_type | E2 | 373481 | squared_error | 0.0157 | 0.0146 | 0.0170 | True | -0.0000 | 0.1499 |
| n_opp_within_3_of_end | E2 | E2r | 373481 | squared_error | -0.0003 | -0.0005 | -0.0002 | True | 0.1499 | 0.1469 |
| n_opp_within_5 | base_type | loc | 1337830 | squared_error | 0.0025 | -0.0005 | 0.0054 | False | 0.1415 | 0.1454 |
| n_opp_within_5 | loc | E0 | 1337830 | squared_error | 0.1977 | 0.1950 | 0.2006 | True | 0.1454 | 0.4501 |
| n_opp_within_5 | E0 | E1 | 1337830 | squared_error | 0.0262 | 0.0254 | 0.0269 | True | 0.4501 | 0.4904 |
| n_opp_within_5 | E1 | E2 | 1337830 | squared_error | 0.0211 | 0.0206 | 0.0216 | True | 0.4904 | 0.5230 |
| n_opp_within_5 | E2 | E2a | 1337830 | squared_error | 0.0088 | 0.0084 | 0.0092 | True | 0.5230 | 0.5366 |
| n_opp_within_5 | base_type | E2 | 1337830 | squared_error | 0.2475 | 0.2424 | 0.2525 | True | 0.1415 | 0.5230 |
| n_opp_within_5 | E2 | E2r | 1337830 | squared_error | -0.0017 | -0.0021 | -0.0013 | True | 0.5230 | 0.5204 |
| nearest_opp_dist | base_type | loc | 1336461 | squared_error | -1.0692 | -1.1859 | -0.9470 | True | 0.1630 | 0.1226 |
| nearest_opp_dist | loc | E0 | 1336461 | squared_error | 12.5672 | 12.3468 | 12.8248 | True | 0.1226 | 0.5968 |
| nearest_opp_dist | E0 | E1 | 1336461 | squared_error | 1.1098 | 1.0821 | 1.1346 | True | 0.5968 | 0.6387 |
| nearest_opp_dist | E1 | E2 | 1336461 | squared_error | 0.7662 | 0.7412 | 0.7909 | True | 0.6387 | 0.6676 |
| nearest_opp_dist | E2 | E2a | 1336461 | squared_error | 0.8022 | 0.7772 | 0.8251 | True | 0.6676 | 0.6978 |
| nearest_opp_dist | base_type | E2 | 1336461 | squared_error | 13.3740 | 13.1387 | 13.6669 | True | 0.1630 | 0.6676 |
| nearest_opp_dist | E2 | E2r | 1336461 | squared_error | -0.3022 | -0.3337 | -0.2730 | True | 0.6676 | 0.6562 |
| nearest_opp_dist_in_cone | base_type | loc | 293797 | squared_error | 19.8572 | 18.7578 | 20.9951 | True | 0.0405 | 0.2441 |
| nearest_opp_dist_in_cone | loc | E0 | 293797 | squared_error | 13.2715 | 12.7098 | 13.8819 | True | 0.2441 | 0.3802 |
| nearest_opp_dist_in_cone | E0 | E1 | 293797 | squared_error | 2.3964 | 2.2464 | 2.5661 | True | 0.3802 | 0.4048 |
| nearest_opp_dist_in_cone | E1 | E2 | 293797 | squared_error | 0.9722 | 0.8572 | 1.0873 | True | 0.4048 | 0.4147 |
| nearest_opp_dist_in_cone | E2 | E2a | 293797 | squared_error | 0.3905 | 0.3118 | 0.4725 | True | 0.4147 | 0.4187 |
| nearest_opp_dist_in_cone | base_type | E2 | 293797 | squared_error | 36.4973 | 35.0830 | 38.1327 | True | 0.0405 | 0.4147 |
| nearest_opp_dist_in_cone | E2 | E2r | 293797 | squared_error | -2.6823 | -3.1363 | -2.1656 | True | 0.4147 | 0.3872 |
| opp_keeper_dist_to_goal_line | base_type | loc | 233791 | squared_error | 2.5788 | 2.4059 | 2.7704 | True | 0.0105 | 0.3221 |
| opp_keeper_dist_to_goal_line | loc | E0 | 233791 | squared_error | 0.6369 | 0.5867 | 0.6861 | True | 0.3221 | 0.3990 |
| opp_keeper_dist_to_goal_line | E0 | E1 | 233791 | squared_error | 0.5168 | 0.4788 | 0.5523 | True | 0.3990 | 0.4615 |
| opp_keeper_dist_to_goal_line | E1 | E2 | 233791 | squared_error | 0.1185 | 0.1033 | 0.1343 | True | 0.4615 | 0.4758 |
| opp_keeper_dist_to_goal_line | E2 | E2a | 233791 | squared_error | 0.0651 | 0.0538 | 0.0750 | True | 0.4758 | 0.4837 |
| opp_keeper_dist_to_goal_line | base_type | E2 | 233791 | squared_error | 3.8510 | 3.6219 | 4.0768 | True | 0.0105 | 0.4758 |
| opp_keeper_dist_to_goal_line | E2 | E2r | 233791 | squared_error | -0.0469 | -0.0607 | -0.0330 | True | 0.4758 | 0.4701 |

## Reliability sensitivity (ball-relative and pass targets)

Students trained on all usable frames (`E2`) or on reliable frames only (`E2r`), scored separately on reliable (>= 7 opponents visible) and unreliable evaluation rows. Unreliable frames under-count opponents by construction, so their labels are biased low.

| target | eval_rows | n | skill_base_type | mae_base_type | skill_E2 | mae_E2 | skill_E2r | mae_E2r | skill_E2a | mae_E2a |
|---|---|---|---|---|---|---|---|---|---|---|
| n_opp_in_cone | reliable | 850149 | 0.003 | 0.472 | 0.161 | 0.418 | 0.169 | 0.425 | 0.160 | 0.418 |
| n_opp_in_lane | reliable | 282132 | -0.004 | 0.631 | 0.282 | 0.470 | 0.282 | 0.474 | 0.491 | 0.381 |
| n_opp_within_10 | reliable | 995227 | 0.061 | 1.087 | 0.612 | 0.696 | 0.616 | 0.696 | 0.621 | 0.686 |
| n_opp_within_3_of_end | reliable | 282132 | -0.002 | 0.172 | 0.152 | 0.157 | 0.152 | 0.161 | 0.435 | 0.112 |
| n_opp_within_5 | reliable | 995227 | 0.125 | 0.618 | 0.514 | 0.429 | 0.516 | 0.430 | 0.527 | 0.419 |
| nearest_opp_dist | reliable | 995227 | 0.138 | 3.158 | 0.656 | 1.891 | 0.657 | 1.882 | 0.684 | 1.811 |
| nearest_opp_dist_in_cone | reliable | 258365 | 0.041 | 7.890 | 0.441 | 5.976 | 0.452 | 5.919 | 0.444 | 5.967 |
| opp_keeper_dist_to_goal_line | reliable | 224670 | 0.013 | 2.096 | 0.477 | 1.516 | 0.475 | 1.514 | 0.485 | 1.504 |
| n_opp_in_cone | unreliable | 263294 | -0.212 | 0.362 | -0.071 | 0.290 | -0.370 | 0.369 | -0.073 | 0.290 |
| n_opp_in_lane | unreliable | 91349 | -0.079 | 0.538 | 0.276 | 0.350 | 0.217 | 0.386 | 0.384 | 0.292 |
| n_opp_within_10 | unreliable | 342603 | -0.177 | 0.970 | 0.520 | 0.554 | 0.458 | 0.598 | 0.537 | 0.540 |
| n_opp_within_3_of_end | unreliable | 91349 | -0.059 | 0.117 | 0.064 | 0.079 | 0.032 | 0.094 | 0.410 | 0.051 |
| n_opp_within_5 | unreliable | 342603 | 0.123 | 0.522 | 0.514 | 0.308 | 0.487 | 0.329 | 0.528 | 0.301 |
| nearest_opp_dist | unreliable | 341234 | 0.130 | 4.145 | 0.657 | 2.604 | 0.626 | 2.680 | 0.693 | 2.452 |
| nearest_opp_dist_in_cone | unreliable | 35432 | 0.031 | 7.393 | 0.193 | 6.709 | -0.152 | 7.917 | 0.210 | 6.668 |
| opp_keeper_dist_to_goal_line | unreliable | 9121 | -0.039 | 2.998 | 0.453 | 2.144 | 0.400 | 2.207 | 0.461 | 2.127 |

## Skill by event type (E2 vs per-type mean)

Within-type R2 (or BSS) of the E2 student; the per-type baseline has zero skill within a type by construction, so this isolates what the event-only features add beyond the type. Contact events (Dispossessed, Foul Won, Dribbled Past, aerial Duels) have a trivially predictable nearest opponent (the paired actor at ~0.14 yd), which inflates their all-type skill; within those types the R2 of the shared student is negative because its residual (MAE ~0.2 yd) exceeds the type's own spread (sd ~0.2 yd).


**block_depth**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 269425 | 0.968 | 2.768 | 20.353 | 0.032 |
| Ball Receipt* | 262689 | 0.954 | 3.293 | 19.955 | 0.032 |
| Carry | 232782 | 0.963 | 2.989 | 20.121 | 0.042 |
| Miscontrol | 7267 | 0.974 | 2.792 | 22.274 | 0.011 |
| Ball Recovery | 19434 | 0.974 | 3.321 | 27.154 | 0.017 |
| Pressure | 12951 | 0.962 | 3.185 | 21.268 | 0.003 |
| Dribble | 8685 | 0.977 | 2.736 | 23.263 | -0.009 |
| Shot | 9178 | 0.790 | 2.002 | 5.628 | -0.011 |
| Duel | 8238 | 0.978 | 2.905 | 25.558 | -0.037 |
| Clearance | 371 | 0.961 | 3.587 | 22.742 | 0.205 |
| Dispossessed | 6104 | 0.978 | 2.552 | 22.232 | -0.009 |
| Interception | 1952 | 0.969 | 3.395 | 24.329 | -0.100 |
| Foul Won | 5438 | 0.964 | 3.231 | 21.961 | -0.043 |
| Block | 2342 | 0.975 | 2.821 | 23.182 | -0.024 |

**def_line**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 269425 | 0.954 | 2.877 | 17.943 | 0.018 |
| Ball Receipt* | 262689 | 0.942 | 3.195 | 17.530 | 0.024 |
| Carry | 232782 | 0.950 | 3.004 | 17.825 | 0.029 |
| Miscontrol | 7267 | 0.965 | 2.686 | 19.515 | -0.021 |
| Ball Recovery | 19434 | 0.964 | 3.565 | 24.748 | -0.023 |
| Pressure | 12951 | 0.952 | 3.106 | 18.935 | 0.010 |
| Dribble | 8685 | 0.967 | 2.773 | 20.464 | -0.064 |
| Shot | 9178 | 0.796 | 1.855 | 5.307 | -0.013 |
| Duel | 8238 | 0.969 | 3.005 | 23.130 | -0.068 |
| Clearance | 371 | 0.937 | 4.444 | 22.374 | 0.359 |
| Dispossessed | 6104 | 0.967 | 2.647 | 19.633 | -0.049 |
| Interception | 1952 | 0.951 | 4.075 | 23.829 | -0.057 |
| Foul Won | 5438 | 0.947 | 3.516 | 19.948 | -0.042 |
| Block | 2342 | 0.966 | 2.798 | 20.628 | -0.039 |

**n_opp_ahead_of_ball**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 269425 | 0.662 | 1.092 | 2.348 | -0.016 |
| Ball Receipt* | 262689 | 0.647 | 1.249 | 2.638 | -0.017 |
| Carry | 232782 | 0.670 | 1.149 | 2.501 | -0.018 |
| Miscontrol | 7267 | 0.524 | 1.183 | 2.164 | 0.004 |
| Ball Recovery | 19434 | 0.624 | 1.198 | 2.456 | -0.014 |
| Pressure | 12951 | 0.560 | 1.288 | 2.438 | 0.009 |
| Dribble | 8685 | 0.526 | 1.217 | 2.212 | -0.004 |
| Shot | 9178 | 0.753 | 1.140 | 2.899 | -0.025 |
| Duel | 8238 | 0.533 | 1.168 | 2.154 | 0.025 |
| Clearance | 371 | 0.319 | 1.191 | 1.753 | -0.106 |
| Dispossessed | 6104 | 0.527 | 1.136 | 2.068 | 0.002 |
| Interception | 1952 | 0.406 | 1.193 | 1.922 | 0.034 |
| Foul Won | 5438 | 0.422 | 1.223 | 2.009 | -0.013 |
| Block | 2342 | 0.498 | 1.301 | 2.298 | 0.005 |

**deep_block**

| f_type | n | base_rate | auc | log_loss | bss |
|---|---|---|---|---|---|
| Pass | 74478 | 0.229 | 0.984 | 0.133 | 0.767 |
| Ball Receipt* | 74639 | 0.229 | 0.982 | 0.141 | 0.754 |
| Carry | 66984 | 0.222 | 0.983 | 0.138 | 0.753 |
| Miscontrol | 1702 | 0.373 | 0.992 | 0.109 | 0.869 |
| Ball Recovery | 3644 | 0.540 | 0.979 | 0.185 | 0.829 |
| Pressure | 3539 | 0.483 | 0.978 | 0.189 | 0.811 |
| Dribble | 2293 | 0.464 | 0.984 | 0.161 | 0.829 |
| Shot | 2522 | 0.967 | 0.982 | 0.060 | 0.968 |
| Duel | 1057 | 0.391 | 0.989 | 0.129 | 0.839 |
| Dispossessed | 1581 | 0.410 | 0.988 | 0.141 | 0.836 |
| Interception | 254 | 0.291 | 0.962 | 0.222 | 0.661 |
| Foul Won | 1147 | 0.147 | 0.979 | 0.125 | 0.724 |
| Block | 598 | 0.557 | 0.981 | 0.178 | 0.844 |

**counter_on**

| f_type | n | base_rate | auc | log_loss | bss |
|---|---|---|---|---|---|
| Pass | 150821 | 0.038 | 0.938 | 0.095 | 0.305 |
| Ball Receipt* | 144060 | 0.101 | 0.915 | 0.199 | 0.338 |
| Carry | 130316 | 0.080 | 0.930 | 0.161 | 0.346 |
| Miscontrol | 2833 | 0.304 | 0.805 | 0.481 | 0.393 |
| Ball Recovery | 7949 | 0.133 | 0.874 | 0.271 | 0.297 |
| Pressure | 4602 | 0.210 | 0.839 | 0.380 | 0.342 |
| Dribble | 3183 | 0.203 | 0.785 | 0.415 | 0.256 |
| Duel | 3822 | 0.382 | 0.846 | 0.468 | 0.533 |
| Clearance | 207 | 0.014 | 0.861 | 0.069 | 0.167 |
| Dispossessed | 2386 | 0.292 | 0.762 | 0.512 | 0.332 |
| Interception | 1013 | 0.158 | 0.841 | 0.325 | 0.271 |
| Foul Won | 3312 | 0.316 | 0.788 | 0.502 | 0.384 |
| Block | 732 | 0.164 | 0.810 | 0.355 | 0.227 |

**n_opp_within_5**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 373481 | 0.428 | 0.448 | 0.763 | 0.001 |
| Ball Receipt* | 352271 | 0.521 | 0.378 | 0.786 | 0.002 |
| Carry | 322790 | 0.474 | 0.336 | 0.682 | 0.002 |
| Miscontrol | 9905 | 0.358 | 0.548 | 0.882 | 0.010 |
| Ball Recovery | 36460 | 0.361 | 0.515 | 0.831 | 0.001 |
| Pressure | 118570 | 0.213 | 0.341 | 0.595 | 0.001 |
| Dribble | 11173 | 0.210 | 0.523 | 0.720 | -0.003 |
| Shot | 10357 | 0.506 | 0.728 | 1.347 | 0.017 |
| Duel | 23600 | 0.543 | 0.404 | 0.845 | 0.002 |
| Clearance | 15061 | 0.269 | 0.538 | 0.862 | 0.001 |
| Dispossessed | 8316 | 0.279 | 0.543 | 0.789 | -0.003 |
| Interception | 8970 | 0.129 | 0.376 | 0.575 | -0.006 |
| Foul Won | 8250 | 0.156 | 0.513 | 0.683 | 0.009 |
| Block | 14753 | 0.316 | 0.434 | 0.752 | -0.001 |

**nearest_opp_dist**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 372911 | 0.607 | 2.130 | 4.975 | -0.004 |
| Ball Receipt* | 351974 | 0.610 | 2.434 | 5.237 | -0.003 |
| Carry | 322420 | 0.634 | 2.370 | 5.283 | -0.001 |
| Miscontrol | 9905 | 0.338 | 1.317 | 2.445 | -0.043 |
| Ball Recovery | 36336 | 0.496 | 2.545 | 5.221 | -0.006 |
| Pressure | 118569 | 0.184 | 1.154 | 1.691 | -0.007 |
| Dribble | 11173 | 0.311 | 0.536 | 0.888 | 0.006 |
| Shot | 10357 | 0.607 | 1.038 | 2.320 | -0.014 |
| Duel | 23600 | 0.102 | 0.531 | 0.790 | -0.008 |
| Clearance | 15061 | 0.174 | 1.448 | 2.190 | -0.007 |
| Dispossessed | 8316 | -2.687 | 0.214 | 0.168 | -0.004 |
| Interception | 8970 | 0.133 | 1.672 | 2.391 | 0.028 |
| Foul Won | 8250 | -2.370 | 0.262 | 0.227 | 0.022 |
| Block | 14753 | 0.072 | 1.139 | 1.640 | 0.001 |

**n_opp_in_cone**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 353359 | 0.124 | 0.419 | 0.563 | -0.001 |
| Ball Receipt* | 340879 | 0.123 | 0.378 | 0.528 | 0.002 |
| Carry | 309610 | 0.157 | 0.377 | 0.543 | -0.002 |
| Miscontrol | 8468 | 0.124 | 0.264 | 0.427 | 0.003 |
| Ball Recovery | 28447 | 0.355 | 0.396 | 0.710 | -0.002 |
| Pressure | 15898 | 0.246 | 0.352 | 0.579 | 0.001 |
| Dribble | 10072 | 0.133 | 0.275 | 0.436 | -0.000 |
| Shot | 10191 | 0.416 | 0.563 | 0.989 | 0.001 |
| Duel | 10750 | 0.106 | 0.245 | 0.417 | 0.009 |
| Clearance | 725 | 0.123 | 0.353 | 0.540 | -0.044 |
| Dispossessed | 6961 | 0.110 | 0.263 | 0.414 | 0.009 |
| Interception | 2962 | 0.117 | 0.297 | 0.455 | 0.001 |
| Foul Won | 6665 | 0.077 | 0.249 | 0.391 | 0.005 |
| Block | 2786 | 0.173 | 0.377 | 0.587 | 0.006 |

**n_opp_in_lane**

| f_type | n | r2 | mae | sd_y | bias |
|---|---|---|---|---|---|
| Pass | 373481 | 0.292 | 0.441 | 0.733 | 0.001 |

## Cross-competition and forward transfer

`skill_transfer`: student trained on the source domain only, scored on the target domain; `skill_in_domain`: the CV out-of-fold prediction (trained on 4/5 of all matches, i.e. including the target domain's other matches) scored on the same rows. `base_*` rows are the source-domain means. Training rows of the transfer fits are thinned to the same cap as the CV fits.


**m2f: men -> women**

| target | feature_set | n | skill_transfer | skill_in_domain | mae | mae_in_domain | auc | log_loss | log_loss_in_domain |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | base_type | 189529 | 0.029 | 0.033 | 17.318 | 17.249 |  |  |  |
| block_depth | loc | 189529 | 0.871 | 0.877 | 5.994 | 5.865 |  |  |  |
| block_depth | E0 | 189529 | 0.947 | 0.954 | 3.805 | 3.564 |  |  |  |
| block_depth | E2 | 189529 | 0.967 | 0.971 | 2.983 | 2.827 |  |  |  |
| block_depth | E2a | 189529 | 0.968 | 0.971 | 2.963 | 2.813 |  |  |  |
| block_length | base_type | 189529 | -0.038 | -0.014 | 5.389 | 5.336 |  |  |  |
| block_length | loc | 189529 | 0.278 | 0.294 | 4.527 | 4.477 |  |  |  |
| block_length | E0 | 189529 | 0.308 | 0.371 | 4.432 | 4.207 |  |  |  |
| block_length | E2 | 189529 | 0.330 | 0.401 | 4.358 | 4.102 |  |  |  |
| block_length | E2a | 189529 | 0.329 | 0.403 | 4.361 | 4.094 |  |  |  |
| block_width | base_type | 189529 | -0.169 | -0.087 | 6.882 | 6.618 |  |  |  |
| block_width | loc | 189529 | 0.032 | 0.105 | 6.245 | 5.985 |  |  |  |
| block_width | E0 | 189529 | 0.088 | 0.298 | 6.086 | 5.310 |  |  |  |
| block_width | E2 | 189529 | 0.227 | 0.397 | 5.578 | 4.932 |  |  |  |
| block_width | E2a | 189529 | 0.231 | 0.398 | 5.565 | 4.922 |  |  |  |
| counter_on | base_type | 96932 | 0.039 | 0.038 |  |  | 0.644 | 0.394 | 0.385 |
| counter_on | loc | 96932 | 0.007 | 0.008 |  |  | 0.566 | 0.410 | 0.399 |
| counter_on | E0 | 96932 | 0.242 | 0.250 |  |  | 0.854 | 0.293 | 0.283 |
| counter_on | E2 | 96932 | 0.312 | 0.326 |  |  | 0.889 | 0.265 | 0.253 |
| counter_on | E2a | 96932 | 0.313 | 0.330 |  |  | 0.890 | 0.263 | 0.251 |
| deep_block | base_type | 49027 | 0.054 | 0.053 |  |  | 0.570 | 0.583 | 0.581 |
| deep_block | loc | 49027 | 0.670 | 0.671 |  |  | 0.963 | 0.218 | 0.217 |
| deep_block | E0 | 49027 | 0.715 | 0.724 |  |  | 0.973 | 0.188 | 0.183 |
| deep_block | E2 | 49027 | 0.783 | 0.790 |  |  | 0.984 | 0.146 | 0.141 |
| deep_block | E2a | 49027 | 0.786 | 0.792 |  |  | 0.985 | 0.144 | 0.140 |
| def_line | base_type | 189529 | 0.029 | 0.032 | 15.426 | 15.354 |  |  |  |
| def_line | loc | 189529 | 0.852 | 0.859 | 5.592 | 5.474 |  |  |  |
| def_line | E0 | 189529 | 0.934 | 0.944 | 3.748 | 3.495 |  |  |  |
| def_line | E2 | 189529 | 0.954 | 0.960 | 3.083 | 2.908 |  |  |  |
| def_line | E2a | 189529 | 0.954 | 0.961 | 3.071 | 2.894 |  |  |  |
| n_opp_ahead_of_ball | base_type | 189529 | -0.147 | -0.075 | 2.060 | 2.006 |  |  |  |
| n_opp_ahead_of_ball | loc | 189529 | -0.013 | 0.056 | 1.942 | 1.880 |  |  |  |
| n_opp_ahead_of_ball | E0 | 189529 | 0.426 | 0.495 | 1.461 | 1.362 |  |  |  |
| n_opp_ahead_of_ball | E2 | 189529 | 0.589 | 0.630 | 1.234 | 1.170 |  |  |  |
| n_opp_ahead_of_ball | E2a | 189529 | 0.594 | 0.631 | 1.227 | 1.167 |  |  |  |
| n_opp_in_cone | base_type | 295447 | 0.002 | 0.008 | 0.434 | 0.427 |  |  |  |
| n_opp_in_cone | loc | 295447 | 0.047 | 0.055 | 0.409 | 0.404 |  |  |  |
| n_opp_in_cone | E0 | 295447 | 0.133 | 0.150 | 0.373 | 0.361 |  |  |  |
| n_opp_in_cone | E2 | 295447 | 0.168 | 0.187 | 0.363 | 0.351 |  |  |  |
| n_opp_in_cone | E2a | 295447 | 0.169 | 0.186 | 0.363 | 0.351 |  |  |  |
| n_opp_in_lane | base_type | 100549 | -0.033 | -0.018 | 0.636 | 0.639 |  |  |  |
| n_opp_in_lane | loc | 100549 | 0.092 | 0.111 | 0.586 | 0.589 |  |  |  |
| n_opp_in_lane | E0 | 100549 | 0.220 | 0.248 | 0.495 | 0.508 |  |  |  |
| n_opp_in_lane | E2 | 100549 | 0.277 | 0.294 | 0.465 | 0.477 |  |  |  |
| n_opp_in_lane | E2a | 100549 | 0.458 | 0.473 | 0.392 | 0.394 |  |  |  |
| n_opp_within_10 | base_type | 374247 | 0.066 | 0.071 | 1.087 | 1.088 |  |  |  |
| n_opp_within_10 | loc | 374247 | 0.222 | 0.238 | 1.016 | 1.012 |  |  |  |
| n_opp_within_10 | E0 | 374247 | 0.517 | 0.544 | 0.765 | 0.753 |  |  |  |
| n_opp_within_10 | E2 | 374247 | 0.622 | 0.641 | 0.674 | 0.664 |  |  |  |
| n_opp_within_10 | E2a | 374247 | 0.627 | 0.647 | 0.665 | 0.656 |  |  |  |
| n_opp_within_3_of_end | base_type | 100549 | -0.008 | -0.004 | 0.172 | 0.180 |  |  |  |
| n_opp_within_3_of_end | loc | 100549 | 0.121 | 0.125 | 0.157 | 0.164 |  |  |  |
| n_opp_within_3_of_end | E0 | 100549 | 0.144 | 0.151 | 0.158 | 0.172 |  |  |  |
| n_opp_within_3_of_end | E2 | 100549 | 0.148 | 0.159 | 0.161 | 0.171 |  |  |  |
| n_opp_within_3_of_end | E2a | 100549 | 0.436 | 0.456 | 0.114 | 0.118 |  |  |  |
| n_opp_within_5 | base_type | 374247 | 0.102 | 0.112 | 0.609 | 0.611 |  |  |  |
| n_opp_within_5 | loc | 374247 | 0.120 | 0.143 | 0.641 | 0.630 |  |  |  |
| n_opp_within_5 | E0 | 374247 | 0.437 | 0.462 | 0.440 | 0.442 |  |  |  |
| n_opp_within_5 | E2 | 374247 | 0.518 | 0.535 | 0.409 | 0.410 |  |  |  |
| n_opp_within_5 | E2a | 374247 | 0.526 | 0.545 | 0.403 | 0.404 |  |  |  |
| nearest_opp_dist | base_type | 373601 | 0.140 | 0.146 | 3.533 | 3.493 |  |  |  |
| nearest_opp_dist | loc | 373601 | 0.117 | 0.134 | 3.882 | 3.792 |  |  |  |
| nearest_opp_dist | E0 | 373601 | 0.612 | 0.633 | 2.282 | 2.155 |  |  |  |
| nearest_opp_dist | E2 | 373601 | 0.680 | 0.697 | 2.032 | 1.937 |  |  |  |
| nearest_opp_dist | E2a | 373601 | 0.705 | 0.721 | 1.952 | 1.856 |  |  |  |
| nearest_opp_dist_in_cone | base_type | 67634 | -0.035 | -0.004 | 7.371 | 7.253 |  |  |  |
| nearest_opp_dist_in_cone | loc | 67634 | 0.122 | 0.155 | 6.568 | 6.453 |  |  |  |
| nearest_opp_dist_in_cone | E0 | 67634 | 0.275 | 0.360 | 5.976 | 5.671 |  |  |  |
| nearest_opp_dist_in_cone | E2 | 67634 | 0.314 | 0.390 | 5.796 | 5.524 |  |  |  |
| nearest_opp_dist_in_cone | E2a | 67634 | 0.321 | 0.395 | 5.777 | 5.506 |  |  |  |
| opp_keeper_dist_to_goal_line | base_type | 53634 | -0.013 | -0.005 | 2.217 | 2.192 |  |  |  |
| opp_keeper_dist_to_goal_line | loc | 53634 | 0.283 | 0.287 | 1.776 | 1.768 |  |  |  |
| opp_keeper_dist_to_goal_line | E0 | 53634 | 0.363 | 0.376 | 1.691 | 1.673 |  |  |  |
| opp_keeper_dist_to_goal_line | E2 | 53634 | 0.429 | 0.446 | 1.625 | 1.598 |  |  |  |
| opp_keeper_dist_to_goal_line | E2a | 53634 | 0.436 | 0.453 | 1.616 | 1.589 |  |  |  |

**f2m: women -> men**

| target | feature_set | n | skill_transfer | skill_in_domain | mae | mae_in_domain | auc | log_loss | log_loss_in_domain |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | base_type | 660620 | 0.018 | 0.029 | 16.358 | 16.403 |  |  |  |
| block_depth | loc | 660620 | 0.827 | 0.855 | 6.663 | 6.113 |  |  |  |
| block_depth | E0 | 660620 | 0.921 | 0.936 | 4.426 | 3.987 |  |  |  |
| block_depth | E2 | 660620 | 0.952 | 0.962 | 3.434 | 3.058 |  |  |  |
| block_depth | E2a | 660620 | 0.953 | 0.963 | 3.394 | 3.029 |  |  |  |
| block_length | base_type | 660620 | -0.035 | 0.015 | 5.727 | 5.600 |  |  |  |
| block_length | loc | 660620 | 0.219 | 0.318 | 4.919 | 4.686 |  |  |  |
| block_length | E0 | 660620 | 0.268 | 0.386 | 4.764 | 4.438 |  |  |  |
| block_length | E2 | 660620 | 0.315 | 0.437 | 4.621 | 4.250 |  |  |  |
| block_length | E2a | 660620 | 0.316 | 0.440 | 4.616 | 4.239 |  |  |  |
| block_width | base_type | 660620 | -0.161 | 0.027 | 7.271 | 6.541 |  |  |  |
| block_width | loc | 660620 | 0.080 | 0.300 | 6.360 | 5.466 |  |  |  |
| block_width | E0 | 660620 | 0.112 | 0.362 | 6.262 | 5.220 |  |  |  |
| block_width | E2 | 660620 | 0.258 | 0.456 | 5.740 | 4.854 |  |  |  |
| block_width | E2a | 660620 | 0.265 | 0.458 | 5.715 | 4.841 |  |  |  |
| counter_on | base_type | 359267 | 0.032 | 0.036 |  |  | 0.646 | 0.260 | 0.241 |
| counter_on | loc | 359267 | 0.000 | 0.004 |  |  | 0.557 | 0.272 | 0.253 |
| counter_on | E0 | 359267 | 0.199 | 0.225 |  |  | 0.876 | 0.192 | 0.176 |
| counter_on | E2 | 359267 | 0.327 | 0.348 |  |  | 0.922 | 0.159 | 0.144 |
| counter_on | E2a | 359267 | 0.325 | 0.352 |  |  | 0.922 | 0.159 | 0.143 |
| deep_block | base_type | 186198 | 0.052 | 0.049 |  |  | 0.560 | 0.534 | 0.530 |
| deep_block | loc | 186198 | 0.634 | 0.637 |  |  | 0.960 | 0.213 | 0.208 |
| deep_block | E0 | 186198 | 0.672 | 0.692 |  |  | 0.969 | 0.192 | 0.178 |
| deep_block | E2 | 186198 | 0.749 | 0.767 |  |  | 0.981 | 0.151 | 0.138 |
| deep_block | E2a | 186198 | 0.750 | 0.768 |  |  | 0.981 | 0.150 | 0.137 |
| def_line | base_type | 660620 | 0.019 | 0.028 | 14.244 | 14.324 |  |  |  |
| def_line | loc | 660620 | 0.795 | 0.840 | 6.157 | 5.508 |  |  |  |
| def_line | E0 | 660620 | 0.900 | 0.924 | 4.272 | 3.762 |  |  |  |
| def_line | E2 | 660620 | 0.933 | 0.949 | 3.468 | 3.056 |  |  |  |
| def_line | E2a | 660620 | 0.933 | 0.950 | 3.462 | 3.038 |  |  |  |
| n_opp_ahead_of_ball | base_type | 660620 | -0.133 | 0.047 | 2.293 | 2.048 |  |  |  |
| n_opp_ahead_of_ball | loc | 660620 | 0.012 | 0.203 | 2.136 | 1.865 |  |  |  |
| n_opp_ahead_of_ball | E0 | 660620 | 0.370 | 0.515 | 1.641 | 1.408 |  |  |  |
| n_opp_ahead_of_ball | E2 | 660620 | 0.551 | 0.668 | 1.386 | 1.166 |  |  |  |
| n_opp_ahead_of_ball | E2a | 660620 | 0.554 | 0.671 | 1.381 | 1.161 |  |  |  |
| n_opp_in_cone | base_type | 817996 | -0.005 | 0.006 | 0.434 | 0.453 |  |  |  |
| n_opp_in_cone | loc | 817996 | 0.032 | 0.046 | 0.421 | 0.435 |  |  |  |
| n_opp_in_cone | E0 | 817996 | 0.094 | 0.117 | 0.400 | 0.412 |  |  |  |
| n_opp_in_cone | E2 | 817996 | 0.128 | 0.149 | 0.389 | 0.401 |  |  |  |
| n_opp_in_cone | E2a | 817996 | 0.128 | 0.149 | 0.390 | 0.400 |  |  |  |
| n_opp_in_lane | base_type | 272932 | -0.040 | -0.003 | 0.626 | 0.597 |  |  |  |
| n_opp_in_lane | loc | 272932 | 0.076 | 0.122 | 0.581 | 0.549 |  |  |  |
| n_opp_in_lane | E0 | 272932 | 0.194 | 0.238 | 0.497 | 0.455 |  |  |  |
| n_opp_in_lane | E2 | 272932 | 0.247 | 0.283 | 0.461 | 0.427 |  |  |  |
| n_opp_in_lane | E2a | 272932 | 0.459 | 0.483 | 0.362 | 0.347 |  |  |  |
| n_opp_within_10 | base_type | 963583 | 0.066 | 0.079 | 1.066 | 1.045 |  |  |  |
| n_opp_within_10 | loc | 963583 | 0.201 | 0.233 | 1.004 | 0.977 |  |  |  |
| n_opp_within_10 | E0 | 963583 | 0.467 | 0.499 | 0.776 | 0.751 |  |  |  |
| n_opp_within_10 | E2 | 963583 | 0.587 | 0.610 | 0.679 | 0.658 |  |  |  |
| n_opp_within_10 | E2a | 963583 | 0.599 | 0.621 | 0.668 | 0.646 |  |  |  |
| n_opp_within_3_of_end | base_type | 272932 | -0.012 | -0.001 | 0.172 | 0.151 |  |  |  |
| n_opp_within_3_of_end | loc | 272932 | 0.109 | 0.121 | 0.155 | 0.135 |  |  |  |
| n_opp_within_3_of_end | E0 | 272932 | 0.125 | 0.137 | 0.143 | 0.128 |  |  |  |
| n_opp_within_3_of_end | E2 | 272932 | 0.129 | 0.142 | 0.139 | 0.126 |  |  |  |
| n_opp_within_3_of_end | E2a | 272932 | 0.388 | 0.423 | 0.099 | 0.090 |  |  |  |
| n_opp_within_5 | base_type | 963583 | 0.122 | 0.145 | 0.606 | 0.587 |  |  |  |
| n_opp_within_5 | loc | 963583 | 0.090 | 0.137 | 0.623 | 0.618 |  |  |  |
| n_opp_within_5 | E0 | 963583 | 0.408 | 0.439 | 0.446 | 0.425 |  |  |  |
| n_opp_within_5 | E2 | 963583 | 0.493 | 0.513 | 0.408 | 0.393 |  |  |  |
| n_opp_within_5 | E2a | 963583 | 0.506 | 0.528 | 0.400 | 0.383 |  |  |  |
| nearest_opp_dist | base_type | 962860 | 0.152 | 0.164 | 3.346 | 3.378 |  |  |  |
| nearest_opp_dist | loc | 962860 | 0.078 | 0.112 | 3.669 | 3.694 |  |  |  |
| nearest_opp_dist | E0 | 962860 | 0.555 | 0.580 | 2.385 | 2.359 |  |  |  |
| nearest_opp_dist | E2 | 962860 | 0.629 | 0.654 | 2.167 | 2.126 |  |  |  |
| nearest_opp_dist | E2a | 962860 | 0.661 | 0.686 | 2.067 | 2.021 |  |  |  |
| nearest_opp_dist_in_cone | base_type | 226163 | -0.028 | 0.033 | 8.149 | 8.002 |  |  |  |
| nearest_opp_dist_in_cone | loc | 226163 | 0.165 | 0.251 | 7.298 | 6.987 |  |  |  |
| nearest_opp_dist_in_cone | E0 | 226163 | 0.288 | 0.374 | 6.786 | 6.431 |  |  |  |
| nearest_opp_dist_in_cone | E2 | 226163 | 0.325 | 0.410 | 6.600 | 6.226 |  |  |  |
| nearest_opp_dist_in_cone | E2a | 226163 | 0.327 | 0.414 | 6.591 | 6.215 |  |  |  |
| opp_keeper_dist_to_goal_line | base_type | 180157 | -0.015 | 0.010 | 2.107 | 2.113 |  |  |  |
| opp_keeper_dist_to_goal_line | loc | 180157 | 0.313 | 0.329 | 1.727 | 1.699 |  |  |  |
| opp_keeper_dist_to_goal_line | E0 | 180157 | 0.362 | 0.403 | 1.674 | 1.615 |  |  |  |
| opp_keeper_dist_to_goal_line | E2 | 180157 | 0.434 | 0.482 | 1.595 | 1.523 |  |  |  |
| opp_keeper_dist_to_goal_line | E2a | 180157 | 0.436 | 0.491 | 1.591 | 1.511 |  |  |  |

**fwd: <= 2023/24 -> Euro 2024 + Women's Euro 2025**

| target | feature_set | n | skill_transfer | skill_in_domain | mae | mae_in_domain | auc | log_loss | log_loss_in_domain |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | base_type | 162468 | 0.035 | 0.035 | 16.828 | 16.817 |  |  |  |
| block_depth | loc | 162468 | 0.859 | 0.859 | 6.217 | 6.207 |  |  |  |
| block_depth | E0 | 162468 | 0.944 | 0.944 | 3.858 | 3.824 |  |  |  |
| block_depth | E2 | 162468 | 0.966 | 0.967 | 2.961 | 2.942 |  |  |  |
| block_depth | E2a | 162468 | 0.967 | 0.968 | 2.940 | 2.912 |  |  |  |
| block_length | base_type | 162468 | 0.020 | 0.020 | 5.393 | 5.394 |  |  |  |
| block_length | loc | 162468 | 0.321 | 0.321 | 4.508 | 4.510 |  |  |  |
| block_length | E0 | 162468 | 0.400 | 0.403 | 4.238 | 4.225 |  |  |  |
| block_length | E2 | 162468 | 0.447 | 0.449 | 4.074 | 4.060 |  |  |  |
| block_length | E2a | 162468 | 0.450 | 0.452 | 4.058 | 4.049 |  |  |  |
| block_width | base_type | 162468 | 0.043 | 0.043 | 6.509 | 6.511 |  |  |  |
| block_width | loc | 162468 | 0.272 | 0.272 | 5.607 | 5.611 |  |  |  |
| block_width | E0 | 162468 | 0.379 | 0.382 | 5.164 | 5.152 |  |  |  |
| block_width | E2 | 162468 | 0.471 | 0.473 | 4.794 | 4.787 |  |  |  |
| block_width | E2a | 162468 | 0.471 | 0.475 | 4.793 | 4.773 |  |  |  |
| counter_on | base_type | 87231 | 0.040 | 0.040 |  |  | 0.655 | 0.270 | 0.270 |
| counter_on | loc | 87231 | 0.005 | 0.005 |  |  | 0.571 | 0.283 | 0.283 |
| counter_on | E0 | 87231 | 0.262 | 0.265 |  |  | 0.899 | 0.190 | 0.189 |
| counter_on | E2 | 87231 | 0.366 | 0.369 |  |  | 0.934 | 0.160 | 0.159 |
| counter_on | E2a | 87231 | 0.369 | 0.374 |  |  | 0.935 | 0.159 | 0.158 |
| deep_block | base_type | 43350 | 0.049 | 0.049 |  |  | 0.561 | 0.565 | 0.564 |
| deep_block | loc | 43350 | 0.650 | 0.652 |  |  | 0.963 | 0.218 | 0.217 |
| deep_block | E0 | 43350 | 0.718 | 0.715 |  |  | 0.975 | 0.178 | 0.180 |
| deep_block | E2 | 43350 | 0.789 | 0.788 |  |  | 0.986 | 0.137 | 0.137 |
| deep_block | E2a | 43350 | 0.792 | 0.789 |  |  | 0.986 | 0.135 | 0.136 |
| def_line | base_type | 162468 | 0.034 | 0.034 | 14.748 | 14.730 |  |  |  |
| def_line | loc | 162468 | 0.844 | 0.845 | 5.612 | 5.598 |  |  |  |
| def_line | E0 | 162468 | 0.934 | 0.934 | 3.639 | 3.618 |  |  |  |
| def_line | E2 | 162468 | 0.955 | 0.956 | 2.959 | 2.941 |  |  |  |
| def_line | E2a | 162468 | 0.956 | 0.956 | 2.938 | 2.925 |  |  |  |
| n_opp_ahead_of_ball | base_type | 162468 | 0.056 | 0.056 | 2.046 | 2.044 |  |  |  |
| n_opp_ahead_of_ball | loc | 162468 | 0.198 | 0.200 | 1.879 | 1.877 |  |  |  |
| n_opp_ahead_of_ball | E0 | 162468 | 0.535 | 0.546 | 1.406 | 1.382 |  |  |  |
| n_opp_ahead_of_ball | E2 | 162468 | 0.671 | 0.683 | 1.186 | 1.159 |  |  |  |
| n_opp_ahead_of_ball | E2a | 162468 | 0.675 | 0.685 | 1.177 | 1.153 |  |  |  |
| n_opp_in_cone | base_type | 209425 | 0.007 | 0.008 | 0.456 | 0.459 |  |  |  |
| n_opp_in_cone | loc | 209425 | 0.054 | 0.055 | 0.434 | 0.437 |  |  |  |
| n_opp_in_cone | E0 | 209425 | 0.131 | 0.135 | 0.402 | 0.405 |  |  |  |
| n_opp_in_cone | E2 | 209425 | 0.166 | 0.168 | 0.391 | 0.395 |  |  |  |
| n_opp_in_cone | E2a | 209425 | 0.166 | 0.168 | 0.390 | 0.394 |  |  |  |
| n_opp_in_lane | base_type | 70295 | -0.000 | -0.000 | 0.615 | 0.615 |  |  |  |
| n_opp_in_lane | loc | 70295 | 0.138 | 0.137 | 0.563 | 0.564 |  |  |  |
| n_opp_in_lane | E0 | 70295 | 0.260 | 0.262 | 0.474 | 0.471 |  |  |  |
| n_opp_in_lane | E2 | 70295 | 0.305 | 0.308 | 0.446 | 0.442 |  |  |  |
| n_opp_in_lane | E2a | 70295 | 0.516 | 0.517 | 0.352 | 0.351 |  |  |  |
| n_opp_within_10 | base_type | 252098 | 0.090 | 0.090 | 1.056 | 1.055 |  |  |  |
| n_opp_within_10 | loc | 252098 | 0.240 | 0.240 | 0.989 | 0.988 |  |  |  |
| n_opp_within_10 | E0 | 252098 | 0.533 | 0.534 | 0.736 | 0.737 |  |  |  |
| n_opp_within_10 | E2 | 252098 | 0.634 | 0.636 | 0.650 | 0.649 |  |  |  |
| n_opp_within_10 | E2a | 252098 | 0.643 | 0.645 | 0.639 | 0.638 |  |  |  |
| n_opp_within_3_of_end | base_type | 70295 | -0.000 | -0.000 | 0.161 | 0.162 |  |  |  |
| n_opp_within_3_of_end | loc | 70295 | 0.143 | 0.144 | 0.146 | 0.146 |  |  |  |
| n_opp_within_3_of_end | E0 | 70295 | 0.172 | 0.168 | 0.142 | 0.143 |  |  |  |
| n_opp_within_3_of_end | E2 | 70295 | 0.179 | 0.179 | 0.142 | 0.142 |  |  |  |
| n_opp_within_3_of_end | E2a | 70295 | 0.481 | 0.483 | 0.096 | 0.096 |  |  |  |
| n_opp_within_5 | base_type | 252098 | 0.152 | 0.152 | 0.588 | 0.588 |  |  |  |
| n_opp_within_5 | loc | 252098 | 0.152 | 0.152 | 0.620 | 0.620 |  |  |  |
| n_opp_within_5 | E0 | 252098 | 0.474 | 0.475 | 0.417 | 0.417 |  |  |  |
| n_opp_within_5 | E2 | 252098 | 0.545 | 0.545 | 0.386 | 0.386 |  |  |  |
| n_opp_within_5 | E2a | 252098 | 0.557 | 0.559 | 0.378 | 0.377 |  |  |  |
| nearest_opp_dist | base_type | 251884 | 0.170 | 0.170 | 3.390 | 3.392 |  |  |  |
| nearest_opp_dist | loc | 251884 | 0.119 | 0.119 | 3.741 | 3.740 |  |  |  |
| nearest_opp_dist | E0 | 251884 | 0.612 | 0.616 | 2.272 | 2.258 |  |  |  |
| nearest_opp_dist | E2 | 251884 | 0.681 | 0.684 | 2.041 | 2.031 |  |  |  |
| nearest_opp_dist | E2a | 251884 | 0.711 | 0.714 | 1.942 | 1.931 |  |  |  |
| nearest_opp_dist_in_cone | base_type | 58683 | 0.041 | 0.042 | 7.938 | 7.939 |  |  |  |
| nearest_opp_dist_in_cone | loc | 58683 | 0.245 | 0.247 | 7.000 | 6.991 |  |  |  |
| nearest_opp_dist_in_cone | E0 | 58683 | 0.387 | 0.390 | 6.349 | 6.335 |  |  |  |
| nearest_opp_dist_in_cone | E2 | 58683 | 0.420 | 0.423 | 6.181 | 6.157 |  |  |  |
| nearest_opp_dist_in_cone | E2a | 58683 | 0.426 | 0.427 | 6.156 | 6.143 |  |  |  |
| opp_keeper_dist_to_goal_line | base_type | 45957 | 0.006 | 0.009 | 2.075 | 2.079 |  |  |  |
| opp_keeper_dist_to_goal_line | loc | 45957 | 0.354 | 0.364 | 1.608 | 1.599 |  |  |  |
| opp_keeper_dist_to_goal_line | E0 | 45957 | 0.427 | 0.452 | 1.537 | 1.507 |  |  |  |
| opp_keeper_dist_to_goal_line | E2 | 45957 | 0.499 | 0.520 | 1.469 | 1.436 |  |  |  |
| opp_keeper_dist_to_goal_line | E2a | 45957 | 0.503 | 0.525 | 1.462 | 1.430 |  |  |  |

## Error analysis (E2 student, out-of-fold)

MAE per bin for continuous / count targets, log-loss for binaries (full tables incl. n, RMSE, within-bin R2 and bias in `soccer_02_by_*.parquet`).


### Elapsed possession time (s)

| poss_elapsed_s | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| [0.0, 2.0) | 95404 | 3.044 | 3.481 | 1.145 |  | 0.262 | 0.424 | 2.154 | 0.330 | 0.517 |
| [2.0, 5.0) | 98631 | 2.968 | 3.205 | 1.145 |  | 0.281 | 0.442 | 1.966 | 0.356 | 0.453 |
| [5.0, 10.0) | 124384 | 3.191 | 3.218 | 1.179 |  | 0.215 | 0.395 | 2.114 | 0.369 | 0.434 |
| [10.0, 20.0) | 182563 | 3.158 | 3.078 | 1.191 | 0.102 | 0.158 | 0.376 | 2.135 | 0.388 | 0.418 |
| [20.0, 40.0) | 203212 | 2.945 | 2.838 | 1.172 | 0.142 | 0.113 | 0.384 | 2.057 | 0.413 | 0.423 |
| [40.0, inf) | 145955 | 2.745 | 2.624 | 1.151 | 0.175 | 0.079 | 0.395 | 1.989 | 0.441 | 0.431 |

### Located events since the opponent's last defensive action (from the sequence block; 20+ = none in the last 20 or none this period)

| events_since_opp_def_action | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| [0, 1) | 96301 | 2.748 | 2.900 | 1.155 | 0.136 | 0.250 | 0.452 | 1.133 | 0.366 | 0.520 |
| [1, 2) | 83570 | 3.063 | 3.095 | 1.226 | 0.150 | 0.307 | 0.455 | 1.773 | 0.361 | 0.504 |
| [2, 3) | 68706 | 2.887 | 2.942 | 1.168 | 0.150 | 0.233 | 0.449 | 1.873 | 0.381 | 0.522 |
| [3, 5) | 96760 | 3.055 | 3.037 | 1.201 | 0.160 | 0.188 | 0.442 | 2.195 | 0.400 | 0.477 |
| [5, 10) | 151278 | 3.007 | 2.938 | 1.166 | 0.149 | 0.131 | 0.388 | 2.314 | 0.403 | 0.427 |
| [10, 20) | 151553 | 3.023 | 2.972 | 1.157 | 0.138 | 0.111 | 0.364 | 2.280 | 0.400 | 0.392 |
| 20+ / none | 201981 | 3.110 | 3.175 | 1.140 | 0.116 | 0.135 | 0.361 | 2.220 | 0.383 | 0.383 |

### Seconds since the opponent's last defensive action

| since_opp_def_action_s | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| [0.0, 2.0) | 203234 | 2.889 | 3.038 | 1.197 | 0.137 | 0.298 | 0.484 | 1.174 | 0.351 | 0.488 |
| [2.0, 5.0) | 126526 | 3.041 | 3.027 | 1.177 | 0.158 | 0.187 | 0.435 | 2.221 | 0.399 | 0.487 |
| [5.0, 10.0) | 112387 | 3.054 | 2.917 | 1.175 | 0.156 | 0.117 | 0.375 | 2.407 | 0.404 | 0.423 |
| [10.0, 30.0) | 202243 | 2.925 | 2.865 | 1.144 | 0.140 | 0.094 | 0.356 | 2.318 | 0.410 | 0.403 |
| [30.0, 60.0) | 103235 | 3.071 | 3.107 | 1.162 | 0.111 | 0.141 | 0.367 | 2.268 | 0.384 | 0.449 |
| [60.0, inf) | 91202 | 3.281 | 3.351 | 1.154 | 0.095 | 0.165 | 0.367 | 2.327 | 0.373 | 0.405 |
| none this period | 11322 | 2.888 | 3.177 | 0.999 | 0.028 | 0.072 | 0.283 | 2.345 | 0.459 | 0.308 |

### Set-piece phase vs open play

| phase | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| open_play | 404976 | 3.096 | 3.216 | 1.163 | 0.138 | 0.199 | 0.402 | 2.056 | 0.359 | 0.425 |
| set_piece | 445173 | 2.925 | 2.848 | 1.171 |  | 0.137 | 0.393 | 2.089 | 0.415 | 0.455 |

### Pitch third of the ball (event team's frame)

| third | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| final | 290654 | 2.448 | 2.145 | 1.194 | 0.283 |  | 0.516 | 1.601 | 0.397 | 0.640 |
| middle | 456199 | 3.174 | 3.226 | 1.151 | 0.057 | 0.167 | 0.377 | 2.050 | 0.425 | 0.382 |
| own | 103296 | 3.838 | 4.599 | 1.163 | 0.001 |  | 0.324 | 2.557 | 0.296 | 0.372 |

### Competition

| competition | n (block_depth) | block_depth (mae) | def_line (mae) | n_opp_ahead_of_ball (mae) | deep_block (log_loss) | counter_on (log_loss) | n_opp_within_5 (mae) | nearest_opp_dist (mae) | n_opp_in_cone (mae) | n_opp_in_lane (mae) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Bundesliga 2023/2024 | 73235 | 3.026 | 3.058 | 1.198 | 0.134 | 0.164 | 0.414 | 1.981 | 0.415 | 0.445 |
| FIFA World Cup 2022 | 121681 | 3.069 | 3.081 | 1.183 | 0.144 | 0.168 | 0.371 | 2.263 | 0.382 | 0.414 |
| La Liga 2020/2021 | 96299 | 3.089 | 3.027 | 1.139 | 0.143 | 0.134 | 0.411 | 2.122 | 0.407 | 0.437 |
| Ligue 1 2021/2022 | 65956 | 3.048 | 3.056 | 1.108 | 0.127 | 0.106 | 0.417 | 2.114 | 0.406 | 0.433 |
| Ligue 1 2022/2023 | 83847 | 3.077 | 3.094 | 1.113 | 0.123 | 0.107 | 0.413 | 2.014 | 0.417 | 0.424 |
| Major League Soccer 2023 | 5099 | 3.395 | 3.458 | 1.468 | 0.144 | 0.320 | 0.400 | 2.071 | 0.378 | 0.437 |
| UEFA Euro 2020 | 101176 | 3.060 | 3.087 | 1.223 | 0.147 | 0.174 | 0.382 | 2.159 | 0.387 | 0.425 |
| UEFA Euro 2024 | 113327 | 3.013 | 2.977 | 1.160 | 0.140 | 0.127 | 0.376 | 2.101 | 0.411 | 0.425 |
| UEFA Women's Euro 2022 | 44137 | 2.846 | 2.908 | 1.170 | 0.120 | 0.267 | 0.402 | 2.011 | 0.342 | 0.473 |
| UEFA Women's Euro 2025 | 49141 | 2.779 | 2.857 | 1.156 | 0.130 | 0.234 | 0.404 | 1.904 | 0.365 | 0.474 |
| Women's World Cup 2023 | 96251 | 2.842 | 2.935 | 1.177 | 0.158 | 0.257 | 0.417 | 1.919 | 0.349 | 0.480 |

## Neural student vs LightGBM (held-out fold 0)

`mlp_E2seq`: scikit-learn `MLPRegressor` on the one-hot E2 design plus a dense per-slot encoding of the 20-event sequence block (hidden (256, 128), Adam, batch 512, early stopping on a random 10% of the training rows, max 40 epochs), trained on 200,000 rows of folds 1-4. `lgbm_E2_same_rows` / `lgbm_E3` are LightGBM on the same rows with E2 / E2 + raw sequence block; `cv_*` rows are the CV students (trained on the larger cap) scored on the same fold.

| target | model | n_train | n_features | epochs_or_rounds | seconds | n | r2 | mae | rmse |
|---|---|---|---|---|---|---|---|---|---|
| block_depth | cv_base_type (train_cap rows) |  |  |  |  | 169539 | 0.032 | 16.342 | 20.147 |
| block_depth | cv_loc (train_cap rows) |  | 5 |  |  | 169539 | 0.854 | 6.141 | 7.837 |
| block_depth | cv_E2 (train_cap rows) |  | 62 |  |  | 169539 | 0.963 | 3.008 | 3.918 |
| block_depth | cv_E2a (train_cap rows) |  | 78 |  |  | 169539 | 0.964 | 2.978 | 3.883 |
| block_depth | lgbm_E2_same_rows | 200000 | 62 | 397 | 21.624 | 169539 | 0.963 | 3.027 | 3.941 |
| block_depth | lgbm_E3 | 200000 | 162 | 400 | 38.181 | 169539 | 0.972 | 2.650 | 3.452 |
| block_depth | mlp_E2seq | 200000 | 467 | 26 | 79.208 | 169539 | 0.965 | 2.950 | 3.836 |
| n_opp_within_5 | cv_base_type (train_cap rows) |  |  |  |  | 265999 | 0.144 | 0.590 | 0.740 |
| n_opp_within_5 | cv_loc (train_cap rows) |  | 5 |  |  | 265999 | 0.146 | 0.619 | 0.739 |
| n_opp_within_5 | cv_E2 (train_cap rows) |  | 62 |  |  | 265999 | 0.524 | 0.394 | 0.552 |
| n_opp_within_5 | cv_E2a (train_cap rows) |  | 78 |  |  | 265999 | 0.536 | 0.385 | 0.544 |
| n_opp_within_5 | lgbm_E2_same_rows | 200000 | 62 | 232 | 6.581 | 265999 | 0.519 | 0.396 | 0.554 |
| n_opp_within_5 | lgbm_E3 | 200000 | 162 | 196 | 15.534 | 265999 | 0.530 | 0.391 | 0.548 |
| n_opp_within_5 | mlp_E2seq | 200000 | 467 | 14 | 41.849 | 265999 | 0.482 | 0.417 | 0.575 |
| nearest_opp_dist | cv_base_type (train_cap rows) |  |  |  |  | 265734 | 0.165 | 3.429 | 4.760 |
| nearest_opp_dist | cv_loc (train_cap rows) |  | 5 |  |  | 265734 | 0.120 | 3.751 | 4.886 |
| nearest_opp_dist | cv_E2 (train_cap rows) |  | 62 |  |  | 265734 | 0.670 | 2.097 | 2.995 |
| nearest_opp_dist | cv_E2a (train_cap rows) |  | 78 |  |  | 265734 | 0.700 | 1.998 | 2.856 |
| nearest_opp_dist | lgbm_E2_same_rows | 200000 | 62 | 218 | 6.769 | 265734 | 0.666 | 2.109 | 3.011 |
| nearest_opp_dist | lgbm_E3 | 200000 | 162 | 201 | 16.385 | 265734 | 0.681 | 2.057 | 2.942 |
| nearest_opp_dist | mlp_E2seq | 200000 | 467 | 13 | 41.842 | 265734 | 0.636 | 2.236 | 3.142 |

## Calibration of the binaries (E2, out-of-fold, equal-count bins)

| target | feature_set | bin | pred | obs | n |
|---|---|---|---|---|---|
| counter_on | E2 | 0 | 0.000 | 0.000 | 45620 |
| counter_on | E2 | 1 | 0.001 | 0.000 | 45620 |
| counter_on | E2 | 2 | 0.002 | 0.001 | 45620 |
| counter_on | E2 | 3 | 0.004 | 0.002 | 45620 |
| counter_on | E2 | 4 | 0.007 | 0.005 | 45620 |
| counter_on | E2 | 5 | 0.014 | 0.014 | 45620 |
| counter_on | E2 | 6 | 0.030 | 0.035 | 45620 |
| counter_on | E2 | 7 | 0.069 | 0.079 | 45620 |
| counter_on | E2 | 8 | 0.177 | 0.192 | 45620 |
| counter_on | E2 | 9 | 0.514 | 0.503 | 45619 |
| deep_block | E2 | 0 | 0.000 | 0.000 | 23523 |
| deep_block | E2 | 1 | 0.000 | 0.000 | 23523 |
| deep_block | E2 | 2 | 0.000 | 0.000 | 23523 |
| deep_block | E2 | 3 | 0.000 | 0.000 | 23523 |
| deep_block | E2 | 4 | 0.002 | 0.002 | 23523 |
| deep_block | E2 | 5 | 0.018 | 0.017 | 23522 |
| deep_block | E2 | 6 | 0.095 | 0.112 | 23522 |
| deep_block | E2 | 7 | 0.472 | 0.471 | 23522 |
| deep_block | E2 | 8 | 0.905 | 0.900 | 23522 |
| deep_block | E2 | 9 | 0.997 | 0.998 | 23522 |

## Feature importance of the final students (gain share, top 10)


**block_depth (E2)**: f_dist_goal 0.49, f_x 0.39, f_position 0.04, f_poss_t_since_ft_entry 0.02, f_poss_start_x 0.01, f_opp_poss_last10s 0.01, f_ball_dx_3 0.01, f_opp_def_x_60s_mean 0.00, f_t_since_opp_def_action 0.00, f_w10_n_own 0.00

**def_line (E2)**: f_x 0.58, f_dist_goal 0.28, f_position 0.04, f_poss_t_since_ft_entry 0.02, f_poss_start_x 0.01, f_opp_poss_last10s 0.01, f_ball_dx_3 0.01, f_poss_elapsed 0.01, f_opp_def_x_60s_mean 0.00, f_goal_opening 0.00

**n_opp_ahead_of_ball (E2)**: f_position 0.33, f_x 0.20, f_poss_t_since_ft_entry 0.06, f_ball_dx_3 0.05, f_opp_poss_last10s 0.04, f_poss_start_x 0.03, f_type_id 0.03, f_w10_n_own 0.02, f_dt_prev 0.02, f_goal_opening 0.02

**deep_block (E2)**: f_x 0.42, f_dist_goal 0.21, f_poss_t_since_ft_entry 0.12, f_poss_in_final_third 0.10, f_position 0.03, f_ball_dx_3 0.02, f_goal_opening 0.02, f_dt_prev 0.01, f_opp_def_x_60s_mean 0.01, f_goal_bearing 0.01

**counter_on (E2)**: f_position 0.33, f_poss_start_x 0.08, f_ball_dx_3 0.06, f_w10_n_own 0.06, f_poss_t_since_ft_entry 0.05, f_type_id 0.04, f_opp_poss_last10s 0.04, f_dt_prev 0.04, f_poss_elapsed 0.03, f_opp_def_x_60s_mean 0.03

**n_opp_within_5 (E2)**: f_type_id 0.19, f_t_since_opp_def_action 0.18, f_position 0.14, f_dist_goal 0.09, f_goal_opening 0.09, f_pass_type 0.03, f_is_possession_team 0.02, f_dt_prev 0.02, f_under_pressure 0.02, f_ball_dx_3 0.02

**nearest_opp_dist (E2)**: f_position 0.28, f_type_id 0.21, f_t_since_opp_def_action 0.10, f_pass_type 0.07, f_under_pressure 0.04, f_dt_prev 0.03, f_dist_goal 0.03, f_is_possession_team 0.03, f_x 0.03, f_goal_opening 0.02

**n_opp_in_cone (E2)**: f_goal_opening 0.16, f_position 0.16, f_poss_t_since_ft_entry 0.10, f_ball_dx_3 0.07, f_x 0.07, f_dist_goal 0.06, f_poss_start_x 0.04, f_type_id 0.03, f_dt_prev 0.03, f_goal_bearing 0.03

**n_opp_in_lane (E2)**: f_t_since_opp_def_action 0.26, f_x 0.21, f_dist_goal 0.11, f_position 0.07, f_under_pressure 0.05, f_dt_prev 0.05, f_pass_type 0.03, f_is_possession_team 0.02, f_pass_body_part 0.02, f_goal_opening 0.01


## Fit summary (CV)

| target | fset | n_train_mean | rounds_median | rounds_min | rounds_max | seconds_mean |
|---|---|---|---|---|---|---|
| block_depth | E0 | 300000.0 | 170.0 | 154 | 255 | 6.9 |
| block_depth | E1 | 300000.0 | 397.0 | 340 | 400 | 13.0 |
| block_depth | E2 | 300000.0 | 400.0 | 393 | 400 | 17.1 |
| block_depth | E2a | 300000.0 | 400.0 | 394 | 400 | 19.6 |
| block_depth | loc | 300000.0 | 54.0 | 38 | 64 | 1.4 |
| block_length | E0 | 300000.0 | 80.0 | 75 | 115 | 3.0 |
| block_length | E1 | 300000.0 | 150.0 | 113 | 185 | 4.7 |
| block_length | E2 | 300000.0 | 167.0 | 146 | 228 | 7.1 |
| block_length | E2a | 300000.0 | 173.0 | 139 | 197 | 8.2 |
| block_length | loc | 300000.0 | 39.0 | 27 | 60 | 1.0 |
| block_width | E0 | 300000.0 | 145.0 | 128 | 163 | 4.7 |
| block_width | E1 | 300000.0 | 185.0 | 146 | 240 | 6.8 |
| block_width | E2 | 300000.0 | 220.0 | 168 | 288 | 9.3 |
| block_width | E2a | 300000.0 | 254.0 | 189 | 332 | 10.8 |
| block_width | loc | 300000.0 | 42.0 | 40 | 44 | 1.4 |
| counter_on | E0 | 300000.0 | 63.0 | 45 | 86 | 2.5 |
| counter_on | E1 | 300000.0 | 95.0 | 74 | 141 | 3.8 |
| counter_on | E2 | 300000.0 | 127.0 | 115 | 196 | 5.7 |
| counter_on | E2a | 300000.0 | 168.0 | 123 | 214 | 7.6 |
| counter_on | loc | 300000.0 | 14.0 | 11 | 16 | 0.8 |
| deep_block | E0 | 188180.0 | 73.0 | 62 | 81 | 1.6 |
| deep_block | E1 | 188180.0 | 79.0 | 77 | 112 | 2.2 |
| deep_block | E2 | 188180.0 | 105.0 | 76 | 158 | 3.2 |
| deep_block | E2a | 188180.0 | 103.0 | 97 | 129 | 3.8 |
| deep_block | loc | 188180.0 | 55.0 | 51 | 61 | 0.8 |
| def_line | E0 | 300000.0 | 163.0 | 99 | 197 | 5.3 |
| def_line | E1 | 300000.0 | 337.0 | 265 | 399 | 8.5 |
| def_line | E2 | 300000.0 | 398.0 | 386 | 400 | 13.3 |
| def_line | E2a | 300000.0 | 397.0 | 384 | 400 | 16.4 |
| def_line | loc | 300000.0 | 51.0 | 36 | 58 | 1.4 |
| n_opp_ahead_of_ball | E0 | 300000.0 | 114.0 | 107 | 133 | 5.9 |
| n_opp_ahead_of_ball | E1 | 300000.0 | 300.0 | 194 | 360 | 10.1 |
| n_opp_ahead_of_ball | E2 | 300000.0 | 397.0 | 393 | 400 | 14.9 |
| n_opp_ahead_of_ball | E2a | 300000.0 | 368.0 | 326 | 395 | 16.9 |
| n_opp_ahead_of_ball | loc | 300000.0 | 41.0 | 31 | 50 | 1.2 |
| n_opp_in_cone | E0 | 300000.0 | 69.0 | 55 | 87 | 2.7 |
| n_opp_in_cone | E1 | 300000.0 | 76.0 | 62 | 119 | 3.4 |
| n_opp_in_cone | E2 | 300000.0 | 78.0 | 53 | 134 | 4.5 |
| n_opp_in_cone | E2a | 300000.0 | 97.0 | 58 | 180 | 5.8 |
| n_opp_in_cone | E2r | 300000.0 | 92.0 | 59 | 110 | 4.5 |
| n_opp_in_cone | loc | 300000.0 | 27.0 | 20 | 41 | 1.0 |
| n_opp_in_lane | E0 | 298784.8 | 78.0 | 63 | 78 | 2.1 |
| n_opp_in_lane | E1 | 298784.8 | 132.0 | 123 | 183 | 4.0 |
| n_opp_in_lane | E2 | 298784.8 | 92.0 | 82 | 119 | 4.2 |
| n_opp_in_lane | E2a | 298784.8 | 171.0 | 150 | 285 | 6.5 |
| n_opp_in_lane | E2r | 225705.6 | 92.0 | 78 | 137 | 3.4 |
| n_opp_in_lane | loc | 298784.8 | 31.0 | 23 | 44 | 0.8 |
| n_opp_within_10 | E0 | 300000.0 | 159.0 | 111 | 217 | 4.5 |
| n_opp_within_10 | E1 | 300000.0 | 379.0 | 355 | 400 | 9.7 |
| n_opp_within_10 | E2 | 300000.0 | 400.0 | 394 | 400 | 11.9 |
| n_opp_within_10 | E2a | 300000.0 | 400.0 | 399 | 400 | 14.0 |
| n_opp_within_10 | E2r | 300000.0 | 387.0 | 350 | 399 | 11.7 |
| n_opp_within_10 | loc | 300000.0 | 45.0 | 36 | 52 | 1.1 |
| n_opp_within_3_of_end | E0 | 298784.8 | 32.0 | 25 | 33 | 3.3 |
| n_opp_within_3_of_end | E1 | 298784.8 | 46.0 | 28 | 65 | 4.1 |
| n_opp_within_3_of_end | E2 | 298784.8 | 53.0 | 37 | 66 | 5.0 |
| n_opp_within_3_of_end | E2a | 298784.8 | 137.0 | 82 | 166 | 7.1 |
| n_opp_within_3_of_end | E2r | 225705.6 | 44.0 | 31 | 51 | 2.1 |
| n_opp_within_3_of_end | loc | 298784.8 | 26.0 | 24 | 40 | 0.7 |
| n_opp_within_5 | E0 | 300000.0 | 125.0 | 116 | 159 | 6.1 |
| n_opp_within_5 | E1 | 300000.0 | 240.0 | 213 | 295 | 9.2 |
| n_opp_within_5 | E2 | 300000.0 | 250.0 | 201 | 301 | 10.9 |
| n_opp_within_5 | E2a | 300000.0 | 303.0 | 281 | 398 | 14.4 |
| n_opp_within_5 | E2r | 300000.0 | 231.0 | 196 | 252 | 8.1 |
| n_opp_within_5 | loc | 300000.0 | 33.0 | 31 | 41 | 1.0 |
| nearest_opp_dist | E0 | 300000.0 | 137.0 | 106 | 176 | 4.2 |
| nearest_opp_dist | E1 | 300000.0 | 245.0 | 233 | 260 | 7.2 |
| nearest_opp_dist | E2 | 300000.0 | 325.0 | 238 | 400 | 10.3 |
| nearest_opp_dist | E2a | 300000.0 | 394.0 | 362 | 400 | 13.8 |
| nearest_opp_dist | E2r | 300000.0 | 254.0 | 206 | 329 | 9.0 |
| nearest_opp_dist | loc | 300000.0 | 37.0 | 26 | 55 | 1.0 |
| nearest_opp_dist_in_cone | E0 | 235037.6 | 77.0 | 60 | 129 | 2.3 |
| nearest_opp_dist_in_cone | E1 | 235037.6 | 128.0 | 99 | 138 | 3.4 |
| nearest_opp_dist_in_cone | E2 | 235037.6 | 116.0 | 78 | 169 | 4.4 |
| nearest_opp_dist_in_cone | E2a | 235037.6 | 111.0 | 87 | 136 | 5.2 |
| nearest_opp_dist_in_cone | E2r | 206692.0 | 103.0 | 96 | 142 | 3.8 |
| nearest_opp_dist_in_cone | loc | 235037.6 | 33.0 | 30 | 42 | 0.7 |
| opp_keeper_dist_to_goal_line | E0 | 187032.8 | 91.0 | 63 | 121 | 4.0 |
| opp_keeper_dist_to_goal_line | E1 | 187032.8 | 179.0 | 144 | 188 | 5.6 |
| opp_keeper_dist_to_goal_line | E2 | 187032.8 | 150.0 | 123 | 199 | 6.2 |
| opp_keeper_dist_to_goal_line | E2a | 187032.8 | 188.0 | 113 | 259 | 8.2 |
| opp_keeper_dist_to_goal_line | E2r | 179736.0 | 147.0 | 114 | 198 | 4.1 |
| opp_keeper_dist_to_goal_line | loc | 187032.8 | 37.0 | 35 | 45 | 0.7 |

## Applied to the non-360 matches

`apply_student.py` writes `processed_dir('soccer')/imputed_no360.parquet` and the shift tables `soccer_02_shift_no360.parquet` / `soccer_02_shift_by_competition.parquet` (see the section appended by that script below, if it has been run).


### Distribution of the imputed values: 2015/16 leagues vs the 360 domain (same event types: Pass, Carry, Shot)

`no360 leagues 2015/16 imputed` = the four full 2015/16 seasons (shots + 25% of passes / carries); `360 oof imputed` / `360 truth` = out-of-fold imputation and 360 label on Pass / Carry / Shot rows of the 360 matches. `360 truth` exists only on labelled rows (reliable frames for the team-shape targets, visible keeper / non-empty cone for the keeper and cone distances), whereas the imputed rows cover the student's whole subset, so the shift between domains is read from the two imputed rows; imputed vs truth is a selection effect unless the labelled rows are the same.

| target | source | n | mean | sd | p05 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| block_depth | no360 leagues 2015/16 imputed | 645662 | 48.328 | 22.020 | 14.038 | 48.400 | 86.053 |
| block_depth | no360 other competitions imputed | 65158 | 47.679 | 22.068 | 13.697 | 47.683 | 85.271 |
| block_depth | 360 oof imputed | 673469 | 49.275 | 21.897 | 15.969 | 48.028 | 86.891 |
| block_depth | 360 truth | 511385 | 43.901 | 20.450 | 14.334 | 41.803 | 82.265 |
| def_line | no360 leagues 2015/16 imputed | 645662 | 37.486 | 19.337 | 8.060 | 36.920 | 71.060 |
| def_line | no360 other competitions imputed | 65158 | 36.642 | 19.394 | 7.663 | 35.719 | 70.243 |
| def_line | 360 oof imputed | 673469 | 38.278 | 19.411 | 10.017 | 36.321 | 72.206 |
| def_line | 360 truth | 511385 | 33.526 | 18.040 | 8.424 | 30.881 | 67.763 |
| block_width | no360 leagues 2015/16 imputed | 645662 | 38.480 | 6.220 | 25.323 | 39.504 | 46.340 |
| block_width | no360 other competitions imputed | 65158 | 39.080 | 6.164 | 26.061 | 40.255 | 46.675 |
| block_width | 360 oof imputed | 673469 | 38.656 | 5.576 | 28.253 | 39.161 | 46.418 |
| block_width | 360 truth | 511385 | 38.196 | 8.385 | 23.923 | 38.493 | 51.299 |
| block_length | no360 leagues 2015/16 imputed | 645662 | 25.212 | 5.209 | 15.970 | 26.081 | 32.333 |
| block_length | no360 other competitions imputed | 65158 | 25.322 | 5.329 | 16.042 | 26.168 | 32.796 |
| block_length | 360 oof imputed | 673469 | 25.073 | 4.857 | 16.521 | 25.468 | 32.572 |
| block_length | 360 truth | 511385 | 24.177 | 7.061 | 12.394 | 24.131 | 35.875 |
| n_opp_ahead_of_ball | no360 leagues 2015/16 imputed | 645662 | 6.861 | 2.122 | 2.802 | 7.233 | 9.529 |
| n_opp_ahead_of_ball | no360 other competitions imputed | 65158 | 6.855 | 2.045 | 2.854 | 7.289 | 9.276 |
| n_opp_ahead_of_ball | 360 oof imputed | 673469 | 6.976 | 1.920 | 3.297 | 7.359 | 9.394 |
| n_opp_ahead_of_ball | 360 truth | 511385 | 6.932 | 2.441 | 2.000 | 7.000 | 10.000 |
| deep_block | no360 leagues 2015/16 imputed | 151124 | 0.232 | 0.383 | 0.000 | 0.001 | 0.999 |
| deep_block | no360 other competitions imputed | 17324 | 0.240 | 0.387 | 0.000 | 0.001 | 0.999 |
| deep_block | 360 oof imputed | 181116 | 0.200 | 0.355 | 0.000 | 0.001 | 0.998 |
| deep_block | 360 truth | 143984 | 0.239 | 0.426 | 0.000 | 0.000 | 1.000 |
| counter_on | no360 leagues 2015/16 imputed | 320421 | 0.063 | 0.140 | 0.000 | 0.008 | 0.370 |
| counter_on | no360 other competitions imputed | 30796 | 0.060 | 0.139 | 0.000 | 0.006 | 0.367 |
| counter_on | 360 oof imputed | 338673 | 0.056 | 0.134 | 0.000 | 0.005 | 0.338 |
| counter_on | 360 truth | 281162 | 0.058 | 0.233 | 0.000 | 0.000 | 1.000 |
| n_opp_within_5 | no360 leagues 2015/16 imputed | 687791 | 0.713 | 0.587 | -0.002 | 0.634 | 1.696 |
| n_opp_within_5 | no360 other competitions imputed | 68491 | 0.663 | 0.589 | -0.020 | 0.565 | 1.658 |
| n_opp_within_5 | 360 oof imputed | 706960 | 0.640 | 0.539 | -0.010 | 0.550 | 1.572 |
| n_opp_within_5 | 360 truth | 706628 | 0.638 | 0.772 | 0.000 | 0.000 | 2.000 |
| n_opp_within_10 | no360 leagues 2015/16 imputed | 687791 | 1.762 | 1.146 | 0.215 | 1.651 | 3.696 |
| n_opp_within_10 | no360 other competitions imputed | 68491 | 1.619 | 1.158 | 0.130 | 1.476 | 3.636 |
| n_opp_within_10 | 360 oof imputed | 706960 | 1.609 | 1.050 | 0.175 | 1.510 | 3.369 |
| n_opp_within_10 | 360 truth | 706628 | 1.607 | 1.343 | 0.000 | 1.000 | 4.000 |
| nearest_opp_dist | no360 leagues 2015/16 imputed | 687791 | 6.339 | 4.585 | 1.584 | 5.217 | 13.864 |
| nearest_opp_dist | no360 other competitions imputed | 68491 | 6.722 | 4.707 | 1.659 | 5.531 | 14.932 |
| nearest_opp_dist | 360 oof imputed | 706960 | 6.589 | 4.222 | 1.624 | 5.579 | 14.152 |
| nearest_opp_dist | 360 truth | 705688 | 6.574 | 5.235 | 1.013 | 5.124 | 16.424 |
| n_opp_in_cone | no360 leagues 2015/16 imputed | 645662 | 0.335 | 0.261 | 0.079 | 0.285 | 0.711 |
| n_opp_in_cone | no360 other competitions imputed | 65158 | 0.322 | 0.260 | 0.077 | 0.267 | 0.712 |
| n_opp_in_cone | 360 oof imputed | 673469 | 0.322 | 0.225 | 0.081 | 0.270 | 0.717 |
| n_opp_in_cone | 360 truth | 673160 | 0.323 | 0.565 | 0.000 | 0.000 | 1.000 |
| nearest_opp_dist_in_cone | no360 leagues 2015/16 imputed | 645662 | 15.902 | 7.318 | 4.905 | 15.780 | 27.243 |
| nearest_opp_dist_in_cone | no360 other competitions imputed | 65158 | 16.369 | 7.383 | 4.977 | 16.291 | 27.547 |
| nearest_opp_dist_in_cone | 360 oof imputed | 673469 | 16.150 | 6.433 | 5.875 | 16.069 | 26.405 |
| nearest_opp_dist_in_cone | 360 truth | 187873 | 16.913 | 9.858 | 2.619 | 16.337 | 33.839 |
| opp_keeper_dist_to_goal_line | no360 leagues 2015/16 imputed | 343957 | 6.142 | 3.431 | 1.784 | 5.500 | 12.451 |
| opp_keeper_dist_to_goal_line | no360 other competitions imputed | 33996 | 5.791 | 3.259 | 1.729 | 5.099 | 11.837 |
| opp_keeper_dist_to_goal_line | 360 oof imputed | 344842 | 6.049 | 3.142 | 1.822 | 5.528 | 11.793 |
| opp_keeper_dist_to_goal_line | 360 truth | 134909 | 3.849 | 2.753 | 0.154 | 3.463 | 8.879 |
| n_opp_within_3_of_end | no360 leagues 2015/16 imputed | 370952 | 0.088 | 0.127 | 0.010 | 0.051 | 0.254 |
| n_opp_within_3_of_end | no360 other competitions imputed | 35986 | 0.089 | 0.137 | 0.009 | 0.049 | 0.267 |
| n_opp_within_3_of_end | 360 oof imputed | 373639 | 0.086 | 0.127 | 0.011 | 0.048 | 0.259 |
| n_opp_within_3_of_end | 360 truth | 373481 | 0.086 | 0.324 | 0.000 | 0.000 | 1.000 |
| n_opp_in_lane | no360 leagues 2015/16 imputed | 370952 | 0.501 | 0.382 | 0.084 | 0.368 | 1.218 |
| n_opp_in_lane | no360 other competitions imputed | 35986 | 0.498 | 0.396 | 0.074 | 0.363 | 1.235 |
| n_opp_in_lane | 360 oof imputed | 373639 | 0.495 | 0.394 | 0.079 | 0.361 | 1.241 |
| n_opp_in_lane | 360 truth | 373481 | 0.494 | 0.733 | 0.000 | 0.000 | 2.000 |

**Shots only**

| target | source | n | mean | sd | p50 |
|---|---|---|---|---|---|
| block_depth | no360 leagues 2015/16 imputed | 37434 | 16.643 | 5.746 | 15.745 |
| block_depth | no360 other competitions imputed | 3761 | 15.894 | 5.750 | 14.963 |
| block_depth | 360 oof imputed | 10201 | 15.449 | 5.125 | 14.890 |
| block_depth | 360 truth | 9178 | 15.273 | 5.628 | 14.909 |
| def_line | no360 leagues 2015/16 imputed | 37434 | 10.685 | 5.165 | 10.095 |
| def_line | no360 other competitions imputed | 3761 | 9.939 | 5.247 | 9.342 |
| def_line | 360 oof imputed | 10201 | 9.531 | 4.739 | 9.045 |
| def_line | 360 truth | 9178 | 9.461 | 5.307 | 9.179 |
| block_width | no360 leagues 2015/16 imputed | 37434 | 24.706 | 3.853 | 24.534 |
| block_width | no360 other competitions imputed | 3761 | 25.296 | 3.927 | 25.270 |
| block_width | 360 oof imputed | 10201 | 24.396 | 3.770 | 24.495 |
| block_width | 360 truth | 9178 | 24.500 | 7.135 | 24.330 |
| block_length | no360 leagues 2015/16 imputed | 37434 | 15.916 | 3.310 | 16.233 |
| block_length | no360 other competitions imputed | 3761 | 15.699 | 3.440 | 16.160 |
| block_length | 360 oof imputed | 10201 | 15.678 | 2.973 | 16.066 |
| block_length | 360 truth | 9178 | 15.487 | 6.023 | 14.815 |
| n_opp_ahead_of_ball | no360 leagues 2015/16 imputed | 37434 | 5.536 | 2.630 | 5.272 |
| n_opp_ahead_of_ball | no360 other competitions imputed | 3761 | 5.583 | 2.644 | 5.312 |
| n_opp_ahead_of_ball | 360 oof imputed | 10201 | 5.403 | 2.509 | 5.138 |
| n_opp_ahead_of_ball | 360 truth | 9178 | 5.670 | 2.899 | 6.000 |
| deep_block | no360 leagues 2015/16 imputed | 10322 | 0.929 | 0.176 | 0.997 |
| deep_block | no360 other competitions imputed | 1002 | 0.948 | 0.145 | 0.997 |
| deep_block | 360 oof imputed | 2845 | 0.961 | 0.127 | 0.998 |
| deep_block | 360 truth | 2522 | 0.967 | 0.178 | 1.000 |
| counter_on | no360 leagues 2015/16 imputed | 197 | 0.181 | 0.228 | 0.064 |
| counter_on | no360 other competitions imputed | 24 | 0.174 | 0.265 | 0.026 |
| counter_on | 360 oof imputed | 26 | 0.263 | 0.303 | 0.100 |
| counter_on | 360 truth | 25 | 0.200 | 0.400 | 0.000 |
| n_opp_within_5 | no360 leagues 2015/16 imputed | 37888 | 1.737 | 0.897 | 1.677 |
| n_opp_within_5 | no360 other competitions imputed | 3799 | 1.731 | 0.920 | 1.679 |
| n_opp_within_5 | 360 oof imputed | 10367 | 1.892 | 0.954 | 1.787 |
| n_opp_within_5 | 360 truth | 10357 | 1.874 | 1.347 | 2.000 |
| n_opp_within_10 | no360 leagues 2015/16 imputed | 37888 | 4.060 | 1.603 | 3.920 |
| n_opp_within_10 | no360 other competitions imputed | 3799 | 4.038 | 1.653 | 3.879 |
| n_opp_within_10 | 360 oof imputed | 10367 | 4.263 | 1.638 | 4.110 |
| n_opp_within_10 | 360 truth | 10357 | 4.249 | 2.074 | 4.000 |
| nearest_opp_dist | no360 leagues 2015/16 imputed | 37888 | 3.027 | 1.875 | 2.765 |
| nearest_opp_dist | no360 other competitions imputed | 3799 | 3.069 | 2.012 | 2.733 |
| nearest_opp_dist | 360 oof imputed | 10367 | 2.854 | 1.725 | 2.684 |
| nearest_opp_dist | 360 truth | 10357 | 2.869 | 2.320 | 2.232 |
| n_opp_in_cone | no360 leagues 2015/16 imputed | 37434 | 0.683 | 0.652 | 0.470 |
| n_opp_in_cone | no360 other competitions imputed | 3761 | 0.689 | 0.654 | 0.484 |
| n_opp_in_cone | 360 oof imputed | 10201 | 0.673 | 0.596 | 0.505 |
| n_opp_in_cone | 360 truth | 10191 | 0.672 | 0.989 | 0.000 |
| nearest_opp_dist_in_cone | no360 leagues 2015/16 imputed | 37434 | 5.610 | 2.885 | 4.775 |
| nearest_opp_dist_in_cone | no360 other competitions imputed | 3761 | 5.715 | 3.029 | 4.860 |
| nearest_opp_dist_in_cone | 360 oof imputed | 10201 | 5.262 | 2.601 | 4.501 |
| nearest_opp_dist_in_cone | 360 truth | 4466 | 6.275 | 4.144 | 5.486 |
| opp_keeper_dist_to_goal_line | no360 leagues 2015/16 imputed | 37406 | 2.871 | 1.035 | 2.690 |
| opp_keeper_dist_to_goal_line | no360 other competitions imputed | 3755 | 2.765 | 1.021 | 2.582 |
| opp_keeper_dist_to_goal_line | 360 oof imputed | 10195 | 2.787 | 0.935 | 2.645 |
| opp_keeper_dist_to_goal_line | 360 truth | 9607 | 2.754 | 1.801 | 2.473 |
| n_opp_within_3_of_end | no360 leagues 2015/16 imputed | 0 |  |  |  |
| n_opp_within_3_of_end | no360 other competitions imputed | 0 |  |  |  |
| n_opp_within_3_of_end | 360 oof imputed | 0 |  |  |  |
| n_opp_within_3_of_end | 360 truth | 0 |  |  |  |
| n_opp_in_lane | no360 leagues 2015/16 imputed | 0 |  |  |  |
| n_opp_in_lane | no360 other competitions imputed | 0 |  |  |  |
| n_opp_in_lane | 360 oof imputed | 0 |  |  |  |
| n_opp_in_lane | 360 truth | 0 |  |  |  |

**Mean imputed value per competition (E2)**

| source | competition | season | gender | n_rows | n_matches | block_depth_mean | def_line_mean | n_opp_ahead_of_ball_mean | n_opp_within_5_mean | nearest_opp_dist_mean | n_opp_in_cone_mean | n_opp_in_lane_mean | deep_block_mean | counter_on_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no360 imputed | 1. Bundesliga | 2023/2024 | male | 1578 | 3 | 43.55 | 33.35 | 7.36 | 0.64 | 6.53 | 0.39 | 0.47 | 0.32 | 0.05 |
| no360 imputed | African Cup of Nations | 2023 | male | 20996 | 52 | 48.19 | 37.08 | 6.91 | 0.63 | 7.04 | 0.32 | 0.47 | 0.23 | 0.06 |
| no360 imputed | Copa America | 2024 | male | 13264 | 32 | 49.32 | 38.02 | 6.92 | 0.65 | 6.95 | 0.32 | 0.48 | 0.22 | 0.06 |
| no360 imputed | FIFA World Cup | 2018 | male | 30201 | 64 | 46.71 | 35.79 | 6.75 | 0.69 | 6.44 | 0.32 | 0.53 | 0.25 | 0.06 |
| no360 imputed | La Liga | 2015/2016 | male | 169367 | 380 | 48.10 | 37.28 | 6.83 | 0.71 | 6.37 | 0.33 | 0.50 | 0.24 | 0.06 |
| no360 imputed | Ligue 1 | 2015/2016 | male | 173190 | 377 | 49.97 | 38.91 | 6.86 | 0.71 | 6.36 | 0.32 | 0.49 | 0.19 | 0.06 |
| no360 imputed | Ligue 1 | 2022/2023 | male | 532 | 1 | 49.78 | 38.57 | 7.11 | 0.67 | 6.36 | 0.34 | 0.46 | 0.21 | 0.04 |
| no360 imputed | Major League Soccer | 2023 | male | 1513 | 3 | 49.38 | 38.41 | 7.21 | 0.67 | 6.59 | 0.36 | 0.45 | 0.21 | 0.05 |
| no360 imputed | Premier League | 2015/2016 | male | 171090 | 380 | 46.90 | 36.26 | 6.83 | 0.73 | 6.29 | 0.34 | 0.51 | 0.26 | 0.06 |
| no360 imputed | Serie A | 2015/2016 | male | 174144 | 380 | 48.32 | 37.47 | 6.92 | 0.71 | 6.34 | 0.34 | 0.50 | 0.24 | 0.06 |
| no360 imputed | UEFA Women's Euro | 2022 | female | 407 | 1 | 47.35 | 36.82 | 6.07 | 0.81 | 6.14 | 0.29 | 0.64 | 0.23 | 0.08 |
| 360 oof imputed | 1. Bundesliga | 2023/2024 | male | 57278 | 31 | 49.04 | 37.98 | 7.29 | 0.64 | 6.43 | 0.36 | 0.47 | 0.20 | 0.04 |
| 360 oof imputed | FIFA World Cup | 2022 | male | 108211 | 64 | 49.56 | 38.21 | 7.17 | 0.56 | 7.16 | 0.32 | 0.44 | 0.18 | 0.05 |
| 360 oof imputed | La Liga | 2020/2021 | male | 69237 | 35 | 47.61 | 37.01 | 7.25 | 0.65 | 6.43 | 0.35 | 0.47 | 0.23 | 0.05 |
| 360 oof imputed | Ligue 1 | 2021/2022 | male | 46198 | 26 | 49.47 | 38.62 | 7.24 | 0.65 | 6.29 | 0.34 | 0.45 | 0.20 | 0.04 |
| 360 oof imputed | Ligue 1 | 2022/2023 | male | 57857 | 31 | 49.34 | 38.44 | 7.26 | 0.65 | 6.25 | 0.34 | 0.46 | 0.19 | 0.04 |
| 360 oof imputed | Major League Soccer | 2023 | male | 4886 | 3 | 49.44 | 38.28 | 7.20 | 0.66 | 6.31 | 0.34 | 0.50 | 0.20 | 0.05 |
| 360 oof imputed | UEFA Euro | 2020 | male | 87113 | 51 | 48.26 | 37.06 | 7.07 | 0.60 | 6.90 | 0.32 | 0.46 | 0.19 | 0.05 |
| 360 oof imputed | UEFA Euro | 2024 | male | 87876 | 51 | 48.22 | 37.09 | 7.19 | 0.58 | 6.96 | 0.33 | 0.46 | 0.22 | 0.05 |
| 360 oof imputed | UEFA Women's Euro | 2022 | female | 46156 | 30 | 51.38 | 40.51 | 6.36 | 0.70 | 6.48 | 0.28 | 0.57 | 0.19 | 0.08 |
| 360 oof imputed | UEFA Women's Euro | 2025 | female | 45779 | 31 | 50.16 | 39.40 | 6.38 | 0.72 | 6.23 | 0.28 | 0.61 | 0.21 | 0.08 |
| 360 oof imputed | Women's World Cup | 2023 | female | 96369 | 64 | 50.71 | 39.88 | 6.32 | 0.74 | 6.13 | 0.28 | 0.60 | 0.19 | 0.08 |

## Caveats

- Labels are 360 freeze frames, not tracking: only players inside the broadcast visible area are present (mean 7.9 opponents visible; reliable share 73%, only 40% in the own third). Team-shape targets are therefore computed on the visible subset of the defence even on reliable frames, and the students learn that biased quantity. Ball-relative targets are less affected (the visible area is centred on the ball).
- The keeper-distance label is filtered on `y_keeper_consistent == 1` (0.34% of visible keepers sit at the wrong end of the pitch, ~114 yd, and a student trained on them extrapolated own-third events to ~90 yd); the keeper student is restricted to possession-team events in the attacking half (x >= 60) and the two cone students to possession-team events, where the cone towards the opponent goal is meaningful.
- The `nearest_opp_dist_in_cone` and `opp_keeper_dist_to_goal_line` students are trained only where the quantity exists (a non-empty cone / a visible keeper, mostly final-third events); applied elsewhere they extrapolate.
- `E2a` uses the realised pass end / carry end / duration, which are consequences of the defensive state (a carry ends where a defender is met); it is the after-the-fact analytics setting, not a pre-instant state model.
- Out-of-fold predictions of a student trained on the cap are what feed the shift tables and the later payoff stage; the final students (fitted on all matches with the median CV round count) are only applied to the non-360 matches.
- Training rows are thinned to the caps above; skills are therefore slightly below what the full 1.3M rows would give (the fit summary shows most students still gain a little at the round cap).
- The two binaries are defined from the same 360 targets (`def_line`, `n_opp_ahead_of_ball`); their threshold / third definitions are choices of this stage, fixed once and stored in the bundle.
- The MLP early-stopping split is a random 10% of the training rows (not match-grouped); it only chooses the epoch count, the test fold is untouched.
- `f_gender` is constant inside each cross-gender training set, so the transferred students never saw the target gender's value of that feature; `f_comp_type` (league / tournament) is constant (tournament) in the women's data.
