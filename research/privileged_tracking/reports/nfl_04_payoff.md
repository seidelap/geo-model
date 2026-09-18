# NFL 04 payoff: does imputed tracking state improve event-only outcome prediction?

Machine-written by `python -m research.privileged_tracking.nfl.payoff`. Question: within the tracked games and on untracked seasons, does adding imputed tracking state (NFL 03 students) to an nflfastR-style event-only feature set improve the prediction of observable outcomes, and how does that compare with the true tracking values (oracle) and with the official NGS participation fields?

## Protocol

* **Tasks.** `pass_completion` = completion on non-sack, non-spike pass plays (binary, target `complete_pass`); `pass_yards` = yards gained on non-sack, non-spike pass plays (reg, target `yards_gained`); `pass_epa` = EPA on non-sack, non-spike pass plays (reg, target `epa`); `run_yards` = yards gained on run plays (reg, target `yards_gained`); `run_success` = nflfastR success (EPA > 0) on run plays (binary, target `success`). Sacks and spikes are flagged by nflfastR (`sack`, `qb_spike` / `play_type`) and excluded from the pass tasks; throwaways are not flagged and stay in (nflfastR `cp` is NaN on them, so the `nflfastR cp` rows and the `cp` deltas cover the cp-present plays only).
* **Feature sets.** `PBP` = pre-snap situation (down, distance, yard line, quarter, clocks, score, `wp`, `ep`, `xpass`, shotgun / no-huddle, home) + strictly-prior-week team tendencies (`tend_*`, from full-season play-by-play) + at-release fields (pass: `air_yards`, `pass_length`, `pass_location`, `qb_hit`, receiver position group, `qb_scramble`; run: `run_location`, `run_gap`, `qb_scramble`). `PBP+PERS` adds the personnel counts and formation (participation charting). `PBP+IMP` adds the F0P out-of-fold pre-snap imputations (n_deep_safeties, mof_open, box_count_tuned, n_dl, def_y_std, cb_cushion, n_wide_left, n_wide_right, n_backfield, qb_depth, shotgun_derived, motion_derived) and the at-release F1T imputations of the within-play targets (pass: time_to_throw, pressure_derived, min_def_dist_qb_throw, n_pass_rushers_derived, separation_at_arrival, n_def_within_r_target, target_depth; run: box_count_run, min_def_dist_carrier_handoff, yards_to_first_contact, n_def_within_r_carrier_first_contact). `PBP+IMP_F0` is the strictly event-only variant (F0 / F0T students, no personnel input). `PBP+ORACLE_PRESNAP` adds the true tracking values of the 12 pre-snap targets, `PBP+ORACLE_WITHIN` the true values of the within-play targets and `PBP+ORACLE` both. The within-play targets are measured at or after the throw / handoff (separation and defenders near the target at ball arrival, time to throw and pressure at the release, yards to first contact and defenders at first contact of the carrier), so they are quasi-outcomes: the `PBP+ORACLE` / `PBP+ORACLE_WITHIN` rows bound a tracking-informed post-hoc description of the play, not what an at-release imputation could recover; `PBP+ORACLE_PRESNAP` is the relevant ceiling for pre-snap imputation. `PBP+NGS` adds personnel + the official participation fields (`defenders_in_box`, `number_of_pass_rushers`, `ngs_air_yards`, `was_pressure`, `time_to_throw`; the last two are absent on sacks, which are excluded anyway). `PBP+IMP+NGS` stacks both. `PBP+IMP_F1(leaky)` (stage a only) uses the NFL 03 F1 students, whose inputs include `complete_pass` / `yards_gained` / `epa`: it is shown to demonstrate the label leak, not as a payoff.
* **Why F1T.** The NFL 03 F1 students see the outcome of the play, so their within-play imputations cannot be features of an outcome model. F1T / F0T students are fitted here with the NFL 03 machinery (same LightGBM settings, same 5 game folds, team encodings fitted inside the training fold) on the F0P / F0 inputs plus the at-release fields only (`play_type_code`, `is_pass`, `is_run`, `qb_dropback`, `qb_scramble`, `pass_length`, `pass_location`, `air_yards`, `sack`, `qb_hit`, `receiver_group`, `run_location`, `run_gap`); their bundles are saved as `models/students_F1T.joblib` / `students_F0T.joblib`. `qb_hit` is a charted flag of a hit on the passer during the play, so 'at release' here means 'up to and including QB contact'; it sits in `PBP` and in the F1T / F0T inputs alike, so it cannot create an imputed-vs-PBP difference, but the pass_yards / EPA models do get a mild post-release hint from it.
* **Splits.** Stage a: 5-fold group k-fold by `gameId` (`common.splits.group_kfold`, seed 0, the NFL 03 folds), LightGBM with early stopping on an inner 20% game holdout of the training fold; every imputed feature of a test play is out-of-fold (its students never saw that game). Stages b-d: train 2018-2021, test 2022 once; the imputed features come from students trained on the 2017 tracked games only (a disjoint domain), the NFL 02 personnel probabilities are out-of-fold by game (2016-2020) or model predictions (2021-2022).
* **Statistics.** Metrics with n; paired bootstrap (1000 resamples, seed 0) of per-play log-loss (binary) or squared / absolute error (regression), play-level and game-clustered 95% CIs; deltas are `from` minus `to`, positive = `to` better. Quote the game-clustered CI. `base_global` = training-fold mean; `nflfastR cp` = the external completion-probability column.
* **Refit noise.** The bootstrap CIs condition on one fit. The `PBP`, `PBP+IMP` and `PBP+IMP_F0` models (stage a also `PBP+ORACLE_PRESNAP`; stage d: `PBP` and `PBP+IMP+IMP_PERS`) are refit under seeds 0, 1, 2 (the inner early-stopping holdout and the LightGBM bagging / feature-fraction seed change; folds and the 2018-2021 / 2022 split do not) and the spread max - min of their mean loss is tabulated in `nfl_04{a,b,d}_refit_noise.parquet`. Every deltas table carries `refit_spread` = the larger spread of the two members of the pair (`nflfastR cp` and `base_global` involve no fit and count as zero) and `refit_measured` = `both` / `from` / `to` / `none` saying which members were actually refit. A delta whose |value| is at or below the spread of a fully measured pair is reported as within refit noise even when its clustered CI excludes zero; when only one member was refit the floor is a lower bound and the verdict line says so.
* **Like with like against `cp`.** `nflfastR cp` is NaN on throwaways, so every completion table also scores the sets on the cp-present rows (`<set> [cp rows]`); only those rows and the `nflfastR cp -> <set>` deltas are comparable with the `cp` row.

## Summary

* **(a) Tracked games (n=6225 non-sack, non-spike pass plays and n=4671 run plays, 91 games).** Completion log-loss PBP 0.6078; adding the imputed state changes it by -0.0009 nats [-0.0042, +0.0023] (positive = better), the strictly event-only variant by -0.0002 nats [-0.0028, +0.0027]; the true tracking values of all targets (oracle) give +0.0565 nats [+0.0457, +0.0662], of which the 12 pre-snap targets alone give +0.0027 nats [+0.0000, +0.0054] and the within-play (post-release) targets alone +0.0578 nats [+0.0473, +0.0674]; the official NGS participation fields +0.0223 nats [+0.0167, +0.0276]. Run yards: pre-snap oracle -0.0227 yd^2 [-0.3302, +0.3193], within-play oracle +15.6654 yd^2 [+13.1471, +18.0900], full oracle +15.8459 yd^2 [+13.2730, +18.3801]. Yards / EPA / run tasks: imputed deltas all straddle zero, full-oracle deltas clearly positive on 4 of 4; the pre-snap oracle alone straddles zero on all of them (tables below). Like with like on the 6155 cp-present plays: nflfastR cp 0.5867 vs PBP 0.6040, i.e. cp is better by +0.0173 nats [+0.0133, +0.0215] (cp is fitted on many seasons incl. 2017: a reference, not a held-out competitor). The leaky F1 imputations reach log-loss 0.256: that is the label, not a payoff.
* **(b) Untracked seasons (train 2018-2021, test 2022, n=18096 pass plays / 271 games).** Completion log-loss PBP 0.5482, PBP+IMP 0.5492 (-0.0010 nats [-0.0024, +0.0004]), PBP+NGS 0.5126 (+0.0356 nats [+0.0316, +0.0397]). Like with like against nflfastR cp: cp is NaN on the 790 throwaways (all incompletions, on which PBP has log-loss 0.020), so on the 17306 cp-present plays cp scores 0.5702 vs PBP 0.5723 on the same rows: **nflfastR cp beats the PBP set by +0.0021 nats [+0.0006, +0.0036]** (the clustered CI excludes zero but the delta is inside the 0.0031-nat refit spread of the PBP model, so cp and the PBP set are within refit noise of each other: the PBP set is not better than cp); cp is fitted on these seasons, so it is a reference, not a held-out competitor. PBP+NGS 0.5358 on the same rows beats cp by +0.0345 nats [+0.0301, +0.0388] (positive = PBP+NGS better). The all-plays PBP figure 0.5482 must not be set next to the cp row. Imputed features add nothing out of domain on any of the five tasks (2 of the 10 PBP -> PBP+IMP / PBP+IMP_F0 deltas have a clustered CI excluding zero, 2 of those are within the refit spread of the same models and 0 are an improvement beyond it (pass_yards PBP -> PBP+IMP_F0 -0.4026 vs refit spread 0.5552; run_success PBP -> PBP+IMP_F0 -0.0021 vs refit spread 0.0024)); stacked on the NGS fields they are neutral or slightly harmful (EPA: -0.0344 EPA^2 [-0.0475, -0.0218]). The imputed values themselves drift little across seasons: max |z_shift| 0.16 over the F0P / F1T columns used by PBP+IMP and 0.28 over all feature sets and seasons (shift table). Refit spread of the PBP / PBP+IMP completion models across seeds: 0.0031 nats.
* **(c) Receiver-week vs NGS (2022 test receiver-weeks, n=1273).** Out-of-sample R2 for NGS `avg_separation`: naive event proxies (air yards, cp, share, n) 0.184; the fair after-the-fact baseline (naive + completion rate, YAC, yards, INT rate, QB-hit rate, EPA, the same fields the F1 student consumes) 0.316; that baseline + the at-release F1T student 0.371 (delta R2 +0.055 [+0.038, +0.072]), + the after-the-fact F1 student 0.412 (+0.096 [+0.072, +0.120]), + F0T 0.377 (+0.061 [+0.041, +0.079]; brackets = paired bootstrap 95% CI over the 2022 receiver-weeks, 2000 resamples). Against the naive-only baseline the F1 student would look like 0.184 -> 0.412, but that comparison mismatches information sets and overstates the payoff about twofold. At play level a linear regression on the outcome fields (air yards, QB hit, cp, completion, YAC, INT, completion x air yards) reproduces the F1 imputed separation with R2 0.67 on 2022 plays (at-release fields alone 0.34; the F1T imputation on the at-release fields 0.50), i.e. most of the F1 imputation is a linear re-encoding of the outcome flags, consistent with the NFL 03 `F1_vs_outcome` finding. For `avg_cushion` nothing works (best R2 0.044).
* **(d) Participation (2022 pass plays, n=18096 test plays, n_train=71762 2018-2021 plays with NFL 02 personnel probabilities).** Imputed personnel probabilities: +0.0002 nats [-0.0008, +0.0011]; true grouping: +0.0004 nats [-0.0006, +0.0013]; imputed -> true: +0.0002 nats [-0.0007, +0.0011]; imputed tracking state + imputed personnel: +0.0013 nats [+0.0001, +0.0026], the only stage-d pair whose clustered CI excludes zero, |delta| within the refit spread 0.0016 nats of the PBP / PBP+IMP+IMP_PERS models under other seeds; the PBP model alone moves by 0.0009 nats between stages b and d (0.5482 vs 0.5491, same 2022 rows, 570 training rows without personnel probabilities dropped). The offense personnel grouping, imputed or true (stage d tests the grouping probabilities, the true one-hot grouping, the position-group counts and the formation on completion and EPA only, not player identity), adds nothing measurable once the at-release fields are known.
* **Bottom line.** Perfectly known pre-snap tracking structure has no measurable play-level payoff once the at-release play-by-play fields are known: the pre-snap oracle (true values of the 12 pre-snap targets) changes completion log-loss by +0.0027 nats [+0.0000, +0.0054] (clustered CI excludes zero; 5% of the full-oracle gain) and run-yards squared error by -0.0227 yd^2 [-0.3302, +0.3193] (clustered CI straddles zero; 0% of the full-oracle gain). The full-oracle gains on every task come from the within-play targets (+0.0578 nats [+0.0473, +0.0674] on completion, +15.6654 yd^2 [+13.1471, +18.0900] on run yards): separation and defenders near the target at ball arrival, time to throw / pressure at the release, yards to first contact, all measured after the throw or handoff and therefore quasi-outcomes. No at-release student can recover them, and the oracle row is a ceiling for a tracking-informed post-hoc description of the play, not for any at-release imputation. Consistently, the event-only students add nothing on any task, in or out of domain: their imputations are functions of the same play-by-play fields the outcome model already sees, and LightGBM extracts that information itself. The only usable extra signal in untracked seasons is the official NGS charting (`was_pressure`, `time_to_throw`), which is after-the-fact. The nflfastR-style PBP set is itself no better than nflfastR's own `cp` on the plays where both exist (cp is marginally ahead, within refit noise); only the NGS charting beats `cp`. For after-the-fact player aggregates (receiver-week separation) the imputation is a modest real gain over the fair after-the-fact baseline (0.32 -> 0.41 R2 with the F1 student, -> 0.37 with the at-release F1T student), not a doubling, and most of the F1 signal is a re-encoding of outcome flags.

## (a) Tracked games 2017 (k-fold by game)

### pass_completion: completion on non-sack, non-spike pass plays

| model | n | log_loss | brier | bss | auc | delta_log_loss_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|---|
| PBP | 6225 | 0.6078 | 0.2096 | 0.1006 | 0.6863 |  |  |  |  |  |
| PBP+PERS | 6225 | 0.6080 | 0.2097 | 0.1001 | 0.6846 | -0.0002 | -0.0019 | 0.0018 | -0.0020 | 0.0015 |
| PBP+IMP | 6225 | 0.6088 | 0.2100 | 0.0988 | 0.6832 | -0.0009 | -0.0038 | 0.0020 | -0.0042 | 0.0023 |
| PBP+IMP_F0 | 6225 | 0.6080 | 0.2097 | 0.1002 | 0.6854 | -0.0002 | -0.0030 | 0.0027 | -0.0028 | 0.0027 |
| PBP+ORACLE_PRESNAP | 6225 | 0.6051 | 0.2084 | 0.1059 | 0.6905 | 0.0027 | 0.0004 | 0.0051 | 0.0000 | 0.0054 |
| PBP+ORACLE_WITHIN | 6225 | 0.5500 | 0.1851 | 0.2056 | 0.7687 | 0.0578 | 0.0501 | 0.0656 | 0.0473 | 0.0674 |
| PBP+ORACLE | 6225 | 0.5514 | 0.1858 | 0.2028 | 0.7670 | 0.0565 | 0.0489 | 0.0643 | 0.0457 | 0.0662 |
| PBP+NGS | 6225 | 0.5855 | 0.1998 | 0.1428 | 0.7200 | 0.0223 | 0.0172 | 0.0274 | 0.0167 | 0.0276 |
| PBP+IMP+NGS | 6225 | 0.5868 | 0.2002 | 0.1407 | 0.7183 | 0.0211 | 0.0160 | 0.0260 | 0.0158 | 0.0265 |
| PBP+IMP_F1(leaky) | 6225 | 0.2562 | 0.0769 | 0.6700 | 0.9601 | 0.3516 | 0.3343 | 0.3688 | 0.3183 | 0.3814 |
| base_global | 6225 | 0.6590 | 0.2331 | -0.0002 | 0.4888 | -0.0511 | -0.0588 | -0.0437 | -0.0582 | -0.0435 |
| nflfastR cp | 6155 | 0.5867 | 0.2002 | 0.1337 | 0.7116 | 0.0173 | 0.0137 | 0.0208 | 0.0133 | 0.0215 |
| PBP [cp rows] | 6155 | 0.6040 | 0.2078 | 0.1009 | 0.6866 |  |  |  |  |  |
| PBP+IMP [cp rows] | 6155 | 0.6056 | 0.2086 | 0.0977 | 0.6824 |  |  |  |  |  |
| PBP+IMP_F0 [cp rows] | 6155 | 0.6049 | 0.2083 | 0.0989 | 0.6844 |  |  |  |  |  |
| PBP+ORACLE_PRESNAP [cp rows] | 6155 | 0.6012 | 0.2066 | 0.1062 | 0.6910 |  |  |  |  |  |
| PBP+ORACLE [cp rows] | 6155 | 0.5502 | 0.1853 | 0.1982 | 0.7648 |  |  |  |  |  |
| PBP+NGS [cp rows] | 6155 | 0.5838 | 0.1990 | 0.1390 | 0.7170 |  |  |  |  |  |
| PBP+IMP+NGS [cp rows] | 6155 | 0.5851 | 0.1995 | 0.1367 | 0.7154 |  |  |  |  |  |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | log_loss | -0.0008 | -0.0035 | 0.0022 | -0.0038 | 0.0025 | 6225 | 91 |
| PBP+IMP_F0 | PBP+IMP | log_loss | -0.0007 | -0.0035 | 0.0019 | -0.0032 | 0.0014 | 6225 | 91 |
| PBP+IMP | PBP+ORACLE | log_loss | 0.0574 | 0.0493 | 0.0654 | 0.0456 | 0.0675 | 6225 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | log_loss | 0.0537 | 0.0457 | 0.0615 | 0.0432 | 0.0633 | 6225 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | log_loss | -0.0014 | -0.0036 | 0.0008 | -0.0037 | 0.0011 | 6225 | 91 |
| PBP+NGS | PBP+IMP+NGS | log_loss | -0.0012 | -0.0041 | 0.0016 | -0.0038 | 0.0014 | 6225 | 91 |
| nflfastR cp | PBP | log_loss | -0.0173 | -0.0208 | -0.0137 | -0.0215 | -0.0133 | 6155 | 91 |
| nflfastR cp | PBP+IMP | log_loss | -0.0189 | -0.0227 | -0.0147 | -0.0228 | -0.0151 | 6155 | 91 |

### pass_yards: yards gained on non-sack, non-spike pass plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| PBP | 6225 | 0.0898 | 6.5418 | 9.1583 |  |  |  |  |  |
| PBP+PERS | 6225 | 0.0839 | 6.5807 | 9.1882 | -0.5492 | -1.1236 | -0.0230 | -0.9920 | -0.0581 |
| PBP+IMP | 6225 | 0.0911 | 6.5634 | 9.1518 | 0.1195 | -0.7152 | 0.9298 | -0.6802 | 1.0339 |
| PBP+IMP_F0 | 6225 | 0.0827 | 6.5717 | 9.1943 | -0.6603 | -1.4618 | 0.0721 | -1.2963 | 0.0041 |
| PBP+ORACLE_PRESNAP | 6225 | 0.0947 | 6.5301 | 9.1339 | 0.4451 | -0.2709 | 1.1535 | -0.2177 | 1.1297 |
| PBP+ORACLE_WITHIN | 6225 | 0.1356 | 6.1844 | 8.9249 | 4.2200 | 2.6181 | 5.7302 | 2.7582 | 5.6841 |
| PBP+ORACLE | 6225 | 0.1411 | 6.1897 | 8.8968 | 4.7216 | 3.1514 | 6.2567 | 3.2966 | 6.0840 |
| PBP+NGS | 6225 | 0.0925 | 6.5285 | 9.1447 | 0.2491 | -0.4070 | 0.8716 | -0.3606 | 0.8621 |
| PBP+IMP+NGS | 6225 | 0.0911 | 6.5493 | 9.1519 | 0.1172 | -0.7977 | 0.9274 | -0.7464 | 0.9606 |
| PBP+IMP_F1(leaky) | 6225 | 0.6144 | 3.6496 | 5.9612 | 48.3379 | 43.3848 | 53.1917 | 42.9809 | 53.3849 |
| base_global | 6225 | -0.0004 | 6.8536 | 9.6014 | -8.3123 | -10.3362 | -6.3824 | -10.4474 | -6.4357 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | 0.6687 | -0.0850 | 1.4380 | -0.0368 | 1.5617 | 6225 | 91 |
| PBP+PERS | PBP+IMP | ae | 0.0173 | -0.0082 | 0.0409 | -0.0057 | 0.0426 | 6225 | 91 |
| PBP+IMP_F0 | PBP+IMP | se | 0.7798 | 0.1033 | 1.4689 | -0.1385 | 1.6717 | 6225 | 91 |
| PBP+IMP_F0 | PBP+IMP | ae | 0.0083 | -0.0134 | 0.0294 | -0.0165 | 0.0343 | 6225 | 91 |
| PBP+IMP | PBP+ORACLE | se | 4.6022 | 3.0213 | 6.0955 | 2.8893 | 6.1271 | 6225 | 91 |
| PBP+IMP | PBP+ORACLE | ae | 0.3738 | 0.3220 | 0.4228 | 0.3135 | 0.4331 | 6225 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | se | 4.2765 | 2.8740 | 5.6658 | 2.9148 | 5.6646 | 6225 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | ae | 0.3404 | 0.2918 | 0.3918 | 0.2858 | 0.3922 | 6225 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | se | 0.5016 | -0.2150 | 1.1825 | -0.2975 | 1.3100 | 6225 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | ae | -0.0053 | -0.0312 | 0.0196 | -0.0349 | 0.0257 | 6225 | 91 |
| PBP+NGS | PBP+IMP+NGS | se | -0.1319 | -0.8879 | 0.6244 | -1.0554 | 0.7987 | 6225 | 91 |
| PBP+NGS | PBP+IMP+NGS | ae | -0.0208 | -0.0454 | 0.0049 | -0.0463 | 0.0063 | 6225 | 91 |

### pass_epa: EPA on non-sack, non-spike pass plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| PBP | 6225 | 0.0140 | 1.1052 | 1.5455 |  |  |  |  |  |
| PBP+PERS | 6225 | 0.0170 | 1.1026 | 1.5431 | 0.0072 | -0.0009 | 0.0160 | -0.0010 | 0.0149 |
| PBP+IMP | 6225 | 0.0163 | 1.1026 | 1.5436 | 0.0056 | -0.0059 | 0.0174 | -0.0081 | 0.0198 |
| PBP+IMP_F0 | 6225 | 0.0122 | 1.1034 | 1.5468 | -0.0043 | -0.0163 | 0.0072 | -0.0156 | 0.0074 |
| PBP+ORACLE_PRESNAP | 6225 | 0.0124 | 1.1044 | 1.5467 | -0.0039 | -0.0141 | 0.0060 | -0.0148 | 0.0074 |
| PBP+ORACLE_WITHIN | 6225 | 0.0534 | 1.0521 | 1.5143 | 0.0954 | 0.0690 | 0.1204 | 0.0684 | 0.1232 |
| PBP+ORACLE | 6225 | 0.0534 | 1.0492 | 1.5143 | 0.0954 | 0.0678 | 0.1215 | 0.0681 | 0.1229 |
| PBP+NGS | 6225 | 0.0175 | 1.0973 | 1.5427 | 0.0086 | -0.0029 | 0.0197 | -0.0027 | 0.0199 |
| PBP+IMP+NGS | 6225 | 0.0186 | 1.0942 | 1.5419 | 0.0111 | -0.0042 | 0.0263 | -0.0028 | 0.0251 |
| PBP+IMP_F1(leaky) | 6225 | 0.4135 | 0.7587 | 1.1919 | 0.9678 | 0.8886 | 1.0529 | 0.8871 | 1.0485 |
| base_global | 6225 | -0.0003 | 1.1205 | 1.5566 | -0.0345 | -0.0491 | -0.0200 | -0.0497 | -0.0187 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | -0.0017 | -0.0147 | 0.0106 | -0.0150 | 0.0132 | 6225 | 91 |
| PBP+PERS | PBP+IMP | ae | -0.0000 | -0.0036 | 0.0030 | -0.0035 | 0.0039 | 6225 | 91 |
| PBP+IMP_F0 | PBP+IMP | se | 0.0099 | -0.0036 | 0.0233 | -0.0035 | 0.0237 | 6225 | 91 |
| PBP+IMP_F0 | PBP+IMP | ae | 0.0008 | -0.0026 | 0.0043 | -0.0032 | 0.0053 | 6225 | 91 |
| PBP+IMP | PBP+ORACLE | se | 0.0898 | 0.0615 | 0.1180 | 0.0599 | 0.1189 | 6225 | 91 |
| PBP+IMP | PBP+ORACLE | ae | 0.0534 | 0.0461 | 0.0609 | 0.0451 | 0.0613 | 6225 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | se | 0.0993 | 0.0740 | 0.1250 | 0.0715 | 0.1256 | 6225 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | ae | 0.0552 | 0.0482 | 0.0624 | 0.0477 | 0.0625 | 6225 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | se | 0.0000 | -0.0134 | 0.0124 | -0.0133 | 0.0128 | 6225 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | ae | 0.0029 | -0.0004 | 0.0063 | -0.0007 | 0.0069 | 6225 | 91 |
| PBP+NGS | PBP+IMP+NGS | se | 0.0025 | -0.0132 | 0.0179 | -0.0122 | 0.0170 | 6225 | 91 |
| PBP+NGS | PBP+IMP+NGS | ae | 0.0031 | -0.0007 | 0.0072 | -0.0007 | 0.0072 | 6225 | 91 |

### run_yards: yards gained on run plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| PBP | 4671 | 0.0245 | 3.8093 | 6.5174 |  |  |  |  |  |
| PBP+PERS | 4671 | 0.0219 | 3.8045 | 6.5260 | -0.1130 | -0.3362 | 0.1020 | -0.3039 | 0.0894 |
| PBP+IMP | 4671 | 0.0254 | 3.8057 | 6.5146 | 0.0361 | -0.2410 | 0.2938 | -0.2085 | 0.2860 |
| PBP+IMP_F0 | 4671 | 0.0247 | 3.8045 | 6.5169 | 0.0062 | -0.3309 | 0.3169 | -0.3094 | 0.3101 |
| PBP+ORACLE_PRESNAP | 4671 | 0.0240 | 3.8084 | 6.5191 | -0.0227 | -0.3517 | 0.3037 | -0.3302 | 0.3193 |
| PBP+ORACLE_WITHIN | 4671 | 0.3843 | 2.5541 | 5.1779 | 15.6654 | 13.2843 | 18.0921 | 13.1471 | 18.0900 |
| PBP+ORACLE | 4671 | 0.3884 | 2.5519 | 5.1604 | 15.8459 | 13.4667 | 18.2541 | 13.2730 | 18.3801 |
| PBP+NGS | 4671 | 0.0276 | 3.7871 | 6.5071 | 0.1336 | -0.0791 | 0.3446 | -0.0442 | 0.3154 |
| PBP+IMP+NGS | 4671 | 0.0269 | 3.8024 | 6.5093 | 0.1047 | -0.1978 | 0.4189 | -0.1761 | 0.3881 |
| PBP+IMP_F1(leaky) | 4671 | 0.8121 | 0.9714 | 2.8603 | 34.2949 | 29.5215 | 39.1430 | 30.2565 | 38.5371 |
| base_global | 4671 | -0.0002 | 3.9048 | 6.5993 | -1.0753 | -1.5537 | -0.6602 | -1.4191 | -0.6943 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | 0.1491 | -0.1000 | 0.3933 | -0.0648 | 0.3540 | 4671 | 91 |
| PBP+PERS | PBP+IMP | ae | -0.0012 | -0.0224 | 0.0176 | -0.0241 | 0.0215 | 4671 | 91 |
| PBP+IMP_F0 | PBP+IMP | se | 0.0299 | -0.2557 | 0.3126 | -0.2070 | 0.2574 | 4671 | 91 |
| PBP+IMP_F0 | PBP+IMP | ae | -0.0013 | -0.0210 | 0.0186 | -0.0222 | 0.0197 | 4671 | 91 |
| PBP+IMP | PBP+ORACLE | se | 15.8098 | 13.4657 | 18.2115 | 13.2521 | 18.3749 | 4671 | 91 |
| PBP+IMP | PBP+ORACLE | ae | 1.2538 | 1.1688 | 1.3371 | 1.1396 | 1.3577 | 4671 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | se | 15.8686 | 13.4578 | 18.3856 | 13.3035 | 18.4324 | 4671 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | ae | 1.2565 | 1.1702 | 1.3432 | 1.1524 | 1.3553 | 4671 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | se | 0.1806 | -0.2787 | 0.6255 | -0.2378 | 0.6233 | 4671 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | ae | 0.0022 | -0.0195 | 0.0225 | -0.0232 | 0.0278 | 4671 | 91 |
| PBP+NGS | PBP+IMP+NGS | se | -0.0289 | -0.3430 | 0.2774 | -0.2995 | 0.2215 | 4671 | 91 |
| PBP+NGS | PBP+IMP+NGS | ae | -0.0153 | -0.0349 | 0.0026 | -0.0376 | 0.0048 | 4671 | 91 |

### run_success: nflfastR success (EPA > 0) on run plays

| model | n | log_loss | brier | bss | auc | delta_log_loss_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|---|
| PBP | 4671 | 0.6432 | 0.2256 | 0.0425 | 0.6136 |  |  |  |  |  |
| PBP+PERS | 4671 | 0.6427 | 0.2255 | 0.0430 | 0.6149 | 0.0005 | -0.0020 | 0.0029 | -0.0021 | 0.0028 |
| PBP+IMP | 4671 | 0.6418 | 0.2250 | 0.0450 | 0.6209 | 0.0014 | -0.0016 | 0.0043 | -0.0021 | 0.0047 |
| PBP+IMP_F0 | 4671 | 0.6450 | 0.2264 | 0.0391 | 0.6084 | -0.0019 | -0.0049 | 0.0011 | -0.0052 | 0.0014 |
| PBP+ORACLE_PRESNAP | 4671 | 0.6422 | 0.2252 | 0.0445 | 0.6200 | 0.0010 | -0.0014 | 0.0033 | -0.0013 | 0.0033 |
| PBP+ORACLE_WITHIN | 4671 | 0.4574 | 0.1465 | 0.3783 | 0.8504 | 0.1857 | 0.1709 | 0.1998 | 0.1680 | 0.2033 |
| PBP+ORACLE | 4671 | 0.4567 | 0.1467 | 0.3777 | 0.8510 | 0.1864 | 0.1717 | 0.2010 | 0.1695 | 0.2034 |
| PBP+NGS | 4671 | 0.6428 | 0.2255 | 0.0432 | 0.6155 | 0.0003 | -0.0023 | 0.0029 | -0.0025 | 0.0032 |
| PBP+IMP+NGS | 4671 | 0.6423 | 0.2252 | 0.0444 | 0.6193 | 0.0009 | -0.0027 | 0.0039 | -0.0026 | 0.0044 |
| PBP+IMP_F1(leaky) | 4671 | 0.2115 | 0.0622 | 0.7360 | 0.9706 | 0.4317 | 0.4152 | 0.4473 | 0.4138 | 0.4479 |
| base_global | 4671 | 0.6643 | 0.2357 | -0.0002 | 0.4904 | -0.0211 | -0.0269 | -0.0155 | -0.0267 | -0.0159 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | log_loss | 0.0009 | -0.0019 | 0.0038 | -0.0022 | 0.0041 | 4671 | 91 |
| PBP+IMP_F0 | PBP+IMP | log_loss | 0.0032 | 0.0004 | 0.0062 | 0.0001 | 0.0062 | 4671 | 91 |
| PBP+IMP | PBP+ORACLE | log_loss | 0.1850 | 0.1703 | 0.1993 | 0.1687 | 0.2021 | 4671 | 91 |
| PBP+ORACLE_PRESNAP | PBP+ORACLE | log_loss | 0.1855 | 0.1702 | 0.1998 | 0.1687 | 0.2027 | 4671 | 91 |
| PBP+ORACLE_WITHIN | PBP+ORACLE | log_loss | 0.0007 | -0.0023 | 0.0039 | -0.0025 | 0.0042 | 4671 | 91 |
| PBP+NGS | PBP+IMP+NGS | log_loss | 0.0005 | -0.0027 | 0.0035 | -0.0026 | 0.0037 | 4671 | 91 |

### Calibration (10 equal-count bins, mean predicted vs observed)

`pass_completion` / `PBP`: 0.34->0.33, 0.45->0.43, 0.54->0.53, 0.60->0.61, 0.65->0.63, 0.69->0.68, 0.72->0.70, 0.74->0.76, 0.78->0.79, 0.83->0.84 (bins of 623)
`pass_completion` / `PBP+IMP`: 0.33->0.34, 0.44->0.42, 0.53->0.55, 0.60->0.62, 0.65->0.62, 0.68->0.68, 0.72->0.70, 0.75->0.75, 0.78->0.77, 0.83->0.85 (bins of 623)
`pass_completion` / `PBP+ORACLE_PRESNAP`: 0.34->0.31, 0.45->0.45, 0.54->0.54, 0.60->0.60, 0.64->0.61, 0.68->0.69, 0.72->0.71, 0.75->0.74, 0.78->0.81, 0.82->0.84 (bins of 623)
`pass_completion` / `PBP+ORACLE`: 0.25->0.27, 0.36->0.34, 0.46->0.44, 0.54->0.52, 0.62->0.63, 0.70->0.72, 0.77->0.77, 0.83->0.82, 0.87->0.86, 0.91->0.93 (bins of 623)
`pass_completion` / `PBP+NGS`: 0.29->0.28, 0.41->0.39, 0.52->0.51, 0.59->0.59, 0.64->0.64, 0.69->0.70, 0.74->0.71, 0.77->0.81, 0.81->0.81, 0.86->0.87 (bins of 623)
`pass_completion` / `nflfastR cp`: 0.30->0.29, 0.41->0.41, 0.52->0.53, 0.59->0.61, 0.65->0.65, 0.70->0.72, 0.74->0.73, 0.77->0.78, 0.80->0.79, 0.86->0.88 (bins of 616)
`run_success` / `PBP`: 0.24->0.25, 0.28->0.28, 0.30->0.34, 0.32->0.34, 0.34->0.33, 0.37->0.37, 0.39->0.37, 0.43->0.39, 0.50->0.54, 0.59->0.59 (bins of 468)
`run_success` / `PBP+IMP`: 0.25->0.25, 0.29->0.27, 0.31->0.32, 0.33->0.33, 0.35->0.34, 0.37->0.36, 0.39->0.40, 0.43->0.40, 0.48->0.53, 0.58->0.60 (bins of 468)
`run_success` / `PBP+ORACLE_PRESNAP`: 0.23->0.26, 0.28->0.26, 0.30->0.30, 0.32->0.36, 0.34->0.32, 0.37->0.37, 0.39->0.42, 0.43->0.39, 0.49->0.53, 0.59->0.60 (bins of 468)
`run_success` / `PBP+ORACLE`: 0.06->0.04, 0.09->0.12, 0.11->0.10, 0.15->0.16, 0.22->0.24, 0.33->0.33, 0.47->0.45, 0.63->0.61, 0.80->0.82, 0.91->0.93 (bins of 468)
`run_success` / `PBP+NGS`: 0.22->0.25, 0.26->0.31, 0.29->0.31, 0.32->0.32, 0.34->0.36, 0.36->0.35, 0.40->0.39, 0.44->0.40, 0.51->0.53, 0.61->0.59 (bins of 468)

### At-release students (F1T / F0T): out-of-fold skill on the tracked plays

Skill = R2 (continuous / count) or Brier skill (binary), k-fold OOF; `skill_F0P` / `skill_F1` are the NFL 03 students on the same rows (F1 sees the outcome), `skill_base_outcome` the per-outcome-class mean (also outcome-aware).

| target | subset | kind | student | n | skill | skill_F0P | skill_F1 | skill_base_outcome |
|---|---|---|---|---|---|---|---|---|
| time_to_throw | pass | reg | F1T | 6613 | 0.330 | 0.040 | 0.355 | 0.212 |
| pressure_derived | pass | binary | F1T | 6613 | 0.270 | 0.006 | 0.275 | 0.284 |
| min_def_dist_qb_throw | pass | reg | F1T | 6613 | 0.274 | 0.023 | 0.287 | 0.251 |
| n_pass_rushers_derived | pass | count | F1T | 6696 | 0.069 | 0.055 | 0.066 | 0.006 |
| separation_at_arrival | pass | reg | F1T | 6136 | 0.185 | 0.022 | 0.304 | 0.090 |
| n_def_within_r_target | pass | count | F1T | 6136 | 0.091 | 0.001 | 0.140 | 0.036 |
| target_depth | pass | reg | F1T | 6136 | 0.898 | 0.019 | 0.899 | 0.082 |
| box_count_run | run | count | F1T | 4671 | 0.435 | 0.432 | 0.432 | 0.031 |
| min_def_dist_carrier_handoff | run | reg | F1T | 4543 | 0.080 | 0.060 | 0.094 | -0.001 |
| yards_to_first_contact | run | reg | F1T | 4132 | -0.000 | -0.000 | 0.534 | 0.468 |
| n_def_within_r_carrier_first_contact | run | count | F1T | 4132 | 0.154 | 0.035 | 0.242 | 0.107 |
| time_to_throw | pass | reg | F0T | 6613 | 0.325 | 0.040 | 0.355 | 0.212 |
| pressure_derived | pass | binary | F0T | 6613 | 0.272 | 0.006 | 0.275 | 0.284 |
| min_def_dist_qb_throw | pass | reg | F0T | 6613 | 0.277 | 0.023 | 0.287 | 0.251 |
| n_pass_rushers_derived | pass | count | F0T | 6696 | 0.057 | 0.055 | 0.066 | 0.006 |
| separation_at_arrival | pass | reg | F0T | 6136 | 0.188 | 0.022 | 0.304 | 0.090 |
| n_def_within_r_target | pass | count | F0T | 6136 | 0.089 | 0.001 | 0.140 | 0.036 |
| target_depth | pass | reg | F0T | 6136 | 0.897 | 0.019 | 0.899 | 0.082 |
| box_count_run | run | count | F0T | 4671 | 0.357 | 0.432 | 0.432 | 0.031 |
| min_def_dist_carrier_handoff | run | reg | F0T | 4543 | 0.067 | 0.060 | 0.094 | -0.001 |
| yards_to_first_contact | run | reg | F0T | 4132 | -0.001 | -0.000 | 0.534 | 0.468 |
| n_def_within_r_carrier_first_contact | run | count | F0T | 4132 | 0.139 | 0.035 | 0.242 | 0.107 |

### Where the gain comes from (LightGBM gain share by feature group, mean over folds)

| task | model | imputed pre-snap | imputed within-play | ngs official | oracle pre-snap | oracle within-play | pbp | personnel |
|---|---|---|---|---|---|---|---|---|
| pass_completion | PBP |  |  |  |  |  | 1.000 |  |
| pass_completion | PBP+IMP | 0.153 | 0.446 |  |  |  | 0.401 |  |
| pass_completion | PBP+IMP+NGS | 0.128 | 0.201 | 0.343 |  |  | 0.319 | 0.009 |
| pass_completion | PBP+IMP_F0 | 0.162 | 0.462 |  |  |  | 0.375 |  |
| pass_completion | PBP+IMP_F1(leaky) | 0.063 | 0.672 |  |  |  | 0.265 |  |
| pass_completion | PBP+NGS |  |  | 0.426 |  |  | 0.554 | 0.020 |
| pass_completion | PBP+ORACLE |  |  |  | 0.068 | 0.591 | 0.341 |  |
| pass_completion | PBP+ORACLE_PRESNAP |  |  |  | 0.136 |  | 0.864 |  |
| pass_completion | PBP+ORACLE_WITHIN |  |  |  |  | 0.616 | 0.384 |  |
| pass_completion | PBP+PERS |  |  |  |  |  | 0.971 | 0.029 |
| pass_epa | PBP |  |  |  |  |  | 1.000 |  |
| pass_epa | PBP+IMP | 0.180 | 0.259 |  |  |  | 0.562 |  |
| pass_epa | PBP+IMP+NGS | 0.189 | 0.256 | 0.086 |  |  | 0.455 | 0.013 |
| pass_epa | PBP+IMP_F0 | 0.233 | 0.280 |  |  |  | 0.487 |  |
| pass_epa | PBP+IMP_F1(leaky) | 0.083 | 0.561 |  |  |  | 0.356 |  |
| pass_epa | PBP+NGS |  |  | 0.166 |  |  | 0.806 | 0.029 |
| pass_epa | PBP+ORACLE |  |  |  | 0.090 | 0.314 | 0.596 |  |
| pass_epa | PBP+ORACLE_PRESNAP |  |  |  | 0.137 |  | 0.863 |  |
| pass_epa | PBP+ORACLE_WITHIN |  |  |  |  | 0.339 | 0.661 |  |
| pass_epa | PBP+PERS |  |  |  |  |  | 0.969 | 0.031 |
| pass_yards | PBP |  |  |  |  |  | 1.000 |  |
| pass_yards | PBP+IMP | 0.146 | 0.176 |  |  |  | 0.678 |  |
| pass_yards | PBP+IMP+NGS | 0.140 | 0.185 | 0.079 |  |  | 0.592 | 0.003 |
| pass_yards | PBP+IMP_F0 | 0.160 | 0.177 |  |  |  | 0.663 |  |
| pass_yards | PBP+IMP_F1(leaky) | 0.047 | 0.549 |  |  |  | 0.404 |  |
| pass_yards | PBP+NGS |  |  | 0.142 |  |  | 0.845 | 0.013 |
| pass_yards | PBP+ORACLE |  |  |  | 0.065 | 0.290 | 0.645 |  |
| pass_yards | PBP+ORACLE_PRESNAP |  |  |  | 0.126 |  | 0.874 |  |
| pass_yards | PBP+ORACLE_WITHIN |  |  |  |  | 0.330 | 0.670 |  |
| pass_yards | PBP+PERS |  |  |  |  |  | 0.985 | 0.015 |
| run_success | PBP |  |  |  |  |  | 1.000 |  |
| run_success | PBP+IMP | 0.216 | 0.140 |  |  |  | 0.644 |  |
| run_success | PBP+IMP+NGS | 0.227 | 0.118 | 0.018 |  |  | 0.623 | 0.014 |
| run_success | PBP+IMP_F0 | 0.242 | 0.138 |  |  |  | 0.620 |  |
| run_success | PBP+IMP_F1(leaky) | 0.078 | 0.643 |  |  |  | 0.279 |  |
| run_success | PBP+NGS |  |  | 0.026 |  |  | 0.930 | 0.044 |
| run_success | PBP+ORACLE |  |  |  | 0.054 | 0.569 | 0.377 |  |
| run_success | PBP+ORACLE_PRESNAP |  |  |  | 0.183 |  | 0.817 |  |
| run_success | PBP+ORACLE_WITHIN |  |  |  |  | 0.587 | 0.413 |  |
| run_success | PBP+PERS |  |  |  |  |  | 0.946 | 0.054 |
| run_yards | PBP |  |  |  |  |  | 1.000 |  |
| run_yards | PBP+IMP | 0.326 | 0.183 |  |  |  | 0.491 |  |
| run_yards | PBP+IMP+NGS | 0.302 | 0.152 | 0.025 |  |  | 0.509 | 0.011 |
| run_yards | PBP+IMP_F0 | 0.308 | 0.197 |  |  |  | 0.494 |  |
| run_yards | PBP+IMP_F1(leaky) | 0.023 | 0.923 |  |  |  | 0.054 |  |
| run_yards | PBP+NGS |  |  | 0.041 |  |  | 0.923 | 0.036 |
| run_yards | PBP+ORACLE |  |  |  | 0.072 | 0.639 | 0.289 |  |
| run_yards | PBP+ORACLE_PRESNAP |  |  |  | 0.227 |  | 0.773 |  |
| run_yards | PBP+ORACLE_WITHIN |  |  |  |  | 0.658 | 0.342 |  |
| run_yards | PBP+PERS |  |  |  |  |  | 0.955 | 0.045 |

**Verdict (a).**

  - pass_completion: PBP -> PBP+IMP: -0.0009 log_loss [game CI -0.0042, +0.0023], no distinguishable change (n=6225)
  - pass_completion: PBP -> PBP+IMP_F0: -0.0002 log_loss [game CI -0.0028, +0.0027], no distinguishable change (n=6225)
  - pass_completion: PBP -> PBP+ORACLE_PRESNAP: +0.0027 log_loss [game CI +0.0000, +0.0054], better (clustered CI excludes 0 and |delta| > refit spread 0.0022 of the pair) (n=6225)
  - pass_completion: PBP -> PBP+ORACLE_WITHIN: +0.0578 log_loss [game CI +0.0473, +0.0674], better (clustered CI excludes 0 and |delta| > refit spread 0.0018 of `PBP` alone; other member not refit) (n=6225)
  - pass_completion: PBP -> PBP+ORACLE: +0.0565 log_loss [game CI +0.0457, +0.0662], better (clustered CI excludes 0 and |delta| > refit spread 0.0018 of `PBP` alone; other member not refit) (n=6225)
  - pass_completion: PBP+ORACLE_PRESNAP -> PBP+ORACLE: +0.0537 log_loss [game CI +0.0432, +0.0633], better (clustered CI excludes 0 and |delta| > refit spread 0.0022 of `PBP+ORACLE_PRESNAP` alone; other member not refit) (n=6225)
  - pass_completion: PBP -> PBP+NGS: +0.0223 log_loss [game CI +0.0167, +0.0276], better (clustered CI excludes 0 and |delta| > refit spread 0.0018 of `PBP` alone; other member not refit) (n=6225)
  - pass_completion: PBP+NGS -> PBP+IMP+NGS: -0.0012 log_loss [game CI -0.0038, +0.0014], no distinguishable change (n=6225)
  - pass_completion: nflfastR cp -> PBP: -0.0173 log_loss [game CI -0.0215, -0.0133], worse (clustered CI excludes 0 and |delta| > refit spread 0.0018 of the pair) (n=6155)
  - pass_completion: nflfastR cp -> PBP+NGS: +0.0029 log_loss [game CI -0.0029, +0.0085], no distinguishable change (n=6155)
  - pass_yards: PBP -> PBP+IMP: +0.1195 se [game CI -0.6802, +1.0339], no distinguishable change (n=6225)
  - pass_yards: PBP -> PBP+IMP_F0: -0.6603 se [game CI -1.2963, +0.0041], no distinguishable change (n=6225)
  - pass_yards: PBP -> PBP+ORACLE_PRESNAP: +0.4451 se [game CI -0.2177, +1.1297], no distinguishable change (n=6225)
  - pass_yards: PBP -> PBP+ORACLE_WITHIN: +4.2200 se [game CI +2.7582, +5.6841], better (clustered CI excludes 0 and |delta| > refit spread 1.8272 of `PBP` alone; other member not refit) (n=6225)
  - pass_yards: PBP -> PBP+ORACLE: +4.7216 se [game CI +3.2966, +6.0840], better (clustered CI excludes 0 and |delta| > refit spread 1.8272 of `PBP` alone; other member not refit) (n=6225)
  - pass_yards: PBP+ORACLE_PRESNAP -> PBP+ORACLE: +4.2765 se [game CI +2.9148, +5.6646], better (clustered CI excludes 0 and |delta| > refit spread 0.5711 of `PBP+ORACLE_PRESNAP` alone; other member not refit) (n=6225)
  - pass_yards: PBP -> PBP+NGS: +0.2491 se [game CI -0.3606, +0.8621], no distinguishable change (n=6225)
  - pass_yards: PBP+NGS -> PBP+IMP+NGS: -0.1319 se [game CI -1.0554, +0.7987], no distinguishable change (n=6225)
  - pass_epa: PBP -> PBP+IMP: +0.0056 se [game CI -0.0081, +0.0198], no distinguishable change (n=6225)
  - pass_epa: PBP -> PBP+IMP_F0: -0.0043 se [game CI -0.0156, +0.0074], no distinguishable change (n=6225)
  - pass_epa: PBP -> PBP+ORACLE_PRESNAP: -0.0039 se [game CI -0.0148, +0.0074], no distinguishable change (n=6225)
  - pass_epa: PBP -> PBP+ORACLE_WITHIN: +0.0954 se [game CI +0.0684, +0.1232], better (clustered CI excludes 0 and |delta| > refit spread 0.0064 of `PBP` alone; other member not refit) (n=6225)
  - pass_epa: PBP -> PBP+ORACLE: +0.0954 se [game CI +0.0681, +0.1229], better (clustered CI excludes 0 and |delta| > refit spread 0.0064 of `PBP` alone; other member not refit) (n=6225)
  - pass_epa: PBP+ORACLE_PRESNAP -> PBP+ORACLE: +0.0993 se [game CI +0.0715, +0.1256], better (clustered CI excludes 0 and |delta| > refit spread 0.0047 of `PBP+ORACLE_PRESNAP` alone; other member not refit) (n=6225)
  - pass_epa: PBP -> PBP+NGS: +0.0086 se [game CI -0.0027, +0.0199], no distinguishable change (n=6225)
  - pass_epa: PBP+NGS -> PBP+IMP+NGS: +0.0025 se [game CI -0.0122, +0.0170], no distinguishable change (n=6225)
  - run_yards: PBP -> PBP+IMP: +0.0361 se [game CI -0.2085, +0.2860], no distinguishable change (n=4671)
  - run_yards: PBP -> PBP+IMP_F0: +0.0062 se [game CI -0.3094, +0.3101], no distinguishable change (n=4671)
  - run_yards: PBP -> PBP+ORACLE_PRESNAP: -0.0227 se [game CI -0.3302, +0.3193], no distinguishable change (n=4671)
  - run_yards: PBP -> PBP+ORACLE_WITHIN: +15.6654 se [game CI +13.1471, +18.0900], better (clustered CI excludes 0 and |delta| > refit spread 0.1247 of `PBP` alone; other member not refit) (n=4671)
  - run_yards: PBP -> PBP+ORACLE: +15.8459 se [game CI +13.2730, +18.3801], better (clustered CI excludes 0 and |delta| > refit spread 0.1247 of `PBP` alone; other member not refit) (n=4671)
  - run_yards: PBP+ORACLE_PRESNAP -> PBP+ORACLE: +15.8686 se [game CI +13.3035, +18.4324], better (clustered CI excludes 0 and |delta| > refit spread 0.0882 of `PBP+ORACLE_PRESNAP` alone; other member not refit) (n=4671)
  - run_yards: PBP -> PBP+NGS: +0.1336 se [game CI -0.0442, +0.3154], no distinguishable change (n=4671)
  - run_yards: PBP+NGS -> PBP+IMP+NGS: -0.0289 se [game CI -0.2995, +0.2215], no distinguishable change (n=4671)
  - run_success: PBP -> PBP+IMP: +0.0014 log_loss [game CI -0.0021, +0.0047], no distinguishable change (n=4671)
  - run_success: PBP -> PBP+IMP_F0: -0.0019 log_loss [game CI -0.0052, +0.0014], no distinguishable change (n=4671)
  - run_success: PBP -> PBP+ORACLE_PRESNAP: +0.0010 log_loss [game CI -0.0013, +0.0033], no distinguishable change (n=4671)
  - run_success: PBP -> PBP+ORACLE_WITHIN: +0.1857 log_loss [game CI +0.1680, +0.2033], better (clustered CI excludes 0 and |delta| > refit spread 0.0030 of `PBP` alone; other member not refit) (n=4671)
  - run_success: PBP -> PBP+ORACLE: +0.1864 log_loss [game CI +0.1695, +0.2034], better (clustered CI excludes 0 and |delta| > refit spread 0.0030 of `PBP` alone; other member not refit) (n=4671)
  - run_success: PBP+ORACLE_PRESNAP -> PBP+ORACLE: +0.1855 log_loss [game CI +0.1687, +0.2027], better (clustered CI excludes 0 and |delta| > refit spread 0.0018 of `PBP+ORACLE_PRESNAP` alone; other member not refit) (n=4671)
  - run_success: PBP -> PBP+NGS: +0.0003 log_loss [game CI -0.0025, +0.0032], no distinguishable change (n=4671)
  - run_success: PBP+NGS -> PBP+IMP+NGS: +0.0005 log_loss [game CI -0.0026, +0.0037], no distinguishable change (n=4671)

Refit noise (k-fold OOF mean loss of the same model under three seeds; `spread` = max - min):

| task | model | loss | n_seeds | min_loss | max_loss | spread |
|---|---|---|---|---|---|---|
| pass_completion | PBP | log_loss | 3 | 0.6064 | 0.6082 | 0.0018 |
| pass_completion | PBP+IMP | log_loss | 3 | 0.6071 | 0.6088 | 0.0016 |
| pass_completion | PBP+IMP_F0 | log_loss | 3 | 0.6080 | 0.6082 | 0.0002 |
| pass_completion | PBP+ORACLE_PRESNAP | log_loss | 3 | 0.6051 | 0.6073 | 0.0022 |
| pass_yards | PBP | se | 3 | 83.8741 | 85.7013 | 1.8272 |
| pass_yards | PBP+IMP | se | 3 | 83.7546 | 84.4604 | 0.7058 |
| pass_yards | PBP+IMP_F0 | se | 3 | 84.5344 | 85.6162 | 1.0818 |
| pass_yards | PBP+ORACLE_PRESNAP | se | 3 | 83.4290 | 84.0001 | 0.5711 |
| pass_epa | PBP | se | 3 | 2.3866 | 2.3931 | 0.0064 |
| pass_epa | PBP+IMP | se | 3 | 2.3812 | 2.3953 | 0.0141 |
| pass_epa | PBP+IMP_F0 | se | 3 | 2.3840 | 2.3961 | 0.0121 |
| pass_epa | PBP+ORACLE_PRESNAP | se | 3 | 2.3920 | 2.3968 | 0.0047 |
| run_yards | PBP | se | 3 | 42.3513 | 42.4760 | 0.1247 |
| run_yards | PBP+IMP | se | 3 | 42.3626 | 42.4399 | 0.0773 |
| run_yards | PBP+IMP_F0 | se | 3 | 42.2225 | 42.5203 | 0.2978 |
| run_yards | PBP+ORACLE_PRESNAP | se | 3 | 42.4147 | 42.5029 | 0.0882 |
| run_success | PBP | log_loss | 3 | 0.6430 | 0.6459 | 0.0030 |
| run_success | PBP+IMP | log_loss | 3 | 0.6408 | 0.6436 | 0.0028 |
| run_success | PBP+IMP_F0 | log_loss | 3 | 0.6420 | 0.6450 | 0.0030 |
| run_success | PBP+ORACLE_PRESNAP | log_loss | 3 | 0.6412 | 0.6430 | 0.0018 |

## (b) Untracked seasons: train 2018-2021, test 2022

### pass_completion: completion on non-sack, non-spike pass plays

| model | n | log_loss | brier | bss | auc | delta_log_loss_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|---|
| base_global | 18096 | 0.6527 | 0.2300 | -0.0001 | 0.5000 | -0.1045 | -0.1100 | -0.0992 | -0.1100 | -0.0987 |
| PBP | 18096 | 0.5482 | 0.1850 | 0.1956 | 0.7415 |  |  |  |  |  |
| PBP+PERS | 18096 | 0.5482 | 0.1850 | 0.1957 | 0.7413 | -0.0000 | -0.0009 | 0.0008 | -0.0009 | 0.0009 |
| PBP+IMP | 18096 | 0.5492 | 0.1854 | 0.1938 | 0.7404 | -0.0010 | -0.0024 | 0.0004 | -0.0024 | 0.0004 |
| PBP+IMP_F0 | 18096 | 0.5484 | 0.1852 | 0.1950 | 0.7412 | -0.0002 | -0.0014 | 0.0010 | -0.0015 | 0.0010 |
| PBP+NGS | 18096 | 0.5126 | 0.1708 | 0.2576 | 0.7844 | 0.0356 | 0.0320 | 0.0393 | 0.0316 | 0.0397 |
| PBP+IMP+NGS | 18096 | 0.5138 | 0.1713 | 0.2552 | 0.7842 | 0.0344 | 0.0310 | 0.0377 | 0.0307 | 0.0379 |
| nflfastR cp | 17306 | 0.5702 | 0.1926 | 0.1281 | 0.7087 | 0.0021 | 0.0007 | 0.0035 | 0.0006 | 0.0036 |
| PBP [cp rows] | 17306 | 0.5723 | 0.1934 | 0.1243 | 0.7057 |  |  |  |  |  |
| PBP+IMP [cp rows] | 17306 | 0.5734 | 0.1939 | 0.1224 | 0.7044 |  |  |  |  |  |
| PBP+IMP_F0 [cp rows] | 17306 | 0.5723 | 0.1936 | 0.1237 | 0.7054 |  |  |  |  |  |
| PBP+NGS [cp rows] | 17306 | 0.5358 | 0.1786 | 0.1917 | 0.7545 |  |  |  |  |  |
| PBP+IMP+NGS [cp rows] | 17306 | 0.5370 | 0.1791 | 0.1890 | 0.7543 |  |  |  |  |  |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | log_loss | -0.0010 | -0.0023 | 0.0005 | -0.0022 | 0.0004 | 18096 | 271 |
| PBP+IMP_F0 | PBP+IMP | log_loss | -0.0008 | -0.0020 | 0.0004 | -0.0019 | 0.0002 | 18096 | 271 |
| PBP+NGS | PBP+IMP+NGS | log_loss | -0.0013 | -0.0026 | 0.0002 | -0.0028 | 0.0002 | 18096 | 271 |
| nflfastR cp | PBP | log_loss | -0.0021 | -0.0035 | -0.0007 | -0.0036 | -0.0006 | 17306 | 271 |
| nflfastR cp | PBP+IMP | log_loss | -0.0032 | -0.0047 | -0.0016 | -0.0048 | -0.0016 | 17306 | 271 |

### pass_yards: yards gained on non-sack, non-spike pass plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| base_global | 18096 | -0.0005 | 6.8223 | 9.4494 | -12.1459 | -13.4689 | -10.8395 | -13.4230 | -10.7938 |
| PBP | 18096 | 0.1356 | 6.1436 | 8.7833 |  |  |  |  |  |
| PBP+PERS | 18096 | 0.1345 | 6.1599 | 8.7889 | -0.0997 | -0.3805 | 0.1829 | -0.3836 | 0.2037 |
| PBP+IMP | 18096 | 0.1347 | 6.1457 | 8.7880 | -0.0839 | -0.4188 | 0.2402 | -0.4387 | 0.2642 |
| PBP+IMP_F0 | 18096 | 0.1311 | 6.1327 | 8.8062 | -0.4026 | -0.8586 | 0.0380 | -0.8530 | -0.0018 |
| PBP+NGS | 18096 | 0.1509 | 6.0082 | 8.7053 | 1.3634 | 0.9680 | 1.7619 | 0.9088 | 1.7855 |
| PBP+IMP+NGS | 18096 | 0.1484 | 6.0220 | 8.7183 | 1.1367 | 0.6874 | 1.5706 | 0.6984 | 1.5657 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | 0.0159 | -0.3315 | 0.3297 | -0.3130 | 0.3482 | 18096 | 271 |
| PBP+PERS | PBP+IMP | ae | 0.0143 | 0.0055 | 0.0232 | 0.0047 | 0.0235 | 18096 | 271 |
| PBP+IMP_F0 | PBP+IMP | se | 0.3187 | -0.1654 | 0.7764 | -0.1410 | 0.8053 | 18096 | 271 |
| PBP+IMP_F0 | PBP+IMP | ae | -0.0130 | -0.0252 | -0.0007 | -0.0264 | 0.0003 | 18096 | 271 |
| PBP+NGS | PBP+IMP+NGS | se | -0.2267 | -0.6580 | 0.1651 | -0.6340 | 0.1666 | 18096 | 271 |
| PBP+NGS | PBP+IMP+NGS | ae | -0.0139 | -0.0246 | -0.0033 | -0.0253 | -0.0023 | 18096 | 271 |

### pass_epa: EPA on non-sack, non-spike pass plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| base_global | 18096 | -0.0008 | 1.1188 | 1.5300 | -0.1084 | -0.1240 | -0.0932 | -0.1235 | -0.0946 |
| PBP | 18096 | 0.0455 | 1.0558 | 1.4941 |  |  |  |  |  |
| PBP+PERS | 18096 | 0.0433 | 1.0564 | 1.4959 | -0.0052 | -0.0129 | 0.0027 | -0.0144 | 0.0029 |
| PBP+IMP | 18096 | 0.0439 | 1.0541 | 1.4954 | -0.0038 | -0.0131 | 0.0060 | -0.0139 | 0.0069 |
| PBP+IMP_F0 | 18096 | 0.0483 | 1.0538 | 1.4920 | 0.0064 | 0.0002 | 0.0133 | -0.0003 | 0.0128 |
| PBP+NGS | 18096 | 0.0790 | 1.0292 | 1.4677 | 0.0783 | 0.0613 | 0.0968 | 0.0593 | 0.0966 |
| PBP+IMP+NGS | 18096 | 0.0642 | 1.0396 | 1.4794 | 0.0438 | 0.0298 | 0.0580 | 0.0297 | 0.0579 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | 0.0015 | -0.0091 | 0.0114 | -0.0086 | 0.0120 | 18096 | 271 |
| PBP+PERS | PBP+IMP | ae | 0.0023 | 0.0001 | 0.0047 | 0.0001 | 0.0046 | 18096 | 271 |
| PBP+IMP_F0 | PBP+IMP | se | -0.0102 | -0.0189 | -0.0010 | -0.0180 | -0.0016 | 18096 | 271 |
| PBP+IMP_F0 | PBP+IMP | ae | -0.0003 | -0.0023 | 0.0016 | -0.0024 | 0.0016 | 18096 | 271 |
| PBP+NGS | PBP+IMP+NGS | se | -0.0344 | -0.0471 | -0.0216 | -0.0475 | -0.0218 | 18096 | 271 |
| PBP+NGS | PBP+IMP+NGS | ae | -0.0104 | -0.0135 | -0.0072 | -0.0133 | -0.0074 | 18096 | 271 |

### run_yards: yards gained on run plays

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| base_global | 14377 | -0.0002 | 3.9127 | 6.3750 | -1.7800 | -2.0025 | -1.5687 | -2.0417 | -1.5195 |
| PBP | 14377 | 0.0436 | 3.7519 | 6.2338 |  |  |  |  |  |
| PBP+PERS | 14377 | 0.0425 | 3.7567 | 6.2375 | -0.0458 | -0.1307 | 0.0421 | -0.1387 | 0.0422 |
| PBP+IMP | 14377 | 0.0428 | 3.7398 | 6.2365 | -0.0331 | -0.1266 | 0.0571 | -0.1392 | 0.0641 |
| PBP+IMP_F0 | 14377 | 0.0434 | 3.7496 | 6.2343 | -0.0063 | -0.1001 | 0.0935 | -0.1102 | 0.0986 |
| PBP+NGS | 14377 | 0.0441 | 3.7487 | 6.2321 | 0.0210 | -0.0867 | 0.1272 | -0.1095 | 0.1463 |
| PBP+IMP+NGS | 14377 | 0.0454 | 3.7285 | 6.2280 | 0.0726 | -0.0312 | 0.1764 | -0.0502 | 0.1998 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | se | 0.0127 | -0.0952 | 0.1175 | -0.0970 | 0.1202 | 14377 | 271 |
| PBP+PERS | PBP+IMP | ae | 0.0170 | 0.0095 | 0.0245 | 0.0089 | 0.0264 | 14377 | 271 |
| PBP+IMP_F0 | PBP+IMP | se | -0.0268 | -0.1249 | 0.0675 | -0.1251 | 0.0673 | 14377 | 271 |
| PBP+IMP_F0 | PBP+IMP | ae | 0.0098 | 0.0027 | 0.0165 | 0.0020 | 0.0175 | 14377 | 271 |
| PBP+NGS | PBP+IMP+NGS | se | 0.0516 | -0.0503 | 0.1498 | -0.0592 | 0.1622 | 14377 | 271 |
| PBP+NGS | PBP+IMP+NGS | ae | 0.0202 | 0.0124 | 0.0278 | 0.0101 | 0.0310 | 14377 | 271 |

### run_success: nflfastR success (EPA > 0) on run plays

| model | n | log_loss | brier | bss | auc | delta_log_loss_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|---|
| base_global | 14377 | 0.6835 | 0.2452 | -0.0006 | 0.5000 | -0.0358 | -0.0401 | -0.0316 | -0.0401 | -0.0316 |
| PBP | 14377 | 0.6477 | 0.2281 | 0.0693 | 0.6465 |  |  |  |  |  |
| PBP+PERS | 14377 | 0.6481 | 0.2282 | 0.0686 | 0.6455 | -0.0004 | -0.0019 | 0.0011 | -0.0018 | 0.0010 |
| PBP+IMP | 14377 | 0.6486 | 0.2284 | 0.0678 | 0.6446 | -0.0009 | -0.0023 | 0.0004 | -0.0024 | 0.0005 |
| PBP+IMP_F0 | 14377 | 0.6498 | 0.2290 | 0.0654 | 0.6419 | -0.0021 | -0.0036 | -0.0007 | -0.0037 | -0.0005 |
| PBP+NGS | 14377 | 0.6471 | 0.2278 | 0.0704 | 0.6478 | 0.0006 | -0.0009 | 0.0021 | -0.0008 | 0.0020 |
| PBP+IMP+NGS | 14377 | 0.6473 | 0.2278 | 0.0703 | 0.6479 | 0.0004 | -0.0011 | 0.0019 | -0.0012 | 0.0019 |

Pairwise deltas (positive = `to` better):

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+PERS | PBP+IMP | log_loss | -0.0005 | -0.0022 | 0.0011 | -0.0021 | 0.0010 | 14377 | 271 |
| PBP+IMP_F0 | PBP+IMP | log_loss | 0.0012 | -0.0002 | 0.0026 | -0.0004 | 0.0026 | 14377 | 271 |
| PBP+NGS | PBP+IMP+NGS | log_loss | -0.0002 | -0.0014 | 0.0010 | -0.0015 | 0.0010 | 14377 | 271 |

### Calibration on 2022 (10 equal-count bins)

`pass_completion` / `PBP`: 0.18->0.17, 0.42->0.40, 0.53->0.54, 0.62->0.63, 0.68->0.69, 0.74->0.74, 0.77->0.77, 0.80->0.79, 0.84->0.83, 0.88->0.86 (bins of 1810)
`pass_completion` / `PBP+IMP`: 0.18->0.18, 0.41->0.40, 0.53->0.53, 0.62->0.62, 0.68->0.69, 0.73->0.75, 0.77->0.76, 0.80->0.81, 0.84->0.81, 0.88->0.86 (bins of 1810)
`pass_completion` / `PBP+NGS`: 0.12->0.12, 0.38->0.34, 0.51->0.51, 0.61->0.61, 0.68->0.71, 0.74->0.74, 0.79->0.79, 0.83->0.81, 0.87->0.89, 0.92->0.90 (bins of 1810)
`pass_completion` / `PBP+IMP+NGS`: 0.14->0.12, 0.39->0.35, 0.51->0.51, 0.61->0.62, 0.68->0.71, 0.73->0.74, 0.78->0.77, 0.82->0.82, 0.86->0.88, 0.91->0.91 (bins of 1810)
`pass_completion` / `nflfastR cp`: 0.32->0.31, 0.46->0.46, 0.57->0.57, 0.64->0.65, 0.70->0.71, 0.74->0.73, 0.77->0.78, 0.80->0.81, 0.84->0.83, 0.88->0.86 (bins of 1731)
`run_success` / `PBP`: 0.24->0.24, 0.30->0.31, 0.33->0.35, 0.35->0.36, 0.37->0.39, 0.40->0.41, 0.44->0.45, 0.50->0.50, 0.57->0.57, 0.70->0.70 (bins of 1438)
`run_success` / `PBP+IMP`: 0.26->0.25, 0.31->0.32, 0.33->0.35, 0.36->0.36, 0.38->0.38, 0.41->0.41, 0.44->0.44, 0.50->0.52, 0.57->0.57, 0.69->0.70 (bins of 1438)
`run_success` / `PBP+NGS`: 0.24->0.23, 0.30->0.32, 0.33->0.34, 0.35->0.37, 0.38->0.37, 0.41->0.42, 0.45->0.45, 0.50->0.51, 0.58->0.57, 0.70->0.71 (bins of 1438)
`run_success` / `PBP+IMP+NGS`: 0.25->0.25, 0.31->0.30, 0.33->0.34, 0.36->0.36, 0.38->0.39, 0.41->0.42, 0.44->0.44, 0.50->0.51, 0.57->0.58, 0.69->0.70 (bins of 1438)

### Gain share by feature group (2022 models)

| task | model | imputed pre-snap | imputed within-play | ngs official | pbp | personnel |
|---|---|---|---|---|---|---|
| pass_completion | PBP |  |  |  | 1.000 |  |
| pass_completion | PBP+IMP | 0.067 | 0.474 |  | 0.459 |  |
| pass_completion | PBP+IMP+NGS | 0.035 | 0.141 | 0.530 | 0.274 | 0.019 |
| pass_completion | PBP+IMP_F0 | 0.049 | 0.533 |  | 0.418 |  |
| pass_completion | PBP+NGS |  |  | 0.544 | 0.434 | 0.022 |
| pass_completion | PBP+PERS |  |  |  | 0.982 | 0.018 |
| pass_epa | PBP |  |  |  | 1.000 |  |
| pass_epa | PBP+IMP | 0.208 | 0.162 |  | 0.630 |  |
| pass_epa | PBP+IMP+NGS | 0.138 | 0.133 | 0.236 | 0.465 | 0.028 |
| pass_epa | PBP+IMP_F0 | 0.128 | 0.142 |  | 0.731 |  |
| pass_epa | PBP+NGS |  |  | 0.280 | 0.675 | 0.045 |
| pass_epa | PBP+PERS |  |  |  | 0.871 | 0.129 |
| pass_yards | PBP |  |  |  | 1.000 |  |
| pass_yards | PBP+IMP | 0.061 | 0.089 |  | 0.850 |  |
| pass_yards | PBP+IMP+NGS | 0.064 | 0.118 | 0.163 | 0.648 | 0.008 |
| pass_yards | PBP+IMP_F0 | 0.106 | 0.097 |  | 0.798 |  |
| pass_yards | PBP+NGS |  |  | 0.222 | 0.759 | 0.019 |
| pass_yards | PBP+PERS |  |  |  | 0.977 | 0.023 |
| run_success | PBP |  |  |  | 1.000 |  |
| run_success | PBP+IMP | 0.109 | 0.151 |  | 0.740 |  |
| run_success | PBP+IMP+NGS | 0.082 | 0.120 | 0.029 | 0.754 | 0.015 |
| run_success | PBP+IMP_F0 | 0.136 | 0.136 |  | 0.728 |  |
| run_success | PBP+NGS |  |  | 0.036 | 0.924 | 0.039 |
| run_success | PBP+PERS |  |  |  | 0.948 | 0.052 |
| run_yards | PBP |  |  |  | 1.000 |  |
| run_yards | PBP+IMP | 0.313 | 0.196 |  | 0.491 |  |
| run_yards | PBP+IMP+NGS | 0.274 | 0.172 | 0.033 | 0.509 | 0.012 |
| run_yards | PBP+IMP_F0 | 0.311 | 0.239 |  | 0.450 |  |
| run_yards | PBP+NGS |  |  | 0.063 | 0.914 | 0.022 |
| run_yards | PBP+PERS |  |  |  | 0.957 | 0.043 |

### Distribution shift of the imputed values, 2017 OOF vs 2018-2022

`z_shift` = (season mean - 2017 OOF mean) / 2017 OOF sd; `truth2017` = tracked truth. One row per target and season for the feature sets used by `PBP+IMP` (F0P pre-snap, F1T within-play); the full table (all sets) is in `nfl_04b_shift.parquet`.

| target | feature_set | oof2017_mean | oof2017_sd | truth2017_mean | z_2018 | z_2019 | z_2020 | z_2021 | z_2022 |
|---|---|---|---|---|---|---|---|---|---|
| box_count_run | F1T | 6.90 | 0.61 | 6.91 | -0.08 | -0.05 | -0.04 | 0.02 | -0.09 |
| box_count_tuned | F0P | 6.50 | 0.78 | 6.50 | -0.05 | -0.04 | -0.04 | 0.00 | -0.04 |
| cb_cushion | F0P | 4.43 | 1.10 | 4.44 | 0.07 | 0.10 | 0.06 | 0.04 | 0.08 |
| def_y_std | F0P | 7.72 | 0.99 | 7.71 | 0.04 | 0.03 | 0.03 | -0.03 | 0.03 |
| min_def_dist_carrier_handoff | F1T | 4.26 | 0.30 | 4.24 | -0.13 | -0.12 | -0.13 | -0.13 | -0.14 |
| min_def_dist_qb_throw | F1T | 2.78 | 0.71 | 2.78 | 0.04 | 0.02 | 0.08 | 0.09 | 0.08 |
| mof_open | F0P | 0.28 | 0.23 | 0.28 | -0.01 | 0.01 | -0.03 | -0.04 | -0.05 |
| motion_derived | F0P | 0.15 | 0.10 | 0.16 | 0.00 | 0.00 | -0.03 | -0.04 | -0.04 |
| n_backfield | F0P | 1.23 | 0.42 | 1.23 | -0.02 | -0.02 | -0.04 | -0.07 | -0.03 |
| n_deep_safeties | F0P | 1.22 | 0.50 | 1.21 | -0.04 | -0.02 | -0.08 | -0.08 | -0.07 |
| n_def_within_r_carrier_first_contact | F1T | 3.32 | 0.66 | 3.35 | -0.01 | -0.01 | 0.00 | 0.02 | -0.04 |
| n_def_within_r_target | F1T | 1.17 | 0.25 | 1.15 | 0.02 | 0.05 | 0.04 | 0.02 | 0.02 |
| n_dl | F0P | 5.24 | 0.47 | 5.24 | -0.04 | -0.05 | -0.03 | -0.01 | 0.00 |
| n_pass_rushers_derived | F1T | 4.26 | 0.26 | 4.27 | -0.04 | -0.02 | 0.02 | 0.00 | 0.03 |
| n_wide_left | F0P | 1.22 | 0.33 | 1.22 | 0.03 | 0.02 | 0.03 | -0.04 | 0.04 |
| n_wide_right | F0P | 1.16 | 0.37 | 1.16 | 0.07 | 0.06 | 0.07 | 0.04 | 0.08 |
| pressure_derived | F1T | 0.29 | 0.24 | 0.29 | -0.04 | -0.03 | -0.06 | -0.09 | -0.06 |
| qb_depth | F0P | 3.57 | 1.64 | 3.56 | 0.05 | 0.07 | 0.09 | 0.07 | 0.11 |
| separation_at_arrival | F1T | 2.98 | 0.99 | 3.01 | -0.06 | -0.08 | -0.08 | -0.06 | -0.04 |
| shotgun_derived | F0P | 0.58 | 0.48 | 0.58 | 0.08 | 0.10 | 0.12 | 0.11 | 0.16 |
| target_depth | F1T | 7.48 | 9.37 | 7.99 | 0.03 | 0.03 | 0.01 | -0.00 | -0.00 |
| time_to_throw | F1T | 2.79 | 0.66 | 2.75 | -0.03 | -0.03 | -0.08 | -0.08 | -0.05 |
| yards_to_first_contact | F1T | 1.01 | 0.54 | 0.99 | 0.05 | 0.03 | 0.02 | -0.01 | -0.02 |

**Verdict (b).**

  - pass_completion: PBP -> PBP+IMP: -0.0010 log_loss [game CI -0.0024, +0.0004], no distinguishable change (n=18096)
  - pass_completion: PBP -> PBP+IMP_F0: -0.0002 log_loss [game CI -0.0015, +0.0010], no distinguishable change (n=18096)
  - pass_completion: PBP -> PBP+NGS: +0.0356 log_loss [game CI +0.0316, +0.0397], better (clustered CI excludes 0 and |delta| > refit spread 0.0031 of `PBP` alone; other member not refit) (n=18096)
  - pass_completion: PBP+NGS -> PBP+IMP+NGS: -0.0013 log_loss [game CI -0.0028, +0.0002], no distinguishable change (n=18096)
  - pass_completion: PBP+PERS -> PBP+IMP: -0.0010 log_loss [game CI -0.0022, +0.0004], no distinguishable change (n=18096)
  - pass_completion: nflfastR cp -> PBP: -0.0021 log_loss [game CI -0.0036, -0.0006], CI excludes 0 (worse) but |delta| <= refit spread 0.0031 of the pair: within refit noise (n=17306)
  - pass_completion: nflfastR cp -> PBP+NGS: +0.0345 log_loss [game CI +0.0301, +0.0388], better (clustered CI excludes 0 and |delta| > refit spread 0.0000 of `nflfastR cp` alone; other member not refit) (n=17306)
  - pass_yards: PBP -> PBP+IMP: -0.0839 se [game CI -0.4387, +0.2642], no distinguishable change (n=18096)
  - pass_yards: PBP -> PBP+IMP_F0: -0.4026 se [game CI -0.8530, -0.0018], CI excludes 0 (worse) but |delta| <= refit spread 0.5552 of the pair: within refit noise (n=18096)
  - pass_yards: PBP -> PBP+NGS: +1.3634 se [game CI +0.9088, +1.7855], better (clustered CI excludes 0 and |delta| > refit spread 0.2952 of `PBP` alone; other member not refit) (n=18096)
  - pass_yards: PBP+NGS -> PBP+IMP+NGS: -0.2267 se [game CI -0.6340, +0.1666], no distinguishable change (n=18096)
  - pass_yards: PBP+PERS -> PBP+IMP: +0.0159 se [game CI -0.3130, +0.3482], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+IMP: -0.0038 se [game CI -0.0139, +0.0069], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+IMP_F0: +0.0064 se [game CI -0.0003, +0.0128], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+NGS: +0.0783 se [game CI +0.0593, +0.0966], better (clustered CI excludes 0 and |delta| > refit spread 0.0047 of `PBP` alone; other member not refit) (n=18096)
  - pass_epa: PBP+NGS -> PBP+IMP+NGS: -0.0344 se [game CI -0.0475, -0.0218], worse (clustered CI excludes 0; refit noise not measured for this pair) (n=18096)
  - pass_epa: PBP+PERS -> PBP+IMP: +0.0015 se [game CI -0.0086, +0.0120], no distinguishable change (n=18096)
  - run_yards: PBP -> PBP+IMP: -0.0331 se [game CI -0.1392, +0.0641], no distinguishable change (n=14377)
  - run_yards: PBP -> PBP+IMP_F0: -0.0063 se [game CI -0.1102, +0.0986], no distinguishable change (n=14377)
  - run_yards: PBP -> PBP+NGS: +0.0210 se [game CI -0.1095, +0.1463], no distinguishable change (n=14377)
  - run_yards: PBP+NGS -> PBP+IMP+NGS: +0.0516 se [game CI -0.0592, +0.1622], no distinguishable change (n=14377)
  - run_yards: PBP+PERS -> PBP+IMP: +0.0127 se [game CI -0.0970, +0.1202], no distinguishable change (n=14377)
  - run_success: PBP -> PBP+IMP: -0.0009 log_loss [game CI -0.0024, +0.0005], no distinguishable change (n=14377)
  - run_success: PBP -> PBP+IMP_F0: -0.0021 log_loss [game CI -0.0037, -0.0005], CI excludes 0 (worse) but |delta| <= refit spread 0.0024 of the pair: within refit noise (n=14377)
  - run_success: PBP -> PBP+NGS: +0.0006 log_loss [game CI -0.0008, +0.0020], no distinguishable change (n=14377)
  - run_success: PBP+NGS -> PBP+IMP+NGS: -0.0002 log_loss [game CI -0.0015, +0.0010], no distinguishable change (n=14377)
  - run_success: PBP+PERS -> PBP+IMP: -0.0005 log_loss [game CI -0.0021, +0.0010], no distinguishable change (n=14377)

Refit noise (2022 mean loss of the same model under three seeds; `spread` = max - min):

| task | model | loss | n_seeds | min_loss | max_loss | spread |
|---|---|---|---|---|---|---|
| pass_completion | PBP | log_loss | 3 | 0.5482 | 0.5513 | 0.0031 |
| pass_completion | PBP+IMP | log_loss | 3 | 0.5477 | 0.5492 | 0.0015 |
| pass_completion | PBP+IMP_F0 | log_loss | 3 | 0.5476 | 0.5487 | 0.0010 |
| pass_yards | PBP | se | 3 | 77.1458 | 77.4410 | 0.2952 |
| pass_yards | PBP+IMP | se | 3 | 77.2296 | 77.3941 | 0.1645 |
| pass_yards | PBP+IMP_F0 | se | 3 | 76.9932 | 77.5484 | 0.5552 |
| pass_epa | PBP | se | 3 | 2.2277 | 2.2324 | 0.0047 |
| pass_epa | PBP+IMP | se | 3 | 2.2287 | 2.2361 | 0.0075 |
| pass_epa | PBP+IMP_F0 | se | 3 | 2.2259 | 2.2358 | 0.0098 |
| run_yards | PBP | se | 3 | 38.8045 | 38.8603 | 0.0558 |
| run_yards | PBP+IMP | se | 3 | 38.8418 | 38.8934 | 0.0516 |
| run_yards | PBP+IMP_F0 | se | 3 | 38.8616 | 38.8943 | 0.0327 |
| run_success | PBP | log_loss | 3 | 0.6477 | 0.6501 | 0.0024 |
| run_success | PBP+IMP | log_loss | 3 | 0.6479 | 0.6486 | 0.0007 |
| run_success | PBP+IMP_F0 | log_loss | 3 | 0.6488 | 0.6498 | 0.0010 |

## (c) Receiver-week aggregation vs NGS weekly receiving (2018-2022)

Imputed separation (`separation_at_arrival`, students F1 = after the fact, F1T = at release, F0T = at release without personnel) and cushion (`cb_cushion`, F0P / F0) averaged over a receiver's targeted non-sack passes in a week, joined to the NGS weekly `avg_separation` / `avg_cushion` of the same season / week / gsis id. Naive event-only proxies: mean `air_yards`, mean nflfastR `cp`, target share (targets / team targets that week). After-the-fact event aggregates (the fields the F1 student consumes, so the fair after-the-fact baseline): completion rate, mean YAC per target (0 on incompletions), mean yards gained, interception rate, QB-hit rate, mean EPA. `within_player_pearson` demeans by (season, player). NGS weekly rows exist only for receivers with enough targets (min 5 in the file).

| quantity | value |
|---|---|
| receiver-weeks (event data, >=1 target) | 21223.000 |
| NGS weekly receiving rows 2018-2022 REG | 6449.000 |
| joined receiver-weeks | 6449.000 |
| joined share of NGS rows | 1.000 |
| mean \|n_targets - ngs_targets\| on joined rows | 0.040 |

Correlations over all joined receiver-weeks (every NGS weekly row joins, so no extra target threshold is applied):

| ngs_field | proxy | n | pearson | spearman | within_player_pearson | n_within |
|---|---|---|---|---|---|---|
| avg_separation | mean_imp_separation_at_arrival__F1 | 6449 | 0.590 | 0.574 | 0.537 | 6218 |
| avg_separation | mean_imp_separation_at_arrival__F1T | 6449 | 0.456 | 0.432 | 0.370 | 6218 |
| avg_separation | mean_imp_separation_at_arrival__F0T | 6449 | 0.480 | 0.458 | 0.387 | 6218 |
| avg_separation | mean_imp_cb_cushion__F0P | 6449 | 0.057 | 0.052 | 0.050 | 6218 |
| avg_separation | mean_imp_cb_cushion__F0 | 6449 | 0.061 | 0.058 | 0.057 | 6218 |
| avg_separation | mean_imp_target_depth__F1T | 6449 | -0.386 | -0.392 | -0.294 | 6218 |
| avg_separation | mean_air_yards | 6449 | -0.384 | -0.391 | -0.294 | 6218 |
| avg_separation | mean_cp | 6449 | 0.418 | 0.413 | 0.333 | 6218 |
| avg_separation | mean_complete_pass | 6449 | 0.369 | 0.365 | 0.344 | 6218 |
| avg_separation | mean_yac_per_target | 6449 | 0.408 | 0.461 | 0.372 | 6218 |
| avg_separation | mean_yards_gained | 6449 | 0.112 | 0.122 | 0.174 | 6218 |
| avg_separation | mean_interception | 6449 | -0.081 | -0.079 | -0.065 | 6218 |
| avg_separation | mean_qb_hit | 6449 | -0.070 | -0.067 | -0.059 | 6218 |
| avg_separation | mean_epa | 6449 | 0.116 | 0.122 | 0.148 | 6218 |
| avg_separation | target_share | 6449 | -0.033 | -0.015 | 0.000 | 6218 |
| avg_cushion | mean_imp_separation_at_arrival__F1 | 6447 | 0.146 | 0.130 | 0.140 | 6216 |
| avg_cushion | mean_imp_separation_at_arrival__F1T | 6447 | 0.178 | 0.162 | 0.171 | 6216 |
| avg_cushion | mean_imp_separation_at_arrival__F0T | 6447 | 0.181 | 0.165 | 0.176 | 6216 |
| avg_cushion | mean_imp_cb_cushion__F0P | 6447 | 0.153 | 0.150 | 0.149 | 6216 |
| avg_cushion | mean_imp_cb_cushion__F0 | 6447 | 0.163 | 0.155 | 0.157 | 6216 |
| avg_cushion | mean_imp_target_depth__F1T | 6447 | -0.140 | -0.143 | -0.122 | 6216 |
| avg_cushion | mean_air_yards | 6447 | -0.136 | -0.141 | -0.119 | 6216 |
| avg_cushion | mean_cp | 6447 | 0.187 | 0.180 | 0.156 | 6216 |
| avg_cushion | mean_complete_pass | 6447 | 0.099 | 0.096 | 0.093 | 6216 |
| avg_cushion | mean_yac_per_target | 6447 | 0.029 | 0.033 | 0.014 | 6216 |
| avg_cushion | mean_yards_gained | 6447 | -0.062 | -0.060 | -0.042 | 6216 |
| avg_cushion | mean_interception | 6447 | 0.006 | 0.002 | -0.003 | 6216 |
| avg_cushion | mean_qb_hit | 6447 | -0.020 | -0.024 | -0.024 | 6216 |
| avg_cushion | mean_epa | 6447 | -0.045 | -0.049 | -0.024 | 6216 |
| avg_cushion | target_share | 6447 | -0.032 | -0.028 | -0.010 | 6216 |

Out-of-sample proxy regressions: OLS of the NGS field on the listed receiver-week proxies, fitted on 2018-2021 receiver-weeks and scored on 2022 (`test_pearson`, `test_r2`). The naive model combines mean air yards, mean `cp`, target share and target count; `naive + after-the-fact` adds the after-the-fact aggregates; the imputed rows add the student means. The gain of a student is read against the row with the same event information (`naive + after-the-fact` for F1, which consumes the outcome; the F1T / F0T at-release students are shown against both). `delta_r2_vs_baseline` is the R2 gain over `naive + after-the-fact` on the same 2022 receiver-weeks with a paired bootstrap 95% CI (`ci_low` / `ci_high`, 2000 resamples of receiver-weeks, per-week squared errors).

| ngs_field | proxy_model | n_train | n_test | test_pearson | test_r2 | delta_r2_vs_baseline | ci_low | ci_high | baseline |
|---|---|---|---|---|---|---|---|---|---|
| avg_separation | naive (air yards, cp, share, n) | 5176 | 1273 | 0.434 | 0.184 | -0.132 | -0.168 | -0.096 | naive + after-the-fact |
| avg_separation | after-the-fact aggregates only | 5176 | 1273 | 0.562 | 0.307 | -0.009 | -0.018 | 0.001 | naive + after-the-fact |
| avg_separation | naive + after-the-fact | 5176 | 1273 | 0.569 | 0.316 |  |  |  | naive + after-the-fact |
| avg_separation | imputed F1T only | 5176 | 1273 | 0.510 | 0.252 | -0.064 | -0.107 | -0.019 | naive + after-the-fact |
| avg_separation | imputed F1 only | 5176 | 1273 | 0.642 | 0.406 | 0.090 | 0.056 | 0.124 | naive + after-the-fact |
| avg_separation | naive + imputed F0T | 5176 | 1273 | 0.529 | 0.278 | -0.038 | -0.078 | -0.000 | naive + after-the-fact |
| avg_separation | naive + imputed F1T | 5176 | 1273 | 0.521 | 0.267 | -0.049 | -0.087 | -0.011 | naive + after-the-fact |
| avg_separation | naive + imputed F1 | 5176 | 1273 | 0.646 | 0.412 | 0.096 | 0.063 | 0.128 | naive + after-the-fact |
| avg_separation | naive + after-the-fact + imputed F0T | 5176 | 1273 | 0.619 | 0.377 | 0.061 | 0.041 | 0.079 | naive + after-the-fact |
| avg_separation | naive + after-the-fact + imputed F1T | 5176 | 1273 | 0.616 | 0.371 | 0.055 | 0.038 | 0.072 | naive + after-the-fact |
| avg_separation | naive + after-the-fact + imputed F1 | 5176 | 1273 | 0.647 | 0.412 | 0.096 | 0.072 | 0.120 | naive + after-the-fact |
| avg_cushion | naive (air yards, cp, share, n) | 5174 | 1273 | 0.184 | 0.002 | -0.021 | -0.033 | -0.010 | naive + after-the-fact |
| avg_cushion | after-the-fact aggregates only | 5174 | 1273 | 0.187 | 0.001 | -0.022 | -0.033 | -0.012 | naive + after-the-fact |
| avg_cushion | naive + after-the-fact | 5174 | 1273 | 0.235 | 0.023 |  |  |  | naive + after-the-fact |
| avg_cushion | imputed F1T only | 5174 | 1273 | 0.224 | 0.018 | -0.005 | -0.030 | 0.019 | naive + after-the-fact |
| avg_cushion | imputed F1 only | 5174 | 1273 | 0.187 | 0.001 | -0.022 | -0.046 | 0.003 | naive + after-the-fact |
| avg_cushion | naive + imputed F0T | 5174 | 1273 | 0.241 | 0.028 | 0.005 | -0.015 | 0.026 | naive + after-the-fact |
| avg_cushion | naive + imputed F1T | 5174 | 1273 | 0.237 | 0.026 | 0.003 | -0.016 | 0.023 | naive + after-the-fact |
| avg_cushion | naive + imputed F1 | 5174 | 1273 | 0.225 | 0.020 | -0.003 | -0.022 | 0.016 | naive + after-the-fact |
| avg_cushion | naive + after-the-fact + imputed F0T | 5174 | 1273 | 0.272 | 0.044 | 0.021 | 0.004 | 0.039 | naive + after-the-fact |
| avg_cushion | naive + after-the-fact + imputed F1T | 5174 | 1273 | 0.269 | 0.042 | 0.019 | 0.004 | 0.035 | naive + after-the-fact |
| avg_cushion | naive + after-the-fact + imputed F1 | 5174 | 1273 | 0.262 | 0.038 | 0.015 | 0.000 | 0.031 | naive + after-the-fact |

Play-level linear re-encoding check: OLS of each imputed column on the at-release fields and on the outcome fields of the same play (fitted on 2018-2021 non-sack pass plays with a receiver, `r2_test` on 2022). A high R2 means the imputation is mostly a function of those fields.

| imputed_col | regressors | n_train | n_test | r2_train | r2_test |
|---|---|---|---|---|---|
| imp_separation_at_arrival__F1 | at-release fields (air yards, QB hit, cp) | 69679 | 17306 | 0.343 | 0.344 |
| imp_separation_at_arrival__F1T | at-release fields (air yards, QB hit, cp) | 69679 | 17306 | 0.482 | 0.497 |
| imp_separation_at_arrival__F0T | at-release fields (air yards, QB hit, cp) | 69679 | 17306 | 0.492 | 0.506 |
| imp_target_depth__F1T | at-release fields (air yards, QB hit, cp) | 69679 | 17306 | 0.989 | 0.989 |
| imp_separation_at_arrival__F1 | outcome fields (+ completion, YAC, INT, completion x air yards) | 69679 | 17306 | 0.674 | 0.675 |
| imp_separation_at_arrival__F1T | outcome fields (+ completion, YAC, INT, completion x air yards) | 69679 | 17306 | 0.522 | 0.534 |
| imp_separation_at_arrival__F0T | outcome fields (+ completion, YAC, INT, completion x air yards) | 69679 | 17306 | 0.535 | 0.545 |
| imp_target_depth__F1T | outcome fields (+ completion, YAC, INT, completion x air yards) | 69679 | 17306 | 0.989 | 0.989 |

**Verdict (c).** The student's contribution is the `naive + after-the-fact + imputed` row minus the `naive + after-the-fact` row (same event information on both sides); the `naive` -> `naive + imputed F1` difference mixes the student with the after-the-fact fields it consumes and overstates it. The summary above quotes the fair numbers.

## (d) Participation payoff on 2022 pass plays

`PBP+IMP_PERS` adds the NFL 02 offense-grouping probabilities (`p_off_*`, 10 classes; out-of-fold by game for 2016-2020, model predictions for 2021-2022), `PBP+TRUE_PERS` the one-hot true grouping, `+FORM` the true formation, `PBP+PERS` the true position-group counts + formation; `PBP+IMP+*` stack the imputed tracking state on top.

### pass_completion

| model | n | log_loss | brier | bss | auc | delta_log_loss_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|---|
| base_global | 18096 | 0.6527 | 0.2301 | -0.0001 | 0.5000 |  |  |  |  |  |
| PBP | 18096 | 0.5491 | 0.1854 | 0.1939 | 0.7395 |  |  |  |  |  |
| PBP+IMP_PERS | 18096 | 0.5489 | 0.1853 | 0.1943 | 0.7400 | 0.0002 | -0.0008 | 0.0010 | -0.0008 | 0.0011 |
| PBP+TRUE_PERS | 18096 | 0.5487 | 0.1853 | 0.1945 | 0.7404 | 0.0004 | -0.0006 | 0.0013 | -0.0006 | 0.0013 |
| PBP+TRUE_PERS+FORM | 18096 | 0.5487 | 0.1853 | 0.1945 | 0.7406 | 0.0004 | -0.0005 | 0.0013 | -0.0005 | 0.0012 |
| PBP+PERS | 18096 | 0.5484 | 0.1851 | 0.1954 | 0.7408 | 0.0007 | -0.0001 | 0.0014 | -0.0000 | 0.0015 |
| PBP+IMP+IMP_PERS | 18096 | 0.5478 | 0.1849 | 0.1962 | 0.7420 | 0.0013 | 0.0002 | 0.0025 | 0.0001 | 0.0026 |
| PBP+IMP+TRUE_PERS | 18096 | 0.5483 | 0.1851 | 0.1955 | 0.7416 | 0.0008 | -0.0004 | 0.0021 | -0.0004 | 0.0020 |

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+IMP_PERS | PBP+TRUE_PERS | log_loss | 0.0002 | -0.0007 | 0.0011 | -0.0007 | 0.0011 | 18096 | 271 |
| PBP+PERS | PBP+TRUE_PERS | log_loss | -0.0003 | -0.0013 | 0.0006 | -0.0012 | 0.0006 | 18096 | 271 |
| PBP+TRUE_PERS | PBP+TRUE_PERS+FORM | log_loss | 0.0000 | -0.0009 | 0.0010 | -0.0010 | 0.0010 | 18096 | 271 |
| PBP+IMP+IMP_PERS | PBP+IMP+TRUE_PERS | log_loss | -0.0005 | -0.0013 | 0.0003 | -0.0013 | 0.0002 | 18096 | 271 |

### pass_epa

| model | n | r2 | mae | rmse | delta_se_vs_PBP | ci_low | ci_high | ci_low_game | ci_high_game |
|---|---|---|---|---|---|---|---|---|---|
| base_global | 18096 | -0.0009 | 1.1190 | 1.5300 |  |  |  |  |  |
| PBP | 18096 | 0.0458 | 1.0574 | 1.4939 |  |  |  |  |  |
| PBP+IMP_PERS | 18096 | 0.0435 | 1.0560 | 1.4957 | -0.0053 | -0.0133 | 0.0032 | -0.0127 | 0.0027 |
| PBP+TRUE_PERS | 18096 | 0.0465 | 1.0564 | 1.4934 | 0.0017 | -0.0044 | 0.0077 | -0.0041 | 0.0067 |
| PBP+TRUE_PERS+FORM | 18096 | 0.0441 | 1.0570 | 1.4952 | -0.0038 | -0.0119 | 0.0041 | -0.0122 | 0.0042 |
| PBP+PERS | 18096 | 0.0436 | 1.0558 | 1.4956 | -0.0050 | -0.0132 | 0.0032 | -0.0141 | 0.0035 |
| PBP+IMP+IMP_PERS | 18096 | 0.0456 | 1.0581 | 1.4940 | -0.0004 | -0.0070 | 0.0056 | -0.0075 | 0.0066 |
| PBP+IMP+TRUE_PERS | 18096 | 0.0423 | 1.0574 | 1.4966 | -0.0080 | -0.0171 | 0.0012 | -0.0177 | 0.0016 |

| from | to | loss | delta | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| PBP+IMP_PERS | PBP+TRUE_PERS | se | 0.0070 | 0.0007 | 0.0137 | -0.0001 | 0.0131 | 18096 | 271 |
| PBP+IMP_PERS | PBP+TRUE_PERS | ae | -0.0003 | -0.0017 | 0.0011 | -0.0019 | 0.0010 | 18096 | 271 |
| PBP+PERS | PBP+TRUE_PERS | se | 0.0067 | -0.0016 | 0.0145 | -0.0021 | 0.0149 | 18096 | 271 |
| PBP+PERS | PBP+TRUE_PERS | ae | -0.0006 | -0.0023 | 0.0012 | -0.0023 | 0.0014 | 18096 | 271 |
| PBP+TRUE_PERS | PBP+TRUE_PERS+FORM | se | -0.0055 | -0.0129 | 0.0023 | -0.0126 | 0.0021 | 18096 | 271 |
| PBP+TRUE_PERS | PBP+TRUE_PERS+FORM | ae | -0.0006 | -0.0024 | 0.0012 | -0.0022 | 0.0009 | 18096 | 271 |
| PBP+IMP+IMP_PERS | PBP+IMP+TRUE_PERS | se | -0.0077 | -0.0150 | -0.0001 | -0.0155 | -0.0001 | 18096 | 271 |
| PBP+IMP+IMP_PERS | PBP+IMP+TRUE_PERS | ae | 0.0006 | -0.0010 | 0.0023 | -0.0009 | 0.0022 | 18096 | 271 |

| task | model | imputed personnel | imputed pre-snap | imputed within-play | pbp | personnel | true personnel |
|---|---|---|---|---|---|---|---|
| pass_completion | PBP |  |  |  | 1.000 |  |  |
| pass_completion | PBP+IMP+IMP_PERS | 0.027 | 0.045 | 0.440 | 0.487 |  |  |
| pass_completion | PBP+IMP+TRUE_PERS |  | 0.058 | 0.445 | 0.497 |  | 0.001 |
| pass_completion | PBP+IMP_PERS | 0.047 |  |  | 0.953 |  |  |
| pass_completion | PBP+PERS |  |  |  | 0.988 | 0.012 |  |
| pass_completion | PBP+TRUE_PERS |  |  |  | 0.996 |  | 0.004 |
| pass_completion | PBP+TRUE_PERS+FORM |  |  |  | 0.989 | 0.008 | 0.003 |
| pass_epa | PBP |  |  |  | 1.000 |  |  |
| pass_epa | PBP+IMP+IMP_PERS | 0.052 | 0.120 | 0.140 | 0.688 |  |  |
| pass_epa | PBP+IMP+TRUE_PERS |  | 0.155 | 0.159 | 0.685 |  | 0.001 |
| pass_epa | PBP+IMP_PERS | 0.120 |  |  | 0.880 |  |  |
| pass_epa | PBP+PERS |  |  |  | 0.879 | 0.121 |  |
| pass_epa | PBP+TRUE_PERS |  |  |  | 0.998 |  | 0.002 |
| pass_epa | PBP+TRUE_PERS+FORM |  |  |  | 0.900 | 0.073 | 0.027 |

**Verdict (d).**

  - pass_completion: PBP -> PBP+IMP_PERS: +0.0002 log_loss [game CI -0.0008, +0.0011], no distinguishable change (n=18096)
  - pass_completion: PBP -> PBP+TRUE_PERS: +0.0004 log_loss [game CI -0.0006, +0.0013], no distinguishable change (n=18096)
  - pass_completion: PBP -> PBP+IMP+IMP_PERS: +0.0013 log_loss [game CI +0.0001, +0.0026], CI excludes 0 (better) but |delta| <= refit spread 0.0016 of the pair: within refit noise (n=18096)
  - pass_completion: PBP+IMP_PERS -> PBP+TRUE_PERS: +0.0002 log_loss [game CI -0.0007, +0.0011], no distinguishable change (n=18096)
  - pass_completion: PBP+IMP+IMP_PERS -> PBP+IMP+TRUE_PERS: -0.0005 log_loss [game CI -0.0013, +0.0002], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+IMP_PERS: -0.0053 se [game CI -0.0127, +0.0027], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+TRUE_PERS: +0.0017 se [game CI -0.0041, +0.0067], no distinguishable change (n=18096)
  - pass_epa: PBP -> PBP+IMP+IMP_PERS: -0.0004 se [game CI -0.0075, +0.0066], no distinguishable change (n=18096)
  - pass_epa: PBP+IMP_PERS -> PBP+TRUE_PERS: +0.0070 se [game CI -0.0001, +0.0131], no distinguishable change (n=18096)
  - pass_epa: PBP+IMP+IMP_PERS -> PBP+IMP+TRUE_PERS: -0.0077 se [game CI -0.0155, -0.0001], worse (clustered CI excludes 0 and |delta| > refit spread 0.0046 of `PBP+IMP+IMP_PERS` alone; other member not refit) (n=18096)

Refit noise (2022 mean loss of the same model under three seeds; `spread` = max - min):

| task | model | loss | n_seeds | min_loss | max_loss | spread |
|---|---|---|---|---|---|---|
| pass_completion | PBP | log_loss | 3 | 0.5485 | 0.5501 | 0.0016 |
| pass_completion | PBP+IMP+IMP_PERS | log_loss | 3 | 0.5478 | 0.5486 | 0.0008 |
| pass_epa | PBP | se | 3 | 2.2293 | 2.2326 | 0.0033 |
| pass_epa | PBP+IMP+IMP_PERS | se | 3 | 2.2321 | 2.2367 | 0.0046 |

## Caveats

* The NFL 03 F1 students are inadmissible as outcome-model features (their inputs contain the label); the `PBP+IMP_F1(leaky)` row in stage a quantifies the leak and must not be read as a payoff. The F1T / F0T students used instead see the play only up to the release / handoff.
* Stage a uses the NFL 03 folds, so every test-fold imputation comes from students that never saw that game; the training-fold imputations were produced by students fitted on the other folds (the usual stacking protocol), a second-order optimism that affects the `PBP+IMP*` rows only and, given their null result, cannot change the conclusion.
* The imputed features are deterministic functions of the event inputs already in `PBP` (+ personnel), so `PBP+IMP` can only help through the representation the students learned from tracking targets (learning with privileged information); the tables measure exactly that channel. In stage b the paired CIs are game-clustered over 271 games.
* The bootstrap CIs condition on a single fit; the refit-noise tables (`nfl_04{a,b,d}_refit_noise.parquet`, three seeds) give the spread of the mean loss of the same model, and the verdict lines flag a delta as within refit noise when |delta| is at or below the larger spread of the two models of the pair (`refit_measured` = `both`); pairs with only one refit member are labelled as such and pairs with none keep the plain CI verdict. The `PBP` model also differs between stages b and d (identical 2022 rows, 570 fewer training rows in d because plays without NFL 02 personnel probabilities are dropped), which is the same order of magnitude.
* The oracle rows are not a ceiling for at-release imputation: `PBP+ORACLE_WITHIN` / `PBP+ORACLE` contain within-play targets measured at or after the throw / handoff (separation at arrival, defenders near the target, yards to first contact), which are quasi-outcomes of the play; `PBP+ORACLE_PRESNAP` is the ceiling that matters for pre-snap imputation. `qb_hit` (a charted post-play flag of a hit on the passer) is part of the at-release set on every side of every comparison; 'at release' therefore means 'up to and including QB contact'.
* `nflfastR cp` is NaN on throwaways (about 4% of non-sack pass plays); its metrics row and the `cp` deltas cover the cp-present plays only. Compare it only with the `<set> [cp rows]` rows: the all-plays rows include the throwaways, on which every model predicts a near-zero completion probability and so scores a near-zero loss, which lowers their average.
* Stage c: the after-the-fact aggregates use the outcome of the aggregated targets by design (an after-the-fact analytics setting, not prediction); the F1 student consumes the same fields, so its gain is quoted against `naive + after-the-fact`. Mean YAC per target counts incompletions as 0 yards (so every receiver-week has a value); averaging YAC over completions only and dropping receiver-weeks without a completion gives a slightly lower baseline on slightly fewer weeks. The play-level regressions in `nfl_04c_imputation_vs_outcome.parquet` show how much of each imputation those fields reproduce.
* The `PBP+IMP` set includes the F0P pre-snap students, whose inputs are the participation personnel / formation strings; `PBP+IMP_F0` is the strictly event-only variant and `PBP+PERS` isolates the personnel information itself.
* Team target encodings inside the students are 2017 values; tendencies in `PBP` are strictly-prior weeks of the same season (week 1 uses the constant priors). nflfastR `cp` is a model fitted on many seasons (including these), so it is a reference, not a held-out competitor; it is NaN on throwaways.
* Stage b/d LightGBM uses learning rate 0.1 with early stopping on an inner 20% game holdout of 2018-2021 (the 80% model is scored on 2022, no refit); stage a uses the NFL 03 settings (0.05). No subsampling anywhere.
* Stage c joins on gsis id / season / week; NGS weekly rows cover only receivers with >= 5 targets and use NGS' own target count (`ngs_targets`), which differs slightly from the nflfastR count of targeted passes.
* Run times: see the console log.
