# NFL 02: participation as a structured latent variable

Question: how well can the tracking-derived participation labels in the nflverse files (personnel groupings, box count, pass rushers, coverage, the 22 participant ids) be inferred from play-by-play situation plus team and player tendencies computed from strictly earlier games?

Protocol: regular-season scrimmage plays with a non-null `offense_personnel` (special teams, kneels and spikes excluded; `no_play` penalty snaps kept because personnel is pre-snap). Train seasons 2016-2020, validation 2021 (all settings: `num_leaves` and early-stopping rounds), test 2022 (reported once). LightGBM, `n_jobs=2`, lr=0.05, `min_data_in_leaf=100`, feature/bagging fraction 0.8, max 1500 rounds, early stopping 50. Paired bootstrap CIs (1000 resamples) are reported twice: over plays (`ci_low` / `ci_high`) and game-clustered (`ci_low_game` / `ci_high_game`: whole games resampled with replacement). Plays within a game are correlated, so the clustered interval is the one to trust; the play-level one is kept for comparison. Seeds fixed (0).

## Headline results (test season 2022)

- **Offense personnel grouping** (10 classes, n=35969 plays): S2 accuracy 0.647 / log-loss 0.984 / top-2 0.831, against majority 0.619 / 1.278 and the best tendency baseline (team x down-distance, prior games) 0.626 / 1.116. Situation alone (S0) gives 1.156, shotgun/no-huddle (S1) 1.126; tendencies are the largest step (0.142 nats). Knowing pass/run post hoc adds only 0.0013 nats. The grouping is mostly *not* recoverable from pbp: the model is essentially a calibrated '11 personnel unless the team/situation says 12 or 21' prior.
- With the test season's labels blanked entirely (tendencies from earlier seasons only, the 'labels disappear in-season' scenario) S2 drops to accuracy 0.630 / log-loss 1.085.
- **defense_personnel** (n=35969, 271 games): accuracy / log-loss: team-prior baseline 0.448 / 1.482; S2 0.506 / 1.257; + imputed offense personnel 0.514 / 1.243; + true offense personnel 0.579 / 1.121. Paired-bootstrap gain over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true 0.1369 [0.1311, 0.1435; 0.1214, 0.1528], imputed 0.0149 [0.0114, 0.0182; 0.0055, 0.0238].
- **defenders_in_box** (n=35090, 271 games): MAE / R2: team-prior baseline 0.791 / 0.015; S2 0.593 / 0.453; + imputed 0.594 / 0.453; + true 0.563 / 0.514. Paired-bootstrap gain over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true 0.0665 [0.0620, 0.0713; 0.0596, 0.0737], imputed -0.0004 [-0.0023, 0.0016; -0.0039, 0.0028].
- **number_of_pass_rushers** (n=20186, 271 games): MAE / R2: team-prior baseline 0.555 / 0.004; S2 0.535 / 0.059; + imputed 0.533 / 0.061; + true 0.535 / 0.067. Paired-bootstrap gain over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true 0.0049 [0.0021, 0.0081; 0.0019, 0.0078], imputed 0.0015 [-0.0002, 0.0032; -0.0007, 0.0035].
- **man_zone** (n=17979, 271 games): accuracy / log-loss: team-prior baseline 0.712 / 0.593; S2 0.749 / 0.530; + imputed offense personnel 0.749 / 0.530; + true offense personnel 0.750 / 0.527. Paired-bootstrap gain over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true 0.0030 [0.0021, 0.0040; 0.0018, 0.0041], imputed 0.0001 [-0.0008, 0.0009; -0.0013, 0.0014].
- **coverage_type** (n=17979, 271 games): accuracy / log-loss: team-prior baseline 0.323 / 1.643; S2 0.395 / 1.511; + imputed offense personnel 0.395 / 1.511; + true offense personnel 0.394 / 1.510. Paired-bootstrap gain over S2 as mean [play-level 95% CI; game-clustered 95% CI]: true 0.0014 [-0.0004, 0.0033; -0.0016, 0.0040], imputed -0.0000 [-0.0021, 0.0020; -0.0031, 0.0031].
- **Marginalisation (d)**: true offense personnel matters a lot for the defense's personnel and box count and nothing for pass-rush count, man/zone or coverage type; the imputed grouping recovers only a small fraction of that gain (judge the small `S2 -> S2+imp_pers` deltas by the game-clustered CI), so a downstream team-level model should not expect the participation step to add much unless the grouping is observed.
- **Player-level participation (c)**: offense: per-slot accuracy 0.834 (exact lineup 0.136) with the true grouping, 0.811 (0.090) with the imputed grouping, prior-usage-rank baseline 0.816 (0.111); defense: per-slot accuracy 0.783 (exact lineup 0.063) with the true grouping, 0.765 (0.036) with the imputed grouping, prior-usage-rank baseline 0.774 (0.053). Misses concentrate in DL rotation, LB, nickel/dime DBs, WR3+, RB2+ and TE2+; RB1 / TE1 / WR1-2 / CB1-2 are almost never missed but are not deterministic either (on-field rates well below 1 for RB1 and TE1). The candidate set is the 48-man game-day active list (weekly roster status `ACT`; declared inactives carry `INA`), so the residual QB / OL errors are not unknown inactives. The depth-chart QB1 is on the ACT list on 91.3% of test plays and is the QB on the field on 91.1% of them; 4.3% of those plays had him dressed but never taking a snap (stale depth chart / listed non-starter), and when he did play that day he is off the field on 4.8% of plays (in-game changes). On 7.5% of plays the listed QB1 was declared inactive and on 1.2% he was on a reserve list or missing from the depth chart; the weekly roster status resolves those pre-kickoff.

## Data

| split | season | plays | games | pass_plays | box_labelled | rushers_labelled | coverage_labelled |
|---|---|---|---|---|---|---|---|
| test | 2022 | 35969 | 271 | 21372 | 35090 | 20291 | 17979 |
| train | 2016 | 34814 | 256 | 21269 | 33770 | 19339 | 0 |
| train | 2017 | 34358 | 256 | 20585 | 33453 | 18626 | 0 |
| train | 2018 | 34252 | 256 | 20926 | 33290 | 19717 | 17510 |
| train | 2019 | 34605 | 256 | 21190 | 33735 | 19904 | 17717 |
| train | 2020 | 34646 | 256 | 21064 | 33832 | 19979 | 17879 |
| val | 2021 | 35288 | 269 | 21404 | 34435 | 20205 | 18036 |

Collapsed classes (chosen on the training seasons to cover ~95% of plays, remainder = `other`):

- offense personnel: ['1 RB, 1 TE, 3 WR', '1 RB, 2 TE, 2 WR', '2 RB, 1 TE, 2 WR', '1 RB, 3 TE, 1 WR', '2 RB, 2 TE, 1 WR', '1 RB, 0 TE, 4 WR', '6 OL, 1 RB, 1 TE, 2 WR', '6 OL, 1 RB, 2 TE, 1 WR', '2 RB, 0 TE, 3 WR', 'other']
- defense personnel: ['4 DL, 2 LB, 5 DB', '4 DL, 3 LB, 4 DB', '3 DL, 3 LB, 5 DB', '2 DL, 4 LB, 5 DB', '3 DL, 4 LB, 4 DB', '2 DL, 3 LB, 6 DB', '4 DL, 1 LB, 6 DB', '3 DL, 2 LB, 6 DB', '1 DL, 4 LB, 6 DB', '5 DL, 2 LB, 4 DB', 'other']
- coverage type (2018+): ['COVER_3', 'COVER_1', 'COVER_2', 'COVER_4', 'COVER_6', 'COVER_0', 'other']

## Feature sets

- **S0** situation: down, ydstogo, yardline_100, qtr, half_seconds_remaining, score_differential, wp (nflfastR pre-play), goal_to_go, season, week, home/away.
- **S1** = S0 + pre-snap pbp observables: shotgun, no_huddle.
- **S2** = S1 + strictly-prior tendencies: league running share (`t_lg_*`), posteam previous-season share (`t_offprev_*`), posteam in-season share over earlier games (`t_off_*`, shrunk towards the previous season with alpha=100.0 plays), posteam share by down x distance bucket (`t_offdd_*`, shrunk towards the in-season share, alpha=50.0), defteam share of groupings faced (`t_deffaced_*`), posteam prior shotgun rate. For the defense targets S2 additionally holds the defteam's prior-game defense-grouping shares (overall, previous season, by down x distance), prior mean box / pass rushers / man share / coverage shares, and the posteam's faced-defense shares. Numeric tendencies are shrunk league -> previous season -> in-season; the league running mean uses strictly earlier (season, week) slots and is shrunk towards target-free constants ({'shotgun': 0.5, 'box': 6.0, 'rushers': 4.0, 'man_zone': 0.5}) with alpha=10.0 plays, so no feature contains its own or a later week.
- **S2+play_type** adds post-play information (pass / rush flag, qb_dropback): an upper bound on what a pbp-observable could add, not a deployable feature.
- **S2+true_pers** adds the true offense grouping, its RB/TE/WR/OL slot counts and the defteam's prior-game defense-grouping shares conditional on that offense grouping; **+form** adds the charted offense formation.
- **S2+imp_pers** replaces the true grouping by the stage (a) S2 model's out-of-sample class probabilities (out-of-fold by game on the training seasons), their argmax, expected slot counts, and the conditional defense shares looked up with the imputed argmax.

## (a) Offense personnel grouping

Validation grid for `num_leaves` (S2 features):

| num_leaves | best_iter | accuracy | log_loss | top2 | n |
|---|---|---|---|---|---|
| 15 | 115 | 0.6281 | 1.0531 | 0.8252 | 35288 |
| 31 | 90 | 0.6261 | 1.0569 | 0.8246 | 35288 |
| 63 | 81 | 0.6241 | 1.0571 | 0.8228 | 35288 |

Chosen `num_leaves` = 15. Training-season class shares:

| class | train_share |
|---|---|
| 1 RB, 1 TE, 3 WR | 0.5932 |
| 1 RB, 2 TE, 2 WR | 0.1809 |
| 2 RB, 1 TE, 2 WR | 0.0739 |
| 1 RB, 3 TE, 1 WR | 0.0330 |
| 2 RB, 2 TE, 1 WR | 0.0243 |
| 1 RB, 0 TE, 4 WR | 0.0181 |
| 6 OL, 1 RB, 1 TE, 2 WR | 0.0114 |
| 6 OL, 1 RB, 2 TE, 1 WR | 0.0107 |
| 2 RB, 0 TE, 3 WR | 0.0092 |
| other | 0.0454 |

Metrics (multiclass log-loss in nats; top-2 = true class within the two most probable):

| model | split | n | accuracy | log_loss | top2 |
|---|---|---|---|---|---|
| S0 | val | 35288 | 0.5923 | 1.2344 | 0.7943 |
| S0 | test | 35969 | 0.6211 | 1.1560 | 0.8033 |
| S1 | val | 35288 | 0.6084 | 1.1837 | 0.7966 |
| S1 | test | 35969 | 0.6275 | 1.1265 | 0.8062 |
| S2 | val | 35288 | 0.6281 | 1.0531 | 0.8252 |
| S2 | test | 35969 | 0.6470 | 0.9843 | 0.8313 |
| S2+play_type | val | 35288 | 0.6274 | 1.0535 | 0.8245 |
| S2+play_type | test | 35969 | 0.6478 | 0.9830 | 0.8316 |
| baseline: majority (train freq) | val | 35288 | 0.5889 | 1.3542 | 0.7908 |
| baseline: team prior-games | val | 35288 | 0.5983 | 1.2361 | 0.8096 |
| baseline: team x down-distance prior-games | val | 35288 | 0.5974 | 1.2138 | 0.8089 |
| baseline: team previous season | val | 35288 | 0.5889 | 1.3218 | 0.7896 |
| baseline: majority (train freq) | test | 35969 | 0.6187 | 1.2777 | 0.8042 |
| baseline: team prior-games | test | 35969 | 0.6219 | 1.1447 | 0.8255 |
| baseline: team x down-distance prior-games | test | 35969 | 0.6257 | 1.1162 | 0.8287 |
| baseline: team previous season | test | 35969 | 0.6078 | 1.2496 | 0.8060 |
| S2 (test-season labels blanked: no in-season tendencies) | test | 35969 | 0.6305 | 1.0850 | 0.8103 |

Paired bootstrap of the per-play log-loss difference on test (positive = `to` better than `from`):

| from | to | delta_loss | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|
| baseline: majority (train freq) | baseline: team prior-games | 0.1330 | 0.1275 | 0.1385 | 0.1170 | 0.1499 | 35969 | 271 |
| baseline: team prior-games | baseline: team x down-distance prior-games | 0.0285 | 0.0260 | 0.0310 | 0.0246 | 0.0326 | 35969 | 271 |
| baseline: team x down-distance prior-games | S2 | 0.1318 | 0.1264 | 0.1378 | 0.1201 | 0.1439 | 35969 | 271 |
| S0 | S1 | 0.0295 | 0.0261 | 0.0331 | 0.0222 | 0.0366 | 35969 | 271 |
| S1 | S2 | 0.1422 | 0.1369 | 0.1474 | 0.1246 | 0.1603 | 35969 | 271 |
| S2 | S2+play_type | 0.0013 | -0.0000 | 0.0026 | -0.0012 | 0.0036 | 35969 | 271 |

Per-class test report for S2:

| class | support | share | precision | recall | f1 | mean_prob |
|---|---|---|---|---|---|---|
| 1 RB, 1 TE, 3 WR | 22253 | 0.6187 | 0.7191 | 0.9049 | 0.8014 | 0.6162 |
| 1 RB, 2 TE, 2 WR | 6672 | 0.1855 | 0.3982 | 0.3350 | 0.3639 | 0.1872 |
| 2 RB, 1 TE, 2 WR | 2792 | 0.0776 | 0.3938 | 0.2317 | 0.2918 | 0.0781 |
| 1 RB, 3 TE, 1 WR | 1373 | 0.0382 | 0.2288 | 0.0197 | 0.0362 | 0.0344 |
| 2 RB, 2 TE, 1 WR | 912 | 0.0254 | 0.3491 | 0.0647 | 0.1092 | 0.0211 |
| 1 RB, 0 TE, 4 WR | 205 | 0.0057 | 0.0000 | 0.0000 |  | 0.0100 |
| 6 OL, 1 RB, 1 TE, 2 WR | 308 | 0.0086 |  | 0.0000 |  | 0.0079 |
| 6 OL, 1 RB, 2 TE, 1 WR | 458 | 0.0127 | 0.3939 | 0.0284 | 0.0530 | 0.0090 |
| 2 RB, 0 TE, 3 WR | 137 | 0.0038 |  | 0.0000 |  | 0.0044 |
| other | 859 | 0.0239 | 0.4005 | 0.1804 | 0.2488 | 0.0317 |

Test confusion for S2 (rows = true, columns = predicted):

| true | 1 RB, 0 TE, 4 WR | 1 RB, 1 TE, 3 WR | 1 RB, 2 TE, 2 WR | 1 RB, 3 TE, 1 WR | 2 RB, 1 TE, 2 WR | 2 RB, 2 TE, 1 WR | 6 OL, 1 RB, 2 TE, 1 WR | other |
|---|---|---|---|---|---|---|---|---|
| 1 RB, 0 TE, 4 WR | 0 | 204 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1 RB, 1 TE, 3 WR | 5 | 20136 | 1648 | 13 | 358 | 13 | 1 | 79 |
| 1 RB, 2 TE, 2 WR | 0 | 4011 | 2235 | 32 | 284 | 38 | 5 | 67 |
| 1 RB, 3 TE, 1 WR | 0 | 608 | 681 | 27 | 27 | 9 | 3 | 18 |
| 2 RB, 0 TE, 3 WR | 1 | 97 | 8 | 0 | 31 | 0 | 0 | 0 |
| 2 RB, 1 TE, 2 WR | 0 | 1653 | 444 | 4 | 647 | 24 | 1 | 19 |
| 2 RB, 2 TE, 1 WR | 0 | 401 | 174 | 9 | 240 | 59 | 2 | 27 |
| 6 OL, 1 RB, 1 TE, 2 WR | 0 | 128 | 150 | 6 | 19 | 0 | 1 | 4 |
| 6 OL, 1 RB, 2 TE, 1 WR | 0 | 212 | 176 | 21 | 8 | 10 | 13 | 18 |
| other | 0 | 550 | 96 | 6 | 29 | 16 | 7 | 155 |

Out-of-fold S2 on the training seasons (4 game-grouped folds): accuracy 0.6347, log-loss 1.0437, n=172675.


Top S2 features by gain share:

| feature | gain_share |
|---|---|
| shotgun | 0.2557 |
| t_offdd_2RB_1TE_2WR | 0.0628 |
| half_seconds_remaining | 0.0396 |
| t_offdd_other | 0.0378 |
| down | 0.0344 |
| t_offdd_1RB_0TE_4WR | 0.0339 |
| yardline_100 | 0.0326 |
| wp | 0.0287 |
| t_offdd_1RB_1TE_3WR | 0.0284 |
| t_offdd_1RB_2TE_2WR | 0.0237 |
| t_off_2RB_1TE_2WR | 0.0233 |
| t_offsg | 0.0221 |
| t_off_2RB_2TE_1WR | 0.0178 |
| t_offdd_2RB_2TE_1WR | 0.0175 |
| t_off_1RB_1TE_3WR | 0.0162 |

## (b) Defense personnel, box, pass rush and coverage given the offense

Targets: `defense_personnel` (collapsed), `defenders_in_box` (all plays with a value), `number_of_pass_rushers` (pass plays only), `man_zone` (MAN=1 vs ZONE=0, 2018+ pass plays), `coverage_type` (collapsed, 2018+). Classification: accuracy / log-loss / top-2; counts: MAE / R2 / RMSE (LightGBM L2 objective).


### defense_personnel

| model | split | n | accuracy | log_loss | top2 |
|---|---|---|---|---|---|
| S2 | val | 35288 | 0.5384 | 1.2193 | 0.7864 |
| S2 | test | 35969 | 0.5062 | 1.2575 | 0.7625 |
| S2+true_pers | val | 35288 | 0.6117 | 1.0810 | 0.8245 |
| S2+true_pers | test | 35969 | 0.5792 | 1.1206 | 0.8090 |
| S2+true_pers+form | val | 35288 | 0.6120 | 1.0799 | 0.8254 |
| S2+true_pers+form | test | 35969 | 0.5774 | 1.1178 | 0.8060 |
| S2+imp_pers | val | 35288 | 0.5360 | 1.2187 | 0.7870 |
| S2+imp_pers | test | 35969 | 0.5136 | 1.2425 | 0.7675 |
| baseline: majority (train freq) | val | 35288 | 0.2894 | 2.0499 | 0.4062 |
| baseline: team prior-games | val | 35288 | 0.4718 | 1.4754 | 0.6828 |
| baseline: team x down-distance prior-games | val | 35288 | 0.4896 | 1.3973 | 0.7302 |
| baseline: majority (train freq) | test | 35969 | 0.2627 | 2.0601 | 0.3633 |
| baseline: team prior-games | test | 35969 | 0.4483 | 1.4822 | 0.6918 |
| baseline: team x down-distance prior-games | test | 35969 | 0.4751 | 1.4011 | 0.7267 |

### defenders_in_box

| model | split | n | mae | r2 | rmse |
|---|---|---|---|---|---|
| S2 | val | 34435 | 0.6110 | 0.4574 | 0.7886 |
| S2 | test | 35090 | 0.5934 | 0.4530 | 0.7728 |
| S2+true_pers | val | 34435 | 0.5827 | 0.5164 | 0.7444 |
| S2+true_pers | test | 35090 | 0.5630 | 0.5139 | 0.7285 |
| S2+true_pers+form | val | 34435 | 0.5665 | 0.5518 | 0.7166 |
| S2+true_pers+form | test | 35090 | 0.5465 | 0.5478 | 0.7027 |
| S2+imp_pers | val | 34435 | 0.6131 | 0.4542 | 0.7909 |
| S2+imp_pers | test | 35090 | 0.5936 | 0.4526 | 0.7731 |
| baseline: global mean (train) | val | 34435 | 0.8472 | -0.0004 | 1.0707 |
| baseline: team prior-games | val | 34435 | 0.8242 | 0.0235 | 1.0578 |
| baseline: global mean (train) | test | 35090 | 0.8241 | -0.0001 | 1.0450 |
| baseline: team prior-games | test | 35090 | 0.7907 | 0.0155 | 1.0368 |

### number_of_pass_rushers

| model | split | n | mae | r2 | rmse |
|---|---|---|---|---|---|
| S2 | val | 20122 | 0.5556 | 0.0611 | 0.8232 |
| S2 | test | 20186 | 0.5352 | 0.0591 | 0.7837 |
| S2+true_pers | val | 20122 | 0.5579 | 0.0639 | 0.8220 |
| S2+true_pers | test | 20186 | 0.5355 | 0.0665 | 0.7806 |
| S2+true_pers+form | val | 20122 | 0.5533 | 0.0684 | 0.8201 |
| S2+true_pers+form | test | 20186 | 0.5348 | 0.0655 | 0.7810 |
| S2+imp_pers | val | 20122 | 0.5561 | 0.0587 | 0.8243 |
| S2+imp_pers | test | 20186 | 0.5334 | 0.0613 | 0.7828 |
| baseline: global mean (train) | val | 20122 | 0.6010 | -0.0031 | 0.8509 |
| baseline: team prior-games | val | 20122 | 0.5825 | 0.0040 | 0.8479 |
| baseline: global mean (train) | test | 20186 | 0.5728 | -0.0004 | 0.8081 |
| baseline: team prior-games | test | 20186 | 0.5549 | 0.0043 | 0.8062 |

### man_zone

| model | split | n | accuracy | log_loss | top2 |
|---|---|---|---|---|---|
| S2 | val | 18036 | 0.7445 | 0.5362 |  |
| S2 | test | 17979 | 0.7495 | 0.5300 |  |
| S2+true_pers | val | 18036 | 0.7427 | 0.5360 |  |
| S2+true_pers | test | 17979 | 0.7503 | 0.5269 |  |
| S2+true_pers+form | val | 18036 | 0.7451 | 0.5366 |  |
| S2+true_pers+form | test | 17979 | 0.7488 | 0.5286 |  |
| S2+imp_pers | val | 18036 | 0.7436 | 0.5374 |  |
| S2+imp_pers | test | 17979 | 0.7490 | 0.5299 |  |
| baseline: base rate (train) | val | 18036 | 0.6971 | 0.6194 |  |
| baseline: team prior-games | val | 18036 | 0.6952 | 0.6055 |  |
| baseline: base rate (train) | test | 17979 | 0.7125 | 0.6103 |  |
| baseline: team prior-games | test | 17979 | 0.7125 | 0.5930 |  |

### coverage_type

| model | split | n | accuracy | log_loss | top2 |
|---|---|---|---|---|---|
| S2 | val | 18036 | 0.4001 | 1.4854 | 0.6402 |
| S2 | test | 17979 | 0.3947 | 1.5110 | 0.6283 |
| S2+true_pers | val | 18036 | 0.4030 | 1.4840 | 0.6401 |
| S2+true_pers | test | 17979 | 0.3936 | 1.5097 | 0.6275 |
| S2+true_pers+form | val | 18036 | 0.3997 | 1.4848 | 0.6396 |
| S2+true_pers+form | test | 17979 | 0.3957 | 1.5065 | 0.6300 |
| S2+imp_pers | val | 18036 | 0.4000 | 1.4868 | 0.6376 |
| S2+imp_pers | test | 17979 | 0.3948 | 1.5111 | 0.6247 |
| baseline: majority (train freq) | val | 18036 | 0.3292 | 1.7035 | 0.5672 |
| baseline: team prior-games | val | 18036 | 0.3332 | 1.6305 | 0.5779 |
| baseline: majority (train freq) | test | 17979 | 0.3215 | 1.7155 | 0.5491 |
| baseline: team prior-games | test | 17979 | 0.3227 | 1.6430 | 0.5607 |

### (d) Marginalisation check: paired bootstrap of per-play loss on test

Loss = log-loss for classification targets, squared error for counts. Positive `delta_loss` means the `to` model is better. `S2 -> S2+imp_pers` is what the participation step buys when personnel must be imputed; `S2+imp_pers -> S2+true_pers` is the cost of imputation.

| target | from | to | delta_loss | ci_low | ci_high | ci_low_game | ci_high_game | n | n_games |
|---|---|---|---|---|---|---|---|---|---|
| defense_personnel | baseline: team prior-games | S2 | 0.2248 | 0.2170 | 0.2321 | 0.2048 | 0.2458 | 35969 | 271 |
| defense_personnel | S2 | S2+imp_pers | 0.0149 | 0.0114 | 0.0182 | 0.0055 | 0.0238 | 35969 | 271 |
| defense_personnel | S2 | S2+true_pers | 0.1369 | 0.1311 | 0.1435 | 0.1214 | 0.1528 | 35969 | 271 |
| defense_personnel | S2+imp_pers | S2+true_pers | 0.1220 | 0.1157 | 0.1286 | 0.1080 | 0.1373 | 35969 | 271 |
| defense_personnel | S2+true_pers | S2+true_pers+form | 0.0028 | 0.0003 | 0.0049 | -0.0030 | 0.0094 | 35969 | 271 |
| defenders_in_box | baseline: team prior-games | S2 | 0.4777 | 0.4609 | 0.4938 | 0.4485 | 0.5067 | 35090 | 271 |
| defenders_in_box | S2 | S2+imp_pers | -0.0004 | -0.0023 | 0.0016 | -0.0039 | 0.0028 | 35090 | 271 |
| defenders_in_box | S2 | S2+true_pers | 0.0665 | 0.0620 | 0.0713 | 0.0596 | 0.0737 | 35090 | 271 |
| defenders_in_box | S2+imp_pers | S2+true_pers | 0.0669 | 0.0625 | 0.0718 | 0.0596 | 0.0742 | 35090 | 271 |
| defenders_in_box | S2+true_pers | S2+true_pers+form | 0.0370 | 0.0330 | 0.0407 | 0.0321 | 0.0418 | 35090 | 271 |
| number_of_pass_rushers | baseline: team prior-games | S2 | 0.0358 | 0.0301 | 0.0412 | 0.0304 | 0.0418 | 20186 | 271 |
| number_of_pass_rushers | S2 | S2+imp_pers | 0.0015 | -0.0002 | 0.0032 | -0.0007 | 0.0035 | 20186 | 271 |
| number_of_pass_rushers | S2 | S2+true_pers | 0.0049 | 0.0021 | 0.0081 | 0.0019 | 0.0078 | 20186 | 271 |
| number_of_pass_rushers | S2+imp_pers | S2+true_pers | 0.0034 | 0.0007 | 0.0064 | 0.0007 | 0.0062 | 20186 | 271 |
| number_of_pass_rushers | S2+true_pers | S2+true_pers+form | -0.0007 | -0.0032 | 0.0018 | -0.0037 | 0.0020 | 20186 | 271 |
| man_zone | baseline: team prior-games | S2 | 0.0630 | 0.0577 | 0.0685 | 0.0573 | 0.0690 | 17979 | 271 |
| man_zone | S2 | S2+imp_pers | 0.0001 | -0.0008 | 0.0009 | -0.0013 | 0.0014 | 17979 | 271 |
| man_zone | S2 | S2+true_pers | 0.0030 | 0.0021 | 0.0040 | 0.0018 | 0.0041 | 17979 | 271 |
| man_zone | S2+imp_pers | S2+true_pers | 0.0029 | 0.0020 | 0.0039 | 0.0016 | 0.0043 | 17979 | 271 |
| man_zone | S2+true_pers | S2+true_pers+form | -0.0017 | -0.0026 | -0.0008 | -0.0030 | -0.0004 | 17979 | 271 |
| coverage_type | baseline: team prior-games | S2 | 0.1319 | 0.1228 | 0.1404 | 0.1215 | 0.1429 | 17979 | 271 |
| coverage_type | S2 | S2+imp_pers | -0.0000 | -0.0021 | 0.0020 | -0.0031 | 0.0031 | 17979 | 271 |
| coverage_type | S2 | S2+true_pers | 0.0014 | -0.0004 | 0.0033 | -0.0016 | 0.0040 | 17979 | 271 |
| coverage_type | S2+imp_pers | S2+true_pers | 0.0014 | -0.0008 | 0.0034 | -0.0016 | 0.0041 | 17979 | 271 |
| coverage_type | S2+true_pers | S2+true_pers+form | 0.0032 | 0.0013 | 0.0051 | 0.0006 | 0.0061 | 17979 | 271 |

Top features by gain share for `defense_personnel` (S2, S2+true_pers, S2+imp_pers):

| model | feature | gain_share |
|---|---|---|
| S2 | t_dgrpdd_4DL_2LB_5DB | 0.1269 |
| S2 | shotgun | 0.0811 |
| S2 | t_dgrpdd_2DL_4LB_5DB | 0.0665 |
| S2 | t_dgrpdd_3DL_4LB_4DB | 0.0492 |
| S2 | t_dgrpdd_4DL_3LB_4DB | 0.0468 |
| S2 | t_dgrpdd_3DL_3LB_5DB | 0.0438 |
| S2 | t_dgrpdd_2DL_3LB_6DB | 0.0359 |
| S2 | t_dgrpdd_4DL_1LB_6DB | 0.0357 |
| S2 | half_seconds_remaining | 0.0299 |
| S2 | down | 0.0252 |
| S2 | t_dgrp_2DL_4LB_5DB | 0.0226 |
| S2 | t_dgrpdd_1DL_4LB_6DB | 0.0211 |
| S2+true_pers | t_dgrp_giv_4DL_2LB_5DB | 0.1085 |
| S2+true_pers | slot_WR | 0.1001 |
| S2+true_pers | t_dgrp_giv_2DL_4LB_5DB | 0.0731 |
| S2+true_pers | t_dgrp_giv_3DL_4LB_4DB | 0.0701 |
| S2+true_pers | t_dgrp_giv_4DL_3LB_4DB | 0.0668 |
| S2+true_pers | t_dgrp_giv_3DL_3LB_5DB | 0.0334 |
| S2+true_pers | t_dgrpdd_2DL_3LB_6DB | 0.0296 |
| S2+true_pers | off_grp_code | 0.0227 |
| S2+true_pers | down | 0.0227 |
| S2+true_pers | t_dgrpdd_4DL_1LB_6DB | 0.0216 |
| S2+true_pers | t_dgrpdd_4DL_2LB_5DB | 0.0203 |
| S2+true_pers | half_seconds_remaining | 0.0192 |
| S2+imp_pers | t_dgrp_giv_2DL_4LB_5DB | 0.0730 |
| S2+imp_pers | t_dgrp_giv_4DL_2LB_5DB | 0.0704 |
| S2+imp_pers | slot_WR | 0.0701 |
| S2+imp_pers | t_dgrpdd_4DL_2LB_5DB | 0.0647 |
| S2+imp_pers | t_dgrpdd_2DL_3LB_6DB | 0.0401 |
| S2+imp_pers | t_dgrpdd_3DL_3LB_5DB | 0.0346 |
| S2+imp_pers | t_dgrpdd_3DL_4LB_4DB | 0.0333 |
| S2+imp_pers | p_off_1RB_1TE_3WR | 0.0311 |
| S2+imp_pers | t_dgrpdd_4DL_3LB_4DB | 0.0297 |
| S2+imp_pers | t_dgrpdd_2DL_4LB_5DB | 0.0275 |
| S2+imp_pers | t_dgrpdd_1DL_4LB_6DB | 0.0205 |
| S2+imp_pers | t_dgrp_giv_4DL_3LB_4DB | 0.0205 |

## (c) Player-level participation

Seeded subsample of 15000 plays from 2021 (training; rounds chosen on a game-held-out fifth) and 15000 plays from 2022 (test), restricted to plays with exactly 11 participants per side. Candidates = the team's `ACT` weekly roster (see below), side-matched position groups (K/P/LS excluded). Per-candidate features: position group and depth-chart position, depth-chart rank that week (min over listed positions, 4 if unlisted), season-to-date / last-3 / previous-season snap share from strictly earlier games, weeks since last snap, prior-usage rank within the team-week position group (season-to-date share, previous season for week 1, ties by depth rank), group size, the slots the play's grouping gives the candidate's group (true, or imputed argmax and expected value from stages (a)/(b)), slot minus rank, and the situation. `imputed grouping` for the defense uses the stage (b) `S2+imp_pers` defense-personnel model (itself fed by imputed offense personnel). `Candidates` are the weekly roster rows with status `ACT`: in these files that is exactly the 48-man game-day active list (declared inactives carry `INA`, ~6 per team-week, none of them takes a snap), fixed about 90 minutes before kickoff, so the candidate set is pre-play but not mid-week information.

Candidate coverage:

| season | side | plays | candidates_per_play | true_players_in_candidates | imputed_grouping_slots_correct |
|---|---|---|---|---|---|
| 2021 | offense | 15000 | 21.9926 | 0.9984 | 0.6215 |
| 2021 | defense | 15000 | 22.9959 | 0.9987 | 0.5393 |
| 2022 | offense | 15000 | 21.9086 | 0.9993 | 0.6467 |
| 2022 | defense | 15000 | 23.0889 | 0.9998 | 0.5113 |

Fits (LightGBM binary, on-field or not):

| side | variant | rounds | n_train_rows | top_features |
|---|---|---|---|---|
| offense | true | 218 | 329889 | slot_minus_rank_true, snap_last3, snap_std, depth_rank, slot_true |
| offense | imp | 179 | 329889 | slot_minus_rank_exp, slot_minus_rank_imp, snap_last3, depth_rank, snap_std |
| defense | true | 640 | 344939 | slot_minus_rank_true, snap_last3, snap_std, snap_prev, dcp_code |
| defense | imp | 643 | 344939 | slot_minus_rank_imp, snap_last3, slot_minus_rank_exp, snap_std, snap_prev |

Slot decoding on test: exact-lineup accuracy (all 11 correct) and per-slot accuracy (correct players / 11):

| decoder | side | plays | exact_lineup_acc | per_slot_acc | mean_errors_per_play |
|---|---|---|---|---|---|
| baseline: depth chart rank \| true grouping | defense | 15000 | 0.0547 | 0.7687 | 2.5447 |
| baseline: depth chart rank \| true grouping | offense | 15000 | 0.1019 | 0.8159 | 2.0255 |
| baseline: prior usage rank \| imputed grouping | defense | 15000 | 0.0306 | 0.7562 | 2.6817 |
| baseline: prior usage rank \| imputed grouping | offense | 15000 | 0.0753 | 0.7970 | 2.2330 |
| baseline: prior usage rank \| true grouping | defense | 15000 | 0.0533 | 0.7738 | 2.4877 |
| baseline: prior usage rank \| true grouping | offense | 15000 | 0.1109 | 0.8162 | 2.0222 |
| model \| imputed grouping | defense | 15000 | 0.0361 | 0.7655 | 2.5795 |
| model \| imputed grouping | offense | 15000 | 0.0901 | 0.8112 | 2.0767 |
| model \| true grouping | defense | 15000 | 0.0634 | 0.7835 | 2.3818 |
| model \| true grouping | offense | 15000 | 0.1363 | 0.8342 | 1.8242 |
| model(true feats) \| imputed slots | defense | 15000 | 0.0341 | 0.7642 | 2.5936 |
| model(true feats) \| imputed slots | offense | 15000 | 0.0913 | 0.8142 | 2.0437 |

Same, split by whether the imputed grouping gave the correct slot counts:

| side | imp_grp_ok | plays | exact_lineup_acc | per_slot_acc |
|---|---|---|---|---|
| defense | False | 7330 | 0.0000 | 0.7420 |
| defense | True | 7670 | 0.0707 | 0.7880 |
| offense | False | 5300 | 0.0000 | 0.7683 |
| offense | True | 9700 | 0.1394 | 0.8346 |

Per position subgroup on test (precision / recall / F1 after decoding; AUC of the raw score among candidates of that subgroup; WR/TE/RB/CB tiers use the prior-usage rank):

| decoder | subgroup | n_candidates | n_on_field | on_field_rate | precision | recall | f1 | auc | false_negatives | false_positives |
|---|---|---|---|---|---|---|---|---|---|---|
| model \| true grouping | QB | 30514 | 14982 | 0.4910 | 0.8972 | 0.9005 | 0.8989 | 0.9412 | 1491 | 1545 |
| model \| true grouping | OL | 120182 | 75432 | 0.6276 | 0.9052 | 0.9058 | 0.9055 | 0.9293 | 7103 | 7158 |
| model \| true grouping | RB1 | 15000 | 9123 | 0.6082 | 0.6205 | 0.9950 | 0.7643 | 0.6529 | 46 | 5552 |
| model \| true grouping | RB2+ | 39400 | 7547 | 0.1915 | 0.7396 | 0.2002 | 0.3151 | 0.8044 | 6036 | 532 |
| model \| true grouping | WR1-2 | 30000 | 23614 | 0.7871 | 0.8274 | 0.9839 | 0.8989 | 0.8170 | 381 | 4845 |
| model \| true grouping | WR3+ | 46715 | 14658 | 0.3138 | 0.7493 | 0.5211 | 0.6147 | 0.8074 | 7020 | 2556 |
| model \| true grouping | TE1 | 15000 | 10519 | 0.7013 | 0.7126 | 0.9990 | 0.8319 | 0.7132 | 11 | 4237 |
| model \| true grouping | TE2+ | 31818 | 9006 | 0.2830 | 0.7984 | 0.4275 | 0.5568 | 0.8241 | 5156 | 972 |
| model \| true grouping | DL | 103966 | 47484 | 0.4567 | 0.6621 | 0.6624 | 0.6622 | 0.7662 | 16031 | 16054 |
| model \| true grouping | LB | 105235 | 44908 | 0.4267 | 0.7975 | 0.7976 | 0.7976 | 0.9052 | 9089 | 9094 |
| model \| true grouping | CB1-2 | 13438 | 12358 | 0.9196 | 0.9266 | 0.9950 | 0.9596 | 0.7456 | 62 | 974 |
| model \| true grouping | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7851 | 0.7562 | 0.7704 | 0.8833 | 6638 | 5633 |
| model \| true grouping | S | 58193 | 32997 | 0.5670 | 0.8799 | 0.8825 | 0.8812 | 0.9305 | 3876 | 3973 |
| model \| imputed grouping | QB | 30514 | 14982 | 0.4910 | 0.8974 | 0.8985 | 0.8979 | 0.9413 | 1521 | 1539 |
| model \| imputed grouping | OL | 120182 | 75432 | 0.6276 | 0.9040 | 0.8990 | 0.9015 | 0.9282 | 7615 | 7200 |
| model \| imputed grouping | RB1 | 15000 | 9123 | 0.6082 | 0.6103 | 0.9957 | 0.7568 | 0.6315 | 39 | 5800 |
| model \| imputed grouping | RB2+ | 39400 | 7547 | 0.1915 | 0.4903 | 0.0567 | 0.1017 | 0.7486 | 7119 | 445 |
| model \| imputed grouping | WR1-2 | 30000 | 23614 | 0.7871 | 0.7967 | 0.9916 | 0.8835 | 0.7474 | 198 | 5976 |
| model \| imputed grouping | WR3+ | 46715 | 14658 | 0.3138 | 0.6146 | 0.5136 | 0.5596 | 0.7756 | 7130 | 4721 |
| model \| imputed grouping | TE1 | 15000 | 10519 | 0.7013 | 0.7021 | 0.9913 | 0.8220 | 0.6590 | 91 | 4425 |
| model \| imputed grouping | TE2+ | 31818 | 9006 | 0.2830 | 0.6183 | 0.1874 | 0.2877 | 0.7329 | 7318 | 1042 |
| model \| imputed grouping | DL | 103966 | 47484 | 0.4567 | 0.6419 | 0.6509 | 0.6464 | 0.7507 | 16577 | 17243 |
| model \| imputed grouping | LB | 105235 | 44908 | 0.4267 | 0.7779 | 0.7574 | 0.7675 | 0.8915 | 10895 | 9712 |
| model \| imputed grouping | CB1-2 | 13438 | 12358 | 0.9196 | 0.9263 | 0.9919 | 0.9580 | 0.7504 | 100 | 975 |
| model \| imputed grouping | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7577 | 0.7385 | 0.7480 | 0.8720 | 7118 | 6429 |
| model \| imputed grouping | S | 58193 | 32997 | 0.5670 | 0.8701 | 0.8796 | 0.8748 | 0.9248 | 3972 | 4334 |
| model(true feats) \| imputed slots | QB | 30514 | 14982 | 0.4910 | 0.8971 | 0.8981 | 0.8976 | 0.9412 | 1526 | 1544 |
| model(true feats) \| imputed slots | OL | 120182 | 75432 | 0.6276 | 0.9069 | 0.9019 | 0.9044 | 0.9293 | 7399 | 6984 |
| model(true feats) \| imputed slots | RB1 | 15000 | 9123 | 0.6082 | 0.6156 | 0.9522 | 0.7478 | 0.6529 | 436 | 5424 |
| model(true feats) \| imputed slots | RB2+ | 39400 | 7547 | 0.1915 | 0.6713 | 0.1464 | 0.2404 | 0.8044 | 6442 | 541 |
| model(true feats) \| imputed slots | WR1-2 | 30000 | 23614 | 0.7871 | 0.8013 | 0.9895 | 0.8855 | 0.8170 | 248 | 5793 |
| model(true feats) \| imputed slots | WR3+ | 46715 | 14658 | 0.3138 | 0.6089 | 0.5185 | 0.5601 | 0.8074 | 7058 | 4882 |
| model(true feats) \| imputed slots | TE1 | 15000 | 10519 | 0.7013 | 0.7031 | 0.9732 | 0.8164 | 0.7132 | 282 | 4323 |
| model(true feats) \| imputed slots | TE2+ | 31818 | 9006 | 0.2830 | 0.6153 | 0.2065 | 0.3093 | 0.8241 | 7146 | 1163 |
| model(true feats) \| imputed slots | DL | 103966 | 47484 | 0.4567 | 0.6405 | 0.6495 | 0.6450 | 0.7662 | 16642 | 17308 |
| model(true feats) \| imputed slots | LB | 105235 | 44908 | 0.4267 | 0.7758 | 0.7553 | 0.7654 | 0.9052 | 10988 | 9805 |
| model(true feats) \| imputed slots | CB1-2 | 13438 | 12358 | 0.9196 | 0.9254 | 0.9935 | 0.9582 | 0.7456 | 80 | 990 |
| model(true feats) \| imputed slots | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7577 | 0.7375 | 0.7474 | 0.8833 | 7147 | 6419 |
| model(true feats) \| imputed slots | S | 58193 | 32997 | 0.5670 | 0.8687 | 0.8783 | 0.8734 | 0.9305 | 4016 | 4382 |
| baseline: prior usage rank \| true grouping | QB | 30514 | 14982 | 0.4910 | 0.8619 | 0.8650 | 0.8634 | 0.8644 | 2023 | 2077 |
| baseline: prior usage rank \| true grouping | OL | 120182 | 75432 | 0.6276 | 0.8823 | 0.8830 | 0.8827 | 0.8968 | 8827 | 8882 |
| baseline: prior usage rank \| true grouping | RB1 | 15000 | 9123 | 0.6082 | 0.6133 | 0.9975 | 0.7596 | 0.5810 | 23 | 5737 |
| baseline: prior usage rank \| true grouping | RB2+ | 39400 | 7547 | 0.1915 | 0.5967 | 0.1451 | 0.2334 | 0.7079 | 6452 | 740 |
| baseline: prior usage rank \| true grouping | WR1-2 | 30000 | 23614 | 0.7871 | 0.8180 | 0.9895 | 0.8956 | 0.6621 | 247 | 5199 |
| baseline: prior usage rank \| true grouping | WR3+ | 46715 | 14658 | 0.3138 | 0.7393 | 0.4896 | 0.5891 | 0.7318 | 7482 | 2530 |
| baseline: prior usage rank \| true grouping | TE1 | 15000 | 10519 | 0.7013 | 0.7116 | 1.0000 | 0.8315 | 0.6361 | 0 | 4263 |
| baseline: prior usage rank \| true grouping | TE2+ | 31818 | 9006 | 0.2830 | 0.8038 | 0.4270 | 0.5578 | 0.6745 | 5160 | 939 |
| baseline: prior usage rank \| true grouping | DL | 103966 | 47484 | 0.4567 | 0.6551 | 0.6554 | 0.6552 | 0.7155 | 16363 | 16386 |
| baseline: prior usage rank \| true grouping | LB | 105235 | 44908 | 0.4267 | 0.7871 | 0.7872 | 0.7872 | 0.8751 | 9555 | 9560 |
| baseline: prior usage rank \| true grouping | CB1-2 | 13438 | 12358 | 0.9196 | 0.9199 | 1.0000 | 0.9583 | 0.6240 | 0 | 1076 |
| baseline: prior usage rank \| true grouping | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7707 | 0.7371 | 0.7535 | 0.8333 | 7156 | 5970 |
| baseline: prior usage rank \| true grouping | S | 58193 | 32997 | 0.5670 | 0.8694 | 0.8724 | 0.8709 | 0.9106 | 4210 | 4324 |
| baseline: depth chart rank \| true grouping | QB | 30514 | 14982 | 0.4910 | 0.8971 | 0.9003 | 0.8987 | 0.8904 | 1493 | 1547 |
| baseline: depth chart rank \| true grouping | OL | 120182 | 75432 | 0.6276 | 0.8881 | 0.8888 | 0.8885 | 0.8955 | 8390 | 8445 |
| baseline: depth chart rank \| true grouping | RB1 | 15000 | 9123 | 0.6082 | 0.6302 | 0.9171 | 0.7471 | 0.5849 | 756 | 4910 |
| baseline: depth chart rank \| true grouping | RB2+ | 39400 | 7547 | 0.1915 | 0.5399 | 0.2429 | 0.3350 | 0.6464 | 5714 | 1562 |
| baseline: depth chart rank \| true grouping | WR1-2 | 30000 | 23614 | 0.7871 | 0.8365 | 0.9399 | 0.8852 | 0.6653 | 1419 | 4339 |
| baseline: depth chart rank \| true grouping | WR3+ | 46715 | 14658 | 0.3138 | 0.6717 | 0.5379 | 0.5974 | 0.7017 | 6773 | 3853 |
| baseline: depth chart rank \| true grouping | TE1 | 15000 | 10519 | 0.7013 | 0.7355 | 0.8964 | 0.8080 | 0.6236 | 1090 | 3391 |
| baseline: depth chart rank \| true grouping | TE2+ | 31818 | 9006 | 0.2830 | 0.6487 | 0.4860 | 0.5557 | 0.6493 | 4629 | 2370 |
| baseline: depth chart rank \| true grouping | DL | 103966 | 47484 | 0.4567 | 0.6501 | 0.6504 | 0.6503 | 0.7040 | 16599 | 16622 |
| baseline: depth chart rank \| true grouping | LB | 105235 | 44908 | 0.4267 | 0.7857 | 0.7857 | 0.7857 | 0.8601 | 9622 | 9627 |
| baseline: depth chart rank \| true grouping | CB1-2 | 13438 | 12358 | 0.9196 | 0.9257 | 0.9845 | 0.9542 | 0.6546 | 192 | 976 |
| baseline: depth chart rank \| true grouping | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7596 | 0.7333 | 0.7462 | 0.8086 | 7260 | 6319 |
| baseline: depth chart rank \| true grouping | S | 58193 | 32997 | 0.5670 | 0.8604 | 0.8646 | 0.8625 | 0.8890 | 4467 | 4628 |
| baseline: prior usage rank \| imputed grouping | QB | 30514 | 14982 | 0.4910 | 0.8621 | 0.8631 | 0.8626 | 0.8644 | 2051 | 2069 |
| baseline: prior usage rank \| imputed grouping | OL | 120182 | 75432 | 0.6276 | 0.8836 | 0.8788 | 0.8812 | 0.8968 | 9145 | 8730 |
| baseline: prior usage rank \| imputed grouping | RB1 | 15000 | 9123 | 0.6082 | 0.6096 | 0.9975 | 0.7567 | 0.5810 | 23 | 5829 |
| baseline: prior usage rank \| imputed grouping | RB2+ | 39400 | 7547 | 0.1915 | 0.4867 | 0.0534 | 0.0962 | 0.7079 | 7144 | 425 |
| baseline: prior usage rank \| imputed grouping | WR1-2 | 30000 | 23614 | 0.7871 | 0.7891 | 0.9968 | 0.8809 | 0.6621 | 75 | 6290 |
| baseline: prior usage rank \| imputed grouping | WR3+ | 46715 | 14658 | 0.3138 | 0.6035 | 0.4863 | 0.5386 | 0.7318 | 7530 | 4684 |
| baseline: prior usage rank \| imputed grouping | TE1 | 15000 | 10519 | 0.7013 | 0.7013 | 0.9999 | 0.8244 | 0.6361 | 1 | 4480 |
| baseline: prior usage rank \| imputed grouping | TE2+ | 31818 | 9006 | 0.2830 | 0.6186 | 0.1775 | 0.2759 | 0.6745 | 7407 | 986 |
| baseline: prior usage rank \| imputed grouping | DL | 103966 | 47484 | 0.4567 | 0.6353 | 0.6442 | 0.6397 | 0.7155 | 16894 | 17560 |
| baseline: prior usage rank \| imputed grouping | LB | 105235 | 44908 | 0.4267 | 0.7658 | 0.7456 | 0.7556 | 0.8751 | 11423 | 10240 |
| baseline: prior usage rank \| imputed grouping | CB1-2 | 13438 | 12358 | 0.9196 | 0.9196 | 1.0000 | 0.9581 | 0.6240 | 0 | 1080 |
| baseline: prior usage rank \| imputed grouping | CB3+ (nickel/dime) | 65502 | 27222 | 0.4156 | 0.7440 | 0.7231 | 0.7334 | 0.8333 | 7538 | 6772 |
| baseline: prior usage rank \| imputed grouping | S | 58193 | 32997 | 0.5670 | 0.8624 | 0.8685 | 0.8654 | 0.9106 | 4340 | 4574 |

Where the misses are (true on-field players not selected), by subgroup:

| subgroup | baseline: depth chart rank \| true grouping | baseline: prior usage rank \| imputed grouping | baseline: prior usage rank \| true grouping | model \| imputed grouping | model \| true grouping | model(true feats) \| imputed slots |
|---|---|---|---|---|---|---|
| CB1-2 | 0.003 |  |  | 0.001 | 0.001 | 0.001 |
| CB3+ (nickel/dime) | 0.106 | 0.102 | 0.106 | 0.102 | 0.105 | 0.103 |
| DL | 0.243 | 0.230 | 0.242 | 0.238 | 0.255 | 0.240 |
| LB | 0.141 | 0.155 | 0.142 | 0.156 | 0.144 | 0.158 |
| OL | 0.123 | 0.124 | 0.131 | 0.109 | 0.113 | 0.107 |
| QB | 0.022 | 0.028 | 0.030 | 0.022 | 0.024 | 0.022 |
| RB1 | 0.011 | 0.000 | 0.000 | 0.001 | 0.001 | 0.006 |
| RB2+ | 0.084 | 0.097 | 0.096 | 0.102 | 0.096 | 0.093 |
| S | 0.065 | 0.059 | 0.062 | 0.057 | 0.062 | 0.058 |
| TE1 | 0.016 | 0.000 |  | 0.001 | 0.000 | 0.004 |
| TE2+ | 0.068 | 0.101 | 0.076 | 0.105 | 0.082 | 0.103 |
| WR1-2 | 0.021 | 0.001 | 0.004 | 0.003 | 0.006 | 0.004 |
| WR3+ | 0.099 | 0.102 | 0.111 | 0.102 | 0.112 | 0.102 |

QB check on test. Because candidates are the `ACT` (game-day active) list, a listed QB1 who was declared inactive is never a candidate; those plays are counted from the weekly roster status. Among plays with an ACT depth-1 QB the residual is split with the post-game snap-count file (it lists only players who took a snap, so this split is diagnostic, not a feature): `dressed non-starter` = the depth chart listed a QB who never played that day, `took a snap` = the residual is an in-game change (injury, garbage time, wildcat / trick plays). The depth-chart QB1 is on the ACT list on 91.3% of test plays and is the QB on the field on 91.1% of them; 4.3% of those plays had him dressed but never taking a snap (stale depth chart / listed non-starter), and when he did play that day he is off the field on 4.8% of plays (in-game changes). On 7.5% of plays the listed QB1 was declared inactive and on 1.2% he was on a reserve list or missing from the depth chart; the weekly roster status resolves those pre-kickoff.

| candidate | condition | plays | share_of_plays | on_field_rate |
|---|---|---|---|---|
| depth-chart QB1 | depth-1 QB is on the ACT list (a candidate) | 13701 | 0.9134 | 0.9112 |
| depth-chart QB1 |   ... and took a snap that day (post-game split) | 13114 | 0.8743 | 0.9520 |
| depth-chart QB1 |   ... and took no snap that day: dressed non-starter (post-game split) | 587 | 0.0391 | 0.0000 |
| depth-chart QB1 | no depth-1 QB on the ACT list: listed QB1 declared inactive (INA) | 1126 | 0.0751 |  |
| depth-chart QB1 | no depth-1 QB on the ACT list: listed QB1 on a reserve / other list | 119 | 0.0079 |  |
| depth-chart QB1 | no depth-1 QB on the ACT list: no QB1 on the depth chart | 54 | 0.0036 |  |
| prior-usage QB1 | all plays | 15000 | 1.0000 | 0.8621 |
| prior-usage QB1 |   ... and took a snap that day (post-game split) | 13649 | 0.9099 | 0.9474 |
| prior-usage QB1 |   ... and took no snap that day: dressed non-starter (post-game split) | 1351 | 0.0901 | 0.0000 |

## Caveats

- Tendency features are computed from *true* labels of earlier games. In a fully label-free deployment (in-season, college) those histories would themselves have to be imputed, so S2 is optimistic about the tendency channel; S0/S1 are the label-free floor.
- `wp` is nflfastR's pre-play model output, itself a function of situation and pre-game spread; it is not tracking-derived.
- Bootstrap CIs are given over plays and game-clustered; the clustered interval is the honest one and is the wider of the two for the small `S2 -> S2+imp_pers` deltas. Per-play test losses of every stage (a) / (b) model (and baselines) are saved as `processed/nfl/participation_test_losses_a.parquet` / `participation_test_losses_b.parquet` and per-game loss sums as `nfl_02a_offense_game_losses.parquet` / `nfl_02b_defense_game_losses.parquet`, so any clustered comparison can be recomputed without refitting.
- The 2023 participation file has a different provenance (position-level personnel strings, different coverage vocabulary) and was not used.
- Stage (c) uses a seeded subsample of plays for speed (sizes stated in that section). Its candidate set is the weekly roster with status `ACT`, which in these files is exactly the 48-man game-day active list (the declared inactives carry status `INA`, about 6 per team-week, and none of them takes a snap). Both statuses are fixed about 90 minutes before kickoff, so the candidate set is pre-play but not mid-week information; a mid-week forecast would have to model the inactive list itself. The residual QB / OL errors therefore come from dressed non-starters, in-game substitutions and depth-chart staleness, not from unknown inactives (see the QB check in section (c)).
