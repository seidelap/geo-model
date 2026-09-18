# Soccer 04 - transfer of the 360-trained students to seasons without 360

Machine-written by `research/privileged_tracking/soccer/transfer.py`. Question: does the event-only defensive-state student trained on 2020-2025 360 data transfer to event-only club football (2015/16 Premier League, La Liga, Serie A, Ligue 1) and to non-360 tournaments (World Cup 2018, Copa America 2024, AFCON 2023) - as a measurement (vs the shot freeze frame), as a downstream feature (xG, xPass) and at team level?

## Headline

- Shots: 37,488 non-penalty shots / 3,569 goals in 1,517 2015/16 matches; 3,541 / 296 in 148 non-360 tournament matches; 29,577 shots with a rebuilt key pass.
- xG log-loss, leagues_2015/16 (within-domain CV): BASE 0.3144, LOC 0.2760, EVENT 0.2552, EVENT+IMP 0.2551, EVENT+ORACLESHOT 0.2489, _full 0.2473; zero-shot 360 models: EVENT 0.2599, EVENT+IMP 0.2600, EVENT+ORACLESHOT 0.2533; StatsBomb xG 0.2484.
- xG log-loss, tournaments_no360 (within-domain CV): BASE 0.2877, LOC n/a, EVENT 0.2528, EVENT+IMP 0.2521, EVENT+ORACLESHOT 0.2509, _full n/a; zero-shot 360 models: EVENT 0.2469, EVENT+IMP 0.2454, EVENT+ORACLESHOT 0.2430; StatsBomb xG 0.2398.
- xPass 2015/16 (after): n = 149,621, completion 0.775; log-loss BASE 0.5332, EVENT 0.2515, EVENT+IMP 0.2502, EVENT-nodur 0.2666.
- xPass 2015/16 (noafter): n = 149,621, completion 0.775; log-loss BASE 0.5332, EVENT 0.4097, EVENT+IMP 0.4078.

### Reading

- xG in the 2015/16 leagues (within-domain CV, frozen students): EVENT+IMP vs EVENT no measurable difference (+0.0000, CI [-0.0004, +0.0005]); EVENT+ORACLESHOT significant gain (+0.0063); StatsBomb xG significant gain (+0.0068).
- Zero-shot stage-03 xG models on 2015/16 shots: EVENT (zero-shot) vs within-2015/16 EVENT significant loss (-0.0047); EVENT+IMP (zero-shot) no measurable difference (-0.0001, CI [-0.0005, +0.0003]); EVENT+ORACLESHOT (zero-shot) significant gain (+0.0065) (deltas of the zero-shot IMP / ORACLESHOT models are vs the zero-shot EVENT model; the zero-shot vs within-CV EVENT gap mixes domain shift with training-set size: 10,233 360 shots vs ~30k 2015/16 shots per training fold).
- Per league (5-fold CV inside the league): PL 2015/16: IMP +0.0017 [+0.0004, +0.0029], ORACLESHOT +0.0048 [+0.0026, +0.0069]; La Liga 2015/16: IMP -0.0006 [-0.0017, +0.0004], ORACLESHOT +0.0055 [+0.0028, +0.0081]; Serie A 2015/16: IMP -0.0005 [-0.0016, +0.0006], ORACLESHOT +0.0061 [+0.0038, +0.0085]; Ligue 1 2015/16: IMP +0.0010 [-0.0003, +0.0022], ORACLESHOT +0.0057 [+0.0032, +0.0082]. 1 of 4 leagues show a nominally significant IMP gain; with four tests one nominal hit is expected by chance, so the pooled estimate is the primary result.
- xPass 2015/16: with the destination known EVENT+IMP significant gain (+0.0013), dropping the realised duration from EVENT significant loss (-0.0151); pre-instant EVENT+IMP significant gain (+0.0019).
- Oracle check (imputed vs shot.freeze_frame; the in-domain bias is the shot-frame vs 360-frame measurement gap, the transfer-specific part is the change from the 360 OOF rows to 2015/16): n_opp_in_cone: R2 vs shot frame 0.409 (360 OOF) -> 0.408 (2015/16), MAE 0.60 -> 0.56, bias -0.18 -> -0.09 (transfer-specific +0.09); nearest_opp_dist_in_cone: R2 vs shot frame 0.430 (360 OOF) -> 0.424 (2015/16), MAE 2.38 -> 2.50, bias +0.53 -> +0.36 (transfer-specific -0.17); opp_keeper_dist_to_goal_line: R2 vs shot frame 0.150 (360 OOF) -> 0.208 (2015/16), MAE 1.14 -> 1.19, bias +0.50 -> +0.27 (transfer-specific -0.23).
- Team level (2015/16): Spearman of the imputed block depth conceded with def_line_x +0.79 (team-season, n 80) / +0.77 (team-match, n 3034); with ppda -0.55 (team-season, n 80) / -0.33 (team-match, n 3034); with passes allowed +0.07 (team-season, n 80) / +0.00 (team-match, n 3034). Comparators at team-season level vs def_line_x: mean ball x alone -0.78 (sign reversed by construction), E0 student +0.79 - the ranking is mostly where the ball is, not what the window features add.

### Verdict

- **Does the 360-trained student transfer to event-only club data?** NO measurable downstream value: on 2015/16 shots the frozen imputations change the xG log-loss by +0.0000 [-0.0004, +0.0005] vs an event-only model that already sees every student input, while the shot freeze frame (the real oracle) gives +0.0063 [+0.0051, +0.0075].
- As a *measurement* the student survives essentially unchanged: against the shot freeze frame its R2 goes n_opp_in_cone 0.41 -> 0.41, nearest_opp_dist_in_cone 0.43 -> 0.42, opp_keeper_dist_to_goal_line 0.15 -> 0.21 from the 360 out-of-fold shots to 2015/16 (MAE / bias / calibration in the oracle tables). The shot state it recovers was already weak in-domain, so what transfers is a weak signal.
- Passes: the frozen pass-state students give +0.0013 [+0.0007, +0.0019] with the destination known and +0.0019 [+0.0014, +0.0024] pre-instant (both significant; in-domain stage 03: +0.0005 / +0.0016).
- Team-level aggregates of the imputed block depth rank teams consistently with the event-only style proxies (Spearman table), but the E2 student reads the opponent's recent defensive-action x directly, so this is a consistency check, not independent validation.

## Protocol

- Populations: Shot rows of `events_no360.parquet` in the four 2015/16 leagues (pooled and per league) and in the three non-360 tournaments (World Cup 2018, Copa America 2024, AFCON 2023, pooled); penalties excluded; label `post_shot_outcome == 'Goal'`. Passes: the 2015/16 Pass rows (the build stage kept a fixed 25% of passes per match), `Unknown` / `Injury Clearance` dropped, match-stratified subsample. Stray non-360 matches of other competitions (9 360-flagged fall-backs) are excluded.
- Students: frozen stage-02 bundles (`students_E0/E2/E2a.joblib`, trained on all 417 360 matches) as applied by `apply_student` (`imputed_no360.parquet`); the assist pass of every shot is rebuilt with the build-stage feature code (`process_match`, all passes of the match) because the 25% pass sample misses most key passes, then imputed with the same bundles; the receiver-distance student (no stage-02 model) is fitted once on the 360 Pass rows and applied here. No non-360 match ever enters a student's training set.
- Downstream models: LightGBM (stage-03 settings), `group_kfold` by match inside the target population (seed 0, 5 folds), rounds by early stopping on an inner match holdout; xG refitted on all training rows and averaged over seeds (pooled 3, per league 2); xPass single seed without refit. BASE = training-fold positive rate. Zero-shot = the stage-03 `xg_models.joblib` (EVENT / EVENT+IMP / EVENT+ORACLESHOT fitted on the 10,233 360 shots) applied as is; constant design columns of a single-gender / single-competition-type domain (`f_gender`, `f_comp_type`) are restored from the undropped design.
- Features: EVENT = stage-02 E2 design of the shot minus `f_after_duration` plus the key pass's attributes (incl. its realised trajectory and duration, complete before the shot); EVENT+IMP adds the frozen E2 shot-state and E2a assist-lane imputations (clipped at 0); EVENT+ORACLESHOT the same quantities from `shot.freeze_frame`; `_full` every `sff_*` column. xPass EVENT = the E2a (after) or E2 (noafter) design incl. `f_after_duration` in the after design (every student input is in the baseline); EVENT+IMP adds the seven pass-state imputations (E2a / E2 students). Never used as features: `shot.freeze_frame` outside the oracle variants, `statsbomb_xg`, `one_on_one`, `open_goal`, `post_*`, any `y_*`.
- Metrics: log-loss, Brier, AUC, ECE (10 equal-count bins); paired per-sample bootstrap of the per-sample loss (2000 resamples for shots, 500 for passes, seed 0) with a match-clustered CI next to it; `significant` = the per-sample CI excludes 0. Positive delta = better than the reference.
- Oracle check: imputed E2 shot state vs the `shot.freeze_frame` quantity on every shot where both exist (`sff_nearest_opp_dist_in_cone` is NaN when the cone is empty, ~49% of shots; the students were trained on the 360 frame, whose cone count agrees with the shot frame only to r 0.75, so the 360 out-of-fold rows scored against the same shot frame are the like-for-like reference). Calibration = equal-count bins of the imputed value, mean imputed vs mean observed.

## (a) Oracle check under domain shift: imputed shot state vs shot.freeze_frame

Per group: n shots where both values exist, R2 / MAE / bias (imputed - observed) / correlation, means; for counts the exact and within-1 agreement of the rounded imputation. `360 ... (OOF)` rows are the stage-02 out-of-fold imputations of the 360 shots scored against the same shot frame (like for like) and against the 360 frame (their own label).

### Summary: R2 / MAE / bias (imputed - observed) vs the shot frame per group; the last column is the 360 out-of-fold R2 against the 360 frame (the students' own label)

| target | metric | leagues_2015/16 | PL 2015/16 | La Liga 2015/16 | Serie A 2015/16 | Ligue 1 2015/16 | WC 2018 | Copa 2024 | AFCON 2023 | 360 all (OOF) | 360 male club season (OOF) | 360 male tournament (OOF) | 360 female tournament (OOF) | 360 all (OOF) vs 360 frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n_opp_in_cone | r2 | 0.408 | 0.378 | 0.419 | 0.402 | 0.434 | 0.233 | 0.390 | 0.430 | 0.409 | 0.453 | 0.381 | 0.386 | 0.421 |
| n_opp_in_cone | mae | 0.563 | 0.586 | 0.541 | 0.557 | 0.567 | 0.617 | 0.582 | 0.576 | 0.600 | 0.550 | 0.591 | 0.658 |  |
| n_opp_in_cone | bias | -0.090 | -0.145 | -0.037 | -0.062 | -0.115 | -0.124 | -0.094 | -0.135 | -0.179 | -0.101 | -0.175 | -0.260 |  |
| nearest_opp_dist_in_cone | r2 | 0.424 | 0.408 | 0.426 | 0.453 | 0.403 | 0.463 | 0.450 | 0.462 | 0.430 | 0.444 | 0.420 | 0.428 | 0.427 |
| nearest_opp_dist_in_cone | mae | 2.500 | 2.446 | 2.479 | 2.488 | 2.599 | 2.687 | 2.441 | 2.659 | 2.382 | 2.210 | 2.501 | 2.390 |  |
| nearest_opp_dist_in_cone | bias | 0.363 | 0.196 | 0.535 | 0.557 | 0.164 | -0.344 | 0.429 | 0.310 | 0.535 | 0.399 | 0.686 | 0.479 |  |
| opp_keeper_dist_to_goal_line | r2 | 0.208 | 0.197 | 0.230 | 0.128 | 0.240 | -0.158 | 0.337 | 0.115 | 0.150 | 0.160 | 0.171 | 0.063 | 0.247 |
| opp_keeper_dist_to_goal_line | mae | 1.192 | 1.166 | 1.244 | 1.137 | 1.230 | 1.487 | 1.389 | 1.249 | 1.141 | 1.148 | 1.081 | 1.206 |  |
| opp_keeper_dist_to_goal_line | bias | 0.270 | 0.227 | 0.264 | 0.454 | 0.116 | 1.015 | 0.417 | 0.470 | 0.505 | 0.370 | 0.476 | 0.671 |  |
| n_opp_within_5 | r2 | 0.506 | 0.487 | 0.498 | 0.523 | 0.514 | 0.498 | 0.569 | 0.566 | 0.525 | 0.511 | 0.539 | 0.513 | 0.495 |
| n_opp_within_5 | mae | 0.675 | 0.681 | 0.682 | 0.665 | 0.671 | 0.695 | 0.690 | 0.675 | 0.720 | 0.702 | 0.703 | 0.757 |  |
| n_opp_within_5 | bias | 0.030 | 0.013 | -0.017 | 0.052 | 0.072 | 0.003 | 0.003 | -0.004 | -0.029 | -0.027 | -0.043 | -0.013 |  |

**n_opp_in_cone**

| group | observed | n | r2 | mae | bias | corr | mean_imp | mean_obs | sd_obs | exact | within_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leagues_2015/16 | shot.freeze_frame | 37034 | 0.408 | 0.563 | -0.090 | 0.646 | 0.686 | 0.776 | 0.972 | 0.559 | 0.943 |
| PL 2015/16 | shot.freeze_frame | 9726 | 0.378 | 0.586 | -0.145 | 0.632 | 0.676 | 0.820 | 0.996 | 0.546 | 0.935 |
| La Liga 2015/16 | shot.freeze_frame | 8957 | 0.419 | 0.541 | -0.037 | 0.650 | 0.680 | 0.717 | 0.948 | 0.582 | 0.951 |
| Serie A 2015/16 | shot.freeze_frame | 9756 | 0.402 | 0.557 | -0.062 | 0.642 | 0.713 | 0.775 | 0.938 | 0.549 | 0.949 |
| Ligue 1 2015/16 | shot.freeze_frame | 8595 | 0.434 | 0.567 | -0.115 | 0.669 | 0.672 | 0.787 | 1.005 | 0.560 | 0.938 |
| WC 2018 | shot.freeze_frame | 1625 | 0.233 | 0.617 | -0.124 | 0.524 | 0.674 | 0.798 | 0.948 | 0.523 | 0.910 |
| Copa 2024 | shot.freeze_frame | 731 | 0.390 | 0.582 | -0.094 | 0.634 | 0.688 | 0.782 | 0.968 | 0.544 | 0.943 |
| AFCON 2023 | shot.freeze_frame | 1150 | 0.430 | 0.576 | -0.135 | 0.669 | 0.713 | 0.849 | 1.022 | 0.553 | 0.932 |
| 360 all (OOF) | shot.freeze_frame | 10067 | 0.409 | 0.600 | -0.179 | 0.669 | 0.672 | 0.851 | 1.054 | 0.538 | 0.930 |
| 360 male club season (OOF) | shot.freeze_frame | 3086 | 0.453 | 0.550 | -0.101 | 0.685 | 0.638 | 0.738 | 1.007 | 0.572 | 0.945 |
| 360 male tournament (OOF) | shot.freeze_frame | 3808 | 0.381 | 0.591 | -0.175 | 0.646 | 0.644 | 0.818 | 0.989 | 0.529 | 0.937 |
| 360 female tournament (OOF) | shot.freeze_frame | 3173 | 0.386 | 0.658 | -0.260 | 0.675 | 0.739 | 0.999 | 1.153 | 0.516 | 0.908 |
| 360 all (OOF) | 360 frame | 10057 | 0.421 | 0.560 | -0.009 | 0.650 | 0.672 | 0.681 | 0.992 | 0.565 | 0.946 |
| 360 male club season (OOF) | 360 frame | 3082 | 0.442 | 0.537 | 0.022 | 0.666 | 0.638 | 0.616 | 0.969 | 0.589 | 0.946 |
| 360 male tournament (OOF) | 360 frame | 3803 | 0.391 | 0.548 | -0.006 | 0.625 | 0.644 | 0.649 | 0.927 | 0.562 | 0.952 |
| 360 female tournament (OOF) | 360 frame | 3172 | 0.423 | 0.595 | -0.042 | 0.657 | 0.739 | 0.782 | 1.079 | 0.547 | 0.939 |

**nearest_opp_dist_in_cone**

| group | observed | n | r2 | mae | bias | corr | mean_imp | mean_obs | sd_obs |
|---|---|---|---|---|---|---|---|---|---|
| leagues_2015/16 | shot.freeze_frame | 19080 | 0.424 | 2.500 | 0.363 | 0.660 | 6.532 | 6.169 | 4.239 |
| PL 2015/16 | shot.freeze_frame | 5237 | 0.408 | 2.446 | 0.196 | 0.643 | 6.254 | 6.058 | 4.034 |
| La Liga 2015/16 | shot.freeze_frame | 4339 | 0.426 | 2.479 | 0.535 | 0.666 | 6.520 | 5.985 | 4.251 |
| Serie A 2015/16 | shot.freeze_frame | 5139 | 0.453 | 2.488 | 0.557 | 0.688 | 6.839 | 6.282 | 4.328 |
| Ligue 1 2015/16 | shot.freeze_frame | 4365 | 0.403 | 2.599 | 0.164 | 0.642 | 6.517 | 6.354 | 4.349 |
| WC 2018 | shot.freeze_frame | 871 | 0.463 | 2.687 | -0.344 | 0.684 | 6.483 | 6.828 | 4.903 |
| Copa 2024 | shot.freeze_frame | 380 | 0.450 | 2.441 | 0.429 | 0.679 | 6.223 | 5.794 | 4.265 |
| AFCON 2023 | shot.freeze_frame | 627 | 0.462 | 2.659 | 0.310 | 0.685 | 6.759 | 6.449 | 4.570 |
| 360 all (OOF) | shot.freeze_frame | 5412 | 0.430 | 2.382 | 0.535 | 0.670 | 5.964 | 5.429 | 3.954 |
| 360 male club season (OOF) | shot.freeze_frame | 1500 | 0.444 | 2.210 | 0.399 | 0.675 | 6.000 | 5.602 | 3.726 |
| 360 male tournament (OOF) | shot.freeze_frame | 2040 | 0.420 | 2.501 | 0.686 | 0.670 | 6.276 | 5.590 | 4.101 |
| 360 female tournament (OOF) | shot.freeze_frame | 1872 | 0.428 | 2.390 | 0.479 | 0.665 | 5.594 | 5.115 | 3.949 |
| 360 all (OOF) | 360 frame | 4466 | 0.427 | 2.473 | 0.020 | 0.654 | 6.295 | 6.275 | 4.144 |
| 360 male club season (OOF) | 360 frame | 1257 | 0.433 | 2.260 | 0.050 | 0.658 | 6.285 | 6.235 | 3.836 |
| 360 male tournament (OOF) | 360 frame | 1676 | 0.433 | 2.594 | 0.065 | 0.659 | 6.606 | 6.541 | 4.359 |
| 360 female tournament (OOF) | 360 frame | 1533 | 0.412 | 2.516 | -0.054 | 0.642 | 5.964 | 6.019 | 4.129 |

**opp_keeper_dist_to_goal_line**

| group | observed | n | r2 | mae | bias | corr | mean_imp | mean_obs | sd_obs |
|---|---|---|---|---|---|---|---|---|---|
| leagues_2015/16 | shot.freeze_frame | 36974 | 0.208 | 1.192 | 0.270 | 0.478 | 2.890 | 2.620 | 1.965 |
| PL 2015/16 | shot.freeze_frame | 9707 | 0.197 | 1.166 | 0.227 | 0.462 | 2.848 | 2.621 | 1.847 |
| La Liga 2015/16 | shot.freeze_frame | 8940 | 0.230 | 1.244 | 0.264 | 0.494 | 2.870 | 2.606 | 2.192 |
| Serie A 2015/16 | shot.freeze_frame | 9745 | 0.128 | 1.137 | 0.454 | 0.468 | 2.910 | 2.455 | 1.705 |
| Ligue 1 2015/16 | shot.freeze_frame | 8582 | 0.240 | 1.230 | 0.116 | 0.493 | 2.938 | 2.822 | 2.097 |
| WC 2018 | shot.freeze_frame | 1623 | -0.158 | 1.487 | 1.015 | 0.295 | 2.720 | 1.705 | 2.108 |
| Copa 2024 | shot.freeze_frame | 726 | 0.337 | 1.389 | 0.417 | 0.618 | 2.897 | 2.480 | 2.585 |
| AFCON 2023 | shot.freeze_frame | 1149 | 0.115 | 1.249 | 0.470 | 0.435 | 2.844 | 2.375 | 1.888 |
| 360 all (OOF) | shot.freeze_frame | 10059 | 0.150 | 1.141 | 0.505 | 0.464 | 2.811 | 2.306 | 1.976 |
| 360 male club season (OOF) | shot.freeze_frame | 3085 | 0.160 | 1.148 | 0.370 | 0.431 | 2.902 | 2.532 | 2.383 |
| 360 male tournament (OOF) | shot.freeze_frame | 3802 | 0.171 | 1.081 | 0.476 | 0.486 | 2.787 | 2.311 | 1.860 |
| 360 female tournament (OOF) | shot.freeze_frame | 3172 | 0.063 | 1.206 | 0.671 | 0.492 | 2.750 | 2.080 | 1.617 |
| 360 all (OOF) | 360 frame | 9474 | 0.247 | 1.036 | -0.006 | 0.498 | 2.779 | 2.785 | 1.794 |
| 360 male club season (OOF) | 360 frame | 2997 | 0.238 | 1.091 | -0.037 | 0.492 | 2.877 | 2.914 | 2.017 |
| 360 male tournament (OOF) | 360 frame | 3601 | 0.244 | 1.027 | 0.042 | 0.494 | 2.761 | 2.719 | 1.693 |
| 360 female tournament (OOF) | 360 frame | 2876 | 0.259 | 0.991 | -0.033 | 0.510 | 2.698 | 2.732 | 1.658 |

**n_opp_within_5**

| group | observed | n | r2 | mae | bias | corr | mean_imp | mean_obs | sd_obs | exact | within_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leagues_2015/16 | shot.freeze_frame | 37488 | 0.506 | 0.675 | 0.030 | 0.712 | 1.758 | 1.728 | 1.248 | 0.460 | 0.916 |
| PL 2015/16 | shot.freeze_frame | 9817 | 0.487 | 0.681 | 0.013 | 0.698 | 1.783 | 1.769 | 1.234 | 0.457 | 0.913 |
| La Liga 2015/16 | shot.freeze_frame | 9071 | 0.498 | 0.682 | -0.017 | 0.706 | 1.735 | 1.752 | 1.254 | 0.461 | 0.913 |
| Serie A 2015/16 | shot.freeze_frame | 9877 | 0.523 | 0.665 | 0.052 | 0.725 | 1.736 | 1.683 | 1.253 | 0.463 | 0.921 |
| Ligue 1 2015/16 | shot.freeze_frame | 8723 | 0.514 | 0.671 | 0.072 | 0.720 | 1.781 | 1.709 | 1.251 | 0.460 | 0.916 |
| WC 2018 | shot.freeze_frame | 1638 | 0.498 | 0.695 | 0.003 | 0.706 | 1.738 | 1.734 | 1.279 | 0.441 | 0.911 |
| Copa 2024 | shot.freeze_frame | 741 | 0.569 | 0.690 | 0.003 | 0.760 | 1.833 | 1.830 | 1.378 | 0.470 | 0.906 |
| AFCON 2023 | shot.freeze_frame | 1162 | 0.566 | 0.675 | -0.004 | 0.757 | 1.748 | 1.751 | 1.348 | 0.454 | 0.917 |
| 360 all (OOF) | shot.freeze_frame | 10233 | 0.525 | 0.720 | -0.029 | 0.726 | 1.914 | 1.943 | 1.356 | 0.435 | 0.901 |
| 360 male club season (OOF) | shot.freeze_frame | 3143 | 0.511 | 0.702 | -0.027 | 0.716 | 1.828 | 1.855 | 1.302 | 0.443 | 0.905 |
| 360 male tournament (OOF) | shot.freeze_frame | 3859 | 0.539 | 0.703 | -0.043 | 0.737 | 1.831 | 1.874 | 1.337 | 0.437 | 0.914 |
| 360 female tournament (OOF) | shot.freeze_frame | 3231 | 0.513 | 0.757 | -0.013 | 0.716 | 2.098 | 2.111 | 1.413 | 0.425 | 0.881 |
| 360 all (OOF) | 360 frame | 10223 | 0.495 | 0.730 | 0.016 | 0.704 | 1.914 | 1.899 | 1.338 | 0.433 | 0.893 |
| 360 male club season (OOF) | 360 frame | 3139 | 0.459 | 0.715 | 0.022 | 0.678 | 1.829 | 1.807 | 1.265 | 0.438 | 0.901 |
| 360 male tournament (OOF) | 360 frame | 3854 | 0.504 | 0.714 | 0.012 | 0.710 | 1.830 | 1.818 | 1.315 | 0.440 | 0.905 |
| 360 female tournament (OOF) | 360 frame | 3230 | 0.502 | 0.765 | 0.014 | 0.708 | 2.097 | 2.083 | 1.414 | 0.420 | 0.872 |

### Calibration (equal-count bins of the imputed value; mean imputed vs mean observed)


**n_opp_in_cone**

| bin | 360 all (OOF): imputed | 360 all (OOF): observed | leagues_2015/16: imputed | leagues_2015/16: observed | WC 2018: imputed | WC 2018: observed | Copa 2024: imputed | Copa 2024: observed | AFCON 2023: imputed | AFCON 2023: observed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.09 | 0.18 | 0.09 | 0.17 | 0.08 | 0.21 | 0.10 | 0.22 | 0.09 | 0.14 |
| 1 | 0.19 | 0.25 | 0.18 | 0.24 | 0.18 | 0.29 | 0.19 | 0.29 | 0.17 | 0.23 |
| 2 | 0.27 | 0.39 | 0.26 | 0.38 | 0.26 | 0.56 | 0.27 | 0.33 | 0.25 | 0.38 |
| 3 | 0.36 | 0.48 | 0.35 | 0.46 | 0.35 | 0.54 | 0.35 | 0.49 | 0.34 | 0.53 |
| 4 | 0.45 | 0.62 | 0.43 | 0.56 | 0.44 | 0.69 | 0.44 | 0.56 | 0.44 | 0.57 |
| 5 | 0.55 | 0.72 | 0.52 | 0.64 | 0.53 | 0.73 | 0.53 | 0.58 | 0.55 | 0.60 |
| 6 | 0.68 | 0.82 | 0.65 | 0.76 | 0.65 | 0.77 | 0.66 | 0.89 | 0.69 | 0.85 |
| 7 | 0.84 | 1.02 | 0.84 | 0.94 | 0.83 | 0.98 | 0.83 | 0.99 | 0.91 | 1.14 |
| 8 | 1.16 | 1.44 | 1.21 | 1.29 | 1.14 | 1.23 | 1.18 | 1.21 | 1.33 | 1.53 |
| 9 | 2.13 | 2.59 | 2.33 | 2.33 | 2.29 | 1.99 | 2.35 | 2.29 | 2.37 | 2.51 |

**nearest_opp_dist_in_cone**

| bin | 360 all (OOF): imputed | 360 all (OOF): observed | leagues_2015/16: imputed | leagues_2015/16: observed | WC 2018: imputed | WC 2018: observed | Copa 2024: imputed | Copa 2024: observed | AFCON 2023: imputed | AFCON 2023: observed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.53 | 1.93 | 2.59 | 2.60 | 2.58 | 2.26 | 2.54 | 2.12 | 2.53 | 2.21 |
| 1 | 3.29 | 2.82 | 3.39 | 3.28 | 3.28 | 3.51 | 3.28 | 2.20 | 3.28 | 2.79 |
| 2 | 3.80 | 3.32 | 4.02 | 3.64 | 3.84 | 3.90 | 3.94 | 3.79 | 4.03 | 3.47 |
| 3 | 4.35 | 3.87 | 4.74 | 4.31 | 4.47 | 4.60 | 4.53 | 4.27 | 4.91 | 4.34 |
| 4 | 4.96 | 4.32 | 5.50 | 5.13 | 5.25 | 5.67 | 5.15 | 4.48 | 5.66 | 5.85 |
| 5 | 5.70 | 5.18 | 6.38 | 6.07 | 6.14 | 6.41 | 5.97 | 5.62 | 6.65 | 6.85 |
| 6 | 6.59 | 6.16 | 7.45 | 7.36 | 7.31 | 8.21 | 7.02 | 6.27 | 7.76 | 7.81 |
| 7 | 7.70 | 7.46 | 8.60 | 8.43 | 8.50 | 9.42 | 8.10 | 8.44 | 8.92 | 8.74 |
| 8 | 9.06 | 8.73 | 9.99 | 9.61 | 10.15 | 10.96 | 9.39 | 9.80 | 10.59 | 10.73 |
| 9 | 11.66 | 10.52 | 12.67 | 11.26 | 13.35 | 13.40 | 12.30 | 10.94 | 13.46 | 11.88 |

**opp_keeper_dist_to_goal_line**

| bin | 360 all (OOF): imputed | 360 all (OOF): observed | leagues_2015/16: imputed | leagues_2015/16: observed | WC 2018: imputed | WC 2018: observed | Copa 2024: imputed | Copa 2024: observed | AFCON 2023: imputed | AFCON 2023: observed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.57 | 1.44 | 1.54 | 1.55 | 1.48 | 1.12 | 1.55 | 1.30 | 1.51 | 1.40 |
| 1 | 2.10 | 1.65 | 2.14 | 1.80 | 2.08 | 1.35 | 2.11 | 1.46 | 2.10 | 1.60 |
| 2 | 2.28 | 1.77 | 2.31 | 1.91 | 2.25 | 1.34 | 2.27 | 1.80 | 2.26 | 1.71 |
| 3 | 2.43 | 1.85 | 2.46 | 2.01 | 2.39 | 1.34 | 2.42 | 1.74 | 2.37 | 1.85 |
| 4 | 2.58 | 1.94 | 2.61 | 2.22 | 2.50 | 1.48 | 2.57 | 1.91 | 2.52 | 2.02 |
| 5 | 2.74 | 2.13 | 2.79 | 2.52 | 2.64 | 1.49 | 2.76 | 1.98 | 2.72 | 2.47 |
| 6 | 2.92 | 2.30 | 3.00 | 2.78 | 2.82 | 1.52 | 2.97 | 2.32 | 2.93 | 2.38 |
| 7 | 3.18 | 2.57 | 3.29 | 3.09 | 3.04 | 1.73 | 3.25 | 2.70 | 3.23 | 2.62 |
| 8 | 3.60 | 3.00 | 3.73 | 3.56 | 3.45 | 2.12 | 3.77 | 3.53 | 3.72 | 3.63 |
| 9 | 4.72 | 4.41 | 5.03 | 4.77 | 4.56 | 3.57 | 5.35 | 6.13 | 5.10 | 4.08 |

## (b) Goal prediction on 2015/16 shots (students frozen from the 360 domain)


### leagues_2015/16

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 37488 | 3569 | 0.3144 | 0.0861 | 0.4916 | 0.0038 | 0.0952 |
| LOC | 37488 | 3569 | 0.2760 | 0.0782 | 0.7497 | 0.0051 | 0.0952 |
| EVENT | 37488 | 3569 | 0.2552 | 0.0725 | 0.8038 | 0.0034 | 0.0947 |
| EVENT+IMP(E0) | 37488 | 3569 | 0.2550 | 0.0724 | 0.8041 | 0.0033 | 0.0947 |
| EVENT+IMP | 37488 | 3569 | 0.2551 | 0.0724 | 0.8036 | 0.0031 | 0.0947 |
| EVENT+ORACLESHOT | 37488 | 3569 | 0.2489 | 0.0708 | 0.8170 | 0.0040 | 0.0947 |
| EVENT+ORACLESHOT_full | 37488 | 3569 | 0.2473 | 0.0701 | 0.8185 | 0.0033 | 0.0947 |
| EVENT (zero-shot) | 37488 | 3569 | 0.2599 | 0.0736 | 0.7944 | 0.0103 | 0.0875 |
| EVENT+IMP (zero-shot) | 37488 | 3569 | 0.2600 | 0.0736 | 0.7945 | 0.0122 | 0.0881 |
| EVENT+ORACLESHOT (zero-shot) | 37488 | 3569 | 0.2533 | 0.0719 | 0.8069 | 0.0068 | 0.0924 |
| STATSBOMB_XG | 37488 | 3569 | 0.2484 | 0.0699 | 0.8155 | 0.0097 | 0.0916 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 37488 | -0.0593 [-0.0628, -0.0559] | [-0.0628, -0.0557] | -0.01366 [-0.01472, -0.01262] | True |
| LOC | 37488 | -0.0208 [-0.0227, -0.0189] | [-0.0227, -0.0186] | -0.00575 [-0.00637, -0.00514] | True |
| EVENT+IMP(E0) | 37488 | +0.0001 [-0.0002, +0.0005] | [-0.0002, +0.0005] | +0.00004 [-0.00008, +0.00017] | False |
| EVENT+IMP | 37488 | +0.0000 [-0.0004, +0.0005] | [-0.0004, +0.0005] | +0.00004 [-0.00010, +0.00017] | False |
| EVENT+ORACLESHOT | 37488 | +0.0063 [+0.0051, +0.0075] | [+0.0052, +0.0074] | +0.00169 [+0.00132, +0.00208] | True |
| EVENT+ORACLESHOT_full | 37488 | +0.0079 [+0.0066, +0.0092] | [+0.0067, +0.0091] | +0.00234 [+0.00192, +0.00277] | True |
| EVENT (zero-shot) | 37488 | -0.0047 [-0.0059, -0.0035] | [-0.0059, -0.0035] | -0.00114 [-0.00150, -0.00080] | True |
| EVENT+IMP (zero-shot) | 37488 | -0.0048 [-0.0060, -0.0036] | [-0.0060, -0.0036] | -0.00112 [-0.00148, -0.00078] | True |
| EVENT+ORACLESHOT (zero-shot) | 37488 | +0.0018 [+0.0003, +0.0033] | [+0.0003, +0.0033] | +0.00061 [+0.00014, +0.00109] | True |
| STATSBOMB_XG | 37488 | +0.0068 [+0.0050, +0.0086] | [+0.0050, +0.0086] | +0.00255 [+0.00199, +0.00309] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 37488 | -0.0001 [-0.0005, +0.0003] | [-0.0005, +0.0003] | +0.00002 [-0.00011, +0.00015] | False |
| EVENT+ORACLESHOT (zero-shot) | 37488 | +0.0065 [+0.0053, +0.0078] | [+0.0053, +0.0077] | +0.00176 [+0.00135, +0.00217] | True |
| STATSBOMB_XG | 37488 | +0.0115 [+0.0096, +0.0134] | [+0.0096, +0.0134] | +0.00369 [+0.00309, +0.00431] | True |

### tournaments_no360

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 3541 | 296 | 0.2877 | 0.0766 | 0.4729 | 0.0096 | 0.0836 |
| EVENT | 3541 | 296 | 0.2528 | 0.0696 | 0.7487 | 0.0114 | 0.0781 |
| EVENT+IMP | 3541 | 296 | 0.2521 | 0.0690 | 0.7461 | 0.0102 | 0.0777 |
| EVENT+ORACLESHOT | 3541 | 296 | 0.2509 | 0.0688 | 0.7504 | 0.0089 | 0.0785 |
| EVENT (zero-shot) | 3541 | 296 | 0.2469 | 0.0685 | 0.7698 | 0.0128 | 0.0857 |
| EVENT+IMP (zero-shot) | 3541 | 296 | 0.2454 | 0.0680 | 0.7732 | 0.0114 | 0.0872 |
| EVENT+ORACLESHOT (zero-shot) | 3541 | 296 | 0.2430 | 0.0673 | 0.7770 | 0.0097 | 0.0884 |
| STATSBOMB_XG | 3541 | 296 | 0.2398 | 0.0656 | 0.7837 | 0.0131 | 0.0896 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 3541 | -0.0349 [-0.0436, -0.0260] | [-0.0438, -0.0257] | -0.00703 [-0.00927, -0.00475] | True |
| EVENT+IMP | 3541 | +0.0006 [-0.0022, +0.0035] | [-0.0023, +0.0033] | +0.00057 [-0.00026, +0.00143] | False |
| EVENT+ORACLESHOT | 3541 | +0.0019 [-0.0020, +0.0057] | [-0.0021, +0.0061] | +0.00080 [-0.00031, +0.00190] | False |
| EVENT (zero-shot) | 3541 | +0.0059 [+0.0005, +0.0113] | [+0.0010, +0.0108] | +0.00110 [-0.00051, +0.00280] | True |
| EVENT+IMP (zero-shot) | 3541 | +0.0073 [+0.0021, +0.0126] | [+0.0024, +0.0124] | +0.00159 [-0.00003, +0.00324] | True |
| EVENT+ORACLESHOT (zero-shot) | 3541 | +0.0098 [+0.0033, +0.0162] | [+0.0030, +0.0165] | +0.00232 [+0.00010, +0.00443] | True |
| STATSBOMB_XG | 3541 | +0.0129 [+0.0061, +0.0197] | [+0.0051, +0.0201] | +0.00403 [+0.00183, +0.00616] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 3541 | +0.0015 [+0.0001, +0.0029] | [+0.0001, +0.0028] | +0.00049 [+0.00008, +0.00090] | True |
| EVENT+ORACLESHOT (zero-shot) | 3541 | +0.0039 [+0.0001, +0.0077] | [-0.0002, +0.0082] | +0.00122 [-0.00007, +0.00246] | True |
| STATSBOMB_XG | 3541 | +0.0070 [+0.0012, +0.0127] | [+0.0005, +0.0131] | +0.00293 [+0.00119, +0.00471] | True |

### PL 2015/16

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 9817 | 914 | 0.3099 | 0.0845 | 0.4722 | 0.0124 | 0.0931 |
| EVENT | 9817 | 914 | 0.2674 | 0.0753 | 0.7651 | 0.0086 | 0.0917 |
| EVENT+IMP | 9817 | 914 | 0.2657 | 0.0749 | 0.7701 | 0.0083 | 0.0911 |
| EVENT+ORACLESHOT | 9817 | 914 | 0.2626 | 0.0738 | 0.7747 | 0.0039 | 0.0916 |
| EVENT (zero-shot) | 9817 | 914 | 0.2648 | 0.0745 | 0.7709 | 0.0070 | 0.0902 |
| EVENT+IMP (zero-shot) | 9817 | 914 | 0.2640 | 0.0742 | 0.7727 | 0.0057 | 0.0904 |
| EVENT+ORACLESHOT (zero-shot) | 9817 | 914 | 0.2595 | 0.0731 | 0.7824 | 0.0088 | 0.0932 |
| STATSBOMB_XG | 9817 | 914 | 0.2529 | 0.0705 | 0.7954 | 0.0077 | 0.0917 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 9817 | -0.0425 [-0.0481, -0.0365] | [-0.0484, -0.0368] | -0.00914 [-0.01068, -0.00750] | True |
| EVENT+IMP | 9817 | +0.0017 [+0.0004, +0.0029] | [+0.0005, +0.0029] | +0.00046 [+0.00007, +0.00083] | True |
| EVENT+ORACLESHOT | 9817 | +0.0048 [+0.0026, +0.0069] | [+0.0026, +0.0070] | +0.00159 [+0.00086, +0.00231] | True |
| EVENT (zero-shot) | 9817 | +0.0025 [+0.0000, +0.0049] | [+0.0001, +0.0048] | +0.00079 [+0.00008, +0.00150] | True |
| EVENT+IMP (zero-shot) | 9817 | +0.0034 [+0.0011, +0.0057] | [+0.0010, +0.0057] | +0.00112 [+0.00041, +0.00182] | True |
| EVENT+ORACLESHOT (zero-shot) | 9817 | +0.0079 [+0.0048, +0.0110] | [+0.0044, +0.0113] | +0.00227 [+0.00129, +0.00326] | True |
| STATSBOMB_XG | 9817 | +0.0145 [+0.0105, +0.0185] | [+0.0105, +0.0183] | +0.00485 [+0.00363, +0.00605] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 9817 | +0.0009 [+0.0000, +0.0017] | [+0.0001, +0.0017] | +0.00033 [+0.00005, +0.00058] | True |
| EVENT+ORACLESHOT (zero-shot) | 9817 | +0.0053 [+0.0032, +0.0075] | [+0.0030, +0.0077] | +0.00147 [+0.00072, +0.00219] | True |
| STATSBOMB_XG | 9817 | +0.0120 [+0.0082, +0.0155] | [+0.0084, +0.0156] | +0.00405 [+0.00295, +0.00521] | True |

### La Liga 2015/16

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 9071 | 945 | 0.3343 | 0.0933 | 0.4837 | 0.0074 | 0.1042 |
| EVENT | 9071 | 945 | 0.2702 | 0.0781 | 0.8066 | 0.0069 | 0.1025 |
| EVENT+IMP | 9071 | 945 | 0.2708 | 0.0782 | 0.8051 | 0.0073 | 0.1023 |
| EVENT+ORACLESHOT | 9071 | 945 | 0.2647 | 0.0766 | 0.8186 | 0.0075 | 0.1024 |
| EVENT (zero-shot) | 9071 | 945 | 0.2708 | 0.0783 | 0.8105 | 0.0163 | 0.0935 |
| EVENT+IMP (zero-shot) | 9071 | 945 | 0.2719 | 0.0786 | 0.8095 | 0.0186 | 0.0935 |
| EVENT+ORACLESHOT (zero-shot) | 9071 | 945 | 0.2638 | 0.0765 | 0.8227 | 0.0109 | 0.0994 |
| STATSBOMB_XG | 9071 | 945 | 0.2555 | 0.0736 | 0.8328 | 0.0116 | 0.0996 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 9071 | -0.0641 [-0.0713, -0.0574] | [-0.0717, -0.0560] | -0.01529 [-0.01745, -0.01321] | True |
| EVENT+IMP | 9071 | -0.0006 [-0.0017, +0.0004] | [-0.0017, +0.0005] | -0.00016 [-0.00051, +0.00017] | False |
| EVENT+ORACLESHOT | 9071 | +0.0055 [+0.0028, +0.0081] | [+0.0030, +0.0079] | +0.00144 [+0.00053, +0.00231] | True |
| EVENT (zero-shot) | 9071 | -0.0007 [-0.0036, +0.0022] | [-0.0039, +0.0024] | -0.00028 [-0.00121, +0.00060] | False |
| EVENT+IMP (zero-shot) | 9071 | -0.0017 [-0.0046, +0.0012] | [-0.0051, +0.0015] | -0.00057 [-0.00150, +0.00032] | False |
| EVENT+ORACLESHOT (zero-shot) | 9071 | +0.0064 [+0.0028, +0.0100] | [+0.0026, +0.0099] | +0.00159 [+0.00046, +0.00268] | True |
| STATSBOMB_XG | 9071 | +0.0147 [+0.0107, +0.0188] | [+0.0103, +0.0192] | +0.00442 [+0.00307, +0.00574] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 9071 | -0.0011 [-0.0020, -0.0002] | [-0.0019, -0.0002] | -0.00029 [-0.00059, -0.00000] | True |
| EVENT+ORACLESHOT (zero-shot) | 9071 | +0.0070 [+0.0045, +0.0097] | [+0.0044, +0.0097] | +0.00187 [+0.00100, +0.00276] | True |
| STATSBOMB_XG | 9071 | +0.0154 [+0.0115, +0.0194] | [+0.0113, +0.0195] | +0.00470 [+0.00337, +0.00610] | True |

### Serie A 2015/16

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 9877 | 858 | 0.2953 | 0.0793 | 0.4832 | 0.0057 | 0.0869 |
| EVENT | 9877 | 858 | 0.2389 | 0.0669 | 0.8071 | 0.0071 | 0.0846 |
| EVENT+IMP | 9877 | 858 | 0.2394 | 0.0670 | 0.8047 | 0.0063 | 0.0845 |
| EVENT+ORACLESHOT | 9877 | 858 | 0.2328 | 0.0654 | 0.8212 | 0.0074 | 0.0846 |
| EVENT (zero-shot) | 9877 | 858 | 0.2395 | 0.0669 | 0.8052 | 0.0127 | 0.0816 |
| EVENT+IMP (zero-shot) | 9877 | 858 | 0.2398 | 0.0669 | 0.8045 | 0.0166 | 0.0827 |
| EVENT+ORACLESHOT (zero-shot) | 9877 | 858 | 0.2328 | 0.0650 | 0.8170 | 0.0096 | 0.0850 |
| STATSBOMB_XG | 9877 | 858 | 0.2299 | 0.0642 | 0.8265 | 0.0117 | 0.0836 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 9877 | -0.0564 [-0.0632, -0.0498] | [-0.0628, -0.0505] | -0.01241 [-0.01440, -0.01048] | True |
| EVENT+IMP | 9877 | -0.0005 [-0.0016, +0.0006] | [-0.0016, +0.0005] | -0.00005 [-0.00037, +0.00030] | False |
| EVENT+ORACLESHOT | 9877 | +0.0061 [+0.0038, +0.0085] | [+0.0038, +0.0084] | +0.00156 [+0.00086, +0.00230] | True |
| EVENT (zero-shot) | 9877 | -0.0006 [-0.0032, +0.0017] | [-0.0028, +0.0015] | +0.00001 [-0.00073, +0.00071] | False |
| EVENT+IMP (zero-shot) | 9877 | -0.0009 [-0.0035, +0.0015] | [-0.0031, +0.0013] | +0.00003 [-0.00070, +0.00072] | False |
| EVENT+ORACLESHOT (zero-shot) | 9877 | +0.0060 [+0.0029, +0.0090] | [+0.0032, +0.0090] | +0.00188 [+0.00095, +0.00278] | True |
| STATSBOMB_XG | 9877 | +0.0090 [+0.0054, +0.0125] | [+0.0051, +0.0125] | +0.00270 [+0.00167, +0.00374] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 9877 | -0.0003 [-0.0010, +0.0005] | [-0.0010, +0.0005] | +0.00002 [-0.00021, +0.00027] | False |
| EVENT+ORACLESHOT (zero-shot) | 9877 | +0.0067 [+0.0045, +0.0089] | [+0.0046, +0.0089] | +0.00187 [+0.00113, +0.00263] | True |
| STATSBOMB_XG | 9877 | +0.0096 [+0.0060, +0.0131] | [+0.0061, +0.0132] | +0.00270 [+0.00160, +0.00371] | True |

### Ligue 1 2015/16

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 8723 | 852 | 0.3200 | 0.0881 | 0.4899 | 0.0039 | 0.0977 |
| EVENT | 8723 | 852 | 0.2661 | 0.0755 | 0.7883 | 0.0067 | 0.0957 |
| EVENT+IMP | 8723 | 852 | 0.2651 | 0.0751 | 0.7890 | 0.0094 | 0.0956 |
| EVENT+ORACLESHOT | 8723 | 852 | 0.2604 | 0.0738 | 0.7991 | 0.0109 | 0.0951 |
| EVENT (zero-shot) | 8723 | 852 | 0.2659 | 0.0753 | 0.7887 | 0.0131 | 0.0851 |
| EVENT+IMP (zero-shot) | 8723 | 852 | 0.2660 | 0.0753 | 0.7887 | 0.0149 | 0.0862 |
| EVENT+ORACLESHOT (zero-shot) | 8723 | 852 | 0.2588 | 0.0734 | 0.8023 | 0.0091 | 0.0925 |
| STATSBOMB_XG | 8723 | 852 | 0.2567 | 0.0719 | 0.8041 | 0.0101 | 0.0921 |

Paired deltas vs EVENT (within-domain CV):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 8723 | -0.0539 [-0.0608, -0.0470] | [-0.0607, -0.0475] | -0.01262 [-0.01467, -0.01054] | True |
| EVENT+IMP | 8723 | +0.0010 [-0.0003, +0.0022] | [-0.0003, +0.0023] | +0.00038 [-0.00000, +0.00076] | False |
| EVENT+ORACLESHOT | 8723 | +0.0057 [+0.0032, +0.0082] | [+0.0033, +0.0081] | +0.00172 [+0.00096, +0.00252] | True |
| EVENT (zero-shot) | 8723 | +0.0002 [-0.0028, +0.0031] | [-0.0026, +0.0032] | +0.00024 [-0.00064, +0.00108] | False |
| EVENT+IMP (zero-shot) | 8723 | +0.0001 [-0.0028, +0.0030] | [-0.0027, +0.0029] | +0.00024 [-0.00068, +0.00107] | False |
| EVENT+ORACLESHOT (zero-shot) | 8723 | +0.0073 [+0.0036, +0.0108] | [+0.0037, +0.0110] | +0.00207 [+0.00092, +0.00320] | True |
| STATSBOMB_XG | 8723 | +0.0094 [+0.0050, +0.0136] | [+0.0048, +0.0140] | +0.00360 [+0.00238, +0.00488] | True |

Paired deltas vs EVENT (zero-shot):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (zero-shot) | 8723 | -0.0001 [-0.0010, +0.0008] | [-0.0010, +0.0008] | -0.00000 [-0.00027, +0.00026] | False |
| EVENT+ORACLESHOT (zero-shot) | 8723 | +0.0071 [+0.0046, +0.0095] | [+0.0044, +0.0098] | +0.00183 [+0.00098, +0.00267] | True |
| STATSBOMB_XG | 8723 | +0.0091 [+0.0050, +0.0132] | [+0.0048, +0.0134] | +0.00336 [+0.00217, +0.00465] | True |

### In-domain reference (stage 03, 360 matches)

| task | variant | n | log_loss | auc | delta_log_loss | ci_low | ci_high | design |
|---|---|---|---|---|---|---|---|---|
| xG (360, 5-fold) | BASE | 10233 | 0.3286 | 0.4820 | -0.0541 | -0.0604 | -0.0478 |  |
| xG (360, 5-fold) | LOC | 10233 | 0.2945 | 0.7242 | -0.0200 | -0.0238 | -0.0164 |  |
| xG (360, 5-fold) | EVENT | 10233 | 0.2745 | 0.7847 |  |  |  |  |
| xG (360, 5-fold) | EVENT+IMP(E0) | 10233 | 0.2744 | 0.7836 | 0.0001 | -0.0008 | 0.0010 |  |
| xG (360, 5-fold) | EVENT+IMP | 10233 | 0.2748 | 0.7842 | -0.0003 | -0.0013 | 0.0007 |  |
| xG (360, 5-fold) | EVENT+ORACLE360 | 10233 | 0.2716 | 0.7891 | 0.0029 | 0.0013 | 0.0045 |  |
| xG (360, 5-fold) | EVENT+ORACLESHOT | 10233 | 0.2666 | 0.8003 | 0.0078 | 0.0055 | 0.0102 |  |
| xG (360, 5-fold) | EVENT+ORACLESHOT_full | 10233 | 0.2639 | 0.8039 | 0.0106 | 0.0080 | 0.0133 |  |
| xG (360, 5-fold) | STATSBOMB_XG | 10233 | 0.2617 | 0.8091 | 0.0128 | 0.0090 | 0.0168 |  |
| xPass (360, 5-fold) | BASE | 149591 | 0.4624 | 0.4916 | -0.2535 | -0.2565 | -0.2506 | after |
| xPass (360, 5-fold) | EVENT | 149591 | 0.2089 | 0.9498 |  |  |  | after |
| xPass (360, 5-fold) | EVENT-nodur | 149591 | 0.2229 | 0.9426 | -0.0140 | -0.0148 | -0.0131 | after |
| xPass (360, 5-fold) | EVENT+IMP | 149591 | 0.2083 | 0.9498 | 0.0005 | -0.0001 | 0.0011 | after |
| xPass (360, 5-fold) | EVENT+ORACLE | 149591 | 0.2009 | 0.9542 | 0.0080 | 0.0072 | 0.0088 | after |
| xPass (360, 5-fold) | BASE | 149591 | 0.4624 | 0.4916 | -0.1138 | -0.1160 | -0.1114 | noafter |
| xPass (360, 5-fold) | EVENT | 149591 | 0.3486 | 0.8234 |  |  |  | noafter |
| xPass (360, 5-fold) | EVENT+IMP | 149591 | 0.3470 | 0.8258 | 0.0016 | 0.0010 | 0.0022 | noafter |
| xPass (360, 5-fold) | EVENT+ORACLE | 149591 | 0.2783 | 0.9039 | 0.0703 | 0.0685 | 0.0721 | noafter |

### Calibration, 2015/16 pooled (equal-count bins: mean predicted vs observed goal rate)

| bin | obs_EVENT | obs_EVENT (zero-shot) | obs_EVENT+IMP | obs_EVENT+IMP (zero-shot) | obs_EVENT+ORACLESHOT | obs_STATSBOMB_XG | pred_EVENT | pred_EVENT (zero-shot) | pred_EVENT+IMP | pred_EVENT+IMP (zero-shot) | pred_EVENT+ORACLESHOT | pred_STATSBOMB_XG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.010 | 0.014 | 0.011 | 0.014 | 0.008 | 0.019 | 0.013 | 0.022 | 0.013 | 0.024 | 0.013 | 0.009 |
| 1 | 0.018 | 0.024 | 0.018 | 0.027 | 0.016 | 0.014 | 0.020 | 0.025 | 0.019 | 0.027 | 0.019 | 0.019 |
| 2 | 0.025 | 0.028 | 0.025 | 0.025 | 0.023 | 0.021 | 0.027 | 0.029 | 0.027 | 0.031 | 0.025 | 0.026 |
| 3 | 0.037 | 0.033 | 0.035 | 0.033 | 0.030 | 0.031 | 0.036 | 0.036 | 0.036 | 0.038 | 0.033 | 0.034 |
| 4 | 0.048 | 0.042 | 0.043 | 0.042 | 0.041 | 0.034 | 0.046 | 0.044 | 0.046 | 0.046 | 0.042 | 0.044 |
| 5 | 0.053 | 0.055 | 0.054 | 0.055 | 0.052 | 0.050 | 0.060 | 0.054 | 0.059 | 0.055 | 0.055 | 0.057 |
| 6 | 0.079 | 0.081 | 0.081 | 0.082 | 0.073 | 0.078 | 0.080 | 0.070 | 0.078 | 0.071 | 0.073 | 0.073 |
| 7 | 0.119 | 0.118 | 0.114 | 0.116 | 0.118 | 0.103 | 0.112 | 0.098 | 0.110 | 0.097 | 0.106 | 0.100 |
| 8 | 0.178 | 0.179 | 0.185 | 0.182 | 0.191 | 0.190 | 0.177 | 0.154 | 0.178 | 0.153 | 0.180 | 0.159 |
| 9 | 0.386 | 0.378 | 0.385 | 0.376 | 0.400 | 0.411 | 0.376 | 0.345 | 0.381 | 0.339 | 0.401 | 0.394 |

### Slices, 2015/16 pooled (log-loss per variant; delta vs EVENT with per-shot CI)


**domain**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLESHOT | EVENT (zero-shot) | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLESHOT) | d(EVENT (zero-shot)) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| La Liga 2015/16 | 9071 | 945 | 0.2655 | 0.2663 | 0.2588 | 0.2708 | 0.2555 | -0.0008 [-0.0017, +0.0001] | +0.0067 [+0.0041, +0.0092] | -0.0053 [-0.0079, -0.0030] | +0.0100 [+0.0063, +0.0136] |
| Ligue 1 2015/16 | 8723 | 852 | 0.2622 | 0.2616 | 0.2551 | 0.2659 | 0.2567 | +0.0006 [-0.0003, +0.0015] | +0.0071 [+0.0049, +0.0095] | -0.0036 [-0.0062, -0.0010] | +0.0055 [+0.0016, +0.0092] |
| PL 2015/16 | 9817 | 914 | 0.2606 | 0.2605 | 0.2558 | 0.2648 | 0.2529 | +0.0001 [-0.0007, +0.0009] | +0.0048 [+0.0027, +0.0069] | -0.0043 [-0.0066, -0.0019] | +0.0077 [+0.0038, +0.0113] |
| Serie A 2015/16 | 9877 | 858 | 0.2341 | 0.2339 | 0.2275 | 0.2395 | 0.2299 | +0.0002 [-0.0007, +0.0010] | +0.0066 [+0.0045, +0.0088] | -0.0054 [-0.0077, -0.0033] | +0.0042 [+0.0008, +0.0074] |

**distance_band**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLESHOT | EVENT (zero-shot) | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLESHOT) | d(EVENT (zero-shot)) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-8 | 2936 | 918 | 0.5408 | 0.5382 | 0.5275 | 0.5565 | 0.5060 | +0.0026 [+0.0006, +0.0045] | +0.0133 [+0.0070, +0.0194] | -0.0157 [-0.0226, -0.0088] | +0.0348 [+0.0257, +0.0440] |
| 8-12 | 5709 | 891 | 0.3878 | 0.3873 | 0.3750 | 0.3950 | 0.3704 | +0.0005 [-0.0010, +0.0020] | +0.0128 [+0.0088, +0.0170] | -0.0072 [-0.0109, -0.0034] | +0.0173 [+0.0118, +0.0232] |
| 12-18 | 8399 | 1010 | 0.3293 | 0.3300 | 0.3193 | 0.3315 | 0.3185 | -0.0007 [-0.0017, +0.0004] | +0.0100 [+0.0069, +0.0133] | -0.0022 [-0.0048, +0.0004] | +0.0108 [+0.0065, +0.0153] |
| 18-25 | 9630 | 500 | 0.1936 | 0.1937 | 0.1901 | 0.1958 | 0.1919 | -0.0001 [-0.0007, +0.0005] | +0.0035 [+0.0017, +0.0052] | -0.0023 [-0.0042, -0.0006] | +0.0017 [-0.0007, +0.0041] |
| 25+ | 10814 | 250 | 0.1049 | 0.1051 | 0.1044 | 0.1094 | 0.1098 | -0.0001 [-0.0005, +0.0002] | +0.0006 [+0.0001, +0.0011] | -0.0044 [-0.0061, -0.0029] | -0.0048 [-0.0078, -0.0020] |

**play_pattern**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLESHOT | EVENT (zero-shot) | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLESHOT) | d(EVENT (zero-shot)) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| counter | 1706 | 265 | 0.3543 | 0.3542 | 0.3409 | 0.3627 | 0.3425 | +0.0001 [-0.0022, +0.0027] | +0.0134 [+0.0059, +0.0196] | -0.0084 [-0.0157, -0.0016] | +0.0118 [+0.0002, +0.0230] |
| other | 679 | 67 | 0.2379 | 0.2388 | 0.2233 | 0.2535 | 0.2242 | -0.0009 [-0.0047, +0.0027] | +0.0146 [+0.0072, +0.0223] | -0.0156 [-0.0262, -0.0060] | +0.0137 [+0.0034, +0.0245] |
| regular | 12613 | 1238 | 0.2533 | 0.2535 | 0.2475 | 0.2573 | 0.2462 | -0.0002 [-0.0010, +0.0004] | +0.0058 [+0.0038, +0.0078] | -0.0040 [-0.0059, -0.0022] | +0.0070 [+0.0040, +0.0099] |
| set_piece | 22490 | 1999 | 0.2493 | 0.2490 | 0.2435 | 0.2537 | 0.2431 | +0.0002 [-0.0003, +0.0007] | +0.0058 [+0.0044, +0.0071] | -0.0044 [-0.0061, -0.0028] | +0.0061 [+0.0036, +0.0085] |

**has_assist**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLESHOT | EVENT (zero-shot) | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLESHOT) | d(EVENT (zero-shot)) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assisted | 27035 | 2596 | 0.2609 | 0.2610 | 0.2551 | 0.2646 | 0.2526 | -0.0001 [-0.0006, +0.0004] | +0.0058 [+0.0046, +0.0071] | -0.0037 [-0.0050, -0.0023] | +0.0083 [+0.0064, +0.0102] |
| unassisted | 10453 | 973 | 0.2404 | 0.2400 | 0.2330 | 0.2477 | 0.2373 | +0.0004 [-0.0005, +0.0012] | +0.0074 [+0.0051, +0.0097] | -0.0073 [-0.0099, -0.0047] | +0.0031 [-0.0013, +0.0072] |

## (c) Pass completion in 2015/16 (no pass oracle without 360)

`after` = realised end location / length / height / flags / duration are features (E2a students); `noafter` = pre-instant only (E2 students). `EVENT-nodur` drops the realised duration from the after design (what that one field is worth).

| design | variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|---|
| after | BASE | 149621 | 115950 | 0.5332 | 0.1744 | 0.4934 | 0.0085 | 0.7750 |
| after | EVENT | 149621 | 115950 | 0.2515 | 0.0784 | 0.9413 | 0.0045 | 0.7755 |
| after | EVENT-nodur | 149621 | 115950 | 0.2666 | 0.0834 | 0.9340 | 0.0041 | 0.7754 |
| after | EVENT+IMP | 149621 | 115950 | 0.2502 | 0.0782 | 0.9419 | 0.0035 | 0.7754 |
| noafter | BASE | 149621 | 115950 | 0.5332 | 0.1744 | 0.4934 | 0.0085 | 0.7750 |
| noafter | EVENT | 149621 | 115950 | 0.4097 | 0.1295 | 0.8104 | 0.0044 | 0.7752 |
| noafter | EVENT+IMP | 149621 | 115950 | 0.4078 | 0.1289 | 0.8131 | 0.0027 | 0.7752 |

Paired deltas vs EVENT:

| design | variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|---|
| after | BASE | 149621 | -0.2818 [-0.2845, -0.2789] | [-0.2850, -0.2787] | -0.09599 [-0.09713, -0.09477] | True |
| after | EVENT-nodur | 149621 | -0.0151 [-0.0160, -0.0143] | [-0.0161, -0.0143] | -0.00493 [-0.00522, -0.00459] | True |
| after | EVENT+IMP | 149621 | +0.0013 [+0.0007, +0.0019] | [+0.0006, +0.0019] | +0.00026 [+0.00006, +0.00046] | True |
| noafter | BASE | 149621 | -0.1235 [-0.1258, -0.1215] | [-0.1258, -0.1213] | -0.04491 [-0.04586, -0.04405] | True |
| noafter | EVENT+IMP | 149621 | +0.0019 [+0.0014, +0.0024] | [+0.0014, +0.0025] | +0.00057 [+0.00037, +0.00076] | True |

Receiver-distance student fitted on the 360 Pass rows:

| target | fset | rounds | n_train | seconds |
|---|---|---|---|---|
| nearest_opp_to_receiver | E2a | 397 | 300000 | 15.7678 |
| nearest_opp_to_receiver | E2 | 242 | 300000 | 11.0321 |

### xPass calibration (after design)

| bin | obs_EVENT | obs_EVENT+IMP | obs_EVENT-nodur | pred_EVENT | pred_EVENT+IMP | pred_EVENT-nodur |
|---|---|---|---|---|---|---|
| 0 | 0.101 | 0.101 | 0.125 | 0.103 | 0.108 | 0.130 |
| 1 | 0.334 | 0.335 | 0.346 | 0.328 | 0.329 | 0.340 |
| 2 | 0.599 | 0.596 | 0.595 | 0.603 | 0.592 | 0.595 |
| 3 | 0.830 | 0.830 | 0.818 | 0.846 | 0.838 | 0.831 |
| 4 | 0.934 | 0.935 | 0.924 | 0.938 | 0.939 | 0.929 |
| 5 | 0.973 | 0.974 | 0.967 | 0.970 | 0.973 | 0.966 |
| 6 | 0.987 | 0.987 | 0.985 | 0.984 | 0.987 | 0.982 |
| 7 | 0.994 | 0.995 | 0.993 | 0.991 | 0.993 | 0.990 |
| 8 | 0.998 | 0.998 | 0.998 | 0.995 | 0.997 | 0.994 |
| 9 | 0.999 | 1.000 | 0.999 | 0.998 | 0.998 | 0.997 |

### xPass slices (after design)


**domain**

| level | n | positives | EVENT | EVENT+IMP | EVENT-nodur | d(EVENT+IMP) | d(EVENT-nodur) |
|---|---|---|---|---|---|---|---|
| La Liga 2015/16 | 36916 | 28186 | 0.2589 | 0.2562 | 0.2758 | +0.0027 [+0.0016, +0.0038] | -0.0169 [-0.0189, -0.0152] |
| Ligue 1 2015/16 | 37763 | 29418 | 0.2537 | 0.2526 | 0.2686 | +0.0011 [-0.0002, +0.0022] | -0.0149 [-0.0168, -0.0132] |
| PL 2015/16 | 37076 | 28742 | 0.2547 | 0.2533 | 0.2685 | +0.0014 [+0.0002, +0.0024] | -0.0138 [-0.0157, -0.0118] |
| Serie A 2015/16 | 37866 | 29604 | 0.2389 | 0.2388 | 0.2538 | +0.0000 [-0.0011, +0.0011] | -0.0149 [-0.0167, -0.0131] |

**length_band**

| level | n | positives | EVENT | EVENT+IMP | EVENT-nodur | d(EVENT+IMP) | d(EVENT-nodur) |
|---|---|---|---|---|---|---|---|
| 0-10 | 29239 | 22250 | 0.2218 | 0.2201 | 0.2797 | +0.0017 [+0.0003, +0.0031] | -0.0578 [-0.0611, -0.0540] |
| 10-20 | 58342 | 51303 | 0.2002 | 0.1985 | 0.2057 | +0.0017 [+0.0009, +0.0025] | -0.0055 [-0.0065, -0.0045] |
| 20-35 | 39238 | 30952 | 0.2440 | 0.2432 | 0.2473 | +0.0008 [-0.0003, +0.0020] | -0.0034 [-0.0043, -0.0024] |
| 35+ | 22802 | 11445 | 0.4336 | 0.4331 | 0.4389 | +0.0005 [-0.0010, +0.0021] | -0.0053 [-0.0069, -0.0036] |

**height**

| level | n | positives | EVENT | EVENT+IMP | EVENT-nodur | d(EVENT+IMP) | d(EVENT-nodur) |
|---|---|---|---|---|---|---|---|
| Ground Pass | 94243 | 85943 | 0.1551 | 0.1532 | 0.1720 | +0.0019 [+0.0013, +0.0026] | -0.0169 [-0.0181, -0.0157] |
| High Pass | 34684 | 15367 | 0.4922 | 0.4924 | 0.4980 | -0.0003 [-0.0015, +0.0011] | -0.0058 [-0.0071, -0.0045] |
| Low Pass | 20694 | 14640 | 0.2870 | 0.2861 | 0.3096 | +0.0009 [-0.0007, +0.0028] | -0.0226 [-0.0256, -0.0192] |

**play_pattern**

| level | n | positives | EVENT | EVENT+IMP | EVENT-nodur | d(EVENT+IMP) | d(EVENT-nodur) |
|---|---|---|---|---|---|---|---|
| counter | 982 | 852 | 0.2172 | 0.2130 | 0.2221 | +0.0042 [-0.0024, +0.0111] | -0.0049 [-0.0142, +0.0043] |
| other | 4204 | 3170 | 0.2635 | 0.2601 | 0.2768 | +0.0034 [+0.0001, +0.0071] | -0.0133 [-0.0197, -0.0073] |
| regular | 63282 | 52133 | 0.2111 | 0.2097 | 0.2271 | +0.0014 [+0.0005, +0.0022] | -0.0160 [-0.0173, -0.0144] |
| set_piece | 81153 | 59795 | 0.2828 | 0.2817 | 0.2974 | +0.0011 [+0.0002, +0.0019] | -0.0147 [-0.0159, -0.0134] |

**under_pressure**

| level | n | positives | EVENT | EVENT+IMP | EVENT-nodur | d(EVENT+IMP) | d(EVENT-nodur) |
|---|---|---|---|---|---|---|---|
| pressed | 23162 | 15978 | 0.2974 | 0.2964 | 0.3179 | +0.0009 [-0.0007, +0.0024] | -0.0206 [-0.0236, -0.0179] |
| unpressed | 126459 | 99972 | 0.2431 | 0.2417 | 0.2572 | +0.0014 [+0.0007, +0.0019] | -0.0141 [-0.0150, -0.0133] |

### xPass slices (noafter design)


**domain**

| level | n | positives | EVENT | EVENT+IMP | d(EVENT+IMP) |
|---|---|---|---|---|---|
| La Liga 2015/16 | 36916 | 28186 | 0.4181 | 0.4156 | +0.0025 [+0.0014, +0.0035] |
| Ligue 1 2015/16 | 37763 | 29418 | 0.4070 | 0.4049 | +0.0022 [+0.0011, +0.0032] |
| PL 2015/16 | 37076 | 28742 | 0.4080 | 0.4058 | +0.0022 [+0.0010, +0.0031] |
| Serie A 2015/16 | 37866 | 29604 | 0.4060 | 0.4051 | +0.0009 [-0.0000, +0.0019] |

**length_band**

| level | n | positives | EVENT | EVENT+IMP | d(EVENT+IMP) |
|---|---|---|---|---|---|
| 0-10 | 29239 | 22250 | 0.4329 | 0.4317 | +0.0012 [-0.0001, +0.0023] |
| 10-20 | 58342 | 51303 | 0.2936 | 0.2913 | +0.0023 [+0.0017, +0.0030] |
| 20-35 | 39238 | 30952 | 0.3802 | 0.3776 | +0.0026 [+0.0014, +0.0035] |
| 35+ | 22802 | 11445 | 0.7281 | 0.7273 | +0.0008 [-0.0010, +0.0029] |

## (d) Team-level sanity (2015/16 leagues)

Imputed block depth / defensive line conceded = mean of the E2 (and E0) imputations over the opponent's possession Pass / Carry rows (25% sample) per match, attributed to the defending team; `ball_x_mean` = mean attacking-frame x of those rows (location-only comparator). Proxies from `team_match.parquet`: `def_line_x` (mean x of the team's defensive actions in its own frame), `ppda`, `ppda_passes_allowed`, `possession`. Spearman rank correlations:

| level | imputed | proxy | n | spearman | p_value |
|---|---|---|---|---|---|
| team-match | imp_block_depth__E2 | def_line_x | 3034 | 0.767 | 0.000 |
| team-match | imp_block_depth__E2 | ppda | 3034 | -0.326 | 0.000 |
| team-match | imp_block_depth__E2 | ppda_passes_allowed | 3034 | 0.002 | 0.909 |
| team-match | imp_block_depth__E2 | possession | 3034 | 0.459 | 0.000 |
| team-match | imp_block_depth__E0 | def_line_x | 3034 | 0.754 | 0.000 |
| team-match | imp_block_depth__E0 | ppda | 3034 | -0.285 | 0.000 |
| team-match | imp_block_depth__E0 | ppda_passes_allowed | 3034 | 0.036 | 0.045 |
| team-match | imp_block_depth__E0 | possession | 3034 | 0.434 | 0.000 |
| team-match | imp_def_line__E2 | def_line_x | 3034 | 0.762 | 0.000 |
| team-match | imp_def_line__E2 | ppda | 3034 | -0.305 | 0.000 |
| team-match | imp_def_line__E2 | ppda_passes_allowed | 3034 | 0.023 | 0.213 |
| team-match | imp_def_line__E2 | possession | 3034 | 0.441 | 0.000 |
| team-match | imp_def_line__E0 | def_line_x | 3034 | 0.751 | 0.000 |
| team-match | imp_def_line__E0 | ppda | 3034 | -0.274 | 0.000 |
| team-match | imp_def_line__E0 | ppda_passes_allowed | 3034 | 0.043 | 0.017 |
| team-match | imp_def_line__E0 | possession | 3034 | 0.427 | 0.000 |
| team-match | ball_x_mean | def_line_x | 3034 | -0.706 | 0.000 |
| team-match | ball_x_mean | ppda | 3034 | 0.172 | 0.000 |
| team-match | ball_x_mean | ppda_passes_allowed | 3034 | -0.143 | 0.000 |
| team-match | ball_x_mean | possession | 3034 | -0.343 | 0.000 |
| team-season | imp_block_depth__E2 | def_line_x | 80 | 0.793 | 0.000 |
| team-season | imp_block_depth__E2 | ppda | 80 | -0.555 | 0.000 |
| team-season | imp_block_depth__E2 | ppda_passes_allowed | 80 | 0.069 | 0.540 |
| team-season | imp_block_depth__E2 | possession | 80 | 0.480 | 0.000 |
| team-season | imp_block_depth__E0 | def_line_x | 80 | 0.794 | 0.000 |
| team-season | imp_block_depth__E0 | ppda | 80 | -0.536 | 0.000 |
| team-season | imp_block_depth__E0 | ppda_passes_allowed | 80 | 0.062 | 0.585 |
| team-season | imp_block_depth__E0 | possession | 80 | 0.490 | 0.000 |
| team-season | imp_def_line__E2 | def_line_x | 80 | 0.793 | 0.000 |
| team-season | imp_def_line__E2 | ppda | 80 | -0.544 | 0.000 |
| team-season | imp_def_line__E2 | ppda_passes_allowed | 80 | 0.078 | 0.489 |
| team-season | imp_def_line__E2 | possession | 80 | 0.471 | 0.000 |
| team-season | imp_def_line__E0 | def_line_x | 80 | 0.794 | 0.000 |
| team-season | imp_def_line__E0 | ppda | 80 | -0.531 | 0.000 |
| team-season | imp_def_line__E0 | ppda_passes_allowed | 80 | 0.066 | 0.560 |
| team-season | imp_def_line__E0 | possession | 80 | 0.487 | 0.000 |
| team-season | ball_x_mean | def_line_x | 80 | -0.782 | 0.000 |
| team-season | ball_x_mean | ppda | 80 | 0.476 | 0.000 |
| team-season | ball_x_mean | ppda_passes_allowed | 80 | -0.173 | 0.125 |
| team-season | ball_x_mean | possession | 80 | -0.419 | 0.000 |
| team-season La Liga | imp_block_depth__E2 | def_line_x | 20 | 0.830 | 0.000 |
| team-season La Liga | imp_block_depth__E2 | ppda | 20 | -0.597 | 0.005 |
| team-season La Liga | imp_def_line__E2 | def_line_x | 20 | 0.832 | 0.000 |
| team-season La Liga | imp_def_line__E2 | ppda | 20 | -0.586 | 0.007 |
| team-season Ligue 1 | imp_block_depth__E2 | def_line_x | 20 | 0.875 | 0.000 |
| team-season Ligue 1 | imp_block_depth__E2 | ppda | 20 | -0.681 | 0.001 |
| team-season Ligue 1 | imp_def_line__E2 | def_line_x | 20 | 0.883 | 0.000 |
| team-season Ligue 1 | imp_def_line__E2 | ppda | 20 | -0.684 | 0.001 |
| team-season Premier League | imp_block_depth__E2 | def_line_x | 20 | 0.783 | 0.000 |
| team-season Premier League | imp_block_depth__E2 | ppda | 20 | -0.708 | 0.000 |
| team-season Premier League | imp_def_line__E2 | def_line_x | 20 | 0.808 | 0.000 |
| team-season Premier League | imp_def_line__E2 | ppda | 20 | -0.720 | 0.000 |
| team-season Serie A | imp_block_depth__E2 | def_line_x | 20 | 0.925 | 0.000 |
| team-season Serie A | imp_block_depth__E2 | ppda | 20 | -0.627 | 0.003 |
| team-season Serie A | imp_def_line__E2 | def_line_x | 20 | 0.928 | 0.000 |
| team-season Serie A | imp_def_line__E2 | ppda | 20 | -0.606 | 0.005 |

Team-season means (top / bottom 5 by imputed block depth conceded, pooled leagues):

| competition | team | n_matches | imp_block_depth__E2 | imp_def_line__E2 | imp_block_depth__E0 | ball_x_mean | def_line_x | ppda | ppda_passes_allowed |
|---|---|---|---|---|---|---|---|---|---|
| Serie A | Napoli | 38 | 55.33 | 43.49 | 54.93 | 54.85 | 55.75 | 1.82 | 321.68 |
| Ligue 1 | Lyon | 38 | 54.55 | 42.82 | 53.97 | 56.31 | 54.87 | 1.63 | 295.97 |
| Serie A | Lazio | 38 | 53.41 | 41.76 | 52.87 | 57.18 | 54.03 | 1.74 | 333.05 |
| Ligue 1 | Lille | 38 | 53.04 | 41.50 | 52.55 | 57.43 | 53.43 | 1.81 | 333.84 |
| La Liga | Barcelona | 38 | 52.93 | 41.40 | 52.46 | 58.88 | 53.90 | 1.74 | 244.00 |
| Premier League | West Bromwich Albion | 38 | 47.89 | 37.16 | 48.08 | 63.09 | 49.48 | 2.32 | 355.16 |
| Premier League | Crystal Palace | 38 | 47.71 | 36.98 | 47.66 | 63.34 | 50.35 | 1.97 | 316.50 |
| Premier League | Everton | 38 | 47.69 | 36.94 | 47.68 | 63.77 | 47.36 | 2.30 | 307.03 |
| Premier League | Newcastle United | 38 | 47.58 | 36.79 | 47.66 | 64.03 | 46.55 | 2.23 | 324.61 |
| Serie A | Frosinone | 38 | 47.31 | 36.62 | 47.00 | 64.02 | 49.24 | 1.86 | 323.95 |

### Domain shift: distribution of the imputed quantities by domain vs the 360 domain

Possession Pass / Carry rows and Shot rows; `360 ... OOF imputed` = stage-02 out-of-fold imputations on the same event types, `truth` = the 360 label where valid (reliable frames only for shape targets, so truth and imputed rows differ in selection).


**Pass/Carry (possession)**

| target | source | n | mean | sd | p05 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| block_depth | PL 2015/16 | 150754 | 48.89 | 21.38 | 15.83 | 48.35 | 86.04 |
| block_depth | La Liga 2015/16 | 149536 | 50.03 | 21.27 | 16.24 | 49.80 | 86.76 |
| block_depth | Serie A 2015/16 | 154113 | 50.32 | 21.23 | 16.63 | 49.82 | 86.97 |
| block_depth | Ligue 1 2015/16 | 153825 | 51.84 | 20.59 | 17.41 | 52.26 | 86.90 |
| block_depth | WC 2018 | 27067 | 48.60 | 20.91 | 16.20 | 48.18 | 84.98 |
| block_depth | Copa 2024 | 11805 | 51.38 | 21.33 | 16.63 | 51.38 | 86.46 |
| block_depth | AFCON 2023 | 18867 | 50.20 | 21.21 | 15.82 | 50.43 | 86.06 |
| def_line | PL 2015/16 | 150754 | 37.95 | 18.84 | 9.74 | 36.87 | 71.12 |
| def_line | La Liga 2015/16 | 149536 | 38.91 | 18.76 | 10.11 | 38.17 | 71.77 |
| def_line | Serie A 2015/16 | 154113 | 39.16 | 18.71 | 10.54 | 38.24 | 71.82 |
| def_line | Ligue 1 2015/16 | 153825 | 40.50 | 18.26 | 11.16 | 40.43 | 72.07 |
| def_line | WC 2018 | 27067 | 37.38 | 18.48 | 10.18 | 36.16 | 70.12 |
| def_line | Copa 2024 | 11805 | 39.75 | 18.87 | 10.36 | 39.12 | 71.52 |
| def_line | AFCON 2023 | 18867 | 38.76 | 18.80 | 9.65 | 38.45 | 71.23 |
| n_opp_ahead_of_ball | PL 2015/16 | 150754 | 6.92 | 2.08 | 2.98 | 7.29 | 9.53 |
| n_opp_ahead_of_ball | La Liga 2015/16 | 149536 | 6.92 | 2.07 | 3.00 | 7.28 | 9.51 |
| n_opp_ahead_of_ball | Serie A 2015/16 | 154113 | 7.00 | 2.04 | 3.13 | 7.34 | 9.54 |
| n_opp_ahead_of_ball | Ligue 1 2015/16 | 153825 | 6.94 | 2.03 | 3.13 | 7.29 | 9.48 |
| n_opp_ahead_of_ball | WC 2018 | 27067 | 6.82 | 1.96 | 3.08 | 7.21 | 9.17 |
| n_opp_ahead_of_ball | Copa 2024 | 11805 | 7.01 | 1.96 | 3.17 | 7.44 | 9.27 |
| n_opp_ahead_of_ball | AFCON 2023 | 18867 | 7.00 | 1.99 | 3.09 | 7.47 | 9.24 |
| n_opp_within_5 | PL 2015/16 | 150754 | 0.64 | 0.51 | 0.00 | 0.57 | 1.54 |
| n_opp_within_5 | La Liga 2015/16 | 149536 | 0.63 | 0.50 | 0.00 | 0.55 | 1.52 |
| n_opp_within_5 | Serie A 2015/16 | 154113 | 0.63 | 0.50 | 0.00 | 0.55 | 1.53 |
| n_opp_within_5 | Ligue 1 2015/16 | 153825 | 0.63 | 0.50 | 0.00 | 0.55 | 1.53 |
| n_opp_within_5 | WC 2018 | 27067 | 0.62 | 0.49 | 0.00 | 0.55 | 1.48 |
| n_opp_within_5 | Copa 2024 | 11805 | 0.56 | 0.49 | 0.00 | 0.44 | 1.47 |
| n_opp_within_5 | AFCON 2023 | 18867 | 0.55 | 0.49 | 0.00 | 0.43 | 1.48 |
| nearest_opp_dist | PL 2015/16 | 150754 | 6.67 | 4.74 | 1.67 | 5.53 | 14.26 |
| nearest_opp_dist | La Liga 2015/16 | 149536 | 6.75 | 4.68 | 1.68 | 5.64 | 14.46 |
| nearest_opp_dist | Serie A 2015/16 | 154113 | 6.72 | 4.66 | 1.67 | 5.66 | 14.11 |
| nearest_opp_dist | Ligue 1 2015/16 | 153825 | 6.73 | 4.67 | 1.65 | 5.68 | 14.28 |
| nearest_opp_dist | WC 2018 | 27067 | 6.78 | 4.64 | 1.76 | 5.63 | 14.93 |
| nearest_opp_dist | Copa 2024 | 11805 | 7.39 | 4.83 | 1.75 | 6.30 | 15.68 |
| nearest_opp_dist | AFCON 2023 | 18867 | 7.44 | 4.97 | 1.76 | 6.34 | 15.67 |
| block_depth | 360 all OOF imputed | 663268 | 49.80 | 21.65 | 17.01 | 48.50 | 87.03 |
| block_depth | 360 all truth | 502207 | 44.42 | 20.25 | 15.22 | 42.31 | 82.48 |
| block_depth | 360 tournaments OOF imputed | 439109 | 50.08 | 21.62 | 16.64 | 49.18 | 86.59 |
| block_depth | 360 tournaments truth | 306047 | 43.37 | 20.06 | 14.43 | 41.21 | 81.01 |
| block_depth | 360 club seasons OOF imputed | 224159 | 49.25 | 21.69 | 17.71 | 47.22 | 87.93 |
| block_depth | 360 club seasons truth | 196160 | 46.06 | 20.43 | 16.71 | 44.06 | 84.49 |
| def_line | 360 all OOF imputed | 663268 | 38.72 | 19.22 | 10.94 | 36.75 | 72.34 |
| def_line | 360 all truth | 502207 | 33.97 | 17.89 | 9.18 | 31.31 | 67.97 |
| def_line | 360 tournaments OOF imputed | 439109 | 38.92 | 19.35 | 10.58 | 37.17 | 72.34 |
| def_line | 360 tournaments truth | 306047 | 32.91 | 17.81 | 8.48 | 30.11 | 67.14 |
| def_line | 360 club seasons OOF imputed | 224159 | 38.33 | 18.96 | 11.66 | 36.03 | 72.35 |
| def_line | 360 club seasons truth | 196160 | 35.61 | 17.89 | 10.67 | 33.33 | 69.18 |
| n_opp_ahead_of_ball | 360 all OOF imputed | 663268 | 7.00 | 1.90 | 3.38 | 7.38 | 9.39 |
| n_opp_ahead_of_ball | 360 all truth | 502207 | 6.95 | 2.43 | 2.00 | 7.00 | 10.00 |
| n_opp_ahead_of_ball | 360 tournaments OOF imputed | 439109 | 6.86 | 1.87 | 3.28 | 7.25 | 9.15 |
| n_opp_ahead_of_ball | 360 tournaments truth | 306047 | 6.72 | 2.42 | 2.00 | 7.00 | 10.00 |
| n_opp_ahead_of_ball | 360 club seasons OOF imputed | 224159 | 7.28 | 1.93 | 3.61 | 7.67 | 9.63 |
| n_opp_ahead_of_ball | 360 club seasons truth | 196160 | 7.32 | 2.39 | 3.00 | 8.00 | 10.00 |
| n_opp_within_5 | 360 all OOF imputed | 663268 | 0.60 | 0.51 | 0.00 | 0.51 | 1.52 |
| n_opp_within_5 | 360 all truth | 662969 | 0.60 | 0.74 | 0.00 | 0.00 | 2.00 |
| n_opp_within_5 | 360 tournaments OOF imputed | 439109 | 0.60 | 0.51 | 0.00 | 0.49 | 1.52 |
| n_opp_within_5 | 360 tournaments truth | 438908 | 0.59 | 0.73 | 0.00 | 0.00 | 2.00 |
| n_opp_within_5 | 360 club seasons OOF imputed | 224159 | 0.62 | 0.50 | 0.00 | 0.53 | 1.53 |
| n_opp_within_5 | 360 club seasons truth | 224061 | 0.62 | 0.76 | 0.00 | 0.00 | 2.00 |
| nearest_opp_dist | 360 all OOF imputed | 663268 | 6.79 | 4.25 | 1.67 | 5.82 | 14.32 |
| nearest_opp_dist | 360 all truth | 662042 | 6.78 | 5.29 | 1.07 | 5.36 | 16.65 |
| nearest_opp_dist | 360 tournaments OOF imputed | 439109 | 6.94 | 4.43 | 1.63 | 5.91 | 14.82 |
| nearest_opp_dist | 360 tournaments truth | 438011 | 6.93 | 5.46 | 1.04 | 5.44 | 17.18 |
| nearest_opp_dist | 360 club seasons OOF imputed | 224159 | 6.49 | 3.85 | 1.75 | 5.68 | 13.15 |
| nearest_opp_dist | 360 club seasons truth | 224031 | 6.48 | 4.93 | 1.15 | 5.23 | 15.47 |

**Shot**

| target | source | n | mean | sd | p05 | p50 | p95 |
|---|---|---|---|---|---|---|---|
| n_opp_in_cone | PL 2015/16 | 9726 | 0.68 | 0.63 | 0.10 | 0.48 | 2.21 |
| n_opp_in_cone | La Liga 2015/16 | 8957 | 0.68 | 0.67 | 0.10 | 0.46 | 2.34 |
| n_opp_in_cone | Serie A 2015/16 | 9756 | 0.71 | 0.67 | 0.10 | 0.49 | 2.37 |
| n_opp_in_cone | Ligue 1 2015/16 | 8595 | 0.67 | 0.65 | 0.10 | 0.46 | 2.31 |
| n_opp_in_cone | WC 2018 | 1625 | 0.67 | 0.64 | 0.09 | 0.48 | 2.25 |
| n_opp_in_cone | Copa 2024 | 731 | 0.69 | 0.66 | 0.11 | 0.48 | 2.35 |
| n_opp_in_cone | AFCON 2023 | 1150 | 0.71 | 0.67 | 0.10 | 0.49 | 2.35 |
| nearest_opp_dist_in_cone | PL 2015/16 | 9726 | 5.51 | 2.69 | 2.41 | 4.80 | 10.84 |
| nearest_opp_dist_in_cone | La Liga 2015/16 | 8957 | 5.52 | 2.85 | 2.43 | 4.62 | 11.19 |
| nearest_opp_dist_in_cone | Serie A 2015/16 | 9756 | 5.83 | 3.06 | 2.39 | 4.95 | 11.81 |
| nearest_opp_dist_in_cone | Ligue 1 2015/16 | 8595 | 5.60 | 2.98 | 2.38 | 4.65 | 11.69 |
| nearest_opp_dist_in_cone | WC 2018 | 1625 | 5.79 | 3.05 | 2.45 | 4.93 | 11.99 |
| nearest_opp_dist_in_cone | Copa 2024 | 731 | 5.52 | 2.86 | 2.46 | 4.60 | 11.32 |
| nearest_opp_dist_in_cone | AFCON 2023 | 1150 | 5.77 | 3.19 | 2.25 | 4.87 | 12.39 |
| opp_keeper_dist_to_goal_line | PL 2015/16 | 9720 | 2.85 | 0.93 | 1.77 | 2.67 | 4.50 |
| opp_keeper_dist_to_goal_line | La Liga 2015/16 | 8951 | 2.87 | 1.07 | 1.72 | 2.68 | 4.61 |
| opp_keeper_dist_to_goal_line | Serie A 2015/16 | 9747 | 2.91 | 1.04 | 1.73 | 2.72 | 4.65 |
| opp_keeper_dist_to_goal_line | Ligue 1 2015/16 | 8588 | 2.94 | 1.05 | 1.74 | 2.73 | 4.81 |
| opp_keeper_dist_to_goal_line | WC 2018 | 1624 | 2.72 | 0.86 | 1.66 | 2.57 | 4.26 |
| opp_keeper_dist_to_goal_line | Copa 2024 | 726 | 2.90 | 1.24 | 1.76 | 2.66 | 4.76 |
| opp_keeper_dist_to_goal_line | AFCON 2023 | 1150 | 2.84 | 1.03 | 1.62 | 2.60 | 4.62 |
| n_opp_within_5 | PL 2015/16 | 9817 | 1.78 | 0.86 | 0.42 | 1.71 | 3.42 |
| n_opp_within_5 | La Liga 2015/16 | 9071 | 1.74 | 0.88 | 0.30 | 1.65 | 3.43 |
| n_opp_within_5 | Serie A 2015/16 | 9877 | 1.74 | 0.87 | 0.21 | 1.68 | 3.33 |
| n_opp_within_5 | Ligue 1 2015/16 | 8723 | 1.78 | 0.91 | 0.16 | 1.70 | 3.51 |
| n_opp_within_5 | WC 2018 | 1638 | 1.74 | 0.86 | 0.25 | 1.67 | 3.45 |
| n_opp_within_5 | Copa 2024 | 741 | 1.83 | 0.93 | 0.45 | 1.73 | 3.57 |
| n_opp_within_5 | AFCON 2023 | 1162 | 1.75 | 0.91 | 0.07 | 1.71 | 3.43 |
| block_depth | PL 2015/16 | 9726 | 16.36 | 5.52 | 8.98 | 15.61 | 26.43 |
| block_depth | La Liga 2015/16 | 8957 | 16.26 | 5.66 | 9.06 | 15.36 | 26.17 |
| block_depth | Serie A 2015/16 | 9756 | 17.14 | 5.87 | 9.30 | 16.23 | 27.61 |
| block_depth | Ligue 1 2015/16 | 8595 | 16.85 | 6.02 | 8.97 | 15.81 | 27.89 |
| block_depth | WC 2018 | 1625 | 15.70 | 5.17 | 8.98 | 14.83 | 25.06 |
| block_depth | Copa 2024 | 731 | 16.47 | 7.34 | 8.63 | 15.27 | 26.75 |
| block_depth | AFCON 2023 | 1150 | 15.90 | 5.64 | 8.45 | 14.91 | 26.40 |
| n_opp_in_cone | 360 all OOF imputed | 10201 | 0.67 | 0.60 | 0.10 | 0.51 | 2.06 |
| n_opp_in_cone | 360 all truth | 10191 | 0.67 | 0.99 | 0.00 | 0.00 | 3.00 |
| n_opp_in_cone | 360 tournaments OOF imputed | 7079 | 0.69 | 0.59 | 0.10 | 0.52 | 2.00 |
| n_opp_in_cone | 360 tournaments truth | 7073 | 0.70 | 1.00 | 0.00 | 0.00 | 3.00 |
| n_opp_in_cone | 360 club seasons OOF imputed | 3122 | 0.64 | 0.61 | 0.09 | 0.47 | 2.18 |
| n_opp_in_cone | 360 club seasons truth | 3118 | 0.61 | 0.97 | 0.00 | 0.00 | 2.00 |
| nearest_opp_dist_in_cone | 360 all OOF imputed | 10201 | 5.26 | 2.60 | 2.41 | 4.50 | 10.48 |
| nearest_opp_dist_in_cone | 360 all truth | 4466 | 6.28 | 4.14 | 0.95 | 5.49 | 13.76 |
| nearest_opp_dist_in_cone | 360 tournaments OOF imputed | 7079 | 5.32 | 2.68 | 2.37 | 4.55 | 10.77 |
| nearest_opp_dist_in_cone | 360 tournaments truth | 3209 | 6.29 | 4.26 | 0.93 | 5.43 | 14.29 |
| nearest_opp_dist_in_cone | 360 club seasons OOF imputed | 3122 | 5.12 | 2.40 | 2.45 | 4.43 | 9.97 |
| nearest_opp_dist_in_cone | 360 club seasons truth | 1257 | 6.23 | 3.84 | 0.99 | 5.63 | 13.20 |
| opp_keeper_dist_to_goal_line | 360 all OOF imputed | 10195 | 2.79 | 0.93 | 1.62 | 2.64 | 4.42 |
| opp_keeper_dist_to_goal_line | 360 all truth | 9607 | 2.75 | 1.80 | 0.67 | 2.47 | 5.78 |
| opp_keeper_dist_to_goal_line | 360 tournaments OOF imputed | 7073 | 2.75 | 0.92 | 1.65 | 2.59 | 4.35 |
| opp_keeper_dist_to_goal_line | 360 tournaments truth | 6575 | 2.69 | 1.69 | 0.66 | 2.44 | 5.62 |
| opp_keeper_dist_to_goal_line | 360 club seasons OOF imputed | 3122 | 2.88 | 0.97 | 1.45 | 2.77 | 4.52 |
| opp_keeper_dist_to_goal_line | 360 club seasons truth | 3032 | 2.89 | 2.02 | 0.69 | 2.57 | 6.01 |
| n_opp_within_5 | 360 all OOF imputed | 10367 | 1.89 | 0.95 | 0.48 | 1.79 | 3.81 |
| n_opp_within_5 | 360 all truth | 10357 | 1.87 | 1.35 | 0.00 | 2.00 | 4.00 |
| n_opp_within_5 | 360 tournaments OOF imputed | 7188 | 1.93 | 0.97 | 0.52 | 1.81 | 3.92 |
| n_opp_within_5 | 360 tournaments truth | 7182 | 1.91 | 1.38 | 0.00 | 2.00 | 4.00 |
| n_opp_within_5 | 360 club seasons OOF imputed | 3179 | 1.81 | 0.90 | 0.30 | 1.75 | 3.52 |
| n_opp_within_5 | 360 club seasons truth | 3175 | 1.79 | 1.27 | 0.00 | 2.00 | 4.00 |
| block_depth | 360 all OOF imputed | 10201 | 15.45 | 5.13 | 8.31 | 14.89 | 24.27 |
| block_depth | 360 all truth | 9178 | 15.27 | 5.63 | 7.21 | 14.91 | 24.51 |
| block_depth | 360 tournaments OOF imputed | 7079 | 15.23 | 5.30 | 7.99 | 14.62 | 24.44 |
| block_depth | 360 tournaments truth | 6413 | 15.06 | 5.75 | 7.03 | 14.58 | 24.59 |
| block_depth | 360 club seasons OOF imputed | 3122 | 15.94 | 4.66 | 9.20 | 15.41 | 24.01 |
| block_depth | 360 club seasons truth | 2765 | 15.76 | 5.29 | 8.04 | 15.53 | 24.41 |

## (e) Worked example: imputed block depth over time in one 2015/16 match

Match 3754145: West Bromwich Albion vs Manchester City (Premier League 2015/2016); 538 imputed possession rows, 539 defensive actions. For every team: the imputed block depth of that team (E2 student, from the opponent's possession Pass / Carry rows, yards from the team's own goal line) next to the mean x of the team's own defensive actions in the same minutes (also yards from its own goal line). 5-minute bins (row-weighted); the per-minute table is in `soccer_04_example_minutes.parquet`.

| minutes | West Bromwich Albion: imputed block depth | West Bromwich Albion: n rows | West Bromwich Albion: def-action x | West Bromwich Albion: n actions | Manchester City: imputed block depth | Manchester City: n rows | Manchester City: def-action x | Manchester City: n actions |
|---|---|---|---|---|---|---|---|---|
| 0-5 | 38.9 | 22 | 35.0 | 26 | 71.3 | 5 | 58.3 | 16 |
| 5-10 | 38.6 | 44 | 33.0 | 15 |  | 0 | 91.1 | 5 |
| 10-15 | 33.8 | 24 | 32.7 | 17 | 66.0 | 9 | 54.8 | 14 |
| 15-20 | 43.4 | 38 | 32.7 | 24 | 57.8 | 9 | 75.8 | 11 |
| 20-25 | 43.6 | 25 | 40.8 | 12 | 73.1 | 3 | 46.2 | 7 |
| 25-30 | 48.0 | 20 | 39.8 | 9 | 55.2 | 16 | 64.3 | 20 |
| 30-35 | 46.0 | 25 | 48.3 | 19 | 52.9 | 2 | 50.6 | 7 |
| 35-40 | 47.6 | 29 | 37.5 | 32 | 55.8 | 9 | 81.4 | 9 |
| 40-45 | 43.9 | 14 | 40.3 | 18 | 46.1 | 10 | 55.9 | 14 |
| 45-50 | 62.9 | 7 | 55.0 | 16 | 39.2 | 16 | 45.1 | 24 |
| 50-55 | 41.9 | 26 | 32.3 | 11 | 53.8 | 12 | 37.0 | 12 |
| 55-60 | 14.4 | 2 | 34.9 | 14 | 46.9 | 7 | 57.5 | 16 |
| 60-65 | 45.0 | 14 | 54.4 | 21 | 63.5 | 5 | 42.7 | 14 |
| 65-70 | 61.2 | 6 | 44.2 | 3 | 54.1 | 19 | 45.4 | 22 |
| 70-75 | 50.7 | 4 | 76.2 | 7 | 24.3 | 10 | 34.0 | 11 |
| 75-80 | 56.7 | 19 | 57.8 | 17 | 29.9 | 8 | 49.6 | 13 |
| 80-85 | 33.7 | 18 | 46.4 | 15 | 29.8 | 6 | 47.3 | 9 |
| 85-90 | 43.2 | 32 | 26.5 | 24 | 80.4 | 3 | 72.3 | 7 |
| 90-95 | 39.5 | 20 | 34.2 | 7 |  | 0 | 98.5 | 1 |

Correlation between the two series (over minutes / bins where both exist):

| team_id | team | n_minutes | pearson | spearman | mean_imputed | mean_observed | level |
|---|---|---|---|---|---|---|---|
| 27 | West Bromwich Albion | 67 | 0.614 | 0.706 | 46.005 | 39.870 | per minute |
| 36 | Manchester City | 48 | 0.757 | 0.729 | 50.748 | 51.150 | per minute |
| 27 | West Bromwich Albion | 19 | 0.491 | 0.647 | 43.836 | 42.213 | 5-minute bins |
| 36 | Manchester City | 17 | 0.424 | 0.424 | 52.957 | 54.014 | 5-minute bins |

## Fits

| population | variant | rounds_median | seconds | fits |
|---|---|---|---|---|
| La Liga 2015/16 | EVENT | 140.0 | 7.3 | 10 |
| La Liga 2015/16 | EVENT+IMP | 134.0 | 7.4 | 10 |
| La Liga 2015/16 | EVENT+ORACLESHOT | 139.5 | 6.5 | 10 |
| Ligue 1 2015/16 | EVENT | 131.0 | 6.0 | 10 |
| Ligue 1 2015/16 | EVENT+IMP | 123.0 | 6.5 | 10 |
| Ligue 1 2015/16 | EVENT+ORACLESHOT | 143.0 | 6.7 | 10 |
| PL 2015/16 | EVENT | 113.0 | 5.7 | 10 |
| PL 2015/16 | EVENT+IMP | 128.5 | 7.8 | 10 |
| PL 2015/16 | EVENT+ORACLESHOT | 143.5 | 7.7 | 10 |
| Serie A 2015/16 | EVENT | 156.0 | 6.2 | 10 |
| Serie A 2015/16 | EVENT+IMP | 133.0 | 6.8 | 10 |
| Serie A 2015/16 | EVENT+ORACLESHOT | 122.0 | 6.5 | 10 |
| leagues_2015/16 | EVENT | 201.0 | 29.7 | 15 |
| leagues_2015/16 | EVENT+IMP | 219.0 | 36.2 | 15 |
| leagues_2015/16 | EVENT+IMP(E0) | 219.0 | 35.6 | 15 |
| leagues_2015/16 | EVENT+ORACLESHOT | 209.0 | 34.6 | 15 |
| leagues_2015/16 | EVENT+ORACLESHOT_full | 214.0 | 38.0 | 15 |
| leagues_2015/16 | LOC | 107.0 | 8.2 | 15 |
| tournaments_no360 | EVENT | 90.0 | 5.0 | 15 |
| tournaments_no360 | EVENT+IMP | 90.0 | 5.6 | 15 |
| tournaments_no360 | EVENT+ORACLESHOT | 91.0 | 4.8 | 15 |

| variant | design | rounds_median | seconds | fits |
|---|---|---|---|---|
| EVENT | after | 455.0 | 31.2 | 5 |
| EVENT | noafter | 397.0 | 22.7 | 5 |
| EVENT+IMP | after | 399.0 | 27.9 | 5 |
| EVENT+IMP | noafter | 360.0 | 24.5 | 5 |
| EVENT-nodur | after | 460.0 | 29.8 | 5 |

Stage wall time (s): report 74

## Caveats

- The shot-state students were weak in-domain (OOF R2 vs the 360 frame 0.16 / 0.41 / 0.48 for cone count / cone distance / keeper distance, stage 02), so a small transfer loss on top does not change the downstream picture; the interesting oracle (the shot freeze frame) exists in 2015/16 and is not imputed by anyone here.
- The 2015/16 event data is older StatsBomb collection (2015/16 was coded retrospectively); collection conventions (pressure events, freeze-frame completeness) may differ from 2020-25. The domain shift therefore mixes era, competition type (club vs tournament; only a handful of single-club seasons were in the 360 training set) and collection.
- The key-pass rebuild recomputes the build-stage features with the same code and settings; 9 of the non-360 matches are 360-flagged fall-backs whose passes are rebuilt through the same fall-back path. Shots whose key pass is missing from the raw events keep `a_has_assist = 0`.
- The team-level check is partly circular: `f_opp_def_x_60s_mean` (the opponent's recent defensive-action x) is an E2 input, and `def_line_x` in `team_match.parquet` is the match mean of the same quantity. The E0 student (no window features) and the raw mean ball x are shown as comparators. Imputed block depth is a visible-area-truncated 360 quantity (stage-02 caveat), not tracking.
- The worked example uses a 25% pass / carry sample per match, so most minutes have 0-3 imputed rows; read the 5-minute bins. The observed series (mean x of the team's defensive actions) is where the team acted, not where its block stood, and is one of the student's own inputs.
- Zero-shot xG models were fitted on shots from both genders and mostly tournaments; their `f_gender` / `f_comp_type` inputs take constant values here. The within-2015/16 CV models have ~4x more shots than the 360 xG models had, so a lower within-CV log-loss is expected even without any domain shift.
- Stage-02 open issues respected: imputed counts / distances clipped at 0 wherever they are used or scored here; the transfer BSS reference issue does not apply (no binary targets used); the NFL highlight check is not used.
- Single machine, 2 threads; LightGBM is seeded but not deterministic across machines.
