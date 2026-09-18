# Soccer 06 - payoff of the sequence student (seq20): xG and xPass on the stage-05 folds

Machine-written by `python -m research.privileged_tracking.soccer.payoff_seq`. Question:
stage 03 found no xG payoff and a small xPass payoff for imputed defensive state, using
the stage-02 LightGBM E2 students. Stage 05 showed that the `seq20` GRU over the raw
20-event history imputes seven state quantities better than LightGBM on 7/7 targets.
Does the better imputation change the payoff? Same protocol as stage 03, restricted to
the 3 stage-02 folds (0, 1, 2; 251 matches) on which `seq20`
out-of-fold predictions exist.

## Headline

- xG (6,202 non-penalty shots, 621 goals; 3 seeds; deltas =
  log-loss(EVENT) - log-loss(variant), positive = better; per-shot paired bootstrap
  95% CI, match-clustered CI in parentheses):
  EVENT+IMP (stage-03 LightGBM block) -0.0005 [-0.0021, +0.0012] (clustered [-0.0021, +0.0013]);
  EVENT+SEQ7(lgbm) +0.0008 [-0.0006, +0.0023] (clustered [-0.0006, +0.0023]);
  **EVENT+SEQ7(seq20) +0.0000 [-0.0017, +0.0017] (clustered [-0.0016, +0.0016])**;
  EVENT+IMP+SEQ7(seq20) +0.0000 [-0.0018, +0.0019] (clustered [-0.0018, +0.0019]);
  EVENT+ORACLE360 +0.0017 [-0.0005, +0.0039] (clustered [-0.0006, +0.0039]); EVENT+ORACLESHOT +0.0061 [+0.0032, +0.0090] (clustered [+0.0031, +0.0090]).
- Direct pair, same seven inputs: EVENT+SEQ7(lgbm) -> EVENT+SEQ7(seq20) -0.0008 [-0.0026, +0.0010] (clustered [-0.0025, +0.0010]); stage-03 block -> plus the
  seven `seq20` columns +0.0005 [-0.0007, +0.0018] (clustered [-0.0008, +0.0017]).
- xPass (89,285 pass attempts of the stage-03 match-stratified subsample, single
  seed), pre-instant design:
  EVENT+IMP +0.0020 [+0.0012, +0.0028] (clustered [+0.0011, +0.0029]);
  EVENT+SEQ7(lgbm) +0.0013 [+0.0006, +0.0019] (clustered [+0.0006, +0.0019]);
  **EVENT+SEQ7(seq20) +0.0026 [+0.0019, +0.0034] (clustered [+0.0018, +0.0033])**;
  EVENT+IMP+SEQ7(seq20) +0.0030 [+0.0022, +0.0039] (clustered [+0.0021, +0.0039]);
  EVENT+ORACLE360 +0.0683 [+0.0659, +0.0708] (clustered [+0.0656, +0.0711]). Direct pair lgbm -> seq20:
  +0.0013 [+0.0006, +0.0020] (clustered [+0.0006, +0.0020]).
- xPass, destination-known design: EVENT+IMP +0.0008 [+0.0000, +0.0017] (clustered [-0.0000, +0.0016]);
  EVENT+SEQ7(seq20) +0.0002 [-0.0004, +0.0009] (clustered [-0.0005, +0.0009]);
  EVENT+IMP+SEQ7(seq20) +0.0012 [+0.0003, +0.0021] (clustered [+0.0003, +0.0021]); direct pair lgbm -> seq20:
  -0.0000 [-0.0006, +0.0006] (clustered [-0.0006, +0.0006]).

## Protocol

* **Rows.** The stage-03 shot table (`payoff_cache/payoff_shots.parquet`, penalties
  excluded) and pass subsample restricted to folds [0, 1, 2] of the stage-02
  5-fold match split (`group_kfold` by `match_id`, seed 0): 6,202 shots and
  89,285 pass attempts in 251 matches.
* **Imputed columns.** `imp_<t>__E2` = stage-02 LightGBM out-of-fold predictions;
  `imp_<t>__seq20` = stage-05 `seq20` out-of-fold predictions
  (`processed/soccer/imputed_oof_seq.parquet`; for fold k the network was trained on the
  other four folds with an inner match holdout). Seven targets: block_depth, def_line, n_opp_ahead_of_ball, nearest_opp_dist, n_opp_in_cone, deep_block, counter_on.
  Counts / distances are clipped at 0 (`payoff_features.clip_state`); NaN where the
  stage-02 subset rule leaves the quantity undefined (e.g. cone counts outside possession
  events), identically for both sources.
* **Variants.** `EVENT` = EVENT + nothing; `EVENT+IMP` = EVENT + imp_shot, imp_assist; `EVENT+SEQ7(lgbm)` = EVENT + seq7_lgbm; `EVENT+SEQ7(seq20)` = EVENT + seq7_seq20; `EVENT+IMP+SEQ7(seq20)` = EVENT + imp_shot, imp_assist, seq7_seq20; `EVENT+ORACLE360` = EVENT + oracle_shot, oracle_assist; `EVENT+ORACLESHOT` = EVENT + oracleshot_shot, oracle_assist. `imp_shot` / `imp_assist` / `imp_pass` are the stage-03
  blocks (SHOT_STATE E2, ASSIST_STATE E2a, PASS_STATE E2 or E2a); `seq7_lgbm` /
  `seq7_seq20` are the seven stage-05 targets from LightGBM E2 or from the sequence
  student; `oracle_*` the 360-frame values; `oracleshot_shot` the shot-freeze-frame
  values.
* **Models.** Stage-03 LightGBM settings (`PayoffConfig.xg` / `.xpass`), cross-validated
  over the 3 folds (train on the other two, early stopping on an inner
  match holdout, xG refit on all training rows), so every test row's imputations come
  from students that never saw its match. Training rows' imputations were produced by
  students that saw the test fold's 360 labels (the stage-03 second-order stacking
  coupling, unchanged).
* **Statistics.** Paired bootstrap (2,000 resamples, seed 0) of per-shot /
  per-pass log-loss, plus a match-clustered bootstrap; deltas are
  `loss(reference) - loss(variant)`, positive = the variant is better. `significant
  (clustered)` uses the clustered interval.

## Imputation quality on exactly these rows

Skill of each source against the 360 label on the shot rows and on the pass rows
(identical rows per target):

| population | target | source | n | r2 | mae | log_loss | auc |
|---|---|---|---|---|---|---|---|
| shots | block_depth | E2 | 5476 | 0.788 | 1.984 |  |  |
| shots | block_depth | seq20 | 5476 | 0.810 | 1.915 |  |  |
| shots | def_line | E2 | 5476 | 0.795 | 1.839 |  |  |
| shots | def_line | seq20 | 5476 | 0.809 | 1.763 |  |  |
| shots | n_opp_ahead_of_ball | E2 | 5476 | 0.747 | 1.146 |  |  |
| shots | n_opp_ahead_of_ball | seq20 | 5476 | 0.786 | 1.031 |  |  |
| shots | nearest_opp_dist | E2 | 6198 | 0.560 | 1.037 |  |  |
| shots | nearest_opp_dist | seq20 | 6198 | 0.547 | 1.053 |  |  |
| shots | n_opp_in_cone | E2 | 6092 | 0.428 | 0.564 |  |  |
| shots | n_opp_in_cone | seq20 | 6092 | 0.434 | 0.532 |  |  |
| shots | deep_block | E2 | 1543 |  |  | 0.060 | 0.984 |
| shots | deep_block | seq20 | 1543 |  |  | 0.057 | 0.982 |
| shots | counter_on | E2 | 14 |  |  |  |  |
| shots | counter_on | seq20 | 14 |  |  |  |  |
| passes | block_depth | E2 | 65292 | 0.969 | 2.757 |  |  |
| passes | block_depth | seq20 | 65292 | 0.973 | 2.558 |  |  |
| passes | def_line | E2 | 65292 | 0.954 | 2.878 |  |  |
| passes | def_line | seq20 | 65292 | 0.959 | 2.706 |  |  |
| passes | n_opp_ahead_of_ball | E2 | 65292 | 0.666 | 1.087 |  |  |
| passes | n_opp_ahead_of_ball | seq20 | 65292 | 0.697 | 1.029 |  |  |
| passes | nearest_opp_dist | E2 | 89108 | 0.608 | 2.116 |  |  |
| passes | nearest_opp_dist | seq20 | 89108 | 0.618 | 2.114 |  |  |
| passes | n_opp_in_cone | E2 | 84553 | 0.121 | 0.420 |  |  |
| passes | n_opp_in_cone | seq20 | 84553 | 0.128 | 0.415 |  |  |
| passes | deep_block | E2 | 17979 |  |  | 0.129 | 0.985 |
| passes | deep_block | seq20 | 17979 |  |  | 0.125 | 0.986 |
| passes | counter_on | E2 | 36727 |  |  | 0.094 | 0.939 |
| passes | counter_on | seq20 | 36727 |  |  | 0.086 | 0.954 |

Coverage (share of rows with a prediction; both sources follow the stage-02 subset
rules):

| population | target | n_rows | share_lgbm_E2 | share_seq20 | share_both |
|---|---|---|---|---|---|
| shots | block_depth | 6202 | 0.983 | 0.983 | 0.983 |
| shots | def_line | 6202 | 0.983 | 0.983 | 0.983 |
| shots | n_opp_ahead_of_ball | 6202 | 0.983 | 0.983 | 0.983 |
| shots | nearest_opp_dist | 6202 | 1.000 | 1.000 | 1.000 |
| shots | n_opp_in_cone | 6202 | 0.983 | 0.983 | 0.983 |
| shots | deep_block | 6202 | 0.281 | 0.281 | 0.281 |
| shots | counter_on | 6202 | 0.002 | 0.002 | 0.002 |
| passes | block_depth | 89285 | 0.947 | 0.947 | 0.947 |
| passes | def_line | 89285 | 0.947 | 0.947 | 0.947 |
| passes | n_opp_ahead_of_ball | 89285 | 0.947 | 0.947 | 0.947 |
| passes | nearest_opp_dist | 89285 | 1.000 | 1.000 | 1.000 |
| passes | n_opp_in_cone | 89285 | 0.947 | 0.947 | 0.947 |
| passes | deep_block | 89285 | 0.249 | 0.249 | 0.249 |
| passes | counter_on | 89285 | 0.486 | 0.486 | 0.486 |

## xG

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 6202 | 621 | 0.3258 | 0.0902 | 0.4775 | 0.0161 | 0.1002 |
| EVENT | 6202 | 621 | 0.2713 | 0.0771 | 0.7849 | 0.0095 | 0.0969 |
| EVENT+IMP | 6202 | 621 | 0.2717 | 0.0771 | 0.7834 | 0.0115 | 0.0964 |
| EVENT+SEQ7(lgbm) | 6202 | 621 | 0.2704 | 0.0770 | 0.7878 | 0.0090 | 0.0960 |
| EVENT+SEQ7(seq20) | 6202 | 621 | 0.2712 | 0.0770 | 0.7858 | 0.0121 | 0.0973 |
| EVENT+IMP+SEQ7(seq20) | 6202 | 621 | 0.2712 | 0.0771 | 0.7850 | 0.0097 | 0.0966 |
| EVENT+ORACLE360 | 6202 | 621 | 0.2695 | 0.0762 | 0.7841 | 0.0088 | 0.0966 |
| EVENT+ORACLESHOT | 6202 | 621 | 0.2652 | 0.0753 | 0.7969 | 0.0092 | 0.0971 |
| STATSBOMB_XG | 6202 | 621 | 0.2556 | 0.0723 | 0.8138 | 0.0077 | 0.0997 |

Deltas vs EVENT:

| variant | reference | n | delta_log_loss [per-sample CI] (match-clustered CI) | significant (clustered) |
|---|---|---|---|---|
| BASE | EVENT | 6202 | -0.0545 [-0.0627, -0.0469] (clustered [-0.0620, -0.0476]) | True |
| EVENT+IMP | EVENT | 6202 | -0.0005 [-0.0021, +0.0012] (clustered [-0.0021, +0.0013]) | False |
| EVENT+SEQ7(lgbm) | EVENT | 6202 | +0.0008 [-0.0006, +0.0023] (clustered [-0.0006, +0.0023]) | False |
| EVENT+SEQ7(seq20) | EVENT | 6202 | +0.0000 [-0.0017, +0.0017] (clustered [-0.0016, +0.0016]) | False |
| EVENT+IMP+SEQ7(seq20) | EVENT | 6202 | +0.0000 [-0.0018, +0.0019] (clustered [-0.0018, +0.0019]) | False |
| EVENT+ORACLE360 | EVENT | 6202 | +0.0017 [-0.0005, +0.0039] (clustered [-0.0006, +0.0039]) | False |
| EVENT+ORACLESHOT | EVENT | 6202 | +0.0061 [+0.0032, +0.0090] (clustered [+0.0031, +0.0090]) | True |
| STATSBOMB_XG | EVENT | 6202 | +0.0157 [+0.0108, +0.0208] (clustered [+0.0105, +0.0208]) | True |

Direct pairs:

| from | to | n | delta_log_loss [per-sample CI] (match-clustered CI) | significant (clustered) |
|---|---|---|---|---|
| EVENT+SEQ7(lgbm) | EVENT+SEQ7(seq20) | 6202 | -0.0008 [-0.0026, +0.0010] (clustered [-0.0025, +0.0010]) | False |
| EVENT+IMP | EVENT+IMP+SEQ7(seq20) | 6202 | +0.0005 [-0.0007, +0.0018] (clustered [-0.0008, +0.0017]) | False |
| EVENT+SEQ7(seq20) | EVENT+ORACLE360 | 6202 | +0.0017 [-0.0006, +0.0040] (clustered [-0.0006, +0.0040]) | False |

## xPass

| design | variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|---|
| after | BASE | 89285 | 73953 | 0.4587 | 0.1423 | 0.4933 | 0.0155 | 0.8283 |
| after | EVENT | 89285 | 73953 | 0.2122 | 0.0651 | 0.9471 | 0.0056 | 0.8295 |
| after | EVENT+IMP | 89285 | 73953 | 0.2114 | 0.0650 | 0.9474 | 0.0045 | 0.8294 |
| after | EVENT+SEQ7(lgbm) | 89285 | 73953 | 0.2119 | 0.0650 | 0.9472 | 0.0049 | 0.8294 |
| after | EVENT+SEQ7(seq20) | 89285 | 73953 | 0.2120 | 0.0651 | 0.9472 | 0.0052 | 0.8292 |
| after | EVENT+IMP+SEQ7(seq20) | 89285 | 73953 | 0.2110 | 0.0649 | 0.9477 | 0.0041 | 0.8290 |
| after | EVENT+ORACLE360 | 89285 | 73953 | 0.2044 | 0.0632 | 0.9517 | 0.0051 | 0.8296 |
| noafter | BASE | 89285 | 73953 | 0.4587 | 0.1423 | 0.4933 | 0.0155 | 0.8283 |
| noafter | EVENT | 89285 | 73953 | 0.3490 | 0.1062 | 0.8195 | 0.0032 | 0.8282 |
| noafter | EVENT+IMP | 89285 | 73953 | 0.3470 | 0.1057 | 0.8229 | 0.0046 | 0.8281 |
| noafter | EVENT+SEQ7(lgbm) | 89285 | 73953 | 0.3478 | 0.1058 | 0.8213 | 0.0034 | 0.8285 |
| noafter | EVENT+SEQ7(seq20) | 89285 | 73953 | 0.3464 | 0.1056 | 0.8241 | 0.0028 | 0.8280 |
| noafter | EVENT+IMP+SEQ7(seq20) | 89285 | 73953 | 0.3460 | 0.1054 | 0.8245 | 0.0036 | 0.8285 |
| noafter | EVENT+ORACLE360 | 89285 | 73953 | 0.2807 | 0.0863 | 0.9007 | 0.0056 | 0.8290 |

Deltas vs EVENT (per design):

| design | variant | reference | n | delta_log_loss [per-sample CI] (match-clustered CI) | significant (clustered) |
|---|---|---|---|---|---|
| after | BASE | EVENT | 89285 | -0.2465 [-0.2502, -0.2427] (clustered [-0.2526, -0.2408]) | True |
| after | EVENT+IMP | EVENT | 89285 | +0.0008 [+0.0000, +0.0017] (clustered [-0.0000, +0.0016]) | False |
| after | EVENT+SEQ7(lgbm) | EVENT | 89285 | +0.0002 [-0.0004, +0.0008] (clustered [-0.0004, +0.0009]) | False |
| after | EVENT+SEQ7(seq20) | EVENT | 89285 | +0.0002 [-0.0004, +0.0009] (clustered [-0.0005, +0.0009]) | False |
| after | EVENT+IMP+SEQ7(seq20) | EVENT | 89285 | +0.0012 [+0.0003, +0.0021] (clustered [+0.0003, +0.0021]) | True |
| after | EVENT+ORACLE360 | EVENT | 89285 | +0.0078 [+0.0066, +0.0088] (clustered [+0.0067, +0.0088]) | True |
| noafter | BASE | EVENT | 89285 | -0.1096 [-0.1125, -0.1065] (clustered [-0.1133, -0.1060]) | True |
| noafter | EVENT+IMP | EVENT | 89285 | +0.0020 [+0.0012, +0.0028] (clustered [+0.0011, +0.0029]) | True |
| noafter | EVENT+SEQ7(lgbm) | EVENT | 89285 | +0.0013 [+0.0006, +0.0019] (clustered [+0.0006, +0.0019]) | True |
| noafter | EVENT+SEQ7(seq20) | EVENT | 89285 | +0.0026 [+0.0019, +0.0034] (clustered [+0.0018, +0.0033]) | True |
| noafter | EVENT+IMP+SEQ7(seq20) | EVENT | 89285 | +0.0030 [+0.0022, +0.0039] (clustered [+0.0021, +0.0039]) | True |
| noafter | EVENT+ORACLE360 | EVENT | 89285 | +0.0683 [+0.0659, +0.0708] (clustered [+0.0656, +0.0711]) | True |

Direct pairs:

| design | from | to | n | delta_log_loss [per-sample CI] (match-clustered CI) | significant (clustered) |
|---|---|---|---|---|---|
| after | EVENT+SEQ7(lgbm) | EVENT+SEQ7(seq20) | 89285 | -0.0000 [-0.0006, +0.0006] (clustered [-0.0006, +0.0006]) | False |
| after | EVENT+IMP | EVENT+IMP+SEQ7(seq20) | 89285 | +0.0004 [-0.0001, +0.0009] (clustered [-0.0001, +0.0009]) | False |
| after | EVENT+SEQ7(seq20) | EVENT+ORACLE360 | 89285 | +0.0075 [+0.0064, +0.0086] (clustered [+0.0065, +0.0086]) | True |
| noafter | EVENT+SEQ7(lgbm) | EVENT+SEQ7(seq20) | 89285 | +0.0013 [+0.0006, +0.0020] (clustered [+0.0006, +0.0020]) | True |
| noafter | EVENT+IMP | EVENT+IMP+SEQ7(seq20) | 89285 | +0.0010 [+0.0004, +0.0016] (clustered [+0.0005, +0.0015]) | True |
| noafter | EVENT+SEQ7(seq20) | EVENT+ORACLE360 | 89285 | +0.0657 [+0.0634, +0.0682] (clustered [+0.0630, +0.0686]) | True |

## Fits

| task | variant | fits | rounds | seconds | d |
|---|---|---|---|---|---|
| xg | EVENT | 9 | 101.9 | 3.4 | 75 |
| xg | EVENT+IMP | 9 | 93.7 | 3.7 | 82 |
| xg | EVENT+IMP+SEQ7(seq20) | 9 | 96.4 | 5.0 | 89 |
| xg | EVENT+ORACLE360 | 9 | 101.0 | 4.5 | 82 |
| xg | EVENT+ORACLESHOT | 9 | 99.6 | 4.4 | 82 |
| xg | EVENT+SEQ7(lgbm) | 9 | 116.8 | 4.1 | 82 |
| xg | EVENT+SEQ7(seq20) | 9 | 83.9 | 4.1 | 82 |
| xpass | EVENT | 6 | 325.2 | 16.9 | 69 |
| xpass | EVENT+IMP | 6 | 249.5 | 14.3 | 76 |
| xpass | EVENT+IMP+SEQ7(seq20) | 6 | 233.8 | 15.2 | 83 |
| xpass | EVENT+ORACLE360 | 6 | 289.7 | 15.8 | 76 |
| xpass | EVENT+SEQ7(lgbm) | 6 | 292.5 | 15.9 | 76 |
| xpass | EVENT+SEQ7(seq20) | 6 | 272.2 | 15.0 | 76 |

## Caveats

* Only the seven stage-05 targets have `seq20` predictions; the stage-03 shot-state block
  also holds `nearest_opp_dist_in_cone`, `opp_keeper_dist_to_goal_line` and
  `n_opp_within_5`, and the pass-state block `n_opp_in_lane`, `n_opp_within_3_of_end` and
  `nearest_opp_to_receiver`, which stay LightGBM-imputed in the `EVENT+IMP+SEQ7` variant.
  `EVENT+SEQ7(lgbm)` vs `EVENT+SEQ7(seq20)` is the like-for-like comparison.
* Three folds instead of five: the xG models train on ~4,134 shots per
  fold instead of ~8k, so the absolute log-losses are slightly worse than stage 03 and
  the intervals wider; the stage-03 numbers are not comparable row for row.
* The sequence student's binary heads are less calibrated than LightGBM's (stage 05); the
  downstream LightGBM can re-calibrate them, so this is not a handicap for the payoff
  test.
* xPass uses one seed and the stage-03 40% match-stratified pass subsample; xG averages
  3 seeds.

## Timing

report: 72 s
