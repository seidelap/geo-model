# Soccer 05 - sequence student: neural model over the 20-event history vs LightGBM E2

Source: `processed_dir('soccer')/events360.parquet` + stage-02 `imputed_oof.parquet` (1,357,506 rows, 417 matches). This stage scores folds [0, 1, 2] of the stage-02 5-fold match-grouped split (251 held-out matches, 812,427 rows); every model and baseline below is scored on exactly the same held-out rows (label valid and every compared model has a prediction), so the LightGBM numbers differ slightly from the 5-fold figures of `soccer_02_imputation.md`.

## Protocol

- Folds: the `fold` column of `imputed_oof.parquet` (`group_kfold` by `match_id`, seed 0, verified against a recomputation). Labels: the stage-02 `y_<target>` columns (same subsets and label rules: team-shape targets on possession-team reliable frames, `deep_block` on settled possession, `counter_on` in the middle third, ball-relative targets on every usable frame, `n_opp_in_cone` on possession-team events).
- Network (GRU, PyTorch CPU, `torch.set_num_threads(2)`): sequence branch = type-id embedding + linear projection of 8 per-slot numerics (x/120, y/80, log1p(seconds before the current event), own / opponent / pad flags, displacement to the current event) + learned slot position, d_model 64, tokens oldest -> newest; output = newest hidden state and masked mean. Current-event branch = MLP [256, 128] over the E2 design (numeric z-scores + missing indicators, 8-dim embeddings per categorical column; training-fold statistics). Shared head 128 -> 7 outputs: MSE on standardised continuous / count targets, BCE for the two binaries, masked per row, equal weights.
- Optimisation: AdamW lr 0.002, weight decay 0.01, batch 1024, grad-clip 1, LR halved on validation plateau, early stopping (patience 3, max 15 epochs, time cap 10 min per fold) on an inner 15% match holdout of the training folds; training rows thinned uniformly to 300,000 (validation 60,000); seed 0 + fold. Count / distance predictions are clipped at 0.
- References: `lgbm_E2` = stage-02 LightGBM E2 out-of-fold predictions (one model per target, 300k training rows each, as stored, unclipped); `base_type` = stage-02 per-event-type training mean. Skill = R2 (continuous / count) or Brier skill score vs the training base rate (binary). Deltas: paired per-row bootstrap on squared error / log-loss (1000 resamples, 95% CI, positive = the `to` model is better) with a match-clustered bootstrap CI next to it (rows within a match are correlated; the clustered interval is the conservative one).

### Variants

| model | description |
|---|---|
| seq20 | GRU over the 20-slot event history + E2 current-event MLP (multi-task) |
| seq1 | same network, history truncated to the previous event only (window length 1) |
| seq0 | no sequence branch: the E2 current-event MLP alone (neural analogue of LightGBM E2) |
| lgbm_E2 | stage-02 LightGBM E2 student (out-of-fold, one model per target) |
| base_type | stage-02 per-event-type training mean / base rate |

## Summary

- `seq20` beats `lgbm_E2` on 7/7 targets with the match-clustered 95% CI excluding zero (worse on: none). Skill lgbm_E2 -> seq20: block_depth 0.964 -> 0.973; def_line 0.952 -> 0.961; n_opp_ahead_of_ball 0.671 -> 0.728; nearest_opp_dist 0.666 -> 0.683; n_opp_in_cone 0.160 -> 0.175; deep_block 0.774 -> 0.792; counter_on 0.343 -> 0.410.
- `seq0` (the same E2 inputs through the MLP, no sequence branch) is worse than `lgbm_E2` on 6/7 targets and better on 0/7: the neural architecture alone does not beat the trees; the gain of `seq20` comes from the raw event history.
- `seq20` beats `seq1` on 7/7 targets (clustered CI), so slots 2-20 carry information beyond the previous event and the E2 window counts; `seq1` alone beats `lgbm_E2` on 5/7 and loses on 2/7 (`nearest_opp_dist`, `deep_block`).

## Headline: skill per target (held-out folds, identical rows)

Skill = R2 or BSS. `n` is the number of scored rows per target.

| target | kind | n | seq20 | seq1 | seq0 | lgbm_E2 | base_type |
|---|---|---|---|---|---|---|---|
| block_depth | reg | 514853 | 0.973 | 0.966 | 0.960 | 0.964 | 0.032 |
| def_line | reg | 514853 | 0.961 | 0.953 | 0.948 | 0.952 | 0.030 |
| n_opp_ahead_of_ball | count | 514853 | 0.728 | 0.687 | 0.659 | 0.671 | 0.054 |
| nearest_opp_dist | reg | 799585 | 0.683 | 0.665 | 0.652 | 0.666 | 0.163 |
| n_opp_in_cone | count | 666441 | 0.175 | 0.167 | 0.158 | 0.160 | 0.009 |
| deep_block | binary | 143078 | 0.792 | 0.770 | 0.757 | 0.774 | 0.053 |
| counter_on | binary | 277676 | 0.410 | 0.360 | 0.327 | 0.343 | 0.036 |

### Continuous / count targets: MAE (yd or players)

| target | seq20 | seq1 | seq0 | lgbm_E2 | base_type |
|---|---|---|---|---|---|
| block_depth | 2.609 | 2.952 | 3.208 | 3.005 | 16.545 |
| def_line | 2.710 | 2.961 | 3.169 | 3.024 | 14.509 |
| n_opp_ahead_of_ball | 1.057 | 1.136 | 1.183 | 1.162 | 2.034 |
| nearest_opp_dist | 2.038 | 2.083 | 2.128 | 2.072 | 3.402 |
| n_opp_in_cone | 0.379 | 0.387 | 0.384 | 0.389 | 0.446 |

### Binary targets: AUC / log-loss


auc:

| target | seq20 | seq1 | seq0 | lgbm_E2 | base_type |
|---|---|---|---|---|---|
| deep_block | 0.9875 | 0.9847 | 0.9825 | 0.9847 | 0.5594 |
| counter_on | 0.9498 | 0.9363 | 0.9263 | 0.9278 | 0.6457 |

log_loss:

| target | seq20 | seq1 | seq0 | lgbm_E2 | base_type |
|---|---|---|---|---|---|
| deep_block | 0.1250 | 0.1376 | 0.1460 | 0.1349 | 0.5316 |
| counter_on | 0.1446 | 0.1583 | 0.1682 | 0.1649 | 0.2696 |

ece:

| target | seq20 | seq1 | seq0 | lgbm_E2 | base_type |
|---|---|---|---|---|---|
| deep_block | 0.0094 | 0.0102 | 0.0089 | 0.0029 | 0.0108 |
| counter_on | 0.0083 | 0.0065 | 0.0090 | 0.0045 | 0.0051 |

## Paired deltas (loss(from) - loss(to); positive = `to` better)

| target | from | to | n | loss | delta_loss | ci_low | ci_high | significant | ci_low_clustered | ci_high_clustered | significant_clustered | skill_from | skill_to |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block_depth | lgbm_E2 | seq20 | 514853 | squared_error | 3.95872 | 3.89148 | 4.02350 | True | 3.74901 | 4.16954 | True | 0.96416 | 0.97338 |
| block_depth | lgbm_E2 | seq1 | 514853 | squared_error | 0.58996 | 0.53583 | 0.64582 | True | 0.46673 | 0.71523 | True | 0.96416 | 0.96553 |
| block_depth | lgbm_E2 | seq0 | 514853 | squared_error | -1.93378 | -1.98490 | -1.88604 | True | -2.04567 | -1.81918 | True | 0.96416 | 0.95965 |
| block_depth | base_type | seq20 | 514853 | squared_error | 403.99187 | 402.45397 | 405.52214 | True | 395.96878 | 412.06201 | True | 0.03200 | 0.97338 |
| block_depth | seq1 | seq20 | 514853 | squared_error | 3.36876 | 3.31026 | 3.42673 | True | 3.18792 | 3.55344 | True | 0.96553 | 0.97338 |
| block_depth | seq0 | seq20 | 514853 | squared_error | 5.89250 | 5.82390 | 5.96857 | True | 5.65852 | 6.13139 | True | 0.95965 | 0.97338 |
| block_depth | seq0 | seq1 | 514853 | squared_error | 2.52373 | 2.46715 | 2.58415 | True | 2.36538 | 2.67986 | True | 0.95965 | 0.96553 |
| def_line | lgbm_E2 | seq20 | 514853 | squared_error | 3.10098 | 3.03907 | 3.16299 | True | 2.92178 | 3.28729 | True | 0.95168 | 0.96095 |
| def_line | lgbm_E2 | seq1 | 514853 | squared_error | 0.55732 | 0.50457 | 0.60931 | True | 0.44565 | 0.67234 | True | 0.95168 | 0.95335 |
| def_line | lgbm_E2 | seq0 | 514853 | squared_error | -1.27970 | -1.32578 | -1.22885 | True | -1.38243 | -1.18019 | True | 0.95168 | 0.94786 |
| def_line | base_type | seq20 | 514853 | squared_error | 311.50945 | 310.19753 | 312.77875 | True | 304.90754 | 317.85736 | True | 0.03038 | 0.96095 |
| def_line | seq1 | seq20 | 514853 | squared_error | 2.54367 | 2.49185 | 2.59903 | True | 2.38651 | 2.73646 | True | 0.95335 | 0.96095 |
| def_line | seq0 | seq20 | 514853 | squared_error | 4.38068 | 4.31792 | 4.44471 | True | 4.18321 | 4.58299 | True | 0.94786 | 0.96095 |
| def_line | seq0 | seq1 | 514853 | squared_error | 1.83701 | 1.78556 | 1.88947 | True | 1.70213 | 1.96603 | True | 0.94786 | 0.95335 |
| n_opp_ahead_of_ball | lgbm_E2 | seq20 | 514853 | squared_error | 0.36538 | 0.35863 | 0.37211 | True | 0.34507 | 0.38601 | True | 0.67142 | 0.72778 |
| n_opp_ahead_of_ball | lgbm_E2 | seq1 | 514853 | squared_error | 0.10210 | 0.09683 | 0.10745 | True | 0.09010 | 0.11420 | True | 0.67142 | 0.68717 |
| n_opp_ahead_of_ball | lgbm_E2 | seq0 | 514853 | squared_error | -0.08127 | -0.08586 | -0.07674 | True | -0.09116 | -0.07131 | True | 0.67142 | 0.65888 |
| n_opp_ahead_of_ball | base_type | seq20 | 514853 | squared_error | 4.36895 | 4.34894 | 4.39070 | True | 4.26866 | 4.47059 | True | 0.05379 | 0.72778 |
| n_opp_ahead_of_ball | seq1 | seq20 | 514853 | squared_error | 0.26328 | 0.25751 | 0.26894 | True | 0.24748 | 0.27946 | True | 0.68717 | 0.72778 |
| n_opp_ahead_of_ball | seq0 | seq20 | 514853 | squared_error | 0.44665 | 0.44011 | 0.45351 | True | 0.42747 | 0.46553 | True | 0.65888 | 0.72778 |
| n_opp_ahead_of_ball | seq0 | seq1 | 514853 | squared_error | 0.18337 | 0.17840 | 0.18836 | True | 0.17273 | 0.19356 | True | 0.65888 | 0.68717 |
| nearest_opp_dist | lgbm_E2 | seq20 | 799585 | squared_error | 0.43014 | 0.40793 | 0.45385 | True | 0.38793 | 0.47755 | True | 0.66649 | 0.68278 |
| nearest_opp_dist | lgbm_E2 | seq1 | 799585 | squared_error | -0.03934 | -0.06120 | -0.01636 | True | -0.07211 | -0.00661 | True | 0.66649 | 0.66500 |
| nearest_opp_dist | lgbm_E2 | seq0 | 799585 | squared_error | -0.39292 | -0.41514 | -0.37092 | True | -0.42506 | -0.35874 | True | 0.66649 | 0.65162 |
| nearest_opp_dist | base_type | seq20 | 799585 | squared_error | 13.73469 | 13.61679 | 13.84852 | True | 13.37179 | 14.12109 | True | 0.16268 | 0.68278 |
| nearest_opp_dist | seq1 | seq20 | 799585 | squared_error | 0.46948 | 0.44765 | 0.48964 | True | 0.42878 | 0.51316 | True | 0.66500 | 0.68278 |
| nearest_opp_dist | seq0 | seq20 | 799585 | squared_error | 0.82306 | 0.79963 | 0.84718 | True | 0.77357 | 0.87019 | True | 0.65162 | 0.68278 |
| nearest_opp_dist | seq0 | seq1 | 799585 | squared_error | 0.35358 | 0.33324 | 0.37361 | True | 0.32344 | 0.38341 | True | 0.65162 | 0.66500 |
| n_opp_in_cone | lgbm_E2 | seq20 | 666441 | squared_error | 0.00474 | 0.00430 | 0.00516 | True | 0.00412 | 0.00537 | True | 0.15976 | 0.17513 |
| n_opp_in_cone | lgbm_E2 | seq1 | 666441 | squared_error | 0.00218 | 0.00183 | 0.00257 | True | 0.00177 | 0.00267 | True | 0.15976 | 0.16682 |
| n_opp_in_cone | lgbm_E2 | seq0 | 666441 | squared_error | -0.00040 | -0.00077 | 0.00002 | False | -0.00092 | 0.00013 | False | 0.15976 | 0.15847 |
| n_opp_in_cone | base_type | seq20 | 666441 | squared_error | 0.05136 | 0.05021 | 0.05244 | True | 0.04902 | 0.05366 | True | 0.00859 | 0.17513 |
| n_opp_in_cone | seq1 | seq20 | 666441 | squared_error | 0.00256 | 0.00221 | 0.00288 | True | 0.00204 | 0.00308 | True | 0.16682 | 0.17513 |
| n_opp_in_cone | seq0 | seq20 | 666441 | squared_error | 0.00514 | 0.00474 | 0.00552 | True | 0.00455 | 0.00572 | True | 0.15847 | 0.17513 |
| n_opp_in_cone | seq0 | seq1 | 666441 | squared_error | 0.00258 | 0.00227 | 0.00290 | True | 0.00211 | 0.00307 | True | 0.15847 | 0.16682 |
| deep_block | lgbm_E2 | seq20 | 143078 | log_loss | 0.00995 | 0.00861 | 0.01145 | True | 0.00749 | 0.01231 | True | 0.77356 | 0.79180 |
| deep_block | lgbm_E2 | seq1 | 143078 | log_loss | -0.00265 | -0.00399 | -0.00126 | True | -0.00449 | -0.00074 | True | 0.77356 | 0.76960 |
| deep_block | lgbm_E2 | seq0 | 143078 | log_loss | -0.01105 | -0.01223 | -0.00989 | True | -0.01311 | -0.00911 | True | 0.77356 | 0.75650 |
| deep_block | base_type | seq20 | 143078 | log_loss | 0.40661 | 0.40346 | 0.41008 | True | 0.39696 | 0.41667 | True | 0.05251 | 0.79180 |
| deep_block | seq1 | seq20 | 143078 | log_loss | 0.01260 | 0.01129 | 0.01405 | True | 0.01046 | 0.01486 | True | 0.76960 | 0.79180 |
| deep_block | seq0 | seq20 | 143078 | log_loss | 0.02100 | 0.01943 | 0.02253 | True | 0.01825 | 0.02367 | True | 0.75650 | 0.79180 |
| deep_block | seq0 | seq1 | 143078 | log_loss | 0.00840 | 0.00699 | 0.00978 | True | 0.00641 | 0.01056 | True | 0.75650 | 0.76960 |
| counter_on | lgbm_E2 | seq20 | 277676 | log_loss | 0.02037 | 0.01926 | 0.02144 | True | 0.01824 | 0.02249 | True | 0.34310 | 0.41006 |
| counter_on | lgbm_E2 | seq1 | 277676 | log_loss | 0.00663 | 0.00584 | 0.00743 | True | 0.00543 | 0.00782 | True | 0.34310 | 0.36002 |
| counter_on | lgbm_E2 | seq0 | 277676 | log_loss | -0.00325 | -0.00389 | -0.00252 | True | -0.00421 | -0.00222 | True | 0.34310 | 0.32680 |
| counter_on | base_type | seq20 | 277676 | log_loss | 0.12505 | 0.12305 | 0.12697 | True | 0.11980 | 0.13019 | True | 0.03645 | 0.41006 |
| counter_on | seq1 | seq20 | 277676 | log_loss | 0.01374 | 0.01281 | 0.01457 | True | 0.01213 | 0.01549 | True | 0.36002 | 0.41006 |
| counter_on | seq0 | seq20 | 277676 | log_loss | 0.02362 | 0.02256 | 0.02468 | True | 0.02162 | 0.02591 | True | 0.32680 | 0.41006 |
| counter_on | seq0 | seq1 | 277676 | log_loss | 0.00988 | 0.00911 | 0.01063 | True | 0.00871 | 0.01112 | True | 0.32680 | 0.36002 |

## Error vs elapsed possession time

MAE (continuous / count) or log-loss (binary) per bin of `f_poss_elapsed` (seconds since the possession started); n per bin.


block_depth (MAE):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 2.906 | 3.280 | 3.404 | 3.060 | 22.567 | 56916 |
| [2.0, 5.0) | 2.810 | 3.101 | 3.199 | 2.967 | 20.422 | 59822 |
| [5.0, 10.0) | 2.824 | 3.157 | 3.381 | 3.188 | 17.585 | 75390 |
| [10.0, 20.0) | 2.661 | 3.021 | 3.319 | 3.149 | 15.205 | 110972 |
| [20.0, 40.0) | 2.436 | 2.785 | 3.095 | 2.936 | 14.198 | 123419 |
| [40.0, inf) | 2.273 | 2.613 | 2.960 | 2.754 | 14.112 | 88334 |

def_line (MAE):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 3.393 | 3.662 | 3.749 | 3.485 | 20.526 | 56916 |
| [2.0, 5.0) | 3.100 | 3.295 | 3.352 | 3.189 | 18.086 | 59822 |
| [5.0, 10.0) | 2.931 | 3.165 | 3.345 | 3.206 | 15.378 | 75390 |
| [10.0, 20.0) | 2.692 | 2.957 | 3.209 | 3.086 | 13.259 | 110972 |
| [20.0, 40.0) | 2.431 | 2.689 | 2.956 | 2.842 | 12.298 | 123419 |
| [40.0, inf) | 2.232 | 2.494 | 2.771 | 2.638 | 12.126 | 88334 |

n_opp_ahead_of_ball (MAE):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 1.103 | 1.167 | 1.176 | 1.149 | 2.264 | 56916 |
| [2.0, 5.0) | 1.086 | 1.145 | 1.168 | 1.141 | 2.160 | 59822 |
| [5.0, 10.0) | 1.073 | 1.148 | 1.192 | 1.174 | 2.007 | 75390 |
| [10.0, 20.0) | 1.058 | 1.145 | 1.202 | 1.184 | 1.957 | 110972 |
| [20.0, 40.0) | 1.035 | 1.120 | 1.180 | 1.164 | 1.960 | 123419 |
| [40.0, inf) | 1.023 | 1.109 | 1.167 | 1.147 | 2.024 | 88334 |

nearest_opp_dist (MAE):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 2.116 | 2.192 | 2.229 | 2.160 | 4.102 | 96004 |
| [2.0, 5.0) | 1.957 | 2.000 | 2.035 | 1.976 | 3.412 | 101165 |
| [5.0, 10.0) | 2.088 | 2.128 | 2.173 | 2.113 | 3.384 | 127339 |
| [10.0, 20.0) | 2.088 | 2.130 | 2.174 | 2.129 | 3.356 | 175298 |
| [20.0, 40.0) | 2.012 | 2.052 | 2.105 | 2.052 | 3.241 | 179388 |
| [40.0, inf) | 1.955 | 1.993 | 2.048 | 1.987 | 3.162 | 120391 |

n_opp_in_cone (MAE):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 0.323 | 0.332 | 0.323 | 0.332 | 0.416 | 84012 |
| [2.0, 5.0) | 0.349 | 0.358 | 0.350 | 0.357 | 0.438 | 81294 |
| [5.0, 10.0) | 0.361 | 0.369 | 0.366 | 0.370 | 0.435 | 103866 |
| [10.0, 20.0) | 0.378 | 0.386 | 0.385 | 0.389 | 0.440 | 145458 |
| [20.0, 40.0) | 0.404 | 0.412 | 0.412 | 0.415 | 0.460 | 150164 |
| [40.0, inf) | 0.429 | 0.436 | 0.438 | 0.441 | 0.479 | 101647 |

deep_block (log-loss):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [10.0, 20.0) | 0.093 | 0.102 | 0.107 | 0.100 | 0.480 | 46879 |
| [20.0, 40.0) | 0.127 | 0.140 | 0.148 | 0.137 | 0.532 | 55456 |
| [40.0, inf) | 0.159 | 0.176 | 0.188 | 0.173 | 0.590 | 40743 |

counter_on (log-loss):

| poss_elapsed_s | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| [0.0, 2.0) | 0.249 | 0.266 | 0.271 | 0.261 | 0.417 | 27351 |
| [2.0, 5.0) | 0.265 | 0.277 | 0.289 | 0.278 | 0.495 | 31698 |
| [5.0, 10.0) | 0.185 | 0.204 | 0.218 | 0.212 | 0.344 | 42118 |
| [10.0, 20.0) | 0.128 | 0.145 | 0.157 | 0.156 | 0.237 | 63445 |
| [20.0, 40.0) | 0.093 | 0.103 | 0.112 | 0.112 | 0.178 | 67970 |
| [40.0, inf) | 0.060 | 0.068 | 0.075 | 0.075 | 0.136 | 45094 |

## Skill by event type (types with >= 2,000 scored rows)


block_depth:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.973 | 0.967 | 0.964 | 0.968 | -0.001 | 163274 |
| Ball Receipt* | 0.971 | 0.965 | 0.949 | 0.954 | -0.001 | 159380 |
| Carry | 0.971 | 0.959 | 0.958 | 0.963 | -0.001 | 141136 |
| Ball Recovery | 0.982 | 0.976 | 0.971 | 0.974 | -0.000 | 11609 |
| Pressure | 0.972 | 0.964 | 0.953 | 0.962 | -0.000 | 7781 |
| Shot | 0.805 | 0.739 | 0.646 | 0.784 | -0.000 | 5534 |
| Dribble | 0.981 | 0.973 | 0.971 | 0.976 | -0.000 | 5210 |
| Duel | 0.981 | 0.977 | 0.974 | 0.978 | -0.000 | 4893 |
| Miscontrol | 0.979 | 0.972 | 0.969 | 0.973 | -0.001 | 4381 |
| Dispossessed | 0.981 | 0.975 | 0.973 | 0.978 | -0.000 | 3613 |
| Foul Won | 0.974 | 0.956 | 0.955 | 0.965 | -0.000 | 3290 |

def_line:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.959 | 0.953 | 0.950 | 0.954 | -0.001 | 163274 |
| Ball Receipt* | 0.958 | 0.953 | 0.938 | 0.942 | -0.001 | 159380 |
| Carry | 0.958 | 0.946 | 0.946 | 0.950 | -0.001 | 141136 |
| Ball Recovery | 0.973 | 0.966 | 0.961 | 0.965 | -0.000 | 11609 |
| Pressure | 0.963 | 0.957 | 0.945 | 0.952 | -0.000 | 7781 |
| Shot | 0.810 | 0.750 | 0.677 | 0.793 | -0.000 | 5534 |
| Dribble | 0.970 | 0.962 | 0.962 | 0.967 | -0.001 | 5210 |
| Duel | 0.973 | 0.967 | 0.966 | 0.970 | -0.000 | 4893 |
| Miscontrol | 0.971 | 0.963 | 0.961 | 0.965 | -0.001 | 4381 |
| Dispossessed | 0.971 | 0.963 | 0.963 | 0.967 | -0.000 | 3613 |
| Foul Won | 0.956 | 0.940 | 0.939 | 0.948 | -0.000 | 3290 |

n_opp_ahead_of_ball:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.696 | 0.661 | 0.654 | 0.664 | -0.000 | 163274 |
| Ball Receipt* | 0.735 | 0.708 | 0.634 | 0.648 | -0.000 | 159380 |
| Carry | 0.720 | 0.656 | 0.658 | 0.670 | -0.000 | 141136 |
| Ball Recovery | 0.701 | 0.651 | 0.601 | 0.621 | -0.000 | 11609 |
| Pressure | 0.646 | 0.599 | 0.531 | 0.555 | -0.000 | 7781 |
| Shot | 0.792 | 0.738 | 0.714 | 0.752 | -0.000 | 5534 |
| Dribble | 0.593 | 0.498 | 0.477 | 0.518 | -0.000 | 5210 |
| Duel | 0.566 | 0.515 | 0.498 | 0.529 | -0.000 | 4893 |
| Miscontrol | 0.606 | 0.506 | 0.500 | 0.518 | -0.001 | 4381 |
| Dispossessed | 0.587 | 0.516 | 0.497 | 0.534 | -0.002 | 3613 |
| Foul Won | 0.563 | 0.386 | 0.403 | 0.435 | -0.002 | 3290 |

nearest_opp_dist:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.617 | 0.604 | 0.595 | 0.609 | -0.000 | 222957 |
| Ball Receipt* | 0.625 | 0.616 | 0.597 | 0.607 | -0.000 | 210866 |
| Carry | 0.660 | 0.615 | 0.608 | 0.631 | -0.000 | 192960 |
| Pressure | 0.193 | 0.188 | 0.174 | 0.186 | -0.000 | 71039 |
| Ball Recovery | 0.558 | 0.548 | 0.450 | 0.500 | -0.000 | 21677 |
| Duel | -0.260 | 0.044 | 0.044 | 0.103 | -0.001 | 13936 |
| Clearance | 0.180 | 0.170 | 0.111 | 0.173 | -0.000 | 8940 |
| Block | 0.106 | 0.093 | 0.042 | 0.080 | -0.002 | 8811 |
| Dribble | -0.057 | 0.008 | -0.090 | 0.299 | -0.000 | 6699 |
| Shot | 0.590 | 0.567 | 0.562 | 0.606 | -0.001 | 6279 |
| Miscontrol | 0.383 | 0.310 | 0.268 | 0.336 | -0.000 | 5929 |
| Foul Committed | -0.185 | -0.102 | -0.105 | -0.062 | -0.001 | 5409 |
| Interception | 0.063 | 0.106 | 0.072 | 0.127 | -0.000 | 5344 |
| Foul Won | -5.476 | -2.798 | -8.069 | -3.008 | -0.001 | 4977 |
| Dispossessed | -16.356 | -11.717 | -11.004 | -2.560 | -0.000 | 4919 |
| Dribbled Past | -33.655 | -16.793 | -12.175 | -12.984 | -0.000 | 3969 |
| Goal Keeper | 0.520 | 0.490 | 0.267 | 0.351 | -0.000 | 3493 |

n_opp_in_cone:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.129 | 0.124 | 0.122 | 0.122 | -0.000 | 211313 |
| Ball Receipt* | 0.149 | 0.146 | 0.123 | 0.123 | -0.000 | 204234 |
| Carry | 0.169 | 0.153 | 0.153 | 0.156 | -0.000 | 185313 |
| Ball Recovery | 0.374 | 0.364 | 0.340 | 0.351 | -0.000 | 16996 |
| Pressure | 0.274 | 0.262 | 0.246 | 0.247 | -0.000 | 9530 |
| Duel | 0.120 | 0.112 | 0.116 | 0.094 | -0.000 | 6376 |
| Shot | 0.437 | 0.423 | 0.422 | 0.422 | -0.000 | 6173 |
| Dribble | 0.168 | 0.138 | 0.133 | 0.138 | -0.000 | 6016 |
| Miscontrol | 0.133 | 0.112 | 0.109 | 0.122 | -0.000 | 5091 |
| Dispossessed | 0.112 | 0.113 | 0.103 | 0.101 | -0.000 | 4109 |
| Foul Won | 0.110 | 0.080 | 0.081 | 0.092 | -0.000 | 4028 |

deep_block:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Ball Receipt* | 0.789 | 0.774 | 0.738 | 0.755 | 0.005 | 45492 |
| Pass | 0.779 | 0.758 | 0.757 | 0.770 | 0.005 | 45287 |
| Carry | 0.772 | 0.734 | 0.734 | 0.753 | 0.008 | 40762 |
| Ball Recovery | 0.860 | 0.843 | 0.822 | 0.831 | 0.244 | 2167 |
| Pressure | 0.801 | 0.806 | 0.771 | 0.803 | 0.180 | 2107 |

counter_on:

| f_type | seq20 | seq1 | seq0 | lgbm_E2 | base_type | n |
|---|---|---|---|---|---|---|
| Pass | 0.355 | 0.308 | 0.296 | 0.312 | 0.055 | 91757 |
| Ball Receipt* | 0.426 | 0.388 | 0.321 | 0.340 | 0.003 | 87931 |
| Carry | 0.401 | 0.338 | 0.333 | 0.344 | 0.000 | 79413 |
| Ball Recovery | 0.408 | 0.336 | 0.287 | 0.314 | 0.021 | 4753 |
| Pressure | 0.425 | 0.399 | 0.307 | 0.343 | 0.093 | 2766 |
| Duel | 0.546 | 0.530 | 0.531 | 0.530 | 0.274 | 2292 |

## Fits

| variant | fold | n_train | n_val | epochs | best_epoch | best_val_loss | stop_reason | minutes | n_params | n_train_block_depth | n_train_def_line | n_train_n_opp_ahead_of_ball | n_train_nearest_opp_dist | n_train_n_opp_in_cone | n_train_deep_block | n_train_counter_on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq20 | 0 | 300000 | 60000 | 12 | 8 | 1.735 | early_stopping | 6.289 | 142743 | 188815 | 188815 | 188815 | 299692 | 249393 | 52467 | 101187 |
| seq20 | 1 | 300000 | 60000 | 14 | 10 | 1.785 | early_stopping | 7.452 | 142743 | 190321 | 190321 | 190321 | 299672 | 250102 | 52365 | 101937 |
| seq20 | 2 | 300000 | 60000 | 15 | 11 | 1.765 | early_stopping | 7.830 | 142743 | 188943 | 188943 | 188943 | 299676 | 249339 | 52011 | 100852 |
| seq1 | 0 | 300000 | 60000 | 15 | 11 | 1.847 | early_stopping | 2.139 | 142743 | 188815 | 188815 | 188815 | 299692 | 249393 | 52467 | 101187 |
| seq1 | 1 | 300000 | 60000 | 15 | 11 | 1.886 | early_stopping | 2.149 | 142743 | 190321 | 190321 | 190321 | 299672 | 250102 | 52365 | 101937 |
| seq1 | 2 | 300000 | 60000 | 15 | 11 | 1.860 | early_stopping | 2.152 | 142743 | 188943 | 188943 | 188943 | 299676 | 249339 | 52011 | 100852 |
| seq0 | 0 | 300000 | 60000 | 15 | 12 | 1.925 | max_epochs | 1.618 | 97175 | 188815 | 188815 | 188815 | 299692 | 249393 | 52467 | 101187 |
| seq0 | 1 | 300000 | 60000 | 15 | 13 | 1.969 | max_epochs | 1.612 | 97175 | 190321 | 190321 | 190321 | 299672 | 250102 | 52365 | 101937 |
| seq0 | 2 | 300000 | 60000 | 15 | 14 | 1.946 | max_epochs | 1.621 | 97175 | 188943 | 188943 | 188943 | 299676 | 249339 | 52011 | 100852 |

## Caveats

- Only folds [0, 1, 2] of the five stage-02 folds were trained (CPU budget); the LightGBM reference is scored on the same rows, so the comparison is paired, but the absolute numbers are on a subset of the 417 matches.
- The multi-task network sees one thinned training set for all seven targets, so per target it trains on fewer labelled rows than the corresponding LightGBM student (see `n_train_<target>` in the fits table vs 300,000 per stage-02 student); the comparison favours LightGBM on the sparse subsets (`deep_block`, `counter_on`).
- The stage-02 review's mechanism caveat stands: `block_depth` / `def_line` / `deep_block` skill is mostly ball location plus visible-area truncation; the per-event-type baseline shows how little event type alone recovers.
- Sequence predictions are clipped at 0 for counts / distances, the stored LightGBM predictions are not (the stage-02 review's unclipped-output issue); clipping is loss-reducing by construction but changes squared error only where the prediction was negative.
- Per-row bootstrap intervals treat rows as independent; the match-clustered interval next to each delta accounts for within-match correlation and is the one to trust for significance.
- Within event types whose `nearest_opp_dist` label is a near-constant ~0.14 yd (Duel, Dribbled Past, Dispossessed, Foul Won, see `soccer_01_build.md`), the network is worse than LightGBM (more negative within-type R2 in the per-type table: trees isolate `f_type` exactly, the shared MSE head does not); the aggregate gain comes from Pass / Carry / Ball Receipt* rows. Absolute errors on those types stay ~1 yd.
- `seq0` reached the 15-epoch cap with the best epoch near the end (validation loss still improving by < 0.01 per epoch), so those variants are marginally under-trained; the gaps to `seq1` / `seq20` (fits table, `best_val_loss`) are an order of magnitude larger.
