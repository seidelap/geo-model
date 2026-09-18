# Soccer 03 - payoff of imputed defensive state: xG and xPass in the 360 matches

Machine-written by `research/privileged_tracking/soccer/payoff.py`. Question: does the stage-02 imputed 360 state (out-of-fold student predictions from event data only) improve an event-only xG / xPass model, how far is that from the 360 and shot-freeze-frame oracles, does distilling an oracle teacher into an event-only student beat training on goals, does the gain survive a cross-gender domain shift, and where does the imputed state help?

## Headline

- Shots: n = 10,233 non-penalty shots, 1,039 goals (0.102), 417 matches, 6,715 with a key pass; 5-fold match-grouped CV (stage-02 folds), 3 seeds averaged per variant.
- xG log-loss: BASE 0.3286, LOC 0.2945, EVENT 0.2745, EVENT+IMP 0.2748, EVENT+ORACLE360 0.2716, EVENT+ORACLESHOT 0.2666, EVENT+ORACLESHOT_full 0.2639, StatsBomb xG 0.2617.
- Paired log-loss deltas vs EVENT (positive = better; per-shot bootstrap 95% CI): EVENT+IMP -0.0003 [-0.0013, +0.0007]; EVENT+ORACLE360 +0.0029 [+0.0013, +0.0045]; EVENT+ORACLESHOT +0.0078 [+0.0055, +0.0102]; StatsBomb xG +0.0128 [+0.0090, +0.0168].
- Distillation EVENT<-EVENT+ORACLE360@1 (distilled): -0.0001 [-0.0013, +0.0011] vs EVENT (direct).
- Distillation EVENT<-EVENT+ORACLE360@0.5 (distilled): +0.0006 [+0.0000, +0.0012] vs EVENT (direct).
- Distillation EVENT<-EVENT+ORACLESHOT@1 (distilled): +0.0001 [-0.0012, +0.0013] vs EVENT (direct).
- Distillation EVENT+IMP<-EVENT+ORACLE360@1 (distilled): -0.0002 [-0.0014, +0.0011] vs EVENT (direct).
- Learning curve, 10% of the training matches: EVENT+IMP vs EVENT +0.0000 [-0.0012, +0.0013] (EVENT log-loss 0.2918).
- Learning curve, 25% of the training matches: EVENT+IMP vs EVENT +0.0015 [-0.0001, +0.0031] (EVENT log-loss 0.2856).
- Learning curve, 50% of the training matches: EVENT+IMP vs EVENT -0.0012 [-0.0023, -0.0000] (EVENT log-loss 0.2784).
- Learning curve, 100% of the training matches: EVENT+IMP vs EVENT -0.0003 [-0.0013, +0.0007] (EVENT log-loss 0.2745).
- xPass (after): n = 149,591 passes, completion 0.826; log-loss BASE 0.4624, EVENT 0.2089, EVENT+IMP 0.2083 (+0.0005 [-0.0001, +0.0011]), EVENT+ORACLE 0.2009 (+0.0080 [+0.0072, +0.0088]).
- xPass (noafter): n = 149,591 passes, completion 0.826; log-loss BASE 0.4624, EVENT 0.3486, EVENT+IMP 0.3470 (+0.0016 [+0.0010, +0.0022]), EVENT+ORACLE 0.2783 (+0.0703 [+0.0685, +0.0721]).
- Transfer m2f: EVENT+IMP -0.0004 [-0.0021, +0.0013], EVENT+ORACLE360 +0.0040 [+0.0014, +0.0066] vs EVENT (students and xG trained on the source gender only).
- Transfer f2m: EVENT+IMP -0.0006 [-0.0017, +0.0004], EVENT+ORACLE360 +0.0019 [-0.0003, +0.0040] vs EVENT (students and xG trained on the source gender only).

### Reading

- xG, imputed state (EVENT+IMP vs EVENT): no measurable difference (-0.0003, CI [-0.0013, +0.0007]); the 360 oracle gives significant gain (+0.0029) and the shot freeze frame significant gain (+0.0078). The event-only student recovers none of the oracle gain on shots.
- Distillation: best run EVENT<-EVENT+ORACLE360@0.5 (distilled): significant gain (+0.0006).
- Learning curve (EVENT+IMP vs EVENT at each training share): 10% no measurable difference (+0.0000, CI [-0.0012, +0.0013]); 25% no measurable difference (+0.0015, CI [-0.0001, +0.0031]); 50% significant loss (-0.0012); 100% no measurable difference (-0.0003, CI [-0.0013, +0.0007]).
- xPass with the destination known: EVENT+IMP no measurable difference (+0.0005, CI [-0.0001, +0.0011]); EVENT+ORACLE significant gain (+0.0080); dropping the realised duration from EVENT costs significant loss (-0.0140). Without the destination: EVENT+IMP significant gain (+0.0016) (the no-after oracle gap is not attainable, see caveats).
- Cross-gender transfer (source-only students and xG): m2f: EVENT+IMP no measurable difference (-0.0004, CI [-0.0021, +0.0013]), EVENT+ORACLE360 significant gain (+0.0040); f2m: EVENT+IMP no measurable difference (-0.0006, CI [-0.0017, +0.0004]), EVENT+ORACLE360 no measurable difference (+0.0019, CI [-0.0003, +0.0040]).
- Slices: no play-pattern, distance-band or reliability slice shows a significant EVENT+IMP effect on xG (per-shot CIs); the oracle gains are largest on close-range shots (distance_band table).

## Protocol

- Population: Shot rows of `events360.parquet` (417 matches with usable 360); penalties excluded (yes); label = `post_shot_outcome == 'Goal'`. Shots keep their 360-unusable / unreliable frames (oracle columns are NaN there; the reliability slice separates them); a both-frames-usable subset is reported separately.
- Splits: the stage-02 `fold` column (`group_kfold` by match, seed 0, 5 folds), so every imputed feature of a test-fold shot comes from students that never saw that fold's matches (the imputations are `imputed_oof.parquet` columns). Rounds by early stopping on an inner 15% match holdout, then refit on all training rows; probabilities averaged over seeds. BASE = training-fold goal rate.
- Features: see the feature-set table. EVENT reads only `f_*` columns of the shot (stage-02 E2 design minus the post-instant `f_after_duration`) plus the key pass's attributes (`shot.key_pass_id` from the raw events; that pass is complete before the shot instant). Never used: `shot.freeze_frame`, `statsbomb_xg`, `one_on_one`, `open_goal`, `post_*`, any `y_*` outside the named oracle blocks. Imputed counts / distances are clipped at 0 (stage-02 outputs are unclipped).
- Metrics: log-loss, Brier, AUC, ECE (10 equal-count bins); deltas are paired per-shot bootstraps of the per-shot loss (2000 resamples, seed 0) with a match-clustered CI next to them; `significant` = the per-shot CI excludes 0.
- StatsBomb xG is the vendor's model output as stored in the events (a reference, not a variant trained here; its training set may include these matches).

### Feature sets

| task | variant | n_features | description |
|---|---|---|---|
| xG | BASE | 0 | fold-wise goal rate of the training matches |
| xG | LOC | 5 | x, y, distance, opening angle, bearing (LightGBM) |
| xG | EVENT | 72 | event-only: stage-02 E2 design on the shot (location geometry, body part, technique, shot type, first time, play pattern, position, under pressure, minute / period, score, possession context, 10-event window, opponent defensive actions, gender, competition type) + assist (key pass) attributes |
| xG | EVENT+IMP | 79 | EVENT + out-of-fold E2 student predictions of the five shot-state quantities + E2a predictions of the assist's pass-lane quantities |
| xG | EVENT+IMP(E0) | 79 | ablation: EVENT + out-of-fold E0 (current-event-only) student predictions |
| xG | EVENT+ORACLE360 | 79 | EVENT + the same quantities measured on the 360 frame (stage-02 label rules: NaN where the frame is unusable / unreliable / the cone is empty) |
| xG | EVENT+ORACLESHOT | 79 | EVENT + the same quantities from shot.freeze_frame (n_opp_ahead_of_ball stands in for block depth) + the assist's 360 pass-lane quantities |
| xG | EVENT+ORACLESHOT_full | 89 | EVENT + every sff_* column of the shot freeze frame |
| xPass | BASE |  | fold-wise completion rate of the training matches |
| xPass | EVENT |  | event-only: stage-02 E2a design on the pass (start / end location, length, angle, height, duration, switch / cross / through-ball / cut-back flags, pass type, body part, under pressure, play pattern, position, possession context, window, gender, competition type); the no-after design drops the realised end / length / height / flags |
| xPass | EVENT-nodur |  | ablation (after design only): EVENT without the realised pass duration, the one E2a student input that the task's xPass feature list does not name |
| xPass | EVENT+IMP |  | EVENT + out-of-fold student predictions of the seven pass-state quantities (E2a students for the with-after design, E2 students for the no-after design) |
| xPass | EVENT+ORACLE |  | EVENT + the same quantities measured on the 360 frame |
| xG | STATSBOMB_XG | 0 | StatsBomb's own xG as given (reference; its training data may include these matches) |

## (a) xG: event-only vs imputed vs oracle state

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 10233 | 1039 | 0.3286 | 0.0912 | 0.4820 | 0.0073 | 0.1016 |
| LOC | 10233 | 1039 | 0.2945 | 0.0839 | 0.7242 | 0.0107 | 0.1011 |
| EVENT | 10233 | 1039 | 0.2745 | 0.0783 | 0.7847 | 0.0067 | 0.0997 |
| EVENT+IMP(E0) | 10233 | 1039 | 0.2744 | 0.0782 | 0.7836 | 0.0065 | 0.0998 |
| EVENT+IMP | 10233 | 1039 | 0.2748 | 0.0784 | 0.7842 | 0.0078 | 0.0998 |
| EVENT+ORACLE360 | 10233 | 1039 | 0.2716 | 0.0773 | 0.7891 | 0.0082 | 0.0995 |
| EVENT+ORACLESHOT | 10233 | 1039 | 0.2666 | 0.0759 | 0.8003 | 0.0081 | 0.0997 |
| EVENT+ORACLESHOT_full | 10233 | 1039 | 0.2639 | 0.0750 | 0.8039 | 0.0071 | 0.0999 |
| STATSBOMB_XG | 10233 | 1039 | 0.2617 | 0.0742 | 0.8091 | 0.0099 | 0.0991 |

Paired deltas vs EVENT (log-loss, positive = better than EVENT):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 10233 | -0.0541 [-0.0604, -0.0478] | [-0.0601, -0.0478] | -0.01293 [-0.01481, -0.01102] | True |
| LOC | 10233 | -0.0200 [-0.0238, -0.0164] | [-0.0237, -0.0163] | -0.00557 [-0.00675, -0.00438] | True |
| EVENT+IMP(E0) | 10233 | +0.0001 [-0.0008, +0.0010] | [-0.0008, +0.0009] | +0.00013 [-0.00015, +0.00043] | False |
| EVENT+IMP | 10233 | -0.0003 [-0.0013, +0.0007] | [-0.0013, +0.0006] | -0.00013 [-0.00045, +0.00022] | False |
| EVENT+ORACLE360 | 10233 | +0.0029 [+0.0013, +0.0045] | [+0.0014, +0.0044] | +0.00100 [+0.00048, +0.00152] | True |
| EVENT+ORACLESHOT | 10233 | +0.0078 [+0.0055, +0.0102] | [+0.0055, +0.0103] | +0.00238 [+0.00161, +0.00312] | True |
| EVENT+ORACLESHOT_full | 10233 | +0.0106 [+0.0080, +0.0133] | [+0.0079, +0.0134] | +0.00337 [+0.00252, +0.00426] | True |
| STATSBOMB_XG | 10233 | +0.0128 [+0.0090, +0.0168] | [+0.0089, +0.0166] | +0.00413 [+0.00290, +0.00539] | True |

Both-frames-usable subset (`sff_present == 1` and `y_frame_ok == 1`):

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| BASE | 10223 | 1039 | 0.3288 | 0.0913 | 0.4819 | 0.0070 | 0.1016 |
| LOC | 10223 | 1039 | 0.2946 | 0.0840 | 0.7242 | 0.0107 | 0.1011 |
| EVENT | 10223 | 1039 | 0.2747 | 0.0784 | 0.7847 | 0.0067 | 0.0998 |
| EVENT+IMP(E0) | 10223 | 1039 | 0.2746 | 0.0783 | 0.7836 | 0.0065 | 0.0998 |
| EVENT+IMP | 10223 | 1039 | 0.2750 | 0.0785 | 0.7842 | 0.0079 | 0.0998 |
| EVENT+ORACLE360 | 10223 | 1039 | 0.2717 | 0.0774 | 0.7891 | 0.0083 | 0.0995 |
| EVENT+ORACLESHOT | 10223 | 1039 | 0.2668 | 0.0760 | 0.8003 | 0.0083 | 0.0997 |
| EVENT+ORACLESHOT_full | 10223 | 1039 | 0.2640 | 0.0750 | 0.8039 | 0.0072 | 0.0999 |
| STATSBOMB_XG | 10223 | 1039 | 0.2619 | 0.0743 | 0.8091 | 0.0099 | 0.0991 |

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| BASE | 10223 | -0.0541 [-0.0603, -0.0479] | [-0.0601, -0.0478] | -0.01294 [-0.01483, -0.01113] | True |
| LOC | 10223 | -0.0200 [-0.0238, -0.0164] | [-0.0237, -0.0163] | -0.00558 [-0.00679, -0.00443] | True |
| EVENT+IMP(E0) | 10223 | +0.0001 [-0.0009, +0.0009] | [-0.0008, +0.0009] | +0.00013 [-0.00016, +0.00041] | False |
| EVENT+IMP | 10223 | -0.0003 [-0.0013, +0.0007] | [-0.0013, +0.0006] | -0.00013 [-0.00045, +0.00021] | False |
| EVENT+ORACLE360 | 10223 | +0.0029 [+0.0014, +0.0046] | [+0.0014, +0.0045] | +0.00101 [+0.00051, +0.00155] | True |
| EVENT+ORACLESHOT | 10223 | +0.0078 [+0.0054, +0.0102] | [+0.0055, +0.0103] | +0.00238 [+0.00161, +0.00312] | True |
| EVENT+ORACLESHOT_full | 10223 | +0.0106 [+0.0079, +0.0132] | [+0.0079, +0.0134] | +0.00337 [+0.00245, +0.00427] | True |
| STATSBOMB_XG | 10223 | +0.0128 [+0.0087, +0.0165] | [+0.0089, +0.0166] | +0.00412 [+0.00288, +0.00535] | True |

### Calibration (equal-count bins: mean predicted vs observed goal rate)

| bin | obs_EVENT | obs_EVENT+IMP | obs_EVENT+ORACLE360 | obs_EVENT+ORACLESHOT | obs_STATSBOMB_XG | pred_EVENT | pred_EVENT+IMP | pred_EVENT+ORACLE360 | pred_EVENT+ORACLESHOT | pred_STATSBOMB_XG |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.018 | 0.021 | 0.020 | 0.019 | 0.018 | 0.022 | 0.023 | 0.023 | 0.021 | 0.012 |
| 1 | 0.021 | 0.022 | 0.019 | 0.018 | 0.022 | 0.028 | 0.030 | 0.029 | 0.029 | 0.024 |
| 2 | 0.034 | 0.030 | 0.028 | 0.029 | 0.026 | 0.035 | 0.036 | 0.035 | 0.034 | 0.032 |
| 3 | 0.039 | 0.044 | 0.048 | 0.032 | 0.024 | 0.042 | 0.044 | 0.042 | 0.041 | 0.041 |
| 4 | 0.044 | 0.039 | 0.040 | 0.045 | 0.048 | 0.052 | 0.053 | 0.051 | 0.049 | 0.051 |
| 5 | 0.067 | 0.065 | 0.070 | 0.065 | 0.065 | 0.066 | 0.066 | 0.064 | 0.061 | 0.064 |
| 6 | 0.099 | 0.089 | 0.087 | 0.090 | 0.070 | 0.087 | 0.087 | 0.084 | 0.079 | 0.081 |
| 7 | 0.119 | 0.131 | 0.125 | 0.115 | 0.118 | 0.120 | 0.119 | 0.116 | 0.114 | 0.110 |
| 8 | 0.191 | 0.186 | 0.186 | 0.188 | 0.203 | 0.182 | 0.180 | 0.180 | 0.182 | 0.172 |
| 9 | 0.384 | 0.388 | 0.393 | 0.415 | 0.419 | 0.364 | 0.360 | 0.372 | 0.388 | 0.406 |

### Gain share by feature block (mean over folds and seeds)

| variant | assist_event | assist_state | shot_event | shot_state |
|---|---|---|---|---|
| EVENT | 0.141 |  | 0.859 |  |
| EVENT+IMP | 0.115 | 0.019 | 0.747 | 0.118 |
| EVENT+IMP(E0) | 0.123 | 0.012 | 0.753 | 0.112 |
| EVENT+ORACLE360 | 0.107 | 0.007 | 0.751 | 0.135 |
| EVENT+ORACLESHOT | 0.101 | 0.004 | 0.670 | 0.225 |
| EVENT+ORACLESHOT_full | 0.089 | 0.003 | 0.607 | 0.301 |
| LOC |  |  | 1.000 |  |

Top features per variant:

| variant | feature | gain_share |
|---|---|---|
| EVENT | f_goal_opening | 0.238 |
| EVENT | f_dist_goal | 0.137 |
| EVENT | a_pass_through_ball | 0.044 |
| EVENT | f_position | 0.043 |
| EVENT | f_w10_n_other | 0.034 |
| EVENT | f_x | 0.029 |
| EVENT | f_t_since_opp_def_action | 0.029 |
| EVENT | f_opp_def_x_60s_mean | 0.027 |
| EVENT+IMP | f_goal_opening | 0.231 |
| EVENT+IMP | f_dist_goal | 0.137 |
| EVENT+IMP | s_opp_keeper_dist_to_goal_line | 0.047 |
| EVENT+IMP | a_pass_through_ball | 0.039 |
| EVENT+IMP | f_position | 0.037 |
| EVENT+IMP | s_nearest_opp_dist_in_cone | 0.030 |
| EVENT+IMP | f_w10_n_other | 0.028 |
| EVENT+IMP | f_t_since_opp_def_action | 0.026 |
| EVENT+IMP(E0) | f_goal_opening | 0.225 |
| EVENT+IMP(E0) | f_dist_goal | 0.134 |
| EVENT+IMP(E0) | a_pass_through_ball | 0.043 |
| EVENT+IMP(E0) | s_opp_keeper_dist_to_goal_line | 0.038 |
| EVENT+IMP(E0) | f_position | 0.036 |
| EVENT+IMP(E0) | f_w10_n_other | 0.032 |
| EVENT+IMP(E0) | f_t_since_opp_def_action | 0.027 |
| EVENT+IMP(E0) | s_nearest_opp_dist_in_cone | 0.025 |
| EVENT+ORACLE360 | f_goal_opening | 0.220 |
| EVENT+ORACLE360 | f_dist_goal | 0.131 |
| EVENT+ORACLE360 | s_opp_keeper_dist_to_goal_line | 0.065 |
| EVENT+ORACLE360 | f_position | 0.039 |
| EVENT+ORACLE360 | a_pass_through_ball | 0.030 |
| EVENT+ORACLE360 | s_block_depth | 0.029 |
| EVENT+ORACLE360 | f_x | 0.028 |
| EVENT+ORACLE360 | f_w10_n_other | 0.027 |
| EVENT+ORACLESHOT | f_goal_opening | 0.214 |
| EVENT+ORACLESHOT | f_dist_goal | 0.105 |
| EVENT+ORACLESHOT | s_block_depth | 0.082 |
| EVENT+ORACLESHOT | s_opp_keeper_dist_to_goal_line | 0.071 |
| EVENT+ORACLESHOT | s_nearest_opp_dist_in_cone | 0.061 |
| EVENT+ORACLESHOT | f_position | 0.035 |
| EVENT+ORACLESHOT | f_dt_prev | 0.025 |
| EVENT+ORACLESHOT | f_w10_n_other | 0.023 |
| EVENT+ORACLESHOT_full | f_goal_opening | 0.213 |
| EVENT+ORACLESHOT_full | f_dist_goal | 0.095 |
| EVENT+ORACLESHOT_full | s_sff_n_opp_ahead_of_ball | 0.076 |
| EVENT+ORACLESHOT_full | s_sff_opp_keeper_dist_to_goal_line | 0.057 |
| EVENT+ORACLESHOT_full | s_sff_nearest_opp_dist_in_cone | 0.049 |
| EVENT+ORACLESHOT_full | s_sff_nearest_opp_dist | 0.046 |
| EVENT+ORACLESHOT_full | s_sff_opp_keeper_in_cone | 0.032 |
| EVENT+ORACLESHOT_full | f_position | 0.032 |
| LOC | f_goal_opening | 0.434 |
| LOC | f_dist_goal | 0.330 |
| LOC | f_x | 0.096 |
| LOC | f_goal_bearing | 0.078 |
| LOC | f_y | 0.062 |

### Distillation: event-only student trained on an oracle teacher's xG

Runs are `student<-teacher@alpha` (alpha = weight of the teacher's probability in the soft label, 1 - alpha on the goal). Nested protocol: for every outer test fold the teacher is re-fitted by an inner CV over the training folds only, so the student never sees a test-fold label through the teacher; the student uses LightGBM's cross-entropy objective on the soft labels. Evaluated on goals.

| variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|
| EVENT (direct) | 10233 | 1039 | 0.2745 | 0.0783 | 0.7847 | 0.0067 | 0.0997 |
| EVENT+IMP (direct) | 10233 | 1039 | 0.2748 | 0.0784 | 0.7842 | 0.0078 | 0.0998 |
| EVENT+ORACLE360 (teacher) | 10233 | 1039 | 0.2716 | 0.0773 | 0.7891 | 0.0082 | 0.0995 |
| EVENT+ORACLESHOT (teacher) | 10233 | 1039 | 0.2666 | 0.0759 | 0.8003 | 0.0081 | 0.0997 |
| EVENT<-EVENT+ORACLE360@1 (distilled) | 10233 | 1039 | 0.2745 | 0.0784 | 0.7862 | 0.0132 | 0.0982 |
| EVENT<-EVENT+ORACLE360@0.5 (distilled) | 10233 | 1039 | 0.2738 | 0.0782 | 0.7863 | 0.0087 | 0.0989 |
| EVENT<-EVENT+ORACLESHOT@1 (distilled) | 10233 | 1039 | 0.2744 | 0.0784 | 0.7868 | 0.0135 | 0.0984 |
| EVENT+IMP<-EVENT+ORACLE360@1 (distilled) | 10233 | 1039 | 0.2746 | 0.0785 | 0.7863 | 0.0129 | 0.0981 |

Paired deltas vs EVENT (direct):

| variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|
| EVENT+IMP (direct) | 10233 | -0.0003 [-0.0013, +0.0007] | [-0.0013, +0.0006] | -0.00013 [-0.00045, +0.00022] | False |
| EVENT+ORACLE360 (teacher) | 10233 | +0.0029 [+0.0013, +0.0045] | [+0.0014, +0.0044] | +0.00100 [+0.00048, +0.00152] | True |
| EVENT+ORACLESHOT (teacher) | 10233 | +0.0078 [+0.0055, +0.0102] | [+0.0055, +0.0103] | +0.00238 [+0.00161, +0.00312] | True |
| EVENT<-EVENT+ORACLE360@1 (distilled) | 10233 | -0.0001 [-0.0013, +0.0011] | [-0.0012, +0.0011] | -0.00008 [-0.00046, +0.00029] | False |
| EVENT<-EVENT+ORACLE360@0.5 (distilled) | 10233 | +0.0006 [+0.0000, +0.0012] | [+0.0000, +0.0012] | +0.00015 [-0.00004, +0.00033] | True |
| EVENT<-EVENT+ORACLESHOT@1 (distilled) | 10233 | +0.0001 [-0.0012, +0.0013] | [-0.0012, +0.0014] | -0.00007 [-0.00048, +0.00032] | False |
| EVENT+IMP<-EVENT+ORACLE360@1 (distilled) | 10233 | -0.0002 [-0.0014, +0.0011] | [-0.0013, +0.0011] | -0.00015 [-0.00054, +0.00025] | False |

Student-teacher fidelity (out-of-fold student vs the teacher's own out-of-fold xG):

| run | corr_with_teacher_oof | mae_vs_teacher | mean_student | mean_teacher |
|---|---|---|---|---|
| EVENT<-EVENT+ORACLE360@1 | 0.9498 | 0.0188 | 0.0982 | 0.0995 |
| EVENT<-EVENT+ORACLE360@0.5 | 0.9592 | 0.0162 | 0.0989 | 0.0995 |
| EVENT<-EVENT+ORACLESHOT@1 | 0.9114 | 0.0258 | 0.0984 | 0.0997 |
| EVENT+IMP<-EVENT+ORACLE360@1 | 0.9490 | 0.0188 | 0.0981 | 0.0995 |

### Learning curve: does imputed state help when goal labels are scarce?

Each xG variant refitted with only a share of the training matches (test folds complete, same seeds); delta = paired log-loss gain over EVENT at the same share.

| train_share | variant | n | log_loss | brier | auc | delta [95% CI] |
|---|---|---|---|---|---|---|
| 0.1000 | EVENT | 10233 | 0.2918 | 0.0834 | 0.7477 |  |
| 0.1000 | EVENT+IMP | 10233 | 0.2918 | 0.0834 | 0.7479 | +0.0000 [-0.0012, +0.0013] |
| 0.1000 | EVENT+ORACLE360 | 10233 | 0.2886 | 0.0823 | 0.7568 | +0.0032 [+0.0016, +0.0049] |
| 0.1000 | EVENT+ORACLESHOT | 10233 | 0.2839 | 0.0811 | 0.7663 | +0.0079 [+0.0060, +0.0099] |
| 0.2500 | EVENT | 10233 | 0.2856 | 0.0814 | 0.7592 |  |
| 0.2500 | EVENT+IMP | 10233 | 0.2841 | 0.0809 | 0.7626 | +0.0015 [-0.0001, +0.0031] |
| 0.2500 | EVENT+ORACLE360 | 10233 | 0.2813 | 0.0803 | 0.7692 | +0.0043 [+0.0022, +0.0063] |
| 0.2500 | EVENT+ORACLESHOT | 10233 | 0.2767 | 0.0788 | 0.7798 | +0.0089 [+0.0064, +0.0113] |
| 0.5000 | EVENT | 10233 | 0.2784 | 0.0796 | 0.7777 |  |
| 0.5000 | EVENT+IMP | 10233 | 0.2796 | 0.0799 | 0.7751 | -0.0012 [-0.0023, -0.0000] |
| 0.5000 | EVENT+ORACLE360 | 10233 | 0.2750 | 0.0783 | 0.7813 | +0.0034 [+0.0017, +0.0051] |
| 0.5000 | EVENT+ORACLESHOT | 10233 | 0.2717 | 0.0776 | 0.7914 | +0.0066 [+0.0043, +0.0090] |
| 1.0000 | EVENT | 10233 | 0.2745 | 0.0783 | 0.7847 |  |
| 1.0000 | EVENT+IMP | 10233 | 0.2748 | 0.0784 | 0.7842 | -0.0003 [-0.0013, +0.0007] |
| 1.0000 | EVENT+ORACLE360 | 10233 | 0.2716 | 0.0773 | 0.7891 | +0.0029 [+0.0013, +0.0045] |
| 1.0000 | EVENT+ORACLESHOT | 10233 | 0.2666 | 0.0759 | 0.8003 | +0.0078 [+0.0055, +0.0102] |

## (b) xPass: pass completion in the 360 matches

Population: a match-stratified subsample of 149,591 pass attempts (target 150,000; `Unknown` / `Injury Clearance` outcomes dropped) of the 360 matches; label = `post_pass_outcome == Complete`. Two designs: `after` = the pass's realised end location / length / angle / height / switch / cross / through-ball / cut-back are features (the usual xPass setting: the destination is known) with E2a students; `noafter` = only what is known at the pass instant, with E2 students. The receiver-distance student (`nearest_opp_to_receiver`) had no stage-02 model and was fitted here out-of-fold on all Pass rows with the stage-02 settings (fit table below). Single seed; rounds by early stopping on an inner match holdout without refit.

| design | variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|---|
| after | BASE | 149591 | 123546 | 0.4624 | 0.1438 | 0.4916 | 0.0058 | 0.8259 |
| after | EVENT | 149591 | 123546 | 0.2089 | 0.0643 | 0.9498 | 0.0051 | 0.8264 |
| after | EVENT-nodur | 149591 | 123546 | 0.2229 | 0.0687 | 0.9426 | 0.0042 | 0.8261 |
| after | EVENT+IMP | 149591 | 123546 | 0.2083 | 0.0641 | 0.9498 | 0.0033 | 0.8262 |
| after | EVENT+ORACLE | 149591 | 123546 | 0.2009 | 0.0623 | 0.9542 | 0.0050 | 0.8265 |
| noafter | BASE | 149591 | 123546 | 0.4624 | 0.1438 | 0.4916 | 0.0058 | 0.8259 |
| noafter | EVENT | 149591 | 123546 | 0.3486 | 0.1065 | 0.8234 | 0.0040 | 0.8257 |
| noafter | EVENT+IMP | 149591 | 123546 | 0.3470 | 0.1061 | 0.8258 | 0.0039 | 0.8257 |
| noafter | EVENT+ORACLE | 149591 | 123546 | 0.2783 | 0.0858 | 0.9039 | 0.0046 | 0.8259 |

Paired deltas vs EVENT:

| design | variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|---|
| after | BASE | 149591 | -0.2535 [-0.2565, -0.2506] | [-0.2587, -0.2487] | -0.07952 [-0.08065, -0.07835] | True |
| after | EVENT-nodur | 149591 | -0.0140 [-0.0148, -0.0131] | [-0.0150, -0.0130] | -0.00443 [-0.00475, -0.00413] | True |
| after | EVENT+IMP | 149591 | +0.0005 [-0.0001, +0.0011] | [-0.0000, +0.0011] | +0.00016 [-0.00005, +0.00033] | False |
| after | EVENT+ORACLE | 149591 | +0.0080 [+0.0072, +0.0088] | [+0.0072, +0.0088] | +0.00198 [+0.00171, +0.00225] | True |
| noafter | BASE | 149591 | -0.1138 [-0.1160, -0.1114] | [-0.1169, -0.1110] | -0.03732 [-0.03813, -0.03643] | True |
| noafter | EVENT+IMP | 149591 | +0.0016 [+0.0010, +0.0022] | [+0.0011, +0.0022] | +0.00041 [+0.00022, +0.00062] | True |
| noafter | EVENT+ORACLE | 149591 | +0.0703 [+0.0685, +0.0721] | [+0.0681, +0.0723] | +0.02072 [+0.02005, +0.02135] | True |

Receiver-distance student (out-of-fold on all Pass rows):

| target | fset | fold | rounds | n_train | seconds |
|---|---|---|---|---|---|
| nearest_opp_to_receiver | E2a | 0 | 310 | 297373 | 11.3330 |
| nearest_opp_to_receiver | E2a | 1 | 217 | 296549 | 8.4698 |
| nearest_opp_to_receiver | E2a | 2 | 228 | 297547 | 8.6096 |
| nearest_opp_to_receiver | E2a | 3 | 302 | 295631 | 10.9351 |
| nearest_opp_to_receiver | E2a | 4 | 272 | 297440 | 10.2739 |
| nearest_opp_to_receiver | E2 | 0 | 186 | 297373 | 7.3700 |
| nearest_opp_to_receiver | E2 | 1 | 211 | 296549 | 7.7702 |
| nearest_opp_to_receiver | E2 | 2 | 188 | 297547 | 7.0548 |
| nearest_opp_to_receiver | E2 | 3 | 235 | 295631 | 9.0266 |
| nearest_opp_to_receiver | E2 | 4 | 164 | 297440 | 6.5027 |

### xPass calibration (after design)

| bin | obs_EVENT | obs_EVENT+IMP | obs_EVENT+ORACLE | obs_EVENT-nodur | pred_EVENT | pred_EVENT+IMP | pred_EVENT+ORACLE | pred_EVENT-nodur |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.143 | 0.144 | 0.139 | 0.172 | 0.140 | 0.146 | 0.132 | 0.168 |
| 1 | 0.468 | 0.466 | 0.459 | 0.468 | 0.460 | 0.456 | 0.454 | 0.463 |
| 2 | 0.777 | 0.778 | 0.773 | 0.767 | 0.796 | 0.785 | 0.791 | 0.782 |
| 3 | 0.923 | 0.920 | 0.929 | 0.915 | 0.932 | 0.930 | 0.936 | 0.922 |
| 4 | 0.971 | 0.974 | 0.973 | 0.965 | 0.970 | 0.972 | 0.976 | 0.965 |
| 5 | 0.988 | 0.987 | 0.992 | 0.984 | 0.985 | 0.987 | 0.989 | 0.982 |
| 6 | 0.994 | 0.994 | 0.996 | 0.993 | 0.991 | 0.993 | 0.994 | 0.990 |
| 7 | 0.997 | 0.997 | 0.998 | 0.997 | 0.995 | 0.996 | 0.996 | 0.994 |
| 8 | 0.999 | 0.999 | 0.999 | 0.998 | 0.997 | 0.998 | 0.998 | 0.996 |
| 9 | 1.000 | 1.000 | 1.000 | 0.999 | 0.998 | 0.999 | 0.999 | 0.998 |

### xPass slices (after design; log-loss per variant, delta vs EVENT with per-pass CI)


**play_pattern**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| counter | 916 | 797 | 0.2662 | 0.2564 | 0.2572 | +0.0098 [+0.0019, +0.0181] | +0.0090 [-0.0028, +0.0219] |
| other | 5130 | 4177 | 0.2297 | 0.2258 | 0.2172 | +0.0039 [+0.0004, +0.0077] | +0.0124 [+0.0078, +0.0175] |
| regular | 65462 | 56267 | 0.1819 | 0.1821 | 0.1734 | -0.0002 [-0.0011, +0.0006] | +0.0085 [+0.0073, +0.0098] |
| set_piece | 78083 | 62305 | 0.2294 | 0.2286 | 0.2222 | +0.0008 [-0.0000, +0.0017] | +0.0072 [+0.0062, +0.0083] |

**length_band**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| 0-10 | 31683 | 25008 | 0.2138 | 0.2132 | 0.1977 | +0.0005 [-0.0009, +0.0016] | +0.0161 [+0.0140, +0.0181] |
| 10-20 | 62625 | 56728 | 0.1633 | 0.1626 | 0.1551 | +0.0007 [+0.0001, +0.0014] | +0.0082 [+0.0070, +0.0095] |
| 20-35 | 39443 | 33193 | 0.1954 | 0.1951 | 0.1911 | +0.0003 [-0.0007, +0.0013] | +0.0043 [+0.0030, +0.0057] |
| 35+ | 15840 | 8617 | 0.4125 | 0.4123 | 0.4126 | +0.0002 [-0.0019, +0.0024] | -0.0000 [-0.0025, +0.0022] |

**height**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| Ground Pass | 111212 | 102898 | 0.1344 | 0.1339 | 0.1263 | +0.0006 [+0.0000, +0.0011] | +0.0081 [+0.0073, +0.0090] |
| High Pass | 23403 | 10634 | 0.4935 | 0.4927 | 0.4890 | +0.0008 [-0.0015, +0.0028] | +0.0046 [+0.0024, +0.0069] |
| Low Pass | 14976 | 10014 | 0.3169 | 0.3170 | 0.3048 | -0.0001 [-0.0022, +0.0021] | +0.0122 [+0.0095, +0.0151] |

**reliable_frame**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| reliable | 112945 | 92974 | 0.2193 | 0.2185 | 0.2104 | +0.0008 [+0.0001, +0.0014] | +0.0089 [+0.0078, +0.0098] |
| unreliable | 36646 | 30572 | 0.1768 | 0.1770 | 0.1716 | -0.0002 [-0.0012, +0.0009] | +0.0052 [+0.0038, +0.0067] |

**under_pressure**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| pressed | 22468 | 16529 | 0.2800 | 0.2792 | 0.2671 | +0.0008 [-0.0008, +0.0022] | +0.0129 [+0.0103, +0.0150] |
| unpressed | 127123 | 107017 | 0.1963 | 0.1958 | 0.1892 | +0.0005 [-0.0002, +0.0011] | +0.0071 [+0.0062, +0.0079] |

**gender**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE | d(EVENT+IMP) | d(EVENT+ORACLE) |
|---|---|---|---|---|---|---|---|
| female | 39906 | 30268 | 0.2671 | 0.2671 | 0.2591 | +0.0000 [-0.0011, +0.0012] | +0.0080 [+0.0061, +0.0098] |
| male | 109685 | 93278 | 0.1877 | 0.1870 | 0.1797 | +0.0007 [+0.0001, +0.0013] | +0.0080 [+0.0070, +0.0088] |

## (c) Cross-gender robustness of the xG gain

Train on one gender's shots, test on the other's. Every imputed feature comes from students trained on the training gender only: out-of-fold within the source gender (stage `students`, 5-fold by match, train cap 150,000) for the training shots and the stage-02 transfer fits (`imputation_cache/transfer__<dir>__*`, trained on every source-gender row) for the test shots. `(in-domain CV)` rows are the main 5-fold predictions on the same test shots (students and xG trained on both genders' other matches) for comparison; BASE (target rate) is an oracle base rate.

| direction | variant | n | positives | log_loss | brier | auc | ece | mean_pred |
|---|---|---|---|---|---|---|---|---|
| m2f | BASE (source rate) | 3231 | 302 | 0.3113 | 0.0849 | 0.5000 | 0.0210 | 0.1053 |
| m2f | BASE (target rate) | 3231 | 302 | 0.3105 | 0.0847 | 0.5000 | 0.0187 | 0.0935 |
| m2f | EVENT | 3231 | 302 | 0.2634 | 0.0742 | 0.7779 | 0.0114 | 0.1029 |
| m2f | EVENT+IMP | 3231 | 302 | 0.2638 | 0.0742 | 0.7743 | 0.0128 | 0.1021 |
| m2f | EVENT+ORACLE360 | 3231 | 302 | 0.2594 | 0.0729 | 0.7851 | 0.0120 | 0.1007 |
| m2f | EVENT+ORACLESHOT | 3231 | 302 | 0.2542 | 0.0713 | 0.7939 | 0.0054 | 0.0936 |
| m2f | STATSBOMB_XG | 3231 | 302 | 0.2483 | 0.0691 | 0.8100 | 0.0133 | 0.0945 |
| m2f | EVENT (in-domain CV) | 3231 | 302 | 0.2610 | 0.0735 | 0.7826 | 0.0094 | 0.0969 |
| m2f | EVENT+IMP (in-domain CV) | 3231 | 302 | 0.2609 | 0.0735 | 0.7832 | 0.0099 | 0.0970 |
| m2f | EVENT+ORACLE360 (in-domain CV) | 3231 | 302 | 0.2582 | 0.0726 | 0.7884 | 0.0084 | 0.0963 |
| f2m | BASE (source rate) | 7002 | 737 | 0.3373 | 0.0943 | 0.5000 | 0.0160 | 0.0935 |
| f2m | BASE (target rate) | 7002 | 737 | 0.3365 | 0.0942 | 0.5000 | 0.0131 | 0.1053 |
| f2m | EVENT | 7002 | 737 | 0.2895 | 0.0833 | 0.7737 | 0.0244 | 0.0881 |
| f2m | EVENT+IMP | 7002 | 737 | 0.2901 | 0.0834 | 0.7714 | 0.0249 | 0.0900 |
| f2m | EVENT+ORACLE360 | 7002 | 737 | 0.2875 | 0.0825 | 0.7755 | 0.0223 | 0.0901 |
| f2m | EVENT+ORACLESHOT | 7002 | 737 | 0.2790 | 0.0801 | 0.7900 | 0.0147 | 0.0933 |
| f2m | STATSBOMB_XG | 7002 | 737 | 0.2679 | 0.0765 | 0.8086 | 0.0086 | 0.1012 |
| f2m | EVENT (in-domain CV) | 7002 | 737 | 0.2807 | 0.0805 | 0.7855 | 0.0075 | 0.1010 |
| f2m | EVENT+IMP (in-domain CV) | 7002 | 737 | 0.2813 | 0.0807 | 0.7845 | 0.0099 | 0.1011 |
| f2m | EVENT+ORACLE360 (in-domain CV) | 7002 | 737 | 0.2777 | 0.0795 | 0.7892 | 0.0078 | 0.1010 |

Paired deltas vs EVENT (transferred):

| direction | variant | n | delta_log_loss [95% CI] | clustered CI | delta_brier [95% CI] | significant |
|---|---|---|---|---|---|---|
| m2f | BASE (source rate) | 3231 | -0.0479 [-0.0586, -0.0365] | [-0.0589, -0.0369] | -0.01063 [-0.01397, -0.00723] | True |
| m2f | BASE (target rate) | 3231 | -0.0471 [-0.0583, -0.0352] | [-0.0587, -0.0357] | -0.01050 [-0.01396, -0.00693] | True |
| m2f | EVENT+IMP | 3231 | -0.0004 [-0.0021, +0.0013] | [-0.0021, +0.0013] | +0.00005 [-0.00051, +0.00062] | False |
| m2f | EVENT+ORACLE360 | 3231 | +0.0040 [+0.0014, +0.0066] | [+0.0016, +0.0066] | +0.00133 [+0.00044, +0.00225] | True |
| m2f | EVENT+ORACLESHOT | 3231 | +0.0092 [+0.0052, +0.0134] | [+0.0046, +0.0135] | +0.00292 [+0.00161, +0.00428] | True |
| m2f | STATSBOMB_XG | 3231 | +0.0151 [+0.0082, +0.0221] | [+0.0072, +0.0225] | +0.00511 [+0.00303, +0.00726] | True |
| m2f | EVENT (in-domain CV) | 3231 | +0.0024 [-0.0002, +0.0051] | [+0.0000, +0.0048] | +0.00072 [-0.00015, +0.00161] | False |
| m2f | EVENT+IMP (in-domain CV) | 3231 | +0.0025 [-0.0003, +0.0054] | [+0.0000, +0.0052] | +0.00074 [-0.00026, +0.00172] | False |
| m2f | EVENT+ORACLE360 (in-domain CV) | 3231 | +0.0052 [+0.0018, +0.0083] | [+0.0022, +0.0081] | +0.00160 [+0.00048, +0.00265] | True |
| f2m | BASE (source rate) | 7002 | -0.0478 [-0.0547, -0.0412] | [-0.0546, -0.0412] | -0.01103 [-0.01293, -0.00928] | True |
| f2m | BASE (target rate) | 7002 | -0.0470 [-0.0535, -0.0407] | [-0.0533, -0.0407] | -0.01089 [-0.01268, -0.00919] | True |
| f2m | EVENT+IMP | 7002 | -0.0006 [-0.0017, +0.0004] | [-0.0017, +0.0004] | -0.00013 [-0.00041, +0.00015] | False |
| f2m | EVENT+ORACLE360 | 7002 | +0.0019 [-0.0003, +0.0040] | [+0.0000, +0.0039] | +0.00075 [+0.00016, +0.00132] | False |
| f2m | EVENT+ORACLESHOT | 7002 | +0.0104 [+0.0070, +0.0138] | [+0.0069, +0.0140] | +0.00317 [+0.00209, +0.00430] | True |
| f2m | STATSBOMB_XG | 7002 | +0.0216 [+0.0161, +0.0272] | [+0.0161, +0.0271] | +0.00676 [+0.00497, +0.00859] | True |
| f2m | EVENT (in-domain CV) | 7002 | +0.0088 [+0.0053, +0.0123] | [+0.0055, +0.0125] | +0.00275 [+0.00166, +0.00390] | True |
| f2m | EVENT+IMP (in-domain CV) | 7002 | +0.0082 [+0.0048, +0.0117] | [+0.0048, +0.0118] | +0.00256 [+0.00148, +0.00367] | True |
| f2m | EVENT+ORACLE360 (in-domain CV) | 7002 | +0.0117 [+0.0078, +0.0157] | [+0.0080, +0.0160] | +0.00382 [+0.00255, +0.00517] | True |

## (d) Where does the imputed state help? xG log-loss by slice

Per-slice log-loss of each variant and paired delta vs EVENT (positive = better; per-shot bootstrap CI, 1000 resamples). Slices with fewer than 50 shots are omitted.


**play_pattern**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE360 | EVENT+ORACLESHOT | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLE360) | d(EVENT+ORACLESHOT) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| counter | 477 | 65 | 0.3316 | 0.3358 | 0.3239 | 0.3173 | 0.3157 | -0.0042 [-0.0090, +0.0010] | +0.0077 [-0.0011, +0.0177] | +0.0143 [-0.0007, +0.0293] | +0.0159 [-0.0038, +0.0377] |
| other | 221 | 35 | 0.4005 | 0.4009 | 0.3966 | 0.3919 | 0.4129 | -0.0004 [-0.0057, +0.0049] | +0.0038 [-0.0059, +0.0136] | +0.0086 [-0.0072, +0.0247] | -0.0124 [-0.0502, +0.0246] |
| regular | 3527 | 375 | 0.2854 | 0.2853 | 0.2832 | 0.2782 | 0.2693 | +0.0001 [-0.0017, +0.0020] | +0.0022 [-0.0005, +0.0049] | +0.0072 [+0.0036, +0.0111] | +0.0161 [+0.0104, +0.0223] |
| set_piece | 6008 | 564 | 0.2589 | 0.2592 | 0.2560 | 0.2512 | 0.2474 | -0.0003 [-0.0016, +0.0011] | +0.0029 [+0.0011, +0.0048] | +0.0077 [+0.0046, +0.0106] | +0.0115 [+0.0071, +0.0167] |

**distance_band**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE360 | EVENT+ORACLESHOT | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLE360) | d(EVENT+ORACLESHOT) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-8 | 988 | 291 | 0.5379 | 0.5382 | 0.5295 | 0.5159 | 0.4826 | -0.0003 [-0.0044, +0.0041] | +0.0084 [+0.0017, +0.0150] | +0.0220 [+0.0106, +0.0332] | +0.0553 [+0.0383, +0.0727] |
| 8-12 | 1724 | 248 | 0.3700 | 0.3707 | 0.3615 | 0.3573 | 0.3469 | -0.0007 [-0.0041, +0.0026] | +0.0086 [+0.0037, +0.0133] | +0.0127 [+0.0053, +0.0202] | +0.0231 [+0.0123, +0.0345] |
| 12-18 | 2631 | 308 | 0.3255 | 0.3256 | 0.3236 | 0.3156 | 0.3174 | -0.0001 [-0.0025, +0.0025] | +0.0019 [-0.0022, +0.0057] | +0.0099 [+0.0043, +0.0163] | +0.0081 [-0.0007, +0.0168] |
| 18-25 | 2734 | 133 | 0.1889 | 0.1892 | 0.1878 | 0.1861 | 0.1828 | -0.0002 [-0.0012, +0.0008] | +0.0011 [-0.0003, +0.0026] | +0.0028 [+0.0007, +0.0051] | +0.0062 [+0.0020, +0.0103] |
| 25+ | 2156 | 59 | 0.1236 | 0.1240 | 0.1242 | 0.1223 | 0.1244 | -0.0005 [-0.0014, +0.0005] | -0.0006 [-0.0015, +0.0003] | +0.0013 [-0.0005, +0.0034] | -0.0009 [-0.0080, +0.0055] |

**reliable_frame**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE360 | EVENT+ORACLESHOT | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLE360) | d(EVENT+ORACLESHOT) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reliable | 9183 | 836 | 0.2585 | 0.2590 | 0.2560 | 0.2520 | 0.2471 | -0.0005 [-0.0015, +0.0005] | +0.0025 [+0.0010, +0.0039] | +0.0064 [+0.0041, +0.0088] | +0.0113 [+0.0075, +0.0150] |
| unreliable | 1050 | 203 | 0.4144 | 0.4131 | 0.4077 | 0.3943 | 0.3890 | +0.0014 [-0.0033, +0.0064] | +0.0067 [-0.0010, +0.0151] | +0.0202 [+0.0086, +0.0322] | +0.0254 [+0.0063, +0.0452] |

**has_assist**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE360 | EVENT+ORACLESHOT | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLE360) | d(EVENT+ORACLESHOT) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assisted | 6715 | 694 | 0.2835 | 0.2848 | 0.2809 | 0.2761 | 0.2708 | -0.0013 [-0.0024, -0.0001] | +0.0026 [+0.0008, +0.0045] | +0.0074 [+0.0047, +0.0103] | +0.0127 [+0.0083, +0.0171] |
| unassisted | 3518 | 345 | 0.2572 | 0.2558 | 0.2539 | 0.2486 | 0.2443 | +0.0014 [-0.0005, +0.0033] | +0.0034 [+0.0005, +0.0060] | +0.0086 [+0.0046, +0.0128] | +0.0129 [+0.0055, +0.0201] |

**gender**

| level | n | positives | EVENT | EVENT+IMP | EVENT+ORACLE360 | EVENT+ORACLESHOT | STATSBOMB_XG | d(EVENT+IMP) | d(EVENT+ORACLE360) | d(EVENT+ORACLESHOT) | d(STATSBOMB_XG) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| female | 3231 | 302 | 0.2610 | 0.2609 | 0.2582 | 0.2514 | 0.2483 | +0.0002 [-0.0016, +0.0021] | +0.0028 [+0.0002, +0.0055] | +0.0097 [+0.0057, +0.0137] | +0.0127 [+0.0057, +0.0196] |
| male | 7002 | 737 | 0.2807 | 0.2813 | 0.2777 | 0.2737 | 0.2679 | -0.0006 [-0.0018, +0.0005] | +0.0029 [+0.0009, +0.0049] | +0.0070 [+0.0041, +0.0096] | +0.0128 [+0.0086, +0.0173] |

## Fits

| variant | rounds_median | seconds | fits |
|---|---|---|---|
| EVENT | 118.0 | 10.8 | 15 |
| EVENT+IMP | 102.0 | 9.9 | 15 |
| EVENT+IMP(E0) | 115.0 | 11.7 | 15 |
| EVENT+ORACLE360 | 117.0 | 10.4 | 15 |
| EVENT+ORACLESHOT | 132.0 | 10.7 | 15 |
| EVENT+ORACLESHOT_full | 126.0 | 12.7 | 15 |
| LOC | 101.0 | 4.6 | 15 |

| variant | design | rounds_median | seconds | fits |
|---|---|---|---|---|
| EVENT | after | 505.0 | 36.2 | 5 |
| EVENT | noafter | 441.0 | 30.0 | 5 |
| EVENT+IMP | after | 385.0 | 30.8 | 5 |
| EVENT+IMP | noafter | 439.0 | 31.9 | 5 |
| EVENT+ORACLE | after | 559.0 | 41.9 | 5 |
| EVENT+ORACLE | noafter | 335.0 | 28.3 | 5 |
| EVENT-nodur | after | 501.0 | 34.6 | 5 |

Stage wall time (s): report 62

## Caveats

- The imputed shot state is out-of-fold with respect to the xG folds, but the students that produced the training-fold imputations were fitted on the other four folds including the test fold's 360 labels (not its goal labels): the usual second-order coupling of stacked out-of-fold features, shared with every phase-1 stage.
- EVENT already contains every input the students read (the stage-02 E2 design), so EVENT+IMP can only add what the 360 supervision taught the students beyond what ~10k goal labels teach the xG model directly; the learning curve tests that at smaller label budgets.
- `f_under_pressure` is StatsBomb's flag for a Pressure event overlapping the shot; for shots it is contemporaneous with the shot instant and a standard xG input, but not strictly pre-instant.
- The 360 oracle follows the stage-02 label rules: `nearest_opp_dist_in_cone` is NaN when the cone is empty (informative missingness), `block_depth` is NaN on unreliable frames, keeper distance is NaN when the keeper is not visible or inconsistently flagged. The shot-frame oracle has no block depth; `sff_n_opp_ahead_of_ball` stands in. Both oracles are visible-area truncations, not tracking.
- StatsBomb xG is a reference trained by the vendor on far more shots (possibly including these matches) with the shot freeze frame; treat it as an upper reference, not a fair competitor.
- ~10k shots / ~1.1k goals: single-model log-loss differences of 0.002-0.005 are within noise; read the CIs. Slice CIs are wider still.
- xPass: the subsample is match-stratified (every match keeps the same expected share); the with-after design conditions on the realised end location, which for incomplete passes is where the ball was won. The stage-02 E2a students also read the realised pass duration, so the with-after EVENT design keeps `f_after_duration` (otherwise the imputations smuggle it in: with duration removed from EVENT only, EVENT+IMP beat even the 360 oracle on short passes); `EVENT-nodur` shows what duration alone is worth. In the no-after design the oracle's lane / end-location quantities are defined relative to the realised end, so the no-after ORACLE gap is not an attainable target for a pre-instant model; the E2 students never see the end and are comparable.
- Transfer students use a smaller training cap than the stage-02 CV students and single fits; their in-domain skill is slightly lower, so the transfer gap mixes domain shift with a small capacity effect.
- Stage-02 open issues respected: imputed counts / distances clipped at 0 here; `deep_block` / `counter_on` and the NFL highlight check are not used by this stage.
