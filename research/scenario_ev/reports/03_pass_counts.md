# Scenario +EV 03 - player pass counts: does the per-pass gain move a count line?

Machine-written by `research/scenario_ev/pass_counts.py`. Every number below is read from a committed `research/scenario_ev/reports/03_pass_*.parquet` table. Discovery numbers are exploratory and multiplicity-inflated; confirmation numbers are scored once and are the headline.

## Headline

The programme's one positive imputed-state result was pass difficulty. This stage asked whether it converts into a player pass-count prop. It does not, and the chain breaks in three independent places.

**1. The per-pass gain itself is mostly an artefact of the subsample it was measured on.** Refitting the programme's own xPass variants -- same stage-02 folds, same LightGBM settings, same pre-instant `E2` design, same out-of-fold receiver student -- on *all* 371,844 pass attempts of the 417 360-matches instead of its 149,591-pass subsample, and then scoring on exactly the 149,591 rows the programme scored, `EVENT+IMP` beats `EVENT` by +0.00017 [-0.00031, +0.00066] nats per pass against the programme's published +0.00162 [+0.00104, +0.00220] on the same rows. The 360 oracle gap reproduces almost exactly (+0.0703 here against +0.0703 published), so the pipeline is sound: the event-only model simply catches up once it is given the passes the subsample threw away. The sequence student's channel is the one that survives -- on its 251 matches `EVENT+IMP+SEQ` is still worth +0.0019 [+0.0013, +0.0024] nats per pass at full training size.

**2. Aggregating to a count destroys about 90% of whatever per-pass skill exists.** With the realised attempts given, the *perfect* 360 frame is worth 0.0703 nats per pass, which over the 29.6 attempts of an average player-match sums to 2.079 nats; the completed-pass distribution actually improves by 0.1970 nats, a retention of 9.5%. A count throws away *which* passes failed, and that is where the information is. Downstream of a 9.5% retention rate, a per-pass gain of 0.002 nats is worth ~0.006 nats on a player-match count -- on a distribution whose own entropy is 1.98 nats.

**3. At a plausible half-line the imputed state moves the price a lot and improves it not at all.** At a mean line of 23.0 completed passes (confirmation, n=5,038, over rate 0.530), `EVENT+IMP` shifts P(over) by 4.8 percentage points on average (max 34 pp) and changes Brier by -0.00074 [-0.00235, +0.00089]. The sequence variant shifts 5.9 pp for +0.00219 [-0.00081, +0.00496]. Only the true 360 frame both moves (17.5 pp) and improves (+0.05194 [+0.04686, +0.05722]). A 5-point move at an even-money line is commercially enormous *if it is information*; here it is noise.

### The ceiling: a completed-pass prop is a volume market

Of the variance of completed passes in the confirmation half (sd 17.7 passes, n=13,053), only **2.2%** is the trial-by-trial Bernoulli term that a per-pass completion model can touch at all; 97.8% is attempt volume and the player-match rate level. The end-to-end decomposition agrees: swapping the attempts channel from the rolling-mean proxy to a model is worth +0.0724 [+0.0627, +0.0831] nats per player-match, swapping the completion-rate channel is worth +0.0213 [+0.0182, +0.0246], and both together +0.0771 [+0.0657, +0.0897]. At the margin the rate channel adds +0.0047 nats (6% of the joint gain) and the volume channel adds +0.0557 (72%). **Volume is the market.** And the rate channel that does exist is bought with ordinary rolling and opponent features, not with tracking: adding the imputed block depth to the attempts model is worth +0.00081 [-0.00091, +0.00251] nats, against a refit-noise spread of 0.00149 over 3 seeds. The tracking-derived work is irrelevant to this market and the stage should say so plainly.

### The gate

Five scenarios were pre-registered on discovery, including the one the earlier discussion proposed (the team's usual deep-lying passer is absent) and the opponent sitting deep. On the confirmation scenario rows the specialist is **worse** than the generalist in all five, by -0.0103 to -0.0054 nats. Targeting has no room for this target: a model fitted on all rows already prices the scenario rows better than one fitted only on them.

### On expected value

Against a market priced off the rolling-mean proxy the model shows +25.0% ROI at 6% hold ([22.7%, 27.4%], 9,815 bets on confirmation). That is a measure of how bad the proxy is, not of an edge, and two calibrations say so:

- *In line units.* Our model's mean differs from the proxy's by 3.84 completed passes on average, 21.8% of the outcome's sd, and it would post a different line from the proxy in 90% of player-matches. A real bookmaker's implied mean total goals differs from the identical proxy by 0.200 goals, 12.3% of that outcome's sd, differing by half a line in 6.3% of matches. We are claiming to disagree with a rolling mean roughly 1.8x as hard as a real book does.
- *Against a book that sets its own line.* Giving the opponent a mean 0.565 of the way from the proxy to our model -- the factor that matches the goals-market disagreement above -- and letting it post its own line, the model's ROI at 6% hold falls to +9.0% ([6.4%, 11.8%], 8,024 bets).

That residual is not zero, and honesty requires saying so. But it rests entirely on an assumption the stage cannot test: that a real pass-prop book knows only 56% of what our model knows. The feature ablation makes that implausible. All of the model's advantage over the proxy comes from the player's own rolling block, his position, and above all the team / opponent rolling context (+0.0293 nats after player and position, +0.0630 after team and opponent, +0.0638 after the imputed state) -- public averages that every trading desk computes. There is no private information in this model. **The stage's conclusion is a null result: no tradeable edge is demonstrated, and none of the apparent one is attributable to the tracking-derived channel this programme was about.**

## Protocol

- **Discovery / confirmation.** Fixed in code before any modelling (`common.chronological_split`, 60% of matches by date), taken separately inside each population being scored: a single split over all 2,090 StatsBomb matches would put every club season in discovery and only international tournaments in confirmation, which is a regime change, not a held-out sample. Every threshold, cut, feature choice and bet rule below is chosen on discovery.
- **Populations.** Part (a): the 417 StatsBomb 360 matches, every pass attempt. Parts (b) and (c): the four league seasons StatsBomb covers end to end (Premier League, La Liga, Serie A, Ligue 1, 2015/2016), starters with at least 3 strictly prior appearances and both teams with at least 3 prior matches. The population is **not** filtered on realised minutes: minutes are an outcome and an early substitution is exactly the risk a prop prices.
- **What counts as known at pricing time.** Whether the player starts, his starting position, and the two teams -- the facts a book has when it posts a player prop after team news. Nothing else from the match being predicted enters any feature.
- **Hygiene.** Rolling features come from strictly earlier matches, verified by brute force rather than asserted; folds grouped by match; fixed seeds; match-clustered bootstrap intervals (2000 resamples) on every model-vs-model and ROI claim; a refit-noise floor over 3 seeds.

### Strictly-prior audit

Prior sums and counts recomputed by direct filtering for a random sample of rows; the shrunk rate reconstructed from them for every row; and the opponent join checked to be the opponent's own prior value. A non-zero error here means leakage -- an earlier version of the last-5 feature had one, and this check is what caught it.

| quantity | n_checked | max_abs_sum_error | max_abs_count_error |
|---|---|---|---|
| passes_raw | 120 | 0 | 0 |
| minutes_raw | 120 | 9.09495e-13 | 0 |
| p_att_last5_bruteforce | 150 | 0 | 0 |
| p_att_per90_reconstruction | 59532 | 0 | 0 |
| team_possession_prior | 100 | 3.55271e-15 | 0 |
| opponent_join_consistency | 29081 | 0 | 0 |

## (a) The completion channel

### (a.1) Per-pass models refitted on every attempt

The programme's cache holds a match-stratified 149,591-pass subsample, about 40% of the 371,844 attempts, so its player-match counts would be ~40% of full size and a count line cannot be studied on it. The same models are refitted here on every attempt. Variants: `EVENT` = event-only pre-instant design; `EVENT+IMP` = plus the seven out-of-fold LightGBM student predictions of the 360 pass state; `EVENT+ORACLE` = plus the same quantities measured on the 360 frame; `EVENT+IMP+SEQ` = plus the soccer-06 sequence student's seven quantities, on the 251 matches it covers. `source = programme_cache` rows are the published models scored on their own subsample.

| population | source | variant | n | log_loss | delta |
|---|---|---|---|---|---|
| all_attempts | refit_all_attempts | BASE | 371844 | 0.4623 | -0.11793 [-0.12052, -0.11545] |
| all_attempts | refit_all_attempts | EVENT | 371844 | 0.3444 | +0.00000 [+0.00000, +0.00000] |
| all_attempts | refit_all_attempts | EVENT+IMP | 371844 | 0.3443 | +0.00001 [-0.00032, +0.00034] |
| all_attempts | refit_all_attempts | EVENT+ORACLE | 371844 | 0.2740 | +0.07031 [+0.06879, +0.07180] |
| cached_subsample_rows | refit_all_attempts | BASE | 149591 | 0.4624 | -0.11846 [-0.12165, -0.11535] |
| cached_subsample_rows | refit_all_attempts | EVENT | 149591 | 0.3439 | +0.00000 [+0.00000, +0.00000] |
| cached_subsample_rows | refit_all_attempts | EVENT+IMP | 149591 | 0.3437 | +0.00017 [-0.00031, +0.00066] |
| cached_subsample_rows | refit_all_attempts | EVENT+ORACLE | 149591 | 0.2738 | +0.07010 [+0.06803, +0.07236] |
| cached_subsample_rows | programme_cache | EVENT | 149591 | 0.3486 | +0.00000 [+0.00000, +0.00000] |
| cached_subsample_rows | programme_cache | EVENT+IMP | 149591 | 0.3470 | +0.00162 [+0.00104, +0.00220] |
| cached_subsample_rows | programme_cache | EVENT+ORACLE | 149591 | 0.2783 | +0.07032 [+0.06829, +0.07247] |
| seq_matches | refit_all_attempts | EVENT | 210538 | 0.3342 | +0.00000 [+0.00000, +0.00000] |
| seq_matches | refit_all_attempts | EVENT+IMP | 210538 | 0.3334 | +0.00082 [+0.00038, +0.00127] |
| seq_matches | refit_all_attempts | EVENT+IMP+SEQ | 210538 | 0.3323 | +0.00188 [+0.00134, +0.00238] |

The `cached_subsample_rows` block is the comparison that matters: on **the same 149,591 evaluation rows**, the programme's models give `EVENT+IMP` a significant gain and the refitted models give it nothing. The only thing that changed is how many passes the event-only model saw in training. The oracle gap is unchanged, so the 360 frame still carries real information about a single pass; what evaporates is the *imputed* version's marginal value once the event-only model has enough data to learn the same thing from the events directly.

### (a.2) Player-match completed-pass distributions

Given the realised attempts, completed passes are a sum of independent Bernoulli trials with unequal probabilities, so the exact predictive distribution is a Poisson binomial. `count_log_score` is the mean negative log probability assigned to the realised count; deltas are paired and match-clustered. **Caveat:** conditioning on realised attempts isolates the completion channel and is not tradeable -- a real prop must predict attempts too, which is what part (b) does.

| population | where | variant | n | n_matches | mean_attempts | mean_completed | count_log_score | mean_pred | mean_sd | mae | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all_matches | discovery | BASE | 7540 | 250 | 30.2328 | 25.3231 | 2.8612 | 24.9678 | 1.9250 | 2.7546 | -0.9022 [-0.9477, -0.8511] |
| all_matches | discovery | EVENT | 7540 | 250 | 30.2328 | 25.3231 | 1.9590 | 25.3575 | 1.6333 | 1.5342 | +0.0000 [+0.0000, +0.0000] |
| all_matches | discovery | EVENT+IMP | 7540 | 250 | 30.2328 | 25.3231 | 1.9612 | 25.3569 | 1.6279 | 1.5343 | -0.0022 [-0.0065, +0.0020] |
| all_matches | discovery | EVENT+ORACLE | 7540 | 250 | 30.2328 | 25.3231 | 1.7728 | 25.3571 | 1.4607 | 1.2762 | +0.1862 [+0.1714, +0.2018] |
| all_matches | confirmation | BASE | 5038 | 167 | 28.5607 | 23.0623 | 3.0098 | 23.5929 | 1.8628 | 2.8620 | -0.9894 [-1.0596, -0.9205] |
| all_matches | confirmation | EVENT | 5038 | 167 | 28.5607 | 23.0623 | 2.0204 | 22.9864 | 1.6945 | 1.6219 | +0.0000 [+0.0000, +0.0000] |
| all_matches | confirmation | EVENT+IMP | 5038 | 167 | 28.5607 | 23.0623 | 2.0240 | 22.9773 | 1.6932 | 1.6243 | -0.0037 [-0.0085, +0.0011] |
| all_matches | confirmation | EVENT+ORACLE | 5038 | 167 | 28.5607 | 23.0623 | 1.8071 | 23.0166 | 1.5140 | 1.3319 | +0.2132 [+0.1931, +0.2336] |
| all_matches | all | BASE | 12578 | 417 | 29.5630 | 24.4176 | 2.9207 | 24.4171 | 1.9001 | 2.7976 | -0.9371 [-0.9763, -0.8977] |
| all_matches | all | EVENT | 12578 | 417 | 29.5630 | 24.4176 | 1.9836 | 24.4078 | 1.6579 | 1.5693 | +0.0000 [+0.0000, +0.0000] |
| all_matches | all | EVENT+IMP | 12578 | 417 | 29.5630 | 24.4176 | 1.9863 | 24.4038 | 1.6541 | 1.5704 | -0.0028 [-0.0060, +0.0004] |
| all_matches | all | EVENT+ORACLE | 12578 | 417 | 29.5630 | 24.4176 | 1.7865 | 24.4196 | 1.4821 | 1.2985 | +0.1970 [+0.1851, +0.2091] |
| seq_matches | discovery | EVENT | 4538 | 151 | 28.7232 | 24.8938 | 1.9298 | 24.9034 | 1.5505 | 1.4876 | +0.0000 [+0.0000, +0.0000] |
| seq_matches | discovery | EVENT+IMP | 4538 | 151 | 28.7232 | 24.8938 | 1.9258 | 24.9080 | 1.5417 | 1.4781 | +0.0040 [-0.0018, +0.0098] |
| seq_matches | discovery | EVENT+IMP+SEQ | 4538 | 151 | 28.7232 | 24.8938 | 1.9269 | 24.9137 | 1.5414 | 1.4789 | +0.0029 [-0.0039, +0.0098] |
| seq_matches | confirmation | EVENT | 3011 | 100 | 26.6330 | 22.4082 | 1.9835 | 22.3548 | 1.6030 | 1.5750 | +0.0000 [+0.0000, +0.0000] |
| seq_matches | confirmation | EVENT+IMP | 3011 | 100 | 26.6330 | 22.4082 | 1.9808 | 22.3365 | 1.5968 | 1.5676 | +0.0027 [-0.0052, +0.0104] |
| seq_matches | confirmation | EVENT+IMP+SEQ | 3011 | 100 | 26.6330 | 22.4082 | 1.9789 | 22.3334 | 1.5979 | 1.5635 | +0.0046 [-0.0045, +0.0136] |
| seq_matches | all | EVENT | 7549 | 251 | 27.8895 | 23.9024 | 1.9512 | 23.8869 | 1.5714 | 1.5225 | +0.0000 [+0.0000, +0.0000] |
| seq_matches | all | EVENT+IMP | 7549 | 251 | 27.8895 | 23.9024 | 1.9477 | 23.8823 | 1.5637 | 1.5138 | +0.0035 [-0.0013, +0.0081] |
| seq_matches | all | EVENT+IMP+SEQ | 7549 | 251 | 27.8895 | 23.9024 | 1.9476 | 23.8845 | 1.5639 | 1.5127 | +0.0036 [-0.0018, +0.0093] |

The predicted spread is honest but slightly tight: the Poisson binomial's mean sd is 1.66 completions while the mean absolute error is 1.57, so passes inside a match are mildly positively dependent and independence understates the tails. That matters little here because no imputed variant's count gain is significant: `EVENT+IMP` -0.0037 [-0.0085, +0.0011], `EVENT+IMP+SEQ` +0.0046 [-0.0045, +0.0136], against the oracle's +0.2132 [+0.1931, +0.2336].

### (a.3) Retention: per-pass nats in, count nats out

| population | variant | mean_attempts | per_pass_nats | summed_over_attempts | count_nats | retention |
|---|---|---|---|---|---|---|
| all_matches | EVENT+IMP | 29.56305 | 0.00001 | 0.00038 | -0.00277 | -7.29483 |
| all_matches | EVENT+ORACLE | 29.56305 | 0.07031 | 2.07872 | 0.19703 | 0.09478 |
| seq_matches | EVENT+IMP | 27.88952 | 0.00082 | 0.02283 | 0.00348 | 0.15260 |
| seq_matches | EVENT+IMP+SEQ | 27.88952 | 0.00188 | 0.05233 | 0.00356 | 0.06794 |

### (a.4) Half-lines around the model's own mean

Lines are `floor(EVENT's mean) + 0.5 + offset`, so offset 0 is at the money and +/-6 probe the tails. `mean_abs_shift_vs_EVENT` is how far the variant moves P(over) from `EVENT`'s price; `delta_brier_vs_EVENT` is whether the move was an improvement.

| population | offset | variant | n | mean_line | over_rate | mean_p | brier | delta | mean_abs_shift_vs_EVENT | max_abs_shift_vs_EVENT |
|---|---|---|---|---|---|---|---|---|---|---|
| all_matches | -6 | BASE | 5038 | 17.6334 | 0.9748 | 0.9468 | 0.0473 | -0.02981 [-0.03520, -0.02450] | 0.0587 | 0.9787 |
| all_matches | -6 | EVENT | 5038 | 17.6334 | 0.9748 | 0.9835 | 0.0175 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| all_matches | -6 | EVENT+IMP | 5038 | 17.6334 | 0.9748 | 0.9832 | 0.0174 | +0.00010 [-0.00017, +0.00037] | 0.0032 | 0.1689 |
| all_matches | -6 | EVENT+ORACLE | 5038 | 17.6334 | 0.9748 | 0.9816 | 0.0153 | +0.00221 [+0.00130, +0.00311] | 0.0108 | 0.6071 |
| all_matches | -3 | BASE | 5038 | 20.1624 | 0.9131 | 0.8771 | 0.1388 | -0.06665 [-0.07522, -0.05833] | 0.1289 | 0.9628 |
| all_matches | -3 | EVENT | 5038 | 20.1624 | 0.9131 | 0.9335 | 0.0722 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| all_matches | -3 | EVENT+IMP | 5038 | 20.1624 | 0.9131 | 0.9313 | 0.0721 | +0.00005 [-0.00044, +0.00051] | 0.0149 | 0.2478 |
| all_matches | -3 | EVENT+ORACLE | 5038 | 20.1624 | 0.9131 | 0.9263 | 0.0605 | +0.01166 [+0.00914, +0.01437] | 0.0546 | 0.7447 |
| all_matches | 0 | BASE | 5038 | 22.9827 | 0.5302 | 0.6463 | 0.3840 | -0.14479 [-0.15540, -0.13322] | 0.2989 | 0.9148 |
| all_matches | 0 | EVENT | 5038 | 22.9827 | 0.5302 | 0.5211 | 0.2392 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| all_matches | 0 | EVENT+IMP | 5038 | 22.9827 | 0.5302 | 0.5200 | 0.2400 | -0.00074 [-0.00235, +0.00089] | 0.0479 | 0.3406 |
| all_matches | 0 | EVENT+ORACLE | 5038 | 22.9827 | 0.5302 | 0.5243 | 0.1873 | +0.05194 [+0.04686, +0.05722] | 0.1749 | 0.5629 |
| all_matches | 3 | BASE | 5038 | 25.9827 | 0.0798 | 0.2066 | 0.1523 | -0.08470 [-0.09685, -0.07250] | 0.1795 | 0.9420 |
| all_matches | 3 | EVENT | 5038 | 25.9827 | 0.0798 | 0.0452 | 0.0676 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| all_matches | 3 | EVENT+IMP | 5038 | 25.9827 | 0.0798 | 0.0460 | 0.0678 | -0.00021 [-0.00061, +0.00020] | 0.0101 | 0.1552 |
| all_matches | 3 | EVENT+ORACLE | 5038 | 25.9827 | 0.0798 | 0.0573 | 0.0551 | +0.01252 [+0.00981, +0.01559] | 0.0406 | 0.7703 |
| all_matches | 6 | BASE | 5038 | 28.9827 | 0.0093 | 0.0491 | 0.0274 | -0.01838 [-0.02240, -0.01453] | 0.0478 | 0.9899 |
| all_matches | 6 | EVENT | 5038 | 28.9827 | 0.0093 | 0.0017 | 0.0091 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| all_matches | 6 | EVENT+IMP | 5038 | 28.9827 | 0.0093 | 0.0018 | 0.0090 | +0.00001 [-0.00003, +0.00006] | 0.0006 | 0.0452 |
| all_matches | 6 | EVENT+ORACLE | 5038 | 28.9827 | 0.0093 | 0.0037 | 0.0079 | +0.00116 [+0.00043, +0.00207] | 0.0032 | 0.7612 |
| seq_matches | -6 | EVENT | 3011 | 17.0038 | 0.9777 | 0.9859 | 0.0164 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| seq_matches | -6 | EVENT+IMP | 3011 | 17.0038 | 0.9777 | 0.9854 | 0.0165 | -0.00009 [-0.00044, +0.00028] | 0.0034 | 0.1720 |
| seq_matches | -6 | EVENT+IMP+SEQ | 3011 | 17.0038 | 0.9777 | 0.9852 | 0.0165 | -0.00015 [-0.00054, +0.00022] | 0.0035 | 0.1918 |
| seq_matches | -3 | EVENT | 3011 | 19.5259 | 0.9226 | 0.9412 | 0.0657 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| seq_matches | -3 | EVENT+IMP | 3011 | 19.5259 | 0.9226 | 0.9384 | 0.0653 | +0.00034 [-0.00042, +0.00109] | 0.0159 | 0.2445 |
| seq_matches | -3 | EVENT+IMP+SEQ | 3011 | 19.5259 | 0.9226 | 0.9379 | 0.0653 | +0.00032 [-0.00042, +0.00111] | 0.0171 | 0.2261 |
| seq_matches | 0 | EVENT | 3011 | 22.3429 | 0.5291 | 0.5304 | 0.2421 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| seq_matches | 0 | EVENT+IMP | 3011 | 22.3429 | 0.5291 | 0.5269 | 0.2404 | +0.00171 [-0.00072, +0.00408] | 0.0540 | 0.3370 |
| seq_matches | 0 | EVENT+IMP+SEQ | 3011 | 22.3429 | 0.5291 | 0.5262 | 0.2399 | +0.00219 [-0.00081, +0.00496] | 0.0591 | 0.3129 |
| seq_matches | 3 | EVENT | 3011 | 25.3429 | 0.0737 | 0.0375 | 0.0634 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| seq_matches | 3 | EVENT+IMP | 3011 | 25.3429 | 0.0737 | 0.0378 | 0.0635 | -0.00003 [-0.00065, +0.00058] | 0.0098 | 0.2238 |
| seq_matches | 3 | EVENT+IMP+SEQ | 3011 | 25.3429 | 0.0737 | 0.0378 | 0.0632 | +0.00020 [-0.00048, +0.00085] | 0.0105 | 0.1934 |
| seq_matches | 6 | EVENT | 3011 | 28.3429 | 0.0050 | 0.0012 | 0.0049 | +0.00000 [+0.00000, +0.00000] | 0.0000 | 0.0000 |
| seq_matches | 6 | EVENT+IMP | 3011 | 28.3429 | 0.0050 | 0.0012 | 0.0048 | +0.00002 [-0.00002, +0.00010] | 0.0005 | 0.0736 |
| seq_matches | 6 | EVENT+IMP+SEQ | 3011 | 28.3429 | 0.0050 | 0.0012 | 0.0048 | +0.00000 [-0.00002, +0.00004] | 0.0005 | 0.0592 |

Discovery is in `03_pass_a_count_lines.parquet` (column `where`) and tells the same story.

### (a.5) Calibration

Deciles of the predicted mean, then of the tail probability at the at-the-money line.

| bin | pred_BASE | pred_EVENT | pred_EVENT+IMP | pred_EVENT+ORACLE | obs_BASE | obs_EVENT | obs_EVENT+IMP | obs_EVENT+ORACLE |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.419 | 1.949 | 1.953 | 1.928 | 2.027 | 1.901 | 1.901 | 1.867 |
| 1 | 6.109 | 5.341 | 5.334 | 5.293 | 5.472 | 5.236 | 5.228 | 5.200 |
| 2 | 9.885 | 8.753 | 8.759 | 8.670 | 8.767 | 8.549 | 8.545 | 8.492 |
| 3 | 13.634 | 12.313 | 12.311 | 12.208 | 12.342 | 12.071 | 12.075 | 12.072 |
| 4 | 17.698 | 16.308 | 16.301 | 16.244 | 16.153 | 16.041 | 16.064 | 16.054 |
| 5 | 22.392 | 21.094 | 21.088 | 20.974 | 20.963 | 20.907 | 20.907 | 20.877 |
| 6 | 27.825 | 26.920 | 26.920 | 26.890 | 26.886 | 26.843 | 26.827 | 26.839 |
| 7 | 34.982 | 35.018 | 35.012 | 35.046 | 34.801 | 34.940 | 34.948 | 34.994 |
| 8 | 44.762 | 46.343 | 46.331 | 46.472 | 46.630 | 46.642 | 46.630 | 46.662 |
| 9 | 65.152 | 70.032 | 70.021 | 70.464 | 70.842 | 71.037 | 71.043 | 71.111 |

| bin | pred_BASE | pred_EVENT | pred_EVENT+IMP | pred_EVENT+ORACLE | obs_BASE | obs_EVENT | obs_EVENT+IMP | obs_EVENT+ORACLE |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.021 | 0.368 | 0.335 | 0.141 | 0.624 | 0.336 | 0.360 | 0.125 |
| 1 | 0.135 | 0.433 | 0.412 | 0.262 | 0.532 | 0.456 | 0.430 | 0.243 |
| 2 | 0.308 | 0.459 | 0.450 | 0.348 | 0.543 | 0.491 | 0.455 | 0.325 |
| 3 | 0.459 | 0.484 | 0.480 | 0.421 | 0.512 | 0.474 | 0.504 | 0.418 |
| 4 | 0.586 | 0.508 | 0.506 | 0.491 | 0.540 | 0.501 | 0.526 | 0.494 |
| 5 | 0.690 | 0.532 | 0.533 | 0.562 | 0.523 | 0.543 | 0.517 | 0.557 |
| 6 | 0.785 | 0.555 | 0.561 | 0.634 | 0.529 | 0.554 | 0.559 | 0.641 |
| 7 | 0.859 | 0.579 | 0.592 | 0.708 | 0.516 | 0.573 | 0.576 | 0.712 |
| 8 | 0.928 | 0.608 | 0.631 | 0.794 | 0.499 | 0.597 | 0.603 | 0.797 |
| 9 | 0.982 | 0.707 | 0.731 | 0.907 | 0.419 | 0.711 | 0.707 | 0.924 |

## (b) The volume channel

| where | n_rows | n_matches | n_players | date_min | date_max | mean_attempts | sd_attempts | mean_completed | sd_completed | mean_completion_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| discovery | 16028 | 790 | 1625 | 2015-08-28 | 2016-02-03 | 41.685 | 19.170 | 31.952 | 17.614 | 0.766 |
| confirmation | 13053 | 607 | 1640 | 2016-02-03 | 2016-05-17 | 41.772 | 19.244 | 32.119 | 17.741 | 0.769 |

**The book proxy.** Expected minutes = shrunk mean of the player's minutes in his prior starts (k=4 appearances); rate = shrunk prior attempts per 90; `mu_proxy_att = rate * expected_minutes / 90`, and `mu_proxy_comp` multiplies by his shrunk prior completion rate. Team and opponent rolling means (possession, PPDA, defensive-line x, passes, imputed block depth) use k=5 matches. Before being scored the proxy gets a one-dimensional Poisson recalibration (`log E[y] = a + b log mu`) fitted out of sample, so it is not penalised for a level or slope error no book would make.

**The models.** LightGBM Poisson on the player / team / opponent rolling block. `model_noimp` omits the four imputed-state features; `model` adds them. Discovery is match-grouped 5-fold CV; confirmation is fitted on all of discovery and scored once.

| where | target | model | n | mae | r2 | pois_dev | log_score |
|---|---|---|---|---|---|---|---|
| discovery | passes | proxy | 16028 | 10.9479 | 0.4369 | 4.7962 | 4.0237 |
| discovery | passes | model_noimp | 16028 | 10.1313 | 0.5090 | 4.1497 | 3.9556 |
| discovery | passes | model | 16028 | 10.1318 | 0.5091 | 4.1499 | 3.9558 |
| confirmation | passes | proxy | 13053 | 10.7231 | 0.4612 | 4.6005 | 4.0038 |
| confirmation | passes | model_noimp | 13053 | 9.9136 | 0.5389 | 3.9681 | 3.9408 |
| confirmation | passes | model | 13053 | 9.9083 | 0.5413 | 3.9546 | 3.9400 |
| discovery | passes_completed | proxy | 16028 | 9.5771 | 0.4732 | 4.7293 | 3.8484 |
| discovery | passes_completed | model_noimp | 16028 | 8.8699 | 0.5377 | 4.1030 | 3.7777 |
| discovery | passes_completed | model | 16028 | 8.8707 | 0.5374 | 4.1016 | 3.7774 |
| confirmation | passes_completed | proxy | 13053 | 9.4466 | 0.4902 | 4.6373 | 3.8448 |
| confirmation | passes_completed | model_noimp | 13053 | 8.7531 | 0.5644 | 3.9952 | 3.7753 |
| confirmation | passes_completed | model | 13053 | 8.7516 | 0.5664 | 3.9830 | 3.7747 |

| where | target | model | vs | n | delta |
|---|---|---|---|---|---|
| discovery | passes | model_noimp | proxy | 16028 | +0.06803 [+0.06035, +0.07547] |
| discovery | passes | model | proxy | 16028 | +0.06781 [+0.06021, +0.07498] |
| discovery | passes | model | model_noimp | 16028 | -0.00022 [-0.00184, +0.00139] |
| confirmation | passes | model_noimp | proxy | 13053 | +0.06297 [+0.05480, +0.07198] |
| confirmation | passes | model | proxy | 13053 | +0.06377 [+0.05573, +0.07273] |
| confirmation | passes | model | model_noimp | 13053 | +0.00081 [-0.00091, +0.00251] |
| discovery | passes_completed | model_noimp | proxy | 16028 | +0.07073 [+0.06169, +0.07937] |
| discovery | passes_completed | model | proxy | 16028 | +0.07109 [+0.06195, +0.07960] |
| discovery | passes_completed | model | model_noimp | 16028 | +0.00036 [-0.00177, +0.00252] |
| confirmation | passes_completed | model_noimp | proxy | 13053 | +0.06948 [+0.05921, +0.08073] |
| confirmation | passes_completed | model | proxy | 13053 | +0.07014 [+0.06029, +0.08115] |
| confirmation | passes_completed | model | model_noimp | 13053 | +0.00065 [-0.00146, +0.00289] |

The model beats the proxy on attempts by +0.0630 [+0.0548, +0.0720] nats per player-match and lifts R2 from 0.461 to 0.541. The imputed block depth contributes +0.00081 [-0.00091, +0.00251].

### Where the advantage comes from

Nested feature sets, each scored against the recalibrated proxy. `F0` is the proxy mean handed to LightGBM as a single feature, which is slightly *worse* than the proxy itself (the tree cannot improve on a monotone recalibration and pays a little variance).

| where | feature_set | n_features | n | r2 | log_score | delta |
|---|---|---|---|---|---|---|
| discovery | F0 proxy mean only | 1 | 16028 | 0.4327 | 4.0249 | -0.00120 [-0.00299, +0.00051] |
| confirmation | F0 proxy mean only | 1 | 13053 | 0.4641 | 4.0092 | -0.00536 [-0.00773, -0.00302] |
| discovery | F1 + player rolling | 11 | 16028 | 0.4535 | 4.0004 | +0.02323 [+0.01919, +0.02711] |
| confirmation | F1 + player rolling | 11 | 13053 | 0.4751 | 3.9902 | +0.01365 [+0.00916, +0.01834] |
| discovery | F2 + position / venue | 14 | 16028 | 0.4712 | 3.9832 | +0.04050 [+0.03521, +0.04589] |
| confirmation | F2 + position / venue | 14 | 13053 | 0.4944 | 3.9745 | +0.02935 [+0.02359, +0.03536] |
| discovery | F3 + team and opponent rolling | 25 | 16028 | 0.5090 | 3.9556 | +0.06803 [+0.06035, +0.07547] |
| confirmation | F3 + team and opponent rolling | 25 | 13053 | 0.5389 | 3.9408 | +0.06297 [+0.05480, +0.07198] |
| discovery | F4 + imputed block depth | 29 | 16028 | 0.5091 | 3.9558 | +0.06781 [+0.06021, +0.07498] |
| confirmation | F4 + imputed block depth | 29 | 13053 | 0.5413 | 3.9400 | +0.06377 [+0.05573, +0.07273] |

### Refit-noise floor

| target | model | n_seeds | mean_log_score | min | max | spread |
|---|---|---|---|---|---|---|
| passes | model | 3 | 3.94023 | 3.93994 | 3.94071 | 0.00077 |
| passes | model_noimp | 3 | 3.94041 | 3.93945 | 3.94094 | 0.00149 |

### The gate: is there room for a specialist?

A generalist fitted on all training rows against a specialist fitted only on the scenario's training rows, both scored on held-out scenario rows (and, as the reverse check, on held-out rows outside it). A positive `delta_specialist_minus_generalist` means targeting has room. The scenarios, and every cut in them, are quantiles of the discovery rows; `metronome_out` flags the team-matches whose usual highest-volume passer -- the player of the recent squad with the highest attempts per 90 over his strictly earlier appearances for that team -- did not play, and scores his team-mates.

| scenario | cut | n_discovery | n_confirmation | share_discovery | mean_attempts_in | mean_attempts_out |
|---|---|---|---|---|---|---|
| metronome_out |  | 4636 | 3555 | 0.2892 | 41.4061 | 41.8489 |
| deep_opponent | 48.1884 | 5348 | 4875 | 0.3337 | 43.2301 | 40.9078 |
| pressing_opponent | 2.0062 | 5348 | 5434 | 0.3337 | 39.2683 | 43.1712 |
| high_volume_passer | 45.7866 | 5343 | 4679 | 0.3334 | 55.8456 | 34.2985 |
| possession_edge | 0.0210 | 5346 | 4667 | 0.3335 | 48.7627 | 38.0281 |

| scenario | where | scored_on | n | n_matches | mean_y | log_score_generalist | log_score_specialist | delta | mae_generalist | mae_specialist |
|---|---|---|---|---|---|---|---|---|---|---|
| metronome_out | discovery | scenario | 4636 | 390 | 41.1808 | 3.9375 | 3.9506 | -0.0131 [-0.0200, -0.0067] | 10.0063 | 10.1613 |
| metronome_out | discovery | off_scenario | 11392 | 720 | 41.8905 | 3.9541 | 3.9697 | -0.0156 [-0.0206, -0.0109] | 10.1828 | 10.4112 |
| metronome_out | confirmation | scenario | 3555 | 285 | 41.6999 | 3.9321 | 3.9401 | -0.0079 [-0.0157, +0.0003] | 9.8288 | 9.9421 |
| metronome_out | confirmation | off_scenario | 9498 | 559 | 41.7989 | 3.9305 | 3.9430 | -0.0126 [-0.0175, -0.0079] | 9.9381 | 10.1169 |
| deep_opponent | discovery | scenario | 5348 | 412 | 43.2382 | 3.9671 | 3.9807 | -0.0136 [-0.0199, -0.0065] | 10.3615 | 10.4907 |
| deep_opponent | discovery | off_scenario | 10680 | 676 | 40.9076 | 3.9403 | 3.9750 | -0.0347 [-0.0425, -0.0271] | 10.0167 | 10.4334 |
| deep_opponent | confirmation | scenario | 4875 | 335 | 43.2211 | 3.9610 | 3.9712 | -0.0103 [-0.0171, -0.0043] | 10.2025 | 10.3125 |
| deep_opponent | confirmation | off_scenario | 8178 | 488 | 40.9080 | 3.9130 | 3.9545 | -0.0415 [-0.0510, -0.0315] | 9.7329 | 10.1637 |
| pressing_opponent | discovery | scenario | 5348 | 418 | 38.9731 | 3.8904 | 3.9004 | -0.0100 [-0.0158, -0.0042] | 9.5478 | 9.6796 |
| pressing_opponent | discovery | off_scenario | 10680 | 685 | 43.0434 | 3.9787 | 4.0109 | -0.0322 [-0.0391, -0.0251] | 10.4242 | 10.7918 |
| pressing_opponent | confirmation | scenario | 5434 | 392 | 39.5589 | 3.8748 | 3.8834 | -0.0085 [-0.0132, -0.0036] | 9.3618 | 9.4341 |
| pressing_opponent | confirmation | off_scenario | 7619 | 494 | 43.3503 | 3.9709 | 3.9924 | -0.0215 [-0.0295, -0.0139] | 10.2981 | 10.4750 |
| high_volume_passer | discovery | scenario | 5343 | 779 | 55.7797 | 4.2393 | 4.2459 | -0.0067 [-0.0109, -0.0027] | 12.9285 | 13.0348 |
| high_volume_passer | discovery | off_scenario | 10685 | 790 | 34.6373 | 3.8042 | 4.2871 | -0.4829 [-0.5024, -0.4627] | 8.7333 | 13.8028 |
| high_volume_passer | confirmation | scenario | 4679 | 602 | 55.9209 | 4.1981 | 4.2059 | -0.0078 [-0.0117, -0.0039] | 12.4798 | 12.6150 |
| high_volume_passer | confirmation | off_scenario | 8374 | 607 | 33.8661 | 3.7816 | 4.3204 | -0.5388 [-0.5642, -0.5128] | 8.4715 | 14.1309 |
| possession_edge | discovery | scenario | 5346 | 524 | 48.6094 | 4.0570 | 4.0722 | -0.0153 [-0.0209, -0.0098] | 11.3185 | 11.5182 |
| possession_edge | discovery | off_scenario | 10682 | 789 | 38.2199 | 3.8953 | 3.9632 | -0.0679 [-0.0799, -0.0562] | 9.5379 | 10.4084 |
| possession_edge | confirmation | scenario | 4667 | 434 | 48.9383 | 4.0600 | 4.0654 | -0.0054 [-0.0108, +0.0001] | 11.2872 | 11.3871 |
| possession_edge | confirmation | off_scenario | 8386 | 607 | 37.7837 | 3.8591 | 3.9355 | -0.0764 [-0.0905, -0.0634] | 9.1409 | 10.0856 |

Every confirmation row scored on its own scenario is negative. The reverse direction is much more negative still, as it must be: a specialist trained on deep-block matches is badly wrong about the rest. There is no scenario here in which the generalist underfits.

## Proxy honesty check: total goals, the one count market with a real line

The identical shrunk rolling-mean construction is applied to total goals on football-data.co.uk matches of the same four divisions (E0, SP1, I1, F1), Bet365's Over/Under 2.5 price is de-vigged proportionally, and the two are scored against each other. This is the only place in the stage where a real bookmaker can be measured.

### At the standard 2.5 line

| population | where | model | n | base_rate | log_loss | brier | mean_p | delta_vs_proxy_cal | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| four_leagues_all_seasons | discovery | climatology | 17232 | 0.48659 | 0.69279 | 0.24982 | 0.48659 | -0.00504 | -0.00653 | -0.00361 |
| four_leagues_all_seasons | discovery | proxy_raw | 17232 | 0.48659 | 0.68847 | 0.24766 | 0.47612 | -0.00072 | -0.00130 | -0.00016 |
| four_leagues_all_seasons | discovery | proxy_calibrated | 17232 | 0.48659 | 0.68774 | 0.24732 | 0.48660 | 0.00000 | 0.00000 | 0.00000 |
| four_leagues_all_seasons | discovery | bet365_novig | 17232 | 0.48659 | 0.67721 | 0.24222 | 0.48426 | 0.01053 | 0.00881 | 0.01228 |
| four_leagues_all_seasons | confirmation | climatology | 11438 | 0.51338 | 0.69422 | 0.25054 | 0.48659 | -0.00747 | -0.00951 | -0.00550 |
| four_leagues_all_seasons | confirmation | proxy_raw | 11438 | 0.51338 | 0.68709 | 0.24697 | 0.50209 | -0.00034 | -0.00094 | 0.00026 |
| four_leagues_all_seasons | confirmation | proxy_calibrated | 11438 | 0.51338 | 0.68675 | 0.24681 | 0.50643 | 0.00000 | 0.00000 | 0.00000 |
| four_leagues_all_seasons | confirmation | bet365_novig | 11438 | 0.51338 | 0.67615 | 0.24161 | 0.51445 | 0.01060 | 0.00805 | 0.01301 |
| four_leagues_2015_16 | discovery | climatology | 792 | 0.47980 | 0.69233 | 0.24959 | 0.47980 | -0.00091 | -0.00381 | 0.00199 |
| four_leagues_2015_16 | discovery | proxy_raw | 792 | 0.47980 | 0.69480 | 0.25074 | 0.46933 | -0.00338 | -0.00921 | 0.00244 |
| four_leagues_2015_16 | discovery | proxy_calibrated | 792 | 0.47980 | 0.69142 | 0.24914 | 0.47977 | 0.00000 | 0.00000 | 0.00000 |
| four_leagues_2015_16 | discovery | bet365_novig | 792 | 0.47980 | 0.67751 | 0.24250 | 0.47465 | 0.01391 | 0.00429 | 0.02308 |
| four_leagues_2015_16 | confirmation | climatology | 608 | 0.51480 | 0.69516 | 0.25101 | 0.47980 | -0.00403 | -0.00727 | -0.00060 |
| four_leagues_2015_16 | confirmation | proxy_raw | 608 | 0.51480 | 0.68948 | 0.24816 | 0.47958 | 0.00164 | -0.00476 | 0.00752 |
| four_leagues_2015_16 | confirmation | proxy_calibrated | 608 | 0.51480 | 0.69113 | 0.24899 | 0.48342 | 0.00000 | 0.00000 | 0.00000 |
| four_leagues_2015_16 | confirmation | bet365_novig | 608 | 0.51480 | 0.67086 | 0.23934 | 0.49710 | 0.02027 | 0.00853 | 0.03207 |

### At a line the proxy itself would have set (the exact analogue)

Restricted to the matches whose proxy mean rounds to the 2.5 line, so the book is pricing a line the proxy also posts -- the same situation as the pass simulation, where the line is always the proxy's own mean.

| where | n | base_rate | log_loss_proxy | log_loss_bet365 | book_minus_proxy_nats | ci_lo | ci_hi | relative_gap | sd_p_proxy | sd_p_bet365 |
|---|---|---|---|---|---|---|---|---|---|---|
| discovery | 15828 | 0.47858 | 0.68960 | 0.68065 | 0.00896 | 0.00723 | 0.01067 | 0.01299 | 0.03616 | 0.06624 |
| confirmation | 9691 | 0.49933 | 0.68891 | 0.67760 | 0.01131 | 0.00875 | 0.01412 | 0.01642 | 0.03516 | 0.08107 |

A real book beats the identical rolling-mean proxy by +0.01131 [+0.00875, +0.01412] nats at a proxy-set line, 1.64% of the proxy's log loss, and carries 2.3x the spread of opinion (sd of P(over) 0.081 vs 0.035).

### In line units

Inverting each side's probability back through the fitted count distribution gives the mean each is pricing, which is the quantity a market actually expresses as a line.

| market | n | sd_outcome | mean_abs_diff | sd_diff | abs_diff_over_sd_outcome | share_diff_ge_half_unit | share_diff_ge_1_unit | share_diff_ge_2_units |
|---|---|---|---|---|---|---|---|---|
| total_goals_bet365_vs_proxy | 25519 | 1.6247 | 0.1996 | 0.2659 | 0.1228 | 0.0631 | 0.0063 | 0.0000 |
| completed_passes_model_vs_proxy | 29081 | 17.6705 | 3.8444 | 5.2211 | 0.2176 | 0.9039 | 0.8100 | 0.6383 |

### What a real book makes betting into a proxy-priced market

| hold | threshold | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi |
|---|---|---|---|---|---|---|---|---|
| 0.0400 | 0.0000 | 11438 | 8696 | 0.7603 | 0.0888 | 0.0952 | 0.0746 | 0.1155 |
| 0.0400 | 0.0200 | 11438 | 7301 | 0.6383 | 0.1039 | 0.1207 | 0.0992 | 0.1427 |
| 0.0400 | 0.0500 | 11438 | 5454 | 0.4768 | 0.1271 | 0.1451 | 0.1199 | 0.1686 |
| 0.0600 | 0.0000 | 11438 | 7361 | 0.6436 | 0.0824 | 0.0976 | 0.0767 | 0.1190 |
| 0.0600 | 0.0200 | 11438 | 6099 | 0.5332 | 0.0973 | 0.1182 | 0.0943 | 0.1413 |
| 0.0600 | 0.0500 | 11438 | 4327 | 0.3783 | 0.1231 | 0.1436 | 0.1153 | 0.1710 |
| 0.0800 | 0.0000 | 11438 | 6170 | 0.5394 | 0.0761 | 0.0974 | 0.0751 | 0.1191 |
| 0.0800 | 0.0200 | 11438 | 4900 | 0.4284 | 0.0933 | 0.1097 | 0.0828 | 0.1354 |
| 0.0800 | 0.0500 | 11438 | 3362 | 0.2939 | 0.1202 | 0.1386 | 0.1086 | 0.1695 |

| log_loss_gap_book_minus_proxy_nats | ci_lo | ci_hi | roi_book_vs_proxy_mean_over_holds | roi_haircut |
|---|---|---|---|---|
| 0.01060 | 0.00805 | 0.01301 | 0.11619 | 0.11619 |

## (c) End to end

### (c.1) The ceiling

Exact law of total variance for `C = sum of Bernoulli(p_t)` given attempts `A`: `Var(C) = Var(A * pbar) + E[A * pbar * (1 - pbar)]`. The second term is all a per-pass completion model can ever touch.

| n | sd_completed | sd_attempts | mean_rate | sd_rate | var_completed | var_within_match_bernoulli | var_between | share_volume_and_rate_level | share_irreducible_bernoulli | where |
|---|---|---|---|---|---|---|---|---|---|---|
| 29081 | 17.6705 | 19.2024 | 0.7676 | 0.1269 | 312.2474 | 6.8654 | 305.3820 | 0.9780 | 0.0220 | all |
| 13053 | 17.7398 | 19.2433 | 0.7689 | 0.1271 | 314.7016 | 6.8462 | 307.8555 | 0.9782 | 0.0218 | confirmation |

### (c.2) Which channel carries the improvement

Completed passes are modelled as a compound: attempts ~ negative binomial with a dispersion fitted on discovery, completed | attempts ~ Binomial(attempts, rate). Each row swaps one channel from the proxy to the model.

| where | combination | n | mean_completed | log_score | mae | r2 | delta |
|---|---|---|---|---|---|---|---|
| discovery | proxy_att x proxy_rate | 16028 | 31.9516 | 3.8552 | 9.5771 | 0.4732 | +0.0000 [+0.0000, +0.0000] |
| discovery | model_att x proxy_rate | 16028 | 31.9516 | 3.7805 | 8.8737 | 0.5388 | +0.0747 [+0.0660, +0.0832] |
| discovery | proxy_att x model_rate | 16028 | 31.9516 | 3.8342 | 9.3839 | 0.4901 | +0.0210 [+0.0187, +0.0232] |
| discovery | model_att x model_rate | 16028 | 31.9516 | 3.7746 | 8.8237 | 0.5417 | +0.0806 [+0.0706, +0.0902] |
| confirmation | proxy_att x proxy_rate | 13053 | 32.1186 | 3.8503 | 9.4466 | 0.4902 | +0.0000 [+0.0000, +0.0000] |
| confirmation | model_att x proxy_rate | 13053 | 32.1186 | 3.7779 | 8.7679 | 0.5654 | +0.0724 [+0.0627, +0.0831] |
| confirmation | proxy_att x model_rate | 13053 | 32.1186 | 3.8290 | 9.2665 | 0.5102 | +0.0213 [+0.0182, +0.0246] |
| confirmation | model_att x model_rate | 13053 | 32.1186 | 3.7732 | 8.7405 | 0.5686 | +0.0771 [+0.0657, +0.0897] |

### (c.3) Betting a line the proxy sets

The proxy posts `floor(its own mean) + 0.5` and prices both sides to the stated hold. By construction its own fair probability sits near 0.5 there, which is what a line-based market is: the disagreement is expressed in the *line*, not the price. The model bets whichever side its own distribution makes positive EV, with an edge threshold chosen on discovery. **Confirmation only.** `roi_after_haircut` subtracts the ROI a real book makes against the goals proxy.

| market | source | hold | threshold_chosen_on_discovery |
|---|---|---|---|
| completed_passes | model_att x proxy_rate | 0.0400 | 0.0500 |
| completed_passes | model_att x proxy_rate | 0.0600 | 0.0500 |
| completed_passes | model_att x proxy_rate | 0.0800 | 0.0500 |
| completed_passes | proxy_att x model_rate | 0.0400 | 0.0500 |
| completed_passes | proxy_att x model_rate | 0.0600 | 0.0500 |
| completed_passes | proxy_att x model_rate | 0.0800 | 0.0500 |
| completed_passes | model_att x model_rate | 0.0400 | 0.0500 |
| completed_passes | model_att x model_rate | 0.0600 | 0.0500 |
| completed_passes | model_att x model_rate | 0.0800 | 0.0500 |
| pass_attempts | model_att | 0.0400 | 0.0500 |
| pass_attempts | model_att | 0.0600 | 0.0500 |
| pass_attempts | model_att | 0.0800 | 0.0500 |

| market | source | hold | threshold | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | mean_p_model | mean_p_book | realized_a_rate | roi_after_haircut | roi_lo_after_haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| completed_passes | model_att x proxy_rate | 0.0400 | 0.0500 | 13053 | 10120 | 0.7753 | 0.2631 | 0.2562 | 0.2315 | 0.2807 | 0.4799 | 0.4596 | 0.4957 | 0.1401 | 0.1153 |
| completed_passes | model_att x proxy_rate | 0.0600 | 0.0500 | 13053 | 9471 | 0.7256 | 0.2530 | 0.2459 | 0.2215 | 0.2704 | 0.4813 | 0.4595 | 0.4970 | 0.1297 | 0.1053 |
| completed_passes | model_att x proxy_rate | 0.0800 | 0.0500 | 13053 | 8821 | 0.6758 | 0.2437 | 0.2389 | 0.2140 | 0.2634 | 0.4826 | 0.4594 | 0.4964 | 0.1227 | 0.0978 |
| completed_passes | proxy_att x model_rate | 0.0400 | 0.0500 | 13053 | 8566 | 0.6562 | 0.1946 | 0.1813 | 0.1558 | 0.2076 | 0.4924 | 0.4597 | 0.5023 | 0.0651 | 0.0397 |
| completed_passes | proxy_att x model_rate | 0.0600 | 0.0500 | 13053 | 7627 | 0.5843 | 0.1883 | 0.1683 | 0.1415 | 0.1963 | 0.4970 | 0.4597 | 0.5040 | 0.0522 | 0.0254 |
| completed_passes | proxy_att x model_rate | 0.0800 | 0.0500 | 13053 | 6746 | 0.5168 | 0.1828 | 0.1631 | 0.1366 | 0.1917 | 0.5028 | 0.4598 | 0.5071 | 0.0469 | 0.0204 |
| completed_passes | model_att x model_rate | 0.0400 | 0.0500 | 13053 | 10405 | 0.7971 | 0.2781 | 0.2660 | 0.2432 | 0.2901 | 0.4885 | 0.4595 | 0.5005 | 0.1498 | 0.1270 |
| completed_passes | model_att x model_rate | 0.0600 | 0.0500 | 13053 | 9815 | 0.7519 | 0.2669 | 0.2504 | 0.2273 | 0.2741 | 0.4901 | 0.4595 | 0.5025 | 0.1342 | 0.1111 |
| completed_passes | model_att x model_rate | 0.0800 | 0.0500 | 13053 | 9201 | 0.7049 | 0.2570 | 0.2393 | 0.2156 | 0.2638 | 0.4923 | 0.4594 | 0.5021 | 0.1231 | 0.0994 |
| pass_attempts | model_att | 0.0400 | 0.0500 | 13053 | 10085 | 0.7726 | 0.2591 | 0.2592 | 0.2365 | 0.2810 | 0.4884 | 0.4672 | 0.4914 | 0.1430 | 0.1203 |
| pass_attempts | model_att | 0.0600 | 0.0500 | 13053 | 9432 | 0.7226 | 0.2489 | 0.2530 | 0.2299 | 0.2745 | 0.4898 | 0.4671 | 0.4920 | 0.1369 | 0.1137 |
| pass_attempts | model_att | 0.0800 | 0.0500 | 13053 | 8777 | 0.6724 | 0.2396 | 0.2453 | 0.2222 | 0.2676 | 0.4912 | 0.4671 | 0.4949 | 0.1291 | 0.1060 |

These numbers are the badness of the proxy, not an edge. They are reported because the protocol asks for them, and immediately qualified in (c.5) and (c.6).

### (c.4) Closing-line-value check: who is right where they disagree?

| bin | n | diff_lo | diff_hi | mean_p_model | mean_p_book | observed | log_loss_model | log_loss_book | model_better |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2611 | -0.4521 | -0.1158 | 0.2462 | 0.4598 | 0.2650 | 0.5554 | 0.6579 | 0.1024 |
| 1 | 2611 | -0.1158 | -0.0113 | 0.3985 | 0.4599 | 0.4408 | 0.6870 | 0.6864 | -0.0007 |
| 2 | 2611 | -0.0113 | 0.0736 | 0.4918 | 0.4600 | 0.5067 | 0.6907 | 0.6967 | 0.0060 |
| 3 | 2610 | 0.0737 | 0.1644 | 0.5774 | 0.4597 | 0.5820 | 0.6778 | 0.7095 | 0.0317 |
| 4 | 2610 | 0.1644 | 0.4701 | 0.6972 | 0.4586 | 0.6897 | 0.6151 | 0.7286 | 0.1135 |

The model is closer to the truth in every bin where it disagrees materially. That establishes it beats a rolling mean; it says nothing about beating a bookmaker.

### (c.5) A book that sets its own line

The opponent's mean is moved a fraction `kappa` of the way from the proxy to the model and it posts *its own* line there, which is what a real book does with a disagreement. `kind = goals_calibrated` is the kappa at which the opponent's mean sits as far from the proxy, relative to the outcome's sd, as a real Bet365 line sits from the identical proxy on total goals.

| kind | kappa | hold | mean_line | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | log_loss_book | log_loss_model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goals_calibrated | 0.5647 | 0.0400 | 31.8511 | 9236 | 0.7076 | 0.1241 | 0.0980 | 0.0732 | 0.1246 | 0.6943 | 0.6863 |
| goals_calibrated | 0.5647 | 0.0600 | 31.8511 | 8024 | 0.6147 | 0.1169 | 0.0901 | 0.0643 | 0.1182 | 0.6943 | 0.6863 |
| goals_calibrated | 0.5647 | 0.0800 | 31.8511 | 6913 | 0.5296 | 0.1100 | 0.0945 | 0.0680 | 0.1243 | 0.6943 | 0.6863 |
| quarter | 0.2500 | 0.0400 | 31.8305 | 10711 | 0.8206 | 0.1990 | 0.1826 | 0.1594 | 0.2058 | 0.6950 | 0.6671 |
| quarter | 0.2500 | 0.0600 | 31.8305 | 9932 | 0.7609 | 0.1894 | 0.1717 | 0.1484 | 0.1958 | 0.6950 | 0.6671 |
| quarter | 0.2500 | 0.0800 | 31.8305 | 9194 | 0.7044 | 0.1799 | 0.1580 | 0.1338 | 0.1823 | 0.6950 | 0.6671 |
| half | 0.5000 | 0.0400 | 31.8463 | 9740 | 0.7462 | 0.1408 | 0.1086 | 0.0852 | 0.1347 | 0.6940 | 0.6832 |
| half | 0.5000 | 0.0600 | 31.8463 | 8611 | 0.6597 | 0.1335 | 0.1080 | 0.0828 | 0.1349 | 0.6940 | 0.6832 |
| half | 0.5000 | 0.0800 | 31.8463 | 7602 | 0.5824 | 0.1261 | 0.0999 | 0.0730 | 0.1285 | 0.6940 | 0.6832 |
| three_quarters | 0.7500 | 0.0400 | 31.8650 | 6834 | 0.5236 | 0.0755 | 0.0463 | 0.0170 | 0.0786 | 0.6932 | 0.6918 |
| three_quarters | 0.7500 | 0.0600 | 31.8650 | 5129 | 0.3929 | 0.0703 | 0.0373 | 0.0036 | 0.0734 | 0.6932 | 0.6918 |
| three_quarters | 0.7500 | 0.0800 | 31.8650 | 3680 | 0.2819 | 0.0664 | 0.0173 | -0.0240 | 0.0628 | 0.6932 | 0.6918 |

### (c.6) The ladder: how much would the opponent have to know?

Here the line stays at the proxy's mean and only the opponent's *price* improves, as a logit blend of proxy and model. It is the price-side twin of (c.5).

| lam | opponent_log_loss | proxy_log_loss | rel_gap_over_proxy | goals_rel_gap | at_or_past_book_quality | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.6958 | 0.6958 | 0.0000 | 0.0154 | False | 10696 | 0.8194 | 0.2478 | 0.2354 | 0.2131 | 0.2585 |
| 0.1000 | 0.6850 | 0.6958 | 0.0155 | 0.0154 | True | 10432 | 0.7992 | 0.2076 | 0.1960 | 0.1746 | 0.2187 |
| 0.2000 | 0.6756 | 0.6958 | 0.0291 | 0.0154 | True | 10060 | 0.7707 | 0.1719 | 0.1597 | 0.1385 | 0.1815 |
| 0.3000 | 0.6675 | 0.6958 | 0.0407 | 0.0154 | True | 9612 | 0.7364 | 0.1389 | 0.1264 | 0.1058 | 0.1475 |
| 0.5000 | 0.6553 | 0.6958 | 0.0582 | 0.0154 | True | 7988 | 0.6120 | 0.0825 | 0.0678 | 0.0468 | 0.0886 |
| 0.7000 | 0.6480 | 0.6958 | 0.0687 | 0.0154 | True | 3746 | 0.2870 | 0.0359 | 0.0302 | 0.0025 | 0.0567 |
| 1.0000 | 0.6452 | 0.6958 | 0.0727 | 0.0154 | True | 0 | 0.0000 |  |  |  |  |

`03_pass_c_relative_scale.parquet` additionally normalises both markets by their own log score. That normalisation is reported for completeness but is **not** the right scale for an EV claim: a count log score is dominated by the distribution's irreducible entropy, so relative gaps there are not comparable to a binary log loss at a line.

| goals_log_loss_proxy | goals_log_loss_book | goals_relative_gap | passes_count_log_score_proxy | passes_count_log_score_model | passes_relative_gap | implied_book_count_log_score | model_minus_implied_book_nats |
|---|---|---|---|---|---|---|---|
| 0.68675 | 0.67615 | 0.01544 | 3.85030 | 3.77325 | 0.02001 | 3.79085 | -0.01760 |

## Answering the three questions the stage was set

1. *Does a +0.0026 nats per-pass gain move a count line, and by how much in probability terms at a typical line?* It moves P(over) at a 22.3-completion line by 5.9 percentage points and improves Brier by +0.00219 [-0.00081, +0.00496] -- a movement large enough to matter commercially and an improvement indistinguishable from zero. At the count level it is worth +0.0046 [-0.0045, +0.0136] nats per player-match. And the LightGBM-imputation version of the gain does not survive refitting on the full pass population at all.
2. *Is there a scenario where targeting helps?* No. All five pre-registered scenarios fail the gate on confirmation, including the deep-lying-passer-absent scenario the stage was asked to test.
3. *What fraction of the achievable improvement is volume rather than completion?* 72% of the joint gain is unique to the volume channel and 6% unique to the completion-rate channel; the variance decomposition puts the hard ceiling on the completion side at 2.2% of the outcome's variance. Since the completion-rate gain that exists is bought with rolling and opponent features rather than tracking, **the tracking-derived work is irrelevant to this market.**

## Caveats

- **No pass-prop lines exist in this data.** Every EV number is against a constructed proxy. The transfer from the goals market that turns it into a claim about a real book assumes a bookmaker's relative disagreement with a rolling mean is the same across markets, which is an assumption, not a measurement, and probably a generous one: player pass counts are far more predictable from public information (minutes, role, opponent) than match goals are, so a real desk's model is likely much closer to ours than the goals-calibrated kappa implies.
- **Part (a) conditions on realised attempts.** It answers 'does better per-pass completion move a completed-pass line', not 'can you price the prop'.
- **Starting position is taken from the match's own line-up.** A book posting before team news would not have it; the `F1 -> F2` step of the ablation (+0.0157 nats, which also includes venue and competition) bounds what it is worth.
- **The per-pass refit is single-seed**, matching the programme's xPass setting (`n_seeds_xpass = 1`). The count-level deltas it feeds are all well inside their own intervals, so seed noise cannot flip the sign of the conclusion, but the per-pass numbers carry an unmeasured seed component. The volume-channel models do have a 3-seed refit floor.
- **The sequence-student variants live on 251 of the 417 matches**, so they are only comparable through the `seq_matches` population, which is why both populations are reported everywhere.
- **StatsBomb 2015/16 is one season of four leagues**; the confirmation half is February to May 2016. Nothing here is tested across seasons or in other leagues.
- **Passes include throw-ins, corners and free kicks** as StatsBomb counts them; a real prop's settlement rules may differ, and so may its definition of a completed pass.
- **The compound model assumes completion is independent of attempts given the features.** Real player-matches violate this mildly (see a.2).

## Reproduction

```shell
cd /home/user/geo-model
python -m research.scenario_ev.pass_counts --stage passpreds   # ~6 min, 2 threads
python -m research.scenario_ev.pass_counts --stage a           # ~20 s
python -m research.scenario_ev.pass_counts --stage b           # ~2 min
python -m research.scenario_ev.pass_counts --stage goals       # ~15 s
python -m research.scenario_ev.pass_counts --stage c           # ~40 s
python -m research.scenario_ev.pass_counts --stage report
python -m pytest research/scenario_ev/tests/test_pass_counts.py -q
```

Cached intermediates (not committed) live under `<PRIV_DATA_DIR>/processed/scenario_ev/`: `pass_preds_full.parquet` (per-pass out-of-fold probabilities), `pass_panel.parquet` (the player-match panel), `pass_a_counts.parquet`, `pass_b_preds.parquet`, `pass_c_preds.parquet`, `goals_proxy.parquet`.

