# 04 Fouls and the referee channel: is a foul market priceable against a rolling mean?

Stage 04 of the scenario expected-value phase. Machine-written by `research/scenario_ev/fouls.py` from the committed `research/scenario_ev/reports/04_fouls_*.parquet` tables, so every number below is read from one of them. Discovery numbers are exploratory and multiplicity-inflated; **confirmation numbers are scored once and are the headline**.

```
cd /home/user/geo-model
python -m research.scenario_ev.fouls --stage build   # ~2 min, both universes + the leakage audit
python -m research.scenario_ev.fouls --stage a       # ~8 min, odds-table team and match foul totals
python -m research.scenario_ev.fouls --stage b       # ~2 min, player fouls, referee and matchup channels
python -m research.scenario_ev.fouls --stage c       # ~20 s, the imputed-state layer
python -m research.scenario_ev.fouls --stage d       # ~1 min, betting simulation and the goals haircut
python -m research.scenario_ev.fouls --stage report  # writes this file
python -m pytest research/scenario_ev/tests -q
```

## Verdict

1. **A rolling mean is already good at fouls, unlike corners.** The book proxy explains R2 = 0.2188 of a single team's foul count on confirmation (n = 84030 team-matches) against 0.033 for corners in stage 02, because foul rates are dominated by persistent competition and officiating effects that a rolling mean captures by construction.
2. **The event-only model beats it, by a real but small margin.** The full model beats the *recalibrated* proxy by +0.0055 [+0.0046, +0.0064] nats per team-match and +0.0086 [+0.0072, +0.0100] per match total. Against a 5-seed refit spread of 0.00019 that is about 30x the noise, so it is a real gap and not a seed.
3. **Division effects are large and already absorbed.** Between-division variation is 11.1% of the variance of a team's fouls on confirmation and division-season 13.8%. But adding division identifiers and league levels to a model that already has the proxy and the rolling event block is worth +0.0000 [-0.0003, +0.0003] nats. The proxy carries the division; the model does not have to be told about it.
4. **The matchup channel is real in exactly one direction, at about 3x refit noise.** For fouls *won*, adding the opponent's foul propensity to the event + referee model is worth +0.0023 [+0.0011, +0.0036] nats per appearance on confirmation, against a 0.00078-nat spread of that same increment over 5 refits at the tuned parameters. For fouls *committed*, adding the opponent's dribbling down this player's flank -- the mechanism the card hypothesis rested on -- is worth +0.0003 [-0.0003, +0.0008], inside its 0.00082-nat spread. Fouls won are a matchup stat; fouls committed are a player-and-official stat. Two caveats travel with this and are stated in section 3.1: both fouls-won channels were inside noise on *discovery* and cleared zero only on confirmation, and four (channel, target) cells were tested, so the claim rests on sign consistency across halves and seeds rather than on independent replication.
5. **The referee channel is real but is mostly a league effect.** Appearances under a top-tercile referee average 1.2993 fouls against 1.0253 in the bottom tercile, but the referee prior's R2 on a team's fouls falls from 0.0955 to 0.0247 once competition means are removed, and adding the referee block to the player model is worth +0.0001 [-0.0005, +0.0007] nats per appearance.
6. **The imputed defensive state adds nothing**: -0.0002 [-0.0006, +0.0002] nats for fouls committed and +0.0001 [-0.0003, +0.0005] for fouls won, both inside the refit noise. That is the fourth market in a row in which the programme's tracking-derived channel fails to move an outcome model that already reads the same events.
7. **And none of it is +EV.** Betting the model into a market priced off the proxy at a 6% hold returns ROI +0.1090 [+0.0943, +0.1242] on team fouls (15304 bets on confirmation) and +0.1395 [+0.1216, +0.1568] on match totals (10238 bets). On the one count market whose line we do observe, a real bookmaker beats the identical proxy by +0.0090 [+0.0079, +0.0100] nats and turns the same kind of simulated profit into a loss: a haircut of 20.7 ROI points. After it, team fouls return -0.0983 [-0.1130, -0.0831] and match totals -0.0678 [-0.0857, -0.0505]. Both intervals lie entirely below zero.

**Read plainly: a null result on the betting question.** Our margin over a rolling mean on fouls is smaller than the margin a real bookmaker already holds over that same rolling mean on goals. It is a positive result on two smaller questions: the asymmetry of the foul matchup, and the fact that the apparent referee channel is almost entirely a league effect.

## 1. Protocol and universes

**Discovery / confirmation.** Matches are cut chronologically by `common.chronological_split`, separately inside each population, before any modelling. Every threshold, shrinkage, capacity choice, calibration map, dispersion and bet rule below is chosen on discovery; confirmation is scored once.

- Odds-table universe: 210076 team-rows (105038 matches) of the 121,721 football-data matches that record fouls, after requiring at least 5 strictly-prior matches with fouls recorded for both teams and both directions. Discovery 126046 team-rows, confirmation 84030.
- Player universe: 33329 appearances by 1770 players in 1398 StatsBomb matches (the four 2015/16 league seasons), non-goalkeepers with at least 3 strictly-prior appearances. Discovery 19359, confirmation 13970. Referee known on 96.5% of rows; a direct opponent is identified on 96.5%.
- Targets: team fouls (mean 12.31 on confirmation), match total fouls, player fouls committed (mean 1.179, sd 1.246) and player fouls won (mean 1.121, sd 1.302).
- Football-data carries a handful of impossible foul counts (up to 145 in one match); team counts are winsorised at 40, which touches 5 of 210,076 team-rows (0.002%).

**Reuse.** The odds layer is built with the corner stage's prior-feature primitives (`corners_features.to_team_match`, `add_prior_features`, `prior_by_date`, `shrink`), whose `TEAM_STATS` already accumulate fouls; only the two pieces the corner proxy did not need (raw prior sums/counts for fouls and the home/away league levels) are added, in `fouls_features.py`. The player layer is built on the card stage's cached StatsBomb player-match table, adding a position-group backstop, the direct opponent's prior *fouls committed* rate, the opponent's prior fouls by pitch side, and the imputed defensive-state priors. Nothing in `research/privileged_tracking` or in stages 01-03 is modified.

**Leakage audit, recomputed by brute force rather than asserted.** Run inside the builder on the *unfiltered* team-match frame, because that is the frame the priors were accumulated over.

| check | n | max_abs_error | passed |
|---|---|---|---|
| prior_foul_sum_strictly_earlier | 120 | 0.0000000000 | True |
| prior_foul_count_strictly_earlier | 120 | 0.0000000000 | True |
| league_level_constant_within_day | 477716 | 0.0000000000 | True |

The third check matters for same-day fixtures: the league level is a function of (division-season, date) alone, so two matches played on the same day cannot enter one another's reference level.

**Two smaller points of the same kind, stated rather than buried.** (i) The team-level prior foul rate *by pitch side* (`fouls_features.flank_foul_priors`, which feeds `opp_tp_fouls_<side>_pm` and `opp_mirror_fouls_pm` inside the opponent-foul block) shrinks toward a global per-side mean. That target is itself accumulated over strictly earlier dates only, with a fixed constant (4.8) for the first date, where there is nothing earlier to average; an earlier version of this stage used the whole table's mean, which was a single scalar per side and numerically negligible but not strictly prior. (ii) The player universe conditions on `minutes > 0`, i.e. on the appearance having happened. A real player prop cannot do that -- it is priced before the team sheet is certain, and a late withdrawal is a void bet, not a zero. No expected-value claim in this stage depends on it, because the betting simulation is team- and match-level only; but the player-level nats are conditional on playing and should be read that way.

## 2. (a) The odds-table layer: 105k matches

### 2.1 The book proxy and its shrinkage, chosen on discovery

The proxy is the standard side-market heuristic: expected fouls = the league's home or away level x the team's shrunk prior foul rate / league x the opponent's shrunk prior fouls-conceded rate / league, everything from strictly earlier dates in the same division-season. Its shrinkage and functional form are chosen on discovery by squared error, so the opponent is the strongest rolling mean available rather than a straw man.

| form | k | n | mse | r2 | corr | sd_pred | sd_y | bias |
|---|---|---|---|---|---|---|---|---|
| mult | 10.0000 | 126046 | 14.9776 | 0.3220 | 0.5683 | 2.7137 | 4.7001 | 0.1365 |
| add | 10.0000 | 126046 | 14.9796 | 0.3219 | 0.5682 | 2.7180 | 4.7001 | 0.1375 |
| mult | 15.0000 | 126046 | 14.9948 | 0.3212 | 0.5679 | 2.6340 | 4.7001 | 0.1661 |
| add | 15.0000 | 126046 | 14.9950 | 0.3212 | 0.5679 | 2.6369 | 4.7001 | 0.1669 |
| mult | 6.0000 | 126046 | 15.0472 | 0.3189 | 0.5661 | 2.8154 | 4.7001 | 0.1020 |
| add | 6.0000 | 126046 | 15.0531 | 0.3186 | 0.5659 | 2.8217 | 4.7001 | 0.1032 |

### 2.2 LightGBM capacity, also chosen on discovery

Chosen on an inner forward split *inside* discovery (first 75% of discovery dates to fit, last 25% to score), so confirmation is untouched and a null result cannot be confused with a capacity artefact.

| params | n_train | n_valid | loss |
|---|---|---|---|
| {'n_estimators': 300, 'num_leaves': 15, 'min_child_samples': 500, 'learning_rate': 0.04} | 85490 | 40556 | 2.76012 |
| {'n_estimators': 400, 'num_leaves': 31, 'min_child_samples': 200, 'learning_rate': 0.04} | 85490 | 40556 | 2.76040 |
| {'n_estimators': 200, 'num_leaves': 7, 'min_child_samples': 500, 'learning_rate': 0.05} | 85490 | 40556 | 2.76095 |
| {'n_estimators': 100, 'num_leaves': 7, 'min_child_samples': 1000, 'learning_rate': 0.05} | 85490 | 40556 | 2.76413 |

### 2.3 How much of a team's fouls is the division?

| split | level | n | var_share_explained | sd_between | sd_total |
|---|---|---|---|---|---|
| discovery | division | 126046 | 0.1817 | 1.9975 | 4.6855 |
| discovery | division_season | 126046 | 0.2613 | 2.3951 | 4.6855 |
| confirmation | division | 84030 | 0.1111 | 1.3564 | 4.0703 |
| confirmation | division_season | 84030 | 0.1384 | 1.5142 | 4.0703 |

### 2.4 The model ladder

`C0_proxy_raw` is the proxy as it comes; `C0_proxy_cal` adds a one-dimensional Poisson recalibration fitted on discovery, and is the reference every delta is measured against (an *un*calibrated proxy is beaten on scale alone, which would manufacture an edge out of nothing). `F1` deliberately contains no division identifier or league level, so the division question is answered by the F1 -> F2 step rather than assumed. Losses are negative-binomial log scores in nats, with the dispersion fitted on discovery; positive `delta_vs_ref` means better than the calibrated proxy.

| model | split | n | mean_y | nb_log_score | poisson_deviance | r2 | mae | delta_vs_ref | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_proxy_raw | discovery | 126046 | 13.0361 | 2.7496 | 1.1570 | 0.3238 | 3.0499 | -0.0006 | -0.0009 | -0.0004 |
| C0_proxy_raw | confirmation | 84030 | 12.3055 | 2.6869 | 1.0694 | 0.2179 | 2.8579 | -0.0005 | -0.0007 | -0.0002 |
| C0_proxy_cal | discovery | 126046 | 13.0361 | 2.7489 | 1.1555 | 0.3248 | 3.0434 | 0.0000 | 0.0000 | 0.0000 |
| C0_proxy_cal | confirmation | 84030 | 12.3055 | 2.6864 | 1.0684 | 0.2188 | 2.8520 | 0.0000 | 0.0000 | 0.0000 |
| F0_proxy | discovery | 126046 | 13.0361 | 2.7475 | 1.1522 | 0.3272 | 3.0387 | 0.0014 | 0.0010 | 0.0019 |
| F0_proxy | confirmation | 84030 | 12.3055 | 2.6864 | 1.0684 | 0.2190 | 2.8519 | -0.0000 | -0.0004 | 0.0004 |
| F1_event_no_division | discovery | 126046 | 13.0361 | 2.7410 | 1.1375 | 0.3361 | 3.0184 | 0.0079 | 0.0072 | 0.0087 |
| F1_event_no_division | confirmation | 84030 | 12.3055 | 2.6814 | 1.0571 | 0.2269 | 2.8387 | 0.0050 | 0.0043 | 0.0058 |
| F2_plus_division | discovery | 126046 | 13.0361 | 2.7401 | 1.1354 | 0.3374 | 3.0155 | 0.0089 | 0.0081 | 0.0096 |
| F2_plus_division | confirmation | 84030 | 12.3055 | 2.6814 | 1.0571 | 0.2270 | 2.8377 | 0.0051 | 0.0043 | 0.0059 |
| F3_plus_market | discovery | 126046 | 13.0361 | 2.7391 | 1.1332 | 0.3388 | 3.0116 | 0.0098 | 0.0090 | 0.0106 |
| F3_plus_market | confirmation | 84030 | 12.3055 | 2.6809 | 1.0560 | 0.2278 | 2.8367 | 0.0055 | 0.0046 | 0.0064 |

Match totals:

| model | split | n | mean_y | nb_log_score | poisson_deviance | r2 | mae | delta_vs_ref | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_proxy_raw | discovery | 63023 | 26.0721 | 3.1859 | 1.3586 | 0.4183 | 4.6698 | -0.0010 | -0.0014 | -0.0007 |
| C0_proxy_raw | confirmation | 42015 | 24.6109 | 3.0850 | 1.1511 | 0.2893 | 4.2128 | -0.0007 | -0.0011 | -0.0003 |
| C0_proxy_cal | discovery | 63023 | 26.0721 | 3.1848 | 1.3557 | 0.4196 | 4.6584 | 0.0000 | 0.0000 | 0.0000 |
| C0_proxy_cal | confirmation | 42015 | 24.6109 | 3.0842 | 1.1491 | 0.2906 | 4.2025 | 0.0000 | 0.0000 | 0.0000 |
| F0_proxy | discovery | 63023 | 26.0721 | 3.1823 | 1.3485 | 0.4232 | 4.6462 | 0.0025 | 0.0018 | 0.0033 |
| F0_proxy | confirmation | 42015 | 24.6109 | 3.0840 | 1.1484 | 0.2913 | 4.2024 | 0.0002 | -0.0005 | 0.0010 |
| F1_event_no_division | discovery | 63023 | 26.0721 | 3.1734 | 1.3242 | 0.4340 | 4.6015 | 0.0114 | 0.0103 | 0.0126 |
| F1_event_no_division | confirmation | 42015 | 24.6109 | 3.0767 | 1.1289 | 0.3030 | 4.1670 | 0.0075 | 0.0062 | 0.0089 |
| F2_plus_division | discovery | 63023 | 26.0721 | 3.1717 | 1.3195 | 0.4361 | 4.5943 | 0.0131 | 0.0119 | 0.0144 |
| F2_plus_division | confirmation | 42015 | 24.6109 | 3.0766 | 1.1286 | 0.3033 | 4.1634 | 0.0076 | 0.0063 | 0.0090 |
| F3_plus_market | discovery | 63023 | 26.0721 | 3.1710 | 1.3175 | 0.4370 | 4.5890 | 0.0138 | 0.0125 | 0.0151 |
| F3_plus_market | confirmation | 42015 | 24.6109 | 3.0757 | 1.1261 | 0.3049 | 4.1599 | 0.0086 | 0.0072 | 0.0100 |

### 2.5 Does the proxy already absorb the division?

| model | split | n | nb_log_score | r2 | delta_vs_ref | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| F0_proxy_plus_division | discovery | 126046 | 2.7456 | 0.3299 | 0.0033 | 0.0028 | 0.0039 |
| F0_proxy_plus_division | confirmation | 84030 | 2.6866 | 0.2188 | -0.0001 | -0.0007 | 0.0004 |
| F2_plus_division minus F1_event_no_division | discovery | 126046 | 2.7401 | 0.3374 | 0.0009 | 0.0007 | 0.0011 |
| F2_plus_division minus F1_event_no_division | confirmation | 84030 | 2.6814 | 0.2270 | 0.0000 | -0.0003 | 0.0003 |
| F0_proxy_plus_division minus F0_proxy | discovery | 126046 | 2.7456 | 0.3299 | 0.0019 | 0.0016 | 0.0022 |
| F0_proxy_plus_division minus F0_proxy | confirmation | 84030 | 2.6866 | 0.2188 | -0.0001 | -0.0005 | 0.0003 |

Yes. Division identifiers are worth +0.0000 [-0.0003, +0.0003] nats on confirmation on top of the event model, and -0.0001 [-0.0005, +0.0003] on top of the proxy alone. A rolling mean taken inside a division-season is already a division model.

### 2.6 Direct binary classifiers at the standard half-lines

Protocol step 5 asks for both a full count distribution and direct binary models at the half-lines. They agree: the negative binomial built on the count model's mean and a classifier trained directly on the line score within a whisker of each other, so nothing below depends on which is used.

| market | line | model | split | n | base_rate | log_loss | brier | delta_vs_proxy | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| team_fouls | 10.5000 | C0_proxy_cal | discovery | 126046 | 0.6857 | 0.5367 | 0.1815 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 10.5000 | C0_proxy_cal | confirmation | 84030 | 0.6515 | 0.5809 | 0.1987 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 10.5000 | F3_count_nb | discovery | 126046 | 0.6857 | 0.5319 | 0.1797 | 0.0049 | 0.0043 | 0.0054 |
| team_fouls | 10.5000 | F3_count_nb | confirmation | 84030 | 0.6515 | 0.5776 | 0.1974 | 0.0033 | 0.0027 | 0.0040 |
| team_fouls | 10.5000 | F3_direct_binary | discovery | 126046 | 0.6857 | 0.5317 | 0.1796 | 0.0050 | 0.0043 | 0.0056 |
| team_fouls | 10.5000 | F3_direct_binary | confirmation | 84030 | 0.6515 | 0.5779 | 0.1975 | 0.0030 | 0.0022 | 0.0039 |
| team_fouls | 11.5000 | C0_proxy_cal | discovery | 126046 | 0.5978 | 0.5753 | 0.1972 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 11.5000 | C0_proxy_cal | confirmation | 84030 | 0.5519 | 0.6170 | 0.2143 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 11.5000 | F3_count_nb | discovery | 126046 | 0.5978 | 0.5701 | 0.1951 | 0.0053 | 0.0047 | 0.0058 |
| team_fouls | 11.5000 | F3_count_nb | confirmation | 84030 | 0.5519 | 0.6134 | 0.2128 | 0.0035 | 0.0029 | 0.0042 |
| team_fouls | 11.5000 | F3_direct_binary | discovery | 126046 | 0.5978 | 0.5702 | 0.1951 | 0.0052 | 0.0045 | 0.0058 |
| team_fouls | 11.5000 | F3_direct_binary | confirmation | 84030 | 0.5519 | 0.6138 | 0.2130 | 0.0031 | 0.0023 | 0.0040 |
| team_fouls | 12.5000 | C0_proxy_cal | discovery | 126046 | 0.5091 | 0.5839 | 0.2002 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 12.5000 | C0_proxy_cal | confirmation | 84030 | 0.4529 | 0.6154 | 0.2134 | 0.0000 | 0.0000 | 0.0000 |
| team_fouls | 12.5000 | F3_count_nb | discovery | 126046 | 0.5091 | 0.5785 | 0.1980 | 0.0054 | 0.0049 | 0.0060 |
| team_fouls | 12.5000 | F3_count_nb | confirmation | 84030 | 0.4529 | 0.6118 | 0.2119 | 0.0036 | 0.0029 | 0.0043 |
| team_fouls | 12.5000 | F3_direct_binary | discovery | 126046 | 0.5091 | 0.5786 | 0.1980 | 0.0053 | 0.0046 | 0.0060 |
| team_fouls | 12.5000 | F3_direct_binary | confirmation | 84030 | 0.4529 | 0.6121 | 0.2120 | 0.0033 | 0.0025 | 0.0041 |
| match_fouls | 22.5000 | C0_proxy_cal | discovery | 63023 | 0.6525 | 0.5268 | 0.1785 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 22.5000 | C0_proxy_cal | confirmation | 42015 | 0.6114 | 0.5791 | 0.1984 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 22.5000 | F3_count_nb | discovery | 63023 | 0.6525 | 0.5191 | 0.1757 | 0.0076 | 0.0068 | 0.0085 |
| match_fouls | 22.5000 | F3_count_nb | confirmation | 42015 | 0.6114 | 0.5725 | 0.1958 | 0.0066 | 0.0055 | 0.0077 |
| match_fouls | 22.5000 | F3_direct_binary | discovery | 63023 | 0.6525 | 0.5191 | 0.1757 | 0.0076 | 0.0065 | 0.0088 |
| match_fouls | 22.5000 | F3_direct_binary | confirmation | 42015 | 0.6114 | 0.5748 | 0.1968 | 0.0043 | 0.0027 | 0.0059 |
| match_fouls | 23.5000 | C0_proxy_cal | discovery | 63023 | 0.5984 | 0.5429 | 0.1846 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 23.5000 | C0_proxy_cal | confirmation | 42015 | 0.5456 | 0.5947 | 0.2048 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 23.5000 | F3_count_nb | discovery | 63023 | 0.5984 | 0.5349 | 0.1816 | 0.0080 | 0.0071 | 0.0090 |
| match_fouls | 23.5000 | F3_count_nb | confirmation | 42015 | 0.5456 | 0.5881 | 0.2022 | 0.0066 | 0.0054 | 0.0078 |
| match_fouls | 23.5000 | F3_direct_binary | discovery | 63023 | 0.5984 | 0.5351 | 0.1817 | 0.0078 | 0.0066 | 0.0090 |
| match_fouls | 23.5000 | F3_direct_binary | confirmation | 42015 | 0.5456 | 0.5903 | 0.2031 | 0.0044 | 0.0028 | 0.0059 |
| match_fouls | 24.5000 | C0_proxy_cal | discovery | 63023 | 0.5417 | 0.5491 | 0.1863 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 24.5000 | C0_proxy_cal | confirmation | 42015 | 0.4795 | 0.5942 | 0.2042 | 0.0000 | 0.0000 | 0.0000 |
| match_fouls | 24.5000 | F3_count_nb | discovery | 63023 | 0.5417 | 0.5413 | 0.1834 | 0.0078 | 0.0068 | 0.0087 |
| match_fouls | 24.5000 | F3_count_nb | confirmation | 42015 | 0.4795 | 0.5881 | 0.2018 | 0.0061 | 0.0050 | 0.0073 |
| match_fouls | 24.5000 | F3_direct_binary | discovery | 63023 | 0.5417 | 0.5412 | 0.1834 | 0.0079 | 0.0067 | 0.0090 |
| match_fouls | 24.5000 | F3_direct_binary | confirmation | 42015 | 0.4795 | 0.5901 | 0.2027 | 0.0041 | 0.0026 | 0.0056 |

### 2.7 Refit-noise floor

| seed | split | model | delta_vs_proxy | nb_log_score | spread |
|---|---|---|---|---|---|
| 20260918 | discovery | F3_plus_market | 0.00981 | 2.73913 | 0.00032 |
| 20260918 | confirmation | F3_plus_market | 0.00551 | 2.68093 | 0.00019 |
| 20260919 | discovery | F3_plus_market | 0.01005 | 2.73890 | 0.00032 |
| 20260919 | confirmation | F3_plus_market | 0.00556 | 2.68088 | 0.00019 |
| 20260920 | discovery | F3_plus_market | 0.00996 | 2.73899 | 0.00032 |
| 20260920 | confirmation | F3_plus_market | 0.00542 | 2.68102 | 0.00019 |
| 20260921 | discovery | F3_plus_market | 0.00973 | 2.73922 | 0.00032 |
| 20260921 | confirmation | F3_plus_market | 0.00540 | 2.68104 | 0.00019 |
| 20260922 | discovery | F3_plus_market | 0.00987 | 2.73908 | 0.00032 |
| 20260922 | confirmation | F3_plus_market | 0.00558 | 2.68085 | 0.00019 |

## 3. (b) The player layer: fouls committed and fouls won

Two proxies are reported. `P0_proxy` is the naive one a lazy side market is hung off -- the player's shrunk prior rate per 90 times his expected minutes. `P0_proxy_strong` is what protocol step 3 asks for, and mirrors the card stage's: the same rate and minutes multiplied by an opponent factor (the mirror statistic -- how many fouls the opponent *draws* when the target is fouls committed, how many it *gives away* when the target is fouls won) and a referee factor. Both are recalibrated on discovery by a one-dimensional Poisson GLM, so neither is beaten on scale alone. The opponent factor is deliberately kept out of the modelling ladder's P0/P1 blocks, so that the matchup channel measured in 3.1 is not partly inside its own reference. The ladder then adds the player's own rolling block and team context (`P1`), the referee (`P1r`), the matchup blocks (`P2`), and -- in section 4 -- the imputed defensive state (`P3`).

| market | model | split | n | mean_y | nb_log_score | r2 | mae | delta_vs_ref | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| player_fouls | P0_proxy_raw | discovery | 19359 | 1.1967 | 1.4460 | 0.0561 | 0.9276 | -0.0054 | -0.0071 | -0.0038 |
| player_fouls | P0_proxy_raw | confirmation | 13970 | 1.1555 | 1.4052 | 0.0824 | 0.9031 | 0.0004 | -0.0013 | 0.0021 |
| player_fouls | P0_proxy_cal | discovery | 19359 | 1.1967 | 1.4406 | 0.0682 | 0.9467 | 0.0000 | 0.0000 | 0.0000 |
| player_fouls | P0_proxy_cal | confirmation | 13970 | 1.1555 | 1.4057 | 0.0811 | 0.9238 | 0.0000 | 0.0000 | 0.0000 |
| player_fouls | P0_proxy_strong_raw | discovery | 19359 | 1.1967 | 1.4404 | 0.0690 | 0.9270 | 0.0002 | -0.0024 | 0.0029 |
| player_fouls | P0_proxy_strong_raw | confirmation | 13970 | 1.1555 | 1.4096 | 0.0753 | 0.9173 | -0.0040 | -0.0072 | -0.0009 |
| player_fouls | P0_proxy_strong_cal | discovery | 19359 | 1.1967 | 1.4354 | 0.0768 | 0.9426 | 0.0052 | 0.0032 | 0.0074 |
| player_fouls | P0_proxy_strong_cal | confirmation | 13970 | 1.1555 | 1.4101 | 0.0696 | 0.9341 | -0.0045 | -0.0072 | -0.0017 |
| player_fouls | P0_proxy | discovery | 19359 | 1.1967 | 1.3878 | 0.1517 | 0.9011 | 0.0528 | 0.0484 | 0.0572 |
| player_fouls | P0_proxy | confirmation | 13970 | 1.1555 | 1.3555 | 0.1666 | 0.8694 | 0.0501 | 0.0450 | 0.0553 |
| player_fouls | P1_event | discovery | 19359 | 1.1967 | 1.3832 | 0.1613 | 0.8952 | 0.0574 | 0.0530 | 0.0618 |
| player_fouls | P1_event | confirmation | 13970 | 1.1555 | 1.3550 | 0.1688 | 0.8706 | 0.0507 | 0.0454 | 0.0559 |
| player_fouls | P1r_event_plus_referee | discovery | 19359 | 1.1967 | 1.3827 | 0.1626 | 0.8945 | 0.0579 | 0.0536 | 0.0624 |
| player_fouls | P1r_event_plus_referee | confirmation | 13970 | 1.1555 | 1.3549 | 0.1695 | 0.8715 | 0.0508 | 0.0454 | 0.0560 |
| player_fouls | P2_plus_matchup | discovery | 19359 | 1.1967 | 1.3803 | 0.1678 | 0.8922 | 0.0603 | 0.0558 | 0.0647 |
| player_fouls | P2_plus_matchup | confirmation | 13970 | 1.1555 | 1.3549 | 0.1697 | 0.8721 | 0.0508 | 0.0453 | 0.0561 |
| player_fouls | P1r_plus_dribble_side | discovery | 19359 | 1.1967 | 1.3822 | 0.1639 | 0.8941 | 0.0584 | 0.0540 | 0.0628 |
| player_fouls | P1r_plus_dribble_side | confirmation | 13970 | 1.1555 | 1.3546 | 0.1705 | 0.8710 | 0.0511 | 0.0456 | 0.0562 |
| player_fouls | P1r_plus_opponent_foul | discovery | 19359 | 1.1967 | 1.3806 | 0.1670 | 0.8923 | 0.0600 | 0.0556 | 0.0644 |
| player_fouls | P1r_plus_opponent_foul | confirmation | 13970 | 1.1555 | 1.3548 | 0.1696 | 0.8718 | 0.0509 | 0.0454 | 0.0563 |
| player_fouls_won | P0_proxy_raw | discovery | 19359 | 1.1438 | 1.4128 | 0.1028 | 0.9299 | -0.0073 | -0.0092 | -0.0054 |
| player_fouls_won | P0_proxy_raw | confirmation | 13970 | 1.0894 | 1.3619 | 0.1401 | 0.8989 | 0.0010 | -0.0013 | 0.0032 |
| player_fouls_won | P0_proxy_cal | discovery | 19359 | 1.1438 | 1.4055 | 0.1287 | 0.9394 | 0.0000 | 0.0000 | 0.0000 |
| player_fouls_won | P0_proxy_cal | confirmation | 13970 | 1.0894 | 1.3629 | 0.1344 | 0.9212 | 0.0000 | 0.0000 | 0.0000 |
| player_fouls_won | P0_proxy_strong_raw | discovery | 19359 | 1.1438 | 1.4091 | 0.1144 | 0.9294 | -0.0036 | -0.0064 | -0.0010 |
| player_fouls_won | P0_proxy_strong_raw | confirmation | 13970 | 1.0894 | 1.3649 | 0.1417 | 0.9103 | -0.0021 | -0.0056 | 0.0013 |
| player_fouls_won | P0_proxy_strong_cal | discovery | 19359 | 1.1438 | 1.4052 | 0.1269 | 0.9415 | 0.0002 | -0.0022 | 0.0026 |
| player_fouls_won | P0_proxy_strong_cal | confirmation | 13970 | 1.0894 | 1.3706 | 0.1235 | 0.9361 | -0.0077 | -0.0113 | -0.0043 |
| player_fouls_won | P0_proxy | discovery | 19359 | 1.1438 | 1.3467 | 0.2177 | 0.8858 | 0.0587 | 0.0534 | 0.0639 |
| player_fouls_won | P0_proxy | confirmation | 13970 | 1.0894 | 1.3052 | 0.2250 | 0.8580 | 0.0577 | 0.0515 | 0.0642 |
| player_fouls_won | P1_event | discovery | 19359 | 1.1438 | 1.3415 | 0.2294 | 0.8794 | 0.0640 | 0.0586 | 0.0690 |
| player_fouls_won | P1_event | confirmation | 13970 | 1.0894 | 1.3044 | 0.2272 | 0.8602 | 0.0584 | 0.0521 | 0.0647 |
| player_fouls_won | P1r_event_plus_referee | discovery | 19359 | 1.1438 | 1.3404 | 0.2332 | 0.8777 | 0.0650 | 0.0597 | 0.0702 |
| player_fouls_won | P1r_event_plus_referee | confirmation | 13970 | 1.0894 | 1.3034 | 0.2278 | 0.8603 | 0.0595 | 0.0531 | 0.0656 |
| player_fouls_won | P2_plus_matchup | discovery | 19359 | 1.1438 | 1.3390 | 0.2362 | 0.8754 | 0.0664 | 0.0610 | 0.0717 |
| player_fouls_won | P2_plus_matchup | confirmation | 13970 | 1.0894 | 1.3010 | 0.2352 | 0.8576 | 0.0618 | 0.0555 | 0.0682 |
| player_fouls_won | P1r_plus_dribble_side | discovery | 19359 | 1.1438 | 1.3403 | 0.2332 | 0.8776 | 0.0652 | 0.0597 | 0.0704 |
| player_fouls_won | P1r_plus_dribble_side | confirmation | 13970 | 1.0894 | 1.3021 | 0.2334 | 0.8565 | 0.0608 | 0.0544 | 0.0668 |
| player_fouls_won | P1r_plus_opponent_foul | discovery | 19359 | 1.1438 | 1.3392 | 0.2360 | 0.8758 | 0.0663 | 0.0609 | 0.0717 |
| player_fouls_won | P1r_plus_opponent_foul | confirmation | 13970 | 1.0894 | 1.3010 | 0.2356 | 0.8557 | 0.0619 | 0.0554 | 0.0682 |

The richer proxy does not survive its own confirmation split. Relative to the naive one it is worth -0.0045 [-0.0072, -0.0017] nats per appearance on fouls committed and -0.0077 [-0.0113, -0.0043] on fouls won -- both negative, after being positive on discovery (+0.00525 and +0.00025). Multiplying a player's rate by an opponent factor and a referee factor is the right idea in the wrong functional form. The naive proxy is therefore the *harder* reference out of sample, and it is the one every delta in this section is measured against; against the richer proxy the full model wins by more, +0.0553 [+0.0501, +0.0606] on fouls committed and +0.0695 [+0.0633, +0.0760] on fouls won. Either way the margin is not an artefact of choosing the weakest available rolling mean.

A caution that the ladder makes plain: most of the gap to the proxy is functional form, not information. On player fouls the `P0_proxy` model -- LightGBM on the proxy's own inputs plus the position-group backstop and the starter flag -- already beats the calibrated proxy by +0.0501 nats, and the entire event, referee and matchup apparatus adds +0.0006 more. A rate times expected minutes is simply the wrong shape for a count, and a booster fixes that before it learns anything about football.

### 3.1 The two matchup channels, added one at a time

| market | step | split | n | delta_nats_per_appearance | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| player_fouls | P0_proxy_cal -> P0_proxy_strong_cal | discovery | 19359 | 0.00525 | 0.00320 | 0.00737 |
| player_fouls | P0_proxy_cal -> P0_proxy_strong_cal | confirmation | 13970 | -0.00447 | -0.00724 | -0.00170 |
| player_fouls | P0_proxy_strong_cal -> P2_plus_matchup | discovery | 19359 | 0.05505 | 0.05074 | 0.05916 |
| player_fouls | P0_proxy_strong_cal -> P2_plus_matchup | confirmation | 13970 | 0.05526 | 0.05005 | 0.06065 |
| player_fouls | P1_event -> P1r_event_plus_referee | discovery | 19359 | 0.00056 | -0.00002 | 0.00118 |
| player_fouls | P1_event -> P1r_event_plus_referee | confirmation | 13970 | 0.00012 | -0.00049 | 0.00074 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | discovery | 19359 | 0.00049 | 0.00003 | 0.00097 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | confirmation | 13970 | 0.00028 | -0.00028 | 0.00084 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | discovery | 19359 | 0.00209 | 0.00121 | 0.00300 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | confirmation | 13970 | 0.00010 | -0.00097 | 0.00110 |
| player_fouls | P1r_event_plus_referee -> P2_plus_matchup | discovery | 19359 | 0.00235 | 0.00131 | 0.00335 |
| player_fouls | P1r_event_plus_referee -> P2_plus_matchup | confirmation | 13970 | -0.00000 | -0.00118 | 0.00107 |
| player_fouls_won | P0_proxy_cal -> P0_proxy_strong_cal | discovery | 19359 | 0.00025 | -0.00222 | 0.00263 |
| player_fouls_won | P0_proxy_cal -> P0_proxy_strong_cal | confirmation | 13970 | -0.00770 | -0.01130 | -0.00430 |
| player_fouls_won | P0_proxy_strong_cal -> P2_plus_matchup | discovery | 19359 | 0.06617 | 0.06077 | 0.07146 |
| player_fouls_won | P0_proxy_strong_cal -> P2_plus_matchup | confirmation | 13970 | 0.06953 | 0.06332 | 0.07602 |
| player_fouls_won | P1_event -> P1r_event_plus_referee | discovery | 19359 | 0.00107 | 0.00009 | 0.00212 |
| player_fouls_won | P1_event -> P1r_event_plus_referee | confirmation | 13970 | 0.00109 | 0.00008 | 0.00218 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | discovery | 19359 | 0.00016 | -0.00071 | 0.00105 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | confirmation | 13970 | 0.00126 | 0.00030 | 0.00218 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | discovery | 19359 | 0.00126 | -0.00015 | 0.00255 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | confirmation | 13970 | 0.00234 | 0.00111 | 0.00364 |
| player_fouls_won | P1r_event_plus_referee -> P2_plus_matchup | discovery | 19359 | 0.00138 | -0.00007 | 0.00270 |
| player_fouls_won | P1r_event_plus_referee -> P2_plus_matchup | confirmation | 13970 | 0.00232 | 0.00096 | 0.00367 |

Refit-noise floor over 5 seeds. Each row refits the named quantity end to end at the *tuned* parameters and re-measures it, so the spread is the spread of the number the text compares, not of some other model's. Two quantities are measured because two different claims are made: the ladder claim (`P0_proxy_cal -> P2_plus_matchup`, how far the full model is ahead of the proxy) and the channel claim (`P1r_event_plus_referee -> P1r_plus_<channel>`, what one block adds). The two spreads are of similar absolute size -- on fouls won the ladder delta moves 0.00066 nats across seeds, the opponent-foul increment 0.00078 and the dribble-side increment 0.00073 -- and that is exactly why they must not be swapped, because the claims they support differ by two orders of magnitude: the ladder delta is 0.0618 nats and clears its own spread about 93x, while the opponent-foul increment is 0.00234 and clears the same-sized spread about 3x. An earlier version of this stage quoted the ladder spread -- and, worse, an *untuned* refit of it, because the tuned parameters were never passed through to the refit loop -- as the floor for the channel claim, which made a 3x result read as 9x.

**Five seeds here, not the three the other stages use.** The channel increments are a few ten-thousandths of a nat and a three-seed range is a thin estimate of their spread: on the first three seeds alone the opponent-foul increment on fouls won spans only 0.00007 nats and the same claim would have read as 34x noise. Two more refits put it at 0.00078 and 3x. The lesson is general and is carried into the sweep report as well: a three-seed range can understate a spread by an order of magnitude, so a refit floor is a screen for downgrading claims, never a certificate.

| market | quantity | seed | split | n | delta | refit_spread |
|---|---|---|---|---|---|---|
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260918 | discovery | 19359 | 0.06030 | 0.00101 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260918 | discovery | 19359 | 0.00049 | 0.00034 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260918 | discovery | 19359 | 0.00209 | 0.00060 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260918 | confirmation | 13970 | 0.05079 | 0.00085 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260918 | confirmation | 13970 | 0.00028 | 0.00082 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260918 | confirmation | 13970 | 0.00010 | 0.00073 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260919 | discovery | 19359 | 0.06051 | 0.00101 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260919 | discovery | 19359 | 0.00056 | 0.00034 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260919 | discovery | 19359 | 0.00198 | 0.00060 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260919 | confirmation | 13970 | 0.05053 | 0.00085 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260919 | confirmation | 13970 | -0.00054 | 0.00082 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260919 | confirmation | 13970 | -0.00055 | 0.00073 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260920 | discovery | 19359 | 0.05950 | 0.00101 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260920 | discovery | 19359 | 0.00068 | 0.00034 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260920 | discovery | 19359 | 0.00204 | 0.00060 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260920 | confirmation | 13970 | 0.04993 | 0.00085 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260920 | confirmation | 13970 | 0.00024 | 0.00082 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260920 | confirmation | 13970 | 0.00015 | 0.00073 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260921 | discovery | 19359 | 0.06012 | 0.00101 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260921 | discovery | 19359 | 0.00047 | 0.00034 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260921 | discovery | 19359 | 0.00257 | 0.00060 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260921 | confirmation | 13970 | 0.05046 | 0.00085 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260921 | confirmation | 13970 | 0.00023 | 0.00082 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260921 | confirmation | 13970 | 0.00018 | 0.00073 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260922 | discovery | 19359 | 0.06046 | 0.00101 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260922 | discovery | 19359 | 0.00033 | 0.00034 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260922 | discovery | 19359 | 0.00207 | 0.00060 |
| player_fouls | P0_proxy_cal -> P2_plus_matchup | 20260922 | confirmation | 13970 | 0.05047 | 0.00085 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260922 | confirmation | 13970 | -0.00015 | 0.00082 |
| player_fouls | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260922 | confirmation | 13970 | -0.00029 | 0.00073 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260918 | discovery | 19359 | 0.06642 | 0.00118 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260918 | discovery | 19359 | 0.00016 | 0.00112 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260918 | discovery | 19359 | 0.00126 | 0.00064 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260918 | confirmation | 13970 | 0.06183 | 0.00066 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260918 | confirmation | 13970 | 0.00126 | 0.00073 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260918 | confirmation | 13970 | 0.00234 | 0.00078 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260919 | discovery | 19359 | 0.06760 | 0.00118 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260919 | discovery | 19359 | 0.00056 | 0.00112 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260919 | discovery | 19359 | 0.00129 | 0.00064 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260919 | confirmation | 13970 | 0.06179 | 0.00066 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260919 | confirmation | 13970 | 0.00077 | 0.00073 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260919 | confirmation | 13970 | 0.00233 | 0.00078 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260920 | discovery | 19359 | 0.06649 | 0.00118 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260920 | discovery | 19359 | 0.00093 | 0.00112 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260920 | discovery | 19359 | 0.00119 | 0.00064 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260920 | confirmation | 13970 | 0.06163 | 0.00066 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260920 | confirmation | 13970 | 0.00129 | 0.00073 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260920 | confirmation | 13970 | 0.00240 | 0.00078 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260921 | discovery | 19359 | 0.06648 | 0.00118 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260921 | discovery | 19359 | 0.00054 | 0.00112 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260921 | discovery | 19359 | 0.00183 | 0.00064 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260921 | confirmation | 13970 | 0.06229 | 0.00066 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260921 | confirmation | 13970 | 0.00096 | 0.00073 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260921 | confirmation | 13970 | 0.00191 | 0.00078 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260922 | discovery | 19359 | 0.06757 | 0.00118 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260922 | discovery | 19359 | 0.00128 | 0.00112 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260922 | discovery | 19359 | 0.00151 | 0.00064 |
| player_fouls_won | P0_proxy_cal -> P2_plus_matchup | 20260922 | confirmation | 13970 | 0.06192 | 0.00066 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_dribble_side | 20260922 | confirmation | 13970 | 0.00056 | 0.00073 |
| player_fouls_won | P1r_event_plus_referee -> P1r_plus_opponent_foul | 20260922 | confirmation | 13970 | 0.00269 | 0.00078 |

The asymmetry is the finding of this section, and it is an asymmetry of the *target*, not of the channel. On fouls won, the opponent-foul block is worth +0.0023 [+0.0011, +0.0036] against a 0.00078-nat spread of that same increment (3.0x), and the dribble-side block +0.0013 [+0.0003, +0.0022] against 0.00073 (1.7x) -- both clear of zero. On fouls committed, the same two blocks are worth +0.0003 [-0.0003, +0.0008] (spread 0.00082) and +0.0001 [-0.0010, +0.0011] (spread 0.00073) -- neither clear of zero, and both inside the noise. Who a player is up against moves how many fouls he draws; it does not measurably move how many he commits.

**How much weight this carries.** Less than a first reading suggests, and the same standard is applied here as to the direction that failed. Both fouls-won channels were *inside* noise on discovery -- opponent-foul +0.00126 [-0.00015, +0.00255], dribble-side +0.00016 [-0.00071, +0.00105] -- and cleared zero only on confirmation. Four (channel, target) cells were tested. So this is not a discovery finding replicated on confirmation; it is a confirmation finding whose discovery half agreed in sign but not in significance. What it rests on is sign consistency: the same sign in both halves and in every refit seed, in the direction the mechanism predicts, while the mirror direction is flat in both halves.

Note also how the two halves disagree for fouls committed: the opponent-foul block was worth +0.00209 [+0.00121, +0.00300] on discovery, an interval comfortably clear of zero, and collapsed to nothing on confirmation. That is exactly the multiplicity the split exists to catch, and it is the reason a discovery-only version of this stage would have reported a matchup effect in both directions.

### 3.2 Is the mechanism there at all? Observed rates

| contrast | line | opp_flank_dribbles_t | own_dribble_t | opp_foul_rate_t | ref_foul_t | n | mean | ci_lo | ci_hi | mean_minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| fouls_by_line_x_opponent_flank_dribbles | DEF | high | n/a | n/a | n/a | 4855 | 1.0958 | 1.0623 | 1.1267 | 86.2956 |
| fouls_by_line_x_opponent_flank_dribbles | DEF | low | n/a | n/a | n/a | 2481 | 1.1125 | 1.0682 | 1.1576 | 86.2994 |
| fouls_by_line_x_opponent_flank_dribbles | DEF | mid | n/a | n/a | n/a | 4352 | 1.1119 | 1.0818 | 1.1433 | 84.9730 |
| fouls_by_line_x_opponent_flank_dribbles | FWD | high | n/a | n/a | n/a | 2874 | 1.0456 | 1.0017 | 1.0942 | 63.9657 |
| fouls_by_line_x_opponent_flank_dribbles | FWD | low | n/a | n/a | n/a | 3809 | 1.1378 | 1.0996 | 1.1780 | 66.5801 |
| fouls_by_line_x_opponent_flank_dribbles | FWD | mid | n/a | n/a | n/a | 2543 | 1.0735 | 1.0263 | 1.1221 | 64.1767 |
| fouls_by_line_x_opponent_flank_dribbles | MID | high | n/a | n/a | n/a | 4049 | 1.3564 | 1.3165 | 1.3951 | 70.5676 |
| fouls_by_line_x_opponent_flank_dribbles | MID | low | n/a | n/a | n/a | 4867 | 1.2712 | 1.2368 | 1.3063 | 69.8058 |
| fouls_by_line_x_opponent_flank_dribbles | MID | mid | n/a | n/a | n/a | 3499 | 1.3267 | 1.2834 | 1.3727 | 70.6537 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | high | high | n/a | 4667 | 1.5485 | 1.5069 | 1.5980 | 68.8122 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | high | low | n/a | 4201 | 1.2471 | 1.2069 | 1.2865 | 70.8403 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | high | mid | n/a | 3068 | 1.4286 | 1.3805 | 1.4822 | 70.0115 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | low | high | n/a | 4677 | 0.9972 | 0.9634 | 1.0295 | 82.4649 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | low | low | n/a | 4122 | 0.7057 | 0.6801 | 0.7317 | 83.4177 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | low | mid | n/a | 3103 | 0.8582 | 0.8176 | 0.8949 | 82.1541 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | mid | high | n/a | 3539 | 1.2235 | 1.1771 | 1.2700 | 68.9781 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | mid | low | n/a | 3010 | 0.8900 | 0.8483 | 0.9278 | 68.0563 |
| fouls_won_by_own_dribbling_x_opponent_foul_rate | n/a | n/a | mid | mid | n/a | 2942 | 1.1105 | 1.0640 | 1.1558 | 71.1232 |
| fouls_by_referee_tercile | n/a | n/a | n/a | n/a | high | 11681 | 1.2993 | 1.2770 | 1.3228 | 73.4801 |
| fouls_by_referee_tercile | n/a | n/a | n/a | n/a | low | 10932 | 1.0253 | 1.0027 | 1.0463 | 74.9740 |
| fouls_by_referee_tercile | n/a | n/a | n/a | n/a | mid | 10716 | 1.2060 | 1.1788 | 1.2315 | 74.4937 |
| fouls_won_by_referee_tercile | n/a | n/a | n/a | n/a | high | 11681 | 1.2342 | 1.2107 | 1.2574 | 73.4801 |
| fouls_won_by_referee_tercile | n/a | n/a | n/a | n/a | low | 10932 | 0.9745 | 0.9536 | 0.9952 | 74.9740 |
| fouls_won_by_referee_tercile | n/a | n/a | n/a | n/a | mid | 10716 | 1.1470 | 1.1211 | 1.1707 | 74.4937 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | high | 4429 | 1.2775 | 1.2433 | 1.3186 | 73.2689 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | low | 524 | 1.1088 | 1.0208 | 1.2214 | 74.2490 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | mid | 3427 | 1.2419 | 1.2042 | 1.2891 | 73.7587 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | high | 2133 | 1.2893 | 1.2309 | 1.3422 | 73.6977 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | low | 2661 | 1.1052 | 1.0513 | 1.1507 | 74.1995 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | mid | 3435 | 1.1499 | 1.0974 | 1.1960 | 74.5065 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | high | 23 | 0.8696 | 0.8696 | 0.8696 | 73.8572 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | low | 7510 | 0.9866 | 0.9661 | 1.0098 | 75.3645 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | mid | 790 | 1.0228 | 0.9455 | 1.1090 | 77.5592 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | high | 5096 | 1.3244 | 1.2914 | 1.3607 | 73.5708 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | low | 237 | 1.1730 | 1.0279 | 1.3036 | 72.8968 |
| fouls_by_competition_x_referee_tercile | n/a | n/a | n/a | n/a | mid | 3064 | 1.2758 | 1.2344 | 1.3246 | 74.5109 |

Defenders facing top-tercile flank dribbling commit *fewer* fouls than those facing bottom-tercile flank dribbling, not more. Fouls won, by contrast, move monotonically in both the player's own dribbling and the opponent's foul rate.

### 3.3 The referee channel: a referee effect or a league effect?

The referee's prior foul rate is cut into terciles at roughly 28 and 31 fouls per match, which is very nearly the difference between the Premier League and Serie A. Two checks separate the two explanations.

| split | n | n_referees | r2_raw | r2_within_competition | sd_referee_prior | sd_referee_prior_within_competition |
|---|---|---|---|---|---|---|
| discovery | 1667 | 90 | 0.0672 | 0.0266 | 1.4385 | 1.0560 |
| confirmation | 1034 | 89 | 0.0955 | 0.0247 | 1.7224 | 1.0266 |

| competition | ref_foul_t | n | mean | ci_lo | ci_hi |
|---|---|---|---|---|---|
| La Liga | high | 4429 | 1.2775 | 1.2433 | 1.3186 |
| La Liga | low | 524 | 1.1088 | 1.0208 | 1.2214 |
| La Liga | mid | 3427 | 1.2419 | 1.2042 | 1.2891 |
| Ligue 1 | high | 2133 | 1.2893 | 1.2309 | 1.3422 |
| Ligue 1 | low | 2661 | 1.1052 | 1.0513 | 1.1507 |
| Ligue 1 | mid | 3435 | 1.1499 | 1.0974 | 1.1960 |
| Premier League | high | 23 | 0.8696 | 0.8696 | 0.8696 |
| Premier League | low | 7510 | 0.9866 | 0.9661 | 1.0098 |
| Premier League | mid | 790 | 1.0228 | 0.9455 | 1.1090 |
| Serie A | high | 5096 | 1.3244 | 1.2914 | 1.3607 |
| Serie A | low | 237 | 1.1730 | 1.0279 | 1.3036 |
| Serie A | mid | 3064 | 1.2758 | 1.2344 | 1.3246 |

And the model ablation, in which the referee block is added to an event model stripped of its competition feature:

| market | split | n | step | delta_nats_per_appearance | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| player_fouls | discovery | 19359 | P1_event_without_competition -> + referee block | 0.00128 | 0.00052 | 0.00205 |
| player_fouls | confirmation | 13970 | P1_event_without_competition -> + referee block | 0.00009 | -0.00064 | 0.00081 |
| player_fouls_won | discovery | 19359 | P1_event_without_competition -> + referee block | 0.00093 | 0.00033 | 0.00155 |
| player_fouls_won | confirmation | 13970 | P1_event_without_competition -> + referee block | 0.00076 | 0.00005 | 0.00150 |

| split | n | r2_referee_prior_on_team_fouls | corr | sd_team_fouls |
|---|---|---|---|---|
| discovery | 1667 | 0.0672 | 0.3301 | 4.9954 |
| confirmation | 1034 | 0.0955 | 0.3359 | 4.4858 |

## 4. (c) The imputed-state layer

Prior-match aggregates of the programme's E2 student predictions (nearest-opponent distance, opponents within 5m, block depth, counter-attack flag) for both teams, never the 360 truth. Usable on 33329 of 33329 rows (100.0%), so the like-for-like comparison is the whole universe.

| market | model | split | n | nb_log_score | r2 | mae | delta_vs_ref | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| player_fouls_state_subset | P0_proxy_cal | discovery | 19359 | 1.44063 | 0.06825 | 0.94669 | 0.00000 | 0.00000 | 0.00000 |
| player_fouls_state_subset | P0_proxy_cal | confirmation | 13970 | 1.40566 | 0.08110 | 0.92384 | 0.00000 | 0.00000 | 0.00000 |
| player_fouls_state_subset | P2_plus_matchup | discovery | 19359 | 1.38032 | 0.16779 | 0.89224 | 0.06030 | 0.05577 | 0.06474 |
| player_fouls_state_subset | P2_plus_matchup | confirmation | 13970 | 1.35487 | 0.16970 | 0.87212 | 0.05079 | 0.04525 | 0.05615 |
| player_fouls_state_subset | P3_plus_state | discovery | 19359 | 1.38038 | 0.16754 | 0.89212 | 0.06025 | 0.05577 | 0.06474 |
| player_fouls_state_subset | P3_plus_state | confirmation | 13970 | 1.35503 | 0.16909 | 0.87230 | 0.05063 | 0.04507 | 0.05604 |
| player_fouls_state_subset | P0_proxy_plus_state | discovery | 19359 | 1.43264 | 0.07990 | 0.94166 | 0.00799 | 0.00573 | 0.01014 |
| player_fouls_state_subset | P0_proxy_plus_state | confirmation | 13970 | 1.40333 | 0.08409 | 0.92250 | 0.00233 | 0.00001 | 0.00460 |
| player_fouls_state_subset | P0_proxy_features_only | discovery | 19359 | 1.43406 | 0.07728 | 0.94430 | 0.00657 | 0.00444 | 0.00855 |
| player_fouls_state_subset | P0_proxy_features_only | confirmation | 13970 | 1.40276 | 0.08467 | 0.91780 | 0.00290 | 0.00062 | 0.00509 |
| player_fouls_state_subset | P3_plus_state minus P2_plus_matchup | discovery | 19359 | 1.38038 | 0.16754 | 0.89212 | -0.00005 | -0.00042 | 0.00034 |
| player_fouls_state_subset | P3_plus_state minus P2_plus_matchup | confirmation | 13970 | 1.35503 | 0.16909 | 0.87230 | -0.00016 | -0.00057 | 0.00025 |
| player_fouls_state_subset | P0_proxy_plus_state minus P0_proxy_features_only | discovery | 19359 | 1.43264 | 0.07990 | 0.94166 | 0.00142 | 0.00045 | 0.00238 |
| player_fouls_state_subset | P0_proxy_plus_state minus P0_proxy_features_only | confirmation | 13970 | 1.40333 | 0.08409 | 0.92250 | -0.00057 | -0.00163 | 0.00045 |
| player_fouls_won_state_subset | P0_proxy_cal | discovery | 19359 | 1.40547 | 0.12875 | 0.93939 | 0.00000 | 0.00000 | 0.00000 |
| player_fouls_won_state_subset | P0_proxy_cal | confirmation | 13970 | 1.36286 | 0.13441 | 0.92123 | 0.00000 | 0.00000 | 0.00000 |
| player_fouls_won_state_subset | P2_plus_matchup | discovery | 19359 | 1.33693 | 0.23818 | 0.87747 | 0.06853 | 0.06344 | 0.07356 |
| player_fouls_won_state_subset | P2_plus_matchup | confirmation | 13970 | 1.30173 | 0.23282 | 0.86519 | 0.06113 | 0.05513 | 0.06678 |
| player_fouls_won_state_subset | P3_plus_state | discovery | 19359 | 1.33734 | 0.23755 | 0.87799 | 0.06813 | 0.06299 | 0.07313 |
| player_fouls_won_state_subset | P3_plus_state | confirmation | 13970 | 1.30166 | 0.23322 | 0.86428 | 0.06120 | 0.05520 | 0.06667 |
| player_fouls_won_state_subset | P0_proxy_plus_state | discovery | 19359 | 1.38845 | 0.15059 | 0.93490 | 0.01702 | 0.01419 | 0.01990 |
| player_fouls_won_state_subset | P0_proxy_plus_state | confirmation | 13970 | 1.35331 | 0.14153 | 0.92605 | 0.00955 | 0.00615 | 0.01282 |
| player_fouls_won_state_subset | P0_proxy_features_only | discovery | 19359 | 1.38869 | 0.14989 | 0.93484 | 0.01678 | 0.01392 | 0.01957 |
| player_fouls_won_state_subset | P0_proxy_features_only | confirmation | 13970 | 1.35272 | 0.14374 | 0.92022 | 0.01014 | 0.00686 | 0.01337 |
| player_fouls_won_state_subset | P3_plus_state minus P2_plus_matchup | discovery | 19359 | 1.33734 | 0.23755 | 0.87799 | -0.00040 | -0.00077 | -0.00004 |
| player_fouls_won_state_subset | P3_plus_state minus P2_plus_matchup | confirmation | 13970 | 1.30166 | 0.23322 | 0.86428 | 0.00007 | -0.00034 | 0.00050 |
| player_fouls_won_state_subset | P0_proxy_plus_state minus P0_proxy_features_only | discovery | 19359 | 1.38845 | 0.15059 | 0.93490 | 0.00024 | -0.00045 | 0.00086 |
| player_fouls_won_state_subset | P0_proxy_plus_state minus P0_proxy_features_only | confirmation | 13970 | 1.35331 | 0.14153 | 0.92605 | -0.00059 | -0.00138 | 0.00016 |

Added to the rich model the state is worth nothing; added to a bare proxy model it is worth nothing either. This is the same null the programme found for xG and for pass completion, in a fourth market.

## 5. (d) The betting simulation and the haircut that governs it

### 5.1 Protocol step 4: how much better is a real book than this rolling mean?

We have no foul lines, only goals lines. So the identical proxy machinery is pointed at total goals on the *same* matches, turned into a probability at 2.5 (negative binomial plus a Platt map, both fitted on discovery) and compared with Bet365's no-vig over/under price.

| split | model | n | log_loss | brier | delta_vs_proxy_cal | ci_lo | ci_hi | mean_p | obs_rate |
|---|---|---|---|---|---|---|---|---|---|
| discovery | base_rate | 53327 | 0.6931 | 0.2500 | -0.0051 | -0.0059 | -0.0042 | 0.4944 | 0.4944 |
| confirmation | base_rate | 42004 | 0.6933 | 0.2501 | -0.0083 | -0.0094 | -0.0072 | 0.4944 | 0.5028 |
| discovery | proxy_raw | 53327 | 0.6885 | 0.2477 | -0.0004 | -0.0007 | -0.0002 | 0.4896 | 0.4944 |
| confirmation | proxy_raw | 42004 | 0.6854 | 0.2462 | -0.0004 | -0.0008 | -0.0001 | 0.4935 | 0.5028 |
| discovery | proxy_cal | 53327 | 0.6880 | 0.2475 | 0.0000 | 0.0000 | 0.0000 | 0.4944 | 0.4944 |
| confirmation | proxy_cal | 42004 | 0.6850 | 0.2459 | 0.0000 | 0.0000 | 0.0000 | 0.4973 | 0.5028 |
| discovery | model | 53327 | 0.6851 | 0.2460 | 0.0030 | 0.0022 | 0.0037 | 0.4944 | 0.4944 |
| confirmation | model | 42004 | 0.6810 | 0.2440 | 0.0040 | 0.0030 | 0.0049 | 0.5005 | 0.5028 |
| discovery | bet365_novig | 53327 | 0.6829 | 0.2450 | 0.0052 | 0.0044 | 0.0059 | 0.4942 | 0.4944 |
| confirmation | bet365_novig | 42004 | 0.6760 | 0.2416 | 0.0090 | 0.0079 | 0.0100 | 0.5011 | 0.5028 |

Bet365 beats the calibrated rolling mean by +0.0090 [+0.0079, +0.0100] nats on confirmation. Our own goals model, built from the same feature ladder, beats it by +0.0040 [+0.0030, +0.0049] -- less than half as much. On the one market where the comparison can be made, this modelling apparatus is markedly worse than the bookmaker it would have to beat.

In ROI terms, the same goals model is made to bet into three markets: one priced off the rolling-mean proxy at the stated hold, one priced off Bet365's *no-vig* probability at that same hold, and one at Bet365's actual posted prices. The second leg exists because the third cannot be margin-matched: Bet365's realised two-way overround on these prices is 5.78%, fixed, so differencing it against a proxy leg priced at 4% or 8% would mix a margin difference into the sharpness difference the haircut is supposed to isolate. `haircut_roi_points` is therefore the margin-matched difference (`roi_vs_proxy - roi_vs_real_book_matched`) and is what section 5.2 subtracts; `haircut_roi_points_actual_prices` is the raw difference against the posted prices, kept for reference. At the 6% headline row the two are nearly identical (20.7 against 20.7 ROI points), because 6% is 0.22 points from Bet365's realised hold; at 4% and 8% they differ by 3.1 and 1.8 points, which is exactly the margin mismatch this leg removes.

| split | hold | haircut_nats | roi_vs_proxy | roi_vs_real_book_matched | roi_vs_real_book_actual_prices | real_book_hold | haircut_roi_points | haircut_roi_points_actual_prices |
|---|---|---|---|---|---|---|---|---|
| discovery | 0.0400 | 0.0052 | 0.1144 | -0.0286 | -0.0564 | 0.0669 | 0.1430 | 0.1708 |
| discovery | 0.0600 | 0.0052 | 0.1060 | -0.0483 | -0.0564 | 0.0669 | 0.1543 | 0.1624 |
| discovery | 0.0800 | 0.0052 | 0.0968 | -0.0848 | -0.0564 | 0.0669 | 0.1816 | 0.1532 |
| confirmation | 0.0400 | 0.0090 | 0.1159 | -0.0577 | -0.0891 | 0.0578 | 0.1736 | 0.2050 |
| confirmation | 0.0600 | 0.0090 | 0.1176 | -0.0897 | -0.0891 | 0.0578 | 0.2073 | 0.2067 |
| confirmation | 0.0800 | 0.0090 | 0.1206 | -0.1076 | -0.0891 | 0.0578 | 0.2282 | 0.2097 |

All three legs simulate the *same* goals model under the same discovery-chosen threshold. Between the first two only the opponent's fair probability changes (rolling mean vs Bet365 de-vigged), which is what makes their difference a pure sharpness gap; the third also changes the margin, to whatever Bet365 actually charged. Full per-hold detail, including bet counts and mean edges, is in `04_fouls_d_goals_roi.parquet`.

### 5.2 The foul markets

For each match the proxy posts the ladder rung closest to its own mean, both sides are priced at a stated hold, and the model bets when its EV clears a threshold chosen on discovery. `roi_after_haircut` subtracts the ROI-point haircut from the row above.

| market | split | hold | threshold | mean_line | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | haircut_roi_points | roi_after_haircut | roi_lo_after_haircut | roi_hi_after_haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| team_fouls | discovery | 0.0400 | 0.0800 | 13.0263 | 30587 | 0.2427 | 0.1512 | 0.1512 | 0.1404 | 0.1620 | 0.1430 | 0.0082 | -0.0026 | 0.0189 |
| team_fouls | confirmation | 0.0400 | 0.0800 | 12.2926 | 21137 | 0.2515 | 0.1411 | 0.1115 | 0.0981 | 0.1249 | 0.1736 | -0.0620 | -0.0754 | -0.0487 |
| team_fouls | discovery | 0.0600 | 0.0800 | 13.0263 | 22811 | 0.1810 | 0.1501 | 0.1537 | 0.1402 | 0.1662 | 0.1543 | -0.0006 | -0.0140 | 0.0120 |
| team_fouls | confirmation | 0.0600 | 0.0800 | 12.2926 | 15304 | 0.1821 | 0.1386 | 0.1090 | 0.0943 | 0.1242 | 0.2073 | -0.0983 | -0.1130 | -0.0831 |
| team_fouls | discovery | 0.0800 | 0.0800 | 13.0263 | 17067 | 0.1354 | 0.1487 | 0.1533 | 0.1394 | 0.1670 | 0.1816 | -0.0283 | -0.0422 | -0.0146 |
| team_fouls | confirmation | 0.0800 | 0.0800 | 12.2926 | 11045 | 0.1314 | 0.1361 | 0.0958 | 0.0783 | 0.1135 | 0.2282 | -0.1324 | -0.1499 | -0.1146 |
| match_fouls | discovery | 0.0400 | 0.0800 | 26.0630 | 19807 | 0.3143 | 0.1616 | 0.1598 | 0.1472 | 0.1726 | 0.1430 | 0.0168 | 0.0041 | 0.0296 |
| match_fouls | confirmation | 0.0400 | 0.0800 | 24.5786 | 13349 | 0.3177 | 0.1516 | 0.1367 | 0.1204 | 0.1524 | 0.1736 | -0.0369 | -0.0531 | -0.0212 |
| match_fouls | discovery | 0.0600 | 0.0800 | 26.0630 | 15223 | 0.2415 | 0.1609 | 0.1614 | 0.1464 | 0.1767 | 0.1543 | 0.0071 | -0.0079 | 0.0225 |
| match_fouls | confirmation | 0.0600 | 0.0800 | 24.5786 | 10238 | 0.2437 | 0.1482 | 0.1395 | 0.1216 | 0.1568 | 0.2073 | -0.0678 | -0.0857 | -0.0505 |
| match_fouls | discovery | 0.0800 | 0.0800 | 26.0630 | 11756 | 0.1865 | 0.1600 | 0.1554 | 0.1392 | 0.1718 | 0.1816 | -0.0262 | -0.0424 | -0.0098 |
| match_fouls | confirmation | 0.0800 | 0.0800 | 24.5786 | 7690 | 0.1830 | 0.1460 | 0.1398 | 0.1202 | 0.1605 | 0.2282 | -0.0884 | -0.1079 | -0.0676 |

### 5.3 How sharp may the book be before the edge disappears?

`book_lambda` moves the opponent from the rolling mean (0) to our own model (1) on the logit scale; confirmation rows only.

| market | hold | book_lambda | n_bets | mean_edge | roi | roi_lo | roi_hi |
|---|---|---|---|---|---|---|---|
| team_fouls | 0.0400 | 0.0000 | 21137 | 0.1411 | 0.1115 | 0.0974 | 0.1240 |
| team_fouls | 0.0400 | 0.2500 | 10145 | 0.1180 | 0.0790 | 0.0612 | 0.0944 |
| team_fouls | 0.0400 | 0.5000 | 1480 | 0.0963 | 0.0784 | 0.0369 | 0.1167 |
| team_fouls | 0.0400 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| team_fouls | 0.0400 | 1.0000 | 0 | n/a | n/a | n/a | n/a |
| team_fouls | 0.0600 | 0.0000 | 15304 | 0.1386 | 0.1090 | 0.0946 | 0.1236 |
| team_fouls | 0.0600 | 0.2500 | 6020 | 0.1160 | 0.0702 | 0.0483 | 0.0919 |
| team_fouls | 0.0600 | 0.5000 | 442 | 0.0924 | 0.0259 | -0.0517 | 0.1076 |
| team_fouls | 0.0600 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| team_fouls | 0.0600 | 1.0000 | 0 | n/a | n/a | n/a | n/a |
| team_fouls | 0.0800 | 0.0000 | 11045 | 0.1361 | 0.0958 | 0.0781 | 0.1157 |
| team_fouls | 0.0800 | 0.2500 | 3584 | 0.1131 | 0.0728 | 0.0442 | 0.0986 |
| team_fouls | 0.0800 | 0.5000 | 87 | 0.0893 | -0.1110 | -0.2745 | 0.0558 |
| team_fouls | 0.0800 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| team_fouls | 0.0800 | 1.0000 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0400 | 0.0000 | 13349 | 0.1516 | 0.1367 | 0.1201 | 0.1537 |
| match_fouls | 0.0400 | 0.2500 | 7164 | 0.1243 | 0.1186 | 0.0982 | 0.1394 |
| match_fouls | 0.0400 | 0.5000 | 1464 | 0.0982 | 0.0997 | 0.0574 | 0.1360 |
| match_fouls | 0.0400 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0400 | 1.0000 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0600 | 0.0000 | 10238 | 0.1482 | 0.1395 | 0.1213 | 0.1555 |
| match_fouls | 0.0600 | 0.2500 | 4623 | 0.1217 | 0.1274 | 0.1029 | 0.1521 |
| match_fouls | 0.0600 | 0.5000 | 526 | 0.0927 | 0.0820 | 0.0135 | 0.1441 |
| match_fouls | 0.0600 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0600 | 1.0000 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0800 | 0.0000 | 7690 | 0.1460 | 0.1398 | 0.1206 | 0.1590 |
| match_fouls | 0.0800 | 0.2500 | 2954 | 0.1188 | 0.1206 | 0.0911 | 0.1509 |
| match_fouls | 0.0800 | 0.5000 | 103 | 0.0910 | 0.2107 | 0.0974 | 0.3357 |
| match_fouls | 0.0800 | 0.7500 | 0 | n/a | n/a | n/a | n/a |
| match_fouls | 0.0800 | 1.0000 | 0 | n/a | n/a | n/a | n/a |

### 5.4 Closing-line-value style check: who is right where they disagree?

| market | bin | n | diff_lo | diff_hi | mean_p_model | mean_p_book | observed | log_loss_model | log_loss_book | model_better |
|---|---|---|---|---|---|---|---|---|---|---|
| team_fouls | 0 | 16806 | -0.2723 | -0.0354 | 0.4050 | 0.4837 | 0.4203 | 0.6771 | 0.6870 | 0.0099 |
| team_fouls | 1 | 16806 | -0.0354 | -0.0012 | 0.4659 | 0.4826 | 0.4647 | 0.6882 | 0.6888 | 0.0006 |
| team_fouls | 2 | 16806 | -0.0012 | 0.0230 | 0.4941 | 0.4830 | 0.4827 | 0.6910 | 0.6908 | -0.0002 |
| team_fouls | 3 | 16806 | 0.0230 | 0.0496 | 0.5191 | 0.4834 | 0.5036 | 0.6912 | 0.6916 | 0.0003 |
| team_fouls | 4 | 16806 | 0.0496 | 0.2354 | 0.5582 | 0.4833 | 0.5458 | 0.6861 | 0.6942 | 0.0081 |
| match_fouls | 0 | 8403 | -0.3469 | -0.0393 | 0.4005 | 0.4892 | 0.4055 | 0.6700 | 0.6885 | 0.0185 |
| match_fouls | 1 | 8403 | -0.0393 | -0.0008 | 0.4711 | 0.4897 | 0.4692 | 0.6910 | 0.6919 | 0.0009 |
| match_fouls | 2 | 8403 | -0.0008 | 0.0284 | 0.5037 | 0.4895 | 0.4907 | 0.6920 | 0.6916 | -0.0004 |
| match_fouls | 3 | 8403 | 0.0284 | 0.0590 | 0.5333 | 0.4902 | 0.5154 | 0.6916 | 0.6926 | 0.0009 |
| match_fouls | 4 | 8403 | 0.0590 | 0.2997 | 0.5780 | 0.4895 | 0.5658 | 0.6815 | 0.6955 | 0.0141 |

In the bins where the model disagrees most with the proxy, the model is closer to the truth -- the edge over the proxy is real. The proxy is simply not the opponent a bettor faces.

## 6. The gate

The gate for this stage's two targets is run by the cross-candidate sweep in `reports/05_gate_sweep.md`, through the single shared implementation (`common.run_gate`), so that its numbers are directly comparable with the other six markets. Positive `delta_gen_minus_spec` means the specialist is better, i.e. targeting has room. Both halves are shown; the `discovery_cv` verdicts are exploratory and only the `confirmation` rows count.

| target_label | scenario | where | n | n_train_scenario | loss_generalist | loss_specialist | delta_gen_minus_spec | ci_lo | ci_hi | refit_spread | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| team_fouls | tight_match | discovery_cv | 27220 | 27220 | 2.77798 | 2.78031 | -0.00233 | -0.00335 | -0.00132 | 0.00074 | no room |
| team_fouls | tight_match | confirmation | 21760 | 27220 | 2.70470 | 2.70557 | -0.00087 | -0.00189 | 0.00013 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | away_underdog | discovery_cv | 24634 | 24634 | 2.77311 | 2.77475 | -0.00164 | -0.00282 | -0.00044 | n/a | no room |
| team_fouls | away_underdog | confirmation | 14799 | 24634 | 2.70105 | 2.70175 | -0.00071 | -0.00200 | 0.00072 | n/a | inconclusive (interval spans zero) |
| team_fouls | high_foul_division | discovery_cv | 42016 | 42016 | 2.85727 | 2.85693 | 0.00033 | -0.00026 | 0.00094 | n/a | inconclusive (interval spans zero) |
| team_fouls | high_foul_division | confirmation | 16108 | 42016 | 2.78945 | 2.78993 | -0.00048 | -0.00134 | 0.00037 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_low | discovery_cv | 42016 | 42016 | 2.62134 | 2.62055 | 0.00078 | 0.00026 | 0.00135 | n/a | room |
| team_fouls | proxy_low | confirmation | 32212 | 42016 | 2.59735 | 2.59756 | -0.00021 | -0.00085 | 0.00042 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_high | discovery_cv | 42016 | 42016 | 2.87233 | 2.87232 | 0.00001 | -0.00051 | 0.00055 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_high | confirmation | 17507 | 42016 | 2.81284 | 2.81245 | 0.00039 | -0.00033 | 0.00109 | n/a | inconclusive (interval spans zero) |
| player_fouls | defenders_only | discovery_cv | 6758 | 6758 | 1.35357 | 1.37270 | -0.01912 | -0.02362 | -0.01459 | 0.00104 | no room |
| player_fouls | defenders_only | confirmation | 4930 | 6758 | 1.34775 | 1.36630 | -0.01854 | -0.02382 | -0.01308 | 0.00102 | no room |
| player_fouls | defender_x_flank | discovery_cv | 2679 | 2679 | 1.34611 | 1.36656 | -0.02045 | -0.02807 | -0.01274 | n/a | no room |
| player_fouls | defender_x_flank | confirmation | 2176 | 2679 | 1.34070 | 1.36194 | -0.02124 | -0.02924 | -0.01244 | n/a | no room |
| player_fouls | high_foul_referee | discovery_cv | 6467 | 6467 | 1.44133 | 1.44775 | -0.00643 | -0.00923 | -0.00342 | n/a | no room |
| player_fouls | high_foul_referee | confirmation | 5214 | 6467 | 1.41045 | 1.41430 | -0.00385 | -0.00662 | -0.00119 | n/a | no room |
| player_fouls | proxy_low | discovery_cv | 6454 | 6454 | 1.20791 | 1.21139 | -0.00348 | -0.00591 | -0.00111 | n/a | no room |
| player_fouls | proxy_low | confirmation | 5095 | 6454 | 1.16531 | 1.16417 | 0.00115 | -0.00152 | 0.00384 | n/a | inconclusive (interval spans zero) |
| player_fouls | proxy_high | discovery_cv | 6453 | 6453 | 1.55850 | 1.60324 | -0.04474 | -0.05145 | -0.03800 | n/a | no room |
| player_fouls | proxy_high | confirmation | 5214 | 6453 | 1.52382 | 1.55687 | -0.03305 | -0.04045 | -0.02562 | n/a | no room |

## 7. Limitations

- **No foul lines exist in this data.** Every ROI number is a simulation against a proxy-priced market, which is why the goals haircut is applied to all of them. The haircut is measured on goals and assumed to transfer to fouls; if a real foul market were *softer* than a real goals market the haircut would be too harsh, and if it were thinner and more heavily margined it would be too kind. The stage cannot settle that, and says so.
- **The player layer is one season of four leagues.** 1,398 matches is enough to separate the two matchup directions but not enough to price a niche prop, and its rows condition on a realised appearance (section 1), which a real prop cannot.
- **The fouls-won channel is a confirmation-only clearance.** It was inside noise on discovery, four (channel, target) cells were tested, and the correctly measured refit floor puts it at roughly three times noise rather than nine. Section 3.1 gives the numbers. It is the phase's most interesting surviving signal and it is also the one most in need of an independent replication.
- **Foul counts are recorded, not adjudicated.** football-data foul counts come from feeds with known coding differences between divisions and eras, which is part of what the division and season features absorb.
- **The referee identity is available only in the StatsBomb layer.** The 105k-match odds table has no referee, so the referee channel could only be tested on 1,398 matches.

