# 02 Corners: does style or state beat a rolling mean, and is the gap bettable?

Stage 02 of the scenario expected-value programme. The question is not whether a model is better on average but whether, in a pre-registered scenario, it prices corner totals better than the rolling-mean heuristic a side market is hung off, by enough to survive a realistic margin. Reproduce with `python -m research.scenario_ev.corners --stage all`.

## Verdict

1. **For the match total, a rolling mean is almost all there is.** The book proxy explains R2 = 0.0328 of match-total corners on discovery (correlation 0.184); no shrinkage or functional form does better. Adding Elo, form, prior shots and fouls and the market's own 1X2 / over-2.5 prices changes confirmation Poisson deviance by 0.0017 [0.0002, 0.0032] and the over/under lines by 0.0001 nats at 9.5 -- an order of magnitude less than the haircut in point 5.
2. **The gate says targeting has no room.** For every pre-registered scenario a specialist fitted on that scenario's rows is *worse* on those rows than a generalist fitted on everything, on discovery and again on confirmation (deltas from -0.0572 to -0.0024 on discovery). The general model is not underfitting the favourites, the mismatches or the high totals; it has nothing extra to give them.
3. **The one real signal is the split, not the total.** At team level the same model beats the same proxy by 0.0596 [0.0560, 0.0630] Poisson deviance (R2 0.082 -> 0.120, n = 84302 team-matches) and by 0.0117 [0.0108, 0.0125] nats at the 5.5 team line. The ablation says where it comes from: the pre-match market block alone is worth 0.0569 [0.0534, 0.0603] of the 0.0596 [0.0560, 0.0630]. A team's corner count follows the game script; the match total is nearly conserved between the two sides, so it barely moves.
4. **Style and imputed defensive state carry corner signal, but not new signal.** On their own they beat the proxy (style only 0.0270 [0.0119, 0.0428], imputed defensive state only 0.0175 [0.0065, 0.0282] in match-grouped CV). On top of the odds table, the team's own style adds 0.0029 and the opponent's compactness a further 0.0007, both inside the intervals. The stated hypothesis -- crossing teams against *compact* opponents -- keeps its first half and loses its second.
5. **The haircut is the size of the whole effect.** A real bookmaker beats the identical rolling-mean proxy on the one count market whose line we observe (total goals at 2.5) by 0.0089 [0.0080, 0.0099] nats on confirmation, worth ROI 0.125 [0.110, 0.140] betting into rolling-mean prices at a 6% hold and a 2% edge threshold.
6. **So: no +EV claim survives.** At a 6% hold and the discovery threshold the match-total market returns ROI 0.0140 [-0.0111, 0.0391] against proxy prices, -0.1107 after the haircut. The team-line market returns 0.1324 [0.1230, 0.1420] -- large, and almost exactly the haircut, leaving 0.0077 net. Read plainly: our team model is about as far ahead of a rolling mean as a real book is, and a real book is what you would be betting against.

## 1. Data, universe and the fixed split

- Source: `data/raw/privileged/odds/matches.parquet` -- 238,858 football-data.co.uk matches, of which 122,111 record both corner counts (51%, 22 division codes).
- Usable universe after requiring at least 5 prior matches with corners recorded for both teams (so that the proxy has something to average): 105378 matches.
- **Discovery** (earlier 60% of matches): n = 63227. **Confirmation** (later 40%): n = 42151. Every threshold, feature choice, shrinkage and bet rule below was chosen on discovery; confirmation was scored once. Discovery numbers are exploratory and multiplicity-inflated.
- Grouping: every split holds out whole matches; every rolling feature uses matches with a strictly earlier date, so same-day fixtures cannot leak into each other.

### Leakage audit (recomputed by brute force, not asserted)

| check | n | max_abs_error | passed |
|---|---|---|---|
| prior_sum_strictly_earlier | 120 | 0.0000 | True |
| prior_count_strictly_earlier | 120 | 0.0000 | True |
| same_day_rows_exist_and_are_excluded | 67528 | 0.0000 | True |

## 2. C0, the book proxy, and how its shrinkage was chosen

The proxy is the standard heuristic: expected corners = league home/away level x the team's shrunk prior corners-for ratio x the opponent's shrunk prior corners-against ratio, all from strictly earlier matches of the same division-season, with the league level itself shrunk toward the division's all-history mean. Its one free parameter is the shrinkage `k`, chosen on discovery by squared error over both functional forms so that the opponent is the strongest rolling mean available, not a straw man.

| stat | form | k | split | n | r2 | corr | sd_pred | sd_y | bias |
|---|---|---|---|---|---|---|---|---|---|
| corners_for | mult | 40.0000 | discovery | 63227 | 0.0328 | 0.1841 | 0.7714 | 3.5651 | 0.0261 |
| corners_for | add | 15.0000 | discovery | 63227 | 0.0325 | 0.1821 | 0.7353 | 3.5651 | 0.0273 |
| corners_for | mult | 60.0000 | discovery | 63227 | 0.0324 | 0.1814 | 0.7198 | 3.5651 | 0.0280 |
| corners_for | add | 10.0000 | discovery | 63227 | 0.0324 | 0.1830 | 0.7668 | 3.5651 | 0.0261 |
| corners_for | add | 25.0000 | discovery | 63227 | 0.0317 | 0.1792 | 0.7002 | 3.5651 | 0.0288 |
| corners_for | add | 6.0000 | discovery | 63227 | 0.0314 | 0.1827 | 0.8073 | 3.5651 | 0.0247 |
| corners_for | mult | 25.0000 | discovery | 63227 | 0.0309 | 0.1843 | 0.8522 | 3.5651 | 0.0233 |
| corners_for | mult | 100.0000 | discovery | 63227 | 0.0305 | 0.1755 | 0.6778 | 3.5651 | 0.0299 |

Chosen: multiplicative form, k = 40 (the discovery optimum). The same form and shrinkage are used for the goals proxy of the honesty check.

### LightGBM capacity, also chosen on discovery

The corner signal is weak enough that model capacity decides the answer. A conventional 400-tree / 31-leaf LightGBM loses to the proxy outright; a small one beats it. The choice is made on an inner forward split *inside* discovery -- fit on its first 75% of dates, score the last 25% -- so confirmation is untouched. Without this step a null result here could not be told apart from a capacity artefact.

| target | params | n_train | n_valid | loss |
|---|---|---|---|---|
| over_9.5 | {"n_estimators": 100, "num_leaves": 7, "min_child_samples": 1000, "learning_rate": 0.05} | 42906 | 20321 | 0.678612 |
| over_9.5 | {"n_estimators": 100, "num_leaves": 3, "min_child_samples": 2000, "learning_rate": 0.05} | 42906 | 20321 | 0.678707 |
| over_9.5 | {"n_estimators": 200, "num_leaves": 7, "min_child_samples": 500, "learning_rate": 0.05} | 42906 | 20321 | 0.678895 |
| over_9.5 | C0_proxy | 42906 | 20321 | 0.679539 |
| over_9.5 | {"n_estimators": 600, "num_leaves": 7, "min_child_samples": 2000, "learning_rate": 0.03} | 42906 | 20321 | 0.679880 |
| over_9.5 | {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.04} | 42906 | 20321 | 0.680666 |
| over_9.5 | {"n_estimators": 400, "num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.04} | 42906 | 20321 | 0.683758 |
| total_count | {"n_estimators": 200, "num_leaves": 7, "min_child_samples": 500, "learning_rate": 0.05} | 42906 | 20321 | 1.167712 |
| total_count | {"n_estimators": 100, "num_leaves": 7, "min_child_samples": 1000, "learning_rate": 0.05} | 42906 | 20321 | 1.167793 |
| total_count | {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.04} | 42906 | 20321 | 1.168601 |
| total_count | {"n_estimators": 100, "num_leaves": 3, "min_child_samples": 2000, "learning_rate": 0.05} | 42906 | 20321 | 1.168662 |
| total_count | {"n_estimators": 600, "num_leaves": 7, "min_child_samples": 2000, "learning_rate": 0.03} | 42906 | 20321 | 1.168838 |
| total_count | C0_proxy | 42906 | 20321 | 1.171085 |
| total_count | {"n_estimators": 400, "num_leaves": 31, "min_child_samples": 200, "learning_rate": 0.04} | 42906 | 20321 | 1.173160 |

## 3. Proxy honesty check on goals: the haircut (protocol step 4)

We have no corner or card lines, only the 1X2 and over/under 2.5 goals prices. So the proxy's weakness is calibrated on the one count market where a real line is observed: the identical rolling-mean machinery predicts total goals, is turned into a probability at 2.5 (negative binomial, dispersion and a Platt recalibration both fitted on discovery), and is compared with Bet365's no-vig over/under 2.5 price on the same matches. The gap is the estimate of how much better a real book is than a rolling mean on a count market.

| split | model | n | log_loss | brier | delta_vs_proxy_cal | ci_lo | ci_hi | mean_p | obs_rate |
|---|---|---|---|---|---|---|---|---|---|
| discovery | base_rate | 53531 | 0.6931 | 0.2500 | -0.0051 | -0.0060 | -0.0042 | 0.4938 | 0.4938 |
| discovery | proxy_raw | 53531 | 0.6884 | 0.2477 | -0.0005 | -0.0007 | -0.0002 | 0.4892 | 0.4938 |
| discovery | proxy_cal | 53531 | 0.6880 | 0.2474 | 0.0000 | 0.0000 | 0.0000 | 0.4938 | 0.4938 |
| discovery | bet365_novig | 53531 | 0.6828 | 0.2449 | 0.0052 | 0.0044 | 0.0060 | 0.4938 | 0.4938 |
| confirmation | base_rate | 42140 | 0.6933 | 0.2501 | -0.0083 | -0.0095 | -0.0072 | 0.4938 | 0.5028 |
| confirmation | proxy_raw | 42140 | 0.6854 | 0.2462 | -0.0004 | -0.0008 | -0.0001 | 0.4935 | 0.5028 |
| confirmation | proxy_cal | 42140 | 0.6850 | 0.2459 | 0.0000 | 0.0000 | 0.0000 | 0.4971 | 0.5028 |
| confirmation | bet365_novig | 42140 | 0.6761 | 0.2416 | 0.0089 | 0.0080 | 0.0099 | 0.5011 | 0.5028 |

On confirmation the real book beats the calibrated rolling-mean proxy by 0.0089 [0.0080, 0.0099] nats (discovery 0.0052 [0.0044, 0.0060]). Expressed as money: betting the book's fair probability into prices built from the rolling-mean proxy plus a hold returns

| split | hold | threshold | bettor | book | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | n_over | profit | mean_p_model | mean_p_book | realized_a_rate | mean_odds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| confirmation | 0.0400 | 0.0000 | bet365_novig | goals_rolling_proxy | 42140 | 27894 | 0.6619 | 0.0679 | 0.1106 | 0.0992 | 0.1216 | 14299 | 3085.3558 | 0.5064 | 0.5001 | 0.5075 | 1.9538 |
| confirmation | 0.0400 | 0.0200 | bet365_novig | goals_rolling_proxy | 42140 | 21688 | 0.5147 | 0.0845 | 0.1295 | 0.1169 | 0.1421 | 11368 | 2808.7541 | 0.5104 | 0.5018 | 0.5149 | 1.9602 |
| confirmation | 0.0400 | 0.0500 | bet365_novig | goals_rolling_proxy | 42140 | 14183 | 0.3366 | 0.1110 | 0.1529 | 0.1378 | 0.1686 | 7827 | 2169.1569 | 0.5192 | 0.5048 | 0.5282 | 1.9683 |
| confirmation | 0.0600 | 0.0000 | bet365_novig | goals_rolling_proxy | 42140 | 21895 | 0.5196 | 0.0634 | 0.1075 | 0.0944 | 0.1198 | 11464 | 2353.3328 | 0.5103 | 0.5018 | 0.5146 | 1.9230 |
| confirmation | 0.0600 | 0.0200 | bet365_novig | goals_rolling_proxy | 42140 | 16548 | 0.3927 | 0.0808 | 0.1246 | 0.1101 | 0.1395 | 8964 | 2062.3214 | 0.5160 | 0.5039 | 0.5240 | 1.9282 |
| confirmation | 0.0600 | 0.0500 | bet365_novig | goals_rolling_proxy | 42140 | 10430 | 0.2475 | 0.1082 | 0.1511 | 0.1328 | 0.1691 | 5975 | 1575.9774 | 0.5268 | 0.5070 | 0.5382 | 1.9380 |
| confirmation | 0.0800 | 0.0000 | bet365_novig | goals_rolling_proxy | 42140 | 16863 | 0.4002 | 0.0596 | 0.1017 | 0.0877 | 0.1162 | 9126 | 1714.7016 | 0.5155 | 0.5037 | 0.5240 | 1.8918 |
| confirmation | 0.0800 | 0.0200 | bet365_novig | goals_rolling_proxy | 42140 | 12379 | 0.2938 | 0.0778 | 0.1215 | 0.1046 | 0.1379 | 6958 | 1503.9574 | 0.5226 | 0.5059 | 0.5328 | 1.8991 |
| confirmation | 0.0800 | 0.0500 | bet365_novig | goals_rolling_proxy | 42140 | 7531 | 0.1787 | 0.1061 | 0.1459 | 0.1252 | 0.1669 | 4565 | 1098.9006 | 0.5370 | 0.5090 | 0.5473 | 1.9055 |

**The haircut.** Any edge claimed against the rolling-mean proxy on corners must exceed this before it can be called +EV: about 0.0089 nats of log-loss, or ROI 0.125 at a 6% hold and a 2% edge threshold -- and that ROI figure is the *one-directional* form, which is a lower bound; the phase reference is the margin-matched round trip in the table below. Two caveats, both stated rather than buried: goals is the sharpest market a book prices, so this is probably an upper bound on a book's superiority over a rolling mean on corners; and the corner market's true margin is wider than the 42140-match goals market's (observed Bet365 over/under 2.5 hold averages 6.6%), which pushes the bar the other way.

**Cross-check on a different universe.** The cards stage (01) ran the same protocol step on the matches with card counts rather than corner counts, n = 64790 on confirmation, with its own rolling machinery, its own goals model and its own split date, and measures the gap as 0.0111 [0.0101, 0.0122] nats against 0.0089 [0.0080, 0.0099] here. The two intervals do not quite overlap -- different match sets, different eras, and card matches skew to the leagues and years with the most complete data. Note what this is and is not: the two universes are genuinely different, but both stages de-vig with the same shared helper (`common.novig_two_way`) on the same Bet365 price columns, so this is one construction measured on two match sets, not two independent constructions. Stage 04's foul universe is a *subset* of this one and agrees with it to five decimals, so it is the same measurement again rather than a third. `06_haircut.md` quantifies the overlap.

**Three haircut definitions, and which one this stage's results are read under.** The phase measures the same gap in money three ways. They are not interchangeable, and the corner team line survives under one of them and not the other two, so all three are shown (confirmation, 6% hold):

| definition | stage | value | what |
|---|---|---|---|
| book-earns (one-directional) | 02 corners | 0.1246 | ROI a real book makes betting its de-vigged price into proxy prices |
| margin-matched round trip | 04 fouls | 0.2073 | ROI(model vs proxy prices) - ROI(model vs the real book's no-vig price re-priced at the same hold) |
| actual-prices round trip | 01 cards | 0.3062 | the same, against Bet365's posted prices: 0.177 the goals model makes against proxy prices plus the 0.129 it loses against the posted price |

Every betting cell of this stage after each haircut (count-NB source, 6% hold, 0.02 threshold, confirmation):

| market | n_bets | ROI vs proxy | - book-earns | - margin-matched | - actual-prices |
|---|---|---|---|---|---|
| `match_total x all` | 5976 | 0.0140 | -0.111 | -0.193 | -0.292 |
| `match_total x fav_strong` | 2502 | 0.0154 | -0.109 | -0.192 | -0.291 |
| `match_total x fav_strong_low_total` | 207 | 0.0533 | -0.071 | -0.154 | -0.253 |
| `match_total x high_total` | 1776 | 0.0342 | -0.090 | -0.173 | -0.272 |
| `team_total x all` | 43833 | 0.1324 | +0.008 | -0.075 | -0.174 |
| `team_total x fav_strong` | 9267 | 0.2619 | +0.137 | +0.055 | -0.044 |
| `team_total x fav_strong_low_total` | 969 | 0.1575 | +0.033 | -0.050 | -0.149 |
| `team_total x high_total` | 8471 | 0.2212 | +0.097 | +0.014 | -0.085 |

Under the one-directional haircut 4 of 8 cells stay positive (`team_total x all`, `team_total x fav_strong`, `team_total x fav_strong_low_total`, `team_total x high_total`); under the margin-matched round trip 2 do (`team_total x fav_strong`, `team_total x high_total`); under the actual-prices round trip none do. The match total is clearly negative under all three. **Which definition is used therefore decides the team line and nothing else**, and the conclusion below reads it under the margin-matched one, which is the definition the phase settles on.

## 4. (a) Large-sample layer: C0 vs C1

C1 adds to the proxy's inputs: Elo and the Elo gap, form, division, rest days, prior shots / shots on target / fouls for both teams, and the match's pre-match no-vig 1X2 and over/under 2.5 probabilities. Two specifications are reported. `C1_plain` is a free-standing LightGBM; `C1_offset` takes the proxy as an `init_score` so the trees only learn the deviation from it, which removes the league-level drift that otherwise punishes a stationary model. Positive delta = better than the proxy.

| model | split | seed | n | value | r2 | mean_p | obs_rate | delta_c0_minus_model | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_proxy | discovery | 0 | 63227 | 1.1838 | 0.0328 | 10.5759 | 10.5498 | 0.0000 | 0.0000 | 0.0000 |
| C0_proxy | confirmation | 0 | 42151 | 1.1701 | 0.0237 | 9.8513 | 9.7745 | 0.0000 | 0.0000 | 0.0000 |
| C1_plain | discovery | 0 | 63227 | 1.1802 | 0.0357 | 10.5501 | 10.5498 | 0.0036 | 0.0023 | 0.0048 |
| C1_plain | confirmation | 0 | 42151 | 1.1686 | 0.0250 | 9.9573 | 9.7745 | 0.0015 | -0.0001 | 0.0032 |
| C1_offset | discovery | 0 | 63227 | 1.1798 | 0.0361 | 10.5501 | 10.5498 | 0.0040 | 0.0028 | 0.0050 |
| C1_offset | confirmation | 0 | 42151 | 1.1685 | 0.0250 | 9.9538 | 9.7745 | 0.0017 | 0.0002 | 0.0032 |
| C1_offset | discovery | 1 | 63227 | 1.1799 | 0.0360 | 10.5501 | 10.5498 | 0.0039 | 0.0028 | 0.0050 |
| C1_offset | confirmation | 1 | 42151 | 1.1688 | 0.0247 | 9.9638 | 9.7745 | 0.0013 | -0.0002 | 0.0028 |
| C1_offset | discovery | 2 | 63227 | 1.1798 | 0.0361 | 10.5507 | 10.5498 | 0.0040 | 0.0029 | 0.0051 |
| C1_offset | confirmation | 2 | 42151 | 1.1687 | 0.0248 | 9.9611 | 9.7745 | 0.0014 | -0.0002 | 0.0029 |

At the standard half-lines (log-loss, and the paired delta against C0):

| line | model | split | seed | n | value | brier | obs_rate | mean_p | delta_c0_minus_model | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8.5000 | C0_proxy | discovery | 0 | 63227 | 0.5975 | 0.2042 | 0.7035 | 0.7035 | 0.0000 | 0.0000 | 0.0000 |
| 8.5000 | base_rate | discovery | 0 | 63227 | 0.6079 | 0.2086 | 0.7035 | 0.7035 | 0.0000 | 0.0000 | 0.0000 |
| 8.5000 | C0_proxy | confirmation | 0 | 42151 | 0.6548 | 0.2312 | 0.6242 | 0.6402 | 0.0000 | 0.0000 | 0.0000 |
| 8.5000 | base_rate | confirmation | 0 | 42151 | 0.6764 | 0.2408 | 0.6242 | 0.7035 | 0.0000 | 0.0000 | 0.0000 |
| 8.5000 | C1_plain | discovery | 0 | 63227 | 0.5974 | 0.2041 | 0.7035 | 0.7035 | 0.0002 | -0.0002 | 0.0005 |
| 8.5000 | C1_plain | confirmation | 0 | 42151 | 0.6547 | 0.2312 | 0.6242 | 0.6465 | 0.0001 | -0.0004 | 0.0006 |
| 8.5000 | C1_offset | discovery | 0 | 63227 | 0.5971 | 0.2040 | 0.7035 | 0.7034 | 0.0005 | 0.0001 | 0.0008 |
| 8.5000 | C1_offset | confirmation | 0 | 42151 | 0.6545 | 0.2310 | 0.6242 | 0.6476 | 0.0003 | -0.0002 | 0.0008 |
| 9.5000 | C0_proxy | discovery | 0 | 63227 | 0.6645 | 0.2359 | 0.5920 | 0.5920 | 0.0000 | 0.0000 | 0.0000 |
| 9.5000 | base_rate | discovery | 0 | 63227 | 0.6761 | 0.2415 | 0.5920 | 0.5920 | 0.0000 | 0.0000 | 0.0000 |
| 9.5000 | C0_proxy | confirmation | 0 | 42151 | 0.6860 | 0.2465 | 0.5068 | 0.5207 | 0.0000 | 0.0000 | 0.0000 |
| 9.5000 | base_rate | confirmation | 0 | 42151 | 0.7078 | 0.2572 | 0.5068 | 0.5920 | 0.0000 | 0.0000 | 0.0000 |
| 9.5000 | C1_plain | discovery | 0 | 63227 | 0.6641 | 0.2357 | 0.5920 | 0.5921 | 0.0004 | 0.0000 | 0.0008 |
| 9.5000 | C1_plain | confirmation | 0 | 42151 | 0.6858 | 0.2463 | 0.5068 | 0.5269 | 0.0002 | -0.0003 | 0.0007 |
| 9.5000 | C1_offset | discovery | 0 | 63227 | 0.6637 | 0.2356 | 0.5920 | 0.5920 | 0.0007 | 0.0003 | 0.0011 |
| 9.5000 | C1_offset | confirmation | 0 | 42151 | 0.6859 | 0.2464 | 0.5068 | 0.5270 | 0.0001 | -0.0003 | 0.0006 |
| 9.5000 | C1_offset | discovery | 1 | 63227 | 0.6637 | 0.2356 | 0.5920 | 0.5919 | 0.0007 | 0.0003 | 0.0011 |
| 9.5000 | C1_offset | confirmation | 1 | 42151 | 0.6858 | 0.2464 | 0.5068 | 0.5268 | 0.0002 | -0.0003 | 0.0007 |
| 9.5000 | C1_offset | discovery | 2 | 63227 | 0.6637 | 0.2355 | 0.5920 | 0.5922 | 0.0008 | 0.0004 | 0.0012 |
| 9.5000 | C1_offset | confirmation | 2 | 42151 | 0.6859 | 0.2464 | 0.5068 | 0.5286 | 0.0001 | -0.0004 | 0.0006 |
| 10.5000 | C0_proxy | discovery | 0 | 63227 | 0.6811 | 0.2441 | 0.4794 | 0.4794 | 0.0000 | 0.0000 | 0.0000 |
| 10.5000 | base_rate | discovery | 0 | 63227 | 0.6923 | 0.2496 | 0.4794 | 0.4794 | 0.0000 | 0.0000 | 0.0000 |
| 10.5000 | C0_proxy | confirmation | 0 | 42151 | 0.6619 | 0.2347 | 0.3913 | 0.4090 | 0.0000 | 0.0000 | 0.0000 |
| 10.5000 | base_rate | confirmation | 0 | 42151 | 0.6850 | 0.2459 | 0.3913 | 0.4794 | 0.0000 | 0.0000 | 0.0000 |
| 10.5000 | C1_plain | discovery | 0 | 63227 | 0.6808 | 0.2439 | 0.4794 | 0.4794 | 0.0003 | -0.0001 | 0.0007 |
| 10.5000 | C1_plain | confirmation | 0 | 42151 | 0.6617 | 0.2346 | 0.3913 | 0.4133 | 0.0002 | -0.0003 | 0.0007 |
| 10.5000 | C1_offset | discovery | 0 | 63227 | 0.6806 | 0.2438 | 0.4794 | 0.4794 | 0.0005 | 0.0001 | 0.0009 |
| 10.5000 | C1_offset | confirmation | 0 | 42151 | 0.6615 | 0.2345 | 0.3913 | 0.4129 | 0.0004 | -0.0001 | 0.0008 |

### Team lines: home and away corners (4.5 / 5.5)

Same comparison at team-match level, both rows of a match always in the same fold, every model a proxy-offset correction. Three nested feature blocks separate the two channels: `C1_market_only` is the pre-match market view (no-vig 1X2 and over/under 2.5, Elo, home/away) and nothing else; `C1_events_only` is the two teams' strictly-prior event histories; `C1_full` is both.

| model | split | n | loss_C0 | loss_C1 | r2_C0 | r2_C1 | delta_c0_minus_c1 | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| C1_market_only | discovery | 126454 | 1.4440 | 1.4067 | 0.0943 | 0.1182 | 0.0373 | 0.0346 | 0.0400 |
| C1_market_only | confirmation | 84302 | 1.5111 | 1.4542 | 0.0818 | 0.1179 | 0.0569 | 0.0534 | 0.0603 |
| C1_events_only | discovery | 126454 | 1.4440 | 1.4123 | 0.0943 | 0.1147 | 0.0317 | 0.0294 | 0.0340 |
| C1_events_only | confirmation | 84302 | 1.5111 | 1.4633 | 0.0818 | 0.1122 | 0.0478 | 0.0446 | 0.0508 |
| C1_full | discovery | 126454 | 1.4440 | 1.4034 | 0.0943 | 0.1204 | 0.0405 | 0.0379 | 0.0432 |
| C1_full | confirmation | 84302 | 1.5111 | 1.4515 | 0.0818 | 0.1196 | 0.0596 | 0.0560 | 0.0630 |

| line | model | split | n | loss_C0 | loss_C1 | brier_C0 | brier_C1 | obs_rate | delta_c0_minus_c1 | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.5000 | C1_market_only | discovery | 126454 | 0.6536 | 0.6462 | 0.2309 | 0.2274 | 0.5631 | 0.0074 | 0.0067 | 0.0081 |
| 4.5000 | C1_market_only | confirmation | 84302 | 0.6660 | 0.6553 | 0.2368 | 0.2318 | 0.5033 | 0.0107 | 0.0098 | 0.0116 |
| 4.5000 | C1_events_only | discovery | 126454 | 0.6536 | 0.6473 | 0.2309 | 0.2279 | 0.5631 | 0.0063 | 0.0058 | 0.0070 |
| 4.5000 | C1_events_only | confirmation | 84302 | 0.6660 | 0.6568 | 0.2368 | 0.2325 | 0.5033 | 0.0091 | 0.0083 | 0.0099 |
| 4.5000 | C1_full | discovery | 126454 | 0.6536 | 0.6452 | 0.2309 | 0.2270 | 0.5631 | 0.0084 | 0.0078 | 0.0091 |
| 4.5000 | C1_full | confirmation | 84302 | 0.6660 | 0.6545 | 0.2368 | 0.2314 | 0.5033 | 0.0115 | 0.0106 | 0.0123 |
| 5.5000 | C1_market_only | discovery | 126454 | 0.6483 | 0.6413 | 0.2284 | 0.2252 | 0.4198 | 0.0070 | 0.0063 | 0.0077 |
| 5.5000 | C1_market_only | confirmation | 84302 | 0.6291 | 0.6184 | 0.2194 | 0.2148 | 0.3658 | 0.0107 | 0.0099 | 0.0116 |
| 5.5000 | C1_events_only | discovery | 126454 | 0.6483 | 0.6424 | 0.2284 | 0.2257 | 0.4198 | 0.0059 | 0.0053 | 0.0065 |
| 5.5000 | C1_events_only | confirmation | 84302 | 0.6291 | 0.6203 | 0.2194 | 0.2156 | 0.3658 | 0.0088 | 0.0080 | 0.0095 |
| 5.5000 | C1_full | discovery | 126454 | 0.6483 | 0.6403 | 0.2284 | 0.2248 | 0.4198 | 0.0080 | 0.0073 | 0.0087 |
| 5.5000 | C1_full | confirmation | 84302 | 0.6291 | 0.6175 | 0.2194 | 0.2143 | 0.3658 | 0.0117 | 0.0108 | 0.0125 |

Split by side (full model only):

| target | line | split | side | n | loss_C0 | loss_C1 | delta_c0_minus_c1 | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| team_count |  | discovery | home | 63227 | 1.4430 | 1.3985 | 0.0445 | 0.0409 | 0.0481 |
| team_count |  | discovery | away | 63227 | 1.4449 | 1.4084 | 0.0366 | 0.0334 | 0.0399 |
| team_count |  | confirmation | home | 42151 | 1.5115 | 1.4504 | 0.0611 | 0.0563 | 0.0658 |
| team_count |  | confirmation | away | 42151 | 1.5107 | 1.4526 | 0.0581 | 0.0537 | 0.0623 |
| team_over_line | 4.5000 | discovery | home | 63227 | 0.6325 | 0.6236 | 0.0089 | 0.0080 | 0.0098 |
| team_over_line | 4.5000 | discovery | away | 63227 | 0.6747 | 0.6667 | 0.0080 | 0.0071 | 0.0089 |
| team_over_line | 4.5000 | confirmation | home | 42151 | 0.6651 | 0.6531 | 0.0120 | 0.0108 | 0.0133 |
| team_over_line | 4.5000 | confirmation | away | 42151 | 0.6669 | 0.6559 | 0.0110 | 0.0098 | 0.0121 |
| team_over_line | 5.5000 | discovery | home | 63227 | 0.6739 | 0.6652 | 0.0087 | 0.0078 | 0.0097 |
| team_over_line | 5.5000 | discovery | away | 63227 | 0.6227 | 0.6155 | 0.0073 | 0.0064 | 0.0081 |
| team_over_line | 5.5000 | confirmation | home | 42151 | 0.6619 | 0.6497 | 0.0122 | 0.0110 | 0.0134 |
| team_over_line | 5.5000 | confirmation | away | 42151 | 0.5963 | 0.5852 | 0.0111 | 0.0100 | 0.0122 |

**This is the one place in the stage where the richer model clearly beats the rolling mean.** On confirmation the full team model cuts Poisson deviance by 0.0596 [0.0560, 0.0630] (n = 84302) and the 5.5 team line by 0.0117 [0.0108, 0.0125] nats. The ablation says where it comes from: the market-only block alone is worth 0.0569 [0.0534, 0.0603] and the event-history block alone 0.0478 [0.0446, 0.0508]. In other words a team's corner count follows the *game script* -- who the market thinks will be on top -- and a rolling corner mean is blind to it, while the match total is nearly conserved between the two teams and so barely moves.

Read this with the obvious caveat attached: a real bookmaker pricing a team corner line starts from the match odds, because they are the most visible input on the screen. Our proxy deliberately does not. So this gap measures how much the *proxy* is missing, not how much a real market is missing -- which is exactly what section 3's haircut is for.

### Refit-noise floor (three seeds)

The match models first. These are the C1_offset headline fits; the confirmation column is the one the deltas above are read against.

| target | line | model | split | min | max | count | spread |
|---|---|---|---|---|---|---|---|
| over_line | 9.5000 | C1_offset | confirmation | 0.6858 | 0.6859 | 3 | 0.0001 |
| over_line | 9.5000 | C1_offset | discovery | 0.6637 | 0.6637 | 3 | 0.0001 |
| total_count |  | C1_offset | confirmation | 1.1685 | 1.1688 | 3 | 0.0004 |
| total_count |  | C1_offset | discovery | 1.1798 | 1.1799 | 3 | 0.0001 |

And the **team** models, which the match-model floor above does not cover. The team line is the one quantity in this stage whose margin over the proxy exceeds the goals haircut in nats, so protocol step 6 wants a floor belonging to that model rather than to a different one. Confirmation only, since that is where the headline is read; the confirmation prediction is a single fit on all discovery rows, so re-seeding it is exactly the relevant noise.

| target | line | model | split | min | max | count | spread | values |
|---|---|---|---|---|---|---|---|---|
| team_count |  | C1_full | confirmation | 1.4514 | 1.4517 | 3 | 0.0003 | 1.451396;1.451697;1.451609 |
| team_over_line | 4.5000 | C1_full | confirmation | 0.6543 | 0.6544 | 3 | 0.0001 | 0.654407;0.654342;0.654352 |
| team_over_line | 5.5000 | C1_full | confirmation | 0.6174 | 0.6175 | 3 | 0.0001 | 0.617514;0.617434;0.617499 |

The 5.5 team line's floor is 0.000081 nats against a measured margin over the proxy of 0.0117 [0.0108, 0.0125] -- about 144x the floor, so the margin is real as a measurement. Whether it is tradeable is a separate question and is answered by the haircut, not by this floor.

## 5. The gate: is the generalist underfitting any scenario?

A generalist is fitted on all training rows, a specialist on the scenario's training rows only, and both are scored on held-out scenario rows -- 5-fold with whole matches held out inside discovery, then once on confirmation. Positive `delta_gen_minus_spec` means the specialist is better, i.e. targeting has room. The reverse direction (the specialist off its own scenario) is shown too.

| scenario | split | region | n | n_train_scenario | share_of_rows | loss_generalist | loss_specialist | delta_gen_minus_spec | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| fav_strong | discovery | in_scenario | 6220 | 6220 | 0.0984 | 1.2258 | 1.2363 | -0.0105 | -0.0158 | -0.0046 |
| fav_strong | discovery | off_scenario | 57007 | 6220 | 0.0984 | 1.1753 | 1.1890 | -0.0137 | -0.0157 | -0.0118 |
| fav_strong | confirmation | in_scenario | 4636 | 6220 | 0.1100 | 1.2190 | 1.2305 | -0.0115 | -0.0187 | -0.0049 |
| fav_strong | confirmation | off_scenario | 37515 | 6220 | 0.1100 | 1.1624 | 1.1750 | -0.0126 | -0.0152 | -0.0099 |
| fav_strong_low_total | discovery | in_scenario | 815 | 815 | 0.0129 | 1.2089 | 1.2661 | -0.0572 | -0.0941 | -0.0203 |
| fav_strong_low_total | discovery | off_scenario | 62412 | 815 | 0.0129 | 1.1799 | 1.2317 | -0.0519 | -0.0557 | -0.0481 |
| fav_strong_low_total | confirmation | in_scenario | 489 | 815 | 0.0116 | 1.2535 | 1.3558 | -0.1022 | -0.1588 | -0.0494 |
| fav_strong_low_total | confirmation | off_scenario | 41662 | 815 | 0.0116 | 1.1676 | 1.2201 | -0.0525 | -0.0568 | -0.0479 |
| elo_mismatch_low_total | discovery | in_scenario | 2506 | 2506 | 0.0396 | 1.2163 | 1.2233 | -0.0070 | -0.0160 | 0.0017 |
| elo_mismatch_low_total | discovery | off_scenario | 60721 | 2506 | 0.0396 | 1.1787 | 1.1930 | -0.0143 | -0.0163 | -0.0122 |
| elo_mismatch_low_total | confirmation | in_scenario | 1730 | 2506 | 0.0410 | 1.1786 | 1.1922 | -0.0136 | -0.0248 | -0.0027 |
| elo_mismatch_low_total | confirmation | off_scenario | 40421 | 2506 | 0.0410 | 1.1682 | 1.1836 | -0.0155 | -0.0178 | -0.0132 |
| high_total | discovery | in_scenario | 3317 | 3317 | 0.0525 | 1.2033 | 1.2116 | -0.0083 | -0.0158 | -0.0010 |
| high_total | discovery | off_scenario | 59910 | 3317 | 0.0525 | 1.1789 | 1.2025 | -0.0236 | -0.0260 | -0.0212 |
| high_total | confirmation | in_scenario | 4995 | 3317 | 0.1185 | 1.2181 | 1.2304 | -0.0123 | -0.0182 | -0.0065 |
| high_total | confirmation | off_scenario | 37156 | 3317 | 0.1185 | 1.1619 | 1.1967 | -0.0348 | -0.0385 | -0.0310 |
| home_big_fav | discovery | in_scenario | 5113 | 5113 | 0.0809 | 1.2412 | 1.2500 | -0.0088 | -0.0146 | -0.0028 |
| home_big_fav | discovery | off_scenario | 58114 | 5113 | 0.0809 | 1.1749 | 1.1933 | -0.0184 | -0.0207 | -0.0162 |
| home_big_fav | confirmation | in_scenario | 3587 | 5113 | 0.0851 | 1.2091 | 1.2234 | -0.0143 | -0.0217 | -0.0070 |
| home_big_fav | confirmation | off_scenario | 38564 | 5113 | 0.0851 | 1.1648 | 1.2005 | -0.0357 | -0.0392 | -0.0319 |
| even_match | discovery | in_scenario | 20259 | 20259 | 0.3204 | 1.1739 | 1.1767 | -0.0028 | -0.0044 | -0.0012 |
| even_match | discovery | off_scenario | 42968 | 20259 | 0.3204 | 1.1832 | 1.1873 | -0.0041 | -0.0056 | -0.0026 |
| even_match | confirmation | in_scenario | 13346 | 20259 | 0.3166 | 1.1587 | 1.1615 | -0.0028 | -0.0048 | -0.0009 |
| even_match | confirmation | off_scenario | 28805 | 20259 | 0.3166 | 1.1732 | 1.1729 | 0.0003 | -0.0016 | 0.0022 |
| proxy_low | discovery | in_scenario | 12646 | 12646 | 0.2000 | 1.1849 | 1.1877 | -0.0028 | -0.0052 | -0.0002 |
| proxy_low | discovery | off_scenario | 50581 | 12646 | 0.2000 | 1.1791 | 1.2282 | -0.0492 | -0.0531 | -0.0448 |
| proxy_low | confirmation | in_scenario | 23613 | 12646 | 0.5602 | 1.1671 | 1.1705 | -0.0034 | -0.0049 | -0.0018 |
| proxy_low | confirmation | off_scenario | 18538 | 12646 | 0.5602 | 1.1706 | 1.1774 | -0.0069 | -0.0114 | -0.0024 |
| proxy_high | discovery | in_scenario | 12646 | 12646 | 0.2000 | 1.1669 | 1.1693 | -0.0024 | -0.0050 | 0.0001 |
| proxy_high | discovery | off_scenario | 50581 | 12646 | 0.2000 | 1.1836 | 1.2573 | -0.0738 | -0.0790 | -0.0686 |
| proxy_high | confirmation | in_scenario | 880 | 12646 | 0.0209 | 1.1931 | 1.2014 | -0.0084 | -0.0197 | 0.0023 |
| proxy_high | confirmation | off_scenario | 41271 | 12646 | 0.0209 | 1.1681 | 1.3293 | -0.1613 | -0.1686 | -0.1538 |

| scenario | split | region | n | n_train_scenario | loss_generalist | loss_specialist | delta_gen_minus_spec | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|---|---|
| fav_strong | discovery | in_scenario | 6220 | 6220 | 0.6597 | 0.6610 | -0.0013 | -0.0030 | 0.0005 |
| fav_strong | discovery | off_scenario | 57007 | 6220 | 0.6646 | 0.6679 | -0.0034 | -0.0040 | -0.0027 |
| fav_strong | confirmation | in_scenario | 4636 | 6220 | 0.6819 | 0.6852 | -0.0033 | -0.0052 | -0.0014 |
| fav_strong | confirmation | off_scenario | 37515 | 6220 | 0.6863 | 0.6907 | -0.0044 | -0.0052 | -0.0035 |
| fav_strong_low_total | discovery | in_scenario | 815 | 815 | 0.6731 | 0.6888 | -0.0157 | -0.0272 | -0.0043 |
| fav_strong_low_total | discovery | off_scenario | 62412 | 815 | 0.6640 | 0.6788 | -0.0148 | -0.0161 | -0.0135 |
| fav_strong_low_total | confirmation | in_scenario | 489 | 815 | 0.6778 | 0.7102 | -0.0324 | -0.0484 | -0.0158 |
| fav_strong_low_total | confirmation | off_scenario | 41662 | 815 | 0.6859 | 0.6977 | -0.0118 | -0.0133 | -0.0103 |
| elo_mismatch_low_total | discovery | in_scenario | 2506 | 2506 | 0.6753 | 0.6884 | -0.0130 | -0.0195 | -0.0071 |
| elo_mismatch_low_total | discovery | off_scenario | 60721 | 2506 | 0.6636 | 0.6794 | -0.0158 | -0.0171 | -0.0144 |
| elo_mismatch_low_total | confirmation | in_scenario | 1730 | 2506 | 0.6835 | 0.6941 | -0.0106 | -0.0154 | -0.0055 |
| elo_mismatch_low_total | confirmation | off_scenario | 40421 | 2506 | 0.6859 | 0.6901 | -0.0042 | -0.0052 | -0.0033 |
| high_total | discovery | in_scenario | 3317 | 3317 | 0.6684 | 0.6697 | -0.0013 | -0.0038 | 0.0015 |
| high_total | discovery | off_scenario | 59910 | 3317 | 0.6638 | 0.6706 | -0.0067 | -0.0077 | -0.0057 |
| high_total | confirmation | in_scenario | 4995 | 3317 | 0.6776 | 0.6811 | -0.0036 | -0.0055 | -0.0017 |
| high_total | confirmation | off_scenario | 37156 | 3317 | 0.6869 | 0.7028 | -0.0159 | -0.0175 | -0.0142 |
| home_big_fav | discovery | in_scenario | 5113 | 5113 | 0.6555 | 0.6574 | -0.0019 | -0.0039 | 0.0001 |
| home_big_fav | discovery | off_scenario | 58114 | 5113 | 0.6648 | 0.6710 | -0.0061 | -0.0071 | -0.0052 |
| home_big_fav | confirmation | in_scenario | 3587 | 5113 | 0.6820 | 0.6851 | -0.0031 | -0.0051 | -0.0010 |
| home_big_fav | confirmation | off_scenario | 38564 | 5113 | 0.6862 | 0.6978 | -0.0117 | -0.0129 | -0.0103 |
| even_match | discovery | in_scenario | 20259 | 20259 | 0.6658 | 0.6665 | -0.0007 | -0.0014 | 0.0001 |
| even_match | discovery | off_scenario | 42968 | 20259 | 0.6633 | 0.6652 | -0.0020 | -0.0026 | -0.0014 |
| even_match | confirmation | in_scenario | 13346 | 20259 | 0.6856 | 0.6872 | -0.0016 | -0.0026 | -0.0007 |
| even_match | confirmation | off_scenario | 28805 | 20259 | 0.6859 | 0.6868 | -0.0009 | -0.0017 | -0.0002 |
| proxy_low | discovery | in_scenario | 12646 | 12646 | 0.6879 | 0.6885 | -0.0006 | -0.0016 | 0.0005 |
| proxy_low | discovery | off_scenario | 50581 | 12646 | 0.6581 | 0.6766 | -0.0185 | -0.0202 | -0.0169 |
| proxy_low | confirmation | in_scenario | 23613 | 12646 | 0.6863 | 0.6879 | -0.0016 | -0.0024 | -0.0009 |
| proxy_low | confirmation | off_scenario | 18538 | 12646 | 0.6852 | 0.6879 | -0.0027 | -0.0046 | -0.0008 |
| proxy_high | discovery | in_scenario | 12646 | 12646 | 0.6198 | 0.6206 | -0.0008 | -0.0018 | 0.0003 |
| proxy_high | discovery | off_scenario | 50581 | 12646 | 0.6751 | 0.6951 | -0.0199 | -0.0216 | -0.0181 |
| proxy_high | confirmation | in_scenario | 880 | 12646 | 0.6631 | 0.6627 | 0.0004 | -0.0037 | 0.0044 |
| proxy_high | confirmation | off_scenario | 41271 | 12646 | 0.6863 | 0.7209 | -0.0347 | -0.0369 | -0.0323 |

Pre-registered primary scenario (the least-bad on discovery): **proxy_high** -- proxy corner mean in the highest discovery quintile.
On confirmation it gives -0.0084 [-0.0197, 0.0023] (n = 880).

Note what the table says: **no** scenario has a positive discovery delta for the match total, so there was nothing to pre-register in the first place. The specialists are uniformly worse on their own rows, and worse again off them; the loss grows as the scenario shrinks (`fav_strong_low_total`, 1.3% of rows, is the worst). That is the signature of a variance problem, not of a generalist that is underfitting a corner of the space. The single exception in the whole table is `fav_strong` at team level on confirmation (+0.0085 [0.0035, 0.0136]), whose discovery counterpart is +0.0012 [-0.0027, 0.0054] and therefore was not, and could not have been, pre-registered.

## 6. Betting-shaped evaluation

The proxy hangs the half-line its own mean implies, prices both sides at the stated two-way hold, and the model bets whichever side clears the edge threshold. Stakes are flat; ROI intervals resample whole matches. Thresholds were read off the discovery grid.

`haircut_roi` is what the *real* bookmaker earned in the identical simulation on goals at the same hold and threshold (section 3), so the two are directly comparable: same pricing rule, same margin, same bet filter. `roi_after_haircut` subtracts it. That subtraction is a point estimate against a point estimate -- the haircut itself carries an interval (0.110 to 0.140 at a 6% hold and a 2% threshold), so a net of a few tenths of a percent is not distinguishable from zero.

Confirmation, all matches (the full grid -- both splits, every scenario and every threshold -- is in `02_corners_bets.parquet`):

| market | scenario | source | hold | threshold | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | haircut_roi | roi_after_haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| match_total | all | count_nb | 0.0400 | 0.0000 | 42151 | 16747 | 0.3973 | 0.0348 | 0.0030 | -0.0120 | 0.0185 | 0.1106 | -0.1076 |
| match_total | all | count_nb | 0.0400 | 0.0200 | 42151 | 10133 | 0.2404 | 0.0516 | 0.0141 | -0.0044 | 0.0348 | 0.1295 | -0.1154 |
| match_total | all | count_nb | 0.0400 | 0.0500 | 42151 | 4306 | 0.1022 | 0.0760 | 0.0376 | 0.0074 | 0.0668 | 0.1529 | -0.1153 |
| match_total | all | count_nb | 0.0600 | 0.0000 | 42151 | 10336 | 0.2452 | 0.0311 | -0.0048 | -0.0234 | 0.0148 | 0.1075 | -0.1123 |
| match_total | all | count_nb | 0.0600 | 0.0200 | 42151 | 5976 | 0.1418 | 0.0471 | 0.0140 | -0.0111 | 0.0391 | 0.1246 | -0.1107 |
| match_total | all | count_nb | 0.0600 | 0.0500 | 42151 | 2172 | 0.0515 | 0.0721 | 0.0277 | -0.0125 | 0.0701 | 0.1511 | -0.1234 |
| match_total | all | count_nb | 0.0800 | 0.0000 | 42151 | 6201 | 0.1471 | 0.0267 | -0.0071 | -0.0322 | 0.0194 | 0.1017 | -0.1088 |
| match_total | all | count_nb | 0.0800 | 0.0200 | 42151 | 3174 | 0.0753 | 0.0436 | 0.0138 | -0.0210 | 0.0498 | 0.1215 | -0.1077 |
| match_total | all | count_nb | 0.0800 | 0.0500 | 42151 | 1007 | 0.0239 | 0.0673 | 0.0534 | -0.0079 | 0.1144 | 0.1459 | -0.0925 |
| match_total | all | binary_direct | 0.0400 | 0.0000 | 40096 | 21226 | 0.5294 | 0.0441 | -0.0021 | -0.0161 | 0.0120 | 0.1106 | -0.1127 |
| match_total | all | binary_direct | 0.0400 | 0.0200 | 40096 | 14416 | 0.3595 | 0.0606 | 0.0067 | -0.0088 | 0.0240 | 0.1295 | -0.1228 |
| match_total | all | binary_direct | 0.0400 | 0.0500 | 40096 | 7662 | 0.1911 | 0.0842 | 0.0238 | 0.0012 | 0.0463 | 0.1529 | -0.1291 |
| match_total | all | binary_direct | 0.0600 | 0.0000 | 40096 | 14599 | 0.3641 | 0.0401 | -0.0128 | -0.0286 | 0.0035 | 0.1075 | -0.1203 |
| match_total | all | binary_direct | 0.0600 | 0.0200 | 40096 | 9638 | 0.2404 | 0.0558 | -0.0046 | -0.0238 | 0.0166 | 0.1246 | -0.1292 |
| match_total | all | binary_direct | 0.0600 | 0.0500 | 40096 | 4569 | 0.1140 | 0.0802 | 0.0149 | -0.0156 | 0.0461 | 0.1511 | -0.1362 |
| match_total | all | binary_direct | 0.0800 | 0.0000 | 40096 | 9873 | 0.2462 | 0.0354 | -0.0234 | -0.0427 | -0.0038 | 0.1017 | -0.1251 |
| match_total | all | binary_direct | 0.0800 | 0.0200 | 40096 | 6120 | 0.1526 | 0.0513 | -0.0057 | -0.0318 | 0.0195 | 0.1215 | -0.1272 |
| match_total | all | binary_direct | 0.0800 | 0.0500 | 40096 | 2584 | 0.0644 | 0.0760 | 0.0360 | -0.0006 | 0.0737 | 0.1459 | -0.1099 |
| team_total | all | count_nb | 0.0400 | 0.0000 | 84302 | 61286 | 0.7270 | 0.1024 | 0.1169 | 0.1085 | 0.1249 | 0.1106 | 0.0063 |
| team_total | all | count_nb | 0.0400 | 0.0200 | 84302 | 51842 | 0.6150 | 0.1193 | 0.1380 | 0.1288 | 0.1468 | 0.1295 | 0.0085 |
| team_total | all | count_nb | 0.0400 | 0.0500 | 84302 | 39949 | 0.4739 | 0.1446 | 0.1645 | 0.1544 | 0.1746 | 0.1529 | 0.0116 |
| team_total | all | count_nb | 0.0600 | 0.0000 | 84302 | 52180 | 0.6190 | 0.0976 | 0.1152 | 0.1067 | 0.1237 | 0.1075 | 0.0077 |
| team_total | all | count_nb | 0.0600 | 0.0200 | 84302 | 43833 | 0.5200 | 0.1143 | 0.1324 | 0.1230 | 0.1420 | 0.1246 | 0.0077 |
| team_total | all | count_nb | 0.0600 | 0.0500 | 84302 | 33033 | 0.3918 | 0.1404 | 0.1602 | 0.1498 | 0.1717 | 0.1511 | 0.0091 |
| team_total | all | count_nb | 0.0800 | 0.0000 | 84302 | 44308 | 0.5256 | 0.0927 | 0.1103 | 0.1012 | 0.1196 | 0.1017 | 0.0086 |
| team_total | all | count_nb | 0.0800 | 0.0200 | 84302 | 36684 | 0.4351 | 0.1099 | 0.1301 | 0.1196 | 0.1410 | 0.1215 | 0.0086 |
| team_total | all | count_nb | 0.0800 | 0.0500 | 84302 | 27194 | 0.3226 | 0.1363 | 0.1544 | 0.1426 | 0.1657 | 0.1459 | 0.0085 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0000 | 84302 | 58277 | 0.6913 | 0.0974 | 0.1188 | 0.1106 | 0.1274 | 0.1106 | 0.0082 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0200 | 84302 | 50213 | 0.5956 | 0.1114 | 0.1372 | 0.1284 | 0.1459 | 0.1295 | 0.0077 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0500 | 84302 | 37693 | 0.4471 | 0.1368 | 0.1688 | 0.1586 | 0.1794 | 0.1529 | 0.0159 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0000 | 84302 | 50540 | 0.5995 | 0.0898 | 0.1149 | 0.1067 | 0.1236 | 0.1075 | 0.0074 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0200 | 84302 | 41845 | 0.4964 | 0.1064 | 0.1359 | 0.1260 | 0.1459 | 0.1246 | 0.0113 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0500 | 84302 | 30393 | 0.3605 | 0.1335 | 0.1619 | 0.1501 | 0.1734 | 0.1511 | 0.0108 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0000 | 84302 | 42367 | 0.5026 | 0.0849 | 0.1132 | 0.1042 | 0.1226 | 0.1017 | 0.0116 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0200 | 84302 | 34281 | 0.4066 | 0.1026 | 0.1322 | 0.1217 | 0.1429 | 0.1215 | 0.0108 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0500 | 84302 | 24032 | 0.2851 | 0.1317 | 0.1616 | 0.1485 | 0.1734 | 0.1459 | 0.0157 |
| team_total | all | binary_direct | 0.0400 | 0.0000 | 72156 | 53482 | 0.7412 | 0.0896 | 0.1082 | 0.0997 | 0.1173 | 0.1106 | -0.0024 |
| team_total | all | binary_direct | 0.0400 | 0.0200 | 72156 | 44274 | 0.6136 | 0.1062 | 0.1269 | 0.1171 | 0.1372 | 0.1295 | -0.0026 |
| team_total | all | binary_direct | 0.0400 | 0.0500 | 72156 | 32730 | 0.4536 | 0.1315 | 0.1563 | 0.1451 | 0.1676 | 0.1529 | 0.0034 |
| team_total | all | binary_direct | 0.0600 | 0.0000 | 72156 | 44651 | 0.6188 | 0.0846 | 0.1041 | 0.0944 | 0.1139 | 0.1075 | -0.0034 |
| team_total | all | binary_direct | 0.0600 | 0.0200 | 72156 | 36446 | 0.5051 | 0.1014 | 0.1252 | 0.1145 | 0.1356 | 0.1246 | 0.0005 |
| team_total | all | binary_direct | 0.0600 | 0.0500 | 72156 | 26308 | 0.3646 | 0.1274 | 0.1538 | 0.1417 | 0.1657 | 0.1511 | 0.0027 |
| team_total | all | binary_direct | 0.0800 | 0.0000 | 72156 | 36905 | 0.5115 | 0.0800 | 0.1026 | 0.0917 | 0.1129 | 0.1017 | 0.0009 |
| team_total | all | binary_direct | 0.0800 | 0.0200 | 72156 | 29541 | 0.4094 | 0.0976 | 0.1226 | 0.1112 | 0.1338 | 0.1215 | 0.0011 |
| team_total | all | binary_direct | 0.0800 | 0.0500 | 72156 | 21294 | 0.2951 | 0.1221 | 0.1487 | 0.1356 | 0.1623 | 0.1459 | 0.0027 |

The same rows inside the pre-registered scenarios:

| market | scenario | source | hold | threshold | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | haircut_roi | roi_after_haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| match_total | fav_strong | count_nb | 0.0400 | 0.0200 | 4636 | 3064 | 0.6609 | 0.0679 | 0.0298 | -0.0073 | 0.0660 | 0.1295 | -0.0997 |
| match_total | fav_strong | count_nb | 0.0600 | 0.0200 | 4636 | 2502 | 0.5397 | 0.0560 | 0.0154 | -0.0243 | 0.0550 | 0.1246 | -0.1092 |
| match_total | fav_strong | count_nb | 0.0800 | 0.0200 | 4636 | 1701 | 0.3669 | 0.0487 | 0.0173 | -0.0301 | 0.0647 | 0.1215 | -0.1041 |
| match_total | fav_strong_low_total | count_nb | 0.0400 | 0.0200 | 489 | 310 | 0.6339 | 0.0539 | 0.0721 | -0.0402 | 0.1843 | 0.1295 | -0.0574 |
| match_total | fav_strong_low_total | count_nb | 0.0600 | 0.0200 | 489 | 207 | 0.4233 | 0.0459 | 0.0533 | -0.0889 | 0.1883 | 0.1246 | -0.0713 |
| match_total | fav_strong_low_total | count_nb | 0.0800 | 0.0200 | 489 | 121 | 0.2474 | 0.0374 | 0.0837 | -0.0972 | 0.2615 | 0.1215 | -0.0378 |
| match_total | high_total | count_nb | 0.0400 | 0.0200 | 4995 | 2384 | 0.4773 | 0.0621 | 0.0344 | -0.0081 | 0.0747 | 0.1295 | -0.0951 |
| match_total | high_total | count_nb | 0.0600 | 0.0200 | 4995 | 1776 | 0.3556 | 0.0529 | 0.0342 | -0.0147 | 0.0809 | 0.1246 | -0.0905 |
| match_total | high_total | count_nb | 0.0800 | 0.0200 | 4995 | 1140 | 0.2282 | 0.0460 | 0.0385 | -0.0169 | 0.0983 | 0.1215 | -0.0830 |
| match_total | fav_strong | binary_direct | 0.0400 | 0.0200 | 4251 | 3182 | 0.7485 | 0.0822 | 0.0279 | -0.0069 | 0.0631 | 0.1295 | -0.1016 |
| match_total | fav_strong | binary_direct | 0.0600 | 0.0200 | 4251 | 2762 | 0.6497 | 0.0695 | 0.0152 | -0.0232 | 0.0528 | 0.1246 | -0.1094 |
| match_total | fav_strong | binary_direct | 0.0800 | 0.0200 | 4251 | 2188 | 0.5147 | 0.0599 | -0.0035 | -0.0454 | 0.0381 | 0.1215 | -0.1250 |
| match_total | fav_strong_low_total | binary_direct | 0.0400 | 0.0200 | 478 | 258 | 0.5397 | 0.0591 | 0.0690 | -0.0570 | 0.1926 | 0.1295 | -0.0605 |
| match_total | fav_strong_low_total | binary_direct | 0.0600 | 0.0200 | 478 | 176 | 0.3682 | 0.0526 | 0.1328 | -0.0176 | 0.2799 | 0.1246 | 0.0082 |
| match_total | fav_strong_low_total | binary_direct | 0.0800 | 0.0200 | 478 | 118 | 0.2469 | 0.0442 | 0.1679 | -0.0089 | 0.3373 | 0.1215 | 0.0465 |
| match_total | high_total | binary_direct | 0.0400 | 0.0200 | 4523 | 2868 | 0.6341 | 0.0729 | 0.0036 | -0.0324 | 0.0408 | 0.1295 | -0.1259 |
| match_total | high_total | binary_direct | 0.0600 | 0.0200 | 4523 | 2266 | 0.5010 | 0.0639 | -0.0075 | -0.0502 | 0.0331 | 0.1246 | -0.1322 |
| match_total | high_total | binary_direct | 0.0800 | 0.0200 | 4523 | 1669 | 0.3690 | 0.0562 | -0.0078 | -0.0556 | 0.0393 | 0.1215 | -0.1293 |
| team_total | fav_strong | count_nb | 0.0400 | 0.0200 | 9272 | 9271 | 0.9999 | 0.2312 | 0.2860 | 0.2656 | 0.3056 | 0.1295 | 0.1565 |
| team_total | fav_strong | count_nb | 0.0600 | 0.0200 | 9272 | 9267 | 0.9995 | 0.2080 | 0.2619 | 0.2419 | 0.2813 | 0.1246 | 0.1373 |
| team_total | fav_strong | count_nb | 0.0800 | 0.0200 | 9272 | 9211 | 0.9934 | 0.1867 | 0.2406 | 0.2210 | 0.2595 | 0.1215 | 0.1191 |
| team_total | fav_strong_low_total | count_nb | 0.0400 | 0.0200 | 978 | 977 | 0.9990 | 0.1533 | 0.1791 | 0.1136 | 0.2443 | 0.1295 | 0.0496 |
| team_total | fav_strong_low_total | count_nb | 0.0600 | 0.0200 | 978 | 969 | 0.9908 | 0.1325 | 0.1575 | 0.0920 | 0.2217 | 0.1246 | 0.0329 |
| team_total | fav_strong_low_total | count_nb | 0.0800 | 0.0200 | 978 | 916 | 0.9366 | 0.1173 | 0.1443 | 0.0780 | 0.2088 | 0.1215 | 0.0228 |
| team_total | high_total | count_nb | 0.0400 | 0.0200 | 9990 | 8817 | 0.8826 | 0.1996 | 0.2369 | 0.2161 | 0.2591 | 0.1295 | 0.1074 |
| team_total | high_total | count_nb | 0.0600 | 0.0200 | 9990 | 8471 | 0.8479 | 0.1838 | 0.2212 | 0.2007 | 0.2423 | 0.1246 | 0.0966 |
| team_total | high_total | count_nb | 0.0800 | 0.0200 | 9990 | 8061 | 0.8069 | 0.1695 | 0.2099 | 0.1891 | 0.2302 | 0.1215 | 0.0884 |
| team_total | fav_strong | count_nb_market_only | 0.0400 | 0.0200 | 9272 | 9272 | 1.0000 | 0.2185 | 0.2859 | 0.2656 | 0.3053 | 0.1295 | 0.1563 |
| team_total | fav_strong | count_nb_market_only | 0.0600 | 0.0200 | 9272 | 9272 | 1.0000 | 0.1956 | 0.2616 | 0.2417 | 0.2807 | 0.1246 | 0.1370 |
| team_total | fav_strong | count_nb_market_only | 0.0800 | 0.0200 | 9272 | 9240 | 0.9965 | 0.1740 | 0.2396 | 0.2197 | 0.2586 | 0.1215 | 0.1181 |
| team_total | fav_strong_low_total | count_nb_market_only | 0.0400 | 0.0200 | 978 | 978 | 1.0000 | 0.1451 | 0.1795 | 0.1143 | 0.2448 | 0.1295 | 0.0500 |
| team_total | fav_strong_low_total | count_nb_market_only | 0.0600 | 0.0200 | 978 | 978 | 1.0000 | 0.1235 | 0.1573 | 0.0933 | 0.2213 | 0.1246 | 0.0326 |
| team_total | fav_strong_low_total | count_nb_market_only | 0.0800 | 0.0200 | 978 | 966 | 0.9877 | 0.1038 | 0.1362 | 0.0723 | 0.2000 | 0.1215 | 0.0147 |
| team_total | high_total | count_nb_market_only | 0.0400 | 0.0200 | 9990 | 8703 | 0.8712 | 0.1906 | 0.2399 | 0.2191 | 0.2611 | 0.1295 | 0.1104 |
| team_total | high_total | count_nb_market_only | 0.0600 | 0.0200 | 9990 | 8313 | 0.8321 | 0.1756 | 0.2263 | 0.2055 | 0.2486 | 0.1246 | 0.1017 |
| team_total | high_total | count_nb_market_only | 0.0800 | 0.0200 | 9990 | 7874 | 0.7882 | 0.1618 | 0.2099 | 0.1876 | 0.2317 | 0.1215 | 0.0884 |
| team_total | fav_strong | binary_direct | 0.0400 | 0.0200 | 5837 | 5830 | 0.9988 | 0.2054 | 0.3038 | 0.2811 | 0.3261 | 0.1295 | 0.1743 |
| team_total | fav_strong | binary_direct | 0.0600 | 0.0200 | 5837 | 5796 | 0.9930 | 0.1837 | 0.2807 | 0.2579 | 0.3024 | 0.1246 | 0.1561 |
| team_total | fav_strong | binary_direct | 0.0800 | 0.0200 | 5837 | 5677 | 0.9726 | 0.1649 | 0.2618 | 0.2380 | 0.2849 | 0.1215 | 0.1403 |
| team_total | fav_strong_low_total | binary_direct | 0.0400 | 0.0200 | 672 | 667 | 0.9926 | 0.1478 | 0.1914 | 0.1134 | 0.2698 | 0.1295 | 0.0619 |
| team_total | fav_strong_low_total | binary_direct | 0.0600 | 0.0200 | 672 | 653 | 0.9717 | 0.1286 | 0.1771 | 0.1008 | 0.2530 | 0.1246 | 0.0525 |
| team_total | fav_strong_low_total | binary_direct | 0.0800 | 0.0200 | 672 | 619 | 0.9211 | 0.1130 | 0.1546 | 0.0786 | 0.2343 | 0.1215 | 0.0331 |
| team_total | high_total | binary_direct | 0.0400 | 0.0200 | 7295 | 6195 | 0.8492 | 0.1684 | 0.2268 | 0.2034 | 0.2499 | 0.1295 | 0.0972 |
| team_total | high_total | binary_direct | 0.0600 | 0.0200 | 7295 | 5857 | 0.8029 | 0.1542 | 0.2213 | 0.1977 | 0.2442 | 0.1246 | 0.0967 |
| team_total | high_total | binary_direct | 0.0800 | 0.0200 | 7295 | 5421 | 0.7431 | 0.1427 | 0.2122 | 0.1874 | 0.2352 | 0.1215 | 0.0907 |

Discovery, for comparison (this is where the threshold was chosen; these numbers are exploratory):

| market | scenario | source | hold | threshold | n_rows | n_bets | bet_rate | mean_edge | roi | roi_lo | roi_hi | haircut_roi | roi_after_haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| match_total | all | count_nb | 0.0400 | 0.0000 | 63227 | 22978 | 0.3634 | 0.0256 | 0.0325 | 0.0199 | 0.0455 | 0.1106 | -0.0781 |
| match_total | all | count_nb | 0.0400 | 0.0200 | 63227 | 11078 | 0.1752 | 0.0433 | 0.0493 | 0.0310 | 0.0683 | 0.1295 | -0.0802 |
| match_total | all | count_nb | 0.0400 | 0.0500 | 63227 | 3032 | 0.0480 | 0.0728 | 0.0836 | 0.0456 | 0.1219 | 0.1529 | -0.0693 |
| match_total | all | count_nb | 0.0600 | 0.0000 | 63227 | 11409 | 0.1804 | 0.0229 | 0.0248 | 0.0063 | 0.0424 | 0.1075 | -0.0827 |
| match_total | all | count_nb | 0.0600 | 0.0200 | 63227 | 4809 | 0.0761 | 0.0423 | 0.0467 | 0.0185 | 0.0737 | 0.1246 | -0.0780 |
| match_total | all | count_nb | 0.0600 | 0.0500 | 63227 | 1220 | 0.0193 | 0.0731 | 0.0218 | -0.0352 | 0.0766 | 0.1511 | -0.1293 |
| match_total | all | count_nb | 0.0800 | 0.0000 | 63227 | 5100 | 0.0807 | 0.0217 | 0.0285 | 0.0020 | 0.0565 | 0.1017 | -0.0732 |
| match_total | all | count_nb | 0.0800 | 0.0200 | 63227 | 1988 | 0.0314 | 0.0423 | 0.0390 | -0.0058 | 0.0818 | 0.1215 | -0.0825 |
| match_total | all | count_nb | 0.0800 | 0.0500 | 63227 | 528 | 0.0084 | 0.0719 | 0.0846 | 0.0009 | 0.1727 | 0.1459 | -0.0614 |
| match_total | all | binary_direct | 0.0400 | 0.0000 | 43735 | 19534 | 0.4466 | 0.0348 | 0.0143 | 0.0006 | 0.0277 | 0.1106 | -0.0964 |
| match_total | all | binary_direct | 0.0400 | 0.0200 | 43735 | 11422 | 0.2612 | 0.0530 | 0.0357 | 0.0177 | 0.0544 | 0.1295 | -0.0938 |
| match_total | all | binary_direct | 0.0400 | 0.0500 | 43735 | 4839 | 0.1106 | 0.0800 | 0.0597 | 0.0311 | 0.0874 | 0.1529 | -0.0932 |
| match_total | all | binary_direct | 0.0600 | 0.0000 | 43735 | 11660 | 0.2666 | 0.0324 | 0.0157 | -0.0021 | 0.0335 | 0.1075 | -0.0918 |
| match_total | all | binary_direct | 0.0600 | 0.0200 | 43735 | 6560 | 0.1500 | 0.0505 | 0.0241 | -0.0003 | 0.0483 | 0.1246 | -0.1005 |
| match_total | all | binary_direct | 0.0600 | 0.0500 | 43735 | 2561 | 0.0586 | 0.0779 | 0.0727 | 0.0333 | 0.1107 | 0.1511 | -0.0784 |
| match_total | all | binary_direct | 0.0800 | 0.0000 | 43735 | 6792 | 0.1553 | 0.0300 | 0.0081 | -0.0143 | 0.0314 | 0.1017 | -0.0935 |
| match_total | all | binary_direct | 0.0800 | 0.0200 | 43735 | 3601 | 0.0823 | 0.0484 | 0.0436 | 0.0116 | 0.0756 | 0.1215 | -0.0779 |
| match_total | all | binary_direct | 0.0800 | 0.0500 | 43735 | 1283 | 0.0293 | 0.0765 | 0.0566 | 0.0008 | 0.1116 | 0.1459 | -0.0893 |
| team_total | all | count_nb | 0.0400 | 0.0000 | 126454 | 87823 | 0.6945 | 0.0929 | 0.0947 | 0.0879 | 0.1011 | 0.1106 | -0.0159 |
| team_total | all | count_nb | 0.0400 | 0.0200 | 126454 | 72486 | 0.5732 | 0.1105 | 0.1133 | 0.1061 | 0.1205 | 0.1295 | -0.0162 |
| team_total | all | count_nb | 0.0400 | 0.0500 | 126454 | 53901 | 0.4262 | 0.1368 | 0.1365 | 0.1284 | 0.1450 | 0.1529 | -0.0165 |
| team_total | all | count_nb | 0.0600 | 0.0000 | 126454 | 73042 | 0.5776 | 0.0889 | 0.0918 | 0.0849 | 0.0991 | 0.1075 | -0.0157 |
| team_total | all | count_nb | 0.0600 | 0.0200 | 126454 | 59927 | 0.4739 | 0.1062 | 0.1072 | 0.0995 | 0.1149 | 0.1246 | -0.0175 |
| team_total | all | count_nb | 0.0600 | 0.0500 | 126454 | 43864 | 0.3469 | 0.1326 | 0.1336 | 0.1247 | 0.1432 | 0.1511 | -0.0175 |
| team_total | all | count_nb | 0.0800 | 0.0000 | 126454 | 60634 | 0.4795 | 0.0848 | 0.0852 | 0.0773 | 0.0932 | 0.1017 | -0.0165 |
| team_total | all | count_nb | 0.0800 | 0.0200 | 126454 | 49087 | 0.3882 | 0.1024 | 0.1007 | 0.0916 | 0.1089 | 0.1215 | -0.0208 |
| team_total | all | count_nb | 0.0800 | 0.0500 | 126454 | 35361 | 0.2796 | 0.1288 | 0.1313 | 0.1211 | 0.1411 | 0.1459 | -0.0146 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0000 | 126454 | 83381 | 0.6594 | 0.0935 | 0.0961 | 0.0894 | 0.1031 | 0.1106 | -0.0145 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0200 | 126454 | 70551 | 0.5579 | 0.1088 | 0.1076 | 0.1000 | 0.1154 | 0.1295 | -0.0219 |
| team_total | all | count_nb_market_only | 0.0400 | 0.0500 | 126454 | 51635 | 0.4083 | 0.1358 | 0.1327 | 0.1236 | 0.1417 | 0.1529 | -0.0202 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0000 | 126454 | 71062 | 0.5620 | 0.0872 | 0.0861 | 0.0789 | 0.0932 | 0.1075 | -0.0214 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0200 | 126454 | 58078 | 0.4593 | 0.1045 | 0.1032 | 0.0953 | 0.1115 | 0.1246 | -0.0214 |
| team_total | all | count_nb_market_only | 0.0600 | 0.0500 | 126454 | 41020 | 0.3244 | 0.1337 | 0.1265 | 0.1172 | 0.1367 | 0.1511 | -0.0246 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0000 | 126454 | 58761 | 0.4647 | 0.0831 | 0.0820 | 0.0743 | 0.0903 | 0.1017 | -0.0197 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0200 | 126454 | 46480 | 0.3676 | 0.1024 | 0.0955 | 0.0860 | 0.1047 | 0.1215 | -0.0260 |
| team_total | all | count_nb_market_only | 0.0800 | 0.0500 | 126454 | 32611 | 0.2579 | 0.1315 | 0.1271 | 0.1156 | 0.1370 | 0.1459 | -0.0189 |
| team_total | all | binary_direct | 0.0400 | 0.0000 | 95568 | 65989 | 0.6905 | 0.0827 | 0.0925 | 0.0851 | 0.1004 | 0.1106 | -0.0181 |
| team_total | all | binary_direct | 0.0400 | 0.0200 | 95568 | 53561 | 0.5604 | 0.0996 | 0.1116 | 0.1035 | 0.1200 | 0.1295 | -0.0179 |
| team_total | all | binary_direct | 0.0400 | 0.0500 | 95568 | 38170 | 0.3994 | 0.1260 | 0.1392 | 0.1289 | 0.1489 | 0.1529 | -0.0137 |
| team_total | all | binary_direct | 0.0600 | 0.0000 | 95568 | 54008 | 0.5651 | 0.0782 | 0.0901 | 0.0817 | 0.0983 | 0.1075 | -0.0173 |
| team_total | all | binary_direct | 0.0600 | 0.0200 | 95568 | 42983 | 0.4498 | 0.0958 | 0.1074 | 0.0979 | 0.1170 | 0.1246 | -0.0173 |
| team_total | all | binary_direct | 0.0600 | 0.0500 | 95568 | 30149 | 0.3155 | 0.1220 | 0.1367 | 0.1258 | 0.1477 | 0.1511 | -0.0144 |
| team_total | all | binary_direct | 0.0800 | 0.0000 | 95568 | 43575 | 0.4560 | 0.0745 | 0.0864 | 0.0778 | 0.0953 | 0.1017 | -0.0153 |
| team_total | all | binary_direct | 0.0800 | 0.0200 | 95568 | 34281 | 0.3587 | 0.0921 | 0.1056 | 0.0954 | 0.1161 | 0.1215 | -0.0159 |
| team_total | all | binary_direct | 0.0800 | 0.0500 | 95568 | 23364 | 0.2445 | 0.1190 | 0.1325 | 0.1201 | 0.1453 | 0.1459 | -0.0134 |

Headline (confirmation, all matches, 6% hold, 2% edge threshold): 5976 bets from 42151 matches (0.142), mean modelled edge 0.0471, ROI 0.0140 [-0.0111, 0.0391]. After the step-4 haircut of 0.1246: **-0.1107**.

Team lines (confirmation, 6% hold, 2% edge threshold): 43833 bets from 84302 team-matches (0.520), mean edge 0.1143, ROI 0.1324 [0.1230, 0.1420]; after the haircut **0.0077**.

### Closing-line-value style calibration

In the bins where model and proxy disagree most, is the model or the proxy closer to the truth? Positive `delta_proxy_minus_model` = the model.

| bin | n | mean_abs_gap | ll_model | ll_proxy | delta_proxy_minus_model | obs_rate | mean_model | mean_proxy | split | market | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12646 | 0.0027 | 0.6902 | 0.6902 | -0.0000 | 0.4730 | 0.4708 | 0.4709 | discovery | match_total | count_nb |
| 2 | 12646 | 0.0085 | 0.6880 | 0.6884 | 0.0004 | 0.4673 | 0.4695 | 0.4703 | discovery | match_total | count_nb |
| 3 | 12645 | 0.0150 | 0.6880 | 0.6883 | 0.0003 | 0.4646 | 0.4686 | 0.4706 | discovery | match_total | count_nb |
| 4 | 12645 | 0.0233 | 0.6888 | 0.6902 | 0.0014 | 0.4691 | 0.4672 | 0.4704 | discovery | match_total | count_nb |
| 5 | 12645 | 0.0401 | 0.6875 | 0.6910 | 0.0035 | 0.4781 | 0.4758 | 0.4698 | discovery | match_total | count_nb |
| 1 | 8431 | 0.0027 | 0.6871 | 0.6872 | 0.0001 | 0.4583 | 0.4692 | 0.4689 | confirmation | match_total | count_nb |
| 2 | 8430 | 0.0084 | 0.6878 | 0.6881 | 0.0002 | 0.4617 | 0.4709 | 0.4690 | confirmation | match_total | count_nb |
| 3 | 8430 | 0.0152 | 0.6865 | 0.6873 | 0.0009 | 0.4595 | 0.4766 | 0.4704 | confirmation | match_total | count_nb |
| 4 | 8430 | 0.0251 | 0.6895 | 0.6889 | -0.0006 | 0.4635 | 0.4876 | 0.4709 | confirmation | match_total | count_nb |
| 5 | 8430 | 0.0468 | 0.6893 | 0.6916 | 0.0023 | 0.4974 | 0.5162 | 0.4716 | confirmation | match_total | count_nb |
| 1 | 25291 | 0.0060 | 0.6846 | 0.6846 | 0.0001 | 0.4494 | 0.4523 | 0.4521 | discovery | team_total | count_nb |
| 2 | 25291 | 0.0196 | 0.6835 | 0.6843 | 0.0008 | 0.4508 | 0.4529 | 0.4525 | discovery | team_total | count_nb |
| 3 | 25291 | 0.0380 | 0.6801 | 0.6838 | 0.0038 | 0.4527 | 0.4513 | 0.4527 | discovery | team_total | count_nb |
| 4 | 25291 | 0.0643 | 0.6758 | 0.6843 | 0.0086 | 0.4508 | 0.4484 | 0.4522 | discovery | team_total | count_nb |
| 5 | 25290 | 0.1199 | 0.6524 | 0.6849 | 0.0326 | 0.4581 | 0.4570 | 0.4522 | discovery | team_total | count_nb |
| 1 | 16861 | 0.0066 | 0.6819 | 0.6819 | 0.0000 | 0.4347 | 0.4504 | 0.4495 | confirmation | team_total | count_nb |
| 2 | 16861 | 0.0221 | 0.6818 | 0.6830 | 0.0012 | 0.4468 | 0.4564 | 0.4521 | confirmation | team_total | count_nb |
| 3 | 16860 | 0.0427 | 0.6780 | 0.6829 | 0.0049 | 0.4407 | 0.4538 | 0.4500 | confirmation | team_total | count_nb |
| 4 | 16860 | 0.0706 | 0.6676 | 0.6808 | 0.0131 | 0.4388 | 0.4563 | 0.4510 | confirmation | team_total | count_nb |
| 5 | 16860 | 0.1291 | 0.6408 | 0.6884 | 0.0476 | 0.4668 | 0.4774 | 0.4520 | confirmation | team_total | count_nb |

## 7. (b) Style layer on the StatsBomb overlap

Join: 1517 StatsBomb matches of the 2015/2016 Premier League, La Liga, Serie A and Ligue 1 against the odds table. The team-name map is built explicitly by fixture-date overlap (maximum-weight bipartite matching per division), not assumed: 80 names mapped, join rate 1.000, final scores agree on 1.000 of joined matches, 0 needed a one-day shift.

Every model here is a multiplicative correction to the same C0 proxy fitted as a regularised Poisson GLM (the sample is small, so a linear correction is the low-variance choice), at team-corner level with matches held out together. `cv_all` is match-grouped 5-fold over the whole overlap -- more power, no temporal holdout; `forward_confirmation` is the protocol's chronological headline.

| feature_set | scheme | n_rows | n_matches | poisson_deviance | delta_vs_B0 | ci_lo | ci_hi | delta_ll_4.5 | delta_ll_5.5 |
|---|---|---|---|---|---|---|---|---|---|
| B0_proxy | cv_all | 2634 | 1317 | 1.4617 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B0_proxy | cv_discovery | 1580 | 790 | 1.5072 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B0_proxy | forward_confirmation | 1054 | 527 | 1.3935 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B1_odds_event_market | cv_all | 2634 | 1317 | 1.4265 | 0.0351 | 0.0212 | 0.0486 | 0.0069 | 0.0065 |
| B1_odds_event_market | cv_discovery | 1580 | 790 | 1.4669 | 0.0403 | 0.0237 | 0.0573 | 0.0080 | 0.0076 |
| B1_odds_event_market | forward_confirmation | 1054 | 527 | 1.3699 | 0.0236 | -0.0010 | 0.0480 | 0.0042 | 0.0043 |
| B2a_plus_own_style | cv_all | 2634 | 1317 | 1.4237 | 0.0380 | 0.0225 | 0.0536 | 0.0076 | 0.0068 |
| B2a_plus_own_style | cv_discovery | 1580 | 790 | 1.4667 | 0.0405 | 0.0208 | 0.0597 | 0.0082 | 0.0073 |
| B2a_plus_own_style | forward_confirmation | 1054 | 527 | 1.3627 | 0.0307 | 0.0046 | 0.0553 | 0.0059 | 0.0056 |
| B2_plus_style | cv_all | 2634 | 1317 | 1.4248 | 0.0369 | 0.0198 | 0.0544 | 0.0072 | 0.0062 |
| B2_plus_style | cv_discovery | 1580 | 790 | 1.4660 | 0.0412 | 0.0194 | 0.0632 | 0.0084 | 0.0073 |
| B2_plus_style | forward_confirmation | 1054 | 527 | 1.3692 | 0.0243 | -0.0026 | 0.0511 | 0.0041 | 0.0039 |
| B2c_plus_sb_corner_history | cv_all | 2634 | 1317 | 1.4249 | 0.0368 | 0.0195 | 0.0548 | 0.0072 | 0.0062 |
| B2c_plus_sb_corner_history | cv_discovery | 1580 | 790 | 1.4660 | 0.0411 | 0.0190 | 0.0640 | 0.0083 | 0.0073 |
| B2c_plus_sb_corner_history | forward_confirmation | 1054 | 527 | 1.3697 | 0.0238 | -0.0037 | 0.0508 | 0.0040 | 0.0038 |
| B3_plus_imputed_state | cv_all | 2634 | 1317 | 1.4243 | 0.0374 | 0.0200 | 0.0561 | 0.0073 | 0.0065 |
| B3_plus_imputed_state | cv_discovery | 1580 | 790 | 1.4655 | 0.0417 | 0.0190 | 0.0641 | 0.0087 | 0.0077 |
| B3_plus_imputed_state | forward_confirmation | 1054 | 527 | 1.3681 | 0.0253 | -0.0023 | 0.0524 | 0.0041 | 0.0043 |
| B3opp_opponent_compactness_only | cv_all | 2634 | 1317 | 1.4241 | 0.0376 | 0.0203 | 0.0555 | 0.0073 | 0.0065 |
| B3opp_opponent_compactness_only | cv_discovery | 1580 | 790 | 1.4652 | 0.0420 | 0.0196 | 0.0645 | 0.0087 | 0.0078 |
| B3opp_opponent_compactness_only | forward_confirmation | 1054 | 527 | 1.3678 | 0.0256 | -0.0019 | 0.0527 | 0.0040 | 0.0041 |
| B3own_own_compactness_only | cv_all | 2634 | 1317 | 1.4250 | 0.0366 | 0.0196 | 0.0543 | 0.0072 | 0.0062 |
| B3own_own_compactness_only | cv_discovery | 1580 | 790 | 1.4664 | 0.0407 | 0.0189 | 0.0630 | 0.0083 | 0.0072 |
| B3own_own_compactness_only | forward_confirmation | 1054 | 527 | 1.3696 | 0.0239 | -0.0033 | 0.0504 | 0.0041 | 0.0040 |
| S_style_only | cv_all | 2634 | 1317 | 1.4346 | 0.0270 | 0.0119 | 0.0428 | 0.0060 | 0.0044 |
| S_style_only | cv_discovery | 1580 | 790 | 1.4751 | 0.0320 | 0.0118 | 0.0524 | 0.0072 | 0.0056 |
| S_style_only | forward_confirmation | 1054 | 527 | 1.3758 | 0.0177 | -0.0048 | 0.0403 | 0.0035 | 0.0021 |
| S_imputed_only | cv_all | 2634 | 1317 | 1.4442 | 0.0175 | 0.0065 | 0.0282 | 0.0036 | 0.0042 |
| S_imputed_only | cv_discovery | 1580 | 790 | 1.4864 | 0.0208 | 0.0060 | 0.0349 | 0.0039 | 0.0051 |
| S_imputed_only | forward_confirmation | 1054 | 527 | 1.3789 | 0.0146 | 0.0012 | 0.0297 | 0.0030 | 0.0031 |

Reading of the hypothesis as stated -- *teams that cross heavily and get shots blocked against compact opponents generate corners well above their average*:

- Style alone (crossing, blocked shots, passes into the box, PPDA, possession, line height, for both teams, no corner history at all) beats the rolling corner mean: 0.0270 [0.0119, 0.0428] in CV, 0.0177 [-0.0048, 0.0403] forward. The channel is real.
- The imputed defensive state alone (block depth, defensive line, deep-block share of both teams, from the completed programme's students) also beats it: 0.0175 [0.0065, 0.0282] in CV, 0.0146 [0.0012, 0.0297] forward.
- Added on top of what the odds table already says, the team's **own** style is worth 0.0029 deviance in CV and 0.0071 forward -- small, and the only increment in this layer that is positive in both evaluations.
- Adding the **opponent's** style on top of that is worth -0.0012 in CV and -0.0064 forward: nothing, or slightly negative.
- Adding the opponent's **imputed compactness** (block depth, defensive line, deep-block share, measured from their prior matches by the completed programme's students) on top of both styles is worth 0.0007 in CV and 0.0013 forward; the team's own compactness -0.0002 and -0.0004. Every one of these increments is an order of magnitude smaller than the interval around it.
- Extra StatsBomb corner history (`B2c`) is worth -0.0001, which is the useful control: the odds table's own corner history has already extracted what there is.

**So the hypothesis splits in two and only one half survives.** *Teams that cross heavily* is weakly supported -- own style adds a little, and style on its own beats the rolling mean outright. *Against compact opponents* is not: neither the opponent's measured style nor its imputed defensive shape adds anything once the team's own profile is known. And the whole layer is underpowered for the sizes involved -- 527 confirmation matches, with forward intervals wide enough to contain both zero and twice the effect.

## 8. Limitations

- There are no corner lines in this data. The opponent is a *simulated* book, not a real one; the haircut in section 3 is the only evidence about the distance between the two, and it is measured on a different (sharper) market.
- The style layer is one season of four leagues; its confirmation half is 527 matches, so its intervals are wide and a null there is weak evidence, not strong.
- Corner counts are recorded on 51% of the odds table; divisions and seasons without them are absent, and the corner level drifts over time, which is why the proxy's league-level tracking matters as much as its team ratios.
- The proxy's negative-binomial dispersion and Platt recalibration are fitted on all discovery rows, including the held-out folds of the discovery cross-validation; with two parameters over tens of thousands of rows the effect is negligible, but it flatters C0 rather than the models under test.
- The team-line result rests on features derived from the 1X2 market. A model whose edge over a rolling mean comes from re-expressing the match odds has, by construction, no edge over a book that also reads the match odds. The honest statement is the negative one: a rolling-mean proxy is a poor model of how a real book prices a team corner line, so the +13% ROI against it is a measure of the proxy's weakness, not of a market inefficiency.
- The haircut is transferred from goals to corners unchanged. Goals is the market a book works hardest on, so the transfer probably overstates a corner book's edge; corner props also carry a wider margin than the 6.6% seen on the goals line, which understates the bar. Neither correction is measurable here, and the net result (team lines roughly break even, match totals lose about 11 points of ROI) is far enough from the boundary in the match-total case and far enough inside the haircut's own interval in the team-line case that neither would flip.
- Discovery spans 2000-2019 and confirmation 2019-2026; the corner rate falls from 10.55 to 9.77 per match across that boundary. That drift is why a free-standing classifier is punished on confirmation and why every model here is specified as a correction to the proxy, which tracks the league level by construction.

