# 06 The haircut, reconciled: one definition, one bar

Code: `research/scenario_ev/haircut.py`. Inputs: the committed result tables of stages 01-04 plus the raw odds table (for the universe overlap only). Nothing here refits a model.

```
cd /home/user/geo-model
python -m research.scenario_ev.haircut     # ~20 s
```

## 1. Why this exists

Every EV claim in this phase is conditional on protocol step 4: a real bookmaker is sharper than a rolling mean, and by how much is measured on total goals, the one count market with an observed line. Stages 01-04 each ran that measurement, but expressed it in money three different ways on overlapping universes. Which market 'survives the haircut' therefore depended on which stage's arithmetic a reader picked up. This module fixes one definition and applies it to everything.

## 2. How independent are the four measurements?

The synthesis originally called the nats gap 'measured four times independently'. The odds table says otherwise:

| universe_a | universe_b | n_a | n_b | n_intersection | jaccard | a_subset_of_b |
|---|---|---|---|---|---|---|
| 01 cards goals check | 02 corners goals check | 156684 | 110752 | 110752 | 0.7068 | False |
| 01 cards goals check | 04 fouls goals check | 156684 | 110362 | 110362 | 0.7044 | False |
| 02 corners goals check | 01 cards goals check | 110752 | 156684 | 110752 | 0.7068 | True |
| 02 corners goals check | 04 fouls goals check | 110752 | 110362 | 110362 | 0.9965 | False |
| 04 fouls goals check | 01 cards goals check | 110362 | 156684 | 110362 | 0.7044 | True |
| 04 fouls goals check | 02 corners goals check | 110362 | 110752 | 110362 | 0.9965 | True |

Read the `a_subset_of_b` column. The foul pool is a **subset** of the corner pool (110362 of 110752 matches, Jaccard 0.9965), which is why stages 02 and 04 agree to five decimals on both the proxy and the Bet365 log-loss: they are the same measurement, run twice. The corner pool is in turn a subset of the card pool (110752 of 156684), because stage 01's goals check does not condition on card counts at all -- it conditions on a result, a price and prior history, so its pool is the whole priced table. Stage 03's four-league universe is nested inside all of them. **The three pools are a chain of nested sets, not three samples.**

What separates the card measurement from the corner/foul one is therefore not the matches available but the **chronological cut**, which puts the confirmation halves in different eras:

| universe_a | universe_b | split_a | split_b | n_conf_a | n_conf_b | n_intersection | share_of_a |
|---|---|---|---|---|---|---|---|
| 01 cards goals check | 02 corners goals check | 2017-11-30 | 2019-08-31 | 65203 | 48889 | 48889 | 0.7498 |
| 01 cards goals check | 04 fouls goals check | 2017-11-30 | 2019-08-31 | 65203 | 48887 | 48887 | 0.7498 |
| 02 corners goals check | 01 cards goals check | 2019-08-31 | 2017-11-30 | 48889 | 65203 | 48889 | 1.0000 |
| 02 corners goals check | 04 fouls goals check | 2019-08-31 | 2019-08-31 | 48889 | 48887 | 48887 | 1.0000 |
| 04 fouls goals check | 01 cards goals check | 2019-08-31 | 2017-11-30 | 48887 | 65203 | 48887 | 1.0000 |
| 04 fouls goals check | 02 corners goals check | 2019-08-31 | 2019-08-31 | 48887 | 48889 | 48887 | 1.0000 |

And that does not separate them either: the corner/foul confirmation half is 100.0% contained in the card confirmation half (48889 of its 65203 matches). The card measurement is the corner measurement plus about 16,314 extra matches from the 2017-12 to 2019-08 window, which is why its gap is larger: that window is where the rolling mean falls furthest behind.

**The phase has one construction, on one nested match pool, scored on two nested row sets** -- not four independent replications, and not even two. All four stages also call the same de-vig helper (`common.novig_two_way`) on the same Bet365 price columns. What the agreement does establish is that the number is not an artefact of one stage's proxy tuning, capacity choice or split date, since those differ between the stages and the answer does not. That is worth something. It is not four independent estimates, and the synthesis no longer says it is.

Bet365's realised two-way overround on the Over/Under 2.5 price, which is what the actual-prices round trip charges:

| universe | era | n | mean_overround |
|---|---|---|---|
| 01 cards goals check | all years | 156684 | 0.0660 |
| 01 cards goals check | cards confirmation (from 2017-12) | 65203 | 0.0592 |
| 01 cards goals check | corner/foul confirmation (from 2019-09) | 52059 | 0.0584 |
| 02 corners goals check | all years | 110752 | 0.0629 |
| 02 corners goals check | cards confirmation (from 2017-12) | 61091 | 0.0585 |
| 02 corners goals check | corner/foul confirmation (from 2019-09) | 48889 | 0.0577 |
| 04 fouls goals check | all years | 110362 | 0.0628 |
| 04 fouls goals check | cards confirmation (from 2017-12) | 61085 | 0.0585 |
| 04 fouls goals check | corner/foul confirmation (from 2019-09) | 48887 | 0.0577 |

## 3. The gap in nats

| stage | universe | split | n | gap_nats | ci_lo | ci_hi | source_table |
|---|---|---|---|---|---|---|---|
| 01 cards | 01 cards goals check | confirmation | 64790 | 0.0111 | 0.0101 | 0.0122 | 01_cards_c_goals_gap |
| 02 corners | 02 corners goals check | confirmation | 42140 | 0.0089 | 0.0080 | 0.0099 | 02_corners_goals_haircut_metrics |
| 03 pass counts | 03 four leagues (nested in 02 / 04) | confirmation | 11438 | 0.0106 | 0.0081 | 0.0130 | 03_pass_goals_gap |
| 04 fouls | 04 fouls goals check | confirmation | 42004 | 0.0090 | 0.0079 | 0.0100 | 04_fouls_d_goals_metrics |
| 01 cards | 01 cards goals check | discovery | 90538 | 0.0058 | 0.0050 | 0.0064 | 01_cards_c_goals_gap |
| 02 corners | 02 corners goals check | discovery | 53531 | 0.0052 | 0.0044 | 0.0060 | 02_corners_goals_haircut_metrics |
| 03 pass counts | 03 four leagues (nested in 02 / 04) | discovery | 17232 | 0.0105 | 0.0088 | 0.0123 | 03_pass_goals_gap |
| 04 fouls | 04 fouls goals check | discovery | 53327 | 0.0052 | 0.0044 | 0.0059 | 04_fouls_d_goals_metrics |

This is the robust, comparable form of the bar and the one the synthesis leans on: a real book is roughly 0.009-0.011 nats better than a shrunk, discovery-tuned, recalibrated rolling mean on a count market. Section 2 is the caveat to read it with: these are nested row sets scored over two overlapping eras, so the four intervals are not four independent estimates and the spread between 0.0089 and 0.0111 is mostly the era, not sampling.

## 4. The three definitions in ROI, side by side

* **`book_earns_one_way`** -- ROI a real book makes betting its de-vigged goals price into prices built from the rolling-mean proxy at the stated hold. One-directional: it charges the profit that evaporates but not the loss that appears, so it is a LOWER BOUND.
* **`margin_matched_round_trip`** -- ROI(our goals model vs proxy prices at hold h) minus ROI(our goals model vs the real book's no-vig price re-priced at the same hold h). Both legs carry the same margin, so the difference is sharpness alone. This is the phase reference.
* **`actual_prices_round_trip`** -- the same round trip, but the second leg uses Bet365's posted prices at whatever hold Bet365 actually charged. It mixes a margin difference into the sharpness difference whenever that hold differs from h.

| definition | stage | universe | hold | haircut_roi_points | leg_vs_proxy | leg_vs_book |
|---|---|---|---|---|---|---|
| actual_prices_round_trip | 01 cards | 01 cards goals check | 0.0400 | 0.3053 | 0.1760 | -0.1293 |
| actual_prices_round_trip | 01 cards | 01 cards goals check | 0.0600 | 0.3150 | 0.1857 | -0.1293 |
| actual_prices_round_trip | 01 cards | 01 cards goals check | 0.0800 | 0.2982 | 0.1689 | -0.1293 |
| actual_prices_round_trip | 04 fouls | 04 fouls goals check | 0.0400 | 0.2050 | 0.1159 | -0.0891 |
| actual_prices_round_trip | 04 fouls | 04 fouls goals check | 0.0600 | 0.2067 | 0.1176 | -0.0891 |
| actual_prices_round_trip | 04 fouls | 04 fouls goals check | 0.0800 | 0.2097 | 0.1206 | -0.0891 |
| book_earns_one_way | 02 corners | 02 corners goals check | 0.0400 | 0.1295 | 0.1295 |  |
| book_earns_one_way | 02 corners | 02 corners goals check | 0.0600 | 0.1246 | 0.1246 |  |
| book_earns_one_way | 02 corners | 02 corners goals check | 0.0800 | 0.1215 | 0.1215 |  |
| book_earns_one_way | 03 pass counts | 03 four leagues (nested in 02 / 04) | 0.0400 | 0.1162 | 0.1162 |  |
| book_earns_one_way | 03 pass counts | 03 four leagues (nested in 02 / 04) | 0.0600 | 0.1162 | 0.1162 |  |
| book_earns_one_way | 03 pass counts | 03 four leagues (nested in 02 / 04) | 0.0800 | 0.1162 | 0.1162 |  |
| margin_matched_round_trip | 01 cards | 01 cards goals check | 0.0400 | 0.2898 | 0.1760 | -0.1138 |
| margin_matched_round_trip | 01 cards | 01 cards goals check | 0.0600 | 0.3099 | 0.1857 | -0.1241 |
| margin_matched_round_trip | 01 cards | 01 cards goals check | 0.0800 | 0.3043 | 0.1689 | -0.1354 |
| margin_matched_round_trip | 04 fouls | 04 fouls goals check | 0.0400 | 0.1736 | 0.1159 | -0.0577 |
| margin_matched_round_trip | 04 fouls | 04 fouls goals check | 0.0600 | 0.2073 | 0.1176 | -0.0897 |
| margin_matched_round_trip | 04 fouls | 04 fouls goals check | 0.0800 | 0.2282 | 0.1206 | -0.1076 |

**The phase reference is `margin_matched_round_trip`** at the matching universe: 04 fouls 0.2073, 01 cards 0.3099 at a 6% hold. It is the definition a bettor actually experiences and the only one that holds the margin fixed between its two legs. The one-directional `book_earns_one_way` number is kept everywhere as a **lower bound**, never as the bar.

## 5. Every published confirmation betting row, after each haircut

`haircut_*` is what was subtracted; `after_*` is the point estimate that remains; `after_*_lo` applies the same subtraction to the ROI interval's lower bound. Card rows are haircut against the card universe, everything else against the corner/foul universe on which it was simulated.

| candidate | market | scenario | source | hold | n_bets | bet_rate | roi | roi_lo | haircut_margin_matched_round_trip | after_margin_matched_round_trip | after_margin_matched_round_trip_lo | after_book_earns_one_way | after_actual_prices_round_trip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cards | match_cards_proxy_line | all | count_nb | 0.0400 | 551 | 0.0085 | 0.3441 | 0.2545 | 0.2898 | 0.0543 | -0.0354 | 0.2146 | 0.0388 |
| cards | match_cards_proxy_line | all | count_nb | 0.0600 | 376 | 0.0058 | 0.3054 | 0.1922 | 0.3099 | -0.0045 | -0.1177 | 0.1807 | -0.0097 |
| cards | match_cards_proxy_line | all | count_nb | 0.0800 | 263 | 0.0041 | 0.3446 | 0.2155 | 0.3043 | 0.0403 | -0.0888 | 0.2231 | 0.0464 |
| cards | player_carded_0.5 | all | binary_direct | 0.0400 | 822 | 0.0543 | 0.2395 | 0.0345 | 0.2898 | -0.0504 | -0.2553 | 0.1099 | -0.0659 |
| cards | player_carded_0.5 | all | binary_direct | 0.0600 | 722 | 0.0477 | 0.2651 | 0.0621 | 0.3099 | -0.0447 | -0.2477 | 0.1405 | -0.0499 |
| cards | player_carded_0.5 | all | binary_direct | 0.0800 | 626 | 0.0413 | 0.2605 | 0.0335 | 0.3043 | -0.0439 | -0.2709 | 0.1390 | -0.0378 |
| corners | match_total | all | count_nb | 0.0400 | 10133 | 0.2404 | 0.0141 | -0.0044 | 0.1736 | -0.1595 | -0.1780 | -0.1154 | -0.1909 |
| corners | match_total | all | count_nb | 0.0600 | 5976 | 0.1418 | 0.0140 | -0.0111 | 0.2073 | -0.1933 | -0.2184 | -0.1107 | -0.1927 |
| corners | match_total | all | count_nb | 0.0800 | 3174 | 0.0753 | 0.0138 | -0.0210 | 0.2282 | -0.2143 | -0.2492 | -0.1077 | -0.1959 |
| corners | match_total | fav_strong | count_nb | 0.0400 | 3064 | 0.6609 | 0.0298 | -0.0073 | 0.1736 | -0.1438 | -0.1809 | -0.0997 | -0.1752 |
| corners | match_total | fav_strong | count_nb | 0.0600 | 2502 | 0.5397 | 0.0154 | -0.0243 | 0.2073 | -0.1919 | -0.2316 | -0.1092 | -0.1913 |
| corners | match_total | fav_strong | count_nb | 0.0800 | 1701 | 0.3669 | 0.0173 | -0.0301 | 0.2282 | -0.2108 | -0.2582 | -0.1041 | -0.1924 |
| corners | match_total | fav_strong_low_total | count_nb | 0.0400 | 310 | 0.6339 | 0.0721 | -0.0402 | 0.1736 | -0.1014 | -0.2137 | -0.0574 | -0.1329 |
| corners | match_total | fav_strong_low_total | count_nb | 0.0600 | 207 | 0.4233 | 0.0533 | -0.0889 | 0.2073 | -0.1540 | -0.2962 | -0.0713 | -0.1534 |
| corners | match_total | fav_strong_low_total | count_nb | 0.0800 | 121 | 0.2474 | 0.0837 | -0.0972 | 0.2282 | -0.1444 | -0.3254 | -0.0378 | -0.1260 |
| corners | match_total | high_total | count_nb | 0.0400 | 2384 | 0.4773 | 0.0344 | -0.0081 | 0.1736 | -0.1392 | -0.1817 | -0.0951 | -0.1706 |
| corners | match_total | high_total | count_nb | 0.0600 | 1776 | 0.3556 | 0.0342 | -0.0147 | 0.2073 | -0.1731 | -0.2220 | -0.0905 | -0.1726 |
| corners | match_total | high_total | count_nb | 0.0800 | 1140 | 0.2282 | 0.0385 | -0.0169 | 0.2282 | -0.1896 | -0.2451 | -0.0830 | -0.1712 |
| corners | team_total | all | count_nb | 0.0400 | 51842 | 0.6150 | 0.1380 | 0.1288 | 0.1736 | -0.0356 | -0.0448 | 0.0085 | -0.0670 |
| corners | team_total | all | count_nb | 0.0600 | 43833 | 0.5200 | 0.1324 | 0.1230 | 0.2073 | -0.0749 | -0.0843 | 0.0077 | -0.0744 |
| corners | team_total | all | count_nb | 0.0800 | 36684 | 0.4351 | 0.1301 | 0.1196 | 0.2282 | -0.0981 | -0.1086 | 0.0086 | -0.0797 |
| corners | team_total | fav_strong | count_nb | 0.0400 | 9271 | 0.9999 | 0.2860 | 0.2656 | 0.1736 | 0.1124 | 0.0920 | 0.1565 | 0.0810 |
| corners | team_total | fav_strong | count_nb | 0.0600 | 9267 | 0.9995 | 0.2619 | 0.2419 | 0.2073 | 0.0546 | 0.0346 | 0.1373 | 0.0552 |
| corners | team_total | fav_strong | count_nb | 0.0800 | 9211 | 0.9934 | 0.2406 | 0.2210 | 0.2282 | 0.0125 | -0.0072 | 0.1191 | 0.0309 |
| corners | team_total | fav_strong_low_total | count_nb | 0.0400 | 977 | 0.9990 | 0.1791 | 0.1136 | 0.1736 | 0.0055 | -0.0600 | 0.0496 | -0.0259 |
| corners | team_total | fav_strong_low_total | count_nb | 0.0600 | 969 | 0.9908 | 0.1575 | 0.0920 | 0.2073 | -0.0498 | -0.1153 | 0.0329 | -0.0492 |
| corners | team_total | fav_strong_low_total | count_nb | 0.0800 | 916 | 0.9366 | 0.1443 | 0.0780 | 0.2282 | -0.0838 | -0.1501 | 0.0228 | -0.0654 |
| corners | team_total | high_total | count_nb | 0.0400 | 8817 | 0.8826 | 0.2369 | 0.2161 | 0.1736 | 0.0634 | 0.0426 | 0.1074 | 0.0319 |
| corners | team_total | high_total | count_nb | 0.0600 | 8471 | 0.8479 | 0.2212 | 0.2007 | 0.2073 | 0.0139 | -0.0066 | 0.0966 | 0.0145 |
| corners | team_total | high_total | count_nb | 0.0800 | 8061 | 0.8069 | 0.2099 | 0.1891 | 0.2282 | -0.0183 | -0.0390 | 0.0884 | 0.0001 |
| fouls | match_fouls | all | count_nb | 0.0400 | 13349 | 0.3177 | 0.1367 | 0.1204 | 0.1736 | -0.0369 | -0.0531 | 0.0072 | -0.0684 |
| fouls | match_fouls | all | count_nb | 0.0600 | 10238 | 0.2437 | 0.1395 | 0.1216 | 0.2073 | -0.0678 | -0.0857 | 0.0149 | -0.0672 |
| fouls | match_fouls | all | count_nb | 0.0800 | 7690 | 0.1830 | 0.1398 | 0.1202 | 0.2282 | -0.0884 | -0.1079 | 0.0183 | -0.0700 |
| fouls | team_fouls | all | count_nb | 0.0400 | 21137 | 0.2515 | 0.1115 | 0.0981 | 0.1736 | -0.0620 | -0.0754 | -0.0180 | -0.0935 |
| fouls | team_fouls | all | count_nb | 0.0600 | 15304 | 0.1821 | 0.1090 | 0.0943 | 0.2073 | -0.0983 | -0.1130 | -0.0156 | -0.0977 |
| fouls | team_fouls | all | count_nb | 0.0800 | 11045 | 0.1314 | 0.0958 | 0.0783 | 0.2282 | -0.1324 | -0.1499 | -0.0257 | -0.1139 |
| pass_counts | completed_passes | all | model_att x proxy_rate | 0.0400 | 10120 | 0.7753 | 0.2562 | 0.2315 | 0.1736 | 0.0827 | 0.0579 | 0.1267 | 0.0512 |
| pass_counts | completed_passes | all | proxy_att x model_rate | 0.0400 | 8566 | 0.6562 | 0.1813 | 0.1558 | 0.1736 | 0.0077 | -0.0177 | 0.0518 | -0.0237 |
| pass_counts | completed_passes | all | model_att x model_rate | 0.0400 | 10405 | 0.7971 | 0.2660 | 0.2432 | 0.1736 | 0.0924 | 0.0696 | 0.1365 | 0.0610 |
| pass_counts | completed_passes | all | model_att x proxy_rate | 0.0600 | 9471 | 0.7256 | 0.2459 | 0.2215 | 0.2073 | 0.0386 | 0.0142 | 0.1212 | 0.0391 |
| pass_counts | completed_passes | all | proxy_att x model_rate | 0.0600 | 7627 | 0.5843 | 0.1683 | 0.1415 | 0.2073 | -0.0389 | -0.0657 | 0.0437 | -0.0384 |
| pass_counts | completed_passes | all | model_att x model_rate | 0.0600 | 9815 | 0.7519 | 0.2504 | 0.2273 | 0.2073 | 0.0431 | 0.0200 | 0.1257 | 0.0436 |
| pass_counts | completed_passes | all | model_att x proxy_rate | 0.0800 | 8821 | 0.6758 | 0.2389 | 0.2140 | 0.2282 | 0.0107 | -0.0142 | 0.1174 | 0.0291 |
| pass_counts | completed_passes | all | proxy_att x model_rate | 0.0800 | 6746 | 0.5168 | 0.1631 | 0.1366 | 0.2282 | -0.0650 | -0.0916 | 0.0416 | -0.0466 |
| pass_counts | completed_passes | all | model_att x model_rate | 0.0800 | 9201 | 0.7049 | 0.2393 | 0.2156 | 0.2282 | 0.0112 | -0.0126 | 0.1178 | 0.0296 |
| pass_counts | pass_attempts | all | model_att | 0.0400 | 10085 | 0.7726 | 0.2592 | 0.2365 | 0.1736 | 0.0857 | 0.0630 | 0.1297 | 0.0542 |
| pass_counts | pass_attempts | all | model_att | 0.0600 | 9432 | 0.7226 | 0.2530 | 0.2299 | 0.2073 | 0.0458 | 0.0226 | 0.1284 | 0.0463 |
| pass_counts | pass_attempts | all | model_att | 0.0800 | 8777 | 0.6724 | 0.2453 | 0.2222 | 0.2282 | 0.0171 | -0.0060 | 0.1238 | 0.0356 |

**Under the phase reference, 9 of 48 rows keep a lower bound above zero**: `corners/team_total/fav_strong/count_nb` at 4% (+0.112 [+0.092]), `corners/team_total/fav_strong/count_nb` at 6% (+0.055 [+0.035]), `corners/team_total/high_total/count_nb` at 4% (+0.063 [+0.043]), `pass_counts/completed_passes/all/model_att x proxy_rate` at 4% (+0.083 [+0.058]), `pass_counts/completed_passes/all/model_att x model_rate` at 4% (+0.092 [+0.070]), `pass_counts/completed_passes/all/model_att x proxy_rate` at 6% (+0.039 [+0.014]), `pass_counts/completed_passes/all/model_att x model_rate` at 6% (+0.043 [+0.020]), `pass_counts/pass_attempts/all/model_att` at 4% (+0.086 [+0.063]), `pass_counts/pass_attempts/all/model_att` at 6% (+0.046 [+0.023]).

Under the one-directional lower bound 33 of 48 rows have a positive point estimate, which is exactly the freedom this module removes: the lenient definition promotes rows the reference definition kills, and the two were measured on essentially the same matches.

## 6. What this changes in the phase's conclusions

* The corner **team** line splits. On all rows it clears the one-directional lower bound (+0.008 at a 6% hold) and does **not** clear the reference (-0.075 [-0.084]), so the phase's previously reported '+0.008 residual' was an artefact of the lenient definition. Inside `fav_strong` it clears the reference too (+0.055 [+0.035]) -- but that rule bets 99.95% of the matches in the scenario, so it is not a selective edge, it is the proxy being wrong about every strong-favourite match because it cannot see the 1X2 price. Stage 02's ablation puts 95% of the team model's gain in exactly that market block.
* The **pass-count** residual survives the reference haircut (+0.043 [+0.020] at a 6% hold). The haircut does not argue it away and the synthesis does not pretend it does. What argues it away is stated in stage 03: every feature carrying the advantage is a public rolling average, the model claims to disagree with the rolling mean about 1.8x as hard as a real book does on goals, and against a book allowed to set its own line at the goals-matched disagreement the ROI falls to +0.090 [+0.064, +0.118] before any haircut.
* One instrument points the other way and is reported rather than suppressed. `03_pass_c_relative_scale.parquet` normalises each market by its own proxy's log score: on goals a real book beats the proxy by 1.54% of the proxy's log loss, while our pass model beats its proxy by 2.00% of its count log score, so a book that was only *goals-equivalently* sharp on pass props would still score 0.0176 nats per player-match **worse** than our model. Stage 03 reports that ratio for completeness and explicitly declines to use it, because a count log score is dominated by irreducible entropy and its relative gaps are not comparable to a binary log loss at a line. It is quoted here so that a reader is not left with the impression that every instrument points at a null.
* Card and foul rows are negative under the reference definition at the 6% headline hold, which is what the synthesis already said. Two card rows turn slightly positive at 4% and 8% (+0.054 and +0.040) on point estimate only, with lower bounds of -0.035 and -0.089; the 6% row is -0.004 [-0.118]. The card conclusion is 'indistinguishable from zero and thin' rather than 'clearly negative', and the 0.4-0.9% bet rates are why.

## 7. Limitations

* The reference haircut is still a **transfer**: it is measured on goals and applied to corners, cards, fouls and pass props, on the assumption that a side-market book is as sharp relative to a rolling mean as a goals book is. Goals is the market a book works hardest on, so the transfer probably overstates a side book's sharpness; side-market margins are wider than the 5.8-6.6% observed here, which pushes the bar the other way. Neither correction is measurable in this data.
* A flat ROI subtraction across markets is crude. Bet rates differ by two orders of magnitude between the card rules (0.4-0.9%) and the pass rules (70-80%), and the haircut was measured at bet rates of 25-50%. The nats form of the bar (section 3) does not have this problem and is the form to prefer.
* The card universe's margin-matched leg re-prices Bet365's no-vig probability at the synthetic hold. That removes the margin mismatch but keeps the assumption that the de-vig is proportional; stage 01 checked Shin and additive de-vigs and found they move the nats gap by ten-thousandths.

