# NFL 07 - forward receiving-props test: next-game receiving yards and targets

Machine-written by `python -m research.privileged_tracking.nfl.props_forward`. Question: a
player-prop price is forward-looking. Given a receiver's play-by-play history up to week t
(with or without the imputed tracking state of those plays), how well is his NEXT game's
receiving line predicted, and does the imputed state add anything to the event-only history?

## Headline

- **next_rec_yards, population `all`** (n = 11,422 test receiver-games, 2020-2022): R2 BASE_STD 0.213, BASE_L3 0.134, HIST 0.294 (MAE 19.92); deltas vs HIST (squared error, positive = better): HIST+IMP_F0T +0.51 [-2.84, +3.71] (week-clustered [-2.47, +3.65]); HIST+IMP_F1T +3.08 [+0.10, +6.18] (week-clustered [+0.15, +6.00]); HIST+IMP_F1 +1.56 [-1.05, +4.00] (week-clustered [-0.79, +3.91]); HIST+IMP_ALL +2.45 [-1.12, +5.98] (week-clustered [-0.81, +5.74]); HIST+NGS -0.92 [-4.87, +2.94] (week-clustered [-5.34, +3.10]).
- **next_rec_yards, population `regular`** (n = 5,835 test receiver-games, 2020-2022): R2 BASE_STD 0.198, BASE_L3 0.082, HIST 0.232 (MAE 23.51); deltas vs HIST (squared error, positive = better): HIST+IMP_F0T +1.47 [-3.89, +6.80] (week-clustered [-5.22, +7.65]); HIST+IMP_F1T -2.94 [-8.34, +2.58] (week-clustered [-8.19, +2.28]); HIST+IMP_F1 -0.25 [-4.43, +4.13] (week-clustered [-3.71, +3.14]); HIST+IMP_ALL -2.98 [-9.09, +3.01] (week-clustered [-9.69, +4.11]); HIST+NGS +0.81 [-4.34, +5.94] (week-clustered [-4.53, +5.73]).
- **next_n_targets, population `all`** (n = 11,422 test receiver-games, 2020-2022): R2 BASE_STD 0.344, BASE_L3 0.289, HIST 0.398 (MAE 1.87); deltas vs HIST (squared error, positive = better): HIST+IMP_F0T +0.02 [-0.00, +0.04] (week-clustered [-0.01, +0.04]); HIST+IMP_F1T +0.02 [-0.00, +0.04] (week-clustered [-0.00, +0.04]); HIST+IMP_F1 +0.01 [-0.01, +0.03] (week-clustered [-0.02, +0.04]); HIST+IMP_ALL +0.01 [-0.02, +0.03] (week-clustered [-0.02, +0.03]); HIST+NGS -0.00 [-0.03, +0.02] (week-clustered [-0.03, +0.02]).
- **next_n_targets, population `regular`** (n = 5,835 test receiver-games, 2020-2022): R2 BASE_STD 0.253, BASE_L3 0.159, HIST 0.279 (MAE 2.20); deltas vs HIST (squared error, positive = better): HIST+IMP_F0T -0.00 [-0.04, +0.03] (week-clustered [-0.03, +0.03]); HIST+IMP_F1T -0.01 [-0.05, +0.03] (week-clustered [-0.05, +0.03]); HIST+IMP_F1 -0.01 [-0.05, +0.02] (week-clustered [-0.04, +0.01]); HIST+IMP_ALL -0.01 [-0.06, +0.03] (week-clustered [-0.05, +0.02]); HIST+NGS -0.04 [-0.08, -0.00] (week-clustered [-0.08, -0.00]).

## Protocol

* **Rows.** Targeted pass plays of the 2018-2022 regular seasons (`payoff_pbp_<season>.parquet`,
  the stage-04 season frames with every student applied) aggregated per (season, week,
  receiver): `n_targets`, `receptions`, `rec_yards` (yards gained on completions), mean
  `air_yards` / `cp` / `epa`, `target_share`, and the mean imputed separation / cushion /
  target depth of the targeted plays. One row per receiver-game that has a later game with
  >= 1 target in the same season; the target is that next game's `rec_yards` / `n_targets`.
* **Features** (all from games <= t of the same season): current game (`cur_`), season-to-date
  mean (`std_`), last-three-game mean (`l3_`) of each quantity, games played, week. Feature
  sets: `HIST` (23 columns); `HIST+IMP_F0T` (29 columns); `HIST+IMP_F1T` (32 columns); `HIST+IMP_F1` (26 columns); `HIST+IMP_ALL` (41 columns); `HIST+NGS` (32 columns).
  `HIST+NGS` adds the official NGS weekly `avg_separation` / `avg_cushion` /
  `avg_intended_air_yards` (published for receivers with >= 5 targets; NaN otherwise) as a
  privileged reference. Baselines: `BASE_STD` / `BASE_L3` predict the season-to-date /
  last-three mean of the target; `BASE_GLOBAL` the training seasons' mean.
* **Populations.** `all` = every receiver-game with a later game in the season; `regular` = >= 3 games played and >= 3 targets per game season-to-date (props-eligible).
* **Split.** Rolling origin: test seasons 2020, 2021, 2022, each
  predicted by LightGBM trained on the seasons before it (rounds by early stopping on the
  last training season, refit on every training season with that round count; 2020 uses
  2018-2019 with 2019 as the validation season). Metrics pooled over the three test seasons
  and per season.
* **Statistics.** Paired bootstrap (2,000 resamples, seed 0) of per-row
  squared error and absolute error against `HIST` (delta = loss(HIST) - loss(model),
  positive = the model is better; `delta_r2` = delta squared error / test variance), with a
  (season, week)-clustered interval; `refit_spread_se` = max - min of the mean squared error
  of the pair's members under seeds 0, 1, 2
  (`HIST`, `HIST+IMP_F1T`, `HIST+IMP_ALL` are refit); a delta inside it is noise whatever its
  interval says.
* **Leakage.** The imputed columns come from students trained on the 2017 tracked games only;
  no feature uses a week later than t; NGS fields of weeks <= t are after-the-fact for those
  weeks but precede the target game.

## Sample

| population | season | receiver_games | receivers | ngs_available | mean_next_rec_yards | mean_next_n_targets |
|---|---|---|---|---|---|---|
| all | 2018 | 3610 | 441 | 0.312 | 32.833 | 4.329 |
| all | 2019 | 3591 | 436 | 0.316 | 32.492 | 4.340 |
| all | 2020 | 3701 | 446 | 0.327 | 32.076 | 4.259 |
| all | 2021 | 3877 | 458 | 0.328 | 31.034 | 4.243 |
| all | 2022 | 3844 | 449 | 0.305 | 30.189 | 4.116 |
| regular | 2018 | 1875 | 239 | 0.487 | 42.502 | 5.498 |
| regular | 2019 | 1864 | 230 | 0.488 | 42.114 | 5.504 |
| regular | 2020 | 1898 | 232 | 0.510 | 41.889 | 5.513 |
| regular | 2021 | 2031 | 244 | 0.504 | 39.921 | 5.363 |
| regular | 2022 | 1906 | 224 | 0.501 | 40.230 | 5.373 |

## next_rec_yards

### population `all` (every receiver-game with a later game in the season)

Pooled test seasons 2020-2022:

| model | n | r2 | mae | rmse | bias |
|---|---|---|---|---|---|
| BASE_GLOBAL | 11422 | -0.0014 | 24.7789 | 32.0232 | 1.3149 |
| BASE_STD | 11422 | 0.2128 | 20.5732 | 28.3926 | 0.4595 |
| BASE_L3 | 11422 | 0.1339 | 21.4565 | 29.7823 | 0.2380 |
| HIST | 11422 | 0.2944 | 19.9238 | 26.8813 | 0.6344 |
| HIST+IMP_F0T | 11422 | 0.2949 | 19.8853 | 26.8718 | 0.6324 |
| HIST+IMP_F1T | 11422 | 0.2974 | 19.8638 | 26.8240 | 0.6560 |
| HIST+IMP_F1 | 11422 | 0.2959 | 19.8755 | 26.8522 | 0.6688 |
| HIST+IMP_ALL | 11422 | 0.2968 | 19.8568 | 26.8358 | 0.6577 |
| HIST+NGS | 11422 | 0.2935 | 19.9555 | 26.8985 | 0.6916 |

Deltas vs `HIST` (pooled test rows):

| model | n | delta_r2 | delta squared error [CI] (clustered) | delta absolute error [CI] (clustered) | refit_spread_se | inside_refit_noise |
|---|---|---|---|---|---|---|
| BASE_GLOBAL | 11422 | -0.2958 | -302.88 [-325.22, -282.08] (week-clustered [-323.56, -281.99]) | -4.855 [-5.115, -4.611] (week-clustered [-5.157, -4.555]) | 1.9648 | False |
| BASE_STD | 11422 | -0.0816 | -83.53 [-98.88, -67.41] (week-clustered [-120.50, -53.18]) | -0.649 [-0.818, -0.465] (week-clustered [-1.013, -0.339]) | 1.9648 | False |
| BASE_L3 | 11422 | -0.1605 | -164.38 [-183.83, -145.01] (week-clustered [-195.29, -138.60]) | -1.533 [-1.746, -1.303] (week-clustered [-1.832, -1.283]) | 1.9648 | False |
| HIST+IMP_F0T | 11422 | 0.0005 | +0.51 [-2.84, +3.71] (week-clustered [-2.47, +3.65]) | +0.038 [-0.007, +0.081] (week-clustered [-0.003, +0.081]) | 1.9648 | True |
| HIST+IMP_F1T | 11422 | 0.0030 | +3.08 [+0.10, +6.18] (week-clustered [+0.15, +6.00]) | +0.060 [+0.018, +0.103] (week-clustered [+0.019, +0.103]) | 1.9648 | False |
| HIST+IMP_F1 | 11422 | 0.0015 | +1.56 [-1.05, +4.00] (week-clustered [-0.79, +3.91]) | +0.048 [+0.010, +0.084] (week-clustered [+0.015, +0.082]) | 1.9648 | True |
| HIST+IMP_ALL | 11422 | 0.0024 | +2.45 [-1.12, +5.98] (week-clustered [-0.81, +5.74]) | +0.067 [+0.017, +0.117] (week-clustered [+0.022, +0.110]) | 1.9648 | False |
| HIST+NGS | 11422 | -0.0009 | -0.92 [-4.87, +2.94] (week-clustered [-5.34, +3.10]) | -0.032 [-0.083, +0.021] (week-clustered [-0.093, +0.022]) | 1.9648 | True |

R2 per test season:

| model | 2020 | 2021 | 2022 |
|---|---|---|---|
| BASE_GLOBAL | -0.0003 | -0.0020 | -0.0036 |
| BASE_L3 | 0.1371 | 0.1278 | 0.1351 |
| BASE_STD | 0.2100 | 0.2053 | 0.2221 |
| HIST | 0.2868 | 0.2987 | 0.2965 |
| HIST+IMP_ALL | 0.2891 | 0.2983 | 0.3018 |
| HIST+IMP_F0T | 0.2892 | 0.2971 | 0.2971 |
| HIST+IMP_F1 | 0.2877 | 0.3002 | 0.2987 |
| HIST+IMP_F1T | 0.2892 | 0.2967 | 0.3052 |
| HIST+NGS | 0.2845 | 0.2931 | 0.3019 |

### population `regular` (>= 3 games played and >= 3 targets per game season-to-date (props-eligible))

Pooled test seasons 2020-2022:

| model | n | r2 | mae | rmse | bias |
|---|---|---|---|---|---|
| BASE_GLOBAL | 5835 | -0.0014 | 27.5051 | 35.0344 | 1.3570 |
| BASE_STD | 5835 | 0.1979 | 23.9849 | 31.3562 | 2.4736 |
| BASE_L3 | 5835 | 0.0817 | 25.4106 | 33.5497 | 1.9909 |
| HIST | 5835 | 0.2321 | 23.5113 | 30.6808 | 0.6568 |
| HIST+IMP_F0T | 5835 | 0.2332 | 23.5122 | 30.6569 | 0.7524 |
| HIST+IMP_F1T | 5835 | 0.2297 | 23.5144 | 30.7286 | 0.6518 |
| HIST+IMP_F1 | 5835 | 0.2318 | 23.5144 | 30.6848 | 0.6644 |
| HIST+IMP_ALL | 5835 | 0.2296 | 23.5285 | 30.7293 | 0.7120 |
| HIST+NGS | 5835 | 0.2327 | 23.4478 | 30.6676 | 0.6311 |

Deltas vs `HIST` (pooled test rows):

| model | n | delta_r2 | delta squared error [CI] (clustered) | delta absolute error [CI] (clustered) | refit_spread_se | inside_refit_noise |
|---|---|---|---|---|---|---|
| BASE_GLOBAL | 5835 | -0.2334 | -286.10 [-315.37, -257.32] (week-clustered [-308.73, -263.82]) | -3.994 [-4.333, -3.665] (week-clustered [-4.364, -3.639]) | 3.1834 | False |
| BASE_STD | 5835 | -0.0342 | -41.90 [-60.17, -24.26] (week-clustered [-60.20, -24.69]) | -0.474 [-0.686, -0.258] (week-clustered [-0.699, -0.246]) | 3.1834 | False |
| BASE_L3 | 5835 | -0.1503 | -184.27 [-213.75, -155.86] (week-clustered [-211.45, -159.16]) | -1.899 [-2.249, -1.556] (week-clustered [-2.214, -1.580]) | 3.1834 | False |
| HIST+IMP_F0T | 5835 | 0.0012 | +1.47 [-3.89, +6.80] (week-clustered [-5.22, +7.65]) | -0.001 [-0.075, +0.070] (week-clustered [-0.089, +0.082]) | 3.1834 | True |
| HIST+IMP_F1T | 5835 | -0.0024 | -2.94 [-8.34, +2.58] (week-clustered [-8.19, +2.28]) | -0.003 [-0.071, +0.068] (week-clustered [-0.084, +0.070]) | 3.1834 | True |
| HIST+IMP_F1 | 5835 | -0.0002 | -0.25 [-4.43, +4.13] (week-clustered [-3.71, +3.14]) | -0.003 [-0.059, +0.053] (week-clustered [-0.053, +0.047]) | 3.1834 | True |
| HIST+IMP_ALL | 5835 | -0.0024 | -2.98 [-9.09, +3.01] (week-clustered [-9.69, +4.11]) | -0.017 [-0.097, +0.055] (week-clustered [-0.106, +0.065]) | 3.1834 | True |
| HIST+NGS | 5835 | 0.0007 | +0.81 [-4.34, +5.94] (week-clustered [-4.53, +5.73]) | +0.064 [-0.009, +0.137] (week-clustered [-0.013, +0.141]) | 3.1834 | True |

R2 per test season:

| model | 2020 | 2021 | 2022 |
|---|---|---|---|
| BASE_GLOBAL | -0.0001 | -0.0041 | -0.0015 |
| BASE_L3 | 0.0628 | 0.0979 | 0.0819 |
| BASE_STD | 0.1740 | 0.2062 | 0.2120 |
| HIST | 0.2069 | 0.2564 | 0.2301 |
| HIST+IMP_ALL | 0.2005 | 0.2515 | 0.2345 |
| HIST+IMP_F0T | 0.2113 | 0.2540 | 0.2319 |
| HIST+IMP_F1 | 0.2064 | 0.2543 | 0.2322 |
| HIST+IMP_F1T | 0.2078 | 0.2483 | 0.2305 |
| HIST+NGS | 0.2074 | 0.2532 | 0.2351 |

## next_n_targets

### population `all` (every receiver-game with a later game in the season)

Pooled test seasons 2020-2022:

| model | n | r2 | mae | rmse | bias |
|---|---|---|---|---|---|
| BASE_GLOBAL | 11422 | -0.0010 | 2.5329 | 3.1432 | 0.1057 |
| BASE_STD | 11422 | 0.3441 | 1.8771 | 2.5444 | -0.0341 |
| BASE_L3 | 11422 | 0.2888 | 1.9438 | 2.6494 | -0.0068 |
| HIST | 11422 | 0.3977 | 1.8653 | 2.4383 | 0.0386 |
| HIST+IMP_F0T | 11422 | 0.3994 | 1.8592 | 2.4347 | 0.0362 |
| HIST+IMP_F1T | 11422 | 0.3994 | 1.8574 | 2.4347 | 0.0364 |
| HIST+IMP_F1 | 11422 | 0.3987 | 1.8594 | 2.4361 | 0.0368 |
| HIST+IMP_ALL | 11422 | 0.3984 | 1.8628 | 2.4368 | 0.0345 |
| HIST+NGS | 11422 | 0.3974 | 1.8624 | 2.4388 | 0.0394 |

Deltas vs `HIST` (pooled test rows):

| model | n | delta_r2 | delta squared error [CI] (clustered) | delta absolute error [CI] (clustered) | refit_spread_se | inside_refit_noise |
|---|---|---|---|---|---|---|
| BASE_GLOBAL | 11422 | -0.3986 | -3.93 [-4.15, -3.72] (week-clustered [-4.13, -3.75]) | -0.668 [-0.698, -0.639] (week-clustered [-0.699, -0.637]) | 0.0150 | False |
| BASE_STD | 11422 | -0.0536 | -0.53 [-0.64, -0.42] (week-clustered [-0.77, -0.34]) | -0.012 [-0.028, +0.004] (week-clustered [-0.033, +0.009]) | 0.0150 | False |
| BASE_L3 | 11422 | -0.1088 | -1.07 [-1.22, -0.94] (week-clustered [-1.26, -0.91]) | -0.079 [-0.098, -0.058] (week-clustered [-0.098, -0.060]) | 0.0150 | False |
| HIST+IMP_F0T | 11422 | 0.0018 | +0.02 [-0.00, +0.04] (week-clustered [-0.01, +0.04]) | +0.006 [+0.003, +0.009] (week-clustered [+0.002, +0.010]) | 0.0150 | False |
| HIST+IMP_F1T | 11422 | 0.0017 | +0.02 [-0.00, +0.04] (week-clustered [-0.00, +0.04]) | +0.008 [+0.004, +0.011] (week-clustered [+0.004, +0.012]) | 0.0265 | True |
| HIST+IMP_F1 | 11422 | 0.0011 | +0.01 [-0.01, +0.03] (week-clustered [-0.02, +0.04]) | +0.006 [+0.003, +0.009] (week-clustered [+0.002, +0.010]) | 0.0150 | True |
| HIST+IMP_ALL | 11422 | 0.0007 | +0.01 [-0.02, +0.03] (week-clustered [-0.02, +0.03]) | +0.002 [-0.001, +0.006] (week-clustered [-0.002, +0.007]) | 0.0261 | True |
| HIST+NGS | 11422 | -0.0003 | -0.00 [-0.03, +0.02] (week-clustered [-0.03, +0.02]) | +0.003 [-0.001, +0.007] (week-clustered [-0.001, +0.007]) | 0.0150 | True |

R2 per test season:

| model | 2020 | 2021 | 2022 |
|---|---|---|---|
| BASE_GLOBAL | -0.0006 | -0.0004 | -0.0032 |
| BASE_L3 | 0.2896 | 0.2722 | 0.3040 |
| BASE_STD | 0.3409 | 0.3247 | 0.3660 |
| HIST | 0.3893 | 0.3918 | 0.4112 |
| HIST+IMP_ALL | 0.3869 | 0.3943 | 0.4132 |
| HIST+IMP_F0T | 0.3923 | 0.3939 | 0.4113 |
| HIST+IMP_F1 | 0.3899 | 0.3953 | 0.4103 |
| HIST+IMP_F1T | 0.3913 | 0.3936 | 0.4126 |
| HIST+NGS | 0.3858 | 0.3927 | 0.4130 |

### population `regular` (>= 3 games played and >= 3 targets per game season-to-date (props-eligible))

Pooled test seasons 2020-2022:

| model | n | r2 | mae | rmse | bias |
|---|---|---|---|---|---|
| BASE_GLOBAL | 5835 | -0.0005 | 2.6445 | 3.2762 | 0.0762 |
| BASE_STD | 5835 | 0.2533 | 2.2309 | 2.8304 | 0.1728 |
| BASE_L3 | 5835 | 0.1594 | 2.3494 | 3.0030 | 0.2119 |
| HIST | 5835 | 0.2788 | 2.1979 | 2.7816 | -0.0020 |
| HIST+IMP_F0T | 5835 | 0.2787 | 2.1967 | 2.7818 | 0.0041 |
| HIST+IMP_F1T | 5835 | 0.2777 | 2.1973 | 2.7836 | -0.0065 |
| HIST+IMP_F1 | 5835 | 0.2776 | 2.1991 | 2.7839 | 0.0005 |
| HIST+IMP_ALL | 5835 | 0.2776 | 2.1990 | 2.7839 | -0.0056 |
| HIST+NGS | 5835 | 0.2749 | 2.2040 | 2.7890 | 0.0128 |

Deltas vs `HIST` (pooled test rows):

| model | n | delta_r2 | delta squared error [CI] (clustered) | delta absolute error [CI] (clustered) | refit_spread_se | inside_refit_noise |
|---|---|---|---|---|---|---|
| BASE_GLOBAL | 5835 | -0.2793 | -3.00 [-3.25, -2.74] (week-clustered [-3.21, -2.78]) | -0.447 [-0.482, -0.413] (week-clustered [-0.479, -0.413]) | 0.0136 | False |
| BASE_STD | 5835 | -0.0255 | -0.27 [-0.41, -0.15] (week-clustered [-0.40, -0.16]) | -0.033 [-0.052, -0.014] (week-clustered [-0.052, -0.016]) | 0.0136 | False |
| BASE_L3 | 5835 | -0.1194 | -1.28 [-1.49, -1.08] (week-clustered [-1.45, -1.12]) | -0.151 [-0.180, -0.123] (week-clustered [-0.175, -0.128]) | 0.0136 | False |
| HIST+IMP_F0T | 5835 | -0.0001 | -0.00 [-0.04, +0.03] (week-clustered [-0.03, +0.03]) | +0.001 [-0.004, +0.007] (week-clustered [-0.003, +0.006]) | 0.0136 | True |
| HIST+IMP_F1T | 5835 | -0.0011 | -0.01 [-0.05, +0.03] (week-clustered [-0.05, +0.03]) | +0.001 [-0.006, +0.007] (week-clustered [-0.005, +0.006]) | 0.0136 | True |
| HIST+IMP_F1 | 5835 | -0.0012 | -0.01 [-0.05, +0.02] (week-clustered [-0.04, +0.01]) | -0.001 [-0.006, +0.004] (week-clustered [-0.006, +0.004]) | 0.0136 | True |
| HIST+IMP_ALL | 5835 | -0.0012 | -0.01 [-0.06, +0.03] (week-clustered [-0.05, +0.02]) | -0.001 [-0.007, +0.005] (week-clustered [-0.007, +0.005]) | 0.0204 | True |
| HIST+NGS | 5835 | -0.0039 | -0.04 [-0.08, -0.00] (week-clustered [-0.08, -0.00]) | -0.006 [-0.013, +0.000] (week-clustered [-0.012, +0.000]) | 0.0136 | False |

R2 per test season:

| model | 2020 | 2021 | 2022 |
|---|---|---|---|
| BASE_GLOBAL | -0.0000 | -0.0019 | -0.0008 |
| BASE_L3 | 0.1289 | 0.1954 | 0.1507 |
| BASE_STD | 0.2188 | 0.2781 | 0.2607 |
| HIST | 0.2459 | 0.3041 | 0.2841 |
| HIST+IMP_ALL | 0.2479 | 0.3039 | 0.2786 |
| HIST+IMP_F0T | 0.2509 | 0.3018 | 0.2811 |
| HIST+IMP_F1 | 0.2455 | 0.3028 | 0.2822 |
| HIST+IMP_F1T | 0.2499 | 0.3060 | 0.2747 |
| HIST+NGS | 0.2393 | 0.2974 | 0.2861 |

## Refit noise

| target | population | model | mean_se_by_seed | refit_spread_se |
|---|---|---|---|---|
| next_rec_yards | all | HIST | 722.605, 721.130, 720.640 | 1.9648 |
| next_rec_yards | all | HIST+IMP_F1T | 719.529, 720.043, 719.806 | 0.5140 |
| next_rec_yards | all | HIST+IMP_ALL | 720.159, 720.921, 719.779 | 1.1424 |
| next_rec_yards | regular | HIST | 941.312, 942.697, 944.495 | 3.1834 |
| next_rec_yards | regular | HIST+IMP_F1T | 944.248, 944.463, 941.685 | 2.7777 |
| next_rec_yards | regular | HIST+IMP_ALL | 944.291, 946.795, 945.028 | 2.5047 |
| next_n_targets | all | HIST | 5.945, 5.947, 5.932 | 0.0150 |
| next_n_targets | all | HIST+IMP_F1T | 5.928, 5.921, 5.947 | 0.0265 |
| next_n_targets | all | HIST+IMP_ALL | 5.938, 5.916, 5.942 | 0.0261 |
| next_n_targets | regular | HIST | 7.737, 7.741, 7.751 | 0.0136 |
| next_n_targets | regular | HIST+IMP_F1T | 7.749, 7.741, 7.746 | 0.0077 |
| next_n_targets | regular | HIST+IMP_ALL | 7.750, 7.767, 7.770 | 0.0204 |

## Caveats

* The target is the receiver's next game with >= 1 target, so rows are conditioned on the
  player being targeted again (injuries, benchings and byes are not modelled); a props line
  is only posted for players expected to play, which the `regular` population approximates.
* No market lines are on disk, so this measures predictive skill against event-only history,
  not against a bookmaker's number; a gain that is inside the refit spread or whose clustered
  interval covers zero would not move a price.
* Season-to-date aggregates restart every season (no prior-season carry-over), which limits
  every model equally in the first weeks; `n_games` and `week` let the model weight them.
* The F1 student reads the outcome flags of the plays it aggregates (after the fact); it is
  admissible here because those plays precede the target game, but it encodes past outcomes,
  not tracking.

Timing: 122 s on 2 threads.
