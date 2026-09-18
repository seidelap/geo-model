# Soccer 01 - build: 360 defensive-state targets and event-only features

Source: StatsBomb open data (`data/raw/privileged/sb`). Outputs under `processed_dir('soccer')`: `events360.parquet` (one row per event with a 360 freeze frame) and `events_no360.parquet` (event-only features for matches without 360). Nothing here is modelled; this is the data layer for the imputation and payoff stages.

## Run summary

- matches processed: 2090 of 2090 (417 with usable 360, 1673 without, of which 9 are 360-flagged matches whose frames were unusable, see bottom)
- `events360.parquet`: 1,357,506 rows from 1,357,626 frames (1,551,841 events in those matches; 0 frames on events without a location, which are unusable: `y_frame_orientation = no_location`)
- `events_no360.parquet`: 756,282 rows = all 41,687 Shots (41,279 with a `shot.freeze_frame`) + a fixed random 25% of Pass/Carry events per match (seeded by match id; `--frac 1.0` keeps all). 406,938 passes and 307,657 carries kept.
- wall time 12.2 min with 2 worker processes
- 360 frame orientation: 1,274,497 frames ok (93.9%), 63,333 repaired (4.7%: mirrored 37,230, mirrored + flag swap 26,103), 19,676 unusable (1.4%; all `y_*` geometry targets NaN, `y_reliable` = 0). See the orientation section.
- `f_after_*` columns (realised pass trajectory, carry end, duration) describe what happened AFTER the instant of the event; they are event-only but not pre-instant state (`instant` = post in the column table).
- penalty shoot-outs (period 5) are dropped; goals for the score state are counted in periods 1-4 only; all windows and the sequence block reset at each period start
- coordinates in every row are in the event team's attacking frame (goal at x=120); opponent events are mirrored with (120-x, 80-y) when they enter windows/sequence
- privileged fields (`shot.freeze_frame`, `statsbomb_xg`, `one_on_one`, `open_goal`) are emitted only under the `sff_` / `oracle_` prefixes; `post_` columns are outcomes. Event-only feature sets must be selected as `f_*` + `seq_*` columns only, and results reported with and without the post-instant `f_after_*` subset.

## Matches and events per competition

| competition | season | gender | has_360 | n_matches | n_events_total | n_frames | frame_coverage | n_rows | n_shots | n_pass | n_carry |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. Bundesliga | 2023/2024 | male | True | 31 | 125905 | 108394 | 0.861 | 108394 | 819 | 30203 | 26256 |
| FIFA World Cup | 2022 | male | True | 64 | 234637 | 203882 | 0.869 | 203848 | 1412 | 57393 | 49406 |
| La Liga | 2020/2021 | male | True | 35 | 139030 | 128840 | 0.927 | 128840 | 834 | 36912 | 31491 |
| Ligue 1 | 2021/2022 | male | True | 26 | 101766 | 86352 | 0.849 | 86352 | 660 | 23816 | 21722 |
| Ligue 1 | 2022/2023 | male | True | 31 | 125464 | 108222 | 0.863 | 108222 | 798 | 29684 | 27375 |
| Major League Soccer | 2023 | male | True | 3 | 10629 | 9600 | 0.903 | 9600 | 68 | 2603 | 2215 |
| UEFA Euro | 2020 | male | True | 51 | 192664 | 166871 | 0.866 | 166871 | 1215 | 46237 | 39661 |
| UEFA Euro | 2024 | male | True | 51 | 187924 | 164530 | 0.876 | 164512 | 1281 | 46202 | 40393 |
| UEFA Women's Euro | 2022 | female | True | 30 | 102046 | 90850 | 0.890 | 90850 | 842 | 24800 | 20514 |
| UEFA Women's Euro | 2025 | female | True | 31 | 105658 | 90559 | 0.857 | 90530 | 847 | 24124 | 20808 |
| Women's World Cup | 2023 | female | True | 64 | 226118 | 199526 | 0.882 | 199487 | 1591 | 51665 | 43113 |
| 1. Bundesliga | 2023/2024 | male | False | 3 | 11860 | 10213 |  | 1578 | 73 | 838 | 667 |
| African Cup of Nations | 2023 | male | False | 52 | 162903 | 1 |  | 20996 | 1184 | 11128 | 8684 |
| Copa America | 2024 | male | False | 32 | 100324 | 0 |  | 13264 | 751 | 7050 | 5463 |
| FIFA World Cup | 2018 | male | False | 64 | 227825 | 0 |  | 30201 | 1667 | 15697 | 12837 |
| La Liga | 2015/2016 | male | False | 380 | 1295354 | 0 |  | 169367 | 9168 | 91251 | 68948 |
| Ligue 1 | 2015/2016 | male | False | 377 | 1358593 | 0 |  | 173190 | 8814 | 93688 | 70688 |
| Ligue 1 | 2022/2023 | male | False | 1 | 4026 | 3548 |  | 532 | 20 | 294 | 218 |
| Major League Soccer | 2023 | male | False | 3 | 11157 | 10079 |  | 1513 | 76 | 763 | 674 |
| Premier League | 2015/2016 | male | False | 380 | 1313773 | 0 |  | 171090 | 9908 | 92275 | 68907 |
| Serie A | 2015/2016 | male | False | 380 | 1353739 | 0 |  | 174144 | 9998 | 93738 | 70408 |
| UEFA Women's Euro | 2022 | female | False | 1 | 3095 | 0 |  | 407 | 28 | 216 | 163 |

`n_rows` counts output rows (frames for 360 matches; shots + sampled passes/carries otherwise). `frame_coverage` = frames / events in the match.

## 360 visibility

| metric | scope | n | mean | std | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|
| y_n_teammates_visible | all | 1337830 | 6.730 | 1.832 | 4.000 | 6.000 | 7.000 | 8.000 | 9.000 |
| y_n_opponents_visible | all | 1337830 | 7.898 | 2.181 | 4.000 | 6.000 | 8.000 | 10.000 | 11.000 |
| y_visible_area_frac | all | 1357506 | 0.289 | 0.096 | 0.155 | 0.215 | 0.277 | 0.355 | 0.461 |
| y_opp_keeper_visible | all | 1337830 | 0.191 |  |  |  |  |  |  |
| y_actor_visible | all | 1357506 | 0.999 |  |  |  |  |  |  |
| y_reliable | all | 1357506 | 0.733 |  |  |  |  |  |  |

Share of reliable frames (>= 7 opponents visible): 0.733 (n = 1,357,506).

### By event type (14 most frequent)

| f_type | n | frame_ok_share | reliable_share | opp_visible_mean | tm_visible_mean | keeper_visible_share | visible_area_mean | in_possession_share |
|---|---|---|---|---|---|---|---|---|
| Pass | 373639 | 1.000 | 0.755 | 8.019 | 6.735 | 0.188 | 0.297 | 0.946 |
| Ball Receipt* | 371155 | 0.949 | 0.730 | 8.083 | 6.619 | 0.203 | 0.297 | 0.959 |
| Carry | 322954 | 0.999 | 0.746 | 7.946 | 6.703 | 0.180 | 0.297 | 0.959 |
| Pressure | 118572 | 1.000 | 0.715 | 7.505 | 7.022 | 0.155 | 0.278 | 0.134 |
| Ball Recovery | 36464 | 1.000 | 0.659 | 7.267 | 6.717 | 0.199 | 0.254 | 0.780 |
| Duel | 23688 | 0.996 | 0.692 | 7.476 | 7.163 | 0.140 | 0.266 | 0.456 |
| Clearance | 15064 | 1.000 | 0.471 | 6.317 | 8.010 | 0.010 | 0.210 | 0.048 |
| Block | 14765 | 0.999 | 0.634 | 7.147 | 7.309 | 0.138 | 0.238 | 0.189 |
| Dribble | 11198 | 0.998 | 0.850 | 8.306 | 5.965 | 0.358 | 0.263 | 0.902 |
| Shot | 10367 | 0.999 | 0.895 | 8.879 | 5.004 | 0.939 | 0.185 | 0.984 |
| Miscontrol | 9925 | 0.998 | 0.836 | 8.272 | 5.983 | 0.318 | 0.269 | 0.855 |
| Interception | 8975 | 0.999 | 0.610 | 6.957 | 7.433 | 0.065 | 0.278 | 0.330 |
| Foul Committed | 8947 | 0.999 | 0.755 | 7.778 | 6.858 | 0.194 | 0.283 | 0.204 |
| Foul Won | 8482 | 0.973 | 0.755 | 7.874 | 6.768 | 0.126 | 0.284 | 0.810 |

### By competition

| competition | season | n | reliable_share | opp_visible_mean | visible_area_mean |
|---|---|---|---|---|---|
| 1. Bundesliga | 2023/2024 | 108394 | 0.792 | 8.283 | 0.302 |
| FIFA World Cup | 2022 | 203848 | 0.692 | 7.602 | 0.292 |
| La Liga | 2020/2021 | 128840 | 0.854 | 8.714 | 0.321 |
| Ligue 1 | 2021/2022 | 86352 | 0.863 | 8.760 | 0.326 |
| Ligue 1 | 2022/2023 | 108222 | 0.884 | 8.925 | 0.338 |
| Major League Soccer | 2023 | 9600 | 0.622 | 7.081 | 0.252 |
| UEFA Euro | 2020 | 166871 | 0.703 | 7.689 | 0.284 |
| UEFA Euro | 2024 | 164512 | 0.795 | 8.336 | 0.305 |
| UEFA Women's Euro | 2022 | 90850 | 0.590 | 6.983 | 0.242 |
| UEFA Women's Euro | 2025 | 90530 | 0.670 | 7.366 | 0.257 |
| Women's World Cup | 2023 | 199487 | 0.599 | 7.034 | 0.247 |

### By ball zone (possession-team events only)

| zone | n | reliable_share | opp_visible_mean |
|---|---|---|---|
| own_third | 258221 | 0.402 | 6.029 |
| middle | 554926 | 0.824 | 8.393 |
| final | 316051 | 0.915 | 9.133 |

## 360 frame orientation (StatsBomb paired-event defect)

For paired events StatsBomb stores the 360 frame of the *other* team's event: the coordinates are in the other team's attacking frame and, for some pairs, the `teammate` flags are also from the other team's perspective. Detected per row by comparing the flagged actor with the event location (`y_actor_dist_to_event`; 0 for correct frames, the mirrored actor is within 5 yd of the event location for a mirrored frame). A mirrored frame is repaired by mirroring all coordinates (`mirrored`) or by mirroring and swapping the teammate flags (`mirrored_swapped`); which one is decided by an *anchor* (a frame of the other team with identical coordinates whose own actor sits on its event location: same flags => swap, inverted flags => pure mirror), falling back to the keeper ends after the mirror. `far` frames (actor away from both the event location and its mirror: a frame from another instant, almost all incomplete-pass receipts whose frame is the opponent's next action) and `unresolved` mirrored frames get NaN geometry targets and `y_reliable = 0`. `y_keeper_consistent` checks, after the resolution, that every visible keeper is at the expected end of the pitch.

| y_frame_orientation | n | share | frame_ok | actor_dist_median | keeper_consistent | keeper_n |
|---|---|---|---|---|---|---|
| far | 19672 | 0.014 | 0.000 | 70.867 |  | 0 |
| mirrored | 37230 | 0.027 | 1.000 | 69.785 | 0.992 | 13415 |
| mirrored_swapped | 26103 | 0.019 | 1.000 | 75.646 | 0.994 | 11710 |
| no_actor | 2 | 0.000 | 0.000 |  |  | 0 |
| ok | 1274497 | 0.939 | 1.000 | 0.000 | 0.995 | 407128 |
| unresolved | 2 | 0.000 | 0.000 | 4.437 |  | 0 |

### By event type (18 most frequent)

| f_type | n | ok | mirrored | mirrored_swapped | far | unresolved | other | mis_oriented_share | repaired_share | usable_share |
|---|---|---|---|---|---|---|---|---|---|---|
| Pass | 373639 | 373474 | 0 | 7 | 158 | 0 | 0 | 0.000 | 0.000 | 1.000 |
| Ball Receipt* | 371155 | 328732 | 2303 | 21236 | 18884 | 0 | 0 | 0.114 | 0.063 | 0.949 |
| Carry | 322954 | 321864 | 707 | 219 | 164 | 0 | 0 | 0.003 | 0.003 | 0.999 |
| Pressure | 118572 | 118568 | 0 | 2 | 2 | 0 | 0 | 0.000 | 0.000 | 1.000 |
| Ball Recovery | 36464 | 36455 | 0 | 5 | 4 | 0 | 0 | 0.000 | 0.000 | 1.000 |
| Duel | 23688 | 12956 | 6364 | 4280 | 88 | 0 | 0 | 0.453 | 0.449 | 0.996 |
| Clearance | 15064 | 15053 | 0 | 8 | 3 | 0 | 0 | 0.001 | 0.001 | 1.000 |
| Block | 14765 | 14749 | 0 | 4 | 12 | 0 | 0 | 0.001 | 0.000 | 0.999 |
| Dribble | 11198 | 6758 | 4372 | 43 | 25 | 0 | 0 | 0.396 | 0.394 | 0.998 |
| Shot | 10367 | 10357 | 0 | 0 | 10 | 0 | 0 | 0.001 | 0.000 | 0.999 |
| Miscontrol | 9925 | 9904 | 0 | 1 | 20 | 0 | 0 | 0.002 | 0.000 | 0.998 |
| Interception | 8975 | 8961 | 0 | 9 | 5 | 0 | 0 | 0.002 | 0.001 | 0.999 |
| Foul Committed | 8947 | 8936 | 0 | 6 | 5 | 0 | 0 | 0.001 | 0.001 | 0.999 |
| Foul Won | 8482 | 52 | 8070 | 128 | 230 | 2 | 0 | 0.994 | 0.967 | 0.973 |
| Dispossessed | 8360 | 94 | 8176 | 46 | 43 | 0 | 1 | 0.989 | 0.983 | 0.995 |
| Dribbled Past | 6600 | 9 | 6535 | 39 | 17 | 0 | 0 | 0.999 | 0.996 | 0.997 |
| Goal Keeper | 6028 | 6025 | 0 | 1 | 2 | 0 | 0 | 0.000 | 0.000 | 1.000 |
| 50/50 | 1506 | 781 | 703 | 21 | 0 | 0 | 1 | 0.481 | 0.481 | 0.999 |

### Repaired frames by decision method

| y_frame_orientation | y_frame_method | n | keeper_consistent | keeper_n |
|---|---|---|---|---|
| mirrored | anchor | 37230 | 0.992 | 13415 |
| mirrored_swapped | anchor | 26103 | 0.994 | 11710 |

### Effect on the affected event types

`y_nearest_opp_dist` on reliable frames in the previous build (no orientation check; from the pre-fix `soccer_01_targets_by_type.parquet`) versus now. `mis_oriented_share` = share of the type's rows whose raw frame was not `ok`. Dispossessed / Dribbled Past were not in the previous per-type table; their raw all-frame medians were ~50 yd (the actor's mirror image).

| f_type | n_reliable_now | n_rows | mis_oriented_share | pre_fix_nearest_opp_mean | pre_fix_nearest_opp_sd | pre_fix_nearest_opp_median | now_nearest_opp_mean | now_nearest_opp_sd | now_nearest_opp_median | now_n_opp_within_5_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Ball Receipt* | 270821 | 371155 | 0.114 | 9.990 | 12.810 | 6.730 | 6.694 | 4.581 | 5.727 | 0.606 |
| Duel | 16400 | 23688 | 0.453 | 23.810 | 30.580 | 3.940 | 0.672 | 0.747 | 0.141 | 1.526 |
| Dribble | 9513 | 11198 | 0.396 | 22.730 | 31.230 | 2.060 | 0.924 | 0.880 | 0.770 | 1.634 |
| Foul Won | 6402 | 8482 | 0.994 | 38.310 | 24.650 | 34.860 | 0.162 | 0.205 | 0.141 | 1.551 |
| Dispossessed | 7100 | 8360 | 0.989 |  |  |  | 0.157 | 0.168 | 0.141 | 1.770 |
| Dribbled Past | 4120 | 6600 | 0.999 |  |  |  | 0.147 | 0.108 | 0.141 | 1.224 |
| 50/50 | 1187 | 1506 | 0.481 |  |  |  | 0.738 | 0.776 | 0.418 | 1.585 |
| Carry | 240787 | 322954 | 0.003 | 7.220 | 5.080 | 6.370 | 7.097 | 4.600 | 6.336 | 0.497 |

## Target distributions by event type (reliable frames only)

Reliable = `y_frame_ok == 1` (orientation ok or repaired) and >= 7 opponents visible. Cells are mean (sd). Full long-format table with p10/p50/p90 for all 16 key targets in `soccer_01_targets_by_type.parquet`.

| f_type | n_reliable | y_block_depth | y_def_line | y_block_width | y_n_opp_ahead_of_ball | y_n_opp_within_5 | y_nearest_opp_dist | y_n_opp_in_cone |
|---|---|---|---|---|---|---|---|---|
| Pass | 282132 | 45.1 (21.0) | 34.6 (18.7) | 38.2 (8.4) | 7.0 (2.4) | 0.8 (0.8) | 5.1 (4.5) | 0.4 (0.6) |
| Ball Receipt* | 270821 | 44.6 (20.3) | 34.1 (17.9) | 38.4 (8.2) | 6.4 (2.7) | 0.6 (0.8) | 6.7 (4.6) | 0.3 (0.6) |
| Carry | 240787 | 45.6 (20.8) | 35.0 (18.5) | 39.0 (8.2) | 6.7 (2.5) | 0.5 (0.7) | 7.1 (4.6) | 0.3 (0.6) |
| Pressure | 84735 | 58.5 (25.6) | 45.2 (24.8) | 45.8 (12.9) | 4.8 (2.1) | 1.1 (0.6) | 2.8 (1.6) | 0.2 (0.5) |
| Ball Recovery | 24016 | 49.2 (29.2) | 38.1 (27.0) | 37.0 (11.7) | 5.8 (2.4) | 0.9 (0.9) | 4.7 (4.0) | 0.4 (0.8) |
| Duel | 16400 | 57.9 (28.1) | 45.7 (26.6) | 38.8 (12.2) | 4.5 (2.1) | 1.5 (1.0) | 0.7 (0.7) | 0.2 (0.4) |
| Clearance | 7096 | 93.2 (20.0) | 80.4 (21.1) | 36.5 (11.2) | 5.9 (1.8) | 1.4 (1.0) | 2.6 (2.0) | 0.2 (0.4) |
| Block | 9359 | 64.6 (32.1) | 52.4 (31.0) | 40.4 (12.6) | 4.9 (2.0) | 1.4 (0.8) | 2.1 (1.5) | 0.2 (0.5) |
| Dribble | 9513 | 41.9 (24.5) | 32.0 (21.7) | 36.7 (9.5) | 4.7 (2.2) | 1.6 (0.7) | 0.9 (0.9) | 0.2 (0.4) |
| Shot | 9281 | 15.4 (5.9) | 9.6 (5.5) | 24.6 (7.3) | 5.7 (2.9) | 1.9 (1.4) | 2.9 (2.3) | 0.7 (1.0) |
| Miscontrol | 8300 | 43.6 (23.8) | 33.2 (21.1) | 36.6 (9.1) | 4.3 (2.1) | 1.4 (0.9) | 2.5 (2.3) | 0.2 (0.4) |
| Interception | 5475 | 65.4 (24.3) | 51.7 (24.5) | 47.3 (11.8) | 5.3 (1.8) | 1.0 (0.6) | 3.1 (2.4) | 0.2 (0.4) |
| Foul Committed | 6751 | 49.4 (24.1) | 37.0 (22.3) | 42.2 (13.0) | 4.9 (2.2) | 1.4 (0.8) | 1.3 (1.0) | 0.2 (0.5) |
| Foul Won | 6402 | 58.4 (23.9) | 46.4 (22.2) | 38.6 (9.4) | 4.5 (2.0) | 1.6 (0.7) | 0.2 (0.2) | 0.2 (0.4) |

### All frames vs reliable frames

| f_type | target | n | mean | std | p10 | p50 | p90 |
|---|---|---|---|---|---|---|---|
| <all reliable> | y_block_depth | 995227 | 47.123 | 22.949 | 19.192 | 44.395 | 80.527 |
| <all frames> | y_block_depth | 1336457 | 53.950 | 25.670 | 21.168 | 51.621 | 91.218 |
| <all reliable> | y_def_line | 995227 | 36.279 | 20.732 | 12.540 | 32.746 | 66.123 |
| <all frames> | y_def_line | 1336457 | 43.602 | 24.671 | 14.257 | 39.685 | 80.016 |
| <all reliable> | y_block_length | 995227 | 24.531 | 7.170 | 15.277 | 24.430 | 33.637 |
| <all frames> | y_block_length | 1331159 | 22.946 | 7.663 | 13.021 | 22.971 | 32.636 |
| <all reliable> | y_block_width | 995227 | 38.994 | 9.489 | 27.199 | 38.900 | 50.387 |
| <all frames> | y_block_width | 1331159 | 37.230 | 10.290 | 24.133 | 37.491 | 49.419 |
| <all reliable> | y_block_area | 995227 | 614.817 | 288.845 | 273.916 | 583.197 | 983.671 |
| <all frames> | y_block_area | 1315925 | 542.539 | 296.164 | 192.485 | 508.909 | 924.012 |
| <all reliable> | y_n_opp_ahead_of_ball | 995227 | 6.367 | 2.574 | 3.000 | 7.000 | 10.000 |
| <all frames> | y_n_opp_ahead_of_ball | 1337830 | 5.691 | 2.624 | 2.000 | 6.000 | 9.000 |
| <all reliable> | y_n_opp_within_5 | 995227 | 0.785 | 0.843 | 0.000 | 1.000 | 2.000 |
| <all frames> | y_n_opp_within_5 | 1337830 | 0.720 | 0.805 | 0.000 | 1.000 | 2.000 |
| <all reliable> | y_n_opp_within_10 | 995227 | 1.908 | 1.444 | 0.000 | 2.000 | 4.000 |
| <all frames> | y_n_opp_within_10 | 1337830 | 1.715 | 1.391 | 0.000 | 1.000 | 4.000 |
| <all reliable> | y_nearest_opp_dist | 995227 | 5.471 | 4.533 | 1.026 | 4.184 | 11.751 |
| <all frames> | y_nearest_opp_dist | 1336461 | 6.016 | 5.148 | 1.065 | 4.472 | 13.157 |
| <all reliable> | y_n_opp_in_box | 995227 | 0.804 | 1.928 | 0.000 | 0.000 | 4.000 |
| <all frames> | y_n_opp_in_box | 1337830 | 0.618 | 1.709 | 0.000 | 0.000 | 3.000 |
| <all reliable> | y_n_opp_in_cone | 995227 | 0.329 | 0.573 | 0.000 | 0.000 | 1.000 |
| <all frames> | y_n_opp_in_cone | 1337830 | 0.277 | 0.532 | 0.000 | 0.000 | 1.000 |
| <all reliable> | y_nearest_opp_dist_in_cone | 280891 | 16.574 | 9.862 | 3.960 | 15.815 | 29.733 |
| <all frames> | y_nearest_opp_dist_in_cone | 322357 | 16.587 | 9.791 | 3.999 | 15.893 | 29.600 |
| <all reliable> | y_n_opp_within_3_of_end | 282132 | 0.101 | 0.354 | 0.000 | 0.000 | 0.000 |
| <all frames> | y_n_opp_within_3_of_end | 373481 | 0.086 | 0.324 | 0.000 | 0.000 | 0.000 |
| <all reliable> | y_n_opp_in_lane | 282132 | 0.545 | 0.775 | 0.000 | 0.000 | 2.000 |
| <all frames> | y_n_opp_in_lane | 373481 | 0.494 | 0.733 | 0.000 | 0.000 | 1.000 |
| <all reliable> | y_nearest_opp_to_receiver | 282132 | 8.180 | 5.168 | 2.229 | 7.187 | 15.542 |
| <all frames> | y_nearest_opp_to_receiver | 372911 | 8.914 | 5.658 | 2.435 | 7.846 | 16.820 |
| <all reliable> | y_opp_keeper_dist_to_goal_line | 242249 | 4.250 | 6.202 | 0.848 | 3.531 | 7.763 |
| <all frames> | y_opp_keeper_dist_to_goal_line | 255259 | 4.414 | 7.089 | 0.840 | 3.549 | 7.935 |

## Shots: 360-derived vs shot.freeze_frame-derived cone counts

Both are the number of outfield opponents inside the triangle ball -> posts. The shot freeze frame lists every player StatsBomb coded from video (not limited to the broadcast visible area), so it is the closer-to-truth reference; disagreement mostly reflects 360 visibility. Shots with an unusable 360 frame (10 of 10,245 shots with both frames) are excluded.

| subset | n | exact_agreement | within_1 | mae | mean_360 | mean_sff | pearson_r | keeper_in_cone_agreement | mean_opp_visible_360 | mean_opp_listed_sff | nearest_in_cone_n | nearest_in_cone_mae | nearest_in_cone_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all shots with both frames | 10235 | 0.643 | 0.952 | 0.414 | 0.674 | 0.842 | 0.748 | 0.843 | 8.892 | 9.004 | 3806 | 1.541 | 0.756 |
| reliable 360 frames (>=7 opp visible) | 9192 | 0.621 | 0.947 | 0.442 | 0.731 | 0.911 | 0.744 | 0.843 | 9.297 | 9.405 | 3682 | 1.564 | 0.751 |
| unreliable 360 frames | 1043 | 0.838 | 0.997 | 0.166 | 0.180 | 0.233 | 0.537 | 0.841 | 5.321 | 5.471 | 124 | 0.854 | 0.919 |
| open-play shots | 9889 | 0.656 | 0.960 | 0.391 | 0.610 | 0.775 | 0.726 | 0.842 | 8.835 | 8.952 | 3513 | 1.597 | 0.741 |
| male competitions | 7000 | 0.656 | 0.960 | 0.391 | 0.629 | 0.775 | 0.741 | 0.858 | 8.891 | 8.990 | 2460 | 1.498 | 0.751 |
| female competitions | 3235 | 0.615 | 0.935 | 0.464 | 0.773 | 0.988 | 0.756 | 0.808 | 8.894 | 9.035 | 1346 | 1.619 | 0.764 |

Cross-tab of cone counts (360 rows, shot-frame columns, counts capped at 4+):

| count | sff=0 | sff=1 | sff=2 | sff=3 | sff=4+ |
|---|---|---|---|---|---|
| 360=0 | 4060 | 1435 | 182 | 31 | 10 |
| 360=1 | 666 | 1843 | 453 | 79 | 19 |
| 360=2 | 44 | 211 | 449 | 153 | 40 |
| 360=3 | 1 | 22 | 72 | 148 | 78 |
| 360=4+ | 0 | 3 | 19 | 61 | 156 |

## Column definitions

260 columns in both files. Groups: id, event_feature (`f_`), sequence (`seq_`), target_360 (`y_`), shot_freeze_frame (`sff_`), oracle (`oracle_`), post_event (`post_`). `instant` = pre (known at the instant of the event) or post (`f_after_*`: realised trajectory / end point / duration, i.e. after the instant the 360 frame shows); 16 columns are post-instant.

| column | dtype | group | instant | definition |
|---|---|---|---|---|
| match_id | int64 | id |  | StatsBomb match id (group key for all splits) |
| event_id | string | id |  | StatsBomb event uuid |
| event_index | int32 | id |  | event order within the match |
| period | int32 | id |  | 1-2 regulation, 3-4 extra time (period 5 shoot-out dropped) |
| team_id | int64 | id |  | team performing the event |
| opp_team_id | int64 | id |  | the other team |
| possession_team_id | int64 | id |  | team in possession per StatsBomb |
| player_id | int64 | id |  | actor |
| competition | string | id |  | competition name |
| competition_id | int32 | id |  | StatsBomb competition id |
| season | string | id |  | season name |
| season_id | int32 | id |  | StatsBomb season id |
| gender | string | id |  | competition gender (male/female) |
| match_date | string | id |  | kick-off date (YYYY-MM-DD) |
| match_week | int32 | id |  | StatsBomb match week |
| home_team_id | int64 | id |  | home team |
| away_team_id | int64 | id |  | away team |
| has_360 | bool | id |  | match has usable 360 frames (True for every events360 row; False in events_no360, including 360-flagged matches whose frames were unusable) |
| f_type | string | event_feature | pre | event type name |
| f_type_id | int32 | event_feature | pre | compact integer type id (see type vocab) |
| f_x | float | event_feature | pre | event x (event team's attacking frame, goal at 120) |
| f_y | float | event_feature | pre | event y (0-80) |
| f_dist_goal | float | event_feature | pre | distance from event location to goal centre (120, 40) |
| f_goal_opening | float | event_feature | pre | angle subtended by the posts from the event location (rad) |
| f_goal_bearing | float | event_feature | pre | atan2(y-40, 120-x): 0 = straight at goal (rad) |
| f_has_location | bool | event_feature | pre | event carries a location |
| f_play_pattern | string | event_feature | pre | StatsBomb play pattern of the possession |
| f_position | string | event_feature | pre | actor's position name |
| f_under_pressure | bool | event_feature | pre | StatsBomb under_pressure flag |
| f_counterpress | bool | event_feature | pre | StatsBomb counterpress flag |
| f_after_duration | float | event_feature | post | POST-INSTANT: event duration (s): for a pass/carry the time until the ball arrives / the carry ends |
| f_minute | float | event_feature | pre | match minute incl. stoppage (minute + second/60) |
| f_t_period | float | event_feature | pre | seconds since the start of the period |
| f_ts_repaired | bool | event_feature | pre | timestamp was > 60 s before the previous located event (StatsBomb glitch) and was replaced by that event's time |
| f_period | int32 | event_feature | pre | same as period |
| f_home | bool | event_feature | pre | event team is the home team |
| f_is_possession_team | bool | event_feature | pre | event team == possession team |
| f_score_for | int32 | event_feature | pre | event team's goals before this event (periods 1-4) |
| f_score_against | int32 | event_feature | pre | opponent goals before this event |
| f_score_diff | int32 | event_feature | pre | for - against |
| f_goals_total | int32 | event_feature | pre | for + against |
| f_dt_prev | float | event_feature | pre | seconds since the previous located event (same period); NaN if none |
| f_poss_idx | int32 | event_feature | pre | StatsBomb possession index |
| f_poss_elapsed | float | event_feature | pre | seconds since the first event of the possession |
| f_poss_n_events | int32 | event_feature | pre | located events earlier in this possession (both teams) |
| f_poss_n_passes | int32 | event_feature | pre | passes by the possession team earlier in this possession |
| f_poss_ball_dist | float | event_feature | pre | sum of pass lengths + carry lengths so far in the possession (yd) |
| f_poss_start_type | string | event_feature | pre | how the possession started (kick_off/set_piece/recovery/interception/duel/keeper/open_play/other) |
| f_poss_t_since_ft_entry | float | event_feature | pre | seconds since the ball last crossed into the final third (possession team's frame); NaN if it has not |
| f_poss_in_final_third | bool | event_feature | pre | ball currently in the possession team's final third |
| f_poss_start_x | float | event_feature | pre | possession start x (event team's frame) |
| f_poss_start_y | float | event_feature | pre | possession start y (event team's frame) |
| f_pass_type | string | event_feature | pre | Open Play or set-piece type (known before the pass) |
| f_pass_body_part | string | event_feature | pre | body part used (at the instant of the pass) |
| f_after_pass_length | float | event_feature | post | POST-INSTANT: realised pass length (yd) |
| f_after_pass_angle | float | event_feature | post | POST-INSTANT: realised pass angle (rad, StatsBomb) |
| f_after_pass_height | string | event_feature | post | POST-INSTANT: Ground/Low/High Pass (realised trajectory) |
| f_after_pass_switch | bool | event_feature | post | POST-INSTANT: switch of play |
| f_after_pass_cross | bool | event_feature | post | POST-INSTANT: cross |
| f_after_pass_through_ball | bool | event_feature | post | POST-INSTANT: through ball |
| f_after_pass_cut_back | bool | event_feature | post | POST-INSTANT: cut back |
| f_after_pass_end_x | float | event_feature | post | POST-INSTANT: pass end x (an incomplete pass ends where the opponent won it) |
| f_after_pass_end_y | float | event_feature | post | POST-INSTANT: pass end y |
| f_after_pass_end_dist_goal | float | event_feature | post | POST-INSTANT: distance from pass end to goal centre |
| f_after_pass_progress | float | event_feature | post | POST-INSTANT: end x - start x |
| post_pass_outcome | string | post_event |  | POST-EVENT: Complete / Incomplete / Out / ... (not a state feature) |
| f_after_carry_length | float | event_feature | post | POST-INSTANT: carry length (yd) - a consequence of the nearest defender's distance |
| f_after_carry_end_x | float | event_feature | post | POST-INSTANT: carry end x |
| f_after_carry_end_y | float | event_feature | post | POST-INSTANT: carry end y |
| f_after_carry_progress | float | event_feature | post | POST-INSTANT: carry end x - start x |
| f_shot_body_part | string | event_feature | pre | shot body part |
| f_shot_technique | string | event_feature | pre | shot technique |
| f_shot_type | string | event_feature | pre | Open Play / Free Kick / Penalty / Corner |
| f_shot_first_time | bool | event_feature | pre | first-time shot |
| post_shot_outcome | string | post_event |  | POST-EVENT shot outcome (Goal, Saved, ...) |
| oracle_statsbomb_xg | float | oracle |  | ORACLE: StatsBomb xG (freeze-frame based) - never a feature |
| oracle_one_on_one | float | oracle |  | ORACLE: shot.one_on_one flag |
| oracle_open_goal | float | oracle |  | ORACLE: shot.open_goal |
| f_w10_n_pass | int32 | event_feature | pre | count of 'pass' events among the previous 10 (both teams) |
| f_w10_n_carry | int32 | event_feature | pre | count of 'carry' events among the previous 10 (both teams) |
| f_w10_n_receipt | int32 | event_feature | pre | count of 'receipt' events among the previous 10 (both teams) |
| f_w10_n_pressure | int32 | event_feature | pre | count of 'pressure' events among the previous 10 (both teams) |
| f_w10_n_duel | int32 | event_feature | pre | count of 'duel' events among the previous 10 (both teams) |
| f_w10_n_dribble | int32 | event_feature | pre | count of 'dribble' events among the previous 10 (both teams) |
| f_w10_n_shot | int32 | event_feature | pre | count of 'shot' events among the previous 10 (both teams) |
| f_w10_n_interception | int32 | event_feature | pre | count of 'interception' events among the previous 10 (both teams) |
| f_w10_n_block | int32 | event_feature | pre | count of 'block' events among the previous 10 (both teams) |
| f_w10_n_clearance | int32 | event_feature | pre | count of 'clearance' events among the previous 10 (both teams) |
| f_w10_n_recovery | int32 | event_feature | pre | count of 'recovery' events among the previous 10 (both teams) |
| f_w10_n_foul | int32 | event_feature | pre | count of 'foul' events among the previous 10 (both teams) |
| f_w10_n_keeper | int32 | event_feature | pre | count of 'keeper' events among the previous 10 (both teams) |
| f_w10_n_loss | int32 | event_feature | pre | count of 'loss' events among the previous 10 (both teams) |
| f_w10_n_other | int32 | event_feature | pre | count of 'other' events among the previous 10 (both teams) |
| f_w10_n_own | int32 | event_feature | pre | of the previous 10 located events, how many by the event team |
| f_w10_n_opp_def | int32 | event_feature | pre | of the previous 10, opponent defensive actions (Pressure/Duel/Interception/Block/Clearance) |
| f_w10_n | int32 | event_feature | pre | number of previous located events available in the window (same period) |
| f_opp_def_x_60s_mean | float | event_feature | pre | mean x (event team's frame, 120 - x_opp) of opponent defensive actions in the last 60 s; NaN if none |
| f_opp_def_n_60s | int32 | event_feature | pre | count of opponent defensive actions in the last 60 s |
| f_opp_poss_last10s | bool | event_feature | pre | any event in the last 10 s with the opponent as possession team |
| f_ball_speed_3 | float | event_feature | pre | path length over the last 3 event transitions / elapsed time (yd/s) |
| f_ball_dx_3 | float | event_feature | pre | x displacement of the ball over the last 3 previous events |
| f_ball_dy_3 | float | event_feature | pre | y displacement over the last 3 previous events |
| f_t_since_opp_def_action | float | event_feature | pre | seconds since the opponent's last defensive action (same period); NaN if none |
| y_frame_orientation | string | target_360 |  | 360: ok / mirrored / mirrored_swapped (repaired) / far / unresolved / no_actor / no_location / empty (unusable); see frame_features.resolve_frame_orientation |
| y_frame_ok | float | target_360 |  | 360: geometry targets were computed (orientation ok or repaired) |
| y_frame_repaired | float | target_360 |  | 360: frame was mirrored (and flag-swapped for mirrored_swapped) |
| y_frame_method | string | target_360 |  | 360: how a mirrored frame's flag perspective was decided: anchor (identical-coordinate ok frame of the other team) or keeper |
| y_actor_dist_to_event | float | target_360 |  | 360: distance (yd) between the raw flagged actor and the event location (0 for ok frames) |
| y_keeper_consistent | float | target_360 |  | 360: after resolution every visible keeper is at the expected end (opp x>60, own x<60); NaN if no keeper visible |
| y_n_teammates_visible | float | target_360 |  | 360: visible teammates excluding the actor |
| y_n_opponents_visible | float | target_360 |  | 360: visible opponents incl. keeper |
| y_opp_keeper_visible | float | target_360 |  | 360: opponent keeper visible |
| y_tm_keeper_visible | float | target_360 |  | 360: own keeper visible |
| y_actor_visible | float | target_360 |  | 360: actor present in frame |
| y_reliable | float | target_360 |  | 360: frame usable (y_frame_ok) and n_opponents_visible >= 7 |
| y_visible_area_frac | float | target_360 |  | 360: visible_area polygon (clipped to pitch) / 9600 |
| y_n_opp_outfield | float | target_360 |  | 360: visible outfield opponents |
| y_block_depth | float | target_360 |  | 360: 120 - mean x of outfield opponents (distance from their goal) |
| y_def_line | float | target_360 |  | 360: 120 - max x of outfield opponents (deepest defender) |
| y_block_length | float | target_360 |  | 360: x range of outfield opponents |
| y_block_width | float | target_360 |  | 360: y range |
| y_block_area | float | target_360 |  | 360: convex-hull area of outfield opponents (NaN < 3 players) |
| y_block_centroid_y | float | target_360 |  | 360: mean y of outfield opponents |
| y_block_std_x | float | target_360 |  | 360: std of x |
| y_block_std_y | float | target_360 |  | 360: std of y |
| y_n_opp_in_box | float | target_360 |  | 360: outfield opponents in the penalty box (x>=102, 18<=y<=62) |
| y_n_tm_in_box | float | target_360 |  | 360: teammates in the box |
| y_opp_keeper_dist_to_goal_line | float | target_360 |  | 360: 120 - keeper x (NaN if not visible) |
| y_opp_keeper_y | float | target_360 |  | 360: keeper y |
| y_n_opp_ahead_of_ball | float | target_360 |  | 360: opponents (incl. keeper) with x > ball x |
| y_n_opp_within_5 | float | target_360 |  | 360: opponents within 5 yd of the ball |
| y_n_opp_within_10 | float | target_360 |  | 360: opponents within 10 yd |
| y_nearest_opp_dist | float | target_360 |  | 360: nearest opponent (yd) |
| y_n_tm_ahead_of_ball | float | target_360 |  | 360: teammates with x > ball x |
| y_n_tm_within_10 | float | target_360 |  | 360: teammates within 10 yd |
| y_nearest_tm_dist | float | target_360 |  | 360: nearest teammate |
| y_cone_area | float | target_360 |  | area of the shot cone |
| y_n_opp_in_cone | float | target_360 |  | 360: outfield opponents in triangle ball-(120,36)-(120,44) |
| y_nearest_opp_dist_in_cone | float | target_360 |  | 360: nearest outfield opponent inside the cone (NaN if none) |
| y_opp_keeper_in_cone | float | target_360 |  | 360: keeper inside the cone |
| y_n_opp_within_3_of_end | float | target_360 |  | 360 (Pass only): opponents within 3 yd of the pass end |
| y_n_opp_in_lane | float | target_360 |  | 360 (Pass only): opponents within 2 yd of the start->end segment |
| y_nearest_opp_to_end | float | target_360 |  | 360 (Pass only): nearest opponent to the pass end |
| y_nearest_opp_to_receiver | float | target_360 |  | 360 (Pass only): nearest opponent to the receiver (teammate nearest the end location) |
| y_receiver_dist_to_end | float | target_360 |  | 360 (Pass only): receiver distance to the pass end |
| sff_present | float | shot_freeze_frame |  | shot.freeze_frame available (Shot rows; penalties usually lack it) |
| sff_n_opp | float | shot_freeze_frame |  | shot frame: opponents listed |
| sff_n_tm | float | shot_freeze_frame |  | shot frame: teammates listed |
| sff_opp_keeper_visible | float | shot_freeze_frame |  | shot frame: keeper listed |
| sff_n_opp_in_cone | float | shot_freeze_frame |  | shot frame: outfield opponents in the cone |
| sff_nearest_opp_dist_in_cone | float | shot_freeze_frame |  | shot frame: nearest outfield opponent in the cone |
| sff_opp_keeper_in_cone | float | shot_freeze_frame |  | shot frame: keeper in the cone |
| sff_cone_area | float | shot_freeze_frame |  | shot cone area |
| sff_n_opp_within_5 | float | shot_freeze_frame |  | shot frame: opponents within 5 yd |
| sff_n_opp_within_10 | float | shot_freeze_frame |  | shot frame: opponents within 10 yd |
| sff_nearest_opp_dist | float | shot_freeze_frame |  | shot frame: nearest opponent |
| sff_n_opp_in_box | float | shot_freeze_frame |  | shot frame: outfield opponents in the box |
| sff_n_opp_ahead_of_ball | float | shot_freeze_frame |  | shot frame: opponents with x > ball x |
| sff_opp_keeper_dist_to_goal_line | float | shot_freeze_frame |  | shot frame: 120 - keeper x |
| sff_opp_keeper_y | float | shot_freeze_frame |  | shot frame: keeper y |

Sequence block: `seq_<field>_<k>` for k = 01..20 (k = 1 most recent located event of the same period), fields type (int16 id, 0 pad), x, y (event team's frame, NaN pad), dt (seconds before the event, NaN pad), same (1 own team, 0 opponent, -1 pad).

### Type vocabulary (`f_type_id`, `seq_type_*`)

| type_name | type_id | window_group |
|---|---|---|
| <pad> | 0 | pad |
| Pass | 1 | pass |
| Ball Receipt* | 2 | receipt |
| Carry | 3 | carry |
| Pressure | 4 | pressure |
| Ball Recovery | 5 | recovery |
| Duel | 6 | duel |
| Block | 7 | block |
| Clearance | 8 | clearance |
| Goal Keeper | 9 | keeper |
| Dribble | 10 | dribble |
| Shot | 11 | shot |
| Interception | 12 | interception |
| Foul Committed | 13 | foul |
| Miscontrol | 14 | loss |
| Dispossessed | 15 | loss |
| Foul Won | 16 | foul |
| Dribbled Past | 17 | dribble |
| 50/50 | 18 | other |
| Shield | 19 | other |
| Error | 20 | other |
| Own Goal Against | 21 | other |
| Own Goal For | 22 | other |
| Offside | 23 | other |
| Injury Stoppage | 24 | other |
| Referee Ball-Drop | 25 | other |
| Bad Behaviour | 26 | other |
| Player Off | 27 | other |
| Player On | 28 | other |
| Substitution | 29 | other |
| Tactical Shift | 30 | other |
| Half Start | 31 | other |
| Half End | 32 | other |
| Starting XI | 33 | other |
| Camera On | 34 | other |
| Camera off | 35 | other |
| <unknown> | 36 | other |

## Caveats

- 360 frames only contain players inside the broadcast `visible_area`; block-shape targets on unreliable frames are biased towards the players near the ball. Downstream stages should train on `y_reliable == 1` (or weight by visibility) and report both.
- `y_reliable` now requires `y_frame_ok == 1`; rows with `y_frame_ok == 0` have NaN geometry targets by construction. Repaired frames (`y_frame_repaired == 1`) are validated per row (anchor flags or keeper ends); downstream stages can exclude them with `y_frame_orientation == 'ok'` as a robustness check. After a flag swap the actor is re-identified as the teammate nearest the event location (within 5 yd), so `y_actor_visible` can be 0 on `mirrored_swapped` rows. `y_visible_area_frac` is computed from the raw polygon (its area is mirror-invariant).
- `f_after_*` (pass end/length/angle/height/switch/cross/through-ball/cut-back/progress, carry end/length/progress, duration) are post-instant: a carry's length depends on the nearest defender and an incomplete pass ends where the opponent won the ball. They are legitimate event-only inputs for imputation on non-360 matches, but any imputation / payoff result must be reported with and without them (`instant` column of `soccer_01_columns.parquet`).
- Lint scope: `ruff check research/privileged_tracking/soccer research/privileged_tracking/tests/test_soccer_*.py` is clean; the NFL test files in the same tests directory belong to the NFL item and are not linted here.
- `y_block_*` describe the *opponents of the event team*; on defending-team events (Pressure, Duel, ...) that is the attacking team's shape. Filter with `f_is_possession_team` when a defensive-block target is wanted.
- Pass geometry targets exist only for Pass events with an end location; they are NaN elsewhere and cannot be built from `shot.freeze_frame`.
- `f_poss_ball_dist` uses pass lengths and carry lengths only (not receipts/dribbles).
- `f_opp_def_x_60s_mean` and `f_t_since_opp_def_action` look back only within the current period; extra-time periods restart the windows.
- Tendency features (team/player rolling means) are deliberately absent here; they must be built from strictly earlier matches in the modelling stage.
- 33 events360 rows and 0 events_no360 rows carry `f_ts_repaired = True` (receipt stamped 00:00:00 late in a period; time replaced by the previous event's).

### 9 360-flagged matches without usable frames

`frames_missing` = corrupt/absent 360 file (3845506 is corrupt upstream too: a run of null bytes); `frames_unmatched` = no frame `event_uuid` matches any event (360 and event files out of sync); `frames_too_few` = fewer than 500 frames joined to events. These matches fall back to the event-only output (`events_no360`, `has_360 = False`).

| match_id | competition | season | n_frames | error | n_rows |
|---|---|---|---|---|---|
| 3837706 | Ligue 1 | 2022/2023 | 3548 | frames_unmatched | 532 |
| 3845506 | UEFA Women's Euro | 2022 | 0 | frames_missing | 407 |
| 3877115 | Major League Soccer | 2023 | 3808 | frames_unmatched | 556 |
| 3877170 | Major League Soccer | 2023 | 3097 | frames_unmatched | 496 |
| 3877194 | Major League Soccer | 2023 | 3174 | frames_unmatched | 461 |
| 3895158 | 1. Bundesliga | 2023/2024 | 3318 | frames_unmatched | 488 |
| 3895266 | 1. Bundesliga | 2023/2024 | 3615 | frames_unmatched | 600 |
| 3895309 | 1. Bundesliga | 2023/2024 | 3280 | frames_unmatched | 490 |
| 3923880 | African Cup of Nations | 2023 | 1 | frames_too_few | 446 |
