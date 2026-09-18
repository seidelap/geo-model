# NFL 05 - audit-fix pass over the NFL chain (01 build -> 02 participation -> 03 imputation -> 04 payoff)

Written by the phase-2 `audit-fix` stage for `nfl` (2026-09-18) after the cross-cutting review returned `fix_needed` with two majors and five minors. Every affected script was edited and re-run from a clean shell (`python -m research.privileged_tracking.nfl.ood_highlights --max-plays 150`, 0.3 min; `python -m research.privileged_tracking.nfl.payoff --stages a,b,c,d`, 3.5 min, then `--stages a` again after adding the pre-snap oracle to the refit sets), the parquet tables and markdown reports were regenerated, and the test suite was re-run. `nfl_05_audit_fix_changes.parquet` holds every number that changed (before = the pre-fix tables, after = the regenerated ones).

## What was wrong and what changed

| # | severity | issue | fix | where |
|---|---|---|---|---|
| 1 | major | OOD highlight check: one play (2018102200/3269) had its ball row 25.6 yd off the field at both `ball_snap` frames, so the recomputed box count was 0 against an F0P prediction of 7.9; the reported box_count_tuned R2 F0 0.21 / F0P 0.31 / F1 0.31 and the sentence 'box recovery degrades out of distribution' rested on it. Also: the snap frame was 'first in file order' among two tagged frames on 7 plays, 16 postseason / Pro Bowl plays were silently lost in the REG-only join, and one play listed twice in the index was counted twice. | `ball_row_valid` (in the field, within 2.5 yd of the LOS); `choose_snap_frame` = earliest plausible `ball_snap` frame; centre / OL-median lateral fallback (`lateral_reference`) when no frame is plausible; season-type accounting; index de-duplication; play-level bootstrap CIs and a data-driven reading (`reference_position`). Conclusion withdrawn. | `nfl/ood_highlights.py`, `reports/nfl_03_ood_highlights.md`, `nfl_03_ood_highlights{,_summary,_bias}.parquet`, `phase1_handoffs.md` (NFL 03 key result, caveat, resolution note) |
| 2 | major | NFL 04 bottom line framed the oracle gain as 'the part that matters' that students cannot recover, although `PBP+ORACLE` mixes pre-snap structure with post-release quasi-outcome state. | New sets `PBP+ORACLE_PRESNAP` (12 pre-snap targets) and `PBP+ORACLE_WITHIN` (within-play targets) in stage a, new pairs and verdict lines, pre-snap oracle added to the refit sets; summary (a), protocol, caveats and the bottom line rewritten from the tables (`_bottom_line_oracle`, `_straddle_text`). | `nfl/payoff.py`, `reports/nfl_04_payoff.md`, `nfl_04a_*.parquet` |
| 3 | minor | Stage-c receiver-week gains had no CI. | `proxy_regression_table` fits every model on the same receiver-weeks and adds `delta_r2_vs_baseline` with a paired bootstrap CI (per-week squared errors / test variance, 2000 resamples). | `nfl_04c_receiver_week_regression.parquet`, summary (c) |
| 4 | minor | Refit-noise floor borrowed from the PBP / PBP+IMP models for pairs never refit. | `attach_refit_spread` gives each pair the larger spread of its own two members and a `refit_measured` flag (`both` / `from` / `to` / `none`; `nflfastR cp` and `base_global` are zero-spread fixed references); `PBP+IMP_F0` added to the stage a / b refit sets; `_sig` names half-measured pairs. | `nfl_04{a,b,d}_deltas.parquet`, verdict lines |
| 5 | minor | Summary (a) quoted only the pass n, (d) omitted n_train, and 'who is on the field' over-generalised stage d. | Both n in (a), n / n_train in (d), sentence narrowed to the offense personnel grouping. | `reports/nfl_04_payoff.md` |
| 6 | minor | `qb_hit` is a charted post-play flag inside the 'at-release' set. | Kept (symmetric across every set) and stated in the protocol and caveats: 'at release' = 'up to and including QB contact'. | `reports/nfl_04_payoff.md`, `nfl/CLAUDE.md` |

## Before / after

### OOD highlight check (n=126 regular-season 2018-19 highlight plays after de-duplication; play-level bootstrap 95% CI)

| quantity | before | after | note |
|---|---|---|---|
| box_count_tuned R2 F0 on highlight plays | 0.211 | 0.341 | n 127 -> 126; after: 95% CI [0.221, 0.433], 2017 OOF 0.434 |
| box_count_tuned R2 F0P on highlight plays | 0.314 | 0.475 | n 127 -> 126; after: 95% CI [0.342, 0.573], 2017 OOF 0.509 |
| box_count_tuned R2 F1 on highlight plays | 0.313 | 0.473 | n 127 -> 126; after: 95% CI [0.338, 0.577], 2017 OOF 0.513 |
| n_deep_safeties R2 F0 on highlight plays | 0.383 | 0.388 | n 127 -> 126; after: 95% CI [0.021, 0.518], 2017 OOF 0.400 |
| n_deep_safeties R2 F0P on highlight plays | 0.376 | 0.383 | n 127 -> 126; after: 95% CI [-0.013, 0.524], 2017 OOF 0.403 |
| n_deep_safeties R2 F1 on highlight plays | 0.410 | 0.415 | n 127 -> 126; after: 95% CI [-0.013, 0.573], 2017 OOF 0.418 |
| mof_open BSS F0 on highlight plays | 0.004 | 0.025 | n 127 -> 126; after: 95% CI [-0.249, 0.211], 2017 OOF 0.255 |
| mof_open BSS F0P on highlight plays | -0.003 | 0.025 | n 127 -> 126; after: 95% CI [-0.255, 0.220], 2017 OOF 0.264 |
| mof_open BSS F1 on highlight plays | 0.015 | 0.037 | n 127 -> 126; after: 95% CI [-0.210, 0.216], 2017 OOF 0.265 |
| play 2018102200/3269 recomputed box_count_tuned (F0P prediction 7.92) | 0.000 | 8.000 | snap frame 126 -> 98, lateral reference centre |
| play 2019101306/3030 recomputed box_count_tuned (F0P prediction 7.16) | 7.000 | 9.000 | snap frame 128 -> 48, lateral reference ball |
| highlight plays scored | 127.000 | 126.000 | one (gameId, playId) listed twice in the index was counted twice |

Reading: box_count_tuned R2 F0 0.34 [0.22, 0.43] / F0P 0.48 [0.34, 0.57] / F1 0.47 [0.34, 0.58] vs 2017 OOF 0.43 / 0.51 / 0.51: F0P and F1 inside their interval, the F0 reference at the upper edge of its interval (0.434 vs 0.433), so 'no visible degradation at n~125' for the personnel-aware students and 'neither shown nor excluded' for F0; the earlier 'box recovery degrades out of distribution' is withdrawn. The synthesis must not cite the old 0.21 / 0.31 numbers.

### Stage a oracle decomposition (k-fold by game, 91 games; deltas are `from` minus `to`, positive = `to` better, game-clustered CI)

| quantity | before | after | note |
|---|---|---|---|
| pass_completion: PBP -> PBP+ORACLE_PRESNAP (nats) |  | 0.0027 | new row; game CI [+0.0000, +0.0054], refit spread 0.0022 (both), n 6225 |
| pass_completion: PBP -> PBP+ORACLE_WITHIN (nats) |  | 0.0578 | new row; game CI [+0.0473, +0.0674] |
| pass_completion: PBP -> PBP+ORACLE (nats) | 0.0565 | 0.0565 | unchanged (reproduced exactly) |
| pass_yards: PBP -> PBP+ORACLE_PRESNAP (yd^2) |  | 0.4451 | new row; game CI [-0.2177, +1.1297], refit spread 1.8272 (both), n 6225 |
| pass_yards: PBP -> PBP+ORACLE_WITHIN (yd^2) |  | 4.2200 | new row; game CI [+2.7582, +5.6841] |
| pass_yards: PBP -> PBP+ORACLE (yd^2) | 4.7216 | 4.7216 | unchanged (reproduced exactly) |
| pass_epa: PBP -> PBP+ORACLE_PRESNAP (EPA^2) |  | -0.0039 | new row; game CI [-0.0148, +0.0074], refit spread 0.0064 (both), n 6225 |
| pass_epa: PBP -> PBP+ORACLE_WITHIN (EPA^2) |  | 0.0954 | new row; game CI [+0.0684, +0.1232] |
| pass_epa: PBP -> PBP+ORACLE (EPA^2) | 0.0954 | 0.0954 | unchanged (reproduced exactly) |
| run_yards: PBP -> PBP+ORACLE_PRESNAP (yd^2) |  | -0.0227 | new row; game CI [-0.3302, +0.3193], refit spread 0.1247 (both), n 4671 |
| run_yards: PBP -> PBP+ORACLE_WITHIN (yd^2) |  | 15.6654 | new row; game CI [+13.1471, +18.0900] |
| run_yards: PBP -> PBP+ORACLE (yd^2) | 15.8459 | 15.8459 | unchanged (reproduced exactly) |
| run_success: PBP -> PBP+ORACLE_PRESNAP (nats) |  | 0.0010 | new row; game CI [-0.0013, +0.0033], refit spread 0.0030 (both), n 4671 |
| run_success: PBP -> PBP+ORACLE_WITHIN (nats) |  | 0.1857 | new row; game CI [+0.1680, +0.2033] |
| run_success: PBP -> PBP+ORACLE (nats) | 0.1864 | 0.1864 | unchanged (reproduced exactly) |

Completion (n=6225): pre-snap oracle +0.0027 nats [+0.0000, +0.0054] vs within-play oracle +0.0578 nats [+0.0473, +0.0674] and full oracle +0.0565 nats [+0.0457, +0.0662]. Run yards (n=4671): pre-snap -0.0227 yd^2 [-0.3302, +0.3193] vs within-play +15.6654 yd^2 [+13.1471, +18.0900] and full +15.8459 yd^2 [+13.2730, +18.3801]. Perfectly known pre-snap structure has no measurable play-level payoff once the at-release fields are known; the whole oracle gain is post-release within-play state that no at-release imputation could recover. The reviewer's replication (+0.0027 nats [+0.0000, +0.0054]; -0.02 yd^2 [-0.33, +0.32]) agrees.

### Stage c gains with CIs (2022 test receiver-weeks, R2 for NGS `avg_separation` over `naive + after-the-fact`)

| quantity | before | after | note |
|---|---|---|---|
| avg_separation R2 gain over the fair baseline: naive + after-the-fact + imputed F1T | 0.055 | 0.055 | paired bootstrap 95% CI [+0.038, +0.072] (2000 resamples, n_test 1273); R2 unchanged |
| avg_separation R2 gain over the fair baseline: naive + after-the-fact + imputed F0T | 0.061 | 0.061 | paired bootstrap 95% CI [+0.041, +0.079] (2000 resamples, n_test 1273); R2 unchanged |
| avg_separation R2 gain over the fair baseline: naive + after-the-fact + imputed F1 | 0.096 | 0.096 | paired bootstrap 95% CI [+0.072, +0.120] (2000 resamples, n_test 1273); R2 unchanged |

### Refit floor per pair (stage b examples; the verdict wording follows `refit_measured`)

| item | quantity | before | after | note |
|---|---|---|---|---|
| NFL 04b refit floor | pass_completion: nflfastR cp -> PBP refit_spread | 0.0031 | 0.0031 | after: refit_measured = both; delta -0.0021, game CI [-0.0036, -0.0006] |
| NFL 04b refit floor | run_success: PBP -> PBP+IMP_F0 refit_spread | 0.0024 | 0.0024 | after: refit_measured = both; delta -0.0021, game CI [-0.0037, -0.0005] |
| NFL 04b refit floor | pass_yards: PBP -> PBP+IMP_F0 refit_spread | 0.2952 | 0.5552 | after: refit_measured = both; delta -0.4026, game CI [-0.8530, -0.0018] |
| NFL 04b refit floor | pass_epa: PBP+NGS -> PBP+IMP+NGS refit_spread | 0.0075 |  | after: refit_measured = none; delta -0.0344, game CI [-0.0475, -0.0218] |
| NFL 04a refit floor | pass_completion: PBP+ORACLE_PRESNAP refit spread (3 seeds) |  | 0.0022 | new refit set |
| NFL 04a refit floor | pass_yards: PBP+ORACLE_PRESNAP refit spread (3 seeds) |  | 0.5711 | new refit set |
| NFL 04a refit floor | pass_epa: PBP+ORACLE_PRESNAP refit spread (3 seeds) |  | 0.0047 | new refit set |
| NFL 04a refit floor | run_yards: PBP+ORACLE_PRESNAP refit spread (3 seeds) |  | 0.0882 | new refit set |
| NFL 04a refit floor | run_success: PBP+ORACLE_PRESNAP refit spread (3 seeds) |  | 0.0018 | new refit set |

## Reproducibility checks

* The at-release student bundles `models/students_F1T.joblib` / `students_F0T.joblib` re-saved by the stage-a re-run are identical to the pre-fix bundles (model strings, rounds, team encodings), and all 22 F1T / F0T out-of-fold columns of `payoff_oof_tracked.parquet` are bit-identical; the cached 2018-2022 imputations therefore remain valid and stages b-d reproduce every metric to 0.0 (52 shared stage-a rows, 41 stage-b rows, 16 stage-d rows, 22 stage-c R2 values compared).
* The OOD numbers with the fix agree with the reviewer's exclude-the-play recomputation (F0 0.346 / F0P 0.476 / F1 0.474 at n=127) up to the corrected play now being kept with the centre-y reference and the duplicated index play being counted once (n=126).
* Tests: `python -m pytest research/privileged_tracking/tests -q` from a clean shell after the fix: 153 passed (148 before; new tests cover `ball_row_valid`, `choose_snap_frame`, `lateral_reference`, `skill_bootstrap_ci`, `skill_summary` / `reference_position`, the oracle split, `attach_refit_spread` / `_sig` with `refit_measured`, and the stage-c paired CIs).
* One verdict wording changed because the floor is now the pair's own: stage b `pass_yards` PBP -> PBP+IMP_F0 (-0.40 yd^2, clustered CI excluding zero) was 'worse beyond the 0.30 refit spread of PBP' and is now 'within refit noise' because PBP+IMP_F0's own three-seed spread is 0.56 yd^2; the summary count of out-of-domain imputed deltas beyond refit noise stays 0 of 10. The stage-b `nflfastR cp -> PBP` pair keeps its 'within refit noise' label, now on the measured grounds that `cp` is a fixed column and PBP's spread is the pair's.

## Caveats

* The OOD sample is 126 plays; intervals on an R2 of ~0.4 are about +-0.15 wide. The F0 box student's 2017 reference sits at the edge of its highlight interval; the check rules out gross failure, not a mild season shift.
* Three of the four lateral-fallback plays are shotgun snaps already travelling back at the tagged frame (2.6-3.5 yd behind the LOS); there the centre / OL reference differs from the ball's y by <= 0.4 yd, so the 2.5-yd tolerance is conservative rather than wrong.
* The pre-snap oracle delta on completion (+0.0027 nats) has a clustered CI whose lower bound touches zero and sits above the refit spread of the pair; it is a real but negligible effect (0.4% of the PBP log-loss), which is the point.
* `qb_hit` stays in the at-release set on every side of every comparison; a sensitivity row without it was not added.
* The qb_spike minor from the NFL 03 review (19 of 6,613 pass plays) is documented but unchanged.
