# Learning game state from rare tracking data: what transferred, what did not

Cross-sport synthesis of the privileged-tracking research (NFL, then soccer). This is the
document to read first; every number below is copied from a stage report table in this
directory and carries its sample size and, for every model-vs-model claim, a paired
bootstrap 95% interval. The machine-readable companion is `results_index.parquet` (one row
per headline metric: `sport, stage, experiment, feature_set, metric, value, ci_low, ci_high,
ci_type, n, source_report, source_table, row_key`; built by
`python -m research.privileged_tracking.synthesis.results_index`, 1,150 rows from 50 tables).
This revision (fix pass) adds three experiments the first review found missing: the NFL
participation -> alignment chain with imputed personnel (§2.4, `nfl_06_participation_chain.md`),
a forward receiving-props test (§2.5, `nfl_07_props_forward.md`) and the payoff of the soccer
sequence student's imputations (§3.6, `soccer_06_seq_payoff.md`).

Conventions used throughout: a **delta** is `loss(reference) - loss(candidate)`, so **positive =
the candidate is better**; log-loss deltas are in nats; "game-clustered" / "match-clustered"
intervals resample whole games or matches and are the ones to trust. The soccer stage reports
label significance by the per-shot / per-pass interval; every soccer delta quoted below prints
both as `[per-sample] / [clustered]`. Over the 120 soccer delta rows that carry both, the
interval bounds differ by at most 0.0022 nats (xPass) and 0.0014 (xG), and the zero-exclusion
verdict flips in four rows, each flagged where it is quoted (men -> women in-domain EVENT and
EVENT+IMP, women -> men EVENT+ORACLE360, tournaments EVENT+ORACLESHOT zero-shot). Skill is R2
for continuous and count targets and Brier skill score (BSS) for binaries. "Oracle" means the
true privileged value used directly as a feature. Numbers printed without an interval (skill
tables, transfer R2s, decoder accuracies, Spearman correlations, external 2018 R2s) are point
estimates; the paired-interval rule applies to every model-vs-model delta.

## Verdict in brief

1. **Event-only students recover a real part of pre-snap / pre-instant state** (NFL box count
   R2 0.43-0.51, soccer block depth R2 0.96 and nearest-opponent distance R2 0.67 out of fold),
   almost nothing of within-play state before the fact (NFL time-to-throw, pressure,
   separation < 0.05 before the release), and their after-the-fact skill is mostly a
   re-encoding of the outcome flags.
2. **The recovered state adds nothing measurable to outcome models that already see the same
   event inputs.** NFL completion: PBP -> PBP+IMP -0.0009 nats [-0.0042, +0.0023]
   (game-clustered, n = 6,225) in the tracked games and -0.0010 [-0.0024, +0.0004] (n = 18,096)
   in 2022; soccer xG: EVENT -> EVENT+IMP -0.0003 [-0.0013, +0.0007] / [-0.0013, +0.0006]
   (n = 10,233 shots) in the 360 matches and +0.0000 [-0.0004, +0.0005] / [-0.0004, +0.0005]
   (n = 37,488) in 2015/16 club football. Learning curves, distillation, cross-gender transfer,
   per-slice analyses and the better sequence-student imputations (§3.6: +0.0000 [-0.0017,
   +0.0017] / [-0.0016, +0.0016], n = 6,202) find no regime where it appears. Feeding the
   imputed personnel grouping to the NFL alignment students (§2.4) recovers none of the
   charted-personnel step either (box R2 0.434 -> 0.442 vs 0.509 charted).
3. **Even perfectly known pre-instant structure is worth little at play level.** The NFL
   pre-snap oracle changes completion log-loss by +0.0027 nats [+0.0000, +0.0054]; the soccer
   360-frame oracle adds +0.0029 [+0.0013, +0.0045] / [+0.0014, +0.0044] to xG and the shot
   freeze frame +0.0078 [+0.0055, +0.0102] / [+0.0055, +0.0103]. The large oracle gains in both sports (+0.058 nats on NFL completion,
   +0.070 on pre-instant soccer xPass) come from post-release quantities that are
   quasi-outcomes and that no pre-instant imputation could recover.
4. **Small, significant exceptions exist only for pass difficulty in soccer:** EVENT+IMP gains
   +0.0016 nats [+0.0010, +0.0022] / [+0.0011, +0.0022] pre-instant in the 360 matches
   (n = 149,591) and +0.0019 [+0.0014, +0.0024] / [+0.0014, +0.0025] in 2015/16 (n = 149,621),
   about 2% of the corresponding 360-oracle gap. The sequence student's better imputations
   enlarge it (+0.0026 [+0.0019, +0.0034] / [+0.0018, +0.0033] on the 89,285 passes of its
   folds; LightGBM -> GRU imputations of the same seven quantities +0.0013 [+0.0006, +0.0020] /
   [+0.0006, +0.0020]) to about 4% of the oracle gap, still a fraction of a hundredth of a nat.
5. **The students transfer as measurements** (NFL 2017 -> 2018 official fields, soccer 360
   tournaments 2020-25 -> 2015/16 leagues with R2 vs the shot freeze frame 0.41 -> 0.41, 0.43
   -> 0.42, 0.15 -> 0.21) and the raw event history is where extra imputation skill sits
   (soccer GRU over 20 events beats LightGBM on 7/7 targets; the same network without the
   history loses on 6/7). What transfers is a weak signal.
6. **The one robust after-the-fact payoff is descriptive:** NFL receiver-week separation
   R2 0.316 -> 0.371 with an at-release student (+0.055 [+0.038, +0.072], n = 1,273 weeks) and
   -> 0.412 with the after-the-fact student (+0.096 [+0.072, +0.120]); analytics, not pricing.
   Forward, it is nothing measurable (§2.5): with a receiver's history through week t, the
   imputed separation changes next-game receiving-yards R2 by +0.003 on all receiver-games
   (+3.1 yd^2 [+0.1, +6.0] week-clustered, n = 11,422, just outside the 2.0 yd^2 refit spread),
   by -0.002 on the props-eligible receivers (-2.9 [-8.2, +2.3], n = 5,835) and nothing for
   next-game targets; even the true NGS separation history adds nothing (-0.9 [-5.3, +3.1]).

## 1. The question and the data actually available

**Question.** Player-tracking data lets one measure game state (who is on the field, how the
defence is aligned, how far the nearest defender is, when the ball was released). Such data is
rare. Can a model trained where tracking exists ("privileged information" at training time) be
used to impute that state where only event / play-by-play data exists, and does the imputed
state then improve the things one actually wants to predict?

**What was on disk** (`data/raw/privileged/`, see `nfl_01_build.md` and `soccer_01_build.md`):

| sport | frame-level tracking | abundant privileged signal | event / play-by-play data |
|---|---|---|---|
| NFL | 2017 Big Data Bowl: **91 games, weeks 1-6 of one season**, 10 Hz, 11,518 non-special-teams tracked plays with a `ball_snap` tag (6,696 pass plays), 253,370 player-plays | nflverse `pbp_participation` (NGS charting from the same tracking): personnel, formation, defenders in box, pass rushers, the 22 participant ids for 2016-2022 (243,932 regular-season scrimmage plays), coverage from 2018, `was_pressure` / `time_to_throw` / `air_yards` on ~54% of scrimmage plays (never on sacks); NGS weekly receiving (separation, cushion) | nflfastR play-by-play 2016-2022 |
| soccer | none | StatsBomb 360 freeze frames: **426 flagged matches, 417 usable** (9 have corrupt or unmatched frames), 1,357,506 events with a frame (2020-2025 tournaments and single-club seasons, men and women); a frame is the broadcast visible area only (mean 7.9 opponents visible, 73.3% of frames show >= 7); `shot.freeze_frame` (all coded players, not just the visible area) on nearly every shot in every match (41,279 of the 41,687 non-360 shots; penalties usually lack it) | StatsBomb events for 1,673 non-360 matches (2015/16 PL, La Liga, Serie A, Ligue 1; WC 2018, Copa 2024, AFCON 2023): 41,687 shots plus a fixed 25% of passes / carries per match (756,282 rows) |

Two structural differences shape everything that follows. In the NFL the tracking is complete
but tiny (one season fragment), while a rich charted feed exists for seven seasons, so the
imputation students compete with an official privileged field that is simply available. In
soccer the "tracking" is a freeze frame at every event, so there are 1.36 M labelled instants,
but each label is a visible-area truncation, and the genuinely informative privileged field for
shots (the freeze frame) exists everywhere anyway.

**Hygiene** (audited twice per chain, see `phase1_handoffs.md`, `nfl_05_audit_fix.md`): splits
hold out whole games / matches or run forward in time; tendencies use strictly earlier games;
imputed features fed to a downstream model are out-of-fold on the same folds or come from a
disjoint domain; every event-only feature list is checked against the privileged / outcome
columns; outcomes of the play being predicted are never features; every comparison carries a
paired bootstrap interval; the NFL 04 stage also reports a three-seed refit spread and treats a
delta inside it as noise even when its interval excludes zero.

## 2. NFL results

### 2.1 Participation inference: who is on the field (`nfl_02_participation.md`)

Protocol: regular-season scrimmage plays, train 2016-2020, validate 2021, test 2022 once
(n = 35,969 plays, 271 games). Feature sets: S0 situation, S1 + shotgun / no-huddle, S2 + team
tendencies from strictly earlier games.

| offense personnel grouping (10 classes), test 2022 | accuracy | log-loss | top-2 |
|---|---|---|---|
| majority class | 0.619 | 1.278 | 0.804 |
| team x down-distance prior games | 0.626 | 1.116 | 0.829 |
| S0 situation | 0.621 | 1.156 | 0.803 |
| S1 + shotgun / no-huddle | 0.627 | 1.126 | 0.806 |
| **S2 + tendencies** | **0.647** | **0.984** | **0.831** |
| S2 + post-hoc pass / run flag | 0.648 | 0.983 | 0.832 |
| S2 with the test season's labels blanked | 0.630 | 1.085 | 0.810 |

S1 -> S2 is 0.142 nats [0.125, 0.160] (game-clustered); S2 -> S2+play_type 0.0013 [-0.0012,
+0.0036]. The grouping is mostly *not* recoverable: S2 is a calibrated "11 personnel unless the
team and situation say 12 or 21" prior (recall 0.905 for `1 RB, 1 TE, 3 WR`, 0.335 for `1 RB, 2 TE,
2 WR`, ~0 for every rare class).

| defensive target given the offense grouping (test 2022) | n | S2 | + imputed grouping (delta, game-clustered CI) | + true grouping (delta) |
|---|---|---|---|---|
| defense_personnel (11 classes, log-loss) | 35,969 | 1.257 | +0.0149 [+0.0055, +0.0238] (refit-sensitive: 0.0088 in a previous fit) | +0.1369 [+0.1214, +0.1528] |
| defenders_in_box (R2 / squared error) | 35,090 | 0.453 | -0.0004 [-0.0039, +0.0028] | +0.0665 [+0.0596, +0.0737] (R2 0.514) |
| number_of_pass_rushers (R2 / squared error) | 20,186 | 0.059 | +0.0015 [-0.0007, +0.0035] | +0.0049 [+0.0019, +0.0078] |
| man_zone (log-loss) | 17,979 | 0.530 | +0.0001 [-0.0013, +0.0014] | +0.0030 [+0.0018, +0.0041] |
| coverage_type (log-loss) | 17,979 | 1.511 | -0.0000 [-0.0031, +0.0031] | +0.0014 [-0.0016, +0.0040] |

Player identity (15,000 test plays per side, candidates = the 48-man game-day active list):

| decoder | offense per-slot / exact-11 | defense per-slot / exact-11 |
|---|---|---|
| prior-usage rank, true grouping | 0.816 / 0.111 | 0.774 / 0.053 |
| model, true grouping | 0.834 / 0.136 | 0.783 / 0.063 |
| model, imputed grouping | 0.811 / 0.090 | 0.765 / 0.036 |
| prior-usage rank, imputed grouping | 0.797 / 0.075 | 0.756 / 0.031 |

(Decoder accuracies are point estimates: the NFL 02 stage-(c) tables carry no paired interval.)

Where the noise sits (share of true on-field players missed, model with the true grouping):
DL 25.5%, LB 14.4%, OL 11.3%, WR3+ 11.2%, nickel / dime CB 10.5%, RB2+ 9.6%, TE2+ 8.2%, S 6.2%,
QB 2.4%, WR1-2 0.6%, CB1-2 0.1%. On-field F1: CB1-2 0.960, OL 0.906, QB 0.899, WR1-2 0.899,
S 0.881, TE1 0.832, LB 0.798, CB3+ 0.770, RB1 0.764, DL 0.662, WR3+ 0.615, TE2+ 0.557, RB2+ 0.315.
RB1 / TE1 are on the field on only 60.8% / 70.1% of plays, so "near-deterministic" applies to
QB, OL, WR1-2 and CB1-2 only; the depth-chart QB1 is on the active list on 91.3% of test plays
and on the field on 91.1% of those (4.3% dressed non-starter, 4.8% in-game change when he did
play; 7.5% of plays had the listed QB1 declared inactive, which the weekly roster resolves 90
minutes before kick-off). Exact lineups are 0.000 whenever the imputed slot counts are wrong
(35.3% of offense, 48.9% of defense plays).

**Reading.** The team-level participation step is worth a lot when the grouping is observed and
almost nothing when it must be imputed from play-by-play; player identity beyond the fixed
starters is rotation noise that no pre-play information resolves.

### 2.2 Alignment and movement imputation (`nfl_03_imputation.md`, `nfl_03_ood_highlights.md`)

Students: LightGBM per target, 5-fold by game on the 91 tracked games (out-of-fold for every
play), plus a forward split (weeks 1-4 -> 5-6). Feature sets: F0n situation + prior-week team
tendencies; F0 + team target encodings (training-fold games only); F0P + charted personnel /
formation; F1 + post-play fields (after the fact); F2 + official box / rusher counts.
`base_outcome` = per-outcome-class training mean (an after-the-fact baseline). F0P reads the
charted (privileged) personnel string, so only the F0-based sets are event-only; the chain the
brief asked for, participation *imputed* from play-by-play feeding the alignment students, was
not part of NFL 03 and is tested in §2.4 (`F0I`).

| target (k-fold OOF, n) | base_outcome | F0 | F0P | F1 | F2 | official field as oracle | F0 verdict |
|---|---|---|---|---|---|---|---|
| shotgun_derived (11,518) | 0.216 | 0.939 | 0.957 | 0.957 | 0.958 | | well (pipeline check: nflfastR `shotgun` is description-derived) |
| qb_depth (11,471) | 0.203 | 0.868 | 0.883 | 0.883 | 0.884 | | well (inherits shotgun) |
| def_y_std (11,518) | 0.175 | 0.443 | 0.538 | 0.539 | 0.574 | | partly |
| box_count_tuned (11,518) | 0.165 | 0.434 | 0.509 | 0.513 | 0.751 | 0.699 | partly |
| n_deep_safeties (11,518) | 0.047 | 0.400 | 0.403 | 0.418 | 0.495 | | partly |
| box_count_run (4,671) | 0.031 | 0.356 | 0.432 | 0.432 | 0.658 | 0.598 | partly |
| mof_open (11,518) | 0.044 | 0.255 | 0.264 | 0.265 | 0.367 | | partly |
| cb_cushion (11,518) | 0.043 | 0.239 | 0.287 | 0.291 | 0.308 | | partly |
| n_wide_right / n_wide_left (11,518) | 0.082 / 0.076 | 0.179 / 0.159 | 0.240 / 0.209 | 0.246 / 0.221 | 0.255 / 0.224 | | weakly |
| n_backfield (11,518) | 0.059 | 0.144 | 0.487 | 0.491 | 0.495 | | weakly (personnel makes it) |
| n_dl (11,518) | 0.004 | 0.117 | 0.124 | 0.132 | 0.216 | | weakly |
| motion_derived (11,518) | 0.010 | 0.067 | 0.067 | 0.072 | 0.075 | | weakly |
| n_pass_rushers_derived (6,696) | 0.006 | 0.043 | 0.055 | 0.066 | 0.642 | 0.605 | not |
| time_to_throw (6,613) | 0.212 | 0.028 | 0.040 | 0.355 | 0.371 | 0.992 | not |
| min_def_dist_qb_throw (6,613) | 0.251 | 0.025 | 0.023 | 0.287 | 0.302 | | not |
| pressure_derived (6,613) | 0.284 | 0.008 | 0.006 | 0.275 | 0.276 | 0.786 | not |
| separation_at_arrival (6,136) | 0.090 | 0.014 | 0.022 | 0.304 | 0.308 | | not |
| n_def_within_r_target (6,136) | 0.036 | 0.007 | 0.001 | 0.140 | 0.150 | | not |
| target_depth (6,136) | 0.082 | 0.010 | 0.019 | 0.899 | 0.899 | 0.939 | not (F1 sees air yards) |
| min_def_dist_carrier_handoff (4,543) | -0.001 | 0.038 | 0.060 | 0.094 | 0.097 | | not |
| yards_to_first_contact (4,132) | 0.468 | -0.001 | -0.000 | 0.534 | 0.535 | | not |
| n_def_within_r_carrier_first_contact (4,132) | 0.107 | 0.031 | 0.035 | 0.242 | 0.244 | | not |

What the table says:

* **Pre-snap structure is partly recoverable** from situation plus tendencies (box R2 0.43, deep
  safeties 0.40, lateral spread 0.44, cushion 0.24), and the charted personnel / formation
  string adds a further step where formation is the quantity (n_backfield 0.14 -> 0.49,
  def_y_std 0.44 -> 0.54, box 0.43 -> 0.51). The forward split (n = 3,618) is within a few
  hundredths of the k-fold numbers (box F0 0.430, deep safeties 0.420, mof_open 0.239). Part of
  the F0 skill on box / safeties / cushion is team identity: the no-encoding F0n column gives
  cb_cushion 0.120 vs F0 0.239.
* **Within-play state is not recoverable before the fact** (every F0 / F0P skill < 0.05).
* **After the fact, F1 skill is mostly outcome-class identification.** Against the per-outcome-
  class mean (`F1_vs_outcome` = 1 - MSE(F1) / MSE(base_outcome)): time_to_throw 0.182,
  min_def_dist_qb_throw 0.048, pressure_derived -0.012 (F1 does not beat the class mean; inside
  clean plays, n = 5,624, BSS 0.010 / AUC 0.591), separation_at_arrival 0.235 (0.300 inside
  completions, 0.048 inside incompletions), target_depth 0.890 (air yards is the same quantity),
  yards_to_first_contact 0.124, n_def_within_r_carrier_first_contact 0.152. F1 beats
  base_outcome by game-clustered CI on 9 of 11 within-play targets, not on pressure_derived and
  n_def_within_r_target.
* **Cross-season check against the official 2018 NGS fields** (F1 students, never fitted on
  2018): box_count_tuned R2 0.538 (n = 32,315) vs 0.564 on 2017 OOF (ceiling of the derived
  definition itself 0.695); time_to_throw 0.244 (n = 17,595, non-sack plays) vs 0.234;
  pressure AUC 0.708 (n = 17,594) vs 0.707; target_depth 0.960 vs 0.937; n_pass_rushers 0.059
  vs 0.065 (point estimates; the external table carries no interval). Imputed values drift
  little across 2018-2022: max |z-shift| 0.16 over the columns the payoff stage uses, 0.28 over
  every set (`nfl_04b_shift.parquet`).
* **Out-of-distribution highlight check (revised).** On 126 regular-season 2018-19 NGS
  highlight plays (touchdowns and long gains, so a biased sample) with a validated snap frame
  and ball row, box_count_tuned R2 is F0 0.341 [0.221, 0.433] / F0P 0.475 [0.342, 0.573] / F1
  0.473 [0.338, 0.577] against 2017 OOF 0.434 / 0.509 / 0.513, and n_deep_safeties 0.388
  [0.021, 0.518] / 0.383 / 0.415 against 0.400 / 0.403 / 0.418 (play-level bootstrap): no visible
  degradation for the personnel-aware students, "neither shown nor excluded" for F0. The one
  degradation the stage report does flag is `mof_open`: BSS 0.025 / 0.025 / 0.037 (F0 / F0P /
  F1) on the highlight plays vs 0.255 / 0.264 / 0.265 on 2017 OOF, AUC 0.71 / 0.72 / 0.70 vs
  0.81, with the 2017 skill outside the highlight interval for all three students (F0 [-0.25,
  +0.21]); on this biased sample the open-middle look is rarer (0.21 vs 0.28) and the intervals
  are wide, so it is a visible but poorly measured loss. The earlier "box recovery degrades out
  of distribution" statement rested on one play with a ball row 25.6 yd off the field and is
  withdrawn (`nfl_05_audit_fix.md`); do not cite the old 0.21 / 0.31.

### 2.3 Payoff: does imputed state improve outcome prediction? (`nfl_04_payoff.md`)

Feature sets: `PBP` = pre-snap situation + prior-week tendencies + the at-release play-by-play
fields (air yards, pass length / location, `qb_hit`, receiver group, run location / gap);
`PBP+IMP` adds the F0P pre-snap imputations and at-release students (F1T: fitted on F0P inputs
plus the at-release fields, no outcome), so in the untracked seasons it reads the charted
personnel through its students and is *not* an event-only set; `PBP+IMP_F0` is the strictly
event-only variant (F0 / F0T students);
`PBP+ORACLE_PRESNAP` the true values of the 12 pre-snap targets, `PBP+ORACLE_WITHIN` the true
within-play targets (measured at or after the throw / handoff: quasi-outcomes), `PBP+NGS` the
official charting (`was_pressure`, `time_to_throw`, box, rushers; after the fact). Deltas vs PBP,
game-clustered CI; refit spread = max - min of the mean loss over three seeds.

**(a) Tracked 2017 games, 5-fold by game** (n = 6,225 non-sack pass plays, 4,671 runs, 91 games):

| task (loss) | PBP | PBP+IMP | PBP+IMP_F0 | pre-snap oracle | within-play oracle | full oracle | PBP+NGS |
|---|---|---|---|---|---|---|---|
| completion (log-loss) | 0.6078 | -0.0009 [-0.0042, +0.0023] | -0.0002 [-0.0028, +0.0027] | +0.0027 [+0.0000, +0.0054] | +0.0578 [+0.0473, +0.0674] | +0.0565 [+0.0457, +0.0662] | +0.0223 [+0.0167, +0.0276] |
| pass yards (squared error, yd^2) | R2 0.090 | +0.12 [-0.68, +1.03] | -0.66 [-1.30, +0.00] | +0.45 [-0.22, +1.13] | +4.22 [+2.76, +5.68] | +4.72 [+3.30, +6.08] | +0.25 [-0.36, +0.86] |
| pass EPA (squared error) | R2 0.014 | +0.006 [-0.008, +0.020] | -0.004 [-0.016, +0.007] | -0.004 [-0.015, +0.007] | +0.095 [+0.068, +0.123] | +0.095 [+0.068, +0.123] | +0.009 [-0.003, +0.020] |
| run yards (squared error, yd^2) | R2 0.025 | +0.04 [-0.21, +0.29] | +0.01 [-0.31, +0.31] | -0.02 [-0.33, +0.32] | +15.67 [+13.15, +18.09] | +15.85 [+13.27, +18.38] | +0.13 [-0.04, +0.32] |
| run success (log-loss) | 0.6432 | +0.0014 [-0.0021, +0.0047] | -0.0019 [-0.0052, +0.0014] | +0.0010 [-0.0013, +0.0033] | +0.1857 [+0.1680, +0.2033] | +0.1864 [+0.1695, +0.2034] | +0.0003 [-0.0025, +0.0032] |

Refit spread of the PBP completion model: 0.0018 nats (k-fold), so the pre-snap oracle's +0.0027
is real but negligible (0.4% of the PBP log-loss) and every imputed delta is inside noise. Like
with like on the 6,155 cp-present plays, nflfastR `cp` (0.5867) beats the PBP set (0.6040) by
+0.0173 [+0.0133, +0.0215]; `cp` is fitted on many seasons including 2017 and is a reference. The
leaky F1 students reach completion log-loss 0.256: that is the label, not a payoff.

**(b) Untracked seasons, train 2018-2021, test 2022** (n = 18,096 pass / 14,377 run plays, 271
games; students trained on the 2017 tracked games only):

| task | PBP | PBP -> PBP+IMP | PBP -> PBP+IMP_F0 | PBP -> PBP+NGS | PBP+NGS -> PBP+IMP+NGS |
|---|---|---|---|---|---|
| completion (log-loss) | 0.5482 | -0.0010 [-0.0024, +0.0004] | -0.0002 [-0.0015, +0.0010] | +0.0356 [+0.0316, +0.0397] | -0.0013 [-0.0028, +0.0002] |
| pass yards (yd^2) | R2 0.136 | -0.08 [-0.44, +0.26] | -0.40 [-0.85, -0.00] (within the 0.56 refit spread of the pair) | +1.36 [+0.91, +1.79] | -0.23 [-0.63, +0.17] |
| pass EPA | R2 0.046 | -0.004 [-0.014, +0.007] | +0.006 [-0.000, +0.013] | +0.078 [+0.059, +0.097] | -0.034 [-0.048, -0.022] |
| run yards (yd^2) | R2 0.044 | -0.03 [-0.14, +0.06] | -0.01 [-0.11, +0.10] | +0.02 [-0.11, +0.15] | +0.05 [-0.06, +0.16] |
| run success (log-loss) | 0.6477 | -0.0009 [-0.0024, +0.0005] | -0.0021 [-0.0037, -0.0005] (within the 0.0024 refit spread) | +0.0006 [-0.0008, +0.0020] | -0.0002 [-0.0015, +0.0010] |

Of the ten PBP -> imputed deltas, none is an improvement; two are nominal degradations inside
refit noise. On the 17,306 cp-present plays nflfastR `cp` (0.5702) is ahead of the PBP set
(0.5723) by +0.0021 [+0.0006, +0.0036], inside the 0.0031-nat refit spread of the PBP model; only
`PBP+NGS` (0.5358 on those rows) beats `cp` clearly (+0.0345 [+0.0301, +0.0388]). Stacked on the
official charting the imputations are neutral or slightly harmful (EPA -0.034 EPA^2).

**(c) The props angle: receiver-week separation vs NGS** (OLS fitted on 2018-2021 receiver-weeks,
scored on 2022, n = 1,273; NGS weekly rows exist for receivers with >= 5 targets):

| proxy set for NGS `avg_separation` | test R2 | delta R2 vs the fair baseline (per-week bootstrap CI) |
|---|---|---|
| naive event proxies (air yards, cp, target share, n) | 0.184 | -0.132 |
| after-the-fact aggregates only (completion rate, YAC, yards, INT, QB-hit rate, EPA) | 0.307 | -0.009 |
| **naive + after-the-fact (fair baseline)** | **0.316** | |
| + at-release student F1T | 0.371 | +0.055 [+0.038, +0.072] |
| + at-release student F0T (no personnel) | 0.377 | +0.061 [+0.041, +0.079] |
| + after-the-fact student F1 | 0.412 | +0.096 [+0.072, +0.120] |

The earlier "0.184 -> 0.412" reading compared the F1 student against a baseline lacking the outcome
fields it consumes and overstated the gain twofold. At play level a linear regression on outcome
fields reproduces the F1 imputed separation with R2 0.67 (at-release fields alone 0.34), so most
of the F1 signal is a re-encoding of outcome flags. For `avg_cushion` nothing works (best R2 0.044).
This is a same-week, after-the-fact reproduction of an NGS field; the forward question a props
price needs is §2.5.

**(d) Participation payoff on 2022 pass plays** (n = 18,096, n_train = 71,762): imputed offense
grouping probabilities +0.0002 nats [-0.0008, +0.0011]; true grouping +0.0004 [-0.0006, +0.0013];
imputed -> true +0.0002 [-0.0007, +0.0011]; imputed state + imputed grouping +0.0013 [+0.0001,
+0.0026], the only pair with an interval excluding zero and inside the 0.0016-nat refit spread of
the pair. Once the at-release fields are known, the personnel grouping adds nothing measurable.

**Bottom line (NFL).** Perfectly known pre-snap structure has no measurable play-level payoff
given the at-release play-by-play fields; the whole oracle gain is post-release within-play
state; the event-only students are deterministic functions of fields the outcome model already
sees and LightGBM extracts that information itself. The only usable extra signal in untracked
seasons is the official NGS charting, which is after the fact.

### 2.4 The participation -> alignment chain with imputed personnel (`nfl_06_participation_chain.md`)

The brief's three-stage chain (participation -> alignment -> movement) was never wired in
phase 1: NFL 03's `F0` uses no personnel and `F0P` the charted string. This stage adds `F0I` =
F0 + the ten NFL 02 offense-grouping probabilities (`p_off_*`, out-of-fold by game over
2016-2020; on the tracked plays the argmax matches the participation label on 63.4% vs a
58.4% majority class) and `F0PI` = F0P + the same probabilities, on the 12 pre-snap targets
with the NFL 03 protocol (5-fold by game, n = 11,518; forward split weeks 1-4 -> 5-6).

| target (k-fold OOF) | F0 | F0I | F0P | F0PI | F0 -> F0I (game-clustered CI) | F0I -> F0P |
|---|---|---|---|---|---|---|
| box_count_tuned | 0.434 | 0.442 | 0.509 | 0.513 | +0.0018 yd [-0.0033, +0.0062] | +0.0285 [+0.0223, +0.0344] |
| def_y_std | 0.443 | 0.443 | 0.538 | 0.537 | -0.0002 [-0.0049, +0.0047] | +0.0734 [+0.0653, +0.0817] |
| n_backfield | 0.144 | 0.148 | 0.487 | 0.486 | +0.0001 [-0.0025, +0.0027] | +0.1128 [+0.1016, +0.1228] |
| cb_cushion | 0.239 | 0.230 | 0.287 | 0.277 | -0.0032 [-0.0101, +0.0043] | +0.0539 [+0.0428, +0.0652] |
| n_deep_safeties | 0.400 | 0.393 | 0.403 | 0.396 | -0.0013 [-0.0045, +0.0016] | +0.0004 [-0.0036, +0.0043] |
| mof_open (BSS, log-loss delta) | 0.255 | 0.257 | 0.264 | 0.259 | +0.0015 [-0.0020, +0.0052] | +0.0042 [+0.0006, +0.0077] |
| qb_depth | 0.868 | 0.874 | 0.883 | 0.885 | +0.0108 [+0.0026, +0.0208] | +0.0089 [+0.0022, +0.0147] |
| n_wide_right | 0.179 | 0.183 | 0.240 | 0.240 | +0.0019 [+0.0001, +0.0038] | +0.0344 [+0.0303, +0.0386] |
| shotgun_derived (BSS) | 0.939 | 0.939 | 0.957 | 0.956 | -0.0006 [-0.0035, +0.0017] | +0.0219 [+0.0140, +0.0312] |

(Deltas are absolute-error or log-loss reductions, positive = the second set better; the other
three targets, n_dl, n_wide_left and motion_derived, are in the stage table.) `F0 -> F0I` is
better by clustered CI on 2 of 12 targets (qb_depth, n_wide_right) and worse on none; `F0I ->
F0P` is better on 10 of 12; `F0P -> F0PI` on 1 of 12 (n_wide_left) and worse on 1
(shotgun_derived). The forward split agrees (box 0.430 / 0.430 / 0.495 / 0.499). The imputed
grouping is a calibrated "11 personnel unless the situation says otherwise" prior that the
situation features already encode, so chaining it into the alignment students recovers none of
the charted-personnel step; the payoff stage had reached the same null for the same
probabilities as direct outcome features (§2.3 d). The chain was not carried further into
NFL 04 with `F0I` students: `PBP+IMP_F0` remains the strictly event-only payoff variant.

### 2.5 The props angle, forward: next-game receiving yards and targets (`nfl_07_props_forward.md`)

§2.3(c) reproduces a same-week NGS field after the fact. A prop price is forward-looking, so
this stage predicts a receiver's NEXT game from his history: 87,433 targeted passes of
2018-2022 aggregated per receiver-game (n_targets, receptions, receiving yards, air yards,
`cp`, EPA, target share, mean imputed separation / cushion / target depth of the targeted
plays), features = current game, season-to-date and last-three-game means plus games played
and week (all from weeks <= t), target = receiving yards / targets of the receiver's next
game with a target in the season. LightGBM, rolling origin (test 2020, 2021, 2022, each
trained on the seasons before it), paired bootstrap of per-row squared error vs the
event-only `HIST` model with a (season, week)-clustered interval, three-seed refit spread.
Populations: `all` (n = 11,422 test receiver-games) and `regular` (>= 3 games and >= 3
targets per game season-to-date: the props-eligible receivers, n = 5,835).

| next-game target, population | BASE_STD R2 | HIST R2 (MAE) | + IMP_F0T | + IMP_F1T | + IMP_F1 (after the fact) | + IMP_ALL | + NGS history (oracle) |
|---|---|---|---|---|---|---|---|
| receiving yards, all | 0.213 | 0.294 (19.9 yd) | +0.5 yd^2 [-2.5, +3.6] | +3.1 [+0.1, +6.0] | +1.6 [-0.8, +3.9] | +2.4 [-0.8, +5.7] | -0.9 [-5.3, +3.1] |
| receiving yards, regular | 0.198 | 0.232 (23.5 yd) | +1.5 [-5.2, +7.6] | -2.9 [-8.2, +2.3] | -0.2 [-3.7, +3.1] | -3.0 [-9.7, +4.1] | +0.8 [-4.5, +5.7] |
| targets, all | 0.344 | 0.398 (1.87) | +0.017 [-0.006, +0.040] | +0.017 [-0.004, +0.039] | +0.011 [-0.016, +0.037] | +0.007 [-0.016, +0.031] | -0.003 [-0.027, +0.022] |
| targets, regular | 0.253 | 0.279 (2.20) | -0.001 [-0.031, +0.027] | -0.012 [-0.050, +0.027] | -0.013 [-0.038, +0.013] | -0.013 [-0.047, +0.022] | -0.042 [-0.079, -0.002] |

(`BASE_STD` = the season-to-date mean of the target as the prediction; deltas are squared-error
reductions vs HIST, positive = better, week-clustered 95% CI; the receiving-yards refit spread
is 2.0 yd^2 on `all` and 3.2 on `regular`, the targets spread 0.014-0.027.) The only interval
that excludes zero in the imputed direction is +3.1 yd^2 for the at-release F1T student on all
receiver-games (delta R2 +0.003), which reverses sign on the props-eligible population and does
not reproduce for targets; the true NGS separation history is no better. What predicts the next
game is the event history itself (HIST beats the season-to-date mean by +84 yd^2 [+53, +120]).
No market lines are on disk, so this is skill vs event-only history, not vs a bookmaker.

## 3. Soccer results

### 3.1 What state is recoverable from events (`soccer_02_imputation.md`)

Students: LightGBM per target, 5-fold by match over the 417 usable-360 matches (training rows
thinned to 300k per fit), out-of-fold for every event. Feature sets: `loc` location only; `E0`
current event; `E1` + possession context; `E2` + 10-event window counts, opponent defensive
actions in the last 60 s, ball speed / displacement; `E2a` + the 16 post-instant `f_after_*`
columns (realised pass end / length / height, carry end, duration).

| target (subset, n) | base_type | loc | E0 | E2 | E2a | E2 MAE | verdict |
|---|---|---|---|---|---|---|---|
| block_depth (possession, reliable frames; 850,149) | 0.032 | 0.861 | 0.941 | **0.964** | 0.965 | 3.0 yd | recoverable |
| def_line (850,149) | 0.031 | 0.845 | 0.929 | **0.952** | 0.952 | 3.0 yd | recoverable |
| deep_block (settled possession; 235,225; BSS) | 0.050 | 0.645 | 0.700 | **0.772** | 0.774 | AUC 0.984 | recoverable |
| n_opp_ahead_of_ball (850,149) | 0.054 | 0.200 | 0.527 | **0.671** | 0.674 | 1.17 | recoverable |
| nearest_opp_dist (all usable frames; 1,336,461) | 0.163 | 0.123 | 0.597 | **0.668** | 0.698 | 2.07 yd | recoverable |
| n_opp_within_10 (1,337,830) | 0.080 | 0.237 | 0.514 | **0.621** | 0.630 | 0.66 | recoverable |
| n_opp_within_5 (1,337,830) | 0.142 | 0.145 | 0.450 | **0.523** | 0.537 | 0.40 | recoverable |
| opp_keeper_dist_to_goal_line (attacking half, keeper visible; 233,791) | 0.010 | 0.322 | 0.399 | **0.476** | 0.484 | 1.5 yd | partly |
| block_width (850,149) | 0.039 | 0.286 | 0.372 | **0.464** | 0.466 | 4.9 yd | partly |
| block_length (850,149) | 0.019 | 0.320 | 0.390 | **0.436** | 0.438 | 4.2 yd | partly |
| nearest_opp_dist_in_cone (non-empty cone; 293,797) | 0.040 | 0.244 | 0.380 | **0.415** | 0.419 | 6.1 yd | partly |
| counter_on (middle third; 456,199; BSS) | 0.037 | 0.005 | 0.233 | **0.341** | 0.345 | AUC 0.927 | partly |
| n_opp_in_lane (Pass; 373,481) | -0.000 | 0.125 | 0.247 | **0.292** | 0.484 | 0.44 | partly |
| n_opp_in_cone (possession; 1,113,443) | 0.008 | 0.050 | 0.127 | **0.160** | 0.160 | 0.39 | weak |
| n_opp_within_3_of_end (Pass; 373,481) | -0.000 | 0.124 | 0.144 | **0.150** | 0.436 | 0.14 | weak |

Every nested step base_type -> loc -> E0 -> E1 -> E2 is significant under a match-clustered
bootstrap except n_opp_within_3_of_end E1 -> E2 and n_opp_within_5 base_type -> loc. Mechanism
caveats that matter downstream: block depth / defensive line / deep block are mostly ball
location plus visible-area truncation (loc alone gives R2 0.86 on block depth; the incremental
event-only skill is 0.10 R2, MAE 6.1 -> 3.0 yd); the post-instant `f_after_*` block matters only
where the target is defined by the realised trajectory (pass-lane counts 0.29 -> 0.48, 0.15 ->
0.44; carry length reveals the nearest defender, 0.668 -> 0.698); shot-state quantities (cone
count 0.16, cone distance 0.42, keeper distance 0.48) are the weakest and are exactly the ones
xG needs. Reliability: training on reliable frames only never helps (E2r). Error grows with the
time since the last informative event: nearest-opponent MAE 1.13 yd when the opponent's last
defensive action was the previous event, 1.77 at one event, 2.20 at 3-5 events, 2.2-2.3 beyond
(n 95k-378k per bin); counter_on log-loss 0.25 at 0 events vs 0.11-0.14 at 10+; block depth and
def line are flat (2.7-3.2 yd) over that dimension and over elapsed possession time.

### 3.2 Cross-competition and forward transfer of the students (`soccer_02_imputation.md`)

E2 students trained on one domain and scored on the other (in-domain reference = the CV
out-of-fold prediction on the same rows):

| target | men -> women (in-domain) | women -> men (in-domain) | <= 2023/24 -> Euro 2024 + Women's Euro 2025 (in-domain) |
|---|---|---|---|
| block_depth | 0.967 (0.971) | 0.952 (0.962) | 0.966 (0.967) |
| def_line | 0.954 (0.960) | 0.933 (0.949) | 0.955 (0.956) |
| deep_block (BSS) | 0.783 (0.790) | 0.749 (0.767) | 0.789 (0.788) |
| n_opp_ahead_of_ball | 0.589 (0.630) | 0.551 (0.668) | 0.671 (0.683) |
| nearest_opp_dist | 0.680 (0.697) | 0.629 (0.654) | 0.681 (0.684) |
| n_opp_within_5 | 0.518 (0.535) | 0.493 (0.513) | 0.545 (0.545) |
| block_width | 0.227 (0.397) | 0.258 (0.456) | 0.471 (0.473) |
| counter_on (BSS) | 0.312 (0.326) | 0.327 (0.348) | 0.366 (0.369) |

(n women's rows 189,529 for shape targets and 374,247 for ball targets; men's 660,620 / 963,583;
forward 162,468 / 252,098.) These are point estimates: the stage-02 transfer table carries no
paired interval, so "forward transfer loses <= 0.021 on every target" is a difference of skills,
not a tested claim; cross-gender transfer loses little except on block width / length and
n_opp_ahead_of_ball. The stage-02 review noted that the transferred BSS of the two binaries uses
the source base rate as reference and understates the counter_on gap (0.023 m2f / 0.060 f2m with
a consistent reference); the deep_block / counter_on cells above are the uncorrected values, and
the conclusion "small loss" stands for every other target.

### 3.3 Sequence model vs tabular student (`soccer_05_sequence.md`, `soccer_02_imputation.md`)

A PyTorch multi-task GRU over the raw 20-event history (`seq20`) plus an MLP over the E2 design,
trained on folds 0-2 of the stage-02 split (251 held-out matches, 812,427 rows, CPU budget) and
scored against the stage-02 LightGBM E2 out-of-fold predictions on identical rows. `seq1` sees
only the previous event; `seq0` has no sequence branch.

| target (n rows) | base_type | lgbm_E2 | seq0 | seq1 | seq20 | lgbm_E2 -> seq20: loss reduction (match-clustered CI) |
|---|---|---|---|---|---|---|
| block_depth (514,853) | 0.032 | 0.964 | 0.960 | 0.966 | **0.973** (MAE 3.01 -> 2.61 yd) | squared error +3.96 yd^2 [+3.75, +4.17] |
| def_line (514,853) | 0.030 | 0.952 | 0.948 | 0.953 | **0.961** | +3.10 [+2.92, +3.29] |
| n_opp_ahead_of_ball (514,853) | 0.054 | 0.671 | 0.659 | 0.687 | **0.728** | +0.365 [+0.345, +0.386] |
| nearest_opp_dist (799,585) | 0.163 | 0.666 | 0.652 | 0.665 | **0.683** | +0.430 [+0.388, +0.478] |
| n_opp_in_cone (666,441) | 0.009 | 0.160 | 0.158 | 0.167 | **0.175** | +0.0047 [+0.0041, +0.0054] |
| deep_block (143,078; BSS) | 0.053 | 0.774 | 0.757 | 0.770 | **0.792** | log-loss +0.0100 [+0.0075, +0.0123] |
| counter_on (277,676; BSS) | 0.036 | 0.343 | 0.327 | 0.360 | **0.410** | log-loss +0.0204 [+0.0182, +0.0225] |

(The last column is `delta_loss` = loss(lgbm_E2) - loss(seq20), positive = seq20 better.)
seq20 beats LightGBM on 7/7 targets, seq0 loses on 6/7 (the architecture alone does not beat
trees), seq20 beats seq1 on 7/7: the gain is the raw history. It grows with possession age for
the shape targets (block-depth MAE 2.91 vs 3.06 yd under 2 s of possession, 2.27 vs 2.75 beyond
40 s) and is largest for counter_on early in a possession (log-loss 0.185 vs 0.212 at 5-10 s).
The neural student is worse on contact events whose label is a near-constant 0.14 yd, and its
binary heads are 2-3x less calibrated (ECE 0.008-0.009 vs 0.003-0.005). The stage-02
scikit-learn MLP over the one-hot E2 design *plus a flattened 20-event block* (`mlp_E2seq`, 467
features) had lost to LightGBM E2 on the same rows (nearest_opp_dist 0.636 vs 0.666 on fold 0),
so the stage-05 gain is specific to the GRU encoding of the history, not to having the history
as an input; LightGBM with the raw sequence block (E3) reached 0.972 / 0.681 on block depth /
nearest_opp_dist, so trees plus the history close part of the gap too. Whether the better
imputations pay off downstream is §3.6.

### 3.4 Payoff: xG and xPass in the 360 matches (`soccer_03_payoff.md`)

xG on 10,233 non-penalty shots (1,039 goals, 417 matches), 5-fold by match on the stage-02
folds, 3 seeds. `EVENT` = the stage-02 E2 design of the shot plus the key pass's attributes (72
features; it contains every input the students read); `EVENT+IMP` adds the out-of-fold E2 shot-
state and E2a assist-lane imputations; `EVENT+ORACLE360` the same quantities from the 360 frame;
`EVENT+ORACLESHOT` from `shot.freeze_frame`; `STATSBOMB_XG` the vendor's model (possibly trained
on these matches: an upper reference).

| xG variant | log-loss | AUC | delta vs EVENT (per-shot CI / match-clustered CI) |
|---|---|---|---|
| BASE (training-fold goal rate) | 0.3286 | 0.482 | -0.0541 [-0.0604, -0.0478] / [-0.0601, -0.0478] |
| LOC (5 geometry features) | 0.2945 | 0.724 | -0.0200 [-0.0238, -0.0164] / [-0.0237, -0.0163] |
| **EVENT** | **0.2745** | 0.785 | |
| EVENT+IMP(E0) | 0.2744 | 0.784 | +0.0001 [-0.0008, +0.0010] / [-0.0008, +0.0009] |
| EVENT+IMP | 0.2748 | 0.784 | -0.0003 [-0.0013, +0.0007] / [-0.0013, +0.0006] |
| EVENT+ORACLE360 | 0.2716 | 0.789 | +0.0029 [+0.0013, +0.0045] / [+0.0014, +0.0044] |
| EVENT+ORACLESHOT | 0.2666 | 0.800 | +0.0078 [+0.0055, +0.0102] / [+0.0055, +0.0103] |
| EVENT+ORACLESHOT_full (every `sff_*` column) | 0.2639 | 0.804 | +0.0106 [+0.0080, +0.0133] / [+0.0079, +0.0134] |
| STATSBOMB_XG | 0.2617 | 0.809 | +0.0128 [+0.0090, +0.0168] / [+0.0089, +0.0166] |

ECE is 0.007-0.010 for every variant; the imputed block receives 12-14% of the LightGBM gain
share (it is used, it does not help). Three follow-ups all come back null:

* **Distillation** (event-only student trained on an oracle teacher's xG, nested CV): EVENT <-
  ORACLE360 at alpha 1 -0.0001 [-0.0013, +0.0011] / [-0.0012, +0.0011]; at alpha 0.5 +0.0006
  [+0.0000, +0.0012] / [+0.0000, +0.0012] (borderline, best of four); EVENT <- ORACLESHOT
  +0.0001 [-0.0012, +0.0013] / [-0.0012, +0.0014]. Students reproduce their teacher's
  out-of-fold xG with correlation 0.91-0.96, which does not translate into goals.
* **Learning curve** (share of training matches, EVENT+IMP vs EVENT): 10% +0.0000 [-0.0012,
  +0.0013] / [-0.0013, +0.0013] (EVENT 0.2918); 25% +0.0015 [-0.0001, +0.0031] / [-0.0000,
  +0.0030] (0.2856); 50% -0.0012 [-0.0023, -0.0000] / [-0.0024, -0.0000] (0.2784); 100% -0.0003
  (0.2745). ORACLESHOT stays +0.007 to +0.009 at every share.
* **Cross-gender** (students and xG trained on the source gender only): men -> women (n = 3,231)
  EVENT 0.2634, EVENT+IMP -0.0004 [-0.0021, +0.0013] / [-0.0021, +0.0013], ORACLE360 +0.0040
  [+0.0014, +0.0066] / [+0.0016, +0.0066], ORACLESHOT +0.0092 [+0.0052, +0.0134] / [+0.0046,
  +0.0135]; women -> men (n = 7,002) EVENT 0.2895, EVENT+IMP -0.0006 [-0.0017, +0.0004] /
  [-0.0017, +0.0004], ORACLE360 +0.0019 [-0.0003, +0.0040] per-shot but [+0.0000, +0.0039]
  match-clustered (a verdict flip: borderline, not null), ORACLESHOT +0.0104 [+0.0070, +0.0138] /
  [+0.0069, +0.0140]. The in-domain CV EVENT model is better than the transferred one by
  +0.0024 [-0.0002, +0.0051] / [+0.0000, +0.0048] for women (a second flip: borderline; the
  in-domain EVENT+IMP row flips the same way, +0.0025 [-0.0003, +0.0054] / [+0.0000, +0.0052])
  and by +0.0088 [+0.0053, +0.0123] / [+0.0055, +0.0125] for men (clear).
* **Slices**: no play-pattern, distance or frame-reliability slice shows a significant EVENT+IMP
  effect; oracle gains concentrate close to goal (ORACLESHOT +0.0220 [+0.0106, +0.0332] /
  [+0.0115, +0.0346] at 0-8 yd, n = 988; +0.0013 [-0.0005, +0.0034] / [-0.0004, +0.0031] at 25+
  yd, n = 2,156).

xPass on 149,591 match-stratified pass attempts (completion 0.826), single seed:

| design | BASE | EVENT | EVENT+IMP (delta, per-pass CI / match-clustered CI) | EVENT+ORACLE (delta) | EVENT-nodur |
|---|---|---|---|---|---|
| destination known (`after`, E2a students) | 0.4624 | 0.2089 | 0.2083 (+0.0005 [-0.0001, +0.0011] / [-0.0000, +0.0011]) | 0.2009 (+0.0080 [+0.0072, +0.0088] / [+0.0072, +0.0088]) | 0.2229 (-0.0140 [-0.0148, -0.0131] / [-0.0150, -0.0130]) |
| pre-instant (`noafter`, E2 students) | 0.4624 | 0.3486 | 0.3470 (+0.0016 [+0.0010, +0.0022] / [+0.0011, +0.0022]) | 0.2783 (+0.0703 [+0.0685, +0.0721] / [+0.0681, +0.0723]; not attainable, the lane targets are defined by the realised end) | |

A methodological finding of this stage: with the realised pass duration absent from EVENT (as
the task's feature list implied) but present in the E2a students, EVENT+IMP appeared to gain
+0.009 and to beat the 360 oracle on short passes; the imputations were a conduit for an omitted
event field. Any downstream use of imputations must give the event-only baseline every field the
students read.

### 3.5 Transfer to seasons without 360 (`soccer_04_transfer.md`)

Frozen stage-02 bundles (trained on all 417 360 matches) applied to 2015/16 club football and to
three non-360 tournaments; no non-360 match ever entered a student's training set.

**Oracle check as a measurement** (imputed E2 shot state vs `shot.freeze_frame`; the in-domain
bias is the shot-frame vs 360-frame measurement gap):

| quantity | 360 OOF vs 360 frame (own label) | 360 OOF vs shot frame, R2 / MAE / bias | 2015/16 leagues vs shot frame | WC 2018 |
|---|---|---|---|---|
| n_opp_in_cone | 0.421 (n 10,057) | 0.409 / 0.60 / -0.18 (n 10,067) | 0.408 / 0.56 / -0.09 (n 37,034) | 0.233 (n 1,625) |
| nearest_opp_dist_in_cone (yd) | 0.427 (n 4,466) | 0.430 / 2.38 / +0.53 (n 5,412) | 0.424 / 2.50 / +0.36 (n 19,080) | 0.463 (n 871) |
| opp_keeper_dist_to_goal_line (yd) | 0.247 (n 9,474) | 0.150 / 1.14 / +0.50 (n 10,059) | 0.208 / 1.19 / +0.27 (n 36,974) | -0.158, bias +1.0 (n 1,623) |
| n_opp_within_5 | 0.495 (n 10,223) | 0.525 / 0.72 / -0.03 (n 10,233) | 0.506 / 0.68 / +0.03 (n 37,488) | 0.498 (n 1,638) |

**xG in the 2015/16 leagues** (37,488 non-penalty shots, 3,569 goals, 1,517 matches; 5-fold by
match inside 2015/16, 3 seeds; deltas vs the within-2015/16 EVENT model, per-shot CI /
match-clustered CI):

| variant | log-loss | delta vs EVENT |
|---|---|---|
| BASE / LOC | 0.3144 / 0.2760 | -0.0593 [-0.0628, -0.0559] / [-0.0628, -0.0557]; -0.0208 [-0.0227, -0.0189] / [-0.0227, -0.0186] |
| **EVENT** | **0.2552** (AUC 0.804) | |
| EVENT+IMP | 0.2551 | +0.0000 [-0.0004, +0.0005] / [-0.0004, +0.0005] |
| EVENT+IMP(E0) | 0.2550 | +0.0001 [-0.0002, +0.0005] / [-0.0002, +0.0005] |
| EVENT+ORACLESHOT | 0.2489 | +0.0063 [+0.0051, +0.0075] / [+0.0052, +0.0074] |
| EVENT+ORACLESHOT_full | 0.2473 | +0.0079 [+0.0066, +0.0092] / [+0.0067, +0.0091] |
| STATSBOMB_XG | 0.2484 | +0.0068 [+0.0050, +0.0086] / [+0.0050, +0.0086] |
| EVENT, zero-shot stage-03 model | 0.2599 | -0.0047 [-0.0059, -0.0035] / [-0.0059, -0.0035] (mixes domain shift with 4x fewer training shots; under-predicts the goal rate, 0.0875 vs 0.0952) |
| EVENT+IMP, zero-shot | 0.2600 | -0.0048 [-0.0060, -0.0036] / [-0.0060, -0.0036]; vs the zero-shot EVENT model -0.0001 [-0.0005, +0.0003] / [-0.0005, +0.0003] (`soccer_04_xg_deltas_zeroshot.parquet`) |
| EVENT+ORACLESHOT, zero-shot | 0.2533 | +0.0018 [+0.0003, +0.0033] / [+0.0003, +0.0033] (better than the within-CV event model) |

Per league (2 seeds) EVENT+IMP: PL +0.0017 [+0.0004, +0.0029] / [+0.0005, +0.0029], La Liga
-0.0006 [-0.0017, +0.0004] / [-0.0017, +0.0005], Serie A -0.0005 [-0.0016, +0.0006] / [-0.0016,
+0.0005], Ligue 1 +0.0010 [-0.0003, +0.0022] / [-0.0003, +0.0023]: one nominal hit in four
tests, the pooled null is primary; ORACLESHOT is +0.0048 to +0.0061 and significant in every
league under both intervals. The only positive slice is 0-8 yd (+0.0026 [+0.0006, +0.0045] /
[+0.0006, +0.0045], n = 2,936), where the shot frame gives +0.0133 [+0.0070, +0.0194] /
[+0.0067, +0.0193]. Non-360 tournaments (3,541 shots, 296 goals, 148 matches): EVENT 0.2528,
EVENT+IMP +0.0006 [-0.0022, +0.0035] / [-0.0023, +0.0033], ORACLESHOT +0.0019 [-0.0020,
+0.0057] / [-0.0021, +0.0061]; here the zero-shot 360 model (0.2469) beats the small
within-tournament CV (+0.0059 [+0.0005, +0.0113] / [+0.0010, +0.0108]) and the zero-shot
imputations add +0.0015 [+0.0001, +0.0029] / [+0.0001, +0.0028] over the zero-shot event
model, the smallest-n and least consistent of the positive xG slices (the zero-shot ORACLESHOT
on the same rows is the fourth verdict flip: +0.0039 [+0.0001, +0.0077] per-shot, [-0.0002,
+0.0082] clustered).

**xPass 2015/16** (149,621 attempts, completion 0.775): destination known EVENT 0.2515,
EVENT+IMP 0.2502 (+0.0013 [+0.0007, +0.0019] / [+0.0006, +0.0019]), EVENT-nodur 0.2666
(-0.0151 [-0.0160, -0.0143] / [-0.0161, -0.0143]); pre-instant EVENT 0.4097, EVENT+IMP 0.4078
(+0.0019 [+0.0014, +0.0024] / [+0.0014, +0.0025]); per league pre-instant +0.0009 (Serie A,
[-0.0000, +0.0019] / [-0.0000, +0.0020]) to +0.0025 (La Liga, [+0.0014, +0.0035] / [+0.0012,
+0.0035]).

**Team level** (3,034 team-matches, 80 team-seasons; Spearman point estimates with p-values,
no paired interval): the imputed E2 block depth conceded correlates with the event-only
defensive-line proxy `def_line_x` at +0.79 (team-season) / +0.77 (team-match), with PPDA
-0.55 / -0.33. But mean ball x alone gives -0.78 (sign reversed by construction) and the E0
student +0.79, so the ranking is where the ball is, not what the window features add;
`def_line_x` is also one of the E2 inputs. Worked example (West Bromwich Albion vs Manchester
City, 538 imputed rows): per-minute imputed block depth vs the team's own defensive-action x
correlates 0.61 (WBA, 67 minutes) / 0.76 (City, 48 minutes).

### 3.6 Payoff of the sequence student's imputations (`soccer_06_seq_payoff.md`)

The first review noted that the payoff null rested on the LightGBM imputations only. This
stage refits the stage-03 xG and xPass models on the 251 matches of the stage-05 folds
(6,202 non-penalty shots, 621 goals; 89,285 passes of the stage-03 subsample; 3-fold CV by
match, every imputed column out-of-fold) with the seven `seq20` heads as features:
`EVENT+SEQ7(lgbm)` = EVENT + the LightGBM E2 predictions of the same seven quantities (the
like-for-like row), `EVENT+SEQ7(seq20)` = EVENT + the GRU predictions, `EVENT+IMP+SEQ7(seq20)`
= the stage-03 block plus the seven GRU columns. On these rows the GRU is the better
measurement of all seven quantities on the pass rows (e.g. counter_on log-loss 0.094 -> 0.086,
block depth R2 0.969 -> 0.973) and of five of the six measurable on the shot rows (block depth
0.788 -> 0.810, opponents ahead of the ball 0.747 -> 0.786, cone count 0.428 -> 0.434; nearest
opponent distance slightly worse, 0.560 -> 0.547; counter_on is undefined on all but 14 shots).

| task (n) | EVENT | EVENT+IMP | EVENT+SEQ7(lgbm) | EVENT+SEQ7(seq20) | EVENT+IMP+SEQ7(seq20) | lgbm -> seq20 (same seven inputs) | EVENT+ORACLE360 |
|---|---|---|---|---|---|---|---|
| xG (6,202) | 0.2713 | -0.0005 [-0.0021, +0.0012] / [-0.0021, +0.0013] | +0.0008 [-0.0006, +0.0023] / [-0.0006, +0.0023] | +0.0000 [-0.0017, +0.0017] / [-0.0016, +0.0016] | +0.0000 [-0.0018, +0.0019] / [-0.0018, +0.0019] | -0.0008 [-0.0026, +0.0010] / [-0.0025, +0.0010] | +0.0017 [-0.0005, +0.0039] / [-0.0006, +0.0039] |
| xPass pre-instant (89,285) | 0.3490 | +0.0020 [+0.0012, +0.0028] / [+0.0011, +0.0029] | +0.0013 [+0.0006, +0.0019] / [+0.0006, +0.0019] | +0.0026 [+0.0019, +0.0034] / [+0.0018, +0.0033] | +0.0030 [+0.0022, +0.0039] / [+0.0021, +0.0039] | +0.0013 [+0.0006, +0.0020] / [+0.0006, +0.0020] | +0.0683 [+0.0659, +0.0708] / [+0.0656, +0.0711] |
| xPass destination known (89,285) | 0.2122 | +0.0008 [+0.0000, +0.0017] / [-0.0000, +0.0016] | +0.0002 [-0.0004, +0.0008] / [-0.0004, +0.0009] | +0.0002 [-0.0004, +0.0009] / [-0.0005, +0.0009] | +0.0012 [+0.0003, +0.0021] / [+0.0003, +0.0021] | -0.0000 [-0.0006, +0.0006] / [-0.0006, +0.0006] | +0.0078 [+0.0066, +0.0088] / [+0.0067, +0.0088] |

(Deltas vs EVENT in nats, per-shot or per-pass CI / match-clustered CI; on this 3-fold subset
even the 360 oracle is not significant for xG, +0.0017 vs +0.0029 on the full five folds, and
the shot-frame oracle is +0.0061 [+0.0032, +0.0090] / [+0.0031, +0.0090].) The better
imputations change nothing for xG: the GRU columns are worth -0.0008 [-0.0026, +0.0010] /
[-0.0025, +0.0010] against the LightGBM columns of the same seven quantities. For pre-instant
pass difficulty they double the small gain (+0.0013 [+0.0006, +0.0020] / [+0.0006, +0.0020]
over the LightGBM columns; +0.0010 [+0.0004, +0.0016] / [+0.0005, +0.0015] on top of the full
stage-03 block), to about 4% of the 360-oracle gap; with the destination known the extra
history adds nothing. The programme's xG null therefore holds for the better student too, and
the one place imputed state pays (pre-instant xPass) scales with imputation quality but stays
at the level of a few thousandths of a nat.

## 4. What generalises across the two sports, and what does not

**Generalises.**

* *Team-level state is recoverable, individual within-play quantities are not.* Both chains
  recover the coarse collective picture before the instant (NFL box / safeties / spread at R2
  0.4-0.5, soccer block depth / defensive line at 0.95-0.96 of which 0.10 is beyond ball
  location, nearest opponent 0.67) and fail on the quantities that decide the outcome (NFL
  separation, pressure, time to throw < 0.05 before the release; soccer cone count 0.16, cone
  distance 0.42, keeper distance 0.48). In both, what is "recovered" is largely a
  deterministic function of location and phase (NFL: shotgun / formation / down-distance;
  soccer: ball x plus the visible-area truncation), which is exactly why it has no downstream
  value: the downstream model already sees those inputs.
* *After-the-fact skill is outcome re-encoding.* NFL F1 within-play students do not beat the
  per-outcome-class mean on pressure and barely on QB distance; soccer E2a students read the
  realised pass duration and carry length. Both projects had to re-derive fair baselines
  (NFL 04c, soccer 03 xPass) after an imputation looked useful only because the baseline lacked
  a field the student had.
* *The payoff null and its shape.* Imputed state adds nothing to outcome models in domain and
  out of domain (NFL -0.0009 / -0.0010 nats; soccer -0.0003 / +0.0000), whether the imputations
  come from trees or from the better sequence student (soccer xG +0.0000 with the GRU columns),
  and nothing forward either (NFL next-game receiving yards / targets); the pre-instant oracles
  are worth little (+0.0027 nats NFL pre-snap; +0.0029 soccer 360 frame; +0.0078 shot frame
  with all coded players); the big oracle gains are post-release quantities (+0.058 nats NFL
  completion, +15.7 yd^2 run yards; +0.070 soccer pre-instant xPass with end-relative lanes).
  In both sports the only privileged input that beats the event baseline clearly and is
  available at scale is charted, after-the-fact and vendor-supplied (NGS `was_pressure` /
  `time_to_throw`, StatsBomb `shot.freeze_frame` / xG).
* *The students transfer as measurements* across seasons (NFL 2017 -> 2018 box R2 0.54 vs 0.56;
  z-shifts <= 0.16 to 2022) and across eras / competitions / genders (soccer 360 tournaments ->
  2015/16 leagues with R2 vs the shot frame unchanged; men <-> women within 0.02 on most
  targets), because they encode location-and-phase regularities that are stable. That is also
  why they carry no new information.

**Sport-specific.**

* *Error growth with time since informative events.* Soccer is a continuous flow: nearest-
  opponent error doubles from 1.1 to 2.2 yd as the opponent's last defensive action recedes
  beyond three events, counter_on log-loss halves after ten events, and the sequence model's
  advantage grows with possession age. The NFL pre-snap state is a discrete start state
  re-initialised every play, so there is no within-possession decay to exploit; what carries
  across plays is team tendency (worth 0.14 nats for the offense grouping, R2 0.12 -> 0.24 for
  cushion), and within-play movement is unobservable before the release regardless of history.
* *Value of the participation step.* NFL: knowing the true offense grouping is worth 0.137
  nats for defense personnel and +0.06 R2 for the box count, the imputed grouping 0.015 nats
  and nothing for the box, and personnel / formation lifts the pre-snap students (n_backfield
  0.14 -> 0.49) without any play-level payoff (PBP+PERS and stage d all null). Chaining the
  *imputed* grouping into the alignment students (§2.4, `F0I`) recovers none of that lift
  (box 0.434 -> 0.442 vs 0.509 charted; 2 of 12 targets improve by clustered CI, the largest
  qb_depth 0.868 -> 0.874), so the participation -> alignment chain with event-only inputs is
  a null at both of its links; in the untracked seasons `PBP+IMP` reads charted personnel
  through its F0P students and only `PBP+IMP_F0` is event-only. Soccer lineups are observed,
  so the analogous step (who is on the pitch) was never a bottleneck; the analogue of
  "personnel" is the possession phase, which the E1 -> E2 window features capture.
* *Where the small significant gain lives.* Only soccer xPass shows a reproducible imputed-
  state gain (+0.0016 pre-instant in domain, +0.0019 in 2015/16, +0.0013 with the destination
  known in 2015/16), about a fiftieth of the oracle gap, rising to +0.0026 (about 4%) with the
  sequence student's imputations; the NFL has no analogue at any task, including the forward
  receiving-props test.
* *Sequence models.* The raw 20-event history adds 0.01-0.06 skill in soccer when a GRU
  encodes it (a flattened block through a scikit-learn MLP lost to trees, and trees with the
  raw block gain 0.01 R2), and that extra skill reaches the payoff only for pre-instant pass
  difficulty (+0.0013 nats over the LightGBM imputations of the same quantities, nothing for
  xG); the NFL chain never modelled the play sequence within a drive, so this is untested
  there.

## 5. Implications for the betting and portfolio goals

Honest effect sizes first. The quantities that an event-only model could learn from tracking
supervision move play-level and shot-level probabilities by less than a thousandth of a nat of
log-loss in either sport, with intervals that include zero; the two soccer xPass gains that are
significant are 0.0013-0.0019 nats on models whose log-loss is 0.25-0.41. Even the unattainable
ceilings are modest: perfect pre-snap structure is worth +0.0027 nats on NFL completion and the
complete shot freeze frame +0.006-0.008 on xG (about 2-3% of the model's log-loss). Nothing in
this programme produces a probability edge of the size that survives a bookmaker's margin on a
per-play, per-shot or per-match price, and the portion of the oracle gain that is large (post-
release NFL state, end-relative soccer pass lanes) is by construction unavailable before the
event settles. For pre-match markets the picture is the same one level up: an xG model built
from events is within 0.013 nats of StatsBomb's own xG and within 0.006 of a freeze-frame model,
and the imputed state closes none of that gap, so match totals or team-strength ratings built
on "imputed 360 state" would be numerically indistinguishable from those built on plain events.

For the portfolio the useful conclusions are negative and specific. Learned-state features are
not a source of edge and should not receive further modelling budget; what is worth paying for
is data that is privileged and available at scale (StatsBomb freeze frames and xG for shots, NGS
charting for pass plays), and even that is after-the-fact and therefore only relevant to
settlement-time analytics, in-play models with a latency that tolerates charting delays, or
player-level ratings. The one place the imputation helps is descriptive: receiver-week
separation R2 0.32 -> 0.37 with an at-release student (+0.055 [+0.038, +0.072]) and -> 0.41 after
the fact, and season-level team rankings of defensive depth that agree with simple event
proxies (Spearman 0.79) but not beyond them. For player props the forward test (§2.5) is the
one that matters and it is measured, not assumed: next-game receiving yards and targets are
predicted no better with the imputed separation history than from the event history alone
(the single interval that excludes zero, +3.1 yd^2 on all receiver-games, is a +0.003 change in
R2 that reverses on the props-eligible receivers), and the true NGS separation history does no
better; the event history itself is what carries the prior (+84 yd^2 over the season-to-date
mean). The
evaluation machinery built here (game / match-grouped splits, out-of-fold stacking, clustered
intervals, refit-noise floors, leakage tiers) is the reusable asset, and it is precisely what
exposed the two apparent gains (NFL receiver separation "doubling", soccer xPass +0.009) as
information-set mismatches.

## 6. Limitations and next steps

* **NFL sample.** 91 games of one season fragment: team encodings and tendencies rest on at
  most five prior games per team, the k-fold protocol lets a team's other 2017 games inform its
  encoding (the forward split and the 2018 external check are the honest numbers), and
  differences below ~0.003 nats are refit noise (three-seed spreads 0.002-0.003 nats). Targets
  are derived definitions that agree with the official charting on 71% (box) / 80% (rushers) /
  96.5% (pressure) of plays. `qb_hit` is a charted post-play flag inside the "at-release" set
  (present on every side of every comparison). The highlight OOD check has n = 126 and rules
  out gross failure only.
* **Soccer labels are not tracking.** 360 frames show the broadcast visible area (40% of
  own-third frames are reliable), so shape targets are visibility-biased; the deep-block
  threshold and counter definition are choices of stage 02; f_under_pressure is partly
  post-instant on carries; the training-fold imputations come from students that saw the test
  fold's 360 labels (standard second-order stacking coupling); stage 05 trained 3 of 5 folds;
  xPass uses a fixed 40% subsample and one seed; StatsBomb xG may have been trained on these
  matches. 2015/16 is older retrospective collection, so its domain shift mixes era,
  competition type and coding conventions; WC 2018 is the one domain where the keeper student
  fails (R2 -0.16, bias +1.0 yd).
* **Statistical power.** ~10k shots / 1k goals in the 360 matches means 0.002-0.005 nats is
  noise at shot level; two auto-generated "significant" labels in stage 03 (distillation
  alpha 0.5, 50% learning-curve point) sit at the boundary and should be read as null.
* **Open items carried from the reviews (not fixed here).** (a) One garbage play in
  `plays_tracked.parquet` (gameId 2017091700 / playId 180, an injury stoppage with
  n_deep_safeties = 11 and qb_depth = -46.4) is still in every NFL 03 / 04 / 06 table
  unflagged; it is 1 of 11,518 rows and moves no reported number, but a regression on qb_depth
  sees a 40-yd outlier. (b) 19 `qb_spike` plays sit inside the NFL 03 pass subset (n = 6,696);
  the NFL 04 tasks exclude them. (c) The soccer 02 transfer BSS of deep_block / counter_on uses
  the source-domain base rate as reference (§3.2 prints the uncorrected cells). (d) The
  workflow rule "every module has a test file" was violated for `nfl/imputation.py`,
  `nfl/participation.py`, `soccer/imputation.py` and `soccer/build_tables.py`; this pass adds
  smoke tests of their pure helpers (outcome classes, multiclass metrics, row thinning, target
  resolution, output schema), not driver runs. (e) Nothing of phase 2 is committed: the
  untracked stage files and reports and 9 modified tracked files under
  `research/privileged_tracking/` (125 untracked and 9 modified after this pass) await a commit
  on the `claude/*` branch, so the reproduction section assumes the working tree.
* **Not tried.** A possession-level tactical block label (rather than the instantaneous
  def_line), a within-drive sequence model for the NFL, transformer variants of the soccer
  sequence student (implemented, not run under the CPU budget), tuning of the sequence student
  (single configuration), `F0I` students carried through the NFL 04 payoff (the chain stops at
  the alignment step, where it is already null), prior-season carry-over and market lines in
  the forward props test (no lines are on disk), and imputation of the shot freeze frame
  itself from events (the one privileged block that carries downstream value and exists in
  every StatsBomb match, so a student for it could be trained on 41k shots rather than 10k).
* **If one more experiment were funded,** it should be the last item: an event-only student of
  `sff_*` trained on all 51k shots with freeze frames, evaluated exactly as here, because the
  freeze-frame oracle is the only pre-instant privileged block with a consistent +0.006-0.008
  nat payoff. Everything else in this programme argues against further tracking-to-event
  imputation for outcome prediction.

## 7. Reproduction

Run from the repository root with `PRIV_DATA_DIR` pointing at the raw data (default
`data/raw/privileged/`). Every stage caches under `processed/<sport>/` and writes its report and
`<sport>_<NN>_*.parquet` tables into this directory. Each invocation below was kept under ~25
minutes on 2 CPU threads with two exceptions: the soccer 02 chain is split into cached
invocations of < 14 minutes each (~85 minutes in total) and the soccer 05 sequence student ran
33 minutes for three folds in one invocation (a 10-minute-per-fold training cap; run it per
fold with `--folds k` to stay under the budget).

```shell
# tests (180 passed: 153 before the synthesis, +5 for results_index, +22 in the fix pass)
python -m pytest research/privileged_tracking/tests -q

# NFL chain
python -m research.privileged_tracking.nfl.build_tables                       # 01, 0.8 min
python -m research.privileged_tracking.nfl.participation --stages a,b,c --n-jobs 2   # 02, 15.5 min
python -m research.privileged_tracking.nfl.imputation                         # 03, 3.7 min (incl. 2018 application)
python -m research.privileged_tracking.nfl.apply_student --season 2018 --feature-sets F0,F0P,F1
python -m research.privileged_tracking.nfl.ood_highlights --max-plays 150     # 03b, 0.3 min
python -m research.privileged_tracking.nfl.payoff --stages a,b,c,d --n-jobs 2 # 04, ~3.5 min with cached season applications (first run applies the students to 2018-2022)
python -m research.privileged_tracking.nfl.imputation_chain --n-jobs 2         # 06, 1.1 min (participation -> alignment chain, F0I / F0PI)
python -m research.privileged_tracking.nfl.props_forward --n-jobs 2            # 07, 1 min (forward receiving-props test)

# soccer chain
python -m research.privileged_tracking.soccer.build_tables --workers 2 --frac 0.25          # 01, 12 min
python -m research.privileged_tracking.soccer.imputation --stage cv --targets shape         # 02: then --targets ball, pass; then
python -m research.privileged_tracking.soccer.imputation --stage transfer                   #     transfer, forward, mlp, final, report (~85 min total, each < 14 min)
python -m research.privileged_tracking.soccer.apply_student                                 # 02: students -> events_no360
python -m research.privileged_tracking.soccer.payoff --stage all                            # 03, 19 min (stages cached individually)
python -m research.privileged_tracking.soccer.transfer --stage shots                        # 04: then xg, passes, xpass, report (8 / 5 / 0.5 / 2.5 / 1.5 min)
python -m research.privileged_tracking.soccer.sequence_student --variants seq20,seq1,seq0 --folds 0,1,2   # 05, 33 min training + 5.5 min report
python -m research.privileged_tracking.soccer.payoff_seq --stage all --n-jobs 2             # 06, 0.5 + 1.6 + 1.2 min (xg, xpass, report)

# synthesis
python -m research.privileged_tracking.synthesis.results_index                              # results_index.parquet
```

Reports in this directory, in reading order: `phase1_handoffs.md` (every phase-1 handoff and
the open review issues, with the NFL 03 resolution note), `nfl_01_build.md`,
`nfl_02_participation.md`, `nfl_03_imputation.md`, `nfl_03_ood_highlights.md`,
`nfl_04_payoff.md`, `nfl_05_audit_fix.md`, `nfl_06_participation_chain.md`,
`nfl_07_props_forward.md`, `soccer_01_build.md`, `soccer_02_imputation.md`,
`soccer_03_payoff.md`, `soccer_04_transfer.md`, `soccer_05_sequence.md`,
`soccer_06_seq_payoff.md`. The parquet tables next to each report are the source of every number
above; `results_index.parquet` maps each headline number to its table and row.
