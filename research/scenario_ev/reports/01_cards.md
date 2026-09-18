# 01 Cards: can a card model be +EV against a bookmaker's rolling mean?

Stage 01 of the scenario-EV phase. Code: `research/scenario_ev/cards.py` (+ shared
machinery in `research/scenario_ev/common.py`), tests in `research/scenario_ev/tests/`,
result tables `research/scenario_ev/reports/01_cards_*.parquet`.

```
cd /home/user/geo-model
python -m research.scenario_ev.cards --stage build --force  # 15 s, builds all five intermediates
python -m research.scenario_ev.cards --stage a              # 2-4 min    player card props
python -m research.scenario_ev.cards --stage b              # 5-12 min   foul -> card mechanism
python -m research.scenario_ev.cards --stage c              # 7-15 min   team card totals + the goals anchor
python -m pytest research/scenario_ev/tests/test_common.py research/scenario_ev/tests/test_cards.py -q   # 39 tests
python -m pytest research/privileged_tracking/tests -q      # 180 tests, unchanged by this stage
```

## Verdict in one paragraph

Targeting has **no room** for cards. On every scenario tested, at player level and at team
level, a specialist fitted only on the scenario's rows is **worse** on held-out scenario rows
than a generalist fitted on everything, and the gap widens as the scenario narrows. The
event-only models *do* beat the rolling-mean book proxy — by +0.0064 nats on "player is
carded" (n = 15,148 confirmation appearances) and +0.0067 nats at the 3.5 match-cards line
(n = 64,550 confirmation matches) — and simulated betting against that proxy returns
+26.5% ROI on player props and +30.5% on team totals. That ROI is an artefact of the strawman.
Calibrated on the one count market where a real line is observed, **Bet365 beats the identical
rolling-mean proxy on total goals by +0.0111 nats [+0.0101, +0.0122], and turns a +17.7%
simulated ROI into −12.9% [−16.7%, −9.2%] when the same model bets into its real prices** — a
30.6-point ROI haircut. Our entire margin over the rolling mean, on both card markets, is
**smaller than the margin a real bookmaker already holds over that same rolling mean on
goals.** After the haircut no card ROI has an interval clear of zero (point estimates −0.001 to
+0.038, lower bounds −0.11 to −0.05), and the book-strength ladder puts break-even at a book
only two-thirds as sharp as a real one. The matchup channel that motivated the stage (foul-prone full-back vs
high-volume dribbler under a card-happy referee) adds nothing over the plain event-only model
(+0.0006 nats, inside the 0.0012-nat three-seed refit spread), and part (b) shows why: fouls on
a dribbling opponent *are* more carded overall (OR 1.21 [1.08, 1.40]) but **less** so in the
defending third, exactly where a full-back's duels happen — the interaction runs the wrong way.

This is a null result on the betting question and a positive result on two smaller ones: the
event-only model really does beat a rolling mean, and the foul -> card channel is real and
strongly location- and time-driven.

## 1. Protocol

Everything below obeys the phase protocol, and the parts that are easy to assert and hard to
verify were verified.

**Discovery / confirmation.** Matches are cut chronologically. Every scenario definition,
tercile threshold, feature set, boosting-iteration count, calibration map, dispersion
parameter and bet threshold is chosen on discovery. Confirmation is scored once and its
numbers are the headline. Both are printed side by side throughout, and discovery numbers are
labelled exploratory because the same rows chose the knobs.

**The gate.** For each target: a *generalist* fitted on all training rows and a *specialist*
fitted only on the scenario's training rows, both scored on the *same* held-out scenario rows,
first by 5-fold group CV inside discovery and then once on confirmation. A positive
`delta_specialist_minus_generalist` means the specialist is better. The reverse direction (the
specialist scored off-scenario) is reported too, as a sanity check that the specialist is a
real model and not a broken one.

**The book proxy.** A shrunk rolling mean over strictly prior matches, recalibrated on
discovery (logistic for binaries, a two-parameter Poisson GLM plus a fitted negative-binomial
dispersion for counts). The recalibration matters: an *un*calibrated strawman is beaten on
scale alone, which would have manufactured an edge out of nothing.

**Hygiene, verified not asserted.** `common.prior_expanding` builds every rolling feature from
strictly earlier rows; `common.audit_strictly_prior` re-derives a random sample by brute-force
filtering and `cards.audit_player_table` re-derives the three headline priors (player card
rate, team card rate, referee card rate) directly from the raw StatsBomb tables for 120 random
player-matches:

| feature | max_abs_error | n_checked |
|---|---|---|
| pl_card_rate | 0.0 | 120 |
| own_tp_yellow | 0.0 | 120 |
| ref_card_rate | 0.0 | 120 |

Group-level backstops (the position-group card rate, the division card rate) are accumulated
over *(group, date)* aggregates rather than rows, so two rows sharing a date can never enter
one another's prior. The minutes a player actually played in the match being predicted are
**not** a feature (a sent-off player plays fewer minutes); only the expected minutes from his
prior appearances are. Lineup facts known when a prop is priced — starter, starting position,
opponent, competition, home/away — are used.

## 2. (a) Player card props

### 2.1 Universe

StatsBomb player-appearances, restricted to the four 2015/16 club seasons where a rolling
history is meaningful, outfield players only, and at least 3 strictly-prior appearances (a
prop market needs a history to price off).

| | |
|---|---|
| player-appearances in the full StatsBomb table | 59,532 |
| modelling universe (4 club seasons, outfield, >= 3 prior apps) | 33,329 over 1,398 matches, 1,770 players |
| discovery | 18,181 rows, 791 matches, 2015-08-28 to 2016-02-02 |
| confirmation | 15,148 rows, 607 matches, 2016-02-03 to 2016-05-17 |
| carded rate (yellow or second yellow or red) | 0.1761 discovery, 0.1679 confirmation |
| median prior appearances per row | 13 |
| rows with a known referee | 96.5% |
| rows with an identified direct opponent | 97.1% |

Target `y_carded` = yellow OR second yellow OR red; `y_yellow` (yellow or second yellow) is
reported separately and behaves identically.

### 2.2 The models

* **P0 book proxy** — `lambda = shrunk_card_rate_per_90 x (expected_minutes / 90) x team_factor
  x referee_factor`, `p = 1 - exp(-lambda)`, then a discovery-fitted logistic recalibration.
  The player rate is empirical-Bayes shrunk to his position-group rate; the referee factor
  comes from that referee's strictly prior matches.
* **P1 event-only** — P0's inputs plus prior foul rates (total, by pitch side, defensive
  third), prior fouls won, dribbled-past, tackles, pressures, dribbles, position, starter,
  home/away, competition, match week and the team's and opponent's prior foul / card / dribble
  / possession / points rates. 53 features (P2 adds 10, P3 adds 8).
* **P2 matchup** — P1 plus the opponent's prior dribble and fouls-won volume on the *mirrored*
  flank, the identified direct opposing starter's prior dribble and fouls-won rates, and four
  explicit referee x matchup and foul x matchup products. (A right back faces the opponent's
  left-sided attacker: positions are named in each team's own attacking frame, so the physical
  mirror is left <-> right.)
* **P3 state** — P2 plus the programme's imputed defensive-state aggregates (nearest-opponent
  distance, opponents within 5 m, block depth, counter-on) for both teams, taken from
  `imputed_oof` / `imputed_no360` (`E2` student), aggregated per team-match over the events in
  which the *other* team has possession, then rolled forward over strictly prior matches.

Boosting iterations are chosen per feature set on discovery OOF log-loss (30/60/100/200/350;
60-100 wins everywhere, so the models are small) and every model's probabilities are
recalibrated by a logistic map fitted on the discovery OOF predictions — the same treatment P0
gets, so the comparison is about information, not about calibration. Raw (uncalibrated)
variants are in `01_cards_a_metrics.parquet` as `P1_raw` / `P2_raw` / `P3_raw` and differ by
less than 0.0004 nats.

### 2.3 Results

`delta_vs_P0` is `log_loss(P0) - log_loss(model)`, so positive means better; the interval is a
match-clustered bootstrap.

**Confirmation (headline), target `y_carded`, n = 15,148:**

| model | log_loss | brier | auc | delta_vs_P0 | 95% CI | skill vs P0 |
|---|---|---|---|---|---|---|
| climatology | 0.4528 | 0.1398 | 0.500 | -0.0161 | [-0.0191, -0.0131] | -3.69% |
| P0 book proxy | 0.4367 | 0.1353 | 0.636 | 0 | — | 0 |
| P1 event-only | 0.4309 | 0.1338 | 0.658 | **+0.0058** | [+0.0036, +0.0080] | +1.33% |
| P2 matchup | 0.4304 | 0.1336 | 0.661 | **+0.0064** | [+0.0042, +0.0084] | +1.46% |
| P3 + imputed state | 0.4302 | 0.1335 | 0.661 | +0.0065 | [+0.0043, +0.0085] | +1.49% |

**Discovery OOF (exploratory), n = 18,181:** P0 0.4522, P1 0.4435 (+0.0088 [+0.0067, +0.0110]),
P2 0.4435 (+0.0087), P3 0.4429 (+0.0093). `y_yellow` is the same picture
(P0 0.4303 -> P1 0.4235, +0.0067 [+0.0047, +0.0087] on confirmation).

**Three-seed refit spread** (same pipeline, seeds 20260918/19/20, confirmation log-loss):
P1 0.4309 / 0.4297 / 0.4306, P2 0.4304 / 0.4305 / 0.4302 — a spread of **0.0012 nats**.

Read against that floor:

* **P0 -> P1 is real**: +0.0058, about 5x the refit spread, interval clear of zero. Prior foul
  rates and team context beat a shrunk card rate. Cards are rare (a player with 20
  appearances has ~3); fouls are not (~25), so the foul channel is a better-measured proxy
  for the same propensity.
* **P1 -> P2 is not**: +0.0006, *inside* the refit spread. The matchup features — the
  hypothesis this stage was built to test — add nothing detectable.
* **P2 -> P3 is not**: +0.0001 [-0.0007, +0.0009] on the like-for-like subset. Consistent with
  the completed programme's finding that imputed state adds nothing to models that already
  read the same events.

**Imputed-state coverage.** 3,947 of 4,180 team-matches (94.4%) have at least one strictly
prior match with usable imputations; within the modelling universe every row has a state prior
for both teams (median 21 prior matches), so the P3-vs-P2 comparison above is already
like-for-like on the full universe, not a subset. `01_cards_a_p3_subset.parquet` repeats it on
the explicit `state_available` mask and gets the same numbers.

### 2.4 The gate: does targeting have room?

Scenario pre-registered on discovery: full-back or wing-back, opponent's mirrored-flank prior
dribble volume in the **top tercile** (cut 6.956 dribbles/match), referee card rate in the
**top tercile** (cut 0.1757 cards per appearance). Two looser rungs are reported because the
narrow scenario is small. Feature set P2, LightGBM at the discovery-selected 60 iterations.

| scenario | n disc | n conf | conf base rate |
|---|---|---|---|
| fullback_only | 3,194 | 2,695 | 0.188 |
| fullback x top-tercile flank | 1,065 | 1,000 | 0.190 |
| fullback x flank x top-tercile referee | 348 | 400 | 0.225 |

| where | scenario | n | generalist LL | specialist LL | delta (spec - gen) | 95% CI |
|---|---|---|---|---|---|---|
| discovery CV | fullback_only | 3,194 | 0.4900 | 0.4956 | **-0.0056** | [-0.0096, -0.0019] |
| confirmation | fullback_only | 2,695 | 0.4738 | 0.4751 | -0.0014 | [-0.0055, +0.0030] |
| confirmation, off-scenario | fullback_only | 12,453 | 0.4212 | 0.4381 | -0.0169 | [-0.0195, -0.0142] |
| discovery CV | fullback x flank | 1,065 | 0.5183 | 0.5162 | +0.0022 | [-0.0084, +0.0129] |
| confirmation | fullback x flank | 1,000 | 0.4793 | 0.4879 | -0.0086 | [-0.0205, +0.0036] |
| confirmation, off-scenario | fullback x flank | 14,148 | 0.4271 | 0.4506 | -0.0235 | [-0.0264, -0.0204] |
| discovery CV | fullback x flank x ref | 348 | 0.5852 | 0.5851 | +0.0001 | [-0.0185, +0.0195] |
| confirmation | fullback x flank x ref | 400 | 0.5240 | 0.5448 | **-0.0208** | [-0.0395, -0.0011] |
| confirmation, off-scenario | fullback x flank x ref | 14,748 | 0.4280 | 0.4761 | -0.0481 | [-0.0529, -0.0435] |

**Targeting has no room for player card props.** The specialist is never better on
confirmation; on the narrowest scenario it is significantly worse, and the two discovery-CV
rungs where it draws level (+0.0022, +0.0001) have intervals straddling zero and are the
multiplicity-inflated numbers by construction. The off-scenario rows confirm the specialist is
a working model that has simply been starved of data: it degrades smoothly as its training set
shrinks (3,194 -> 1,065 -> 348 rows), which is the ordinary bias-variance story, not a
scenario-specific signal the generalist was missing.

### 2.5 Calibration and where the model disagrees with the proxy

Both P0 and P2 are well calibrated on confirmation (full 10-bin tables in
`01_cards_a_calibration.parquet`). The closing-line-value style check bins confirmation rows by
`p_model - p_book` and asks who is closer to the truth:

| bin | n | diff range | mean p_model | mean p_book | observed | LL model | LL book | model better by |
|---|---|---|---|---|---|---|---|---|
| 0 (model much lower) | 3,030 | -0.237 .. -0.052 | 0.146 | 0.229 | **0.153** | 0.4102 | 0.4297 | +0.0194 |
| 1 | 3,030 | -0.052 .. -0.025 | 0.137 | 0.175 | 0.146 | 0.3930 | 0.3980 | +0.0050 |
| 2 | 3,030 | -0.025 .. -0.002 | 0.146 | 0.160 | 0.152 | 0.4041 | 0.4050 | +0.0009 |
| 3 | 3,029 | -0.002 .. +0.026 | 0.164 | 0.152 | 0.164 | 0.4230 | 0.4232 | +0.0002 |
| 4 (model much higher) | 3,029 | +0.026 .. +0.207 | 0.231 | 0.175 | **0.224** | 0.5215 | 0.5278 | +0.0063 |

The model is closer to the truth in both tails, and most of its advantage is in the bin where
it says a player is *safer* than the proxy does — the proxy's biggest error is over-pricing
card-prone-looking players. That is genuine information, and it is what the betting simulation
below converts into a fictitious edge.

### 2.6 Betting simulation ("player to be carded", a 0.5 line)

Flat stakes, the proxy's fair probability marked up to `1 + hold` on both sides, bet the side
with the larger expected value when that EV exceeds a threshold chosen on discovery
(grid 0 .. 0.40).

| book | model | hold | threshold | where | n bets | bet rate | mean edge | ROI | 95% CI | mean p_model | mean p_book | realised | mean odds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 proxy | P2 | 0.04 | 0.40 | discovery | 1,649 | 9.1% | 0.581 | +0.478 | [+0.345, +0.612] | 0.254 | 0.156 | 0.233 | 7.00 |
| P0 proxy | P2 | 0.04 | 0.40 | **confirmation** | 822 | 5.4% | 0.593 | **+0.239** | [+0.035, +0.455] | 0.205 | 0.126 | 0.159 | 9.29 |
| P0 proxy | P2 | 0.06 | 0.40 | **confirmation** | 722 | 4.8% | 0.587 | **+0.265** | [+0.062, +0.485] | 0.203 | 0.123 | 0.165 | 9.30 |
| P0 proxy | P2 | 0.08 | 0.40 | **confirmation** | 626 | 4.1% | 0.584 | **+0.260** | [+0.033, +0.492] | 0.202 | 0.120 | 0.163 | 9.31 |
| P1 as the book | P2 | 0.06 | 0.20 | **confirmation** | 722 | 4.8% | 0.280 | +0.023 | [-0.175, +0.224] | 0.183 | 0.135 | 0.154 | 8.51 |

Note what the "realised" column says: on the bets placed, the model itself is *over*-confident
(it says 0.203, the truth is 0.165) — but the proxy says 0.123 and the mean price is 9.30, so
the bet still wins. The entire return comes from the proxy being badly wrong on long shots, not
from the model being right. (ROI is `mean(win x odds) - 1` over the placed bets, not
`mean(win) x mean(odds) - 1`: wins and prices are negatively correlated here, which is why
0.265 and not 0.165 x 9.30 - 1.)

**How sharp may the book be before this dies?** A synthetic book interpolated on the logit
scale between the proxy (lam = 0) and our own model (lam = 1). `book_gain` is how much that
book beats the proxy on confirmation, in nats — directly comparable to the goals-market
measurement in section 4.2.

| lam | book gain over proxy (nats) | hold | n bets | ROI | 95% CI |
|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.06 | 722 | +0.265 | [+0.062, +0.485] |
| 0.25 | 0.0034 | 0.06 | 229 | +0.138 | [-0.218, +0.542] |
| 0.50 | 0.0055 | 0.06 | 78 | +0.219 | [-0.423, +0.848] |
| 0.75 | 0.0065 | 0.06 | 89 | -0.112 | [-0.577, +0.405] |
| 1.00 | 0.0064 | 0.06 | 0 | — | — |

A book only **0.0034 nats** better than the rolling mean — less than a third of the 0.0111 nats
that Bet365 actually holds over the same construction on goals — is already enough for the
interval to include zero. Our whole margin over the proxy is 0.0064 nats, so against a
real-book-strength opponent there is nothing left to bet.

### Verdict (a)

* Does targeting have room? **No.** The specialist never beats the generalist on confirmation
  scenario rows; on the tightest, pre-registered scenario it is worse by 0.0208 nats
  [0.0011, 0.0395].
* Does any model beat the proxy by more than the haircut? **No.** P2 beats the proxy by
  +0.0064 nats, less than the +0.0111 nats a real book holds over the same proxy on goals. The
  +26.5% simulated ROI carries a 30.6-point haircut (section 4.2), which takes it negative
  outright — and that is before the extra fact that player-prop holds are far above 6%.
* What would a bet rule look like? Bet "player to be carded" when the model's EV at the
  proxy's price exceeds 40%: about 5% of appearances, mean modelled edge 59%, realised
  +26.5% ROI against the strawman. Against any plausible real book: nothing.

## 3. (b) Foul -> card: the mechanism behind (a)

60,511 StatsBomb fouls; the modelling universe is the same four club seasons (45,517 fouls,
27,524 discovery / 17,993 confirmation), with the remaining 14,994 (tournaments, other
seasons) held out entirely as an out-of-domain check. 12.9% of fouls are carded (12.1% yellow,
0.77% red or second yellow); 63% are fouls on a dribbling opponent.

Features are all pre-foul: pitch location in the fouling team's attacking frame (own goal at
x = 0), minute and period, score state reconstructed from the running score in the programme's
event tables, whether the fouled player was dribbling, fouled and fouler position lines, the
team's and the player's foul and card counts *so far in this match*, the referee's prior
cards-per-foul rate, and the fouler's prior season card and foul rates.

| where | model | n | base rate | log_loss | brier | auc |
|---|---|---|---|---|---|---|
| discovery OOF | foul LGBM | 27,524 | 0.1302 | 0.3594 | 0.1066 | 0.688 |
| discovery OOF | base rate | 27,524 | 0.1302 | 0.3867 | 0.1132 | 0.500 |
| **confirmation** | **foul LGBM** | 17,993 | 0.1264 | **0.3509** | 0.1033 | **0.695** |
| confirmation | base rate | 17,993 | 0.1264 | 0.3796 | 0.1105 | 0.500 |
| out-of-domain | foul LGBM | 14,994 | 0.1012 | 0.3150 | 0.0889 | 0.672 |
| out-of-domain | base rate | 14,994 | 0.1012 | 0.3316 | 0.0918 | 0.500 |

**The channel is real**: 7.6% log-loss skill over the base rate on confirmation, and it
transfers to tournaments and other seasons it never saw (5.0% skill, AUC 0.672).

### 3.1 Effect sizes

Logistic regression, continuous terms standardised so odds ratios are per standard deviation,
200-replicate match-clustered bootstrap. Confirmation split (the full table, including
discovery, is `01_cards_b_odds_ratios.parquet`).

| term | odds ratio | 95% CI | reading |
|---|---|---|---|
| distance to own goal (per SD) | 0.466 | [0.329, 0.605] | the single strongest term: fouls near your own goal are far more carded |
| minute (per SD) | 1.334 | [1.225, 1.457] | cards accumulate through the match |
| dangerous play | 1.951 | [1.256, 2.758] | |
| penalty conceded | 1.458 | [1.010, 2.024] | |
| **fouled player was dribbling** | **1.211** | **[1.082, 1.400]** | the matchup channel, in the right direction |
| **dribbling x own third** | **0.827** | **[0.631, 1.050]** | but it *shrinks* where a full-back defends |
| own third | 1.289 | [1.031, 1.814] | |
| referee cards-per-foul (per SD) | 1.088 | [1.032, 1.146] | referee tendency is real but small |
| fouler's prior card rate (per SD) | 1.149 | [1.092, 1.215] | |
| fouler's prior fouls per 90 (per SD) | 0.934 | [0.892, 0.978] | high-foul players are carded *less per foul* |
| player's cards so far this match | 0.808 | [0.757, 0.852] | already-booked players are carded less afterwards |
| player's fouls so far this match | 1.079 | [1.022, 1.140] | |
| team's cards so far this match | 0.901 | [0.841, 0.944] | |
| handball | 0.221 | [0.129, 0.381] | |
| fouler is a forward / midfielder (vs defender) | 0.697 / 0.757 | [0.585, 0.818] / [0.673, 0.835] | |

Raw rates say the same thing without any model:

| zone | dribbling | n | card rate | 95% CI | where |
|---|---|---|---|---|---|
| final third | no | 2,575 | 0.0687 | [0.0595, 0.0776] | confirmation |
| final third | yes | 2,181 | 0.0779 | [0.0665, 0.0893] | confirmation |
| middle third | no | 3,223 | 0.1049 | [0.0941, 0.1157] | confirmation |
| middle third | yes | 6,454 | **0.1277** | [0.1200, 0.1353] | confirmation |
| own third | no | 908 | **0.2357** | [0.2073, 0.2648] | confirmation |
| own third | yes | 2,652 | **0.2081** | [0.1924, 0.2247] | confirmation |

And by minute and by accumulated team fouls (confirmation):

| minute bucket | n | card rate | 95% CI |
|---|---|---|---|
| 0-15 | 2,612 | 0.0532 | [0.0446, 0.0620] |
| 15-30 | 2,823 | 0.1130 | [0.1004, 0.1259] |
| 30-45 | 2,922 | 0.1348 | [0.1220, 0.1483] |
| 45-60 | 3,087 | 0.1328 | [0.1219, 0.1447] |
| 60-75 | 2,824 | 0.1611 | [0.1479, 0.1746] |
| 75+ | 3,725 | 0.1498 | [0.1384, 0.1614] |

| team fouls so far | n | card rate | 95% CI |
|---|---|---|---|
| 0-2 | 3,642 | 0.0623 | [0.0546, 0.0703] |
| 3-5 | 3,630 | 0.1262 | [0.1152, 0.1374] |
| 6-8 | 3,502 | 0.1442 | [0.1323, 0.1558] |
| 9-11 | 3,007 | 0.1463 | [0.1336, 0.1592] |
| 12+ | 4,212 | 0.1531 | [0.1428, 0.1635] |

### Verdict (b)

* Does targeting have room? The question is different here — (b) is a mechanism check, not a
  market. What it says about (a) is decisive: **the mechanism exists but points away from the
  scenario.** A foul on a dribbling opponent is 21% more likely to be carded overall, but the
  premium is *smaller* in the defending third (interaction OR 0.83, raw rates 0.208 vs 0.236),
  which is precisely where the "foul-prone full-back vs dangerous winger" duel takes place.
  The variance that a card model can actually explain is where and when the foul happened, and
  neither is knowable before kick-off.
* Effect sizes worth keeping: location (OR 0.47 per SD of distance to own goal) dominates
  everything; referee tendency is real but worth only OR 1.09 per SD; being already booked
  *reduces* the chance of another card (OR 0.81), which is the opposite of the naive
  "accumulator" intuition a prop model might encode.
* One caveat: `fouled_line_unknown` carries a large odds ratio (3.28) but is entangled with
  handballs (OR 0.22), which have no fouled player; the two largely offset and neither is used
  for any claim above.

## 4. (c) Team card totals, and the proxy honesty check that anchors the phase

### 4.1 Universe

football-data.co.uk matches with yellow-card counts and at least 5 strictly prior
card-recorded matches for both teams: **125,411 matches, 22 division codes, 2000-09-08 to
2026-09-03**. The chronological cut is taken over all 238,858 matches in the table, so
discovery ends 2017-11-30.

| | |
|---|---|
| discovery | 60,861 matches (to 2017-11-30) |
| confirmation | 64,550 matches (2017-12-01 onward) |
| mean total yellows per match | 3.691 (variance 4.448 — mildly overdispersed) |
| P(> 2.5 / 3.5 / 4.5 / 5.5 / 6.5) | 0.691 / 0.502 / 0.326 / 0.188 / 0.098 |

Teams are keyed by country + name so a club's history follows it through promotion and
relegation. Rolling features cover cards for and against, fouls for and against, shots,
shots on target, corners, goals, points, division-level card and foul levels (accumulated over
*(division, date)* aggregates), and a pairwise "derby" prior from the two clubs' previous
meetings. Market features (no-vig 1X2 probabilities, favourite strength, the no-vig Over 2.5
goals probability) are held in a separate nested set so their contribution is visible.

* **C0 proxy** — `mu = home_team_prior_match_cards + away_team_prior_match_cards -
  division_prior`, recalibrated by a two-parameter Poisson GLM on discovery, with a
  negative-binomial dispersion fitted on discovery; line probabilities from the NB survival
  function, then a per-line logistic recalibration.
* **C1 event-only** — LightGBM Poisson on the rolling team / division / pair features
  (no market odds), same NB + recalibration treatment.
* **C2 event + market** — C1 plus the six market features.
* **C2 binary** — direct LightGBM binary classifiers at 3.5 / 4.5 / 5.5, for comparison with
  the count-distribution route.

### 4.2 THE ANCHOR: how much better is a real bookmaker than a rolling mean?

We have no card lines. We do have real Over/Under 2.5 **goals** prices. So the identical
rolling-mean machinery was built for total goals on the same table (155,328 matches with
Bet365 O/U 2.5 prices, 90,538 discovery / 64,790 confirmation, mean Bet365 overround 1.0660),
Bet365's price was de-vigged proportionally, and both were scored at the 2.5 line.

| where | model | n | log_loss | brier | auc | delta vs proxy | 95% CI |
|---|---|---|---|---|---|---|---|
| confirmation | climatology | 64,790 | 0.6938 | 0.2503 | 0.500 | -0.0056 | [-0.0063, -0.0049] |
| confirmation | rolling-mean proxy | 64,790 | 0.6882 | 0.2476 | 0.557 | 0 | — |
| confirmation | our event-only goals model | 64,790 | 0.6857 | 0.2463 | 0.567 | +0.0025 | [+0.0019, +0.0032] |
| **confirmation** | **Bet365 (no-vig)** | 64,790 | **0.6771** | **0.2421** | **0.600** | **+0.0111** | **[+0.0101, +0.0122]** |

(Discovery: proxy 0.6880, model 0.6850 (+0.0029), Bet365 0.6822 (+0.0058). The gap is larger on
confirmation, i.e. in the more recent, more efficient era.)

The goals model is given exactly the mirror of the card models' feature set — the same rolling
team / division / pair machinery *and* its own proxy mean as a feature — and deliberately no
market features, since the market is what it is being scored against.

**A real bookmaker is +0.0111 nats better than the rolling-mean proxy. Our modelling machinery,
applied to the same market with the same care, closes only 23% of that gap.**

Stage 02 (corners) built the same check on the subset of matches that carry corner counts and
measured +0.0089 nats [+0.0080, +0.0099] (n = 42,140 confirmation matches,
`02_corners_goals_haircut_metrics.parquet`). Read that agreement for what it is rather than as
independent replication: `06_haircut.md` shows the corner pool is a **subset** of this stage's
pool (which does not condition on card counts at all -- the goals check needs a result, a price
and prior history, nothing else), the corner confirmation half is entirely contained in this
one, and both stages de-vig with the same shared helper. What the agreement establishes is that
the number does not depend on a stage's proxy tuning, capacity choice or split date, since
those differ and the answer does not. The residual spread, 0.0089 against 0.0111, is mostly the
extra 2017-12 to 2019-08 window this stage's earlier cut includes.

The same comparison as a betting dress rehearsal — the *same* event-only goals model, the
*same* threshold-selection procedure, bet first into a synthetic book priced off the proxy and
then into Bet365's real prices:

| book | hold | threshold | where | n bets | bet rate | mean edge | ROI | 95% CI |
|---|---|---|---|---|---|---|---|---|
| rolling-mean proxy | 0.04 | 0.20 | confirmation | 1,296 | 2.0% | 0.258 | **+0.176** | [+0.119, +0.232] |
| rolling-mean proxy | 0.06 | 0.20 | confirmation | 895 | 1.4% | 0.255 | **+0.186** | [+0.119, +0.250] |
| rolling-mean proxy | 0.08 | 0.20 | confirmation | 629 | 1.0% | 0.251 | **+0.169** | [+0.091, +0.249] |
| **Bet365 real prices** | 0.066 (actual) | 0.20 | **confirmation** | 4,527 | 7.0% | 0.299 | **-0.129** | **[-0.167, -0.092]** |

Bet365's realised two-way overround on these rows is **6.60%**, not 6%, so differencing the
actual-price leg against a 6%-hold proxy leg mixes 0.6 points of margin into what is meant to be
a sharpness comparison. A third leg removes that: Bet365's *no-vig* probability re-priced at the
same synthetic hold as the proxy leg. That is the phase's reference definition
(`06_haircut.md`), and on this universe it lands almost exactly where the actual-price version
does, because 6.60% is close to 6% (`01_cards_c_haircut_by_hold.parquet`):

| hold | ROI vs proxy prices | ROI vs Bet365 re-priced at the same hold | ROI vs Bet365's posted prices | **margin-matched haircut** | actual-prices haircut |
|---|---|---|---|---|---|
| 0.04 | +0.1760 | -0.1138 | -0.1293 | **0.2898** | 0.3053 |
| 0.06 | +0.1857 | -0.1241 | -0.1293 | **0.3099** | 0.3150 |
| 0.08 | +0.1689 | -0.1354 | -0.1293 | **0.3043** | 0.2982 |

| quantity | value |
|---|---|
| log-loss gap, real book minus rolling-mean proxy (confirmation) | **+0.0111 nats** |
| mean simulated ROI against the proxy-priced book, over 4/6/8% holds | +0.1769 |
| realised ROI against Bet365's actual prices | -0.1293 |
| ROI haircut, actual prices (mean-over-holds form used in the tables below) | 0.3062 |
| **ROI haircut, margin-matched at a 6% hold (the phase reference)** | **0.3099** |

A model that looks like it prints +17.7% against a rolling-mean book loses 12.9% against a real
one. That ~30-point gap is the haircut, and it is applied to every card ROI below. The two
definitions differ by 0.4 ROI points here, so nothing in this stage turns on which is used --
that is not true elsewhere in the phase, which is why `06_haircut.md` exists. (On discovery the
same rule bet only 1.4% of matches into Bet365 for -0.036 [-0.104, +0.037]; the larger
confirmation bet rate, 7.0%, is the model disagreeing with a sharper modern market more often
and being wrong about it.)

Two caveats, stated because they cut in opposite directions. (i) Goals is the single most
efficient market in football; a card book is probably *not* 0.0111 nats better than a rolling
mean, so the haircut likely over-corrects for cards. (ii) The haircut is a flat ROI
subtraction measured at bet rates of 1-7%, and transferring it to a rule that bets 0.6% of
matches at a 49% claimed edge is crude. Section 4.5's book-strength ladder is the better
instrument and does not require transferring a number between markets.

### 4.3 Do the models beat the proxy?

`delta_vs_proxy` is `log_loss(C0) - log_loss(model)`, match-clustered bootstrap, n = 64,550
confirmation matches.

| line | model | log_loss | brier | auc | delta vs proxy | 95% CI |
|---|---|---|---|---|---|---|
| 2.5 | climatology | 0.5960 | 0.2025 | 0.500 | -0.0379 | [-0.0399, -0.0358] |
| 2.5 | C0 proxy | 0.5582 | 0.1874 | 0.628 | 0 | — |
| 2.5 | C1 event | 0.5523 | 0.1846 | 0.638 | +0.0059 | [+0.0047, +0.0070] |
| 2.5 | C2 event+market | 0.5520 | 0.1845 | 0.639 | +0.0061 | [+0.0050, +0.0073] |
| **3.5** | climatology | 0.7126 | 0.2597 | 0.500 | -0.0478 | [-0.0503, -0.0454] |
| **3.5** | **C0 proxy** | 0.6648 | 0.2362 | 0.632 | 0 | — |
| **3.5** | **C1 event** | 0.6585 | 0.2331 | 0.639 | **+0.0063** | [+0.0051, +0.0075] |
| **3.5** | **C2 event+market** | 0.6581 | 0.2329 | 0.641 | **+0.0067** | [+0.0055, +0.0080] |
| **3.5** | C2 binary | 0.6579 | 0.2329 | 0.641 | +0.0069 | [+0.0056, +0.0082] |
| 4.5 | C0 proxy | 0.6365 | 0.2221 | 0.643 | 0 | — |
| 4.5 | C1 event | 0.6298 | 0.2194 | 0.650 | +0.0067 | [+0.0055, +0.0079] |
| 4.5 | C2 event+market | 0.6294 | 0.2192 | 0.651 | +0.0071 | [+0.0059, +0.0084] |
| 4.5 | C2 binary | 0.6310 | 0.2199 | 0.649 | +0.0055 | [+0.0041, +0.0068] |
| 5.5 | C0 proxy | 0.5097 | 0.1655 | 0.660 | 0 | — |
| 5.5 | C2 event+market | 0.5038 | 0.1636 | 0.668 | +0.0059 | [+0.0048, +0.0070] |
| 6.5 | C0 proxy | 0.3519 | 0.1026 | 0.683 | 0 | — |
| 6.5 | C2 event+market | 0.3474 | 0.1016 | 0.691 | +0.0045 | [+0.0036, +0.0054] |

Three-seed refit spread at the 3.5 line: 0.657859 / 0.658066 / 0.658091 — **0.0002 nats**, so
the +0.0067 margin is 30x the noise floor and the market features' +0.0004 contribution is not.
The direct binary classifiers and the count-distribution route agree to within 0.0017 nats:
modelling the whole distribution is neither better nor worse than fitting each line.

**Yes, the event-only model beats the rolling mean — by +0.0067 nats. And a real book beats
the rolling mean by +0.0111 nats on the comparably-priced goals line (base rates 0.557 and
0.503, both near even money). We are 60% of the way to a real bookmaker, from below.**

### 4.4 The gate

Pre-registered on discovery: combined prior fouls-per-match of the two teams in the **top
tercile** (cut 27.70 fouls/match), absolute Elo gap in the **bottom tercile** (cut 47.95,
i.e. an evenly matched game), division prior card level in the **top tercile** (cut 3.371). Target: match
yellows over 3.5. Feature set C2.

| where | scenario | n | generalist LL | specialist LL | delta (spec - gen) | 95% CI |
|---|---|---|---|---|---|---|
| discovery CV | high foul both | 20,287 | 0.6311 | 0.6380 | **-0.0069** | [-0.0087, -0.0051] |
| confirmation | high foul both | 26,015 | 0.6216 | 0.6303 | **-0.0087** | [-0.0102, -0.0072] |
| confirmation, off-scenario | high foul both | 38,535 | 0.6849 | 0.6956 | -0.0106 | [-0.0123, -0.0089] |
| discovery CV | + tight Elo | 6,478 | 0.6197 | 0.6486 | **-0.0289** | [-0.0354, -0.0227] |
| confirmation | + tight Elo | 8,545 | 0.6080 | 0.6303 | **-0.0223** | [-0.0268, -0.0175] |
| confirmation, off-scenario | + tight Elo | 56,005 | 0.6672 | 0.6944 | -0.0272 | [-0.0292, -0.0251] |
| discovery CV | + cardy division | 5,915 | 0.6145 | 0.6483 | **-0.0338** | [-0.0406, -0.0270] |
| confirmation | + cardy division | 8,535 | 0.6080 | 0.6337 | **-0.0258** | [-0.0310, -0.0205] |
| confirmation, off-scenario | + cardy division | 56,015 | 0.6672 | 0.6971 | -0.0298 | [-0.0321, -0.0278] |

**Targeting has no room for team card totals either**, and here the sample is large enough that
the answer is unambiguous: every specialist is worse, on both splits, with intervals well clear
of zero, and the penalty grows as the scenario narrows (-0.0087 -> -0.0223 -> -0.0258) even
though the narrowest scenario still has 5,915 discovery rows. A model fitted on 61,000 matches
knows more about a high-foul, evenly matched game in a card-happy division than a model fitted
on the 5,915 such games alone.

### 4.5 Betting simulation

For each match the line is the standard half-line at which the proxy's over-probability is
closest to 0.5 (3.5 for 54,697 matches, 2.5 for 40,251, 4.5 for 26,054, 5.5 for 4,407, 6.5 for
2), the proxy's fair probability is marked up to `1 + hold` on both sides, and the model bets
the better side when its EV exceeds a threshold chosen on discovery (grid 0 .. 0.40).

| model | hold | threshold | where | n bets | bet rate | mean edge | ROI | 95% CI | realised | mean odds | ROI after haircut | lower bound after haircut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 event | 0.06 | 0.40 | discovery | 453 | 0.74% | 0.523 | +0.472 | [+0.357, +0.587] | 0.561 | 2.59 | +0.166 | +0.051 |
| C1 event | 0.06 | 0.40 | **confirmation** | 353 | 0.55% | 0.484 | **+0.338** | [+0.225, +0.447] | 0.584 | 2.16 | **+0.031** | **-0.081** |
| C2 event+market | 0.04 | 0.40 | **confirmation** | 551 | 0.85% | 0.483 | **+0.344** | [+0.254, +0.433] | 0.572 | 2.19 | **+0.038** | **-0.052** |
| C2 event+market | 0.06 | 0.40 | **confirmation** | 376 | 0.58% | 0.487 | **+0.305** | [+0.192, +0.416] | 0.572 | 2.16 | **-0.001** | **-0.114** |
| C2 event+market | 0.08 | 0.40 | **confirmation** | 263 | 0.41% | 0.491 | **+0.345** | [+0.216, +0.475] | 0.597 | 2.14 | **+0.038** | **-0.091** |

The discovery search picked the **largest** threshold on the grid (0.40) in every configuration,
which is a warning sign in itself: the procedure is saying "bet less and less", and the rule
that survives bets 0.4-0.9% of matches — roughly one bet per division per fortnight. Before the
haircut the ROI is a clean +0.31 to +0.34 with intervals excluding zero. After the flat
30.6-point haircut the point estimates collapse to -0.001 to +0.038 and every lower bound is
negative: **no demonstrated edge.**

Where the model and the proxy disagree, the model is closer to the truth in every bin — but it
is also over-confident in the bin that drives the betting (bin 4: model 0.624, proxy 0.495,
truth 0.590):

| bin | n | diff range | mean p_model | mean p_book | observed | LL model | LL book | model better by |
|---|---|---|---|---|---|---|---|---|
| 0 | 12,910 | -0.338 .. -0.020 | 0.426 | 0.497 | 0.458 | 0.6821 | 0.6874 | +0.0053 |
| 1 | 12,910 | -0.020 .. +0.023 | 0.499 | 0.496 | 0.527 | 0.6862 | 0.6866 | +0.0004 |
| 2 | 12,910 | +0.023 .. +0.054 | 0.536 | 0.497 | 0.556 | 0.6790 | 0.6852 | +0.0061 |
| 3 | 12,910 | +0.054 .. +0.088 | 0.570 | 0.499 | 0.569 | 0.6736 | 0.6837 | +0.0101 |
| 4 | 12,910 | +0.088 .. +0.532 | 0.624 | 0.495 | 0.590 | 0.6687 | 0.6870 | +0.0183 |

**The book-strength ladder** (synthetic book interpolated on the logit scale between the proxy
and our own model; `book gain` is how much that book beats the proxy on confirmation, directly
comparable to the +0.0111 nats Bet365 holds on goals):

| lam | book gain over proxy (nats) | hold | threshold | n bets | ROI | 95% CI |
|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.06 | 0.40 | 376 | +0.305 | [+0.192, +0.416] |
| 0.25 | 0.0045 | 0.06 | 0.30 | 114 | +0.222 | [+0.053, +0.391] |
| 0.50 | 0.0073 | 0.06 | 0.12 | 420 | **+0.005** | [-0.073, +0.082] |
| 0.75 | 0.0085 | 0.06 | 0.02 | 552 | **-0.057** | [-0.119, +0.009] |
| 1.00 | 0.0080 | 0.06 | 0.00 | 0 | — | — |

The edge reaches zero at a book **0.0073 nats** better than the rolling mean and is negative at
0.0085 — both *below* the 0.0111 nats that a real bookmaker demonstrably achieves on the
comparable goals line. This is the same conclusion as the flat haircut, reached without
transferring an ROI between markets, and it is the one to trust.

### Verdict (c)

* Does targeting have room? **No**, unambiguously: every specialist is worse than the
  generalist on held-out scenario rows, on both splits, with the penalty growing as the
  scenario narrows (-0.0087 -> -0.0223 -> -0.0258 nats).
* Does any model beat the proxy by more than the haircut? **No.** C2 beats the proxy by
  +0.0067 nats [+0.0055, +0.0080] at the 3.5 line, versus the +0.0111 nats [+0.0101, +0.0122] a
  real book holds over that same proxy on goals. The simulated +30.5% ROI against the strawman
  falls to -0.1% [-11.4%] after the flat haircut and to +0.5% [-7.3%, +8.2%] once the synthetic
  book is only two-thirds of the way to real-book sharpness.
* What would a bet rule look like? "Bet the proxy's own line when the model's EV exceeds 40%":
  0.58% of matches (376 bets in 64,550 confirmation matches at a 6% hold), mean modelled edge
  48.7%, ROI +0.305 [+0.192, +0.416] against the rolling mean, and no demonstrable edge against
  anything sharper. The threshold sitting at the top of the grid says the rule is living in the
  tail of the disagreement distribution, which is exactly where a strawman book is most wrong
  and a real one is not.

## 5. Verdicts

| question | (a) player card props | (b) foul -> card | (c) team card totals |
|---|---|---|---|
| does targeting have room? | **no** (specialist -0.0208 [-0.0395, -0.0011] on the pre-registered scenario, n = 400) | n/a (mechanism check) | **no** (specialist -0.0258 [-0.0310, -0.0205], n = 8,535) |
| does a model beat the book proxy? | yes, +0.0064 nats [+0.0042, +0.0084] | yes, 7.6% skill over the base rate | yes, +0.0067 nats [+0.0055, +0.0080] |
| by more than the haircut? | **no** (margin 0.0064 < 0.0111) | n/a | **no** (margin 0.0067 < 0.0111) |
| bet rule in practice | 4.8% of appearances, edge 59%, ROI +0.265 [+0.062, +0.485] vs the strawman; +0.023 [-0.175, +0.224] vs a P1-strength book | n/a | 0.58% of matches, edge 49%, ROI +0.305 [+0.192, +0.416] vs the strawman; +0.005 [-0.073, +0.082] vs a two-thirds-sharp book; -0.001 [-0.114] after the flat haircut |

**The phase-level lesson.** Both card markets give the same answer twice, by two independent
routes. Scenario targeting does not help, because a specialist starved of rows loses more to
variance than it gains from specificity — and that is not a modelling failure to be engineered
around, it is a measurement: the conditional structure a scenario isolates is already learned
by the generalist from the other rows. And the margin that our event-only models *do* hold over
a rolling-mean book (0.0064-0.0067 nats) is smaller than the margin a real bookmaker holds over
the same rolling mean on the one count market where we can check (0.0111 nats). A simulated
edge against a proxy book is therefore not evidence of an edge; on this data it is mostly
evidence that the proxy is bad.

The specific hypothesis that opened this stage — a foul-prone full-back against a high-volume
dribbler with a card-happy referee — fails at three independent points: the matchup features add
+0.0006 nats, inside the refit noise floor; the gate says a specialist for exactly that
situation is worse than a generalist; and the foul-level mechanism says the dribble-contact card
premium is *smaller*, not larger, in the third of the pitch where a full-back defends.

## 6. What is and is not established

* The event-only advantage over a rolling mean is **real and replicated** (two different
  targets, two different data sources, 15,148 and 64,550 confirmation rows, both intervals well
  clear of zero and far above the refit noise floors of 0.0012 and 0.0002 nats). The mechanism
  is mundane: fouls are a better-measured proxy for card propensity than cards are, and a
  rolling card mean throws that away.
* The haircut is measured on **goals, not cards**. Card books may well be softer than goals
  books, which would make the haircut too harsh. The book-strength ladder is offered precisely
  so a reader who believes card books are softer can read off the answer at their own assumed
  sharpness; at any book better than 0.0073 nats over the rolling mean, the card edge is gone.
* The card-market simulation prices both sides off the proxy at a stated hold. Real card props
  also carry limits, line movement, and the fact that a 40%-edge bet is not available at the
  size the ROI implies. None of that is modelled, and all of it cuts against the edge.
* Player-level results are on **one season of four leagues** (1,398 matches). The StatsBomb
  tournaments were excluded from part (a) because players arrive with almost no in-sample
  history; part (b) uses them as a genuine out-of-domain check and the mechanism transfers.
* Goalkeepers are excluded from part (a). Their card rate is very low and player card props are
  rarely offered on them; including them would have inflated AUC without changing any verdict.
* Bet365's Over/Under 2.5 price is the *opening or closing* price recorded by
  football-data.co.uk, not necessarily the price available at any moment; the de-vig is
  proportional. A different de-vig (e.g. Shin) would move the +0.0111 nats gap by a few
  ten-thousandths, not by a factor.
* The post-haircut team-card ROI (-0.001 to +0.038, lower bounds -0.11 to -0.05) is reported as
  "no demonstrated edge", not as "an established loss". The haircut itself is a point estimate
  from one market; a reader who thinks card books are softer than goals books should read the
  book-strength ladder instead, which says the edge dies at 0.0073 nats of book sharpness.

## 7. Result tables

All in `research/scenario_ev/reports/`, one parquet per table:

`01_cards_a_setup`, `a_audit`, `a_audit_features`, `a_proxy_constants`, `a_fit_info`,
`a_metrics`, `a_calibration`, `a_p3_subset`, `a_state_coverage`, `a_scenario_defs`, `a_gate`,
`a_refit`, `a_bets`, `a_book_ladder`, `a_disagreement`;
`01_cards_b_counts`, `b_metrics`, `b_odds_ratios`, `b_rate_by_zone_dribbling`,
`b_rate_by_team_fouls`, `b_rate_by_minute`, `b_rate_by_fouled_line`, `b_importance`;
`01_cards_c_setup`, `c_goals_setup`, `c_goals_gap`, `c_goals_bets`, `c_haircut`,
`c_haircut_by_hold`, `c_metrics`,
`c_fit_info`, `c_gate`, `c_scenario_defs`, `c_bets`, `c_book_ladder`, `c_line_mix`,
`c_disagreement`, `c_refit`.

Processed intermediates (not committed) live in
`$PRIV_DATA_DIR/processed/scenario_ev/`: `cards_player_match.parquet`,
`cards_team_state.parquet`, `cards_foul_level.parquet`, `cards_goal_times.parquet`,
`cards_odds_matches.parquet`.
