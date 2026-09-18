# Scenario expected value: can a model price one stat, in one scenario, better than a bookmaker?

Synthesis of the four-candidate phase in `research/scenario_ev/`. Sources: the six stage
reports in this directory (`01_cards.md`, `02_corners.md`, `03_pass_counts.md`,
`04_fouls.md`, `05_gate_sweep.md`, `06_haircut.md`) and the parquet tables beside them.
**Every number below is read from one of those tables**; the machine-checkable subset is
re-derived from them by `research/scenario_ev/results_index.py` into
`results_index.parquet` (116 rows), and `research/scenario_ev/report_sources.py` pins every
report in this directory to a digest of the tables it was rendered from, so a stage that is
re-run without its reports being re-rendered fails the test suite instead of being
discovered by a reader. Discovery numbers are exploratory and multiplicity-inflated;
**confirmation numbers were scored once and are the headline**, and every number is
labelled.

---

## 0. The answer in one paragraph

**No — but the null is narrower than "nothing survives", and the two things that do survive
are named here rather than buried.** On four markets — cards, corners, player pass counts
and fouls — scenario-specific modelling has no room to work in: a specialist beats a
generalist on held-out confirmation rows of its own scenario in 1 of 38 pre-registered
cells, and that one cell is the expected false positive. The teacher-student imputed-state
channel that motivated the whole programme adds nothing incremental to any of the four
outcome models. Our models *do* beat a bookmaker's rolling-mean pricing heuristic —
reliably, on large samples, with intervals far clear of zero — and against a market priced
off that heuristic they simulate ROIs of +11% to +31%. But on the one count market where a
real line is observable (total goals at 2.5), **a real bookmaker beats the identical rolling
mean by 0.0089–0.0111 nats**, which is *larger* than the margin our card, foul and
match-corner models hold over the same rolling mean. Charging that gap as the phase's single
reference haircut (section 3), the card, foul and match-corner ROIs all go to zero or below.
**Two results do not.** The team corner line inside `fav_strong` keeps +0.055 [+0.035] — but it
bets 99.95% of the matches in that scenario, and 95% of the model's gain is the 1X2 price
the proxy cannot see, so it measures the proxy's blindness rather than a market
inefficiency. Player pass counts keep +0.043 [+0.020] — but every feature carrying the
advantage is a public rolling average, and the model claims to disagree with a rolling mean
1.8× as hard as a real book does on goals. Neither is killed by the haircut and this
document does not pretend otherwise; both are argued away on separate grounds, which is a
weaker form of evidence and is labelled as such. There is no demonstrated tradeable edge
here, and the honest reason is not that our models are bad: it is that the opponent we could
construct is a strawman, and the one measurement we have of the real opponent says it is
sharper than we are.

---

## 1. The question, and why it differs from the completed programme's

The completed programme (`research/privileged_tracking/reports/README.md`) asked whether
game state that needs player tracking can be learned from event data and imputed where
tracking is absent. It answered yes for recoverability (soccer block depth R² 0.96 out of
fold) and **no for value**: adding imputed state to outcome models that already read the
same events changed nothing measurable (soccer xG −0.0003 nats [−0.0013, +0.0007],
n = 10,233; NFL completion −0.0009 [−0.0042, +0.0023], n = 6,225). Its sharpest finding was
an inverse relationship: the quantities whose oracle value is highest (near-goal cone
occupancy, worth +0.0220 nats at 0–8 yd) are the least recoverable (R² 0.16), and the most
recoverable (block depth) is worth almost nothing.

Those were **model-vs-model comparisons on mean log-loss over all rows**. This phase asked
a different question, the user's question:

> Forget models that are better on average. Can a model predict **one specific stat** with
> **positive expected value** in **specific scenarios**, against a **bookmaker's pricing
> heuristic** rather than against our own strong event-only model?

Two things change. The opponent is much weaker — side markets are priced off rolling
season averages plus a margin, not off a tuned model. And the metric is a betting decision
at a line, not mean log-loss: a model can be worse on average and still profitable if it
is right where it disagrees.

Four candidates were chosen because their channel was untested and their market is soft:
**cards**, **fouls**, **corners**, and **player pass counts** (the one place the completed
programme found a real imputed-state signal, +0.0026 nats per pass, never converted to a
count line).

Two structural facts govern everything that follows and are stated here rather than
buried. First, **there are no corner, card, foul or pass-prop lines anywhere in this
data** — only 1X2 and Over/Under 2.5 *goals* odds. Every "bookmaker" in this phase is a
constructed rolling-mean proxy. Second, that is exactly why the protocol's step 4 (the
haircut, section 3) exists, and why it, not the simulated ROIs, is the load-bearing part
of the phase.

### The protocol every stage followed

1. **Discovery / confirmation split fixed in code** before any modelling, chronological,
   whole matches, ~60/40. Every scenario definition, tercile cut, feature set, capacity
   choice, calibration map, dispersion parameter and bet threshold was chosen on
   discovery. Confirmation was scored once.
2. **The gate** (section 2): generalist vs specialist, both scored on the *same* held-out
   scenario rows, with the reverse (off-scenario) direction as a sanity check.
3. **The book proxy**: a shrunk rolling mean over strictly prior matches, *recalibrated on
   discovery* — an uncalibrated strawman is beaten on scale alone, which would manufacture
   an edge out of nothing. Shrinkage `k` and functional form were swept on discovery, so
   the opponent is the strongest rolling mean available.
4. **The proxy honesty check** (section 3): the same machinery pointed at total goals,
   scored against Bet365's de-vigged Over/Under 2.5 price.
5. **Betting-shaped evaluation**: full count distributions (LightGBM Poisson / negative
   binomial with a fitted dispersion) *and* direct binary classifiers at the half-lines;
   both sides priced at 4 / 6 / 8% two-way hold; bets filtered by a discovery-chosen edge
   threshold; ROI with match-clustered bootstrap intervals; a CLV-style check of who is
   right where model and proxy disagree.
6. **Hygiene verified, not asserted**: every rolling feature re-derived by brute force for
   random samples (max abs error 0.0 in every stage's audit table), splits grouped by
   match, same-day fixtures excluded by construction, fixed seeds, and a **refit-noise
   floor** so small deltas are not oversold.

---

## 2. The gate: is there room for scenario-specific modelling at all?

This is the test that governs the whole exercise. Scenario-specific modelling can only
help if the general model is **underfitting** the scenario. That is directly testable: fit
a generalist on all training rows, fit a specialist on the scenario's training rows only,
score both on the *same* held-out scenario rows.

`delta = loss(generalist) − loss(specialist)`, so **positive means the specialist is
better, i.e. targeting has room.** Intervals are match-clustered bootstraps (2,000
resamples) on the paired per-row loss difference. The uniform run is
`gate_sweep.py`, which imports each market's own universe, feature set and LightGBM
settings and runs them all through one implementation (`common.run_gate`).

### 2.1 All 38 cells, collapsed by market (confirmation)

| market | cells | room | no room | inconclusive | best delta | median delta | worst delta | refit floor |
|---|---|---|---|---|---|---|---|---|
| match_corners | 5 | 0 | 4 | 1 | −0.00115 | −0.00168 | −0.00533 | 0.00009 |
| team_corners | 5 | **1** | 1 | 3 | **+0.00538** | +0.00006 | −0.00120 | 0.00045 |
| player_cards | 4 | 0 | 1 | 3 | −0.00125 | −0.00152 | −0.02080 | 0.00344 |
| team_cards | 4 | 0 | 3 | 1 | +0.00032 | −0.00480 | −0.00830 | 0.00268 |
| player_fouls | 5 | 0 | 4 | 1 | +0.00115 | −0.01854 | −0.03305 | 0.00102 |
| team_fouls | 5 | 0 | 0 | 5 | +0.00039 | −0.00048 | −0.00087 | 0.00028 |
| player_passes | 5 | 0 | 5 | 0 | −0.00527 | −0.00914 | −0.01130 | 0.00211 |
| player_passes_completed | 5 | 0 | 3 | 2 | +0.00034 | −0.00988 | −0.01455 | 0.00107 |
| **total** | **38** | **1** | **21** | **16** | | −0.00237 | | |

Source: `05_gate_sweep_headline.parquet`, report `05_gate_sweep.md`. Losses are binary log
loss for `player_cards` and negative-binomial log scores for the seven count markets, so
deltas are comparable *within* a market, not across markets.

**The gate is shut.** One cell in 38 has a confirmation interval clear of zero in the
specialist's favour; 21 are clear of zero *against* it. Three further facts say the single
positive is not a finding:

* Its discovery delta was +0.00079 against a confirmation +0.00538 — the halves agree on
  sign but differ by an order of magnitude in size, so it could not have been
  pre-registered as an effect of that size.
* With 38 cells tested at a 5% level, one or two intervals clearing zero by chance is the
  expectation.
* It is `team_corners / fav_strong`, and section 4.2 shows that is precisely the market
  where the haircut removes the whole margin anyway. A real cell there would still not be
  tradeable.

### 2.2 Why the specialist loses, and the check that it is a real model

* **The specialist loses hardest where the scenario is narrowest.** Across the 38 cells the
  delta correlates +0.56 with the log of the scenario's training rows; the median delta
  rises from −0.00945 in the smallest third of scenarios to −0.00167 and −0.00071
  (`05_gate_sweep.md`).
* **Off-scenario the specialist also loses, in 37 of 38 cells** — the expected direction,
  confirming the specialists are working models starved of rows rather than broken ones.
* **Discovery and confirmation agree on the sign in 29 of 38 cells**; where they do not,
  the cell is labelled inconclusive rather than read as a result.

Stage 01's own team-card gate makes the point at scale with a scenario that is *not*
small: the penalty **grows** as the scenario narrows (−0.0087 → −0.0223 → −0.0258 nats at
the 3.5 line, n = 26,015 / 8,545 / 8,535 confirmation) even though the narrowest rung
still has 5,915 discovery rows to fit on. A model fitted on 61,000 matches knows more
about a high-foul, evenly matched game in a card-happy division than a model fitted on the
5,915 such games alone.

### 2.3 The pre-registered hypothesis cells, one per candidate (confirmation)

| candidate | pre-registered scenario | n | delta (spec − gen) | 95% CI | verdict |
|---|---|---|---|---|---|
| cards | full-back × top-tercile opponent flank dribbles × card-happy referee | 400 | **−0.02080** | [−0.04088, −0.00064] | no room |

| corners | favourite no-vig probability ≥ 0.65 (team line) | 9,272 | **+0.00538** | [+0.00317, +0.00744] | room (the one cell; see 2.1) |
| pass counts | the team's usual highest-volume passer is absent | 3,555 | **−0.01130** | [−0.01878, −0.00348] | no room |
| fouls | defender facing top-tercile opponent flank dribbling | 2,176 | **−0.02124** | [−0.02924, −0.01244] | no room |

All four intervals come from the uniform sweep. Where a stage also ran the same cell in its
own local copy of the loop, the point estimate agrees exactly and the interval differs
slightly because the bootstraps were drawn separately — the cards cell is [−0.04088,
−0.00064] here and [−0.03952, −0.00114] in `01_cards.md`. `05_gate_sweep.md` line 47 explains
why the stages grew local copies; the sweep's numbers are the ones to cite.

### 2.4 The structural caveat, and the second test that answers it

The gate tests whether *refitting the same model family on a subset* helps. It does not
test whether a **different feature set**, specific to a scenario, could help. Each stage
therefore tested exactly that for its own hypothesis, as a nested feature block on top of
the generalist. Confirmation, against each stage's own refit-noise floor:

| candidate | hypothesis, as a feature block | delta (nats) | 95% CI | n | refit floor | reading |
|---|---|---|---|---|---|---|
| cards | matchup block (P1 → P2): opponent mirrored-flank dribbling, direct opposing starter, referee × matchup products | +0.0006 | (inside the floor) | 15,148 | 0.0012 | noise |
| corners | own style over the odds table (crossing, blocked shots, passes into box) | +0.0071 fwd / +0.0029 CV | intervals contain zero | 1,054 rows / 527 matches | — | weak |
| corners | opponent style, and opponent imputed compactness, on top | −0.0064 fwd / −0.0012 CV | intervals contain zero | 1,054 | — | nothing |
| pass counts | imputed opponent block depth on the attempts model | +0.00081 | [−0.00091, +0.00251] | 13,053 | 0.00149 | noise |
| fouls | opponent-foul block on **fouls won** | **+0.00234** (5-seed mean +0.00234, range 0.00191–0.00269) | [+0.00111, +0.00364] | 13,970 | 0.00078 (5 seeds) | ~3× noise, survives |
| fouls | dribble-side block on **fouls won** | +0.00126 (5-seed mean **+0.00097**, range 0.00056–0.00129) | [+0.00030, +0.00218] | 13,970 | 0.00073 | 1.3× noise at the seed mean, 0.8× at the worst seed |
| fouls | opponent-foul block on **fouls committed** | +0.00010 | [−0.00097, +0.00110] | 13,970 | 0.00073 | noise |
| fouls | dribble-side block on **fouls committed** | +0.00028 | [−0.00028, +0.00084] | 13,970 | 0.00082 | noise |

The dribble-side row is quoted two ways on purpose. The headline seed gives +0.00126 and an
earlier draft reported only that, which is the **largest** of the five seeds
(`04_fouls_b_refit.parquet`: 0.00126, 0.00077, 0.00129, 0.00096, 0.00056). At the seed mean
it is +0.00097, about 1.3× its own 0.00073 spread, and the weakest seed is below it. The
opponent-foul row is robust to the same treatment (headline +0.00234, seed mean +0.00234).
The multiplicity is also larger than an earlier draft declared: `04_fouls_b_channels.parquet`
holds **12** confirmation increments across the two targets, not four, and on fouls won the
referee step (+0.00109 [+0.00009, +0.00218]) and the full `P1r → P2_plus_matchup` step
(+0.00232 [+0.00096, +0.00367]) also clear zero.

**One target in eight produced channels that cleared zero on its confirmation half: fouls
*won*.** Who a player is up against moves how many fouls he **draws**; it does not
measurably move how many he **commits**. Stage 04 states how far that goes and no further,
and the same standard is applied to it as to the failures: both fouls-won channels were
*inside* noise on discovery (+0.00126 [−0.00015, +0.00255] and +0.00016 [−0.00071,
+0.00105]), 12 confirmation increments were tested across two targets, and the dribble-side
channel's five seeds straddle its own noise floor, so this is a confirmation-only clearance
resting on sign consistency across both halves, not on independent replication. A few thousandths of a nat per appearance is a mechanism finding,
not a price.

Stage 01 explains *why* the card version of the same hypothesis fails, at the level of the
individual foul (45,517 StatsBomb fouls, 27,524 discovery / 17,993 confirmation, 12.9%
carded; LightGBM log-loss 0.3509 vs a 0.3796 base rate, AUC 0.695, transferring to 14,994
held-out tournament fouls at AUC 0.672). Fouls on a dribbling opponent *are* more carded
overall (OR 1.21 [1.08, 1.40] per match-clustered bootstrap) — but the dribbling × own-third
interaction is 0.83 [0.63, 1.05], and the raw confirmation rates in the defending third run
**the wrong way**: 0.208 [0.192, 0.225] with dribbling against 0.236 [0.207, 0.265]
without. The dribble-contact card premium *reverses* exactly where a full-back defends.
The variance a card model can explain is where and when the foul happened (distance to own
goal, OR 0.47 per SD; minute, OR 1.33 per SD), and neither is knowable before kick-off.

---

## 3. The proxy honesty check: how much better is a real bookmaker than a rolling mean?

This is protocol step 4, and it is what makes the rest credible. We have no corner, card,
foul or pass lines. We do have real Over/Under 2.5 **goals** prices. So each stage built the
*identical* rolling-mean machinery for total goals on its own universe, de-vigged Bet365's
price, and measured the gap.

### 3.1 In nats, at the 2.5 goals line (confirmation)

| stage | universe | n | proxy log-loss | Bet365 log-loss | **gap** | 95% CI |
|---|---|---|---|---|---|---|
| 01 cards | every priced match with prior history | 64,790 | 0.6882 | 0.6771 | **+0.0111** | [+0.0101, +0.0122] |
| 02 corners | those of them recording corners | 42,140 | 0.6850 | 0.6761 | **+0.0089** | [+0.0080, +0.0099] |
| 03 pass counts | four leagues, all seasons | 11,438 | 0.6868 | 0.6762 | **+0.0106** | [+0.0081, +0.0130] |
| 04 fouls | those recording fouls | 42,004 | 0.6850 | 0.6760 | **+0.0090** | [+0.0079, +0.0100] |

Discovery, for comparison: +0.0058, +0.0052, +0.0105, +0.0052 — the gap is **larger on
confirmation** in all four stages, and markedly so in three of them: the more recent, more
efficient era is where the rolling mean falls furthest behind.

**These are not four independent replications, and an earlier draft of this document said
they were.** `06_haircut.md` measures the overlap directly from the odds table, and the
pools are a chain of nested sets: the foul pool is a subset of the corner pool (110,362 of
110,752, Jaccard 0.9965 — which is why stages 02 and 04 agree to five decimals on both
log-losses; they are the same measurement run twice), and the corner pool is in turn a subset
of the card pool, because stage 01's goals check never conditions on card counts at all.
Stage 03's four leagues are nested inside all of them. Nor do the split dates separate them:
the corner/foul confirmation half (48,889 priced matches after 2019-08-31) is **100%**
contained in the card confirmation half (65,203 after 2017-11-30). All four stages also call
the same de-vig helper (`common.novig_two_way`) on the same Bet365 price columns.

What the agreement *does* establish is that the number does not depend on a stage's proxy
shrinkage, functional form, capacity choice or split date — those differ between the stages
and the answer does not — and the 0.0089-to-0.0111 spread is mostly the extra 2017-12 to
2019-08 window the card cut includes, not sampling noise. **The phase has one measurement of
this quantity, well instrumented, not four.**

**A real bookmaker is 0.009–0.011 nats better than a shrunk, discovery-tuned, recalibrated
rolling mean on a count market.** That is the bar. Two independent checks on the number
itself: a Shin / additive / power de-vig moves stage 01's +0.0111 by ten-thousandths, not by
a factor; and our own event-only goals model, given the exact mirror of the card models'
feature set, closes only 23% of that gap (+0.0025 [+0.0019, +0.0032], n = 64,790) — 45% in
stage 04's version (+0.0040 [+0.0030, +0.0049], n = 42,004). On the one market where the
comparison is possible, this modelling apparatus is markedly worse than the bookmaker it
would have to beat.

### 3.2 In ROI: three definitions, and the one the phase now uses

The stages expressed the same gap in money three different ways. That was a real
inconsistency: which market "survived the haircut" depended on which stage's arithmetic a
reader picked up, and the two stages that disagreed most — corners with the most lenient
definition, fouls with the harshest — ran on a 99.7%-identical match set.
`research/scenario_ev/haircut.py` removes the freedom. It collects all three into one table,
names one the phase reference, and applies all three uniformly to every confirmation betting
row the phase published.

| definition | what it charges | 6% hold, card pool | 6% hold, corner/foul pool |
|---|---|---|---|
| `book_earns_one_way` | the ROI a real book makes betting its de-vigged price into rolling-mean prices. One-directional — it charges the profit that evaporates but not the loss that appears | — | **0.1246** [+0.1101, +0.1395] |
| **`margin_matched_round_trip`** (**the reference**) | `ROI(our goals model vs proxy prices at hold h) − ROI(our goals model vs the real book's no-vig price re-priced at the same h)`. Both legs carry the same margin, so the difference is sharpness alone | **0.3099** | **0.2073** |
| `actual_prices_round_trip` | the same round trip against Bet365's *posted* prices, at whatever hold Bet365 charged | 0.3062 | 0.2067 |

At 4% / 6% / 8% holds the reference is 0.2898 / 0.3099 / 0.3043 on the card pool and
0.1736 / 0.2073 / 0.2282 on the corner/foul pool. **The one-directional number is kept as a
lower bound and never as the bar**; the phase's earlier "+0.008 corner residual" was that
lenient definition applied to the corner team line, and it becomes −0.075 under the
reference.

Two corrections to the earlier draft's account of these numbers. First, the 0.3062 cards
haircut was described as running against "5.78% realised hold"; Bet365's realised two-way
overround on stage 01's own rows is **6.60%** (`01_cards_c_goals_bets.parquet`, `hold` =
0.065983), and 5.78% is stage 04's confirmation-era figure. Second, stage 01 now computes
the margin-matched leg too (`01_cards_c_haircut_by_hold.parquet`), which removes the 0.6
points of margin mismatch that pairing a 6.60%-hold leg with a 6%-hold leg carried: on that
universe it lands at 0.3099, within 0.4 points of the actual-prices version, so nothing in
the card stage turns on the choice.

Why the card pool's haircut (0.31) is so much larger than the corner/foul pool's (0.21): the
card measurement's confirmation half starts 21 months earlier and contains 16,314 extra
matches from a window where the rolling mean is comparatively worse, and the goals model's
threshold search lands on a different bet rate. Card rows are therefore haircut against the
card pool and everything else against the corner/foul pool, which is the pool each was
simulated on.

### 3.3 The instrument that does not require transferring a number between markets

Stages 01, 03 and 04 also built a **book-strength ladder**: a synthetic opponent
interpolated on the logit scale between the proxy (λ = 0) and our own model (λ = 1), whose
`book_gain` over the proxy is measured in the same nats as section 3.1. A reader who
believes card or corner books are softer than goals books can pick their own sharpness and
read off the answer, without believing the transfer. It is the better instrument and is
used in section 4.

---


## 4. Results by candidate

Format for each: what the model is worth against the proxy (confirmation, with n and CI),
the scenario result, and the betting simulation at 4 / 6 / 8% hold, raw and after the
haircut.

### 4.1 Cards (stage 01)

**Universes.** Player props: 33,329 StatsBomb player-appearances (1,398 matches of the four
2015/16 club seasons, outfield, ≥ 3 prior appearances); discovery 18,181, confirmation
15,148, carded rate 0.168 on confirmation. Team totals: 125,411 football-data matches with
yellow-card counts, 22 divisions, 2000–2026; discovery 60,861 (to 2017-11-30), confirmation
64,550.

**Against the proxy (confirmation).** All deltas are `log_loss(proxy) − log_loss(model)`;
positive is better.

| target | model | n | log-loss | AUC | delta vs proxy | 95% CI |
|---|---|---|---|---|---|---|
| player carded | climatology | 15,148 | 0.4528 | 0.500 | −0.0161 | [−0.0191, −0.0131] |
| player carded | P0 book proxy | 15,148 | 0.4367 | 0.636 | 0 | — |
| player carded | P1 event-only | 15,148 | 0.4309 | 0.658 | **+0.0058** | [+0.0036, +0.0080] |
| player carded | P2 matchup | 15,148 | 0.4304 | 0.661 | **+0.0064** | [+0.0042, +0.0084] |
| player carded | P3 + imputed state | 15,148 | 0.4302 | 0.661 | +0.0065 | [+0.0043, +0.0085] |
| match cards > 3.5 | C0 book proxy | 64,550 | 0.6648 | 0.632 | 0 | — |
| match cards > 3.5 | C1 event-only | 64,550 | 0.6585 | 0.639 | **+0.0063** | [+0.0051, +0.0075] |
| match cards > 3.5 | C2 event + market | 64,550 | 0.6581 | 0.641 | **+0.0067** | [+0.0055, +0.0080] |

Refit-noise floors (three seeds, confirmation log-loss): 0.0012 nats for the player models,
0.0002 at the 3.5 line. So P0 → P1 is ~5× the floor and real; **P1 → P2 (+0.0006) is inside
it** and the matchup hypothesis adds nothing detectable. The same picture holds at the 2.5 /
4.5 / 5.5 / 6.5 lines, and direct binary classifiers agree with the count-distribution route
to within 0.0017 nats.

**Scenario (confirmation).** −0.0208 [−0.0395, −0.0011] on the pre-registered player
scenario (n = 400); −0.0087 / −0.0223 / −0.0258 on the three team rungs. No room, twice.
(That interval is stage 01's own run of the cell; the sweep's uniform re-run in section 2.3
gives the same point estimate with a slightly wider interval, [−0.0409, −0.0006], because
the two bootstraps were drawn separately — see `05_gate_sweep.md` line 47 on why the stages
grew local copies of the loop.)

**Betting (confirmation).** Bet the side whose EV at the proxy's price exceeds a threshold
chosen on discovery (grid 0 … 0.40; the search picked the *top* of the grid in every card
configuration — the procedure saying "bet less and less").

| market | hold | threshold | n bets | bet rate | mean edge | ROI | 95% CI | after 0.306 haircut |
|---|---|---|---|---|---|---|---|---|
| player carded (0.5 line) | 0.04 | 0.40 | 822 | 5.4% | 0.593 | +0.239 | [+0.035, +0.455] | −0.050 [−0.255] |
| player carded (0.5 line) | 0.06 | 0.40 | 722 | 4.8% | 0.587 | **+0.265** | [+0.062, +0.485] | **−0.045 [−0.248]** |
| player carded (0.5 line) | 0.08 | 0.40 | 626 | 4.1% | 0.584 | +0.260 | [+0.033, +0.492] | −0.044 [−0.271] |
| match cards (proxy's line) | 0.04 | 0.40 | 551 | 0.85% | 0.483 | +0.344 | [+0.254, +0.433] | +0.054 [−0.035] |
| match cards (proxy's line) | 0.06 | 0.40 | 376 | 0.58% | 0.487 | **+0.305** | [+0.192, +0.416] | **−0.004 [−0.118]** |
| match cards (proxy's line) | 0.08 | 0.40 | 263 | 0.41% | 0.491 | +0.345 | [+0.216, +0.475] | +0.040 [−0.089] |

After-haircut columns are the **reference** haircut (margin-matched round trip on the card
pool: 0.2898 / 0.3099 / 0.3043 at 4 / 6 / 8%), read from `06_haircut_applied.parquet` — not
hand arithmetic, and not the flat mean-over-holds number the earlier draft applied. An
earlier draft also attributed a bet-rate-matched sensitivity of "0.294–0.373" to a reviewer;
no such table exists and the sentence has been removed. What is measured is in section 3.2:
the margin-matched and actual-prices definitions differ by 0.4 ROI points on this pool.

**No card ROI has a lower bound clear of zero after the haircut**, at any of the three holds;
the point estimates straddle it (−0.004 at the 6% headline, +0.04 to +0.05 at 4% and 8%),
which makes the card verdict "indistinguishable from zero on a very thin rule" rather than
"clearly negative". The 0.4–0.9% bet rates are why the intervals are this wide. The book-strength ladder
(6% hold, confirmation) is the cleaner statement:

| book's gain over the proxy (nats) | market | n bets | ROI | 95% CI |
|---|---|---|---|---|
| 0.0000 (the proxy itself) | match cards | 376 | +0.305 | [+0.192, +0.416] |
| 0.0045 | match cards | 114 | +0.222 | [+0.053, +0.391] |
| **0.0073** | match cards | 420 | **+0.005** | [−0.073, +0.082] |
| 0.0085 | match cards | 552 | −0.057 | [−0.119, +0.009] |
| 0.0034 | player props | 229 | +0.138 | [−0.218, +0.542] |

The team-card edge reaches zero at a book **0.0073 nats** sharper than a rolling mean and is
negative at 0.0085 — both *below* the 0.0089–0.0111 a real book demonstrably achieves. The
player-prop edge already has an interval spanning zero at 0.0034 nats. (Caveat the stage
states: the ladder's rungs re-choose their threshold on discovery, 0.40 / 0.30 / 0.12 / 0.02,
so they are not matched bet populations.)

**Why the simulated ROI exists at all.** On the bets placed, the model itself is
*over*-confident (it says 0.203, the truth is 0.165) — but the proxy says 0.123 and the mean
price is 9.30. The entire return comes from the proxy being badly wrong on long shots.

### 4.2 Corners (stage 02)

**Universe.** 105,378 of the 122,111 football-data matches recording corners (22 divisions,
2000–2026); discovery 63,227 (to 2019-08-31), confirmation 42,151. The corner rate drifts
from 10.55 per match in discovery to 9.77 in confirmation, which is why every model is
specified as a correction to the proxy rather than free-standing.

**The match total: nothing.** The proxy (multiplicative, shrinkage k = 40, chosen on
discovery from 18 cells) explains R² 0.033 discovery / 0.024 confirmation. Adding Elo, form,
rest, prior shots and fouls *and* the no-vig 1X2 and Over-2.5 prices is worth:

| target | n (confirmation) | delta vs proxy | 95% CI | refit spread |
|---|---|---|---|---|
| match total (Poisson deviance) | 42,151 | +0.0017 | [+0.0002, +0.0032] | 0.00036 |
| over 8.5 | 42,151 | +0.0003 | [−0.0002, +0.0008] | — |
| over 9.5 | 42,151 | +0.0001 | [−0.0003, +0.0006] | 0.00010 |
| over 10.5 | 42,151 | +0.0004 | [−0.0001, +0.0008] | — |

Every line interval straddles zero and every point estimate is 20–30× below the haircut.
Capacity was the hinge and was settled on an inner forward split *inside* discovery: a
conventional 400-tree/31-leaf LightGBM **loses to the proxy outright** (0.6838 vs 0.6795 at
9.5), while the chosen 100-tree/7-leaf model wins (0.6786). Without that step this null
could not be told apart from a capacity artefact.

**The team line: the one real signal in the phase, and what it is made of.**

| model | n (confirmation team-matches) | delta vs proxy (Poisson deviance) | 95% CI |
|---|---|---|---|
| C1_full (market + events) | 84,302 | **+0.0596** | [+0.0560, +0.0630] |
| C1_market_only | 84,302 | +0.0569 | [+0.0534, +0.0603] |
| C1_events_only | 84,302 | +0.0478 | [+0.0446, +0.0508] |
| C1_full at the 5.5 team line (nats) | 84,302 | **+0.0117** | [+0.0108, +0.0125] |
| C1_full at the 4.5 team line (nats) | 84,302 | +0.0115 | [+0.0106, +0.0123] |

The team line now has a refit-noise floor of its own (`02_corners_team_refit.parquet`, three
seeds, confirmation): **0.000081 nats at the 5.5 line** and 0.00030 Poisson deviance on the
team count. The margin is ~144× that floor, so it is real as a measurement. An earlier draft
printed 0.0001 in this column, which was the *match* model's floor (`C1_offset` at the 9.5
match line) standing in for a model it does not describe; the two happen to be close, but
that was a labelling error, not a measurement.

R² rises 0.082 → 0.120, and the CLV table is monotone: in the top-disagreement bin
(n = 16,860) the model beats the proxy by +0.0476 nats. **This is the only margin in the
phase larger than the haircut in nats** (+0.0117 against +0.0089). A team's corner count
follows the game script; the match total is nearly conserved between the two sides and
barely moves.

But the ablation locates it: the market block alone carries 95% of the full model's gain.
A rolling corner mean is blind to who the market thinks will be on top; a real bookmaker is
not, because the match odds are the most visible input on the screen. So this measures the
*proxy's* weakness, not a market inefficiency — and the betting simulation closes it:

| market | hold | n bets | bet rate | mean edge | ROI | 95% CI | − lower bound (one-way) | **− reference** | ref. lower bound |
|---|---|---|---|---|---|---|---|---|---|
| match total | 0.04 | 10,133 | 24.0% | 0.052 | +0.014 | [−0.004, +0.035] | −0.115 | **−0.160** | −0.178 |
| match total | 0.06 | 5,976 | 14.2% | 0.047 | +0.014 | [−0.011, +0.039] | −0.111 | **−0.193** | −0.218 |
| match total | 0.08 | 3,174 | 7.5% | 0.044 | +0.014 | [−0.021, +0.050] | −0.108 | **−0.214** | −0.249 |
| team total | 0.04 | 51,842 | 61.5% | 0.119 | +0.138 | [+0.129, +0.147] | +0.009 | **−0.036** | −0.045 |
| team total | 0.06 | 43,833 | 52.0% | 0.114 | **+0.132** | [+0.123, +0.142] | +0.008 | **−0.075** | −0.084 |
| team total | 0.08 | 36,684 | 43.5% | 0.110 | +0.130 | [+0.120, +0.141] | +0.009 | **−0.098** | −0.109 |

Threshold 0.02 throughout, count-NB source, confirmation; "one-way" is the lenient
`book_earns_one_way` lower bound this stage originally used (0.1295 / 0.1246 / 0.1215) and
"reference" is the phase's margin-matched round trip (0.1736 / 0.2073 / 0.2282). All rows
from `06_haircut_applied.parquet`.

**The earlier draft's "+0.008 residual, indistinguishable from zero" was an artefact of the
lenient definition.** Under the phase reference the team total on all rows is
**−0.075 [−0.084]** — clearly below zero, not marginally above it. An earlier draft also
attributed a correction of "+0.005 rather than +0.008" to a reviewer; no such correction
exists in `02_corners.md` or its tables and the sentence has been removed.

**Scenario — and the largest after-haircut number in the phase.** All eight pre-registered
corner scenarios have negative discovery deltas for the match total (−0.0572 to −0.0024) and
negative confirmation deltas, so **no scenario had positive discovery room for the match
total and there was nothing genuinely to pre-register.** The `fav_strong` team-level cell is
the single gate cell in 38 with confirmation room (+0.0085 in the stage's own run, +0.0054
[+0.0032, +0.0074] in the uniform sweep), and it is disclaimed by the stage itself because
its discovery counterpart is +0.0012 [−0.0027, +0.0054]. Its betting simulation was in
`02_corners_bets.parquet` all along and did not reach the earlier draft of this synthesis.
It should have, because it is precisely the question the phase was asked — one stat, one
scenario, +EV — on the one cell the gate said had room:

| scenario | n rows | n bets | bet rate | mean edge | ROI | 95% CI | − one-way | **− reference** | ref. lower bound |
|---|---|---|---|---|---|---|---|---|---|
| `team_total × fav_strong` | 9,272 | 9,267 | **99.95%** | 0.208 | **+0.262** | [+0.242, +0.281] | +0.137 | **+0.055** | **+0.035** |
| `team_total × high_total` | 9,990 | 8,471 | 84.8% | 0.184 | +0.221 | [+0.201, +0.242] | +0.097 | +0.014 | −0.007 |
| `team_total × fav_strong_low_total` | 978 | 969 | 99.1% | 0.133 | +0.158 | [+0.092, +0.222] | +0.033 | −0.050 | −0.115 |
| `match_total × fav_strong` | 4,636 | 2,502 | 54.0% | 0.056 | +0.015 | [−0.024, +0.055] | −0.109 | −0.192 | −0.232 |

6% hold, 0.02 threshold, count-NB, confirmation. **`team_total × fav_strong` is the only
corner row whose lower bound clears zero after the reference haircut at the headline hold**
(it also clears at 4%, +0.112 [+0.092]; at 8% it is +0.013 [−0.007]). Across the whole phase
9 of 48 published confirmation betting rows keep a lower bound above zero after the
reference haircut: these two, `team_total × high_total` at 4% only (+0.063 [+0.043]), and six
pass-count rows at the 4% and 6% holds (section 4.3). The full list is in `06_haircut.md`
section 5.

Three things say it is the proxy's blindness rather than an edge, and none of them is the
haircut:

* **The bet rate is 99.95%.** The rule fires on 9,267 of 9,272 rows. That is not a selective
  edge in a scenario; it is the model and the proxy disagreeing about *every* strong-favourite
  match, which is what you would expect from an opponent that cannot see who the favourite is.
* **95% of the team model's margin is the market block** (`C1_market_only` +0.0569 of
  `C1_full`'s +0.0596 Poisson-deviance gain). The one input carrying it is the 1X2 price —
  the most visible number on a real trading screen, and the one input a real corner book
  certainly has.
* **The discovery half is +0.199 [+0.182, +0.217] against the confirmation +0.262**, and the
  gate delta that selected the cell was an order of magnitude smaller on discovery than on
  confirmation. The cell is the expected false positive at 38 tests; its ROI is stable across
  halves because the proxy's blindness is stable, not because an edge was pre-registered.

The honest summary is: *against a bookmaker who prices team corners off a rolling mean and
ignores the match odds, this is worth +5.5% after the strictest haircut we can measure.* No
such bookmaker is known to exist.

**Style (StatsBomb overlap, 1,517 matches joined exactly by fixture-date bipartite matching;
scores agree on 100%).** The hypothesis "teams that cross heavily, against compact
opponents, generate corners above their average" splits in two and only one half survives:
style alone beats the rolling mean (+0.0270 [+0.0119, +0.0428] CV) and imputed defensive
state alone does too (+0.0175 [+0.0065, +0.0282] CV), but on top of the odds table the
team's **own** style adds +0.0029 CV / +0.0071 forward, the **opponent's** style −0.0012 /
−0.0064, and the opponent's imputed compactness a further +0.0007 / +0.0013. The layer is
one season of four leagues, 527 confirmation matches, with forward intervals wide enough to
contain both zero and twice the effect.

### 4.3 Player pass counts (stage 03)

This was the candidate with a real prior: the completed programme's one positive
imputed-state result, +0.0026 nats per pass. Three independent links in the chain break.

**(1) The motivating per-pass gain does not survive a bigger training set.** Refitting the
identical xPass variants — same folds, same LightGBM settings, same pre-instant `E2`
design — on all 371,844 pass attempts of the 417 360-matches instead of the programme's
149,591-pass subsample, and scoring on **exactly the same 149,591 rows**:

| source | variant | n | delta vs EVENT (nats/pass) | 95% CI |
|---|---|---|---|---|
| programme cache | EVENT+IMP | 149,591 | **+0.00162** | [+0.00104, +0.00220] |
| refit on all attempts | EVENT+IMP | 149,591 | **+0.00017** | [−0.00031, +0.00066] |
| programme cache | EVENT+ORACLE | 149,591 | +0.07032 | [+0.06829, +0.07247] |
| refit on all attempts | EVENT+ORACLE | 149,591 | +0.07010 | [+0.06803, +0.07236] |

The oracle gap reproduces to four decimals, so the pipeline is sound — **the event-only
model simply catches up once it is given the passes the subsample discarded.**

How far the seed is separated from the training size, stated exactly, because an earlier
draft of this document claimed a four-seed control that does not exist and was never run.
What is measured is this: the two rows differ in *both* training population and seed
(stage 03's per-pass refit is single-seed, matching the programme's own `n_seeds_xpass = 1`,
and `03_pass_a_per_pass.parquet` has no seed column). That same joint change moves the
**oracle** variant's delta by 0.00022 (+0.07032 → +0.07010) and the **imputed** variant's by
0.00145 (+0.00162 → +0.00017). A seed effect large enough to explain the second would have
to be six times smaller on the first, for no reason anyone has offered. That is strong
circumstantial evidence for the training-size reading, and it is not a controlled
measurement; `03_pass_counts.md` says the same in its caveats, and a multi-seed per-pass
refit remains the missing experiment.

The *sequence* student's channel does survive at full size: +0.0019 [+0.0013, +0.0024] nats
per pass on its 251 matches (n = 210,538).

**(2) Aggregating to a count destroys ~90% of per-pass skill.** With realised attempts
given, the perfect 360 frame is worth 0.0703 nats/pass, which over the 29.6 attempts of an
average player-match sums to 2.079 nats — but the exact Poisson-binomial completed-pass
distribution improves by only 0.197 nats, a retention of **9.5%**. A count throws away
*which* passes failed, and that is where the information is.

**(3) At a plausible line the imputed state moves the price a lot and improves it not at
all.** Confirmation, n = 5,038, mean line 23.0 completions, over rate 0.530:

| variant | mean absolute shift in P(over) vs EVENT | max shift | delta Brier | 95% CI |
|---|---|---|---|---|
| EVENT+IMP | 4.8 pp | 34 pp | −0.00074 | [−0.00235, +0.00089] |
| EVENT+IMP+SEQ (its own 251-match population) | 5.9 pp | 31 pp | +0.00219 | [−0.00081, +0.00496] |
| EVENT+ORACLE (the true frame) | 17.5 pp | 56 pp | **+0.05194** | [+0.04686, +0.05722] |

A 5-point move at an even-money line is commercially enormous *if it is information*; here
it is noise. Only the true 360 frame both moves and improves.

**The structural ceiling.** Of the variance of completed passes (confirmation, sd 17.7,
n = 13,053) only **2.2%** is the trial-by-trial Bernoulli term a completion model can touch;
97.8% is attempt volume and the player-match rate level. The end-to-end channel split
agrees: model attempts × proxy rate +0.0724 [+0.0627, +0.0831]; proxy attempts × model rate
+0.0213 [+0.0182, +0.0246]; both +0.0771 [+0.0657, +0.0897]. **Volume is the market.**

**The volume model against the proxy (confirmation, n = 13,053):** +0.0630 [+0.0548,
+0.0720] nats per player-match on attempts (R² 0.461 → 0.541) and +0.0701 [+0.0603, +0.0812]
on completed passes. The nested ablation puts the whole advantage in public features —
player rolling +0.0137, + position/venue +0.0293, + team/opponent rolling +0.0630, +
imputed state +0.0638 — and the imputed block depth contributes +0.00081 [−0.00091,
+0.00251] against a 3-seed refit spread of 0.00149.

**Betting (confirmation).** Against a proxy-priced market:

| hold | n bets | bet rate | mean edge | ROI | 95% CI | − one-way (0.1162) | **− reference** | ref. lower bound |
|---|---|---|---|---|---|---|---|---|
| 0.04 | 10,405 | 79.7% | 0.278 | +0.266 | [+0.243, +0.290] | +0.150 | **+0.092** | **+0.070** |
| 0.06 | 9,815 | 75.2% | 0.267 | **+0.250** | [+0.227, +0.274] | +0.134 | **+0.043** | **+0.020** |
| 0.08 | 9,201 | 70.5% | 0.257 | +0.239 | [+0.216, +0.264] | +0.123 | +0.011 | −0.013 |

Threshold 0.05, `model attempts × model rate`, confirmation. **This residual is not killed
by the haircut and this document does not claim it is.** At the 4% and 6% holds its interval
lies entirely above zero under the phase's strictest measured definition (+0.092 [+0.070] and
+0.043 [+0.020]); only at 8% does the lower bound cross. An earlier draft of section 0 said
"no market's ROI interval clears zero" after the haircut, which was contradicted by this
table three sections later. It is the largest surviving residual in the phase, and what
argues against it is three separate calibrations rather than the haircut. **In line units**, our model's
mean differs from the proxy's by 3.84 completed passes (21.8% of the outcome's sd) and would
post a different line in 90% of player-matches, while a real bookmaker's implied mean total
goals differs from the identical proxy by 0.200 goals (12.3% of that outcome's sd),
differing by half a line in 6.3% of matches: we claim to disagree with a rolling mean about
1.8× as hard as a real book does. **Against a book that sets its own line** at κ = 0.565 of
the way from the proxy to our model — the factor matching the goals-market disagreement —
ROI falls to +0.098 / **+0.090 [+0.064, +0.118]** / +0.095 at 4 / 6 / 8% hold, before any
haircut. (The κ ladder and the haircut are alternative ways of modelling a sharper opponent,
not to be composed: subtracting the haircut from the κ-ladder number would charge the same
sharpness twice.) That residual rests on assuming a real pass-prop book knows only 56% of
what a model built from public rolling averages knows, which the third calibration — the
feature ablation — makes implausible: every step of the advantage is a public rolling
average (player rolling +0.0137, + position/venue +0.0293, + team/opponent rolling +0.0630),
so there is no private information in this model and a trading desk computes all of it. The
stage's own caveat cuts the other way: the goals proxy is opponent-adjusted while the pass
proxy is not, which biases this transfer *toward* the model.

**One instrument points the other way, and is reported rather than suppressed.**
`03_pass_c_relative_scale.parquet` normalises each market by its own proxy's score. On goals
a real book beats the proxy by 1.54% of the proxy's log loss; our pass model beats its proxy
by **2.00%** of its count log score. Scaled that way, a book that were only
*goals-equivalently* sharp on pass props would post a count log score of 3.7908 against our
model's 3.7732 — i.e. it would be **0.0176 nats per player-match worse than us**, not
better. Stage 03 computes this for completeness and explicitly declines to use it
(`03_pass_counts.md` line 464): a count log score is dominated by the distribution's
irreducible entropy, so a relative gap there is not comparable to a binary log loss at a
line. That is the right call, and it is why this number is not the phase's instrument — but
a reader is entitled to know that the one normalisation putting the two markets on a common
relative scale favours the model, and that this synthesis is not selecting only the
instruments that agree with its conclusion.

**Scenario.** All five pre-registered scenarios fail on confirmation, by −0.0103 to −0.0054
nats, including the one the discussion proposed ("the usual deep-lying passer is absent").

### 4.4 Fouls (stage 04)

**Universes.** Odds table: 210,076 team-rows (105,038 matches) recording fouls; discovery
126,046 team-rows, confirmation 84,030. Player layer: the same 33,329 StatsBomb appearances
as the cards stage; discovery 19,359, confirmation 13,970.

**A rolling mean is already good at fouls, unlike corners.** The proxy explains R² 0.2188 of
a single team's foul count on confirmation, against 0.024 for corners — foul rates are
dominated by persistent competition and officiating effects a rolling mean captures by
construction. Division effects are large (11.1% of the variance of a team's fouls between
divisions on confirmation) **and already absorbed**: adding division identifiers to a model
that has the proxy and the rolling block is worth +0.0000 [−0.0003, +0.0003].

**Against the proxy (confirmation, negative-binomial log score, dispersion fitted on
discovery):**

| target | n | delta vs recalibrated proxy | 95% CI | refit spread (5 seeds) |
|---|---|---|---|---|
| team fouls | 84,030 | **+0.0055** | [+0.0046, +0.0064] | 0.00019 |
| match total fouls | 42,015 | **+0.0086** | [+0.0072, +0.0100] | — |

~29× the noise floor, so real — and **smaller than the +0.0090 [+0.0079, +0.0100] a real
book holds over the same proxy on goals**. At the half-lines the count route and direct
binary classifiers agree within a whisker (team fouls 12.5: +0.0036 vs +0.0033).

One honest caution the ladder makes plain: most of the gap to the *player* proxy is
functional form, not information. On player fouls, LightGBM on the proxy's own inputs
already beats the calibrated proxy by +0.0501 nats, and the entire event, referee and
matchup apparatus adds +0.0006 more. A rate times expected minutes is the wrong shape for a
count, and a booster fixes that before it learns anything about football.

**The referee channel is real but is mostly a league effect.** Appearances under a
top-tercile referee average 1.2993 [1.2770, 1.3228] fouls against 1.0253 [1.0027, 1.0463] in
the bottom tercile — but the referee prior's R² on a team's fouls falls from 0.0955 to
0.0247 once competition means are removed, and adding the referee block to the player model
is worth +0.0001 [−0.0005, +0.0007] on fouls committed and +0.0011 [+0.0001, +0.0022] on
fouls won (n = 13,970 confirmation; stripping the competition feature first leaves
+0.0008 [+0.0001, +0.0015] on fouls won and +0.0001 [−0.0006, +0.0008] on fouls committed). The tercile cuts, ~28 and ~31 fouls per match, are very nearly the difference between
the Premier League and Serie A.

**The matchup asymmetry** is the phase's most interesting surviving signal and is set out in
section 2.4 with its caveats.

**Betting (confirmation).** The proxy posts the rung closest to its own mean; threshold 0.08
chosen on discovery. `haircut_roi_points` is the stage's margin-matched goals haircut.

Stage 04's `haircut_roi_points` **is** the phase reference definition — it is the stage that
invented it — so these numbers are unchanged by the reconciliation in section 3.2.

| market | hold | n bets | bet rate | mean edge | ROI | 95% CI | haircut (reference) | after haircut | CI |
|---|---|---|---|---|---|---|---|---|---|
| team fouls | 0.04 | 21,137 | 25.2% | 0.141 | +0.112 | [+0.098, +0.125] | 0.1736 | −0.062 | [−0.075, −0.049] |
| team fouls | 0.06 | 15,304 | 18.2% | 0.139 | **+0.109** | [+0.094, +0.124] | 0.2073 | **−0.098** | [−0.113, −0.083] |
| team fouls | 0.08 | 11,045 | 13.1% | 0.136 | +0.096 | [+0.078, +0.114] | 0.2282 | −0.132 | [−0.150, −0.115] |
| match fouls | 0.04 | 13,349 | 31.8% | 0.152 | +0.137 | [+0.120, +0.152] | 0.1736 | −0.037 | [−0.053, −0.021] |
| match fouls | 0.06 | 10,238 | 24.4% | 0.148 | **+0.139** | [+0.122, +0.157] | 0.2073 | **−0.068** | [−0.086, −0.050] |
| match fouls | 0.08 | 7,690 | 18.3% | 0.146 | +0.140 | [+0.120, +0.161] | 0.2282 | −0.088 | [−0.108, −0.068] |

**Both markets' after-haircut intervals lie entirely below zero** — the clearest negative in
the phase, and the one that does not depend on a marginal judgement. The book-strength
ladder agrees: at a book only a quarter of the way from the proxy to our model, team-foul
ROI falls to +0.070 [+0.048, +0.092] and at half way to +0.026 [−0.052, +0.108] on 442 bets.

**Scenario.** Five cells, no room or inconclusive in all five; the pre-registered
defender × flank cell is −0.0212 [−0.0292, −0.0124] on 2,176 confirmation rows.

### 4.5 The four candidates side by side

All haircuts here are the phase reference (section 3.2) at the 6% headline hold: 0.3099 on
the card pool, 0.2073 on the corner/foul pool. The refit floors are each headline model's
own, which is not the same quantity as the gate sweep's per-market floors in section 2.1 —
those come from the sweep's own fits and are used only to downgrade a positive gate cell.

| | cards (team, 3.5) | corners (team, 5.5) | pass counts (attempts) | fouls (team) |
|---|---|---|---|---|
| margin over the proxy (confirmation) | +0.0067 nats | +0.0117 nats | +0.0630 nats/player-match ‡ | +0.0055 nats |
| 95% CI | [+0.0055, +0.0080] | [+0.0108, +0.0125] | [+0.0548, +0.0720] | [+0.0046, +0.0064] |
| n | 64,550 | 84,302 | 13,053 | 84,030 |
| refit floor (this model, 3 seeds) | 0.0002 | 0.000081 | 0.0015 | 0.0002 |
| that pool's goals haircut (nats) | 0.0111 | 0.0090 | 0.0090 | 0.0090 |
| margin > haircut in nats? | **no** | yes (but 95% market) | not comparable ‡ | **no** |
| ROI vs proxy, 6% hold | +0.305 | +0.132 | +0.250 | +0.109 |
| **ROI after the reference haircut** | **−0.004 [−0.118]** | **−0.075 [−0.084]** | **+0.043 [+0.020]** | **−0.098 [−0.113]** |
| best scenario cell, after the same haircut | — | **+0.055 [+0.035]** (`fav_strong`, 99.95% bet rate) | — | — |
| gate verdict | no room | 1 of 5 cells, disclaimed | no room | no room |

‡ the pass-count margin is a count log score per player-match, not a binary log loss at a
line, and a count log score is dominated by irreducible entropy — the two are not on the
same scale and the stage says so explicitly. Its line-based instruments are the κ ladder and
the book-sets-its-own-line simulation quoted in 4.3.

Read the bold row with section 4.2 and 4.3 beside it: two of the six after-haircut cells are
positive with lower bounds clear of zero, and both are argued away by evidence other than
the haircut — a 99.95% bet rate and a 95%-market ablation for the corner cell, a public-only
feature ablation and a 1.8× disagreement ratio for the pass cell.

---


## 5. Where the teacher-student / imputed-state channel enters, and whether it ever helps

The completed programme's students (out-of-fold `E2` LightGBM predictions of 360-frame
state: nearest-opponent distance, opponents within 5 m, block depth, counter-on, and the
soccer-06 sequence student) were offered to all four candidates as prior-match aggregates.

Coverage was not the limitation *where the channel could be offered at all*: 94.4% of the
**4,180 StatsBomb team-matches** have a usable state prior (3,947 of 4,180,
`01_cards_a_state_coverage.parquet`), and within the card and foul player universes every
row has one for both teams (100.0% in stage 04). But that denominator is the StatsBomb
overlap, not the odds table. The students only exist for the 1,398-match StatsBomb club
seasons, so the channel could never be offered on the 105k-match universes where every
betting conclusion in this document lives — 84,302 confirmation team-corner rows, 84,030
team-foul rows, 64,550 match-card rows. The single exception is stage 02's corner style
layer, whose forward-confirmation half is 527 matches and is underpowered by its own
account. **The imputed-state null is established at player and team level on one season of
four leagues; it is untested at odds-market scale.**

| candidate | where the imputed state was added | delta (nats) | 95% CI | n | split |
|---|---|---|---|---|---|
| cards | P2 matchup → P3 + imputed defensive state (player carded) | +0.0001 | [−0.0007, +0.0009] | 15,148 | confirmation |
| corners | opponent imputed compactness on top of odds + both styles | +0.0007 CV / +0.0013 fwd | contains zero | 2,634 / 1,054 | cv_all / confirmation |
| corners | imputed state **alone** vs the rolling mean | +0.0175 | [+0.0065, +0.0282] | 2,634 | cv_all |
| pass counts | imputed block depth on the attempts model | +0.00081 | [−0.00091, +0.00251] | 13,053 | confirmation |
| pass counts | EVENT → EVENT+IMP per pass, refit at full training size | +0.00017 | [−0.00031, +0.00066] | 149,591 | cv_all |
| pass counts | EVENT → EVENT+IMP at a 23.0-completion line (Brier) | −0.00074 | [−0.00235, +0.00089] | 5,038 | confirmation |
| fouls | P2 → P3 + imputed state, fouls committed | −0.00016 | [−0.00057, +0.00025] | 13,970 | confirmation |
| fouls | P2 → P3 + imputed state, fouls won | +0.00007 | [−0.00034, +0.00050] | 13,970 | confirmation |

**Eight tests, in four markets, and not one *incremental* test clears its refit-noise
floor.** The one row that does clear — imputed state **alone** against the rolling corner
mean, +0.0175 [+0.0065, +0.0282] — is not incremental, and is the subject of the first
bullet below. That is
the completed programme's null reproduced four more times, on outcomes it had never
touched — and reproduced in the shape the programme predicted. Two observations sharpen it:

* **The channel is real, it is just not incremental.** Imputed state *alone* beats a rolling
  corner mean (+0.0175 [+0.0065, +0.0282]). It stops being worth anything the moment the
  model already reads the events the students were trained on. This is the completed
  programme's exact finding, now at the level of side markets rather than xG.
* **The one place the oracle is worth a great deal, the student cannot reach.** In pass
  counts the *true* 360 frame improves the at-the-money price by +0.0519 Brier [+0.0469,
  +0.0572] and shifts it 17.5 pp; the imputed version shifts it 4.8 pp and improves it by
  nothing. That is the inverse relationship between oracle value and student recoverability,
  observed one more time — and now with a second mechanism layered on top, the **9.5% count
  retention** of section 4.3: even a perfect per-pass model loses 90% of its value the
  moment the market is a count rather than a single event.

And the phase found one thing the programme's framing did not anticipate: the motivating
per-pass gain itself (+0.00162 [+0.00104, +0.00220]) **does not survive being refit on the
full pass population** (+0.00017 [−0.00031, +0.00066] on the identical evaluation rows).
That is a correction to the completed programme, not just to this phase, and it should be
carried back: the LightGBM-imputation gain published in soccer 03 was a small-training-set
artefact. The *sequence* student's channel is the one that survives full training size
(+0.0019 [+0.0013, +0.0024] nats/pass on 251 matches).

---

## 6. Verdict: is any of this worth betting real money on?

No, and the reason is specific rather than general. Our models are not weak: they beat a
shrunk, tuned, recalibrated rolling mean on every market tested, at sample sizes of 13,000
to 84,000 confirmation rows, with margins roughly 5× to 140× their own refit-noise floors,
and
they are closer to the truth than the proxy in every bin where the two disagree most. What
sinks the EV case is that the only opponent we can construct is that rolling mean, and the
one time we can measure the real opponent — the goals line, priced by Bet365 on the same
matches — the real book beats the rolling mean by 0.0089–0.0111 nats, which is *more* than
our card (0.0064–0.0067), foul (0.0055) and match-corner (0.0001) models beat it by.

**What the haircut settles, and what it does not.** Charging the phase's single reference
definition (section 3.2), the card ROI is −0.004 [−0.118], team fouls −0.098 [−0.113], match
fouls −0.068 [−0.086] and the match corner total −0.193 [−0.218]. Those four are settled.
Two rows are not:

* **Team corner lines inside `fav_strong`: +0.055 [+0.035]** after the reference haircut,
  9,267 bets. The haircut does not kill it. What does: the rule bets 99.95% of the scenario's
  matches, and 95% of the model's margin is the 1X2 price the proxy cannot see. This is the
  proxy's blindness measured at its most extreme, not an inefficiency in any real market. On
  *all* rows the same model is −0.075 [−0.084], so the "edge" is entirely the sub-population
  where the constructed opponent is worst.
* **Player pass counts: +0.043 [+0.020]** after the reference haircut at 6% (+0.092 [+0.070]
  at 4%, +0.011 [−0.013] at 8%). The haircut does not kill this one either. What does: every
  feature carrying the advantage is a public rolling average; the model claims to disagree
  with a rolling mean 1.8× as hard as a real book does on goals; and against a book allowed
  to set its own line at the goals-matched disagreement the ROI falls to +0.090 [+0.064,
  +0.118] before any haircut. One instrument — the relative-scale normalisation in 4.3 —
  points the other way, and stage 03 explains why it should not be used. That is an argument,
  not a measurement, and it is offered as one.

Meanwhile the scenario premise, which is what the phase was actually asked about, fails
independently and much harder: 1 room in 38 pre-registered cells, 21 clear losses, the
penalty growing as the scenario narrows, and no imputed-state channel clearing noise
anywhere. The one condition under which some of this becomes tradeable is stated exactly, and
a reader can check it against their own beliefs: **if a real card book is less than 0.0073
nats sharper than a rolling mean (0.0034 for player card props), the card edge is positive**;
below that the ladder is interval-zero. Since the observed number for the market a book works
hardest on is 0.009–0.011, and side markets carry wider margins, thinner limits, line
movement and availability constraints that none of these simulations model — all of which
cut against the edge — the answer under any assumption we can defend is that there is no
demonstrated edge here. The two surviving residuals are the honest qualification on that
sentence, and both would be settled by a few hundred observed side-market lines.

---


## 7. Limitations, and what would settle the open questions

**The limitation that dominates all others.** There are no corner, card, foul or pass-prop
lines in this data. Every "book" is simulated, and the distance to a real one is estimated
by analogy from goals — the single most efficient market a book prices, so the transfer
probably *overstates* a side-market book's sharpness, while side-market margins are wider
than the 5.8–6.6% observed on the goals line, which *understates* the bar. Neither
correction is measurable here. The book-strength ladder exists so a reader can substitute
their own assumption.

Also:

* **The haircut was defined three different ways across the four stages** (section 3.2), from
  0.116 to 0.310 ROI points at a 6% hold, and **which one a stage happened to use decided
  whether its market survived.** That is now fixed: `haircut.py` names the margin-matched
  round trip the phase reference and applies all three uniformly
  (`06_haircut_applied.parquet`). It is not true, as an earlier draft claimed, that every
  conclusion survives either definition — the corner team line is +0.008 under the lenient
  one and −0.075 under the reference. Across all 48 published confirmation betting rows, 33
  have a positive point estimate under the lenient definition against 18 under the
  reference, and 21 keep a lower bound above zero under the lenient one against 9 under the
  reference. The nats version (0.0089–0.0111) remains the
  robust, comparable quantity; the cross-market transfer of a flat ROI subtraction is still
  crude, especially across bet rates that differ by two orders of magnitude.
* **The reference haircut differs by pool** (0.3099 on the card pool, 0.2073 on the
  corner/foul pool) and the difference is mostly era, not market. A single phase-wide number
  would be tidier and would also be wrong.
* **Player-level results are one season of four leagues** (1,398 StatsBomb matches; 33,329
  appearances). Cards part (a), fouls part (b) and the pass volume channel all live there.
  Nothing player-level is tested across seasons or competitions; the card *mechanism* is the
  exception and does transfer to 14,994 held-out tournament fouls.
* **The card and foul player universes condition on having played** (`minutes > 0`,
  `cards.py:879`), which a real prop cannot: it is priced before the team sheet is certain,
  and a late withdrawal is a void bet, not a zero. No EV claim depends on this, because the
  card and foul betting simulations are team- and match-level, but those player-level nats
  are conditional on playing. **This does not apply to the pass stage**, contrary to an
  earlier draft: `pass_counts.league_population` is explicitly not filtered on realised
  minutes, because minutes are an outcome of the match and an early substitution is exactly
  the risk the market prices. (A dead `PassConfig.min_minutes` knob documented a filter that
  was never applied; it has been removed.)
* **Starting position comes from the match's own line-up** in the pass stage. A book pricing
  before team news would not have it; the F1 → F2 ablation step (+0.0157 nats, which also
  bundles venue and competition) bounds what it is worth.
* **Thresholds sit at the top of their grid** in every card configuration (0.40), so the
  surviving card rules bet 0.4–0.9% of matches (263–551 confirmation bets). These are thin,
  tail-dwelling rules and exactly the region where a strawman book is most wrong and a real
  one is not.
* **The corner style layer is underpowered**: 527 confirmation matches, with forward
  intervals wide enough to contain both zero and twice the effect, so its null on opponent
  compactness is weak evidence rather than strong.
* **The fouls-won channels are confirmation-only clearances** from a tested family of 12
  confirmation increments over two targets, not four, whose discovery halves agreed in sign
  but not in significance. The opponent-foul channel is ~3× its five-seed spread at both its
  headline seed and its seed mean; the dribble-side channel is 1.7× at its headline seed but
  only 1.3× at the seed mean and 0.8× at its weakest seed, so it should be read as suggestive
  rather than as surviving. Both are the phase's most interesting signals and the ones most
  in need of independent replication.
* **A three-seed refit floor can understate a spread by an order of magnitude.** Stage 04
  found one channel increment spanning 0.00007 nats over three seeds and 0.00078 over five.
  Stages 01–03 and the gate sweep use three seeds; the sweep's floor is therefore used only
  to *downgrade* a positive cell, never to promote one.
* **Data coverage**: corners and fouls are recorded on ~51% of the odds table, yellows on
  53%; the corner rate drifts 10.55 → 9.77 per match across the split boundary, and
  football-data foul counts carry known feed differences between divisions and eras.
* **The completed programme's per-pass result needed correcting** (section 5). Any future
  work quoting soccer 03's +0.00162 should quote the full-training-size +0.00017 instead —
  with the caveat stated in 4.3 that the refit changed the seed as well as the training
  population, and the separation rests on the oracle variant moving 6× less under the same
  joint change rather than on a multi-seed control, which was never run.
* **The two surviving after-haircut residuals are argued away, not measured away.** The
  corner `fav_strong` cell and the pass-count residual both keep lower bounds above zero
  under the strictest haircut this data can support. The arguments against them (bet rate,
  market ablation, feature ablation, disagreement ratio) are good ones and one instrument
  disagrees with them (4.3), but a reader who rejects the arguments is left with two positive
  results and should know it.
* **`06_haircut.md`'s universe analysis is reconstructed, not read back from the stages.** It
  rebuilds each stage's stated filter from the raw odds table, which reproduces the pools but
  not the prior-history filters, so its pool sizes exceed the stages' scored n by a few
  thousand rows each. The nesting relations, which are what the argument rests on, are not
  sensitive to that.

### What would actually settle this

1. **Historical prop lines and, above all, closing lines** for cards, corners, fouls and
   player props. Everything in this phase is a bound derived by analogy; one season of real
   corner/card lines with opening and closing prices would replace the entire haircut
   apparatus with a measurement, and closing-line value against a real close is the only
   honest test of an edge. This is the single highest-value missing input.
2. **A book-quality measurement on a side market.** Even without prop history, a few hundred
   observed card or corner lines would locate the real book on the ladder of section 4.1 and
   turn "if a card book is less than 0.0073 nats sharper than a rolling mean" into a fact.
3. **Independent replication of the fouls-won matchup channel** on another season or league,
   pre-registered from this stage's confirmation result, with five or more seeds.
4. **A proxy given the 1X2 prices**, then asking what is left. Stage 02's team-line result
   is the phase's one large margin and it is almost entirely the match odds; handing the
   proxy those odds and re-measuring is a one-line experiment that would say whether
   anything remains.
5. **Multi-season player data** for the prop-level questions, and pricing the market as a
   book does — before team news, with void rules — rather than conditional on the appearance.

---

## 8. Reproduction

Commands verified against each module's `argparse`. Run from the repository root.

```shell
cd /home/user/geo-model

# Stage 01 -- cards (--stage build|a|b|c|all, plus --force to rebuild intermediates)
python -m research.scenario_ev.cards --stage build --force   # ~15 s
python -m research.scenario_ev.cards --stage a               # 2-4 min   player card props
python -m research.scenario_ev.cards --stage b               # 5-12 min  foul -> card mechanism
python -m research.scenario_ev.cards --stage c               # 7-15 min  team totals + the goals anchor

# Stage 02 -- corners (--stage all|build|audit|proxy|tune|honesty|models|team|gate|bets|style|report,
#                      plus --force-build and --seeds N)
python -m research.scenario_ev.corners --stage all

# Stage 03 -- pass counts (--stage all|passpreds|a|b|goals|c|report, plus --force and --smoke N)
python -m research.scenario_ev.pass_counts --stage passpreds  # ~6 min, 2 threads
python -m research.scenario_ev.pass_counts --stage a          # ~20 s
python -m research.scenario_ev.pass_counts --stage b          # ~2 min
python -m research.scenario_ev.pass_counts --stage goals      # ~15 s
python -m research.scenario_ev.pass_counts --stage c          # ~40 s
python -m research.scenario_ev.pass_counts --stage report

# Stage 04 -- fouls (--stage all|build|a|b|c|d|report, plus --force)
python -m research.scenario_ev.fouls --stage build   # ~2 min
python -m research.scenario_ev.fouls --stage a       # ~8 min
python -m research.scenario_ev.fouls --stage b       # ~2 min
python -m research.scenario_ev.fouls --stage c       # ~20 s
python -m research.scenario_ev.fouls --stage d       # ~1 min
python -m research.scenario_ev.fouls --stage report

# Stage 05 -- the uniform gate sweep (--targets all|<comma-separated>, --seeds, --folds, --report)
python -m research.scenario_ev.gate_sweep --targets all   # ~25 min over the eight markets
python -m research.scenario_ev.gate_sweep --report

# Stage 06 -- reconcile the haircut definitions and apply one of them to everything
python -m research.scenario_ev.haircut                  # ~20 s, reads committed tables only

# This synthesis: rebuild results_index.parquet from the committed tables (--print to dump it)
python -m research.scenario_ev.results_index

# Anti-staleness: check every report against a digest of the tables it was rendered from.
# Run --update only AFTER re-rendering the affected reports, never instead of it.
python -m research.scenario_ev.report_sources
python -m research.scenario_ev.report_sources --update

# Tests
python -m pytest research/scenario_ev/tests -q          # 157 tests
python -m pytest research/privileged_tracking/tests -q  # 180 tests, untouched by this phase
```

The order that matters: re-run a stage, re-render **every** report that reads its tables
(`report_sources.DEPENDENCIES` lists them; `02_corners.md` reads three of stage 01's and
stage 04's tables, which is how it went stale once), then `haircut`, then `results_index`,
then `report_sources --update`.

Processed intermediates (not committed) live under
`$PRIV_DATA_DIR/processed/scenario_ev/`; each stage report lists its own.

---

## 9. The results index, and the staleness guard

`results_index.parquet` (116 rows) is the machine-readable form of this document, built by
`research/scenario_ev/results_index.py`. One row per headline metric, with columns:

`candidate, stage, experiment, model, metric, value, ci_low, ci_high, ci_type, n, split,
source_report, source_table`

Each row is **read out of the source parquet at build time**, not transcribed: a spec whose
filters no longer select exactly one row raises rather than reporting a stale number.
Composition:

| experiment | rows | what it holds |
|---|---|---|
| `gate` | 38 | every (market, scenario) cell of the sweep, confirmation split |
| `betting` | 23 | ROI at the 6% headline hold, on the book ladders, in the scenario cells, and after the reference haircut |
| `vs_proxy` | 13 | each candidate's model margin over its book proxy |
| `haircut` | 12 | the nats measurements, the ROI haircuts, the reconciled reference and the pass stage's relative-scale instrument |
| `imputed_state` | 12 | every place the teacher-student channel was offered |
| `refit_floor` | 8 | the noise floors small deltas are read against, including the corner team line's own |
| `matchup_channel` | 6 | the two foul channels × two targets, plus the two fouls-won channels at their five-seed mean |
| `mechanism`, `ceiling` | 4 | the foul → card model, and the pass-count retention / variance ceiling |

Conventions worth knowing when reading it: for `gate` rows a **positive** value means the
specialist is better (targeting has room); for `vs_proxy` rows a positive value means the
model beats the proxy; `n` is the number of bets for betting rows and the number of scored
rows elsewhere; `ci_type` is `none` for quantities with no interval (refit spreads, variance
shares, point-estimate haircuts), `seed_range_5` where the interval columns hold a seed
min/max rather than a bootstrap, and it marks lower-bound-only intervals explicitly; `split`
is `cv_all` for the match-grouped cross-fitted quantities that have no temporal holdout (the
per-pass models and the corner style layer's CV scheme), and those are labelled as such
rather than called confirmation.

**The staleness guard.** `report_sources.py` pins every report in this directory to an MD5
of each parquet it was rendered from — its own stage's tables by prefix, plus the
cross-stage tables its renderer reads by name — in `report_sources.json`, and
`tests/test_report_sources.py` fails when any of them has moved. This exists because it
already happened once: stage 01's haircut table was regenerated after `02_corners.md` was
last rendered, and the committed corner report went on quoting superseded numbers until
someone re-ran its own documented render command. Cross-stage dependencies are the dangerous
ones, because a same-stage table is usually regenerated by the same command that re-renders
the report. `--update` re-pins; run it only after re-rendering, never instead.
