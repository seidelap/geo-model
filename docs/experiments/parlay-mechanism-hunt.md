# Parlay mechanism hunt: where could a cross-market correlation hide?

Companion to `parlay-interaction-effects.md` (which found the "explaining away"
cross-game correlation is about 0.002 against a 0.098 hurdle). This document is a
research survey, not a backtest: it enumerates other mechanisms by which two bets
that a sportsbook prices as independent could in fact be correlated, sizes each one
with the same Gaussian shared-variance / total-variance reasoning as
`gaussian_explaining_away_corr`, cites the prior evidence, states what data a test
needs and whether it is reachable through the GitHub-only proxy, and reports a set
of quick empirical bounds computed from the nflverse games file (1999-2025, REG,
6967 games with closing spread and total).

**Bottom line.** Anything that acts through the score is diluted by single-game
noise (sigma about 13.2 points) and cannot reach the hurdle: to get phi = 0.098 the
two legs need a *shared, unpriced* shock with standard deviation of about 5.2
points, and the largest unpriced shared component we can find in 27 seasons is
below 0.01 in correlation units. The only mechanisms that clear the hurdle are
*structural*: the outcome of one leg logically changes the meaning of the other
(final-week incentive contingencies, futures/derivative legs on the same team).
Those are real, computable by construction, rare, and mostly blocked or repriced by
house rules. A null on every score-mediated mechanism is the expected result and is
what we find.

## 0. The hurdle, restated in the right units

- 2-leg parlay, -110 per leg: pays 3.645; break-even phi between the leg-win
  indicators is **0.098** (same-sign rate 54.9%). Computed with
  `geo_model.parlay.backtest.breakeven_correlation`.
- For a bivariate normal split at its medians, phi = (2/pi) arcsin(rho), so phi =
  0.098 needs **Pearson rho = 0.153** between the two residuals.
- NFL closing-line residuals have variance 174 (spread) and 179 (total). A shared
  factor with variance v induces rho = v / (v + sigma^2), so rho = 0.153 needs
  **v = 27 points^2, i.e. a shared shock with std 5.2 points** that is common to
  both games *and unknown to the book at close*.
- A common cause that the book *knows* (a bye, a referee crew, a forecast) is not a
  correlation; it is either priced (nothing) or mispriced (a *marginal* edge on each
  leg). Marginal edges compound in parlays without any correlation: two 55% legs at
  -110 give a parlay EV of +10.3% versus +5.0% per single. The hunt for
  correlation is only interesting where the marginals are fair and the joint is not.
- Break-even phi depends on leg prices. At -110/-110 it is 0.098; at -120/-120 it is
  0.190; with a +150 second leg whose true probability is 0.40 it falls to 0.039
  (computed by hand as `(1/(d1*d2) - p1*p2) / sqrt(p1(1-p1)p2(1-p2))` with d1 = 1.909, d2 = 2.5; `breakeven_correlation` takes a single per-leg decimal).

Four classes of mechanism, by whether game noise dilutes them:

| class | what carries the dependence | diluted by sigma^2? | typical phi |
|---|---|---|---|
| A. structural / logical | leg 1's outcome changes leg 2's incentives or definition | no | 0.1 to 1.0 by construction, rare |
| B. score-mediated shared factor | a random common cause moves both scores | yes | v / 180 |
| C. derivative-market inconsistency | two prices for the same random variable disagree | n/a (arbitrage, not correlation) | n/a |
| D. marginal bias mistaken for correlation | a known common cause is mispriced on each leg | n/a | 0 |

## 1. Ranked table

`phi` is the order-of-magnitude expected correlation between leg-win indicators at
closing prices. `GitHub data` says whether a test is possible under the proxy
constraint (only raw.githubusercontent.com and github.com release assets reachable).
`status` says what books do about it today.

| rank | mechanism | class | legs (direction) | expected phi | evidence | GitHub data | status | testable now |
|---|---|---|---|---|---|---|---|---|
| 1 | Final-week incentive contingency (scoreboard dependence) | A | A wins & B fails to cover (negative) or A wins & B covers (positive), by scenario | 0.10 to 0.25 when live; 0 otherwise | by construction; NFL schedules linked games simultaneously to blunt it | yes (games.csv, standings computable) | allowed as a normal cross-game parlay; rare (a few games/season) | yes, low power |
| 2 | Futures / derivative leg + same-team game leg | A | team ML this week & team makes playoffs / wins division (positive) | 0.10 to 0.40; 1.0 in must-win games | by construction | rules audit only; no futures prices on GitHub | blocked or SGP-priced by house rules | no |
| 3 | Season win-total / make-playoffs vs. remaining schedule consistency | C | futures price vs. simulation from game lines | n/a (arbitrage) | win-total boards documented inefficient (hold, +1 win shading) | no futures archive on GitHub | markets closed once contingent on one game | no |
| 4 | MLB series carry-over (day-1 marathon -> day-2 bullpens) | B | day-1 over & day-2 over (sign unclear; FanGraphs finds scoring slightly *down*) | <= 0.05 | FanGraphs: run scoring "a little down" after 12+ inning games | retrosheet gamelogs 404 on GitHub; no MLB lines | day-2 lines posted after day 1 | no |
| 5 | Cross-day tanking / rest contingencies in NBA, MLB, soccer | A | rare contingencies; mostly marginal | 0.1+ when a contingency is live, else 0 | Soebbing & Humphreys: spreads already adjust for tanking | no NBA/MLB lines on GitHub | priced marginally; contingency rare | no |
| 6 | League-wide common-mode error (rule changes, scoring environment, officiating emphasis) | B | any two totals same week (positive) | measured: ICC 0.000 [-0.008, +0.008] | Bundesliga COVID home-advantage mispricing; NBA Feb-2024 scoring drop | yes | n/a | done here: null |
| 7 | Weather-system cluster (shared forecast error, same day, nearby stadiums) | B | two outdoor totals same day (positive, both under) | measured: ICC +0.008 [-0.013, +0.028] | Borghesi 2007/2008: weather is a *marginal* under bias | yes (wind/temp/roof in games.csv; no forecast archive) | n/a | done here: null |
| 8 | Sequential-window information flow (early games -> late lines; TNF -> Sunday) | B | early-window resid & late-window resid same day | measured: r = -0.02 (totals), -0.00 (spreads) | none | yes | n/a | done here: null |
| 9 | Explaining-away with more granular state (QB / unit-level Kalman) | B | H's next game & A's next game (positive) | <= 0.0006 at fitted s = 2.4; <= 0.068 even at s = 12 | previous doc: 0.002 measured | yes | n/a | analytically dead |
| 10 | Referee / umpire crew tendencies | D | none cross-game (a crew works one game per week) | 0 (marginal only); measured OOS slope 0.15 +- 0.21 | NBA/MLB crew effects exist but are marginal | yes (officials.csv 2015+; games.csv referee 1999+) | partially priced | done here: null |
| 11 | Rest / travel / bye / short week | D | none cross-game | 0 (marginal only) | Lopez & Bliss: bye over-priced (home off bye covers 44.6%) | yes (home_rest/away_rest) | priced/over-priced | done here: marginal only |
| 12 | Injury-news propagation | none | independent shocks per game | 0 | books reprice within about a minute | injuries release reachable | priced | n/a |
| 13 | Public-bias line shading co-movement (Levitt 2004) | D | all favourites same week | bounded by ICC 0.000 | Levitt 2004: shading is systematic, not random | yes | n/a | done here: null |
| 14 | Same-game correlated legs (spread+total, props+ML, Wong teasers) | A | favourite covers & over (positive) | 0.03 to 0.5 | JPM paper: over 53.3% when 3+ pt favourites cover; SGP correlations 30-50% | yes for spread+total | SGP-priced or blocked; teasers repriced | priced |
| 15 | Same-competitor multi-event (MLB doubleheaders, golf rounds + outright, tennis) | A | same competitor in two events | large | house rules cite it explicitly | no | voided ("same competitor in different matchups") | blocked |
| 16 | Pari-mutuel derivative pools (Dr. Z place/show) | C | win pool vs. place/show pool | n/a | Hausch, Ziemba & Rubinstein 1981 | n/a | historical; not a sportsbook parlay | analog only |

## 2. Quick empirical bounds (NFL, 1999-2025)

Data: `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
(same file that feeds `data/raw/nfl_games_lines.parquet`, but with `roof`, `wind`,
`temp`, `referee`, `home_rest`, `away_rest`, `gametime`, `weekday`). Residuals as in
the main experiment: `resid = result - spread_line`, `tresid = total - total_line`.

### 2.1 Shared-factor bound: intraclass correlation of residuals

The one-way random-effects ICC of residuals grouped by slate is exactly the
pairwise correlation of two games in the same group, i.e. the largest cross-game
correlation *any* shared factor at that grouping level could induce, whatever its
cause. 95% CIs are cluster bootstraps over groups (2000 resamples).

| grouping | groups | games | ICC | 95% CI |
|---|---|---|---|---|
| total resid, same (season, week) | 464 | 6967 | -0.0000 | [-0.0078, +0.0084] |
| spread resid, same (season, week) | 464 | 6967 | -0.0002 | [-0.0083, +0.0080] |
| total resid, same gameday | 1255 | 6967 | +0.0045 | [-0.0110, +0.0203] |
| spread resid, same gameday | 1255 | 6967 | -0.0053 | [-0.0189, +0.0105] |
| total resid, same season | 27 | 6967 | -0.0004 | [-0.0018, +0.0008] |
| spread resid, same season | 27 | 6967 | -0.0004 | [-0.0026, +0.0019] |
| total resid, same season, weeks 1-4 | 27 | 1669 | +0.0026 | [-0.0064, +0.0105] |
| spread resid, same season, weeks 1-4 | 27 | 1669 | -0.0031 | [-0.0119, +0.0079] |
| total resid, same season, 2018+ | 8 | 2127 | -0.0003 | [-0.0027, +0.0013] |
| home-cover indicator, same season | 27 | 6778 | -0.0019 | [-0.0030, -0.0008] |

Reading: the upper CI bound at the weekly level is 0.008 in Pearson units, versus
0.153 needed. Equivalent in points: the std of the season-mean total residual across
seasons is 0.79 versus 0.83 expected under independence (no detectable common-mode
season error); for weeks 1-4 it is 1.84 versus 1.70 expected, an excess variance of
0.5 points^2, i.e. rho = 0.003. The largest early-season common-mode errors were
2002 (+4.3 points), 2020 (+4.0) and 2008 (+3.6) over 1-4 weeks: real, but small
relative to the 27 points^2 needed and gone by mid-season.

### 2.2 Weather cluster

Outdoor games with a wind reading: 4980. ICC of total residuals on the same gameday:
+0.0081 [-0.0132, +0.0284] (1037 days). Restricted to wind >= 15 mph on both games:
-0.072 [-0.182, +0.043] (342 days, 649 games). The *marginal* effect is visible and
consistent with Borghesi: mean total residual +1.43 (se 0.36, n = 1517) at wind <= 5
mph versus -2.25 (se 1.32, n = 115) at wind > 20 mph. That is a single-bet lean on
windy unders, not a cross-game correlation.

### 2.3 Referee crew (out of sample)

For each season s >= 2000, each referee's mean total residual over seasons < s
(shrunk with k = 60 games, requiring >= 30 prior games) was regressed on that
season's residuals: n = 5449, slope 0.15 (se 0.21, p = 0.46); predictor std 0.88
points. Top-decile "over" crews went over 47.6% (n = 552), bottom-decile 47.9%
(n = 559). No marginal edge, and no cross-game channel exists in any case.

### 2.4 Final week, sequential windows, Thursday

- Final regular-season week (n = 429): spread residual std 12.98 vs 13.20 in other
  weeks (Levene p = 0.65); home cover 53.0%; over 50.9%; ICC of spread residuals
  within the final-week slate +0.0125 [-0.033, +0.056]. No sign of unpriced effort
  shocks in aggregate; the incentive mechanism (Section 3.1) has to be tested on the
  specific contingent games, not the whole week.
- Early (1pm) window mean residual vs late (4pm) window mean residual on the same
  Sunday, n = 443 Sundays: r = -0.021 (p = 0.67) for totals, -0.001 (p = 0.99) for
  spreads.
- Thursday-night residual vs mean residual of the rest of the week, n = 281 weeks:
  r = +0.036 (totals), +0.006 (spreads).

### 2.5 Rest (marginal only)

Home team off a bye vs opponent on normal rest: n = 360, mean spread residual -0.13
(se 0.66), home covers 47.7%. Away team off a bye: n = 342, residual -1.20 (se 0.69),
home covers 46.4% (away covers 53.6%). Short week (<= 5 days rest): home n = 305,
-0.38 (se 0.73); away n = 304, -0.48 (se 0.73). Directionally matches Lopez & Bliss
(the market gives about +1 point for a bye, the true effect is closer to +0.3), but
these are marginals with p > 0.05 and no parlay relevance.

### 2.6 Marginals for context

Overall over rate 49.5% (n = 6868, se 0.6%); mean total residual +0.69 points (se
0.16) - the mean is positive because the residual distribution is right-skewed, the
median is not. Week-1 over rate 45.3% (n = 424, se 2.4%), i.e. unders 54.7%, in line
with the BetMGM "Week 1 unders 55% since 2005" trend; weeks 1-4 over rate 48.9%.
Home cover rate 49.0% (n = 6778).

### 2.7 Multiple-comparison exposure

About 30 statistics over 24 subsets (10 ICC groupings, 2 weather ICCs plus 5 wind
buckets, 1 referee regression plus 2 deciles, 3 final-week statistics, 2
window correlations, 2 Thursday correlations, 4 rest cells, 4 marginals). Nothing
was pre-registered beyond the mechanism list above; the smallest p-value among the
correlation tests is 0.46. With this many looks, any |ICC| below 0.02 should be
read as noise; none exceeded it.

## 3. Mechanisms

Each section: causal story; legs and direction; magnitude and reasoning; prior
evidence; data and reachability; test spec against phi = 0.098; practical
constraints; rank.

### 3.1 Final-week incentive contingency (rank 1)

**Story.** In the last regular-season week (and occasionally earlier), whether team
B has anything to play for depends on the result of team A's game. If A wins, B is
eliminated (or clinched) and rests starters or plays at reduced effort; if A loses,
B plays a full-effort game. The book must price B's line as an average over the two
states, but a parlay locks both legs before A kicks off, so the joint outcome is
mispriced by construction. Because the channel is *effort*, not score noise, the
correlation is not diluted by sigma^2.

**Legs.** A moneyline (or spread) and B spread. Sign depends on the scenario: "A wins
kills B's stake" gives negative correlation between A-wins and B-covers (bet A wins +
B fails to cover); "A wins gives B something to play for" gives positive.

**Magnitude.** Contingency tree with P(A wins) = 0.5 and B's cover probability 35%
if the stake dies vs 50% otherwise: P(B covers) = 0.425, phi = -0.152, parlay EV
(A wins, B fails) = +18% at -110/-110. At 40% vs 50%: phi = -0.101 (break-even). At
45% vs 50%: phi = -0.050 (loses). Resting starters is worth many points (Week-18
lines move 7-10 points when a QB is announced out), so 35% vs 50% is conservative
*when the contingency is live and unresolved at kickoff*.

**Evidence.** By construction. The NFL announces Week 18 kickoff times only after
Week 17 and places games with linked implications in the same window, which removes
the sequential (early/late) version but not the simultaneous scoreboard-watching
version (B's second-half effort responds to A's in-progress score). Examples of
motivation mismatches in Week 18 are routine in the betting press (see the
DraftKings Network and CBS pieces below), but those are announced rests, i.e.
priced marginals; the unpriced part is the *unresolved* contingency.

**Data.** games.csv gives results, kickoff times, and closing lines back to 1999;
standings and clinch/elimination scenarios are computable (tie-breakers need care;
a Monte Carlo over remaining games using closing spreads is an adequate proxy).
Reachable now.

**Test.** For each season, identify (A, B) pairs in the final two weeks where B's
playoff probability, simulated before the slate, differs by >= 30 percentage points
between "A wins" and "A loses". Record phi between I(A wins) and I(B covers) with
the scenario sign, pooled across seasons; compare to 0.098 with a bootstrap CI. Also
run the two-sided parlay ROI (`two_sided_parlay_roi` with the scenario sign).
Expect roughly 2-5 qualifying pairs per season (about 60-130 over 27 seasons), so
the CI will be about +-0.2: the test can confirm the sign and rule out zero only if
the effect is at the high end. Report EV by construction alongside.

**Constraints.** Rare; both legs must be posted (books sometimes hold Week-18 lines
until rest decisions are announced); the book reprices B once A's result is known,
so a straight bet after A is not available at the stale price but the parlay is; a
book may void a parlay it judges correlated after the fact (house rules reserve this).

### 3.2 Futures / derivative leg plus a same-team game leg (rank 2)

**Story.** "Team X makes the playoffs" or "wins the division" is a function of X's
remaining results. Parlaying it with "X wins this week" at independent multiplication
is +EV whenever the futures price and the game price are each fair.

**Magnitude.** With P(win) = 0.5 and playoff probability 60% if win / 40% if lose:
phi = 0.20; with a 20-point swing, 0.40; a Super Bowl future at 15% / 5%: 0.157; in
a playoff game the future is 0 if the team loses, so phi = 1. Break-even phi with a
+150 futures leg (true p = 0.40) is 0.039, so even a 5-point swing clears it.

**Evidence.** Arithmetic. The Journal of Prediction Markets college-football paper
documents the analogous same-game case and notes books "have generally been too
conservative in refusing such bets", i.e. books know.

**Data / status.** No futures price archive is reachable on GitHub. House rules
(e.g. Fanatics) refuse "parlays where the outcome of one part of the wager
contributes to the outcome of another" and cancel them if taken in error; the major
US books route same-team combinations through same-game-parlay pricing engines
(SGP/SGPx) that apply a correlation tax. The only test is a rules audit per book.

### 3.3 Season win totals and make-playoffs prices vs. the remaining schedule (rank 3)

**Story.** Not a correlation but a consistency check: a team's win-total or playoff
price implies a distribution over its remaining games; game lines imply another.
Late in the season a win-total "over 9.5" for a 9-7 team is a clone of its last-game
moneyline.

**Evidence.** Win-total boards sum to 273 wins for 272 games (one win of optimism
shaded across teams), carry higher holds than sides, and differ by a full win across
books (nfelo, analytics.bet, atsstats). Academic work finds the win-total market
"highly inefficient" in the sense of profitable rules (Claremont thesis review).

**Data / status.** No archive reachable. Books close or reprice these markets once
they collapse to a single game. Arbitrage, not parlay.

### 3.4 MLB series carry-over (rank 4)

**Story.** After a marathon game both bullpens are depleted; both teams meet again
tomorrow, so day-2 scoring shifts for both. Day-1 total and day-2 total would be
correlated if day-2 lines were posted before day 1.

**Magnitude.** If the day-2 shift is beta times the day-1 total residual, rho =
beta (equal variances). A 0.5-run shift after a +6 residual gives beta = 0.08, phi
about 0.05. FanGraphs finds run scoring "a little down" (not up) after 12+ inning
games and no rise in extra-inning frequency (8.9% vs 9.0%), so the sign is not even
clearly positive and the size is small.

**Data / status.** Retrosheet game logs on the chadwickbureau GitHub mirror returned
404 at the path tried; no MLB closing-line archive is on GitHub. Day-2 lines are
usually posted after day 1 ends, so the parlay does not exist at pre-day-1 prices.
Not testable here.

### 3.5 Cross-day tanking / rest contingencies in other leagues (rank 5)

**Story.** NBA lottery standings, play-in seeding, MLB final weekend, soccer final
matchdays: team B's incentive tonight depends on A's result. Same structure as 3.1.

**Evidence.** Soebbing & Humphreys (Contemporary Economic Policy, 2013) find NBA
point spreads already adjust for eliminated teams' tanking incentives (a priced
marginal). Sportico and Gambling Insider document that 2025-26 tanking produced
35% blowouts in tank-vs-non-tank games and that mid-game star pulls defeat injury-
report-based edges: variance, not correlation. UEFA and domestic leagues play final
matchdays simultaneously for exactly this reason.

**Data / status.** No NBA/MLB/soccer lines reachable on GitHub. The contingent
subset is small; the book prices the marginal; the parlay-relevant unresolved
contingency is rarer than in the NFL because most leagues finish simultaneously.

### 3.6 League-wide common-mode error (rank 6)

**Story.** A rule change, a ball change, an officiating point of emphasis, or a
scoring-environment shift moves every game's expected total, and the market adapts
with a lag. During the lag all totals share an unpriced bias delta; as a random
effect with variance v it induces rho = v / (v + 180).

**Magnitude.** A 3-point unpriced league-wide shift gives rho = 9/189 = 0.048, phi
about 0.03. Measured NFL bound: ICC within week 0.000 [-0.008, +0.008]; within
season -0.0004; weeks 1-4 +0.003. The documented lag cases elsewhere (Bundesliga
home advantage after COVID, Deutscher, Winkelmann & Otting 2020, arXiv:2008.05417;
NBA March-2024 scoring drop of about 7 points with totals lagging) are *marginal*
edges: once you know the sign you bet every leg the same way, and the correlation is
a by-product too small to matter at -110.

**Data / test.** Done (Section 2.1). Null in the NFL.

### 3.7 Weather-system cluster (rank 7)

**Story.** Realized wind or rain across nearby outdoor stadiums on the same afternoon
is correlated, so the forecast errors the book could not price are shared.

**Magnitude.** Wind moves totals by roughly 0.3 points per mph above 10 mph; a 4 mph
forecast error shared with r = 0.5 between two cities gives shared variance 0.7
points^2 and rho = 0.004. Measured: same-day outdoor ICC +0.008 [-0.013, +0.028];
wind >= 15 subset -0.07 [-0.18, +0.04].

**Evidence.** Borghesi (Journal of Sports Economics 2007; Applied Financial
Economics 2008): weather is under-incorporated in totals, exploitable *marginally*
near kickoff. Our wind-bucket marginals agree.

**Data / test.** wind/temp/roof in games.csv; no forecast archive on GitHub, so only
realized-weather clustering is testable. Done; null.

### 3.8 Sequential-window information flow (rank 8)

**Story.** Early games reveal something league-wide (officiating, weather, a
scoring regime) that moves late lines; a parlay placed before the early window holds
the late leg at a stale price.

**Measured.** Early-window mean vs late-window mean residual on the same Sunday:
r = -0.02 (totals), -0.00 (spreads), n = 443; Thursday vs rest of week: r = +0.04 /
+0.01, n = 281. Nothing to carry.

### 3.9 Explaining-away with more granular state (rank 9)

**Story.** The user's track (a): replace team-strength errors with QB-, unit- or
matchup-level errors and accumulate over the whole season.

**Bound.** The one-shared-game correlation is a function of the *total* per-team
market error s and game noise sigma only; partitioning s^2 into finer components
cannot raise it. With sigma = 13.3: s = 2.37 (fitted) gives next-game rho 0.0009
(phi 0.0006); s = 6 gives 0.021; s = 8 gives 0.046 (phi 0.029); s = 12, i.e. the
market wrong by 12 points per team on average, gives rho 0.106 (phi 0.068), still
below 0.098. A granular filter could only help by finding *more* market error than
the team-level filter (s > 12), which the marginal check (lag-1 residual
autocorrelation -0.006; Kalman mean slope -0.83 +- 0.73 in the main doc) rules out.
Analytically dead.

### 3.10 Referee and umpire crews (rank 10)

**Story.** Crews differ in penalty rates (NFL: about 2 accepted penalties per game
between the highest and lowest crews), free throws (NBA), and strike zones (MLB,
0.2-0.3 runs per game between extreme umpires). If unpriced, every game a crew
works shares a bias. But a crew works one game per week, so there is no same-slate
pair; across weeks the bias is a constant, i.e. a marginal.

**Measured.** NFL out-of-sample crew effect on totals: slope 0.15 (se 0.21), decile
over rates 47.6% vs 47.9%. No marginal either.

**Evidence.** Belasen, Belasen & Olbrecht (Journal of Sports Economics 2025) on
NBA last-two-minute calls vs. the line; practitioner umpire guides for MLB. Data:
nflverse `officials.csv` (2015+) is reachable; games.csv has `referee` from 1999.

### 3.11 Rest, travel, bye, short week (rank 11)

Known common causes; priced (and per Lopez & Bliss, Frontiers in Behavioral
Economics 2024, over-priced for byes: home teams off a bye cover 44.6% since 2011).
Our cells agree directionally (Section 2.5). No cross-game channel: two legs sharing
a rest state share a *constant* mispricing, which is a marginal.

### 3.12 Injury-news propagation (rank 12)

Books pull or move lines within about a minute of injury news; a parlay placed
midweek carries stale prices on both legs, but the shocks are independent across
games, so the parlay's variance rises and its correlation does not. The only shared
version is same-team across weeks (a Thursday-game injury and that team's next
line), which is priced before the second leg is posted.

### 3.13 Public-bias line shading co-movement (rank 13)

Levitt (Economic Journal 2004) shows books shade lines toward bettor biases so that
the popular side loses more than half the time; the shading is systematic (a
marginal on favourites/overs), not a random weekly factor. Any weekly randomness in
it is inside the weekly ICC bound of 0.008.

### 3.14 Same-game correlated legs (already priced)

Favourite covers and over (Journal of Prediction Markets, college football
2005-2015: over 52.5% when favourites cover, 53.3% for 3+ point favourites);
QB passing yards and team win; first-half and full-game lines; team total and game
total. Correlations of 0.03 to 0.5. Books either block these or price them through
SGP engines with a correlation tax (typical SGP hold 12-18%). Wong teasers (6-point
teasers crossing 3 and 7, historically 73-76% per leg) are a marginal-edge product
that books now price at -120/-130 or worse.

### 3.15 Same-competitor multi-event (blocked)

MLB doubleheaders (game-2 lines are posted after game 1), a golfer's round matchup
plus outright, a tennis player's singles and doubles: house rules void "the same
competitor in different matchups". Listed-pitcher rules void or reprice a leg when
the starter changes; DraftKings voids a parlay containing a player who does not
play. These are the cases books have already found.

### 3.16 Pari-mutuel derivative pools (historical analog)

Hausch, Ziemba & Rubinstein (Management Science 1981) showed place and show pools
were priced inconsistently with the win pool, exploitable with a Kelly-style
optimizer ("Dr. Z"). This is the canonical documented derivative-market
inconsistency in wagering and the template for class C. Nothing comparable exists in
fixed-odds sportsbook parlays, which are priced by one book from one model.

## 4. What books already price, and what is unavailable

- **Priced via SGP engines:** any two legs from the same game; player props with
  team results; first-half/full-game; team totals with sides.
- **Blocked or voided:** futures with a same-team game leg; same competitor in two
  events; nested futures (division + conference); parlays the book judges
  "correlated" after the fact (house rules reserve cancellation).
- **Repriced or restricted:** Wong teasers; listed-pitcher legs; parlays with a
  non-playing player (voided).
- **Allowed and priced as independent:** cross-game sides and totals, which is why
  the survey concentrates on them and finds the shared-factor bound at 0.008.

## 5. Recommendation

The only mechanism that is (a) allowed as a normal cross-game parlay, (b) not
diluted by game noise, and (c) testable from reachable data is 3.1, the final-week
incentive contingency. It is +EV by construction when the contingency is
unresolved at kickoff and B's response is large, but the qualifying sample over 27
seasons is on the order of 60-130 pairs, so a backtest can confirm sign and rough
size, not tight significance. Everything score-mediated is bounded near zero and
should not be pursued further; the granular-state Kalman idea is bounded
analytically (phi <= 0.03 even if the market were wrong by 8 points per team).

## 6. Caveats

- All bounds are NFL closing lines from one aggregator; opening-line correlations
  could be larger but are not parlay-relevant unless the parlay is placed at open.
- ICC is a two-sided bound on shared variance at a grouping level; it cannot see a
  mechanism that is positive for some pairs and negative for others in the same slate
  (the incentive mechanism is such a case, which is why it needs a scenario-signed
  test).
- Contingency-tree phis are illustrative parameterizations, not estimates.
- Web sources for practitioner claims (SGP holds, referee trends, tanking) are not
  peer-reviewed; academic citations are listed where they exist.
- No MLB/NBA/soccer test was possible under the GitHub-only proxy.

## 7. Sources

Academic and primary:
- Sauer, R. D. (1998). The Economics of Wagering Markets. JEL 36(4): 2021-2064. https://ideas.repec.org/a/aea/jeclit/v36y1998i4p2021-2064.html
- Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? Economic Journal 114. Summarized in https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=2102&context=cmc_theses
- Paul, R. J. & Weinbach, A. P. (2002). Market Efficiency and a Profitable Betting Rule: Evidence From Totals on Professional Football. Journal of Sports Economics. https://journals.sagepub.com/doi/10.1177/1527002502003003003
- Borghesi, R. (2007). The Home Team Weather Advantage and Biases in the NFL Betting Market. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2149682
- Borghesi, R. (2008). Weather biases in the NFL totals market. Applied Financial Economics 18(12). https://www.tandfonline.com/doi/full/10.1080/09603100701335432
- Correlated Parlay Betting: An Analysis of Betting Market Profitability Scenarios in College Football. Journal of Prediction Markets. https://www.ubplj.org/index.php/jpm/article/view/1562
- Soebbing, B. P. & Humphreys, B. R. (2013). Do Gamblers Think That Teams Tank? Evidence from the NBA. Contemporary Economic Policy. https://onlinelibrary.wiley.com/doi/10.1111/j.1465-7287.2011.00298.x
- Deutscher, C., Winkelmann, D. & Otting, M. (2020). Bookmakers' mispricing of the disappeared home advantage in the German Bundesliga after the COVID-19 break. arXiv:2008.05417. https://arxiv.org/abs/2008.05417
- Lopez, M. J. & Bliss, T. (2024). Bye-bye, bye advantage: estimating the competitive impact of rest differential in the NFL. Frontiers in Behavioral Economics. https://www.frontiersin.org/journals/behavioral-economics/articles/10.3389/frbhe.2024.1479832/full
- Belasen, A. R., Belasen, A. T. & Olbrecht, A. M. (2025). With the Game on the (Betting) Line: NBA Referee Performance in the Last Two Minutes. Journal of Sports Economics. https://journals.sagepub.com/doi/10.1177/15270025251369447
- Hausch, D. B., Ziemba, W. T. & Rubinstein, M. (1981). Efficiency of the Market for Racetrack Betting. Management Science 27(12). https://pubsonline.informs.org/doi/abs/10.1287/mnsc.27.12.1435
- Whelan, K. (2026). The Parlay Puzzle: Expected Utility and Multi-Leg Betting. UCD WP. https://www.karlwhelan.com/Papers/Parlays.pdf
- Arbitrage Analysis in Polymarket NBA Markets (ML vs spread arbitrage, mostly live). https://arxiv.org/pdf/2605.00864
- JQAS 4(2) 2008, article 7 (NBA betting percentages: favourites and overs over-bet), hosted at http://users.nber.org/~jwolfers/Papers/NBABetting.pdf

Practitioner and house rules:
- Wizard of Odds, Same-Game Parlays: The Mathematics of Correlation. https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/
- OddsIndex, SGP correlation tax. https://oddsindex.com/guides/same-game-parlay-correlation
- Fanatics Sportsbook house rules (correlated parlays cancelled). https://sportsbook.fanatics.com/legal/tn/house-rules/
- Action Network, MLB listed-pitcher vs action rules. https://www.actionnetwork.com/mlb/mlb-betting-rules-for-scratched-pitchers-action-vs-listed
- Action Network, parlays voided when a player does not play. https://www.actionnetwork.com/mlb/props-parlays-sgps-if-the-player-isnt-in-the-lineup
- FanGraphs, What Happens the Game After a Marathon Extra-Inning Game? https://blogs.fangraphs.com/what-happens-the-game-after-a-marathon-game/
- BetMGM, Week 1 unders trend. https://sports.betmgm.com/en/blog/nfl/nfl-betting-trends-week-1-bm16/
- Sharp Football Analysis, referee crew penalty trends. https://www.sharpfootballanalysis.com/betting/nfl-referee-assignments-penalty-trends-betting-impact/
- Sportico, NBA tanking data 2025-26. https://www.sportico.com/leagues/basketball/2026/nba-tanking-data-draft-lottery-odds-1234890511/
- Gambling Insider, How bookmakers are adjusting to NBA tankathon. https://www.gamblinginsider.com/news/114641/how-bookmakers-are-adjusting-to-nba-tankathon
- DraftKings Network, Which teams will rest starters in Week 18. https://dknetwork.draftkings.com/2026/01/01/which-teams-will-rest-starters-in-week-18-3/
- OddsShopper, Wong teaser strategy status. https://www.oddsshopper.com/articles/betting-101/wong-teaser-strategy
- nfelo, win-total board sums to 273. https://www.nfeloapp.com/nfl-power-ratings/nfl-win-totals/
- nflverse-data releases (officials 2015+, injuries, pbp, snap counts, depth charts reachable via github.com release assets). https://github.com/nflverse/nflverse-data/releases

## Appendix: regenerating Section 2

No module was added for this survey (research track). The numbers in Section 2 come
from the following self-contained script; run it from the repo root with the system
python (pandas, numpy, scipy). Bootstrap CIs use seed 0 and 2000 resamples.

```bash
D="${GEO_MODEL_DATA_DIR:-data/raw}"; mkdir -p "$D"
curl -sSL -o "$D/nfl_games_full.csv" https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
GAMES="$D/nfl_games_full.csv" python - <<'PY'
import os
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(0)
raw = pd.read_csv(os.environ["GAMES"], low_memory=False)
df = raw[raw.result.notna() & raw.spread_line.notna() & raw.total_line.notna()
         & (raw.game_type == "REG") & raw.season.between(1999, 2025)].copy()
df["resid"] = df.result - df.spread_line
df["tresid"] = df.total - df.total_line

def icc(values, groups, n_boot=2000):
    """One-way random-effects ICC (pairwise within-group correlation) with a cluster bootstrap."""
    d = pd.DataFrame({"y": values, "g": groups}).dropna()
    grp = d.groupby("g")["y"]
    n = grp.size().to_numpy(float); s1 = grp.sum().to_numpy(); s2 = (d.y ** 2).groupby(d.g).sum().to_numpy()
    def _icc(n, s1, s2):
        k, N = len(n), n.sum()
        gm = s1.sum() / N
        msb = (n * (s1 / n - gm) ** 2).sum() / (k - 1)
        msw = (s2 - s1 ** 2 / n).sum() / (N - k)
        n0 = (N - (n ** 2).sum() / N) / (k - 1)
        return (msb - msw) / (msb + (n0 - 1) * msw)
    est = _icc(n, s1, s2); k = len(n)
    boots = [_icc(*(a[ix] for a in (n, s1, s2))) for ix in (rng.integers(0, k, k) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi, k, int(n.sum())

wk = df.season.astype(str) + "-" + df.week.astype(str)
for label, col, g in [("total|week", "tresid", wk), ("spread|week", "resid", wk),
                      ("total|gameday", "tresid", df.gameday), ("spread|gameday", "resid", df.gameday),
                      ("total|season", "tresid", df.season), ("spread|season", "resid", df.season),
                      ("total|season wk1-4", "tresid", df.season.where(df.week <= 4)),
                      ("spread|season wk1-4", "resid", df.season.where(df.week <= 4)),
                      ("total|season 2018+", "tresid", df.season.where(df.season >= 2018))]:
    print(label, "ICC %+.4f [%+.4f, %+.4f] groups=%d n=%d" % icc(df[col], g))
o = df[df.roof.isin(["outdoors", "open"]) & df.wind.notna()]
print("weather|outdoor gameday", "ICC %+.4f [%+.4f, %+.4f] groups=%d n=%d" % icc(o.tresid, o.gameday))
w = o[o.wind >= 15]
print("weather|wind>=15 gameday", "ICC %+.4f [%+.4f, %+.4f] groups=%d n=%d" % icc(w.tresid, w.gameday))
print(o.groupby(pd.cut(o.wind, [-1, 5, 10, 15, 20, 100]), observed=True).tresid.agg(["size", "mean", "sem"]).round(3))
r = df[df.referee.notna()].copy(); r["pm"] = np.nan; r["pn"] = np.nan
for s in sorted(r.season.unique()):
    past = r[r.season < s].groupby("referee").tresid.agg(["mean", "size"]); m = r.season == s
    r.loc[m, "pm"] = r.loc[m, "referee"].map(past["mean"]); r.loc[m, "pn"] = r.loc[m, "referee"].map(past["size"])
rr = r[r.pn >= 30].assign(shrunk=lambda x: x.pm * x.pn / (x.pn + 60.0))
res = stats.linregress(rr.shrunk, rr.tresid)
print("referee OOS n=%d slope=%.3f se=%.3f p=%.3f" % (len(rr), res.slope, res.stderr, res.pvalue))
top, bot = rr[rr.shrunk >= rr.shrunk.quantile(.9)], rr[rr.shrunk <= rr.shrunk.quantile(.1)]
print("decile over rates %.3f (n=%d) vs %.3f (n=%d)" % ((top.tresid > 0).mean(), len(top), (bot.tresid > 0).mean(), len(bot)))
fw = df.groupby("season").week.transform("max"); last, other = df[df.week == fw], df[df.week != fw]
print("final week n=%d std %.2f vs %.2f Levene p=%.3f; ICC %+.4f [%+.4f, %+.4f]" % (
    len(last), last.resid.std(), other.resid.std(), stats.levene(last.resid, other.resid).pvalue, *icc(last.resid, last.season)[:3]))
sun = df[(df.weekday == "Sunday") & df.gametime.notna()].copy(); sun["hour"] = sun.gametime.str[:2].astype(int)
e = sun[sun.hour <= 13].groupby("gameday").agg(e_t=("tresid", "mean"), e_s=("resid", "mean"), n=("resid", "size"))
l = sun[sun.hour.between(15, 17)].groupby("gameday").agg(l_t=("tresid", "mean"), l_s=("resid", "mean"), n2=("resid", "size"))
j = e.join(l, how="inner"); j = j[(j.n >= 4) & (j.n2 >= 2)]
print("early vs late n=%d totals r=%+.3f spreads r=%+.3f" % (len(j), stats.pearsonr(j.e_t, j.l_t)[0], stats.pearsonr(j.e_s, j.l_s)[0]))
tn = df[df.weekday == "Thursday"].groupby(["season", "week"]).agg(t=("tresid", "mean"), s=("resid", "mean"))
rest = df[df.weekday != "Thursday"].groupby(["season", "week"]).agg(t2=("tresid", "mean"), s2=("resid", "mean"), n=("resid", "size"))
j = tn.join(rest, how="inner"); j = j[j.n >= 8]
print("TNF vs rest n=%d totals r=%+.3f spreads r=%+.3f" % (len(j), stats.pearsonr(j.t, j.t2)[0], stats.pearsonr(j.s, j.s2)[0]))
for lab, m in [("home off bye", (df.home_rest >= 13) & (df.away_rest <= 8)), ("away off bye", (df.away_rest >= 13) & (df.home_rest <= 8)),
               ("home short week", df.home_rest <= 5), ("away short week", df.away_rest <= 5)]:
    g = df[m]; print("%s n=%d mean resid %+.2f (se %.2f) home cover %.3f" % (lab, len(g), g.resid.mean(), g.resid.sem(), (g.resid[g.resid != 0] > 0).mean()))
nz = df[df.tresid != 0]; w1 = nz[nz.week == 1]
print("over rate %.4f n=%d; week1 over %.4f n=%d; home cover %.4f" % ((nz.tresid > 0).mean(), len(nz), (w1.tresid > 0).mean(), len(w1), (df.resid[df.resid != 0] > 0).mean()))
PY
```

Theoretical numbers (Sections 0 and 3.9) come from
`geo_model.parlay.backtest.gaussian_explaining_away_corr(s, 13.3)` for
s in {2.37, 4, 6, 8, 10, 12} and `breakeven_correlation(...)`; the contingency-tree
phis are `cov / sqrt(p1(1-p1)p2(1-p2))` with `cov = P(A wins) * (P(B covers | A wins) - P(B covers))`.
