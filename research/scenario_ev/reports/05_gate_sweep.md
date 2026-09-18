# 05 The gate sweep: is there room for scenario-specific modelling at all?

Machine-written by `research/scenario_ev/gate_sweep.py` from `reports/05_gate_sweep_rows.parquet` (every fit) and `reports/05_gate_sweep_summary.parquet` (the collapse). This is the cross-candidate deliverable of the phase: the same test, one implementation, one sign convention, on every market the phase touches.

```
cd /home/user/geo-model
python -m research.scenario_ev.gate_sweep --targets all   # ~25 min over the eight markets
python -m research.scenario_ev.gate_sweep --report        # rebuild the summary and this file
```

## The answer

**The gate is shut.** Across 38 pre-registered (market, scenario) cells spanning 8 markets, the specialist beats the generalist on held-out confirmation scenario rows in **1**. It is beaten outright in **21**, and the remaining 16 are inconclusive (interval spanning zero, or a delta inside the market's refit-noise floor). The damage is not marginal either: the worst cell (player_fouls / proxy_high) costs -0.0330 nats and the best gains only +0.0054.

### The one exception: team_corners / fav_strong

`team_corners` in the scenario *no-vig favourite probability >= 0.65* is the single cell where a specialist wins: +0.00538 nats [+0.00317, +0.00744] on 9272 held-out confirmation rows, against a market refit-noise floor of 0.00045. Three things keep this from being a finding. Its discovery delta was only +0.00079, an order of magnitude smaller than what confirmation produced, so the two halves agree on sign but not on size. With 38 cells tested at a 5% level, one or two intervals clearing zero by chance is the expectation, not a surprise. And stage 02 already established what this market is worth commercially: the team corner line is exactly where its model does beat the rolling-mean proxy, and exactly there the goals haircut removes the entire margin. A cell that is real would still not be tradeable.

That is the answer to the question this phase was set up to ask. Scenario-specific modelling can only pay if the general model is *underfitting* the scenario. With one arguable exception out of 38, it is not. On these targets a model fitted on all rows already prices the scenario rows at least as well as a model fitted only on them -- usually much better, because the scenario mostly throws away training data that was carrying transferable structure. No amount of teacher-student machinery, imputed state or market cleverness downstream can create room that the gate says is not there.

## Method

For one market and one scenario:

1. a **generalist** is fitted on all training rows and a **specialist** on the training rows inside the scenario;
2. both are scored on the **same** held-out scenario rows -- 5-fold match-grouped CV inside the discovery half, then once on confirmation after a single fit on all of discovery;
3. the reported quantity is `delta = loss(generalist) - loss(specialist)`, so **positive means the specialist is better, i.e. targeting has room**;
4. the interval is a match-clustered bootstrap (2,000 resamples) on the paired per-row loss difference;
5. the same pair is scored **off** the scenario as a sanity check that the specialist is a real model rather than a broken one;
6. `refit_spread` is the range of the delta over three seeds; only each market's primary scenario is refitted that way, so `refit_noise_floor` carries that market's measured spread across all of its cells and the verdict requires a delta to clear it. A cell with `n_seeds = 1` has no spread of its own (`refit_spread` is blank), not a spread of zero. If a whole market were ever run under a single seed, its `refit_noise_floor` would print as `n/a` and a positive interval there would be called `inconclusive (no measured refit floor)` rather than `room`: there would be nothing for the delta to clear. A three-seed range is itself a thin estimate of a spread -- stage 04 re-measured one channel increment over five seeds instead of three and its range grew by an order of magnitude -- so the floor here is a weak screen, used only to *downgrade* a positive cell and never to promote one. Every market in the table below carries a measured floor.

Losses are in nats throughout: binary log loss for `player_cards`, and a negative-binomial log score for the seven count markets, with the dispersion fitted once on the discovery generalist and shared by both models so that the comparison is about the conditional mean and nothing else.

**Every market keeps its owner's design.** The sweep does not invent features or model settings for someone else's target; it imports the owning stage's universe filter, feature set, categorical handling and LightGBM settings:

| market | owned by | features | rows |
|---|---|---|---|
| match_corners | `corners.py` | 102 | 105378 |
| player_cards | `cards.py` | 63 | 33329 |
| player_fouls | `fouls.py` | 61 | 33329 |
| player_passes | `pass_counts.py` | 25 | 29081 |
| player_passes_completed | `pass_counts.py` | 25 | 29081 |
| team_cards | `cards.py` | 54 | 125411 |
| team_corners | `corners.py` | 115 | 210756 |
| team_fouls | `fouls.py` | 47 | 210076 |

Stages 01-03 each grew a local copy of this loop before the shared version existed (`cards._gate`, `corners.stage_gate`, `pass_counts.gate`). They agree on the design but differ in loss and in the sign of the reported delta, which makes their tables hard to read side by side; the shared implementation now lives in `common.run_gate` and this stage runs everything through it. The individual stages' own gate tables are left as they were and their conclusions are unchanged.

**Scenarios.** Each market contributes the scenarios its own stage pre-registered on discovery, plus two that are defined identically everywhere -- `proxy_low` and `proxy_high`, the bottom and top discovery terciles of that market's own book proxy -- so that at least one cell is comparable across all eight markets.

## The table

Confirmation is the headline; `delta_discovery` is shown beside it because a sign flip between the two is exactly what the split exists to catch. `delta_off_scenario` is the same pair scored outside the scenario.

| market | scenario | kind | n_discovery | n_confirmation | n_train_scenario | loss_generalist | loss_specialist | delta_gen_minus_spec | ci_lo | ci_hi | delta_discovery | delta_off_scenario | refit_spread | n_seeds | refit_noise_floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| match_corners | even_match | count | 20259 | 13346 | 20259 | 2.60185 | 2.60309 | -0.00125 | -0.00210 | -0.00040 | -0.00090 | 0.00014 | n/a | 1 | 0.00009 | no room |
| match_corners | fav_strong | count | 6220 | 4636 | 6220 | 2.65756 | 2.66254 | -0.00499 | -0.00813 | -0.00212 | -0.00423 | -0.00557 | 0.00009 | 3 | 0.00009 | no room |
| match_corners | high_total | count | 3317 | 4995 | 3317 | 2.66240 | 2.66773 | -0.00533 | -0.00794 | -0.00282 | -0.00344 | -0.01528 | n/a | 1 | 0.00009 | no room |
| match_corners | proxy_high | count | 21076 | 2360 | 21076 | 2.68357 | 2.68525 | -0.00168 | -0.00357 | 0.00010 | -0.00018 | -0.06185 | n/a | 1 | 0.00009 | inconclusive (interval spans zero) |
| match_corners | proxy_low | count | 21076 | 31208 | 21076 | 2.60225 | 2.60340 | -0.00114 | -0.00157 | -0.00070 | -0.00075 | -0.00168 | n/a | 1 | 0.00009 | no room |
| player_cards | fullback_only | binary | 3194 | 2695 | 3194 | 0.47378 | 0.47514 | -0.00136 | -0.00554 | 0.00283 | -0.00561 | -0.01692 | 0.00344 | 3 | 0.00344 | inconclusive (interval spans zero) |
| player_cards | fullback_x_flank_x_ref | binary | 348 | 400 | 348 | 0.52399 | 0.54479 | -0.02080 | -0.04088 | -0.00064 | 0.00014 | -0.04806 | n/a | 1 | 0.00344 | no room |
| player_cards | proxy_high | binary | 6061 | 5406 | 6061 | 0.54810 | 0.54977 | -0.00167 | -0.00390 | 0.00050 | -0.00348 | -0.01138 | n/a | 1 | 0.00344 | inconclusive (interval spans zero) |
| player_cards | proxy_low | binary | 6061 | 5621 | 6061 | 0.33200 | 0.33325 | -0.00125 | -0.00331 | 0.00080 | -0.00407 | -0.01698 | n/a | 1 | 0.00344 | inconclusive (interval spans zero) |
| player_fouls | defender_x_flank | count | 2679 | 2176 | 2679 | 1.34070 | 1.36194 | -0.02124 | -0.02924 | -0.01244 | -0.02045 | -0.07759 | n/a | 1 | 0.00102 | no room |
| player_fouls | defenders_only | count | 6758 | 4930 | 6758 | 1.34775 | 1.36630 | -0.01854 | -0.02382 | -0.01308 | -0.01912 | -0.08981 | 0.00102 | 3 | 0.00102 | no room |
| player_fouls | high_foul_referee | count | 6467 | 5214 | 6467 | 1.41045 | 1.41430 | -0.00385 | -0.00662 | -0.00119 | -0.00643 | -0.01414 | n/a | 1 | 0.00102 | no room |
| player_fouls | proxy_high | count | 6453 | 5214 | 6453 | 1.52382 | 1.55687 | -0.03305 | -0.04045 | -0.02562 | -0.04474 | -0.09620 | n/a | 1 | 0.00102 | no room |
| player_fouls | proxy_low | count | 6454 | 5095 | 6454 | 1.16531 | 1.16417 | 0.00115 | -0.00152 | 0.00384 | -0.00348 | -0.01175 | n/a | 1 | 0.00102 | inconclusive (interval spans zero) |
| player_passes | deep_opponent | count | 5348 | 4875 | 5348 | 3.96374 | 3.97350 | -0.00976 | -0.01670 | -0.00336 | -0.01115 | -0.04023 | n/a | 1 | 0.00211 | no room |
| player_passes | high_volume_passer | count | 5343 | 4679 | 5343 | 4.19962 | 4.20877 | -0.00914 | -0.01331 | -0.00488 | -0.00372 | -0.51423 | 0.00211 | 3 | 0.00211 | no room |
| player_passes | metronome_out | count | 4636 | 3555 | 4636 | 3.93146 | 3.94275 | -0.01130 | -0.01878 | -0.00348 | -0.01326 | -0.01391 | n/a | 1 | 0.00211 | no room |
| player_passes | proxy_high | count | 5343 | 4593 | 5343 | 4.20778 | 4.21461 | -0.00684 | -0.01140 | -0.00267 | -0.00612 | -0.63999 | n/a | 1 | 0.00211 | no room |
| player_passes | proxy_low | count | 5343 | 4832 | 5343 | 3.62460 | 3.62987 | -0.00527 | -0.00928 | -0.00115 | -0.00515 | -0.18755 | n/a | 1 | 0.00211 | no room |
| player_passes_completed | deep_opponent | count | 5348 | 4875 | 5348 | 3.81119 | 3.82107 | -0.00988 | -0.01983 | -0.00081 | -0.01220 | -0.04148 | n/a | 1 | 0.00107 | no room |
| player_passes_completed | high_volume_passer | count | 5343 | 4679 | 5343 | 4.09040 | 4.10227 | -0.01187 | -0.01761 | -0.00642 | -0.00298 | -0.49028 | 0.00107 | 3 | 0.00107 | no room |
| player_passes_completed | metronome_out | count | 4636 | 3555 | 4636 | 3.76383 | 3.77837 | -0.01455 | -0.02484 | -0.00370 | -0.01970 | -0.01716 | n/a | 1 | 0.00107 | no room |
| player_passes_completed | proxy_high | count | 5343 | 4449 | 5343 | 4.12269 | 4.12235 | 0.00034 | -0.00450 | 0.00547 | -0.00436 | -0.67511 | n/a | 1 | 0.00107 | inconclusive (interval spans zero) |
| player_passes_completed | proxy_low | count | 5343 | 4898 | 5343 | 3.39508 | 3.39815 | -0.00307 | -0.00750 | 0.00133 | 0.00495 | -0.33970 | n/a | 1 | 0.00107 | inconclusive (interval spans zero) |
| team_cards | high_foul_both | count | 20287 | 26015 | 20287 | 2.16481 | 2.16988 | -0.00506 | -0.00639 | -0.00367 | -0.00031 | -0.00461 | 0.00268 | 3 | 0.00268 | no room |
| team_cards | high_foul_tight | count | 6478 | 8545 | 6478 | 2.16719 | 2.17549 | -0.00829 | -0.01149 | -0.00479 | -0.00825 | -0.01826 | n/a | 1 | 0.00268 | no room |
| team_cards | proxy_high | count | 20287 | 33399 | 20287 | 2.17137 | 2.17591 | -0.00453 | -0.00568 | -0.00341 | -0.00092 | -0.00948 | n/a | 1 | 0.00268 | no room |
| team_cards | proxy_low | count | 20287 | 10517 | 20287 | 2.01886 | 2.01854 | 0.00032 | -0.00145 | 0.00215 | -0.00354 | -0.08588 | n/a | 1 | 0.00268 | inconclusive (interval spans zero) |
| team_corners | even_match | count | 40518 | 26692 | 40518 | 2.32385 | 2.32379 | 0.00006 | -0.00043 | 0.00056 | 0.00002 | -0.02681 | n/a | 1 | 0.00045 | inconclusive (interval spans zero) |
| team_corners | fav_strong | count | 12440 | 9272 | 12440 | 2.33149 | 2.32611 | 0.00538 | 0.00317 | 0.00744 | 0.00079 | -0.06504 | 0.00045 | 3 | 0.00045 | room |
| team_corners | high_total | count | 6634 | 9990 | 6634 | 2.34826 | 2.34706 | 0.00120 | -0.00089 | 0.00325 | -0.00159 | -0.00920 | n/a | 1 | 0.00045 | inconclusive (interval spans zero) |
| team_corners | proxy_high | count | 42152 | 12403 | 42152 | 2.47923 | 2.48043 | -0.00119 | -0.00188 | -0.00047 | 0.00025 | -0.07992 | n/a | 1 | 0.00045 | no room |
| team_corners | proxy_low | count | 42152 | 41892 | 42152 | 2.24646 | 2.24646 | 0.00000 | -0.00038 | 0.00039 | 0.00034 | -0.02424 | n/a | 1 | 0.00045 | inconclusive (interval spans zero) |
| team_fouls | away_underdog | count | 24634 | 14799 | 24634 | 2.70105 | 2.70175 | -0.00071 | -0.00200 | 0.00072 | -0.00164 | -0.00829 | n/a | 1 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | high_foul_division | count | 42016 | 16108 | 42016 | 2.78945 | 2.78993 | -0.00048 | -0.00134 | 0.00037 | 0.00033 | -0.02769 | n/a | 1 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | proxy_high | count | 42016 | 17507 | 42016 | 2.81284 | 2.81245 | 0.00039 | -0.00033 | 0.00109 | 0.00001 | -0.23142 | n/a | 1 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | proxy_low | count | 42016 | 32212 | 42016 | 2.59735 | 2.59756 | -0.00021 | -0.00085 | 0.00042 | 0.00078 | -0.19930 | n/a | 1 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | tight_match | count | 27220 | 21760 | 27220 | 2.70470 | 2.70557 | -0.00087 | -0.00189 | 0.00013 | -0.00233 | -0.01061 | 0.00028 | 3 | 0.00028 | inconclusive (interval spans zero) |

### Scenario definitions

| market | scenario | description |
|---|---|---|
| match_corners | even_match | no-vig favourite probability <= 0.42 |
| match_corners | fav_strong | no-vig favourite probability >= 0.65 |
| match_corners | high_total | P(over 2.5 goals) >= 0.60 |
| match_corners | proxy_high | book proxy (proxy_total) in the top discovery tercile (>= 10.942) |
| match_corners | proxy_low | book proxy (proxy_total) in the bottom discovery tercile (<= 10.291) |
| player_cards | fullback_only | full-backs only |
| player_cards | fullback_x_flank_x_ref | full-back, opponent flank dribbles in the top discovery tercile, card-happy referee |
| player_cards | proxy_high | book proxy (p0_logit) in the top discovery tercile (>= -1.408) |
| player_cards | proxy_low | book proxy (p0_logit) in the bottom discovery tercile (<= -1.794) |
| player_fouls | defender_x_flank | defender facing opponent flank dribbles in the top discovery tercile (>= 6.73 per match) |
| player_fouls | defenders_only | defenders only |
| player_fouls | high_foul_referee | referee's prior fouls per match in the top discovery tercile (>= 31.1) |
| player_fouls | proxy_high | book proxy (proxy_mu_fouls) in the top discovery tercile (>= 1.168) |
| player_fouls | proxy_low | book proxy (proxy_mu_fouls) in the bottom discovery tercile (<= 0.962) |
| player_passes | deep_opponent | opponent's imputed block depth in the deepest discovery tercile |
| player_passes | high_volume_passer | the player's prior attempts per 90 in the top discovery tercile |
| player_passes | metronome_out | the team's usual highest-volume passer is absent |
| player_passes | proxy_high | book proxy (mu_proxy_att) in the top discovery tercile (>= 44.948) |
| player_passes | proxy_low | book proxy (mu_proxy_att) in the bottom discovery tercile (<= 36.852) |
| player_passes_completed | deep_opponent | opponent's imputed block depth in the deepest discovery tercile |
| player_passes_completed | high_volume_passer | the player's prior attempts per 90 in the top discovery tercile |
| player_passes_completed | metronome_out | the team's usual highest-volume passer is absent |
| player_passes_completed | proxy_high | book proxy (mu_proxy_comp) in the top discovery tercile (>= 35.199) |
| player_passes_completed | proxy_low | book proxy (mu_proxy_comp) in the bottom discovery tercile (<= 27.553) |
| team_cards | high_foul_both | both teams' prior foul rates sum into the top discovery tercile |
| team_cards | high_foul_tight | the same, and the Elo gap in the bottom discovery tercile |
| team_cards | proxy_high | book proxy (proxy_mu_cards) in the top discovery tercile (>= 3.545) |
| team_cards | proxy_low | book proxy (proxy_mu_cards) in the bottom discovery tercile (<= 3.020) |
| team_corners | even_match | no-vig favourite probability <= 0.42 |
| team_corners | fav_strong | no-vig favourite probability >= 0.65 |
| team_corners | high_total | P(over 2.5 goals) >= 0.60 |
| team_corners | proxy_high | book proxy (proxy_team) in the top discovery tercile (>= 5.659) |
| team_corners | proxy_low | book proxy (proxy_team) in the bottom discovery tercile (<= 4.878) |
| team_fouls | away_underdog | away side with a no-vig win probability <= 0.25 |
| team_fouls | high_foul_division | division-season foul level in the top discovery tercile (>= 13.94 per team-match) |
| team_fouls | proxy_high | book proxy (proxy_fouls) in the top discovery tercile (>= 13.950) |
| team_fouls | proxy_low | book proxy (proxy_fouls) in the bottom discovery tercile (<= 11.690) |
| team_fouls | tight_match | |Elo gap| in the bottom discovery tercile (<= 48) |

## Reading it

- **The specialist loses hardest where the scenario is narrowest.** Across the 38 cells the delta correlates +0.56 with the log of the scenario's training rows, and the median delta rises from -0.00945 in the smallest third of scenarios to -0.00167 and -0.00071. What looks like a targeting opportunity is mostly a sample-size cut; the correlation is not perfect, so size is not the whole story, but it is most of it.
- **Off-scenario, the specialist loses too** (37 of 38 cells), which is the expected direction and confirms the specialists are working models rather than broken ones.
- **Discovery and confirmation agree on the sign in 29 of 38 cells.** Where they do not, the cell is inconclusive and is labelled as such rather than read as a result.
- **The one structural caveat.** This tests whether *refitting the same model family on a subset* helps. It does not test whether a different feature set, specific to a scenario, could help -- but stages 01-04 tested exactly that for their own hypotheses (the full-back / flank-dribbler / referee interaction for cards, crossing style against compact defences for corners, the absent metronome for passes, the foul matchup for fouls) and exactly one *target* produced channels that cleared zero on their confirmation half: fouls won, where both the opponent's foul propensity (+0.0023) and the dribble-side block (+0.0013) did. Stage 04 section 3.1 states how far that goes and no further -- both were inside noise on discovery, they sit at about 3x and 1.7x their own five-seed refit spreads, and a few thousandths of a nat per appearance is a mechanism finding and not a price.

## Per-cell detail

All four regions and both halves, as fitted. The per-row `verdict` is computed cell by cell, so `discovery_cv` rows carry one too; those are exploratory and do not enter the counts above, which use confirmation rows only.

| target_label | scenario | where | region | n | n_matches | mean_y | loss_generalist | loss_specialist | delta_gen_minus_spec | ci_lo | ci_hi | refit_spread | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| match_corners | even_match | confirmation | in_scenario | 13346 | 13346 | 9.61839 | 2.60185 | 2.60309 | -0.00125 | -0.00210 | -0.00040 | n/a | no room |
| match_corners | even_match | confirmation | off_scenario | 28805 | 28805 | 9.84687 | 2.62025 | 2.62011 | 0.00014 | -0.00070 | 0.00098 | n/a | inconclusive (interval spans zero) |
| match_corners | even_match | discovery_cv | in_scenario | 20259 | 20259 | 10.49534 | 2.65332 | 2.65422 | -0.00090 | -0.00162 | -0.00020 | n/a | no room |
| match_corners | even_match | discovery_cv | off_scenario | 42968 | 42968 | 10.57554 | 2.66128 | 2.66284 | -0.00156 | -0.00224 | -0.00090 | n/a | no room |
| match_corners | fav_strong | confirmation | in_scenario | 4636 | 4636 | 10.20125 | 2.65756 | 2.66254 | -0.00499 | -0.00813 | -0.00212 | 0.00009 | no room |
| match_corners | fav_strong | confirmation | off_scenario | 37515 | 37515 | 9.72179 | 2.60909 | 2.61466 | -0.00557 | -0.00673 | -0.00437 | 0.00099 | no room |
| match_corners | fav_strong | discovery_cv | in_scenario | 6220 | 6220 | 10.67428 | 2.68383 | 2.68806 | -0.00423 | -0.00658 | -0.00163 | 0.00265 | no room |
| match_corners | fav_strong | discovery_cv | off_scenario | 57007 | 57007 | 10.53627 | 2.65599 | 2.66191 | -0.00592 | -0.00680 | -0.00505 | 0.00024 | no room |
| match_corners | high_total | confirmation | in_scenario | 4995 | 4995 | 10.30370 | 2.66240 | 2.66773 | -0.00533 | -0.00794 | -0.00282 | n/a | no room |
| match_corners | high_total | confirmation | off_scenario | 37156 | 37156 | 9.70339 | 2.60797 | 2.62325 | -0.01528 | -0.01693 | -0.01363 | n/a | no room |
| match_corners | high_total | discovery_cv | in_scenario | 3317 | 3317 | 10.62466 | 2.67296 | 2.67639 | -0.00344 | -0.00660 | -0.00013 | n/a | no room |
| match_corners | high_total | discovery_cv | off_scenario | 59910 | 59910 | 10.54570 | 2.65794 | 2.66798 | -0.01004 | -0.01111 | -0.00899 | n/a | no room |
| match_corners | proxy_high | confirmation | in_scenario | 2360 | 2360 | 10.96229 | 2.68357 | 2.68525 | -0.00168 | -0.00357 | 0.00010 | n/a | inconclusive (interval spans zero) |
| match_corners | proxy_high | confirmation | off_scenario | 39791 | 39791 | 9.70408 | 2.61032 | 2.67217 | -0.06185 | -0.06494 | -0.05874 | n/a | no room |
| match_corners | proxy_high | discovery_cv | in_scenario | 21076 | 21076 | 11.23197 | 2.68651 | 2.68668 | -0.00018 | -0.00082 | 0.00046 | n/a | inconclusive (interval spans zero) |
| match_corners | proxy_high | discovery_cv | off_scenario | 42151 | 42151 | 10.20877 | 2.64484 | 2.67530 | -0.03046 | -0.03280 | -0.02797 | n/a | no room |
| match_corners | proxy_low | confirmation | in_scenario | 31208 | 31208 | 9.54428 | 2.60225 | 2.60340 | -0.00114 | -0.00157 | -0.00070 | n/a | no room |
| match_corners | proxy_low | confirmation | off_scenario | 10943 | 10943 | 10.43114 | 2.64913 | 2.65082 | -0.00168 | -0.00401 | 0.00066 | n/a | inconclusive (interval spans zero) |
| match_corners | proxy_low | discovery_cv | in_scenario | 21076 | 21076 | 9.82240 | 2.62739 | 2.62814 | -0.00075 | -0.00136 | -0.00016 | n/a | no room |
| match_corners | proxy_low | discovery_cv | off_scenario | 42151 | 42151 | 10.91357 | 2.67440 | 2.69308 | -0.01868 | -0.02051 | -0.01689 | n/a | no room |
| player_cards | fullback_only | confirmation | in_scenario | 2695 | 607 | 0.18776 | 0.47378 | 0.47514 | -0.00136 | -0.00554 | 0.00283 | 0.00344 | inconclusive (interval spans zero) |
| player_cards | fullback_only | confirmation | off_scenario | 12453 | 607 | 0.16366 | 0.42118 | 0.43810 | -0.01692 | -0.01945 | -0.01432 | 0.00090 | no room |
| player_cards | fullback_only | discovery_cv | in_scenario | 3194 | 790 | 0.20288 | 0.49000 | 0.49561 | -0.00561 | -0.00975 | -0.00139 | 0.00031 | no room |
| player_cards | fullback_only | discovery_cv | off_scenario | 14987 | 791 | 0.17041 | 0.43373 | 0.44750 | -0.01377 | -0.01631 | -0.01120 | 0.00112 | no room |
| player_cards | fullback_x_flank_x_ref | confirmation | in_scenario | 400 | 169 | 0.22500 | 0.52399 | 0.54479 | -0.02080 | -0.04088 | -0.00064 | n/a | no room |
| player_cards | fullback_x_flank_x_ref | confirmation | off_scenario | 14748 | 607 | 0.16640 | 0.42800 | 0.47606 | -0.04806 | -0.05253 | -0.04335 | n/a | no room |
| player_cards | fullback_x_flank_x_ref | discovery_cv | in_scenario | 348 | 176 | 0.27874 | 0.58520 | 0.58505 | 0.00014 | -0.01808 | 0.02027 | n/a | inconclusive (interval spans zero) |
| player_cards | fullback_x_flank_x_ref | discovery_cv | off_scenario | 17833 | 791 | 0.17412 | 0.44085 | 0.48366 | -0.04281 | -0.04684 | -0.03873 | n/a | no room |
| player_cards | proxy_high | confirmation | in_scenario | 5406 | 485 | 0.24639 | 0.54810 | 0.54977 | -0.00167 | -0.00390 | 0.00050 | n/a | inconclusive (interval spans zero) |
| player_cards | proxy_high | confirmation | off_scenario | 9742 | 607 | 0.12441 | 0.36530 | 0.37668 | -0.01138 | -0.01406 | -0.00874 | n/a | no room |
| player_cards | proxy_high | discovery_cv | in_scenario | 6061 | 699 | 0.24649 | 0.54814 | 0.55162 | -0.00348 | -0.00603 | -0.00096 | n/a | no room |
| player_cards | proxy_high | discovery_cv | off_scenario | 12120 | 791 | 0.14092 | 0.39134 | 0.40197 | -0.01063 | -0.01294 | -0.00813 | n/a | no room |
| player_cards | proxy_low | confirmation | in_scenario | 5621 | 591 | 0.10639 | 0.33200 | 0.33325 | -0.00125 | -0.00331 | 0.00080 | n/a | inconclusive (interval spans zero) |
| player_cards | proxy_low | confirmation | off_scenario | 9527 | 605 | 0.20426 | 0.48867 | 0.50565 | -0.01698 | -0.02056 | -0.01352 | n/a | no room |
| player_cards | proxy_low | discovery_cv | in_scenario | 6061 | 773 | 0.11087 | 0.33756 | 0.34162 | -0.00407 | -0.00655 | -0.00165 | n/a | no room |
| player_cards | proxy_low | discovery_cv | off_scenario | 12120 | 790 | 0.20875 | 0.49665 | 0.51055 | -0.01390 | -0.01711 | -0.01068 | n/a | no room |
| player_fouls | defender_x_flank | confirmation | in_scenario | 2176 | 448 | 1.07445 | 1.34070 | 1.36194 | -0.02124 | -0.02924 | -0.01244 | n/a | no room |
| player_fouls | defender_x_flank | confirmation | off_scenario | 11794 | 559 | 1.17043 | 1.35183 | 1.42942 | -0.07759 | -0.08467 | -0.07087 | n/a | no room |
| player_fouls | defender_x_flank | discovery_cv | in_scenario | 2679 | 691 | 1.11310 | 1.34611 | 1.36656 | -0.02045 | -0.02807 | -0.01274 | n/a | no room |
| player_fouls | defender_x_flank | discovery_cv | off_scenario | 16680 | 839 | 1.21013 | 1.38176 | 1.45669 | -0.07493 | -0.08037 | -0.06941 | n/a | no room |
| player_fouls | defenders_only | confirmation | in_scenario | 4930 | 559 | 1.09310 | 1.34775 | 1.36630 | -0.01854 | -0.02382 | -0.01308 | 0.00102 | no room |
| player_fouls | defenders_only | confirmation | off_scenario | 9040 | 559 | 1.18949 | 1.35137 | 1.44117 | -0.08981 | -0.09822 | -0.08150 | 0.00027 | no room |
| player_fouls | defenders_only | discovery_cv | in_scenario | 6758 | 838 | 1.11423 | 1.35357 | 1.37270 | -0.01912 | -0.02362 | -0.01459 | 0.00104 | no room |
| player_fouls | defenders_only | discovery_cv | off_scenario | 12601 | 839 | 1.24093 | 1.38930 | 1.48158 | -0.09228 | -0.09958 | -0.08526 | 0.00239 | no room |
| player_fouls | high_foul_referee | confirmation | in_scenario | 5214 | 207 | 1.26295 | 1.41045 | 1.41430 | -0.00385 | -0.00662 | -0.00119 | n/a | no room |
| player_fouls | high_foul_referee | confirmation | off_scenario | 8756 | 352 | 1.09148 | 1.31415 | 1.32829 | -0.01414 | -0.01740 | -0.01090 | n/a | no room |
| player_fouls | high_foul_referee | discovery_cv | in_scenario | 6467 | 271 | 1.32859 | 1.44133 | 1.44775 | -0.00643 | -0.00923 | -0.00342 | n/a | no room |
| player_fouls | high_foul_referee | discovery_cv | off_scenario | 12892 | 568 | 1.13055 | 1.34447 | 1.36315 | -0.01868 | -0.02170 | -0.01540 | n/a | no room |
| player_fouls | proxy_high | confirmation | in_scenario | 5214 | 559 | 1.51438 | 1.52382 | 1.55687 | -0.03305 | -0.04045 | -0.02562 | n/a | no room |
| player_fouls | proxy_high | confirmation | off_scenario | 8756 | 559 | 0.94175 | 1.24664 | 1.34284 | -0.09620 | -0.10374 | -0.08844 | n/a | no room |
| player_fouls | proxy_high | discovery_cv | in_scenario | 6453 | 837 | 1.58066 | 1.55850 | 1.60324 | -0.04474 | -0.05145 | -0.03800 | n/a | no room |
| player_fouls | proxy_high | discovery_cv | off_scenario | 12906 | 839 | 1.00473 | 1.28599 | 1.38755 | -0.10156 | -0.10847 | -0.09461 | n/a | no room |
| player_fouls | proxy_low | confirmation | in_scenario | 5095 | 559 | 0.80177 | 1.16531 | 1.16417 | 0.00115 | -0.00152 | 0.00384 | n/a | inconclusive (interval spans zero) |
| player_fouls | proxy_low | confirmation | off_scenario | 8875 | 559 | 1.35854 | 1.45617 | 1.46792 | -0.01175 | -0.01588 | -0.00773 | n/a | no room |
| player_fouls | proxy_low | discovery_cv | in_scenario | 6454 | 838 | 0.88519 | 1.20791 | 1.21139 | -0.00348 | -0.00591 | -0.00111 | n/a | no room |
| player_fouls | proxy_low | discovery_cv | off_scenario | 12905 | 839 | 1.35250 | 1.46130 | 1.47350 | -0.01220 | -0.01504 | -0.00938 | n/a | no room |
| player_passes | deep_opponent | confirmation | in_scenario | 4875 | 335 | 43.22113 | 3.96374 | 3.97350 | -0.00976 | -0.01670 | -0.00336 | n/a | no room |
| player_passes | deep_opponent | confirmation | off_scenario | 8178 | 488 | 40.90805 | 3.91286 | 3.95309 | -0.04023 | -0.05007 | -0.03009 | n/a | no room |
| player_passes | deep_opponent | discovery_cv | in_scenario | 5348 | 412 | 43.23822 | 3.96616 | 3.97731 | -0.01115 | -0.01792 | -0.00393 | n/a | no room |
| player_passes | deep_opponent | discovery_cv | off_scenario | 10680 | 676 | 40.90758 | 3.94044 | 3.97168 | -0.03124 | -0.03897 | -0.02388 | n/a | no room |
| player_passes | high_volume_passer | confirmation | in_scenario | 4679 | 602 | 55.92092 | 4.19962 | 4.20877 | -0.00914 | -0.01331 | -0.00488 | 0.00211 | no room |
| player_passes | high_volume_passer | confirmation | off_scenario | 8374 | 607 | 33.86613 | 3.78225 | 4.29648 | -0.51423 | -0.53928 | -0.48862 | 0.00927 | no room |
| player_passes | high_volume_passer | discovery_cv | in_scenario | 5343 | 779 | 55.77971 | 4.24010 | 4.24382 | -0.00372 | -0.00758 | -0.00001 | 0.00084 | no room |
| player_passes | high_volume_passer | discovery_cv | off_scenario | 10685 | 790 | 34.63734 | 3.80346 | 4.27210 | -0.46863 | -0.48790 | -0.44910 | 0.05572 | no room |
| player_passes | metronome_out | confirmation | in_scenario | 3555 | 285 | 41.69986 | 3.93146 | 3.94275 | -0.01130 | -0.01878 | -0.00348 | n/a | no room |
| player_passes | metronome_out | confirmation | off_scenario | 9498 | 559 | 41.79891 | 3.93201 | 3.94592 | -0.01391 | -0.01924 | -0.00879 | n/a | no room |
| player_passes | metronome_out | discovery_cv | in_scenario | 4636 | 390 | 41.18076 | 3.93846 | 3.95172 | -0.01326 | -0.02002 | -0.00698 | n/a | no room |
| player_passes | metronome_out | discovery_cv | off_scenario | 11392 | 720 | 41.89054 | 3.95332 | 3.96748 | -0.01417 | -0.01868 | -0.00977 | n/a | no room |
| player_passes | proxy_high | confirmation | in_scenario | 4593 | 604 | 56.25561 | 4.20778 | 4.21461 | -0.00684 | -0.01140 | -0.00267 | n/a | no room |
| player_passes | proxy_high | confirmation | off_scenario | 8460 | 607 | 33.90863 | 3.78206 | 4.42205 | -0.63999 | -0.66532 | -0.61322 | n/a | no room |
| player_passes | proxy_high | discovery_cv | in_scenario | 5343 | 785 | 56.04286 | 4.23722 | 4.24333 | -0.00612 | -0.00979 | -0.00218 | n/a | no room |
| player_passes | proxy_high | discovery_cv | off_scenario | 10685 | 790 | 34.50576 | 3.80491 | 4.44395 | -0.63905 | -0.66145 | -0.61590 | n/a | no room |
| player_passes | proxy_low | confirmation | in_scenario | 4832 | 607 | 28.68233 | 3.62460 | 3.62987 | -0.00527 | -0.00928 | -0.00115 | n/a | no room |
| player_passes | proxy_low | confirmation | off_scenario | 8221 | 607 | 49.46552 | 4.11245 | 4.30000 | -0.18755 | -0.21001 | -0.16493 | n/a | no room |
| player_passes | proxy_low | discovery_cv | in_scenario | 5343 | 790 | 28.17649 | 3.61848 | 3.62362 | -0.00515 | -0.00943 | -0.00105 | n/a | no room |
| player_passes | proxy_low | discovery_cv | off_scenario | 10685 | 790 | 48.44024 | 4.11430 | 4.33001 | -0.21570 | -0.23760 | -0.19343 | n/a | no room |
| player_passes_completed | deep_opponent | confirmation | in_scenario | 4875 | 335 | 33.52410 | 3.81119 | 3.82107 | -0.00988 | -0.01983 | -0.00081 | n/a | no room |
| player_passes_completed | deep_opponent | confirmation | off_scenario | 8178 | 488 | 31.28075 | 3.74710 | 3.78858 | -0.04148 | -0.05418 | -0.02897 | n/a | no room |
| player_passes_completed | deep_opponent | discovery_cv | in_scenario | 5348 | 412 | 33.54506 | 3.79797 | 3.81017 | -0.01220 | -0.01992 | -0.00410 | n/a | no room |
| player_passes_completed | deep_opponent | discovery_cv | off_scenario | 10680 | 676 | 31.15365 | 3.76284 | 3.79876 | -0.03592 | -0.04629 | -0.02628 | n/a | no room |
| player_passes_completed | high_volume_passer | confirmation | in_scenario | 4679 | 602 | 45.45971 | 4.09040 | 4.10227 | -0.01187 | -0.01761 | -0.00642 | 0.00107 | no room |
| player_passes_completed | high_volume_passer | confirmation | off_scenario | 8374 | 607 | 24.66420 | 3.59260 | 4.08287 | -0.49028 | -0.51615 | -0.46433 | 0.04277 | no room |
| player_passes_completed | high_volume_passer | discovery_cv | in_scenario | 5343 | 779 | 45.09751 | 4.11864 | 4.12161 | -0.00298 | -0.00782 | 0.00188 | 0.00162 | inconclusive (interval spans zero) |
| player_passes_completed | high_volume_passer | discovery_cv | off_scenario | 10685 | 790 | 25.37801 | 3.60251 | 4.12291 | -0.52039 | -0.54218 | -0.49882 | 0.04189 | no room |
| player_passes_completed | metronome_out | confirmation | in_scenario | 3555 | 285 | 32.05963 | 3.76383 | 3.77837 | -0.01455 | -0.02484 | -0.00370 | n/a | no room |
| player_passes_completed | metronome_out | confirmation | off_scenario | 9498 | 559 | 32.14066 | 3.77374 | 3.79090 | -0.01716 | -0.02483 | -0.01023 | n/a | no room |
| player_passes_completed | metronome_out | discovery_cv | in_scenario | 4636 | 390 | 31.33865 | 3.75179 | 3.77149 | -0.01970 | -0.02942 | -0.00996 | n/a | no room |
| player_passes_completed | metronome_out | discovery_cv | off_scenario | 11392 | 720 | 32.20102 | 3.78384 | 3.79810 | -0.01426 | -0.02049 | -0.00861 | n/a | no room |
| player_passes_completed | proxy_high | confirmation | in_scenario | 4449 | 596 | 46.31355 | 4.12269 | 4.12235 | 0.00034 | -0.00450 | 0.00547 | n/a | inconclusive (interval spans zero) |
| player_passes_completed | proxy_high | confirmation | off_scenario | 8604 | 607 | 24.77859 | 3.58920 | 4.26431 | -0.67511 | -0.70436 | -0.64563 | n/a | no room |
| player_passes_completed | proxy_high | discovery_cv | in_scenario | 5343 | 778 | 45.56972 | 4.14086 | 4.14522 | -0.00436 | -0.00912 | 0.00019 | n/a | inconclusive (interval spans zero) |
| player_passes_completed | proxy_high | discovery_cv | off_scenario | 10685 | 790 | 25.14188 | 3.59140 | 4.28709 | -0.69569 | -0.72148 | -0.67039 | n/a | no room |
| player_passes_completed | proxy_low | confirmation | in_scenario | 4898 | 607 | 19.83381 | 3.39508 | 3.39815 | -0.00307 | -0.00750 | 0.00133 | n/a | inconclusive (interval spans zero) |
| player_passes_completed | proxy_low | confirmation | off_scenario | 8155 | 607 | 39.49700 | 3.99684 | 4.33654 | -0.33970 | -0.37465 | -0.30547 | n/a | no room |
| player_passes_completed | proxy_low | discovery_cv | in_scenario | 5343 | 789 | 19.02882 | 3.34192 | 3.33697 | 0.00495 | 0.00027 | 0.00974 | n/a | room |
| player_passes_completed | proxy_low | discovery_cv | off_scenario | 10685 | 790 | 38.41357 | 3.99091 | 4.34998 | -0.35908 | -0.39282 | -0.32411 | n/a | no room |
| team_cards | high_foul_both | confirmation | in_scenario | 26015 | 26015 | 4.54645 | 2.16481 | 2.16988 | -0.00506 | -0.00639 | -0.00367 | 0.00268 | no room |
| team_cards | high_foul_both | confirmation | off_scenario | 38535 | 38535 | 3.59696 | 2.07323 | 2.07784 | -0.00461 | -0.00611 | -0.00309 | 0.00262 | no room |
| team_cards | high_foul_both | discovery_cv | in_scenario | 20287 | 20287 | 4.14162 | 2.06274 | 2.06305 | -0.00031 | -0.00148 | 0.00093 | 0.00125 | inconclusive (interval spans zero) |
| team_cards | high_foul_both | discovery_cv | off_scenario | 40574 | 40574 | 3.00545 | 1.98275 | 1.99906 | -0.01631 | -0.01818 | -0.01453 | 0.00192 | no room |
| team_cards | high_foul_tight | confirmation | in_scenario | 8545 | 8545 | 4.71949 | 2.16719 | 2.17549 | -0.00829 | -0.01149 | -0.00479 | n/a | no room |
| team_cards | high_foul_tight | confirmation | off_scenario | 56005 | 56005 | 3.86674 | 2.10144 | 2.11970 | -0.01826 | -0.02000 | -0.01652 | n/a | no room |
| team_cards | high_foul_tight | discovery_cv | in_scenario | 6478 | 6478 | 4.23232 | 2.06173 | 2.06998 | -0.00825 | -0.01246 | -0.00409 | n/a | no room |
| team_cards | high_foul_tight | discovery_cv | off_scenario | 54383 | 54383 | 3.28314 | 2.00318 | 2.02466 | -0.02147 | -0.02328 | -0.01967 | n/a | no room |
| team_cards | proxy_high | confirmation | in_scenario | 33399 | 33399 | 4.49052 | 2.17137 | 2.17591 | -0.00453 | -0.00568 | -0.00341 | n/a | no room |
| team_cards | proxy_high | confirmation | off_scenario | 31151 | 31151 | 3.43186 | 2.04449 | 2.05397 | -0.00948 | -0.01144 | -0.00758 | n/a | no room |
| team_cards | proxy_high | discovery_cv | in_scenario | 20287 | 20287 | 4.18524 | 2.07828 | 2.07921 | -0.00092 | -0.00217 | 0.00028 | n/a | inconclusive (interval spans zero) |
| team_cards | proxy_high | discovery_cv | off_scenario | 40574 | 40574 | 2.98363 | 1.97498 | 1.99408 | -0.01911 | -0.02106 | -0.01721 | n/a | no room |
| team_cards | proxy_low | confirmation | in_scenario | 10517 | 10517 | 3.23305 | 2.01886 | 2.01854 | 0.00032 | -0.00145 | 0.00215 | n/a | inconclusive (interval spans zero) |
| team_cards | proxy_low | confirmation | off_scenario | 54033 | 54033 | 4.12494 | 2.12791 | 2.21379 | -0.08588 | -0.09003 | -0.08161 | n/a | no room |
| team_cards | proxy_low | discovery_cv | in_scenario | 20287 | 20287 | 2.78153 | 1.94668 | 1.95022 | -0.00354 | -0.00491 | -0.00211 | n/a | no room |
| team_cards | proxy_low | discovery_cv | off_scenario | 40574 | 40574 | 3.68549 | 2.04078 | 2.10641 | -0.06562 | -0.06932 | -0.06201 | n/a | no room |
| team_corners | even_match | confirmation | in_scenario | 26692 | 13346 | 4.80919 | 2.32385 | 2.32379 | 0.00006 | -0.00043 | 0.00056 | n/a | inconclusive (interval spans zero) |
| team_corners | even_match | confirmation | off_scenario | 57610 | 28805 | 4.92343 | 2.32328 | 2.35009 | -0.02681 | -0.02862 | -0.02509 | n/a | no room |
| team_corners | even_match | discovery_cv | in_scenario | 40518 | 20259 | 5.24767 | 2.35120 | 2.35118 | 0.00002 | -0.00037 | 0.00043 | n/a | inconclusive (interval spans zero) |
| team_corners | even_match | discovery_cv | off_scenario | 85936 | 42968 | 5.28777 | 2.35478 | 2.37401 | -0.01924 | -0.02066 | -0.01789 | n/a | no room |
| team_corners | fav_strong | confirmation | in_scenario | 9272 | 4636 | 5.10063 | 2.33149 | 2.32611 | 0.00538 | 0.00317 | 0.00744 | 0.00045 | room |
| team_corners | fav_strong | confirmation | off_scenario | 75030 | 37515 | 4.86090 | 2.32247 | 2.38751 | -0.06504 | -0.06710 | -0.06289 | 0.00793 | no room |
| team_corners | fav_strong | discovery_cv | in_scenario | 12440 | 6220 | 5.33714 | 2.34456 | 2.34377 | 0.00079 | -0.00085 | 0.00244 | 0.00115 | inconclusive (interval spans zero) |
| team_corners | fav_strong | discovery_cv | off_scenario | 114014 | 57007 | 5.26813 | 2.35462 | 2.39104 | -0.03642 | -0.03789 | -0.03483 | 0.01474 | no room |
| team_corners | high_total | confirmation | in_scenario | 9990 | 4995 | 5.15185 | 2.34826 | 2.34706 | 0.00120 | -0.00089 | 0.00325 | n/a | inconclusive (interval spans zero) |
| team_corners | high_total | confirmation | off_scenario | 74312 | 37156 | 4.85169 | 2.32012 | 2.32933 | -0.00920 | -0.01013 | -0.00825 | n/a | no room |
| team_corners | high_total | discovery_cv | in_scenario | 6634 | 3317 | 5.31233 | 2.35077 | 2.35236 | -0.00159 | -0.00412 | 0.00108 | n/a | inconclusive (interval spans zero) |
| team_corners | high_total | discovery_cv | off_scenario | 119820 | 59910 | 5.27285 | 2.35379 | 2.36063 | -0.00684 | -0.00748 | -0.00614 | n/a | no room |
| team_corners | proxy_high | confirmation | in_scenario | 12403 | 12369 | 6.25712 | 2.47923 | 2.48043 | -0.00119 | -0.00188 | -0.00047 | n/a | no room |
| team_corners | proxy_high | confirmation | off_scenario | 71899 | 42117 | 4.65095 | 2.29659 | 2.37651 | -0.07992 | -0.08231 | -0.07748 | n/a | no room |
| team_corners | proxy_high | discovery_cv | in_scenario | 42152 | 41292 | 6.27557 | 2.45998 | 2.45973 | 0.00025 | -0.00018 | 0.00069 | n/a | inconclusive (interval spans zero) |
| team_corners | proxy_high | discovery_cv | off_scenario | 84302 | 62367 | 4.77458 | 2.30045 | 2.36814 | -0.06769 | -0.07006 | -0.06531 | n/a | no room |
| team_corners | proxy_low | confirmation | in_scenario | 41892 | 36007 | 4.25313 | 2.24646 | 2.24646 | 0.00000 | -0.00038 | 0.00039 | n/a | inconclusive (interval spans zero) |
| team_corners | proxy_low | confirmation | off_scenario | 42410 | 36266 | 5.51365 | 2.39952 | 2.42376 | -0.02424 | -0.02673 | -0.02189 | n/a | no room |
| team_corners | proxy_low | discovery_cv | in_scenario | 42152 | 40285 | 4.32653 | 2.25123 | 2.25090 | 0.00034 | -0.00010 | 0.00073 | n/a | inconclusive (interval spans zero) |
| team_corners | proxy_low | discovery_cv | off_scenario | 84302 | 61360 | 5.74913 | 2.40483 | 2.45312 | -0.04829 | -0.05048 | -0.04615 | n/a | no room |
| team_fouls | away_underdog | confirmation | in_scenario | 14799 | 14799 | 12.42239 | 2.70105 | 2.70175 | -0.00071 | -0.00200 | 0.00072 | n/a | inconclusive (interval spans zero) |
| team_fouls | away_underdog | confirmation | off_scenario | 69231 | 42015 | 12.28100 | 2.67648 | 2.68476 | -0.00829 | -0.00933 | -0.00723 | n/a | no room |
| team_fouls | away_underdog | discovery_cv | in_scenario | 24634 | 24634 | 13.63055 | 2.77311 | 2.77475 | -0.00164 | -0.00282 | -0.00044 | n/a | no room |
| team_fouls | away_underdog | discovery_cv | off_scenario | 101412 | 63023 | 12.89277 | 2.73229 | 2.74592 | -0.01363 | -0.01466 | -0.01262 | n/a | no room |
| team_fouls | high_foul_division | confirmation | in_scenario | 16108 | 8054 | 14.62391 | 2.78945 | 2.78993 | -0.00048 | -0.00134 | 0.00037 | n/a | inconclusive (interval spans zero) |
| team_fouls | high_foul_division | confirmation | off_scenario | 67922 | 33961 | 11.75618 | 2.65504 | 2.68273 | -0.02769 | -0.02940 | -0.02599 | n/a | no room |
| team_fouls | high_foul_division | discovery_cv | in_scenario | 42016 | 21008 | 15.82980 | 2.85727 | 2.85693 | 0.00033 | -0.00026 | 0.00094 | n/a | inconclusive (interval spans zero) |
| team_fouls | high_foul_division | discovery_cv | off_scenario | 84030 | 42015 | 11.64051 | 2.68176 | 2.71895 | -0.03719 | -0.03906 | -0.03539 | n/a | no room |
| team_fouls | proxy_high | confirmation | in_scenario | 17507 | 12853 | 15.13492 | 2.81284 | 2.81245 | 0.00039 | -0.00033 | 0.00109 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_high | confirmation | off_scenario | 66523 | 37361 | 11.56138 | 2.64606 | 2.87747 | -0.23142 | -0.23648 | -0.22651 | n/a | no room |
| team_fouls | proxy_high | discovery_cv | in_scenario | 42016 | 26650 | 16.13226 | 2.87233 | 2.87232 | 0.00001 | -0.00051 | 0.00055 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_high | discovery_cv | off_scenario | 84030 | 47657 | 11.48928 | 2.67423 | 2.90099 | -0.22676 | -0.23160 | -0.22191 | n/a | no room |
| team_fouls | proxy_low | confirmation | in_scenario | 32212 | 22618 | 10.45039 | 2.59735 | 2.59756 | -0.00021 | -0.00085 | 0.00042 | n/a | inconclusive (interval spans zero) |
| team_fouls | proxy_low | confirmation | off_scenario | 51818 | 32421 | 13.45936 | 2.73268 | 2.93198 | -0.19930 | -0.20587 | -0.19243 | n/a | no room |
| team_fouls | proxy_low | discovery_cv | in_scenario | 42016 | 28813 | 10.42262 | 2.62134 | 2.62055 | 0.00078 | 0.00026 | 0.00135 | n/a | room |
| team_fouls | proxy_low | discovery_cv | off_scenario | 84030 | 49820 | 14.34416 | 2.79973 | 3.21830 | -0.41856 | -0.42760 | -0.40983 | n/a | no room |
| team_fouls | tight_match | confirmation | in_scenario | 21760 | 10880 | 12.90492 | 2.70470 | 2.70557 | -0.00087 | -0.00189 | 0.00013 | 0.00028 | inconclusive (interval spans zero) |
| team_fouls | tight_match | confirmation | off_scenario | 62270 | 31135 | 12.09658 | 2.67246 | 2.68306 | -0.01061 | -0.01184 | -0.00943 | 0.00071 | no room |
| team_fouls | tight_match | discovery_cv | in_scenario | 27220 | 13610 | 14.17796 | 2.77798 | 2.78031 | -0.00233 | -0.00335 | -0.00132 | 0.00074 | no room |
| team_fouls | tight_match | discovery_cv | off_scenario | 98826 | 49413 | 12.72269 | 2.72988 | 2.74094 | -0.01106 | -0.01205 | -0.01010 | 0.00050 | no room |

