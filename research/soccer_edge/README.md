# Soccer edge detectors (superseded, kept for reference)

Started as a test of five retail-scale betting hypotheses (fullback cards from
dribbler volume, corners from crossing style, pass lines after role changes,
late-season motivation, top-down line shopping). The work was redirected to the
privileged-information tracking research in `research/privileged_tracking/`
before hypotheses 1-4 were run. What exists here:

- `sb_parse.py`: StatsBomb open-data parser producing `matches`, `player_match`,
  `team_match`, `fouls` and `shapes360` Parquet tables (used by the tracking
  research as well). Run with `SOCCER_EDGE_DATA=<dir with events/lineups/three-sixty/matches>`.
- `common.py`: odds-table loader, no-vig conversion, bootstrap helpers.
- `h4_motivation.py`: hypothesis 4 (late-season dead rubbers), written but never
  executed; treat as untested.
- `h5_lines.py`: hypothesis 5, top-down line shopping, on 211k matches (2005-2026,
  football-data.co.uk via a GitHub mirror). Result: backing the best of ~17 books
  when its implied probability sits at least 3 points below Bet365's no-vig price
  returned roughly +7% ROI (95% CI +2% to +12%, n=5,173 bets, 0.8% of outcomes)
  across all divisions, rising with the threshold but shrinking to a noisy zero in
  the top-5 leagues (n=600, CI -17% to +15%). Bet365's own price on those rows lost
  17%, so the signal is the outlier book, not the consensus. Caveat: near-closing
  odds, not closing, and no Pinnacle (blocked from this environment).

Data sources blocked from this sandbox: ESPN, football-data.co.uk, Understat,
FBref, Kaggle. Only GitHub raw content and PyPI were reachable.
