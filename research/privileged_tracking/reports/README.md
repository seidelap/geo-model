- Deliverable state: every stage module, report and result table under
  `research/privileged_tracking/` is committed on branch `claude/soccer-betting-edge-29h9yo`
  (phase 1 in commit 25e942d, phase 2 including this synthesis in the following commit);
  processed data and model bundles under `data/raw/privileged/` are not committed.

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
