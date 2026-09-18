"""Scenario-specific expected-value research (phase 2 of the privileged-tracking programme).

The question this package asks is deliberately different from the one answered in
``research.privileged_tracking``: not "is a model better on average" but "can a model
price one specific stat, in one specific scenario, better than a bookmaker's rolling-mean
heuristic, by enough to be +EV after margin".

Modules
-------
common:
    Shared loaders, strictly-prior rolling features, book proxies, betting simulation.
cards:
    Stage 01 -- player card props, the foul -> card mechanism, and team card totals.
corners, corners_features, corners_style:
    Stage 02 -- match and team corner lines, and whether StatsBomb style or the imputed
    defensive state of the opponent beats a rolling corner mean.
pass_counts:
    Stage 03 -- player pass counts: whether the programme's one positive imputed-state result
    (per-pass completion) survives aggregation to a player-match count line, and whether the
    volume channel that dominates such a prop is priceable against a rolling mean.
fouls, fouls_features:
    Stage 04 -- team and match foul totals on the odds table, player fouls committed and
    fouls won on StatsBomb, the referee channel, and the two directions of the foul
    matchup.
gate_sweep:
    Stage 05 -- the generalist-vs-specialist gate run uniformly over all eight markets of
    the phase through the single shared implementation in ``common.run_gate``.
results_index:
    Synthesis -- reads every headline metric back out of the committed result tables into
    ``reports/results_index.parquet``. The phase verdict is ``reports/README.md``.

Every stage obeys the same protocol: a chronological discovery / confirmation split fixed
in code, a generalist-vs-specialist gate run before any modelling, a shrunk rolling-mean
book proxy, and a betting-shaped evaluation whose EV claims are haircut by how much a real
bookmaker beats the same proxy on the one count market where a line is observed (goals).
The haircut has one definition phase-wide -- the margin-matched round trip named by
``haircut.REFERENCE`` -- and the lenient one-directional form is only ever a lower bound.
"""
from __future__ import annotations

__all__ = ["common"]
