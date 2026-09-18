"""Build ``reports/results_index.parquet``: one row per headline metric of the project.

Every row is read from a machine-written ``reports/<stage>_*.parquet`` table (never typed
by hand), so the numbers quoted in ``reports/README.md`` can be traced back to the table
and row they came from. A pick that matches zero or several rows is an error, which is
what protects the index against silent schema drift in the stage reports.

Run from the repo root::

    python -m research.privileged_tracking.synthesis.results_index

Columns of the output: ``sport``, ``stage``, ``experiment``, ``feature_set``, ``metric``,
``value``, ``ci_low``, ``ci_high``, ``ci_type``, ``n``, ``source_report``, ``source_table``,
``row_key``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from research.privileged_tracking.common.io import reports_dir

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

INDEX_COLUMNS = [
    "sport",
    "stage",
    "experiment",
    "feature_set",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "ci_type",
    "n",
    "source_report",
    "source_table",
    "row_key",
]

Where = tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class Pick:
    """One headline metric to extract from one row of one report table.

    Attributes:
        table: parquet basename without extension under ``reports_dir()``.
        where: equality filters ``((column, value), ...)`` that must select exactly one row.
        value: column holding the metric value.
        metric: metric name written to the index (e.g. ``log_loss``, ``delta_log_loss``).
        sport, stage, experiment, feature_set: descriptive keys written to the index.
        ci_low, ci_high: columns holding the 95% interval, or ``None``.
        ci_type: how the interval was computed (``game-clustered``, ``per-sample``, ...).
        n: column holding the sample size, or ``None``.
        source_report: markdown report the number is printed in.
    """

    table: str
    where: Where
    value: str
    metric: str
    sport: str
    stage: str
    experiment: str
    feature_set: str
    ci_low: str | None = None
    ci_high: str | None = None
    ci_type: str = ""
    n: str | None = "n"
    source_report: str = ""


@dataclass
class PickError:
    """A pick that did not resolve to exactly one row."""

    pick: Pick
    n_matched: int
    missing_columns: tuple[str, ...] = field(default_factory=tuple)


def match_rows(df: pd.DataFrame, where: Where) -> pd.DataFrame:
    """Rows of ``df`` satisfying every equality filter in ``where``.

    Args:
        df: any table with the filter columns; shape ``[n_rows, n_cols]``.
        where: ``((column, value), ...)``; a NaN value matches NaN cells.

    Returns:
        The matching sub-frame (possibly empty). Raises ``KeyError`` for an unknown column.
    """
    mask = np.ones(len(df), dtype=bool)
    for col, val in where:
        if col not in df.columns:
            raise KeyError(col)
        s = df[col]
        if isinstance(val, float) and np.isnan(val):
            mask &= s.isna().to_numpy()
        else:
            mask &= (s == val).to_numpy()
    return df.loc[mask]


def _cell(row: pd.Series, col: str | None) -> float:
    if col is None or col not in row.index:
        return float("nan")
    v = row[col]
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def resolve_pick(tables: Mapping[str, pd.DataFrame], pick: Pick) -> dict | PickError:
    """Turn one pick into an index row, or a ``PickError`` when it does not resolve.

    Args:
        tables: ``{basename: DataFrame}`` of the report tables already loaded.
        pick: the pick to resolve.
    """
    if pick.table not in tables:
        return PickError(pick, 0, (pick.table,))
    df = tables[pick.table]
    needed = [c for c, _ in pick.where] + [pick.value]
    needed += [c for c in (pick.ci_low, pick.ci_high, pick.n) if c is not None]
    missing = tuple(c for c in needed if c not in df.columns)
    if missing:
        return PickError(pick, 0, missing)
    rows = match_rows(df, pick.where)
    if len(rows) != 1:
        return PickError(pick, len(rows))
    row = rows.iloc[0]
    value = _cell(row, pick.value)
    if np.isnan(value):
        return PickError(pick, 1, (pick.value,))
    n = _cell(row, pick.n)
    return {
        "sport": pick.sport,
        "stage": pick.stage,
        "experiment": pick.experiment,
        "feature_set": pick.feature_set,
        "metric": pick.metric,
        "value": value,
        "ci_low": _cell(row, pick.ci_low),
        "ci_high": _cell(row, pick.ci_high),
        "ci_type": pick.ci_type if pick.ci_low else "",
        "n": int(n) if not np.isnan(n) else None,
        "source_report": pick.source_report,
        "source_table": pick.table + ".parquet",
        "row_key": "; ".join(f"{c}={v}" for c, v in pick.where),
    }


def build_index(
    tables: Mapping[str, pd.DataFrame], picks: Iterable[Pick]
) -> tuple[pd.DataFrame, list[PickError]]:
    """Resolve every pick; return the index frame and the list of unresolved picks.

    Returns:
        ``(index, errors)`` where ``index`` has the ``INDEX_COLUMNS`` and one row per resolved pick.
    """
    rows: list[dict] = []
    errors: list[PickError] = []
    for p in picks:
        r = resolve_pick(tables, p)
        if isinstance(r, PickError):
            errors.append(r)
        else:
            rows.append(r)
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    index["n"] = index["n"].astype("Int64")
    return index, errors


def load_tables(names: Iterable[str], root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the named report tables (``<name>.parquet``) that exist under ``root``."""
    root = root or reports_dir()
    out: dict[str, pd.DataFrame] = {}
    for name in sorted(set(names)):
        p = root / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


# ---------------------------------------------------------------------------------------
# Pick catalogue. Grouped by stage; every entry names the markdown report it appears in.
# ---------------------------------------------------------------------------------------


def _w(**kw: object) -> Where:
    return tuple(kw.items())


def nfl_picks() -> list[Pick]:
    picks: list[Pick] = []
    # NFL 01 build: agreement of the tracking-derived definitions with the NGS charting.
    rep = "nfl_01_build.md"
    for a, b, col, metric in [
        (
            "box_count_tuned (weeks 4-6 held out)",
            "plays.defendersInTheBox",
            "exact",
            "exact_agreement",
        ),
        (
            "n_pass_rushers_tuned (weeks 4-6 held out)",
            "plays.numberOfPassRushers",
            "exact",
            "exact_agreement",
        ),
        ("time_to_throw (derived, s)", "participation.time_to_throw", "corr", "pearson_r"),
        (
            "target_depth (derived, target == pbp receiver)",
            "participation.ngs_air_yards",
            "exact",
            "exact_agreement",
        ),
        ("derived_RB", "plays.personnel_offense RB", "exact", "exact_agreement"),
        ("derived_DL", "plays.personnel_defense DL", "exact", "exact_agreement"),
    ]:
        picks.append(
            Pick(
                "nfl_01_agreement",
                _w(a=a, b=b),
                col,
                metric,
                "nfl",
                "01_build",
                f"{a} vs {b}",
                "tracking definition",
                source_report=rep,
            )
        )
    # NFL 02 participation.
    rep = "nfl_02_participation.md"
    for model in [
        "S0",
        "S1",
        "S2",
        "S2+play_type",
        "baseline: majority (train freq)",
        "baseline: team x down-distance prior-games",
        "S2 (test-season labels blanked: no in-season tendencies)",
    ]:
        for col in ["log_loss", "accuracy"]:
            picks.append(
                Pick(
                    "nfl_02a_offense_metrics",
                    _w(model=model, split="test"),
                    col,
                    col,
                    "nfl",
                    "02_participation",
                    "offense personnel grouping, test 2022",
                    model,
                    source_report=rep,
                )
            )
    for frm, to in [
        ("S1", "S2"),
        ("S2", "S2+play_type"),
        ("baseline: team x down-distance prior-games", "S2"),
    ]:
        picks.append(
            Pick(
                "nfl_02a_offense_deltas",
                _w(**{"from": frm, "to": to}),
                "delta_loss",
                "delta_log_loss",
                "nfl",
                "02_participation",
                "offense personnel grouping, test 2022",
                f"{frm} -> {to}",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    for target in [
        "defense_personnel",
        "defenders_in_box",
        "number_of_pass_rushers",
        "man_zone",
        "coverage_type",
    ]:
        for to in ["S2+imp_pers", "S2+true_pers"]:
            picks.append(
                Pick(
                    "nfl_02b_defense_deltas",
                    _w(target=target, **{"from": "S2", "to": to}),
                    "delta_loss",
                    "delta_loss",
                    "nfl",
                    "02_participation",
                    f"{target} given the offense grouping, test 2022",
                    f"S2 -> {to}",
                    "ci_low_game",
                    "ci_high_game",
                    "game-clustered",
                    source_report=rep,
                )
            )
    for target, col in [
        ("defense_personnel", "log_loss"),
        ("defense_personnel", "accuracy"),
        ("defenders_in_box", "r2"),
        ("number_of_pass_rushers", "r2"),
        ("man_zone", "log_loss"),
        ("coverage_type", "accuracy"),
    ]:
        for model in ["S2", "S2+imp_pers", "S2+true_pers"]:
            picks.append(
                Pick(
                    "nfl_02b_defense_metrics",
                    _w(target=target, model=model, split="test"),
                    col,
                    col,
                    "nfl",
                    "02_participation",
                    f"{target}, test 2022",
                    model,
                    source_report=rep,
                )
            )
    picks.append(
        Pick(
            "nfl_02b_defense_metrics",
            _w(target="defenders_in_box", model="baseline: team prior-games", split="test"),
            "r2",
            "r2",
            "nfl",
            "02_participation",
            "defenders_in_box, test 2022",
            "baseline: team prior-games",
            source_report=rep,
        )
    )
    for decoder in [
        "model | true grouping",
        "model | imputed grouping",
        "baseline: prior usage rank | true grouping",
        "baseline: prior usage rank | imputed grouping",
    ]:
        for side in ["offense", "defense"]:
            for col in ["per_slot_acc", "exact_lineup_acc"]:
                picks.append(
                    Pick(
                        "nfl_02c_players_summary",
                        _w(decoder=decoder, side=side),
                        col,
                        col,
                        "nfl",
                        "02_participation",
                        f"player-level lineup, {side}, test 2022",
                        decoder,
                        n="plays",
                        source_report=rep,
                    )
                )
    for sub in [
        "QB",
        "OL",
        "WR1-2",
        "CB1-2",
        "RB1",
        "TE1",
        "S",
        "LB",
        "CB3+ (nickel/dime)",
        "WR3+",
        "TE2+",
        "RB2+",
        "DL",
    ]:
        picks.append(
            Pick(
                "nfl_02c_players_per_subgroup",
                _w(decoder="model | true grouping", subgroup=sub),
                "f1",
                "f1",
                "nfl",
                "02_participation",
                f"player-level on-field F1, {sub}, test 2022",
                "model | true grouping",
                n="n_candidates",
                source_report=rep,
            )
        )
    # NFL 03 imputation.
    rep = "nfl_03_imputation.md"
    targets_03 = [
        "shotgun_derived",
        "qb_depth",
        "def_y_std",
        "box_count_tuned",
        "n_deep_safeties",
        "box_count_run",
        "mof_open",
        "cb_cushion",
        "n_wide_right",
        "n_wide_left",
        "n_backfield",
        "n_dl",
        "motion_derived",
        "n_pass_rushers_derived",
        "min_def_dist_carrier_handoff",
        "n_def_within_r_carrier_first_contact",
        "time_to_throw",
        "min_def_dist_qb_throw",
        "separation_at_arrival",
        "target_depth",
        "pressure_derived",
        "n_def_within_r_target",
        "yards_to_first_contact",
    ]
    for t in targets_03:
        for fs in ["F0n", "F0", "F0P", "F1", "F2"]:
            picks.append(
                Pick(
                    "nfl_03_ranking",
                    _w(target=t),
                    f"{fs}_skill",
                    "skill_r2_or_bss",
                    "nfl",
                    "03_imputation",
                    f"{t}, 5-fold by game OOF",
                    fs,
                    n="n_kfold",
                    source_report=rep,
                )
            )
        picks.append(
            Pick(
                "nfl_03_ranking",
                _w(target=t),
                "base_outcome_skill",
                "skill_r2_or_bss",
                "nfl",
                "03_imputation",
                f"{t}, 5-fold by game OOF",
                "base_outcome",
                n="n_kfold",
                source_report=rep,
            )
        )
        picks.append(
            Pick(
                "nfl_03_ranking",
                _w(target=t),
                "F0_skill_forward",
                "skill_r2_or_bss",
                "nfl",
                "03_imputation",
                f"{t}, forward split weeks 1-4 -> 5-6",
                "F0",
                n="n_forward",
                source_report=rep,
            )
        )
    for t in [
        "box_count_tuned",
        "box_count_run",
        "time_to_throw",
        "pressure_derived",
        "target_depth",
        "n_pass_rushers_derived",
    ]:
        picks.append(
            Pick(
                "nfl_03_ranking",
                _w(target=t),
                "oracle_skill",
                "skill_r2_or_bss",
                "nfl",
                "03_imputation",
                f"{t}, official NGS field used as the prediction",
                "oracle",
                n="n_kfold",
                source_report=rep,
            )
        )
    for t in [
        "time_to_throw",
        "pressure_derived",
        "min_def_dist_qb_throw",
        "n_pass_rushers_derived",
        "separation_at_arrival",
        "n_def_within_r_target",
        "target_depth",
        "box_count_run",
        "min_def_dist_carrier_handoff",
        "yards_to_first_contact",
        "n_def_within_r_carrier_first_contact",
    ]:
        picks.append(
            Pick(
                "nfl_03_outcome_decomposition",
                _w(target=t),
                "F1_vs_outcome",
                "skill_vs_base_outcome",
                "nfl",
                "03_imputation",
                f"{t}, after-the-fact F1 beyond the per-outcome-class mean",
                "F1",
                source_report=rep,
            )
        )
        picks.append(
            Pick(
                "nfl_03_outcome_decomposition",
                _w(target=t),
                "F1_skill_in_largest_class",
                "skill_within_largest_outcome_class",
                "nfl",
                "03_imputation",
                f"{t}, F1 inside the largest outcome class",
                "F1",
                n="largest_class_n",
                source_report=rep,
            )
        )
    for t, fs, season, col in [
        ("box_count_tuned", "F1", 2018, "r2"),
        ("box_count_tuned", "F1", 2017, "r2"),
        ("box_count_tuned", "F0P", 2018, "r2"),
        ("box_count_tuned", "F0", 2018, "r2"),
        ("box_count_run", "F0P", 2018, "r2"),
        ("time_to_throw", "F1", 2018, "r2"),
        ("time_to_throw", "F1", 2017, "r2"),
        ("pressure_derived", "F1", 2018, "auc"),
        ("pressure_derived", "F1", 2017, "auc"),
        ("target_depth", "F1", 2018, "r2"),
        ("n_pass_rushers_derived", "F1", 2018, "r2"),
    ]:
        picks.append(
            Pick(
                "nfl_03_external_2018",
                _w(target=t, feature_set=fs, season=season),
                col,
                col,
                "nfl",
                "03_imputation",
                f"{t} vs the official NGS field, season {season}",
                fs,
                source_report=rep,
            )
        )
    for t in ["box_count_tuned", "n_deep_safeties", "mof_open"]:
        for fs in ["F0", "F0P", "F1"]:
            picks.append(
                Pick(
                    "nfl_03_ood_highlights_summary",
                    _w(target=t, model=fs),
                    "skill_highlights",
                    "skill_r2_or_bss",
                    "nfl",
                    "03_ood_highlights",
                    f"{t}, 2018-19 NGS highlight plays (OOD)",
                    fs,
                    "ci_low",
                    "ci_high",
                    "play-level bootstrap",
                    n="n_highlights",
                    source_report="nfl_03_ood_highlights.md",
                )
            )
            picks.append(
                Pick(
                    "nfl_03_ood_highlights_summary",
                    _w(target=t, model=fs),
                    "skill_2017_oof_all",
                    "skill_r2_or_bss",
                    "nfl",
                    "03_ood_highlights",
                    f"{t}, 2017 OOF reference for the OOD check",
                    fs,
                    n=None,
                    source_report="nfl_03_ood_highlights.md",
                )
            )
    # NFL 04 payoff.
    rep = "nfl_04_payoff.md"
    tasks = {
        "pass_completion": ("log_loss", "log_loss"),
        "run_success": ("log_loss", "log_loss"),
        "pass_yards": ("r2", "se"),
        "pass_epa": ("r2", "se"),
        "run_yards": ("r2", "se"),
    }
    sets_a = [
        "PBP",
        "PBP+PERS",
        "PBP+IMP",
        "PBP+IMP_F0",
        "PBP+ORACLE_PRESNAP",
        "PBP+ORACLE_WITHIN",
        "PBP+ORACLE",
        "PBP+NGS",
        "PBP+IMP+NGS",
        "PBP+IMP_F1(leaky)",
        "base_global",
    ]
    for task, (mcol, loss) in tasks.items():
        for m in sets_a:
            picks.append(
                Pick(
                    "nfl_04a_metrics",
                    _w(task=task, model=m),
                    mcol,
                    mcol,
                    "nfl",
                    "04_payoff",
                    f"{task}, tracked 2017 games, 5-fold by game",
                    m,
                    source_report=rep,
                )
            )
        for to in [
            "PBP+PERS",
            "PBP+IMP",
            "PBP+IMP_F0",
            "PBP+ORACLE_PRESNAP",
            "PBP+ORACLE_WITHIN",
            "PBP+ORACLE",
            "PBP+NGS",
        ]:
            picks.append(
                Pick(
                    "nfl_04a_deltas",
                    _w(task=task, **{"from": "PBP", "to": to}, loss=loss),
                    "delta",
                    f"delta_{loss}",
                    "nfl",
                    "04_payoff",
                    f"{task}, tracked 2017 games, 5-fold by game",
                    f"PBP -> {to}",
                    "ci_low_game",
                    "ci_high_game",
                    "game-clustered",
                    source_report=rep,
                )
            )
        picks.append(
            Pick(
                "nfl_04a_deltas",
                _w(task=task, **{"from": "PBP+NGS", "to": "PBP+IMP+NGS"}, loss=loss),
                "delta",
                f"delta_{loss}",
                "nfl",
                "04_payoff",
                f"{task}, tracked 2017 games, 5-fold by game",
                "PBP+NGS -> PBP+IMP+NGS",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    for m in ["nflfastR cp", "PBP [cp rows]", "PBP+NGS [cp rows]"]:
        picks.append(
            Pick(
                "nfl_04a_metrics",
                _w(task="pass_completion", model=m),
                "log_loss",
                "log_loss",
                "nfl",
                "04_payoff",
                "pass_completion, tracked 2017 games, cp-present plays",
                m,
                source_report=rep,
            )
        )
    for to in ["PBP", "PBP+NGS"]:
        picks.append(
            Pick(
                "nfl_04a_deltas",
                _w(task="pass_completion", **{"from": "nflfastR cp", "to": to}),
                "delta",
                "delta_log_loss",
                "nfl",
                "04_payoff",
                "pass_completion, tracked 2017 games, cp-present plays",
                f"nflfastR cp -> {to}",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    for task, (mcol, loss) in tasks.items():
        for m in [
            "PBP",
            "PBP+PERS",
            "PBP+IMP",
            "PBP+IMP_F0",
            "PBP+NGS",
            "PBP+IMP+NGS",
            "base_global",
        ]:
            picks.append(
                Pick(
                    "nfl_04b_metrics",
                    _w(task=task, model=m),
                    mcol,
                    mcol,
                    "nfl",
                    "04_payoff",
                    f"{task}, train 2018-2021, test 2022",
                    m,
                    source_report=rep,
                )
            )
        for to in ["PBP+PERS", "PBP+IMP", "PBP+IMP_F0", "PBP+NGS"]:
            picks.append(
                Pick(
                    "nfl_04b_deltas",
                    _w(task=task, **{"from": "PBP", "to": to}, loss=loss),
                    "delta",
                    f"delta_{loss}",
                    "nfl",
                    "04_payoff",
                    f"{task}, train 2018-2021, test 2022",
                    f"PBP -> {to}",
                    "ci_low_game",
                    "ci_high_game",
                    "game-clustered",
                    source_report=rep,
                )
            )
        picks.append(
            Pick(
                "nfl_04b_deltas",
                _w(task=task, **{"from": "PBP+NGS", "to": "PBP+IMP+NGS"}, loss=loss),
                "delta",
                f"delta_{loss}",
                "nfl",
                "04_payoff",
                f"{task}, train 2018-2021, test 2022",
                "PBP+NGS -> PBP+IMP+NGS",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    for m in ["nflfastR cp", "PBP [cp rows]", "PBP+NGS [cp rows]"]:
        picks.append(
            Pick(
                "nfl_04b_metrics",
                _w(task="pass_completion", model=m),
                "log_loss",
                "log_loss",
                "nfl",
                "04_payoff",
                "pass_completion, test 2022, cp-present plays",
                m,
                source_report=rep,
            )
        )
    for to in ["PBP", "PBP+NGS"]:
        picks.append(
            Pick(
                "nfl_04b_deltas",
                _w(task="pass_completion", **{"from": "nflfastR cp", "to": to}),
                "delta",
                "delta_log_loss",
                "nfl",
                "04_payoff",
                "pass_completion, test 2022, cp-present plays",
                f"nflfastR cp -> {to}",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    for student in ["F1T", "F0T"]:
        for t in [
            "time_to_throw",
            "pressure_derived",
            "min_def_dist_qb_throw",
            "n_pass_rushers_derived",
            "separation_at_arrival",
            "n_def_within_r_target",
            "target_depth",
            "box_count_run",
            "min_def_dist_carrier_handoff",
            "yards_to_first_contact",
            "n_def_within_r_carrier_first_contact",
        ]:
            picks.append(
                Pick(
                    "nfl_04a_students",
                    _w(target=t, student=student),
                    "skill",
                    "skill_r2_or_bss",
                    "nfl",
                    "04_payoff",
                    f"{t}, at-release student, 5-fold OOF",
                    student,
                    source_report=rep,
                )
            )
    for pm in [
        "naive (air yards, cp, share, n)",
        "after-the-fact aggregates only",
        "naive + after-the-fact",
        "naive + imputed F1",
        "naive + after-the-fact + imputed F0T",
        "naive + after-the-fact + imputed F1T",
        "naive + after-the-fact + imputed F1",
    ]:
        picks.append(
            Pick(
                "nfl_04c_receiver_week_regression",
                _w(ngs_field="avg_separation", proxy_model=pm),
                "test_r2",
                "r2",
                "nfl",
                "04_payoff",
                "receiver-week NGS avg_separation, OLS fit 2018-2021, test 2022",
                pm,
                n="n_test",
                source_report=rep,
            )
        )
    for pm in [
        "naive + after-the-fact + imputed F0T",
        "naive + after-the-fact + imputed F1T",
        "naive + after-the-fact + imputed F1",
    ]:
        picks.append(
            Pick(
                "nfl_04c_receiver_week_regression",
                _w(ngs_field="avg_separation", proxy_model=pm),
                "delta_r2_vs_baseline",
                "delta_r2_vs_naive_plus_after_the_fact",
                "nfl",
                "04_payoff",
                "receiver-week NGS avg_separation, test 2022",
                pm,
                "ci_low",
                "ci_high",
                "per-receiver-week bootstrap",
                n="n_test",
                source_report=rep,
            )
        )
    picks.append(
        Pick(
            "nfl_04c_receiver_week_regression",
            _w(ngs_field="avg_cushion", proxy_model="naive + after-the-fact + imputed F0T"),
            "test_r2",
            "r2",
            "nfl",
            "04_payoff",
            "receiver-week NGS avg_cushion, test 2022",
            "naive + after-the-fact + imputed F0T",
            n="n_test",
            source_report=rep,
        )
    )
    for to in ["PBP+IMP_PERS", "PBP+TRUE_PERS", "PBP+PERS", "PBP+IMP+IMP_PERS"]:
        picks.append(
            Pick(
                "nfl_04d_deltas",
                _w(task="pass_completion", **{"from": "PBP", "to": to}),
                "delta",
                "delta_log_loss",
                "nfl",
                "04_payoff",
                "pass_completion, participation payoff, test 2022",
                f"PBP -> {to}",
                "ci_low_game",
                "ci_high_game",
                "game-clustered",
                source_report=rep,
            )
        )
    picks.append(
        Pick(
            "nfl_04d_deltas",
            _w(task="pass_completion", **{"from": "PBP+IMP_PERS", "to": "PBP+TRUE_PERS"}),
            "delta",
            "delta_log_loss",
            "nfl",
            "04_payoff",
            "pass_completion, participation payoff, test 2022",
            "PBP+IMP_PERS -> PBP+TRUE_PERS",
            "ci_low_game",
            "ci_high_game",
            "game-clustered",
            source_report=rep,
        )
    )
    # NFL 06: participation -> alignment chain (fix pass)
    rep = "nfl_06_participation_chain.md"
    presnap = [
        "n_deep_safeties",
        "mof_open",
        "box_count_tuned",
        "n_dl",
        "def_y_std",
        "cb_cushion",
        "n_wide_left",
        "n_wide_right",
        "n_backfield",
        "qb_depth",
        "shotgun_derived",
        "motion_derived",
    ]
    for t in presnap:
        for fset in ["F0", "F0I", "F0P", "F0PI"]:
            picks.append(
                Pick(
                    "nfl_06_chain_metrics",
                    _w(target=t, split="kfold", feature_set=fset),
                    "skill",
                    "skill_r2_or_bss",
                    "nfl",
                    "06_chain",
                    f"{t}, 5-fold by game, tracked 2017 games",
                    fset,
                    source_report=rep,
                )
            )
        for a, b in [("F0", "F0I"), ("F0I", "F0P"), ("F0P", "F0PI")]:
            picks.append(
                Pick(
                    "nfl_06_chain_deltas",
                    _w(target=t, split="kfold", a=a, b=b),
                    "delta_a_minus_b",
                    "delta_loss",
                    "nfl",
                    "06_chain",
                    f"{t}, 5-fold by game, tracked 2017 games",
                    f"{a} -> {b}",
                    "ci_low_game",
                    "ci_high_game",
                    "game-clustered",
                    source_report=rep,
                )
            )
    picks.append(
        Pick(
            "nfl_06_chain_coverage",
            _w(),
            "argmax_accuracy",
            "accuracy",
            "nfl",
            "06_chain",
            "imputed offense grouping (argmax of p_off_*) vs the NFL 02 label, tracked plays",
            "p_off argmax",
            n="n_plays",
            source_report=rep,
        )
    )
    # NFL 07: forward receiving-props test (fix pass)
    rep = "nfl_07_props_forward.md"
    for target in ["next_rec_yards", "next_n_targets"]:
        for pop in ["all", "regular"]:
            exp = f"{target}, population {pop}, rolling test 2020-2022"
            for model in [
                "BASE_GLOBAL",
                "BASE_STD",
                "BASE_L3",
                "HIST",
                "HIST+IMP_F0T",
                "HIST+IMP_F1T",
                "HIST+IMP_F1",
                "HIST+IMP_ALL",
                "HIST+NGS",
            ]:
                picks.append(
                    Pick(
                        "nfl_07_props_metrics",
                        _w(target=target, population=pop, model=model, scope="pooled"),
                        "r2",
                        "r2",
                        "nfl",
                        "07_props",
                        exp,
                        model,
                        source_report=rep,
                    )
                )
                if model != "HIST":
                    picks.append(
                        Pick(
                            "nfl_07_props_deltas",
                            _w(target=target, population=pop, model=model),
                            "delta_se",
                            "delta_squared_error",
                            "nfl",
                            "07_props",
                            exp,
                            f"HIST -> {model}",
                            "ci_low_week",
                            "ci_high_week",
                            "week-clustered",
                            source_report=rep,
                        )
                    )
    return picks


def soccer_picks() -> list[Pick]:
    picks: list[Pick] = []
    rep = "soccer_01_build.md"
    for col, metric in [("exact_agreement", "exact_agreement"), ("pearson_r", "pearson_r")]:
        picks.append(
            Pick(
                "soccer_01_cone_agreement",
                _w(subset="all shots with both frames"),
                col,
                metric,
                "soccer",
                "01_build",
                "shot cone count: 360 frame vs shot.freeze_frame",
                "360 frame",
                source_report=rep,
            )
        )
    rep = "soccer_02_imputation.md"
    targets_02 = [
        "block_depth",
        "def_line",
        "deep_block",
        "n_opp_ahead_of_ball",
        "nearest_opp_dist",
        "n_opp_within_10",
        "n_opp_within_5",
        "opp_keeper_dist_to_goal_line",
        "block_width",
        "block_length",
        "nearest_opp_dist_in_cone",
        "counter_on",
        "n_opp_in_lane",
        "n_opp_in_cone",
        "n_opp_within_3_of_end",
    ]
    for t in targets_02:
        for fs in ["base_type", "loc", "E0", "E1", "E2", "E2a"]:
            picks.append(
                Pick(
                    "soccer_02_ranking",
                    _w(target=t),
                    fs,
                    "skill_r2_or_bss",
                    "soccer",
                    "02_imputation",
                    f"{t}, 5-fold by match OOF (417 matches)",
                    fs,
                    source_report=rep,
                )
            )
        for col, exp in [
            ("E2_m2f", "men -> women transfer"),
            ("E2_f2m", "women -> men transfer"),
            ("E2_fwd", "forward transfer to Euro 2024 + Women's Euro 2025"),
        ]:
            picks.append(
                Pick(
                    "soccer_02_ranking",
                    _w(target=t),
                    col,
                    "skill_r2_or_bss",
                    "soccer",
                    "02_imputation",
                    f"{t}, {exp}",
                    "E2",
                    n=None,
                    source_report=rep,
                )
            )
        if t not in ("deep_block", "counter_on"):
            picks.append(
                Pick(
                    "soccer_02_ranking",
                    _w(target=t),
                    "mae_E2",
                    "mae",
                    "soccer",
                    "02_imputation",
                    f"{t}, 5-fold by match OOF (417 matches)",
                    "E2",
                    source_report=rep,
                )
            )
        for frm, to in [("loc", "E0"), ("E1", "E2"), ("E2", "E2a")]:
            picks.append(
                Pick(
                    "soccer_02_deltas",
                    _w(target=t, **{"from": frm, "to": to}),
                    "delta_loss",
                    "delta_loss",
                    "soccer",
                    "02_imputation",
                    f"{t}, 5-fold by match OOF (417 matches)",
                    f"{frm} -> {to}",
                    "ci_low",
                    "ci_high",
                    "match-clustered",
                    source_report=rep,
                )
            )
    for t in ["block_depth", "n_opp_within_5", "nearest_opp_dist"]:
        for m in ["lgbm_E2_same_rows", "lgbm_E3", "mlp_E2seq"]:
            picks.append(
                Pick(
                    "soccer_02_mlp",
                    _w(target=t, model=m),
                    "r2",
                    "r2",
                    "soccer",
                    "02_imputation",
                    f"{t}, neural vs LightGBM on held-out fold 0",
                    m,
                    source_report=rep,
                )
            )
    rep = "soccer_03_payoff.md"
    xg_variants = [
        "BASE",
        "LOC",
        "EVENT",
        "EVENT+IMP(E0)",
        "EVENT+IMP",
        "EVENT+ORACLE360",
        "EVENT+ORACLESHOT",
        "EVENT+ORACLESHOT_full",
        "STATSBOMB_XG",
    ]
    for v in xg_variants:
        for col in ["log_loss", "auc"]:
            picks.append(
                Pick(
                    "soccer_03_xg_metrics",
                    _w(variant=v),
                    col,
                    col,
                    "soccer",
                    "03_payoff",
                    "xG, 360 matches, 5-fold by match, 3 seeds",
                    v,
                    source_report=rep,
                )
            )
    for v in xg_variants:
        if v == "EVENT":
            continue
        picks.append(
            Pick(
                "soccer_03_xg_deltas",
                _w(variant=v, reference="EVENT"),
                "delta_log_loss",
                "delta_log_loss",
                "soccer",
                "03_payoff",
                "xG, 360 matches, 5-fold by match, 3 seeds",
                f"EVENT -> {v}",
                "ci_low",
                "ci_high",
                "per-shot bootstrap",
                source_report=rep,
            )
        )
    for design in ["after", "noafter"]:
        for v in ["BASE", "EVENT", "EVENT-nodur", "EVENT+IMP", "EVENT+ORACLE"]:
            if design == "noafter" and v == "EVENT-nodur":
                continue
            picks.append(
                Pick(
                    "soccer_03_xpass_metrics",
                    _w(design=design, variant=v),
                    "log_loss",
                    "log_loss",
                    "soccer",
                    "03_payoff",
                    f"xPass ({design}), 360 matches, 5-fold by match",
                    v,
                    source_report=rep,
                )
            )
            if v != "EVENT":
                picks.append(
                    Pick(
                        "soccer_03_xpass_deltas",
                        _w(design=design, variant=v, reference="EVENT"),
                        "delta_log_loss",
                        "delta_log_loss",
                        "soccer",
                        "03_payoff",
                        f"xPass ({design}), 360 matches, 5-fold by match",
                        f"EVENT -> {v}",
                        "ci_low",
                        "ci_high",
                        "per-pass bootstrap",
                        source_report=rep,
                    )
                )
    for v in [
        "EVENT<-EVENT+ORACLE360@1 (distilled)",
        "EVENT<-EVENT+ORACLE360@0.5 (distilled)",
        "EVENT<-EVENT+ORACLESHOT@1 (distilled)",
        "EVENT+IMP<-EVENT+ORACLE360@1 (distilled)",
    ]:
        picks.append(
            Pick(
                "soccer_03_xg_distill_deltas",
                _w(variant=v, reference="EVENT (direct)"),
                "delta_log_loss",
                "delta_log_loss",
                "soccer",
                "03_payoff",
                "xG distillation from an oracle teacher, nested CV",
                f"EVENT (direct) -> {v}",
                "ci_low",
                "ci_high",
                "per-shot bootstrap",
                source_report=rep,
            )
        )
    for share in [0.1, 0.25, 0.5, 1.0]:
        picks.append(
            Pick(
                "soccer_03_xg_curve",
                _w(train_share=share, variant="EVENT"),
                "log_loss",
                "log_loss",
                "soccer",
                "03_payoff",
                f"xG learning curve, {int(share * 100)}% of training matches",
                "EVENT",
                source_report=rep,
            )
        )
        for v in ["EVENT+IMP", "EVENT+ORACLESHOT"]:
            picks.append(
                Pick(
                    "soccer_03_xg_curve",
                    _w(train_share=share, variant=v),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "03_payoff",
                    f"xG learning curve, {int(share * 100)}% of training matches",
                    f"EVENT -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    for direction in ["m2f", "f2m"]:
        for v in [
            "EVENT+IMP",
            "EVENT+ORACLE360",
            "EVENT+ORACLESHOT",
            "STATSBOMB_XG",
            "EVENT (in-domain CV)",
        ]:
            picks.append(
                Pick(
                    "soccer_03_xg_transfer_deltas",
                    _w(direction=direction, variant=v, reference="EVENT"),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "03_payoff",
                    f"xG cross-gender transfer {direction} "
                    "(students and xG trained on the source gender)",
                    f"EVENT -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    for lvl in ["0-8", "8-12", "12-18", "18-25", "25+"]:
        for v in ["EVENT+IMP", "EVENT+ORACLESHOT"]:
            picks.append(
                Pick(
                    "soccer_03_xg_slices",
                    _w(slice="distance_band", level=lvl, variant=v),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "03_payoff",
                    f"xG by shot distance band {lvl} yd",
                    f"EVENT -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    picks.append(
        Pick(
            "soccer_03_xg_slices",
            _w(slice="play_pattern", level="counter", variant="EVENT+IMP"),
            "delta_log_loss",
            "delta_log_loss",
            "soccer",
            "03_payoff",
            "xG on counter-attack shots",
            "EVENT -> EVENT+IMP",
            "ci_low",
            "ci_high",
            "per-shot bootstrap",
            source_report=rep,
        )
    )
    rep = "soccer_04_transfer.md"
    v04 = [
        "BASE",
        "LOC",
        "EVENT",
        "EVENT+IMP(E0)",
        "EVENT+IMP",
        "EVENT+ORACLESHOT",
        "EVENT+ORACLESHOT_full",
        "EVENT (zero-shot)",
        "EVENT+IMP (zero-shot)",
        "EVENT+ORACLESHOT (zero-shot)",
        "STATSBOMB_XG",
    ]
    for v in v04:
        picks.append(
            Pick(
                "soccer_04_xg_metrics",
                _w(population="leagues_2015/16", variant=v),
                "log_loss",
                "log_loss",
                "soccer",
                "04_transfer",
                "xG, 2015/16 PL + La Liga + Serie A + Ligue 1, 5-fold by match",
                v,
                source_report=rep,
            )
        )
        if v != "EVENT":
            # The parquet stores every delta against the within-CV EVENT model; the
            # zero-shot-vs-zero-shot deltas are printed in the markdown only.
            picks.append(
                Pick(
                    "soccer_04_xg_deltas",
                    _w(population="leagues_2015/16", variant=v, reference="EVENT"),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "04_transfer",
                    "xG, 2015/16 PL + La Liga + Serie A + Ligue 1, 5-fold by match",
                    f"EVENT -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    for pop in ["leagues_2015/16", "tournaments_no360"]:
        for v in ["EVENT+IMP (zero-shot)", "EVENT+ORACLESHOT (zero-shot)"]:
            picks.append(
                Pick(
                    "soccer_04_xg_deltas_zeroshot",
                    _w(population=pop, variant=v, reference="EVENT (zero-shot)"),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "04_transfer",
                    f"xG, {pop}, stage-03 models applied zero-shot",
                    f"EVENT (zero-shot) -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    for pop in ["PL 2015/16", "La Liga 2015/16", "Serie A 2015/16", "Ligue 1 2015/16"]:
        for v in ["EVENT+IMP", "EVENT+ORACLESHOT"]:
            picks.append(
                Pick(
                    "soccer_04_xg_deltas",
                    _w(population=pop, variant=v, reference="EVENT"),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "04_transfer",
                    f"xG, {pop}, 5-fold by match, 2 seeds",
                    f"EVENT -> {v}",
                    "ci_low",
                    "ci_high",
                    "per-shot bootstrap",
                    source_report=rep,
                )
            )
    for v in ["EVENT", "EVENT+IMP", "EVENT+ORACLESHOT", "EVENT (zero-shot)", "STATSBOMB_XG"]:
        picks.append(
            Pick(
                "soccer_04_xg_metrics",
                _w(population="tournaments_no360", variant=v),
                "log_loss",
                "log_loss",
                "soccer",
                "04_transfer",
                "xG, WC 2018 + Copa 2024 + AFCON 2023 (no 360)",
                v,
                source_report=rep,
            )
        )
    for v in [
        "EVENT+IMP",
        "EVENT+ORACLESHOT",
        "EVENT (zero-shot)",
        "EVENT+IMP (zero-shot)",
        "STATSBOMB_XG",
    ]:
        picks.append(
            Pick(
                "soccer_04_xg_deltas",
                _w(population="tournaments_no360", variant=v, reference="EVENT"),
                "delta_log_loss",
                "delta_log_loss",
                "soccer",
                "04_transfer",
                "xG, WC 2018 + Copa 2024 + AFCON 2023 (no 360)",
                f"EVENT -> {v}",
                "ci_low",
                "ci_high",
                "per-shot bootstrap",
                source_report=rep,
            )
        )
    for design in ["after", "noafter"]:
        for v in ["BASE", "EVENT", "EVENT-nodur", "EVENT+IMP"]:
            if design == "noafter" and v == "EVENT-nodur":
                continue
            picks.append(
                Pick(
                    "soccer_04_xpass_metrics",
                    _w(design=design, variant=v),
                    "log_loss",
                    "log_loss",
                    "soccer",
                    "04_transfer",
                    f"xPass ({design}), 2015/16 leagues, 5-fold by match",
                    v,
                    source_report=rep,
                )
            )
            if v != "EVENT":
                picks.append(
                    Pick(
                        "soccer_04_xpass_deltas",
                        _w(design=design, variant=v, reference="EVENT"),
                        "delta_log_loss",
                        "delta_log_loss",
                        "soccer",
                        "04_transfer",
                        f"xPass ({design}), 2015/16 leagues, 5-fold by match",
                        f"EVENT -> {v}",
                        "ci_low",
                        "ci_high",
                        "per-pass bootstrap",
                        source_report=rep,
                    )
                )
    for t in [
        "n_opp_in_cone",
        "nearest_opp_dist_in_cone",
        "opp_keeper_dist_to_goal_line",
        "n_opp_within_5",
    ]:
        for grp, obs in [
            ("360 all (OOF)", "shot.freeze_frame"),
            ("leagues_2015/16", "shot.freeze_frame"),
            ("WC 2018", "shot.freeze_frame"),
            ("360 all (OOF)", "360 frame"),
        ]:
            for col in ["r2", "mae", "bias"]:
                picks.append(
                    Pick(
                        "soccer_04_oracle_check",
                        _w(group=grp, target=t, observed=obs),
                        col,
                        col,
                        "soccer",
                        "04_transfer",
                        f"imputed E2 {t} vs {obs}, {grp}",
                        "E2 (frozen)",
                        source_report=rep,
                    )
                )
    for level, imputed, proxy in [
        ("team-season", "imp_block_depth__E2", "def_line_x"),
        ("team-season", "imp_block_depth__E2", "ppda"),
        ("team-season", "imp_block_depth__E0", "def_line_x"),
        ("team-season", "ball_x_mean", "def_line_x"),
        ("team-match", "imp_block_depth__E2", "def_line_x"),
    ]:
        picks.append(
            Pick(
                "soccer_04_team_spearman",
                _w(level=level, imputed=imputed, proxy=proxy),
                "spearman",
                "spearman",
                "soccer",
                "04_transfer",
                f"{level} {imputed} vs {proxy}, 2015/16 leagues",
                imputed,
                source_report=rep,
            )
        )
    rep = "soccer_05_sequence.md"
    kinds_05 = {
        "block_depth": "r2",
        "def_line": "r2",
        "n_opp_ahead_of_ball": "r2",
        "nearest_opp_dist": "r2",
        "n_opp_in_cone": "r2",
        "deep_block": "bss",
        "counter_on": "bss",
    }
    for t, col in kinds_05.items():
        for m in ["seq20", "seq1", "seq0", "lgbm_E2", "base_type"]:
            picks.append(
                Pick(
                    "soccer_05_metrics",
                    _w(target=t, model=m),
                    col,
                    "skill_r2_or_bss",
                    "soccer",
                    "05_sequence",
                    f"{t}, held-out folds 0-2 (251 matches), identical rows",
                    m,
                    source_report=rep,
                )
            )
        for frm, to in [
            ("lgbm_E2", "seq20"),
            ("lgbm_E2", "seq1"),
            ("lgbm_E2", "seq0"),
            ("seq1", "seq20"),
        ]:
            picks.append(
                Pick(
                    "soccer_05_deltas",
                    _w(target=t, **{"from": frm, "to": to}),
                    "delta_loss",
                    "delta_loss",
                    "soccer",
                    "05_sequence",
                    f"{t}, held-out folds 0-2 (251 matches), identical rows",
                    f"{frm} -> {to}",
                    "ci_low_clustered",
                    "ci_high_clustered",
                    "match-clustered",
                    source_report=rep,
                )
            )
    # Soccer 06: payoff of the sequence student (fix pass)
    rep = "soccer_06_seq_payoff.md"
    v06 = [
        "EVENT",
        "EVENT+IMP",
        "EVENT+SEQ7(lgbm)",
        "EVENT+SEQ7(seq20)",
        "EVENT+IMP+SEQ7(seq20)",
        "EVENT+ORACLE360",
        "EVENT+ORACLESHOT",
    ]
    for v in v06:
        picks.append(
            Pick(
                "soccer_06_xg_metrics",
                _w(variant=v),
                "log_loss",
                "log_loss",
                "soccer",
                "06_seq_payoff",
                "xG, stage-02 folds 0-2 (251 matches), 3-fold by match",
                v,
                source_report=rep,
            )
        )
        if v != "EVENT":
            picks.append(
                Pick(
                    "soccer_06_xg_deltas",
                    _w(variant=v, reference="EVENT"),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "06_seq_payoff",
                    "xG, stage-02 folds 0-2 (251 matches), 3-fold by match",
                    f"EVENT -> {v}",
                    "ci_low_clustered",
                    "ci_high_clustered",
                    "match-clustered",
                    source_report=rep,
                )
            )
        for design in ["after", "noafter"]:
            if v == "EVENT+ORACLESHOT":
                continue
            picks.append(
                Pick(
                    "soccer_06_xpass_metrics",
                    _w(design=design, variant=v),
                    "log_loss",
                    "log_loss",
                    "soccer",
                    "06_seq_payoff",
                    f"xPass ({design}), stage-02 folds 0-2, 3-fold by match",
                    v,
                    source_report=rep,
                )
            )
            if v != "EVENT":
                picks.append(
                    Pick(
                        "soccer_06_xpass_deltas",
                        _w(design=design, variant=v, reference="EVENT"),
                        "delta_log_loss",
                        "delta_log_loss",
                        "soccer",
                        "06_seq_payoff",
                        f"xPass ({design}), stage-02 folds 0-2, 3-fold by match",
                        f"EVENT -> {v}",
                        "ci_low_clustered",
                        "ci_high_clustered",
                        "match-clustered",
                        source_report=rep,
                    )
                )
    for frm, to in [
        ("EVENT+SEQ7(lgbm)", "EVENT+SEQ7(seq20)"),
        ("EVENT+IMP", "EVENT+IMP+SEQ7(seq20)"),
    ]:
        picks.append(
            Pick(
                "soccer_06_xg_pairs",
                _w(**{"from": frm, "to": to}),
                "delta_log_loss",
                "delta_log_loss",
                "soccer",
                "06_seq_payoff",
                "xG, stage-02 folds 0-2 (251 matches), 3-fold by match",
                f"{frm} -> {to}",
                "ci_low_clustered",
                "ci_high_clustered",
                "match-clustered",
                source_report=rep,
            )
        )
        for design in ["after", "noafter"]:
            picks.append(
                Pick(
                    "soccer_06_xpass_pairs",
                    _w(design=design, **{"from": frm, "to": to}),
                    "delta_log_loss",
                    "delta_log_loss",
                    "soccer",
                    "06_seq_payoff",
                    f"xPass ({design}), stage-02 folds 0-2, 3-fold by match",
                    f"{frm} -> {to}",
                    "ci_low_clustered",
                    "ci_high_clustered",
                    "match-clustered",
                    source_report=rep,
                )
            )
    return picks


def all_picks() -> list[Pick]:
    """The full pick catalogue (NFL then soccer)."""
    return nfl_picks() + soccer_picks()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", default=None, help="output parquet (default reports_dir()/results_index.parquet)"
    )
    args = parser.parse_args(argv)
    picks = all_picks()
    tables = load_tables(p.table for p in picks)
    index, errors = build_index(tables, picks)
    out = Path(args.out) if args.out else reports_dir() / "results_index.parquet"
    index.to_parquet(out, index=False)
    print(f"wrote {out} with {len(index)} rows from {index['source_table'].nunique()} tables")
    if errors:
        print(f"{len(errors)} picks did not resolve:")
        for e in errors:
            print(
                f"  {e.pick.table} {e.pick.where}: matched {e.n_matched}, "
                f"missing {e.missing_columns}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
