"""Build the phase-level index of headline results.

Every number quoted in ``research/scenario_ev/reports/README.md`` comes from a result
table written by one of the four candidate stages or by the gate sweep. This module
declares, one :class:`MetricSpec` per headline metric, *where* that number lives, reads
it back out of the committed parquet, and writes a single tidy table:

``research/scenario_ev/reports/results_index.parquet``

with columns ``candidate, stage, experiment, model, metric, value, ci_low, ci_high,
ci_type, n, split, source_report, source_table``.

The index is therefore not a transcription: if a stage is re-run and a number moves, the
index moves with it, and a spec that no longer resolves to exactly one row fails loudly
rather than silently reporting a stale value.

Run as::

    cd /home/user/geo-model
    python -m research.scenario_ev.results_index
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.scenario_ev import common as C

# ---------------------------------------------------------------------------
# Spec model
# ---------------------------------------------------------------------------

BOOTSTRAP_CI = "match_clustered_bootstrap_95"
NO_CI = "none"


@dataclass(frozen=True)
class MetricSpec:
    """One headline metric and the table row it is read from.

    Attributes:
        candidate: Market family the metric belongs to (``cards``, ``corners``,
            ``pass_counts``, ``fouls``).
        stage: Two-digit stage number that produced it (``"01"`` .. ``"05"``).
        experiment: Short slug for the experiment (``gate``, ``vs_proxy``, ``betting``,
            ``haircut``, ``imputed_state``, ``mechanism``, ``refit_floor``, ``ceiling``).
        model: Model or contrast the value describes.
        metric: Name of the quantity (units are nats, ROI fractions, or shares).
        source_table: Parquet stem under ``reports/`` holding the row.
        source_report: Markdown report that discusses it.
        split: ``discovery`` (exploratory) or ``confirmation`` (scored once).
        value_col: Column holding the value.
        filters: Column -> value equality filters selecting the row(s). Floats are
            matched with ``np.isclose``; ``None`` matches null.
        ci_lo_col: Column holding the lower interval bound, if any.
        ci_hi_col: Column holding the upper interval bound, if any.
        ci_type: Label describing the interval, or ``"none"``.
        n_col: Column holding the sample size, if any.
        agg: ``"one"`` requires exactly one matching row; ``"spread"`` takes the range
            (max - min) of ``value_col`` over the matching rows, which is how a
            refit-noise floor is expressed; ``"mean"`` takes the mean over the matching
            rows and reports the min and max in the interval columns, which is how a
            multi-seed channel estimate is expressed.
    """

    candidate: str
    stage: str
    experiment: str
    model: str
    metric: str
    source_table: str
    source_report: str
    split: str
    value_col: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    ci_lo_col: str | None = None
    ci_hi_col: str | None = None
    ci_type: str = BOOTSTRAP_CI
    n_col: str | None = "n"
    agg: str = "one"


INDEX_COLUMNS: tuple[str, ...] = (
    "candidate",
    "stage",
    "experiment",
    "model",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "ci_type",
    "n",
    "split",
    "source_report",
    "source_table",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def select_rows(df: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    """Select the rows of ``df`` matching every filter.

    Args:
        df: Table to filter, shape ``[n_rows, n_cols]``.
        filters: Column -> value equality filters. Floats are compared with
            ``np.isclose`` (so ``0.06`` matches a stored ``0.060000000000000005``),
            ``None`` selects null entries, everything else compares with ``==``.

    Returns:
        The matching sub-frame, shape ``[n_matched, n_cols]``.

    Raises:
        KeyError: If a filter names a column the table does not have.
    """
    mask = np.ones(len(df), dtype=bool)
    for col, want in filters.items():
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in table (have {list(df.columns)})")
        series = df[col]
        if want is None:
            mask &= series.isna().to_numpy()
        elif isinstance(want, float):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            mask &= np.isclose(values, want, rtol=0.0, atol=1e-9)
        else:
            mask &= (series == want).to_numpy()
    return df.loc[mask]


def _float_or_nan(row: pd.Series, col: str | None) -> float:
    """Read a float from a row, returning NaN when the column is not requested."""
    if col is None:
        return float("nan")
    return float(row[col])


def resolve_spec(spec: MetricSpec, df: pd.DataFrame) -> dict[str, Any]:
    """Turn one spec plus its source table into one index row.

    Args:
        spec: The metric declaration.
        df: The full source table, shape ``[n_rows, n_cols]``.

    Returns:
        A dict with the keys of :data:`INDEX_COLUMNS`.

    Raises:
        ValueError: If the filters do not select the number of rows the spec's ``agg``
            requires (exactly one for ``"one"``, at least two for ``"spread"`` and
            ``"mean"``).
    """
    sub = select_rows(df, spec.filters)
    if spec.agg in ("spread", "mean"):
        if len(sub) < 2:
            raise ValueError(
                f"{spec.source_table}/{spec.metric}: {spec.agg} needs >= 2 rows, "
                f"got {len(sub)}"
            )
        values = pd.to_numeric(sub[spec.value_col], errors="coerce").to_numpy(dtype=float)
        if spec.agg == "spread":
            value = float(np.nanmax(values) - np.nanmin(values))
            ci_low = ci_high = float("nan")
        else:
            # a seed mean, reported with the seed range in place of an interval, so a
            # channel quoted at its luckiest seed can be read against its own spread
            value = float(np.nanmean(values))
            ci_low, ci_high = float(np.nanmin(values)), float(np.nanmax(values))
        n = float(sub[spec.n_col].iloc[0]) if spec.n_col else float("nan")
    else:
        if len(sub) != 1:
            raise ValueError(
                f"{spec.source_table}/{spec.metric}: filters {dict(spec.filters)} "
                f"selected {len(sub)} rows, expected 1"
            )
        row = sub.iloc[0]
        value = float(row[spec.value_col])
        ci_low = _float_or_nan(row, spec.ci_lo_col)
        ci_high = _float_or_nan(row, spec.ci_hi_col)
        n = _float_or_nan(row, spec.n_col)
    return {
        "candidate": spec.candidate,
        "stage": spec.stage,
        "experiment": spec.experiment,
        "model": spec.model,
        "metric": spec.metric,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_type": spec.ci_type,
        "n": n,
        "split": spec.split,
        "source_report": spec.source_report,
        "source_table": spec.source_table,
    }


GATE_MARKET_CANDIDATE: dict[str, str] = {
    "match_corners": "corners",
    "team_corners": "corners",
    "player_cards": "cards",
    "team_cards": "cards",
    "player_fouls": "fouls",
    "team_fouls": "fouls",
    "player_passes": "pass_counts",
    "player_passes_completed": "pass_counts",
}


def gate_index(headline: pd.DataFrame) -> pd.DataFrame:
    """Expand the gate sweep's headline table into index rows.

    One row per (market, scenario) cell, carrying the confirmation-split paired delta
    ``loss(generalist) - loss(specialist)``: positive means the specialist is better,
    i.e. targeting has room.

    Args:
        headline: ``05_gate_sweep_headline`` as written by ``gate_sweep.py``,
            shape ``[n_cells, 18]``.

    Returns:
        Index rows, shape ``[n_cells, len(INDEX_COLUMNS)]``.

    Raises:
        KeyError: If a market in the table has no candidate mapping.
    """
    rows: list[dict[str, Any]] = []
    for _, r in headline.iterrows():
        market = str(r["market"])
        if market not in GATE_MARKET_CANDIDATE:
            raise KeyError(f"unmapped gate market {market!r}")
        rows.append(
            {
                "candidate": GATE_MARKET_CANDIDATE[market],
                "stage": "05",
                "experiment": "gate",
                "model": f"{market} / {r['scenario']}",
                "metric": "delta_gen_minus_spec_nats",
                "value": float(r["delta_gen_minus_spec"]),
                "ci_low": float(r["ci_lo"]),
                "ci_high": float(r["ci_hi"]),
                "ci_type": BOOTSTRAP_CI,
                "n": float(r["n_confirmation"]),
                "split": "confirmation",
                "source_report": "05_gate_sweep.md",
                "source_table": "05_gate_sweep_headline",
            }
        )
    return pd.DataFrame(rows, columns=list(INDEX_COLUMNS))


# ---------------------------------------------------------------------------
# The headline specs
# ---------------------------------------------------------------------------

CARDS_MD = "01_cards.md"
CORNERS_MD = "02_corners.md"
PASS_MD = "03_pass_counts.md"
FOULS_MD = "04_fouls.md"
HAIRCUT_MD = "06_haircut.md"

SPECS: tuple[MetricSpec, ...] = (
    # ---------------- the anchor: how much a real book beats a rolling mean -------
    MetricSpec(
        candidate="cards", stage="01", experiment="haircut", model="bet365_novig_vs_proxy",
        metric="haircut_nats_goals_2.5", source_table="01_cards_c_goals_gap",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_proxy",
        filters={"where": "confirmation", "model": "bet365_novig"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="haircut", model="event_only_goals_model",
        metric="model_gain_vs_proxy_nats_goals_2.5", source_table="01_cards_c_goals_gap",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_proxy",
        filters={"where": "confirmation", "model": "event_only_model"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="haircut", model="bet365_novig_vs_proxy",
        metric="haircut_nats_goals_2.5", source_table="02_corners_goals_haircut_metrics",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_vs_proxy_cal",
        filters={"split": "confirmation", "model": "bet365_novig"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="haircut", model="bet365_novig_vs_proxy",
        metric="haircut_nats_goals_2.5", source_table="03_pass_goals_gap",
        source_report=PASS_MD, split="confirmation", value_col="delta_vs_proxy_cal",
        filters={"population": "four_leagues_all_seasons", "where": "confirmation",
                 "model": "bet365_novig"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="haircut", model="bet365_novig_vs_proxy",
        metric="haircut_nats_goals_2.5", source_table="04_fouls_d_goals_metrics",
        source_report=FOULS_MD, split="confirmation", value_col="delta_vs_proxy_cal",
        filters={"split": "confirmation", "model": "bet365_novig"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="haircut", model="goals_dress_rehearsal",
        metric="haircut_roi_points", source_table="01_cards_c_haircut",
        source_report=CARDS_MD, split="confirmation", value_col="roi_haircut",
        filters={}, ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="haircut", model="goals_margin_matched_6pct",
        metric="haircut_roi_points", source_table="04_fouls_d_haircut",
        source_report=FOULS_MD, split="confirmation", value_col="haircut_roi_points",
        filters={"split": "confirmation", "hold": 0.06}, ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="haircut", model="goals_book_vs_proxy",
        metric="haircut_roi_points", source_table="03_pass_goals_haircut",
        source_report=PASS_MD, split="confirmation", value_col="roi_haircut",
        filters={}, ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="haircut", model="goals_book_vs_proxy_6pct",
        metric="haircut_roi_points", source_table="02_corners_goals_haircut_roi",
        source_report=CORNERS_MD, split="confirmation", value_col="roi",
        filters={"split": "confirmation", "hold": 0.06, "threshold": 0.02},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    # ---------------- cards: does a model beat the proxy? -------------------------
    MetricSpec(
        candidate="cards", stage="01", experiment="vs_proxy", model="P1_event_only",
        metric="delta_log_loss_vs_proxy_nats", source_table="01_cards_a_metrics",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_P0",
        filters={"target": "y_carded", "where": "confirmation", "model": "P1"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="vs_proxy", model="P2_matchup",
        metric="delta_log_loss_vs_proxy_nats", source_table="01_cards_a_metrics",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_P0",
        filters={"target": "y_carded", "where": "confirmation", "model": "P2"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="vs_proxy", model="C1_event_only_3.5",
        metric="delta_log_loss_vs_proxy_nats", source_table="01_cards_c_metrics",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_proxy",
        filters={"where": "confirmation", "line": 3.5, "model": "C1_event"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="vs_proxy", model="C2_event_plus_market_3.5",
        metric="delta_log_loss_vs_proxy_nats", source_table="01_cards_c_metrics",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_proxy",
        filters={"where": "confirmation", "line": 3.5, "model": "C2_event_plus_market"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="imputed_state", model="P3_minus_P2",
        metric="delta_log_loss_nats", source_table="01_cards_a_p3_subset",
        source_report=CARDS_MD, split="confirmation", value_col="delta_vs_P2",
        filters={"where": "confirmation", "model": "P3", "subset": "state_available"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="refit_floor", model="P1_event_only",
        metric="refit_spread_nats_3_seeds", source_table="01_cards_a_refit",
        source_report=CARDS_MD, split="confirmation", value_col="confirmation_log_loss",
        filters={"model": "P1"}, ci_type=NO_CI, n_col=None, agg="spread",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="refit_floor", model="P2_matchup",
        metric="refit_spread_nats_3_seeds", source_table="01_cards_a_refit",
        source_report=CARDS_MD, split="confirmation", value_col="confirmation_log_loss",
        filters={"model": "P2"}, ci_type=NO_CI, n_col=None, agg="spread",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="refit_floor",
        model="C2_event_plus_market_3.5", metric="refit_spread_nats_3_seeds",
        source_table="01_cards_c_refit", source_report=CARDS_MD, split="confirmation",
        value_col="confirmation_log_loss", filters={"model": "C2_event_plus_market"},
        ci_type=NO_CI, n_col=None, agg="spread",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="mechanism", model="foul_level_lgbm",
        metric="log_loss", source_table="01_cards_b_metrics", source_report=CARDS_MD,
        split="confirmation", value_col="log_loss",
        filters={"where": "confirmation", "model": "foul_lgb"}, ci_type=NO_CI,
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="mechanism", model="base_rate",
        metric="log_loss", source_table="01_cards_b_metrics", source_report=CARDS_MD,
        split="confirmation", value_col="log_loss",
        filters={"where": "confirmation", "model": "base_rate"}, ci_type=NO_CI,
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="betting", model="P2_vs_proxy_6pct",
        metric="roi", source_table="01_cards_a_bets", source_report=CARDS_MD,
        split="confirmation", value_col="roi",
        filters={"book": "P0_proxy", "model": "P2", "hold": 0.06, "threshold": 0.4,
                 "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="betting", model="C2_vs_proxy_6pct",
        metric="roi", source_table="01_cards_c_bets", source_report=CARDS_MD,
        split="confirmation", value_col="roi",
        filters={"book": "C0_proxy", "model": "C2_event_plus_market", "hold": 0.06,
                 "threshold": 0.4, "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="betting", model="C2_vs_proxy_6pct",
        metric="roi_after_haircut", source_table="01_cards_c_bets", source_report=CARDS_MD,
        split="confirmation", value_col="roi_after_haircut",
        filters={"book": "C0_proxy", "model": "C2_event_plus_market", "hold": 0.06,
                 "threshold": 0.4, "where": "confirmation"},
        ci_lo_col="roi_lo_after_haircut", ci_hi_col=None, n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="betting",
        model="C2_vs_book_0.0073_nats_sharper", metric="roi",
        source_table="01_cards_c_book_ladder", source_report=CARDS_MD,
        split="confirmation", value_col="roi",
        filters={"lam": 0.5, "hold": 0.06, "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="cards", stage="01", experiment="betting",
        model="P2_vs_book_0.0034_nats_sharper", metric="roi",
        source_table="01_cards_a_book_ladder", source_report=CARDS_MD,
        split="confirmation", value_col="roi",
        filters={"lam": 0.25, "hold": 0.06, "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    # ---------------- corners ------------------------------------------------------
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_offset_match_total",
        metric="delta_poisson_deviance_vs_proxy_nats", source_table="02_corners_models",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_model",
        filters={"target": "total_count", "model": "C1_offset", "split": "confirmation",
                 "seed": 0},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_offset_match_9.5",
        metric="delta_log_loss_vs_proxy_nats", source_table="02_corners_models",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_model",
        filters={"target": "over_line", "line": 9.5, "model": "C1_offset",
                 "split": "confirmation", "seed": 0},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_full_team_count",
        metric="delta_poisson_deviance_vs_proxy_nats", source_table="02_corners_team",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_c1",
        filters={"target": "team_count", "model": "C1_full", "split": "confirmation",
                 "side": "both"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_market_only_team_count",
        metric="delta_poisson_deviance_vs_proxy_nats", source_table="02_corners_team",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_c1",
        filters={"target": "team_count", "model": "C1_market_only", "split": "confirmation",
                 "side": "both"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_events_only_team_count",
        metric="delta_poisson_deviance_vs_proxy_nats", source_table="02_corners_team",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_c1",
        filters={"target": "team_count", "model": "C1_events_only", "split": "confirmation",
                 "side": "both"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="vs_proxy", model="C1_full_team_5.5",
        metric="delta_log_loss_vs_proxy_nats", source_table="02_corners_team",
        source_report=CORNERS_MD, split="confirmation", value_col="delta_c0_minus_c1",
        filters={"target": "team_over_line", "line": 5.5, "model": "C1_full",
                 "split": "confirmation", "side": "both"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="imputed_state",
        model="S_imputed_only_vs_proxy", metric="delta_poisson_deviance_vs_proxy_nats",
        source_table="02_corners_style", source_report=CORNERS_MD, split="cv_all",
        value_col="delta_vs_B0",
        filters={"feature_set": "S_imputed_only", "scheme": "cv_all"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi", n_col="n_rows",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="imputed_state",
        model="B3_plus_imputed_state_vs_proxy", metric="delta_poisson_deviance_vs_proxy_nats",
        source_table="02_corners_style", source_report=CORNERS_MD, split="confirmation",
        value_col="delta_vs_B0",
        filters={"feature_set": "B3_plus_imputed_state", "scheme": "forward_confirmation"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi", n_col="n_rows",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="imputed_state",
        model="B2_plus_style_vs_proxy", metric="delta_poisson_deviance_vs_proxy_nats",
        source_table="02_corners_style", source_report=CORNERS_MD, split="confirmation",
        value_col="delta_vs_B0",
        filters={"feature_set": "B2_plus_style", "scheme": "forward_confirmation"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi", n_col="n_rows",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="refit_floor", model="C1_offset_match_9.5",
        metric="refit_spread_nats_3_seeds", source_table="02_corners_refit_spread",
        source_report=CORNERS_MD, split="confirmation", value_col="spread",
        filters={"target": "over_line", "line": 9.5, "model": "C1_offset",
                 "split": "confirmation"},
        ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="betting", model="count_nb_match_total_6pct",
        metric="roi", source_table="02_corners_bets", source_report=CORNERS_MD,
        split="confirmation", value_col="roi",
        filters={"market": "match_total", "scenario": "all", "source": "count_nb",
                 "hold": 0.06, "threshold": 0.02, "split": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="betting", model="count_nb_team_total_6pct",
        metric="roi", source_table="02_corners_bets", source_report=CORNERS_MD,
        split="confirmation", value_col="roi",
        filters={"market": "team_total", "scenario": "all", "source": "count_nb",
                 "hold": 0.06, "threshold": 0.02, "split": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    # ---------------- pass counts ---------------------------------------------------
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="EVENT+IMP_programme_cache", metric="delta_per_pass_nats_vs_event",
        source_table="03_pass_a_per_pass", source_report=PASS_MD, split="cv_all",
        value_col="delta_vs_EVENT",
        filters={"population": "cached_subsample_rows", "source": "programme_cache",
                 "variant": "EVENT+IMP"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="EVENT+IMP_refit_all_attempts", metric="delta_per_pass_nats_vs_event",
        source_table="03_pass_a_per_pass", source_report=PASS_MD, split="cv_all",
        value_col="delta_vs_EVENT",
        filters={"population": "cached_subsample_rows", "source": "refit_all_attempts",
                 "variant": "EVENT+IMP"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="EVENT+ORACLE_refit_all_attempts", metric="delta_per_pass_nats_vs_event",
        source_table="03_pass_a_per_pass", source_report=PASS_MD, split="cv_all",
        value_col="delta_vs_EVENT",
        filters={"population": "cached_subsample_rows", "source": "refit_all_attempts",
                 "variant": "EVENT+ORACLE"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="ceiling", model="EVENT+ORACLE",
        metric="count_retention_of_per_pass_gain", source_table="03_pass_a_retention",
        source_report=PASS_MD, split="cv_all", value_col="retention",
        filters={"population": "all_matches", "variant": "EVENT+ORACLE"},
        ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="ceiling",
        model="completed_pass_variance", metric="share_irreducible_bernoulli",
        source_table="03_pass_c_variance", source_report=PASS_MD, split="confirmation",
        value_col="share_irreducible_bernoulli", filters={"where": "confirmation"},
        ci_type=NO_CI,
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="EVENT+IMP_at_the_money_line", metric="delta_brier_vs_event",
        source_table="03_pass_a_count_lines", source_report=PASS_MD, split="confirmation",
        value_col="delta_brier_vs_EVENT",
        filters={"population": "all_matches", "where": "confirmation", "offset": 0,
                 "variant": "EVENT+IMP"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="EVENT+ORACLE_at_the_money_line", metric="delta_brier_vs_event",
        source_table="03_pass_a_count_lines", source_report=PASS_MD, split="confirmation",
        value_col="delta_brier_vs_EVENT",
        filters={"population": "all_matches", "where": "confirmation", "offset": 0,
                 "variant": "EVENT+ORACLE"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="vs_proxy",
        model="volume_model_noimp_attempts", metric="delta_count_log_score_vs_proxy_nats",
        source_table="03_pass_b_deltas", source_report=PASS_MD, split="confirmation",
        value_col="delta_log_score",
        filters={"where": "confirmation", "target": "passes", "model": "model_noimp",
                 "vs": "proxy"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="imputed_state",
        model="volume_model_minus_model_noimp_attempts", metric="delta_count_log_score_nats",
        source_table="03_pass_b_deltas", source_report=PASS_MD, split="confirmation",
        value_col="delta_log_score",
        filters={"where": "confirmation", "target": "passes", "model": "model",
                 "vs": "model_noimp"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="refit_floor", model="volume_model_noimp",
        metric="refit_spread_nats_3_seeds", source_table="03_pass_b_refit",
        source_report=PASS_MD, split="confirmation", value_col="spread",
        filters={"target": "passes", "model": "model_noimp"}, ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="betting",
        model="model_att_x_model_rate_vs_proxy_6pct", metric="roi",
        source_table="03_pass_c_confirmation_roi", source_report=PASS_MD,
        split="confirmation", value_col="roi",
        filters={"market": "completed_passes", "source": "model_att x model_rate",
                 "hold": 0.06, "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="betting",
        model="model_att_x_model_rate_after_haircut_6pct", metric="roi_after_haircut",
        source_table="03_pass_c_confirmation_roi", source_report=PASS_MD,
        split="confirmation", value_col="roi_after_haircut",
        filters={"market": "completed_passes", "source": "model_att x model_rate",
                 "hold": 0.06, "where": "confirmation"},
        ci_lo_col="roi_lo_after_haircut", ci_hi_col="roi_hi_after_haircut", n_col="n_bets",
    ),
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="betting",
        model="vs_goals_calibrated_line_setting_book_6pct", metric="roi",
        source_table="03_pass_c_book_sets_line", source_report=PASS_MD,
        split="confirmation", value_col="roi",
        filters={"kind": "goals_calibrated", "hold": 0.06, "where": "confirmation"},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    # ---------------- fouls ---------------------------------------------------------
    MetricSpec(
        candidate="fouls", stage="04", experiment="vs_proxy", model="F3_plus_market_team_fouls",
        metric="delta_nb_log_score_vs_proxy_nats", source_table="04_fouls_a_team_models",
        source_report=FOULS_MD, split="confirmation", value_col="delta_vs_ref",
        filters={"market": "team_fouls", "model": "F3_plus_market", "split": "confirmation"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="vs_proxy", model="F3_plus_market_match_fouls",
        metric="delta_nb_log_score_vs_proxy_nats", source_table="04_fouls_a_match_models",
        source_report=FOULS_MD, split="confirmation", value_col="delta_vs_ref",
        filters={"market": "match_fouls", "model": "F3_plus_market", "split": "confirmation"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="refit_floor", model="F3_plus_market_team_fouls",
        metric="refit_spread_nats_5_seeds", source_table="04_fouls_a_refit",
        source_report=FOULS_MD, split="confirmation", value_col="nb_log_score",
        filters={"split": "confirmation", "model": "F3_plus_market"},
        ci_type=NO_CI, n_col=None, agg="spread",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="fouls_won_plus_opponent_foul_block", metric="delta_nats_per_appearance",
        source_table="04_fouls_b_channels", source_report=FOULS_MD, split="confirmation",
        value_col="delta_nats_per_appearance",
        filters={"market": "player_fouls_won", "split": "confirmation",
                 "step": "P1r_event_plus_referee -> P1r_plus_opponent_foul"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="fouls_won_plus_dribble_side_block", metric="delta_nats_per_appearance",
        source_table="04_fouls_b_channels", source_report=FOULS_MD, split="confirmation",
        value_col="delta_nats_per_appearance",
        filters={"market": "player_fouls_won", "split": "confirmation",
                 "step": "P1r_event_plus_referee -> P1r_plus_dribble_side"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="fouls_committed_plus_opponent_foul_block", metric="delta_nats_per_appearance",
        source_table="04_fouls_b_channels", source_report=FOULS_MD, split="confirmation",
        value_col="delta_nats_per_appearance",
        filters={"market": "player_fouls", "split": "confirmation",
                 "step": "P1r_event_plus_referee -> P1r_plus_opponent_foul"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="fouls_committed_plus_dribble_side_block", metric="delta_nats_per_appearance",
        source_table="04_fouls_b_channels", source_report=FOULS_MD, split="confirmation",
        value_col="delta_nats_per_appearance",
        filters={"market": "player_fouls", "split": "confirmation",
                 "step": "P1r_event_plus_referee -> P1r_plus_dribble_side"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="imputed_state",
        model="P3_minus_P2_fouls_committed", metric="delta_nb_log_score_nats",
        source_table="04_fouls_c_models", source_report=FOULS_MD, split="confirmation",
        value_col="delta_vs_ref",
        filters={"market": "player_fouls_state_subset", "split": "confirmation",
                 "model": "P3_plus_state minus P2_plus_matchup"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="imputed_state",
        model="P3_minus_P2_fouls_won", metric="delta_nb_log_score_nats",
        source_table="04_fouls_c_models", source_report=FOULS_MD, split="confirmation",
        value_col="delta_vs_ref",
        filters={"market": "player_fouls_won_state_subset", "split": "confirmation",
                 "model": "P3_plus_state minus P2_plus_matchup"},
        ci_lo_col="ci_lo", ci_hi_col="ci_hi",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="betting", model="team_fouls_vs_proxy_6pct",
        metric="roi", source_table="04_fouls_d_bets", source_report=FOULS_MD,
        split="confirmation", value_col="roi",
        filters={"market": "team_fouls", "split": "confirmation", "hold": 0.06},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="betting",
        model="team_fouls_after_haircut_6pct", metric="roi_after_haircut",
        source_table="04_fouls_d_bets", source_report=FOULS_MD, split="confirmation",
        value_col="roi_after_haircut",
        filters={"market": "team_fouls", "split": "confirmation", "hold": 0.06},
        ci_lo_col="roi_lo_after_haircut", ci_hi_col="roi_hi_after_haircut", n_col="n_bets",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="betting", model="match_fouls_vs_proxy_6pct",
        metric="roi", source_table="04_fouls_d_bets", source_report=FOULS_MD,
        split="confirmation", value_col="roi",
        filters={"market": "match_fouls", "split": "confirmation", "hold": 0.06},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="betting",
        model="match_fouls_after_haircut_6pct", metric="roi_after_haircut",
        source_table="04_fouls_d_bets", source_report=FOULS_MD, split="confirmation",
        value_col="roi_after_haircut",
        filters={"market": "match_fouls", "split": "confirmation", "hold": 0.06},
        ci_lo_col="roi_lo_after_haircut", ci_hi_col="roi_hi_after_haircut", n_col="n_bets",
    ),
    # ---------------- the scenario-level betting rows -----------------------------
    # The phase's question is about specific scenarios, so the scenario rows of the one
    # market with a real margin belong in the index next to the all-rows numbers.
    MetricSpec(
        candidate="corners", stage="02", experiment="betting",
        model="count_nb_team_total_fav_strong_6pct", metric="roi",
        source_table="02_corners_bets", source_report=CORNERS_MD, split="confirmation",
        value_col="roi",
        filters={"market": "team_total", "scenario": "fav_strong", "source": "count_nb",
                 "split": "confirmation", "hold": 0.06, "threshold": 0.02},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="betting",
        model="count_nb_team_total_fav_strong_6pct", metric="roi",
        source_table="02_corners_bets", source_report=CORNERS_MD, split="discovery",
        value_col="roi",
        filters={"market": "team_total", "scenario": "fav_strong", "source": "count_nb",
                 "split": "discovery", "hold": 0.06, "threshold": 0.02},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="betting",
        model="count_nb_team_total_fav_strong_6pct", metric="bet_rate",
        source_table="02_corners_bets", source_report=CORNERS_MD, split="confirmation",
        value_col="bet_rate",
        filters={"market": "team_total", "scenario": "fav_strong", "source": "count_nb",
                 "split": "confirmation", "hold": 0.06, "threshold": 0.02},
        n_col="n_rows", ci_type=NO_CI,
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="betting",
        model="count_nb_team_total_high_total_6pct", metric="roi",
        source_table="02_corners_bets", source_report=CORNERS_MD, split="confirmation",
        value_col="roi",
        filters={"market": "team_total", "scenario": "high_total", "source": "count_nb",
                 "split": "confirmation", "hold": 0.06, "threshold": 0.02},
        ci_lo_col="roi_lo", ci_hi_col="roi_hi", n_col="n_bets",
    ),
    # ---------------- the corner team line's own refit-noise floor ----------------
    MetricSpec(
        candidate="corners", stage="02", experiment="refit_floor",
        model="C1_full_team_5.5", metric="refit_spread_nats_3_seeds",
        source_table="02_corners_team_refit", source_report=CORNERS_MD,
        split="confirmation", value_col="spread",
        filters={"target": "team_over_line", "line": 5.5, "model": "C1_full"},
        ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="corners", stage="02", experiment="refit_floor",
        model="C1_full_team_count", metric="refit_spread_poisson_deviance_3_seeds",
        source_table="02_corners_team_refit", source_report=CORNERS_MD,
        split="confirmation", value_col="spread",
        filters={"target": "team_count", "model": "C1_full"},
        ci_type=NO_CI, n_col=None,
    ),
    # ---------------- the pass stage's sharpest instrument ------------------------
    # Not a haircut but the same idea in the market's own units: what a book that is as
    # sharp on pass props, relative to this proxy, as Bet365 is on goals would score.
    MetricSpec(
        candidate="pass_counts", stage="03", experiment="haircut",
        model="implied_line_setting_book", metric="model_minus_implied_book_nats",
        source_table="03_pass_c_relative_scale", source_report=PASS_MD,
        split="confirmation", value_col="model_minus_implied_book_nats",
        filters={}, ci_type=NO_CI, n_col=None,
    ),
    # ---------------- the fouls-won channels at their seed mean -------------------
    # The synthesis previously quoted the largest of five seeds for the dribble-side
    # channel; the mean with the seed range beside it is the honest form.
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="P1r_plus_dribble_side_fouls_won_seed_mean",
        metric="delta_nats_per_appearance_5_seed_mean", source_table="04_fouls_b_refit",
        source_report=FOULS_MD, split="confirmation", value_col="delta",
        filters={"market": "player_fouls_won", "split": "confirmation",
                 "quantity": "P1r_event_plus_referee -> P1r_plus_dribble_side"},
        n_col="n", ci_type="seed_range_5", agg="mean",
    ),
    MetricSpec(
        candidate="fouls", stage="04", experiment="matchup_channel",
        model="P1r_plus_opponent_foul_fouls_won_seed_mean",
        metric="delta_nats_per_appearance_5_seed_mean", source_table="04_fouls_b_refit",
        source_report=FOULS_MD, split="confirmation", value_col="delta",
        filters={"market": "player_fouls_won", "split": "confirmation",
                 "quantity": "P1r_event_plus_referee -> P1r_plus_opponent_foul"},
        n_col="n", ci_type="seed_range_5", agg="mean",
    ),
    # ---------------- the reconciled haircut (stage 06) ---------------------------
    MetricSpec(
        candidate="cards", stage="06", experiment="haircut",
        model="margin_matched_round_trip_card_universe_6pct",
        metric="haircut_roi_points", source_table="06_haircut_definitions",
        source_report=HAIRCUT_MD, split="confirmation", value_col="haircut_roi_points",
        filters={"definition": "margin_matched_round_trip", "stage": "01 cards",
                 "split": "confirmation", "hold": 0.06},
        ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="fouls", stage="06", experiment="haircut",
        model="margin_matched_round_trip_odds_universe_6pct",
        metric="haircut_roi_points", source_table="06_haircut_definitions",
        source_report=HAIRCUT_MD, split="confirmation", value_col="haircut_roi_points",
        filters={"definition": "margin_matched_round_trip", "stage": "04 fouls",
                 "split": "confirmation", "hold": 0.06},
        ci_type=NO_CI, n_col=None,
    ),
    MetricSpec(
        candidate="corners", stage="06", experiment="betting",
        model="team_total_all_after_reference_haircut_6pct",
        metric="roi_after_margin_matched_haircut", source_table="06_haircut_applied",
        source_report=HAIRCUT_MD, split="confirmation",
        value_col="after_margin_matched_round_trip",
        filters={"candidate": "corners", "market": "team_total", "scenario": "all",
                 "hold": 0.06},
        ci_lo_col="after_margin_matched_round_trip_lo", n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
    MetricSpec(
        candidate="corners", stage="06", experiment="betting",
        model="team_total_fav_strong_after_reference_haircut_6pct",
        metric="roi_after_margin_matched_haircut", source_table="06_haircut_applied",
        source_report=HAIRCUT_MD, split="confirmation",
        value_col="after_margin_matched_round_trip",
        filters={"candidate": "corners", "market": "team_total",
                 "scenario": "fav_strong", "hold": 0.06},
        ci_lo_col="after_margin_matched_round_trip_lo", n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
    MetricSpec(
        candidate="pass_counts", stage="06", experiment="betting",
        model="model_att_x_model_rate_after_reference_haircut_6pct",
        metric="roi_after_margin_matched_haircut", source_table="06_haircut_applied",
        source_report=HAIRCUT_MD, split="confirmation",
        value_col="after_margin_matched_round_trip",
        filters={"candidate": "pass_counts", "market": "completed_passes",
                 "source": "model_att x model_rate", "hold": 0.06},
        ci_lo_col="after_margin_matched_round_trip_lo", n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
    MetricSpec(
        candidate="cards", stage="06", experiment="betting",
        model="match_cards_after_reference_haircut_6pct",
        metric="roi_after_margin_matched_haircut", source_table="06_haircut_applied",
        source_report=HAIRCUT_MD, split="confirmation",
        value_col="after_margin_matched_round_trip",
        filters={"candidate": "cards", "market": "match_cards_proxy_line", "hold": 0.06},
        ci_lo_col="after_margin_matched_round_trip_lo", n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
    MetricSpec(
        candidate="fouls", stage="06", experiment="betting",
        model="team_fouls_after_reference_haircut_6pct",
        metric="roi_after_margin_matched_haircut", source_table="06_haircut_applied",
        source_report=HAIRCUT_MD, split="confirmation",
        value_col="after_margin_matched_round_trip",
        filters={"candidate": "fouls", "market": "team_fouls", "hold": 0.06},
        ci_lo_col="after_margin_matched_round_trip_lo", n_col="n_bets",
        ci_type="match_clustered_bootstrap_95_lower_bound_only",
    ),
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_table(stem: str) -> pd.DataFrame:
    """Read a committed result table from the reports directory.

    Args:
        stem: Parquet file stem, e.g. ``"01_cards_a_metrics"``.

    Returns:
        The table.

    Raises:
        FileNotFoundError: If the table has not been produced yet.
    """
    path = C.reports_dir() / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing result table {path}; re-run the owning stage")
    return pd.read_parquet(path)


def build_index(specs: Sequence[MetricSpec] = SPECS) -> pd.DataFrame:
    """Resolve every spec and the gate sweep into the phase results index.

    Args:
        specs: Metric declarations to resolve (defaults to :data:`SPECS`).

    Returns:
        The index, shape ``[n_metrics, len(INDEX_COLUMNS)]``, sorted by candidate,
        stage and experiment.
    """
    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.source_table not in cache:
            cache[spec.source_table] = load_table(spec.source_table)
        rows.append(resolve_spec(spec, cache[spec.source_table]))
    out = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    out = pd.concat([out, gate_index(load_table("05_gate_sweep_headline"))],
                    ignore_index=True)
    order = {"cards": 0, "corners": 1, "pass_counts": 2, "fouls": 3}
    out = out.assign(_o=out["candidate"].map(order).fillna(9))
    out = out.sort_values(["_o", "stage", "experiment", "model", "metric"],
                          kind="stable").drop(columns="_o").reset_index(drop=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write ``results_index.parquet``.

    Args:
        argv: Command-line arguments (``None`` reads ``sys.argv``).

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--print", action="store_true", help="print the whole index")
    args = ap.parse_args(argv)
    idx = build_index()
    path = C.write_table(idx, "results_index")
    print(f"wrote {path} ({len(idx)} rows, {idx['candidate'].nunique()} candidates)")
    counts = idx.groupby(["candidate", "experiment"]).size().rename("n_metrics")
    print(counts.to_string())
    if args.print:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(idx.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
