"""Markdown table rendering shared by the report writers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cell(v: object, floatfmt: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return floatfmt.format(float(v))
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Integers print without decimals, floats with ``floatfmt``, NaN as an empty cell.
    Integer-valued float columns (e.g. counts that went through a merge) print as ints.
    """
    d = df.reset_index() if index else df
    cols = [str(c).replace("|", "\\|") for c in d.columns]
    int_like = {}
    for c in d.columns:
        s = d[c]
        int_like[c] = bool(pd.api.types.is_float_dtype(s) and s.notna().any()
                           and np.all(np.mod(s.dropna().to_numpy(dtype=float), 1) == 0)
                           and s.dropna().abs().max() < 1e12 and c in _COUNT_HINTS)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # iterate column-wise lists: ``iterrows`` would upcast every numeric cell to float
    columns = [d[c].tolist() for c in d.columns]
    for values in zip(*columns):
        cells = []
        for c, v in zip(d.columns, values):
            if int_like[c] and not (v is None or (isinstance(v, float) and np.isnan(v))):
                cells.append(str(int(v)))
            else:
                cells.append(_cell(v, floatfmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_COUNT_HINTS = {"n", "plays", "games", "support", "n_candidates", "n_on_field", "false_negatives", "false_positives",
                "missed_players", "best_iter", "num_leaves", "rounds", "n_train_rows", "pass_plays", "box_labelled",
                "rushers_labelled", "coverage_labelled", "season", "week"}
