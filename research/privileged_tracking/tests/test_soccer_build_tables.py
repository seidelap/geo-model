"""Smoke tests for the pure helpers of the soccer 01 build driver (no data access)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from research.privileged_tracking.soccer import build_tables as bt


def test_output_schema_is_typed_and_prefixed() -> None:
    cols = bt.output_columns()
    assert cols[: len(bt.ID_COLS)] == list(bt.ID_COLS)
    assert len(cols) == len(set(cols))
    schema = bt.output_schema()
    assert schema.names == cols
    assert any(c.startswith("f_") for c in cols) and any(c.startswith("y_") for c in cols)
    assert any(c.startswith("sff_") for c in cols) and any(c.startswith("seq_type_") for c in cols)
    assert bt._pa_type("match_id") == pa.int64()
    assert bt._pa_type("seq_type_1") == pa.int16()
    assert bt._pa_type("f_w10_n_pass") == pa.int32()
    assert bt._pa_type("f_x") == pa.float32()


def test_quantiles_and_md_table_handle_empty_and_nan() -> None:
    q = bt._quantiles(pd.Series([1.0, np.nan, 3.0, 5.0]))
    assert q["n"] == 3 and q["mean"] == 3.0 and q["p50"] == 3.0
    e = bt._quantiles(pd.Series([], dtype=float))
    assert e["n"] == 0 and np.isnan(e["mean"])
    md = bt.md_table(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}))
    lines = md.split("\n")
    assert lines[0] == "| a | b |" and len(lines) == 4
