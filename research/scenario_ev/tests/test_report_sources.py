"""Tests for the report staleness guard.

The unit tests run on a temporary directory of synthetic files. The last test runs
against the committed reports and is the one that actually protects the phase: if a stage
is re-run and a report is not re-rendered, it fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.scenario_ev import common as C
from research.scenario_ev import report_sources as RS


@pytest.fixture()
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A reports directory with one report, one own table and one cross-stage table."""
    monkeypatch.setitem(RS.DEPENDENCIES, "zz_demo.md", (("zz_demo_",), ("xx_other.parquet",)))
    for name in list(RS.DEPENDENCIES):
        if name != "zz_demo.md":
            monkeypatch.delitem(RS.DEPENDENCIES, name)
    (tmp_path / "zz_demo.md").write_text("# demo\n")
    (tmp_path / "zz_demo_a.parquet").write_bytes(b"own-table-v1")
    (tmp_path / "xx_other.parquet").write_bytes(b"cross-stage-v1")
    (tmp_path / "zz_unrelated.parquet").write_bytes(b"not a dependency")
    return tmp_path


def test_tables_for_takes_prefix_plus_named_extras(tiny: Path) -> None:
    assert RS.tables_for("zz_demo.md", tiny) == ["xx_other.parquet", "zz_demo_a.parquet"]


def test_check_reports_a_missing_manifest(tiny: Path) -> None:
    bad = RS.check(tiny)
    assert len(bad) == 1 and RS.MANIFEST in bad[0]


def test_update_then_check_is_clean(tiny: Path) -> None:
    RS.update(tiny)
    assert RS.check(tiny) == []


def test_cross_stage_table_change_is_caught(tiny: Path) -> None:
    """The exact failure this module exists for: a *different* stage's table moved."""
    RS.update(tiny)
    (tiny / "xx_other.parquet").write_bytes(b"cross-stage-v2")
    bad = RS.check(tiny)
    assert len(bad) == 1
    assert "xx_other.parquet" in bad[0] and "zz_demo.md" in bad[0]


def test_own_table_change_is_caught(tiny: Path) -> None:
    RS.update(tiny)
    (tiny / "zz_demo_a.parquet").write_bytes(b"own-table-v2")
    assert any("zz_demo_a.parquet" in line for line in RS.check(tiny))


def test_unrelated_table_change_is_ignored(tiny: Path) -> None:
    RS.update(tiny)
    (tiny / "zz_unrelated.parquet").write_bytes(b"still not a dependency")
    assert RS.check(tiny) == []


def test_new_dependency_table_is_caught(tiny: Path) -> None:
    """A table added under a watched prefix means the report has not seen it."""
    RS.update(tiny)
    (tiny / "zz_demo_b.parquet").write_bytes(b"a new table")
    assert any("zz_demo_b.parquet" in line for line in RS.check(tiny))


def test_digest_of_missing_file(tmp_path: Path) -> None:
    assert RS.digest(tmp_path / "nope.parquet") == "missing"


def test_manifest_round_trips_through_json(tiny: Path) -> None:
    p = RS.update(tiny)
    assert json.loads(p.read_text()) == RS.manifest(tiny)


@pytest.mark.skipif(not (C.reports_dir() / RS.MANIFEST).exists(),
                    reason="manifest not built yet")
def test_committed_reports_are_current() -> None:
    """Every committed report is pinned to the tables currently on disk."""
    bad = RS.check()
    assert bad == [], "stale report(s):\n  " + "\n  ".join(bad)
