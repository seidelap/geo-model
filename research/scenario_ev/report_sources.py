"""Anti-staleness guard: every committed report is pinned to the tables it was built from.

Each report in ``reports/`` claims that its numbers are read out of the parquet tables
beside it rather than transcribed. That claim is only worth something if a report cannot
quietly fall behind a table that was regenerated afterwards -- which is exactly what
happened once in this phase: stage 01's haircut table was rebuilt after ``02_corners.md``
was last rendered, and the committed corner report went on quoting the old numbers, so
re-running its own documented render command changed it.

This module pins each report to a digest of every table it reads. ``check`` recomputes the
digests and names anything that moved; the test suite calls it, so a stale report fails
CI instead of being discovered by a reader. After re-running a stage **and** re-rendering
the reports that depend on it, refresh the manifest with ``--update``.

Dependencies are declared, not inferred: a report depends on its own stage's tables (by
filename prefix) plus the cross-stage tables its renderer reads by name. The
cross-stage entries are the ones that matter -- a same-stage table is almost always
regenerated in the same command that re-renders the report, a cross-stage one is not.

Run::

    cd /home/user/geo-model
    python -m research.scenario_ev.report_sources            # check, exit 1 on drift
    python -m research.scenario_ev.report_sources --update   # re-pin after a re-render
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from research.scenario_ev import common as C

#: Manifest file, committed beside the reports.
MANIFEST = "report_sources.json"

#: ``report -> (own-table prefixes, extra tables read by name)``.
#:
#: The synthesis reads from every stage, so its prefix list is all of them; that is the
#: point, because the synthesis is the document most exposed to a stage being re-run.
DEPENDENCIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "01_cards.md": (("01_cards_",), ()),
    "02_corners.md": (("02_corners_",),
                      ("01_cards_c_goals_gap.parquet", "01_cards_c_haircut.parquet",
                       "04_fouls_d_haircut.parquet")),
    "03_pass_counts.md": (("03_pass_",), ()),
    "04_fouls.md": (("04_fouls_",), ("05_gate_sweep_summary.parquet",)),
    "05_gate_sweep.md": (("05_gate_sweep_",),
                         ("01_cards_a_fit_info.parquet", "01_cards_c_fit_info.parquet")),
    "06_haircut.md": (("06_haircut_",),
                      ("01_cards_a_bets.parquet", "01_cards_c_bets.parquet",
                       "01_cards_c_goals_gap.parquet",
                       "02_corners_bets.parquet",
                       "02_corners_goals_haircut_metrics.parquet",
                       "02_corners_goals_haircut_roi.parquet",
                       "03_pass_c_confirmation_roi.parquet",
                       "03_pass_goals_gap.parquet", "03_pass_goals_haircut.parquet",
                       "04_fouls_d_bets.parquet", "04_fouls_d_goals_metrics.parquet",
                       "04_fouls_d_haircut.parquet")),
    "README.md": (("01_cards_", "02_corners_", "03_pass_", "04_fouls_",
                   "05_gate_sweep_", "06_haircut_"), ("results_index.parquet",)),
}


def digest(path: Path) -> str:
    """MD5 of a file's bytes.

    Args:
        path: File to digest.

    Returns:
        Hex digest, or ``"missing"`` when the file does not exist.
    """
    if not path.exists():
        return "missing"
    return hashlib.md5(path.read_bytes()).hexdigest()


def tables_for(report: str, rd: Path) -> list[str]:
    """Names of the tables a report depends on, sorted.

    Args:
        report: Report file name, a key of :data:`DEPENDENCIES`.
        rd: Reports directory.

    Returns:
        Sorted file names (not paths) of the parquet tables the report reads.
    """
    prefixes, extra = DEPENDENCIES[report]
    names = {p.name for p in rd.glob("*.parquet")
             if any(p.name.startswith(pref) for pref in prefixes)}
    names.update(extra)
    return sorted(names)


def manifest(rd: Path | None = None) -> dict[str, dict[str, str]]:
    """Current digests of every report's source tables.

    Args:
        rd: Reports directory; defaults to the package's own.

    Returns:
        ``{report: {table_name: md5}}``.
    """
    rd = rd or C.reports_dir()
    return {rep: {name: digest(rd / name) for name in tables_for(rep, rd)}
            for rep in sorted(DEPENDENCIES)}


def load(rd: Path | None = None) -> dict[str, dict[str, str]]:
    """Read the committed manifest.

    Args:
        rd: Reports directory; defaults to the package's own.

    Returns:
        The stored ``{report: {table: md5}}``, or an empty dict when absent.
    """
    rd = rd or C.reports_dir()
    p = rd / MANIFEST
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def check(rd: Path | None = None) -> list[str]:
    """Compare the committed manifest with the tables on disk.

    Args:
        rd: Reports directory; defaults to the package's own.

    Returns:
        Human-readable descriptions of every drift found; empty when every report is
        current. A missing manifest is itself reported as drift.
    """
    rd = rd or C.reports_dir()
    stored, now = load(rd), manifest(rd)
    if not stored:
        return [f"{MANIFEST} is missing; run "
                "`python -m research.scenario_ev.report_sources --update`"]
    out: list[str] = []
    for rep in sorted(set(stored) | set(now)):
        if rep not in stored:
            out.append(f"{rep}: not in the manifest")
            continue
        if rep not in now:
            out.append(f"{rep}: in the manifest but no dependency rule")
            continue
        a, b = stored[rep], now[rep]
        for name in sorted(set(a) | set(b)):
            if a.get(name) != b.get(name):
                out.append(f"{rep}: {name} changed since the report was rendered "
                           f"({a.get(name, 'absent')} -> {b.get(name, 'absent')})")
    return out


def update(rd: Path | None = None) -> Path:
    """Re-pin every report to the tables currently on disk.

    Call this only after re-rendering the reports, never instead of re-rendering them.

    Args:
        rd: Reports directory; defaults to the package's own.

    Returns:
        Path of the written manifest.
    """
    rd = rd or C.reports_dir()
    p = rd / MANIFEST
    p.write_text(json.dumps(manifest(rd), indent=1, sort_keys=True) + "\n")
    return p


def as_frame(rd: Path | None = None) -> pd.DataFrame:
    """The manifest as a tidy frame, for inspection.

    Args:
        rd: Reports directory; defaults to the package's own.

    Returns:
        One row per (report, table) with the stored and current digests.
    """
    stored, now = load(rd), manifest(rd)
    rows = [{"report": rep, "table": name, "stored": stored.get(rep, {}).get(name),
             "current": digests.get(name)}
            for rep, digests in now.items() for name in digests]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI: check the manifest, or refresh it with ``--update``.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        ``0`` when current (or after an update), ``1`` when drift was found.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true",
                    help="re-pin the manifest to the tables on disk")
    args = ap.parse_args(argv)
    rd = C.reports_dir()
    if args.update:
        print(f"wrote {update(rd)}")
        return 0
    bad = check(rd)
    for line in bad:
        print(f"STALE: {line}")
    print("report_sources: " + (f"{len(bad)} drift(s)" if bad else "all reports current"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
