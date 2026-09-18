"""Shared IO helpers for the privileged-tracking research code."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "privileged"


def data_dir() -> Path:
    """Raw data root, configurable through ``PRIV_DATA_DIR``."""
    return Path(os.environ.get("PRIV_DATA_DIR", DEFAULT_DATA_DIR))


def nfl_dir() -> Path:
    """Directory holding ``bdb2017/`` and ``nflverse/``."""
    return data_dir() / "nfl"


def sb_dir() -> Path:
    """Directory holding StatsBomb ``events/``, ``lineups/``, ``three-sixty/``, ``matches/``."""
    return data_dir() / "sb"


def reports_dir() -> Path:
    """Directory for machine-written result tables (committed)."""
    p = REPO_ROOT / "research" / "privileged_tracking" / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(sport: str) -> Path:
    """Per-sport processed-data directory (not committed)."""
    p = data_dir() / "processed" / sport
    p.mkdir(parents=True, exist_ok=True)
    return p
