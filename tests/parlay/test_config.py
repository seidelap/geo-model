from __future__ import annotations

from pathlib import Path

import pytest

from geo_model.parlay.config import ParlayConfig, american_to_decimal


def test_american_to_decimal() -> None:
    assert american_to_decimal(-110) == pytest.approx(1.9090909)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEO_MODEL_DATA_DIR", "/tmp/xyz")
    cfg = ParlayConfig()
    assert cfg.data_dir == Path("/tmp/xyz")
    assert cfg.games_parquet.suffix == ".parquet"
    assert cfg.leg_decimal == pytest.approx(american_to_decimal(-110))
