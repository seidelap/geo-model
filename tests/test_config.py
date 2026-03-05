"""Tests for global configuration and constants."""

from __future__ import annotations

from geo_model.config import (
    NUM_TIME_BINS,
    TIME_BINS,
    ModelConfig,
    PathConfig,
    PLOVERType,
    TrainingConfig,
)


def test_time_bins_count():
    assert len(TIME_BINS) == NUM_TIME_BINS == 17


def test_time_bins_contiguous():
    """Time bins must cover [0, 365] with no gaps."""
    for i in range(len(TIME_BINS) - 1):
        assert TIME_BINS[i][1] == TIME_BINS[i + 1][0], f"Gap between bin {i} and {i + 1}"
    assert TIME_BINS[0][0] == 0
    assert TIME_BINS[-1][1] == 365


def test_plover_types_count():
    assert len(PLOVERType) == 18


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.embedding_dim == 256
    assert cfg.num_time_bins == 17
    assert 0.95 <= cfg.ema_alpha <= 1.0


def test_path_config_defaults():
    cfg = PathConfig()
    assert cfg.data_raw.name == "raw"


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.negatives_per_positive == 10
    assert cfg.phase == 0
