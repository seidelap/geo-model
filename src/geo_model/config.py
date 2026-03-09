"""Global configuration for geo-model.

All hyperparameters, paths, and model settings are defined here as Pydantic
models. Nothing is hardcoded in module code — import from this module.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class PLOVERType(str, Enum):
    """The 18 PLOVER event categories."""

    AID = "AID"
    AGREE = "AGREE"
    CONSULT = "CONSULT"
    COOP = "COOP"
    DEMAND = "DEMAND"
    DISAPPROVE = "DISAPPROVE"
    ENGAGE = "ENGAGE"
    FIGHT = "FIGHT"
    INVESTIGATE = "INVESTIGATE"
    MOBILIZE = "MOBILIZE"
    PROTEST = "PROTEST"
    REDUCE = "REDUCE"
    REJECT = "REJECT"
    SANCTION = "SANCTION"
    SEIZE = "SEIZE"
    THREATEN = "THREATEN"
    YIELD = "YIELD"
    OTHER = "OTHER"


class ActorType(str, Enum):
    """Actor type prefixes for canonical IDs."""

    STATE = "state"
    IGO = "igo"
    REGIONAL_ORG = "regional_org"
    NSAG = "nsag"
    COMPANY = "company"
    LEADER = "leader"
    SUBNATIONAL = "subnational"


# K=17 non-uniform time bins (days). Each tuple is (start, end).
TIME_BINS: list[tuple[float, float]] = [
    (0, 1), (1, 2), (2, 3), (3, 5), (5, 7),
    (7, 10), (10, 14), (14, 21), (21, 30), (30, 45),
    (45, 60), (60, 90), (90, 120), (120, 150), (150, 180),
    (180, 270), (270, 365),
]
NUM_TIME_BINS: int = 17

SYMMETRIC_EVENT_TYPES: set[str] = {"AGREE", "CONSULT", "ENGAGE", "COOP"}


class PathConfig(BaseSettings):
    """Data and artifact paths. Override via environment variables."""

    data_raw: Path = Field(default=Path("data/raw"))
    data_curated: Path = Field(default=Path("data/curated"))
    data_processed: Path = Field(default=Path("data/processed"))
    checkpoints: Path = Field(default=Path("checkpoints"))
    configs: Path = Field(default=Path("configs"))

    model_config = {"env_prefix": "GEO_"}


class ModelConfig(BaseSettings):
    """Model hyperparameters."""

    # Embedding
    embedding_dim: int = Field(default=256, description="Actor embedding dimension d")
    sketch_dim: int = Field(default=64, description="Sketch vector dimension for text filtering")

    # ConfliBERT
    text_encoder_name: str = "eventdata/ConfliBERT"
    text_encoder_dim: int = 768  # BERT-base hidden size
    max_seq_length: int = 512

    # Actor memory
    ema_alpha: float = Field(default=0.98, ge=0.95, le=1.0)
    memory_gate_type: str = "gru"  # "gru" or "sigmoid"

    # Attention (Layer 4)
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Prediction head (Layer 5)
    num_time_bins: int = NUM_TIME_BINS
    hawkes_event_types: list[str] = Field(
        default=["CONSULT", "ENGAGE", "COOP", "DISAPPROVE"],
        description="Event types modeled with Hawkes intensity (high-frequency)",
    )

    model_config = {"env_prefix": "GEO_MODEL_"}


class TrainingConfig(BaseSettings):
    """Training pipeline parameters."""

    # Phase control
    phase: int = Field(default=0, ge=0, le=3)

    # General
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Negative sampling
    negatives_per_positive: int = 10
    hard_negative_ratio: float = 0.15

    # Loss weights
    survival_loss_weight: float = 1.0
    hawkes_loss_weight: float = 0.5
    concordance_loss_weight: float = 0.1
    focal_loss_gamma: float = 2.0

    # Temporal weighting
    temporal_half_life_days: int = 270

    # Data splits
    train_end: str = "2024-09-30"
    val_start: str = "2024-10-01"
    val_end: str = "2025-03-31"
    test_start: str = "2025-04-01"
    test_end: str = "2026-03-31"

    model_config = {"env_prefix": "GEO_TRAIN_"}
