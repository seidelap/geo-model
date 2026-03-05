"""Domain data schemas shared across components.

All structured data flowing between components uses these dataclasses.
No raw dicts for domain objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class RawArticle:
    """Raw article from Common Crawl or other text source (C1 §1)."""

    article_id: str  # SHA-256 of (url + publish_date)
    url: str
    title: str
    body_text: str
    publish_date: date
    source_domain: str
    language: str
    ingestion_timestamp: datetime


@dataclass(frozen=True)
class CuratedArticle:
    """Article that passed all text filters (C1 §1.3)."""

    article_id: str
    title: str
    body_text: str
    publish_date: date
    source_domain: str
    language: str
    relevance_score: float
    minhash_signature: bytes


@dataclass(frozen=True)
class NormalizedEvent:
    """Event normalized across GDELT/POLECAT/ICEWS (C1 §2.4)."""

    event_id: str  # SHA-256 of (source, target, type, date, source_url)
    source_actor_id: str  # canonical actor ID, e.g. "state:USA"
    target_actor_id: str
    event_type: str  # PLOVER category
    event_mode: str  # verbal / hypothetical / actual
    goldstein_score: float  # normalized to [-1, +1]
    event_date: date
    magnitude_dead: int = 0
    magnitude_injured: int = 0
    magnitude_size: int = 0
    data_source: str = ""  # polecat / gdelt / icews
    confidence: float = 0.0  # [0, 1]
    source_url: str = ""


@dataclass
class ActorRecord:
    """Actor registry entry (C2 §1.2)."""

    actor_id: str  # e.g. "state:USA", "igo:NATO"
    actor_type: str  # state / igo / regional_org / nsag / company / leader / subnational
    name: str
    aliases: list[str] = field(default_factory=list)
    iso_alpha3: str | None = None
    parent_actor_id: str | None = None
    active_from: date | None = None
    active_to: date | None = None
    source_code_mappings: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SurvivalTrainingExample:
    """Primary training example with survival target (C3 §8.1)."""

    source_actor_id: str
    target_actor_id: str
    event_type: str
    reference_date: date
    event_occurred: bool
    days_to_event: float
    censored: bool
    time_bin_index: int
    is_negative: bool = False
    sampling_method: str = "n/a"
    negative_confidence: float = 1.0
    temporal_weight: float = 1.0
    event_type_weight: float = 1.0
    context_window_key: str = ""


@dataclass(frozen=True)
class IntensityTrainingExample:
    """Training example for Hawkes intensity targets (C3 §8.2)."""

    source_actor_id: str
    target_actor_id: str
    event_type: str
    period_start: date
    period_end: date
    event_times: tuple[float, ...] = ()  # days from period_start
    event_count: int = 0
    mean_goldstein: float = 0.0
    temporal_weight: float = 1.0
    event_type_weight: float = 1.0
    context_window_key: str = ""
