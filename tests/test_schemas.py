"""Tests for domain data schemas."""

from __future__ import annotations

from datetime import date, datetime

from geo_model.schemas import ActorRecord, CuratedArticle, NormalizedEvent


def test_normalized_event_creation():
    event = NormalizedEvent(
        event_id="abc123",
        source_actor_id="state:USA",
        target_actor_id="state:RUS",
        event_type="SANCTION",
        event_mode="actual",
        goldstein_score=-0.8,
        event_date=date(2024, 3, 1),
        data_source="polecat",
        confidence=0.95,
    )
    assert event.source_actor_id == "state:USA"
    assert event.event_type == "SANCTION"
    assert event.goldstein_score == -0.8


def test_actor_record_creation():
    actor = ActorRecord(
        actor_id="state:USA",
        actor_type="state",
        name="United States",
        aliases=["US", "USA", "United States of America"],
        iso_alpha3="USA",
    )
    assert actor.actor_id == "state:USA"
    assert len(actor.aliases) == 3
    assert actor.source_code_mappings == {}


def test_curated_article_creation():
    article = CuratedArticle(
        article_id="sha256hash",
        title="Test Article",
        body_text="Body text content here.",
        publish_date=date(2024, 6, 15),
        source_domain="reuters.com",
        language="en",
        relevance_score=0.42,
        minhash_signature=b"\x00\x01",
    )
    assert article.relevance_score == 0.42
    assert article.language == "en"
