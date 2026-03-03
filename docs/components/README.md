# Component Architecture

This directory contains precise specifications for each component of the geopolitical event prediction model. Each component is a self-contained work stream with defined inputs, outputs, and interfaces.

These components cover everything needed to produce a trained, validated model. The application layer (visualization, scenario simulation, API) lives in a separate repository.

## Component Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                     1. TRAINING DATA CURATION                        │
│                                                                      │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ Text Corpus │  │ Structured Event │  │ Structural/Static Data  │ │
│  │ Pipeline    │  │ Pipeline         │  │ Pipeline                │ │
│  └──────┬──────┘  └────────┬─────────┘  └───────────┬─────────────┘ │
│         │                  │                        │               │
└─────────┼──────────────────┼────────────────────────┼───────────────┘
          │                  │                        │
          ▼                  │                        ▼
┌──────────────────┐         │         ┌──────────────────────────────┐
│ 2. ACTOR         │◄────────┼────────►│ 3. TARGET EVENT DEFINITION   │
│    REGISTRY &    │         │         │    & DATASET CONSTRUCTION    │
│    INITIALIZATION│◄────────┘         │                              │
└────────┬─────────┘                   └──────────────┬───────────────┘
         │                                            │
         │         ┌──────────────────────┐           │
         └────────►│ 4. MODEL             │◄──────────┘
                   │    ARCHITECTURE      │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │ 5. TRAINING          │
                   │    PIPELINE          │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │ 6. MODEL VALIDATION  │
                   │    & CALIBRATION     │
                   └──────────────────────┘
```

## Components

| # | Component | Document | Purpose |
|---|-----------|----------|---------|
| 1 | [Training Data Curation](01-training-data-curation.md) | `01-training-data-curation.md` | Ingest, filter, normalize, and store raw data from all sources |
| 2 | [Actor Registry & Initialization](02-actor-registry.md) | `02-actor-registry.md` | Define the actor set, collect structural features, bootstrap embeddings |
| 3 | [Target Event Definition](03-target-event-definition.md) | `03-target-event-definition.md` | Define prediction targets, construct train/val/test datasets, negative sampling |
| 4 | [Model Architecture](04-model-architecture.md) | `04-model-architecture.md` | The 6-layer neural architecture: memory, text stream, event stream, graph, prediction, calibration |
| 5 | [Training Pipeline](05-training-pipeline.md) | `05-training-pipeline.md` | Multi-phase training: tabular baseline → structural pretrain → self-supervised → supervised |
| 6 | [Model Validation & Calibration](06-model-validation.md) | `06-model-validation.md` | Metrics, evaluation protocol, calibration, benchmark comparisons |

## Dependency Order

Components have a natural dependency order for implementation:

1. **Training Data Curation** — no dependencies, start here
2. **Actor Registry & Initialization** — depends on structural data from (1)
3. **Target Event Definition** — depends on event data from (1) and actor registry from (2)
4. **Model Architecture** — depends on data schemas from (1–3)
5. **Training Pipeline** — depends on all of (1–4)
6. **Model Validation** — depends on trained model from (5)

Components 1, 2, and 3 can be developed partially in parallel. Component 4 (architecture) can be designed while data pipelines are being built. Components 5 and 6 require everything upstream.

## Cross-Component Interfaces

Key data structures that flow between components:

- **Curated Article** (1 → 2, 4): filtered text with metadata, actor mentions, timestamp
- **Normalized Event** (1 → 3, 4): `(source_actor_id, event_type, target_actor_id, timestamp, goldstein, mode, magnitude)`
- **Actor Record** (2 → 3, 4, 5): `(actor_id, actor_type, structural_features, initial_embedding)`
- **Training Example** (3 → 5): `(positive_event, negative_events, context_window, temporal_weight)`
- **Actor Memory Vector** (4 → 5, 6): `h_i(t) ∈ R^d` — the persistent state per actor
- **Prediction Output** (4 → 6): `P(event_type | actor_i, actor_j, horizon)` with confidence
