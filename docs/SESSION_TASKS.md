# Session Task Breakdown

Each session is scoped to 1-3 hours of Claude Code work. Sessions within the same
component are sequential. Sessions marked ⊕ can run in parallel once their
prerequisites are met.

---

## Phase 0 Prerequisites (Sessions 1-2)

### Session 1: Project Skeleton ✅
- [x] Create directory structure, pyproject.toml, .gitignore
- [x] Define shared config (config.py): PLOVERType, TIME_BINS, ModelConfig, TrainingConfig
- [x] Define shared schemas (schemas.py): NormalizedEvent, ActorRecord, CuratedArticle, SurvivalTrainingExample
- [x] Write tests for config and schemas

### Session 2: CAMEO→PLOVER Mapping + Event Schema Validation
- [ ] Create `src/geo_model/data/cameo_plover_map.py` with the ~250→18 mapping table
- [ ] Create `src/geo_model/data/event_normalization.py`: normalize Goldstein [-10,+10]→[-1,+1], validate event schema
- [ ] Write unit tests with example CAMEO codes and edge cases
- [ ] Store mapping as a versioned Parquet file in `configs/`

---

## Component 1: Training Data Curation (Sessions 3-7)

### Session 3: GDELT Ingestion ⊕
- [ ] `src/geo_model/data/gdelt_ingest.py`: date-bounded queries via gdeltPyR
- [ ] Parse raw GDELT records into NormalizedEvent via CAMEO→PLOVER map
- [ ] Deduplicate within-source (same actor pair + type + date → keep highest confidence)
- [ ] Write to Parquet partitioned by event_date (monthly files)
- [ ] Tests with mock GDELT data (no live API calls in tests)
- **Spec ref:** C1 §2.2, §2.4

### Session 4: POLECAT Ingestion ⊕
- [ ] `src/geo_model/data/polecat_ingest.py`: download weekly files from Harvard Dataverse
- [ ] Parse into NormalizedEvent (PLOVER-native, simpler than GDELT)
- [ ] Actor code normalization (ISO alpha-3 → canonical actor IDs)
- [ ] Cross-source dedup with GDELT (prefer POLECAT records)
- [ ] Tests with sample POLECAT data
- **Spec ref:** C1 §2.1, §2.4

### Session 5: ICEWS Historical Backfill
- [ ] `src/geo_model/data/icews_ingest.py`: bulk load from Harvard Dataverse
- [ ] CAMEO→PLOVER mapping (reuse Session 2 module)
- [ ] Store as Parquet partitioned by month, 1995–2023
- [ ] Cross-source dedup with GDELT/POLECAT
- **Spec ref:** C1 §2.3

### Session 6: Text Pipeline — Common Crawl Ingestion + Filtering
- [ ] `src/geo_model/data/text_ingest.py`: download daily WARC from S3, parse articles
- [ ] `src/geo_model/data/text_filter.py`: 4-stage pipeline:
  1. Language filter (fasttext lid.176.bin)
  2. Geopolitical relevance (TF-IDF cosine ≥ 0.15)
  3. Near-duplicate removal (MinHash LSH, Jaccard ≥ 0.8)
  4. Quality filter (100-5000 words, boilerplate removal)
- [ ] Output: CuratedArticle Parquet partitioned by publish_date
- [ ] Tests for each filter stage independently
- **Spec ref:** C1 §1.1–§1.3

### Session 7: Structural Data Pipeline
- [ ] `src/geo_model/data/structural_ingest.py`:
  - COW CINC/MID/Alliance → Parquet
  - V-Dem democracy indices → Parquet
  - Voeten UNGA ideal points → Parquet
  - World Bank indicators via API → Parquet
  - SIPRI military expenditure + arms transfers → Parquet
- [ ] `src/geo_model/data/structural_features.py`: merge into per-actor-per-year panel
  - Log normalization, z-scoring, missing value handling (interpolation + missingness flags)
  - Dyadic features: bilateral trade, alliance type, geographic distance, Voeten distance
- [ ] Tests for normalization and missing value imputation
- **Spec ref:** C1 §3.1–§3.7

---

## Component 2: Actor Registry (Sessions 8-9)

### Session 8: Actor Registry Core ⊕ (after Session 2)
- [ ] `src/geo_model/actors/registry.py`: ActorRegistry class
  - Load/save from Parquet
  - CRUD operations for ActorRecord
  - Source code mapping (GDELT CAMEO codes, POLECAT ISO, ICEWS → canonical IDs)
  - Unmapped actor logging and review queue
- [ ] `src/geo_model/actors/bootstrap.py`: initialize registry with ~193 UN states + ~35 IGOs/regional orgs
- [ ] Tests: actor ID format validation, mapping lookups, alias resolution
- **Spec ref:** C2 §1.1–§1.3

### Session 9: Actor Initialization + Sketch Vectors
- [ ] `src/geo_model/actors/initialization.py`:
  - Path A: structural projection (PCA before Phase 1, learned W_struct after)
  - Path B: text-derived (ConfliBERT mean-pool + projection)
  - Path C: neighbor average fallback
  - Name identity encoding (ConfliBERT on actor names/aliases)
  - Gated combination: h_base + gate * h_name
- [ ] `src/geo_model/actors/sketch.py`: TF-IDF sketch vectors (64-dim) for text relevance pre-filter
- [ ] Tests: output shapes [d], Path A/B/C routing logic, sketch vector computation
- **Spec ref:** C2 §3.1–§3.5, §4

---

## Component 3: Target Events & Dataset Construction (Sessions 10-12)

### Session 10: Positive Example Construction + Temporal Splits
- [ ] `src/geo_model/targets/positives.py`:
  - Event→TrainingPositive conversion
  - Within/cross-source dedup and aggregation
  - Symmetric type expansion (AGREE, CONSULT, ENGAGE, COOP → both directions)
- [ ] `src/geo_model/targets/splits.py`:
  - Strict temporal train/val/test splits
  - Expanding window robustness splits
  - Validation: no future leakage check
- [ ] Tests: directionality expansion, split boundary enforcement
- **Spec ref:** C3 §2, §6

### Session 11: Negative Sampling + Feasibility Filter
- [ ] `src/geo_model/targets/negatives.py`:
  - Mixed random/hard sampling (85/15 ratio, K=10 per positive)
  - Noise distribution (frequency^0.75, monthly recomputation)
  - Source/target/relation corruption strategies
- [ ] `src/geo_model/targets/feasibility.py`:
  - Structural feasibility filter (geographic proximity for FIGHT/MOBILIZE/SEIZE, state-level for SANCTION)
  - Temporal buffer (±3 days around known positives)
  - Negative confidence scoring based on media coverage density
- [ ] Tests: filter rejection rates, temporal buffer enforcement
- **Spec ref:** C3 §3, §4

### Session 12: Dataset Assembly + Context Windows
- [ ] `src/geo_model/targets/dataset.py`:
  - Assemble SurvivalTrainingExample and IntensityTrainingExample
  - Context window construction (dyad events 365d, actor events 180d, articles 30d)
  - Temporal and event-type weighting
  - Count targets for high-frequency event types (negative binomial)
- [ ] `src/geo_model/targets/stats.py`: compute and log dataset statistics (base rates, active dyads, coverage heatmaps)
- [ ] PyTorch Dataset/DataLoader wrappers
- [ ] Tests: context window sizes, weight calculations, dataset iteration
- **Spec ref:** C3 §4–§5, §7–§9

---

## Phase 0: LightGBM Baseline (Sessions 13-14)

### Session 13: Feature Engineering for Tabular Baseline
- [ ] `src/geo_model/training/phase0_features.py`:
  - 108 dyadic event features (18 types × 6 aggregations)
  - 16 quad-class aggregates
  - 14 structural features (GDP ratio, democracy diff, alliance, trade, etc.)
  - 4 temporal features (month, year, days since last event)
  - Total: ~142 features per dyad-month
- [ ] Tests: feature count validation, NaN handling
- **Spec ref:** C5 §2.2

### Session 14: LightGBM Training + BSS Floor
- [ ] `src/geo_model/training/phase0_train.py`:
  - LightGBM binary + Poisson objectives
  - Hyperparameter config via TrainingConfig
  - Temporal cross-validation (no future leakage)
- [ ] `src/geo_model/evaluation/bss.py`: Brier Skill Score vs climatological baseline
- [ ] `src/geo_model/evaluation/metrics.py`: ECE, log-loss, per-event-type breakdown
- [ ] Store Phase 0 BSS as the floor — all subsequent phases must beat it
- [ ] Tests: BSS calculation on synthetic data, metric computation
- **Spec ref:** C5 §2, C6 §1

---

## Component 4: Neural Architecture (Sessions 15-20)

### Session 15: Layer 1 — Actor Memory Store
- [ ] `src/geo_model/model/memory.py`:
  - Persistent h_i(t) vectors, shape [d]
  - EMA baseline decay (α ∈ [0.95, 1.0])
  - Temporal decay toward EMA baseline
  - Memory read/write interface
- [ ] Tests: decay convergence, shape invariants, EMA constraint
- **Spec ref:** C4 §2

### Session 16: Layer 2 — Text Processing Stream
- [ ] `src/geo_model/model/text_stream.py`:
  - ConfliBERT encoding (frozen or fine-tuned depending on phase)
  - Cross-attention: actor queries over document tokens
  - Gated memory update (scalar + vector gates)
  - Sketch vector pre-filter for article-actor relevance
- [ ] Tests: attention output shapes, gate value ranges, memory update
- **Spec ref:** C4 §3

### Session 17: Layer 3 — Structured Event Stream
- [ ] `src/geo_model/model/event_stream.py`:
  - GRU update from coded events
  - Event embedding (type + Goldstein + magnitude → vector)
  - Fast sequential update (no attention, just GRU)
- [ ] Tests: GRU state evolution, event embedding shapes
- **Spec ref:** C4 §4

### Session 18: Layer 4 — Actor Self-Attention
- [ ] `src/geo_model/model/self_attention.py`:
  - Full multi-head attention across all actors
  - Sparsemax/entmax for learned sparse attention patterns
  - Runs once per simulated day
- [ ] Tests: attention weight sparsity, output shape [n_actors, d]
- **Spec ref:** C4 §5

### Session 19: Layer 5 — Event Prediction Head
- [ ] `src/geo_model/model/prediction_head.py`:
  - Discrete hazard parameterization → survival curves (17 bins)
  - Monotonicity guarantee: survival = cumprod(1 - hazard)
  - Hawkes process intensity for high-frequency types
  - Dyadic scoring: f(h_i, h_j, r) → hazard/intensity
- [ ] Tests: survival monotonicity, hazard ∈ (0,1), intensity positivity
- **Spec ref:** C4 §6

### Session 20: Full Model Assembly + Layer 6 Calibration
- [ ] `src/geo_model/model/full_model.py`: wire Layers 1-5, forward pass
- [ ] `src/geo_model/model/calibration.py`: per-bin temperature scaling (Layer 6)
- [ ] End-to-end forward pass test with dummy data
- [ ] Tests: output shape [batch, 17], calibration preserves monotonicity
- **Spec ref:** C4 §7, C6

---

## Component 5: Training Pipeline (Sessions 21-24)

### Session 21: Loss Functions
- [ ] `src/geo_model/training/losses.py`:
  - DeepHit NLL (survival) + concordance ranking
  - Focal loss for class imbalance
  - Hawkes NLL for intensity targets
  - Composite loss with configurable weights
- [ ] Tests: loss computation on known inputs, gradient flow
- **Spec ref:** C5 §5 (from architecture doc §15)

### Session 22: Phase 1 — Structural Pretraining
- [ ] `src/geo_model/training/phase1_pretrain.py`:
  - Train structural projection W_struct on actor features → embedding space
  - Objective: structural neighbors should be close in embedding space
  - Quick (GPU, minutes)
- [ ] Tests: embedding distances correlate with structural similarity
- **Spec ref:** C5 §3

### Session 23: Phase 2 — Self-Supervised Pretraining
- [ ] `src/geo_model/training/phase2_selfsup.py`:
  - CPC objective on text + memory
  - Event-type prediction as auxiliary task
  - ConfliBERT fine-tuning (unfrozen)
  - Temporal text exclusion: no val/test period articles
- [ ] Tests: CPC loss decreases, temporal exclusion enforced
- **Spec ref:** C5 §4

### Session 24: Phase 3 — Supervised Fine-Tuning
- [ ] `src/geo_model/training/phase3_finetune.py`:
  - End-to-end training with survival + Hawkes losses
  - Hard negative mining (model-in-the-loop)
  - Learning rate scheduling, gradient clipping
  - Checkpoint management with BSS tracking
- [ ] Tests: training loop runs, checkpoints save/load, BSS ≥ Phase 0
- **Spec ref:** C5 §5

---

## Component 6: Validation (Sessions 25-27)

### Session 25: Evaluation Protocol
- [ ] `src/geo_model/evaluation/evaluate.py`:
  - Streaming evaluation (memory updates during test period)
  - Batch evaluation (frozen memory, diagnostic)
  - Per-event-type metric breakdown
- [ ] Tests: evaluation runs on synthetic predictions
- **Spec ref:** C6 §1

### Session 26: Calibration + Reliability Diagrams
- [ ] `src/geo_model/evaluation/calibration.py`:
  - Per-bin temperature scaling fitting on validation set
  - Reliability diagrams, ECE computation
  - Calibration-aware survival curve outputs
- [ ] Tests: perfect calibration → ECE = 0
- **Spec ref:** C6 §2-3

### Session 27: Benchmarking + Final Report
- [ ] `src/geo_model/evaluation/benchmark.py`:
  - ViEWS comparison at monthly horizon
  - Polymarket comparison where available
  - Ablation studies: drop each layer, measure BSS impact
- [ ] `src/geo_model/evaluation/report.py`: generate comprehensive evaluation report
- [ ] Tests: report generation, metric formatting
- **Spec ref:** C6 §4-5

---

## Parallelization Map

```
Session 1 (skeleton) ✅
    │
Session 2 (CAMEO map)
    │
    ├── Session 3 (GDELT) ──┐
    ├── Session 4 (POLECAT) ─┤──→ Session 5 (ICEWS)
    ├── Session 8 (registry) ┘
    │
    ├── Session 6 (text pipeline)
    │
    └── Session 7 (structural data)
            │
            └── Session 9 (actor init) ──→ Sessions 10-12 (targets)
                                                │
                                          Sessions 13-14 (Phase 0 / BSS floor)
                                                │
                                          Sessions 15-20 (architecture)
                                                │
                                          Sessions 21-24 (training)
                                                │
                                          Sessions 25-27 (validation)
```

After Session 2, you can run **two parallel Claude Code sessions**:
- **Session A:** GDELT + POLECAT + ICEWS ingestion (Sessions 3-5)
- **Session B:** Actor registry core (Session 8)

After Session 7, you can potentially run:
- **Session A:** Actor initialization (Session 9)
- **Session B:** Text pipeline (Session 6) — if not already done

Sessions 13+ are strictly sequential (each phase depends on the previous).
