# Event Ingestion

Handles acquisition, normalization, and processing of high-frequency structured event data from GDELT, POLECAT, and the ICEWS archive. This is the fast-update channel that provides continuous low-latency signals for actor memory updates.

## Scope

This component covers event data retrieval, cleaning, deduplication, normalization, and representation encoding. It feeds into both the training pipeline (historical events) and the structured event stream (Layer 3) that runs in parallel with text processing during model operation.

## Data Sources

### GDELT (Global Database of Events, Language, and Tone)

The largest open event dataset.

- **Volume:** ~100,000+ structured event records per day from 300+ event categories
- **Coverage:** 1979 to present, updated every 15 minutes
- **Access:** Free via Google BigQuery (1 TB/month free) or bulk CSV at `data.gdeltproject.org`
- **Python:** `gdeltPyR` library (`github.com/linwoodc3/gdeltPyR`)
- **Key fields:** Source actor, target actor, CAMEO event code, Goldstein scale score (-10 to +10), quad-class (verbal/material x cooperation/conflict), tone, article URL, date
- **Known issues:**
  - Opaque coding pipeline
  - Significant event duplication (one real-world event generates many records from multiple articles)
  - English/Western media bias
  - Weekend periodicity artifacts
  - Actor resolution errors

### POLECAT (POLitical Event Classification, Attributes, and Types)

The active successor to ICEWS, substantially higher precision than GDELT.

- **Coverage:** 2018 to present
- **Access:** Free weekly downloads at Harvard Dataverse (`doi:10.7910/DVN/AJGVIT`)
- **Advantages over GDELT:**
  - Trained ML classifiers (transformer-based NGEC pipeline) rather than dictionary pattern matching
  - Wikipedia-based entity linking for actor resolution
  - PLOVER ontology (cleaner than CAMEO)
- **License:** No restrictions on ML use

### ICEWS Archive (1995–April 2023)

The predecessor to POLECAT. Standard benchmark dataset for temporal knowledge graph research.

- **Volume:** ~270 million events covering 1995–2023
- **Access:** Harvard Dataverse (`doi:10.7910/DVN/28075`)
- **Benchmark usage:** TComplEx, TNTComplEx, and RE-NET all evaluate on ICEWS14 and ICEWS05-15 subsets
- **License:** No restrictions on ML use

### Important Exclusion

**ACLED:** 2025 terms of service explicitly prohibit using its data to train, test, develop, or improve ML models, LLMs, or AI systems. **Do not use ACLED for training.**

## Event Ontology

### CAMEO (Legacy, Still in GDELT)
- ~250 hierarchically arranged codes across 20 top-level categories
- Maps to Goldstein scale (-10 to +10)
- Quad-class aggregation (verbal/material x cooperation/conflict) most commonly used in ML

### PLOVER (Current Standard, Used in POLECAT)
- 18 event types with event-mode-context scheme
- Separates *what* happened from *how* (verbal, hypothetical, actual) and *why*
- Adds magnitude fields (dead, injured, size)
- JSON interchange format

**The 18 PLOVER categories:**

| Code | Category | Goldstein Range |
|------|----------|----------------|
| AID | Provide aid | +3 to +8 |
| AGREE | Make agreement | +4 to +8 |
| CONSULT | Consult | +1 to +4 |
| COOP | Cooperate | +2 to +6 |
| DEMAND | Make demand | -2 to -5 |
| DISAPPROVE | Express disapproval | -1 to -4 |
| ENGAGE | Engage in diplomatic exchange | +1 to +3 |
| FIGHT | Use conventional military force | -8 to -10 |
| INVESTIGATE | Investigate | 0 to -2 |
| MOBILIZE | Mobilize/increase readiness | -3 to -6 |
| PROTEST | Protest/demonstrate | -1 to -3 |
| REDUCE | Reduce relations | -2 to -5 |
| REJECT | Reject | -2 to -4 |
| SANCTION | Impose sanctions | -4 to -8 |
| SEIZE | Seize/arrest | -4 to -7 |
| THREATEN | Threaten | -4 to -7 |
| YIELD | Concede/yield | +2 to +5 |
| OTHER | Other | varies |

**Ontology specification:** `github.com/openeventdata/PLOVER`

## Event Representation

Each structured event `(source_actor, event_type, target_actor, t, goldstein)` is encoded as:

```python
e_event = concat([
    e_r,                    # learnable relation type embedding, dim d
    goldstein_scalar,       # normalized to [-1, 1]
    event_mode_encoding,    # verbal/hypothetical/actual (3-dim one-hot)
    magnitude_features,     # casualties, group size if available
    time2vec(t),            # temporal encoding
])  # total shape: [d + 6 or so]
```

Each of the 18 PLOVER event types gets a dense embedding vector `e_r ∈ R^d` in the same space as actor vectors. Training discovers the geometry — event types that co-occur between similar actor pairs will cluster in the embedding space.

## Critical Normalization

**Always normalize event counts against total article volume in the same time window.** Raw event counts correlate with media attention cycles, not just real-world activity. This is the single most important data quality step.

## Deduplication

GDELT generates many records from a single real-world event (one event → multiple articles → multiple records). Deduplication is essential before count-based features or training signals are computed.

## Fast GRU Update (Layer 3)

The structured event stream provides fast, low-latency actor memory updates running in parallel with the text stream:

```python
# Source actor update (active role)
h_source(t) = GRU_source(h_source(t⁻), concat(e_event, h_target(t⁻)))

# Target actor update (passive role)
h_target(t) = GRU_target(h_target(t⁻), concat(e_event, h_source(t⁻)))
```

Two GRUs share parameters but have different roles. Both actors' memories update, conditioned on their respective position (initiator vs. recipient). This is faster and cheaper than the text stream — no encoder to run.

When the text stream later processes articles about the same event, the gating mechanism naturally suppresses redundant updates.

## Poisson Count Modeling

For high-frequency event types (verbal cooperation, diplomatic consultations), model the *count* per dyad per time period rather than binary occurrence:

```python
# Negative Binomial count model (handles overdispersion)
count_pred = NB(mu=exp(score(s, r, o, t)), alpha=dispersion_param)
L_count = -NB.log_prob(observed_count)
```

This handles zero-inflation naturally and avoids awkward binary framing for events that occur multiple times per period.

## Key Dependencies

- `gdeltPyR` — GDELT data retrieval (`github.com/linwoodc3/gdeltPyR`)
- `PLOVER` — Event ontology specification (`github.com/openeventdata/PLOVER`)
- `PETRARCH3` — Rule-based event coder, PLOVER-compatible (`github.com/oudalab/petrarch3`)
- Google BigQuery client — For GDELT access at scale

## Build Phase Mapping

| Phase | Role |
|-------|------|
| Phase 1 (Months 1–2) | GDELT event counts by CAMEO quad-class as tabular features |
| Phase 2 (Months 3–5) | Temporal knowledge graph edges (countries as nodes, events as typed timestamped edges) |
| Phase 3 (Months 6–12) | Real-time GDELT/POLECAT ingestion. Structured event fast-update stream (Layer 3) |

## Architecture Reference

Corresponds to **Data Sources — Event Data** (Section 3.1), **Event Ontology** (Section 4), and **Layer 3: Structured Event Stream** (Section 8) in the architecture design document.
