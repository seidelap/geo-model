# Structured Data Curation

Handles acquisition, cleaning, normalization, and feature engineering for slow-moving structural data sources. These provide the backbone of actor initialization and the static features that anchor the embedding space geometry.

## Scope

This component covers structural (slow-moving, high-signal) data acquisition and preparation for training. It does **not** cover high-frequency event data (see `event_ingestion`) or text corpora (see `text_data_curation`).

## Data Sources

### UN General Assembly Voting Data (Voeten)

The cleanest continuous signal of state foreign-policy alignment. Provides both revealed preferences (votes) and stated preferences (speeches).

- **Voeten dataset:** Ideal point estimates from roll-call votes spanning sessions 1–78 (1946–2023), decomposed across six issue categories
- **Access:** `unvotes.unige.ch`; R package `unvotes` provides tidy access to ~738,764 vote records
- **Raw UN votes:** `un.org/en/ga/` (accessible via API)
- **Usage:** Voeten ideal point distances serve as key structural features for actor initialization and as training features in the Phase 1 tabular baseline

### Correlates of War (COW) Project

Deep historical structural data. All datasets are free CSV downloads from `correlatesofwar.org`.

- **National Material Capabilities (CINC v6.0, 1816–2016):** Military expenditure, military personnel, energy consumption, iron/steel production, urban population, total population
- **Militarized Interstate Disputes (MID v5.0, 1816–2014):** Recorded military confrontations between states
- **Alliance data (v4.1, 1816–2012):** Formal defensive, offensive, and neutrality agreements
- **Limitation:** Most datasets end 2010–2016
- **Access:** R package `peacesciencer` simplifies retrieval (`github.com/svmiller/peacesciencer`)

### SIPRI Databases

- **Arms Transfer Database:** Trend Indicator Values for major conventional weapons transfers, 1950–2024, freely downloadable
- **Military Expenditure Database:** 171 countries since 1988
- **Usage:** Arms transfer patterns encode security alignment; military expenditure encodes capability

### World Bank Open Data

- **API:** `api.worldbank.org/v2/` — 1,400+ development indicators, free unauthenticated access
- **Key indicators:** GDP, GDP per capita, trade as % of GDP, FDI flows, population
- **Usage:** Economic structural features for actor initialization and tabular baseline

### UN Comtrade / CEPII BACI (Bilateral Trade)

- **Comtrade:** 500 free API calls/day
- **CEPII BACI:** Reconciles Comtrade at HS 6-digit level (free for academic use, 1995–present)
- **Usage:** Bilateral trade volume is a key input for the structural feasibility filter (determines whether economic or military events between two actors are plausible) and for graph edge weights

### V-Dem (Varieties of Democracy)

- Democracy scores across multiple dimensions (electoral, liberal, participatory, deliberative, egalitarian)
- **Usage:** Regime type dimensions for actor initialization

## Data Processing Pipeline

### Normalization
All structural features must be normalized before use in actor initialization:

```python
structural_features = concat([
    normalize(cinc_components),          # military, economic, demographic
    normalize(vdem_democracy_scores),    # regime type dimensions
    normalize(world_bank_indicators),    # GDP, trade, development
    voeten_ideal_points,                 # UNGA alignment dimensions
])
```

### Temporal Alignment
Structural data sources update at different cadences (annually, irregularly). Align all sources to a common temporal grid, forward-filling where necessary. Most COW datasets end 2010–2016 — decide on imputation strategy for the gap to present.

### Missing Data Handling
Data density is deeply asymmetric:
- **G7 nations:** Dense coverage across all sources
- **Mid-tier states:** Adequate structured coverage
- **Small states:** Primarily UN voting records and World Bank indicators
- **Non-state actors:** No dedicated structural datasets — must construct from available signals

For sparse actors, the model relies on structural initialization (where in the embedding space does this actor belong given available characteristics?) and cross-actor similarity rather than dense feature vectors.

### Feature Engineering for Phase 1 Baseline

The Phase 1 tabular baseline (LightGBM) requires hand-engineered features from these sources:

- GDELT event counts by CAMEO quad-class (4 features per dyad)
- Lagged conflict indicators
- Structural features: Polity score, GDP, population, Voeten ideal point distances
- Recent UNGA voting record similarity
- Bilateral trade volume
- Alliance membership overlap
- Geographic proximity / contiguity

## Structural Feasibility Filter

Structural data powers the feasibility mask that filters negative samples during training:

```python
def feasible_pair(actor_i, actor_j, event_type):
    if event_type in MILITARY_EVENTS:
        return (trade_volume(i, j) > threshold or
                conflict_history(i, j) > 0 or
                geographic_proximity(i, j) < max_distance)
    if event_type in ECONOMIC_EVENTS:
        return trade_volume(i, j) > 0
    return True  # diplomatic events always feasible
```

This ensures the model trains on distinguishing plausible-but-didn't-happen from actually-happened, rather than wasting capacity on structural impossibilities (e.g., USA-military-attack-Canada).

## Storage Estimate

~500GB for full GDELT history + structural data + embeddings (S3 estimate: ~$12/month).

## Key Dependencies

- `peacesciencer` (R) — COW + V-Dem data access
- `wbdata` or `wbgapi` (Python) — World Bank API client
- `unvotes` (R) — UN voting data access
- `lightgbm` — Phase 1 tabular baseline
- `pandas`, `numpy` — Data manipulation

## Build Phase Mapping

| Phase | Role |
|-------|------|
| Phase 1 (Months 1–2) | Primary data source. Hand-engineered features for LightGBM baseline |
| Phase 2 (Months 3–5) | Actor initialization vectors. Graph edge construction |
| Phase 3 (Months 6–12) | Continuous structural feature updates. Feasibility filter for negative sampling |

## Architecture Reference

Corresponds to **Data Sources — Structural Data** (Section 3.2), **Actor Initialization** (Section 6.2), and **Negative Sampling — Structural Feasibility Filter** (Section 14.4) in the architecture design document.
