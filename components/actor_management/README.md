# Actor Management

Handles the taxonomy, initialization, and lifecycle of actor entities in the system. Actors are the fundamental units of the model — each one maintains a persistent high-dimensional vector encoding its dispositions, relationships, and alignments.

## Scope

This component covers actor registration, type classification, structural feature assembly, embedding initialization, and sparse actor handling. It does **not** cover the ongoing memory update mechanism (see `model_training`) or memory persistence infrastructure (see `model_storage`).

## Actor Taxonomy

| Type | Examples | Data Quality | Primary Sources |
|------|----------|-------------|-----------------|
| UN member states (193) | USA, China, Kenya | Excellent for major powers; poor for small states | GDELT, POLECAT, COW, Voeten |
| Major IGOs | UN, NATO, EU, WTO, IMF | Good | GDELT, official documents |
| Regional organizations | ASEAN, AU, SCO, MERCOSUR | Moderate | GDELT, treaty texts |
| Non-state armed groups | Hamas, Wagner Group, Hezbollah | Sparse and biased | GDELT (noisy), conflict databases |
| Multinational companies | Shell, Huawei, BlackRock | Poor (no dedicated geo-political datasets) | Financial news, SEC filings |
| Individual leaders | Heads of state, foreign ministers | Moderate | UNGDC, ParlSpeech, GDELT GKG persons |
| Subnational entities | Kurdistan, Taiwan, Hong Kong | Sparse | GDELT, specialized sources |

## Data Sparsity by Actor Type

Data density is deeply asymmetric — this is one of the hardest practical challenges:

- **G7 nations:** Dense coverage across all data sources; hundreds of thousands of events per year
- **Mid-tier states:** Adequate structured event coverage; limited text depth
- **Small states:** Primarily UN voting records and GDELT mentions; minimal parliamentary speech data
- **Non-state actors:** Highly variable; groups that receive media attention (ISIS, Wagner) have moderate coverage; others near zero
- **Companies and individuals:** No dedicated geopolitical event datasets; must build from financial news and social media

## Actor Memory Vector

Each actor `i` maintains a persistent memory vector:

```
h_i(t) ∈ R^d,   d = 256 or 512
```

Stored in a key-value database indexed by actor ID. Persists across documents and updates in-place. Conceptually, it is the actor's continuously-updated world-state representation — encoding dispositions, relationships, capabilities, and recent behavior.

## Initialization Strategies

**Critical:** Do not initialize from random. The initialization sets the geometric structure of the embedding space and dramatically accelerates training.

### Nation-States

```python
structural_features = concat([
    normalize(cinc_components),          # military, economic, demographic
    normalize(vdem_democracy_scores),    # regime type dimensions
    normalize(world_bank_indicators),    # GDP, trade, development
    voeten_ideal_points,                 # UNGA alignment dimensions
])
h_i(0) = Linear(structural_features, d)  # learned projection to d dims
```

### Leaders and Individuals

```python
text = concat(wikipedia_bio, major_speeches, party_manifestos)
token_embeddings = ConfliBERT(text)
h_i(0) = Linear(mean_pool(token_embeddings), d)
```

### Organizations (IGOs, Regional Orgs)

```python
text = concat(charter, founding_treaty, membership_list_descriptions)
h_i(0) = Linear(mean_pool(ConfliBERT(text)), d)
```

### Companies

```python
features = concat(
    mean_pool(ConfliBERT(sec_filings + news_descriptions)),
    sector_embeddings,
    market_cap_normalized,
    geographic_exposure_vector
)
h_i(0) = Linear(features, d)
```

## Embedding Space Properties

The geometry of the embedding space is the central representation. Training produces these properties (verified post-hoc, not specified by hand):

- **Proximity:** Actors with similar behavior patterns have nearby vectors
- **Interpretable dimensions:** Dimensions correspond to geopolitical axes (security alignment, economic orientation, ideological regime type, etc.)
- **Interaction encoding:** Dot product between actor vectors encodes expected interaction intensity
- **Asymmetry encoding:** Difference vector between actors encodes relationship asymmetry

## Sparse Actor Strategies

For actors with minimal data, the model relies on:

1. **Structural initialization** — Place the actor in the embedding space based on whatever structural characteristics are available (regime type, GDP, geographic region, alliance membership)
2. **Cross-actor similarity** — An actor's sparse event history resembles other actors' histories; transfer learning from similar actors fills gaps
3. **Graph propagation** — Information flows from well-observed neighbors through the temporal graph (Layer 4)

## Actor Sketch Vectors

In addition to the full d-dimensional memory vector, each actor maintains a lightweight sketch vector:

- Low-dimensional: 64-dim TF-IDF or small embedding projection
- Used by the text pre-filtering pipeline for fast actor relevance determination
- Updated cheaply whenever the full memory update runs
- Enables the sketch-based actor relevance filter (see `text_data_curation`)

## Open Research Questions

- **Optimal dimensionality:** NOMINATE works in 2D for legislatures (83% variance). What is the effective dimensionality of geopolitical actor space? Early BPTD experiments on ICEWS suggest 20–50 latent dimensions, but not systematically studied for the full actor set including non-state actors.
- **Memory half-life by actor type:** The exponential decay constant λ is assumed uniform. In reality, a small state's economic posture may be stable (long half-life) while security alignment after a coup changes overnight (short half-life). Learning adaptive half-lives per dimension per actor type is open.

## Memory Requirements

```
Actor memories: 5,000 actors × 512 floats × 4 bytes = ~10 MB (trivial)
```

## Architecture Reference

Corresponds to **Actor Taxonomy and Coverage** (Section 5) and **Layer 1: Actor Memory Store** (Section 6) in the architecture design document.
