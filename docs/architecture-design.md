# Geopolitical Event Prediction System: Architecture Design

*A comprehensive technical design document for a system that encodes the dispositions of geopolitical actors as high-dimensional time-varying vectors, and uses those vectors to predict the probability of future events.*

-----

## Table of Contents

1. [Motivation and Problem Framing](#1-motivation-and-problem-framing)
1. [System Overview](#2-system-overview)
1. [Data Sources](#3-data-sources)
1. [Event Ontology](#4-event-ontology)
1. [Actor Taxonomy and Coverage](#5-actor-taxonomy-and-coverage)
1. [Layer 1: Actor Memory Store](#6-layer-1-actor-memory-store)
1. [Layer 2: Text Processing Stream](#7-layer-2-text-processing-stream)
1. [Layer 3: Structured Event Stream](#8-layer-3-structured-event-stream)
1. [Layer 4: Actor Self-Attention](#9-layer-4-actor-self-attention)
1. [Layer 5: Event Prediction Head](#10-layer-5-event-prediction-head)
1. [Layer 6: Calibration](#11-layer-6-calibration)
1. [Gating Mechanisms](#12-gating-mechanisms)
1. [Training Strategy](#13-training-strategy)
1. [Negative Sampling](#14-negative-sampling)
1. [Loss Functions](#15-loss-functions)
1. [Evaluation and Benchmarking](#16-evaluation-and-benchmarking)
1. [Computational Budget](#17-computational-budget)
1. [Phased Build Plan](#18-phased-build-plan)
1. [Key Open-Source Dependencies](#19-key-open-source-dependencies)
1. [Open Research Questions](#20-open-research-questions)

-----

## 1. Motivation and Problem Framing

### 1.1 The Core Idea

This system aims to represent geopolitical actors — nation-states, international organizations, sub-state actors, companies, and influential individuals — as persistent high-dimensional vectors whose geometry encodes their dispositions, relationships, and alignments across many political, economic, and security dimensions simultaneously. These vectors evolve continuously as new information arrives, and their geometric relationships are used to predict the probability of future events.

The ultimate goals are threefold:

**Visualization and education.** Given arbitrary user-defined axes (e.g., “alignment with Western institutions” vs. “military assertiveness”), project actor vectors onto those axes and render an interactive map of the international system at any point in time.

**Scenario simulation.** Introduce a hypothetical event (e.g., a US-China trade deal, a coup in a given country), propagate its effects through the actor network, and observe how the predicted disposition vectors of all actors shift.

**Event probability prediction.** Given the current state of all actor vectors, estimate the probability that specific events occur within a defined time horizon — with accuracy competitive with or superior to prediction markets such as Polymarket.

### 1.2 Why This Is Harder Than Legislative Vote Prediction

The domestic legislative setting is almost ideally suited to ideal point models: votes are binary and formally recorded, actors vote repeatedly on many issues, the action space is constrained, and party discipline compresses most variance onto one or two dimensions.

Geopolitics breaks each of these properties:

- The action space is unbounded and heterogeneous. Countries do not just vote — they form alliances, impose sanctions, deploy troops, offer aid, conduct cyberattacks, and maintain conspicuous silences.
- State preferences are neither stable nor unitary. A country’s foreign policy position is the output of an internal coalition process and can shift dramatically with elections, leadership changes, or external shocks.
- The intrinsic dimensionality is genuinely high. Security alignment, economic interdependence, ideological regime type, historical grievance, resource geopolitics, and regional dynamics are all partially independent axes. A country like India is not reducible to a position on one or two dimensions.
- Strategic behavior is pervasive and sophisticated. The gap between stated position and actual preference is often the most important signal, and many actors cultivate deliberate ambiguity as a policy tool.
- The interaction structure is heterogeneous. What matters for US-China relations is largely orthogonal to what matters for India-Pakistan relations. There is no single shared “issue space” the way there is in a legislature.

### 1.3 The A↔B Dynamical System

The system formalizes geopolitics as a coupled dynamical system:

```
A(t)   = vector of actor dispositions at time t
B(t)   = set of events that occurred at time t

A(t+1) = f(A(t), B(t))    [events update dispositions]
P(B(t+1)) = g(A(t))       [dispositions generate event probabilities]
```

The functions `f` and `g` are what the model learns. The challenge is that `f` and `g` are both complex and the system has long-range dependencies — a disposition shift in 2014 is still generating events in 2024. Both functions are trained jointly, end-to-end.

### 1.4 Why This Approach Can Beat Prediction Markets

Prediction markets aggregate distributed human knowledge effectively but have structural weaknesses a model-based approach can exploit:

- **Attention asymmetry.** Markets are heavily weighted toward events salient to their user base. Lower-salience geopolitical events have thin depth and prices often anchored to base rates with minimal analysis.
- **Correlated events.** If event X and event Y are jointly caused by the same underlying disposition shift, a model can identify when the market has correctly priced X but not yet updated Y.
- **Update speed.** A model that continuously ingests diplomatic text, structured events, and economic signals can detect disposition shifts before they manifest as events that market participants notice.
- **Combinatorial coverage.** Prediction markets only cover specific, resolved questions. This system generates probability estimates across all actor-dyad-event-type combinations simultaneously.

-----

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT STREAMS                           │
│                                                                 │
│  Raw News Text          Structured Events     Static Structure  │
│  (Common Crawl,         (GDELT, POLECAT,      (COW, World Bank, │
│   scraped sources)       ICEWS archive)        SIPRI, Voeten)   │
└──────────┬──────────────────────┬──────────────────┬───────────┘
           │                      │                  │
           ▼                      ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│              LAYER 1: ACTOR MEMORY STORE                     │
│   h_i ∈ R^d for each actor i (nation, org, leader, company)  │
│   Initialized from structural features, evolves continuously  │
└──────────────────────────────┬───────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   │
┌─────────────────┐  ┌──────────────────┐          │
│  LAYER 2:       │  │  LAYER 3:        │          │
│  TEXT STREAM    │  │  STRUCTURED      │          │
│                 │  │  EVENT STREAM    │          │
│  - Sketch filter│  │                  │          │
│  - Sparse actor │  │  - Fast GRU      │          │
│    gating       │  │    update per    │          │
│  - ConfliBERT   │  │    coded event   │          │
│    encoding     │  │  - Goldstein     │          │
│  - Mention      │  │    score         │          │
│    extraction   │  │  - Event type    │          │
│  - Cross-actor  │  │    embedding     │          │
│    attention    │  │                  │          │
│  - Gated memory │  │                  │          │
│    update       │  │                  │          │
└────────┬────────┘  └────────┬─────────┘          │
         │                    │                    │
         └──────────┬─────────┘                    │
                    │                              │
                    ▼                              │
┌──────────────────────────────────────────────┐  │
│          LAYER 4: ACTOR SELF-ATTENTION       │◄─┘
│          (periodic, daily)                   │
│                                              │
│  Full multi-head self-attention across all   │
│  actors. Each actor attends to every other.  │
│  No explicit graph — topology emerges from   │
│  learned attention patterns.                 │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│         LAYER 5: EVENT PREDICTION HEAD       │
│                                              │
│  Query: P(event_r | actor_i, actor_j, τ)    │
│  Hawkes process for temporal prediction      │
│  Survival model for time-to-event           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│           LAYER 6: CALIBRATION               │
│  Temperature scaling per event type          │
│  Reliability diagram monitoring              │
└──────────────────────────────────────────────┘
```

-----

## 3. Data Sources

### 3.1 Event Data (Dynamic, High-Frequency)

#### GDELT (Global Database of Events, Language, and Tone)

The largest open event dataset. Produces roughly 100,000+ structured event records per day from 300+ event categories across 100+ languages from print, broadcast, and web news.

- **Access:** Free via Google BigQuery (1 TB/month free) or bulk CSV at `data.gdeltproject.org`
- **Python:** `gdeltPyR` library (`github.com/linwoodc3/gdeltPyR`)
- **Key fields:** Source actor, target actor, CAMEO event code, Goldstein scale score (–10 to +10), quad-class (verbal/material × cooperation/conflict), tone, article URL, date
- **Coverage:** 1979 to present, updated every 15 minutes
- **Known issues:** Opaque coding pipeline, significant event duplication (one real-world event generates many records from multiple articles), English/Western media bias, weekend periodicity artifacts, actor resolution errors

**Critical normalization:** Always normalize event counts against total article volume in the same time window. Raw event counts correlate with media attention cycles, not just real-world activity.

#### POLECAT (POLitical Event Classification, Attributes, and Types)

The active successor to ICEWS, produced using the PLOVER ontology and the Neural Global Event Coder (NGEC) — a transformer-based pipeline. Substantially higher precision than GDELT for structured events.

- **Access:** Free weekly downloads at Harvard Dataverse (`doi:10.7910/DVN/AJGVIT`)
- **Coverage:** 2018 to present
- **Advantage over GDELT:** Trained ML classifiers rather than dictionary pattern matching; Wikipedia-based entity linking for actor resolution; PLOVER ontology (cleaner than CAMEO)
- **License:** No restrictions on ML use

#### ICEWS Archive (1995–April 2023)

The predecessor to POLECAT. The standard benchmark dataset for temporal knowledge graph research — TComplEx, TNTComplEx, and RE-NET all evaluate on ICEWS14 and ICEWS05-15 subsets.

- **Access:** Harvard Dataverse (`doi:10.7910/DVN/28075`)
- **~270 million events** covering 1995–2023
- **License:** No restrictions on ML use

**Note on ACLED:** ACLED’s 2025 terms of service explicitly prohibit using its data to train, test, develop, or improve ML models, LLMs, or AI systems. Do not use ACLED for training.

### 3.2 Structural Data (Slow-Moving, High Signal)

#### UN General Assembly Voting Data

The cleanest continuous signal of state foreign-policy alignment. Provides both revealed preferences (votes) and stated preferences (speeches).

- **Voeten dataset:** Ideal point estimates from roll-call votes spanning sessions 1–78 (1946–2023), decomposed across six issue categories. Available at `unvotes.unige.ch`; R package `unvotes` provides tidy access to ~738,764 vote records
- **UN General Debate Corpus:** 7,300+ full-text country speeches from 1970–2016 (Baturo, Dasandi & Mikhaylov, 2017)
- **Raw UN votes:** `un.org/en/ga/` (accessible via API)

#### Correlates of War (COW) Project

Deep historical structural data. All datasets are free CSV downloads from `correlatesofwar.org`.

- **National Material Capabilities (CINC v6.0, 1816–2016):** Military expenditure, military personnel, energy consumption, iron/steel production, urban population, total population
- **Militarized Interstate Disputes (MID v5.0, 1816–2014):** Recorded military confrontations between states
- **Alliance data (v4.1, 1816–2012):** Formal defensive, offensive, and neutrality agreements
- **Limitation:** Most datasets end 2010–2016. The R package `peacesciencer` simplifies access

#### SIPRI Databases

- **Arms Transfer Database:** Trend Indicator Values for major conventional weapons transfers, 1950–2024, freely downloadable
- **Military Expenditure Database:** 171 countries since 1988

#### World Bank Open Data

1,400+ development indicators via free unauthenticated API (`api.worldbank.org/v2/`). Key indicators: GDP, GDP per capita, trade as % of GDP, FDI flows, population.

#### UN Comtrade / CEPII BACI

Bilateral trade flows by commodity. Comtrade: 500 free API calls/day. CEPII BACI reconciles Comtrade at HS 6-digit level (free for academic use, 1995–present).

### 3.3 Text Corpora (For Encoder Pretraining and Memory Updates)

#### Common Crawl News

Daily WARC files since 2016, 1,000+ news sources. The primary free source for large-scale news text.

- **Access:** `s3://commoncrawl/crawl-data/CC-NEWS/` (free S3 public dataset)
- **Volume:** Varies, typically 200,000–500,000 articles/day across all languages

#### Parliamentary Speech Corpora

- **ParlSpeech V2:** 6.3 million speeches from 9 democracies, 1987–2018 (Harvard Dataverse)
- **ParlaMint (CLARIN):** Additional EU parliaments in TEI XML
- **UK Hansard:** Digital since 1803, via TheyWorkForYou API
- **US Congressional Record:** Via `govinfo.gov` bulk data

#### Expert Survey Data

- **Chapel Hill Expert Survey (CHES):** 279 parties across 31 countries, expert-estimated ideological positions, free at `chesdata.eu`
- **Manifesto Project:** 5,285 election manifestos across 67 countries since 1945, with `manifestoberta` multilingual LLM (`manifestoproject.wzb.eu`)

-----

## 4. Event Ontology

### 4.1 CAMEO (Legacy, Still in GDELT)

~250 hierarchically arranged codes across 20 top-level categories. Maps to a Goldstein scale (–10 to +10). The quad-class aggregation (verbal/material × cooperation/conflict) is most commonly used in ML applications.

### 4.2 PLOVER (Current Standard)

18 event types with an event-mode-context scheme separating *what* happened from *how* (verbal, hypothetical, actual) and *why*. Adds magnitude fields (dead, injured, size). Uses JSON interchange format.

**The 18 PLOVER categories:**

|Code       |Category                       |Goldstein Range|
|-----------|-------------------------------|---------------|
|AID        |Provide aid                    |+3 to +8       |
|AGREE      |Make agreement                 |+4 to +8       |
|CONSULT    |Consult                        |+1 to +4       |
|COOP       |Cooperate                      |+2 to +6       |
|DEMAND     |Make demand                    |-2 to -5       |
|DISAPPROVE |Express disapproval            |-1 to -4       |
|ENGAGE     |Engage in diplomatic exchange  |+1 to +3       |
|FIGHT      |Use conventional military force|-8 to -10      |
|INVESTIGATE|Investigate                    |0 to -2        |
|MOBILIZE   |Mobilize/increase readiness    |-3 to -6       |
|PROTEST    |Protest/demonstrate            |-1 to -3       |
|REDUCE     |Reduce relations               |-2 to -5       |
|REJECT     |Reject                         |-2 to -4       |
|SANCTION   |Impose sanctions               |-4 to -8       |
|SEIZE      |Seize/arrest                   |-4 to -7       |
|THREATEN   |Threaten                       |-4 to -7       |
|YIELD      |Concede/yield                  |+2 to +5       |
|OTHER      |Other                          |varies         |

**Available at:** `github.com/openeventdata/PLOVER`

### 4.3 Relationship Between Event Type and Relation Embedding

Each of the 18 PLOVER event types gets a dense embedding vector `e_r ∈ R^d` in the same embedding space as actor vectors. This is *not* compressed into fewer dimensions — each relation type occupies its own point in the full d-dimensional space. Training discovers the geometry: event types that co-occur between similar actor pairs, or that tend to precede or follow each other, will cluster in the embedding space.

The abstract “relationship” between any two actors is then an implicit object — computed on demand as a function of their current memory vectors — rather than an explicit categorical label. This representation:

- Generalizes to actor pairs that have never directly interacted
- Is continuous, enabling similarity comparisons between relationships
- Decomposes naturally into interpretable dimensions that can be probed post-hoc
- Avoids the information loss inherent in discretizing relationships into categories

-----

## 5. Actor Taxonomy and Coverage

### 5.1 Actor Types

|Type                   |Examples                         |Data Quality                                     |Primary Sources                     |
|-----------------------|---------------------------------|-------------------------------------------------|------------------------------------|
|UN member states (193) |USA, China, Kenya                |Excellent for major powers; poor for small states|GDELT, POLECAT, COW, Voeten         |
|Major IGOs             |UN, NATO, EU, WTO, IMF           |Good                                             |GDELT, official documents           |
|Regional organizations |ASEAN, AU, SCO, MERCOSUR         |Moderate                                         |GDELT, treaty texts                 |
|Non-state armed groups |Hamas, Wagner Group, Hezbollah   |Sparse and biased                                |GDELT (noisy), conflict databases   |
|Multinational companies|Shell, Huawei, BlackRock         |Poor (no dedicated geo-political datasets)       |Financial news, SEC filings         |
|Individual leaders     |Heads of state, foreign ministers|Moderate                                         |UNGDC, ParlSpeech, GDELT GKG persons|
|Subnational entities   |Kurdistan, Taiwan, Hong Kong     |Sparse                                           |GDELT, specialized sources          |

### 5.2 Data Sparsity by Actor

Data density is deeply asymmetric:

- **G7 nations:** Dense coverage across all data sources; hundreds of thousands of events per year
- **Mid-tier states:** Adequate structured event coverage; limited text depth
- **Small states:** Primarily UN voting records and GDELT mentions; minimal parliamentary speech data
- **Non-state actors:** Highly variable; groups that receive media attention (ISIS, Wagner) have moderate coverage; others near zero
- **Companies and individuals:** No dedicated geopolitical event datasets; must build from financial news and social media

For sparse actors, the model relies more heavily on structural initialization (where in the embedding space does this actor belong given its structural characteristics?) and cross-actor similarity (this actor’s sparse event history resembles these other actors’ histories) rather than dense event-by-event updates.

-----

## 6. Layer 1: Actor Memory Store

### 6.1 Structure

Each actor `i` maintains a persistent memory vector:

```
h_i(t) ∈ R^d,   d = 256 or 512
```

This vector is stored in a key-value database indexed by actor ID. It persists across documents and updates in-place as new information arrives. Conceptually, it is the actor’s continuously-updated world-state representation — encoding everything the model has learned about that actor’s dispositions, relationships, capabilities, and recent behavior.

### 6.2 Initialization

Do not initialize from random. The initialization sets the geometric structure of the embedding space and dramatically accelerates training.

**For nation-states:**

```python
structural_features = concat([
    normalize(cinc_components),          # military, economic, demographic
    normalize(vdem_democracy_scores),    # regime type dimensions
    normalize(world_bank_indicators),    # GDP, trade, development
    voeten_ideal_points,                 # UNGA alignment dimensions
])
h_i(0) = Linear(structural_features, d)  # learned projection to d
```

**For leaders and individuals:**

```python
text = concat(wikipedia_bio, major_speeches, party_manifestos)
token_embeddings = ConfliBERT(text)
h_i(0) = Linear(mean_pool(token_embeddings), d)
```

**For organizations:**

```python
text = concat(charter, founding_treaty, membership_list_descriptions)
h_i(0) = Linear(mean_pool(ConfliBERT(text)), d)
```

**For companies:**

```python
features = concat(
    mean_pool(ConfliBERT(sec_filings + news_descriptions)),
    sector_embeddings,
    market_cap_normalized,
    geographic_exposure_vector
)
h_i(0) = Linear(features, d)
```

### 6.3 The Embedding Space

The geometry of the embedding space is the central representation. The model is trained so that:

- Actors with similar behavior patterns have nearby vectors
- Dimensions that emerge from training correspond to interpretable geopolitical axes (security alignment, economic orientation, ideological regime type, etc.)
- The dot product between actor vectors encodes expected interaction intensity
- The difference vector between actor vectors encodes the asymmetry of their relationship

These properties are not specified by hand — they emerge from the training objective. But they can be verified post-hoc by inspecting which actors cluster in which regions of the space, and which held-out relationships are correctly predicted.

-----

## 7. Layer 2: Text Processing Stream

This layer handles the primary information source: raw news articles. The key design decision is to update actor memory vectors *directly from text*, rather than first extracting structured events and then using those as inputs. This preserves richer signal — hedging, tone, causal framing, forward-looking language — that structured event extraction discards.

### 7.1 Pre-filtering Pipeline

Running the full text encoder on all 200,000–500,000 daily Common Crawl News articles is wasteful. Filter aggressively before encoding:

```
Raw stream:               ~500,000 articles/day
→ Language filter:        ~200,000  (keep English + major languages)
→ Geopolitical relevance: ~80,000   (TF-IDF cosine vs. keyword dictionary)
→ Deduplication:          ~40,000   (near-duplicate removal via MinHash)
→ Full encoding:          ~40,000 articles/day
```

The relevance filter is a lightweight bag-of-words TF-IDF cosine similarity against a geopolitical keyword dictionary — runs at millions of articles per minute on CPU, costs essentially nothing, and reduces encoding workload by 5–10x before any GPU compute is spent.

### 7.2 Sketch-Based Actor Relevance Filter

For each article that passes pre-filtering, determine which actors to update without running the full encoder first:

```python
# Compute lightweight sketch of article
article_sketch = tfidf_vectorizer.transform([article_text])  # sparse, cheap

# Compute relevance to each actor via dot product against actor sketch vectors
relevance_scores = article_sketch @ actor_sketch_matrix.T  # shape: [N_actors]

# Sparsemax: learned sparse selection of relevant actors
actor_weights = sparsemax(relevance_scores / temperature)

# Only proceed with nonzero-weight actors
active_actors = [i for i, w in enumerate(actor_weights) if w > 0]
# Typical result: 5–20 actors per article
```

The actor sketch vectors are low-dimensional (64-dim TF-IDF or small embedding) projections maintained alongside the full memory vectors. They are updated cheaply whenever the full memory update runs.

This sketch-based filtering achieves most of the benefit of soft attention (no hard entity resolution errors, captures indirect relevance through semantic similarity) while avoiding the noise of updating every actor with every article.

### 7.3 Full Document Encoding

For articles passing the sketch filter:

```python
# Encode full document
T = ConfliBERT(article_text)  # shape: [seq_len, d_model=768]

# For each active actor i, extract mention representation
# by pooling contextual embeddings at mention spans
m_i = mean_pool(T[mention_spans_of_actor_i])  # shape: [d_model]

# If no explicit mention spans (indirect relevance from sketch):
m_i = weighted_pool(T, attention_weights=actor_weights[i])  # soft version
```

**On entity resolution vs. soft attention:** Full hard entity resolution (identifying exact mention spans) is the most precise approach but requires a robust NER + disambiguation pipeline. Soft pooling weighted by the sketch-derived actor relevance weights is a reasonable alternative that sidesteps entity resolution errors at the cost of some precision. The two approaches can be combined: use hard entity resolution where confident, fall back to soft pooling for actors identified only by sketch similarity.

### 7.4 Cross-Actor Interaction Representation

An article about bilateral negotiations isn’t just updating two actors independently — the *relationship* between them in this document is the signal. Compute a cross-actor interaction representation:

```python
# For each pair (i, j) of co-active actors in this article:
# Cross-attention: each actor attends to the other's mention tokens
# and to the full document context

query_i = W_Q @ m_i           # actor i as query
key_j   = W_K @ T             # full document as key/value
val_j   = W_V @ T

c_ij = softmax(query_i @ key_j.T / sqrt(d_k)) @ val_j
# c_ij captures: what is happening between i and j in this document
```

This cross-actor representation feeds into the memory update for both actors, ensuring that what each actor “learns” from this document is conditioned on what other actors they’re interacting with and in what way.

### 7.5 Gated Memory Update

Update each active actor’s memory using a residual formulation with multi-scale gating:

```python
for i in active_actors:
    # Aggregate all relational context involving actor i
    relational_context = sum([c_ij for j in active_actors if j != i])

    # Combined input to update
    update_input = concat(h_i, m_i, relational_context, time2vec(t))

    # Scalar gate: how much should this article update actor i at all?
    gate_scalar = sparsemax(W_scalar @ update_input)  # competition across actors

    # Dimensional gate: which memory dimensions should update?
    gate_dims = sigmoid(W_dims @ update_input)        # independent per dimension

    # Candidate update (residual)
    delta_h = MLP(update_input) * gate_scalar

    # Temporal decay + gated residual update
    h_i = h_i * exp(-lambda_decay * delta_t) + gate_dims * delta_h
```

**Key design choices in this update:**

- **Residual rather than interpolation:** Rather than interpolating between old and new state (GRU-style), add a learned delta. This preserves direct gradient paths through the identity connection regardless of gate values.
- **Separate scalar and dimensional gates:** The scalar gate handles “is this article relevant to this actor at all?” The dimensional gate handles “which aspects of this actor’s disposition does this article speak to?” These serve different purposes and use different functional forms.
- **Exponential temporal decay:** Memory decays toward a per-actor baseline (computed from structural features and name encoding through learned shared projections) between updates, reflecting that old information becomes less reliable. The decay constant λ corresponds to a half-life of roughly 3–6 months for geopolitical dispositions.

-----

## 8. Layer 3: Structured Event Stream

Running in parallel with the text stream, the structured event stream provides faster, lower-latency updates from GDELT and POLECAT.

### 8.1 Event Representation

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

### 8.2 Fast GRU Update

```python
# Source actor update (active role)
h_source(t) = GRU_source(h_source(t⁻), concat(e_event, h_target(t⁻)))

# Target actor update (passive role)
h_target(t) = GRU_target(h_target(t⁻), concat(e_event, h_source(t⁻)))
```

The two GRUs share parameters but have different roles: the source actor is the agent initiating the event, the target is the recipient. Both actors’ memories update, but with different conditioning — each learns from the event as experienced from their respective position.

This structured stream is faster and cheaper than the text stream (no encoder to run) and provides a continuous update signal even when articles haven’t yet been processed. When the text stream later processes the articles about the same event, it provides a richer contextual update that may modify the initial structured-event update.

Both streams write to the same memory vectors. The gating mechanism handles fusion naturally: if a structured event and its corresponding articles update the same actor within a short window, the gate activations will naturally suppress redundant updates.

-----

## 9. Layer 4: Actor Self-Attention

Once per simulated day, propagate information across the full actor population via standard multi-head self-attention. This captures second-order effects that direct observation misses: if Germany and France have a major summit, the France-Algeria relationship is also affected even if no direct event was coded between them.

### 9.1 Design: Full Attention Over All Actors

Every actor attends to every other actor — no explicit graph construction, no edge types, no sparsity mask. The actor memory matrix H ∈ R^{N×d} is treated as a sequence and passed through a standard transformer block.

```python
# Each layer: standard transformer block
H_norm = LayerNorm(H)
H = H + MultiHeadAttention(H_norm, H_norm, H_norm)   # self-attention + residual
H = H + FFN(LayerNorm(H))                             # feed-forward + residual
```

**Why full attention instead of sparse graph edges:**
- **N is small.** With 500–2000 actors, full self-attention (N² attention entries) is trivial — <1ms on GPU. Article encoding (Layer 2) dominates compute by orders of magnitude.
- **Text-informed topology.** Actors discussed together in articles develop correlated memories. Attention discovers this without needing coded events to define edges.
- **Emergent typed interactions.** With multi-head attention, different heads specialize (cooperative vs. adversarial relationships) — the equivalent of typed edges, learned from data.

### 9.2 Multi-Layer Stacking

Stack 2–3 self-attention layers. Each layer allows information to propagate one additional hop through the actor population. With 2 layers, information flows from any actor to any other actor through an intermediate.

In practice 2 layers is sufficient for capturing the relevant international system structure without over-smoothing actor representations.

-----

## 10. Layer 5: Event Prediction Head

### 10.1 Dyadic Representation

For a query “what is the probability of event type `r` between actors `i` and `j` within `τ` days?”:

```python
# Construct dyadic feature vector
d_ij = concat([
    h_i,                    # source actor state
    h_j,                    # target actor state
    h_i * h_j,             # element-wise product: compatibility per dimension
    h_i - h_j,             # difference: asymmetry per dimension
    abs(h_i - h_j),        # absolute difference: distance per dimension
    time2vec(τ),            # forecast horizon encoding
    e_r,                    # relation type embedding (for multi-task head)
])
```

The element-wise product `h_i ⊙ h_j` captures symmetric compatibility. The difference `h_i - h_j` captures asymmetry: who is the more militarily capable actor, who is more Western-aligned, who is the economic dominant partner. Including both is critical because geopolitical relationships are neither purely symmetric nor purely differential.

### 10.2 Multi-Task Prediction Head

```python
# One MLP head per event type (multi-task)
scores = {}
for r in range(18):
    scores[r] = MLP_r(d_ij)  # shared trunk, relation-specific head

# Event probabilities
P_event = {r: sigmoid(scores[r]) for r in range(18)}

# Goldstein scale intensity (auxiliary regression task)
goldstein_pred = MLP_goldstein(d_ij)

# Escalation probability (binary: will any conflict-class event occur?)
P_escalation = sigmoid(MLP_escalation(d_ij))
```

The multi-task setup shares statistical strength across event types. Rare events (military attacks) benefit from representations learned on common events (verbal cooperation, diplomatic consultations).

### 10.3 Hawkes Process for Temporal Prediction

For predicting *when* rather than just *whether* an event occurs:

```python
def hawkes_intensity(h_i, h_j, event_history_ij, t):
    """
    Conditional intensity function: rate at which event type r occurs
    between actors i and j at time t, given history.
    """
    # Base rate from actor states
    mu = softplus(MLP_base(concat(h_i, h_j)))

    # Self-excitation: past events increase future rate
    # with exponential decay
    excitation = sum(
        exp(-beta * (t - t_k)) * w_type[r_k]
        for t_k, r_k in event_history_ij
        if t_k < t
    )

    # Total intensity
    lambda_r = mu + softplus(excitation)
    return lambda_r
```

The Hawkes process captures a fundamental empirical regularity in conflict data: events beget events. A military skirmish between two countries raises the short-term probability of further skirmishes. Diplomatic agreements cluster similarly. The exponential decay kernel means this excitation effect fades over time.

### 10.4 Survival Model for Time-to-Event Prediction

For questions like “how many days until the next sanctions event between actors i and j?”, frame as a survival analysis problem using DeepHit (Lee et al., AAAI 2018):

```python
# Discrete-time hazard model (DeepHit)
hazard_t = MLP_hazard(concat(d_ij, time2vec(t)))  # hazard at each time step
survival_t = cumprod(1 - hazard_t)                 # survival function
event_prob_by_t = 1 - survival_t                   # CDF
```

DeepHit handles competing risks naturally (coup vs. election vs. revolution), does not require proportional hazards assumptions, and works with censored observations (ongoing peace periods where we observe the actor surviving without conflict but do not know the eventual outcome time).

Implementation via the `pycox` library (`github.com/havakv/pycox`).

-----

## 11. Layer 6: Calibration

Neural networks trained on imbalanced data are systematically miscalibrated. A model that outputs P=0.8 for rare events should be correct 80% of the time — but in practice will usually be overconfident. Calibration corrects this.

### 11.1 Temperature Scaling

The simplest and most robust calibration method (Guo et al., ICML 2017):

```python
P_calibrated(r) = sigmoid(score_r / T_r)
```

Where `T_r` is a scalar learned separately per event type on a held-out calibration set. Apply after the full model is trained — temperature scaling does not change the ranking of predictions, only their absolute magnitudes. The `temperature_scaling` repository (`github.com/gpleiss/temperature_scaling`) provides a reference implementation.

**Why per-event-type:** Rare events (coups, wars) will be systematically miscalibrated differently from common events (verbal diplomatic exchanges). A single global temperature is insufficient.

### 11.2 Monitoring

Produce reliability diagrams at each evaluation checkpoint: bin predictions by predicted probability, plot mean predicted probability vs. observed frequency. A perfectly calibrated model lies on the diagonal. Compute Expected Calibration Error (ECE) as a scalar summary.

```python
def ece(y_pred, y_true, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
        if mask.sum() > 0:
            bin_conf = y_pred[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += mask.mean() * abs(bin_conf - bin_acc)
    return ece
```

-----

## 12. Gating Mechanisms

The gate controlling memory updates is one of the most important design choices. It functions analogously to a ReLU or sigmoid — a learned thresholding function — but the choice of functional form affects what the model learns, how sparse the updates are, and whether the computation can be efficiently skipped for zero-weight actors.

### 12.1 The Two-Scale Gating Problem

Two distinct questions require different functional forms:

**Scalar gate (relevance):** Should this document update actor i at all? This is a competition across actors — we want most actors to receive zero weight from most documents.

**Dimensional gate (specificity):** Which dimensions of actor i’s memory vector should update? This is independent per dimension — a document can simultaneously be relevant to the security axis and the economic axis without these competing.

### 12.2 Sigmoid (Baseline)

```
z = σ(Wx + b),   output ∈ (0, 1) per dimension
```

The standard choice. Each actor’s relevance is computed independently, meaning all actors can simultaneously receive high weight. Appropriate for the dimensional gate (no reason for dimensions to compete), but too permissive for the scalar gate (leads to noisy updates across many actors).

### 12.3 Sparsemax (Recommended for Scalar Gate)

Projects scores onto the probability simplex with exact zeros for low-scoring inputs:

```
sparsemax(z) = argmin_{p ∈ Δ} ||p - z||²
```

Unlike softmax which produces small positive values everywhere, sparsemax produces **exactly zero** for actors below a dynamically computed threshold. The sparsity pattern is learned — no manual threshold required. Differentiable everywhere except at the sparsity boundary.

**α-entmax generalization:**

```
α = 1.0  → softmax (dense)
α = 1.5  → partial sparsity
α = 2.0  → sparsemax (exactly sparse)
α → ∞   → argmax (one-hot)
```

α can be treated as a learnable hyperparameter or made input-dependent: documents with narrow focus use high α (sparse updates), multilateral event coverage uses low α (diffuse updates). This is a meaningful inductive bias that can be learned from the data.

### 12.4 Gumbel-Sigmoid (Hard Decisions with Gradient Flow)

Enables hard binary decisions during inference while maintaining differentiable training:

```python
def gumbel_sigmoid(logits, temperature=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)

    if hard:
        # Hard in forward pass (binary), soft in backward pass (gradient)
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft  # straight-through estimator
    return y_soft
```

During training: high temperature → smooth sigmoid → clean gradients. Anneal temperature toward 0 over training. At inference: temperature near 0 → near-binary output. Set hard threshold and skip update computation entirely for zero-gate actors. This is the only approach that enables true computational savings at inference time.

### 12.5 L0 Regularization (Hard Concrete Gate)

Directly penalizes the number of active gates using the hard concrete distribution:

```python
def hard_concrete_gate(log_alpha, beta=0.667, gamma=-0.1, zeta=1.1):
    u = torch.zeros_like(log_alpha).uniform_().clamp(1e-8, 1-1e-8)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    z = s_bar.clamp(0, 1)
    return z

# L0 regularization term
L0 = sum(sigmoid(log_alpha_i - beta * log(-gamma / zeta))
         for i in all_actors)

# Total loss
L_total = L_prediction + lambda_L0 * L0
```

Tune `lambda_L0` to control average number of actors updated per document. This makes the sparsity level a directly tunable hyperparameter rather than an emergent property of the threshold.

### 12.6 Recommended Configuration

```python
# Scalar gate: sparsemax across actors (learned sparse selection)
actor_weights = entmax(relevance_scores, alpha=1.5)

# Dimensional gate: sigmoid per dimension (independent, no competition)
gate_dims = sigmoid(W_dims @ update_input)

# Update
delta_h = MLP(update_input)
h_i = h_i * exp(-lambda * dt) + gate_dims * (actor_weights[i] * delta_h)
```

-----

## 13. Training Strategy

### 13.1 Phase 1: Structural Pretraining

Initialize actor embeddings from structural features (Section 6.2). No text or event data yet. Run a few gradient steps to ensure the initialization is well-conditioned and the projection matrices are well-scaled. This phase is fast (minutes, not hours).

### 13.2 Phase 2: Self-Supervised Text Pretraining

Pretrain the text encoder and memory update mechanism before introducing supervised event labels. Two objectives:

**Masked entity prediction:** Mask actor name spans in documents. Predict the masked actor from the surrounding document context and the current memory states of other actors mentioned in the document. This forces the memory vectors to encode geopolitically meaningful representations — if you mask “Russia” and the model must identify it from context (“military exercises near Ukrainian border…”), it must encode that such events are associated with a specific cluster of actors.

```python
# Mask actor spans and predict them
L_entity = CrossEntropy(
    actor_classifier(T[mask_positions]),  # predict from context
    true_actor_ids
)
```

**Temporal ordering:** Given two paragraphs from different time points, predict which came first. This teaches the model about causal sequencing and event precedence.

```python
# Contrastive temporal ordering
L_temporal = BCE(
    temporal_classifier(mean_pool(T1), mean_pool(T2)),
    (t1 < t2).float()
)
```

**Contrastive document similarity:** Documents describing the same event or the same actors in the same context should have similar representations. Use GDELT event codes as weak supervision for pairing: articles that generate the same structured event code on the same day are positive pairs.

### 13.3 Phase 3: Supervised Event Prediction Fine-Tuning

Fine-tune end-to-end on the event prediction task. Training signal flows from prediction errors all the way back into the encoder weights.

**Curriculum learning:** Start with high-frequency event types (diplomatic consultations, verbal cooperation) and gradually introduce rare events (military attacks, sanctions). Prevents early training from being dominated by trivially easy examples.

```python
# Linear curriculum pacing
def get_event_weight(event_type, training_step, total_steps):
    event_freq = base_frequencies[event_type]
    rarity = 1.0 / event_freq
    curriculum_factor = min(1.0, training_step / (curriculum_warmup * total_steps))
    return 1.0 + (rarity - 1.0) * curriculum_factor
```

**Multi-task training:** Predict all 18 event types simultaneously plus auxiliary tasks (Goldstein score regression, escalation binary classification, actor-type classification from memory vectors). Rare event types benefit from shared representations learned on common events.

### 13.4 Handling Long-Range Temporal Dependencies

The key difficulty in training a dynamical model is that gradients must flow back through many steps of memory updates. Two mitigations:

**Truncated backpropagation through time (TBPTT):** Only backpropagate through the last K memory update steps (K=50–100). This limits gradient path length while still allowing the model to learn from recent history.

**Auxiliary losses at intermediate timesteps:** Add event prediction losses not just at the final timestep but also at intermediate checkpoints during training rollouts. This creates shorter gradient paths and prevents vanishing gradients.

-----

## 14. Negative Sampling

### 14.1 The Problem

The space of possible events is combinatorially enormous: N_actors² × 18 event types × time steps. The vast majority of these tuples are never observed. Training cannot enumerate all non-events.

This is the open-world assumption (OWA) problem: unobserved triples are *unknown* (either false or simply unrecorded), not *false* by assumption. GDELT and POLECAT only observe what media reports. A diplomatic meeting that received no English-language coverage is unobserved but not a non-event.

### 14.2 The Word2Vec Connection

The solution follows Word2Vec’s negative sampling directly. For each observed event (source, relation, target, t), contrast it against a small number of randomly sampled non-events. Train the model to score observed events higher than sampled noise via binary cross-entropy:

```
L_NS = -log σ(score(s,r,o,t)) - Σₖ Eₙ~P(n)[log σ(-score(s,r,n,t))]
```

Where k=5–20 negative samples per positive, and P(n) is the noise distribution over actors.

The noise distribution should be the marginal actor frequency distribution raised to the 3/4 power (exactly as in Word2Vec), computed within a rolling time window rather than globally to account for the non-stationarity of geopolitical event frequency.

### 14.3 Hard Negative Mining

Random negatives become trivially easy as the model improves. USA-sanctions-Tuvalu is an obvious negative; USA-sanctions-India is a meaningful hard negative that reflects a real geopolitical counterfactual.

```python
def mixed_negatives(observed_event, model, ratio_hard=0.15):
    """
    Sample negatives: ~85% random, ~15% hard (model-predicted plausible non-events).
    """
    s, r, o, t = observed_event
    n_negatives = 10

    # Random negatives (corrupt one element)
    random_negs = [corrupt(s, r, o, t, noise_distribution) 
                   for _ in range(int(n_negatives * (1 - ratio_hard)))]

    # Hard negatives: sample from model's top predictions that didn't occur
    scores = model.score_all_targets(s, r, t)  # score all possible targets
    scores[o] = -inf  # exclude the true event
    hard_targets = topk(scores, int(n_negatives * ratio_hard))
    hard_negs = [(s, r, t_hard, t) for t_hard in hard_targets]

    return random_negs + hard_negs
```

Use pure hard negatives only sparingly — they can cause mode collapse (the model chases its own tail). The 85/15 random/hard split is empirically stable across dense retrieval literature.

### 14.4 Structural Feasibility Filter

Some events are structural zeros — essentially impossible given the actors’ relationship structure — rather than merely unobserved. USA-military-attack-Canada is a structural zero; USA-military-attack-Russia is a true negative worth training on.

Build a feasibility mask from structural data:

```python
def feasible_pair(actor_i, actor_j, event_type):
    """
    Return True if this event is plausible given structural constraints.
    """
    if event_type in MILITARY_EVENTS:
        return (trade_volume(i, j) > threshold or
                conflict_history(i, j) > 0 or
                geographic_proximity(i, j) < max_distance)
    if event_type in ECONOMIC_EVENTS:
        return trade_volume(i, j) > 0
    return True  # diplomatic events are always feasible
```

Only generate negatives from the feasible set. This makes the discrimination problem well-posed: the model learns to distinguish plausible-but-didn’t-happen from actually-happened, which is the meaningful geopolitical question.

### 14.5 Poisson Count Modeling for Common Events

For high-frequency event types (verbal cooperation, diplomatic consultations), model the *count* per dyad per time period rather than binary occurrence:

```python
# Negative Binomial count model (handles overdispersion)
count_pred = NB(mu=exp(score(s, r, o, t)), alpha=dispersion_param)
L_count = -NB.log_prob(observed_count)
```

This naturally handles zero-inflation (zeros are expected Poisson/NB outcomes with low rate, not missing data) and avoids the awkward binary framing for events that occur multiple times per period.

-----

## 15. Loss Functions

### 15.1 Primary: Binary Cross-Entropy with Focal Weighting

```
L_BCE = -[y · log(p) + (1-y) · log(1-p)]

L_focal = -α_t · (1 - p_t)^γ · log(p_t)
```

Where γ=2 is standard and α_t is the inverse-frequency class weight. Focal loss down-weights easy negatives (common non-events the model gets right with high confidence) and focuses learning on hard examples.

### 15.2 Auxiliary: Goldstein Scale Regression

```
L_goldstein = MSE(goldstein_pred, goldstein_observed)
```

Predicting event intensity as a continuous value regularizes the representation and ensures the model learns an ordering over events, not just their occurrence.

### 15.3 Survival/Hazard Loss (DeepHit)

For time-to-event predictions:

```
L_DeepHit = L_NLL + η · L_ranking

L_NLL = -log(P(T = t_i | X_i))    # negative log-likelihood on event time
L_ranking = E[max(0, σ - (S_i(t) - S_j(t)))]  # concordance ranking loss
```

Where S_i(t) is the survival function for subject i, and the ranking loss ensures the model correctly orders subjects by their risk.

### 15.4 Memory Regularization

Prevent memory vectors from exploding or collapsing:

```
L_mem_reg = λ_reg · Σᵢ ||h_i||²     # L2 regularization on memory norms
```

### 15.5 Total Loss

```
L_total = L_focal
        + λ₁ · L_goldstein
        + λ₂ · L_DeepHit
        + λ₃ · L_mem_reg
        + λ₄ · L0_gate_penalty
```

Tune λ hyperparameters via grid search on held-out validation Brier score.

### 15.6 Temporal Discounting

Weight more recent events more heavily in the training loss:

```
w(t) = exp(-λ_discount · (T_current - t))
```

With λ_discount corresponding to a half-life of roughly 6–12 months. Events from 5 years ago are less representative of current system dynamics than events from last quarter.

-----

## 16. Evaluation and Benchmarking

### 16.1 Primary Metrics

**Brier Score (BS):**

```
BS = (1/N) · Σᵢ (p_i - y_i)²
```

The mean squared error of probabilistic forecasts. Lower is better, range [0, 1]. Strictly proper scoring rule.

**Brier Skill Score (BSS):**

```
BSS = 1 - BS_model / BS_climatology
```

Measures improvement over the climatological base rate (always predicting the historical event frequency). Positive BSS means the model beats the naive baseline.

**Log Loss:**

```
L = -(1/N) · Σᵢ [y_i · log(p_i) + (1-y_i) · log(1-p_i)]
```

More sensitive than Brier to confident wrong predictions. Use as training objective; report alongside Brier at evaluation.

**PR-AUC (not ROC-AUC):** For rare events, ROC-AUC is misleading (the large number of true negatives inflates it). Precision-Recall AUC is the appropriate metric for imbalanced problems.

**Expected Calibration Error (ECE):**

```python
ECE = Σ_b (|B_b| / N) · |acc(B_b) - conf(B_b)|
```

Measures how well predicted probabilities match empirical frequencies.

### 16.2 Benchmark Comparisons

**ViEWS system:** The primary benchmark. Open forecasts and evaluation code at `github.com/views-platform`. The ViEWS 2023/24 competition provides standardized country-month conflict fatality predictions against which to compare.

**Polymarket comparison methodology:**

1. For each Polymarket geopolitical question, identify the relevant actors and event type in your system
1. Query your model’s probability at the same time Polymarket opened
1. Compute head-to-head Brier scores over a held-out set of resolved questions
1. Control for question difficulty (Polymarket questions may be non-randomly selected)

**Superforecaster baseline:** Good Judgment Open provides aggregate human forecasts. Superforecasters achieve mean Brier ≈ 0.14 on geopolitical questions. This is the target for a competitive system.

**No-change baseline:** Always predict that the current situation continues. This deceptively strong baseline beats many conflict forecasting models on escalation prediction. Any proposed model must beat this baseline to be considered meaningful.

### 16.3 What the ViEWS Competitions Taught Us

The ViEWS 2020/21 competition (15 teams) found that most sophisticated ML models were “surprised by conflict outbreak in previously peaceful locations” and beaten by a basic no-change model on escalation. The unweighted ensemble of all submitted models was highly competitive, suggesting model diversity matters more than individual model sophistication.

The 2023/24 competition found that simple tree-based models (XGBoost, Random Forest) with carefully engineered features often outperform complex neural approaches. Feature engineering quality dominates architectural choices. The key implication: **build the LightGBM baseline first and ensure it achieves positive BSS before investing in neural architecture.**

-----

## 17. Computational Budget

### 17.1 Daily Encoding Cost

Assuming 40,000 articles/day after pre-filtering (from ~500,000 raw):

```
40,000 articles × 400 tokens/article = 16,000,000 tokens/day
```

|Hardware        |Throughput (tokens/sec)|Time for 16M tokens|Cost/day (spot) |
|----------------|-----------------------|-------------------|----------------|
|T4 (Colab/GCP)  |~8,000–12,000          |~25–35 min         |~$0.15–0.25     |
|A10G (AWS g5)   |~25,000–35,000         |~8–12 min          |~$0.20–0.30     |
|A100 (80GB)     |~60,000–80,000         |~3–5 min           |~$0.50–0.80     |
|RTX 4090 (owned)|~40,000–55,000         |~5–7 min           |electricity only|

### 17.2 Total Monthly Budget Estimate (Hobby)

```
Daily encoding (40K articles, T4 spot, ~30 min):       ~$0.20/day  → ~$6/month
Actor self-attention (daily batch, T4, <1 min):          ~$0.01/day  → ~$0.30/month
Training runs (weekly, A10G spot, 4 hours):             ~$2.00/week → ~$8/month
Storage (GDELT history, embeddings, ~500GB S3):         ~$12/month
Total:                                                  ~$28/month
```

This is achievable on a hobby budget. The bottleneck is engineering and research quality, not compute.

### 17.3 Model Size Tradeoffs

**ConfliBERT (BERT-base, 12L, 110M params):** Baseline. Good domain adaptation for geopolitical text.

**DistilConfliBERT (6L, ~66M params):** ~1.7x faster, minimal accuracy loss. Achievable via knowledge distillation from ConfliBERT using HuggingFace’s distillation toolkit.

**Custom 4-layer domain model (4L, ~30M params):** ~2.5x faster than ConfliBERT. Train from scratch or from BERT-small initialization. Fine-tune on geopolitical text with masked entity prediction + temporal ordering objectives. For this narrow domain, a smaller model trained specifically on relevant corpora may outperform a larger general model.

**Recommended:** Start with ConfliBERT to establish baselines. Distill once the architecture is stable.

### 17.4 Memory Requirements

```
Actor memories: 5,000 actors × 512 floats × 4 bytes = ~10 MB (trivial)
ConfliBERT inference: ~2 GB VRAM (batch size 32)
Actor self-attention (1000 actors, 2 layers): ~50 MB VRAM
Total VRAM for inference: ~4-6 GB
Training (with gradients + optimizer states): ~16-20 GB
→ Fits on RTX 3090/4090 (24 GB) or A10G (24 GB)
```

-----

## 18. Phased Build Plan

### Phase 1: Tabular Baseline (Months 1–2, CPU only)

Build a country-month panel combining:

- GDELT event counts by CAMEO quad-class (4 features per dyad)
- Lagged conflict indicators from UCDP
- Structural features: Polity score, GDP, population, Voeten ideal point distances
- Recent Voeten UNGA voting record similarity

Train LightGBM classifiers per event type. Evaluate with Brier score, BSS, reliability diagrams.

**Expected outcome:** Positive BSS over climatological baseline. This is the bar every subsequent component must improve upon.

**Tools:** `gdeltPyR`, `peacesciencer`, World Bank API, `lightgbm`, `sklearn`

### Phase 2: Neural Embeddings (Months 3–5, single GPU)

Add ConfliBERT-based text feature extraction. Implement actor self-attention propagation (standard transformer blocks over the actor population). Multi-task prediction heads for all 18 event types. Apply focal loss + class weights. Post-hoc temperature scaling.

**Expected outcome:** Improvement over Phase 1 baseline, especially for events with strong textual precursors.

**Tools:** `transformers` (ConfliBERT), `pycox` (survival analysis), `mordecai3` (geoparsing)

### Phase 3: Full Architecture (Months 6–12, single GPU)

Implement TGN-style per-actor memory with gated text updates (Layer 2). Add Neural ODE continuous dynamics via `torchdyn`. Add structured event fast-update stream (Layer 3). Full actor self-attention propagation (Layer 4). Transformer Hawkes Process for temporal event density prediction. Ensemble across model families. Real-time GDELT/POLECAT ingestion pipeline. Benchmarking against Polymarket and ViEWS. Interactive visualization dashboard.

**Expected outcome:** Competitive with or superior to Polymarket on low-salience geopolitical events. Publishable architecture contribution (entity memory + actor self-attention + Hawkes process coupling).

-----

## 19. Key Open-Source Dependencies

|Component                 |Repository                                                 |Purpose                                              |
|--------------------------|-----------------------------------------------------------|-----------------------------------------------------|
|PyTorch Geometric Temporal|`github.com/benedekrozemberczki/pytorch_geometric_temporal`|Reference for dynamic GNN baselines (comparison only) |
|RE-NET                    |`github.com/INK-USC/RE-Net`                                |Autoregressive temporal KG event prediction          |
|TComplEx/TNTComplEx       |`github.com/facebookresearch/tkbc`                         |Static+temporal KG embedding baselines               |
|Transformer Hawkes Process|`github.com/SimiaoZuo/Transformer-Hawkes-Process`          |Neural temporal point process                        |
|Latent ODE                |`github.com/YuliaRubanova/latent_ode`                      |Continuous-time actor dynamics                       |
|BPTD                      |`github.com/aschein/bptd`                                  |Bayesian Poisson Tucker decomposition (interpretable)|
|ConfliBERT                |`github.com/eventdata/ConfliBERT`                          |Domain-adapted BERT for conflict text                |
|Mordecai 3                |`github.com/ahalterman/mordecai3`                          |Neural geoparsing / toponym resolution               |
|PETRARCH3                 |`github.com/oudalab/petrarch3`                             |Rule-based event coder (PLOVER-compatible)           |
|PLOVER                    |`github.com/openeventdata/PLOVER`                          |Event ontology specification                         |
|pycox                     |`github.com/havakv/pycox`                                  |Survival analysis (DeepHit, Cox-Time)                |
|ViEWS pipeline            |`github.com/views-platform`                                |Conflict prediction MLOps, evaluation infrastructure |
|gdeltPyR                  |`github.com/linwoodc3/gdeltPyR`                            |GDELT data retrieval                                 |
|temperature_scaling       |`github.com/gpleiss/temperature_scaling`                   |Post-hoc calibration                                 |
|torchdyn                  |`github.com/DiffEqML/torchdyn`                             |Neural ODE implementations                           |
|peacesciencer             |CRAN / `github.com/svmiller/peacesciencer`                 |COW + V-Dem data access                              |
|sparsemax                 |`github.com/KrisKorrel/sparsemax-pytorch`                  |Sparsemax / entmax implementation                    |

-----

## 20. Open Research Questions

Several aspects of this design remain genuinely open and represent opportunities for novel contribution:

**The OWA problem for geopolitics.** The combination of structural feasibility filtering, Poisson count modeling, and hard negative mining addresses the open-world assumption but does not fully solve it. A principled probabilistic treatment of the difference between “unobserved because media didn’t cover it” and “unobserved because it didn’t happen” is an open problem.

**Optimal dimensionality.** NOMINATE works in 2D and captures 83% of vote variance in legislatures. What is the effective dimensionality of geopolitical actor space? Early experiments with the BPTD framework on ICEWS data suggest 20–50 latent dimensions capture most of the variance, but this has not been systematically studied for the full actor set including non-state actors and leaders.

**Memory half-life by actor type.** The exponential decay constant λ is assumed uniform across actors and dimensions. In reality, a small state’s economic posture may be very stable (long half-life) while its security alignment after a coup may change overnight (short half-life). Learning adaptive half-lives per dimension per actor type is an open architectural question.

**Cross-lingual signal.** GDELT and Common Crawl News cover 100+ languages, but the encoder stack (ConfliBERT) is primarily English-trained. Events reported only in Arabic, Chinese, or Russian language media — which often carry different perspectives on the same events — are underrepresented. Multilingual pretraining or cross-lingual alignment is needed to fully exploit these sources.

**Strategic communication modeling.** The model takes text at face value. A country that deliberately issues misleading statements will corrupt its own memory representation. Adversarial training or explicit modeling of strategic information environments (who is more or less credible on which topics) would be a meaningful extension.

**Composing actor-level predictions to question-level probabilities.** Prediction markets ask specific, resolved questions: “Will Russia and Ukraine sign a ceasefire before December 31, 2025?” The model predicts probabilities over abstract event types between actor pairs. The mapping from the latter to the former requires a semantic layer — ideally an LLM that parses the question, identifies relevant actors and event types, and aggregates the model’s dyadic probability estimates into a question-specific probability. This interface layer is understudied and practically important for the Polymarket comparison goal.

-----

*Document reflects design discussions through March 2026. Architecture is at the proposal stage; no implementation exists yet.*