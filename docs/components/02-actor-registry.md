# Component 2: Actor Registry & Initialization

## Purpose

Define the canonical set of actors the model tracks, assign stable IDs, collect and normalize structural features for each actor, and compute initial embedding vectors that bootstrap the Actor Memory Store (Layer 1) before any training data is processed.

**Inputs:** Structural/static data from Component 1 (COW, V-Dem, Voeten, World Bank, SIPRI). Text corpora for non-state actor initialization.

**Outputs:** An actor registry table and a set of initial embedding vectors `h_i(0) ∈ R^d` for every actor.

---

## 1. Actor Set Definition

### 1.1 Actor Types

| Type | Count (approximate) | Inclusion Criteria |
|------|---------------------|--------------------|
| UN member states | 193 | All current members. Always included. |
| Major IGOs | ~15 | UN, NATO, EU, WTO, IMF, World Bank, IAEA, WHO, ICC, OSCE, G7, G20, BRICS, SCO, OECD |
| Regional organizations | ~20 | ASEAN, AU, ECOWAS, MERCOSUR, GCC, SADC, EAC, CARICOM, Pacific Islands Forum, etc. Include if they appear in ≥100 POLECAT events/year. |
| Non-state armed groups | ~30–50 | Include groups that appear in ≥50 POLECAT events total. Examples: Hamas, Hezbollah, Wagner Group, ISIS, Taliban, Houthis. Review and update annually. |
| Multinational companies | ~20–50 | Include companies that appear in ≥100 geopolitically-relevant news articles/year. Examples: major defense contractors, sanctioned entities, energy majors. Phase 2+ only. |
| Individual leaders | ~50–100 | Current heads of state and foreign ministers for top-50 countries by event volume. Rotate as leadership changes. Phase 2+ only. |
| Subnational entities | ~10–20 | Only where they are major actors in international politics: Taiwan, Kurdistan, Hong Kong, Crimea, etc. |

**Total actor count:** ~350–500 in Phase 1 (states + major IGOs + regional orgs), scaling to ~500–800 in Phase 2+.

### 1.2 Actor Registry Schema

| Field | Type | Description |
|-------|------|-------------|
| `actor_id` | `string` | Canonical unique identifier. Format: `{type_prefix}:{code}`. Examples: `state:USA`, `igo:NATO`, `nsag:HAMAS`, `corp:HUAWEI`, `leader:USA_POTUS` |
| `actor_type` | `enum` | `state` / `igo` / `regional_org` / `nsag` / `company` / `leader` / `subnational` |
| `name` | `string` | Primary display name |
| `aliases` | `list[string]` | Alternative names, abbreviations, transliterations |
| `iso_alpha3` | `string` | ISO 3166-1 alpha-3 code (states only, null otherwise) |
| `parent_actor_id` | `string` | For leaders: the state they lead. For subnational: the containing state. Null otherwise. |
| `active_from` | `date` | Date this actor enters the system (e.g., country independence date, org founding date) |
| `active_to` | `date` | Null if still active. Set when actor ceases to exist (e.g., leadership change for individual leaders) |
| `source_code_mappings` | `dict` | Maps from source-specific codes to this actor. Keys: `gdelt_codes`, `polecat_codes`, `icews_codes`. Values: lists of source-native codes that map to this actor. |

### 1.3 Source Code Mapping

Each data source uses its own actor coding. The registry maintains explicit mappings:

**GDELT → canonical:** GDELT uses 3-letter CAMEO country codes plus role suffixes (`USAGOV` = US Government, `USAMIL` = US Military, `USAOPN` = US Opposition). Map these:
- `USA*` → `state:USA` (collapse role suffixes for state-level modeling)
- Or optionally: `USAGOV` → `leader:USA_POTUS`, `USAMIL` → `state:USA` (role-aware mapping for Phase 2+)

**POLECAT → canonical:** POLECAT uses ISO alpha-3 for states and Wikipedia entity IDs for non-state actors. Mapping is more straightforward.

**ICEWS → canonical:** Similar to GDELT CAMEO codes. Use the same mapping table with ICEWS-specific extensions.

Unmapped codes are logged to a review queue. Actors that accumulate ≥50 unmapped event references are flagged for inclusion in the registry.

---

## 2. Structural Feature Collection

### 2.1 Feature Vector for Nation-States

For each state and each year, assemble the structural feature vector from Component 1's structural data output:

```python
structural_features_state = [
    # Military capability (from COW CINC + SIPRI)
    cinc_score,                # composite [0, 1]
    log_military_personnel,    # log-normalized
    log_military_expenditure,  # log-normalized, constant USD
    log_arms_imports_tiv,      # SIPRI TIV
    log_arms_exports_tiv,      # SIPRI TIV

    # Economic (from World Bank)
    log_gdp,                   # log-normalized, constant USD
    log_gdp_per_capita,        # log-normalized
    trade_pct_gdp,             # trade openness
    log_fdi_net_inflows,       # log-normalized

    # Demographic (from World Bank + COW)
    log_population,            # log-normalized
    urban_pct,                 # urbanization rate

    # Governance (from V-Dem)
    electoral_democracy,       # [0, 1]
    liberal_democracy,         # [0, 1]
    participatory_democracy,   # [0, 1]
    rule_of_law,               # [0, 1]
    corruption,                # [0, 1]

    # Alignment (from Voeten)
    voeten_ideal_dim1,         # continuous
    voeten_ideal_dim2,         # continuous
]
# Total: ~18 features per state per year
```

**Normalization:** All features are z-score normalized using statistics computed over the full historical panel (all states, all years). Log transformations are applied before z-scoring for heavy-tailed distributions (GDP, population, military expenditure).

**Missing value handling:**
- Linear interpolation for single-year gaps.
- Forward-fill for end-of-coverage gaps (e.g., COW ending 2016 → forward-fill to present).
- Explicit missingness indicator: for each feature, add a binary `{feature}_missing` flag. The initialization projection can learn to handle missingness.
- For actors with no data at all for a feature block (e.g., non-state actors have no CINC score), fill with the median value for that actor type and set the missingness flag.

### 2.2 Feature Vector for Non-State Actors

Non-state actors (IGOs, armed groups, companies) lack most structural datasets. Their initialization relies more heavily on text:

**For IGOs and regional organizations:**
```python
structural_features_org = [
    member_count,                    # number of member states
    mean_member_gdp,                 # average GDP of member states
    mean_member_democracy,           # average democracy score of members
    founding_year_normalized,        # how old the organization is
    # + text-derived features (see Section 3.2)
]
```

**For non-state armed groups:**
```python
structural_features_nsag = [
    event_count_last_5y,             # total POLECAT events involving this group
    mean_goldstein_as_source,        # average Goldstein score when acting as source
    mean_goldstein_as_target,        # average Goldstein score when acting as target
    geographic_region_onehot,        # 7-dim: Africa, Asia, Europe, MENA, Americas, Oceania, Global
    # + text-derived features
]
```

**For companies and leaders:** Primarily text-derived (see Section 3).

---

## 3. Initial Embedding Computation

### 3.1 Initialization Strategy Overview

The initial embedding `h_i(0)` for each actor is computed *before* training begins. It sets the starting geometry of the embedding space. Good initialization dramatically accelerates training because actors start in approximately the right neighborhoods.

Three initialization pathways depending on actor type and data availability:

```
                    ┌──────────────────────┐
                    │ Structural features  │
                    │ available?           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
              yes   │                      │  no
          ┌─────────┤                      ├─────────┐
          │         └──────────────────────┘         │
          ▼                                          ▼
┌─────────────────┐                       ┌─────────────────┐
│ Path A:         │                       │ Text available? │
│ Structural      │                       └────────┬────────┘
│ projection      │                           yes  │  no
│ (states)        │                          ┌─────┴─────┐
└─────────────────┘                          ▼           ▼
                                   ┌──────────────┐  ┌───────────┐
                                   │ Path B:      │  │ Path C:   │
                                   │ Text-derived │  │ Neighbor  │
                                   │ (orgs,       │  │ average   │
                                   │  leaders)    │  │ (sparse)  │
                                   └──────────────┘  └───────────┘
```

### 3.2 Path A: Structural Projection (Nation-States)

For nation-states with full structural feature vectors:

```python
def init_state_embedding(structural_features: np.ndarray, d: int) -> np.ndarray:
    """
    Project structural features into d-dimensional embedding space.

    structural_features: shape [n_features] (e.g., 18 features)
    d: embedding dimension (256 or 512)
    Returns: h_i(0) of shape [d]
    """
    # Learned linear projection (trained in Phase 1 of training pipeline)
    # Before Phase 1 training, use PCA as a deterministic initialization
    h_i_0 = W_struct @ structural_features + b_struct  # shape: [d]
    return h_i_0
```

**Pre-training initialization:** Before the Phase 1 structural pretraining (Component 5), use PCA on the full structural feature matrix of all states to produce a d-dimensional projection. This is deterministic and reproducible.

**Post-Phase-1 initialization:** After structural pretraining, `W_struct` and `b_struct` are learned parameters. New actors added to the registry after training can be initialized by applying this learned projection to their structural features.

### 3.3 Path B: Text-Derived Initialization (IGOs, Leaders, Companies)

For actors with limited structural data but available text:

```python
def init_text_embedding(texts: list[str], d: int) -> np.ndarray:
    """
    Initialize embedding from representative text documents.

    texts: list of relevant documents (Wikipedia bio, charter text,
           speeches, founding treaty, major news articles about the actor)
    d: embedding dimension
    Returns: h_i(0) of shape [d]
    """
    # Encode each text with ConfliBERT
    token_embeddings = [conflibert.encode(text) for text in texts]  # each: [seq_len, 768]

    # Mean-pool each document, then average across documents
    doc_embeddings = [mean_pool(emb) for emb in token_embeddings]  # each: [768]
    avg_embedding = mean(doc_embeddings)  # [768]

    # Project to embedding dimension
    h_i_0 = W_text @ avg_embedding + b_text  # [d]
    return h_i_0
```

**Text sources per actor type:**

| Actor type | Text sources for initialization |
|------------|--------------------------------|
| IGOs | Charter/founding treaty + Wikipedia article + recent annual reports |
| Regional orgs | Founding treaty + Wikipedia article + recent communiqués |
| Non-state armed groups | Wikipedia article + top-20 most relevant POLECAT-linked articles |
| Companies | Wikipedia article + most recent 10-K filing summary + top-20 geopolitically-relevant news articles |
| Leaders | Wikipedia biography + top-5 major foreign policy speeches + party manifesto excerpts |

### 3.4 Path C: Neighbor Average (Sparse Actors)

For actors with neither structural features nor sufficient text (rare):

```python
def init_neighbor_embedding(actor_id: str, d: int) -> np.ndarray:
    """
    Initialize from average of similar actors' embeddings.

    Find actors of the same type in the same geographic region
    with existing embeddings, and average them.
    """
    same_type = [a for a in registry if a.actor_type == actor.actor_type]
    same_region = [a for a in same_type if a.region == actor.region]

    if len(same_region) >= 3:
        neighbors = same_region
    else:
        neighbors = same_type  # fall back to type-level average

    h_i_0 = mean([h_j_0 for j in neighbors if h_j_0 is not None])
    return h_i_0
```

This is the weakest initialization and should only be used as a fallback. The model will update these vectors quickly from event data once training begins.

### 3.5 Name Identity Encoding

Every actor, regardless of initialization path, receives a name encoding vector that gives the model a lexical anchor for connecting actors to their mentions in text.

```python
def compute_name_encoding(actor: Actor, d: int) -> Tensor:
    """
    Encode an actor's canonical name and aliases through ConfliBERT.

    This gives the model a direct lexical hook — when the actor later
    cross-attends over a document (Component 4, Layer 2), its query
    vector includes information about what its name looks like as text.
    Without this, two structurally similar countries (e.g., Czech Republic
    and Slovakia) would attend to documents identically at initialization.
    """
    # Collect name variants
    name_texts = [actor.name] + actor.aliases  # e.g., ["Russia", "Russian Federation", "РФ"]

    # Encode each variant through ConfliBERT (frozen at this stage)
    name_embeddings = []
    for name in name_texts:
        tokens = tokenizer(name, return_tensors="pt")
        emb = conflibert(**tokens).last_hidden_state.mean(dim=1)  # [768]
        name_embeddings.append(emb)

    # Average across variants, project to embedding dimension
    avg_name = torch.stack(name_embeddings).mean(dim=0)  # [768]
    name_enc = W_name @ avg_name + b_name                # [d]

    return name_enc
```

**Integration with initialization paths:**

All three paths (A, B, C) produce a base embedding from structural/text/neighbor features. The name encoding is added as a separate component:

```python
def init_actor_embedding(actor: Actor, d: int) -> Tensor:
    # 1. Compute base embedding via appropriate path
    if has_structural_features(actor):
        h_base = init_state_embedding(actor, d)      # Path A
    elif has_text(actor):
        h_base = init_text_embedding(actor, d)        # Path B
    else:
        h_base = init_neighbor_embedding(actor, d)    # Path C

    # 2. Add name encoding (all actors get this)
    h_name = compute_name_encoding(actor, d)

    # 3. Combine via learned gated addition
    gate = sigmoid(W_gate @ torch.cat([h_base, h_name]))  # [d]
    h_i_0 = h_base + gate * h_name

    return h_i_0  # this becomes h_baseline_i
```

**Why gated addition, not concatenation:** Concatenation would double the embedding dimension. The gate lets the model learn how much weight to give the name encoding vs. structural features per dimension. For states with rich structural data, the gate may be small. For newly added non-state actors, the name encoding may dominate initially.

**Why this matters for cross-attention:** When this actor later queries a document via `W_Q @ h_i`, the name encoding means the query vector has affinity for tokens that resemble the actor's name. Russia's query vector will naturally attend more strongly to "Russia", "Russian", "Moscow", "Kremlin" — not because of a hardcoded rule, but because those tokens' ConfliBERT representations are similar to the name encoding that's baked into the memory.

---

## 4. Actor Sketch Vectors

Alongside the full d-dimensional embedding, each actor maintains a lightweight sketch vector for the text processing pipeline's fast relevance filter (Component 4, Layer 2).

### 4.1 Sketch Vector Computation

```python
def compute_sketch_vector(actor: Actor, dim: int = 64) -> np.ndarray:
    """
    Compute a lightweight TF-IDF sketch vector for fast article-actor relevance scoring.

    Built from the actor's name, aliases, and a sample of articles known to mention them.
    """
    # Collect text snippets associated with this actor
    actor_texts = [actor.name] + actor.aliases

    # Add snippets from articles known to reference this actor
    # (from POLECAT/GDELT event records with source URLs)
    representative_articles = get_articles_mentioning(actor.actor_id, limit=100)
    actor_texts.extend([truncate(art.body_text, 500) for art in representative_articles])

    # Compute TF-IDF over this actor's text corpus
    combined_text = " ".join(actor_texts)
    sketch = tfidf_vectorizer.transform([combined_text])  # sparse vector

    # Reduce to fixed dimension via random projection or truncated SVD
    sketch_dense = random_projection(sketch, dim)  # shape: [dim]
    return sketch_dense
```

### 4.2 Sketch Update Schedule

Sketch vectors are recomputed:
- At initialization
- Monthly (to incorporate recent articles)
- When a new alias is added to the actor registry

Sketch vectors do **not** need to be recomputed on every training step. They are a cheap pre-filter, not a precision instrument.

---

## 5. Registry Maintenance

### 5.1 Adding New Actors

When a new actor becomes geopolitically relevant (new leader takes office, new organization formed, new armed group emerges):

1. Add entry to the registry with canonical ID, type, aliases, source code mappings.
2. Compute structural features where available.
3. Initialize embedding via the appropriate path (A, B, or C).
4. Compute sketch vector.
5. The model will update the embedding from data during the next training or inference run.

### 5.2 Retiring Actors

When an actor ceases to be relevant (leader leaves office, organization dissolves):

1. Set `active_to` date in the registry.
2. Freeze the embedding at its last state. Do not delete — the historical embedding is needed for retrospective analysis and training on historical data.
3. Stop ingesting new events for this actor.

### 5.3 Leader Succession

When a head of state changes:
1. Retire the outgoing leader's actor entry.
2. Create a new actor entry for the incoming leader.
3. Initialize the new leader's embedding from their available text (speeches, biography) via Path B.
4. The state-level actor (`state:USA`) is unaffected — it persists across leadership changes. The state's embedding will naturally drift as the new leader's actions update it through the event and text streams.

---

## 6. Output Artifacts

This component produces:

| Artifact | Format | Description |
|----------|--------|-------------|
| `actor_registry.parquet` | Parquet | The full actor registry table |
| `structural_features.parquet` | Parquet | Per-actor-per-year structural feature panel |
| `dyadic_features.parquet` | Parquet | Per-actor-pair-per-year dyadic features |
| `initial_embeddings.npy` | NumPy | Matrix of shape `[n_actors, d]` with initial embeddings |
| `sketch_vectors.npy` | NumPy | Matrix of shape `[n_actors, sketch_dim]` with sketch vectors |
| `source_code_mappings.json` | JSON | GDELT/POLECAT/ICEWS → canonical ID mapping tables |
