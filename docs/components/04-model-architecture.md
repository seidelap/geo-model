# Component 4: Model Architecture

## Purpose

Define the neural network architecture that transforms curated data into event predictions. The architecture has six layers, each with a specific role. This document specifies the inputs, outputs, parameters, and computation of each layer.

**Inputs:** Actor registry with initial embeddings (Component 2). Curated text and event data (Component 1). Training datasets (Component 3).

**Outputs:** Probability estimates P(event_type r | actor_i, actor_j, horizon τ) for all queryable combinations.

---

## 1. Architecture Overview

```
Layer 1: Actor Memory Store      Persistent state vectors h_i(t) ∈ R^d per actor
Layer 2: Text Processing Stream  Updates memory from news articles (high latency, high richness)
Layer 3: Structured Event Stream Updates memory from coded events (low latency, lower richness)
Layer 4: Temporal Graph           Propagates information across the actor network periodically
Layer 5: Event Prediction Head    Produces event probabilities from actor state pairs
Layer 6: Calibration             Post-hoc probability correction
```

Layers 2 and 3 run in parallel, both writing to Layer 1. Layer 4 runs periodically (daily/weekly). Layers 5 and 6 are invoked at query time.

**Key hyperparameter:** d = embedding dimension. Start with d=256. Scale to d=512 if capacity is insufficient.

---

## 2. Layer 1: Actor Memory Store

### 2.1 Data Structure

Each actor i in the registry maintains:

| Field | Shape | Description |
|-------|-------|-------------|
| `h_i` | `[d]` | The memory vector. Updated by Layers 2, 3, and 4. |
| `t_last_updated_i` | `scalar` | Timestamp of last memory update. Used for temporal decay. |
| `sketch_i` | `[sketch_dim]` | Lightweight sketch vector for text relevance filtering. |

The full memory store is a matrix H of shape `[N_actors, d]` plus metadata. For N=500 actors and d=512, this is ~1 MB — entirely in-memory.

### 2.2 Temporal Decay

Between updates, memory decays toward a learned baseline:

```python
def apply_decay(h_i: Tensor, t_last: float, t_now: float, lambda_decay: float, h_baseline_i: Tensor) -> Tensor:
    """
    Exponential decay toward a learned per-actor baseline.
    lambda_decay corresponds to a half-life: t_half = ln(2) / lambda_decay
    Target half-life: 90–180 days (λ ≈ 0.004–0.008 per day)
    """
    dt = t_now - t_last  # in days
    decay_factor = math.exp(-lambda_decay * dt)
    h_decayed = h_baseline_i + decay_factor * (h_i - h_baseline_i)
    return h_decayed
```

`h_baseline_i` is a learned parameter per actor (or per actor type). It represents the actor's "resting state" — what the model assumes about the actor in the absence of recent information. Initialized from the structural embedding.

### 2.3 Storage

In training: H is a PyTorch tensor on GPU, updated in-place within each training rollout. Between rollouts, H is checkpointed to disk.

In inference: H is persisted in a key-value store (SQLite for single-machine; Redis for multi-process). Updated continuously as new data arrives.

---

## 3. Layer 2: Text Processing Stream

### 3.1 Purpose

Extract information from news articles and update actor memory vectors. This is the richest signal source — text preserves hedging, tone, causal framing, and forward-looking language that structured event coding discards.

### 3.2 Pipeline

```
Curated article
  → Sketch-based actor relevance filter (CPU, fast)
  → Full document encoding with ConfliBERT (GPU)
  → Actor mention extraction
  → Cross-actor interaction representation
  → Gated memory update
```

### 3.3 Sketch-Based Actor Relevance Filter

Determine which actors each article should update without running the full encoder:

```python
def filter_relevant_actors(article_text: str, actor_sketches: Tensor, temperature: float = 1.0) -> list[tuple[str, float]]:
    """
    Fast relevance filtering using pre-computed actor sketch vectors.

    article_text: raw article text
    actor_sketches: [N_actors, sketch_dim] matrix
    Returns: list of (actor_id, weight) for actors with nonzero weight
    """
    # Compute lightweight article sketch
    article_sketch = tfidf_vectorizer.transform([article_text])  # sparse, [1, vocab]
    article_dense = random_projection(article_sketch, sketch_dim)  # [sketch_dim]

    # Dot product relevance scores
    relevance_scores = actor_sketches @ article_dense  # [N_actors]

    # Sparsemax: learned sparse selection
    actor_weights = sparsemax(relevance_scores / temperature)  # [N_actors], mostly zeros

    # Return only nonzero-weight actors (typically 5–20 per article)
    active = [(actor_id, w) for actor_id, w in zip(actor_ids, actor_weights) if w > 0]
    return active
```

**Compute cost:** Microseconds per article on CPU. The entire daily batch (40K articles) takes seconds.

### 3.4 Full Document Encoding

For articles that pass the sketch filter:

```python
def encode_document(article_text: str) -> Tensor:
    """
    Encode article with ConfliBERT.

    Returns: token-level contextual embeddings [seq_len, 768]
    """
    tokens = tokenizer(article_text, max_length=512, truncation=True, return_tensors="pt")
    T = conflibert(**tokens).last_hidden_state  # [1, seq_len, 768]
    return T.squeeze(0)  # [seq_len, 768]
```

**Model:** ConfliBERT (BERT-base, 12 layers, 110M params). Domain-adapted for conflict/political text. Available at `github.com/eventdata/ConfliBERT`.

**Throughput:** ~8,000–12,000 tokens/sec on T4; ~40,000 articles/day takes ~25–35 minutes.

### 3.5 Actor Mention Extraction

For each active actor, extract their representation from the document:

```python
def extract_actor_mention(T: Tensor, actor_id: str, article_text: str, actor_weight: float) -> Tensor:
    """
    Extract an actor-specific representation from encoded document.

    Two strategies, combined:
    1. Hard mention pooling: average embeddings at token positions where the actor is named
    2. Soft weighted pooling: weight all tokens by actor relevance (fallback for indirect mentions)
    """
    mention_spans = find_mention_spans(article_text, actor_id)  # NER + alias matching

    if mention_spans:
        # Hard mention: average contextual embeddings at mention positions
        mention_tokens = torch.cat([T[start:end] for start, end in mention_spans])
        m_hard = mention_tokens.mean(dim=0)  # [768]

        # Blend with soft attention for broader context
        attn_weights = softmax(T @ m_hard / math.sqrt(768))  # [seq_len]
        m_soft = (attn_weights.unsqueeze(-1) * T).sum(dim=0)  # [768]

        m_i = 0.7 * m_hard + 0.3 * m_soft  # weighted blend
    else:
        # No explicit mention found — use soft pooling only (indirect relevance from sketch)
        # Use a learned query vector per actor type as attention query
        query = actor_type_queries[actor_type_of(actor_id)]  # [768]
        attn_weights = softmax(T @ query / math.sqrt(768))
        m_i = (attn_weights.unsqueeze(-1) * T).sum(dim=0)  # [768]

    return m_i  # [768]
```

### 3.6 Cross-Actor Interaction Representation

When multiple actors co-occur in an article, compute a pairwise interaction representation:

```python
def compute_cross_actor(m_i: Tensor, m_j: Tensor, T: Tensor) -> Tensor:
    """
    What is happening between actors i and j in this document?
    Cross-attention: actor i attends to document context conditioned on actor j.
    """
    query = W_Q(m_i)          # [d_k]
    keys = W_K(T)             # [seq_len, d_k]
    values = W_V(T)           # [seq_len, d_v]

    # Condition query on the other actor
    query_conditioned = query + W_cond(m_j)  # [d_k], adds relational bias

    attn_scores = query_conditioned @ keys.T / math.sqrt(d_k)  # [seq_len]
    attn_weights = softmax(attn_scores)
    c_ij = attn_weights @ values  # [d_v]

    return c_ij
```

This is computed for every pair of co-active actors in the article. For an article mentioning 5 actors, this produces 5×4 = 20 pairwise representations. In practice, most articles mention 2–3 actors, keeping this manageable.

### 3.7 Gated Memory Update

Update each active actor's memory:

```python
def update_memory_from_text(
    h_i: Tensor,           # current memory [d]
    m_i: Tensor,           # actor mention representation [768]
    c_ij_list: list[Tensor],  # cross-actor representations involving i
    t: float,              # current timestamp
    t_last: float,         # last update timestamp
) -> Tensor:
    """
    Gated residual update to actor memory.
    """
    # Aggregate relational context
    if c_ij_list:
        relational_context = torch.stack(c_ij_list).mean(dim=0)  # [d_v]
    else:
        relational_context = torch.zeros(d_v)

    # Project mention and context to memory dimension
    update_input = W_proj(torch.cat([h_i, m_i, relational_context, time2vec(t)]))  # [d]

    # Scalar gate: should this article update this actor at all?
    gate_scalar = sparsemax(W_scalar(update_input))  # scalar in [0, 1], competing across actors

    # Dimensional gate: which dimensions to update?
    gate_dims = sigmoid(W_dims(update_input))  # [d], independent per dimension

    # Candidate update (residual)
    delta_h = MLP_update(update_input)  # [d]

    # Apply temporal decay first, then add gated update
    h_decayed = apply_decay(h_i, t_last, t, lambda_decay, h_baseline_i)
    h_new = h_decayed + gate_dims * (gate_scalar * delta_h)

    return h_new
```

**Learnable parameters in this layer:**
- ConfliBERT encoder: 110M params (fine-tuned, not frozen)
- W_Q, W_K, W_V, W_cond: cross-attention projections, 4 × (768 × d_k)
- W_proj: input projection, (768 + 768 + d_v + time_dim) × d
- W_scalar: scalar gate, d × 1
- W_dims: dimensional gate, d × d
- MLP_update: 2-layer MLP, d × 4d × d
- h_baseline per actor: N_actors × d
- time2vec parameters: time_dim

---

## 4. Layer 3: Structured Event Stream

### 4.1 Purpose

Fast, low-latency memory updates from coded event records. Cheaper than the text stream (no encoder) and provides a continuous signal even before articles are processed.

### 4.2 Event Encoding

```python
def encode_event(event: NormalizedEvent) -> Tensor:
    """
    Encode a structured event into a dense vector.
    """
    e = torch.cat([
        relation_embeddings[event.event_type],  # [d], learnable per PLOVER type
        torch.tensor([event.goldstein_score]),   # [1], normalized to [-1, 1]
        mode_encoding[event.event_mode],         # [3], one-hot: verbal/hypothetical/actual
        torch.tensor([
            math.log1p(event.magnitude_dead),
            math.log1p(event.magnitude_injured),
            math.log1p(event.magnitude_size),
        ]),                                      # [3], log-transformed magnitudes
        time2vec(event.event_date),              # [time_dim]
    ])
    return e  # [d + 7 + time_dim]
```

### 4.3 GRU Memory Update

Two GRUs — one for the source actor (active role), one for the target (passive role):

```python
def update_memory_from_event(
    h_source: Tensor,       # [d]
    h_target: Tensor,       # [d]
    e_event: Tensor,        # encoded event
) -> tuple[Tensor, Tensor]:
    """
    Update both actors' memories from a structured event.
    Source GRU conditions on the target's state (and vice versa).
    """
    # Source actor update (the actor initiating the event)
    source_input = torch.cat([e_event, h_target])
    h_source_new = GRU_source(source_input, h_source)  # [d]

    # Target actor update (the actor receiving the event)
    target_input = torch.cat([e_event, h_source])
    h_target_new = GRU_target(target_input, h_target)  # [d]

    return h_source_new, h_target_new
```

GRU_source and GRU_target share architecture but have independent parameters. Each is a standard GRU cell: input dimension = (d + 7 + time_dim + d), hidden dimension = d.

### 4.4 Fusion with Text Stream

Both streams write to the same memory vectors. When a structured event and its corresponding articles arrive within a short window:
- The structured event provides a fast initial update.
- The text stream later provides a richer, contextualized update.
- The gating mechanism in Layer 2 naturally suppresses redundant updates — if the event already moved the memory in the right direction, the text update gate will activate less strongly.

No explicit fusion logic is needed. The shared memory and learned gates handle it.

---

## 5. Layer 4: Temporal Graph Propagation

### 5.1 Purpose

Propagate information across the actor network periodically. Captures second-order effects: if Germany and France have a major summit, the France-Algeria relationship is also affected through linkages.

### 5.2 Execution Schedule

Run graph propagation as a batch process:
- **During training:** After every N memory update steps (N = 50–100 events/articles).
- **During inference:** Daily or weekly, depending on compute budget.

### 5.3 Graph Construction

```python
def build_graph(events_recent: list[NormalizedEvent], decay_beta: float = 0.01) -> tuple[Tensor, Tensor, Tensor]:
    """
    Construct the multi-relational graph from recent events.

    Returns:
        edge_index: [2, E] source/target actor index pairs
        edge_type: [E] PLOVER type index (0–17) per edge
        edge_weight: [E] recency-weighted interaction strength per edge
    """
    edges = defaultdict(float)

    for event in events_recent:
        i = actor_index[event.source_actor_id]
        j = actor_index[event.target_actor_id]
        r = plover_index[event.event_type]
        days_ago = (reference_date - event.event_date).days

        # Exponential recency weighting
        weight = math.exp(-decay_beta * days_ago)
        edges[(i, j, r)] += weight

    # Filter: only keep edges above minimum weight threshold
    threshold = 0.01
    filtered = [(i, j, r, w) for (i, j, r), w in edges.items() if w > threshold]

    edge_index = torch.tensor([[i, j] for i, j, r, w in filtered]).T  # [2, E]
    edge_type = torch.tensor([r for i, j, r, w in filtered])          # [E]
    edge_weight = torch.tensor([w for i, j, r, w in filtered])        # [E]

    return edge_index, edge_type, edge_weight
```

### 5.4 Multi-Relational Graph Attention

```python
class RelationalGraphAttentionLayer(nn.Module):
    """
    One layer of multi-relational graph attention.
    Separate weight matrices per relation type.
    """
    def __init__(self, d: int, n_relations: int = 18):
        self.W_r = nn.ParameterList([nn.Linear(d, d) for _ in range(n_relations)])
        self.a_r = nn.ParameterList([nn.Linear(2 * d + time_dim, 1) for _ in range(n_relations)])
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, H: Tensor, edge_index: Tensor, edge_type: Tensor, edge_weight: Tensor) -> Tensor:
        """
        H: [N_actors, d] current actor memories
        Returns: [N_actors, d] updated actor memories
        """
        messages = torch.zeros_like(H)  # [N_actors, d]

        for r in range(18):
            # Edges of this relation type
            mask = edge_type == r
            if not mask.any():
                continue

            src = edge_index[0, mask]  # source actor indices
            tgt = edge_index[1, mask]  # target actor indices
            w = edge_weight[mask]      # edge weights

            # Transform actor representations with relation-specific weights
            h_src_r = self.W_r[r](H[src])  # [E_r, d]
            h_tgt_r = self.W_r[r](H[tgt])  # [E_r, d]

            # Attention coefficients
            attn_input = torch.cat([h_src_r, h_tgt_r], dim=-1)  # [E_r, 2d]
            alpha = F.leaky_relu(self.a_r[r](attn_input).squeeze(-1))  # [E_r]
            alpha = alpha * w  # weight by recency

            # Softmax per target node (each node normalizes over its incoming edges)
            alpha = scatter_softmax(alpha, tgt, dim=0)

            # Weighted message aggregation
            msg = alpha.unsqueeze(-1) * h_src_r  # [E_r, d]
            messages.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)

        # Residual connection + layer norm
        H_new = H + self.layer_norm(messages)
        return H_new
```

### 5.5 Multi-Layer Stacking

Stack 2–3 graph attention layers. Each layer allows information to propagate one hop further:

```python
class TemporalGraphPropagation(nn.Module):
    def __init__(self, d: int, n_layers: int = 2):
        self.layers = nn.ModuleList([RelationalGraphAttentionLayer(d) for _ in range(n_layers)])

    def forward(self, H: Tensor, edge_index: Tensor, edge_type: Tensor, edge_weight: Tensor) -> Tensor:
        for layer in self.layers:
            H = layer(H, edge_index, edge_type, edge_weight)
        return H
```

**Learnable parameters:**
- Per layer: 18 × (d × d) relation projections + 18 × ((2d + time_dim) × 1) attention vectors + layer norm
- Total for 2 layers: ~2 × (18 × d² + 18 × 2d) ≈ 2 × 18 × d² ≈ 2.4M params at d=256

---

## 6. Layer 5: Event Prediction Head

### 6.1 Dyadic Representation

For a query P(event_type r | actor_i, actor_j, horizon τ):

```python
def build_dyadic_representation(h_i: Tensor, h_j: Tensor, r: int, tau: int) -> Tensor:
    """
    Construct the feature vector for a prediction query.

    h_i: [d] source actor memory
    h_j: [d] target actor memory
    r: event type index (0–17)
    tau: prediction horizon in days
    """
    d_ij = torch.cat([
        h_i,                            # [d]    source state
        h_j,                            # [d]    target state
        h_i * h_j,                      # [d]    element-wise product (symmetric compatibility)
        h_i - h_j,                      # [d]    difference (asymmetry)
        torch.abs(h_i - h_j),           # [d]    absolute difference (distance)
        time2vec(tau),                   # [time_dim]  horizon encoding
        relation_embeddings[r],         # [d]    relation type embedding
    ])
    return d_ij  # [5d + time_dim + d] = [6d + time_dim]
```

**Why these five combinations of h_i and h_j:**
- `h_i, h_j` separately: the individual actor states matter (a strong military actor behaves differently)
- `h_i * h_j`: symmetric pairwise compatibility per dimension — high product means both actors are strong on that dimension
- `h_i - h_j`: directional asymmetry — who is more powerful, more Western-aligned, more democratic
- `|h_i - h_j|`: undirected distance per dimension — how different are they, regardless of direction

### 6.2 Multi-Task Prediction

```python
class EventPredictionHead(nn.Module):
    def __init__(self, input_dim: int, d: int, n_event_types: int = 18):
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 4 * d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d, 2 * d),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Per-event-type heads
        self.event_heads = nn.ModuleList([
            nn.Linear(2 * d, 1) for _ in range(n_event_types)
        ])
        # Auxiliary heads
        self.goldstein_head = nn.Linear(2 * d, 1)       # intensity regression
        self.escalation_head = nn.Linear(2 * d, 1)      # binary escalation

    def forward(self, d_ij: Tensor, event_type: int) -> dict:
        trunk = self.shared(d_ij)  # [2d]

        # Event probability
        logit = self.event_heads[event_type](trunk)  # [1]
        p_event = torch.sigmoid(logit)

        # Auxiliary predictions
        goldstein_pred = self.goldstein_head(trunk)   # [1], raw regression
        p_escalation = torch.sigmoid(self.escalation_head(trunk))  # [1]

        return {
            "p_event": p_event,
            "logit": logit,
            "goldstein_pred": goldstein_pred,
            "p_escalation": p_escalation,
        }
```

### 6.3 Hawkes Process for Temporal Intensity

For predicting *when* (not just whether) events occur:

```python
class HawkesIntensity(nn.Module):
    """
    Conditional intensity function: rate at which events occur between a dyad,
    given their current states and event history.

    λ(t) = μ(h_i, h_j) + Σ_k α_rk * exp(-β * (t - t_k))

    μ: base rate from actor states
    Σ: self-excitation from past events (events beget events)
    """
    def __init__(self, d: int):
        self.base_rate_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
            nn.Softplus(),  # ensure positive rate
        )
        self.excitation_weights = nn.Parameter(torch.ones(18) * 0.1)  # per event type
        self.decay_beta = nn.Parameter(torch.tensor(0.1))  # learnable decay

    def forward(self, h_i: Tensor, h_j: Tensor, event_history: list[tuple], t: float) -> Tensor:
        # Base rate from current actor states
        mu = self.base_rate_mlp(torch.cat([h_i, h_j]))

        # Self-excitation from historical events
        excitation = torch.tensor(0.0)
        for t_k, r_k in event_history:
            if t_k < t:
                excitation += self.excitation_weights[r_k] * torch.exp(-self.decay_beta * (t - t_k))

        lambda_t = mu + F.softplus(excitation)
        return lambda_t
```

### 6.4 Survival Model (DeepHit)

For time-to-event prediction ("how many days until the next X between i and j?"):

```python
class DeepHitHead(nn.Module):
    """
    Discrete-time hazard model. Outputs hazard at each of K discrete time bins.
    """
    def __init__(self, input_dim: int, d: int, n_time_bins: int = 30):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, n_time_bins),
        )
        # Time bins: [0-7, 7-14, 14-30, 30-60, 60-90, 90-120, 120-150, 150-180, ...]
        # Non-uniform binning: finer resolution for near-term, coarser for far-term

    def forward(self, d_ij: Tensor) -> dict:
        logits = self.mlp(d_ij)                        # [n_time_bins]
        hazard = torch.sigmoid(logits)                 # [n_time_bins], per-bin hazard
        survival = torch.cumprod(1 - hazard, dim=-1)   # [n_time_bins], survival function
        event_cdf = 1 - survival                       # [n_time_bins], cumulative event prob

        return {
            "hazard": hazard,
            "survival": survival,
            "event_cdf": event_cdf,
        }
```

---

## 7. Gating Mechanism Details

### 7.1 Scalar Gate: Sparsemax (α-entmax)

Controls whether an article/event updates a given actor at all:

```python
def entmax(scores: Tensor, alpha: float = 1.5) -> Tensor:
    """
    α-entmax: generalized sparse attention.
    α=1.0 → softmax (dense), α=1.5 → partial sparsity, α=2.0 → sparsemax (hard sparse)

    Produces exact zeros for low-scoring inputs.
    """
    # Use entmax15 from the entmax library for α=1.5
    return entmax15(scores)
```

α can be:
- Fixed at 1.5 (good default)
- Learned globally (one α for the whole model)
- Input-dependent: articles with narrow focus get higher α (sparser updates), multilateral coverage gets lower α

### 7.2 Dimensional Gate: Sigmoid

Controls which memory dimensions update:

```python
gate_dims = torch.sigmoid(W_dims @ update_input)  # [d], each element in (0,1)
```

Independent per dimension. No competition between dimensions — an article can simultaneously be relevant to security and economic dimensions.

### 7.3 Combined Update Formula

```python
delta_h = MLP_update(update_input)
h_new = h_decayed + gate_dims * (gate_scalar * delta_h)
```

Where:
- `gate_scalar ∈ [0, 1]` is from entmax (sparse across actors)
- `gate_dims ∈ (0, 1)^d` is from sigmoid (independent per dimension)
- `delta_h ∈ R^d` is the candidate update from the MLP

---

## 8. Time Encoding

### 8.1 time2vec

Continuous time encoding that captures multiple periodic and aperiodic components:

```python
class Time2Vec(nn.Module):
    """
    Learnable time encoding (Kazemi et al., 2019).
    First element: linear (aperiodic trend)
    Remaining elements: periodic with learned frequencies
    """
    def __init__(self, dim: int = 16):
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.randn(1))
        self.periodic_weights = nn.Parameter(torch.randn(dim - 1))
        self.periodic_biases = nn.Parameter(torch.randn(dim - 1))

    def forward(self, t: Tensor) -> Tensor:
        """t: scalar or [batch] of timestamps (in days)"""
        linear = self.linear_weight * t + self.linear_bias  # [1] or [batch, 1]
        periodic = torch.sin(self.periodic_weights * t.unsqueeze(-1) + self.periodic_biases)
        return torch.cat([linear.unsqueeze(-1), periodic], dim=-1)  # [dim]
```

### 8.2 Usage

Time2Vec appears in three places:
1. **Text memory update:** encodes the current timestamp to condition the update on where we are in the temporal sequence.
2. **Event encoding:** encodes the event timestamp for the structured event stream.
3. **Prediction head:** encodes the forecast horizon τ.

Each use has its own Time2Vec instance with independent learned frequencies, since they encode different temporal concepts (absolute time vs. horizon length).

---

## 9. Relation Embeddings

### 9.1 Structure

Each of the 18 PLOVER event types has a learnable dense embedding vector:

```python
relation_embeddings = nn.Embedding(18, d)  # [18, d]
```

These are shared across the event encoding (Layer 3), dyadic representation (Layer 5), and graph construction (Layer 4). Training discovers their geometry: event types that co-occur between similar dyads, or that tend to follow each other temporally, will cluster in the embedding space.

### 9.2 Initialization

Initialize relation embeddings based on the PLOVER Goldstein scale ordering. Events with similar Goldstein scores start nearby:

```python
# Initialize from Goldstein midpoints, projected into d dimensions
goldstein_midpoints = torch.tensor([5.5, 6.0, 2.5, 4.0, -3.5, -2.5, 2.0, -9.0, ...])  # per type
relation_embeddings.weight.data[:, 0] = normalize(goldstein_midpoints)
# Remaining dimensions: small random noise
relation_embeddings.weight.data[:, 1:] = torch.randn(18, d - 1) * 0.01
```

---

## 10. Parameter Count Summary

| Component | Parameters | Notes |
|-----------|------------|-------|
| ConfliBERT encoder | 110M | Fine-tuned during Phase 2–3 |
| Cross-attention (W_Q, W_K, W_V, W_cond) | ~2.4M | At d_k = d_v = 256 |
| Memory update MLP | ~0.8M | 2-layer, d × 4d × d |
| Scalar + dimensional gates | ~0.2M | |
| GRU source + target | ~1.6M | Two GRU cells, each d × d input |
| Graph attention (2 layers) | ~2.4M | 18 relations × d² per layer |
| Prediction head (shared trunk) | ~2.1M | 6d → 4d → 2d |
| Per-event-type heads (18) | ~9K | 18 × 2d → 1 |
| Hawkes process | ~0.1M | |
| DeepHit head | ~0.5M | |
| Relation embeddings | ~5K | 18 × d |
| Actor baselines | ~0.1M | N_actors × d |
| Time2Vec (3 instances) | ~100 | Tiny |
| **Total (excluding encoder)** | **~10M** | |
| **Total (including encoder)** | **~120M** | |

At d=256 and N_actors=500. Scales roughly as d² for most components.
