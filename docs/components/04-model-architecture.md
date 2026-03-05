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
Layer 4: Actor Self-Attention     Propagates information across all actors via full self-attention
Layer 5: Event Prediction Head    Produces event probabilities from actor state pairs
Layer 6: Calibration             Post-hoc probability correction
```

Layers 2 and 3 run in parallel, both writing to Layer 1. Layer 4 runs once per simulated day. Layers 5 and 6 are invoked at query time.

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

Between updates, memory decays toward the actor's baseline. The baseline itself is not fixed — it evolves as a slow-moving exponential moving average (EMA) of the actor's memory, tracking gradual shifts in the actor's structural position over time. Memory decays toward this moving target, not toward a static point from t=0.

```python
def apply_decay(
    h_i: Tensor,
    b_i: Tensor,            # current EMA baseline for actor i
    t_last: float,
    t_now: float,
    lambda_decay: float,
) -> Tensor:
    """
    Exponential decay toward per-actor EMA baseline.
    lambda_decay corresponds to a half-life: t_half = ln(2) / lambda_decay
    Target half-life: 90–180 days (λ ≈ 0.004–0.008 per day)
    """
    dt = t_now - t_last  # in days
    decay_factor = math.exp(-lambda_decay * dt)
    h_decayed = b_i + decay_factor * (h_i - b_i)
    return h_decayed


def update_baseline(b_i: Tensor, h_i: Tensor, alpha_baseline: float) -> Tensor:
    """
    EMA update to the actor's baseline. Runs once per day, after all
    memory updates (Layers 2, 3, 4) for that day are complete.

    alpha_baseline is close to 1 (e.g., 0.99), so the baseline moves
    much more slowly than memory. This is a shared learned scalar,
    constrained to (0.95, 1.0) via sigmoid reparameterization:
        alpha_baseline = 0.95 + 0.05 * sigmoid(alpha_raw)
    """
    b_i_new = alpha_baseline * b_i + (1 - alpha_baseline) * h_i
    return b_i_new
```

**Two-timescale dynamics:** The memory `h_i` operates on a fast timescale — updated daily by articles, events, and self-attention, decaying toward the baseline between updates. The baseline `b_i` operates on a slow timescale — a weighted average of recent memory states. With `alpha_baseline ≈ 0.99`, the baseline's effective half-life is ~69 days (`ln(2) / (1 - 0.99)`), meaning it tracks drift over months rather than days. This means:
- A **one-off spike** (a single dramatic event) barely moves the baseline. Memory decays back toward the pre-spike normal.
- A **sustained shift** (e.g., a sanctions regime change) gradually pulls the baseline to reflect the new normal. After several months, the decay target has moved — the model no longer tries to pull memory back to a pre-sanctions resting state.

**Causality guarantee:** The baseline at time t depends only on memory states from times ≤ t. No future information can leak backward through the EMA. This is the same causal guarantee that Adam's momentum terms provide — the running average is strictly a function of past gradients.

**Initialization:** At epoch start, `b_i(0) = h_baseline_init_i`, the structural projection baseline computed from the actor's fixed features and name encoding (see Component 2, Section 3.5 and Section 11.2 below). The EMA then evolves from there as the rollout proceeds. The structural projection provides a meaningful starting point; the EMA allows it to drift.

**`alpha_baseline` parameterization:** A single shared learned scalar, constrained to (0.95, 1.0) via sigmoid reparameterization. This range ensures the baseline always moves substantially slower than the memory decay rate. If the optimizer finds that a nearly-fixed baseline works best, α will push toward 1.0. If faster adaptation helps, it can move toward 0.95 (half-life ~14 days). The constraint prevents degenerate solutions where the baseline simply tracks memory in real time (which would eliminate the decay mechanism entirely).

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
  → For each relevant actor:
      Actor memory cross-attention over document (memory is the query)
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

**Model:** ConfliBERT (BERT-base, 12 layers, 110M params). Domain-adapted for conflict/political text. Available at `github.com/eventdata/ConfliBERT`. Base weights are **frozen**; adaptation is via LoRA (see Section 3.4.1).

**Throughput:** ~8,000–12,000 tokens/sec on T4; ~40,000 articles/day takes ~25–35 minutes.

#### 3.4.1 LoRA Adapter Configuration

ConfliBERT's 110M base parameters are frozen during all training phases. Adaptation uses **Low-Rank Adaptation (LoRA)** applied to the query and value projection matrices in each self-attention layer.

```python
# LoRA configuration for ConfliBERT
lora_config = {
    "r": 8,                          # rank of low-rank matrices
    "lora_alpha": 16,                # scaling factor (effective scale = alpha/r = 2)
    "target_modules": ["query", "value"],  # Q and V projections in each attention layer
    "lora_dropout": 0.05,            # dropout on LoRA input
    "bias": "none",                  # no bias adaptation
    "modules_to_save": None,         # no additional modules unfrozen
}
```

**Parameter breakdown:**
- Each LoRA adapter adds two matrices per target module: `A [768, 8]` and `B [8, 768]`
- Per attention layer: 2 target modules × 2 matrices × 768 × 8 = 24,576 params
- 12 attention layers × 24,576 = ~295K params
- With lora_alpha scaling and layer norms: **~0.6M trainable parameters** total

**Rationale:**
- Saves ~880MB optimizer states (Adam maintains 2 state tensors per parameter; freezing 110M params eliminates ~880MB)
- TBPTT with K=75 days is more comfortable on 24GB VRAM
- ConfliBERT is already domain-adapted for conflict text; LoRA provides task-specific fine-tuning without catastrophic forgetting
- LoRA adapters tolerate higher learning rates than full fine-tuning: `lr=1e-4` for LoRA vs the `lr=1e-5` that full fine-tuning would require

**Training phases:**
- Phase 0: ConfliBERT not used (LightGBM baseline)
- Phase 1 (structural pretrain): ConfliBERT frozen, LoRA disabled — only structural components trained
- Phase 2 (self-supervised): LoRA adapters enabled and trained (`lr=1e-4`)
- Phase 3 (supervised fine-tune): LoRA adapters continue training (`lr=1e-4`)

**Implementation:** Use `peft` library (`from peft import LoraConfig, get_peft_model`) or manual implementation with `nn.Linear` wrappers. The `peft` approach is preferred for compatibility with HuggingFace checkpointing.

### 3.5 Actor Memory Cross-Attention Over Document

Each relevant actor uses its current memory vector as a query to attend over the full encoded document. There is no mention extraction, no NER, and no separate event context pathway. The memory vector `h_i` already encodes everything the actor knows — its name (baked into `h_baseline_init_i` via Component 2, Section 3.5, and preserved through the EMA baseline), its structural profile, and its accumulated history from *both* the text and event streams. The two streams share the same memory, so event-stream updates naturally inform the next text-stream read and vice versa.

```python
def actor_reads_document(
    h_i: Tensor,           # current actor memory [d]
    T: Tensor,             # encoded document [seq_len, 768]
) -> Tensor:
    """
    Actor i reads the document by cross-attending over all tokens.

    The actor's memory is the query. The document tokens are keys/values.
    What the actor extracts depends on its current state — a country
    that just processed a FIGHT event via Layer 3 will attend to different
    tokens than one in a trade negotiation, even in the same article.

    No separate cross-stream grounding is needed: since Layers 2 and 3
    both write to h_i, the event stream's updates are already present in
    the memory when the text stream reads the next article.
    """
    # Project actor memory to query space
    query = W_Q(h_i)              # [d_k]

    # Project document tokens to key/value space (shared across all actors)
    keys = W_K(T)                 # [seq_len, d_k]
    values = W_V(T)               # [seq_len, d_v]

    # Standard scaled dot-product attention
    attn_scores = query @ keys.T / math.sqrt(d_k)  # [seq_len]
    attn_weights = softmax(attn_scores)             # [seq_len]
    m_i = attn_weights @ values                     # [d_v]

    return m_i  # what actor i extracted from this document
```

**How the actor knows who it is — two sources in the query:**
1. **Name encoding** (from `h_baseline_init_i`, see Component 2, Section 3.5): Gives lexical affinity — Russia's query attends more to tokens like "Russia", "Moscow", "Kremlin" because its baseline initialization encodes its name through ConfliBERT. This signal persists even as the EMA baseline drifts, because the name encoding is a component of the structural projection that anchors the baseline's starting point, and the EMA's slow update rate preserves it.
2. **Accumulated memory** (`h_i`): Encodes the actor's full recent history from both streams. A country that just processed a FIGHT event via the GRU (Layer 3) has a different memory state than one processing COOP events — and this shifts its attention pattern when reading the next article.

**Cross-stream interaction via shared memory:** The text and event streams don't need an explicit bridge because they write to the same `h_i`. Within each simulated day (see Component 5, Section 5.3), the processing order is: articles first (Layer 2), then structured events (Layer 3), then actor self-attention (Layer 4). Each step sees the cumulative effect of all prior steps that day. Across days, the previous day's event-stream updates are fully reflected in `h_i` when the next day's articles are read.

**Role-specific projections (Q/K/V):** The single memory vector `h_i` serves multiple roles — as a query when reading documents, as a key/value when participating in actor self-attention, and as input to the prediction head. Learned projection matrices (W_Q, W_K, W_V, etc.) create role-specific views of the same underlying state. This is the standard transformer approach: one representation, multiple projections. If capacity becomes a bottleneck (e.g., self-attention quality degrades text reading), we can split into separate vectors later, but the single-vector design is simpler and avoids the question of which vector the GRU writes to.

**Why this works for new entities:** A newly added actor starts with `h_baseline_init_i`, which includes the name encoding (lexical anchor) and structural/text-derived features (neighborhood anchor). The first few articles and events rapidly specialize the memory, and the EMA baseline begins tracking the actor's emerging identity from there.

**Compared to the alternatives:**
- No mention spans to find → no NER errors, no pipeline fragility
- No pairwise cross-actor attention → O(A) per article instead of O(A²), where A is the number of relevant actors
- No separate event context pathway → simpler architecture, cross-stream interaction is free via shared memory

### 3.6 Gated Memory Update

Update each active actor's memory:

```python
def update_memory_from_text(
    h_i: Tensor,           # current memory [d]
    m_i: Tensor,           # what actor i read from the document [d_v]
    t: float,              # current timestamp
    t_last: float,         # last update timestamp
) -> Tensor:
    """
    Gated residual update to actor memory.
    """
    # Project document reading and current state to memory dimension
    update_input = W_proj(torch.cat([h_i, m_i, time2vec(t)]))  # [d]

    # Scalar gate: should this article update this actor at all?
    # Competes across actors via sparsemax — an article about NATO-Russia tensions
    # should update Russia strongly, NATO somewhat, and Portugal barely at all
    gate_scalar = sparsemax(W_scalar(update_input))  # scalar in [0, 1]

    # Dimensional gate: which dimensions to update?
    gate_dims = sigmoid(W_dims(update_input))  # [d], independent per dimension

    # Candidate update (residual)
    delta_h = MLP_update(update_input)  # [d]

    # Gated residual update (no temporal decay here — decay is applied once per day)
    h_new = h_i + gate_dims * (gate_scalar * delta_h)

    return h_new
```

**Learnable parameters in this layer:**
- ConfliBERT encoder: 110M params (adapted via LoRA, ~0.6M trainable)
- W_Q: d × d_k (actor memory → query projection)
- W_K: 768 × d_k (document tokens → key projection)
- W_V: 768 × d_v (document tokens → value projection)
- W_proj: input projection, (d + d_v + time_dim) × d
- W_scalar: scalar gate, d × 1
- W_dims: dimensional gate, d × d
- MLP_update: 2-layer MLP, d × 4d × d
- h_baseline_init per actor: N_actors × d (computed from fixed inputs through shared projections; see Component 2, Section 3.5 and Section 11 below)
- alpha_baseline: 1 (shared learned EMA decay rate for baseline updates; see Section 2.2)
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

## 5. Layer 4: Actor Self-Attention

### 5.1 Purpose

Propagate information across the full actor population. Captures second-order effects: if Germany and France have a major summit, the France-Algeria relationship is also affected even if no direct event was coded between them.

**Design choice:** Every actor attends to every other actor via standard multi-head self-attention (a transformer block over the actor "sequence"). There is no explicit graph construction — no edges, no typed adjacency, no sparsity mask. The model learns which actors should attend to which entirely from data.

**Why full attention instead of sparse graph edges:**
1. **N is small.** With 500–2000 actors, full self-attention is O(N² × d) — a single matrix multiply on GPU, negligible compared to encoding thousands of articles through ConfliBERT.
2. **Text-informed topology.** Actors discussed together in articles develop correlated memories, and attention picks this up — even with zero coded PLOVER events between them.
3. **Emergent typed interactions.** With multi-head attention, different heads can specialize (cooperative vs. adversarial relationships), giving the equivalent of typed edges without hand-coding 18 PLOVER categories into the propagation structure.
4. **Multi-hop for free.** Stacking 2–3 transformer layers gives multi-hop propagation without explicit message-passing rounds.

### 5.2 Execution Schedule

Run actor self-attention **once per simulated day**, in both training and inference. This ensures identical cadence — the model never sees a different update rhythm than what it will encounter in production.

- **During training:** At the end of each simulated day in the chronological rollout, after processing all events and articles for that day, and before advancing to the next day.
- **During inference:** Once daily, after processing the day's event and article streams.

Within a single day, events and articles update actor memories individually (Layers 2 and 3). Then the daily self-attention step (Layer 4) allows those updates to ripple across the population. This two-phase structure (local updates → global propagation) repeats every day.

### 5.3 Actor Self-Attention Block

Each layer is a standard transformer block: multi-head self-attention followed by a feed-forward network, with residual connections and layer normalization.

```python
class ActorSelfAttentionLayer(nn.Module):
    """
    Standard transformer block over the actor population.
    Each actor attends to all other actors.
    """
    def __init__(self, d: int, n_heads: int = 8):
        self.mha = nn.MultiheadAttention(embed_dim=d, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, H: Tensor) -> Tensor:
        """
        H: [N_actors, d] current actor memories
        Returns: [N_actors, d] updated actor memories
        """
        # Self-attention: every actor attends to every other actor
        H_norm = self.norm1(H)
        H_attn, _ = self.mha(H_norm.unsqueeze(0), H_norm.unsqueeze(0), H_norm.unsqueeze(0))
        H = H + H_attn.squeeze(0)  # residual

        # Feed-forward
        H = H + self.ffn(self.norm2(H))  # residual

        return H
```

**Compute cost:** For N=1000 actors, d=256, 8 heads: the attention matrix is 1000×1000 — ~1M entries. This is trivial. The entire Layer 4 forward pass takes <1ms on a T4, compared to ~30 minutes for Layer 2 (article encoding).

### 5.4 Multi-Layer Stacking

Stack 2–3 self-attention layers. Each layer allows information to propagate one additional hop through the actor population:

```python
class ActorPropagation(nn.Module):
    def __init__(self, d: int, n_layers: int = 2, n_heads: int = 8):
        self.layers = nn.ModuleList([
            ActorSelfAttentionLayer(d, n_heads) for _ in range(n_layers)
        ])

    def forward(self, H: Tensor) -> Tensor:
        for layer in self.layers:
            H = layer(H)
        return H
```

**Learnable parameters:**
- Per layer: MHA projections (4 × d × d) + FFN (d × 4d + 4d × d = 8d²) + 2 × LayerNorm (4d)
- Total per layer: ~12d² ≈ 0.8M at d=256
- Total for 2 layers: ~1.6M params at d=256

### 5.5 Daily Memory Maintenance

At the end of each simulated day, after all articles (Layer 2) and events (Layer 3) have been processed, the following maintenance steps run in order:

```python
def daily_maintenance(model, active_actors, day_date):
    """
    Daily memory maintenance: decay, self-attention, EMA baseline update.
    Called once per simulated day, after all articles and events are processed.
    """
    # 1. Temporal decay toward EMA baseline for all actors
    for actor_id in active_actors:
        dt = (day_date - model.t_last_decay[actor_id]).days
        if dt > 0:
            decay = math.exp(-model.lambda_decay * dt)
            model.H[actor_id] = model.B[actor_id] + decay * (model.H[actor_id] - model.B[actor_id])
            model.t_last_decay[actor_id] = day_date

    # 2. Actor self-attention (Layer 4)
    active_H = gather_active(model.H, active_actors)
    updated_H = model.actor_propagation(active_H)
    write_back_active(model.H, updated_H, active_actors)

    # 3. EMA baseline update
    for actor_id in active_actors:
        model.B[actor_id] = model.alpha_baseline * model.B[actor_id] + \
                            (1 - model.alpha_baseline) * model.H[actor_id]
```

**Why once per day:** Applying decay per-article or per-event would create inconsistency between the text and event streams — an actor updated by 50 articles would experience 50 micro-decays, while one updated by 5 events would experience only 5. Daily decay is stream-agnostic and aligns with the natural temporal granularity of the data (events and articles are date-stamped, not hour-stamped).

---

## 6. Layer 5: Event Prediction Head

### 6.1 Dyadic Representation

The model's primary output is a survival curve over time, not a single probability. The dyadic representation therefore does **not** include a horizon embedding — the full temporal distribution is produced from one forward pass.

```python
def build_dyadic_representation(
    h_i: Tensor, h_j: Tensor, r: int,
    surprise_i: Tensor, surprise_j: Tensor,
) -> Tensor:
    """
    Construct the feature vector for a dyad-event query.
    No horizon parameter — the model outputs a full survival curve.

    h_i: [d] source actor memory
    h_j: [d] target actor memory
    r: event type index (0–17)
    surprise_i: [2] source actor surprise features (CPC score, event-type KL)
    surprise_j: [2] target actor surprise features
    """
    d_ij = torch.cat([
        h_i,                            # [d]    source state
        h_j,                            # [d]    target state
        h_i * h_j,                      # [d]    element-wise product (symmetric compatibility)
        h_i - h_j,                      # [d]    difference (asymmetry)
        torch.abs(h_i - h_j),           # [d]    absolute difference (distance)
        relation_embeddings[r],         # [d]    relation type embedding
        surprise_i,                     # [2]    source surprise (CPC score, event-type KL)
        surprise_j,                     # [2]    target surprise (CPC score, event-type KL)
    ])
    return d_ij  # [6d + 4]
```

**Why these five combinations of h_i and h_j:**
- `h_i, h_j` separately: the individual actor states matter (a strong military actor behaves differently)
- `h_i * h_j`: symmetric pairwise compatibility per dimension — high product means both actors are strong on that dimension
- `h_i - h_j`: directional asymmetry — who is more powerful, more Western-aligned, more democratic
- `|h_i - h_j|`: undirected distance per dimension — how different are they, regardless of direction

**Surprise features (from CPC + event-type prediction heads):**
- `surprise_i, surprise_j`: each is a 2-element vector containing [CPC_score, event_type_KL] for that actor at the current timestep. CPC_score = 1 − cosine_sim(predicted, actual) from the CPC head; event_type_KL = KL divergence between predicted and actual event-type distribution from the event-type prediction head.
- These features encode "how surprised was the model by recent inputs for this actor" — a signal that correlates with escalation and de-escalation dynamics. An actor whose news stream suddenly diverges from expectations (high surprise) may be entering a transition period.
- In Phase 2, these heads are trained by the self-supervised objectives. In Phase 3, they continue training as auxiliary losses (0.1 weight) while their outputs feed into the prediction heads as features.
- During the first few days of a rollout (before enough history for meaningful CPC scores), surprise features are set to zero.

### 6.2 Survival Curve Output (Primary)

The primary prediction head outputs a discrete hazard function over K non-uniform time bins, from which the full survival curve and CDF are derived:

```python
class SurvivalHead(nn.Module):
    """
    Primary output: discrete-time hazard model (DeepHit-style).
    Outputs hazard at each of K time bins → survival curve → CDF.

    Any fixed-horizon probability query is a CDF lookup.
    Monotonicity (P(30d) >= P(7d)) is guaranteed by construction.
    """
    # Non-uniform time bins: finer near-term, coarser far-term
    TIME_BIN_BOUNDARIES = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180, 270, 365]
    K = len(TIME_BIN_BOUNDARIES) - 1  # 17 bins

    def __init__(self, input_dim: int, d: int, n_event_types: int = 18):
        # Shared trunk: dyadic representation → compressed features
        # input_dim = 6d + 4 (5 actor-state combinations + relation embedding + surprise features)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 4 * d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d, 2 * d),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Dropout before per-type heads (reduces overfitting for rare event types)
        self.hazard_dropout = nn.Dropout(0.2)

        # Per-event-type hazard heads: each outputs K hazard values
        self.hazard_heads = nn.ModuleList([
            nn.Linear(2 * d, self.K) for _ in range(n_event_types)
        ])

        # Auxiliary heads
        self.goldstein_head = nn.Linear(2 * d, 1)       # intensity regression
        self.escalation_head = nn.Linear(2 * d, 1)      # binary escalation

    def forward(self, d_ij: Tensor, event_type: int) -> dict:
        trunk = self.shared(d_ij)  # [2d]

        # Per-bin hazard: probability of event in bin k, given survival to bin k
        logits = self.hazard_heads[event_type](self.hazard_dropout(trunk))  # [K]
        hazard = torch.sigmoid(logits)                    # [K], each in (0, 1)

        # Survival function: probability of no event up to end of bin k
        survival = torch.cumprod(1 - hazard, dim=-1)      # [K]

        # CDF: probability of at least one event by end of bin k
        cdf = 1 - survival                                # [K]

        # PDF (probability mass per bin): P(event in bin k)
        survival_shifted = torch.cat([torch.ones(1), survival[:-1]])
        pdf = hazard * survival_shifted                    # [K]

        # Auxiliary predictions
        goldstein_pred = self.goldstein_head(trunk)        # [1]
        p_escalation = torch.sigmoid(self.escalation_head(trunk))  # [1]

        return {
            "hazard": hazard,          # [K] per-bin conditional hazard
            "survival": survival,      # [K] survival function
            "cdf": cdf,               # [K] cumulative event probability
            "pdf": pdf,               # [K] probability mass per bin
            "goldstein_pred": goldstein_pred,
            "p_escalation": p_escalation,
        }

    def query_horizon(self, cdf: Tensor, tau_days: float) -> float:
        """
        Read off the event probability at an arbitrary horizon.
        Linearly interpolates between bin boundaries.
        """
        boundaries = self.TIME_BIN_BOUNDARIES
        for k in range(len(boundaries) - 1):
            if boundaries[k] <= tau_days <= boundaries[k + 1]:
                # Linear interpolation within bin
                frac = (tau_days - boundaries[k]) / (boundaries[k + 1] - boundaries[k])
                cdf_lo = cdf[k - 1] if k > 0 else 0.0
                cdf_hi = cdf[k]
                return cdf_lo + frac * (cdf_hi - cdf_lo)
        return cdf[-1].item()  # beyond last bin
```

### 6.3 Design Note: Temporal Clustering Without Hawkes Excitation

The architecture does not include a separate Hawkes self-excitation component. Instead, temporal clustering ("events beget events") is captured through the actor memory mechanism:

- When a FIGHT event occurs, the GRU (Layer 3) updates both actors' memories, shifting them toward conflict-associated regions of the embedding space.
- The updated memories produce elevated hazard logits in the near-term bins, naturally reflecting higher short-term risk after recent activity.
- Cross-type excitation (e.g., THREATEN → FIGHT escalation) is captured through the same mechanism: a THREATEN event shifts actor states in a direction that increases FIGHT hazard.

This approach avoids the need to maintain per-dyad event histories at prediction time and keeps the prediction head stateless (dependent only on current actor memories). If empirical evaluation shows poor calibration on bursty event types, a Hawkes excitation term can be re-introduced as an additive component on hazard logits.

### 6.4 CPC Prediction Head (Self-Supervised / Auxiliary)

Projects actor memory into a "prediction space" for contrastive predictive coding. Used as a self-supervised objective in Phase 2 and carried over as an auxiliary loss in Phase 3.

```python
class CPCPredictionHead(nn.Module):
    """
    Contrastive Predictive Coding: project actor memory into a space where
    it can distinguish the actor's actual next-day article signal from negatives.
    """
    def __init__(self, d: int, d_cpc: int = 128):
        # Predictor: maps memory → prediction space
        self.W_pred = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d_cpc),
        )
        # Target encoder: maps article aggregate embedding → same prediction space
        self.W_target = nn.Sequential(
            nn.Linear(768, d),
            nn.ReLU(),
            nn.Linear(d, d_cpc),
        )

    def forward(self, h_i: Tensor, target_embedding: Tensor) -> tuple[Tensor, Tensor]:
        """
        h_i: [d] actor memory state (post Layer 4 self-attention)
        target_embedding: [768] mean-pooled ConfliBERT embedding of next-day articles

        Returns: (z_pred, z_target), both [d_cpc], L2-normalized
        """
        z_pred = F.normalize(self.W_pred(h_i), dim=-1)
        z_target = F.normalize(self.W_target(target_embedding), dim=-1)
        return z_pred, z_target
```

**Surprise score at inference:** `1 - cosine_sim(z_pred, z_target)` provides a per-actor, per-day anomaly score. High values mean the actor's actual news diverged from what the memory predicted. This score is concatenated to the dyadic representation (Section 6.1) as an additional feature for the survival head during Phase 3.

### 6.5 Event-Type Prediction Head (Self-Supervised / Auxiliary)

Predicts the distribution of PLOVER event types an actor will be involved in during the next 7-day window. Provides an interpretable "surprise" signal.

```python
class EventTypePredictionHead(nn.Module):
    """
    Predict next-window event-type distribution from actor memory.
    """
    def __init__(self, d: int, n_event_types: int = 18):
        self.predictor = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, n_event_types),
        )

    def forward(self, h_i: Tensor) -> Tensor:
        """
        h_i: [d] actor memory state
        Returns: [n_event_types] logits (apply softmax for distribution)
        """
        return self.predictor(h_i)

    def surprise_score(self, h_i: Tensor, actual_counts: Tensor) -> float:
        """
        KL divergence between predicted and actual event-type distribution.
        Interpretable: "expected CONSULT but saw THREATEN" = high surprise.
        """
        p_pred = F.softmax(self.predictor(h_i), dim=-1)
        p_actual = actual_counts / actual_counts.sum().clamp(min=1)
        return F.kl_div(p_pred.log(), p_actual, reduction="sum").item()
```

**Interpretability:** Unlike the CPC score (which operates in opaque embedding space), the event-type prediction gives human-readable diagnostics. You can inspect *which* event types the model expected vs. what occurred.

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

Time2Vec appears in two places:
1. **Text memory update:** encodes the current timestamp to condition the update on where we are in the temporal sequence.
2. **Event encoding:** encodes the event timestamp for the structured event stream.

Each use has its own Time2Vec instance with independent learned frequencies. The prediction head does not use time2vec because it outputs a full temporal curve rather than a single-horizon prediction.

---

## 9. Relation Embeddings

### 9.1 Structure

Each of the 18 PLOVER event types has a learnable dense embedding vector:

```python
relation_embeddings = nn.Embedding(18, d)  # [18, d]
```

These are shared across the event encoding (Layer 3) and the dyadic representation (Layer 5). Layer 4 (actor self-attention) does not use relation embeddings — it learns inter-actor structure directly from the memory vectors. Training discovers their geometry: event types that co-occur between similar dyads, or that tend to follow each other temporally, will cluster in the embedding space.

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
| ConfliBERT encoder (frozen) | 110M | Base weights frozen; LoRA adapters (~0.6M) trained during Phase 2–3 |
| Cross-attention (W_Q, W_K, W_V, W_cond) | ~2.4M | At d_k = d_v = 256 |
| Memory update MLP | ~0.8M | 2-layer, d × 4d × d |
| Scalar + dimensional gates | ~0.2M | |
| GRU source + target | ~1.6M | Two GRU cells, each d × d input |
| Actor self-attention (2 layers) | ~1.6M | 2 × (MHA + FFN + LayerNorm) |
| Survival head (shared trunk) | ~2.1M | 6d → 4d → 2d |
| Per-event-type hazard heads (18) | ~156K | 18 × (2d → 17 bins) |
| CPC prediction head | ~0.5M | W_pred (d→d→d_cpc) + W_target (768→d→d_cpc) |
| Event-type prediction head | ~0.1M | MLP (d→d→18) |
| Relation embeddings | ~5K | 18 × d |
| Baseline projections (W_struct, W_text) | ~0.2M | Shared projections for computing h_baseline_init_i |
| `alpha_baseline` | 1 | EMA baseline decay rate (shared scalar) |
| Time2Vec (2 instances) | ~64 | Tiny |
| **Total (excluding encoder)** | **~9.8M** | |
| **Total (including encoder)** | **~120M** | Encoder adapted via LoRA (~0.6M trainable) |

At d=256 and N_actors=500. Scales roughly as d² for most components.

---

## 11. Parameter Classification: Actor-Specific vs Shared

Every value in the model falls into one of three categories. Getting this wrong — especially carrying actor-specific state across epochs — would leak future information into earlier time periods.

### 11.1 Actor-Specific State (Reset Each Epoch)

These are **state variables**, not learned parameters. They are populated during the chronological rollout and reset at the start of each training epoch.

| Value | Shape | Description | In Optimizer? |
|-------|-------|-------------|---------------|
| `h_i(t)` | `[d]` per actor | Actor memory vector. Evolves through the rollout via Layers 2, 3, 4. | **No.** This is computed state, not a gradient-updated parameter. |
| `b_i(t)` | `[d]` per actor | EMA baseline vector. Updated once per day as a slow-moving average of `h_i`. Memory decays toward this. | **No.** This is computed state, evolved via the EMA update rule. |
| `t_last_updated_i` | scalar per actor | Timestamp of last memory update. Used for temporal decay. | No. |

**At epoch start:** For each actor i, set `h_i(0) = h_baseline_init_i` (the structural projection baseline), `b_i(0) = h_baseline_init_i`, and `t_last_updated_i = epoch_start_date`. Both memory and EMA baseline start from the same structural projection, then diverge as the rollout proceeds. Gradients flow through both the memory and baseline updates (within each TBPTT window), but neither is carried from one epoch to the next.

**Why reset?** Each epoch replays the full training period chronologically. If memories or baselines from the previous epoch persisted, the actor states at time t would contain information from events after t (from the previous epoch's rollout). This is temporal leakage. Resetting ensures the model can only use information from the past within each rollout.

### 11.2 Actor-Specific Computed Values (Deterministic, Not Directly Optimized)

These are **computed from fixed inputs through learned shared projections**. They are not directly in the optimizer — gradients flow through them to update the projections (W_struct, W_name, W_gate, etc.), but the per-actor vectors are never independently tuned.

| Value | Shape | Description | In Optimizer? |
|-------|-------|-------------|---------------|
| `h_baseline_init_i` | `[d]` per actor | Structural projection baseline. Initializes both `h_i(0)` and `b_i(0)` at epoch start. Computed from the actor's structural features and name encoding via learned projections (Component 2, Section 3.5). | **No.** Computed deterministically from fixed inputs. Gradients flow through to the shared projections. |
| `sketch_i` | `[sketch_dim]` per actor | TF-IDF sketch vector for text relevance filtering. | **No.** Recomputed periodically (see Component 2, Section 4.2). Not learned by gradient descent. |

**`h_baseline_init_i` details:** This is a deterministic function of the actor's fixed inputs (structural features, name) passed through learned shared projections:

```python
h_baseline_init_i = f(structural_features_i, name_i; W_struct, W_name, W_gate)
```

The projections (W_struct, W_name, W_gate) are shared across all actors and updated by the optimizer. The per-actor vectors are recomputed from these projections at the start of each epoch. This design prevents temporal leakage: if the initialization were a free parameter directly tuned by gradients from predictions at time t=2000, it could encode actor-specific future information that then initializes the rollout at t=0. Constraining the initialization to be a function of fixed inputs through shared projections ensures the model can only learn generalizable structure ("how to map structural features to useful starting geometry"), not actor-specific temporal signals.

**Relationship to the EMA baseline:** `h_baseline_init_i` provides the starting point for the EMA baseline `b_i(t)`. During the rollout, `b_i(t)` evolves as a slow-moving average of the actor's memory (see Section 2.2). The structural projection anchors where each actor *starts*; the EMA tracks where each actor *drifts to*. Memory always decays toward the current EMA baseline `b_i(t)`, not toward the static initialization.

### 11.3 Shared Learned Parameters (Persist Across Epochs)

All transformation weights are shared across actors. When the model applies `W_Q @ h_i` to compute an attention query, the same `W_Q` is used for every actor. This is critical — the model learns *how* to update memories, not *what* each actor's memory should be at a given time.

| Component | Parameters | In Optimizer? |
|-----------|-----------|---------------|
| ConfliBERT encoder (base) | 110M BERT weights | **No** (frozen) |
| ConfliBERT LoRA adapters | ~0.6M (rank-8 on Q/V, 12 layers) | Yes (lr=1e-4) |
| Cross-attention projections | W_Q (d × d_k), W_K (768 × d_k), W_V (768 × d_v) | Yes |
| Name encoding projection | W_name (768 × d), b_name, W_gate (2d × d) | Yes |
| Memory update gate/MLP | W_proj, W_scalar, W_dims, MLP_update | Yes |
| GRU cells | GRU_source, GRU_target (all internal weights) | Yes |
| Actor self-attention layers | MHA projections, FFN weights, LayerNorm (2 layers) | Yes |
| Survival head | Shared trunk MLP, 18 per-type hazard heads | Yes |
| Relation embeddings | Embedding(18, d) | Yes |
| Time2Vec | 2 instances, all frequency/phase parameters | Yes |
| `lambda_decay` | Scalar (shared) or per-type (18 values) | Yes |
| `alpha_baseline` | Scalar (shared). Constrained to (0.95, 1.0) via sigmoid reparameterization. | Yes |

**Key invariant:** No shared parameter depends on actor identity. The same GRU processes events for Russia and for Luxembourg. Actor-specific behavior emerges entirely from the actor-specific state (`h_i`, `b_i`, `h_baseline_init_i`) flowing through shared transformations.

### 11.4 Epoch Structure Summary

```
Epoch k:
  1. Recompute h_baseline_init_i from fixed inputs through updated projections for every actor i
  2. Reset all h_i(0) ← h_baseline_init_i
  3. Reset all b_i(0) ← h_baseline_init_i  (EMA baseline starts at structural projection)
  4. Reset all t_last_updated_i ← training_start_date
  5. Rollout chronologically through the training period:
     a. For each day t in [train_start, train_end]:
        - Process all articles for day t (Layer 2: actor reads document → gated update)
        - Process all structured events for day t (Layer 3: GRU update)
        - Run actor self-attention (Layer 4: all actors attend to all others)
        - Update EMA baselines: b_i(t) ← alpha_baseline * b_i(t-1) + (1 - alpha_baseline) * h_i(t)
        - Every K steps: compute loss, backprop (TBPTT), update shared params
  6. Evaluate on validation set
  7. Checkpoint shared parameters

Shared parameters (including baseline projections W_struct, W_name, W_gate and alpha_baseline) carry over to epoch k+1.
h_baseline_init_i is recomputed from fixed inputs through the updated projections at each epoch start.
h_i(t), b_i(t), and t_last_updated_i do NOT carry over — they are recomputed from scratch.
```
