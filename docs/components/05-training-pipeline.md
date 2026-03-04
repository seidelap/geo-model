# Component 5: Training Pipeline

## Purpose

Orchestrate the multi-phase training process that takes the model from untrained parameters to a production-ready prediction system. Each phase has specific objectives, data requirements, loss functions, and success criteria.

**Inputs:** Curated data (Component 1), actor registry with initial embeddings (Component 2), constructed training datasets (Component 3), model architecture (Component 4).

**Outputs:** Trained model weights at each phase checkpoint. Final model ready for validation (Component 6).

---

## 1. Phase Overview

```
Phase 0: Tabular Baseline     LightGBM on hand-crafted features      CPU, days
    ↓
Phase 1: Structural Pretrain   Initialize embeddings from structure   GPU, minutes
    ↓
Phase 2: Self-Supervised       Train encoder + memory on text         GPU, days
    ↓
Phase 3: Supervised Fine-Tune  End-to-end event prediction            GPU, days–weeks
```

Each phase builds on the previous. Phase 0 is a non-neural baseline that all subsequent phases must beat. Phases 1–3 progressively unlock model components.

---

## 2. Phase 0: Tabular Baseline (LightGBM)

### 2.1 Purpose

Establish a meaningful ML baseline using traditional features and gradient-boosted trees. This baseline:
- Validates the data pipeline (if the features can't predict events in a tree model, the data has problems)
- Sets the minimum bar for the neural architecture to beat
- Provides a strong ensemble member for later combination

### 2.2 Feature Engineering

Build a country-month panel with the following feature groups:

**Dyadic event features (from Component 1 structured events):**
```python
for r in PLOVER_TYPES:  # 18 types
    features[f"count_{r}_1m"] = event_count(i, j, r, window=30)
    features[f"count_{r}_3m"] = event_count(i, j, r, window=90)
    features[f"count_{r}_6m"] = event_count(i, j, r, window=180)
    features[f"count_{r}_12m"] = event_count(i, j, r, window=365)
    features[f"mean_goldstein_{r}_3m"] = mean_goldstein(i, j, r, window=90)
    features[f"trend_{r}_3m"] = (count_recent - count_prior) / max(count_prior, 1)
# Total: 18 × 6 = 108 dyadic event features
```

**Quad-class aggregates:**
```python
features["verbal_coop_3m"] = sum of CONSULT + ENGAGE + AGREE + COOP + AID counts
features["verbal_conf_3m"] = sum of DEMAND + DISAPPROVE + REJECT + THREATEN counts
features["material_coop_3m"] = sum of AID(actual) + COOP(actual) + YIELD counts
features["material_conf_3m"] = sum of FIGHT + SANCTION + SEIZE + MOBILIZE counts
# 4 quad classes × 4 windows = 16 features
```

**Structural features (from Component 2):**
```python
features["source_gdp"] = log_gdp(i)
features["target_gdp"] = log_gdp(j)
features["gdp_ratio"] = log_gdp(i) - log_gdp(j)
features["source_democracy"] = electoral_democracy(i)
features["target_democracy"] = electoral_democracy(j)
features["democracy_diff"] = abs(electoral_democracy(i) - electoral_democracy(j))
features["voeten_distance"] = voeten_distance(i, j)
features["alliance"] = has_alliance(i, j)
features["bilateral_trade"] = log_trade(i, j)
features["geographic_distance"] = log_distance(i, j)
features["contiguous"] = is_contiguous(i, j)
features["source_cinc"] = cinc_score(i)
features["target_cinc"] = cinc_score(j)
features["cinc_ratio"] = cinc_score(i) / max(cinc_score(j), 0.001)
# ~14 structural features
```

**Temporal features:**
```python
features["month"] = t.month  # seasonality
features["year"] = t.year
features["days_since_last_event"] = days_since_last(i, j, any_type)
features["days_since_last_conflict"] = days_since_last(i, j, conflict_types)
# 4 temporal features
```

**Total:** ~142 features per dyad-month observation.

### 2.3 Model Configuration

```python
import lightgbm as lgb

params = {
    "objective": "binary",           # or "poisson" for count targets
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "scale_pos_weight": auto,        # inverse class frequency
    "verbose": -1,
}

# Train one model per event type
for r in PLOVER_TYPES:
    model_r = lgb.train(
        params,
        train_data=lgb_train[r],
        valid_sets=[lgb_val[r]],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)],
    )
```

### 2.4 Success Criteria for Phase 0

- **Brier Skill Score (BSS) > 0 for at least 12 of 18 event types.** This means the model beats the climatological base rate.
- **Positive BSS against the no-change baseline** for at least 8 of 18 event types.
- **Feature importance analysis:** structural and recent-event features should dominate. If month/year features are most important, the model is fitting temporal trends rather than learning geopolitical dynamics.

If Phase 0 fails these criteria, the data pipeline has issues that must be resolved before proceeding to neural approaches.

### 2.5 Compute

CPU only. Training 18 LightGBM models on a few million dyad-month observations takes minutes. Feature engineering on the full event history takes hours (dominated by counting events in rolling windows). Total: ~1 day including feature engineering.

---

## 3. Phase 1: Structural Pretraining

### 3.1 Purpose

Train the structural projection matrices (W_struct, b_struct from Component 2, Section 3.2) and verify that the initial embedding space has reasonable geometry.

### 3.2 Objective

Given structural feature vectors for all states, learn a projection that places geopolitically similar states near each other:

```python
def structural_pretraining_loss(H_init: Tensor, structural_features: Tensor, events: list) -> Tensor:
    """
    H_init: [N_states, d] initial embeddings from structural projection
    """
    loss = 0.0

    # 1. Contrastive: states that interact frequently should be nearby
    for event in sample_events(events, n=1000):
        i, j = event.source_actor_id, event.target_actor_id
        h_i, h_j = H_init[i], H_init[j]
        pos_score = (h_i * h_j).sum()

        # Random negative
        k = random_actor(exclude=[i, j])
        neg_score = (h_i * H_init[k]).sum()

        loss += F.relu(1.0 - pos_score + neg_score)  # margin-based contrastive

    # 2. Regularization: don't collapse to a single point
    loss += 0.01 * (H_init.norm(dim=1).mean() - 1.0).pow(2)  # encourage unit-ish norm

    return loss
```

### 3.3 Procedure

1. Initialize H from PCA of structural features (deterministic).
2. Run 100–500 gradient steps with Adam (lr=1e-3) on the contrastive loss.
3. Verify: plot t-SNE or UMAP of the resulting embeddings. Check that NATO allies cluster, that authoritarian states cluster, that geographic neighbors are nearby.

### 3.4 Compute

Minutes on a single GPU. This is a tiny model (just the projection matrices, no encoder).

---

## 4. Phase 2: Self-Supervised Text Pretraining

### 4.1 Purpose

Train the text encoder (ConfliBERT) and memory update mechanism (Layer 2 of the architecture) before introducing supervised event labels. This phase teaches the model to:
- Extract geopolitically meaningful representations from text
- Update actor memories from documents in a way that captures who is mentioned, in what context, and how actors relate to each other in the text

### 4.2 Training Objectives

Three self-supervised objectives, weighted and summed:

**Objective A — Masked entity prediction:**

Mask actor name spans in documents. Predict the masked actor from surrounding context plus the memory states of other actors mentioned.

```python
def masked_entity_loss(article: str, actor_ids: list[str], H: Tensor) -> Tensor:
    """
    Mask one actor mention, predict which actor it is from context.
    """
    # Choose an actor to mask
    masked_actor = random.choice(actor_ids)
    masked_text = mask_actor_spans(article, masked_actor)  # replace name with [MASK]

    # Encode masked document
    T_masked = conflibert(masked_text)  # [seq_len, 768]

    # Pool representations at mask positions
    mask_positions = find_mask_positions(masked_text)
    mask_repr = T_masked[mask_positions].mean(dim=0)  # [768]

    # Predict actor from mask representation + other actors' memories
    other_memories = torch.stack([H[a] for a in actor_ids if a != masked_actor])
    context = torch.cat([mask_repr, other_memories.mean(dim=0)])
    logits = actor_classifier(context)  # [N_actors]

    return F.cross_entropy(logits, actor_index[masked_actor])
```

**Why this works:** If the model must identify "Russia" from context like "military exercises near Ukrainian border…", it must encode that such events are associated with a specific cluster of actors. This forces the memories to capture geopolitical meaning.

**Objective B — Temporal ordering:**

Given two paragraphs from different time points, predict which came first.

```python
def temporal_ordering_loss(article_a: str, article_b: str, t_a: date, t_b: date) -> Tensor:
    """
    Binary classification: did article_a come before article_b?
    """
    repr_a = conflibert(article_a).mean(dim=1)  # [768]
    repr_b = conflibert(article_b).mean(dim=1)  # [768]

    logit = temporal_classifier(torch.cat([repr_a, repr_b]))  # scalar
    label = float(t_a < t_b)

    return F.binary_cross_entropy_with_logits(logit, torch.tensor(label))
```

**Objective C — Contrastive document similarity:**

Articles describing the same event should have similar representations. Use GDELT/POLECAT event codes as weak supervision: articles that generate the same structured event on the same day are positive pairs.

```python
def contrastive_doc_loss(anchor: str, positive: str, negatives: list[str], temperature: float = 0.07) -> Tensor:
    """
    InfoNCE contrastive loss. Anchor and positive describe the same event; negatives don't.
    """
    z_anchor = F.normalize(conflibert(anchor).mean(dim=1), dim=-1)
    z_pos = F.normalize(conflibert(positive).mean(dim=1), dim=-1)
    z_negs = torch.stack([F.normalize(conflibert(n).mean(dim=1), dim=-1) for n in negatives])

    pos_score = (z_anchor * z_pos).sum() / temperature
    neg_scores = (z_anchor @ z_negs.T) / temperature

    logits = torch.cat([pos_score.unsqueeze(0), neg_scores])
    labels = torch.tensor(0)  # positive is at index 0

    return F.cross_entropy(logits, labels)
```

### 4.3 Combined Phase 2 Loss

```python
L_phase2 = L_masked_entity + 0.5 * L_temporal_ordering + 0.3 * L_contrastive_doc
```

Weights are tuned on a held-out set of downstream event prediction performance (using the Phase 0 LightGBM with ConfliBERT embeddings as additional features).

### 4.4 Data Sampling

- Process articles in chronological order to simulate the streaming setting.
- Batch size: 32 articles per GPU step.
- For masked entity prediction: sample articles that mention ≥2 actors (so there's meaningful context from the non-masked actors).
- For temporal ordering: sample pairs from the same actor dyad, separated by 7–365 days.
- For contrastive pairs: match articles via POLECAT event codes.

### 4.5 Training Schedule

- **Optimizer:** AdamW, lr=2e-5 for ConfliBERT (standard BERT fine-tuning rate), lr=1e-3 for memory update parameters.
- **Warmup:** Linear warmup over first 10% of steps.
- **Duration:** 2–5 epochs over the training period text corpus (40K articles/day × ~2,500 training days ≈ 100M articles, but heavily subsampled to ~5M for Phase 2).
- **Early stopping:** Monitor masked entity prediction accuracy on held-out articles. Stop when it plateaus.

### 4.6 Success Criteria

- Masked entity prediction accuracy ≥ 50% (random baseline: 1/N_actors ≈ 0.2%).
- t-SNE of memory vectors at phase end shows meaningful geopolitical clustering: NATO allies together, BRICS together, conflict dyads separated from cooperative dyads.
- ConfliBERT embeddings (mean-pooled article representations) added as features to the Phase 0 LightGBM improve BSS on ≥10 of 18 event types.

### 4.7 Compute

- ConfliBERT fine-tuning on 5M articles: ~20–40 GPU-hours on A10G.
- At spot pricing (~$1/hour): ~$20–40.
- With memory update training overhead: ~$30–60 total.

---

## 5. Phase 3: Supervised Event Prediction Fine-Tuning

### 5.1 Purpose

Fine-tune the entire model end-to-end on the event prediction task. This is where the model learns to predict events from actor states.

### 5.2 What Gets Trained

All parameters are trainable, with different learning rates:

| Component | Learning Rate | Rationale |
|-----------|--------------|-----------|
| ConfliBERT encoder | 1e-5 | Small LR: preserve pretraining; just fine-tune |
| Memory update parameters | 5e-4 | Medium: needs to adapt to supervised signal |
| Graph attention layers | 5e-4 | Medium: new component, learning from scratch |
| Survival/hazard heads | 1e-3 | Larger: top-level heads, most supervised gradient |
| Hawkes excitation parameters | 1e-3 | Larger: learning temporal dynamics |
| Intensity head (high-freq types) | 1e-3 | Larger: learning rate dynamics |
| Relation embeddings | 5e-4 | Medium: should shift to capture prediction structure |

### 5.3 Training Loop: Chronological Rollout

Training processes the full training period day by day. Within each day, events and articles update actor memories. At the end of each day, graph propagation runs. Prediction losses are computed at TBPTT boundaries.

```python
def phase3_epoch(model: FullModel, data: ChronologicalStream, K: int = 75):
    """
    One epoch of Phase 3: chronological rollout over the training period.

    The rollout simulates the passage of time. Each day:
      1. Activate/deactivate actors whose lifecycle boundaries fall on this day
      2. Process all articles for this day (Layer 2)
      3. Process all structured events for this day (Layer 3)
      4. Run graph propagation (Layer 4)
      5. At TBPTT boundaries: compute losses, backprop, update parameters
    """
    # --- Epoch initialization (see Component 4, Section 11) ---
    # Reset all actor memories to their learned baselines
    for actor in registry.all_actors():
        model.H[actor.actor_id] = model.h_baseline[actor.actor_id].clone()
        model.t_last[actor.actor_id] = data.start_date

    # Track which actors are currently active
    active_actors = set()
    memory_step_count = 0

    for day in data.iter_days():  # each day in [train_start, train_end]

        # --- 1. Actor lifecycle: activate/deactivate ---
        newly_active, newly_retired = registry.lifecycle_changes_on(day)
        for actor in newly_active:
            activate_actor(model, actor, day)
            active_actors.add(actor.actor_id)
        for actor in newly_retired:
            deactivate_actor(model, actor, day)
            active_actors.discard(actor.actor_id)

        # --- 2. Process articles for this day (Layer 2) ---
        for article in day.articles:
            relevant_actors = model.filter_relevant_actors(article, active_actors)
            T = model.encode_document(article)
            for actor_id, weight in relevant_actors:
                m_i = model.actor_reads_document(model.H[actor_id], T)
                model.H[actor_id] = model.update_memory_from_text(
                    model.H[actor_id], m_i, day.date, model.t_last[actor_id]
                )
                model.t_last[actor_id] = day.date
            memory_step_count += 1

        # --- 3. Process structured events for this day (Layer 3) ---
        for event in day.events:
            if event.source_actor_id in active_actors and event.target_actor_id in active_actors:
                model.H[event.source_actor_id], model.H[event.target_actor_id] = \
                    model.update_from_event(event)
                model.t_last[event.source_actor_id] = day.date
                model.t_last[event.target_actor_id] = day.date
            memory_step_count += 1

        # --- 4. Graph propagation (once per day, Layer 4) ---
        active_H, active_edges = build_active_subgraph(model.H, active_actors, day.date)
        updated_H = model.graph_propagation(active_H, *active_edges)
        write_back_active(model.H, updated_H, active_actors)

        # --- 5. TBPTT: compute losses and backprop at boundaries ---
        if memory_step_count >= K:
            losses = compute_prediction_losses(day, model, active_actors)
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.detach_memories()  # truncate gradient path
            memory_step_count = 0
```

### 5.4 Actor Lifecycle During Training

Actors enter and leave the system during the chronological rollout. The model must handle this explicitly — not all 500 actors exist at all points in time.

#### 5.4.1 Activating an Actor

When the rollout reaches an actor's `active_from` date:

```python
def activate_actor(model: FullModel, actor: Actor, date: date):
    """
    Bring a new actor into the model during the training rollout.

    Called when the rollout date reaches actor.active_from.
    """
    # Initialize memory from learned baseline
    model.H[actor.actor_id] = model.h_baseline[actor.actor_id].clone()
    model.t_last[actor.actor_id] = date

    # The actor now participates in:
    # - Text processing: its sketch vector is included in relevance filtering
    # - Event processing: events involving it trigger GRU updates
    # - Graph propagation: it receives messages from neighbors
    # - Prediction queries: dyads involving it are now queryable
```

**Where `h_baseline` comes from for new actors:** The baseline is initialized from Component 2 (structural projection for states, text-derived for leaders/orgs, neighbor average as fallback). During training, the optimizer updates `h_baseline` via gradients — but only from time steps *after* the actor is active.

#### 5.4.2 Deactivating an Actor

When the rollout reaches an actor's `active_to` date:

```python
def deactivate_actor(model: FullModel, actor: Actor, date: date):
    """
    Remove an actor from active participation during the training rollout.

    The actor's memory is frozen but retained — it is needed for
    historical graph edges and for any training examples that reference
    this actor's state at earlier time points.
    """
    # Freeze: stop updating this actor's memory
    model.H[actor.actor_id] = model.H[actor.actor_id].detach()

    # The actor no longer participates in:
    # - New text/event processing (no more memory updates)
    # - Graph propagation (excluded from active subgraph)
    # - New prediction queries (dyads involving it are not sampled)
    #
    # But its frozen state is still available:
    # - As a graph neighbor for historical edge construction
    # - As source/target state for loss computation on examples
    #   with reference dates before the deactivation
```

#### 5.4.3 Leader Succession Example

The most common lifecycle event is a leadership change:

```
Day 1021 (2023-01-20): leader:USA_POTUS_45 reaches active_to
  → deactivate_actor(model, "leader:USA_POTUS_45", day)
  → Freeze memory at current state

Day 1021 (2023-01-20): leader:USA_POTUS_46 reaches active_from
  → activate_actor(model, "leader:USA_POTUS_46", day)
  → Initialize from h_baseline (text-derived from speeches, biography)

Note: state:USA is unaffected — it remains active throughout.
The state's memory continues evolving; the new leader's actions
will gradually shift the state embedding through events/articles.
```

#### 5.4.4 Gradient Flow for Actor-Specific Baselines

When computing losses for training examples that involve actor i:
- Gradients flow through `h_i(t)` back through the memory update chain (within the TBPTT window)
- At the start of the TBPTT window, `h_i` was either: (a) the output of a previous TBPTT window (detached), or (b) `h_baseline_i` (at epoch start or actor activation)
- In case (b), the gradient reaches `h_baseline_i` and updates it. This is how the optimizer learns each actor's resting state.
- Actors that appear in more training examples receive more gradient signal for their baselines. This is correct — we have more information about those actors.

### 5.5 Curriculum Learning

Don't present all event types at full weight from the start. Gradually increase the weight on rare events:

```python
def curriculum_weight(event_type: str, step: int, total_steps: int, warmup_frac: float = 0.2) -> float:
    """
    Ramp up rare-event weights over the first 20% of training.
    """
    base_freq = event_base_rates[event_type]
    rarity = 1.0 / max(base_freq, 0.001)

    progress = min(1.0, step / (warmup_frac * total_steps))
    # At step 0: all events weighted equally (rarity factor = 1)
    # At warmup complete: full inverse-frequency weighting applied
    weight = 1.0 + (min(rarity, 100.0) - 1.0) * progress
    return weight
```

**Phase schedule:**
1. Steps 0–20% of training: curriculum warmup, increasing rare event weight
2. Steps 20–80%: full training at target weights
3. Steps 80–100%: reduce learning rate by 10x (cosine decay), continue training

### 5.6 Truncated Backpropagation Through Time (TBPTT)

The model processes events and articles chronologically, updating actor memories at each step. Full backpropagation through the entire history is infeasible. Instead, truncate gradients:

```python
def train_with_tbptt(data_stream: Iterator, model: FullModel, K: int = 75):
    """
    Process K memory update steps, compute loss, backprop, then detach.
    K=50-100 is typical. Higher K = longer gradient paths = better long-range learning but more memory.
    """
    step = 0
    for item in data_stream:
        if isinstance(item, NormalizedEvent):
            model.update_from_event(item)
        elif isinstance(item, CuratedArticle):
            model.update_from_text(item)

        step += 1

        if step % K == 0:
            # Compute prediction losses on recent events
            losses = compute_prediction_losses(recent_predictions)
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Detach memory vectors to truncate gradient path
            model.detach_memories()
```

### 5.7 Auxiliary Losses at Intermediate Timesteps

To create shorter gradient paths and prevent vanishing gradients, add prediction losses not just at the final step but also at intermediate checkpoints:

```python
# Every M steps within each TBPTT window, compute an intermediate prediction loss
M = K // 5  # e.g., every 15 steps within a 75-step window

if step % M == 0:
    intermediate_loss = compute_prediction_losses(sample_predictions)
    intermediate_loss *= 0.3  # lower weight than final loss
    total_loss += intermediate_loss
```

### 5.8 Training Schedule

- **Optimizer:** AdamW with separate parameter groups (different LRs per component, see Section 5.2).
- **Batch size:** Process events/articles in chronological order. Each TBPTT window of K=75 steps constitutes one "batch."
- **Total steps:** ~100K TBPTT windows (≈ 7.5M memory update steps covering the training period multiple times).
- **Learning rate schedule:** Linear warmup (1000 steps) → constant → cosine decay (last 20%).
- **Gradient clipping:** Max norm 1.0 to prevent exploding gradients from long temporal chains.
- **Early stopping:** Monitor validation concordance index (C-index) and Brier scores derived from the CDF. Stop if no improvement for 10 evaluation epochs.

### 5.9 Evaluation During Training

Every 5,000 steps, run evaluation on the validation set (Component 3, Section 6):
- Derive fixed-horizon Brier scores from the survival CDF at standard horizons (7, 30, 90, 180 days).
- Compute concordance index (C-index) for event ordering quality.
- Compute BSS against Phase 0 baseline at the 30-day horizon (primary comparison).
- Log reliability diagrams for visual inspection at each standard horizon.
- If BSS is worse than Phase 0 baseline for >50% of event types at the 30-day horizon, something is wrong — investigate before continuing.

### 5.10 Success Criteria

- **BSS > Phase 0 BSS at 30-day horizon for ≥14 of 18 event types** (the neural model should improve on the tree baseline for most types).
- **Aggregate Brier score improvement ≥ 5% over Phase 0 at 30-day horizon** (weighted by event type importance).
- **C-index ≥ 0.70** for survival-target event types (model correctly orders dyads by time-to-event).
- **Calibration:** ECE < 0.10 at all standard horizons before post-hoc calibration (Layer 6 will improve this further).
- **No regression:** The model should not be worse than Phase 0 on any event type by more than 10% relative BSS at the 30-day horizon. If it is, that type needs debugging.

### 5.11 Compute

- Full Phase 3 training: ~50–100 GPU-hours on A10G.
- At spot pricing: ~$50–100.
- With hyperparameter search (5–10 runs): ~$250–500.

---

## 6. Epoch Structure and Memory Reset

### 6.1 What Happens Between Epochs

Each epoch replays the entire training period from `train_start` to `train_end`. Between epochs:

**Reset (actor-specific state):**
- All `h_i(t)` → `h_baseline_i` (learned baseline)
- All `t_last_updated_i` → `train_start_date`
- Active actor set → recomputed from scratch based on `active_from` dates

**Persist (learned parameters):**
- `h_baseline_i` for every actor (updated by optimizer during the epoch)
- All shared weights (encoder, projections, gates, GRUs, heads, etc.)
- Optimizer state (momentum, adaptive learning rates)

### 6.2 Why Memory Reset Is Necessary

The model's value comes from learning shared transformations (how to update memories, how to propagate information, how to predict events), not from memorizing actor states at particular times. If actor memories carried over between epochs:

1. **Temporal leakage:** At the start of epoch k+1, `h_Russia(t=0)` would contain information from events at `t=2000 days` (end of epoch k). The model would start with future knowledge baked into its states.
2. **Gradient confusion:** The optimizer would receive conflicting signals — gradients from early in the rollout would push memories one way, but the carried-over state already reflects the end of the previous epoch.
3. **Baseline learning failure:** The optimizer would have no incentive to learn good `h_baseline_i` values, since memories would never actually start from baseline.

By resetting, each epoch provides a clean forward pass. The only information that persists is encoded in the shared weights and baselines — which is exactly the generalizable knowledge we want.

### 6.3 Practical Considerations

- **Epoch count:** 3–5 epochs is typical. More epochs risk overfitting the shared weights to the training period's specific event sequence.
- **Stochasticity across epochs:** The event/article ordering within a single day can be shuffled between epochs (since intra-day ordering is arbitrary). This provides mild data augmentation without violating temporal ordering.
- **Checkpoint strategy:** Save shared weights + baselines after each epoch. The best checkpoint is selected on validation performance (see Section 5.9).

---

## 7. Loss Functions (Detailed)

### 7.1 DeepHit Survival Loss (Primary — Survival-Target Types)

```python
def deephit_loss(
    pred: dict,
    event_time_bin: int,
    censored: bool,
    event_type_weight: float,
    eta: float = 0.5,
) -> Tensor:
    """
    DeepHit: NLL + ranking loss for survival prediction.

    For uncensored examples: maximize P(event in observed bin).
    For censored examples: maximize P(survival to censoring time).
    """
    pdf = pred["pdf"]         # [K] probability mass per bin
    survival = pred["survival"]  # [K] survival function

    if not censored:
        # NLL: negative log probability of event at the observed time bin
        L_nll = -torch.log(pdf[event_time_bin] + 1e-8)
    else:
        # Censored: event didn't happen yet. Maximize survival to censoring time.
        L_nll = -torch.log(survival[event_time_bin] + 1e-8)

    # Ranking loss (computed across a batch): subjects that experienced events
    # earlier should have lower survival values. Ensures correct ordering.
    # (Implemented at batch level in the training loop, omitted here for clarity.)
    L_ranking = 0.0  # placeholder; computed by batch-level ranking function

    return L_nll + eta * L_ranking
```

### 7.2 Hawkes NLL (Primary — Intensity-Target Types)

```python
def hawkes_nll(
    intensity_head: IntensityHead,
    h_i: Tensor, h_j: Tensor,
    event_type: int,
    event_times: list[float],
    T_window: float,
) -> Tensor:
    """
    Negative log-likelihood of a Hawkes process over a time window.
    Used for high-frequency event types modeled as rates.

    event_times: exact event times within the window (days from window start)
    T_window: length of observation window in days
    """
    # Log-likelihood of observed events: sum of log-intensity at each event time
    if len(event_times) > 0:
        event_t = torch.tensor(event_times)
        lambda_at_events = intensity_head(h_i, h_j, event_times, event_type, 0.0, event_t)
        ll = torch.log(lambda_at_events + 1e-8).sum()
    else:
        ll = torch.tensor(0.0)

    # Integral of intensity (compensator) — approximated by trapezoidal rule
    t_grid = torch.linspace(0, T_window, 200)
    lambda_grid = intensity_head(h_i, h_j, event_times, event_type, 0.0, t_grid)
    integral = torch.trapezoid(lambda_grid, t_grid)

    return -(ll - integral)
```

### 7.3 Auxiliary Losses

```python
# Goldstein scale regression (for uncensored survival examples)
L_goldstein = F.mse_loss(pred["goldstein_pred"], event_goldstein)

# Memory regularization: prevent memory vectors from exploding
L_mem = lambda_mem * (h_i.norm().pow(2) + h_j.norm().pow(2))  # λ_mem = 0.01

# Gate sparsity: encourage sparse memory updates
L_gate = lambda_gate * model.get_gate_penalty()  # λ_gate = 0.01
```

### 7.4 Total Loss

The total loss depends on example type:

```python
# For survival-target event types (rare/moderate):
L_survival_example = (
    L_deephit                            # primary: survival NLL + ranking
    + lambda_1 * L_goldstein             # intensity regression (λ₁ = 0.1)
    + L_mem + L_gate                     # regularization
)

# For intensity-target event types (high-frequency):
L_intensity_example = (
    L_hawkes_nll                         # primary: Hawkes process NLL
    + L_mem + L_gate                     # regularization
)
```

Both are weighted by `temporal_weight * event_type_weight` before aggregation.

---

## 8. Checkpointing and Reproducibility

### 8.1 Checkpoint Contents

Every evaluation epoch, save:

| Artifact | Contents |
|----------|----------|
| `model_weights.pt` | All shared model parameters (encoder, projections, heads, etc.) |
| `actor_baselines.pt` | Per-actor `h_baseline_i` vectors (the only actor-specific learned values) |
| `optimizer_state.pt` | Optimizer state (for resuming training) |
| `training_config.json` | All hyperparameters, random seeds, data version |
| `metrics.json` | Validation metrics at this checkpoint |

Note: actor memory vectors `h_i(t)` are **not** checkpointed between epochs — they are recomputed from baselines during each rollout. They are only checkpointed mid-epoch for crash recovery.

### 8.2 Reproducibility

- **Random seeds:** Fix PyTorch, NumPy, and Python random seeds at the start of each phase.
- **Data ordering:** Events and articles are processed in strict chronological order (deterministic).
- **Data version:** Record the data manifest hash from Component 1 in the training config.
- **Code version:** Record the git commit hash.

### 8.3 Best Model Selection

After Phase 3 training completes:
1. Identify the checkpoint with the best **aggregate validation metric**: weighted combination of C-index (survival types) and Hawkes NLL (intensity types), plus Brier score at 30-day horizon derived from the CDF.
2. Load that checkpoint's shared weights and actor baselines.
3. For validation/test evaluation: run a fresh chronological rollout through the relevant period, starting from baselines. This mirrors production inference.
4. Pass to Component 6 (Validation & Calibration) for final evaluation on the held-out test set.
