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

Five self-supervised objectives, weighted and summed:

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

**Objective D — Contrastive Predictive Coding (CPC):**

Given an actor's current memory state, distinguish the actor's actual next-day article aggregate from negatives. This forces the memory to be *generatively predictive* of the information stream, not merely a summary of past inputs.

```python
def cpc_loss(
    H: dict[str, Tensor],
    tomorrow_aggregates: dict[str, Tensor],
    day: date,
    temperature: float = 0.07,
    n_negatives: int = 15,
) -> Tensor:
    """
    Contrastive Predictive Coding: actor memory predicts next-day article signal.

    Run after Layer 4 (actor self-attention) each day, using tomorrow's
    aggregate article embedding as the positive target.

    H: current actor memory states (post self-attention)
    tomorrow_aggregates: {actor_id: mean-pooled ConfliBERT embedding of tomorrow's articles}
    """
    loss = torch.tensor(0.0)
    count = 0

    for actor_id, h_i in H.items():
        if actor_id not in tomorrow_aggregates:
            continue  # no news tomorrow for this actor — skip

        # Project memory into prediction space
        z_pred = F.normalize(W_cpc_pred(h_i), dim=-1)  # [d_cpc]

        # Positive: tomorrow's actual aggregate embedding for this actor
        positive = F.normalize(W_cpc_target(tomorrow_aggregates[actor_id]), dim=-1)

        # Negatives (three types, ~5 each):
        # 1. Different actor, same day: forces encoding of *who* specifically
        other_actors = [a for a in tomorrow_aggregates if a != actor_id]
        neg_diff_actor = random.sample(other_actors, min(5, len(other_actors)))

        # 2. Same actor, different time period: forces encoding of *when* in trajectory
        neg_diff_time = sample_from_other_dates(actor_id, day, n=5)

        # 3. Random: easy negatives for training stability
        neg_random = sample_random_aggregates(n=5)

        all_negatives = [tomorrow_aggregates[a] for a in neg_diff_actor] + neg_diff_time + neg_random
        neg_embeddings = torch.stack([F.normalize(W_cpc_target(n), dim=-1) for n in all_negatives])

        # InfoNCE loss
        pos_score = (z_pred * positive).sum() / temperature
        neg_scores = (z_pred @ neg_embeddings.T) / temperature
        logits = torch.cat([pos_score.unsqueeze(0), neg_scores])
        loss += F.cross_entropy(logits, torch.tensor(0))
        count += 1

    return loss / max(count, 1)
```

**Why CPC and not point prediction:** There are many possible "next articles" for a given actor — an article about trade, about military exercises, about diplomatic meetings, or no article at all. Predicting a single point in embedding space would just converge to the uninformative mean. CPC sidesteps this: it only asks the memory to *distinguish* relevant future articles from irrelevant ones, which is well-defined even when the future is stochastic.

**Three negative types, in order of importance:**
1. **Different actor, same day** — prevents the model from encoding only "what's happening globally" rather than actor-specific state
2. **Same actor, different time period** — prevents the model from encoding only actor identity rather than temporal trajectory
3. **Random** — easy negatives for numerical stability during early training

**Surprise signal at inference time:** The CPC score (cosine similarity between predicted and actual embedding) provides a natural anomaly detection signal. Low score = the model was surprised by what it read. This is available as a feature for downstream prediction heads without any additional training.

**Objective E — Categorical event-type prediction:**

From memory state, predict the distribution of CAMEO/PLOVER event types the actor will be involved in during the next time window. This provides an interpretable "surprise" signal and a direct bridge between the self-supervised memory and the supervised prediction task.

```python
def event_type_prediction_loss(
    H: dict[str, Tensor],
    next_window_events: dict[str, Tensor],
    n_event_types: int = 18,
) -> Tensor:
    """
    Predict distribution over event types in the next 7-day window.

    H: current actor memory states
    next_window_events: {actor_id: [n_event_types] count vector of events in next 7 days}
    """
    loss = torch.tensor(0.0)
    count = 0

    for actor_id, h_i in H.items():
        if actor_id not in next_window_events:
            continue

        # Predict event-type distribution from memory
        logits = event_type_predictor(h_i)  # [n_event_types]
        p_predicted = F.softmax(logits, dim=-1)

        # Actual distribution (normalized counts from structured event stream)
        counts = next_window_events[actor_id]  # [n_event_types]
        if counts.sum() == 0:
            continue  # no events in window
        p_actual = counts / counts.sum()

        # KL divergence: how surprised would the model be?
        loss += F.kl_div(p_predicted.log(), p_actual, reduction="batchmean")
        count += 1

    return loss / max(count, 1)
```

**Why this complements CPC:** CPC operates in embedding space (high-dimensional, hard to interpret). Event-type prediction operates in a discrete, interpretable space. Together they provide both a rich training signal (CPC) and an interpretable diagnostic (event-type prediction). At inference time, you can literally say "the model expected CONSULT events but saw THREATEN — surprise score 3.2."

**Weak supervision source:** Event-type labels come from the GDELT/POLECAT structured event stream (Component 1), not from human annotation. This makes it free to compute at scale.

### 4.3 Combined Phase 2 Loss

```python
L_phase2 = (
    1.0 * L_masked_entity
    + 0.5 * L_temporal_ordering
    + 0.3 * L_contrastive_doc
    + 0.5 * L_cpc
    + 0.3 * L_event_type_pred
)
```

Weights are tuned on a held-out set of downstream event prediction performance (using the Phase 0 LightGBM with ConfliBERT embeddings as additional features).

**Design rationale for weights:** Masked entity prediction (1.0) is the primary objective — it most directly forces geopolitically meaningful memory representations. CPC (0.5) is weighted equally with temporal ordering because both test temporal dynamics of the memory. Event-type prediction (0.3) and contrastive docs (0.3) are auxiliary signals that complement the primary objectives.

### 4.4 Data Sampling

- Process articles in chronological order to simulate the streaming setting.
- Batch size: 32 articles per GPU step.
- For masked entity prediction: sample articles that mention ≥2 actors (so there's meaningful context from the non-masked actors).
- For temporal ordering: sample pairs from the same actor dyad, separated by 7–365 days.
- For contrastive pairs: match articles via POLECAT event codes.
- For CPC: compute per-actor daily aggregate embeddings (mean-pooled ConfliBERT over all articles mentioning actor i on day t). CPC loss is computed after each day's Layer 4 pass, using tomorrow's aggregates as positives. Negatives are sampled from the same batch.
- For event-type prediction: aggregate PLOVER event counts per actor over a 7-day forward window. Labels come from the structured event stream (Component 1), not from human annotation.

**Temporal leakage constraint:** Phase 2 pretraining must use only articles from the **training time period**. Articles from the validation or test periods must not appear in Phase 2, even though Phase 2 is self-supervised. The encoder fine-tuned on future articles could learn distributional patterns specific to the test period, creating subtle leakage. Enforce this by filtering the article corpus to `[train_start, train_end]` before Phase 2 begins.

### 4.5 Training Schedule

- **Optimizer:** AdamW, lr=1e-4 for ConfliBERT LoRA adapters, lr=1e-3 for memory update parameters.
- **Warmup:** Linear warmup over first 10% of steps.
- **Duration:** 2–5 epochs over the training period text corpus (40K articles/day × ~2,500 training days ≈ 100M articles, but heavily subsampled to ~5M for Phase 2).
- **Early stopping:** Monitor masked entity prediction accuracy on held-out articles. Stop when it plateaus.

### 4.6 Success Criteria

- Masked entity prediction accuracy ≥ 50% (random baseline: 1/N_actors ≈ 0.2%).
- **CPC accuracy ≥ 70%** on held-out days (can the memory distinguish next-day articles from negatives? Random baseline: 1/(1+n_negatives) ≈ 6%).
- **Event-type prediction KL divergence** decreasing over training, indicating the memory learns to anticipate what kinds of events actors will be involved in.
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

### 5.2 Phase 2 Auxiliary Loss Carryover

Phase 2's CPC and event-type prediction objectives carry over into Phase 3 as auxiliary losses. This serves two purposes:
1. **Regularization:** Prevents the memory from overfitting to just the 18 target event types — the CPC loss keeps the memory generally predictive of the information stream.
2. **Catastrophic forgetting prevention:** Without these auxiliary signals, fine-tuning on supervised losses can degrade the general-purpose representations learned in Phase 2.

```python
# Phase 3 total loss (per TBPTT window):
L_phase3 = (
    L_supervised                            # DeepHit + Goldstein (see Section 7)
    + 0.1 * L_cpc                           # carried from Phase 2, downweighted 5x
    + 0.1 * L_event_type_pred               # carried from Phase 2, downweighted 3x
    + L_mem + L_gate                        # regularization (unchanged)
)
```

The 0.1 weight ensures these losses influence training without dominating the supervised signal. The CPC and event-type prediction heads continue to be trained in Phase 3 — their parameters are included in the memory update parameter group (lr=5e-4).

**Surprise features for prediction heads:** The CPC score and event-type prediction error from the most recent timestep are concatenated to the dyadic representation (Section 6.1 in Component 4) as additional features for the survival head. This gives the prediction heads access to "how surprised was the model by recent inputs for this actor" — a signal that is naturally correlated with escalation and de-escalation dynamics.

### 5.3 What Gets Trained

All parameters are trainable, with different learning rates:

| Component | Learning Rate | Rationale |
|-----------|--------------|-----------|
| ConfliBERT LoRA adapters | 1e-4 | Moderate LR: LoRA adapters on frozen encoder |
| Memory update parameters | 5e-4 | Medium: needs to adapt to supervised signal |
| CPC + event-type prediction heads | 5e-4 | Medium: continuing Phase 2 objectives as auxiliary |
| Actor self-attention layers | 5e-4 | Medium: new component, learning from scratch |
| Survival/hazard heads | 1e-3 | Larger: top-level heads, most supervised gradient |
| Relation embeddings | 5e-4 | Medium: should shift to capture prediction structure |

### 5.3 Training Loop: Chronological Rollout

Training processes the full training period day by day. Within each day, events and articles update actor memories. At the end of each day, actor self-attention runs. Prediction losses are computed at TBPTT boundaries.

```python
def phase3_epoch(model: FullModel, data: ChronologicalStream, K: int = 75):
    """
    One epoch of Phase 3: chronological rollout over the training period.

    K is measured in simulated days (not individual article/event steps).
    Each day:
      1. Activate/deactivate actors whose lifecycle boundaries fall on this day
      2. Process all articles for this day (Layer 2) — no temporal decay within day
      3. Process all structured events for this day (Layer 3)
      4. Daily maintenance: temporal decay, self-attention (Layer 4), EMA baseline update
      5. CPC + event-type prediction (auxiliary losses)
      6. At TBPTT boundaries (every K days): compute losses, backprop, update parameters
    """
    # --- Epoch initialization (see Component 4, Section 11) ---
    for actor in registry.all_actors():
        model.H[actor.actor_id] = model.h_baseline[actor.actor_id].clone()
        model.B[actor.actor_id] = model.h_baseline[actor.actor_id].clone()
        model.t_last_decay[actor.actor_id] = data.start_date

    active_actors = set()
    day_count = 0

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
                    model.H[actor_id], m_i, day.date
                )

        # --- 3. Process structured events for this day (Layer 3) ---
        for event in day.events:
            if event.source_actor_id in active_actors and event.target_actor_id in active_actors:
                model.H[event.source_actor_id], model.H[event.target_actor_id] = \
                    model.update_from_event(event)

        # --- 4. Daily maintenance (Component 4, Section 5.5) ---
        model.daily_maintenance(active_actors, day.date)
        day_count += 1

        # --- 5. CPC + event-type prediction (auxiliary, carried from Phase 2) ---
        if day.has_next_day:
            L_cpc = cpc_loss(model.H, day.next_day.article_aggregates, day.date)
            L_event_pred = event_type_prediction_loss(model.H, day.next_day.event_counts)
        else:
            L_cpc = L_event_pred = torch.tensor(0.0)

        # --- 6. TBPTT: compute losses and backprop every K days ---
        if day_count >= K:
            L_supervised = compute_prediction_losses(day, model, active_actors)
            losses = L_supervised + 0.1 * L_cpc + 0.1 * L_event_pred
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            model.detach_memories()  # truncate gradient path
            day_count = 0
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
    # - Actor self-attention: it attends to and is attended by all other active actors
    # - Prediction queries: dyads involving it are now queryable
```

**Where `h_baseline` comes from for new actors:** The baseline is computed from the actor's fixed inputs (structural features, name encoding) through learned shared projections (Component 2, Section 3.5). It is not a free parameter — the optimizer updates the shared projections (`W_struct`, `W_name`, `W_gate`), not the per-actor vectors directly. This prevents temporal leakage from future predictions into the initial state.

#### 5.4.2 Deactivating an Actor

When the rollout reaches an actor's `active_to` date:

```python
def deactivate_actor(model: FullModel, actor: Actor, date: date):
    """
    Remove an actor from active participation during the training rollout.

    The actor's memory is frozen but retained — it is needed for
    any training examples that reference this actor's state at
    earlier time points.
    """
    # Freeze: stop updating this actor's memory
    model.H[actor.actor_id] = model.H[actor.actor_id].detach()

    # The actor no longer participates in:
    # - New text/event processing (no more memory updates)
    # - Actor self-attention (excluded from active set)
    # - New prediction queries (dyads involving it are not sampled)
    #
    # But its frozen state is still available:
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

#### 5.4.4 Gradient Flow for Baseline Projections

When computing losses for training examples that involve actor i:
- Gradients flow through `h_i(t)` back through the memory update chain (within the TBPTT window)
- At the start of the TBPTT window, `h_i` was either: (a) the output of a previous TBPTT window (detached), or (b) `h_baseline_i` (at epoch start or actor activation)
- In case (b), the gradient reaches `h_baseline_i` and flows through to the shared projections (`W_struct`, `W_name`, `W_gate`). This is how the optimizer learns how to map structural features and names into useful starting geometry.
- Because the projections are shared across all actors, no single actor's gradients can push its baseline to encode actor-specific temporal information. The projections learn generalizable structure only.

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

The model processes events and articles chronologically, updating actor memories each day. Full backpropagation through the entire training period (~2500 days) is infeasible. Instead, truncate gradients every K simulated days:

**K is measured in days, not individual article/event steps.** Within each day, all articles and events build the computation graph (the number of per-actor updates varies depending on article relevance). The TBPTT counter increments once per day. At K=75 days (~2.5 months), the gradient path captures most multi-step escalation sequences while keeping VRAM manageable.

```python
# K=75 days: gradients flow through ~75 daily self-attention passes
# + all within-day article/event updates for those 75 days.
# At TBPTT boundaries: compute loss, backprop, detach memories.
```

### 5.7 Auxiliary Losses at Intermediate Timesteps

To create shorter gradient paths and prevent vanishing gradients, add prediction losses at intermediate checkpoints within each TBPTT window:

```python
# Every M days within each TBPTT window, compute an intermediate prediction loss
M = K // 5  # e.g., every 15 days within a 75-day window

if day_count % M == 0 and day_count > 0:
    intermediate_loss = compute_prediction_losses(sample_predictions)
    intermediate_loss *= 0.3  # lower weight than final loss
    total_loss += intermediate_loss
```

### 5.8 Training Schedule

- **Optimizer:** AdamW with separate parameter groups (different LRs per component, see Section 5.3).
- **TBPTT window:** K=75 days. Each window processes all articles/events for those days, constituting one gradient update.
- **Total steps:** Training period (~2500 days) / K (75 days) × epochs (3-5) ≈ 100-170 gradient updates per epoch.
- **Learning rate schedule:** Linear warmup (first 10% of updates) → constant → cosine decay (last 20%).
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
- All `h_baseline_i` recomputed from fixed inputs through updated projections
- All `h_i(t)` → `h_baseline_i` (freshly computed)
- All `t_last_updated_i` → `train_start_date`
- Active actor set → recomputed from scratch based on `active_from` dates

**Persist (learned parameters):**
- All shared weights: encoder, projections (including baseline projections W_struct, W_name, W_gate), gates, GRUs, heads, etc.
- Optimizer state (momentum, adaptive learning rates)

### 6.2 Why Memory Reset Is Necessary

The model's value comes from learning shared transformations (how to update memories, how to propagate information, how to predict events), not from memorizing actor states at particular times. If actor memories carried over between epochs:

1. **Temporal leakage:** At the start of epoch k+1, `h_Russia(t=0)` would contain information from events at `t=2000 days` (end of epoch k). The model would start with future knowledge baked into its states.
2. **Gradient confusion:** The optimizer would receive conflicting signals — gradients from early in the rollout would push memories one way, but the carried-over state already reflects the end of the previous epoch.
3. **Projection learning failure:** The optimizer would have no incentive to learn good baseline projections, since memories would never actually start from baseline.

By resetting, each epoch provides a clean forward pass. The only information that persists is encoded in the shared weights (including the baseline projections) — which is exactly the generalizable knowledge we want.

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

### 7.2 Auxiliary Losses

```python
# Goldstein scale regression (for uncensored survival examples)
L_goldstein = F.mse_loss(pred["goldstein_pred"], event_goldstein)

# Memory regularization: prevent memory vectors from exploding
L_mem = lambda_mem * (h_i.norm().pow(2) + h_j.norm().pow(2))  # λ_mem = 0.01

# Gate sparsity: encourage sparse memory updates
L_gate = lambda_gate * model.get_gate_penalty()  # λ_gate = 0.01
```

### 7.3 Total Loss

All event types use the same loss structure:

```python
# For all event types (survival curve output):
L_example = (
    L_deephit                            # primary: survival NLL + ranking
    + lambda_1 * L_goldstein             # intensity regression (λ₁ = 0.1)
    + L_mem + L_gate                     # regularization
)
```

Each example is weighted by `temporal_weight * event_type_weight` before aggregation.

---

## 8. Checkpointing and Reproducibility

### 8.1 Checkpoint Contents

Every evaluation epoch, save:

| Artifact | Contents |
|----------|----------|
| `model_weights.pt` | All shared model parameters (encoder, projections, heads, baseline projections, etc.) |
| `optimizer_state.pt` | Optimizer state (for resuming training) |
| `training_config.json` | All hyperparameters, random seeds, data version |
| `metrics.json` | Validation metrics at this checkpoint |

Note: per-actor `h_baseline_i` vectors and actor memory vectors `h_i(t)` are **not** checkpointed — baselines are recomputed from fixed inputs through the learned projections, and memories are recomputed from baselines during each rollout. Actor memories are only checkpointed mid-epoch for crash recovery.

### 8.2 Reproducibility

- **Random seeds:** Fix PyTorch, NumPy, and Python random seeds at the start of each phase.
- **Data ordering:** Events and articles are processed in strict chronological order (deterministic).
- **Data version:** Record the data manifest hash from Component 1 in the training config.
- **Code version:** Record the git commit hash.

### 8.3 Best Model Selection

After Phase 3 training completes:
1. Identify the checkpoint with the best **aggregate validation metric**: C-index across all event types plus Brier score at 30-day horizon derived from the CDF.
2. Load that checkpoint's shared weights and actor baselines.
3. For validation/test evaluation: run a fresh chronological rollout through the relevant period, starting from baselines. This mirrors production inference.
4. Pass to Component 6 (Validation & Calibration) for final evaluation on the held-out test set.
