# Model Training

Handles the end-to-end training pipeline including phased training strategy, loss functions, negative sampling, gating mechanisms, and optimization. This is the core learning component that trains the functions `f` (events update dispositions) and `g` (dispositions generate event probabilities) jointly.

## Scope

This component covers everything involved in training the model: the three-phase strategy, all loss functions, negative sampling, gating mechanism configuration, curriculum learning, and temporal dependency handling. It does **not** cover model architecture definitions for graph propagation (see `graph_propagation`) or prediction heads (see `prediction_heads`), though it orchestrates their training.

## The A↔B Dynamical System

The model learns two coupled functions:

```
A(t+1) = f(A(t), B(t))    [events update dispositions]
P(B(t+1)) = g(A(t))       [dispositions generate event probabilities]
```

Both are trained jointly, end-to-end. The challenge is that both are complex and the system has long-range dependencies — a disposition shift in 2014 is still generating events in 2024.

## Three-Phase Training Strategy

### Phase 1: Structural Pretraining

Initialize actor embeddings from structural features (see `actor_management`). No text or event data yet. Run a few gradient steps to ensure initialization is well-conditioned and projection matrices are well-scaled.

- **Duration:** Minutes, not hours
- **Purpose:** Establish geometric structure of embedding space

### Phase 2: Self-Supervised Text Pretraining

Pretrain text encoder and memory update mechanism before introducing supervised event labels.

**Masked entity prediction:**
```python
L_entity = CrossEntropy(
    actor_classifier(T[mask_positions]),
    true_actor_ids
)
```

**Temporal ordering:**
```python
L_temporal = BCE(
    temporal_classifier(mean_pool(T1), mean_pool(T2)),
    (t1 < t2).float()
)
```

**Contrastive document similarity:** Documents describing the same event should have similar representations. Use GDELT event codes as weak supervision for pairing.

### Phase 3: Supervised Event Prediction Fine-Tuning

Fine-tune end-to-end on event prediction. Training signal flows from prediction errors back into encoder weights.

**Curriculum learning:** Start with high-frequency event types, gradually introduce rare events:

```python
def get_event_weight(event_type, training_step, total_steps):
    event_freq = base_frequencies[event_type]
    rarity = 1.0 / event_freq
    curriculum_factor = min(1.0, training_step / (curriculum_warmup * total_steps))
    return 1.0 + (rarity - 1.0) * curriculum_factor
```

**Multi-task training:** Predict all 18 event types simultaneously plus auxiliary tasks (Goldstein regression, escalation classification, actor-type classification from memory vectors).

## Loss Functions

### Primary: Focal Loss

```
L_focal = -α_t · (1 - p_t)^γ · log(p_t)
```

Where γ=2 is standard, α_t is inverse-frequency class weight. Down-weights easy negatives, focuses on hard examples.

### Auxiliary: Goldstein Scale Regression

```
L_goldstein = MSE(goldstein_pred, goldstein_observed)
```

Regularizes the representation and ensures the model learns an ordering over events.

### Survival/Hazard Loss (DeepHit)

```
L_DeepHit = L_NLL + η · L_ranking
L_NLL = -log(P(T = t_i | X_i))
L_ranking = E[max(0, σ - (S_i(t) - S_j(t)))]
```

### Memory Regularization

```
L_mem_reg = λ_reg · Σᵢ ||h_i||²
```

Prevents memory vectors from exploding or collapsing.

### Total Loss

```
L_total = L_focal
        + λ₁ · L_goldstein
        + λ₂ · L_DeepHit
        + λ₃ · L_mem_reg
        + λ₄ · L0_gate_penalty
```

Tune λ hyperparameters via grid search on held-out validation Brier score.

### Temporal Discounting

```
w(t) = exp(-λ_discount · (T_current - t))
```

Half-life ~6–12 months. Events from 5 years ago are less representative than events from last quarter.

## Negative Sampling

### The Problem
The space of possible events is N_actors² × 18 event types × time steps. The vast majority are never observed. Training cannot enumerate all non-events.

This is the open-world assumption (OWA) problem: unobserved triples are *unknown*, not *false*. GDELT/POLECAT only observe what media reports.

### Word2Vec-Style Negative Sampling

```
L_NS = -log σ(score(s,r,o,t)) - Σₖ E[log σ(-score(s,r,n,t))]
```

k=5–20 negatives per positive. Noise distribution: marginal actor frequency raised to 3/4 power, computed within a rolling time window.

### Hard Negative Mining

```python
def mixed_negatives(observed_event, model, ratio_hard=0.15):
    # ~85% random, ~15% hard (model-predicted plausible non-events)
    random_negs = [corrupt(s, r, o, t, noise_distribution)
                   for _ in range(int(n_negatives * (1 - ratio_hard)))]
    scores = model.score_all_targets(s, r, t)
    scores[o] = -inf
    hard_targets = topk(scores, int(n_negatives * ratio_hard))
    hard_negs = [(s, r, t_hard, t) for t_hard in hard_targets]
    return random_negs + hard_negs
```

85/15 random/hard split. Pure hard negatives can cause mode collapse.

### Structural Feasibility Filter

Only generate negatives from feasible actor pairs (see `structured_data_curation`). This makes the model learn to distinguish plausible-but-didn't-happen from actually-happened.

## Gating Mechanisms

### Two-Scale Gating

Two distinct questions require different functional forms:

1. **Scalar gate (relevance):** Should this document update actor i at all? Competition across actors — most should receive zero weight from most documents.
2. **Dimensional gate (specificity):** Which dimensions of actor i's memory should update? Independent per dimension.

### Recommended Configuration

```python
# Scalar gate: sparsemax across actors (learned sparse selection)
actor_weights = entmax(relevance_scores, alpha=1.5)

# Dimensional gate: sigmoid per dimension (independent, no competition)
gate_dims = sigmoid(W_dims @ update_input)

# Update
delta_h = MLP(update_input)
h_i = h_i * exp(-lambda * dt) + gate_dims * (actor_weights[i] * delta_h)
```

### Gating Options

| Gate Type | Form | Use Case |
|-----------|------|----------|
| Sigmoid | σ(Wx + b) | Dimensional gate (independent per dimension) |
| Sparsemax | argmin on simplex | Scalar gate (exact zeros for irrelevant actors) |
| α-entmax | Generalizes softmax↔sparsemax | Learnable sparsity level |
| Gumbel-sigmoid | Hard decisions with gradient flow | Inference-time computational savings |
| Hard concrete (L0) | Directly penalizes active gate count | Tunable sparsity via λ_L0 |

## Handling Long-Range Temporal Dependencies

**Truncated BPTT:** Only backpropagate through last K memory update steps (K=50–100). Limits gradient path length while learning from recent history.

**Auxiliary losses at intermediate timesteps:** Add event prediction losses at intermediate checkpoints during training rollouts. Creates shorter gradient paths, prevents vanishing gradients.

## Computational Budget

```
Training runs (weekly, A10G spot, 4 hours):  ~$2.00/week → ~$8/month
Training VRAM (with gradients + optimizer):   ~16-20 GB
→ Fits on RTX 3090/4090 (24 GB) or A10G (24 GB)
```

## Key Dependencies

- `torch` — Core training framework
- `sparsemax-pytorch` — Sparsemax / entmax gating
- `pycox` — DeepHit survival loss
- `lightgbm` — Phase 1 tabular baseline
- `transformers` — ConfliBERT fine-tuning

## Build Phase Mapping

| Phase | Focus |
|-------|-------|
| Phase 1 (Months 1–2) | LightGBM classifiers per event type. Brier score, BSS, reliability diagrams |
| Phase 2 (Months 3–5) | ConfliBERT + EvolveGCN. Focal loss + class weights. Temperature scaling |
| Phase 3 (Months 6–12) | Full end-to-end training. TGN memory, Neural ODE, Hawkes process, ensemble |

## Architecture Reference

Corresponds to **Training Strategy** (Section 13), **Negative Sampling** (Section 14), **Loss Functions** (Section 15), and **Gating Mechanisms** (Section 12) in the architecture design document.
