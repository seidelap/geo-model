# Prediction Heads

Handles the final prediction layer that transforms actor memory vectors into event probability estimates. Given two actors and a time horizon, this component answers: "What is the probability of event type `r` between actors `i` and `j` within `τ` days?"

## Scope

This component covers dyadic representation construction, multi-task prediction heads, the Hawkes process for temporal prediction, and survival models for time-to-event forecasting. It consumes actor memory vectors from the memory store (after text/event/graph updates) and produces calibrated probability estimates (calibration itself is in `model_testing`).

## Dyadic Representation

For a query about event type `r` between actors `i` and `j` within `τ` days:

```python
d_ij = concat([
    h_i,                    # source actor state
    h_j,                    # target actor state
    h_i * h_j,             # element-wise product: compatibility per dimension
    h_i - h_j,             # difference: asymmetry per dimension
    abs(h_i - h_j),        # absolute difference: distance per dimension
    time2vec(τ),            # forecast horizon encoding
    e_r,                    # relation type embedding
])
```

### Why These Features Matter

- **Element-wise product `h_i ⊙ h_j`:** Captures symmetric compatibility — do these actors align on each dimension?
- **Difference `h_i - h_j`:** Captures asymmetry — who is more militarily capable, more Western-aligned, the economic dominant partner?
- **Absolute difference `|h_i - h_j|`:** Captures distance without direction — how far apart are they on each dimension?

Including both is critical because geopolitical relationships are neither purely symmetric nor purely differential. A military alliance (symmetric compatibility) and an aid relationship (asymmetric dependency) both matter.

## Multi-Task Prediction Head

```python
# One MLP head per event type (multi-task with shared trunk)
scores = {}
for r in range(18):
    scores[r] = MLP_r(d_ij)  # shared trunk, relation-specific head

# Event probabilities
P_event = {r: sigmoid(scores[r]) for r in range(18)}

# Goldstein scale intensity (auxiliary regression)
goldstein_pred = MLP_goldstein(d_ij)

# Escalation probability (binary: any conflict-class event?)
P_escalation = sigmoid(MLP_escalation(d_ij))
```

### Multi-Task Benefits

The multi-task setup shares statistical strength across event types. **Rare events (military attacks) benefit from representations learned on common events (verbal cooperation, diplomatic consultations).** This is critical given the severe class imbalance in geopolitical event data.

### Prediction Targets

| Target | Type | Purpose |
|--------|------|---------|
| P(event_r) for each of 18 PLOVER types | Binary probability | Core prediction |
| Goldstein score | Continuous regression | Event intensity, regularizes representations |
| P(escalation) | Binary probability | Any conflict-class event? Operationally useful |
| Event count (high-freq types) | Count (Negative Binomial) | Handles events that occur multiple times per period |

## Hawkes Process for Temporal Prediction

For predicting *when* rather than just *whether* an event occurs:

```python
def hawkes_intensity(h_i, h_j, event_history_ij, t):
    """
    Conditional intensity: rate of event type r between actors i, j at time t.
    """
    # Base rate from actor states
    mu = softplus(MLP_base(concat(h_i, h_j)))

    # Self-excitation: past events increase future rate with exponential decay
    excitation = sum(
        exp(-beta * (t - t_k)) * w_type[r_k]
        for t_k, r_k in event_history_ij
        if t_k < t
    )

    # Total intensity
    lambda_r = mu + softplus(excitation)
    return lambda_r
```

### Why Hawkes?

The Hawkes process captures a fundamental empirical regularity in conflict data: **events beget events.** A military skirmish raises the short-term probability of further skirmishes. Diplomatic agreements cluster similarly. The exponential decay kernel means this excitation effect fades over time.

Implementation reference: `github.com/SimiaoZuo/Transformer-Hawkes-Process`

## Survival Model for Time-to-Event

For questions like "how many days until the next sanctions event between actors i and j?":

```python
# Discrete-time hazard model (DeepHit)
hazard_t = MLP_hazard(concat(d_ij, time2vec(t)))  # hazard at each time step
survival_t = cumprod(1 - hazard_t)                 # survival function
event_prob_by_t = 1 - survival_t                   # CDF
```

### Why DeepHit?

- Handles **competing risks** naturally (coup vs. election vs. revolution)
- Does **not** require proportional hazards assumptions
- Works with **censored observations** (ongoing peace periods where the eventual outcome time is unknown)

Implementation via `pycox` library (`github.com/havakv/pycox`).

## Relation Type Embeddings

Each of the 18 PLOVER event types has a dense embedding vector `e_r ∈ R^d` in the same space as actor vectors. These are **not** compressed into fewer dimensions — each occupies a point in the full d-dimensional space.

Training discovers the geometry:
- Event types that co-occur between similar actor pairs cluster together
- Event types that tend to precede or follow each other are nearby
- The "relationship" between any two actors is computed on demand as a function of their memory vectors, not stored as a label

## Key Dependencies

- `torch` — MLP heads, training
- `pycox` — DeepHit survival analysis (`github.com/havakv/pycox`)
- `Transformer-Hawkes-Process` — Neural temporal point process (`github.com/SimiaoZuo/Transformer-Hawkes-Process`)

## Build Phase Mapping

| Phase | Implementation |
|-------|---------------|
| Phase 1 (Months 1–2) | LightGBM classifiers per event type (no neural prediction head) |
| Phase 2 (Months 3–5) | Multi-task MLP prediction heads for all 18 event types. Focal loss + class weights |
| Phase 3 (Months 6–12) | Transformer Hawkes Process. DeepHit survival model. Full dyadic representation with all interaction features |

## Open Research Questions

- **Composing actor-level predictions to question-level probabilities:** Prediction markets ask specific resolved questions ("Will Russia and Ukraine sign a ceasefire before December 31?"). The model predicts abstract event-type probabilities between actor pairs. The mapping requires a semantic layer — ideally an LLM that parses the question, identifies relevant actors/event types, and aggregates dyadic estimates. This interface is understudied.

## Architecture Reference

Corresponds to **Layer 5: Event Prediction Head** (Section 10), **Hawkes Process** (Section 10.3), and **Survival Model** (Section 10.4) in the architecture design document.
