# Component 3: Target Event Definition & Dataset Construction

## Purpose

Define precisely what the model predicts, how positive and negative training examples are derived from curated event data, and how train/validation/test splits are constructed. This component bridges the gap between raw curated data (Component 1) and the training pipeline (Component 5).

**Inputs:** Curated events from Component 1. Actor registry from Component 2.

**Outputs:** Train/validation/test datasets of (positive event, negative events, context window, temporal weight) tuples ready for model training.

---

## 1. Prediction Target Specification

### 1.1 What the Model Predicts

The model answers queries of the form:

> **P(event_type r occurs between actor i and actor j within the next τ days)**

Concretely:
- **event_type r**: one of the 18 PLOVER categories (AID, AGREE, CONSULT, COOP, DEMAND, DISAPPROVE, ENGAGE, FIGHT, INVESTIGATE, MOBILIZE, PROTEST, REDUCE, REJECT, SANCTION, SEIZE, THREATEN, YIELD, OTHER)
- **actor i**: source actor (the actor initiating the event)
- **actor j**: target actor (the actor receiving the event)
- **τ**: prediction horizon in days

The model produces a probability in [0, 1] for each such query.

### 1.2 Prediction Horizons

The model is trained and evaluated at multiple horizons simultaneously:

| Horizon | τ (days) | Use case |
|---------|----------|----------|
| Short-term | 7 | Near-term tactical prediction |
| Medium-term | 30 | Monthly forecasting, ViEWS benchmark alignment |
| Long-term | 90 | Quarterly strategic forecasting |
| Extended | 180 | Long-range outlook |

Each training example generates prediction targets at all four horizons. The prediction head receives τ as an input via a time embedding, so a single model serves all horizons.

### 1.3 Temporal Binning

For a given reference time t₀, the target for horizon τ is:

```
y(i, j, r, t₀, τ) = 1  if ∃ event of type r from actor i to actor j in [t₀, t₀ + τ]
                     0  otherwise
```

This is a binary indicator: did at least one event of this type occur in the window?

**For high-frequency event types** (verbal cooperation, diplomatic consultations) where the binary indicator is almost always 1 for active dyads, use count-based targets instead:

```
y_count(i, j, r, t₀, τ) = count of events of type r from i to j in [t₀, t₀ + τ]
```

Modeled as a Poisson or negative binomial count (see Section 5).

### 1.4 Determining Which Event Types Use Binary vs. Count Targets

Compute the empirical base rate for each event type across all active dyads:

```python
for r in PLOVER_TYPES:
    base_rate = (number of (i,j,month) triples with ≥1 event of type r) / (total active dyad-months)
    if base_rate > 0.10:
        target_type[r] = "count"   # too common for binary to be informative
    else:
        target_type[r] = "binary"
```

Expected classification:
- **Binary targets:** FIGHT, SANCTION, SEIZE, MOBILIZE, THREATEN, PROTEST, AID, AGREE (rare or moderately rare)
- **Count targets:** CONSULT, ENGAGE, COOP, DISAPPROVE (common for active dyads)
- **Borderline:** DEMAND, REDUCE, REJECT, YIELD — evaluate empirically

---

## 2. Positive Example Construction

### 2.1 From Curated Events to Training Positives

Each curated event record from Component 1 generates one or more positive training examples:

```python
def event_to_positives(event: NormalizedEvent) -> list[TrainingPositive]:
    """
    Convert a single curated event into positive training examples.
    """
    positives = []

    # The event itself is a positive for its exact type
    positives.append(TrainingPositive(
        source_actor_id=event.source_actor_id,
        target_actor_id=event.target_actor_id,
        event_type=event.event_type,
        event_date=event.event_date,
        goldstein_score=event.goldstein_score,
        event_mode=event.event_mode,
        magnitude_dead=event.magnitude_dead,
        magnitude_injured=event.magnitude_injured,
        magnitude_size=event.magnitude_size,
        data_source=event.data_source,
    ))

    return positives
```

### 2.2 Event Aggregation

Raw event feeds contain massive duplication — one real-world event generates many coded records from different articles. Before constructing training examples, aggregate:

**Within-source dedup:** Group events by (source_actor_id, target_actor_id, event_type, event_date). Within each group:
- Keep the record with the highest confidence score.
- Average the Goldstein scores.
- Sum the magnitude fields (casualties are cumulative across reports).
- Store the count of pre-dedup records as `report_count` (useful as a media attention proxy).

**Cross-source dedup:** When the same real-world event appears in both GDELT and POLECAT (matched by actor pair + event type + date):
- Prefer the POLECAT record (higher coding precision).
- Retain the GDELT report_count as additional metadata.

### 2.3 Directed vs. Undirected Events

Events are directed: (source → target). "USA sanctions Russia" and "Russia sanctions USA" are different events. The model maintains this directionality.

However, some event types are inherently symmetric:
- AGREE (both parties agree)
- CONSULT (both parties consult)
- ENGAGE (both parties engage)
- COOP (both parties cooperate)

For these symmetric types, each observed event generates two directed training examples: (i → j) and (j → i). For asymmetric types (FIGHT, SANCTION, THREATEN, DEMAND, etc.), only the observed direction is used.

```python
SYMMETRIC_TYPES = {"AGREE", "CONSULT", "ENGAGE", "COOP"}

def expand_directionality(positive: TrainingPositive) -> list[TrainingPositive]:
    if positive.event_type in SYMMETRIC_TYPES:
        reverse = positive.copy()
        reverse.source_actor_id = positive.target_actor_id
        reverse.target_actor_id = positive.source_actor_id
        return [positive, reverse]
    return [positive]
```

---

## 3. Negative Example Construction

### 3.1 The Core Problem

The space of possible events is combinatorially vast: N_actors² × 18 event types × T time steps. The vast majority are never observed. Training cannot enumerate all non-events, so we must sample.

This is complicated by the **open-world assumption**: an unobserved event is *unknown* — it may be false (didn't happen) or simply unrecorded (happened but wasn't covered by media). GDELT and POLECAT only observe what media reports.

### 3.2 Negative Sampling Strategy

For each positive event (s, r, o, t), sample K negative events. K = 10 negatives per positive (tunable).

**Mixed random/hard sampling** with a 85/15 ratio:

```python
def sample_negatives(
    positive: TrainingPositive,
    model: Optional[Model],  # None during Phase 0/1, available during Phase 3
    K: int = 10,
    hard_ratio: float = 0.15,
) -> list[TrainingNegative]:
    s, r, o, t = positive.source_actor_id, positive.event_type, positive.target_actor_id, positive.event_date
    negatives = []

    n_hard = int(K * hard_ratio) if model is not None else 0
    n_random = K - n_hard

    # --- Random negatives ---
    for _ in range(n_random):
        corruption = random.choice(["source", "target", "relation"])
        if corruption == "source":
            s_neg = sample_actor(noise_distribution, exclude=s)
            neg = (s_neg, r, o, t)
        elif corruption == "target":
            o_neg = sample_actor(noise_distribution, exclude=o)
            neg = (s, r, o_neg, t)
        else:
            r_neg = sample_relation(exclude=r)
            neg = (s, r_neg, o, t)

        # Feasibility check
        if is_feasible(neg):
            negatives.append(TrainingNegative(*neg, sampling_method="random"))

    # --- Hard negatives (Phase 3 only) ---
    if model is not None and n_hard > 0:
        scores = model.score_all_targets(s, r, t)
        scores[o] = float("-inf")  # exclude the true target
        hard_targets = topk_indices(scores, n_hard)
        for o_hard in hard_targets:
            if is_feasible((s, r, o_hard, t)):
                negatives.append(TrainingNegative(s, r, o_hard, t, sampling_method="hard"))

    return negatives
```

### 3.3 Noise Distribution

The probability of sampling actor a as a negative is proportional to its marginal event frequency raised to the 3/4 power (following Word2Vec):

```python
def compute_noise_distribution(events: list[NormalizedEvent], window_months: int = 12) -> dict[str, float]:
    """
    Compute noise distribution over actors within a rolling time window.
    """
    recent_events = [e for e in events if e.event_date >= cutoff_date]

    # Count actor frequency (as source or target)
    actor_counts = Counter()
    for e in recent_events:
        actor_counts[e.source_actor_id] += 1
        actor_counts[e.target_actor_id] += 1

    # Raise to 3/4 power (smooths distribution, upweights rare actors relative to frequency)
    total = sum(c ** 0.75 for c in actor_counts.values())
    noise_dist = {actor: (count ** 0.75) / total for actor, count in actor_counts.items()}
    return noise_dist
```

The distribution is recomputed monthly using a rolling 12-month window to account for non-stationarity.

### 3.4 Structural Feasibility Filter

Some event-dyad combinations are structural zeros — essentially impossible given the actors' relationship structure. Including them as negatives wastes capacity on trivially easy discriminations.

```python
def is_feasible(event_tuple: tuple) -> bool:
    """
    Filter out structurally impossible events.
    Training negatives should be plausible counterfactuals, not absurd combinations.
    """
    s, r, o, t = event_tuple
    actor_s = registry.get(s)
    actor_o = registry.get(o)

    # Self-events are not meaningful for most types
    if s == o:
        return False

    # Military events require geographic proximity OR existing conflict history
    if r in {"FIGHT", "MOBILIZE", "SEIZE"}:
        return (geographic_distance(s, o) < 5000  # km
                or has_conflict_history(s, o)
                or has_alliance_with_adversary(s, o))

    # Sanctions require state-level actors (or against companies)
    if r == "SANCTION":
        return actor_s.actor_type in {"state", "igo"} or actor_o.actor_type == "company"

    # Aid typically flows from richer to poorer
    if r == "AID":
        return True  # don't over-constrain; direction is handled by the model

    # Diplomatic events are always feasible between any pair
    return True
```

### 3.5 Avoiding Contaminated Negatives

A "negative" that actually occurred but was not observed (open-world problem) corrupts training. Mitigate this:

1. **Cross-reference across sources:** If an event is absent from POLECAT but present in GDELT, it's not a true negative. Check all sources before labeling a (s, r, o, t) tuple as negative.

2. **Temporal buffer:** Never sample negatives from within ±3 days of a known positive event involving the same actor pair. Near-misses are likely unreported events, not true negatives.

3. **Confidence weighting:** Weight the loss contribution of negative examples by (1 - P(false_negative)), where P(false_negative) is estimated from the actor pair's overall media coverage density:

```python
def negative_confidence(s: str, o: str, t: date) -> float:
    """
    How confident are we that this is a true negative (event really didn't happen)
    vs. just unreported?

    High media coverage → more confident in negatives.
    Low media coverage → less confident.
    """
    coverage = media_coverage_density(s, o, t)  # articles/day mentioning this dyad
    # Sigmoid: high coverage → confidence near 1; low → confidence near 0.5
    confidence = 0.5 + 0.5 * sigmoid(coverage - coverage_threshold)
    return confidence
```

---

## 4. Context Window Construction

Each training example needs a temporal context: the history leading up to the prediction reference time.

### 4.1 Context Window Definition

For a prediction at reference time t₀:

```python
context_window = {
    # Recent events involving either actor in this dyad
    "dyad_events": get_events(source=i, target=j, t_start=t0 - 365, t_end=t0),

    # Recent events involving the source actor with any partner
    "source_events": get_events(source=i, t_start=t0 - 180, t_end=t0, limit=200),

    # Recent events involving the target actor with any partner
    "target_events": get_events(target=j, t_start=t0 - 180, t_end=t0, limit=200),

    # Recent articles mentioning either actor (for text stream)
    "articles": get_articles(actors=[i, j], t_start=t0 - 30, t_end=t0, limit=50),

    # Current structural features for both actors
    "source_structural": get_structural_features(i, year=t0.year),
    "target_structural": get_structural_features(j, year=t0.year),
}
```

The context window is what the model processes to update actor memory vectors to their state at t₀, from which predictions are made.

### 4.2 Context Window Sizes

| Context type | Lookback | Max records | Rationale |
|--------------|----------|-------------|-----------|
| Dyad event history | 365 days | Unlimited (typically <500) | Full year of bilateral history |
| Actor event history | 180 days | 200 per actor | Recent behavior, capped for compute |
| Text articles | 30 days | 50 per dyad | Most recent text context |
| Structural features | Current year | 1 per actor | Slow-moving, one snapshot suffices |

---

## 5. Count Targets for Common Event Types

### 5.1 When to Use Count Targets

For event types where the binary indicator is nearly always 1 for active dyads (base rate >10%), binary prediction is uninformative. Instead, predict the event count per period:

```python
@dataclass
class CountTarget:
    source_actor_id: str
    target_actor_id: str
    event_type: str
    period_start: date
    period_end: date
    observed_count: int       # actual number of events in this period
    report_count: int         # total pre-dedup records (media attention proxy)
    mean_goldstein: float     # average Goldstein score of events in this period
```

### 5.2 Count Distribution

Model counts with a negative binomial distribution to handle overdispersion (variance > mean, which is typical for event data due to bursty clustering):

```
y ~ NegativeBinomial(mu=exp(score), alpha=dispersion)

L_count = -NB.log_prob(observed_count)
```

Where `score` is the model's output for this dyad-event-period combination and `alpha` is a learned dispersion parameter per event type.

### 5.3 Zero-Inflated Variant

Some dyad-event combinations have structural zeros (USA-FIGHT-Canada is never observed, not just low-rate). A zero-inflated negative binomial separates "always zero" from "low rate":

```
P(y=0) = π + (1-π) * NB(y=0|mu,alpha)
P(y=k) = (1-π) * NB(y=k|mu,alpha)    for k > 0
```

Where π is the structural-zero probability, modeled as `sigmoid(feasibility_score)`.

---

## 6. Temporal Train/Validation/Test Splits

### 6.1 Split Strategy

**Strictly temporal splits.** No random splitting — the model must be evaluated on its ability to predict the future from the past, not to interpolate within observed data.

```
Historical data: 1995 ──────────────────────────────────────── 2026
                 │                          │        │        │
                 │        Training          │  Val   │  Test  │
                 │    1995 ─── 2022-06      │  2022  │  2023  │
                 │                          │  -07   │  -01   │
                 │                          │  to    │  to    │
                 │                          │  2022  │  2023  │
                 │                          │  -12   │  -12   │
```

- **Training:** All data up to 2022-06-30.
- **Validation:** 2022-07-01 to 2022-12-31 (6 months). Used for hyperparameter tuning, early stopping, calibration set fitting.
- **Test:** 2023-01-01 to 2023-12-31 (12 months). Evaluated once at the end. Never used for model selection.

### 6.2 Expanding Window for Robustness

To verify the model isn't overfit to a specific temporal split:

| Split name | Train end | Val period | Test period |
|------------|-----------|------------|-------------|
| Primary | 2022-06 | 2022-07 – 2022-12 | 2023-01 – 2023-12 |
| Robustness 1 | 2021-06 | 2021-07 – 2021-12 | 2022-01 – 2022-12 |
| Robustness 2 | 2020-06 | 2020-07 – 2020-12 | 2021-01 – 2021-12 |

Report metrics on all three test periods. A model that works well on only one split is overfit.

### 6.3 Embargo Period

Between training data end and validation/test start, enforce a gap equal to the longest prediction horizon (180 days for the extended horizon):

```
Training data ends:    t_train_end
Embargo period:        [t_train_end, t_train_end + 180 days]
Validation starts:     t_train_end + 180 days
```

Wait — this is overly conservative. The embargo only needs to be as long as the prediction horizon being evaluated. For the 7-day horizon, a 7-day gap suffices. For the 180-day horizon, the 180-day gap applies. In practice, use the gap equal to the longest horizon to keep things simple, or evaluate each horizon with its own appropriate gap.

**Recommended:** Use a 30-day embargo (matching the medium-term horizon that is the primary evaluation target). Evaluate the 90-day and 180-day horizons with the understanding that their earliest predictions partially overlap with the training period.

---

## 7. Temporal Weighting

### 7.1 Training Loss Weights

More recent training examples receive higher weight in the loss:

```python
def temporal_weight(event_date: date, reference_date: date, half_life_days: int = 270) -> float:
    """
    Exponential decay weighting. Events from ~9 months ago get half weight.
    Events from >2 years ago get ~10% weight.
    """
    days_ago = (reference_date - event_date).days
    weight = math.exp(-math.log(2) * days_ago / half_life_days)
    return max(weight, 0.05)  # floor at 5% to avoid completely ignoring old data
```

**Rationale:** The geopolitical system is non-stationary. Patterns from 5 years ago are less representative of current dynamics than patterns from last quarter. But old data still provides useful structural priors, especially for rare events, so we down-weight rather than exclude.

### 7.2 Per-Event-Type Weighting

Rare event types receive higher weight to prevent the loss from being dominated by common events:

```python
def event_type_weight(event_type: str, base_frequencies: dict[str, float]) -> float:
    """
    Inverse-frequency weighting, capped to avoid extreme weights on very rare events.
    """
    freq = base_frequencies[event_type]
    weight = 1.0 / max(freq, 0.001)
    return min(weight, 100.0)  # cap at 100x to avoid instability
```

---

## 8. Training Example Schema

### 8.1 Binary Target Example

```python
@dataclass
class BinaryTrainingExample:
    # Prediction query
    source_actor_id: str
    target_actor_id: str
    event_type: str              # PLOVER category
    reference_date: date         # t₀: the "as of" date for prediction
    horizon_days: int            # τ: how far ahead to predict

    # Target
    label: int                   # 1 if event occurred in [t₀, t₀ + τ], else 0

    # Metadata for the positive event (if label=1)
    event_goldstein: float       # Goldstein score of the actual event
    event_mode: str              # verbal / hypothetical / actual
    event_magnitude_dead: int
    event_magnitude_injured: int
    event_magnitude_size: int

    # Negative sampling metadata (if label=0)
    sampling_method: str         # "random" / "hard" / "n/a"
    negative_confidence: float   # how confident this is a true negative [0.5, 1.0]

    # Weights
    temporal_weight: float       # decay weight based on recency
    event_type_weight: float     # inverse-frequency weight

    # Context reference (not stored inline — loaded at training time)
    context_window_key: str      # key to look up the context window
```

### 8.2 Count Target Example

```python
@dataclass
class CountTrainingExample:
    source_actor_id: str
    target_actor_id: str
    event_type: str
    period_start: date
    period_end: date
    observed_count: int
    mean_goldstein: float
    temporal_weight: float
    event_type_weight: float
    context_window_key: str
```

---

## 9. Dataset Statistics to Compute and Track

Before training, compute and log:

| Statistic | Purpose |
|-----------|---------|
| Base rate per event type | Determines binary vs. count targets; sets climatological baseline for BSS |
| Active dyad count per month | Number of (i,j) pairs with ≥1 event; tracks system activity level |
| Event count distribution per type | Mean, median, variance, max — for negative binomial parameterization |
| Actor event frequency distribution | Identifies sparse vs. dense actors; informs noise distribution |
| Temporal coverage heatmap | Events per month per source — verifies no gaps or anomalies |
| Positive/negative ratio per type | Verifies sampling is producing intended ratios |
| Feasibility filter rejection rate | If >50% of random negatives are filtered, the feasibility constraints may be too strict |
