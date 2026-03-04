# Component 6: Model Validation & Calibration

## Purpose

Evaluate the trained model on held-out test data, apply post-hoc calibration, and benchmark against external baselines. This component validates every major architectural decision — from the self-supervised pretraining objectives through the memory dynamics to the final survival curve outputs — producing the evidence needed to determine whether the model is useful and where it fails.

**Inputs:** Best model checkpoint from Component 5. Test dataset from Component 3 (Section 6). External benchmark data (ViEWS, Polymarket).

**Outputs:** Comprehensive evaluation report with per-event-type metrics, calibration parameters, reliability diagrams, memory quality analysis, surprise signal evaluation, ablation results, and benchmark comparisons.

---

## 1. Evaluation Protocol

### 1.1 Strict Temporal Separation

The test set is strictly future relative to all training data. No information from the test period may leak into training, validation, or calibration. This includes Phase 2 self-supervised pretraining — ConfliBERT must not be fine-tuned on articles from the validation or test period (see Component 5, Section 4.4).

```
Training:    ... ─── 2024-09-30
Validation:  2024-10-01 ─── 2025-03-31   (model selection, hyperparameter tuning, calibration fitting)
Test:        2025-04-01 ─── 2026-03-31   (final evaluation, reported once)
```

### 1.2 Evaluation Procedure

```python
def evaluate(model: FullModel, test_data: TestDataset, calibration_params: dict) -> EvaluationReport:
    """
    Full evaluation pipeline.
    """
    predictions = {}
    actuals = {}
    surprise_scores = {}

    # 1. Replay context: bring actor memories to the state at the start of the test period
    #    by running a full chronological rollout over the training period
    model.replay_to(date(2024, 9, 30))

    # 2. For each test reference date, make predictions
    for reference_date in test_data.reference_dates:
        # Update memories with data up to reference_date (simulating streaming inference)
        model.update_to(reference_date)

        # Record surprise scores for all active actors (CPC + event-type KL)
        for actor_id in model.active_actors:
            surprise_scores[(actor_id, reference_date)] = {
                "cpc_score": model.get_cpc_score(actor_id),
                "event_type_kl": model.get_event_type_kl(actor_id),
            }

        # Predict for all test queries at this reference date
        for query in test_data.queries_at(reference_date):
            event_history = get_dyad_event_history(query.source, query.target)
            survival_curve = model.predict_survival(
                query.source, query.target, query.event_type, event_history
            )

            # Apply calibration (per-bin temperature scaling)
            cal_curve = calibrate_curve(survival_curve, query.event_type, calibration_params)

            predictions[query.id] = cal_curve
            actuals[query.id] = query.actual_event_time

    # 3. Compute all metrics
    return compute_all_metrics(predictions, actuals, surprise_scores, test_data)
```

### 1.3 Streaming vs. Batch Evaluation

Two evaluation modes:

**Streaming (realistic):** The model processes events and articles in real-time order during the test period. Actor memories update continuously. CPC and event-type prediction heads compute surprise scores daily. Predictions are made at each reference date using the current memory state, including surprise features. This tests the model as it would actually be deployed.

**Batch (diagnostic):** The model's memories are frozen at the training cutoff. No updates during the test period. Surprise features are fixed at their last training-period values. This isolates the predictive value of the initial state from the model's ability to incorporate new information.

Report both modes. If streaming performance is substantially better than batch, the memory update mechanism and surprise signal are contributing value. If they are similar, the model is relying primarily on structural features and historical patterns.

---

## 2. Primary Metrics

The model outputs survival curves, not single probabilities. Evaluation uses both **survival-native metrics** (which evaluate the full temporal distribution) and **derived fixed-horizon metrics** (which evaluate CDF read-offs at standard horizons for comparability with baselines).

### 2.1 Concordance Index (C-index) — Survival-Native

```python
def concordance_index(predicted_cdfs: list[Tensor], actual_times: list[float], censored: list[bool]) -> float:
    """
    Fraction of comparable pairs where the model correctly ranks who experiences
    the event first. C-index = 0.5 is random, 1.0 is perfect ordering.

    A pair (i, j) is comparable if i's event time < j's event time and i is uncensored.
    The model is concordant if it assigns higher CDF (higher risk) to i than j
    at the time of i's event.
    """
    concordant, discordant, tied = 0, 0, 0
    for i in range(len(actual_times)):
        if censored[i]:
            continue
        for j in range(len(actual_times)):
            if i == j or actual_times[j] <= actual_times[i]:
                continue
            risk_i = interpolate_cdf(predicted_cdfs[i], actual_times[i])
            risk_j = interpolate_cdf(predicted_cdfs[j], actual_times[i])
            if risk_i > risk_j:
                concordant += 1
            elif risk_i < risk_j:
                discordant += 1
            else:
                tied += 1
    return (concordant + 0.5 * tied) / max(concordant + discordant + tied, 1)
```

**Target:** C-index ≥ 0.70 per event type. Report macro-averaged across event types.

### 2.2 Integrated Brier Score (IBS) — Survival-Native

```python
def integrated_brier_score(predicted_cdfs: list[Tensor], actual_times: list[float],
                           censored: list[bool], t_max: float = 365) -> float:
    """
    Time-integrated Brier score: evaluates the full survival curve, not just a single horizon.
    IBS = (1/t_max) * integral_0^t_max BS(t) dt

    Uses inverse-probability-of-censoring weighting (IPCW) to handle censored observations.
    """
    t_grid = np.linspace(0, t_max, 200)
    bs_values = []
    for t in t_grid:
        sq_errors = []
        for i in range(len(actual_times)):
            predicted_cdf_at_t = interpolate_cdf(predicted_cdfs[i], t)
            actual_indicator = float(actual_times[i] <= t and not censored[i])
            sq_errors.append((predicted_cdf_at_t - actual_indicator) ** 2)
        bs_values.append(np.mean(sq_errors))
    return np.trapz(bs_values, t_grid) / t_max
```

IBS evaluates the *entire* curve rather than cherry-picking horizons. Lower is better.

### 2.3 Fixed-Horizon Brier Score (Derived)

```python
def fixed_horizon_brier(predicted_cdfs: list[Tensor], actual_times: list[float],
                        horizon_days: float) -> float:
    """
    Standard Brier score at a specific horizon, derived from the survival CDF.
    """
    predictions = [interpolate_cdf(cdf, horizon_days) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return np.mean([(p - y) ** 2 for p, y in zip(predictions, actuals)])
```

Report at standard horizons (7, 30, 90, 180 days) for comparison with Phase 0 LightGBM and external benchmarks. These are read-offs from the same survival curve, so consistency (P(30d) ≥ P(7d)) is guaranteed.

### 2.4 Brier Skill Score (BSS)

```python
def brier_skill_score(bs_model: float, bs_baseline: float) -> float:
    return 1.0 - bs_model / bs_baseline
```

Compute BSS at the 30-day horizon (primary) against three baselines:
1. **Climatological baseline:** Always predict the historical base rate for each event type.
2. **No-change baseline:** Predict that whatever happened last month will happen this month.
3. **Phase 0 LightGBM:** The tabular baseline from Component 5.

### 2.5 Hawkes Intensity Metrics (High-Frequency Event Types)

For event types modeled as rates (CONSULT, ENGAGE, COOP, DISAPPROVE), evaluate the Hawkes process output:

```python
def hawkes_evaluation(predicted_intensities: list, actual_event_times: list, T_windows: list) -> dict:
    """
    Evaluate the Hawkes intensity model on high-frequency event types.
    """
    # 1. Log-likelihood ratio vs. homogeneous Poisson baseline
    ll_model = sum(hawkes_log_likelihood(pred, actual, T) for pred, actual, T in zip(...))
    ll_poisson = sum(poisson_log_likelihood(mean_rate, actual, T) for ...)
    ll_ratio = ll_model - ll_poisson

    # 2. Count calibration: predicted vs. actual event counts per window
    predicted_counts = [integrate_intensity(pred, T) for pred, T in zip(...)]
    actual_counts = [len(times) for times in actual_event_times]
    count_mae = mean_absolute_error(predicted_counts, actual_counts)

    # 3. Temporal clustering: does the model correctly predict bursty periods?
    # Compare predicted intensity peaks with actual event clusters
    return {"ll_ratio": ll_ratio, "count_mae": count_mae}
```

### 2.6 Log Loss and PR-AUC (Derived)

```python
def log_loss_at_horizon(predicted_cdfs, actual_times, horizon_days, eps=1e-7):
    predictions = [np.clip(interpolate_cdf(cdf, horizon_days), eps, 1 - eps) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for p, y in zip(predictions, actuals)])

def pr_auc_at_horizon(predicted_cdfs, actual_times, horizon_days):
    """PR-AUC: appropriate for rare events where ROC-AUC is misleading."""
    predictions = [interpolate_cdf(cdf, horizon_days) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return average_precision_score(actuals, predictions)
```

---

## 3. Calibration

### 3.1 Per-Bin Temperature Scaling

The survival curve consists of K hazard logits per time bin. Calibrate by applying a learned temperature per event type per time bin on the validation set:

```python
def fit_bin_temperatures(val_hazard_logits: Tensor, val_event_bins: Tensor,
                         val_censored: Tensor, K: int) -> Tensor:
    """
    Fit per-bin temperatures that minimize the DeepHit NLL on the validation set.
    """
    T = nn.Parameter(torch.ones(K) * 1.5)
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        calibrated_hazard = torch.sigmoid(val_hazard_logits / T.unsqueeze(0))
        survival = torch.cumprod(1 - calibrated_hazard, dim=-1)
        survival_shifted = torch.cat([torch.ones(len(survival), 1), survival[:, :-1]], dim=1)
        pdf = calibrated_hazard * survival_shifted

        loss = 0.0
        for i in range(len(val_event_bins)):
            if not val_censored[i]:
                loss -= torch.log(pdf[i, val_event_bins[i]] + 1e-8)
            else:
                loss -= torch.log(survival[i, val_event_bins[i]] + 1e-8)
        loss /= len(val_event_bins)
        loss.backward()
        return loss

    optimizer.step(closure)
    return T.detach()

# Fit per event type, apply at test time
def calibrate_curve(raw_logits: Tensor, event_type: str, bin_temperatures: dict) -> dict:
    T = bin_temperatures[event_type]
    hazard = torch.sigmoid(raw_logits / T)
    survival = torch.cumprod(1 - hazard, dim=-1)
    cdf = 1 - survival
    return {"hazard": hazard, "survival": survival, "cdf": cdf}
```

**Why per-bin:** The model may be systematically overconfident at near-term horizons but well-calibrated long-term, or vice versa. Per-bin temperatures correct this independently.

### 3.2 Expected Calibration Error (ECE)

```python
def ece_at_horizon(predicted_cdfs, actual_times, horizon_days, n_bins=15):
    predictions = [interpolate_cdf(cdf, horizon_days) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (np.array(predictions) >= bin_lower) & (np.array(predictions) < bin_upper)
        if mask.sum() > 0:
            bin_conf = np.array(predictions)[mask].mean()
            bin_acc = np.array(actuals)[mask].mean()
            ece += mask.mean() * abs(bin_conf - bin_acc)
    return ece
```

| Metric | Pre-calibration | Post-calibration | Target |
|--------|----------------|-----------------|--------|
| ECE at 7 days | Measure | Measure | < 0.05 |
| ECE at 30 days | Measure | Measure | < 0.05 |
| ECE at 90 days | Measure | Measure | < 0.05 |
| ECE at 180 days | Measure | Measure | < 0.05 |
| ECE worst (any type × horizon) | Measure | Measure | < 0.10 |

### 3.3 Reliability Diagrams

For each event type, produce reliability diagrams at each standard horizon (7, 30, 90, 180 days):
- X-axis: predicted CDF(horizon), binned
- Y-axis: observed event frequency within that bin
- Perfect calibration: points on the diagonal

Additionally, produce **survival curve comparison plots**: overlay the average predicted survival curve against the Kaplan-Meier empirical survival curve for each event type. These should agree across the full temporal range, not just at fixed horizons.

Generate diagrams for: (a) the full test set, (b) the first 6 months of test, (c) the last 6 months of test. If calibration degrades over time, the model may be drifting and needs more frequent recalibration.

---

## 4. Memory & Representation Quality

These evaluations test whether the model's internal representations — memory vectors, surprise signals, self-attention patterns — are working as designed. Unlike Section 2 (which tests prediction outputs), this section inspects the model's internals.

### 4.1 Phase 2 Transfer Validation

Phase 2 self-supervised pretraining trains the encoder and memory dynamics before supervised labels are introduced. Validate that this pretraining transfers meaningfully to Phase 3:

**CPC accuracy on held-out test data:**

```python
def evaluate_cpc_transfer(model, test_data):
    """
    Can the memory distinguish next-day articles from negatives on the test set?
    If CPC accuracy is near chance on test data, Phase 2 didn't learn generalizable
    predictive dynamics — it may have overfit to training-period patterns.
    """
    correct, total = 0, 0
    for day in test_data.iter_days():
        for actor_id in model.active_actors:
            if actor_id not in day.next_day.article_aggregates:
                continue
            z_pred, z_pos = model.cpc_head(model.H[actor_id], day.next_day.article_aggregates[actor_id])
            negatives = sample_negatives(actor_id, day, n=15)
            z_negs = torch.stack([model.cpc_head.W_target(n) for n in negatives])
            pos_score = (z_pred * z_pos).sum()
            neg_scores = z_pred @ F.normalize(z_negs, dim=-1).T
            if pos_score > neg_scores.max():
                correct += 1
            total += 1
    return correct / total  # target: ≥ 60% (lower than Phase 2's 70% due to distribution shift)
```

**Event-type prediction accuracy on test data:**

Measure KL divergence between predicted and actual event-type distributions on the test set. Compare to: (a) a baseline that always predicts the training-period marginal distribution, and (b) Phase 2 performance. If Phase 3 fine-tuning degrades event-type prediction relative to Phase 2, the auxiliary loss weight (0.1) may need to be higher.

### 4.2 Memory Vector Quality

**Geopolitical clustering:**

After rolling out actor memories through the test period, visualize with t-SNE/UMAP:
- NATO allies should cluster together
- BRICS states should form a recognizable group
- Active conflict dyads should be separated from cooperative dyads
- Leaders should be near their parent states

If clustering degrades compared to Phase 2's end-of-training vectors, Phase 3 may be distorting the memory space to overfit to specific prediction targets.

**Memory dynamics inspection:**

For a handful of well-known actors (USA, Russia, China, Ukraine), plot memory vector norms and key PCA dimensions over the test period:

```python
def plot_memory_trajectories(model, test_data, actors=["state:USA", "state:RUS", "state:CHN", "state:UKR"]):
    """
    Visualize how memory vectors evolve during the test period.
    Look for: meaningful responses to known events, smooth trajectories between events,
    decay toward baseline during quiet periods.
    """
    for actor_id in actors:
        norms = []
        baseline_distances = []
        for day in test_data.iter_days():
            model.update_to(day.date)
            h = model.H[actor_id]
            b = model.B[actor_id]
            norms.append(h.norm().item())
            baseline_distances.append((h - b).norm().item())
        # Plot: norm over time, distance-from-baseline over time
        # Annotate with known events (e.g., elections, conflicts, summits)
```

**What to look for:**
- Memory should spike after significant events and decay back toward baseline during quiet periods
- The EMA baseline should drift gradually in response to sustained shifts (e.g., prolonged sanctions)
- Actors with no news should converge toward their baseline (temporal decay working correctly)
- Memory norms should stay bounded (regularization working)

### 4.3 Self-Attention Analysis

Layer 4 (actor self-attention) runs daily over all actors. Inspect the learned attention patterns:

```python
def analyze_attention_patterns(model, test_data, sample_days=10):
    """
    Extract and analyze self-attention weights from Layer 4.
    """
    for day in random.sample(test_data.days, sample_days):
        model.update_to(day.date)
        H = model.gather_active_memories()
        # Run self-attention with attention weight output
        _, attn_weights = model.actor_propagation.layers[0].mha(
            H.unsqueeze(0), H.unsqueeze(0), H.unsqueeze(0), need_weights=True
        )
        # attn_weights: [1, n_heads, N_actors, N_actors]

        # Per-head analysis: do different heads specialize?
        # Head 0 might capture alliance relationships, Head 1 might capture adversarial dynamics
        for head in range(n_heads):
            top_pairs = topk_attention_pairs(attn_weights[0, head], k=20)
            # Log: which actor pairs attend most strongly to each other?
```

**What to look for:**
- Different attention heads should specialize (cooperative vs. adversarial vs. geographic vs. organizational)
- Attention should be non-trivial — not uniform, not collapsed to self-attention only
- Actors involved in active crises should attend more strongly to each other than baseline
- Regional blocs should show higher within-group attention

### 4.4 Cross-Attention Analysis

Layer 2 (text processing) uses actor memory as queries over document tokens. Validate that actors attend to relevant tokens:

```python
def analyze_cross_attention(model, sample_articles, actors):
    """
    For a sample of articles, inspect which tokens each actor attends to.
    """
    for article in sample_articles:
        T = model.encode_document(article.text)
        for actor_id in actors:
            m_i, attn_weights = model.actor_reads_document(
                model.H[actor_id], T, return_attention=True
            )
            # Map attention weights back to tokens
            top_tokens = get_top_attended_tokens(attn_weights, article.tokens, k=10)
            # Log: does Russia attend to "Russia", "Moscow", "sanctions"?
            # Does the attention shift based on the actor's recent memory state?
```

**What to look for:**
- Actors should attend to their own name mentions (name encoding working)
- Attention should also capture context beyond name mentions — an actor in a FIGHT state should attend to conflict-related tokens even in articles that mention them indirectly
- Two actors with different memory states should attend to different parts of the same article

---

## 5. Surprise Signal Evaluation

The CPC score and event-type prediction KL are fed as features to the prediction heads (Component 4, Section 6.1). This section evaluates whether these surprise signals carry genuine predictive value.

### 5.1 Surprise as Escalation Predictor

```python
def surprise_escalation_analysis(surprise_scores, test_events, lookforward_days=30):
    """
    Do high surprise scores precede escalatory events?

    For each actor-day with a surprise score, check if an escalatory event
    (FIGHT, THREATEN, MOBILIZE, SANCTION) occurs in the next N days.
    """
    results = []
    for (actor_id, date), scores in surprise_scores.items():
        future_events = get_events_involving(actor_id, date, date + timedelta(days=lookforward_days))
        escalatory = [e for e in future_events if e.event_type in ESCALATORY_TYPES]
        results.append({
            "cpc_score": scores["cpc_score"],
            "event_type_kl": scores["event_type_kl"],
            "escalation_followed": len(escalatory) > 0,
            "n_escalatory_events": len(escalatory),
        })

    # Compute: ROC-AUC and PR-AUC of surprise scores for predicting escalation
    # If AUC > 0.6, the surprise signal has genuine predictive value beyond the memory state alone
    df = pd.DataFrame(results)
    cpc_auc = roc_auc_score(df["escalation_followed"], df["cpc_score"])
    kl_auc = roc_auc_score(df["escalation_followed"], df["event_type_kl"])
    return {"cpc_escalation_auc": cpc_auc, "kl_escalation_auc": kl_auc}
```

### 5.2 Surprise Feature Importance

Measure how much the surprise features contribute to the survival head's predictions via feature permutation:

```python
def surprise_feature_importance(model, test_data):
    """
    Permute surprise features while holding everything else fixed.
    Measure degradation in C-index and Brier score.
    """
    # Baseline: full model with real surprise features
    baseline_metrics = evaluate(model, test_data)

    # Permuted: shuffle surprise_i and surprise_j across dyads
    model_permuted = model.with_permuted_surprise()
    permuted_metrics = evaluate(model_permuted, test_data)

    # Zeroed: set all surprise features to zero
    model_zeroed = model.with_zeroed_surprise()
    zeroed_metrics = evaluate(model_zeroed, test_data)

    return {
        "baseline_c_index": baseline_metrics.c_index,
        "permuted_c_index": permuted_metrics.c_index,
        "zeroed_c_index": zeroed_metrics.c_index,
        "surprise_contribution": baseline_metrics.c_index - zeroed_metrics.c_index,
    }
```

If `surprise_contribution` is near zero, the surprise features are not adding value beyond what the memory vectors already encode, and they could be removed to simplify the architecture.

### 5.3 Event-Type Prediction Interpretability

For a sample of high-surprise days, produce human-readable diagnostics:

```python
def interpret_surprise(model, actor_id, date):
    """
    What did the model expect vs. what actually happened?
    """
    predicted_dist = F.softmax(model.event_type_head(model.H[actor_id]), dim=-1)
    actual_counts = get_event_counts(actor_id, date, date + timedelta(days=7))
    actual_dist = actual_counts / actual_counts.sum()

    # Top-3 expected event types vs. top-3 actual
    expected_top3 = predicted_dist.topk(3)
    actual_top3 = actual_dist.topk(3)

    # Per-type surprise: which types were most unexpected?
    per_type_surprise = actual_dist * (actual_dist / predicted_dist.clamp(min=1e-6)).log()
    most_surprising_type = PLOVER_TYPES[per_type_surprise.argmax()]
    return {
        "expected": [(PLOVER_TYPES[i], p) for i, p in zip(expected_top3.indices, expected_top3.values)],
        "actual": [(PLOVER_TYPES[i], p) for i, p in zip(actual_top3.indices, actual_top3.values)],
        "most_surprising_type": most_surprising_type,
    }
```

Include these interpretable diagnostics in the evaluation report for the top-50 highest-surprise actor-days in the test period. This provides qualitative validation: do the "surprises" correspond to events that a human analyst would also consider unexpected?

---

## 6. Benchmark Comparisons

### 6.1 ViEWS Benchmark

The ViEWS platform (`github.com/views-platform`) provides standardized conflict prediction evaluations.

**Comparison methodology:**
1. Map PLOVER event types to ViEWS conflict categories (primarily FIGHT → ViEWS state-based conflict).
2. Aggregate dyad-level predictions to country-month level (ViEWS predicts at country-month, not dyad-month). For a given country-month, take max CDF(30d) across all dyads involving that country for the relevant event type.
3. Evaluate on the same test period as ViEWS 2023/24 competition submissions.
4. Report: our Brier score vs. ViEWS submitted models' Brier scores.

**Expected outcome:** Competitive with top ViEWS submissions for state-based conflict, while also covering the other 17 event types that ViEWS doesn't predict.

### 6.2 Polymarket Comparison

**Comparison methodology:**
1. Collect resolved Polymarket geopolitical questions from the test period.
2. For each question, manually map to relevant (actor_i, actor_j, event_type) queries and compute the corresponding horizon from the question's resolution date.
3. Query our model's survival CDF at the time the Polymarket question opened, reading off the CDF at the question's horizon.
4. Compute head-to-head Brier scores: our CDF(horizon) vs. Polymarket's implied probability at the same time.

**Challenges:**
- Polymarket questions are specific and nuanced ("Will Russia and Ukraine sign a ceasefire before December 31, 2025?"), while our model predicts generic event types. The mapping is imperfect, but the survival curve makes it natural — the question's deadline maps directly to a CDF read-off.
- Small sample size: Polymarket may have only 20–50 resolved geopolitical questions in a given year.
- Report comparisons on the full question set and stratified by liquidity.

### 6.3 Superforecaster Baseline

Good Judgment Open provides aggregate human forecasts. Superforecasters achieve mean Brier ≈ 0.14 on geopolitical questions. This is an aspirational target. Comparison is approximate due to different question sets and evaluation periods.

### 6.4 No-Change Baseline

```python
def no_change_baseline(dyad_history, reference_date, horizon):
    """
    Predict that whatever happened in the last period will happen in the next.
    """
    lookback_start = reference_date - timedelta(days=horizon)
    recent = [e for e in dyad_history if lookback_start <= e.date < reference_date]
    return 1.0 if len(recent) > 0 else 0.0
```

The no-change baseline is deceptively strong for conflict prediction (most peaceful dyads stay peaceful, most conflict dyads continue fighting). The ViEWS 2020/21 competition showed many sophisticated models fail to beat this.

**Requirement:** Our model must beat the no-change baseline on aggregate BSS. Failure indicates the model is not capturing meaningful dynamics beyond persistence.

---

## 7. Per-Event-Type Analysis

### 7.1 Event Type Report Card

For each of the 18 PLOVER event types, report:

| Metric | Value | Comparison |
|--------|-------|------------|
| Target type | survival / intensity / both | — |
| Base rate at 30d (test set) | % | — |
| C-index | | ≥ 0.70 target |
| IBS (integrated Brier) | | vs. Phase 0 |
| Brier at 7d (derived from CDF) | | vs. Phase 0 |
| Brier at 30d (derived from CDF) | | vs. Phase 0 |
| Brier at 90d (derived from CDF) | | vs. Phase 0 |
| BSS at 30d vs. climatological | | positive = good |
| BSS at 30d vs. no-change | | positive = model beats persistence |
| BSS at 30d vs. Phase 0 (LightGBM) | | positive = neural beats tabular |
| PR-AUC at 30d | | vs. Phase 0 |
| ECE at 30d (post-calibration) | | < 0.05 target |
| Hawkes LL ratio (intensity types) | | vs. homogeneous Poisson |
| Mean surprise score before event | | higher = model was surprised |
| Mean predicted vs. Kaplan-Meier | | visual agreement |

**State-transition event types** (SANCTION, FIGHT, MOBILIZE, SEIZE, AID, REDUCE — see Component 3, Section 6.3): Report START and END predictions separately. START accuracy is typically harder (predicting onset is harder than predicting continuation). Verify that the eligibility constraints are enforced — the model should never predict START for a dyad already in an active state.

### 7.2 Error Analysis

For each event type, categorize prediction errors:

**False positives (predicted event, didn't happen):**
- Were there near-misses (events of a similar type, or events that almost happened)?
- Was the model picking up on genuine escalation that didn't materialize?
- Was the surprise signal elevated before the false positive? If so, the model detected a real change in dynamics, even though the specific predicted event didn't occur.

**False negatives (missed actual events):**
- Was the event a genuine surprise (no precursors in the data)? Check CPC score — a high surprise score on the day of the event suggests the model's memory was not positioned for it.
- Did the model have sufficient data on both actors? Check actor sparsity category.
- Was there a data pipeline failure (event not coded, or coded late)?

**Confident wrong predictions (|p - y| > 0.8):**
- These are the most dangerous errors for deployment.
- Log and manually review every prediction where |p - y| > 0.8.
- For each, record: the surprise scores for both actors, the event-type prediction distribution, and the top-attended tokens from recent articles. This provides a complete diagnostic picture.
- Look for systematic patterns: specific dyads, specific event types, or specific time periods where the model is consistently overconfident.

---

## 8. Ablation Studies

### 8.1 Component Ablations

Test the contribution of each model component by removing it and measuring performance degradation:

| Ablation | What's removed | Expected impact |
|----------|---------------|-----------------|
| No text stream | Remove Layer 2, rely only on structured events | Moderate degradation, especially for events with textual precursors (THREATEN, DEMAND) |
| No event stream | Remove Layer 3, rely only on text | Moderate degradation, especially for high-frequency event types modeled as intensities |
| No actor self-attention | Remove Layer 4 (skip daily self-attention) | Small-moderate degradation, mainly for second-order effects between indirectly connected actors |
| No Hawkes excitation | Remove temporal self-excitation from survival head | Degradation on bursty event types (FIGHT, PROTEST) |
| No temporal decay | Set λ_decay=0 (memories never decay) | Memories become stale; late-test performance degrades |
| No EMA baseline | Fixed baseline (no drift) — decay always toward h_baseline_init | Degradation for actors with sustained shifts (sanctions regimes, alliance changes) |
| No surprise features | Zero out surprise_i and surprise_j in dyadic representation | Quantifies the marginal value of the CPC and event-type prediction signal |
| No Phase 2 auxiliary losses | Remove L_cpc and L_event_type_pred from Phase 3 loss | Tests whether auxiliary losses provide meaningful regularization |
| No Phase 2 pretraining | Skip Phase 2 entirely; initialize Phase 3 from Phase 1 | Tests whether self-supervised pretraining accelerates or improves convergence |
| No name encoding | Remove name encoding from h_baseline_init | Tests whether lexical identity helps cross-attention actor discrimination |
| No curriculum learning | Full event-type weights from step 0 | Possible degradation on rare events |
| Random initialization | Skip Phase 1 structural pretraining | Slower convergence, possible small degradation |

### 8.2 Hyperparameter Sensitivity

| Hyperparameter | Test values | Metric to watch |
|----------------|-------------|-----------------|
| Embedding dimension d | 128, 256, 512 | C-index + IBS |
| Memory decay half-life | 30, 90, 180, 365 days | Late-test-period C-index |
| EMA baseline α | 0.95, 0.97, 0.99, 0.999 | Memory trajectory smoothness, BSS |
| TBPTT window K | 25, 50, 75, 100 | C-index + training stability |
| Self-attention layers | 1, 2, 3 | IBS on second-order effects |
| Self-attention heads | 4, 8, 16 | Head specialization quality |
| Number of time bins K | 10, 17, 30 | IBS (more bins = finer curve, more parameters) |
| CPC negative count | 5, 15, 30 | CPC accuracy, downstream BSS |
| Phase 2 auxiliary loss weight | 0.0, 0.05, 0.1, 0.2 | Phase 3 BSS, CPC transfer accuracy |
| CPC temperature τ | 0.03, 0.07, 0.15 | CPC accuracy, downstream BSS |

---

## 9. Robustness Checks

### 9.1 Temporal Robustness

Evaluate on multiple test periods (Component 3, Section 6.2):

| Test period | C-index | IBS | BSS at 30d | Notes |
|-------------|---------|-----|------------|-------|
| 2025-04 – 2026-03 (primary) | | | | Primary evaluation |
| 2024-04 – 2025-03 (robustness 1) | | | | Should be similar |
| 2023-04 – 2024-03 (robustness 2) | | | | Should be similar |

If performance varies drastically across test periods, the model may be fitting to temporal artifacts.

### 9.2 Actor Sparsity Robustness

Stratify evaluation by actor data density:

| Actor category | Dyad count | C-index | IBS | BSS at 30d |
|----------------|------------|---------|-----|------------|
| Dense-Dense (G7 × G7) | | | | Expected: best performance |
| Dense-Sparse (G7 × small state) | | | | Expected: moderate |
| Sparse-Sparse (small × small) | | | | Expected: weakest |

The model should degrade gracefully with data sparsity. If Sparse-Sparse performance is worse than the climatological baseline, those predictions should not be trusted.

### 9.3 Event Type Novelty

Test on events between actor pairs that had **no prior events of that type** in the training data:

```python
novel_predictions = [
    (p, y) for query, p, y in test_results
    if count_prior_events(query.source, query.target, query.event_type, training_period) == 0
]
```

If novel-event performance is much worse than overall performance, the model is memorizing dyad-event-type co-occurrence patterns rather than learning general dynamics.

### 9.4 Actor Lifecycle Transitions

Evaluate specifically on actors that were activated during the test period (new leaders, new organizations):

```python
def evaluate_new_actors(model, test_data, registry):
    """
    How well does the model perform on actors it hasn't seen during training?
    These actors start from h_baseline_init (structural projection + name encoding)
    and must learn from the test-period data stream alone.
    """
    new_actors = [a for a in registry.all_actors() if a.active_from > date(2024, 9, 30)]
    new_actor_queries = [q for q in test_data.queries if q.source in new_actors or q.target in new_actors]
    # Compute metrics on just these queries
```

For leader transitions specifically: compare predictions involving `leader:USA_POTUS_46` (activated mid-rollout) against predictions involving `state:USA` (continuous). The state-level actor should be more reliable since its memory was built over the full training period.

### 9.5 Data Source Robustness

Evaluate the model's sensitivity to the structured event data source:

| Evaluation variant | Description | Expected behavior |
|-------------------|-------------|-------------------|
| POLECAT only | Remove GDELT events from test-period streaming | Slight degradation (less event volume) |
| GDELT only | Remove POLECAT events from test-period streaming | Moderate degradation (noisier coding) |
| Text only (no events) | Remove all structured events from test-period streaming | Tests text stream self-sufficiency |
| Events only (no text) | Remove all articles from test-period streaming | Tests event stream self-sufficiency |

This tests whether the model is robust to data source availability — important for deployment where one source may have outages.

---

## 10. Evaluation Report Structure

The final evaluation report should contain:

1. **Executive summary:** Overall Brier, BSS vs. each baseline, C-index, 1-paragraph conclusion on deployment readiness.
2. **Per-event-type report card** (Section 7.1 table for all 18 types, plus separate rows for START/END state-transition types).
3. **Reliability diagrams** (pre- and post-calibration, per event type, at each standard horizon).
4. **Survival curve comparison plots** (predicted vs. Kaplan-Meier, per event type).
5. **Memory & representation quality** (Section 4: clustering visualizations, memory trajectories, attention analysis, Phase 2 transfer metrics).
6. **Surprise signal evaluation** (Section 5: escalation prediction AUC, feature importance, interpretable examples of top-50 highest-surprise actor-days).
7. **Benchmark comparisons** (ViEWS, Polymarket, superforecaster, no-change).
8. **Ablation results** (Section 8 tables, with commentary on which components provide the most value).
9. **Robustness checks** (Section 9: temporal, actor sparsity, novelty, lifecycle, data source).
10. **Error analysis** (manually reviewed confident wrong predictions, with surprise diagnostics and attention patterns for each).
11. **Recommendations:** which event types and actor categories the model is reliable enough to deploy on, which need more work, and which ablation findings should inform the next training iteration.

All metrics, predictions, surprise scores, and analysis code are saved alongside the model checkpoint for reproducibility.
