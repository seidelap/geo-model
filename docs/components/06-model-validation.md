# Component 6: Model Validation & Calibration

## Purpose

Evaluate the trained model on held-out test data, apply post-hoc calibration, and benchmark against external baselines. This component produces the evidence needed to determine whether the model is useful and where it fails.

**Inputs:** Best model checkpoint from Component 5. Test dataset from Component 3 (Section 6). External benchmark data (ViEWS, Polymarket).

**Outputs:** Comprehensive evaluation report with per-event-type metrics, calibration parameters, reliability diagrams, and benchmark comparisons.

---

## 1. Evaluation Protocol

### 1.1 Strict Temporal Separation

The test set is strictly future relative to all training data. No information from the test period may leak into training, validation, or calibration.

```
Training:    ... ─── 2024-09-30
Validation:  2024-10-01 ─── 2025-03-31   (model selection, hyperparameter tuning)
Calibration: fit on validation set         (temperature scaling parameters)
Test:        2025-04-01 ─── 2026-03-31    (final evaluation, reported once)
```

### 1.2 Evaluation Procedure

```python
def evaluate(model: FullModel, test_data: TestDataset, calibration_params: dict) -> EvaluationReport:
    """
    Full evaluation pipeline.
    """
    predictions = {}
    actuals = {}

    # 1. Replay context: bring actor memories to the state at the start of the test period
    #    using data up to the training cutoff
    model.replay_to(date(2024, 9, 30))

    # 2. For each test reference date, make predictions
    for reference_date in test_data.reference_dates:
        # Update memories with data up to reference_date (simulating streaming inference)
        model.update_to(reference_date)

        # Predict for all test queries at this reference date
        for query in test_data.queries_at(reference_date):
            # Model outputs a full survival curve, not a single probability
            event_history = get_dyad_event_history(query.source, query.target)
            survival_curve = model.predict_survival(
                query.source, query.target, query.event_type, event_history
            )

            # Apply calibration (per-bin temperature scaling)
            cal_curve = calibrate_curve(survival_curve, query.event_type, calibration_params)

            predictions[query.id] = cal_curve  # full CDF
            actuals[query.id] = query.actual_event_time  # days to event, or censored

    # 3. Compute metrics (survival metrics + derived fixed-horizon Brier)
    return compute_all_metrics(predictions, actuals, test_data)
```

### 1.3 Streaming vs. Batch Evaluation

Two evaluation modes:

**Streaming (realistic):** The model processes events and articles in real-time order during the test period. Actor memories update continuously. Predictions are made at each reference date using the current memory state. This tests the model as it would actually be deployed.

**Batch (diagnostic):** The model's memories are frozen at the training cutoff. No updates during the test period. This isolates the predictive value of the initial state from the model's ability to incorporate new information. If batch performance is much worse than streaming, the memory update mechanism is contributing significant value.

Report both modes.

---

## 2. Primary Metrics

The model outputs survival curves, not single probabilities. Evaluation uses both **survival-native metrics** (which evaluate the full temporal distribution) and **derived fixed-horizon metrics** (which evaluate CDF read-offs at standard horizons for comparability with baselines and benchmarks).

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
            # i experienced the event before j
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

    BS(t) = (1/N) * Σ_i [F_i(t) - I(T_i <= t)]^2 * W_i(t)

    where W_i(t) is an inverse-probability-of-censoring weight (IPCW) to handle
    censored observations.
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

Fixed-horizon Brier scores are derived from the survival CDF for comparability with baselines and benchmarks:

```python
def fixed_horizon_brier(predicted_cdfs: list[Tensor], actual_times: list[float],
                        horizon_days: float) -> float:
    """
    Standard Brier score at a specific horizon, derived from the survival CDF.

    prediction = CDF(horizon_days) = P(event within horizon_days)
    actual = 1 if event occurred within horizon_days, else 0
    """
    predictions = [interpolate_cdf(cdf, horizon_days) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return np.mean([(p - y) ** 2 for p, y in zip(predictions, actuals)])
```

Report at standard horizons (7, 30, 90, 180 days) for comparison with Phase 0 LightGBM and external benchmarks. These are not separate predictions — they are read-offs from the same survival curve, so consistency (P(30d) ≥ P(7d)) is guaranteed.

### 2.4 Brier Skill Score (BSS)

```python
def brier_skill_score(bs_model: float, bs_baseline: float) -> float:
    """
    Improvement over a reference baseline. Positive means model beats baseline.
    BSS = 1 - BS_model / BS_baseline
    """
    return 1.0 - bs_model / bs_baseline
```

Compute BSS at the 30-day horizon (primary) against three baselines:
1. **Climatological baseline:** Always predict the historical base rate for each event type.
2. **No-change baseline:** Predict that whatever happened last month will happen this month. (Surprisingly strong — see ViEWS competition results.)
3. **Phase 0 LightGBM:** The tabular baseline from Component 5.

### 2.5 Log Loss (Derived)

```python
def log_loss_at_horizon(predicted_cdfs: list[Tensor], actual_times: list[float],
                        horizon_days: float, eps: float = 1e-7) -> float:
    """
    Negative log-likelihood at a specific horizon, derived from the CDF.
    More sensitive than Brier to confident wrong predictions.
    """
    predictions = [np.clip(interpolate_cdf(cdf, horizon_days), eps, 1 - eps) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for p, y in zip(predictions, actuals)])
```

### 2.6 Precision-Recall AUC (PR-AUC, Derived)

```python
from sklearn.metrics import average_precision_score

def pr_auc_at_horizon(predicted_cdfs: list[Tensor], actual_times: list[float],
                      horizon_days: float) -> float:
    """
    PR-AUC at a specific horizon, derived from the CDF.
    Appropriate for rare events where ROC-AUC is misleading.
    """
    predictions = [interpolate_cdf(cdf, horizon_days) for cdf in predicted_cdfs]
    actuals = [1 if t <= horizon_days else 0 for t in actual_times]
    return average_precision_score(actuals, predictions)
```

### 2.7 Expected Calibration Error (ECE, Derived)

```python
def ece_at_horizon(predicted_cdfs: list[Tensor], actual_times: list[float],
                   horizon_days: float, n_bins: int = 15) -> float:
    """
    ECE at a specific horizon. Measures whether CDF(horizon) matches
    empirical event frequency at that horizon.
    """
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

ECE is reported at each standard horizon (7, 30, 90, 180 days), before and after calibration.

---

## 3. Calibration

### 3.1 Per-Bin Temperature Scaling

The survival curve consists of K hazard logits per time bin. Calibrate by applying a learned temperature per event type per time bin on the validation set:

```python
def fit_bin_temperatures(val_hazard_logits: Tensor, val_event_bins: Tensor,
                         val_censored: Tensor, K: int) -> Tensor:
    """
    Fit per-bin temperatures that minimize the DeepHit NLL on the validation set.

    val_hazard_logits: [N_val, K] raw hazard logits
    val_event_bins: [N_val] observed time bin per example
    val_censored: [N_val] censoring indicators

    Returns: [K] temperatures, one per time bin
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

# Fit temperatures per event type
bin_temperatures = {}
for r in PLOVER_TYPES:
    bin_temperatures[r] = fit_bin_temperatures(
        val_hazard_logits[r], val_event_bins[r], val_censored[r], K
    )

# Apply at test time: divide hazard logits by temperature before sigmoid
def calibrate_curve(raw_logits: Tensor, event_type: str, bin_temperatures: dict) -> dict:
    T = bin_temperatures[event_type]
    hazard = torch.sigmoid(raw_logits / T)
    survival = torch.cumprod(1 - hazard, dim=-1)
    cdf = 1 - survival
    return {"hazard": hazard, "survival": survival, "cdf": cdf}
```

**Why per-bin:** The model may be systematically overconfident at near-term horizons (hazard too high in early bins) but well-calibrated at long-term horizons, or vice versa. Per-bin temperatures correct this.

**Fallback (simpler):** A single temperature per event type applied to all bins. Faster to fit, less expressive. Use as a starting point; upgrade to per-bin if calibration varies significantly across the curve.

### 3.2 Calibration Validation

After fitting temperatures on the validation set, evaluate ECE at each standard horizon on the test set:

| Metric | Pre-calibration | Post-calibration | Target |
|--------|----------------|-----------------|--------|
| ECE at 7 days | Measure | Measure | < 0.05 |
| ECE at 30 days | Measure | Measure | < 0.05 |
| ECE at 90 days | Measure | Measure | < 0.05 |
| ECE at 180 days | Measure | Measure | < 0.05 |
| ECE worst (any type × horizon) | Measure | Measure | < 0.10 |

If post-calibration ECE > 0.10 for any event type at any horizon, investigate: the model may be systematically biased for that type/horizon combination.

### 3.3 Reliability Diagrams

For each event type, produce reliability diagrams **at each standard horizon** (7, 30, 90, 180 days):
- X-axis: predicted CDF(horizon) (binned)
- Y-axis: observed event frequency within that bin at that horizon
- Perfect calibration: points lie on the diagonal
- Overconfident: points below the diagonal (predicted probability too high)
- Underconfident: points above the diagonal

Additionally, produce **survival curve comparison plots**: overlay the average predicted survival curve against the Kaplan-Meier empirical survival curve for each event type. These should be close if the model is well-calibrated across the full temporal range, not just at fixed horizons.

Generate diagrams for: (a) the full test set, (b) the first 6 months of test, (c) the last 6 months of test. If calibration degrades over time, the model may be drifting and needs more frequent recalibration.

---

## 4. Benchmark Comparisons

### 4.1 ViEWS Benchmark

The ViEWS platform (`github.com/views-platform`) provides standardized conflict prediction evaluations.

**Comparison methodology:**
1. Map our PLOVER event types to ViEWS conflict categories (primarily FIGHT → ViEWS state-based conflict).
2. Aggregate our country-pair predictions to country-month level (ViEWS predicts at country-month, not dyad-month).
3. Evaluate on the same test period as ViEWS 2023/24 competition submissions.
4. Report: our Brier score vs. ViEWS submitted models' Brier scores.

**Expected outcome:** Our model should be competitive with top ViEWS submissions for state-based conflict, while also covering the other 17 event types that ViEWS doesn't predict.

### 4.2 Polymarket Comparison

**Comparison methodology:**
1. Collect resolved Polymarket geopolitical questions from the test period.
2. For each question, manually map to relevant (actor_i, actor_j, event_type) queries and compute the corresponding horizon from the question's resolution date.
3. Query our model's survival CDF at the time the Polymarket question opened, reading off the CDF at the question's horizon.
4. Compute head-to-head Brier scores: our CDF(horizon) vs. Polymarket's implied probability at the same time.

**Challenges:**
- Polymarket questions are specific and nuanced ("Will Russia and Ukraine sign a ceasefire before December 31, 2025?"), while our model predicts generic event types. The mapping is imperfect, but the survival curve makes it natural — the question's deadline maps directly to a CDF read-off.
- Polymarket questions are non-randomly selected (biased toward salient, uncertain events). Report comparisons on the full question set and stratified by salience/liquidity.
- Small sample size: Polymarket may have only 20–50 resolved geopolitical questions in a given year. Statistical power is limited.

### 4.3 Superforecaster Baseline

Good Judgment Open provides aggregate human forecasts. Superforecasters achieve mean Brier ≈ 0.14 on geopolitical questions.

- This is the aspirational target for the full system.
- Comparison is approximate (different question sets, different evaluation periods).
- Report our aggregate Brier on comparable questions alongside the 0.14 benchmark.

### 4.4 No-Change Baseline

```python
def no_change_baseline(dyad_history: dict, reference_date: date, horizon: int) -> float:
    """
    Predict that whatever happened in the last period will happen in the next period.
    For binary targets: if any event of this type occurred in the last τ days, predict 1.
    """
    lookback_start = reference_date - timedelta(days=horizon)
    recent_events = dyad_history.get((source, target, event_type), [])
    recent = [e for e in recent_events if lookback_start <= e.date < reference_date]
    return 1.0 if len(recent) > 0 else 0.0
```

The no-change baseline is deceptively strong for conflict prediction (most peaceful dyads stay peaceful, most conflict dyads continue fighting). The ViEWS 2020/21 competition showed many sophisticated models fail to beat this.

**Requirement:** Our model must beat the no-change baseline on aggregate BSS. Failure indicates the model is not capturing meaningful dynamics.

---

## 5. Per-Event-Type Analysis

### 5.1 Event Type Report Card

For each of the 18 PLOVER event types, report:

| Metric | Value | Comparison |
|--------|-------|------------|
| Target type | survival / intensity | — |
| Base rate at 30d (test set) | % | — |
| C-index | | ≥ 0.70 target |
| IBS (integrated Brier) | | vs. Phase 0 |
| Brier at 7d (derived from CDF) | | vs. Phase 0 |
| Brier at 30d (derived from CDF) | | vs. Phase 0 |
| Brier at 90d (derived from CDF) | | vs. Phase 0 |
| BSS at 30d vs. climatological | | positive = good |
| BSS at 30d vs. no-change | | positive = neural beats persistence |
| BSS at 30d vs. Phase 0 (LightGBM) | | positive = neural beats tabular |
| PR-AUC at 30d | | vs. Phase 0 |
| ECE at 30d (post-calibration) | | < 0.05 target |
| Mean predicted vs. Kaplan-Meier | | visual agreement |

### 5.2 Error Analysis Categories

For each event type, categorize prediction errors:

**False positives (predicted event, didn't happen):**
- Were there near-misses (events of a similar type, or events that almost happened)?
- Was the model picking up on genuine escalation that didn't materialize?
- Is the model confusing verbal/hypothetical events with actual ones?

**False negatives (missed actual events):**
- Was the event a genuine surprise (no precursors in the data)?
- Did the model have sufficient data on both actors?
- Was there a data pipeline failure (event not coded, or coded late)?

**Confident wrong predictions (high probability, wrong direction):**
- These are the most dangerous errors for deployment.
- Log and manually review every prediction where |p - y| > 0.8.
- Look for systematic patterns (e.g., model consistently overconfident on specific dyads).

---

## 6. Ablation Studies

### 6.1 Component Ablations

Test the contribution of each model component by removing it and measuring performance degradation:

| Ablation | What's removed | Expected impact |
|----------|---------------|-----------------|
| No text stream | Remove Layer 2, rely only on structured events | Moderate degradation, especially for events with textual precursors |
| No event stream | Remove Layer 3, rely only on text | Moderate degradation, especially for high-frequency events |
| No actor self-attention | Remove Layer 4 | Small-moderate degradation, mainly for second-order effects |
| No Hawkes process | Remove temporal self-excitation | Degradation on bursty event types (FIGHT, PROTEST) |
| No curriculum learning | Full event-type weights from step 0 | Possible degradation on rare events |
| No temporal decay | Remove memory decay (λ=0) | Memories become stale; late-test performance degrades |
| Random initialization | Skip Phase 1 structural pretraining | Slower convergence, possible degradation |
| No Phase 2 pretraining | Skip self-supervised text pretraining | Possible degradation on text-driven predictions |

### 6.2 Hyperparameter Sensitivity

Test sensitivity to key hyperparameters:

| Hyperparameter | Test values | Metric to watch |
|----------------|-------------|-----------------|
| Embedding dimension d | 128, 256, 512 | C-index + IBS |
| Memory decay half-life | 30, 90, 180, 365 days | Late-test-period C-index |
| TBPTT window K | 25, 50, 75, 100 | C-index + training stability |
| Negative sampling ratio K_neg | 5, 10, 20 | C-index on rare events |
| Hard negative ratio | 0.0, 0.15, 0.30 | C-index on rare events |
| Self-attention layers | 1, 2, 3 | IBS on second-order effects |
| Number of time bins K | 10, 17, 30 | IBS (more bins = finer curve, but more parameters) |

---

## 7. Robustness Checks

### 7.1 Temporal Robustness

Evaluate on multiple test periods (Component 3, Section 6.2):

| Test period | C-index | IBS | BSS at 30d | Notes |
|-------------|---------|-----|------------|-------|
| 2025-04 – 2026-03 (primary) | | | | Primary evaluation |
| 2024-04 – 2025-03 (robustness 1) | | | | Should be similar |
| 2023-04 – 2024-03 (robustness 2) | | | | Should be similar |

If performance varies drastically across test periods, the model may be fitting to temporal artifacts rather than learning stable geopolitical dynamics.

### 7.2 Actor Sparsity Robustness

Stratify evaluation by actor data density:

| Actor category | Dyad count | C-index | IBS | BSS at 30d |
|----------------|------------|---------|-----|------------|
| Dense-Dense (G7 × G7) | | | | Expected: best performance |
| Dense-Sparse (G7 × small state) | | | | Expected: moderate |
| Sparse-Sparse (small × small) | | | | Expected: weakest |

The model should degrade gracefully with data sparsity, not catastrophically. If Sparse-Sparse performance is worse than the climatological baseline, the model is not generalizing to sparse actors and those predictions should not be trusted.

### 7.3 Event Type Novelty

Test on events between actor pairs that had **no prior events of that type** in the training data:

```python
novel_predictions = [
    (p, y) for query, p, y in test_results
    if count_prior_events(query.source, query.target, query.event_type, training_period) == 0
]
novel_brier = brier_score([p for p, _ in novel_predictions], [y for _, y in novel_predictions])
```

If novel-event performance is much worse than overall performance, the model is memorizing dyad-event-type co-occurrence patterns rather than learning general geopolitical dynamics.

---

## 8. Evaluation Report Structure

The final evaluation report should contain:

1. **Executive summary:** Overall Brier, BSS vs. each baseline, 1-paragraph conclusion.
2. **Per-event-type report card** (Section 5.1 table for all 18 types).
3. **Reliability diagrams** (pre- and post-calibration, per event type).
4. **Benchmark comparisons** (ViEWS, Polymarket, superforecaster, no-change).
5. **Ablation results** (Section 6 tables).
6. **Robustness checks** (Section 7 tables).
7. **Error analysis** (manually reviewed confident wrong predictions).
8. **Recommendations:** which event types and actor categories the model is reliable enough to deploy on, and which need more work.

All metrics, predictions, and analysis code are saved alongside the model checkpoint for reproducibility.
