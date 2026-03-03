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
Training:    ... ─── 2022-06-30
Validation:  2022-07-01 ─── 2022-12-31   (model selection, hyperparameter tuning)
Calibration: fit on validation set         (temperature scaling parameters)
Test:        2023-01-01 ─── 2023-12-31    (final evaluation, reported once)
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
    model.replay_to(date(2022, 6, 30))

    # 2. For each test reference date, make predictions
    for reference_date in test_data.reference_dates:
        # Update memories with data up to reference_date (simulating streaming inference)
        model.update_to(reference_date)

        # Predict for all test queries at this reference date
        for query in test_data.queries_at(reference_date):
            raw_pred = model.predict(query.source, query.target, query.event_type, query.horizon)

            # Apply calibration
            cal_pred = calibrate(raw_pred, query.event_type, calibration_params)

            predictions[query.id] = cal_pred
            actuals[query.id] = query.actual_label

    # 3. Compute metrics
    return compute_all_metrics(predictions, actuals, test_data)
```

### 1.3 Streaming vs. Batch Evaluation

Two evaluation modes:

**Streaming (realistic):** The model processes events and articles in real-time order during the test period. Actor memories update continuously. Predictions are made at each reference date using the current memory state. This tests the model as it would actually be deployed.

**Batch (diagnostic):** The model's memories are frozen at the training cutoff. No updates during the test period. This isolates the predictive value of the initial state from the model's ability to incorporate new information. If batch performance is much worse than streaming, the memory update mechanism is contributing significant value.

Report both modes.

---

## 2. Primary Metrics

### 2.1 Brier Score (BS)

```python
def brier_score(predictions: list[float], actuals: list[int]) -> float:
    """
    Mean squared error of probabilistic forecasts. Lower is better. Range [0, 1].
    Strictly proper scoring rule: minimized when predicted probabilities equal true probabilities.
    """
    return np.mean([(p - y) ** 2 for p, y in zip(predictions, actuals)])
```

Report per event type and aggregated (macro-average across event types, not micro-average across examples, to avoid domination by common types).

### 2.2 Brier Skill Score (BSS)

```python
def brier_skill_score(bs_model: float, bs_baseline: float) -> float:
    """
    Improvement over a reference baseline. Positive means model beats baseline.
    BSS = 1 - BS_model / BS_baseline
    """
    return 1.0 - bs_model / bs_baseline
```

Compute BSS against three baselines:
1. **Climatological baseline:** Always predict the historical base rate for each event type.
2. **No-change baseline:** Predict that whatever happened last month will happen this month. (Surprisingly strong — see ViEWS competition results.)
3. **Phase 0 LightGBM:** The tabular baseline from Component 5.

### 2.3 Log Loss

```python
def log_loss(predictions: list[float], actuals: list[int], eps: float = 1e-7) -> float:
    """
    Negative log-likelihood. More sensitive than Brier to confident wrong predictions.
    """
    preds = np.clip(predictions, eps, 1 - eps)
    return -np.mean([y * np.log(p) + (1 - y) * np.log(1 - p) for p, y in zip(preds, actuals)])
```

Log loss penalizes confident wrong predictions catastrophically. A model that says P=0.99 when the actual outcome is 0 gets a huge penalty. This matters for deployment: overconfident wrong predictions are dangerous.

### 2.4 Precision-Recall AUC (PR-AUC)

```python
from sklearn.metrics import average_precision_score

def pr_auc(predictions: list[float], actuals: list[int]) -> float:
    """
    Area under the precision-recall curve. Appropriate for imbalanced/rare events.
    NOT ROC-AUC, which is misleading when true negatives dominate.
    """
    return average_precision_score(actuals, predictions)
```

Report PR-AUC per event type. For rare events (FIGHT, SEIZE), this is more informative than accuracy or ROC-AUC.

### 2.5 Expected Calibration Error (ECE)

```python
def expected_calibration_error(predictions: list[float], actuals: list[int], n_bins: int = 15) -> float:
    """
    Measures how well predicted probabilities match empirical frequencies.
    Perfect calibration: ECE = 0.
    """
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

ECE is reported before and after calibration (Section 3) to quantify calibration improvement.

---

## 3. Calibration

### 3.1 Temperature Scaling

Post-hoc calibration using a per-event-type learned temperature (Guo et al., ICML 2017):

```python
def fit_temperature(val_logits: Tensor, val_labels: Tensor) -> float:
    """
    Find the temperature T that minimizes NLL on the validation set.
    T > 1 softens (reduces confidence), T < 1 sharpens.
    """
    T = nn.Parameter(torch.tensor(1.5))
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(val_logits / T, val_labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return T.item()

# Fit one temperature per event type
temperature = {}
for r in PLOVER_TYPES:
    val_logits_r = get_val_logits(r)
    val_labels_r = get_val_labels(r)
    temperature[r] = fit_temperature(val_logits_r, val_labels_r)

# Apply at test time
def calibrate(raw_logit: float, event_type: str, temperature: dict) -> float:
    T = temperature[event_type]
    return sigmoid(raw_logit / T)
```

**Why per-event-type:** Rare events (FIGHT: base rate ~0.1%) and common events (CONSULT: base rate ~15%) have fundamentally different calibration profiles. A single global temperature cannot correct both.

### 3.2 Calibration Validation

After fitting temperatures on the validation set, verify on the test set:

| Metric | Pre-calibration | Post-calibration | Target |
|--------|----------------|-----------------|--------|
| ECE (macro avg) | Measure | Measure | < 0.05 |
| ECE per type (worst) | Measure | Measure | < 0.10 |

If post-calibration ECE > 0.10 for any event type, investigate: the model may be systematically biased for that type.

### 3.3 Reliability Diagrams

For each event type, produce a reliability diagram:
- X-axis: predicted probability (binned)
- Y-axis: observed frequency within that bin
- Perfect calibration: points lie on the diagonal
- Overconfident: points below the diagonal (model predicts higher probability than observed)
- Underconfident: points above the diagonal

Generate these diagrams for: (a) the full test set, (b) the first 6 months of test, (c) the last 6 months of test. If calibration degrades over time, the model may be drifting and needs more frequent recalibration.

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
2. For each question, manually map to relevant (actor_i, actor_j, event_type, horizon) queries.
3. Query our model at the time the Polymarket question opened.
4. Compute head-to-head Brier scores: our prediction vs. Polymarket's implied probability at the same time.

**Challenges:**
- Polymarket questions are specific and nuanced ("Will Russia and Ukraine sign a ceasefire before December 31, 2023?"), while our model predicts generic event types ("AGREE between state:RUS and state:UKR within 180 days"). The mapping is imperfect.
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
| Base rate (test set) | % | — |
| Brier Score | | vs. Phase 0 |
| BSS vs. climatological | | positive = good |
| BSS vs. no-change | | positive = good |
| BSS vs. Phase 0 (LightGBM) | | positive = neural beats tabular |
| Log Loss | | vs. Phase 0 |
| PR-AUC | | vs. Phase 0 |
| ECE (post-calibration) | | < 0.05 target |
| Temperature T_r | | >1 = overconfident, <1 = underconfident |

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
| No graph propagation | Remove Layer 4 | Small-moderate degradation, mainly for second-order effects |
| No Hawkes process | Remove temporal self-excitation | Degradation on bursty event types (FIGHT, PROTEST) |
| No curriculum learning | Full event-type weights from step 0 | Possible degradation on rare events |
| No temporal decay | Remove memory decay (λ=0) | Memories become stale; late-test performance degrades |
| Random initialization | Skip Phase 1 structural pretraining | Slower convergence, possible degradation |
| No Phase 2 pretraining | Skip self-supervised text pretraining | Possible degradation on text-driven predictions |

### 6.2 Hyperparameter Sensitivity

Test sensitivity to key hyperparameters:

| Hyperparameter | Test values | Metric to watch |
|----------------|-------------|-----------------|
| Embedding dimension d | 128, 256, 512 | Aggregate BSS |
| Memory decay half-life | 30, 90, 180, 365 days | Late-test-period BSS |
| TBPTT window K | 25, 50, 75, 100 | Aggregate BSS + training stability |
| Negative sampling ratio K_neg | 5, 10, 20 | PR-AUC on rare events |
| Hard negative ratio | 0.0, 0.15, 0.30 | PR-AUC on rare events |
| Graph attention layers | 1, 2, 3 | BSS on second-order effects |
| Focal loss gamma | 0, 1, 2, 3 | PR-AUC on rare events |

---

## 7. Robustness Checks

### 7.1 Temporal Robustness

Evaluate on multiple test periods (Component 3, Section 6.2):

| Test period | Brier | BSS vs. baseline | Notes |
|-------------|-------|-------------------|-------|
| 2023 (primary) | | | Primary evaluation |
| 2022 (robustness 1) | | | Should be similar |
| 2021 (robustness 2) | | | Should be similar |

If performance varies drastically across test periods, the model may be fitting to temporal artifacts rather than learning stable geopolitical dynamics.

### 7.2 Actor Sparsity Robustness

Stratify evaluation by actor data density:

| Actor category | Dyad count | Brier | BSS |
|----------------|------------|-------|-----|
| Dense-Dense (G7 × G7) | | | Expected: best performance |
| Dense-Sparse (G7 × small state) | | | Expected: moderate |
| Sparse-Sparse (small × small) | | | Expected: weakest |

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
