# Model Testing

Handles evaluation metrics, calibration, benchmarking against external systems, and validation methodology. This component ensures the model produces well-calibrated probability estimates that demonstrably outperform baselines.

## Scope

This component covers all evaluation, calibration, and benchmarking infrastructure. It includes the calibration layer (Layer 6) since calibration is fundamentally an evaluation/correction mechanism rather than a core model component. It does **not** cover training loop logic (see `model_training`).

## Primary Metrics

### Brier Score (BS)

```
BS = (1/N) · Σᵢ (p_i - y_i)²
```

Mean squared error of probabilistic forecasts. Lower is better, range [0, 1]. Strictly proper scoring rule — the only way to minimize it is to report your true beliefs.

### Brier Skill Score (BSS)

```
BSS = 1 - BS_model / BS_climatology
```

Measures improvement over the climatological base rate (always predicting the historical event frequency). **Positive BSS means the model beats the naive baseline.** This is the minimum bar — any component that doesn't improve BSS over the previous phase should be reconsidered.

### Log Loss

```
L = -(1/N) · Σᵢ [y_i · log(p_i) + (1-y_i) · log(1-p_i)]
```

More sensitive than Brier to confident wrong predictions. Used as training objective; reported alongside Brier at evaluation.

### PR-AUC (Not ROC-AUC)

For rare events, ROC-AUC is misleading — the large number of true negatives inflates it. **Precision-Recall AUC is the appropriate metric for imbalanced problems.** Always report PR-AUC for rare event types (FIGHT, SANCTION, SEIZE).

### Expected Calibration Error (ECE)

```python
ECE = Σ_b (|B_b| / N) · |acc(B_b) - conf(B_b)|
```

Measures how well predicted probabilities match empirical frequencies. A perfectly calibrated model has ECE = 0.

## Calibration (Layer 6)

Neural networks trained on imbalanced data are systematically miscalibrated. A model outputting P=0.8 for rare events should be correct 80% of the time — but typically is overconfident.

### Temperature Scaling

The simplest and most robust calibration method (Guo et al., ICML 2017):

```python
P_calibrated(r) = sigmoid(score_r / T_r)
```

`T_r` is a scalar learned separately **per event type** on a held-out calibration set. Applied after training — does not change prediction rankings, only absolute magnitudes.

**Why per-event-type:** Rare events (coups, wars) are miscalibrated differently from common events (verbal diplomatic exchanges). A single global temperature is insufficient.

Reference implementation: `github.com/gpleiss/temperature_scaling`

### Reliability Diagrams

At each evaluation checkpoint:
1. Bin predictions by predicted probability
2. Plot mean predicted probability vs. observed frequency
3. A perfectly calibrated model lies on the diagonal
4. Compute ECE as scalar summary

```python
def ece(y_pred, y_true, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0
    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
        if mask.sum() > 0:
            bin_conf = y_pred[mask].mean()
            bin_acc = y_true[mask].mean()
            ece_val += mask.mean() * abs(bin_conf - bin_acc)
    return ece_val
```

## Benchmark Comparisons

### ViEWS System (Primary Academic Benchmark)

The primary benchmark for conflict forecasting. Open forecasts and evaluation code at `github.com/views-platform`.

- The ViEWS 2023/24 competition provides standardized country-month conflict fatality predictions
- **Key finding from ViEWS 2020/21:** Most sophisticated ML models were "surprised by conflict outbreak in previously peaceful locations" and beaten by a basic no-change model on escalation
- **Key finding from ViEWS 2023/24:** Simple tree-based models (XGBoost, Random Forest) with carefully engineered features often outperform complex neural approaches. Feature engineering quality dominates architectural choices

**Implication:** Build the LightGBM baseline first (Phase 1) and ensure positive BSS before investing in neural architecture.

### Polymarket Comparison

1. For each Polymarket geopolitical question, identify relevant actors and event type
2. Query the model's probability at the same time Polymarket opened
3. Compute head-to-head Brier scores over resolved questions
4. Control for question difficulty (Polymarket questions may be non-randomly selected)

**Note:** The mapping from dyadic event-type probabilities to question-specific probabilities requires a semantic layer — ideally an LLM that parses the question, identifies relevant actors/event types, and aggregates. This interface is understudied (see Open Research Questions in the architecture doc).

### Superforecaster Baseline

Good Judgment Open provides aggregate human forecasts. Superforecasters achieve mean Brier ≈ 0.14 on geopolitical questions. This is the target for a competitive system.

### No-Change Baseline

Always predict the current situation continues. This deceptively strong baseline beats many conflict forecasting models on escalation prediction. **Any proposed model must beat this baseline to be considered meaningful.**

### Unweighted Ensemble Baseline

ViEWS competitions found that unweighted ensembles of all submitted models are highly competitive, suggesting model diversity matters more than individual sophistication. Consider ensemble approaches across model families.

## Evaluation Methodology

### Temporal Train/Test Split

Never evaluate on data from the same time period as training. Use strict temporal splits:

- **Training window:** Historical data up to time T
- **Validation window:** T to T + Δ (for hyperparameter tuning, calibration)
- **Test window:** T + Δ to T + 2Δ (for final evaluation, never touched during development)

### Per-Event-Type Evaluation

Report all metrics broken down by event type. Aggregate metrics can mask poor performance on rare but important event types. A model that perfectly predicts CONSULT events but fails on FIGHT events is not useful.

### Evaluation Cadence

- **During training:** Track validation Brier score per epoch; save best checkpoint
- **Per training run:** Full metric suite on validation set
- **Per phase transition:** Full metric suite on held-out test set + benchmark comparisons

## Lessons from ViEWS Competitions

These empirical findings from the research community should guide evaluation:

1. Most neural models are beaten by no-change baseline on escalation — test this explicitly
2. Feature engineering quality dominates architecture choices — evaluate feature ablations
3. Unweighted ensembles are highly competitive — consider ensemble evaluation
4. Calibration is critical for rare events — always check reliability diagrams
5. Models struggle most with conflict *onset* in previously peaceful locations — create specific test sets for this

## Key Dependencies

- `sklearn.metrics` — Standard classification metrics
- `temperature_scaling` — Post-hoc calibration (`github.com/gpleiss/temperature_scaling`)
- `matplotlib` / `plotly` — Reliability diagrams, metric visualization
- `pycox` — Survival analysis evaluation metrics

## Build Phase Mapping

| Phase | Evaluation Focus |
|-------|-----------------|
| Phase 1 (Months 1–2) | Brier score, BSS, reliability diagrams for LightGBM baseline |
| Phase 2 (Months 3–5) | All metrics. Temperature scaling. Compare neural vs. Phase 1 baseline |
| Phase 3 (Months 6–12) | Full benchmark suite. ViEWS comparison. Polymarket head-to-head. Ensemble evaluation |

## Architecture Reference

Corresponds to **Layer 6: Calibration** (Section 11) and **Evaluation and Benchmarking** (Section 16) in the architecture design document.
