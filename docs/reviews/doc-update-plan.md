# Documentation Update Plan

Based on the design gap analysis review and owner decisions on gaps 4, 7, 8, 11, 15, and 16.

---

## Summary of Decisions

| Gap | Decision | Rationale |
|-----|----------|-----------|
| 4   | Move temporal decay to once-per-day | Apply alongside daily self-attention, not per-update |
| 7   | Remove Hawkes excitation entirely | Actor memories already encode recent history; eliminates per-dyad event history blocker |
| 8   | Clarify TBPTT counts in days, not items | K=75 days; per-actor updates per day are small |
| 11  | Update architecture doc to match C4 | C4's survival curve is authoritative |
| 15  | Add dropout to per-type hazard heads | Same architecture, reduced overfitting risk |
| 16  | Dismiss — no change needed | Coverage *about* blackouts provides signal through the text stream |

---

## Change 1: Remove Hawkes Excitation (Gap 7)

Removes the Hawkes self-excitation component and the separate IntensityHead for high-frequency events. All 18 event types use the survival curve (discrete hazard) as their primary output. High-frequency types that previously used an intensity framing are handled by the survival head producing near-zero short-term survival (which is the correct output for frequent events).

### Files to modify:

**`docs/architecture-design.md`:**
- Section 2 (system overview diagram): Remove "Hawkes process for temporal prediction" from Layer 5 box. Keep "Survival model for time-to-event".
- Section 10.3: Delete the entire Hawkes Process section (lines ~603-628). Fold the insight ("events beget events") into a note in Section 10.4 explaining that actor memories encode recent event history, so the hazard logits naturally reflect temporal clustering.
- Section 10.4: Keep DeepHit survival model. Note that this is the sole output for all 18 types.
- Section 15.5 (Total Loss): Remove L_hawkes references.
- Section 18 (Phase 3 description): Remove "Transformer Hawkes Process" from the list. Update expected outcome text.
- Section 19 (dependencies): Remove "Transformer Hawkes Process" row from the table.

**`docs/components/03-target-event-definition.md`:**
- Section 1.4: Remove the intensity/Hawkes framing for high-frequency types. All types use the survival curve. Add a note that high-frequency types will have near-zero short-term survival and the model learns this from data.
- Remove the "Which event types use which framing" table or simplify to: all types use survival curves.

**`docs/components/04-model-architecture.md`:**
- Section 6.3 (HawkesExcitation class): Delete entirely.
- Section 6.4 (IntensityHead class): Delete entirely.
- Update the SurvivalHead.forward() to remove `excitation` addition — hazard is simply `sigmoid(logits)`.
- Section 10 (Parameter Count): Remove Hawkes excitation (~7K) and Intensity head (~0.2M) rows. Update totals.
- Section 11.3 (shared learned parameters): Remove Hawkes excitation and Intensity head rows.

**`docs/components/05-training-pipeline.md`:**
- Section 5.3 (training loop): Remove Hawkes excitation parameters from the LR table.
- Section 7.2: Delete the entire `hawkes_nll` loss function section.
- Section 7.4: Remove `L_intensity_example` block. All examples use the survival loss.
- Section 8.3: Update best model selection metric to remove Hawkes NLL.

**`docs/components/06-model-validation.md`:**
- Section 2.5: Delete the Hawkes Intensity Metrics section.
- Section 8.1 (ablations): Remove "No Hawkes excitation" row. Add a note: "Hawkes excitation was considered but removed during design review. If bursty-event calibration is poor, re-introduce as an additive excitation term on hazard logits."

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 7 status to "Resolved: removed Hawkes excitation from architecture."
- Update summary tables.

---

## Change 2: Temporal Decay Once Per Day (Gap 4)

Move temporal decay from per-update (inside Layer 2 text processing) to once-per-day (alongside Layer 4 self-attention and EMA baseline update). This means within a single day, neither the text stream nor event stream applies decay — they both operate on the current-day memory state. At day's end: (1) apply temporal decay to all actors based on days since last decay, (2) run self-attention, (3) update EMA baselines.

### Files to modify:

**`docs/components/04-model-architecture.md`:**
- Section 3.6 (text memory update): Remove `apply_decay()` from the per-article update. The update becomes: `h_new = h_i + gate_dims * delta_h` (no decay step).
- Section 4.3 (GRU memory update): Unchanged — was already not decaying (which was the gap).
- Add a new section or subsection under Layer 4 (or between L4 and L5): "Daily Memory Maintenance" describing the once-per-day sequence:
  1. Apply `h_i = b_i + exp(-λ * dt) * (h_i - b_i)` for all actors where dt = days since last decay
  2. Run actor self-attention (Layer 4)
  3. Update EMA baselines: `b_i = α * b_i + (1-α) * h_i`

**`docs/components/05-training-pipeline.md`:**
- Section 5.3 (training loop): Move the decay step from inside the article-processing loop to the daily maintenance block (after events, before self-attention). Add explicit `apply_decay()` call.

**`docs/architecture-design.md`:**
- Section 6.2 (temporal decay): Clarify that decay is applied once per day, not per update.
- Section 12.6 (recommended gating config): Update the decay formula to reflect daily application.

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 4 status to "Resolved: temporal decay moved to once-per-day."

---

## Change 3: TBPTT Counts in Days (Gap 8)

Clarify that TBPTT window K=75 means 75 *simulated days*, not 75 individual article/event processing steps. Within each day, all articles and events are processed (building the computation graph), then self-attention runs. The TBPTT counter increments once per day.

### Files to modify:

**`docs/components/05-training-pipeline.md`:**
- Section 5.3 (training loop pseudocode): Change `memory_step_count` to `day_count`. Increment after each day's processing, not after each article/event. The TBPTT boundary check becomes `if day_count >= K`.
- Section 5.6 (TBPTT explanation): Clarify "K=75 days" rather than "K=75 memory update steps". Note: within a single day, gradients flow through all article/event updates for that day, plus self-attention.
- Section 5.7 (intermediate losses): Update M = K // 5 to mean "every 15 days" rather than "every 15 steps."
- Section 5.8: Update "~100K TBPTT windows" to reflect the new counting.

**`docs/architecture-design.md`:**
- Section 13.4: Update TBPTT description to reference days, not individual updates.

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 8 status to "Resolved: TBPTT counts in days, not individual updates."

---

## Change 4: Architecture Doc Section 10 → Match C4 (Gap 11)

The architecture doc's Section 10 describes an earlier design (per-horizon binary classifier with `time2vec(τ)` input). Update to match C4's authoritative survival curve design.

### Files to modify:

**`docs/architecture-design.md`:**
- Section 10.1 (dyadic representation): Remove `time2vec(τ)` from the concatenation. Add `surprise_i`, `surprise_j` features. Match C4 Section 6.1.
- Section 10.2 (multi-task prediction head): Replace binary `P_event = sigmoid(scores[r])` with the discrete hazard → survival curve formulation from C4 Section 6.2. Reference the SurvivalHead class.
- Section 10.3: Already deleted (Hawkes removal, Change 1).
- Section 10.4: Promote to the primary prediction mechanism. Remove the old `MLP_hazard(concat(d_ij, time2vec(t)))` formulation. Replace with the C4 hazard-bins approach.

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 11 status to "Resolved: architecture doc updated to match C4."

---

## Change 5: Add Dropout to Per-Type Hazard Heads (Gap 15)

Keep the same architecture (shared trunk + 18 × `nn.Linear(2d, K=17)`) but add dropout (0.2) before each per-type hazard head. This reduces overfitting risk for rare event types without changing the architecture structure.

### Files to modify:

**`docs/components/04-model-architecture.md`:**
- Section 6.2 (SurvivalHead class): Add `nn.Dropout(0.2)` applied to `trunk` before passing to each per-type hazard head. Update the code snippet.

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 15 status to "Resolved: added dropout before per-type hazard heads."

---

## Change 6: Update Gap Analysis Summary (Gap 16)

Dismiss Gap 16 with explanation.

### Files to modify:

**`docs/reviews/design-gap-analysis.md`:**
- Update Gap 16 status to "Dismissed: media coverage *about* blackouts (internet shutdowns, censorship, conflict zone media restrictions) provides signal through the text stream. Articles describing the blackout itself update actor memories with conflict-associated representations, countering the false-decay concern."
- Move from "Important" to "Dismissed" in the summary table.
