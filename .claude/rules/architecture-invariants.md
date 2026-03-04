# Architecture Invariants

- Actor memory vectors h_i(t) must always be shape [d] where d comes from config.
- Survival curves must be monotonically non-increasing. Enforce via discrete hazard
  parameterization: hazard h[k] ∈ (0,1), survival = cumulative product of (1 - h[k]).
- Time bins are fixed at K=17 non-uniform intervals:
  [0-1, 1-2, 2-3, 3-5, 5-7, 7-10, 10-14, 14-21, 21-30, 30-45,
   45-60, 60-90, 90-120, 120-150, 150-180, 180-270, 270-365]
- Negative samples must pass the feasibility filter (docs/components/03, Section 4).
- Phase 0 LightGBM BSS is the floor. No model variant ships if it regresses below this.
- Temporal splits are strictly chronological. No future data leakage.
  Self-supervised pretraining text must not include validation/test period articles.
- Event types are the 18 PLOVER categories. Do not add, remove, or merge categories
  without updating docs/architecture-design.md Section 4.
- EMA baseline is a computed value (not a learned parameter). Constrain α ∈ [0.95, 1.0].
