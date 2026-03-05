# Design Gap Analysis: Tensor Flow Mapping & Identified Issues

*Review of the full architecture design and component specifications (C1–C6)*

---

## Part 1: Complete Tensor Flow Map

Every tensor in the forward pass, traced from raw input through to prediction output.

### Layer 1: Actor Memory Store

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `h_i(t)` | `[d]` | Actor i's full state at time t — encodes dispositions, recent behavior, relationships | Updated by L2, L3, L4 | L2 (as query), L3 (as GRU hidden state), L4 (as attention token), L5 (prediction input) |
| `b_i(t)` | `[d]` | EMA baseline — slow-moving average of h_i, the "normal" for actor i | EMA of h_i, updated daily | Temporal decay target for h_i |
| `h_baseline_init_i` | `[d]` | Structural projection — deterministic function of fixed actor features + name | W_struct, W_name, W_gate applied to structural features and name encoding | Initializes h_i(0) and b_i(0) at epoch start |
| `t_last_updated_i` | scalar | Timestamp of last memory update for actor i | Set on each update | Temporal decay computation (dt = t_now - t_last) |
| `H` | `[N_actors, d]` | Full memory matrix — all actor states stacked | Concatenation of all h_i | Layer 4 self-attention input |
| `sketch_i` | `[sketch_dim]` | TF-IDF sketch for fast article-actor relevance | Random projection of TF-IDF vector | Layer 2 sketch filter |

### Layer 2: Text Processing Stream

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `article_sketch` | `[sketch_dim]` | Article's lightweight representation for relevance scoring | Random projection of TF-IDF | Dot product with actor sketches |
| `relevance_scores` | `[N_actors]` | Per-actor relevance to this article | `actor_sketches @ article_dense` | Sparsemax to get actor_weights |
| `actor_weights` | `[N_actors]` | Sparse selection weights — which actors this article updates | `sparsemax(relevance_scores / temperature)` | Filter to active_actors, scalar gate input |
| `T` | `[seq_len, 768]` | ConfliBERT token embeddings for the full article | `conflibert(article_text).last_hidden_state` | Cross-attention keys/values |
| `query_i` | `[d_k]` | Actor i's query vector for reading the document | `W_Q @ h_i` | Dot product with keys |
| `keys` | `[seq_len, d_k]` | Document tokens projected to key space | `W_K @ T` | Attention score computation |
| `values` | `[seq_len, d_v]` | Document tokens projected to value space | `W_V @ T` | Weighted sum to produce m_i |
| `attn_scores` | `[seq_len]` | Raw attention scores — actor i's affinity to each token | `query_i @ keys.T / sqrt(d_k)` | Softmax |
| `attn_weights` | `[seq_len]` | Normalized attention — which tokens actor i reads | `softmax(attn_scores)` | Weighted sum over values |
| `m_i` | `[d_v]` | What actor i extracted from this document | `attn_weights @ values` | Memory update input |
| `update_input` | `[d + d_v + time_dim]` | Concatenation of current state, reading, and time | `cat(h_i, m_i, time2vec(t))` | Gates and MLP |
| `gate_scalar` | scalar | Should this article update actor i at all? | `sparsemax(W_scalar @ update_input)` | Scales delta_h |
| `gate_dims` | `[d]` | Which memory dimensions should update? | `sigmoid(W_dims @ update_input)` | Element-wise mask on delta_h |
| `delta_h` | `[d]` | Candidate memory update | `MLP_update(update_input) * gate_scalar` | Added to decayed memory |
| `h_decayed` | `[d]` | Memory after temporal decay toward baseline | `b_i + exp(-λ*dt) * (h_i - b_i)` | Base for gated update |
| `h_new` | `[d]` | Updated actor memory | `h_decayed + gate_dims * delta_h` | Replaces h_i |

### Layer 3: Structured Event Stream

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `e_r` | `[d]` | Learnable relation type embedding for this PLOVER type | `relation_embeddings[event_type]` | Event encoding, also used in L5 |
| `goldstein_scalar` | `[1]` | Normalized conflict-cooperation score [-1, 1] | `event.goldstein_score / 10` | Event encoding |
| `mode_encoding` | `[3]` | One-hot: verbal / hypothetical / actual | One-hot of event_mode | Event encoding |
| `magnitude_features` | `[3]` | Log-transformed casualties, injuries, group size | `log1p(dead, injured, size)` | Event encoding |
| `time2vec(t)` | `[time_dim]` | Temporal encoding of event timestamp | Time2Vec module | Event encoding |
| `e_event` | `[d + 7 + time_dim]` | Full encoded event | Concatenation of above | GRU input |
| `source_input` | `[d + 7 + time_dim + d]` | GRU input for source actor | `cat(e_event, h_target)` | GRU_source |
| `target_input` | `[d + 7 + time_dim + d]` | GRU input for target actor | `cat(e_event, h_source)` | GRU_target |
| `h_source_new` | `[d]` | Updated source actor memory | `GRU_source(source_input, h_source)` | Replaces h_source |
| `h_target_new` | `[d]` | Updated target actor memory | `GRU_target(target_input, h_target)` | Replaces h_target |

### Layer 4: Actor Self-Attention

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `H` | `[N_active, d]` | All active actor memories, stacked | Gathered from per-actor h_i | Self-attention |
| `H_norm` | `[N_active, d]` | LayerNorm'd actor memories | `LayerNorm(H)` | MHA Q, K, V |
| `H_attn` | `[N_active, d]` | Self-attention output | `MHA(H_norm, H_norm, H_norm)` | Residual addition |
| `H_updated` | `[N_active, d]` | Post-attention + FFN actor memories | `H + FFN(LayerNorm(H + H_attn))` | Written back to per-actor h_i |
| `attn_weights` | `[n_heads, N_active, N_active]` | Per-head attention matrix — which actors attend to which | MHA internal | Diagnostic only (analysis) |

### Layer 5: Event Prediction Head

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `h_i` | `[d]` | Source actor state at prediction time | Layer 1 (post L4) | Dyadic representation |
| `h_j` | `[d]` | Target actor state at prediction time | Layer 1 (post L4) | Dyadic representation |
| `h_i * h_j` | `[d]` | Element-wise product — symmetric compatibility | Computation | Dyadic representation |
| `h_i - h_j` | `[d]` | Difference — directional asymmetry | Computation | Dyadic representation |
| `abs(h_i - h_j)` | `[d]` | Absolute difference — undirected distance | Computation | Dyadic representation |
| `e_r` | `[d]` | Relation type embedding | `relation_embeddings[r]` | Dyadic representation |
| `surprise_i` | `[2]` | Source actor surprise: [CPC_score, event_type_KL] | CPC + EventType heads | Dyadic representation |
| `surprise_j` | `[2]` | Target actor surprise: [CPC_score, event_type_KL] | CPC + EventType heads | Dyadic representation |
| `d_ij` | `[6d + 4]` | Full dyadic representation | Concatenation of above | Survival head shared trunk |
| `trunk` | `[2d]` | Compressed dyadic features | `shared_MLP(d_ij)`: (6d+4) → 4d → 2d | Per-type hazard heads |
| `logits` | `[K=17]` | Raw hazard logits per time bin | `hazard_heads[r](trunk)` | Sigmoid → hazard |
| `excitation` | `[K=17]` | Hawkes self-excitation contribution | HawkesExcitation from event history | Added to logits |
| `hazard` | `[K=17]` | Per-bin conditional hazard P(event in bin k \| survived to k) | `sigmoid(logits + excitation)` | Survival computation |
| `survival` | `[K=17]` | Survival function S(τ) per bin | `cumprod(1 - hazard)` | CDF derivation |
| `cdf` | `[K=17]` | Cumulative event probability F(τ) per bin | `1 - survival` | Fixed-horizon lookups, loss |
| `pdf` | `[K=17]` | Probability mass per bin | `hazard * survival_shifted` | DeepHit NLL loss |
| `goldstein_pred` | `[1]` | Predicted Goldstein score (auxiliary) | `goldstein_head(trunk)` | MSE loss |
| `p_escalation` | `[1]` | Binary escalation probability (auxiliary) | `sigmoid(escalation_head(trunk))` | BCE loss |

### CPC + Event-Type Prediction Heads (Auxiliary)

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `z_pred` | `[d_cpc=128]` | Memory projected to prediction space | `W_pred(h_i)`, L2-normalized | CPC contrastive loss |
| `z_target` | `[d_cpc=128]` | Next-day article aggregate in prediction space | `W_target(article_aggregate)`, L2-normalized | CPC contrastive loss |
| `article_aggregate` | `[768]` | Mean-pooled ConfliBERT embedding of next-day articles for actor i | `mean_pool(conflibert(articles))` | CPC target encoder |
| `event_type_logits` | `[18]` | Predicted event-type distribution for next 7-day window | `event_type_predictor(h_i)` | KL divergence loss |
| `event_type_counts` | `[18]` | Actual PLOVER event counts in next 7-day window | Counted from structured events | KL divergence target |

### Intensity Head (High-Frequency Events)

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `mu` | scalar | Base event rate (events/day) from actor states | `base_rate_mlp(cat(h_i, h_j))` | Intensity function |
| `excitation` | `[T_query]` | Hawkes self-excitation at each query time | Sum of exp-decayed past events | Added to mu |
| `lambda_t` | `[T_query]` | Total intensity at each query time | `mu + excitation` | Hawkes NLL loss |

### Layer 6: Calibration

| Tensor | Shape | Semantic Meaning | Source | Consumed By |
|--------|-------|-----------------|--------|-------------|
| `T_cal` | `[K=17]` | Per-bin temperature (per event type) | LBFGS-fitted on validation set | Divides hazard logits |
| `calibrated_hazard` | `[K=17]` | Temperature-scaled hazard | `sigmoid(logits / T_cal)` | Calibrated survival curve |

---

## Part 2: Identified Design Gaps and Issues

### GAP 1: Cross-Attention Dimension Mismatch (Layer 2)

**Tensors involved:** `h_i [d]` → `W_Q` → `query_i [d_k]` vs `T [seq_len, 768]` → `W_K` → `keys [seq_len, d_k]`

**Issue:** The actor memory lives in `d`-dimensional space (256 or 512) while ConfliBERT output is 768-dimensional. The design specifies `W_Q: d → d_k` and `W_K: 768 → d_k`, but `d_v` (the value projection output dimension) feeds into the memory update as `m_i [d_v]`. The update_input is `cat(h_i [d], m_i [d_v], time2vec [time_dim])` → projected to `[d]`.

The gap: if `d_v ≠ d`, this is fine because W_proj handles the dimension change. But the design never explicitly specifies the relationship between d_k, d_v, and d. The parameter count table says `W_Q, W_K, W_V, W_cond` at `d_k = d_v = 256`, but the architecture doc's Section 7.3 shows `m_i = mean_pool(T[mention_spans])` which would be `[768]`, not `[d_v]`. There are two different code paths described — one with cross-attention (Component 4 Section 3.5, producing `m_i [d_v]`) and one with mean-pooling of mention spans (architecture doc Section 7.3, producing `m_i [768]`).

**Risk:** If these aren't reconciled during implementation, the text stream will either use the wrong dimension for memory updates or silently truncate information.

**Recommendation:** Standardize on the cross-attention path from Component 4. Explicitly set `d_k = d_v = d` (or d/n_heads for multi-head) and document it. Drop the mention-span pooling path from the architecture doc.

---

### GAP 2: Scalar Gate Semantics Conflict Between Sketch Filter and Memory Update

**Tensors involved:** `actor_weights [N_actors]` from sketch filter sparsemax vs `gate_scalar` from `W_scalar @ update_input`

**Issue:** There are two separate sparsity mechanisms that serve overlapping purposes:

1. **Sketch-based filter** (Section 3.3): `sparsemax(relevance_scores / temperature)` → produces `actor_weights` that determine which actors are "active" for this article. Only actors with weight > 0 proceed.
2. **Scalar gate** (Section 3.6): `sparsemax(W_scalar @ update_input)` → a second gate that further controls whether the article should update this actor.

The architecture doc (Section 7.5) says `gate_scalar = sparsemax(W_scalar @ update_input)` and describes it as "competition across actors." But in Component 4 (Section 3.6), the gate_scalar is computed *per actor independently* from `update_input` which includes `h_i` and `m_i` for that specific actor. Sparsemax over a single scalar doesn't produce sparsity — sparsemax requires a vector of competing alternatives.

**Risk:** If gate_scalar is computed per-actor independently (not across actors), the sparsemax call is meaningless (it's just clamp-to-[0,1] on a scalar). If it IS computed across actors, then the computation requires knowing all actors' update_inputs simultaneously, which conflicts with the per-article sequential processing loop shown in Component 5 Section 5.3.

**Recommendation:** Clarify the design. Two clean options:
- **Option A:** Sketch filter provides the cross-actor competition (sparsemax). The per-actor gate_scalar uses sigmoid (independent decision per actor). This is simpler and avoids the batch-computation issue.
- **Option B:** Collect all active actors' update_inputs, run sparsemax across them. But this creates a dependency between actors' updates within a single article, complicating the training loop.

Option A is strongly recommended.

---

### GAP 3: GRU Hidden State Dimension vs Input Dimension (Layer 3)

**Tensors involved:** `source_input [d + 7 + time_dim + d]` as GRU input, `h_source [d]` as GRU hidden state

**Issue:** The GRU cell expects `input_dim` and `hidden_dim`. The design specifies:
- Input: `cat(e_event [d + 7 + time_dim], h_target [d])` = `[2d + 7 + time_dim]`
- Hidden state: `h_source [d]`

Standard GRU: `h_new = GRU(input, h_prev)` where input_dim can differ from hidden_dim. This is fine.

However, the GRU *replaces* the actor's memory entirely (`h_source_new = GRU_source(source_input, h_source)`). There is no temporal decay applied before the GRU update (unlike the text stream which applies `apply_decay` first), and no gating mechanism to control update magnitude. A single coded event can arbitrarily shift the entire memory vector.

**Risk:** In practice, GDELT produces 100K+ events/day. If many events hit the same actor on the same day, the GRU will be called repeatedly with each event replacing the memory. The GRU's internal gates provide *some* stability, but there's no per-event relevance gating (analogous to the text stream's scalar gate). This makes the event stream much "noisier" per-update than the text stream, despite the design's intent that text provides "richer" signal.

**Overfitting risk:** The GRU has full capacity to memorize event sequences. With 2d+7+time_dim input dimension and d hidden dimension, each GRU cell has ~3*(input_dim * d + d^2 + d) parameters. At d=256 and time_dim=16, this is ~3*(535*256 + 256^2 + 256) ≈ 608K per GRU. Two GRUs = 1.2M parameters devoted to processing relatively simple structured events (7 scalar features + a learned embedding).

**Recommendation:**
1. Apply temporal decay *before* each GRU update, same as the text stream
2. Add a scalar gate to the event stream: `gate = sigmoid(W_event_gate @ cat(e_event, h_source))`, then `h_source_new = h_source + gate * (GRU(...) - h_source)` — residual-gated rather than full-replacement
3. Consider deduplicating events within a day before processing (the design mentions dedup in C1 and C3, but the training loop in C5 processes `day.events` without mention of intra-day dedup)

---

### GAP 4: Temporal Decay Not Applied Before GRU in Event Stream

**Tensors involved:** `h_source [d]` and `h_target [d]` before GRU update in Layer 3

**Issue:** The text stream (Layer 2, Section 3.6) explicitly applies temporal decay before the gated update:
```
h_decayed = apply_decay(h_i, b_i, t_last, t, lambda_decay)
h_new = h_decayed + gate_dims * delta_h
```

But the event stream (Layer 3, Section 4.3) does not:
```
h_source_new = GRU_source(source_input, h_source)  # no decay first
```

**Risk:** If the text stream processed an article at t=100 and the next event for that actor arrives at t=130, the event stream's GRU sees a memory that's 30 days stale without decay. The memory should have decayed toward baseline during those 30 days. This creates inconsistency: the same actor's memory will look different depending on which stream processes it next.

The training loop in C5 Section 5.3 processes articles first, then events, within each day. So within a day there's no gap. But across days, if an actor has events on day 50 and day 80 with no intervening articles, 30 days of un-decayed memory will feed the GRU.

**Recommendation:** Apply `apply_decay(h_i, b_i, t_last, t_now, lambda_decay)` at the start of every memory update operation, regardless of which stream triggers it.

---

### GAP 5: Asymmetric Dyad Representation vs Symmetric Event Types

**Tensors involved:** `d_ij [6d+4]` includes `h_i - h_j [d]` (directional) alongside `abs(h_i - h_j) [d]` (symmetric)

**Issue:** The dyadic representation includes both `h_i - h_j` (directional asymmetry) and `|h_i - h_j|` (symmetric distance). For symmetric event types (AGREE, CONSULT, ENGAGE, COOP), the model receives a directional signal that shouldn't matter — an agreement between USA and Japan should have the same probability regardless of which is labeled "source."

Component 3 Section 2.3 handles this at the training data level: symmetric types generate two directed examples (i→j and j→i). But this doubles the training data for these types and forces the model to learn that `h_i - h_j` should be ignored for symmetric events, rather than encoding symmetry structurally.

**Risk:** Moderate. The model can learn to ignore the directional component for symmetric types, but:
- It wastes capacity (half the dyadic representation is useless for 4 of 18 types)
- The doubled training data inflates the loss contribution of symmetric types
- At inference time, you must query both directions and average (or pick one), adding complexity

**Recommendation:** For symmetric event types, either:
- Zero out `h_i - h_j` in the dyadic representation and use only the symmetric components
- Or use a separate symmetric prediction head for AGREE/CONSULT/ENGAGE/COOP that operates on `cat(h_i + h_j, h_i * h_j, |h_i - h_j|, e_r)`

---

### GAP 6: EMA Baseline and Gradient Flow Interaction

**Tensors involved:** `b_i(t) [d]`, `h_i(t) [d]`, `alpha_baseline` (scalar)

**Issue:** The EMA baseline update is `b_i(t) = alpha * b_i(t-1) + (1-alpha) * h_i(t)`. This runs once per day. The decay toward baseline is `h_decayed = b_i + exp(-λ*dt) * (h_i - b_i)`.

During TBPTT, gradients flow through h_i(t) backward through the memory update chain. The question is: do gradients also flow through b_i(t)?

If b_i(t) is part of the computation graph (which it should be, since alpha_baseline is a learned parameter), then the gradient path includes:
```
loss → h_i(t) → decay uses b_i(t) → b_i(t) = alpha * b_i(t-1) + (1-alpha) * h_i(t) → h_i(t) (circular!)
```

Within a single TBPTT window, `b_i` is updated K/daily_events_per_actor times (once per day within the window). Each update feeds back into the decay computation. This creates a recurrence through b_i that lengthens the effective gradient path beyond K steps.

**Risk:** The gradient path through the EMA baseline effectively extends beyond the TBPTT truncation window. With alpha=0.99, b_i barely moves per step, so gradients through it will be tiny (attenuated by 0.01 per day). This is probably fine in practice, but it's an unexamined interaction.

**Recommendation:** Document whether b_i(t) is inside or outside the computation graph during TBPTT. If inside: verify gradient magnitudes through the baseline path don't cause instability. If outside (detached): document that alpha_baseline only receives gradients through the decay formula `h_decayed = b_i + ...` where b_i is treated as a constant.

---

### GAP 7: Missing Event History for Hawkes Excitation at Prediction Time

**Tensors involved:** `event_history: list[tuple]` for `HawkesExcitation.forward()`

**Issue:** The Hawkes excitation component (Section 6.3) requires `event_history` — a list of `(event_time, event_type)` tuples for the specific dyad being predicted. But the design never specifies how this history is maintained or passed to the prediction head during training.

The training loop (C5 Section 5.3) processes events and articles day by day, updating memories. When `compute_prediction_losses()` is called at TBPTT boundaries, it needs to query the survival head for sampled dyads. Each query requires the dyad's event history — but the training loop doesn't maintain per-dyad event histories.

**Risk:** Implementation will hit this gap immediately. Options:
1. Maintain a per-dyad event buffer during the rollout (memory cost: O(N_actors^2 * avg_events))
2. Pre-compute event histories and look them up by (actor_i, actor_j, reference_date)
3. Approximate with a running summary (last-K events per dyad)

**Recommendation:** Add an explicit `DyadEventHistory` data structure to the training loop that maintains a sliding window of recent events per dyad. The context window definition in C3 Section 4.1 already specifies "dyad_events: last 365 days" — this should be materialized during the rollout.

---

### GAP 8: TBPTT Window Size vs Daily Processing Granularity

**Tensors involved:** All memory tensors across K=75 update steps

**Issue:** The TBPTT window is K=75 "memory update steps." But the training loop counts steps differently for articles and events:
- Each article processing increments `memory_step_count += 1`
- Each event processing increments `memory_step_count += 1`
- Daily self-attention does NOT increment the counter

With ~40K articles/day and ~100K events/day, 75 steps covers a tiny fraction of a single day. This means:
- Each TBPTT window spans <1 day of simulated time
- The daily self-attention (Layer 4) might run 0 times within most TBPTT windows (it only runs at day boundaries)
- Gradients from the prediction loss rarely flow through Layer 4

**Risk:** Layer 4 (actor self-attention) may be severely undertrained. If TBPTT boundaries rarely coincide with Layer 4 execution, the self-attention parameters receive sparse gradient signal.

The design says "run actor self-attention once per simulated day" and "compute losses at TBPTT boundaries." If a single day has 140K update steps (40K articles + 100K events), and K=75, there are ~1,866 TBPTT windows per day, but self-attention only runs once. Only 1 of those 1,866 windows will have Layer 4 in its computation graph.

**Recommendation:** This is a critical issue. Options:
1. **Batch articles and events per day, not per step.** Count TBPTT in days, not individual update steps. K=75 days is the window.
2. **Sub-sample articles and events.** Don't process all 40K articles per day — sample a manageable number (e.g., 100–500 most relevant articles per day).
3. **Decouple Layer 4 gradient from TBPTT.** Run self-attention daily but detach its input from the computation graph, treating Layer 4 as a separate module with its own loss (e.g., the CPC loss computed after self-attention).

Option 2 is likely necessary regardless for compute reasons (40K full ConfliBERT encodings per day per epoch is enormous). Option 1 changes the TBPTT semantics but aligns better with the daily processing rhythm.

---

### GAP 9: Phase 2 → Phase 3 Transition: Surprise Feature Cold Start

**Tensors involved:** `surprise_i [2]`, `surprise_j [2]` in the dyadic representation

**Issue:** The surprise features (CPC score and event-type KL) are computed from the CPC and event-type prediction heads. In Phase 3, these are concatenated to the dyadic representation and fed to the survival head. But:

1. At the start of each epoch's rollout, memories are reset to baselines. The CPC and event-type heads produce surprise scores by comparing today's predictions with tomorrow's actuals. For the first few days of the rollout, the memories are near-baseline and surprise scores are not yet meaningful.

2. The design says "set surprise features to zero during the first few days." But how many days? The CPC head needs at least 1 day of articles to produce a target. The event-type head needs a 7-day forward window. So surprise features are undefined for the first 7 days of each rollout.

3. More importantly: in Phase 2, the CPC and event-type heads are trained with weights 0.5 and 0.3 respectively. In Phase 3, they're downweighted to 0.1 each. This abrupt weight change at the Phase 2→3 transition may cause the heads to drift, making their surprise scores less reliable over the course of Phase 3 training.

**Risk:** The survival head may learn to ignore surprise features if they're frequently zero (early rollout) or noisy (after Phase 2→3 weight drop). The ablation study (C6 Section 8.1) will detect this, but it's better to prevent it.

**Recommendation:**
1. Use a linear warmup for surprise features during the first 14 days of each rollout (multiply by `min(1, day_index/14)`)
2. Consider keeping the CPC and event-type loss weights at 0.3 in Phase 3 (not 0.1) to maintain head quality, or use a gradual weight decay schedule across Phase 3

---

### GAP 10: State-Transition Event Types Create a Hidden State Machine Not Tracked in Memory

**Tensors involved:** No tensor tracks state-transition status (active/inactive per event type per dyad)

**Issue:** Component 3 Section 6.3 defines state-transition event types (SANCTION, FIGHT, MOBILIZE, SEIZE, AID, REDUCE) that have START and END variants. The eligibility function `is_eligible()` checks `current_state.is_active` to determine whether a START or END prediction is valid.

But where does `current_state` come from? The model's actor memories `h_i(t)` encode everything implicitly. There's no explicit binary state variable tracking "is there an active sanction between USA and Russia right now?"

During training, `is_eligible()` presumably checks the ground-truth event history. But at inference time, the model must infer the current state from its own predictions and event history. If the model predicts a SANCTION_START but no corresponding SANCTION_END, is the sanction assumed to be ongoing? For how long?

**Risk:**
- Training/inference skew: training uses ground-truth eligibility; inference must self-determine eligibility
- Missing ground-truth state: the event data (GDELT/POLECAT) codes point events, not durations. How are START and END events derived from the raw event stream? The design says "preprocess to identify onset and termination dates" but doesn't specify the algorithm.
- The survival curve predicts time-to-START or time-to-END, but doesn't track whether we're currently in an active state

**Recommendation:**
1. Define an explicit state tracker: for each dyad and state-transition event type, maintain a binary `is_active` flag derived from the event history. This is a data structure, not a learned parameter.
2. Specify the START/END derivation algorithm: e.g., SANCTION_START = first SANCTION event in a cluster (no SANCTION events in prior 90 days). SANCTION_END = 90 days with no SANCTION events after a SANCTION_START.
3. At inference time, use the observed event history (not model predictions) to determine eligibility.

---

### GAP 11: Inconsistency Between Architecture Doc and Component 4 on Prediction Head

**Tensors involved:** `d_ij` composition differs between documents

**Issue:** The architecture doc (Section 10.1) includes `time2vec(τ)` (horizon encoding) in the dyadic representation, while Component 4 (Section 6.1) explicitly states "No horizon parameter — the model outputs a full survival curve." The architecture doc also shows `P_event = sigmoid(scores[r])` (binary prediction per type), while Component 4 defines the SurvivalHead with K=17 hazard bins.

These are two fundamentally different prediction architectures:
1. **Architecture doc:** Per-horizon binary classifier with time2vec(τ) input
2. **Component 4:** Horizon-free discrete hazard model outputting a full survival curve

**Risk:** If implemented per the architecture doc, you get a separate prediction per horizon (inconsistent across horizons). If implemented per Component 4, the survival curve guarantees monotonicity. Component 4 is clearly the more refined design.

**Recommendation:** Update the architecture doc Sections 10.1–10.4 to match Component 4. The architecture doc appears to be an earlier draft; Component 4 is the authoritative spec.

---

### GAP 12: No Mechanism for Handling Variable Actor Count in Self-Attention

**Tensors involved:** `H [N_active, d]` where N_active changes over time

**Issue:** The actor lifecycle system (C5 Section 5.4) activates and deactivates actors during the rollout. Layer 4 operates on `H [N_active, d]` where N_active varies. Standard `nn.MultiheadAttention` handles variable sequence lengths, so this isn't a PyTorch problem.

However, the EMA baseline update and temporal decay are applied per-actor. When a new actor is activated mid-rollout:
- Its memory starts at `h_baseline_init_i`
- Its EMA baseline starts at `h_baseline_init_i`
- It immediately participates in self-attention with all other actors

The other actors' attention patterns must suddenly accommodate a new entry. With pre-norm attention (`LayerNorm` before `MHA`), this is handled reasonably, but there's no explicit warmup period for new actors' participation in self-attention.

**Risk:** Low, but worth noting. A new actor injected into self-attention with a baseline-initialized memory may cause transient attention pattern disruptions for existing actors, especially if the new actor's initial memory is far from the population mean.

**Recommendation:** Consider a "warmup mask" for newly activated actors: for the first N days after activation, the new actor can attend to others (receive information) but others cannot attend to the new actor (don't propagate uninitialized state). This is a minor refinement.

---

### GAP 13: ConfliBERT Sequence Length Truncation at 512 Tokens

**Tensors involved:** `T [seq_len, 768]` where seq_len ≤ 512

**Issue:** ConfliBERT (BERT-base) truncates at 512 tokens. The text filtering pipeline (C1 Section 1.2) keeps articles between 100–5000 words. At ~1.3 tokens per word, a 5000-word article is ~6500 tokens — 93% of which are discarded by truncation. Even a 1000-word article (~1300 tokens) loses 60% of its content.

The cross-attention mechanism reads only the first 512 tokens. For long articles, the most relevant geopolitical content may be in later paragraphs (after background context in the lede).

**Risk:** Systematic information loss for longer articles. The actors mentioned later in articles won't have their mentions captured by cross-attention.

**Recommendation:**
1. **Chunk and pool:** Split long articles into overlapping 512-token chunks, encode each, and concatenate the token sequences before cross-attention. This preserves all content at the cost of ~2-3x encoding time for long articles.
2. **Or:** Filter more aggressively in C1 — set the max article length to 1500 words (keeping within ~2000 tokens, truncating only 75% of content at most).
3. **Or:** Use a sentence-extraction step to select the most relevant sentences before encoding.

---

### GAP 14: Negative Sampling Corruption Only Corrupts One Element

**Tensors involved:** Negative training examples

**Issue:** The negative sampling strategy (C3 Section 3.2) corrupts exactly one element of the triple (source, relation, or target). This means every negative shares 2 of 3 elements with the positive. The model never sees "fully random" negatives where both actors and the relation type are different.

More critically, the corruption is chosen uniformly: `random.choice(["source", "target", "relation"])`. This means:
- 1/3 of negatives corrupt the source actor
- 1/3 corrupt the target actor
- 1/3 corrupt the relation type

But corrupting the relation type produces a much "easier" negative for most cases — if the positive is (USA, FIGHT, Iraq), then (USA, CONSULT, Iraq) is a very different event that the model can distinguish from the Goldstein score alone. Corrupted-actor negatives are harder because they test whether the model understands *who* is likely to interact.

**Risk:** The model may learn to discriminate event types easily (via relation embedding geometry) while undertrained on actor discrimination. The 1/3 relation-corruption rate may be too high.

**Recommendation:** Weight corruption types proportional to difficulty. Consider 45/45/10 (source/target/relation) instead of 33/33/33. Or follow the TransE convention: corrupt only actors, never relations, since relation discrimination is handled by the per-type hazard heads.

---

### GAP 15: Potential for Overfitting: Per-Event-Type Hazard Heads

**Tensors involved:** 18 separate `nn.Linear(2d, K=17)` hazard heads

**Issue:** The survival head has a shared trunk (6d+4 → 4d → 2d, ~2.1M params) followed by 18 separate per-type hazard heads (each 2d → 17, total ~156K params). The shared trunk provides cross-type feature sharing. But the per-type heads are independently parameterized.

For rare event types (FIGHT, SEIZE, SANCTION), the number of positive training examples may be small relative to the per-type head capacity. At d=256, each hazard head has 512*17 + 17 = 8,721 parameters. If a rare event type has <1000 positive examples in the training set, the per-type head is at risk of overfitting.

The curriculum learning (C5 Section 5.5) and event-type weighting (C3 Section 7.2) mitigate this, but they increase the effective gradient magnitude for rare types, which can *accelerate* overfitting rather than prevent it.

**Risk:** Moderate for the rarest event types (SEIZE, potentially FIGHT in certain dyad categories).

**Recommendation:**
1. Add dropout (0.2–0.3) specifically to the per-type hazard heads
2. Consider factored hazard heads: instead of 18 independent linear layers, use a shared base + per-type residual: `logits = shared_hazard(trunk) + type_specific_residual[r](trunk)`, where the residual has fewer parameters
3. Monitor per-type overfitting via train/val loss divergence per event type

---

### GAP 16: No Explicit Handling of Media Blackout Periods

**Tensors involved:** `h_i(t)` decay toward `b_i(t)` when no updates occur

**Issue:** The temporal decay mechanism causes memories to decay toward the EMA baseline when no updates occur. This is correct for actors that are genuinely "quiet." But media blackouts (internet shutdowns, conflict zones, censorship) also produce gaps in the data stream — and for the opposite reason.

An actor involved in a major conflict that shuts down media coverage will appear to be "quiet" to the model. Its memory will decay toward the pre-conflict baseline, making it *less* likely to predict ongoing conflict — exactly backwards.

The data quality monitoring (C1 Section 4.1) alerts on volume drops, but there's no mechanism to pause decay or flag an actor as "data-unavailable vs. genuinely quiet."

**Risk:** The model will systematically underpredict conflict persistence for actors in media-blackout situations. This is a known problem in GDELT-based research.

**Recommendation:**
1. Track per-actor article volume. When an actor's daily article count drops below 20% of its 30-day moving average, flag as "potential blackout" and reduce decay rate (increase effective half-life).
2. Consider a "data confidence" feature: append the actor's recent article volume (z-scored relative to its own history) to the dyadic representation. This gives the prediction head information about data reliability.

---

### GAP 17: Phase 0 Baseline and Neural Model Target Mismatch

**Tensors involved:** Phase 0 outputs binary/Poisson predictions; Phase 3 outputs survival curves

**Issue:** Phase 0 (LightGBM) trains 18 separate binary classifiers per event type with `objective: "binary"`. The neural model (Phase 3) outputs survival curves. Comparison between them uses "BSS at 30-day horizon" which derives a binary prediction from the CDF.

But the Phase 0 model predicts "will event type r happen between actors i and j in the next month?" while the neural model predicts "time to first event type r from i to j." These are subtly different:
- Phase 0 uses a country-month panel — one prediction per dyad per month
- Phase 3 can make predictions at any reference date, producing a continuous CDF

The BSS comparison at 30 days requires aligning these: Phase 0 predictions are presumably made at the start of each month, while Phase 3 can be evaluated at any point. If Phase 3 is evaluated at the same monthly cadence, fine. But if evaluated daily, Phase 3 has an advantage (more recent information).

**Risk:** Unfair comparison that overstates the neural model's improvement over the baseline.

**Recommendation:** Evaluate Phase 3 at the exact same monthly reference dates as Phase 0 for the head-to-head BSS comparison. Phase 3 can additionally be evaluated at daily cadence as a separate metric.

---

### GAP 18: Relation Embeddings Shared Between Event Encoding and Prediction Head

**Tensors involved:** `relation_embeddings [18, d]` used in Layer 3 event encoding AND Layer 5 dyadic representation

**Issue:** The same relation embedding `e_r` is used in two very different contexts:
1. **Layer 3 (event encoding):** `e_r` is concatenated with event metadata and fed to a GRU. Here it serves as "what kind of event happened."
2. **Layer 5 (dyadic representation):** `e_r` is concatenated with actor states. Here it serves as "what kind of event are we predicting."

These are semantically different roles. In Layer 3, the embedding interacts with the GRU to modify actor memory — it needs to encode "how this event type should change the actor's state." In Layer 5, it needs to encode "what kind of interaction we're asking about."

**Risk:** The shared embedding must serve both purposes, potentially compromising one. For example, the GRU might push FIGHT's embedding toward a direction that efficiently updates memories for conflict events, while the survival head needs FIGHT's embedding to be in a direction that correctly scales hazard for actor pairs.

**Recommendation:** Use separate embedding tables for Layer 3 and Layer 5. The event stream gets `relation_embeddings_event [18, d]` and the prediction head gets `relation_embeddings_pred [18, d]`. They can be initialized identically (from Goldstein scores) but allowed to diverge during training. Cost: +5K parameters (negligible).

---

### GAP 19: Underfitting Risk from Excessive Regularization on Rare Events

**Tensors involved:** All tensors in the loss computation

**Issue:** The design applies multiple overlapping regularization mechanisms to rare event types:
1. Curriculum learning (C5 Section 5.5): ramp up rare event weights over 20% of training
2. Inverse-frequency weighting (C3 Section 7.2): weight_r = 1/freq, capped at 100x
3. Temporal weighting (C3 Section 7.1): exponential decay with 270-day half-life
4. Negative confidence weighting (C3 Section 3.5): weights negatives by media coverage
5. Focal loss down-weighting of easy negatives (CLAUDE.md mentions focal loss as primary)
6. Memory regularization L_mem and gate sparsity L_gate
7. Dropout (0.1) in the survival head

For a rare event type (e.g., SEIZE between two sparsely-covered actors in a historical period), the effective training signal is: `temporal_weight (0.1) * event_type_weight (100) * curriculum_factor (0.5 mid-training) * negative_confidence (0.6) * focal_weight (0.1 if easy negative)`. These can interact unpredictably — the event_type_weight upweights while focal loss downweights, potentially canceling out.

**Risk:** The interaction of 5+ weighting schemes creates a complex, hard-to-debug effective learning rate landscape. Rare events may oscillate between being over- and under-emphasized depending on which weighting factors dominate.

**Recommendation:**
1. Log the effective per-example loss weight during training (product of all weighting factors). Verify the distribution across event types.
2. Consider simplifying: use focal loss OR inverse-frequency weighting, not both simultaneously. Focal loss already upweights hard examples, which rare events tend to be.

---

### GAP 20: No Explicit Mechanism for Dyadic Relationship History Beyond Actor Memories

**Tensors involved:** `h_i [d]` and `h_j [d]` — individual actor states, not dyad-specific

**Issue:** The model represents the state of the world through individual actor memories `h_i(t)`. The relationship between actors i and j is computed on-the-fly from their individual states: `d_ij = f(h_i, h_j)`. There is no persistent dyad-specific state.

This means: if USA and Russia have a complex bilateral history (decades of events), that history must be encoded *entirely* within `h_USA` and `h_Russia` individually. But each actor has hundreds of bilateral relationships. A 256-dimensional vector must encode USA's relationships with Russia, China, UK, NATO, Iran, etc. simultaneously.

The Hawkes excitation component partially addresses this by taking the dyad's event history as input. But that history is a list of (time, type) tuples — a sparse summary of the raw event record, not a learned representation.

**Risk:** The model may struggle with dyad-specific patterns that don't generalize from the individual actor states. For example, USA-Cuba and USA-Mexico are radically different bilateral relationships, but the model can only distinguish them via what's encoded in `h_Cuba` vs `h_Mexico`.

This is a fundamental architectural choice, not necessarily a bug. The design explicitly avoids per-dyad state to prevent O(N^2) memory. But it's worth noting as a potential underfitting source for dyad-specific dynamics.

**Recommendation:** Consider adding a lightweight dyadic state for the most active dyads. For example, maintain a `d_ij_context [d_small]` vector (e.g., d_small=32) for the top-1000 most active dyads, updated by a small GRU when events occur between them. This is ~32K additional state (1000 * 32), negligible. The dyadic context would be concatenated to the dyadic representation in Layer 5.

---

## Part 3: Summary of Critical vs. Important vs. Nice-to-Have

### Critical (will cause training failures or major performance issues)

| # | Gap | Risk |
|---|-----|------|
| 8 | TBPTT window size vs daily granularity | Layer 4 receives almost no gradient signal; self-attention will not learn |
| 4 | No temporal decay before GRU update | Memory staleness in event stream |
| 7 | Missing event history data structure for Hawkes | Implementation blocker |
| 10 | State-transition tracking undefined | Cannot train START/END predictions without ground-truth state |
| 11 | Architecture doc vs Component 4 prediction head inconsistency | Implementation confusion |

### Important (will degrade performance or cause subtle issues)

| # | Gap | Risk |
|---|-----|------|
| 2 | Scalar gate semantics conflict | Sparsemax on scalar is meaningless |
| 3 | GRU full-replacement without gating | Noisy event stream updates |
| 6 | EMA baseline gradient flow unspecified | Potential gradient instability |
| 9 | Surprise feature cold start | Survival head may ignore surprise features |
| 13 | 512-token truncation losing article content | Systematic information loss |
| 15 | Per-type hazard heads overfitting on rare events | Poor generalization for rare types |
| 16 | Media blackout → false decay | Underprediction during crises |
| 18 | Shared relation embeddings serving two roles | Compromised representations |
| 19 | Overlapping regularization schemes | Unpredictable effective weights |
| 20 | No dyad-specific persistent state | Underfitting on bilateral dynamics |

### Nice-to-Have (minor refinements)

| # | Gap | Risk |
|---|-----|------|
| 1 | Cross-attention dimension mismatch | Implementer confusion |
| 5 | Asymmetric representation for symmetric events | Wasted capacity |
| 12 | No warmup mask for new actors in self-attention | Transient disruption |
| 14 | Uniform corruption type distribution | Suboptimal negative difficulty |
| 17 | Phase 0 vs Phase 3 evaluation alignment | Unfair comparison |
