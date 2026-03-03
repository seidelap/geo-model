# Graph Propagation

Handles periodic information propagation across the full international system graph using multi-relational temporal graph attention. This captures second-order effects that direct observation misses — if Germany and France have a major summit, the France-Algeria relationship is also affected through indirect linkages.

## Scope

This component covers graph construction, multi-relational graph attention, multi-layer propagation, and the temporal graph network infrastructure. It operates on actor memory vectors produced by the text and event streams, and feeds updated vectors into the prediction heads.

## Why Graph Propagation?

Direct observation (text + events) only captures what is explicitly reported about each actor. But the international system is deeply interconnected:

- A US-China trade deal affects EU trade policy through second-order effects
- NATO expansion shifts security calculations for non-member states
- Regional power shifts propagate through alliance networks

Without graph propagation, the model can only update actors based on events they're directly involved in. With it, information flows through the network structure, enabling the model to anticipate indirect effects.

## Graph Construction

```
Nodes:      All active actors with current memory vectors {h_i}
Edge types: One per PLOVER event category (18 types)
Edge weights: Function of recency × frequency of interaction

    w_ij(r) = Σ_{t_k ∈ recent} exp(-β(t - t_k))   for events of type r
```

Edge weights decay exponentially with time since last interaction — recently active relationships receive more weight.

## Multi-Relational Graph Attention

```python
for r in range(18):  # one per PLOVER event type
    # Neighbors under relation r
    N_r_i = [j for j in actors if w_ij(r) > threshold]

    # Attention coefficients with temporal decay
    for j in N_r_i:
        attn_input = concat(W_r @ h_i, W_r @ h_j, time2vec(t - t_last_ij))
        alpha_ijr = LeakyReLU(a_r.T @ attn_input)

    alpha_i = softmax([alpha_ijr for j in N_r_i])

    # Aggregated neighborhood message under relation r
    msg_r = sum(alpha_ijr * W_r @ h_j for j in N_r_i)

# Update: residual connection + sum across all relation types
h_i_new = h_i + LayerNorm(sum(msg_r for r in range(18)))
```

### Key Design Decisions

- **Relation-specific weight matrices `W_r`:** One per PLOVER event type. Military alliances propagate security-dimension information differently than trade agreements propagate economic-dimension information.
- **Temporal decay in attention:** Recently active relationships receive more weight. A dormant alliance matters less than an active one.
- **Residual connection:** The identity path ensures that graph propagation refines actor representations without overwriting direct observation signal.

## Multi-Layer Propagation

Stack 2–3 layers of graph attention for multi-hop information flow:

```python
for layer in range(num_layers):
    h_new = graph_attention_layer(h_current, edge_index, edge_types, edge_weights)
    h_current = h_new
```

| Layers | Information Reach | Trade-off |
|--------|------------------|-----------|
| 1 | Direct partners only | Minimal propagation |
| 2 | Partners' partners | Good balance (recommended) |
| 3 | 3-hop neighborhoods | Risk of over-smoothing |

**In practice, 2 layers is usually sufficient** for capturing relevant international system structure without over-smoothing actor representations (making all actors look similar).

## Propagation Schedule

Graph propagation runs periodically — **daily or weekly** — not on every event. This is a design trade-off:

- **Too frequent:** Computationally expensive, may amplify noise
- **Too infrequent:** Misses time-sensitive indirect effects
- **Recommended:** Daily propagation, with the option for event-triggered propagation on major events

## Computational Cost

```
Graph propagation (daily batch, T4, ~10 min):  ~$0.07/day → ~$2/month
Graph attention (200 nodes, 18 relations):      ~500 MB VRAM
```

The graph is relatively small (~5,000 actors with most having sparse connections), so propagation is cheap compared to text encoding.

## Implementation Options

### EvolveGCN (Phase 2)
- From PyTorch Geometric Temporal
- Adapts GCN parameters over time using an RNN
- Good starting point for Phase 2

### Temporal Graph Networks / TGN (Phase 3)
- Per-node memory with message-passing updates
- Better suited to the full architecture's continuous memory paradigm
- Reference: `github.com/twitter-research/tgn`

### RE-NET (Baseline Comparison)
- Autoregressive temporal knowledge graph event prediction
- Reference: `github.com/INK-USC/RE-Net`

## Key Dependencies

- `pytorch_geometric_temporal` — EvolveGCN, A3TGCN implementations (`github.com/benedekrozemberczki/pytorch_geometric_temporal`)
- `tgn` — Temporal Graph Networks (`github.com/twitter-research/tgn`)
- `torch_geometric` — Core graph neural network primitives
- `torchdyn` — Neural ODE for continuous-time dynamics (`github.com/DiffEqML/torchdyn`)

## Build Phase Mapping

| Phase | Implementation |
|-------|---------------|
| Phase 1 (Months 1–2) | No graph component — tabular features only |
| Phase 2 (Months 3–5) | EvolveGCN via PyTorch Geometric Temporal. Countries as nodes, CAMEO/PLOVER events as typed timestamped edges |
| Phase 3 (Months 6–12) | Full TGN-style multi-relational temporal graph attention. Neural ODE continuous dynamics. Multi-layer propagation |

## Architecture Reference

Corresponds to **Layer 4: Temporal Graph Propagation** (Section 9) in the architecture design document.
