# Model Storage

Handles persistence, checkpointing, and versioning of actor memory vectors, model weights, and embedding snapshots. Actor memories are the central mutable state of the system — they must be reliably stored, versioned, and recoverable.

## Scope

This component covers the storage infrastructure for actor memory vectors, model checkpoints, and training artifacts. It does **not** cover the training logic itself (see `model_training`) or actor initialization (see `actor_management`).

## Actor Memory Store

### Structure

Each actor maintains a persistent memory vector:

```
h_i(t) ∈ R^d,   d = 256 or 512
```

Stored in a key-value database indexed by actor ID. These vectors persist across documents and update in-place as new information arrives. They are the continuously-updated world-state representations — not ephemeral hidden states that reset between batches.

### Memory Footprint

```
Actor memories: 5,000 actors × 512 floats × 4 bytes = ~10 MB
```

The memory vectors themselves are trivially small. The storage challenge is versioning and temporal snapshots, not raw size.

### Key-Value Store Requirements

- **Fast read/write:** Memory vectors are read and written on every update (per article, per event)
- **Atomic updates:** Multiple actors may update simultaneously from the same article
- **Temporal versioning:** Must be able to retrieve the state of all actor vectors at any past time point (for training rollouts, evaluation, and scenario simulation)
- **Snapshot capability:** Periodic full snapshots for checkpointing and recovery

### Candidate Storage Backends

| Backend | Pros | Cons |
|---------|------|------|
| In-memory dict + periodic pickle | Simplest, fast | No versioning, no crash recovery |
| SQLite with blob columns | Single-file, versioning via timestamps | Slower for batch reads |
| Redis | Fast K-V, TTL support | External dependency, no built-in versioning |
| HDF5 / Zarr | Efficient array storage, compression | Not a database — append-only snapshots |
| LevelDB / RocksDB | Fast embedded K-V, ordered keys | More complex setup |

**Recommendation for Phase 1–2:** In-memory dict with periodic HDF5 snapshots. Migrate to a proper K-V store (RocksDB or similar) only when scale demands it.

## Model Checkpointing

### What to Checkpoint

1. **Actor memory vectors** — The full state of all `h_i(t)` at checkpoint time
2. **Model weights** — All neural network parameters (ConfliBERT, GRUs, graph attention, prediction heads, gating networks)
3. **Optimizer state** — For training resumption
4. **Training metadata** — Step count, epoch, curriculum stage, learning rate schedule position
5. **Actor sketch vectors** — The 64-dim TF-IDF projections used for text pre-filtering

### Checkpoint Frequency

- **Actor memory snapshots:** Daily (these change continuously)
- **Full model checkpoints:** Weekly or per-training-run
- **Best-model checkpoint:** Saved whenever validation Brier score improves

### Checkpoint Format

Standard PyTorch checkpoint format:

```python
checkpoint = {
    'step': global_step,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'actor_memories': actor_memory_store.snapshot(),
    'actor_sketches': actor_sketch_matrix,
    'curriculum_stage': curriculum_stage,
    'best_brier_score': best_brier,
    'timestamp': datetime.utcnow().isoformat(),
}
torch.save(checkpoint, checkpoint_path)
```

## Temporal Snapshots

The model needs to reconstruct actor states at arbitrary past time points for:

- **Training rollouts:** Replay historical event sequences starting from past states
- **Evaluation:** Assess predictions made from past states against known outcomes
- **Scenario simulation:** Branch from a historical state and propagate hypothetical events

### Snapshot Strategy

- Store full snapshots at regular intervals (weekly or monthly)
- Between snapshots, store the delta (events processed, updates applied)
- Reconstruct intermediate states by replaying deltas from the nearest snapshot

## Embedding Space Versioning

As the model trains across phases, the meaning of embedding dimensions shifts. Track:

- **Phase transitions:** Clear version boundary when moving from Phase 1 → 2 → 3
- **Training run ID:** Each training run produces a distinct embedding space
- **Comparability metadata:** Store PCA/UMAP projections at each checkpoint for visual comparison across versions

## Storage Budget

```
Daily actor memory snapshots: 5K actors × 512 × 4 bytes × 365 days  ≈ 3.7 GB/year
Weekly model checkpoints: ~500MB each × 52 weeks                    ≈ 26 GB/year
GDELT history + structural data + embeddings                        ≈ 500 GB
Total S3 estimate:                                                  ~$12/month
```

## Key Dependencies

- `torch` — Checkpoint save/load
- `h5py` or `zarr` — Efficient array snapshot storage
- `pandas` — Metadata tracking

## Architecture Reference

Corresponds to **Layer 1: Actor Memory Store** (Section 6), **Computational Budget — Storage** (Section 17), and the persistence aspects of the overall system in the architecture design document.
