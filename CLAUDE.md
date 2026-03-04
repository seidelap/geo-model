# geo-model

Geopolitical event prediction system. Encodes actors (nations, orgs, leaders) as
time-varying vectors and predicts event probabilities between actor pairs using
survival curves over 18 PLOVER event types.

- **Architecture spec:** `docs/architecture-design.md`
- **Component specs:** `docs/components/01-06` (dependency order in `docs/components/README.md`)
- **Status:** Pre-implementation (architecture designed, no code yet)

## Architecture at a Glance

Six-layer neural pipeline (details in architecture-design.md Sections 6-11):

| Layer | Name                    | Role                                                     |
|-------|-------------------------|----------------------------------------------------------|
| 1     | Actor Memory Store      | Persistent h_i(t) vectors per actor, EMA decay           |
| 2     | Text Processing Stream  | ConfliBERT encoding → cross-attention → gated memory update |
| 3     | Structured Event Stream | GRU update from coded GDELT/POLECAT events               |
| 4     | Actor Self-Attention    | Full multi-head attention across all actors               |
| 5     | Event Prediction Head   | Discrete hazard → survival curves + Hawkes intensity     |
| 6     | Calibration             | Per-bin temperature scaling                              |

**Prediction target:** S(τ | i, j, r, t₀) — survival curve over K=17 non-uniform time bins.

**Build phases:** 0 (LightGBM baseline) → 1 (structural pretrain) → 2 (self-supervised) → 3 (supervised fine-tune). Each phase must beat the previous.

**Implementation order:** C1 (data) → C2 (actors) → C3 (targets) → C4 (model) → C5 (training) → C6 (validation). Components 1-3 can partially parallelize.

## Tech Stack

- **Language:** Python 3.11+
- **ML framework:** PyTorch
- **Text encoder:** ConfliBERT (BERT-base, 110M params) via HuggingFace `transformers`
- **Baseline model:** LightGBM
- **Data format:** Parquet (tabular), NumPy .npy (dense embeddings), PyTorch .pt (checkpoints)
- **Survival analysis:** pycox (DeepHit, Cox-Time)
- **Event data:** GDELT, POLECAT (Harvard Dataverse), ICEWS
- **Geoparsing:** Mordecai 3
- **Other:** datasketch (MinHash LSH), sparsemax-pytorch (entmax), torchdyn (Neural ODE)
- **ACLED is prohibited** — license forbids ML training/testing (see architecture doc §3)

## Key Terminology

| Term              | Meaning                                                           |
|-------------------|-------------------------------------------------------------------|
| Actor             | Any entity: state, IGO, non-state group, company, leader          |
| Dyad              | Ordered pair (source actor, target actor)                         |
| PLOVER            | Event ontology: AID, AGREE, CONSULT, COOP, DEMAND, DISAPPROVE, ENGAGE, FIGHT, INVESTIGATE, MOBILIZE, PROTEST, REDUCE, REJECT, SANCTION, SEIZE, THREATEN, YIELD, OTHER |
| CAMEO             | Legacy event coding (GDELT); convert to PLOVER                   |
| Goldstein score   | Cooperation/conflict intensity, -10 to +10                       |
| Quad-class        | 2×2: verbal/material × cooperation/conflict                      |
| Survival curve    | S(τ) = P(no event in [t₀, t₀+τ]); primary prediction output     |
| Discrete hazard   | h[k] = P(event in bin k | survived to bin k); K=17 bins          |
| Hawkes process    | Self-exciting point process for high-frequency event intensity    |
| Actor memory      | h_i(t) ∈ ℝ^d — persistent embedding updated by Layers 2-4       |
| EMA baseline      | Slow-moving exponential average of actor memory; decay target     |
| Sketch vector     | Lightweight TF-IDF vector for fast article-actor relevance filter |
| BSS               | Brier Skill Score — improvement over climatological baseline      |
| ECE               | Expected Calibration Error                                       |
| CPC               | Contrastive Predictive Coding (Phase 2 pretraining objective)     |
| DeepHit           | Survival model loss: NLL + concordance ranking                   |
| Focal loss        | Down-weights easy negatives; primary training loss               |
| Feasibility filter| Only sample negatives that are structurally plausible             |

## Commands

<!-- UPDATE this section as implementation proceeds -->

```shell
# Tests — TODO
# pytest tests/ -x -q
# pytest tests/ -m "not slow"   # skip GPU/data-heavy tests

# Linting — TODO
# ruff check . && ruff format --check .

# Training — TODO
# python -m geo_model.train --phase 0   # LightGBM baseline
# python -m geo_model.train --phase 3   # supervised fine-tune

# Evaluation — TODO
# python -m geo_model.evaluate --split test
```

## Coding Standards

- Type hints on all function signatures (`from __future__ import annotations`)
- Docstrings: Google style. Include tensor shapes in bracket notation: `[batch, seq_len, d]`
- Dataclasses or Pydantic models for all domain objects — no raw dicts for configs or schemas
- Actor IDs: string format per Component 2 spec (e.g., `"state:USA"`, `"igo:UN"`)
- Timestamps: always UTC. Durations in days (float)
- Tabular data: Parquet only, never CSV
- Embedding dimension `d` and all loss weights: config parameters, never hardcoded
- Test files mirror source: `src/geo_model/foo/bar.py` → `tests/foo/test_bar.py`
- File naming: `lowercase_snake_case`

## Development Workflow

**Branch convention:** `claude/*` for all AI-assisted work.

### Two-Layer Agent Workflow
1. **Claude Code (primary):** All implementation, architecture, refactoring, tests.
   Read component specs before implementing. Commit to `claude/*` branches.
2. **Cursor (secondary):** Debugging when Claude Code hits a wall.
   Use for: stepping through runtime errors, inspecting tensor shapes, data exploration.
   Do NOT make architectural decisions in Cursor — reflect findings back to Claude Code.

### Every PR must
- Not regress Phase 0 BSS (once established)
- Include or update relevant tests
- Reference the relevant component spec (C1-C6)
- Update CLAUDE.md if conventions changed

## Meta: Keeping This File Current

This is a living document. Update it when:

1. **New command or entrypoint created** → add to Commands section
2. **Coding pattern established** → add to Coding Standards
3. **New domain term used repeatedly** → add to Terminology table
4. **Module directory created** → consider subdirectory CLAUDE.md (see below)
5. **Debugging discovery** → add non-obvious pattern to relevant section or subdirectory file

### Reflection (after completing any major task)
- Did I discover a convention that should be documented here?
- Is there a command I ran repeatedly that should be in Commands?
- Should a subdirectory CLAUDE.md be created for the module I just built?

### Subdirectory CLAUDE.md Files
Create a CLAUDE.md in any `geo_model/` subdirectory when the module has 5+ files
or module-specific conventions not captured here. Keep 20-40 lines each. Reference
the corresponding `docs/components/0X-*.md` spec rather than restating it.

### Root vs Subdirectory
- **Root:** project-wide conventions, architecture, commands, terminology
- **Subdirectory:** module-specific patterns, data format gotchas, internal APIs
