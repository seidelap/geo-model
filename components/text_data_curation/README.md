# Text Data Curation

Handles acquisition, filtering, and encoding of unstructured text for actor memory updates. This is the primary information channel — raw news articles contain richer signal (hedging, tone, causal framing, forward-looking language) than structured event extractions.

## Scope

This component covers everything from raw text acquisition through encoded document representations ready for actor memory updates. It does **not** cover the memory update itself (see `model_training`) or real-time ingestion pipelines (separate application repo).

## Data Sources

### Common Crawl News (Primary)
- Daily WARC files since 2016, 1,000+ news sources
- Access: `s3://commoncrawl/crawl-data/CC-NEWS/` (free S3 public dataset)
- Volume: ~200,000–500,000 articles/day across all languages
- This is the primary free source for large-scale news text

### Parliamentary Speech Corpora
- **ParlSpeech V2:** 6.3 million speeches from 9 democracies, 1987–2018 (Harvard Dataverse)
- **ParlaMint (CLARIN):** Additional EU parliaments in TEI XML
- **UK Hansard:** Digital since 1803, via TheyWorkForYou API
- **US Congressional Record:** Via `govinfo.gov` bulk data

### Expert Survey Data
- **Chapel Hill Expert Survey (CHES):** 279 parties across 31 countries, expert-estimated ideological positions (`chesdata.eu`)
- **Manifesto Project:** 5,285 election manifestos across 67 countries since 1945, with `manifestoberta` multilingual LLM (`manifestoproject.wzb.eu`)

### UN General Debate Corpus
- 7,300+ full-text country speeches from 1970–2016 (Baturo, Dasandi & Mikhaylov, 2017)

## Pre-filtering Pipeline

The filtering pipeline reduces ~500K daily articles down to ~40K before any GPU compute:

```
Raw stream:               ~500,000 articles/day
→ Language filter:        ~200,000  (keep English + major languages)
→ Geopolitical relevance: ~80,000   (TF-IDF cosine vs. keyword dictionary)
→ Deduplication:          ~40,000   (near-duplicate removal via MinHash)
→ Full encoding:          ~40,000 articles/day
```

### Language Filtering
Keep English plus major geopolitically relevant languages. The architecture notes GDELT's English/Western media bias — multilingual coverage is a known gap and open research question.

### Relevance Filtering
Lightweight bag-of-words TF-IDF cosine similarity against a geopolitical keyword dictionary. Runs at millions of articles per minute on CPU. Reduces encoding workload by 5–10x before any GPU compute is spent.

### Deduplication
Near-duplicate removal via MinHash. One real-world event generates many records from multiple articles — this is a known issue in GDELT as well. Critical normalization step.

## Sketch-Based Actor Relevance Filter

For each article that passes pre-filtering, determine which actors to update without running the full encoder:

```python
# Compute lightweight sketch of article
article_sketch = tfidf_vectorizer.transform([article_text])  # sparse, cheap

# Compute relevance to each actor via dot product against actor sketch vectors
relevance_scores = article_sketch @ actor_sketch_matrix.T  # shape: [N_actors]

# Sparsemax: learned sparse selection of relevant actors
actor_weights = sparsemax(relevance_scores / temperature)

# Only proceed with nonzero-weight actors
active_actors = [i for i, w in enumerate(actor_weights) if w > 0]
# Typical result: 5–20 actors per article
```

Actor sketch vectors are low-dimensional (64-dim TF-IDF or small embedding) projections maintained alongside full memory vectors. Updated cheaply whenever the full memory update runs.

This achieves most of the benefit of soft attention (no hard entity resolution errors, captures indirect relevance through semantic similarity) while avoiding noise from updating every actor with every article.

## Full Document Encoding

### ConfliBERT Encoder
- Domain-adapted BERT for conflict/political text (`github.com/eventdata/ConfliBERT`)
- BERT-base architecture: 12 layers, 110M parameters
- Output: `[seq_len, d_model=768]` per article

### Mention Extraction
For each active actor, extract mention representation by pooling contextual embeddings at mention spans:

```python
T = ConfliBERT(article_text)  # [seq_len, 768]

# Hard entity resolution where confident
m_i = mean_pool(T[mention_spans_of_actor_i])

# Soft pooling fallback for indirect relevance
m_i = weighted_pool(T, attention_weights=actor_weights[i])
```

The two approaches can be combined: use hard entity resolution where confident, fall back to soft pooling for actors identified only by sketch similarity.

### Cross-Actor Interaction Representation
Articles about bilateral interactions aren't just updating two actors independently — the *relationship* between them is the signal:

```python
# Cross-attention: each actor attends to the other's mention tokens
query_i = W_Q @ m_i
key_j   = W_K @ T
val_j   = W_V @ T
c_ij = softmax(query_i @ key_j.T / sqrt(d_k)) @ val_j
```

This feeds into the memory update for both actors, ensuring what each actor "learns" from the document is conditioned on the interaction context.

## Model Size Tradeoffs

| Model | Params | Speed | Notes |
|-------|--------|-------|-------|
| ConfliBERT (BERT-base, 12L) | 110M | Baseline | Good domain adaptation |
| DistilConfliBERT (6L) | ~66M | ~1.7x faster | Minimal accuracy loss, via HuggingFace distillation |
| Custom 4-layer domain model | ~30M | ~2.5x faster | Train from scratch on geopolitical text |

**Recommendation:** Start with ConfliBERT to establish baselines. Distill once the architecture is stable.

## Computational Cost

Assuming 40,000 articles/day after pre-filtering:

```
40,000 articles × 400 tokens/article = 16,000,000 tokens/day
```

| Hardware | Throughput | Time for 16M tokens | Cost/day (spot) |
|----------|-----------|---------------------|-----------------|
| T4 (Colab/GCP) | ~8K–12K tok/sec | ~25–35 min | ~$0.15–0.25 |
| A10G (AWS g5) | ~25K–35K tok/sec | ~8–12 min | ~$0.20–0.30 |
| RTX 4090 (owned) | ~40K–55K tok/sec | ~5–7 min | electricity only |

## Key Dependencies

- `transformers` — ConfliBERT model loading and inference
- `mordecai3` — Neural geoparsing / toponym resolution (`github.com/ahalterman/mordecai3`)
- `sparsemax-pytorch` — Sparsemax / entmax for actor selection (`github.com/KrisKorrel/sparsemax-pytorch`)
- MinHash library (e.g., `datasketch`) for deduplication

## Self-Supervised Pretraining Objectives (Phase 2)

Before supervised event prediction, the text encoder is pretrained with:

1. **Masked entity prediction:** Mask actor name spans in documents, predict masked actor from surrounding context. Forces memory vectors to encode geopolitically meaningful representations.
2. **Temporal ordering:** Given two paragraphs from different time points, predict which came first. Teaches causal sequencing.
3. **Contrastive document similarity:** Documents describing the same event should have similar representations. Use GDELT event codes as weak supervision for pairing.

## Open Research Questions

- **Cross-lingual signal:** ConfliBERT is primarily English-trained. Events reported only in Arabic, Chinese, or Russian media carry different perspectives. Multilingual pretraining or cross-lingual alignment needed.
- **Strategic communication modeling:** The model takes text at face value. Deliberately misleading statements corrupt memory representations. Adversarial training or credibility modeling is an open extension.

## Architecture Reference

Corresponds to **Layer 2: Text Processing Stream** (Section 7) and **Text Corpora** (Section 3.3) in the architecture design document.
