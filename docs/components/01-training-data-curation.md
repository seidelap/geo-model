# Component 1: Training Data Curation

## Purpose

Ingest raw data from all sources, apply filtering and normalization, and produce clean datasets in consistent formats for downstream components. This component produces three categories of curated data: text corpora, structured events, and structural/static actor features.

---

## 1. Text Corpus Pipeline

### 1.1 Source: Common Crawl News

**What it is:** Daily WARC archives of news articles from 1,000+ sources, available since 2016.

**Access:** `s3://commoncrawl/crawl-data/CC-NEWS/` (public S3 bucket, free).

**Raw volume:** 200,000–500,000 articles/day across all languages.

**Ingestion process:**

```
S3 WARC files (daily)
  → Download daily archive
  → Parse WARC records → extract (url, title, body_text, publish_date, source_domain, language)
  → Store raw articles in staging table
```

**Storage schema for raw articles:**

| Field | Type | Description |
|-------|------|-------------|
| `article_id` | `string` | SHA-256 of (url + publish_date) |
| `url` | `string` | Source URL |
| `title` | `string` | Article title |
| `body_text` | `string` | Full article text |
| `publish_date` | `date` | Publication date |
| `source_domain` | `string` | e.g. `reuters.com` |
| `language` | `string` | ISO 639-1 code |
| `ingestion_timestamp` | `datetime` | When we downloaded it |

### 1.2 Text Filtering Pipeline

Four sequential filters reduce ~500K raw articles/day to ~40K for encoding:

**Stage 1 — Language filter:**
- Keep: English plus up to 5 additional major languages (Arabic, Chinese, French, Russian, Spanish) when cross-lingual encoding is enabled.
- Implementation: `fasttext` language ID model (`lid.176.bin`). Single-pass, CPU, microseconds per article.
- Expected yield: ~200,000 articles.

**Stage 2 — Geopolitical relevance filter:**
- Method: TF-IDF cosine similarity of article text against a curated geopolitical keyword dictionary.
- Keyword dictionary: ~2,000 terms covering country names, leader names, organization names, military terms, diplomatic terms, economic policy terms. Maintained as a versioned text file in this repository.
- Threshold: cosine similarity ≥ 0.15 (tuned on a manually labeled sample of 1,000 articles).
- Implementation: `sklearn.feature_extraction.text.TfidfVectorizer` with precomputed dictionary vectors. Runs at millions of articles/minute on CPU.
- Expected yield: ~80,000 articles.

**Stage 3 — Near-duplicate removal:**
- Method: MinHash LSH with Jaccard similarity threshold ≥ 0.8.
- Implementation: `datasketch.MinHashLSH` with 128 permutations, computed on 5-gram shingles of article body text.
- Within each duplicate cluster, keep the earliest published article (by `publish_date`), discard rest.
- Expected yield: ~40,000 unique articles/day.

**Stage 4 — Quality filter (optional):**
- Remove articles shorter than 100 words (wire service stubs, paywalled excerpts).
- Remove articles longer than 5,000 words (multi-topic longform that dilutes actor signal).
- Remove articles where ≥50% of body is boilerplate (detected by repeated n-gram ratio).

### 1.3 Text Output Schema

Articles that pass all filters are stored as the curated text corpus:

| Field | Type | Description |
|-------|------|-------------|
| `article_id` | `string` | Same as raw |
| `title` | `string` | Article title |
| `body_text` | `string` | Full article text |
| `publish_date` | `date` | Publication date |
| `source_domain` | `string` | Source domain |
| `language` | `string` | ISO 639-1 |
| `relevance_score` | `float` | TF-IDF cosine score from Stage 2 |
| `minhash_signature` | `bytes` | For future dedup lookups |

**Storage format:** Parquet files partitioned by `publish_date`, one file per day. Typical size: ~50–100 MB/day compressed.

### 1.4 Source: Parliamentary Speech Corpora (Supplementary)

Lower priority than news text but useful for Phase 2+ self-supervised pretraining.

| Corpus | Coverage | Access |
|--------|----------|--------|
| ParlSpeech V2 | 6.3M speeches, 9 democracies, 1987–2018 | Harvard Dataverse |
| ParlaMint (CLARIN) | EU parliaments, TEI XML | CLARIN infrastructure |
| UK Hansard | 1803–present | TheyWorkForYou API |
| US Congressional Record | All sessions | `govinfo.gov` bulk data |

These are static corpora (not daily feeds). Download once, filter for foreign-policy-relevant speeches (committee assignments, keyword filtering), store in same Parquet schema as news articles with `source_domain = "parlspeech"` etc.

### 1.5 Source: UN General Debate Corpus

7,300+ full-text country speeches from 1970–2016 (Baturo, Dasandi & Mikhaylov, 2017). Download once. Each speech is attributed to a specific country and year, making it directly usable for actor-attributed text pretraining.

---

## 2. Structured Event Pipeline

### 2.1 Source: POLECAT (Primary)

**What it is:** Political events coded using the PLOVER ontology via the NGEC transformer-based pipeline. Higher precision than GDELT.

**Access:** Free weekly downloads at Harvard Dataverse (`doi:10.7910/DVN/AJGVIT`).

**Coverage:** 2018–present, weekly updates.

**Raw fields we use:**

| Field | Type | Description |
|-------|------|-------------|
| `source_actor` | `string` | PLOVER actor code |
| `target_actor` | `string` | PLOVER actor code |
| `event_type` | `string` | One of 18 PLOVER categories |
| `event_mode` | `string` | `verbal` / `hypothetical` / `actual` |
| `goldstein_score` | `float` | –10 to +10 cooperation-conflict scale |
| `event_date` | `date` | When the event occurred |
| `magnitude_dead` | `int` | Casualties (if applicable) |
| `magnitude_injured` | `int` | Injuries (if applicable) |
| `magnitude_size` | `int` | Group size (protests, mobilizations) |
| `source_url` | `string` | Article the event was coded from |
| `confidence` | `float` | Coder confidence score |

### 2.2 Source: GDELT (Secondary / Historical)

**What it is:** Largest open event dataset. ~100,000+ events/day from 300+ CAMEO categories across 100+ languages.

**Access:** Google BigQuery (1 TB/month free tier) or bulk CSV from `data.gdeltproject.org`.

**Python library:** `gdeltPyR` (`github.com/linwoodc3/gdeltPyR`).

**Coverage:** 1979–present, updated every 15 minutes.

**Critical normalization requirements:**
- GDELT event counts must **always** be normalized against total article volume in the same time window. Raw counts reflect media attention cycles, not real-world activity.
- GDELT produces massive event duplication (one real-world event → many records from many articles). Deduplicate by grouping events with the same (source_actor, target_actor, event_type, date) and keeping the record with the highest confidence or most detailed metadata.
- CAMEO codes must be mapped to PLOVER categories for consistency with POLECAT. The mapping table is maintained at `github.com/openeventdata/PLOVER`.

**CAMEO → PLOVER mapping approach:**
- CAMEO has ~250 codes organized in a 2-level hierarchy (20 top-level × ~12 subcategories).
- PLOVER has 18 categories.
- The mapping is many-to-one. A static lookup table (CSV) maps each CAMEO code to exactly one PLOVER category.
- Ambiguous CAMEO codes (e.g., `040: Consult` vs. `050: Diplomatic cooperation`) map to the closest PLOVER category. Document edge cases in the mapping file.

### 2.3 Source: ICEWS Archive (Historical Backfill)

**What it is:** 270 million events covering 1995–April 2023. The standard benchmark for temporal knowledge graph research.

**Access:** Harvard Dataverse (`doi:10.7910/DVN/28075`).

**Use:** Historical backfill for training. ICEWS events use CAMEO coding and require the same CAMEO → PLOVER mapping as GDELT.

### 2.4 Event Normalization Pipeline

All event sources are normalized to a common schema:

```
Raw POLECAT events  ──────────────────────┐
                                          │
Raw GDELT events ─→ CAMEO→PLOVER map ─→  ├─→ Normalize actor codes ─→ Deduplicate ─→ Curated Events
                                          │
Raw ICEWS events ─→ CAMEO→PLOVER map ─→  ┘
```

**Normalized event schema:**

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `string` | SHA-256 of (source, target, type, date, source_url) |
| `source_actor_id` | `string` | Canonical actor ID (from Actor Registry) |
| `target_actor_id` | `string` | Canonical actor ID |
| `event_type` | `enum` | One of 18 PLOVER categories |
| `event_mode` | `enum` | `verbal` / `hypothetical` / `actual` |
| `goldstein_score` | `float` | Normalized to [–1, +1] from raw [–10, +10] |
| `event_date` | `date` | Date of occurrence |
| `magnitude_dead` | `int` | 0 if not applicable |
| `magnitude_injured` | `int` | 0 if not applicable |
| `magnitude_size` | `int` | 0 if not applicable |
| `data_source` | `enum` | `polecat` / `gdelt` / `icews` |
| `confidence` | `float` | Source-specific confidence score, normalized to [0, 1] |
| `source_url` | `string` | Source article URL |

**Actor code normalization:** GDELT, POLECAT, and ICEWS each use slightly different actor coding schemes. Normalize to canonical actor IDs defined by the Actor Registry (Component 2). This requires a mapping table from each source's actor codes to canonical IDs.

- POLECAT uses ISO 3166-1 alpha-3 for countries and Wikipedia-linked entity IDs for others.
- GDELT uses CAMEO actor codes (3-letter country codes + role suffixes like `GOV`, `MIL`, `OPP`).
- ICEWS uses a custom coding scheme similar to CAMEO.

The mapping tables are maintained as versioned CSV files in this repository. Unmapped actors are logged and periodically reviewed for inclusion in the Actor Registry.

**Cross-source deduplication:** When the same real-world event appears in multiple sources (e.g., both GDELT and POLECAT code the same news article), prefer the POLECAT record (higher precision) and mark the GDELT record as a duplicate. Match on: same source/target actors, same event type, same date, overlapping source URLs.

**Storage format:** Parquet files partitioned by `event_date`, one file per month. Typical size: ~200–500 MB/month compressed for all sources combined.

### 2.5 ACLED Exclusion

ACLED's 2025 terms of service explicitly prohibit using its data to train, test, develop, or improve ML models or AI systems. **Do not use ACLED data in any pipeline.**

---

## 3. Structural / Static Data Pipeline

These are slow-moving datasets that characterize actors structurally. They are updated infrequently (annually or less) and used primarily for actor initialization (Component 2) and as features in the tabular baseline (Component 5, Phase 0).

### 3.1 Source: Correlates of War (COW)

| Dataset | Key Fields | Coverage | Access |
|---------|------------|----------|--------|
| CINC v6.0 | Military expenditure, military personnel, energy consumption, iron/steel production, urban pop, total pop | 1816–2016 | `correlatesofwar.org` CSV |
| MID v5.0 | Militarized interstate disputes | 1816–2014 | `correlatesofwar.org` CSV |
| Alliance v4.1 | Formal defense/offense/neutrality pacts | 1816–2012 | `correlatesofwar.org` CSV |

**Limitation:** Most datasets end 2010–2016. For recent years, supplement with:
- SIPRI for military expenditure (1988–2024)
- World Bank for economic/demographic indicators
- Manual alliance coding from treaty databases

**Access helper:** R package `peacesciencer` simplifies COW data retrieval and merging.

### 3.2 Source: V-Dem (Varieties of Democracy)

Democracy and governance indices for 202 countries, 1789–2023. ~400 indicators covering electoral, liberal, participatory, deliberative, and egalitarian dimensions.

**Key fields for actor initialization:**
- `v2x_polyarchy` (electoral democracy index)
- `v2x_libdem` (liberal democracy index)
- `v2x_partipdem` (participatory democracy index)
- `v2x_rule` (rule of law index)
- `v2x_corr` (corruption index)

**Access:** `v-dem.net/data/` CSV download or API.

### 3.3 Source: UN General Assembly Voting (Voeten)

**What it is:** Ideal point estimates derived from roll-call votes, decomposed across six issue categories.

**Dataset:** Voeten dataset, sessions 1–78 (1946–2023), ~738,764 individual vote records.

**Key fields:**
- Country ideal points per session (continuous, ~2D)
- Vote-level records: (country, resolution, vote [yes/no/abstain], session, date)
- Issue category: Palestinian conflict, nuclear weapons, colonialism, human rights, economic development, arms control

**Access:** `unvotes.unige.ch`; R package `unvotes`.

### 3.4 Source: World Bank Open Data

1,400+ development indicators via free unauthenticated API (`api.worldbank.org/v2/`).

**Key indicators for actor features:**
- `NY.GDP.MKTP.CD` — GDP (current US$)
- `NY.GDP.PCAP.CD` — GDP per capita
- `NE.TRD.GNFS.ZS` — Trade as % of GDP
- `BX.KLT.DINV.CD.WD` — FDI net inflows
- `SP.POP.TOTL` — Total population
- `SP.URB.TOTL.IN.ZS` — Urban population %

**Ingestion:** Annual snapshots via API. Interpolate missing values linearly between available years.

### 3.5 Source: SIPRI

| Dataset | Coverage | Access |
|---------|----------|--------|
| Arms Transfer Database (TIV) | Major conventional weapons transfers, 1950–2024 | `sipri.org` download |
| Military Expenditure Database | 171 countries, 1988–present | `sipri.org` download |

### 3.6 Source: UN Comtrade / CEPII BACI

Bilateral trade flows by commodity.

- **Comtrade:** 500 free API calls/day. Raw but comprehensive.
- **CEPII BACI:** Reconciled version at HS 6-digit level. Free for academic use, 1995–present.

**Use:** Bilateral trade volume and commodity composition as dyadic features (actor-pair level, not just actor level).

### 3.7 Structural Data Output Schema

All structural data is normalized to a common per-actor-per-year panel:

| Field | Type | Description |
|-------|------|-------------|
| `actor_id` | `string` | Canonical actor ID |
| `year` | `int` | Year of observation |
| `cinc_score` | `float` | Composite military capability [0, 1] |
| `milper` | `float` | Military personnel (log-normalized) |
| `milex` | `float` | Military expenditure (log-normalized, constant USD) |
| `energy` | `float` | Energy consumption (log-normalized) |
| `gdp` | `float` | GDP (log-normalized, constant USD) |
| `gdp_per_capita` | `float` | GDP per capita (log-normalized) |
| `population` | `float` | Total population (log-normalized) |
| `trade_pct_gdp` | `float` | Trade openness [0, 1+] |
| `polity_score` | `float` | V-Dem electoral democracy index [0, 1] |
| `liberal_democracy` | `float` | V-Dem liberal democracy index [0, 1] |
| `voeten_ideal_1` | `float` | Voeten ideal point, dimension 1 |
| `voeten_ideal_2` | `float` | Voeten ideal point, dimension 2 |
| `arms_imports_tiv` | `float` | SIPRI TIV arms imports (log-normalized) |
| `arms_exports_tiv` | `float` | SIPRI TIV arms exports (log-normalized) |

**Dyadic structural features** (actor-pair level, for bilateral relationships):

| Field | Type | Description |
|-------|------|-------------|
| `actor_i_id` | `string` | Actor i canonical ID |
| `actor_j_id` | `string` | Actor j canonical ID |
| `year` | `int` | Year of observation |
| `bilateral_trade_volume` | `float` | Log-normalized bilateral trade |
| `trade_balance` | `float` | Normalized net trade (i exports to j minus j exports to i) |
| `alliance_type` | `enum` | `none` / `defense` / `offense` / `neutrality` / `entente` |
| `voeten_distance` | `float` | Euclidean distance between Voeten ideal points |
| `mid_history_count` | `int` | Count of MIDs between actors in last 10 years |
| `geographic_distance_km` | `float` | Capital-to-capital distance |
| `contiguous` | `bool` | Whether states share a land border |

**Storage format:** Parquet files, one per year, ~5–10 MB each. Dyadic features stored separately due to quadratic actor count.

---

## 4. Data Quality and Monitoring

### 4.1 Volume Monitoring

Track daily counts for each pipeline stage. Alert on:
- Text pipeline: <20,000 or >80,000 articles after filtering (indicates source disruption or filter drift)
- Event pipeline: <10,000 or >300,000 events/day from GDELT (indicates API issues or processing errors)
- Unmapped actors exceeding 5% of daily event volume (indicates Actor Registry needs updating)

### 4.2 Known Data Quality Issues

**GDELT-specific:**
- Weekend periodicity: event counts drop 20–40% on weekends due to media cycles, not real-world activity. The tabular baseline and any count-based features must account for day-of-week effects.
- English/Western media bias: events involving non-Western actors are systematically undercounted. This is an inherent limitation, partially mitigated by including POLECAT (which uses multilingual sources) and eventually adding cross-lingual encoding.
- Actor resolution errors: GDELT's pattern-matching actor coder produces frequent misattributions. Cross-reference with POLECAT's Wikipedia-linked entity IDs where possible.

**POLECAT-specific:**
- Weekly batch updates introduce 1–7 day latency. Not suitable for real-time inference without supplementing with GDELT for the most recent days.
- Coverage starts 2018, limiting historical training depth.

**Structural data gaps:**
- COW datasets largely end 2010–2016. Recent years require imputation or supplementation from other sources.
- Small states and non-state actors have sparse or missing structural data. Handle with default values and explicit missingness indicators.

### 4.3 Versioning

All curated datasets are versioned by processing date. A data manifest file records:
- Source URLs and access dates
- Filter thresholds applied
- Record counts at each pipeline stage
- Schema version
- Processing code git commit hash

This enables reproducible training runs tied to specific data snapshots.
