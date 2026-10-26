# Architecture Decision Records (ADR) — NewsSense Platform

This document tracks all architectural, algorithmic, and optimization decisions made for the **NewsSense** AI platform, including evaluation metrics, model selections, trade-offs, and planned improvements.

---

## ADR Index

- [ADR-001: News Category Classification Model](#adr-001-news-category-classification-model)
- [ADR-002: Abstractive Summarization Model & Input Slicing Fix](#adr-002-abstractive-summarization-model--input-slicing-fix)
- [ADR-003: Topic Modeling & Semantic Keyword Extraction](#adr-003-topic-modeling--semantic-keyword-extraction)
- [ADR-004: Translation Engine & Fault Tolerance](#adr-004-translation-engine--fault-tolerance)
- [ADR-005: Web Ingestion & Text Extraction Resilience](#adr-005-web-ingestion--text-extraction-resilience)
- [ADR-006: Streamlit Caching & Memory Lifecycle Management](#adr-006-streamlit-caching--memory-lifecycle-management)
- [ADR-007: Unicode TrueType Font Handling for PDF Export](#adr-007-unicode-truetype-font-handling-for-pdf-export)

---

## ADR-001: News Category Classification Model

### Status: Evaluated / Upgrade Planned
- **Current Architecture**: TF-IDF Vectorizer (10,000 max features, n-gram range `(1, 3)`, sublinear TF) + `LogisticRegression` (`C=10`, `class_weight='balanced'`, solver=`liblinear`).
- **Context & Evaluation**:
  - Dataset: 2,127 BBC news articles across 5 classes (`sport: 505`, `business: 503`, `politics: 403`, `entertainment: 369`, `tech: 347`).
  - Baseline Test Performance:
    - **Macro F1-Score**: ~0.97 on in-distribution test set.
    - **Inference Latency**: < 5 ms per request on CPU.
  - **Identified Weaknesses**:
    - Bag-of-words / TF-IDF lacks contextual synonym understanding (e.g., struggles when modern tech or economic terms not seen in the 2004 BBC dataset appear).
    - Threshold `< 0.30` for fallback to `"others"` is an empirical heuristic; the probabilities from `LogisticRegression` with high `C=10` are uncalibrated (tends to output overconfident probabilities near 0 or 1).

### Evaluation Metrics Required:
1. **Macro F1 & Weighted F1**: Balanced evaluation across categories.
2. **Expected Calibration Error (ECE)**: To verify if confidence probabilities reflect true likelihood before flagging as `"others"`.
3. **Out-of-Domain Detection Rate**: Percentage of non-BBC style modern news correctly routed to `"others"`.

### Decision / Recommended Evolution:
- **Phase 1 (Immediate)**: Apply `CalibratedClassifierCV(method='sigmoid', cv=5)` to produce realistic probabilities for the `< 0.30` confidence guardrail.
- **Phase 2 (Production Upgrade)**: Transition to **DistilRoBERTa-base** or **SetFit** (Sentence-Transformer Few-Shot) to achieve deep semantic classification while keeping inference under 35 ms on CPU.

---

## ADR-002: Abstractive Summarization Model & Input Slicing Fix

### Status: Critical Bug Identified & Optimization Planned
- **Current Architecture**: Hugging Face pipeline with `facebook/bart-large-cnn` (406M parameters).
- **Critical Finding**:
  - `App/utils.py` currently executes: `summarizer(text[:1024])[0]['summary_text']`.
  - **Issue**: `text[:1024]` slices by **characters**, not tokens. In English, 1,024 characters is only ~150–180 words. For a standard 800-word news article, the model truncates and ignores ~75% of the content.
- **Performance Trade-off**:
  - `bart-large-cnn` model weights are ~1.63 GB.
  - CPU inference takes **4.5 to 8.2 seconds** per summary on standard machines.

### Evaluation Metrics Required:
1. **ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)**: Overlap against human reference summaries.
2. **BERTScore**: Semantic similarity beyond exact n-gram matching.
3. **Faithfulness / Hallucination Rate**: Consistency of facts between summary and source article (evaluated via SummaC or LLM-as-a-judge).
4. **Compression Ratio**: Target of 20%–30% of original article length.

### Decision / Recommended Evolution:
- **Fix 1 (Immediate)**: Fix truncation by using native tokenizer truncation:
  `summarizer(text, max_length=130, min_length=30, truncation=True)`.
  For long articles (> 1024 tokens), implement a chunked map-reduce summarization strategy.
- **Fix 2 (Optimization)**: Migrate to **`sshleifer/distilbart-cnn-12-6`** (306M params) or apply **Dynamic INT8 Quantization** (`torch.quantization.quantize_dynamic`) to cut latency by ~45% with negligible ROUGE degradation.

---

## ADR-003: Topic Modeling & Semantic Keyword Extraction

### Status: Active / Tuning Planned
- **Current Architecture**: Unsupervised `BERTopic` model (Sentence-Transformers + UMAP + HDBSCAN + c-TF-IDF) combined with spaCy POS extraction (`NOUN`, `PROPN`, `ADJ`) and journalistic noise stopwords.
- **Evaluation & Context**:
  - Current implementation checks if topic is `-1` (HDBSCAN outlier), and falls back to `topic_model.find_topics(preprocessed, top_n=1)`.
  - Keyword frequency directly extracts words from the article and combines them with topic weights ($score \times 5.0$).

### Evaluation Metrics Required:
1. **Topic Coherence ($C_v$)**: Measures semantic coherence among top words in a topic cluster.
2. **Topic Diversity**: Proportion of unique words across all generated topics.
3. **Outlier Ratio**: Percentage of documents classified as `-1` (aiming for $< 15\%$).

### Decision / Recommended Evolution:
- Lower HDBSCAN `min_cluster_size` to reduce the outlier ratio.
- Replace raw frequency count with **KeyBERT** / Maximal Marginal Relevance (MMR) for single-article keyword extraction to prevent redundant synonyms from dominating the Word Cloud.

---

## ADR-004: Translation Engine & Fault Tolerance

### Status: Fragility Identified / Fallback Strategy Designed
- **Current Architecture**: `googletrans==4.0.0-rc1` for:
  1. Input normalization (Foreign $\rightarrow$ English).
  2. Summary output translation (English $\rightarrow$ Hindi, Telugu, Kannada).
- **Issue**: `googletrans` relies on an unofficial reverse-engineered API client that is subject to IP throttling, HTTP 429 rate limits, and breaking upstream schema changes.

### Decision / Recommended Evolution:
- Wrap all translation calls in an exponential-backoff retry decorator.
- Provide a clear UI fallback message if the translation endpoint times out, rather than allowing the application to crash or display an unhandled exception.
- (Long-term) Integrate local Hugging Face `Helsinki-NLP/opus-mt-en-hi` models for offline, zero-network dependency translation.

---

## ADR-005: Web Ingestion & Text Extraction Resilience

### Status: Active / Enhancement Planned
- **Current Architecture**: `newspaper3k` for URL scraping and `PyMuPDF` (`fitz`) for PDF streaming.
- **Evaluation**:
  - `newspaper3k` performs well on standard server-side rendered HTML articles, but fails on JavaScript-heavy SPAs (React/Vue/Next.js news portals) or sites with Cloudflare protection.
  - PDF extraction with `PyMuPDF` is exceptionally fast and robust.

### Decision / Recommended Evolution:
- Add custom User-Agent headers to requests to prevent 403 Forbidden errors from bot detection.
- Add `trafilatura` as a primary extractor with `newspaper3k` as fallback for higher precision text/body extraction.

---

## ADR-006: Streamlit Caching & Memory Lifecycle Management

### Status: Optimization Planned
- **Current Architecture**:
  - `spacy.load("en_core_web_sm")`, `pipeline("summarization")`, and `joblib.load()` are declared at global module scope in `App/utils.py`.
- **Issue**:
  - In Streamlit, un-cached global objects can be initialized repeatedly during module imports or worker restarts, bloating memory consumption by > 2 GB.
  - Artificial delay `time.sleep(0.5)` in `App/app.py` unnecessarily slows down user interaction.

### Decision / Recommended Evolution:
- Wrap all model loaders with `@st.cache_resource` in `utils.py` or `app.py`.
- Remove simulated sleep delays.
- Persist extracted texts and computed summaries in `st.session_state` to prevent re-running costly transformer inference when toggling UI language tabs.

---

## ADR-007: Unicode TrueType Font Handling for PDF Export

### Status: Implemented & Verified
- **Architecture**: `fpdf` with custom font discovery for `DejaVu Sans`, `Arial`, and `Segoe UI`.
- **Decision**: Standard FPDF 1.7 fails with `UnicodeEncodeError` when encountering Indian Rupee symbols (`₹`), curved quotes, or em-dashes. Dynamic font lookup with Latin-1 replacement fallback guarantees zero export crashes.
