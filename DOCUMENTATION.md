# Amazon ML Challenge 2026 — Business Entity Resolution
## Technical Solution Documentation

---

### 1. Problem & Objective
The challenge requires resolving noisy business entities across three heterogeneous datasets: **Source 1** (reference queries) and **Source 2 / Source 3** (potential target records). The primary real-world difficulties stem from extreme data noise:
- **Typographical & spelling variations**, abbreviations, and phonetic shifts.
- **Legal suffix permutations** (e.g., *LLC, Inc, Ltd, Private Limited*).
- **Multilingual and transliterated entity names** (e.g., Devanagari/Indic to Latin).
- **Inconsistent address representations**, missing fields, and variable postal formatting.

Because the evaluation metric is **Macro $F_{0.5}$**, precision is weighted twice as heavily as recall ($\beta = 0.5$). False positive matches heavily penalize the official score. To maximize $F_{0.5}$, we design a two-stage decoupled architecture: **high-recall multi-view candidate generation** (to ensure true matches are never pruned) coupled with a **50-feature high-precision ranking classifier and surgical post-processing layer**.

---

### 2. Solution Architecture

```
                       [ Source 1 Query Record ]
                                   │
                                   ▼
               [ Canonical Normalization & Tokenization ]
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
 [ Stage-5 Multi-Pass Blocking ]              [ Multi-View TF-IDF Retrieval ]
   • Core & Name Inverted Index                 • Name Word Unigram / Bigram
   • Transliteration & Phonetic Skeletons       • Address Word Unigram / Bigram
   • Structured Address & House Number          • Name Character 3-gram Sparse View
           │                                               │
           └───────────────────────┬───────────────────────┘
                                   ▼
                   [ Candidate Set Union & Top-K ]
                       (Capacity ≤ 200 per S1)
                                   │
                                   ▼
                   [ 50-Feature Extraction Engine ]
                    (Ultra-Fast Precomputed Invariants)
                                   │
                                   ▼
                    [ LightGBM Ultra Classifier ]
                     (Pairwise Match Probabilities)
                                   │
                                   ▼
                   [ V1.3 Surgical Decision Layer ]
                     • Calibrated Base Threshold (0.965)
                     • Dynamic Exact Evidence Relaxation
                     • Singleton Protection Guard (0.900)
                     • Competition Margin Gating (0.020)
                     • Global Target Exclusivity
                                   │
                                   ▼
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
[ matching_results.tsv ]                        [ candidate_pairs.tsv ]
 (Final High-Precision Matches)                   (Complete Evaluated Pool)
```

---

### 3. Normalization & Candidate Generation
Candidate generation reduces the $O(N \times M)$ comparison space while guaranteeing near-perfect recall ceiling:
1. **Canonical Normalization**: Standardizes punctuation, whitespace, casing, and canonical legal entity suffixes. Extracts a stripped *core business name*.
2. **Cross-Lingual Transliteration & Phonetics**: Converts Indic scripts to Latin phonemes and computes double-metaphone and phonetic skeleton tokens for robust phonetic matching.
3. **Structured Address Keying**: Decomposes addresses into numerical tokens, street keywords, house numbers, and postal codes to construct compound blocking keys (e.g., `num_word`, `ph_num`).
4. **Multi-View Sparse TF-IDF Retrieval**: Computes country-sharded cosine similarities across three complementary sparse vector spaces: name word unigram/bigram, address word unigram/bigram, and name character 3-grams.
5. **Candidate Pool Management**: Prunes excessively frequent blocking keys (>500 entities) and combines candidate sets up to a strict capacity of 200 targets per query.

---

### 4. 50-Feature Ranking Model
Pairwise match decisions are modeled with LightGBM (`models/lightgbm_v1_1_ultra.txt`) trained over 50 numerical features:

| Feature Family | Features & Signals |
| :--- | :--- |
| **Name Similarity (13)** | Exact normalized/core match, Levenshtein ratio, token sort/set ratio, weighted token Jaccard (IDF-weighted), character 3-gram overlap, containment ratio, Jaro-Winkler similarity, Longest Common Subsequence (LCS) ratio, minimum token overlap ratio. |
| **Address Similarity (15)** | Exact address match, missing address flag, token sort/set ratio, token Jaccard, weighted Jaccard, character 3-gram overlap, numeric overlap, Jaro-Winkler, LCS ratio, address containment ratio. |
| **Structured Address (7)** | House number match/conflict, postal code match/conflict, 3-digit postal prefix match, address digit overlap count, digit conflict flag. |
| **Corpus Frequency & Metadata (10)** | S1/Target name log-frequency, S1/Target address log-frequency, country match flag, Source 2 / Source 3 origin indicators, token count difference, exact numeric overlap. |
| **Cross-Modal & Retrieval (5)** | Name-address joint similarity, retrieval view support count, max TF-IDF retrieval score. |

---

### 5. V1.3 Surgical Decision Layer
Rather than applying a single naive threshold, predictions are processed through a multi-tier surgical decision engine:
- **Calibrated Base Threshold ($0.965$)**: High-precision cutoff tuned to eliminate ambiguous predictions.
- **Dynamic Evidence Overrides**: Relaxes threshold ($0.800 - 0.895$) only when unambiguous deterministic evidence is present (e.g., exact core name match + exact address match, or verified postal + house number match).
- **Singleton Protection Guard ($0.900$)**: Suppresses weak marginal single candidates with low joint similarity ($<0.75$), preventing non-matching queries from becoming false positive singletons.
- **Competition Margin Gating ($\Delta p \ge 0.020$)**: Resolves top candidate ties by enforcing a minimum margin against competing candidates unless strong exact name evidence exists.
- **Joint Similarity Floor ($0.450$)**: Hard safety boundary preventing model hallucination on severe name/address conflicts.
- **Global Target Exclusivity**: Enforces 1-to-1 consistency for mutually exclusive target entities.

---

### 6. Inference Optimization & Production Engineering
To process the 2,206,821 Source 1 queries within strict memory and compute limits without modifying model semantics:
- **Precomputed Target Invariants**: Target entities are indexed into memory with pre-tokenized sets, character 3-grams, extracted address primitives, and log frequencies.
- **Vectorized Country-Sharded Retrieval**: Batch TF-IDF matrix multiplications use sparse GEMM operations, avoiding per-record vectorizer overhead.
- **Allocation-Free Feature Extraction**: Vectorized string routines and precomputed properties in `extract_ultra_features_fast()` eliminate repetitive allocations and achieve 100.0% numerical equivalence with the original feature definitions.
- **Streaming Chunk Processing**: Queries are evaluated in memory-bounded chunks of 5,000 records, flushing results directly to disk.

---

### 7. Validation & Pre-Flight Benchmark Results

All components underwent deterministic verification against the fixed validation cohort and official test pre-flight suite:

#### A. Deterministic Pre-Flight Equivalence Checks
- **Feature Calculation Equivalence**: 900 feature values verified; Maximum Absolute Delta = **`0.0000000000000000`**; Mismatches ($>10^{-12}$) = **0**.
- **Candidate Set Equivalence**: 1,000 queries tested; Candidate Set Differences = **0 / 1,000** (**100.0% identical**).
- **Model Output Equivalence**: 136,506 candidate pairs scored; Max Probability Delta = **`0.0000000000000000`**; Prediction Differences = **0**.
- **Candidate Recall Ceiling**: **`99.943%`** (3,515 / 3,517 validation ground-truth links captured).

#### B. Validation Performance & Throughput Metrics

| Metric Category | Metric Name | Measured Value | Target / Requirement | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Validation Quality** | Macro $F_{0.5}$ | **97.95%** | Maximized | [✓] |
| | Macro Precision | **98.72%** | High Precision ($\beta=0.5$) | [✓] |
| | Macro Recall | **96.12%** | High Coverage | [✓] |
| | Singleton Accuracy | **90.74%** | $\ge 90.0\%$ | [✓] |
| **Inference Speed** | Feature Extraction Time | **8.58 ms / S1** | $\le 12.0\text{ ms / S1}$ | [✓] |
| | Single-Core Throughput | **60.81 S1 / sec** | $\ge 50.0\text{ S1 / sec}$ | [✓] |
| | Projected 8-vCPU Throughput | **413.5 S1 / sec** | $\ge 350.0\text{ S1 / sec}$ | [✓] |
| **Resource Utilization** | Peak RAM Consumption | **1.88 GiB** | $< 10.0\text{ GiB}$ (64 GiB budget) | [✓] |
| | Projected 2.2M Full Runtime | **~1.48 hours (1h 29m)** | $\le 2.0\text{ hours}$ | [✓] |

*(Note: Validation scores are measured on the fixed validation cohort; test set evaluation is executed separately).*

---

### 8. Output & Engineering Validation
- **Challenge Compliant Output**: `matching_results.tsv` and `candidate_pairs.tsv` strictly follow the tab-separated submission specifications.
- **Strict Subset Invariant**: Validated that for every record, $\text{matched\_set} \subseteq \text{candidate\_set}$.
- **Uniqueness & Format Integrity**: Zero duplicate match entity IDs, zero fabricated entity IDs, and safe UTF-8 encoding across all international records.
- **Self-Contained Pipeline**: Runs entirely offline using standard open-source libraries without external API dependencies or external data lookups.

---

### 9. Conclusion
The proposed solution combines **multi-pass blocking and sparse TF-IDF retrieval** for high candidate recall, a **50-feature LightGBM ranking model** capturing granular orthographic and semantic signals, and a **V1.3 surgical post-processor** tailored specifically for the precision-weighted Macro $F_{0.5}$ metric. With targeted inference optimizations, the pipeline achieves 60.81 S1/sec throughput and 1.88 GiB peak memory, ensuring high accuracy, scalability, and deterministic reproducibility.
