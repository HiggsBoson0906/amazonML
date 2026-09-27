# Amazon ML Challenge 2026 — Final Pipeline Report (V1.0)

**Project:** Business Entity Resolution Challenge  
**Pipeline Version:** `v1.0-final-candidate`  
**Repository:** https://github.com/HiggsBoson0906/amazonML  
**Evaluation Benchmark:** Fixed 2,000 Source-1 Validation Entities (6,876 True Ground-Truth Pairs)  

---

## 1. Executive Summary & Comparison with Stage-5 Baseline

| Metric | Stage-5 Baseline | V1.0 Optimized Final | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Candidate Recall Ceiling** | 96.84% | **99.11%** | **+2.27%** (+156 true matches) |
| **End-to-End Macro $F_{0.5}$** | 90.42% | **96.20%** | **+5.78%** |
| **Macro Precision** | 93.66% | **97.62%** | **+3.96%** |
| **Macro Recall** | 84.68% | **93.10%** | **+8.42%** |
| **Singleton Accuracy** | 88.89% | **94.02%** | **+5.13%** |
| **Validation ROC-AUC** | 0.99826 | **0.99967** | **+0.00141** |
| **Validation PR-AUC** | 0.97050 | **0.99194** | **+0.02144** |
| **Average Candidates / S1** | 75.06 | **125.97** | Capped at 160 budget |
| **Feature Extraction Speedup**| 1.0x (Legacy) | **1.72x (Fast Vectorized)**| Precomputed primitives |

---

## 2. End-to-End Pipeline Architecture

```
                       +---------------------------------------+
                       |           Raw TSV Records             |
                       |  (Source 1, Source 2, Source 3)       |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Normalization & Precomputation      |
                       |  - Multilingual Transliteration (IN)  |
                       |  - Double Metaphone / Phonetics       |
                       |  - Core Business Name Extraction      |
                       |  - Address Normalization & Components |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |    Multi-View Candidate Retrieval     |
                       |  1. Stage-5 Inverted Index Blocker    |
                       |  2. Name TF-IDF Top-30                |
                       |  3. Address TF-IDF Top-30             |
                       |  4. Char 3-gram TF-IDF Top-30         |
                       |  Union Budget: Max 160 Candidates/S1  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |        Pairwise Feature Engine        |
                       |  - 25 Fine-Grained String Similarities|
                       |  - 6 Structured Address Component Evid|
                       |  - 4 Corpus Rarity Log Frequencies    |
                       |  - 2 Retrieval Consensus & Max Score  |
                       |  Total: 37 Float32 Features           |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       LightGBM Binary Classifier      |
                       |  - 45 Leaves, Depth 7, GBDT           |
                       |  - Optimal Decision Threshold: 0.960  |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |           Decision Layer              |
                       |  - Singleton Protection Guard         |
                       |  - Global Target Consistency Postproc |
                       +-------------------+-------------------+
                                           |
                         +-----------------+-----------------+
                         |                                   |
                         v                                   v
             matching_results.tsv                   candidate_pairs.tsv
      (Strict Subset of candidate_pairs)      (Exact Candidates Fed to LightGBM)
```

---

## 3. Candidate Generation & Multi-View Retrieval
Candidate generation combines Stage-5 deterministic inverted indexing with sparse linear TF-IDF vector spaces:
1. **Stage-5 Inverted Index Blocker:** Multi-word address signatures, acronym signatures, ordered 2/3-token signatures, Indic transliteration phonetic compounds, and tiered quota budgeting.
2. **Name Word TF-IDF:** Top-30 nearest neighbors in country-partitioned TF-IDF vector spaces.
3. **Address Word TF-IDF:** Top-30 nearest neighbors matching street/locality word combinations.
4. **Name Character 3-Gram TF-IDF:** Top-30 sub-word character n-gram cosine similarities for misspelling tolerance.
5. **Candidate Union:** All views are merged and deduplicated, respecting a strict maximum capacity cap of 160 candidates per Source-1 entity.

---

## 4. Complete 37-Feature Specification

### A. Base Name Similarities (13 Features)
1. `exact_norm_name`: Exact equality of normalized names (1.0 or 0.0)
2. `exact_core_name`: Exact equality of core business names (1.0 or 0.0)
3. `name_ratio`: Levenshtein similarity ratio between normalized names
4. `name_wratio`: RapidFuzz WRatio weighted similarity
5. `name_token_sort`: Token sort ratio on normalized names
6. `name_token_set`: Token set ratio on normalized names
7. `name_partial`: Partial substring ratio on normalized names
8. `core_ratio`: Levenshtein similarity on stripped core names
9. `core_token_sort`: Token sort ratio on core names
10. `core_token_set`: Token set ratio on core names
11. `name_jaccard`: Token-level Jaccard intersection over union
12. `name_char_3gram`: Character 3-gram Jaccard similarity
13. `name_len_diff`: Absolute difference in name lengths

### B. Base Address Similarities (9 Features)
14. `addr_missing`: Indicator if either address is null/empty
15. `addr_exact`: Exact equality of normalized addresses
16. `addr_ratio`: Levenshtein similarity ratio on addresses
17. `addr_token_sort`: Token sort ratio on addresses
18. `addr_token_set`: Token set ratio on addresses
19. `addr_jaccard`: Token-level Jaccard similarity on address words
20. `addr_char_3gram`: Character 3-gram Jaccard on address strings
21. `addr_num_match`: Binary overlap indicator between numeric digits in addresses
22. `addr_len_diff`: Absolute difference in address character lengths

### C. Metadata & Source Features (3 Features)
23. `country_match`: Exact equality of country strings
24. `is_source2`: Binary indicator for Target Source 2 origin
25. `is_source3`: Binary indicator for Target Source 3 origin

### D. Structured Address Components (6 Features)
26. `addr_hnum_match`: Exact match of parsed house number
27. `addr_hnum_conflict`: Conflict indicator where both records have different house numbers
28. `addr_postal_match`: Exact match of parsed 5/6-digit postal/PIN code
29. `addr_postal_conflict`: Conflict indicator where postal codes are present and disagree
30. `addr_digits_overlap`: Count of overlapping numeric tokens
31. `addr_digits_conflict`: Conflict indicator when numeric sets are non-empty and disjoint

### E. Corpus Rarity & Frequency (4 Features)
32. `s1_name_log_freq`: $\log(1 + \text{count})$ of Source-1 name in corpus
33. `tgt_name_log_freq`: $\log(1 + \text{count})$ of Target name in corpus
34. `s1_addr_log_freq`: $\log(1 + \text{count})$ of Source-1 address in corpus
35. `tgt_addr_log_freq`: $\log(1 + \text{count})$ of Target address in corpus

### F. Retrieval Evidence & Consensus (2 Features)
36. `retrieval_views_count`: Number of distinct retrieval views generating this pair (1.0 to 4.0)
37. `max_tfidf_score`: Maximum TF-IDF cosine score across all sparse retrieval views

---

## 5. Machine Learning Model & Training Parameters
- **Architecture:** Gradient Boosted Decision Trees (LightGBM Booster)
- **Objective:** `binary` (Log-Loss)
- **Metric:** `auc`
- **Boosting Type:** `gbdt`
- **Learning Rate:** `0.04`
- **Num Leaves:** `45`
- **Max Depth:** `7`
- **Feature Fraction:** `0.85`
- **Bagging Fraction:** `0.85`
- **Bagging Freq:** `1`
- **Min Child Samples:** `20`
- **Early Stopping:** 30 rounds on validation set
- **Decision Threshold:** `0.960`

---

## 6. Post-Processing & Decision Layer

### A. Singleton Protection Guard
For S1 entities where candidate probabilities are borderline ($\ge 0.960$ but $< 0.992$) and multiple candidates have small probability margins ($< 0.05$), the singleton protection layer suppresses false multi-match collisions to prevent the fatal $0.0$ singleton penalty.

### B. Global Target Consistency (Exclusivity Conflict Resolution)
In real-world business directories, a distinct physical Target record cannot simultaneously represent two unrelated Source-1 entities. If multiple Source-1 entities predict a match with the same Target ID, the post-processor resolves the conflict greedily by assigning the Target entity to the Source-1 query with the highest model confidence.

---

## 7. Data Leakage & Validation Integrity
- **Disjoint Partitioning:** Validation cohort consists of exactly 2,000 S1 entities strictly disjoint from the 7,997 training entities (0 overlap).
- **No Ground-Truth Leakage:** TF-IDF vectorizers, corpus frequency counters, and InvertedIndexBlockers are fit purely on raw text records without using ground-truth match labels.
- **Unseen Test Compatibility:** Country is treated as an open-set field (supporting US, India, France, and any novel country tokens). Missing addresses and fields are never imputed or fabricated.

---

## 8. Output Integrity & Official Validator Compliance
- **Matching Results:** `outputs/matching_results.tsv` (exactly 1 row per S1, tab-separated).
- **Candidate Pairs:** `outputs/candidate_pairs.tsv` (exactly 1 row per S1, tab-separated).
- **100% Subset Invariant:** For every Source-1 entity, $\text{set}(\text{matched\_entity\_ids}) \subseteq \text{set}(\text{candidate\_entity\_ids})$.
- **Official Validator:** Verified with `utils/validate_submission.py`.

---

## 9. Reproducibility Commands

### To Re-run All Experiments (E0 through E12):
```bash
python scripts/run_all_experiments.py
```

### To Retrain and Persist the V1.0 Final Model:
```bash
python scripts/train.py
```

### To Run Validation on Any Submission:
```bash
python utils/validate_submission.py \
    --matching outputs/matching_results.tsv \
    --candidate outputs/candidate_pairs.tsv \
    --test-dir dataset/test
```
