# Amazon ML Challenge 2026: Dataset Inspection Report

**Date:** September 2026  
**Workspace:** `C:\Users\tmtec\Desktop\Amazon-ML`  
**Dataset Path:** `dataset/` (Train: `dataset/train/`, Test: `dataset/test/`)

---

## 1. Executive Summary

A comprehensive, zero-data-loss streaming audit was conducted across all 7 dataset TSV files (4 training files, 3 test files). 
* **Total Records Analyzed:** 26,355,995 rows (~2.5 GB on disk).
* **Format & Encoding:** Standard Tab-Separated Values (`.tsv`), UTF-8 encoded with multilingual characters (Latin, accented French, Devanagari script for Hindi/Marathi, special symbols).
* **File Integrity:** **0 malformed lines**, **0 header errors**, clean 1:1 TSV delimiter formatting.

---

## 2. File Specifications & Statistics

### Training Files (`dataset/train/`)

| File Name | Size (MB) | Total Lines | Data Rows | Malformed Rows | Null/Empty Fields |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`train_source1.tsv`** | 200.34 MB | 2,206,822 | **2,206,821** | 0 | None (100% complete) |
| **`train_source2.tsv`** | 466.63 MB | 5,034,617 | **5,034,616** | 0 | `business_address`: 168,967 (3.36%) |
| **`train_source3.tsv`** | 480.37 MB | 5,285,604 | **5,285,603** | 0 | `business_address`: 175,916 (3.33%) |
| **`train_ground_truth.tsv`** | 121.13 MB | 2,206,822 | **2,206,821** | 0 | `matched_entity_ids`: 123,247 (5.58%) |

### Test Files (`dataset/test/`)

| File Name | Size (MB) | Total Lines | Data Rows | Malformed Rows | Null/Empty Fields |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`test_source1.tsv`** | 166.91 MB | 1,732,545 | **1,732,544** | 0 | None (100% complete) |
| **`test_source2.tsv`** | 485.86 MB | 4,887,274 | **4,887,273** | 0 | `business_address`: 129,408 (2.65%) |
| **`test_source3.tsv`** | 482.56 MB | 5,082,317 | **5,082,316** | 0 | `business_address`: 136,098 (2.68%) |

---

## 3. Country & Language Distribution

| Dataset | Total Records | US (%) | India (%) | France (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | 2,206,821 | 1,323,633 (60.0%) | 883,188 (40.0%) | *0 (0.0%)* |
| **Train Source 2** | 5,034,616 | 3,016,817 (59.9%) | 2,017,799 (40.1%) | *0 (0.0%)* |
| **Train Source 3** | 5,285,603 | 3,170,056 (60.0%) | 2,115,547 (40.0%) | *0 (0.0%)* |
| **Test Source 1** | 1,732,544 | 663,106 (38.3%) | 809,986 (46.8%) | **259,452 (15.0%)** |
| **Test Source 2** | 4,887,273 | 1,871,330 (38.3%) | 2,312,565 (47.3%) | **703,378 (14.4%)** |
| **Test Source 3** | 5,082,316 | 1,945,701 (38.3%) | 2,405,000 (47.3%) | **731,615 (14.4%)** |

> [!IMPORTANT]
> **Open-Set Country Notice (`France` in Test):**
> Notice that `France` appears in the test dataset (~15%) but is **not** in the training set. All normalization, blocking, and matching pipelines must support French entity names (accents, French legal suffixes like `SARL`, `SAS`, `SCI`, `SASU`) and address tokens without hardcoded US/India assumptions.

---

## 4. Ground Truth Matching Characteristics

Analysis of `train_ground_truth.tsv` (2,206,821 Source 1 entities):

* **Singletons (0 Matches):** 123,247 entities (**5.58%**) have no matching entities in Source 2/Source 3. Under macro $F_{0.5}$, predicting an empty match list for singletons awards a score of 1.0, while false positives yield 0.0.
* **1 Match:** 119,157 (5.40%)
* **2 Matches:** 375,212 (17.00%)
* **3 Matches:** 530,841 (24.05%)
* **4 Matches:** 484,115 (21.94%)
* **5 Matches:** 321,957 (14.59%)
* **6 Matches:** 164,868 (7.47%)
* **7 Matches:** 63,968 (2.90%)
* **8 Matches:** 18,680 (0.85%)
* **9–11 Matches:** 4,596 (0.21%)
* **Mean Matches per S1:** ~3.34 matches.

---

## 5. Noise Patterns and Entity Variation Identified

1. **Devanagari & Multilingual Scripts:**
   - Examples in India: `राम मार्केटिंग प्राइवेट लिमिटेड` (Ram Marketing Private Limited), `आदित्य प्रॉपर्टीज एलएलपी` (Aditya Properties LLP), `मॉडर्न फाइनेंस` (Modern Finance).
   - Transliteration variations occur between Devanagari script and Latinized English spellings.
2. **French Entity Structures:**
   - Accented characters: `À`, `é`, `è`, `ô`, `ç` (e.g. `SCI Ptit Àmicale`, `Fractales Amis Groupe S.A.S`).
   - French legal suffixes: `SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SA`, `SNC`.
   - French address tokens: `Rue`, `Boulevard` / `Blvd` / `Bd`, `Avenue` / `Av`, `R. DE DIEPPE`.
3. **Punctuation and Formatting Noise:**
   - Decorative symbols: `-- Holloway Peak Inc Seafood`, `<< Team Ecole`, `B+ Retail Inc`.
   - Domain-style entity names: `wilfordhancock.com`.
4. **Missing Addresses:**
   - ~3% of records in Source 2 and Source 3 have blank/empty addresses (`""`). Address comparison must gracefully handle missing addresses without producing NaN errors or false negatives.
5. **Address Abbreviations & Landmark Styles:**
   - US: State abbreviations (`NC` ↔ `North Carolina`, `OK` ↔ `Oklahoma`, `TX` ↔ `Texas`), street abbreviations (`Rd`, `Ct`, `Ave`, `St`, `Dr`).
   - India: Plot / Khasra / Khata numbers (`KH NO. -570/13`), Landmark indicators (`Near SBI ATM`), District / Taluk formats (`HUNSUR TQMYSORE DIST.`), State codes (`DL` ↔ `Delhi`, `HR` ↔ `Haryana`).

---

## 6. Pipeline Design Implications

1. **Polars & Memory Efficiency:**
   - With 26M total rows across sources, full Cartesian product ($2.2M \times 10.3M \approx 2.2 \times 10^{13}$ pairs) is mathematically impossible on 16 GB RAM.
   - We must use multi-pass inverted-index / key-based blocking with Polars lazy frames or chunked streaming.
2. **Blocking Key Strategy:**
   - Partitioning by `country` is safe (no cross-country matches expected).
   - High-precision blocking keys: normalized clean name, first 2-3 significant tokens, soundex/metaphone or rare token keys, address PIN/Zip or city tokens.
3. **Robust Text Normalizer:**
   - Unicode NFKD normalization (strip accents while preserving ASCII base for French/Latin).
   - Removal of noise prefixes (`--`, `<<`, `**`, etc.).
   - Canonical replacement of `&` $\to$ `and`, `+` $\to$ `plus`.
   - Legal suffix stripping / standardizing (US, India, France).
4. **Deterministic Baseline:**
   - Stage 1: Build the deterministic baseline and evaluate $F_{0.5}$ on holdout validation.
   - Stage 2: Train LightGBM ranker on hard negatives and optimize decision threshold.
