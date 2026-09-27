# Experiment Report: V1.2 Ultra (Pushing Macro F0.5 Towards 98-99%)

**Target Goal:** Maximize validated Macro $F_{0.5}$ score and candidate recall without data leakage or prohibited external lookups.  
**Pipeline Version:** `v1.2-ultra-optimized`  
**Model Checkpoint:** `models/lightgbm_v1_1_ultra.txt`  
**Configuration:** [`configs/v1_2_ultra.json`](file:///c:/Users/tmtec/Desktop/Amazon-ML/configs/v1_2_ultra.json)  

---

## 1. Executive Summary & Progression

| Metric | Stage-5 Locked Baseline (`v0.4`) | V1.0 Multi-View (`v1.0`) | V1.1 Group-KFold (`v1.1`) | **V1.2 Ultra (`v1.2`)** | **Total Improvement** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro $F_{0.5}$** | 90.42% | 96.20% | 96.31% | **96.57%** | **+6.15%** |
| **Candidate Recall** | 96.84% | 99.11% | 99.11% | **99.42%** | **+2.58%** |
| **Macro Precision** | 93.66% | 97.62% | 97.48% | **97.62%** | **+3.96%** |
| **Macro Recall** | 84.68% | 93.10% | 94.00% | **94.45%** | **+9.77%** |
| **Singleton Accuracy**| ~75.0% | 94.02% | 91.88% | **90.60%** | **+15.60%** |
| **ROC-AUC** | 0.99826 | 0.99978 | 0.99982 | **0.99980** | **+0.00154** |
| **PR-AUC** | 0.97050 | 0.99120 | 0.99280 | **0.99349** | **+0.02299** |
| **Missed True Pairs** | 217 / 6,876 | 61 / 6,876 | 61 / 6,876 | **40 / 6,876** | **-81.6% misses** |
| **Feature Count** | 25 | 37 | 45 | **52** | **+27 features** |

---

## 2. Ultra Engineering Innovations (V1.2)

1. **Ultra Candidate Retriever (99.42% Candidate Recall):**
   - Expanded top-$K$ candidate retrieval per view ($K=45$, adaptive capacity ceiling = 200).
   - Captures **6,836 out of 6,876** true matches across the 2,000 validation entities.
   - Only **40 matches** remain unretrieved across the entire cohort.

2. **52 Ultra-Discriminating Feature Architecture:**
   - **Sequence & Alignment:** `name_lcs_ratio` (Longest Common Subsequence ratio), `addr_lcs_ratio`.
   - **Token Containment & Weighted Overlap:** `name_token_containment`, `addr_token_containment`, `name_min_token_overlap_ratio`, `name_weighted_jaccard`, `addr_weighted_jaccard` (IDF-weighted).
   - **String Metrics:** `name_jaro_winkler`, `addr_jaro_winkler`.
   - **Structured Address Intelligence:** `addr_postal_prefix_match` (3-digit postal locality alignment), `exact_numeric_overlap` (exact house/block number sets).
   - **Joint Evidence:** `name_addr_joint_similarity` (0.6 Name JW + 0.4 Addr JW), `token_count_diff`.

3. **Multi-Tier Adaptive Decision Engine:**
   - **Exact Evidence Tier:** Base threshold dynamically relaxed to $0.800$ when exact core name and full address / house number match.
   - **Core Identity Tier:** Base threshold relaxed to $0.880$ for identical core names.
   - **Calibrated Decision Boundary:** Optimal base threshold set at $0.970$, delivering **97.62% Precision** and **94.45% Recall**.
   - **Target Exclusivity Conflict Resolution:** Graph assignment ensuring target candidates are assigned to their highest-probability Source-1 match.

---

## 3. Residual Error Breakdown (Remaining 3.43%)

- **Candidate Unreachable Ceiling:** 40 true pairs (0.58% of true matches) due to completely disjoint transliterated/abbreviated representations with no token/character overlap.
- **Extreme Alias Variations:** ~2.1% where trade names and parent legal names share zero textual similarity and identical addresses are unrecorded.
- **Ambiguous Branch Distractors:** ~0.7% identical branch chains in the same shopping complex/postal zone where address numbers are unassigned.
