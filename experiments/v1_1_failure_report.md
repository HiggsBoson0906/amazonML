# V1.0 Pipeline Deep Residual Error Analysis Report

**Evaluation Cohort:** 2,000 Source-1 Validation Entities  
**Ground-Truth Matches:** 6,876 True Pairs  
**Total False Negatives:** 452 (6.57% of true matches)  
**Total False Positives:** 125  

---

## 1. False Negative Category Distribution

| Category | Count | Percentage | Primary Root Cause & Fix Opportunity |
| :--- | :---: | :---: | :--- |
| **threshold_boundary** | 282 | 62.4% | Probability in [0.70, 0.960): add stronger string/rarity features and margin-based decision function. |
| **candidate_missing** | 61 | 13.5% | Unreachable candidate in retrieval: add targeted Char 4/5-gram and transliteration TF-IDF channels. |
| **score_too_low** | 44 | 9.7% | Score too low due to insufficient similarity signal. |
| **multilingual_or_abbreviation** | 42 | 9.3% | Heavy name variation / abbreviation: add token containment, Dice, and Jaro-Winkler features. |
| **postprocessing_conflict_suppression** | 19 | 4.2% | Target exclusivity conflict: refine multi-assignment resolution with probability margins. |
| **address_divergence** | 4 | 0.9% | Different address representation / landmark: weight core name higher when name match is exact. |

---

## 2. False Positive Category Distribution

| Category | Count | Percentage | Primary Root Cause & Fix Opportunity |
| :--- | :---: | :---: | :--- |
| **high_similarity_wrong_entity** | 92 | 73.6% | Branch / subsidiary similarity: compute candidate-competition score margins to distinguish true vs nearby entity. |
| **generic_name_different_address** | 17 | 13.6% | Common business name occurring at distinct location: penalize via corpus frequency and address conflict features. |
| **shared_address_different_name** | 9 | 7.2% | Commercial building / business park sharing address: enforce core name similarity requirement. |
| **singleton_false_positive** | 7 | 5.6% | Singleton entity falsely matching distractor: raise singleton margin floor and ambiguity filters. |

---

## 3. Targeted Optimization Action Plan for V1.1

1. **Candidate Retrieval (Remaining 61 Misses):**
   - Add **Char 4-gram / 5-gram TF-IDF** and **Transliterated Name TF-IDF** to push candidate recall from 99.11% toward 99.5%+.
2. **Precision & Rarity Weighting:**
   - Incorporate **Token IDF weights**, **House Number/Postal Conflict penalties**, and **Corpus Rarity Log Counts** to suppress generic name false positives.
3. **Candidate-Competition Features:**
   - Add **Probability Margins** ($\Delta = P_{	ext{top1}} - P_{	ext{top2}}$) and **Candidate Ranks** to distinguish true entities from competing branch distractors.
4. **Margin-Aware Decision Function:**
   - Accept matches with high confidence ($P \ge 0.95$) and sufficient margin, while allowing exact name + exact address overrides.
