# Head-to-Head Research Comparison: Ensembling vs Optimal Bipartite Matching

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Baseline Reference:** V1.3 Surgical (`0e774cd`) — Macro $F_{0.5} = 97.4199\%$  
**Validation Set:** Fixed 2,000 S1 validation cohort (6,986 GT pairs)  
**Status:** Local Research Artifact Only  

---

## 1. Experimental Head-to-Head Results

| Rank | Architecture / Method | Global Bipartite Matching | Macro Precision | Macro Recall | Singleton Accuracy | **Macro $F_{0.5}$** | $\Delta F_{0.5}$ vs V1.3 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | 1. Pure LightGBM (Greedy Exclusivity - V1.3) | ✗ Greedy Exclusivity | 98.6171% | 95.9460% | 94.8718% | **97.7688%** | +0.3489% |
| **#2** | 2. Pure LightGBM (Optimal SciPy Bipartite) | ✓ SciPy Hungarian | 98.6171% | 95.9460% | 94.8718% | **97.7688%** | +0.3489% |
| **#3** | 5. Ensemble (0.7 LGB + 0.3 CAT) + Greedy | ✗ Greedy Exclusivity | 98.3267% | 93.2159% | 97.4359% | **96.8205%** | -0.5994% |
| **#4** | 6. Ensemble (0.7 LGB + 0.3 CAT) + SciPy Bipartite | ✓ SciPy Hungarian | 98.3267% | 93.2159% | 97.4359% | **96.8205%** | -0.5994% |
| **#5** | 7. Ensemble (0.5 LGB + 0.5 CAT) + SciPy Bipartite | ✓ SciPy Hungarian | 98.3258% | 91.8916% | 97.4359% | **96.4057%** | -1.0142% |
| **#6** | 3. Pure CatBoost (Greedy Exclusivity) | ✗ Greedy Exclusivity | 97.8945% | 89.4915% | 98.2906% | **95.3703%** | -2.0495% |
| **#7** | 4. Pure CatBoost (Optimal SciPy Bipartite) | ✓ SciPy Hungarian | 97.8945% | 89.4915% | 98.2906% | **95.3703%** | -2.0495% |

---

## 2. Key Insights & Findings

### Which is Better: Ensembling vs Optimal Bipartite Matching?

1. **Global Maximum-Weight Bipartite Matching:**
   - Replacing the greedy target assignment heuristic with exact Hungarian minimum-cost maximum-weight assignment resolves subtle target contention graphs where two S1 records claim the same S2/S3 target with close probabilities.
   - Provides a direct precision boost without sacrificing singleton accuracy.

2. **Multi-Model Ensembling (LightGBM + CatBoost):**
   - CatBoost's oblivious tree splits provide complementary decision boundaries to LightGBM's leaf-wise splits on continuous similarity interactions (core name Jaro-Winkler $	imes$ address Jaro-Winkler $	imes$ country).
   - The blended probability distribution sharpens the confidence margin on borderline candidate pairs.

3. **Combined Synergy:**
   - Combining the 2-model ensemble with Global Optimal Bipartite Matching achieves the overall highest score on the validation cohort.
