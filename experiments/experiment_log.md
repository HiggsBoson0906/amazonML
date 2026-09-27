# Amazon ML Challenge 2026 — Experiment Log

This document tracks all optimization hypotheses, implementations, quantitative validation metrics, decisions (KEEP/REJECT), and rationales across the experimentation lifecycle.

## Evaluation Benchmark Setup
- **Validation Cohort:** Fixed 2,000 Source-1 entities (from disjoint train/val partition)
- **Ground-Truth Target Matches:** 6,876 true pairs
- **Evaluation Metrics:** Official Macro F0.5 (precision-heavy, singleton penalization), Macro Precision, Macro Recall, Pairwise F0.5, Candidate Recall Ceiling, Singleton Accuracy, Inference Runtime, Peak RAM.

---

## Experiment Matrix & Quantitative Results

| Exp ID | Configuration / Hypothesis | Candidate Recall | Avg Cands/S1 | Macro F0.5 | Macro Prec | Macro Rec | Singleton Acc | Decision | Reason & Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0** | **Stage-5 Locked Baseline** (LightGBM 25 feats, thr=0.980) | **96.84%** | 75.06 | **90.42%** | 93.66% | 84.68% | 88.89% | **BASELINE** | Immutable reference benchmark |
| **E1** | **Performance Optimization** (Precomputed target primitives, fast array extraction) | 96.84% | 75.06 | 90.42% | 93.66% | 84.68% | 88.89% | **KEEP** | 100% numerical parity with 1.72x feature speedup |
| **E2** | **+ Multi-View TF-IDF Candidate Retrieval** (Name + Addr + Char n-gram top-30 sparse union) | **99.11%** | 125.97 | 84.57% | 86.47% | 84.82% | 73.50% | **KEEP** | Recovers +156 additional true matches (candidate ceiling jumps to 99.11%) |
| **E4-E7** | **+ Rich Features** (Structured Address Components + Corpus Rarity + Retrieval Evidence) | 99.11% | 125.97 | **95.96%** | 97.55% | 92.65% | 88.89% | **KEEP** | Macro F0.5 surges from 90.42% to 95.96% (ROC-AUC: 0.99967, PR-AUC: 0.99194) |
| **E9** | **+ Iterative Hard Negative Mining** | 99.11% | 125.97 | 95.73% | 98.08% | 90.53% | 94.87% | **REJECT** | Oversuppresses recall slightly on boundary non-singletons |
| **E10** | **+ Singleton Protection Guard** | 99.11% | 125.97 | 95.97% | 97.57% | 92.62% | 88.89% | **KEEP** | Filters ambiguous multi-candidate collisions below confidence floor |
| **E11** | **+ Global Target Consistency** (Exclusivity Conflict Resolution) | **99.11%** | 125.97 | **96.20%** | **97.62%** | **93.10%** | **94.02%** | **KEEP** | Resolves cross-S1 target assignment collisions; yields new all-time best Macro F0.5 |
| **E12** | **+ Model Ensemble** (LightGBM + HistGradientBoosting) | 99.11% | 125.97 | 95.88% | 98.18% | 90.76% | 94.02% | **REJECT** | Single LightGBM + Global Consistency outperforms ensemble (96.20% vs 95.88%) and runs faster |

---

## Key Milestone Decisions:
1. **Multi-View Retrieval (E2):** Adding sparse TF-IDF (word on name, word on address, and char 3-gram on name) lifted candidate recall from **96.84% to 99.11%**, capturing 6,815 out of 6,876 true matches.
2. **Rich Feature Engineering (E4-E7):** Adding structured address components (house number match/conflict, postal code match/conflict, digits overlap/conflict) + corpus rarity log frequencies + retrieval consensus count boosted classifier precision to 97.55% and Macro F0.5 to 95.96%.
3. **Global Target Consistency (E11):** Post-processing target assignment conflicts where multiple S1 entities claim the same Target ID improved Macro Precision to **97.62%** and overall Macro F0.5 to **96.20%**.
4. **Final Model Choice:** Single LightGBM with 37 features at threshold `0.960` + Global Target Consistency is selected as the winning architecture (V1.0 Final).
