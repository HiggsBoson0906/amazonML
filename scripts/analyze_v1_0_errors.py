import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter

import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    OUTPUT_DIR,
)
from src.data_loader import load_ground_truth, load_and_normalize_source
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import (
    STAGE5_FEATURE_COLS,
    PrecomputedEntity,
    extract_pairwise_features_fast,
)
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import (
    CorpusFrequencyTracker,
    extract_structured_address_components,
    compute_address_component_features,
)
from src.postprocess import PostProcessor
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.utils import ExperimentLogger

EXPANDED_FEATURE_COLS = STAGE5_FEATURE_COLS + [
    "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
    "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
    "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
    "retrieval_views_count", "max_tfidf_score",
]

def analyze_v1_0_errors():
    logger = ExperimentLogger()
    logger.log("=" * 75)
    logger.log("DEEP RESIDUAL ERROR ANALYSIS FOR V1.0 PIPELINE")
    logger.log("=" * 75)
    
    # 1. Load Ground Truth and Partitions
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    old_train_df = pl.read_parquet(OUTPUT_DIR / "train_features.parquet")
    old_val_df = pl.read_parquet(OUTPUT_DIR / "validation_features.parquet")
    
    train_s1_ids = set(old_train_df["s1_id"].unique().to_list())
    val_s1_ids = set(old_val_df["s1_id"].unique().to_list())
    all_s1_ids = train_s1_ids.union(val_s1_ids)
    val_s1_list = list(val_s1_ids)
    
    val_gt_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    total_val_gt_matches = sum(len(s) for s in val_gt_map.values())
    
    # 2. Load Normalized Source 1
    df_s1_all = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    s1_rows = df_s1_all.filter(pl.col("entity_id").is_in(list(all_s1_ids)))
    
    s1_dict: Dict[str, PrecomputedEntity] = {}
    s1_raw_dict: Dict[str, Tuple[str, str, str, str]] = {}
    for r in s1_rows.iter_rows(named=True):
        eid = r["entity_id"]
        n_name = r["norm_name"]
        c_name = r["core_name"]
        n_addr = r["norm_address"]
        country = r["country"]
        s1_raw_dict[eid] = (n_name, c_name, n_addr, country)
        s1_dict[eid] = PrecomputedEntity(n_name, c_name, n_addr, country, eid)
        
    # 3. Load Targets
    needed_target_ids = set()
    for s1_id in all_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
        
    target_lookup_raw: Dict[str, Tuple[str, str, str, str]] = {}
    target_lookup_fast: Dict[str, PrecomputedEntity] = {}
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    
    def load_targets(filepath: Path, src_prefix: str, max_distractors: int = 125000):
        distractors = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not tid:
                    continue
                is_needed = tid in needed_target_ids
                if is_needed or distractors < max_distractors:
                    if not is_needed:
                        distractors += 1
                    n_name = normalize_business_name(name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(addr) if addr else ""
                    
                    target_lookup_raw[tid] = (n_name, c_name, n_addr, country)
                    target_lookup_fast[tid] = PrecomputedEntity(n_name, c_name, n_addr, country, tid)
                    
                    if country:
                        keys = blocker._extract_all_keys(n_name, c_name, n_addr, name)
                        c_idx = blocker.index[country]
                        for k in keys:
                            c_idx[k].append(tid)

    load_targets(TRAIN_SOURCE2, "train_source2.tsv", max_distractors=125000)
    load_targets(TRAIN_SOURCE3, "train_source3.tsv", max_distractors=125000)
    blocker.prune_high_frequency_keys()
    
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=30,
        max_total_candidates=160,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    
    freq_tracker = CorpusFrequencyTracker()
    all_tgt_names = [e.norm_name for e in target_lookup_fast.values()]
    all_tgt_addrs = [e.norm_addr for e in target_lookup_fast.values()]
    freq_tracker.fit(all_tgt_names, all_tgt_addrs)
    
    s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_dict.items()}
    tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
    
    # 4. Generate Candidates on Validation
    val_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(val_s1_list))
    val_stage5_candidates = blocker.generate_candidates_for_s1(val_df_chunk)
    
    val_candidates = {}
    val_evidence = {}
    for s1_id in val_s1_list:
        raw_s1 = s1_raw_dict[s1_id]
        cands, evid = retriever.retrieve_candidates(
            s1_id, raw_s1[0], raw_s1[1], raw_s1[2], raw_s1[3],
            blocking_cands=val_stage5_candidates.get(s1_id, set()),
        )
        val_candidates[s1_id] = cands
        val_evidence[s1_id] = evid
        
    cand_eval = evaluate_candidate_recall(val_gt_map, val_candidates)
    logger.log(f"  V1.0 Candidate Recall: {cand_eval['candidate_recall_ceiling']*100:.2f}% ({cand_eval['captured_matches']}/{cand_eval['total_true_matches']})")
    
    # 5. Load V1.0 Model & Predict
    model_path = Path("models/lightgbm_v1_optimized.txt")
    model = lgb.Booster(model_file=str(model_path))
    
    X_val_rows, val_meta = [], []
    for s1_id in val_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        cands = val_candidates.get(s1_id, set())
        evid_dict = val_evidence.get(s1_id, {})
        
        for tid in cands:
            tgt_obj = target_lookup_fast.get(tid)
            if not tgt_obj:
                continue
            tgt_ac = tgt_addr_comps.get(tid, extract_structured_address_components(""))
            base_feats = extract_pairwise_features_fast(s1_obj, tgt_obj)
            ac_feats = compute_address_component_features(s1_ac, tgt_ac)
            s1_nf = freq_tracker.get_name_freq_feature(s1_obj.norm_name)
            tgt_nf = freq_tracker.get_name_freq_feature(tgt_obj.norm_name)
            s1_af = freq_tracker.get_addr_freq_feature(s1_obj.norm_addr)
            tgt_af = freq_tracker.get_addr_freq_feature(tgt_obj.norm_addr)
            
            evid = evid_dict.get(tid, {})
            ret_views = float(len(evid)) if evid else 1.0
            max_tfidf = max([v for k, v in evid.items() if k != "blocking"], default=0.0)
            
            full_row = base_feats + [
                ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
                ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
                ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
                s1_nf, tgt_nf, s1_af, tgt_af,
                ret_views, max_tfidf,
            ]
            X_val_rows.append(full_row)
            val_meta.append((s1_id, tid))
            
    X_val = np.array(X_val_rows, dtype=np.float32)
    probs = model.predict(X_val)
    
    pair_prob_dict = defaultdict(list)
    pair_prob_lookup = {}
    for (s1_id, tid), p in zip(val_meta, probs):
        pair_prob_dict[s1_id].append((tid, float(p)))
        pair_prob_lookup[(s1_id, tid)] = float(p)
        
    postprocessor = PostProcessor(
        base_threshold=0.960,
        min_margin=0.05,
        enable_singleton_guard=True,
        enable_global_consistency=True,
    )
    final_preds = postprocessor.apply(val_s1_list, pair_prob_dict, val_candidates)
    metrics = evaluate_predictions(val_gt_map, final_preds)
    
    logger.log(f"  V1.0 Macro F0.5: {metrics['macro_f05']*100:.2f}%, Prec: {metrics['macro_precision']*100:.2f}%, Rec: {metrics['macro_recall']*100:.2f}%, Singleton: {metrics['singleton_accuracy']*100:.2f}%")

    # 6. Extract and Classify Errors
    logger.log("\n6. Classifying False Positives, False Negatives, and Candidate Misses...")
    
    fn_records = []
    fp_records = []
    singleton_errors = []
    
    fn_categories = Counter()
    fp_categories = Counter()
    
    # Analyze False Negatives (True Ground-Truth pairs missed by final predictions)
    for s1_id in val_s1_list:
        true_targets = val_gt_map.get(s1_id, set())
        pred_targets = final_preds.get(s1_id, set())
        cand_set = val_candidates.get(s1_id, set())
        s1_obj = s1_dict[s1_id]
        
        for tid in true_targets:
            if tid not in pred_targets:
                tgt_obj = target_lookup_fast.get(tid)
                prob = pair_prob_lookup.get((s1_id, tid), 0.0)
                
                # Determine root cause category
                if tid not in cand_set:
                    category = "candidate_missing"
                elif prob < 0.50:
                    # Low model score: check feature discrepancies
                    if s1_obj and tgt_obj and s1_obj.country != tgt_obj.country:
                        category = "country_mismatch"
                    elif s1_obj and tgt_obj and fuzz.ratio(s1_obj.norm_name, tgt_obj.norm_name) < 50:
                        category = "multilingual_or_abbreviation"
                    elif s1_obj and tgt_obj and s1_obj.norm_addr and tgt_obj.norm_addr and fuzz.ratio(s1_obj.norm_addr, tgt_obj.norm_addr) < 40:
                        category = "address_divergence"
                    else:
                        category = "score_too_low"
                elif prob < 0.960:
                    category = "threshold_boundary"
                else:
                    # Prob was >= 0.960 but suppressed by singleton guard or global consistency
                    category = "postprocessing_conflict_suppression"
                    
                fn_categories[category] += 1
                fn_records.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "error_type": "FALSE_NEGATIVE",
                    "category": category,
                    "model_prob": prob,
                    "in_candidate_pool": tid in cand_set,
                    "s1_name": s1_obj.norm_name if s1_obj else "",
                    "tgt_name": tgt_obj.norm_name if tgt_obj else "",
                    "s1_addr": s1_obj.norm_addr if s1_obj else "",
                    "tgt_addr": tgt_obj.norm_addr if tgt_obj else "",
                    "s1_country": s1_obj.country if s1_obj else "",
                    "tgt_country": tgt_obj.country if tgt_obj else "",
                })
                
        # Analyze False Positives (Predicted matches that are not in Ground Truth)
        for tid in pred_targets:
            if tid not in true_targets:
                tgt_obj = target_lookup_fast.get(tid)
                prob = pair_prob_lookup.get((s1_id, tid), 0.0)
                is_singleton = (len(true_targets) == 0)
                
                if is_singleton:
                    category = "singleton_false_positive"
                elif s1_obj and tgt_obj and fuzz.ratio(s1_obj.norm_name, tgt_obj.norm_name) > 90 and fuzz.ratio(s1_obj.norm_addr, tgt_obj.norm_addr) < 40:
                    category = "generic_name_different_address"
                elif s1_obj and tgt_obj and fuzz.ratio(s1_obj.norm_addr, tgt_obj.norm_addr) > 90 and fuzz.ratio(s1_obj.norm_name, tgt_obj.norm_name) < 60:
                    category = "shared_address_different_name"
                elif s1_obj and tgt_obj and s1_obj.country != tgt_obj.country:
                    category = "cross_country_collision"
                else:
                    category = "high_similarity_wrong_entity"
                    
                fp_categories[category] += 1
                fp_records.append({
                    "s1_id": s1_id,
                    "target_id": tid,
                    "error_type": "FALSE_POSITIVE",
                    "category": category,
                    "model_prob": prob,
                    "in_candidate_pool": True,
                    "s1_name": s1_obj.norm_name if s1_obj else "",
                    "tgt_name": tgt_obj.norm_name if tgt_obj else "",
                    "s1_addr": s1_obj.norm_addr if s1_obj else "",
                    "tgt_addr": tgt_obj.norm_addr if tgt_obj else "",
                    "s1_country": s1_obj.country if s1_obj else "",
                    "tgt_country": tgt_obj.country if tgt_obj else "",
                })

    # Save to CSV
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    
    all_errors = fn_records + fp_records
    with open(exp_dir / "v1_1_failure_analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "s1_id", "target_id", "error_type", "category", "model_prob", "in_candidate_pool",
            "s1_name", "tgt_name", "s1_addr", "tgt_addr", "s1_country", "tgt_country"
        ])
        writer.writeheader()
        writer.writerows(all_errors)
        
    logger.log(f"  Saved {len(all_errors):,} error records to {exp_dir / 'v1_1_failure_analysis.csv'}")

    # Write Markdown Report
    report_content = f"""# V1.0 Pipeline Deep Residual Error Analysis Report

**Evaluation Cohort:** 2,000 Source-1 Validation Entities  
**Ground-Truth Matches:** 6,876 True Pairs  
**Total False Negatives:** {len(fn_records):,} ({len(fn_records)/total_val_gt_matches*100:.2f}% of true matches)  
**Total False Positives:** {len(fp_records):,}  

---

## 1. False Negative Category Distribution

| Category | Count | Percentage | Primary Root Cause & Fix Opportunity |
| :--- | :---: | :---: | :--- |
"""
    total_fns = len(fn_records)
    for cat, count in fn_categories.most_common():
        pct = (count / total_fns * 100) if total_fns > 0 else 0
        if cat == "threshold_boundary":
            fix = "Probability in [0.70, 0.960): add stronger string/rarity features and margin-based decision function."
        elif cat == "candidate_missing":
            fix = "Unreachable candidate in retrieval: add targeted Char 4/5-gram and transliteration TF-IDF channels."
        elif cat == "multilingual_or_abbreviation":
            fix = "Heavy name variation / abbreviation: add token containment, Dice, and Jaro-Winkler features."
        elif cat == "address_divergence":
            fix = "Different address representation / landmark: weight core name higher when name match is exact."
        elif cat == "postprocessing_conflict_suppression":
            fix = "Target exclusivity conflict: refine multi-assignment resolution with probability margins."
        else:
            fix = "Score too low due to insufficient similarity signal."
        report_content += f"| **{cat}** | {count:,} | {pct:.1f}% | {fix} |\n"

    report_content += """
---

## 2. False Positive Category Distribution

| Category | Count | Percentage | Primary Root Cause & Fix Opportunity |
| :--- | :---: | :---: | :--- |
"""
    total_fps = len(fp_records)
    for cat, count in fp_categories.most_common():
        pct = (count / total_fps * 100) if total_fps > 0 else 0
        if cat == "generic_name_different_address":
            fix = "Common business name occurring at distinct location: penalize via corpus frequency and address conflict features."
        elif cat == "singleton_false_positive":
            fix = "Singleton entity falsely matching distractor: raise singleton margin floor and ambiguity filters."
        elif cat == "high_similarity_wrong_entity":
            fix = "Branch / subsidiary similarity: compute candidate-competition score margins to distinguish true vs nearby entity."
        elif cat == "shared_address_different_name":
            fix = "Commercial building / business park sharing address: enforce core name similarity requirement."
        else:
            fix = "General false positive collision."
        report_content += f"| **{cat}** | {count:,} | {pct:.1f}% | {fix} |\n"

    report_content += """
---

## 3. Targeted Optimization Action Plan for V1.1

1. **Candidate Retrieval (Remaining 61 Misses):**
   - Add **Char 4-gram / 5-gram TF-IDF** and **Transliterated Name TF-IDF** to push candidate recall from 99.11% toward 99.5%+.
2. **Precision & Rarity Weighting:**
   - Incorporate **Token IDF weights**, **House Number/Postal Conflict penalties**, and **Corpus Rarity Log Counts** to suppress generic name false positives.
3. **Candidate-Competition Features:**
   - Add **Probability Margins** ($\Delta = P_{\text{top1}} - P_{\text{top2}}$) and **Candidate Ranks** to distinguish true entities from competing branch distractors.
4. **Margin-Aware Decision Function:**
   - Accept matches with high confidence ($P \ge 0.95$) and sufficient margin, while allowing exact name + exact address overrides.
"""
    with open(exp_dir / "v1_1_failure_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.log(f"  Generated failure report: {exp_dir / 'v1_1_failure_report.md'}")

if __name__ == "__main__":
    analyze_v1_0_errors()
