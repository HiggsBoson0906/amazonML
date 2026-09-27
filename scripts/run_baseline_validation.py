import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set
import polars as pl
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    OUTPUT_DIR,
)
from src.data_loader import load_and_normalize_source, load_ground_truth
from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features
from src.evaluate import evaluate_predictions, evaluate_candidate_recall

def run_evaluation_benchmark(n_s1_sample: int = 5000):
    print("=" * 65)
    print(f"STAGE: DETERMINISTIC BASELINE VALIDATION ({n_s1_sample:,} S1 ENTITIES)")
    print("=" * 65)
    
    t_start = time.time()
    
    # 1. Load Ground Truth
    print("\n[Step 1] Loading Ground Truth...")
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    # 2. Sample S1 entities
    print(f"[Step 2] Loading & Normalizing {n_s1_sample:,} Source 1 records...")
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=n_s1_sample)
    s1_dict = {row["entity_id"]: row for row in df_s1.iter_rows(named=True)}
    eval_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in s1_dict.keys()}
    
    needed_target_ids = set()
    for t_ids in eval_gt.values():
        needed_target_ids.update(t_ids)
        
    print(f"  Loaded {len(s1_dict):,} S1 entities ({len(needed_target_ids):,} true target IDs)")
    
    # 3. Stream & Load S2/S3 Target Records (True targets + 200k background distractor records)
    print(f"\n[Step 3] Loading & Normalizing Target Records (True Targets + 200,000 Distractors)...")
    target_records = {}
    
    def extract_targets(path, needed_ids, max_distractors=100000):
        found = 0
        distractors = 0
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline() # skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if tid in needed_ids:
                    rows.append((tid, name, addr, country))
                    found += 1
                elif distractors < max_distractors:
                    rows.append((tid, name, addr, country))
                    distractors += 1
        return rows, found
        
    t_load = time.time()
    s2_rows, s2_found = extract_targets(TRAIN_SOURCE2, needed_target_ids, max_distractors=100000)
    s3_rows, s3_found = extract_targets(TRAIN_SOURCE3, needed_target_ids, max_distractors=100000)
    
    all_target_rows = s2_rows + s3_rows
    df_targets = pl.DataFrame(
        all_target_rows,
        schema=["entity_id", "business_name", "business_address", "country"],
        orient="row",
    )
    
    # Normalize targets
    names = df_targets["business_name"].to_list()
    addrs = df_targets["business_address"].to_list()
    from src.normalize import normalize_business_name, extract_core_business_name, normalize_business_address
    
    norm_names = [normalize_business_name(n) for n in names]
    core_names = [extract_core_business_name(n) for n in norm_names]
    norm_addrs = [normalize_business_address(a) for a in addrs]
    
    df_targets = df_targets.with_columns(
        pl.Series("norm_name", norm_names, dtype=pl.Utf8),
        pl.Series("core_name", core_names, dtype=pl.Utf8),
        pl.Series("norm_address", norm_addrs, dtype=pl.Utf8),
    )
    print(f"  Target pool assembled: {len(df_targets):,} records (Loaded in {time.time() - t_load:.2f}s)")
    
    # 4. Inverted Index Blocking
    print("\n[Step 4] Building Multi-Pass Inverted Index Blocker...")
    t_block = time.time()
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=120, max_per_key=25)
    blocker.add_target_records(df_targets)
    blocker.prune_high_frequency_keys()
    
    print("  Generating candidates for S1 entities...")
    candidates = blocker.generate_candidates_for_s1(df_s1)
    
    # Evaluate Blocking Recall and Candidate Distribution
    cand_counts = [len(c) for c in candidates.values()]
    p95_cands = float(np.percentile(cand_counts, 95)) if cand_counts else 0.0
    max_cands = int(np.max(cand_counts)) if cand_counts else 0
    
    cand_metrics = evaluate_candidate_recall(eval_gt, candidates)
    print("\n" + "-" * 55)
    print("ENHANCED BLOCKING & CANDIDATE GENERATION METRICS:")
    print(f"  Candidate Recall Ceiling: {cand_metrics['candidate_recall_ceiling']*100:.2f}%")
    print(f"  Total True Matches:       {cand_metrics['total_true_matches']:,}")
    print(f"  Captured Matches:         {cand_metrics['captured_matches']:,}")
    print(f"  Average Candidates / S1:  {cand_metrics['avg_candidates_per_s1']:.2f}")
    print(f"  P95 Candidates / S1:      {p95_cands:.1f}")
    print(f"  Max Candidates / S1:      {max_cands}")
    print(f"  Median Candidates / S1:   {cand_metrics['median_candidates_per_s1']:.1f}")
    print(f"  Blocking Time:            {time.time() - t_block:.2f}s")
    print("-" * 55)
    
    # 5. Deterministic Matching
    print("\n[Step 5] Applying Deterministic Matching Rules...")
    t_match = time.time()
    
    target_lookup = {
        row["entity_id"]: row
        for row in df_targets.iter_rows(named=True)
    }
    
    predictions: Dict[str, Set[str]] = {}
    
    for s1_id, s1_rec in s1_dict.items():
        s1_norm_name = s1_rec["norm_name"]
        s1_core_name = s1_rec["core_name"]
        s1_norm_addr = s1_rec["norm_address"]
        s1_country = s1_rec["country"]
        
        cand_ids = candidates.get(s1_id, set())
        matched_set = set()
        
        for tid in cand_ids:
            if tid not in target_lookup:
                continue
            tgt = target_lookup[tid]
            if s1_country != tgt["country"]:
                continue
                
            feats = extract_pairwise_features(
                s1_norm_name,
                s1_core_name,
                s1_norm_addr,
                s1_country,
                tgt["norm_name"],
                tgt["core_name"],
                tgt["norm_address"],
                tgt["country"],
                tid,
            )
            
            # High-Precision Matching Rules (Targeting Macro F0.5):
            # Rule 1: Exact core name + address match / missing address / strong token jaccard
            if feats["exact_core_name"] == 1.0:
                if feats["addr_missing"] == 1.0 or feats["addr_exact"] == 1.0 or feats["addr_token_set"] >= 0.70 or feats["addr_jaccard"] >= 0.35 or feats["addr_num_match"] == 1.0:
                    matched_set.add(tid)
                    continue
                    
            # Rule 2: Very high core token sort / set ratio >= 0.90 + address agreement
            if (feats["core_token_set"] >= 0.92 or feats["core_token_sort"] >= 0.90) and (feats["addr_token_set"] >= 0.75 or feats["addr_jaccard"] >= 0.40):
                matched_set.add(tid)
                continue
                
            # Rule 3: High name ratio >= 0.85 + exact address match
            if feats["name_ratio"] >= 0.85 and feats["addr_exact"] == 1.0:
                matched_set.add(tid)
                continue
                
            # Rule 4: Distinctive compound address match + reasonable core name similarity >= 0.60
            if feats["addr_exact"] == 1.0 and feats["core_token_set"] >= 0.60:
                matched_set.add(tid)
                continue
                
        predictions[s1_id] = matched_set
        
    print(f"  Matching completed in {time.time() - t_match:.2f}s")
    
    # 6. Evaluate Predictions
    eval_results = evaluate_predictions(eval_gt, predictions)
    print("\n" + "=" * 65)
    print("FINAL DETERMINISTIC BASELINE RESULTS:")
    print(f"  Macro F0.5 Score:      {eval_results['macro_f05']*100:.2f}%")
    print(f"  Macro Precision:       {eval_results['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:          {eval_results['macro_recall']*100:.2f}%")
    print(f"  Singleton Accuracy:    {eval_results['singleton_accuracy']*100:.2f}%")
    print(f"  Total S1 Evaluated:    {eval_results['total_evaluated']:,}")
    print(f"  Total Runtime:         {time.time() - t_start:.2f}s")
    print("=" * 65)
    
    # Print sample False Positives and False Negatives for inspection
    fp_examples = []
    fn_examples = []
    
    for s1_id, true_set in eval_gt.items():
        pred_set = predictions.get(s1_id, set())
        fps = pred_set - true_set
        fns = true_set - pred_set
        
        for tid in fps:
            if len(fp_examples) < 3 and tid in target_lookup:
                fp_examples.append((s1_dict[s1_id], target_lookup[tid]))
        for tid in fns:
            if len(fn_examples) < 3 and tid in target_lookup:
                fn_examples.append((s1_dict[s1_id], target_lookup[tid]))
                
    if fp_examples:
        print("\n--- SAMPLE FALSE POSITIVES (Predicted Match, but Ground Truth says No) ---")
        for idx, (s1, tgt) in enumerate(fp_examples, 1):
            print(f"FP #{idx}:")
            print(f"  S1:  [{s1['entity_id']}] '{s1['business_name']}' | Addr: '{s1['business_address']}'")
            print(f"  Tgt: [{tgt['entity_id']}] '{tgt['business_name']}' | Addr: '{tgt['business_address']}'")
            
    if fn_examples:
        print("\n--- SAMPLE FALSE NEGATIVES (Ground Truth Match, but Model Missed) ---")
        for idx, (s1, tgt) in enumerate(fn_examples, 1):
            print(f"FN #{idx}:")
            print(f"  S1:  [{s1['entity_id']}] '{s1['business_name']}' | Addr: '{s1['business_address']}'")
            print(f"  Tgt: [{tgt['entity_id']}] '{tgt['business_name']}' | Addr: '{tgt['business_address']}'")

if __name__ == "__main__":
    run_evaluation_benchmark(n_s1_sample=5000)
