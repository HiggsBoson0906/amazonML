import os
import sys
import time
from pathlib import Path
from typing import Dict, Set
import polars as pl

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

def run_deterministic_baseline(sample_size: int = 20000):
    print("=" * 60)
    print(f"RUNNING DETERMINISTIC BASELINE ON SAMPLE OF {sample_size:,} S1 ENTITIES")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load Ground Truth
    print("\n1. Loading Ground Truth...")
    full_gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    print(f"Total Ground Truth entries loaded: {len(full_gt):,}")
    
    # 2. Load Source 1 Sample
    print(f"\n2. Loading and normalizing {sample_size:,} Source 1 records...")
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=sample_size)
    s1_ids = set(df_s1["entity_id"].to_list())
    eval_gt = {s1_id: full_gt.get(s1_id, set()) for s1_id in s1_ids}
    
    # 3. Load Target Sources (S2 and S3)
    # To ensure full candidate recall on the sample, we can load a proportionate chunk or full target
    target_sample_size = sample_size * 5
    print(f"\n3. Loading and normalizing {target_sample_size:,} target records from S2 and S3...")
    df_s2 = load_and_normalize_source(TRAIN_SOURCE2, n_rows=target_sample_size)
    df_s3 = load_and_normalize_source(TRAIN_SOURCE3, n_rows=target_sample_size)
    
    # 4. Build Blocker Index
    print("\n4. Building Inverted Index Blocker...")
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=80)
    blocker.add_target_records(df_s2)
    blocker.add_target_records(df_s3)
    blocker.prune_high_frequency_keys()
    
    # 5. Generate Candidates
    print("5. Generating candidates for S1 entities...")
    candidates = blocker.generate_candidates_for_s1(df_s1)
    
    # 6. Evaluate Candidate Recall
    cand_metrics = evaluate_candidate_recall(eval_gt, candidates)
    print("\n--- CANDIDATE GENERATION (BLOCKING) METRICS ---")
    print(f"Candidate Recall Ceiling: {cand_metrics['candidate_recall_ceiling']*100:.2f}%")
    print(f"Total True Matches: {cand_metrics['total_true_matches']:,}")
    print(f"Captured Matches: {cand_metrics['captured_matches']:,}")
    print(f"Avg Candidates per S1: {cand_metrics['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates per S1: {cand_metrics['median_candidates_per_s1']:.1f}")
    
    # 7. Fast Index lookup for Target Features
    print("\n7. Running Deterministic Matcher...")
    
    # Combine targets into lookup dict for fast feature calculation
    target_lookup = {}
    for row in df_s2.iter_rows(named=True):
        target_lookup[row["entity_id"]] = (
            row["norm_name"],
            row["core_name"],
            row["norm_address"],
            row["country"],
        )
    for row in df_s3.iter_rows(named=True):
        target_lookup[row["entity_id"]] = (
            row["norm_name"],
            row["core_name"],
            row["norm_address"],
            row["country"],
        )
        
    s1_rows = df_s1.iter_rows(named=True)
    predictions: Dict[str, Set[str]] = {}
    
    for row in s1_rows:
        s1_id = row["entity_id"]
        s1_norm_name = row["norm_name"]
        s1_core_name = row["core_name"]
        s1_norm_addr = row["norm_address"]
        s1_country = row["country"]
        
        cand_ids = candidates.get(s1_id, set())
        matched_set = set()
        
        for tgt_id in cand_ids:
            if tgt_id not in target_lookup:
                continue
            tgt_norm_name, tgt_core_name, tgt_norm_addr, tgt_country = target_lookup[tgt_id]
            
            # Country must match
            if s1_country != tgt_country:
                continue
                
            feats = extract_pairwise_features(
                s1_norm_name,
                s1_core_name,
                s1_norm_addr,
                s1_country,
                tgt_norm_name,
                tgt_core_name,
                tgt_norm_addr,
                tgt_country,
                tgt_id,
            )
            
            # Deterministic Matching Heuristics:
            # Rule 1: Exact core name match + reasonable address agreement
            if feats["exact_core_name"] == 1.0:
                if feats["addr_missing"] == 1.0 or feats["addr_exact"] == 1.0 or feats["addr_token_set"] >= 0.70 or feats["addr_jaccard"] >= 0.40 or feats["addr_num_match"] == 1.0:
                    matched_set.add(tgt_id)
                    continue
                    
            # Rule 2: Very high core token set / sort ratio + strong address agreement
            if (feats["core_token_set"] >= 0.92 or feats["core_token_sort"] >= 0.90) and (feats["addr_token_set"] >= 0.75 or feats["addr_jaccard"] >= 0.45):
                matched_set.add(tgt_id)
                continue
                
            # Rule 3: High name ratio + exact address match
            if feats["name_ratio"] >= 0.85 and feats["addr_exact"] == 1.0:
                matched_set.add(tgt_id)
                continue
                
        predictions[s1_id] = matched_set
        
    # 8. Evaluate Predictions
    eval_results = evaluate_predictions(eval_gt, predictions)
    print("\n" + "=" * 60)
    print("--- DETERMINISTIC BASELINE RESULTS ---")
    print(f"Macro F0.5 Score:      {eval_results['macro_f05']*100:.2f}%")
    print(f"Macro Precision:       {eval_results['macro_precision']*100:.2f}%")
    print(f"Macro Recall:          {eval_results['macro_recall']*100:.2f}%")
    print(f"Singleton Accuracy:    {eval_results['singleton_accuracy']*100:.2f}%")
    print(f"Total Evaluated:       {eval_results['total_evaluated']:,}")
    print("=" * 60)
    print(f"Total Baseline Pipeline Time: {time.time() - t0:.2f}s")
    
if __name__ == "__main__":
    run_deterministic_baseline(sample_size=20000)
