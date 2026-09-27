import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
import numpy as np
import lightgbm as lgb

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
from src.data_loader import load_ground_truth, load_and_normalize_source
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features
from src.evaluate import evaluate_predictions, evaluate_candidate_recall, compute_f05

FEATURE_COLS = [
    "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
    "name_token_sort", "name_token_set", "name_partial", "core_ratio",
    "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
    "name_len_diff", "addr_missing", "addr_exact", "addr_ratio",
    "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_char_3gram",
    "addr_num_match", "addr_len_diff", "country_match", "is_source2", "is_source3",
]

def run_integrity_audit():
    print("=" * 70)
    print("VALIDATION INTEGRITY AUDIT")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load Parquet Data to get exact S1 IDs for Train & Val
    train_parquet_path = OUTPUT_DIR / "train_features.parquet"
    val_parquet_path = OUTPUT_DIR / "validation_features.parquet"
    
    df_train_feats = pl.read_parquet(train_parquet_path)
    df_val_feats = pl.read_parquet(val_parquet_path)
    
    train_s1_ids = set(df_train_feats["s1_id"].unique().to_list())
    val_s1_ids = set(df_val_feats["s1_id"].unique().to_list())
    
    print(f"\n[Check 1] Train/Val S1 Entity Disjointness:")
    intersection = train_s1_ids.intersection(val_s1_ids)
    print(f"  Train S1 Count:       {len(train_s1_ids):,}")
    print(f"  Val S1 Count:         {len(val_s1_ids):,}")
    print(f"  Overlapping S1 IDs:   {len(intersection)} (Disjoint: {len(intersection) == 0})")
    
    # 2. Load Ground Truth for ALL validation S1 entities
    print(f"\n[Check 2] Ground Truth Match Universe for Validation S1s:")
    full_gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    val_gt = {s1_id: full_gt.get(s1_id, set()) for s1_id in val_s1_ids}
    
    total_val_s1 = len(val_gt)
    total_true_match_pairs = sum(len(matches) for matches in val_gt.values())
    val_singletons = [s1 for s1, m in val_gt.items() if len(m) == 0]
    val_non_singletons = [s1 for s1, m in val_gt.items() if len(m) > 0]
    
    print(f"  Total Validation S1 Entities:     {total_val_s1:,}")
    print(f"  Total True Match Pairs in GT:     {total_true_match_pairs:,}")
    print(f"  Singleton S1 Entities (0 matches): {len(val_singletons):,} ({len(val_singletons)/total_val_s1*100:.2f}%)")
    print(f"  Non-Singleton S1 Entities:        {len(val_non_singletons):,} ({len(val_non_singletons)/total_val_s1*100:.2f}%)")
    
    # 3. Load Target Records (True Targets + 250k Distractors)
    print(f"\n[Check 3] Target Records Pool & Blocker Candidate Generation:")
    needed_target_ids = set()
    for s1_id in train_s1_ids.union(val_s1_ids):
        needed_target_ids.update(full_gt.get(s1_id, set()))
        
    def extract_targets(path, needed_ids, max_distractors=125000):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            distractors = 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if tid in needed_ids:
                    rows.append((tid, name, addr, country))
                elif distractors < max_distractors:
                    rows.append((tid, name, addr, country))
                    distractors += 1
        return rows
        
    s2_rows = extract_targets(TRAIN_SOURCE2, needed_target_ids, max_distractors=125000)
    s3_rows = extract_targets(TRAIN_SOURCE3, needed_target_ids, max_distractors=125000)
    
    df_targets = pl.DataFrame(
        s2_rows + s3_rows,
        schema=["entity_id", "business_name", "business_address", "country"],
        orient="row",
    )
    
    names = df_targets["business_name"].to_list()
    addrs = df_targets["business_address"].to_list()
    norm_names = [normalize_business_name(n) for n in names]
    core_names = [extract_core_business_name(n) for n in norm_names]
    norm_addrs = [normalize_business_address(a) for a in addrs]
    
    df_targets = df_targets.with_columns(
        pl.Series("norm_name", norm_names, dtype=pl.Utf8),
        pl.Series("core_name", core_names, dtype=pl.Utf8),
        pl.Series("norm_address", norm_addrs, dtype=pl.Utf8),
    )
    
    target_lookup = {r["entity_id"]: r for r in df_targets.iter_rows(named=True)}
    print(f"  Target pool assembled: {len(df_targets):,} records")
    
    # Load S1 validation normalized records
    df_s1_all = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    val_s1_df = df_s1_all.filter(pl.col("entity_id").is_in(list(val_s1_ids)))
    val_s1_records = {r["entity_id"]: r for r in val_s1_df.iter_rows(named=True)}
    
    # 4. Generate Candidates STRICTLY using Blocker
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    blocker.add_target_records(df_targets)
    blocker.prune_high_frequency_keys()
    
    candidates = blocker.generate_candidates_for_s1(val_s1_df)
    
    # Compute True Candidate Recall
    cand_metrics = evaluate_candidate_recall(val_gt, candidates)
    true_pairs_in_candidates = cand_metrics["captured_matches"]
    true_pairs_absent = total_true_match_pairs - true_pairs_in_candidates
    candidate_recall = true_pairs_in_candidates / total_true_match_pairs if total_true_match_pairs > 0 else 0.0
    
    print(f"  Candidate Recall Ceiling:       {candidate_recall*100:.2f}% ({true_pairs_in_candidates:,} / {total_true_match_pairs:,})")
    print(f"  True Pairs Absent from Candidates: {true_pairs_absent:,} ({true_pairs_absent/total_true_match_pairs*100:.2f}%)")
    print(f"  Average Candidates per S1:       {cand_metrics['avg_candidates_per_s1']:.2f}")
    
    # 5. Score ONLY generated candidate pairs with LightGBM
    print(f"\n[Check 4] Scoring Generated Candidates with Trained LightGBM Model:")
    model_txt_path = OUTPUT_DIR / "lightgbm_entity_resolution.txt"
    model = lgb.Booster(model_file=str(model_txt_path))
    
    # Extract features for all generated candidates
    cand_pair_rows = []
    cand_pair_meta = []
    
    for s1_id in val_s1_ids:
        s1_rec = val_s1_records[s1_id]
        s1_norm_name = s1_rec["norm_name"]
        s1_core_name = s1_rec["core_name"]
        s1_norm_addr = s1_rec["norm_address"]
        s1_country = s1_rec["country"]
        
        cand_ids = candidates.get(s1_id, set())
        for tid in cand_ids:
            if tid not in target_lookup:
                continue
            tgt = target_lookup[tid]
            feats = extract_pairwise_features(
                s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                tid,
            )
            cand_pair_rows.append([feats[c] for c in FEATURE_COLS])
            cand_pair_meta.append((s1_id, tid))
            
    print(f"  Total Candidate Pairs to score: {len(cand_pair_rows):,}")
    
    X_cand = np.array(cand_pair_rows, dtype=np.float32)
    cand_probs = model.predict(X_cand)
    
    # Structure predictions per S1
    s1_cand_predictions = {s1_id: [] for s1_id in val_s1_ids}
    for (s1_id, tid), prob in zip(cand_pair_meta, cand_probs):
        s1_cand_predictions[s1_id].append((tid, prob))
        
    # 6. Evaluate End-to-End Metrics across Thresholds
    print(f"\n[Check 5] End-to-End Evaluation across Decision Thresholds:")
    
    threshold_audit_rows = []
    best_e2e_f05 = -1.0
    best_e2e_thresh = 0.77
    best_e2e_metrics = {}
    
    for thresh in [round(x, 2) for x in np.arange(0.05, 1.00, 0.02)]:
        final_preds: Dict[str, Set[str]] = {}
        total_tp = 0
        total_fp = 0
        
        for s1_id in val_s1_ids:
            pairs = s1_cand_predictions.get(s1_id, [])
            matched = {tid for tid, prob in pairs if prob >= thresh}
            final_preds[s1_id] = matched
            
            true_set = val_gt.get(s1_id, set())
            tp = len(matched.intersection(true_set))
            fp = len(matched - true_set)
            total_tp += tp
            total_fp += fp
            
        m = evaluate_predictions(val_gt, final_preds)
        total_fn = total_true_match_pairs - total_tp
        
        row_data = {
            "threshold": thresh,
            "end_to_end_macro_f05": m["macro_f05"],
            "end_to_end_macro_precision": m["macro_precision"],
            "end_to_end_macro_recall": m["macro_recall"],
            "candidate_recall": candidate_recall,
            "singleton_accuracy": m["singleton_accuracy"],
            "final_true_positives": total_tp,
            "final_false_positives": total_fp,
            "final_false_negatives": total_fn,
            "true_pairs_outside_candidates": true_pairs_absent,
            "total_predicted_matches": sum(len(v) for v in final_preds.values()),
        }
        threshold_audit_rows.append(row_data)
        
        if m["macro_f05"] > best_e2e_f05:
            best_e2e_f05 = m["macro_f05"]
            best_e2e_thresh = thresh
            best_e2e_metrics = row_data
            
    # Print Top Thresholds
    sorted_audit = sorted(threshold_audit_rows, key=lambda x: x["end_to_end_macro_f05"], reverse=True)
    print("\n--- TOP 10 THRESHOLDS (END-TO-END PIPELINE) ---")
    print(f"{'Thresh':>8s} | {'E2E Macro F0.5':>15s} | {'Precision':>10s} | {'Recall':>9s} | {'Singleton Acc':>14s} | {'TP':>6s} | {'FP':>6s} | {'FN':>6s}")
    print("-" * 90)
    for r in sorted_audit[:10]:
        print(f"{r['threshold']:8.2f} | {r['end_to_end_macro_f05']*100:14.2f}% | {r['end_to_end_macro_precision']*100:9.2f}% | {r['end_to_end_macro_recall']*100:8.2f}% | {r['singleton_accuracy']*100:13.2f}% | {r['final_true_positives']:6,d} | {r['final_false_positives']:6,d} | {r['final_false_negatives']:6,d}")
        
    # Also evaluate threshold 0.77 explicitly
    r_77 = next(r for r in threshold_audit_rows if abs(r["threshold"] - 0.77) < 1e-4)
    
    print("\n" + "=" * 70)
    print("EXACT METRIC AUDIT AT THRESHOLD 0.77:")
    print(f"  Candidate Recall Ceiling:               {r_77['candidate_recall']*100:.2f}%")
    print(f"  True Match Pairs Outside Candidates:    {r_77['true_pairs_outside_candidates']:,}")
    print(f"  Final True Positives (TP):              {r_77['final_true_positives']:,}")
    print(f"  Final False Positives (FP):             {r_77['final_false_positives']:,}")
    print(f"  Final False Negatives (FN):             {r_77['final_false_negatives']:,}")
    print(f"  Final End-to-End Macro Precision:       {r_77['end_to_end_macro_precision']*100:.2f}%")
    print(f"  Final End-to-End Macro Recall:          {r_77['end_to_end_macro_recall']*100:.2f}%")
    print(f"  Final End-to-End Macro F0.5 Score:      {r_77['end_to_end_macro_f05']*100:.2f}%")
    print(f"  Singleton Accuracy:                     {r_77['singleton_accuracy']*100:.2f}%")
    print("=" * 70)
    
    # 7. Save Audit Artifacts
    audit_json_path = OUTPUT_DIR / "validation_integrity_audit.json"
    audit_csv_path = OUTPUT_DIR / "validation_integrity_audit.csv"
    
    audit_report = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_val_s1_disjoint": bool(len(intersection) == 0),
        "validation_s1_count": total_val_s1,
        "total_ground_truth_pairs": total_true_match_pairs,
        "true_pairs_in_candidate_pool": true_pairs_in_candidates,
        "true_pairs_absent_from_candidate_pool": true_pairs_absent,
        "candidate_recall_ceiling": float(candidate_recall),
        "threshold_0_77_results": {
            "end_to_end_macro_f05": float(r_77["end_to_end_macro_f05"]),
            "end_to_end_macro_precision": float(r_77["end_to_end_macro_precision"]),
            "end_to_end_macro_recall": float(r_77["end_to_end_macro_recall"]),
            "singleton_accuracy": float(r_77["singleton_accuracy"]),
            "final_true_positives": int(r_77["final_true_positives"]),
            "final_false_positives": int(r_77["final_false_positives"]),
            "final_false_negatives": int(r_77["final_false_negatives"]),
        },
        "optimal_end_to_end_threshold": float(best_e2e_thresh),
        "optimal_end_to_end_macro_f05": float(best_e2e_f05),
        "interpretation": {
            "reported_98_46_recall_was": "B) recall ONLY over ground-truth pairs that were present in validation_features.parquet (model classifier recall conditional on candidate presence).",
            "true_end_to_end_recall_is": f"{r_77['end_to_end_macro_recall']*100:.2f}% (bounded by candidate recall ceiling {candidate_recall*100:.2f}%).",
            "reported_f05_was": "Macro-average F0.5 computed per S1 entity, but evaluated on pairs present in validation_features.parquet.",
            "true_end_to_end_macro_f05_is": f"{r_77['end_to_end_macro_f05']*100:.2f}%."
        }
    }
    
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
        
    pl.DataFrame(threshold_audit_rows).write_csv(audit_csv_path)
    
    print(f"\nSaved Integrity Audit JSON to: {audit_json_path}")
    print(f"Saved Threshold Audit CSV to:  {audit_csv_path}")
    print(f"Audit completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_integrity_audit()
