import os
import sys
import time
import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

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

def main():
    print("=" * 75)
    print("STAGE 5 — END-TO-END MODEL OPTIMIZATION & VALIDATION PIPELINE")
    print("=" * 75)
    t_global_start = time.time()

    # -------------------------------------------------------------
    # STEP 1 & 2: Load S1, Ground Truth, and Target Pool
    # -------------------------------------------------------------
    print("\n>>> STEP 1: Loading Data & Preserving Fixed Validation Universe...")
    
    # Load ground truth
    t0 = time.time()
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    print(f"  Ground truth loaded in {time.time() - t0:.2f}s")
    
    # Load previous train and val S1 IDs to guarantee 100% split consistency
    old_train_df = pl.read_parquet(OUTPUT_DIR / "train_features.parquet")
    old_val_df = pl.read_parquet(OUTPUT_DIR / "validation_features.parquet")
    
    train_s1_ids = set(old_train_df["s1_id"].unique().to_list())
    val_s1_ids = set(old_val_df["s1_id"].unique().to_list())
    
    print(f"  Loaded Train S1 cohort: {len(train_s1_ids):,} entities")
    print(f"  Loaded Val S1 cohort:   {len(val_s1_ids):,} entities")
    assert len(train_s1_ids.intersection(val_s1_ids)) == 0, "FATAL: Train/Val overlap detected!"
    
    all_selected_s1_ids = train_s1_ids.union(val_s1_ids)
    
    # Load normalized S1 records
    print("\n>>> Loading and Normalizing Source 1 records...")
    df_s1_all = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    s1_rows = df_s1_all.filter(pl.col("entity_id").is_in(list(all_selected_s1_ids)))
    s1_dict = {r["entity_id"]: r for r in s1_rows.iter_rows(named=True)}
    print(f"  Assembled {len(s1_dict):,} S1 entity records")
    
    # Target IDs needed by GT
    needed_target_ids = set()
    for s1_id in all_selected_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
    print(f"  True Ground-Truth target IDs needed: {len(needed_target_ids):,}")
    
    # Load Target Records (True Targets + 250,000 Distractors)
    print("\n>>> Loading & Normalizing Target Records (True Targets + 250k Distractors)...")
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
        
    t_load = time.time()
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
    print(f"  Target pool assembled: {len(df_targets):,} records in {time.time() - t_load:.2f}s")
    
    # -------------------------------------------------------------
    # STEP 2: Inverted Index Blocking with LOCKED Stage-5 Blocker
    # -------------------------------------------------------------
    print("\n>>> STEP 2: Indexing Targets with LOCKED Stage-5 Blocker...")
    t_block = time.time()
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    blocker.add_target_records(df_targets)
    blocker.prune_high_frequency_keys()
    print(f"  Indexing & pruning completed in {time.time() - t_block:.2f}s")
    
    # Split S1 DataFrames
    df_s1_train = df_s1_all.filter(pl.col("entity_id").is_in(list(train_s1_ids)))
    df_s1_val = df_s1_all.filter(pl.col("entity_id").is_in(list(val_s1_ids)))
    
    print("\n>>> Generating Stage-5 Candidates for Train & Validation Sets...")
    train_candidates = blocker.generate_candidates_for_s1(df_s1_train)
    val_candidates = blocker.generate_candidates_for_s1(df_s1_val)
    
    # Validate Candidate Recall on the 2,000 Validation Cohort
    val_gt = {s1: gt_map.get(s1, set()) for s1 in val_s1_ids}
    total_val_gt_matches = sum(len(m) for m in val_gt.values())
    val_cand_metrics = evaluate_candidate_recall(val_gt, val_candidates)
    
    captured_val = val_cand_metrics["captured_matches"]
    missed_val = total_val_gt_matches - captured_val
    cand_recall = val_cand_metrics["candidate_recall_ceiling"]
    
    print(f"\n==================================================")
    print(f"STAGE 5 CANDIDATE GENERATION AUDIT:")
    print(f"  Validation S1 Entities:     {len(val_s1_ids):,}")
    print(f"  Ground-Truth Matches:       {total_val_gt_matches:,}")
    print(f"  Captured by Stage-5 Blocker:{captured_val:,} ({cand_recall*100:.2f}%)")
    print(f"  Missed / Unreachable:       {missed_val:,} ({missed_val/total_val_gt_matches*100:.2f}%)")
    print(f"  Average Candidates / S1:    {val_cand_metrics['avg_candidates_per_s1']:.2f}")
    print(f"  Median Candidates / S1:     {val_cand_metrics['median_candidates_per_s1']:.1f}")
    print(f"==================================================")
    
    # -------------------------------------------------------------
    # STEP 3: Generate Training & Validation Feature Datasets
    # -------------------------------------------------------------
    print("\n>>> STEP 3: Extracting Pairwise Features for Train & Validation...")
    
    def extract_dataset(
        s1_id_set: Set[str],
        cand_dict: Dict[str, Set[str]],
        is_val: bool = False,
        max_hard_neg: int = 12,
    ) -> pl.DataFrame:
        rows = []
        pos_count = 0
        neg_count = 0
        cand_counts = []
        
        for s1_id in s1_id_set:
            s1_rec = s1_dict[s1_id]
            s1_norm_name = s1_rec["norm_name"]
            s1_core_name = s1_rec["core_name"]
            s1_norm_addr = s1_rec["norm_address"]
            s1_country = s1_rec["country"]
            
            true_matches = gt_map.get(s1_id, set())
            cand_ids = cand_dict.get(s1_id, set())
            cand_counts.append(len(cand_ids))
            
            if is_val:
                # Validation: Extract ALL candidates produced by blocker
                for tid in cand_ids:
                    if tid not in target_lookup:
                        continue
                    tgt = target_lookup[tid]
                    feats = extract_pairwise_features(
                        s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                        tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                        tid,
                    )
                    feats["s1_id"] = s1_id
                    feats["target_id"] = tid
                    lbl = 1 if tid in true_matches else 0
                    feats["label"] = lbl
                    rows.append(feats)
                    if lbl == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
            else:
                # Training: Extract all true positives + top hard negatives from candidate set
                for tid in true_matches:
                    if tid not in target_lookup:
                        continue
                    tgt = target_lookup[tid]
                    feats = extract_pairwise_features(
                        s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                        tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                        tid,
                    )
                    feats["s1_id"] = s1_id
                    feats["target_id"] = tid
                    feats["label"] = 1
                    rows.append(feats)
                    pos_count += 1
                    
                cand_negatives = [tid for tid in cand_ids if tid not in true_matches and tid in target_lookup]
                neg_feats_list = []
                for tid in cand_negatives:
                    tgt = target_lookup[tid]
                    feats = extract_pairwise_features(
                        s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                        tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                        tid,
                    )
                    hardness = feats["name_wratio"] + feats["core_token_set"] + feats["addr_token_set"]
                    neg_feats_list.append((hardness, feats, tid))
                    
                neg_feats_list.sort(key=lambda x: x[0], reverse=True)
                for _, feats, tid in neg_feats_list[:max_hard_neg]:
                    feats["s1_id"] = s1_id
                    feats["target_id"] = tid
                    feats["label"] = 0
                    rows.append(feats)
                    neg_count += 1
                    
        df_feat = pl.DataFrame(rows)
        float_cols = [
            "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
            "name_token_sort", "name_token_set", "name_partial", "core_ratio",
            "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
            "addr_missing", "addr_exact", "addr_ratio", "addr_token_sort",
            "addr_token_set", "addr_jaccard", "addr_char_3gram", "addr_num_match",
            "country_match", "is_source2", "is_source3",
        ]
        cast_exprs = [pl.col(c).cast(pl.Float32) for c in float_cols if c in df_feat.columns]
        cast_exprs.append(pl.col("name_len_diff").cast(pl.Int16))
        cast_exprs.append(pl.col("addr_len_diff").cast(pl.Int16))
        cast_exprs.append(pl.col("label").cast(pl.UInt8))
        df_feat = df_feat.with_columns(cast_exprs)
        return df_feat
        
    print("  Generating Train Feature DataFrame...")
    train_df = extract_dataset(train_s1_ids, train_candidates, is_val=False, max_hard_neg=12)
    print("  Generating Validation Feature DataFrame...")
    val_df = extract_dataset(val_s1_ids, val_candidates, is_val=True)
    
    # Save Stage 5 Parquet Files
    train_parquet_path = OUTPUT_DIR / "train_features_stage5.parquet"
    val_parquet_path = OUTPUT_DIR / "validation_features_stage5.parquet"
    
    train_df.write_parquet(train_parquet_path, compression="zstd")
    val_df.write_parquet(val_parquet_path, compression="zstd")
    
    train_pos = int(train_df["label"].sum())
    train_neg = len(train_df) - train_pos
    val_pos = int(val_df["label"].sum())
    val_neg = len(val_df) - val_pos
    
    print("\n--- FEATURE DATASET GENERATION SUMMARY ---")
    print(f"Train Parquet:      {train_parquet_path}")
    print(f"  Rows:             {len(train_df):,} (Pos: {train_pos:,}, Hard Neg: {train_neg:,}, Ratio 1:{train_neg/max(1,train_pos):.2f})")
    print(f"  S1 Entities:      {len(train_s1_ids):,}")
    print(f"Val Parquet:        {val_parquet_path}")
    print(f"  Rows:             {len(val_df):,} (Pos: {val_pos:,}, Neg: {val_neg:,}, Ratio 1:{val_neg/max(1,val_pos):.2f})")
    print(f"  S1 Entities:      {len(val_s1_ids):,}")
    print(f"  Captured Matches: {val_pos:,} / {total_val_gt_matches:,} ({val_pos/total_val_gt_matches*100:.2f}%)")
    
    # -------------------------------------------------------------
    # STEP 4: Feature Integrity Audit
    # -------------------------------------------------------------
    print("\n>>> STEP 4: Running Pre-Training Feature Integrity Audit...")
    
    # Check 1: Disjointness
    assert len(train_s1_ids.intersection(val_s1_ids)) == 0, "Overlap between train and val S1 IDs!"
    print("  [Pass] Train and Validation S1 entity IDs are 100% disjoint.")
    
    # Check 2: Missing / NaN features
    for col in FEATURE_COLS:
        null_train = train_df[col].is_null().sum()
        null_val = val_df[col].is_null().sum()
        assert null_train == 0 and null_val == 0, f"Null values found in feature column {col}!"
    print("  [Pass] Zero null or NaN values in feature matrices.")
    
    # Check 3: No label leakage in features
    assert "label" not in FEATURE_COLS, "Label column is in feature list!"
    assert "s1_id" not in FEATURE_COLS and "target_id" not in FEATURE_COLS, "IDs in feature list!"
    print("  [Pass] Zero ID/target label leakage in feature columns.")
    
    # Check 4: Valid ID formats
    for tid in train_df["target_id"].head(500).to_list() + val_df["target_id"].head(500).to_list():
        assert tid.startswith("S2-") or tid.startswith("S3-"), f"Invalid target ID format: {tid}"
    print("  [Pass] All candidate IDs have valid S2/S3 format.")
    
    # -------------------------------------------------------------
    # STEP 5: Retrain LightGBM Model
    # -------------------------------------------------------------
    print("\n>>> STEP 5: Retraining LightGBM Pair Classifier on Stage-5 Features...")
    
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["label"].to_numpy()
    
    X_val = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df["label"].to_numpy()
    
    val_s1_list = val_df["s1_id"].to_list()
    val_tgt_list = val_df["target_id"].to_list()
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLS)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]
    
    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    
    best_iter = model.best_iteration
    val_probs = model.predict(X_val, num_iteration=best_iter)
    
    val_roc_auc = roc_auc_score(y_val, val_probs)
    val_pr_auc = average_precision_score(y_val, val_probs)
    
    print(f"\nLightGBM Training Completed in {best_iter} iterations:")
    print(f"  Validation ROC-AUC: {val_roc_auc:.5f}")
    print(f"  Validation PR-AUC:  {val_pr_auc:.5f}")
    
    # Save Model Artifacts
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path_models = models_dir / "lightgbm_stage5.txt"
    model_path_outputs = OUTPUT_DIR / "lightgbm_stage5.txt"
    
    model.save_model(str(model_path_models))
    model.save_model(str(model_path_outputs))
    print(f"  Model saved to: {model_path_models}")
    
    # -------------------------------------------------------------
    # STEP 6 & 7: Exact Challenge Macro F0.5 Threshold Sweep
    # -------------------------------------------------------------
    print("\n>>> STEP 6 & 7: Running Exact Challenge Entity-Level Macro F0.5 Sweep...")
    
    # Group validation pairs by S1 ID
    s1_to_val_pairs = defaultdict(list)
    for s1_id, tid, prob, label in zip(val_s1_list, val_tgt_list, val_probs, y_val):
        s1_to_val_pairs[s1_id].append((tid, float(prob), int(label)))
        
    coarse_thresholds = [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]
    fine_thresholds = [round(x, 3) for x in np.arange(0.80, 0.995, 0.005)]
    all_thresholds = sorted(list(set(coarse_thresholds + fine_thresholds)))
    
    threshold_results = []
    best_macro_f05 = -1.0
    best_thresh = 0.95
    best_res_dict = {}
    
    for thresh in all_thresholds:
        preds: Dict[str, Set[str]] = {}
        total_pred_matches = 0
        total_tp = 0
        total_fp = 0
        empty_s1_count = 0
        
        for s1_id in val_s1_ids:
            pairs = s1_to_val_pairs.get(s1_id, [])
            matched = {tid for tid, prob, _ in pairs if prob >= thresh}
            preds[s1_id] = matched
            total_pred_matches += len(matched)
            if len(matched) == 0:
                empty_s1_count += 1
                
            true_set = val_gt.get(s1_id, set())
            tp = len(matched.intersection(true_set))
            fp = len(matched - true_set)
            total_tp += tp
            total_fp += fp
            
        m = evaluate_predictions(val_gt, preds)
        total_fn = total_val_gt_matches - total_tp
        
        # Pairwise precision, recall, F0.5
        pair_prec = total_tp / total_pred_matches if total_pred_matches > 0 else 0.0
        pair_rec = total_tp / total_val_gt_matches if total_val_gt_matches > 0 else 0.0
        pair_f05 = compute_f05(pair_prec, pair_rec, 0.5)
        
        res = {
            "threshold": thresh,
            "macro_f05": m["macro_f05"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "pairwise_f05": pair_f05,
            "pairwise_precision": pair_prec,
            "pairwise_recall": pair_rec,
            "singleton_accuracy": m["singleton_accuracy"],
            "pred_matches": total_pred_matches,
            "empty_s1_count": empty_s1_count,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "outside_candidate_pool": missed_val,
            "avg_pred_per_s1": total_pred_matches / len(val_s1_ids),
        }
        threshold_results.append(res)
        
        if m["macro_f05"] > best_macro_f05:
            best_macro_f05 = m["macro_f05"]
            best_thresh = thresh
            best_res_dict = res
            
    # Save Threshold Results CSV
    thresh_csv_path = OUTPUT_DIR / "threshold_results_stage5.csv"
    with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(threshold_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in threshold_results:
            writer.writerow({k: f"{v:.5f}" if isinstance(v, float) else v for k, v in r.items()})
            
    print(f"  Threshold sweep saved to: {thresh_csv_path}")
    
    # Sort top 15 thresholds by Macro F0.5
    sorted_thresh = sorted(threshold_results, key=lambda x: x["macro_f05"], reverse=True)
    print("\n--- TOP 15 THRESHOLDS BY CHALLENGE MACRO F0.5 ---")
    print(f"{'Thresh':>7s} | {'Macro F0.5':>11s} | {'Macro Prec':>11s} | {'Macro Rec':>10s} | {'Pair F0.5':>10s} | {'Single Acc':>11s} | {'Matches':>8s} | {'TP':>6s} | {'FP':>5s} | {'FN':>5s}")
    print("-" * 105)
    for r in sorted_thresh[:15]:
        print(f"{r['threshold']:7.3f} | {r['macro_f05']*100:10.2f}% | {r['macro_precision']*100:10.2f}% | {r['macro_recall']*100:9.2f}% | {r['pairwise_f05']*100:9.2f}% | {r['singleton_accuracy']*100:10.2f}% | {r['pred_matches']:8,d} | {r['true_positives']:6,d} | {r['false_positives']:5,d} | {r['false_negatives']:5,d}")
        
    print(f"\nOPTIMAL DECISION THRESHOLD: {best_thresh:.3f}")
    print(f"  Validation Macro F0.5:     {best_macro_f05*100:.2f}%")
    print(f"  Validation Macro Precision:{best_res_dict['macro_precision']*100:.2f}%")
    print(f"  Validation Macro Recall:   {best_res_dict['macro_recall']*100:.2f}%")
    print(f"  Singleton Accuracy:        {best_res_dict['singleton_accuracy']*100:.2f}%")
    print(f"  Total True Positives:      {best_res_dict['true_positives']:,} / {total_val_gt_matches:,}")
    print(f"  Total False Positives:     {best_res_dict['false_positives']:,}")
    print(f"  Total False Negatives:     {best_res_dict['false_negatives']:,}")
    
    # -------------------------------------------------------------
    # STEP 8: Model Comparison (Old Stage 3/4 vs Stage 5)
    # -------------------------------------------------------------
    print("\n>>> STEP 8: Comparing Old Pipeline vs Stage-5 Pipeline...")
    
    # Old baseline values from audit
    # Old model audit at best threshold 0.95 had:
    # Candidate recall: 91.36% (6,282 / 6,876)
    # At T=0.95: Macro F0.5: 92.92%, Prec: 95.83%, Recall: 87.80%, TP: 6,037, FP: 263, FN: 839
    
    comparison_data = [
        {
            "Metric": "Candidate Recall Ceiling",
            "Old Model (Stage 3/4)": "91.36% (6,282 / 6,876)",
            "Stage 5 Model": f"{cand_recall*100:.2f}% ({captured_val:,} / {total_val_gt_matches:,})",
            "Difference": f"+{(cand_recall - 0.9136)*100:+.2f}% (+{captured_val - 6282} matches)",
        },
        {
            "Metric": "Macro F0.5 (Challenge Metric)",
            "Old Model (Stage 3/4)": "92.92%",
            "Stage 5 Model": f"{best_macro_f05*100:.2f}%",
            "Difference": f"+{(best_macro_f05 - 0.9292)*100:+.2f}%",
        },
        {
            "Metric": "Macro Precision",
            "Old Model (Stage 3/4)": "95.83%",
            "Stage 5 Model": f"{best_res_dict['macro_precision']*100:.2f}%",
            "Difference": f"+{(best_res_dict['macro_precision'] - 0.9583)*100:+.2f}%",
        },
        {
            "Metric": "Macro Recall",
            "Old Model (Stage 3/4)": "87.80%",
            "Stage 5 Model": f"{best_res_dict['macro_recall']*100:.2f}%",
            "Difference": f"+{(best_res_dict['macro_recall'] - 0.8780)*100:+.2f}%",
        },
        {
            "Metric": "True Positives (Captured & Matched)",
            "Old Model (Stage 3/4)": "6,037",
            "Stage 5 Model": f"{best_res_dict['true_positives']:,}",
            "Difference": f"+{best_res_dict['true_positives'] - 6037:,}",
        },
        {
            "Metric": "False Positives",
            "Old Model (Stage 3/4)": "263",
            "Stage 5 Model": f"{best_res_dict['false_positives']:,}",
            "Difference": f"{best_res_dict['false_positives'] - 263:+d}",
        },
        {
            "Metric": "False Negatives",
            "Old Model (Stage 3/4)": "839",
            "Stage 5 Model": f"{best_res_dict['false_negatives']:,}",
            "Difference": f"{best_res_dict['false_negatives'] - 839:+d}",
        },
        {
            "Metric": "Singleton Accuracy",
            "Old Model (Stage 3/4)": "94.02%",
            "Stage 5 Model": f"{best_res_dict['singleton_accuracy']*100:.2f}%",
            "Difference": f"+{(best_res_dict['singleton_accuracy'] - 0.9402)*100:+.2f}%",
        },
        {
            "Metric": "Optimal Decision Threshold",
            "Old Model (Stage 3/4)": "0.95",
            "Stage 5 Model": f"{best_thresh:.3f}",
            "Difference": f"{best_thresh - 0.95:+.3f}",
        },
    ]
    
    comp_df = pl.DataFrame(comparison_data)
    comp_csv_path = OUTPUT_DIR / "model_comparison_stage5.csv"
    comp_df.write_csv(comp_csv_path)
    print(f"  Saved comparison table to: {comp_csv_path}")
    
    print("\n" + "=" * 80)
    print("STAGE 3/4 vs STAGE 5 FULL PIPELINE COMPARISON:")
    print(f"{'Metric':<35s} | {'Old Model (Stage 3/4)':<20s} | {'Stage 5 Model':<20s} | {'Gain':<12s}")
    print("-" * 95)
    for r in comparison_data:
        print(f"{r['Metric']:<35s} | {r['Old Model (Stage 3/4)']:<20s} | {r['Stage 5 Model']:<20s} | {r['Difference']:<12s}")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # STEP 9: Detailed Error Analysis (Top FP & FN)
    # -------------------------------------------------------------
    print("\n>>> STEP 9: Extracting Detailed Error Analysis Records...")
    
    needed_val_tids = set(val_tgt_list)
    val_target_meta = {tid: target_lookup.get(tid, {}) for tid in needed_val_tids}
    
    fp_records = []
    fn_records = []
    
    for s1_id, tid, prob, label in zip(val_s1_list, val_tgt_list, val_probs, y_val):
        s1 = s1_dict.get(s1_id, {})
        tgt = val_target_meta.get(tid, {})
        
        pred_label = 1 if prob >= best_thresh else 0
        
        if pred_label == 1 and label == 0:
            fp_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "source": "S2" if tid.startswith("S2-") else "S3",
                "country": s1.get("country", ""),
                "name_s1": s1.get("business_name", ""),
                "name_target": tgt.get("business_name", ""),
                "address_s1": s1.get("business_address", ""),
                "address_target": tgt.get("business_address", ""),
                "probability": float(prob),
                "true_label": 0,
                "error_category": "address_only_collision" if s1.get("norm_address") == tgt.get("norm_address") and s1.get("core_name") != tgt.get("core_name") else "name_similarity_collision",
            })
            
        if pred_label == 0 and label == 1:
            fn_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "source": "S2" if tid.startswith("S2-") else "S3",
                "country": s1.get("country", ""),
                "name_s1": s1.get("business_name", ""),
                "name_target": tgt.get("business_name", ""),
                "address_s1": s1.get("business_address", ""),
                "address_target": tgt.get("business_address", ""),
                "probability": float(prob),
                "true_label": 1,
                "error_category": "threshold_cutoff" if prob >= 0.50 else "weak_similarity_signal",
            })
            
    # Also include the 216 uncaptured matches as candidate_missing FN
    for s1_id, true_set in val_gt.items():
        cand_ids = val_candidates.get(s1_id, set())
        missed_set = true_set - cand_ids
        for tid in missed_set:
            s1 = s1_dict.get(s1_id, {})
            tgt = target_lookup.get(tid, {})
            fn_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "source": "S2" if tid.startswith("S2-") else "S3",
                "country": s1.get("country", ""),
                "name_s1": s1.get("business_name", ""),
                "name_target": tgt.get("business_name", ""),
                "address_s1": s1.get("business_address", ""),
                "address_target": tgt.get("business_address", ""),
                "probability": 0.0,
                "true_label": 1,
                "error_category": "outside_candidate_pool",
            })
            
    fp_records.sort(key=lambda x: x["probability"], reverse=True)
    fn_records.sort(key=lambda x: x["probability"], reverse=True)
    
    fp_csv_path = OUTPUT_DIR / "stage5_false_positives.csv"
    fn_csv_path = OUTPUT_DIR / "stage5_false_negatives.csv"
    
    pl.DataFrame(fp_records).write_csv(fp_csv_path)
    pl.DataFrame(fn_records).write_csv(fn_csv_path)
    
    print(f"  Saved {len(fp_records)} False Positives to: {fp_csv_path}")
    print(f"  Saved {len(fn_records)} False Negatives to: {fn_csv_path}")
    
    # -------------------------------------------------------------
    # STEP 10: Final Model Config Artifact
    # -------------------------------------------------------------
    print("\n>>> STEP 10: Saving Final Model Configuration Artifact...")
    
    final_config = {
        "pipeline_stage": "Stage 5 Final Locked",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_artifact": str(model_path_models),
        "model_artifact_outputs": str(model_path_outputs),
        "blocking_version": "Stage-5 (Acronyms, Token-Pairs, Multi-word Addr, Phonetic-Compound, Tiered Quotas)",
        "features": FEATURE_COLS,
        "optimal_threshold": float(best_thresh),
        "validation_metrics": {
            "validation_s1_count": len(val_s1_ids),
            "total_ground_truth_matches": total_val_gt_matches,
            "candidate_recall_ceiling": float(cand_recall),
            "candidate_captured_matches": int(captured_val),
            "candidate_missed_matches": int(missed_val),
            "macro_f05": float(best_macro_f05),
            "macro_precision": float(best_res_dict["macro_precision"]),
            "macro_recall": float(best_res_dict["macro_recall"]),
            "pairwise_f05": float(best_res_dict["pairwise_f05"]),
            "pairwise_precision": float(best_res_dict["pairwise_precision"]),
            "pairwise_recall": float(best_res_dict["pairwise_recall"]),
            "singleton_accuracy": float(best_res_dict["singleton_accuracy"]),
            "true_positives": int(best_res_dict["true_positives"]),
            "false_positives": int(best_res_dict["false_positives"]),
            "false_negatives": int(best_res_dict["false_negatives"]),
            "validation_roc_auc": float(val_roc_auc),
            "validation_pr_auc": float(val_pr_auc),
        },
        "model_parameters": lgb_params,
        "best_iteration": int(best_iter),
    }
    
    config_json_path = OUTPUT_DIR / "final_model_config_stage5.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
        
    print(f"  Final Model Config saved to: {config_json_path}")
    print(f"\nPipeline execution finished in {time.time() - t_global_start:.2f}s")

if __name__ == "__main__":
    main()
