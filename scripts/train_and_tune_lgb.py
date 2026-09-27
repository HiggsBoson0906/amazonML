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
from src.evaluate import evaluate_predictions, compute_f05

FEATURE_COLS = [
    "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
    "name_token_sort", "name_token_set", "name_partial", "core_ratio",
    "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
    "name_len_diff", "addr_missing", "addr_exact", "addr_ratio",
    "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_char_3gram",
    "addr_num_match", "addr_len_diff", "country_match", "is_source2", "is_source3",
]

def main():
    print("=" * 65)
    print("STAGE 3: LIGHTGBM TRAINING & THRESHOLD OPTIMIZATION")
    print("=" * 65)
    
    t0 = time.time()
    
    # 1. Load Parquet Data
    train_parquet_path = OUTPUT_DIR / "train_features.parquet"
    val_parquet_path = OUTPUT_DIR / "validation_features.parquet"
    
    print(f"\n1. Loading feature datasets:")
    print(f"   Train: {train_parquet_path}")
    print(f"   Val:   {val_parquet_path}")
    
    train_df = pl.read_parquet(train_parquet_path)
    val_df = pl.read_parquet(val_parquet_path)
    
    print(f"   Loaded Train shape: {train_df.shape} (Pos: {train_df['label'].sum():,}, Neg: {(train_df['label'] == 0).sum():,})")
    print(f"   Loaded Val shape:   {val_df.shape} (Pos: {val_df['label'].sum():,}, Neg: {(val_df['label'] == 0).sum():,})")
    
    X_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df["label"].to_numpy()
    
    X_val = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df["label"].to_numpy()
    
    val_s1_ids = val_df["s1_id"].to_list()
    val_tgt_ids = val_df["target_id"].to_list()
    
    # 2. Train LightGBM Model
    print("\n2. Training LightGBM Model...")
    
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
    
    best_iteration = model.best_iteration
    val_preds_prob = model.predict(X_val, num_iteration=best_iteration)
    
    val_roc_auc = roc_auc_score(y_val, val_preds_prob)
    val_pr_auc = average_precision_score(y_val, val_preds_prob)
    print(f"\nLightGBM Training Completed in {model.best_iteration} rounds:")
    print(f"  Validation ROC-AUC: {val_roc_auc:.5f}")
    print(f"  Validation PR-AUC:  {val_pr_auc:.5f}")
    
    # 3. Load Ground Truth for S1 validation entities
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    unique_val_s1 = set(val_s1_ids)
    eval_val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in unique_val_s1}
    
    # 4. Threshold Optimization for Macro F0.5
    print("\n3. Threshold Optimization for Macro F0.5...")
    
    thresholds = [round(x, 2) for x in np.arange(0.05, 1.00, 0.02)]
    threshold_results = []
    
    best_thresh = 0.5
    best_f05 = -1.0
    best_metrics = {}
    
    # Pre-structure validation pairs per S1
    s1_to_pairs = {}
    for s1_id, tid, prob, label in zip(val_s1_ids, val_tgt_ids, val_preds_prob, y_val):
        if s1_id not in s1_to_pairs:
            s1_to_pairs[s1_id] = []
        s1_to_pairs[s1_id].append((tid, prob, label))
        
    for thresh in thresholds:
        predictions: Dict[str, Set[str]] = {}
        total_pred_matches = 0
        total_fps = 0
        total_fns = 0
        
        for s1_id in unique_val_s1:
            pairs = s1_to_pairs.get(s1_id, [])
            matched = set()
            for tid, prob, label in pairs:
                if prob >= thresh:
                    matched.add(tid)
                    if label == 0:
                        total_fps += 1
                else:
                    if label == 1:
                        total_fns += 1
            predictions[s1_id] = matched
            total_pred_matches += len(matched)
            
        m = evaluate_predictions(eval_val_gt, predictions)
        m["threshold"] = thresh
        m["pred_matches"] = total_pred_matches
        m["false_positives"] = total_fps
        m["false_negatives"] = total_fns
        threshold_results.append(m)
        
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thresh = thresh
            best_metrics = m
            
    # Save Threshold Analysis CSV
    thresh_csv_path = OUTPUT_DIR / "threshold_results.csv"
    with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["threshold", "macro_f05", "macro_precision", "macro_recall", "singleton_accuracy", "pred_matches", "false_positives", "false_negatives"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in threshold_results:
            writer.writerow({
                "threshold": r["threshold"],
                "macro_f05": f"{r['macro_f05']:.5f}",
                "macro_precision": f"{r['macro_precision']:.5f}",
                "macro_recall": f"{r['macro_recall']:.5f}",
                "singleton_accuracy": f"{r['singleton_accuracy']:.5f}",
                "pred_matches": r["pred_matches"],
                "false_positives": r["false_positives"],
                "false_negatives": r["false_negatives"],
            })
            
    # Sort top 10 thresholds by F0.5
    sorted_thresh = sorted(threshold_results, key=lambda x: x["macro_f05"], reverse=True)
    print("\n--- TOP 10 THRESHOLDS BY MACRO F0.5 ---")
    print(f"{'Thresh':>8s} | {'Macro F0.5':>11s} | {'Precision':>10s} | {'Recall':>9s} | {'Singleton Acc':>14s} | {'Matches':>8s}")
    print("-" * 75)
    for r in sorted_thresh[:10]:
        print(f"{r['threshold']:8.2f} | {r['macro_f05']*100:10.2f}% | {r['macro_precision']*100:9.2f}% | {r['macro_recall']*100:8.2f}% | {r['singleton_accuracy']*100:13.2f}% | {r['pred_matches']:8,d}")
        
    print(f"\nOptimal Decision Threshold: {best_thresh:.2f} (Macro F0.5: {best_f05*100:.2f}%)")
    
    # 5. Deterministic Baseline vs LightGBM Comparison
    print("\n4. Comparative Evaluation on Validation Set (2,000 S1 Entities):")
    
    # Deterministic Predictions on Val Set
    det_predictions = {}
    for s1_id in unique_val_s1:
        matched = set()
        for idx in range(len(val_df)):
            if val_df["s1_id"][idx] != s1_id:
                continue
            tid = val_df["target_id"][idx]
            # Baseline rules
            ex_name = val_df["exact_core_name"][idx]
            c_set = val_df["core_token_set"][idx]
            c_sort = val_df["core_token_sort"][idx]
            n_ratio = val_df["name_ratio"][idx]
            a_exact = val_df["addr_exact"][idx]
            a_set = val_df["addr_token_set"][idx]
            a_jacc = val_df["addr_jaccard"][idx]
            a_miss = val_df["addr_missing"][idx]
            a_num = val_df["addr_num_match"][idx]
            
            if ex_name == 1.0 and (a_miss == 1.0 or a_exact == 1.0 or a_set >= 0.70 or a_jacc >= 0.35 or a_num == 1.0):
                matched.add(tid)
            elif (c_set >= 0.92 or c_sort >= 0.90) and (a_set >= 0.75 or a_jacc >= 0.40):
                matched.add(tid)
            elif n_ratio >= 0.85 and a_exact == 1.0:
                matched.add(tid)
            elif a_exact == 1.0 and c_set >= 0.60:
                matched.add(tid)
        det_predictions[s1_id] = matched
        
    det_metrics = evaluate_predictions(eval_val_gt, det_predictions)
    
    # Best LightGBM Predictions
    lgb_predictions = {}
    for s1_id in unique_val_s1:
        pairs = s1_to_pairs.get(s1_id, [])
        lgb_predictions[s1_id] = {tid for tid, prob, _ in pairs if prob >= best_thresh}
    lgb_best_metrics = evaluate_predictions(eval_val_gt, lgb_predictions)
    
    print("\n" + "=" * 65)
    print("BASELINE vs LIGHTGBM COMPARISON:")
    print(f"{'Metric':<22s} | {'Deterministic Baseline':<22s} | {'LightGBM (T=' + str(best_thresh) + ')':<20s} | {'Gain':<10s}")
    print("-" * 80)
    print(f"{'Macro F0.5 Score':<22s} | {det_metrics['macro_f05']*100:20.2f}% | {lgb_best_metrics['macro_f05']*100:18.2f}% | +{(lgb_best_metrics['macro_f05'] - det_metrics['macro_f05'])*100:5.2f}%")
    print(f"{'Macro Precision':<22s} | {det_metrics['macro_precision']*100:20.2f}% | {lgb_best_metrics['macro_precision']*100:18.2f}% | +{(lgb_best_metrics['macro_precision'] - det_metrics['macro_precision'])*100:5.2f}%")
    print(f"{'Macro Recall':<22s} | {det_metrics['macro_recall']*100:20.2f}% | {lgb_best_metrics['macro_recall']*100:18.2f}% | +{(lgb_best_metrics['macro_recall'] - det_metrics['macro_recall'])*100:5.2f}%")
    print(f"{'Singleton Accuracy':<22s} | {det_metrics['singleton_accuracy']*100:20.2f}% | {lgb_best_metrics['singleton_accuracy']*100:18.2f}% | +{(lgb_best_metrics['singleton_accuracy'] - det_metrics['singleton_accuracy'])*100:5.2f}%")
    print("=" * 65)
    
    # 6. Subgroup Performance (Source 2 vs 3, US vs India, Singletons vs Non-Singletons)
    print("\n5. Subgroup Breakdown Performance (LightGBM):")
    
    # Load S1 records metadata
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    s1_meta = {r["entity_id"]: r for r in df_s1.iter_rows(named=True)}
    
    # By Country
    for country in ["US", "India"]:
        c_s1s = [s1 for s1 in unique_val_s1 if s1_meta.get(s1, {}).get("country") == country]
        if c_s1s:
            c_gt = {s1: eval_val_gt[s1] for s1 in c_s1s}
            c_preds = {s1: lgb_predictions[s1] for s1 in c_s1s}
            c_res = evaluate_predictions(c_gt, c_preds)
            print(f"  Country [{country:5s} ({len(c_s1s):,d} S1s)]: F0.5={c_res['macro_f05']*100:.2f}%, Prec={c_res['macro_precision']*100:.2f}%, Rec={c_res['macro_recall']*100:.2f}%")
            
    # By Singleton vs Non-Singleton
    singletons = [s1 for s1 in unique_val_s1 if len(eval_val_gt[s1]) == 0]
    non_singletons = [s1 for s1 in unique_val_s1 if len(eval_val_gt[s1]) > 0]
    
    s_gt = {s1: eval_val_gt[s1] for s1 in singletons}
    s_preds = {s1: lgb_predictions[s1] for s1 in singletons}
    s_res = evaluate_predictions(s_gt, s_preds)
    print(f"  Singletons     ({len(singletons):,d} S1s): F0.5={s_res['macro_f05']*100:.2f}%, Acc={s_res['singleton_accuracy']*100:.2f}%")
    
    ns_gt = {s1: eval_val_gt[s1] for s1 in non_singletons}
    ns_preds = {s1: lgb_predictions[s1] for s1 in non_singletons}
    ns_res = evaluate_predictions(ns_gt, ns_preds)
    print(f"  Non-Singletons ({len(non_singletons):,d} S1s): F0.5={ns_res['macro_f05']*100:.2f}%, Prec={ns_res['macro_precision']*100:.2f}%, Rec={ns_res['macro_recall']*100:.2f}%")
    
    # By Target Source (S2 vs S3 pairwise precision/recall)
    for src, prefix in [("Source 2", "S2-"), ("Source 3", "S3-")]:
        src_tp = sum(1 for s1 in unique_val_s1 for tid in lgb_predictions[s1] if tid.startswith(prefix) and tid in eval_val_gt[s1])
        src_pred = sum(1 for s1 in unique_val_s1 for tid in lgb_predictions[s1] if tid.startswith(prefix))
        src_true = sum(1 for s1 in unique_val_s1 for tid in eval_val_gt[s1] if tid.startswith(prefix))
        src_prec = src_tp / src_pred if src_pred > 0 else 0.0
        src_rec = src_tp / src_true if src_true > 0 else 0.0
        src_f05 = compute_f05(src_prec, src_rec, 0.5)
        print(f"  Target [{src:8s}]: Pairwise F0.5={src_f05*100:.2f}%, Prec={src_prec*100:.2f}% ({src_tp}/{src_pred}), Rec={src_rec*100:.2f}% ({src_tp}/{src_true})")
        
    # 7. Feature Importance
    print("\n6. Feature Importance:")
    gain_imp = model.feature_importance(importance_type="gain")
    split_imp = model.feature_importance(importance_type="split")
    
    feat_imp_df = pl.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": gain_imp,
        "importance_split": split_imp,
    }).sort("importance_gain", descending=True)
    
    feat_imp_csv = OUTPUT_DIR / "feature_importance.csv"
    feat_imp_df.write_csv(feat_imp_csv)
    print(f"  Saved Feature Importance to: {feat_imp_csv}")
    print("\n  Top 10 Features by Gain:")
    for row in feat_imp_df.head(10).iter_rows(named=True):
        print(f"    {row['feature']:<20s} | Gain: {row['importance_gain']:10.1f} | Split: {row['importance_split']:5d}")
        
    # 8. Error Analysis: Save False Positives & False Negatives
    print("\n7. Extracting Error Analysis Records (Top FP & FN)...")
    
    # Load target metadata for error analysis
    needed_tids = set(val_tgt_ids)
    target_meta = {}
    def load_t_meta(path):
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == 4 and parts[0] in needed_tids:
                    target_meta[parts[0]] = {"business_name": parts[1], "business_address": parts[2], "country": parts[3]}
    load_t_meta(TRAIN_SOURCE2)
    load_t_meta(TRAIN_SOURCE3)
    
    fp_records = []
    fn_records = []
    
    for s1_id, tid, prob, label in zip(val_s1_ids, val_tgt_ids, val_preds_prob, y_val):
        s1 = s1_meta.get(s1_id, {})
        tgt = target_meta.get(tid, {})
        
        pred_label = 1 if prob >= best_thresh else 0
        
        # False Positive
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
            })
            
        # False Negative
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
            })
            
    # Sort FPs by confidence (highest prob first)
    fp_records.sort(key=lambda x: x["probability"], reverse=True)
    top20_fp = fp_records[:20]
    
    # Sort FNs: 20 highest prob FNs (near threshold) + 20 lowest prob FNs (deep misses)
    fn_records.sort(key=lambda x: x["probability"], reverse=True)
    top20_fn_high_prob = fn_records[:20]
    top20_fn_low_prob = sorted(fn_records, key=lambda x: x["probability"])[:20]
    
    fp_csv_path = OUTPUT_DIR / "false_positives.csv"
    fn_csv_path = OUTPUT_DIR / "false_negatives.csv"
    
    pl.DataFrame(top20_fp).write_csv(fp_csv_path)
    pl.DataFrame(top20_fn_high_prob + top20_fn_low_prob).write_csv(fn_csv_path)
    
    print(f"  Saved False Positives to: {fp_csv_path} ({len(top20_fp)} records)")
    print(f"  Saved False Negatives to: {fn_csv_path} ({len(top20_fn_high_prob) + len(top20_fn_low_prob)} records)")
    
    # 9. Save Model Artifact and Metrics
    model_txt_path = OUTPUT_DIR / "lightgbm_entity_resolution.txt"
    model.save_model(str(model_txt_path))
    model_size_kb = os.path.getsize(model_txt_path) / 1024
    print(f"\n8. Saved Trained Model to: {model_txt_path} ({model_size_kb:.1f} KB)")
    
    metrics_json_path = OUTPUT_DIR / "model_metrics.json"
    metrics_data = {
        "model_type": "LightGBM GBDT",
        "best_iteration": int(best_iteration),
        "optimal_threshold": float(best_thresh),
        "validation_macro_f05": float(lgb_best_metrics["macro_f05"]),
        "validation_macro_precision": float(lgb_best_metrics["macro_precision"]),
        "validation_macro_recall": float(lgb_best_metrics["macro_recall"]),
        "validation_singleton_accuracy": float(lgb_best_metrics["singleton_accuracy"]),
        "validation_roc_auc": float(val_roc_auc),
        "validation_pr_auc": float(val_pr_auc),
        "baseline_macro_f05": float(det_metrics["macro_f05"]),
        "f05_improvement": float(lgb_best_metrics["macro_f05"] - det_metrics["macro_f05"]),
        "model_size_kb": float(model_size_kb),
        "training_time_seconds": float(time.time() - t0),
    }
    
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"  Saved Model Metrics to: {metrics_json_path}")
    print(f"\nStage 3 Total Runtime: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
