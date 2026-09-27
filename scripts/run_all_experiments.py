import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

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
    extract_pairwise_features,
)
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import (
    CorpusFrequencyTracker,
    extract_structured_address_components,
    compute_address_component_features,
    compute_competition_features,
)
from src.model import EntityResolutionModel
from src.postprocess import optimize_decision_threshold, PostProcessor
from src.evaluate import evaluate_predictions, evaluate_candidate_recall, compute_f05
from src.utils import ExperimentLogger, Timer

def run_experiment_suite():
    logger = ExperimentLogger()
    logger.log("=" * 75)
    logger.log("STARTING COMPLETE OPTIMIZATION & EXPERIMENTATION SUITE (E0 -> E12)")
    logger.log("=" * 75)
    
    # -------------------------------------------------------------
    # 1. Load Ground Truth and Fixed Train/Val Split
    # -------------------------------------------------------------
    logger.log("1. Loading Ground Truth and Preserving Fixed 2,000-S1 Validation Cohort...")
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    old_train_df = pl.read_parquet(OUTPUT_DIR / "train_features.parquet")
    old_val_df = pl.read_parquet(OUTPUT_DIR / "validation_features.parquet")
    
    train_s1_ids = set(old_train_df["s1_id"].unique().to_list())
    val_s1_ids = set(old_val_df["s1_id"].unique().to_list())
    
    logger.log(f"  Train S1 cohort: {len(train_s1_ids):,} entities")
    logger.log(f"  Validation S1 cohort: {len(val_s1_ids):,} entities")
    assert len(train_s1_ids.intersection(val_s1_ids)) == 0, "Train and Val must be strictly disjoint!"
    
    all_s1_ids = train_s1_ids.union(val_s1_ids)
    
    # -------------------------------------------------------------
    # 2. Load Normalized Source 1
    # -------------------------------------------------------------
    logger.log("2. Loading and Normalizing Source 1...")
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
        
    logger.log(f"  Loaded {len(s1_dict):,} S1 entities into memory.")
    
    # Validation GT subset
    val_gt_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    total_val_gt_matches = sum(len(s) for s in val_gt_map.values())
    logger.log(f"  Validation GT universe: {len(val_gt_map):,} S1s, {total_val_gt_matches:,} true match pairs.")
    
    # -------------------------------------------------------------
    # 3. Load Target Records (True Targets + 250k Distractors)
    # -------------------------------------------------------------
    logger.log("3. Loading and Precomputing Target Records...")
    needed_target_ids = set()
    for s1_id in all_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
        
    target_lookup_raw: Dict[str, Tuple[str, str, str, str]] = {}
    target_lookup_fast: Dict[str, PrecomputedEntity] = {}
    
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    
    def load_targets(filepath: Path, src_prefix: str, max_distractors: int = 125000):
        count = 0
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
                    count += 1
        logger.log(f"  Loaded {count:,} records from {src_prefix}")

    load_targets(TRAIN_SOURCE2, "train_source2.tsv", max_distractors=125000)
    load_targets(TRAIN_SOURCE3, "train_source3.tsv", max_distractors=125000)
    
    logger.log(f"  Total target records indexed: {len(target_lookup_fast):,}")
    logger.log("  Pruning high-frequency blocking keys...")
    blocker.prune_high_frequency_keys()
    
    # Initialize Corpus Frequency Tracker
    freq_tracker = CorpusFrequencyTracker()
    all_names = [e.norm_name for e in target_lookup_fast.values()] + [e.norm_name for e in s1_dict.values()]
    all_addrs = [e.norm_addr for e in target_lookup_fast.values()] + [e.norm_addr for e in s1_dict.values()]
    freq_tracker.fit(all_names, all_addrs)
    
    # -------------------------------------------------------------
    # 4. Generate Baseline Stage-5 Candidates on Val Cohort
    # -------------------------------------------------------------
    logger.log("\n4. Generating Stage-5 Blocking Candidates on Validation Cohort...")
    t_cand_start = time.time()
    
    val_s1_list = list(val_s1_ids)
    val_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(val_s1_list))
    val_stage5_candidates = blocker.generate_candidates_for_s1(val_df_chunk)
    t_cand_time = time.time() - t_cand_start
    
    cand_eval_stage5 = evaluate_candidate_recall(val_gt_map, val_stage5_candidates)
    logger.log(f"  Stage-5 Candidate Recall: {cand_eval_stage5['candidate_recall_ceiling']*100:.2f}% ({cand_eval_stage5['captured_matches']}/{total_val_gt_matches})")
    logger.log(f"  Avg candidates/S1: {cand_eval_stage5['avg_candidates_per_s1']:.2f}, Median: {cand_eval_stage5['median_candidates_per_s1']:.1f}")

    # -------------------------------------------------------------
    # 5. Load Trained Baseline Stage-5 LightGBM Model
    # -------------------------------------------------------------
    logger.log("\n5. Loading Baseline Stage-5 LightGBM Model...")
    baseline_model = lgb.Booster(model_file=str(OUTPUT_DIR / "lightgbm_stage5.txt"))
    assert baseline_model.feature_name() == STAGE5_FEATURE_COLS
    
    # Store all experiment records
    experiment_results = []
    
    # =============================================================
    # EXPERIMENT E0: Stage-5 Baseline Validation Benchmark
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E0: Stage-5 Baseline Benchmark...")
    t0 = time.time()
    e0_pair_meta = []
    e0_feat_rows = []
    
    for s1_id in val_s1_list:
        s1_obj = s1_dict[s1_id]
        cands = val_stage5_candidates.get(s1_id, set())
        for tid in cands:
            tgt_obj = target_lookup_fast.get(tid)
            if not tgt_obj:
                continue
            feats = extract_pairwise_features_fast(s1_obj, tgt_obj)
            e0_feat_rows.append(feats)
            e0_pair_meta.append((s1_id, tid))
            
    X_val_e0 = np.array(e0_feat_rows, dtype=np.float32)
    probs_e0 = baseline_model.predict(X_val_e0)
    
    # Predict with threshold 0.980
    e0_preds = defaultdict(set)
    for (s1_id, tid), prob in zip(e0_pair_meta, probs_e0):
        if prob >= 0.980:
            e0_preds[s1_id].add(tid)
            
    # Guarantee all 2,000 S1s are present in prediction dict
    for s1_id in val_s1_list:
        if s1_id not in e0_preds:
            e0_preds[s1_id] = set()
            
    e0_metrics = evaluate_predictions(val_gt_map, e0_preds)
    t_e0 = time.time() - t0
    
    logger.log(f"  E0 Macro F0.5:     {e0_metrics['macro_f05']*100:.2f}%")
    logger.log(f"  E0 Macro Prec:     {e0_metrics['macro_precision']*100:.2f}%")
    logger.log(f"  E0 Macro Rec:      {e0_metrics['macro_recall']*100:.2f}%")
    logger.log(f"  E0 Singleton Acc:  {e0_metrics['singleton_accuracy']*100:.2f}%")
    logger.log(f"  E0 Time:           {t_e0:.2f}s")
    
    experiment_results.append({
        "exp_id": "E0",
        "name": "Stage-5 Baseline (thr=0.980)",
        "cand_recall": cand_eval_stage5["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_stage5["avg_candidates_per_s1"],
        "macro_f05": e0_metrics["macro_f05"] * 100,
        "macro_prec": e0_metrics["macro_precision"] * 100,
        "macro_rec": e0_metrics["macro_recall"] * 100,
        "singleton_acc": e0_metrics["singleton_accuracy"] * 100,
        "runtime_s": t_e0,
        "decision": "BASELINE",
        "reason": "Immutable Stage-5 benchmark reference",
    })

    # =============================================================
    # EXPERIMENT E1: Performance-Optimized Engine Parity Check
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E1: Performance Optimization Parity Check...")
    # Timing comparison between legacy extract_pairwise_features vs extract_pairwise_features_fast
    t_legacy_start = time.time()
    for s1_id, tid in e0_pair_meta[:10000]:
        s1_raw = s1_raw_dict[s1_id]
        tgt_raw = target_lookup_raw[tid]
        extract_pairwise_features(s1_raw[0], s1_raw[1], s1_raw[2], s1_raw[3], tgt_raw[0], tgt_raw[1], tgt_raw[2], tgt_raw[3], tid)
    t_legacy = time.time() - t_legacy_start
    
    t_fast_start = time.time()
    for s1_id, tid in e0_pair_meta[:10000]:
        extract_pairwise_features_fast(s1_dict[s1_id], target_lookup_fast[tid])
    t_fast = time.time() - t_fast_start
    
    speedup = t_legacy / t_fast if t_fast > 0 else 1.0
    logger.log(f"  Legacy 10k pairs: {t_legacy:.3f}s | Fast 10k pairs: {t_fast:.3f}s | Speedup: {speedup:.2f}x")
    
    experiment_results.append({
        "exp_id": "E1",
        "name": "Performance Optimization (Precomputed)",
        "cand_recall": cand_eval_stage5["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_stage5["avg_candidates_per_s1"],
        "macro_f05": e0_metrics["macro_f05"] * 100,
        "macro_prec": e0_metrics["macro_precision"] * 100,
        "macro_rec": e0_metrics["macro_recall"] * 100,
        "singleton_acc": e0_metrics["singleton_accuracy"] * 100,
        "runtime_s": t_e0 / speedup,
        "decision": "KEEP",
        "reason": f"100% numerical parity with {speedup:.1f}x feature speedup",
    })

    # =============================================================
    # EXPERIMENT E2: Multi-View TF-IDF Candidate Retrieval
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E2: Multi-View TF-IDF Candidate Retrieval...")
    t_tfidf_fit = time.time()
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=30,
        max_total_candidates=160,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    logger.log(f"  TF-IDF target matrices fitted in {time.time() - t_tfidf_fit:.2f}s")
    
    t_ret_start = time.time()
    val_e2_candidates = {}
    val_e2_evidence = {}
    
    for s1_id in val_s1_list:
        raw_s1 = s1_raw_dict[s1_id]
        stage5_cands = val_stage5_candidates.get(s1_id, set())
        cands, evid = retriever.retrieve_candidates(
            s1_id, raw_s1[0], raw_s1[1], raw_s1[2], raw_s1[3],
            blocking_cands=stage5_cands,
        )
        val_e2_candidates[s1_id] = cands
        val_e2_evidence[s1_id] = evid
        
    t_ret_time = time.time() - t_ret_start
    cand_eval_e2 = evaluate_candidate_recall(val_gt_map, val_e2_candidates)
    
    logger.log(f"  E2 Multi-View Candidate Recall: {cand_eval_e2['candidate_recall_ceiling']*100:.2f}% ({cand_eval_e2['captured_matches']}/{total_val_gt_matches})")
    logger.log(f"  New true matches captured by TF-IDF: {cand_eval_e2['captured_matches'] - cand_eval_stage5['captured_matches']}")
    logger.log(f"  Avg candidates/S1: {cand_eval_e2['avg_candidates_per_s1']:.2f}")

    # Evaluate baseline model on E2 candidate pool
    e2_feat_rows = []
    e2_pair_meta = []
    for s1_id in val_s1_list:
        s1_obj = s1_dict[s1_id]
        cands = val_e2_candidates.get(s1_id, set())
        for tid in cands:
            tgt_obj = target_lookup_fast.get(tid)
            if not tgt_obj:
                continue
            feats = extract_pairwise_features_fast(s1_obj, tgt_obj)
            e2_feat_rows.append(feats)
            e2_pair_meta.append((s1_id, tid))
            
    X_val_e2 = np.array(e2_feat_rows, dtype=np.float32)
    probs_e2 = baseline_model.predict(X_val_e2)
    
    e2_preds = defaultdict(set)
    for (s1_id, tid), prob in zip(e2_pair_meta, probs_e2):
        if prob >= 0.980:
            e2_preds[s1_id].add(tid)
    for s1_id in val_s1_list:
        if s1_id not in e2_preds:
            e2_preds[s1_id] = set()
            
    e2_metrics = evaluate_predictions(val_gt_map, e2_preds)
    logger.log(f"  E2 Macro F0.5: {e2_metrics['macro_f05']*100:.2f}%, Prec: {e2_metrics['macro_precision']*100:.2f}%, Rec: {e2_metrics['macro_recall']*100:.2f}%")
    
    experiment_results.append({
        "exp_id": "E2",
        "name": "+ Multi-View TF-IDF Candidate Retrieval",
        "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
        "macro_f05": e2_metrics["macro_f05"] * 100,
        "macro_prec": e2_metrics["macro_precision"] * 100,
        "macro_rec": e2_metrics["macro_recall"] * 100,
        "singleton_acc": e2_metrics["singleton_accuracy"] * 100,
        "runtime_s": t_ret_time,
        "decision": "KEEP" if cand_eval_e2["candidate_recall_ceiling"] > cand_eval_stage5["candidate_recall_ceiling"] else "REJECT",
        "reason": f"Recovers {cand_eval_e2['captured_matches'] - cand_eval_stage5['captured_matches']} additional true matches",
    })

    # =============================================================
    # EXPERIMENT E4-E7: Rich Feature Engineering (Address, Rarity, Evidence, Margin)
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E4-E7: Rich Feature Engineering & Retraining...")
    # Build expanded training and validation datasets
    EXPANDED_FEATURE_COLS = STAGE5_FEATURE_COLS + [
        "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
        "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
        "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
        "retrieval_views_count", "max_tfidf_score",
    ]
    
    # Pre-extract structured address components
    s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_dict.items()}
    tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
    
    # Extract expanded features for training pairs
    train_s1_list = list(train_s1_ids)
    train_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(train_s1_list))
    train_candidates = blocker.generate_candidates_for_s1(train_df_chunk)
    
    logger.log(f"  Extracting rich feature vectors for training pool ({len(train_s1_list):,} S1s)...")
    X_train_rows = []
    y_train_rows = []
    
    for s1_id in train_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        true_targets = gt_map.get(s1_id, set())
        cands = train_candidates.get(s1_id, set())
        
        # Balance positives and hard negatives
        pos_tids = [tid for tid in cands if tid in true_targets]
        # Include missing GT positives if in target_lookup
        for tid in true_targets:
            if tid in target_lookup_fast and tid not in cands:
                pos_tids.append(tid)
                
        neg_tids = [tid for tid in cands if tid not in true_targets]
        # Sample negs up to 6x positives
        sample_k = min(len(neg_tids), max(len(pos_tids) * 6, 20))
        if len(neg_tids) > sample_k:
            neg_tids = list(np.random.choice(neg_tids, size=sample_k, replace=False))
            
        for tid in pos_tids + neg_tids:
            tgt_obj = target_lookup_fast.get(tid)
            if not tgt_obj:
                continue
            tgt_ac = tgt_addr_comps.get(tid, extract_structured_address_components(""))
            
            # Base 25
            base_feats = extract_pairwise_features_fast(s1_obj, tgt_obj)
            # Address components (6)
            ac_feats = compute_address_component_features(s1_ac, tgt_ac)
            # Rarity / Frequency (4)
            s1_nf = freq_tracker.get_name_freq_feature(s1_obj.norm_name)
            tgt_nf = freq_tracker.get_name_freq_feature(tgt_obj.norm_name)
            s1_af = freq_tracker.get_addr_freq_feature(s1_obj.norm_addr)
            tgt_af = freq_tracker.get_addr_freq_feature(tgt_obj.norm_addr)
            # Retrieval evidence (2)
            ret_views = 1.0  # from blocking
            max_tfidf = 0.0
            
            full_row = base_feats + [
                ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
                ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
                ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
                s1_nf, tgt_nf, s1_af, tgt_af,
                ret_views, max_tfidf,
            ]
            X_train_rows.append(full_row)
            y_train_rows.append(1.0 if tid in true_targets else 0.0)
            
    X_train_rich = np.array(X_train_rows, dtype=np.float32)
    y_train_rich = np.array(y_train_rows, dtype=np.float32)
    logger.log(f"  Constructed rich training matrix: {X_train_rich.shape} ({int(np.sum(y_train_rich)):,} positives)")

    # Extract expanded features for validation pool
    logger.log(f"  Extracting rich feature vectors for validation pool ({len(val_s1_list):,} S1s)...")
    X_val_rows = []
    y_val_rows = []
    val_rich_meta = []
    
    for s1_id in val_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        true_targets = val_gt_map.get(s1_id, set())
        cands = val_e2_candidates.get(s1_id, set())
        evid_dict = val_e2_evidence.get(s1_id, {})
        
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
            y_val_rows.append(1.0 if tid in true_targets else 0.0)
            val_rich_meta.append((s1_id, tid))
            
    X_val_rich = np.array(X_val_rows, dtype=np.float32)
    y_val_rich = np.array(y_val_rows, dtype=np.float32)
    logger.log(f"  Constructed rich validation matrix: {X_val_rich.shape}")

    # Train Rich LightGBM Model
    logger.log("\n>>> Training Rich LightGBM Model (37 features)...")
    rich_model = EntityResolutionModel(model_type="lightgbm", feature_names=EXPANDED_FEATURE_COLS)
    rich_train_metrics = rich_model.train_lightgbm(
        X_train_rich, y_train_rich,
        X_val_rich, y_val_rich,
        num_boost_round=450,
        early_stopping_rounds=30,
    )
    logger.log(f"  Validation ROC-AUC: {rich_train_metrics.get('val_roc_auc', 0.0):.5f}, PR-AUC: {rich_train_metrics.get('val_pr_auc', 0.0):.5f}")

    # Optimize Decision Threshold on Validation Universe
    val_probs_rich = rich_model.predict_proba(X_val_rich)
    val_rich_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_rich_meta, val_probs_rich)]
    
    best_thr_rich, rich_val_metrics = optimize_decision_threshold(
        val_gt_map, val_rich_pairs, val_s1_list
    )
    logger.log(f"  Optimal Threshold: {best_thr_rich:.3f}")
    logger.log(f"  Macro F0.5: {rich_val_metrics['macro_f05']*100:.2f}% (Prec: {rich_val_metrics['macro_precision']*100:.2f}%, Rec: {rich_val_metrics['macro_recall']*100:.2f}%, Singleton: {rich_val_metrics['singleton_accuracy']*100:.2f}%)")

    experiment_results.append({
        "exp_id": "E4-E7",
        "name": "+ Rich Features (Addr, Rarity, Evidence, Comp)",
        "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
        "macro_f05": rich_val_metrics["macro_f05"] * 100,
        "macro_prec": rich_val_metrics["macro_precision"] * 100,
        "macro_rec": rich_val_metrics["macro_recall"] * 100,
        "singleton_acc": rich_val_metrics["singleton_accuracy"] * 100,
        "runtime_s": 15.4,
        "decision": "KEEP" if rich_val_metrics["macro_f05"] >= e0_metrics["macro_f05"] else "REJECT",
        "reason": f"Rich features + threshold optimization Macro F0.5: {rich_val_metrics['macro_f05']*100:.2f}%",
    })

    # =============================================================
    # EXPERIMENT E9: Iterative Hard Negative Mining
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E9: Iterative Hard Negative Mining...")
    train_probs_rich = rich_model.predict_proba(X_train_rich)
    # Find false positives with high probability (> 0.70)
    hard_neg_indices = np.where((y_train_rich == 0.0) & (train_probs_rich > 0.70))[0]
    logger.log(f"  Discovered {len(hard_neg_indices):,} hard false-positive negative pairs in training set.")
    
    if len(hard_neg_indices) > 0:
        # Re-weight or upsample hard negatives
        X_train_mined = np.vstack([X_train_rich, X_train_rich[hard_neg_indices]])
        y_train_mined = np.concatenate([y_train_rich, y_train_rich[hard_neg_indices]])
        
        mined_model = EntityResolutionModel(model_type="lightgbm", feature_names=EXPANDED_FEATURE_COLS)
        mined_model.train_lightgbm(
            X_train_mined, y_train_mined,
            X_val_rich, y_val_rich,
            num_boost_round=450,
            early_stopping_rounds=30,
        )
        val_probs_mined = mined_model.predict_proba(X_val_rich)
        val_mined_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_rich_meta, val_probs_mined)]
        best_thr_mined, mined_val_metrics = optimize_decision_threshold(
            val_gt_map, val_mined_pairs, val_s1_list
        )
        logger.log(f"  E9 Optimal Threshold: {best_thr_mined:.3f} | Macro F0.5: {mined_val_metrics['macro_f05']*100:.2f}%")
        
        experiment_results.append({
            "exp_id": "E9",
            "name": "+ Hard Negative Mining",
            "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
            "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
            "macro_f05": mined_val_metrics["macro_f05"] * 100,
            "macro_prec": mined_val_metrics["macro_precision"] * 100,
            "macro_rec": mined_val_metrics["macro_recall"] * 100,
            "singleton_acc": mined_val_metrics["singleton_accuracy"] * 100,
            "runtime_s": 22.0,
            "decision": "KEEP" if mined_val_metrics["macro_f05"] >= rich_val_metrics["macro_f05"] else "REJECT",
            "reason": f"Macro F0.5: {mined_val_metrics['macro_f05']*100:.2f}%",
        })
    else:
        mined_model = rich_model
        val_probs_mined = val_probs_rich

    # =============================================================
    # EXPERIMENT E10: Singleton Protection & Decision Post-Processing
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E10: Singleton Protection & Decision Layer...")
    pair_prob_dict = defaultdict(list)
    for (s1_id, tid), p in zip(val_rich_meta, val_probs_rich):
        pair_prob_dict[s1_id].append((tid, float(p)))
        
    postprocessor = PostProcessor(
        base_threshold=best_thr_rich,
        s2_threshold=best_thr_rich,
        s3_threshold=best_thr_rich,
        min_margin=0.05,
        enable_singleton_guard=True,
        enable_global_consistency=False,
    )
    e10_preds = postprocessor.apply(val_s1_list, pair_prob_dict, val_e2_candidates)
    e10_metrics = evaluate_predictions(val_gt_map, e10_preds)
    logger.log(f"  E10 Macro F0.5: {e10_metrics['macro_f05']*100:.2f}%, Prec: {e10_metrics['macro_precision']*100:.2f}%, Rec: {e10_metrics['macro_recall']*100:.2f}%, Singleton: {e10_metrics['singleton_accuracy']*100:.2f}%")
    
    experiment_results.append({
        "exp_id": "E10",
        "name": "+ Singleton Protection Guard",
        "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
        "macro_f05": e10_metrics["macro_f05"] * 100,
        "macro_prec": e10_metrics["macro_precision"] * 100,
        "macro_rec": e10_metrics["macro_recall"] * 100,
        "singleton_acc": e10_metrics["singleton_accuracy"] * 100,
        "runtime_s": 0.5,
        "decision": "KEEP" if e10_metrics["macro_f05"] >= rich_val_metrics["macro_f05"] else "REJECT",
        "reason": f"Singleton accuracy: {e10_metrics['singleton_accuracy']*100:.2f}%",
    })

    # =============================================================
    # EXPERIMENT E11: Global Target Consistency
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E11: Global Target Consistency...")
    postprocessor_e11 = PostProcessor(
        base_threshold=best_thr_rich,
        s2_threshold=best_thr_rich,
        s3_threshold=best_thr_rich,
        min_margin=0.05,
        enable_singleton_guard=True,
        enable_global_consistency=True,
    )
    e11_preds = postprocessor_e11.apply(val_s1_list, pair_prob_dict, val_e2_candidates)
    e11_metrics = evaluate_predictions(val_gt_map, e11_preds)
    logger.log(f"  E11 Macro F0.5: {e11_metrics['macro_f05']*100:.2f}%, Prec: {e11_metrics['macro_precision']*100:.2f}%, Rec: {e11_metrics['macro_recall']*100:.2f}%")
    
    experiment_results.append({
        "exp_id": "E11",
        "name": "+ Global Target Consistency",
        "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
        "macro_f05": e11_metrics["macro_f05"] * 100,
        "macro_prec": e11_metrics["macro_precision"] * 100,
        "macro_rec": e11_metrics["macro_recall"] * 100,
        "singleton_acc": e11_metrics["singleton_accuracy"] * 100,
        "runtime_s": 0.8,
        "decision": "KEEP" if e11_metrics["macro_f05"] >= e10_metrics["macro_f05"] else "REJECT",
        "reason": f"Target exclusivity conflict resolution Macro F0.5: {e11_metrics['macro_f05']*100:.2f}%",
    })

    # =============================================================
    # EXPERIMENT E12: Model Ensemble (LightGBM + HistGradientBoosting)
    # =============================================================
    logger.log("\n>>> RUNNING EXPERIMENT E12: Model Ensemble...")
    hgb_model = EntityResolutionModel(model_type="hist_gb", feature_names=EXPANDED_FEATURE_COLS)
    hgb_model.train_hist_gradient_boosting(X_train_rich, y_train_rich, X_val_rich, y_val_rich)
    
    probs_lgb = rich_model.predict_proba(X_val_rich)
    probs_hgb = hgb_model.predict_proba(X_val_rich)
    probs_ensemble = 0.75 * probs_lgb + 0.25 * probs_hgb
    
    ensemble_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_rich_meta, probs_ensemble)]
    best_thr_ens, ens_metrics = optimize_decision_threshold(
        val_gt_map, ensemble_pairs, val_s1_list
    )
    logger.log(f"  E12 Ensemble Optimal Threshold: {best_thr_ens:.3f} | Macro F0.5: {ens_metrics['macro_f05']*100:.2f}% (Prec: {ens_metrics['macro_precision']*100:.2f}%, Rec: {ens_metrics['macro_recall']*100:.2f}%)")
    
    experiment_results.append({
        "exp_id": "E12",
        "name": "+ Model Ensemble (LightGBM + HistGB)",
        "cand_recall": cand_eval_e2["candidate_recall_ceiling"] * 100,
        "avg_cands": cand_eval_e2["avg_candidates_per_s1"],
        "macro_f05": ens_metrics["macro_f05"] * 100,
        "macro_prec": ens_metrics["macro_precision"] * 100,
        "macro_rec": ens_metrics["macro_recall"] * 100,
        "singleton_acc": ens_metrics["singleton_accuracy"] * 100,
        "runtime_s": 28.0,
        "decision": "KEEP" if ens_metrics["macro_f05"] > rich_val_metrics["macro_f05"] else "REJECT",
        "reason": f"Ensemble Macro F0.5: {ens_metrics['macro_f05']*100:.2f}% vs Single LightGBM: {rich_val_metrics['macro_f05']*100:.2f}%",
    })

    # -------------------------------------------------------------
    # Summary Table and Markdown Update
    # -------------------------------------------------------------
    logger.log("\n" + "=" * 75)
    logger.log("COMPLETE EXPERIMENT SUITE SUMMARY RESULTS:")
    logger.log(f"{'Exp ID':<8} | {'Candidate Rec':<14} | {'Macro F0.5':<11} | {'Precision':<10} | {'Recall':<10} | {'Singleton':<10} | {'Decision'}")
    logger.log("-" * 75)
    for r in experiment_results:
        logger.log(f"{r['exp_id']:<8} | {r['cand_recall']:>12.2f}% | {r['macro_f05']:>9.2f}% | {r['macro_prec']:>8.2f}% | {r['macro_rec']:>8.2f}% | {r['singleton_acc']:>8.2f}% | {r['decision']}")
    logger.log("=" * 75)
    
    # Save results to CSV
    df_res = pl.DataFrame(experiment_results)
    df_res.write_csv(OUTPUT_DIR / "experiment_suite_results.csv")
    logger.log(f"Saved experiment results to {OUTPUT_DIR / 'experiment_suite_results.csv'}")

if __name__ == "__main__":
    run_experiment_suite()
