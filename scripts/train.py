import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
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
)
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import (
    CorpusFrequencyTracker,
    extract_structured_address_components,
    compute_address_component_features,
)
from src.model import EntityResolutionModel
from src.postprocess import optimize_decision_threshold, PostProcessor
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.utils import ExperimentLogger

EXPANDED_FEATURE_COLS = STAGE5_FEATURE_COLS + [
    "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
    "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
    "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
    "retrieval_views_count", "max_tfidf_score",
]

def train_and_save_final_pipeline():
    logger = ExperimentLogger()
    logger.log("=" * 75)
    logger.log("TRAINING FINAL HIGH-PERFORMANCE ENTITY RESOLUTION PIPELINE (V1.0)")
    logger.log("=" * 75)
    
    # 1. Load Ground Truth and Partitions
    logger.log("1. Loading Ground Truth and Preserving Validation Universe...")
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    old_train_df = pl.read_parquet(OUTPUT_DIR / "train_features.parquet")
    old_val_df = pl.read_parquet(OUTPUT_DIR / "validation_features.parquet")
    
    train_s1_ids = set(old_train_df["s1_id"].unique().to_list())
    val_s1_ids = set(old_val_df["s1_id"].unique().to_list())
    all_s1_ids = train_s1_ids.union(val_s1_ids)
    
    # 2. Load Normalized Source 1
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
        
    val_gt_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    
    # 3. Load Targets
    logger.log("3. Loading and Precomputing Target Records...")
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
    
    # Fit TF-IDF Retriever
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=30,
        max_total_candidates=160,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    
    # Fit Frequency Tracker
    freq_tracker = CorpusFrequencyTracker()
    all_names = [e.norm_name for e in target_lookup_fast.values()] + [e.norm_name for e in s1_dict.values()]
    all_addrs = [e.norm_addr for e in target_lookup_fast.values()] + [e.norm_addr for e in s1_dict.values()]
    freq_tracker.fit(all_names, all_addrs)
    
    s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_dict.items()}
    tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
    
    # 4. Generate Training Pairs
    logger.log("4. Constructing Training Features...")
    train_s1_list = list(train_s1_ids)
    train_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(train_s1_list))
    train_candidates = blocker.generate_candidates_for_s1(train_df_chunk)
    
    X_train_rows, y_train_rows = [], []
    for s1_id in train_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        true_targets = gt_map.get(s1_id, set())
        cands = train_candidates.get(s1_id, set())
        
        pos_tids = [tid for tid in cands if tid in true_targets]
        for tid in true_targets:
            if tid in target_lookup_fast and tid not in cands:
                pos_tids.append(tid)
                
        neg_tids = [tid for tid in cands if tid not in true_targets]
        sample_k = min(len(neg_tids), max(len(pos_tids) * 6, 20))
        if len(neg_tids) > sample_k:
            neg_tids = list(np.random.choice(neg_tids, size=sample_k, replace=False))
            
        for tid in pos_tids + neg_tids:
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
            
            full_row = base_feats + [
                ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
                ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
                ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
                s1_nf, tgt_nf, s1_af, tgt_af,
                1.0, 0.0,
            ]
            X_train_rows.append(full_row)
            y_train_rows.append(1.0 if tid in true_targets else 0.0)
            
    X_train = np.array(X_train_rows, dtype=np.float32)
    y_train = np.array(y_train_rows, dtype=np.float32)
    
    # 5. Generate Validation Features
    logger.log("5. Constructing Validation Features...")
    val_s1_list = list(val_s1_ids)
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
        
    X_val_rows, y_val_rows, val_meta = [], [], []
    for s1_id in val_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        true_targets = val_gt_map.get(s1_id, set())
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
            y_val_rows.append(1.0 if tid in true_targets else 0.0)
            val_meta.append((s1_id, tid))
            
    X_val = np.array(X_val_rows, dtype=np.float32)
    y_val = np.array(y_val_rows, dtype=np.float32)
    
    # 6. Train Final LightGBM Model
    logger.log("6. Training Final LightGBM Model...")
    final_model = EntityResolutionModel(model_type="lightgbm", feature_names=EXPANDED_FEATURE_COLS)
    final_model.train_lightgbm(X_train, y_train, X_val, y_val, num_boost_round=450, early_stopping_rounds=30)
    
    # Save Model Artifacts (Without Overwriting Stage-5 Baseline)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / "lightgbm_v1_optimized.txt"
    final_model.save_model(model_save_path)
    logger.log(f"  Saved final model to: {model_save_path}")
    
    # 7. Evaluate and Optimize Threshold with Global Consistency
    val_probs = final_model.predict_proba(X_val)
    val_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_meta, val_probs)]
    
    best_thr, raw_metrics = optimize_decision_threshold(val_gt_map, val_pairs, val_s1_list)
    
    pair_prob_dict = defaultdict(list)
    for (s1_id, tid), p in zip(val_meta, val_probs):
        pair_prob_dict[s1_id].append((tid, float(p)))
        
    postprocessor = PostProcessor(
        base_threshold=best_thr,
        min_margin=0.05,
        enable_singleton_guard=True,
        enable_global_consistency=True,
    )
    final_preds = postprocessor.apply(val_s1_list, pair_prob_dict, val_candidates)
    final_metrics = evaluate_predictions(val_gt_map, final_preds)
    cand_eval = evaluate_candidate_recall(val_gt_map, val_candidates)
    
    logger.log("\n" + "=" * 75)
    logger.log("FINAL VALIDATED PIPELINE PERFORMANCE:")
    logger.log(f"  Candidate Recall:    {cand_eval['candidate_recall_ceiling']*100:.2f}% ({cand_eval['captured_matches']}/{cand_eval['total_true_matches']})")
    logger.log(f"  Avg Candidates / S1: {cand_eval['avg_candidates_per_s1']:.2f}")
    logger.log(f"  Decision Threshold:  {best_thr:.3f}")
    logger.log(f"  Macro F0.5 Score:    {final_metrics['macro_f05']*100:.2f}%")
    logger.log(f"  Macro Precision:     {final_metrics['macro_precision']*100:.2f}%")
    logger.log(f"  Macro Recall:        {final_metrics['macro_recall']*100:.2f}%")
    logger.log(f"  Singleton Accuracy:  {final_metrics['singleton_accuracy']*100:.2f}%")
    logger.log("=" * 75)
    
    # Save Pipeline Config JSON
    config_dict = {
        "pipeline_version": "v1.0-final-candidate",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_artifact": str(model_save_path),
        "features": EXPANDED_FEATURE_COLS,
        "feature_count": len(EXPANDED_FEATURE_COLS),
        "decision_threshold": best_thr,
        "singleton_guard": True,
        "global_consistency": True,
        "candidate_recall": cand_eval["candidate_recall_ceiling"],
        "macro_f05": final_metrics["macro_f05"],
        "macro_precision": final_metrics["macro_precision"],
        "macro_recall": final_metrics["macro_recall"],
        "singleton_accuracy": final_metrics["singleton_accuracy"],
    }
    with open(OUTPUT_DIR / "final_model_config_v1.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.log(f"Saved configuration to: {OUTPUT_DIR / 'final_model_config_v1.json'}")

if __name__ == "__main__":
    train_and_save_final_pipeline()
