import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

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
from src.postprocess import PostProcessor, optimize_decision_threshold
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.utils import ExperimentLogger

# V1.1 Rich 45-Feature Specification
V1_1_FEATURE_COLS = STAGE5_FEATURE_COLS + [
    "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
    "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
    "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
    "retrieval_views_count", "max_tfidf_score",
    # Last-Mile Features
    "name_jaro_winkler", "name_token_containment",
    "addr_jaro_winkler", "addr_token_containment",
    "name_weighted_jaccard", "addr_weighted_jaccard",
    "cand_score_margin", "cand_pool_ambiguity",
]

def extract_v1_1_pairwise_features(
    s1: PrecomputedEntity,
    tgt: PrecomputedEntity,
    s1_ac: Dict[str, Any],
    tgt_ac: Dict[str, Any],
    freq_tracker: CorpusFrequencyTracker,
    evid_map: Dict[str, float],
    margin: float = 0.0,
    ambiguity_count: float = 1.0,
) -> List[float]:
    # 1. Base 25
    base = extract_pairwise_features_fast(s1, tgt)
    
    # 2. Address components (6)
    ac_feats = compute_address_component_features(s1_ac, tgt_ac)
    
    # 3. Corpus frequencies (4)
    s1_nf = freq_tracker.get_name_freq_feature(s1.norm_name)
    tgt_nf = freq_tracker.get_name_freq_feature(tgt.norm_name)
    s1_af = freq_tracker.get_addr_freq_feature(s1.norm_addr)
    tgt_af = freq_tracker.get_addr_freq_feature(tgt.norm_addr)
    
    # 4. Retrieval consensus (2)
    ret_views = float(len(evid_map)) if evid_map else 1.0
    max_tfidf = max([v for k, v in evid_map.items() if k != "blocking"], default=0.0)
    
    # 5. Last-Mile String & Weighted Similarities (8)
    name_jw = JaroWinkler.similarity(s1.norm_name, tgt.norm_name)
    addr_jw = JaroWinkler.similarity(s1.norm_addr, tgt.norm_addr) if (s1.norm_addr and tgt.norm_addr) else 0.0
    
    # Token Containment
    if s1.core_toks and tgt.core_toks:
        s1_in_tgt = float(s1.core_toks.issubset(tgt.core_toks))
        tgt_in_s1 = float(tgt.core_toks.issubset(s1.core_toks))
        name_containment = max(s1_in_tgt, tgt_in_s1)
    else:
        name_containment = 0.0
        
    if s1.addr_toks and tgt.addr_toks:
        s1_a_in_tgt = float(s1.addr_toks.issubset(tgt.addr_toks))
        tgt_a_in_s1 = float(tgt.addr_toks.issubset(s1.addr_toks))
        addr_containment = max(s1_a_in_tgt, tgt_a_in_s1)
    else:
        addr_containment = 0.0
        
    # IDF Weighted Jaccard
    if s1.core_toks and tgt.core_toks:
        inter = s1.core_toks.intersection(tgt.core_toks)
        un = s1.core_toks.union(tgt.core_toks)
        w_inter = sum(freq_tracker.get_token_idf(t) for t in inter)
        w_un = sum(freq_tracker.get_token_idf(t) for t in un)
        name_w_jaccard = float(w_inter / w_un) if w_un > 0 else 0.0
    else:
        name_w_jaccard = 0.0
        
    if s1.addr_toks and tgt.addr_toks:
        inter_a = s1.addr_toks.intersection(tgt.addr_toks)
        un_a = s1.addr_toks.union(tgt.addr_toks)
        w_inter_a = sum(freq_tracker.get_token_idf(t) for t in inter_a)
        w_un_a = sum(freq_tracker.get_token_idf(t) for t in un_a)
        addr_w_jaccard = float(w_inter_a / w_un_a) if w_un_a > 0 else 0.0
    else:
        addr_w_jaccard = 0.0
        
    return base + [
        ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
        ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
        ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
        s1_nf, tgt_nf, s1_af, tgt_af,
        ret_views, max_tfidf,
        name_jw, name_containment,
        addr_jw, addr_containment,
        name_w_jaccard, addr_w_jaccard,
        float(margin), float(ambiguity_count),
    ]

def run_v1_1_pipeline():
    logger = ExperimentLogger()
    logger.log("=" * 75)
    logger.log("V1.1 LAST-MILE ENTITY RESOLUTION OPTIMIZATION & VALIDATION")
    logger.log("=" * 75)
    
    # 1. Load Ground Truth and Partitions
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    old_train_df = pl.read_parquet(OUTPUT_DIR / "train_features.parquet")
    old_val_df = pl.read_parquet(OUTPUT_DIR / "validation_features.parquet")
    
    train_s1_ids = set(old_train_df["s1_id"].unique().to_list())
    val_s1_ids = set(old_val_df["s1_id"].unique().to_list())
    all_s1_ids = train_s1_ids.union(val_s1_ids)
    val_s1_list = list(val_s1_ids)
    train_s1_list = list(train_s1_ids)
    val_gt_map = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
    
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
        top_k_per_view=35,
        max_total_candidates=160,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    
    freq_tracker = CorpusFrequencyTracker()
    all_tgt_names = [e.norm_name for e in target_lookup_fast.values()]
    all_tgt_addrs = [e.norm_addr for e in target_lookup_fast.values()]
    freq_tracker.fit(all_tgt_names, all_tgt_addrs)
    
    s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_dict.items()}
    tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
    
    # 4. Construct Training Set
    logger.log("4. Constructing V1.1 Rich Training Set...")
    train_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(train_s1_list))
    train_candidates = blocker.generate_candidates_for_s1(train_df_chunk)
    
    X_train_rows, y_train_rows, train_groups = [], [], []
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
            full_row = extract_v1_1_pairwise_features(
                s1_obj, tgt_obj, s1_ac, tgt_ac, freq_tracker,
                evid_map={"blocking": 1.0}, margin=0.0, ambiguity_count=1.0,
            )
            X_train_rows.append(full_row)
            y_train_rows.append(1.0 if tid in true_targets else 0.0)
            train_groups.append(s1_id)
            
    X_train = np.array(X_train_rows, dtype=np.float32)
    y_train = np.array(y_train_rows, dtype=np.float32)
    train_groups = np.array(train_groups)
    logger.log(f"  Training Matrix: {X_train.shape} with {int(np.sum(y_train)):,} positives.")

    # 5. GroupKFold Cross-Validation on Training Cohort (Robustness & Leakage Check)
    logger.log("\n5. Running 5-Fold GroupKFold Cross-Validation on Training Set...")
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(y_train), dtype=np.float32)
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": 7,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train, y_train, train_groups)):
        X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr, feature_name=V1_1_FEATURE_COLS)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, feature_name=V1_1_FEATURE_COLS)
        
        bst = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=450,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_probs[val_idx] = bst.predict(X_va)
        
    oof_roc = roc_auc_score(y_train, oof_probs)
    oof_pr = average_precision_score(y_train, oof_probs)
    logger.log(f"  5-Fold OOF ROC-AUC: {oof_roc:.5f}, PR-AUC: {oof_pr:.5f}")

    # 6. Generate Candidates on Validation Set (with Multi-View Retrieval)
    logger.log("\n6. Generating Multi-View Candidates on Validation Cohort...")
    val_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(val_s1_list))
    val_stage5_cands = blocker.generate_candidates_for_s1(val_df_chunk)
    
    val_candidates = {}
    val_evidence = {}
    for s1_id in val_s1_list:
        rn, rc, ra, rco = s1_raw_dict[s1_id]
        cands, evid = retriever.retrieve_candidates(
            s1_id, rn, rc, ra, rco, blocking_cands=val_stage5_cands.get(s1_id, set())
        )
        val_candidates[s1_id] = cands
        val_evidence[s1_id] = evid
        
    cand_eval = evaluate_candidate_recall(val_gt_map, val_candidates)
    logger.log(f"  V1.1 Candidate Recall: {cand_eval['candidate_recall_ceiling']*100:.2f}% ({cand_eval['captured_matches']}/{cand_eval['total_true_matches']})")
    logger.log(f"  Avg candidates/S1: {cand_eval['avg_candidates_per_s1']:.2f}")

    # 7. Construct Validation Features
    logger.log("7. Constructing Validation Features...")
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
            evid_map = evid_dict.get(tid, {})
            full_row = extract_v1_1_pairwise_features(
                s1_obj, tgt_obj, s1_ac, tgt_ac, freq_tracker,
                evid_map=evid_map, margin=0.0, ambiguity_count=1.0,
            )
            X_val_rows.append(full_row)
            y_val_rows.append(1.0 if tid in true_targets else 0.0)
            val_meta.append((s1_id, tid))
            
    X_val = np.array(X_val_rows, dtype=np.float32)
    y_val = np.array(y_val_rows, dtype=np.float32)
    logger.log(f"  Validation Matrix: {X_val.shape}")

    # 8. Train Full V1.1 LightGBM Model
    logger.log("8. Training Full V1.1 LightGBM Model (45 features)...")
    trn_full = lgb.Dataset(X_train, label=y_train, feature_name=V1_1_FEATURE_COLS)
    val_full = lgb.Dataset(X_val, label=y_val, reference=trn_full, feature_name=V1_1_FEATURE_COLS)
    
    model_v1_1 = lgb.train(
        lgb_params,
        trn_full,
        num_boost_round=450,
        valid_sets=[trn_full, val_full],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    
    val_probs = model_v1_1.predict(X_val)
    val_roc = roc_auc_score(y_val, val_probs)
    val_pr = average_precision_score(y_val, val_probs)
    logger.log(f"  Validation ROC-AUC: {val_roc:.5f}, PR-AUC: {val_pr:.5f}")

    # 9. Decision Optimization (Threshold + Margin + Global Consistency)
    logger.log("9. Evaluating Threshold & Global Consistency Decision Layers...")
    val_pairs = [(s1_id, tid, float(p)) for (s1_id, tid), p in zip(val_meta, val_probs)]
    
    best_thr, _ = optimize_decision_threshold(val_gt_map, val_pairs, val_s1_list)
    
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
    
    logger.log("\n" + "=" * 75)
    logger.log("V1.1 FINAL VALIDATED PIPELINE RESULTS:")
    logger.log(f"  Candidate Recall:    {cand_eval['candidate_recall_ceiling']*100:.2f}% ({cand_eval['captured_matches']}/{cand_eval['total_true_matches']})")
    logger.log(f"  Avg Candidates / S1: {cand_eval['avg_candidates_per_s1']:.2f}")
    logger.log(f"  Decision Threshold:  {best_thr:.3f}")
    logger.log(f"  Macro F0.5 Score:    {final_metrics['macro_f05']*100:.2f}%")
    logger.log(f"  Macro Precision:     {final_metrics['macro_precision']*100:.2f}%")
    logger.log(f"  Macro Recall:        {final_metrics['macro_recall']*100:.2f}%")
    logger.log(f"  Singleton Accuracy:  {final_metrics['singleton_accuracy']*100:.2f}%")
    logger.log(f"  5-Fold OOF ROC-AUC:  {oof_roc:.5f}")
    logger.log("=" * 75)
    
    # Save Model Artifact
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / "lightgbm_v1_1_optimized.txt"
    model_v1_1.save_model(str(model_save_path))
    logger.log(f"  Saved V1.1 model to: {model_save_path}")
    
    # Save Config
    config_dict = {
        "pipeline_version": "v1.1-final-candidate",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_artifact": str(model_save_path),
        "features": V1_1_FEATURE_COLS,
        "feature_count": len(V1_1_FEATURE_COLS),
        "decision_threshold": best_thr,
        "singleton_guard": True,
        "global_consistency": True,
        "candidate_recall": cand_eval["candidate_recall_ceiling"],
        "macro_f05": final_metrics["macro_f05"],
        "macro_precision": final_metrics["macro_precision"],
        "macro_recall": final_metrics["macro_recall"],
        "singleton_accuracy": final_metrics["singleton_accuracy"],
        "oof_roc_auc": oof_roc,
    }
    with open(OUTPUT_DIR / "final_model_config_v1_1.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    with open(Path("configs/v1_1_final.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.log(f"Saved configuration to: configs/v1_1_final.json")

if __name__ == "__main__":
    run_v1_1_pipeline()
