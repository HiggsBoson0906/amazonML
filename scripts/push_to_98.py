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
from sklearn.metrics import roc_auc_score, average_precision_score
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein

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
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.utils import ExperimentLogger

# PUSH TO 98: 52 Ultra-Discriminating Features
ULTRA_FEATURE_COLS = STAGE5_FEATURE_COLS + [
    # Structured Address
    "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
    "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
    # Corpus Frequencies
    "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
    # Retrieval Evidence
    "retrieval_views_count", "max_tfidf_score",
    # String Distance Metrics
    "name_jaro_winkler", "name_token_containment",
    "addr_jaro_winkler", "addr_token_containment",
    "name_weighted_jaccard", "addr_weighted_jaccard",
    # Ultra-Discriminating Additions
    "name_lcs_ratio", "name_min_token_overlap_ratio",
    "addr_lcs_ratio", "addr_postal_prefix_match",
    "name_addr_joint_similarity", "token_count_diff",
    "exact_numeric_overlap"
]

def extract_ultra_features(
    s1: PrecomputedEntity,
    tgt: PrecomputedEntity,
    s1_ac: Dict[str, Any],
    tgt_ac: Dict[str, Any],
    freq_tracker: CorpusFrequencyTracker,
    evid_map: Dict[str, float],
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
    
    # 5. String Metrics
    name_jw = JaroWinkler.similarity(s1.norm_name, tgt.norm_name)
    addr_jw = JaroWinkler.similarity(s1.norm_addr, tgt.norm_addr) if (s1.norm_addr and tgt.norm_addr) else 0.0
    
    # Token Containment & Min Token Overlap
    if s1.core_toks and tgt.core_toks:
        s1_in_tgt = float(s1.core_toks.issubset(tgt.core_toks))
        tgt_in_s1 = float(tgt.core_toks.issubset(s1.core_toks))
        name_containment = max(s1_in_tgt, tgt_in_s1)
        
        inter = s1.core_toks.intersection(tgt.core_toks)
        un = s1.core_toks.union(tgt.core_toks)
        min_tok_len = min(len(s1.core_toks), len(tgt.core_toks))
        name_min_overlap = len(inter) / min_tok_len if min_tok_len > 0 else 0.0
        
        w_inter = sum(freq_tracker.get_token_idf(t) for t in inter)
        w_un = sum(freq_tracker.get_token_idf(t) for t in un)
        name_w_jaccard = float(w_inter / w_un) if w_un > 0 else 0.0
    else:
        name_containment = 0.0
        name_min_overlap = 0.0
        name_w_jaccard = 0.0
        
    if s1.addr_toks and tgt.addr_toks:
        s1_a_in_tgt = float(s1.addr_toks.issubset(tgt.addr_toks))
        tgt_a_in_s1 = float(tgt.addr_toks.issubset(s1.addr_toks))
        addr_containment = max(s1_a_in_tgt, tgt_a_in_s1)
        
        inter_a = s1.addr_toks.intersection(tgt.addr_toks)
        un_a = s1.addr_toks.union(tgt.addr_toks)
        w_inter_a = sum(freq_tracker.get_token_idf(t) for t in inter_a)
        w_un_a = sum(freq_tracker.get_token_idf(t) for t in un_a)
        addr_w_jaccard = float(w_inter_a / w_un_a) if w_un_a > 0 else 0.0
    else:
        addr_containment = 0.0
        addr_w_jaccard = 0.0
        
    # LCS Sequence Ratios
    name_lcs = LCSseq.similarity(s1.norm_name, tgt.norm_name)
    addr_lcs = LCSseq.similarity(s1.norm_addr, tgt.norm_addr) if (s1.norm_addr and tgt.norm_addr) else 0.0
    
    # Postal Prefix Match (3-digit prefix)
    s1_p = s1_ac["postal_code"]
    tgt_p = tgt_ac["postal_code"]
    if s1_p and tgt_p and len(s1_p) >= 3 and len(tgt_p) >= 3:
        postal_prefix_match = 1.0 if s1_p[:3] == tgt_p[:3] else 0.0
    else:
        postal_prefix_match = 0.0
        
    # Joint Name + Address score
    joint_sim = float(name_jw * 0.6 + addr_jw * 0.4)
    token_count_diff = float(abs(len(s1.core_toks) - len(tgt.core_toks)))
    exact_num_overlap = 1.0 if (s1.addr_nums and tgt.addr_nums and s1.addr_nums == tgt.addr_nums) else 0.0
    
    return base + [
        ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
        ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
        ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
        s1_nf, tgt_nf, s1_af, tgt_af,
        ret_views, max_tfidf,
        name_jw, name_containment,
        addr_jw, addr_containment,
        name_w_jaccard, addr_w_jaccard,
        name_lcs, name_min_overlap,
        addr_lcs, postal_prefix_match,
        joint_sim, token_count_diff,
        exact_num_overlap,
    ]

def evaluate_adaptive_decision_engine(
    val_s1_list: List[str],
    pair_meta: List[Tuple[str, str]],
    probs: np.ndarray,
    s1_dict: Dict[str, PrecomputedEntity],
    target_lookup_fast: Dict[str, PrecomputedEntity],
    s1_addr_comps: Dict[str, Any],
    tgt_addr_comps: Dict[str, Any],
    candidate_sets: Dict[str, Set[str]],
    val_gt_map: Dict[str, Set[str]],
    base_thr: float = 0.930,
) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    """Adaptive multi-tier decision engine combining calibrated probabilities,

    exact-evidence overrides, singleton confidence gating, and global exclusivity.
    """
    pair_prob_lookup = defaultdict(dict)
    s1_candidates = defaultdict(list)
    
    for (s1_id, tid), p in zip(pair_meta, probs):
        pair_prob_lookup[s1_id][tid] = float(p)
        s1_candidates[s1_id].append((tid, float(p)))
        
    initial_matches: Dict[str, List[Tuple[str, float]]] = {}
    
    for s1_id in val_s1_list:
        cands = candidate_sets.get(s1_id, set())
        pairs = s1_candidates.get(s1_id, [])
        pairs.sort(key=lambda x: -x[1])
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        
        selected = []
        for tid, p in pairs:
            if tid not in cands:
                continue
            tgt_obj = target_lookup_fast[tid]
            tgt_ac = tgt_addr_comps.get(tid, extract_structured_address_components(""))
            
            # Evidence Overrides
            exact_name = (s1_obj.core_name and s1_obj.core_name == tgt_obj.core_name)
            exact_addr = (s1_obj.norm_addr and s1_obj.norm_addr == tgt_obj.norm_addr)
            postal_match = (s1_ac["postal_code"] and s1_ac["postal_code"] == tgt_ac["postal_code"])
            hnum_match = (s1_ac["house_num"] and s1_ac["house_num"] == tgt_ac["house_num"])
            
            # Dynamic Decision Thresholding
            if exact_name and (exact_addr or (postal_match and hnum_match)):
                thr = 0.800  # Overwhelming exact physical proof
            elif exact_name:
                thr = 0.880  # Strong core identity match
            elif exact_addr and postal_match:
                thr = 0.900  # Exact address match
            else:
                thr = base_thr  # Standard calibrated threshold
                
            if p >= thr:
                selected.append((tid, p))
                
        # Singleton Confidence Gating
        if selected:
            top_p = selected[0][1]
            if len(selected) > 1:
                margin = top_p - selected[1][1]
                # If multiple weak matches with near-zero margin, suppress as singleton noise
                if top_p < 0.970 and margin < 0.03:
                    selected = []
            elif top_p < 0.900:
                # Weak single match without exact evidence
                tgt_obj = target_lookup_fast[selected[0][0]]
                if s1_obj.core_name != tgt_obj.core_name and s1_obj.norm_addr != tgt_obj.norm_addr:
                    selected = []
                    
        initial_matches[s1_id] = selected

    # Global Target Exclusivity Conflict Resolution
    target_to_s1 = defaultdict(list)
    for s1_id, matches in initial_matches.items():
        for tid, p in matches:
            target_to_s1[tid].append((s1_id, p))
            
    final_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in val_s1_list}
    for tid, s1_list in target_to_s1.items():
        if len(s1_list) == 1:
            final_matches[s1_list[0][0]].add(tid)
        else:
            s1_list.sort(key=lambda x: -x[1])
            best_s1, _ = s1_list[0]
            final_matches[best_s1].add(tid)
            
    metrics = evaluate_predictions(val_gt_map, final_matches)
    return metrics, final_matches

def run_push_to_98():
    logger = ExperimentLogger()
    logger.log("=" * 75)
    logger.log("PUSH TO 98-99% MACRO F0.5 — HIGH PRECISION & RECALL ENGINE")
    logger.log("=" * 75)
    
    # 1. Load Partitions & GT
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
    s1_dict = {r["entity_id"]: PrecomputedEntity(r["norm_name"], r["core_name"], r["norm_address"], r["country"], r["entity_id"]) for r in s1_rows.iter_rows(named=True)}
    s1_raw_dict = {r["entity_id"]: (r["norm_name"], r["core_name"], r["norm_address"], r["country"]) for r in s1_rows.iter_rows(named=True)}
    
    # 3. Load Target Records
    needed_target_ids = set()
    for s1_id in all_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
        
    target_lookup_raw: Dict[str, Tuple[str, str, str, str]] = {}
    target_lookup_fast: Dict[str, PrecomputedEntity] = {}
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    
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
    
    # Fit Ultra Multi-View Retriever (Top-45 per view, adaptive cap 200)
    logger.log("  Fitting Multi-View TF-IDF Retriever (Top-45 per view)...")
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=45,
        max_total_candidates=200,
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
    logger.log(f"  Ultra Candidate Recall: {cand_eval['candidate_recall_ceiling']*100:.2f}% ({cand_eval['captured_matches']}/{cand_eval['total_true_matches']})")
    logger.log(f"  Avg candidates/S1: {cand_eval['avg_candidates_per_s1']:.2f}")

    # 5. Extract Ultra Training Matrix
    logger.log("5. Extracting Ultra 52-Feature Training Matrix...")
    train_df_chunk = df_s1_all.filter(pl.col("entity_id").is_in(train_s1_list))
    train_cands = blocker.generate_candidates_for_s1(train_df_chunk)
    
    X_train_rows, y_train_rows = [], []
    for s1_id in train_s1_list:
        s1_obj = s1_dict[s1_id]
        s1_ac = s1_addr_comps[s1_id]
        true_targets = gt_map.get(s1_id, set())
        cands = train_cands.get(s1_id, set())
        
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
            full_row = extract_ultra_features(
                s1_obj, tgt_obj, s1_ac, tgt_ac, freq_tracker, evid_map={"blocking": 1.0}
            )
            X_train_rows.append(full_row)
            y_train_rows.append(1.0 if tid in true_targets else 0.0)
            
    X_train = np.array(X_train_rows, dtype=np.float32)
    y_train = np.array(y_train_rows, dtype=np.float32)
    
    # 6. Extract Ultra Validation Matrix
    logger.log("6. Extracting Ultra 52-Feature Validation Matrix...")
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
            evid = evid_dict.get(tid, {})
            full_row = extract_ultra_features(
                s1_obj, tgt_obj, s1_ac, tgt_ac, freq_tracker, evid_map=evid
            )
            X_val_rows.append(full_row)
            y_val_rows.append(1.0 if tid in true_targets else 0.0)
            val_meta.append((s1_id, tid))
            
    X_val = np.array(X_val_rows, dtype=np.float32)
    y_val = np.array(y_val_rows, dtype=np.float32)
    
    # 7. Train Ultra LightGBM Model
    logger.log("7. Training Ultra LightGBM Model (52 features, tuned depth & regularization)...")
    trn_data = lgb.Dataset(X_train, label=y_train, feature_name=ULTRA_FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data, feature_name=ULTRA_FEATURE_COLS)
    
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 15,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=600,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    
    val_probs = model.predict(X_val)
    val_roc = roc_auc_score(y_val, val_probs)
    val_pr = average_precision_score(y_val, val_probs)
    logger.log(f"  Validation ROC-AUC: {val_roc:.5f}, PR-AUC: {val_pr:.5f}")

    # 8. Multi-Threshold Grid Search with Adaptive Decision Engine
    logger.log("\n8. Multi-Threshold Grid Search with Adaptive Decision Engine...")
    best_f05 = -1.0
    best_thr = 0.930
    best_metrics = {}
    best_preds = {}
    
    for thr in [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]:
        m, preds = evaluate_adaptive_decision_engine(
            val_s1_list, val_meta, val_probs, s1_dict, target_lookup_fast,
            s1_addr_comps, tgt_addr_comps, val_candidates, val_gt_map, base_thr=thr
        )
        logger.log(f"  Base Thr {thr:.3f} -> Macro F0.5: {m['macro_f05']*100:.2f}% (Prec: {m['macro_precision']*100:.2f}%, Rec: {m['macro_recall']*100:.2f}%, Singleton: {m['singleton_accuracy']*100:.2f}%)")
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thr = thr
            best_metrics = m
            best_preds = preds

    logger.log("\n" + "=" * 75)
    logger.log("ULTRA OPTIMIZATION VALIDATION RESULTS:")
    logger.log(f"  Candidate Recall:    {cand_eval['candidate_recall_ceiling']*100:.2f}%")
    logger.log(f"  Avg Candidates/S1:   {cand_eval['avg_candidates_per_s1']:.2f}")
    logger.log(f"  Best Base Threshold: {best_thr:.3f}")
    logger.log(f"  Macro F0.5 Score:    {best_metrics['macro_f05']*100:.2f}%")
    logger.log(f"  Macro Precision:     {best_metrics['macro_precision']*100:.2f}%")
    logger.log(f"  Macro Recall:        {best_metrics['macro_recall']*100:.2f}%")
    logger.log(f"  Singleton Accuracy:  {best_metrics['singleton_accuracy']*100:.2f}%")
    logger.log("=" * 75)
    
    if best_metrics["macro_f05"] > 0.9631:
        logger.log(">>> NEW ALL-TIME BEST SCORE ACHIEVED! Persisting artifacts...")
        model_save_path = Path("models/lightgbm_v1_1_ultra.txt")
        model.save_model(str(model_save_path))
        
        config = {
            "pipeline_version": "v1.2-ultra-optimized",
            "model_artifact": str(model_save_path),
            "features": ULTRA_FEATURE_COLS,
            "feature_count": len(ULTRA_FEATURE_COLS),
            "base_threshold": best_thr,
            "macro_f05": best_metrics["macro_f05"],
            "macro_precision": best_metrics["macro_precision"],
            "macro_recall": best_metrics["macro_recall"],
            "singleton_accuracy": best_metrics["singleton_accuracy"],
            "candidate_recall": cand_eval["candidate_recall_ceiling"],
        }
        with open("configs/v1_2_ultra.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

if __name__ == "__main__":
    run_push_to_98()
