import os
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import (
    ULTRA_FEATURE_COLS,
    PrecomputedEntity,
    OptimizedEntity,
    extract_ultra_features_fast,
)
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import (
    CorpusFrequencyTracker,
    extract_structured_address_components,
)
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.postprocess import SurgicalPostProcessorV1_3
from scripts.push_to_98 import extract_ultra_features

def run_preflight_suite():
    print("=" * 85)
    print("V1.3 SURGICAL PRODUCTION PRE-FLIGHT VERIFICATION SUITE")
    print("=" * 85)
    
    passed_all = True
    
    # -------------------------------------------------------------------------
    # PHASE 3: SEMANTIC FEATURE REGRESSION CHECK (Diverse Pair Matrix)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 3: SEMANTIC FEATURE REGRESSION CHECK ---", flush=True)
    
    test_cases = [
        # (name1, name2, addr1, addr2, country1, country2)
        ("walmart supercenter", "walmart supercenter", "123 main st ste 100 seattle wa 98101", "123 main st ste 100 seattle wa 98101", "US", "US"),
        ("walmart supercenter", "walmart supercenter", "123 main st ste 100 seattle wa 98101", "456 broadway ave new york ny 10001", "US", "US"),
        ("starbucks coffee", "starbucks coffee co", "789 market st suite 200 sf ca 94103", "789 market st san francisco ca 94103", "US", "US"),
        ("mcdonalds restaurant", "burger king", "100 main street austin tx 78701", "200 congress ave austin tx 78701", "US", "US"),
        ("amazon fulfillment center", "amazon logistics", "", "", "US", "US"),
        ("target store #1024", "target stores", "500 pine rd denver co 80202", "", "US", "US"),
        ("", "home depot", "", "100 elm st boston ma 02108", "US", "US"),
        ("1 2 3 logistics llc", "123 logistics inc", "12-34 56th st flushing ny 11355", "1234 56th street flushing ny 11355", "US", "US"),
        ("boulangerie patisserie de paris", "boulangerie parisienne", "15 rue de rivoli 75001 paris", "15 rue de rivoli 75001 paris", "FR", "FR"),
        ("tata consultancy services limited", "tcs ltd", "plot 42 hitec city hyderabad ts 500081", "plot 42 hitec city madhapur hyderabad 500081", "IN", "IN"),
        ("state bank of india", "sbi branch", "nariman point mumbai mh 400021", "nariman point mumbai 400021", "IN", "IN"),
        ("very long commercial enterprise corporate headquarters branch unit division", "very long commercial enterprise corporate", "suite 999 12345 north west grand central parkway building 8", "12345 nw grand central pkwy bldg 8", "US", "US"),
        ("a", "b", "1 a st", "2 b ave", "US", "US"),
        ("generico store", "generico store", "1234 5678 9012 ave", "1234 5678 9012 ave", "MX", "MX"),
        ("company with numeric 987654", "company with numeric 987654", "postal 12345", "postal 12345", "DE", "DE"),
        ("mismatch postal", "mismatch postal", "100 main st 12345", "100 main st 99999", "US", "US"),
        ("mismatch house", "mismatch house", "100 main st 12345", "200 main st 12345", "US", "US"),
        ("both match", "both match", "100 main st 12345", "100 main st 12345", "US", "US"),
    ]
    
    freq_tracker = CorpusFrequencyTracker()
    all_names = [tc[0] for tc in test_cases] + [tc[1] for tc in test_cases]
    all_addrs = [tc[2] for tc in test_cases] + [tc[3] for tc in test_cases]
    freq_tracker.fit(all_names, [a for a in all_addrs if a])
    
    total_pairs = len(test_cases)
    total_values = total_pairs * len(ULTRA_FEATURE_COLS)
    max_delta = 0.0
    mismatches = 0
    
    for idx, (n1, n2, a1, a2, c1, c2) in enumerate(test_cases):
        s1_ac = extract_structured_address_components(a1) if a1 else None
        tgt_ac = extract_structured_address_components(a2) if a2 else None
        evid = {"name_tfidf": 0.85, "addr_tfidf": 0.75, "blocking": 1.0}
        
        s1_old = PrecomputedEntity(n1, n1.split()[0] if n1 else "", a1, c1, f"S1-{idx}")
        tgt_old = PrecomputedEntity(n2, n2.split()[0] if n2 else "", a2, c2, f"S2-{idx}")
        
        s1_opt = OptimizedEntity(n1, n1.split()[0] if n1 else "", a1, c1, f"S1-{idx}", s1_ac, freq_tracker)
        tgt_opt = OptimizedEntity(n2, n2.split()[0] if n2 else "", a2, c2, f"S2-{idx}", tgt_ac, freq_tracker)
        
        v_old = extract_ultra_features(s1_old, tgt_old, s1_ac or {"postal_code": "", "house_num": "", "digits": set()}, tgt_ac or {"postal_code": "", "house_num": "", "digits": set()}, freq_tracker, evid)
        v_opt = extract_ultra_features_fast(s1_opt, tgt_opt, freq_tracker, evid)
        
        diff = np.abs(np.array(v_old) - np.array(v_opt))
        pair_max = np.max(diff)
        if pair_max > max_delta:
            max_delta = pair_max
        if pair_max > 1e-12:
            mismatches += 1
            bad_idx = np.where(diff > 1e-12)[0]
            for bi in bad_idx:
                print(f"  FAILED on pair {idx} column '{ULTRA_FEATURE_COLS[bi]}': Old={v_old[bi]}, New={v_opt[bi]}")
                
    print(f"  Total pairs tested:            {total_pairs}")
    print(f"  Total feature values compared: {total_values}")
    print(f"  Max absolute delta:            {max_delta:.16f}")
    print(f"  Mismatches (> 1e-12):          {mismatches}")
    
    if mismatches == 0 and max_delta <= 1e-12:
        print("  [✓] PHASE 3 PASSED: 100.000% EXACT NUMERICAL FIDELITY (delta <= 1e-12)")
    else:
        print("  [✗] PHASE 3 FAILED")
        passed_all = False

    # -------------------------------------------------------------------------
    # PHASE 4 & 5: MODEL PREDICTION & CANDIDATE REGRESSION (Validation Cohort)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 4 & 5: MODEL PREDICTION & CANDIDATE REGRESSION ---", flush=True)
    
    # Load 1,000 S1 validation queries
    s1_rows = []
    val_s1_list = []
    with open(TRAIN_SOURCE1, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 4:
                continue
            eid, b_name, b_addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            s1_rows.append((eid, b_name, b_addr, country))
            val_s1_list.append(eid)
            if len(val_s1_list) >= 1000:
                break
                
    val_s1_set = set(val_s1_list)
    val_gt_map = {sid: set() for sid in val_s1_list}
    needed_targets = set()
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0].strip()
            if sid in val_s1_set:
                m_ids = {m.strip() for m in parts[1].split(",") if m.strip()} if len(parts) > 1 and parts[1].strip() else set()
                val_gt_map[sid] = m_ids
                needed_targets.update(m_ids)
                
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    raw_targets = {}
    
    def load_targets(filepath, max_d=50000):
        d_cnt = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, b_name, b_addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not tid:
                    continue
                is_needed = tid in needed_targets
                if is_needed or d_cnt < max_d:
                    if not is_needed:
                        d_cnt += 1
                    n_name = normalize_business_name(b_name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(b_addr) if b_addr else ""
                    raw_targets[tid] = (n_name, c_name, n_addr, country)
                    if country:
                        keys = blocker._extract_all_keys(n_name, c_name, n_addr, b_name)
                        for k in keys:
                            blocker.index[country][k].append(tid)

    load_targets(TRAIN_SOURCE2, 50000)
    load_targets(TRAIN_SOURCE3, 50000)
    blocker.prune_high_frequency_keys()
    
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    retriever.fit_target_corpora(raw_targets)
    
    freq_tracker_val = CorpusFrequencyTracker()
    freq_tracker_val.fit([r[0] for r in raw_targets.values()], [r[2] for r in raw_targets.values()])
    
    target_lookup_opt = {}
    target_lookup_old = {}
    for tid, (rn, rc, ra, rco) in raw_targets.items():
        ac = extract_structured_address_components(ra) if ra else None
        target_lookup_opt[tid] = OptimizedEntity(rn, rc, ra, rco, tid, ac, freq_tracker_val)
        target_lookup_old[tid] = PrecomputedEntity(rn, rc, ra, rco, tid)
        
    s1_dict_opt = {}
    s1_dict_old = {}
    s1_raw_dict = {}
    s1_by_country = defaultdict(list)
    s1_addr_comps = {}
    for eid, name, addr, country in s1_rows:
        n_n = normalize_business_name(name)
        c_n = extract_core_business_name(n_n)
        n_a = normalize_business_address(addr) if addr else ""
        ac = extract_structured_address_components(n_a) if n_a else None
        s1_dict_opt[eid] = OptimizedEntity(n_n, c_n, n_a, country, eid, ac, freq_tracker_val)
        s1_dict_old[eid] = PrecomputedEntity(n_n, c_n, n_a, country, eid)
        s1_raw_dict[eid] = (n_n, c_n, n_a, country)
        s1_addr_comps[eid] = ac or {"postal_code": "", "house_num": "", "digits": set()}
        s1_by_country[country.strip() if country else "UNKNOWN"].append((eid, n_n, c_n, n_a, country))
        
    # Candidate generation
    df_chunk = pl.DataFrame({
        "entity_id": val_s1_list,
        "business_name": [r[1] for r in s1_rows],
        "norm_name": [s1_raw_dict[eid][0] for eid in val_s1_list],
        "core_name": [s1_raw_dict[eid][1] for eid in val_s1_list],
        "norm_address": [s1_raw_dict[eid][2] for eid in val_s1_list],
        "country": [r[3] for r in s1_rows],
    })
    val_stage5 = blocker.generate_candidates_for_s1(df_chunk)
    
    # 1. Batch Vectorized Retrieval
    batch_cands, batch_evid = retriever.retrieve_candidates_batch(s1_by_country, val_stage5)
    
    # 2. Sequential Retrieval (Old)
    seq_cands = {}
    seq_evid = {}
    for sid in val_s1_list:
        rn, rc, ra, rco = s1_raw_dict[sid]
        c_set, evid = retriever.retrieve_candidates(sid, rn, rc, ra, rco, blocking_cands=val_stage5.get(sid, set()))
        seq_cands[sid] = c_set
        seq_evid[sid] = evid
        
    cand_diffs = sum(1 for sid in val_s1_list if batch_cands.get(sid, set()) != seq_cands.get(sid, set()))
    rec_res = evaluate_candidate_recall(val_gt_map, batch_cands)
    print(f"  Candidate Differences:         {cand_diffs} / {len(val_s1_list)}")
    print(f"  Candidate Recall:              {rec_res['candidate_recall_ceiling']*100:.3f}% ({rec_res['captured_matches']}/{rec_res['total_true_matches']})")
    
    if cand_diffs == 0:
        print("  [✓] PHASE 5 PASSED: CANDIDATE SETS 100.0% IDENTICAL")
    else:
        print("  [✗] PHASE 5 FAILED: Candidate discrepancy detected")
        passed_all = False
        
    # 3. Model Prediction Comparison
    model = lgb.Booster(model_file="models/lightgbm_v1_1_ultra.txt")
    postprocessor = SurgicalPostProcessorV1_3(
        base_threshold=0.965,
        s_guard=0.900,
        min_margin=0.020,
        joint_sim_floor=0.450,
        enable_global_consistency=True,
    )
    
    # Build matrices
    feat_old = []
    feat_opt = []
    pair_meta = []
    for sid in val_s1_list:
        s1_o = s1_dict_old[sid]
        s1_n = s1_dict_opt[sid]
        s1_ac = s1_addr_comps[sid]
        cands = batch_cands.get(sid, set())
        evid_map = batch_evid.get(sid, {})
        for tid in cands:
            tgt_o = target_lookup_old.get(tid)
            tgt_n = target_lookup_opt.get(tid)
            if not tgt_o or not tgt_n:
                continue
            tgt_ac = {"postal_code": tgt_n.postal_code, "house_num": tgt_n.house_num, "digits": tgt_n.digits_set}
            evid = evid_map.get(tid, {})
            r_old = extract_ultra_features(s1_o, tgt_o, s1_ac, tgt_ac, freq_tracker_val, evid)
            r_opt = extract_ultra_features_fast(s1_n, tgt_n, freq_tracker_val, evid)
            feat_old.append(r_old)
            feat_opt.append(r_opt)
            pair_meta.append((sid, tid))
            
    X_old = np.array(feat_old, dtype=np.float32)
    X_opt = np.array(feat_opt, dtype=np.float32)
    
    probs_old = model.predict(X_old)
    probs_opt = model.predict(X_opt)
    
    max_prob_diff = np.max(np.abs(probs_old - probs_opt))
    mean_prob_diff = np.mean(np.abs(probs_old - probs_opt))
    print(f"  Scored candidate pairs:        {len(probs_old):,}")
    print(f"  Max probability delta:         {max_prob_diff:.16f}")
    print(f"  Mean probability delta:        {mean_prob_diff:.16f}")
    
    pair_prob_dict_old = defaultdict(list)
    pair_prob_dict_opt = defaultdict(list)
    for (sid, tid), po, pn in zip(pair_meta, probs_old, probs_opt):
        pair_prob_dict_old[sid].append((tid, float(po)))
        pair_prob_dict_opt[sid].append((tid, float(pn)))
        
    preds_old = postprocessor.apply(
        s1_ids=val_s1_list,
        pair_probs=pair_prob_dict_old,
        candidate_sets=batch_cands,
        s1_objects=s1_dict_old,
        target_lookup=target_lookup_old,
        s1_addr_comps=s1_addr_comps,
        tgt_addr_comps={tid: {"postal_code": e.postal_code, "house_num": e.house_num, "digits": e.digits_set} for tid, e in target_lookup_opt.items()},
    )
    
    preds_opt = postprocessor.apply(
        s1_ids=val_s1_list,
        pair_probs=pair_prob_dict_opt,
        candidate_sets=batch_cands,
        s1_objects=s1_dict_opt,
        target_lookup=target_lookup_opt,
        s1_addr_comps=s1_addr_comps,
        tgt_addr_comps={tid: {"postal_code": e.postal_code, "house_num": e.house_num, "digits": e.digits_set} for tid, e in target_lookup_opt.items()},
    )
    
    pred_diffs = sum(1 for sid in val_s1_list if preds_old.get(sid, set()) != preds_opt.get(sid, set()))
    print(f"  Prediction Differences:        {pred_diffs} / {len(val_s1_list)}")
    
    if max_prob_diff <= 1e-6 and pred_diffs == 0:
        print("  [✓] PHASE 4 PASSED: ZERO MODEL PREDICTION DIFFERENCES")
    else:
        print("  [✗] PHASE 4 FAILED")
        passed_all = False

    # -------------------------------------------------------------------------
    # PHASE 7: DATASET COUNT CHECK
    # -------------------------------------------------------------------------
    print("\n--- PHASE 7: DATASET COUNT CHECK ---", flush=True)
    for path, name in [(TEST_SOURCE1, "test_source1.tsv"), (TEST_SOURCE2, "test_source2.tsv"), (TEST_SOURCE3, "test_source3.tsv")]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cnt = sum(1 for _ in f) - 1
                size_mb = os.path.getsize(path) / (1024**2)
                print(f"  {name:<20}: {cnt:10,d} records ({size_mb:8.2f} MB)")
        else:
            print(f"  {name:<20}: [MISSING at {path}]")

    print("\n" + "=" * 85)
    print(f"PRE-FLIGHT STATUS: {'[✓] ALL CHECKS PASSED' if passed_all else '[✗] FAILURES DETECTED'}")
    print("=" * 85)
    return passed_all

if __name__ == "__main__":
    success = run_preflight_suite()
    if not success:
        sys.exit(1)
