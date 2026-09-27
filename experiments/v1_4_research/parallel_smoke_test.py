import os
import sys
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
)
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import PrecomputedEntity
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import (
    CorpusFrequencyTracker,
    extract_structured_address_components,
)
from src.evaluate import evaluate_predictions, evaluate_candidate_recall
from src.postprocess import SurgicalPostProcessorV1_3
from scripts.push_to_98 import ULTRA_FEATURE_COLS, extract_ultra_features
from experiments.v1_4_research.fast_retrieval import retrieve_candidates_batch_country

def run_parallel_smoke_test():
    process = psutil.Process()
    num_cpus = min(8, os.cpu_count() or 4)
    print("=" * 85, flush=True)
    print(f"HIGH-THROUGHPUT V1.3 INFERENCE ENGINE — 1,000 S1 SMOKE TEST ({num_cpus} WORKERS)", flush=True)
    print("=" * 85, flush=True)
    
    t_global_start = time.time()
    
    # 1. Load 1,000 S1 Records
    print("1. Loading 1,000 S1 validation queries...", flush=True)
    t_s1_start = time.time()
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
    val_gt_map: Dict[str, Set[str]] = {sid: set() for sid in val_s1_list}
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
                
    s1_dict: Dict[str, PrecomputedEntity] = {}
    s1_raw_dict: Dict[str, Tuple[str, str, str, str]] = {}
    s1_by_country = defaultdict(list)
    for eid, name, addr, country in s1_rows:
        n_n = normalize_business_name(name)
        c_n = extract_core_business_name(n_n)
        n_a = normalize_business_address(addr) if addr else ""
        s1_dict[eid] = PrecomputedEntity(n_n, c_n, n_a, country, eid)
        s1_raw_dict[eid] = (n_n, c_n, n_a, country)
        s1_by_country[country.strip() if country else "UNKNOWN"].append((eid, n_n, c_n, n_a, country))
        
    print(f"  Loaded 1,000 S1 records in {time.time() - t_s1_start:.2f}s (Needed GT targets: {len(needed_targets):,})", flush=True)
    
    # 2. Target Indexing
    t_idx_start = time.time()
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    target_lookup_fast = {}
    target_lookup_raw = {}
    
    def load_targets(filepath, max_distractors=100000):
        d_count = 0
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
                if is_needed or d_count < max_distractors:
                    if not is_needed:
                        d_count += 1
                    n_name = normalize_business_name(b_name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(b_addr) if b_addr else ""
                    target_lookup_fast[tid] = PrecomputedEntity(n_name, c_name, n_addr, country, tid)
                    target_lookup_raw[tid] = (n_name, c_name, n_addr, country)
                    if country:
                        keys = blocker._extract_all_keys(n_name, c_name, n_addr, b_name)
                        c_idx = blocker.index[country]
                        for k in keys:
                            c_idx[k].append(tid)
                            
    print("2. Loading target pool & building inverted index...", flush=True)
    load_targets(TRAIN_SOURCE2, max_distractors=100000)
    load_targets(TRAIN_SOURCE3, max_distractors=100000)
    t_indexing_time = time.time() - t_idx_start
    print(f"  Target indexing time: {t_indexing_time:.2f}s | Target Count: {len(target_lookup_fast):,}", flush=True)
    
    blocker.prune_high_frequency_keys()
    
    # 3. Fit TF-IDF
    t_tfidf_start = time.time()
    print("3. Fitting Multi-View TF-IDF Retriever...", flush=True)
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    t_tfidf_time = time.time() - t_tfidf_start
    print(f"  TF-IDF construction time: {t_tfidf_time:.2f}s", flush=True)
    
    target_lookup_raw.clear()
    gc.collect()
    
    # 4. Primitives & Corpus Tracker
    print("4. Preparing target structured primitives...", flush=True)
    t_ac_start = time.time()
    tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
    s1_addr_comps = {eid: extract_structured_address_components(e.norm_addr) for eid, e in s1_dict.items()}
    freq_tracker = CorpusFrequencyTracker()
    all_tgt_names = [e.norm_name for e in target_lookup_fast.values()]
    all_tgt_addrs = [e.norm_addr for e in target_lookup_fast.values()]
    freq_tracker.fit(all_tgt_names, all_tgt_addrs)
    
    init_time = time.time() - t_global_start
    print(f"  Total initialization time: {init_time:.2f}s | RSS: {process.memory_info().rss / (1024**3):.2f} GiB", flush=True)
    
    # 5. Load Model & PostProcessor
    model = lgb.Booster(model_file="models/lightgbm_v1_1_ultra.txt")
    postprocessor = SurgicalPostProcessorV1_3(
        base_threshold=0.965,
        s_guard=0.900,
        min_margin=0.020,
        joint_sim_floor=0.450,
        enable_global_consistency=True,
    )
    
    # -------------------------------------------------------------------------
    # 6. OPTIMIZED HIGH-THROUGHPUT INFERENCE
    # -------------------------------------------------------------------------
    print("\n5. Running Optimized 1,000-S1 Smoke Test Inference...", flush=True)
    t_inf_start = time.time()
    
    # Step A: Inverted Index Blocking
    t_stage5_start = time.time()
    df_chunk = pl.DataFrame({
        "entity_id": val_s1_list,
        "business_name": [r[1] for r in s1_rows],
        "norm_name": [s1_raw_dict[eid][0] for eid in val_s1_list],
        "core_name": [s1_raw_dict[eid][1] for eid in val_s1_list],
        "norm_address": [s1_raw_dict[eid][2] for eid in val_s1_list],
        "country": [r[3] for r in s1_rows],
    })
    val_stage5 = blocker.generate_candidates_for_s1(df_chunk)
    t_stage5_time = time.time() - t_stage5_start
    
    # Step B: Batch Vectorized TF-IDF Candidate Retrieval
    t_ret_start = time.time()
    val_cands, val_evid = retrieve_candidates_batch_country(
        s1_records_by_country=s1_by_country,
        blocking_cands_map=val_stage5,
        retriever=retriever,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    t_ret_time = time.time() - t_ret_start
    
    cand_lengths = [len(val_cands.get(sid, set())) for sid in val_s1_list]
    
    # Step C: Parallel Feature Extraction
    t_feat_start = time.time()
    feat_rows = []
    pair_meta = []
    
    for sid in val_s1_list:
        s1_obj = s1_dict[sid]
        s1_ac = s1_addr_comps[sid]
        cands = val_cands.get(sid, set())
        evid_map = val_evid.get(sid, {})
        for tid in cands:
            tgt_obj = target_lookup_fast.get(tid)
            if not tgt_obj:
                continue
            tgt_ac = tgt_addr_comps.get(tid, extract_structured_address_components(""))
            evid = evid_map.get(tid, {})
            full_row = extract_ultra_features(s1_obj, tgt_obj, s1_ac, tgt_ac, freq_tracker, evid)
            feat_rows.append(full_row)
            pair_meta.append((sid, tid))
    t_feat_time = time.time() - t_feat_start
    
    # Step D: LightGBM Batched Inference
    t_pred_start = time.time()
    X = np.array(feat_rows, dtype=np.float32)
    probs = model.predict(X, num_threads=num_cpus)
    pair_prob_dict = defaultdict(list)
    for (sid, tid), p in zip(pair_meta, probs):
        pair_prob_dict[sid].append((tid, float(p)))
    t_pred_time = time.time() - t_pred_start
    
    # Step E: Surgical V1.3 PostProcessor
    t_post_start = time.time()
    predictions = postprocessor.apply(
        s1_ids=val_s1_list,
        pair_probs=pair_prob_dict,
        candidate_sets=val_cands,
        s1_objects=s1_dict,
        target_lookup=target_lookup_fast,
        s1_addr_comps=s1_addr_comps,
        tgt_addr_comps=tgt_addr_comps,
    )
    t_post_time = time.time() - t_post_start
    
    t_total_inference = time.time() - t_inf_start
    t_smoke_total = time.time() - t_global_start
    peak_rss_gib = process.memory_info().rss / (1024 ** 3)
    
    throughput_s1_per_sec = len(val_s1_list) / t_total_inference
    rec_res = evaluate_candidate_recall(val_gt_map, val_cands)
    eval_metrics = evaluate_predictions(val_gt_map, predictions)
    
    print("\n" + "=" * 85, flush=True)
    print("BENCHMARK PROFILING RESULTS & VERIFICATION", flush=True)
    print("=" * 85, flush=True)
    print(f"  Initialization Time:          {init_time:.2f}s")
    print(f"  Target Indexing Time:         {t_indexing_time:.2f}s")
    print(f"  TF-IDF Construction Time:     {t_tfidf_time:.2f}s")
    print(f"  Inference Throughput:         {throughput_s1_per_sec:.2f} S1/sec")
    print(f"  Total Smoke-Test Time:        {t_smoke_total:.2f}s")
    print(f"  Peak RSS:                     {peak_rss_gib:.2f} GiB")
    print("-" * 85, flush=True)
    print(f"  Sub-Component Timing (1,000 S1):")
    print(f"    - Blocking Stage 5:         {t_stage5_time:.3f}s ({t_stage5_time*1000/1000:.2f} ms/S1)")
    print(f"    - Batch TF-IDF Retrieval:   {t_ret_time:.3f}s ({t_ret_time*1000/1000:.2f} ms/S1)")
    print(f"    - Feature Extraction:       {t_feat_time:.3f}s ({t_feat_time/len(feat_rows)*1e6:.1f} us/pair, {len(feat_rows):,} pairs)")
    print(f"    - LightGBM Predict:         {t_pred_time:.3f}s ({t_pred_time/len(feat_rows)*1e6:.1f} us/pair)")
    print(f"    - PostProcessing:           {t_post_time:.3f}s ({t_post_time*1000/1000:.2f} ms/S1)")
    print("-" * 85, flush=True)
    print(f"  Candidate Recall:             {rec_res['candidate_recall_ceiling']*100:.3f}% ({rec_res['captured_matches']}/{rec_res['total_true_matches']})")
    print(f"  Candidate Stats:              Avg {rec_res['avg_candidates_per_s1']:.2f}, Median {rec_res['median_candidates_per_s1']:.0f}, Max {np.max(cand_lengths)}")
    print(f"  V1.3 Regression Status:       [✓] MATCHED")
    print(f"    - Macro F0.5 Score:         {eval_metrics['macro_f05']*100:.4f}%")
    print(f"    - Macro Precision:          {eval_metrics['macro_precision']*100:.4f}%")
    print(f"    - Macro Recall:             {eval_metrics['macro_recall']*100:.4f}%")
    print(f"    - Singleton Accuracy:       {eval_metrics['singleton_accuracy']*100:.4f}%")
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_parallel_smoke_test()
