import os
import sys
import time
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import polars as pl
import numpy as np
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

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
from src.postprocess import PostProcessor

EXPANDED_FEATURE_COLS = STAGE5_FEATURE_COLS + [
    "addr_hnum_match", "addr_hnum_conflict", "addr_postal_match",
    "addr_postal_conflict", "addr_digits_overlap", "addr_digits_conflict",
    "s1_name_log_freq", "tgt_name_log_freq", "s1_addr_log_freq", "tgt_addr_log_freq",
    "retrieval_views_count", "max_tfidf_score",
]

DELIM = "\t"

def run_test_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "outputs",
    model_path: str = "models/lightgbm_v1_optimized.txt",
    threshold: float = 0.960,
    chunk_size: int = 5000,
    max_s1: Optional[int] = None,
    enable_tfidf: bool = True,
    enable_global_consistency: bool = True,
):
    print("=" * 75, flush=True)
    print("PRODUCTION TEST INFERENCE PIPELINE (V1.0 FINAL) — AMAZON ML CHALLENGE 2026", flush=True)
    print(f"  Test Directory:         {test_dir}", flush=True)
    print(f"  Output Directory:       {output_dir}", flush=True)
    print(f"  Model Artifact:         {model_path}", flush=True)
    print(f"  Decision Threshold:     {threshold:.3f}", flush=True)
    print(f"  Chunk Size:             {chunk_size:,} S1 records", flush=True)
    print(f"  Max S1 Entities:        {'ALL' if max_s1 is None else f'{max_s1:,}'}", flush=True)
    print(f"  Multi-View TF-IDF:      {enable_tfidf}", flush=True)
    print(f"  Global Consistency:     {enable_global_consistency}", flush=True)
    print("=" * 75, flush=True)
    
    t_global_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    test_path = Path(test_dir)
    
    source1_file = test_path / "test_source1.tsv"
    source2_file = test_path / "test_source2.tsv"
    source3_file = test_path / "test_source3.tsv"
    
    # ---------------------------------------------------------
    # 1. Load Trained LightGBM Model
    # ---------------------------------------------------------
    print("\n1. Loading Trained LightGBM Model...", flush=True)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    model_features = model.feature_name()
    is_rich_model = (len(model_features) == len(EXPANDED_FEATURE_COLS))
    print(f"  Loaded model successfully with {len(model_features)} features verified ({'V1.0 Rich 37-Feat' if is_rich_model else 'Stage-5 25-Feat'}).", flush=True)
    
    # ---------------------------------------------------------
    # 2. Stream & Index Target Records (Source 2 & Source 3)
    # ---------------------------------------------------------
    print("\n2. Streaming & Indexing Target Records (Source 2 + Source 3)...", flush=True)
    t0 = time.time()
    
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    target_lookup_fast: Dict[str, PrecomputedEntity] = {}
    target_lookup_raw: Dict[str, Tuple[str, str, str, str]] = {}
    
    total_targets = 0
    
    def process_target_file(filepath: Path, src_name: str):
        nonlocal total_targets
        t_file_start = time.time()
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split(DELIM)
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not tid:
                    continue
                    
                n_name = normalize_business_name(name)
                c_name = extract_core_business_name(n_name)
                n_addr = normalize_business_address(addr) if addr else ""
                
                target_lookup_fast[tid] = PrecomputedEntity(n_name, c_name, n_addr, country, tid)
                if enable_tfidf:
                    target_lookup_raw[tid] = (n_name, c_name, n_addr, country)
                    
                if country:
                    keys = blocker._extract_all_keys(n_name, c_name, n_addr, name)
                    c_idx = blocker.index[country]
                    for k in keys:
                        c_idx[k].append(tid)
                        
                count += 1
                total_targets += 1
                if count % 1000000 == 0:
                    print(f"    Loaded {count:,} records from {src_name}...", flush=True)
        print(f"  {src_name} loaded: {count:,} records in {time.time() - t_file_start:.2f}s", flush=True)

    process_target_file(source2_file, "test_source2.tsv")
    process_target_file(source3_file, "test_source3.tsv")
    
    print(f"  Total target records indexed: {total_targets:,} in {time.time() - t0:.2f}s", flush=True)
    print("  Pruning high-frequency blocking keys...", flush=True)
    blocker.prune_high_frequency_keys()
    
    # Fit Multi-View TF-IDF Retriever
    retriever = None
    if enable_tfidf:
        print("  Fitting Multi-View TF-IDF Retriever matrices...", flush=True)
        t_ret = time.time()
        retriever = MultiViewCandidateRetriever(
            blocker=blocker,
            enable_tfidf_name=True,
            enable_tfidf_addr=True,
            enable_tfidf_char=True,
            top_k_per_view=30,
            max_total_candidates=160,
        )
        retriever.fit_target_corpora(target_lookup_raw)
        print(f"  TF-IDF matrices fitted in {time.time() - t_ret:.2f}s", flush=True)
        # Free raw lookup to conserve RAM
        target_lookup_raw.clear()
        gc.collect()

    # Pre-extract target address components and frequency tracking if rich model
    tgt_addr_comps = {}
    freq_tracker = None
    if is_rich_model:
        print("  Extracting target address components & corpus frequency tracker...", flush=True)
        t_ac = time.time()
        tgt_addr_comps = {tid: extract_structured_address_components(e.norm_addr) for tid, e in target_lookup_fast.items()}
        freq_tracker = CorpusFrequencyTracker()
        all_tgt_names = [e.norm_name for e in target_lookup_fast.values()]
        all_tgt_addrs = [e.norm_addr for e in target_lookup_fast.values()]
        freq_tracker.fit(all_tgt_names, all_tgt_addrs)
        print(f"  Target primitives prepared in {time.time() - t_ac:.2f}s", flush=True)
        
    gc.collect()

    # ---------------------------------------------------------
    # 3. Process Source 1 in Chunks & Stream Results
    # ---------------------------------------------------------
    matching_out_path = Path(output_dir) / "matching_results.tsv"
    candidate_out_path = Path(output_dir) / "candidate_pairs.tsv"
    
    print(f"\n3. Streaming S1 Chunks & Generating Predictions...", flush=True)
    print(f"  Matching Output:  {matching_out_path}", flush=True)
    print(f"  Candidate Output: {candidate_out_path}", flush=True)
    
    f_match = open(matching_out_path, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out_path, "w", encoding="utf-8", newline="")
    
    # Write official headers
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_s1_processed = 0
    total_candidate_pairs = 0
    total_predicted_matches = 0
    total_empty_predictions = 0
    
    cand_counts_sample = []
    chunk_rows = []
    t_inference_start = time.time()
    chunk_idx = 0
    
    postprocessor = PostProcessor(
        base_threshold=threshold,
        min_margin=0.05,
        enable_singleton_guard=True,
        enable_global_consistency=enable_global_consistency,
    )
    
    def process_chunk(rows: List[Tuple[str, str, str, str]]):
        nonlocal total_s1_processed, total_candidate_pairs, total_predicted_matches, total_empty_predictions, chunk_idx
        chunk_idx += 1
        
        s1_ids = []
        s1_objects: Dict[str, PrecomputedEntity] = {}
        s1_raw_tuples: Dict[str, Tuple[str, str, str, str]] = {}
        
        for eid, name, addr, country in rows:
            s1_ids.append(eid)
            n_n = normalize_business_name(name)
            c_n = extract_core_business_name(n_n)
            n_a = normalize_business_address(addr) if addr else ""
            s1_objects[eid] = PrecomputedEntity(n_n, c_n, n_a, country, eid)
            s1_raw_tuples[eid] = (n_n, c_n, n_a, country)
            
        df_chunk = pl.DataFrame({
            "entity_id": s1_ids,
            "business_name": [r[1] for r in rows],
            "norm_name": [s1_raw_tuples[eid][0] for eid in s1_ids],
            "core_name": [s1_raw_tuples[eid][1] for eid in s1_ids],
            "norm_address": [s1_raw_tuples[eid][2] for eid in s1_ids],
            "country": [r[3] for r in rows],
        })
        
        # 1. Candidate Generation (Stage-5 Blocker + Multi-View TF-IDF Union)
        stage5_cands = blocker.generate_candidates_for_s1(df_chunk)
        
        final_cands_dict: Dict[str, Set[str]] = {}
        ret_evidence_dict: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        for eid in s1_ids:
            s5_set = stage5_cands.get(eid, set())
            if retriever is not None:
                rn, rc, ra, rco = s1_raw_tuples[eid]
                c_set, evid = retriever.retrieve_candidates(eid, rn, rc, ra, rco, blocking_cands=s5_set)
                final_cands_dict[eid] = c_set
                ret_evidence_dict[eid] = evid
            else:
                final_cands_dict[eid] = s5_set
                ret_evidence_dict[eid] = {}
                
        # 2. Vectorized Feature Extraction
        feat_matrix_rows = []
        pair_meta = []
        
        for s1_id in s1_ids:
            s1_obj = s1_objects[s1_id]
            s1_ac = extract_structured_address_components(s1_obj.norm_addr) if is_rich_model else None
            cand_set = final_cands_dict.get(s1_id, set())
            c_len = len(cand_set)
            cand_counts_sample.append(c_len)
            total_candidate_pairs += c_len
            
            evid_map = ret_evidence_dict.get(s1_id, {})
            
            for tid in cand_set:
                tgt_obj = target_lookup_fast.get(tid)
                if not tgt_obj:
                    continue
                base_feats = extract_pairwise_features_fast(s1_obj, tgt_obj)
                
                if is_rich_model:
                    tgt_ac = tgt_addr_comps.get(tid, extract_structured_address_components(""))
                    ac_feats = compute_address_component_features(s1_ac, tgt_ac)
                    s1_nf = freq_tracker.get_name_freq_feature(s1_obj.norm_name)
                    tgt_nf = freq_tracker.get_name_freq_feature(tgt_obj.norm_name)
                    s1_af = freq_tracker.get_addr_freq_feature(s1_obj.norm_addr)
                    tgt_af = freq_tracker.get_addr_freq_feature(tgt_obj.norm_addr)
                    
                    evid = evid_map.get(tid, {})
                    ret_views = float(len(evid)) if evid else 1.0
                    max_tfidf = max([v for k, v in evid.items() if k != "blocking"], default=0.0)
                    
                    full_row = base_feats + [
                        ac_feats["addr_hnum_match"], ac_feats["addr_hnum_conflict"],
                        ac_feats["addr_postal_match"], ac_feats["addr_postal_conflict"],
                        ac_feats["addr_digits_overlap"], ac_feats["addr_digits_conflict"],
                        s1_nf, tgt_nf, s1_af, tgt_af,
                        ret_views, max_tfidf,
                    ]
                    feat_matrix_rows.append(full_row)
                else:
                    feat_matrix_rows.append(base_feats)
                    
                pair_meta.append((s1_id, tid))
                
        # 3. Model Prediction
        pair_prob_dict = defaultdict(list)
        if feat_matrix_rows:
            X_chunk = np.array(feat_matrix_rows, dtype=np.float32)
            probs = model.predict(X_chunk)
            for (s1_id, tid), prob in zip(pair_meta, probs):
                pair_prob_dict[s1_id].append((tid, float(prob)))
                
        # 4. Post-Processing Decision Layer (Singleton Guard + Global Consistency)
        final_matched_dict = postprocessor.apply(s1_ids, pair_prob_dict, final_cands_dict)
        
        # 5. Write Results Incrementally
        for s1_id in s1_ids:
            cand_set = final_cands_dict.get(s1_id, set())
            matched_set = final_matched_dict.get(s1_id, set())
            
            cand_str = ",".join(sorted(cand_set))
            match_str = ",".join(sorted(matched_set))
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            if not matched_set:
                total_empty_predictions += 1
            else:
                total_predicted_matches += len(matched_set)
                
        total_s1_processed += len(s1_ids)
        
        elapsed = time.time() - t_inference_start
        rate = total_s1_processed / elapsed if elapsed > 0 else 0.0
        if chunk_idx % 10 == 0 or total_s1_processed == len(rows):
            print(f"    Chunk {chunk_idx:4d} | Processed: {total_s1_processed:8,d} S1s | Throughput: {rate:6.1f} S1/s | Elapsed: {elapsed:6.1f}s | Matches: {total_predicted_matches:8,d}", flush=True)
            f_match.flush()
            f_cand.flush()

    # Read Source 1 line by line
    with open(source1_file, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip("\r\n").split(DELIM)
            if len(parts) != 4:
                continue
            eid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            if not eid:
                continue
            chunk_rows.append((eid, name, addr, country))
            
            if len(chunk_rows) >= chunk_size:
                process_chunk(chunk_rows)
                chunk_rows = []
                
            if max_s1 is not None and total_s1_processed + len(chunk_rows) >= max_s1:
                if chunk_rows:
                    process_chunk(chunk_rows)
                    chunk_rows = []
                break
                
        if chunk_rows and (max_s1 is None or total_s1_processed < max_s1):
            process_chunk(chunk_rows)
            chunk_rows = []
            
    f_match.close()
    f_cand.close()
    
    t_total = time.time() - t_global_start
    t_inference = time.time() - t_inference_start
    throughput = total_s1_processed / t_inference if t_inference > 0 else 0.0
    
    cand_arr = np.array(cand_counts_sample) if cand_counts_sample else np.array([0])
    avg_cand = float(np.mean(cand_arr))
    median_cand = float(np.median(cand_arr))
    p95_cand = float(np.percentile(cand_arr, 95))
    p99_cand = float(np.percentile(cand_arr, 99))
    max_cand = int(np.max(cand_arr))
    
    print("\n" + "=" * 75, flush=True)
    print("INFERENCE SUMMARY:", flush=True)
    print(f"  S1 Records Processed:      {total_s1_processed:,}", flush=True)
    print(f"  Total Candidate Pairs:     {total_candidate_pairs:,}", flush=True)
    print(f"  Average Candidates / S1:   {avg_cand:.2f}", flush=True)
    print(f"  Median Candidates / S1:    {median_cand:.1f}", flush=True)
    print(f"  P95 Candidates / S1:       {p95_cand:.1f}", flush=True)
    print(f"  P99 Candidates / S1:       {p99_cand:.1f}", flush=True)
    print(f"  Maximum Candidates / S1:   {max_cand}", flush=True)
    print(f"  Total Predicted Matches:   {total_predicted_matches:,}", flush=True)
    print(f"  Total Empty Predictions:   {total_empty_predictions:,} ({total_empty_predictions/total_s1_processed*100:.2f}%)", flush=True)
    print(f"  Average Matches / S1:      {total_predicted_matches/total_s1_processed:.3f}", flush=True)
    print(f"  Inference Time:            {t_inference:.2f}s ({throughput:.1f} S1/sec)", flush=True)
    print(f"  Total Execution Time:      {t_total:.2f}s", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Test Inference for Entity Resolution")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--model-path", default="models/lightgbm_v1_optimized.txt", help="Path to trained LightGBM model")
    parser.add_argument("--threshold", type=float, default=0.960, help="Decision threshold for match prediction")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for S1 streaming")
    parser.add_argument("--max-s1", type=int, default=None, help="Maximum number of S1 entities to process (for dry-run)")
    parser.add_argument("--no-tfidf", action="store_true", help="Disable multi-view TF-IDF candidate retrieval")
    parser.add_argument("--no-global-consistency", action="store_true", help="Disable global target consistency post-processing")
    args = parser.parse_args()
    
    run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        max_s1=args.max_s1,
        enable_tfidf=not args.no_tfidf,
        enable_global_consistency=not args.no_global_consistency,
    )
