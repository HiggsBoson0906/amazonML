import os
import sys
import time
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

import polars as pl
import numpy as np
import lightgbm as lgb

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features

FEATURE_COLS = [
    "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
    "name_token_sort", "name_token_set", "name_partial", "core_ratio",
    "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
    "name_len_diff", "addr_missing", "addr_exact", "addr_ratio",
    "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_char_3gram",
    "addr_num_match", "addr_len_diff", "country_match", "is_source2", "is_source3",
]

DELIM = "\t"

def run_test_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "outputs",
    model_path: str = "models/lightgbm_stage5.txt",
    threshold: float = 0.980,
    chunk_size: int = 5000,
    max_s1: int = None,
):
    print("=" * 75)
    print("PRODUCTION TEST INFERENCE PIPELINE — AMAZON ML CHALLENGE 2026")
    print(f"  Test Directory:     {test_dir}")
    print(f"  Output Directory:   {output_dir}")
    print(f"  Model Artifact:     {model_path}")
    print(f"  Decision Threshold: {threshold:.3f}")
    print(f"  Chunk Size:         {chunk_size:,} S1 records")
    print(f"  Max S1 Entities:    {'ALL' if max_s1 is None else f'{max_s1:,}'}")
    print("=" * 75)
    
    t_global_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    test_path = Path(test_dir)
    
    source1_file = test_path / "test_source1.tsv"
    source2_file = test_path / "test_source2.tsv"
    source3_file = test_path / "test_source3.tsv"
    
    # ---------------------------------------------------------
    # 1. Load Trained LightGBM Model
    # ---------------------------------------------------------
    print("\n1. Loading Trained LightGBM Model...")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    model_features = model.feature_name()
    assert model_features == FEATURE_COLS, f"Feature mismatch! Expected {FEATURE_COLS}, got {model_features}"
    print(f"  Loaded model successfully with {len(model_features)} features verified.")
    
    # ---------------------------------------------------------
    # 2. Stream & Index Target Records (Source 2 & Source 3)
    # ---------------------------------------------------------
    print("\n2. Streaming & Indexing Target Records (Source 2 + Source 3)...")
    t0 = time.time()
    
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    target_lookup = {}  # tid -> (norm_name, core_name, norm_addr, country)
    
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
                
                target_lookup[tid] = (n_name, c_name, n_addr, country)
                
                if country:
                    keys = blocker._extract_all_keys(n_name, c_name, n_addr, name)
                    c_idx = blocker.index[country]
                    for k in keys:
                        c_idx[k].append(tid)
                        
                count += 1
                total_targets += 1
                if count % 1000000 == 0:
                    print(f"    Loaded {count:,} records from {src_name}...")
        print(f"  {src_name} loaded: {count:,} records in {time.time() - t_file_start:.2f}s")

    process_target_file(source2_file, "test_source2.tsv")
    process_target_file(source3_file, "test_source3.tsv")
    
    print(f"  Total target records indexed: {total_targets:,} in {time.time() - t0:.2f}s")
    
    print("  Pruning high-frequency blocking keys...")
    t_prune = time.time()
    blocker.prune_high_frequency_keys()
    print(f"  Pruning completed in {time.time() - t_prune:.2f}s")
    gc.collect()
    
    # ---------------------------------------------------------
    # 3. Process Source 1 in Chunks & Stream Results
    # ---------------------------------------------------------
    matching_out_path = Path(output_dir) / "matching_results.tsv"
    candidate_out_path = Path(output_dir) / "candidate_pairs.tsv"
    
    print(f"\n3. Streaming S1 Chunks & Generating Predictions...")
    print(f"  Matching Output:  {matching_out_path}")
    print(f"  Candidate Output: {candidate_out_path}")
    
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
    
    def process_chunk(rows: List[Tuple[str, str, str, str]]):
        nonlocal total_s1_processed, total_candidate_pairs, total_predicted_matches, total_empty_predictions, chunk_idx
        chunk_idx += 1
        t_chunk_start = time.time()
        
        # 1. Normalize S1 rows in chunk
        s1_ids = []
        norm_names = []
        core_names = []
        raw_names = []
        norm_addrs = []
        countries = []
        
        for eid, name, addr, country in rows:
            s1_ids.append(eid)
            raw_names.append(name)
            n_n = normalize_business_name(name)
            norm_names.append(n_n)
            core_names.append(extract_core_business_name(n_n))
            norm_addrs.append(normalize_business_address(addr) if addr else "")
            countries.append(country)
            
        df_chunk = pl.DataFrame({
            "entity_id": s1_ids,
            "business_name": raw_names,
            "norm_name": norm_names,
            "core_name": core_names,
            "norm_address": norm_addrs,
            "country": countries,
        })
        
        # 2. Generate candidates for S1 chunk
        cands_dict = blocker.generate_candidates_for_s1(df_chunk)
        
        # 3. Prepare feature extraction rows
        feat_matrix_rows = []
        pair_meta = []  # (s1_id, tid)
        
        for s1_id, n_name, c_name, n_addr, country in zip(s1_ids, norm_names, core_names, norm_addrs, countries):
            cand_set = cands_dict.get(s1_id, set())
            c_len = len(cand_set)
            cand_counts_sample.append(c_len)
            total_candidate_pairs += c_len
            
            for tid in cand_set:
                if tid not in target_lookup:
                    continue
                tgt_n_name, tgt_c_name, tgt_n_addr, tgt_country = target_lookup[tid]
                feats = extract_pairwise_features(
                    n_name, c_name, n_addr, country,
                    tgt_n_name, tgt_c_name, tgt_n_addr, tgt_country,
                    tid,
                )
                feat_matrix_rows.append([feats[c] for c in FEATURE_COLS])
                pair_meta.append((s1_id, tid))
                
        # 4. Predict with LightGBM in vectorized batch
        matched_per_s1 = defaultdict(list)
        if feat_matrix_rows:
            X_chunk = np.array(feat_matrix_rows, dtype=np.float32)
            probs = model.predict(X_chunk)
            
            for (s1_id, tid), prob in zip(pair_meta, probs):
                if prob >= threshold:
                    matched_per_s1[s1_id].append(tid)
                    
        # 5. Write to output files
        for s1_id in s1_ids:
            cand_set = cands_dict.get(s1_id, set())
            matched_list = matched_per_s1.get(s1_id, [])
            
            cand_str = ",".join(sorted(cand_set))
            match_str = ",".join(sorted(matched_list))
            
            f_cand.write(f"{s1_id}\t{cand_str}\n")
            f_match.write(f"{s1_id}\t{match_str}\n")
            
            if not matched_list:
                total_empty_predictions += 1
            else:
                total_predicted_matches += len(matched_list)
                
        total_s1_processed += len(s1_ids)
        
        # Periodic Progress Logging
        elapsed = time.time() - t_inference_start
        rate = total_s1_processed / elapsed if elapsed > 0 else 0.0
        if chunk_idx % 10 == 0 or total_s1_processed == len(rows):
            print(f"    Chunk {chunk_idx:4d} | Processed: {total_s1_processed:8,d} S1s | Throughput: {rate:6.1f} S1/s | Elapsed: {elapsed:6.1f}s | Matches: {total_predicted_matches:8,d}")
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
    
    # Calculate Candidate Statistics
    cand_arr = np.array(cand_counts_sample) if cand_counts_sample else np.array([0])
    avg_cand = float(np.mean(cand_arr))
    median_cand = float(np.median(cand_arr))
    p95_cand = float(np.percentile(cand_arr, 95))
    p99_cand = float(np.percentile(cand_arr, 99))
    max_cand = int(np.max(cand_arr))
    
    print("\n" + "=" * 75)
    print("INFERENCE SUMMARY:")
    print(f"  S1 Records Processed:      {total_s1_processed:,}")
    print(f"  Total Candidate Pairs:     {total_candidate_pairs:,}")
    print(f"  Average Candidates / S1:   {avg_cand:.2f}")
    print(f"  Median Candidates / S1:    {median_cand:.1f}")
    print(f"  P95 Candidates / S1:       {p95_cand:.1f}")
    print(f"  P99 Candidates / S1:       {p99_cand:.1f}")
    print(f"  Maximum Candidates / S1:   {max_cand}")
    print(f"  Total Predicted Matches:   {total_predicted_matches:,}")
    print(f"  Total Empty Predictions:   {total_empty_predictions:,} ({total_empty_predictions/total_s1_processed*100:.2f}%)")
    print(f"  Average Matches / S1:      {total_predicted_matches/total_s1_processed:.3f}")
    print(f"  Inference Time:            {t_inference:.2f}s ({throughput:.1f} S1/sec)")
    print(f"  Total Execution Time:      {t_total:.2f}s")
    print("=" * 75)
    
    return {
        "s1_processed": total_s1_processed,
        "candidate_pairs": total_candidate_pairs,
        "avg_candidates": avg_cand,
        "median_candidates": median_cand,
        "p95_candidates": p95_cand,
        "p99_candidates": p99_cand,
        "max_candidates": max_cand,
        "predicted_matches": total_predicted_matches,
        "empty_predictions": total_empty_predictions,
        "avg_matches_per_s1": total_predicted_matches / total_s1_processed,
        "inference_time": t_inference,
        "throughput": throughput,
        "matching_path": str(matching_out_path),
        "candidate_path": str(candidate_out_path),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Test Inference for Entity Resolution")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--model-path", default="models/lightgbm_stage5.txt", help="Path to trained LightGBM model")
    parser.add_argument("--threshold", type=float, default=0.980, help="Decision threshold for match prediction")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for S1 streaming")
    parser.add_argument("--max-s1", type=int, default=None, help="Maximum number of S1 entities to process (for dry-run)")
    args = parser.parse_args()
    
    run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        max_s1=args.max_s1,
    )
