import os
import sys
import time
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, LCSseq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import (
    STAGE5_FEATURE_COLS,
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
from src.postprocess import SurgicalPostProcessorV1_3

DELIM = "\t"

def run_test_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "outputs",
    model_path: str = "models/lightgbm_v1_1_ultra.txt",
    threshold: float = 0.965,
    s_guard: float = 0.900,
    min_margin: float = 0.020,
    joint_sim_floor: float = 0.450,
    chunk_size: int = 5000,
    max_s1: Optional[int] = None,
    enable_tfidf: bool = True,
    enable_global_consistency: bool = True,
):
    print("=" * 80, flush=True)
    print("PRODUCTION TEST INFERENCE PIPELINE (V1.3 SURGICAL OPTIMIZED) — AMAZON ML 2026", flush=True)
    print("=" * 80, flush=True)
    print(f"  Test Directory:         {test_dir}", flush=True)
    print(f"  Output Directory:       {output_dir}", flush=True)
    print(f"  Model Artifact:         {model_path}", flush=True)
    print(f"  Base Threshold:         {threshold:.3f}", flush=True)
    print(f"  Singleton Guard:        {s_guard:.3f}", flush=True)
    print(f"  Minimum Margin:         {min_margin:.3f}", flush=True)
    print(f"  Joint Sim Floor:        {joint_sim_floor:.3f}", flush=True)
    print(f"  Feature Dimensions:     {len(ULTRA_FEATURE_COLS)} features", flush=True)
    print(f"  Chunk Size:             {chunk_size:,} S1 records", flush=True)
    print(f"  Max S1 Entities:        {'ALL' if max_s1 is None else f'{max_s1:,}'}", flush=True)
    print(f"  Multi-View TF-IDF:      {enable_tfidf} (Top-45 per view, capacity 200)", flush=True)
    print(f"  Global Consistency:     {enable_global_consistency}", flush=True)
    print("=" * 80, flush=True)
    
    t_global_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    test_path = Path(test_dir)
    
    source1_file = test_path / "test_source1.tsv"
    source2_file = test_path / "test_source2.tsv"
    source3_file = test_path / "test_source3.tsv"
    
    # -------------------------------------------------------------------------
    # 1. LOAD TRAINED LIGHTGBM MODEL & PRODUCTION SELF-CHECK
    # -------------------------------------------------------------------------
    print("\n1. Loading Trained LightGBM Model & Running Self-Check...", flush=True)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"V1.3 Model file not found at: {model_path}")
    
    model = lgb.Booster(model_file=str(model_path))
    model_features = model.feature_name()
    print(f"  Loaded model successfully with {len(model_features)} features.", flush=True)
    
    # Strict Self-Check Assertions
    if len(model_features) != len(ULTRA_FEATURE_COLS):
        raise ValueError(
            f"Model feature count mismatch: expected {len(ULTRA_FEATURE_COLS)} (V1.3 Ultra), "
            f"but found {len(model_features)} in '{model_path}'. "
            f"Ensure models/lightgbm_v1_1_ultra.txt is used!"
        )
    print("  [✓] V1.3 Feature schema verified against production config.", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. STREAM & INDEX TARGET RECORDS (SOURCE 2 & SOURCE 3)
    # -------------------------------------------------------------------------
    print("\n2. Streaming & Indexing Target Records (Source 2 + Source 3)...", flush=True)
    t0 = time.time()
    
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    freq_tracker = CorpusFrequencyTracker()
    raw_targets: Dict[str, Tuple[str, str, str, str]] = {}
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
                
                raw_targets[tid] = (n_name, c_name, n_addr, country)
                
                # Stream frequency statistics on the fly (zero intermediate lists)
                if n_name:
                    freq_tracker.name_counts[n_name] += 1
                    for t in set(n_name.split()):
                        freq_tracker.token_doc_counts[t] += 1
                if n_addr:
                    freq_tracker.addr_counts[n_addr] += 1
                freq_tracker.total_docs += 1
                    
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
    
    # Fit Multi-View TF-IDF Retriever (V1.3 Ultra Settings: Top-45, cap 200)
    retriever = None
    if enable_tfidf:
        print("  Fitting Multi-View TF-IDF Retriever matrices (Top-45 per view, capacity 200)...", flush=True)
        t_ret = time.time()
        retriever = MultiViewCandidateRetriever(
            blocker=blocker,
            enable_tfidf_name=True,
            enable_tfidf_addr=True,
            enable_tfidf_char=True,
            top_k_per_view=45,
            max_total_candidates=200,
        )
        retriever.fit_target_corpora(raw_targets)
        print(f"  TF-IDF matrices fitted in {time.time() - t_ret:.2f}s", flush=True)

    print("  [✓] Memory-safe target indexing and frequency tracking ready (zero object bloat).", flush=True)
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. STREAM S1 IN CHUNKS & GENERATE PREDICTIONS
    # -------------------------------------------------------------------------
    matching_out_path = Path(output_dir) / "matching_results.tsv"
    candidate_out_path = Path(output_dir) / "candidate_pairs.tsv"
    
    print(f"\n3. Streaming S1 Chunks & Generating V1.3 Predictions...", flush=True)
    print(f"  Matching Output:  {matching_out_path}", flush=True)
    print(f"  Candidate Output: {candidate_out_path}", flush=True)
    
    f_match = open(matching_out_path, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out_path, "w", encoding="utf-8", newline="")
    
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
    
    # Exact V1.3 Surgical PostProcessor
    postprocessor = SurgicalPostProcessorV1_3(
        base_threshold=threshold,
        s_guard=s_guard,
        min_margin=min_margin,
        joint_sim_floor=joint_sim_floor,
        enable_global_consistency=enable_global_consistency,
    )
    
    def process_chunk(rows: List[Tuple[str, str, str, str]]):
        nonlocal total_s1_processed, total_candidate_pairs, total_predicted_matches, total_empty_predictions, chunk_idx
        chunk_idx += 1
        
        s1_ids = []
        s1_objects: Dict[str, OptimizedEntity] = {}
        s1_raw_tuples: Dict[str, Tuple[str, str, str, str]] = {}
        s1_by_country = defaultdict(list)
        s1_addr_comps_chunk: Dict[str, Any] = {}
        
        for eid, name, addr, country in rows:
            s1_ids.append(eid)
            n_n = normalize_business_name(name)
            c_n = extract_core_business_name(n_n)
            n_a = normalize_business_address(addr) if addr else ""
            ac = extract_structured_address_components(n_a) if n_a else None
            s1_objects[eid] = OptimizedEntity(
                norm_name=n_n,
                core_name=c_n,
                norm_addr=n_a,
                country=country,
                src_id=eid,
                ac=ac,
                freq_tracker=freq_tracker
            )
            s1_raw_tuples[eid] = (n_n, c_n, n_a, country)
            s1_by_country[country.strip() if country else "UNKNOWN"].append((eid, n_n, c_n, n_a, country))
            s1_addr_comps_chunk[eid] = ac or {"postal_code": "", "house_num": "", "digits": set()}
            
        df_chunk = pl.DataFrame({
            "entity_id": s1_ids,
            "business_name": [r[1] for r in rows],
            "norm_name": [s1_raw_tuples[eid][0] for eid in s1_ids],
            "core_name": [s1_raw_tuples[eid][1] for eid in s1_ids],
            "norm_address": [s1_raw_tuples[eid][2] for eid in s1_ids],
            "country": [r[3] for r in rows],
        })
        
        stage5_cands = blocker.generate_candidates_for_s1(df_chunk)
        
        if retriever is not None:
            final_cands_dict, ret_evidence_dict = retriever.retrieve_candidates_batch(
                s1_by_country=s1_by_country,
                blocking_cands_map=stage5_cands
            )
        else:
            final_cands_dict = stage5_cands
            ret_evidence_dict = {}
                
        # Materialize OptimizedEntity ONLY for the candidate targets in this active chunk
        chunk_tids = {tid for c_set in final_cands_dict.values() for tid in c_set}
        chunk_target_objects: Dict[str, OptimizedEntity] = {}
        for tid in chunk_tids:
            tgt_tuple = raw_targets.get(tid)
            if tgt_tuple:
                chunk_target_objects[tid] = OptimizedEntity(
                    norm_name=tgt_tuple[0],
                    core_name=tgt_tuple[1],
                    norm_addr=tgt_tuple[2],
                    country=tgt_tuple[3],
                    src_id=tid,
                    ac=None,
                    freq_tracker=freq_tracker
                )

        feat_matrix_rows = []
        pair_meta = []
        
        for s1_id in s1_ids:
            s1_obj = s1_objects[s1_id]
            cand_set = final_cands_dict.get(s1_id, set())
            c_len = len(cand_set)
            cand_counts_sample.append(c_len)
            total_candidate_pairs += c_len
            
            evid_map = ret_evidence_dict.get(s1_id, {})
            for tid in cand_set:
                tgt_obj = chunk_target_objects.get(tid)
                if not tgt_obj:
                    continue
                evid = evid_map.get(tid, {})
                full_row = extract_ultra_features_fast(s1_obj, tgt_obj, freq_tracker, evid)
                feat_matrix_rows.append(full_row)
                pair_meta.append((s1_id, tid))
                
        pair_prob_dict = defaultdict(list)
        if feat_matrix_rows:
            X_chunk = np.array(feat_matrix_rows, dtype=np.float32)
            probs = model.predict(X_chunk)
            for (s1_id, tid), prob in zip(pair_meta, probs):
                pair_prob_dict[s1_id].append((tid, float(prob)))
                
        # Apply exact V1.3 decision logic with competition margins & singleton guard
        final_matched_dict = postprocessor.apply(
            s1_ids=s1_ids,
            pair_probs=pair_prob_dict,
            candidate_sets=final_cands_dict,
            s1_objects=s1_objects,
            target_lookup=chunk_target_objects,
            s1_addr_comps=s1_addr_comps_chunk,
            tgt_addr_comps=None,
        )
        
        for s1_id in s1_ids:
            cand_set = final_cands_dict.get(s1_id, set())
            matched_set = final_matched_dict.get(s1_id, set())
            
            # Invariant Check: matched_set must strictly be subset of cand_set
            invalid_matches = matched_set - cand_set
            if invalid_matches:
                raise AssertionError(f"Fatal: Matched IDs {invalid_matches} not in candidate set for S1 {s1_id}")
            
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
        if chunk_idx % 10 == 0 or (max_s1 is not None and total_s1_processed >= max_s1):
            print(f"    Chunk {chunk_idx:4d} | Processed: {total_s1_processed:8,d} S1s | Throughput: {rate:6.1f} S1/s | Elapsed: {elapsed:6.1f}s | Matches: {total_predicted_matches:8,d}", flush=True)
            f_match.flush()
            f_cand.flush()
        chunk_target_objects.clear()

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
    
    t_total_end = time.time()
    total_runtime = t_total_end - t_global_start
    inf_runtime = t_total_end - t_inference_start
    overall_throughput = total_s1_processed / inf_runtime if inf_runtime > 0 else 0.0
    
    print("\n" + "=" * 80, flush=True)
    print("PRODUCTION INFERENCE RUN COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)
    print(f"  Total S1 Entities Processed:    {total_s1_processed:,}", flush=True)
    print(f"  Total Candidate Pairs Scored:   {total_candidate_pairs:,}", flush=True)
    print(f"  Total Matches Predicted:        {total_predicted_matches:,}", flush=True)
    print(f"  Singletons (Empty Predictions): {total_empty_predictions:,} ({total_empty_predictions/total_s1_processed*100:.2f}%)", flush=True)
    if cand_counts_sample:
        print(f"  Average Candidates / S1:        {np.mean(cand_counts_sample):.2f}", flush=True)
        print(f"  Median Candidates / S1:         {np.median(cand_counts_sample):.0f}", flush=True)
        print(f"  Max Candidates / S1:            {np.max(cand_counts_sample)}", flush=True)
    print(f"  Total Pipeline Time:            {total_runtime:.2f}s ({total_runtime/60:.2f} min)", flush=True)
    print(f"  Inference-Only Time:            {inf_runtime:.2f}s ({inf_runtime/60:.2f} min)", flush=True)
    print(f"  Sustained Inference Throughput: {overall_throughput:.2f} S1/sec", flush=True)
    print(f"  Matching TSV:                   {matching_out_path} ({os.path.getsize(matching_out_path)/1024/1024:.2f} MB)", flush=True)
    print(f"  Candidate TSV:                  {candidate_out_path} ({os.path.getsize(candidate_out_path)/1024/1024:.2f} MB)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Production Test Inference (V1.3 Surgical Optimized)")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test dataset folder")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output TSVs")
    parser.add_argument("--model-path", type=str, default="models/lightgbm_v1_1_ultra.txt", help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.965, help="V1.3 base threshold")
    parser.add_argument("--s-guard", type=float, default=0.900, help="V1.3 singleton guard threshold")
    parser.add_argument("--min-margin", type=float, default=0.020, help="V1.3 minimum competition margin")
    parser.add_argument("--joint-sim-floor", type=float, default=0.450, help="V1.3 joint similarity floor")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of S1 records per chunk")
    parser.add_argument("--max-s1", type=int, default=None, help="Maximum S1 records to evaluate (for smoke test)")
    parser.add_argument("--disable-tfidf", action="store_true", help="Disable multi-view TF-IDF retrieval")
    parser.add_argument("--disable-global-consistency", action="store_true", help="Disable global target exclusivity")
    args = parser.parse_args()
    
    run_test_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        s_guard=args.s_guard,
        min_margin=args.min_margin,
        joint_sim_floor=args.joint_sim_floor,
        chunk_size=args.chunk_size,
        max_s1=args.max_s1,
        enable_tfidf=not args.disable_tfidf,
        enable_global_consistency=not args.disable_global_consistency,
    )
