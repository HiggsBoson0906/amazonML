"""
V2 Production Inference Pipeline — Amazon ML Challenge 2026
============================================================

Architecture redesign for 100x+ wall-clock speedup over V1.3:

KEY OPTIMIZATIONS:
1. MULTIPROCESSING: Dynamic CPU detection. Scales across all vCPUs using
   Linux fork copy-on-write. Zero memory duplication of target structures.
2. COLUMNAR TARGET STORE: All target-side data stored in arrays/lists (TargetStore).
   No per-chunk OptimizedEntity construction. ~2-3 GB instead of 50+ GB.
3. TWO-STAGE RANKING:
   Stage A: Ultra-cheap filter using token Jaccard, n-gram overlap, country/postal.
            Reduces ~100-200 candidates to ~15-50 survivors. Zero RapidFuzz calls.
   Stage B: Full 50-feature LightGBM scoring only on survivors.
4. LAZY TARGET PRIMITIVE CREATION: Full dict-primitives only built for Stage-B
   survivors (~15-50 per S1), not for all ~200 candidates.
5. ELIMINATED REDUNDANT COMPUTATION:
   - Postprocessor reuses JW scores from feature extraction
   - No re-creation of OptimizedEntity objects per chunk
   - frozenset ops instead of repeated set() construction
6. MICROBATCHING: 500 S1 per microbatch, distributed to workers.
7. STREAMING OUTPUT: Results written per-microbatch in order.

PRESERVED:
- All 50 LightGBM features (exact numerical fidelity)
- V1.3 surgical postprocessing (thresholds, guards, consistency)
- Same output format (matching_results.tsv, candidate_pairs.tsv)
"""

import os
import sys
import time
import gc
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import lightgbm as lgb
import polars as pl

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
from src.features import ULTRA_FEATURE_COLS
from src.features_v2 import (
    TargetStore,
    make_s1_primitives,
    filter_candidates_stage_a,
    extract_ultra_features_v2,
    STAGE_A_THRESHOLD,
)
from src.retrieval import MultiViewCandidateRetriever
from src.ranking import CorpusFrequencyTracker
from src.postprocess_v2 import PostProcessorV2

DELIM = "\t"


# =========================================================================
# GLOBAL READ-ONLY STATE FOR WORKERS
# =========================================================================
# On Linux, these are inherited via fork() copy-on-write, consuming 0 extra RAM.
G_MODEL = None
G_BLOCKER = None
G_FREQ_TRACKER = None
G_TARGET_STORE = None
G_RETRIEVER = None
G_STAGE_A_MAX = 50
G_POSTPROCESSOR = None


def init_worker():
    """Worker initialization: ensure single-threaded execution for LightGBM and BLAS."""
    # Prevent hidden thread pools from oversubscribing the machine
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    if G_MODEL is not None:
        G_MODEL.set_num_threads(1)


def process_microbatch_worker(args: Tuple[int, List[Tuple[str, str, str, str]]]):
    """
    Processes a single microbatch of S1 records in a worker process.
    Reads from G_* globals, returns compact string results.
    """
    batch_idx, rows = args
    
    s1_ids = []
    s1_prims: Dict[str, dict] = {}
    s1_by_country = defaultdict(list)

    for eid, name, addr, country in rows:
        s1_ids.append(eid)
        nn = normalize_business_name(name)
        cn = extract_core_business_name(nn)
        na = normalize_business_address(addr) if addr else ""
        s1_prims[eid] = make_s1_primitives(nn, cn, na, country, G_FREQ_TRACKER)
        c = country.strip() if country else "UNKNOWN"
        s1_by_country[c].append((eid, nn, cn, na, country))

    # --- Candidate Generation ---
    df_mb = pl.DataFrame({
        "entity_id": s1_ids,
        "business_name": [r[1] for r in rows],
        "norm_name": [s1_prims[eid]['nn'] for eid in s1_ids],
        "core_name": [s1_prims[eid]['cn'] for eid in s1_ids],
        "norm_address": [s1_prims[eid]['na'] for eid in s1_ids],
        "country": [r[3] for r in rows],
    })

    stage5_cands = G_BLOCKER.generate_candidates_for_s1(df_mb)

    if G_RETRIEVER is not None:
        final_cands, ret_evidence = G_RETRIEVER.retrieve_candidates_batch(
            s1_by_country=s1_by_country,
            blocking_cands_map=stage5_cands,
        )
    else:
        final_cands = stage5_cands
        ret_evidence = {}

    # --- Stage A + Stage B ---
    feat_rows = []
    pair_meta = []
    total_raw_cands = 0
    total_stage_a_survivors = 0

    for s1_id in s1_ids:
        sp = s1_prims[s1_id]
        cand_set = final_cands.get(s1_id, set())
        total_raw_cands += len(cand_set)

        evid_s1 = ret_evidence.get(s1_id, {})

        # STAGE A: Cheap pre-filter
        survivors = filter_candidates_stage_a(
            sp['nn'], sp['cn'], sp['na'], sp['ct'],
            sp['ct_f'], sp['ng3'], sp['pc'], sp['hn'],
            cand_set, G_TARGET_STORE, evid_s1,
            max_survivors=G_STAGE_A_MAX,
        )
        total_stage_a_survivors += len(survivors)

        # STAGE B: Full 50-feature extraction
        for tid, sa_score in survivors:
            tgt_p = G_TARGET_STORE.get_primitives(tid)
            if tgt_p is None:
                continue
            evid = evid_s1.get(tid, {})
            feat_vec = extract_ultra_features_v2(sp, tgt_p, G_FREQ_TRACKER, evid)
            feat_rows.append(feat_vec)

            pair_meta.append((
                s1_id, tid,
                feat_vec[37],  # name_jw
                feat_vec[39],  # addr_jw
                feat_vec[1] > 0.5,   # exact_core_name
                feat_vec[14] > 0.5,  # addr_exact
                feat_vec[27] > 0.5,  # addr_postal_match
                feat_vec[25] > 0.5,  # addr_hnum_match
            ))

    # --- LightGBM Batch Prediction ---
    pair_results: Dict[str, List[dict]] = defaultdict(list)
    if feat_rows:
        X = np.array(feat_rows, dtype=np.float32)
        probs = G_MODEL.predict(X)

        for (s1_id, tid, njw, ajw, ec, ea, pm, hm), prob in zip(pair_meta, probs):
            pair_results[s1_id].append({
                'tid': tid,
                'prob': float(prob),
                'name_jw': njw,
                'addr_jw': ajw,
                'exact_core': ec,
                'exact_addr': ea,
                'postal_match': pm,
                'hnum_match': hm,
            })

    # --- V1.3 Postprocessing ---
    final_matched = G_POSTPROCESSOR.apply(s1_ids, pair_results)

    # --- Prepare Return Data (Compact Strings) ---
    match_output = []
    cand_output = []
    total_matches = 0
    total_empty = 0

    for s1_id in s1_ids:
        cand_set = final_cands.get(s1_id, set())
        matched_set = final_matched.get(s1_id, set())

        cand_output.append(f"{s1_id}\t{','.join(sorted(cand_set))}\n")
        match_output.append(f"{s1_id}\t{','.join(sorted(matched_set))}\n")

        if matched_set:
            total_matches += len(matched_set)
        else:
            total_empty += 1

    stats = {
        's1_count': len(s1_ids),
        'raw_cands': total_raw_cands,
        'stage_a_surv': total_stage_a_survivors,
        'matches': total_matches,
        'empty': total_empty
    }

    return (batch_idx, match_output, cand_output, stats)


def run_v2_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "outputs",
    model_path: str = "models/lightgbm_v1_1_ultra.txt",
    threshold: float = 0.965,
    s_guard: float = 0.900,
    min_margin: float = 0.020,
    joint_sim_floor: float = 0.450,
    microbatch_size: int = 500,
    stage_a_max_survivors: int = 50,
    max_s1: Optional[int] = None,
    enable_tfidf: bool = True,
    enable_global_consistency: bool = True,
    workers: str = "auto",
):
    global G_MODEL, G_BLOCKER, G_FREQ_TRACKER, G_TARGET_STORE, G_RETRIEVER, G_STAGE_A_MAX, G_POSTPROCESSOR

    if workers.lower() == "auto":
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(workers)

    print("=" * 80, flush=True)
    print("V2 PRODUCTION INFERENCE — AMAZON ML CHALLENGE 2026", flush=True)
    print("=" * 80, flush=True)
    print(f"  Test Directory:         {test_dir}", flush=True)
    print(f"  Output Directory:       {output_dir}", flush=True)
    print(f"  Model:                  {model_path}", flush=True)
    print(f"  Workers:                {n_workers} (Detected: {os.cpu_count()})", flush=True)
    print(f"  Microbatch Size:        {microbatch_size}", flush=True)
    print(f"  Max S1:                 {'ALL' if max_s1 is None else f'{max_s1:,}'}", flush=True)
    print("=" * 80, flush=True)

    t_global = time.time()
    os.makedirs(output_dir, exist_ok=True)
    test_path = Path(test_dir)

    source1_file = test_path / "test_source1.tsv"
    source2_file = test_path / "test_source2.tsv"
    source3_file = test_path / "test_source3.tsv"

    # =========================================================================
    # 1. LOAD MODEL
    # =========================================================================
    print("\n1. Loading LightGBM Model...", flush=True)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = lgb.Booster(model_file=str(model_path))
    G_MODEL = model
    print(f"  [✓] Model loaded: {len(model.feature_name())} features.", flush=True)

    # =========================================================================
    # 2. STREAM & INDEX TARGETS
    # =========================================================================
    print("\n2. Streaming & Indexing Target Records...", flush=True)
    t0 = time.time()

    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    freq_tracker = CorpusFrequencyTracker()
    raw_targets: Dict[str, Tuple[str, str, str, str]] = {}
    total_targets = 0

    def process_target_file(filepath: Path, src_name: str):
        nonlocal total_targets
        t_file = time.time()
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()
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
                    print(f"    {src_name}: {count:,}...", flush=True)
        print(f"  {src_name}: {count:,} in {time.time() - t_file:.1f}s", flush=True)

    process_target_file(source2_file, "S2")
    process_target_file(source3_file, "S3")

    print(f"  Total: {total_targets:,} in {time.time() - t0:.1f}s", flush=True)
    blocker.prune_high_frequency_keys()

    # =========================================================================
    # 2b. BUILD COLUMNAR TARGET STORE
    # =========================================================================
    print("\n  Building TargetStore (columnar)...", flush=True)
    target_store = TargetStore()
    target_store.build(raw_targets, freq_tracker)

    # =========================================================================
    # 2c. FIT TF-IDF RETRIEVER
    # =========================================================================
    retriever = None
    if enable_tfidf:
        print("  Fitting TF-IDF...", flush=True)
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
        print(f"  TF-IDF fitted in {time.time() - t_ret:.1f}s", flush=True)

    # Free raw_targets — TargetStore and TF-IDF have consumed the data
    del raw_targets
    gc.collect()

    # Bind to globals for fork()
    G_BLOCKER = blocker
    G_FREQ_TRACKER = freq_tracker
    G_TARGET_STORE = target_store
    G_RETRIEVER = retriever
    G_STAGE_A_MAX = stage_a_max_survivors
    G_POSTPROCESSOR = PostProcessorV2(
        base_threshold=threshold,
        s_guard=s_guard,
        min_margin=min_margin,
        joint_sim_floor=joint_sim_floor,
        enable_global_consistency=enable_global_consistency,
    )

    try:
        import psutil
        rss_gb = psutil.Process().memory_info().rss / (1024**3)
        print(f"  [✓] Init complete (raw_targets freed). RSS: {rss_gb:.2f} GB", flush=True)
    except Exception:
        print("  [✓] Init complete.", flush=True)

    # =========================================================================
    # 3. STREAMING INFERENCE
    # =========================================================================
    matching_out = Path(output_dir) / "matching_results.tsv"
    candidate_out = Path(output_dir) / "candidate_pairs.tsv"

    print(f"\n3. Inference (Stage-A → Stage-B → Postprocess)...", flush=True)

    f_match = open(matching_out, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_out, "w", encoding="utf-8", newline="")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    def batch_generator():
        chunk_rows = []
        batch_idx = 0
        with open(source1_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split(DELIM)
                if len(parts) != 4:
                    continue
                eid, name, addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not eid:
                    continue
                
                chunk_rows.append((eid, name, addr, country))

                if len(chunk_rows) >= microbatch_size:
                    yield (batch_idx, chunk_rows)
                    batch_idx += 1
                    chunk_rows = []
                    
                if max_s1 is not None and (batch_idx * microbatch_size + len(chunk_rows)) >= max_s1:
                    break

            if chunk_rows:
                yield (batch_idx, chunk_rows)

    t_inf = time.time()
    
    total_s1 = 0
    total_raw_cands = 0
    total_stage_a_survivors = 0
    total_matches = 0
    total_empty = 0

    # Ensure parent process also limits LightGBM threads if we fall back to sequential
    init_worker()

    if n_workers > 1 and sys.platform != "win32":
        # Linux / MacOS: Safe to fork with copy-on-write
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=n_workers, initializer=init_worker)
        iterator = pool.imap(process_microbatch_worker, batch_generator())
    elif n_workers > 1 and sys.platform == "win32":
        print("  [!] WARNING: Multiprocessing on Windows uses spawn. Bypassing parallel execution for local testing.", flush=True)
        print("  [!] (On Linux/EC2, this will use fork and utilize all cores).", flush=True)
        iterator = map(process_microbatch_worker, batch_generator())
        pool = None
    else:
        # Sequential
        iterator = map(process_microbatch_worker, batch_generator())
        pool = None

    report_interval = max(microbatch_size * 20, 5000)

    try:
        for batch_idx, match_output, cand_output, stats in iterator:
            # Write results (imap ensures order)
            f_match.writelines(match_output)
            f_cand.writelines(cand_output)

            # Update stats
            total_s1 += stats['s1_count']
            total_raw_cands += stats['raw_cands']
            total_stage_a_survivors += stats['stage_a_surv']
            total_matches += stats['matches']
            total_empty += stats['empty']

            if total_s1 % report_interval == 0 or total_s1 == max_s1:
                elapsed = time.time() - t_inf
                rate = total_s1 / elapsed if elapsed > 0 else 0.0
                avg_cands = total_raw_cands / total_s1 if total_s1 else 0
                avg_surv = total_stage_a_survivors / total_s1 if total_s1 else 0
                try:
                    rss = psutil.Process().memory_info().rss / (1024**3)
                    rss_str = f"{rss:.1f}GB"
                except Exception:
                    rss_str = "N/A"
                print(
                    f"    {total_s1:>9,d} S1 | {rate:>7.1f} S1/s | "
                    f"cands: {avg_cands:.0f}→{avg_surv:.1f} | "
                    f"matches: {total_matches:,} | RSS: {rss_str} | {elapsed:.0f}s",
                    flush=True,
                )
                f_match.flush()
                f_cand.flush()
                
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    f_match.close()
    f_cand.close()

    # =========================================================================
    # 4. SUMMARY
    # =========================================================================
    t_total = time.time() - t_global
    t_inf_total = time.time() - t_inf
    rate = total_s1 / t_inf_total if t_inf_total > 0 else 0.0
    avg_c = total_raw_cands / max(total_s1, 1)
    avg_s = total_stage_a_survivors / max(total_s1, 1)

    print("\n" + "=" * 80, flush=True)
    print("V2 INFERENCE COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"  S1 Processed:       {total_s1:,}", flush=True)
    print(f"  Raw Candidates:     {total_raw_cands:,} (avg {avg_c:.1f}/S1)", flush=True)
    print(f"  Stage-A Survivors:  {total_stage_a_survivors:,} (avg {avg_s:.1f}/S1)", flush=True)
    print(f"  Stage-A Reduction:  {(1 - avg_s/max(avg_c,1))*100:.1f}%", flush=True)
    print(f"  Matches:            {total_matches:,}", flush=True)
    print(f"  Empty:              {total_empty:,} ({total_empty/max(total_s1,1)*100:.1f}%)", flush=True)
    print(f"  Total Time:         {t_total:.1f}s ({t_total/60:.1f} min)", flush=True)
    print(f"  Inference Time:     {t_inf_total:.1f}s ({t_inf_total/60:.1f} min)", flush=True)
    print(f"  Throughput:         {rate:.1f} S1/sec", flush=True)
    print(f"  Output:             {matching_out}", flush=True)
    try:
        rss = psutil.Process().memory_info().rss / (1024**3)
        print(f"  Final RSS:          {rss:.2f} GB", flush=True)
    except Exception:
        pass
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Production Inference — Amazon ML 2026")
    parser.add_argument("--test-dir", type=str, default="dataset/test")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-path", type=str, default="models/lightgbm_v1_1_ultra.txt")
    parser.add_argument("--threshold", type=float, default=0.965)
    parser.add_argument("--s-guard", type=float, default=0.900)
    parser.add_argument("--min-margin", type=float, default=0.020)
    parser.add_argument("--joint-sim-floor", type=float, default=0.450)
    parser.add_argument("--microbatch-size", type=int, default=500)
    parser.add_argument("--stage-a-max", type=int, default=50)
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--disable-tfidf", action="store_true")
    parser.add_argument("--disable-global-consistency", action="store_true")
    parser.add_argument("--workers", type=str, default="auto", help="auto, or int (e.g. 8)")
    args = parser.parse_args()

    run_v2_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        s_guard=args.s_guard,
        min_margin=args.min_margin,
        joint_sim_floor=args.joint_sim_floor,
        microbatch_size=args.microbatch_size,
        stage_a_max_survivors=args.stage_a_max,
        max_s1=args.max_s1,
        enable_tfidf=not args.disable_tfidf,
        enable_global_consistency=not args.disable_global_consistency,
        workers=args.workers,
    )
