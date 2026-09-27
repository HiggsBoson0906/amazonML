import os
import sys
import time
import psutil
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import PrecomputedEntity
from src.retrieval import MultiViewCandidateRetriever

def benchmark_indexing():
    process = psutil.Process()
    print("=" * 80)
    print("TARGET INDEXING & MEMORY PROFILING (TEST SOURCE 2 + SOURCE 3)")
    print("=" * 80)
    
    t0 = time.time()
    s2_file = Path("dataset/test/test_source2.tsv")
    s3_file = Path("dataset/test/test_source3.tsv")
    
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    target_lookup_fast = {}
    target_lookup_raw = {}
    
    total_targets = 0
    
    def process_file(filepath: Path, name: str):
        nonlocal total_targets
        t_start = time.time()
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, b_name, b_addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not tid:
                    continue
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
                
                count += 1
                total_targets += 1
                if count % 1000000 == 0:
                    rss_gb = process.memory_info().rss / (1024 ** 3)
                    print(f"    [{name}] {count:,} records loaded... RSS: {rss_gb:.2f} GiB", flush=True)
                    
        print(f"  {name}: {count:,} records in {time.time() - t_start:.2f}s | Current RSS: {process.memory_info().rss / (1024**3):.2f} GiB", flush=True)
        
    print("1. Indexing Source 2...", flush=True)
    process_file(s2_file, "Source 2")
    print("2. Indexing Source 3...", flush=True)
    process_file(s3_file, "Source 3")
    
    print(f"\nTotal Target records loaded: {total_targets:,} in {time.time() - t0:.2f}s")
    print(f"Peak RSS after loading: {process.memory_info().rss / (1024**3):.2f} GiB", flush=True)
    
    print("\n3. Pruning high-frequency blocking keys...", flush=True)
    t_prune = time.time()
    blocker.prune_high_frequency_keys()
    print(f"Pruned high-frequency keys in {time.time() - t_prune:.2f}s | RSS: {process.memory_info().rss / (1024**3):.2f} GiB")
    
    print("\n4. Fitting Multi-View TF-IDF Retriever...", flush=True)
    t_tfidf = time.time()
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    print(f"Multi-View TF-IDF fitted in {time.time() - t_tfidf:.2f}s | RSS: {process.memory_info().rss / (1024**3):.2f} GiB")
    
    target_lookup_raw.clear()
    gc.collect()
    print(f"RSS after clearing raw lookups: {process.memory_info().rss / (1024**3):.2f} GiB")
    print(f"Total initialization time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    benchmark_indexing()
