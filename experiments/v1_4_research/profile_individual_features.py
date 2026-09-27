import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import PrecomputedEntity
from src.ranking import CorpusFrequencyTracker, extract_structured_address_components, compute_address_component_features

def profile_features():
    print("=" * 80)
    print("GRANULAR FEATURE EXTRACTION CPU PROFILER (50,000 PAIRS)")
    print("=" * 80)
    
    # Synthetic realistic pairs
    pairs = []
    for i in range(50000):
        s1 = PrecomputedEntity(
            norm_name=f"amazon web services inc branch {i%100}",
            core_name=f"amazon web services {i%100}",
            norm_addr=f"{100 + i%900} main street suite {i%50} seattle wa 98101",
            country="US",
            src_id=f"S1-{i}"
        )
        tgt = PrecomputedEntity(
            norm_name=f"amazon web services llc {i%100}",
            core_name=f"amazon web services {i%100}",
            norm_addr=f"{100 + (i+1)%900} main st ste {i%50} seattle wa 98101",
            country="US",
            src_id=f"S2-{i}"
        )
        s1_ac = extract_structured_address_components(s1.norm_addr)
        tgt_ac = extract_structured_address_components(tgt.norm_addr)
        evid = {"name_tfidf": 0.85, "addr_tfidf": 0.72}
        pairs.append((s1, tgt, s1_ac, tgt_ac, evid))
        
    freq_tracker = CorpusFrequencyTracker()
    freq_tracker.fit([p[1].norm_name for p in pairs[:5000]], [p[1].norm_addr for p in pairs[:5000]])
    
    timings = defaultdict(float)
    N = len(pairs)
    
    # 1. fuzz.ratio norm_name
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.ratio(s1.norm_name, tgt.norm_name)
    timings["fuzz.ratio(norm_name)"] = time.time() - t0
    
    # 2. fuzz.WRatio norm_name
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.WRatio(s1.norm_name, tgt.norm_name)
    timings["fuzz.WRatio(norm_name)"] = time.time() - t0
    
    # 3. fuzz.token_sort_ratio norm_name
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.token_sort_ratio(s1.norm_name, tgt.norm_name)
    timings["fuzz.token_sort_ratio(norm_name)"] = time.time() - t0
    
    # 4. fuzz.token_set_ratio norm_name
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.token_set_ratio(s1.norm_name, tgt.norm_name)
    timings["fuzz.token_set_ratio(norm_name)"] = time.time() - t0
    
    # 5. fuzz.partial_ratio norm_name
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.partial_ratio(s1.norm_name, tgt.norm_name)
    timings["fuzz.partial_ratio(norm_name)"] = time.time() - t0
    
    # 6. Core name fuzz metrics (ratio, sort, set)
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.ratio(s1.core_name, tgt.core_name)
        _ = fuzz.token_sort_ratio(s1.core_name, tgt.core_name)
        _ = fuzz.token_set_ratio(s1.core_name, tgt.core_name)
    timings["fuzz.core_name (ratio+sort+set)"] = time.time() - t0
    
    # 7. Address fuzz metrics (ratio, sort, set)
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = fuzz.ratio(s1.norm_addr, tgt.norm_addr)
        _ = fuzz.token_sort_ratio(s1.norm_addr, tgt.norm_addr)
        _ = fuzz.token_set_ratio(s1.norm_addr, tgt.norm_addr)
    timings["fuzz.addr (ratio+sort+set)"] = time.time() - t0
    
    # 8. JaroWinkler similarity
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = JaroWinkler.similarity(s1.norm_name, tgt.norm_name)
        _ = JaroWinkler.similarity(s1.norm_addr, tgt.norm_addr)
    timings["JaroWinkler (name + addr)"] = time.time() - t0
    
    # 9. LCSseq similarity
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        _ = LCSseq.similarity(s1.norm_name, tgt.norm_name)
        _ = LCSseq.similarity(s1.norm_addr, tgt.norm_addr)
    timings["LCSseq (name + addr)"] = time.time() - t0
    
    # 10. Address components compute
    t0 = time.time()
    for s1, tgt, s1_ac, tgt_ac, _ in pairs:
        _ = compute_address_component_features(s1_ac, tgt_ac)
    timings["compute_address_components"] = time.time() - t0
    
    # 11. Token IDF weighted Jaccard
    t0 = time.time()
    for s1, tgt, _, _, _ in pairs:
        inter = s1.core_toks.intersection(tgt.core_toks)
        un = s1.core_toks.union(tgt.core_toks)
        _ = sum(freq_tracker.get_token_idf(t) for t in inter)
        _ = sum(freq_tracker.get_token_idf(t) for t in un)
    timings["Token IDF weighted Jaccard"] = time.time() - t0
    
    # 12. Python list concatenation and construction
    t0 = time.time()
    for _ in range(N):
        _ = [0.0] * 50
    timings["Python list construction (50 floats)"] = time.time() - t0
    
    total_time = sum(timings.values())
    print(f"\nTotal time for {N:,} pairs: {total_time:.3f}s ({total_time/N*1e6:.1f} us/pair)")
    print("-" * 80)
    for op, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / total_time) * 100
        print(f"  {op:<38}: {t:.3f}s ({t/N*1e6:5.1f} us/pair, {pct:5.1f}%)")

if __name__ == "__main__":
    profile_features()
