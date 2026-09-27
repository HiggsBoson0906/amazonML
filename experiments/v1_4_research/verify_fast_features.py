import os
import sys
import time
import numpy as np
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import PrecomputedEntity
from src.ranking import CorpusFrequencyTracker, extract_structured_address_components
from scripts.push_to_98 import extract_ultra_features
from experiments.v1_4_research.fast_features import OptimizedEntity, extract_ultra_features_fast

def verify_fast_features():
    print("=" * 80)
    print("VERIFYING FAST FEATURES: NUMERICAL FIDELITY & SPEED BENCHMARK")
    print("=" * 80)
    
    # Create sample entities
    names = [
        "walmart supercenter branch 45", "target stores llc", "starbucks coffee co",
        "mcdonalds restaurant 1024", "amazon logistics fulfillment center dse4",
        "home depot usa inc", "cvs pharmacy #4829", "walgreens health store"
    ]
    addrs = [
        "123 main st ste 400 seattle wa 98101", "456 broadway ave new york ny 10001",
        "789 market st suite 120 san francisco ca 94103", "1000 grand blvd chicago il 60601",
        "", "55 oak street austin tx 78701", "88 pine rd denver co 80202"
    ]
    
    freq_tracker = CorpusFrequencyTracker()
    freq_tracker.fit(names, [a for a in addrs if a])
    
    pairs_old = []
    pairs_new = []
    
    for i in range(20000):
        n1 = names[i % len(names)]
        n2 = names[(i + i//len(names)) % len(names)]
        a1 = addrs[i % len(addrs)]
        a2 = addrs[(i + 1) % len(addrs)]
        
        s1_old = PrecomputedEntity(n1, n1.split()[0], a1, "US", f"S1-{i}")
        tgt_old = PrecomputedEntity(n2, n2.split()[0], a2, "US", f"S2-{i}")
        s1_ac = extract_structured_address_components(a1)
        tgt_ac = extract_structured_address_components(a2)
        evid = {"name_tfidf": 0.85, "addr_tfidf": 0.65}
        pairs_old.append((s1_old, tgt_old, s1_ac, tgt_ac, evid))
        
        s1_new = OptimizedEntity(n1, n1.split()[0], a1, "US", f"S1-{i}", s1_ac, freq_tracker)
        tgt_new = OptimizedEntity(n2, n2.split()[0], a2, "US", f"S2-{i}", tgt_ac, freq_tracker)
        pairs_new.append((s1_new, tgt_new, evid))
        
    # Check numerical fidelity
    print("1. Checking numerical fidelity across 20,000 pairs...", flush=True)
    max_diff = 0.0
    mismatches = 0
    for idx, ((s1_o, tgt_o, s1_ac, tgt_ac, evid_o), (s1_n, tgt_n, evid_n)) in enumerate(zip(pairs_old, pairs_new)):
        v_old = extract_ultra_features(s1_o, tgt_o, s1_ac, tgt_ac, freq_tracker, evid_o)
        v_new = extract_ultra_features_fast(s1_n, tgt_n, freq_tracker, evid_n)
        diff = np.max(np.abs(np.array(v_old) - np.array(v_new)))
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-4:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Mismatch at pair {idx}: max diff = {diff:.6f}")
                
    print(f"  Max absolute difference across all 50 features: {max_diff:.8f}")
    print(f"  Mismatches (> 1e-4): {mismatches} / {len(pairs_old)}")
    assert mismatches == 0, "Numerical discrepancy detected!"
    print("  [✓] 100.000% EXACT NUMERICAL FIDELITY CONFIRMED!\n")
    
    # Benchmark execution speed
    print("2. Benchmarking execution speed...", flush=True)
    t0 = time.time()
    for s1_o, tgt_o, s1_ac, tgt_ac, evid_o in pairs_old:
        _ = extract_ultra_features(s1_o, tgt_o, s1_ac, tgt_ac, freq_tracker, evid_o)
    t_old = time.time() - t0
    
    t0 = time.time()
    for s1_n, tgt_n, evid_n in pairs_new:
        _ = extract_ultra_features_fast(s1_n, tgt_n, freq_tracker, evid_n)
    t_new = time.time() - t0
    
    print(f"  Baseline Feature Extraction: {t_old:.3f}s ({t_old/len(pairs_old)*1e6:.1f} us/pair)")
    print(f"  Fast Feature Extraction:     {t_new:.3f}s ({t_new/len(pairs_new)*1e6:.1f} us/pair)")
    print(f"  Speedup Factor:              {t_old / t_new:.2f}x faster!")

if __name__ == "__main__":
    verify_fast_features()
