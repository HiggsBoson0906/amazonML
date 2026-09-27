import os
import sys
import time
import csv
from pathlib import Path
from collections import defaultdict, Counter
import polars as pl
from rapidfuzz import fuzz

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    OUTPUT_DIR,
)
from src.data_loader import load_and_normalize_source, load_ground_truth
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
    extract_name_blocking_keys,
    extract_address_blocking_keys,
)
from src.blocking import InvertedIndexBlocker
from src.evaluate import evaluate_candidate_recall

def is_non_latin(text: str) -> bool:
    """Check if string contains non-Latin characters (e.g., Devanagari, Telugu, etc.)."""
    for c in text:
        if ord(c) > 0x0590:  # Beyond Latin, Greek, Hebrew
            return True
    return False

def analyze_failures(n_s1_sample: int = 5000):
    print("=" * 65)
    print(f"TASK 1: ANALYZING CANDIDATE-GENERATION FAILURES ({n_s1_sample:,} S1 sample)")
    print("=" * 65)
    
    # 1. Load Ground Truth
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    # 2. Sample S1 entities
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=n_s1_sample)
    s1_dict = {row["entity_id"]: row for row in df_s1.iter_rows(named=True)}
    eval_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in s1_dict.keys()}
    
    needed_target_ids = set()
    for t_ids in eval_gt.values():
        needed_target_ids.update(t_ids)
        
    print(f"Loaded {len(s1_dict):,} S1 entities ({len(needed_target_ids):,} true target IDs)")
    
    # 3. Stream & Load S2/S3 Target Records (True Targets + 200,000 Distractors)
    def extract_targets(path, needed_ids, max_distractors=100000):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            distractors = 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if tid in needed_ids:
                    rows.append((tid, name, addr, country))
                elif distractors < max_distractors:
                    rows.append((tid, name, addr, country))
                    distractors += 1
        return rows
        
    s2_rows = extract_targets(TRAIN_SOURCE2, needed_target_ids, max_distractors=100000)
    s3_rows = extract_targets(TRAIN_SOURCE3, needed_target_ids, max_distractors=100000)
    
    all_target_rows = s2_rows + s3_rows
    df_targets = pl.DataFrame(
        all_target_rows,
        schema=["entity_id", "business_name", "business_address", "country"],
        orient="row",
    )
    
    names = df_targets["business_name"].to_list()
    addrs = df_targets["business_address"].to_list()
    
    norm_names = [normalize_business_name(n) for n in names]
    core_names = [extract_core_business_name(n) for n in norm_names]
    norm_addrs = [normalize_business_address(a) for a in addrs]
    
    df_targets = df_targets.with_columns(
        pl.Series("norm_name", norm_names, dtype=pl.Utf8),
        pl.Series("core_name", core_names, dtype=pl.Utf8),
        pl.Series("norm_address", norm_addrs, dtype=pl.Utf8),
    )
    
    target_lookup = {
        row["entity_id"]: row
        for row in df_targets.iter_rows(named=True)
    }
    
    # 4. Build Blocker and Track Keys
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=100)
    blocker.add_target_records(df_targets)
    
    # Keep copy of unpruned keys to check if high-frequency pruning caused the drop
    raw_key_counts = {
        country: {k: len(v) for k, v in keys_dict.items()}
        for country, keys_dict in blocker.index.items()
    }
    
    blocker.prune_high_frequency_keys()
    candidates = blocker.generate_candidates_for_s1(df_s1)
    
    # 5. Analyze Missed Ground Truth Pairs
    failure_records = []
    category_counts = Counter()
    
    for s1_id, true_set in eval_gt.items():
        cand_set = candidates.get(s1_id, set())
        missed_set = true_set - cand_set
        s1_rec = s1_dict[s1_id]
        s1_country = s1_rec["country"].strip()
        
        s1_keys = set(extract_name_blocking_keys(s1_rec["core_name"], s1_rec["norm_name"]))
        s1_keys.update(extract_address_blocking_keys(s1_rec["norm_address"]))
        
        for tid in missed_set:
            if tid not in target_lookup:
                # Malformed or missing from dataset
                cat = "malformed_record"
                tgt_rec = {"business_name": "", "business_address": "", "country": "", "core_name": "", "norm_name": "", "norm_address": ""}
                tgt_keys = set()
            else:
                tgt_rec = target_lookup[tid]
                tgt_country = tgt_rec["country"].strip()
                
                tgt_keys = set(extract_name_blocking_keys(tgt_rec["core_name"], tgt_rec["norm_name"]))
                tgt_keys.update(extract_address_blocking_keys(tgt_rec["norm_address"]))
                
                # Check failure root causes
                if s1_country != tgt_country:
                    cat = "missing_or_incorrect_country"
                elif is_non_latin(s1_rec["business_name"]) or is_non_latin(tgt_rec["business_name"]):
                    cat = "multilingual_transliteration_issue"
                else:
                    overlapping_keys = s1_keys.intersection(tgt_keys)
                    if overlapping_keys:
                        # Shared key existed! Why missed?
                        # Check if pruned due to high frequency
                        was_pruned = any(
                            raw_key_counts.get(s1_country, {}).get(k, 0) > 500
                            for k in overlapping_keys
                        )
                        if was_pruned:
                            cat = "high_frequency_token_pruning"
                        else:
                            # S1 candidate capacity (100) filled by earlier keys
                            cat = "candidate_capacity_exceeded"
                    else:
                        # No shared keys between S1 and Target
                        name_sim = fuzz.token_set_ratio(s1_rec["core_name"], tgt_rec["core_name"])
                        addr_sim = fuzz.token_set_ratio(s1_rec["norm_address"], tgt_rec["norm_address"]) if (s1_rec["norm_address"] and tgt_rec["norm_address"]) else 0
                        
                        if name_sim >= 75:
                            cat = "name_normalization_failure"
                        elif addr_sim >= 75:
                            cat = "address_normalization_failure"
                        else:
                            cat = "no_blocking_key_overlap"
                            
            category_counts[cat] += 1
            failure_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "source": "S2" if tid.startswith("S2-") else "S3",
                "country": s1_country,
                "name_s1": s1_rec["business_name"],
                "name_target": tgt_rec["business_name"],
                "address_s1": s1_rec["business_address"],
                "address_target": tgt_rec["business_address"],
                "failure_category": cat,
                "blocking_keys_s1": "; ".join(sorted(s1_keys)),
                "blocking_keys_target": "; ".join(sorted(tgt_keys)),
            })
            
    # Save failure analysis CSV
    out_csv = OUTPUT_DIR / "blocking_failure_analysis.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source1_id", "target_id", "source", "country",
            "name_s1", "name_target", "address_s1", "address_target",
            "failure_category", "blocking_keys_s1", "blocking_keys_target"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_records)
        
    cand_metrics = evaluate_candidate_recall(eval_gt, candidates)
    print(f"\nSaved failure analysis to: {out_csv}")
    print(f"Total Missed Pairs Analyzed: {len(failure_records):,} / {cand_metrics['total_true_matches']:,}")
    print(f"Candidate Recall Ceiling: {cand_metrics['candidate_recall_ceiling']*100:.2f}%")
    print("\n--- AGGREGATE COUNTS BY FAILURE CATEGORY ---")
    for cat, count in category_counts.most_common():
        pct = (count / len(failure_records)) * 100 if failure_records else 0.0
        print(f"  {cat:35s}: {count:5d} ({pct:5.2f}%)")
    print("=" * 65)

if __name__ == "__main__":
    analyze_failures(n_s1_sample=5000)
