import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set

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
)
from src.data_loader import load_and_normalize_source, load_ground_truth
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
    extract_name_blocking_keys,
    extract_address_blocking_keys,
)
from src.features import extract_pairwise_features
from src.evaluate import evaluate_predictions, evaluate_candidate_recall

def test_blocking_on_true_pairs(n_s1_sample: int = 5000):
    print("=" * 60)
    print(f"EVALUATING BLOCKING RECALL ON GROUND TRUTH PAIRS ({n_s1_sample:,} S1 sample)")
    print("=" * 60)
    
    # 1. Load Ground Truth
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    # Select first n_s1_sample S1 records
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=n_s1_sample)
    s1_dict = {
        row["entity_id"]: row
        for row in df_s1.iter_rows(named=True)
    }
    
    eval_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in s1_dict.keys()}
    
    # Collect all true target IDs needed for this sample
    needed_target_ids = set()
    for s1_id, t_ids in eval_gt.items():
        needed_target_ids.update(t_ids)
        
    print(f"Sample S1 Entities: {len(df_s1):,}")
    print(f"Total True Target IDs to find: {len(needed_target_ids):,}")
    
    # Stream S2 and S3 to extract only the needed true target records + a background sample of distractor records
    print("\nExtracting target records from S2 and S3...")
    target_records = {}
    
    def extract_targets_from_tsv(path, needed_ids, max_distractors=50000):
        found = 0
        distractors = 0
        records = {}
        with open(path, "r", encoding="utf-8") as f:
            header = [c.strip() for c in f.readline().rstrip("\r\n").split("\t")]
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if tid in needed_ids:
                    n_name = normalize_business_name(name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(addr)
                    records[tid] = {
                        "entity_id": tid,
                        "business_name": name,
                        "business_address": addr,
                        "country": country,
                        "norm_name": n_name,
                        "core_name": c_name,
                        "norm_address": n_addr,
                    }
                    found += 1
                elif distractors < max_distractors:
                    n_name = normalize_business_name(name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(addr)
                    records[tid] = {
                        "entity_id": tid,
                        "business_name": name,
                        "business_address": addr,
                        "country": country,
                        "norm_name": n_name,
                        "core_name": c_name,
                        "norm_address": n_addr,
                    }
                    distractors += 1
        return records, found
        
    t0 = time.time()
    s2_recs, s2_found = extract_targets_from_tsv(TRAIN_SOURCE2, needed_target_ids, max_distractors=100000)
    s3_recs, s3_found = extract_targets_from_tsv(TRAIN_SOURCE3, needed_target_ids, max_distractors=100000)
    
    target_records.update(s2_recs)
    target_records.update(s3_recs)
    print(f"Extracted {len(target_records):,} total target records (Found {s2_found + s3_found:,} / {len(needed_target_ids):,} true matches) in {time.time() - t0:.2f}s")
    
    # Now let's test our blocking index on this target pool
    print("\nBuilding Inverted Index on the target records...")
    index = defaultdict(lambda: defaultdict(list))
    for tid, rec in target_records.items():
        country = rec["country"].strip()
        keys = extract_name_blocking_keys(rec["core_name"], rec["norm_name"])
        keys.extend(extract_address_blocking_keys(rec["norm_address"]))
        for k in keys:
            index[country][k].append(tid)
            
    # Generate candidates for each S1 using prioritized key ordering
    def key_priority(k: str) -> int:
        if k.startswith("name:"):
            return 1
        if k.startswith("addr:"):
            return 2
        if k.startswith("tok2:"):
            return 3
        if k.startswith("sort2:"):
            return 4
        if k.startswith("tok1:"):
            return 5
        return 6

    candidates = {}
    for s1_id, s1_rec in s1_dict.items():
        country = s1_rec["country"].strip()
        keys = extract_name_blocking_keys(s1_rec["core_name"], s1_rec["norm_name"])
        keys.extend(extract_address_blocking_keys(s1_rec["norm_address"]))
        # Sort keys by priority (specific -> broad)
        keys = sorted(keys, key=key_priority)
        
        cand_set = set()
        if country in index:
            c_idx = index[country]
            for k in keys:
                if k in c_idx:
                    for tid in c_idx[k]:
                        cand_set.add(tid)
                        if len(cand_set) >= 100:
                            break
                if len(cand_set) >= 100:
                    break
        candidates[s1_id] = cand_set
        
    cand_metrics = evaluate_candidate_recall(eval_gt, candidates)
    print("\n" + "=" * 60)
    print("--- REALISTIC BLOCKING RECALL CEILING ---")
    print(f"Candidate Recall Ceiling: {cand_metrics['candidate_recall_ceiling']*100:.2f}%")
    print(f"Total True Matches: {cand_metrics['total_true_matches']:,}")
    print(f"Captured Matches: {cand_metrics['captured_matches']:,}")
    print(f"Avg Candidates per S1: {cand_metrics['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates per S1: {cand_metrics['median_candidates_per_s1']:.1f}")
    
    # Let's inspect any missed true matches to see WHY they were missed and how to improve blocking!
    missed_examples = []
    for s1_id, true_set in eval_gt.items():
        cand_set = candidates.get(s1_id, set())
        missed = true_set - cand_set
        for tid in missed:
            if tid in target_records:
                s1_rec = s1_dict[s1_id]
                tgt_rec = target_records[tid]
                missed_examples.append((s1_rec, tgt_rec))
                if len(missed_examples) >= 10:
                    break
        if len(missed_examples) >= 10:
            break
            
    print(f"\nSample Missed True Pairs (Total missed in sample: {cand_metrics['total_true_matches'] - cand_metrics['captured_matches']}):")
    for idx, (s1, tgt) in enumerate(missed_examples[:5], 1):
        print(f"\n[Missed #{idx}]")
        print(f"  S1  [{s1['entity_id']}]: Name='{s1['business_name']}' | Addr='{s1['business_address']}' | Country={s1['country']}")
        print(f"  Tgt [{tgt['entity_id']}]: Name='{tgt['business_name']}' | Addr='{tgt['business_address']}' | Country={tgt['country']}")
        s1_keys = extract_name_blocking_keys(s1['core_name'], s1['norm_name'])
        tgt_keys = extract_name_blocking_keys(tgt['core_name'], tgt['norm_name'])
        print(f"  S1 Keys: {s1_keys}")
        print(f"  Tgt Keys: {tgt_keys}")

if __name__ == "__main__":
    test_blocking_on_true_pairs(n_s1_sample=5000)
