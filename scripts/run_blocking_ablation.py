import os
import sys
import time
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import polars as pl
import numpy as np

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
from src.data_loader import load_ground_truth, load_and_normalize_source
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
    extract_name_blocking_keys,
    extract_address_blocking_keys,
    GENERIC_STOP_WORDS,
    ADDRESS_ABBREVIATIONS,
)
from src.transliterate import transliterate_indic_to_latin
from src.evaluate import evaluate_candidate_recall

# -------------------------------------------------------------
# Modular Key Generators for Ablation
# -------------------------------------------------------------

from src.phonetic import phonetic_skeleton

def get_translit_blocking_keys(name: str) -> List[str]:
    """Generate cross-lingual phonetic and transliterated keys for Indic-English matching."""
    if not name:
        return []
    
    # Check if Indic script is present
    has_indic = any(ord(c) >= 0x0900 for c in name)
    latin_name = transliterate_indic_to_latin(name) if has_indic else name
    
    core_name = extract_core_business_name(normalize_business_name(latin_name))
    tokens = [t for t in core_name.split() if len(t) >= 3 and t not in GENERIC_STOP_WORDS]
    
    keys = []
    # 1. Transliterated exact name & prefix if Indic
    if has_indic:
        compact = "".join(tokens)
        if len(compact) >= 4:
            keys.append(f"tr_name:{compact[:25]}")
            keys.append(f"tr_pfx4:{compact[:4]}")
            
    # 2. Phonetic token pairs & single significant tokens (for BOTH Latin and Indic records)
    ph_tokens = [phonetic_skeleton(t) for t in tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
    if len(ph_tokens) >= 2:
        keys.append(f"ph_tok2:{ph_tokens[0]}_{ph_tokens[1]}")
    elif len(ph_tokens) == 1 and len(ph_tokens[0]) >= 3:
        keys.append(f"ph_tok1:{ph_tokens[0]}")
        
    return keys

def get_enhanced_address_keys(norm_addr: str) -> List[str]:
    """Extract fine-grained address blocking keys."""
    if not norm_addr:
        return []
    
    keys = []
    tokens = norm_addr.split()
    
    # Extract numbers and distinctive street/locality tokens
    nums = []
    words = []
    for t in tokens:
        if t.isdigit() and 1 <= len(t) <= 8:
            nums.append(str(int(t)))
        elif len(t) >= 4 and t not in ADDRESS_ABBREVIATIONS.values():
            words.append(t)
            
    # Compound Number + Street Word (e.g., 1795 + westchester, 17560 + ellis)
    for n in nums[:2]:
        for w in words[:2]:
            keys.append(f"num_word:{n}_{w}")
            
    # Distinctive Locality/City + Street Word if no numbers
    if len(words) >= 2:
        keys.append(f"loc_pair:{words[-1]}_{words[0]}")
        
    return keys

# -------------------------------------------------------------
# Parameterized Inverted Index Blocker for Ablation
# -------------------------------------------------------------

class ModularAblationBlocker:
    def __init__(
        self,
        use_translit: bool = False,
        use_enhanced_address: bool = False,
        use_tiered_quotas: bool = False,
        use_freq_aware: bool = False,
        max_key_freq: int = 500,
        max_candidates: int = 140,
    ):
        self.use_translit = use_translit
        self.use_enhanced_address = use_enhanced_address
        self.use_tiered_quotas = use_tiered_quotas
        self.use_freq_aware = use_freq_aware
        self.max_key_freq = max_key_freq
        self.max_candidates = max_candidates
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def extract_record_keys(self, row: dict) -> List[str]:
        keys = []
        c_name = row["core_name"]
        n_name = row["norm_name"]
        n_addr = row["norm_address"]
        raw_name = row.get("business_name", "")
        
        # Base keys
        keys.extend(extract_name_blocking_keys(c_name, n_name))
        keys.extend(extract_address_blocking_keys(n_addr))
        
        # Translit keys
        if self.use_translit:
            keys.extend(get_translit_blocking_keys(raw_name))
            
        # Enhanced Address keys
        if self.use_enhanced_address:
            keys.extend(get_enhanced_address_keys(n_addr))
            
        return keys

    def add_target_records(self, df_targets: pl.DataFrame):
        for row in df_targets.iter_rows(named=True):
            country = row["country"].strip() if row["country"] else ""
            if not country:
                continue
            rec_id = row["entity_id"]
            keys = self.extract_record_keys(row)
            c_idx = self.index[country]
            for k in keys:
                c_idx[k].append(rec_id)
                
    def prune_and_weigh_keys(self):
        for country, keys_dict in self.index.items():
            to_delete = []
            for k, id_list in keys_dict.items():
                if self.use_freq_aware:
                    # Specific high-value keys (exact name, PIN, house number) tolerate higher frequency
                    if k.startswith(("name:", "pin:", "hnum:", "tr_name:")):
                        if len(id_list) > 1000:
                            to_delete.append(k)
                    elif len(id_list) > self.max_key_freq:
                        to_delete.append(k)
                else:
                    if len(id_list) > self.max_key_freq:
                        to_delete.append(k)
            for k in to_delete:
                del keys_dict[k]

    def _get_key_quota(self, k: str) -> int:
        if not self.use_tiered_quotas:
            return 30
        # Tier 1: High Specificity (Exact names, PIN, House number, Translit name, Compound address)
        if k.startswith(("name:", "pin:", "hnum:", "tr_name:", "addr:", "num_word:")):
            return 40
        # Tier 2: Medium Specificity (Token pairs, Prefix 4, First-Last, Phonetic pairs)
        if k.startswith(("tok3:", "tok2:", "ph_tok2:", "tr_tok2:", "fl:", "sort2:", "pfx4:", "tr_pfx4:")):
            return 20
        # Tier 3: Low Specificity / Broad (3-char prefix, single tokens, phonetic single)
        if k.startswith(("pfx3:", "num:", "tok1:", "ph_tok1:", "tr_tok1:", "loc_pair:")):
            return 8
        return 12

    def _key_priority(self, k: str) -> int:
        if k.startswith("name:"):
            return 1
        if k.startswith("tr_name:"):
            return 2
        if k.startswith("pin:"):
            return 3
        if k.startswith("hnum:"):
            return 4
        if k.startswith("addr:"):
            return 5
        if k.startswith("num_word:"):
            return 6
        if k.startswith("tok3:"):
            return 7
        if k.startswith("tok2:"):
            return 8
        if k.startswith("ph_tok2:"):
            return 9
        if k.startswith("tr_tok2:"):
            return 10
        if k.startswith("fl:"):
            return 11
        if k.startswith("sort2:"):
            return 12
        if k.startswith("pfx4:"):
            return 13
        if k.startswith("tr_pfx4:"):
            return 14
        if k.startswith("pfx3:"):
            return 15
        if k.startswith("loc_pair:"):
            return 16
        if k.startswith("num:"):
            return 17
        if k.startswith("tok1:"):
            return 18
        if k.startswith("ph_tok1:"):
            return 19
        if k.startswith("tr_tok1:"):
            return 20
        return 22

    def generate_candidates(self, df_s1: pl.DataFrame) -> Dict[str, Set[str]]:
        candidates = {}
        for row in df_s1.iter_rows(named=True):
            s1_id = row["entity_id"]
            country = row["country"].strip() if row["country"] else ""
            cand_set = set()
            
            if country in self.index:
                c_idx = self.index[country]
                keys = self.extract_record_keys(row)
                keys.sort(key=self._key_priority)
                
                for k in keys:
                    if k in c_idx:
                        id_list = c_idx[k]
                        quota = self._get_key_quota(k)
                        added = 0
                        for tid in id_list:
                            if tid not in cand_set:
                                cand_set.add(tid)
                                added += 1
                                if added >= quota:
                                    break
                            if len(cand_set) >= self.max_candidates:
                                break
                    if len(cand_set) >= self.max_candidates:
                        break
                        
            candidates[s1_id] = cand_set
        return candidates

def run_failure_analysis_v2(eval_gt, s1_dict, target_lookup, candidates):
    """Diagnose the exact missed pairs on the 2,000 S1 validation set."""
    failure_records = []
    category_counts = Counter()
    
    for s1_id, true_set in eval_gt.items():
        cand_set = candidates.get(s1_id, set())
        missed = true_set - cand_set
        s1_rec = s1_dict[s1_id]
        s1_country = s1_rec["country"].strip()
        
        s1_keys = set(extract_name_blocking_keys(s1_rec["core_name"], s1_rec["norm_name"]))
        s1_keys.update(extract_address_blocking_keys(s1_rec["norm_address"]))
        
        for tid in missed:
            if tid not in target_lookup:
                cat = "malformed_or_missing_record"
                tgt_rec = {"business_name": "", "business_address": "", "country": "", "core_name": "", "norm_name": "", "norm_address": ""}
                tgt_keys = set()
            else:
                tgt_rec = target_lookup[tid]
                tgt_keys = set(extract_name_blocking_keys(tgt_rec["core_name"], tgt_rec["norm_name"]))
                tgt_keys.update(extract_address_blocking_keys(tgt_rec["norm_address"]))
                
                # Assign failure root cause
                has_indic = any(ord(c) >= 0x0900 for c in s1_rec["business_name"] + tgt_rec["business_name"])
                if has_indic:
                    cat = "multilingual_transliteration"
                elif len(s1_keys.intersection(tgt_keys)) > 0:
                    cat = "candidate_capacity_quota"
                elif not s1_rec["norm_address"] or not tgt_rec["norm_address"]:
                    cat = "address_normalization"
                else:
                    cat = "no_blocking_overlap"
                    
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
            
    out_csv = OUTPUT_DIR / "blocking_failure_analysis_v2.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source1_id", "target_id", "source", "country",
            "name_s1", "name_target", "address_s1", "address_target",
            "failure_category", "blocking_keys_s1", "blocking_keys_target"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_records)
        
    print(f"\n[Task 1] Saved Failure Analysis v2 to: {out_csv} ({len(failure_records)} missed pairs)")
    print("Aggregate Missed Categories:")
    for cat, count in category_counts.most_common():
        pct = (count / len(failure_records)) * 100 if failure_records else 0.0
        print(f"  {cat:32s}: {count:5d} ({pct:5.2f}%)")

def main():
    print("=" * 70)
    print("STAGE 4: CANDIDATE-GENERATION ABLATION STUDY (2,000 S1 ENTITIES)")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Load exact validation S1 IDs
    val_parquet_path = OUTPUT_DIR / "validation_features.parquet"
    df_val_feats = pl.read_parquet(val_parquet_path)
    val_s1_ids = set(df_val_feats["s1_id"].unique().to_list())
    
    # 2. Load Ground Truth
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    eval_gt = {s1: gt_map.get(s1, set()) for s1 in val_s1_ids}
    total_true_matches = sum(len(m) for m in eval_gt.values())
    
    print(f"Loaded Validation Cohort: {len(val_s1_ids):,} S1 Entities ({total_true_matches:,} Ground Truth Matches)")
    
    # 3. Load S1 and Target pool
    needed_tids = set()
    for t_set in eval_gt.values():
        needed_tids.update(t_set)
        
    df_s1_all = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    val_s1_df = df_s1_all.filter(pl.col("entity_id").is_in(list(val_s1_ids)))
    s1_dict = {r["entity_id"]: r for r in val_s1_df.iter_rows(named=True)}
    
    def extract_targets(path, needed_ids, max_distractors=125000):
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
        
    s2_rows = extract_targets(TRAIN_SOURCE2, needed_tids, max_distractors=125000)
    s3_rows = extract_targets(TRAIN_SOURCE3, needed_tids, max_distractors=125000)
    
    df_targets = pl.DataFrame(
        s2_rows + s3_rows,
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
    
    target_lookup = {r["entity_id"]: r for r in df_targets.iter_rows(named=True)}
    print(f"Target Pool Assembled: {len(df_targets):,} records")
    
    # 4. Ablation Study Configurations
    configs = [
        ("1. Baseline Stage 2", False, False, False, False, 100),
        ("2. + Transliteration", True, False, False, False, 100),
        ("3. + Address Improvements", False, True, False, False, 100),
        ("4. + Quota Improvements", False, False, True, False, 140),
        ("5. + Frequency-Aware Blocking", False, False, False, True, 100),
        ("6. + ALL COMBINED (Optimal)", True, True, True, True, 160),
    ]
    
    ablation_results = []
    
    for name, tr, addr, quota, freq, max_cands in configs:
        t_start = time.time()
        blocker = ModularAblationBlocker(
            use_translit=tr,
            use_enhanced_address=addr,
            use_tiered_quotas=quota,
            use_freq_aware=freq,
            max_key_freq=500,
            max_candidates=max_cands,
        )
        blocker.add_target_records(df_targets)
        blocker.prune_and_weigh_keys()
        
        cands = blocker.generate_candidates(val_s1_df)
        elapsed = time.time() - t_start
        
        # Calculate failure analysis v2 on baseline
        if name == "1. Baseline Stage 2":
            run_failure_analysis_v2(eval_gt, s1_dict, target_lookup, cands)
            
        cand_metrics = evaluate_candidate_recall(eval_gt, cands)
        cand_counts = [len(c) for c in cands.values()]
        
        captured = cand_metrics["captured_matches"]
        missed = total_true_matches - captured
        recall = cand_metrics["candidate_recall_ceiling"]
        avg_cands = float(np.mean(cand_counts))
        med_cands = float(np.median(cand_counts))
        p95_cands = float(np.percentile(cand_counts, 95))
        p99_cands = float(np.percentile(cand_counts, 99))
        max_c = int(np.max(cand_counts))
        total_cands = int(sum(cand_counts))
        
        ablation_results.append({
            "configuration": name,
            "candidate_recall": recall,
            "captured_matches": captured,
            "missed_matches": missed,
            "avg_candidates_per_s1": avg_cands,
            "median_candidates_per_s1": med_cands,
            "p95_candidates_per_s1": p95_cands,
            "p99_candidates_per_s1": p99_cands,
            "max_candidates_per_s1": max_c,
            "total_candidates": total_cands,
            "runtime_seconds": elapsed,
        })
        
    # Print Ablation Table
    print("\n" + "=" * 95)
    print("CANDIDATE-GENERATION ABLATION STUDY RESULTS (2,000 S1 Entities, 6,876 True Matches):")
    print(f"{'Configuration':<30s} | {'Recall':>8s} | {'Captured':>8s} | {'Missed':>6s} | {'Avg Cand':>8s} | {'P95':>5s} | {'P99':>5s} | {'Max':>4s} | {'Time':>6s}")
    print("-" * 95)
    for r in ablation_results:
        print(f"{r['configuration']:<30s} | {r['candidate_recall']*100:7.2f}% | {r['captured_matches']:8,d} | {r['missed_matches']:6,d} | {r['avg_candidates_per_s1']:8.2f} | {r['p95_candidates_per_s1']:5.0f} | {r['p99_candidates_per_s1']:5.0f} | {r['max_candidates_per_s1']:4d} | {r['runtime_seconds']:5.2f}s")
    print("=" * 95)
    
    # Save Ablation Results
    ablation_csv_path = OUTPUT_DIR / "blocking_ablation_results.csv"
    pl.DataFrame(ablation_results).write_csv(ablation_csv_path)
    print(f"\nSaved Ablation Results to: {ablation_csv_path}")

if __name__ == "__main__":
    main()
