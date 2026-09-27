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
from src.phonetic import phonetic_skeleton
from src.blocking import InvertedIndexBlocker, get_translit_blocking_keys, get_enhanced_address_keys
from src.evaluate import evaluate_candidate_recall

# -------------------------------------------------------------
# STEP 1: Exact Failure Analysis v3 on the 343 Missed Pairs
# -------------------------------------------------------------

def analyze_stage4_misses(eval_gt, val_s1_df, df_targets, target_lookup, baseline_cands):
    print("\n" + "=" * 70)
    print("STEP 1: DETAILED FAILURE ANALYSIS V3 ON STAGE-4 MISSED PAIRS")
    print("=" * 70)
    
    s1_dict = {r["entity_id"]: r for r in val_s1_df.iter_rows(named=True)}
    
    # Rebuild Stage 4 index with unpruned frequency tracking to detect frequency pruning
    blocker_stage4 = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    blocker_stage4.add_target_records(df_targets)
    
    raw_key_freqs = {
        country: {k: len(v) for k, v in keys_dict.items()}
        for country, keys_dict in blocker_stage4.index.items()
    }
    blocker_stage4.prune_high_frequency_keys()
    
    failure_records = []
    category_counts = Counter()
    
    for s1_id, true_set in eval_gt.items():
        cand_set = baseline_cands.get(s1_id, set())
        missed_set = true_set - cand_set
        if not missed_set:
            continue
            
        s1_rec = s1_dict[s1_id]
        s1_country = s1_rec["country"].strip()
        s1_keys = set(blocker_stage4._extract_all_keys(s1_rec["norm_name"], s1_rec["core_name"], s1_rec["norm_address"], s1_rec["business_name"]))
        
        for tid in missed_set:
            if tid not in target_lookup:
                cat = "malformed_or_missing_record"
                evidence = "Target ID missing from target record pool."
                tgt_rec = {"business_name": "", "business_address": "", "country": "", "core_name": "", "norm_name": "", "norm_address": ""}
                tgt_keys = set()
                name_sim = 0.0
                tr_sim = 0.0
                ph_sim = 0.0
                addr_sim = 0.0
                num_overlap = 0
            else:
                tgt_rec = target_lookup[tid]
                tgt_country = tgt_rec["country"].strip()
                tgt_keys = set(blocker_stage4._extract_all_keys(tgt_rec["norm_name"], tgt_rec["core_name"], tgt_rec["norm_address"], tgt_rec["business_name"]))
                
                # Similarities
                name_sim = fuzz.token_set_ratio(s1_rec["core_name"], tgt_rec["core_name"])
                
                # Translit similarity
                tr_s1 = transliterate_indic_to_latin(s1_rec["business_name"])
                tr_tgt = transliterate_indic_to_latin(tgt_rec["business_name"])
                tr_sim = fuzz.token_set_ratio(tr_s1, tr_tgt)
                
                # Phonetic similarity
                ph_s1 = " ".join(phonetic_skeleton(t) for t in tr_s1.split())
                ph_tgt = " ".join(phonetic_skeleton(t) for t in tr_tgt.split())
                ph_sim = fuzz.token_set_ratio(ph_s1, ph_tgt)
                
                # Address similarity
                addr_sim = fuzz.token_set_ratio(s1_rec["norm_address"], tgt_rec["norm_address"]) if (s1_rec["norm_address"] and tgt_rec["norm_address"]) else 0.0
                
                s1_nums = {t for t in s1_rec["norm_address"].split() if t.isdigit()}
                tgt_nums = {t for t in tgt_rec["norm_address"].split() if t.isdigit()}
                num_overlap = len(s1_nums.intersection(tgt_nums))
                
                # Key Overlap & Pruning / Quota Analysis
                shared_keys = s1_keys.intersection(tgt_keys)
                has_shared_key = len(shared_keys) > 0
                
                # Check if shared key was pruned due to frequency
                was_freq_pruned = False
                if has_shared_key:
                    for k in shared_keys:
                        if raw_key_freqs.get(s1_country, {}).get(k, 0) > 500 and not k.startswith(("name:", "pin:", "hnum:", "tr_name:")):
                            was_freq_pruned = True
                            break
                            
                has_indic = any(ord(c) >= 0x0900 for c in s1_rec["business_name"] + tgt_rec["business_name"])
                
                # Assign failure category with evidence
                if s1_country != tgt_country:
                    cat = "country_mismatch"
                    evidence = f"Country label disagreement: S1='{s1_country}' vs Tgt='{tgt_country}'"
                elif has_shared_key and was_freq_pruned:
                    cat = "frequency_pruning"
                    evidence = f"Shared key(s) {shared_keys} exceeded frequency ceiling > 500 and was pruned."
                elif has_shared_key and not was_freq_pruned:
                    cat = "quota_eviction"
                    evidence = f"Shared key(s) {shared_keys} existed in index, but target was pushed beyond S1 quota limit."
                elif has_indic and ph_sim < 65:
                    cat = "transliteration_failure"
                    evidence = f"Indic script with divergent phonetic token rendering: S1='{s1_rec['business_name']}' vs Tgt='{tgt_rec['business_name']}' (ph_sim={ph_sim:.1f})"
                elif has_indic and ph_sim >= 65:
                    cat = "phonetic_failure"
                    evidence = f"Phonetic similarity is high ({ph_sim:.1f}) but no exact phonetic token-pair key overlapped."
                elif (s1_rec["norm_address"] and tgt_rec["norm_address"]) and (addr_sim >= 75 or num_overlap > 0):
                    cat = "address_key_failure"
                    evidence = f"Address similarity is high ({addr_sim:.1f}, num_overlap={num_overlap}) but address tokens were formatted/ordered differently without shared compound key."
                elif name_sim >= 75 or ph_sim >= 75:
                    cat = "abbreviation_or_acronym_failure"
                    evidence = f"Name tokens have acronym, abbreviation, or word transposition variation (name_sim={name_sim:.1f})."
                elif not s1_rec["norm_address"] or not tgt_rec["norm_address"]:
                    cat = "normalization_failure"
                    evidence = "One of the records has an empty/blank address string with no shared name tokens."
                else:
                    cat = "no_blocking_overlap"
                    evidence = f"Divergent name and address representations: name_sim={name_sim:.1f}, addr_sim={addr_sim:.1f}"
                    
            category_counts[cat] += 1
            failure_records.append({
                "source1_id": s1_id,
                "target_id": tid,
                "source": "S2" if tid.startswith("S2-") else "S3",
                "country": s1_country,
                "name_s1": s1_rec["business_name"],
                "name_target": tgt_rec.get("business_name", ""),
                "address_s1": s1_rec["business_address"],
                "address_target": tgt_rec.get("business_address", ""),
                "blocking_keys_s1": "; ".join(sorted(s1_keys)),
                "blocking_keys_target": "; ".join(sorted(tgt_keys)),
                "has_shared_key": int(len(s1_keys.intersection(tgt_keys)) > 0),
                "name_sim": name_sim,
                "translit_sim": tr_sim,
                "phonetic_sim": ph_sim,
                "address_sim": addr_sim,
                "num_overlap": num_overlap,
                "failure_category": cat,
                "evidence_reason": evidence,
            })
            
    out_csv = OUTPUT_DIR / "blocking_failure_analysis_v3.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source1_id", "target_id", "source", "country",
            "name_s1", "name_target", "address_s1", "address_target",
            "has_shared_key", "name_sim", "translit_sim", "phonetic_sim", "address_sim", "num_overlap",
            "failure_category", "evidence_reason", "blocking_keys_s1", "blocking_keys_target"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_records)
        
    print(f"Saved Failure Analysis v3 to: {out_csv} ({len(failure_records):,} missed pairs)")
    print("\nFailure Category Breakdown of the 343 Misses:")
    for cat, count in category_counts.most_common():
        pct = (count / len(failure_records)) * 100 if failure_records else 0.0
        print(f"  {cat:32s}: {count:5d} ({pct:5.2f}%)")
        
    return failure_records

# -------------------------------------------------------------
# STEP 2 & 3: Targeted Stage 5 Improvements
# -------------------------------------------------------------

def get_stage5_acronym_and_token_keys(name: str) -> List[str]:
    """Extract acronym keys and 3-gram character prefix keys for abbreviation matching."""
    if not name:
        return []
    
    # Transliterate first if Indic
    has_indic = any(ord(c) >= 0x0900 for c in name)
    latin_name = transliterate_indic_to_latin(name) if has_indic else name
    core = extract_core_business_name(normalize_business_name(latin_name))
    tokens = [t for t in core.split() if len(t) >= 2 and t not in GENERIC_STOP_WORDS]
    
    keys = []
    # 1. Acronym / Initials key if >= 2 words (e.g., Tata Consultancy Services -> tcs)
    if len(tokens) >= 2:
        acronym = "".join(t[0] for t in tokens if t[0].isalnum())
        if 2 <= len(acronym) <= 6:
            keys.append(f"acro:{acronym}")
            
    # 2. First 3 non-generic tokens combined (e.g. tok3:t1_t2_t3)
    if len(tokens) >= 3:
        keys.append(f"tok3_ord:{tokens[0]}_{tokens[1]}_{tokens[2]}")
        
    # 3. Rare 3-char prefix on distinctive first token
    if tokens and len(tokens[0]) >= 4 and tokens[0] not in GENERIC_STOP_WORDS:
        keys.append(f"pfx3_tok1:{tokens[0][:3]}")
        
    return keys

def get_stage5_address_keys(norm_addr: str) -> List[str]:
    """Extract robust street + postal token combinations."""
    if not norm_addr:
        return []
    
    keys = []
    tokens = norm_addr.split()
    
    # Extract distinct non-generic alphanumeric words
    distinct_words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_ABBREVIATIONS.values()]
    numbers = [str(int(t)) for t in tokens if t.isdigit() and 1 <= len(t) <= 8]
    
    # 1. Number + First distinct word
    if numbers and distinct_words:
        keys.append(f"num_w1:{numbers[0]}_{distinct_words[0]}")
        if len(distinct_words) >= 2:
            keys.append(f"num_w2:{numbers[0]}_{distinct_words[-1]}")
            
    # 2. First two distinct address words (e.g. street + city)
    if len(distinct_words) >= 2:
        keys.append(f"addr_word2:{distinct_words[0]}_{distinct_words[1]}")
        
    return keys

class Stage5ModularBlocker:
    def __init__(
        self,
        use_stage5_acronyms: bool = False,
        use_stage5_address: bool = False,
        use_stage5_phonetic_expansion: bool = False,
        max_candidates: int = 160,
    ):
        self.use_stage5_acronyms = use_stage5_acronyms
        self.use_stage5_address = use_stage5_address
        self.use_stage5_phonetic_expansion = use_stage5_phonetic_expansion
        self.max_candidates = max_candidates
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def _extract_all_keys(self, n_name: str, c_name: str, n_addr: str, raw_name: str = "") -> List[str]:
        # Start with Stage 4 baseline keys
        keys = []
        keys.extend(extract_name_blocking_keys(c_name, n_name))
        keys.extend(extract_address_blocking_keys(n_addr))
        keys.extend(get_translit_blocking_keys(raw_name if raw_name else n_name))
        keys.extend(get_enhanced_address_keys(n_addr))
        
        # Stage 5 Additions
        if self.use_stage5_acronyms:
            keys.extend(get_stage5_acronym_and_token_keys(raw_name if raw_name else n_name))
        if self.use_stage5_address:
            keys.extend(get_stage5_address_keys(n_addr))
        if self.use_stage5_phonetic_expansion:
            # Phonetic first token + address number
            has_indic = any(ord(c) >= 0x0900 for c in (raw_name or n_name))
            latin_name = transliterate_indic_to_latin(raw_name or n_name) if has_indic else (raw_name or n_name)
            core = extract_core_business_name(normalize_business_name(latin_name))
            tokens = [t for t in core.split() if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
            addr_nums = [str(int(t)) for t in n_addr.split() if t.isdigit() and 1 <= len(t) <= 8]
            if tokens and addr_nums:
                ph = phonetic_skeleton(tokens[0])
                keys.append(f"ph_num:{ph}_{addr_nums[0]}")
                
        return keys

    def add_target_records(self, df_targets: pl.DataFrame):
        for row in df_targets.iter_rows(named=True):
            country = row["country"].strip() if row["country"] else ""
            if not country:
                continue
            rec_id = row["entity_id"]
            keys = self._extract_all_keys(row["norm_name"], row["core_name"], row["norm_address"], row.get("business_name", ""))
            c_idx = self.index[country]
            for k in keys:
                c_idx[k].append(rec_id)
                
    def prune_high_frequency_keys(self):
        for country, keys_dict in self.index.items():
            to_delete = []
            for k, id_list in keys_dict.items():
                if k.startswith(("name:", "pin:", "hnum:", "tr_name:", "acro:", "num_w1:")):
                    if len(id_list) > 1000:
                        to_delete.append(k)
                elif len(id_list) > 500:
                    to_delete.append(k)
            for k in to_delete:
                del keys_dict[k]

    @staticmethod
    def _key_priority(k: str) -> int:
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
        if k.startswith("num_w1:"):
            return 7
        if k.startswith("num_w2:"):
            return 8
        if k.startswith("tok3:"):
            return 9
        if k.startswith("tok3_ord:"):
            return 10
        if k.startswith("tok2:"):
            return 11
        if k.startswith("ph_tok2:"):
            return 12
        if k.startswith("tr_tok2:"):
            return 13
        if k.startswith("fl:"):
            return 14
        if k.startswith("sort2:"):
            return 15
        if k.startswith("pfx4:"):
            return 16
        if k.startswith("tr_pfx4:"):
            return 17
        if k.startswith("ph_num:"):
            return 18
        if k.startswith("acro:"):
            return 19
        if k.startswith("pfx3:"):
            return 20
        if k.startswith("pfx3_tok1:"):
            return 21
        if k.startswith("addr_word2:"):
            return 22
        if k.startswith("loc_pair:"):
            return 23
        if k.startswith("num:"):
            return 24
        if k.startswith("tok1:"):
            return 25
        if k.startswith("ph_tok1:"):
            return 26
        if k.startswith("tr_tok1:"):
            return 27
        return 30

    @staticmethod
    def _get_key_quota(k: str) -> int:
        # High specificity
        if k.startswith(("name:", "pin:", "hnum:", "tr_name:", "addr:", "num_word:", "num_w1:", "num_w2:", "ph_num:")):
            return 40
        # Medium specificity
        if k.startswith(("tok3:", "tok3_ord:", "tok2:", "ph_tok2:", "tr_tok2:", "fl:", "sort2:", "pfx4:", "tr_pfx4:", "acro:")):
            return 20
        # Broad specificity
        if k.startswith(("pfx3:", "pfx3_tok1:", "addr_word2:", "num:", "tok1:", "ph_tok1:", "tr_tok1:", "loc_pair:")):
            return 8
        return 12

    def generate_candidates_for_s1(self, df_s1: pl.DataFrame) -> Dict[str, Set[str]]:
        candidates = {}
        for row in df_s1.iter_rows(named=True):
            s1_id = row["entity_id"]
            country = row["country"].strip() if row["country"] else ""
            cand_set = set()
            
            if country in self.index:
                c_idx = self.index[country]
                keys = self._extract_all_keys(row["norm_name"], row["core_name"], row["norm_address"], row.get("business_name", ""))
                keys.sort(key=self._key_priority)
                
                for k in keys:
                    if k in c_idx:
                        quota = self._get_key_quota(k)
                        added = 0
                        for tid in c_idx[k]:
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

# -------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------

def main():
    print("=" * 75)
    print("STAGE 5: RECOVERING REMAINING MISSED GROUND-TRUTH PAIRS (2,000 S1 COHORT)")
    print("=" * 75)
    
    t0 = time.time()
    
    # 1. Load Validation IDs and Ground Truth
    val_parquet_path = OUTPUT_DIR / "validation_features.parquet"
    df_val_feats = pl.read_parquet(val_parquet_path)
    val_s1_ids = set(df_val_feats["s1_id"].unique().to_list())
    
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    eval_gt = {s1: gt_map.get(s1, set()) for s1 in val_s1_ids}
    total_true_matches = sum(len(m) for m in eval_gt.values())
    
    # 2. Load S1 and Target pool
    needed_tids = set()
    for t_set in eval_gt.values():
        needed_tids.update(t_set)
        
    df_s1_all = load_and_normalize_source(TRAIN_SOURCE1, n_rows=10000)
    val_s1_df = df_s1_all.filter(pl.col("entity_id").is_in(list(val_s1_ids)))
    
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
    
    # 3. Stage 4 Baseline Run
    print("\nRunning Stage-4 Immutable Baseline...")
    b4 = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=160)
    b4.add_target_records(df_targets)
    b4.prune_high_frequency_keys()
    stage4_cands = b4.generate_candidates_for_s1(val_s1_df)
    m4 = evaluate_candidate_recall(eval_gt, stage4_cands)
    stage4_captured = m4["captured_matches"]
    stage4_missed = total_true_matches - stage4_captured
    print(f"Stage-4 Baseline Recall: {m4['candidate_recall_ceiling']*100:.2f}% ({stage4_captured:,} captured, {stage4_missed:,} missed)")
    
    # 4. Failure Analysis v3
    analyze_stage4_misses(eval_gt, val_s1_df, df_targets, target_lookup, stage4_cands)
    
    # 5. Ablation Configurations for Stage 5
    print("\n" + "=" * 75)
    print("STEP 3: STAGE-5 CANDIDATE-GENERATION ABLATION STUDY")
    print("=" * 75)
    
    configs = [
        ("Stage-4 Baseline", False, False, False, 160),
        ("+ Acronym & Ordered Tokens", True, False, False, 160),
        ("+ Address Word Combinations", False, True, False, 160),
        ("+ Phonetic-Address Compound", False, False, True, 160),
        ("+ Acronym + Address Combined", True, True, False, 160),
        ("+ ALL STAGE-5 COMBINED (Optimal)", True, True, True, 160),
    ]
    
    ablation_stage5_rows = []
    
    for c_name, acro, addr, ph_exp, max_c in configs:
        t_sub = time.time()
        blocker5 = Stage5ModularBlocker(
            use_stage5_acronyms=acro,
            use_stage5_address=addr,
            use_stage5_phonetic_expansion=ph_exp,
            max_candidates=max_c,
        )
        blocker5.add_target_records(df_targets)
        blocker5.prune_high_frequency_keys()
        
        cands = blocker5.generate_candidates_for_s1(val_s1_df)
        elapsed = time.time() - t_sub
        
        m = evaluate_candidate_recall(eval_gt, cands)
        counts = [len(c) for c in cands.values()]
        
        captured = m["captured_matches"]
        missed = total_true_matches - captured
        gain_pct = (m["candidate_recall_ceiling"] - m4["candidate_recall_ceiling"]) * 100
        recovered_count = captured - stage4_captured
        
        ablation_stage5_rows.append({
            "Configuration": c_name,
            "Candidate Recall": f"{m['candidate_recall_ceiling']*100:.2f}%",
            "Captured Matches": captured,
            "Missed Matches": missed,
            "Recovered Count": recovered_count,
            "Recall Gain": f"+{gain_pct:.2f}%" if gain_pct >= 0 else f"{gain_pct:.2f}%",
            "Avg Cand/S1": f"{np.mean(counts):.2f}",
            "Median": f"{np.median(counts):.0f}",
            "P95": f"{np.percentile(counts, 95):.0f}",
            "P99": f"{np.percentile(counts, 99):.0f}",
            "Max Cand": int(np.max(counts)),
            "Runtime": f"{elapsed:.2f}s",
        })
        
    # Print Stage 5 Ablation Table
    print("\n" + "=" * 105)
    print(f"{'Configuration':<35s} | {'Recall':>8s} | {'Captured':>8s} | {'Missed':>6s} | {'Recovered':>9s} | {'Avg Cand':>8s} | {'P95':>5s} | {'P99':>5s} | {'Max':>4s} | {'Time':>6s}")
    print("-" * 105)
    for r in ablation_stage5_rows:
        print(f"{r['Configuration']:<35s} | {r['Candidate Recall']:>8s} | {r['Captured Matches']:8,d} | {r['Missed Matches']:6,d} | {r['Recovered Count']:+9d} | {r['Avg Cand/S1']:>8s} | {r['P95']:>5s} | {r['P99']:>5s} | {r['Max Cand']:4d} | {r['Runtime']:>6s}")
    print("=" * 105)
    
    out_ablation_csv = OUTPUT_DIR / "blocking_ablation_stage5.csv"
    pl.DataFrame(ablation_stage5_rows).write_csv(out_ablation_csv)
    print(f"\nSaved Stage 5 Ablation Results to: {out_ablation_csv}")
    print(f"Total Execution Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
