import os
import sys
import time
import random
from pathlib import Path
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
from src.data_loader import load_and_normalize_source, load_ground_truth
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)
from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features

def build_training_and_val_datasets(
    n_total_s1: int = 10000,
    train_ratio: float = 0.8,
    max_hard_negatives_per_s1: int = 10,
    random_seed: int = 42,
):
    print("=" * 65)
    print(f"TASK 4 & 5: GENERATING LIGHTGBM TRAINING & VALIDATION DATASETS")
    print(f"Total S1 sample: {n_total_s1:,} ({int(n_total_s1*train_ratio):,} Train / {int(n_total_s1*(1-train_ratio)):,} Val)")
    print("=" * 65)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    t_start = time.time()
    
    # 1. Load Ground Truth
    print("\n1. Loading Ground Truth...")
    gt_map = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    # 2. Load Source 1 Sample
    print(f"2. Loading & Normalizing {n_total_s1:,} Source 1 records...")
    df_s1 = load_and_normalize_source(TRAIN_SOURCE1, n_rows=n_total_s1)
    s1_rows = df_s1.iter_rows(named=True)
    all_s1_records = {r["entity_id"]: r for r in s1_rows}
    
    # Collect all needed true target IDs
    needed_target_ids = set()
    for s1_id in all_s1_records.keys():
        needed_target_ids.update(gt_map.get(s1_id, set()))
        
    print(f"  Total S1 entities: {len(all_s1_records):,} ({len(needed_target_ids):,} ground truth target IDs)")
    
    # 3. Stream & Load Target Records (True Targets + 250,000 Distractors)
    print("\n3. Loading & Normalizing Target Records (True Targets + 250,000 Distractors)...")
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
        
    t_load = time.time()
    s2_rows = extract_targets(TRAIN_SOURCE2, needed_target_ids, max_distractors=125000)
    s3_rows = extract_targets(TRAIN_SOURCE3, needed_target_ids, max_distractors=125000)
    
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
    print(f"  Target pool assembled: {len(df_targets):,} records in {time.time() - t_load:.2f}s")
    
    # 4. Inverted Index Blocking
    print("\n4. Indexing & Generating Candidate Pairs...")
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=120, max_per_key=25)
    blocker.add_target_records(df_targets)
    blocker.prune_high_frequency_keys()
    
    candidates = blocker.generate_candidates_for_s1(df_s1)
    
    # 5. Split S1 IDs into Train / Val (Disjoint sets to prevent leakage)
    all_s1_id_list = list(all_s1_records.keys())
    random.shuffle(all_s1_id_list)
    
    split_idx = int(len(all_s1_id_list) * train_ratio)
    train_s1_ids = set(all_s1_id_list[:split_idx])
    val_s1_ids = set(all_s1_id_list[split_idx:])
    
    print(f"\n5. Train/Val S1 Split: {len(train_s1_ids):,} Train S1s, {len(val_s1_ids):,} Validation S1s")
    
    def extract_dataset_rows(s1_id_subset: Set[str], desc: str) -> pl.DataFrame:
        print(f"\nExtracting features for {desc} ({len(s1_id_subset):,} S1 entities)...")
        rows = []
        pos_count = 0
        neg_count = 0
        
        for s1_id in s1_id_subset:
            s1_rec = all_s1_records[s1_id]
            s1_norm_name = s1_rec["norm_name"]
            s1_core_name = s1_rec["core_name"]
            s1_norm_addr = s1_rec["norm_address"]
            s1_country = s1_rec["country"]
            
            true_matches = gt_map.get(s1_id, set())
            cand_ids = candidates.get(s1_id, set())
            
            # 1. Positives (all true matches present in target pool)
            for tid in true_matches:
                if tid not in target_lookup:
                    continue
                tgt = target_lookup[tid]
                feats = extract_pairwise_features(
                    s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                    tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                    tid,
                )
                feats["s1_id"] = s1_id
                feats["target_id"] = tid
                feats["label"] = 1
                rows.append(feats)
                pos_count += 1
                
            # 2. Hard Negatives (candidate pairs that are not true matches)
            cand_negatives = [tid for tid in cand_ids if tid not in true_matches and tid in target_lookup]
            
            # Score negatives by quick similarity to prioritize hard negatives
            neg_feats_list = []
            for tid in cand_negatives:
                tgt = target_lookup[tid]
                feats = extract_pairwise_features(
                    s1_norm_name, s1_core_name, s1_norm_addr, s1_country,
                    tgt["norm_name"], tgt["core_name"], tgt["norm_address"], tgt["country"],
                    tid,
                )
                # Hardness score = name similarity + address similarity
                hardness = feats["name_wratio"] + feats["core_token_set"] + feats["addr_token_set"]
                neg_feats_list.append((hardness, feats, tid))
                
            # Sort by hardness descending and take top N hard negatives
            neg_feats_list.sort(key=lambda x: x[0], reverse=True)
            for _, feats, tid in neg_feats_list[:max_hard_negatives_per_s1]:
                feats["s1_id"] = s1_id
                feats["target_id"] = tid
                feats["label"] = 0
                rows.append(feats)
                neg_count += 1
                
        print(f"  {desc} Extracted: {len(rows):,} pairs ({pos_count:,} Positives, {neg_count:,} Hard Negatives, Pos:Neg = 1:{neg_count/max(1,pos_count):.2f})")
        
        # Build Polars DataFrame with optimized compact dtypes
        df_feat = pl.DataFrame(rows)
        
        # Cast columns to compact numeric types
        float_cols = [
            "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
            "name_token_sort", "name_token_set", "name_partial", "core_ratio",
            "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
            "addr_missing", "addr_exact", "addr_ratio", "addr_token_sort",
            "addr_token_set", "addr_jaccard", "addr_char_3gram", "addr_num_match",
            "country_match", "is_source2", "is_source3",
        ]
        
        cast_exprs = [pl.col(c).cast(pl.Float32) for c in float_cols if c in df_feat.columns]
        cast_exprs.append(pl.col("name_len_diff").cast(pl.Int16))
        cast_exprs.append(pl.col("addr_len_diff").cast(pl.Int16))
        cast_exprs.append(pl.col("label").cast(pl.UInt8))
        
        df_feat = df_feat.with_columns(cast_exprs)
        return df_feat
        
    train_df = extract_dataset_rows(train_s1_ids, "Train Set")
    val_df = extract_dataset_rows(val_s1_ids, "Validation Set")
    
    # 6. Save Parquet Files
    train_parquet_path = OUTPUT_DIR / "train_features.parquet"
    val_parquet_path = OUTPUT_DIR / "validation_features.parquet"
    
    print("\n6. Saving Parquet Files to Disk...")
    train_df.write_parquet(train_parquet_path, compression="zstd")
    val_df.write_parquet(val_parquet_path, compression="zstd")
    
    train_size_mb = os.path.getsize(train_parquet_path) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_parquet_path) / (1024 * 1024)
    
    print("=" * 65)
    print("DATASET GENERATION SUMMARY:")
    print(f"  Train Parquet:       {train_parquet_path} ({len(train_df):,} rows, {train_size_mb:.2f} MB)")
    print(f"  Validation Parquet:  {val_parquet_path} ({len(val_df):,} rows, {val_size_mb:.2f} MB)")
    print(f"  Total Features:      {len(train_df.columns) - 3} similarity features")
    print(f"  Execution Time:      {time.time() - t_start:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    build_training_and_val_datasets(n_total_s1=10000)
