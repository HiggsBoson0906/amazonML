import os
import sys
import time
from collections import Counter

# Set UTF-8 encoding for Windows console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

def inspect_tsv_file(filepath, expected_cols_count=None, is_ground_truth=False):
    print(f"\n=======================================================")
    print(f"Inspecting: {filepath}")
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    
    start_time = time.time()
    
    total_lines = 0
    blank_lines = 0
    malformed_lines = 0
    null_counts = Counter()
    country_counts = Counter()
    match_count_dist = Counter()
    sample_rows = []
    header = []
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        total_lines += 1
        if not header_line:
            print("ERROR: Empty file!")
            return
        
        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        print(f"Header ({len(header)} columns): {header}")
        
        expected_len = expected_cols_count if expected_cols_count else len(header)
        
        for line_idx, line in enumerate(f, start=2):
            total_lines += 1
            raw_line = line.rstrip("\r\n")
            if not raw_line.strip():
                blank_lines += 1
                continue
            
            parts = raw_line.split("\t")
            if len(parts) != expected_len:
                malformed_lines += 1
                if malformed_lines <= 5:
                    print(f"  [Malformed line {line_idx}]: parts count={len(parts)}, line='{raw_line[:100]}...'")
                continue
            
            # Store first 5 samples
            if len(sample_rows) < 5:
                sample_rows.append(parts)
            
            # Column analysis
            for col_name, val in zip(header, parts):
                if not val.strip():
                    null_counts[col_name] += 1
            
            if "country" in header:
                country_idx = header.index("country")
                country_counts[parts[country_idx].strip()] += 1
                
            if is_ground_truth and len(parts) >= 2:
                matched = parts[1].strip()
                if not matched:
                    match_count_dist[0] += 1
                else:
                    m_ids = [m.strip() for m in matched.split(",") if m.strip()]
                    match_count_dist[len(m_ids)] += 1
    
    elapsed = time.time() - start_time
    data_rows = total_lines - 1 - blank_lines - malformed_lines
    print(f"Total Lines: {total_lines:,}")
    print(f"Data Rows: {data_rows:,}")
    print(f"Blank Lines: {blank_lines:,}")
    print(f"Malformed Lines: {malformed_lines:,}")
    if null_counts:
        print(f"Empty/Null Fields by Column: {dict(null_counts)}")
    else:
        print("Empty/Null Fields: None (All columns populated)")
    if country_counts:
        print(f"Country Distribution: {dict(country_counts.most_common())}")
    if is_ground_truth:
        zero_matches = match_count_dist[0]
        total_gt = sum(match_count_dist.values())
        print(f"Ground Truth Match Distribution (Total S1 entities: {total_gt:,}):")
        print(f"  Singletons (0 matches): {zero_matches:,} ({zero_matches/total_gt*100:.2f}%)")
        for num_m in sorted(match_count_dist.keys())[:10]:
            if num_m > 0:
                print(f"  {num_m} matches: {match_count_dist[num_m]:,} ({match_count_dist[num_m]/total_gt*100:.2f}%)")
        if max(match_count_dist.keys()) > 10:
            print(f"  Max matches for single entity: {max(match_count_dist.keys())}")
    
    print("\nFirst 3 Sample Rows:")
    for s in sample_rows[:3]:
        print("  ", s)
    print(f"Inspection Time: {elapsed:.2f}s")
    print(f"=======================================================")

def main():
    print("=== STARTING DATASET INSPECTION ===")
    
    # Train files
    train_files = [
        ("train_source1.tsv", 4, False),
        ("train_source2.tsv", 4, False),
        ("train_source3.tsv", 4, False),
        ("train_ground_truth.tsv", 2, True),
    ]
    
    for fname, cols, is_gt in train_files:
        path = os.path.join(TRAIN_DIR, fname)
        if os.path.exists(path):
            inspect_tsv_file(path, expected_cols_count=cols, is_ground_truth=is_gt)
        else:
            print(f"MISSING FILE: {path}")
            
    # Test files
    test_files = [
        ("test_source1.tsv", 4, False),
        ("test_source2.tsv", 4, False),
        ("test_source3.tsv", 4, False),
    ]
    
    for fname, cols, is_gt in test_files:
        path = os.path.join(TEST_DIR, fname)
        if os.path.exists(path):
            inspect_tsv_file(path, expected_cols_count=cols, is_ground_truth=is_gt)
        else:
            print(f"MISSING FILE: {path}")

if __name__ == "__main__":
    main()
