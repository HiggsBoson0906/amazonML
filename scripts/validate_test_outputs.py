import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Set, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

def validate_test_outputs(
    matching_path: str,
    candidate_path: str,
    source1_path: str,
    expected_s1_count: int = None,
) -> bool:
    print("=" * 70)
    print("TEST OUTPUT STRUCTURAL & INTEGRITY VALIDATOR")
    print(f"  Matching File:  {matching_path}")
    print(f"  Candidate File: {candidate_path}")
    print(f"  Source 1 File:  {source1_path}")
    print("=" * 70)
    
    errors = []
    warnings = []
    
    # 1. Read Expected S1 IDs
    if not os.path.isfile(source1_path):
        print(f"ERROR: Source 1 file not found: {source1_path}")
        return False
        
    required_s1_ids = []
    with open(source1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split(DELIM)
            if parts and parts[0].strip():
                required_s1_ids.append(parts[0].strip())
                if expected_s1_count and len(required_s1_ids) >= expected_s1_count:
                    break
                    
    expected_s1_set = set(required_s1_ids)
    print(f"\n1. Target Universe: {len(expected_s1_set):,} expected S1 entities")
    
    # 2. Validate Matching Results TSV
    print("\n2. Validating matching_results.tsv...")
    if not os.path.isfile(matching_path):
        errors.append(f"Matching file not found: {matching_path}")
        return False
        
    matched_mapping: Dict[str, Set[str]] = {}
    seen_matching_s1 = set()
    matching_empties = 0
    total_matches = 0
    
    with open(matching_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append("Matching file is empty.")
            return False
        header = [c.strip().lower() for c in header_line.rstrip("\r\n").split(DELIM)]
        if header != MATCHING_HEADER:
            errors.append(f"Matching file invalid header: {header}. Expected {MATCHING_HEADER}")
            
        for line_idx, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(DELIM)
            if len(parts) != 2:
                errors.append(f"Matching file line {line_idx} does not have exactly 2 tab-separated fields: {line!r}")
                continue
            s1_id, match_str = parts[0].strip(), parts[1].strip()
            
            if s1_id in seen_matching_s1:
                errors.append(f"Duplicate S1 ID in matching_results: {s1_id}")
            seen_matching_s1.add(s1_id)
            
            if not match_str:
                matching_empties += 1
                matched_mapping[s1_id] = set()
            else:
                m_ids = [m.strip() for m in match_str.split(",") if m.strip()]
                if len(m_ids) != len(set(m_ids)):
                    errors.append(f"Duplicate match ID inside matching list for S1 {s1_id}: {match_str}")
                for mid in m_ids:
                    if not mid.startswith(("S2-", "S3-")):
                        errors.append(f"Invalid target ID format in matching list for S1 {s1_id}: {mid}")
                    if mid.startswith("S1-"):
                        errors.append(f"Self-match (S1 ID) in matching list for S1 {s1_id}: {mid}")
                matched_mapping[s1_id] = set(m_ids)
                total_matches += len(m_ids)
                
    print(f"  matching_results.tsv: {len(seen_matching_s1):,} S1 rows, {matching_empties:,} empty predictions ({matching_empties/len(seen_matching_s1)*100:.2f}%), {total_matches:,} total matches ({total_matches/len(seen_matching_s1):.3f} matches/S1)")
    
    # 3. Validate Candidate Pairs TSV
    print("\n3. Validating candidate_pairs.tsv...")
    if not os.path.isfile(candidate_path):
        errors.append(f"Candidate file not found: {candidate_path}")
        return False
        
    candidate_mapping: Dict[str, Set[str]] = {}
    seen_candidate_s1 = set()
    candidate_empties = 0
    total_candidates = 0
    cand_counts = []
    
    with open(candidate_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            errors.append("Candidate file is empty.")
            return False
        header = [c.strip().lower() for c in header_line.rstrip("\r\n").split(DELIM)]
        if header != CANDIDATE_HEADER:
            errors.append(f"Candidate file invalid header: {header}. Expected {CANDIDATE_HEADER}")
            
        for line_idx, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split(DELIM)
            if len(parts) != 2:
                errors.append(f"Candidate file line {line_idx} does not have exactly 2 tab-separated fields: {line!r}")
                continue
            s1_id, cand_str = parts[0].strip(), parts[1].strip()
            
            if s1_id in seen_candidate_s1:
                errors.append(f"Duplicate S1 ID in candidate_pairs: {s1_id}")
            seen_candidate_s1.add(s1_id)
            
            if not cand_str:
                candidate_empties += 1
                candidate_mapping[s1_id] = set()
                cand_counts.append(0)
            else:
                c_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
                if len(c_ids) != len(set(c_ids)):
                    errors.append(f"Duplicate candidate ID inside candidate list for S1 {s1_id}: {cand_str}")
                for cid in c_ids:
                    if not cid.startswith(("S2-", "S3-")):
                        errors.append(f"Invalid candidate ID format for S1 {s1_id}: {cid}")
                    if cid.startswith("S1-"):
                        errors.append(f"Self-candidate (S1 ID) in candidate list for S1 {s1_id}: {cid}")
                candidate_mapping[s1_id] = set(c_ids)
                total_candidates += len(c_ids)
                cand_counts.append(len(c_ids))
                
    import numpy as np
    print(f"  candidate_pairs.tsv:  {len(seen_candidate_s1):,} S1 rows, {candidate_empties:,} empty, {total_candidates:,} total candidates")
    print(f"    Avg: {np.mean(cand_counts):.2f}, Median: {np.median(cand_counts):.1f}, P95: {np.percentile(cand_counts, 95):.1f}, P99: {np.percentile(cand_counts, 99):.1f}, Max: {max(cand_counts) if cand_counts else 0}")
    
    # 4. Set Consistency & Coverage Checks
    print("\n4. Checking S1 Identity & Invariant Consistency...")
    
    # Coverage check
    missing_in_matching = expected_s1_set - seen_matching_s1
    extra_in_matching = seen_matching_s1 - expected_s1_set
    missing_in_cand = expected_s1_set - seen_candidate_s1
    extra_in_cand = seen_candidate_s1 - expected_s1_set
    
    if missing_in_matching:
        errors.append(f"{len(missing_in_matching)} expected S1 IDs missing from matching_results.tsv (e.g. {list(missing_in_matching)[:5]})")
    if extra_in_matching:
        errors.append(f"{len(extra_in_matching)} unexpected S1 IDs present in matching_results.tsv (e.g. {list(extra_in_matching)[:5]})")
    if missing_in_cand:
        errors.append(f"{len(missing_in_cand)} expected S1 IDs missing from candidate_pairs.tsv (e.g. {list(missing_in_cand)[:5]})")
    if extra_in_cand:
        errors.append(f"{len(extra_in_cand)} unexpected S1 IDs present in candidate_pairs.tsv (e.g. {list(extra_in_cand)[:5]})")
        
    # CRITICAL SUBSET INVARIANT: matched_entity_ids MUST BE SUBSET OF candidate_entity_ids
    subset_violations = []
    for s1_id in expected_s1_set:
        matched_set = matched_mapping.get(s1_id, set())
        candidate_set = candidate_mapping.get(s1_id, set())
        invalid_matches = matched_set - candidate_set
        if invalid_matches:
            subset_violations.append((s1_id, invalid_matches))
            
    if subset_violations:
        errors.append(f"FATAL INVARIANT VIOLATION: {len(subset_violations)} S1 entities have matched IDs that are NOT in candidate_pairs.tsv! (e.g. {subset_violations[:3]})")
    else:
        print("  [PASS] 100% SUBSET INVARIANT: matched_entity_ids is a strict subset of candidate_entity_ids for all S1 entities.")
        
    print("\n" + "=" * 70)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return False
    else:
        print("VALIDATION PASSED: All structural, schema, ID prefix, and subset invariants verified!")
        print("=" * 70)
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate test output TSV files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--source1", default="dataset/test/test_source1.tsv", help="Path to test_source1.tsv")
    parser.add_argument("--expected-count", type=int, default=None, help="Expected S1 count (e.g. 5000 for dry run)")
    args = parser.parse_args()
    
    success = validate_test_outputs(
        args.matching,
        args.candidate,
        args.source1,
        expected_s1_count=args.expected_count,
    )
    sys.exit(0 if success else 1)
