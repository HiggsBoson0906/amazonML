import os
import sys
import time
from pathlib import Path
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from src.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
from src.normalize import normalize_business_name, extract_core_business_name, normalize_business_address
from src.blocking import InvertedIndexBlocker
from src.features import PrecomputedEntity
from src.retrieval import MultiViewCandidateRetriever
from src.evaluate import evaluate_candidate_recall
from experiments.v1_4_research.fast_retrieval import retrieve_candidates_batch_country

def test_speed_and_equivalence():
    print("Testing Vectorized Candidate Retrieval Speed & Equivalence...")
    
    # Load 1,000 S1
    s1_rows = []
    val_s1_list = []
    with open(TRAIN_SOURCE1, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 4:
                continue
            eid, b_name, b_addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            s1_rows.append((eid, b_name, b_addr, country))
            val_s1_list.append(eid)
            if len(val_s1_list) >= 1000:
                break
                
    val_s1_set = set(val_s1_list)
    val_gt_map = {sid: set() for sid in val_s1_list}
    needed_targets = set()
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            sid = parts[0].strip()
            if sid in val_s1_set:
                m_ids = {m.strip() for m in parts[1].split(",") if m.strip()} if len(parts) > 1 and parts[1].strip() else set()
                val_gt_map[sid] = m_ids
                needed_targets.update(m_ids)
                
    # Load targets (50k distractors)
    blocker = InvertedIndexBlocker(max_key_frequency=500, max_candidates_per_s1=200)
    target_lookup_fast = {}
    target_lookup_raw = {}
    
    def load_targets(filepath, max_d=25000):
        d_cnt = 0
        with open(filepath, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 4:
                    continue
                tid, b_name, b_addr, country = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                if not tid:
                    continue
                is_needed = tid in needed_targets
                if is_needed or d_cnt < max_d:
                    if not is_needed:
                        d_cnt += 1
                    n_name = normalize_business_name(b_name)
                    c_name = extract_core_business_name(n_name)
                    n_addr = normalize_business_address(b_addr) if b_addr else ""
                    target_lookup_fast[tid] = PrecomputedEntity(n_name, c_name, n_addr, country, tid)
                    target_lookup_raw[tid] = (n_name, c_name, n_addr, country)
                    if country:
                        keys = blocker._extract_all_keys(n_name, c_name, n_addr, b_name)
                        for k in keys:
                            blocker.index[country][k].append(tid)

    load_targets(TRAIN_SOURCE2)
    load_targets(TRAIN_SOURCE3)
    blocker.prune_high_frequency_keys()
    
    retriever = MultiViewCandidateRetriever(
        blocker=blocker,
        enable_tfidf_name=True,
        enable_tfidf_addr=True,
        enable_tfidf_char=True,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    retriever.fit_target_corpora(target_lookup_raw)
    
    # S1 prepared
    s1_country_dict = defaultdict(list)
    s1_raw_dict = {}
    for eid, name, addr, country in s1_rows:
        n_n = normalize_business_name(name)
        c_n = extract_core_business_name(n_n)
        n_a = normalize_business_address(addr) if addr else ""
        s1_country_dict[country.strip() if country else "UNKNOWN"].append((eid, n_n, c_n, n_a, country))
        s1_raw_dict[eid] = (n_n, c_n, n_a, country)
        
    df_chunk = pl.DataFrame({
        "entity_id": val_s1_list,
        "business_name": [r[1] for r in s1_rows],
        "norm_name": [s1_raw_dict[eid][0] for eid in val_s1_list],
        "core_name": [s1_raw_dict[eid][1] for eid in val_s1_list],
        "norm_address": [s1_raw_dict[eid][2] for eid in val_s1_list],
        "country": [r[3] for r in s1_rows],
    })
    val_stage5 = blocker.generate_candidates_for_s1(df_chunk)
    
    # Measure Vectorized Batch Retrieval
    t0 = time.time()
    batch_cands, batch_evid = retrieve_candidates_batch_country(
        s1_records_by_country=s1_country_dict,
        blocking_cands_map=val_stage5,
        retriever=retriever,
        top_k_per_view=45,
        max_total_candidates=200,
    )
    t_batch = time.time() - t0
    print(f"Batch Vectorized Retrieval Time: {t_batch:.4f}s for 1,000 S1 ({t_batch*1000:.2f} ms total, {t_batch/1000*1000:.3f} ms/S1)")
    
    rec_res = evaluate_candidate_recall(batch_cands, val_gt_map)
    print(f"Batch Retrieval Candidate Recall: {rec_res['candidate_recall']*100:.3f}% ({rec_res['found_gt_pairs']}/{rec_res['total_gt_pairs']})")

if __name__ == "__main__":
    test_speed_and_equivalence()
