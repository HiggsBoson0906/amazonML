import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np

class CorpusFrequencyTracker:
    """Computes unsupervised corpus token and entity frequencies from training/target corpora."""
    def __init__(self):
        self.name_counts: Counter = Counter()
        self.addr_counts: Counter = Counter()
        self.token_doc_counts: Counter = Counter()
        self.total_docs = 0

    def fit(self, names: List[str], addrs: List[str]):
        self.total_docs = len(names)
        for n, a in zip(names, addrs):
            if n:
                self.name_counts[n] += 1
                unique_tokens = set(n.split())
                for t in unique_tokens:
                    self.token_doc_counts[t] += 1
            if a:
                self.addr_counts[a] += 1

    def get_token_idf(self, token: str) -> float:
        df = self.token_doc_counts.get(token, 0)
        if df == 0 or self.total_docs == 0:
            return 1.0
        return float(np.log((self.total_docs + 1.0) / (df + 1.0)))

    def get_name_freq_feature(self, norm_name: str) -> float:
        cnt = self.name_counts.get(norm_name, 0)
        return float(np.log1p(cnt))

    def get_addr_freq_feature(self, norm_addr: str) -> float:
        cnt = self.addr_counts.get(norm_addr, 0)
        return float(np.log1p(cnt))

def extract_structured_address_components(addr: str) -> Dict[str, Any]:
    """Extract structured components: house number, postal code, street tokens, numeric set."""
    if not addr:
        return {
            "house_num": "",
            "postal_code": "",
            "digits": set(),
            "tokens": set(),
        }
    tokens = addr.split()
    digits = {t for t in tokens if t.isdigit()}
    
    # House number heuristic: first numeric token
    house_num = ""
    for t in tokens:
        if t.isdigit():
            house_num = t
            break
            
    # Postal code heuristic: 5-digit (US/FR) or 6-digit (IN) token
    postal_code = ""
    for t in tokens:
        if t.isdigit() and len(t) in (5, 6):
            postal_code = t
            break
            
    return {
        "house_num": house_num,
        "postal_code": postal_code,
        "digits": digits,
        "tokens": set(tokens),
    }

def compute_address_component_features(
    s1_addr_comp: Dict[str, Any],
    tgt_addr_comp: Dict[str, Any],
) -> Dict[str, float]:
    """Compute agreement and conflict features for structured address components."""
    s1_h = s1_addr_comp["house_num"]
    tgt_h = tgt_addr_comp["house_num"]
    
    if s1_h and tgt_h:
        hnum_match = 1.0 if s1_h == tgt_h else 0.0
        hnum_conflict = 1.0 if s1_h != tgt_h else 0.0
    else:
        hnum_match = 0.0
        hnum_conflict = 0.0
        
    s1_p = s1_addr_comp["postal_code"]
    tgt_p = tgt_addr_comp["postal_code"]
    if s1_p and tgt_p:
        postal_match = 1.0 if s1_p == tgt_p else 0.0
        postal_conflict = 1.0 if s1_p != tgt_p else 0.0
    else:
        postal_match = 0.0
        postal_conflict = 0.0
        
    s1_dig = s1_addr_comp["digits"]
    tgt_dig = tgt_addr_comp["digits"]
    if s1_dig and tgt_dig:
        digits_overlap = float(len(s1_dig.intersection(tgt_dig)))
        digits_conflict = 1.0 if not s1_dig.intersection(tgt_dig) else 0.0
    else:
        digits_overlap = 0.0
        digits_conflict = 0.0
        
    return {
        "addr_hnum_match": hnum_match,
        "addr_hnum_conflict": hnum_conflict,
        "addr_postal_match": postal_match,
        "addr_postal_conflict": postal_conflict,
        "addr_digits_overlap": digits_overlap,
        "addr_digits_conflict": digits_conflict,
    }

def compute_competition_features(
    pair_scores: List[Tuple[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Compute relative competition features (margin to top-1, candidate rank, pool size).

    
    pair_scores: list of (target_id, similarity_or_prelim_prob)
    Returns:
        target_id -> dict of relative features
    """
    if not pair_scores:
        return {}
        
    # Sort descending by score
    sorted_pairs = sorted(pair_scores, key=lambda x: -x[1])
    pool_size = len(sorted_pairs)
    top1_score = sorted_pairs[0][1]
    top2_score = sorted_pairs[1][1] if pool_size > 1 else 0.0
    
    comp_feats = {}
    for rank, (tid, score) in enumerate(sorted_pairs):
        margin_to_top = float(score - top1_score) if rank > 0 else float(top1_score - top2_score)
        comp_feats[tid] = {
            "cand_pool_size": float(pool_size),
            "cand_rank": float(rank),
            "cand_score_margin": margin_to_top,
            "cand_top1_score": float(top1_score),
            "cand_is_top1": 1.0 if rank == 0 else 0.0,
        }
    return comp_feats
