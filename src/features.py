from typing import Dict, List, Tuple, Any
from rapidfuzz import fuzz
import numpy as np

def compute_token_jaccard(toks1: set, toks2: set) -> float:
    """Compute Jaccard similarity between two sets of tokens."""
    if not toks1 or not toks2:
        return 0.0
    intersection = len(toks1.intersection(toks2))
    union = len(toks1.union(toks2))
    return float(intersection / union) if union > 0 else 0.0

def compute_char_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if not str1 or not str2:
        return 0.0
    if len(str1) < n or len(str2) < n:
        return 1.0 if str1 == str2 else 0.0
    ngrams1 = {str1[i:i+n] for i in range(len(str1) - n + 1)}
    ngrams2 = {str2[i:i+n] for i in range(len(str2) - n + 1)}
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return float(intersection / union) if union > 0 else 0.0

def extract_pairwise_features(
    s1_norm_name: str,
    s1_core_name: str,
    s1_norm_addr: str,
    s1_country: str,
    tgt_norm_name: str,
    tgt_core_name: str,
    tgt_norm_addr: str,
    tgt_country: str,
    tgt_id: str,
) -> Dict[str, float]:
    """Extract fine-grained similarity features between a Source-1 entity and a Target entity."""
    # Name similarities
    exact_norm_name = 1.0 if s1_norm_name and s1_norm_name == tgt_norm_name else 0.0
    exact_core_name = 1.0 if s1_core_name and s1_core_name == tgt_core_name else 0.0
    
    name_ratio = fuzz.ratio(s1_norm_name, tgt_norm_name) / 100.0
    name_wratio = fuzz.WRatio(s1_norm_name, tgt_norm_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_norm_name, tgt_norm_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_norm_name, tgt_norm_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_norm_name, tgt_norm_name) / 100.0
    
    core_ratio = fuzz.ratio(s1_core_name, tgt_core_name) / 100.0
    core_token_sort = fuzz.token_sort_ratio(s1_core_name, tgt_core_name) / 100.0
    core_token_set = fuzz.token_set_ratio(s1_core_name, tgt_core_name) / 100.0
    
    # Name tokens Jaccard and character 3-gram
    s1_name_toks = set(s1_core_name.split())
    tgt_name_toks = set(tgt_core_name.split())
    name_jaccard = compute_token_jaccard(s1_name_toks, tgt_name_toks)
    name_char_3gram = compute_char_ngram_jaccard(s1_norm_name, tgt_norm_name, n=3)
    
    name_len_diff = abs(len(s1_norm_name) - len(tgt_norm_name))
    
    # Address similarities
    addr_missing = 1.0 if (not s1_norm_addr or not tgt_norm_addr) else 0.0
    if addr_missing:
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0
        addr_char_3gram = 0.0
        addr_num_match = 0.0
        addr_len_diff = 0
    else:
        addr_exact = 1.0 if s1_norm_addr == tgt_norm_addr else 0.0
        addr_ratio = fuzz.ratio(s1_norm_addr, tgt_norm_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_norm_addr, tgt_norm_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_norm_addr, tgt_norm_addr) / 100.0
        
        s1_addr_toks = set(s1_norm_addr.split())
        tgt_addr_toks = set(tgt_norm_addr.split())
        addr_jaccard = compute_token_jaccard(s1_addr_toks, tgt_addr_toks)
        addr_char_3gram = compute_char_ngram_jaccard(s1_norm_addr, tgt_norm_addr, n=3)
        
        s1_nums = {t for t in s1_addr_toks if t.isdigit()}
        tgt_nums = {t for t in tgt_addr_toks if t.isdigit()}
        if s1_nums and tgt_nums:
            addr_num_match = 1.0 if s1_nums.intersection(tgt_nums) else 0.0
        else:
            addr_num_match = 0.0
            
        addr_len_diff = abs(len(s1_norm_addr) - len(tgt_norm_addr))
            
    country_match = 1.0 if s1_country == tgt_country else 0.0
    is_source2 = 1.0 if tgt_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if tgt_id.startswith("S3-") else 0.0
    
    return {
        "exact_norm_name": exact_norm_name,
        "exact_core_name": exact_core_name,
        "name_ratio": name_ratio,
        "name_wratio": name_wratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_partial": name_partial,
        "core_ratio": core_ratio,
        "core_token_sort": core_token_sort,
        "core_token_set": core_token_set,
        "name_jaccard": name_jaccard,
        "name_char_3gram": name_char_3gram,
        "name_len_diff": float(name_len_diff),
        "addr_missing": addr_missing,
        "addr_exact": addr_exact,
        "addr_ratio": addr_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "addr_char_3gram": addr_char_3gram,
        "addr_num_match": addr_num_match,
        "addr_len_diff": float(addr_len_diff),
        "country_match": country_match,
        "is_source2": is_source2,
        "is_source3": is_source3,
    }
