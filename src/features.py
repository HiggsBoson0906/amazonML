from typing import Dict, List, Tuple, Any, Set, Optional
from rapidfuzz import fuzz
import numpy as np

STAGE5_FEATURE_COLS = [
    "exact_norm_name", "exact_core_name", "name_ratio", "name_wratio",
    "name_token_sort", "name_token_set", "name_partial", "core_ratio",
    "core_token_sort", "core_token_set", "name_jaccard", "name_char_3gram",
    "name_len_diff", "addr_missing", "addr_exact", "addr_ratio",
    "addr_token_sort", "addr_token_set", "addr_jaccard", "addr_char_3gram",
    "addr_num_match", "addr_len_diff", "country_match", "is_source2", "is_source3",
]

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

class PrecomputedEntity:
    """Optimized container with precomputed tokens, sets, and lengths to accelerate pairwise feature extraction."""
    __slots__ = (
        'norm_name', 'core_name', 'norm_addr', 'country', 'src_id',
        'core_toks', 'addr_toks', 'addr_nums', 'name_ngrams3', 'addr_ngrams3',
        'name_len', 'addr_len', 'is_s2', 'is_s3'
    )
    def __init__(self, norm_name: str, core_name: str, norm_addr: str, country: str, src_id: str = ""):
        self.norm_name = norm_name or ""
        self.core_name = core_name or ""
        self.norm_addr = norm_addr or ""
        self.country = country or ""
        self.src_id = src_id
        
        self.core_toks = set(self.core_name.split()) if self.core_name else set()
        self.addr_toks = set(self.norm_addr.split()) if self.norm_addr else set()
        self.addr_nums = {t for t in self.addr_toks if t.isdigit()}
        
        # 3-grams
        nn = self.norm_name
        self.name_ngrams3 = {nn[i:i+3] for i in range(len(nn)-2)} if len(nn) >= 3 else set()
        na = self.norm_addr
        self.addr_ngrams3 = {na[i:i+3] for i in range(len(na)-2)} if len(na) >= 3 else set()
        
        self.name_len = len(self.norm_name)
        self.addr_len = len(self.norm_addr)
        self.is_s2 = 1.0 if src_id.startswith("S2-") else 0.0
        self.is_s3 = 1.0 if src_id.startswith("S3-") else 0.0

def extract_pairwise_features_fast(
    s1: PrecomputedEntity,
    tgt: PrecomputedEntity,
) -> List[float]:
    """Extract fine-grained similarity feature vector using precomputed structures."""
    # 1. Name similarities
    exact_norm_name = 1.0 if s1.norm_name and s1.norm_name == tgt.norm_name else 0.0
    exact_core_name = 1.0 if s1.core_name and s1.core_name == tgt.core_name else 0.0
    
    name_ratio = fuzz.ratio(s1.norm_name, tgt.norm_name) / 100.0
    name_wratio = fuzz.WRatio(s1.norm_name, tgt.norm_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1.norm_name, tgt.norm_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1.norm_name, tgt.norm_name) / 100.0
    name_partial = fuzz.partial_ratio(s1.norm_name, tgt.norm_name) / 100.0
    
    core_ratio = fuzz.ratio(s1.core_name, tgt.core_name) / 100.0
    core_token_sort = fuzz.token_sort_ratio(s1.core_name, tgt.core_name) / 100.0
    core_token_set = fuzz.token_set_ratio(s1.core_name, tgt.core_name) / 100.0
    
    # Name tokens Jaccard and character 3-gram
    if s1.core_toks and tgt.core_toks:
        inter = len(s1.core_toks.intersection(tgt.core_toks))
        un = len(s1.core_toks.union(tgt.core_toks))
        name_jaccard = float(inter / un) if un > 0 else 0.0
    else:
        name_jaccard = 0.0
        
    if not s1.norm_name or not tgt.norm_name:
        name_char_3gram = 0.0
    elif s1.name_len < 3 or tgt.name_len < 3:
        name_char_3gram = 1.0 if s1.norm_name == tgt.norm_name else 0.0
    else:
        inter = len(s1.name_ngrams3.intersection(tgt.name_ngrams3))
        un = len(s1.name_ngrams3.union(tgt.name_ngrams3))
        name_char_3gram = float(inter / un) if un > 0 else 0.0
        
    name_len_diff = float(abs(s1.name_len - tgt.name_len))
    
    # 2. Address similarities
    if not s1.norm_addr or not tgt.norm_addr:
        addr_missing = 1.0
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0
        addr_char_3gram = 0.0
        addr_num_match = 0.0
        addr_len_diff = 0.0
    else:
        addr_missing = 0.0
        addr_exact = 1.0 if s1.norm_addr == tgt.norm_addr else 0.0
        addr_ratio = fuzz.ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        
        inter_addr = len(s1.addr_toks.intersection(tgt.addr_toks))
        un_addr = len(s1.addr_toks.union(tgt.addr_toks))
        addr_jaccard = float(inter_addr / un_addr) if un_addr > 0 else 0.0
        
        if s1.addr_len < 3 or tgt.addr_len < 3:
            addr_char_3gram = 1.0 if s1.norm_addr == tgt.norm_addr else 0.0
        else:
            inter_ng = len(s1.addr_ngrams3.intersection(tgt.addr_ngrams3))
            un_ng = len(s1.addr_ngrams3.union(tgt.addr_ngrams3))
            addr_char_3gram = float(inter_ng / un_ng) if un_ng > 0 else 0.0
            
        if s1.addr_nums and tgt.addr_nums:
            addr_num_match = 1.0 if s1.addr_nums.intersection(tgt.addr_nums) else 0.0
        else:
            addr_num_match = 0.0
            
        addr_len_diff = float(abs(s1.addr_len - tgt.addr_len))
        
    country_match = 1.0 if s1.country == tgt.country else 0.0
    
    return [
        exact_norm_name, exact_core_name, name_ratio, name_wratio,
        name_token_sort, name_token_set, name_partial, core_ratio,
        core_token_sort, core_token_set, name_jaccard, name_char_3gram,
        name_len_diff, addr_missing, addr_exact, addr_ratio,
        addr_token_sort, addr_token_set, addr_jaccard, addr_char_3gram,
        addr_num_match, addr_len_diff, country_match, tgt.is_s2, tgt.is_s3
    ]

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
    """Legacy feature extraction preserving exact dictionary output."""
    s1 = PrecomputedEntity(s1_norm_name, s1_core_name, s1_norm_addr, s1_country)
    tgt = PrecomputedEntity(tgt_norm_name, tgt_core_name, tgt_norm_addr, tgt_country, tgt_id)
    vec = extract_pairwise_features_fast(s1, tgt)
    return dict(zip(STAGE5_FEATURE_COLS, vec))
