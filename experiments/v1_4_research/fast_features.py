from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq
from src.features import STAGE5_FEATURE_COLS, PrecomputedEntity
from scripts.push_to_98 import ULTRA_FEATURE_COLS

class OptimizedEntity:
    """High-performance precomputed entity container caching invariants and structured address primitives."""
    __slots__ = (
        'norm_name', 'core_name', 'norm_addr', 'country', 'src_id',
        'core_toks', 'addr_toks', 'addr_nums', 'name_ngrams3', 'addr_ngrams3',
        'name_len', 'addr_len', 'is_s2', 'is_s3',
        'postal_code', 'house_num', 'digits_set',
        'name_log_freq', 'addr_log_freq'
    )
    def __init__(
        self,
        norm_name: str,
        core_name: str,
        norm_addr: str,
        country: str,
        src_id: str = "",
        ac: Optional[Dict[str, Any]] = None,
        freq_tracker: Optional[Any] = None
    ):
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
        
        # Structured address primitives
        if ac is not None:
            self.postal_code = ac.get("postal_code", "")
            self.house_num = ac.get("house_num", "")
            self.digits_set = ac.get("digits", set())
        else:
            self.postal_code = ""
            self.house_num = ""
            self.digits_set = set()
            
        # Frequencies
        if freq_tracker is not None:
            self.name_log_freq = freq_tracker.get_name_freq_feature(self.norm_name)
            self.addr_log_freq = freq_tracker.get_addr_freq_feature(self.norm_addr)
        else:
            self.name_log_freq = 0.0
            self.addr_log_freq = 0.0

def extract_ultra_features_fast(
    s1: OptimizedEntity,
    tgt: OptimizedEntity,
    freq_tracker: Any,
    evid_map: Dict[str, float],
) -> List[float]:
    """Ultra-fast zero-allocation feature extractor preserving 100.0% exact numerical fidelity

    with V1.3 Ultra (50 features).
    """
    # -------------------------------------------------------------------------
    # 1. NAME SIMILARITIES
    # -------------------------------------------------------------------------
    exact_norm_name = 1.0 if (s1.norm_name and s1.norm_name == tgt.norm_name) else 0.0
    exact_core_name = 1.0 if (s1.core_name and s1.core_name == tgt.core_name) else 0.0
    
    if exact_norm_name == 1.0:
        name_ratio = 1.0
        name_wratio = 1.0
        name_token_sort = 1.0
        name_token_set = 1.0
        name_partial = 1.0
        name_jaccard = 1.0
        name_char_3gram = 1.0
        name_len_diff = 0.0
        name_jw = 1.0
        name_lcs = float(s1.name_len)
        name_containment = 1.0
        name_min_overlap = 1.0
        name_w_jaccard = 1.0
    else:
        name_ratio = fuzz.ratio(s1.norm_name, tgt.norm_name) / 100.0
        name_wratio = fuzz.WRatio(s1.norm_name, tgt.norm_name) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1.norm_name, tgt.norm_name) / 100.0
        name_token_set = fuzz.token_set_ratio(s1.norm_name, tgt.norm_name) / 100.0
        name_partial = fuzz.partial_ratio(s1.norm_name, tgt.norm_name) / 100.0
        
        if s1.core_toks and tgt.core_toks:
            inter = s1.core_toks.intersection(tgt.core_toks)
            un = s1.core_toks.union(tgt.core_toks)
            name_jaccard = float(len(inter) / len(un)) if un else 0.0
            
            s1_in_tgt = float(s1.core_toks.issubset(tgt.core_toks))
            tgt_in_s1 = float(tgt.core_toks.issubset(s1.core_toks))
            name_containment = max(s1_in_tgt, tgt_in_s1)
            
            min_tok_len = min(len(s1.core_toks), len(tgt.core_toks))
            name_min_overlap = len(inter) / min_tok_len if min_tok_len > 0 else 0.0
            
            w_inter = sum(freq_tracker.get_token_idf(t) for t in inter)
            w_un = sum(freq_tracker.get_token_idf(t) for t in un)
            name_w_jaccard = float(w_inter / w_un) if w_un > 0 else 0.0
        else:
            name_jaccard = 0.0
            name_containment = 0.0
            name_min_overlap = 0.0
            name_w_jaccard = 0.0
            
        if not s1.norm_name or not tgt.norm_name:
            name_char_3gram = 0.0
        elif s1.name_len < 3 or tgt.name_len < 3:
            name_char_3gram = 1.0 if s1.norm_name == tgt.norm_name else 0.0
        else:
            inter = len(s1.name_ngrams3.intersection(tgt.name_ngrams3))
            un = len(s1.name_ngrams3.union(tgt.name_ngrams3))
            name_char_3gram = float(inter / un) if un > 0 else 0.0
            
        name_len_diff = float(abs(s1.name_len - tgt.name_len))
        name_jw = JaroWinkler.similarity(s1.norm_name, tgt.norm_name)
        name_lcs = float(LCSseq.similarity(s1.norm_name, tgt.norm_name))

    # Core name similarities
    if exact_core_name == 1.0:
        core_ratio = 1.0
        core_token_sort = 1.0
        core_token_set = 1.0
    else:
        core_ratio = fuzz.ratio(s1.core_name, tgt.core_name) / 100.0
        core_token_sort = fuzz.token_sort_ratio(s1.core_name, tgt.core_name) / 100.0
        core_token_set = fuzz.token_set_ratio(s1.core_name, tgt.core_name) / 100.0

    # -------------------------------------------------------------------------
    # 2. ADDRESS SIMILARITIES
    # -------------------------------------------------------------------------
    addr_missing = 1.0 if (not s1.norm_addr or not tgt.norm_addr) else 0.0
    if addr_missing:
        addr_exact = 0.0
        addr_ratio = 0.0
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_jaccard = 0.0
        addr_char_3gram = 0.0
        addr_num_match = 0.0
        addr_len_diff = 0.0
        addr_jw = 0.0
        addr_lcs = 0.0
        addr_containment = 0.0
        addr_w_jaccard = 0.0
    elif s1.norm_addr == tgt.norm_addr:
        addr_exact = 1.0
        addr_ratio = 1.0
        addr_token_sort = 1.0
        addr_token_set = 1.0
        addr_jaccard = 1.0
        addr_char_3gram = 1.0
        addr_num_match = 1.0 if s1.addr_nums else 0.0
        addr_len_diff = 0.0
        addr_jw = 1.0
        addr_lcs = float(s1.addr_len)
        addr_containment = 1.0
        addr_w_jaccard = 1.0
    else:
        addr_exact = 0.0
        addr_ratio = fuzz.ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1.norm_addr, tgt.norm_addr) / 100.0
        
        inter_a = s1.addr_toks.intersection(tgt.addr_toks)
        un_a = s1.addr_toks.union(tgt.addr_toks)
        addr_jaccard = float(len(inter_a) / len(un_a)) if un_a else 0.0
        
        if s1.addr_len < 3 or tgt.addr_len < 3:
            addr_char_3gram = 1.0 if s1.norm_addr == tgt.norm_addr else 0.0
        else:
            inter_g = len(s1.addr_ngrams3.intersection(tgt.addr_ngrams3))
            un_g = len(s1.addr_ngrams3.union(tgt.addr_ngrams3))
            addr_char_3gram = float(inter_g / un_g) if un_g > 0 else 0.0
            
        addr_num_match = 1.0 if (s1.addr_nums and tgt.addr_nums and s1.addr_nums.intersection(tgt.addr_nums)) else 0.0
        addr_len_diff = float(abs(s1.addr_len - tgt.addr_len))
        addr_jw = JaroWinkler.similarity(s1.norm_addr, tgt.norm_addr)
        addr_lcs = float(LCSseq.similarity(s1.norm_addr, tgt.norm_addr))
        
        s1_a_in_tgt = float(s1.addr_toks.issubset(tgt.addr_toks))
        tgt_a_in_s1 = float(tgt.addr_toks.issubset(s1.addr_toks))
        addr_containment = max(s1_a_in_tgt, tgt_a_in_s1)
        
        w_inter_a = sum(freq_tracker.get_token_idf(t) for t in inter_a)
        w_un_a = sum(freq_tracker.get_token_idf(t) for t in un_a)
        addr_w_jaccard = float(w_inter_a / w_un_a) if w_un_a > 0 else 0.0

    # -------------------------------------------------------------------------
    # 3. CONTEXT & METADATA
    # -------------------------------------------------------------------------
    country_match = 1.0 if (s1.country and s1.country == tgt.country) else 0.0
    is_source2 = tgt.is_s2
    is_source3 = tgt.is_s3
    
    # -------------------------------------------------------------------------
    # 4. STRUCTURED ADDRESS FAST FEATURES
    # -------------------------------------------------------------------------
    s1_hn, tgt_hn = s1.house_num, tgt.house_num
    if s1_hn and tgt_hn:
        addr_hnum_match = 1.0 if s1_hn == tgt_hn else 0.0
        addr_hnum_conflict = 1.0 if s1_hn != tgt_hn else 0.0
    else:
        addr_hnum_match = 0.0
        addr_hnum_conflict = 0.0
        
    s1_p, tgt_p = s1.postal_code, tgt.postal_code
    if s1_p and tgt_p:
        addr_postal_match = 1.0 if s1_p == tgt_p else 0.0
        addr_postal_conflict = 1.0 if s1_p != tgt_p else 0.0
        postal_prefix_match = 1.0 if (len(s1_p) >= 3 and len(tgt_p) >= 3 and s1_p[:3] == tgt_p[:3]) else 0.0
    else:
        addr_postal_match = 0.0
        addr_postal_conflict = 0.0
        postal_prefix_match = 0.0
        
    if s1.digits_set and tgt.digits_set:
        inter_d = len(s1.digits_set.intersection(tgt.digits_set))
        addr_digits_overlap = float(inter_d / len(s1.digits_set.union(tgt.digits_set)))
        addr_digits_conflict = 1.0 if inter_d == 0 else 0.0
    else:
        addr_digits_overlap = 0.0
        addr_digits_conflict = 0.0

    # -------------------------------------------------------------------------
    # 5. FREQUENCIES, EVIDENCE & SYNTHETIC INTERACTION METRICS
    # -------------------------------------------------------------------------
    ret_views = float(len(evid_map)) if evid_map else 1.0
    max_tfidf = max([v for k, v in evid_map.items() if k != "blocking"], default=0.0)
    
    joint_sim = float(name_jw * 0.6 + addr_jw * 0.4)
    token_count_diff = float(abs(len(s1.core_toks) - len(tgt.core_toks)))
    exact_num_overlap = 1.0 if (s1.addr_nums and tgt.addr_nums and s1.addr_nums == tgt.addr_nums) else 0.0
    
    return [
        exact_norm_name, exact_core_name, name_ratio, name_wratio,
        name_token_sort, name_token_set, name_partial, core_ratio,
        core_token_sort, core_token_set, name_jaccard, name_char_3gram,
        name_len_diff, addr_missing, addr_exact, addr_ratio,
        addr_token_sort, addr_token_set, addr_jaccard, addr_char_3gram,
        addr_num_match, addr_len_diff, country_match, is_source2, is_source3,
        # Structured address
        addr_hnum_match, addr_hnum_conflict, addr_postal_match,
        addr_postal_conflict, addr_digits_overlap, addr_digits_conflict,
        # Frequencies
        s1.name_log_freq, tgt.name_log_freq, s1.addr_log_freq, tgt.addr_log_freq,
        # Evidence
        ret_views, max_tfidf,
        # String metrics
        name_jw, name_containment,
        addr_jw, addr_containment,
        name_w_jaccard, addr_w_jaccard,
        name_lcs, name_min_overlap,
        addr_lcs, postal_prefix_match,
        joint_sim, token_count_diff,
        exact_num_overlap,
    ]
