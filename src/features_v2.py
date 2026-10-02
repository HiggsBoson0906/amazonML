"""
V2 Feature Extraction — Optimized for Production Speed + Memory Safety
=======================================================================

Key design: Target-side invariants stored in a COLUMNAR layout using arrays
and a string-interning approach, NOT as millions of Python dicts/objects.

Architecture:
1. TargetStore: Columnar storage for all target-side data
2. Stage-A: Ultra-cheap filter using precomputed values
3. Stage-B: Full 50-feature extraction for survivors only
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq
import numpy as np
import time

from src.features import ULTRA_FEATURE_COLS  # Keep canonical feature list


class TargetStore:
    """Columnar storage for target primitives.
    
    Stores ~10M targets in compact arrays instead of Python objects.
    String data is interned into a string pool; token sets are stored
    as variable-length lists keyed by integer index.
    
    Memory estimate for 10M targets:
    - norm_name strings: ~10M × ~30 bytes avg = ~300MB (Python intern)
    - core_name strings: similar
    - norm_addr strings: similar 
    - Arrays (s2/s3/freq/lengths): 10M × 4 bytes × ~8 = ~320MB
    - Token sets kept as-is only for Stage-B accessed targets (lazy)
    Total: ~2-3 GB vs ~50+ GB for dict-of-dicts approach
    """
    
    def __init__(self):
        self.tids: List[str] = []            # Ordered target IDs
        self.tid_to_idx: Dict[str, int] = {} # tid -> integer index
        
        # String arrays (Python lists — more memory-efficient than dict-per-target)
        self.norm_names: List[str] = []
        self.core_names: List[str] = []
        self.norm_addrs: List[str] = []
        self.countries: List[str] = []
        
        # Numeric arrays (NumPy)
        self.name_lens: np.ndarray = None    # int32
        self.addr_lens: np.ndarray = None    # int32
        self.is_s2: np.ndarray = None        # float32
        self.is_s3: np.ndarray = None        # float32
        self.name_log_freq: np.ndarray = None  # float32
        self.addr_log_freq: np.ndarray = None  # float32
        
        # Structured address (Python lists of strings — compact)
        self.postal_codes: List[str] = []
        self.house_nums: List[str] = []
        
        # Size
        self.n = 0
    
    def build(self, raw_targets: Dict[str, Tuple[str, str, str, str]], freq_tracker: Any):
        """Build columnar store from raw target dict. One-time operation."""
        t0 = time.time()
        n = len(raw_targets)
        self.n = n
        
        self.tids = list(raw_targets.keys())
        self.tid_to_idx = {tid: i for i, tid in enumerate(self.tids)}
        
        # Pre-allocate lists
        self.norm_names = [""] * n
        self.core_names = [""] * n
        self.norm_addrs = [""] * n
        self.countries = [""] * n
        self.postal_codes = [""] * n
        self.house_nums = [""] * n
        
        # NumPy arrays
        name_lens = np.zeros(n, dtype=np.int32)
        addr_lens = np.zeros(n, dtype=np.int32)
        is_s2 = np.zeros(n, dtype=np.float32)
        is_s3 = np.zeros(n, dtype=np.float32)
        name_lf = np.zeros(n, dtype=np.float32)
        addr_lf = np.zeros(n, dtype=np.float32)
        
        for i, tid in enumerate(self.tids):
            nn, cn, na, ct = raw_targets[tid]
            nn = nn or ""
            cn = cn or ""
            na = na or ""
            ct = ct or ""
            
            self.norm_names[i] = nn
            self.core_names[i] = cn
            self.norm_addrs[i] = na
            self.countries[i] = ct
            
            name_lens[i] = len(nn)
            addr_lens[i] = len(na)
            is_s2[i] = 1.0 if tid.startswith("S2-") else 0.0
            is_s3[i] = 1.0 if tid.startswith("S3-") else 0.0
            name_lf[i] = freq_tracker.get_name_freq_feature(nn)
            addr_lf[i] = freq_tracker.get_addr_freq_feature(na)
            
            # Structured address extraction (inline)
            house_num = ""
            postal_code = ""
            if na:
                for t in na.split():
                    if t.isdigit():
                        if not house_num:
                            house_num = t
                        if not postal_code and len(t) in (5, 6):
                            postal_code = t
            self.postal_codes[i] = postal_code
            self.house_nums[i] = house_num
            
            if (i + 1) % 2000000 == 0:
                print(f"    TargetStore: {i+1:,}/{n:,} targets indexed...", flush=True)
        
        self.name_lens = name_lens
        self.addr_lens = addr_lens
        self.is_s2 = is_s2
        self.is_s3 = is_s3
        self.name_log_freq = name_lf
        self.addr_log_freq = addr_lf
        
        print(f"  TargetStore built: {n:,} targets in {time.time()-t0:.1f}s", flush=True)
    
    def get_primitives(self, tid: str) -> Optional[dict]:
        """Lazily build primitives dict for a single target. Used in Stage-B only."""
        idx = self.tid_to_idx.get(tid)
        if idx is None:
            return None
        
        nn = self.norm_names[idx]
        cn = self.core_names[idx]
        na = self.norm_addrs[idx]
        
        addr_tokens = na.split() if na else []
        
        return {
            'nn': nn, 'cn': cn, 'na': na, 'ct': self.countries[idx],
            'ct_f': frozenset(cn.split()) if cn else frozenset(),
            'at_f': frozenset(addr_tokens),
            'an_f': frozenset(t for t in addr_tokens if t.isdigit()),
            'ng3': frozenset(nn[i:i+3] for i in range(len(nn)-2)) if len(nn) >= 3 else frozenset(),
            'ag3': frozenset(na[i:i+3] for i in range(len(na)-2)) if len(na) >= 3 else frozenset(),
            'nl': int(self.name_lens[idx]),
            'al': int(self.addr_lens[idx]),
            's2': float(self.is_s2[idx]),
            's3': float(self.is_s3[idx]),
            'pc': self.postal_codes[idx],
            'hn': self.house_nums[idx],
            'ds': frozenset(t for t in addr_tokens if t.isdigit()),
            'nf': float(self.name_log_freq[idx]),
            'af': float(self.addr_log_freq[idx]),
        }
    
    def stage_a_score(self, s1_nn: str, s1_cn: str, s1_na: str, s1_ct: str,
                      s1_core_toks: frozenset, s1_ng3: frozenset,
                      s1_pc: str, s1_hn: str,
                      tid: str, evid: dict) -> float:
        """Ultra-cheap Stage-A compatibility score using columnar lookups.
        
        This avoids creating ANY Python objects for the target — just array lookups.
        """
        idx = self.tid_to_idx.get(tid)
        if idx is None:
            return -1.0
        
        tgt_nn = self.norm_names[idx]
        tgt_cn = self.core_names[idx]
        
        # Exact name match — guaranteed pass
        if s1_nn and s1_nn == tgt_nn:
            return 1.0
        
        score = 0.0
        
        # Exact core name
        if s1_cn and s1_cn == tgt_cn:
            score += 0.45
        
        # Token Jaccard on core name (need to tokenize target)
        if s1_core_toks and tgt_cn:
            tgt_core_toks = frozenset(tgt_cn.split())
            if tgt_core_toks:
                inter = len(s1_core_toks & tgt_core_toks)
                union = len(s1_core_toks | tgt_core_toks)
                if union > 0:
                    jaccard = inter / union
                    score += 0.25 * jaccard
                    min_len = min(len(s1_core_toks), len(tgt_core_toks))
                    if min_len > 0:
                        score += 0.10 * (inter / min_len)
        
        # Name char 3-gram Jaccard (lightweight — frozenset ops)
        if s1_ng3:
            tgt_nl = int(self.name_lens[idx])
            if tgt_nl >= 3:
                tgt_ng3 = frozenset(tgt_nn[i:i+3] for i in range(tgt_nl - 2))
                ng_inter = len(s1_ng3 & tgt_ng3)
                ng_union = len(s1_ng3 | tgt_ng3)
                if ng_union > 0:
                    score += 0.15 * (ng_inter / ng_union)
        
        # Country match/conflict
        tgt_ct = self.countries[idx]
        if s1_ct and tgt_ct:
            if s1_ct == tgt_ct:
                score += 0.02
            else:
                score -= 0.05
        
        # Postal code
        if s1_pc:
            tgt_pc = self.postal_codes[idx]
            if tgt_pc:
                if s1_pc == tgt_pc:
                    score += 0.03
                else:
                    score -= 0.02
        
        # TF-IDF evidence boost
        if evid:
            max_tfidf = max((v for k, v in evid.items() if k != "blocking"), default=0.0)
            score += 0.05 * max_tfidf
            if "blocking" in evid:
                score += 0.02
        
        return score


# Conservative threshold: tuned to preserve >99.5% candidate recall
STAGE_A_THRESHOLD = 0.08


def filter_candidates_stage_a(
    s1_nn: str, s1_cn: str, s1_na: str, s1_ct: str,
    s1_core_toks: frozenset, s1_ng3: frozenset,
    s1_pc: str, s1_hn: str,
    candidate_tids: Set[str],
    target_store: 'TargetStore',
    tfidf_scores: dict,
    max_survivors: int = 50,
) -> List[Tuple[str, float]]:
    """Apply Stage-A cheap filter to reduce candidates.
    
    Returns (tid, score) tuples for survivors, sorted descending, capped.
    """
    scored = []
    for tid in candidate_tids:
        evid = tfidf_scores.get(tid, {})
        sa = target_store.stage_a_score(
            s1_nn, s1_cn, s1_na, s1_ct,
            s1_core_toks, s1_ng3, s1_pc, s1_hn,
            tid, evid,
        )
        if sa >= STAGE_A_THRESHOLD:
            scored.append((tid, sa))
    
    scored.sort(key=lambda x: -x[1])
    return scored[:max_survivors]


def make_s1_primitives(
    norm_name: str, core_name: str, norm_addr: str, country: str,
    freq_tracker: Any,
) -> dict:
    """Create S1-side primitives dict."""
    nn = norm_name or ""
    cn = core_name or ""
    na = norm_addr or ""
    ct = country or ""
    
    core_toks = frozenset(cn.split()) if cn else frozenset()
    addr_tokens = na.split() if na else []
    addr_toks = frozenset(addr_tokens)
    addr_nums = frozenset(t for t in addr_tokens if t.isdigit())
    
    name_ngrams3 = frozenset(nn[i:i+3] for i in range(len(nn)-2)) if len(nn) >= 3 else frozenset()
    addr_ngrams3 = frozenset(na[i:i+3] for i in range(len(na)-2)) if len(na) >= 3 else frozenset()
    
    house_num = ""
    postal_code = ""
    for t in addr_tokens:
        if t.isdigit():
            if not house_num:
                house_num = t
            if not postal_code and len(t) in (5, 6):
                postal_code = t
    
    return {
        'nn': nn, 'cn': cn, 'na': na, 'ct': ct,
        'ct_f': core_toks, 'at_f': addr_toks, 'an_f': addr_nums,
        'ng3': name_ngrams3, 'ag3': addr_ngrams3,
        'nl': len(nn), 'al': len(na),
        's2': 0.0, 's3': 0.0,
        'pc': postal_code, 'hn': house_num, 'ds': addr_nums,
        'nf': freq_tracker.get_name_freq_feature(nn),
        'af': freq_tracker.get_addr_freq_feature(na),
    }


# ---------------------------------------------------------------------------
# STAGE B — Full 50-feature extraction (only for Stage-A survivors)
# ---------------------------------------------------------------------------

def extract_ultra_features_v2(
    s1: dict, tgt: dict,
    freq_tracker: Any,
    evid_map: Dict[str, float],
) -> List[float]:
    """V2 ultra feature extractor using dict-based primitives.
    
    Produces the EXACT same 50-feature vector as V1.3 extract_ultra_features_fast.
    
    Preserves short-circuit optimizations for exact matches.
    """
    s1_nn, tgt_nn = s1['nn'], tgt['nn']
    s1_cn, tgt_cn = s1['cn'], tgt['cn']
    s1_na, tgt_na = s1['na'], tgt['na']
    
    # -------------------------------------------------------------------------
    # 1. NAME SIMILARITIES
    # -------------------------------------------------------------------------
    exact_norm_name = 1.0 if (s1_nn and s1_nn == tgt_nn) else 0.0
    exact_core_name = 1.0 if (s1_cn and s1_cn == tgt_cn) else 0.0
    
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
        name_lcs = float(s1['nl'])
        name_containment = 1.0
        name_min_overlap = 1.0
        name_w_jaccard = 1.0
    else:
        name_ratio = fuzz.ratio(s1_nn, tgt_nn) / 100.0
        name_wratio = fuzz.WRatio(s1_nn, tgt_nn) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_nn, tgt_nn) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_nn, tgt_nn) / 100.0
        name_partial = fuzz.partial_ratio(s1_nn, tgt_nn) / 100.0
        
        s1_ct = s1['ct_f']
        tgt_ct = tgt['ct_f']
        if s1_ct and tgt_ct:
            inter = s1_ct & tgt_ct
            un = s1_ct | tgt_ct
            name_jaccard = float(len(inter) / len(un)) if un else 0.0
            
            s1_in_tgt = float(s1_ct.issubset(tgt_ct))
            tgt_in_s1 = float(tgt_ct.issubset(s1_ct))
            name_containment = max(s1_in_tgt, tgt_in_s1)
            
            min_tok_len = min(len(s1_ct), len(tgt_ct))
            name_min_overlap = len(inter) / min_tok_len if min_tok_len > 0 else 0.0
            
            w_inter = sum(freq_tracker.get_token_idf(t) for t in inter)
            w_un = sum(freq_tracker.get_token_idf(t) for t in un)
            name_w_jaccard = float(w_inter / w_un) if w_un > 0 else 0.0
        else:
            name_jaccard = 0.0
            name_containment = 0.0
            name_min_overlap = 0.0
            name_w_jaccard = 0.0
        
        if not s1_nn or not tgt_nn:
            name_char_3gram = 0.0
        elif s1['nl'] < 3 or tgt['nl'] < 3:
            name_char_3gram = 1.0 if s1_nn == tgt_nn else 0.0
        else:
            ng_inter = len(s1['ng3'] & tgt['ng3'])
            ng_union = len(s1['ng3'] | tgt['ng3'])
            name_char_3gram = float(ng_inter / ng_union) if ng_union > 0 else 0.0
        
        name_len_diff = float(abs(s1['nl'] - tgt['nl']))
        name_jw = JaroWinkler.similarity(s1_nn, tgt_nn)
        name_lcs = float(LCSseq.similarity(s1_nn, tgt_nn))
    
    # Core name similarities
    if exact_core_name == 1.0:
        core_ratio = 1.0
        core_token_sort = 1.0
        core_token_set = 1.0
    else:
        core_ratio = fuzz.ratio(s1_cn, tgt_cn) / 100.0
        core_token_sort = fuzz.token_sort_ratio(s1_cn, tgt_cn) / 100.0
        core_token_set = fuzz.token_set_ratio(s1_cn, tgt_cn) / 100.0
    
    # -------------------------------------------------------------------------
    # 2. ADDRESS SIMILARITIES
    # -------------------------------------------------------------------------
    addr_missing = 1.0 if (not s1_na or not tgt_na) else 0.0
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
    elif s1_na == tgt_na:
        addr_exact = 1.0
        addr_ratio = 1.0
        addr_token_sort = 1.0
        addr_token_set = 1.0
        addr_jaccard = 1.0
        addr_char_3gram = 1.0
        addr_num_match = 1.0 if s1['an_f'] else 0.0
        addr_len_diff = 0.0
        addr_jw = 1.0
        addr_lcs = float(s1['al'])
        addr_containment = 1.0
        addr_w_jaccard = 1.0
    else:
        addr_exact = 0.0
        addr_ratio = fuzz.ratio(s1_na, tgt_na) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_na, tgt_na) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_na, tgt_na) / 100.0
        
        s1_at = s1['at_f']
        tgt_at = tgt['at_f']
        inter_a = s1_at & tgt_at
        un_a = s1_at | tgt_at
        addr_jaccard = float(len(inter_a) / len(un_a)) if un_a else 0.0
        
        if s1['al'] < 3 or tgt['al'] < 3:
            addr_char_3gram = 1.0 if s1_na == tgt_na else 0.0
        else:
            ag_inter = len(s1['ag3'] & tgt['ag3'])
            ag_union = len(s1['ag3'] | tgt['ag3'])
            addr_char_3gram = float(ag_inter / ag_union) if ag_union > 0 else 0.0
        
        addr_num_match = 1.0 if (s1['an_f'] and tgt['an_f'] and (s1['an_f'] & tgt['an_f'])) else 0.0
        addr_len_diff = float(abs(s1['al'] - tgt['al']))
        addr_jw = JaroWinkler.similarity(s1_na, tgt_na)
        addr_lcs = float(LCSseq.similarity(s1_na, tgt_na))
        
        s1_a_in_tgt = float(s1_at.issubset(tgt_at))
        tgt_a_in_s1 = float(tgt_at.issubset(s1_at))
        addr_containment = max(s1_a_in_tgt, tgt_a_in_s1)
        
        w_inter_a = sum(freq_tracker.get_token_idf(t) for t in inter_a)
        w_un_a = sum(freq_tracker.get_token_idf(t) for t in un_a)
        addr_w_jaccard = float(w_inter_a / w_un_a) if w_un_a > 0 else 0.0
    
    # -------------------------------------------------------------------------
    # 3. CONTEXT & METADATA
    # -------------------------------------------------------------------------
    country_match = 1.0 if (s1['ct'] and s1['ct'] == tgt['ct']) else 0.0
    
    # -------------------------------------------------------------------------
    # 4. STRUCTURED ADDRESS
    # -------------------------------------------------------------------------
    s1_hn, tgt_hn = s1['hn'], tgt['hn']
    if s1_hn and tgt_hn:
        addr_hnum_match = 1.0 if s1_hn == tgt_hn else 0.0
        addr_hnum_conflict = 1.0 if s1_hn != tgt_hn else 0.0
    else:
        addr_hnum_match = 0.0
        addr_hnum_conflict = 0.0
    
    s1_p, tgt_p = s1['pc'], tgt['pc']
    if s1_p and tgt_p:
        addr_postal_match = 1.0 if s1_p == tgt_p else 0.0
        addr_postal_conflict = 1.0 if s1_p != tgt_p else 0.0
        postal_prefix_match = 1.0 if (len(s1_p) >= 3 and len(tgt_p) >= 3 and s1_p[:3] == tgt_p[:3]) else 0.0
    else:
        addr_postal_match = 0.0
        addr_postal_conflict = 0.0
        postal_prefix_match = 0.0
    
    s1_ds = s1['ds']
    tgt_ds = tgt['ds']
    if s1_ds and tgt_ds:
        inter_d = len(s1_ds & tgt_ds)
        addr_digits_overlap = float(inter_d)
        addr_digits_conflict = 1.0 if inter_d == 0 else 0.0
    else:
        addr_digits_overlap = 0.0
        addr_digits_conflict = 0.0
    
    # -------------------------------------------------------------------------
    # 5. FREQUENCIES, EVIDENCE & SYNTHETIC
    # -------------------------------------------------------------------------
    ret_views = float(len(evid_map)) if evid_map else 1.0
    max_tfidf = max((v for k, v in evid_map.items() if k != "blocking"), default=0.0)
    
    joint_sim = float(name_jw * 0.6 + addr_jw * 0.4)
    token_count_diff = float(abs(len(s1['ct_f']) - len(tgt['ct_f'])))
    exact_num_overlap = 1.0 if (s1['an_f'] and tgt['an_f'] and s1['an_f'] == tgt['an_f']) else 0.0
    
    return [
        exact_norm_name, exact_core_name, name_ratio, name_wratio,
        name_token_sort, name_token_set, name_partial, core_ratio,
        core_token_sort, core_token_set, name_jaccard, name_char_3gram,
        name_len_diff, addr_missing, addr_exact, addr_ratio,
        addr_token_sort, addr_token_set, addr_jaccard, addr_char_3gram,
        addr_num_match, addr_len_diff, country_match, tgt['s2'], tgt['s3'],
        # Structured address
        addr_hnum_match, addr_hnum_conflict, addr_postal_match,
        addr_postal_conflict, addr_digits_overlap, addr_digits_conflict,
        # Frequencies
        s1['nf'], tgt['nf'], s1['af'], tgt['af'],
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
