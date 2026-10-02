from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from tqdm import tqdm

from src.normalize import (
    extract_name_blocking_keys,
    extract_address_blocking_keys,
    extract_core_business_name,
    normalize_business_name,
    normalize_business_address,
    GENERIC_STOP_WORDS,
    ADDRESS_ABBREVIATIONS,
)

from src.phonetic import phonetic_skeleton
from src.transliterate import transliterate_indic_to_latin

def get_translit_blocking_keys(name: str) -> List[str]:
    """Generate cross-lingual phonetic and transliterated keys for Indic-English matching."""
    if not name:
        return []
    has_indic = any(ord(c) >= 0x0900 for c in name)
    latin_name = transliterate_indic_to_latin(name) if has_indic else name
    
    core_name = extract_core_business_name(normalize_business_name(latin_name))
    tokens = [t for t in core_name.split() if len(t) >= 3 and t not in GENERIC_STOP_WORDS]
    
    keys = []
    if has_indic:
        compact = "".join(tokens)
        if len(compact) >= 4:
            keys.append(f"tr_name:{compact[:25]}")
            keys.append(f"tr_pfx4:{compact[:4]}")
            
    ph_tokens = [phonetic_skeleton(t) for t in tokens if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
    if len(ph_tokens) >= 2:
        keys.append(f"ph_tok2:{ph_tokens[0]}_{ph_tokens[1]}")
    elif len(ph_tokens) == 1 and len(ph_tokens[0]) >= 3:
        keys.append(f"ph_tok1:{ph_tokens[0]}")
        
    return keys

def get_enhanced_address_keys(norm_addr: str) -> List[str]:
    """Extract fine-grained address blocking keys."""
    if not norm_addr:
        return []
    keys = []
    tokens = norm_addr.split()
    nums = []
    words = []
    for t in tokens:
        if t.isdigit() and 1 <= len(t) <= 8:
            nums.append(str(int(t)))
        elif len(t) >= 4 and t not in ADDRESS_ABBREVIATIONS.values():
            words.append(t)
            
    for n in nums[:2]:
        for w in words[:2]:
            keys.append(f"num_word:{n}_{w}")
            
    if len(words) >= 2:
        keys.append(f"loc_pair:{words[-1]}_{words[0]}")
    return keys

def get_stage5_acronym_and_token_keys(name: str) -> List[str]:
    """Extract acronym keys and ordered token keys for abbreviation matching."""
    if not name:
        return []
    has_indic = any(ord(c) >= 0x0900 for c in name)
    latin_name = transliterate_indic_to_latin(name) if has_indic else name
    core = extract_core_business_name(normalize_business_name(latin_name))
    tokens = [t for t in core.split() if len(t) >= 2 and t not in GENERIC_STOP_WORDS]
    
    keys = []
    if len(tokens) >= 2:
        acronym = "".join(t[0] for t in tokens if t[0].isalnum())
        if 2 <= len(acronym) <= 6:
            keys.append(f"acro:{acronym}")
    if len(tokens) >= 3:
        keys.append(f"tok3_ord:{tokens[0]}_{tokens[1]}_{tokens[2]}")
    if tokens and len(tokens[0]) >= 4 and tokens[0] not in GENERIC_STOP_WORDS:
        keys.append(f"pfx3_tok1:{tokens[0][:3]}")
    return keys

def get_stage5_address_keys(norm_addr: str) -> List[str]:
    """Extract robust street + postal word combinations."""
    if not norm_addr:
        return []
    keys = []
    tokens = norm_addr.split()
    distinct_words = [t for t in tokens if len(t) >= 4 and not t.isdigit() and t not in ADDRESS_ABBREVIATIONS.values()]
    numbers = [str(int(t)) for t in tokens if t.isdigit() and 1 <= len(t) <= 8]
    
    if numbers and distinct_words:
        keys.append(f"num_w1:{numbers[0]}_{distinct_words[0]}")
        if len(distinct_words) >= 2:
            keys.append(f"num_w2:{numbers[0]}_{distinct_words[-1]}")
    if len(distinct_words) >= 2:
        keys.append(f"addr_word2:{distinct_words[0]}_{distinct_words[1]}")
    return keys

class InvertedIndexBlocker:
    """Production Multi-pass Inverted Index Blocker with Tiered Quotas, Transliteration, and Address Signatures."""
    
    def __init__(
        self,
        max_key_frequency: int = 500,
        max_candidates_per_s1: int = 160,
    ):
        self.max_key_frequency = max_key_frequency
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
    def _extract_all_keys(self, n_name: str, c_name: str, n_addr: str, raw_name: str = "") -> List[str]:
        keys = []
        # Stage 4 Base keys
        keys.extend(extract_name_blocking_keys(c_name, n_name))
        keys.extend(extract_address_blocking_keys(n_addr))
        keys.extend(get_translit_blocking_keys(raw_name if raw_name else n_name))
        keys.extend(get_enhanced_address_keys(n_addr))
        
        # Stage 5 Additions
        keys.extend(get_stage5_acronym_and_token_keys(raw_name if raw_name else n_name))
        keys.extend(get_stage5_address_keys(n_addr))
        
        # Phonetic token + address number compound
        has_indic = any(ord(c) >= 0x0900 for c in (raw_name or n_name))
        latin_name = transliterate_indic_to_latin(raw_name or n_name) if has_indic else (raw_name or n_name)
        core = extract_core_business_name(normalize_business_name(latin_name))
        tokens = [t for t in core.split() if len(t) >= 4 and t not in GENERIC_STOP_WORDS]
        addr_nums = [str(int(t)) for t in n_addr.split() if t.isdigit() and 1 <= len(t) <= 8]
        if tokens and addr_nums:
            ph = phonetic_skeleton(tokens[0])
            keys.append(f"ph_num:{ph}_{addr_nums[0]}")
            
        return keys

    def add_target_records(
        self,
        df: Any,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        norm_name_col: str = "norm_name",
        core_name_col: str = "core_name",
        norm_addr_col: str = "norm_address",
        country_col: str = "country",
    ):
        ids = df[id_col].to_list()
        norm_names = df[norm_name_col].to_list()
        core_names = df[core_name_col].to_list()
        raw_names = df[name_col].to_list() if name_col in df.columns else [""] * len(ids)
        norm_addrs = df[norm_addr_col].to_list() if norm_addr_col in df.columns else [""] * len(ids)
        countries = df[country_col].to_list()
        
        for record_id, n_name, c_name, raw_name, n_addr, country in zip(ids, norm_names, core_names, raw_names, norm_addrs, countries):
            country_clean = country.strip() if country else ""
            if not country_clean:
                continue
            
            keys = self._extract_all_keys(n_name, c_name, n_addr, raw_name)
            country_idx = self.index[country_clean]
            for k in keys:
                country_idx[k].append(record_id)
                
    def prune_high_frequency_keys(self):
        for country, keys_dict in self.index.items():
            to_delete = []
            for k, id_list in keys_dict.items():
                if k.startswith(("name:", "pin:", "hnum:", "tr_name:", "acro:", "num_w1:")):
                    if len(id_list) > 1000:
                        to_delete.append(k)
                elif len(id_list) > self.max_key_frequency:
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
        if k.startswith(("name:", "pin:", "hnum:", "tr_name:", "addr:", "num_word:", "num_w1:", "num_w2:", "ph_num:")):
            return 40
        if k.startswith(("tok3:", "tok3_ord:", "tok2:", "ph_tok2:", "tr_tok2:", "fl:", "sort2:", "pfx4:", "tr_pfx4:", "acro:")):
            return 20
        if k.startswith(("pfx3:", "pfx3_tok1:", "addr_word2:", "num:", "tok1:", "ph_tok1:", "tr_tok1:", "loc_pair:")):
            return 8
        return 12

    def generate_candidates_for_s1(
        self,
        df_s1: Any,
        id_col: str = "entity_id",
        name_col: str = "business_name",
        norm_name_col: str = "norm_name",
        core_name_col: str = "core_name",
        norm_addr_col: str = "norm_address",
        country_col: str = "country",
    ) -> Dict[str, Set[str]]:
        if isinstance(df_s1, dict):
            ids = df_s1.get(id_col, [])
            norm_names = df_s1.get(norm_name_col, [])
            core_names = df_s1.get(core_name_col, [])
            raw_names = df_s1.get(name_col, [""] * len(ids))
            norm_addrs = df_s1.get(norm_addr_col, [""] * len(ids))
            countries = df_s1.get(country_col, [])
        else:
            ids = df_s1[id_col].to_list()
            norm_names = df_s1[norm_name_col].to_list()
            core_names = df_s1[core_name_col].to_list()
            raw_names = df_s1[name_col].to_list() if name_col in df_s1.columns else [""] * len(ids)
            norm_addrs = df_s1[norm_addr_col].to_list() if norm_addr_col in df_s1.columns else [""] * len(ids)
            countries = df_s1[country_col].to_list()
        
        candidates: Dict[str, Set[str]] = {}
        
        for s1_id, n_name, c_name, raw_name, n_addr, country in zip(ids, norm_names, core_names, raw_names, norm_addrs, countries):
            country_clean = country.strip() if country else ""
            cand_set: Set[str] = set()
            
            if country_clean in self.index:
                c_idx = self.index[country_clean]
                keys = self._extract_all_keys(n_name, c_name, n_addr, raw_name)
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
                            if len(cand_set) >= self.max_candidates_per_s1:
                                break
                    if len(cand_set) >= self.max_candidates_per_s1:
                        break
                        
            candidates[s1_id] = cand_set
            
        return candidates
