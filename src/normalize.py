import re
import unicodedata
from src.config import LEGAL_SUFFIXES, ADDRESS_ABBREVIATIONS

# Compiled regexes for fast processing
RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
RE_WHITESPACE = re.compile(r"\s+")
RE_LEADING_NOISE = re.compile(r"^[^\w]+|[^\w]+$")

# Legal suffix regex builder
sorted_suffixes = sorted(LEGAL_SUFFIXES, key=len, reverse=True)
escaped_suffixes = [re.escape(s) for s in sorted_suffixes]
SUFFIX_PATTERN_STR = r"\b(?:" + "|".join(escaped_suffixes) + r")\b"
RE_LEGAL_SUFFIXES = re.compile(SUFFIX_PATTERN_STR, re.IGNORECASE)

# Address abbreviation pattern builder
sorted_abbr = sorted(ADDRESS_ABBREVIATIONS.keys(), key=len, reverse=True)
ABBR_PATTERN_STR = r"\b(?:" + "|".join(re.escape(k) for k in sorted_abbr) + r")\b"
RE_ADDR_ABBR = re.compile(ABBR_PATTERN_STR, re.IGNORECASE)

def _expand_abbr(match):
    token = match.group(0).lower()
    return ADDRESS_ABBREVIATIONS.get(token, token)

# Domain extension stripping
RE_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9]+)\.(?:com|org|net|io|co|in|fr|gov|edu|biz|info)\b", re.IGNORECASE)
RE_HANDLE_PREFIX = re.compile(r"^[@#+\-]+")

# Generic business terms that shouldn't be single-token blocking keys
GENERIC_STOP_WORDS = {
    "services", "service", "enterprises", "enterprise", "group", "holdings",
    "holding", "solutions", "solution", "management", "consulting", "consultants",
    "consultant", "associates", "associate", "international", "national",
    "global", "trading", "industries", "industry", "ventures", "venture",
    "retail", "commercial", "finance", "financial", "store", "stores",
    "center", "centre", "corporation", "company", "limited", "private",
    "agency", "marketing", "media", "technologies", "technology", "tech",
    "systems", "system", "properties", "property", "realty", "real", "estate",
    "hotel", "restaurant", "cafe", "clinic", "hospital", "school", "academy",
    "general", "public", "plus", "and", "the", "for", "with", "club",
}

def strip_accents(text: str) -> str:
    """Normalize unicode characters, stripping combining diacritical marks (e.g. é -> e, à -> a)

    while preserving non-Latin scripts (e.g., Devanagari Hindi/Marathi).
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")

def normalize_text_basic(text: str) -> str:
    """Basic normalization: strip accents, lower, strip domain extensions, handle symbols, clean noise."""
    if not text:
        return ""
    text = strip_accents(str(text)).lower()
    
    # Strip domain names (e.g., primemoney.com -> primemoney)
    text = RE_DOMAIN.sub(r"\1", text)
    
    # Strip leading handles (@primemoney -> primemoney)
    text = RE_HANDLE_PREFIX.sub("", text)
    
    text = text.replace("&", " and ").replace("+", " plus ").replace("@", " ")
    text = RE_PUNCT.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text

def normalize_business_name(name: str) -> str:
    """Standardized normalized business name."""
    return normalize_text_basic(name)

def extract_core_business_name(norm_name: str) -> str:
    """Remove legal entity suffixes (e.g., 'inc', 'llc', 'pvt ltd', 'sarl')

    from the normalized business name.
    """
    if not norm_name:
        return ""
    cleaned = RE_LEGAL_SUFFIXES.sub(" ", norm_name)
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()
    return cleaned if cleaned else norm_name

def normalize_business_address(address: str) -> str:
    """Standardize and expand address abbreviations."""
    if not address:
        return ""
    norm = normalize_text_basic(address)
    if not norm:
        return ""
    expanded = RE_ADDR_ABBR.sub(_expand_abbr, norm)
    return RE_WHITESPACE.sub(" ", expanded).strip()

# Regex for PIN/Zip codes and Indian house numbers
RE_PIN_ZIP = re.compile(r"\b\d{5,6}\b")
RE_HOUSE_NUM = re.compile(r"\b\d+[-/]\d+(?:[-/][a-z0-9]+)?\b", re.IGNORECASE)

def extract_name_blocking_keys(core_name: str, norm_name: str) -> list[str]:
    """Extract diverse blocking keys for candidate generation:

    1. Compact core name
    2. 4-character prefix & 3-character prefix (handles typos & phonetic shifts)
    3. First 2 tokens
    4. First 3 tokens (if >= 3 tokens)
    5. First & Last token (handles dropped middle words)
    6. Non-generic significant single tokens
    7. Sorted 2 tokens
    """
    keys = set()
    target = core_name if core_name else norm_name
    if not target:
        return []
    
    tokens = [t for t in target.split() if len(t) > 1]
    if not tokens:
        tokens = target.split()
        
    if not tokens:
        return []
        
    # Key 1: Compact core name
    compact_name = "".join(tokens)
    if len(compact_name) >= 3:
        keys.add(f"name:{compact_name[:30]}")
        
    # Key 2: 4-character prefix and 3-character prefix
    if len(compact_name) >= 4:
        keys.add(f"pfx4:{compact_name[:4]}")
    if len(compact_name) >= 3:
        keys.add(f"pfx3:{compact_name[:3]}")
        
    # Key 3: First 2 tokens
    if len(tokens) >= 2:
        keys.add(f"tok2:{tokens[0]}_{tokens[1]}")
    elif len(tokens) == 1 and len(tokens[0]) >= 3 and tokens[0] not in GENERIC_STOP_WORDS:
        keys.add(f"tok1:{tokens[0]}")
        
    # Key 4: First 3 tokens if available
    if len(tokens) >= 3:
        keys.add(f"tok3:{tokens[0]}_{tokens[1]}_{tokens[2]}")
        
    # Key 5: First & Last token (handles dropped middle words)
    if len(tokens) >= 3:
        keys.add(f"fl:{tokens[0]}_{tokens[-1]}")
        
    # Key 6: Non-generic significant single tokens
    for t in tokens:
        if len(t) >= 4 and t not in GENERIC_STOP_WORDS:
            keys.add(f"tok1:{t}")
            
    # Key 7: Sorted 2 tokens
    if len(tokens) >= 2:
        sorted_toks = sorted(tokens[:4])
        keys.add(f"sort2:{sorted_toks[0]}_{sorted_toks[1]}")
        
    return list(keys)

def extract_address_blocking_keys(norm_address: str) -> list[str]:
    """Extract compound address keys: PIN/ZIP, complex house numbers, number + street."""
    if not norm_address:
        return []
    
    keys = []
    
    # 1. Complex house numbers (e.g., 16-11-23, 570/13)
    house_nums = RE_HOUSE_NUM.findall(norm_address)
    for hn in house_nums[:2]:
        keys.append(f"hnum:{hn}")
        
    # 2. PIN / ZIP codes (5 or 6 digits)
    pincodes = RE_PIN_ZIP.findall(norm_address)
    for pin in pincodes[:2]:
        keys.append(f"pin:{pin}")
        
    # 3. Unpadded number + adjacent street token
    tokens = norm_address.split()
    for i, tok in enumerate(tokens):
        if tok.isdigit() and 1 <= len(tok) <= 8:
            unpadded = str(int(tok))
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if len(next_tok) >= 3 and not next_tok.isdigit() and next_tok not in ADDRESS_ABBREVIATIONS.values():
                    keys.append(f"addr:{unpadded}_{next_tok}")
            if len(unpadded) >= 4:
                keys.append(f"num:{unpadded}")
                
    return keys

