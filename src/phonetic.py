import re

# Fast Phonetic Consonant Skeleton / Reducer
RE_VOWELS = re.compile(r"[aeiouy]+")
RE_DUPES = re.compile(r"(.)\1+")

def phonetic_skeleton(token: str) -> str:
    """Compute phonetic skeleton for cross-lingual English-Indic matching (e.g. dream <-> dreem -> drim)."""
    if not token or len(token) < 2:
        return token
    t = token.lower()
    
    # 1. Phonetic substitutions
    t = t.replace("ph", "f").replace("c", "k").replace("q", "k").replace("x", "ks")
    t = t.replace("ee", "i").replace("ea", "i").replace("oo", "u").replace("ou", "u")
    t = t.replace("sh", "s").replace("ch", "s").replace("th", "t").replace("dh", "d")
    t = t.replace("bh", "b").replace("gh", "g").replace("kh", "k").replace("jh", "j")
    t = t.replace("v", "w").replace("z", "s")
    
    # 2. Collapse repeated characters
    t = RE_DUPES.sub(r"\1", t)
    
    # 3. Preserve first char, collapse inner vowels
    if len(t) > 3:
        first = t[0]
        rest = RE_VOWELS.sub("", t[1:])
        return (first + rest)[:6]
    return t[:5]
