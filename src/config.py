import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Train file paths
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test file paths
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Output submission paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Column names
ENTITY_ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"

GT_S1_COL = "source1_entity_id"
GT_MATCHES_COL = "matched_entity_ids"
CANDIDATE_COL = "candidate_entity_ids"

# Supported countries
COUNTRIES = ["US", "India", "France"]

# Legal corporate suffixes across jurisdictions (US, India, France, General)
LEGAL_SUFFIXES = [
    # India
    "private limited", "pvt ltd", "pvt limited", "private ltd", "p limited",
    "limited", "ltd", "llp", "limited liability partnership",
    "opc", "one person company", "proprietorship", "enterprises",
    
    # US
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited liability company", "llc", "l l c", "l.l.c.", "l.l.c",
    "limited liability partnership", "llp", "l.l.p.",
    "professional corporation", "pc", "p.c.",
    "pllc", "p.l.l.c.", "general partnership", "gp",
    
    # France
    "societe a responsabilite limitee", "sarl", "s.a.r.l.", "s.a.r.l",
    "societe par actions simplifiee", "sas", "s.a.s.", "s.a.s",
    "societe par actions simplifiee unipersonnelle", "sasu", "s.a.s.u.",
    "societe civile immobiliere", "sci", "s.c.i.",
    "entreprise unipersonnelle a responsabilite limitee", "eurl", "e.u.r.l.",
    "societe anonyme", "sa", "s.a.",
    "societe en nom collectif", "snc",
    "groupement d interet economique", "gie",
    "succursale", "ets", "etablissements",
]

# Common address token expansions / standardizations
ADDRESS_ABBREVIATIONS = {
    # US / General
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pkwy": "parkway",
    "hwy": "highway",
    "fwy": "freeway",
    "expwy": "expressway",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "trl": "trail",
    "pk": "park",
    "pky": "parkway",
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "adj": "adjacent",
    "ext": "extension",
    
    # French address tokens
    "r": "rue",
    "rte": "route",
    "all": "allee",
    "imp": "impasse",
    "pl": "place",
    "pas": "passage",
    "faub": "faubourg",
    "chem": "chemin",
}
