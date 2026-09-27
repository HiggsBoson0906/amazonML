import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import polars as pl

from src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    ENTITY_ID_COL,
    NAME_COL,
    ADDRESS_COL,
    COUNTRY_COL,
    GT_S1_COL,
    GT_MATCHES_COL,
)
from src.normalize import (
    normalize_business_name,
    extract_core_business_name,
    normalize_business_address,
)

def load_ground_truth(path: Path = TRAIN_GROUND_TRUTH) -> Dict[str, Set[str]]:
    """Load train ground truth into a dictionary mapping source1_entity_id -> set of matched_entity_ids."""
    gt_map: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                matched_ids = {m.strip() for m in parts[1].split(",") if m.strip()}
            else:
                matched_ids = set()
            gt_map[s1_id] = matched_ids
    return gt_map

def load_source_tsv(
    path: Path,
    n_rows: Optional[int] = None,
    filter_country: Optional[str] = None,
) -> pl.DataFrame:
    """Load a source TSV file into a Polars DataFrame with standardized types."""
    df = pl.read_csv(
        path,
        separator="\t",
        n_rows=n_rows,
        has_header=True,
        schema={
            ENTITY_ID_COL: pl.Utf8,
            NAME_COL: pl.Utf8,
            ADDRESS_COL: pl.Utf8,
            COUNTRY_COL: pl.Utf8,
        },
        null_values=["", "NULL", "null", "None"],
    )
    
    # Fill null addresses with empty string
    df = df.with_columns(
        pl.col(ADDRESS_COL).fill_null(""),
        pl.col(NAME_COL).fill_null(""),
        pl.col(COUNTRY_COL).fill_null(""),
    )
    
    if filter_country:
        df = df.filter(pl.col(COUNTRY_COL) == filter_country)
        
    return df

def load_and_normalize_source(
    path: Path,
    n_rows: Optional[int] = None,
    filter_country: Optional[str] = None,
) -> pl.DataFrame:
    """Load a source TSV file and compute normalized string columns."""
    df = load_source_tsv(path, n_rows=n_rows, filter_country=filter_country)
    
    # Extract columns as lists for fast batch Python normalization
    names = df[NAME_COL].to_list()
    addresses = df[ADDRESS_COL].to_list()
    
    norm_names = [normalize_business_name(n) for n in names]
    core_names = [extract_core_business_name(n) for n in norm_names]
    norm_addresses = [normalize_business_address(a) for a in addresses]
    
    df = df.with_columns(
        pl.Series("norm_name", norm_names, dtype=pl.Utf8),
        pl.Series("core_name", core_names, dtype=pl.Utf8),
        pl.Series("norm_address", norm_addresses, dtype=pl.Utf8),
    )
    
    return df
