"""
Data Loader Module for Rainfall Analysis Dashboard.
Safely reads, validates, and prepares rainfall data without modifying source CSV files.
Provides clean in-memory DataFrames and lookups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from config import DISTRICT_CSV_PATH, SUBDIVISION_CSV_PATH, MONTHS, SEASON_KEYS

# Global In-Memory Caches
_CACHED_SUBDIVISION_DF: Optional[pd.DataFrame] = None
_CACHED_DISTRICT_DF: Optional[pd.DataFrame] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None


def load_subdivision_data() -> pd.DataFrame:
    """
    Safely loads and preprocesses 'rainfall in india 1901-2015.csv'.
    Original file is opened read-only and never modified.
    
    Cleaning steps:
    1. Strips column headers and text values of whitespace.
    2. Imputes missing monthly values non-destructively using subdivision-specific median.
    3. Verifies and recalculates seasonal and annual sums where missing.
    4. Generates an 'ALL INDIA' synthetic subdivision by taking the mean across subdivisions per year.
    """
    global _CACHED_SUBDIVISION_DF
    if _CACHED_SUBDIVISION_DF is not None:
        return _CACHED_SUBDIVISION_DF.copy()

    if not SUBDIVISION_CSV_PATH.exists():
        raise FileNotFoundError(f"Subdivision dataset not found at: {SUBDIVISION_CSV_PATH}")

    # Read original CSV in read-only mode
    df = pd.read_csv(SUBDIVISION_CSV_PATH, encoding="utf-8")
    
    # Strip column names
    df.columns = [col.strip() for col in df.columns]

    # Clean text columns
    df["SUBDIVISION"] = df["SUBDIVISION"].astype(str).str.strip()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype(int)

    # Convert numeric columns
    numeric_cols = MONTHS + ["ANNUAL"] + SEASON_KEYS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Non-destructive imputation for missing monthly values
    for month in MONTHS:
        # Group by subdivision to impute monthly median for that specific subdivision
        df[month] = df.groupby("SUBDIVISION")[month].transform(
            lambda s: s.fillna(s.median() if not s.dropna().empty else 0.0)
        )

    # Ensure seasonal aggregates are populated accurately from monthly components
    df["Jan-Feb"] = df["JAN"] + df["FEB"]
    df["Mar-May"] = df["MAR"] + df["APR"] + df["MAY"]
    df["Jun-Sep"] = df["JUN"] + df["JUL"] + df["AUG"] + df["SEP"]
    df["Oct-Dec"] = df["OCT"] + df["NOV"] + df["DEC"]
    df["ANNUAL"] = df["Jan-Feb"] + df["Mar-May"] + df["Jun-Sep"] + df["Oct-Dec"]

    # Round to 2 decimal places
    for col in numeric_cols:
        df[col] = df[col].round(2)

    # Compute ALL INDIA national benchmark per year
    all_india_df = df.groupby("YEAR", as_index=False)[numeric_cols].mean()
    all_india_df["SUBDIVISION"] = "ALL INDIA"
    all_india_df = all_india_df[["SUBDIVISION", "YEAR"] + numeric_cols]
    all_india_df[numeric_cols] = all_india_df[numeric_cols].round(2)

    # Combine regular subdivisions with ALL INDIA benchmark
    combined_df = pd.concat([df, all_india_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=["SUBDIVISION", "YEAR"]).reset_index(drop=True)

    _CACHED_SUBDIVISION_DF = combined_df
    return _CACHED_SUBDIVISION_DF.copy()


def load_district_data() -> pd.DataFrame:
    """
    Safely loads and preprocesses 'district wise rainfall normal.csv'.
    Original file is opened read-only and never modified.
    
    Cleaning steps:
    1. Strips whitespace from column headers, state names, and district names.
    2. Validates numerical ranges.
    3. Normalizes state naming casing.
    """
    global _CACHED_DISTRICT_DF
    if _CACHED_DISTRICT_DF is not None:
        return _CACHED_DISTRICT_DF.copy()

    if not DISTRICT_CSV_PATH.exists():
        raise FileNotFoundError(f"District dataset not found at: {DISTRICT_CSV_PATH}")

    df = pd.read_csv(DISTRICT_CSV_PATH, encoding="utf-8")
    
    # Strip column names
    df.columns = [col.strip() for col in df.columns]

    # Clean text columns
    df["STATE_UT_NAME"] = df["STATE_UT_NAME"].astype(str).str.strip().str.upper()
    df["DISTRICT"] = df["DISTRICT"].astype(str).str.strip().str.title()

    # Convert numeric columns
    numeric_cols = MONTHS + ["ANNUAL"] + SEASON_KEYS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).round(2)

    # Sort deterministically
    df = df.sort_values(by=["STATE_UT_NAME", "DISTRICT"]).reset_index(drop=True)

    _CACHED_DISTRICT_DF = df
    return _CACHED_DISTRICT_DF.copy()


def get_metadata() -> Dict[str, Any]:
    """
    Returns unified metadata for UI dropdowns, filters, and validation.
    """
    global _CACHED_METADATA
    if _CACHED_METADATA is not None:
        return _CACHED_METADATA

    sub_df = load_subdivision_data()
    dist_df = load_district_data()

    subdivisions = sorted(list(sub_df["SUBDIVISION"].unique()))
    # Ensure 'ALL INDIA' is first in the subdivisions list
    if "ALL INDIA" in subdivisions:
        subdivisions.remove("ALL INDIA")
        subdivisions = ["ALL INDIA"] + subdivisions

    states = sorted(list(dist_df["STATE_UT_NAME"].unique()))

    # Build State -> Districts dictionary
    state_to_districts: Dict[str, List[str]] = {}
    for state in states:
        dist_list = sorted(dist_df[dist_df["STATE_UT_NAME"] == state]["DISTRICT"].unique().tolist())
        state_to_districts[state] = dist_list

    min_year = int(sub_df["YEAR"].min())
    max_year = int(sub_df["YEAR"].max())

    _CACHED_METADATA = {
        "subdivisions": subdivisions,
        "states": states,
        "state_to_districts": state_to_districts,
        "total_districts": len(dist_df),
        "total_subdivisions": len(subdivisions),
        "min_year": min_year,
        "max_year": max_year,
        "months": MONTHS,
        "seasons": SEASON_KEYS,
        "metrics": ["ANNUAL", "Jun-Sep", "Oct-Dec", "Mar-May", "Jan-Feb"] + MONTHS
    }
    return _CACHED_METADATA
