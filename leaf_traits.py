"""
Leaf Traits Module
==================
Functions for loading and processing leaf morphology data from XLSX files.

Note: Needleleaf (conifer) trees don't have data in "without petiole" columns
because needles don't have petioles. For these, we use "with petiole" values.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_leaf_traits(xlsx_path: Path) -> pd.DataFrame:
    """
    Load leaf morphology data from XLSX file.
    
    Parameters
    ----------
    xlsx_path : Path
        Path to the leaf morphology XLSX file
        (e.g., 021_003_017_leaf_morphology_2023_2025.xlsx)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with processed leaf traits per tree
    """
    xlsx_path = Path(xlsx_path)
    
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Leaf traits file not found: {xlsx_path}")
    
    df = pd.read_excel(xlsx_path, sheet_name="Data_raw")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Ensure ID_tree is integer
    df["ID_tree"] = pd.to_numeric(df["ID_tree"], errors="coerce").astype("Int64")
    
    # Calculate derived metrics
    # For leaf area and weights: use "without petiole" if available,
    # otherwise fall back to "with petiole" (for needleleaf species)
    
    # Average leaf size
    df["leaf_area_cm2"] = np.nan
    if "Average size leaf without petiole" in df.columns:
        df["leaf_area_cm2"] = df["Average size leaf without petiole"]
    if "Average size leaf with petiole" in df.columns:
        # Fill NaN with "with petiole" values (for needleleaf)
        df["leaf_area_cm2"] = df["leaf_area_cm2"].fillna(df["Average size leaf with petiole"])
    
    # Fresh weight
    df["leaf_fresh_weight_g"] = np.nan
    if "Fresh weight leaf without petiole" in df.columns:
        df["leaf_fresh_weight_g"] = df["Fresh weight leaf without petiole"]
    if "Fresh weight leaf with petiole" in df.columns:
        df["leaf_fresh_weight_g"] = df["leaf_fresh_weight_g"].fillna(df["Fresh weight leaf with petiole"])
    
    # Dry weight
    df["leaf_dry_weight_g"] = np.nan
    if "Dry weight leaf without petiole" in df.columns:
        df["leaf_dry_weight_g"] = df["Dry weight leaf without petiole"]
    if "Dry weight leaf with petiole" in df.columns:
        df["leaf_dry_weight_g"] = df["leaf_dry_weight_g"].fillna(df["Dry weight leaf with petiole"])
    
    # Average thickness (from 5 measurements)
    thickness_cols = [f"Thickness_{i}" for i in range(1, 6)]
    available_thickness = [c for c in thickness_cols if c in df.columns]
    if available_thickness:
        df["leaf_thickness_mm"] = df[available_thickness].mean(axis=1)
    else:
        df["leaf_thickness_mm"] = np.nan
    
    # Average toughness
    toughness_cols = [f"Toughness_{i}" for i in range(1, 6)]
    available_toughness = [c for c in toughness_cols if c in df.columns]
    if available_toughness:
        df["toughness"] = df[available_toughness].mean(axis=1)
    else:
        df["toughness"] = np.nan
    
    # LDMC = Leaf Dry Matter Content = Dry weight / Fresh weight
    if "leaf_fresh_weight_g" in df.columns and "leaf_dry_weight_g" in df.columns:
        df["ldmc"] = df["leaf_dry_weight_g"] / df["leaf_fresh_weight_g"].replace(0, np.nan)
    else:
        df["ldmc"] = np.nan
    
    # Select relevant columns
    output_cols = [
        "ID_tree",
        "Species",
        "leaf_area_cm2",
        "leaf_thickness_mm",
        "leaf_fresh_weight_g",
        "leaf_dry_weight_g",
        "ldmc",
        "toughness",
    ]
    
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    
    return df[output_cols].copy()


def get_leaf_traits_for_tree(
    leaf_df: pd.DataFrame,
    numeric_id: int,
) -> dict:
    """
    Get leaf traits for a specific tree by numeric ID.
    
    Parameters
    ----------
    leaf_df : pd.DataFrame
        DataFrame from load_leaf_traits()
    numeric_id : int
        Tree numeric ID
        
    Returns
    -------
    dict
        Dictionary with leaf trait values (or None if not found)
    """
    row = leaf_df[leaf_df["ID_tree"] == numeric_id]
    
    if len(row) == 0:
        return {
            "leaf_area_cm2": None,
            "leaf_thickness_mm": None,
            "leaf_fresh_weight_g": None,
            "leaf_dry_weight_g": None,
            "ldmc": None,
            "toughness": None,
        }
    
    # Take first row if multiple exist
    row = row.iloc[0]
    
    return {
        "leaf_area_cm2": _safe_float(row.get("leaf_area_cm2")),
        "leaf_thickness_mm": _safe_float(row.get("leaf_thickness_mm")),
        "leaf_fresh_weight_g": _safe_float(row.get("leaf_fresh_weight_g")),
        "leaf_dry_weight_g": _safe_float(row.get("leaf_dry_weight_g")),
        "ldmc": _safe_float(row.get("ldmc")),
        "toughness": _safe_float(row.get("toughness")),
    }


def _safe_float(val) -> Optional[float]:
    """Convert to float, return None for NaN/None."""
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
