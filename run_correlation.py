"""
Data Preparation and Correlation Analysis for Tree Sound Absorption
====================================================================
Merges acoustic and structural data, checks collinearity,
creates correlation heatmaps and TF visualizations.

Usage:
    python run_correlation.py --output-dir ./output

Expects in output/csv/:
    - tree_acoustic_summary.csv (from Python pipeline)
    - tree_structural_traits.csv (from R pipeline)

Outputs:
    CSV:
        - output/csv/merged_data.csv
        - output/csv/correlation_matrix.csv
        - output/csv/collinearity_report.txt
    
    Figures (output/figures/combined/):
        - trait_correlations.png
        - predictor_response_all.png
        - predictor_response_broadleaf.png
        - predictor_response_needleleaf.png
        - predictor_response_mixed.png
        - tfs_broadleaf.png
        - tfs_needleleaf.png
        - tfs_mixed.png
"""
import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

from config import (
    EXCLUDE_IDS,
    CB_INDIGO, CB_ROSE, CB_PURPLE,
    FREQ_LOW, FREQ_HIGH,
    OUT_DIR,
)
from leaf_traits import load_leaf_traits

# Predictor/response definitions (kept here since they're for statistical analysis)
# Note: Precursor predictors removed (densest_slice_height, distance, pts_per_voxel, dry/fresh weight)
STRUCTURAL_PREDICTORS = [
    "height_m", "crown_volume_m3", "vol_fill_fraction", "gap_fraction", 
    "leaf_voxel_density", "ENL", "FHD", "LAD",
]

LEAF_PREDICTORS = [
    "leaf_area_cm2", "leaf_thickness_mm", "ldmc", "toughness",
]

# Precursor predictors - computed but excluded from analysis
# (used to derive other metrics or are measurement artifacts)
PRECURSOR_PREDICTORS = [
    "densest_slice_height_m",  # Used to determine measurement height
    "distance_m",              # Measurement setup parameter
    "pts_per_voxel",           # TLS artifact, not biological
    "leaf_fresh_weight_g",     # Precursor to LDMC
    "leaf_dry_weight_g",       # Precursor to LDMC
]

ALL_PREDICTORS = STRUCTURAL_PREDICTORS + LEAF_PREDICTORS

RESPONSE_VARS = [
    "attenuation_low_db", "attenuation_mid_db",
    "attenuation_high_db", "attenuation_overall_db",
]

warnings.filterwarnings("ignore")


def load_and_merge_data(output_dir: Path, leaf_xlsx: Path = None) -> pd.DataFrame:
    """Load and merge acoustic + structural + leaf trait data."""
    output_dir = Path(output_dir)
    
    # Load acoustic data
    acoustic_path = output_dir / "csv" / "tree_acoustic_summary.csv"
    if not acoustic_path.exists():
        raise FileNotFoundError(f"Acoustic data not found: {acoustic_path}")
    
    df_acoustic = pd.read_csv(acoustic_path)
    print(f"Loaded acoustic data: {len(df_acoustic)} trees")
    
    # Load structural data
    structural_path = output_dir / "csv" / "tree_structural_traits.csv"
    if not structural_path.exists():
        raise FileNotFoundError(f"Structural data not found: {structural_path}")
    
    df_structural = pd.read_csv(structural_path)
    print(f"Loaded structural data: {len(df_structural)} trees")
    
    # Load leaf traits from xlsx
    df_leaf = None
    if leaf_xlsx is not None:
        leaf_xlsx = Path(leaf_xlsx)
        if leaf_xlsx.exists():
            try:
                df_leaf = load_leaf_traits(leaf_xlsx)
                # Rename ID_tree to numeric_id for merging
                if "ID_tree" in df_leaf.columns:
                    df_leaf = df_leaf.rename(columns={"ID_tree": "numeric_id"})
                print(f"Loaded leaf traits: {len(df_leaf)} entries")
            except Exception as e:
                print(f"Warning: Could not load leaf traits: {e}")
        else:
            print(f"Warning: Leaf traits file not found: {leaf_xlsx}")
    
    # Filter excluded IDs
    df_acoustic = df_acoustic[~df_acoustic["numeric_id"].isin(EXCLUDE_IDS)]
    df_structural = df_structural[~df_structural["numeric_id"].isin(EXCLUDE_IDS)]
    
    # Merge acoustic + structural
    df = pd.merge(df_acoustic, df_structural, on="numeric_id", how="inner", suffixes=("", "_struct"))
    
    # Merge leaf traits if available
    if df_leaf is not None:
        df_leaf = df_leaf[~df_leaf["numeric_id"].isin(EXCLUDE_IDS)]
        df = pd.merge(df, df_leaf, on="numeric_id", how="left")
        print(f"  Merged leaf traits for {df['leaf_area_cm2'].notna().sum()} trees")
    
    # Ensure leaf_type column exists
    if "leaf_type" not in df.columns:
        df["leaf_type"] = df["is_needleleaf"].apply(lambda x: "needleleaf" if x else "broadleaf")
    
    print(f"\nMerged dataset: {len(df)} trees")
    print(f"  Broadleaf: {sum(df['leaf_type'] == 'broadleaf')}")
    print(f"  Needleleaf: {sum(df['leaf_type'] == 'needleleaf')}")
    
    # Diagnostic: show which predictors are available
    print("\nAvailable predictors in merged data:")
    available_struct = [p for p in STRUCTURAL_PREDICTORS if p in df.columns]
    available_leaf = [p for p in LEAF_PREDICTORS if p in df.columns]
    missing_struct = [p for p in STRUCTURAL_PREDICTORS if p not in df.columns]
    missing_leaf = [p for p in LEAF_PREDICTORS if p not in df.columns]
    
    print(f"  Structural ({len(available_struct)}/{len(STRUCTURAL_PREDICTORS)}): {available_struct}")
    if missing_struct:
        print(f"    MISSING: {missing_struct}")
    print(f"  Leaf traits ({len(available_leaf)}/{len(LEAF_PREDICTORS)}): {available_leaf}")
    if missing_leaf:
        print(f"    MISSING: {missing_leaf}")
    
    return df


def standardize_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize all numeric predictors."""
    df_out = df.copy()
    available = [p for p in ALL_PREDICTORS if p in df.columns]
    
    for col in available:
        values = df[col].values.astype(float)
        valid = ~np.isnan(values)
        if valid.sum() > 1:
            mean, std = np.nanmean(values), np.nanstd(values)
            df_out[f"{col}_z"] = (values - mean) / std if std > 0 else 0.0
        else:
            df_out[f"{col}_z"] = np.nan
    
    return df_out


def check_collinearity(df: pd.DataFrame, output_dir: Path):
    """Check collinearity among predictors."""
    available = [p for p in ALL_PREDICTORS if p in df.columns and df[p].notna().sum() > 3]
    
    corr = df[available].corr()
    corr.to_csv(output_dir / "correlation_matrix.csv")
    
    high_corr = []
    for i, col1 in enumerate(available):
        for j, col2 in enumerate(available):
            if i < j:
                r = corr.loc[col1, col2]
                if not np.isnan(r) and abs(r) > 0.7:
                    high_corr.append((col1, col2, r))
    
    vif_results = {}
    for col in available:
        other_cols = [c for c in available if c != col]
        subset = df[[col] + other_cols].dropna()
        if len(subset) > len(other_cols) + 1:
            X, y = subset[other_cols].values, subset[col].values
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            vif_results[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    
    report_path = output_dir / "collinearity_report.txt"
    with open(report_path, "w") as f:
        f.write("COLLINEARITY ANALYSIS REPORT\n" + "=" * 60 + "\n\n")
        f.write("HIGH CORRELATIONS (|r| > 0.7):\n" + "-" * 40 + "\n")
        for v1, v2, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            f.write(f"  {v1} <-> {v2}: r = {r:.3f}\n")
        f.write("\nVARIANCE INFLATION FACTORS:\n" + "-" * 40 + "\n")
        for var, vif in sorted(vif_results.items(), key=lambda x: -x[1]):
            flag = " *** EXCLUDE" if vif > 10 else " ** HIGH" if vif > 5 else ""
            f.write(f"  {var:25s}: {vif:6.2f}{flag}\n")
    
    print(f"  Collinearity report: {report_path.name}")
    return corr, high_corr, vif_results


def setup_style():
    """Clean plot styling."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def plot_trait_correlation_heatmap(df: pd.DataFrame, save_path: Path):
    """Heatmap of correlations between ALL predictors."""
    available = [c for c in ALL_PREDICTORS if c in df.columns and df[c].notna().sum() > 3]
    if len(available) < 2:
        return
    
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Predictor Correlations", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  {save_path.name}")


def plot_predictor_response_heatmap(df: pd.DataFrame, leaf_type: str, save_path: Path):
    """Heatmap of correlations between predictors and acoustic responses."""
    if leaf_type == "mixed":
        df_sub = df
        title_suffix = " (Mixed - All Trees)"
    else:
        df_sub = df[df["leaf_type"] == leaf_type]
        title_suffix = f" ({leaf_type.capitalize()})"
    
    available_preds = [c for c in ALL_PREDICTORS if c in df_sub.columns and df_sub[c].notna().sum() > 3]
    available_resp = [c for c in RESPONSE_VARS if c in df_sub.columns]
    
    if len(available_preds) < 1 or len(df_sub) < 4:
        return
    
    corr_data, annot_data = [], []
    for pred in available_preds:
        row_corr, row_annot = [], []
        for resp in available_resp:
            valid = df_sub[[pred, resp]].dropna()
            if len(valid) > 3:
                r, p = stats.pearsonr(valid[pred], valid[resp])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""
                row_corr.append(r)
                row_annot.append(f"{r:.2f}{stars}")
            else:
                row_corr.append(np.nan)
                row_annot.append("")
        corr_data.append(row_corr)
        annot_data.append(row_annot)
    
    col_labels = [c.replace("attenuation_", "").replace("_db", "") for c in available_resp]
    corr_df = pd.DataFrame(corr_data, index=available_preds, columns=col_labels)
    annot_df = pd.DataFrame(annot_data, index=available_preds, columns=col_labels)
    
    fig, ax = plt.subplots(figsize=(7, 10))
    sns.heatmap(corr_df, annot=annot_df, fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Predictors vs Attenuation{title_suffix}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  {save_path.name}")


def plot_tfs_by_leaf_type(df: pd.DataFrame, output_dir: Path, base_output_dir: Path):
    """TF overlay plots for broadleaf, needleleaf, and mixed."""
    pkl_dir = base_output_dir / "pkl"
    if not pkl_dir.exists():
        return
    
    # Color scheme: red(broad) + blue(needle) = purple(mixed)
    colors = {"broadleaf": CB_ROSE, "needleleaf": CB_INDIGO, "mixed": CB_PURPLE}
    
    for leaf_type in ["broadleaf", "needleleaf", "mixed"]:
        subset = df if leaf_type == "mixed" else df[df["leaf_type"] == leaf_type]
        color = colors[leaf_type]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        n_plotted = 0
        
        for _, row in subset.iterrows():
            pkl_path = pkl_dir / f"{row['tree_id']}.pkl"
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    tree = pickle.load(f)
                freqs = getattr(tree, "freqs", None)
                tf_db = getattr(tree, "tf_magnitude_db", None)
                if freqs is not None and tf_db is not None:
                    ax.plot(freqs, tf_db, color=color, alpha=0.6, linewidth=1.0)
                    n_plotted += 1
            except Exception:
                continue
        
        if n_plotted == 0:
            plt.close(fig)
            continue
        
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlim(FREQ_LOW, FREQ_HIGH)
        ax.set_ylim(-60, 20)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [dB]")
        ax.set_title(f"Transfer Functions - {leaf_type.capitalize()} (n={n_plotted})", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"tfs_{leaf_type}.png", dpi=200)
        plt.close(fig)
        print(f"  tfs_{leaf_type}.png ({n_plotted} trees)")


def plot_tfs_by_trait(df: pd.DataFrame, trait: str, leaf_type: str, 
                      output_dir: Path, base_output_dir: Path):
    """
    Plot TFs colored by a continuous trait value.
    Separate for broadleaf, needleleaf, or mixed.
    """
    if trait not in df.columns or df[trait].isna().all():
        return
    
    if leaf_type == "mixed":
        subset = df
    else:
        subset = df[df["leaf_type"] == leaf_type]
    
    if len(subset) < 2:
        return
    
    trait_vals = subset[trait].dropna()
    if len(trait_vals) < 2:
        return
    
    pkl_dir = base_output_dir / "pkl"
    if not pkl_dir.exists():
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    vmin, vmax = trait_vals.min(), trait_vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    
    n_plotted = 0
    
    for _, row in subset.iterrows():
        tree_id = row["tree_id"]
        trait_val = row[trait]
        
        if pd.isna(trait_val):
            continue
        
        pkl_path = pkl_dir / f"{tree_id}.pkl"
        if not pkl_path.exists():
            continue
        
        try:
            with open(pkl_path, "rb") as f:
                tree = pickle.load(f)
            
            freqs = getattr(tree, "freqs", None)
            tf_db = getattr(tree, "tf_magnitude_db", None)
            
            if freqs is not None and tf_db is not None:
                color = cmap(norm(trait_val))
                ax.plot(freqs, tf_db, color=color, alpha=0.7, linewidth=1.2)
                n_plotted += 1
        except Exception:
            continue
    
    if n_plotted == 0:
        plt.close(fig)
        return
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(trait, fontsize=10)
    
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(FREQ_LOW, FREQ_HIGH)
    ax.set_ylim(-60, 20)
    
    ticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("Amplitude [dB]", fontsize=11)
    
    title_suffix = f" - {leaf_type.capitalize()}" if leaf_type != "mixed" else " - Mixed"
    ax.set_title(f"TFs by {trait}{title_suffix} (n={n_plotted})", 
                 fontsize=13, fontweight="bold")
    
    fig.tight_layout()
    
    save_path = output_dir / f"tfs_{leaf_type}_by_{trait}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  tfs_{leaf_type}_by_{trait}.png ({n_plotted} trees)")


def create_all_figures(df: pd.DataFrame, output_dir: Path):
    """Create all visualization figures."""
    setup_style()
    combined_dir = output_dir / "figures" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating figures in: {combined_dir}")
    plot_trait_correlation_heatmap(df, combined_dir / "trait_correlations.png")
    
    # Predictor-response heatmaps: broadleaf, needleleaf, mixed
    for lt in ["broadleaf", "needleleaf", "mixed"]:
        plot_predictor_response_heatmap(df, lt, combined_dir / f"predictor_response_{lt}.png")
    
    print("\nCreating TF plots...")
    plot_tfs_by_leaf_type(df, combined_dir, output_dir)
    
    # TF plots colored by trait values
    print("\nCreating TF-by-trait plots...")
    traits_to_plot = ALL_PREDICTORS  # Use the filtered predictor list (no precursors)
    for lt in ["broadleaf", "needleleaf", "mixed"]:
        lt_dir = combined_dir / f"tfs_by_trait_{lt}"
        lt_dir.mkdir(parents=True, exist_ok=True)
        for trait in traits_to_plot:
            plot_tfs_by_trait(df, trait, lt, lt_dir, output_dir)


def run_pipeline(output_dir: Path = None, leaf_xlsx: Path = None):
    """Run the complete data preparation and visualization pipeline."""
    if output_dir is None:
        output_dir = OUT_DIR
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("TREE SOUND ABSORPTION - DATA MERGING & CORRELATION ANALYSIS")
    print("=" * 70)
    
    df = load_and_merge_data(output_dir, leaf_xlsx)
    df = standardize_predictors(df)
    
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    merged_path = csv_dir / "merged_data.csv"
    df.to_csv(merged_path, index=False)
    print(f"\nMerged data saved to: {merged_path}")
    
    print("\nChecking collinearity...")
    check_collinearity(df, csv_dir)
    
    create_all_figures(df, output_dir)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nNext step: Run stepwise models in R:")
    print(f"  Rscript run_stepwise_modeling.R --data {merged_path} --output output/figures/models")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Data preparation and correlation analysis")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--leaf-xlsx", type=Path, default=None, 
                        help="Path to leaf morphology XLSX file (required for leaf traits)")
    args = parser.parse_args()
    return run_pipeline(args.output_dir, args.leaf_xlsx)


if __name__ == "__main__":
    main()
