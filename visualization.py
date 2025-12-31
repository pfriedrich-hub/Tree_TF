"""
Visualization Module
====================
Plotting functions for tree acoustic analysis.

Design principles:
- All plots saved as PNGs
- Per-tree plots go in subfolders
- Colorblind-friendly palette
- No imports within functions
"""
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyfar
import seaborn as sns

from config import (
    FREQ_HIGH, FREQ_LOW, BAND_LOW, BAND_MID, BAND_HIGH,
    CB_INDIGO, CB_ROSE, CB_PURPLE,
    OUT_FIG_TF, OUT_FIG_IR, OUT_FIG_COMBINED,
)


def setup_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })


def _save_or_show(fig, save_path: Optional[Path] = None, show: bool = False):
    """Save figure as PNG or show interactively."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# TRANSFER FUNCTION PLOT - Main result visualization
# =============================================================================

def plot_transfer_function(
    tree,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot transfer function with band shading and attenuation annotations.
    
    Parameters
    ----------
    tree : Tree
        Tree object with freqs and tf_magnitude_db
    save_path : Path, optional
        Path to save PNG
    show : bool
        Whether to display interactively
    """
    if tree.freqs is None or tree.tf_magnitude_db is None:
        return
    
    freqs = tree.freqs
    tf_db = tree.tf_magnitude_db
    
    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # Remove spines for clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(
        axis="both", which="both",
        color="black", labelcolor="black",
        bottom=True, left=True, labelsize=10
    )
    
    # --- Band shading (subtle) ---
    band_colors = {
        "low": ("#3498db", 0.12),   # blue
        "mid": ("#f39c12", 0.12),   # orange
        "high": ("#9b59b6", 0.12),  # purple
    }
    bands = {
        "low": BAND_LOW,
        "mid": BAND_MID,
        "high": BAND_HIGH,
    }
    
    for band_name, (f_lo, f_hi) in bands.items():
        color, alpha = band_colors[band_name]
        ax.axvspan(f_lo, f_hi, alpha=alpha, color=color, zorder=0)
    
    # --- Main TF curve ---
    ax.plot(freqs, tf_db, color="#CC0077", linewidth=2.2, zorder=2)
    
    # --- Reference line at 0 dB ---
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.7, zorder=1)
    
    # --- Band attenuation annotations ---
    attenuation_vals = {
        "low": tree.attenuation_low,
        "mid": tree.attenuation_mid,
        "high": tree.attenuation_high,
    }
    
    for band_name, (f_lo, f_hi) in bands.items():
        att = attenuation_vals.get(band_name)
        if att is not None:
            f_mid = np.sqrt(f_lo * f_hi)  # geometric mean
            ax.annotate(
                f"{att:.1f} dB",
                xy=(f_mid, -45),
                fontsize=10, fontweight="bold",
                ha="center", va="bottom",
                color=band_colors[band_name][0],
            )
    
    # --- Axis setup ---
    ax.set_xscale("log")
    ax.set_xlim(FREQ_LOW, FREQ_HIGH)
    ax.set_ylim(-50, 10)
    
    # Clean frequency ticks
    ticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    
    ax.set_xlabel("Frequency [Hz]", fontsize=12, color="black", labelpad=8)
    ax.set_ylabel("Amplitude [dB]", fontsize=12, color="black", labelpad=8)
    
    # Title with tree info
    ax.set_title(
        f"{tree.species_short} ({tree.numeric_id}) — {tree.direction}",
        fontsize=13, fontweight="bold", pad=10
    )
    
    # No grid for clean look
    ax.grid(False)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# =============================================================================
# IR COMPARISON PLOT - Raw vs Windowed
# =============================================================================

def plot_ir_comparison(
    tree,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot impulse response before and after windowing.
    Clean minimal style matching the TF plots.
    
    Parameters
    ----------
    tree : Tree
        Tree object with ir_full and ir_windowed
    save_path : Path, optional
        Path to save PNG
    show : bool
        Whether to display interactively
    """
    if tree.ir_full is None or tree.ir_windowed is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    fig.patch.set_facecolor("white")
    
    for ax in axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(color="black", labelcolor="black", bottom=True, left=True)
    
    # --- Time domain ---
    ax = axes[0]
    pyfar.plot.time(tree.ir_full, unit="ms", dB=True, ax=ax, 
                    color="#0088CC", alpha=0.6, linewidth=1.5, label="Full IR")
    pyfar.plot.time(tree.ir_windowed, unit="ms", dB=True, ax=ax,
                    color="#CC0077", linewidth=2.0, label="Windowed IR")
    ax.set_xlim(0, 15)
    ax.set_title(f"Impulse Response — {tree.tree_id}", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(False)
    
    # --- Frequency domain ---
    ax = axes[1]
    pyfar.plot.freq(tree.ir_full, dB=True, ax=ax,
                    color="#0088CC", alpha=0.6, linewidth=1.5, label="Full IR")
    pyfar.plot.freq(tree.ir_windowed, dB=True, ax=ax,
                    color="#CC0077", linewidth=2.0, label="Windowed IR")
    ax.set_xlim(FREQ_LOW, FREQ_HIGH)
    ax.set_ylim(-60, 20)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Transfer Function", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(False)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# =============================================================================
# OVERLAY PLOT - All TFs together
# =============================================================================

def plot_all_tfs_overlay(
    trees: List,
    save_path: Optional[Path] = None,
    show: bool = False,
    color_by: str = "leaf_type",
):
    """
    Overlay all tree transfer functions on one plot.
    
    Parameters
    ----------
    trees : list
        List of Tree objects
    save_path : Path, optional
        Path to save PNG
    show : bool
        Whether to display interactively
    color_by : str
        How to color lines: "leaf_type" or "mixed"
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(color="black", labelcolor="black", bottom=True, left=True)
    
    # Plot each tree
    for tree in trees:
        if tree.freqs is None:
            continue
        
        if color_by == "mixed":
            # Purple for mixed (red + blue = purple)
            color = CB_PURPLE
        else:
            # Rose (reddish) for broadleaf, Indigo (bluish) for needleleaf
            color = CB_INDIGO if tree.is_needleleaf else CB_ROSE
        
        alpha = 0.5
        
        ax.plot(tree.freqs, tree.tf_magnitude_db,
                color=color, alpha=alpha, linewidth=1.0)
    
    # Reference line
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    
    # Legend
    if color_by == "mixed":
        legend_elements = [
            Line2D([0], [0], color=CB_PURPLE, linewidth=2, label="All Trees"),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color=CB_INDIGO, linewidth=2, label="Needleleaf"),
            Line2D([0], [0], color=CB_ROSE, linewidth=2, label="Broadleaf"),
        ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=10)
    
    ax.set_xscale("log")
    ax.set_xlim(FREQ_LOW, FREQ_HIGH)
    ax.set_ylim(-60, 20)
    
    ticks = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    
    ax.set_xlabel("Frequency [Hz]", fontsize=12)
    ax.set_ylabel("Amplitude [dB]", fontsize=12)
    
    title = "All Transfer Functions — Mixed" if color_by == "mixed" else "All Transfer Functions — By Leaf Type"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(False)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# =============================================================================
# HEATMAP - Attenuation across frequency bands
# =============================================================================

def plot_attenuation_heatmap(
    trees: List,
    n_bands: int = 20,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    Heatmap of attenuation across frequency bands for all trees.
    Trees sorted by species for easier comparison.
    
    Parameters
    ----------
    trees : list
        List of Tree objects
    n_bands : int
        Number of frequency bands
    save_path : Path, optional
        Path to save PNG
    show : bool
        Whether to display interactively
    """
    tree_labels = []
    heatmap_data = []
    band_edges = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), n_bands + 1)
    
    for tree in trees:
        if tree.freqs is None or tree.tf_magnitude_db is None:
            continue
        
        freqs = tree.freqs
        tf_db = tree.tf_magnitude_db
        
        band_means = []
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_means.append(np.mean(tf_db[mask]))
            else:
                band_means.append(np.nan)
        
        heatmap_data.append(band_means)
        tree_labels.append(f"{tree.species_short} ({tree.numeric_id})")
    
    if not heatmap_data:
        return
    
    heatmap_data = np.array(heatmap_data)
    
    # Sort by species
    sorted_idx = np.argsort(tree_labels)
    heatmap_data = heatmap_data[sorted_idx]
    tree_labels = [tree_labels[i] for i in sorted_idx]
    
    # Band labels (simplified)
    band_labels = [f"{int(lo)}" for lo in band_edges[:-1]]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    
    sns.heatmap(
        heatmap_data,
        xticklabels=band_labels,
        yticklabels=tree_labels,
        cmap="RdBu_r",
        center=0,
        vmin=-40, vmax=10,
        cbar_kws={"label": "Attenuation [dB]"},
        ax=ax,
    )
    
    ax.set_xlabel("Frequency [Hz]", fontsize=11)
    ax.set_ylabel("Tree", fontsize=11)
    ax.set_title("Attenuation Heatmap", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    
    fig.tight_layout()
    _save_or_show(fig, save_path, show)


# =============================================================================
# MAIN REPORT FUNCTION - Creates all figures
# =============================================================================

def create_all_figures(trees: List, output_dir: Path):
    """
    Create all output figures as PNGs.
    
    Output structure:
        output_dir/
        ├── transfer_functions/
        │   ├── 270_5.4_240SW_tf.png
        │   └── ...
        ├── ir_comparison/
        │   ├── 270_5.4_240SW_ir.png
        │   └── ...
        ├── combined/
        │   ├── all_tfs_by_leaf_type.png
        │   └── all_tfs_mixed.png
        └── attenuation_heatmap.png
    
    Parameters
    ----------
    trees : list
        List of Tree objects
    output_dir : Path
        Base output directory for figures
    """
    setup_style()
    output_dir = Path(output_dir)
    
    # Create subdirectories
    tf_dir = output_dir / "transfer_functions"
    ir_dir = output_dir / "ir_comparison"
    combined_dir = output_dir / "combined"
    tf_dir.mkdir(parents=True, exist_ok=True)
    ir_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating figures in: {output_dir}")
    
    # Per-tree plots
    for tree in trees:
        if tree.freqs is not None:
            plot_transfer_function(tree, save_path=tf_dir / f"{tree.tree_id}_tf.png")
        
        if tree.ir_full is not None and tree.ir_windowed is not None:
            plot_ir_comparison(tree, save_path=ir_dir / f"{tree.tree_id}_ir.png")
    
    # TF overlay by leaf type
    plot_all_tfs_overlay(trees, save_path=combined_dir / "all_tfs_by_leaf_type.png", 
                         color_by="leaf_type")
    
    # TF overlay mixed (for statistical analysis with combined data)
    plot_all_tfs_overlay(trees, save_path=combined_dir / "all_tfs_mixed.png",
                         color_by="mixed")
    
    # Attenuation heatmap goes in main figures folder
    plot_attenuation_heatmap(trees, save_path=output_dir / "attenuation_heatmap.png")
    
    print(f"  Transfer functions: {tf_dir}/")
    print(f"  IR comparisons: {ir_dir}/")
    print(f"  Combined plots: {combined_dir}/")
    print(f"  Attenuation heatmap: attenuation_heatmap.png")
