#!/usr/bin/env python3
"""
analyse.py

Full analysis pipeline for tree acoustic data.

Generates PDFs:
- Per-tree TF (distance-only)
- Per-tree TF (median scaling)
- Heatmaps of band attenuation (distance-only + median scaling)

notes:
- Expects per-tree .pkl files to contain 'recording', 'ref_dist', and 'ref_med'
  where ref_dist/ref_med are the *raw* scaled slab.Sound objects produced by
  distance_scale or by applying median scaling (i.e. NOT post-processed by compute_tf).
- Computes transfer functions (deconvolution) on-the-fly from those raw signals
  using compute_tf_from_signals(), which does NOT re-apply distance scaling.
"""

import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pyfar
import slab
import logging

# --- CONFIG ---
DATA_DIR = Path.cwd() / "data"
WINDOW_SIZE = 120  # ms
ROLLING_WINDOW = 5  # bins for smoothing TF
SHOW = False

LOG_IDS = {"227", "257", "247", "277", "499", "502", "332", "274", "232",
           "467", "270", "281", "298", "327", "518", "333", "342"}

# --- Tree mappings ---
species_map_short = {
    "227": "Lar dec", "247": "Pop tre", "257": "Pse men", "277": "Pru avi",
    "499": "Til tom", "502": "Ced deo", "332": "Pin nig", "274": "Pop tre",
    "232": "Pru avi", "467": "Til tom", "281": "Lar dec", "270": "Aln glu",
    "298": "Abi gra", "327": "Sal cap", "518": "Til tom", "342": "Pse men",
    "333": "Pin nig", "313": "Aln glu", "344": "Abi gra", "353": "Sal cap",
}

species_long_map = {
    "Lar dec": "Larix decidua", "Pop tre": "Populus tremula", "Pse men": "Pseudotsuga menziesii",
    "Pru avi": "Prunus avium", "Til tom": "Tilia tomentosa", "Ced deo": "Cedrus deodara",
    "Pin nig": "Pinus nigra", "Aln glu": "Alnus glutinosa", "Abi gra": "Abies grandis",
    "Sal cap": "Salix caprea",
}

needleleaf_set = {"Lar dec", "Pse men", "Pin nig", "Abi gra", "Ced deo"}
broadleaf_set  = {"Pop tre", "Pru avi", "Til tom", "Aln glu", "Sal cap"}

# --- Utilities ---

def parse_distance(foldername: str) -> float:
    try:
        dist_str = foldername.split("_")[1]
        return float(dist_str.replace(",", "."))
    except Exception:
        return None

def get_tree_dirs(base_dir: Path):
    for f in base_dir.iterdir():
        if not f.is_dir():
            continue
        name = f.name
        if name in ["ref", "ref_linear", "rcx"]:
            continue
        if name.endswith("_ref"):
            continue
        if name.startswith("353_") and name != "353_8.1_255W":
            continue
        yield f

def find_first_pkl(folder: Path):
    pkl_all = sorted(folder.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pkl_all[0] if pkl_all else None

def load_tree_pkl(tree_dir: Path):
    pkl_path = find_first_pkl(tree_dir)
    if pkl_path is None:
        logging.warning(f"No .pkl files in {tree_dir}")
        return None
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        logging.warning(f"Unexpected pickle format in {pkl_path}: {type(obj)}")
        return None
    # must contain recording, ref_dist, ref_med
    if not ("recording" in obj and ("ref_dist" in obj or "ref_med" in obj)):
        logging.warning(f"Pickle {pkl_path} missing expected keys (recording/ref_dist/ref_med).")
        return None
    return obj

def get_tree_traits(tree_id: str):
    short = species_map_short.get(tree_id.split("_")[0], "Unknown")
    leaf_type = "needleleaf" if short in needleleaf_set else "broadleaf"
    return {
        "tree_id": tree_id,
        "species_short": short,
        "species_long": species_long_map.get(short, "Unknown"),
        "leaf_type": leaf_type,
    }

# --- New TF computation (no distance_scale inside) ---

def compute_tf_from_signals(recording, reference, window_size=WINDOW_SIZE):
    """
    Compute TFs (raw + windowed) from slab.Sound recording and slab.Sound reference.
    DOES NOT apply distance scaling; assumes 'reference' is already the correctly scaled slab.Sound.
    Returns: raw_tf (slab.Filter), windowed_tf (slab.Filter)
    """
    # convert slab.Sound -> pyfar.Signal (pyfar expects samples x channels; pyfar uses shape (n_samples, n_channels) or (n_channels, n_samples)?)
    # In earlier code we used data.T, so follow that.
    rec_pf = pyfar.Signal(data=recording.data.T, sampling_rate=recording.samplerate)
    ref_pf = pyfar.Signal(data=reference.data.T, sampling_rate=reference.samplerate)

    # Regularized inversion of the reference spectrum
    reference_inv = pyfar.dsp.regularized_spectrum_inversion(ref_pf, frequency_range=(20, 19.75e3))
    ir_deconvolved = rec_pf * reference_inv

    # Window the IR to remove late reflections
    fs = ir_deconvolved.sampling_rate
    win_samples = max(1, int(round(window_size * 1e-3 * fs)))
    ir_windowed = pyfar.dsp.time_window(ir_deconvolved, (0, win_samples), 'boxcar', unit='samples', crop='window')
    ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples)

    # Return magnitude TFs as slab.Filter objects (consistent with previous code)
    raw_mag = np.abs(ir_deconvolved.freq)
    win_mag = np.abs(ir_windowed.freq)
    raw_tf = slab.Filter(data=raw_mag, samplerate=ir_deconvolved.sampling_rate, fir='TF')
    windowed_tf = slab.Filter(data=win_mag, samplerate=ir_deconvolved.sampling_rate, fir='TF')
    return raw_tf, windowed_tf

# --- Plotting ---

def plot_tf_variant(tf_obj, rec, traits, label, pdf):
    """Plot transfer function in dB vs frequency, with log-binned smoothing and mean lines."""
    if tf_obj is None:
        return

    # magnitude spectrum → dB
    mag = np.asarray(getattr(tf_obj, "data", tf_obj)).squeeze()
    tf_db = 20 * np.log10(np.maximum(mag, 1e-12))

    # frequency axis
    if hasattr(tf_obj, "frequencies"):
        freqs = np.asarray(tf_obj.frequencies).squeeze()
    else:
        fs = getattr(rec, "samplerate", None) or 48000
        freqs = np.linspace(0, fs / 2, num=tf_db.size)

    # drop DC bin if 0 Hz
    if freqs[0] == 0:
        freqs, tf_db = freqs[1:], tf_db[1:]

    # --- log-binned smoothing ---
    n_bins = 200
    log_edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_bins)
    smoothed = []
    bin_centers = []
    for lo, hi in zip(log_edges[:-1], log_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        smoothed.append(np.mean(tf_db[mask]))
        bin_centers.append(np.sqrt(lo * hi))  # geometric mean
    smoothed = np.array(smoothed)
    bin_centers = np.array(bin_centers)

    # --- mean attenuation lines ---
    mean_all = np.mean(tf_db)
    low_mask = (freqs < 1000) & (freqs >= 20)
    high_mask = (freqs >= 1000) & (freqs <= 20000)
    mean_low = np.mean(tf_db[low_mask]) if np.any(low_mask) else np.nan
    mean_high = np.mean(tf_db[high_mask]) if np.any(high_mask) else np.nan

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogx(freqs, tf_db, alpha=0.4, label="TF (raw)")
    ax.semilogx(bin_centers, smoothed, color="C1", linewidth=1.5, label="Log-binned mean")

    # global mean (black solid)
    ax.axhline(mean_all, color="black", linewidth=1.2, linestyle="-",
               label=f"Mean all: {mean_all:.1f} dB")

    # low-band mean (<1 kHz)
    if not np.isnan(mean_low):
        ax.hlines(mean_low, 20, 1000, color="green", linewidth=1.2, linestyle="--",
                  label=f"<1 kHz mean: {mean_low:.1f} dB")

    # high-band mean (≥1 kHz)
    if not np.isnan(mean_high):
        ax.hlines(mean_high, 1000, 20000, color="green", linewidth=1.2, linestyle=":",
                  label=f">=1 kHz mean: {mean_high:.1f} dB")

    # axes & labels
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-20, 30)
    ax.axhline(0, color="grey", linewidth=2.0)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title(f"TF — {traits['tree_id']} ({traits['species_short']}) [{label}]")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_tf_heatmap_variant(all_trees, key, pdf, n_bands=20):
    """
    Build heatmap from windowed TFs computed on-the-fly per tree.
    key: 'dist' or 'med' to select which reference to use.
    """
    tree_labels = []
    heatmap_data = []

    for t in all_trees:
        tree_id = t["tree_id"]
        # print(tree_id)
        base_id = tree_id.split("_")[0]
        # skip special short-id folders if you don't want them here
        #if base_id in {"313", "344", "353"}:
        #    continue

        rec = t["data"].get("recording")
        if rec is None:
            continue

        # choose raw reference object
        if key == "dist":
            ref_obj = t["data"].get("ref_dist")
        else:
            ref_obj = t["data"].get("ref_med")
        if ref_obj is None:
            continue

        # compute TFs on the fly
        try:
            raw_tf, windowed_tf = compute_tf_from_signals(rec, ref_obj, window_size=WINDOW_SIZE)
        except Exception as e:
            logging.warning(f"Failed to compute TF for {tree_id} ({key}): {e}")
            continue

        mag = np.asarray(getattr(windowed_tf, "data", windowed_tf)).squeeze()
        tf_db = 20 * np.log10(np.maximum(mag, 1e-12))

        # get freqs from pyfar/slab: windowed_tf.samplerate is sampling rate of TF freq axis
        freqs = np.linspace(0, (getattr(rec, "samplerate", 48000) / 2.0), num=tf_db.size)
        if freqs[0] == 0:
            freqs, tf_db = freqs[1:], tf_db[1:]

        band_edges = np.logspace(np.log10(20), np.log10(20000), n_bands+1)
        band_means = []
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            band_means.append(np.mean(tf_db[mask]) if np.any(mask) else np.nan)
        heatmap_data.append(band_means)
        short_name = species_map_short.get(base_id, base_id)
        tree_labels.append(short_name)

    if not heatmap_data:
        logging.warning("No data for heatmap")
        return

    heatmap_data = np.array(heatmap_data)
    sorted_idx = np.argsort(tree_labels)
    heatmap_data = heatmap_data[sorted_idx]
    tree_labels = [tree_labels[i] for i in sorted_idx]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"{i+1}" for i in range(n_bands)],
        yticklabels=tree_labels,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Mean attenuation [dB]"}
    )
    plt.xlabel("Frequency band")
    plt.ylabel("Tree")
    plt.title(f"Mean attenuation per frequency band ({key})")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# --- Main ---

def main():
    all_trees = []
    for tree_dir in get_tree_dirs(DATA_DIR):
        tree_id = tree_dir.name
        # print(tree_dir)
        if tree_id in ['313', '344', '353']:
            print('skipping', tree_id)
            continue
        tree_data = load_tree_pkl(tree_dir)
        if tree_data is None:
            continue
        traits = get_tree_traits(tree_id)
        traits["data"] = tree_data
        all_trees.append(traits)

    # --- Per-tree TFs (distance-only) ---
    with PdfPages("figures/per_tree_tf_dist.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            ref = t["data"].get("ref_dist")
            if rec is None or ref is None:
                continue
            try:
                raw_tf, windowed_tf = compute_tf_from_signals(rec, ref, window_size=WINDOW_SIZE)
            except Exception as e:
                logging.warning(f"TF compute failed for {t['tree_id']} (dist): {e}")
                continue
            plot_tf_variant(windowed_tf, rec, t, "distance-only", pdf)

    # --- Per-tree TFs (median scaling) ---
    with PdfPages("figures/per_tree_tf_med.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            ref = t["data"].get("ref_med")
            if rec is None or ref is None:
                continue
            try:
                raw_tf, windowed_tf = compute_tf_from_signals(rec, ref, window_size=WINDOW_SIZE)
            except Exception as e:
                logging.warning(f"TF compute failed for {t['tree_id']} (med): {e}")
                continue
            plot_tf_variant(windowed_tf, rec, t, "median-scaled", pdf)

    # --- Heatmaps ---
    with PdfPages("figures/tf_heatmap_dist.pdf") as pdf:
        plot_tf_heatmap_variant(all_trees, "dist", pdf)
    with PdfPages("figures/tf_heatmap_med.pdf") as pdf:
        plot_tf_heatmap_variant(all_trees, "med", pdf)

    print("Analysis completed.")

if __name__ == "__main__":
    main()

exit()
