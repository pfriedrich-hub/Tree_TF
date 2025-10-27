#!/usr/bin/env python3
"""
analyse.py

Full analysis pipeline for tree acoustic data.

Generates PDFs:
- trimming of 0-line
- window exploration with fixed lengths
- windowing with selected length and offset
- Per-tree TF (different variants: ISO-implementation, baselining, RMS scaling)
- Heatmaps of band attenuation (1 selected variant of TFs)

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
from scipy.stats import pearsonr
import seaborn as sns
import pyfar
import slab
import logging
import os

# --- CONFIG ---
DATA_DIR = Path.cwd() / "data"
WINDOW_SIZE = 120  # ms # not needed -> shortened
ROLLING_WINDOW = 5  # bins for smoothing TF # not needed
SHOW = False

LOG_IDS = {
    "227",
    "257",
    "247",
    "277",
    "499",
    "502",
    "332",
    "274",
    "232",
    "467",
    "270",
    "281",
    "298",
    "327",
    "518",
    "333",
    "342",
}

# --- Tree mappings ---
species_map_short = {
    "227": "Lar dec",
    "247": "Pop tre",
    "257": "Pse men",
    "277": "Pru avi",
    "499": "Til tom",
    "502": "Ced deo",
    "332": "Pin nig",
    "274": "Pop tre",
    "232": "Pru avi",
    "467": "Til tom",
    "281": "Lar dec",
    "270": "Aln glu",
    "298": "Abi gra",
    "327": "Sal cap",
    "518": "Til tom",
    "342": "Pse men",
    "333": "Pin nig",
    "313": "Aln glu",
    "344": "Abi gra",
    "353": "Sal cap",
}

species_long_map = {
    "Lar dec": "Larix decidua",
    "Pop tre": "Populus tremula",
    "Pse men": "Pseudotsuga menziesii",
    "Pru avi": "Prunus avium",
    "Til tom": "Tilia tomentosa",
    "Ced deo": "Cedrus deodara",
    "Pin nig": "Pinus nigra",
    "Aln glu": "Alnus glutinosa",
    "Abi gra": "Abies grandis",
    "Sal cap": "Salix caprea",
}

needleleaf_set = {"Lar dec", "Pse men", "Pin nig", "Abi gra", "Ced deo"}
broadleaf_set = {"Pop tre", "Pru avi", "Til tom", "Aln glu", "Sal cap"}

# --- Utilities ---


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
    pkl_all = sorted(
        folder.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
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
        logging.warning(
            f"Pickle {pkl_path} missing expected keys (recording/ref_dist/ref_med)."
        )
        return None
    return obj


def get_tree_traits(
    tree_id: str,
):  # will be updated with structural traits from tls later?
    short = species_map_short.get(tree_id.split("_")[0], "Unknown")
    leaf_type = "needleleaf" if short in needleleaf_set else "broadleaf"
    return {
        "tree_id": tree_id,
        "species_short": short,
        "species_long": species_long_map.get(short, "Unknown"),
        "leaf_type": leaf_type,
    }


# TODO:
# def compute_ISO():


# --- trimming ---
def auto_trim_signal(
    recording, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=False
):
    """
    Trim the silent 0-line at the beginning of a slab.Sound recording.

    Returns:
    - trimmed_recording: new slab.Sound object with silence removed
    - start_idx (optional): index (in samples) of where the trimmed signal begins in the original
    """
    if recording is None:
        return (None, 0) if return_index else None

    data = recording.data
    fs = getattr(recording, "samplerate", 48000)
    arr = np.asarray(data)
    if arr.ndim == 1:
        abs_sig = np.abs(arr)
    else:
        abs_sig = np.mean(np.abs(arr), axis=1)

    peak = np.max(abs_sig) if abs_sig.size > 0 else 0.0
    thresh = threshold_rel * peak

    if peak == 0.0:
        return (recording, 0) if return_index else recording

    # above = np.where(abs_sig > thresh)[0]
    # if above.size == 0:
    #     return (recording, 0) if return_index else recording
    # first = int(above[0])

    above = np.where(abs_sig > thresh)[0]
    if above.size == 0:
        return (recording, 0) if return_index else recording

    # parameters for checking the surrounding area
    fs = getattr(recording, "samplerate", 48000)
    check_window_ms = 1  # how many ms to check around the candidate onset
    check_window = int(round(check_window_ms * 2e-2 * fs))

    valid_start = None
    for idx in above:
        # take a small window before and after the candidate
        lo = max(0, idx - check_window)
        hi = min(len(abs_sig), idx + check_window)
        local_mean = np.mean(abs_sig[lo:hi])

        # If the local average is reasonably strong, treat it as real onset
        if local_mean > 0.5 * thresh:
            valid_start = idx
            break

    # fallback if no valid start found
    if valid_start is None:
        valid_start = int(above[0])

    first = valid_start

    margin_samples = int(round(safety_margin_ms * 2e-2 * fs))
    start_idx = max(0, first - margin_samples)

    if arr.ndim == 1:
        trimmed = arr[start_idx:]
    else:
        trimmed = arr[start_idx:, :]

    try:
        trimmed_sound = slab.Sound(data=trimmed, samplerate=fs)
    except Exception:
        trimmed_sound = recording
        trimmed_sound.data = trimmed

    if return_index:
        return trimmed_sound, start_idx
    return trimmed_sound


def plot_time_overlay_raw_vs_trimmed(
    original_rec, trimmed_rec, traits, pdf, start_idx=0
):
    """
    Overlay the raw and trimmed recordings in the time domain.
    The trimmed recording is plotted with a time offset so it starts at the correct position.
    """
    if original_rec is None or trimmed_rec is None:
        return

    fs = getattr(original_rec, "samplerate", 48000)

    def mono(x):
        arr = np.asarray(x.data)
        return np.mean(arr, axis=1) if arr.ndim > 1 else arr

    sig_raw = mono(original_rec)
    sig_trim = mono(trimmed_rec)

    # full time axes
    t_raw = np.arange(sig_raw.size) / fs
    t_trim = np.arange(sig_trim.size) / fs + (start_idx / fs)  # offset

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_raw, sig_raw, label="raw", alpha=0.6)
    ax.plot(t_trim, sig_trim, label="trimmed", alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Trim check — {traits['tree_id']} ({traits['species_short']})")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_tf_window_grid(t, window_sizes_samples, freq_range):
    """
    Create a column of time–frequency comparison plots (before vs after windowing)
    for one tree.
    Adds an extra plot where the window extends from 0 to onset+window_length.
    Adds a fixed margin before the onset index for all other windows.

    Parameters:
        t: dict with tree data ('recording', 'ref_dist' or 'ref_med', etc.)
        window_sizes_samples: list of window lengths in samples
        freq_range: tuple with speaker frequency range
    """

    rec = t["data"].get("recording")
    ref = t["data"].get("ref_dist") or t["data"].get("ref_med")
    if rec is None or ref is None:
        return None

    # --- Trim both signals ---
    rec_trimmed, start_idx = auto_trim_signal(
        rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
    )
    ref_data = np.asarray(ref.data)
    ref_trimmed_data = (
        ref_data[start_idx:] if ref_data.ndim == 1 else ref_data[start_idx:, :]
    )
    ref_trimmed = slab.Sound(data=ref_trimmed_data, samplerate=ref.samplerate)

    fs = getattr(rec_trimmed, "samplerate", 48000)
    rec_pf = pyfar.Signal(rec_trimmed.data.T, fs)
    ref_pf = pyfar.Signal(ref_trimmed.data.T, fs)

    try:
        ref_inv = pyfar.dsp.regularized_spectrum_inversion(
            ref_pf, frequency_range=(20, 19.75e3)
        )
        ir_deconvolved = rec_pf * ref_inv
    except Exception as e:
        print(f"⚠️ Deconvolution failed for {t['tree_id']}: {e}")
        return None

    # --- Detect onset in first 5000 samples --- # there is a pyfar function which could do this directly, need to try it!
    ir_arr = np.ravel(ir_deconvolved.time)
    abs_ir = np.abs(ir_arr)
    search_window = min(5000, len(abs_ir))
    onset_idx = int(np.argmax(abs_ir[:search_window]))
    print(f"Detected onset: {onset_idx} samples ({onset_idx / fs * 1000:.2f} ms)")

    # --- Setup figure ---
    n_rows = len(window_sizes_samples)
    fig, axes_grid = plt.subplots(n_rows, 1, figsize=(8, 3.8 * n_rows), squeeze=False)

    for i, win_samples in enumerate(window_sizes_samples):
        ax_parent = axes_grid[i, 0]
        gs = ax_parent.get_subplotspec().subgridspec(2, 1, hspace=0.6)
        ax_time = fig.add_subplot(gs[0])
        ax_freq = fig.add_subplot(gs[1])
        ax_parent.set_visible(False)

        # --- Apply window: start at 0, extend to offset + window_size ---
        start_samp = 0
        end_samp = int(start_samp + onset_idx + win_samples)
        win_label = f"0 → offset+{win_samples} samples"

        try:
            ir_windowed = pyfar.dsp.time_window(
                ir_deconvolved,
                (start_samp, end_samp),
                "boxcar",
                unit="samples",
                crop="none",
            )
        except Exception as e:
            print(f"⚠️ Window failed for {t['tree_id']} ({win_label}): {e}")
            continue

        # --- Time–frequency plots ---
        pyfar.plot.time_freq(
            ir_deconvolved,
            unit="samples",
            dB_time=True,
            label="Before Windowing",
            ax=[ax_time, ax_freq],
        )
        pyfar.plot.time_freq(
            ir_windowed,
            unit="samples",
            dB_time=True,
            label="After Windowing",
            ax=[ax_time, ax_freq],
        )

        # --- Axis styling ---
        ax_time.set_xlim(0, len(ir_deconvolved.time[0]))
        ax_freq.set_ylim(-40, 50)
        ax_freq.legend(loc="lower left", fontsize=7)
        ax_freq.axvline(freq_range[0], color="green", ls="--", lw=2)
        ax_freq.axvline(freq_range[1], color="green", ls="--", lw=2)

        # --- Titles ---
        if i == len(window_sizes_samples) - 1:
            title_text = f"Window: {win_label} (full IR visible)"
        else:
            win_ms = win_samples / fs * 1000
            freq_res = fs / win_samples
            title_text = (
                f"Window: {win_label} ({win_ms:.1f} ms) | Δf ≈ {freq_res:.1f} Hz"
            )
        ax_time.set_title(title_text, fontsize=9)

    fig.suptitle(
        f"Tree {t['tree_id']} ({t['species_short']}) — TF window sweep",
        fontsize=11,
    )
    fig.subplots_adjust(wspace=0, hspace=0.3)
    return fig


def get_windowed(t):
    """
    Extract a fixed windowed IR (onset + 2 ms) from a tree's recording.
    Returns full and windowed IR signals.
    TODO: detect window onset & smooth fft with attribute of pyfar
    """
    rec = t["data"].get("recording")
    ref = t["data"].get("ref_dist") or t["data"].get("ref_med")
    if rec is None or ref is None:
        print(f"⚠️ Missing recording or reference for {t['tree_id']}")
        return None

    # --- Trim both signals ---
    rec_trimmed, start_idx = auto_trim_signal(
        rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
    )
    ref_data = np.asarray(ref.data)
    ref_trimmed_data = (
        ref_data[start_idx:] if ref_data.ndim == 1 else ref_data[start_idx:, :]
    )
    ref_trimmed = slab.Sound(data=ref_trimmed_data, samplerate=ref.samplerate)

    fs = getattr(rec_trimmed, "samplerate", 48000)
    rec_pf = pyfar.Signal(rec_trimmed.data.T, fs)
    ref_pf = pyfar.Signal(ref_trimmed.data.T, fs)

    # --- Deconvolution ---
    try:
        ref_inv = pyfar.dsp.regularized_spectrum_inversion(
            ref_pf, frequency_range=(20, 19_750)
        )
        ir_full = rec_pf * ref_inv
    except Exception as e:
        print(f"⚠️ Deconvolution failed for {t['tree_id']}: {e}")
        return None

    # --- Detect onset ---
    ir_arr = np.ravel(ir_full.time)
    abs_ir = np.abs(ir_arr)
    search_window = min(5000, len(abs_ir))
    onset_idx = int(np.argmax(abs_ir[:search_window]))

    # --- Fixed window: onset + 2 ms ---
    win_samples = int(0.002 * fs)
    start_samp = 0
    end_samp = onset_idx + win_samples

    try:
        ir_windowed = pyfar.dsp.time_window(
            ir_full,
            (start_samp, end_samp),
            "boxcar",
            unit="samples",
            crop="none",
        )
    except Exception as e:
        print(f"⚠️ Window failed for {t['tree_id']} ({win_samples} samples): {e}")
        return None

    # print(pyfar.dsp.correlate(ir_windowed, ir_full))

    result = {
        "tree_id": t["tree_id"],
        "fs": fs,
        "window_samples": win_samples,
        "window_ms": win_samples / fs * 1000,
        "signals": {
            "ir_full": ir_full,
            "ir_windowed": ir_windowed,
        },
    }

    print(
        f"✅ Tree {t['tree_id']}: window = {win_samples} samples "
        f"({win_samples / fs * 1000:.2f} ms)"
    )

    return result


def plot_windowed(t, result, freq_range=(74, 20000)):
    """
    Plot impulse response (IR) and transfer function (TF) comparison for a tree.
    Shows full IR and windowed IR using pyfar plotting.

    Parameters
    ----------
    t : dict
        Tree metadata (must contain 'tree_id')
    result : dict
        Output from get_windowed
    freq_range : tuple
        Frequency range to highlight in TF plot
    """
    ir_full = result["signals"]["ir_full"]
    ir_windowed = result["signals"]["ir_windowed"]

    # --- Create figure ---
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 3], hspace=0.4)

    # --- Top: Impulse responses and TF ---
    ax_time = fig.add_subplot(gs[0])
    ax_freq = fig.add_subplot(gs[1])

    pyfar.plot.time_freq(
        ir_full,
        unit="samples",
        dB_time=True,
        label="Full IR",
        ax=[ax_time, ax_freq],
    )
    pyfar.plot.time_freq(
        ir_windowed,
        unit="samples",
        dB_time=True,
        label="Windowed IR",
        ax=[ax_time, ax_freq],
    )

    # --- Styling ---
    ax_time.set_title(
        f"Tree {t['tree_id']} ({t.get('species_short', '')})\n"
        f"Window: {result['window_samples']} samples "
        f"({result['window_ms']:.1f} ms), ",
        # f"Uncertainty = {result['uncertainty']:.3f}",
        fontsize=10,
    )
    ax_time.set_xlabel("Time [samples]")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(frameon=False, fontsize=8)

    ax_freq.set_xscale("log")
    ax_freq.set_xlim(freq_range)
    ax_freq.set_ylim(-40, 50)
    ax_freq.axvline(freq_range[0], color="green", ls="--", lw=1.5)
    ax_freq.axvline(freq_range[1], color="green", ls="--", lw=1.5)
    ax_freq.grid(True, which="both", alpha=0.3)
    ax_freq.legend(frameon=False, fontsize=8)
    ax_freq.set_title("Transfer Function Comparison", fontsize=10)

    fig.tight_layout()
    return fig


# TODO:
# def baseline_tf():


# --- TF computation ---
# TODO: use new window computation: use get_windowed() instead of compute_tf_from_signals() and return both TFs as slab filters (correlation of Filters possible?)
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
    reference_inv = pyfar.dsp.regularized_spectrum_inversion(
        ref_pf, frequency_range=(20, 19.75e3)
    )
    ir_deconvolved = rec_pf * reference_inv  # in frequency domain

    # Window the IR to remove late reflections
    fs = ir_deconvolved.sampling_rate
    win_samples = max(1, int(round(window_size * 1e-3 * fs)))
    ir_windowed = pyfar.dsp.time_window(
        ir_deconvolved, (0, win_samples), "boxcar", unit="samples", crop="window"
    )
    ir_windowed = pyfar.dsp.pad_zeros(
        ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples
    )

    # Return magnitude TFs as slab.Filter objects (consistent with previous code)
    raw_mag = np.abs(ir_deconvolved.freq)
    win_mag = np.abs(ir_windowed.freq)
    raw_tf = slab.Filter(
        data=raw_mag, samplerate=ir_deconvolved.sampling_rate, fir="TF"
    )
    windowed_tf = slab.Filter(
        data=win_mag, samplerate=ir_deconvolved.sampling_rate, fir="TF"
    )
    return raw_tf, windowed_tf


# TODO:
# apply baselining/ISO-implementation before plotting
# annotate frequencies for bands
# log binned smoothing and mean not needed -> based on windowed version
# compare before and after baselining, ISO and RMS?
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
    ax.semilogx(
        bin_centers, smoothed, color="C1", linewidth=1.5, label="Log-binned mean"
    )

    # global mean (black solid)
    ax.axhline(
        mean_all,
        color="black",
        linewidth=1.2,
        linestyle="-",
        label=f"Mean all: {mean_all:.1f} dB",
    )

    # low-band mean (<1 kHz)
    if not np.isnan(mean_low):
        ax.hlines(
            mean_low,
            20,
            1000,
            color="green",
            linewidth=1.2,
            linestyle="--",
            label=f"<1 kHz mean: {mean_low:.1f} dB",
        )

    # high-band mean (≥1 kHz)
    if not np.isnan(mean_high):
        ax.hlines(
            mean_high,
            1000,
            20000,
            color="green",
            linewidth=1.2,
            linestyle=":",
            label=f">=1 kHz mean: {mean_high:.1f} dB",
        )

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


# TODO:
# plot only 1 variant, selection based on TF vs frequency plots
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
        # if base_id in {"313", "344", "353"}:
        #    continue

        rec = t["data"].get("recording")
        # rec = auto_trim_signal(t["data"].get("recording"), threshold_rel=1e-2, safety_margin_ms=1)
        rec, start_idx = auto_trim_signal(
            rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
        )

        if rec is None:
            continue

        # choose raw reference object
        if key == "dist":
            ref_obj = t["data"].get("ref_dist")
        else:
            ref_obj = t["data"].get("ref_med")
        if ref_obj is None:
            continue

        # Trim the reference by the same amount (keep length aligned)
        ref_data = np.asarray(ref_obj.data)
        if ref_data.ndim == 1:
            ref_trimmed_data = ref_data[start_idx:]
        else:
            ref_trimmed_data = ref_data[start_idx:, :]
        ref_obj = slab.Sound(data=ref_trimmed_data, samplerate=ref_obj.samplerate)

        # compute TFs on the fly
        try:
            raw_tf, windowed_tf = compute_tf_from_signals(
                rec, ref_obj, window_size=WINDOW_SIZE
            )
        except Exception as e:
            logging.warning(f"Failed to compute TF for {tree_id} ({key}): {e}")
            continue

        mag = np.asarray(getattr(windowed_tf, "data", windowed_tf)).squeeze()
        tf_db = 20 * np.log10(np.maximum(mag, 1e-12))

        # get freqs from pyfar/slab: windowed_tf.samplerate is sampling rate of TF freq axis
        freqs = np.linspace(
            0, (getattr(rec, "samplerate", 48000) / 2.0), num=tf_db.size
        )
        if freqs[0] == 0:
            freqs, tf_db = freqs[1:], tf_db[1:]

        band_edges = np.logspace(np.log10(20), np.log10(20000), n_bands + 1)
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
        xticklabels=[f"{i + 1}" for i in range(n_bands)],
        yticklabels=tree_labels,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Mean attenuation [dB]"},
    )
    plt.xlabel("Frequency band")
    plt.ylabel("Tree")
    plt.title(f"Mean attenuation per frequency band ({key})")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# --- Main ---


def main():
    # ensure figures dir exists
    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_trees = []
    for tree_dir in get_tree_dirs(DATA_DIR):
        tree_id = tree_dir.name
        # print(tree_dir)
        if tree_id in ["313", "344", "353"]:
            print("skipping", tree_id)
            continue
        tree_data = load_tree_pkl(tree_dir)
        if tree_data is None:
            continue
        traits = get_tree_traits(tree_id)
        traits["data"] = tree_data
        all_trees.append(traits)

    # --- Visual check: raw vs trimmed waveform overlay ---
    with PdfPages("figures/trim_check_overlay.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            if rec is None:
                continue
            # rec_trimmed = auto_trim_signal(rec, threshold_rel=1e-2, safety_margin_ms=1)
            rec_trimmed, start_idx = auto_trim_signal(
                rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
            )

            try:
                # plot_time_overlay_raw_vs_trimmed(rec, rec_trimmed, t, pdf)
                plot_time_overlay_raw_vs_trimmed(
                    rec, rec_trimmed, t, pdf, start_idx=start_idx
                )
            except Exception as e:
                logging.warning(f"Trim overlay failed for {t['tree_id']}: {e}")
                continue

    # === Time–Frequency window sweeps for all trees ===
    pdf_path = Path("figures/tf_window_exploration.pdf")
    pdf_path.parent.mkdir(exist_ok=True)
    window_sizes_samples = (
        120,
        480,
        1200,
        2400,
        4800,
        4800,
    )  # ≈ 2.5, 10, 25, 50, 100 ms at 48kHz
    offset_samples = 0
    freq_range = (74, 20000)

    with PdfPages(pdf_path) as pdf:
        for t in all_trees:
            fig = create_tf_window_grid(t, window_sizes_samples, freq_range)
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
                print(f"✅ Added TF window grid for {t['tree_id']}")
            else:
                print(f"⚠️ Skipped {t['tree_id']} (no data or error)")
    print(f"✅ All tree window sweeps saved to: {pdf_path}")

    # --- TF window optimization for all trees ---
    # TODO: rename: no optimization, use pyfar attributes and modules to detect onset, apply a fiter instead of windowing and smooth fft
    pdf_path = figures_dir / "tf_window_optimization.pdf"
    with PdfPages(pdf_path) as pdf:
        for t in all_trees:
            print(f"--- Optimizing TF window for tree {t['tree_id']} ---")

            result = get_windowed(t)
            if result is None:
                continue
            fig = plot_windowed(t, result)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"✅ All window optimization plots saved to {pdf_path}")

    # PLOTS for evaluation of absorption potential:
    # --- Per-tree TFs (distance-only) --- # TODO: baselining/ISO/RMS scaling, comparison of all versions
    with PdfPages("figures/per_tree_tf_dist.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            ref = t["data"].get("ref_dist")
            if rec is None or ref is None:
                continue
            # trim recording before further processing
            # rec_trimmed = auto_trim_signal(rec, threshold_rel=1e-2, safety_margin_ms=1)
            rec_trimmed, start_idx = auto_trim_signal(
                rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
            )

            # Trim the reference by the same amount (keep length aligned)
            ref_data = np.asarray(ref.data)
            if ref_data.ndim == 1:
                ref_trimmed_data = ref_data[start_idx:]
            else:
                ref_trimmed_data = ref_data[start_idx:, :]
            ref_trimmed = slab.Sound(data=ref_trimmed_data, samplerate=ref.samplerate)

            try:
                raw_tf, windowed_tf = compute_tf_from_signals(
                    rec_trimmed, ref_trimmed, window_size=WINDOW_SIZE
                )
            except Exception as e:
                logging.warning(f"TF compute failed for {t['tree_id']} (dist): {e}")
                continue
            plot_tf_variant(windowed_tf, rec_trimmed, t, "distance-only", pdf)

    # --- Heatmaps --- TODO: choose one variant before plotting
    with PdfPages("figures/tf_heatmap_dist.pdf") as pdf:
        plot_tf_heatmap_variant(all_trees, "dist", pdf)

    print("Analysis completed.")


# TODO: modeling

if __name__ == "__main__":
    main()

# redundant?
#
# def parse_distance(foldername: str) -> float:  # redundant?
#     try:
#         dist_str = foldername.split("_")[1]
#         return float(dist_str.replace(",", "."))
#     except Exception:
#         return None


# def plot_fft_trimmed_recording(trimmed_rec, traits, pdf):  # redundant?
#     """
#     Compute and plot the FFT of the trimmed raw recording in dB.
#     Saves one page per tree into the provided PdfPages object.
#     """
#     if trimmed_rec is None:
#         return

#     fs = getattr(trimmed_rec, "samplerate", 48000)
#     data = np.asarray(trimmed_rec.data)
#     if data.ndim > 1:
#         data = np.mean(data, axis=1)

#     n = data.size
#     fft_vals = np.fft.rfft(data, n=n)
#     freqs = np.fft.rfftfreq(n, d=1.0 / fs)
#     mag = np.abs(fft_vals)
#     db_mag = 20 * np.log10(np.maximum(mag, 1e-12))

#     if freqs[0] == 0:
#         freqs, db_mag = freqs[1:], db_mag[1:]

#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.semilogx(freqs, db_mag, color="C0", lw=1.5)
#     ax.set_xlim(20, 20000)
#     ax.set_xlabel("Frequency [Hz]")
#     ax.set_ylabel("Magnitude [dB]")
#     tree_label = traits.get("tree_id", "unknown")
#     species = traits.get("species_short", "")
#     ax.set_title(f"FFT of trimmed recording — {tree_label} ({species})")
#     fig.tight_layout()
#     pdf.savefig(fig)
#     plt.close(fig)

# --- Per-tree TFs (median scaling) --- not needed anymore
# with PdfPages("figures/per_tree_tf_med.pdf") as pdf:
#     for t in all_trees:
#         rec = t["data"].get("recording")
#         ref = t["data"].get("ref_med")
#         if rec is None or ref is None:
#             continue
#         # trim recording before further processing
#         # rec_trimmed = auto_trim_signal(rec, threshold_rel=1e-2, safety_margin_ms=1)
#         rec_trimmed, start_idx = auto_trim_signal(
#             rec, threshold_rel=2e-2, safety_margin_ms=0.0, return_index=True
#         )

#         # Trim the reference by the same amount (keep length aligned)
#         ref_data = np.asarray(ref.data)
#         if ref_data.ndim == 1:
#             ref_trimmed_data = ref_data[start_idx:]
#         else:
#             ref_trimmed_data = ref_data[start_idx:, :]
#         ref_trimmed = slab.Sound(data=ref_trimmed_data, samplerate=ref.samplerate)

#         try:
#             raw_tf, windowed_tf = compute_tf_from_signals(
#                 rec_trimmed, ref_trimmed, window_size=WINDOW_SIZE
#             )
#         except Exception as e:
#             logging.warning(f"TF compute failed for {t['tree_id']} (med): {e}")
#             continue
#         plot_tf_variant(windowed_tf, rec_trimmed, t, "median-scaled", pdf)

# with PdfPages("figures/tf_heatmap_med.pdf") as pdf:  # not needed
#     plot_tf_heatmap_variant(all_trees, "med", pdf)


# --- FFT of trimmed recordings --- # not needed
# with PdfPages("figures/fft_trimmed_recordings.pdf") as pdf:
#     for t in all_trees:
#         rec = t["data"].get("recording")
#         if rec is None:
#             continue
#         rec_trimmed = auto_trim_signal(
#             rec, threshold_rel=2e-2, safety_margin_ms=0.0
#         )
#         try:
#             plot_fft_trimmed_recording(rec_trimmed, t, pdf)
#         except Exception as e:
#             logging.warning(f"FFT plot failed for {t['tree_id']}: {e}")
#             continue
