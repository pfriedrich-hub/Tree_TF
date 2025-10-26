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

"""
TODO:
    1. eliminate unrealistic low frequencies (no sound from speaker or not relevant in anthropospheres)
        - research: meaning of the window size: noteboook on git
            notes:
                - window size not in ms but in samples
                - need to try different values and see what happens in the frequency domain
        - research: relevant frequencies in anthropospheres
            notes:
                - there are studies suggesting highest noise pollution in cities from low frequencies: https://www.sciencedirect.com/science/article/pii/S0048969721006689
                - others start at 100Hz: https://doi.org/10.1177/1351010X16678218
                - or slightly below 100Hz: https://www.mdpi.com/1424-8220/23/4/1912
                - low frequency noise seems to be less healthy: https://www.thieme-connect.de/products/ejournals/abstract/10.1055/s-0043-1769497
                - so cut the frequencies according to the speaker, not human made sounds
                - idea: separate frequency bands according to typical sources and map tree per source
        - cut 0-line of raw recording (automatization code), compare fft of raw recording with and without 0-line, also look at per-tree TFs with distance and median scaling and heatmaps of band attenuation
        - play with window size according to notebook and compare tfs (median/distance scaled, heatmaps of band attenuation)
        - find optimal window size for the TF starting at 100 Hz/realistic frequency, look at tf (distance/median scaling and heatmap of band attenuation)
        - next steps for finding the window:
            - understand window, ripples in frequency domain, comb filter (last prompt chatty):
                - what causes the 'comb' effect: reflections return the direct sound with a delay, the sound is made of waves at multiple frequencies, so the delay returns some frequencies in phase (amplification), some out of phase (canceling) and some in between (attenuation)
                - Once you’ve deconvolved, you’re no longer looking at the sweep; you’re looking at the impulse.
                - All the information from the whole sweep has been time-compressed into a short burst.
                - The impulse response (IR) is the time-domain fingerprint of the tree
                - A short IR sounds like a filtered “click” (showing the tonal color of the tree).
                - IR is a filter that describes how the tree modifies each frequency component of the sound that passes through
                - IR: how the tree colors or filters sound
                - if we would play white noise through the tree, then we would hear the tone of the tree?
                - the IR is the pattern of echos which the tree generates for all frequencies at once,
                - the IR is the pattern that defines how all frequencies interact in time
                - The tree’s IR shows the timing and strength of acoustic interactions (direct path, leaf scattering, trunk reflection).
                - The tree’s transfer function (FFT of IR) shows which frequencies are transmitted, absorbed, or cancelled due to those interactions.
                ! automatize finding the window length based on window_grid function but in a new function
                ! tradeoff between accuracy in low frequencies and 'noise'/high variability in high frequencies
                ! optimize window length between 1-30ms, optimize according to low/high frequencies in TF
                - is it reasonable to use the full IR for the lower frequencies and the windowed for the TF of the higher frequencies or a dynamic window length,
                so that each frequency bin's amplitude is taken from a similar neighbourhood of frequencies?
                multiresolution approach? long window + smoothing at higher frequencies?
                - infer an uncertainty metric for the chosen window length per tree
            - implement and check optimization code (chatty)
            - implement correct specs of the speaker
            - push to git
    2. fix reference scaling/ampification of low frequencies:
        - research: ISO-implementation-code
        - research: multiplication in frequency domain convolution in time domain, so is the deconvolution done by a scaling in frequency domain? is this the trick?
        - try baselining of lowest frequency in frequency domain before or after deconvolution and look at tfs with different scalings and heatmaps
    3. minor fixes:
        - root mean square (RMS) instead of median scaling
        - annotate real frequencies in bands on x-axis
        - octave scaling instead of logarithmic
    4. summary;
        - TLS per tree from CloudCompare (just a picture) with important information:
            - uncertainty of window length
            - mean attenuation
            - frequency range of highest attenuation
            - sound of each tree (play the same sound with the tree filter...or before tree and after tree? the same sweep?)
        - print this summary
    4. TLS-Scans:
        - structural/architectonical traits of every tree
        - which parameters are most relevant for acoustics?
        - add parameters from extra data Arboretum
        - try to estimate a pore-size per tree?
    5. model absorption potential predicted by tree traits -> look into satellite data and find silent places from tree traits
    6. write protocol:
        - motivation to think on my own because of low frequency of intense meetings
        - possibility to apply tools in a free and self-responsible way: best experience of the whole masters program
        - exploring things at my preferred tempo without the need to proof anything to anyone was an experience that i really needed
    7. look with Paul at window selection plots (tf_window_exploration)
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
WINDOW_SIZE = 120  # ms
ROLLING_WINDOW = 5  # bins for smoothing TF
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


# --- New functions: trimming + FFT comparison ---


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


def plot_fft_trimmed_recording(trimmed_rec, traits, pdf):
    """
    Compute and plot the FFT of the trimmed raw recording in dB.
    Saves one page per tree into the provided PdfPages object.
    """
    if trimmed_rec is None:
        return

    fs = getattr(trimmed_rec, "samplerate", 48000)
    data = np.asarray(trimmed_rec.data)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    n = data.size
    fft_vals = np.fft.rfft(data, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(fft_vals)
    db_mag = 20 * np.log10(np.maximum(mag, 1e-12))

    if freqs[0] == 0:
        freqs, db_mag = freqs[1:], db_mag[1:]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(freqs, db_mag, color="C0", lw=1.5)
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    tree_label = traits.get("tree_id", "unknown")
    species = traits.get("species_short", "")
    ax.set_title(f"FFT of trimmed recording — {tree_label} ({species})")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


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


def create_tf_window_grid(
    t, window_sizes_samples, offset_samples, freq_range, pre_offset_samples=100
):
    """
    Create a column of time–frequency comparison plots (before vs after windowing)
    for one tree.
    Adds an extra plot where the window extends from 0 to onset+window_length.
    Adds a fixed margin before the onset index for all other windows.

    Parameters:
        t: dict with tree data ('recording', 'ref_dist' or 'ref_med', etc.)
        window_sizes_samples: list of window lengths in samples
        offset_samples: not used here but kept for compatibility
        freq_range: tuple with speaker frequency range
        pre_offset_samples: number of samples to show before onset in windowed IR
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

    # --- Detect onset in first 5000 samples ---
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


def find_optimal_window_length(
    t,
    freq_range=(100, 19000),
    alpha=0.3,
):
    """
    Automatically find the optimal window length (in samples) per tree
    by balancing spectral fidelity (correlation to full IR) and
    high-frequency smoothness.

    Parameters
    ----------
    t : dict
        Tree dictionary with entries 'data' -> {'recording', 'ref_dist' or 'ref_med'}
    freq_range : tuple
        Frequency range (Hz) to evaluate the TF.
    alpha : float
        Weighting factor for penalizing high-frequency variability.

    Returns
    -------
    result : dict
        Includes best window info, uncertainty, diagnostics, and signals.
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
            ref_pf, frequency_range=(20, 19.75e3)
        )
        ir_deconvolved = rec_pf * ref_inv
    except Exception as e:
        print(f"⚠️ Deconvolution failed for {t['tree_id']}: {e}")
        return None

    # --- Detect onset ---
    ir_arr = np.ravel(ir_deconvolved.time)
    abs_ir = np.abs(ir_arr)
    search_window = min(5000, len(abs_ir))
    onset_idx = int(np.argmax(abs_ir[:search_window]))

    # --- Candidate window lengths: 0.5–2 ms in samples ---
    window_lengths_samples = np.arange(
        int(fs * 0.0005), int(fs * 0.002) + 1, int(fs * 0.001)
    )
    ###### ✅ LOG-BINNED SMOOTHING (for full IR)
    # --- Compute magnitude TF for full IR ---
    full_tf_db = 20 * np.log10(np.abs(ir_deconvolved.freq).flatten() + 1e-12)
    freqs = np.linspace(0, fs / 2, len(full_tf_db))
    valid_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_valid = freqs[valid_mask]
    tf_valid = full_tf_db[valid_mask]

    # --- Log-binned smoothing ---
    n_log_bins = 200
    log_edges = np.logspace(
        np.log10(freqs_valid[0]), np.log10(freqs_valid[-1]), n_log_bins + 1
    )
    smoothed_full, bin_centers = [], []
    for lo, hi in zip(log_edges[:-1], log_edges[1:]):
        mask = (freqs_valid >= lo) & (freqs_valid < hi)
        if not np.any(mask):
            continue
        smoothed_full.append(np.mean(tf_valid[mask]))
        bin_centers.append(np.sqrt(lo * hi))  # geometric mean = log midpoint
    smoothed_full = np.array(smoothed_full)
    bin_centers = np.array(bin_centers)
    smoothed_full -= np.mean(smoothed_full)

    # --- Lists for diagnostics ---
    r_low_arr, r_high_arr, var_high_arr, score_arr = [], [], [], []

    for win_samples in window_lengths_samples:
        start_samp = 0
        end_samp = int(onset_idx + win_samples)

        try:
            ir_windowed = pyfar.dsp.time_window(
                ir_deconvolved,
                (start_samp, end_samp),
                "boxcar",
                unit="samples",
                crop="none",
            )
        except Exception as e:
            print(f"⚠️ Window failed for {t['tree_id']} ({win_samples} samples): {e}")
            continue

        # --- Windowed TF ---
        win_tf_db = 20 * np.log10(np.abs(ir_windowed.freq).flatten() + 1e-12)
        win_tf_valid = win_tf_db[valid_mask]

        # --- Log-binned smoothing (same bins as full TF) ---
        smoothed_win = []
        for lo, hi in zip(log_edges[:-1], log_edges[1:]):
            mask = (freqs_valid >= lo) & (freqs_valid < hi)
            if not np.any(mask):
                continue
            smoothed_win.append(np.mean(win_tf_valid[mask]))
        smoothed_win = np.array(smoothed_win)
        smoothed_win -= np.mean(smoothed_win)

        # --- Split low/high frequencies ---
        split_idx = len(smoothed_full) // 2
        win_low, win_high = smoothed_win[:split_idx], smoothed_win[split_idx:]
        full_low, full_high = smoothed_full[:split_idx], smoothed_full[split_idx:]

        # --- Correlations ---
        r_low = (
            pearsonr(full_low, win_low)[0] if np.all(np.isfinite(win_low)) else np.nan
        )
        r_high = (
            pearsonr(full_high, win_high)[0]
            if np.all(np.isfinite(win_high))
            else np.nan
        )

        # --- High-frequency variance (consistently log-binned) ---
        var_high = (
            np.var(win_high - np.mean(win_high))
            if np.all(np.isfinite(win_high))
            else np.nan
        )

        # --- Store diagnostics ---
        r_low_arr.append(r_low)
        r_high_arr.append(r_high)
        var_high_arr.append(var_high)
        # (score_arr will be computed later as before)

        # --- Combined score ---
        r_combined = np.nanmean(
            [r_low, r_high]
        )  # more weight on correlation to high frequencies?
        score_arr.append(r_combined - alpha * var_high)

    r_low_arr = np.array(r_low_arr)
    r_high_arr = np.array(r_high_arr)
    var_high_arr = np.array(var_high_arr)
    score_arr = np.array(score_arr)

    # --- Choose best window ---
    best_idx = np.nanargmax(score_arr)
    best_win = window_lengths_samples[best_idx]
    uncertainty = 1 - np.nanmean([r_low_arr[best_idx], r_high_arr[best_idx]])

    # --- Best IR ---
    ir_best = pyfar.dsp.time_window(
        ir_deconvolved,
        (0, int(onset_idx + best_win)),
        "boxcar",
        unit="samples",
        crop="none",
    )

    result = {
        "tree_id": t["tree_id"],
        "fs": fs,
        "best_window_samples": int(best_win),
        "best_window_ms": best_win / fs * 1000,
        "best_score": float(score_arr[best_idx]),
        "uncertainty": float(uncertainty),
        "diagnostics": {
            "window_lengths": window_lengths_samples,
            "r_low_log": r_low_arr,
            "r_high_log": r_high_arr,
            "var_highfreq": var_high_arr,
            "score": score_arr,
        },
        "signals": {
            "ir_full": ir_deconvolved,
            "ir_best": ir_best,
            "ir_smoothed": smoothed_full,
            "log_freqs": bin_centers,
        },
    }

    print(
        f"✅ Tree {t['tree_id']}: best window = {best_win} samples "
        f"({best_win / fs * 1000:.2f} ms), uncertainty = {uncertainty:.3f}"
    )

    return result


def find_optimal_window_length_old(
    t,
    freq_range=(100, 19000),
    alpha=0.5,
    low_cutoff_hz=2000.0,
):
    """
    Find optimal window length (samples) per tree.

    Improvements over prior version:
      - compute r_allfreq and r_low (low-frequency correlation)
      - compute var_high (HF std)
      - normalize var_high across the tested windows before combining
      - score = r_low - alpha * var_high_norm
      - uncertainty = 1 - r_all_best (global-frequency uncertainty)

    Returns the same result dict shape as before, but with extra diagnostics:
      'r_low', 'r_all', 'var_high', 'var_high_norm', 'score'
    """
    rec = t["data"].get("recording")
    ref = t["data"].get("ref_dist") or t["data"].get("ref_med")
    if rec is None or ref is None:
        print(f"⚠️ Missing recording or reference for {t.get('tree_id', '?')}")
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
            ref_pf, frequency_range=(20, 19.75e3)
        )
        ir_deconvolved = rec_pf * ref_inv
    except Exception as e:
        print(f"⚠️ Deconvolution failed for {t.get('tree_id', '?')}: {e}")
        return None

    # --- Detect onset ---
    ir_arr = np.ravel(ir_deconvolved.time)
    abs_ir = np.abs(ir_arr)
    search_window = min(5000, len(abs_ir))
    onset_idx = int(np.argmax(abs_ir[:search_window]))
    print(f"Detected onset: {onset_idx} samples ({onset_idx / fs * 1000:.2f} ms)")

    # --- Candidate window lengths: 1–30 ms in samples (step 1 ms) ---
    ms_step_samples = max(1, int(round(0.001 * fs)))
    min_samples = int(round(0.001 * fs))
    max_samples = int(round(0.030 * fs))
    window_lengths_samples = np.arange(
        min_samples, max_samples + 1, ms_step_samples, dtype=int
    )

    # --- Full TF magnitude (consistent with compute_tf_from_signals) ---
    try:
        full_mag = np.abs(ir_deconvolved.freq)
    except Exception as e:
        print(f"⚠️ Couldn't access full TF magnitudes for {t.get('tree_id', '?')}: {e}")
        return None
    full_mag = np.asarray(full_mag).squeeze()
    if full_mag.ndim > 1:
        full_mag = np.mean(full_mag, axis=1)
    full_tf_db = 20.0 * np.log10(np.maximum(full_mag, 1e-12))

    # frequency axis (bins)
    n_bins = full_tf_db.shape[0]
    freqs = np.linspace(0.0, fs / 2.0, n_bins)

    valid_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if valid_mask.sum() == 0:
        print(
            f"⚠️ No frequency bins in freq_range {freq_range} for {t.get('tree_id', '?')}"
        )
        return None
    full_tf_valid = full_tf_db[valid_mask]

    # low/high masks (for focused metrics)
    low_mask = freqs <= low_cutoff_hz
    # ensure we intersect with overall valid_mask when computing r_low
    low_mask_valid = valid_mask & low_mask
    high_mask = freqs > ((freq_range[0] + freq_range[1]) / 2.0)
    high_mask_valid = valid_mask & high_mask

    # diagnostics arrays
    r_all_arr = []
    r_low_arr = []
    var_high_arr = []
    score_arr = []

    for win_samples in window_lengths_samples:
        start_samp = 0
        end_samp = int(onset_idx + win_samples)

        try:
            ir_windowed = pyfar.dsp.time_window(
                ir_deconvolved,
                (start_samp, end_samp),
                "boxcar",
                unit="samples",
                crop="none",
            )
        except Exception as e:
            # keep NaNs for failed windows
            r_all_arr.append(np.nan)
            r_low_arr.append(np.nan)
            var_high_arr.append(np.nan)
            score_arr.append(np.nan)
            continue

        # TF magnitude for windowed IR
        try:
            win_mag = np.abs(ir_windowed.freq).squeeze()
            if win_mag.ndim > 1:
                win_mag = np.mean(win_mag, axis=1)
            win_tf_db = 20.0 * np.log10(np.maximum(win_mag, 1e-12))
        except Exception as e:
            r_all_arr.append(np.nan)
            r_low_arr.append(np.nan)
            var_high_arr.append(np.nan)
            score_arr.append(np.nan)
            continue

        # valid portions
        win_tf_valid = win_tf_db[valid_mask]

        # global correlation (all frequencies in valid_mask)
        if np.all(np.isfinite(win_tf_valid)) and np.all(np.isfinite(full_tf_valid)):
            try:
                r_all, _ = pearsonr(full_tf_valid, win_tf_valid)
            except Exception:
                r_all = np.nan
        else:
            r_all = np.nan
        r_all_arr.append(r_all)

        # low-frequency correlation (focus on preserving LF content)
        if (
            low_mask_valid.sum() > 1
            and np.all(np.isfinite(full_tf_db[low_mask_valid]))
            and np.all(np.isfinite(win_tf_db[low_mask_valid]))
        ):
            try:
                r_low, _ = pearsonr(
                    full_tf_db[low_mask_valid], win_tf_db[low_mask_valid]
                )
            except Exception:
                r_low = np.nan
        else:
            r_low = np.nan
        r_low_arr.append(r_low)

        # high-frequency variability (noise/wiggliness) measured as std in HF band
        if high_mask_valid.sum() > 0:
            var_h = float(np.nanstd(win_tf_db[high_mask_valid]))
        else:
            var_h = np.nan
        var_high_arr.append(var_h)

        # placeholder for score; will normalize var_high after loop
        score_arr.append(np.nan)

    r_all_arr = np.array(r_all_arr, dtype=float)
    r_low_arr = np.array(r_low_arr, dtype=float)
    var_high_arr = np.array(var_high_arr, dtype=float)

    # Normalize var_high to [0,1] across valid windows (lower -> better)
    finite_var = var_high_arr[np.isfinite(var_high_arr)]
    if finite_var.size == 0:
        var_high_norm = np.full_like(var_high_arr, np.nan)
    else:
        vmin = finite_var.min()
        vmax = finite_var.max()
        denom = vmax - vmin if vmax > vmin else 1.0
        var_high_norm = (var_high_arr - vmin) / denom
        # clip to [0,1]
        var_high_norm = np.clip(var_high_norm, 0.0, 1.0)

    # Compose score: maximize r_low while minimizing normalized HF variability
    # score = r_low - alpha * var_high_norm
    score_arr = np.where(np.isfinite(r_low_arr), r_low_arr, -np.inf) - alpha * np.where(
        np.isfinite(var_high_norm), var_high_norm, 1.0
    )

    # Pick best window
    safe_scores = np.where(np.isfinite(score_arr), score_arr, -np.inf)
    if np.all(~np.isfinite(safe_scores)):
        print(f"⚠️ No valid scores for {t.get('tree_id', '?')}")
        return None
    best_idx = int(np.nanargmax(safe_scores))
    best_win = int(window_lengths_samples[best_idx])

    best_r_all = (
        float(r_all_arr[best_idx]) if np.isfinite(r_all_arr[best_idx]) else np.nan
    )
    uncertainty = float(1.0 - best_r_all) if np.isfinite(best_r_all) else np.nan

    # Best IR windowed signal
    ir_best = pyfar.dsp.time_window(
        ir_deconvolved,
        (0, int(onset_idx + best_win)),
        "boxcar",
        unit="samples",
        crop="none",
    )

    result = {
        "tree_id": t.get("tree_id", None),
        "fs": fs,
        "onset_idx": onset_idx,
        "best_window_samples": int(best_win),
        "best_window_ms": best_win / fs * 1000.0,
        "best_score": float(safe_scores[best_idx]),
        "uncertainty": float(uncertainty),
        "diagnostics": {
            "window_lengths": window_lengths_samples,
            "r_allfreq": r_all_arr,
            "r_low": r_low_arr,
            "var_highfreq": var_high_arr,
            "var_high_norm": var_high_norm,
            "score": score_arr,
            "freqs": freqs,
            "valid_mask": valid_mask,
        },
        "signals": {
            "ir_full": ir_deconvolved,
            "ir_best": ir_best,
        },
    }

    print(
        f"✅ Tree {t.get('tree_id', '?')}: best window = {best_win} samples ({best_win / fs * 1000:.2f} ms), "
        f"uncertainty = {uncertainty:.3f}"
    )
    return result


def plot_window_optimization_old(t, result, freq_range=(100, 19000)):
    """
    One-page diagnostic plot per tree:
      - Top: time–frequency comparison (full vs best windowed IR)
      - Bottom: optimization metrics
    """

    ir_full = result["signals"]["ir_full"]
    ir_best = result["signals"]["ir_best"]
    fs = result["fs"]

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 1], hspace=0.5)

    # --- Top panels: time–frequency ---
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
        ir_best,
        unit="samples",
        dB_time=True,
        label="Best windowed IR",
        ax=[ax_time, ax_freq],
    )

    ax_time.set_xlim(0, len(ir_full.time[0]))
    ax_freq.set_ylim(-40, 50)
    ax_freq.legend(loc="lower left", fontsize=7)
    ax_freq.axvline(freq_range[0], color="green", ls="--", lw=2)
    ax_freq.axvline(freq_range[1], color="green", ls="--", lw=2)

    title_text = (
        f"Tree {t['tree_id']} ({t.get('species_short', '')})\n"
        f"Best window: {result['best_window_samples']} samples "
        f"({result['best_window_ms']:.1f} ms), "
        f"Uncertainty = {result['uncertainty']:.3f}"
    )
    ax_time.set_title(title_text, fontsize=10)

    # --- Bottom panel: metrics ---
    ax_diag = fig.add_subplot(gs[2])
    wl = result["diagnostics"]["window_lengths"]

    ax_diag.plot(wl, result["diagnostics"]["r_low_log"], label="r_low_log", marker="o")
    ax_diag.plot(
        wl, result["diagnostics"]["r_high_log"], label="r_high_log", marker="^"
    )
    ax_diag.plot(
        wl, result["diagnostics"]["var_highfreq"], label="var_highfreq", marker="x"
    )
    ax_diag.plot(wl, result["diagnostics"]["score"], label="combined_score", marker="s")
    ax_diag.axvline(result["best_window_samples"], color="red", lw=1.5, ls="--")

    ax_diag.set_xlabel("Window length (samples)")
    ax_diag.set_ylabel("Metric value")
    ax_diag.legend(fontsize=8)
    ax_diag.grid(True, ls="--", alpha=0.3)

    fig.suptitle(f"Window Optimization Diagnostics — Tree {t['tree_id']}", fontsize=12)
    fig.tight_layout()

    return fig


def plot_window_optimization_odd(t, result, freq_range=(100, 19000), pdf=None):
    """
    Plot diagnostics for optimal window length:
        - top: time–frequency before/after windowing
        - middle: log-binned smoothed TFs + mean TF
        - bottom: metrics: low/high frequency correlations, high-freq variance, combined score
    """

    ir_full = result["signals"]["ir_full"]
    ir_best = result["signals"]["ir_best"]
    fs = result["fs"]

    # --- Create figure ---
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1], hspace=0.4)

    # --- Top: TF comparison (full vs best windowed) ---
    ax_top = fig.add_subplot(gs[0])
    pyfar.plot.time_freq(
        ir_full, unit="samples", dB_time=True, label="Full IR", ax=[ax_top]
    )
    pyfar.plot.time_freq(
        ir_best, unit="samples", dB_time=True, label="Best windowed IR", ax=[ax_top]
    )
    ax_top.set_title(
        f"Tree {t['tree_id']} ({t['species_short']}) — Time-Frequency IR", fontsize=10
    )

    # --- Middle: log-binned smoothed TF + mean ---
    ax_mid = fig.add_subplot(gs[1])

    # compute raw magnitude spectra
    tf_full = np.abs(ir_full.freq)  # shape: channels x samples
    tf_best = np.abs(ir_best.freq)

    # convert to dB
    tf_full_db = 20 * np.log10(np.maximum(tf_full, 1e-12))
    tf_best_db = 20 * np.log10(np.maximum(tf_best, 1e-12))

    # frequency axis
    n_samples = tf_full_db.shape[1]
    freqs = np.linspace(0, fs / 2, n_samples)
    if freqs[0] == 0:
        freqs, tf_full_db, tf_best_db = freqs[1:], tf_full_db[:, 1:], tf_best_db[:, 1:]

    # log-binned smoothing
    n_bins = 200
    log_edges = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), n_bins)
    smoothed_full = []
    smoothed_best = []
    bin_centers = []

    for lo, hi in zip(log_edges[:-1], log_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            continue
        # collapse channel dimension
        smoothed_full.append(np.mean(tf_full_db[:, mask], axis=None))
        smoothed_best.append(np.mean(tf_best_db[:, mask], axis=None))
        bin_centers.append(np.sqrt(lo * hi))

    smoothed_full = np.array(smoothed_full)
    smoothed_best = np.array(smoothed_best)
    bin_centers = np.array(bin_centers)

    # plot smoothed TFs
    ax_mid.plot(
        bin_centers, smoothed_full, label="Full IR (mean, log-binned)", color="blue"
    )
    ax_mid.plot(
        bin_centers,
        smoothed_best,
        label="Best windowed IR (mean, log-binned)",
        color="orange",
    )
    ax_mid.set_xscale("log")
    ax_mid.set_xlim(freq_range)
    ax_mid.set_ylabel("Magnitude (dB)")
    ax_mid.set_xlabel("Frequency (Hz)")
    ax_mid.set_title("Log-binned Smoothed TF Comparison")
    ax_mid.legend(fontsize=8)
    ax_mid.grid(True, ls="--", alpha=0.3)

    # --- Bottom: metrics ---
    ax_bot = fig.add_subplot(gs[2])
    diag = result["diagnostics"]
    wl = diag["window_lengths"]
    ax_bot.plot(wl, diag["r_low_log"], label="r_low_log", marker="o")
    ax_bot.plot(wl, diag["r_high_log"], label="r_high_log", marker="x")
    ax_bot.plot(wl, diag["var_highfreq"], label="var_highfreq", marker="s")
    ax_bot.plot(wl, diag["score"], label="score", marker="^")
    ax_bot.axvline(result["best_window_samples"], color="red", lw=1.5, ls="--")
    ax_bot.set_xlabel("Window length (samples)")
    ax_bot.set_ylabel("Metric value")
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, ls="--", alpha=0.3)
    ax_bot.set_title("Window Optimization Metrics")

    fig.suptitle(f"Tree {t['tree_id']} — Optimal Window Diagnostics", fontsize=12)
    fig.tight_layout()

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_window_diagnostics_old(t, result, freq_range=(100, 19000), n_log_bins=200):
    """
    One-page diagnostic plot per tree:
      - top: time–frequency comparison (full vs best windowed)
      - overlay: mean TF (log-binned)
      - bottom: metric diagnostics (r_allfreq, var_highfreq, score)
    """
    ir_full = result["signals"]["ir_full"]
    ir_best = result["signals"]["ir_best"]
    fs = result["fs"]

    # --- Flatten TF arrays safely ---
    tf_full = np.abs(ir_full.freq).flatten()
    tf_best = np.abs(ir_best.freq).flatten()

    # --- Compute frequency axis from FFT ---
    n_samples = ir_full.n_samples
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # --- Apply frequency mask ---
    valid_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    tf_full_valid = tf_full[valid_mask]
    tf_best_valid = tf_best[valid_mask]
    freqs_valid = freqs[valid_mask]

    # --- Compute log-binned mean TF for overlay ---
    freq_bins = np.logspace(
        np.log10(freq_range[0]), np.log10(freq_range[1]), n_log_bins
    )
    tf_mean = np.zeros(len(freq_bins) - 1)
    bin_centers = np.zeros(len(freq_bins) - 1)

    for i in range(len(freq_bins) - 1):
        mask = (freqs_valid >= freq_bins[i]) & (freqs_valid < freq_bins[i + 1])
        bin_centers[i] = (freq_bins[i] + freq_bins[i + 1]) / 2
        if np.any(mask):
            tf_mean[i] = np.nanmean(tf_full_valid[mask])
        else:
            tf_mean[i] = np.nan

    # --- Create figure ---
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 2.5, 1], hspace=0.5)

    # --- Top two panels: Time–Frequency plots ---
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
        ir_best,
        unit="samples",
        dB_time=True,
        label="Best windowed IR",
        ax=[ax_time, ax_freq],
    )

    # --- Overlay mean TF ---
    ax_freq.plot(
        bin_centers,
        20 * np.log10(tf_mean + 1e-12),
        "r--",
        lw=1.5,
        label="Mean TF (log-binned)",
    )

    ax_time.set_xlim(0, len(ir_full.time[0]))
    ax_freq.set_ylim(-40, 50)
    ax_freq.legend(loc="lower left", fontsize=7)
    ax_freq.axvline(freq_range[0], color="green", ls="--", lw=2)
    ax_freq.axvline(freq_range[1], color="green", ls="--", lw=2)

    title_text = (
        f"Tree {t['tree_id']} ({t['species_short']})\n"
        f"Best window: {result['best_window_samples']} samples "
        f"({result['best_window_ms']:.1f} ms), "
        f"Uncertainty = {result['uncertainty']:.3f}"
    )
    ax_time.set_title(title_text, fontsize=10)

    # --- Diagnostics subplot ---
    ax_diag = fig.add_subplot(gs[2])
    wl = result["diagnostics"]["window_lengths"]
    ax_diag.plot(wl, result["diagnostics"]["r_allfreq"], label="r_allfreq", marker="o")
    ax_diag.plot(
        wl, result["diagnostics"]["var_highfreq"], label="var_highfreq", marker="x"
    )
    ax_diag.plot(wl, result["diagnostics"]["score"], label="score", marker="s")
    ax_diag.axvline(result["best_window_samples"], color="red", lw=1.5, ls="--")
    ax_diag.set_xlabel("Window length (samples)")
    ax_diag.set_ylabel("Metric value")
    ax_diag.legend(fontsize=8)
    ax_diag.grid(True, ls="--", alpha=0.3)

    fig.suptitle(f"Window Optimization Diagnostics — Tree {t['tree_id']}", fontsize=12)
    fig.tight_layout()
    # plt.show()

    return fig


def plot_window_optimization_odd(result, freq_range=(100, 19000)):
    """
    Plot IR and TF diagnostics from find_optimal_window_length result.

    Parameters
    ----------
    result : dict
        Output from find_optimal_window_length().
    freq_range : tuple
        Frequency range to highlight in the TF plot.
    """
    ir_full = result["signals"]["ir_full"]
    ir_best = result["signals"]["ir_best"]
    fs = result["fs"]

    # --- Impulse Response Panel ---
    t_ir = np.arange(ir_full.n_samples) / fs
    ir_full_data = ir_full.time.flatten()
    ir_best_data = ir_best.time.flatten()

    # --- Transfer Function Panel ---
    freqs = np.fft.rfftfreq(ir_full.n_samples, 1 / fs)
    tf_full = 20 * np.log10(np.abs(ir_full.freq).flatten() + 1e-12)
    tf_best = 20 * np.log10(np.abs(ir_best.freq).flatten() + 1e-12)
    tf_mean = (tf_full + tf_best) / 2

    # --- Create figure ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    # Panel 1: IR
    axes[0].plot(t_ir * 1000, ir_full_data, label="Full IR", color="C0")
    axes[0].plot(t_ir * 1000, ir_best_data, label="Windowed IR", color="C1", alpha=0.7)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Tree {result['tree_id']} - Impulse Response")
    axes[0].legend()
    axes[0].grid(True)

    # Panel 2: TF
    axes[1].plot(freqs, tf_full, label="Full IR TF", color="C0")
    axes[1].plot(freqs, tf_best, label="Windowed IR TF", color="C1", alpha=0.7)
    axes[1].plot(freqs, tf_mean, label="Mean TF", color="C2", linestyle="--")

    # Highlight frequency range
    axes[1].axvline(freq_range[0], color="k", linestyle=":", label="Freq Range")
    axes[1].axvline(freq_range[1], color="k", linestyle=":")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].set_title("Transfer Function")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_window_optimization(t, result, freq_range=(100, 19000)):
    """
    Plot impulse response (IR) and transfer function (TF) comparison for a tree.
    Shows full IR, best windowed IR, and smoothed TF.

    Parameters
    ----------
    t : dict
        Tree metadata (must contain 'tree_id')
    result : dict
        Output from find_optimal_window_length
    freq_range : tuple
        Frequency range to highlight in TF plot
    """
    # --- Extract signals ---
    ir_full = result["signals"]["ir_full"]
    ir_best = result["signals"]["ir_best"]
    smoothed_tf = result["signals"]["ir_smoothed"]
    fs = result["fs"]

    # --- Compute TFs ---
    # Full and windowed TFs in dB
    tf_full_db = 20 * np.log10(np.abs(ir_full.freq).flatten() + 1e-12)
    tf_best_db = 20 * np.log10(np.abs(ir_best.freq).flatten() + 1e-12)
    freqs = np.linspace(0, fs / 2, len(tf_full_db))

    # --- Create figure ---
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 3], hspace=0.4)

    # --- Top: Impulse responses ---
    ax_time = fig.add_subplot(gs[0])
    pyfar.plot.time(ir_full, unit="samples", dB=True, label="Full IR", ax=ax_time)
    pyfar.plot.time(
        ir_best, unit="samples", dB=True, label="Best windowed IR", ax=ax_time
    )
    ax_time.set_title(
        f"Tree {t['tree_id']} ({t.get('species_short', '')})\n"
        f"Best window: {result['best_window_samples']} samples "
        f"({result['best_window_ms']:.1f} ms), "
        f"Uncertainty = {result['uncertainty']:.3f}",
        fontsize=10,
    )
    ax_time.set_xlabel("Time [samples]")
    ax_time.set_ylabel("Amplitude")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(frameon=False, fontsize=8)

    # --- Bottom: Transfer functions ---
    ax_freq = fig.add_subplot(gs[1])

    # Full and windowed TF (linear frequency)
    ax_freq.plot(freqs, tf_full_db, label="Full TF", lw=1, alpha=0.6)
    ax_freq.plot(freqs, tf_best_db, label="Best windowed TF", lw=1)

    # Smoothed TF (log-binned)
    n_log_bins = len(smoothed_tf)
    log_freqs = np.logspace(
        np.log10(freq_range[0]), np.log10(freq_range[1]), n_log_bins
    )
    ax_freq.plot(log_freqs, smoothed_tf, label="Smoothed TF", lw=1.2, ls="--")

    # --- Axis settings ---
    ax_freq.set_xscale("log")
    ax_freq.set_xlim(freq_range)  # ensures green lines are visible
    ax_freq.set_ylim(-40, 50)  # adjust if needed
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel("Magnitude [dB]")

    # --- Vertical lines for frequency range ---
    ax_freq.axvline(freq_range[0], color="green", ls="--", lw=1.5)
    ax_freq.axvline(freq_range[1], color="green", ls="--", lw=1.5)

    ax_freq.grid(True, which="both", alpha=0.3)
    ax_freq.legend(frameon=False, fontsize=8)
    ax_freq.set_title("Transfer Function Comparison", fontsize=10)

    fig.tight_layout()
    return fig


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

    # --- TF window optimization for all trees ---
    pdf_path = figures_dir / "tf_window_optimization.pdf"
    with PdfPages(pdf_path) as pdf:
        for t in all_trees:
            print(f"--- Optimizing TF window for tree {t['tree_id']} ---")

            result = find_optimal_window_length(t)
            if result is None:
                continue
            fig = plot_window_optimization(t, result)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ All window optimization plots saved to {pdf_path}")
    exit()
    # --- FFT of trimmed recordings ---
    with PdfPages("figures/fft_trimmed_recordings.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            if rec is None:
                continue
            rec_trimmed = auto_trim_signal(
                rec, threshold_rel=2e-2, safety_margin_ms=0.0
            )
            try:
                plot_fft_trimmed_recording(rec_trimmed, t, pdf)
            except Exception as e:
                logging.warning(f"FFT plot failed for {t['tree_id']}: {e}")
                continue

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

    # pdf_path = figures_dir / "tf_window_optimization_auto.pdf"
    # with PdfPages(pdf_path) as pdf:
    #     # === Find optimal window length ===
    #     print(f"\n--- Optimizing TF window for tree {tree_id} ---")
    #     result = find_optimal_window_length(
    #         t=traits,
    #         freq_range=(100, 20000),  # full frequency range
    #         alpha=0.3,
    #     )

    #     # if result is None:
    #     #     print(f"⚠️ Optimization failed for tree {tree_id}")
    #     #     continue

    #     # === Plot diagnostics and TF comparison ===
    #     fig = plot_window_diagnostics(traits, result, freq_range=(100, 20000))
    #     pdf.savefig(fig)
    #     plt.close(fig)

    #     print(
    #         f"Tree {tree_id}: optimal window = {result['best_window_samples']} samples "
    #         f"(≈ {result['best_window_ms']:.1f} ms), "
    #         f"uncertainty = {result['uncertainty']:.3f}"
    #     )
    exit()
    # === Time–Frequency window sweeps for all trees ===
    pdf_path = Path("figures/tf_window_exploration.pdf")
    pdf_path.parent.mkdir(exist_ok=True)

    """
    notes:
        - If you start your window too late you lose early high-frequency energy that arrives first in the impulse.
        Offsets shift the windowed pulse in time; if the offset omits the direct path’s earliest samples,
        the high-frequency content is diminished.
        - Time and frequency resolution are reciprocal: to resolve frequency f_min reasonably well you need a
        window roughly T ≈ 1 / f_min. To include ~100 Hz content you need order 10 ms (≈ 480 samples at 48 kHz).
        Therefore short windows (e.g. 240 samples) don’t capture below ~600 Hz properly.
        - Windowing ≈ smoothing in frequency domain
        -> exact calculation is possible:
            considering sampling rate (S) and length of the window in samples (l)
            then the duration of the window in time is T=l/S seconds
            and the frequency resolution is 1/T in Hz
            resolution means: smallest spacing where the FFT can distinguish separate frequencies
            so at one bin of the FFT you have an average of the surrounding 1/T Hz neighbourhood
        → short windows broaden the main lobe and smooth spectral ripples,
        reducing comb-filter wiggles but at the cost of frequency resolution.
        - So the task is fundamentally a trade-off:
            - keep window long enough to preserve low-frequency content,
            - but keep it short enough to remove reflections that cause spectral ripples (wiggliness).
            - so we want low wiggliness (remove reflections, short IR) and the window being as short as possible (without loosing spectral resolution from the short window).
    """

    window_sizes_samples = (
        120,
        480,
        1200,
        2400,
        4800,
        4800,
    )  # ≈ 2.5, 10, 25, 50, 100 ms at 48kHz
    offset_samples = 0
    freq_range = (100, 20000)

    with PdfPages(pdf_path) as pdf:
        for t in all_trees:
            fig = create_tf_window_grid(
                t, window_sizes_samples, offset_samples, freq_range
            )
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
                print(f"✅ Added TF window grid for {t['tree_id']}")
            else:
                print(f"⚠️ Skipped {t['tree_id']} (no data or error)")
    print(f"✅ All tree window sweeps saved to: {pdf_path}")

    # --- Per-tree TFs (distance-only) ---
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

    # --- Per-tree TFs (median scaling) ---
    with PdfPages("figures/per_tree_tf_med.pdf") as pdf:
        for t in all_trees:
            rec = t["data"].get("recording")
            ref = t["data"].get("ref_med")
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
                logging.warning(f"TF compute failed for {t['tree_id']} (med): {e}")
                continue
            plot_tf_variant(windowed_tf, rec_trimmed, t, "median-scaled", pdf)

    # --- Heatmaps ---
    with PdfPages("figures/tf_heatmap_dist.pdf") as pdf:
        plot_tf_heatmap_variant(all_trees, "dist", pdf)
    with PdfPages("figures/tf_heatmap_med.pdf") as pdf:
        plot_tf_heatmap_variant(all_trees, "med", pdf)

    print("Analysis completed.")


if __name__ == "__main__":
    main()

exit()
