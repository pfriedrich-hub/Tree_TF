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
            - find optimal hspace and wspace
            - implement and check optimization code (chatty)
            - push to git
    2. fix reference scaling/ampification of low frequencies:
        - research: ISO-implementation-code
        - research: multiplication in frequency domain convolution in time domain, so is the deconvolution done by a scaling in frequency domain? is this the trick?
        - try baselining of lowest frequency in frequency domain before or after deconvolution and look at tfs with different scalings and heatmaps
    3. minor fixes:
        - root mean square (RMS) instead of median scaling
        - annotate real frequencies in bands on x-axis
        - octave scaling instead of logarithmic
    4. TLS-Scans:
        - important information of every tree
        - printed
    5. model absorption potential predicted by tree traits -> look into satellite data and find silent places from tree traits
    6. write protocol:
        - motivation to think on my own because of low frequency of intense meetings
        - possibility to apply tools in a free and self-responsible way: best experience of the whole masters program
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


def create_tf_window_grid(t, window_sizes_samples, offsets_samples, freq_range):
    """
    Create a grid of time–frequency comparison plots (before vs after windowing)
    for one tree. Each grid cell shows two independent subplots:
      • top: time-domain (samples)
      • bottom: frequency-domain (Hz, log-scaled)
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

    # --- Figure grid (wider layout) ---
    n_rows = len(offsets_samples)
    n_cols = len(window_sizes_samples)
    fig, axes_grid = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False
    )

    for i, offset_samples in enumerate(offsets_samples):
        for j, win_samples in enumerate(window_sizes_samples):
            ax_parent = axes_grid[i, j]

            # --- Create nested 2-row subplots (no shared axes!) ---
            gs = ax_parent.get_subplotspec().subgridspec(2, 1, hspace=0.35)
            ax_time = fig.add_subplot(gs[0])
            ax_freq = fig.add_subplot(gs[1])
            ax_parent.set_visible(False)

            # --- Apply window ---
            start_samp = int(offset_samples)
            end_samp = int(offset_samples + win_samples)

            try:
                # ir_windowed = pyfar.dsp.time_window(
                #     ir_deconvolved,
                #     (start_samp, end_samp),
                #     "boxcar",
                #     unit="samples",
                #     crop="window",
                # )
                ir_windowed = pyfar.dsp.time_window(
                    ir_deconvolved,
                    (start_samp, end_samp),
                    "boxcar",
                    unit="samples",
                    crop="none",  # keep full signal length, preserve offset
                )
            except Exception as e:
                print(
                    f"⚠️ Window failed for {t['tree_id']} ({offset_samples},{win_samples}): {e}"
                )
                continue

            # --- Time–frequency plots ---
            axes_tf = [ax_time, ax_freq]
            pyfar.plot.time_freq(
                ir_deconvolved,
                unit="samples",
                dB_time=True,
                label="Before Windowing",
                ax=axes_tf,
            )
            pyfar.plot.time_freq(
                ir_windowed,
                unit="samples",
                dB_time=True,
                label="After Windowing",
                ax=axes_tf,
            )

            # --- Customize axes ---
            ax_time.set_xlim(left=0)
            ax_freq.set_ylim(-40, 50)
            ax_freq.legend(loc="lower left", fontsize=7)

            # --- Speaker band markers ---
            ax_freq.axvline(freq_range[0], color="green", ls="--", lw=2)
            ax_freq.axvline(freq_range[1], color="green", ls="--", lw=2)
            ax_freq.text(
                freq_range[0],
                0,
                f"{freq_range[0]} Hz",
                color="green",
                fontsize=7,
                va="bottom",
                ha="left",
            )
            ax_freq.text(
                freq_range[1],
                20,
                f"{freq_range[1] / 1000:.1f} kHz",
                color="green",
                fontsize=7,
                va="bottom",
                ha="right",
            )

            ax_time.set_title(
                f"{offset_samples} ms offset | {win_samples} ms window", fontsize=8
            )

    fig.suptitle(
        f"Tree {t['tree_id']} ({t['species_short']}) — TF window sweep", fontsize=10
    )
    fig.subplots_adjust(wspace=0.55, hspace=0.3)
    # fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    window_sizes_samples = (120, 240, 360)
    offsets_samples = (0, 15, 30)
    freq_range = (100, 20000)

    with PdfPages(pdf_path) as pdf:
        for t in all_trees:
            fig = create_tf_window_grid(
                t, window_sizes_samples, offsets_samples, freq_range
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
