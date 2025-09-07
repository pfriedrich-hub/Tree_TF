#!/usr/bin/env python3
"""
homogenize.py

Homogenize tree acoustic measurements (post-recording pipeline).
Adds diagnostic plots of recording vs scaled references.

Outputs (per tree):
    data/<id>/<id>.pkl  -- contains raw recording, ref_dist, ref_med
Also:
    results_all_trees.pdf  -- overlay plots (recording + ref_dist + ref_med)
"""

import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from slab import Sound
from record_tf import *
import numpy as np
import scipy.signal as sps
import copy
import csv
from scipy.signal import hilbert

# CONFIG
DATA_DIR = Path.cwd() / "data"
REF_DIST = 2.0
WINDOW_SIZE = 120  # ms
SHOW = False

LOG_IDS = {"227", "257", "247", "277", "499", "502", "332", "274", "232",
           "467", "270", "281", "298", "327", "518", "333", "342"}
SPECIAL_IDS = {"313", "353", "344"}


def parse_distance(foldername: str) -> float:
    try:
        dist_str = foldername.split("_")[1]
        return float(dist_str.replace(",", "."))
    except Exception:
        return None


def get_tree_dirs(base_dir: Path):
    """Yield long-ID dirs, excluding refs & short-ID dirs."""
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


def compute_alpha_median(recording, reference, sr, band=(200, 2000)):
    rec = np.squeeze(np.asarray(recording.data))
    ref = np.squeeze(np.asarray(reference.data))
    n = min(len(rec), len(ref))
    rec, ref = rec[:n], ref[:n]

    freqs, P_rec = sps.welch(rec, fs=sr, nperseg=4096)
    _, P_ref = sps.welch(ref, fs=sr, nperseg=4096)
    A_rec = np.sqrt(np.maximum(P_rec, 1e-20))
    A_ref = np.sqrt(np.maximum(P_ref, 1e-20))

    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 1.0
    ratios = A_rec[mask] / np.maximum(A_ref[mask], 1e-12)
    alpha = float(np.median(ratios))
    return alpha if np.isfinite(alpha) and alpha > 0 else 1.0


def write_special_pickle(tree_id: str, recording: Sound, reference: Sound):
    """Save direct reference recording for special trees (short-ID dirs)."""
    id_dict = {'recording': recording, 'reference': reference}
    tree_dir = DATA_DIR / tree_id
    tree_dir.mkdir(parents=True, exist_ok=True)
    file_path = tree_dir / f"{tree_id}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved direct-ref .pkl for {tree_id} -> {file_path}")


def overlay_plot(recording, ref_dist, ref_med, sr, tree_id, pdf, alpha_med):
    """Overlay recording, distance-scaled ref, and median-scaled ref (envelopes)."""
    t = np.arange(len(recording.data)) / sr
    rec = np.squeeze(recording.data)
    r_dist = np.squeeze(ref_dist.data)[:len(t)]
    r_med = np.squeeze(ref_med.data)[:len(t)]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, rec, color="black", linewidth=1.2, label="Recording")

    env_med = np.abs(hilbert(r_med))
    ax.plot(t, env_med, color="C1", linewidth=2.0,
            label=f"Ref median-scaled (α={alpha_med:.2f}, {20*np.log10(alpha_med):.1f} dB)")
    ax.plot(t, -env_med, color="C1", linewidth=2.0)

    env_dist = np.abs(hilbert(r_dist))
    ax.plot(t, env_dist, color="crimson", linewidth=2.0, label="Ref dist-scaled")
    ax.plot(t, -env_dist, color="crimson", linewidth=2.0)

    ax.set_title(f"{tree_id}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    # --- Manually specified valid folders ---
    valid_folders = [
        "313_4.6_20N",
        "353_8.1_255W",
        "344_3.15_261W"
    ]

    for folder_name in valid_folders:
        tree_id = folder_name.split('_')[0]  # Assuming tree_id is the first part of the folder name
        tree_folder = DATA_DIR / folder_name
        print(tree_folder)

        rec_path = tree_folder / f"{folder_name}_rec.wav"
        ref_path = DATA_DIR / f"{folder_name}_ref" / f"{folder_name}_ref.wav"

        if rec_path.exists() and ref_path.exists():
            recording = Sound(str(rec_path))
            reference = Sound(str(ref_path))
            write_special_pickle(tree_id, recording, reference)

    # --- overlay plots for all trees ---
    csv_file = open("scaling_factors.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["tree_id", "alpha_median", "dB_median"])

    pdf_path = Path("figures/results_all_trees.pdf")
    with PdfPages(pdf_path) as pdf:
        for tree_dir in get_tree_dirs(DATA_DIR):
            tree_id = tree_dir.name
            rec_distance = parse_distance(tree_id)
            if rec_distance is None:
                continue

            wav_path = tree_dir / f"{tree_id}_rec.wav"
            if not wav_path.exists():
                continue
            recording = Sound.read(str(wav_path))

            base = DATA_DIR
            if tree_id.split("_")[0] in LOG_IDS:
                ref_path = base / "ref" / "ref_rec.wav"
            else:
                ref_path = base / "ref_linear" / "ref_linear_rec.wav"
            if not ref_path.exists():
                continue
            reference = Sound.read(str(ref_path))

            # distance-only scaling (raw)
            ref_dist = distance_scale(reference, input_distance=REF_DIST, output_distance=rec_distance)

            # empirical scaling
            alpha_med = compute_alpha_median(recording, ref_dist, recording.samplerate)
            db_med = 20*np.log10(alpha_med) if alpha_med > 0 else np.nan
            writer.writerow([tree_id, f"{alpha_med:.4f}", f"{db_med:.2f}"])

            ref_med = copy.deepcopy(ref_dist); ref_med.data *= alpha_med

            # overlay plot
            overlay_plot(recording, ref_dist, ref_med, recording.samplerate, tree_id, pdf, alpha_med)

            # save raw references + TFs
            id_dict = {
                "recording": recording,
                "ref_dist": ref_dist,   # raw distance-scaled reference
                "ref_med": ref_med,     # raw median-scaled reference
                # "raw_tf_dist": raw_tf_dist,
                # "windowed_tf_dist": windowed_tf_dist,
                # "raw_tf_med": raw_tf_med,
                # "windowed_tf_med": windowed_tf_med,
            }
            file_path = tree_dir / f"{tree_id}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)

    csv_file.close()
    print(f"All results saved. Plots written to {pdf_path}")
