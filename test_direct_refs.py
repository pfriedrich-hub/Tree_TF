#!/usr/bin/env python3
"""
test_direct_refs.py
testing the scaling with the trees where direct references exist
For special trees (313, 353, 344), generate two PDFs:
1. recording vs direct reference (direct measurement)
2. recording vs distance-scaled reference (2 m reference scaled to canopy distance)
"""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

DATA_DIR = Path.cwd() / "data"
SPECIAL_IDS = {"313", "353", "344"}

OUTPUT_DIRECT = Path("figures/special_trees_direct.pdf")
OUTPUT_SCALED = Path("figures/special_trees_scaled.pdf")


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_array(obj):
    if obj is None:
        return None
    if hasattr(obj, "data"):
        arr = np.asarray(obj.data)
    else:
        arr = np.asarray(obj)
    return np.squeeze(arr)


def plot(rec, ref, sr, tree_id, label, pdf):
    n = min(len(rec), len(ref))
    t = np.arange(n) / sr
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, rec[:n], label="Recording", alpha=0.7)
    ax.plot(t, ref[:n], label=f"Reference ({label})", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{tree_id} — {label}")
    ax.legend()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # --- Direct reference plots (short-ID dirs) ---
    with PdfPages(OUTPUT_DIRECT) as pdf:
        for tree_id in SPECIAL_IDS:
            pkl = DATA_DIR / tree_id / f"{tree_id}.pkl"
            if not pkl.exists():
                continue
            data = load_pickle(pkl)
            rec = to_array(data.get("recording"))
            ref = to_array(data.get("reference"))
            if rec is None or ref is None:
                continue
            sr = getattr(data.get("recording"), "samplerate", 48000)
            plot(rec, ref, sr, tree_id, "direct", pdf)

    print(f"Direct reference PDF saved: {OUTPUT_DIRECT}")

    # --- Distance-scaled reference plots (long-ID dirs) ---
    with PdfPages(OUTPUT_SCALED) as pdf:
        for tree_dir in DATA_DIR.iterdir():
            if not tree_dir.is_dir():
                continue
            if tree_dir.name.split("_")[0] not in SPECIAL_IDS:
                continue
            pkl = tree_dir / f"{tree_dir.name}.pkl"
            if not pkl.exists():
                continue
            data = load_pickle(pkl)
            rec = to_array(data.get("recording"))
            ref = to_array(data.get("ref_dist"))  # raw distance-scaled reference
            if rec is None or ref is None:
                continue
            sr = getattr(data.get("recording"), "samplerate", 48000)
            tree_id = tree_dir.name
            plot(rec, ref, sr, tree_id, "dist-scaled", pdf)

    print(f"Scaled reference PDF saved: {OUTPUT_SCALED}")


if __name__ == "__main__":
    main()
