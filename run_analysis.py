"""
Tree Sound Absorption Analysis Pipeline
========================================
Processes tree recordings into transfer functions (generic low/mid/high/overall bands).
Filters 4 urban noise scenarios through each tree for sonification.

Usage:
    python run_analysis.py --sounds-dir sounds/
    python run_analysis.py --slider-only --model-csv output/figures/models/stepwise_coefficients.csv
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import slab

from config import DATA_DIR, TREE_IDS, OUT_DIR
from noise_profiles import load_scenario_audio, SOUNDS_DIR
from processing import (
    baseline_reference, deconvolve, extract_transfer_function,
    generate_filtered_audio, trim_signals, window_ir,
)
from tree import Tree
from visualization import create_all_figures, setup_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_reference(data_dir: Path) -> slab.Sound:
    ref_path = data_dir / "ref" / "ref_rec.wav"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")
    return slab.Sound.read(str(ref_path))


def load_recording(data_dir: Path, tree_id: str) -> slab.Sound:
    rec_path = data_dir / tree_id / f"{tree_id}_rec.wav"
    if not rec_path.exists():
        raise FileNotFoundError(f"Recording not found: {rec_path}")
    return slab.Sound.read(str(rec_path))


def process_tree(
    tree_id: str,
    data_dir: Path,
    reference: slab.Sound,
    scenario_audio: dict,
) -> Tree:
    """Process one tree: TF pipeline + filter scenario sounds."""
    logger.info(f"Processing {tree_id}...")
    tree = Tree(tree_id=tree_id)

    try:
        tree.recording = load_recording(data_dir, tree_id)
    except FileNotFoundError as e:
        logger.warning(f"Skipping {tree_id}: {e}")
        return tree

    tree.reference = reference

    # Acoustic pipeline (unchanged)
    tree.reference_baselined = baseline_reference(tree.recording, reference)
    tree.recording_trimmed, tree.reference_trimmed, tree.trim_start_idx = trim_signals(
        tree.recording, tree.reference_baselined)
    tree.ir_full = deconvolve(tree.recording_trimmed, tree.reference_trimmed)
    tree.ir_windowed, tree.window = window_ir(tree.ir_full)
    tree.freqs, tree.tf_magnitude_db = extract_transfer_function(tree.ir_windowed)

    # Attenuation metrics (generic low/mid/high/overall — unchanged)
    tree.compute_attenuation_metrics()

    # Sonification: filter each scenario sound through this tree's IR
    for skey, (audio, sr) in scenario_audio.items():
        tree.filtered_scenarios[skey] = generate_filtered_audio(
            tree.ir_windowed, audio, sr)

    logger.info(f"  {tree_id}: att={tree.attenuation_overall:.1f} dB, "
                f"scenarios={list(tree.filtered_scenarios.keys())}")
    return tree


def generate_sliders(output_dir: Path, model_csv: str = None):
    """Generate slider HTML from R model coefficients."""
    from slider import generate_slider_html, load_models_from_csv
    slider_dir = output_dir / "sliders"
    slider_dir.mkdir(parents=True, exist_ok=True)

    if model_csv and Path(model_csv).exists():
        banks = load_models_from_csv(model_csv)
        for leaf_type, bank in banks.items():
            out = slider_dir / f"slider_{leaf_type}.html"
            generate_slider_html(bank, str(out))
            logger.info(f"Slider: {out}")
    else:
        logger.info("No model CSV — run R stepwise first, then:")
        logger.info("  python run_analysis.py --slider-only --model-csv <path>")


def run_pipeline(
    data_dir: Path = None,
    output_dir: Path = None,
    tree_ids: list = None,
    sounds_dir: Path = None,
    model_csv: str = None,
):
    if data_dir is None: data_dir = DATA_DIR
    if output_dir is None: output_dir = OUT_DIR
    if sounds_dir is None: sounds_dir = SOUNDS_DIR

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    setup_style()

    out_pkl = output_dir / "pkl"
    out_fig = output_dir / "figures"
    out_wav = output_dir / "wav"
    out_csv = output_dir / "csv"
    for d in [out_pkl, out_fig, out_wav, out_csv]:
        d.mkdir(parents=True, exist_ok=True)

    # Load scenario sounds (highway, tram, construction, children)
    logger.info("=== Loading scenario sounds ===")
    scenario_audio = load_scenario_audio(sounds_dir)

    # Save unfiltered references
    for skey, (audio, sr) in scenario_audio.items():
        sf.write(out_wav / f"{skey}_unfiltered.wav",
                 (audio / np.max(np.abs(audio)) * 0.9).astype(np.float32), sr)

    # Load acoustic reference
    logger.info("=== Processing trees ===")
    reference = load_reference(data_dir)

    if tree_ids is None:
        tree_ids = TREE_IDS

    # Process trees
    trees = []
    for tree_id in tree_ids:
        tree = process_tree(tree_id, data_dir, reference, scenario_audio)
        trees.append(tree)
        tree.save(out_pkl / f"{tree_id}.pkl")

        # Save filtered WAVs per scenario
        for skey, audio in tree.filtered_scenarios.items():
            sf.write(out_wav / f"{tree_id}_filtered_{skey}.wav", audio, tree.sample_rate)

    # CSV (same columns as before: low/mid/high/overall)
    logger.info("=== Exporting CSV ===")
    summary_df = pd.DataFrame([t.to_dict() for t in trees])
    summary_df.to_csv(out_csv / "tree_acoustic_summary.csv", index=False)

    # Figures
    logger.info("=== Creating figures ===")
    create_all_figures(trees, out_fig)

    # Sliders
    if model_csv:
        generate_sliders(output_dir, model_csv)

    logger.info(f"=== DONE: {len(trees)} trees, {len(scenario_audio)} scenarios ===")
    return trees


def main():
    parser = argparse.ArgumentParser(description="Tree Sound Absorption Pipeline")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sounds-dir", type=Path, default=None)
    parser.add_argument("--trees", nargs="+", default=None)
    parser.add_argument("--model-csv", type=str, default=None)
    parser.add_argument("--slider-only", action="store_true")
    args = parser.parse_args()

    if args.slider_only:
        generate_sliders(args.output_dir or OUT_DIR, args.model_csv)
    else:
        run_pipeline(
            data_dir=args.data_dir, output_dir=args.output_dir,
            tree_ids=args.trees, sounds_dir=args.sounds_dir,
            model_csv=args.model_csv)


if __name__ == "__main__":
    main()
