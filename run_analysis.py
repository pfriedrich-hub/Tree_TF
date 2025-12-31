"""
Tree Sound Absorption Analysis Pipeline
========================================
Main script to process all tree recordings and generate outputs.

Usage (command line):
    python run_analysis.py --data-dir /path/to/data --output-dir /path/to/output

Or with defaults from config:
    python run_analysis.py
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import slab

from config import (
    DATA_DIR, TREE_IDS,
    OUT_DIR,
    WHITE_NOISE_DURATION_S, PLAYGROUND_NOISE_FILE,
)
from processing import (
    baseline_reference,
    deconvolve,
    extract_transfer_function,
    generate_filtered_noise,
    generate_filtered_audio,
    load_playground_noise,
    trim_signals,
    window_ir,
)
from tree import Tree
from visualization import create_all_figures, setup_style

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_reference(data_dir: Path) -> slab.Sound:
    """Load the reference recording."""
    ref_path = data_dir / "ref" / "ref_rec.wav"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")
    return slab.Sound.read(str(ref_path))


def load_recording(data_dir: Path, tree_id: str) -> slab.Sound:
    """Load a tree's recording."""
    rec_path = data_dir / tree_id / f"{tree_id}_rec.wav"
    if not rec_path.exists():
        raise FileNotFoundError(f"Recording not found: {rec_path}")
    return slab.Sound.read(str(rec_path))


def process_tree(
    tree_id: str,
    data_dir: Path,
    reference: slab.Sound,
    playground_audio: np.ndarray = None,
    playground_sr: int = None,
) -> Tree:
    """
    Process a single tree through the acoustic pipeline.
    
    Steps:
    1. Load recording
    2. Baseline reference
    3. Trim signals
    4. Deconvolve
    5. Window IR
    6. Extract TF and metrics
    7. Generate sonifications
    
    Parameters
    ----------
    tree_id : str
        Tree identifier (e.g., "270_5.4_240SW")
    data_dir : Path
        Path to data directory
    reference : slab.Sound
        Reference recording (shared across trees)
    playground_audio : np.ndarray
        Playground noise for sonification (optional)
    playground_sr : int
        Sample rate of playground audio
        
    Returns
    -------
    Tree
        Processed tree object
    """
    logger.info(f"Processing {tree_id}...")
    
    # Create tree object
    tree = Tree(tree_id=tree_id)
    
    # Load recording
    try:
        tree.recording = load_recording(data_dir, tree_id)
    except FileNotFoundError as e:
        logger.warning(f"Skipping {tree_id}: {e}")
        return tree
    
    tree.reference = reference
    
    # Step 1: Baseline reference
    tree.reference_baselined = baseline_reference(tree.recording, reference)
    
    # Step 2: Trim signals
    tree.recording_trimmed, tree.reference_trimmed, tree.trim_start_idx = trim_signals(
        tree.recording,
        tree.reference_baselined,
    )
    
    # Step 3: Deconvolve
    tree.ir_full = deconvolve(tree.recording_trimmed, tree.reference_trimmed)
    
    # Step 4: Window IR
    tree.ir_windowed, tree.window = window_ir(tree.ir_full)
    
    # Step 5: Extract TF
    tree.freqs, tree.tf_magnitude_db = extract_transfer_function(tree.ir_windowed)
    
    # Step 6: Compute attenuation metrics
    tree.compute_attenuation_metrics()
    
    # Step 7: Generate filtered noise for sonification
    tree.filtered_noise = generate_filtered_noise(
        tree.ir_windowed,
        duration_s=WHITE_NOISE_DURATION_S,
    )
    
    # Step 8: Generate filtered playground noise if available
    if playground_audio is not None and playground_sr is not None:
        tree.filtered_playground = generate_filtered_audio(
            tree.ir_windowed,
            playground_audio,
            playground_sr,
        )
    
    logger.info(f"  {tree_id}: attenuation = {tree.attenuation_overall:.1f} dB")
    
    return tree


def run_pipeline(
    data_dir: Path = None,
    output_dir: Path = None,
    tree_ids: list = None,
):
    """
    Run the full acoustic analysis pipeline.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing recordings (default: config.DATA_DIR)
    output_dir : Path
        Directory for outputs (default: config.OUT_DIR)
    tree_ids : list
        List of tree IDs to process (default: all)
    """
    # Use defaults from config if not specified
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUT_DIR
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    setup_style()
    
    # Create output directories
    out_pkl = output_dir / "pkl"
    out_fig = output_dir / "figures"
    out_wav = output_dir / "wav"
    out_csv = output_dir / "csv"
    
    for d in [out_pkl, out_fig, out_wav, out_csv]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load reference
    logger.info("Loading reference recording...")
    reference = load_reference(data_dir)
    
    # Load playground noise if available
    playground_audio, playground_sr = load_playground_noise(out_wav)
    if playground_audio is None:
        # Also check data directory
        playground_audio, playground_sr = load_playground_noise(data_dir)
    
    if playground_audio is not None:
        logger.info(f"Loaded playground noise ({len(playground_audio)/playground_sr:.1f}s)")
    else:
        logger.info("Playground noise not found - skipping playground sonification")
    
    # Determine which trees to process
    if tree_ids is None:
        tree_ids = TREE_IDS
    
    # Process all trees
    trees = []
    for tree_id in tree_ids:
        tree = process_tree(
            tree_id, data_dir, reference,
            playground_audio, playground_sr
        )
        trees.append(tree)
        
        # Save individual tree pickle
        tree.save(out_pkl / f"{tree_id}.pkl")
        
        # Save filtered noise WAV
        if tree.filtered_noise is not None:
            wav_path = out_wav / f"{tree_id}_filtered_noise.wav"
            sf.write(wav_path, tree.filtered_noise, tree.sample_rate)
        
        # Save filtered playground WAV
        if tree.filtered_playground is not None:
            wav_path = out_wav / f"{tree_id}_filtered_playground.wav"
            sf.write(wav_path, tree.filtered_playground, tree.sample_rate)
    
    # Also save white noise reference
    noise = np.random.normal(0, 1, int(WHITE_NOISE_DURATION_S * reference.samplerate))
    noise = (noise / np.max(np.abs(noise)) * 0.9).astype(np.float32)
    sf.write(out_wav / "white_noise_reference.wav", noise, reference.samplerate)
    
    # Export summary CSV
    logger.info("Exporting summary CSV...")
    summary_df = pd.DataFrame([t.to_dict() for t in trees])
    summary_df.to_csv(out_csv / "tree_acoustic_summary.csv", index=False)
    
    # Create figures
    logger.info("Creating figures...")
    create_all_figures(trees, out_fig)
    
    logger.info("Pipeline complete!")
    logger.info(f"  Processed {len(trees)} trees")
    logger.info(f"  Outputs in: {output_dir}")
    
    return trees


def main():
    parser = argparse.ArgumentParser(
        description="Tree Sound Absorption Analysis Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Directory containing tree recordings (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--trees",
        nargs="+",
        default=None,
        help="Specific tree IDs to process (default: all)",
    )
    
    args = parser.parse_args()
    
    trees = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tree_ids=args.trees,
    )
    
    return trees


if __name__ == "__main__":
    main()
