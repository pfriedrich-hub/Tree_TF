"""
Noise Scenarios
===============
Converts scenario MP3s to WAV and loads them for sonification.

Sounds (in sounds/):
    highway.mp3, tram.mp3, construction.mp3, children.mp3

Usage:
    from noise_profiles import load_scenario_audio
    scenarios = load_scenario_audio()  # {key: (audio_array, sr)}
"""
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SOUNDS_DIR = Path(__file__).parent / "sounds"
NOISE_WAV_DIR = Path(__file__).parent / "noise_wav"

SCENARIOS = {
    "highway": "highway.mp3",
    "tram": "tram.mp3",
    "construction": "construction.mp3",
    "children": "children.mp3",
}


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sr: int = 44100) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(mp3_path),
           "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16", str(wav_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")
    return wav_path


def ensure_wavs(sounds_dir: Path = None) -> dict:
    """Convert MP3s to WAVs if needed. Returns {key: wav_path}."""
    if sounds_dir is None:
        sounds_dir = SOUNDS_DIR
    wav_dir = NOISE_WAV_DIR

    paths = {}
    for key, filename in SCENARIOS.items():
        mp3 = sounds_dir / filename
        if not mp3.exists():
            logger.warning(f"Missing: {mp3}")
            continue
        wav = wav_dir / f"{key}.wav"
        if not wav.exists() or wav.stat().st_mtime < mp3.stat().st_mtime:
            logger.info(f"Converting {filename} → {wav.name}")
            convert_mp3_to_wav(mp3, wav)
        paths[key] = wav
    return paths


def load_scenario_audio(sounds_dir: Path = None) -> dict:
    """
    Load all scenario sounds as numpy arrays.
    Returns {key: (audio_array, sample_rate)}.
    """
    import soundfile as sf

    wav_paths = ensure_wavs(sounds_dir)
    audio_dict = {}
    for key, wav_path in wav_paths.items():
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio_dict[key] = (audio, sr)
        logger.info(f"  Loaded: {key} ({len(audio)/sr:.1f}s)")
    return audio_dict
