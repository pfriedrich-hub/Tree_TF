"""
Tree Data Model
===============
Class-based representation of a tree with acoustic and structural data.
Each Tree object holds all raw data, processed signals, and derived metrics.
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyfar
import slab

from config import (
    BAND_HIGH, BAND_LOW, BAND_MID, FREQ_HIGH, FREQ_LOW,
    get_species_info, parse_tree_id, SAMPLE_RATE
)


@dataclass
class Tree:
    """
    Represents a single tree with acoustic measurements and structural traits.
    
    This class encapsulates:
    - Raw acoustic recordings (recording, reference)
    - Processed signals (trimmed, deconvolved, windowed)
    - Transfer function and attenuation metrics
    - Structural traits from TLS
    - Leaf traits from measurements
    
    The processing pipeline is:
    1. Load raw recording and reference
    2. Baseline reference in frequency domain
    3. Trim leading silence
    4. Deconvolve to get impulse response (IR)
    5. Window IR to remove reflections
    6. Extract transfer function (TF) and attenuation metrics
    """
    
    # Identity
    tree_id: str
    numeric_id: int = field(init=False)
    distance_m: float = field(init=False)
    direction: Optional[str] = field(init=False)
    
    # Species info
    species_short: str = field(init=False)
    species_long: str = field(init=False)
    is_needleleaf: bool = field(init=False)
    is_broadleaf: bool = field(init=False)
    
    # Raw audio (slab.Sound objects)
    recording: Optional[slab.Sound] = field(default=None, repr=False)
    reference: Optional[slab.Sound] = field(default=None, repr=False)
    reference_baselined: Optional[slab.Sound] = field(default=None, repr=False)
    
    # Trimmed audio
    recording_trimmed: Optional[slab.Sound] = field(default=None, repr=False)
    reference_trimmed: Optional[slab.Sound] = field(default=None, repr=False)
    trim_start_idx: int = 0
    
    # Processed signals (pyfar.Signal objects)
    ir_full: Optional[pyfar.Signal] = field(default=None, repr=False)
    ir_windowed: Optional[pyfar.Signal] = field(default=None, repr=False)
    window: Optional[pyfar.Signal] = field(default=None, repr=False)
    
    # Transfer function data
    freqs: Optional[np.ndarray] = field(default=None, repr=False)
    tf_magnitude_db: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Attenuation metrics (dB, negative = attenuation)
    attenuation_overall: Optional[float] = None
    attenuation_low: Optional[float] = None
    attenuation_mid: Optional[float] = None
    attenuation_high: Optional[float] = None
    
    # Sonification outputs
    filtered_noise: Optional[np.ndarray] = field(default=None, repr=False)
    filtered_playground: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Parse tree_id and set species info."""
        parsed = parse_tree_id(self.tree_id)
        self.numeric_id = parsed["numeric_id"]
        self.distance_m = parsed["distance_m"]
        self.direction = parsed["direction"]
        
        species = get_species_info(self.tree_id)
        self.species_short = species["species_short"]
        self.species_long = species["species_long"]
        self.is_needleleaf = species["is_needleleaf"]
        self.is_broadleaf = species["is_broadleaf"]
    
    @property
    def sample_rate(self) -> int:
        """Get sample rate from recording."""
        if self.recording is not None:
            return self.recording.samplerate
        return SAMPLE_RATE
    
    @property
    def leaf_type(self) -> str:
        """Return 'needleleaf' or 'broadleaf'."""
        return "needleleaf" if self.is_needleleaf else "broadleaf"
    
    def compute_attenuation_metrics(self):
        """
        Compute band-averaged attenuation from the transfer function.
        
        Attenuation is the mean magnitude in dB within each frequency band.
        Negative values indicate attenuation (signal reduction).
        """
        if self.freqs is None or self.tf_magnitude_db is None:
            return
        
        freqs = self.freqs
        tf_db = self.tf_magnitude_db
        
        # Overall attenuation (full range)
        mask_all = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
        self.attenuation_overall = float(np.mean(tf_db[mask_all]))
        
        # Low band
        mask_low = (freqs >= BAND_LOW[0]) & (freqs < BAND_LOW[1])
        self.attenuation_low = float(np.mean(tf_db[mask_low]))
        
        # Mid band
        mask_mid = (freqs >= BAND_MID[0]) & (freqs < BAND_MID[1])
        self.attenuation_mid = float(np.mean(tf_db[mask_mid]))
        
        # High band
        mask_high = (freqs >= BAND_HIGH[0]) & (freqs <= BAND_HIGH[1])
        self.attenuation_high = float(np.mean(tf_db[mask_high]))
    
    def to_dict(self) -> dict:
        """
        Export tree data to a dictionary for CSV/DataFrame export.
        Includes only scalar metrics, not arrays.
        Note: Structural and leaf traits are merged later in run_correlation.py
        """
        return {
            # Identity
            "tree_id": self.tree_id,
            "numeric_id": self.numeric_id,
            "distance_m": self.distance_m,
            "direction": self.direction,
            
            # Species
            "species_short": self.species_short,
            "species_long": self.species_long,
            "is_needleleaf": self.is_needleleaf,
            "is_broadleaf": self.is_broadleaf,
            "leaf_type": self.leaf_type,
            
            # Attenuation
            "attenuation_overall_db": self.attenuation_overall,
            "attenuation_low_db": self.attenuation_low,
            "attenuation_mid_db": self.attenuation_mid,
            "attenuation_high_db": self.attenuation_high,
        }
    
    def save(self, path: Path):
        """Save complete tree object to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: Path) -> "Tree":
        """Load tree object from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def __repr__(self):
        if self.attenuation_overall is not None:
            return (
                f"Tree({self.tree_id}, {self.species_short}, "
                f"att={self.attenuation_overall:.1f}dB)"
            )
        return f"Tree({self.tree_id})"
