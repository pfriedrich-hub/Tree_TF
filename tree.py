"""
Tree Data Model
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
    get_species_info, parse_tree_id, SAMPLE_RATE,
)


@dataclass
class Tree:
    tree_id: str
    numeric_id: int = field(init=False)
    distance_m: float = field(init=False)
    direction: Optional[str] = field(init=False)

    species_short: str = field(init=False)
    species_long: str = field(init=False)
    is_needleleaf: bool = field(init=False)
    is_broadleaf: bool = field(init=False)

    recording: Optional[slab.Sound] = field(default=None, repr=False)
    reference: Optional[slab.Sound] = field(default=None, repr=False)
    reference_baselined: Optional[slab.Sound] = field(default=None, repr=False)

    recording_trimmed: Optional[slab.Sound] = field(default=None, repr=False)
    reference_trimmed: Optional[slab.Sound] = field(default=None, repr=False)
    trim_start_idx: int = 0

    ir_full: Optional[pyfar.Signal] = field(default=None, repr=False)
    ir_windowed: Optional[pyfar.Signal] = field(default=None, repr=False)
    window: Optional[pyfar.Signal] = field(default=None, repr=False)

    freqs: Optional[np.ndarray] = field(default=None, repr=False)
    tf_magnitude_db: Optional[np.ndarray] = field(default=None, repr=False)

    # Attenuation metrics (generic bands — unchanged)
    attenuation_overall: Optional[float] = None
    attenuation_low: Optional[float] = None
    attenuation_mid: Optional[float] = None
    attenuation_high: Optional[float] = None

    # Sonification: scenario sounds filtered through this tree's IR
    # {"highway": np.ndarray, "tram": np.ndarray, ...}
    filtered_scenarios: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
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
        return self.recording.samplerate if self.recording else SAMPLE_RATE

    @property
    def leaf_type(self) -> str:
        return "needleleaf" if self.is_needleleaf else "broadleaf"

    def compute_attenuation_metrics(self):
        """Generic low/mid/high/overall attenuation — unchanged."""
        if self.freqs is None or self.tf_magnitude_db is None:
            return
        f, db = self.freqs, self.tf_magnitude_db
        self.attenuation_overall = float(np.mean(db[(f >= FREQ_LOW) & (f <= FREQ_HIGH)]))
        self.attenuation_low = float(np.mean(db[(f >= BAND_LOW[0]) & (f < BAND_LOW[1])]))
        self.attenuation_mid = float(np.mean(db[(f >= BAND_MID[0]) & (f < BAND_MID[1])]))
        self.attenuation_high = float(np.mean(db[(f >= BAND_HIGH[0]) & (f <= BAND_HIGH[1])]))

    def to_dict(self) -> dict:
        """Same columns as before — no scenario columns."""
        return {
            "tree_id": self.tree_id, "numeric_id": self.numeric_id,
            "distance_m": self.distance_m, "direction": self.direction,
            "species_short": self.species_short, "species_long": self.species_long,
            "is_needleleaf": self.is_needleleaf, "is_broadleaf": self.is_broadleaf,
            "leaf_type": self.leaf_type,
            "attenuation_overall_db": self.attenuation_overall,
            "attenuation_low_db": self.attenuation_low,
            "attenuation_mid_db": self.attenuation_mid,
            "attenuation_high_db": self.attenuation_high,
        }

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "Tree":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        if self.attenuation_overall is not None:
            return f"Tree({self.tree_id}, {self.species_short}, att={self.attenuation_overall:.1f}dB)"
        return f"Tree({self.tree_id})"
