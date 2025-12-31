"""
Configuration for Tree Sound Absorption Analysis (Python Pipeline)
==================================================================
Central configuration file with paths, species mappings, and processing parameters.

Note: R scripts (extract_structural_traits.R, run_stepwise_modeling.R) have their
own configuration sections - this file is for Python modules only.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ------------------------------------------------
# PATHS - Adjust these to your system
# ------------------------------------------------
WORK_DIR = Path(__file__).parent.resolve()
DATA_DIR = WORK_DIR / "data"

# Leaf morphology data
LEAF_XLSX = "/home/tigris/DATA/Arboretum/021_003_017_leaf_morphology_2023_2025.xlsx"

# ------------------------------------------------
# OUTPUT DIRECTORIES
# ------------------------------------------------
OUT_DIR = WORK_DIR / "output"

OUT_PKL = OUT_DIR / "pkl"           # Pickle files (Tree objects)
OUT_WAV = OUT_DIR / "wav"           # Audio outputs (filtered noise, sonifications)
OUT_CSV = OUT_DIR / "csv"           # CSV data files

# Figure directories
OUT_FIG = OUT_DIR / "figures"
OUT_FIG_TF = OUT_FIG / "transfer_functions"
OUT_FIG_IR = OUT_FIG / "ir_comparison"
OUT_FIG_COMBINED = OUT_FIG / "combined"


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    dirs = [OUT_PKL, OUT_WAV, OUT_CSV, OUT_FIG_TF, OUT_FIG_IR, OUT_FIG_COMBINED]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# ACOUSTIC PROCESSING PARAMETERS
# ------------------------------------------------
SAMPLE_RATE = 48828  # Hz (TDT processor)
FREQ_LOW = 125       # Hz - lower bound of analysis
FREQ_HIGH = 18000    # Hz - upper bound

# Band definitions for attenuation metrics (Hz)
BAND_LOW = (125, 500) # traffic, machinery
BAND_MID = (500, 2000) # human speech
BAND_HIGH = (2000, 18000) # bird songs

# Sonification parameters
WHITE_NOISE_DURATION_S = 3.0
PLAYGROUND_NOISE_FILE = "playground.wav"  # Any length works; checked in OUT_WAV then DATA_DIR

# ------------------------------------------------
# TREE IDS - All trees with acoustic measurements
# ------------------------------------------------
TREE_IDS = [
    "227_6.6_320NW",
    "232_6.2_230SW",
    "247_8.2_318NW",
    "257_5_273W",
    "270_5.4_240SW",
    "274_5.1_250W",
    "277_6.6_120SO",
    "281_5.8_190S",
    "298_4.5_336NW",
    "327_9_170S",
    "332_5_55NO",
    "333_6_325NW",
    "342_6.2_80O",
    "467_6.1_270W",
    "499_4.6_40NO",
    "502_3.8_46NO",
    "518_4.6_0N",
]

# Trees to exclude (linear sweeps)
EXCLUDE_IDS = [313, 344, 353]

# ------------------------------------------------
# SPECIES MAPPINGS
# ------------------------------------------------
SPECIES_MAP_SHORT = {
    "227": "Lar dec", "232": "Pru avi", "247": "Pop tre", "257": "Pse men",
    "270": "Aln glu", "274": "Pop tre", "277": "Pru avi", "281": "Lar dec",
    "298": "Abi gra", "313": "Aln glu", "327": "Sal cap", "332": "Pin nig",
    "333": "Pin nig", "342": "Pse men", "344": "Abi gra", "353": "Sal cap",
    "467": "Til tom", "499": "Til tom", "502": "Ced deo", "518": "Til tom",
}

SPECIES_LONG_MAP = {
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

# Leaf type classification
NEEDLELEAF_SPECIES = {"Lar dec", "Pse men", "Pin nig", "Abi gra", "Ced deo"}
BROADLEAF_SPECIES = {"Pop tre", "Pru avi", "Til tom", "Aln glu", "Sal cap"}

# ------------------------------------------------
# COLORBLIND-FRIENDLY PALETTE
# Color logic: red(broad) + blue(needle) = purple(mixed)
# ------------------------------------------------
CB_ROSE = "#CC6677"       # Broadleaf (reddish)
CB_INDIGO = "#332288"     # Needleleaf (bluish)
CB_PURPLE = "#AA4499"     # Mixed analysis (purple = red + blue)
CB_FOREST = "#117733"     # Tree/vegetation (darker)
CB_GOLD = "#DDCC77"       # Highlights/markers


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

def parse_tree_id(tree_id: str) -> dict:
    """
    Parse tree ID string into components.
    
    Example: "270_5.4_240SW" -> {
        "tree_id": "270_5.4_240SW",
        "numeric_id": 270,
        "distance_m": 5.4,
        "direction": "240SW"
    }
    """
    parts = tree_id.split("_")
    return {
        "tree_id": tree_id,
        "numeric_id": int(parts[0]),
        "distance_m": float(parts[1]),
        "direction": parts[2] if len(parts) > 2 else None,
    }


def get_species_info(tree_id: str) -> dict:
    """Get species information for a tree."""
    numeric_id = str(parse_tree_id(tree_id)["numeric_id"])
    short = SPECIES_MAP_SHORT.get(numeric_id, "Unknown")
    return {
        "species_short": short,
        "species_long": SPECIES_LONG_MAP.get(short, "Unknown"),
        "is_needleleaf": short in NEEDLELEAF_SPECIES,
        "is_broadleaf": short in BROADLEAF_SPECIES,
    }
