"""
Configuration for Tree Sound Absorption Analysis (Python Pipeline)
==================================================================
Central configuration file with paths, species mappings, and processing parameters.

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

OUT_PKL = OUT_DIR / "pkl"
OUT_WAV = OUT_DIR / "wav"
OUT_CSV = OUT_DIR / "csv"

# Figure directories
OUT_FIG = OUT_DIR / "figures"
OUT_FIG_TF = OUT_FIG / "transfer_functions"
OUT_FIG_IR = OUT_FIG / "ir_comparison"
OUT_FIG_COMBINED = OUT_FIG / "combined"
OUT_FIG_SLIDER = OUT_FIG / "sliders"


def ensure_output_dirs():
    """Create all output directories if they don't exist."""
    dirs = [OUT_PKL, OUT_WAV, OUT_CSV, OUT_FIG_TF, OUT_FIG_IR,
            OUT_FIG_COMBINED, OUT_FIG_SLIDER]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# ACOUSTIC PROCESSING PARAMETERS
# ------------------------------------------------
SAMPLE_RATE = 48828  # Hz (TDT processor)
FREQ_LOW = 125       # Hz - lower bound of analysis
FREQ_HIGH = 18000    # Hz - upper bound


# Sound profiles: no band extraction — the acoustic analysis uses generic low/mid/high/overall.
BAND_LOW = (125, 500)
BAND_MID = (500, 2000)
BAND_HIGH = (2000, 18000)

# Sonification parameters
WHITE_NOISE_DURATION_S = 3.0
PLAYGROUND_NOISE_FILE = "playground.wav"

# ------------------------------------------------
# TREE IDS
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

NEEDLELEAF_SPECIES = {"Lar dec", "Pse men", "Pin nig", "Abi gra", "Ced deo"}
BROADLEAF_SPECIES = {"Pop tre", "Pru avi", "Til tom", "Aln glu", "Sal cap"}

# ------------------------------------------------
# COLORBLIND-FRIENDLY PALETTE
# ------------------------------------------------
CB_ROSE = "#CC6677"
CB_INDIGO = "#332288"
CB_PURPLE = "#AA4499"
CB_FOREST = "#117733"
CB_GOLD = "#DDCC77"


# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

def parse_tree_id(tree_id: str) -> dict:
    parts = tree_id.split("_")
    return {
        "tree_id": tree_id,
        "numeric_id": int(parts[0]),
        "distance_m": float(parts[1]),
        "direction": parts[2] if len(parts) > 2 else None,
    }


def get_species_info(tree_id: str) -> dict:
    numeric_id = str(parse_tree_id(tree_id)["numeric_id"])
    short = SPECIES_MAP_SHORT.get(numeric_id, "Unknown")
    return {
        "species_short": short,
        "species_long": SPECIES_LONG_MAP.get(short, "Unknown"),
        "is_needleleaf": short in NEEDLELEAF_SPECIES,
        "is_broadleaf": short in BROADLEAF_SPECIES,
    }
