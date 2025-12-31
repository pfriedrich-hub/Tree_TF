# Tree Sound Absorption Analysis Pipeline

Analysis pipeline for investigating structural and leaf traits that influence sound attenuation by tree canopies.

## Overview

This project combines:
1. **Acoustic measurements**: Transfer functions from speaker-microphone recordings through tree canopies
2. **Structural traits**: 3D metrics from terrestrial laser scanning (TLS)
3. **Leaf traits**: Morphological measurements (area, thickness, toughness, etc.)

## Directory Structure

```
tree_acoustics/
├── config.py                    # Python configuration (paths, species, colors)
├── tree.py                      # Tree dataclass
├── processing.py                # Audio/signal processing functions
├── leaf_traits.py               # Leaf trait loading from Excel
├── visualization.py             # Plotting functions
├── run_analysis.py              # Main acoustic processing pipeline
├── run_correlation.py           # Data merging & correlation analysis
├── extract_structural_traits.R  # TLS processing (R)
├── run_stepwise_modeling.R      # Statistical modeling (R)
├── data/                        # Input recordings
│   ├── ref/                     # Reference recordings
│   └── {tree_id}/               # Per-tree recordings
└── output/                      # Generated outputs
    ├── pkl/                     # Pickled Tree objects
    ├── wav/                     # Filtered audio (sonifications)
    ├── csv/                     # Data summaries
    └── figures/                 # All plots
```

## Usage

### Step 1: Process Acoustic Data (Python)

```bash
python run_analysis.py --data-dir ./data --output-dir ./output
```

Outputs:
- `output/pkl/{tree_id}.pkl` - Tree objects with TFs
- `output/wav/{tree_id}_filtered_noise.wav` - White noise through tree filter
- `output/wav/{tree_id}_filtered_playground.wav` - Playground noise through tree filter (if playground.wav exists)
- `output/csv/tree_acoustic_summary.csv` - Attenuation metrics
- `output/figures/transfer_functions/` - Per-tree TF plots

### Step 2: Extract Structural Traits (R)

```bash
Rscript extract_structural_traits.R --data-dir /path/to/LAS/files --output-dir ./output
```

Outputs:
- `output/csv/tree_structural_traits.csv` - Height, crown volume, gap fraction, ENL, FHD, LAD
- `output/figures/structural/` - Top-down and projection plots per tree

### Step 3: Correlation Analysis (Python)

```bash
python run_correlation.py --output-dir ./output --leaf-xlsx /path/to/021_003_017_leaf_morphology_2023_2025.xlsx
```

Merges acoustic + structural + leaf trait data, checks collinearity, creates heatmaps.

Outputs:
- `output/csv/merged_data.csv` - Combined dataset with all traits
- `output/figures/combined/` - Correlation heatmaps, TF overlays by leaf type

### Step 4: Statistical Modeling (R)

```bash
Rscript run_stepwise_modeling.R --data output/csv/merged_data.csv --output output/figures/models
```

Stepwise forward selection with multiple criteria (Adj.R², F-test, LOOCV, BIC).

Outputs:
- `output/figures/models/stepwise_summary.csv` - All model results
- `output/figures/models/model_summary_*.png` - Heatmaps of final models
- `output/figures/models/model_fits_*.png` - Scatter plots with fits
- `output/figures/models/stepwise_*.png` - Forward selection step by step
- `output/figures/models/predictor_correlations_*.png` - multicollinearity

## Dependencies

### Python
- numpy, scipy, pandas
- matplotlib, seaborn
- pyfar, slab, soundfile
- scikit-learn
- openpyxl (for Excel leaf traits)

### R
- lidR, VoxR (TLS processing)
- data.table, dplyr
- ggplot2, patchwork
- optparse
- ggrepel (optional, for label placement)

## Key Variables

### Acoustic Metrics
- `attenuation_overall_db`: Mean attenuation 125-18000 Hz
- `attenuation_low_db`: Mean attenuation 125-500 Hz
- `attenuation_mid_db`: Mean attenuation 500-2000 Hz
- `attenuation_high_db`: Mean attenuation 2000-18000 Hz

### Structural Traits
- `height_m`: Tree height (99th percentile)
- `crown_volume_m3`: Convex hull volume
- `vol_fill_fraction`: Occupied voxels / hull voxels
- `gap_fraction`: Empty area in speaker view projection
- `leaf_voxel_density`: Point density in leaf voxels
- `ENL`: Effective Number of Layers (vertical complexity)
- `FHD`: Foliage Height Diversity (Shannon entropy)
- `LAD`: Leaf Area Density (from gap fraction via Beer-Lambert)

### Leaf Traits
- `leaf_area_cm2`: Mean leaf area
- `leaf_thickness_mm`: Leaf thickness
- `ldmc`: Leaf Dry Matter Content (dry weight / fresh weight)
- `toughness`: Leaf toughness

### Precursor Variables (computed but excluded from analysis)
- `densest_slice_height_m`: Used to determine measurement height
- `distance_m`: Speaker-microphone distance (measurement setup)
- `pts_per_voxel`: TLS scanning artifact, not biologically meaningful
- `leaf_fresh_weight_g`, `leaf_dry_weight_g`: Used to compute LDMC

## Sonification

The pipeline creates "sonified" versions of each tree's acoustic filter:
- **White noise**: 3-second white noise filtered through the tree's transfer function
- **Playground noise**: Optional real-world recording (any length) filtered through the tree

Place `playground.wav` in the output/wav/ or data/ directory.

## Color Scheme

Colorblind-friendly palette with logical color mixing:
- **Rose (#CC6677)**: Broadleaf species (reddish)
- **Indigo (#332288)**: Needleleaf species (bluish)
- **Purple (#AA4499)**: Mixed/combined analysis (red + blue = purple)
- **Forest (#117733)**: Tree canopy in structural plots
- **Gold (#DDCC77)**: Markers/highlights in structural plots

## Species

**Needleleaf (5):** Larix decidua, Pseudotsuga menziesii, Pinus nigra, Abies grandis, Cedrus deodara

**Broadleaf (5):** Populus tremula, Prunus avium, Tilia tomentosa, Alnus glutinosa, Salix caprea

## Notes

- Trees 313, 344, 353 are excluded (recorded with linear sweeps instead of logarithmic)
- Arrow length in top-down plots = distance from tree center to first canopy hit point (sound path length)
