# Tree Sound Absorption Analysis Pipeline

Analysis pipeline for investigating how structural and leaf traits of trees influence sound attenuation. Includes interactive sonification with urban noise scenarios and real-time slider tools with live frequency spectrum visualization.

## Overview

This project combines three data sources:

1. **Acoustic measurements** — Transfer functions from speaker–microphone recordings through 17 tree canopies (10 species, Großpösna Arboretum)
2. **Structural traits** — 3D metrics from terrestrial laser scanning (TLS)
3. **Leaf traits** — Morphological measurements (leaf area, thickness, LDMC, toughness)

The pipeline produces statistical models, per-tree sonifications with 4 urban noise scenarios, and interactive slider tools where you can hear and see how tree traits shape sound filtering.

## Quick Start

```bash
# 1. Acoustic processing + sonification
python run_analysis.py --sounds-dir sounds/

# 2. Structural traits from TLS (R)
Rscript extract_structural_traits.R --data-dir ~/DATA/Arboretum --output-dir output

# 3. Merge all data (leaf traits loaded automatically from config.py)
python run_correlation.py --output-dir output

# 4. Statistical modeling
Rscript run_stepwise_modeling.R --data output/csv/merged_data.csv --output output/figures/models

# 5. Generate interactive sliders
python run_analysis.py --slider-only --model-csv output/figures/models/stepwise_coefficients.csv
```

## Viewing Results

After running the pipeline, there are three ways to explore the results: the presentation, the sliders, and the filtered audio files. All three require a local HTTP server because browsers block local file access for audio.

### Start the server (do this first)

```bash
# From the project root directory:
python -m http.server 8000
```

Leave this running in a terminal. All URLs below assume this server is active.

### View the presentation

The presentation is a Quarto reveal.js slideshow. To render it:

```bash
# Install Quarto if needed: https://quarto.org/docs/get-started/
quarto render TreeSoundAbsorption_Presentation_EN.qmd
```

Then open in your browser:

```
http://localhost:8000/TreeSoundAbsorption_Presentation_EN.html
```

The presentation contains:

- Processing pipeline with formulas
- Transfer function plots (per-tree and overlaid by leaf type)
- Attenuation heatmap across frequency bands
- Structural trait extraction with gap fraction projections
- Statistical modeling results (predictor consensus, model accuracy)
- Best model equations for the slider
- Comprehensive limitations (16 problems with proposed fixes)
- **Interactive sonification map** — select a noise scenario (highway / tram / construction / children) from the dropdown, click a tree dot to hear that noise filtered through the tree
- Links to the interactive sliders

### Open the interactive sliders

With the server running, open directly:

```
http://localhost:8000/output/sliders/slider_broadleaf.html
http://localhost:8000/output/sliders/slider_needleleaf.html
```

Each slider page has:

- **Trait sliders** — one per predictor in the fitted models (e.g., LDMC, ENL, leaf thickness). Moving a slider recalculates predicted attenuation per frequency band.
- **Scenario dropdown** — switch between highway, tram, construction, and children as the audio source.
- **Play button** — starts playing the selected scenario through a 3-band parametric EQ (low shelf @ 250 Hz, peaking @ 1 kHz, high shelf @ 6 kHz) whose gains are set by the model predictions.
- **Frequency spectrum canvas** — shows the original scenario spectrum (dashed grey) and the filtered spectrum (solid green) on a log-frequency axis. The shaded area between the curves shows attenuation (green) or boost (red). Moving sliders changes the green curve in real time.
- **Band gain display** — shows the predicted dB value for each band (low / mid / high), color-coded green (attenuation) or red (boost).

The spectrum is computed via FFT from the actual scenario WAV when it's first loaded, then cached. Switching scenarios updates both the audio and the spectrum instantly.

### Listen to filtered audio files directly

All per-tree filtered WAVs are in `output/wav/`:

```
output/wav/270_5.4_240SW_filtered_highway.wav
output/wav/270_5.4_240SW_filtered_tram.wav
output/wav/270_5.4_240SW_filtered_construction.wav
output/wav/270_5.4_240SW_filtered_children.wav
```

Plus unfiltered references for A/B comparison:

```
output/wav/highway_unfiltered.wav
output/wav/tram_unfiltered.wav
output/wav/construction_unfiltered.wav
output/wav/children_unfiltered.wav
```

## Directory Structure

```
tree_acoustics/
├── config.py                        # Paths, species maps, band definitions, colors
├── tree.py                          # Tree dataclass
├── processing.py                    # Signal processing (baseline, trim, deconvolve, window)
├── leaf_traits.py                   # Leaf trait loading from Excel
├── visualization.py                 # Plotting functions
├── noise_profiles.py                # Converts scenario MP3s to WAV and loads them
├── slider.py                        # Interactive slider HTML generator (parametric EQ + spectrum)
├── run_analysis.py                  # Main pipeline: acoustic processing + sonification
├── run_correlation.py               # Merge acoustic + structural + leaf data
├── extract_structural_traits.R      # TLS point cloud → structural metrics (R)
├── run_stepwise_modeling.R          # Forward stepwise regression (R)
│
├── sounds/                          # Urban noise recordings (MP3)
│   ├── highway.mp3
│   ├── tram.mp3
│   ├── construction.mp3
│   └── children.mp3
│
├── data/                            # Acoustic recordings
│   ├── ref/ref_rec.wav
│   └── {tree_id}/{tree_id}_rec.wav
│
├── output/
│   ├── pkl/                         # Pickled Tree objects
│   ├── wav/                         # Filtered audio (4 scenarios × 17 trees + 4 references)
│   ├── csv/                         # tree_acoustic_summary.csv, tree_structural_traits.csv, merged_data.csv
│   ├── figures/
│   │   ├── transfer_functions/      # Per-tree TF plots
│   │   ├── ir_comparison/           # IR windowing comparisons
│   │   ├── combined/                # TF overlays, heatmaps
│   │   ├── structural/              # Top-down + projection plots
│   │   └── models/                  # Stepwise results, consensus, accuracy, coefficients
│   └── sliders/
│       ├── slider_broadleaf.html
│       └── slider_needleleaf.html
│
├── TreeSoundAbsorption_Presentation_EN.qmd
├── Protocol.qmd
├── Arboretum_Choir.png
├── Arboretum_front.png
└── field_setup.jpg
```

## Pipeline Steps in Detail

### Step 1: Acoustic Processing + Sonification

```bash
python run_analysis.py --sounds-dir sounds/
```

Converts scenario MP3s to WAV (via ffmpeg), processes all 17 trees through the acoustic pipeline (baseline → trim → deconvolve → window → TF → band attenuation), and filters each scenario through each tree's IR.

**Outputs:** `output/csv/tree_acoustic_summary.csv`, per-tree WAVs, per-tree figures, pickled Tree objects.

### Step 2: Structural Traits (R)

```bash
Rscript extract_structural_traits.R --data-dir ~/DATA/Arboretum --output-dir output
```

Processes TLS point clouds into height, crown volume, gap fraction, ENL, FHD, LAD, etc.

### Step 3: Merge Data

```bash
python run_correlation.py --output-dir output
```

Merges acoustic + structural + leaf traits into `merged_data.csv`. Leaf traits are loaded automatically from the path in `config.py` (`LEAF_XLSX`). Override with `--leaf-xlsx /other/path.xlsx`.

### Step 4: Statistical Modeling (R)

```bash
Rscript run_stepwise_modeling.R --data output/csv/merged_data.csv --output output/figures/models
```

Forward stepwise selection with 4 criteria, separately for broadleaf (n=9), needleleaf (n=8), and mixed (n=17).

**Key outputs:** `stepwise_summary.csv`, `stepwise_coefficients.csv` (for sliders), `combined_predictor_consensus.png`, `combined_model_accuracy.png`.

### Step 5: Generate Sliders

```bash
python run_analysis.py --slider-only --model-csv output/figures/models/stepwise_coefficients.csv
```

Reads coefficients from R, loads actual predictor ranges from `merged_data.csv`, and generates interactive HTML pages.

## Key Variables

### Response Variables (Acoustic)
| Variable | Band | Range (Hz) | Context |
|----------|------|------------|---------|
| `attenuation_low_db` | Low | 125–500 | Traffic, machinery |
| `attenuation_mid_db` | Mid | 500–2,000 | Speech |
| `attenuation_high_db` | High | 2,000–18,000 | Broadband |
| `attenuation_overall_db` | Overall | 125–18,000 | Full range |

### Predictors: Structural (from TLS)
| Variable | Description |
|----------|-------------|
| `height_m` | Tree height (99th–1st percentile) |
| `crown_volume_m3` | 3D convex hull volume |
| `vol_fill_fraction` | Occupied voxels / hull voxels |
| `gap_fraction` | Empty area in speaker-direction projection |
| `leaf_voxel_density` | Voxel count at measurement height |
| `ENL` | Effective Number of Layers (Simpson diversity) |
| `FHD` | Foliage Height Diversity (Shannon entropy) |
| `LAD` | Leaf Area Density (Beer–Lambert, k=0.5) |

### Predictors: Leaf Traits (from lab)
| Variable | Description |
|----------|-------------|
| `leaf_area_cm2` | Mean leaf area |
| `leaf_thickness_mm` | Mean thickness (5 measurements) |
| `ldmc` | Leaf Dry Matter Content (dry/fresh weight) |
| `toughness` | Mean toughness (5 measurements) |

### Excluded Precursors
`densest_slice_height_m`, `distance_m`, `pts_per_voxel`, `leaf_fresh_weight_g`, `leaf_dry_weight_g` — computed but excluded from modeling.

## Noise Scenarios

| Scenario | File | Description |
|----------|------|-------------|
| Highway | `sounds/highway.mp3` | Highway traffic |
| Tram | `sounds/tram.mp3` | Tram / urban street |
| Construction | `sounds/construction.mp3` | Construction site |
| Children | `sounds/children.mp3` | Playground / children |

These are used for sonification (filtering through tree IRs) and as audio sources in the interactive sliders. The statistical modeling uses generic frequency bands (low/mid/high/overall), not scenario-specific bands.

## Dependencies

### Python
```
numpy scipy pandas matplotlib seaborn
pyfar slab soundfile
scikit-learn openpyxl
```

### R
```
lidR VoxR data.table dplyr
ggplot2 patchwork optparse
ggrepel  # optional
```

### System
- **ffmpeg** — MP3 → WAV conversion
- **quarto** — presentation rendering (https://quarto.org)

## Species

| Leaf type | Species | Tree IDs |
|-----------|---------|----------|
| Needleleaf | *Larix decidua* | 227, 281 |
| | *Pseudotsuga menziesii* | 257, 342 |
| | *Pinus nigra* | 332, 333 |
| | *Abies grandis* | 298 |
| | *Cedrus deodara* | 502 |
| Broadleaf | *Populus tremula* | 247, 274 |
| | *Prunus avium* | 277, 232 |
| | *Tilia tomentosa* | 467, 499, 518 |
| | *Alnus glutinosa* | 270 |
| | *Salix caprea* | 327 |

Trees 313, 344, 353 excluded (recorded with linear sweeps).

## Color Scheme

| Color | Hex | Use |
|-------|-----|-----|
| Rose | `#CC6677` | Broadleaf |
| Indigo | `#332288` | Needleleaf |
| Purple | `#AA4499` | Mixed (red + blue) |
| Forest | `#117733` | Canopy / structural plots |
| Gold | `#DDCC77` | Highlights / markers |
