# Analysis of Tree Acoustic Measurements

This repository contains code and tools for recording and analyzing the acoustic attenuation properties of trees.

---

## Overview

- **`main.py`**
  Record acoustic data and perform a quick analysis in the field.

- **`homogenize.py` & `analyse.py` / `test_direct_refs.py`**
  Post-recording analysis scripts for processing and comparing recordings using different reference signals.

- **`record_tf.py`**
  Contains utility functions for recording and handling transfer functions.

- **`tf.py`**
  Example script used for teaching the theory behind transfer functions and acoustic filtering.

- **`clean_pkls.py`**
  Can be used for cleaning up data dirs, if correct pkls cannot be found otherwise.

- **`comb_filter.py`**
  Interactive Visualization of impulse response and transfer function

---

## Usage

1. **Record data in the field:**
   ```bash
   python main.py
   ```

2. **Run post-processing after recordings:**

   ```bash
   python homogenize.py
   python analyse.py
   # or
   python test_direct_refs.py
   ```

3. **Explore the theory with:**

   ```bash
   python tf.py
   # and
   python comb_filter.py
   ```

---

## Notes

* Make sure all dependencies are installed (e.g. NumPy, SciPy, matplotlib, etc.). Have a look at the environmetn.yml.
* Data output and plots will be saved in respective folders (e.g. `figures/`).
* figures_0line shows figures before trimming the signals, with a fixed window length of 120ms for a distance scaled and median scaled reference
