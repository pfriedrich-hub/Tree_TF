"""
Slider Module
=============
Parametric equalizer driven by fitted linear models.

Takes model coefficients (from R stepwise output) and creates:
1. Offline WAV generation: predict attenuation per band, shape filter, apply to audio
2. HTML slider widget: standalone HTML page with Web Audio API for real-time interaction

Usage:
    from slider import SliderModel, SliderBank, generate_slider_html

    # Define a model for one band
    low_model = SliderModel(
        band_label="engine_rumble",
        center_hz=125,
        filter_type="lowshelf",
        coefficients={"intercept": -5.2, "ldmc": -18.3, "ENL": 2.1},
        predictor_ranges={"ldmc": (0.25, 0.55), "ENL": (1.5, 6.0)},
    )

    # Combine into a bank for one leaf type
    bank = SliderBank(
        leaf_type="broadleaf",
        models=[low_model, mid_model, high_model],
    )

    # Generate HTML slider page
    generate_slider_html(bank, "broadleaf_slider.html")
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SliderModel:
    """
    One frequency band's linear model: predicted_db = intercept + sum(coef_i * x_i).

    Parameters
    ----------
    band_label : str
        Name matching the noise profile band (e.g. "engine_rumble")
    center_hz : float
        Center frequency for the parametric EQ filter
    filter_type : str
        "lowshelf", "peaking", or "highshelf"
    coefficients : dict
        {"intercept": float, "predictor_name": float, ...}
    predictor_ranges : dict
        {"predictor_name": (min, max), ...} for slider bounds
    adj_r2 : float
        Model fit quality (for display)
    criterion : str
        Which selection criterion produced this model
    q_value : float
        Q factor for peaking filter (ignored for shelves)
    """
    band_label: str
    center_hz: float
    filter_type: str  # "lowshelf" | "peaking" | "highshelf"
    coefficients: dict
    predictor_ranges: dict
    adj_r2: float = 0.0
    criterion: str = ""
    q_value: float = 0.7  # default Q for peaking

    @property
    def predictor_names(self) -> list:
        """Return predictor names (excluding intercept)."""
        return [k for k in self.coefficients if k != "intercept"]

    def predict(self, values: dict) -> float:
        """
        Predict attenuation in dB from predictor values.

        Parameters
        ----------
        values : dict
            {"predictor_name": float, ...}

        Returns
        -------
        float
            Predicted attenuation in dB
        """
        result = self.coefficients.get("intercept", 0.0)
        for name, coef in self.coefficients.items():
            if name == "intercept":
                continue
            if name in values:
                result += coef * values[name]
            else:
                # Use midpoint of range as default
                lo, hi = self.predictor_ranges.get(name, (0, 1))
                result += coef * (lo + hi) / 2
        return result


@dataclass
class SliderBank:
    """
    Collection of band models for one leaf type.

    Combines 2-3 SliderModels (one per frequency band) into a unified
    parametric EQ that can be controlled by trait sliders.
    """
    leaf_type: str
    noise_profile_key: str = "generic"
    models: list = field(default_factory=list)

    @property
    def all_predictors(self) -> list:
        """Unique predictors across all band models, sorted."""
        preds = set()
        for m in self.models:
            preds.update(m.predictor_names)
        return sorted(preds)

    @property
    def all_predictor_ranges(self) -> dict:
        """Merged predictor ranges across all models."""
        ranges = {}
        for m in self.models:
            for name, (lo, hi) in m.predictor_ranges.items():
                if name in ranges:
                    old_lo, old_hi = ranges[name]
                    ranges[name] = (min(lo, old_lo), max(hi, old_hi))
                else:
                    ranges[name] = (lo, hi)
        return ranges

    def predict_all_bands(self, values: dict) -> dict:
        """Predict attenuation for each band given predictor values."""
        return {m.band_label: m.predict(values) for m in self.models}

    def to_json(self) -> dict:
        """Export bank config as JSON (for HTML widget)."""
        return {
            "leaf_type": self.leaf_type,
            "noise_profile": self.noise_profile_key,
            "predictors": {
                name: {
                    "min": rng[0],
                    "max": rng[1],
                    "default": (rng[0] + rng[1]) / 2,
                }
                for name, rng in self.all_predictor_ranges.items()
            },
            "models": [
                {
                    "band_label": m.band_label,
                    "center_hz": m.center_hz,
                    "filter_type": m.filter_type,
                    "q_value": m.q_value,
                    "coefficients": m.coefficients,
                    "adj_r2": m.adj_r2,
                    "criterion": m.criterion,
                }
                for m in self.models
            ],
        }


# ============================================================
# OFFLINE AUDIO GENERATION
# ============================================================

def apply_parametric_eq(
    audio: np.ndarray,
    sr: int,
    band_gains_db: dict,
    models: list,
) -> np.ndarray:
    """
    Apply parametric EQ to audio based on predicted band gains.

    Uses scipy biquad filters: low shelf, peaking, high shelf.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    band_gains_db : dict
        {band_label: gain_db} from SliderBank.predict_all_bands()
    models : list
        List of SliderModel (for filter config)

    Returns
    -------
    np.ndarray
        Filtered audio
    """
    from scipy.signal import sosfilt

    result = audio.copy().astype(np.float64)

    for m in models:
        gain_db = band_gains_db.get(m.band_label, 0.0)
        if abs(gain_db) < 0.1:  # skip near-zero gains
            continue

        sos = _biquad_sos(m.filter_type, m.center_hz, gain_db, sr, m.q_value)
        result = sosfilt(sos, result)

    # Normalize
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.9

    return result.astype(np.float32)


def _biquad_sos(filter_type: str, freq: float, gain_db: float, sr: int, q: float = 0.7):
    """
    Compute second-order section for a biquad filter.

    Based on Robert Bristow-Johnson's Audio EQ Cookbook.
    """
    A = 10 ** (gain_db / 40)  # amplitude
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    if filter_type == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    elif filter_type == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Normalize
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sos


# ============================================================
# HTML SLIDER WIDGET
# ============================================================

def generate_slider_html(bank: SliderBank, output_path: str, wav_dir: str = "../wav"):
    """
    Generate standalone HTML with trait sliders + scenario sound selector.

    The page loads one of 4 scenario WAVs (highway, tram, construction, children)
    and applies a real-time parametric EQ (low shelf + peaking + high shelf)
    controlled by trait sliders. A dropdown switches which sound is playing.

    Parameters
    ----------
    bank : SliderBank
        Model configuration
    output_path : str
        Path for output HTML file
    wav_dir : str
        Relative path from HTML location to the wav directory
    """
    config = json.dumps(bank.to_json(), indent=2)

    pred_labels = {
        "ldmc": "LDMC (dry/fresh)",
        "ENL": "Effective No. Layers",
        "FHD": "Foliage Height Diversity",
        "LAD": "Leaf Area Density",
        "leaf_thickness_mm": "Leaf Thickness (mm)",
        "leaf_area_cm2": "Leaf Area (cm²)",
        "crown_volume_m3": "Crown Volume (m³)",
        "gap_fraction": "Gap Fraction",
        "height_m": "Height (m)",
        "vol_fill_fraction": "Vol. Fill Fraction",
        "leaf_voxel_density": "Voxel Density",
        "toughness": "Toughness",
    }

    scenarios_json = json.dumps({
        "highway": f"{wav_dir}/highway_unfiltered.wav",
        "tram": f"{wav_dir}/tram_unfiltered.wav",
        "construction": f"{wav_dir}/construction_unfiltered.wav",
        "children": f"{wav_dir}/children_unfiltered.wav",
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tree Sound Filter — {bank.leaf_type.title()}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: 2em auto; background: #fafafa; }}
  h1 {{ color: #2d5016; }}
  .controls {{ display: flex; gap: 1em; align-items: center; margin: 1em 0; }}
  .play-btn {{ font-size: 1.2em; padding: 0.6em 2em; border: none; border-radius: 6px; cursor: pointer; background: #2d5016; color: white; }}
  .play-btn:hover {{ background: #3a6b1e; }}
  select {{ font-size: 1em; padding: 0.5em 1em; border-radius: 6px; border: 1px solid #ccc; }}
  .slider-group {{ margin: 0.8em 0; padding: 1em; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .slider-group label {{ display: block; font-weight: 600; margin-bottom: 4px; }}
  .slider-group input[type=range] {{ width: 100%; }}
  .slider-value {{ float: right; font-family: monospace; color: #666; }}
  .band-display {{ display: flex; gap: 1em; margin: 1em 0; }}
  .band-box {{ flex: 1; padding: 0.8em; background: white; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .band-box .db {{ font-size: 1.8em; font-weight: 700; }}
  .band-box .label {{ font-size: 0.85em; color: #666; }}
  .eq-label {{ font-size: 0.75em; color: #999; }}
  .info {{ font-size: 0.85em; color: #888; margin-top: 2em; }}
  .status {{ font-size: 0.9em; color: #666; margin: 0.5em 0; }}
</style>
</head>
<body>

<h1>Tree Sound Filter — {bank.leaf_type.title()}</h1>
<p>Pick a noise scenario, move trait sliders, hear how trees filter urban sound.</p>

<div class="controls">
  <button class="play-btn" id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <select id="scenarioSelect" onchange="onScenarioChange()">
    <option value="highway">Highway traffic</option>
    <option value="tram">Tram / urban street</option>
    <option value="construction">Construction site</option>
    <option value="children">Playground / children</option>
  </select>
</div>

<div class="status" id="status">Select a scenario and press Play</div>

<canvas id="eqCanvas" width="760" height="200" style="width:100%;background:white;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);margin:1em 0;"></canvas>

<div class="band-display" id="bandDisplay"></div>

<div id="sliders"></div>

<div class="info">
  <p>Models: {', '.join(m.criterion + ' (R²=' + f'{m.adj_r2:.2f}' + ')' for m in bank.models)}</p>
  <p>Filter: 3-band parametric EQ (low shelf @ 250 Hz + peaking @ 1 kHz + high shelf @ 6 kHz)</p>
</div>

<script>
const CONFIG = {config};
const LABELS = {json.dumps(pred_labels)};
const SCENARIOS = {scenarios_json};

let audioCtx = null;
let sourceNode = null;
let filters = [];
let audioBuffers = {{}};
let spectrumCache = {{}};  // cached PSD per scenario
let isPlaying = false;
let currentScenario = 'highway';

// Compute smoothed PSD from an AudioBuffer (once per scenario)
function computeSpectrum(buffer) {{
  const data = buffer.getChannelData(0);
  const fftSize = 4096;
  const nBins = fftSize / 2;
  const sr = buffer.sampleRate;

  // Average multiple windows (Welch-style)
  const hop = fftSize / 2;
  const nWindows = Math.max(1, Math.floor((data.length - fftSize) / hop));
  const avgPower = new Float64Array(nBins);

  for (let w = 0; w < nWindows; w++) {{
    const offset = w * hop;
    const segment = new Float64Array(fftSize);
    for (let i = 0; i < fftSize; i++) {{
      // Hann window
      const hann = 0.5 * (1 - Math.cos(2 * Math.PI * i / fftSize));
      segment[i] = (data[offset + i] || 0) * hann;
    }}
    // Simple DFT at the log-spaced frequencies we care about
    // (too slow for full DFT — use the real FFT trick below)
    // Actually, just use the offline AnalyserNode approach:
    // We'll compute a simple periodogram instead
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    for (let i = 0; i < fftSize; i++) {{ re[i] = segment[i]; im[i] = 0; }}
    fftInPlace(re, im, fftSize);
    for (let k = 0; k < nBins; k++) {{
      avgPower[k] += (re[k] * re[k] + im[k] * im[k]) / nWindows;
    }}
  }}

  // Convert to dB, build freq axis
  const freqAxis = new Float64Array(nBins);
  const psdDb = new Float64Array(nBins);
  for (let k = 0; k < nBins; k++) {{
    freqAxis[k] = k * sr / fftSize;
    psdDb[k] = 10 * Math.log10(Math.max(avgPower[k], 1e-20));
  }}

  // Smooth (moving average, ~1/6 octave)
  const smoothed = new Float64Array(nBins);
  const halfW = 8;
  for (let k = 0; k < nBins; k++) {{
    let sum = 0, cnt = 0;
    for (let j = Math.max(0, k - halfW); j <= Math.min(nBins - 1, k + halfW); j++) {{
      sum += psdDb[j]; cnt++;
    }}
    smoothed[k] = sum / cnt;
  }}

  return {{ freqAxis, psdDb: smoothed }};
}}

// Simple in-place radix-2 FFT (Cooley-Tukey)
function fftInPlace(re, im, n) {{
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {{
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {{
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }}
  }}
  // FFT butterflies
  for (let len = 2; len <= n; len *= 2) {{
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {{
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {{
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        const vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[i + j + len/2] = uRe - vRe; im[i + j + len/2] = uIm - vIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }}
    }}
  }}
}}

async function init() {{
  // Slider UI
  const container = document.getElementById('sliders');
  for (const [name, cfg] of Object.entries(CONFIG.predictors)) {{
    const div = document.createElement('div');
    div.className = 'slider-group';
    const label = LABELS[name] || name;
    div.innerHTML = `
      <label>${{label}} <span class="slider-value" id="val_${{name}}">${{cfg.default.toFixed(3)}}</span></label>
      <input type="range" id="slider_${{name}}" min="${{cfg.min}}" max="${{cfg.max}}"
             step="${{((cfg.max - cfg.min) / 200).toFixed(6)}}" value="${{cfg.default}}"
             oninput="onSliderChange()">
    `;
    container.appendChild(div);
  }}

  // Band display
  const bandDiv = document.getElementById('bandDisplay');
  for (const m of CONFIG.models) {{
    const box = document.createElement('div');
    box.className = 'band-box';
    box.innerHTML = `
      <div class="db" id="db_${{m.band_label}}">0.0</div>
      <div class="label">${{m.band_label}}</div>
      <div class="eq-label">${{m.filter_type}} @ ${{m.center_hz}} Hz</div>
    `;
    bandDiv.appendChild(box);
  }}

  updatePredictions();
}}

function getSliderValues() {{
  const vals = {{}};
  for (const name of Object.keys(CONFIG.predictors)) {{
    const el = document.getElementById('slider_' + name);
    if (el) vals[name] = parseFloat(el.value);
  }}
  return vals;
}}

function predict(model, values) {{
  let result = model.coefficients.intercept || 0;
  for (const [name, coef] of Object.entries(model.coefficients)) {{
    if (name === 'intercept') continue;
    if (name in values) result += coef * values[name];
  }}
  return result;
}}

function updatePredictions() {{
  const vals = getSliderValues();
  for (const [name] of Object.entries(CONFIG.predictors)) {{
    document.getElementById('val_' + name).textContent = vals[name].toFixed(3);
  }}
  for (let i = 0; i < CONFIG.models.length; i++) {{
    const m = CONFIG.models[i];
    const db = predict(m, vals);
    const el = document.getElementById('db_' + m.band_label);
    el.textContent = db.toFixed(1);
    el.style.color = db < -1 ? '#2d5016' : db > 1 ? '#a02020' : '#666';
    if (filters[i]) {{
      filters[i].gain.setValueAtTime(db, audioCtx.currentTime);
    }}
  }}
  drawEQ();
}}

function drawEQ() {{
  const canvas = document.getElementById('eqCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const fMin = 20, fMax = 20000;
  const nPts = 256;
  const logFreqs = new Float32Array(nPts);
  for (let i = 0; i < nPts; i++) {{
    logFreqs[i] = fMin * Math.pow(fMax / fMin, i / (nPts - 1));
  }}

  // Get EQ response (dB) at each frequency
  const eqDb = new Float32Array(nPts);
  if (filters.length > 0) {{
    for (const f of filters) {{
      const mag = new Float32Array(nPts);
      const phase = new Float32Array(nPts);
      f.getFrequencyResponse(logFreqs, mag, phase);
      for (let i = 0; i < nPts; i++) {{
        eqDb[i] += 20 * Math.log10(Math.max(mag[i], 1e-12));
      }}
    }}
  }} else {{
    const vals = getSliderValues();
    for (const m of CONFIG.models) {{
      const db = predict(m, vals);
      for (let i = 0; i < nPts; i++) {{
        const f = logFreqs[i];
        if (m.band_label === 'low' && f < 500) eqDb[i] += db;
        else if (m.band_label === 'mid' && f >= 500 && f < 2000) eqDb[i] += db;
        else if (m.band_label === 'high' && f >= 2000) eqDb[i] += db;
      }}
    }}
  }}

  // Get scenario spectrum if available
  const spec = spectrumCache[currentScenario];
  const hasSpectrum = !!spec;

  // Determine dB range from spectrum or use fixed
  let dbMin, dbMax;
  if (hasSpectrum) {{
    // Find range of the spectrum
    let sMin = Infinity, sMax = -Infinity;
    for (let i = 0; i < nPts; i++) {{
      const f = logFreqs[i];
      const idx = Math.round(f / (spec.freqAxis[1] - spec.freqAxis[0]));
      if (idx >= 0 && idx < spec.psdDb.length) {{
        const v = spec.psdDb[idx];
        if (v < sMin) sMin = v;
        if (v > sMax) sMax = v;
      }}
    }}
    dbMax = sMax + 5;
    dbMin = sMin - 5;
  }} else {{
    dbMin = -30; dbMax = 30;
  }}
  const dbSpan = dbMax - dbMin;

  function yFromDb(db) {{ return H - ((db - dbMin) / dbSpan) * H; }}

  // Draw grid
  ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 0.5;
  const gridStep = hasSpectrum ? 10 : 10;
  for (let db = Math.ceil(dbMin / gridStep) * gridStep; db <= dbMax; db += gridStep) {{
    const y = yFromDb(db);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = '10px sans-serif';
    ctx.fillText(db.toFixed(0) + ' dB', 4, y - 3);
  }}
  for (const hz of [100, 1000, 10000]) {{
    const x = W * Math.log10(hz / fMin) / Math.log10(fMax / fMin);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillStyle = '#aaa';
    ctx.fillText(hz >= 1000 ? (hz/1000) + 'k' : hz + '', x + 3, H - 5);
  }}

  if (hasSpectrum) {{
    // Interpolate spectrum to our log frequency axis
    const origDb = new Float32Array(nPts);
    const filtDb = new Float32Array(nPts);
    const df = spec.freqAxis[1] - spec.freqAxis[0];
    for (let i = 0; i < nPts; i++) {{
      const idx = Math.min(Math.round(logFreqs[i] / df), spec.psdDb.length - 1);
      origDb[i] = idx >= 0 ? spec.psdDb[idx] : -100;
      filtDb[i] = origDb[i] + eqDb[i];
    }}

    // Draw original spectrum (grey)
    ctx.strokeStyle = '#999'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const x = W * i / (nPts - 1);
      const y = Math.max(2, Math.min(H - 2, yFromDb(origDb[i])));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw filtered spectrum (green, filled)
    ctx.strokeStyle = '#2d5016'; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const x = W * i / (nPts - 1);
      const y = Math.max(2, Math.min(H - 2, yFromDb(filtDb[i])));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Fill between original and filtered
    ctx.globalAlpha = 0.12;
    for (let i = 0; i < nPts - 1; i++) {{
      const x1 = W * i / (nPts - 1);
      const x2 = W * (i + 1) / (nPts - 1);
      const yO1 = yFromDb(origDb[i]), yO2 = yFromDb(origDb[i+1]);
      const yF1 = yFromDb(filtDb[i]), yF2 = yFromDb(filtDb[i+1]);
      ctx.fillStyle = eqDb[i] < 0 ? '#2d5016' : '#a02020';
      ctx.beginPath();
      ctx.moveTo(x1, yO1); ctx.lineTo(x2, yO2);
      ctx.lineTo(x2, yF2); ctx.lineTo(x1, yF1);
      ctx.fill();
    }}
    ctx.globalAlpha = 1.0;

    // Legend
    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#999';
    ctx.fillText('- - - Original', W - 160, 16);
    ctx.fillStyle = '#2d5016';
    ctx.fillText('\u2014\u2014 Filtered', W - 160, 30);

  }} else {{
    // No spectrum loaded yet — show EQ curve only
    const midY = H / 2;
    ctx.strokeStyle = '#888'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

    ctx.strokeStyle = '#2d5016'; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i < nPts; i++) {{
      const x = W * i / (nPts - 1);
      const y = midY - (eqDb[i] / 30) * midY;
      const yc = Math.max(2, Math.min(H - 2, y));
      if (i === 0) ctx.moveTo(x, yc); else ctx.lineTo(x, yc);
    }}
    ctx.stroke();

    ctx.font = '11px sans-serif'; ctx.fillStyle = '#999';
    ctx.fillText('EQ curve (press Play to see scenario spectrum)', W / 2 - 160, 16);
  }}
}}

function onSliderChange() {{ updatePredictions(); }}

async function loadAudio(scenario) {{
  if (audioBuffers[scenario]) return audioBuffers[scenario];

  const url = SCENARIOS[scenario];
  document.getElementById('status').textContent = 'Loading ' + scenario + '...';

  try {{
    const resp = await fetch(url);
    const arrayBuf = await resp.arrayBuffer();
    const decoded = await audioCtx.decodeAudioData(arrayBuf);
    audioBuffers[scenario] = decoded;
    // Compute and cache spectrum for visualization
    spectrumCache[scenario] = computeSpectrum(decoded);
    return decoded;
  }} catch (e) {{
    document.getElementById('status').textContent = 'Error loading ' + scenario + ': ' + e.message;
    return null;
  }}
}}

async function startPlaying(scenario) {{
  const buffer = await loadAudio(scenario);
  if (!buffer) return;

  // Create source
  sourceNode = audioCtx.createBufferSource();
  sourceNode.buffer = buffer;
  sourceNode.loop = true;

  // Create filter chain
  filters = [];
  let chain = sourceNode;
  for (const m of CONFIG.models) {{
    const f = audioCtx.createBiquadFilter();
    f.type = m.filter_type;
    f.frequency.value = m.center_hz;
    if (m.filter_type === 'peaking') f.Q.value = m.q_value;
    f.gain.value = 0;
    chain.connect(f);
    chain = f;
    filters.push(f);
  }}
  chain.connect(audioCtx.destination);
  sourceNode.start();
  isPlaying = true;

  document.getElementById('playBtn').textContent = '\\u23F9 Stop';
  document.getElementById('status').textContent = 'Playing: ' + scenario;
  updatePredictions();
}}

function stopAudio() {{
  if (sourceNode) {{
    sourceNode.stop();
    sourceNode.disconnect();
    sourceNode = null;
  }}
  filters = [];
  isPlaying = false;
  document.getElementById('playBtn').textContent = '\\u25B6 Play';
  document.getElementById('status').textContent = 'Stopped';
}}

async function togglePlay() {{
  if (!audioCtx) audioCtx = new AudioContext();
  if (isPlaying) {{
    stopAudio();
  }} else {{
    currentScenario = document.getElementById('scenarioSelect').value;
    await startPlaying(currentScenario);
  }}
}}

async function onScenarioChange() {{
  currentScenario = document.getElementById('scenarioSelect').value;
  if (isPlaying) {{
    stopAudio();
    await startPlaying(currentScenario);
  }} else if (audioCtx) {{
    // Preload spectrum even if not playing, so the chart updates
    await loadAudio(currentScenario);
    drawEQ();
  }}
}}

init();
</script>
</body>
</html>"""

    Path(output_path).write_text(html)
    logger.info(f"Slider HTML written to {output_path}")


# ============================================================
# LOAD MODELS FROM R OUTPUT
# ============================================================

def load_models_from_csv(csv_path: str, data_csv: str = None) -> dict:
    """
    Load fitted model coefficients from R stepwise_coefficients.csv.

    Models are for generic bands (low/mid/high). The slider applies them
    as a 3-band parametric EQ (low shelf + peaking + high shelf) to
    whichever scenario sound is playing.

    Parameters
    ----------
    csv_path : str
        Path to stepwise_coefficients.csv (from R)
    data_csv : str
        Path to merged_data.csv — used to extract real predictor ranges
        for the sliders. If None, tries output/csv/merged_data.csv.

    Returns {leaf_type: SliderBank}
    """
    import pandas as pd

    BAND_CONFIG = [
        ("attenuation_low_db",  "low",  250,  "lowshelf"),
        ("attenuation_mid_db",  "mid",  1000, "peaking"),
        ("attenuation_high_db", "high", 6000, "highshelf"),
    ]

    df = pd.read_csv(csv_path)

    # Load merged data for real predictor ranges
    pred_ranges = {}
    if data_csv is None:
        # Try standard location relative to csv_path
        candidate = Path(csv_path).parent.parent / "csv" / "merged_data.csv"
        if candidate.exists():
            data_csv = str(candidate)
    if data_csv and Path(data_csv).exists():
        df_data = pd.read_csv(data_csv)
        for col in df_data.columns:
            if df_data[col].dtype in ['float64', 'int64'] and df_data[col].notna().sum() > 2:
                pred_ranges[col] = (float(df_data[col].min()), float(df_data[col].max()))
        logger.info(f"Loaded predictor ranges from {data_csv} ({len(pred_ranges)} columns)")

    banks = {}

    for leaf_type in df["leaf_type"].unique():
        lt_df = df[df["leaf_type"] == leaf_type]
        models = []

        for resp_var, label, center, ftype in BAND_CONFIG:
            resp_df = lt_df[lt_df["response_var"] == resp_var]
            if resp_df.empty:
                continue

            best = resp_df.loc[resp_df["adj_r2"].idxmax()]
            coeffs = {"intercept": float(best.get("intercept", 0))}
            ranges = {}
            for i in range(1, 4):
                pred = best.get(f"predictor_{i}")
                coef = best.get(f"coef_{i}")
                if pd.notna(pred) and pd.notna(coef):
                    pred_name = str(pred)
                    coeffs[pred_name] = float(coef)
                    # Use real range from data, fallback to (0, 1)
                    ranges[pred_name] = pred_ranges.get(pred_name, (0.0, 1.0))

            models.append(SliderModel(
                band_label=label,
                center_hz=center,
                filter_type=ftype,
                coefficients=coeffs,
                predictor_ranges=ranges,
                adj_r2=float(best.get("adj_r2", 0)),
                criterion=str(best.get("criterion", "")),
            ))

        banks[leaf_type] = SliderBank(
            leaf_type=leaf_type,
            noise_profile_key="generic",
            models=models,
        )

    return banks
