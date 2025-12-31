"""
Acoustic Processing Module
==========================
Functions for processing tree sound recordings into transfer functions.

Processing Pipeline:
1. baseline_reference() - Align reference spectrum to recording
2. trim_signals() - Remove leading silence
3. deconvolve() - Compute impulse response via spectral division
4. window_ir() - Apply time window to remove reflections
5. extract_transfer_function() - Get magnitude spectrum in dB
6. generate_filtered_audio() - Filter audio through tree TF
"""
import numpy as np
import pyfar
import slab
import soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve

from config import FREQ_HIGH, FREQ_LOW, SAMPLE_RATE


def baseline_reference(recording: slab.Sound, reference: slab.Sound) -> slab.Sound:
    """
    Baseline the reference signal in the frequency domain.
    
    Shifts the reference spectrum so its low-frequency energy matches
    the recording. This compensates for different recording levels
    and ensures consistent deconvolution across trees.
    
    Parameters
    ----------
    recording : slab.Sound
        The tree recording (signal through canopy)
    reference : slab.Sound  
        The free-field reference recording
        
    Returns
    -------
    slab.Sound
        Reference with adjusted amplitude spectrum
    """
    rec = np.squeeze(np.asarray(recording.data))
    ref = np.squeeze(np.asarray(reference.data))
    sr = recording.samplerate
    
    # FFT
    rec_spec = np.fft.rfft(rec)
    ref_spec = np.fft.rfft(ref)
    
    # Magnitudes in dB
    rec_mag_db = 20 * np.log10(np.maximum(np.abs(rec_spec), 1e-12))
    ref_mag_db = 20 * np.log10(np.maximum(np.abs(ref_spec), 1e-12))
    
    # Match low-frequency baseline (first 200 bins ≈ 200 Hz)
    low_bins = 200
    shift_db = np.mean(rec_mag_db[:low_bins]) - np.mean(ref_mag_db[:low_bins])
    
    # Apply shift to reference
    ref_mag_db_shifted = ref_mag_db + shift_db
    ref_mag_shifted = 10 ** (ref_mag_db_shifted / 20)
    
    # Reconstruct complex spectrum (preserve phase)
    ref_phase = np.angle(ref_spec)
    ref_spec_shifted = ref_mag_shifted * np.exp(1j * ref_phase)
    
    # Inverse FFT
    ref_baselined = np.fft.irfft(ref_spec_shifted, n=len(ref))
    
    return slab.Sound(ref_baselined, samplerate=sr)


def trim_signals(
    recording: slab.Sound,
    reference: slab.Sound,
    threshold_rel: float = 0.02,
    safety_margin_ms: float = 0.0,
) -> tuple:
    """
    Trim leading silence from recording and reference.
    
    Detects signal onset by finding where amplitude exceeds threshold,
    then trims both signals to maintain alignment.
    
    Parameters
    ----------
    recording : slab.Sound
        Recording to trim
    reference : slab.Sound
        Reference to trim (same amount)
    threshold_rel : float
        Onset threshold as fraction of peak amplitude
    safety_margin_ms : float
        Extra samples to keep before onset
        
    Returns
    -------
    tuple
        (trimmed_recording, trimmed_reference, start_index)
    """
    fs = recording.samplerate
    arr = np.asarray(recording.data)
    
    # Handle multi-channel by averaging
    if arr.ndim > 1:
        abs_sig = np.mean(np.abs(arr), axis=1)
    else:
        abs_sig = np.abs(arr)
    
    peak = np.max(abs_sig)
    if peak == 0:
        return recording, reference, 0
    
    thresh = threshold_rel * peak
    above = np.where(abs_sig > thresh)[0]
    
    if len(above) == 0:
        return recording, reference, 0
    
    # Find robust onset (check local mean to avoid spikes)
    check_window = int(round(1e-3 * fs))  # 1 ms window
    valid_start = None
    
    for idx in above:
        lo = max(0, idx - check_window)
        hi = min(len(abs_sig), idx + check_window)
        local_mean = np.mean(abs_sig[lo:hi])
        
        if local_mean > 0.5 * thresh:
            valid_start = idx
            break
    
    if valid_start is None:
        valid_start = int(above[0])
    
    # Apply safety margin
    margin_samples = int(round(safety_margin_ms * 1e-3 * fs))
    start_idx = max(0, valid_start - margin_samples)
    
    # Trim both signals (reference and recording)
    rec_data = arr[start_idx:] if arr.ndim == 1 else arr[start_idx:, :]
    rec_trimmed = slab.Sound(data=rec_data, samplerate=fs)
    
    ref_data = np.asarray(reference.data)
    ref_data = ref_data[start_idx:] if ref_data.ndim == 1 else ref_data[start_idx:, :]
    ref_trimmed = slab.Sound(data=ref_data, samplerate=reference.samplerate)
    
    return rec_trimmed, ref_trimmed, start_idx


def deconvolve(
    recording: slab.Sound,
    reference: slab.Sound,
    freq_range: tuple = (FREQ_LOW, FREQ_HIGH),
) -> pyfar.Signal:
    """
    Deconvolve recording with reference to obtain impulse response.
    
    Uses regularized spectrum inversion to avoid division by small
    values at frequencies where the reference has low energy.
    
    Parameters
    ----------
    recording : slab.Sound
        Tree recording (through canopy)
    reference : slab.Sound
        Free-field reference
    freq_range : tuple
        (low_freq, high_freq) for regularization
        
    Returns
    -------
    pyfar.Signal
        Deconvolved impulse response
    """
    fs = recording.samplerate
    
    # Convert to pyfar signals
    rec_pf = pyfar.Signal(np.asarray(recording.data).T, fs)
    ref_pf = pyfar.Signal(np.asarray(reference.data).T, fs)
    
    # Regularized inversion of reference
    ref_inv = pyfar.dsp.regularized_spectrum_inversion(
        ref_pf, frequency_range=freq_range
    )
    
    # Deconvolution: multiplication in frequency domain
    ir_full = rec_pf * ref_inv
    
    return ir_full


def window_ir(
    ir_full: pyfar.Signal,
    fade_in_ms: float = 0.25,
    sustain_ms: float = 4.8,
    fade_out_ms: float = 1.0,
) -> tuple:
    """
    Apply time window to impulse response to remove late reflections.
    
    The window is centered around the detected IR onset:
    - Short fade-in before onset
    - Sustain period capturing direct sound
    - Fade-out to smoothly attenuate reflections
    
    Parameters
    ----------
    ir_full : pyfar.Signal
        Full impulse response from deconvolution
    fade_in_ms : float
        Duration before onset for fade-in
    sustain_ms : float
        Duration of sustain after onset
    fade_out_ms : float
        Duration of fade-out
        
    Returns
    -------
    tuple
        (windowed_ir, window_signal)
    """
    fs = ir_full.sampling_rate
    n_samples = ir_full.n_samples
    
    # Detect IR onset
    onset_idx = int(pyfar.dsp.find_impulse_response_start(ir_full, threshold=20)[0])
    
    # Calculate window boundaries in samples
    fade_in_start = onset_idx - int(fade_in_ms * 1e-3 * fs)
    fade_in_end = onset_idx
    fade_out_start = onset_idx + int(sustain_ms * 1e-3 * fs)
    fade_out_end = onset_idx + int((sustain_ms + fade_out_ms) * 1e-3 * fs)
    
    # Clamp to valid range
    fade_in_start = max(fade_in_start, 0)
    fade_in_end = max(fade_in_end, 0)
    fade_out_start = min(fade_out_start, n_samples - 1)
    fade_out_end = min(fade_out_end, n_samples - 1)
    
    # Ensure proper ordering
    samples = tuple(sorted([fade_in_start, fade_in_end, fade_out_start, fade_out_end]))
    
    # Apply Hann window
    ir_windowed, window = pyfar.dsp.time_window(
        ir_full, samples, "hann", unit="samples", crop="end", return_window=True
    )
    
    return ir_windowed, window


def extract_transfer_function(ir: pyfar.Signal) -> tuple:
    """
    Extract transfer function (magnitude spectrum) from impulse response.
    
    Parameters
    ----------
    ir : pyfar.Signal
        Impulse response (windowed)
        
    Returns
    -------
    tuple
        (frequencies_hz, magnitude_db)
    """
    sr = ir.sampling_rate
    n = ir.n_samples
    
    # Frequency axis
    freqs = np.linspace(0, sr / 2, n // 2 + 1)
    
    # Magnitude spectrum in dB
    mag = np.abs(ir.freq).squeeze()
    mag = np.maximum(mag, 1e-12)  # avoid log(0)
    mag_db = 20 * np.log10(mag)
    
    return freqs, mag_db


def generate_filtered_noise(
    ir: pyfar.Signal,
    duration_s: float = 3.0,
    normalize_peak: float = 0.9,
) -> np.ndarray:
    """
    Generate white noise filtered through the tree's transfer function.
    
    Creates an audio file that demonstrates how the tree affects sound,
    useful for intuitive presentation of results.
    
    Parameters
    ----------
    ir : pyfar.Signal
        Impulse response to use as filter
    duration_s : float
        Duration of noise in seconds
    normalize_peak : float
        Peak normalization level (0-1)
        
    Returns
    -------
    np.ndarray
        Filtered noise audio (float32)
    """
    fs = ir.sampling_rate
    n_samples = int(duration_s * fs)
    
    # Generate white noise
    noise = np.random.normal(0, 1, n_samples).astype(np.float64)
    
    # Get IR as numpy array
    ir_data = np.ravel(ir.time).astype(np.float64)
    
    # Convolve
    filtered = fftconvolve(noise, ir_data, mode="full")[:n_samples]
    
    # Normalize
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = (filtered / peak) * normalize_peak
    
    return filtered.astype(np.float32)


def generate_filtered_audio(
    ir: pyfar.Signal,
    input_audio: np.ndarray,
    input_sr: int,
    normalize_peak: float = 0.9,
) -> np.ndarray:
    """
    Filter arbitrary audio through the tree's transfer function.
    
    Parameters
    ----------
    ir : pyfar.Signal
        Impulse response to use as filter
    input_audio : np.ndarray
        Input audio signal
    input_sr : int
        Sample rate of input audio
    normalize_peak : float
        Peak normalization level (0-1)
        
    Returns
    -------
    np.ndarray
        Filtered audio (float32)
    """
    ir_sr = ir.sampling_rate
    
    # Resample input if necessary
    if input_sr != ir_sr:
        from scipy.signal import resample
        n_new = int(len(input_audio) * ir_sr / input_sr)
        audio = resample(input_audio, n_new).astype(np.float64)
    else:
        audio = input_audio.astype(np.float64)
    
    # Get IR as numpy array
    ir_data = np.ravel(ir.time).astype(np.float64)
    
    # Convolve
    filtered = fftconvolve(audio, ir_data, mode="full")[:len(audio)]
    
    # Normalize
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = (filtered / peak) * normalize_peak
    
    return filtered.astype(np.float32)


def load_playground_noise(wav_dir: Path) -> tuple:
    """
    Load playground noise file for sonification.
    
    Parameters
    ----------
    wav_dir : Path
        Directory containing playground.wav
        
    Returns
    -------
    tuple
        (audio_data, sample_rate) or (None, None) if not found
    """
    playground_path = wav_dir / "playground.wav"
    
    if not playground_path.exists():
        return None, None
    
    audio, sr = sf.read(playground_path)
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    return audio, sr
