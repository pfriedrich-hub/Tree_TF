"""
Play and record an FM Sweep to obtain the transfer function (tree canopy).
Distance correction is applied to the FREE-FIELD REFERENCE (measured at 1 m)
so that it matches the canopy mic distance before deconvolution.
"""
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from pathlib import Path
import numpy
import slab
import pyfar
import logging
import freefield
fs = 48828  # sampling rate of the TDT processor
slab.set_default_samplerate(fs)

def record(id, signal, n_recordings, rec_distance, show=True, axis=None):
    # record n_recordings times and average
    logging.info('Recording...')
    recordings = []
    for r in range(n_recordings):
        recordings.append(freefield.play_and_record_headphones('left', signal, distance=rec_distance, equalize=False))
    recording = slab.Sound(numpy.mean(numpy.asarray(recordings), axis=0))  # average
    recording.data -= numpy.mean(recording.data, axis=0)  # subtract the mean to center recording around zero
    # plot waveform
    if show:
        recording.waveform(axis)
        plt.title(f'{id} Waveform')
    # write to file
    counter = 1  # prevent overwriting
    while Path.exists('data' / id / f'{id}_rec.wav'):
        id = f'{id}_{counter}'
        counter += 1
    recording.write('data' / id / f'{id}_rec.wav')
    data_dir = Path.cwd() / 'data' / id
    data_dir.mkdir(parents=True, exist_ok=True)
    out_wav = data_dir / f'{id}_rec.wav'
    counter = 1
    while out_wav.exists():
        out_wav = data_dir / f'{id}_rec_{counter}.wav'
        counter += 1
    recording.write(out_wav)
    return recording

def compute_tf(id=None, rec_distance=1.0, recording=None, reference=None, window_size=120):
    """
    Compute the Transfer Function of the system (tree canopy).
    We scale the free-field reference (measured at 1 m) to the canopy mic
    distance (rec_distance) to remove geometric spreading (inverse square law)
    :param id (string): if given, auto-load recording and reference from disk
    :param rec_distance (float): distance source->canopy mic [m]
    :param recording (slab.Sound): canopy-path recording (tree)
    :param reference (slab.Sound): free-field reference recorded at 1 m
    :param window_size (int): window length in milliseconds for IR (removes late reflections)
    :return: raw_tf (slab.Filter), windowed_tf (slab.Filter)
    """
    # --- Load signals if needed ---
    if recording is None and id:
        recording_path = Path.cwd() / 'data' / id / f'{id}.wav'
        try:
            logging.info(f'Load recording from {recording_path}')
            recording = slab.Sound.read(recording_path)
        except FileNotFoundError:
            logging.error('Must provide id or recording data to compute TF.')
    if reference is None and id:
        reference_path = Path.cwd() / 'data' / f'{id}_ref' / f'{id}_ref.wav'
        try:
            logging.info(f'Load reference from {reference_path}')
            reference = slab.Sound.read(reference_path)
        except FileNotFoundError:
            logging.error('Must provide id or reference data to compute TF.')
    if recording is None or reference is None:
        raise ValueError("compute_tf needs both 'recording' and 'reference'.")
    if rec_distance <= 0:
        raise ValueError("rec_distance must be positive and non-zero.")
    # --- Distance-correct the FREE-FIELD reference from 1 m -> rec_distance ---
    if rec_distance != 1.0:
        reference = distance_scale(reference, input_distance=1.0, output_distance=rec_distance)
    # --- Convert to pyfar.Signal ---
    reference_pf = pyfar.Signal(data=reference.data.T, sampling_rate=reference.samplerate)
    recording_pf = pyfar.Signal(data=recording.data.T, sampling_rate=recording.samplerate)
    # --- Deconvolution via regularized inversion of the reference ---
    reference_inv = pyfar.dsp.regularized_spectrum_inversion(reference_pf, frequency_range=(20, 19.75e3))
    ir_deconvolved = recording_pf * reference_inv  # convolution in time domain = multiplication in frequency domain
    # --- Window the IR to remove late reflections ---
    fs = ir_deconvolved.sampling_rate  # convert window_size to samples
    win_samples = max(1, int(round(window_size * 1e-3 * fs)))
    ir_windowed = pyfar.dsp.time_window(ir_deconvolved, (0, win_samples), 'boxcar', unit='samples', crop='window')
    ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples)
    # --- Wrap as slab.Filter (magnitude TFs) ---
    raw_tf = slab.Filter(data=numpy.abs(ir_deconvolved.freq),
                         samplerate=ir_deconvolved.sampling_rate, fir='TF')
    windowed_tf = slab.Filter(data=numpy.abs(ir_windowed.freq),
                              samplerate=ir_deconvolved.sampling_rate, fir='TF')
    return raw_tf, windowed_tf


def distance_scale(sound, input_distance, output_distance=1.0):
    """
    Scale a recording by the inverse square law.
    For audio waveforms, sound pressure amplitude falls off by 1/distance.
    :param sound (slab.Sound): the recording to scale
    :param input_distance (float): distance of the input sound in meters
    :param output_distance (float): desired distance of the output sound in meters
    :return: slab.Sound: scaled recording
    """
    if input_distance <= 0 or output_distance <= 0:
        raise ValueError("Distances must be positive and non-zero.")
    # amplitude scaling factor (not squared, since we scale pressure not intensity)
    scale_factor = input_distance / output_distance
    import copy
    scaled = copy.deepcopy(sound)
    scaled.data *= scale_factor
    return scaled

def plot(recording, raw_tf, windowed_tf):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    recording.waveform(axis=axes[0])
    axes[0].set_title('Raw recording')
    raw_tf.tf(axis=axes[1])
    axes[1].set_title('Raw TF')
    axes[1].set_xlim(20, 20e3)
    axes[1].set_ylim(-70, 70)
    windowed_tf.tf(axis=axes[2])
    axes[2].set_title('Windowed TF')
    axes[2].set_xlim(20, 20e3)
    axes[2].set_ylim(-70, 70)
    return fig, axes


def write(id, recording, raw_tf, windowed_tf):
    """
    Write the recording and the transfer functions to the disk.
    """
    id_dict = {'recording': recording, 'raw_tf': raw_tf, 'windowed_tf': windowed_tf}
    data_dir = Path.cwd() / 'data' / id
    counter = 1
    while Path.exists(data_dir / f'{id}.pkl'):
        id = f'{id}_{counter}'
        counter += 1
    import pickle
    with open(data_dir / f'{id}.pkl', 'wb') as f:
        pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)


# deprecated
# def compute_tf(id=None, distance=1, recording=None, reference=None, window_size=120):
#     """
#     Compute the Transfer Function of the system. For a step by step explanation see "tf.py"
#     :param id (string): if given an id, automatically find the corresponding recording and reference file
#      to compute the transfer function.
#     :param recording (slab.Sound, optional): the raw recording of the system (tree recording)
#     :param reference (slab.Sound, optional): the reference recording to be removed from the recording (ref recording)
#     :param window_size (int): size of the window to apply to the impulse response (removes late reflections)
#     :return: raw_tf (slab.Filter): the resulting raw transfer function
#     :return: windowed_tf (slab.Filter): the windowed transfer function
#     """
#     if not recording:
#         if id:
#             recording_path = Path.cwd() / 'data' / id / f'{id}.wav'
#             try:
#                 logging.info(f'Load recording from {recording_path}')
#                 recording = slab.Sound.read(recording_path)
#             except FileNotFoundError:
#                 logging.error('Must provide id or recording data to compute TF.')
#     if not reference:
#         if id:
#             reference_path = Path.cwd() / 'data' / f'{id}_ref' / f'{id}_ref.wav'
#             try:
#                 logging.info(f'Load reference from {reference_path}')
#                 reference = slab.Sound.read(reference_path)
#             except FileNotFoundError:
#                 logging.error('Must provide id or reference data to compute TF.')
#     # convert slab.Sound to pyfar.Signal:
#     reference = pyfar.Signal(data=reference.data.T, sampling_rate=reference.samplerate)
#     recording = pyfar.Signal(recording.data.T, sampling_rate=recording.samplerate)
#     # invert reference and deconvolve with recording to compute tf
#     reference_inverted = pyfar.dsp.regularized_spectrum_inversion(reference, frequency_range=(20, 19.75e3))
#     ir_deconvolved = recording * reference_inverted  # convolution in time domain = multiplication in frequency domain
#     # window the impulse response
#     ir_windowed = pyfar.dsp.time_window(ir_deconvolved, (0, window_size), 'boxcar', unit='samples', crop='window')
#     ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples)
#     raw_tf = slab.Filter(data=numpy.abs(ir_deconvolved.freq), samplerate=ir_deconvolved.sampling_rate, fir='TF')
#     windowed_tf = slab.Filter(data=numpy.abs(ir_windowed.freq), samplerate=ir_deconvolved.sampling_rate, fir='TF')
#     return raw_tf, windowed_tf
