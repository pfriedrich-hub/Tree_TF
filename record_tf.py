"""
Play and record an FM Sweep to obtain the transfer function
"""
import matplotlib
matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # todo test this
from matplotlib import pyplot as plt
from pathlib import Path
import numpy
import pickle
import slab
import pyfar
import logging
import freefield
fs = 48828  # sampling rate of the TDT processor
slab.set_default_samplerate(fs)


def record(id, signal, n_recordings, distance, show=True, axis=None):
    # record n_recordings times
    print('Recording...')
    recordings = []
    for r in range(n_recordings):
        recordings.append(freefield.play_and_record_headphones('left', signal, distance=distance, equalize=False))
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
    return recording


def compute_tf(id=None, recording=None, reference=None, window_size=120):
    """
    Compute the Transfer Function of the system. For a step by step explanation see "tf.py"
    :param id (string): if given an id, automatically find the corresponding recording and reference file and write
                        the resulting transfer function to "id"_tf.pkl.
    :param recording (slab.Sound, optional): the raw recording of from the system (tree)
    :param reference (slab.Sound, optional): the reference recording to be removed from the recording (no tree)
    :param window_size (int): the window to apply to the impulse response to remove late reflections
    :return: tf (slab.Filter): the resulting transfer function
    """
    if not recording:
        if id:
            recording_path = Path.cwd() / 'data' / id / f'{id}.wav'
            try:
                logging.info(f'Load recording from {recording_path}')
                recording = slab.Sound.read(recording_path)
            except FileNotFoundError:
                logging.error('Must provide id or recording data to compute TF.')
    if not reference:
        if id:
            reference_path = Path.cwd() / 'data' / f'{id}_ref' / f'{id}_ref.wav'  # todo take universal ref and scale by distance square law
            try:
                logging.info(f'Load reference from {reference_path}')
                reference = slab.Sound.read(reference_path)
            except FileNotFoundError:
                logging.error('Must provide id or reference data to compute TF.')
    # convert slab.Sound to pyfar.Signal:
    reference = pyfar.Signal(data=reference.data.T, sampling_rate=reference.samplerate)
    recording = pyfar.Signal(recording.data.T, sampling_rate=recording.samplerate)
    # invert reference and deconvolve with recording to compute tf
    reference_inverted = pyfar.dsp.regularized_spectrum_inversion(reference, frequency_range=(20, 19.75e3))
    ir_deconvolved = recording * reference_inverted  # convolution in time domain = multiplication in frequency domain
    # window the impulse response
    ir_windowed = pyfar.dsp.time_window(ir_deconvolved, (0, window_size), 'boxcar', unit='samples', crop='window')
    ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples)
    raw_tf = slab.Filter(data=numpy.abs(ir_deconvolved.freq), samplerate=ir_deconvolved.sampling_rate, fir='TF')
    windowed_tf = slab.Filter(data=numpy.abs(ir_windowed.freq), samplerate=ir_deconvolved.sampling_rate, fir='TF')
    return raw_tf, windowed_tf


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
    id_dict = {'recording': recording, 'raw_tf': raw_tf, 'windowed_tf:': windowed_tf}
    data_dir = Path.cwd() / 'data' / id
    counter = 1
    while Path.exists(data_dir / f'{id}.pkl'):
        id = f'{id}_{counter}'
        counter += 1
    with open(data_dir / f'{id}.pkl', 'rw') as f:
        pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)