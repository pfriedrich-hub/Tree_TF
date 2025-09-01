"""
Play and record an FM Sweep to obtain the transfer function
"""
import matplotlib
matplotlib.use('TkAgg')
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

def record(id, signal, n_recordings, distance, show, axis=None):
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

def compute_tf(id=None, recording=None, reference=None, window_size=120, show=True, axis=None):
    """
    Compute the Transfer Function of the system. For a step by step explanation see "tf.py"
    :param id (string): if given an id, automatically find the corresponding recording and reference file and write
                        the resulting transfer function to "id"_tf.pkl.
    :param recording (slab.Sound, optional): the raw recording of from the system (tree)
    :param reference (slab.Sound, optional): the reference recording to be removed from the recording (no tree)
    :param window_size (int): the window to apply to the impulse response to remove late reflections
    :param show (bool): whether to plot the resulting tf
    :return: tf (slab.Filter): the resulting transfer function
    """
    if id:  # load recording and reference
        recording_path = Path('data' / id / f'{id}.wav')
        reference_path = Path('data' / 'reference.wav')
        if Path('data' / id / f'{id}.wav').exists():
            logging.info(f'Load recording from {recording_path}')
            recording = slab.Sound.read(recording_path)
        else: logging.error(f'Recording file not found: {recording_path}')
        if reference_path.exists():
            logging.info(f'Load reference from {reference_path}')
            reference = slab.Sound.read(reference_path)
        else: logging.error(f'Reference file not found: {reference_path}')
    elif not(recording or reference):
        logging.error('Must provide id to load an existing recording and reference or directly specify the'
                      ' recording/reference in the function call to compute a transfer function.')
    # get recording and reference signal from input arguments (must be slab.Sound objects)
    # convert to pyfar.Signal objects:
    reference = pyfar.Signal(data=reference.data.T, sampling_rate=reference.samplerate)
    recording = pyfar.Signal(recording.data.T, sampling_rate=recording.samplerate)
    reference_inverted = pyfar.dsp.regularized_spectrum_inversion(reference, frequency_range=(20, 19.75e3))
    ir_deconvolved = recording * reference_inverted  # convolution in time domain = multiplication in frequency domain
    if show:  # plot
        plt.figure()
        ax = pyfar.plot.time_freq(ir_deconvolved, unit='samples')
        ax[0].set_xlim(0, 1e3)
        ax[0].set_title('raw tf')
        ax[1].set_ylim(-40, 20)
    # window the impulse response
    ir_windowed = pyfar.dsp.time_window(ir_deconvolved, (0, window_size), 'boxcar', unit='samples', crop='window')
    ir_windowed = pyfar.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples)
    if show:  # plot
        pyfar.plot.freq(ir_windowed)
        ax.set_xlim(0, 2.1e4)
        ax.set_ylim(-40, 20)
        plt.title('windowed tf')
    # convert to slab.Filter and return
    ir_windowed.data
    return raw_tf, windowed_tf

def write(id, recording, raw_tf, windowed_tf):
    id_dict = {'recording': recording, 'raw_tf': raw_tf, 'windowed_tf:': windowed_tf}
    data_dir = Path.cwd() / 'data' / id
    counter = 1
    while Path.exists(data_dir / f'{id}.pkl'):
        id = f'{id}_{counter}'
        counter += 1
    with open(data_dir / f'{id}.pkl', 'wb') as f:
        pickle.dump(id_dict, f, pickle.HIGHEST_PROTOCOL)
