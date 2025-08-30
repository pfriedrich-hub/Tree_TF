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
import freefield
fs = 48828  # sampling rate of the TDT processor
slab.set_default_samplerate(fs)

# settings
id = '353_9.1_255W_ref'  # ID of the recording (will create a new subfolder)
distance = 0  # distance between microphone and speaker in meters
n_recordings = 10  # number of recordings to average
level = 85  # signal level
duration = 0.5  # signal duration
low_freq = 20  # signal frequencies
high_freq = 20000

def record(distance, n_recordings):
    # make signal
    signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq,
                              kind='linear')
    signal = signal.ramp(when='both', duration=0.005)
    signal_fft = numpy.fft.rfft(signal.data[:, 0])
    print('Recording...')
    recordings = []
    for r in range(n_recordings):    # record
        recordings.append(freefield.play_and_record_headphones('left', signal, distance=distance, equalize=False))
    rec = slab.Sound(numpy.mean(numpy.asarray(recordings), axis=0))  # average
    rec.data -= numpy.mean(rec.data, axis=0)  # baseline
    with numpy.errstate(divide='ignore'):
        tf = numpy.abs(numpy.fft.rfft(rec.data[:, 0]) / signal_fft)  # compute tf and get power spectrum
        tf = slab.Filter(tf.T, fs, fir='TF')  # store in a slab filter object
    return tf, rec

def write(id, rec, tf):
    # write data
    data_dir = Path.cwd() / 'data' / id
    data_dir.mkdir(parents=True, exist_ok=True)  # create condition directory if it doesnt exist
    counter = 1
    while Path.exists(data_dir / f'{id}.wav'):
        id = f'{id}_{counter}'
        counter += 1
    with open(data_dir / f'{id}_TF.pkl', 'wb') as f:
        pickle.dump(tf, f, pickle.HIGHEST_PROTOCOL)
    rec.write(data_dir / f'{id}.wav')
    # plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    rec.waveform(axis=axes[0])
    tf.tf(axis=axes[1])
    axes[1].set_xlim(low_freq, high_freq)
    plt.title(id)
    plt.savefig(data_dir / f'{id}.png')

if __name__ == "__main__":
    # initialize
    if not freefield.PROCESSORS.mode:
        proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_play_rec_buf.rcx']]
        freefield.initialize('headphones', device=proc_list, connection='USB', zbus=False)
        freefield.PROCESSORS.mode = 'bi_play_rec'
        freefield.set_logger('info')

    tf, rec = record(distance, n_recordings)  # record and compute tf
    write(id, rec, tf)  # write and plot
