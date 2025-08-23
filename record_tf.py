"""
Play and record an FM Sweep to obtain the transfer function
"""
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import numpy
import pickle
import slab
import freefield
fs = 48828
slab.set_default_samplerate(fs)

# settings
id = 'free mic'  # ID of the recording
distance = 0  # distance between microphone and speaker in meters
n_recordings = 10  # number of recordings to average
level = 70  # signal level
duration = 0.5  # signal duration
low_freq = 20  # signal frequencies
high_freq = 20000
write = True  # save recordings.wav and tf.pkl

def record_tf(signal, distance, n_recordings):
    print('Recording...')
    recordings = []
    for r in range(n_recordings):    # record
        recordings.append(freefield.play_and_record_headphones('left', signal, distance=distance, equalize=False))
    rec = slab.Sound(numpy.mean(numpy.asarray(recordings), axis=0))  # average
    rec.data -= numpy.mean(rec.data, axis=0)  # baseline
    with numpy.errstate(divide='ignore'):
        tf = numpy.fft.rfft(rec.data[:, 0]) / signal_fft  # compute tf
        tf = slab.Filter(tf.T, fs, fir='TF')  # create slab filter
    return tf, rec

if __name__ == "__main__":
    # initialize
    if not freefield.PROCESSORS.mode:
        proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_play_rec_buf.rcx']]
        freefield.initialize('headphones', device=proc_list, connection='USB', zbus=False)
        freefield.PROCESSORS.mode = 'bi_play_rec'
        freefield.set_logger('info')  # todo check recording delay
    # make signal
    signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq,
                              kind='linear')
    signal = signal.ramp(when='both', duration=0.005)
    signal_fft = numpy.fft.rfft(signal.data[:, 0])
    # record and compute tf
    tf, rec = record_tf(signal, distance, n_recordings)

    # write data
    if write:
        data_dir = Path.cwd() / 'data' / id
        # write recordings and tf
        data_dir.mkdir(parents=True, exist_ok=True)  # create condition directory if it doesnt exist
        with open(data_dir / 'TF.pkl', 'wb') as f:
            pickle.dump(tf, f, pickle.HIGHEST_PROTOCOL)
        rec.write(data_dir / f'{id}.wav')
