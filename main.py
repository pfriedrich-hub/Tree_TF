import matplotlib.pyplot as plt
from record_tf import *

# --- Settings --- #
id = '313_4.6_20N'  # ID of the recording (will create a new subfolder)
rec_distance = 1.8 # distance between microphone and speaker in meters
n_recordings = 10  # number of recordings to average
level = 85  # signal level
duration = 0.5  # signal duration
low_freq = 20  # signal frequencies
high_freq = 20000
fs = 48828  # sampling rate of the TDT processor
window_size = 120  # time window in ms applied to the resulting IR to remove reflections
show = True  # whether to show a plot of the recording and resulting transfer function

# make signal
slab.set_default_samplerate(fs)
signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq,
                          kind='logarithmic')  # make signal
signal = signal.ramp(when='both', duration=0.005)  # ramp signal to avoid clicks

if __name__ == "__main__":
    # initialize processors (this function should not return any warnings in order to record)
    proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_play_rec_buf.rcx']]
    freefield.initialize('headphones', device=proc_list, connection='USB', zbus=False)
    freefield.PROCESSORS.mode = 'bi_play_rec'
    freefield.set_logger('info')  # set to 'debug' if you want full report from the processor
    # record a signal and write to sound file in /data / id / id_rec.wav
    recording = record(id, n_recordings, rec_distance, show=False)
    # compute the tf
    raw_tf, windowed_tf = compute_tf(id, rec_distance, window_size=window_size)
    # plot results
    if show:
        fig, axes = plot(recording, raw_tf, windowed_tf)
        fig.suptitle(id)
    # save to results in a pickle file (data / id / id.pkl)
    write(id=id, recording=recording, raw_tf=raw_tf, windowed_tf=windowed_tf)
