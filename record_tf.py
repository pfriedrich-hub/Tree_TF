"""
Play and record an FM Sweep to obtain the transfer function
"""
import numpy
from pathlib import Path
import slab
import freefield
import datetime
import pickle
date = datetime.datetime.now()
from copy import deepcopy
fs = 97656  # 97656.25, 195312.5
slab.set_default_samplerate(fs)

# file settings
id = 'free mic'  # ID of the recording
data_dir = Path.cwd() / 'data' / id

# probe signal settings
level = 80  # minimize to reduce reverb ripple effect with miniature mics, kemar recordings are not affected
duration = 0.1  # short chirps <0.05s introduce variation in low freq (4-5 kHz)
ramp_duration = 0.01
low_freq = 20
high_freq = 20000
repetitions = 30
signal = slab.Sound.chirp(duration=duration, level=level, from_frequency=low_freq, to_frequency=high_freq, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
signal_fft = numpy.fft.rfft(signal.data[:, 0])

#todo calibrate loudspeaker

def record_tf():
    # initialize
    if not freefield.PROCESSORS.mode:
        proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_rec_buf.rcx']]
        freefield.initialize('dome', device=proc_list)
        freefield.PROCESSORS.mode = 'play_rec'
        # freefield.load_equalization(file=)

    # record
    freefield.set_logger('info')
    recording = play_and_record()

    # compute tf
    with numpy.errstate(divide='ignore'):
        tf = numpy.fft.rfft(recording) / signal_fft  # compute tf
        tf = slab.Filter(tf.T, fs, fir='TF')  # create slab filter

    # write recordings and tf
    data_dir.mkdir(parents=True, exist_ok=True)  # create condition directory if it doesnt exist
    with open(data_dir / 'TF.pkl', 'wb') as f:
        pickle.dump(tf, f, pickle.HIGHEST_PROTOCOL)
    recording.write(f'{id}.wav')
    return tf, recording

def play_and_record():
    print('Recording...')
    [speaker] = freefield.pick_speakers(1) #todo init ff speaker object
    # speaker = freefield.Speaker()
    # Speaker(index=int(row[0]), analog_channel=int(row[1]), analog_proc=row[2],
    #         azimuth=float(row[3]), digital_channel=int(row[5]) if row[5] else None,
    #         elevation=float(row[4]), digital_proc=row[6] if row[6] else None)
    # get avg of n recordings from each sound source location
    recordings = []
    for r in range(repetitions):
        recordings.append(freefield.play_and_record(speaker, signal, equalize=False))
    rec = slab.Binaural(numpy.mean(numpy.asarray(recs), axis=0))  # average
    rec.data -= numpy.mean(rec.data, axis=0)  # baseline
    rec = slab.Binaural.ramp(rec, when='both', duration=ramp_duration)
    azimuth = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][1]
    elevation = sources[numpy.where(sources[:, 0] == speaker_id)[0][0]][2]
    recordings.append([azimuth, elevation, rec])
    print('progress: %i %%' % (int((idx+1)/len(speaker_ids)*100)), end="\r", flush=True)
    return recordings

def create_src_txt(recordings):
    # convert interaural_polar to vertical_polar coordinates for sofa file
    # interaural polar to cartesian
    # interaural_polar = numpy.asarray(recordings)[:, :2].astype('float')  # deprecated numpy
    interaural_polar = sources[:, 1:]
    cartesian = numpy.zeros((len(interaural_polar), 3))
    vertical_polar = numpy.zeros((len(interaural_polar), 3))
    azimuths = numpy.deg2rad(interaural_polar[:, 0])
    elevations = numpy.deg2rad(90 - interaural_polar[:, 1])
    r = 1.4  # get radii of sound sources
    cartesian[:, 0] = r * numpy.cos(azimuths) * numpy.sin(elevations)
    cartesian[:, 1] = r * numpy.sin(azimuths)
    cartesian[:, 2] = r * numpy.cos(elevations) * numpy.cos(azimuths)
    # cartesian to vertical polar
    xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
    vertical_polar[:, 0] = numpy.rad2deg(numpy.arctan2(cartesian[:, 1], cartesian[:, 0]))
    vertical_polar[vertical_polar[:, 0] < 0, 0] += 360
    vertical_polar[:, 1] = 90 - numpy.rad2deg(numpy.arctan2(numpy.sqrt(xy), cartesian[:, 2]))
    vertical_polar[:, 2] = numpy.sqrt(xy + cartesian[:, 2] ** 2)
    return vertical_polar.astype('float16')

# def record_in_intervals(signal, speaker, repetitions, rec_samplerate):
#     recording_samplerate = fs
#     direct_delay = freefield.get_recording_delay(distance=1.4, sample_rate=recording_samplerate,
#                                             play_from="RX8", rec_from="RP2") + 50
#     reverb_delay = freefield.get_recording_delay(distance=3, sample_rate=recording_samplerate,
#                                             play_from="RX8", rec_from="RP2")
#     n_slice = reverb_delay - direct_delay
#
#     freefield.set_signal_and_speaker(signal, speaker, equalize=True)  # write to RX8 buffers, set output channels
#     freefield.write(tag="n_slice", value=n_slice, processors=["RX81", "RX82"])  # set playbuflen to n_slice datapoints
#     # set slice + delay as recording length
#     freefield.write(tag="n_slice", value=n_slice + direct_delay, processors="RP2")
#     # record until the whole signal (including signal delays) is captured by the recording buffer
#     n_rec = signal.n_samples
#     delay_ids = numpy.empty(0)
#     delay_start = 0
#     recs = []
#     for i in range(repetitions):
#         while not (freefield.read('buf_idx', processor='RP2', n_samples=1) >= n_rec):
#             freefield.play('zBusA')  # iterate over slices
#             freefield.wait_to_finish_playing()
#             n_rec += direct_delay
#             delay_stop = delay_start + direct_delay
#             delay_ids = numpy.concatenate((delay_ids, numpy.arange(delay_start, delay_stop)))
#             delay_start = delay_stop + n_slice
#         freefield.play('zBusB') # reset buffer index
#         rec_l = read(tag='datal', processor='RP2', n_samples=n_rec)
#         rec_r = read(tag='datar', processor='RP2', n_samples=n_rec)
#         # remove direct delays before each slice
#         rec_l = numpy.delete(rec_l, delay_ids)
#         rec_r = numpy.delete(rec_r, delay_ids)
#         recs.append[rec_l, rec_r]
#
#         rec = slab.Binaural(numpy.mean(recs, axis=0), samplerate=recording_samplerate)
#         return rec

if __name__ == "__main__":
    recordings, sources, hrtf = record_hrtf(subject_id, data_dir, condition, signal, repetitions, n_directions, safe, speakers, kemar)
    sources = list(range(hrtf.n_sources-1, -1, -1))  # works for 0°/+/-17.5° cone
    hrtf.plot_tf(sources, xlim=(4000, 16000))
    # fig, axis = plt.subplots(2, 1)
    # hrtf.plot_hrtf_image(hrtf, sources, plot_bins, kind='waterfall', axis=axis[0], ear=plot_ear, xlim=(4000, 16000), dfe=dfe)
    # hrtf.vsi_across_bands(hrtf, sources, n_bins=plot_bins, axis=axis[1], dfe=dfe)
    # axis[0].set_title(subject_id)
    # # hrtf.plot_tf(sources, xlim=(low_freq, high_freq), ear=plot_ear)
    # hrtf.plot_tf(sources, xlim=(4000, 16000), ear=plot_ear)
