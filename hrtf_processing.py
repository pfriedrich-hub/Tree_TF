# Auto-converted from notebook: hrtf_processing.ipynb

import pyfar as pf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import pooch
import os
# %matplotlib ipympl

# adjust this path to your needs. Using `None` will download the file to your
# system cash.
path = None

# Leave this as it is: This is the URL from which the data will be downloaded
# and a hash for checking if the download worked.
url = 'https://github.com/pyfar/files/raw/refs/heads/main/education/VAR_TUB/hrtf_post_processing_and_normalization.far?download='
hash = '1f4e7ad698ce65c1e359d914918fa9f4f81ca611eb9812243b55798fb2462732'

file = pooch.retrieve(
    url, hash, fname='hrtf_post_processing_and_normalization.far', path=path)

# read data
data = pf.io.read(file)

# sweeps recorded at the ears and corresponding microphone positions
ear_pressure = data['ear_pressure']
ear_positions = data['ear_positions']

# sweeps recorded at the center of the loudspeaker array with the dummy
# being absent and the corresponding position
reference_pressure = data['reference_pressure']
reference_position = data['reference_position']

# source positions for with the recorded sweeps were taken
source_positions = data['source_positions']

# Leave this as it is: This is the URL from which the data will be downloaded
url = 'https://github.com/pyfar/open-educational-resources/tree/main/docs/oer/courses/Virtual_Acoustic_Reality_TUB/hrtf_processing/spherical_head.py?raw=true'
hash = '7e30984c122dbc6df2c76ee073c8c6bc4c3d750ddb10186cd547ab5c18ee23c6'

# Download to the directory of this notebook for importing
_ = pooch.retrieve(
    url, hash, fname='spherical_head.py', path=os.getcwd())

from hrtf.processing.spherical_head import spherical_head

### BEGIN SOLUTION
# ignore warning when plotting a single coordinate point
with np.errstate(divide='ignore', invalid='ignore'):
    ax = source_positions.show(color='k')
    ear_positions[0].show(label='left ear', ax=ax)
    ear_positions[1].show(label='right ear', ax=ax)
    reference_position.show(label='center of array', ax=ax)
ax.legend()
plt.show()

print('Source positions:')
for n in range(source_positions.csize):
    pos = pf.rad2deg(source_positions[n].spherical_elevation).flatten()
    print(f'{n}: {pos[0]:3.0f} deg. azimuth, {pos[1]:2.0f} deg. elevation, {pos[2]:1.1f} m radius, ')
### END SOLUTION

### BEGIN SOLUTION
data = ear_pressure[1]
plt.figure()
ax = pf.plot.time(data, label=['left', 'right'])
ax.legend()
ax.set_title('Recorded sine sweep')
plt.show()
### END SOLUTION

### BEGIN SOLUTION
sweep_player = Audio(
    data.time, rate=data.sampling_rate)
display('Recorded sweep:')
display(sweep_player)
### END SOLUTION

# sweep inversion
### BEGIN SOLUTION
reference_pressure_inverted = pf.dsp.regularized_spectrum_inversion(
    reference_pressure, frequency_range=(110, 21000))

plt.figure()
ax = pf.plot.freq(reference_pressure_inverted)
ax.set_title('Inverted reference measurements')
ax.set_xlim(50, 22050)
plt.show()
### END SOLUTION

### BEGIN SOLUTION
hrir_deconvolved = ear_pressure * reference_pressure_inverted

plt.figure()
ax = pf.plot.time(hrir_deconvolved, dB=True)
ax.set_title('Deconcolved HRIRs')
ax.set_xlim(-.1, 1.6)
plt.show()
### END SOLUTION

### BEGIN SOLUTION
hrir_shifted = pf.dsp.time_shift(hrir_deconvolved, 2000)

plt.figure()
ax = pf.plot.time(hrir_shifted, dB=True, unit='ms')
ax.set_title('Time shifted HRIRs')
plt.show()
### END SOLUTION

### BEGIN SOLUTION
# frontal HRIR is at index 0
idx = 0

# detect and visualize onsets
onsets = pf.dsp.find_impulse_response_start(hrir_shifted[idx])

plt.figure()
ax = pf.plot.time(hrir_shifted[idx], unit='ms')
ax.axvline(onsets[0] / hrir_shifted.sampling_rate * 1000,
           color=pf.plot.color('b'), linestyle='--',
           label=f'left={onsets[0]} samples')
ax.axvline(onsets[0] / hrir_shifted.sampling_rate * 1000,
           color=pf.plot.color('r'), linestyle='--',
           label=f'right={onsets[1]} samples')
ax.set_xlim(44, 48)
ax.legend()
ax.set_title('Detected onset times for frontal HRIR')
plt.show()
### END SOLUTION

### BEGIN SOLUTION
# align HRIRs and visualize results
hrir_aligned = pf.dsp.time_shift(
    hrir_shifted, -np.min(onsets) / hrir_shifted.sampling_rate + .001,
    unit='s')

plt.figure()
ax = pf.plot.time(hrir_aligned[idx], unit='ms')
ax.axvline(1, color='k', linestyle='--',
           label='1 ms: desired onset time for frontal HRIR')
ax.set_xlim(0, 5)
ax.legend()
ax.set_title('Aligned fontal HRIR')
plt.show()
### END SOLUTION

### BEGIN SOLUTION
# align HRIRs and visualize results
hrir_aligned = pf.dsp.time_shift(
    hrir_shifted, -np.min(onsets) / hrir_shifted.sampling_rate + .001,
    unit='s')

plt.figure()
ax = pf.plot.time(hrir_aligned, unit='ms')
ax.axvline(1, color='k', linestyle='--',
           label='1 ms: desired onset time for frontal HRIR')
ax.set_xlim(0, 5)
ax.legend()
ax.set_title('Aligned HRIRs')
plt.show()
### END SOLUTION

# find the earliest onset in the HRIR dataset
### BEGIN SOLUTION
onsets = pf.dsp.find_impulse_response_start(hrir_aligned, threshold=20)
onsets_min = np.min(onsets) / hrir_aligned.sampling_rate  # onset in seconds
### END SOLUTION

# window the HRIRs
### BEGIN SOLUTION
times = (onsets_min - .00025,  # start of fade-in
         onsets_min,           # end if fade-in
         onsets_min + .0048,   # start of fade_out
         onsets_min + .0058)   # end of_fade_out
hrir_windowed, window = pf.dsp.time_window(
    hrir_aligned, times, 'hann', unit='s', crop='end', return_window=True)
### END SOLUTION

# plot windowed HRIRs and window
### BEGIN SOLUTION
plt.figure()
ax = pf.plot.time(hrir_windowed, unit='ms', dB=True)
pf.plot.time(window, unit='ms', dB=True, color='k', linestyle='--',
             label='window')
ax.set_xlim(0, 10)
plt.title('Windowed HRIRs')
ax.legend()
plt.show()
### END SOLUTION

# plot HRTFs before and after windowing
### BEGIN SOLUTION
idx = 1
freq_scale = 'log'
plt.figure()
ax = pf.plot.freq(hrir_aligned[idx], freq_scale=freq_scale, color=[.6, .6, .6])
pf.plot.freq(hrir_windowed[idx], freq_scale=freq_scale,
             label=['windowed left', 'windowed right'])
ax.set_title(f'Aligned and windowed HRIRs (position {idx})')
ax.set_xlim(50, 22050)
plt.legend()
plt.show()
### END SOLUTION

# Compute spherical head transfer functions using `spherical_head()`
### BEGIN SOLUTION
shtf = spherical_head(
    source_positions,
    n_samples=hrir_windowed.n_samples,
    sampling_rate=hrir_windowed.sampling_rate)

idx = 1
plt.figure()
ax = pf.plot.time_freq(shtf[idx])
ax[1].set_ylim(-10, 10)
plt.show()
### END SOLUTION

### BEGIN SOLUTION
# frequency below which the data is extrapolated
f_extrap = 400
mask_extrap = hrir_windowed.frequencies >= f_extrap

# frequency below which the HRTF magnitude is assumed to be constant
f_target_idx = hrir_windowed.find_nearest_frequency(150)
f_target = hrir_windowed.frequencies[f_target_idx]

# valid frequencies and magnitude values
frequencies = hrir_windowed.frequencies[mask_extrap]
magnitude = np.abs(
    hrir_windowed.freq[..., mask_extrap])

# concatenate target gains and frequencies
magnitude = np.concatenate((
    np.abs(shtf.freq[..., 0, None]),    # 0 Hz
    np.abs(shtf.freq[..., 1, None]),    # f_target
    magnitude), -1)
frequencies = np.concatenate(([0], [f_target], frequencies))

# interpolate magnitude
magnitude_interpolated = np.empty_like(hrir_windowed.freq)
for source in range(magnitude.shape[0]):
    for ear in range(magnitude.shape[1]):
        magnitude_interpolated[source, ear] = np.interp(
            hrir_windowed.frequencies, frequencies, magnitude[source, ear])

# apply new magnitude response
hrir_extrapolated = hrir_windowed.copy()
hrir_extrapolated.freq = \
    magnitude_interpolated * np.exp(1j * np.angle(hrir_windowed.freq))

# plot HRTFs before and after low-frequency extrapolation
idx = 1
freq_scale = 'log'
plt.figure()
ax = pf.plot.freq(hrir_aligned[idx], freq_scale=freq_scale, color=[.6, .6, .6])
pf.plot.freq(hrir_extrapolated[idx], freq_scale=freq_scale,
             label=['extrapolated left', 'extrapolated right'])
ax.set_title(f'HRTFs after low-frequency extrapolation (position {idx})')
ax.set_xlim(50, 22050)
ax.legend()
plt.show()
### END SOLUTION

# compute the DVFs using the function `spherical_head()`
### BEGIN SOLUTION
source_positions_far_field = source_positions.copy()
source_positions_far_field.radius = 100
shtf_far_field = spherical_head(
    source_positions_far_field,
    n_samples=hrir_windowed.n_samples,
    sampling_rate=hrir_windowed.sampling_rate)
dvf = shtf_far_field / shtf

# plot distance variation functions
idx = 1
plt.figure()
ax = pf.plot.freq(dvf[idx], label=['left', 'right'])
ax.set_title(f'Distance Variation Functions (position {idx})')
ax.set_ylim(-5, 5)
plt.legend()
plt.show()
### END SOLUTION


# estimate the far field HRTFs by applying the DVFs
### BEGIN SOLUTION
hrir_far_field = hrir_extrapolated * dvf

# plot HRTFs before and after far-field extrapolation
idx = 1
plt.figure()
ax = pf.plot.freq(hrir_extrapolated[idx], color=[.6, .6, .6])
pf.plot.freq(hrir_far_field[idx],
             label=['extrapolated left', 'extrapolated right'])
ax.set_title(f'HRTFs after far-field extrapolation (position {idx})')
ax.set_xlim(50, 22050)
plt.legend()
plt.show()
### END SOLUTION

# window to final length
### BEGIN SOLUTION
n_samples = 256
times = [0, 10, 246, n_samples]
hrir_final = pf.dsp.time_window(
    hrir_far_field, times, 'hann', crop='end')
### END SOLUTION

# compare against full length (pad zeros to increase FFT resolution)
### BEGIN SOLUTION
idx = 1
plt.figure()
ax = pf.plot.freq(hrir_far_field[idx], color=[.6, .6, .6])
pf.plot.freq(hrir_final[idx], label=['final left', 'final right'])
ax.legend(loc='lower left')
ax.set_title(f'final HRTFs (position {idx})')
ax.set_xlim(50, 22050)
plt.show()
### END SOLUTION

# compute the CTF
### BEGIN SOLUTION
ctf = pf.dsp.average(hrir_final, mode='power', caxis=0)
ctf_inverse = pf.dsp.regularized_spectrum_inversion(ctf, [0, 21000])
ctf_inverse = pf.dsp.minimum_phase(ctf_inverse, truncate=False)
### END SOLUTION

# plot CTF and CTF inverse
### BEGIN SOLUTION
plt.figure()
ax = pf.plot.freq(ctf, color='k')
pf.plot.freq(ctf_inverse, label=['CTF inverse left', 'CTF inverse right'])
ax.set_xlim(50, 22050)
ax.legend()
ax.set_title('Common Transfer Functions')
plt.show()
### END SOLUTION

# compute CTFs
### BEGIN SOLUTION
dir = hrir_final * ctf_inverse
### END SOLUTION

# compare HRTFs and DTFs
### BEGIN SOLUTION
idx = 1
plt.figure()
ax = pf.plot.freq(hrir_final[idx], color=[.6, .6, .6])
pf.plot.freq(dir[idx], label=['DTF left', 'DTF right'])
ax.legend()
ax.set_title('Common Transfer Functions')
ax.set_title(f'HRTFs and DTFs (position {idx})')
ax.set_xlim(50, 22050)
plt.show()
### END SOLUTION
