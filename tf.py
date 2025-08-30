"""
Obtain the transfer function of a tree by deconvolving a recording with its reference.
"""
from pathlib import Path
from matplotlib import pyplot as plt
import pyfar as pf
import slab
data_dir = Path.cwd() / 'data'

# specify file names
recording_name = '344_3.15_261W'
reference_name = '344_3.15_261W_ref'

# load recording and reference signal as slab.Sound object:
recording = slab.Sound.read(data_dir / recording_name / f'{recording_name}.wav')
reference = slab.Sound.read(data_dir / reference_name / f'{reference_name}.wav')

# convert to pyfar.Signal object:
reference = pf.Signal(data=reference.data.T, sampling_rate=reference.samplerate)
recording = pf.Signal(recording.data.T, sampling_rate=recording.samplerate)

# I
# Obtain the raw TF by means of deconvolution, i.e., H = Y / X
# Where H is the TF (complex spectrum), Y the signal recorded behind the tree, and X the reference signal.
# Note that regularized inversion is often used to compute the inverse 1/X.

# Deconvolve the recording with reference (multiply by inverse) to obtain the TF:
reference_inverted = pf.dsp.regularized_spectrum_inversion(reference, frequency_range=(20, 19.75e3))
ir_deconvolved = recording * reference_inverted  # convolution in time domain = multiplication in frequency domain
# The resulting filter can be viewed in the frequency domain as a transfer function (TF, bottom)
# and in the time domain as an impulse response (IR, top):
plt.figure()
ax = pf.plot.time_freq(ir_deconvolved, unit='samples')
ax[0].set_xlim(0, 1e3)
ax[0].set_title('raw tf')
ax[1].set_ylim(-40, 20)

# II
# Acoustic measurements usually contain reflections from the measurement equipment itself
# (other loudspeakers, supporting construction, etc.) or from the environment (other trees, floor, etc.).
# Reflections show up in the impulse response as peaks that follow the direct sound. In the spectrum they cause a
# ripple (comb-filter) effect.
# Window (shorten) the IR and find a window that is as long as possible to maintain the frequency response at low
# frequencies, and  short enough to discard the reflection(s). Plot your result to see the effect of the time window.

# apply time window to the IR:
ir_windowed = pf.dsp.time_window(ir_deconvolved, (0, 120), 'boxcar', unit='samples', crop='window')
# pad to original length for plotting and further processing:
ir_windowed = pf.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples-ir_windowed.n_samples)

plt.figure()
ax = pf.plot.freq(ir_windowed)
ax.set_xlim(0, 2.1e4)
ax.set_ylim(-40, 20)
plt.title('windowed tf')

# todo see which time window works
# think about reflections in the arboretum