"""
Obtain the transfer function of a tree by deconvolving a recording with its reference.
Plots the transformation from the raw signal to the tranfer function for an example tree.
"""
from pathlib import Path
from matplotlib import pyplot as plt
import pyfar as pf
import slab
data_dir = Path.cwd() / 'data'

# specify file names
recording_name = '353_8.1_255W'
reference_name = '353_8.1_255W_ref'
window_size=120
show=True

# load recording and reference signal as slab.Sound object:
recording = slab.Sound.read(data_dir / recording_name / f'{recording_name}_rec.wav')
reference = slab.Sound.read(data_dir / reference_name / f'{reference_name}.wav')

def compute_tf(recording, reference, window_size, show):
    """
    Compute the Transfer Function of the system.
    :param recording: the raw recording of from the system (tree)
    :param reference: the reference recording to be removed from the recording (no tree)
    :param window_size: the window to apply to the impulse response to remove late reflections
    :param show: whether to plot the resulting tf
    :return: tf - the resulting transfer function
    """
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
    if show:
        plt.figure()
        ax = pf.plot.time_freq(ir_deconvolved, unit='samples')
        ax[0].set_xlim(0, 1e3)
        ax[0].set_title('raw tf as IR in time domain and TF in frequency domain')
        ax[1].set_ylim(-40, 20)
    # II
    # Acoustic measurements usually contain reflections from the measurement equipment itself
    # (other loudspeakers, supporting construction, etc.) or from the environment (other trees, floor, etc.).
    # Reflections show up in the impulse response as peaks that follow the direct sound. In the spectrum they cause a
    # ripple (comb-filter) effect.
    # Window (shorten) the IR and find a window that is as long as possible to maintain the frequency response at low
    # frequencies, and  short enough to discard the reflection(s). Plot your result to see the effect of the time window.
    # apply time window to the IR:
    ir_windowed = pf.dsp.time_window(ir_deconvolved, (0, window_size), 'boxcar', unit='samples', crop='window')
    # pad to original length for plotting and further processing:
    ir_windowed = pf.dsp.pad_zeros(ir_windowed, ir_deconvolved.n_samples-ir_windowed.n_samples)
    if show:
        plt.figure()
        ax = pf.plot.freq(ir_windowed)
        ax.set_xlim(0, 2.1e4)
        ax.set_ylim(-40, 20)
        plt.title('windowed tf')
    if show:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        # 1. Raw recording
        time_axis = recording.times
        axs[0].plot(time_axis, recording.time[0])
        axs[0].set_title('Raw Recording (Time Domain)')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_xlim(0, time_axis[-1])

        # 2. Impulse responses
        ir_time = ir_deconvolved.times
        axs[1].plot(ir_time, ir_deconvolved.time[0], label='Raw IR')
        axs[1].plot(ir_time, ir_windowed.time[0], label='Windowed IR', linestyle='--')
        axs[1].set_title('Impulse Response (Raw and Windowed)')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[1].set_xlim(0, 0.05)

        # 3. Transfer functions (frequency domain) using pyfar's freq plot
        ax3 = axs[2]
        pf.plot.freq(ir_deconvolved, ax=ax3)
        pf.plot.freq(ir_windowed, ax=ax3, linestyle='--')
        ax3.set_title('Transfer Function (Raw and Windowed)')
        ax3.legend(['Raw TF', 'Windowed TF'])
        ax3.set_xlim(20, 20000)
        ax3.set_ylim(-40, 20)

        plt.tight_layout()
        output_path = Path.cwd() / f"{recording_name}_tf_summary.png"
        plt.savefig(output_path, dpi=300)
        print(f"✅ Summary figure saved to: {output_path}")

    # todo see which time window works
    # think about reflections in the arboretum
compute_tf(recording, reference, window_size, show)
plt.show()
