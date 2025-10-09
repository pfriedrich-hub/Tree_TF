"""
Compute and visualize the transfer function of a tree by deconvolving a recording
with its reference. Produces a single 3-panel figure using only Pyfar plotting functions.
"""

from pathlib import Path
from matplotlib import pyplot as plt
import pyfar as pf
import slab

# === Paths and parameters ===
data_dir = Path.cwd() / "data"
recording_name = "353_8.1_255W"
reference_name = "353_8.1_255W_ref"
window_size = 120
show = True

# === Load recording and reference signals ===
recording = slab.Sound.read(data_dir / recording_name / f"{recording_name}_rec.wav")
reference = slab.Sound.read(data_dir / reference_name / f"{reference_name}.wav")


def compute_tf(recording, reference, window_size, show):
    """
    Compute the transfer function and plot a 3-panel summary (Pyfar-only).
    """
    # Convert to pyfar.Signal
    reference = pf.Signal(reference.data.T, reference.samplerate)  # = reference
    recording = pf.Signal(recording.data.T, recording.samplerate)  # = ir_recorded

    # Deconvolution (H = Y / X)
    reference_inverted = pf.dsp.regularized_spectrum_inversion(
        reference, frequency_range=(20, 19.75e3)
    )
    ir_deconvolved = recording * reference_inverted

    # Apply time window to remove reflections
    ir_windowed = pf.dsp.time_window(
        ir_deconvolved, (0, window_size), "boxcar", unit="samples", crop="window"
    )
    ir_windowed = pf.dsp.pad_zeros(
        ir_windowed, ir_deconvolved.n_samples - ir_windowed.n_samples
    )

    # === Create 3-panel Pyfar figure ===
    if show:
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))

        # --- Panel 1: Raw recording (time domain)
        pf.plot.time(recording, ax=axs[0])
        axs[0].set_title("Raw Recording (Time Domain)")
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_xlim(0, recording.times[-1])

        # --- Panel 2: Impulse responses (raw + windowed, in samples)
        pf.plot.time(ir_deconvolved, unit="samples", ax=axs[1])
        pf.plot.time(ir_windowed, unit="samples", ax=axs[1], linestyle="--")
        axs[1].set_title("Impulse Responses (Raw and Windowed, unit: samples)")
        axs[1].set_xlabel("Samples")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend(["Raw IR", "Windowed IR"])
        axs[1].set_xlim(0, 1000)  # adjust if needed

        # --- Panel 3: Transfer functions (raw + windowed)
        pf.plot.freq(ir_deconvolved, ax=axs[2])
        pf.plot.freq(ir_windowed, ax=axs[2], linestyle="--")
        axs[2].set_title("Transfer Functions (Raw and Windowed)")
        axs[2].set_xlim(20, 20000)
        axs[2].set_ylim(-40, 20)
        axs[2].legend(["Raw TF", "Windowed TF"])

        plt.tight_layout()
        output_path = Path.cwd() / f"{recording_name}_tf_summary_pyfar.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 3-panel figure saved to: {output_path}")
        # === Create second figure: Pyfar time-frequency comparison ===
    if show:
        fig_tf, ax_tf = plt.subplots(2, 1, figsize=(10, 6))

        # Plot raw IR before windowing
        ax = pf.plot.time_freq(
            ir_deconvolved,
            unit="samples",
            dB_time=True,
            label="Before Windowing",
            ax=ax_tf,
        )
        # Plot windowed IR
        pf.plot.time_freq(
            ir_windowed, unit="samples", dB_time=True, label="After Windowing", ax=ax_tf
        )

        # Customize axes
        ax[0].set_xlim(0, 1e3)  # samples axis limit
        ax[1].set_ylim(-40, 20)  # dB axis limit
        ax[1].legend(loc="lower left")

        plt.tight_layout()
        tf_output_path = Path.cwd() / f"{recording_name}_timefreq_comparison.png"
        fig_tf.savefig(tf_output_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tf)
        print(f"✅ Time-frequency comparison figure saved to: {tf_output_path}")

    return ir_deconvolved, ir_windowed


# === Run computation ===
ir_deconvolved, ir_windowed = compute_tf(recording, reference, window_size, show)
print("✅ Processing complete.")
