import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import chirp, fftconvolve
import sounddevice as sd

# --- Parameters ---
fs = 48000
T_sweep = 0.05  # 50 ms sweep
t = np.linspace(0, T_sweep, int(fs * T_sweep), endpoint=False)
sweep = chirp(t, f0=100, f1=10000, t1=T_sweep, method="logarithmic")

reflections = []  # list of (delay_samples, amplitude)
base_index = 100
win_ms_default = 10.0


# --- Helper functions ---
def record_with_reflections(sweep, reflections):
    """Convolve sweep with synthetic IR made from reflections."""
    n = len(sweep)
    ir = np.zeros(n)
    ir[base_index] = 1.0
    for d, a in reflections:
        if base_index + d < n:
            ir[base_index + d] += a
    y = fftconvolve(sweep, ir, mode="full")[:n]
    return y, ir


def deconvolve(y, x):
    """Simple spectral deconvolution."""
    S = np.fft.rfft(x, len(y))
    Y = np.fft.rfft(y)
    H = Y / (S + 1e-12)
    h = np.fft.irfft(H)
    return h, H


def apply_window(signal, win_ms):
    win_samples = int(win_ms * 1e-3 * fs)
    window = np.zeros_like(signal)
    window[:win_samples] = 1
    return signal * window


def compute_fft(signal):
    H = np.fft.rfft(signal)
    f = np.fft.rfftfreq(len(signal), 1 / fs)
    return f, 20 * np.log10(np.abs(H) + 1e-12)


def play_ir(ir):
    """Play scaled impulse response."""
    audio = ir / np.max(np.abs(ir))
    sd.play(audio, fs)
    sd.wait()


# --- Initial data ---
y, ir_true = record_with_reflections(sweep, reflections)
ir, H = deconvolve(y, sweep)
h_win = apply_window(ir, win_ms_default)
f, mag = compute_fft(h_win)

# --- Figure setup ---
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.28)

(line_sweep,) = ax[0].plot(sweep, lw=1, label="Sweep + reflections")
(line_ir,) = ax[1].plot(h_win, lw=1.5, label="Impulse Response")
(line_H,) = ax[2].plot(f, mag, lw=1.5, label="|H(f)|")

ax[0].set_title("Click to add reflections (Shift+Click to remove)")
ax[0].set_xlim(0, len(sweep))
ax[1].set_xlim(0, 2000)
ax[2].set_xlim(0, 10000)
ax[2].set_ylim(-40, 20)
for a in ax:
    a.legend()

# --- Sliders ---
ax_win = plt.axes([0.15, 0.15, 0.65, 0.03])
slider_win = Slider(ax_win, "Window (ms)", 1.0, 20.0, valinit=win_ms_default)

ax_amp = plt.axes([0.15, 0.10, 0.65, 0.03])
slider_amp = Slider(ax_amp, "Next reflection amplitude", 0.0, 1.0, valinit=0.5)

# --- Buttons ---
ax_play = plt.axes([0.15, 0.02, 0.1, 0.05])
button_play = Button(ax_play, "Play IR")

ax_reset = plt.axes([0.8, 0.02, 0.1, 0.05])
button_reset = Button(ax_reset, "Reset")


# --- Callbacks ---
def onclick(event):
    if event.inaxes != ax[0]:
        return
    delay_samples = int(event.xdata)
    if event.key == "shift":
        if reflections:
            nearest = min(reflections, key=lambda r: abs(r[0] - delay_samples))
            reflections.remove(nearest)
    else:
        reflections.append((delay_samples, slider_amp.val))
    update(None)


def update(val):
    win_ms = slider_win.val
    y, ir_true = record_with_reflections(sweep, reflections)
    ir, H = deconvolve(y, sweep)
    h_win = apply_window(ir, win_ms)
    f, mag = compute_fft(h_win)
    line_sweep.set_ydata(y / np.max(np.abs(y)))
    line_ir.set_ydata(h_win / np.max(np.abs(h_win)))
    line_H.set_ydata(mag)
    fig.canvas.draw_idle()


def reset(event):
    reflections.clear()
    slider_win.reset()
    slider_amp.reset()
    update(None)


def on_play(event):
    # play windowed IR
    y, ir_true = record_with_reflections(sweep, reflections)
    ir, _ = deconvolve(y, sweep)
    h_win = apply_window(ir, slider_win.val)
    play_ir(h_win)


slider_win.on_changed(update)
button_reset.on_clicked(reset)
button_play.on_clicked(on_play)
fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
