# Example: Ideal Exponential Sine Sweep (Farina)
#
# This is an example iPython notebook, illustrating the Exponential Sine Sweep
# method implementation. The result shows a Dirac-like impulse (with minimal
# pre- and post-ringing) [1].
#
# References
# ----------
# [1] A. Farina, “Advancements in impulse response measurements by sine
# sweeps,” Audio Engineering Society Convention 122, p. 22, 2007.

# %% Import libraries
from acoustdsp import sweep, utils
from matplotlib import pyplot as plt
import numpy as np

# %% Initialise variables
FS = 192000
F_START = 1
F_FINAL = FS / 2
T_SWEEP = 15
T_IDLE = 10
FADE_IN = 0

t = np.linspace(0, T_IDLE + T_SWEEP, FS * (T_IDLE + T_SWEEP))
t_inverse = np.linspace(0, T_SWEEP, FS * T_SWEEP)
t_rir = np.linspace(0, 2 * T_SWEEP + T_IDLE, FS * (2 * T_SWEEP + T_IDLE) - 1)

[signal, inverse] = sweep.ess_gen_farina(F_START, F_FINAL, T_SWEEP, T_IDLE,
                                         FS, FADE_IN)

rir = sweep.ess_parse_farina(signal, inverse, T_SWEEP, T_IDLE, FS)

rir_scl = rir / np.max(np.abs(rir))
sig_scl = signal / np.max(np.abs(signal))

# %%
plt.rc("text", usetex=True)
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
axs[0].plot(t, sig_scl)
axs[0].grid(True)
axs[0].set_xlim((0, 30))
axs[0].set_title("Excitation sweep signal $s(t)$")
axs[0].set_xlabel("Time (seconds)")
axs[0].set_ylabel("Normalized amplitude")

axs[1].plot(t_inverse, inverse)
axs[1].grid(True)
axs[1].set_title("Inverse filter $c(t)$")
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Normalized amplitude")

axs[2].plot(t_rir, rir_scl)
axs[2].grid(True)
axs[2].set_title(r"Estimated Room Impulse Response $\hat{h}(t)$")
axs[2].set_xlabel("Time (seconds)")
axs[2].set_ylabel("Normalized amplitude")
plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
utils.spectrogram(rir_scl, FS, 'hann', FS // 100, ax)
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
ax.set_title("Spectrogram Exponential Sine Sweep")
ax.set_xlim((0, 30))
plt.tight_layout()
plt.show()
