"""
    Module which contains functions relating to transfer-function measurements
    using Exponential Sine Sweep methods, as presented in [1] and [2].

    References
    ----------
    [1] A. Farina, “Advancements in impulse response measurements by sine
    sweeps,” Audio Engineering Society Convention 122, p. 22, 2007.
    [2] A. Novak, P. Lotton, and L. Simon, “Synchronized Swept-Sine: Theory,
    Application, and Implementation,” J. Audio Eng. Soc., vol. 63, no. 10,
    pp. 786–798, Nov. 2015
"""


import math
import numpy as np
from scipy import signal as sig
from typing import Tuple


def ess_gen_farina(f_start: int, f_final: int, t_sweep: float, t_idle: float,
                   fs: int, fade_in: int = 0, cut_zerocross: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single Exponential Sine Sweep (ESS) and the inverse signal,
    which is used to calculate Room Impulse Responses (RIRs) according to
    Farina [1].

    Parameters
    ----------
        f_start: int
            Starting frequency in Hz.
        f_final: int
            Final frequency in Hz.
        t_sweep: float
            Duration of the sweep in seconds.
        fs: int
            Sampling frequency in Hz.
        fade_in: int, optional
            Number of window length fade in samples. Defaults to zero.
        cut_at_zerocross: bool, optional
            If this flag is set to `True`, cut ESS at the last zero-crossing,
            reducing the signal duration. This is done to prevent abrupt
            termination of the ESS (resulting in pulsive sound). Defaults to
            `False`.
    Returns
    -------
        sweep: np.ndarray
            Generated Exponential Sine Sweep.
        inverse: np.ndarray
            Inverse signal, which is the scaled time-inverse of the ESS.
    """
    R = float(f_final) / f_start
    C = (2 * f_final * math.log(R)) / ((f_final - f_start) * t_sweep)
    t = np.linspace(0, t_sweep, int(np.floor(fs * t_sweep)))
    sweep = np.sin(((2 * math.pi * f_start * t_sweep) / math.log(R)) * (
                   np.power(R, (t / t_sweep)) - 1))

    if (fade_in > 0):
        sweep[0:fade_in] = (sweep[0:fade_in] * np.sin(np.linspace(0, np.pi / 2,
                            fade_in)))

    if (cut_zerocross):
        for idx, sample in enumerate(sweep[::-1]):
            if abs(sample) < 0.001:
                max_freq = (f_start * math.exp((t[-idx-1] / t_sweep) *
                                               math.log(R)))
                print("Warning: sweep cutoff at last zero-crossing. Final "
                      f"frequency is: {np.floor(max_freq)} Hz")
                sweep[-idx:] = np.zeros(idx)
                break

    inverse = C * np.power(R, -(t/t_sweep)) * np.flip(sweep)

    # Add idle time after ESS
    sweep = np.append(sweep, np.zeros(t_idle * fs))
    return (sweep, inverse)


def ess_parse_farina(sweep: np.ndarray, inverse: np.ndarray, t_sweep: float,
                     t_idle: float, fs: int, causality: bool = False
                     ) -> np.ndarray:
    """
    Process the input Exponential Sine Sweep (ESS) and output the
    resulting Room Impulse Response (RIR) according to Farina [1].

    Parameters
    ----------
        sweep: np.ndarray
            Input Exponential Sine Sweep (ESS).
        inverse: np.ndarray
            Inverse signal of the ESS.
        t_sweep: float
            Duration of the active sweep in seconds.
        t_idle: float
            Idle time in seconds following a single ESS.
        fs: int
            Sampling frequency in Hz
        causality: bool, optional
            If this flag is set to `True`, only return the causal part of the
            RIR. Otherwise, return the full RIR. Defaults to `False`.
    Returns
    -------
        rir: np.ndarray
            The resulting Room Impulse Response.
    """
    if sweep.ndim > 1:
        raise Exception("Input has more than one dimension. Please input"
                        "a one dimensional vector containing the ESS.")
    duration = int(np.floor((t_sweep + t_idle) * fs))
    rir = np.array(sig.fftconvolve(sweep[:duration], inverse,
                                   mode='full'))
    if causality:
        rir = rir[int(np.floor(t_sweep * fs)):duration]
    return rir.real


def ess_gen_novak(f_start: int, f_final: int, t_sweep: float, t_idle: float,
                  fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Synchronized Exponential Sine Sweep (ESS) and its inverse-time
    signal, which is used to calculate Room Impulse Responses (RIRs) according
    to Novak et al. [2].

    Parameters
    ----------
        f_start: int
            Starting frequency in Hz.
        f_final: int
            Final frequency in Hz.
        t_sweep: float
            Estimated duration of the sweep in seconds.
        t_idle: float
            Idle time in seconds following a sweep. This idle time captures
            the remaining reverberation of the room.
        fs: int
            Sampling frequency in Hz.
    Returns
    -------
        sweep: np.ndarray
            Generated Exponential Sine Sweep.
        inverse_spec: np.ndarray
            Inverse filter of the generated sweep in the frequency domain.
    """

    # Generate sweep signal using eqs. (47) and (49), described in [2].
    L = round(f_start / np.log(f_final / f_start) * t_sweep) / f_start
    t_sweep = L * np.log(f_final / f_start)

    n_sweep = int(np.round(t_sweep * fs))
    n_total = n_sweep + t_idle * fs

    t = np.arange(0, int(np.ceil(t_sweep * fs))) / fs
    signal = np.zeros(n_total)

    # Calculate Synchronized ESS
    signal[:n_sweep] = np.sin(2 * np.pi * f_start * L * np.exp(t[:n_sweep]
                              / L))

    fft_length = int(2**np.ceil(np.log2(signal.shape[0])))
    f_axis = fs * np.arange(0, fft_length) / fft_length

    # Ignore division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Inverse filter in frequency domain (single-sided spectrum) (eq. 43)
        inverse_spec = (2 * np.sqrt(f_axis / L) *
                        np.exp(-2j * np.pi * f_axis * L *
                        (1 - np.log(f_axis / f_start)) + 1j * np.pi / 4))
    inverse_spec[0] = 0  # Eliminate Inf DC component in spectrum
    return (signal, inverse_spec)


def ess_parse_novak(sweep: np.ndarray, inverse_spec: np.ndarray,
                    fs: int, t_idle: float = 0,
                    causality: bool = False) -> np.ndarray:
    """
    Calculated the Room Impulse Response (RIR) by convoluting the input
    Synchronized Exponential Sine Sweep (ESS) with the inverse signal,
    according to Novak et al. [2].

    Parameters
    ----------
        sweep: np.ndarray
            Input Exponential Sine Sweep (ESS).
        inverse_spec: np.ndarray
            Inverse filter of the generated sweep in the frequency domain.
        fs: int
            Sampling frequency in Hz
        t_idle: float, optional
            Idle time in seconds following a sweep. This idle time captures
            the remaining reverberation of the room. Defaults to `0`.
        causality: bool, optional
            If this flag is set to `True`, only return the causal part of the
            RIR. Otherwise, return the full RIR. Defaults to `False`.
    Returns
    -------
        rir: np.ndarray
            The resulting Room Impulse Response.
    """
    fft_length = int(2 ** np.ceil(np.log2(sweep.shape[0])))

    # Convert signal to FFT domain
    X = np.fft.fft(sweep, n=fft_length) / fs
    pos_freq_spec = (X * inverse_spec)
    h = np.fft.irfft(pos_freq_spec, n=fft_length)
    if causality:
        return h[:t_idle * fs]
    else:
        return np.fft.ifftshift(h)
