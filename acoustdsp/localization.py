"""
Module which implements several acoustical sound source localization related
methods.

References
----------
[1] C. Knapp and G. Carter, “The generalized correlation method for estimation
    of time delay,” IEEE Trans. Acoust., Speech, Signal Process., vol. 24,
    no. 4, pp. 320–327, Aug. 1976, doi: 10.1109/TASSP.1976.1162830.
[2] Xiaoming Lai and H. Torp, “Interpolation methods for time-delay estimation
    using cross-correlation method for blood velocity measurement,” IEEE Trans.
    Ultrason., Ferroelect., Freq. Contr., vol. 46, no. 2, pp. 277–290, Mar.
    1999, doi: 10.1109/58.753016.
[3] Lei Zhang and Xiaolin Wu, “On Cross Correlation Based Discrete Time Delay
    Estimation,” in Proceedings. (ICASSP ’05). IEEE International Conference on
    Acoustics, Speech, and Signal Processing, 2005., Philadelphia,
    Pennsylvania, USA, 2005, vol. 4, pp. 981–984.
    doi: 10.1109/ICASSP.2005.1416175.

"""
import itertools
import warnings

import numpy as np


def gcc(sig: np.ndarray, refsig: np.ndarray,
        weighting: str = "direct") -> np.ndarray:
    """
    Compute the Generalized Cross-Correlation according to [1].

    Parameters
    ----------
    sig: np.ndarray
        Input signal, specified as an SxN matrix, with S being the number
        of signals of size N.
    refsig: np.ndarray
        Reference signal, specified as a column or row vector of size
        N.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    weighting: str, optional
        Define the weighting function for the generalized
        cross-correlation. Defaults to 'direct' weighting.
    Returns
    -------
    R: np.ndarray
        Cross-correlation between the input signal and the reference
        signal. `R` has a size of (2N-1) x S.
    """
    if (weighting.lower() != "direct" and weighting.lower() != "phat"):
        raise ValueError("This function currently only supports Direct and "
                         "PHAT weighting.")

    fft_len = 2 * max([sig.shape[0], refsig.shape[0]])

    SIG = np.fft.rfft(sig, n=fft_len, axis=0)
    REFSIG = np.fft.rfft(refsig, n=fft_len, axis=0)

    G = np.conj(REFSIG) * SIG   # Calculate Cross-Spectral Density
    W = np.abs(G) if weighting.lower() == "phat" else 1

    # Apply weighting and retrieve cross-correlation.
    R = np.fft.ifftshift(np.fft.irfft(G / W, n=fft_len, axis=0), axes=0)
    return R


def cc_parabolic_interp(R: np.ndarray, tau: float, fs: int = 1):
    """
    Fit a parabolic function of the form: `ax^2 + bx + c` to the maximum
    value of a Cross-Correlation function. Returns the x-position of the
    vertex of the fitted parabolic function [2].

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    R = np.atleast_2d(R)
    max_indices = np.argmax(np.abs(R), axis=0)

    # Retrieve the values around the maximum of R
    y = np.array([R[idx - 1: idx + 2, i] for i, idx in enumerate(max_indices)])

    # Perform parabolic interpolation and return the improved tau value.
    d1 = y[:, 1] - y[:, 0]
    d2 = y[:, 2] - y[:, 0]
    a = -d1 + d2 / 2
    b = 2 * d1 - d2 / 2
    vertices = -b / (2 * a)
    # vertex - 1 is the sample-offset from maximum point of R
    return tau + (vertices - 1) / fs


def cc_gaussian_interp(R: np.ndarray, tau: float, fs: int = 1):
    """
    Fit a gaussian function of the form: `a * exp(-b(x - c)^2)` to the
    maximum value of a Cross-Correlation function. Returns the x-position
    of the vertex of the fitted gaussian function [3].

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    R = np.atleast_2d(R)
    max_indices = np.argmax(np.abs(R), axis=0)

    # Retrieve the values around the maximum of R. R needs to be positive for
    # indices around the maximum value. If this is not the case, take the
    # absolute value of the point (impacts fitting).
    y = np.array([R[idx - 1: idx + 2, i] for i, idx in enumerate(max_indices)])

    if (y < 0).any():
        warnings.warn("Gaussian interpolation encountered negative R values. "
                      "Interpolation may not be correct.", RuntimeWarning)
        y = np.array([p + 2 * abs(np.min(p)) if (p < 0).any() else p
                      for p in y])

    c = (np.log(y[:, 2]) - np.log(y[:, 0])) / (4 * np.log(y[:, 1]) - 2
                                               * np.log(y[:, 0]) - 2
                                               * np.log(y[:, 2]))
    # vertex - 1 is the sample-offset from maximum point of R
    return tau + c / fs


def cc_sinc_interp(R: np.ndarray, tau: float, interp_mul: int, fs: int,
                   half_width: float = 0.002):
    """
    Fit a critically sampled sinc function to the maximum value of the
    cross-correlation function. Returns the improved time-delay found by the
    fitting.

    Parameters
    ----------
    R: np.ndarray
        Input cross-correlation signal. R has a size of (2N-1) x S. Where
        S is the number of cross-correlations.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    interp_mul: int
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    half_width: float
        interpolation half width of the sinc fitting. Specifies the maximum
        time-delay to fit the sinc funtion around the maximum of the
        cross-correlation function.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if(interp_mul <= 0):
        raise ValueError("Interpolation multiplier has to be a strictly"
                         " positive integer.")

    R = np.atleast_2d(R)
    max_ind = np.argmax(np.abs(R), axis=0)

    fs_res = fs * interp_mul
    max_ind_res = max_ind * interp_mul

    # Search 10 samples around the direct path component
    n_margin_res = int(5 * interp_mul)
    search_area = np.array([d + np.arange(-n_margin_res, n_margin_res + 1)
                           for d in max_ind_res]).T / fs_res

    amplitudes = [R[idx, i] for i, idx in enumerate(max_ind)]
    cost_vector = np.zeros(search_area.shape)

    n_half_width = int(half_width * fs)
    window = np.array([idx + np.arange(-n_half_width, n_half_width + 1)
                       for idx in max_ind]).T
    t = window / fs

    for i, r in enumerate(R.T):
        for j, t_0 in enumerate(search_area[:, i]):
            cost_vector[j, i] = np.sum(np.square(np.sinc(fs * (t[:, i] - t_0))
                                       - r[window[:, i]] / amplitudes[i]))
    minima = np.argmin(cost_vector, axis=0)
    return (minima - n_margin_res) / fs_res + tau


def calculate_tdoa(rirs: np.ndarray, mic_array: np.ndarray, fs: int = 1,
                   c: float = 343, weighting: str = "direct",
                   interp: str = "None"):
    """
    Calculate the Time Difference of Arrival using the
    Generalized Cross-Correlation method.

    Parameters
    ----------
    rirs: np.ndarray
        Input Room Impulse Responses measured using the input
        mic_array. Shape of `rirs` needs to be equal to shape of
        `mic_array` (N x M), where N is the length of the microphone
        signal and M is the number of microphones.
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    c: float, optional
        Speed of sound in meters per second. Defaults to 343 m/s.
    weighting: str, optional
        Define the weighting function for the generalized
        cross-correlation. Defaults to 'direct' weighting.
    interp: str, optional
        Specify which interpolation method to use for improving the
        TDOA. Possible interpolation methods are: `None`, `Parabolic` or
        `Gaussian`. Defaults to `None`.
    Returns
    -------
    tau_hat: np.ndarray
        Time Difference of Arrival between all microphone pairs. The
        number of microphone pairs is: num_mics * (num_mics - 1) / 2
    """
    if (rirs.shape[1] != mic_array.shape[0]):
        raise ValueError("First dimension of rirs and mic_array needs to be "
                         "equal.")

    if (weighting.lower() not in ["direct", "phat"]):
        raise ValueError("This function currently only supports Direct and "
                         "PHAT weighting.")

    if (interp.lower() not in ["none", "parabolic", "gaussian"]):
        raise ValueError("This function currently only supports Parabolic and "
                         "Gaussian interpolation.")
    # All possible microphone pairs P
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    # Get the maximum time-delay in samples for the input microphone array.
    max_td, _ = _mic_array_properties(mic_array, fs, c)

    offset = rirs.shape[0]
    tdoa_region = offset + np.arange(-max_td, max_td + 1)

    # Estimate the time difference of arrival using GCC
    r = gcc(rirs[:, mic_pairs[:, 0]], rirs[:, mic_pairs[:, 1]])

    max_idices = np.argmax(np.abs(r[tdoa_region, :]), axis=0)
    tau_hat = (max_idices - max_td) / fs

    # Perform interpolation method
    if interp.lower() == "parabolic":
        tau_hat = cc_parabolic_interp(r, tau_hat, fs)
    elif interp.lower() == "gaussian":
        tau_hat = cc_gaussian_interp(r, tau_hat, fs)
    # Return estimated TDOA
    return tau_hat


def calculate_doa(tau_hat: np.ndarray, mic_array: np.ndarray, fs: int = 1,
                  c: float = 343):
    """
    Calculate the DOA from the slowness vector obtained with the
    Cross-Correlation method.

    Parameters
    ----------
    tau_hat: np.ndarray
        Estimated Time Difference of Arrival between all microphone pairs.
        The number of microphone pairs is: num_mics * (num_mics - 1) / 2.
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    c: float, optional
        Speed of sound in meters per second. Defaults to 343 m/s.
    Returns
    -------
    doa: np.ndarray of floats
        Estimated Direction of Arrival in Carthesian coordinates (1xD)
    """
    # Get the sensor vector matrix of the mic_array
    _, V = _mic_array_properties(mic_array, fs, c)
    # Calculate slowness vector k
    k_hat = np.inner(np.linalg.pinv(V), tau_hat)
    # Calculate and return direction of arrival
    return k_hat if np.sum(k_hat) == 0 else -k_hat / np.linalg.norm(k_hat, 2)


def _mic_array_properties(mic_array: np.ndarray, fs: int = 1, c: float = 343.):
    """
    Calculates the maximum time it takes for a wavefront to propagate
    through a given microphone array. The input of `mic_array` is a
    (M x D) matrix, where M is the number of microphones and D is the
    number of spatial dimensions.

    Parameters
    ----------
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.

    c: float, optional
        Speed of sound in meters per second. Defaults to 343.0 m/s.
    Returns
    -------
    max_td: int
        The maximum time in samples for a wavefront to propagate through
        the given microphone array configuration.
    V: np.ndarray
        The sensor vector matrix in 3D space. V is an (P x D)
        matrix, with P number of microphone pairs and D physical
        dimensions.
    """
    # All possible microphone pairs P
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))
    # Define the sensor vector matrix in 3D space (P x D)
    V = mic_array[mic_pairs[:, 0], :] - mic_array[mic_pairs[:, 1], :]
    # Maximum distance and time delay
    max_distance = max(np.linalg.norm(V, 2, axis=1))
    # Maximum accepted time delay difference
    max_td = int(np.ceil(max_distance / c * fs)) + 1
    return (max_td, V)
