"""
Module which implements the Spatial Decomposition Method by Tervo et al.

References
----------
[1] S Tervo, J Pätynen and T Lokki: Spatial Decomposition Method for Room
    Impulse Responses. J. Audio Eng. Soc 61(1):1–13, 2013.

"""
import itertools

import numpy as np
from scipy.linalg import hankel

import localization as loc


def spatial_decomposition_method(rirs: np.ndarray, ref_rir: np.ndarray,
                                 mic_array: np.ndarray, fs: int,
                                 threshold_db: float = 60,
                                 win_size: int = None, c: float = 343.):
    """
    Compute the Spatial Decomposition Method by Tervo et al. This method
    divides an input Room Impulse Response (RIR) in small short-time windows
    and computes the Direction of Arrival (DoA) for every short-time window.
    This function returns a DoA estimate for every sample in the reference RIR.

    Parameters
    ----------
    rirs: np.ndarray
        Input Room Impulse Responses measured using the input
        mic_array. Shape of `rirs` needs to be equal to shape of
        `mic_array` (N x M), where N is the length of the microphone
        signal and M is the number of microphones.
    ref_rir: np.ndarray
        Reference pressure signal of size (N x 1). This pressure signal
        has to be located in the geometric center of the microphone array.
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    fs: int
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    threshold_db: float, optional
        The specified correlation threshold in dB. Correlation values lower
        than this theshold will be omitted from the TDOA based DOA estimation.
        The default threshold is -60 dB.
    win_size: int, optional
        Define the window size of the SDM analysis in samples. This window
        size should be larger than the propagation time of a sound wave
        through the microphone array. If this variable is not set, the default
        window size equals the maximum propagation time + 8 samples.
    c: float, optional
        Speed of sound in meters per second. Defaults to 343.0 m/s.
    Returns
    -------
    doa: np.ndarray of floats
        Returns the estimated Direction of Arrival in Carthesian coordinates
        (Nx3), where N is the number of samples in the input Room Impulse
        Response.
    """
    max_td, V = loc.get_propagation_time(mic_array, fs, c)
    rir_size = rirs.shape[0]
    num_mics = mic_array.shape[0]

    win_size = win_size if win_size else max_td + 8
    win = np.hanning(win_size)

    direct_sound_amplitude = np.max(np.abs(ref_rir))

    num_frames = rir_size - win_size + 1
    frames = np.array([hankel(rirs[:num_frames, mic], rirs[-win_size:, mic]).T
                       for mic in range(num_mics)]).T

    # perform windowing
    frames = np.einsum("ijk, j -> ijk", frames, win)
    # Get all possible microphone pairs P
    mic_pairs = np.array(list(itertools.combinations(range(mic_array.shape[0]),
                                                     2)))

    distances = (np.arange(1, num_frames + 1) + win_size // 2) / fs * c

    threshold = (direct_sound_amplitude ** 2) * (10 ** (-abs(threshold_db)
                                                        / 10))

    tdoas = np.zeros((num_frames, mic_pairs.shape[0]))
    for idx, frame in enumerate(frames):
        offset = frame.shape[0]
        tdoa_region = offset + np.arange(-max_td, max_td + 1)

        # Estimate the time difference of arrival using GCC
        r = loc.gcc(frame[:, mic_pairs[:, 0]], frame[:, mic_pairs[:, 1]])
        max_idices = np.argmax(r[tdoa_region, :], axis=0) - max_td

        # Make sure the correlations are sufficiently high w.r.t. the signal
        # amplitude. If too low, TDOA estimation be inaccurate.
        if (r[max_idices + offset, range(r.shape[1])] > threshold).all():
            tau_hat = max_idices / fs
            tdoas[idx] = loc.cc_gaussian_interp(r, tdoa_region, tau_hat, fs)

    doas = np.full((ref_rir.shape[0], 3), np.nan)
    doas[win_size // 2:num_frames + (win_size // 2), :] = (
                                        distances * loc.calculate_doa(tdoas, V)
                                        ).T
    return doas
