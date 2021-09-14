"""
    Module which contains functions relating to the calculation of Objective
    Room Acoustic parameters.

    References
    ----------
    [1] M. R. Schroeder, “New Method of Measuring Reverberation Time,” The
    Journal of the Acoustical Society of America, vol. 37, no. 3, pp. 409–412,
    Mar. 1965
    [2] A. Gade, “Acoustics in Halls for Speech and Music,” in Springer
    Handbook of Acoustics, Thomas D. Rossing, Ed. New York, NY: Springer New
    York, 2007, pp. 301–350.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from acoustdsp import utils


def energy_decay_curve(rir: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Calculate the energy decay curve from a causal input Room
    Impulse Response (RIR) using the Schroeder inverse-integration method [1].

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        plot: bool, optional
            If set to `True`, the energy decay curve will
            be plotted. Defaults to `False`.
    Returns
    -------
        edc: np.ndarray
            The energy decay curve in dB.
    """
    power = np.square(rir)
    decay_curve = integrate.cumtrapz(power[::-1])[::-1]
    decay_curve_db = utils.mag2db(decay_curve / np.max(decay_curve), 'power')
    if (plot):
        plt.plot(decay_curve_db)
    return decay_curve_db


def rt(rir: np.ndarray, fs: int, db: int, db_start: int = 5, db_end: int = 35,
       plot: bool = False) -> float:
    """
    Calculate the Reverberation Time (RT) from a causal input Room Impulse
    Response (RIR) using the Schroeder inverse-integration method [1]. The RT
    is determined from the decay rate (dB/s), as found when fitting a
    regression line (determined from a relevant interval) to the Energy Decay
    Curve (EDC). The standard RT is calculated by extrapolating the decay rate
    from -5 dB to -35 dB.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        db: int
            Reverberation time threshold.
        db_start: int, optional
            Sound pressure decay starting point for fitting a regression line
            to the EDC. Defaults to `5`.
        db_end: int, optional
            Sound pressure decay endpoint for fitting a regression line to
            the EDC. Defaults to `35`.
        plot: bool, optional
            If set to `True`, the RT calculations will be plotted. Defaults to
            `False`.
    Returns
    -------
        rt: float
            RT in seconds.
    """
    if (abs(db_start) > abs(db_end)):
        raise ValueError("Initial starting dB is larger than the final dB.")

    edc = energy_decay_curve(rir)
    p1 = abs(edc + abs(db_start)).argmin()
    p2 = abs(edc + abs(db_end)).argmin()
    rt = db * float(abs(p2 - p1)) / (fs * (db_end - db_start))

    if plot:
        ax = plt.subplot(111)
        t = (np.arange(0, edc.shape[0]) - p1) / fs
        ax.plot(t, edc, color='k', label="Energy Decay Curve")
        ax.axline((0, edc[p1]), ((p2 - p1) / fs, edc[p2]), color="g",
                  linestyle="--", label="Linear fit")
        ax.axvline(0, color='c', linestyle="--",
                   label=f"-{abs(db_start)} dB reference point")
        ax.axvline((p2 - p1) / fs, color='b', linestyle="--",
                   label=f"-{abs(db_end)} dB reference point")
        ax.axhline(-abs(db + db_start), color='r', linestyle="--",
                   label=f"-{abs(db + db_start)} dB")
        ax.axvline(rt, color='r', linestyle="--",
                   label="$\\widehat{RT}$"f"{abs(db)}")
        ax.legend()
    return rt


def edt(rir: np.ndarray, fs: int) -> float:
    """
    Calculate the Early Decay Time (EDT) of an input Room Impulse Response
    (RIR). The EDT is the time it takes for the RIR to decay to -60 dB. The
    decay rate is calculated using the interval from 0 dB to -10 dB, relative
    to the direct sound.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.

    Returns
    -------
        edt: float
            The EDT in seconds.
    """
    return rt(rir, fs, 60, 0, 10)


def energy_ratio(rir: np.ndarray, early: tuple, late: tuple) -> float:
    """
    Calculate the energy ratio between the early, and late part of the input
    Room Impulse Response (RIR).

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        early: tuple of ints
            The start- and endpoint of the desired early reverberation
            in number of samples.
        late: tuple of ints
            The start- and endpoint of the desired late reverberation
            in number of samples.

    Returns
    -------
        ratio: float
            Ratio between early and late energy.
    """
    power = np.square(rir)
    early_power = np.sum(power[early[0]:early[1]])
    late_power = np.sum(power[late[0]:late[1]])
    return (early_power / late_power)


def clarity(rir: np.ndarray, fs: int, threshold: float = 0.08) -> float:
    """
    Calculate the Clarity of an input Room Impulse Response (RIR). Clarity is
    the ratio between energy in the RIR before and after 80ms relative to the
    direct sound.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        threshold: float, optional
            The time threshold in seconds relative to the direct sound.
            Defaults to 80ms.
    Returns
    -------
        clarity: float
            Clarity expressed in dB.
    """
    direct_idx = abs(rir).argmax()
    threshold_idx = direct_idx + threshold * fs
    return 10 * np.log10(energy_ratio(rir, (direct_idx, threshold_idx),
                         (threshold_idx, -1)))


def definition(rir: np.ndarray, fs: int, threshold: float = 0.08) -> float:
    """
    Calculate the Definition of an input Room Impulse Response (RIR).
    Definition describes the ratio between the early energy, and the
    total energy in a RIR.

    Parameters
    ----------
        rir: np.ndarray
            The input RIR.
        fs: int
            Sampling frequency of the input RIR in Hz.
        threshold: float, optional
            The time threshold in seconds relative to the direct sound.
            Defaults to 80ms.
    Returns
    -------
        definition: float
            Definition expressed as a ratio.
    """
    direct_idx = abs(rir).argmax()
    threshold_idx = direct_idx + threshold * fs
    return energy_ratio(rir, (direct_idx, threshold_idx), (direct_idx, -1))
