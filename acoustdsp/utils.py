"""
Module which implements several utility functions that are used throughout
this library.
"""


import numpy as np
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
from scipy import signal as sig
from typing import Tuple


def mag2db(signal: np.ndarray, input_mode: str = "amplitude") -> np.ndarray:
    """
    Convert magnitude to decibels.

    Parameters
    ----------
        signal: np.ndarray
            Input array, specified as scalar or vector.
        mode: {"amplitude", "power"}, optional
            Express input array as either `amplitude` or
            `power` measurement. Default input array is expressed as
            `amplitude`.
    Returns
    -------
        np.ndarray
            Magnitude measurement expressed in decibels.
    """
    scaling = 10 if input_mode == "power" else 20
    return scaling * np.log10(np.abs(signal))


def normalize(signal: np.ndarray) -> np.ndarray:
    """
    Scale an input array so that it bounds are between [-1, 1].

    Parameters
    ----------
        signal: np.ndarray
            Input signal.
    Returns
    -------
        np.ndarray
            Normalized input signal.
    """
    return signal / (np.max(np.abs(signal)))


def spectrogram(signal: np.ndarray, fs: int, win: str or tuple,
                win_length: int, ax: plt.axes = None,
                clims: Tuple[float, float] = (None, None)
                ) -> Tuple[QuadMesh, Tuple[float, float]]:
    """
    Calculate and draw the spectrogram of the input signal on the
    provided axis.

    Parameters
    ----------
        signal: np.ndarray
            Input signal, specified as a 1-D array.
        fs: int
            Sampling frequency `fs`.
        win: str or tuple
            Desired window to use. For more information, see
            scipy.signal.spectrogram documentation.
        ax: plt.axes, optional
            Matplotlib axes on which the spectrogram will be drawn on. If
            no axes is specified, the spectrogram will be drawn on the
            current axes.
        clims: tuple of floats, optional
            Specify minimum and maximum colormap value respectively, of the
            spectrogram.
    Returns
    -------
        matplotlib.collections.QuadMesh
            Quadrilateral mesh, used for drawing the colorbar of the
            spectrogram.
        tuple of floats (min, max)
            Colormap limits of the spectrogram.
    """
    if (ax is None):
        ax = plt.gca()
    f, t, Sxx = sig.spectrogram(signal, fs, win, win_length)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), vmin=clims[0], vmax=clims[1],
                        shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    return pcm, pcm.get_clim()


def cart2sph(x: float or np.ndarray, y: float or np.ndarray,
             z: float or np.ndarray):
    """
    Transform Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
        x: float or np.ndarray
            Cartesian x-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).
        y: float or np.ndarray
            Cartesian y-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).
        z: float or np.ndarray
            Cartesian z-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).

    Returns
    -------
        azimuth: float or np.ndarray
            Azimuth angle, in radians, measured from the positive x-axis.
            Value ranges from [-pi, pi] as a scalar or np.ndarray of size
            (N x 1).
        elevation: float or np.ndarray
            Elevation angle, in radians, measured from the x-y plane.
            Value ranges from [-pi / 2, pi / 2] as a scalar or np.ndarray of
            size (N x 1).
        r: float or np.ndarray
            Radius. The distance from the origin to a point located in the
            x-y-z plane as a scalar or np.ndarray of size (N x 1).
    """
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return (azimuth, elevation, r)


def sph2cart(azimuth: float or np.ndarray, elevation: float or np.ndarray,
             r: float or np.ndarray):
    """
    Transform spherical coordinates to Cartesian coordinates.

    Parameters
    -------
        azimuth: float or np.ndarray
            Azimuth angle, in radians, measured from the positive x-axis.
            Value ranges from [-pi, pi] as a scalar or np.ndarray of size
            (N x 1).
        elevation: float or np.ndarray
            Elevation angle, in radians, measured from the x-y plane.
            Value ranges from [-pi / 2, pi / 2] as a scalar or np.ndarray of
            size (N x 1).
        r: float or np.ndarray
            Radius. The distance from the origin to a point located in the
            x-y-z plane as a scalar or np.ndarray of size (N x 1).

    Returns
    ----------
        x: float or np.ndarray
            Cartesian x-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).
        y: float or np.ndarray
            Cartesian y-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).
        z: float or np.ndarray
            Cartesian z-coordinate, specified as a scalar or np.ndarray of size
            (N x 1).
    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return (x, y, z)
