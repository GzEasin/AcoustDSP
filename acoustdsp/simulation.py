"""
    Module which contains functions relating to the simulation of virtual
    room acoustics.

    References
    ----------
    [1] J. Allen and D. Berkley, “Image method for efficiently simulating
        small-room acoustics,” The Journal of the Acoustical Society of
        America, vol. 65, pp. 943–950, 1979, doi: 10.1121/1.382599.
    [2] De Sena, E., Antonello, N., Moonen, M. and Van Waterschoot, T., 2015.
        On the modeling of rectangular geometries in room acoustic simulations.
        IEEE/ACM Transactions on Audio, Speech, and Language Processing, 23(4),
        pp.774-786.
    [3] T. I. Laakso, V. Valimaki, M. Karjalainen, and U. K. Laine, “Splitting
        the unit delay [FIR/all pass filters design],” IEEE Signal Process.
        Mag., vol. 13, no. 1, pp. 30–60, Jan. 1996, doi: 10.1109/79.482137.
    [4] V. Valimaki and A. Haghparast, “Fractional Delay Filter Design Based
        on Truncated Lagrange Interpolation,” IEEE Signal Process. Lett.,
        vol. 14, no. 11, pp. 816–819, Nov. 2007, doi: 10.1109/LSP.2007.898856.
    [5] A. D. Pierce, Acoustics: An Introduction to Its Physical Principles
        and Applications. Cham: Springer International Publishing, 2019.
        doi: 10.1007/978-3-030-11214-1.
"""
import numpy as np
from scipy.signal import lfilter


def lagrange_fd_filter(delay: np.ndarray, N: int):
    """
    Calculate the coefficients of a Lagrange Fractional Delay FIR filter
    [3].

    Parameters
    ----------
    delay: np.ndarray
        An `(M, 1)` vector containing the desired fractional delays of in
        samples. For every fractional delay value, a Lagrange FIR filter is
        created.
    N: int
        Specifies the filter order of the Fractional Delay FIR filters.
    Returns
    -------
    filter_taps: np.ndarray
        A `(M, N)` matrix containing M Lagrange FIR filters with N+1 filter
        taps.
    """
    delay = np.atleast_1d(delay)
    filter_taps = np.zeros((N + 1, delay.shape[0]))
    for n in range(N+1):
        filter_taps[n, :] = np.prod([(delay + N//2 - k) / (n - k)
                                    for k in np.arange(0, N+1) if k != n],
                                    axis=0)
    return filter_taps


def lagrange_fd_filter_truncated(delay: np.ndarray, N: int, K: int):
    """
    Calculate the coefficients for an N-order truncated Lagrange Fractional
    Delay (FD) FIR filter. K coefficients are removed from both each end
    of the prototype filter [4]. A trunctated FD filter has a wider
    magnitude response, at the cost of a ripple in the passband of the
    filter.

    Parameters
    ----------
    delay: np.ndarray
        An `(M, 1)` vector containing the desired fractional delays of in
        samples. For every fractional delay value, a truncated Lagrange FIR
        filter is created.
    N: int
        Specifies the filter order of the truncated Fractional Delay FIR
        filters.
    K: int
        Specifies the number of coefficients that are set to zero at each end
        of the Lagrange prototype filter.
    Returns
    -------
    filter_taps: np.ndarray
        A `(M, N + 2K)` matrix containing `M` truncated Lagrange FIR filters.
    """
    delay = np.atleast_1d(delay)
    M = N + 2*K
    filter_taps = np.zeros((M + 1, delay.shape[0]))
    for n in range(N + 1):
        filter_taps[n + K] = np.prod([(delay + M//2 - k) / (n + K - k)
                                     for k in np.arange(0, M+1) if k != n + K],
                                     axis=0)
    return filter_taps


def simulate_direct_sound(distance: np.ndarray, fs: int, N: int = 20,
                          K: int = 0, c: float = 343):
    """
        Simulate the ideal direct sound propagation measured by a microphone at
        a given distance from a sound source.

    Parameters
    ----------
    distance: np.ndarray
        Distances between a sound source and the microphone in meters,
        specified as an (M, 1) vector.
    fs: int
        Sampling frequency in Hertz.
    N: int
        Lagrange fractional delay filter order `N`. Defaults to `N = 20`.
    K: int
        Number of coefficients of the Lagrange fractional delay filter that
        are set to zero at each end of the Lagrange prototype filter.
    Returns
    -------
    signals: np.ndarray
        Return M signals of length `(distance / c + 1) * fs` which represent
        the direct sound propagation between a sound source and a microphone.
    """
    num_signals = distance.shape[0]

    signals = np.array([np.zeros(np.round(distance[i] / c + 1).astype(int)
                                 * fs) for i in range(num_signals)])

    for i in range(num_signals):
        # Add one second to the total duration
        delay = distance[i] / c * fs
        # Get integer and fractional part of the delay in samples
        i_delay = int(delay // 1)
        f_delay = delay - i_delay
        # Dirac delta function at integer delay point
        signals[i][i_delay] = 1
        # Filter dirac delta function with a fractional delay FIR filter
        filter_taps = lagrange_fd_filter_truncated(f_delay, N, K)
        # Account for FIR filter delay
        filter_delay = filter_taps.shape[0] // 2
        signals[i][:-filter_delay] = lfilter(filter_taps.squeeze(), 1,
                                             signals[i])[filter_delay:]
    return signals


def speed_of_sound(temperature: float) -> float:
    """
    Calculate the propagation speed of dry air at a specific temperature,
    expressed in degrees Celcius [5]. Note that this calculation only
    approximates the actual propagation speed.

    Parameters
    ----------
    temperature: float
        The temperature in degrees Celcius.
    Returns
    -------
    c: float
        Speed of sound constant in meters per second.
    """
    return 331 + 0.6 * temperature
