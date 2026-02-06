"""
File: custom_baseline_methods.py
Author: Oliver Bayley - Noël Research Group - 2024
GitHub: https://github.com/ombayley

Description: Custom baseline correction methods for NMR data.
"""

import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import nmrglue as ng


def baseline(data, method="als", **kwargs):
    """
    Apply baseline correction to the given data using the specified method.

    Parameters
    ----------
    data : array_like
        Input data array (e.g., NMR spectrum).
    method : str, optional
        Baseline correction method to use. Available methods are:
        - 'als'        : Asymmetric Least Squares
        - 'whittaker'  : Whittaker Smoothing
        - 'polynomial' : Polynomial Fitting
        - 'savgol'     : Savitzky-Golay Filter
        - 'flatfit'    : FlatFit Algorithm
        - 'median'     : Median Filter (using nmrglue)

        Default is 'als'.
    **kwargs
        Additional keyword arguments specific to the chosen method.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.

    Raises
    ------
    ValueError
        If an unknown method is specified.

    Examples
    --------
    >>> corrected_data, baseline = baseline(spectrum_data, method='savgol', window_length=101, polyorder=2)
    """
    methods = {
        "als": _baseline_als,
        "whittaker": _baseline_whittaker,
        "polynomial": _baseline_polynomial_correction,
        "savgol": _baseline_sav_gol,
        "flatfit": _baseline_flatfit,
        "median": _baseline_median,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown baseline correction method '{method}'. "
            f"Available methods are: {list(methods.keys())}"
        )

    # Call the selected baseline correction method with additional arguments
    corrected_data, baseline = methods[method](data, **kwargs)
    return corrected_data, baseline


def _baseline_polynomial_correction(data, degree=0):
    """
    Perform polynomial baseline correction.

    Parameters
    ----------
    data : array_like
        Input data array (e.g., NMR spectrum).
    degree : int, optional
        Degree of the polynomial to fit. Default is 2.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.
    """
    x = np.arange(len(data))
    coefficients = np.polyfit(x, data, degree)
    baseline = np.polyval(coefficients, x)
    corrected_data = data - baseline
    return corrected_data, baseline


def _baseline_als(data, lam=1e50, p=0.5, niter=1):
    """
    Asymmetric Least Squares (ALS) baseline correction.

    Parameters
    ----------
    data : array_like
        Input data (e.g., NMR spectrum).
    lam : float, optional
        Smoothness parameter. Larger values make the baseline smoother. Default is 1e5.
    p : float, optional
        Asymmetry parameter between 0 and 1. Closer to 0 places less weight on positive deviations. Default is 0.01.
    niter : int, optional
        Number of iterations to perform. Default is 10.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.
    """
    y = np.array(data)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L), format="csc")
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    baseline = z
    corrected_data = y - baseline
    return corrected_data, baseline


def _baseline_whittaker(y, lam=100, d=2, p=0.01, max_iter=1):
    """
    Perform baseline correction using the Whittaker smoother with asymmetric least squares.

    Parameters:
    ----------
    y : array_like
        Input data (e.g., NMR spectrum).
    lam : float, optional
        Smoothing parameter for the Whittaker smoother. Default is 1e6.
    d : int, optional
        The order of differences to use for smoothing (e.g., d=2 for second differences).
    p : float, optional
        Asymmetry parameter for penalizing negative deviations from the baseline. Default is 0.01.
    max_iter : int, optional
        Maximum number of iterations for the correction. Default is 10.

    Returns:
    -------
    corrected : ndarray
        The baseline-corrected data.
    baseline : ndarray
        The estimated baseline.
    """
    y = np.array(y)
    L = len(y)
    w = np.ones(L)  # Initialize weights as ones

    # Create difference matrix D for the specified order `d`
    E = sparse.eye(L, format="csc")
    D = E
    for _ in range(d):  # Construct dth order difference matrix
        D = D[1:] - D[:-1]

    for i in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)  # Create diagonal matrix of weights
        # Smooth using the Whittaker smoother
        A = W + lam * (D.T @ D)
        baseline = spsolve(A, W @ y)

        # Update the weights based on asymmetric least squares
        w = p * (y > baseline) + (1 - p) * (y < baseline)

    corrected = y - baseline
    return corrected, baseline


def _baseline_sav_gol(data, window_length=5001, polyorder=0):
    """
    Savitzky-Golay filter baseline correction.

    Parameters
    ----------
    data : array_like
        Input data (e.g., NMR spectrum).
    window_length : int, optional
        Length of the filter window (number of coefficients). Must be a positive odd integer. Default is 101.
    polyorder : int, optional
        Order of the polynomial used to fit the samples. Default is 2.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.

    Raises
    ------
    ValueError
        If `window_length` is not a positive odd integer, or if `polyorder` is greater than `window_length`.
    """
    if window_length % 2 == 0 or window_length <= 0:
        raise ValueError("window_length must be a positive odd integer.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    baseline = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    corrected_data = data - baseline
    return corrected_data, baseline


def _baseline_flatfit(data, smoothness=1e5, p=0.01):
    """
    FlatFit baseline correction.

    Parameters
    ----------
    data : array_like
        Input data (e.g., NMR spectrum).
    smoothness : float, optional
        Smoothness parameter. Default is 1e5.
    p : float, optional
        Relative size parameter for the Savitzky-Golay filter. Default is 0.01.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.
    """
    y = np.array(data)
    L = len(y)

    # Scale lambda to maintain invariance for sampling frequency
    lamb = smoothness * L**4

    # Calculate weights based on slope and curvature
    filter_window = max(5, int(L * p) | 1)  # Ensure filter_window is odd and at least 5
    slope = savgol_filter(y, filter_window, 3, deriv=1)
    curvature = np.gradient(slope)
    slope_squared = slope**2
    curvature_squared = curvature**2

    # Normalize
    slope_norm = slope_squared / np.sum(slope_squared)
    curvature_norm = curvature_squared / np.sum(curvature_squared)

    w = 1 / (slope_norm + curvature_norm + 1e-10)

    # Calculate baseline
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L - 2, L))
    H = lamb * D.T @ D

    W = sparse.spdiags(w, 0, L, L)
    Z = W + H
    z = spsolve(Z, w * y)

    baseline = z
    corrected_data = y - baseline
    return corrected_data, baseline


def _baseline_median(data, mw=1000, sf=20, sigma=1):
    """
    Median filter baseline correction using nmrglue.

    Parameters
    ----------
    data : array_like
        Input data (e.g., NMR spectrum).
    mw : int, optional
        Moving window size. Default is 100.
    sf : int, optional
        Smoothing factor. Default is 20.
    sigma : float, optional
        Gaussian smoothing parameter. Default is 1.

    Returns
    -------
    corrected_data : ndarray
        Baseline-corrected data.
    baseline : ndarray
        Estimated baseline.
    """
    # create extrema array (non extrema values are masked out)
    mask = data == scipy.ndimage.median_filter(data, size=3)
    mask[0] = False  # first pt always extrema
    mask[-1] = False  # last pt always extrema
    e = np.ma.masked_array(data, mask)

    # fill in the median vector
    m = scipy.ndimage.median_filter(e, mw + 1, mode="mirror")
    # using the median_filter might give slightly different results than
    # described algorithm but is MUCH faster

    # convolve with a gaussian
    g = scipy.signal.windows.gaussian(sf, sigma)
    g = g / g.sum()
    baseline = scipy.signal.convolve(m, g, mode="same")

    corrected_data = data - baseline
    return corrected_data, baseline


def _baseline_x(intensity_data):
    corrected_data = ng.proc_bl.baseline_corrector(data=intensity_data, wd=5)
    return corrected_data, None
