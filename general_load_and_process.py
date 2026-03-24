import os
from thz.dataset import DataSet, Constants
import numpy as np

fileDir = r"C:\Users\Samuel\Data\dispersion tests"
# fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-23_MINTS"

# files = [f for f in os.listdir(fileDir) if f.endswith('.csv') and f.startswith('tek')]

import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

try:
    from scipy.optimize import curve_fit
except ImportError as e:
    raise ImportError("This function requires scipy: pip install scipy") from e


PeakModel = Literal["gaussian", "lorentzian", "pvoigt"]


def _gaussian(x, amp, x0, sigma, y0):
    return y0 + amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _lorentzian(x, amp, x0, gamma, y0):
    # gamma is HWHM (half-width at half-maximum) for Lorentzian
    return y0 + amp * (gamma**2 / ((x - x0) ** 2 + gamma**2))


def _pvoigt(x, amp, x0, sigma, gamma, eta, y0):
    # pseudo-Voigt = eta*Lorentz + (1-eta)*Gauss, same amplitude scale at the center
    g = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    l = (gamma**2 / ((x - x0) ** 2 + gamma**2))
    return y0 + amp * (eta * l + (1.0 - eta) * g)


def _fwhm_from_params(model: PeakModel, params: Dict[str, float]) -> float:
    if model == "gaussian":
        sigma = params["sigma"]
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    if model == "lorentzian":
        gamma = params["gamma"]
        return 2.0 * gamma
    if model == "pvoigt":
        # A widely used approximation for Voigt FWHM (Olivero–Longbothum-like form).
        # Uses Gaussian FWHM and Lorentzian FWHM:
        fG = 2.0 * np.sqrt(2.0 * np.log(2.0)) * params["sigma"]
        fL = 2.0 * params["gamma"]
        return 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
    raise ValueError(f"Unknown model: {model}")


def fit_peak_1d(
    x: np.ndarray,
    y: np.ndarray,
    model: PeakModel = "pvoigt",
    x0_guess: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    baseline: Literal["const", "none"] = "const",
    maxfev: int = 50_000,
) -> Dict[str, object]:
    """
    Fit a single peak in 1D data using Gaussian, Lorentzian, or pseudo-Voigt.

    Parameters
    ----------
    x, y : arrays
        1D data (same length). x must be increasing (or at least not pathological).
    model : "gaussian" | "lorentzian" | "pvoigt"
    x0_guess : float, optional
        Initial center guess. If None, uses argmax of y (or of y-baseline estimate).
    window : (xmin, xmax), optional
        If provided, fits only data inside the window.
    baseline : "const" | "none"
        "const" fits y0; "none" sets y0=0 and does not fit it.
    maxfev : int
        Max evaluations for curve_fit.

    Returns
    -------
    dict with keys:
        - model
        - popt (dict of best-fit params)
        - perr (dict of 1-sigma uncertainties)
        - fwhm
        - fwhm_units (same units as x)
        - cov (covariance matrix)
        - y_fit (fit curve on the fitted x)
        - x_fit (x used for fitting)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same shape.")

    # Optional windowing
    if window is not None:
        xmin, xmax = window
        m = (x >= xmin) & (x <= xmax)
        if m.sum() < 6:
            raise ValueError("Too few points in window to fit.")
        x_fit = x[m]
        y_fit = y[m]
    else:
        x_fit, y_fit = x, y

    # Baseline handling
    if baseline == "none":
        y0_init = 0.0
    elif baseline == "const":
        # robust-ish baseline guess: lower percentile
        y0_init = float(np.percentile(y_fit, 10))
    else:
        raise ValueError("baseline must be 'const' or 'none'.")

    # Center guess
    if x0_guess is None:
        # work on baseline-subtracted for center selection
        y_bs = y_fit - y0_init
        x0_init = float(x_fit[np.argmax(y_bs)])
    else:
        x0_init = float(x0_guess)

    # Amplitude guess (positive peak)
    amp_init = float(np.max(y_fit) - y0_init)
    if amp_init <= 0:
        # fallback if the peak is negative or flat: allow negative peaks too
        amp_init = float(np.min(y_fit) - y0_init)

    # Width guess: a fraction of fit span (safe default)
    span = float(x_fit[-1] - x_fit[0]) if len(x_fit) > 1 else 1.0
    dx_med = float(np.median(np.diff(np.sort(x_fit)))) if len(x_fit) > 2 else span
    width_init = max(span / 20.0, 2.0 * dx_med)

    # Parameter bounds to keep widths positive and eta in [0,1]
    # Allow amp to be +/-; x0 within window; widths within sensible range.
    x_min, x_max = float(np.min(x_fit)), float(np.max(x_fit))
    w_min = max(1e-12, 0.25 * dx_med)
    w_max = max(w_min * 10.0, span * 2.0)

    # Build fit function, p0, bounds, and names
    if model == "gaussian":
        if baseline == "none":
            def f(x, amp, x0, sigma): return _gaussian(x, amp, x0, sigma, 0.0)
            p0 = [amp_init, x0_init, width_init]
            bounds = ([-np.inf, x_min, w_min], [np.inf, x_max, w_max])
            names = ["amp", "x0", "sigma"]
        else:
            f = _gaussian
            p0 = [amp_init, x0_init, width_init, y0_init]
            bounds = ([-np.inf, x_min, w_min, -np.inf], [np.inf, x_max, w_max, np.inf])
            names = ["amp", "x0", "sigma", "y0"]

    elif model == "lorentzian":
        if baseline == "none":
            def f(x, amp, x0, gamma): return _lorentzian(x, amp, x0, gamma, 0.0)
            p0 = [amp_init, x0_init, width_init]
            bounds = ([-np.inf, x_min, w_min], [np.inf, x_max, w_max])
            names = ["amp", "x0", "gamma"]
        else:
            f = _lorentzian
            p0 = [amp_init, x0_init, width_init, y0_init]
            bounds = ([-np.inf, x_min, w_min, -np.inf], [np.inf, x_max, w_max, np.inf])
            names = ["amp", "x0", "gamma", "y0"]

    elif model == "pvoigt":
        eta_init = 0.5
        if baseline == "none":
            def f(x, amp, x0, sigma, gamma, eta): return _pvoigt(x, amp, x0, sigma, gamma, eta, 0.0)
            p0 = [amp_init, x0_init, width_init, width_init, eta_init]
            bounds = (
                [-np.inf, x_min, w_min, w_min, 0.0],
                [ np.inf, x_max, w_max, w_max, 1.0],
            )
            names = ["amp", "x0", "sigma", "gamma", "eta"]
        else:
            f = _pvoigt
            p0 = [amp_init, x0_init, width_init, width_init, eta_init, y0_init]
            bounds = (
                [-np.inf, x_min, w_min, w_min, 0.0, -np.inf],
                [ np.inf, x_max, w_max, w_max, 1.0,  np.inf],
            )
            names = ["amp", "x0", "sigma", "gamma", "eta", "y0"]
    else:
        raise ValueError("model must be 'gaussian', 'lorentzian', or 'pvoigt'.")

    # Fit
    popt_arr, pcov = curve_fit(
        f, x_fit, y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev,
    )
    perr_arr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full_like(popt_arr, np.nan)

    popt = {k: float(v) for k, v in zip(names, popt_arr)}
    perr = {k: float(v) for k, v in zip(names, perr_arr)}

    # FWHM
    fwhm = float(_fwhm_from_params(model, popt))

    # Fit curve
    y_model = f(x_fit, *popt_arr)

    return {
        "model": model,
        "popt": popt,
        "perr": perr,
        "fwhm": fwhm,
        "fwhm_units": "x-units",
        "cov": pcov,
        "x_fit": x_fit,
        "y_fit": y_model,
    }

data_set = DataSet(fileDir)
data_set.load_all_data()
data_set.plot_current()

