# -*- coding: utf-8 -*-
"""
FFT with error propagation and careful phase unwrap (after Jepsen 2019)
Refactored to work with THzData.data (N x 3 numpy array).

Assumes columns:
  0: Time (ps)
  1: Mean signal
  2: Std error of signal

Returns a pandas DataFrame with:
  'Frequency (THz)', 'Amplitude', 'Δ(Amplitude)', 'Phase', 'Δ(Phase)'
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq   # rfft returns only positive frequencies
from math import e


def fft_with_error_from_array(
    time_ps: np.ndarray,
    y_mean: np.ndarray,
    y_err: np.ndarray,
    remove_dc: bool = False,
    dc_points: int = 10,
) -> pd.DataFrame:
    """
    Compute FFT with error propagation and informed phase unwrapping
    from Jepsen 2019.

    Parameters
    ----------
    time_ps : np.ndarray
        Time axis in picoseconds (uniformly spaced).
    y_mean : np.ndarray
        Mean time-domain trace.
    y_err : np.ndarray
        Standard error of the mean at each time point.
    remove_dc : bool
        If True, subtracts DC offset estimated from first `dc_points` samples.
        If you already called subtract_dc_offset on THzData, leave this False.
    dc_points : int
        Number of initial points used to estimate DC offset.

    Returns
    -------
    dff : pd.DataFrame
        Frequency-domain data with columns:
        ['Frequency (THz)', 'Amplitude', 'Δ(Amplitude)', 'Phase', 'Δ(Phase)']
    """

    # ensure 1D copies
    time = np.asarray(time_ps, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_err = np.asarray(y_err, dtype=float)

    # define freq axis (rfftfreq units are 1 / (time units), so with ps input → THz)
    dt_ps = time[1] - time[0]
    freq_thz = rfftfreq(len(time), dt_ps)  # 1/ps ≡ THz

    # optional DC offset removal
    if remove_dc and len(y_mean) >= dc_points:
        y_mean = y_mean - np.mean(y_mean[:dc_points])

    # Fourier transforms (only positive freqs), normalised 'ortho'
    ft_y_mean = rfft(y_mean, norm="ortho")
    ft_y_err = rfft(y_err, norm="ortho")

    # Variance of FFT is FFT of variance (Jepsen)
    ft_variance = rfft(y_err**2, norm="ortho")
    sr = np.sqrt(np.abs(ft_variance.real))
    si = np.sqrt(np.abs(ft_variance.imag))

    # Amplitude in polar form
    amplitude = np.abs(ft_y_mean)

    # Phase unwrapping with reference to the main time-domain peak
    t0_ps = time[np.argmax(np.abs(y_mean))]    # position of main pulse (in ps)
    phase0 = 2 * np.pi * t0_ps * freq_thz      # linear phase term

    # Reduce phase before unwrapping
    ft_y_mean_reduced = ft_y_mean * e ** (-1j * phase0)
    phase = np.angle(ft_y_mean_reduced)

    # Unwrap and apply sign convention
    phase = -np.unwrap(phase)
    phase = phase + phase0

    # Error propagation from Cartesian (real, imag) to polar (amp, phase)
    # amplitude error
    with np.errstate(divide="ignore", invalid="ignore"):
        err_amplitude = np.sqrt(
            (sr * ft_y_mean.real) ** 2 + (si * ft_y_mean.imag) ** 2
        ) / np.where(amplitude == 0, 1.0, amplitude)

        # phase error
        denom = np.where(ft_y_mean.real == 0, 1e-30, ft_y_mean.real)
        err_phase = np.sqrt(
            (si / denom) ** 2
            + (ft_y_mean.imag * sr / denom**2) ** 2
        ) / (1.0 + (ft_y_mean.imag / denom) ** 2)

        # account for unwrapped phase magnitude
        err_phase = err_phase * np.abs(phase)

    dff = pd.DataFrame(
        np.column_stack((freq_thz, amplitude, err_amplitude, phase, err_phase)),
        columns=[
            "Frequency (THz)",
            "Amplitude",
            "Δ(Amplitude)",
            "Phase",
            "Δ(Phase)",
        ],
    )

    return dff


def transfer_function(ref: pd.DataFrame, sam: pd.DataFrame, offset) -> pd.DataFrame:
    amplitude = sam["Amplitude"] / ref["Amplitude"]
    # phase = sam["Phase"] - ref["Phase"] - offset
    phase = phaseex(ref, sam) - offset

    err_amplitude = (
        1.0 / (ref["Amplitude"] ** 2)
        * (sam["Δ(Amplitude)"] * ref["Amplitude"] + ref["Δ(Amplitude)"] * sam["Amplitude"])
    )
    err_phase = sam["Δ(Phase)"] + ref["Δ(Phase)"]

    T = pd.DataFrame(
        np.column_stack((ref["Frequency (THz)"], amplitude, err_amplitude, phase, err_phase)),
        columns=["Frequency (THz)", "Amplitude", "Δ(Amplitude)", "Phase", "Δ(Phase)"],
    )
    return T
