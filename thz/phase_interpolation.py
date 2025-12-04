# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:06:03 2023

https://doi.org/10.1007/s10762-019-00578-0

@author: Marco Ballabio
"""
#to do: replace gradient with fit on a small selected region

import numpy as np
from scipy.fft import rfftfreq
from scipy.stats import linregress
from thz.data_structures.decorators import align_to_max_resolution

#extrapolate phase difference to zero. (to avoid refractive index divergence)

# TO DO: consider to extrapolate each dataset and not just the difference,
# maybe helps when the two maxima are way different
@align_to_max_resolution(
    axis_col_name="Frequency (THz)",
    phase_col_names=('Phase', 'Δ(Phase)')
)
def phaseex(ref,sam,show_graph=False):

    breakpoint()
    dff = sam['Phase']-ref['Phase']
    

    #defines range of interest
    low = 0.5
    up = 1.8
    x = ref['Frequency (THz)'].loc[ref['Frequency (THz)'].between(low,up)]
    y = dff.loc[ref['Frequency (THz)'].between(low,up)]
    #least squares linear regression of the phase in the range of interest
    breakpoint()
    lsq = linregress(x,y)
    ycross = lsq.intercept
    # extrapolates the phase from max amplitude to 0 THz using the gradient
    # breakpoint()
    delta_t_ps = -lsq.slope / (2 * np.pi)
    print(f"Delay from phase (phaseex): {delta_t_ps:.3f} ps")

    import matplotlib.pyplot as plt
    plt.plot(sam['Frequency (THz)'], sam['Phase'], label='sample phase')
    plt.plot(ref['Frequency (THz)'], ref['Phase'], label='ref phase')
    plt.legend()
    plt.title('Phaseex data')
    plt.show()
    plt.plot(x, y, 'o', label='data')
    plt.plot(x, lsq.intercept + lsq.slope*x, 'r', label='fit')
    plt.legend()
    plt.show()
    dff -= ycross

    return dff, delta_t_ps


#to account for different time windows starts (use padded data!)
def phaseoffset(ref,sam):
    
    t0r = ref.iloc[0].at['Time (ps)']
    t0s = sam.iloc[0].at['Time (ps)']
    
    #define freq axis
    freq = rfftfreq(len(ref['Time (ps)']), ref.iloc[1].at['Time (ps)']-ref.iloc[0].at['Time (ps)'])
    
    phioffset = 2*np.pi*freq*(t0s-t0r)
    
    return phioffset



import numpy as np
import pandas as pd
from scipy.stats import linregress

# assuming you've already done something like:
# from thz.data_processing.alignment import align_to_max_resolution

@align_to_max_resolution(
    axis_col_name="Frequency (THz)",
    phase_col_names=("Phase",),
    wrap_phase_output=False,   # we want continuous phase inside this function
)
def phaseex_v2(
    ref: pd.DataFrame,
    sam: pd.DataFrame,
    freq_col: str = "Frequency (THz)",
    phase_col: str = "Phase",
    fit_range: tuple[float, float] = (0.5, 1.8),
    remove_slope: bool = False,
    wrap_output: bool = False,
    correction_factor: float = 3.0,
    **kwargs
) -> pd.Series:
    """
    Compute a cleaned phase difference φ_sam - φ_ref, with removal of the
    unphysical constant phase offset (and optionally slope) based on a
    linear fit in a user-specified frequency window.

    Assumptions
    -----------
    - `ref` and `sam` are already aligned to a common frequency axis by
      the align_to_max_resolution decorator.
    - `phase_col` contains the phase in radians.

    Parameters
    ----------
    ref, sam : pd.DataFrame
        FFT results with columns including `freq_col` and `phase_col`.
    freq_col : str, default "Frequency (THz)"
        Name of the frequency column.
    phase_col : str, default "Phase"
        Name of the phase column.
    fit_range : (float, float), default (0.3, 2.0)
        Frequency range [low, high] (in same units as freq_col) over which
        to fit the phase difference with a straight line.
        This should be a clean, high-SNR band.
    remove_slope : bool, default False
        If False (recommended for optical constants):
            only the constant offset (intercept) is removed.
        If True:
            both slope and intercept are removed, so the resulting phase
            is "flattened" in the fit range (useful for diagnostics,
            but discards physical delay information).
    wrap_output : bool, default False
        If True, wrap the corrected phase difference back into [-π, π].
        If False, keep it unwrapped/continuous.
    correction_factor : float, default 3.0
        Informed phase unwrapping inflates the phase difference by a factor of 3, so this factor can be used to correct the computed delay. Made a kwarg for flexibility and testing.

    Returns
    -------
    dphi_corr : pd.Series
        Corrected phase difference φ_sam - φ_ref (after offset removal),
        indexed like `ref[freq_col]`.
    """
    show_graph = kwargs.get("show_graph", False)
    # Extract frequency axis

    freq = ref[freq_col].to_numpy()

    # Extract phase (already unwrapped) from ref & sam
    # phi_ref = np.unwrap(ref[phase_col].to_numpy())
    # phi_sam = np.unwrap(sam[phase_col].to_numpy())
    phi_ref = ref[phase_col].to_numpy()
    phi_sam = sam[phase_col].to_numpy()

    # Raw phase difference (continuous)
    dphi = phi_sam - phi_ref

    if show_graph:
        import matplotlib.pyplot as plt
        plt.plot(ref[freq_col], phi_ref, label='ref phase')
        plt.plot(sam[freq_col], phi_sam, label='sam phase')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase (rad)')
        plt.plot(freq, dphi, label='Raw phase difference')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase difference (rad)')
        plt.legend()
        plt.title('Phaseex_v2 data')
        plt.show()

    # Select fit range
    low, up = fit_range
    mask = (freq >= low) & (freq <= up)

    if mask.sum() < 2:
        raise ValueError(
            f"Not enough points in fit_range {fit_range}. "
            f"Got {mask.sum()} points."
        )

    x_fit = freq[mask]
    y_fit = dphi[mask]

    # Linear regression: dphi ≈ m * f + b in the clean band
    lsq = linregress(x_fit, y_fit)
    m, b = lsq.slope, lsq.intercept

    delta_t_ps = -m / (2 * np.pi)
    print(f"Delay from phase (fit_range={fit_range}): {delta_t_ps:.3f} ps")



    if show_graph:
        plt.plot(x_fit, y_fit, 'o', label='Data in fit range')
        plt.plot(x_fit, m * x_fit + b, 'r', label='Linear fit')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase difference (rad)')
        plt.legend()
        plt.show()

    if remove_slope:
        # Remove both slope and intercept → zero phase in the fit band
        dphi_corr = dphi - (m * freq + b)
    else:
        # Remove only intercept → preserve physical time delay (slope)
        dphi_corr = dphi - b

    if wrap_output:
        dphi_corr = np.angle(np.exp(1j * dphi_corr))

    # Return as a Series aligned with ref's index
    dphi_series = pd.Series(dphi_corr, index=ref.index, name="Phase difference")

    return dphi_series, delta_t_ps/correction_factor
