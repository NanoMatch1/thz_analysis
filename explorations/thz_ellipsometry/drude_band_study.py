"""Which frequency band actually constrains the Drude parameters?

Samuel's argument: dropping the window opens up higher frequencies, where the near-mirror
inversion is better conditioned, so we can fit the Drude response from the high band and
discard the badly conditioned low band.  This asks whether the high band really does carry
the information, using a Fisher (linearised covariance) analysis rather than Monte Carlo.
"""

from __future__ import annotations

import numpy as np

from ellipsometry_conditioning_analysis import (
    drude_permittivity, ellipsometric_ratio, passive_index_from_permittivity,
)

EPSILON_INFINITY = 4.0


def _observables(dc_conductivity, scattering_time, frequencies_hz, incidence_angle_rad):
    index = passive_index_from_permittivity(
        drude_permittivity(frequencies_hz, dc_conductivity, scattering_time, EPSILON_INFINITY))
    ratio = ellipsometric_ratio(1.0, index, incidence_angle_rad)
    return np.concatenate([ratio.real, ratio.imag])


def parameter_uncertainties(dc_conductivity, scattering_time, frequencies_hz,
                            incidence_angle_rad, relative_error):
    """1-sigma on (sigma_dc, tau) from a linearised fit to rho over the band."""
    base = _observables(dc_conductivity, scattering_time, frequencies_hz, incidence_angle_rad)
    noise = relative_error * np.abs(base).mean()
    jacobian = np.column_stack([
        (_observables(dc_conductivity * 1.001, scattering_time, frequencies_hz, incidence_angle_rad)
         - _observables(dc_conductivity * 0.999, scattering_time, frequencies_hz, incidence_angle_rad))
        / (0.002 * dc_conductivity),
        (_observables(dc_conductivity, scattering_time * 1.001, frequencies_hz, incidence_angle_rad)
         - _observables(dc_conductivity, scattering_time * 0.999, frequencies_hz, incidence_angle_rad))
        / (0.002 * scattering_time),
    ])
    covariance = np.linalg.inv(jacobian.T @ jacobian) * noise**2
    return np.sqrt(np.diag(covariance)), covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])


def print_band_study(dc_conductivity=6500.0, scattering_times_fs=(30.0, 100.0),
                     incidence_angle_deg=72.0, relative_error=0.005, points=25):
    angle = np.deg2rad(incidence_angle_deg)
    bands = {
        "0.3-1.5 THz (with window)": (0.3e12, 1.5e12),
        "0.6-3.0 THz (no window)": (0.6e12, 3.0e12),
        "1.0-3.0 THz (high only)": (1.0e12, 3.0e12),
        "0.3-3.0 THz (everything)": (0.3e12, 3.0e12),
    }
    for scattering_time_fs in scattering_times_fs:
        knee_thz = 1.0 / (2 * np.pi * scattering_time_fs * 1e-15) / 1e12
        print(f"\n[drude band] tau = {scattering_time_fs:.0f} fs "
              f"(Drude knee at {knee_thz:.1f} THz), sigma_dc = {dc_conductivity/100:.0f} S/cm, "
              f"{incidence_angle_deg:.0f} deg, {relative_error:.1%} error on rho")
        print(f"   {'band':>28} {'d(sigma)/sigma':>15} {'d(tau)/tau':>12} {'correlation':>13}")
        for label, (low, high) in bands.items():
            frequencies = np.linspace(low, high, points)
            (sigma_error, tau_error), correlation = parameter_uncertainties(
                dc_conductivity, scattering_time_fs * 1e-15, frequencies, angle, relative_error)
            print(f"   {label:>28} {sigma_error/dc_conductivity:>14.2%} "
                  f"{tau_error/(scattering_time_fs*1e-15):>11.2%} {correlation:>13.3f}")


def print_information_density(dc_conductivity=6500.0, scattering_time_fs=30.0,
                              incidence_angle_deg=72.0, relative_error=0.005):
    """Per-octave contribution: where does the tau information actually live?"""
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[information density] single-octave bands, tau = {scattering_time_fs:.0f} fs")
    print(f"   {'octave':>18} {'d(sigma)/sigma':>15} {'d(tau)/tau':>12}")
    for low in (0.25e12, 0.5e12, 1.0e12, 2.0e12):
        frequencies = np.linspace(low, 2 * low, 15)
        (sigma_error, tau_error), _ = parameter_uncertainties(
            dc_conductivity, scattering_time_fs * 1e-15, frequencies, angle, relative_error)
        print(f"   {f'{low/1e12:.2f}-{2*low/1e12:.2f} THz':>18} "
              f"{sigma_error/dc_conductivity:>14.2%} "
              f"{tau_error/(scattering_time_fs*1e-15):>11.2%}")


if __name__ == "__main__":
    print_band_study()
    print_information_density()
