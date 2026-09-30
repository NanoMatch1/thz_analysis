"""Is rotating the sample to 0/90 really better than a single-mount generalised measurement?

The idealised comparison in anisotropic_reflection.print_scheme_comparison says yes, by a wide
margin -- but it assumes the two mounts are otherwise identical.  They are not: re-pressing the
paper changes the contact (F27 measured per-mount gaps of 3.4 vs 5.3 um) and the surface tilt.
This adds an unmodelled per-mount nuisance and asks the question again.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from anisotropic_reflection import (
    CNT_ANISOTROPIC_MODEL, anisotropic_reflection_jones, normalised_jones_observables,
    rotate_tensor_about_normal,
)
from ellipsometry_conditioning_analysis import drude_permittivity

PARAMETER_SCALE = np.array([1e4, 1e4, 1e4, 1e-14, 1.0])


def _mount_observables(parameters, frequencies_hz, incidence_angle_rad, mount_azimuth,
                       use_cross, angle_error_rad=0.0, gain_error=0.0):
    conductivity_1, conductivity_2, conductivity_3, scattering_time, _ = parameters
    values = []
    for frequency in frequencies_hz:
        principal = tuple(
            drude_permittivity(frequency, conductivity, scattering_time,
                               CNT_ANISOTROPIC_MODEL["epsilon_infinity"])
            for conductivity in (conductivity_1, conductivity_2, conductivity_3))
        tensor = rotate_tensor_about_normal(principal, mount_azimuth)
        jones = anisotropic_reflection_jones(tensor, 1.0,
                                             incidence_angle_rad + angle_error_rad)
        ratios = normalised_jones_observables(jones) * (1.0 + gain_error)
        values.extend([ratios[0].real, ratios[0].imag])
        if use_cross:
            values.extend([ratios[1].real, ratios[1].imag])
    return np.array(values)


def _simulate(parameters, frequencies_hz, incidence_angle_rad, scheme, mount_errors):
    if scheme == "generalized_one_mount":
        angle_error, gain_error = mount_errors[0]
        return _mount_observables(parameters, frequencies_hz, incidence_angle_rad,
                                  parameters[4], True, angle_error, gain_error)
    blocks = []
    for mount_index, mount_azimuth in enumerate((0.0, np.pi / 2)):
        angle_error, gain_error = mount_errors[mount_index]
        blocks.append(_mount_observables(parameters, frequencies_hz, incidence_angle_rad,
                                         mount_azimuth, False, angle_error, gain_error))
    return np.concatenate(blocks)


def run(frequencies_hz, incidence_angle_deg=70.0, azimuth_deg=45.0, relative_noise=0.005,
        mount_tilt_deg=0.5, mount_gain=0.02, trials=8, seed=31):
    angle = np.deg2rad(incidence_angle_deg)
    truth = np.array([CNT_ANISOTROPIC_MODEL["conductivity_along_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_across_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_out_of_plane_si"],
                      CNT_ANISOTROPIC_MODEL["scattering_time_s"],
                      np.deg2rad(azimuth_deg)])
    print(f"[mount penalty] per-mount nuisance: {mount_tilt_deg} deg tilt, "
          f"{mount_gain:.0%} gain, UNMODELLED; {relative_noise:.1%} noise, {trials} trials")
    print(f"   {'scheme':>22} {'mounts':>8} {'sigma_along':>18} {'sigma_across':>18} "
          f"{'anisotropy ratio':>20}")
    for scheme in ("generalized_one_mount", "standard_two_mounts"):
        generator = np.random.default_rng(seed)
        recovered = []
        for _ in range(trials):
            mount_errors = [(np.deg2rad(generator.normal(0.0, mount_tilt_deg)),
                             generator.normal(0.0, mount_gain)) for _ in range(2)]
            clean = _simulate(truth, frequencies_hz, angle, scheme, mount_errors)
            noisy = clean + relative_noise * np.abs(clean).mean() * generator.normal(size=clean.size)
            start = truth / PARAMETER_SCALE * generator.uniform(0.7, 1.4, size=5)
            result = least_squares(
                lambda scaled: _simulate(scaled * PARAMETER_SCALE, frequencies_hz, angle,
                                         scheme, [(0.0, 0.0), (0.0, 0.0)]) - noisy,
                start, bounds=([1e-2, 1e-2, 1e-3, 0.1, 0.0], [1e2, 1e2, 1e2, 100.0, np.pi]),
                xtol=1e-10, ftol=1e-10)
            recovered.append(result.x * PARAMETER_SCALE)
        recovered = np.array(recovered)

        def summarise(values):
            return np.median(values), 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))

        along = summarise(recovered[:, 0] / 100)
        across = summarise(recovered[:, 1] / 100)
        ratio = summarise(recovered[:, 0] / recovered[:, 1])
        mounts = 1 if scheme == "generalized_one_mount" else 2
        print(f"   {scheme:>22} {mounts:>8} {along[0]:>9.1f} +/-{along[1]:<7.1f} "
              f"{across[0]:>9.1f} +/-{across[1]:<7.1f} {ratio[0]:>11.2f} +/-{ratio[1]:<7.2f}")
    print("   true: sigma_along 65.0 S/cm, sigma_across 16.2 S/cm, ratio 4.00")


if __name__ == "__main__":
    run(np.linspace(0.8e12, 3.0e12, 9))
