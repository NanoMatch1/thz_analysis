"""Getting the one external number without ever touching the THz beam.

The emitter-polarisation series returns d_p*r_p and d_s*r_s, never r_p and r_s separately, so
one instrument constant C = d_p/d_s has to come from somewhere. The question Samuel asked is
whether it can come from somewhere that does not involve changing the THz optical path or
swapping a reference object into the focus.

It can, and the mechanism is that the DETECTION vector depends on the PROBE polarisation:

    d(phi) = (sin 2phi, 2 cos 2phi)        phi = probe angle from the crystal [001] axis

A half-wave plate in the 800 nm probe arm therefore changes d by a KNOWN amount while every
THz optic, the crystal and the sample stay exactly where they are. Measure the same sample at
two probe azimuths and take the ratio:

    m_1 / m_2 = (C_1 rho) / (C_2 rho) = C_1 / C_2

**The sample cancels.** Whatever is already mounted calibrates the instrument, so there is no
reference object, no remount, and no geometry change at all. The only unknown left is the
crystal's orientation chi relative to the plane of incidence, and the measured ratio determines
it from one real number.

Two practical notes:

- The analyser after the crystal (quarter-wave plate plus Wollaston) does NOT have to co-rotate.
  A mismatch rescales the whole of one measurement, and a scalar common to both polarisations
  cancels inside m_i before the two are ever compared. It costs signal, not accuracy. Keep the
  probe rotation near 30 degrees: at 45 the balanced detection nulls.
- Avoid landing either azimuth on a degenerate orientation (0 or 45 degrees from [001]), where
  one channel is blind.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from thz_ellipsometry.core import materials, model  # noqa: E402

FREQUENCIES = np.linspace(0.8e12, 3.0e12, 60)
INCIDENCE_ANGLE = np.deg2rad(45.0)
BALANCED_PROBE_AZIMUTH = 0.5 * np.arctan(2.0)


def detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad):
    """The <110> crystal's detection vector, rotated into the sample's p/s basis."""
    along = np.sin(2.0 * probe_azimuth_rad)
    across = 2.0 * np.cos(2.0 * probe_azimuth_rad)
    cosine, sine = np.cos(crystal_orientation_rad), np.sin(crystal_orientation_rad)
    return np.array([along * cosine - across * sine, along * sine + across * cosine])


def channel_ratio(probe_azimuth_rad, crystal_orientation_rad):
    vector = detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad)
    return vector[0] / vector[1]


def crystal_orientation_from_probe_pair(measured_ratio_of_ratios, first_azimuth_rad,
                                        second_azimuth_rad):
    """Solve for the crystal orientation from m_1/m_2, in which the sample has cancelled."""
    def residual(parameters):
        predicted = (channel_ratio(first_azimuth_rad, parameters[0])
                     / channel_ratio(second_azimuth_rad, parameters[0]))
        return [predicted - np.real(measured_ratio_of_ratios),
                np.imag(measured_ratio_of_ratios)]

    result = least_squares(residual, [0.0], bounds=([-np.pi / 4], [np.pi / 4]), xtol=1e-14)
    return float(result.x[0])


def demonstrate(true_orientation_deg=17.0, probe_steps_deg=(15.0, 30.0, 45.0),
                relative_noise=0.005, trials=40, seed=11):
    generator = np.random.default_rng(seed)
    true_orientation = np.deg2rad(true_orientation_deg)
    index_sample = materials.doped_silicon_index(FREQUENCIES, 1.0)
    ratio = model.ellipsometric_ratio(index_sample, INCIDENCE_ANGLE)

    print(f"\n[probe rotation] crystal orientation {true_orientation_deg:.2f} deg, "
          f"{relative_noise:.1%} noise, sample cancels in the ratio")
    print(f"   {'HWP-induced step':>18} {'recovered orientation':>23} "
          f"{'recovered C':>14} {'true C':>14}")
    truth = channel_ratio(BALANCED_PROBE_AZIMUTH, true_orientation)
    for step_deg in probe_steps_deg:
        second = BALANCED_PROBE_AZIMUTH + np.deg2rad(step_deg)
        estimates = []
        for _ in range(trials):
            first_measurement = channel_ratio(BALANCED_PROBE_AZIMUTH, true_orientation) * ratio
            second_measurement = channel_ratio(second, true_orientation) * ratio
            scale = relative_noise * np.abs(first_measurement).mean()
            noisy = [value + scale * (generator.normal(size=value.size)
                                      + 1j * generator.normal(size=value.size))
                     for value in (first_measurement, second_measurement)]
            estimates.append(crystal_orientation_from_probe_pair(
                np.mean(noisy[0] / noisy[1]), BALANCED_PROBE_AZIMUTH, second))
        orientation = float(np.median(estimates))
        print(f"   {step_deg:>17.0f}d {np.rad2deg(orientation):>22.3f}d "
              f"{channel_ratio(BALANCED_PROBE_AZIMUTH, orientation):>14.5f} {truth:>14.5f}")
    print("   -> the THz path, the crystal and the sample are untouched throughout.")


if __name__ == "__main__":
    demonstrate()
