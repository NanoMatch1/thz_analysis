"""Does the per-mount sample tilt have to be controlled, or can it be FITTED?

Samuel's objection, and it is the right one to raise: if the method needs the sample and the
gold reference to share an out-of-plane tilt to arcminutes, then it inherits exactly the
mounting-reproducibility problem that defeated the earlier reflection work -- magnetic mount,
3D-printed holders, each sample seating slightly differently, and no way to align visibly.

The resolution turns on a degeneracy. Write the measured channel ratio for a sample tilted out
of plane by psi, with instrument channel ratio C = d_p/d_s:

    m = (C*r_pp + r_ps) / (C*r_ps + r_ss),    r_pp = r_p cos^2 + r_s sin^2, etc.

To first order in psi this is

    m ~ (C*rho + (rho - 1)*psi) / (1 + C*(rho - 1)*psi)

The tilt enters multiplied by (rho - 1), which is MATERIAL- and FREQUENCY-dependent, while C is
frequency-flat. So:

  DISPERSIVE sample (doped Si, CNT)   psi and C separate -> the tilt is fittable from the
                                      sample's own data. Mounting reproducibility NOT needed.
  FLAT sample (gold, HR-Si)           (rho - 1)*psi is frequency-flat too -> psi is degenerate
                                      with C and CANNOT be fitted out.

Which inverts the obvious intuition: the simple, dispersionless wafer is the FRAGILE validation
sample, and the doped one is the robust one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ellipsometry import materials, model  # noqa: E402

FREQUENCIES = np.linspace(0.8e12, 3.0e12, 60)
INCIDENCE_ANGLE = np.deg2rad(45.0)
#: A realistic instrument constant: complex (strain birefringence is allowed) and not unity.
INSTRUMENT_CHANNEL_RATIO = 0.93 + 0.11j
RELATIVE_NOISE = 0.005


def tilted_channel_ratio(index_sample, out_of_plane_tilt_rad, channel_ratio,
                         incidence_angle_rad=INCIDENCE_ANGLE):
    """P/Q that the harmonic fit returns for a tilted sample."""
    reflection_p, reflection_s = model.reflection_coefficients(index_sample, incidence_angle_rad)
    cosine, sine = np.cos(out_of_plane_tilt_rad), np.sin(out_of_plane_tilt_rad)
    diagonal_p = reflection_p * cosine**2 + reflection_s * sine**2
    diagonal_s = reflection_p * sine**2 + reflection_s * cosine**2
    cross = (reflection_p - reflection_s) * sine * cosine
    return ((channel_ratio * diagonal_p + cross)
            / (channel_ratio * cross + diagonal_s))


def _noisy(values, generator):
    scale = RELATIVE_NOISE * np.abs(values).mean()
    return values + scale * (generator.normal(size=values.size)
                             + 1j * generator.normal(size=values.size))


def calibrate_channel_ratio(gold_tilt_rad, generator):
    """C from a gold mirror whose own tilt is unknown and uncorrected."""
    gold = materials.gold_index(FREQUENCIES)
    measured = _noisy(tilted_channel_ratio(gold, gold_tilt_rad, INSTRUMENT_CHANNEL_RATIO),
                      generator)
    return np.mean(measured / model.ellipsometric_ratio(gold, INCIDENCE_ANGLE))


def fit_dispersive_sample(measured, channel_ratio, fit_tilt=True):
    """Fit (resistivity[, tilt]) to a doped-silicon measurement, from its own data only."""
    def residual(parameters):
        index = materials.doped_silicon_index(FREQUENCIES, 10.0 ** parameters[0])
        tilt = np.deg2rad(parameters[1]) if fit_tilt else 0.0
        difference = tilted_channel_ratio(index, tilt, channel_ratio) - measured
        return np.concatenate([difference.real, difference.imag])

    result = least_squares(residual, [0.0, 0.0], bounds=([-2.0, -5.0], [2.0, 5.0]),
                          xtol=1e-13, ftol=1e-13)
    return 10.0 ** result.x[0], result.x[1]


def fit_flat_sample(measured, channel_ratio, fit_tilt=True):
    """Fit (n[, tilt]) to a dispersionless wafer -- where the degeneracy bites."""
    def residual(parameters):
        index = np.full(FREQUENCIES.shape, parameters[0], dtype=complex)
        tilt = np.deg2rad(parameters[1]) if fit_tilt else 0.0
        difference = tilted_channel_ratio(index, tilt, channel_ratio) - measured
        return np.concatenate([difference.real, difference.imag])

    result = least_squares(residual, [3.4, 0.0], bounds=([2.5, -5.0], [4.5, 5.0]),
                           xtol=1e-13, ftol=1e-13)
    return result.x[0], result.x[1]


def report_dispersive_versus_flat(trials=20, seed=3):
    generator = np.random.default_rng(seed)
    print(f"\n[dispersive vs flat] tilt fitted from the sample alone, C known exactly, "
          f"{RELATIVE_NOISE:.1%} noise")
    cases = (
        ("doped Si, rho_dc [ohm.cm], target 1.000",
         lambda: materials.doped_silicon_index(FREQUENCIES, 1.0), fit_dispersive_sample),
        ("HR-Si, n, target 3.4175",
         lambda: np.full(FREQUENCIES.shape, 3.4175, dtype=complex), fit_flat_sample),
    )
    for label, build, fitter in cases:
        print(f"   {label}")
        print(f"      {'tilt [deg]':>11} {'tilt NOT fitted':>20} {'tilt fitted':>18} "
              f"{'tilt recovered':>16}")
        for tilt_deg in (0.0, 0.1, 0.25, 0.5, 1.0):
            fixed, free, tilts = [], [], []
            for _ in range(trials):
                measured = _noisy(
                    tilted_channel_ratio(build(), np.deg2rad(tilt_deg),
                                         INSTRUMENT_CHANNEL_RATIO), generator)
                fixed.append(fitter(measured, INSTRUMENT_CHANNEL_RATIO, fit_tilt=False)[0])
                value, tilt = fitter(measured, INSTRUMENT_CHANNEL_RATIO, fit_tilt=True)
                free.append(value)
                tilts.append(tilt)
            print(f"      {tilt_deg:>11.2f} {np.median(fixed):>20.4f} "
                  f"{np.median(free):>18.4f} {np.median(tilts):>16.3f}")
        print()


def report_independent_mounts(trials=12, seed=5):
    """The decisive test: gold and sample tilted DIFFERENTLY, no shared mount."""
    generator = np.random.default_rng(seed)
    print("[independent mounts] gold tilted one way, sample another, tilt fitted from the "
          "sample alone")
    print("   true rho_dc = 1.000 ohm.cm")
    print(f"   {'gold tilt':>10} {'sample tilt':>12} {'recovered rho_dc':>22} "
          f"{'fitted tilt':>14}")
    for gold_tilt_deg in (0.0, 0.5, 1.0):
        for sample_tilt_deg in (0.0, 0.5, 1.0, -1.0):
            values, tilts = [], []
            for _ in range(trials):
                channel_ratio = calibrate_channel_ratio(np.deg2rad(gold_tilt_deg), generator)
                measured = _noisy(
                    tilted_channel_ratio(materials.doped_silicon_index(FREQUENCIES, 1.0),
                                         np.deg2rad(sample_tilt_deg),
                                         INSTRUMENT_CHANNEL_RATIO), generator)
                value, tilt = fit_dispersive_sample(measured, channel_ratio)
                values.append(value)
                tilts.append(tilt)
            print(f"   {gold_tilt_deg:>10.2f} {sample_tilt_deg:>12.2f} "
                  f"{np.median(values):>16.4f} +/-{np.std(values):<4.3f} "
                  f"{np.median(tilts):>12.3f}")
    print("   -> the fit absorbs the gold's tilt into the sample's fitted tilt, so the")
    print("      DIFFERENCE is what gets estimated. Mounting reproducibility is not required;")
    print("      only keeping the gold roughly aligned, because the residual is second order.")


def main():
    report_dispersive_versus_flat()
    report_independent_mounts()


if __name__ == "__main__":
    main()
