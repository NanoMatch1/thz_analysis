"""Why is k hard to measure in silicon -- physical, instrumental, or analytical?

Samuel's question, and the reason HR-Si and doped Si together are the right validation pair:
one has essentially no absorption and the other has plenty, so comparing them separates a
genuine instrument limitation from a material that simply has nothing to measure.

Three candidate explanations, and this module sizes each:

  PHYSICAL     k of high-resistivity silicon is ~1e-4 at 1 THz.  Nothing can measure that in
               reflection, and failing to is not a fault.
  INSTRUMENTAL near the pseudo-Brewster angle |r_p| collapses, so a fixed FIELD noise floor
               becomes a huge RELATIVE error on rho.  This is the real trade and it sets the
               optimum angle.
  ANALYTICAL   at angles far from Brewster the inversion is simply insensitive to k, whatever
               the signal quality.

The headline result: reflection ellipsometry near Brewster is the most k-sensitive geometry
available, but not AT Brewster -- the SNR collapse wins beyond a point, and the optimum sits a
few degrees off.
"""

from __future__ import annotations

import numpy as np

from ellipsometry_conditioning_analysis import (
    SPEED_OF_LIGHT,
    ellipsometric_ratio,
    fresnel_reflection_p,
    fresnel_reflection_s,
    index_from_ellipsometric_ratio,
)

# Float-zone HR silicon: n = 3.4175, power absorption below ~0.05 /cm at 1 THz.
HIGH_RESISTIVITY_SILICON_INDEX = 3.4175
SILICON_STATIC_PERMITTIVITY = 11.68
VACUUM_PERMITTIVITY = 8.8541878128e-12


def extinction_from_power_absorption(absorption_per_cm, frequency_hz):
    """k from the power absorption coefficient: alpha = 4*pi*k/lambda."""
    wavelength_cm = SPEED_OF_LIGHT / frequency_hz * 100.0
    return absorption_per_cm * wavelength_cm / (4.0 * np.pi)


def doped_silicon_index(frequency_hz, resistivity_ohm_cm, mobility_cm2_per_vs=1400.0):
    """Drude index of doped silicon from its DC resistivity, in the N = n - i*k convention."""
    angular_frequency = 2.0 * np.pi * frequency_hz
    conductivity_si = 100.0 / resistivity_ohm_cm           # S/m from ohm.cm
    carrier_density = conductivity_si / (1.602176634e-19 * mobility_cm2_per_vs * 1e-4)
    effective_mass = 0.26 * 9.1093837015e-31
    scattering_time = (mobility_cm2_per_vs * 1e-4) * effective_mass / 1.602176634e-19
    plasma_squared = (carrier_density * 1.602176634e-19**2
                      / (VACUUM_PERMITTIVITY * effective_mass))
    permittivity = SILICON_STATIC_PERMITTIVITY - plasma_squared / (
        angular_frequency**2 - 1j * angular_frequency / scattering_time)
    index = np.sqrt(permittivity)
    return np.where(index.imag > 0.0, -index, index)


# ---------------------------------------------------------------------------
# Sensitivity of k to a measurement error, with the Brewster SNR collapse included
# ---------------------------------------------------------------------------

def perturbed_ratios(index_sample, incidence_angle_rad, field_noise_fraction,
                     index_incident=1.0):
    """rho values produced by a fixed ADDITIVE field noise on each polarisation channel.

    The measured quantities are the two fields, not their ratio:

        rho_measured = (r_p + n_p) / (r_s + n_s),      |n_p| = |n_s| = nu

    Applying the noise multiplicatively to rho is wrong near the Brewster angle, where
    r_p -> 0: a multiplicative error on a vanishing quantity vanishes with it, which would
    make Brewster look *more* accurate rather than less.  `field_noise_fraction` is nu
    relative to |r_s|, i.e. relative to the strong channel.
    """
    r_p = fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad)
    r_s = fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad)
    noise = field_noise_fraction * abs(r_s)
    for perturbation in (noise, -noise, 1j * noise, -1j * noise):
        yield (r_p + perturbation) / r_s
        yield r_p / (r_s + perturbation)


def relative_ratio_error_from_field_noise(index_sample, incidence_angle_rad,
                                          field_noise_fraction, index_incident=1.0):
    """The resulting RELATIVE error on rho, reported for context only."""
    ratio = ellipsometric_ratio(index_incident, index_sample, incidence_angle_rad)
    return max(abs(perturbed - ratio) for perturbed in
               perturbed_ratios(index_sample, incidence_angle_rad, field_noise_fraction,
                                index_incident)) / abs(ratio)


def _index_from_ratio_nearest(ratio, incidence_angle_rad, reference_index,
                              index_incident=1.0):
    """Invert rho, taking the square-root branch NEAREST the reference index.

    The usual passivity rule (force Im(N) <= 0) is wrong for a sensitivity study on a
    nearly lossless sample: a perturbation that pushes k slightly negative then flips the
    whole index to -N and reports a spurious error of 2n.  A real fit follows the branch by
    continuity, which is what this does.
    """
    sine_squared = np.sin(incidence_angle_rad) ** 2
    tangent_squared = np.tan(incidence_angle_rad) ** 2
    permittivity = index_incident**2 * sine_squared * (
        1.0 + tangent_squared * ((1.0 - ratio) / (1.0 + ratio)) ** 2)
    root = np.sqrt(permittivity)
    return root if abs(root - reference_index) <= abs(-root - reference_index) else -root


def _perturbed_indices(index_sample, incidence_angle_rad, field_noise_fraction,
                       index_incident=1.0):
    for perturbed in perturbed_ratios(index_sample, incidence_angle_rad,
                                      field_noise_fraction, index_incident):
        yield _index_from_ratio_nearest(perturbed, incidence_angle_rad, index_sample,
                                        index_incident)


def extinction_error(index_sample, incidence_angle_rad, field_noise_fraction,
                     index_incident=1.0):
    """Error in k from a fixed additive field noise floor."""
    return max(abs(-recovered.imag - (-index_sample.imag))
               for recovered in _perturbed_indices(index_sample, incidence_angle_rad,
                                                   field_noise_fraction, index_incident))


def refractive_index_error(index_sample, incidence_angle_rad, field_noise_fraction,
                           index_incident=1.0):
    return max(abs(recovered.real - index_sample.real)
               for recovered in _perturbed_indices(index_sample, incidence_angle_rad,
                                                   field_noise_fraction, index_incident))


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def print_what_k_actually_is(frequencies_hz=(0.5e12, 1.0e12, 2.0e12, 3.0e12)):
    print("\n[what k actually is] HR silicon vs doped silicon")
    print(f"   {'f [THz]':>8} {'k(HR, a=0.05/cm)':>18} {'k(HR, a=0.3/cm)':>17} "
          f"{'k(10 ohm.cm)':>14} {'k(1 ohm.cm)':>13} {'k(0.1 ohm.cm)':>15}")
    for frequency in frequencies_hz:
        row = (f"   {frequency/1e12:>8.2f} "
               f"{extinction_from_power_absorption(0.05, frequency):>18.5f} "
               f"{extinction_from_power_absorption(0.3, frequency):>17.5f}")
        for resistivity in (10.0, 1.0, 0.1):
            row += f"{-doped_silicon_index(frequency, resistivity).imag:>14.4f}"
        print(row)
    print("   -> HR silicon's k really is ~1e-4 to 1e-3. No reflection measurement sees that,")
    print("      and failing to is PHYSICAL, not a fault of the instrument or the analysis.")
    print("      Doped silicon is three to four orders of magnitude larger: that is the test.")


def print_k_sensitivity_vs_angle(frequency_hz=1.0e12,
                                 angles_deg=(45.0, 60.0, 65.0, 70.0, 72.0, 73.7, 75.0, 78.0, 80.0),
                                 field_noise_fraction=0.005,
                                 resistivities_ohm_cm=(1.0, 10.0)):
    print(f"\n[k sensitivity vs angle] {frequency_hz/1e12:.1f} THz, field noise "
          f"{field_noise_fraction:.1%} of |r_s|")
    print("   Brewster angle for lossless Si (n=3.4175) is 73.69 deg.")
    samples = {"HR-Si (a=0.05/cm)": HIGH_RESISTIVITY_SILICON_INDEX
               - 1j * extinction_from_power_absorption(0.05, frequency_hz)}
    for resistivity in resistivities_ohm_cm:
        samples[f"doped {resistivity:g} ohm.cm"] = doped_silicon_index(frequency_hz, resistivity)

    for label, index in samples.items():
        print(f"\n   {label}:  n = {index.real:.4f}, k = {-index.imag:.5f}")
        print(f"   {'theta':>7} {'|rho|':>9} {'rel err rho':>12} {'dk':>10} "
              f"{'dk/k':>10} {'dn':>10}")
        best = None
        for angle_deg in angles_deg:
            angle = np.deg2rad(angle_deg)
            ratio = ellipsometric_ratio(1.0, index, angle)
            relative = relative_ratio_error_from_field_noise(index, angle,
                                                             field_noise_fraction)
            delta_k = extinction_error(index, angle, field_noise_fraction)
            delta_n = refractive_index_error(index, angle, field_noise_fraction)
            fractional = delta_k / max(-index.imag, 1e-12)
            if best is None or delta_k < best[1]:
                best = (angle_deg, delta_k)
            print(f"   {angle_deg:>7.1f} {abs(ratio):>9.4f} {relative:>12.4f} "
                  f"{delta_k:>10.5f} {fractional:>10.1f} {delta_n:>10.5f}")
        print(f"   -> best angle for k: {best[0]:.1f} deg (dk = {best[1]:.5f})")


def print_optimum_angle_scan(frequency_hz=1.0e12, field_noise_fraction=0.005,
                             resistivities_ohm_cm=(0.1, 1.0, 10.0)):
    angles = np.deg2rad(np.arange(40.0, 88.0, 0.25))
    print(f"\n[optimum angle for k] fine scan, {frequency_hz/1e12:.1f} THz, "
          f"{field_noise_fraction:.1%} field noise")
    print(f"   {'sample':>22} {'k true':>10} {'theta_opt':>11} {'dk':>10} {'dk/k':>10}")
    entries = [("HR-Si (a=0.05/cm)", HIGH_RESISTIVITY_SILICON_INDEX
                - 1j * extinction_from_power_absorption(0.05, frequency_hz))]
    entries += [(f"doped {r:g} ohm.cm", doped_silicon_index(frequency_hz, r))
                for r in resistivities_ohm_cm]
    for label, index in entries:
        errors = np.array([
            extinction_error(index, angle, field_noise_fraction)
            for angle in angles])
        position = int(np.nanargmin(errors))
        true_k = -index.imag
        print(f"   {label:>22} {true_k:>10.5f} {np.rad2deg(angles[position]):>11.2f} "
              f"{errors[position]:>10.5f} {errors[position]/max(true_k,1e-12):>10.2f}")
    print("   -> the optimum sits a few degrees OFF Brewster: at Brewster the p channel")
    print("      vanishes and the field noise wins. Sensitivity and signal pull opposite ways.")


def print_measurability_verdict(frequency_hz=1.0e12, field_noise_fraction=0.005):
    print(f"\n[verdict] is k measurable? {frequency_hz/1e12:.1f} THz, "
          f"{field_noise_fraction:.1%} field noise, at each sample's own best angle")
    angles = np.deg2rad(np.arange(40.0, 88.0, 0.25))
    entries = [("HR-Si (a=0.05/cm)", HIGH_RESISTIVITY_SILICON_INDEX
                - 1j * extinction_from_power_absorption(0.05, frequency_hz)),
               ("HR-Si (a=0.3/cm)", HIGH_RESISTIVITY_SILICON_INDEX
                - 1j * extinction_from_power_absorption(0.3, frequency_hz))]
    entries += [(f"doped {r:g} ohm.cm", doped_silicon_index(frequency_hz, r))
                for r in (10.0, 1.0, 0.1)]
    print(f"   {'sample':>22} {'k true':>10} {'best dk':>10} {'SNR on k':>10} {'verdict':>22}")
    for label, index in entries:
        errors = np.array([
            extinction_error(index, angle, field_noise_fraction)
            for angle in angles])
        best = np.nanmin(errors)
        true_k = -index.imag
        ratio = true_k / best
        verdict = ("invisible" if ratio < 1 else
                   "marginal" if ratio < 3 else
                   "measurable" if ratio < 30 else "easy")
        print(f"   {label:>22} {true_k:>10.5f} {best:>10.5f} {ratio:>10.2f} {verdict:>22}")


def validate(verbose=True):
    failures = []
    # Known answer: alpha = 4*pi*k/lambda must round trip.
    for absorption, frequency in ((0.05, 1e12), (1.0, 2e12)):
        k = extinction_from_power_absorption(absorption, frequency)
        wavelength_cm = SPEED_OF_LIGHT / frequency * 100.0
        if abs(4 * np.pi * k / wavelength_cm - absorption) > 1e-12:
            failures.append("absorption/k conversion not self-consistent")
    # Doped silicon must be passive and approach the static permittivity as doping vanishes.
    lightly_doped = doped_silicon_index(1e12, 1e6)
    if abs(lightly_doped.real - np.sqrt(SILICON_STATIC_PERMITTIVITY)) > 1e-3:
        failures.append(f"undoped limit wrong: {lightly_doped}")
    if doped_silicon_index(1e12, 1.0).imag > 0:
        failures.append("doped silicon index is not passive")
    # The nearest-branch inverter must round trip exactly on unperturbed data.
    for sample in (HIGH_RESISTIVITY_SILICON_INDEX - 1e-4j, doped_silicon_index(1e12, 1.0)):
        for angle_deg in (45.0, 70.0, 73.7, 80.0):
            angle = np.deg2rad(angle_deg)
            ratio = ellipsometric_ratio(1.0, sample, angle)
            if abs(_index_from_ratio_nearest(ratio, angle, sample) - sample) > 1e-10:
                failures.append(f"nearest-branch inversion failed at {angle_deg} for {sample}")

    # Brewster: |rho| must have its minimum at arctan(n) for a lossless sample.
    angles = np.deg2rad(np.arange(60.0, 85.0, 0.01))
    magnitudes = np.abs(ellipsometric_ratio(1.0, HIGH_RESISTIVITY_SILICON_INDEX - 0j, angles))
    brewster = np.rad2deg(np.arctan(HIGH_RESISTIVITY_SILICON_INDEX))
    if abs(np.rad2deg(angles[int(np.argmin(magnitudes))]) - brewster) > 0.02:
        failures.append("Brewster minimum not at arctan(n)")
    if verbose:
        print("validation:", "PASS (5 known-answer checks)" if not failures else "FAIL")
        for failure in failures:
            print("   ", failure)
    return failures


def main():
    validate()
    print_what_k_actually_is()
    print_k_sensitivity_vs_angle()
    print_optimum_angle_scan()
    print_measurability_verdict()
    print("\ndone.")


if __name__ == "__main__":
    main()
