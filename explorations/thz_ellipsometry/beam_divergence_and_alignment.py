"""Two questions that decide whether the ellipsometry plan survives contact with our bench.

(1) ALIGNMENT AS A FITTED QUANTITY.  Our samples do not reflect visible light usefully, so a
    camera is not an alignment handle.  Can we place the sample approximately and recover the
    geometry from the data itself, using polarisation degrees of freedom on an optical path
    that never changes?

(2) DIFFRACTION-LIMITED ANGULAR BLUR.  A beam of diameter d carries an unavoidable angular
    spread of order lambda/d, so low frequencies diverge far more than high ones.  With a
    1.6 mm spot or a 5 mm aperture, how large is the blur and what does it cost?

An important modelling correction lives here.  The earlier budget averaged rho over the
angular spread, i.e. <rho>.  That is wrong.  The detector measures the coherent field in each
polarisation separately, so the measured ratio is <r_p>/<r_s>: the angular averaging happens
BEFORE the division, and whatever is common to both polarisations cancels.  The correct model
is substantially more forgiving, and it is the one used below.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ellipsometry_conditioning_analysis import (
    SPEED_OF_LIGHT,
    index_error_amplification,
    index_error_from_channel_gain,
    index_error_from_tilt,
    observable_builders,
    drude_permittivity,
    ellipsometric_ratio,
    fresnel_reflection_p,
    fresnel_reflection_s,
    index_from_ellipsometric_ratio,
    passive_index_from_permittivity,
    sample_index,
)

HIGH_RESISTIVITY_SILICON = 3.418 - 0.0j


# ---------------------------------------------------------------------------
# Part 1 -- the diffraction invariant
# ---------------------------------------------------------------------------

def gaussian_divergence_half_angle_rad(beam_diameter_m, wavelength_m):
    """Far-field 1/e^2 half-angle of a Gaussian beam of diameter d = 2*w0.

    theta = lambda / (pi * w0) = 2 * lambda / (pi * d).  This is not an engineering
    limitation that better optics can remove; it is the beam parameter product.
    """
    return 2.0 * wavelength_m / (np.pi * beam_diameter_m)


def airy_half_angle_rad(aperture_diameter_m, wavelength_m):
    """First-zero half-angle of a hard circular aperture, 1.22 * lambda / D."""
    return 1.22 * wavelength_m / aperture_diameter_m


def minimum_beam_diameter_m(target_half_angle_rad, wavelength_m):
    """Smallest spot compatible with a given angular spread -- the inverse invariant."""
    return 2.0 * wavelength_m / (np.pi * target_half_angle_rad)


def print_divergence_invariant(frequencies_hz=(0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12),
                               beam_diameter_mm=1.6, aperture_diameter_mm=5.0):
    print(f"\n[diffraction invariant] angular half-spread for a {beam_diameter_mm} mm Gaussian "
          f"spot and a {aperture_diameter_mm} mm hard aperture")
    print(f"   {'f [THz]':>8} {'lambda [mm]':>12} {'gauss [deg]':>13} {'airy [deg]':>12} "
          f"{'d for 1 deg [mm]':>18}")
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        gaussian = gaussian_divergence_half_angle_rad(beam_diameter_mm * 1e-3, wavelength)
        airy = airy_half_angle_rad(aperture_diameter_mm * 1e-3, wavelength)
        needed = minimum_beam_diameter_m(np.deg2rad(1.0), wavelength) * 1e3
        print(f"   {frequency/1e12:>8.2f} {wavelength*1e3:>12.2f} {np.rad2deg(gaussian):>13.1f} "
              f"{np.rad2deg(airy):>12.1f} {needed:>18.1f}")
    print("   -> a 1 deg spread at 0.3 THz needs a ~37 mm spot, and at 70 deg incidence a")
    print("      ~107 mm footprint. Not achievable on our samples: the low band is blur-limited.")


# ---------------------------------------------------------------------------
# Part 2 -- what the angular blur actually does to the measurement
# ---------------------------------------------------------------------------

def effective_jones_with_divergence(index_sample, incidence_angle_rad, divergence_half_angle_rad,
                                    index_incident=1.0, quadrature_points=15):
    """Angle-averaged reflection Jones matrix for a converging beam.

    Each plane-wave component of the beam hits at its own incidence angle and its own
    orientation of the plane of incidence.  For a component deviating by (a, b) from the
    central ray -- a in the plane of incidence, b out of it -- the local incidence angle is
    theta + a and the local s/p frame is rotated by psi = b / sin(theta).  The measured field
    is the coherent sum, so the AVERAGE IS TAKEN ON THE JONES MATRIX, not on rho.

    Consequence worth noting: the off-diagonal terms are odd in b, so for a symmetric beam the
    divergence-induced cross-polarisation cancels.  It only survives if the beam is clipped
    asymmetrically or is astigmatic.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
    weights = weights / weights.sum()
    sine = np.sin(incidence_angle_rad)

    jones = np.zeros((2, 2), dtype=complex)
    for in_plane, weight_x in zip(nodes, weights):
        local_angle = incidence_angle_rad + divergence_half_angle_rad * in_plane
        r_p = fresnel_reflection_p(index_incident, index_sample, local_angle)
        r_s = fresnel_reflection_s(index_incident, index_sample, local_angle)
        diagonal = np.array([[r_p, 0.0], [0.0, r_s]], dtype=complex)
        for out_of_plane, weight_y in zip(nodes, weights):
            frame_rotation = divergence_half_angle_rad * out_of_plane / sine
            cosine, sine_rotation = np.cos(frame_rotation), np.sin(frame_rotation)
            rotation = np.array([[cosine, -sine_rotation], [sine_rotation, cosine]])
            jones += weight_x * weight_y * (rotation @ diagonal @ rotation.T)
    return jones


def index_error_from_divergence(index_sample, incidence_angle_rad, divergence_half_angle_rad,
                                index_incident=1.0):
    """Index error from treating a divergent beam as a plane wave (correct <r_p>/<r_s> model)."""
    jones = effective_jones_with_divergence(index_sample, incidence_angle_rad,
                                            divergence_half_angle_rad, index_incident)
    recovered = index_from_ellipsometric_ratio(jones[0, 0] / jones[1, 1],
                                               index_incident, incidence_angle_rad)
    return abs(recovered - index_sample), abs(jones[0, 1] / jones[1, 1])


def index_error_from_divergence_naive(index_sample, incidence_angle_rad,
                                      divergence_half_angle_rad, index_incident=1.0,
                                      quadrature_points=15):
    """The earlier, WRONG model: average rho itself.  Kept to size the correction."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
    weights = weights / weights.sum()
    angles = incidence_angle_rad + divergence_half_angle_rad * nodes
    averaged = np.sum(weights * ellipsometric_ratio(index_incident, index_sample, angles))
    recovered = index_from_ellipsometric_ratio(averaged, index_incident, incidence_angle_rad)
    return abs(recovered - index_sample)


def print_divergence_cost(frequencies_hz=(0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12),
                          incidence_angle_deg=70.0, beam_diameter_mm=1.6,
                          aperture_diameter_mm=5.0):
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[divergence cost] index error at {incidence_angle_deg:.0f} deg from treating the "
          f"beam as a plane wave, CNT model")
    print(f"   {'f [THz]':>8} {'|N|':>7} {'spread(1.6mm)':>14} {'|dN|':>8} {'rel':>7} "
          f"{'spread(5mm)':>13} {'|dN|':>8} {'rel':>7} {'naive |dN|':>12}")
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        index = sample_index(frequency)
        row = f"   {frequency/1e12:>8.2f} {abs(index):>7.2f}"
        for diameter_mm in (beam_diameter_mm, aperture_diameter_mm):
            spread = gaussian_divergence_half_angle_rad(diameter_mm * 1e-3, wavelength)
            error, _ = index_error_from_divergence(index, angle, spread)
            row += f" {np.rad2deg(spread):>13.1f} {error:>8.3f} {error/abs(index):>6.1%}"
        spread = gaussian_divergence_half_angle_rad(beam_diameter_mm * 1e-3, wavelength)
        row += f" {index_error_from_divergence_naive(index, angle, spread):>12.3f}"
        print(row)
    print("   'naive' is the <rho> model used in the first budget; the correct <r_p>/<r_s>")
    print("   averaging is far more forgiving because common-mode angular structure cancels.")


def print_divergence_correction_factor(incidence_angle_deg=70.0, frequency_hz=1.0e12,
                                       spreads_deg=(0.5, 1.0, 2.0, 4.0, 7.0, 11.0)):
    angle = np.deg2rad(incidence_angle_deg)
    index = sample_index(frequency_hz)
    print(f"\n[divergence model] correct vs naive, {frequency_hz/1e12:.1f} THz, "
          f"{incidence_angle_deg:.0f} deg, |N| = {abs(index):.2f}")
    print(f"   {'spread [deg]':>13} {'|dN| correct':>14} {'|dN| naive':>12} {'ratio':>8} "
          f"{'|r_ps/r_ss|':>13}")
    for spread_deg in spreads_deg:
        spread = np.deg2rad(spread_deg)
        error, cross = index_error_from_divergence(index, angle, spread)
        naive = index_error_from_divergence_naive(index, angle, spread)
        print(f"   {spread_deg:>13.1f} {error:>14.3f} {naive:>12.3f} "
              f"{naive/max(error,1e-12):>8.1f}x {cross:>13.2e}")
    print("   cross-polarisation from a SYMMETRIC divergent beam cancels by parity "
          "(odd in the out-of-plane angle).")


# ---------------------------------------------------------------------------
# Part 3 -- can we fit the geometry instead of aligning it?
# ---------------------------------------------------------------------------

def out_of_plane_tilt_from_cross_polarisation(jones):
    """Recover an out-of-plane sample tilt from the Jones matrix, MODEL-FREE.

    For an ISOTROPIC sample tilted out of plane by delta, the lab-frame Jones matrix is
    R(delta) diag(r_p, r_s) R(-delta), so

        r_ps / (r_pp - r_ss) = tan(2 delta) / 2

    The material cancels completely: no model, no reference, no knowledge of n.  The tilt is
    therefore SELF-MEASURING on any isotropic sample -- which is exactly what a bare silicon
    wafer or the bare backing flat is.
    """
    return 0.5 * np.arctan(2.0 * jones[0, 1] / (jones[0, 0] - jones[1, 1]))


def tilted_isotropic_jones(index_sample, incidence_angle_rad, tilt_rad, index_incident=1.0):
    r_p = fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad)
    r_s = fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad)
    cosine, sine = np.cos(tilt_rad), np.sin(tilt_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return rotation @ np.array([[r_p, 0.0], [0.0, r_s]], dtype=complex) @ rotation.T


def incidence_angle_uncertainty_from_reference(reference_index, nominal_angle_deg,
                                               relative_error=0.005, frequency_count=20):
    """How well a KNOWN reference sample pins the incidence angle.

    rho depends on theta and on the (known) index, so with the index known there is exactly
    one unknown and many frequencies.  Returns the 1-sigma angle uncertainty in degrees.
    """
    angle = np.deg2rad(nominal_angle_deg)
    ratio = ellipsometric_ratio(1.0, reference_index, angle)
    step = np.deg2rad(0.01)
    derivative = (ellipsometric_ratio(1.0, reference_index, angle + step)
                  - ellipsometric_ratio(1.0, reference_index, angle - step)) / (2 * step)
    noise = relative_error * abs(ratio)
    information = frequency_count * (abs(derivative) ** 2) / noise**2
    return np.rad2deg(1.0 / np.sqrt(information))


def print_angle_from_reference(angles_deg=(45.0, 60.0, 65.0, 70.0, 73.7, 75.0, 80.0),
                               relative_error=0.005, frequency_count=20):
    print(f"\n[angle from a reference] 1-sigma incidence-angle uncertainty, {relative_error:.1%} "
          f"error on rho, {frequency_count} independent frequencies")
    print(f"   {'nominal [deg]':>14} {'HR-Si [deg]':>13} {'gold [deg]':>12} {'CNT [deg]':>12}")
    gold = 1000.0 - 1000.0j
    cnt = sample_index(1.0e12)
    for nominal in angles_deg:
        print(f"   {nominal:>14.1f} "
              f"{incidence_angle_uncertainty_from_reference(HIGH_RESISTIVITY_SILICON, nominal, relative_error, frequency_count):>13.3f} "
              f"{incidence_angle_uncertainty_from_reference(gold, nominal, relative_error, frequency_count):>12.3f} "
              f"{incidence_angle_uncertainty_from_reference(cnt, nominal, relative_error, frequency_count):>12.3f}")
    print("   Si is the angle gauge; gold is useless for it (rho ~ -1 regardless of angle).")


def joint_fit_angle_and_drude(frequencies_hz, true_angle_deg, relative_noise=0.005,
                              trials=20, seed=3, angle_prior_deg=None):
    """Can the incidence angle be fitted jointly with the material, from the sample alone?"""
    true_angle = np.deg2rad(true_angle_deg)
    truth = np.array([6500.0, 30e-15, true_angle])

    def model(parameters):
        index = passive_index_from_permittivity(
            drude_permittivity(frequencies_hz, parameters[0], parameters[1], 4.0))
        ratio = ellipsometric_ratio(1.0, index, parameters[2])
        return np.concatenate([ratio.real, ratio.imag])

    clean = model(truth)
    generator = np.random.default_rng(seed)
    scale = np.array([1e4, 1e-14, 1.0])
    recovered = []
    for _ in range(trials):
        noisy = clean + relative_noise * np.abs(clean).mean() * generator.normal(size=clean.size)

        def residual(scaled):
            difference = model(scaled * scale) - noisy
            if angle_prior_deg is not None:
                penalty = (np.rad2deg(scaled[2]) - true_angle_deg) / angle_prior_deg
                difference = np.concatenate([difference, [penalty * np.abs(clean).mean()]])
            return difference

        start = truth / scale * generator.uniform(0.85, 1.15, size=3)
        result = least_squares(residual, start,
                               bounds=([1e-2, 0.1, np.deg2rad(40) ], [1e2, 100.0, np.deg2rad(89)]),
                               xtol=1e-13, ftol=1e-13)
        recovered.append(result.x * scale)
    recovered = np.array(recovered)
    return recovered


def print_joint_angle_fit(frequencies_hz=None, angles_deg=(70.0, 75.0), relative_noise=0.005):
    if frequencies_hz is None:
        frequencies_hz = np.linspace(0.6e12, 3.0e12, 15)
    print(f"\n[angle fitted from the sample itself] joint Drude + incidence angle, "
          f"{relative_noise:.1%} noise, 20 trials")
    print(f"   {'true angle':>11} {'fitted angle':>22} {'sigma_dc [S/cm]':>22} {'tau [fs]':>18}")
    for true_angle_deg in angles_deg:
        recovered = joint_fit_angle_and_drude(frequencies_hz, true_angle_deg,
                                              relative_noise=relative_noise)
        angles = np.rad2deg(recovered[:, 2])
        conductivity = recovered[:, 0] / 100.0
        scattering = recovered[:, 1] * 1e15
        def summarise(values):
            return np.median(values), 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))
        angle_stat, conductivity_stat, scattering_stat = (summarise(angles),
                                                          summarise(conductivity),
                                                          summarise(scattering))
        print(f"   {true_angle_deg:>11.1f} {angle_stat[0]:>13.2f} +/-{angle_stat[1]:<6.2f} "
              f"{conductivity_stat[0]:>13.1f} +/-{conductivity_stat[1]:<6.1f} "
              f"{scattering_stat[0]:>10.1f} +/-{scattering_stat[1]:<6.1f}")
    print("   true: sigma_dc 65.0 S/cm, tau 30.0 fs")


def print_tilt_self_measurement(tilts_deg=(0.05, 0.1, 0.5, 1.0, 3.0), incidence_angle_deg=70.0,
                                relative_noise=0.005, trials=200, seed=9):
    angle = np.deg2rad(incidence_angle_deg)
    index = sample_index(1.0e12)
    generator = np.random.default_rng(seed)
    print(f"\n[tilt self-measurement] recovering out-of-plane tilt from r_ps/(r_pp - r_ss), "
          f"model-free, {relative_noise:.1%} noise")
    print(f"   {'true tilt [deg]':>16} {'CNT recovered':>20} {'HR-Si recovered':>22}")
    for tilt_deg in tilts_deg:
        row = f"   {tilt_deg:>16.2f}"
        for material in (index, HIGH_RESISTIVITY_SILICON):
            estimates = []
            for _ in range(trials):
                jones = tilted_isotropic_jones(material, angle, np.deg2rad(tilt_deg))
                scale = relative_noise * abs(jones[1, 1])
                noisy = jones + scale * (generator.normal(size=(2, 2))
                                         + 1j * generator.normal(size=(2, 2))) / np.sqrt(2)
                estimates.append(np.rad2deg(
                    out_of_plane_tilt_from_cross_polarisation(noisy).real))
            estimates = np.array(estimates)
            row += (f" {np.median(estimates):>12.3f} "
                    f"+/-{0.5*(np.percentile(estimates,84)-np.percentile(estimates,16)):<6.3f}")
        print(row)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(verbose=True):
    failures = []
    angle = np.deg2rad(70.0)
    index = sample_index(1.0e12)

    # Zero divergence must reproduce the plane-wave answer exactly.
    jones = effective_jones_with_divergence(index, angle, 0.0)
    if abs(jones[0, 0] - fresnel_reflection_p(1.0, index, angle)) > 1e-12:
        failures.append("zero-divergence r_pp wrong")
    if abs(jones[0, 1]) > 1e-14:
        failures.append("zero-divergence cross-pol non-zero")

    # Symmetric divergence must not generate net cross-polarisation.
    jones = effective_jones_with_divergence(index, angle, np.deg2rad(8.0))
    if abs(jones[0, 1] / jones[1, 1]) > 1e-10:
        failures.append(f"symmetric beam produced cross-pol {jones[0,1]/jones[1,1]}")

    # The tilt estimator must invert its own forward model exactly.
    for tilt_deg in (0.1, 1.0, 5.0):
        jones = tilted_isotropic_jones(index, angle, np.deg2rad(tilt_deg))
        recovered = np.rad2deg(out_of_plane_tilt_from_cross_polarisation(jones).real)
        if abs(recovered - tilt_deg) > 1e-8:
            failures.append(f"tilt estimator failed at {tilt_deg}: {recovered}")

    # The diffraction invariant must be self-consistent.
    wavelength = SPEED_OF_LIGHT / 1e12
    diameter = minimum_beam_diameter_m(np.deg2rad(2.0), wavelength)
    if abs(np.rad2deg(gaussian_divergence_half_angle_rad(diameter, wavelength)) - 2.0) > 1e-9:
        failures.append("divergence invariant not self-consistent")

    if verbose:
        print("validation:", "PASS (4 known-answer checks)" if not failures else "FAIL")
        for failure in failures:
            print("   ", failure)
    return failures



def footprint_spread_invariant(wavelength_m, incidence_angle_rad):
    """footprint * angular_spread = 2*lambda/(pi*cos(theta)) -- you cannot beat this.

    The illuminated length along the plane of incidence is d/cos(theta) and the angular
    half-spread is 2*lambda/(pi*d), so their product is independent of d.  Small footprint and
    small angular blur are the same trade, and only a shorter wavelength relaxes it.
    """
    return 2.0 * wavelength_m / (np.pi * np.cos(incidence_angle_rad))


def print_footprint_tradeoff(frequencies_hz=(0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12),
                             incidence_angle_deg=70.0, spreads_deg=(1.0, 2.0, 4.0)):
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[footprint vs blur] the trade is fixed by diffraction, at "
          f"{incidence_angle_deg:.0f} deg incidence")
    print(f"   {'f [THz]':>8} {'invariant [mm.deg]':>20}" +
          "".join(f"{f'footprint@{s:g}deg [mm]':>22}" for s in spreads_deg))
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        invariant = footprint_spread_invariant(wavelength, angle)
        row = f"   {frequency/1e12:>8.2f} {invariant*1e3*180/np.pi:>20.1f}"
        for spread_deg in spreads_deg:
            row += f"{invariant/np.deg2rad(spread_deg)*1e3:>22.1f}"
        print(row)
    print("   Read it as: to hold the blur at X degrees you must illuminate this much flat sample.")


def print_divergence_knowledge_requirement(frequency_hz=1.0e12, incidence_angle_deg=70.0,
                                           true_spread_deg=2.2,
                                           fractional_errors=(0.05, 0.10, 0.25, 0.50)):
    """If we MODEL the angular average, how well must the spread be known?"""
    angle = np.deg2rad(incidence_angle_deg)
    index = sample_index(frequency_hz)
    truth = effective_jones_with_divergence(index, angle, np.deg2rad(true_spread_deg))
    uncorrected, _ = index_error_from_divergence(index, angle, np.deg2rad(true_spread_deg))
    print(f"\n[divergence as a modelled bias] {frequency_hz/1e12:.1f} THz, "
          f"{incidence_angle_deg:.0f} deg, true spread {true_spread_deg} deg")
    print(f"   uncorrected (plane-wave assumption): |dN| = {uncorrected:.3f}")
    print(f"   {'error in assumed spread':>24} {'residual |dN|':>15} {'suppression':>13}")
    for fractional in fractional_errors:
        assumed = np.deg2rad(true_spread_deg * (1.0 + fractional))
        # Fit the index that reproduces the measured Jones ratio under the ASSUMED spread.
        target = truth[0, 0] / truth[1, 1]

        def residual(values):
            trial = values[0] - 1j * abs(values[1])
            jones = effective_jones_with_divergence(trial, angle, assumed)
            difference = jones[0, 0] / jones[1, 1] - target
            return [difference.real, difference.imag]

        result = least_squares(residual, [index.real, -index.imag], xtol=1e-12, ftol=1e-12)
        recovered = result.x[0] - 1j * abs(result.x[1])
        residual_error = abs(recovered - index)
        print(f"   {fractional:>23.0%} {residual_error:>15.3f} "
              f"{uncorrected/max(residual_error,1e-12):>12.1f}x")
    print("   -> modelling the average turns the blur from a systematic into a small residual;")
    print("      knowing the spread to ~10% (a knife-edge measurement) is enough.")



def print_fixed_footprint_budget(footprint_mm=15.0, incidence_angle_deg=70.0,
                                 frequencies_hz=(0.3e12, 0.5e12, 0.8e12, 1.0e12, 1.5e12,
                                                 2.0e12, 3.0e12),
                                 modelling_accuracy=0.10):
    """The practical question: given the flat sample we can actually make, what is the blur?

    The usable low-frequency edge of the band is set by SAMPLE SIZE, not by signal-to-noise
    and not by conditioning -- which is a different limit from any considered so far.
    """
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[fixed footprint] {footprint_mm:.0f} mm of flat sample at "
          f"{incidence_angle_deg:.0f} deg incidence")
    print(f"   {'f [THz]':>8} {'|N|':>7} {'blur [deg]':>11} {'|dN| raw':>10} {'rel':>7} "
          f"{'|dN| modelled':>15} {'rel':>7}")
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        index = sample_index(frequency)
        spread = footprint_spread_invariant(wavelength, angle) / (footprint_mm * 1e-3)
        raw, _ = index_error_from_divergence(index, angle, spread)
        assumed = spread * (1.0 + modelling_accuracy)
        truth = effective_jones_with_divergence(index, angle, spread)
        target = truth[0, 0] / truth[1, 1]

        def residual(values):
            trial = values[0] - 1j * abs(values[1])
            jones = effective_jones_with_divergence(trial, angle, assumed)
            difference = jones[0, 0] / jones[1, 1] - target
            return [difference.real, difference.imag]

        result = least_squares(residual, [index.real, -index.imag], xtol=1e-12, ftol=1e-12)
        modelled = abs((result.x[0] - 1j * abs(result.x[1])) - index)
        print(f"   {frequency/1e12:>8.2f} {abs(index):>7.2f} {np.rad2deg(spread):>11.2f} "
              f"{raw:>10.3f} {raw/abs(index):>6.1%} {modelled:>15.3f} "
              f"{modelled/abs(index):>6.1%}")
    print(f"   'modelled' assumes the angular spread is known to {modelling_accuracy:.0%} "
          "(a knife-edge measurement).")



def print_corrected_budget(frequencies_hz=(0.5e12, 1.0e12, 2.0e12), incidence_angles_deg=(70.0, 75.0),
                           footprint_mm=15.0, relative_observable_error=0.005,
                           angle_uncertainty_deg=0.05, channel_gain_error=0.002,
                           modelling_accuracy=0.10):
    """The error budget with the CORRECTED divergence term.

    Supersedes the divergence row of ellipsometry_conditioning_analysis.combined_error_budget,
    which used both a too-small angular spread and the wrong <rho> averaging.
    """
    print(f"\n[corrected budget] {footprint_mm:.0f} mm footprint, noise "
          f"{relative_observable_error:.1%}, tilt {angle_uncertainty_deg} deg, gain "
          f"{channel_gain_error:.1%}, divergence modelled to {modelling_accuracy:.0%}")
    print(f"   {'f [THz]':>8} {'theta':>7} {'noise':>8} {'tilt':>8} {'gain':>8} "
          f"{'diverg':>8} {'TOTAL':>8} {'rel |N|':>9}")
    for frequency in frequencies_hz:
        index = sample_index(frequency)
        wavelength = SPEED_OF_LIGHT / frequency
        for angle_deg in incidence_angles_deg:
            angle = np.deg2rad(angle_deg)
            noise = (index_error_amplification(observable_builders(1.0, angle)["rho = r_p/r_s"],
                                               index) * relative_observable_error)
            tilt = index_error_from_tilt(index, 1.0, angle, np.deg2rad(angle_uncertainty_deg))
            gain = index_error_from_channel_gain(index, 1.0, angle, channel_gain_error)
            spread = footprint_spread_invariant(wavelength, angle) / (footprint_mm * 1e-3)
            truth = effective_jones_with_divergence(index, angle, spread)
            target = truth[0, 0] / truth[1, 1]
            assumed = spread * (1.0 + modelling_accuracy)

            def residual(values):
                trial = values[0] - 1j * abs(values[1])
                jones = effective_jones_with_divergence(trial, angle, assumed)
                difference = jones[0, 0] / jones[1, 1] - target
                return [difference.real, difference.imag]

            result = least_squares(residual, [index.real, -index.imag], xtol=1e-12, ftol=1e-12)
            divergence = abs((result.x[0] - 1j * abs(result.x[1])) - index)
            total = np.sqrt(noise**2 + tilt**2 + gain**2 + divergence**2)
            print(f"   {frequency/1e12:>8.2f} {angle_deg:>7.0f} {noise:>8.3f} {tilt:>8.3f} "
                  f"{gain:>8.3f} {divergence:>8.3f} {total:>8.3f} {total/abs(index):>8.1%}")




# ---------------------------------------------------------------------------
# Part 4 -- clipping versus focusing: what actually arrives at the sample
# ---------------------------------------------------------------------------

AIRY_FIRST_ZERO_COEFFICIENT = 1.22          # encloses 83.8% of the power
GAUSSIAN_ENERGY_RADIUS_COEFFICIENT = 2.0 / np.pi   # 1/e^2 radius, encloses 86.5%


def aperture_collimation_distance_m(aperture_diameter_m, wavelength_m):
    """Distance over which a hard aperture actually confines the beam (Fresnel number = 1).

    z_c = D^2 / (4 * lambda).  Beyond this the aperture has stopped being a mask and has
    become a diffracting source: the beam is LARGER than if the aperture had not been there.
    """
    return aperture_diameter_m**2 / (4.0 * wavelength_m)


def clipped_beam_radius_m(aperture_diameter_m, wavelength_m, distance_m):
    """Beam radius a distance z after a hard aperture.

    Engineering interpolation between the geometric shadow and the far-field Airy cone:
    w(z) = sqrt((D/2)^2 + (1.22 * lambda * z / D)^2).
    """
    geometric = aperture_diameter_m / 2.0
    diffractive = AIRY_FIRST_ZERO_COEFFICIENT * wavelength_m * distance_m / aperture_diameter_m
    return np.sqrt(geometric**2 + diffractive**2)


def focused_beam_waist_radius_m(collimated_diameter_m, wavelength_m, focal_length_m):
    """Waist radius of a Gaussian beam focused by an optic of focal length f."""
    return wavelength_m * focal_length_m / (np.pi * collimated_diameter_m / 2.0)


def focused_beam_convergence_half_angle_rad(collimated_diameter_m, focal_length_m):
    """Geometric convergence half-angle, w_in / f -- frequency INDEPENDENT."""
    return (collimated_diameter_m / 2.0) / focal_length_m


def print_clipping_is_self_defeating(aperture_diameters_mm=(5.0, 10.0, 16.0),
                                     standoff_mm=100.0,
                                     frequencies_hz=(0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12),
                                     sample_size_mm=10.0):
    print(f"\n[clipping] beam DIAMETER at the sample, {standoff_mm:.0f} mm after a hard aperture "
          f"(sample is {sample_size_mm:.0f} mm)")
    header = f"   {'f [THz]':>8}" + "".join(
        f"{f'{d:g} mm ap [mm]':>17}" for d in aperture_diameters_mm)
    print(header)
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        row = f"   {frequency/1e12:>8.2f}"
        for diameter_mm in aperture_diameters_mm:
            diameter = clipped_beam_radius_m(diameter_mm * 1e-3, wavelength,
                                             standoff_mm * 1e-3) * 2e3
            flag = " " if diameter <= sample_size_mm else "*"
            row += f"{diameter:>16.1f}{flag}"
        print(row)
    print("   * = overfills the sample.  Note the 5 mm aperture is WORSE than no aperture at all")
    print("     below ~1 THz: clipping converts a mask into a diffracting source.")
    print(f"\n   collimation distance of each aperture (Fresnel number = 1), i.e. how far it")
    print(f"   actually confines the beam before it starts to spread:")
    print(f"   {'f [THz]':>8}" + "".join(f"{f'{d:g} mm ap [mm]':>17}" for d in aperture_diameters_mm))
    for frequency in frequencies_hz:
        wavelength = SPEED_OF_LIGHT / frequency
        row = f"   {frequency/1e12:>8.2f}"
        for diameter_mm in aperture_diameters_mm:
            row += f"{aperture_collimation_distance_m(diameter_mm*1e-3, wavelength)*1e3:>17.1f}"
        print(row)
    print("   -> a 5 mm aperture holds the beam for ~6 mm at 0.3 THz and ~21 mm at 1 THz.")
    print("      At a 100 mm standoff it is not a mask, it is an antenna.")


def print_focus_instead(collimated_diameter_mm=16.0, focal_lengths_mm=(75.0, 100.0, 150.0, 200.0),
                        frequencies_hz=(0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12),
                        sample_size_mm=10.0, incidence_angle_deg=70.0):
    allowed_beam_mm = sample_size_mm * np.cos(np.deg2rad(incidence_angle_deg))
    print(f"\n[focusing] spot DIAMETER at the sample for the full {collimated_diameter_mm:g} mm beam, "
          f"and the price in blur")
    print(f"   sample {sample_size_mm:g} mm at {incidence_angle_deg:.0f} deg allows a beam of "
          f"{allowed_beam_mm:.2f} mm")
    print(f"   {'EFL [mm]':>9} {'blur [deg]':>11}" +
          "".join(f"{f'{f/1e12:g} THz':>11}" for f in frequencies_hz))
    for focal_mm in focal_lengths_mm:
        blur = focused_beam_convergence_half_angle_rad(collimated_diameter_mm * 1e-3,
                                                       focal_mm * 1e-3)
        row = f"   {focal_mm:>9.0f} {np.rad2deg(blur):>11.2f}"
        for frequency in frequencies_hz:
            wavelength = SPEED_OF_LIGHT / frequency
            diameter = focused_beam_waist_radius_m(collimated_diameter_mm * 1e-3, wavelength,
                                                   focal_mm * 1e-3) * 2e3
            flag = " " if diameter <= allowed_beam_mm else "*"
            row += f"{diameter:>10.2f}{flag}"
        print(row)
    print("   * = beam wider than the sample allows.  The blur column is frequency-INDEPENDENT")
    print("     (geometric w/f), which is the whole advantage of focusing over clipping.")


def spot_fits_crossover_frequency_hz(spot_diameter_function, sample_size_mm,
                                     incidence_angle_deg=70.0):
    """Lowest frequency whose spot still fits the sample, by bisection."""
    allowed = sample_size_mm * 1e-3 * np.cos(np.deg2rad(incidence_angle_deg))
    low, high = 0.05e12, 10e12
    for _ in range(80):
        middle = 0.5 * (low + high)
        if spot_diameter_function(SPEED_OF_LIGHT / middle) > allowed:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def print_crossover_comparison(sample_size_mm=10.0, incidence_angle_deg=70.0,
                               aperture_diameter_mm=5.0, standoff_mm=100.0,
                               collimated_diameter_mm=16.0, focal_lengths_mm=(100.0, 150.0)):
    print(f"\n[crossover] lowest usable frequency for a {sample_size_mm:g} mm sample at "
          f"{incidence_angle_deg:.0f} deg")
    clipped = spot_fits_crossover_frequency_hz(
        lambda wavelength: 2 * clipped_beam_radius_m(aperture_diameter_mm * 1e-3, wavelength,
                                                     standoff_mm * 1e-3),
        sample_size_mm, incidence_angle_deg)
    print(f"   {aperture_diameter_mm:g} mm aperture, {standoff_mm:.0f} mm standoff: "
          f"{clipped/1e12:.2f} THz")
    for focal_mm in focal_lengths_mm:
        focused = spot_fits_crossover_frequency_hz(
            lambda wavelength: 2 * focused_beam_waist_radius_m(collimated_diameter_mm * 1e-3,
                                                               wavelength, focal_mm * 1e-3),
            sample_size_mm, incidence_angle_deg)
        blur = focused_beam_convergence_half_angle_rad(collimated_diameter_mm * 1e-3,
                                                       focal_mm * 1e-3)
        print(f"   full {collimated_diameter_mm:g} mm beam focused at EFL {focal_mm:.0f} mm:      "
              f"{focused/1e12:.2f} THz   (blur {np.rad2deg(blur):.2f} deg, all frequencies)")
    print("   -> focusing the whole beam beats clipping it by a large factor in usable bandwidth.")


def print_blur_budget_vs_angle(sample_size_mm=10.0, angles_deg=(45.0, 55.0, 60.0, 65.0, 70.0, 75.0),
                               frequencies_hz=(0.5e12, 1.0e12, 2.0e12)):
    """With the sample size fixed, where is the angle optimum now?"""
    print(f"\n[angle optimum] {sample_size_mm:g} mm sample: blur is set by footprint, so the "
          f"diffraction term now FAVOURS lower angles")
    print(f"   {'f [THz]':>8} {'theta':>7} {'blur [deg]':>11} {'|dN| blur':>11} "
          f"{'|dN| noise':>11} {'|dN| total':>11} {'rel |N|':>9}")
    for frequency in frequencies_hz:
        index = sample_index(frequency)
        wavelength = SPEED_OF_LIGHT / frequency
        for angle_deg in angles_deg:
            angle = np.deg2rad(angle_deg)
            spread = footprint_spread_invariant(wavelength, angle) / (sample_size_mm * 1e-3)
            raw, _ = index_error_from_divergence(index, angle, spread)
            noise = index_error_amplification(
                observable_builders(1.0, angle)["rho = r_p/r_s"], index) * 0.005
            tilt = index_error_from_tilt(index, 1.0, angle, np.deg2rad(0.05))
            gain = index_error_from_channel_gain(index, 1.0, angle, 0.002)
            modelled = raw * 0.22       # 10%-accurate forward model, from the suppression study
            total = np.sqrt(noise**2 + tilt**2 + gain**2 + modelled**2)
            print(f"   {frequency/1e12:>8.2f} {angle_deg:>7.0f} {np.rad2deg(spread):>11.2f} "
                  f"{modelled:>11.3f} {noise:>11.3f} {total:>11.3f} {total/abs(index):>8.1%}")


# ---------------------------------------------------------------------------
# Part 5 -- tilt DOES create delay, at second order with a long lever arm
# ---------------------------------------------------------------------------

def tilt_induced_delay_s(tilt_rad, lever_arm_m):
    """Extra optical delay from a sample tilt, via the lengthened path to the EO focus.

    A tilt of delta deviates the reflected beam by 2*delta, so to reach a collection optic a
    distance L along the design axis the ray travels L/cos(2*delta) instead of L:

        extra path = L*(sec(2*delta) - 1) ~ 2*L*delta^2
        delay      = extra path / c

    Second order in delta, so it carries no sign information and vanishes at the null -- but
    with a long lever arm it is far from negligible, which is Samuel's point.
    """
    return lever_arm_m * (1.0 / np.cos(2.0 * np.asarray(tilt_rad)) - 1.0) / SPEED_OF_LIGHT


def tilt_resolution_from_timing_rad(timing_resolution_s, tilt_rad, lever_arm_m):
    """Smallest tilt change detectable from arrival time, at a given working tilt."""
    derivative = (4.0 * lever_arm_m * np.tan(2 * tilt_rad)
                  / (np.cos(2 * tilt_rad) * SPEED_OF_LIGHT))
    derivative = np.where(np.abs(derivative) < 1e-30, 1e-30, derivative)
    return timing_resolution_s / np.abs(derivative)


def print_tilt_delay(lever_arms_mm=(100.0, 300.0), tilts_deg=(0.05, 0.1, 0.5, 1.0, 2.0),
                     timing_resolution_fs=1.5):
    print(f"\n[tilt -> delay] Samuel is right: tilt lengthens the path to the EO focus")
    print(f"   {'tilt [deg]':>11}" + "".join(
        f"{f'delay @{L:g}mm [fs]':>20}" for L in lever_arms_mm) +
        "".join(f"{f'd(tilt) res @{L:g}mm':>22}" for L in lever_arms_mm))
    for tilt_deg in tilts_deg:
        tilt = np.deg2rad(tilt_deg)
        row = f"   {tilt_deg:>11.2f}"
        for lever_mm in lever_arms_mm:
            row += f"{tilt_induced_delay_s(tilt, lever_mm*1e-3)*1e15:>20.1f}"
        for lever_mm in lever_arms_mm:
            resolution = tilt_resolution_from_timing_rad(timing_resolution_fs*1e-15, tilt,
                                                         lever_mm*1e-3)
            row += f"{np.rad2deg(resolution):>21.3f} "
        print(row)
    print(f"   resolution columns: tilt change detectable with {timing_resolution_fs} fs timing.")
    print("   The dependence is QUADRATIC -> no sign information, and blind exactly at the null,")
    print("   which makes it a null-FINDING gauge: scan tilt, minimise arrival time, fit a parabola.")
    print("   It is also common-mode between s and p, so it never corrupts rho -- it is free.")


def print_energy_containment():
    print("\n[aperture vs smooth beam] why a hard edge is worse at the same diameter")
    print(f"   Airy first zero      1.22  * lambda / D   encloses 83.8% of the power")
    print(f"   Gaussian 1/e^2       0.637 * lambda / d   encloses 86.5% of the power")
    print(f"   ratio {AIRY_FIRST_ZERO_COEFFICIENT/GAUSSIAN_ENERGY_RADIUS_COEFFICIENT:.2f}x wider "
          "for the hard aperture at comparable energy containment,")
    print("   plus Airy sidelobes that a Gaussian does not have. Apodise the edge if you must clip.")




def tilt_null_bias_from_axis_offset_rad(axis_offset_m, lever_arm_m, incidence_angle_rad):
    """How far the arrival-time minimum sits from the true tilt null.

    If the goniometer axis misses the beam spot by h, tilting also TRANSLATES the surface, so
    the delay picks up a term linear in the tilt setting on top of the quadratic one:

        delay(t) = (2 cos(theta)/c) * h * (t - t0)  +  (2 L / c) * (t - t0)^2

    The vertex of that parabola is displaced from the true null t0 by h*cos(theta)/(2L).  The
    linear and quadratic terms are separable in a fit, which is why scanning the tilt and
    fitting a parabola is robust rather than merely approximate.
    """
    return axis_offset_m * np.cos(incidence_angle_rad) / (2.0 * lever_arm_m)


def print_tilt_null_procedure(axis_offsets_mm=(0.5, 1.0, 2.0, 5.0),
                              lever_arms_mm=(100.0, 300.0), incidence_angle_deg=70.0):
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[tilt null procedure] bias of the arrival-time minimum from the true tilt null")
    print(f"   {'axis offset [mm]':>18}" + "".join(
        f"{f'bias @{L:g}mm [deg]':>20}" for L in lever_arms_mm))
    for offset_mm in axis_offsets_mm:
        row = f"   {offset_mm:>18.1f}"
        for lever_mm in lever_arms_mm:
            bias = tilt_null_bias_from_axis_offset_rad(offset_mm * 1e-3, lever_mm * 1e-3, angle)
            row += f"{np.rad2deg(bias):>20.3f}"
        print(row)
    print("   Procedure: scan the tilt, record pulse arrival time, fit a + b*t + c*t^2, go to the")
    print("   vertex. Model-free, needs no visible light, works on the sample itself. With the")
    print("   rotation axis within ~1 mm of the beam spot and a 300 mm lever it lands inside the")
    print("   0.2 deg specification; the Si ellipsometric reading then refines it further.")


def print_truncation_consequences(sample_size_mm=10.0, incidence_angle_deg=70.0,
                                  spot_diameters_mm=(3.0, 5.0, 10.0, 15.0)):
    """What overfilling the sample actually costs -- and what it does NOT cost."""
    allowed = sample_size_mm * np.cos(np.deg2rad(incidence_angle_deg))
    print(f"\n[truncation] overfilling a {sample_size_mm:g} mm sample at "
          f"{incidence_angle_deg:.0f} deg (accepts a {allowed:.2f} mm beam)")
    print(f"   {'spot [mm]':>11} {'power collected':>17} {'loss [dB]':>11}")
    for spot_mm in spot_diameters_mm:
        # Gaussian power inside a circular stop of radius a for 1/e^2 radius w:
        fraction = 1.0 - np.exp(-2.0 * (allowed / spot_mm) ** 2)
        print(f"   {spot_mm:>11.1f} {fraction:>16.1%} "
              f"{-10*np.log10(max(fraction,1e-12)):>11.1f}")
    print("   The loss is a SCALAR aperture acting equally on p and s, so it cancels in rho and")
    print("   costs SNR rather than accuracy -- PROVIDED the reference is truncated identically.")
    print("   => make the gold reference the SAME SIZE as the sample, in the same mount.")
    print("   A reference mirror larger than the sample turns a cancelling term into a smooth,")
    print("   frequency-dependent amplitude tilt -- exactly the artefact signature we chase.")


def main():
    validate()
    print_divergence_invariant()
    print_divergence_correction_factor()
    print_divergence_cost()
    print_footprint_tradeoff()
    print_divergence_knowledge_requirement()
    print_fixed_footprint_budget()
    print_corrected_budget()
    print_energy_containment()
    print_clipping_is_self_defeating()
    print_focus_instead()
    print_crossover_comparison()
    print_blur_budget_vs_angle()
    print_tilt_delay()
    print_tilt_null_procedure()
    print_truncation_consequences()
    print_angle_from_reference()
    print_joint_angle_fit()
    print_tilt_self_measurement()
    print("\ndone.")


if __name__ == "__main__":
    main()
