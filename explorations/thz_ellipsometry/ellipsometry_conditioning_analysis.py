"""Quantitative evaluation of reflection ellipsometry versus referenced reflectometry
for highly conductive, poorly conditioned samples (CNT paper).

The question this answers: for a near-mirror sample, does measuring the ellipsometric
ratio rho = r_p / r_s buy enough sensitivity to pay for the loss of an absolute
reference, and how do the two approaches respond to the error channels that actually
dominate our measurements (sample height / timing drift, sample tilt, channel gain)?

Sign convention follows Chen & Pickwell-MacPherson, APL Photonics 7, 071101 (2022):
complex index written as N = n - i*kappa with kappa > 0 for a passive medium, fields
propagating as exp(i(omega t - k z)).  Every function here is pure and takes its
physics by argument so the pieces can be unit tested in isolation.
"""

from __future__ import annotations

import numpy as np

VACUUM_PERMITTIVITY = 8.8541878128e-12
SPEED_OF_LIGHT = 2.99792458e8


# ---------------------------------------------------------------------------
# Material models
# ---------------------------------------------------------------------------

def drude_permittivity(frequency_hz, dc_conductivity_si, scattering_time_s,
                       epsilon_infinity):
    """Drude permittivity in the N = n - i*kappa convention (Im(eps) < 0)."""
    angular_frequency = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    scattering_rate = 1.0 / scattering_time_s
    plasma_frequency_squared = dc_conductivity_si / (VACUUM_PERMITTIVITY * scattering_time_s)
    return epsilon_infinity - plasma_frequency_squared / (
        angular_frequency**2 - 1j * angular_frequency * scattering_rate
    )


def passive_index_from_permittivity(permittivity):
    """sqrt(eps) on the branch with Im(N) <= 0, i.e. N = n - i*kappa, kappa >= 0."""
    index = np.sqrt(np.asarray(permittivity, dtype=complex))
    return np.where(index.imag > 0.0, -index, index)


def real_conductivity_from_index(index, frequency_hz):
    """sigma_1 in S/m for the N = n - i*kappa convention."""
    angular_frequency = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    permittivity = index**2
    return -permittivity.imag * angular_frequency * VACUUM_PERMITTIVITY


# ---------------------------------------------------------------------------
# Fresnel reflection
# ---------------------------------------------------------------------------

def transmitted_cosine(index_incident, index_sample, incidence_angle_rad):
    sin_transmitted = index_incident * np.sin(incidence_angle_rad) / index_sample
    return np.sqrt(1.0 - sin_transmitted**2)


def fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad):
    cos_incident = np.cos(incidence_angle_rad)
    cos_transmitted = transmitted_cosine(index_incident, index_sample, incidence_angle_rad)
    return ((index_incident * cos_incident - index_sample * cos_transmitted)
            / (index_incident * cos_incident + index_sample * cos_transmitted))


def fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad):
    cos_incident = np.cos(incidence_angle_rad)
    cos_transmitted = transmitted_cosine(index_incident, index_sample, incidence_angle_rad)
    return ((index_sample * cos_incident - index_incident * cos_transmitted)
            / (index_sample * cos_incident + index_incident * cos_transmitted))


def ellipsometric_ratio(index_incident, index_sample, incidence_angle_rad):
    return (fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad)
            / fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad))


def index_from_ellipsometric_ratio(ratio, index_incident, incidence_angle_rad):
    """Closed-form single-interface inversion (exact for a bare, isotropic surface)."""
    sin_squared = np.sin(incidence_angle_rad) ** 2
    tan_squared = np.tan(incidence_angle_rad) ** 2
    permittivity = index_incident**2 * sin_squared * (
        1.0 + tan_squared * ((1.0 - ratio) / (1.0 + ratio)) ** 2
    )
    return passive_index_from_permittivity(permittivity)


# ---------------------------------------------------------------------------
# Sensitivity: how a relative error in the observable maps onto the index
# ---------------------------------------------------------------------------

def _index_jacobian(observable_fn, index_sample, relative_step=1e-6):
    """d(observable)/d(n) and d(observable)/d(kappa) by central differences."""
    step_n = relative_step * max(abs(index_sample.real), 1.0)
    step_k = relative_step * max(abs(index_sample.imag), 1.0)
    d_dn = (observable_fn(index_sample + step_n) - observable_fn(index_sample - step_n)) / (2 * step_n)
    d_dk = (observable_fn(index_sample - 1j * step_k) - observable_fn(index_sample + 1j * step_k)) / (2 * step_k)
    return d_dn, d_dk


def index_error_amplification(observable_fn, index_sample):
    """Worst-case |dN| produced by a unit *relative* error in the observable.

    Builds the 2x2 real Jacobian of (Re O, Im O) with respect to (n, kappa),
    scales it by |O| so the input is a relative observable error, and returns the
    largest singular value of the inverse: the amplification factor from a 1
    fractional error in the observable to an absolute error in the index.
    """
    observable = observable_fn(index_sample)
    d_dn, d_dk = _index_jacobian(observable_fn, index_sample)
    jacobian = np.array([[d_dn.real, d_dk.real],
                         [d_dn.imag, d_dk.imag]]) / abs(observable)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    smallest = singular_values.min()
    if smallest <= 0.0:
        return np.inf
    return 1.0 / smallest


def observable_builders(index_incident, incidence_angle_rad):
    """The three competing observables, as callables of the sample index."""
    return {
        "r_s (referenced)": lambda N: fresnel_reflection_s(index_incident, N, incidence_angle_rad),
        "r_p (referenced)": lambda N: fresnel_reflection_p(index_incident, N, incidence_angle_rad),
        "rho = r_p/r_s": lambda N: ellipsometric_ratio(index_incident, N, incidence_angle_rad),
    }


# ---------------------------------------------------------------------------
# Error channels
# ---------------------------------------------------------------------------

def index_error_from_timing_offset(index_sample, index_incident, incidence_angle_rad,
                                   frequency_hz, timing_offset_s):
    """Index error a sample height / timing error injects into REFERENCED reflectometry.

    A displacement of the sample surface along its normal, or any drift of the
    reference timing, multiplies the measured reflection coefficient by
    exp(i*omega*dt).  It multiplies r_p and r_s by the *same* factor, so it cancels
    identically in rho -- that cancellation is the whole point and is asserted in
    validate_self_consistency().
    """
    phase = np.exp(1j * 2.0 * np.pi * frequency_hz * timing_offset_s)
    true_r_s = fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad)
    corrupted = true_r_s * phase
    recovered = _invert_r_s(corrupted, index_incident, incidence_angle_rad)
    return abs(recovered - index_sample)


def _invert_r_s(reflection_s, index_incident, incidence_angle_rad):
    """Invert the s-polarised Fresnel coefficient for the sample index."""
    cos_incident = np.cos(incidence_angle_rad)
    # r_s = (n1 c1 - N c2)/(n1 c1 + N c2)  ->  N c2 = n1 c1 (1-r)/(1+r)
    n_cos_transmitted = index_incident * cos_incident * (1.0 - reflection_s) / (1.0 + reflection_s)
    permittivity = n_cos_transmitted**2 + (index_incident * np.sin(incidence_angle_rad)) ** 2
    return passive_index_from_permittivity(permittivity)


def index_error_from_tilt(index_sample, index_incident, incidence_angle_rad, tilt_rad):
    """Index error a sample tilt injects into ELLIPSOMETRY (rho measured, wrong angle assumed)."""
    measured = ellipsometric_ratio(index_incident, index_sample, incidence_angle_rad + tilt_rad)
    recovered = index_from_ellipsometric_ratio(measured, index_incident, incidence_angle_rad)
    return abs(recovered - index_sample)


def index_error_from_channel_gain(index_sample, index_incident, incidence_angle_rad,
                                  relative_gain_error):
    """Index error from a mis-calibrated p/s detection channel ratio."""
    measured = ellipsometric_ratio(index_incident, index_sample, incidence_angle_rad)
    recovered = index_from_ellipsometric_ratio(measured * (1.0 + relative_gain_error),
                                               index_incident, incidence_angle_rad)
    return abs(recovered - index_sample)


# ---------------------------------------------------------------------------
# Validation (known answers)
# ---------------------------------------------------------------------------

def validate_self_consistency(verbose=True):
    """Known-answer checks that must hold before any number below is believed."""
    failures = []
    angle = np.deg2rad(70.0)
    for index_sample in (3.418 - 0.0j, 6.96 - 8.10j, 1.5 - 0.02j):
        ratio = ellipsometric_ratio(1.0, index_sample, angle)
        recovered = index_from_ellipsometric_ratio(ratio, 1.0, angle)
        if abs(recovered - index_sample) > 1e-9:
            failures.append(f"rho round-trip failed for {index_sample}: {recovered}")

    # A common-mode phase factor must cancel exactly in rho.
    index_sample = 6.96 - 8.10j
    phase = np.exp(1j * 0.37)
    r_p = fresnel_reflection_p(1.0, index_sample, angle) * phase
    r_s = fresnel_reflection_s(1.0, index_sample, angle) * phase
    if abs(r_p / r_s - ellipsometric_ratio(1.0, index_sample, angle)) > 1e-14:
        failures.append("common-mode phase did not cancel in rho")

    # Perfect conductor limit: rho -> -1.
    if abs(ellipsometric_ratio(1.0, 1e6 - 1e6j, angle) + 1.0) > 1e-4:
        failures.append("rho did not approach -1 for a perfect conductor")

    if verbose:
        print("validation:", "PASS (3 known-answer checks)" if not failures else "FAIL")
        for failure in failures:
            print("   ", failure)
    return failures


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CNT_PAPER_MODEL = dict(
    dc_conductivity_si=6500.0,      # 65 S/cm, tuned to reproduce our measured sigma_1
    scattering_time_s=30e-15,
    epsilon_infinity=4.0,
)

HIGH_RESISTIVITY_SILICON = 3.418 - 0.0j


def sample_index(frequency_hz, model=CNT_PAPER_MODEL):
    return passive_index_from_permittivity(drude_permittivity(frequency_hz, **model))


def print_material_model(frequencies_hz):
    print("\n[material] Drude CNT-paper surrogate "
          f"(sigma_dc={CNT_PAPER_MODEL['dc_conductivity_si']/100:.0f} S/cm, "
          f"tau={CNT_PAPER_MODEL['scattering_time_s']*1e15:.0f} fs, "
          f"eps_inf={CNT_PAPER_MODEL['epsilon_infinity']:.1f})")
    print(f"   {'f [THz]':>8} {'n':>8} {'kappa':>8} {'|N|':>8} {'sigma_1 [S/cm]':>15}")
    for frequency in frequencies_hz:
        index = sample_index(frequency)
        conductivity = real_conductivity_from_index(index, frequency) / 100.0
        print(f"   {frequency/1e12:>8.2f} {index.real:>8.2f} {-index.imag:>8.2f} "
              f"{abs(index):>8.2f} {conductivity:>15.1f}")


def print_conditioning_vs_angle(frequency_hz, angles_deg, index_incident=1.0):
    index = sample_index(frequency_hz)
    print(f"\n[conditioning] index error per 1% relative observable error, "
          f"{frequency_hz/1e12:.1f} THz, N = {index.real:.2f} - {-index.imag:.2f}i, "
          f"incident medium n1 = {index_incident:.2f}")
    print(f"   {'theta':>7} {'|1+r_s|':>9} {'tanPsi':>8} {'Delta':>8} "
          f"{'dN r_s':>9} {'dN r_p':>9} {'dN rho':>9} {'rho/r_s':>9}")
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        builders = observable_builders(index_incident, angle)
        amplification = {name: index_error_amplification(fn, index) * 0.01
                         for name, fn in builders.items()}
        ratio = ellipsometric_ratio(index_incident, index, angle)
        pole_distance = abs(1.0 + fresnel_reflection_s(index_incident, index, angle))
        gain = amplification["r_s (referenced)"] / amplification["rho = r_p/r_s"]
        print(f"   {angle_deg:>6.0f} {pole_distance:>9.3f} {abs(ratio):>8.3f} "
              f"{np.rad2deg(np.angle(ratio)):>8.1f} "
              f"{amplification['r_s (referenced)']:>9.3f} "
              f"{amplification['r_p (referenced)']:>9.3f} "
              f"{amplification['rho = r_p/r_s']:>9.3f} {gain:>9.1f}x")


def print_error_channels(frequency_hz, angles_deg, timing_offsets_fs=(1.5, 19.0),
                         tilt_degrees=(0.05, 0.2, 0.5), gain_errors=(0.01, 0.05),
                         index_incident=1.0):
    index = sample_index(frequency_hz)
    print(f"\n[error channels] absolute index error |dN| at {frequency_hz/1e12:.1f} THz "
          f"(true N = {index.real:.2f} - {-index.imag:.2f}i)")
    header = (f"   {'theta':>7}" + "".join(f"{f'ref dt={t:g}fs':>14}" for t in timing_offsets_fs)
              + "".join(f"{f'ell tilt={d:g}deg':>16}" for d in tilt_degrees)
              + "".join(f"{f'ell gain {g:.0%}':>14}" for g in gain_errors))
    print(header)
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        row = f"   {angle_deg:>6.0f}"
        for timing_fs in timing_offsets_fs:
            row += f"{index_error_from_timing_offset(index, index_incident, angle, frequency_hz, timing_fs * 1e-15):>14.3f}"
        for tilt_deg in tilt_degrees:
            row += f"{index_error_from_tilt(index, index_incident, angle, np.deg2rad(tilt_deg)):>16.3f}"
        for gain in gain_errors:
            row += f"{index_error_from_channel_gain(index, index_incident, angle, gain):>14.3f}"
        print(row)


def print_silicon_control(angles_deg, index_incident=1.0):
    print(f"\n[silicon control] HR-Si N = {HIGH_RESISTIVITY_SILICON.real:.3f}, "
          "index error per 1% observable error and per 0.2 deg tilt")
    print(f"   {'theta':>7} {'tanPsi':>8} {'Delta':>8} {'dN r_s':>9} {'dN rho':>9} {'tilt 0.2deg':>13}")
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        builders = observable_builders(index_incident, angle)
        ratio = ellipsometric_ratio(index_incident, HIGH_RESISTIVITY_SILICON, angle)
        print(f"   {angle_deg:>6.0f} {abs(ratio):>8.3f} {np.rad2deg(np.angle(ratio)):>8.1f} "
              f"{index_error_amplification(builders['r_s (referenced)'], HIGH_RESISTIVITY_SILICON)*0.01:>9.3f} "
              f"{index_error_amplification(builders['rho = r_p/r_s'], HIGH_RESISTIVITY_SILICON)*0.01:>9.3f} "
              f"{index_error_from_tilt(HIGH_RESISTIVITY_SILICON, index_incident, angle, np.deg2rad(0.2)):>13.3f}")


def print_optimum_angle(frequencies_hz, index_incident=1.0):
    angles = np.deg2rad(np.arange(5.0, 89.5, 0.25))
    print("\n[optimum angle] incidence minimising the ellipsometric index error")
    print(f"   {'f [THz]':>8} {'theta_opt':>10} {'dN per 1%':>11} "
          f"{'theta_opt(r_s)':>15} {'dN per 1%':>11}")
    for frequency in frequencies_hz:
        index = sample_index(frequency)
        best = {}
        for name in ("rho = r_p/r_s", "r_s (referenced)"):
            errors = np.array([
                index_error_amplification(observable_builders(index_incident, a)[name], index) * 0.01
                for a in angles
            ])
            position = int(np.nanargmin(errors))
            best[name] = (np.rad2deg(angles[position]), errors[position])
        print(f"   {frequency/1e12:>8.2f} {best['rho = r_p/r_s'][0]:>10.1f} "
              f"{best['rho = r_p/r_s'][1]:>11.3f} {best['r_s (referenced)'][0]:>15.1f} "
              f"{best['r_s (referenced)'][1]:>11.3f}")


# ---------------------------------------------------------------------------
# Detection geometry: what a <110> electro-optic crystal actually projects
# ---------------------------------------------------------------------------

def electro_optic_detection_vector(probe_angle_rad):
    """Sensitivity of a <110> zincblende EO crystal to the two THz field components.

    Planken et al., JOSA B 18, 313 (2001): the sampled signal is
        S ~ E_001 * sin(2*phi) + 2 * E_perp * cos(2*phi)
    with phi the probe polarisation angle from [001].  The two entries are REAL and
    frequency-independent (the crystal is cubic, so both THz components see the same
    index, absorption and phase matching) -- the detection channel ratio is therefore
    a single real scalar set by the crystal azimuth, not a complex spectral transfer
    function.  Returned as (sensitivity along [001], sensitivity perpendicular).
    """
    return np.array([np.sin(2.0 * probe_angle_rad), 2.0 * np.cos(2.0 * probe_angle_rad)])


def balanced_probe_angle_rad():
    """Azimuth giving equal sensitivity to both THz components: tan(2 phi) = 2."""
    return 0.5 * np.arctan(2.0)


def print_detection_geometry():
    balanced = balanced_probe_angle_rad()
    vector = electro_optic_detection_vector(balanced)
    print("\n[detection] <110> electro-optic crystal azimuth")
    print(f"   balanced azimuth phi = {np.rad2deg(balanced):.2f} deg from [001]")
    print(f"   detection vector d = ({vector[0]:.3f}, {vector[1]:.3f})  -> ratio {vector[0]/vector[1]:.3f}")
    print(f"   single-channel optimum |d| = 2.000 (THz perp [001], probe parallel to it)")
    print(f"   balanced channel amplitude = {vector[0]:.3f} -> "
          f"{100*vector[0]/2.0:.0f}% of peak sensitivity per channel")
    print(f"   {'phi [deg]':>10} {'d_001':>8} {'d_perp':>8} {'|d|':>8} {'ratio':>8}")
    for probe_angle_deg in (0.0, 15.0, 22.5, np.rad2deg(balanced), 35.0, 45.0):
        d = electro_optic_detection_vector(np.deg2rad(probe_angle_deg))
        ratio = d[0] / d[1] if abs(d[1]) > 1e-12 else np.inf
        print(f"   {probe_angle_deg:>10.2f} {d[0]:>8.3f} {d[1]:>8.3f} "
              f"{np.linalg.norm(d):>8.3f} {ratio:>8.3f}")


def print_polarization_mixing_rank():
    """Show that one detection azimuth cannot close generalised ellipsometry."""
    print("\n[polarisation mixing] rank of the measurement with an emitter-only rotation")
    angle = np.deg2rad(70.0)
    index_parallel, index_perpendicular = 7.0 - 8.0j, 3.0 - 1.0j
    jones_diagonal = np.array([
        [fresnel_reflection_p(1.0, index_parallel, angle), 0.0],
        [0.0, fresnel_reflection_s(1.0, index_perpendicular, angle)],
    ])
    optic_axis = np.deg2rad(30.0)
    rotation = np.array([[np.cos(optic_axis), -np.sin(optic_axis)],
                         [np.sin(optic_axis), np.cos(optic_axis)]])
    jones = rotation @ jones_diagonal @ rotation.T

    input_angles = np.deg2rad(np.arange(0.0, 180.0, 10.0))
    for label, probe_angle_deg in (("single azimuth", [31.72]), ("two azimuths", [31.72, 76.72])):
        rows = []
        for probe_angle in probe_angle_deg:
            d = electro_optic_detection_vector(np.deg2rad(probe_angle))
            for alpha in input_angles:
                incident = np.array([np.cos(alpha), np.sin(alpha)])
                rows.append(np.outer(d, incident).ravel())
        design = np.array(rows)
        singular = np.linalg.svd(design, compute_uv=False)
        print(f"   {label:>15}: design-matrix rank {np.linalg.matrix_rank(design, tol=1e-9)}/4, "
              f"singular values {np.array2string(singular, precision=2)}")
    print("   (4 complex Jones entries -> 3 independent ratios; rank 2 cannot recover them)")


def print_roughness_cancellation():
    """The scalar specular-attenuation factor is polarisation independent."""
    print("\n[roughness] scalar Kirchhoff specular attenuation exp(-(4 pi sigma cos(theta)/lambda)^2)")
    angle = np.deg2rad(70.0)
    index = sample_index(1e12)
    wavelength = SPEED_OF_LIGHT / 1e12
    print(f"   {'sigma_rms [um]':>15} {'attenuation':>12} {'|dN| in rho':>12} {'|dN| in r_s':>12}")
    for roughness_um in (1.0, 5.0, 20.0, 50.0):
        attenuation = np.exp(-((4 * np.pi * roughness_um * 1e-6 * np.cos(angle) / wavelength) ** 2))
        # Attenuate each Fresnel coefficient independently, then form the ratio --
        # the cancellation has to emerge, it is not imposed.
        attenuated_p = fresnel_reflection_p(1.0, index, angle) * attenuation
        attenuated_s = fresnel_reflection_s(1.0, index, angle) * attenuation
        ratio_error = abs(index_from_ellipsometric_ratio(attenuated_p / attenuated_s, 1.0, angle) - index)
        reflect_error = abs(_invert_r_s(attenuated_s, 1.0, angle) - index)
        print(f"   {roughness_um:>15.1f} {attenuation:>12.4f} {ratio_error:>12.2e} {reflect_error:>12.3f}")


def index_error_from_beam_divergence(index_sample, index_incident, incidence_angle_rad,
                                     angular_spread_rad, quadrature_points=41):
    """Bias from averaging rho over the angular spread of a focused THz beam.

    A focused beam illuminates the sample with a distribution of incidence angles.
    Because rho is nonlinear in theta the average of rho is not rho of the average,
    and the residual is a bias that no amount of averaging removes.  Modelled as a
    Gaussian angular distribution of standard deviation angular_spread_rad.
    """
    offsets, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
    weights = weights / weights.sum()
    angles = incidence_angle_rad + angular_spread_rad * offsets
    averaged = np.sum(weights * ellipsometric_ratio(index_incident, index_sample, angles))
    recovered = index_from_ellipsometric_ratio(averaged, index_incident, incidence_angle_rad)
    return abs(recovered - index_sample)


def apparent_cross_polarisation(index_sample, index_incident, incidence_angle_rad,
                                azimuth_error_rad):
    """|r_ps / r_ss| produced purely by a rotation of the sample s/p frame.

    An out-of-plane sample tilt rotates the plane of incidence relative to the
    instrument's polarisation basis.  For an isotropic sample this generates
    off-diagonal Jones terms that mimic anisotropy, so it must be fitted as a
    nuisance azimuth rather than assumed zero.
    """
    r_p = fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad)
    r_s = fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad)
    rotation = np.array([[np.cos(azimuth_error_rad), -np.sin(azimuth_error_rad)],
                         [np.sin(azimuth_error_rad), np.cos(azimuth_error_rad)]])
    jones = rotation @ np.array([[r_p, 0.0], [0.0, r_s]]) @ rotation.T
    return abs(jones[0, 1] / jones[1, 1])


def combined_error_budget(index_sample, index_incident, angles_deg,
                          relative_observable_error, angle_uncertainty_deg,
                          channel_gain_error, angular_spread_deg):
    """Quadrature sum of the independent ellipsometric error terms, per angle."""
    rows = []
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        builders = observable_builders(index_incident, angle)
        noise = (index_error_amplification(builders["rho = r_p/r_s"], index_sample)
                 * relative_observable_error)
        tilt = index_error_from_tilt(index_sample, index_incident, angle,
                                     np.deg2rad(angle_uncertainty_deg))
        gain = index_error_from_channel_gain(index_sample, index_incident, angle,
                                             channel_gain_error)
        divergence = index_error_from_beam_divergence(index_sample, index_incident, angle,
                                                      np.deg2rad(angular_spread_deg))
        total = np.sqrt(noise**2 + tilt**2 + gain**2 + divergence**2)
        rows.append((angle_deg, noise, tilt, gain, divergence, total))
    return rows


def print_combined_error_budget(frequency_hz, angles_deg, index_incident=1.0,
                                relative_observable_error=0.005,
                                angle_uncertainty_deg=0.1,
                                channel_gain_error=0.01,
                                angular_spread_deg=1.0):
    index = sample_index(frequency_hz)
    print(f"\n[budget] combined ellipsometric |dN| at {frequency_hz/1e12:.1f} THz "
          f"(|N| = {abs(index):.2f}); noise {relative_observable_error:.1%}, "
          f"tilt {angle_uncertainty_deg} deg, gain {channel_gain_error:.0%}, "
          f"divergence {angular_spread_deg} deg")
    print(f"   {'theta':>7} {'noise':>9} {'tilt':>9} {'gain':>9} {'diverg':>9} "
          f"{'TOTAL':>9} {'rel |N|':>9}")
    rows = combined_error_budget(index, index_incident, angles_deg,
                                 relative_observable_error, angle_uncertainty_deg,
                                 channel_gain_error, angular_spread_deg)
    for angle_deg, noise, tilt, gain, divergence, total in rows:
        print(f"   {angle_deg:>6.0f} {noise:>9.3f} {tilt:>9.3f} {gain:>9.3f} "
              f"{divergence:>9.3f} {total:>9.3f} {total/abs(index):>8.1%}")
    best = min(rows, key=lambda row: row[5])
    print(f"   -> best of the scanned angles: {best[0]:.0f} deg, "
          f"|dN| = {best[5]:.3f} ({best[5]/abs(index):.1%} of |N|)")


def print_azimuth_crosstalk(frequency_hz, incidence_angle_deg=70.0, index_incident=1.0):
    index = sample_index(frequency_hz)
    angle = np.deg2rad(incidence_angle_deg)
    print(f"\n[azimuth] apparent cross-polarisation from an out-of-plane tilt, "
          f"{incidence_angle_deg:.0f} deg, {frequency_hz/1e12:.1f} THz")
    print(f"   {'azimuth error [deg]':>20} {'|r_ps/r_ss|':>13}")
    for azimuth_deg in (0.1, 0.5, 1.0, 2.0):
        print(f"   {azimuth_deg:>20.1f} "
              f"{apparent_cross_polarisation(index, index_incident, angle, np.deg2rad(azimuth_deg)):>13.4f}")



def print_amplitude_versus_phase_error(frequency_hz, angles_deg, index_incident=1.0,
                                       fractional_error=0.01):
    """Split the rho error budget into the amplitude and phase channels.

    This matters because a <110> electro-optic crystal has a REAL detection vector:
    a mis-set channel gain perturbs |rho| only and leaves arg(rho) untouched.  Any
    error that does move arg(rho) has to come from somewhere else (polariser
    retardance, cross-polarisation, angle), which makes Delta the trustworthy half.
    """
    index = sample_index(frequency_hz)
    print(f"\n[amp vs phase] |dN| from a {fractional_error:.0%} error applied to "
          f"|rho| only or arg(rho) only, {frequency_hz/1e12:.1f} THz")
    print(f"   {'theta':>7} {'amplitude-only':>16} {'phase-only':>12} {'phase [deg]':>12}")
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        ratio = ellipsometric_ratio(index_incident, index, angle)
        amplitude_error = abs(index_from_ellipsometric_ratio(
            ratio * (1.0 + fractional_error), index_incident, angle) - index)
        phase_error = abs(index_from_ellipsometric_ratio(
            ratio * np.exp(1j * fractional_error), index_incident, angle) - index)
        print(f"   {angle_deg:>6.0f} {amplitude_error:>16.3f} {phase_error:>12.3f} "
              f"{np.rad2deg(fractional_error):>12.2f}")


def print_beam_footprint(angles_deg, focal_length_mm=150.0, collimated_diameter_mm=30.0,
                         frequencies_hz=(0.5e12, 1.0e12, 2.0e12)):
    """Illuminated footprint along the plane of incidence, the practical angle limit."""
    print(f"\n[footprint] spot and footprint for EFL {focal_length_mm:.0f} mm, "
          f"{collimated_diameter_mm:.0f} mm collimated beam")
    header = f"   {'theta':>7}" + "".join(f"{f'{f/1e12:g} THz [mm]':>14}" for f in frequencies_hz)
    print(header)
    for angle_deg in angles_deg:
        row = f"   {angle_deg:>6.0f}"
        for frequency in frequencies_hz:
            wavelength_mm = SPEED_OF_LIGHT / frequency * 1e3
            waist_mm = wavelength_mm * focal_length_mm / (np.pi * collimated_diameter_mm / 2.0)
            row += f"{2.0 * waist_mm / np.cos(np.deg2rad(angle_deg)):>14.1f}"
        print(row)



# ---------------------------------------------------------------------------
# Eigenvalue calibration: rho survives an unknown instrument Jones matrix
# ---------------------------------------------------------------------------

GOLD_INDEX = 1000.0 - 1000.0j


def instrument_measurement_matrix(index_sample, index_incident, incidence_angle_rad,
                                  detection_matrix, input_matrix,
                                  instrument_in, instrument_out):
    """The 2x2 complex block a two-input, two-projection measurement actually returns."""
    jones = np.array([
        [fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad), 0.0],
        [0.0, fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad)],
    ])
    return detection_matrix @ instrument_out @ jones @ instrument_in @ input_matrix


def ellipsometric_ratio_from_eigenvalues(sample_matrix, reference_matrix,
                                         reference_ratio):
    """Recover rho from measured blocks without knowing the instrument at all.

    Y G^-1 is similar to J_sample J_reference^-1, and a similarity transform leaves
    the eigenvalues alone.  Their ratio is therefore rho_sample / rho_reference no
    matter what unknown gain imbalance, phase or cross-polarisation the emitter
    optics, off-axis parabolas and detector impose -- provided they are identical
    for both measurements, which is exactly what "never realign" guarantees.
    Returns both branch assignments; pick by continuity in frequency.
    """
    eigenvalues = np.linalg.eigvals(sample_matrix @ np.linalg.inv(reference_matrix))
    first, second = eigenvalues
    return (first / second * reference_ratio, second / first * reference_ratio)


def print_eigenvalue_calibration(frequency_hz, angles_deg, index_incident=1.0, seed=7):
    """Known-answer demonstration plus the eigenvalue separation that limits it."""
    generator = np.random.default_rng(seed)
    instrument_in = np.eye(2) + 0.25 * (generator.normal(size=(2, 2))
                                        + 1j * generator.normal(size=(2, 2)))
    instrument_out = np.eye(2) + 0.25 * (generator.normal(size=(2, 2))
                                         + 1j * generator.normal(size=(2, 2)))
    balanced = electro_optic_detection_vector(balanced_probe_angle_rad())
    detection_matrix = np.array([balanced, [balanced[0], -balanced[1]]])
    input_matrix = np.eye(2)
    index = sample_index(frequency_hz)

    print(f"\n[eigenvalue calibration] recovery of rho through a 25% random unknown "
          f"instrument, {frequency_hz/1e12:.1f} THz")
    print(f"   {'theta':>7} {'|rho| true':>11} {'eigen error':>13} "
          f"{'naive error':>13} {'eig separation':>15}")
    for angle_deg in angles_deg:
        angle = np.deg2rad(angle_deg)
        sample_block = instrument_measurement_matrix(
            index, index_incident, angle, detection_matrix, input_matrix,
            instrument_in, instrument_out)
        reference_block = instrument_measurement_matrix(
            GOLD_INDEX, index_incident, angle, detection_matrix, input_matrix,
            instrument_in, instrument_out)
        reference_ratio = ellipsometric_ratio(index_incident, GOLD_INDEX, angle)
        true_ratio = ellipsometric_ratio(index_incident, index, angle)
        candidates = ellipsometric_ratio_from_eigenvalues(
            sample_block, reference_block, reference_ratio)
        eigen_error = min(abs(candidate - true_ratio) for candidate in candidates)
        naive = ((sample_block[0, 0] / sample_block[1, 1])
                 / (reference_block[0, 0] / reference_block[1, 1]) * reference_ratio)
        eigenvalues = np.linalg.eigvals(sample_block @ np.linalg.inv(reference_block))
        separation = abs(eigenvalues[0] - eigenvalues[1]) / max(abs(eigenvalues))
        print(f"   {angle_deg:>6.0f} {abs(true_ratio):>11.3f} {eigen_error:>13.2e} "
              f"{abs(naive - true_ratio):>13.3f} {separation:>15.3f}")
    print("   (eigen separation is the conditioning of the calibration itself: "
          "small separation -> noise-sensitive branch)")


def main():
    frequencies = [0.3e12, 0.5e12, 1.0e12, 2.0e12, 3.0e12]
    angles = [45.0, 55.0, 65.0, 70.0, 75.0, 80.0, 85.0]
    print("=" * 96)
    print("THz ellipsometry vs referenced reflectometry for a highly conductive sample")
    print("=" * 96)
    validate_self_consistency()
    print_material_model(frequencies)
    print_conditioning_vs_angle(1.0e12, angles)
    print_conditioning_vs_angle(0.5e12, angles)
    print_error_channels(1.0e12, angles)
    print_optimum_angle(frequencies)
    print_silicon_control(angles)
    print_detection_geometry()
    print_polarization_mixing_rank()
    print_roughness_cancellation()
    print_azimuth_crosstalk(1.0e12)
    print_combined_error_budget(1.0e12, angles)
    print_combined_error_budget(1.0e12, angles, angular_spread_deg=0.5,
                                angle_uncertainty_deg=0.05)
    print_combined_error_budget(0.5e12, angles)
    print_combined_error_budget(1.0e12, angles, channel_gain_error=0.002,
                                angle_uncertainty_deg=0.05, angular_spread_deg=0.5)
    print_amplitude_versus_phase_error(1.0e12, angles)
    print_beam_footprint(angles)
    print_eigenvalue_calibration(1.0e12, angles)
    print("\ndone.")


if __name__ == "__main__":
    main()
