"""Reflection Jones matrix of a semi-infinite anisotropic medium, and what a
polarisation series can and cannot recover from it.

The practical case for pressed CNT paper: the surface normal is a principal axis of the
dielectric tensor (fibres lie in the plane), so the tensor in the lab frame is

    eps_lab = R_z(azimuth) . diag(eps_1, eps_2, eps_3) . R_z(-azimuth)

with eps_xz = eps_zx = eps_yz = eps_zy = 0.  For that class the Berreman quartic in the
normal wavevector factorises into a quadratic in q**2, which is solved in closed form here --
no root-finding, no quartic conditioning problems.

Convention matches ellipsometry_conditioning_analysis: N = n - i*kappa, passive media have
Im(eps) < 0, and a forward-decaying mode has Im(q) < 0.
"""

from __future__ import annotations

import numpy as np

from ellipsometry_conditioning_analysis import (
    fresnel_reflection_p,
    fresnel_reflection_s,
    passive_index_from_permittivity,
)


def rotate_tensor_about_normal(principal_permittivities, azimuth_rad):
    """Build the lab-frame tensor from principal values and an in-plane azimuth.

    principal_permittivities = (eps_1, eps_2, eps_3) with eps_3 along the surface normal.
    azimuth_rad is the angle from the plane of incidence (x axis) to principal axis 1.
    """
    eps_1, eps_2, eps_3 = principal_permittivities
    diagonal = np.diag([eps_1, eps_2, eps_3]).astype(complex)
    cosine, sine = np.cos(azimuth_rad), np.sin(azimuth_rad)
    rotation = np.array([[cosine, -sine, 0.0],
                         [sine, cosine, 0.0],
                         [0.0, 0.0, 1.0]], dtype=complex)
    return rotation @ diagonal @ rotation.T


def _normal_wavevectors(permittivity_tensor, reduced_in_plane):
    """The two forward normal wavevectors q, from the quadratic in u = q**2.

    det M(q) = 0 reduces, for eps_xz = eps_yz = 0, to
        eps_zz*u**2 - [(eps_zz - xi**2)(eps_xx + eps_yy - xi**2) + xi**2 (eps_yy - xi**2)]*u
                    + (eps_zz - xi**2)[eps_xx (eps_yy - xi**2) - eps_xy**2] = 0
    The isotropic limit collapses it to the double root u = eps - xi**2, which is the
    known-answer check in validate_against_fresnel().
    """
    eps = permittivity_tensor
    xi_squared = reduced_in_plane**2
    eps_xx, eps_yy, eps_zz, eps_xy = eps[0, 0], eps[1, 1], eps[2, 2], eps[0, 1]

    quadratic = eps_zz
    linear = -((eps_zz - xi_squared) * (eps_xx + eps_yy - xi_squared)
               + xi_squared * (eps_yy - xi_squared))
    constant = (eps_zz - xi_squared) * (eps_xx * (eps_yy - xi_squared) - eps_xy**2)

    discriminant = np.sqrt(linear**2 - 4.0 * quadratic * constant)
    roots = np.array([(-linear + discriminant) / (2.0 * quadratic),
                      (-linear - discriminant) / (2.0 * quadratic)])
    wavevectors = np.sqrt(roots)
    # Forward branch: decaying into the sample, or propagating away from the interface.
    forward = np.where(wavevectors.imag > 0.0, -wavevectors, wavevectors)
    return np.where((forward.imag == 0.0) & (forward.real < 0.0), -forward, forward)


def _wave_matrix(permittivity_tensor, reduced_in_plane, wavevector):
    eps = permittivity_tensor
    xi, q = reduced_in_plane, wavevector
    return np.array([
        [eps[0, 0] - q**2, eps[0, 1], eps[0, 2] + xi * q],
        [eps[1, 0], eps[1, 1] - xi**2 - q**2, eps[1, 2]],
        [eps[2, 0] + xi * q, eps[2, 1], eps[2, 2] - xi**2],
    ], dtype=complex)


DEGENERACY_TOLERANCE = 1e-7


def _mode_fields(permittivity_tensor, reduced_in_plane, wavevectors):
    """Electric fields of the two forward modes, as null vectors of M(q).

    The isotropic (and uniaxial-along-normal) limit is DEGENERATE: both roots collapse to
    q**2 = eps - xi**2 and the null space of M(q) becomes two-dimensional.  Taking one null
    vector per root then returns the same vector twice and the boundary-matching system is
    singular.  Detect that case and take both null directions from the single M(q).
    """
    first, second = wavevectors
    scale = max(abs(first), abs(second), 1.0)
    if abs(first - second) <= DEGENERACY_TOLERANCE * scale:
        _, _, right_vectors = np.linalg.svd(_wave_matrix(permittivity_tensor,
                                                         reduced_in_plane, first))
        return [right_vectors[-1].conj(), right_vectors[-2].conj()], [first, first]
    fields = []
    for wavevector in wavevectors:
        _, _, right_vectors = np.linalg.svd(_wave_matrix(permittivity_tensor,
                                                         reduced_in_plane, wavevector))
        fields.append(right_vectors[-1].conj())
    return fields, list(wavevectors)


def _magnetic_field(reduced_wavevector, electric_field):
    """h = k_hat x E, up to the constant common to both media (cancels in matching)."""
    return np.cross(reduced_wavevector, electric_field)


def anisotropic_reflection_jones(permittivity_tensor, index_incident, incidence_angle_rad):
    """2x2 reflection Jones matrix [[r_pp, r_ps], [r_sp, r_ss]].

    Column order is (p_in, s_in); row order is (p_out, s_out), so r_ps is the p output
    produced by an s input.  p is defined with the same sign convention as
    fresnel_reflection_p, which validate_against_fresnel() enforces.
    """
    xi = index_incident * np.sin(incidence_angle_rad)
    q_incident = np.sqrt(index_incident**2 - xi**2)

    incident_k = np.array([xi, 0.0, q_incident], dtype=complex)
    reflected_k = np.array([xi, 0.0, -q_incident], dtype=complex)

    field_s = np.array([0.0, 1.0, 0.0], dtype=complex)
    incident_p = np.array([q_incident, 0.0, -xi], dtype=complex) / index_incident
    # Sign chosen so that r_pp = -r_ss at normal incidence, matching the Fresnel
    # convention used by fresnel_reflection_p (enforced by validate_against_fresnel).
    reflected_p = -np.array([q_incident, 0.0, xi], dtype=complex) / index_incident

    def tangential(electric, wavevector):
        magnetic = _magnetic_field(wavevector, electric)
        return np.array([electric[0], electric[1], magnetic[0], magnetic[1]])

    transmitted_q = _normal_wavevectors(permittivity_tensor, xi)
    mode_fields, mode_q = _mode_fields(permittivity_tensor, xi, transmitted_q)
    transmitted_columns = [
        tangential(electric, np.array([xi, 0.0, wavevector], dtype=complex))
        for electric, wavevector in zip(mode_fields, mode_q)
    ]

    # Unknowns: [r_p, r_s, t_1, t_2];  incident + reflected = transmitted.
    system = np.column_stack([
        -tangential(reflected_p, reflected_k),
        -tangential(field_s, reflected_k),
        transmitted_columns[0],
        transmitted_columns[1],
    ])

    jones = np.zeros((2, 2), dtype=complex)
    for column, incident_field in enumerate((incident_p, field_s)):
        solution = np.linalg.solve(system, tangential(incident_field, incident_k))
        jones[0, column], jones[1, column] = solution[0], solution[1]
    return jones


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_against_fresnel(verbose=True):
    """The anisotropic solver must reproduce scalar Fresnel exactly."""
    failures = []
    for index_sample in (3.418 - 0.0j, 6.96 - 8.11j, 1.5 - 0.02j, 30.0 - 25.0j):
        tensor = np.eye(3, dtype=complex) * index_sample**2
        for angle_deg in (15.0, 45.0, 70.0, 85.0):
            angle = np.deg2rad(angle_deg)
            jones = anisotropic_reflection_jones(tensor, 1.0, angle)
            expected_p = fresnel_reflection_p(1.0, index_sample, angle)
            expected_s = fresnel_reflection_s(1.0, index_sample, angle)
            if abs(jones[0, 0] - expected_p) > 1e-8:
                failures.append(f"r_pp {index_sample} @{angle_deg}: {jones[0,0]} vs {expected_p}")
            if abs(jones[1, 1] - expected_s) > 1e-8:
                failures.append(f"r_ss {index_sample} @{angle_deg}: {jones[1,1]} vs {expected_s}")
            if max(abs(jones[0, 1]), abs(jones[1, 0])) > 1e-10:
                failures.append(f"isotropic cross-pol not zero @{angle_deg}")

    # Uniaxial with the optic axis ALONG the normal: in-plane isotropic -> still no cross-pol.
    tensor = np.diag([(6.96 - 8.11j)**2, (6.96 - 8.11j)**2, (3.0 - 1.0j)**2]).astype(complex)
    jones = anisotropic_reflection_jones(tensor, 1.0, np.deg2rad(70.0))
    if max(abs(jones[0, 1]), abs(jones[1, 0])) > 1e-10:
        failures.append("uniaxial-along-normal produced cross-polarisation")

    # In-plane anisotropy aligned to the incidence plane -> still no cross-pol.
    tensor = rotate_tensor_about_normal(((7.0 - 8.0j)**2, (3.0 - 1.0j)**2, (4.0 - 2.0j)**2), 0.0)
    jones = anisotropic_reflection_jones(tensor, 1.0, np.deg2rad(70.0))
    if max(abs(jones[0, 1]), abs(jones[1, 0])) > 1e-10:
        failures.append("in-plane axis along s/p produced cross-polarisation")

    # Reciprocity: r_ps = r_sp for a non-magnetic, non-gyrotropic medium.
    tensor = rotate_tensor_about_normal(((7.0 - 8.0j)**2, (3.0 - 1.0j)**2, (4.0 - 2.0j)**2),
                                        np.deg2rad(37.0))
    jones = anisotropic_reflection_jones(tensor, 1.0, np.deg2rad(70.0))
    if abs(abs(jones[0, 1]) - abs(jones[1, 0])) > 1e-8:
        failures.append(f"reciprocity violated: {jones[0,1]} vs {jones[1,0]}")

    if verbose:
        print("validation:", "PASS (5 known-answer checks)" if not failures else "FAIL")
        for failure in failures:
            print("   ", failure)
    return failures


# ---------------------------------------------------------------------------
# What a polarisation series can recover: sample rotation vs detection rotation
# ---------------------------------------------------------------------------

from ellipsometry_conditioning_analysis import (  # noqa: E402
    drude_permittivity, electro_optic_detection_vector, balanced_probe_angle_rad,
)

CNT_ANISOTROPIC_MODEL = dict(
    conductivity_along_fibre_si=6500.0,      # 65 S/cm
    conductivity_across_fibre_si=1625.0,     # 4x anisotropy, mid of our measured 2-7x
    conductivity_out_of_plane_si=650.0,      # layered paper: worst direction
    scattering_time_s=30e-15,
    epsilon_infinity=4.0,
)


def anisotropic_permittivities(frequency_hz, model=CNT_ANISOTROPIC_MODEL):
    """(eps_along_fibre, eps_across_fibre, eps_normal) at one frequency."""
    return tuple(
        drude_permittivity(frequency_hz, conductivity,
                           model["scattering_time_s"], model["epsilon_infinity"])
        for conductivity in (model["conductivity_along_fibre_si"],
                             model["conductivity_across_fibre_si"],
                             model["conductivity_out_of_plane_si"])
    )


def normalised_jones_observables(jones):
    """The 2 independent complex ratios an ellipsometer can measure: (r_pp/r_ss, r_ps/r_ss).

    r_sp carries no extra information -- reciprocity ties it to r_ps -- so a single angle of
    incidence yields FOUR real numbers per frequency, not six.
    """
    return np.array([jones[0, 0] / jones[1, 1], jones[0, 1] / jones[1, 1]])


def print_cross_polarisation_vs_azimuth(frequency_hz, incidence_angle_deg=70.0):
    print(f"\n[cross-pol vs sample azimuth] {frequency_hz/1e12:.1f} THz, "
          f"{incidence_angle_deg:.0f} deg incidence")
    print(f"   {'azimuth [deg]':>14} {'|r_ps/r_ss|':>13} {'|r_pp/r_ss|':>13}")
    angle = np.deg2rad(incidence_angle_deg)
    for azimuth_deg in (0.0, 10.0, 22.5, 30.0, 45.0, 60.0, 80.0, 90.0):
        tensor = rotate_tensor_about_normal(anisotropic_permittivities(frequency_hz),
                                            np.deg2rad(azimuth_deg))
        jones = anisotropic_reflection_jones(tensor, 1.0, angle)
        ratios = normalised_jones_observables(jones)
        print(f"   {azimuth_deg:>14.1f} {abs(ratios[1]):>13.4f} {abs(ratios[0]):>13.4f}")
    print("   -> cross-polarisation peaks near 45 deg and VANISHES at 0/90:")
    print("      mount the fibre axis at ~45 deg or the anisotropy is invisible from one mount.")


def _observable_vector(parameters, frequencies_hz, incidence_angle_rad):
    """Stacked real observables for a Drude-parameterised anisotropic sample."""
    (conductivity_1, conductivity_2, conductivity_3,
     scattering_time, azimuth) = parameters
    values = []
    for frequency in frequencies_hz:
        principal = tuple(
            drude_permittivity(frequency, conductivity, scattering_time,
                               CNT_ANISOTROPIC_MODEL["epsilon_infinity"])
            for conductivity in (conductivity_1, conductivity_2, conductivity_3))
        tensor = rotate_tensor_about_normal(principal, azimuth)
        ratios = normalised_jones_observables(
            anisotropic_reflection_jones(tensor, 1.0, incidence_angle_rad))
        values.extend([ratios[0].real, ratios[0].imag, ratios[1].real, ratios[1].imag])
    return np.array(values)


def print_identifiability(frequency_hz, incidence_angle_deg=70.0, azimuth_deg=45.0):
    """Rank of the single-frequency, single-angle measurement against a free tensor."""
    angle = np.deg2rad(incidence_angle_deg)
    azimuth = np.deg2rad(azimuth_deg)
    principal = np.array(anisotropic_permittivities(frequency_hz))

    def observables(free_parameters):
        values = free_parameters[:6].reshape(3, 2)
        tensor = rotate_tensor_about_normal(values[:, 0] + 1j * values[:, 1],
                                            free_parameters[6])
        ratios = normalised_jones_observables(
            anisotropic_reflection_jones(tensor, 1.0, angle))
        return np.array([ratios[0].real, ratios[0].imag, ratios[1].real, ratios[1].imag])

    base = np.concatenate([np.column_stack([principal.real, principal.imag]).ravel(),
                           [azimuth]])
    jacobian = np.zeros((4, 7))
    for column in range(7):
        step = 1e-6 * max(abs(base[column]), 1.0)
        shifted_up, shifted_down = base.copy(), base.copy()
        shifted_up[column] += step
        shifted_down[column] -= step
        jacobian[:, column] = (observables(shifted_up) - observables(shifted_down)) / (2 * step)

    singular = np.linalg.svd(jacobian, compute_uv=False)
    print(f"\n[identifiability] one frequency, one incidence angle, free biaxial tensor")
    print(f"   observables 4 (r_pp/r_ss and r_ps/r_ss, real+imag; r_sp is NOT independent)")
    print(f"   unknowns    7 (3 complex principal eps + azimuth)")
    print(f"   Jacobian singular values: {np.array2string(singular, precision=3)}")
    print(f"   rank {np.linalg.matrix_rank(jacobian, tol=1e-8)} -> "
          f"UNDER-DETERMINED per frequency by construction.")
    normal_column = np.linalg.norm(jacobian[:, 4:6])
    in_plane_column = np.linalg.norm(jacobian[:, 0:4])
    print(f"   sensitivity to the OUT-OF-PLANE eps is {normal_column/in_plane_column:.3f}x "
          f"that of the in-plane pair -> weakly constrained at this angle.")


def print_fit_recovery(frequencies_hz, incidence_angle_deg=70.0, azimuth_deg=45.0,
                       relative_noise=0.005, trials=12, seed=11):
    """Can a shared Drude model close the problem across frequency? Monte Carlo."""
    from scipy.optimize import least_squares

    angle = np.deg2rad(incidence_angle_deg)
    truth = np.array([CNT_ANISOTROPIC_MODEL["conductivity_along_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_across_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_out_of_plane_si"],
                      CNT_ANISOTROPIC_MODEL["scattering_time_s"],
                      np.deg2rad(azimuth_deg)])
    clean = _observable_vector(truth, frequencies_hz, angle)
    generator = np.random.default_rng(seed)
    scale = np.array([1e4, 1e4, 1e4, 1e-14, 1.0])

    recovered = []
    for _ in range(trials):
        noisy = clean + relative_noise * np.abs(clean).mean() * generator.normal(size=clean.size)
        start = truth / scale * generator.uniform(0.5, 1.6, size=5)
        result = least_squares(
            lambda scaled: _observable_vector(scaled * scale, frequencies_hz, angle) - noisy,
            start, bounds=([1e2/1e4, 1e2/1e4, 1e1/1e4, 1e-15/1e-14, 0.0],
                           [1e6/1e4, 1e6/1e4, 1e6/1e4, 1e-12/1e-14, np.pi]),
            xtol=1e-12, ftol=1e-12)
        recovered.append(result.x * scale)
    recovered = np.array(recovered)

    print(f"\n[fit recovery] shared Drude across {len(frequencies_hz)} frequencies, "
          f"{relative_noise:.1%} noise, {trials} trials, single mount at {azimuth_deg:.0f} deg")
    labels = ["sigma_along [S/cm]", "sigma_across [S/cm]", "sigma_normal [S/cm]",
              "tau [fs]", "azimuth [deg]"]
    display = np.column_stack([recovered[:, 0] / 100, recovered[:, 1] / 100,
                               recovered[:, 2] / 100, recovered[:, 3] * 1e15,
                               np.rad2deg(recovered[:, 4])])
    true_display = [truth[0] / 100, truth[1] / 100, truth[2] / 100,
                    truth[3] * 1e15, np.rad2deg(truth[4])]
    print(f"   {'parameter':>20} {'true':>10} {'median':>10} {'1-sigma':>10} {'bias':>10}")
    for position, label in enumerate(labels):
        column = display[:, position]
        median = np.median(column)
        spread = 0.5 * (np.percentile(column, 84) - np.percentile(column, 16))
        print(f"   {label:>20} {true_display[position]:>10.2f} {median:>10.2f} "
              f"{spread:>10.2f} {median - true_display[position]:>+10.2f}")



def _observable_vector_for_scheme(parameters, frequencies_hz, incidence_angle_rad, scheme):
    """Observables actually available under a given measurement scheme.

    'generalized_one_mount' : r_pp/r_ss AND r_ps/r_ss at a single 45 deg mount.
                              Needs two detection projections; the sample never moves.
    'standard_one_mount'    : r_pp/r_ss only (one detection projection) at 45 deg.
    'standard_two_mounts'   : r_pp/r_ss only, at azimuth 0 and 90 -- the classical
                              "rotate the sample" route, which re-presses the contact.
    """
    conductivity_1, conductivity_2, conductivity_3, scattering_time, azimuth = parameters
    if scheme == "standard_two_mounts":
        mounts, use_cross = (0.0, np.pi / 2), False
    elif scheme == "standard_one_mount":
        mounts, use_cross = (azimuth,), False
    else:
        mounts, use_cross = (azimuth,), True

    values = []
    for mount_azimuth in mounts:
        for frequency in frequencies_hz:
            principal = tuple(
                drude_permittivity(frequency, conductivity, scattering_time,
                                   CNT_ANISOTROPIC_MODEL["epsilon_infinity"])
                for conductivity in (conductivity_1, conductivity_2, conductivity_3))
            tensor = rotate_tensor_about_normal(principal, mount_azimuth)
            ratios = normalised_jones_observables(
                anisotropic_reflection_jones(tensor, 1.0, incidence_angle_rad))
            values.extend([ratios[0].real, ratios[0].imag])
            if use_cross:
                values.extend([ratios[1].real, ratios[1].imag])
    return np.array(values)


def print_scheme_comparison(frequencies_hz, incidence_angle_deg=70.0, azimuth_deg=45.0,
                            relative_noise=0.005, trials=12, seed=23):
    """Polarisation series + 2 detection azimuths vs physically rotating the sample."""
    from scipy.optimize import least_squares

    angle = np.deg2rad(incidence_angle_deg)
    truth = np.array([CNT_ANISOTROPIC_MODEL["conductivity_along_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_across_fibre_si"],
                      CNT_ANISOTROPIC_MODEL["conductivity_out_of_plane_si"],
                      CNT_ANISOTROPIC_MODEL["scattering_time_s"],
                      np.deg2rad(azimuth_deg)])
    scale = np.array([1e4, 1e4, 1e4, 1e-14, 1.0])
    print(f"\n[scheme comparison] {relative_noise:.1%} noise, {trials} trials, "
          f"{len(frequencies_hz)} frequencies, {incidence_angle_deg:.0f} deg")
    print(f"   {'scheme':>22} {'sig_along':>20} {'sig_across':>20} {'ratio':>16} {'azimuth':>14}")
    for scheme in ("generalized_one_mount", "standard_one_mount", "standard_two_mounts"):
        clean = _observable_vector_for_scheme(truth, frequencies_hz, angle, scheme)
        generator = np.random.default_rng(seed)
        recovered = []
        for _ in range(trials):
            noisy = clean + relative_noise * np.abs(clean).mean() * generator.normal(size=clean.size)
            start = truth / scale * generator.uniform(0.6, 1.5, size=5)
            result = least_squares(
                lambda scaled: _observable_vector_for_scheme(
                    scaled * scale, frequencies_hz, angle, scheme) - noisy,
                start, bounds=([1e-2, 1e-2, 1e-3, 0.1, 0.0], [1e2, 1e2, 1e2, 100.0, np.pi]),
                xtol=1e-12, ftol=1e-12)
            recovered.append(result.x * scale)
        recovered = np.array(recovered)

        def summarise(column, factor=1.0):
            values = recovered[:, column] * factor
            return np.median(values), 0.5 * (np.percentile(values, 84) - np.percentile(values, 16))

        along = summarise(0, 1 / 100)
        across = summarise(1, 1 / 100)
        ratios = recovered[:, 0] / recovered[:, 1]
        azimuths = np.rad2deg(recovered[:, 4])
        print(f"   {scheme:>22} {along[0]:>10.1f} +/-{along[1]:<7.1f} "
              f"{across[0]:>10.1f} +/-{across[1]:<7.1f} "
              f"{np.median(ratios):>8.2f} +/-{0.5*(np.percentile(ratios,84)-np.percentile(ratios,16)):<5.2f} "
              f"{np.median(azimuths):>7.1f} +/-{0.5*(np.percentile(azimuths,84)-np.percentile(azimuths,16)):<5.1f}")
    print("   true: sigma_along 65.0, sigma_across 16.2, ratio 4.00, azimuth 45.0")
    print("   NOTE standard_two_mounts fits azimuth only as a dummy (mounts are pinned to 0/90),")
    print("        and it costs a re-press of the sample between the two datasets.")


def main():
    validate_against_fresnel()
    print_cross_polarisation_vs_azimuth(1.0e12)
    print_identifiability(1.0e12)
    print_fit_recovery(np.linspace(0.6e12, 3.0e12, 13))
    print_scheme_comparison(np.linspace(0.6e12, 3.0e12, 13))
    print("\ndone.")


if __name__ == "__main__":
    main()
