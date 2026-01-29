"""
Estimate polarization “mixing” caused by out-of-plane two-mirror steering
when the mirror reflection has different complex coefficients for s and p.

Why this model is useful (and what it misses)
---------------------------------------------
- Mixing is NOT s<->p coupling at a single mirror (that stays diagonal in local s/p).
- The “mixing” you care about comes from:
    (1) the plane of incidence changing between reflections (basis rotations)
    (2) r_s != r_p in amplitude and/or phase (mirror behaves like a retarder)
- If r_s == r_p exactly (same complex number), the chain will NOT create ellipticity
  from pure linear polarization — basis rotations alone just rotate the polarization
  with the beam and are mostly benign.

So we MUST include a nonzero differential phase (and/or amplitude) between r_s and r_p.
For “ideal coatings” you can set |r_s|=|r_p|≈1 and choose a small delta_phase.
A dielectric HR mirror at 45° can easily have a few degrees to tens of degrees
of s-p phase difference depending on coating (and it varies with wavelength).

This code:
- models a 3D field vector, reflected by two mirrors
- lets you set an out-of-plane deflection, an in-plane angle, and a small jitter
- reports how much the output polarization changes: leakage into an orthogonal
  linear state and induced ellipticity (Stokes S3).

No broadband here (single wavelength, fixed rs/rp).
"""

import numpy as np

# -----------------------------
# Utility vector helpers
# -----------------------------
def norm(v):
    v = np.asarray(v, dtype=float)
    return np.sqrt(np.dot(v, v))

def unit(v):
    v = np.asarray(v, dtype=float)
    n = norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n

def orthonormal_basis_from_k(k):
    """
    Build two orthonormal transverse basis vectors (ex, ey) perpendicular to k.
    This is a lab-fixed polarization basis tied to the final propagation direction.
    """
    k = unit(k)
    # pick an arbitrary reference that isn't parallel to k
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(k, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(ref, k))
    e2 = unit(np.cross(k, e1))
    return e1, e2

def stokes_from_field(E, e1, e2):
    """
    Compute Stokes parameters in the transverse basis (e1, e2).
    E is a complex 3-vector.
    Returns (S0,S1,S2,S3) with the usual convention.
    """
    a = np.vdot(e1, E)  # complex amplitude along e1
    b = np.vdot(e2, E)  # complex amplitude along e2
    S0 = (abs(a)**2 + abs(b)**2).real
    S1 = (abs(a)**2 - abs(b)**2).real
    S2 = (2 * (a * np.conj(b)).real).real
    S3 = (-2 * (a * np.conj(b)).imag).real  # sign convention; flip if you prefer
    return S0, S1, S2, S3

# -----------------------------
# Fresnel-like reflection in local s/p basis
# -----------------------------
def reflect_field(E_in, k_in, n_hat, r_s, r_p):
    """
    Reflect a complex electric field E_in off a planar interface with normal n_hat.

    Uses the standard decomposition into local s and p:
      - s is perpendicular to plane of incidence
      - p is in plane of incidence and perpendicular to k

    Applies:
      E_s -> r_s E_s
      E_p -> r_p E_p

    Returns (E_out, k_out).
    """
    k_in = unit(k_in)
    n_hat = unit(n_hat)

    # Reflected propagation direction
    k_out = unit(k_in - 2.0 * np.dot(k_in, n_hat) * n_hat)

    # Define plane of incidence using k_in and n_hat
    # s is perpendicular to incidence plane
    s_hat = np.cross(k_in, n_hat)
    if norm(s_hat) < 1e-12:
        raise ValueError("Incidence undefined: k_in parallel to normal")
    s_hat = unit(s_hat)

    # p is in the incidence plane, perpendicular to k_in
    p_hat = unit(np.cross(s_hat, k_in))

    # Project input field onto s/p (ensure it's transverse to k_in)
    E_in = np.asarray(E_in, dtype=complex)
    # Remove any longitudinal component (numerical hygiene)
    E_in = E_in - np.dot(E_in, k_in) * k_in

    E_s = np.vdot(s_hat, E_in)
    E_p = np.vdot(p_hat, E_in)

    # Apply reflection coefficients
    E_s_out = r_s * E_s
    E_p_out = r_p * E_p

    # Rebuild outgoing field in outgoing transverse basis.
    # For reflection, the s direction stays perpendicular to incidence plane;
    # p direction defined with k_out
    p_hat_out = unit(np.cross(s_hat, k_out))

    E_out = E_s_out * s_hat + E_p_out * p_hat_out
    # Remove any residual longitudinal component w.r.t k_out
    E_out = E_out - np.dot(E_out, k_out) * k_out

    return E_out, k_out

# -----------------------------
# Geometry helpers for "two-mirror out-of-plane jog"
# -----------------------------
def rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])

def mirror_normal_from_dirs(k_in, k_out):
    """
    For specular reflection, the surface normal bisects k_in and k_out:
      n ∥ (k_in - k_out) or (k_in + k_out) depending on convention.
    Using: k_out = k_in - 2 (k_in·n) n
    A consistent choice is n ∥ (k_in - k_out).
    """
    k_in = unit(k_in)
    k_out = unit(k_out)
    n = k_in - k_out
    return unit(n)

# -----------------------------
# Main simulation
# -----------------------------
def simulate(
    out_of_plane_deflect_deg=2.0,
    in_plane_yaw_deg=0.0,
    mirror1_pitch_jitter_urad=0.0,
    mirror1_yaw_jitter_urad=0.0,
    input_beam_yaw_jitter_urad=0.0,
    input_beam_pitch_jitter_urad=0.0,
    delta_phase_deg=10.0,
    amp_mismatch=0.0,
    input_pol_angle_deg=0.0,
):
    """
    Parameters
    ----------
    out_of_plane_deflect_deg:
        Nominal out-of-plane deflection between mirror 1 and 2 (sets k between mirrors).
    in_plane_yaw_deg:
        Optional in-plane yaw for the between-mirror direction (keeps things more general).
    mirror1_pitch_jitter_urad, mirror1_yaw_jitter_urad:
        Small angular perturbation of mirror 1 normal (models vibration).
    input_beam_yaw_jitter_urad, input_beam_pitch_jitter_urad:
        Small angular perturbation of incoming beam direction.
    delta_phase_deg:
        Phase difference between r_p and r_s: r_p = exp(i*delta_phase), r_s = 1.
        (Set 0 for a truly polarization-neutral mirror → you’ll see almost no “mixing”.)
    amp_mismatch:
        Optional amplitude mismatch between |r_p| and |r_s|:
          r_s amplitude = 1
          r_p amplitude = 1 - amp_mismatch
        (e.g. amp_mismatch=0.01 means 1% lower p reflectivity)
    input_pol_angle_deg:
        Linear polarization angle in the FINAL transverse basis (e1,e2):
        0° = along e1, 90° = along e2.
        (For a horizontal beam along +x, e1 defaults ~y and e2 ~z.)
    """

    # Convert small angles
    urad = 1e-6
    d_out = np.deg2rad(out_of_plane_deflect_deg)
    yaw = np.deg2rad(in_plane_yaw_deg)

    # Incoming beam: nominal along +x
    k0 = np.array([1.0, 0.0, 0.0])
    # Apply incoming beam jitter (yaw about z, pitch about y)
    k0 = rot_z(input_beam_yaw_jitter_urad * urad) @ (rot_y(input_beam_pitch_jitter_urad * urad) @ k0)
    k0 = unit(k0)

    # Define the between-mirror direction k1 by pitching down by d_out, and yawing in-plane if desired
    # Start with a pitch down in x-z plane, then yaw about z
    k1 = rot_z(yaw) @ (rot_y(-d_out) @ np.array([1.0, 0.0, 0.0]))
    k1 = unit(k1)

    # We want mirror2 to send k1 back to k2 ~ k0 direction (parallel to incoming)
    k2 = unit(np.array([1.0, 0.0, 0.0]))  # target output direction (nominal)
    # If you want it exactly to match the jittered k0, swap to k2=k0.
    k2 = unit(np.array([1.0, 0.0, 0.0]))

    # Mirror normals (nominal)
    n1 = mirror_normal_from_dirs(k0, k1)
    n2 = mirror_normal_from_dirs(k1, k2)

    # Apply mirror1 normal jitter (yaw about z, pitch about y) as a small rotation of the normal
    n1 = rot_z(mirror1_yaw_jitter_urad * urad) @ (rot_y(mirror1_pitch_jitter_urad * urad) @ n1)
    n1 = unit(n1)

    # Reflection coefficients (simple model)
    delta = np.deg2rad(delta_phase_deg)
    r_s = 1.0 + 0j
    r_p = (1.0 - amp_mismatch) * np.exp(1j * delta)

    # Input polarization: define in transverse basis of k0
    e1_in, e2_in = orthonormal_basis_from_k(k0)
    ang = np.deg2rad(input_pol_angle_deg)
    # Linear polarization at angle ang in (e1_in,e2_in)
    E0 = np.cos(ang) * e1_in + np.sin(ang) * e2_in
    E0 = E0.astype(complex)

    # Propagate through reflections
    E_after_1, k_after_1 = reflect_field(E0, k0, n1, r_s, r_p)
    E_after_2, k_after_2 = reflect_field(E_after_1, k_after_1, n2, r_s, r_p)

    # Evaluate output polarization in a transverse basis tied to k2 (nominal +x)
    e1_out, e2_out = orthonormal_basis_from_k(k_after_2)
    S0, S1, S2, S3 = stokes_from_field(E_after_2, e1_out, e2_out)

    # Define “leakage” into orthogonal linear polarization relative to *best-fit* linear basis:
    # For a quick scalar: compute degree of circular polarization = S3/S0
    docp = 0.0 if S0 == 0 else S3 / S0

    # Another useful metric: how far from perfectly linear is it?
    # Degree of linear polarization (for a fully coherent field it should stay 1, but ellipticity changes)
    dolp = 0.0 if S0 == 0 else np.sqrt(S1**2 + S2**2) / S0

    # Ellipticity angle chi: sin(2chi) = S3/S0
    # chi=0 linear, chi=±45° circular
    chi = 0.5 * np.arcsin(np.clip(docp, -1.0, 1.0))

    return {
        "k_in": k0,
        "k_after_1": k_after_1,
        "k_after_2": k_after_2,
        "stokes": (S0, S1, S2, S3),
        "degree of leakage into orthogonal linear polarization": docp,
        "degree of linear polarization": dolp,
        "ellipticity_angle_deg": np.rad2deg(chi),
    }

def sweep_mirror1_pitch_jitter():
    """
    Example sweep: how ellipticity changes vs mirror1 pitch jitter
    for a chosen out-of-plane jog and chosen mirror s–p phase difference.
    """
    params = dict(
        out_of_plane_deflect_deg=2.0,
        delta_phase_deg=10.0,     # try 0, 5, 10, 20...
        amp_mismatch=0.0,
        input_pol_angle_deg=45.0, # worst-case mixed state in many geometries
    )

    jitters_urad = np.linspace(-200, 200, 41)  # +/-200 urad
    results = []
    for j in jitters_urad:
        out = simulate(mirror1_pitch_jitter_urad=j, **params)
        results.append((j, out["ellipticity_angle_deg"], out["DoCP"]))

    print("pitch_jitter_urad, ellipticity_angle_deg, DoCP")
    for j, chi_deg, docp in results:
        print(f"{j: .1f}, {chi_deg: .6f}, {docp: .6e}")

if __name__ == "__main__":
    # Quick single run (edit these)
    out = simulate(
        out_of_plane_deflect_deg=2.0,
        delta_phase_deg=10.0,       # set to 0 to see why nothing happens if rs==rp
        input_pol_angle_deg=45.0,   # mixed state most sensitive
        mirror1_pitch_jitter_urad=50.0,
    )
    print(out)

    # Uncomment to run a sweep
    # sweep_mirror1_pitch_jitter()
