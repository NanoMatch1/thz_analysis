"""Forward model for THz time-domain ellipsometry.

One definition of the physics, used by the simulator, the fitter and the validator alike, so
there is never a second copy to drift out of step.

Convention throughout the package: the complex refractive index is written ``N = n - i*k`` with
``k >= 0`` for a passive medium, matching ``thz_core`` and the rest of this repo. The Fresnel
primitives themselves are imported from ``thz_core`` rather than reimplemented.

The measurement this package is built around, for emitter polarisation angle ``alpha`` and
angular frequency ``omega``:

    S(alpha, omega) = [ P(omega) cos(alpha) + Q(omega) sin(alpha) ] * exp(i * omega * delay)

with ``P = d_p * r_p * E_0`` and ``Q = d_s * r_s * E_0``.  The emitted spectrum ``E_0``, the beam
path, the detector gain and any timing common to both polarisations all cancel when ``P`` and
``Q`` are divided, which is the entire reason the technique works.
"""

from __future__ import annotations

import numpy as np

import thz_core.thz_core as core

__all__ = [
    "ellipsometric_ratio",
    "index_from_ellipsometric_ratio",
    "measured_amplitude",
    "psi_delta_from_ratio",
    "ratio_from_psi_delta",
    "reflection_coefficients",
]


def reflection_coefficients(index_sample, incidence_angle_rad, index_incident=1.0):
    """(r_p, r_s) for a bare, isotropic, semi-infinite sample."""
    return (
        core.fresnel_reflection_p(index_incident, index_sample, incidence_angle_rad),
        core.fresnel_reflection_s(index_incident, index_sample, incidence_angle_rad),
    )


def ellipsometric_ratio(index_sample, incidence_angle_rad, index_incident=1.0):
    """rho = r_p / r_s."""
    reflection_p, reflection_s = reflection_coefficients(
        index_sample, incidence_angle_rad, index_incident)
    return reflection_p / reflection_s


def index_from_ellipsometric_ratio(ratio, incidence_angle_rad, index_incident=1.0,
                                   reference_index=None):
    """Closed-form inversion of rho for a bare isotropic surface.

        N2 = N1 sin(theta) sqrt( 1 + tan^2(theta) ((1 - rho)/(1 + rho))^2 )

    Exact for a single interface.  The square root has two branches; by default the passive one
    (Im(N) <= 0, i.e. k >= 0) is taken.  Pass ``reference_index`` to select the branch nearest a
    known value instead -- which matters for a nearly lossless sample, where measurement noise
    can push k slightly negative and the passivity rule would then flip the whole index to -N
    and report a spurious error of 2n.
    """
    sine_squared = np.sin(incidence_angle_rad) ** 2
    tangent_squared = np.tan(incidence_angle_rad) ** 2
    permittivity = index_incident**2 * sine_squared * (
        1.0 + tangent_squared * ((1.0 - ratio) / (1.0 + ratio)) ** 2
    )
    root = np.sqrt(np.asarray(permittivity, dtype=complex))
    if reference_index is None:
        return np.where(root.imag > 0.0, -root, root)
    nearest = np.where(np.abs(root - reference_index) <= np.abs(-root - reference_index),
                       root, -root)
    return nearest


def psi_delta_from_ratio(ratio):
    """(tan(Psi), Delta in radians) from rho."""
    ratio = np.asarray(ratio, dtype=complex)
    return np.abs(ratio), np.angle(ratio)


def ratio_from_psi_delta(tan_psi, delta_rad):
    return np.asarray(tan_psi) * np.exp(1j * np.asarray(delta_rad))


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measured_amplitude(emitter_angles_rad, channel_p, channel_s, frequencies_hz,
                       delays_s=None):
    """Complex amplitude for each (emitter angle, frequency).

    Parameters
    ----------
    emitter_angles_rad : (n_angles,) array
    channel_p, channel_s : (n_frequencies,) complex arrays
        P and Q as defined in the module docstring.
    frequencies_hz : (n_frequencies,) array
    delays_s : (n_angles,) array, optional
        Per-acquisition timing offset.  A delay COMMON to every angle cancels in P/Q; only
        the differences between angles matter, and they are a first-order error.

    Returns
    -------
    (n_angles, n_frequencies) complex array
    """
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    harmonic = (np.cos(emitter_angles_rad)[:, None] * np.asarray(channel_p)[None, :]
                + np.sin(emitter_angles_rad)[:, None] * np.asarray(channel_s)[None, :])
    if delays_s is None:
        return harmonic
    phase = np.exp(1j * 2.0 * np.pi
                   * frequencies_hz[None, :] * np.asarray(delays_s, dtype=float)[:, None])
    return harmonic * phase


def channels_from_sample(index_sample, incidence_angle_rad, detection_vector,
                         emitted_spectrum=None, index_incident=1.0):
    """P and Q for a given sample, detection vector and emitted spectrum."""
    reflection_p, reflection_s = reflection_coefficients(
        index_sample, incidence_angle_rad, index_incident)
    spectrum = 1.0 if emitted_spectrum is None else np.asarray(emitted_spectrum)
    return (detection_vector[0] * reflection_p * spectrum,
            detection_vector[1] * reflection_s * spectrum)
