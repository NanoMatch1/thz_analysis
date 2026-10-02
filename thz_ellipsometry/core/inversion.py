"""Invert the ellipsometric ratio for the complex index, and derive the usual quantities.

Two inverters live here. The closed form is exact for a bare isotropic surface illuminated by a
plane wave. The blur-aware one accounts for a converging beam, where each plane-wave component
of the illumination reflects at its own incidence angle.

The blur correction is OPT-IN and requires a MEASURED angular spread. Modelling the average
with a spread known to 10% suppresses the error about fivefold, but a 50%-wrong correction
leaves a larger error than not correcting at all -- so it is better left off than guessed.

Note on the averaging, because it is easy to get wrong: the detector measures each polarisation
coherently, so the angular average happens on the FIELD and the measured quantity is
<r_p>/<r_s>, not <r_p/r_s>.  The 2-D average also has to include the out-of-plane direction,
where the local s/p frame is rotated and r_s leaks into r_pp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import index_from_ellipsometric_ratio, reflection_coefficients

__all__ = [
    "InversionResult",
    "conductivity_from_index",
    "effective_jones_with_divergence",
    "invert_ratio",
    "permittivity_from_index",
]

VACUUM_PERMITTIVITY = 8.8541878128e-12


@dataclass(frozen=True)
class InversionResult:
    frequencies_hz: np.ndarray
    ratio: np.ndarray                 #: (n_frequencies,) complex rho
    index: np.ndarray                 #: (n_frequencies,) complex N = n - i k
    incidence_angle_rad: float
    angular_spread_rad: float         #: 0.0 when no blur correction was applied

    @property
    def refractive_index(self):
        return self.index.real

    @property
    def extinction(self):
        return -self.index.imag

    @property
    def tan_psi(self):
        return np.abs(self.ratio)

    @property
    def delta_rad(self):
        return np.angle(self.ratio)

    @property
    def permittivity(self):
        return permittivity_from_index(self.index)

    @property
    def conductivity_real_si(self):
        return conductivity_from_index(self.index, self.frequencies_hz)


def permittivity_from_index(index):
    return np.asarray(index, dtype=complex) ** 2


def conductivity_from_index(index, frequencies_hz):
    """sigma_1 in S/m, for the N = n - i*k convention where Im(eps) < 0."""
    angular_frequency = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float)
    return -permittivity_from_index(index).imag * angular_frequency * VACUUM_PERMITTIVITY


# ---------------------------------------------------------------------------
# Angular averaging
# ---------------------------------------------------------------------------

def effective_jones_with_divergence(index_sample, incidence_angle_rad, angular_spread_rad,
                                    index_incident=1.0, quadrature_points=11):
    """Angle-averaged reflection Jones matrix for a converging beam.

    A plane-wave component deviating by (a, b) from the central ray -- a in the plane of
    incidence, b out of it -- reflects at incidence angle theta + a with its local s/p frame
    rotated by b / sin(theta).  For a symmetric beam the off-diagonal terms are odd in b and
    cancel, so a clipped or astigmatic beam is the only one that fakes anisotropy.
    """
    if angular_spread_rad <= 0.0:
        reflection_p, reflection_s = reflection_coefficients(
            index_sample, incidence_angle_rad, index_incident)
        return np.array([[reflection_p, 0.0], [0.0, reflection_s]], dtype=complex)

    nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_points)
    weights = weights / weights.sum()
    sine = np.sin(incidence_angle_rad)

    jones = np.zeros((2, 2), dtype=complex)
    for in_plane, weight_x in zip(nodes, weights):
        local_angle = incidence_angle_rad + angular_spread_rad * in_plane
        reflection_p, reflection_s = reflection_coefficients(
            index_sample, local_angle, index_incident)
        diagonal = np.array([[reflection_p, 0.0], [0.0, reflection_s]], dtype=complex)
        for out_of_plane, weight_y in zip(nodes, weights):
            rotation_angle = angular_spread_rad * out_of_plane / sine
            cosine, sine_rotation = np.cos(rotation_angle), np.sin(rotation_angle)
            rotation = np.array([[cosine, -sine_rotation], [sine_rotation, cosine]])
            jones += weight_x * weight_y * (rotation @ diagonal @ rotation.T)
    return jones


def _invert_one_with_blur(ratio, incidence_angle_rad, angular_spread_rad, index_incident,
                          starting_index):
    def residual(values):
        trial = values[0] - 1j * abs(values[1])
        jones = effective_jones_with_divergence(trial, incidence_angle_rad,
                                                angular_spread_rad, index_incident)
        difference = jones[0, 0] / jones[1, 1] - ratio
        return [difference.real, difference.imag]

    result = least_squares(residual, [starting_index.real, -starting_index.imag],
                           xtol=1e-12, ftol=1e-12)
    return result.x[0] - 1j * abs(result.x[1])


def invert_ratio(ratio, frequencies_hz, incidence_angle_rad, index_incident=1.0,
                 angular_spread_rad=0.0, reference_index=None):
    """rho -> complex index, optionally correcting for a converging beam.

    Parameters
    ----------
    reference_index : complex, optional
        Passed to the closed-form inverter to pick the square-root branch nearest a known
        value.  Use it for nearly lossless samples, where noise can push k slightly negative
        and the passivity rule would flip the whole index.
    """
    ratio = np.asarray(ratio, dtype=complex)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    if ratio.shape != frequencies_hz.shape:
        raise ValueError(
            f"ratio {ratio.shape} and frequencies {frequencies_hz.shape} must match")

    closed_form = index_from_ellipsometric_ratio(ratio, incidence_angle_rad, index_incident,
                                                 reference_index=reference_index)

    if angular_spread_rad > 0.0:
        corrected = np.array([
            _invert_one_with_blur(single_ratio, incidence_angle_rad, angular_spread_rad,
                                  index_incident, start)
            for single_ratio, start in zip(ratio, closed_form)
        ])
    else:
        corrected = closed_form

    return InversionResult(
        frequencies_hz=frequencies_hz,
        ratio=ratio,
        index=corrected,
        incidence_angle_rad=float(incidence_angle_rad),
        angular_spread_rad=float(angular_spread_rad),
    )
