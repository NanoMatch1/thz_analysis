"""How errors in rho and in the incidence angle reach the permittivity (plan sec. 8).

Two different kinds of error, kept apart on purpose:

RANDOM -- measurement noise in rho. Propagated per frequency through the inversion's derivative.
Because eps(rho) is analytic, a circular complex error in rho maps to a circular complex error in
N, so the n and k error bars come out equal (sigma_n = sigma_k = |dN/drho| sigma_rho / sqrt 2):
the measurement has isotropic precision in the complex index plane, which is why k is not
intrinsically harder than n, only a smaller number (``silicon_k_sensitivity.py``).

SYSTEMATIC -- a wrong incidence angle. Not random, not reduced by averaging, and NOT detectable
from sample data alone: analysing angle-theta data as theta' gives exactly eps' = A eps + B with
real A and B, which is just as physical as the truth (sec. 8.1). It is reported separately, as
the shift a stated angle uncertainty would cause, never folded into the noise bars.

Conditioning -- d eps/d rho = -4 sin^2 tan^2 (1 - rho)/(1 + rho)^3 grows without bound as
rho -> -1, i.e. for good conductors (sec. 8.2). At 45 degrees that is where metallic samples
become noisy, and the factor is reported per frequency so those bins are flagged, not trusted.
"""

from __future__ import annotations

import numpy as np

from .model import index_from_ellipsometric_ratio

__all__ = [
    "angle_error_affine_coefficients",
    "index_derivative_wrt_angle",
    "index_derivative_wrt_ratio",
    "index_standard_error_from_ratio",
    "permittivity_derivative_wrt_ratio",
]


def permittivity_derivative_wrt_ratio(ratio, incidence_angle_rad, index_incident=1.0):
    """d eps / d rho for the bare isotropic inversion (plan sec. 8.2)."""
    ratio = np.asarray(ratio, dtype=complex)
    sine_squared = np.sin(incidence_angle_rad) ** 2
    tangent_squared = np.tan(incidence_angle_rad) ** 2
    return (-4.0 * index_incident**2 * sine_squared * tangent_squared
            * (1.0 - ratio) / (1.0 + ratio) ** 3)


def index_derivative_wrt_ratio(ratio, incidence_angle_rad, index_incident=1.0, index=None):
    """dN / d rho = (d eps / d rho) / (2 N)."""
    if index is None:
        index = index_from_ellipsometric_ratio(ratio, incidence_angle_rad, index_incident)
    return (permittivity_derivative_wrt_ratio(ratio, incidence_angle_rad, index_incident)
            / (2.0 * np.asarray(index, dtype=complex)))


def index_standard_error_from_ratio(ratio, ratio_variance, incidence_angle_rad,
                                    index_incident=1.0, index=None):
    """Per-frequency standard error of n (= that of k) from Var(rho), to first order."""
    derivative = index_derivative_wrt_ratio(ratio, incidence_angle_rad, index_incident, index)
    return np.abs(derivative) * np.sqrt(np.maximum(np.asarray(ratio_variance, dtype=float),
                                                   0.0) / 2.0)


def index_derivative_wrt_angle(ratio, incidence_angle_rad, index_incident=1.0,
                               step_rad=1e-6, reference_index=None):
    """dN / d theta at fixed rho: how far the recovered index moves per radian of angle error."""
    upper = index_from_ellipsometric_ratio(ratio, incidence_angle_rad + step_rad, index_incident,
                                           reference_index=reference_index)
    lower = index_from_ellipsometric_ratio(ratio, incidence_angle_rad - step_rad, index_incident,
                                           reference_index=reference_index)
    return (upper - lower) / (2.0 * step_rad)


def angle_error_affine_coefficients(true_angle_rad, assumed_angle_rad):
    """(A, B) such that analysing true-angle data at the assumed angle gives eps' = A eps + B.

    Exact, and A, B are real: an angle error rescales and offsets the spectrum without changing
    its shape, so resonance positions and damping survive and oscillator strengths do not.
    Example (plan sec. 8.1): 42 deg data analysed as 45 deg gives A = 1.38, B = -0.12, and
    silicon reads 15.97 instead of 11.68.
    """
    def weight(angle):
        return np.sin(angle) ** 2 * np.tan(angle) ** 2
    scale = weight(assumed_angle_rad) / weight(true_angle_rad)
    offset = np.sin(assumed_angle_rad) ** 2 - scale * np.sin(true_angle_rad) ** 2
    return float(scale), float(offset)
