"""Out-of-plane sample tilt: the one geometric error gold does NOT cancel, and how to fit it.

Self-referencing kills everything that MULTIPLIES both polarisations -- sample height, gain,
roughness loss, truncation -- and nothing that ROTATES them. An out-of-plane tilt psi rotates the
local p/s frame, so the measured channel ratio of a tilted sample is

    m = (C r_pp + r_ps) / (C r_ps + r_ss),     J' = R(psi) diag(r_p, r_s) R(psi)^T

and no division by a gold measurement undoes it (plan sec. 5.1 says placement does not matter;
that is true for the scalars and false for this).

But it can be FITTED, for a dispersive sample (commit 582b43e,
``explorations/thz_ellipsometry/tilt_fitting_removes_remount_problem.py``). To first order
m ~ (C rho + (rho-1) psi)/(1 + C (rho-1) psi): the tilt enters multiplied by (rho - 1), which
varies with frequency for a dispersive sample, while C is flat. So a smooth dispersion model
fitted across the band separates them -- and the model only has to IDENTIFY the tilt: once psi
is known, n and k are recovered per frequency, model-free, by
:func:`invert_with_known_tilt`.

For a FLAT sample (gold, HR-Si) (rho - 1) psi is frequency-flat too, psi is degenerate with C,
and fitting it is meaningless; the fit reports that through its standard error.

Gold's own tilt is absorbed: the fitted psi is the sample's tilt RELATIVE to the gold's, to
first order, which is the quantity that matters.

Dispersion models register themselves with ``@dispersion_model``; each maps a parameter vector
to a complex index in the package convention N = n - i k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import index_from_ellipsometric_ratio, reflection_coefficients

__all__ = [
    "DISPERSION_MODELS",
    "DispersionModel",
    "TiltFit",
    "dispersion_model",
    "fit_tilt_with_dispersion_model",
    "invert_with_known_tilt",
    "tilted_channel_ratio",
]

VACUUM_PERMITTIVITY = 8.8541878128e-12


def tilted_channel_ratio(index_sample, out_of_plane_tilt_rad, channel_ratio,
                         incidence_angle_rad, index_incident=1.0):
    """P/Q the harmonic fit returns for a sample tilted out of plane by psi."""
    reflection_p, reflection_s = reflection_coefficients(index_sample, incidence_angle_rad,
                                                         index_incident)
    cosine, sine = np.cos(out_of_plane_tilt_rad), np.sin(out_of_plane_tilt_rad)
    diagonal_p = reflection_p * cosine**2 + reflection_s * sine**2
    diagonal_s = reflection_p * sine**2 + reflection_s * cosine**2
    cross = (reflection_p - reflection_s) * sine * cosine
    return (channel_ratio * diagonal_p + cross) / (channel_ratio * cross + diagonal_s)


# ---------------------------------------------------------------------------
# Dispersion models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispersionModel:
    name: str
    function: object                  #: (parameters, frequencies_hz) -> complex index
    parameter_names: tuple
    initial: tuple
    lower: tuple
    upper: tuple
    summary: str


DISPERSION_MODELS: dict[str, DispersionModel] = {}


def dispersion_model(*, parameter_names, initial, lower, upper):
    def register(function):
        summary = (function.__doc__ or "").strip().splitlines()
        DISPERSION_MODELS[function.__name__] = DispersionModel(
            name=function.__name__, function=function, parameter_names=tuple(parameter_names),
            initial=tuple(initial), lower=tuple(lower), upper=tuple(upper),
            summary=summary[0] if summary else "")
        return function
    return register


def _passive_root(permittivity):
    index = np.sqrt(np.asarray(permittivity, dtype=complex))
    return np.where(index.imag > 0.0, -index, index)


@dispersion_model(parameter_names=("eps_inf", "log10_resistivity_ohm_cm", "scattering_time_fs"),
                  initial=(11.7, 0.0, 200.0), lower=(1.0, -4.0, 1.0), upper=(30.0, 4.0, 5000.0))
def drude(parameters, frequencies_hz):
    """Drude conductor on a background: eps = eps_inf - wp^2/(w^2 - i w/tau), wp^2 = sigma_dc/(eps0 tau).

    Parametrised by DC resistivity because that is the number a four-point probe gives.
    """
    eps_inf, log10_resistivity, scattering_time_fs = parameters
    angular_frequency = 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float)
    conductivity_si = 100.0 / 10.0 ** log10_resistivity
    scattering_time = scattering_time_fs * 1e-15
    plasma_squared = conductivity_si / (VACUUM_PERMITTIVITY * scattering_time)
    return _passive_root(eps_inf - plasma_squared / (
        angular_frequency**2 - 1j * angular_frequency / scattering_time))


@dispersion_model(parameter_names=("n", "k"), initial=(3.4, 0.0), lower=(1.0, 0.0),
                  upper=(30.0, 30.0))
def constant(parameters, frequencies_hz):
    """Frequency-independent index -- tilt is NOT identifiable with this; it is the control."""
    return np.full(np.shape(frequencies_hz), parameters[0] - 1j * parameters[1], dtype=complex)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TiltFit:
    tilt_rad: float
    tilt_standard_error_rad: float
    model_name: str
    model_parameters: dict
    model_standard_errors: dict
    residual_norm: float

    @property
    def tilt_deg(self):
        return float(np.rad2deg(self.tilt_rad))

    @property
    def tilt_standard_error_deg(self):
        return float(np.rad2deg(self.tilt_standard_error_rad))


def fit_tilt_with_dispersion_model(measured_channel_ratio, channel_ratio, frequencies_hz,
                                   incidence_angle_rad, *, model_name="drude", fixed=None,
                                   initial=None, index_incident=1.0, maximum_tilt_deg=5.0,
                                   ratio_variance=None):
    """Fit (tilt, model parameters) to a sample's calibrated-channel data.

    Parameters
    ----------
    measured_channel_ratio : (n,) complex   P/Q of the sample, already restricted to the band
    channel_ratio : complex                 C from the calibration
    fixed : dict, optional                  model parameters to hold at a given value
    ratio_variance : (n,) array, optional   Var(P/Q), to weight the fit and scale the errors
    """
    try:
        entry = DISPERSION_MODELS[model_name]
    except KeyError:
        raise KeyError(f"unknown dispersion model {model_name!r}; registered: "
                       f"{sorted(DISPERSION_MODELS)}") from None
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(entry.parameter_names)
    if unknown:
        raise ValueError(f"cannot fix {sorted(unknown)}: {model_name} has parameters "
                         f"{list(entry.parameter_names)}")
    free_names = [name for name in entry.parameter_names if name not in fixed]
    start_values = dict(zip(entry.parameter_names, entry.initial))
    start_values.update(initial or {})
    bounds_lower = dict(zip(entry.parameter_names, entry.lower))
    bounds_upper = dict(zip(entry.parameter_names, entry.upper))

    measured = np.asarray(measured_channel_ratio, dtype=complex)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    weights = (np.ones(measured.shape) if ratio_variance is None
               else 1.0 / np.sqrt(np.maximum(np.asarray(ratio_variance, dtype=float), 1e-300)))

    def full_parameters(free_values):
        values = dict(fixed)
        values.update(zip(free_names, free_values))
        return [values[name] for name in entry.parameter_names]

    def residual(vector):
        tilt = np.deg2rad(vector[0])
        index = entry.function(full_parameters(vector[1:]), frequencies_hz)
        difference = (tilted_channel_ratio(index, tilt, channel_ratio, incidence_angle_rad,
                                           index_incident) - measured) * weights
        return np.concatenate([difference.real, difference.imag])

    start = [0.0] + [start_values[name] for name in free_names]
    lower = [-maximum_tilt_deg] + [bounds_lower[name] for name in free_names]
    upper = [maximum_tilt_deg] + [bounds_upper[name] for name in free_names]
    result = least_squares(residual, start, bounds=(lower, upper), xtol=1e-13, ftol=1e-13,
                           x_scale="jac")

    # Standard errors from the Jacobian. Without a noise model, scale by the residual scatter.
    # A rank-deficient Jacobian means the parameters are NOT separately determined -- the flat-
    # sample degeneracy this module's docstring describes -- and a pseudo-inverse would hide that
    # by silently dropping the null direction and reporting a deceptively tight error. So rank
    # deficiency is reported as infinite error instead.
    jacobian = result.jac
    dof = max(result.fun.size - result.x.size, 1)
    scale = 1.0 if ratio_variance is not None else float(np.sum(result.fun**2) / dof)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    determined = (singular_values.size == result.x.size
                  and singular_values[-1] > 1e-8 * singular_values[0])
    if determined:
        covariance = np.linalg.inv(jacobian.T @ jacobian) * scale
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        errors = np.full(result.x.size, np.inf)

    norm = float(np.linalg.norm(np.concatenate([measured.real, measured.imag]) * np.tile(
        weights, 2))) or 1.0
    return TiltFit(
        tilt_rad=float(np.deg2rad(result.x[0])),
        tilt_standard_error_rad=float(np.deg2rad(errors[0])),
        model_name=model_name,
        model_parameters=dict(zip(entry.parameter_names, full_parameters(result.x[1:]))),
        model_standard_errors=dict(zip(free_names, errors[1:].tolist())),
        residual_norm=float(np.linalg.norm(result.fun) / norm),
    )


def invert_with_known_tilt(measured_channel_ratio, channel_ratio, tilt_rad,
                           incidence_angle_rad, *, index_incident=1.0, reference_index=None):
    """Per-frequency index from P/Q with a known tilt removed -- no dispersion model involved.

    Two real equations per frequency for the two real unknowns n and k; started from the
    tilt-free closed form, which is within O(psi) of the answer.
    """
    measured = np.asarray(measured_channel_ratio, dtype=complex)
    starting = index_from_ellipsometric_ratio(measured / channel_ratio, incidence_angle_rad,
                                              index_incident, reference_index=reference_index)
    if tilt_rad == 0.0:
        return starting
    recovered = np.empty_like(starting)
    for position, (value, start) in enumerate(zip(measured, starting)):
        def residual(vector, value=value):
            trial = vector[0] - 1j * vector[1]
            difference = tilted_channel_ratio(trial, tilt_rad, channel_ratio,
                                              incidence_angle_rad, index_incident) - value
            return [difference.real, difference.imag]
        result = least_squares(residual, [start.real, -start.imag], xtol=1e-14, ftol=1e-14)
        recovered[position] = result.x[0] - 1j * result.x[1]
    return recovered
