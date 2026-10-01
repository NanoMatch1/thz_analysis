"""Fit the emitter-polarisation harmonic, and use its redundancy as a drift correction.

At every frequency the measured amplitude MUST be a pure first harmonic in the emitter angle:

    S(alpha) = P cos(alpha) + Q sin(alpha)

Two complex numbers, however finely the angle is sampled.  Taking more than the minimum two
angles therefore buys no new information about the sample -- but it buys the ability to see
anything that does NOT fit that form, and the dominant such thing is a timing drift between
acquisitions at different polarisation settings.

That matters because it is the one first-order error ellipsometry does not remove for free: a
delay COMMON to both polarisations cancels in P/Q, but a delay DIFFERENTIAL between the
acquisitions does not (lab notebook F35).  At 1 THz, 1.5 fs of differential drift is already
~0.9% in rho.

Our timing error is a slow ordered drift rather than independent per-acquisition jitter, so the
default nuisance model is a ONE-PARAMETER linear ramp.  In simulation that is flat at 0.17 index
error from 0 to 200 fs of drift and costs nothing at zero drift, where fitting N-1 free delays
(the published approach) pays a constant penalty of 0.28.

Implementation note: the fit is separable.  For any trial set of delays the remaining problem in
(P, Q) is linear, so the delays are optimised over while (P, Q) are solved in closed form
(variable projection).  That keeps the nonlinear search at one or a few parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

__all__ = ["HarmonicFit", "DRIFT_MODELS", "fit_emitter_harmonic", "harmonic_residual_norm"]


@dataclass(frozen=True)
class HarmonicFit:
    """Result of the emitter-angle harmonic fit."""

    channel_p: np.ndarray            #: (n_frequencies,) complex -- d_p * r_p * E_0
    channel_s: np.ndarray            #: (n_frequencies,) complex -- d_s * r_s * E_0
    delays_s: np.ndarray             #: (n_angles,) fitted per-acquisition delay
    residual_per_frequency: np.ndarray   #: (n_frequencies,) relative residual norm
    residual_norm: float             #: aggregate relative residual -- the quality flag
    degrees_of_freedom: int
    drift_model: str

    @property
    def channel_ratio(self):
        """P / Q.  Equals (d_p/d_s) * rho -- the calibration removes the leading factor."""
        return self.channel_p / self.channel_s


def _design_matrix(emitter_angles_rad):
    angles = np.asarray(emitter_angles_rad, dtype=float)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _delays_from_parameters(parameters, drift_model, angle_count):
    """Map the nuisance parameters onto a per-acquisition delay vector.

    A delay common to every acquisition is degenerate -- it multiplies P and Q by the same
    phase and cancels in their ratio -- so every model below is gauge-fixed with delay[0] = 0.
    """
    if drift_model == "none":
        return np.zeros(angle_count)
    if drift_model == "linear_ramp":
        ramp = np.arange(angle_count, dtype=float) / max(angle_count - 1, 1)
        return parameters[0] * 1e-15 * ramp
    if drift_model == "per_acquisition":
        delays = np.zeros(angle_count)
        delays[1:] = np.asarray(parameters, dtype=float) * 1e-15
        return delays
    raise ValueError(f"unknown drift_model {drift_model!r}; known: {sorted(DRIFT_MODELS)}")


#: Nuisance parameter count for each drift model, given the number of emitter angles.
DRIFT_MODELS = {
    "none": lambda angle_count: 0,
    "linear_ramp": lambda angle_count: 1,
    "per_acquisition": lambda angle_count: max(angle_count - 1, 0),
}


def _solve_channels(design, spectra, frequencies_hz, delays_s):
    """Closed-form (P, Q) for given delays, plus the residual."""
    phase = np.exp(-1j * 2.0 * np.pi
                   * np.asarray(frequencies_hz)[None, :] * np.asarray(delays_s)[:, None])
    derotated = np.asarray(spectra) * phase
    solution, *_ = np.linalg.lstsq(design, derotated, rcond=None)
    residual = derotated - design @ solution
    return solution[0], solution[1], residual


def fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                         drift_model="linear_ramp", maximum_drift_fs=500.0):
    """Fit P and Q (and a drift nuisance) to a polarisation series.

    Parameters
    ----------
    emitter_angles_rad : (n_angles,) array
        Emitter polarisation angles, in the sample's p/s frame.
    spectra : (n_angles, n_frequencies) complex array
        Measured complex amplitude at each angle and frequency.
    frequencies_hz : (n_frequencies,) array
    drift_model : {'none', 'linear_ramp', 'per_acquisition'}
        'linear_ramp' (default) matches our measured error, a slow ordered drift.
    maximum_drift_fs : float
        Bound on the fitted drift, to keep the search well posed.

    Returns
    -------
    HarmonicFit
    """
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)
    spectra = np.asarray(spectra, dtype=complex)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)

    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D (n_angles, n_frequencies); got {spectra.shape}")
    angle_count, frequency_count = spectra.shape
    if emitter_angles_rad.size != angle_count:
        raise ValueError(
            f"emitter_angles_rad has {emitter_angles_rad.size} entries but spectra has "
            f"{angle_count} rows")
    if frequencies_hz.size != frequency_count:
        raise ValueError(
            f"frequencies_hz has {frequencies_hz.size} entries but spectra has "
            f"{frequency_count} columns")
    if angle_count < 2:
        raise ValueError("at least two emitter angles are needed to separate P from Q")

    design = _design_matrix(emitter_angles_rad)
    if np.linalg.matrix_rank(design, tol=1e-10) < 2:
        raise ValueError(
            "the emitter angles are degenerate (all parallel); P and Q cannot be separated")

    parameter_count = DRIFT_MODELS[drift_model](angle_count)
    # Each angle contributes two real numbers per frequency against the four of (P, Q), so a
    # nuisance shared across frequencies needs only one spare angle; a per-acquisition model
    # needs genuine redundancy because its parameter count grows with the angles.
    minimum_angles = 3 if parameter_count else 2
    if drift_model == "per_acquisition":
        minimum_angles = 5
    if angle_count < minimum_angles:
        raise ValueError(
            f"drift_model={drift_model!r} needs at least {minimum_angles} emitter angles "
            f"({parameter_count} nuisance parameters) but only {angle_count} were measured; "
            "use 'linear_ramp' or 'none', or take more angles")

    if parameter_count == 0:
        delays = np.zeros(angle_count)
    else:
        def residual_vector(parameters):
            trial_delays = _delays_from_parameters(parameters, drift_model, angle_count)
            _, _, residual = _solve_channels(design, spectra, frequencies_hz, trial_delays)
            return np.concatenate([residual.real.ravel(), residual.imag.ravel()])

        start = np.zeros(parameter_count)
        bound = maximum_drift_fs
        result = least_squares(residual_vector, start,
                               bounds=(-bound * np.ones(parameter_count),
                                       bound * np.ones(parameter_count)),
                               xtol=1e-12, ftol=1e-12)
        delays = _delays_from_parameters(result.x, drift_model, angle_count)

    channel_p, channel_s, residual = _solve_channels(design, spectra, frequencies_hz, delays)

    scale = np.linalg.norm(spectra, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    residual_per_frequency = np.linalg.norm(residual, axis=0) / scale
    total_scale = np.linalg.norm(spectra)
    residual_norm = float(np.linalg.norm(residual) / (total_scale if total_scale else 1.0))

    return HarmonicFit(
        channel_p=channel_p,
        channel_s=channel_s,
        delays_s=delays,
        residual_per_frequency=residual_per_frequency,
        residual_norm=residual_norm,
        degrees_of_freedom=2 * angle_count * frequency_count
        - (4 * frequency_count + parameter_count),
        drift_model=drift_model,
    )


def harmonic_residual_norm(emitter_angles_rad, spectra, frequencies_hz):
    """Relative residual of the plain two-parameter harmonic -- a reference-free quality flag.

    The signal must be a pure first harmonic in the emitter angle, so whatever is left over is
    instrument error, and its size can be read without knowing what caused it or anything about
    the sample.
    """
    return fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                                drift_model="none").residual_norm
