"""Fit the emitter-polarisation harmonic, and use its redundancy as a drift correction.

At every frequency the measured amplitude of acquisition k MUST have the form

    S_k = [ P cos(alpha_k) + Q sin(alpha_k) + B ] * exp(i w tau_k)

with alpha_k the THz polarisation angle (from p), P = c d_p r_p and Q = c d_s r_s the two
channels whose ratio is the measurement, B an optional non-magnetic background, and tau_k a
timing drift. One row per acquisition: an interleaved acquisition that visits each angle
several times simply contributes several rows at the same angle, and the joint least-squares
fit over every row is the weighted equivalent of the plan's "ratio of averages" (sec. 4.2).

The background term. Optical rectification in the emitter substrate, pump leakage, pickup and
static crystal birefringence do not reverse with the magnet, so they enter every acquisition
identically. The plan removes them by differencing magnet states 180 degrees apart; a constant
column in the fit does the same thing (it is EXACTLY the same for 0/90/180/270), and it also
works for angle sets without reversal pairs.

The drift term. A delay COMMON to every acquisition multiplies P, Q and B by the same phase and
cancels in P/Q; a delay that DIFFERS between acquisitions does not (lab notebook F35), and at
1 THz 1.5 fs of it is ~0.9% in rho. Our timing error is a slow ordered drift, so the default
nuisance is a ONE-parameter ramp. When per-acquisition timestamps are supplied the ramp runs over
real elapsed time, which models an interleaved acquisition exactly; otherwise it runs over
acquisition order, which is the same thing for evenly spaced acquisitions.

Weights. If the per-acquisition spectral noise variance is supplied (from the repeat-scan noise
model), the fit is weighted by it and the returned covariance is the propagated measurement
noise. Otherwise it is unweighted and the covariance is estimated from the residual scatter.

Implementation: the fit is separable. For any trial drift, the problem in (P, Q, B) is linear
and solved in closed form per frequency (variable projection), so the nonlinear search stays at
one or a few parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

__all__ = ["DRIFT_MODELS", "HarmonicFit", "fit_emitter_harmonic", "harmonic_residual_norm"]


@dataclass(frozen=True)
class HarmonicFit:
    """Result of the emitter-angle harmonic fit."""

    channel_p: np.ndarray            #: (n_frequencies,) complex -- d_p * r_p * E_0
    channel_s: np.ndarray            #: (n_frequencies,) complex -- d_s * r_s * E_0
    delays_s: np.ndarray             #: (n_acquisitions,) fitted per-acquisition delay
    residual_per_frequency: np.ndarray   #: (n_frequencies,) relative residual norm
    residual_norm: float             #: aggregate relative residual -- the quality flag
    degrees_of_freedom: int
    drift_model: str
    background: np.ndarray | None = None   #: (n_frequencies,) complex, when fitted
    #: (n_frequencies, n_columns, n_columns) complex covariance of the linear unknowns, in the
    #: column order (P, Q[, B]). Each entry is E[dX dY*] for complex circular errors.
    covariance: np.ndarray | None = None
    #: Weighted residual per degree of freedom. ~1 when the noise model describes the data and
    #: the harmonic model holds; None for an unweighted fit, where it is 1 by construction.
    reduced_chi_square: float | None = None
    elapsed_seconds: np.ndarray | None = None
    emitter_angles_rad: np.ndarray | None = None

    @property
    def channel_ratio(self):
        """P / Q.  Equals (d_p/d_s) * rho -- the calibration removes the leading factor."""
        return self.channel_p / self.channel_s

    @property
    def channel_ratio_variance(self):
        """Var(P/Q) per frequency, propagated to first order from the (P, Q) covariance."""
        if self.covariance is None:
            return None
        gradient_p = 1.0 / self.channel_s
        gradient_q = -self.channel_p / self.channel_s**2
        covariance = self.covariance
        variance = (np.abs(gradient_p) ** 2 * covariance[:, 0, 0].real
                    + np.abs(gradient_q) ** 2 * covariance[:, 1, 1].real
                    + 2.0 * np.real(gradient_p * np.conj(gradient_q) * covariance[:, 0, 1]))
        return np.maximum(variance, 0.0)

    @property
    def channel_signal_to_noise(self):
        """(n_frequencies,) the smaller of |P|/sigma_P and |Q|/sigma_Q, or None."""
        if self.covariance is None:
            return None
        sigma_p = np.sqrt(np.maximum(self.covariance[:, 0, 0].real, 1e-300))
        sigma_q = np.sqrt(np.maximum(self.covariance[:, 1, 1].real, 1e-300))
        return np.minimum(np.abs(self.channel_p) / sigma_p, np.abs(self.channel_s) / sigma_q)

    @property
    def fitted_drift_fs(self):
        return float(np.ptp(self.delays_s) * 1e15)


def _design_matrix(emitter_angles_rad, background_term):
    angles = np.asarray(emitter_angles_rad, dtype=float)
    columns = [np.cos(angles), np.sin(angles)]
    if background_term:
        columns.append(np.ones_like(angles))
    return np.column_stack(columns)


def _ramp_abscissa(acquisition_count, elapsed_seconds):
    """0 at the first acquisition, 1 at the last: in elapsed time when known, else in order."""
    if elapsed_seconds is None:
        return np.arange(acquisition_count, dtype=float) / max(acquisition_count - 1, 1)
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    span = np.ptp(elapsed)
    return (elapsed - elapsed.min()) / span if span > 0 else np.zeros_like(elapsed)


def _delays_from_parameters(parameters, drift_model, acquisition_count, elapsed_seconds=None):
    """Map the nuisance parameters onto a per-acquisition delay vector.

    A delay common to every acquisition is degenerate -- it multiplies P and Q by the same
    phase and cancels in their ratio -- so every model below is gauge-fixed with the first
    acquisition's delay at zero. The ramp parameter is the drift across the whole series, in fs.
    """
    if drift_model == "none":
        return np.zeros(acquisition_count)
    if drift_model == "linear_ramp":
        return parameters[0] * 1e-15 * _ramp_abscissa(acquisition_count, elapsed_seconds)
    if drift_model == "per_acquisition":
        delays = np.zeros(acquisition_count)
        delays[1:] = np.asarray(parameters, dtype=float) * 1e-15
        return delays
    raise ValueError(f"unknown drift_model {drift_model!r}; known: {sorted(DRIFT_MODELS)}")


#: Nuisance parameter count for each drift model, given the number of acquisitions.
DRIFT_MODELS = {
    "none": lambda acquisition_count: 0,
    "linear_ramp": lambda acquisition_count: 1,
    "per_acquisition": lambda acquisition_count: max(acquisition_count - 1, 0),
}


def _solve_linear(design, spectra, frequencies_hz, delays_s, weights):
    """Weighted closed-form solve per frequency for given delays.

    Returns (solution (n_columns, n_frequencies), residual (n_acquisitions, n_frequencies),
    inverse normal matrix (n_frequencies, n_columns, n_columns)).
    """
    phase = np.exp(-1j * 2.0 * np.pi
                   * np.asarray(frequencies_hz)[None, :] * np.asarray(delays_s)[:, None])
    derotated = np.asarray(spectra) * phase
    # normal[f] = A^T diag(w_f) A ; right_hand_side[f] = A^T diag(w_f) b_f
    normal = np.einsum("kf,ki,kj->fij", weights, design, design)
    right_hand_side = np.einsum("kf,ki,kf->fi", weights, design, derotated)
    inverse_normal = np.linalg.inv(normal)
    solution = np.einsum("fij,fj->if", inverse_normal, right_hand_side)
    residual = derotated - design @ solution
    return solution, residual, inverse_normal


def fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                         drift_model="linear_ramp", maximum_drift_fs=500.0, *,
                         elapsed_seconds=None, background_term=False, spectral_variance=None):
    """Fit P and Q (plus an optional background and a drift nuisance) to a polarisation series.

    Parameters
    ----------
    emitter_angles_rad : (n_acquisitions,) array
        THz polarisation angle of each acquisition, from p, in the sample's p/s frame.
        Repeated angles (interleaved cycles) are fine and expected.
    spectra : (n_acquisitions, n_frequencies) complex array
    frequencies_hz : (n_frequencies,) array
    drift_model : {'none', 'linear_ramp', 'per_acquisition'}
    maximum_drift_fs : float
        Bound on the fitted drift, to keep the search well posed.
    elapsed_seconds : (n_acquisitions,) array, optional
        Acquisition times. The ramp runs over these when given, over acquisition order when not.
    background_term : bool
        Fit a polarisation-independent background B alongside P and Q.
    spectral_variance : (n_acquisitions, n_frequencies) array, optional
        E|dS|^2 of each acquisition's spectrum, from the noise model. Weights the fit and makes
        the returned covariance the propagated measurement noise.

    Returns
    -------
    HarmonicFit
    """
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)
    spectra = np.asarray(spectra, dtype=complex)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)

    if spectra.ndim != 2:
        raise ValueError(
            f"spectra must be 2-D (n_acquisitions, n_frequencies); got {spectra.shape}")
    acquisition_count, frequency_count = spectra.shape
    if emitter_angles_rad.size != acquisition_count:
        raise ValueError(
            f"emitter_angles_rad has {emitter_angles_rad.size} entries but spectra has "
            f"{acquisition_count} rows")
    if frequencies_hz.size != frequency_count:
        raise ValueError(
            f"frequencies_hz has {frequencies_hz.size} entries but spectra has "
            f"{frequency_count} columns")
    if elapsed_seconds is not None and np.size(elapsed_seconds) != acquisition_count:
        raise ValueError("elapsed_seconds needs one entry per acquisition")
    if acquisition_count < 2:
        raise ValueError("at least two emitter angles are needed to separate P from Q")

    design = _design_matrix(emitter_angles_rad, background_term)
    column_count = design.shape[1]
    if np.linalg.matrix_rank(design, tol=1e-10) < column_count:
        what = "P, Q and the background" if background_term else "P and Q"
        raise ValueError(
            f"the emitter angles are degenerate; {what} cannot be separated "
            f"(angles: {np.round(np.rad2deg(emitter_angles_rad), 3).tolist()} deg)")

    parameter_count = DRIFT_MODELS[drift_model](acquisition_count)
    # Each acquisition contributes two real numbers per frequency against the 2*n_columns of the
    # linear unknowns, so a nuisance shared across frequencies needs one spare acquisition; a
    # per-acquisition model needs genuine redundancy because its parameter count grows with it.
    minimum_acquisitions = column_count + (1 if parameter_count else 0)
    if drift_model == "per_acquisition":
        minimum_acquisitions = column_count + 3
    if acquisition_count < minimum_acquisitions:
        raise ValueError(
            f"drift_model={drift_model!r}{' with a background term' if background_term else ''} "
            f"needs at least {minimum_acquisitions} acquisitions ({parameter_count} nuisance "
            f"parameters) but only {acquisition_count} were measured; use 'linear_ramp' or "
            "'none', or take more angles")

    if spectral_variance is None:
        weights = np.ones(spectra.shape)
    else:
        spectral_variance = np.asarray(spectral_variance, dtype=float)
        if spectral_variance.shape != spectra.shape:
            raise ValueError(f"spectral_variance {spectral_variance.shape} must match spectra "
                             f"{spectra.shape}")
        if not np.max(spectral_variance) > 0.0:
            raise ValueError("spectral_variance is zero everywhere; omit it for an unweighted "
                             "fit rather than weighting by an infinite precision")
        # Floor zero-variance bins (e.g. a DC bin) so they are heavily but finitely weighted.
        weights = 1.0 / np.maximum(spectral_variance, np.max(spectral_variance) * 1e-12)
    root_weights = np.sqrt(weights)

    if parameter_count == 0:
        delays = np.zeros(acquisition_count)
    else:
        def residual_vector(parameters):
            trial = _delays_from_parameters(parameters, drift_model, acquisition_count,
                                            elapsed_seconds)
            _, residual, _ = _solve_linear(design, spectra, frequencies_hz, trial, weights)
            weighted = residual * root_weights
            return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

        bound = maximum_drift_fs * np.ones(parameter_count)
        result = least_squares(residual_vector, np.zeros(parameter_count),
                               bounds=(-bound, bound), xtol=1e-12, ftol=1e-12)
        delays = _delays_from_parameters(result.x, drift_model, acquisition_count,
                                         elapsed_seconds)

    solution, residual, inverse_normal = _solve_linear(design, spectra, frequencies_hz, delays,
                                                       weights)

    scale = np.linalg.norm(spectra, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    residual_per_frequency = np.linalg.norm(residual, axis=0) / scale
    total_scale = np.linalg.norm(spectra)
    residual_norm = float(np.linalg.norm(residual) / (total_scale if total_scale else 1.0))

    # Complex degrees of freedom per frequency; the shared drift parameters are negligible.
    complex_dof_per_frequency = acquisition_count - column_count
    degrees_of_freedom = 2 * acquisition_count * frequency_count - (
        2 * column_count * frequency_count + parameter_count)
    reduced_chi_square = None
    if spectral_variance is not None:
        covariance = inverse_normal.astype(complex)
        if degrees_of_freedom > 0:
            chi_square = float(np.sum(weights * np.abs(residual) ** 2))
            # E|dS|^2 counts both quadratures, so each complex residual carries 2 real dof.
            reduced_chi_square = 2.0 * chi_square / degrees_of_freedom
    elif complex_dof_per_frequency > 0:
        residual_variance = np.sum(np.abs(residual) ** 2, axis=0) / complex_dof_per_frequency
        covariance = (inverse_normal * residual_variance[:, None, None]).astype(complex)
    else:
        covariance = None

    return HarmonicFit(
        channel_p=solution[0],
        channel_s=solution[1],
        delays_s=delays,
        residual_per_frequency=residual_per_frequency,
        residual_norm=residual_norm,
        degrees_of_freedom=degrees_of_freedom,
        drift_model=drift_model,
        background=solution[2] if background_term else None,
        covariance=covariance,
        reduced_chi_square=reduced_chi_square,
        elapsed_seconds=None if elapsed_seconds is None else np.asarray(elapsed_seconds,
                                                                        dtype=float),
        emitter_angles_rad=emitter_angles_rad,
    )


def harmonic_residual_norm(emitter_angles_rad, spectra, frequencies_hz):
    """Relative residual of the plain two-parameter harmonic -- a reference-free quality flag.

    The signal must be a pure first harmonic in the emitter angle, so whatever is left over is
    instrument error, and its size can be read without knowing what caused it or anything about
    the sample.
    """
    return fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                                drift_model="none").residual_norm
