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

The amplitude term. A laser-power drift, or the broadband part of a purge transient, scales a
whole acquisition. Like a delay, a scale COMMON to every acquisition cancels and one that varies
between them does not, so it gets the same treatment: an optional one-parameter ramp a_k = 1 +
g x_k (or a free value per acquisition), fitted alongside the delay. It is well determined even
for four magnet states (a planted 2% ramp is recovered as 2.01%).

What CANNOT be fitted here: an error in the polarisation angle of individual states. With four
states and a background, each frequency has one complex number of redundancy, and per-state
angle errors have no distinct signature in it -- singular-value ratio ~1e-8 on gold AND on doped
silicon. The angles must come from a calibration (the wire-grid nulls), not from the sample.

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

__all__ = ["AMPLITUDE_MODELS", "DRIFT_MODELS", "HarmonicFit", "fit_emitter_harmonic",
           "harmonic_residual_norm", "revisit_lever_arm", "settling_profile"]


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
    amplitudes: np.ndarray | None = None   #: (n_acquisitions,) fitted scale at 1 THz, first = 1
    #: (n_acquisitions,) fitted d ln(scale)/df per THz, for amplitude_model='tilt_ramp'
    amplitude_tilts_per_thz: np.ndarray | None = None
    amplitude_model: str = "none"
    #: Standard errors of the nuisance parameters in fit order (delay terms in fs, then the
    #: amplitude terms); None when there are none. Large = not separable from P/Q.
    nuisance_parameter_errors: np.ndarray | None = None
    nuisance_parameter_names: tuple = ()

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

    @property
    def fitted_amplitude_change(self):
        """Largest fractional change in acquisition scale across the series (0 if not fitted)."""
        return 0.0 if self.amplitudes is None else float(np.ptp(self.amplitudes))

    @property
    def fitted_tilt_change_per_thz(self):
        """Change in spectral tilt (d ln a / df, per THz) across the series; 0 if not fitted."""
        return (0.0 if self.amplitude_tilts_per_thz is None
                else float(np.ptp(self.amplitude_tilts_per_thz)))


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
    if drift_model == "settling":
        return parameters[0] * 1e-15 * settling_profile(
            _ramp_abscissa(acquisition_count, elapsed_seconds), parameters[1])
    if drift_model == "per_acquisition":
        delays = np.zeros(acquisition_count)
        delays[1:] = np.asarray(parameters, dtype=float) * 1e-15
        return delays
    raise ValueError(f"unknown drift_model {drift_model!r}; known: {sorted(DRIFT_MODELS)}")


def settling_profile(abscissa, rate):
    """0 -> 1 over the block along an exponential settle; ``rate`` = block length / tau.

    (1 - exp(-rate x)) / (1 - exp(-rate)): rate -> 0 is the straight line, rate > 0 a transient
    that is slowing down (a purge settling), rate < 0 one speeding up. This is the EXACT shape of
    a single purge transient, so -- unlike a quadratic -- it leaves no model misfit for other
    terms to absorb. That matters: with a background column, a quadratic's small misfit can leak
    into P/Q through a near-degeneracy that opens for some channel ratios (F40).
    """
    abscissa = np.asarray(abscissa, dtype=float)
    if abs(rate) < 1e-6:
        return abscissa
    return np.expm1(-rate * abscissa) / np.expm1(-rate)


#: drift model -> the names of its parameters for a given number of acquisitions. The one place
#: a drift model is declared; its parameter count is len(names).
DRIFT_MODELS = {
    "none": lambda count: (),
    "linear_ramp": lambda count: ("delay_span_fs",),
    # A purge settling after a disturbance is an exponential with a tau that depends on the
    # disturbance (13-40 min measured); the rate is fitted. Not a quadratic: its small misfit
    # leaks into P/Q through the background when C ~ -1 (F40). purge_spectral_template.py.
    "settling": lambda count: ("delay_span_fs", "settling_rate"),
    "per_acquisition": lambda count: tuple(f"delay_{k}_fs" for k in range(1, count)),
}


#: Frequency about which the spectral tilt is defined, so the gain term means "scale at 1 THz".
TILT_REFERENCE_HZ = 1.0e12


def _amplitudes_from_parameters(parameters, amplitude_model, acquisition_count,
                                elapsed_seconds=None, frequencies_hz=None):
    """Per-acquisition scale, (n_acquisitions, n_frequencies), first acquisition = 1.

    'tilt_ramp' is the purge model: water removal raises the amplitude more at high frequency,
    and on a short record (unresolved lines) that shows as a log-amplitude TILT, linear in f to
    0.3% (purge_spectral_template.py, August purge data): ln a_k(f) = x_k (g + t (f - 1 THz)).
    """
    columns = 1 if frequencies_hz is None else np.size(frequencies_hz)
    if amplitude_model == "none":
        return np.ones((acquisition_count, columns))
    abscissa = _ramp_abscissa(acquisition_count, elapsed_seconds)[:, None]
    if amplitude_model == "linear_ramp":
        return np.repeat(1.0 + parameters[0] * abscissa, columns, axis=1)
    if amplitude_model in ("tilt_ramp", "tilt_settling"):
        offset_thz = (np.asarray(frequencies_hz, dtype=float) - TILT_REFERENCE_HZ)[None, :] / 1e12
        profile = (abscissa if amplitude_model == "tilt_ramp"
                   else settling_profile(abscissa, parameters[2]))
        return np.exp(profile * (parameters[0] + parameters[1] * offset_thz))
    if amplitude_model == "per_acquisition":
        values = np.concatenate([[1.0], np.exp(np.asarray(parameters, dtype=float))])
        return np.repeat(values[:, None], columns, axis=1)
    raise ValueError(f"unknown amplitude_model {amplitude_model!r}; known: "
                     f"{sorted(AMPLITUDE_MODELS)}")


#: amplitude model -> the names of its parameters for a given number of acquisitions.
AMPLITUDE_MODELS = {
    "none": lambda count: (),
    "linear_ramp": lambda count: ("gain",),
    "tilt_ramp": lambda count: ("gain", "tilt_per_thz"),
    # The water part of a purge settles exponentially too (tau ~47 min vs ~38 for the gas), so
    # over a block that starts soon after closing the tilt needs its own settling rate.
    "tilt_settling": lambda count: ("gain", "tilt_per_thz", "tilt_settling_rate"),
    "per_acquisition": lambda count: tuple(f"log_gain_{k}" for k in range(1, count)),
}


def _solve_linear(design, spectra, frequencies_hz, delays_s, weights, amplitudes=None):
    """Weighted closed-form solve per frequency for given delays.

    Returns (solution (n_columns, n_frequencies), residual (n_acquisitions, n_frequencies),
    inverse normal matrix (n_frequencies, n_columns, n_columns)).
    """
    phase = np.exp(-1j * 2.0 * np.pi
                   * np.asarray(frequencies_hz)[None, :] * np.asarray(delays_s)[:, None])
    derotated = np.asarray(spectra) * phase
    if amplitudes is not None:
        # Divide out the scale; the noise is divided with it, so the weights scale by a^2.
        derotated = derotated / amplitudes
        weights = weights * amplitudes**2
    # normal[f] = A^T diag(w_f) A ; right_hand_side[f] = A^T diag(w_f) b_f
    normal = np.einsum("kf,ki,kj->fij", weights, design, design)
    right_hand_side = np.einsum("kf,ki,kf->fi", weights, design, derotated)
    inverse_normal = np.linalg.inv(normal)
    solution = np.einsum("fij,fj->if", inverse_normal, right_hand_side)
    residual = derotated - design @ solution
    if amplitudes is not None:
        residual = residual * amplitudes            # back in measured units
    return solution, residual, inverse_normal


def fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                         drift_model="linear_ramp", maximum_drift_fs=500.0, *,
                         elapsed_seconds=None, background_term=False, spectral_variance=None,
                         amplitude_model="none"):
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
    amplitude_model : {'none', 'linear_ramp', 'per_acquisition'}
        Nuisance scale per acquisition (laser power, broadband purge loss).

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

    if drift_model not in DRIFT_MODELS:
        raise ValueError(f"unknown drift_model {drift_model!r}; known: {sorted(DRIFT_MODELS)}")
    delay_parameter_count = len(DRIFT_MODELS[drift_model](acquisition_count))
    if amplitude_model not in AMPLITUDE_MODELS:
        raise ValueError(f"unknown amplitude_model {amplitude_model!r}; known: "
                         f"{sorted(AMPLITUDE_MODELS)}")
    amplitude_parameter_count = len(AMPLITUDE_MODELS[amplitude_model](acquisition_count))
    parameter_count = delay_parameter_count + amplitude_parameter_count
    # Each acquisition contributes two real numbers per frequency against the 2*n_columns of the
    # linear unknowns, so a nuisance shared across frequencies needs one spare acquisition; a
    # per-acquisition model needs genuine redundancy because its parameter count grows with it.
    minimum_acquisitions = column_count + (1 if parameter_count else 0)
    if "per_acquisition" in (drift_model, amplitude_model):
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

    def nuisance(parameters):
        delays = _delays_from_parameters(parameters[:delay_parameter_count], drift_model,
                                         acquisition_count, elapsed_seconds)
        amplitudes = _amplitudes_from_parameters(parameters[delay_parameter_count:],
                                                 amplitude_model, acquisition_count,
                                                 elapsed_seconds, frequencies_hz)
        return delays, amplitudes

    nuisance_errors = None
    if parameter_count == 0:
        delays = np.zeros(acquisition_count)
        amplitudes = np.ones((acquisition_count, frequency_count))
    else:
        def residual_vector(parameters):
            trial_delays, trial_amplitudes = nuisance(parameters)
            _, residual, _ = _solve_linear(design, spectra, frequencies_hz, trial_delays,
                                           weights, trial_amplitudes)
            weighted = residual * root_weights
            return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

        # Delays in fs within +/- maximum_drift_fs; amplitude parameters within +/- 0.5 (a
        # 50% scale change or e^0.5 is far outside anything physical).
        bound = np.concatenate([maximum_drift_fs * np.ones(delay_parameter_count),
                                0.5 * np.ones(amplitude_parameter_count)])
        start = np.zeros(parameter_count)
        names = (DRIFT_MODELS[drift_model](acquisition_count)
                 + AMPLITUDE_MODELS[amplitude_model](acquisition_count))
        for position, name in enumerate(names):
            if name.endswith("settling_rate"):
                # The rate is block length / tau: +/-10 spans tau from a tenth of the block to
                # infinity. Start at a moderate settle; at rate 0 its gradient vanishes.
                bound[position] = 10.0
                start[position] = 1.0
        result = least_squares(residual_vector, start,
                               bounds=(-bound, bound), xtol=1e-12, ftol=1e-12)
        delays, amplitudes = nuisance(result.x)
        nuisance_errors = _nuisance_standard_errors(result, spectral_variance is not None)

    solution, residual, inverse_normal = _solve_linear(design, spectra, frequencies_hz, delays,
                                                       weights, amplitudes)

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
        amplitudes=(None if amplitude_model == "none"
                    else amplitudes[:, int(np.argmin(np.abs(frequencies_hz
                                                            - TILT_REFERENCE_HZ)))]),
        amplitude_tilts_per_thz=(None if not amplitude_model.startswith("tilt") else np.polyfit(
            frequencies_hz / 1e12, np.log(amplitudes).T, 1)[0]),
        amplitude_model=amplitude_model,
        nuisance_parameter_errors=nuisance_errors,
        nuisance_parameter_names=(DRIFT_MODELS[drift_model](acquisition_count)
                                  + AMPLITUDE_MODELS[amplitude_model](acquisition_count)),
    )




def _nuisance_standard_errors(result, weighted):
    """Standard errors of the drift/amplitude parameters, from the variable-projection Jacobian.

    The Jacobian is of the residual AFTER the linear unknowns (P, Q, B) are re-solved, so a
    nuisance parameter that trades against them -- the case where drift cannot be told apart
    from the channel ratio -- shows up as a large error here. Rank deficiency is reported as
    infinite, never hidden by a pseudo-inverse.
    """
    jacobian = result.jac
    dof = max(result.fun.size - result.x.size, 1)
    # A weighted residual is in units of sigma (reduced chi-square ~1 when the noise model is
    # right); scale by the observed scatter either way so a wrong noise model cannot shrink it.
    scale = float(np.sum(result.fun**2) / dof)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size < result.x.size or singular_values[-1] <= 1e-10 * singular_values[0]:
        return np.full(result.x.size, np.inf)
    covariance = np.linalg.inv(jacobian.T @ jacobian) * scale
    return np.sqrt(np.maximum(np.diag(covariance), 0.0))


def revisit_lever_arm(fit, band=None, similarity_threshold=0.95):
    """How far apart in time the series saw the SAME signal twice, as a fraction of the block.

    A drift is pinned by measuring an identical state at two different times. Compute each
    acquisition's expected signal from the fit (P cos a + Q sin a + B), find pairs that are
    nearly identical (normalised complex correlation above the threshold, same sign), and
    return the largest time separation among them, over the block length. 1.0 = the first and
    last acquisitions repeat each other (best); a small value = only neighbours repeat, and a
    drift between them is indistinguishable from a change of P/Q (F40). Returns 0.0 when no
    pair repeats -- the drift then rests on the frequency structure alone.

    Independent of the noise level, unlike a parameter error, so it reads the same on a quiet
    and a noisy run.
    """
    angles = np.asarray(fit.emitter_angles_rad, dtype=float)
    band = slice(None) if band is None else np.asarray(band, dtype=bool)
    signals = (np.cos(angles)[:, None] * fit.channel_p[None, band]
               + np.sin(angles)[:, None] * fit.channel_s[None, band])
    if fit.background is not None:
        signals = signals + fit.background[None, band]
    norms = np.linalg.norm(signals, axis=1)
    times = (np.arange(angles.size, dtype=float) if fit.elapsed_seconds is None
             else np.asarray(fit.elapsed_seconds, dtype=float))
    span = np.ptp(times) or 1.0
    lever = 0.0
    for first in range(angles.size):
        for second in range(first + 1, angles.size):
            similarity = np.real(np.vdot(signals[first], signals[second])) / (
                norms[first] * norms[second] or 1.0)
            if similarity >= similarity_threshold:
                lever = max(lever, abs(times[second] - times[first]) / span)
    return float(lever)


def harmonic_residual_norm(emitter_angles_rad, spectra, frequencies_hz):
    """Relative residual of the plain two-parameter harmonic -- a reference-free quality flag.

    The signal must be a pure first harmonic in the emitter angle, so whatever is left over is
    instrument error, and its size can be read without knowing what caused it or anything about
    the sample.
    """
    return fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                                drift_model="none").residual_norm
