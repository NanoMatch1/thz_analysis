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

Per-scan rows and box openings. A row need not be a whole acquisition: each repeat scan can be
its own row (``segment_ids`` then says which acquisition, i.e. which magnet state between two
openings of the box, each row belongs to). Inside one acquisition the polarisation is fixed, so
scan-to-scan change there is drift and nothing else -- the drift is measured directly instead of
being inferred from a handful of averages. 'segment_settling' is the model for that layout: a
settling trend over the block (as 'settling') plus, after each opening, a transient A_j exp(-(t - t_j)/T) with its
own amplitude per acquisition and one shared decay time. Its one assumption: after each opening
the delay settles back onto the same trend, so the level jump AT an opening is the transient and
not a permanent step. A palindrome needs no such assumption; the revisit diagnostic still says
whether one was recorded.

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

__all__ = ["AMPLITUDE_MODELS", "DRIFT_MODELS", "HarmonicFit", "SEGMENTED_DRIFT_MODELS",
           "fit_emitter_harmonic", "harmonic_residual_norm", "revisit_lever_arm",
           "segment_transients", "settling_profile"]


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
    #: FFT bins per independent frequency point (zero padding interpolates). The nuisance
    #: errors are already scaled by its square root; ``independent_degrees_of_freedom`` uses it.
    frequency_oversampling: float = 1.0
    #: (n_rows, n_frequencies) the fitted per-row scale (amplitude model), ones without one
    row_scales: np.ndarray | None = None

    def undrifted_spectra(self):
        """(n_rows, n_frequencies) [P cos a + Q sin a (+ B)] x scale: the model without delay."""
        design = _design_matrix(self.emitter_angles_rad, self.background is not None)
        columns = [self.channel_p, self.channel_s] + (
            [] if self.background is None else [self.background])
        scales = 1.0 if self.row_scales is None else self.row_scales
        return (design @ np.vstack(columns)) * scales

    def predicted_spectra(self, frequencies_hz):
        """(n_rows, n_frequencies) what the fitted model says each row should have measured.

        ``undrifted_spectra() x exp(+i 2 pi f tau)``, the convention of the fit itself (it
        derotates each row by exp(-i 2 pi f tau)); with numpy's FFT a positive tau is an EARLIER
        arrival. The measured spectra minus this are the fit residuals. ``frequencies_hz`` must
        be the axis the fit was made on.
        """
        frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        return self.undrifted_spectra() * np.exp(2j * np.pi * frequencies_hz[None, :]
                                                 * np.asarray(self.delays_s)[:, None])
    #: Fitted values of those parameters, same order and units; None when there are none.
    nuisance_parameters: np.ndarray | None = None
    #: (n_rows,) acquisition (between box openings) of each row, for per-scan rows; None when
    #: every row is its own acquisition.
    segment_ids: np.ndarray | None = None

    @property
    def independent_degrees_of_freedom(self):
        """Degrees of freedom counted over independent frequency points, not padded bins.

        The reduced chi-square is unaffected (residual and count shrink together), but its
        spread is set by this: about sqrt(2 / independent_degrees_of_freedom).
        """
        return self.degrees_of_freedom / max(self.frequency_oversampling, 1.0)

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


def _delays_from_parameters(parameters, drift_model, acquisition_count, elapsed_seconds=None,
                            segment_ids=None):
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
    if drift_model == "segment_settling":
        trend = settling_profile(_ramp_abscissa(acquisition_count, elapsed_seconds),
                                 parameters[1])
        delays = (parameters[0] * 1e-15 * trend
                  + segment_transients(parameters[3:], parameters[2], elapsed_seconds,
                                       segment_ids))
        return delays - delays[0]
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


def segment_transients(amplitudes_fs, decay_minutes, elapsed_seconds, segment_ids):
    """Delay (s) of a settling transient after each opening: A_j exp(-(t - t_j)/T) per row.

    t_j is the first row of segment j (the opening itself is not timestamped, so A_j is the
    transient's size at the first scan, not at the opening). Rows of a segment must be contiguous
    in time; ``segment_ids`` are 0..n_segments-1.
    """
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    segments = np.asarray(segment_ids, dtype=int)
    starts = np.array([elapsed[segments == segment].min()
                       for segment in range(segments.max() + 1)])
    since_opening_minutes = (elapsed - starts[segments]) / 60.0
    return (np.asarray(amplitudes_fs, dtype=float)[segments] * 1e-15
            * np.exp(-since_opening_minutes / decay_minutes))


#: drift model -> the names of its parameters for a given number of rows and of segments
#: (acquisitions between box openings). The one place a drift model is declared; its parameter
#: count is len(names).
DRIFT_MODELS = {
    "none": lambda count, segments=1: (),
    "linear_ramp": lambda count, segments=1: ("delay_span_fs",),
    # A purge settling after a disturbance is an exponential with a tau that depends on the
    # disturbance (13-40 min measured); the rate is fitted. Not a quadratic: its small misfit
    # leaks into P/Q through the background when C ~ -1 (F40). purge_spectral_template.py.
    "settling": lambda count, segments=1: ("delay_span_fs", "settling_rate"),
    "per_acquisition": lambda count, segments=1: tuple(f"delay_{k}_fs"
                                                       for k in range(1, count)),
    # Per-scan rows only: the block's purge settle (same exponential trend as 'settling') + one
    # transient per opening, shared decay time (2026-10-07: the first single-pass block measured
    # -0.15 to -0.4 fs/min inside every file and 1-20 fs settles after each opening; per-file
    # averages could not tell that drift from P/Q).
    "segment_settling": lambda count, segments=1: (
        ("delay_span_fs", "settling_rate", "transient_decay_minutes")
        + tuple(f"transient_{k}_fs" for k in range(segments))),
}

#: Drift models that need segment_ids and a timestamp per row.
SEGMENTED_DRIFT_MODELS = frozenset({"segment_settling"})

#: Bounds and starting value of the shared transient decay time, minutes. The cap is what keeps
#: the model identifiable: a transient much longer than its curvature can be seen inside one
#: file is a straight line there, and its amplitude -- the level jump at the opening -- then
#: trades one-for-one with P/Q. Measured 2026-10-07: with a 30 min cap the gold fit wandered
#: between a 0.5 min and a 30 min solution with 6 vs 44 fs transients (n moved 0.6), and on the
#: 10-scan trimmed files the transients came out 100 +/- 140 fs. Capped at ~4 scans (48 s each),
#: a transient can only describe the first scans after an opening; the slow purge belongs to the
#: block-wide trend.
TRANSIENT_DECAY_MINUTES_BOUNDS = (0.25, 3.0)
TRANSIENT_DECAY_MINUTES_START = 1.0


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
    # normal[f] = A^T diag(w_f) A ; right_hand_side[f] = A^T diag(w_f) b_f. Written as matrix
    # products (not einsum): with one row per scan this runs ~800 times per fit.
    column_count = design.shape[1]
    outer = (design[:, :, None] * design[:, None, :]).reshape(design.shape[0], -1)
    normal = (weights.T @ outer).reshape(-1, column_count, column_count)
    right_hand_side = (design.T @ (weights * derotated)).T
    inverse_normal = np.linalg.inv(normal)
    solution = np.matmul(inverse_normal, right_hand_side[:, :, None])[:, :, 0].T
    residual = derotated - design @ solution
    if amplitudes is not None:
        residual = residual * amplitudes            # back in measured units
    return solution, residual, inverse_normal


#: A bin enters the nuisance (drift) search only if its mean per-row signal-to-noise power is
#: above this; for an unweighted fit, if its amplitude is above this fraction of the largest.
#: Noise-only bins carry no drift information (their phase is random), only cost: with one row
#: per scan the search runs ~800 times, and a synthetic record has ~1000 bins, most above the
#: emitter's bandwidth. The final P, Q, B solve still uses every bin.
INFORMATIVE_SIGNAL_TO_NOISE_POWER = 1.0
INFORMATIVE_RELATIVE_AMPLITUDE = 1e-3


def _informative_frequencies(spectra, weights, weighted):
    """Boolean mask of the bins worth searching the drift on (never fewer than all if empty)."""
    power = np.abs(spectra) ** 2
    if weighted:
        keep = np.mean(power * weights, axis=0) > INFORMATIVE_SIGNAL_TO_NOISE_POWER
    else:
        amplitude = np.sqrt(np.mean(power, axis=0))
        keep = amplitude > INFORMATIVE_RELATIVE_AMPLITUDE * amplitude.max()
    return keep if keep.any() else np.ones(spectra.shape[1], dtype=bool)


def fit_emitter_harmonic(emitter_angles_rad, spectra, frequencies_hz,
                         drift_model="linear_ramp", maximum_drift_fs=500.0, *,
                         elapsed_seconds=None, background_term=False, spectral_variance=None,
                         amplitude_model="none", segment_ids=None, frequency_oversampling=1.0):
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
    segment_ids : (n_rows,) int array, optional
        When each row is one repeat scan: the acquisition (magnet state between two openings
        of the box) it belongs to, numbered 0..n-1 in time order. Required by
        'segment_settling'; also restricts the revisit diagnostic to repeats ACROSS openings.
    frequency_oversampling : float
        FFT bins per independent frequency point (``resolution.oversampling`` from the
        preprocessing; 1 for an unpadded record). The drift and amplitude parameters are shared
        across frequencies, so neighbouring padded bins -- which carry the same information --
        would otherwise shrink their errors by its square root.

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
    segment_count = 1
    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids, dtype=int)
        if segment_ids.size != acquisition_count:
            raise ValueError("segment_ids needs one entry per row")
        if set(np.unique(segment_ids)) != set(range(int(segment_ids.max()) + 1)):
            raise ValueError("segment_ids must number the segments 0..n-1 with none missing")
        segment_count = int(segment_ids.max()) + 1
    if drift_model in SEGMENTED_DRIFT_MODELS and (segment_ids is None or elapsed_seconds is None):
        raise ValueError(f"drift_model={drift_model!r} fits a transient after each opening of "
                         "the box, so it needs per-scan rows: segment_ids AND elapsed_seconds")
    drift_parameter_names = DRIFT_MODELS[drift_model](acquisition_count, segment_count)
    delay_parameter_count = len(drift_parameter_names)
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
                                         acquisition_count, elapsed_seconds, segment_ids)
        amplitudes = _amplitudes_from_parameters(parameters[delay_parameter_count:],
                                                 amplitude_model, acquisition_count,
                                                 elapsed_seconds, frequencies_hz)
        return delays, amplitudes

    nuisance_errors = None
    nuisance_values = None
    if parameter_count == 0:
        delays = np.zeros(acquisition_count)
        amplitudes = np.ones((acquisition_count, frequency_count))
    else:
        informative = _informative_frequencies(spectra, weights,
                                               spectral_variance is not None)
        search_spectra = spectra[:, informative]
        search_frequencies = frequencies_hz[informative]
        search_weights = weights[:, informative]
        search_root_weights = root_weights[:, informative]

        def residual_vector(parameters):
            trial_delays, trial_amplitudes = nuisance(parameters)
            if trial_amplitudes.shape[1] > 1:
                trial_amplitudes = trial_amplitudes[:, informative]
            _, residual, _ = _solve_linear(design, search_spectra, search_frequencies,
                                           trial_delays, search_weights, trial_amplitudes)
            weighted = residual * search_root_weights
            return np.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

        # Delays in fs within +/- maximum_drift_fs; amplitude parameters within +/- 0.5 (a
        # 50% scale change or e^0.5 is far outside anything physical).
        bound = np.concatenate([maximum_drift_fs * np.ones(delay_parameter_count),
                                0.5 * np.ones(amplitude_parameter_count)])
        start = np.zeros(parameter_count)
        names = drift_parameter_names + AMPLITUDE_MODELS[amplitude_model](acquisition_count)
        lower = -bound
        for position, name in enumerate(names):
            if name == "transient_decay_minutes":
                lower[position], bound[position] = TRANSIENT_DECAY_MINUTES_BOUNDS
                start[position] = TRANSIENT_DECAY_MINUTES_START
            if name.endswith("settling_rate"):
                # The rate is block length / tau: +/-10 spans tau from a tenth of the block to
                # infinity. Start at a moderate settle; at rate 0 its gradient vanishes.
                bound[position] = 10.0
                start[position] = 1.0
        result = least_squares(residual_vector, start,
                               bounds=(lower, bound), xtol=1e-12, ftol=1e-12)
        delays, amplitudes = nuisance(result.x)
        nuisance_values = np.asarray(result.x, dtype=float)
        nuisance_errors = (_nuisance_standard_errors(result, spectral_variance is not None)
                           * np.sqrt(max(float(frequency_oversampling), 1.0)))

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
        nuisance_parameters=nuisance_values,
        nuisance_parameter_names=(drift_parameter_names
                                  + AMPLITUDE_MODELS[amplitude_model](acquisition_count)),
        segment_ids=segment_ids,
        frequency_oversampling=max(float(frequency_oversampling), 1.0),
        row_scales=np.broadcast_to(amplitudes, spectra.shape).copy(),
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
    and a noisy run. With per-scan rows (``fit.segment_ids``) only pairs from DIFFERENT
    acquisitions count: repeats inside one acquisition pin the drift rate, not a jump at an
    opening of the box.
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
    segments = (np.arange(angles.size) if fit.segment_ids is None
                else np.asarray(fit.segment_ids))
    lever = 0.0
    for first in range(angles.size):
        for second in range(first + 1, angles.size):
            if segments[first] == segments[second]:
                continue
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
