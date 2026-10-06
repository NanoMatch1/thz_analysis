"""Bench-setup analyses: the magnet zero, purge settling, and half-wave-plate beam deviation.

Three small, pure analyses that run on the bench while a measurement is being set up, each
answering one question with a number:

``fit_wire_grid_null``        At what magnet reading is the emission exactly p? (plan sec. 4.4)
``settling_trend``            Has the purge settled after the box was closed? (fs/min)
``analyse_half_wave_plate``   Does rotating the probe half-wave plate walk the probe off the THz
                              focus? (plan sec. 7.8)

All three work on complex spectra and a few scalar settings, with no file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "HalfWavePlateAnalysis",
    "SettlingTrend",
    "WireGridNull",
    "analyse_half_wave_plate",
    "fit_wire_grid_null",
    "fit_wire_grid_nulls",
    "relative_delay_and_amplitude",
    "settling_trend",
    "signed_projection",
]


# ---------------------------------------------------------------------------
# Shared: comparing one spectrum with another
# ---------------------------------------------------------------------------

def signed_projection(spectra, reference_spectrum, band):
    """Signed real amplitude of each spectrum along a reference, over a band.

    Coherent detection keeps the field's SIGN, so a polarisation sweep through a polariser null
    goes through zero and changes sign rather than bottoming out at a noise floor. Projecting
    onto a reference spectrum turns each acquisition into one signed number with all of the
    band's signal-to-noise behind it.
    """
    spectra = np.atleast_2d(np.asarray(spectra, dtype=complex))[:, band]
    reference = np.asarray(reference_spectrum, dtype=complex)[band]
    return np.real(spectra @ np.conj(reference)) / np.real(np.vdot(reference, reference))


def relative_delay_and_amplitude(spectrum, reference_spectrum, frequencies_hz, band):
    """(delay in s, amplitude ratio) of a spectrum relative to a reference of the same shape.

    The delay is the slope of the cross-spectrum phase against angular frequency, weighted by
    the cross-spectrum magnitude; positive means the spectrum arrives LATER. Sub-sample by
    construction, unlike a peak-bin comparison.
    """
    cross = np.asarray(spectrum)[band] * np.conj(np.asarray(reference_spectrum)[band])
    angular = 2.0 * np.pi * np.asarray(frequencies_hz)[band]
    phase = np.unwrap(np.angle(cross))
    weights = np.abs(cross)
    design = np.column_stack([np.ones_like(angular), angular]) * np.sqrt(weights)[:, None]
    coefficients, *_ = np.linalg.lstsq(design, phase * np.sqrt(weights), rcond=None)
    # numpy's forward FFT has exp(-i w t): a later arrival is a more NEGATIVE phase slope.
    delay = -coefficients[1]
    reference_power = np.sum(np.abs(np.asarray(reference_spectrum)[band]) ** 2)
    amplitude = float(np.sum(weights) / reference_power) if reference_power > 0 else np.nan
    return float(delay), amplitude


# ---------------------------------------------------------------------------
# The magnet zero, from a wire-grid null
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireGridNull:
    null_deg: float                 #: magnet reading at the null
    null_standard_error_deg: float
    amplitude: float                #: signed amplitude far from the null (A in A sin(b - b0))
    leakage: float                  #: fitted magnet-independent background (0 if not fitted)
    residual_rms: float
    magnet_angles_deg: np.ndarray
    signed_amplitudes: np.ndarray

    @property
    def leakage_fraction(self):
        """Background relative to the full swing."""
        return abs(self.leakage) / abs(self.amplitude) if self.amplitude else np.inf

    def model(self, magnet_angles_deg):
        """The fitted curve ``A sin(b - b0) + c`` at the given readings."""
        angles = np.deg2rad(np.asarray(magnet_angles_deg, dtype=float))
        return self.amplitude * np.sin(angles - np.deg2rad(self.null_deg)) + self.leakage

    @property
    def residuals(self):
        """Measured minus fitted signed amplitude, at each reading of the sweep."""
        return self.signed_amplitudes - self.model(self.magnet_angles_deg)


def _null_near_data(sine_weight, cosine_weight, angles_rad):
    """The zero of a sin(b) + b' cos(b) nearest the middle of the readings, and the sign of A."""
    null = np.arctan2(-cosine_weight, sine_weight)
    centre = np.angle(np.mean(np.exp(1j * angles_rad)))
    candidates = np.array([null, null + np.pi])
    distance = np.abs(np.angle(np.exp(1j * (candidates - centre))))
    chosen = candidates[np.argmin(distance)]
    sign = 1.0 if np.isclose(chosen, null) else -1.0
    return chosen, sign


def fit_wire_grid_nulls(sweeps, *, fit_background=None):
    """Fit one or two null sweeps through the SAME grid orientation, sharing a background.

    Each sweep is ``(magnet_readings_deg, signed_amplitudes)`` and is modelled as
    ``A_j sin(b - b0_j) + c``. The constant c is a magnet-independent background (optical
    rectification in the emitter substrate, pickup). Over a sweep of +/-20 deg it is nearly
    degenerate with a SHIFT of the null -- cos(b) and a constant are almost collinear there -- so
    it is only fitted when two sweeps 180 deg apart share it: the signal reverses with the
    magnet, the background does not, and the pair separates them (the +/-M trick again). A lone
    sweep is fitted without it; ``fit_background`` overrides.

    The signed amplitudes of a pair must share one projection reference, or c means nothing.
    Returns a list of :class:`WireGridNull`, one per sweep.
    """
    sweeps = [(np.deg2rad(np.asarray(readings, dtype=float)), np.asarray(values, dtype=float))
              for readings, values in sweeps]
    if not 1 <= len(sweeps) <= 2:
        raise ValueError("fit one sweep, or two sweeps 180 deg apart through the same grid")
    for angles, _ in sweeps:
        if angles.size < 4:
            raise ValueError("need at least four magnet readings around each null (two each "
                             "side plus two further out)")
    if fit_background is None:
        fit_background = len(sweeps) == 2

    rows, values = [], []
    column_count = 2 * len(sweeps) + (1 if fit_background else 0)
    for index, (angles, amplitudes) in enumerate(sweeps):
        block = np.zeros((angles.size, column_count))
        block[:, 2 * index] = np.sin(angles)
        block[:, 2 * index + 1] = np.cos(angles)
        if fit_background:
            block[:, -1] = 1.0
        rows.append(block)
        values.append(amplitudes)
    design, values = np.vstack(rows), np.concatenate(values)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    dof = max(values.size - column_count, 1)
    noise_variance = float(residual @ residual) / dof
    covariance = np.linalg.inv(design.T @ design) * noise_variance
    background = float(coefficients[-1]) if fit_background else 0.0

    results = []
    for index, (angles, amplitudes) in enumerate(sweeps):
        sine_weight, cosine_weight = coefficients[2 * index:2 * index + 2]
        amplitude = float(np.hypot(sine_weight, cosine_weight))
        chosen, sign = _null_near_data(sine_weight, cosine_weight, angles)
        gradient = np.zeros(column_count)
        gradient[2 * index:2 * index + 2] = np.array([cosine_weight, -sine_weight]) / amplitude**2
        null_error = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
        # Report the null in the same turn as the readings (a scale may run past 360).
        chosen_deg = float(np.rad2deg(chosen))
        mean_reading = float(np.rad2deg(np.mean(angles)))
        chosen_deg += 360.0 * round((mean_reading - chosen_deg) / 360.0)
        results.append(WireGridNull(
            null_deg=chosen_deg,
            null_standard_error_deg=float(np.rad2deg(null_error)),
            amplitude=sign * amplitude, leakage=background,
            residual_rms=float(np.sqrt(noise_variance)),
            magnet_angles_deg=np.rad2deg(angles), signed_amplitudes=amplitudes))
    return results


def fit_wire_grid_null(magnet_angles_deg, signed_amplitudes, *, fit_background=False):
    """One sweep: the magnet reading at the null of A sin(b - b0) (+ c if asked).

    With the grid passing s, the null is the magnet reading that emits pure p. See
    :func:`fit_wire_grid_nulls` for why the background is off by default for a lone sweep.
    """
    return fit_wire_grid_nulls([(magnet_angles_deg, signed_amplitudes)],
                               fit_background=fit_background)[0]


# ---------------------------------------------------------------------------
# Purge settling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SettlingTrend:
    elapsed_minutes: np.ndarray     #: per scan, since the first scan of the group
    delays_fs: np.ndarray           #: per scan, relative to the group's first scan
    amplitudes: np.ndarray          #: per scan, relative to the group's first scan
    recent_rate_fs_per_minute: float
    recent_amplitude_rate_per_minute: float
    window_scans: int

    def is_settled(self, rate_limit_fs_per_minute=1.0):
        return abs(self.recent_rate_fs_per_minute) <= rate_limit_fs_per_minute


def settling_trend(scan_spectra, elapsed_seconds, frequencies_hz, band, window_scans=3):
    """Delay and amplitude of every scan relative to the first, and how fast they still move.

    The scans must be of the SAME setting (one magnet state): comparing a p pulse with an s pulse
    would measure the sample, not the purge. The rate is a straight-line fit over the last
    ``window_scans`` scans, so it says whether the transient is still running NOW.
    """
    scan_spectra = np.atleast_2d(np.asarray(scan_spectra, dtype=complex))
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    reference = scan_spectra[0]
    pairs = [relative_delay_and_amplitude(spectrum, reference, frequencies_hz, band)
             for spectrum in scan_spectra]
    delays_fs = np.array([pair[0] for pair in pairs]) * 1e15
    amplitudes = np.array([pair[1] for pair in pairs])
    minutes = (elapsed - elapsed[0]) / 60.0

    count = min(window_scans, minutes.size)
    if count >= 2 and np.ptp(minutes[-count:]) > 0:
        rate = float(np.polyfit(minutes[-count:], delays_fs[-count:], 1)[0])
        amplitude_rate = float(np.polyfit(minutes[-count:], amplitudes[-count:], 1)[0])
    else:
        rate, amplitude_rate = np.nan, np.nan
    return SettlingTrend(elapsed_minutes=minutes, delays_fs=delays_fs, amplitudes=amplitudes,
                         recent_rate_fs_per_minute=rate,
                         recent_amplitude_rate_per_minute=amplitude_rate, window_scans=count)


# ---------------------------------------------------------------------------
# Half-wave-plate beam deviation (plan sec. 7.8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HalfWavePlateAnalysis:
    plate_angles_deg: np.ndarray
    reference_plate_deg: float
    #: per plate angle, of the ratio to the reference setting:
    log_amplitude_slope_per_thz: np.ndarray   #: d ln|R| / d f -- walk-off rolls off high f
    delay_fs: np.ndarray                      #: phase slope -- a path change
    constant: np.ndarray                      #: complex value of R at the band centre
    #: harmonic decomposition of the slope over plate angle:
    wedge_slope_amplitude: float      #: 360-deg-periodic part (beam deviation)
    wedge_direction_deg: float
    polarisation_slope_amplitude: float   #: 90-deg-periodic part (pure polarisation)
    same_state_slope_spread: float    #: spread of the slope over settings 90 deg apart
    same_state_delay_spread_fs: float

    @property
    def walk_detected(self):
        """Settings that give the SAME polarisation differ: that is beam deviation."""
        return self.same_state_slope_spread > 0.0


def analyse_half_wave_plate(plate_angles_deg, spectra, frequencies_hz, band,
                            reference_plate_deg=None):
    """Separate beam walk from polarisation in a gold measurement at several HWP angles.

    Divide every spectrum by the one at a reference plate angle. A pure polarisation change
    multiplies the spectrum by a real, frequency-independent number (possibly negative); probe
    walk-off at the crystal shows as a high-frequency amplitude roll-off, because the THz focus
    shrinks as 1/f so a fixed offset matters more there; a path change adds a phase ramp.

    Polarisation repeats every 90 deg of plate rotation, a wedge's deviation only every 360 deg.
    So plate angles 90 deg apart give the SAME polarisation, and any difference between them is
    walk; and fitting the slope over plate angle with a 1x (360 deg) and a 4x (90 deg) harmonic
    splits the two.
    """
    angles = np.asarray(plate_angles_deg, dtype=float)
    spectra = np.atleast_2d(np.asarray(spectra, dtype=complex))
    if reference_plate_deg is None:
        reference_plate_deg = float(angles[0])
    reference_index = int(np.argmin(np.abs(angles - reference_plate_deg)))
    reference = spectra[reference_index]
    frequencies_thz = np.asarray(frequencies_hz)[band] / 1e12
    centre_thz = float(np.mean(frequencies_thz))

    slopes, delays, constants = [], [], []
    for spectrum in spectra:
        ratio = spectrum[band] / reference[band]
        weights = np.abs(reference[band])
        log_amplitude = np.log(np.abs(ratio))
        design = np.column_stack([np.ones_like(frequencies_thz),
                                  frequencies_thz - centre_thz]) * weights[:, None]
        amplitude_fit, *_ = np.linalg.lstsq(design, log_amplitude * weights, rcond=None)
        delay, _ = relative_delay_and_amplitude(spectrum, reference, frequencies_hz, band)
        phase_at_centre = np.angle(np.sum(ratio * weights))
        slopes.append(amplitude_fit[1])
        delays.append(delay * 1e15)
        constants.append(np.exp(amplitude_fit[0]) * np.exp(1j * phase_at_centre))
    slopes, delays = np.array(slopes), np.array(delays)

    radians = np.deg2rad(angles)
    columns = [np.ones_like(radians), np.cos(radians), np.sin(radians)]
    if angles.size >= 5:
        columns += [np.cos(4.0 * radians), np.sin(4.0 * radians)]
    design = np.column_stack(columns)
    harmonic, *_ = np.linalg.lstsq(design, slopes, rcond=None)
    wedge_amplitude = float(np.hypot(harmonic[1], harmonic[2]))
    wedge_direction = float(np.rad2deg(np.arctan2(harmonic[2], harmonic[1])))
    polarisation_amplitude = (float(np.hypot(harmonic[3], harmonic[4]))
                              if harmonic.size == 5 else np.nan)

    residues = np.mod(angles, 90.0)
    same_state = np.isclose(residues, np.mod(reference_plate_deg, 90.0), atol=1.0)
    slope_spread = float(np.ptp(slopes[same_state])) if same_state.sum() > 1 else np.nan
    delay_spread = float(np.ptp(delays[same_state])) if same_state.sum() > 1 else np.nan

    return HalfWavePlateAnalysis(
        plate_angles_deg=angles, reference_plate_deg=float(angles[reference_index]),
        log_amplitude_slope_per_thz=slopes, delay_fs=delays, constant=np.array(constants),
        wedge_slope_amplitude=wedge_amplitude, wedge_direction_deg=wedge_direction,
        polarisation_slope_amplitude=polarisation_amplitude,
        same_state_slope_spread=slope_spread, same_state_delay_spread_fs=delay_spread)
