"""Synthetic ellipsometry measurements -- a first-class deliverable, not a test fixture.

This is how the pipeline gets debugged without beam time, and how every failure mode gets a
known answer. Every error channel the analysis is supposed to survive is injectable here:
measurement noise, timing drift over real elapsed time, a non-magnetic background, sample tilt,
beam divergence, an unknown emitter angle offset and the crystal orientation.

Two levels are provided. ``synthesize_spectra`` produces the frequency-domain arrays the
harmonic fitter consumes, for fast unit tests. ``synthesize_acquisitions`` produces real-valued
repeat scans per acquisition; ``thz_ellipsometry.adapters.synthetic_files`` writes those to disk
as ``.acc`` files so the full driver -- loader, noise model, fit, calibration, inversion -- can
be exercised end to end exactly as it will be on real data. Writing files is I/O, so it lives in
the adapter layer, not here.

An acquisition is one row: one magnet state, one probe setting, a few repeat scans, a time. An
interleaved measurement (plan sec. 4.2) is a schedule of rows that visits every angle once per
cycle; ``interleaved_schedule`` builds one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import BALANCED_PROBE_AZIMUTH_RAD, detection_vector_in_sample_frame
from .inversion import effective_jones_with_divergence
from .model import measured_amplitude, reflection_coefficients

__all__ = [
    "AcquisitionSchedule",
    "SyntheticAcquisitions",
    "SyntheticMeasurement",
    "interleaved_schedule",
    "synthesize_acquisitions",
    "synthesize_spectra",
    "synthesize_time_domain",
]


@dataclass(frozen=True)
class SyntheticMeasurement:
    """Synthetic frequency-domain data plus the truth that generated it."""

    emitter_angles_rad: np.ndarray
    frequencies_hz: np.ndarray
    spectra: np.ndarray                  #: (n_acquisitions, n_frequencies) complex
    true_index: np.ndarray               #: (n_frequencies,) complex
    true_incidence_angle_rad: float
    true_channel_ratio: complex
    true_delays_s: np.ndarray
    label: str
    elapsed_seconds: np.ndarray | None = None


@dataclass(frozen=True)
class AcquisitionSchedule:
    """The order a measurement visits its settings in."""

    polarization_angles_rad: np.ndarray   #: (n_acquisitions,) THz polarisation from p
    elapsed_seconds: np.ndarray           #: (n_acquisitions,) start time of each acquisition
    cycle_numbers: np.ndarray             #: (n_acquisitions,) 1-based cycle of each

    def __len__(self):
        return self.polarization_angles_rad.size


def interleaved_schedule(polarization_angles_deg, cycles=1, seconds_per_acquisition=120.0):
    """Visit every angle once per cycle, in order, for ``cycles`` cycles (plan sec. 4.2).

    Cycling rather than recording each state in one block spreads a slow drift equally over all
    states, which is what lets the ramp model describe it with one parameter.
    """
    angles = np.deg2rad(np.asarray(polarization_angles_deg, dtype=float))
    if angles.size == 0 or cycles < 1:
        raise ValueError("need at least one angle and one cycle")
    polarization = np.tile(angles, cycles)
    cycle_numbers = np.repeat(np.arange(1, cycles + 1), angles.size)
    elapsed = np.arange(polarization.size, dtype=float) * float(seconds_per_acquisition)
    return AcquisitionSchedule(polarization_angles_rad=polarization, elapsed_seconds=elapsed,
                               cycle_numbers=cycle_numbers)


def default_emitted_spectrum(frequencies_hz, centre_thz=1.0, width_thz=0.8):
    """A plausible broadband THz amplitude spectrum -- a lognormal-ish single-lobe envelope.

    Only the SHAPE matters: any common spectral factor cancels in P/Q, so this exists to make
    the synthetic signal-to-noise realistic rather than to model the emitter.
    """
    frequencies_thz = np.asarray(frequencies_hz, dtype=float) / 1e12
    safe = np.where(frequencies_thz > 0, frequencies_thz, 1e-6)
    envelope = safe * np.exp(-((safe - centre_thz) ** 2) / (2.0 * width_thz**2))
    peak = envelope.max()
    return envelope / (peak if peak else 1.0)


def _tilted_jones(index_sample, incidence_angle_rad, out_of_plane_tilt_rad, index_incident,
                  angular_spread_rad):
    if angular_spread_rad > 0.0:
        jones = effective_jones_with_divergence(index_sample, incidence_angle_rad,
                                                angular_spread_rad, index_incident)
    else:
        reflection_p, reflection_s = reflection_coefficients(
            index_sample, incidence_angle_rad, index_incident)
        jones = np.array([[reflection_p, 0.0], [0.0, reflection_s]], dtype=complex)
    if out_of_plane_tilt_rad:
        cosine, sine = np.cos(out_of_plane_tilt_rad), np.sin(out_of_plane_tilt_rad)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        jones = rotation @ jones @ rotation.T
    return jones


def _drift_delays(acquisition_count, elapsed_seconds, drift_span_s):
    """A linear drift reaching ``drift_span_s`` at the last acquisition, over elapsed time.

    An array ``drift_span_s`` is taken as explicit per-acquisition delays.
    """
    if np.ndim(drift_span_s):
        return np.asarray(drift_span_s, dtype=float)
    if elapsed_seconds is None:
        fraction = np.arange(acquisition_count, dtype=float) / max(acquisition_count - 1, 1)
    else:
        elapsed = np.asarray(elapsed_seconds, dtype=float)
        span = np.ptp(elapsed)
        fraction = (elapsed - elapsed.min()) / span if span > 0 else np.zeros_like(elapsed)
    return float(drift_span_s) * fraction


def _amplitude_factors(acquisition_count, elapsed_seconds, amplitude_drift):
    """1 at the first acquisition, 1 + amplitude_drift at the last, linear in elapsed time."""
    return 1.0 + float(amplitude_drift) * (
        _drift_delays(acquisition_count, elapsed_seconds, 1.0))


def _tilt_factors(acquisition_count, elapsed_seconds, frequencies_hz, tilt_drift_per_thz,
                  tilt_profile=None):
    """exp(tilt(t) * (f - 1 THz)): the purge's spectral tilt, 0 at the first acquisition.

    ``tilt_profile`` (n_acquisitions,) gives the tilt per THz explicitly (e.g. an exponential
    settling); otherwise it rises linearly to ``tilt_drift_per_thz`` at the last acquisition.
    """
    if tilt_profile is None:
        tilt_profile = float(tilt_drift_per_thz) * _drift_delays(acquisition_count,
                                                                 elapsed_seconds, 1.0)
    offset_thz = (np.asarray(frequencies_hz, dtype=float) - 1.0e12) / 1e12
    return np.exp(np.asarray(tilt_profile, dtype=float)[:, None] * offset_thz[None, :])


def _channels(index_sample, frequencies_hz, incidence_angle_rad, index_incident, detection,
              out_of_plane_tilt_rad, angular_spread_rad):
    """P and Q per frequency, before the emitted spectrum: d^T J projected on p and s."""
    channel_p = np.zeros(frequencies_hz.size, dtype=complex)
    channel_s = np.zeros(frequencies_hz.size, dtype=complex)
    for position in range(frequencies_hz.size):
        jones = _tilted_jones(index_sample[position], incidence_angle_rad,
                              out_of_plane_tilt_rad, index_incident, angular_spread_rad)
        projected = detection @ jones                      # (2,) row vector d^T J
        channel_p[position], channel_s[position] = projected
    return channel_p, channel_s


def synthesize_spectra(index_sample, frequencies_hz, *, emitter_angles_rad=None,
                       incidence_angle_rad=np.deg2rad(70.0), index_incident=1.0,
                       probe_azimuth_rad=BALANCED_PROBE_AZIMUTH_RAD,
                       crystal_orientation_rad=0.0, emitted_spectrum=None,
                       relative_noise=0.0, drift_span_s=0.0, white_jitter_s=0.0,
                       elapsed_seconds=None, background_relative=0.0, amplitude_drift=0.0,
                       tilt_drift_per_thz=0.0, tilt_profile=None,
                       out_of_plane_tilt_rad=0.0, angular_spread_rad=0.0,
                       emitter_angle_offset_rad=0.0, label="synthetic", seed=0):
    """Build a polarisation series in the frequency domain with every error channel injectable.

    ``emitter_angle_offset_rad`` models a magnet that is repeatable but not accurate: the true
    angles differ from the commanded ones by a constant. ``background_relative`` adds a
    polarisation-independent complex background, as a fraction of the peak signal, carrying the
    emitted spectrum's shape (it rides on the same delay line, so it drifts too).
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    index_sample = np.broadcast_to(np.asarray(index_sample, dtype=complex),
                                   frequencies_hz.shape)
    if emitter_angles_rad is None:
        emitter_angles_rad = np.linspace(0.0, np.pi, 12, endpoint=False)
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)

    detection = detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad)
    spectrum = (default_emitted_spectrum(frequencies_hz) if emitted_spectrum is None
                else np.asarray(emitted_spectrum, dtype=float))

    generator = np.random.default_rng(seed)
    acquisition_count = emitter_angles_rad.size
    delays = (_drift_delays(acquisition_count, elapsed_seconds, drift_span_s)
              + generator.normal(0.0, white_jitter_s, size=acquisition_count))

    channel_p, channel_s = _channels(index_sample, frequencies_hz, incidence_angle_rad,
                                     index_incident, detection, out_of_plane_tilt_rad,
                                     angular_spread_rad)
    true_angles = emitter_angles_rad + emitter_angle_offset_rad
    spectra = measured_amplitude(true_angles, channel_p * spectrum, channel_s * spectrum,
                                 frequencies_hz, delays)
    if background_relative:
        peak = np.max(np.abs(spectra))
        phase = np.exp(1j * 2.0 * np.pi * frequencies_hz[None, :] * delays[:, None])
        spectra = spectra + background_relative * peak * spectrum[None, :] * phase
    if amplitude_drift:
        spectra = spectra * _amplitude_factors(acquisition_count, elapsed_seconds,
                                               amplitude_drift)[:, None]
    if tilt_drift_per_thz or tilt_profile is not None:
        spectra = spectra * _tilt_factors(acquisition_count, elapsed_seconds, frequencies_hz,
                                          tilt_drift_per_thz, tilt_profile)

    if relative_noise:
        scale = relative_noise * np.abs(spectra).max()
        spectra = spectra + scale * (generator.normal(size=spectra.shape)
                                     + 1j * generator.normal(size=spectra.shape)) / np.sqrt(2)

    return SyntheticMeasurement(
        emitter_angles_rad=emitter_angles_rad,
        frequencies_hz=frequencies_hz,
        spectra=spectra,
        true_index=np.asarray(index_sample, dtype=complex),
        true_incidence_angle_rad=float(incidence_angle_rad),
        true_channel_ratio=complex(detection[0] / detection[1]),
        true_delays_s=delays,
        label=label,
        elapsed_seconds=None if elapsed_seconds is None else np.asarray(elapsed_seconds,
                                                                        dtype=float),
    )


# ---------------------------------------------------------------------------
# Time domain, for driving the real loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SyntheticAcquisitions:
    """Repeat-scan time traces for each acquisition, plus the truth that generated them."""

    time_ps: np.ndarray                   #: (n_samples,)
    scans: np.ndarray                     #: (n_acquisitions, n_scans, n_samples)
    polarization_angles_rad: np.ndarray   #: (n_acquisitions,) as commanded
    elapsed_seconds: np.ndarray           #: (n_acquisitions,)
    true_delays_s: np.ndarray             #: (n_acquisitions,)
    seconds_per_scan: float

    @property
    def traces(self):
        """(n_acquisitions, n_samples) repeat-averaged traces."""
        return self.scans.mean(axis=1)


def _single_cycle_pulse(time_ps, centre_ps, width_ps):
    """Derivative-of-Gaussian: the standard single-cycle THz shape, with no DC component."""
    offset = (time_ps - centre_ps) / width_ps
    return -offset * np.exp(-0.5 * offset**2)


def synthesize_acquisitions(index_sample_function, *, polarization_angles_rad,
                            elapsed_seconds=None, scans_per_acquisition=1,
                            seconds_per_scan=10.0,
                            incidence_angle_rad=np.deg2rad(70.0), index_incident=1.0,
                            probe_azimuth_rad=BALANCED_PROBE_AZIMUTH_RAD,
                            crystal_orientation_rad=0.0, emitter_angle_offset_rad=0.0,
                            sample_count=512, time_step_ps=0.05, pulse_centre_ps=5.0,
                            pulse_width_ps=0.11, relative_noise=0.0,
                            multiplicative_noise=0.0, drift_span_s=0.0, amplitude_drift=0.0,
                            tilt_drift_per_thz=0.0, tilt_profile=None,
                            background_relative=0.0, background_delay_ps=0.4,
                            background_width_ps=0.3, out_of_plane_tilt_rad=0.0, seed=0):
    """Repeat-scan time traces for each acquisition of a polarisation series.

    The default pulse width puts the spectral peak near 1.4 THz with usable content to beyond
    3 THz, roughly what the spintronic emitter delivers; a narrower pulse would make the
    synthetic data misleadingly easy at the top of the band. The sample response is applied in
    the frequency domain and transformed back, so the traces carry exactly the physics the
    analysis is meant to recover.

    Noise is added to each SCAN independently: ``relative_noise`` is additive (detector), a
    fraction of the peak; ``multiplicative_noise`` scales with the signal (laser power). That is
    what lets the repeat-scan noise model be exercised. The background is a broader pulse,
    ``background_delay_ps`` after the main one, identical at every polarisation (it does not
    reverse with the magnet) and on the same delay line, so it drifts with everything else.
    """
    angles = np.asarray(polarization_angles_rad, dtype=float)
    acquisition_count = angles.size
    time_ps = np.arange(sample_count, dtype=float) * time_step_ps
    incident = _single_cycle_pulse(time_ps, pulse_centre_ps, pulse_width_ps)
    background = _single_cycle_pulse(time_ps, pulse_centre_ps + background_delay_ps,
                                     background_width_ps)

    frequencies_hz = np.fft.rfftfreq(sample_count, d=time_step_ps * 1e-12)
    incident_spectrum = np.fft.rfft(incident)
    background_spectrum = np.fft.rfft(background)
    detection = detection_vector_in_sample_frame(probe_azimuth_rad, crystal_orientation_rad)

    index_sample = np.asarray(index_sample_function(np.where(frequencies_hz > 0,
                                                             frequencies_hz, 1.0)),
                              dtype=complex)
    index_sample = np.broadcast_to(index_sample, frequencies_hz.shape)
    channel_p, channel_s = _channels(index_sample, frequencies_hz, incidence_angle_rad,
                                     index_incident, detection, out_of_plane_tilt_rad, 0.0)

    delays = _drift_delays(acquisition_count, elapsed_seconds, drift_span_s)
    true_angles = angles + emitter_angle_offset_rad
    clean = np.zeros((acquisition_count, sample_count))
    for position, (angle, delay) in enumerate(zip(true_angles, delays)):
        response = channel_p * np.cos(angle) + channel_s * np.sin(angle)
        shift = np.exp(1j * 2.0 * np.pi * frequencies_hz * delay)
        clean[position] = np.fft.irfft(incident_spectrum * response * shift, n=sample_count)
    peak = np.abs(clean).max()
    if background_relative:
        background_peak = np.abs(background).max()
        for position, delay in enumerate(delays):
            shift = np.exp(1j * 2.0 * np.pi * frequencies_hz * delay)
            clean[position] += (background_relative * peak / background_peak
                                * np.fft.irfft(background_spectrum * shift, n=sample_count))

    if amplitude_drift:
        clean = clean * _amplitude_factors(acquisition_count, elapsed_seconds,
                                           amplitude_drift)[:, None]
    if tilt_drift_per_thz or tilt_profile is not None:
        # The purge tilt acts on the spectrum; apply it there and come back.
        tilt = _tilt_factors(acquisition_count, elapsed_seconds,
                             np.where(frequencies_hz > 0, frequencies_hz, 1.0e12),
                             tilt_drift_per_thz, tilt_profile)
        clean = np.fft.irfft(np.fft.rfft(clean, axis=1) * tilt, n=sample_count, axis=1)

    generator = np.random.default_rng(seed)
    scans = np.repeat(clean[:, None, :], scans_per_acquisition, axis=1)
    if multiplicative_noise:
        scans = scans * (1.0 + multiplicative_noise
                         * generator.normal(size=scans.shape[:2])[..., None])
    if relative_noise:
        scans = scans + relative_noise * peak * generator.normal(size=scans.shape)

    if elapsed_seconds is None:
        elapsed_seconds = (np.arange(acquisition_count, dtype=float)
                           * scans_per_acquisition * seconds_per_scan)
    return SyntheticAcquisitions(
        time_ps=time_ps, scans=scans, polarization_angles_rad=angles,
        elapsed_seconds=np.asarray(elapsed_seconds, dtype=float), true_delays_s=delays,
        seconds_per_scan=float(seconds_per_scan))


def synthesize_time_domain(index_sample_function, *, emitter_angles_rad, **kwargs):
    """Repeat-averaged traces only: ``(time_ps, traces)`` with traces (n_acquisitions, n_samples).

    Convenience wrapper over :func:`synthesize_acquisitions` for callers that do not need the
    individual scans.
    """
    acquisitions = synthesize_acquisitions(index_sample_function,
                                           polarization_angles_rad=emitter_angles_rad, **kwargs)
    return acquisitions.time_ps, acquisitions.traces
