"""Synthetic ellipsometry measurements -- a first-class deliverable, not a test fixture.

This is how the pipeline gets debugged without beam time, and how every failure mode gets a
known answer. Every error channel the analysis is supposed to survive is injectable here:
measurement noise, per-acquisition timing drift, a mis-set channel ratio, sample tilt, beam
divergence and an unknown emitter angle offset.

Two levels are provided. ``synthesize_spectra`` produces the frequency-domain arrays the
harmonic fitter consumes, for fast unit tests. ``write_accumulation_files`` produces real
``.acc`` files on disk, so the full driver -- loader, grouping, preprocessing, fit, calibration,
inversion -- can be exercised end to end exactly as it will be on real data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .inversion import effective_jones_with_divergence
from .model import (
    BALANCED_PROBE_AZIMUTH_RAD,
    electro_optic_detection_vector,
    measured_amplitude,
    reflection_coefficients,
)

__all__ = [
    "SyntheticMeasurement",
    "synthesize_spectra",
    "synthesize_time_domain",
    "write_accumulation_files",
]


@dataclass(frozen=True)
class SyntheticMeasurement:
    """Synthetic data plus the truth that generated it."""

    emitter_angles_rad: np.ndarray
    frequencies_hz: np.ndarray
    spectra: np.ndarray                  #: (n_angles, n_frequencies) complex
    true_index: np.ndarray               #: (n_frequencies,) complex
    true_incidence_angle_rad: float
    true_channel_ratio: complex
    true_delays_s: np.ndarray
    label: str


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


def synthesize_spectra(index_sample, frequencies_hz, *, emitter_angles_rad=None,
                       incidence_angle_rad=np.deg2rad(70.0), index_incident=1.0,
                       probe_azimuth_rad=BALANCED_PROBE_AZIMUTH_RAD,
                       emitted_spectrum=None,
                       relative_noise=0.0, drift_span_s=0.0, white_jitter_s=0.0,
                       out_of_plane_tilt_rad=0.0, angular_spread_rad=0.0,
                       emitter_angle_offset_rad=0.0, label="synthetic", seed=0):
    """Build a polarisation series with every error channel injectable.

    ``emitter_angle_offset_rad`` models a magnet that is repeatable but not accurate: the true
    angles differ from the commanded ones by a constant. The calibration is supposed to absorb
    that, so the simulator must be able to produce it.
    """
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    index_sample = np.broadcast_to(np.asarray(index_sample, dtype=complex),
                                   frequencies_hz.shape)
    if emitter_angles_rad is None:
        emitter_angles_rad = np.linspace(0.0, np.pi, 12, endpoint=False)
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)

    detection = electro_optic_detection_vector(probe_azimuth_rad)
    spectrum = (default_emitted_spectrum(frequencies_hz) if emitted_spectrum is None
                else np.asarray(emitted_spectrum, dtype=float))

    generator = np.random.default_rng(seed)
    angle_count = emitter_angles_rad.size
    elapsed = np.arange(angle_count, dtype=float) / max(angle_count - 1, 1)
    delays = drift_span_s * elapsed + generator.normal(0.0, white_jitter_s, size=angle_count)

    true_angles = emitter_angles_rad + emitter_angle_offset_rad
    spectra = np.zeros((angle_count, frequencies_hz.size), dtype=complex)
    for frequency_position, frequency in enumerate(frequencies_hz):
        jones = _tilted_jones(index_sample[frequency_position], incidence_angle_rad,
                              out_of_plane_tilt_rad, index_incident, angular_spread_rad)
        projected = detection @ jones                      # (2,) row vector d^T J
        channel_p = projected[0] * spectrum[frequency_position]
        channel_s = projected[1] * spectrum[frequency_position]
        spectra[:, frequency_position] = measured_amplitude(
            true_angles, np.array([channel_p]), np.array([channel_s]),
            np.array([frequency]), delays)[:, 0]

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
    )


# ---------------------------------------------------------------------------
# Time domain, for driving the real loader
# ---------------------------------------------------------------------------

def synthesize_time_domain(index_sample_function, *, emitter_angles_rad,
                           incidence_angle_rad=np.deg2rad(70.0), index_incident=1.0,
                           probe_azimuth_rad=BALANCED_PROBE_AZIMUTH_RAD,
                           sample_count=512, time_step_ps=0.05, pulse_centre_ps=5.0,
                           pulse_width_ps=0.11, relative_noise=0.0, drift_span_s=0.0,
                           out_of_plane_tilt_rad=0.0, seed=0):
    """Time-domain traces for each emitter angle.

    The incident pulse is the derivative of a Gaussian -- the standard single-cycle shape, with
    no DC component. The default width puts the spectral peak near 1.4 THz with usable content
    to beyond 3 THz, which is roughly what the spintronic emitter delivers; a narrower pulse
    would make the synthetic data misleadingly easy at the top of the band. The sample response is applied in the frequency domain and transformed
    back, so the traces carry exactly the physics the analysis is meant to recover.

    Returns ``(time_ps, traces)`` with traces shaped (n_angles, sample_count), real valued.
    """
    emitter_angles_rad = np.asarray(emitter_angles_rad, dtype=float)
    time_ps = np.arange(sample_count, dtype=float) * time_step_ps
    offset = (time_ps - pulse_centre_ps) / pulse_width_ps
    incident = -offset * np.exp(-0.5 * offset**2)

    frequencies_hz = np.fft.rfftfreq(sample_count, d=time_step_ps * 1e-12)
    incident_spectrum = np.fft.rfft(incident)
    detection = electro_optic_detection_vector(probe_azimuth_rad)

    index_sample = np.asarray(index_sample_function(np.where(frequencies_hz > 0,
                                                             frequencies_hz, 1.0)),
                              dtype=complex)
    index_sample = np.broadcast_to(index_sample, frequencies_hz.shape)

    channel_p = np.zeros(frequencies_hz.size, dtype=complex)
    channel_s = np.zeros(frequencies_hz.size, dtype=complex)
    for position in range(frequencies_hz.size):
        jones = _tilted_jones(index_sample[position], incidence_angle_rad,
                              out_of_plane_tilt_rad, index_incident, 0.0)
        projected = detection @ jones
        channel_p[position] = projected[0]
        channel_s[position] = projected[1]

    angle_count = emitter_angles_rad.size
    elapsed = np.arange(angle_count, dtype=float) / max(angle_count - 1, 1)
    delays = drift_span_s * elapsed

    generator = np.random.default_rng(seed)
    traces = np.zeros((angle_count, sample_count))
    for position, (angle, delay) in enumerate(zip(emitter_angles_rad, delays)):
        response = channel_p * np.cos(angle) + channel_s * np.sin(angle)
        shifted = response * np.exp(1j * 2.0 * np.pi * frequencies_hz * delay)
        traces[position] = np.fft.irfft(incident_spectrum * shifted, n=sample_count)

    if relative_noise:
        traces = traces + relative_noise * np.abs(traces).max() * generator.normal(
            size=traces.shape)
    return time_ps, traces


def write_accumulation_files(directory, *, index_sample_function, emitter_angles_rad,
                             sample_name="sample", scans_per_angle=2,
                             angle_token="pol", **kwargs):
    """Write one ``.acc`` file per emitter angle, in the real on-disk format.

    Filenames follow the repo's grammar, ``<type>_<key>=<value>.acc``, e.g.
    ``silicon_pol=15.acc`` -- a key=value token that the existing parser extracts regardless of
    position. Each file holds ``scans_per_angle`` repeat scans, as a real accumulation does.

    Returns the list of paths written.
    """
    os.makedirs(directory, exist_ok=True)
    time_ps, traces = synthesize_time_domain(
        index_sample_function, emitter_angles_rad=emitter_angles_rad, **kwargs)

    written = []
    for position, angle in enumerate(np.asarray(emitter_angles_rad, dtype=float)):
        angle_deg = float(np.rad2deg(angle))
        label = f"{angle_deg:g}".replace("-", "m")
        path = os.path.join(directory, f"{sample_name}_{angle_token}={label}.acc")
        lines = []
        for scan_index in range(1, scans_per_angle + 1):
            lines.append(f"%title {sample_name}_{angle_token}={label} acc {scan_index}")
            lines.append("%Created 3")
            lines.append("%type 0")
            lines.append("%Parameters Parameters")
            lines.append(
                f"%param Date and time,2026-10-01 "
                f"{10 + position // 60:02d}:{position % 60:02d}:{scan_index:02d}.000000")
            for time_value, amplitude in zip(time_ps, traces[position]):
                lines.append(f"{time_value:.18e} {amplitude:.18e}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        written.append(path)
    return written
