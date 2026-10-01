"""Load a polarisation series from disk and turn it into spectra.

This is one of the two modules that know the repository and the bench exist. It reads ``.acc``
accumulation files, groups them by the emitter polarisation angle recorded in the filename, and
preprocesses each to a complex spectrum.

Filename grammar follows the repo convention: ``<type>[_<token>]*.acc``, where a token of the
form ``key=value`` is extracted regardless of its position. The emitter angle is expected as
``pol=15`` by default, so ``hr-silicon_pol=15.acc`` and ``ref-gold_pol=15.acc`` both parse.

Why this module does not go through ``DataSet``: that machinery is built around sample/reference
pairing of single acquisitions, whereas a polarisation series is a different shape -- one sample,
many angles. Keeping the MVP loader small and array-based is what makes the package demarcated
and the whole chain testable on synthetic files. The repeat-scan noise model and drift tracking
in ``dataset_core`` remain available for the production path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    "AccumulationFile",
    "PolarisationSeries",
    "load_polarisation_series",
    "parse_key_value_tokens",
    "polarisation_angle_deg_from_filename",
    "read_accumulation_file",
    "spectra_from_traces",
]

DEFAULT_REFERENCE_VOCABULARY = ("gold", "mirror", "ref", "reference")


@dataclass(frozen=True)
class AccumulationFile:
    path: str
    time_ps: np.ndarray            #: (n_samples,)
    scans: np.ndarray              #: (n_scans, n_samples)
    title: str

    @property
    def averaged(self):
        return self.scans.mean(axis=0)


@dataclass(frozen=True)
class PolarisationSeries:
    """One sample measured at several emitter polarisation angles."""

    label: str
    angles_deg: np.ndarray
    time_ps: np.ndarray
    traces: np.ndarray             #: (n_angles, n_samples), repeat-averaged
    scan_counts: np.ndarray
    paths: list

    @property
    def angles_rad(self):
        return np.deg2rad(self.angles_deg)

    def __len__(self):
        return len(self.angles_deg)


def parse_key_value_tokens(filename, delimiter="_"):
    """Extract the ``key=value`` tokens from a filename, following the repo grammar."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    tokens = {}
    for token in stem.split(delimiter):
        if "=" in token:
            key, _, value = token.partition("=")
            tokens[key.strip().lower()] = value.strip()
    return tokens


def polarisation_angle_deg_from_filename(filename, angle_token="pol", delimiter="_"):
    """Emitter angle in degrees, or None when the token is absent.

    A leading ``m`` stands in for a minus sign, because ``-`` is not safe in every acquisition
    program's filename handling: ``pol=m30`` means -30 degrees.
    """
    raw = parse_key_value_tokens(filename, delimiter).get(angle_token.lower())
    if raw is None:
        return None
    text = raw[1:] if raw[:1].lower() == "m" else raw
    sign = -1.0 if raw[:1].lower() == "m" else 1.0
    try:
        return sign * float(text)
    except ValueError:
        return None


def read_accumulation_file(path):
    """Parse one ``.acc`` file: ``%``-prefixed header lines, then two columns per scan.

    A file holds several repeat scans back to back, each introduced by its own header block.
    """
    times, amplitudes, scans, title = [], [], [], ""
    on_first_scan = True
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                if stripped.lower().startswith("%title"):
                    if amplitudes:
                        scans.append(np.asarray(amplitudes, dtype=float))
                        amplitudes = []
                        on_first_scan = False
                    if not title:
                        parts = stripped.split()
                        title = parts[1] if len(parts) > 1 else os.path.basename(path)
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                time_value, amplitude = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if on_first_scan:
                times.append(time_value)
            amplitudes.append(amplitude)
    if amplitudes:
        scans.append(np.asarray(amplitudes, dtype=float))
    if not scans:
        raise ValueError(f"no data rows parsed from {path}")

    sample_count = min(min(scan.size for scan in scans), len(times))
    stacked = np.vstack([scan[:sample_count] for scan in scans])
    time_axis = np.asarray(times[:sample_count], dtype=float)
    if time_axis.size < 8:
        raise ValueError(
            f"{path}: only {time_axis.size} time points parsed; the file does not look like "
            "an .acc accumulation (expected '%'-prefixed headers then two numeric columns)")
    return AccumulationFile(path=path, time_ps=time_axis, scans=stacked,
                            title=title or os.path.basename(path))


def _is_reference(filename, vocabulary):
    lowered = os.path.basename(filename).lower()
    return any(word in lowered for word in vocabulary)


def load_polarisation_series(directory, *, angle_token="pol", delimiter="_",
                             reference_vocabulary=DEFAULT_REFERENCE_VOCABULARY,
                             extension=".acc"):
    """Load every ``.acc`` in a directory and split it into sample and reference series.

    Returns ``(sample_series, reference_series)``; the reference is ``None`` when no file
    matches the reference vocabulary.

    Errors are explicit about what was found versus what was expected, because the filename
    convention is the one part of this pipeline that depends on the acquisition software and
    is therefore the most likely thing to be wrong on first contact with real data.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"no such directory: {directory}")
    candidates = sorted(name for name in os.listdir(directory)
                        if name.lower().endswith(extension))
    if not candidates:
        raise FileNotFoundError(f"no {extension} files in {directory}")

    grouped = {"sample": [], "reference": []}
    missing_token = []
    for name in candidates:
        angle = polarisation_angle_deg_from_filename(name, angle_token, delimiter)
        if angle is None:
            missing_token.append(name)
            continue
        key = "reference" if _is_reference(name, reference_vocabulary) else "sample"
        grouped[key].append((angle, os.path.join(directory, name)))

    if missing_token and not grouped["sample"]:
        raise ValueError(
            f"none of the {len(candidates)} files in {directory} carry a "
            f"'{angle_token}=<degrees>' token. Found filenames like {missing_token[0]!r}. "
            f"Set polarization.angle_token to whatever the acquisition software writes.")
    if not grouped["sample"]:
        raise ValueError(
            f"every file in {directory} matched the reference vocabulary "
            f"{list(reference_vocabulary)}; nothing is left to treat as the sample")

    def build(entries, label):
        if not entries:
            return None
        entries = sorted(entries)
        files = [read_accumulation_file(path) for _, path in entries]
        lengths = {file.time_ps.size for file in files}
        if len(lengths) != 1:
            raise ValueError(
                f"{label}: traces have different lengths {sorted(lengths)}; "
                "a polarisation series must share one time axis")
        return PolarisationSeries(
            label=label,
            angles_deg=np.array([angle for angle, _ in entries], dtype=float),
            time_ps=files[0].time_ps,
            traces=np.vstack([file.averaged for file in files]),
            scan_counts=np.array([file.scans.shape[0] for file in files], dtype=int),
            paths=[path for _, path in entries],
        )

    return build(grouped["sample"], "sample"), build(grouped["reference"], "reference")


# ---------------------------------------------------------------------------
# Preprocessing: baseline, fixed-width window, zero pad, FFT
# ---------------------------------------------------------------------------

def spectra_from_traces(time_ps, traces, *, window_half_width_ps=None,
                        baseline_fraction=0.1, pad_factor=4, window_centre_ps=None):
    """Baseline-subtract, window and transform a set of traces to complex spectra.

    The window is a fixed-width Hann applied at a COMMON centre for every angle, found from the
    mean absolute trace. Using one centre for the whole series matters: windowing each trace at
    its own peak would silently remove the very timing differences between polarisation
    settings that the drift model exists to measure.
    """
    time_ps = np.asarray(time_ps, dtype=float)
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    sample_count = time_ps.size
    if traces.shape[1] != sample_count:
        raise ValueError(f"traces have {traces.shape[1]} samples but the time axis has "
                         f"{sample_count}")

    baseline_count = max(int(baseline_fraction * sample_count), 1)
    corrected = traces - traces[:, :baseline_count].mean(axis=1, keepdims=True)

    if window_centre_ps is None:
        centre_index = int(np.argmax(np.abs(corrected).mean(axis=0)))
    else:
        centre_index = int(np.argmin(np.abs(time_ps - window_centre_ps)))

    if window_half_width_ps is None:
        windowed = corrected
    else:
        time_step_ps = float(np.mean(np.diff(time_ps)))
        half_width = max(int(round(window_half_width_ps / time_step_ps)), 2)
        window = np.zeros(sample_count)
        start = max(centre_index - half_width, 0)
        stop = min(centre_index + half_width + 1, sample_count)
        window[start:stop] = np.hanning(stop - start)
        windowed = corrected * window[None, :]

    padded_length = int(sample_count * max(pad_factor, 1))
    time_step_s = float(np.mean(np.diff(time_ps))) * 1e-12
    spectra = np.fft.rfft(windowed, n=padded_length, axis=1)
    frequencies_hz = np.fft.rfftfreq(padded_length, d=time_step_s)
    return frequencies_hz, spectra
