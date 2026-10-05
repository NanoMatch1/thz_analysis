"""Load polarisation series from disk: one per (role, probe setting).

This module knows the repository and the bench exist. It reads ``.acc`` accumulation files and
groups them by what each file is for (its ROLE: the sample, the channel reference, the angle
reference) and by the probe setting it was taken at.

Filename grammar follows the repo convention: ``<type>[_<token>]*.acc``, where a token of the
form ``key=value`` is extracted regardless of its position:

    doped-si_mag=090_cyc=03.acc          sample, magnet at 90 deg, third cycle
    gold_mag=000_probe=31.72.acc         channel reference, magnet 0, probe at 31.72 deg

``mag`` is the magnet READING. THz polarisation is perpendicular to M, so the reading at which the
emission is p -- and, if calibrated, at which it is s, -p and -s -- is an instrument calibration
from the wire-grid nulls (plan sec. 4.4), supplied as ``magnet_calibration`` and applied here via
``core.emitter``; everything downstream works in polarisation angle from p. A leading ``m`` stands in for a minus sign (``mag=m30``). Files
without a probe token were taken at the configured fixed probe setting. The role comes from a
vocabulary per role matched against the filename; whatever matches no vocabulary is the sample.
Per-scan timestamps from the file headers give each acquisition its elapsed time, which the
drift model needs.

Why this module does not go through ``DataSet``: that machinery is built around sample/reference
pairing of single acquisitions, whereas a polarisation series is a different shape -- one sample,
many angles. The file format itself is NOT re-implemented here: parsing goes through the repo's
standalone ``.acc`` reader (``acquisition_editor.load_file``), so the format has one definition.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import numpy as np
from acquisition_editor import load_file as load_acc_file

from ..core.emitter import polarization_from_reading

__all__ = [
    "AccumulationFile",
    "DEFAULT_ROLE_VOCABULARY",
    "PolarisationSeries",
    "classify_role",
    "load_measurement",
    "load_polarisation_series",
    "parse_key_value_tokens",
    "polarisation_angle_deg_from_filename",
    "read_accumulation_file",
]

DEFAULT_REFERENCE_VOCABULARY = ("gold", "mirror", "ref", "reference")
SAMPLE_ROLE = "sample"
#: role -> filename words. Checked in order; the first role whose vocabulary matches wins, and a
#: file matching none is the sample. The angle reference is opt-in by an explicit word, because a
#: silicon wafer is ALSO a legitimate sample (the validation run measures one as the sample).
DEFAULT_ROLE_VOCABULARY = {
    "angle_reference": ("angleref", "angle-ref", "angle_ref"),
    "channel_reference": ("gold", "mirror"),
}


@dataclass(frozen=True)
class AccumulationFile:
    path: str
    time_ps: np.ndarray            #: (n_samples,)
    scans: np.ndarray              #: (n_scans, n_samples)
    title: str
    scan_timestamps: tuple = ()    #: datetime per scan, from its header; None where absent

    @property
    def averaged(self):
        return self.scans.mean(axis=0)


    @property
    def mean_timestamp(self):
        """The mid-point of the file's scans, or None if any scan lacks a timestamp."""
        if not self.scan_timestamps or any(stamp is None for stamp in self.scan_timestamps):
            return None
        first = self.scan_timestamps[0]
        offsets = [(stamp - first).total_seconds() for stamp in self.scan_timestamps]
        return first + datetime.timedelta(seconds=float(np.mean(offsets)))


@dataclass(frozen=True)
class PolarisationSeries:
    """One object measured at several emitter polarisation angles: one row per acquisition."""

    label: str
    angles_deg: np.ndarray         #: (n_acquisitions,) THz polarisation from p
    time_ps: np.ndarray
    traces: np.ndarray             #: (n_acquisitions, n_samples), repeat-averaged
    scan_counts: np.ndarray
    paths: list
    role: str = SAMPLE_ROLE
    probe_azimuth_deg: float | None = None
    #: (n_acquisitions,) seconds since the earliest acquisition in the directory, or None when
    #: any file lacks timestamps (the drift ramp then runs over acquisition order).
    elapsed_seconds: np.ndarray | None = None
    files: tuple = ()              #: the AccumulationFile behind each row, for the noise model

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


def _scan_timestamp(scan_header):
    """The ``Date and time`` parameter of one scan's header block, or None."""
    for line in scan_header:
        if line.lower().startswith("param date and time,"):
            text = line.split(",", 1)[1].strip()
            try:
                return datetime.datetime.fromisoformat(text)
            except ValueError:
                return None
    return None


def read_accumulation_file(path):
    """Parse one ``.acc`` file into its repeat scans and their timestamps.

    Parsing is delegated to the repo's standalone ``.acc`` reader (``acquisition_editor``), so
    there is one definition of the file format; this function only reshapes its output and pulls
    the per-scan timestamps the drift model needs.
    """
    loaded = load_acc_file(path)
    data = loaded["data"]
    if data.size == 0 or data.shape[1] < 2:
        raise ValueError(
            f"no data rows parsed from {path}; the file does not look like an .acc "
            "accumulation (expected '%'-prefixed headers then two numeric columns)")
    if data.shape[0] < 8:
        raise ValueError(
            f"{path}: only {data.shape[0]} time points parsed; the file does not look like "
            "an .acc accumulation (expected '%'-prefixed headers then two numeric columns)")
    title = ""
    for line in loaded["header"]:
        if line.lower().startswith("title"):
            parts = line.split()
            title = parts[1] if len(parts) > 1 else ""
            break
    return AccumulationFile(
        path=path,
        time_ps=np.asarray(data[:, 0], dtype=float),
        scans=np.asarray(data[:, 1:].T, dtype=float),
        title=title or os.path.basename(path),
        scan_timestamps=tuple(_scan_timestamp(header)
                              for header in loaded["scan_headers"]),
    )


def _is_reference(filename, vocabulary):
    lowered = os.path.basename(filename).lower()
    return any(word in lowered for word in vocabulary)


def classify_role(filename, role_vocabulary=None):
    """The role of a file: the first role whose vocabulary matches, else the sample."""
    vocabulary = DEFAULT_ROLE_VOCABULARY if role_vocabulary is None else role_vocabulary
    for role, words in vocabulary.items():
        if _is_reference(filename, words):
            return role
    return SAMPLE_ROLE


def _float_token(filename, token, delimiter):
    if token is None:
        return None
    return polarisation_angle_deg_from_filename(filename, token, delimiter)


def load_measurement(directory, *, angle_token="mag", probe_token="probe", delimiter="_",
                     role_vocabulary=None, magnet_calibration=None,
                     default_probe_azimuth_deg=None, extension=".acc"):
    """Load every ``.acc`` in a directory into series keyed by ``(role, probe_azimuth_deg)``.

    Rows within a series are in acquisition order when every file has timestamps, else by angle.
    Elapsed times share one origin across the whole directory, so series are comparable.

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

    grouped = {}
    missing_token = []
    for name in candidates:
        angle = polarisation_angle_deg_from_filename(name, angle_token, delimiter)
        if angle is None:
            missing_token.append(name)
            continue
        probe = _float_token(name, probe_token, delimiter)
        probe = default_probe_azimuth_deg if probe is None else probe
        key = (classify_role(name, role_vocabulary), probe)
        grouped.setdefault(key, []).append((float(polarization_from_reading(
                                                angle, magnet_calibration or {0.0: 0.0})),
                                            os.path.join(directory, name)))

    if not grouped:
        raise ValueError(
            f"none of the {len(candidates)} files in {directory} carry a "
            f"'{angle_token}=<degrees>' token. Found filenames like {missing_token[0]!r}. "
            f"Set acquisition.angle_token to whatever the acquisition software writes.")
    if not any(role == SAMPLE_ROLE for role, _ in grouped):
        raise ValueError(
            f"every file in {directory} matched a reference vocabulary "
            f"{dict(role_vocabulary or DEFAULT_ROLE_VOCABULARY)}; nothing is left to treat as "
            "the sample")

    loaded = {key: [(angle, read_accumulation_file(path)) for angle, path in entries]
              for key, entries in grouped.items()}
    stamps = [file.mean_timestamp for entries in loaded.values() for _, file in entries]
    origin = None if any(stamp is None for stamp in stamps) else min(stamps)

    series = {}
    for (role, probe), entries in loaded.items():
        label = role if probe is None else f"{role} @ probe {probe:g} deg"
        lengths = {file.time_ps.size for _, file in entries}
        if len(lengths) != 1:
            raise ValueError(f"{label}: traces have different lengths {sorted(lengths)}; "
                             "a polarisation series must share one time axis")
        if origin is not None:
            entries = sorted(entries, key=lambda entry: entry[1].mean_timestamp)
            elapsed = np.array([(file.mean_timestamp - origin).total_seconds()
                                for _, file in entries])
        else:
            entries = sorted(entries, key=lambda entry: (entry[0], entry[1].path))
            elapsed = None
        files = tuple(file for _, file in entries)
        series[(role, probe)] = PolarisationSeries(
            label=label,
            angles_deg=np.array([angle for angle, _ in entries], dtype=float),
            time_ps=files[0].time_ps,
            traces=np.vstack([file.averaged for file in files]),
            scan_counts=np.array([file.scans.shape[0] for file in files], dtype=int),
            paths=[file.path for file in files],
            role=role, probe_azimuth_deg=probe, elapsed_seconds=elapsed, files=files)
    return series


def load_polarisation_series(directory, *, angle_token="pol", delimiter="_",
                             reference_vocabulary=DEFAULT_REFERENCE_VOCABULARY,
                             extension=".acc"):
    """``(sample_series, reference_series)`` for a directory with one probe setting.

    The two-role view of :func:`load_measurement`; the reference is ``None`` when no file
    matches the reference vocabulary.
    """
    series = load_measurement(directory, angle_token=angle_token, probe_token=None,
                              delimiter=delimiter,
                              role_vocabulary={"reference": tuple(reference_vocabulary)},
                              extension=extension)
    return series.get((SAMPLE_ROLE, None)), series.get(("reference", None))
