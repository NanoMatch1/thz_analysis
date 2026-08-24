"""Per-acquisition tracking: how a THz measurement changes over wall-clock time.

The averaged pipeline (``thz_adapter``) collapses every acquisition in an ``.acc``
file into one mean trace before anything else happens.  That is the right default
for extracting material parameters, but it throws away the axis this module cares
about: **acquisition number / elapsed time**.  A sample that degrades during a
two-hour scan, a laser that warms up, a purge that slowly dries — all of these
appear only when the individual acquisitions are kept apart.

Scope and independence
----------------------
This module reads ``BaseTHzData.raw_data`` directly — the pristine per-scan
traces — so it is **completely independent of the averaged pipeline's state**.
You can run it before, after, or instead of the ``run_me_*`` stages and get the
same answer; nothing here mutates the dataset.  That independence is deliberate:
a drift diagnostic that silently depended on which processing steps had already
run would be useless for diagnosing processing problems.

Structure
---------
``AcquisitionSeries``
    One measurement file's acquisitions on a single frequency grid: timestamps,
    elapsed time, per-scan time traces and per-scan complex spectra.  Built by
    :func:`extract_acquisition_series` (one file) or
    :func:`extract_dataset_acquisitions` (a whole ``DataSet``).

Drift metrics (registry)
    A drift metric maps an ``AcquisitionSeries`` to an ``(n_scans, n_frequencies)``
    real array — one number per acquisition per frequency.  Metrics register
    themselves with :func:`register_drift_metric`, which is the single source of
    truth for both dispatch and presentation (axis label, reference line,
    colormap centring).  Adding a metric therefore means editing exactly one
    place: the metric's own definition.

Plotting
    Every plot function takes a metric *name* and reads its labels from the
    registry, so a newly registered metric is plottable immediately with no
    further changes.

Units
-----
SI internally (seconds, Hz), matching the rest of ``dataset_core``.  Display
helpers convert to ps / THz / minutes at the last moment.
"""

from __future__ import annotations

import datetime
import inspect
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt

from thz_core.thz_core import transfer as core_transfer

from .thz_adapter import _build_symmetric_window

_S_TO_PS = 1e12
_HZ_TO_THZ = 1e-12


# ---------------------------------------------------------------------------
# Measurement parameter headers
#
# The per-scan blocks inside an .acc carry only 'title' and 'Date and time'. The
# instrument settings that matter for comparing files — lock-in sensitivity, time
# constant, accumulations — live in the sibling .dat header, which load_all_data
# skips whenever an .acc of the same name exists (prefer_acc=True). These helpers
# recover them so a sensitivity mismatch between two files can be reported rather
# than silently ratioed.
# ---------------------------------------------------------------------------

_PARAMETER_LINE = re.compile(r'^%param\s+(?P<key>[^,]+),(?P<value>.*)$')


def read_measurement_parameters(filepath: str) -> dict:
    """Parse the ``%param key,value`` header block of a ``.dat`` / ``.acc`` file.

    Values are converted to ``float`` where possible and left as stripped strings
    otherwise.  Reading stops at the first data line, so this is cheap even on a
    large ``.acc``.

    Returns an empty dict if the file cannot be read.
    """
    parameters: dict = {}
    try:
        with open(filepath, 'r') as handle:
            for line in handle:
                if not line.startswith('%'):
                    break  # header block is over
                match = _PARAMETER_LINE.match(line.strip())
                if match is None:
                    continue
                key = match.group('key').strip()
                raw_value = match.group('value').strip()
                try:
                    parameters[key] = float(raw_value)
                except ValueError:
                    parameters[key] = raw_value
    except OSError:
        return {}
    return parameters


def find_measurement_parameters(data_dir: str, filename: str) -> dict:
    """Measurement parameters for *filename*, preferring the sibling ``.dat``.

    ``.dat`` first because its header carries the full instrument settings; the
    ``.acc`` header is the fallback and yields little more than the timestamp.
    """
    stem = os.path.splitext(filename)[0]
    for candidate in (stem + '.dat', filename, stem + '.acc'):
        candidate_path = os.path.join(data_dir, candidate)
        if os.path.isfile(candidate_path):
            parameters = read_measurement_parameters(candidate_path)
            if parameters:
                parameters['source_header_file'] = candidate
                return parameters
    return {}


# ---------------------------------------------------------------------------
# The per-acquisition container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionSeries:
    """Every individual acquisition of ONE measurement file, on one frequency grid.

    Attributes
    ----------
    filename : str
        Key the series was loaded under (the file on disk, not the header title —
        headers keep the name the file had when it was written).
    scan_numbers : np.ndarray, shape (n_scans,)
        Acquisition number from each scan's ``title ... acc N`` header.  This is
        the authoritative count: if acquisitions were culled the header numbers
        show the gap, while a positional index would silently renumber.
    timestamps : list[datetime], length n_scans
        Start time of each acquisition, from the scan header.
    elapsed_seconds : np.ndarray, shape (n_scans,)
        Seconds since the first acquisition of this file.
    time_axis_seconds : np.ndarray, shape (n_time,)
        Shared delay axis, converted from the file's picoseconds to SI seconds.
        Absolute (not re-zeroed), so the axis offset between files is preserved.
    scan_traces : np.ndarray, shape (n_scans, n_time)
        Per-acquisition time-domain traces after baseline subtraction, BEFORE
        windowing — the honest time-domain record.
    window_function : np.ndarray, shape (n_time,)
        The single window applied identically to every acquisition.
    frequency_hz : np.ndarray, shape (n_freq,)
        Frequency grid of ``scan_spectra``.
    scan_spectra : np.ndarray, shape (n_scans, n_freq), complex
        ``rfft`` of each windowed acquisition.
    measurement_parameters : dict
        Instrument settings from the file header (sensitivity, time constant, ...).
    processing : dict
        What was actually done — baseline span, window placement and width, FFT
        length.  Recorded so a figure can be traced back to its processing.
    """

    filename: str
    scan_numbers: np.ndarray
    timestamps: list
    elapsed_seconds: np.ndarray
    time_axis_seconds: np.ndarray
    scan_traces: np.ndarray
    window_function: np.ndarray
    frequency_hz: np.ndarray
    scan_spectra: np.ndarray
    measurement_parameters: dict = field(default_factory=dict)
    processing: dict = field(default_factory=dict)

    # ── shape / axis convenience ────────────────────────────────────────────

    @property
    def n_scans(self) -> int:
        return int(self.scan_spectra.shape[0])

    @property
    def n_frequencies(self) -> int:
        return int(self.scan_spectra.shape[1])

    @property
    def elapsed_minutes(self) -> np.ndarray:
        return self.elapsed_seconds / 60.0

    @property
    def elapsed_hours(self) -> np.ndarray:
        return self.elapsed_seconds / 3600.0

    @property
    def duration_seconds(self) -> float:
        return float(self.elapsed_seconds[-1]) if self.n_scans else 0.0

    @property
    def frequency_thz(self) -> np.ndarray:
        return self.frequency_hz * _HZ_TO_THZ

    @property
    def time_axis_ps(self) -> np.ndarray:
        return self.time_axis_seconds * _S_TO_PS

    # ── derived arrays ──────────────────────────────────────────────────────

    @property
    def amplitude(self) -> np.ndarray:
        """``|Y|`` per acquisition per frequency, shape (n_scans, n_freq)."""
        return np.abs(self.scan_spectra)

    @property
    def mean_spectrum(self) -> np.ndarray:
        """Coherent mean over all acquisitions — what the averaged pipeline sees."""
        return self.scan_spectra.mean(axis=0)

    @property
    def mean_trace(self) -> np.ndarray:
        """Mean time-domain trace over all acquisitions (baseline-subtracted)."""
        return self.scan_traces.mean(axis=0)

    @property
    def peak_amplitude_per_scan(self) -> np.ndarray:
        """Signed amplitude at the mean trace's peak bin, per acquisition.

        Read at a FIXED bin (the mean trace's peak) rather than each scan's own
        maximum: a per-scan argmax would hop between neighbouring bins on noise
        and manufacture drift that is not there.
        """
        peak_index = int(np.argmax(np.abs(self.mean_trace)))
        return self.scan_traces[:, peak_index]

    @property
    def peak_time_seconds_per_scan(self) -> np.ndarray:
        """Time of each acquisition's own extremum — coarse timing-drift check.

        Quantised to the sample step, so it only resolves drift larger than ``dt``.
        For sub-sample timing drift use the ``phase_deviation`` metric instead,
        whose slope in frequency is the delay.
        """
        peak_indices = np.argmax(np.abs(self.scan_traces), axis=1)
        return self.time_axis_seconds[peak_indices]

    # ── lookups ─────────────────────────────────────────────────────────────

    def frequency_index(self, frequency_thz: float) -> int:
        """Index of the bin nearest *frequency_thz*."""
        return int(np.argmin(np.abs(self.frequency_thz - float(frequency_thz))))

    def amplitude_at(self, frequency_thz: float) -> np.ndarray:
        """``|Y|`` vs acquisition at the bin nearest *frequency_thz*."""
        return self.amplitude[:, self.frequency_index(frequency_thz)]

    def spectrum_at(self, frequency_thz: float) -> np.ndarray:
        """Complex ``Y`` vs acquisition at the bin nearest *frequency_thz*."""
        return self.scan_spectra[:, self.frequency_index(frequency_thz)]

    def trusted_mask(self, snr_thresh_db: float = 10.0, tail_fraction: float = 0.25) -> np.ndarray:
        """Boolean (n_freq,) mask of bins above the noise floor of the MEAN spectrum.

        Uses the same high-frequency-tail noise floor estimator as the pipeline's
        SNR mask (``thz_core.transfer``), so "trusted" means the same thing here
        as it does downstream.
        """
        magnitude = np.abs(self.mean_spectrum)
        floor = core_transfer._noise_floor(magnitude, tail_fraction)
        snr_db = 20.0 * np.log10(np.maximum(magnitude, floor) / floor)
        return snr_db >= float(snr_thresh_db)

    def __repr__(self) -> str:
        return (
            f"<AcquisitionSeries {self.filename}: {self.n_scans} acquisitions over "
            f"{self.duration_seconds / 60:.1f} min, {self.n_frequencies} freq bins>"
        )

    def describe(self) -> str:
        """Multi-line human summary of the series and how it was processed."""
        cadence = (f"{np.median(np.diff(self.elapsed_seconds)):.1f} s median"
                   if self.n_scans > 1 else "n/a (single acquisition)")
        half_width_ps = self.processing.get('half_width_ps', float('nan'))
        window_center_ps = self.processing.get('window_center_ps', float('nan'))
        lines = [
            f"AcquisitionSeries '{self.filename}'",
            f"  acquisitions      : {self.n_scans} "
            f"(header numbers {int(self.scan_numbers[0])}..{int(self.scan_numbers[-1])})",
            f"  elapsed           : {self.duration_seconds / 60:.1f} min "
            f"({self.timestamps[0]} -> {self.timestamps[-1]})",
            f"  cadence           : {cadence}",
            f"  delay axis        : {self.time_axis_ps[0]:.2f}..{self.time_axis_ps[-1]:.2f} ps "
            f"({self.time_axis_seconds.size} pts)",
            f"  window            : {self.processing.get('window_type')} "
            f"+/-{half_width_ps:.2f} ps at {window_center_ps:.2f} ps",
            f"  frequency grid    : 0..{self.frequency_thz[-1]:.2f} THz, "
            f"df {np.median(np.diff(self.frequency_thz)) * 1000:.1f} GHz "
            f"(true resolution {self.processing.get('resolution_thz', float('nan')):.3f} THz)",
        ]
        missing = missing_acquisition_numbers(self)
        if missing:
            lines.append(f"  MISSING acq numbers: {missing}")
        sensitivity = self.measurement_parameters.get('Lockin sensitivity')
        if sensitivity is not None:
            lines.append(f"  lock-in sensitivity: {sensitivity}")
        return "\n".join(lines)


def missing_acquisition_numbers(series: AcquisitionSeries) -> list:
    """Header acquisition numbers absent from the series (culled or failed scans).

    An empty list means the record is contiguous.  A non-empty list is a warning
    that ``elapsed_seconds`` has gaps, so a drift trend fitted against acquisition
    *index* would be distorted — fit against elapsed time instead.
    """
    numbers = [int(value) for value in series.scan_numbers if np.isfinite(value)]
    if not numbers:
        return []
    return sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _scan_holder(data_obj):
    """The THzData carrying the per-scan list, unwrapping a reflection container."""
    second_reflection = getattr(data_obj, 'second_reflection', None)
    return second_reflection if second_reflection is not None else data_obj


def extract_acquisition_series(
    data_obj,
    *,
    filename: str | None = None,
    half_width_ps: float | None = None,
    window_type: str = 'hann',
    window_alpha: float = 1.0,
    n_fft: int = 2048,
    baseline_ps: float | None = 1.0,
    region_ps: tuple | None = None,
    measurement_parameters: dict | None = None,
) -> AcquisitionSeries:
    """Build an :class:`AcquisitionSeries` from one ``THzData``'s individual scans.

    Processing, in order, and why each step is the way it is:

    1. **Per-acquisition baseline subtraction.**  The mean of the first
       ``baseline_ps`` of each trace is removed from that trace.  The lock-in DC
       offset wanders over hours; left in, that wander is a slow ramp in every
       trace and would show up as pure low-frequency "drift" that has nothing to
       do with the THz field.  Pass ``baseline_ps=None`` to skip.
    2. **One window position for the whole file.**  The peak is located on the
       MEAN trace, and the same window sits at that bin for every acquisition.
       Locating the peak per-scan would let the window chase noise, adding
       scan-to-scan jitter that then reads as drift.
    3. **One identical window shape**, built by the same helper the pipeline uses
       (``_build_symmetric_window``), so windowing means the same thing here.
    4. **One shared ``n_fft``** for every acquisition, so all spectra land on one
       frequency grid and can be compared bin by bin.

    Parameters
    ----------
    half_width_ps : float or None
        Symmetric window half-width.  ``None`` (default) auto-selects the widest
        half-width that fits inside the record without clipping either edge, and
        records it in ``processing``.  These reflection records are short and
        asymmetric about the pulse, so the auto value is usually what you want.
    region_ps : tuple or None
        ``(start_ps, end_ps)`` to constrain the peak search on the mean trace.
    n_fft : int
        Shared FFT length.  Must be at least the record length.  Zero-padding
        above the record length interpolates the spectrum; it does not add
        resolution (the true resolution is reported in ``processing``).

    Returns
    -------
    AcquisitionSeries
    """
    holder = _scan_holder(data_obj)
    scan_list = getattr(holder, 'data_list', None)
    if not scan_list:
        raise ValueError(
            f"'{filename or getattr(holder, 'filename', '<unknown>')}' has no per-scan "
            f"data_list. Per-acquisition tracking needs an .acc file (load with "
            f"prefer_acc=True, the default)."
        )

    filename = filename or getattr(holder, 'filename', 'unknown_file')

    # Source files store the delay axis in picoseconds; go to SI immediately.
    time_axis_seconds = np.asarray(scan_list[0].raw_data[:, 0], dtype=float) / _S_TO_PS
    n_time = time_axis_seconds.size

    traces = []
    for scan in scan_list:
        scan_array = np.asarray(scan.raw_data, dtype=float)
        if scan_array.shape[0] != n_time:
            raise ValueError(
                f"'{filename}' acquisition {scan.scan_index} has {scan_array.shape[0]} "
                f"samples but the first has {n_time}; the acquisitions do not share a "
                f"time axis and cannot be stacked."
            )
        traces.append(scan_array[:, 1])
    scan_traces = np.array(traces, dtype=float)

    dt_seconds = float(np.median(np.diff(time_axis_seconds)))

    # --- 1. per-acquisition baseline ---
    baseline_samples = 0
    if baseline_ps is not None:
        baseline_samples = int(round(float(baseline_ps) * 1e-12 / dt_seconds))
        baseline_samples = max(1, min(baseline_samples, n_time // 2))
        scan_traces = scan_traces - scan_traces[:, :baseline_samples].mean(axis=1, keepdims=True)

    # --- 2. one window position, from the mean trace ---
    mean_trace = scan_traces.mean(axis=0)
    if region_ps is not None and None not in region_ps:
        in_region = ((time_axis_seconds >= region_ps[0] / _S_TO_PS)
                     & (time_axis_seconds <= region_ps[1] / _S_TO_PS))
        search_indices = np.where(in_region)[0]
        if search_indices.size < 4:
            raise ValueError(
                f"'{filename}': region {region_ps} ps selects fewer than 4 samples."
            )
    else:
        search_indices = np.arange(n_time)
    peak_index = int(search_indices[int(np.argmax(np.abs(mean_trace[search_indices])))])

    # --- 3. one window shape ---
    widest_fit_samples = min(peak_index, n_time - 1 - peak_index)
    if half_width_ps is None:
        half_width_samples = widest_fit_samples
        auto_selected = True
    else:
        half_width_samples = int(round(float(half_width_ps) * 1e-12 / dt_seconds))
        auto_selected = False
    if half_width_samples < 2:
        raise ValueError(
            f"'{filename}': window half-width resolves to {half_width_samples} samples "
            f"at dt={dt_seconds * _S_TO_PS:.4f} ps — too narrow to be meaningful."
        )
    clipped = half_width_samples > widest_fit_samples

    window_length = 2 * half_width_samples + 1
    window_core = _build_symmetric_window(window_length, window_type, window_alpha)

    low = peak_index - half_width_samples
    high = peak_index + half_width_samples + 1
    data_low, data_high = max(low, 0), min(high, n_time)
    core_low = data_low - low
    window_function = np.zeros(n_time)
    window_function[data_low:data_high] = window_core[core_low:core_low + (data_high - data_low)]

    # --- 4. one shared FFT grid ---
    if n_fft < n_time:
        raise ValueError(
            f"'{filename}': n_fft={n_fft} is shorter than the {n_time}-sample record and "
            f"would truncate the pulse. Use n_fft >= {n_time}."
        )
    windowed = scan_traces * window_function
    scan_spectra = np.fft.rfft(windowed, n=n_fft, axis=1)
    frequency_hz = np.fft.rfftfreq(n_fft, dt_seconds)

    # Timestamps / acquisition numbers, falling back to position when a header
    # lacks them so a partially-annotated file still produces a usable series.
    timestamps = [scan.timestamp for scan in scan_list]
    usable_timestamps = all(isinstance(stamp, datetime.datetime) for stamp in timestamps)
    if usable_timestamps:
        elapsed_seconds = np.array(
            [(stamp - timestamps[0]).total_seconds() for stamp in timestamps], dtype=float
        )
    else:
        print(
            f"[extract_acquisition_series] '{filename}': headers carry no parsable "
            f"timestamps; elapsed time falls back to acquisition index."
        )
        elapsed_seconds = np.arange(len(scan_list), dtype=float)
    scan_numbers = np.array(
        [scan.scan_index if scan.scan_index is not None else index + 1
         for index, scan in enumerate(scan_list)],
        dtype=float,
    )

    record_seconds = window_length * dt_seconds
    processing = dict(
        baseline_ps=baseline_ps,
        baseline_samples=baseline_samples,
        window_type=window_type,
        window_alpha=window_alpha,
        half_width_ps=half_width_samples * dt_seconds * _S_TO_PS,
        half_width_samples=half_width_samples,
        half_width_auto=auto_selected,
        window_center_ps=float(time_axis_seconds[peak_index] * _S_TO_PS),
        window_clipped=bool(clipped),
        window_length=window_length,
        n_fft=int(n_fft),
        dt_ps=dt_seconds * _S_TO_PS,
        resolution_thz=1.0 / record_seconds * _HZ_TO_THZ,
    )

    clip_note = "  *** window CLIPPED at the record edge ***" if clipped else ""
    print(
        f"[extract_acquisition_series] '{filename}': {len(scan_list)} acquisitions, "
        f"{processing['half_width_ps']:.2f} ps half-width "
        f"{'(auto)' if auto_selected else '(explicit)'} at "
        f"{processing['window_center_ps']:.2f} ps, resolution "
        f"{processing['resolution_thz']:.3f} THz.{clip_note}"
    )

    return AcquisitionSeries(
        filename=filename,
        scan_numbers=scan_numbers,
        timestamps=timestamps,
        elapsed_seconds=elapsed_seconds,
        time_axis_seconds=time_axis_seconds,
        scan_traces=scan_traces,
        window_function=window_function,
        frequency_hz=frequency_hz,
        scan_spectra=scan_spectra,
        measurement_parameters=measurement_parameters or {},
        processing=processing,
    )


def extract_dataset_acquisitions(
    dataset,
    config: dict | None = None,
    files=None,
) -> dict:
    """Extract an :class:`AcquisitionSeries` for every file in *dataset*.

    Reads its settings from ``config['acquisition_tracking']`` (falling back to
    ``dataset.config``, the project's single-source-of-truth convention), with
    ``config['fft']['n_fft']`` and ``config['window']`` honoured so the drift view
    and the averaged pipeline can be made to window identically when you want them
    to agree.

    Parameters
    ----------
    files : None | str | callable | iterable
        Selector, matching the convention in ``display``: ``None`` for every file,
        a regex/substring, a predicate, or an explicit collection of filenames.

    Returns
    -------
    dict[str, AcquisitionSeries]
        Keyed by filename, in dataset order.  Files without per-scan data are
        skipped with a printed note rather than raising, so one ``.dat`` in a
        folder of ``.acc`` files does not abort the run.
    """
    config = config or getattr(dataset, 'config', None) or {}
    tracking_config = config.get('acquisition_tracking', {})
    window_config = config.get('window', {})
    data_dir = getattr(dataset, 'file_dir', '')

    def selected(filename: str) -> bool:
        if files is None:
            return True
        if callable(files):
            return bool(files(filename))
        if isinstance(files, str):
            return re.search(files, filename) is not None
        return filename in set(files)

    series_by_filename: dict = {}
    for filename, data_obj in dataset.data.data_dict.items():
        if not selected(filename):
            continue
        try:
            series_by_filename[filename] = extract_acquisition_series(
                data_obj,
                filename=filename,
                half_width_ps=tracking_config.get('half_width_ps',
                                                  window_config.get('half_width_ps')),
                window_type=window_config.get('type', 'hann'),
                window_alpha=float(window_config.get('alpha', 1.0)),
                n_fft=int(config.get('fft', {}).get('n_fft', 2048)),
                baseline_ps=tracking_config.get('baseline_ps', 1.0),
                region_ps=tracking_config.get('region_ps'),
                measurement_parameters=find_measurement_parameters(data_dir, filename),
            )
        except ValueError as error:
            print(f"[extract_dataset_acquisitions] skipping '{filename}': {error}")

    _report_parameter_mismatches(series_by_filename)
    return series_by_filename


def _report_parameter_mismatches(series_by_filename: dict) -> None:
    """Warn when files that might be ratioed were taken on different settings.

    A lock-in sensitivity change between a sample and its reference is the classic
    silent factor-of-two in a transfer function.  Whether it actually rescales the
    recorded values depends on the acquisition software, so this reports rather
    than corrects — diagnose, don't silently fix.
    """
    watched_parameters = ('Lockin sensitivity', 'Lockin time constant')
    for parameter in watched_parameters:
        values = {
            filename: series.measurement_parameters.get(parameter)
            for filename, series in series_by_filename.items()
            if series.measurement_parameters.get(parameter) is not None
        }
        if len(set(values.values())) > 1:
            print(
                f"[acquisition_tracking] NOTE: '{parameter}' differs across files: "
                f"{values}. Amplitudes are only directly comparable if the acquisition "
                f"software records absolute units."
            )


# ---------------------------------------------------------------------------
# Drift-metric registry
#
# A metric maps an AcquisitionSeries to an (n_scans, n_frequencies) real array.
# Registering it here is the ONLY place it needs to be declared: dispatch, axis
# labels, reference lines and colormap centring all read from this entry, so the
# plot functions below need no per-metric knowledge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftMetric:
    name: str
    compute: Callable
    label: str
    description: str
    reference_value: float | None   # the "nothing changed" value: line / colormap centre
    diverging: bool                 # True -> symmetric colormap about reference_value


DRIFT_METRICS: dict = {}


def register_drift_metric(
    *,
    name: str,
    label: str,
    description: str,
    reference_value: float | None = None,
    diverging: bool = True,
):
    """Decorator registering a drift metric under *name*.

    The decorated function takes ``(series, **options)`` and returns a real
    ``(n_scans, n_frequencies)`` array.
    """
    def decorator(function: Callable) -> Callable:
        DRIFT_METRICS[name] = DriftMetric(
            name=name,
            compute=function,
            label=label,
            description=description,
            reference_value=reference_value,
            diverging=diverging,
        )
        return function
    return decorator


def available_drift_metrics() -> list:
    """Registered metric names, and print their descriptions."""
    print("Available drift metrics:")
    for name, metric in DRIFT_METRICS.items():
        print(f"  {name:<26} {metric.description}")
    return list(DRIFT_METRICS)


def compute_drift_metric(series: AcquisitionSeries, metric: str = 'amplitude_ratio', **options):
    """Compute a registered metric; returns ``(values, DriftMetric)``.

    Options are filtered against the metric function's own signature, so a caller
    can pass one option dict to several metrics and each takes only what it
    understands.  This keeps the promise the registry makes: a metric declares its
    parameters in exactly one place — its own signature — and nothing else in the
    module (dispatch, plotting, the summary) needs to know which metric accepts
    what.
    """
    if metric not in DRIFT_METRICS:
        raise KeyError(
            f"Unknown drift metric '{metric}'. Registered: {sorted(DRIFT_METRICS)}."
        )
    entry = DRIFT_METRICS[metric]
    parameters = inspect.signature(entry.compute).parameters
    accepts_everything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    accepted = options if accepts_everything else {
        key: value for key, value in options.items() if key in parameters
    }
    return entry.compute(series, **accepted), entry


def baseline_spectrum(series: AcquisitionSeries, baseline_scans: int = 8) -> np.ndarray:
    """Coherent mean of the first *baseline_scans* acquisitions.

    Coherent (complex) rather than ``|Y|``-mean so the noise averages down the same
    way it does in the pipeline's averaged trace.  Averaging several acquisitions
    rather than taking scan 1 alone matters: a single acquisition's noise would be
    baked into the denominator of every ratio and would tilt the whole series.
    """
    count = int(np.clip(baseline_scans, 1, series.n_scans))
    return series.scan_spectra[:count].mean(axis=0)


def cumulative_average_spectra(series: AcquisitionSeries) -> np.ndarray:
    """Running coherent mean of acquisitions 1..N, shape (n_scans, n_freq), complex.

    Row ``i`` is what the averaged pipeline would have produced had the run stopped
    after ``i+1`` acquisitions.  Computed in the frequency domain, which is exactly
    equal to FFT-of-the-running-mean-trace because every acquisition shares one
    window and one ``n_fft`` and the FFT is linear (asserted in the tests).
    """
    running_totals = np.cumsum(series.scan_spectra, axis=0)
    counts = np.arange(1, series.n_scans + 1, dtype=float)[:, None]
    return running_totals / counts


@register_drift_metric(
    name='amplitude_ratio',
    label='|Y| / |Y| at run start',
    description='Per-acquisition amplitude divided by the average of the first few '
                'acquisitions. Flat at 1.0 means stable; a slope is drift.',
    reference_value=1.0,
)
def amplitude_ratio_to_baseline(series: AcquisitionSeries, *, baseline_scans: int = 8) -> np.ndarray:
    """``|Y(f, scan)| / |Y_baseline(f)|`` — the primary drift view."""
    baseline = np.abs(baseline_spectrum(series, baseline_scans))
    floor = core_transfer._noise_floor(np.abs(series.mean_spectrum), 0.25)
    with np.errstate(divide='ignore', invalid='ignore'):
        return series.amplitude / np.maximum(baseline, floor)


@register_drift_metric(
    name='phase_deviation',
    label='arg(Y) - arg(Y at run start)  (rad)',
    description='Per-acquisition phase relative to the run start, wrapped to +/-pi. '
                'A slope in FREQUENCY is a timing drift (delay = -slope/2pi).',
    reference_value=0.0,
)
def phase_deviation_from_baseline(series: AcquisitionSeries, *, baseline_scans: int = 8) -> np.ndarray:
    """``arg(Y(f, scan) / Y_baseline(f))``, wrapped to (-pi, pi].

    This is the sub-sample timing diagnostic: the time-domain peak index only
    resolves drift larger than ``dt``, whereas a delay ``tau`` appears here as a
    phase ramp ``-2*pi*f*tau`` long before it moves a sample.
    """
    baseline = baseline_spectrum(series, baseline_scans)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.angle(series.scan_spectra / baseline)


@register_drift_metric(
    name='cumulative_ratio',
    label='|cumulative mean| / |final mean|',
    description='Amplitude of the running average after N acquisitions, relative to '
                'the final average. Shows how the answer moved as the run went on.',
    reference_value=1.0,
)
def cumulative_average_ratio(series: AcquisitionSeries) -> np.ndarray:
    """``|mean(Y_1..Y_N)| / |mean(Y_1..Y_total)|`` per frequency."""
    cumulative = cumulative_average_spectra(series)
    final = np.abs(cumulative[-1])
    floor = core_transfer._noise_floor(final, 0.25)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(cumulative) / np.maximum(final, floor)


@register_drift_metric(
    name='cumulative_deviation',
    label='|cumulative mean - final| / |final|',
    description='Fractional distance of the running average from the final answer — '
                'read off how many acquisitions were needed to converge.',
    reference_value=0.0,
    diverging=False,
)
def cumulative_average_deviation(series: AcquisitionSeries) -> np.ndarray:
    """Complex distance between the running average and the final average.

    Complex, not just amplitude: an average can be converged in magnitude while
    still moving in phase, and for anything downstream of a transfer function the
    phase is what matters.
    """
    cumulative = cumulative_average_spectra(series)
    final = cumulative[-1]
    floor = core_transfer._noise_floor(np.abs(final), 0.25)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(cumulative - final) / np.maximum(np.abs(final), floor)


# ---------------------------------------------------------------------------
# Quantitative summary
# ---------------------------------------------------------------------------


def fitted_delay_seconds(
    series: AcquisitionSeries,
    *,
    band_thz: tuple = (0.3, 2.0),
    baseline_scans: int = 8,
    snr_thresh_db: float = 15.0,
) -> np.ndarray:
    """Sub-sample delay of each acquisition relative to the run start, in seconds.

    The time-domain peak index (``peak_time_seconds_per_scan``) is quantised to the
    sample step — 50 fs on a typical record here — so it stays pinned to one bin
    while the pulse walks most of the way across it.  A delay ``tau`` is instead
    visible immediately as a phase ramp ``arg(Y_n/Y_baseline) = -2*pi*f*tau``, and
    fitting that slope over the trusted band resolves it to well under a
    femtosecond.

    This matters because a delay that never moves the peak bin can still be large
    enough to change a reflection inversion: for a near-mirror sample the transfer
    phase is what carries the answer.

    Returns
    -------
    np.ndarray, shape (n_scans,)
        Delay in seconds, positive meaning the pulse arrived LATER than at the
        start of the run.  Element 0 is ~0 by construction.
    """
    phase, _entry = compute_drift_metric(
        series, 'phase_deviation', baseline_scans=baseline_scans
    )
    band = (series.trusted_mask(snr_thresh_db)
            & (series.frequency_thz >= band_thz[0])
            & (series.frequency_thz <= band_thz[1]))
    if band.sum() < 3:
        raise ValueError(
            f"'{series.filename}': only {band.sum()} trusted bins in "
            f"{band_thz} THz — cannot fit a delay. Widen band_thz or lower "
            f"snr_thresh_db."
        )
    # np.polyfit fits every column at once: (n_band, n_scans) -> (2, n_scans).
    slopes = np.polyfit(series.frequency_hz[band], phase[:, band].T, 1)[0]
    return -slopes / (2.0 * np.pi)


def delay_drift_rate(series: AcquisitionSeries, **options) -> dict:
    """Fitted timing drift of a run: rate, total, and scan-to-scan scatter.

    Keys are all in seconds (``rate_seconds_per_hour`` per hour).  ``scatter_seconds``
    is the standard deviation of the scan-to-scan DIFFERENCE, which is the honest
    noise floor of the delay measurement — compare the total drift against it to
    judge whether the trend is real.
    """
    delays = fitted_delay_seconds(series, **options)
    hours = series.elapsed_hours
    rate, intercept = np.polyfit(hours, delays, 1)
    edge = max(1, series.n_scans // 10)
    return dict(
        delays_seconds=delays,
        rate_seconds_per_hour=float(rate),
        intercept_seconds=float(intercept),
        total_seconds=float(delays[-edge:].mean() - delays[:edge].mean()),
        scatter_seconds=float(np.std(np.diff(delays))) if series.n_scans > 2 else 0.0,
        sample_step_seconds=float(series.processing['dt_ps'] * 1e-12),
    )


# ---------------------------------------------------------------------------
# Purge equilibration
#
# A leaky purge box exchanging its gas approaches equilibrium as 1 - exp(-t/tau),
# tau = V/Q. That exponential signature is what separates a PURGE transient from
# sample degradation or instrument drift: a purge saturates, degradation does not.
# ---------------------------------------------------------------------------


def _saturating_exponential(t, plateau, span, tau):
    """``y = plateau - span * exp(-t/tau)`` — a box approaching equilibrium."""
    return plateau - span * np.exp(-t / tau)


def _per_point_noise(values: np.ndarray) -> float:
    """Point-to-point noise sigma, estimated from successive differences.

    ``std(diff(y))`` measures the scatter of a *difference* of two independent
    points, whose variance is 2*sigma^2 — hence the sqrt(2). Using successive
    differences rather than the scatter about the mean makes this insensitive to
    the very trend we are trying to fit.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0
    return float(np.std(np.diff(values)) / np.sqrt(2.0))


def fit_exponential_equilibration(
    elapsed_seconds: np.ndarray,
    values: np.ndarray,
    *,
    noise_estimate: float | None = None,
    minimum_improvement: float = 2.0,
    fixed_tau_seconds: float | None = None,
) -> dict:
    """Fit a saturating exponential and judge whether a transient is really there.

    Pure function — arrays in, dict out — so it can be tested and reused on any
    observable, not just the ones this module happens to compute.

    The verdict matters more than the fit.  ``curve_fit`` always returns *a*
    number: on a flat series it happily reports a negative or absurd ``tau``.
    ``transient_detected`` is therefore True only when all three hold:

      * ``tau`` is finite, positive, and shorter than the record (a "time constant"
        far longer than the data is an unconstrained straight line in disguise),
      * the exponential beats a straight line by at least ``minimum_improvement``
        in sum-of-squares, and
      * the fitted span exceeds ``noise_estimate`` (default: the scan-to-scan
        scatter) by 3x, so a fit to noise is rejected.

    Fixing tau from a longer run
    ----------------------------
    ``fixed_tau_seconds`` fits only the plateau and span, holding tau at a value
    measured elsewhere.  This matters more than it sounds: **a short run has no
    lever arm to establish its own tau**, so a run sitting on the slow tail of a
    transient will fit as a straight line and be declared equilibrated.  Measure
    tau once on a long run after deliberately disturbing the chamber, then apply
    that tau to every short run afterwards.

    With tau supplied, the first two guards are skipped — they exist to decide
    *whether* an exponential is the right model, and that question has already been
    answered by the run that measured tau.  Detection then rests on span vs noise.

    Returns
    -------
    dict
        ``tau_seconds``, ``plateau``, ``span``, ``exponential_residual``,
        ``linear_residual``, ``improvement_over_linear``, ``tau_fixed``,
        ``transient_detected``, ``reason`` (why not, when not detected).
    """
    from scipy.optimize import curve_fit

    elapsed_seconds = np.asarray(elapsed_seconds, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(elapsed_seconds) & np.isfinite(values)
    elapsed_seconds, values = elapsed_seconds[finite], values[finite]
    if values.size < 4:
        return dict(tau_seconds=float('nan'), plateau=float('nan'), span=float('nan'),
                    exponential_residual=float('nan'), linear_residual=float('nan'),
                    improvement_over_linear=float('nan'), tau_fixed=False,
                    transient_detected=False, reason='fewer than 4 usable points')

    if noise_estimate is None:
        noise_estimate = _per_point_noise(values)

    linear_coefficients = np.polyfit(elapsed_seconds, values, 1)
    linear_residual = float(np.sum((values - np.polyval(linear_coefficients, elapsed_seconds)) ** 2))

    record_seconds = float(elapsed_seconds[-1] - elapsed_seconds[0]) or 1.0
    tau_fixed = fixed_tau_seconds is not None and np.isfinite(fixed_tau_seconds) \
        and fixed_tau_seconds > 0
    try:
        if tau_fixed:
            def model(t, plateau, span):
                return _saturating_exponential(t, plateau, span, float(fixed_tau_seconds))
            parameters, _covariance = curve_fit(
                model, elapsed_seconds, values,
                p0=[values[-1], values[-1] - values[0]], maxfev=20000,
            )
            plateau, span = (float(value) for value in parameters)
            tau = float(fixed_tau_seconds)
        else:
            parameters, _covariance = curve_fit(
                _saturating_exponential, elapsed_seconds, values,
                p0=[values[-1], values[-1] - values[0], record_seconds / 3.0],
                maxfev=20000,
            )
            plateau, span, tau = (float(value) for value in parameters)
    except (RuntimeError, TypeError, ValueError):
        return dict(tau_seconds=float('nan'), plateau=float('nan'), span=float('nan'),
                    exponential_residual=float('nan'), linear_residual=linear_residual,
                    improvement_over_linear=float('nan'), tau_fixed=tau_fixed,
                    transient_detected=False, reason='exponential fit did not converge')

    exponential_residual = float(np.sum(
        (values - _saturating_exponential(elapsed_seconds, plateau, span, tau)) ** 2
    ))
    improvement = (linear_residual / exponential_residual
                   if exponential_residual > 0 else float('inf'))

    # With tau supplied the model-selection guards are skipped: they answer "is an
    # exponential the right shape?", which the run that measured tau already settled.
    reason = ''
    if not tau_fixed:
        if not np.isfinite(tau) or tau <= 0:
            reason = f'fitted tau is not positive ({tau / 60:.1f} min) — no saturation present'
        elif tau > 5 * record_seconds:
            reason = (f'fitted tau ({tau / 60:.1f} min) far exceeds the '
                      f'{record_seconds / 60:.1f} min record — unconstrained, '
                      f'indistinguishable from a straight line')
        elif improvement < minimum_improvement:
            reason = (f'exponential beats linear by only {improvement:.1f}x '
                      f'(need {minimum_improvement:.1f}x)')
    if not reason and abs(span) <= 3 * noise_estimate:
        reason = (f'fitted span {abs(span):.4g} is within 3x the scatter '
                  f'{noise_estimate:.4g} — consistent with noise')

    return dict(
        tau_seconds=tau, plateau=plateau, span=span,
        exponential_residual=exponential_residual, linear_residual=linear_residual,
        improvement_over_linear=improvement, tau_fixed=tau_fixed,
        transient_detected=not reason, reason=reason,
    )


def time_to_equilibrate(tau_seconds: float, fraction: float = 0.99) -> float:
    """Seconds to reach *fraction* of the way to the plateau: ``-tau*ln(1-fraction)``.

    The number that actually governs lab practice — one time constant is only 63%,
    and settling to 1% costs 4.6 time constants.
    """
    if not np.isfinite(tau_seconds) or tau_seconds <= 0:
        return float('nan')
    return -tau_seconds * np.log(1.0 - float(fraction))


def residual_drift_over_window(
    fit: dict, start_seconds: float, window_seconds: float
) -> float:
    """How much a fitted exponential will still move over a measurement window.

    ``d(t, T) = |span| * exp(-t/tau) * (1 - exp(-T/tau))``

    This — not the distance to the plateau — is the systematic error the purge
    contributes to a measurement of length ``T`` started at time ``t``. A run can
    be far from its plateau and still be perfectly usable if it is slow enough that
    nothing moves while you measure.
    """
    tau = fit.get('tau_seconds', float('nan'))
    if not np.isfinite(tau) or tau <= 0:
        return 0.0
    return float(abs(fit['span']) * np.exp(-start_seconds / tau)
                 * (1.0 - np.exp(-window_seconds / tau)))


def time_until_acceptable_drift(
    fit: dict, window_seconds: float, acceptable_drift: float
) -> float:
    """Invert :func:`residual_drift_over_window` for the wait time, in seconds.

    Solves ``|span|*exp(-t/tau)*(1-exp(-T/tau)) = acceptable_drift`` for ``t``.
    Returns 0.0 when the criterion already holds at ``t = 0``.
    """
    tau = fit.get('tau_seconds', float('nan'))
    if not np.isfinite(tau) or tau <= 0 or acceptable_drift <= 0:
        return 0.0
    initial_drift = abs(fit['span']) * (1.0 - np.exp(-window_seconds / tau))
    if initial_drift <= acceptable_drift:
        return 0.0
    return float(-tau * np.log(acceptable_drift / initial_drift))


def purge_settling_assessment(
    series: AcquisitionSeries,
    *,
    frequency_thz: float = 2.0,
    measurement_window_minutes: float = 30.0,
    acceptable_drift_fraction: float = 0.005,
    baseline_scans: int = 8,
    known_tau_seconds: float | None = None,
) -> dict:
    """When does this run become clean enough to use? Returns a CUT INDEX.

    Design of the criterion
    -----------------------
    The intuitive metric — "within X% of the extrapolated plateau" — is reported
    (``settled_fraction``), but it is not what decides usability, for two reasons.
    The plateau is an extrapolation, badly constrained until roughly one time
    constant has been observed; and distance-from-plateau says nothing about how
    long you intend to measure. A slowly-drifting run far from its plateau can be
    perfectly usable.

    The criterion used instead is **predicted residual drift over the measurement
    window** (:func:`residual_drift_over_window`): the systematic error the purge
    will actually inject into an average of length ``measurement_window_minutes``.

    ``acceptable_drift_fraction`` defaults to 0.5% — the bottom of the measured
    instrument floor (lab notebook F28 puts it at 0.5–2%), so below this the purge
    has stopped being the limiting error term rather than merely being small.

    ``purge_limited`` compares that residual drift against the *statistical*
    uncertainty of the same average. Once the drift falls below the noise, waiting
    longer buys nothing measurable — that is the point of diminishing returns, and
    it is self-calibrating rather than an arbitrary threshold.

    Short runs need a tau from elsewhere
    ------------------------------------
    **Pass ``known_tau_seconds`` for any run shorter than about one time constant.**
    A short run has no lever arm to fit its own tau: sitting on the slow tail of a
    transient, it fits as a straight line and is reported EQUILIBRATED even though
    it is still moving. Measure tau once on a long run started right after
    deliberately disturbing the chamber, then reuse it. ``record_tau_ratio`` in the
    result flags when the record is too short to have established its own.

    Returns
    -------
    dict
        ``first_acceptable_index`` / ``first_acceptable_minutes`` — the cut: use
        acquisitions from here on. ``None`` if the run never becomes acceptable.
        Plus ``fit``, ``settled_fraction``, ``residual_drift``, ``noise_floor``,
        ``purge_limited``, ``record_tau_ratio`` (fit trustworthiness — below ~1 the
        plateau is extrapolated beyond the data), and ``verdict``.
    """
    ratio, _entry = compute_drift_metric(
        series, 'amplitude_ratio', baseline_scans=baseline_scans
    )
    index = series.frequency_index(frequency_thz)
    values = ratio[:, index]
    fit = fit_exponential_equilibration(
        series.elapsed_seconds, values, fixed_tau_seconds=known_tau_seconds
    )

    window_seconds = float(measurement_window_minutes) * 60.0
    cadence = (float(np.median(np.diff(series.elapsed_seconds)))
               if series.n_scans > 1 else window_seconds)
    scans_per_window = max(1.0, window_seconds / max(cadence, 1e-9))
    noise_floor = _per_point_noise(values) / np.sqrt(scans_per_window)

    assessment = dict(
        frequency_thz=float(series.frequency_thz[index]),
        fit=fit,
        noise_floor=float(noise_floor),
        measurement_window_minutes=float(measurement_window_minutes),
        acceptable_drift_fraction=float(acceptable_drift_fraction),
        record_tau_ratio=float('nan'),
    )

    if not fit['transient_detected']:
        assessment.update(
            first_acceptable_index=0,
            first_acceptable_minutes=0.0,
            settled_fraction=1.0,
            residual_drift=0.0,
            purge_limited=False,
            verdict=(f"EQUILIBRATED at {assessment['frequency_thz']:.2f} THz — no purge "
                     f"transient detected; every acquisition is usable ({fit['reason']})."),
        )
        return assessment

    tau = fit['tau_seconds']
    assessment['record_tau_ratio'] = float(series.elapsed_seconds[-1] / tau)

    wait_seconds = time_until_acceptable_drift(
        fit, window_seconds, acceptable_drift_fraction
    )
    acceptable = series.elapsed_seconds >= wait_seconds
    first_index = int(np.argmax(acceptable)) if acceptable.any() else None

    # State at the END of the record: what you would get by measuring from here.
    residual_drift = residual_drift_over_window(
        fit, float(series.elapsed_seconds[-1]), window_seconds
    )
    settled_fraction = 1.0 - abs(
        fit['span'] * np.exp(-series.elapsed_seconds[-1] / tau) / fit['span']
    ) if fit['span'] else 1.0

    assessment.update(
        first_acceptable_index=first_index,
        first_acceptable_minutes=(None if first_index is None
                                  else float(series.elapsed_minutes[first_index])),
        settled_fraction=float(settled_fraction),
        residual_drift=float(residual_drift),
        purge_limited=bool(residual_drift > noise_floor),
    )

    if first_index is None:
        assessment['verdict'] = (
            f"NOT YET ACCEPTABLE at {assessment['frequency_thz']:.2f} THz — needs "
            f"{wait_seconds / 60:.0f} min from the first acquisition, but this run is only "
            f"{series.elapsed_minutes[-1]:.0f} min long. Discard it, or wait "
            f"{(wait_seconds - series.elapsed_seconds[-1]) / 60:.0f} min more before "
            f"re-measuring."
        )
    else:
        kept = series.n_scans - first_index
        assessment['verdict'] = (
            f"CUT AT ACQUISITION {first_index} ({series.elapsed_minutes[first_index]:.0f} min) "
            f"— from there the purge contributes < {100 * acceptable_drift_fraction:.1f}% over a "
            f"{measurement_window_minutes:.0f} min average. Keeps {kept}/{series.n_scans} "
            f"acquisitions."
        )
    return assessment


def print_settling_assessment(assessment: dict) -> dict:
    """Print a settling assessment as a short operator-facing block."""
    fit = assessment['fit']
    print(f"\n[purge_settling_assessment] at {assessment['frequency_thz']:.2f} THz, "
          f"{assessment['measurement_window_minutes']:.0f} min measurement window")
    if fit['transient_detected']:
        print(f"  tau               : {fit['tau_seconds'] / 60:.1f} min "
              f"(record covers {assessment['record_tau_ratio']:.1f} tau"
              f"{' — plateau is EXTRAPOLATED, treat with care' if assessment['record_tau_ratio'] < 1 else ''})")
        print(f"  settled at end    : {100 * assessment['settled_fraction']:.1f}% of the way "
              f"to the plateau")
        print(f"  residual drift    : {100 * assessment['residual_drift']:.3f}% over the next "
              f"{assessment['measurement_window_minutes']:.0f} min "
              f"(threshold {100 * assessment['acceptable_drift_fraction']:.1f}%)")
        limiter = ('PURGE-limited' if assessment['purge_limited']
                   else 'NOISE-limited (waiting longer buys nothing measurable)')
        print(f"  statistical noise : {100 * assessment['noise_floor']:.3f}% on the same average "
              f"-> {limiter}")
    print(f"  VERDICT           : {assessment['verdict']}")
    return assessment


def purge_equilibration_report(
    series: AcquisitionSeries,
    frequencies_thz: Sequence[float] = (0.5, 1.0, 1.6, 2.0),
    *,
    settle_fraction: float = 0.99,
    baseline_scans: int = 8,
) -> dict:
    """Is this run sitting inside a purge transient, and if so how long until it settles?

    Fits the saturating exponential to BOTH observables a gas exchange moves:

    * **timing** — the pulse delay.  Tracks bulk gas composition: N2 is more
      refractive than air (n-1 = 2.98e-4 vs 2.77e-4), so swapping air for nitrogen
      *lengthens* the optical path and the pulse arrives LATER.
    * **amplitude** — strongest at high frequency.  Tracks water vapour, whose
      rotational absorption climbs steeply across this band.

    Both should saturate with a similar time constant.  Water's is typically the
    LONGER of the two, because water keeps desorbing from the chamber walls after
    the bulk gas has already been exchanged — so treat the amplitude time constant
    as the one that governs when it is safe to measure.

    Returns ``{'timing': fit, 'amplitude': {frequency: fit}, 'verdict': str}``.
    """
    minutes = series.elapsed_minutes
    report: dict = {'amplitude': {}}

    delays = fitted_delay_seconds(series, baseline_scans=baseline_scans)
    report['timing'] = fit_exponential_equilibration(series.elapsed_seconds, delays)

    ratio, _entry = compute_drift_metric(
        series, 'amplitude_ratio', baseline_scans=baseline_scans
    )
    for frequency in frequencies_thz:
        index = series.frequency_index(frequency)
        report['amplitude'][float(series.frequency_thz[index])] = (
            fit_exponential_equilibration(series.elapsed_seconds, ratio[:, index])
        )

    detected = [fit for fit in [report['timing'], *report['amplitude'].values()]
                if fit['transient_detected']]
    print(f"\n[purge_equilibration_report] {series.filename} "
          f"({series.n_scans} acquisitions over {minutes[-1]:.1f} min)")

    timing_fit = report['timing']
    if timing_fit['transient_detected']:
        print(f"  timing    : tau {timing_fit['tau_seconds'] / 60:>6.1f} min, plateau "
              f"{timing_fit['plateau'] * 1e15:+.1f} fs  "
              f"(exponential {timing_fit['improvement_over_linear']:.0f}x better than linear)")
    else:
        print(f"  timing    : no transient — {timing_fit['reason']}")

    for frequency, fit in report['amplitude'].items():
        if fit['transient_detected']:
            print(f"  {frequency:>5.2f} THz: tau {fit['tau_seconds'] / 60:>6.1f} min, plateau "
                  f"{100 * (fit['plateau'] - 1):+6.1f}%  "
                  f"(exponential {fit['improvement_over_linear']:.0f}x better)")
        else:
            print(f"  {frequency:>5.2f} THz: no transient — {fit['reason']}")

    if not detected:
        report['verdict'] = 'EQUILIBRATED — no purge transient detected in this run.'
    else:
        slowest = max(fit['tau_seconds'] for fit in detected)
        settle_seconds = time_to_equilibrate(slowest, settle_fraction)
        report['verdict'] = (
            f"PURGE TRANSIENT — slowest tau {slowest / 60:.0f} min, so "
            f"{100 * settle_fraction:.0f}% settling needs {settle_seconds / 60:.0f} min "
            f"({settle_seconds / 3600:.1f} h) MEASURED FROM THE FIRST ACQUISITION of this run "
            f"(this run began at an unknown point on the purge curve; if the chamber was closed "
            f"earlier, the wall-clock wait from closing is longer). This run is "
            f"{minutes[-1]:.0f} min long."
        )
    print(f"  VERDICT   : {report['verdict']}")
    return report


def drift_rate_per_hour(
    series: AcquisitionSeries,
    frequencies_thz: Sequence[float],
    *,
    metric: str = 'amplitude_ratio',
    **metric_options,
) -> dict:
    """Least-squares slope of *metric* against elapsed time, per frequency.

    Fitted against elapsed **hours**, not acquisition index, so a run with gaps or
    a changing cadence still gives a physically meaningful rate.

    Returns ``{frequency_thz: {'slope_per_hour', 'start', 'end', 'total_change'}}``.
    """
    values, _entry = compute_drift_metric(series, metric, **metric_options)
    hours = series.elapsed_hours
    results = {}
    for frequency in frequencies_thz:
        index = series.frequency_index(frequency)
        column = values[:, index]
        finite = np.isfinite(column)
        if finite.sum() < 2:
            continue
        slope, intercept = np.polyfit(hours[finite], column[finite], 1)
        results[float(series.frequency_thz[index])] = dict(
            slope_per_hour=float(slope),
            start=float(intercept),
            end=float(slope * hours[finite][-1] + intercept),
            total_change=float(slope * (hours[finite][-1] - hours[finite][0])),
        )
    return results


def summarise_drift(
    series: AcquisitionSeries,
    frequencies_thz: Sequence[float] = (0.3, 0.5, 0.8, 1.2, 1.6),
    *,
    metric: str = 'amplitude_ratio',
    **metric_options,
) -> dict:
    """Print a drift report for one series and return the fitted rates."""
    entry = DRIFT_METRICS[metric]
    rates = drift_rate_per_hour(series, frequencies_thz, metric=metric, **metric_options)
    print(f"\n[summarise_drift] {series.filename} — {entry.label}")
    print(series.describe())
    peak_values = series.peak_amplitude_per_scan
    early = peak_values[: max(1, series.n_scans // 10)].mean()
    late = peak_values[-max(1, series.n_scans // 10):].mean()
    print(
        f"  time-domain peak   : first 10% {early:+.5g} -> last 10% {late:+.5g} "
        f"({100 * (late / early - 1):+.1f}%)"
    )
    try:
        timing = delay_drift_rate(series)
        step_fraction = 100 * abs(timing['total_seconds']) / timing['sample_step_seconds']
        print(
            f"  timing drift       : {timing['total_seconds'] * 1e15:+.1f} fs total "
            f"({timing['rate_seconds_per_hour'] * 1e15:+.1f} fs/hour, scatter "
            f"{timing['scatter_seconds'] * 1e15:.2f} fs) = {step_fraction:.0f}% of one "
            f"{timing['sample_step_seconds'] * 1e15:.0f} fs sample step"
        )
    except ValueError as error:
        print(f"  timing drift       : not measurable ({error})")
    print(f"  {'freq (THz)':>11}  {'start':>9}  {'end':>9}  {'per hour':>10}")
    for frequency, result in rates.items():
        print(
            f"  {frequency:>11.3f}  {result['start']:>9.4f}  {result['end']:>9.4f}  "
            f"{result['slope_per_hour']:>10.4f}"
        )
    return rates


def save_metric_table(
    series: AcquisitionSeries,
    filepath: str,
    frequencies_thz: Sequence[float] = (0.3, 0.5, 0.8, 1.2, 1.6),
    *,
    metric: str = 'amplitude_ratio',
    **metric_options,
) -> str:
    """Write ``elapsed_minutes, acquisition_number, <metric at each frequency>`` to CSV."""
    values, entry = compute_drift_metric(series, metric, **metric_options)
    indices = [series.frequency_index(frequency) for frequency in frequencies_thz]
    header = ['elapsed_minutes', 'acquisition_number'] + [
        f"{entry.name}_{series.frequency_thz[index]:.3f}THz" for index in indices
    ]
    table = np.column_stack(
        [series.elapsed_minutes, series.scan_numbers] + [values[:, index] for index in indices]
    )
    np.savetxt(filepath, table, delimiter=',', header=','.join(header), comments='')
    print(f"[save_metric_table] wrote {table.shape[0]} rows x {table.shape[1]} cols -> {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DEFAULT_PLOT_CONFIG: dict = {
    "figsize": (9.0, 5.0),
    "title": None,            # None -> built from the series and metric
    "frequency_range_thz": None,   # (lo, hi); None -> the trusted band
    "value_range": None,      # (lo, hi) for the metric axis / colour scale
    "show_mask": True,        # restrict to the trusted (above-noise-floor) band
    "snr_thresh_db": 10.0,
    "tail_fraction": 0.25,
    "colormap": None,         # None -> 'RdBu_r' if the metric diverges, else 'viridis'
    "robust_percentile": 2.0,  # colour limits from this..100-this percentile
    "x_axis": "elapsed_minutes",   # or 'acquisition_number'
    "legend_fontsize": 8,
    "grid": True,
}


def _resolve_plot_config(plot_config: dict | None) -> dict:
    resolved = dict(DEFAULT_PLOT_CONFIG)
    resolved.update(plot_config or {})
    return resolved


def _x_axis_values(series: AcquisitionSeries, plot_config: dict) -> tuple:
    if plot_config['x_axis'] == 'acquisition_number':
        return series.scan_numbers, 'Acquisition number'
    return series.elapsed_minutes, 'Elapsed time (min)'


def _band_mask(series: AcquisitionSeries, plot_config: dict) -> np.ndarray:
    """Frequency bins to display: the trusted band, further limited by a range."""
    if plot_config['show_mask']:
        mask = series.trusted_mask(plot_config['snr_thresh_db'], plot_config['tail_fraction'])
    else:
        mask = np.ones(series.n_frequencies, dtype=bool)
    frequency_range = plot_config['frequency_range_thz']
    if frequency_range is not None:
        mask = mask & (series.frequency_thz >= frequency_range[0]) \
                    & (series.frequency_thz <= frequency_range[1])
    return mask


def _colour_limits(values: np.ndarray, entry: DriftMetric, plot_config: dict) -> tuple:
    """Robust colour limits, made symmetric about the metric's reference value."""
    if plot_config['value_range'] is not None:
        return plot_config['value_range']
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    percentile = plot_config['robust_percentile']
    low, high = np.percentile(finite, [percentile, 100.0 - percentile])
    if entry.diverging and entry.reference_value is not None:
        centre = entry.reference_value
        half_span = max(abs(high - centre), abs(centre - low)) or 1e-9
        return (centre - half_span, centre + half_span)
    return (low, high)


def plot_drift_map(
    series: AcquisitionSeries,
    metric: str = 'amplitude_ratio',
    *,
    plot_config: dict | None = None,
    ax=None,
    **metric_options,
):
    """Heatmap of a drift metric: elapsed time (x) vs frequency (y).

    The one-glance view — a vertical stripe is a moment when everything moved (a
    bump, a purge change), a horizontal stripe is one frequency band drifting on
    its own, and a smooth left-to-right gradient is steady drift.
    """
    plot_config = _resolve_plot_config(plot_config)
    values, entry = compute_drift_metric(series, metric, **metric_options)
    mask = _band_mask(series, plot_config)
    if not mask.any():
        raise ValueError(
            f"'{series.filename}': no frequency bins survive the trusted band and "
            f"frequency_range_thz={plot_config['frequency_range_thz']}. Lower "
            f"plot_config['snr_thresh_db'] or set show_mask=False."
        )
    x_values, x_label = _x_axis_values(series, plot_config)

    if ax is None:
        _figure, ax = plt.subplots(figsize=plot_config['figsize'], layout='constrained')

    low, high = _colour_limits(values[:, mask], entry, plot_config)
    colormap = plot_config['colormap'] or ('RdBu_r' if entry.diverging else 'viridis')
    mesh = ax.pcolormesh(
        x_values,
        series.frequency_thz[mask],
        values[:, mask].T,
        shading='nearest',
        cmap=colormap,
        vmin=low,
        vmax=high,
    )
    colorbar = ax.figure.colorbar(mesh, ax=ax)
    colorbar.set_label(entry.label)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Frequency (THz)')
    ax.set_title(plot_config['title'] or f"{series.filename} — {entry.label}")
    return ax


def plot_frequency_traces(
    series: AcquisitionSeries,
    metric: str = 'amplitude_ratio',
    frequencies_thz: Sequence[float] = (0.3, 0.5, 0.8, 1.2, 1.6),
    *,
    plot_config: dict | None = None,
    ax=None,
    **metric_options,
):
    """One line per selected frequency: the metric against elapsed time.

    The quantitative companion to the heatmap — this is the plot to read a drift
    percentage off.
    """
    plot_config = _resolve_plot_config(plot_config)
    values, entry = compute_drift_metric(series, metric, **metric_options)
    x_values, x_label = _x_axis_values(series, plot_config)

    if ax is None:
        _figure, ax = plt.subplots(figsize=plot_config['figsize'], layout='constrained')

    trusted = series.trusted_mask(plot_config['snr_thresh_db'], plot_config['tail_fraction'])
    for frequency in frequencies_thz:
        index = series.frequency_index(frequency)
        untrusted_note = '' if trusted[index] else ' [below noise floor]'
        ax.plot(
            x_values,
            values[:, index],
            lw=1.2,
            marker='.',
            markersize=3,
            alpha=0.5 if untrusted_note else 0.95,
            label=f"{series.frequency_thz[index]:.2f} THz{untrusted_note}",
        )
    if entry.reference_value is not None:
        ax.axhline(entry.reference_value, color='k', lw=0.8, ls='--', alpha=0.6)
    if plot_config['value_range'] is not None:
        ax.set_ylim(plot_config['value_range'])
    ax.set_xlabel(x_label)
    ax.set_ylabel(entry.label)
    ax.set_title(plot_config['title'] or f"{series.filename} — {entry.label}")
    ax.legend(fontsize=plot_config['legend_fontsize'])
    ax.grid(plot_config['grid'], alpha=0.3)
    return ax


def plot_drift_comparison(
    series_by_filename: dict,
    frequency_thz: float = 0.5,
    metric: str = 'amplitude_ratio',
    *,
    plot_config: dict | None = None,
    ax=None,
    **metric_options,
):
    """One line per FILE at a single frequency — the common-mode separator.

    Drift shared by the sample and a gold reference is the instrument; drift the
    sample shows and gold does not is the sample.  Note the x-axis is each file's
    own elapsed time, so runs taken at different times of day are overlaid by
    duration, not by clock time.
    """
    plot_config = _resolve_plot_config(plot_config)
    if ax is None:
        _figure, ax = plt.subplots(figsize=plot_config['figsize'], layout='constrained')

    entry = DRIFT_METRICS[metric]
    for filename, series in series_by_filename.items():
        values, _entry = compute_drift_metric(series, metric, **metric_options)
        index = series.frequency_index(frequency_thz)
        x_values, x_label = _x_axis_values(series, plot_config)
        ax.plot(x_values, values[:, index], lw=1.2, marker='.', markersize=3,
                label=f"{filename} @ {series.frequency_thz[index]:.2f} THz")
    if entry.reference_value is not None:
        ax.axhline(entry.reference_value, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(entry.label)
    ax.set_title(plot_config['title'] or f"Drift comparison at {frequency_thz:.2f} THz")
    ax.legend(fontsize=plot_config['legend_fontsize'])
    ax.grid(plot_config['grid'], alpha=0.3)
    return ax


def plot_time_domain_drift(
    series: AcquisitionSeries,
    *,
    plot_config: dict | None = None,
    axes=None,
):
    """Time-domain drift: peak amplitude and peak arrival time per acquisition.

    The rawest possible check, free of any windowing or FFT choice.  If this is
    flat and the spectral metrics are not, suspect the processing; if this drifts,
    the measurement drifted.
    """
    plot_config = _resolve_plot_config(plot_config)
    x_values, x_label = _x_axis_values(series, plot_config)

    if axes is None:
        _figure, axes = plt.subplots(
            2, 1, figsize=plot_config['figsize'], sharex=True, layout='constrained'
        )
    amplitude_axis, timing_axis = axes

    peak_values = series.peak_amplitude_per_scan
    amplitude_axis.plot(x_values, peak_values, lw=1.0, marker='.', markersize=3)
    amplitude_axis.set_ylabel('Peak amplitude (a.u.)')
    amplitude_axis.set_title(
        plot_config['title'] or f"{series.filename} — time-domain drift"
    )
    amplitude_axis.grid(plot_config['grid'], alpha=0.3)

    timing_axis.plot(
        x_values, series.peak_time_seconds_per_scan * _S_TO_PS,
        lw=1.0, marker='.', markersize=3, color='tab:orange',
    )
    timing_axis.set_ylabel('Peak time (ps)')
    timing_axis.set_xlabel(x_label)
    timing_axis.grid(plot_config['grid'], alpha=0.3)
    timing_axis.annotate(
        f"quantised to dt = {series.processing['dt_ps']:.3f} ps",
        xy=(0.99, 0.05), xycoords='axes fraction', ha='right', fontsize=7, alpha=0.7,
    )
    return axes


def _draw_equilibration_column(
    axes_column,
    series: AcquisitionSeries,
    frequencies_thz: Sequence[float],
    *,
    baseline_scans: int,
    include_timing: bool,
    plot_config: dict,
    column_title: str,
    fixed_tau_seconds: float | None = None,
) -> dict:
    """Draw one series' fit / residual / timing panels. Returns the fits by label."""
    minutes = series.elapsed_minutes
    dense_minutes = np.linspace(0.0, minutes[-1], 400)
    dense_seconds = dense_minutes * 60.0
    fits: dict = {}
    residual_scale = []   # collects |exponential residual| to scale that panel

    fit_axis = axes_column[0]
    residual_axis = axes_column[1]
    ratio, _entry = compute_drift_metric(
        series, 'amplitude_ratio', baseline_scans=baseline_scans
    )

    for order, frequency in enumerate(frequencies_thz):
        index = series.frequency_index(frequency)
        values = ratio[:, index]
        label = f"{series.frequency_thz[index]:.2f} THz"
        colour = f"C{order}"
        fit = fit_exponential_equilibration(
            series.elapsed_seconds, values, fixed_tau_seconds=fixed_tau_seconds
        )
        fits[label] = fit

        fit_axis.plot(minutes, 100 * (values - 1), '.', ms=3.5, color=colour, alpha=0.55)

        if fit['transient_detected']:
            curve = _saturating_exponential(
                dense_seconds, fit['plateau'], fit['span'], fit['tau_seconds']
            )
            tau_note = 'tau*' if fit['tau_fixed'] else 'tau'
            fit_axis.plot(dense_minutes, 100 * (curve - 1), '-', lw=1.8, color=colour,
                          label=f"{label}: {tau_note} {fit['tau_seconds'] / 60:.0f} min, "
                                f"plateau {100 * (fit['plateau'] - 1):+.1f}%")
            fit_axis.axhline(100 * (fit['plateau'] - 1), color=colour, lw=0.8, ls=':', alpha=0.6)
            exponential_residual = 100 * (values - _saturating_exponential(
                series.elapsed_seconds, fit['plateau'], fit['span'], fit['tau_seconds']))
            residual_axis.plot(minutes, exponential_residual, '.', ms=3.5,
                               color=colour, alpha=0.8)
            residual_scale.append(float(np.nanmax(np.abs(exponential_residual))))
        else:
            fit_axis.plot([], [], '.', color=colour, label=f"{label}: no transient")

        # The degradation null hypothesis, made visible: a straight line has no
        # plateau, so if the data really saturates the linear fit must miss at both
        # ends. Its residual is the curved one.
        linear_coefficients = np.polyfit(series.elapsed_seconds, values, 1)
        linear_values = np.polyval(linear_coefficients, series.elapsed_seconds)
        fit_axis.plot(dense_minutes,
                      100 * (np.polyval(linear_coefficients, dense_seconds) - 1),
                      '--', lw=1.0, color=colour, alpha=0.45)
        residual_axis.plot(minutes, 100 * (values - linear_values), 'x', ms=3,
                           color=colour, alpha=0.28)

        noise = _per_point_noise(values)
        residual_axis.axhspan(-100 * noise, 100 * noise, color=colour, alpha=0.10)
        residual_scale.append(3 * 100 * noise)

    fit_axis.set_ylabel('Amplitude change (%)')
    fit_axis.set_title(f"{column_title}\n{minutes[-1]:.0f} min, {series.n_scans} acquisitions",
                       fontsize=9.5)
    fit_axis.legend(fontsize=plot_config['legend_fontsize'], loc='lower right')
    fit_axis.grid(plot_config['grid'], alpha=0.3)

    residual_axis.axhline(0.0, color='k', lw=0.8)
    residual_axis.set_ylabel('Residual (%)')
    residual_axis.grid(plot_config['grid'], alpha=0.3)
    # Scale to the EXPONENTIAL residuals and the noise band, so those stay readable.
    # The linear residuals then run off the panel — which is the point being made,
    # and is labelled as such in the caption.
    limit = 1.35 * max(residual_scale) if residual_scale else 0.0
    if limit > 0:
        residual_axis.set_ylim(-limit, limit)

    if include_timing:
        timing_axis = axes_column[2]
        try:
            delays_fs = fitted_delay_seconds(series, baseline_scans=baseline_scans) * 1e15
            timing_fit = fit_exponential_equilibration(series.elapsed_seconds, delays_fs)
            fits['timing'] = timing_fit
            timing_axis.plot(minutes, delays_fs, '.', ms=3.5, color='tab:purple', alpha=0.6)
            if timing_fit['transient_detected']:
                timing_axis.plot(
                    dense_minutes,
                    _saturating_exponential(dense_seconds, timing_fit['plateau'],
                                            timing_fit['span'], timing_fit['tau_seconds']),
                    '-', lw=1.8, color='tab:purple',
                    label=f"tau {timing_fit['tau_seconds'] / 60:.0f} min, "
                          f"plateau {timing_fit['plateau']:+.0f} fs")
                timing_axis.axhline(timing_fit['plateau'], color='tab:purple',
                                    lw=0.8, ls=':', alpha=0.6)
            else:
                timing_axis.plot([], [], '.', color='tab:purple', label='no transient')
            timing_axis.legend(fontsize=plot_config['legend_fontsize'], loc='lower right')
        except ValueError as error:
            timing_axis.text(0.5, 0.5, f"timing unavailable:\n{error}", ha='center',
                             va='center', transform=timing_axis.transAxes, fontsize=7)
        timing_axis.set_ylabel('Delay (fs)')
        timing_axis.grid(plot_config['grid'], alpha=0.3)

    axes_column[-1].set_xlabel('Elapsed time (min)')
    return fits


def plot_purge_equilibration(
    series: AcquisitionSeries,
    *,
    frequencies_thz: Sequence[float] = (1.0, 2.0),
    control_series: AcquisitionSeries | None = None,
    control_fixed_tau_seconds: float | None = None,
    include_timing: bool = True,
    baseline_scans: int = 8,
    assessment: dict | None = None,
    plot_config: dict | None = None,
):
    """The purge-vs-degradation figure: exponential fit, residuals, and a control.

    Built to be shown to someone who was not in the room. Three arguments are made
    visually, in order of strength:

    1. **The data saturates.** The solid exponential tracks it and flattens onto a
       plateau (dotted). The dashed line is the straight-line fit — the degradation
       null hypothesis. Degradation has no reason to plateau, so if the data really
       does, the straight line must miss at both ends.
    2. **The residuals say which model is right.** Dots are the exponential's
       residuals, faint crosses the straight line's. The shaded band is the
       point-to-point noise. Structureless residuals inside the band mean the model
       is sufficient; the visible curvature in the linear residuals is the straight
       line failing.
    3. **Two independent observables agree.** The bottom panel is the pulse *delay*
       — a different physical quantity, driven by gas refractive index rather than
       water absorption — saturating with a similar time constant. One process.

    Passing ``control_series`` (a run on the same chamber left undisturbed) adds a
    second column and makes the strongest argument of all: the same analysis
    applied to a run that should be flat, and is.

    Parameters
    ----------
    control_fixed_tau_seconds : float, optional
        Hold the control column's tau at this value (typically the tau fitted from
        the main column) instead of fitting it. A control run is usually far shorter
        than one time constant and cannot establish its own — without this it will
        read "no transient" even while sitting on a slow tail. Fits using it are
        marked ``tau*`` in the legend.
    assessment : dict, optional
        Result of :func:`purge_settling_assessment`. When given, its cut time is
        drawn as a vertical marker and its verdict printed under the figure.

    Returns
    -------
    (figure, axes_grid, fits) — ``fits`` keyed by column then panel label.
    """
    plot_config = _resolve_plot_config(plot_config)
    n_rows = 3 if include_timing else 2
    n_columns = 2 if control_series is not None else 1
    height_ratios = [2.2, 1.0, 1.3][:n_rows]

    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=plot_config['figsize'] if plot_config.get('figsize_explicit')
        else (7.0 * n_columns, 2.6 * n_rows),
        sharex='col', squeeze=False, layout='constrained',
        gridspec_kw={'height_ratios': height_ratios},
    )

    all_fits = {}
    all_fits[series.filename] = _draw_equilibration_column(
        axes[:, 0], series, frequencies_thz, baseline_scans=baseline_scans,
        include_timing=include_timing, plot_config=plot_config,
        column_title=f"{series.filename}\n(chamber disturbed before this run)",
    )
    if control_series is not None:
        all_fits[control_series.filename] = _draw_equilibration_column(
            axes[:, 1], control_series, frequencies_thz, baseline_scans=baseline_scans,
            include_timing=include_timing, plot_config=plot_config,
            column_title=f"{control_series.filename}\nMinimal chamber disturbance",
            fixed_tau_seconds=control_fixed_tau_seconds,
        )
        # Shared y-limits per row so "flat" is visually flat, not rescaled to look busy.
        for row in range(n_rows):
            low = min(axis.get_ylim()[0] for axis in axes[row, :])
            high = max(axis.get_ylim()[1] for axis in axes[row, :])
            for axis in axes[row, :]:
                axis.set_ylim(low, high)

    if assessment is not None and assessment.get('first_acceptable_minutes'):
        for axis in axes[:, 0]:
            axis.axvline(assessment['first_acceptable_minutes'], color='k',
                         lw=1.2, ls='-.', alpha=0.7)
        axes[0, 0].annotate(
            f"usable from {assessment['first_acceptable_minutes']:.0f} min",
            xy=(assessment['first_acceptable_minutes'], axes[0, 0].get_ylim()[1]),
            xytext=(4, -10), textcoords='offset points', fontsize=7, alpha=0.8,
        )

    figure.suptitle(
        'Purge equilibration: does the drift saturate (purge) or continue (degradation)?',
        fontsize=11,
    )
    caption = ('solid = exponential fit, dotted = extrapolated plateau, dashed = straight-line '
               '(degradation) fit;  tau* = tau held fixed from the other run\n'
               'residuals: dots = exponential, faint crosses = linear (these run OFF the panel '
               '— that is the straight line failing), shaded = point-to-point noise')
    if assessment is not None:
        caption += f"\n{assessment['verdict']}"
    figure.text(0.5, -0.01, caption, ha='center', va='top', fontsize=7.5, alpha=0.75)
    return figure, axes, all_fits


def plot_timing_drift(
    series: AcquisitionSeries,
    *,
    plot_config: dict | None = None,
    ax=None,
    **delay_options,
):
    """Sub-sample delay per acquisition, with the quantised peak time for contrast.

    The two lines make the point directly: the phase-fitted delay (left axis, fs)
    resolves drift far below the sample step, while the time-domain peak (right
    axis, faint steps) can sit perfectly flat through the whole thing.
    """
    plot_config = _resolve_plot_config(plot_config)
    x_values, x_label = _x_axis_values(series, plot_config)
    timing = delay_drift_rate(series, **delay_options)
    delays_fs = timing['delays_seconds'] * 1e15

    if ax is None:
        _figure, ax = plt.subplots(figsize=plot_config['figsize'], layout='constrained')

    ax.plot(x_values, delays_fs, lw=1.2, marker='.', markersize=3,
            color='tab:blue', label='phase-fitted delay')
    ax.plot(x_values, timing['rate_seconds_per_hour'] * 1e15 * series.elapsed_hours
            + timing['intercept_seconds'] * 1e15,
            'k--', lw=1.0, alpha=0.7,
            label=f"{timing['rate_seconds_per_hour'] * 1e15:+.1f} fs/hour")
    ax.set_xlabel(x_label)
    ax.set_ylabel('Delay relative to run start (fs)')
    ax.set_title(plot_config['title'] or f"{series.filename} — timing drift")
    ax.grid(plot_config['grid'], alpha=0.3)

    peak_axis = ax.twinx()
    peak_reference = series.peak_time_seconds_per_scan[0]
    peak_axis.plot(
        x_values, (series.peak_time_seconds_per_scan - peak_reference) * 1e15,
        color='tab:grey', lw=1.0, alpha=0.45, drawstyle='steps-mid',
        label=f"time-domain peak (dt = {timing['sample_step_seconds'] * 1e15:.0f} fs)",
    )
    peak_axis.set_ylabel('Peak-bin shift (fs)', color='tab:grey')
    peak_axis.tick_params(axis='y', colors='tab:grey')

    handles, labels = ax.get_legend_handles_labels()
    peak_handles, peak_labels = peak_axis.get_legend_handles_labels()
    ax.legend(handles + peak_handles, labels + peak_labels,
              fontsize=plot_config['legend_fontsize'], loc='best')
    return ax


def plot_scan_spectra(
    series: AcquisitionSeries,
    *,
    plot_config: dict | None = None,
    ax=None,
):
    """Every acquisition's amplitude spectrum, coloured from run start to run end.

    A sanity view before trusting any metric: it shows the dynamic range, where the
    noise floor sits, and whether the spread between acquisitions is a coherent
    drift (an ordered colour gradient) or just noise (colours interleaved).
    """
    plot_config = _resolve_plot_config(plot_config)
    if ax is None:
        _figure, ax = plt.subplots(figsize=plot_config['figsize'], layout='constrained')

    colormap = plt.get_cmap(plot_config['colormap'] or 'viridis')
    for index in range(series.n_scans):
        ax.semilogy(
            series.frequency_thz,
            np.abs(series.scan_spectra[index]),
            lw=0.6,
            alpha=0.5,
            color=colormap(index / max(series.n_scans - 1, 1)),
        )
    ax.semilogy(series.frequency_thz, np.abs(series.mean_spectrum), 'k-', lw=1.6,
                label='coherent mean')

    trusted = series.trusted_mask(plot_config['snr_thresh_db'], plot_config['tail_fraction'])
    if trusted.any():
        ax.axvspan(series.frequency_thz[trusted][0], series.frequency_thz[trusted][-1],
                   color='tab:green', alpha=0.06,
                   label=f"trusted band (>{plot_config['snr_thresh_db']:.0f} dB)")

    frequency_range = plot_config['frequency_range_thz']
    ax.set_xlim(frequency_range if frequency_range else (0, series.frequency_thz[-1]))
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('|Y| (a.u.)')
    ax.set_title(plot_config['title']
                 or f"{series.filename} — {series.n_scans} acquisitions (dark=early, light=late)")
    ax.legend(fontsize=plot_config['legend_fontsize'])
    ax.grid(plot_config['grid'], alpha=0.3, which='both')
    return ax


def show_drift_figures(
    series: AcquisitionSeries,
    frequencies_thz: Sequence[float] = (0.3, 0.5, 0.8, 1.2, 1.6),
    *,
    metrics: Sequence[str] = ('amplitude_ratio', 'cumulative_deviation'),
    plot_config: dict | None = None,
    **metric_options,
) -> list:
    """Build the standard figure set for one series (does not call ``plt.show``).

    Returns the axes so a caller can restyle before showing.
    """
    axes = [
        plot_scan_spectra(series, plot_config=plot_config),
        plot_time_domain_drift(series, plot_config=plot_config),
    ]
    try:
        axes.append(plot_timing_drift(series, plot_config=plot_config))
    except ValueError as error:
        print(f"[show_drift_figures] skipping the timing-drift figure: {error}")
    for metric in metrics:
        # compute_drift_metric filters metric_options against each metric's own
        # signature, so the same options can be handed to every metric.
        axes.append(plot_drift_map(series, metric, plot_config=plot_config, **metric_options))
        axes.append(plot_frequency_traces(series, metric, frequencies_thz,
                                          plot_config=plot_config, **metric_options))
    return axes
