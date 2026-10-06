"""Bench tools: read a directory of .acc files and answer one setup question each.

Registered with ``@bench_tool``; the registry is the single source of truth for the command-line
subcommands, their help text and their dispatch in ``thz_ellipsometry_bench_run_me.py``.
Every tool takes ``(directory, config)`` and returns ``(result, text_report)``. A tool can also
have figures, registered with ``@bench_figures("<tool>")``: a function of ``(directory, config)``
returning ``{filename: matplotlib Figure}``; the drawing itself is in ``bench_plots``.

The analyses themselves are pure and live in ``thz_ellipsometry.core.bench``; this module only
finds the files, parses their filename tokens and builds spectra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from ..core import bench
from ..core.preprocess import transform_traces
from .loader import parse_key_value_tokens, polarisation_angle_deg_from_filename, \
    read_accumulation_file

__all__ = ["BENCH_TOOLS", "NullSweep", "NullSweepCollection", "bench_figures", "bench_tool",
           "bench_tool_names", "collect_null_sweeps", "make_bench_figures", "run_bench_tool"]


@dataclass(frozen=True)
class BenchTool:
    name: str
    function: object
    summary: str
    figures: object = None          #: (directory, config) -> {filename: Figure}, or None


#: name -> BenchTool. The one place a bench tool exists.
BENCH_TOOLS: dict[str, BenchTool] = {}


def bench_tool(function):
    """Register a bench tool; its name is the subcommand, its docstring's first line the help."""
    name = function.__name__.replace("_", "-")
    summary = (function.__doc__ or "").strip().splitlines()
    BENCH_TOOLS[name] = BenchTool(name=name, function=function,
                                  summary=summary[0] if summary else "")
    return function


def bench_figures(tool_name):
    """Register the figures of an already-registered tool (decorator)."""
    def register(function):
        BENCH_TOOLS[tool_name] = replace(BENCH_TOOLS[tool_name], figures=function)
        return function
    return register


def bench_tool_names():
    return sorted(BENCH_TOOLS)


def run_bench_tool(name, directory, config):
    try:
        entry = BENCH_TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown bench tool {name!r}; registered: {bench_tool_names()}") \
            from None
    return entry.function(directory, config)


def make_bench_figures(name, directory, config):
    """The tool's figures as ``{filename: Figure}``; empty if the tool has none."""
    entry = BENCH_TOOLS[name]
    return {} if entry.figures is None else entry.figures(directory, config)


# ---------------------------------------------------------------------------
# Shared file handling
# ---------------------------------------------------------------------------

def _acc_files(directory, *, contains=None):
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"no such directory: {directory}")
    names = sorted(name for name in os.listdir(directory) if name.lower().endswith(".acc"))
    if contains:
        names = [name for name in names if contains.lower() in name.lower()]
    if not names:
        raise FileNotFoundError(f"no matching .acc files in {directory}"
                                + (f" containing {contains!r}" if contains else ""))
    return [os.path.join(directory, name) for name in names]


def _band(frequencies_hz, config):
    band = config.get("band", {})
    low = (band.get("frequency_min_thz") or 0.0) * 1e12
    high = (band.get("frequency_max_thz") or np.inf) * 1e12
    return (frequencies_hz >= low) & (frequencies_hz <= high)


def _transform(time_ps, traces, config, window_centre_ps=None):
    preprocess = dict(config.get("preprocess", {}))
    if window_centre_ps is not None:
        preprocess["window_centre_ps"] = window_centre_ps
    return transform_traces(
        time_ps, traces,
        window_half_width_ps=preprocess.get("window_half_width_ps"),
        baseline_fraction=preprocess.get("baseline_fraction", 0.1),
        pad_factor=preprocess.get("pad_factor", 4),
        window_centre_ps=preprocess.get("window_centre_ps"),
        window_shape=preprocess.get("window_shape", "tukey"),
        taper_fraction=preprocess.get("taper_fraction", 0.5))


def _common_window_centre_ps(files):
    """One window centre for a whole group: the peak of the mean absolute trace."""
    mean_trace = np.mean([np.abs(file.averaged - file.averaged[:max(file.averaged.size // 10, 1)]
                                 .mean()) for file in files], axis=0)
    return float(files[0].time_ps[int(np.argmax(mean_trace))])


def _token(path, key, delimiter="_"):
    return polarisation_angle_deg_from_filename(path, key, delimiter)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NullSweep:
    """One wire-grid null sweep: what was measured, and the fits made to it.

    Everything the report and the figures need, so both read the same numbers.
    """
    polarisation: float | None      #: the ``null=`` token (None if absent)
    name: str                       #: filename stem shared by the sweep's files
    filenames: tuple
    fit: bench.WireGridNull         #: the fit the report uses (shared background if paired)
    lone_fit: bench.WireGridNull    #: the same sweep fitted alone, without a background
    paired: bool                    #: True if a sweep 180 deg away shared the background
    elapsed_minutes: np.ndarray     #: per reading, file mid-time since the sweep's first file
    time_ps: np.ndarray
    traces: np.ndarray              #: (n_readings, n_samples) averaged, baseline removed
    window: np.ndarray              #: the window the transform applied

    @property
    def readings_deg(self):
        return self.fit.magnet_angles_deg

    @property
    def signed_amplitudes(self):
        return self.fit.signed_amplitudes


@dataclass(frozen=True)
class NullSweepCollection:
    sweeps: list                    #: NullSweep, ordered by polarisation label
    file_count: int
    sweep_count: int                #: including the groups too small to fit
    too_small: dict                 #: label -> number of files, for groups below four


def _sweep_name(paths, angle_token):
    stems = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    tokens = [stem.split("_") for stem in stems]
    shared = [token for token in tokens[0]
              if all(token in others for others in tokens[1:])
              and not token.lower().startswith(f"{angle_token.lower()}=")]
    return "_".join(shared) or stems[0]


def _elapsed_minutes(files):
    stamps = [file.mean_timestamp for file in files]
    if any(stamp is None for stamp in stamps):
        return np.full(len(files), np.nan)
    first = min(stamps)
    return np.array([(stamp - first).total_seconds() / 60.0 for stamp in stamps])


def _null_sweeps(groups, config, angle_token):
    """Fit sweeps pairwise where a partner 180 deg away exists (shared background), else alone.

    The two sweeps of a pair are projected onto ONE reference spectrum, so their signed
    amplitudes -- and the background they share -- are on the same scale.
    """
    pending = dict(groups)
    fitted = {}
    while pending:
        polarisation, entries = pending.popitem()
        partner = None
        if polarisation is not None:
            partner = next((key for key in pending if key is not None
                            and np.isclose((key - polarisation) % 360.0, 180.0)), None)
        members = [(polarisation, sorted(entries, key=lambda entry: entry[0]))]
        if partner is not None:
            members.append((partner, sorted(pending.pop(partner), key=lambda entry: entry[0])))
        files = [file for _, sweep in members for _, file in sweep]
        traces = np.vstack([file.averaged for file in files])
        transformed = _transform(files[0].time_ps, traces, config,
                                 _common_window_centre_ps(files))
        band = _band(transformed.frequencies_hz, config)
        strongest = int(np.argmax(np.sum(np.abs(transformed.spectra[:, band]) ** 2, axis=1)))
        amplitudes = bench.signed_projection(transformed.spectra,
                                             transformed.spectra[strongest], band)
        baseline_count = max(int(config.get("preprocess", {}).get("baseline_fraction", 0.1)
                                 * traces.shape[1]), 1)
        corrected = traces - traces[:, :baseline_count].mean(axis=1, keepdims=True)
        sweeps, rows = [], []
        start = 0
        for _, sweep in members:
            rows.append(slice(start, start + len(sweep)))
            sweeps.append(([reading for reading, _ in sweep], amplitudes[rows[-1]]))
            start += len(sweep)
        for (key, sweep), rows_of_sweep, (readings, values), result in zip(
                members, rows, sweeps, bench.fit_wire_grid_nulls(sweeps)):
            sweep_files = [file for _, file in sweep]
            fitted[key] = NullSweep(
                polarisation=key,
                name=_sweep_name([file.path for file in sweep_files], angle_token),
                filenames=tuple(os.path.basename(file.path) for file in sweep_files),
                fit=result, lone_fit=bench.fit_wire_grid_null(readings, values),
                paired=partner is not None,
                elapsed_minutes=_elapsed_minutes(sweep_files),
                time_ps=files[0].time_ps, traces=corrected[rows_of_sweep],
                window=transformed.window)
    return fitted


def collect_null_sweeps(directory, config):
    """Read and fit every wire-grid null sweep in a directory. Returns NullSweepCollection.

    Shared by the ``null`` tool's report and its figures, so both show the same fits.
    """
    settings = config.get("null", {})
    angle_token = settings.get("angle_token", "mag")
    group_token = settings.get("group_token", "null")
    paths = _acc_files(directory, contains=settings.get("filename_contains"))
    groups = {}
    for path in paths:
        reading = _token(path, angle_token)
        if reading is None:
            continue
        groups.setdefault(_token(path, group_token), []).append(
            (reading, read_accumulation_file(path)))
    if not groups:
        raise ValueError(f"no files with a '{angle_token}=' token in {directory}")

    too_small = {key: len(entries) for key, entries in groups.items() if len(entries) < 4}
    usable = {key: entries for key, entries in groups.items() if len(entries) >= 4}
    fitted = _null_sweeps(usable, config, angle_token)
    ordered = [fitted[key] for key in sorted(fitted, key=lambda key: (key is None, key))]
    return NullSweepCollection(sweeps=ordered, file_count=sum(map(len, groups.values())),
                               sweep_count=len(groups), too_small=too_small)


@bench_tool
def null(directory, config):
    """Wire-grid nulls -> the magnet calibration table, from files named ..._null=<pol>_mag=<reading>."""
    group_token = config.get("null", {}).get("group_token", "null")
    collection = collect_null_sweeps(directory, config)
    lines = [f"[null] {collection.file_count} files in {collection.sweep_count} null sweep(s)"]
    lines += [f"   {group_token}={key}: only {count} files; need 4+"
              for key, count in collection.too_small.items()]
    results, table = {}, {}
    for sweep in collection.sweeps:
        polarisation, result = sweep.polarisation, sweep.fit
        count, paired = len(sweep.filenames), sweep.paired
        results[polarisation] = result
        label = ("(no null= token; assumed p)" if polarisation is None
                 else f"polarisation {polarisation:g} deg")
        background = (f"background {result.leakage_fraction:.1%} (shared with the sweep "
                      "180 deg away)" if paired else
                      "no background term (no sweep 180 deg away to separate it)")
        lines.append(f"   {label}: NULL at reading {result.null_deg:.2f} +/- "
                     f"{result.null_standard_error_deg:.2f} deg; {background}; {count} readings")
        readings = result.magnet_angles_deg
        if not (readings.min() < result.null_deg < readings.max()):
            lines.append("      WARNING: the null lies outside the measured readings; extend "
                         "the sweep")
        table[0.0 if polarisation is None else float(polarisation)] = round(result.null_deg, 2)

    if len(table) > 1:
        polarisations = sorted(table)
        gaps = np.diff([table[key] for key in polarisations])
        nominal = np.diff(polarisations)
        lines.append("   spacing between nulls minus nominal: "
                     + ", ".join(f"{gap - step:+.2f}" for gap, step in zip(gaps, nominal))
                     + " deg  (scale or sense error if large)")
    lines += ["   Grid passing s nulls at polarisation 0/180, passing p at 90/270.",
              f"   -> config['geometry']['magnet_calibration'] = {table}"]
    return results, "\n".join(lines)


@bench_figures("null")
def null_figures(directory, config):
    """The fit to every sweep with its residuals, and the time traces behind the points."""
    from . import bench_plots
    sweeps = collect_null_sweeps(directory, config).sweeps
    return {"null_fits.png": bench_plots.plot_null_fits(sweeps),
            "null_traces.png": bench_plots.plot_null_traces(sweeps)}


@bench_tool
def live(directory, config):
    """Purge settling: per-scan delay and amplitude drift for each magnet state, and its rate now."""
    settings = config.get("live", {})
    angle_token = settings.get("angle_token", "mag")
    limit = settings.get("rate_limit_fs_per_minute", 1.0)
    window = settings.get("window_scans", 3)
    paths = _acc_files(directory, contains=settings.get("filename_contains"))
    files = [read_accumulation_file(path) for path in paths]

    groups = {}
    for file in files:
        groups.setdefault(_token(file.path, angle_token), []).append(file)

    results, lines = {}, [f"[live] {len(files)} files in {directory}"]
    for state, members in sorted(groups.items(), key=lambda item: (item[0] is None, item[0])):
        stamps_known = all(file.mean_timestamp is not None for file in members)
        if stamps_known:
            members = sorted(members, key=lambda file: file.mean_timestamp)
        scans = np.vstack([file.scans for file in members])
        stamps = [stamp for file in members for stamp in file.scan_timestamps]
        if any(stamp is None for stamp in stamps):
            elapsed = np.arange(scans.shape[0], dtype=float) * 60.0
        else:
            elapsed = np.array([(stamp - stamps[0]).total_seconds() for stamp in stamps])
        transformed = _transform(members[0].time_ps, scans, config,
                                 _common_window_centre_ps(members))
        band = _band(transformed.frequencies_hz, config)
        trend = bench.settling_trend(transformed.spectra, elapsed, transformed.frequencies_hz,
                                     band, window_scans=window)
        results[state] = trend
        label = "no angle token" if state is None else f"{angle_token}={state:g}"
        verdict = ("SETTLED" if trend.is_settled(limit) else "still drifting") if np.isfinite(
            trend.recent_rate_fs_per_minute) else "need more scans"
        lines.append(f"   {label}: {scans.shape[0]} scans over "
                     f"{trend.elapsed_minutes[-1]:.1f} min; total delay change "
                     f"{trend.delays_fs[-1]:+.1f} fs, amplitude {trend.amplitudes[-1] - 1:+.2%}; "
                     f"last {trend.window_scans} scans {trend.recent_rate_fs_per_minute:+.2f} "
                     f"fs/min -> {verdict} (limit {limit} fs/min)")
    return results, "\n".join(lines)


@bench_tool
def hwp(directory, config):
    """Half-wave-plate test on gold: beam walk (360-deg periodic) vs polarisation (90-deg), files ..._hwp=<deg>."""
    settings = config.get("hwp", {})
    plate_token = settings.get("plate_token", "hwp")
    paths = _acc_files(directory, contains=settings.get("filename_contains"))
    entries = [(_token(path, plate_token), read_accumulation_file(path)) for path in paths]
    entries = sorted([(angle, file) for angle, file in entries if angle is not None],
                     key=lambda entry: entry[0])
    if len(entries) < 3:
        raise ValueError(f"found {len(entries)} files with a '{plate_token}=' token; need at "
                         "least three plate angles, including two 90 deg apart")
    files = [file for _, file in entries]
    transformed = _transform(files[0].time_ps, np.vstack([file.averaged for file in files]),
                             config, _common_window_centre_ps(files))
    band = _band(transformed.frequencies_hz, config)
    angles = np.array([angle for angle, _ in entries])
    result = bench.analyse_half_wave_plate(angles, transformed.spectra,
                                           transformed.frequencies_hz, band,
                                           reference_plate_deg=settings.get("reference_plate_deg"))
    lines = [f"[hwp] {len(files)} plate angles; ratios to plate = "
             f"{result.reference_plate_deg:g} deg",
             "   plate [deg]   roll-off d ln|R|/df [1/THz]   delay [fs]   |R|      arg R [deg]"]
    for angle, slope, delay, constant in zip(angles, result.log_amplitude_slope_per_thz,
                                             result.delay_fs, result.constant):
        lines.append(f"   {angle:9.1f}   {slope:+24.4f}   {delay:+10.2f}   {abs(constant):.4f}"
                     f"   {np.rad2deg(np.angle(constant)):+8.1f}")
    lines += [f"   same polarisation state (90 deg apart): roll-off spread "
              f"{result.same_state_slope_spread:.4f} /THz, delay spread "
              f"{result.same_state_delay_spread_fs:.2f} fs  <- this is beam walk",
              f"   360-deg (wedge) part of the roll-off: {result.wedge_slope_amplitude:.4f} /THz "
              f"towards plate {result.wedge_direction_deg:.0f} deg",
              f"   90-deg (polarisation) part: {result.polarisation_slope_amplitude:.4f} /THz",
              "   Pure polarisation leaves |R| flat and arg R at 0 or 180 deg."]
    return result, "\n".join(lines)
