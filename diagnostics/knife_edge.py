"""Knife-edge beam profiling from THz-TDS traces: spectral amplitude against blade position.

A blade is stepped across the beam and one ``.acc`` file is taken per position, with the position
in the filename as ``..._<pos>mm...`` (``sample_si-HR_knife-edge_7mm.acc``, ``..._7.5mm_x.acc``).
For every file this module returns the spectral amplitude and intensity at a list of chosen
frequencies, so each frequency gives a flat array of signal against blade position -- the knife-
edge curve at that frequency -- and fits each curve with an error function to get the beam radius.

What the curve measures (and why the fit is on the AMPLITUDE, not the intensity)
-----------------------------------------------------------------------------
A power meter behind a knife edge integrates intensity over the unblocked part of the beam, so the
classic erf curve is a curve in POWER. Electro-optic detection does not measure power: it measures
the field projected onto the detection mode (the probe spot in the crystal, imaged back to the
sample plane). If that mode matches the THz beam, the detected field is the overlap integral
of two equal Gaussians, ``integral exp(-2 x^2 / w^2) dx`` over the unblocked part -- which is the
power-meter erf with the same 1/e^2 intensity radius ``w``. So ``|E(f)|`` against position is the
erf curve, and ``|E(f)|^2`` is its square, not an erf. Both are returned; the fit uses amplitude.

If the detection mode is much smaller than the THz beam (or misaligned to it) the curve becomes
the beam's field profile seen through that mode and ``w`` is no longer the intensity radius.
Treat the fitted radius as "the width of the spot the measurement actually sees", which is the
quantity the ellipsometry blur correction needs anyway.

Structure: pure functions (position parsing, spectral extraction, edge fit) take arrays and return
arrays; ``load_knife_edge_files`` and ``write_table_csv`` are the only file I/O; ``plot_knife_edge``
is the only plotting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

from acquisition_editor import load_file as load_acc_file
from thz_ellipsometry.core.preprocess import transform_traces

__all__ = [
    "KnifeEdgeFile", "KnifeEdgeTable", "EdgeFit",
    "position_from_filename", "load_knife_edge_files", "build_knife_edge_table",
    "edge_model", "fit_knife_edge", "fit_all_frequencies", "write_table_csv",
    "print_table", "print_fits", "plot_knife_edge", "plot_fft_traces",
]

#: ``_<number>mm`` anywhere in the name, the number optionally signed and decimal. The match must
#: be followed by a non-alphanumeric character (or the end), so ``_10mmx`` is not read as 10 mm.
POSITION_PATTERN = re.compile(r"_(-?\d+(?:\.\d+)?)mm(?![A-Za-z0-9])")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnifeEdgeFile:
    filename: str
    position_mm: float
    time_ps: np.ndarray        #: (n_samples,)
    scans: np.ndarray          #: (n_scans, n_samples)


def position_from_filename(filename):
    """Blade position in mm from ``..._<pos>mm...``; None if the name carries no position."""
    matches = POSITION_PATTERN.findall(os.path.basename(filename))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"{filename}: more than one '_<pos>mm' token ({matches}); "
                         "the position is ambiguous")
    return float(matches[0])


def load_knife_edge_files(directory, *, exclude_filename_substrings=(), extension=".acc"):
    """Every file in ``directory`` with a position in its name, sorted by position.

    Files without a ``_<pos>mm`` token are skipped and named in the returned ``skipped`` list, so
    a stray reference file is visible rather than silently dropped.
    """
    loaded, skipped = [], []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(extension):
            continue
        if any(substring in filename for substring in exclude_filename_substrings):
            skipped.append((filename, "excluded by config"))
            continue
        position_mm = position_from_filename(filename)
        if position_mm is None:
            skipped.append((filename, "no '_<pos>mm' token"))
            continue
        raw = load_acc_file(os.path.join(directory, filename))["data"]
        loaded.append(KnifeEdgeFile(filename=filename, position_mm=position_mm,
                                    time_ps=np.asarray(raw[:, 0], dtype=float),
                                    scans=np.asarray(raw[:, 1:].T, dtype=float)))
    loaded.sort(key=lambda knife_edge_file: knife_edge_file.position_mm)
    return loaded, skipped


# ---------------------------------------------------------------------------
# Spectral extraction (pure)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnifeEdgeTable:
    """Spectral signal at chosen frequencies for every blade position.

    Rows are files (sorted by position); columns are the requested frequencies. Column ``j`` of
    ``amplitude`` is the flattened knife-edge curve at ``frequencies_thz[j]``.
    """
    filenames: tuple
    positions_mm: np.ndarray            #: (n_files,)
    frequencies_thz: np.ndarray         #: (n_frequencies,) the bins actually used
    requested_frequencies_thz: np.ndarray
    amplitude: np.ndarray               #: (n_files, n_frequencies) |mean spectrum|
    amplitude_error: np.ndarray         #: standard error from scan-to-scan scatter; nan if 1 scan
    scan_counts: np.ndarray             #: (n_files,)
    resolution_thz: float               #: 1 / record length: bins closer than this are not independent
    # What was transformed, kept so the FFT and its window can be inspected (plot_fft_traces).
    time_ps: np.ndarray                 #: (n_samples,)
    raw_traces: np.ndarray              #: (n_files, n_samples) scan-averaged, as loaded
    window: np.ndarray                  #: (n_samples,) the window applied; ones if none
    spectrum_frequencies_thz: np.ndarray    #: (n_grid,) full FFT grid
    mean_spectra: np.ndarray            #: (n_files, n_grid) complex, scan-averaged
    padded_length: int                  #: FFT length

    def transformed_traces(self):
        """The time traces the FFT actually saw (baseline removed, windowed), per file.

        Recovered by inverting ``mean_spectra``, not by re-applying the window, so a mistake in
        the transform itself shows up here instead of being reproduced.
        """
        return np.fft.irfft(self.mean_spectra, n=self.padded_length,
                            axis=1)[:, :self.time_ps.size]

    @property
    def intensity(self):
        return self.amplitude ** 2

    @property
    def intensity_error(self):
        return 2.0 * self.amplitude * self.amplitude_error

    def quantity(self, name):
        """``'amplitude'`` or ``'intensity'`` with its error, as ``(values, errors)``."""
        if name not in ("amplitude", "intensity"):
            raise ValueError(f"unknown quantity {name!r}; use 'amplitude' or 'intensity'")
        return getattr(self, name), getattr(self, f"{name}_error")

    def as_columns(self):
        """Flat named columns: ``position_mm`` then ``amplitude_<f>THz`` / ``intensity_<f>THz``."""
        columns = {"position_mm": self.positions_mm}
        for quantity_name in ("amplitude", "intensity"):
            values, errors = self.quantity(quantity_name)
            for column_index, frequency_thz in enumerate(self.frequencies_thz):
                label = f"{quantity_name}_{frequency_thz:.3f}THz"
                columns[label] = values[:, column_index]
                columns[f"{label}_error"] = errors[:, column_index]
        return columns


def build_knife_edge_table(knife_edge_files, frequencies_thz, *, window_half_width_ps=None,
                           window_shape="tukey", taper_fraction=0.5, pad_factor=4,
                           baseline_fraction=0.1):
    """Spectral amplitude of every file at the requested frequencies.

    All scans of all files go through ONE call to the transform, so they share one window centre
    and one window: a blade that also shifts the pulse in time (a translation axis not parallel to
    the beam) is not then re-centred per file. The amplitude is that of the scan-averaged complex
    spectrum; its error is the scan-to-scan standard error of ``|E|``.
    """
    if not knife_edge_files:
        raise ValueError("no knife-edge files to analyse")
    reference_time_ps = knife_edge_files[0].time_ps
    for knife_edge_file in knife_edge_files[1:]:
        if (knife_edge_file.time_ps.shape != reference_time_ps.shape
                or not np.allclose(knife_edge_file.time_ps, reference_time_ps)):
            raise ValueError(f"{knife_edge_file.filename} has a different time axis from "
                             f"{knife_edge_files[0].filename}; take every position with the "
                             "same scan range")

    all_scans = np.vstack([knife_edge_file.scans for knife_edge_file in knife_edge_files])
    transformed = transform_traces(reference_time_ps, all_scans,
                                   window_half_width_ps=window_half_width_ps,
                                   window_shape=window_shape, taper_fraction=taper_fraction,
                                   pad_factor=pad_factor, baseline_fraction=baseline_fraction)
    grid_thz = transformed.frequencies_hz / 1e12
    requested = np.atleast_1d(np.asarray(frequencies_thz, dtype=float))
    bin_indices = np.array([int(np.argmin(np.abs(grid_thz - frequency)))
                            for frequency in requested])

    amplitude_rows, error_rows, scan_counts, mean_spectra = [], [], [], []
    first_row = 0
    for knife_edge_file in knife_edge_files:
        scan_count = knife_edge_file.scans.shape[0]
        file_spectra = transformed.spectra[first_row:first_row + scan_count]
        first_row += scan_count
        mean_spectra.append(file_spectra.mean(axis=0))
        spectra = file_spectra[:, bin_indices]
        amplitude_rows.append(np.abs(spectra.mean(axis=0)))
        if scan_count > 1:
            error_rows.append(np.abs(spectra).std(axis=0, ddof=1) / np.sqrt(scan_count))
        else:
            error_rows.append(np.full(bin_indices.size, np.nan))
        scan_counts.append(scan_count)

    record_length_ps = reference_time_ps[-1] - reference_time_ps[0]
    return KnifeEdgeTable(
        filenames=tuple(knife_edge_file.filename for knife_edge_file in knife_edge_files),
        positions_mm=np.array([knife_edge_file.position_mm
                               for knife_edge_file in knife_edge_files]),
        frequencies_thz=grid_thz[bin_indices],
        requested_frequencies_thz=requested,
        amplitude=np.array(amplitude_rows),
        amplitude_error=np.array(error_rows),
        scan_counts=np.array(scan_counts),
        resolution_thz=1.0 / record_length_ps,
        time_ps=reference_time_ps,
        raw_traces=np.array([knife_edge_file.scans.mean(axis=0)
                             for knife_edge_file in knife_edge_files]),
        window=transformed.window,
        spectrum_frequencies_thz=grid_thz,
        mean_spectra=np.array(mean_spectra),
        padded_length=transformed.padded_length,
    )


# ---------------------------------------------------------------------------
# Edge fit (pure)
# ---------------------------------------------------------------------------

def edge_model(position_mm, edge_centre_mm, beam_radius_mm, step_height, floor):
    """Knife-edge curve of a Gaussian beam with 1/e^2 intensity radius ``beam_radius_mm``.

    ``step_height`` is signed: negative when the blade enters the beam as the position increases.
    """
    argument = np.sqrt(2.0) * (np.asarray(position_mm) - edge_centre_mm) / beam_radius_mm
    return floor + 0.5 * step_height * (1.0 + erf(argument))


@dataclass(frozen=True)
class EdgeFit:
    frequency_thz: float
    edge_centre_mm: float
    edge_centre_error_mm: float
    beam_radius_mm: float               #: 1/e^2 intensity radius (see the module docstring)
    beam_radius_error_mm: float
    step_height: float
    floor: float
    covered_fraction: float             #: share of the full step the measured range spans
    reduced_chi_square: float = float("nan")   #: nan when there were no scan-to-scan errors
    warnings: tuple = field(default=())

    @property
    def trusted(self):
        return not self.warnings


def fit_knife_edge(positions_mm, values, errors=None, *, frequency_thz=float("nan"),
                   minimum_covered_fraction=0.8, maximum_reduced_chi_square=10.0):
    """Fit one knife-edge curve. Never raises on a bad fit: it reports it in ``warnings``.

    ``covered_fraction`` is how much of the fitted step lies inside the measured positions. Below
    ``minimum_covered_fraction`` the plateau or the floor was extrapolated, and the radius is
    only as good as that extrapolation -- the usual cause is a scan that stops before the blade
    has fully blocked (or fully cleared) the beam.
    """
    positions_mm = np.asarray(positions_mm, dtype=float)
    values = np.asarray(values, dtype=float)
    sigma = None
    if errors is not None:
        errors = np.asarray(errors, dtype=float)
        if np.all(np.isfinite(errors)) and np.all(errors > 0):
            sigma = errors

    span_mm = float(np.ptp(positions_mm)) or 1.0
    order = np.argsort(positions_mm)
    first_value, last_value = values[order[0]], values[order[-1]]
    halfway_value = 0.5 * (first_value + last_value)
    crossing_index = int(np.argmin(np.abs(values - halfway_value)))
    initial_guess = (positions_mm[crossing_index], span_mm / 4.0,
                     last_value - first_value, min(first_value, last_value))

    warnings = []
    try:
        parameters, covariance = curve_fit(
            edge_model, positions_mm, values, p0=initial_guess, sigma=sigma,
            absolute_sigma=sigma is not None, maxfev=20000,
            bounds=([-np.inf, 1e-3 * span_mm, -np.inf, -np.inf],
                    [np.inf, 10.0 * span_mm, np.inf, np.inf]))
        parameter_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    except (RuntimeError, ValueError) as error:
        return EdgeFit(frequency_thz, *(float("nan"),) * 6, 0.0, float("nan"),
                       warnings=(f"fit failed: {error}",))

    edge_centre_mm, beam_radius_mm, step_height, floor = (float(value) for value in parameters)
    # Scan-to-scan errors only see noise. When the erf misses the points by more than that (edge
    # diffraction ripple, a non-Gaussian beam, drift between positions) chi-square says so, and
    # the parameter errors are scaled up by sqrt(chi-square) rather than reported at noise level.
    reduced_chi_square = float("nan")
    degrees_of_freedom = positions_mm.size - len(parameters)
    if sigma is not None and degrees_of_freedom > 0:
        residuals = (values - edge_model(positions_mm, *parameters)) / sigma
        reduced_chi_square = float(np.sum(residuals ** 2) / degrees_of_freedom)
        if reduced_chi_square > 1.0:
            parameter_errors = parameter_errors * np.sqrt(reduced_chi_square)
    fitted_start, fitted_end = edge_model(positions_mm[order[[0, -1]]], *parameters)
    covered_fraction = float(abs(fitted_end - fitted_start) / abs(step_height)) \
        if step_height else 0.0
    if covered_fraction < minimum_covered_fraction:
        warnings.append(f"scan covers only {100 * covered_fraction:.0f}% of the edge: "
                        "extend the positions until the signal stops changing at both ends")
    if reduced_chi_square > maximum_reduced_chi_square:
        warnings.append(f"erf misses the points by {np.sqrt(reduced_chi_square):.0f}x the "
                        "scan-to-scan noise (edge-diffraction ripple, non-Gaussian beam, or "
                        "drift between positions): radius is a shape summary, errors inflated")
    if not np.all(np.isfinite(parameter_errors)):
        warnings.append("covariance undetermined: too few positions for four parameters")
    elif parameter_errors[1] > 0.5 * beam_radius_mm:
        warnings.append("radius uncertain by more than 50%")
    return EdgeFit(frequency_thz=float(frequency_thz), edge_centre_mm=edge_centre_mm,
                   edge_centre_error_mm=float(parameter_errors[0]),
                   beam_radius_mm=beam_radius_mm,
                   beam_radius_error_mm=float(parameter_errors[1]),
                   step_height=step_height, floor=floor, covered_fraction=covered_fraction,
                   reduced_chi_square=reduced_chi_square, warnings=tuple(warnings))


def fit_all_frequencies(table, *, minimum_covered_fraction=0.8, maximum_reduced_chi_square=10.0):
    """One amplitude edge fit per frequency column of ``table``."""
    return [fit_knife_edge(table.positions_mm, table.amplitude[:, column_index],
                           table.amplitude_error[:, column_index],
                           frequency_thz=frequency_thz,
                           minimum_covered_fraction=minimum_covered_fraction,
                           maximum_reduced_chi_square=maximum_reduced_chi_square)
            for column_index, frequency_thz in enumerate(table.frequencies_thz)]


# ---------------------------------------------------------------------------
# Reporting, export, plotting
# ---------------------------------------------------------------------------

def print_table(table):
    print(f"[knife_edge] {len(table.filenames)} positions, "
          f"{table.positions_mm.min():g} to {table.positions_mm.max():g} mm; "
          f"frequency resolution {table.resolution_thz:.3f} THz (1 / record length)")
    header = "  pos (mm)  scans  " + "  ".join(f"|E| {frequency:5.2f}THz"
                                             for frequency in table.frequencies_thz)
    print(header)
    for row_index, position_mm in enumerate(table.positions_mm):
        cells = "  ".join(f"{value:14.4g}" for value in table.amplitude[row_index])
        print(f"  {position_mm:8.2f}  {table.scan_counts[row_index]:5d}  {cells}")
    duplicated = [position for position in np.unique(table.positions_mm)
                  if np.sum(table.positions_mm == position) > 1]
    if duplicated:
        print(f"[knife_edge] WARNING: several files at {duplicated} mm -- all are kept as "
              "separate points; exclude extras with exclude_filename_substrings")


def print_fits(edge_fits):
    print("[knife_edge] erf fit to |E| (radius = 1/e^2 intensity radius)")
    print("  f (THz)   radius (mm)        edge centre (mm)   covered  chi2_red")
    for edge_fit in edge_fits:
        print(f"  {edge_fit.frequency_thz:6.2f}   "
              f"{edge_fit.beam_radius_mm:5.2f} +/- {edge_fit.beam_radius_error_mm:<5.2f}   "
              f"{edge_fit.edge_centre_mm:6.2f} +/- {edge_fit.edge_centre_error_mm:<5.2f}   "
              f"{100 * edge_fit.covered_fraction:4.0f}%   {edge_fit.reduced_chi_square:7.1f}")
        for warning in edge_fit.warnings:
            print(f"           WARNING: {warning}")


def write_table_csv(table, path):
    """The flattened columns of ``table`` as one CSV. Returns the path written."""
    columns = table.as_columns()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savetxt(path, np.column_stack(list(columns.values())), delimiter=",",
               header=",".join(columns), comments="")
    return path


def plot_knife_edge(table, edge_fits=None, *, quantities=("amplitude", "intensity"),
                    normalise=True):
    """One panel per quantity against position, one line per frequency, plus radius vs frequency.

    ``normalise`` divides every frequency's curve by its own maximum so the edges overlay; the
    beam-size comparison is then by eye.
    """
    import matplotlib.pyplot as plt

    show_radius = bool(edge_fits)
    panel_count = len(quantities) + int(show_radius)
    figure, axes = plt.subplots(1, panel_count, figsize=(5.0 * panel_count, 4.2), squeeze=False)
    axes = axes[0]
    colours = plt.cm.viridis(np.linspace(0.0, 0.9, table.frequencies_thz.size))
    dense_positions = np.linspace(table.positions_mm.min(), table.positions_mm.max(), 300)

    for axis, quantity_name in zip(axes, quantities):
        values, errors = table.quantity(quantity_name)
        for column_index, frequency_thz in enumerate(table.frequencies_thz):
            scale = np.nanmax(values[:, column_index]) if normalise else 1.0
            axis.errorbar(table.positions_mm, values[:, column_index] / scale,
                          yerr=errors[:, column_index] / scale, fmt="o", markersize=4,
                          color=colours[column_index], label=f"{frequency_thz:.2f} THz")
            if edge_fits and np.isfinite(edge_fits[column_index].beam_radius_mm):
                edge_fit = edge_fits[column_index]
                fitted = edge_model(dense_positions, edge_fit.edge_centre_mm,
                                    edge_fit.beam_radius_mm, edge_fit.step_height, edge_fit.floor)
                if quantity_name == "intensity":
                    fitted = fitted ** 2
                axis.plot(dense_positions, fitted / scale, "-", color=colours[column_index],
                          linewidth=1.0, alpha=0.8)
        unit = "normalised" if normalise else "arb."
        symbol = "|E(f)|" if quantity_name == "amplitude" else "|E(f)|^2"
        axis.set(xlabel="blade position (mm)", ylabel=f"{symbol} ({unit})",
                 title=f"Knife edge: {quantity_name}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    if show_radius:
        axis = axes[-1]
        for edge_fit, colour in zip(edge_fits, colours):
            marker = "o" if edge_fit.trusted else "x"
            axis.errorbar(edge_fit.frequency_thz, edge_fit.beam_radius_mm,
                          yerr=edge_fit.beam_radius_error_mm, fmt=marker, color=colour)
        axis.set(xlabel="frequency (THz)", ylabel="1/e^2 radius (mm)",
                 title="Fitted beam radius (x = untrusted)")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_fft_traces(table, *, spectrum_max_thz=6.0, show_raw_traces=True):
    """Check the transform: time traces with the window, and the spectra with the traced bins.

    Left: for every position, the trace the FFT saw (solid; baseline removed and windowed,
    recovered from the spectrum) and optionally the raw averaged trace (faint), with the window
    drawn scaled to the largest pulse. The pulse must sit in the window's flat top and the window
    must reach zero before the record ends, or truncation ripple appears in the spectrum.
    Right: |E(f)| on a log scale per position, with the traced frequencies marked.
    """
    import matplotlib.pyplot as plt

    figure, (time_axis, spectrum_axis) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = plt.cm.viridis(np.linspace(0.0, 0.9, table.positions_mm.size))
    transformed = table.transformed_traces()

    for row_index, position_mm in enumerate(table.positions_mm):
        colour = colours[row_index]
        if show_raw_traces:
            time_axis.plot(table.time_ps, table.raw_traces[row_index], color=colour,
                           linewidth=0.8, alpha=0.35)
        time_axis.plot(table.time_ps, transformed[row_index], color=colour, linewidth=1.2,
                       label=f"{position_mm:g} mm")
        spectrum_axis.semilogy(table.spectrum_frequencies_thz,
                               np.abs(table.mean_spectra[row_index]), color=colour,
                               linewidth=1.0)

    window_scale = np.max(np.abs(transformed))
    time_axis.plot(table.time_ps, window_scale * table.window, "k--", linewidth=1.0,
                   label="window (scaled)")
    time_axis.set(xlabel="time (ps)", ylabel="signal",
                  title="Time traces: as transformed (solid), raw (faint)"
                  if show_raw_traces else "Time traces as transformed")
    time_axis.legend(fontsize=7, ncol=2)
    time_axis.grid(alpha=0.3)

    for frequency_thz in table.frequencies_thz:
        spectrum_axis.axvline(frequency_thz, color="grey", linestyle=":", linewidth=1.0)
    in_range = table.spectrum_frequencies_thz <= spectrum_max_thz
    shown = np.abs(table.mean_spectra[:, in_range])
    spectrum_axis.set_ylim(max(shown[shown > 0].min(), shown.max() * 1e-5), shown.max() * 2.0)
    spectrum_axis.set(xlim=(0.0, spectrum_max_thz), xlabel="frequency (THz)",
                      ylabel="|E(f)|",
                      title=f"Spectra (dotted = traced bins; resolution "
                            f"{table.resolution_thz:.2f} THz)")
    spectrum_axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    return figure
