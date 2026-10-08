"""The inspection figures, one or two per processing step, drawn from an InspectionRecord.

Drawing only: every function takes an ``InspectionRecord`` and returns a matplotlib Figure without
showing or saving it. Each is registered at the step (``stages.CHECKPOINTS``) whose output it
shows; ``docs/ELLIPSOMETRY_HARMONIC_FIT_TUTORIAL.md`` explains how to read them.

One column per fitted series (the gold reference first, then the sample), so a reference and
a sample can be compared step by step. Colours follow the polarisation state everywhere.
"""

from __future__ import annotations

import numpy as np
from thz_core.thz_core import conditioning

from .inspection import inspection_figure

__all__ = []

STATE_COLOURS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown",
                 "tab:pink", "tab:olive")


def _state_colour(entry, state_deg):
    position = int(np.nonzero(np.isclose(entry.states_deg, state_deg))[0][0])
    return STATE_COLOURS[position % len(STATE_COLOURS)]


def _columns(record, rows, height_per_row=2.6, sharex="col"):
    import matplotlib.pyplot as plt
    count = max(len(record.series), 1)
    figure, axes = plt.subplots(rows, count, figsize=(5.6 * count, height_per_row * rows + 0.8),
                                squeeze=False, sharex=sharex, constrained_layout=True)
    return figure, axes


def _minutes(entry):
    if entry.row_elapsed_s is None:
        return np.arange(entry.row_traces.shape[0], dtype=float), "row number"
    return (entry.row_elapsed_s - np.min(entry.row_elapsed_s)) / 60.0, "minutes since first scan"


def _segment_starts(entry, minutes):
    if entry.row_segment_ids is None:
        return []
    starts = [minutes[entry.row_segment_ids == segment].min()
              for segment in np.unique(entry.row_segment_ids)]
    return sorted(starts)[1:]


def _shade_band(axis, record, entry):
    if record.band is None or entry.frequencies_hz is None:
        return
    band_thz = entry.frequencies_hz[record.band] / 1e12
    axis.axvspan(band_thz.min(), band_thz.max(), color="0.85", alpha=0.5, linewidth=0,
                 zorder=0)


def _band_axis(record, entry):
    return entry.frequencies_hz[record.band]


def _phase_slope_delay_fs(frequencies_hz, ratio, weights=None):
    """(delay fs, offset deg) of a straight line through arg(ratio) vs angular frequency."""
    phase = np.unwrap(np.angle(ratio))
    slope, intercept = np.polyfit(2 * np.pi * frequencies_hz, phase, 1, w=weights)
    return -slope * 1e15, float(np.rad2deg(intercept))


# ---------------------------------------------------------------------------
# raw_traces
# ---------------------------------------------------------------------------

@inspection_figure("raw_traces")
def raw_scans_over_time(record):
    """Every scan as loaded, coloured by time, and its peak-to-peak per polarisation state."""
    import matplotlib.pyplot as plt
    figure, axes = _columns(record, 2, sharex=False)
    for column, entry in enumerate(record.series):
        trace_axis, peak_axis = axes[0, column], axes[1, column]
        minutes, minutes_label = _minutes(entry)
        colour_scale = plt.Normalize(minutes.min(), max(minutes.max(), minutes.min() + 1e-9))
        colour_map = plt.get_cmap("viridis")
        for trace, minute in zip(entry.row_traces, minutes):
            trace_axis.plot(entry.time_ps, trace * 1e3, color=colour_map(colour_scale(minute)),
                            linewidth=0.6, alpha=0.7)
        baseline_end = entry.time_ps[min(entry.baseline_samples, entry.time_ps.size) - 1]
        trace_axis.axvspan(entry.time_ps[0], baseline_end, color="0.6", alpha=0.25,
                           linewidth=0, label="baseline region")
        trace_axis.set_title(f"{entry.label}\n{entry.row_traces.shape[0]} rows", fontsize=9)
        trace_axis.set_xlabel("time [ps]")
        trace_axis.set_ylabel("signal as loaded [mV]")
        trace_axis.legend(fontsize=7, loc="upper right")
        figure.colorbar(plt.cm.ScalarMappable(norm=colour_scale, cmap=colour_map),
                        ax=trace_axis, label=minutes_label, shrink=0.8)

        peak_to_peak = np.ptp(entry.baseline_corrected_traces, axis=1) * 1e3
        for state in entry.states_deg:
            selected = entry.rows_in_state(state)
            peak_axis.plot(minutes[selected], peak_to_peak[selected], "o", markersize=3,
                           color=_state_colour(entry, state), label=f"{state:.1f} deg")
        for start in _segment_starts(entry, minutes):
            peak_axis.axvline(start, color="0.6", linestyle=":", linewidth=0.8)
        peak_axis.set_xlabel(minutes_label)
        peak_axis.set_ylabel("peak-to-peak [mV]")
        peak_axis.legend(fontsize=7, title="polarisation", title_fontsize=7)
    figure.suptitle("1. Raw traces: what was recorded (dotted line = new file / box opening)")
    return figure


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------

def _full_window(entry):
    """The window's full extent, including any part that fell outside the record."""
    if entry.window_half_width_samples < 1:
        return None, None
    half = entry.window_half_width_samples
    weights = conditioning.apodization_window(2 * half + 1, entry.window_shape,
                                              entry.taper_fraction)
    step = float(np.mean(np.diff(entry.time_ps)))
    first = entry.window_centre_index - half
    return entry.time_ps[0] + (first + np.arange(weights.size)) * step, weights


@inspection_figure("windowing")
def window_on_traces(record):
    """The mean trace per state with the window over it, and what the FFT actually receives."""
    figure, axes = _columns(record, 2)
    for column, entry in enumerate(record.series):
        raw_axis, windowed_axis = axes[0, column], axes[1, column]
        corrected = entry.baseline_corrected_traces
        peak = 1e3 * max(max(float(np.max(np.abs(corrected[entry.rows_in_state(state)]
                                                  .mean(axis=0))))
                             for state in entry.states_deg), 1e-30)
        for state in entry.states_deg:
            mean_trace = corrected[entry.rows_in_state(state)].mean(axis=0) * 1e3
            colour = _state_colour(entry, state)
            raw_axis.plot(entry.time_ps, mean_trace, color=colour, linewidth=1.0,
                          label=f"{state:.1f} deg")
            if entry.window is not None:
                windowed_axis.plot(entry.time_ps, mean_trace * entry.window, color=colour,
                                   linewidth=1.0)
        if entry.window is not None:
            raw_axis.plot(entry.time_ps, entry.window * peak, color="k", linewidth=1.2,
                          label="window x edge taper (applied)")
            full_time, full_weights = _full_window(entry)
            if full_time is not None:
                raw_axis.plot(full_time, full_weights * peak, color="k", linestyle=":",
                              linewidth=1.0, label=f"{entry.window_shape} window, full extent")
            step = float(np.mean(np.diff(entry.time_ps)))
            if entry.window_clipped_before:
                raw_axis.axvspan(entry.time_ps[0] - entry.window_clipped_before * step,
                                 entry.time_ps[0], color="tab:red", alpha=0.15, linewidth=0,
                                 label="window outside the record (cut)")
            if entry.window_clipped_after:
                raw_axis.axvspan(entry.time_ps[-1],
                                 entry.time_ps[-1] + entry.window_clipped_after * step,
                                 color="tab:red", alpha=0.15, linewidth=0)
            windowed_axis.plot(entry.time_ps, entry.window * peak, color="k", linewidth=0.8,
                               alpha=0.5)
        raw_axis.axvspan(entry.time_ps[0], entry.time_ps[entry.baseline_samples - 1],
                         color="0.6", alpha=0.25, linewidth=0, label="baseline region")
        raw_axis.set_title(f"{entry.label}: baseline-corrected mean per state", fontsize=9)
        raw_axis.set_ylabel("signal [mV]")
        raw_axis.legend(fontsize=6, loc="upper right")
        cut = (entry.window_clipped_before + entry.window_clipped_after) * float(
            np.mean(np.diff(entry.time_ps)))
        windowed_axis.set_title(
            "windowed: what is Fourier transformed"
            + (f" ({cut:.2f} ps of window cut by the record)" if cut else ""), fontsize=9)
        windowed_axis.set_xlabel("time [ps]")
        windowed_axis.set_ylabel("signal x window [mV]")
    figure.suptitle(f"2. Windowing: one window at one centre for every row "
                    f"({record.series[0].window_shape if record.series else ''}, edge taper "
                    f"{record.series[0].edge_taper_ps if record.series else 0:g} ps)")
    return figure


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

@inspection_figure("spectra")
def state_spectra(record):
    """Amplitude (with the noise floor) and phase (linear part removed) of each state's mean."""
    figure, axes = _columns(record, 2)
    for column, entry in enumerate(record.series):
        if entry.row_spectra is None:
            continue
        amplitude_axis, phase_axis = axes[0, column], axes[1, column]
        frequencies_thz = entry.frequencies_hz / 1e12
        shown = frequencies_thz <= 5.0
        overall = np.abs(entry.row_spectra).mean(axis=0)
        strong = (overall > 0.05 * overall.max()) & shown
        for state in entry.states_deg:
            selected = entry.rows_in_state(state)
            mean_spectrum = entry.row_spectra[selected].mean(axis=0)
            colour = _state_colour(entry, state)
            amplitude_axis.plot(frequencies_thz[shown],
                                20 * np.log10(np.abs(mean_spectrum[shown]) + 1e-30),
                                color=colour, linewidth=1.0, label=f"{state:.1f} deg")
            if entry.row_variance is not None:
                floor = np.sqrt(entry.row_variance[selected].mean(axis=0) / selected.sum())
                amplitude_axis.plot(frequencies_thz[shown], 20 * np.log10(floor[shown] + 1e-30),
                                    color=colour, linewidth=0.8, linestyle="--", alpha=0.7)
            phase = np.unwrap(np.angle(mean_spectrum))
            line = np.polyfit(frequencies_thz[strong], phase[strong], 1)
            phase_axis.plot(frequencies_thz[strong],
                            np.rad2deg(phase[strong] - np.polyval(line, frequencies_thz[strong])),
                            color=colour, linewidth=1.0)
        _shade_band(amplitude_axis, record, entry)
        _shade_band(phase_axis, record, entry)
        resolution = ("" if not np.isfinite(entry.resolution_df_hz) else
                      f"; resolution {entry.resolution_df_hz / 1e12:.3f} THz, "
                      f"{entry.oversampling:.1f} bins per independent point")
        amplitude_axis.set_title(f"{entry.label}{resolution}", fontsize=9)
        amplitude_axis.set_ylabel("|S| [dB]  (dashed: noise of the mean)")
        amplitude_axis.legend(fontsize=7, title="polarisation", title_fontsize=7)
        phase_axis.set_ylabel("phase - linear fit [deg]")
        phase_axis.set_xlabel("frequency [THz]")
        phase_axis.set_title("phase with the pulse's arrival time (linear part) removed",
                             fontsize=9)
    figure.suptitle("3. Spectra per polarisation state (grey band = the trusted band, when chosen)")
    return figure


# ---------------------------------------------------------------------------
# harmonic_fit
# ---------------------------------------------------------------------------

def _mark_independent(axis, record, frequencies_thz, values, colour):
    if record.independent_band_points is None:
        return
    points = record.independent_band_points
    axis.plot(frequencies_thz[points], values[points], "o", markersize=3, color=colour)


@inspection_figure("harmonic_fit")
def harmonic_components(record):
    """What the harmonic fit extracted: P, Q, background, and the channel ratio P/Q."""
    figure, axes = _columns(record, 3)
    for column, entry in enumerate(record.series):
        if entry.channel_p is None:
            continue
        component_axis, magnitude_axis, phase_axis = axes[:, column]
        frequencies_thz = entry.frequencies_hz / 1e12
        shown = frequencies_thz <= 5.0
        for values, name, colour in ((entry.channel_p, "P (p channel)", "tab:blue"),
                                     (entry.channel_s, "Q (s channel)", "tab:orange"),
                                     (entry.background, "B (background)", "tab:gray")):
            if values is None:
                continue
            component_axis.plot(frequencies_thz[shown],
                                20 * np.log10(np.abs(values[shown]) + 1e-30), color=colour,
                                label=name)
        for variance, colour in ((entry.variance_p, "tab:blue"), (entry.variance_s, "tab:orange")):
            if variance is not None:
                component_axis.plot(frequencies_thz[shown],
                                    20 * np.log10(np.sqrt(variance[shown]) + 1e-30),
                                    color=colour, linestyle="--", linewidth=0.8, alpha=0.7)
        _shade_band(component_axis, record, entry)
        component_axis.set_ylabel("|component| [dB]  (dashed: its error)")
        component_axis.legend(fontsize=7)
        title = entry.label
        if np.isfinite(entry.reduced_chi_square):
            title += f"\nreduced chi-square {entry.reduced_chi_square:.2f} (1 = fits to the noise)"
        component_axis.set_title(title, fontsize=9)

        if record.band is None:
            continue
        band_frequencies = _band_axis(record, entry)
        band_thz = band_frequencies / 1e12
        ratio = (entry.channel_p / entry.channel_s)[record.band]
        magnitude_axis.plot(band_thz, np.abs(ratio), color="tab:purple", alpha=0.5)
        _mark_independent(magnitude_axis, record, band_thz, np.abs(ratio), "tab:purple")
        magnitude_axis.set_ylabel("|P/Q|")
        phase = np.rad2deg(np.unwrap(np.angle(ratio)))
        phase_axis.plot(band_thz, phase, color="tab:purple", alpha=0.5)
        _mark_independent(phase_axis, record, band_thz, phase, "tab:purple")
        delay_fs, offset_deg = _phase_slope_delay_fs(band_frequencies, ratio,
                                                     weights=np.abs(entry.channel_s[record.band]))
        phase_axis.set_ylabel("arg(P/Q) [deg]")
        phase_axis.set_xlabel("frequency [THz]")
        phase_axis.set_title(f"p/s delay in P/Q {delay_fs:+.2f} fs, offset {offset_deg:+.1f} deg "
                             "(instrument x sample)", fontsize=9)
    figure.suptitle("4a. Harmonic fit: the two channels and their ratio "
                    "(P/Q = C x rho; markers = independent points)")
    return figure


def _arrival_delays_fs(entry):
    """The fit's delays as arrival delays: later = positive (the fit's tau is the opposite)."""
    return -entry.delays_s * 1e15


def _row_delays_and_scales(entry, band):
    """Each row's OWN arrival delay (fs, later = positive) and scale, measured against the
    fitted model with its drift taken out."""
    frequencies = entry.frequencies_hz[band]
    model_without_drift = (entry.predicted_spectra[:, band]
                           * np.exp(-2j * np.pi * frequencies[None, :]
                                    * entry.delays_s[:, None]))
    unscaled = model_without_drift / entry.row_scales[:, band]
    measured_delays, measured_scales = [], []
    for spectrum, model, plain in zip(entry.row_spectra[:, band], model_without_drift, unscaled):
        weights = np.abs(model)
        ratio = spectrum / np.where(np.abs(model) > 0, model, 1.0)
        delay_fs, _ = _phase_slope_delay_fs(frequencies, ratio, weights=weights)
        measured_delays.append(delay_fs)
        measured_scales.append(float(np.sum(np.abs(spectrum) * weights)
                                     / np.sum(np.abs(plain) * weights)))
    return np.array(measured_delays), np.array(measured_scales)


@inspection_figure("harmonic_fit")
def harmonic_drift_and_residuals(record):
    """The drift the fit removed, each row's own delay and scale against it, and what is left."""
    figure, axes = _columns(record, 4, sharex=False)
    for column, entry in enumerate(record.series):
        if entry.predicted_spectra is None or record.band is None:
            continue
        delay_axis, scale_axis, row_axis, frequency_axis = axes[:, column]
        minutes, minutes_label = _minutes(entry)
        band = record.band
        measured_delays, measured_scales = _row_delays_and_scales(entry, band)
        frequencies = entry.frequencies_hz[band]
        reference_bin = int(np.argmin(np.abs(frequencies - 1e12)))
        model_scale = entry.row_scales[:, band][:, reference_bin]
        # Each row is compared with the fitted P, Q, B WITHOUT its drift, so its own delay and
        # scale come out on the same footing as the model's (first row = 0 fs, scale 1).
        model_delays = _arrival_delays_fs(entry)
        for state in entry.states_deg:
            selected = entry.rows_in_state(state)
            colour = _state_colour(entry, state)
            delay_axis.plot(minutes[selected], measured_delays[selected], "o", markersize=3,
                            color=colour, label=f"{state:.1f} deg")
            scale_axis.plot(minutes[selected], measured_scales[selected], "o", markersize=3,
                            color=colour)
        order = np.argsort(minutes)
        delay_axis.plot(minutes[order], model_delays[order], color="k", linewidth=1.0,
                        label=f"fitted drift ({entry.drift_model})")
        scale_axis.plot(minutes[order], model_scale[order], color="k", linewidth=1.0,
                        label=f"fitted scale at 1 THz ({entry.amplitude_model})")
        for axis in (delay_axis, scale_axis, row_axis):
            for start in _segment_starts(entry, minutes):
                axis.axvline(start, color="0.6", linestyle=":", linewidth=0.8)
        delay_axis.set_ylabel("arrival delay [fs] (later = +)")
        delay_axis.set_title(f"{entry.label}\npoints: each row's own delay; line: drift model",
                             fontsize=9)
        delay_axis.legend(fontsize=6, ncol=2)
        scale_axis.set_ylabel("scale (first row = 1)")
        scale_axis.legend(fontsize=6)

        residual = entry.row_spectra[:, band] - entry.predicted_spectra[:, band]
        if entry.row_variance is not None:
            normalised = np.abs(residual) ** 2 / entry.row_variance[:, band]
            per_row = np.sqrt(normalised.mean(axis=1))
            per_frequency = np.sqrt(normalised.mean(axis=0))
            row_label, frequency_label = "rms residual / noise", "rms residual / noise"
        else:
            scale = np.abs(entry.row_spectra[:, band])
            per_row = np.sqrt((np.abs(residual) ** 2).mean(axis=1) / (scale ** 2).mean(axis=1))
            per_frequency = np.sqrt((np.abs(residual) ** 2).mean(axis=0)
                                    / (scale ** 2).mean(axis=0))
            row_label, frequency_label = "relative rms residual", "relative rms residual"
        for state in entry.states_deg:
            selected = entry.rows_in_state(state)
            row_axis.plot(minutes[selected], per_row[selected], "o", markersize=3,
                          color=_state_colour(entry, state))
        if entry.row_variance is not None:
            row_axis.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
            frequency_axis.axhline(1.0, color="k", linewidth=0.8, linestyle="--")
        row_axis.set_ylabel(row_label)
        row_axis.set_xlabel(minutes_label)
        frequency_axis.plot(frequencies / 1e12, per_frequency, color="tab:purple", alpha=0.6)
        _mark_independent(frequency_axis, record, frequencies / 1e12, per_frequency,
                          "tab:purple")
        frequency_axis.set_ylabel(frequency_label)
        frequency_axis.set_xlabel("frequency [THz]")
    figure.suptitle("4b. Harmonic fit: drift and scale per row, and the residual by row and by "
                    "frequency (1 = at the noise)")
    return figure


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@inspection_figure("calibration")
def channel_calibration(record):
    """The channel ratio C = d_p/d_s from every known material, against the value applied."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    magnitude_axis, phase_axis, ratio_axis = axes
    if record.band is None or not record.series:
        return figure
    frequencies = record.series[0].frequencies_hz
    band = record.band
    band_thz = frequencies[band] / 1e12
    colours = iter(("tab:orange", "tab:green", "tab:purple", "tab:brown"))
    derived = {}
    for name, ratio in record.calibration_by_material.items():
        colour = next(colours)
        values = np.asarray(ratio)[band]
        derived[name] = values
        delay_fs, offset_deg = _phase_slope_delay_fs(frequencies[band], values)
        label = f"from {name}: delay {delay_fs:+.2f} fs, offset {offset_deg:+.2f} deg"
        magnitude_axis.plot(band_thz, np.abs(values), color=colour, alpha=0.5, label=label)
        _mark_independent(magnitude_axis, record, band_thz, np.abs(values), colour)
        phase = np.rad2deg(np.unwrap(np.angle(values)))
        phase_axis.plot(band_thz, phase, color=colour, alpha=0.5, label=label)
        _mark_independent(phase_axis, record, band_thz, phase, colour)
    applied = record.calibration_applied
    if np.isfinite(applied):
        magnitude_axis.axhline(abs(applied), color="k", linestyle="--",
                               label=f"applied ({record.calibration_name}): one constant")
        phase_axis.axhline(np.rad2deg(np.angle(applied)), color="k", linestyle="--",
                           label="applied: one constant")
    magnitude_axis.set_ylabel("|C|")
    magnitude_axis.set_title("magnitude", fontsize=9)
    phase_axis.set_ylabel("arg C [deg]")
    phase_axis.set_title("phase: a slope is a p/s delay in the instrument", fontsize=9)
    for axis in (magnitude_axis, phase_axis):
        axis.set_xlabel("frequency [THz]")
        axis.legend(fontsize=7)
    names = list(derived)
    if len(names) >= 2:
        ratio = derived[names[1]] / derived[names[0]]
        delay_fs, offset_deg = _phase_slope_delay_fs(frequencies[band], ratio)
        ratio_axis.plot(band_thz, np.abs(ratio), color="tab:purple", label="|ratio|")
        twin = ratio_axis.twinx()
        twin.plot(band_thz, np.rad2deg(np.unwrap(np.angle(ratio))), color="tab:red",
                  label="arg [deg]")
        twin.set_ylabel("arg [deg]", color="tab:red")
        ratio_axis.set_title(f"{names[1]} / {names[0]}: should be 1 at 0 deg\n"
                             f"delay {delay_fs:+.2f} fs, offset {offset_deg:+.2f} deg",
                             fontsize=9)
        ratio_axis.set_ylabel("|ratio|", color="tab:purple")
        ratio_axis.set_xlabel("frequency [THz]")
    else:
        ratio_axis.text(0.5, 0.5, "only one material of known index in this run:\n"
                        "nothing to cross-check C against", ha="center", va="center",
                        transform=ratio_axis.transAxes, fontsize=9)
        ratio_axis.set_axis_off()
    figure.suptitle(f"5. Channel calibration at {record.incidence_angle_deg:.2f} deg incidence "
                    "(C should agree between materials; it is the instrument, not the sample)")
    return figure


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@inspection_figure("result")
def ratio_and_index(record):
    """rho as tan(Psi) and Delta, then n and k with their noise bars and any known reference."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    if record.index is None:
        return figure
    frequencies_thz = record.band_frequencies_hz / 1e12
    points = (record.independent_band_points if record.independent_band_points is not None
              else np.ones(frequencies_thz.size, dtype=bool))
    psi_axis = axes[0]
    psi_axis.plot(frequencies_thz, np.abs(record.ratio), color="tab:blue", label="tan Psi")
    twin = psi_axis.twinx()
    twin.plot(frequencies_thz, np.rad2deg(np.angle(record.ratio)), color="tab:orange",
              label="Delta")
    psi_axis.set_ylabel("tan Psi = |rho|", color="tab:blue")
    twin.set_ylabel("Delta = arg rho [deg]", color="tab:orange")
    psi_axis.set_xlabel("frequency [THz]")
    psi_axis.set_title("ellipsometric ratio rho = r_p / r_s", fontsize=9)
    error = record.index_standard_error
    for axis, values, name, reference in (
            (axes[1], record.index.real, "n", None if record.reference_index is None
             else record.reference_index.real),
            (axes[2], -record.index.imag, "k", None if record.reference_index is None
             else -record.reference_index.imag)):
        axis.plot(frequencies_thz, values, color="tab:blue", alpha=0.45, linewidth=1.0)
        axis.errorbar(frequencies_thz[points], values[points],
                      yerr=None if error is None else error[points], fmt="o", markersize=3.5,
                      capsize=2, color="tab:blue", label=f"{name} (independent points)")
        if reference is not None:
            axis.plot(frequencies_thz, reference, color="k", linestyle="--",
                      label=f"{record.reference_material} reference")
        axis.set_xlabel("frequency [THz]")
        axis.set_ylabel(name)
        axis.legend(fontsize=7)
        axis.set_title(f"{name}: bars are noise only (systematics are in the report)",
                       fontsize=9)
    figure.suptitle("6. Result")
    return figure


@inspection_figure("result")
def run_summary(record):
    """The run report as text: settings, statistics, quality flags and diagnostic findings."""
    import matplotlib.pyplot as plt
    lines = (record.run_report or "(no report: the run did not finish)").splitlines()
    figure = plt.figure(figsize=(12, max(3.0, 0.16 * len(lines) + 0.6)))
    figure.text(0.01, 0.99, "\n".join(lines), family="monospace", fontsize=7, va="top")
    return figure
