"""Figures for the bench tools, so a fit can be checked by eye before its number is used.

Drawing only: every function takes results already computed by ``bench_tools`` and returns a
matplotlib Figure without showing or saving it.

``plot_null_fits``     per sweep: signed amplitude against magnet reading with the reported fit
                       and the lone (no-background) fit, and below it the residuals of both,
                       coloured by acquisition time so drift during a sweep stands out.
``plot_null_traces``   per sweep: the time traces behind the points, coloured by reading, with
                       the window, so the sign reversal through the null can be seen directly.
``plot_settling``      per magnet state: delay and amplitude against time with the fitted
                       exponential settle, its plateau (+/- error) and the file boundaries.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_null_fits", "plot_null_traces", "plot_settling"]


def _null_label(prefix, fit):
    text = f"{prefix}: null {fit.null_deg:.2f} +/- {fit.null_standard_error_deg:.2f} deg"
    if fit.leakage:
        text += f", background {fit.leakage_fraction:.1%}"
    return text


def plot_null_fits(sweeps):
    """Two rows per figure column: data with both fits on top, their residuals below."""
    import matplotlib.pyplot as plt

    if not sweeps:
        raise ValueError("no null sweeps to plot")
    column_count = len(sweeps)
    figure, axes = plt.subplots(2, column_count, figsize=(4.2 * column_count, 6.5),
                                squeeze=False, sharex="col",
                                gridspec_kw={"height_ratios": [2.2, 1.0]},
                                constrained_layout=True)
    timed = [sweep for sweep in sweeps if np.all(np.isfinite(sweep.elapsed_minutes))]
    longest_minutes = max((float(np.max(sweep.elapsed_minutes)) for sweep in timed), default=1.0)
    colour_scale = plt.Normalize(0.0, max(longest_minutes, 1.0))
    scatter = None

    for column, sweep in enumerate(sweeps):
        fit_axis, residual_axis = axes[0, column], axes[1, column]
        readings = sweep.readings_deg
        dense = np.linspace(readings.min() - 2.0, readings.max() + 2.0, 400)

        fit_axis.axhline(0.0, color="0.6", linewidth=0.8)
        fit_axis.plot(dense, sweep.fit.model(dense), color="tab:blue",
                      label=_null_label("reported" + (" (shared sinusoid)"
                                                      if sweep.background_fitted else ""),
                                        sweep.fit))
        fit_axis.axvspan(sweep.fit.null_deg - sweep.fit.null_standard_error_deg,
                         sweep.fit.null_deg + sweep.fit.null_standard_error_deg,
                         color="tab:blue", alpha=0.15, linewidth=0)
        fit_axis.axvline(sweep.fit.null_deg, color="tab:blue", linewidth=1.0)
        if sweep.background_fitted:
            fit_axis.plot(dense, sweep.lone_fit.model(dense), color="tab:orange",
                          linestyle="--", label=_null_label("lone", sweep.lone_fit))
            fit_axis.axvline(sweep.lone_fit.null_deg, color="tab:orange", linestyle="--",
                             linewidth=1.0)
        fit_axis.plot(readings, sweep.signed_amplitudes, "o", color="k", markersize=4,
                      label="measured", zorder=3)
        fit_axis.set_title(sweep.name, fontsize=10)
        fit_axis.set_ylabel("signed amplitude (strongest = +/-1)")
        fit_axis.legend(fontsize=7, loc="best")

        colours = (sweep.elapsed_minutes if np.all(np.isfinite(sweep.elapsed_minutes))
                   else np.zeros(readings.size))
        residual_axis.axhline(0.0, color="0.6", linewidth=0.8)
        scatter = residual_axis.scatter(readings, sweep.fit.residuals, c=colours,
                                        norm=colour_scale, cmap="viridis", marker="o", s=22,
                                        label=f"reported, rms {sweep.fit.residual_rms:.4f}",
                                        zorder=3)
        if sweep.background_fitted:
            residual_axis.scatter(readings, sweep.lone_fit.residuals, marker="s", s=26,
                                  facecolors="none",
                                  edgecolors=plt.get_cmap("viridis")(colour_scale(colours)),
                                  label=f"lone, rms {sweep.lone_fit.residual_rms:.4f}",
                                  zorder=3)
        residual_axis.set_xlabel("magnet reading [deg]")
        residual_axis.set_ylabel("residual")
        residual_axis.legend(fontsize=7, loc="best")

    if scatter is not None and timed:
        figure.colorbar(scatter, ax=axes[1, :].tolist(), shrink=0.9,
                        label="minutes since the sweep's first file")
    figure.suptitle("Wire-grid null sweeps: signed band amplitude against magnet reading")
    return figure


def plot_null_traces(sweeps):
    """One panel per sweep: the averaged, baseline-removed traces coloured by reading."""
    import matplotlib.pyplot as plt

    if not sweeps:
        raise ValueError("no null sweeps to plot")
    column_count = len(sweeps)
    figure, axes = plt.subplots(1, column_count, figsize=(4.2 * column_count, 4.0),
                                squeeze=False, constrained_layout=True)
    colour_map = plt.get_cmap("coolwarm")
    for axis, sweep in zip(axes[0], sweeps):
        readings = sweep.readings_deg
        offsets = readings - sweep.fit.null_deg
        largest = max(float(np.max(np.abs(offsets))), 1e-9)
        peak = float(np.max(np.abs(sweep.traces))) or 1.0
        for reading, offset, trace in zip(readings, offsets, sweep.traces):
            axis.plot(sweep.time_ps, trace * 1e3,
                      color=colour_map(0.5 + 0.5 * offset / largest), linewidth=1.0,
                      label=f"{reading:g}")
        if not np.all(sweep.window == 1.0):
            axis.plot(sweep.time_ps, sweep.window * peak * 1e3, color="0.5", linestyle=":",
                      linewidth=1.0, label="window (scaled)")
        axis.set_title(sweep.name, fontsize=10)
        axis.set_xlabel("time [ps]")
        axis.set_ylabel("signal, baseline removed [mV]")
        axis.legend(fontsize=6, title="reading [deg]", title_fontsize=6, ncol=2, loc="best")
    figure.suptitle("Null sweeps in time: blue below the reported null, red above")
    return figure


def _draw_settle(axis, minutes, values, fitted, settle, scale, unit):
    """Points (hollow if left out of the fit), the fitted curve, and the plateau band."""
    axis.plot(minutes[fitted], values[fitted] * scale, "o", color="k", markersize=3.5,
              label="scans", zorder=3)
    if not np.all(fitted):
        axis.plot(minutes[~fitted], values[~fitted] * scale, "o", markerfacecolor="none",
                  markeredgecolor="k", markersize=4.5, label="not fitted (file start)",
                  zorder=3)
    if settle is None:
        return
    dense = np.linspace(0.0, max(minutes[-1], 1e-9) * 1.25, 400)
    if settle.tau_at_bound:
        axis.plot(dense, settle.model(dense) * scale, color="tab:red", linestyle="--",
                  label="exp fit: no bend yet, plateau unknown")
        return
    if settle.plateau_extrapolated:
        axis.plot(dense, settle.model(dense) * scale, color="tab:orange", linestyle="--",
                  label=f"exp fit, tau {settle.tau_minutes:.0f} min > data: plateau "
                        f"~{settle.plateau * scale:+.1f} {unit}, not reached")
        return
    axis.plot(dense, settle.model(dense) * scale, color="tab:blue",
              label=f"exp fit, tau {settle.tau_minutes:.1f} +/- "
                    f"{settle.tau_standard_error_minutes:.1f} min")
    plateau, error = settle.plateau * scale, settle.plateau_standard_error * scale
    axis.axhline(plateau, color="tab:green", linestyle="--", linewidth=1.0,
                 label=f"plateau {plateau:+.2f} +/- {error:.2f} {unit}")
    axis.axhspan(plateau - error, plateau + error, color="tab:green", alpha=0.15, linewidth=0)


def plot_settling(groups):
    """One column per magnet state: delay on top, amplitude below, both with the fitted settle."""
    import matplotlib.pyplot as plt

    if not groups:
        raise ValueError("no settling groups to plot")
    column_count = len(groups)
    figure, axes = plt.subplots(2, column_count, figsize=(4.6 * column_count, 6.5),
                                squeeze=False, sharex="col", constrained_layout=True)
    for column, group in enumerate(groups):
        trend = group.trend
        delay_axis, amplitude_axis = axes[0, column], axes[1, column]
        _draw_settle(delay_axis, trend.elapsed_minutes, trend.delays_fs, group.fitted,
                     group.delay_settle,
                     1.0, "fs")
        _draw_settle(amplitude_axis, trend.elapsed_minutes, trend.amplitudes - 1.0, group.fitted,
                     None if group.amplitude_settle is None else
                     _shifted(group.amplitude_settle, -1.0), 100.0, "%")
        for axis in (delay_axis, amplitude_axis):
            for start in group.file_start_minutes[1:]:
                axis.axvline(start, color="0.6", linestyle=":", linewidth=1.0)
            axis.legend(fontsize=7, loc="best")
        delay_axis.set_title(f"{group.label} ({len(group.filenames)} file(s))", fontsize=10)
        delay_axis.set_ylabel("delay vs first scan [fs]")
        amplitude_axis.set_ylabel("amplitude vs first scan [%]")
        amplitude_axis.set_xlabel("minutes since first scan"
                                  + ("" if group.timestamps_known else " (1 min/scan assumed)"))
    figure.suptitle("Purge settling per magnet state\n(dotted line = new file)", fontsize=11)
    return figure


def _shifted(settle, offset):
    """The same settle with its plateau moved by ``offset`` (amplitude ratio -> fractional change)."""
    from dataclasses import replace
    return replace(settle, plateau=settle.plateau + offset)
