"""What does the window do? Rerun the whole ellipsometry chain per window setting and overlay.

A tutorial/demonstration tool, outside the main run (each setting is a complete refit):

    .venv/bin/python thz_ellipsometry_window_sweep.py <data dir>                 # half-widths
    .venv/bin/python thz_ellipsometry_window_sweep.py <data dir> --sweep shape
    .venv/bin/python thz_ellipsometry_window_sweep.py --simulate hr_silicon      # no data needed
    ... --half-widths 1.5 2 3 4   --rows acquisition (faster)   --save sweep.png (no window)

Every setting uses the same config as ``thz_ellipsometry_run_me.py`` except the window; the
figure shows, for each setting: the window on the sample's mean trace, the windowed trace, the
spectrum, rho, and n and k. A printed table gives n and k at a few frequencies with the
resolution each window implies. Reading guide: docs/ELLIPSOMETRY_HARMONIC_FIT_TUTORIAL.md sec. 5.

What to expect: a narrower window coarsens the resolution (1/T) and smooths every spectrum; a
window too narrow to hold the p and s pulses distorts rho at low frequency; a window wide enough
to take in an echo adds a ripple at 1/(echo delay). The window SHAPE matters for the drift fit
(flat top vs Hann, see figure 2's notes) more than for rho itself.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
import tempfile

import numpy as np

import thz_ellipsometry_run_me as run_me
from thz_ellipsometry.adapters.inspection import record_from_outcome
from thz_ellipsometry.adapters.stages import run_ellipsometry

#: sweep name -> list of (label, preprocess overrides). The one place a sweep is defined.
def _half_width_settings(half_widths_ps):
    return [(f"tukey, half-width {width:g} ps", {"window_half_width_ps": width})
            for width in half_widths_ps]


def _shape_settings(_):
    return [("tukey 0.25 (wide flat top)", {"window_shape": "tukey", "taper_fraction": 0.25}),
            ("tukey 0.5 (default)", {"window_shape": "tukey", "taper_fraction": 0.5}),
            ("tukey 0.8", {"window_shape": "tukey", "taper_fraction": 0.8}),
            ("hann", {"window_shape": "hann"})]


SWEEPS = {"half_width": _half_width_settings, "shape": _shape_settings}

REPORT_FREQUENCIES_THZ = (1.0, 1.5, 2.0, 2.5)


def run_setting(base_config, overrides):
    """The finished RunOutcome for one window setting (quietly)."""
    config = copy.deepcopy(base_config)
    config["preprocess"].update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_ellipsometry(config)


def plot_sweep(results):
    """Six panels: window on the trace, windowed trace, |S|, rho, n, k -- one colour per setting."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    colours = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(results)))
    for (label, outcome), colour in zip(results, colours):
        record = record_from_outcome(outcome)
        sample = record.series_for("sample")[0]
        state = sample.states_deg[0]
        rows = sample.rows_in_state(state)
        trace = sample.baseline_corrected_traces[rows].mean(axis=0) * 1e3
        peak = float(np.max(np.abs(trace)))
        axes[0, 0].plot(sample.time_ps, sample.window * peak, color=colour, label=label)
        axes[0, 1].plot(sample.time_ps, trace * sample.window, color=colour)
        frequencies_thz = sample.frequencies_hz / 1e12
        shown = frequencies_thz <= 5.0
        spectrum = sample.row_spectra[rows].mean(axis=0)
        axes[0, 2].plot(frequencies_thz[shown], 20 * np.log10(np.abs(spectrum[shown]) + 1e-30),
                        color=colour)
        band_thz = record.band_frequencies_hz / 1e12
        axes[1, 0].plot(band_thz, np.rad2deg(np.angle(record.ratio)), color=colour)
        axes[1, 1].plot(band_thz, record.index.real, color=colour)
        axes[1, 2].plot(band_thz, -record.index.imag, color=colour)
    axes[0, 0].plot(sample.time_ps, trace, color="k", linewidth=1.0, label="sample mean trace")
    if record.reference_index is not None:
        axes[1, 1].plot(band_thz, record.reference_index.real, "k--", label="reference")
        axes[1, 2].plot(band_thz, -record.reference_index.imag, "k--", label="reference")
        axes[1, 1].legend(fontsize=8)
    titles = (("window on the sample's mean trace", "time [ps]", "signal [mV]"),
              ("windowed trace (what is transformed)", "time [ps]", "signal x window [mV]"),
              (f"spectrum of the {state:.0f} deg state", "frequency [THz]", "|S| [dB]"),
              ("Delta = arg rho", "frequency [THz]", "[deg]"),
              ("n", "frequency [THz]", "n"), ("k", "frequency [THz]", "k"))
    for axis, (title, x_label, y_label) in zip(axes.flat, titles):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Window sweep: the same data, the same fit, only the window changed")
    return figure


def format_table(results):
    lines = ["setting                         resolution  "
             + "  ".join(f"n@{f:g}   k@{f:g}  " for f in REPORT_FREQUENCIES_THZ)
             + "  chi2 (ref, sample)"]
    for label, outcome in results:
        inversion = outcome.result.inversion
        frequencies_thz = inversion.frequencies_hz / 1e12
        picks = [int(np.argmin(np.abs(frequencies_thz - f))) for f in REPORT_FREQUENCIES_THZ]
        resolution = outcome.resolution
        chi = [fit.reduced_chi_square for fit in outcome.fits.values()]
        lines.append(
            f"{label:<32}{'' if resolution is None else f'{resolution.df_resolution_hz/1e12:.3f} THz':>10}  "
            + "  ".join(f"{inversion.refractive_index[i]:.3f} {inversion.extinction[i]:6.3f}"
                        for i in picks)
            + "   " + ", ".join("-" if value is None else f"{value:.2f}" for value in chi))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", help="data directory (default: run_me config)")
    parser.add_argument("--simulate", choices=sorted(run_me.SIMULATED_SAMPLES),
                        help="sweep a synthetic dataset instead of real data")
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="half_width")
    parser.add_argument("--half-widths", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0, 4.0],
                        metavar="PS")
    parser.add_argument("--rows", choices=("scan", "acquisition"),
                        help="override config['acquisition']['rows'] (acquisition is faster)")
    parser.add_argument("--save", metavar="PNG", help="write the figure instead of showing it")
    arguments = parser.parse_args(argv)

    config = copy.deepcopy(run_me.config)
    config["general"]["show_graph"] = False
    if arguments.simulate:
        directory = tempfile.mkdtemp(prefix="window_sweep_")
        run_me.build_simulated_dataset(directory, arguments.simulate,
                                       incidence_angle_deg=config["geometry"]["incidence_angle_deg"],
                                       drift_within_acquisition=True)
        config["validation"]["expect"] = arguments.simulate if arguments.simulate in (
            "hr_silicon",) else None
    else:
        directory = arguments.directory or config["data"]["directory"]
    config["data"]["directory"] = directory
    if arguments.rows:
        config["acquisition"]["rows"] = arguments.rows

    results = []
    for label, overrides in SWEEPS[arguments.sweep](arguments.half_widths):
        print(f"[window_sweep] {label} ...", flush=True)
        try:
            results.append((label, run_setting(config, overrides)))
        except ValueError as error:
            print(f"[window_sweep]   skipped: {error}", file=sys.stderr)
    if not results:
        return 2
    print(format_table(results))

    import matplotlib
    if arguments.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plot_sweep(results)
    if arguments.save:
        figure.savefig(arguments.save, dpi=130)
        print(f"[window_sweep] wrote {arguments.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
