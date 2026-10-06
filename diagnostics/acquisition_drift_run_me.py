"""Per-acquisition drift analysis — how the measurement changes DURING a run.

Purpose
-------
Written for the CNT paper doped/undoped relative-reflection measurement, where a
single 2 cm strip carries 1 cm of bare CNT and 1 cm of doped CNT and a lateral
translation stage is the only thing that changes between the two acquisitions.
Before any of that can be trusted we have to know whether the SAMPLE itself is
stable over a two-hour run, or whether it degrades while we measure it.

So this script deliberately does NOT invert anything.  It answers one question:
**what changed, at which frequency, and when?**  Each acquisition inside the
``.acc`` file is kept separate, FFT'd on a common grid, and compared against the
start of its own run.

Reading the output
------------------
``amplitude_ratio``      |Y| relative to the start of the run.  Flat at 1.0 means
                         stable.  A rising line is gain (warm-up, purge drying);
                         a falling line at high frequency with a flat line at low
                         frequency is the signature of a sample getting lossier.
``cumulative_deviation`` how far the running average was from the final answer
                         after N acquisitions — read the convergence off this.
``phase_deviation``      sub-sample timing drift.  A ramp in FREQUENCY is a delay
                         (tau = -slope/2pi); a uniform offset is not.
``timing drift``         the same thing fitted to a single number per acquisition
                         (``fitted_delay_seconds``).  Worth watching closely: the
                         time-domain peak is quantised to the 50 fs sample step,
                         so it reports a flat line — or a spurious 50 fs jump —
                         while the pulse actually walks smoothly across the bin.

Comparing the sample against the gold reference is the common-mode separator:
drift both files share is the instrument, drift only the sample shows is the
sample.  Note the two runs here were taken hours apart, so that comparison is
indicative, not controlled.

What this script is NOT
-----------------------
There is no transfer function here.  The gold file in this dataset is a
placeholder taken at a different time with different lock-in settings, and the
bare-CNT reference (the actual intended reference) has not been measured yet.
When it is, the relative measurement belongs in its own script; this one stays a
pure per-run stability diagnostic.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from dataset_core import DataSet
from dataset_core.adapters import acquisition_tracking as tracking


# ── Data path ────────────────────────────────────────────────────────────────
data_dir = r'C:\Users\Samuel\Data\THz\CNTs\2026-08-20_CNT-paper-doped_2'


# ── Configuration ────────────────────────────────────────────────────────────
config: dict = {
    "acquisition_tracking": {
        # Per-acquisition baseline: mean of the first N ps of each trace is removed
        # from that trace. The lock-in DC offset wanders over hours; left in, it is
        # a slow ramp in every trace and reads as pure low-frequency "drift".
        "baseline_ps": 1.0,

        # Window half-width in ps. None = auto: the widest symmetric window that
        # fits the record without clipping. These reflection records are only ~7 ps
        # long with the pulse ~2 ps from the start, so auto is usually right.
        "half_width_ps": None,

        # (start_ps, end_ps) to constrain the peak search; None = whole trace.
        "region_ps": None,

        # Acquisitions averaged to form the "run start" reference that
        # amplitude_ratio and phase_deviation are measured against. More than one,
        # so a single noisy acquisition does not tilt the whole series.
        "baseline_scans": 8,
    },
    "window": {
        "type": "hann",       # 'hann' | 'tukey' | 'boxcar'
        "alpha": 1.0,         # tukey only
    },
    "fft": {
        # Shared FFT length -> one frequency grid for every acquisition and file.
        # Zero-padding above the record length interpolates; it does not add
        # resolution. The true resolution is printed per file.
        "n_fft": 2048,
    },
    "display": {
        # Frequencies traced as lines against elapsed time.
        "frequencies_thz": (0.3, 0.5, 0.8, 1.2, 1.6),
        # Metrics drawn for each file (any registered name; see the printout).
        "metrics": ("amplitude_ratio", "cumulative_deviation"),
        # Frequency at which files are compared against each other.
        "comparison_frequency_thz": 0.5,
        "plot_config": {
            "snr_thresh_db": 10.0,     # trusted-band threshold for plots
            "x_axis": "elapsed_minutes",  # or 'acquisition_number'
            "frequency_range_thz": None,  # (lo, hi); None = the trusted band
        },
    },
    "purge": {
        # The nitrogen purge box equilibrates as 1 - exp(-t/tau). Fitting that
        # saturation separates a PURGE transient from sample degradation: a purge
        # settles to a plateau, degradation keeps going.
        "check_equilibration": True,
        "frequencies_thz": (0.5, 1.0, 1.6, 2.0),
        # Fraction of the way to the plateau you consider "settled". 1 tau is only
        # 63%; 0.99 costs 4.6 tau. This sets the reported wait time.
        "settle_fraction": 0.5,

        # ---- the usability criterion (see purge_settling_assessment) ----
        # NOT "distance from the plateau" — that ignores how long you measure for.
        # The test is the residual drift the purge will still inject into an average
        # of length measurement_window_minutes, which is the systematic error it
        # actually contributes. Returns a CUT INDEX: use acquisitions from there on.
        "assess_settling": True,
        "assess_frequency_thz": 2.0,      # highest frequency = most water-sensitive
        "measurement_window_minutes": 30.0,
        # 0.5% = the bottom of the measured instrument floor (F28: 0.5-2%), so below
        # this the purge has stopped being the limiting error term.
        "acceptable_drift_fraction": 0.005,

        # ---- the figure ----
        "plot_equilibration": True,
        "figure_frequencies_thz": (1.0, 2.0),
        # Filename substring of a run taken with the chamber left CLOSED. Drawn as a
        # second column — the control that makes the purge-not-degradation case.
        "control_file": "7mm",
        # A control run is usually far shorter than one tau and cannot fit its own.
        # True reuses the main run's tau so a slow tail is not mistaken for "flat".
        "apply_main_tau_to_control": True,
    },
    "export": {
        "save_tables": False,   # True -> write a CSV of the metric per frequency
        "export_dir": None,     # None -> alongside the data
    },
}


# ── Load ─────────────────────────────────────────────────────────────────────
# prefer_acc=True (the default) loads the .acc files, which hold the individual
# acquisitions and their timestamps. The .dat siblings are skipped for data but
# still read for their instrument-settings header.
dataset = DataSet(data_dir, config=config)
dataset.load_all_data()

# Nothing above this line has processed anything — acquisition_tracking reads the
# pristine per-scan raw_data, so it is independent of the averaged pipeline.
series_by_filename = tracking.extract_dataset_acquisitions(dataset)

print()
tracking.available_drift_metrics()


# ── Summary tables ───────────────────────────────────────────────────────────
display_config = config["display"]
tracking_config = config["acquisition_tracking"]

for filename, series in series_by_filename.items():
    tracking.summarise_drift(
        series,
        frequencies_thz=display_config["frequencies_thz"],
        metric="amplitude_ratio",
        baseline_scans=tracking_config["baseline_scans"],
    )
    # Is this run sitting inside a nitrogen-purge transient? A purge SATURATES
    # (exponential), sample degradation does not — that is what separates them.
    if config["purge"]["check_equilibration"]:
        tracking.purge_equilibration_report(
            series,
            frequencies_thz=config["purge"]["frequencies_thz"],
            settle_fraction=config["purge"]["settle_fraction"],
            baseline_scans=tracking_config["baseline_scans"],
        )


# ── purge: usability verdict and the collaborator-facing figure ──────────────
purge_config = config["purge"]
settling_assessments: dict = {}

if purge_config["assess_settling"]:
    for filename, series in series_by_filename.items():
        settling_assessments[filename] = tracking.print_settling_assessment(
            tracking.purge_settling_assessment(
                series,
                frequency_thz=purge_config["assess_frequency_thz"],
                measurement_window_minutes=purge_config["measurement_window_minutes"],
                acceptable_drift_fraction=purge_config["acceptable_drift_fraction"],
                baseline_scans=tracking_config["baseline_scans"],
            )
        )

if purge_config["plot_equilibration"] and series_by_filename:
    # The run with the largest transient is the one that establishes tau; anything
    # matching control_file is the undisturbed-chamber control.
    control_name = next(
        (name for name in series_by_filename
         if purge_config["control_file"] and purge_config["control_file"] in name),
        None,
    )
    transient_candidates = [name for name in series_by_filename if name != control_name]
    if transient_candidates:
        transient_name = max(
            transient_candidates,
            key=lambda name: abs(
                settling_assessments.get(name, {}).get("fit", {}).get("span", 0.0) or 0.0
            ),
        )
        main_assessment = settling_assessments.get(transient_name)
        main_tau = (main_assessment or {}).get("fit", {}).get("tau_seconds")
        tracking.plot_purge_equilibration(
            series_by_filename[transient_name],
            frequencies_thz=purge_config["figure_frequencies_thz"],
            control_series=series_by_filename.get(control_name),
            control_fixed_tau_seconds=(
                main_tau if purge_config["apply_main_tau_to_control"] else None
            ),
            assessment=main_assessment,
            baseline_scans=tracking_config["baseline_scans"],
        )


# ── Figures ──────────────────────────────────────────────────────────────────
for filename, series in series_by_filename.items():
    tracking.show_drift_figures(
        series,
        frequencies_thz=display_config["frequencies_thz"],
        metrics=display_config["metrics"],
        plot_config=display_config["plot_config"],
        baseline_scans=tracking_config["baseline_scans"],
    )

# Common-mode separator: every file at one frequency, on its own elapsed-time axis.
if len(series_by_filename) > 1:
    tracking.plot_drift_comparison(
        series_by_filename,
        frequency_thz=display_config["comparison_frequency_thz"],
        metric="amplitude_ratio",
        plot_config=display_config["plot_config"],
        baseline_scans=tracking_config["baseline_scans"],
    )


# ── Optional CSV export ──────────────────────────────────────────────────────
if config["export"]["save_tables"]:
    export_dir = config["export"]["export_dir"] or data_dir
    os.makedirs(export_dir, exist_ok=True)
    for filename, series in series_by_filename.items():
        tracking.save_metric_table(
            series,
            os.path.join(export_dir, f"{os.path.splitext(filename)[0]}_drift.csv"),
            frequencies_thz=display_config["frequencies_thz"],
            metric="amplitude_ratio",
            baseline_scans=tracking_config["baseline_scans"],
        )

plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# API handles — everything below is for poking at interactively (run with -i, or
# paste into a console after the script). Nothing here executes on a plain run.
#
#   series = series_by_filename['sample_7_CNT-doped-1.acc']
#
#   series.describe()                     # what it is and how it was processed
#   series.n_scans, series.elapsed_minutes, series.scan_numbers
#   series.amplitude_at(0.5)              # |Y| vs acquisition at 0.5 THz
#   series.spectrum_at(0.5)               # complex Y vs acquisition
#   series.scan_spectra                   # (n_scans, n_freq) complex — everything
#   series.scan_traces                    # (n_scans, n_time) time domain
#   series.mean_spectrum                  # what the averaged pipeline would see
#   series.trusted_mask(snr_thresh_db=10) # which bins are above the noise floor
#
#   values, entry = tracking.compute_drift_metric(series, 'phase_deviation')
#   tracking.drift_rate_per_hour(series, [0.5, 1.0])
#   tracking.cumulative_average_spectra(series)   # running mean, row N = "stop here"
#   tracking.missing_acquisition_numbers(series)  # culled/failed acquisitions
#
#   tracking.fitted_delay_seconds(series) * 1e15  # sub-sample delay per acq, in fs
#   tracking.delay_drift_rate(series)             # rate, total, and its noise floor
#
#   # purge: is it settled, and from which acquisition?
#   a = tracking.purge_settling_assessment(series, frequency_thz=2.0,
#                                          measurement_window_minutes=30)
#   a['first_acceptable_index']      # <- the cut; re-average from here
#   tracking.residual_drift_over_window(a['fit'], start_seconds=0, window_seconds=1800)
#   tracking.time_until_acceptable_drift(a['fit'], 1800, 0.005) / 60   # minutes to wait
#   # for a SHORT run, reuse tau from a long one rather than letting it fit its own:
#   tracking.purge_settling_assessment(short_series, known_tau_seconds=47*60)
#
#   fig, axes, fits = tracking.plot_purge_equilibration(
#       series, control_series=other, control_fixed_tau_seconds=47*60, assessment=a)
#   fig.savefig('purge_evidence.png', dpi=200, bbox_inches='tight')
#
#   tracking.plot_drift_map(series, 'phase_deviation')
#   tracking.plot_frequency_traces(series, 'cumulative_ratio', [0.5, 1.0])
#   tracking.plot_timing_drift(series)
#   plt.show()
#
# Adding a metric of your own — one place, and it becomes plottable everywhere:
#
#   @tracking.register_drift_metric(
#       name='amplitude_db', label='change in |Y| (dB)',
#       description='Amplitude change from run start, in dB.', reference_value=0.0)
#   def amplitude_change_db(series, *, baseline_scans=8):
#       import numpy as np
#       baseline = np.abs(tracking.baseline_spectrum(series, baseline_scans))
#       return 20 * np.log10(series.amplitude / baseline)
# ─────────────────────────────────────────────────────────────────────────────
