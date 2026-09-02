"""DEMO (synthetic): how front-pulse self-referencing cancels absolute-time drift.

The SiO2-window reflection geometry (see run_me_low-level.py) records, in EVERY
acquisition, a FRONT reflection (window/air face, ``first_reflection``) and a BACK
reflection (window/sample face, ``second_reflection``) from the same THz pulse, one
window round trip apart. Two separate acquisitions -- a reference scan and a sample
scan -- are taken at different times, so the pulse's ABSOLUTE arrival time in the
recorded trace drifts between them (delay-stage backlash, re-triggering, thermal
drift). That drift is a NUISANCE: it has nothing to do with the sample, but it
still shows up as a big linear phase on the raw back-reflection spectrum.

The mechanism this script demonstrates (``thz_core.self_referenced_transfer`` /
``transfer_function``'s ``self_phase`` mode):

    within each trace   W = Y2 / Y1        (back / front)
    across the traces   H = W_sample / W_reference

Dividing the back spectrum by the front spectrum WITHIN one trace cancels that
trace's own absolute arrival time exactly, because both pulses share the same
recording clock (T0). What survives is only the RELATIVE timing between the two
pulses -- the physically meaningful window round trip -- which differs between the
reference and the sample only because of the sample's back-face response. That
residual difference is exactly the number the pipeline needs.

This script never touches real data: it builds two synthetic traces from Gaussian-
enveloped cosine "pulses" placed at hand-picked times, so every quantity in the
figure can be checked against a known ground truth. The four numbered panels tell
the story top to bottom:

    (A) the two time traces -- first & second reflection, at DIFFERENT absolute
        arrival times, with a DIFFERENT round-trip separation (the sample's answer)
    (B) raw unwrapped phase of all four windowed spectra -- confusing, dominated by
        each trace's own absolute arrival time
    (C) the within-trace ratio W = Y2/Y1, reference vs sample -- the absolute
        arrival time has cancelled; what is left is each trace's round-trip delay
    (D) the final self-referenced H = W_sample/W_reference (correct) against the
        naive H = Y2_sample/Y2_reference (still contaminated by the absolute-time
        drift) -- the punchline

Every stage is exposed on the returned ``results`` dict (time traces, all four
spectra, W_reference/W_sample, H_self_referenced/H_naive, recovered delays) so you
can re-plot or re-style without re-deriving anything.

Run:
    .venv/Scripts/python.exe explorations/reflection_phase_and_windowing/demo_self_referencing_mechanism.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIRECTORY = os.path.join(REPO_ROOT, "explorations", "output")
SECONDS_PER_PICOSECOND = 1e-12

# ---------------------------------------------------------------------------
# SCENARIO -- every physical/acquisition parameter of the synthetic geometry.
# Edit these to explore the mechanism; nothing below this block needs to change.
# ---------------------------------------------------------------------------

# The pulse itself: a Gaussian-enveloped cosine, a stand-in for a THz-TDS pulse.
CENTER_FREQUENCY_THZ = 1.0
ENVELOPE_SIGMA_PS = 0.15

# Acquisition / sampling.
SAMPLE_INTERVAL_PS = 0.02
TRACE_DURATION_PS = 60.0

# ARBITRARY / NUISANCE: the absolute arrival time of the FRONT pulse in each
# trace's own recorded time axis. In a real measurement this drifts scan-to-scan
# and has nothing to do with the sample.
FIRST_REFLECTION_ARRIVAL_REFERENCE_PS = 15.0
FIRST_REFLECTION_ARRIVAL_SAMPLE_PS = 21.4          # 6.4 ps of arbitrary scan-to-scan drift

# PHYSICAL: the round trip through the SiO2 window is common to both traces (same
# window, same thickness). Only the BACK-FACE response differs between the
# reference (e.g. window/air) and the sample (window/material under test) -- this
# extra delay is the one quantity the pipeline is actually trying to recover.
WINDOW_ROUND_TRIP_PS = 25.0
EXTRA_BACK_FACE_DELAY_REFERENCE_PS = 0.0
EXTRA_BACK_FACE_DELAY_SAMPLE_PS = 0.55              # <- the "true" sample answer

# Reflection amplitudes (front-spot coupling drift + sample being lossier at the
# back face). These do not affect any of the phase panels; included for realism.
FIRST_REFLECTION_AMPLITUDE_REFERENCE = 1.00
FIRST_REFLECTION_AMPLITUDE_SAMPLE = 0.85
SECOND_REFLECTION_AMPLITUDE_REFERENCE = 0.90
SECOND_REFLECTION_AMPLITUDE_SAMPLE = 0.55

# Windowing / FFT -- mirrors config['window']['half_width_ps'] in run_me_low-level.py.
WINDOW_HALF_WIDTH_PS = 5.0
# FFT_LENGTH sets the frequency spacing df = 1/(FFT_LENGTH * SAMPLE_INTERVAL_PS). Unwrapping
# a spectrum's phase is only safe (no aliasing) for delays below 1/(2*df); the raw individual
# reflections here carry delays up to ~t1+dt (tens of ps), so df must stay well under
# 1/(2*50ps) = 0.01 THz. 16384 gives df ~ 0.003 THz, safe with margin for delays up to ~160 ps.
FFT_LENGTH = 16384

ANALYSIS_BAND_THZ = (0.3, 2.0)     # band used to fit phase slopes -> recovered delay
DISPLAY_BAND_THZ = (0.05, 3.0)     # band shown on the phase panels

NOISE_AMPLITUDE = 0.0               # set > 0 (e.g. 0.01) to see slope recovery degrade
RANDOM_SEED = 0

# ---------------------------------------------------------------------------
# pulse / trace construction
# ---------------------------------------------------------------------------


def gaussian_cosine_pulse(time_ps, arrival_ps, amplitude, center_frequency_thz, envelope_sigma_ps):
    """A single quasi-few-cycle THz-like pulse: Gaussian envelope times a cosine carrier."""
    relative_time_ps = time_ps - arrival_ps
    envelope = np.exp(-(relative_time_ps ** 2) / (2.0 * envelope_sigma_ps ** 2))
    carrier = np.cos(2.0 * np.pi * center_frequency_thz * relative_time_ps)
    return amplitude * envelope * carrier


def build_synthetic_trace(
    time_ps, first_reflection_arrival_ps, second_reflection_arrival_ps,
    first_reflection_amplitude, second_reflection_amplitude,
    center_frequency_thz=CENTER_FREQUENCY_THZ, envelope_sigma_ps=ENVELOPE_SIGMA_PS,
    noise_amplitude=NOISE_AMPLITUDE, random_seed=RANDOM_SEED,
):
    """One full-trace reflection record: a first-reflection pulse plus a second-reflection pulse."""
    trace = gaussian_cosine_pulse(
        time_ps, first_reflection_arrival_ps, first_reflection_amplitude,
        center_frequency_thz, envelope_sigma_ps,
    )
    trace = trace + gaussian_cosine_pulse(
        time_ps, second_reflection_arrival_ps, second_reflection_amplitude,
        center_frequency_thz, envelope_sigma_ps,
    )
    if noise_amplitude > 0:
        rng = np.random.default_rng(random_seed)
        trace = trace + rng.normal(0.0, noise_amplitude, size=time_ps.shape)
    return trace


# ---------------------------------------------------------------------------
# windowing + spectra (mirrors thz.window_pulses_fixed_width + fft_spectrum)
# ---------------------------------------------------------------------------


def hann_gate_weights(time_ps, center_ps, half_width_ps):
    """Symmetric Hann window fully contained in ``[center - half_width, center + half_width]``.

    The SAME half-width gates every pulse; the window only LOCATES the pulse, it
    never shifts the data -- matching ``thz.window_pulses_fixed_width``.
    """
    weights = np.zeros_like(time_ps)
    inside_gate = np.abs(time_ps - center_ps) <= half_width_ps
    relative_time_ps = time_ps[inside_gate] - center_ps
    weights[inside_gate] = 0.5 * (1.0 + np.cos(np.pi * relative_time_ps / half_width_ps))
    return weights


def windowed_spectrum(time_ps, trace, center_ps, half_width_ps=WINDOW_HALF_WIDTH_PS,
                       sample_interval_ps=SAMPLE_INTERVAL_PS, n_fft=FFT_LENGTH):
    """Gate one reflection with a fixed-width Hann window and FFT it onto a shared grid."""
    gated_trace = trace * hann_gate_weights(time_ps, center_ps, half_width_ps)
    sample_interval_seconds = sample_interval_ps * SECONDS_PER_PICOSECOND
    spectrum = np.fft.rfft(gated_trace, n=n_fft)
    frequency_hz = np.fft.rfftfreq(n_fft, d=sample_interval_seconds)
    return frequency_hz, spectrum


def fit_phase_slope(frequency_hz, spectrum, band_thz=ANALYSIS_BAND_THZ):
    """Unwrap phase over ``band_thz`` and fit a line; return (slope_rad_per_hz, intercept_rad, delay_ps).

    Sign convention matches ``thz_core`` (numpy negative-exponent FFT / ``phase_ramp``):
    a pulse delayed by ``shift`` multiplies its spectrum by ``exp(-2*pi*i*f*shift)``,
    so ``phase(f) = -2*pi*f*shift`` and ``shift = -slope / (2*pi)``.
    """
    frequency_thz = frequency_hz * 1e-12
    in_band = (frequency_thz >= band_thz[0]) & (frequency_thz <= band_thz[1])
    unwrapped_phase = np.unwrap(np.angle(spectrum[in_band]))
    slope_rad_per_hz, intercept_rad = np.polyfit(frequency_hz[in_band], unwrapped_phase, 1)
    recovered_delay_ps = -slope_rad_per_hz / (2.0 * np.pi) * 1e12
    return slope_rad_per_hz, intercept_rad, recovered_delay_ps


# ---------------------------------------------------------------------------
# assemble the demo
# ---------------------------------------------------------------------------


def run_demo():
    """Build both traces, window/FFT all four reflections, and form every ratio.

    Returns a flat dict exposing every intermediate quantity (time traces, all
    four windowed spectra, both within-trace ratios, both candidate transfer
    functions, and the true/recovered delays) so the figure below -- or your own
    analysis -- can use them directly.
    """
    time_ps = np.arange(0.0, TRACE_DURATION_PS, SAMPLE_INTERVAL_PS)

    first_reflection_arrival_reference_ps = FIRST_REFLECTION_ARRIVAL_REFERENCE_PS
    first_reflection_arrival_sample_ps = FIRST_REFLECTION_ARRIVAL_SAMPLE_PS
    round_trip_delay_reference_ps = WINDOW_ROUND_TRIP_PS + EXTRA_BACK_FACE_DELAY_REFERENCE_PS
    round_trip_delay_sample_ps = WINDOW_ROUND_TRIP_PS + EXTRA_BACK_FACE_DELAY_SAMPLE_PS
    second_reflection_arrival_reference_ps = first_reflection_arrival_reference_ps + round_trip_delay_reference_ps
    second_reflection_arrival_sample_ps = first_reflection_arrival_sample_ps + round_trip_delay_sample_ps

    reference_trace = build_synthetic_trace(
        time_ps, first_reflection_arrival_reference_ps, second_reflection_arrival_reference_ps,
        FIRST_REFLECTION_AMPLITUDE_REFERENCE, SECOND_REFLECTION_AMPLITUDE_REFERENCE,
        random_seed=RANDOM_SEED,
    )
    sample_trace = build_synthetic_trace(
        time_ps, first_reflection_arrival_sample_ps, second_reflection_arrival_sample_ps,
        FIRST_REFLECTION_AMPLITUDE_SAMPLE, SECOND_REFLECTION_AMPLITUDE_SAMPLE,
        random_seed=None if RANDOM_SEED is None else RANDOM_SEED + 1,
    )

    frequency_hz, first_reflection_spectrum_reference = windowed_spectrum(
        time_ps, reference_trace, first_reflection_arrival_reference_ps)
    _, second_reflection_spectrum_reference = windowed_spectrum(
        time_ps, reference_trace, second_reflection_arrival_reference_ps)
    _, first_reflection_spectrum_sample = windowed_spectrum(
        time_ps, sample_trace, first_reflection_arrival_sample_ps)
    _, second_reflection_spectrum_sample = windowed_spectrum(
        time_ps, sample_trace, second_reflection_arrival_sample_ps)

    # Within-trace ratio: the front pulse is each trace's own internal clock.
    # (Both pulses share the same noiseless analytic envelope shape, which has an
    # exact spectral null near the Nyquist edge, far outside ANALYSIS_BAND_THZ /
    # DISPLAY_BAND_THZ -- errstate only silences that harmless out-of-band 0/0.)
    with np.errstate(divide="ignore", invalid="ignore"):
        within_trace_ratio_reference = second_reflection_spectrum_reference / first_reflection_spectrum_reference
        within_trace_ratio_sample = second_reflection_spectrum_sample / first_reflection_spectrum_sample

        # The self-referenced transfer function -- the correct, drift-cancelled answer.
        transfer_self_referenced = within_trace_ratio_sample / within_trace_ratio_reference

        # For contrast: the naive ratio of raw back reflections, with NO front-pulse
        # correction. Still carries the arbitrary absolute-arrival-time drift.
        transfer_naive = second_reflection_spectrum_sample / second_reflection_spectrum_reference

    true_delays_ps = dict(
        first_reflection_reference=first_reflection_arrival_reference_ps,
        second_reflection_reference=second_reflection_arrival_reference_ps,
        first_reflection_sample=first_reflection_arrival_sample_ps,
        second_reflection_sample=second_reflection_arrival_sample_ps,
        within_trace_ratio_reference=round_trip_delay_reference_ps,
        within_trace_ratio_sample=round_trip_delay_sample_ps,
        transfer_self_referenced=round_trip_delay_sample_ps - round_trip_delay_reference_ps,
        transfer_naive=second_reflection_arrival_sample_ps - second_reflection_arrival_reference_ps,
    )

    return dict(
        time_ps=time_ps,
        reference_trace=reference_trace,
        sample_trace=sample_trace,
        first_reflection_arrival_reference_ps=first_reflection_arrival_reference_ps,
        second_reflection_arrival_reference_ps=second_reflection_arrival_reference_ps,
        first_reflection_arrival_sample_ps=first_reflection_arrival_sample_ps,
        second_reflection_arrival_sample_ps=second_reflection_arrival_sample_ps,
        frequency_hz=frequency_hz,
        first_reflection_spectrum_reference=first_reflection_spectrum_reference,
        second_reflection_spectrum_reference=second_reflection_spectrum_reference,
        first_reflection_spectrum_sample=first_reflection_spectrum_sample,
        second_reflection_spectrum_sample=second_reflection_spectrum_sample,
        within_trace_ratio_reference=within_trace_ratio_reference,
        within_trace_ratio_sample=within_trace_ratio_sample,
        transfer_self_referenced=transfer_self_referenced,
        transfer_naive=transfer_naive,
        true_delays_ps=true_delays_ps,
    )


def print_summary(results):
    """Print true vs recovered (phase-slope-fitted) delay for every quantity in the demo."""
    spectra_by_label = {
        "first_reflection_reference": results["first_reflection_spectrum_reference"],
        "second_reflection_reference": results["second_reflection_spectrum_reference"],
        "first_reflection_sample": results["first_reflection_spectrum_sample"],
        "second_reflection_sample": results["second_reflection_spectrum_sample"],
        "within_trace_ratio_reference": results["within_trace_ratio_reference"],
        "within_trace_ratio_sample": results["within_trace_ratio_sample"],
        "transfer_self_referenced": results["transfer_self_referenced"],
        "transfer_naive": results["transfer_naive"],
    }
    frequency_hz = results["frequency_hz"]

    print("=== self-referencing phase-cancellation demo ===")
    print(f"{'quantity':<32s} {'true (ps)':>12s} {'recovered (ps)':>16s} {'error (ps)':>12s}")
    recovered_delays_ps = {}
    for label, spectrum in spectra_by_label.items():
        _, _, recovered_delay_ps = fit_phase_slope(frequency_hz, spectrum)
        true_delay_ps = results["true_delays_ps"][label]
        recovered_delays_ps[label] = recovered_delay_ps
        print(f"{label:<32s} {true_delay_ps:12.4f} {recovered_delay_ps:16.4f} "
              f"{recovered_delay_ps - true_delay_ps:+12.5f}")

    absolute_drift_ps = (
        results["first_reflection_arrival_sample_ps"] - results["first_reflection_arrival_reference_ps"]
    )
    print(
        f"\nabsolute arrival-time drift injected between the two scans: {absolute_drift_ps:+.3f} ps\n"
        f"self-referenced H recovers the sample's true extra delay "
        f"({recovered_delays_ps['transfer_self_referenced']:+.4f} ps, true "
        f"{results['true_delays_ps']['transfer_self_referenced']:+.4f} ps) regardless of that drift.\n"
        f"the naive ratio instead reports {recovered_delays_ps['transfer_naive']:+.4f} ps -- "
        f"the sample signal buried under the {absolute_drift_ps:+.3f} ps of arbitrary drift."
    )
    return recovered_delays_ps


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


COLOR_REFERENCE = "tab:blue"
COLOR_SAMPLE = "tab:red"
COLOR_FIRST_REFLECTION_SPAN = "tab:blue"
COLOR_SECOND_REFLECTION_SPAN = "tab:orange"
COLOR_SELF_REFERENCED = "tab:green"
COLOR_NAIVE = "0.4"


def _shade_reflection_windows(axis, first_center_ps, second_center_ps, half_width_ps=WINDOW_HALF_WIDTH_PS):
    axis.axvspan(first_center_ps - half_width_ps, first_center_ps + half_width_ps,
                 color=COLOR_FIRST_REFLECTION_SPAN, alpha=0.12, lw=0)
    axis.axvspan(second_center_ps - half_width_ps, second_center_ps + half_width_ps,
                 color=COLOR_SECOND_REFLECTION_SPAN, alpha=0.12, lw=0)
    for center_ps in (first_center_ps, second_center_ps):
        axis.axvline(center_ps, color="0.3", lw=0.8, ls=":")


def plot_self_referencing_demo(results, display_band_thz=DISPLAY_BAND_THZ):
    """Build the 5-panel figure. Returns (figure, axes_dict) for further styling."""
    frequency_hz = results["frequency_hz"]
    frequency_thz = frequency_hz * 1e-12
    in_display_band = (frequency_thz >= display_band_thz[0]) & (frequency_thz <= display_band_thz[1])
    f_plot = frequency_thz[in_display_band]

    def unwrapped_phase(spectrum):
        return np.unwrap(np.angle(spectrum[in_display_band]))

    figure = plt.figure(figsize=(14, 12), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.1])
    axis_time_reference = figure.add_subplot(grid[0, 0])
    axis_time_sample = figure.add_subplot(grid[0, 1], sharex=axis_time_reference, sharey=axis_time_reference)
    axis_raw_phase = figure.add_subplot(grid[1, :])
    axis_within_trace_phase = figure.add_subplot(grid[2, 0])
    axis_final_phase = figure.add_subplot(grid[2, 1])

    # (A)/(B) time-domain traces: different absolute arrival times, different
    # first/second separation (the sample's answer).
    axis_time_reference.plot(results["time_ps"], results["reference_trace"], color=COLOR_REFERENCE, lw=1.1)
    _shade_reflection_windows(
        axis_time_reference,
        results["first_reflection_arrival_reference_ps"], results["second_reflection_arrival_reference_ps"],
    )
    axis_time_reference.set_title("(A) reference trace")
    axis_time_reference.set_xlabel("time (ps)"); axis_time_reference.set_ylabel("amplitude")

    axis_time_sample.plot(results["time_ps"], results["sample_trace"], color=COLOR_SAMPLE, lw=1.1)
    _shade_reflection_windows(
        axis_time_sample,
        results["first_reflection_arrival_sample_ps"], results["second_reflection_arrival_sample_ps"],
    )
    axis_time_sample.set_title("(B) sample trace -- different arrival times, different round trip")
    axis_time_sample.set_xlabel("time (ps)")
    axis_time_sample.set_xlim(0, max(results["second_reflection_arrival_sample_ps"],
                                      results["second_reflection_arrival_reference_ps"]) + 2 * WINDOW_HALF_WIDTH_PS)

    # (C) raw phase of all four windowed spectra: confusing, dominated by each
    # trace's own absolute arrival time.
    axis_raw_phase.plot(f_plot, unwrapped_phase(results["first_reflection_spectrum_reference"]),
                         color=COLOR_REFERENCE, ls="-", lw=1.4, label="first reflection, reference")
    axis_raw_phase.plot(f_plot, unwrapped_phase(results["second_reflection_spectrum_reference"]),
                         color=COLOR_REFERENCE, ls="--", lw=1.4, label="second reflection, reference")
    axis_raw_phase.plot(f_plot, unwrapped_phase(results["first_reflection_spectrum_sample"]),
                         color=COLOR_SAMPLE, ls="-", lw=1.4, label="first reflection, sample")
    axis_raw_phase.plot(f_plot, unwrapped_phase(results["second_reflection_spectrum_sample"]),
                         color=COLOR_SAMPLE, ls="--", lw=1.4, label="second reflection, sample")
    axis_raw_phase.set_title("(C) raw unwrapped phase -- each trace's absolute arrival time dominates the slope")
    axis_raw_phase.set_xlabel("frequency (THz)"); axis_raw_phase.set_ylabel("unwrapped phase (rad)")
    axis_raw_phase.legend(fontsize=8, ncol=2)

    # (D) within-trace ratio W = Y2/Y1: the absolute arrival time has cancelled.
    axis_within_trace_phase.plot(f_plot, unwrapped_phase(results["within_trace_ratio_reference"]),
                                  color=COLOR_REFERENCE, lw=1.6, label="W_reference = Y2/Y1 (reference)")
    axis_within_trace_phase.plot(f_plot, unwrapped_phase(results["within_trace_ratio_sample"]),
                                  color=COLOR_SAMPLE, lw=1.6, label="W_sample = Y2/Y1 (sample)")
    axis_within_trace_phase.set_title("(D) within-trace ratio -- absolute arrival time cancelled,\nonly the round-trip delay survives")
    axis_within_trace_phase.set_xlabel("frequency (THz)"); axis_within_trace_phase.set_ylabel("unwrapped phase (rad)")
    axis_within_trace_phase.legend(fontsize=8)

    # (E) final transfer function: self-referenced (correct) vs naive (wrong).
    axis_final_phase.plot(f_plot, unwrapped_phase(results["transfer_self_referenced"]),
                           color=COLOR_SELF_REFERENCED, lw=2.0,
                           label="H = W_sample/W_reference (self-referenced, correct)")
    axis_final_phase.plot(f_plot, unwrapped_phase(results["transfer_naive"]),
                           color=COLOR_NAIVE, lw=1.4, ls=":",
                           label="H = Y2_sample/Y2_reference (naive, WRONG)")
    axis_final_phase.set_title("(E) final transfer function: self-referenced vs naive")
    axis_final_phase.set_xlabel("frequency (THz)"); axis_final_phase.set_ylabel("unwrapped phase (rad)")
    axis_final_phase.legend(fontsize=8, loc="lower right")

    true_sample_delay_ps = results["true_delays_ps"]["transfer_self_referenced"]
    _, _, recovered_self_referenced_ps = fit_phase_slope(frequency_hz, results["transfer_self_referenced"])
    _, _, recovered_naive_ps = fit_phase_slope(frequency_hz, results["transfer_naive"])
    axis_final_phase.text(
        0.02, 0.02,
        f"self-ref recovers {recovered_self_referenced_ps:+.3f} ps (true {true_sample_delay_ps:+.3f} ps)\n"
        f"naive gives {recovered_naive_ps:+.3f} ps instead",
        transform=axis_final_phase.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )

    figure.suptitle(
        "Self-referencing cancels absolute pulse-arrival drift, "
        "leaving only the sample's true round-trip phase",
        fontsize=13,
    )

    axes = dict(
        time_reference=axis_time_reference, time_sample=axis_time_sample,
        raw_phase=axis_raw_phase, within_trace_phase=axis_within_trace_phase,
        final_phase=axis_final_phase,
    )
    return figure, axes


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    demo_results = run_demo()
    print_summary(demo_results)
    demo_figure, demo_axes = plot_self_referencing_demo(demo_results)
    output_path = os.path.join(OUTPUT_DIRECTORY, "demo_self_referencing_mechanism.png")
    demo_figure.savefig(output_path, dpi=130)
    print(f"\nSaved figure: {output_path}")
    plt.show()
