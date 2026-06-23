"""DEMO: reflection windowing styles — cancellation and mixing.

We window BOTH reflections of a sample and its reference with a chosen *style* per
reflection, build the self-referenced transfer function
    transfer = (sample_second / sample_first) / (reference_second / reference_first),
and ask two questions:

  1. CANCELLATION — if the SAME imperfect style is applied to both reflections of
     both the sample and the reference, does the imperfection cancel in `transfer`?
     (a too-narrow window that loses resolution; an asymmetric window that keeps the
     tail; a symmetric full-extent window that relies on prepended zeros to stand in
     for un-acquired pre-pulse.)

  2. MIXING — if the first reflection uses one style (say too narrow) and the second
     uses another (normal symmetric), does the mismatch create frequency/phase
     artefacts that do NOT cancel?

Everything is driven by the WINDOW_STYLES and STYLE_COMBINATIONS dictionaries so the
referencing and styles can be changed and re-run easily. The apodization itself uses
the real `thz_core.window_time` (the same function the pipeline will use), so the
demo matches production maths. Pulses stay on one shared time axis per trace, so the
inter-pulse phase relationship is structural (no common-grid step).

Run:
    .venv/Scripts/python.exe explorations/demo_windowing_styles.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset_core.dataset import DataSet  # noqa: E402
import thz_core.thz_core as core  # noqa: E402  (the production window function)

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
ROOT_DIRECTORY = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17"
OUTPUT_DIRECTORY = os.path.join(REPO_ROOT, "explorations", "output")

SAMPLE_FILENAME_KEY = "sample_a-40.0_s-0"
REFERENCE_FILENAME_KEY = "reference_a-40.0_s-0"

FIRST_REFLECTION_RANGE_PS = (152.0, 158.5)
SECOND_REFLECTION_RANGE_PS = (161.0, 168.0)

# Hard crop: a 2nd-order internal-reflection echo in the GaP detection crystal sits
# ~168 ps and adds interpretation-harming oscillations. Drop everything after it.
HARD_CROP_PS = 168.0

SAMPLE_INTERVAL_SECONDS = 50e-15
FFT_LENGTH = 2048
DISPLAY_BAND_THZ = (0.1, 4.5)
TRUSTED_BAND_THZ = (0.5, 4.5)
SECONDS_TO_PICOSECONDS = 1e12

USE_SELF_REFERENCE = True  # flip to False to ratio second reflections only

# A windowing STYLE describes how to window ONE reflection. extents are measured
# from the pulse PEAK; `allow_prepend` lets the window reach earlier than the
# acquisition start by prepending zeros (faking un-acquired baseline pre-pulse).
# Extents are clamped to what actually isolates each pulse: the second reflection
# has only ~5.5 ps before the inter-pulse midpoint and the GaP echo sits ~5.3 ps
# after it, so a symmetric window wider than ~5 ps would reach into the neighbouring
# pulse / echo. 5 ps keeps both reflections cleanly isolated.
WINDOW_STYLES = {
    "symmetric_prepended": dict(
        shape="hann", alpha=1.0, pre_extent_ps=5.0, post_extent_ps=5.0, allow_prepend=True,
    ),
    "symmetric_shortest": dict(
        shape="hann", alpha=1.0, pre_extent_ps=2.3, post_extent_ps=2.3, allow_prepend=False,
    ),
    "asymmetric_tukey": dict(
        shape="tukey", alpha=0.5, pre_extent_ps=2.3, post_extent_ps=5.0, allow_prepend=False,
    ),
}

# Each combination = (first_reflection_style, second_reflection_style). The first
# entry is the reference everything else is compared against.
REFERENCE_COMBINATION_LABEL = "ideal: prepended / prepended"
STYLE_COMBINATIONS = {
    "ideal: prepended / prepended":      ("symmetric_prepended", "symmetric_prepended"),
    "both narrow":                       ("symmetric_shortest", "symmetric_shortest"),
    "both asymmetric":                   ("asymmetric_tukey", "asymmetric_tukey"),
    "mixed: narrow 1st, prepended 2nd":  ("symmetric_shortest", "symmetric_prepended"),
    "mixed: prepended 1st, narrow 2nd":  ("symmetric_prepended", "symmetric_shortest"),
}


# ---------------------------------------------------------------------------
# loading and helpers
# ---------------------------------------------------------------------------
def load_trace(dataset, filename_key):
    """Return (filename, time_seconds, amplitude) for the first matching file."""
    for filename, data_object in dataset.data.items():
        if filename_key in filename.lower():
            raw_scans = np.asarray(data_object.raw_data, float)  # [time_ps, scan1, ...]
            time_picoseconds = raw_scans[:, 0]
            if raw_scans.shape[1] > 2:
                amplitude = raw_scans[:, 1:].mean(axis=1)
            else:
                amplitude = raw_scans[:, 1]
            amplitude = amplitude - amplitude[:20].mean()  # baseline off the pre-pulse
            kept = time_picoseconds <= HARD_CROP_PS  # hard crop the GaP 2nd echo
            return filename, time_picoseconds[kept] / SECONDS_TO_PICOSECONDS, amplitude[kept]
    raise SystemExit(f"no trace matching {filename_key!r}")


def find_peak_time_seconds(time_seconds, amplitude, range_ps):
    """Absolute time of the largest |amplitude| inside range_ps."""
    time_ps = time_seconds * SECONDS_TO_PICOSECONDS
    inside_range = (time_ps >= range_ps[0]) & (time_ps <= range_ps[1])
    indices_inside = np.where(inside_range)[0]
    peak_index = indices_inside[int(np.argmax(np.abs(amplitude[indices_inside])))]
    return float(time_seconds[peak_index])


def prepend_baseline_zeros(time_seconds, amplitude, prepend_seconds):
    """Extend the axis earlier by prepend_seconds of zeros (uniform dt linear axis)."""
    if prepend_seconds <= 0:
        return time_seconds, amplitude
    prepend_count = int(np.ceil(prepend_seconds / SAMPLE_INTERVAL_SECONDS))
    earlier_times = time_seconds[0] - SAMPLE_INTERVAL_SECONDS * np.arange(prepend_count, 0, -1)
    extended_time = np.concatenate([earlier_times, time_seconds])
    extended_amplitude = np.concatenate([np.zeros(prepend_count), amplitude])
    return extended_time, extended_amplitude


def apply_core_window(time_seconds, amplitude, gate_start_seconds, gate_end_seconds, shape, alpha):
    """Apodize one reflection with the production thz_core.window_time over a gate.

    Returns the full-length windowed amplitude (zeros outside the gate).
    """
    window_config = {
        "window": {
            "type": shape,
            "alpha": alpha,
            "gate_start": gate_start_seconds,
            "gate_end": gate_end_seconds,
            "zero_tail": True,
        }
    }
    windowed_amplitude, _metrics, _global_window = core.window_time(
        time_seconds, amplitude, window_config
    )
    return windowed_amplitude


def required_prepend_seconds(first_peak_seconds, acquisition_start_seconds, first_reflection_style):
    """How much earlier than the acquisition start the first-reflection gate reaches."""
    if not first_reflection_style["allow_prepend"]:
        return 0.0
    pre_extent_seconds = first_reflection_style["pre_extent_ps"] / SECONDS_TO_PICOSECONDS
    available_pre_pulse_seconds = first_peak_seconds - acquisition_start_seconds
    return max(0.0, pre_extent_seconds + 0.5e-12 - available_pre_pulse_seconds)


def reflection_spectrum_on_shared_axis(
    shared_time_seconds, shared_amplitude, peak_seconds, style,
    isolation_min_seconds, isolation_max_seconds,
):
    """Window one reflection on the shared axis and return its complex spectrum.

    The spectrum is referenced to absolute time 0 (multiply by exp(-i 2π f t0)) so
    that spectra computed on differently-prepended axes remain phase-comparable.
    Because both reflections of a trace share one axis, this is phase-neutral within
    a trace and only matters when comparing across combinations.

    The gate is clamped to [isolation_min, isolation_max] so it cannot overrun the
    inter-pulse midpoint (catching the neighbouring reflection) or the 168 ps GaP
    echo crop. A clamp that makes the gate asymmetric about the peak is itself a
    finding: it means the requested symmetric extent does not fit the clean window.
    """
    gate_start_seconds = max(peak_seconds - style["pre_extent_ps"] / SECONDS_TO_PICOSECONDS,
                             isolation_min_seconds)
    gate_end_seconds = min(peak_seconds + style["post_extent_ps"] / SECONDS_TO_PICOSECONDS,
                           isolation_max_seconds)
    windowed_amplitude = apply_core_window(
        shared_time_seconds, shared_amplitude, gate_start_seconds, gate_end_seconds,
        style["shape"], style["alpha"],
    )
    frequency_hz = np.fft.rfftfreq(FFT_LENGTH, SAMPLE_INTERVAL_SECONDS)
    spectrum = np.fft.rfft(windowed_amplitude, n=FFT_LENGTH)
    spectrum = spectrum * np.exp(-2j * np.pi * frequency_hz * shared_time_seconds[0])
    return frequency_hz, spectrum, windowed_amplitude, shared_time_seconds


def build_trace_spectra(trace, first_reflection_style, second_reflection_style):
    """Window both reflections of one trace (shared axis) → first/second spectra."""
    prepend_seconds = required_prepend_seconds(
        trace["first_peak_seconds"], trace["time_seconds"][0], first_reflection_style
    )
    shared_time_seconds, shared_amplitude = prepend_baseline_zeros(
        trace["time_seconds"], trace["amplitude"], prepend_seconds
    )
    # Isolation bounds: the inter-pulse midpoint separates the two reflections, the
    # acquisition start bounds the first, and the 168 ps echo crop bounds the second.
    inter_pulse_midpoint_seconds = 0.5 * (trace["first_peak_seconds"] + trace["second_peak_seconds"])
    crop_seconds = HARD_CROP_PS / SECONDS_TO_PICOSECONDS
    frequency_hz, first_spectrum, first_windowed, shared_axis = reflection_spectrum_on_shared_axis(
        shared_time_seconds, shared_amplitude, trace["first_peak_seconds"], first_reflection_style,
        shared_time_seconds[0], inter_pulse_midpoint_seconds,
    )
    _, second_spectrum, second_windowed, _ = reflection_spectrum_on_shared_axis(
        shared_time_seconds, shared_amplitude, trace["second_peak_seconds"], second_reflection_style,
        inter_pulse_midpoint_seconds, crop_seconds,
    )
    return dict(
        frequency_hz=frequency_hz,
        first_spectrum=first_spectrum,
        second_spectrum=second_spectrum,
        first_windowed=first_windowed,
        second_windowed=second_windowed,
        shared_time_seconds=shared_axis,
    )


def divide_with_noise_floor(numerator, denominator, floor_fraction=1e-3):
    """Complex divide, masking bins where the denominator falls into the AC noise floor."""
    floor = np.abs(denominator)[1:].max() * floor_fraction  # exclude DC bin from the floor
    result = np.full_like(numerator, np.nan + 1j * np.nan)
    usable = np.abs(denominator) >= floor
    result[usable] = numerator[usable] / denominator[usable]
    return result


def transfer_function(sample_spectra, reference_spectra, use_self_reference):
    """Self-referenced (or plain) transfer function from the windowed spectra."""
    if use_self_reference:
        sample_intra_ratio = divide_with_noise_floor(
            sample_spectra["second_spectrum"], sample_spectra["first_spectrum"]
        )
        reference_intra_ratio = divide_with_noise_floor(
            reference_spectra["second_spectrum"], reference_spectra["first_spectrum"]
        )
        return divide_with_noise_floor(sample_intra_ratio, reference_intra_ratio)
    return divide_with_noise_floor(
        sample_spectra["second_spectrum"], reference_spectra["second_spectrum"]
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    dataset = DataSet(ROOT_DIRECTORY)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)

    sample_filename, sample_time_seconds, sample_amplitude = load_trace(dataset, SAMPLE_FILENAME_KEY)
    reference_filename, reference_time_seconds, reference_amplitude = load_trace(
        dataset, REFERENCE_FILENAME_KEY
    )

    sample_trace = dict(
        name=sample_filename, time_seconds=sample_time_seconds, amplitude=sample_amplitude,
        first_peak_seconds=find_peak_time_seconds(sample_time_seconds, sample_amplitude, FIRST_REFLECTION_RANGE_PS),
        second_peak_seconds=find_peak_time_seconds(sample_time_seconds, sample_amplitude, SECOND_REFLECTION_RANGE_PS),
    )
    reference_trace = dict(
        name=reference_filename, time_seconds=reference_time_seconds, amplitude=reference_amplitude,
        first_peak_seconds=find_peak_time_seconds(reference_time_seconds, reference_amplitude, FIRST_REFLECTION_RANGE_PS),
        second_peak_seconds=find_peak_time_seconds(reference_time_seconds, reference_amplitude, SECOND_REFLECTION_RANGE_PS),
    )

    print("=== reflection windowing styles demo — CNT-17 s-0 ===")
    print(f"sample    = {sample_trace['name']}")
    print(f"reference = {reference_trace['name']}")
    print(f"sample first/second peak: {sample_trace['first_peak_seconds']*SECONDS_TO_PICOSECONDS:.2f}"
          f" / {sample_trace['second_peak_seconds']*SECONDS_TO_PICOSECONDS:.2f} ps")
    print(f"self_reference = {USE_SELF_REFERENCE}\n")

    # Compute the transfer function for every style combination.
    transfer_by_combination = {}
    spectra_by_combination = {}
    for combination_label, (first_style_name, second_style_name) in STYLE_COMBINATIONS.items():
        first_style = WINDOW_STYLES[first_style_name]
        second_style = WINDOW_STYLES[second_style_name]
        sample_spectra = build_trace_spectra(sample_trace, first_style, second_style)
        reference_spectra = build_trace_spectra(reference_trace, first_style, second_style)
        transfer_by_combination[combination_label] = transfer_function(
            sample_spectra, reference_spectra, USE_SELF_REFERENCE
        )
        spectra_by_combination[combination_label] = sample_spectra

    frequency_hz = spectra_by_combination[REFERENCE_COMBINATION_LABEL]["frequency_hz"]
    frequency_thz = frequency_hz * 1e-12
    display_band = (frequency_thz >= DISPLAY_BAND_THZ[0]) & (frequency_thz <= DISPLAY_BAND_THZ[1])
    trusted_band = (frequency_thz >= TRUSTED_BAND_THZ[0]) & (frequency_thz <= TRUSTED_BAND_THZ[1])
    reference_transfer = transfer_by_combination[REFERENCE_COMBINATION_LABEL]

    # Report: how far each combination's transfer deviates from the ideal.
    print(f"deviation of transfer(f) from '{REFERENCE_COMBINATION_LABEL}':")
    header = f"  {'combination':36s} {'|H| rms frac':>13s} {'phase rms full':>15s} {'phase rms trusted':>18s}"
    print(header)
    for combination_label, transfer in transfer_by_combination.items():
        magnitude_fraction = np.abs(transfer) / np.abs(reference_transfer) - 1.0
        phase_error = np.angle(transfer * np.conj(reference_transfer))
        magnitude_rms = np.sqrt(np.nanmean(magnitude_fraction[display_band] ** 2))
        phase_rms_full = np.sqrt(np.nanmean(phase_error[display_band] ** 2))
        phase_rms_trusted = np.sqrt(np.nanmean(phase_error[trusted_band] ** 2))
        print(f"  {combination_label:36s} {magnitude_rms:>13.4f} "
              f"{phase_rms_full:>15.4f} {phase_rms_trusted:>18.4f}")

    # ---------------------------------------------------------------------
    # figure
    # ---------------------------------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), layout="constrained")
    figure.suptitle(
        f"Reflection windowing styles — CNT-17 s-0 (self_reference={USE_SELF_REFERENCE})",
        fontsize=12,
    )

    # (A) time domain: how the ideal combination isolates the two reflections.
    ideal_sample_spectra = spectra_by_combination[REFERENCE_COMBINATION_LABEL]
    shared_time_ps = ideal_sample_spectra["shared_time_seconds"] * SECONDS_TO_PICOSECONDS
    time_axis = axes[0, 0]
    time_axis.plot(sample_time_seconds * SECONDS_TO_PICOSECONDS, sample_amplitude,
                   color="0.6", lw=1, label="raw sample")
    time_axis.plot(shared_time_ps, ideal_sample_spectra["first_windowed"], lw=1.3, label="first windowed")
    time_axis.plot(shared_time_ps, ideal_sample_spectra["second_windowed"], lw=1.3, label="second windowed")
    time_axis.set_xlim(148, 170)
    time_axis.set_title("(A) ideal styles isolate both reflections on one axis")
    time_axis.set_xlabel("time (ps)"); time_axis.set_ylabel("amplitude"); time_axis.legend(fontsize=8)

    # (B) |transfer| for each combination.
    magnitude_axis = axes[0, 1]
    for combination_label, transfer in transfer_by_combination.items():
        magnitude_axis.plot(frequency_thz[display_band], np.abs(transfer[display_band]),
                            lw=1.2, label=combination_label)
    magnitude_axis.set_title("(B) |transfer(f)|")
    magnitude_axis.set_xlabel("frequency (THz)"); magnitude_axis.set_ylabel("|H|")
    magnitude_axis.legend(fontsize=7)

    # (C) phase(transfer) error vs ideal — the headline.
    phase_axis = axes[1, 0]
    for combination_label, transfer in transfer_by_combination.items():
        if combination_label == REFERENCE_COMBINATION_LABEL:
            continue
        phase_error = np.unwrap(np.angle(transfer[display_band] * np.conj(reference_transfer[display_band])))
        phase_axis.plot(frequency_thz[display_band], phase_error, lw=1.2, label=combination_label)
    phase_axis.axhline(0, color="0.7", lw=0.8)
    phase_axis.axvspan(TRUSTED_BAND_THZ[0], TRUSTED_BAND_THZ[1], color="0.9", zorder=0, label="trusted band")
    phase_axis.set_title("(C) phase(transfer) error vs ideal")
    phase_axis.set_xlabel("frequency (THz)"); phase_axis.set_ylabel("Δ phase (rad)")
    phase_axis.legend(fontsize=7)

    # (D) |transfer| fractional error vs ideal.
    magnitude_error_axis = axes[1, 1]
    for combination_label, transfer in transfer_by_combination.items():
        if combination_label == REFERENCE_COMBINATION_LABEL:
            continue
        magnitude_fraction = np.abs(transfer[display_band]) / np.abs(reference_transfer[display_band]) - 1.0
        magnitude_error_axis.plot(frequency_thz[display_band], magnitude_fraction, lw=1.2, label=combination_label)
    magnitude_error_axis.axhline(0, color="0.7", lw=0.8)
    magnitude_error_axis.set_title("(D) |transfer| fractional error vs ideal")
    magnitude_error_axis.set_xlabel("frequency (THz)"); magnitude_error_axis.set_ylabel("Δ|H| / |H|")
    magnitude_error_axis.legend(fontsize=7)

    output_path = os.path.join(OUTPUT_DIRECTORY, "demo_windowing_styles.png")
    figure.savefig(output_path, dpi=130)
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
    # plt.show()  # comment out if running in headless mode (Agg backend)