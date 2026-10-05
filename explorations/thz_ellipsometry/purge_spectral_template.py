"""What SHAPE does a purge transient have in the spectrum, and is it independent of the sample?

Question (Samuel, 2026-10-05): the August purge runs were taken on a CNT sample, but the RELATIVE
change across the purge window should be modellable whatever the sample, because the purge acts
on the beam path, not on the reflector. If so, one template measured once can be fitted (or just
detected) in any later measurement -- including the ellipsometry harmonic fit, where a humidity
change between magnet states does not cancel.

F31/F32 already established WHEN (exponential settling, tau ~38 min timing / ~47 min water
amplitude) and WHAT (a smooth log-amplitude tilt rising with frequency plus a delay; the water
lines are unresolvable on a ~6.7 ps record). This script adds:

1. The decomposition of each scan's complex log-ratio to the purged end of its run into four
   terms -- flat gain, tilt (log-amplitude per THz), phase offset, delay -- and what is left.
2. Whether ONE spectral shape carries the change (SVD of the log-ratio matrix).
3. Whether that shape is the same on gold and on CNT (the sample-independence hypothesis).
4. What a linear ramp vs an exponential with KNOWN tau leaves behind in a block, for blocks that
   start 10 and 30 min after a disturbance -- Samuel's operating numbers.

Data: ~/data/data_sync/diagnostics/2026-08-20_humidity_and_purge_CNT (12mm = longest, from an
unpurged box; gold = 112 scans earlier the same day). Run from the repo root.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_core.adapters.acquisition_tracking import fit_exponential_equilibration  # noqa: E402
from thz_ellipsometry.adapters.loader import read_accumulation_file  # noqa: E402
from thz_ellipsometry.core.preprocess import transform_traces  # noqa: E402

DATA_DIRECTORY = os.path.expanduser(
    "~/data/data_sync/diagnostics/2026-08-20_humidity_and_purge_CNT")
RUNS = {
    "cnt_12mm (unpurged start)": "sample_CNT-doped-12mm.acc",
    "gold": "reference_7_gold-2.acc",
    "cnt_12mm-2 (follows)": "sample_CNT-doped-12mm-2.acc",
    "cnt_7mm (box closed)": "sample_CNT_doped_7mm.acc",
    "cnt_19mm": "sample_CNT_doped-19mm.acc",
}
BAND_THZ = (0.3, 2.2)
PURGED_END_SCANS = 10          # reference = mean of the last scans (most purged)
PAD_FACTOR = 4


def load_run(filename):
    """Scans, elapsed seconds and spectra of one run, on one common window."""
    accumulation = read_accumulation_file(os.path.join(DATA_DIRECTORY, filename))
    first = accumulation.scan_timestamps[0]
    elapsed = np.array([(stamp - first).total_seconds()
                        for stamp in accumulation.scan_timestamps])
    # The record is short (~6.7 ps); keep all of it with gentle edge tapers.
    record_ps = float(np.ptp(accumulation.time_ps))
    transformed = transform_traces(accumulation.time_ps, accumulation.scans,
                                   window_half_width_ps=record_ps / 2.0 + 0.05,
                                   window_centre_ps=float(np.mean(accumulation.time_ps)),
                                   window_shape="tukey", taper_fraction=0.2,
                                   pad_factor=PAD_FACTOR, baseline_fraction=0.05)
    return elapsed, transformed


def log_ratio_matrix(transformed):
    """Complex ln(Y_k / Y_purged) over the band, per scan."""
    frequencies = transformed.frequencies_hz
    band = (frequencies >= BAND_THZ[0] * 1e12) & (frequencies <= BAND_THZ[1] * 1e12)
    spectra = transformed.spectra[:, band]
    reference = spectra[-PURGED_END_SCANS:].mean(axis=0)
    ratio = spectra / reference[None, :]
    log_amplitude = np.log(np.abs(ratio))
    phase = np.unwrap(np.angle(ratio), axis=1)
    return frequencies[band] / 1e12, log_amplitude, phase, np.abs(reference)


def decompose(frequencies_thz, log_amplitude, phase, weights):
    """Per scan: ln|R| = gain + tilt*f ; arg R = offset + 2*pi*f*delay. Plus what is left."""
    centred = frequencies_thz - frequencies_thz.mean()
    design = np.column_stack([np.ones_like(centred), centred]) * weights[:, None]
    amplitude_terms = np.linalg.lstsq(design, (log_amplitude * weights).T, rcond=None)[0]
    phase_terms = np.linalg.lstsq(design, (phase * weights).T, rcond=None)[0]
    amplitude_residual = log_amplitude - (amplitude_terms[0][:, None]
                                          + amplitude_terms[1][:, None] * centred)
    phase_residual = phase - (phase_terms[0][:, None] + phase_terms[1][:, None] * centred)
    # numpy FFT: a later arrival is a negative phase slope; slope is per THz -> delay in fs.
    delay_fs = -phase_terms[1] / (2.0 * np.pi) * 1e3
    return {"gain": amplitude_terms[0], "tilt_per_thz": amplitude_terms[1],
            "phase_offset": phase_terms[0], "delay_fs": delay_fs,
            "amplitude_residual": amplitude_residual, "phase_residual": phase_residual}


def dominant_shapes(log_amplitude, phase, weights):
    """SVD of the stacked (log-amplitude | phase) matrix after removing each scan's delay.

    Returns the fraction of variance in the first modes and the first mode's spectral shape.
    """
    matrix = np.hstack([log_amplitude * weights, phase * weights])
    matrix = matrix - matrix.mean(axis=0)
    _, singular, right = np.linalg.svd(matrix, full_matrices=False)
    explained = singular**2 / np.sum(singular**2)
    return explained, right[0]


def block_residuals(elapsed_seconds, values, tau_seconds, start_minutes, block_minutes=40.0):
    """RMS left in a block after (a) a line, (b) A*exp(-t/tau) + c, (c) a quadratic."""
    start = start_minutes * 60.0
    inside = (elapsed_seconds >= start) & (elapsed_seconds <= start + block_minutes * 60.0)
    if inside.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    time, data = elapsed_seconds[inside], values[inside]
    line = np.polyval(np.polyfit(time, data, 1), time)
    design = np.column_stack([np.exp(-time / tau_seconds), np.ones_like(time)])
    exponential = design @ np.linalg.lstsq(design, data, rcond=None)[0]
    quadratic = np.polyval(np.polyfit(time, data, 2), time)
    noise = float(np.std(np.diff(data)) / np.sqrt(2.0))
    return (float(np.ptp(data)), float(np.sqrt(np.mean((data - line) ** 2))),
            float(np.sqrt(np.mean((data - exponential) ** 2))),
            float(np.sqrt(np.mean((data - quadratic) ** 2))), noise)


def main():
    results = {}
    for label, filename in RUNS.items():
        elapsed, transformed = load_run(filename)
        frequencies_thz, log_amplitude, phase, reference_amplitude = log_ratio_matrix(transformed)
        weights = reference_amplitude / reference_amplitude.max()
        terms = decompose(frequencies_thz, log_amplitude, phase, weights)
        explained, shape = dominant_shapes(log_amplitude, phase - np.outer(
            -2.0 * np.pi * terms["delay_fs"] * 1e-3, frequencies_thz), weights)
        results[label] = dict(elapsed=elapsed, frequencies_thz=frequencies_thz, terms=terms,
                              explained=explained, shape=shape, weights=weights)

        print(f"\n=== {label}: {elapsed.size} scans over {elapsed[-1] / 60:.1f} min")
        for name, unit in (("delay_fs", "fs"), ("gain", "ln"), ("tilt_per_thz", "ln/THz")):
            values = terms[name]
            fit = fit_exponential_equilibration(elapsed, values)
            tau = fit["tau_seconds"] / 60.0
            print(f"   {name:13s} change {values[-1] - values[0]:+9.4f} {unit:7s} "
                  f"exp-vs-line {fit['improvement_over_linear']:6.1f}x  tau "
                  f"{tau:6.1f} min  transient={fit['transient_detected']}")
        leftover = np.sqrt(np.mean((terms["amplitude_residual"][:, :] * weights) ** 2))
        print(f"   after gain+tilt+offset+delay, rms log-amplitude residual {leftover:.4f}; "
              f"SVD modes (delay removed): {np.round(explained[:3], 3)}")

    print("\n=== Is the change sample-independent? Tilt-per-gain and shape correlation")
    reference_label = "cnt_12mm (unpurged start)"
    common_axis = np.linspace(BAND_THZ[0], BAND_THZ[1], 80)

    def shape_on_common_axis(entry):
        """First mode's log-amplitude half, interpolated and normalised (axes differ by run)."""
        half = entry["shape"][:entry["frequencies_thz"].size]
        values = np.interp(common_axis, entry["frequencies_thz"], half)
        return values / np.linalg.norm(values)

    reference_shape = shape_on_common_axis(results[reference_label])
    for label, entry in results.items():
        terms = entry["terms"]
        tilt_span = terms["tilt_per_thz"][0] - terms["tilt_per_thz"][-1]
        delay_span = terms["delay_fs"][0] - terms["delay_fs"][-1]
        ratio = tilt_span / delay_span if abs(delay_span) > 1.0 else np.nan
        correlation = abs(float(np.dot(shape_on_common_axis(entry), reference_shape)))
        print(f"   {label:28s} tilt span {tilt_span:+.4f} /THz, delay span {delay_span:+7.1f} fs, "
              f"tilt per fs {ratio:+.5f}, first-mode |corr| with 12mm {correlation:.3f}")

    print("\n=== What a 40-min block leaves after drift removal (12mm run, start = box closure)")
    entry = results[reference_label]
    for name, tau_min, scale, unit in (("delay_fs", 38.0, 1.0, "fs"),
                                       ("tilt_per_thz", 47.0, 100.0, "%/THz")):
        for start in (10.0, 30.0, 60.0):
            span, linear, exponential, quadratic, noise = block_residuals(
                entry["elapsed"], entry["terms"][name], tau_min * 60.0, start)
            print(f"   {name:13s} from {start:3.0f} min: change {span * scale:6.2f} {unit}; "
                  f"left: line {linear * scale:5.3f}, exp(tau={tau_min:.0f}) "
                  f"{exponential * scale:5.3f}, quadratic {quadratic * scale:5.3f}; "
                  f"scan noise {noise * scale:5.3f}")


if __name__ == "__main__":
    main()
