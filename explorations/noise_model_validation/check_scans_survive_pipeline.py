"""Phase 0 end-to-end check: do the individual scans survive the real pipeline?

Runs the single-reflection preprocessing chain on real .acc data and reports, at each
step, whether the per-scan matrix is still intact and still consistent with the
averaged trace. Then hands the surviving scans to the noise estimator, which is the
whole point of keeping them.

    python explorations/noise_model_validation/check_scans_survive_pipeline.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
from thz_core.thz_core.noise import drift_corrected_scatter, fit_noise_parameters

DATA_DIR = "/home/match/data/CNTs/2026-08-20_CNT-paper-doped_2/export"

config = {
    "general": {"show_graph": False},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 5},
    "baseline": {"n_points": 10},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "transfer": {"self_reference": False, "apply_snr_mask": True,
                 "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30,
                 "unwrap_phase": True},
}

dataset = DataSet(DATA_DIR, config=config)
dataset.load_all_data()


def report(stage: str) -> None:
    print(f"\n--- after {stage}")
    for filename, data_obj in dataset.data.items():
        status = thz.scan_matrix_status(data_obj)
        matrix = data_obj.processing_dict.get(thz._SCAN_MATRIX_KEY)
        rows = np.asarray(data_obj.data).shape[0]
        if matrix is None:
            print(f"    {filename:<38} (matrix not built yet, trace {rows} rows)")
            continue
        matrix = np.asarray(matrix, dtype=float)
        consistent = np.allclose(matrix[:, 1:].mean(axis=1),
                                 np.asarray(data_obj.data)[:, 1], atol=1e-12)
        mark = "OK  " if status["intact"] else "LOST"
        print(f"    {mark} {filename:<38} {status['n_scans']:>4} scans, "
              f"trace {rows:>4} rows, mean matches trace: {consistent}")
        if status["reason"]:
            print(f"         reason: {status['reason']}")


print("=" * 92)
print("PER-SCAN MATRIX THROUGH THE SINGLE-REFLECTION PREPROCESSING CHAIN")
print("=" * 92)
report("load")

thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
report("taper_and_pad_traces_universal  (row count CHANGES here)")

dataset.group_files(keywords=["type"])
thz.subtract_baseline(dataset)
report("subtract_baseline")

# Snapshot the scans BEFORE windowing. The window multiplies the quiet baseline by
# ~zero, which is exactly what it is for — but it means the additive noise sigma_alpha
# is no longer measurable after it. The model must therefore be FITTED on the
# pre-window trace, where all three terms are still visible, and the window applied
# afterwards during propagation (spectral_noise_moments takes sigma_t and the window
# separately, precisely so these two stay distinct).
pre_window_scans = {
    filename: np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY], dtype=float).copy()
    for filename, data_obj in dataset.data.items()
}

thz.window_single_pulse_fixed_width(dataset, half_width_ps=5, show_graph=False)
report("window_single_pulse_fixed_width")

print()
losses = thz.discarded_scan_matrix_report(dataset)
if not losses:
    print("[scan_matrix] no per-scan data was lost anywhere in the chain.")

print()
print("=" * 92)
print("THE POINT: the noise estimator now runs on pipeline-processed scans")
print("=" * 92)
print("WHERE the model may be fitted turns out to matter, and the reason is structural.")
print("The model assumes sigma_alpha is CONSTANT in time. The pipeline's taper and")
print("window multiply the trace — and therefore its additive noise — by a position-")
print("dependent factor, so after those steps sigma_alpha is no longer constant and the")
print("fit is dragged toward the attenuated samples. Fit on the RAW scans; carry the")
print("pipeline's weighting into the PROPAGATION instead (spectral_noise_moments takes")
print("sigma_t and the window separately, which is exactly what that separation is for).")
print(f"\n    {'acquisition':<34} {'stage':<12} {'sig_a/peak':>11} {'sig_b':>8} {'sig_tau':>9}")
for filename, data_obj in dataset.data.items():
    raw = np.asarray(data_obj.raw_data, dtype=float)
    dt = float(np.median(np.diff(raw[:, 0]))) * 1e-12
    stages = {
        "raw scans": raw[:, 1:].T,
        "pre-window": pre_window_scans[filename][:, 1:].T,
        "post-window": np.asarray(data_obj.processing_dict[thz._SCAN_MATRIX_KEY],
                                  dtype=float)[:, 1:].T,
    }
    if stages["raw scans"].shape[0] < 2:
        print(f"    {filename:<34} (no repeats available)")
        continue
    reference_peak = float(np.max(np.abs(stages["raw scans"].mean(axis=0))))
    for stage, waveforms in stages.items():
        parameters, _ = fit_noise_parameters(waveforms, dt)
        label = filename if stage == "raw scans" else ""
        print(f"    {label:<34} {stage:<12} "
              f"{parameters.sigma_alpha / reference_peak * 100:10.3f}% "
              f"{parameters.sigma_beta * 100:7.3f}% "
              f"{parameters.sigma_tau * 1e15:8.2f}fs")
    drift = drift_corrected_scatter(stages["raw scans"], dt)
    print(f"    {'':<34} {'drift':<12} inflation {drift.drift_inflation:.2f}x, "
          f"amplitude {np.ptp(drift.amplitudes) * 100:.2f}% p-p, "
          f"delay {np.ptp(drift.delays) * 1e15:.1f} fs p-p")
