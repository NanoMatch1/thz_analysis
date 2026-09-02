"""Old floor-based error bars vs the measured ones, on the data being presented.

Runs the single-reflection chain once, then computes the transfer-function
uncertainty both ways on the same processed dataset, so the only difference is the
estimator.

    python explorations/noise_model_validation/compare_error_bar_methods.py
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

DATA_ROOT = os.environ.get("THZ_DATA_ROOT", "/home/match/data")
DATASETS = {
    "2026-08-21 comparison": f"{DATA_ROOT}/CNTs/2026-08-21_CNT-paper-doped_3/comparison",
    "2026-08-20 export": f"{DATA_ROOT}/CNTs/2026-08-20_CNT-paper-doped_2/export",
}

config = {
    "general": {"show_graph": False},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 4.5},
    "baseline": {"n_points": 10},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "resolution": {"limit_to_instrument_resolution": False, "broadening_factor": 1.0},
    "transfer": {"self_reference": False, "apply_snr_mask": True,
                 "min_ref_amp_rel": 1e-3, "regularization_eps": 1e-30,
                 "unwrap_phase": True},
}

for label, data_dir in DATASETS.items():
    if not os.path.isdir(data_dir):
        print(f"\n### {label}: not found at {data_dir}")
        continue

    dataset = DataSet(data_dir, config=config)
    dataset.load_all_data()
    thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
    dataset.group_files(keywords=["type"])
    thz.subtract_baseline(dataset)
    thz.window_single_pulse_fixed_width(dataset, half_width_ps=4.5, show_graph=False)
    thz.fft_spectrum(dataset, n_fft=config["fft"]["n_fft"])
    thz.transfer_function(dataset, ref_type="reference")
    thz.compute_instrument_resolution(dataset, config)

    thz.compute_transfer_uncertainty(dataset)
    floor_sigma = {
        filename: np.asarray(obj.processing_dict["transfer_H_sigma"]).copy()
        for filename, obj in dataset.data.items()
        if obj.processing_dict.get("transfer_H_sigma") is not None
    }

    thz.compute_noise_from_scans(dataset, ref_type="reference", report=False)

    print(f"\n{'=' * 84}")
    print(f"### {label}")
    print("=" * 84)
    for filename, obj in dataset.data.items():
        measured = obj.processing_dict.get("transfer_H_sigma")
        if measured is None or filename not in floor_sigma:
            continue
        measured = np.asarray(measured)
        old = floor_sigma[filename]
        transfer_H = np.abs(np.asarray(obj.processing_dict["transfer_H"]))
        frequency_thz = np.asarray(obj.processing_dict["fft_freq"]) * 1e-12
        drift = obj.processing_dict.get("noise_drift")

        print(f"\n--- {filename}   ({obj.processing_dict.get('noise_source')})")
        if drift is not None:
            print(f"    drift inflation {drift.drift_inflation:.2f}x")
        print(f"    {'f (THz)':>8} {'|H|':>8} {'old sigma':>11} {'measured':>11} "
              f"{'old/new':>8} {'rel. err':>9}")
        for target in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            index = int(np.argmin(np.abs(frequency_thz - target)))
            ratio = old[index] / max(measured[index], 1e-30)
            print(f"    {frequency_thz[index]:8.2f} {transfer_H[index]:8.3f} "
                  f"{old[index]:11.4e} {measured[index]:11.4e} {ratio:8.1f} "
                  f"{measured[index] / max(transfer_H[index], 1e-30):8.2%}")
