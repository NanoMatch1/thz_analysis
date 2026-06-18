from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

import thz_core as core

# from thz_core import (
#     align_on_peak,
#     derive_eps_sigma,
#     fft_spectrum,
#     invert_nk,
#     subtract_baseline,
#     transfer_function,
#     trusted_band_mask,
#     validate_thz_dict,
#     window_time,
#     pad_to_common_grid,
#     extend_grid,
# )
# from thz_core.examples.validate_pipeline import print_metrics

# auto_range for batch mode (index range for peak search)
# Set to None for interactive SpanSelector UI.
AUTO_RANGE = (47, 57)

def load_raw_data(sample_path, reference_path) -> dict:
    """Load reference and sample from text files.

    Returns
    -------
    dict
        ``data_dict`` mapping labels to (N, 2) arrays
        with columns [time_s, amplitude].
    """
    ref = np.loadtxt(reference_path)
    samp = np.loadtxt(sample_path)
    data_dict = {
        "reference": np.column_stack((
            ref[:, 0] * PS_TO_S, ref[:, 1],
        )),
        "sample": np.column_stack((
            samp[:, 0] * PS_TO_S, samp[:, 1],
        )),
    }
    for label, arr in data_dict.items():
        time_col = arr[:, 0]
        dt_ps = np.median(np.diff(time_col)) * 1e12
        print(
            f"Loaded {label:12s}: {time_col.size} pts, "
            f"dt = {dt_ps:.4f} ps, "
            f"range = [{time_col[0]*1e12:.2f}, "
            f"{time_col[-1]*1e12:.2f}] ps"
        )
    return data_dict

def step_baseline(
    data_dict: dict, config: dict
) -> dict:
    """Subtract DC baseline from every trace.

    Returns a new ``data_dict`` with the baseline removed
    from each amplitude column.
    """
    corrected, bl_metrics = subtract_baseline(
        data_dict, config,
    )
    print_metrics("Baseline subtraction", bl_metrics)
    return corrected



# ── Data paths ──────────────────────────────────────────
data_dir = r'C:\Users\Samuel\Data\THz\Sam\testing\cryostat_windows'
ref_path = os.path.join(data_dir, "reference_air_au_mount_1.acc")
samp_path = os.path.join(data_dir, "sample_window-substrate_1_realign_backup.acc")

# ── Physical parameters ─────────────────────────────────
THICKNESS_M = 2.08e-3  # metres (quartz window thickness, measured 2.08 mm)
PS_TO_S = 1e-12

# ── Configuration ───────────────────────────────────────
config: dict = {
    "window": {
        "type": "tukey",
        "alpha": 0.3,
        "zero_tail": True,
    },
    "pad": {
        "extend_factor": 4.0,
    },
    "fft": {
        "norm": "backward",
        "amplitude_scale": 1.0,
    },
    "transfer": {
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 6.0,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "invert": {
        "max_iterations": 30,
        "convergence_tol": 1e-12,
    },
}

data_dict = load_raw_data(samp_path, ref_path)
baseline_correct = step_baseline(data_dict, config)

breakpoint()

