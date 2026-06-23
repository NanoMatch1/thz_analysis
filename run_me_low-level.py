from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import sys
import os

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

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


# ── Data paths ──────────────────────────────────────────
data_dir = r'C:\Users\Samuel\Data\THz\CNTs\CNT-20'
ref_path = os.path.join(data_dir, "reference_air_au_mount_1.acc")
samp_path = os.path.join(data_dir, "sample_window-substrate_1_realign_backup.acc")

# ── Physical parameters ─────────────────────────────────
THICKNESS_M = 2.08e-3  # metres (quartz window thickness, measured 2.08 mm)
PS_TO_S = 1e-12

# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': True,
    },
    "geometry": {
        "theta_external_deg": 45.0,   # external incidence angle
        "polarization": "s",          # s-pol (TE)
        "n_sio2": 1.95,               # SiO2 window refractive index (reference medium)
    },
    "regions": {
        "first_reflection": (153.3, 160.3),  # (start_ps, end_ps) or None for auto
        "second_reflection": (176.25, 185.0),  # (start_ps, end_ps) or None for auto
    },
    "centering": {
        "mode": "crop",  # 'crop' or 'pad'
    },
    "window": {
        "type": "hann",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 1.0,            # tukey only
        "half_width_ps": 3.0,    # fixed half-width — SAME window for every pulse
    },
    "pad": {
        "extend_factor": 1.0,
    },
    "fft": {
        "norm": "backward",
        "amplitude_scale": 1.0,
        "n_fft": 1000,            # shared FFT length (zero-pad for display resolution)
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
        "eps_inf": 1
    },
}

# data_dict = load_raw_data(samp_path, ref_path)
# baseline_correct = step_baseline(data_dict, config)
dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
# dataset.plot_current()
# TODO: Define half-width from minimum max array length
# --- wrap each raw trace as a full-trace reflection (both pulses on ONE shared
#     axis; regions only locate the pulses, they do not size the window) ---
thz.build_full_trace_reflection(dataset)



# --- pair sample <-> reference ---
dataset.group_files(keywords=['type'])
dataset.grouping.show_matches()

# --- T0 calibration on the FRONT pulse (roi locks the cross-correlation onto it) ---
first_region = config['regions'].get('first_reflection')
correlation_roi = first_region if (first_region and None not in first_region) else None
thz.align_to_reference(
    dataset, timing_segment='first_reflection', roi=correlation_roi, show_graph=False,
)

thz.subtract_baseline(dataset)

# if not already defined in the config, this prompts for region selection on the time trace.
thz.define_reflection_regions(dataset, config)

# --- THE window step: one identical fixed-width symmetric Hann on every pulse.
#     half_width_ps sets the (shared) window length; the data is never shifted. ---
thz.window_pulses_fixed_width(
    dataset,
    half_width_ps=config['window']['half_width_ps'],
    config={'window': config['window']},
    show_graph=config['general']['show_graph'],
)

# dataset.plot_current()

# --- FFT both reflections onto ONE frequency grid. Same n_fft for both segments
#     so the grids match exactly; zero-padding happens inside the FFT (no separate
#     zero_pad / pad_to_common_grid step on the shared-axis path). ---
shared_n_fft = config['fft']['n_fft']
thz.fft_spectrum(dataset, segment='second_reflection', n_fft=shared_n_fft)
thz.fft_spectrum(dataset, segment='first_reflection', n_fft=shared_n_fft)

if config['general']['show_graph']:
    thz.plot_fft(dataset, normalise=False, scale='')

# --- self-referenced transfer function: H = (Y2/Y1)_sample / (Y2/Y1)_ref.
#     self_reference=True forms the front/back ratio within each acquisition first,
#     so the sample<->reference timing cancels structurally. ---
thz.transfer_function(
    dataset, config={'transfer': {'self_reference': False}}, ref_type='reference',
)

# --- reflection-mode inversion: H -> n, k for the back-face (SiO2->sample)
#     interface. geometry='window' uses the SiO2 window as the incidence medium. ---
thz.invert_nk_reflection(
    dataset,
    geometry='window',
    theta_deg=config['geometry']['theta_external_deg'],
    polarization=config['geometry']['polarization'],
    n_window=config['geometry']['n_sio2'],
)

# --- complex permittivity + optical conductivity from n, k ---
thz.derive_eps_sigma(dataset)

dataset.save_database()

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)

breakpoint()

