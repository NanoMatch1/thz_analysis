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
data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-06-23_refl_CNT\export'

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
        "first_reflection": (152.5, 159),  # (start_ps, end_ps) or None for auto
        "second_reflection": (177, 183.8),  # (start_ps, end_ps) or None for auto
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
        "n_fft": 2000,            # shared FFT length (zero-pad for display resolution)
    },
    "transfer": {
        "self_reference": True,     # H = (Y2/Y1)_sample / (Y2/Y1)_ref (front-pulse referencing)
        "apply_snr_mask": True,     # build the SNR-based trusted-band mask
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 6.0,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "derive": {
        # Background permittivity subtracted to isolate free-carrier conductivity:
        # sigma = -i*omega*eps0*(eps - eps_background). 1.0 = vacuum; 11.7 = silicon.
        "eps_background": 1,
    },
}

# import acquisition_editor
# acquisition_editor.process_directory(data_dir)

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
# thz.align_to_reference(
#     dataset, timing_segment='first_reflection', roi=correlation_roi, show_graph=True,
# )

thz.subtract_baseline(dataset)

# if not already defined in the config, this prompts for region selection on the time trace.
thz.define_reflection_regions(dataset, config)

# --- THE window step: one identical fixed-width symmetric Hann on every pulse.
#     half_width_ps sets the (shared) window length; the data is never shifted. ---
thz.window_pulses_fixed_width(
    dataset,
    half_width_ps=config['window']['half_width_ps'],
    show_graph=config['general']['show_graph'],
)  # window config read from dataset.config (single source of truth)

# dataset.plot_current()

# --- FFT both reflections onto ONE frequency grid. Same n_fft for both segments
#     so the grids match exactly; zero-padding happens inside the FFT (no separate
#     zero_pad / pad_to_common_grid step on the shared-axis path). ---
# Guard against silent truncation: the windowed trace keeps its FULL shared-axis
# length (the window zeros outside the pulse but does not crop), and rfft(y, n)
# truncates when n < len(y) — which would cut off the pulses, since they sit late
# in the trace. n_fft must be at least the longest segment array.
required_n_fft = thz.minimum_fft_length(dataset)
# TODO: auto-derive n_fft from the data — when config['fft']['n_fft'] is None, set it
#       to the next power of two >= minimum_fft_length(dataset) instead of asserting a
#       hand-typed value (same "derive from the data" idea as the half-width TODO above).
shared_n_fft = config['fft']['n_fft']
assert shared_n_fft >= required_n_fft, (
    f"config['fft']['n_fft'] = {shared_n_fft} would TRUNCATE the windowed trace "
    f"({required_n_fft} samples) and cut off the reflection pulses. "
    f"Set n_fft >= {required_n_fft} (zero-pads above that for finer frequency spacing)."
)
thz.fft_spectrum(dataset, segment='second_reflection', n_fft=shared_n_fft)
thz.fft_spectrum(dataset, segment='first_reflection', n_fft=shared_n_fft)

if config['general']['show_graph']:
    thz.plot_fft(dataset, normalise=False, scale='')

# --- self-referenced transfer function: H = (Y2/Y1)_sample / (Y2/Y1)_ref.
#     transfer + mask settings (incl. self_reference) come from dataset.config so the
#     SNR mask threshold etc. actually carry through (single source of truth). ---
thz.transfer_function(dataset, ref_type='reference')

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

# breakpoint()

