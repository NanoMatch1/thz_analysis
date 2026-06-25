from __future__ import annotations

import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-REFLECTION pipeline (one bounce, gold-referenced).
#
# A clean, dedicated reflection pipeline for the ORDINARY single-bounce geometry:
# the sample reflects once, referenced to a gold mirror (r_gold = -1). It shares all
# the machinery and style of run_me_low-level.py but drops the two-reflection,
# window-coupled self-referencing — here the transfer function is the plain ratio
#     H = Y_sample / Y_reference,     r_sample = r_reference * H   (r_reference = -1)
#
# Built to DEMONSTRATE how a misaligned mirror between the reference and sample
# acquisitions wrecks a reflection measurement: the misalignment delays/distorts the
# sample pulse, which shows up as a linear phase (and amplitude error) in H and a
# corrupted n,k. The fixed-width window NEVER shifts the data, so that timing error is
# preserved and visible rather than silently centred away.
# ─────────────────────────────────────────────────────────────────────────────

# ── Data paths ──────────────────────────────────────────
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-25_misalign_tests'

# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': True,
    },
    "geometry": {
        "theta_external_deg": 45.0,   # external incidence angle (air -> sample)
        "polarization": "s",          # s-pol (TE)
        "r_reference": -1.0,          # gold-mirror reference reflection coefficient
    },
    "regions": {
        # Optional (start_ps, end_ps) to constrain the peak search; None = whole trace.
        "pulse": None,
    },
    "centering": {
        # Recalibration: reset every pulse to a common peak T0 (removes the relative
        # timing). Use it to SIMULATE correcting the delay from angular misalignment —
        # whatever still corrupts H afterwards is the part a delay-correction can't fix.
        "recalibrate": True,        # True to align all peaks to a common T0
        "target_t0_ps": None,        # None = reference peak; else a fixed time
        "subsample": True,           # precise interpolation shift vs integer roll
    },
    "window": {
        "type": "hann",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 1.0,            # tukey only
        "half_width_ps": 3.0,    # fixed half-width — SAME window for every pulse, no shift
    },
    "fft": {
        "norm": "backward",
        "amplitude_scale": 1.0,
        "n_fft": 2000,            # shared FFT length (zero-pad for display resolution)
    },
    "transfer": {
        "self_reference": False,    # plain ratio H = Y_sample / Y_reference (single bounce)
        "apply_snr_mask": True,
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 10,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "derive": {
        # sigma = -i*omega*eps0*(eps - eps_background). 1.0 = vacuum.
        "eps_background": 1.0,
    },
}

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
dataset.plot_current()

# --- pair sample <-> reference (gold) ---
dataset.group_files(keywords=['type'])
dataset.grouping.show_matches()

# --- baseline removal (operates on the single main trace) ---
thz.subtract_baseline(dataset)

# --- OPTIONAL centering recalibration: reset all peaks to a common T0. ---
# OFF by default, so the raw misalignment delay is preserved into H (the default demo).
# Turn ON (config['centering']['recalibrate']) to SIMULATE correcting the angular-
# misalignment delay: the gross linear phase is removed, but the amplitude/cone
# distortion remains — showing what a pure delay-correction cannot rescue.
if config['centering'].get('recalibrate', False):
    thz.recenter_peaks_to_common_t0(
        dataset,
        target_t0_ps=config['centering'].get('target_t0_ps'),
        region_ps=config['regions'].get('pulse'),
        subsample=config['centering'].get('subsample', True),
        show_graph=config['general']['show_graph'],
    )

# --- THE window step: one identical fixed-width symmetric window on each pulse's
#     peak. The data is NEVER shifted, so the sample-vs-reference timing (and any
#     misalignment delay) is preserved into the transfer function. ---
thz.window_single_pulse_fixed_width(
    dataset,
    half_width_ps=config['window']['half_width_ps'],
    region_ps=config['regions'].get('pulse'),
    show_graph=config['general']['show_graph'],
)  # window config read from dataset.config (single source of truth)

# --- FFT every trace onto ONE frequency grid (same n_fft -> identical grid). The
#     fixed-width window keeps the FULL trace length, and rfft(y, n) truncates when
#     n < len(y), which would cut off the pulse. Guard against that. ---
required_n_fft = thz.minimum_fft_length(dataset)
shared_n_fft = config['fft']['n_fft']
assert shared_n_fft >= required_n_fft, (
    f"config['fft']['n_fft'] = {shared_n_fft} would TRUNCATE the windowed trace "
    f"({required_n_fft} samples) and cut off the reflection pulse. "
    f"Set n_fft >= {required_n_fft} (zero-pads above that for finer frequency spacing)."
)
thz.fft_spectrum(dataset, n_fft=shared_n_fft)

if config['general']['show_graph']:
    thz.plot_fft(dataset, normalise=False, scale='')

# --- transfer function: plain single-bounce ratio H = Y_sample / Y_reference. ---
# self_reference is OFF (config) so this is the ordinary reference ratio. A misaligned
# mirror between the two acquisitions delays/distorts the sample pulse -> linear phase
# and amplitude error in H -> corrupted n,k below.
thz.transfer_function(dataset, ref_type='reference')

# --- reflection-mode inversion: H -> n, k for a single AIR -> sample bounce,
#     referenced to a gold mirror (geometry='gold': n_incident = 1, r_reference = -1). ---
thz.invert_nk_reflection(
    dataset,
    geometry='gold',
    theta_deg=config['geometry']['theta_external_deg'],
    polarization=config['geometry']['polarization'],
    r_reference=config['geometry']['r_reference'],
)

# --- complex permittivity + optical conductivity from n, k ---
thz.derive_eps_sigma(dataset)

dataset.save_database()

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)
