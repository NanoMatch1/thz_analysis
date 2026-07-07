from __future__ import annotations

import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import pipeline_registry, session_bundle

# Record the pipeline into dataset.recipe (replayable/reopenable later). Run before the stages.
pipeline_registry.activate_recording()

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
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-25_misalign_tests\gold'
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-24_refl_testing\cone_tests'
data_dir = r'C:\Users\Samuel\Data\THz\diagnostics\2026-06-30_ref_testing'
# data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-1_CNT\export'
data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon\export'


# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': True,
        'air_gap_explorer': False,   # open the interactive air-gap de-embed slider after inversion
        'preprocess_data': False,        # run the preprocessing steps (baseline, window, FFT) before transfer function
        'save_database': False,          # save the dataset database after processing
        'save_session': True,          # True (default dir) or a path -> write a replayable .thzbundle
        'session_notes': '',            # free-text notes stored in the bundle
    },
    "resolution": {
        # True instrument resolution = 1/T_res, T_res = the SHORTER reflection window.
        # n_fft above oversamples that by ~16x (sinc interpolation, not real detail).
        # Plots ALWAYS place markers on the independent-resolution grid; the flag below
        # additionally DECIMATES the stored n/k/sigma/H arrays to that grid.
        "limit_to_instrument_resolution": True,
        "broadening_factor": 1.0,   # >1 (e.g. 2.0) for the conservative Hann main-lobe width
    },
    "geometry": {
        "theta_external_deg": 45.0, # Maybe this is actually closer to 48 or the other way   # external incidence angle (air -> sample)
        "polarization": "s",          # s-pol (TE)/
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
        "recalibrate": False,        # True to align all peaks to a common T0
        "target_t0_ps": None,        # None = reference peak; else a fixed time
        "subsample": False,           # precise interpolation shift vs integer roll
    },
    "window": {
        "type": "hann",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 1.0,            # tukey only
        "half_width_ps": 5,    # fixed half-width — SAME window for every pulse, no shift
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
        "correct_linear_phase": False,  # remove the linear phase from H (misalignment delay)
    },
    "mask": {
        "snr_thresh_db": 10,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "phase_kk": {
        "enabled": True,  # Kramers-Kronig phase correction (arXiv:2412.18662)
        "f_end_thz": None,       # KK truncation freq; None -> top of the trusted SNR band
        "fit_band_thz": (1,5),    # None -> (0.15, 0.85) * f_end
        "use_snr_mask": True,
        # "theta_deg": None,  # incidence angle for the REPORTED shift l only; None -> config['geometry']['theta_external_deg']
    },
    "invert": {
        # Near-mirror singularity floor. The r->n inversion blows up at r=-1 (perfect
        # mirror): a highly reflective sample (|H|~1, e.g. conductive CNT vs gold) makes n
        # spike to infinity wherever H crosses 1+0i. Bins with |1+r| below this floor are
        # ill-conditioned (n unrecoverable) and masked to NaN. 1e-12 = off (legacy);
        # ~0.1 caps the blow-ups at n~10. |H|>1 (unphysical) signals a coupling artifact.
        "min_one_plus_r": 0.1,
    },
    "derive": {
        # sigma = -i*omega*eps0*(eps - eps_background). 1.0 = vacuum.
        "eps_background": 11.7,
    },
    "drude": {
        # Headless Drude fit + KK-consistent thickness calibration (see the stage below).
        "enabled": False,
        "fit_band_thz": (0.35, 2.5),      # band for the joint sigma_1+sigma_2 Drude fit
        "sample": None,                   # substring of the CONDUCTIVE wafer, e.g. 'n-type'
        "optimize_thickness": False,      # scan d for the KK-consistent value, adopt it
        "thickness_bounds_m": (480e-6, 520e-6),
        "slider": True,                  # open the interactive d slider GUI
    },
}


# import acquisition_editor
# acquisition_editor.process_directory(data_dir)

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()

thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
# dataset.plot_current()
# data

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

# dataset.plot_current(title='before truncate')
# thz.global_truncate(dataset)
# dataset.plot_current(title='after truncate')
# --- THE window step: one ldentical fixed-width symmetric window on each pulse's
#     peak. The data is NEVER shifted, so the sample-vs-reference timing (and any
#     misalignment delay) is preserved into the transfer function. ---
thz.window_single_pulse_fixed_width(
    dataset,
    half_width_ps=config['window']['half_width_ps'],
    region_ps=config['regions'].get('pulse'),
    show_graph=config['general']['show_graph'],
)  # window config read from dataset.config (single source of truth)
dataset.plot_current()

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


# fft_dict = dataset.grab_data('fft')
# breakpoint()
# dataset.save_dict(fft_dict, 'cnt_noalign')

# for filename, data_dict in fft_dict.items():
#     dataX = data_dict['fft_freq']
#     dataY = data_dict['fft_spectrum']
#     plt.plot(dataX, np.abs(dataY), label=filename)

# plt.xlabel('Frequency (THz)')
# plt.ylabel('Amplitude')
# plt.title('FFT Spectra')
# plt.legend()
# plt.show()

# --- transfer function: plain single-bounce ratio H = Y_sample / Y_reference. ---
# self_reference is OFF (config) so this is the ordinary reference ratio. A misaligned
# mirror between the two acquisitions delays/distorts the sample pulse -> linear phase
# and amplitude error in H -> corrupted n,k below.
thz.transfer_function(dataset, ref_type='reference')

# --- EXPERIMENTAL: remove the LINEAR (group-delay) phase from H. A ~1 ps group delay from
#     the distorted CNT pulse shape sweeps arg(H) through 0 every ~1 THz, pushing r toward
#     the r=-1 pole and spiking n even when |H|<1. detrend_transfer_phase flattens that slope.
#     It removes any real material group delay too -> for SEEING the fix, not quantitative n.
#     (remove_phase_offset is the WRONG tool here: it strips the intercept and KEEPS the slope.)
if config['transfer'].get('correct_linear_phase', False):
    thz.detrend_transfer_phase(dataset, show_graph=config['general']['show_graph'])

if config['phase_kk'].get('enabled', False):
    thz.phase_correct_kk(dataset, show_graph=config['general']['show_graph'])

# thz.compute_instrument_resolution(dataset, config)
# thz.compute_transfer_uncertainty(dataset)
thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)

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

# --- DRUDE FIT (headless; opt-in) — one joint fit to sigma_1 AND sigma_2. ---
# Geometry-agnostic: works on the reflection sigma exactly as in transmission. Prints
# sigma_DC / tau / plasma freq / joint R^2. (The KK-consistent GEOMETRY optimiser for
# reflection would inject an air-gap re-inversion as apply_value; conductivity_fitting
# stays pipeline-agnostic via that callback.)
from dataset_core.adapters import conductivity_fitting as conductivity
if config.get('drude', {}).get('enabled', False):
    conductivity.fit_conductivity(dataset, fit_band_thz=config['drude'].get('fit_band_thz'))

# --- OPTIONAL: collapse every frequency-domain product onto the true-resolution grid ---
# No-op unless config['resolution']['limit_to_instrument_resolution'] is True. When on,
# decimates fft/H/mask/n/k/eps/sigma to one point per resolution element so the SAVED data
# reflects the measured resolution (not zero-pad interpolation). Run last, after all
# frequency-domain products exist, so they all land on the same decimated grid.
thz.apply_instrument_resolution(dataset, config)

if config['general']['save_database']:
    dataset.save_database()

# --- SAVE A REPLAYABLE SESSION BUNDLE (opt-in): set config['general']['save_session'] to a path
#     or True. Reopen with session_bundle.load_session (fast) or replay_recipe (recompute). ---
_session_target = config['general'].get('save_session', False)
if _session_target:
    _bundle_dir = _session_target if isinstance(_session_target, str) else os.path.join(
        data_dir, f"{dataset.seriesname}.thzbundle")
    session_bundle.save_session(dataset, _bundle_dir, notes=config['general'].get('session_notes', ''))

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)
