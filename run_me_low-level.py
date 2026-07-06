"""Current working implementation of the reflection-mode THz inversion pipeline for samples pressed into the quartz window for internal reflection mode. """

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

def display_crop_regions(dataset):
    """Display the crop regions for each trace in the dataset."""
    config = dataset.config

    for filename, data_obj in dataset.data.items():
        # if dataset.data.is_reference(filename):
            # continue
        dataX = data_obj.data[:, 0] / PS_TO_S  # Assuming the first column is time
        dataY = data_obj.data[:, 1]  # Assuming the second column is amplitude
        first_region = config['regions'].get('first_reflection')
        second_region = config['regions'].get('second_reflection')
        
        plt.plot(dataX, dataY, label='Raw Data')
        # plt.show()
        if first_region is not None:
            start_ps, end_ps = first_region
            plt.axvspan(start_ps, end_ps, color='tab:blue', alpha=0.1, label='First Reflection Region')
            # plt.axvline(x=start_ps, color='r', linestyle='--', label='Start Region')
            # plt.axvline(x=end_ps, color='g', linestyle='--', label='End Region')
        if second_region is not None:
            start_ps, end_ps = second_region
            plt.axvspan(start_ps, end_ps, color='tab:orange', alpha=0.1, label='Second Reflection Region')
            # plt.axvline(x=start_ps, color='b', linestyle='--', label='Start Region 2')
            # plt.axvline(x=end_ps, color='c', linestyle='--', label='End Region 2')
            plt.title(f'Crop Regions for {filename}')
    plt.xlabel('Time (ps)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


# ── Data paths ──────────────────────────────────────────

data_dir = r'C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-21\polarization\silicon\p-pol' # Silicon reference data, silicon pressed into SiO2 window, 45 deg incidence, p-pol. 2.08 mm quartz window thickness.
# data_dir = r'C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-21\polarization\p-pol' # CNT paper pressed into SiO2 window, 45 deg incidence, p-pol. 2.08 mm quartz window thickness.


# ── Physical parameters ─────────────────────────────────
THICKNESS_M = 2.08e-3  # metres (quartz window thickness, measured 2.08 mm)
PS_TO_S = 1e-12

# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': True,
        'air_gap_explorer': False,   # open the interactive air-gap de-embed slider after inversion
        'save_database': True,          # save the dataset database after processing
    },
    "geometry": {
        "theta_external_deg": 45.0,   # external incidence angle
        "polarization": "p",          # p-pol (TM) / s-pol (TE)
        "n_sio2": 1.96,               # SiO2 window refractive index (reference medium)
    },
    "regions": {
        'first_reflection': (146.5, 159.5),  # cnt
        'second_reflection': (170.4, 186.9),  # cnt
        # "first_reflection": (145, 157.3),  # silicon
        # "second_reflection": (174, 183),  # silicon
    },
    "centering": {
        "mode": "crop",  # 'crop' or 'pad'
        "peak_mode": "auto", #, 'auto' or 'manual'
        "taper_ps": 1.0,  # half-cosine taper length in ps
    },
    "window": {
        "type": "hann",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 1.0,            # tukey only
        "half_width_ps": 3,    # fixed half-width — SAME window for every pulse
    },
    "pad": {
        "extend_factor": 1.0,
    },
    "fft": {
        "norm": "backward",
        "amplitude_scale": 1.0,
        "n_fft": 2000,            # shared FFT length (zero-pad for display resolution)
    },
    "resolution": {
        # True instrument resolution = 1/T_res, T_res = the SHORTER reflection window.
        # n_fft above oversamples that by ~16x (sinc interpolation, not real detail).
        # Plots ALWAYS place markers on the independent-resolution grid; the flag below
        # additionally DECIMATES the stored n/k/sigma/H arrays to that grid.
        "limit_to_instrument_resolution": True,
        "broadening_factor": 1.0,   # >1 (e.g. 2.0) for the conservative Hann main-lobe width
    },
    "transfer": {
        "self_reference": False,     # H = (Y2/Y1)_sample / (Y2/Y1)_ref (front-pulse referencing)
        "self_phase": True,          # H = (Y2/Y2)·(C/|C|): front pulse for TIMING only, no amp norm
                                     #   (ignored if self_reference True; replaces align_to_reference)
        "apply_snr_mask": True,     # build the SNR-based trusted-band mask
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 10,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "phase_kk": {
        # Kramers-Kronig (analytical-fit) phase correction (arXiv:2412.18662): removes the
        # sample-vs-reference MISPLACEMENT linear phase while PRESERVING intrinsic dispersion
        # (unlike phase_detrend, which strips material delay too). Polarization-independent;
        # only the downstream invert uses 's'/'p'. Runs between transfer and invert.
        "enabled": False,
        "f_end_thz": 2.5,       # KK truncation freq; None -> top of the trusted SNR band
        "fit_band_thz": (1, 2),    # None -> (0.15, 0.85) * f_end
        "use_snr_mask": True,
    },
    # "invert": {
    #     "min_one_plus_r": 0.1,  # near-mirror singularity floor (|1+r| < this -> n masked to NaN)
    # },
    "selfref_quality": {
        # Front-pulse correction C = Y1_r/Y1_s quality gate (diagnostic only).
        # Flags acquisitions whose front spot drifted (structured |C|). std(|C|)
        # is the discriminator: good alignment ~0.01, misaligned ~0.53.
        "band_thz": (0.2, 2.5),
        "std_threshold": 0.1,
        "median_dev_threshold": 0.15,
        "use_snr_mask": True,
    },
    "derive": {
        # Background permittivity subtracted to isolate free-carrier conductivity:
        # sigma = -i*omega*eps0*(eps - eps_background). 1.0 = vacuum; 11.7 = silicon.
        "eps_background": 11.6
    },
    "air_gap": {
        # Route-A contact-gap de-embed (SiO2 | air d | CNT). When enabled, strips the gap
        # and re-inverts with AIR incidence so the saved n,k,sigma are gap-corrected.
        # Dial position_um / width_um with the slider, then paste the values here.
        # NOTE: d is UNCALIBRATED — a forward Drude fit to r_back prefers a SMALL gap
        # (~0-5 um); larger d over-strips and pushes n below 1. Pin d with the Si
        # benchmark (known n=3.418) before trusting absolute numbers.
        "enabled": False,
        "position_um": 1,    # d_mean: mean gap (round-trip phase strip). PLACEHOLDER pending Si.
        "width_um": 0.0,       # sigma_d: roughness spread (Debye-Waller magnitude un-suppression).
        "max_boost": 0.0e3,    # clip on 1/W so the suppressed high-f tail cannot explode.
    },
}

# import acquisition_editor
# acquisition_editor.process_directory(data_dir)

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
# dataset.plot_current()
# display_crop_regions(dataset)
thz.build_full_trace_reflection(dataset) # defines the reflection dataset - on for refl, off for trans
thz.taper_and_pad_traces(dataset)
# dataset.plot_current()
# thz.center_pulse(dataset, show_graph=True)  # center the pulses in the trace (no crop, no pad)
# TODO: Define half-width from minimum max array length
# --- wrap each raw trace as a full-trace reflection (both pulses on ONE shared
#     axis; regions only locate the pulses, they do not size the window) ---


# --- pair sample <-> reference ---
dataset.group_files(keywords=['type', 'polarization'])
# dataset.group_files(keywords=['type', 'polarization'])
dataset.grouping.show_matches()

# --- NO time alignment in the shared-axis self-reference path. ---
# Self-referencing forms W = Y2/Y1 within each trace, cancelling the absolute time origin
# structurally (the front pulse is each trace's own internal clock). With the symmetric
# absolute-time phase reference in fft_spectrum, W is invariant to a rigid axis shift, so
# align_to_reference is unnecessary here — and shifting only the sample used to LEAK a
# spurious linear phase into H (Audit 1). Still used by the segmented (non-self-ref) path;
# re-enable here only if you switch self_reference off.
# Time-domain alignment is only needed for the PLAIN ratio (self_reference AND self_phase
# both off). self_phase corrects the timing structurally in the frequency domain (via the
# front-pulse phase), so skip the cross-correlation when it is on.
# if not config['transfer'].get('self_reference', False) and not config['transfer'].get('self_phase', False):
#     first_region = config['regions'].get('first_reflection')
#     correlation_roi = first_region if (first_region and None not in first_region) else None
#     thz.align_to_reference(
#         dataset, timing_segment='first_reflection', roi=correlation_roi, show_graph=False, subsample_correction=True
#     )

thz.define_reflection_regions(dataset, config)
thz.subtract_baseline(dataset)


# if not already defined in the config, this prompts for region selection on the time trace.

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
    thz.plot_fft(dataset, normalise=True, scale='log')

# --- self-referenced transfer function: H = (Y2/Y1)_sample / (Y2/Y1)_ref. ---
# The MATH is one pure function, thz_core.self_referenced_transfer (two nested
# divisions: W = Y2/Y1 per trace, then H = W_samp/W_ref); thz.transfer_function is
# the THIN wrapper that fetches the four spectra from the dataset and calls it.
# transfer + mask settings (incl. self_reference) come from dataset.config (single
# source of truth). The wrapper stows the decomposition for inspection below.
thz.transfer_function(dataset, ref_type='reference')

# --- KRAMERS-KRONIG PHASE CORRECTION (opt-in) ---
# Removes the sample-vs-reference misplacement phase (a linear-in-omega term from an
# unknown positioning shift) using the KK relation ln r = ln|r| - i*phi, which ties the
# correct phase to the reliable |r|. Unlike detrend_transfer_phase it preserves the
# intrinsic material dispersion, so it is safe for quantitative n,k. Polarization-agnostic
# (only invert_nk_reflection below picks 's'/'p'). Runs BEFORE the uncertainty/inversion so
# they see the corrected H. See phase_correct_kk docstring for the minimum-phase caveat.
if config.get('phase_kk', {}).get('enabled', False):
    thz.phase_correct_kk(dataset, show_graph=config['general']['show_graph'])

# --- ALIGNMENT QUALITY GATE: warn if the front-pulse correction C is structured. ---
# |C| ≈ 1 (flat) means the front reflection spots of sample and reference landed on the
# same focus -> self-referencing is clean. A structured |C| means the front spot drifted
# (e.g. window rotation sent it off the gate focus) and the self-referenced H is suspect.
# Audit (2026-06-24): std(|C|) cleanly separates good (~0.01) from misaligned (~0.53).
# Diagnostic only — it does NOT modify the data (dividing H by C does not recover shape;
# H/C = Y2_s/Y2_r re-injects the timing offset self-referencing correctly cancels).
thz.selfref_quality(dataset)

# --- TRUE INSTRUMENT RESOLUTION + FREQUENCY-DOMAIN ERROR ---
# n_fft zero-pads the spectra far past the physically independent resolution
# (df_res = 1/T_res, T_res = the shorter reflection window). compute_instrument_resolution
# records df_res and the decimation factor k; plots then mark one point per k bins so the
# eye reads the real resolution, not the interpolation. compute_transfer_uncertainty
# propagates the off-peak spectral noise floor through H = W_s/W_r to give per-bin error
# bars on |H| and phase (they grow where SNR falls — self-consistent with the SNR mask).
thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)

# --- TRANSPARENCY: surface the self-reference decomposition the core function returns,
#     so the two-division math is visible (W_samp, W_ref, the front-pulse drift
#     correction C = Y1_r/Y1_s, and the result H). Pure inspection — no processing. ---
for sample_filename, sample_obj in dataset.data.items():
    if dataset.data.is_reference(sample_filename):
        continue
    processing = sample_obj.processing_dict
    diagnostics = processing.get('transfer_metrics', {}).get('diagnostics', {})
    frequency_hz = processing.get('fft_freq')
    transfer_H = processing.get('transfer_H')
    front_pulse_correction = processing.get('selfref_correction')  # C = Y1_r / Y1_s
    W_sample = diagnostics.get('W_samp')        # Y2_s / Y1_s
    W_reference = diagnostics.get('W_ref')       # Y2_r / Y1_r
    if transfer_H is None or W_sample is None:
        continue  # not a self-referenced run (plain ratio path)

    trusted_band_mask = processing.get('transfer_mask')
    finite_correction = np.isfinite(front_pulse_correction)
    print(
        f"[self-ref decomposition] {sample_filename}: "
        f"H = (Y2_s/Y1_s)/(Y2_r/Y1_r);  |C| median "
        f"{np.nanmedian(np.abs(front_pulse_correction[finite_correction])):.3f}  "
        f"(C = Y1_r/Y1_s, the front-pulse drift correction)"
    )

    if config['general']['show_graph']:
        frequency_thz = frequency_hz * 1e-12
        band = (frequency_thz >= 0.2) & (frequency_thz <= 3.0)
        if trusted_band_mask is not None:
            band = band & trusted_band_mask
        figure, (magnitude_axis, phase_axis) = plt.subplots(
            1, 2, figsize=(13, 4.5), layout='constrained',
        )
        figure.suptitle(f'Self-reference decomposition — {sample_filename}', fontsize=11)
        for series, label, style in (
            (W_sample, 'W_samp = Y2_s/Y1_s', dict(color='C0')),
            (W_reference, 'W_ref = Y2_r/Y1_r', dict(color='C1')),
            (front_pulse_correction, 'C = Y1_r/Y1_s', dict(color='C2', ls=':')),
            (transfer_H, 'H = W_samp/W_ref', dict(color='C3', lw=1.8)),
        ):
            magnitude_axis.plot(frequency_thz[band], np.abs(series)[band], label=label, **style)
            phase_axis.plot(frequency_thz[band], np.angle(series)[band], label=label, **style)
        magnitude_axis.set_title('magnitude'); magnitude_axis.set_xlabel('Frequency (THz)')
        magnitude_axis.set_ylabel('|.|'); magnitude_axis.legend(fontsize=8); magnitude_axis.grid(alpha=0.3)
        phase_axis.set_title('phase'); phase_axis.set_xlabel('Frequency (THz)')
        phase_axis.set_ylabel('arg (rad)'); phase_axis.grid(alpha=0.3)
        plt.show()

# --- reflection-mode inversion: H -> n, k for the back-face (SiO2->sample)
#     interface. geometry='window' uses the SiO2 window as the incidence medium. ---
thz.invert_nk_reflection(
    dataset,
    geometry='window',
    theta_deg=config['geometry']['theta_external_deg'],
    polarization=config['geometry']['polarization'],
    n_window=config['geometry']['n_sio2'],
)

# thz.detrend_transfer_phase(dataset, show_graph=config['general']['show_graph'])

# --- AIR-GAP DE-EMBED (Route A, non-interactive) ---
# Strip the contact gap (SiO2 | air d | CNT) and re-invert with AIR incidence, so the
# SAVED n,k,sigma are the gap-corrected values. Removes the inversion-singularity artifact
# (the spurious ~0.6 THz "Lorentzian") that the residual gap phase manufactures near |r|=1.
# Gap parameters come from config['air_gap'] (dial them with the slider below, then paste
# here). Window-geometry n,k are preserved as processing['n_window']. Must run AFTER
# invert_nk_reflection and BEFORE derive_eps_sigma.
if config['air_gap'].get('enabled', False):
    thz.deembed_air_gap_reflection(dataset)

# --- complex permittivity + optical conductivity from n, k ---
thz.derive_eps_sigma(dataset)

# --- OPTIONAL: collapse every frequency-domain product onto the true-resolution grid ---
# No-op unless config['resolution']['limit_to_instrument_resolution'] is True. When on,
# decimates fft/H/mask/n/k/eps/sigma to one point per resolution element so the SAVED data
# reflects the measured resolution (not zero-pad interpolation). Run last, after all
# frequency-domain products exist, so they all land on the same decimated grid.
thz.apply_instrument_resolution(dataset, config)

# --- AIR-GAP DE-EMBED EXPLORER (interactive sliders) ---
# The CNT is pressed against the SiO2 back face, but the rough surface leaves a thin,
# uneven contact gap (SiO2 | air gap | CNT). The naive window-incidence inversion above
# folds that gap into n,k (the source of n<1 and possibly the suspect Lorentzian). This
# opens a slider GUI that de-embeds a trial gap and re-inverts live:
#   * gap POSITION d_mean (um): strips the round-trip phase (exact single-gap inverse).
#   * gap WIDTH sigma_d (um): undoes the Debye-Waller roughness magnitude loss (single-
#     bounce approx — indicative, not a fitted value).
# Good contact (no sample/reference back-reflection delay) => start near d_mean ~ 0 and
# explore sigma_d for the roughness. Reuses the pipeline's own inverted dataset, so it
# reflects every current correction (symmetric phase ref, fixed-width window, self-ref).
if config['general'].get('air_gap_explorer', False):
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'explorations', 'air_gap_cnt_reflection',
    ))
    import air_gap_slider_explorer
    air_gap_slider_explorer.launch_from_dataset(
        dataset,
        n_sio2=config['geometry']['n_sio2'],
        external_deg=config['geometry']['theta_external_deg'],
        eps_background=config['derive']['eps_background'],
        initial_position_um=0.0,   # good contact: little/no mean gap
        initial_width_um=0.0,      # raise to explore roughness suppression
    )

if config['general'].get('save_database', False):
    dataset.save_database()

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)

# breakpoint()

