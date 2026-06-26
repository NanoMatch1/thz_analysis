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
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-24_refl_testing\main_alignment_tests\CNT' # CNT data
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-24_refl_testing\main_alignment_tests\Si' # Silicon reference data, silicon pressed into SiO2 window, 45 deg incidence, s-pol. 2.08 mm quartz window thickness.
data_dir = r'C:\Users\Samuel\Data\THz\Sam\reflection_testing\2026-06-23_refl_CNT\export\silicon' # Silicon reference data, silicon pressed into SiO2 window, 45 deg incidence, s-pol. 2.08 mm quartz window thickness.
# data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-06-24_refl_testing\main_alignment_tests' # Silicon reference data, silicon pressed into SiO2 window, 45 deg incidence, s-pol. 2.08 mm quartz window thickness.

# ── Physical parameters ─────────────────────────────────
THICKNESS_M = 2.08e-3  # metres (quartz window thickness, measured 2.08 mm)
PS_TO_S = 1e-12

# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': True,
        'air_gap_explorer': False,   # open the interactive air-gap de-embed slider after inversion
        # 'save_database': True,          # save the dataset database after processing
    },
    "geometry": {
        "theta_external_deg": 45.0,   # external incidence angle
        "polarization": "s",          # s-pol (TE)
        "n_sio2": 1.96,               # SiO2 window refractive index (reference medium)
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
        "half_width_ps": 4,    # fixed half-width — SAME window for every pulse
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
        "snr_thresh_db": 10,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
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
        "eps_background": 11.7,
    },
    "air_gap": {
        # Route-A contact-gap de-embed (SiO2 | air d | CNT). When enabled, strips the gap
        # and re-inverts with AIR incidence so the saved n,k,sigma are gap-corrected.
        # Dial position_um / width_um with the slider, then paste the values here.
        # NOTE: d is UNCALIBRATED — a forward Drude fit to r_back prefers a SMALL gap
        # (~0-5 um); larger d over-strips and pushes n below 1. Pin d with the Si
        # benchmark (known n=3.418) before trusting absolute numbers.
        "enabled": False,
        "position_um": 10.0,    # d_mean: mean gap (round-trip phase strip). PLACEHOLDER pending Si.
        "width_um": 5.0,       # sigma_d: roughness spread (Debye-Waller magnitude un-suppression).
        "max_boost": 1.0e3,    # clip on 1/W so the suppressed high-f tail cannot explode.
    },
}

# import acquisition_editor
# acquisition_editor.process_directory(data_dir)

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
# dataset.plot_current()
thz.build_full_trace_reflection(dataset) # defines the reflection dataset - on for refl, off for trans
# TODO: Define half-width from minimum max array length
# --- wrap each raw trace as a full-trace reflection (both pulses on ONE shared
#     axis; regions only locate the pulses, they do not size the window) ---


# --- pair sample <-> reference ---
dataset.group_files(keywords=['type'])
dataset.grouping.show_matches()

# --- NO time alignment in the shared-axis self-reference path. ---
# Self-referencing forms W = Y2/Y1 within each trace, cancelling the absolute time origin
# structurally (the front pulse is each trace's own internal clock). With the symmetric
# absolute-time phase reference in fft_spectrum, W is invariant to a rigid axis shift, so
# align_to_reference is unnecessary here — and shifting only the sample used to LEAK a
# spurious linear phase into H (Audit 1). Still used by the segmented (non-self-ref) path;
# re-enable here only if you switch self_reference off.
# first_region = config['regions'].get('first_reflection')
# correlation_roi = first_region if (first_region and None not in first_region) else None
# thz.align_to_reference(
#     dataset, timing_segment='first_reflection', roi=correlation_roi, show_graph=False,
# )
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
    thz.plot_fft(dataset, normalise=False, scale='')

# --- self-referenced transfer function: H = (Y2/Y1)_sample / (Y2/Y1)_ref. ---
# The MATH is one pure function, thz_core.self_referenced_transfer (two nested
# divisions: W = Y2/Y1 per trace, then H = W_samp/W_ref); thz.transfer_function is
# the THIN wrapper that fetches the four spectra from the dataset and calls it.
# transfer + mask settings (incl. self_reference) come from dataset.config (single
# source of truth). The wrapper stows the decomposition for inspection below.
thz.transfer_function(dataset, ref_type='reference')

# --- ALIGNMENT QUALITY GATE: warn if the front-pulse correction C is structured. ---
# |C| ≈ 1 (flat) means the front reflection spots of sample and reference landed on the
# same focus -> self-referencing is clean. A structured |C| means the front spot drifted
# (e.g. window rotation sent it off the gate focus) and the self-referenced H is suspect.
# Audit (2026-06-24): std(|C|) cleanly separates good (~0.01) from misaligned (~0.53).
# Diagnostic only — it does NOT modify the data (dividing H by C does not recover shape;
# H/C = Y2_s/Y2_r re-injects the timing offset self-referencing correctly cancels).
thz.selfref_quality(dataset)

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

