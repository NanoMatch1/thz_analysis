from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import pipeline_registry, session_bundle

# Record every pipeline call into dataset.recipe so the run is replayable/reopenable later.
# Must run BEFORE the pipeline calls (it wraps the thz.* stages + DataSet setup methods).
pipeline_registry.activate_recording()


# ── Utilities (synthetic-reference generation; not part of the main flow) ─────
def export_thz_data_as_acc(data_obj, dest_path):
    """Export a THzData object to an .acc file using acquisition_editor.save_acc."""
    data_dict = {
        "filename": Path(dest_path).name,
        "header": data_obj.headers,                               # first-scan header lines
        "scan_headers": [s.headers for s in data_obj.data_list],  # per-scan headers
        "data": data_obj.raw_data,                                # (N_pts, 1+N_scans), time in ps
    }
    print(f"Exporting THzData to {dest_path}...")
    return acquisition_editor.save_acc(data_dict, dest_path)


def clone_and_export(data_obj, dest_path, time_shift_ps=0.0):
    """Clone a THzData object, apply a time shift, and export to an .acc file."""
    import copy
    synthetic = copy.deepcopy(data_obj)
    synthetic._data[:, 0] += time_shift_ps
    synthetic.raw_data[:, 0] += time_shift_ps
    synthetic._time_data += time_shift_ps
    return export_thz_data_as_acc(synthetic, dest_path)


# ── Data paths ──────────────────────────────────────────
data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans\ntype'
# data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans\ntype'
# data_dir = r'C:\Users\Samuel\Data\THz\calibration\silicon\chris'
# data_dir = r'C:\Users\Samuel\Data\THz\calibration\silicon\denis'
# Chris data centering info:
# [window_time_fixed_width] 'reference_rt.txt': peak 122.35 ps, window 161 samples (+/- 4.00 ps).
# [window_time_fixed_width] 'sample_rt.txt': peak 125.05 ps, window 161 samples (+/- 4.00 ps).
# [window_time_fixed_width] applied an IDENTICAL hann window of 161 samples to 2 traces (center_in_trace=True).

# ── Configuration ───────────────────────────────────────
config: dict = {
    "general": {
        'show_graph': False,
        'air_gap_explorer': False,   # open the interactive air-gap de-embed slider after inversion
        'preprocess_data': True,        # run the preprocessing steps (baseline, window, FFT) before transfer function
        'save_database': False,          # save the dataset database after processing
        'save_session': True,          # True (default dir) or a path -> write a replayable .thzbundle
        'session_notes': '',            # free-text notes stored in the bundle
        'propagate_uncertainty': True, # Monte-Carlo error bars on n/k/eps/sigma (additive-noise floor)
        'uncertainty_draws': 200,       # MC draws for the above
    },
    "resolution": {
        # True instrument resolution = 1/T_res, T_res = the SHORTER reflection window.
        # n_fft above oversamples that by ~16x (sinc interpolation, not real detail).
        # Plots ALWAYS place markers on the independent-resolution grid; the flag below
        # additionally DECIMATES the stored n/k/sigma/H arrays to that grid.
        "limit_to_instrument_resolution": True,
        "broadening_factor": 1.0,   # >1 (e.g. 2.0) for the conservative Hann main-lobe width
    },
    "sample": {
        # "thickness_m": 492e-6,        # silicon wafer thickness (sets n; your calibration knob)
        "thickness_m": 492e-6,        # silicon wafer thickness (sets n; your calibration knob)
    },
    "regions": {
        # Optional (start_ps, end_ps) to constrain the peak search; None = whole trace.
        "pulse": None,
    },
    "window": {
        "type": "Hann",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 0.1,            # tukey only
        # Fixed half-width window (reflection-style): wide enough for the main pulse,
        # narrow enough to EXCLUDE the Fabry-Perot echo (~2 n d / c after the main pulse)
        # so the single-pass invert_nk stays clean.
        "half_width_ps": 5.0,
        # Peak selection. With a large sample<->reference delay a single region can't
        # cleanly isolate the right pulse, and it isn't always the global max. Options:
        #   interactive_peak=True -> click the pulse you want per trace (snaps to local max)
        #   peak_ps=<float>       -> headless: snap to the local max near this time (ps)
        # else regions.pulse (band) else the global max.
        "interactive_peak": False,
        "peak_ps": None,
        "snap_halfwidth_ps": 0.1,
    },
    "pad": {
        "extend_factor": 1.0,     # trailing zero-pad for finer FFT resolution
    },
    "transfer": {
        "self_reference": False,    # plain ratio H = Y_sample / Y_reference (transmission)
        "apply_snr_mask": True,
        "min_ref_amp_rel": 1e-3,
        "regularization_eps": 1e-30,
        "unwrap_phase": True,
    },
    "mask": {
        "snr_thresh_db": 20,
        "tail_fraction": 0.25,
        "min_contiguous_bins": 3,
    },
    "derive": {
        # sigma = -i*omega*eps0*(eps - eps_background). 1.0 = vacuum (total response);
        # 11.7 = silicon lattice background (isolates free-carrier conductivity).
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

import acquisition_editor
# acquisition_editor.process_directory(data_dir)

show_graph = config['general']['show_graph']

dataset = DataSet(data_dir, config=config)

if config['general'].get('preprocess_data', True):
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    dataset.plot_current()

    # --- pair sample <-> reference ---
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    # --- baseline removal ---
    thz.subtract_baseline(dataset)

    # --- THE window step: select the main peak, CENTER it in the trace (extend the axis so
    #     the peak sits at the midpoint — this preserves the pulse's ABSOLUTE arrival time, so
    #     the sample<->reference group delay that carries n is untouched), then apply one
    #     identical fixed-width symmetric window. Replaces the old whole-trace Tukey + the
    #     interactive centering_manual. ---
    thz.window_time_fixed_width(
        dataset,
        half_width_ps=config['window']['half_width_ps'],
        region_ps=config['regions'].get('pulse'),
        peak_ps=config['window'].get('peak_ps'),
        interactive=config['window'].get('interactive_peak', False),
        snap_halfwidth_ps=config['window'].get('snap_halfwidth_ps', 2.0),
        center_in_trace=True,
        show_graph=show_graph,
    )  # window config read from dataset.config (single source of truth)

    # --- lay every trace on ONE shared absolute-time axis (group-delay-preserving) and
    #     zero-pad for finer FFT resolution. align_to_common_time_axis runs inside zero_pad. ---
    thz.zero_pad(dataset, show_graph=show_graph)
    thz.fft_spectrum(dataset)

    if show_graph:
        thz.plot_fft(dataset, freq_range=(0.0, 10), normalise=False, scale='')

    dataset.save_state()

else:
    dataset.load_state()  # load the preprocessed traces (baseline, window, FFT) from a .state file 

# --- transfer function: plain transmission ratio H = Y_sample / Y_reference. ---
thz.transfer_function(dataset, ref_type='reference')
# thz.remove_phase_offset(dataset)  # optional: remove the constant phase offset from H

# --- n, k from H and the sample thickness (single-pass transmission inversion). ---
thz.invert_nk(dataset, thickness_m=config['sample']['thickness_m'])

thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)
thz.apply_instrument_resolution(dataset, config)
# --- complex permittivity + optical conductivity from n, k ---
thz.derive_eps_sigma(dataset)

# --- MONTE-CARLO UNCERTAINTY PROPAGATION (opt-in) ---
# Pushes the transfer-function uncertainty (transfer_H_sigma / transfer_phase_sigma) through the
# SAME invert+derive, giving error bars on n/k/eps/sigma (shown in the viewer, written to the CSV).
# NOTE: additive-noise-only input -> optimistic lower bound (see uncertainty.py docstring).
if config['general'].get('propagate_uncertainty', False):
    from dataset_core.adapters import uncertainty
    uncertainty.propagate_uncertainty(
        dataset,
        uncertainty.make_transmission_reinvert(config['sample']['thickness_m']),
        n_draws=config['general'].get('uncertainty_draws', 200),
    )

# --- DRUDE FIT + KK-CONSISTENT THICKNESS (headless; opt-in) ---
# sigma_2 is a small difference of large numbers, so it is hypersensitive to d. The
# physically correct thickness is the one where ONE Drude fits BOTH sigma_1 and sigma_2
# (a KK-consistency constraint the fringe method can't use). transfer_H is d-independent,
# so this only re-inverts+refits per trial thickness — fast enough to optimise.
#   * fit_conductivity      : joint sigma_1+sigma_2 Drude fit, prints sigma_DC/tau/R^2.
#   * optimize_thickness    : scans d, picks the KK-consistent value (target the CONDUCTIVE
#                             wafer by name — thickness is per-wafer; an intrinsic wafer
#                             must not vote).
#   * thickness_slider      : GUI modality (d slider + live joint fit + Optimise button).
from dataset_core.adapters import conductivity_fitting as conductivity
if config.get('drude', {}).get('enabled', False):
    drude_cfg = config['drude']
    conductivity.fit_conductivity(dataset, fit_band_thz=drude_cfg.get('fit_band_thz'))
    if drude_cfg.get('optimize_thickness', False):
        result = conductivity.optimize_thickness(
            dataset,
            drude_cfg.get('thickness_bounds_m', (480e-6, 520e-6)),
            samples=drude_cfg.get('sample'),            # e.g. 'n-type' — the conductive wafer
            fit_band_thz=drude_cfg.get('fit_band_thz'),
        )
        config['sample']['thickness_m'] = result['best_value']  # adopt the KK-consistent d
    if drude_cfg.get('slider', False):
        conductivity.thickness_slider(
            dataset, drude_cfg.get('thickness_bounds_m', (480e-6, 520e-6)),
            sample=drude_cfg.get('sample'), fit_band_thz=drude_cfg.get('fit_band_thz'),
        )

if config['general'].get('save_database', True):
    dataset.save_database()

# --- SAVE A REPLAYABLE SESSION BUNDLE (opt-in) ---
# Writes <dir>.thzbundle/ with recipe.json (config + ordered steps + git SHA), snapshot.pkl
# (instant reload for display, no recompute) and report.md. Reopen later with
# session_bundle.load_session(path) (fast) or pipeline_registry.replay_recipe(...) (recompute,
# overridable). Set config['general']['save_session'] to a path, or True for a default beside
# the data.
_session_target = config['general'].get('save_session', False)
if _session_target:
    _bundle_dir = _session_target if isinstance(_session_target, str) else os.path.join(
        data_dir, f"{dataset.seriesname}.thzbundle")
    session_bundle.save_session(dataset, _bundle_dir, notes=config['general'].get('session_notes', ''))

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)
