from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import acquisition_editor
from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz


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
data_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-07-03_silicon_trans'
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
        'air_gap_explorer': True,   # open the interactive air-gap de-embed slider after inversion
        'preprocess_data': False,        # run the preprocessing steps (baseline, window, FFT) before transfer function
        'save_database': True,          # save the dataset database after processing
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
        "thickness_m": 500e-6,        # silicon wafer thickness (sets n; your calibration knob)
    },
    "regions": {
        # Optional (start_ps, end_ps) to constrain the peak search; None = whole trace.
        "pulse": None,
    },
    "window": {
        "type": "tukey",          # symmetric Hann (also 'tukey' / 'boxcar')
        "alpha": 0.1,            # tukey only
        # Fixed half-width window (reflection-style): wide enough for the main pulse,
        # narrow enough to EXCLUDE the Fabry-Perot echo (~2 n d / c after the main pulse)
        # so the single-pass invert_nk stays clean.
        "half_width_ps": 10.0,
        # Peak selection. With a large sample<->reference delay a single region can't
        # cleanly isolate the right pulse, and it isn't always the global max. Options:
        #   interactive_peak=True -> click the pulse you want per trace (snaps to local max)
        #   peak_ps=<float>       -> headless: snap to the local max near this time (ps)
        # else regions.pulse (band) else the global max.
        "interactive_peak": True,
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
}

import acquisition_editor
# acquisition_editor.process_directory(data_dir)

show_graph = config['general']['show_graph']

dataset = DataSet(data_dir, config=config)

if config['general'].get('preprocess_data', True):
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    # dataset.plot_current()

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
thz.remove_phase_offset(dataset)  # optional: remove the constant phase offset from H (for plotting only)

# --- n, k from H and the sample thickness (single-pass transmission inversion). ---
thz.invert_nk(dataset, thickness_m=config['sample']['thickness_m'])

thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)
thz.apply_instrument_resolution(dataset, config)
# --- complex permittivity + optical conductivity from n, k ---
thz.derive_eps_sigma(dataset)

if config['general'].get('save_database', True):
    dataset.save_database()

# --- inspect / launch the interactive result viewer ---
thz.result_viewer(dataset)
