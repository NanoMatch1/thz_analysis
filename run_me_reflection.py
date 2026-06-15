"""Reflection-mode THz-TDS pipeline (window-coupled internal reflection).

Geometry: a sample pressed against the flat back face of a SiO2 window, probed
at 45 deg external incidence, s-polarised.  One acquisition contains both the
front (air->SiO2) first reflection and the back (SiO2->sample) second reflection.
The SiO2-only trace is the reference.

Two phases:

  1. segment(): load the raw traces and crop the first + second reflections into
     separate .acc files under <dir>/first_reflection/ and <dir>/second_reflection/
     (no zero-pad, no taper).  This is the only reflection-specific step.

  2. process(): point a DataSet at the ROOT directory (which contains
     first_reflection/ and second_reflection/ as subdirs).  DataSet.load_all_data()
     auto-detects the layout and builds THzDataReflection objects, so every
     pipeline step can be called once per segment — explicitly and inspectably.

Each pipeline step is called TWICE, once per segment, so both the first and second
reflections go through identical processing you can inspect independently.  There
are no hidden side effects; the segment= kwarg makes every operation explicit.
"""

import os

import numpy as np

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


# ---------------------------------------------------------------------------
# Phase 1 — segment the raw trace
# ---------------------------------------------------------------------------

def segment(dataset: DataSet, config: dict) -> str:
    """Crop the first + second reflections to separate .acc files.

    Returns the root output directory whose first_reflection/ and
    second_reflection/ subdirs can be passed straight to DataSet for phase 2.
    """
    show_graphs = config.get('show_graphs', False)
    interactive = config.get('interactive', False)
    gates = config.get('gates', None)

    segments = None if interactive else gates
    thz.segment_reflections(dataset, segments=segments, show_graph=show_graphs)
    return config['file_dir']


# ---------------------------------------------------------------------------
# Phase 2 — process both reflections through the full pipeline
# ---------------------------------------------------------------------------

def process(dataset: DataSet, pipeline_config: dict) -> DataSet:
    """Run the full two-segment pipeline on a root-dir DataSet.

    pipeline_config keys
    --------------------
    show_graphs : bool
    centering : dict
        Passed to center_pulse; keys: peak_mode, taper_ps.
    align : dict
        auto_range_ps: {segment: (t_start_ps, t_stop_ps) or None}.
        None triggers an interactive SpanSelector.
    window : dict
        Passed to window_time; keys: type, alpha (Tukey / Hann / etc).
    pad : dict
        Use n_samples (absolute) so both segments share the same FFT grid.
    theta_external_deg : float
    polarization : str
    n_sio2 : float
    """
    show = pipeline_config.get('show_graphs', False)

    # --- pair sample <-> reference ---
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    # --- baseline subtraction (removes DC offset from each segment) ---
    thz.subtract_baseline(dataset, segment='second_reflection')
    thz.subtract_baseline(dataset, segment='first_reflection')

    # --- truncate to common time extent, per segment ---
    thz.global_truncate(dataset, segment='second_reflection')
    thz.global_truncate(dataset, segment='first_reflection')

    # --- define peak-search regions interactively (or headless if preset) ---
    # Fills pipeline_config['align']['auto_range_ps'][segment] from SpanSelector
    # when the value is None.  If both are already set, this is a no-op.
    thz.define_alignment_regions(dataset, pipeline_config)

    # --- soft peak alignment (brings all pulses to the same temporal position) ---
    second_range_ps = pipeline_config.get('align', {}).get('auto_range_ps', {}).get('second_reflection')
    first_range_ps  = pipeline_config.get('align', {}).get('auto_range_ps', {}).get('first_reflection')

    thz.pre_window_align_peak(dataset, segment='second_reflection',
                              auto_range_ps=second_range_ps, show_graph=show)
    thz.pre_window_align_peak(dataset, segment='first_reflection',
                              auto_range_ps=first_range_ps,  show_graph=show)

    # --- centre each pulse at the temporal midpoint (zero-pads pre-pulse region) ---
    centering_config = {'centering': pipeline_config.get('centering', {'peak_mode': 'auto', 'taper_ps': 0.5})}

    thz.center_pulse(dataset, segment='second_reflection', config=centering_config, show_graph=show)
    thz.center_pulse(dataset, segment='first_reflection',  config=centering_config, show_graph=show)

    # --- apply time-domain window (Tukey/Hann) symmetrically about the peak ---
    window_config = {'window': pipeline_config.get('window', {'type': 'Hann', 'alpha': 0.1})}

    thz.window_time(dataset, segment='second_reflection', config=window_config, show_graph=show)
    thz.window_time(dataset, segment='first_reflection',  config=window_config, show_graph=show)

    # --- zero-pad to a common length.  Use n_samples (absolute) so both segments
    #     produce the same FFT frequency grid, making W = Y2/Y1 well-defined. ---
    pad_config = {'pad': pipeline_config.get('pad', {'n_samples': 8192})}

    thz.zero_pad(dataset, segment='second_reflection', config=pad_config, show_graph=show)
    thz.zero_pad(dataset, segment='first_reflection',  config=pad_config, show_graph=show)

    # --- FFT.  Second reflection first so its frequency grid can be used to
    #     force the first reflection onto the same n_fft, guaranteeing
    #     W = Y2/Y1 is well-defined sample-by-sample.  The first reflection
    #     also gets an absolute-time phase reference exp(-2πif·t₀) applied. ---
    thz.fft_spectrum(dataset, segment='second_reflection')

    # Derive n_fft from any already-computed second-reflection spectrum so the
    # first-reflection FFT always matches regardless of pad rounding.
    _any_obj = next(iter(dataset.data.values()))
    _second_freq = _any_obj.processing_dict.get('fft_freq')
    n_fft_shared = (2 * (len(_second_freq) - 1)) if _second_freq is not None else None

    thz.fft_spectrum(dataset, segment='first_reflection', n_fft=n_fft_shared)

    if show:
        thz.plot_fft(dataset, normalise=False, scale='')

    # --- H(f) = Y2_sample / Y2_reference, self-referenced to front pulse ---
    thz.transfer_function(
        dataset,
        config={'transfer': {'self_reference': True}},
        ref_type='reference',
    )

    # --- invert to n, k ---
    thz.invert_nk_reflection(
        dataset,
        geometry='window',
        theta_deg=pipeline_config['theta_external_deg'],
        polarization=pipeline_config['polarization'],
        n_window=pipeline_config['n_sio2'],
    )
    thz.derive_eps_sigma(dataset)

    return dataset


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(dataset: DataSet, band_thz: tuple = (0.5, 3.0)) -> None:
    """Print a quick numeric summary of n, k over a band for each sample."""
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        pd_ = data_obj.processing_dict
        freq = pd_.get('fft_freq')
        n    = pd_.get('n')
        k    = pd_.get('k')
        r    = pd_.get('reflection_r')
        if freq is None or n is None:
            print(f"{filename}: no inversion result.")
            continue
        f_thz = freq * 1e-12
        band = (f_thz >= band_thz[0]) & (f_thz <= band_thz[1]) & np.isfinite(n)
        theta_int = pd_.get('theta_internal_rad')
        r_ref     = pd_.get('r_reference')
        print(f"\n=== {filename} ===")
        print(f"  internal angle = {np.rad2deg(theta_int):.2f} deg, "
              f"r_(SiO2->air) = {r_ref:.4f}")
        if band.any():
            print(f"  band {band_thz[0]}-{band_thz[1]} THz ({band.sum()} bins):")
            print(f"    n  mean = {np.nanmean(n[band]):.3f}  "
                  f"(min {np.nanmin(n[band]):.3f}, max {np.nanmax(n[band]):.3f})")
            print(f"    k  mean = {np.nanmean(k[band]):.3f}  "
                  f"(min {np.nanmin(k[band]):.3f}, max {np.nanmax(k[band]):.3f})")
            print(f"    |r_sample| mean = {np.nanmean(np.abs(r[band])):.3f}")
        else:
            print("  no finite n in band.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # CONFIG
    # Root dir must contain first_reflection/ and second_reflection/ as subdirs.
    # DataSet.load_all_data() auto-detects them and builds THzDataReflection objects.
    # -------------------------------------------------------------------------
    # ROOT_DIR = r'C:\Users\Samuel\Data\THz\Sam\2026-06-03_CNT-paper\testing'
    ROOT_DIR = r'C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17'

    pipeline_config = {
        'show_graphs': True,
        'theta_external_deg': 45.0,
        'polarization': 's',
        'n_sio2': 1.95,
        # Centering: half-cosine taper at the pad junction
        'centering': {'peak_mode': 'auto', 'taper_ps': 1},
        # Time-domain window
        'window': {'type': 'Hann', 'alpha': 1},
        # Zero-pad: use n_samples (absolute) so both segments share the same FFT grid.
        # Run the pipeline once to see how many samples your longest gate has, then
        # set this to the next power of 2 above that.
        'pad': {'n_samples': 4096},
        # Peak-search regions for pre_window_align_peak.
        # Set to None to open the interactive SpanSelector on the first run,
        # then paste the printed values here for headless repeats.
        'align': {
            'auto_range_ps': {
                'second_reflection': None,  # e.g. (159.5, 168.8)
                'first_reflection':  None,  # e.g. (151.2, 158.5)
            },
        },
    }

    # -------------------------------------------------------------------------
    # PHASE 1 — segment (only needed once; skip if segmented/ already exists)
    # -------------------------------------------------------------------------
    # raw_dataset = DataSet(ROOT_DIR)
    # raw_dataset.load_all_data(case_insensitive=True)
    # segment(raw_dataset, {'file_dir': ROOT_DIR, 'interactive': True, 'show_graphs': True})

    # -------------------------------------------------------------------------
    # PHASE 2 — process
    # -------------------------------------------------------------------------
    dataset = DataSet(ROOT_DIR)
    dataset.load_all_data(case_insensitive=True)

    ds = process(dataset, pipeline_config)
    report(ds)
    thz.result_viewer(ds)
    dataset.save_database()
