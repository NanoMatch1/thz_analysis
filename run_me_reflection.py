"""Reflection-mode THz-TDS pipeline (window-coupled internal reflection).

Geometry: a sample pressed against the flat back face of a SiO2 window, probed
at 45 deg external incidence, s-polarised.  One acquisition contains both the
front (air->SiO2) first reflection and the back (SiO2->sample) second reflection.
The SiO2-only trace is the reference.

Single-pass flow (no save/reload round-trip):
  1. Load raw traces from ROOT_DIR.
  2. define_reflection_gates: drag SpanSelector (or use preset bounds) to set
     first and second reflection time windows — stored in pipeline_config['gates'].
  3. build_reflection_dataset: crop each raw trace in-memory to both gates,
     building THzDataReflection objects that retain all individual scans.
  4. group_files: pair sample <-> reference by filename keyword.
  5. align_to_reference (timing_segment='first_reflection'): cross-correlate on
     the front pulse (stable T0 keeper), apply integer shift to BOTH segments,
     store subsample residual on data_obj for transfer_function.
  6. Full processing pipeline: subtract_baseline, global_truncate,
     pre_window_align_peak, center_pulse, window_time, zero_pad, fft_spectrum
     — all called TWICE, once per segment.
  7. transfer_function (self_reference=True), invert_nk_reflection, derive_eps_sigma.

Set headless=True in pipeline_config to suppress all interactive windows and
SpanSelectors (requires gates to be preset in the config).
"""

import numpy as np

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # CONFIG
    # ROOT_DIR is the flat directory containing the raw .acc/.dat files.
    # -------------------------------------------------------------------------
    ROOT_DIR = r'C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-17'

    pipeline_config = {
        'headless': False,          # True → no SpanSelectors, no plot windows
        'show_graphs': True,
        'theta_external_deg': 45.0,
        'polarization': 's',
        'n_sio2': 1.95,
        # Centering: half-cosine taper at the pad junction
        'centering': {'peak_mode': 'auto', 'taper_ps': 1},
        # Time-domain window
        'window': {'type': 'Hann', 'alpha': 1},
        # Zero-pad: absolute sample count so both segments share the same FFT grid.
        'pad': {'n_samples': 4096},
        # Gate bounds for segmenting the raw trace into first/second reflections.
        # Set both to None to open interactive SpanSelectors on the first run,
        # then paste the printed values here for headless repeats.
        'gates': {
            'first_reflection':  None,   # e.g. (151.2, 158.5)
            'second_reflection': None,   # e.g. (159.5, 168.8)
        },
        # Set True to also write the gated segments as .acc files under ROOT_DIR/
        # (first_reflection/ and second_reflection/ subfolders).  Useful for
        # caching so a future run can reload from the segmented layout directly.
        'save_segmented': False,
    }

    headless = pipeline_config.get('headless', False)
    show = pipeline_config.get('show_graphs', False) and not headless

    # -------------------------------------------------------------------------
    # STEP 1 — load raw traces
    # -------------------------------------------------------------------------
    dataset = DataSet(ROOT_DIR)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)

    # -------------------------------------------------------------------------
    # STEP 2 — define reflection gates (interactive or from config)
    # -------------------------------------------------------------------------
    thz.define_reflection_gates(dataset, pipeline_config)

    # -------------------------------------------------------------------------
    # STEP 3 — build THzDataReflection objects in-memory (no file I/O)
    # -------------------------------------------------------------------------
    thz.build_reflection_dataset(dataset, pipeline_config['gates'])

    # -------------------------------------------------------------------------
    # STEP 3b — optionally cache gated segments to .acc files
    # Writes first_reflection/ and second_reflection/ under ROOT_DIR so a future
    # run can reload from the segmented layout without repeating segmentation.
    # -------------------------------------------------------------------------
    if pipeline_config.get('save_segmented', False):
        thz.segment_reflections(
            dataset,
            segments=pipeline_config['gates'],
            output_dir=ROOT_DIR,
            show_graph=show,
        )

    # -------------------------------------------------------------------------
    # STEP 4 — pair sample <-> reference
    # -------------------------------------------------------------------------
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    # -------------------------------------------------------------------------
    # STEP 5 — align sample to reference via front pulse (cross-correlation)
    # The integer shift is applied to BOTH segments; subsample residual stored
    # on data_obj.processing_dict for transfer_function.
    # -------------------------------------------------------------------------
    thz.align_to_reference(
        dataset,
        timing_segment='first_reflection',
        show_graph=show,
    )

    # -------------------------------------------------------------------------
    # STEP 6 — process both segments through the full pipeline
    # Each step is called TWICE, once per segment, for full explicitness.
    # -------------------------------------------------------------------------

    # --- baseline subtraction ---
    thz.subtract_baseline(dataset, segment='second_reflection')
    thz.subtract_baseline(dataset, segment='first_reflection')

    # --- truncate to common time extent per segment ---
    thz.global_truncate(dataset, segment='second_reflection')
    thz.global_truncate(dataset, segment='first_reflection')

    # --- soft peak alignment (intra-segment, across files) ---
    thz.pre_window_align_peak(dataset, segment='second_reflection', show_graph=show)
    thz.pre_window_align_peak(dataset, segment='first_reflection',  show_graph=show)

    # --- centre each pulse at the temporal midpoint ---
    centering_config = {'centering': pipeline_config.get('centering', {'peak_mode': 'auto', 'taper_ps': 0.5})}
    thz.center_pulse(dataset, segment='second_reflection', config=centering_config, show_graph=show)
    thz.center_pulse(dataset, segment='first_reflection',  config=centering_config, show_graph=show)

    # --- apply time-domain window ---
    window_config = {'window': pipeline_config.get('window', {'type': 'Hann', 'alpha': 0.1})}
    thz.window_time(dataset, segment='second_reflection', config=window_config, show_graph=show)
    thz.window_time(dataset, segment='first_reflection',  config=window_config, show_graph=show)

    # --- zero-pad to a common absolute length so both segments share the same FFT grid ---
    pad_config = {'pad': pipeline_config.get('pad', {'n_samples': 4096})}
    thz.zero_pad(dataset, segment='second_reflection', config=pad_config, show_graph=show)
    thz.zero_pad(dataset, segment='first_reflection',  config=pad_config, show_graph=show)

    # --- FFT ---
    thz.fft_spectrum(dataset, segment='second_reflection')

    # Force the first reflection onto the same n_fft as the second so W = Y2/Y1
    # is defined sample-by-sample regardless of any rounding in pad.
    _any_obj = next(iter(dataset.data.values()))
    _second_freq = _any_obj.processing_dict.get('fft_freq')
    n_fft_shared = (2 * (len(_second_freq) - 1)) if _second_freq is not None else None
    thz.fft_spectrum(dataset, segment='first_reflection', n_fft=n_fft_shared)

    if show:
        thz.plot_fft(dataset, normalise=False, scale='')

    # -------------------------------------------------------------------------
    # STEP 7 — transfer function, inversion, optical parameters
    # -------------------------------------------------------------------------
    thz.transfer_function(
        dataset,
        config={'transfer': {'self_reference': True}},
        ref_type='reference',
    )

    thz.invert_nk_reflection(
        dataset,
        geometry='window',
        theta_deg=pipeline_config['theta_external_deg'],
        polarization=pipeline_config['polarization'],
        n_window=pipeline_config['n_sio2'],
    )
    thz.derive_eps_sigma(dataset)

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------
    band_thz = (0.5, 3.0)
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

    if show:
        thz.result_viewer(dataset)

    dataset.save_database()
