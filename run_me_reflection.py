"""Reflection-mode THz-TDS pipeline (window-coupled internal reflection).

Geometry: a sample pressed against the flat back face of a SiO2 window, probed at
45 deg external incidence, s-polarised. One acquisition contains both the front
(air->SiO2) first reflection and the back (SiO2->sample) second reflection. The
SiO2-only trace is the reference. We self-reference: H = (Y2/Y1)_sample / (Y2/Y1)_ref.

TWO processing paths are wired here so they can be A/B compared on the same data.
Choose with pipeline_config['processing_path']:

  'shared_axis'  (new, ANALYSIS_NOTES §13-17)
      Keep BOTH reflections on ONE shared time axis the whole way, so the
      inter-pulse phase relationship is structural — no common-grid step, no
      absolute-time phase factor, no sub-sample ramp bookkeeping.
        load -> build_full_trace_reflection (whole trace, no crop)
             -> group -> align_to_reference (front pulse, integer-only is enough)
             -> subtract_baseline -> global_truncate
             -> define_reflection_regions (one call, both regions)
             -> isolate_and_window (one call, both regions, symmetric, shared axis)
             -> fft_spectrum x2 (same n_fft) -> transfer (self_reference)
             -> invert_nk_reflection -> derive_eps_sigma

  'segmented'  (original)
      Crop each reflection to its own gate up front, then run the per-segment
      chain (centering_manual, center_pulse, window_time, zero_pad) twice.

Set headless=True to suppress all SpanSelectors / plot windows (needs presets).
"""

import numpy as np

from dataset_core.dataset import DataSet
from dataset_core.adapters import thz_adapter as thz


# ---------------------------------------------------------------------------
# Path A — shared-axis (new)
# ---------------------------------------------------------------------------

def process_shared_axis(dataset: DataSet, config: dict, show: bool) -> DataSet:
    """Isolate both reflections on one shared axis, then self-reference and invert."""

    # --- wrap each raw trace as a full-trace THzDataReflection. Both segments
    #     start as the WHOLE trace; any GaP echo is excluded later by the second
    #     region's trailing edge (isolate_and_window zeros outside each region). ---
    thz.build_full_trace_reflection(dataset)
    # dataset.plot_current(title="full-trace reflection")

    # --- pair sample <-> reference ---
    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    # --- T0 calibration on the FRONT pulse. In self-reference mode the sample<->ref
    #     timing cancels in W, so integer-sample alignment is enough; we only need a
    #     clean common axis. roi restricts the cross-correlation to the first
    #     reflection so it locks on the front pulse, not the back. ---
    first_region = config['regions'].get('first_reflection')
    correlation_roi = first_region if (first_region and None not in first_region) else None
    thz.align_to_reference(
        dataset, timing_segment='first_reflection', roi=correlation_roi, show_graph=False,
    )

    # --- baseline off the genuine pre-pulse, on the full trace (both holders) ---
    thz.subtract_baseline(dataset, segment='second_reflection')
    thz.subtract_baseline(dataset, segment='first_reflection')

    # --- one common time axis across all files (needed so the FFT grids match) ---
    thz.global_truncate(dataset, segment='second_reflection')
    thz.global_truncate(dataset, segment='first_reflection')

    # --- define BOTH reflection regions (preset bounds or SpanSelector) ---
    thz.define_reflection_regions(dataset, config)

    # --- THE coupling step: isolate + symmetric window both reflections on one
    #     shared axis. center_mode 'pad' keeps the full pulse (pads the short side
    #     with zeros); 'crop' shrinks to the short side. ---
    # thz.isolate_and_window(
    #     dataset,
    #     config={'window': config['window']},
    #     center_mode=config.get('center_mode', 'crop'),
    #     show_graph=True,
    # )
    thz.cent
    # breakpoint()
    # thz.isolate_regions(dataset, config)
    # thz.center_pulses(dataset, mode=config.get('center_mode', 'crop'), show_graph=show)



    dataset.plot_current(title="isolated + windowed reflections")
    # --- FFT both reflections onto ONE frequency grid (zero-pad inside the FFT via
    #     n_fft — no separate zero_pad / pad_to_common_grid step needed). ---
    shared_n_fft = config.get('n_fft', 4096)
    thz.fft_spectrum(dataset, segment='second_reflection', n_fft=shared_n_fft)
    thz.fft_spectrum(dataset, segment='first_reflection', n_fft=shared_n_fft)

    if show:
        thz.plot_fft(dataset, normalise=False, scale='')

    # --- self-referenced transfer, inversion, optical parameters ---
    thz.transfer_function(
        dataset, config={'transfer': {'self_reference': True}}, ref_type='reference',
    )
    thz.invert_nk_reflection(
        dataset, geometry='window',
        theta_deg=config['theta_external_deg'],
        polarization=config['polarization'],
        n_window=config['n_sio2'],
    )
    thz.derive_eps_sigma(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Path B — segmented (original)
# ---------------------------------------------------------------------------

def process_segmented(dataset: DataSet, config: dict, show: bool) -> DataSet:
    """Crop each reflection to its gate up front, then the per-segment chain x2."""

    thz.define_reflection_gates(dataset, config)
    if config.get('save_segmented', False):
        thz.segment_reflections(
            dataset, segments=config['gates'], output_dir=config['root_dir'], show_graph=show,
        )
    thz.build_reflection_dataset(dataset, config['gates'])

    dataset.group_files(keywords=['type'])
    dataset.grouping.show_matches()

    thz.align_to_reference(dataset, timing_segment='first_reflection', show_graph=show)

    for segment in ('second_reflection', 'first_reflection'):
        thz.subtract_baseline(dataset, segment=segment)
        thz.global_truncate(dataset, segment=segment)
        # Use the gate as the peak-search range so this runs headless (otherwise
        # centering_manual opens a SpanSelector and blocks).
        thz.centering_manual(
            dataset, segment=segment,
            auto_range_ps=config['gates'].get(segment), show_graph=show,
        )
        thz.center_pulse(dataset, segment=segment,
                         config={'centering': config['centering']}, show_graph=show)
        thz.window_time(dataset, segment=segment,
                        config={'window': config['window']}, show_graph=show)
        thz.zero_pad(dataset, segment=segment, config={'pad': config['pad']}, show_graph=show)

    thz.fft_spectrum(dataset, segment='second_reflection')
    any_obj = next(iter(dataset.data.values()))
    second_freq = any_obj.processing_dict.get('fft_freq')
    shared_n_fft = (2 * (len(second_freq) - 1)) if second_freq is not None else None
    thz.fft_spectrum(dataset, segment='first_reflection', n_fft=shared_n_fft)

    if show:
        thz.plot_fft(dataset, normalise=False, scale='')

    thz.transfer_function(
        dataset, config={'transfer': {'self_reference': False}}, ref_type='reference',
    )
    thz.invert_nk_reflection(
        dataset, geometry='window',
        theta_deg=config['theta_external_deg'],
        polarization=config['polarization'],
        n_window=config['n_sio2'],
    )
    thz.derive_eps_sigma(dataset)
    return dataset


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(dataset: DataSet, band_thz: tuple = (0.5, 3.0)) -> None:
    """Print band-mean n, k, |r| per sample."""
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        frequency_hz = processing.get('fft_freq')
        refractive_index = processing.get('n')
        extinction = processing.get('k')
        reflection_r = processing.get('reflection_r')
        if frequency_hz is None or refractive_index is None:
            print(f"{filename}: no inversion result.")
            continue
        frequency_thz = frequency_hz * 1e-12
        band = ((frequency_thz >= band_thz[0]) & (frequency_thz <= band_thz[1])
                & np.isfinite(refractive_index))
        print(f"\n=== {filename} ===")
        if band.any():
            print(f"  band {band_thz[0]}-{band_thz[1]} THz ({band.sum()} bins):")
            print(f"    n mean = {np.nanmean(refractive_index[band]):.3f}  "
                  f"(min {np.nanmin(refractive_index[band]):.3f}, "
                  f"max {np.nanmax(refractive_index[band]):.3f})")
            print(f"    k mean = {np.nanmean(extinction[band]):.3f}  "
                  f"(min {np.nanmin(extinction[band]):.3f}, "
                  f"max {np.nanmax(extinction[band]):.3f})")
            if reflection_r is not None:
                print(f"    |r_sample| mean = {np.nanmean(np.abs(reflection_r[band])):.3f}")
        else:
            print("  no finite n in band.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    ROOT_DIR = r'C:\Users\Sam\Data\THz\CNT-17'
    ROOT_DIR = r'C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-16\A'
    ROOT_DIR = r'C:\Users\Samuel\Data\THz\calibration\reflection\2026_06_19_CNT\test1\export'

    pipeline_config = {
        # ---- path selection ----
        # 'processing_path': 'shared_axis',   # 'shared_axis' (new) or 'segmented' (old)
        'processing_path': 'segmented',   # 'shared_axis' (new) or 'segmented' (old)
        'root_dir': ROOT_DIR,
        'headless': False,                  # True -> no SpanSelectors / plot windows
        'show_graphs': True,

        # ---- geometry / inversion ----
        'theta_external_deg': 45.0,
        'polarization': 's',
        'n_sio2': 1.95,

        # ---- shared-axis path ----
        'centering': {'peak_mode': {}, 'taper_ps': 1.0},
        'center_mode': 'crop',               # 'pad' keeps the pulse tail; 'crop' shrinks
        'n_fft': 500,                      # FFT length (zero-pad for display resolution)
        'regions': {                        # first/second reflection regions (ps)
            # The second region's trailing edge excludes the GaP echo (no crop needed).
            'first_reflection':  (152.8, 160.25),
            'second_reflection': (176.3, 185.0),
        },

        # ---- window (shared by both paths) ----
        'window': {'type': 'hann', 'alpha': 1.0},

        # ---- segmented path only ----
        'gates': {'first_reflection':  (152.8, 160.25),
            'second_reflection': (176.3, 185.0)},
        'centering': {'peak_mode': 'auto', 'taper_ps': 1.0},
        'pad': {'n_samples': 500},
        'save_segmented': False,
    }

    headless = pipeline_config.get('headless', False)
    show = pipeline_config.get('show_graphs', False) and not headless

    dataset = DataSet(ROOT_DIR)
    dataset.load_all_data(case_insensitive=True, explicit_dir=True)
    # dataset.plot_current()

    processing_path = pipeline_config['processing_path']
    print(f"\n=== processing path: {processing_path} ===\n")
    if processing_path == 'shared_axis':
        result = process_shared_axis(dataset, pipeline_config, show)
    elif processing_path == 'segmented':
        result = process_segmented(dataset, pipeline_config, show)
    else:
        raise ValueError(f"unknown processing_path {processing_path!r}")

    report(result)

    # # if show:
    # breakpoint()
    # for filename, data_obj in dataset.data.items():
    #     print(f"\n=== {filename} ===")
    #     breakpoint()
    #     # thz.plot_nk(data_obj, freq_range=(0.0, 10))

    thz.result_viewer(result)

    dataset.save_database()
