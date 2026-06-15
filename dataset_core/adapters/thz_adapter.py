import numpy as np
import thz_core.thz_core as core
import matplotlib.pyplot as plt
from dataset_core.dataset import DataSet, DataService
from dataset_core.data_structures.thz import THzDataReflection
"""Used to bridge the DataSet manager and the thz analysis library.

Unit convention
~~~~~~~~~~~~~~~
THzData stores working data in SI units (seconds, Hz).  The adapter
passes arrays straight through to thz_core without conversion.
Display / export helpers convert Hz → THz and s → ps for readability.

Minimal adapter layer: each function extracts arrays from THzData objects,
calls the corresponding thz_core routine, and writes results back onto
the dataset (in-place). All functions return the dataset for chaining.
"""


_HZ_TO_THZ = 1e-12
_S_TO_PS = 1e12

# ---------------------------------------------------------------------------
# Single-line toggle for the sub-sample (fractional) timing correction.
#
# align_to_reference splits its cross-correlation shift into a whole-sample part
# (slid on the time axis) and a sub-sample residual. When this is True the
# residual is applied EXACTLY as a spectral phase ramp in transfer_function
# (Fourier shift theorem — no time-domain interpolation). Flip to False to
# disable it and re-run, e.g. to see how much the residual actually matters.
# A per-call `subsample_correction=` argument on align_to_reference overrides
# this default when you want to set it explicitly.
# ---------------------------------------------------------------------------
SUBSAMPLE_TIMING_CORRECTION = True

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalise(dataset: DataSet, config: dict | None = None, show_graph: bool = False) -> DataSet:
    """Normalise each trace in the dataset by its max absolute amplitude."""
    config = config or {}
    bounds = config.get('bounds', None)
    show_graph = config.get('show_graph', show_graph) # default to kwarg if not in config

    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        if bounds is not None:
            mask = (t >= bounds[0]/_S_TO_PS) & (t <= bounds[1]/_S_TO_PS)
            if not mask.any():
                print(f"Warning: '{filename}' has no samples in normalisation bounds {bounds}, skipping normalisation.")
                continue
        else:
            mask = slice(None)  # all samples
        
        max_amp = np.max(np.abs(y[mask]))
        if max_amp == 0:
            print(f"Warning: '{filename}' has zero max amplitude, skipping normalisation.")
            continue
        norm_y = y / max_amp
        if data_obj.data.shape[1] >= 3:
            norm_err = data_obj.data[:, 2] / max_amp
            data_obj.data = np.column_stack((t, norm_y, norm_err))
        else:
            data_obj.data = np.column_stack((t, norm_y))
        data_obj.processing_dict['normalisation_factor'] = max_amp

        # Apply the same scalar to every individual scan so the per-scan working
        # matrix stays consistent with the normalised averaged trace.
        scan_matrix = _ensure_scan_matrix(data_obj)
        scan_matrix[:, 1:] /= max_amp

        if show_graph:
            plt.plot(t, norm_y, label='{} (normalised)'.format(filename))
            plt.plot(t, y, label='{} (pre-normalisation)'.format(filename), linestyle='--', lw=1, alpha=0.5)
        
    if show_graph:
        plt.legend()
        plt.xlabel('Time (s)')
        plt.title('Normalised Traces')
        plt.show()
    return dataset

def _time_amplitude_array(data_obj) -> np.ndarray:
    """Extract (N, 2) [time_s, mean_amplitude] from a THzData object."""
    return data_obj.data[:, :2].copy()


def _build_data_dict(dataset: DataSet) -> dict:
    """Build {filename: (N,2) array} from all current data objects."""
    return {
        filename: _time_amplitude_array(data_obj)
        for filename, data_obj in dataset.data.items()
    }


def _resolve_segment(data_obj, segment: str):
    """Return the data holder for *segment*, or None to skip this data_obj.

    ``'second_reflection'`` → *data_obj* itself (all object types).
    ``'first_reflection'``  → *data_obj.first_segment* for
    ``THzDataReflection`` only; returns ``None`` for plain ``THzData``.
    """
    if segment == 'first_reflection':
        if not isinstance(data_obj, THzDataReflection):
            return None
        return data_obj.first_segment
    return data_obj


def _build_segment_data_dict(dataset: DataSet, segment: str) -> dict:
    """Build {filename: (N,2) array} for the given segment type.

    Skips data objects that have no holder for the requested segment
    (e.g. plain ``THzData`` when ``segment='first_reflection'``).
    """
    result = {}
    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is not None:
            result[filename] = _time_amplitude_array(holder)
    return result


# ---------------------------------------------------------------------------
# Per-scan working matrix (preserves statistical power through preprocessing)
#
# data_obj.data is the AVERAGED [time_s, mean, stderr] trace that the pipeline
# mutates. To keep every individual acquisition available for segmentation we
# carry a parallel matrix [time_s, scan1, ..., scanN] that the linear
# preprocessing steps (baseline subtract, alignment x-shift, normalise scalar)
# transform IN LOCKSTEP with data. Because all three ops are linear/shared, the
# mean of the scan columns equals data_obj.data[:, 1] at every stage, so the
# averaged pipeline is unchanged while the per-scan scatter survives to the
# segmented .acc files.
# ---------------------------------------------------------------------------
_SCAN_MATRIX_KEY = 'working_scans'


def _ensure_scan_matrix(data_obj) -> np.ndarray:
    """Return the per-scan working matrix ``[time_s, scan1, ..., scanN]``.

    Created lazily from ``raw_data`` (source ps time converted to SI seconds to
    match ``data_obj.data``) on first use and cached in ``processing_dict``.
    Subsequent calls return the same array so in-place transforms by the
    preprocessing steps accumulate. Falls back to the averaged trace as a single
    "scan" if no multi-scan ``raw_data`` is present.
    """
    matrix = data_obj.processing_dict.get(_SCAN_MATRIX_KEY)
    if matrix is not None:
        return matrix
    raw = np.asarray(data_obj.raw_data, dtype=float)
    if raw.ndim == 2 and raw.shape[1] >= 2:
        time_seconds = raw[:, 0] / _S_TO_PS  # ps -> s, matching data_obj.data
        matrix = np.column_stack((time_seconds, raw[:, 1:]))
    else:
        averaged = np.asarray(data_obj.data, dtype=float)
        matrix = np.column_stack((averaged[:, 0], averaged[:, 1]))
    data_obj.processing_dict[_SCAN_MATRIX_KEY] = matrix
    return matrix


def _write_snr_masks(samp_obj, ref_obj, mask_metrics: dict) -> None:
    """Store per-spectrum SNR masks and SNR arrays on sample and reference objects.

    The reference's mask reflects only its own dynamic range (ref_snr_db >= thresh),
    independent of the sample. The sample's mask is just the sample's own DR.
    The intersection (combined + segment-cleaned) lives on the sample as
    'transfer_mask' and is what downstream inversion consumes.
    """
    values = mask_metrics.get('values', {})
    snr_thresh = values.get('snr_thresh_db')
    samp_snr = values.get('samp_snr_db')
    ref_snr = values.get('ref_snr_db')
    if snr_thresh is None or samp_snr is None or ref_snr is None:
        return

    samp_snr_arr = np.asarray(samp_snr)
    ref_snr_arr = np.asarray(ref_snr)

    samp_obj.processing_dict['snr_db'] = samp_snr_arr
    samp_obj.processing_dict['snr_mask'] = samp_snr_arr >= snr_thresh
    if ref_obj is not None:
        ref_obj.processing_dict['snr_db'] = ref_snr_arr
        ref_obj.processing_dict['snr_mask'] = ref_snr_arr >= snr_thresh


def _plot_with_snr_mask(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray | None,
    *,
    color=None,
    label: str | None = None,
    full_alpha: float = 0.9,
    masked_alpha: float = 0.2,
    **plot_kwargs,
):
    """Plot a trace with the trusted region at full alpha and the rest dimmed.

    Strategy: draw the full trace at ``masked_alpha`` (dimmed background), then
    overlay the trusted portion at ``full_alpha``. The legend label attaches to
    the overlay so only the trusted line shows up in the legend.

    If ``mask`` is None or all True, plots a single line at ``full_alpha``.
    If ``mask`` is all False, plots a single dimmed line tagged "(low SNR)".
    """
    if mask is None:
        line, = ax.plot(x, y, color=color, label=label, alpha=full_alpha, **plot_kwargs)
        return line

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.all():
        line, = ax.plot(x, y, color=color, label=label, alpha=full_alpha, **plot_kwargs)
        return line
    if not mask_arr.any():
        dim_label = f"{label} (low SNR)" if label else None
        line, = ax.plot(x, y, color=color, label=dim_label, alpha=masked_alpha, **plot_kwargs)
        return line

    full_line, = ax.plot(x, y, color=color, alpha=masked_alpha, **plot_kwargs)
    actual_color = full_line.get_color()
    y_trusted = np.where(mask_arr, y, np.nan)
    trusted_line, = ax.plot(
        x, y_trusted, color=actual_color, label=label, alpha=full_alpha, **plot_kwargs,
    )
    return trusted_line


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------

def subtract_baseline(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Subtract DC baseline from each trace (removes detector/digitiser offset).

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which segment to process. Call twice to process both.
    """
    config = config or {}
    n_points = int((config.get('baseline', {}) or {}).get('n_points', 10))

    data_dict = _build_segment_data_dict(dataset, segment)
    corrected, metrics = core.subtract_baseline(data_dict, config)

    if show_graph:
        fig, ax = plt.subplots(2, 1, layout='constrained')
        for filename, data in corrected.items():
            ax[1].plot(data[:, 1], label=f'{filename} (corrected)')
            ax[0].plot(data_dict[filename][:, 1], label=f'{filename} (original)',
                       linestyle='--', lw=1, alpha=0.5)
        ax[0].set_title(f'Original Traces — {segment}')
        ax[1].set_title('Baseline-Corrected Traces')
        ax[1].set_xlabel('Time Point Index')
        plt.show()

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None:
            continue
        holder.data = corrected[filename]
        holder.processing_dict['baseline_metrics'] = metrics

        if segment == 'second_reflection':
            # Mirror baseline subtraction onto the per-scan matrix so the
            # per-scan working data stays consistent with the averaged trace.
            # (first_segment is already an averaged trace; no per-scan matrix.)
            scan_matrix = _ensure_scan_matrix(data_obj)
            scan_columns = scan_matrix[:, 1:]
            scan_columns -= scan_columns[:n_points, :].mean(axis=0, keepdims=True)

    return dataset

def global_truncate(
    dataset: DataSet,
    segment: str = 'second_reflection',
    show_graph: bool = False,
) -> DataSet:
    """Truncate all traces of *segment* to their common time range.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which segment to process. Call twice to process both.
    """
    holders = [(fn, _resolve_segment(obj, segment)) for fn, obj in dataset.data.items()]
    holders = [(fn, h) for fn, h in holders if h is not None]

    if not holders:
        print(f"[global_truncate] No data found for segment '{segment}', skipping.")
        return dataset

    min_t = max(float(np.nanmin(h.data[:, 0])) for _, h in holders)
    max_t = min(float(np.nanmax(h.data[:, 0])) for _, h in holders)

    for fn, h in holders:
        mask = (h.data[:, 0] >= min_t) & (h.data[:, 0] <= max_t)
        h.data = h.data[mask, :]
        if show_graph:
            plt.plot(h.data[:, 0] * _S_TO_PS, h.data[:, 1], label=fn)

    print(f"[global_truncate] {segment}: truncated to common time range.")
    if show_graph:
        plt.legend()
        plt.xlabel('Time (ps)')
        plt.title(f'Globally Truncated Traces — {segment}')
        plt.show()
    return dataset

    

def define_alignment_regions(
    dataset: DataSet,
    config: dict,
    segments: tuple = ('second_reflection', 'first_reflection'),
) -> dict:
    """Interactively define peak-search regions for each segment, writing results into *config*.

    For each segment in *segments*, if ``config['align']['auto_range_ps'][segment]``
    is already set (not ``None``), the selector is skipped — headless mode.  Otherwise
    a SpanSelector window opens against a representative trace.

    Parameters
    ----------
    dataset : DataSet
    config : dict
        Modified in-place.  Results land in ``config['align']['auto_range_ps'][segment]``.
    segments : tuple of str
        Segments to define regions for.

    Returns
    -------
    dict
        The modified *config* dict (also mutated in-place).
    """
    align_cfg = config.setdefault('align', {})
    region_cfg = align_cfg.setdefault('auto_range_ps', {})

    for segment in segments:
        if region_cfg.get(segment) is not None:
            print(f"[define_alignment_regions] '{segment}': using preset {region_cfg[segment]} ps.")
            continue

        rep_holder, rep_filename = None, None
        for filename, data_obj in dataset.data.items():
            holder = _resolve_segment(data_obj, segment)
            if holder is not None:
                rep_holder, rep_filename = holder, filename
                break

        if rep_holder is None:
            print(f"[define_alignment_regions] '{segment}': no data found, skipping.")
            continue

        t_ps = rep_holder.data[:, 0] * _S_TO_PS
        y = rep_holder.data[:, 1]
        bounds = _span_select_bounds(
            t_ps, y,
            title=f"Peak region — '{segment}' (file: {rep_filename})",
        )
        region_cfg[segment] = bounds
        print(f"[define_alignment_regions] '{segment}': peak region → {bounds[0]:.2f}–{bounds[1]:.2f} ps.")

    return config


def pre_window_align_peak(
    dataset: DataSet,
    segment: str = 'second_reflection',
    show_graph: bool = False,
    auto_range_ps: tuple | None = None,
    recalibrate: bool = False,
) -> DataSet:
    """Align all traces of *segment* on their main pulse peak.

    Used before windowing so the window function lands at the same T0 distance
    from the edge for every file.

    Parameters
    ----------
    dataset : DataSet
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which segment to align.
    show_graph : bool
    auto_range_ps : tuple[float, float] or None
        ``(t_start_ps, t_end_ps)`` in picoseconds.  Enables headless operation.
        If ``None``, an interactive SpanSelector opens for each trace.
    recalibrate : bool
        If True, prompt interactively to choose a reference file and replace
        all time axes with that file's axis (removes instrumental timing offsets).
    """
    data_dict = _build_segment_data_dict(dataset, segment)

    auto_range_idx = None
    if auto_range_ps is not None:
        first_data = next(iter(data_dict.values()))
        t_s = first_data[:, 0]
        idx_start = int(np.searchsorted(t_s, auto_range_ps[0] * 1e-12))
        idx_end = int(np.searchsorted(t_s, auto_range_ps[1] * 1e-12))
        auto_range_idx = (idx_start, idx_end)

    aligned = core.align_on_peak(data_dict, auto_range=auto_range_idx)

    if recalibrate:
        print("Recalibrating time axis for all files to share same T0, i.e. no instrumental timing offset remains.")
        print(" Select which file's time axis to use:")
        while True:
            for i, filename in enumerate(aligned.keys()):
                print(f"  {i}: {filename}")
            selection = input(f"Enter a number from 0 to {len(aligned)-1}: ")
            try:            
                idx = int(selection)
                if idx < 0 or idx >= len(aligned):
                    print("Must be a valid number from the list.")
                    continue
                selected_filename = list(aligned.keys())[idx]
                break
            except ValueError:
                print("Must be a valid number from the list.")
        print(f"Selected '{selected_filename}' as the T0 reference. Recalibrating all files to share its time axis.")

        ref_data = aligned[selected_filename]
        ref_axis = ref_data[:, 0]

        for filename, data in aligned.items():
            if filename == selected_filename:
                continue
            aligned[filename] = np.column_stack((ref_axis, data[:, 1]))

    if show_graph:
        for filename, data in aligned.items():
            plt.plot(data[:, 1], label=filename)
        plt.legend()
        plt.title(f'Peak-aligned traces — {segment}')
        plt.show()

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is not None:
            holder.data = aligned[filename]

    return dataset


def _pick_peak_manual(t_ps: np.ndarray, y: np.ndarray, title: str) -> float:
    """Open a click-picker window and return the clicked time in ps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ps, y, lw=1)
    ax.set_xlim(float(np.nanmin(t_ps)), float(np.nanmax(t_ps)))
    ax.margins(x=0)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.text(
        0.01, 0.99, "Click on the main pulse peak, then close the window.",
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', alpha=0.2),
    )
    clicked = {}

    def _onclick(event):
        if event.inaxes == ax:
            clicked['t_ps'] = float(event.xdata)
            ax.axvline(clicked['t_ps'], color='tab:red', lw=1.5, linestyle='--',
                       label=f"Peak @ {clicked['t_ps']:.2f} ps")
            ax.legend()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _onclick)
    plt.show()

    if 't_ps' not in clicked:
        raise RuntimeError(f"No peak picked for '{title}'.")
    return clicked['t_ps']


def _center_pulse_trace(
    t: np.ndarray,
    y: np.ndarray,
    peak_mode: str = 'auto',
    taper_ps: float = 0.5,
    picker_title: str = 'Pick main pulse peak',
) -> tuple:
    """Core centering routine — operates on raw arrays, knows nothing about datasets.

    Returns (t_new, y_new, info) where info carries metadata needed by
    callers for logging and by the graph helper for annotation.

    If the pulse is already centred (n_prepend == 0), t_new and y_new are
    copies of the inputs and info['already_centred'] is True.
    """
    dt = float(np.median(np.diff(t)))
    t_ps = t * _S_TO_PS

    if peak_mode == 'auto':
        peak_idx = int(np.argmax(np.abs(y)))
    elif peak_mode == 'manual':
        peak_t_ps = _pick_peak_manual(t_ps, y, picker_title)
        peak_idx = int(np.argmin(np.abs(t_ps - peak_t_ps)))
    else:
        raise ValueError(f"peak_mode must be 'auto' or 'manual', got {peak_mode!r}")

    n_before = peak_idx
    n_after = len(y) - 1 - peak_idx
    n_prepend = max(0, n_after - n_before)

    info = {
        'already_centred': n_prepend == 0,
        'n_prepend': n_prepend,
        'peak_idx_original': peak_idx,
        'peak_idx_new': n_prepend + peak_idx,
        'taper_samples': 0,
        'dt_s': dt,
    }

    if n_prepend == 0:
        return t.copy(), y.copy(), info

    taper_samples = int(round(taper_ps * 1e-12 / dt))
    taper_samples = min(taper_samples, peak_idx)
    info['taper_samples'] = taper_samples

    t_prepend = t[0] - np.arange(n_prepend, 0, -1) * dt
    t_new = np.concatenate([t_prepend, t])
    y_new = np.concatenate([np.zeros(n_prepend), y.copy()])

    if taper_samples > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_samples, endpoint=False)))
        y_new[n_prepend : n_prepend + taper_samples] *= ramp

    return t_new, y_new, info


def _plot_centering_result(filename: str, pre_arr: np.ndarray, new_arr: np.ndarray, info: dict) -> None:
    """Two-panel centering diagnostic: full trace overlay + junction zoom."""
    t_orig_ps = pre_arr[:, 0] * _S_TO_PS
    y_orig = pre_arr[:, 1]
    t_new_ps = new_arr[:, 0] * _S_TO_PS
    y_new = new_arr[:, 1]

    n_prepend = info['n_prepend']
    peak_idx_new = info['peak_idx_new']
    taper_samples = info['taper_samples']
    dt_ps = info['dt_s'] * _S_TO_PS

    pad_end_ps = t_new_ps[n_prepend - 1]
    taper_end_ps = t_new_ps[n_prepend + taper_samples - 1] if taper_samples > 0 else pad_end_ps
    peak_ps = t_new_ps[peak_idx_new]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    fig.suptitle(f"Pulse centering — '{filename}'")

    for ax, (panel_title, xlim) in zip(axes, [
        ('Full trace', (t_new_ps[0], t_new_ps[-1])),
        ('Junction zoom', (t_new_ps[0], peak_ps + 1.0)),
    ]):
        ax.plot(t_orig_ps, y_orig, color='gray', lw=1, linestyle='--', alpha=0.7, label='Original')
        ax.plot(t_new_ps, y_new, color='tab:blue', lw=1.2, label='Centered')
        ax.axvspan(t_new_ps[0], pad_end_ps, alpha=0.15, color='tab:blue',
                   label=f'Prepended pad ({n_prepend} pts, {n_prepend * dt_ps:.2f} ps)')
        if taper_samples > 0:
            ax.axvspan(pad_end_ps, taper_end_ps, alpha=0.30, color='tab:orange',
                       label=f'Taper ({taper_samples * dt_ps:.2f} ps)')
        ax.axvline(peak_ps, color='tab:red', lw=1, linestyle=':', label=f'Peak @ {peak_ps:.2f} ps')
        ax.set_xlim(*xlim)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Amplitude')

        ax.set_ylim(float(np.nanmin(y_new)), float(np.nanmax(y_new)))
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
    # normalise the axis to view the taper region better
    junction_values = y_new[n_prepend : n_prepend + taper_samples] if taper_samples > 0 else []
    junction_values *= 3
    axes[1].set_ylim(float(np.nanmin(junction_values)), float(np.nanmax(junction_values)))



def center_pulse(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Centre the main pulse at the temporal midpoint by pre-padding with zeros.

    Works for any measurement geometry (reflection, transmission) and either
    segment.  Call once per segment for explicit, inspectable processing.

    Parameters
    ----------
    dataset : DataSet
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
    config : dict, optional
        ``config['centering']`` keys:

        ``peak_mode`` : ``'auto'`` (default) or ``'manual'``
        ``taper_ps`` : float, default 0.5
            Duration (ps) of the half-cosine ramp at the pad–signal junction.
    show_graph : bool

    Notes
    -----
    Modifies the segment's data array in-place.  Original trace saved under
    ``processing_dict['pre_centering']``.  Call after ``pre_window_align_peak``
    and before ``window_time``.
    """
    config = config or {}
    centering_cfg = config.get('centering', {})
    peak_mode = centering_cfg.get('peak_mode', 'auto')
    taper_ps = centering_cfg.get('taper_ps', 0.5)

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None:
            continue

        t = holder.data[:, 0]
        y = holder.data[:, 1]
        t_new, y_new, info = _center_pulse_trace(
            t, y, peak_mode=peak_mode, taper_ps=taper_ps,
            picker_title=f"'{filename}' [{segment}] — pick main pulse peak",
        )

        if info['already_centred']:
            print(f"[center_pulse/{segment}] '{filename}': pulse already centred, skipping.")
            continue

        pre_arr = np.column_stack((t, y))
        holder.processing_dict['pre_centering'] = pre_arr
        holder.processing_dict['centering_info'] = info
        holder.data = np.column_stack((t_new, y_new))

        n_prepend = info['n_prepend']
        taper_samples = info['taper_samples']
        dt_ps = info['dt_s'] * _S_TO_PS
        print(
            f"[center_pulse/{segment}] '{filename}': prepended {n_prepend} samples "
            f"({n_prepend * dt_ps:.2f} ps); "
            f"taper {taper_samples} samples ({taper_samples * dt_ps:.2f} ps)."
        )

        if show_graph:
            _plot_centering_result(f"{filename} [{segment}]", pre_arr, holder.data, info)
            plt.show()

    return dataset


def window_time(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Apply a time-domain window to each trace in the dataset.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which data segment to window.  Call once per segment for explicit,
        inspectable processing.
    """
    config = config or {}

    holders = [
        (fn, _resolve_segment(obj, segment))
        for fn, obj in dataset.data.items()
    ]
    holders = [(fn, h) for fn, h in holders if h is not None]

    global_window = None
    for fn, holder in holders:
        t = holder.data[:, 0]
        y = holder.data[:, 1]
        windowed_y, metrics, global_window = core.window_time(t, y, config)

        new_data = np.column_stack((t, windowed_y))
        if holder.data.shape[1] > 2:
            new_data = np.column_stack((new_data, holder.data[:, 2:]))

        holder.processing_dict['pre-window'] = np.column_stack((t, y))
        holder.processing_dict['window_metrics'] = metrics
        holder.data = new_data

    if show_graph:
        fig, axes = plt.subplots(2, 1, layout='constrained')
        fig.suptitle(f'Window applied — {segment}')
        for fn, holder in holders:
            pre = holder.processing_dict.get('pre-window')
            axes[0].plot(pre[:, 1], label=f'{fn} (original)', linestyle='--', lw=1, alpha=0.5)
            axes[1].plot(holder.data[:, 1], label=f'{fn} (windowed)')
        if global_window is not None:
            axes[0].plot(global_window, label='Window Function', alpha=0.5, lw=1)
        axes[0].legend()
        axes[1].legend()
        axes[0].set_title('Original Traces')
        axes[1].set_title('Windowed Traces')
        axes[1].set_xlabel('Time (ps)')
        plt.show()

    return dataset

def _span_select_bounds(t_ps: np.ndarray, y: np.ndarray, title: str) -> tuple:
    """Open a SpanSelector and return the dragged (xmin, xmax) in ps.

    Thin interactive shell for picking gate bounds; for headless/repeat analysis
    supply the bounds directly to segment_reflections instead of opening this
    window.
    """
    from matplotlib.widgets import SpanSelector

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_ps, y, lw=1)
    # Frame the axis to the data extent. This also disables x-autoscale, so the
    # interactive SpanSelector's rectangle patch (initialised near x=0) can no
    # longer stretch the view back to 0.
    ax.set_xlim(float(np.nanmin(t_ps)), float(np.nanmax(t_ps)))
    ax.margins(x=0)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    selected = {}

    def _onselect(xmin, xmax):
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        selected['bounds'] = (float(xmin), float(xmax))
        ax.axvspan(xmin, xmax, alpha=0.2, color='tab:orange')
        fig.canvas.draw_idle()
        print(f"  selected {title}: {xmin:.2f}–{xmax:.2f} ps")

    span = SpanSelector(
        ax, onselect=_onselect, direction='horizontal',
        useblit=True, interactive=True, props=dict(alpha=0.2), minspan=0.0,
    )
    _ = span  # keep alive until window closes
    ax.text(
        0.01, 0.99, f"Drag to select the {title} region, then close the window.",
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', alpha=0.2),
    )
    plt.show()

    if 'bounds' not in selected:
        raise RuntimeError(f"No span selected for '{title}'.")
    return selected['bounds']


def segment_reflections(
    dataset: DataSet,
    *,
    segments: dict | None = None,
    components: tuple = ('first_reflection', 'second_reflection'),
    output_dir: str | None = None,
    bounds_units: str = 'ps',
    show_graph: bool = False,
) -> list[str]:
    """Crop each trace into named reflection components and save them as .acc files.

    A clean splitter, nothing more. For every loaded file it crops the raw
    (multi-scan) trace to each component's time gate — no zero-padding, no
    apodisation — and writes the result as a ``.acc`` file in the same format
    as the source. It does NOT modify the in-memory dataset or run any
    downstream processing: reload the written files and process as normal.

    Components are written into per-name subfolders of ``output_dir`` keeping
    the original filenames, e.g.::

        <output_dir>/first_reflection/<original>.acc
        <output_dir>/second_reflection/<original>.acc

    so pointing a fresh ``DataSet`` at ``second_reflection/`` groups and pairs
    exactly like the originals (no filename suffixes, no pairing ambiguity).
    All files are written, references included.

    Parameters
    ----------
    segments : None | dict
        - ``None``  → interactive SpanSelector per file: drag one span per name
          in ``components`` (a window opens per component).
        - ``{component: (start, stop)}`` → shared bounds for every file.
        - ``{filename: {component: (start, stop)}}`` → per-file bounds.
        Bounds are in ``bounds_units`` (default 'ps').
    components : tuple[str]
        Component names to extract (also the subfolder names), in selection order.
    output_dir : str | None
        Destination root. Defaults to ``<dataset.file_dir>/segmented``.
    bounds_units : {'ps', 's'}
        Units of the supplied/selected bounds. The source time axis is in ps;
        's' bounds are converted to ps before cropping.
    show_graph : bool
        If True, plot each trace with the selected gates shaded for verification.

    Returns
    -------
    list[str]
        Paths of the written .acc files.
    """
    import os
    from acquisition_editor import save_acc

    if bounds_units not in ('ps', 's'):
        raise ValueError("bounds_units must be 'ps' or 's'.")
    to_ps = (lambda v: float(v)) if bounds_units == 'ps' else (lambda v: float(v) * _S_TO_PS)

    base_out = output_dir or dataset.file_dir
    per_file = bool(segments) and all(isinstance(v, dict) for v in segments.values())

    written: list[str] = []
    for filename, data_obj in dataset.data.items():
        # Crop the per-scan working matrix [time_s, scan1, ..., scanN] so every
        # individual acquisition is preserved in the segmented files (statistical
        # power), with the baseline / alignment / normalisation carried through
        # exactly as applied to the averaged trace.
        scan_matrix = np.asarray(_ensure_scan_matrix(data_obj), dtype=float)
        if scan_matrix.shape[0] != np.asarray(data_obj.data).shape[0]:
            print(
                f"Warning: '{filename}' scan matrix ({scan_matrix.shape[0]} rows) "
                f"is out of sync with the averaged trace "
                f"({np.asarray(data_obj.data).shape[0]} rows); a row-count-changing "
                f"step ran before segmentation. Falling back to the averaged trace."
            )
            averaged = np.asarray(data_obj.data, dtype=float)
            scan_matrix = np.column_stack((averaged[:, 0], averaged[:, 1]))
        time_ps = scan_matrix[:, 0] * _S_TO_PS
        scans = scan_matrix[:, 1:]
        mean_y = scans.mean(axis=1)
        scan_headers = [obj.headers for obj in data_obj.data_list]

        if segments is None:
            bounds_ps = {
                name: _span_select_bounds(time_ps, mean_y, title=f"{filename} — {name}")
                for name in components
            }
        else:
            source = segments[filename] if per_file else segments
            bounds_ps = {
                name: (to_ps(source[name][0]), to_ps(source[name][1]))
                for name in components if name in source
            }

        for name, (start, stop) in bounds_ps.items():
            mask = (time_ps >= start) & (time_ps <= stop)
            if np.count_nonzero(mask) < 2:
                raise ValueError(
                    f"Gate '{name}' [{start}, {stop}] ps selects <2 samples of "
                    f"'{filename}'."
                )
            # [time_ps, scan1, ..., scanN] for the gated window — one column per
            # acquisition, so save_acc writes every scan back out.
            cropped = np.column_stack((time_ps[mask], scans[mask, :]))
            dest = os.path.join(base_out, name, filename)
            save_acc(
                {'data': cropped, 'scan_headers': scan_headers, 'header': data_obj.headers},
                dest,
            )
            written.append(dest)

        if show_graph:
            fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
            ax.plot(time_ps, mean_y, color='0.4', lw=1)
            ax.set_xlim(float(np.nanmin(time_ps)), float(np.nanmax(time_ps)))
            for name, (start, stop) in bounds_ps.items():
                ax.axvspan(start, stop, alpha=0.2, label=name)
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Segments — {filename}')
            ax.legend()
            plt.show()

    print(f"Wrote {len(written)} segmented .acc files under '{base_out}'.")
    return written


def align_to_reference(
    dataset: DataSet,
    *,
    ref_type: str = 'reference',
    roi: tuple | None = None,
    max_lag_ps: float | None = None,
    subsample_correction: bool | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Shift each sample in time so it shares its reference's T0 (cross-correlation).

    Calibration step: removes the bulk instrumental timing offset between a
    sample and its reference by cross-correlating their time-domain pulses (with
    sub-sample, parabolic precision) and shifting the SAMPLE onto the reference's
    time axis. The reference is the T0 anchor and is left untouched. The shift is a pure time translation of the sample's time axis, no resampling or interpolation of the Y values.

    Cross-correlation maximises |correlation|, so it aligns correctly even when
    the sample pulse is sign-flipped relative to the reference (e.g. a
    higher-index sample at a window interface). It removes a constant group delay
    only, preserving the sample's genuine (non-linear) reflection phase.

    Run in the processing phase before window_time/FFT, on the traces you intend
    to ratio (e.g. the segmented second reflections).

    Parameters
    ----------
    ref_type : str
        Reference type to align to (passed to ``dataset.get_reference``).
    roi : tuple[float, float] | None
        Optional ``(start_ps, stop_ps)`` restricting the correlation to the
        pulse region. Default None uses the whole trace.
    max_lag_ps : float | None
        Optional cap on the search lag in ps. Default None uses ``align_time``'s
        default (a quarter of the trace length).
    subsample_correction : bool | None
        Override for the module-level ``SUBSAMPLE_TIMING_CORRECTION`` toggle. When
        the effective value is True, only the whole-sample part of the measured
        shift is slid on the time axis; the sub-sample residual is carried to
        ``transfer_function`` and applied there as an exact spectral phase ramp
        (no time-domain interpolation). When False, the residual is dropped — the
        legacy integer-only behaviour. Default None defers to the module toggle.
    show_graph : bool
        If True, plot reference + sample before/after alignment per sample.
    """
    if subsample_correction is None:
        subsample_correction = SUBSAMPLE_TIMING_CORRECTION
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        ref_obj = dataset.get_reference(filename, ref_type=ref_type)
        if ref_obj is None:
            print(f"Warning: no '{ref_type}' reference for '{filename}', skipping alignment.")
            continue

        ref_t = ref_obj.data[:, 0]
        ref_y = ref_obj.data[:, 1]
        samp_t = data_obj.data[:, 0]
        samp_y = data_obj.data[:, 1]

        align_cfg: dict = {}
        if roi is not None:
            align_cfg['roi'] = (roi[0] / _S_TO_PS, roi[1] / _S_TO_PS)
        if max_lag_ps is not None:
            dt = float(np.median(np.diff(ref_t)))
            align_cfg['max_lag_samples'] = int(round((max_lag_ps / _S_TO_PS) / dt))

        _, _, metrics = core.align_time(ref_t, ref_y, samp_t, samp_y, {'align': align_cfg})
        shift = float(metrics['values']['applied_shift_seconds'])

        # Split the measured shift into a whole-sample part and a sub-sample
        # residual. The whole-sample part is slid on the time axis: it is exactly
        # representable on the grid and is picked up by pad_to_common_grid as an
        # integer offset. The residual (|.| <= dt/2) is NOT representable as a grid
        # slide — if left in the axis, pad_to_common_grid would silently round it
        # away, leaving a residual linear phase error in H that biases n. We carry
        # it to transfer_function and apply it there as an exact spectral phase
        # ramp (Fourier shift theorem), so the time-domain Y is never resampled.
        dt = float(np.median(np.diff(samp_t)))
        integer_samples = int(round(shift / dt))
        integer_shift = integer_samples * dt
        subsample_residual = shift - integer_shift
        applied_residual = subsample_residual if subsample_correction else 0.0

        # Y values are unchanged — only the time axis slides, by a whole number of
        # samples. No resampling, no interpolation, no zero-padding here.
        data_obj.processing_dict['pre_align'] = data_obj.data.copy()
        data_obj.processing_dict['align_metrics'] = metrics
        data_obj.processing_dict['subsample_shift_seconds'] = applied_residual
        data_obj.processing_dict['subsample_shift_measured_seconds'] = subsample_residual
        data_obj.processing_dict['subsample_correction_enabled'] = bool(subsample_correction)
        shifted_data = data_obj.data.copy()
        shifted_data[:, 0] = samp_t + integer_shift
        data_obj.data = shifted_data

        # Slide the per-scan working matrix by the same whole-sample shift so the
        # individual acquisitions share the sample's aligned time axis.
        scan_matrix = _ensure_scan_matrix(data_obj)
        scan_matrix[:, 0] = scan_matrix[:, 0] + integer_shift

        corr = metrics['values'].get('corr_peak', float('nan'))
        state = 'ON' if subsample_correction else 'OFF'
        print(
            f"Aligned '{filename}' to '{ref_obj.filename}': "
            f"shift {shift * _S_TO_PS:+.4f} ps = "
            f"{integer_samples:+d} samp ({integer_shift * _S_TO_PS:+.4f} ps grid) + "
            f"{subsample_residual * _S_TO_PS:+.4f} ps sub-sample "
            f"[correction {state}] (corr {corr:.3f})"
        )

    if show_graph:
        for filename, data_obj in dataset.data.items():
            if dataset.data.is_reference(filename):
                continue
            pre = data_obj.processing_dict.get('pre_align')
            ref_obj = dataset.get_reference(filename, ref_type=ref_type)
            fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
            if ref_obj is not None:
                ax.plot(ref_obj.data[:, 0] * _S_TO_PS, ref_obj.data[:, 1],
                        color='0.5', lw=1, label='reference')
            if pre is not None:
                ax.plot(pre[:, 0] * _S_TO_PS, pre[:, 1], '--', lw=1, alpha=0.7,
                        label='sample (before)')            
                ax.plot(data_obj.data[:, 0] * _S_TO_PS, data_obj.data[:, 1], lw=1.4,
                    label='sample (aligned)')
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Align to reference — {filename}')
            ax.legend()
            plt.show()

    return dataset


def plot_current(dataset: DataSet, *, error_style: str = 'shaded') -> None:
    """Plot the current time-domain traces for all files in the dataset.

    Parameters
    ----------
    error_style : {'shaded', 'bars', 'none'}
        Visualisation of the per-acquisition std error (column 2 of ``data``):
        - 'shaded' (default): translucent fill_between band of ±1 std error.
        - 'bars': sparse errorbars (~50 across the axis to keep it readable).
        - 'none': line only, no uncertainty shown.
    """
    import matplotlib.pyplot as plt

    if error_style not in ('shaded', 'bars', 'none'):
        raise ValueError(
            f"error_style must be 'shaded', 'bars', or 'none', got {error_style!r}."
        )

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        t_ps = t * _S_TO_PS
        has_err = data_obj.data.shape[1] > 2 and error_style != 'none'

        if error_style == 'bars' and has_err:
            err = data_obj.data[:, 2]
            errorevery = max(1, t_ps.size // 50)
            ax.errorbar(
                t_ps, y, yerr=err, label=filename,
                errorevery=errorevery, capsize=0, lw=1.0, alpha=0.9,
            )
        else:
            line, = ax.plot(t_ps, y, label=filename)
            if has_err:
                err = data_obj.data[:, 2]
                ax.fill_between(
                    t_ps, y - err, y + err,
                    alpha=0.25, color=line.get_color(), linewidth=0,
                )

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Current Time-Domain Traces')
    ax.legend()
    plt.tight_layout()
    plt.show()


def zero_pad(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Zero-pad traces of *segment* onto a common time grid.

    Accepts either ``config['pad']['extend_factor']`` (relative, default 1.0) or
    ``config['pad']['n_samples']`` (absolute target length).  ``n_samples`` takes
    precedence when both are present.  A ``ValueError`` is raised if ``n_samples``
    is smaller than the longest trace after grid alignment, which would truncate data.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which data segment to pad.  Call once per segment.  Using ``n_samples``
        (absolute) for both segments ensures they share the same FFT frequency grid.
    """
    config = config or {}
    pad_cfg = config.get('pad', {})

    data_dict = _build_segment_data_dict(dataset, segment)
    if not data_dict:
        return dataset

    t_common, padded_dict, metrics = core.pad_to_common_grid(data_dict)

    n_samples = pad_cfg.get('n_samples', None)
    if n_samples is not None:
        n_common = len(t_common)
        if n_samples < n_common:
            raise ValueError(
                f"zero_pad: n_samples={n_samples} is smaller than the longest trace "
                f"({n_common} samples) — this would truncate data, not pad it. "
                f"Use n_samples >= {n_common}."
            )
        # Build extended arrays to EXACTLY n_samples points, bypassing extend_grid's
        # integer rounding so every file in every segment lands on the same grid.
        dt = float(np.median(np.diff(t_common)))
        t_extended = t_common[0] + np.arange(n_samples) * dt
        extended_dict = {
            name: np.concatenate([arr, np.zeros(n_samples - n_common)])
            for name, arr in padded_dict.items()
        }
        metrics = {}
    else:
        extend_factor = pad_cfg.get('extend_factor', 1.0)
        t_extended, extended_dict, metrics = core.extend_grid(t_common, padded_dict, extend_factor)

    if show_graph:
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        fig.suptitle(f'Zero-padded — {segment}')
        for filename, data_obj in dataset.data.items():
            holder = _resolve_segment(data_obj, segment)
            if holder is None or filename not in extended_dict:
                continue
            pre = holder.data
            ax.plot(pre[:, 0] * _S_TO_PS, pre[:, 1], label=f'{filename} (original)', linestyle='--', lw=1, alpha=0.5)
            ax.plot(t_extended * _S_TO_PS, extended_dict[filename], label=f'{filename} (padded)')
        ax.set_xlabel('Time (ps)')
        ax.legend()
        plt.show()

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None or filename not in extended_dict:
            continue
        holder.data = np.column_stack((t_extended, extended_dict[filename]))
        holder.processing_dict['pad_metrics'] = metrics

    return dataset


def fft_spectrum(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    n_fft: int | None = None,
) -> DataSet:
    """Compute the FFT for each trace in *segment* and switch to frequency domain.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which data segment to transform.  Call once per segment.

        For ``'first_reflection'`` an absolute-time phase reference
        ``exp(-2πif·t₀)`` is applied so both segment spectra share the
        experiment time origin, enabling W = Y₂/Y₁ to encode the true
        inter-pulse delay.
    n_fft : int or None
        Optional explicit FFT length.  When provided, ``np.fft.rfft`` is
        called with ``n=n_fft``, zero-padding or truncating the time-domain
        trace as needed.  Use this to force the first- and second-reflection
        spectra onto the same frequency grid when the two segments were
        zero-padded to different lengths (e.g. when ``extend_factor`` rather
        than ``n_samples`` was used in ``zero_pad``).
    """
    config = config or {}

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None:
            continue

        t = holder.data[:, 0]
        y = holder.data[:, 1]

        holder.processing_dict['time_domain_prefft'] = np.column_stack((t, y))

        if n_fft is not None:
            fft_cfg = config.get('fft', {})
            norm = str(fft_cfg.get('norm', 'backward'))
            amplitude_scale = float(fft_cfg.get('amplitude_scale', 1.0))
            dt = float(np.median(np.diff(t)))
            raw = np.fft.rfft(y, n=n_fft, norm=norm) * amplitude_scale
            freq = np.fft.rfftfreq(n_fft, dt)
            spectrum = raw
            metrics = {}
        else:
            freq, spectrum, metrics = core.fft_spectrum(t, y, config)

        if segment == 'first_reflection':
            # Absolute-time phase reference: encodes the true inter-pulse delay
            # in W = Y₂/Y₁ without time-domain resampling.
            spectrum = spectrum * np.exp(-2j * np.pi * freq * t[0])

        holder.processing_dict['fft_freq'] = freq
        holder.processing_dict['fft_spectrum'] = spectrum
        holder.processing_dict['fft_metrics'] = metrics

        holder.data = np.column_stack((
            freq,
            np.abs(spectrum),
            np.angle(spectrum),
        ))
        holder.current_state = 'frequency_domain'

    return dataset


def _absolute_time_spectrum(time_s: np.ndarray, amplitude: np.ndarray, n_fft: int) -> tuple:
    """Hann-windowed rfft of a segment, phase-referenced to ABSOLUTE time.

    The rfft's implicit time origin is the first sample; multiplying by
    ``exp(-i*2*pi*f*t0)`` refers the phase back to the experiment time axis so
    spectra of segments cropped from different gates share a common origin.
    The mean is subtracted first (segments carry no meaningful DC).
    """
    dt = float(np.median(np.diff(time_s)))
    windowed = (amplitude - np.mean(amplitude)) * np.hanning(amplitude.size)
    spectrum = np.fft.rfft(windowed, n=n_fft)
    freq = np.fft.rfftfreq(n_fft, dt)
    return freq, spectrum * np.exp(-2j * np.pi * freq * time_s[0])


def _first_reflection_spectrum(filepath: str, freq: np.ndarray) -> np.ndarray:
    """Spectrum of a segmented first-reflection .acc on the pipeline's grid.

    Loads the file, averages its scans, and computes the absolute-time
    referenced spectrum with the FFT length chosen so the frequency grid
    matches ``freq`` (the grid of the already-FFT'd second-reflection data).
    """
    from dataset_core.io.loaders.acc_loader import ACCLoader

    thz_obj = ACCLoader(filepath).load()
    averaged = np.asarray(thz_obj.data, dtype=float)  # [time_s, mean, stderr]
    time_s = averaged[:, 0]
    amplitude = averaged[:, 1]

    dt = float(np.median(np.diff(time_s)))
    df = float(freq[1] - freq[0])
    n_fft = int(round(1.0 / (df * dt)))
    grid, spectrum = _absolute_time_spectrum(time_s, amplitude, n_fft)
    if grid.size != freq.size or not np.allclose(grid, freq, rtol=0, atol=df * 1e-6):
        raise ValueError(
            f"First-reflection grid of '{filepath}' (dt {dt * _S_TO_PS:.4f} ps, "
            f"{grid.size} bins) does not match the pipeline FFT grid "
            f"({freq.size} bins, df {df * _HZ_TO_THZ * 1e3:.3f} GHz). The first- "
            f"and second-reflection segments must come from the same acquisitions."
        )
    return spectrum


def transfer_function(dataset: DataSet, config: dict | None = None, ref_type: str = 'substrate') -> DataSet:
    """Compute H(f) = Y_sample / Y_reference for each sample-reference pair.

    Also applies an SNR-based trusted-band mask by default (intersection of
    reference and sample dynamic range, with short-segment cleanup). Disable
    via ``config['transfer']['apply_snr_mask'] = False`` to fall back to the
    permissive finite-only mask.

    Per-spectrum SNR masks are stored on both sample and reference objects as
    ``processing_dict['snr_mask']`` for use by frequency-domain visualizations.

    Front-pulse self-referencing (window-coupled reflection)
    --------------------------------------------------------
    ``config['transfer']['self_reference'] = True`` multiplies H by the
    front-pulse drift correction ``C = Y1_ref / Y1_sample`` built from the
    sibling ``first_reflection`` segment folder, giving

        H_new = (Y2_s / Y1_s) / (Y2_ref / Y1_ref)

    — a ratio of intra-trace ratios in which source-spectrum, detector and
    mount-to-mount alignment drift cancel (each trace is referenced to its own
    front-face reflection, which never sees the sample). Validated on CNT-13/D:
    mount drift of 7-10% rms is replaced by a ~1.6% prediction noise floor
    (see ANALYSIS_NOTES §11). Requirements:

    - the dataset directory is a ``second_reflection`` segment folder whose
      sibling ``first_reflection`` folder holds the SAME filenames (the
      ``segment_reflections`` layout); override the location with
      ``config['transfer']['first_reflection_dir']``;
    - both gates must treat instrument echoes (GaP) consistently — either
      both inside or both outside the gate — so the echo factor cancels in W.

    The sub-sample timing ramp is skipped in this mode: the front-pulse
    spectra already carry the true relative timing, so applying the ramp as
    well would double-count the residual.
    """
    import os

    config = config or {}
    transfer_cfg = config.get('transfer', {})
    apply_snr_mask = transfer_cfg.get('apply_snr_mask', True)
    self_reference = transfer_cfg.get('self_reference', False)

    first_reflection_dir = None
    first_spectra_cache: dict = {}
    if self_reference:
        # Datasets loaded from the root dir (THzDataReflection objects) carry their
        # first-segment spectra directly — no file path needed for those.
        # The path is only required for the legacy mode (pointing at second_reflection/).
        explicit_dir = transfer_cfg.get('first_reflection_dir')
        if explicit_dir:
            first_reflection_dir = explicit_dir
        else:
            # New root layout: first_reflection/ is a subdir of dataset.file_dir
            candidate_subdir = os.path.join(dataset.file_dir, 'first_reflection')
            # Legacy layout: first_reflection/ is a sibling of second_reflection/
            candidate_sibling = os.path.join(os.path.dirname(dataset.file_dir), 'first_reflection')
            if os.path.isdir(candidate_subdir):
                first_reflection_dir = candidate_subdir
            elif os.path.isdir(candidate_sibling):
                first_reflection_dir = candidate_sibling

    def _get_first_spectrum(data_object, freq: np.ndarray) -> np.ndarray:
        """Retrieve or compute the first-reflection spectrum for a data object.

        Prefers ``first_segment.processing_dict['fft_spectrum']`` when the object
        is a ``THzDataReflection`` (already computed by the pipeline). Falls back
        to loading from disk via the legacy ``_first_reflection_spectrum`` path.
        """
        if isinstance(data_object, THzDataReflection):
            stored = data_object.first_segment.processing_dict.get('fft_spectrum')
            if stored is not None:
                return stored
        fname = data_object.filename
        if fname not in first_spectra_cache:
            if first_reflection_dir is None or not os.path.isdir(first_reflection_dir):
                raise FileNotFoundError(
                    f"Self-referencing needs the first-reflection segment folder, but "
                    f"none was found and '{fname}' has no pre-computed first segment. "
                    f"Either load the dataset from its root directory (which contains "
                    f"first_reflection/ and second_reflection/ subdirs) or set "
                    f"config['transfer']['first_reflection_dir'] explicitly."
                )
            filepath = os.path.join(first_reflection_dir, fname)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Self-referencing: no first-reflection file for '{fname}' "
                    f"in '{first_reflection_dir}'."
                )
            first_spectra_cache[fname] = _first_reflection_spectrum(filepath, freq)
        return first_spectra_cache[fname]

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        ref_obj = dataset.get_reference(filename, ref_type=ref_type)
        if ref_obj is None:
            print(f"Warning: no reference found for '{filename}', skipping transfer function.")
            continue

        freq = data_obj.processing_dict['fft_freq']
        Y_samp = data_obj.processing_dict['fft_spectrum']
        Y_ref = ref_obj.processing_dict['fft_spectrum']

        H, finite_mask, tf_metrics = core.transfer_function(freq, Y_samp, Y_ref, config)

        # Apply the sub-sample timing residual left over by align_to_reference, as
        # an exact spectral phase ramp (Fourier shift theorem). This completes the
        # T0 alignment to sub-sample precision without ever interpolating the
        # time-domain trace. It is a no-op (residual = 0) when alignment was not
        # run, or when the SUBSAMPLE_TIMING_CORRECTION toggle was off.
        subsample_shift = data_obj.processing_dict.get('subsample_shift_seconds', 0.0)
        if self_reference and subsample_shift:
            print(
                f"[sub-sample] '{filename}': ramp for {subsample_shift * _S_TO_PS:+.4f} ps "
                f"skipped — self-referencing carries the front-pulse timing itself."
            )
        elif subsample_shift:
            H = H * core.phase_ramp(freq, subsample_shift)
            print(
                f"[sub-sample] '{filename}': applied spectral phase ramp for "
                f"{subsample_shift * _S_TO_PS:+.4f} ps residual timing."
            )

        front_samp = front_ref = None
        if self_reference:
            front_samp = _get_first_spectrum(data_obj, freq)
            front_ref = _get_first_spectrum(ref_obj, freq)
            with np.errstate(divide='ignore', invalid='ignore'):
                selfref_correction = front_ref / front_samp
            H = H * selfref_correction
            data_obj.processing_dict['selfref_correction'] = selfref_correction
            band = np.isfinite(selfref_correction)
            print(
                f"[self-ref] '{filename}': applied front-pulse correction "
                f"(|C| median {np.nanmedian(np.abs(selfref_correction[band])):.3f})."
            )

        data_obj.processing_dict['transfer_H'] = H
        data_obj.processing_dict['transfer_metrics'] = tf_metrics
        data_obj.reference_filename = ref_obj.filename

        if apply_snr_mask:
            try:
                snr_mask_combined, mask_metrics = core.trusted_band_mask(
                    freq, Y_ref, Y_samp, H, config,
                )
                if self_reference:
                    # The correction divides by the front-pulse spectra, so bins
                    # where THEY are noise must be excluded too.
                    front_mask, _ = core.trusted_band_mask(
                        freq, front_ref, front_samp, H, config,
                    )
                    snr_mask_combined = snr_mask_combined & front_mask
                data_obj.processing_dict['transfer_mask'] = snr_mask_combined
                data_obj.processing_dict['mask_metrics'] = mask_metrics
                _write_snr_masks(data_obj, ref_obj, mask_metrics)
            except Exception as exc:
                print(
                    f"Warning: SNR mask failed for '{filename}' ({exc}); "
                    f"falling back to finite-only mask."
                )
                data_obj.processing_dict['transfer_mask'] = finite_mask
        else:
            data_obj.processing_dict['transfer_mask'] = finite_mask

        data_obj.data = np.column_stack((
            freq,
            np.abs(H),
            np.angle(H),
        ))

    return dataset


def trusted_band_mask(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Recompute the SNR-based trusted-band mask with explicit config.

    ``transfer_function`` now applies this by default, so this standalone is
    only needed when overriding the threshold or other mask parameters after
    the fact. It overwrites the existing 'transfer_mask' and refreshes the
    per-spectrum 'snr_mask' arrays on both sample and reference.
    """
    config = config or {}

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        H = data_obj.processing_dict.get('transfer_H')
        if H is None:
            continue

        ref_obj = dataset.get_reference(filename, ref_type='substrate')
        if ref_obj is None:
            continue

        freq = data_obj.processing_dict['fft_freq']
        Y_samp = data_obj.processing_dict['fft_spectrum']
        Y_ref = ref_obj.processing_dict['fft_spectrum']

        mask, metrics = core.trusted_band_mask(freq, Y_ref, Y_samp, H, config)

        data_obj.processing_dict['transfer_mask'] = mask
        data_obj.processing_dict['mask_metrics'] = metrics
        _write_snr_masks(data_obj, ref_obj, metrics)

    return dataset


def invert_nk(dataset: DataSet, thickness_m: float, config: dict | None = None) -> DataSet:
    """Extract refractive index n and extinction coefficient k for each sample."""
    config = config or {}

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        H = data_obj.processing_dict.get('transfer_H')
        mask = data_obj.processing_dict.get('transfer_mask')
        if H is None or mask is None:
            print(f"Warning: no transfer function for '{filename}', skipping inversion.")
            continue

        freq = data_obj.processing_dict['fft_freq']
        n, k, metrics = core.invert_nk(freq, H, thickness_m, mask, config)

        data_obj.processing_dict['n'] = n
        data_obj.processing_dict['k'] = k
        data_obj.processing_dict['invert_metrics'] = metrics

        data_obj.data = np.column_stack((freq, n, k))

    return dataset


def _resolve_reflection_geometry(
    geometry: str,
    theta_external_rad: float,
    r_reference: complex,
    n_window: complex | np.ndarray,
) -> tuple:
    """Resolve (n_incident, theta_internal_rad, r_reference_value) for a geometry.

    Single source of truth for the 'gold' vs 'window' reflection model, shared by
    ``invert_nk_reflection`` and ``sweep_time_shift`` so the two never diverge.
    """
    if geometry == 'gold':
        return 1.0, theta_external_rad, r_reference
    if geometry == 'window':
        # Theta is a single angle — use the scalar mean of n_window for Snell's law.
        # r_reference is computed with the full (possibly per-frequency) n_window array.
        n_window_scalar = float(np.real(np.mean(np.atleast_1d(n_window))))
        theta_internal_rad = float(
            np.real(core.snell_refracted_angle(theta_external_rad, 1.0, n_window_scalar))
        )
        r_reference_value = core.fresnel_reflection_s(n_window, 1.0, theta_internal_rad)
        return n_window, theta_internal_rad, r_reference_value
    raise ValueError(f"Unknown geometry '{geometry}'. Use 'gold' or 'window'.")


def invert_nk_reflection(
    dataset: DataSet,
    *,
    geometry: str = 'gold',
    theta_deg: float = 0.0,
    polarization: str = 's',
    r_reference: complex = -1.0 + 0.0j,
    n_window: complex | np.ndarray = 1.95,
    config: dict | None = None,
) -> DataSet:
    """Reflection-mode n,k for each sample (single-interface, semi-infinite).

    Pulls the measured ratio H = Y_samp / Y_ref from
    ``processing_dict['transfer_H']`` (computed by ``transfer_function`` on the
    gated second-reflection spectra) and converts it to a true sample
    reflection coefficient, then inverts with the Fresnel closed form.

    Geometry selects the incident medium, the angle handling, and the reference
    model:

    - ``'gold'`` (external reflection in air): the sample is a flat surface in
      air referenced to a gold mirror. ``r_sample = r_reference * H`` with
      ``r_reference = -1`` for an ideal mirror, incident medium = air,
      incidence angle = ``theta_deg``.
    - ``'window'`` (internal reflection through a window): the wave reflects at
      a window/sample interface accessed through a window of index
      ``n_window`` (e.g. SiO2 ~1.95). ``theta_deg`` is the EXTERNAL angle; the
      internal angle is found by Snell's law, the reference is the computed
      window→air Fresnel coefficient r_{window→air}, and the incident medium is
      the window. Because H = r_{window→sample}/r_{window→air} (the SiO2-only
      second reflection cancels the window path), ``r_sample = r_{window→air} * H``
      recovers the true window→sample reflection.

    Reflection measurements are phase-sensitive — run
    ``phase_correction(dataset, source='transfer')`` first to null any residual
    timing offset before calling this function.

    Parameters
    ----------
    geometry : {'gold', 'window'}
        Reference/incidence model (see above).
    theta_deg : float
        Angle of incidence in degrees. For 'gold' this is the in-air incidence
        angle; for 'window' it is the EXTERNAL angle before refraction.
    polarization : {'s', 'p'}
        Only 's' implemented for now; 'p' raises NotImplementedError.
    r_reference : complex
        Reference reflection coefficient for the 'gold' geometry (-1 = mirror).
        Ignored for 'window' (computed from n_window).
    n_window : complex or np.ndarray
        Window refractive index for the 'window' geometry (scalar, or a
        per-frequency n_SiO2(f) array). Ignored for 'gold'.
    """
    config = config or {}
    theta_external_rad = np.deg2rad(theta_deg)

    n_incident, theta_internal_rad, r_reference_value = _resolve_reflection_geometry(
        geometry, theta_external_rad, r_reference, n_window,
    )

    print("---- Angles ----")
    print(f"External angle = {np.rad2deg(theta_external_rad):.2f} deg")
    print(f"Internal angle = {np.rad2deg(theta_internal_rad):.2f} deg")


    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        H = data_obj.processing_dict.get('transfer_H')
        mask = data_obj.processing_dict.get('transfer_mask')
        if H is None or mask is None:
            print(f"Warning: no transfer function for '{filename}', skipping reflection inversion.")
            continue

        freq = data_obj.processing_dict['fft_freq']
        r_sample = r_reference_value * np.asarray(H)

        n, k, metrics = core.invert_nk_reflection(
            freq, r_sample, mask, config,
            theta_rad=theta_internal_rad, polarization=polarization,
            n_incident=n_incident,
        )

        data_obj.processing_dict['reflection_r'] = r_sample
        data_obj.processing_dict['reflection_geometry'] = geometry
        data_obj.processing_dict['theta_internal_rad'] = theta_internal_rad
        data_obj.processing_dict['r_reference'] = r_reference_value
        data_obj.processing_dict['n'] = n
        data_obj.processing_dict['k'] = k
        data_obj.processing_dict['invert_metrics'] = metrics

        data_obj.data = np.column_stack((freq, n, k))

    return dataset


def characterise_window(
    first_reflection_path: str,
    second_reflection_path: str,
    *,
    thickness_m: float,
    theta_deg: float = 45.0,
    band_thz: tuple = (0.25, 3.5),
    n_initial: float = 1.95,
    fft_length: int = 8192,
    config: dict | None = None,
    show_graph: bool = False,
) -> dict:
    """Window optical constants from a single bare-window two-pulse trace.

    Takes the two segmented components of ONE bare-window acquisition (the
    ``segment_reflections`` output for the same original file) and:

    1. computes the intra-trace ratio ``W = Y_second / Y_first`` — the window
       transfer function, in which the source spectrum, detector response and
       shared air path cancel;
    2. measures the inter-pulse envelope delay (anchors the phase branch);
    3. inverts W for the complex window index n(f) - i*k(f) via
       ``thz_core.invert_window_index`` (s-pol, plane-parallel window model).

    The measured W is also the quantity used by front-pulse self-referencing
    (``transfer_function`` with ``self_reference=True``): the empirical W
    predicts the bare-window second reflection from any measured first
    reflection. Use THIS function to characterise/monitor the window; use the
    transfer-function option to apply the correction in the pipeline.

    Parameters
    ----------
    first_reflection_path, second_reflection_path : str
        Segmented .acc files of the same bare-window acquisition.
    thickness_m : float
        Window thickness in metres (e.g. 0.9e-3). The extracted n scales
        inversely with it: a 1% thickness error is ~1% systematic on n.
    theta_deg : float
        EXTERNAL angle of incidence in degrees.
    band_thz : tuple[float, float]
        Trusted analysis band in THz for the inversion.
    n_initial : float
        Starting index for the iterative inversion.
    fft_length : int
        Zero-padded FFT length applied to both segments (sets the grid).
    config : dict, optional
        Extra ``window`` config keys forwarded to ``invert_window_index``.
    show_graph : bool
        Plot |W|, the extracted n(f) and k(f).

    Returns
    -------
    dict
        ``freq`` (Hz), ``W`` (complex), ``mask``, ``n``, ``k``,
        ``delay_s`` (measured inter-pulse delay), and ``metrics``.
    """
    from dataset_core.io.loaders.acc_loader import ACCLoader

    def _load_segment(filepath):
        averaged = np.asarray(ACCLoader(filepath).load().data, dtype=float)
        return averaged[:, 0], averaged[:, 1]

    time_first, amp_first = _load_segment(first_reflection_path)
    time_second, amp_second = _load_segment(second_reflection_path)

    dt_first = float(np.median(np.diff(time_first)))
    dt_second = float(np.median(np.diff(time_second)))
    if abs(dt_first - dt_second) > 1e-3 * dt_first:
        raise ValueError(
            f"Segment sampling intervals differ ({dt_first * _S_TO_PS:.4f} vs "
            f"{dt_second * _S_TO_PS:.4f} ps) — both segments must come from the "
            f"same acquisition."
        )

    freq, spectrum_first = _absolute_time_spectrum(time_first, amp_first, fft_length)
    _, spectrum_second = _absolute_time_spectrum(time_second, amp_second, fft_length)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_measured = spectrum_second / spectrum_first

    delay_s = (core.envelope_peak_time(time_second, amp_second)
               - core.envelope_peak_time(time_first, amp_first))

    f_thz = freq * _HZ_TO_THZ
    mask = (
        (f_thz >= band_thz[0]) & (f_thz <= band_thz[1])
        & np.isfinite(w_measured)
    )

    window_config = {'window': {'n_initial': n_initial}}
    if config:
        window_config['window'].update(config.get('window', {}))

    theta_external_rad = np.deg2rad(theta_deg)
    n_window, k_window, metrics = core.invert_window_index(
        freq, w_measured, mask,
        thickness_m=thickness_m,
        theta_external_rad=theta_external_rad,
        delay_estimate_s=delay_s,
        config=window_config,
    )

    cos_internal = float(np.real(
        np.cos(core.snell_refracted_angle(theta_external_rad, 1.0, n_initial))
    ))
    nominal_delay_s = 2.0 * n_initial * thickness_m * cos_internal / 299_792_458.0
    print("---- Window characterisation ----")
    print(f"Files: {first_reflection_path}")
    print(f"       {second_reflection_path}")
    print(f"Inter-pulse delay: measured {delay_s * _S_TO_PS:.3f} ps "
          f"(nominal n={n_initial}, d={thickness_m * 1e3:.3f} mm: "
          f"{nominal_delay_s * _S_TO_PS:.3f} ps)")
    print(f"n_window: mean {metrics['values']['n_mean']:.4f} "
          f"+/- {metrics['values']['n_std']:.4f}, "
          f"k mean {metrics['values']['k_mean']:.4f} "
          f"({metrics['values']['trusted_bins']} bins, "
          f"converged={metrics['values']['converged']})")

    if show_graph:
        fig, (ax_w, ax_n, ax_k) = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True, layout='constrained'
        )
        ax_w.plot(f_thz[mask], np.abs(w_measured[mask]))
        ax_w.set_ylabel('|W|')
        ax_w.set_title(
            f'Window transfer W = Y2/Y1 and extracted index '
            f'(d = {thickness_m * 1e3:.3f} mm, {theta_deg:.0f}° external)'
        )
        ax_n.plot(f_thz[mask], n_window[mask])
        ax_n.axhline(n_initial, color='0.5', lw=0.8, ls=':',
                     label=f'n_initial {n_initial}')
        ax_n.set_ylabel('n')
        ax_n.legend()
        ax_k.plot(f_thz[mask], k_window[mask])
        ax_k.set_ylabel('k')
        ax_k.set_xlabel('Frequency (THz)')
        plt.show()

    return {
        'freq': freq,
        'W': w_measured,
        'mask': mask,
        'n': n_window,
        'k': k_window,
        'delay_s': delay_s,
        'metrics': metrics,
    }


def invert_nk_grid(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Extract n and k via brute-force 2D grid search for each sample.

    Geometry is set via config["invert_grid"]["geometry"]:
      - "free_standing"       : sample in air, uses pre-computed transfer_H
      - "substrate_only"      : substrate vs air, computes H internally from FFT spectra
      - "substrate_sandwich"  : sample on substrate, reads n_sub/k_sub from matched substrate file

    For "substrate_sandwich", run "substrate_only" first so that substrate n,k are available.
    """
    config = config or {}
    geometry = config.get("invert_grid", {}).get("geometry", "free_standing")

    if geometry == "free_standing":
        _grid_invert_free_standing(dataset, config)
    elif geometry == "substrate_only":
        _grid_invert_substrate_only(dataset, config)
    elif geometry == "substrate_sandwich":
        _grid_invert_substrate_sandwich(dataset, config)
    else:
        raise ValueError(
            f"Unknown geometry '{geometry}'. Must be 'free_standing', 'substrate_only', or 'substrate_sandwich'."
        )

    return dataset


def _get_transfer_data(data_obj, filename):
    """Return (freq, H, mask) from processing_dict, or (None, None, None) with a warning."""
    freq = data_obj.processing_dict.get('fft_freq')
    H = data_obj.processing_dict.get('transfer_H')
    mask = data_obj.processing_dict.get('transfer_mask')
    if H is None or mask is None:
        print(f"Warning: no transfer function for '{filename}', skipping inversion.")
        return None, None, None
    return freq, H, mask


def _get_fft_data(data_obj, filename):
    """Return (freq, spectrum) from processing_dict, or (None, None) with a warning."""
    freq = data_obj.processing_dict.get('fft_freq')
    spectrum = data_obj.processing_dict.get('fft_spectrum')
    if freq is None or spectrum is None:
        print(f"Warning: no FFT spectrum for '{filename}', skipping.")
        return None, None
    return freq, spectrum


def _store_nk_grid(data_obj, freq, n, k, metrics):
    """Write n, k, and grid-search metrics back into processing_dict."""
    data_obj.processing_dict['n'] = n
    data_obj.processing_dict['k'] = k
    data_obj.processing_dict['invert_metrics'] = metrics
    data_obj.data = np.column_stack((freq, n, k))


def _grid_invert_free_standing(dataset, config):
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        freq, H, mask = _get_transfer_data(data_obj, filename)
        if freq is None:
            continue
        n, k, metrics = core.invert_nk_grid(freq, H, mask, config)
        _store_nk_grid(data_obj, freq, n, k, metrics)


def _grid_invert_substrate_only(dataset, config):
    for filename, data_obj in dataset.data.items():
        file_item = dataset.grouping.file_items.get(filename)
        if file_item is None or file_item.data_type != 'substrate':
            continue

        freq, Y_sub = _get_fft_data(data_obj, filename)
        if freq is None:
            continue

        air_ref_filename = getattr(file_item, 'air_reference', None)
        if air_ref_filename is None:
            print(f"Warning: no air reference for substrate '{filename}', skipping.")
            continue

        air_obj = dataset.data.get(air_ref_filename)
        if air_obj is None:
            print(f"Warning: air reference '{air_ref_filename}' not loaded, skipping '{filename}'.")
            continue

        _, Y_air = _get_fft_data(air_obj, air_ref_filename)
        if Y_air is None:
            continue

        H, _valid_mask, _tf_metrics = core.transfer_function(freq, Y_sub, Y_air, config)
        mask, mask_metrics = core.trusted_band_mask(freq, Y_air, Y_sub, H, config)

        data_obj.processing_dict['transfer_H'] = H
        data_obj.processing_dict['transfer_mask'] = mask
        data_obj.processing_dict['mask_metrics'] = mask_metrics
        _write_snr_masks(data_obj, air_obj, mask_metrics)

        n, k, metrics = core.invert_nk_grid(freq, H, mask, config)
        _store_nk_grid(data_obj, freq, n, k, metrics)


def _grid_invert_substrate_sandwich(dataset, config):
    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        freq, H, mask = _get_transfer_data(data_obj, filename)
        if freq is None:
            continue

        sub_ref_obj = dataset.get_reference(filename, ref_type='substrate')
        if sub_ref_obj is None:
            print(f"Warning: no substrate reference for '{filename}', skipping.")
            continue

        n_sub = sub_ref_obj.processing_dict.get('n')
        k_sub = sub_ref_obj.processing_dict.get('k')
        if n_sub is None or k_sub is None:
            print(f"Warning: substrate reference for '{filename}' has no n,k — run substrate_only first, skipping.")
            continue

        sample_config = {
            **config,
            'invert_grid': {**config.get('invert_grid', {}), 'n_substrate': n_sub - 1j * k_sub},
        }
        n, k, metrics = core.invert_nk_grid(freq, H, mask, sample_config)
        _store_nk_grid(data_obj, freq, n, k, metrics)


def derive_eps_sigma(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Derive complex permittivity and optical conductivity from n, k."""
    config = config or {}

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        n = data_obj.processing_dict.get('n')
        k = data_obj.processing_dict.get('k')
        if n is None or k is None:
            print(f"Warning: no n,k for '{filename}', skipping derivation.")
            continue

        freq = data_obj.processing_dict['fft_freq']
        eps, sigma, metrics = core.derive_eps_sigma(freq, n, k, config)

        data_obj.processing_dict['eps'] = eps
        data_obj.processing_dict['sigma'] = sigma
        data_obj.processing_dict['derive_metrics'] = metrics

        data_obj.data = np.column_stack((
            freq,
            eps.real,
            eps.imag,
        ))

    return dataset


# ---------------------------------------------------------------------------
# Artificial time-shift sweep (qualitative phase-calibration exploration)
# ---------------------------------------------------------------------------

def _select_sample(dataset: DataSet, sample: str | None):
    """Resolve a sample to (filename, data_obj): explicit name, substring, or first."""
    names = [fn for fn, _ in dataset.data.items() if not dataset.data.is_reference(fn)]
    if not names:
        raise ValueError("Dataset has no non-reference samples.")
    if sample is None:
        return names[0], dataset.data[names[0]]
    if sample in names:
        return sample, dataset.data[sample]
    matches = [fn for fn in names if sample.lower() in fn.lower()]
    if len(matches) == 1:
        return matches[0], dataset.data[matches[0]]
    if not matches:
        raise ValueError(f"No sample matching '{sample}'. Available: {names}")
    raise ValueError(f"Ambiguous sample '{sample}' matches {matches}.")


def sweep_time_shift(
    dataset: DataSet,
    shifts_seconds,
    *,
    sample: str | None = None,
    band_thz: tuple | None = None,
    geometry: str = 'window',
    theta_deg: float = 45.0,
    polarization: str = 's',
    r_reference: complex = -1.0 + 0.0j,
    n_window: complex | np.ndarray = 1.95,
    config: dict | None = None,
) -> dict:
    """Apply a series of artificial sub-sample time shifts and re-invert each.

    Reuses the already-computed transfer function H (run the pipeline through
    ``transfer_function`` first). Each shift is applied EXACTLY as a spectral phase
    ramp ``H*exp(-i*2*pi*f*dt)`` (no time-domain interpolation), then the standard
    reflection inversion + eps/sigma derivation runs on the shifted H. Only the
    phase-derived quantities (n, k, sigma) change — |H| and the SNR mask are
    invariant under a pure phase ramp, so they are computed once.

    Parameters
    ----------
    shifts_seconds : array-like
        Artificial time shifts (seconds), applied on top of the current H.
    sample : str | None
        Sample filename (or unique substring). Default: first non-reference sample.
    band_thz : (float, float) | None
        Override the inversion band. By default the SNR ``transfer_mask`` is used,
        which for weak reflections can top out well below the Nyquist range. Pass a
        ``(lo, hi)`` THz window to invert over all finite bins in that range instead
        (useful for *qualitative* exploration past the trusted band — interpret the
        extra reach with the SNR caveat in mind).
    geometry, theta_deg, polarization, r_reference, n_window :
        Same reflection geometry parameters as ``invert_nk_reflection``.

    Returns
    -------
    dict with keys ``sample``, ``shifts`` (s), ``freq`` (Hz), ``mask``,
    ``n``, ``k``, ``sigma1``, ``sigma2`` (each shape ``(n_shifts, n_freq)``),
    plus ``theta_internal_rad`` and ``r_reference``.
    """
    config = config or {}
    filename, data_obj = _select_sample(dataset, sample)
    proc = data_obj.processing_dict
    H = proc.get('transfer_H')
    mask = proc.get('transfer_mask')
    freq = proc.get('fft_freq')
    if H is None or mask is None or freq is None:
        raise ValueError(
            f"'{filename}' has no transfer function; run the pipeline through "
            f"transfer_function before sweeping."
        )
    H = np.asarray(H)
    freq = np.asarray(freq)
    mask = np.asarray(mask)

    if band_thz is not None:
        freq_thz = freq * _HZ_TO_THZ
        mask = (
            (freq_thz >= band_thz[0]) & (freq_thz <= band_thz[1])
            & np.isfinite(H) & (freq > 0.0)
        )

    theta_external_rad = np.deg2rad(theta_deg)
    n_incident, theta_internal_rad, r_reference_value = _resolve_reflection_geometry(
        geometry, theta_external_rad, r_reference, n_window,
    )

    shifts = np.asarray(shifts_seconds, dtype=float)
    n_freq = freq.size
    n_arr = np.full((shifts.size, n_freq), np.nan)
    k_arr = np.full((shifts.size, n_freq), np.nan)
    sigma1 = np.full((shifts.size, n_freq), np.nan)
    sigma2 = np.full((shifts.size, n_freq), np.nan)

    for i, shift in enumerate(shifts):
        H_shifted = H * core.phase_ramp(freq, float(shift))
        r_sample = r_reference_value * H_shifted
        n, k, _ = core.invert_nk_reflection(
            freq, r_sample, mask, config,
            theta_rad=theta_internal_rad, polarization=polarization,
            n_incident=n_incident,
        )
        eps, sigma, _ = core.derive_eps_sigma(freq, n, k, config)
        n_arr[i] = n
        k_arr[i] = k
        sigma1[i] = np.real(sigma)
        sigma2[i] = np.imag(sigma)

    return {
        'sample': filename,
        'shifts': shifts,
        'freq': freq,
        'mask': mask,
        'n': n_arr,
        'k': k_arr,
        'sigma1': sigma1,
        'sigma2': sigma2,
        'theta_internal_rad': theta_internal_rad,
        'r_reference': r_reference_value,
    }


_SWEEP_PANELS = {
    'sigma': (('sigma1', 'σ₁ (S/m)'), ('sigma2', 'σ₂ (S/m)')),
    'nk': (('n', 'n'), ('k', 'k')),
}


def _robust_limits(values: np.ndarray, lo_pct: float, hi_pct: float) -> tuple:
    """Percentile limits over finite values, robust to ill-conditioning blow-up.

    The reflection inversion can spike by 1-2 orders of magnitude at low-SNR
    band edges (the |1+r|->0 conditioning limit). Auto-scaling to min/max then
    squashes the meaningful mid-band structure into a flat line/colour, so axes
    and colour ranges are set from percentiles instead.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def time_shift_waterfall(
    result: dict,
    *,
    quantity: str = 'sigma1',
    freq_range_thz: tuple | None = None,
    ax=None,
    cmap: str = 'viridis',
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Static 2D map of a swept quantity: x = frequency, y = artificial shift.

    ``result`` is the dict from ``sweep_time_shift``. ``quantity`` is one of
    ``'n'``, ``'k'``, ``'sigma1'``, ``'sigma2'``. The colour range defaults to the
    2-98th percentile of the displayed values (robust to the high-frequency
    ill-conditioning blow-up); override with ``vmin``/``vmax``.
    """
    import matplotlib.pyplot as plt

    freq_thz = result['freq'] * _HZ_TO_THZ
    shifts_ps = result['shifts'] * _S_TO_PS
    values = result[quantity]

    if freq_range_thz is not None:
        band = (freq_thz >= freq_range_thz[0]) & (freq_thz <= freq_range_thz[1])
    else:
        band = np.ones(freq_thz.size, dtype=bool)

    displayed = values[:, band]
    auto_lo, auto_hi = _robust_limits(displayed, 2.0, 98.0)
    vmin = auto_lo if vmin is None else vmin
    vmax = auto_hi if vmax is None else vmax

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    mesh = ax.pcolormesh(
        freq_thz[band], shifts_ps, displayed, cmap=cmap, shading='auto',
        vmin=vmin, vmax=vmax,
    )
    ax.figure.colorbar(mesh, ax=ax, label=quantity)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Artificial shift (ps)')
    ax.set_title(f"{result['sample']} — {quantity} vs artificial time shift")
    return ax


def time_shift_slider(
    dataset: DataSet,
    *,
    sample: str | None = None,
    shift_range_ps: tuple = (-0.05, 0.05),
    n_steps: int = 101,
    quantity: str = 'sigma',
    expected: dict | None = None,
    freq_range_thz: tuple | None = None,
    band_thz: tuple | None = None,
    geometry: str = 'window',
    theta_deg: float = 45.0,
    polarization: str = 's',
    r_reference: complex = -1.0 + 0.0j,
    n_window: complex | np.ndarray = 1.95,
    config: dict | None = None,
    show: bool = True,
    block: bool = True,
) -> dict:
    """Interactive slider over an artificial sub-sample time shift.

    Pre-sweeps the whole shift range once (``sweep_time_shift``), so each slider
    frame is just an array lookup + ``set_ydata`` + a ``draw_idle`` (the canonical
    matplotlib widget pattern — fast and smooth for the few-hundred-point curves
    here, and robust, unlike manual blitting which fights the Slider's own redraws).
    Two stacked panels show the swept quantity pair (σ₁/σ₂ for ``quantity='sigma'``;
    n/k for ``'nk'``). y-limits are fixed from the full sweep so the view is stable.

    ``band_thz`` overrides the SNR mask so you can explore past the trusted band
    (n/k/σ are NaN outside the inversion mask, which is why a weak reflection shows
    nothing above its SNR cut-off by default). ``freq_range_thz`` only sets the
    x-view; it does not widen the inversion.

    ``expected`` overlays fixed reference curves: a dict mapping a panel key
    (``'sigma1'``/``'sigma2'``/``'n'``/``'k'``) to ``(freq_thz, values)``.

    Returns a dict with the figure, the slider, the precomputed ``result``, and an
    ``update`` callable (so the view can be driven headlessly in tests).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    if quantity not in _SWEEP_PANELS:
        raise ValueError(f"quantity must be one of {list(_SWEEP_PANELS)}.")

    shifts = np.linspace(
        shift_range_ps[0] * 1e-12, shift_range_ps[1] * 1e-12, int(n_steps),
    )
    result = sweep_time_shift(
        dataset, shifts, sample=sample, band_thz=band_thz, geometry=geometry,
        theta_deg=theta_deg, polarization=polarization, r_reference=r_reference,
        n_window=n_window, config=config,
    )

    freq_thz = result['freq'] * _HZ_TO_THZ
    shifts_ps = result['shifts'] * _S_TO_PS
    if freq_range_thz is not None:
        band = (freq_thz >= freq_range_thz[0]) & (freq_thz <= freq_range_thz[1])
    else:
        band = np.ones(freq_thz.size, dtype=bool)
    freq_plot = freq_thz[band]

    panels = _SWEEP_PANELS[quantity]
    initial_index = int(np.argmin(np.abs(shifts_ps)))  # shift closest to 0

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(9, 7), sharex=True, layout='constrained',
    )
    if len(panels) == 1:
        axes = [axes]

    dynamic_lines = []
    for ax, (key, label) in zip(axes, panels):
        data = result[key][:, band]
        # Robust y-limits over the whole sweep: percentiles, not min/max, so the
        # high-frequency ill-conditioning blow-up doesn't flatten the real curve.
        lo, hi = _robust_limits(data, 1.0, 99.0)
        if lo is not None:
            pad = 0.08 * (hi - lo or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
        if expected is not None and key in expected:
            ef, ev = expected[key]
            ax.plot(ef, ev, '--', color='0.4', lw=1.6, label='expected')
        (line,) = ax.plot(
            freq_plot, data[initial_index], color='C0', lw=1.6, label='swept',
        )
        dynamic_lines.append((ax, line, data))
        ax.axhline(0.0, color='0.85', lw=0.8, zorder=0)
        ax.set_ylabel(label)
        ax.legend(loc='best', fontsize=8)
    axes[-1].set_xlabel('Frequency (THz)')

    slider_ax = fig.add_axes([0.15, 0.005, 0.7, 0.03])
    step_ps = float(shifts_ps[1] - shifts_ps[0]) if shifts_ps.size > 1 else 0.0
    slider = Slider(
        slider_ax, 'shift (ps)', shifts_ps[0], shifts_ps[-1],
        valinit=shifts_ps[initial_index], valstep=step_ps or None,
    )

    title = fig.suptitle(
        f"{result['sample']} — shift {shifts_ps[initial_index]:+.4f} ps"
    )

    def _index_for(val):
        return int(np.clip(round((val - shifts_ps[0]) / step_ps), 0, shifts_ps.size - 1)) \
            if step_ps else 0

    def update(val):
        i = _index_for(val)
        title.set_text(f"{result['sample']} — shift {shifts_ps[i]:+.4f} ps")
        for _ax, line, data in dynamic_lines:
            line.set_ydata(data[i])
        fig.canvas.draw_idle()
        return i

    slider.on_changed(update)
    update(shifts_ps[initial_index])  # draw the initial curve immediately

    if show:
        plt.show(block=block)

    return {
        'figure': fig,
        'slider': slider,
        'result': result,
        'update': update,
        'lines': [line for _, line, _ in dynamic_lines],
    }


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_results(dataset: DataSet, export_dir: str | None = None) -> list[str]:
    """Export per-sample CSV files containing all computed arrays.

    Three files per sample:
      {stem}_results.csv       – freq-domain: freq_THz, fft_mag, n, k,
                                 eps_real, eps_imag, sigma_real, sigma_imag
      {stem}_time_raw.csv      – original averaged time trace (before any
                                 processing): time_ps, amplitude, stderr
      {stem}_time_prefft.csv   – preprocessed time trace just before FFT
                                 (after baseline / align / window / pad):
                                 time_ps, amplitude

    Returns list of written file paths.
    """
    import os

    if export_dir is None:
        export_dir = os.path.join(dataset.file_dir, 'results')
    os.makedirs(export_dir, exist_ok=True)

    written = []
    for fn, data_obj in _sample_items(dataset):
        proc = data_obj.processing_dict
        stem = os.path.splitext(fn)[0]

        # --- raw time trace ---
        td_raw = proc.get('time_domain')
        if td_raw is not None:
            raw_path = os.path.join(export_dir, f"{stem}_time_raw.csv")
            raw_out = td_raw.copy()
            raw_out[:, 0] *= _S_TO_PS          # s → ps
            np.savetxt(raw_path, raw_out,
                       delimiter=',',
                       header='time_ps,amplitude,stderr',
                       comments='')
            written.append(raw_path)

        # --- pre-FFT time trace ---
        td_prefft = proc.get('time_domain_prefft')
        if td_prefft is not None:
            prefft_path = os.path.join(export_dir, f"{stem}_time_prefft.csv")
            prefft_out = td_prefft.copy()
            prefft_out[:, 0] *= _S_TO_PS       # s → ps
            np.savetxt(prefft_path, prefft_out,
                       delimiter=',',
                       header='time_ps,amplitude',
                       comments='')
            written.append(prefft_path)

        # --- frequency-domain results ---
        freq = proc.get('fft_freq')
        if freq is None:
            print(f"Skipping freq-domain for '{fn}': no FFT data.")
            continue

        nan_col = np.full_like(freq, np.nan)

        def _safe(arr):
            return arr if arr is not None else nan_col

        spec = proc.get('fft_spectrum')
        eps = proc.get('eps')
        sigma = proc.get('sigma')

        columns = [
            freq * _HZ_TO_THZ,
            np.abs(spec) if spec is not None else nan_col,
            _safe(proc.get('n')),
            _safe(proc.get('k')),
            eps.real if eps is not None else nan_col,
            eps.imag if eps is not None else nan_col,
            sigma.real if sigma is not None else nan_col,
            sigma.imag if sigma is not None else nan_col,
        ]

        headers = 'freq_THz,fft_mag,n,k,eps_real,eps_imag,sigma_real,sigma_imag'
        data = np.column_stack(columns)

        filepath = os.path.join(export_dir, f"{stem}_results.csv")
        np.savetxt(filepath, data, delimiter=',', header=headers, comments='')
        written.append(filepath)

    print(f"Exported {len(written)} file(s) to {export_dir}")
    return written


# ---------------------------------------------------------------------------
# plotting (standalone)
# ---------------------------------------------------------------------------

def _sample_items(dataset: DataSet):
    """Yield (filename, data_obj) for non-reference files."""
    for filename, data_obj in dataset.data.items():
        if not dataset.data.is_reference(filename):
            yield filename, data_obj


def plot_fft(
    dataset: DataSet,
    freq_range: tuple | None = None,
    normalise: bool = False,
    show_snr_mask: bool = True,
    **kwargs,
) -> None:
    """Plot FFT magnitude for every file (samples and references).

    When ``show_snr_mask`` is True and per-spectrum SNR masks have been
    computed (i.e. ``transfer_function`` has run), regions below each
    spectrum's own SNR threshold are dimmed.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    if kwargs.get('scale', None) == 'log':
        ax.set_yscale('log')

    for filename, data_obj in dataset.data.items():
        freq = data_obj.processing_dict.get('fft_freq')
        spectrum = data_obj.processing_dict.get('fft_spectrum')
        if freq is None or spectrum is None:
            continue
        norm_range = np.where((freq >= (freq_range[0] / _HZ_TO_THZ if freq_range else 0)) &
                              (freq <= (freq_range[1] / _HZ_TO_THZ if freq_range else np.inf)))
        norm = np.abs(spectrum[norm_range]).max() if normalise else 1.0
        mag = np.abs(spectrum) / norm
        snr_mask = data_obj.processing_dict.get('snr_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax, freq * _HZ_TO_THZ, mag, snr_mask, label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('|FFT|')
    ax.set_title('FFT Magnitude')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_transfer_function(
    dataset: DataSet,
    freq_range: tuple | None = None,
    show_snr_mask: bool = True,
) -> None:
    """Plot transfer function magnitude for each sample, dimming low-SNR bins."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in _sample_items(dataset):
        H = data_obj.processing_dict.get('transfer_H')
        freq = data_obj.processing_dict.get('fft_freq')
        if H is None or freq is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax, freq * _HZ_TO_THZ, np.abs(H), mask, label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transfer Function Magnitude')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_transfer_phase(
    dataset: DataSet,
    freq_range: tuple | None = None,
    show_snr_mask: bool = True,
) -> None:
    """Plot unwrapped phase of the transfer function for each sample."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in _sample_items(dataset):
        H = data_obj.processing_dict.get('transfer_H')
        freq = data_obj.processing_dict.get('fft_freq')
        if H is None or freq is None:
            continue
        phase = np.unwrap(np.angle(H))
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax, freq * _HZ_TO_THZ, phase, mask, label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Phase (rad, unwrapped)')
    ax.set_title('Transfer Function Phase (unwrapped)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_nk(
    dataset: DataSet,
    freq_range: tuple | None = None,
    show_snr_mask: bool = True,
) -> None:
    """Plot refractive index n and extinction coefficient k for each sample."""
    import matplotlib.pyplot as plt

    fig, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        n = data_obj.processing_dict.get('n')
        k = data_obj.processing_dict.get('k')
        if freq is None or n is None or k is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax_n, freq * _HZ_TO_THZ, n, mask, label=filename)
        _plot_with_snr_mask(ax_k, freq * _HZ_TO_THZ, k, mask, label=filename)

    if freq_range is not None:
        ax_n.set_xlim(freq_range)
    ax_n.set_ylabel('n')
    ax_n.set_title('Refractive Index')
    ax_n.legend()
    ax_k.set_xlabel('Frequency (THz)')
    ax_k.set_ylabel('k')
    ax_k.set_title('Extinction Coefficient')
    ax_k.legend()
    plt.tight_layout()
    plt.show()


def plot_permittivity(
    dataset: DataSet,
    freq_range: tuple | None = None,
    show_snr_mask: bool = True,
) -> None:
    """Plot real and imaginary parts of the complex permittivity."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        eps = data_obj.processing_dict.get('eps')
        if freq is None or eps is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax_r, freq * _HZ_TO_THZ, eps.real, mask, label=filename)
        _plot_with_snr_mask(ax_i, freq * _HZ_TO_THZ, eps.imag, mask, label=filename)

    if freq_range is not None:
        ax_r.set_xlim(freq_range)
    ax_r.set_ylabel(r'$\varepsilon_r$')
    ax_r.set_title(r'Permittivity — Real Part ($\varepsilon_r = n^2 - k^2$)')
    ax_r.legend()
    ax_i.set_xlabel('Frequency (THz)')
    ax_i.set_ylabel(r'$\varepsilon_i$')
    ax_i.set_title(r'Permittivity — Imaginary Part ($\varepsilon_i = 2nk$)')
    ax_i.legend()
    plt.tight_layout()
    plt.show()


def plot_conductivity(
    dataset: DataSet,
    freq_range: tuple | None = None,
    show_snr_mask: bool = True,
) -> None:
    """Plot real and imaginary parts of the optical conductivity."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        sigma = data_obj.processing_dict.get('sigma')
        if freq is None or sigma is None:
            continue
        mask = data_obj.processing_dict.get('transfer_mask') if show_snr_mask else None
        _plot_with_snr_mask(ax_r, freq * _HZ_TO_THZ, sigma.real, mask, label=filename)
        _plot_with_snr_mask(ax_i, freq * _HZ_TO_THZ, sigma.imag, mask, label=filename)

    if freq_range is not None:
        ax_r.set_xlim(freq_range)
    ax_r.set_ylabel(r'$\sigma_r$ (S/m)')
    ax_r.set_title(r'Optical Conductivity — Real Part')
    ax_r.legend()
    ax_i.set_xlabel('Frequency (THz)')
    ax_i.set_ylabel(r'$\sigma_i$ (S/m)')
    ax_i.set_title(r'Optical Conductivity — Imaginary Part')
    ax_i.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# phase correction demo
# ---------------------------------------------------------------------------

def phase_correction(dataset: DataSet, source: str = 'transfer') -> DataSet:
    """Interactive demo: fit a line to a selected phase region, subtract the
    y-intercept (timing-offset correction), then re-wrap.

    Parameters
    ----------
    dataset : DataSet
        Must already have FFT data (and transfer function if *source='transfer'*).
    source : str
        Which complex spectrum to operate on:
        - ``'transfer'`` - unwrapped phase of H(f)  (default)
        - ``'fft'``      - unwrapped phase of the raw FFT spectrum

    Workflow (per trace, blocking):
        1. Show unwrapped phase vs frequency (THz).
        2. User drags a span to select a "trusted" linear region.
        3. Linear regression is fitted; y-intercept = assumed timing error.
        4. Corrected phase is shown overlaid; a second figure shows wrapped
           comparison (before / after).
        5. The corrected complex spectrum is written back into the dataset.

    Returns the dataset (modified in-place) for chaining.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import SpanSelector

    filenames = list(dataset.data.keys())

    for filename in filenames:
        data_obj = dataset.data[filename]
        proc = data_obj.processing_dict

        freq = proc.get('fft_freq')
        if freq is None:
            print(f"Skipping '{filename}': no FFT data.")
            continue

        if source == 'transfer':
            spectrum = proc.get('transfer_H')
            label = 'H(f)'
            if spectrum is None:
                print(f"Skipping '{filename}': no transfer function.")
                continue
        else:
            spectrum = proc.get('fft_spectrum')
            label = 'FFT'
            if spectrum is None:
                print(f"Skipping '{filename}': no FFT spectrum.")
                continue

        freq_thz = freq * _HZ_TO_THZ
        unwrapped = np.unwrap(np.angle(spectrum))

        # --- interactive selection figure ---
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(freq_thz, unwrapped, color='steelblue', label='unwrapped phase')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title(f'{label} phase — {filename}\n'
                      'Drag to select linear region, then close window')
        ax.legend(loc='upper right')

        selection = {}

        def on_select(xmin, xmax):
            selection['xmin'] = xmin
            selection['xmax'] = xmax

            mask = (freq_thz >= xmin) & (freq_thz <= xmax)
            if mask.sum() < 2:
                return

            coeffs = np.polyfit(freq_thz[mask], unwrapped[mask], 1)
            fit_line = np.polyval(coeffs, freq_thz)

            # Clear previous fit overlay (keep original trace)
            while len(ax.lines) > 1:
                ax.lines[-1].remove()
            ax.axvspan(xmin, xmax, alpha=0.15, color='orange', label='selected')
            ax.plot(freq_thz, fit_line, '--', color='crimson', lw=1.5,
                    label=f'fit: slope={coeffs[0]:.3f}, intercept={coeffs[1]:.3f}')
            ax.legend(loc='upper right', fontsize=8)
            fig.canvas.draw_idle()

            selection['coeffs'] = coeffs

        span = SpanSelector(ax, on_select, 'horizontal',
                            useblit=True, interactive=True,
                            props=dict(alpha=0.25, facecolor='orange'))
        plt.tight_layout()
        plt.show()  # blocks until window closed

        if 'coeffs' not in selection:
            print(f"  No region selected for '{filename}', skipping correction.")
            continue

        slope, intercept = selection['coeffs']
        print(f"  {filename}: slope={slope:.4f} rad/THz, intercept={intercept:.4f} rad")

        # --- apply correction ---
        # The linear phase φ(f) = slope·f + intercept.
        # intercept is the timing-error offset; subtract the full linear trend
        # so that the residual phase is only dispersion.
        correction = np.polyval(selection['coeffs'], freq_thz)
        # corrected_unwrapped = unwrapped - correction
        corrected_unwrapped = unwrapped - intercept  # only remove y-intercept

        plt.plot(freq_thz, unwrapped, color='steelblue', alpha=0.5, label='original')
        plt.plot(freq_thz, corrected_unwrapped, color='darkorange', label='corrected')
        plt.legend()
        plt.title(f'{filename} — unwrapped phase before/after correction')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Phase (rad)')
        plt.show()

        # Rebuild corrected complex spectrum (preserve magnitude)
        corrected_wrapped = np.angle(np.exp(1j * corrected_unwrapped))
        magnitude = np.abs(spectrum)
        corrected_spectrum = magnitude * np.exp(1j * corrected_unwrapped)

        # --- before/after comparison ---
        fig2, (ax_uw, ax_w) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        ax_uw.plot(freq_thz, unwrapped, 'steelblue', alpha=0.5, label='original')
        ax_uw.plot(freq_thz, corrected_unwrapped, 'darkorange', label='corrected')
        ax_uw.set_ylabel('Unwrapped phase (rad)')
        ax_uw.set_title(f'{filename} — unwrapped phase before/after')
        ax_uw.legend(fontsize=8)

        original_wrapped = np.angle(spectrum)
        ax_w.plot(freq_thz, original_wrapped, 'steelblue', alpha=0.5, label='original')
        ax_w.plot(freq_thz, corrected_wrapped, 'darkorange', label='corrected')
        ax_w.set_xlabel('Frequency (THz)')
        ax_w.set_ylabel('Wrapped phase (rad)')
        ax_w.set_title(f'{filename} — wrapped phase before/after')
        ax_w.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        # --- write back ---
        if source == 'transfer':
            proc['transfer_H'] = corrected_spectrum
            data_obj.data = np.column_stack((
                freq, magnitude, np.angle(corrected_spectrum),
            ))
        else:
            proc['fft_spectrum'] = corrected_spectrum
            data_obj.data = np.column_stack((
                freq, magnitude, np.angle(corrected_spectrum),
            ))

        proc['phase_correction'] = {
            'slope_rad_per_THz': slope,
            'intercept_rad': intercept,
            'source': source,
        }
        print(f"  → phase corrected and written back for '{filename}'.")

    return dataset


# ---------------------------------------------------------------------------
# interactive viewer
# ---------------------------------------------------------------------------

class ResultViewer:
    """Interactive matplotlib GUI for browsing all THz-TDS results.

    Features:
    - Radio buttons to switch between plot types
    - Checkboxes to toggle individual sample traces
    - Crosshair cursor with live coordinate readout
    - Home button to reset zoom after panning/zooming
    - All in one window, no re-run required
    """

    PLOT_TYPES = [
        'FFT Magnitude',
        'FFT Phase',
        'Transfer |H|',
        'Transfer Phase',
        'n  (refractive index)',
        'k  (extinction)',
        'Permittivity',
        'Conductivity',
    ]

    def __init__(self, dataset: DataSet, show_snr_mask: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, CheckButtons, TextBox

        self._plt = plt
        self._dataset = dataset
        self._show_snr_mask = bool(show_snr_mask)

        # Collect sample filenames (ordered) and reference filenames
        self._sample_names = [
            fn for fn, _ in _sample_items(dataset)
        ]
        self._all_names = [fn for fn in dataset.data.keys()]
        self._visible = {fn: True for fn in self._sample_names}

        # Assign a stable color per sample for consistency across views
        cmap = plt.cm.get_cmap('tab10')
        self._colours = {
            fn: cmap(i % 10) for i, fn in enumerate(self._sample_names)
        }
        # References get grey tones
        ref_names = [fn for fn in self._all_names if fn not in self._sample_names]
        for i, fn in enumerate(ref_names):
            self._colours[fn] = (0.5, 0.5, 0.5, 0.6)

        self._current_plot = self.PLOT_TYPES[0]
        self._freq_min = None   # persistent x-axis limits (None = auto)
        self._freq_max = None

        # --- layout ---
        self._fig = plt.figure(figsize=(14, 8))
        # Main axes area (leave room on right for widgets)
        self._ax_top = self._fig.add_axes([0.07, 0.54, 0.60, 0.40])
        self._ax_bot = self._fig.add_axes([0.07, 0.08, 0.60, 0.40], sharex=self._ax_top)
        # For single-panel plots, hide bottom and expand top
        self._single_panel_mode = False

        # Radio buttons for plot type
        radio_ax = self._fig.add_axes([0.72, 0.50, 0.26, 0.45])
        radio_ax.set_frame_on(False)
        self._radio = RadioButtons(radio_ax, self.PLOT_TYPES, active=0)
        self._radio.on_clicked(self._on_plot_type_changed)

        # Checkboxes for trace visibility
        short_names = [self._short(fn) for fn in self._sample_names]
        initial_vis = [True] * len(self._sample_names)
        check_ax = self._fig.add_axes([0.72, 0.05, 0.26, 0.42])
        check_ax.set_title('Samples', fontsize=9, loc='left')
        check_ax.set_frame_on(False)
        self._check = CheckButtons(check_ax, short_names, initial_vis)
        self._check.on_clicked(self._on_check_toggled)

        # Frequency range textboxes
        self._fig.text(0.72, 0.48, 'Freq range (THz):', fontsize=8)
        fmin_ax = self._fig.add_axes([0.72, 0.44, 0.10, 0.035])
        fmax_ax = self._fig.add_axes([0.85, 0.44, 0.10, 0.035])
        self._tb_fmin = TextBox(fmin_ax, '', initial='', textalignment='center')
        self._tb_fmax = TextBox(fmax_ax, '', initial='', textalignment='center')
        self._tb_fmin.on_submit(self._on_freq_min_changed)
        self._tb_fmax.on_submit(self._on_freq_max_changed)

        # Coordinate readout text
        self._coord_text = self._fig.text(
            0.07, 0.01, '', fontsize=8, family='monospace',
        )
        self._fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self._draw_current()
        plt.show()

    # --- short display name ---
    @staticmethod
    def _short(filename: str) -> str:
        """Truncate long filenames for checkbox labels."""
        if len(filename) > 30:
            return '...' + filename[-27:]
        return filename

    # --- callbacks ---
    def _on_plot_type_changed(self, label: str):
        self._current_plot = label
        self._draw_current()

    def _on_check_toggled(self, label: str):
        # Reverse-map short name to full name
        for fn in self._sample_names:
            if self._short(fn) == label:
                self._visible[fn] = not self._visible[fn]
                break
        self._draw_current()

    def _on_freq_min_changed(self, text: str):
        text = text.strip()
        self._freq_min = float(text) if text else None
        self._draw_current()

    def _on_freq_max_changed(self, text: str):
        text = text.strip()
        self._freq_max = float(text) if text else None
        self._draw_current()

    def _on_mouse_move(self, event):
        if event.inaxes in (self._ax_top, self._ax_bot):
            self._coord_text.set_text(f'x={event.xdata:.4g}   y={event.ydata:.4g}')
            self._fig.canvas.draw_idle()

    # --- data extractors ---
    def _get(self, filename: str, key: str):
        obj = self._dataset.data.get(filename)
        if obj is None:
            return None
        return obj.processing_dict.get(key)

    # --- drawing ---
    def _draw_current(self):
        plot_type = self._current_plot
        if plot_type in ('Permittivity', 'Conductivity', 'n  (refractive index)', 'k  (extinction)'):
            self._set_dual_panel()
        else:
            self._set_single_panel()

        self._ax_top.clear()
        self._ax_bot.clear()

        draw_fn = {
            'FFT Magnitude':          self._draw_fft,
            'FFT Phase':              self._draw_fft_phase,
            'Transfer |H|':           self._draw_transfer_mag,
            'Transfer Phase':         self._draw_transfer_phase,
            'n  (refractive index)':  self._draw_nk,
            'k  (extinction)':        self._draw_nk,
            'Permittivity':           self._draw_permittivity,
            'Conductivity':           self._draw_conductivity,
        }[plot_type]

        draw_fn()
        self._apply_freq_limits()
        self._fig.canvas.draw_idle()

    def _set_single_panel(self):
        self._ax_top.set_position([0.07, 0.08, 0.60, 0.86])
        self._ax_bot.set_visible(False)
        self._single_panel_mode = True

    def _set_dual_panel(self):
        self._ax_top.set_position([0.07, 0.54, 0.60, 0.40])
        self._ax_bot.set_position([0.07, 0.08, 0.60, 0.40])
        self._ax_bot.set_visible(True)
        self._single_panel_mode = False

    def _apply_freq_limits(self):
        if self._freq_min is not None or self._freq_max is not None:
            lo = self._freq_min if self._freq_min is not None else None
            hi = self._freq_max if self._freq_max is not None else None
            self._ax_top.set_xlim(left=lo, right=hi)
            if self._ax_bot.get_visible():
                self._ax_bot.set_xlim(left=lo, right=hi)

    def _visible_samples(self):
        for fn in self._sample_names:
            if self._visible.get(fn, False):
                yield fn

    def _spec_mask(self, fn: str):
        """Per-spectrum SNR mask (sample or reference) for FFT-domain plots."""
        return self._get(fn, 'snr_mask') if self._show_snr_mask else None

    def _trusted_mask(self, fn: str):
        """Combined trusted-band mask for transfer-derived plots (H, n, k, eps, sigma)."""
        return self._get(fn, 'transfer_mask') if self._show_snr_mask else None

    def _draw_fft(self):
        ax = self._ax_top
        ax.set_yscale('log')
        # Samples at higher alpha, references at lower alpha; SNR mask dims further.
        for fn in self._all_names:
            freq = self._get(fn, 'fft_freq')
            spec = self._get(fn, 'fft_spectrum')
            if freq is None or spec is None:
                continue
            is_sample = fn in self._sample_names
            if is_sample and not self._visible.get(fn, False):
                continue
            full_alpha = 0.9 if is_sample else 0.4
            masked_alpha = 0.2 if is_sample else 0.1
            _plot_with_snr_mask(
                ax, freq * _HZ_TO_THZ, np.abs(spec), self._spec_mask(fn),
                color=self._colours[fn], label=self._short(fn),
                full_alpha=full_alpha, masked_alpha=masked_alpha,
            )
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('|FFT|')
        ax.set_title('FFT Magnitude')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_fft_phase(self):
        ax = self._ax_top
        for fn in self._all_names:
            freq = self._get(fn, 'fft_freq')
            spec = self._get(fn, 'fft_spectrum')
            if freq is None or spec is None:
                continue
            is_sample = fn in self._sample_names
            if is_sample and not self._visible.get(fn, False):
                continue
            phase = np.unwrap(np.angle(spec))
            full_alpha = 0.9 if is_sample else 0.4
            masked_alpha = 0.2 if is_sample else 0.1
            _plot_with_snr_mask(
                ax, freq * _HZ_TO_THZ, phase, self._spec_mask(fn),
                color=self._colours[fn], label=self._short(fn),
                full_alpha=full_alpha, masked_alpha=masked_alpha,
            )
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('FFT Phase (unwrapped)')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_transfer_mag(self):
        ax = self._ax_top
        for fn in self._visible_samples():
            H = self._get(fn, 'transfer_H')
            freq = self._get(fn, 'fft_freq')
            if H is None or freq is None:
                continue
            _plot_with_snr_mask(
                ax, freq * _HZ_TO_THZ, np.abs(H), self._trusted_mask(fn),
                color=self._colours[fn], label=self._short(fn),
            )
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('|H(f)|')
        ax.set_title('Transfer Function Magnitude')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_transfer_phase(self):
        ax = self._ax_top
        for fn in self._visible_samples():
            H = self._get(fn, 'transfer_H')
            freq = self._get(fn, 'fft_freq')
            if H is None or freq is None:
                continue
            phase = np.unwrap(np.angle(H))
            _plot_with_snr_mask(
                ax, freq * _HZ_TO_THZ, phase, self._trusted_mask(fn),
                color=self._colours[fn], label=self._short(fn),
            )
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Transfer Function Phase (unwrapped)')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_nk(self):
        ax_n, ax_k = self._ax_top, self._ax_bot
        for fn in self._visible_samples():
            freq = self._get(fn, 'fft_freq')
            n = self._get(fn, 'n')
            k = self._get(fn, 'k')
            if freq is None or n is None or k is None:
                continue
            mask = self._trusted_mask(fn)
            _plot_with_snr_mask(
                ax_n, freq * _HZ_TO_THZ, n, mask,
                color=self._colours[fn], label=self._short(fn),
            )
            _plot_with_snr_mask(
                ax_k, freq * _HZ_TO_THZ, k, mask,
                color=self._colours[fn], label=self._short(fn),
            )
        ax_n.set_ylabel('n')
        ax_n.set_title('Refractive Index')
        ax_n.legend(fontsize=7, loc='upper right')
        ax_k.set_xlabel('Frequency (THz)')
        ax_k.set_ylabel('k')
        ax_k.set_title('Extinction Coefficient')
        ax_k.legend(fontsize=7, loc='upper right')

    def _draw_permittivity(self):
        ax_r, ax_i = self._ax_top, self._ax_bot
        for fn in self._visible_samples():
            freq = self._get(fn, 'fft_freq')
            eps = self._get(fn, 'eps')
            if freq is None or eps is None:
                continue
            mask = self._trusted_mask(fn)
            _plot_with_snr_mask(
                ax_r, freq * _HZ_TO_THZ, eps.real, mask,
                color=self._colours[fn], label=self._short(fn),
            )
            _plot_with_snr_mask(
                ax_i, freq * _HZ_TO_THZ, eps.imag, mask,
                color=self._colours[fn], label=self._short(fn),
            )
        ax_r.set_ylabel(r'$\varepsilon_r$')
        ax_r.set_title(r'Permittivity — Real ($n^2 - k^2$)')
        ax_r.legend(fontsize=7, loc='upper right')
        ax_i.set_xlabel('Frequency (THz)')
        ax_i.set_ylabel(r'$\varepsilon_i$')
        ax_i.set_title(r'Permittivity — Imaginary ($2nk$)')
        ax_i.legend(fontsize=7, loc='upper right')

    def _draw_conductivity(self):
        ax_r, ax_i = self._ax_top, self._ax_bot
        for fn in self._visible_samples():
            freq = self._get(fn, 'fft_freq')
            sigma = self._get(fn, 'sigma')
            if freq is None or sigma is None:
                continue
            mask = self._trusted_mask(fn)
            _plot_with_snr_mask(
                ax_r, freq * _HZ_TO_THZ, sigma.real, mask,
                color=self._colours[fn], label=self._short(fn),
            )
            _plot_with_snr_mask(
                ax_i, freq * _HZ_TO_THZ, sigma.imag, mask,
                color=self._colours[fn], label=self._short(fn),
            )
        ax_r.set_ylabel(r'$\sigma_r$ (S/m)')
        ax_r.set_title('Optical Conductivity — Real')
        ax_r.legend(fontsize=7, loc='upper right')
        ax_i.set_xlabel('Frequency (THz)')
        ax_i.set_ylabel(r'$\sigma_i$ (S/m)')
        ax_i.set_title('Optical Conductivity — Imaginary')
        ax_i.legend(fontsize=7, loc='upper right')


def result_viewer(dataset: DataSet, show_snr_mask: bool = True) -> ResultViewer:
    """Launch the interactive result viewer GUI.

    Pass ``show_snr_mask=False`` to plot all bins at full alpha (useful when
    you want to inspect noise-dominated regions explicitly).
    """
    return ResultViewer(dataset, show_snr_mask=show_snr_mask)

def validate_thz(dataset: DataSet, verbose=True, label: str = "Validation") -> dict:
    """Check if dataset has the required structure for THz processing."""
    if not dataset.data:
        print("Dataset is empty.")
        return False
    
    validation = core.validate_thz_dict({filename: _time_amplitude_array(data_obj) for filename, data_obj in dataset.data.items()})
    print(validation)

    if verbose:
        print_metrics(label or "Input Validation", validation)

    if any(validation.get('flags', {}).values()):
        print("WARNING: Validation failed with flags:")
        for key, val in validation.get('flags', {}).items():
            if val:
                print(f"  - {key}")
        raise ValueError("Dataset failed THz validation. See flags for details.")

    return validation

# ═══════════════════════════════════════════════════════
#  PRINTING / SUMMARY HELPERS
# ═══════════════════════════════════════════════════════

def print_metrics(label: str, metrics: dict) -> None:
    """Print stage metrics to console."""
    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print(f"{'=' * 54}")
    vals = metrics.get("values", {})
    for key, val in vals.items():
        if isinstance(val, np.ndarray):
            print(
                f"  {key:30s} : array({val.shape})"
            )
        elif isinstance(val, dict):
            print(f"  {key:30s} :")
            for sub_key, sub_val in val.items():
                print(
                    f"    {sub_key:28s} : {sub_val}"
                )
        else:
            print(f"  {key:30s} : {val}")
    flags = metrics.get("flags", {})
    if flags:
        for key, val in flags.items():
            flag_str = "!! FLAGGED" if val else "   ok"
            print(f"  {key:30s} : {flag_str}")
