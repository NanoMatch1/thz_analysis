import numpy as np
import thz_core.thz_core as core
import matplotlib.pyplot as plt
from dataset_core.dataset import DataSet, DataService
from dataset_core.data_structures.thz import THzDataReflection, THzData, BaseTHzData
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

    ``'second_reflection'`` → ``data_obj.second_reflection`` for
    ``THzDataReflection``; *data_obj* itself for plain ``THzData``.
    ``'first_reflection'``  → ``data_obj.first_reflection`` for
    ``THzDataReflection`` only; returns ``None`` for plain ``THzData``.
    """
    if segment == 'first_reflection':
        if not isinstance(data_obj, THzDataReflection):
            return None
        return data_obj.first_reflection
    if isinstance(data_obj, THzDataReflection):
        return data_obj.second_reflection
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
# In-memory reflection construction helpers
# ---------------------------------------------------------------------------

def _crop_thzdata_to_gate(
    data_obj: THzData,
    t_start_ps: float,
    t_stop_ps: float,
) -> THzData:
    """Crop every individual scan in *data_obj* to the time gate [t_start_ps, t_stop_ps].

    ``raw_data[:, 0]`` is in ps (source units), so the comparison is direct.
    Returns a new ``THzData`` built from the cropped scan list — no file I/O.
    The averaged trace, processing_dict, and scan matrix are re-derived from
    the cropped scans so the new object is self-consistent.
    """
    cropped_scans: list[BaseTHzData] = []
    for scan in data_obj.data_list:
        raw = np.asarray(scan.raw_data, dtype=float)
        mask = (raw[:, 0] >= t_start_ps) & (raw[:, 0] <= t_stop_ps)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"Gate [{t_start_ps}, {t_stop_ps}] ps selects <2 samples of "
                f"'{data_obj.filename}'."
            )
        cropped_scan = BaseTHzData(data=raw[mask], headers=scan.headers)
        cropped_scans.append(cropped_scan)
    return THzData(
        data=cropped_scans,
        header=data_obj.headers,
        filename=data_obj.filename,
        data_type=data_obj.data_type,
    )


def build_reflection_dataset(
    dataset: DataSet,
    gates: dict,
    bounds_units: str = 'ps',
) -> DataSet:
    """Crop all loaded traces in-memory to form first+second reflection gates.

    Replaces each ``THzData`` entry in *dataset* with a ``THzDataReflection``
    built from two in-memory crops — no file I/O, no reload round-trip.
    Individual scans are preserved in each segment so per-scan statistics
    survive to the FFT stage.

    Parameters
    ----------
    dataset : DataSet
        Must be loaded from the flat raw directory (not the segmented layout).
    gates : dict
        ``{'first_reflection': (start, stop), 'second_reflection': (start, stop)}``
        Bounds in *bounds_units*.
    bounds_units : {'ps', 's'}
        Units of the supplied bounds. ``raw_data[:, 0]`` is always in ps;
        's' bounds are converted before cropping.

    Returns
    -------
    DataSet
        Same object, with each entry replaced by a ``THzDataReflection``.
    """
    if bounds_units not in ('ps', 's'):
        raise ValueError("bounds_units must be 'ps' or 's'.")
    to_ps = (lambda v: float(v)) if bounds_units == 'ps' else (lambda v: float(v) * _S_TO_PS)

    first_bounds = gates.get('first_reflection')
    second_bounds = gates.get('second_reflection')
    if first_bounds is None or second_bounds is None:
        raise ValueError(
            "gates must contain both 'first_reflection' and 'second_reflection' keys."
        )
    first_start_ps, first_stop_ps = to_ps(first_bounds[0]), to_ps(first_bounds[1])
    second_start_ps, second_stop_ps = to_ps(second_bounds[0]), to_ps(second_bounds[1])

    for filename, data_obj in list(dataset.data.items()):
        first_seg = _crop_thzdata_to_gate(data_obj, first_start_ps, first_stop_ps)
        second_seg = _crop_thzdata_to_gate(data_obj, second_start_ps, second_stop_ps)
        reflection_obj = THzDataReflection.from_thzdata(second_seg, first_seg)
        dataset.data[filename] = reflection_obj

    return dataset


def define_reflection_gates(
    dataset: DataSet,
    config: dict,
) -> dict:
    """Resolve or interactively select the first/second reflection time gates.

    Reads ``config['gates']`` and fills any ``None`` component by opening a
    ``SpanSelector`` on a representative trace.  Returns the filled gates dict
    (also written back into ``config['gates']``).

    The representative trace is the first file's averaged data in ps.
    """
    gates = config.get('gates', {})
    components = ('first_reflection', 'second_reflection')

    first_obj = next(iter(dataset.data.values()))
    raw = np.asarray(first_obj.raw_data, dtype=float)
    time_ps = raw[:, 0]

    # raw_data columns: [time_ps, scan1, ..., scanN] if multi-scan, or [time_ps, amp]
    if raw.shape[1] > 2:
        mean_y = raw[:, 1:].mean(axis=1)
    else:
        mean_y = raw[:, 1]

    for component in components:
        if gates.get(component) is None:
            bounds = _span_select_bounds(time_ps, mean_y, title=f"Select {component}")
            gates[component] = bounds
            print(f"  {component}: {bounds[0]:.3f} – {bounds[1]:.3f} ps")
        else:
            print(f"  {component} (preset): {gates[component][0]:.3f} – {gates[component][1]:.3f} ps")

    config['gates'] = gates
    return gates


# ---------------------------------------------------------------------------
# Shared-axis reflection isolation (build full trace -> define regions ->
# isolate_and_window). Keeps both reflections on ONE time axis so the inter-pulse
# phase relationship is structural — no common-grid step, no absolute-time factor.
# ---------------------------------------------------------------------------

def build_full_trace_reflection(dataset: DataSet) -> DataSet:
    """Wrap each loaded full trace into a THzDataReflection WITHOUT cropping to gates.

    Both segments (``first_reflection`` and ``second_reflection``) start as
    independent copies of the **whole** trace on one shared time axis. The
    reflection regions are isolated later by ``isolate_and_window`` — so the two
    pulses keep a single, shared axis and their true inter-pulse delay.

    No echo cropping happens here: any GaP echo is excluded by the second-reflection
    region's trailing edge, and ``isolate_and_window`` zeros everything outside each
    region before windowing — so the region selection is the only echo filter needed.

    Each object gets ``first_region``/``second_region`` attributes initialised to
    ``None`` for ``define_reflection_regions`` to fill.
    """
    for filename, data_obj in list(dataset.data.items()):
        raw_time_ps = np.asarray(data_obj.raw_data, dtype=float)[:, 0]
        start_ps = float(raw_time_ps.min())
        stop_ps = float(raw_time_ps.max())

        first_full_copy = _crop_thzdata_to_gate(data_obj, start_ps, stop_ps)
        second_full_copy = _crop_thzdata_to_gate(data_obj, start_ps, stop_ps)
        reflection_obj = THzDataReflection.from_thzdata(second_full_copy, first_full_copy)
        reflection_obj.first_region = None
        reflection_obj.second_region = None
        dataset.data[filename] = reflection_obj

        print(
            f"[build_full_trace_reflection] '{filename}': full trace "
            f"[{start_ps:.1f}, {stop_ps:.1f}] ps ({len(first_full_copy.data)} samples) "
            f"wrapped on a shared axis; regions not yet defined."
        )
    return dataset


def define_reflection_regions(
    dataset: DataSet,
    config: dict | None = None,
    show_graph: bool = False,
) -> dict:
    """Define BOTH reflection regions once — from preset bounds or a SpanSelector.

    One function for both regions. ``config['regions']`` may hold
    ``{'first_reflection': (start, stop) | None, 'second_reflection': ... }`` in ps.
    Any entry that is ``None`` is selected interactively on a representative trace.
    The resolved (start, stop) ps tuples are written onto every THzDataReflection as
    ``first_region`` / ``second_region`` attributes (self-describing) and mirrored
    back into ``config['regions']`` for headless repeats.
    """
    config = dataset.config or config or {}
    requested_regions = config.get('regions', {})
    region_names = ('first_reflection', 'second_reflection')

    representative_obj = next(iter(dataset.data.values()))
    representative_time_ps = representative_obj.data[:, 0] * _S_TO_PS
    representative_amplitude = representative_obj.data[:, 1]

    resolved_regions = {}
    for region_name in region_names:
        preset_bounds = requested_regions.get(region_name)
        if preset_bounds is None:
            bounds_ps = _span_select_bounds(
                representative_time_ps, representative_amplitude,
                title=f"Select {region_name}",
            )
            print(f"[define_reflection_regions] {region_name}: selected "
                  f"[{bounds_ps[0]:.2f}, {bounds_ps[1]:.2f}] ps")
        else:
            bounds_ps = (float(preset_bounds[0]), float(preset_bounds[1]))
            print(f"[define_reflection_regions] {region_name} (preset): "
                  f"[{bounds_ps[0]:.2f}, {bounds_ps[1]:.2f}] ps")
        resolved_regions[region_name] = bounds_ps

    for filename, reflection_obj in dataset.data.items():
        reflection_obj.first_region = resolved_regions['first_reflection']
        reflection_obj.second_region = resolved_regions['second_reflection']

    config['regions'] = resolved_regions
    return resolved_regions

def generate_reflection_segments(dataset: DataSet) -> DataSet:
    """Generates the THzDataReflection object from the full trace and defined regions. Required structure for the processing pipeline."""
    regions = dataset.config.get('regions', None)
    if regions is None:
        raise ValueError("Reflection regions must be defined in the dataset config before generating segments.")
    
    for filename, data_obj in dataset.data.items():
        first_reflection = _crop_thzdata_to_gate(data_obj, *regions['first_reflection'])
        second_reflection = _crop_thzdata_to_gate(data_obj, *regions['second_reflection'])
        # from_thzdata(second, first): pass second-reflection first, first-reflection second.
        reflection_obj = THzDataReflection.from_thzdata(second_reflection, first_reflection)
        dataset.data[filename] = reflection_obj
    
    print("Segments generated for all traces based on defined reflection regions.")

    return dataset


def _plan_centered_window(
    time_seconds: np.ndarray,
    amplitude: np.ndarray,
    region_seconds: tuple,
    center_mode: str,
) -> dict:
    """Work out the apodization window for one reflection region.

    The region [start, stop] bounds the real data (everything outside is zeroed).
    The window's Hann/Tukey ALWAYS tapers to zero at its own edges, and the edges
    are chosen to land on real-data boundaries so BOTH sides taper smoothly:

    - ``'crop'``: symmetric window centred on the peak, half-width = the SHORTER
      half (min of pre/post). Tapers to zero at ``peak ± half``, both inside the
      region. Narrower, exactly symmetric about the peak (no centroid shift).
    - ``'pad'`` : the FULL region. Tapers to zero at both region edges, so it keeps
      all the region data and is smooth on both sides. Its centre is the region
      centre, so it is mildly asymmetric about the peak when the region is.

    (A window that extended *past* a region edge — the old 'pad' — left the Hann
    non-zero where the data was cut, giving a sharp step on the padded side. Keeping
    the window edges on the data boundary is what makes both sides smooth.)
    """
    region_start_seconds, region_end_seconds = region_seconds
    inside_region = (time_seconds >= region_start_seconds) & (time_seconds <= region_end_seconds)
    region_indices = np.where(inside_region)[0]
    if region_indices.size < 4:
        raise ValueError(
            f"reflection region [{region_start_seconds * _S_TO_PS:.2f}, "
            f"{region_end_seconds * _S_TO_PS:.2f}] ps selects <4 samples."
        )
    peak_index = region_indices[int(np.argmax(np.abs(amplitude[region_indices])))]
    peak_time_seconds = float(time_seconds[peak_index])

    pre_half_seconds = peak_time_seconds - region_start_seconds
    post_half_seconds = region_end_seconds - peak_time_seconds
    if center_mode == 'crop':
        half_width_seconds = min(pre_half_seconds, post_half_seconds)
        window_start_seconds = peak_time_seconds - half_width_seconds
        window_end_seconds = peak_time_seconds + half_width_seconds
    elif center_mode == 'pad':
        window_start_seconds = region_start_seconds
        window_end_seconds = region_end_seconds
        half_width_seconds = 0.5 * (window_end_seconds - window_start_seconds)
    else:
        raise ValueError("center_mode must be 'crop' or 'pad'.")

    # Never let the window reach past the acquired samples (a region can be set
    # slightly outside the truncated axis); the Hann must taper inside real data.
    window_start_seconds = max(window_start_seconds, float(time_seconds[0]))
    window_end_seconds = min(window_end_seconds, float(time_seconds[-1]))

    return dict(
        peak_time_seconds=peak_time_seconds,
        region_start_seconds=region_start_seconds,
        region_end_seconds=region_end_seconds,
        pre_half_seconds=pre_half_seconds,
        post_half_seconds=post_half_seconds,
        half_width_seconds=half_width_seconds,
        window_start_seconds=window_start_seconds,
        window_end_seconds=window_end_seconds,
        center_mode=center_mode,
    )


def _windowed_pulse_on_common_axis(
    common_time_seconds: np.ndarray,
    common_amplitude: np.ndarray,
    plan: dict,
    window_config: dict,
) -> tuple:
    """Apodize one reflection with a PEAK-ANCHORED window (smooth at both edges).

    The window is 1.0 **at the pulse peak** and tapers to 0 at ``window_start`` and
    ``window_end`` with the Hann (or Tukey) cosine taper — the same maths as
    ``thz_core.window_time``, but split at the peak so each side can taper over a
    different width. That is what keeps BOTH sides smooth while leaving the pulse at
    full weight, even when the region (and hence the window) is asymmetric about the
    peak (``pad`` mode). A symmetric window (``crop``) is the special case where the
    two sides have equal width. Returns ``(windowed_amplitude, window_function)``.
    """
    # shape = str(window_config.get('type', 'hann')).lower()
    shape = 'hann'  # force symmetric Hann for now - asymmetry is bad
    alpha = float(window_config.get('alpha', 1.0))
    peak_time = plan['peak_time_seconds']
    window_start = plan['window_start_seconds']
    window_end = plan['window_end_seconds']

    window_function = np.zeros_like(common_time_seconds)
    pre_width = max(peak_time - window_start, 1e-30)
    post_width = max(window_end - peak_time, 1e-30)

    # breakpoint()
    rising = (common_time_seconds >= window_start) & (common_time_seconds <= peak_time)
    falling = (common_time_seconds > peak_time) & (common_time_seconds <= window_end)
    rise_fraction = (common_time_seconds[rising] - window_start) / pre_width   # 0 -> 1 at peak
    fall_fraction = (common_time_seconds[falling] - peak_time) / post_width    # 0 at peak -> 1

    if shape == 'tukey' and 0.0 < alpha < 1.0:
        # Flat top over the inner (1-alpha) of each side; cosine taper over the
        # outer alpha next to each edge.
        rise_ramp = np.clip(rise_fraction / alpha, 0.0, 1.0)
        fall_ramp = np.clip((1.0 - fall_fraction) / alpha, 0.0, 1.0)
        window_function[rising] = 0.5 * (1.0 - np.cos(np.pi * rise_ramp))
        window_function[falling] = 0.5 * (1.0 - np.cos(np.pi * fall_ramp))
    else:  # hann (alpha >= 1): full cosine taper on each side
        window_function[rising] = 0.5 * (1.0 - np.cos(np.pi * rise_fraction))
        window_function[falling] = 0.5 * (1.0 + np.cos(np.pi * fall_fraction))

    return common_amplitude * window_function, window_function

def isolate_reflection_regions(
    dataset: DataSet,
    show_graph: bool = False,
) -> DataSet:
    """Zero each reflection's data outside its defined region.

    Reads ``first_region`` / ``second_region`` (ps bounds set by
    ``define_reflection_regions``) from each ``THzDataReflection`` and masks
    everything outside the respective region to zero on the shared time axis.
    This removes neighbouring pulses and instrument echoes before windowing.

    Must be called after ``build_full_trace_reflection`` and
    ``define_reflection_regions``.  Call ``window_reflection_pulses`` next to
    apply the apodization taper.
    """
    reflection_items = [
        (fn, obj) for fn, obj in dataset.data.items()
        if isinstance(obj, THzDataReflection)
    ]
    if not reflection_items:
        print("[isolate_reflection_regions] No THzDataReflection objects found, skipping.")
        return dataset

    # Validate shared time axis before touching any data.
    common_time_seconds = reflection_items[0][1].first_reflection.data[:, 0]
    for filename, reflection_obj in reflection_items:
        t = reflection_obj.first_reflection.data[:, 0]
        if t.shape != common_time_seconds.shape:
            raise ValueError(
                f"'{filename}' time axis length ({t.size}) differs from the first file "
                f"({common_time_seconds.size}); run global_truncate before isolating."
            )

    graph_records = {}
    for filename, reflection_obj in reflection_items:
        if reflection_obj.first_reflection is None or reflection_obj.second_reflection is None:
            raise ValueError(
                f"'{filename}' has no regions defined; run define_reflection_regions first."
            )

        first_holder = reflection_obj.first_reflection
        second_holder = reflection_obj.second_reflection
        time_seconds = first_holder.data[:, 0]

        first_region_s = (reflection_obj.first_reflection[0] / _S_TO_PS,
                          reflection_obj.first_reflection[1] / _S_TO_PS)
        second_region_s = (reflection_obj.second_reflection[0] / _S_TO_PS,
                           reflection_obj.second_reflection[1] / _S_TO_PS)

        first_raw = first_holder.data[:, 1].copy()
        second_raw = second_holder.data[:, 1].copy()

        first_mask = (time_seconds >= first_region_s[0]) & (time_seconds <= first_region_s[1])
        second_mask = (time_seconds >= second_region_s[0]) & (time_seconds <= second_region_s[1])

        first_isolated = first_raw * first_mask
        second_isolated = second_raw * second_mask

        first_holder.processing_dict['pre_isolation'] = first_holder.data.copy()
        second_holder.processing_dict['pre_isolation'] = second_holder.data.copy()

        first_holder.data = np.column_stack((time_seconds, first_isolated))
        second_holder.data = np.column_stack((time_seconds, second_isolated))

        print(
            f"[isolate_reflection_regions] '{filename}': "
            f"first [{reflection_obj.first_reflection[0]:.2f}, {reflection_obj.first_reflection[1]:.2f}] ps "
            f"({first_mask.sum()} samples kept); "
            f"second [{reflection_obj.second_reflection[0]:.2f}, {reflection_obj.second_reflection[1]:.2f}] ps "
            f"({second_mask.sum()} samples kept)."
        )

        graph_records[filename] = dict(
            time_seconds=time_seconds,
            first=dict(raw=first_raw, isolated=first_isolated),
            second=dict(raw=second_raw, isolated=second_isolated),
        )

    if show_graph and graph_records:
        _plot_isolated_regions(graph_records)

    return dataset


def _plot_isolated_regions(graph_records: dict) -> None:
    """Diagnostic: full trace (grey) with the isolated regions overlaid."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('isolate_reflection_regions — regions extracted from full trace')
    for filename, record in graph_records.items():
        time_ps = record['time_seconds'] * _S_TO_PS
        axes[0].plot(time_ps, record['first']['raw'], color='0.75', lw=0.8)
        axes[0].plot(time_ps, record['first']['isolated'], lw=1.3, label=filename)
        axes[1].plot(time_ps, record['second']['raw'], color='0.75', lw=0.8)
        axes[1].plot(time_ps, record['second']['isolated'], lw=1.3, label=filename)
    axes[0].set_title('First reflection')
    axes[1].set_title('Second reflection')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel('time (ps)')
    plt.show()


def window_reflection_pulses(
    dataset: DataSet,
    center_mode: str = 'crop',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Apodize both reflections of each THzDataReflection with a peak-anchored window.

    Plans and applies a Hann (or Tukey) window centred on the pulse peak for
    both the first and second reflection of every ``THzDataReflection``.  The
    window tapers to zero at both edges so both sides of the taper are smooth.

    - ``center_mode='crop'``: symmetric window centred on the peak — half-width
      is the SHORTER of pre/post peak.  No centroid shift.
    - ``center_mode='pad'``: window spans the full region.  Keeps all region
      data; mildly asymmetric about the peak when the region is.

    Reads ``first_region`` / ``second_region`` (ps) from each object.  Call
    ``isolate_reflection_regions`` first so that the peak search only sees the
    target pulse.
    """
    config = config or {}
    window_config = config.get('window', {'type': 'hann', 'alpha': 1.0})
    if center_mode not in ('crop', 'pad'):
        raise ValueError("center_mode must be 'crop' or 'pad'.")

    reflection_items = [
        (fn, obj) for fn, obj in dataset.data.items()
        if isinstance(obj, THzDataReflection)
    ]
    if not reflection_items:
        print("[window_reflection_pulses] No THzDataReflection objects found, skipping.")
        return dataset

    common_time_seconds = reflection_items[0][1].first_reflection.data[:, 0]

    graph_records = {}
    for filename, reflection_obj in reflection_items:
        if reflection_obj.first_reflection is None or reflection_obj.second_reflection is None:
            raise ValueError(
                f"'{filename}' has no regions defined; run define_reflection_regions first."
            )

        first_holder = reflection_obj.first_reflection
        second_holder = reflection_obj.second_reflection
        time_seconds = first_holder.data[:, 0]

        if time_seconds.shape != common_time_seconds.shape:
            raise ValueError(
                f"'{filename}' time axis is not aligned with the others "
                f"({time_seconds.size} vs {common_time_seconds.size} samples); "
                f"run global_truncate before windowing."
            )

        first_region_s = (reflection_obj.first_region[0] / _S_TO_PS,
                          reflection_obj.first_region[1] / _S_TO_PS)
        second_region_s = (reflection_obj.second_region[0] / _S_TO_PS,
                           reflection_obj.second_region[1] / _S_TO_PS)

        first_plan = _plan_centered_window(
            time_seconds, first_holder.data[:, 1], first_region_s, center_mode,
        )
        second_plan = _plan_centered_window(
            time_seconds, second_holder.data[:, 1], second_region_s, center_mode,
        )

        first_amplitude = first_holder.data[:, 1].copy()
        second_amplitude = second_holder.data[:, 1].copy()

        first_windowed, first_window_fn = _windowed_pulse_on_common_axis(
            time_seconds, first_amplitude, first_plan, window_config,
        )
        second_windowed, second_window_fn = _windowed_pulse_on_common_axis(
            time_seconds, second_amplitude, second_plan, window_config,
        )

        first_holder.data = np.column_stack((time_seconds, first_windowed))
        second_holder.data = np.column_stack((time_seconds, second_windowed))
        first_holder.processing_dict['window_plan'] = first_plan
        second_holder.processing_dict['window_plan'] = second_plan

        _print_isolate_decision(filename, 'first_reflection', first_plan, window_config)
        _print_isolate_decision(filename, 'second_reflection', second_plan, window_config)

        graph_records[filename] = dict(
            time_seconds=time_seconds,
            first=dict(pre_window=first_amplitude, windowed=first_windowed,
                       window_fn=first_window_fn),
            second=dict(pre_window=second_amplitude, windowed=second_windowed,
                        window_fn=second_window_fn),
        )

    if show_graph and graph_records:
        _plot_windowed_pulses(graph_records)

    return dataset


def _plot_windowed_pulses(graph_records: dict) -> None:
    """Diagnostic: windowed pulses with their window functions on a sample-index axis."""
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('window_reflection_pulses — apodized pulses (sample index)')
    for filename, record in graph_records.items():
        sample_index = np.arange(record['time_seconds'].size)
        peak_amp = max(
            np.max(np.abs(record['first']['windowed'])),
            np.max(np.abs(record['second']['windowed'])),
            1e-30,
        )
        axes[0].plot(sample_index, record['first']['windowed'], lw=1.3, label=filename)
        axes[0].plot(sample_index, record['first']['window_fn'] * peak_amp,
                     '--', lw=1, alpha=0.5)
        axes[1].plot(sample_index, record['second']['windowed'], lw=1.3, label=filename)
        axes[1].plot(sample_index, record['second']['window_fn'] * peak_amp,
                     '--', lw=1, alpha=0.5)
    axes[0].set_title('First reflection')
    axes[1].set_title('Second reflection')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel('sample index')
    plt.show()




def isolate_and_window(
    dataset: DataSet,
    config: dict | None = None,
    center_mode: str = 'crop',
    show_graph: bool = False,
) -> DataSet:
    """Isolate + apodize BOTH reflections on one shared axis (the coupling step).

    Kept for backwards compatibility.  For explicit, inspectable pipelines call
    ``isolate_reflection_regions`` then ``window_reflection_pulses`` instead.

    Replaces ``centering_manual`` + ``center_pulse`` + per-segment windowing.
    Reads each object's ``first_region`` / ``second_region`` (set by
    ``define_reflection_regions``) and apodizes each reflection in place on the
    shared axis. The window always tapers to zero at its own edges, and the edges
    sit on real-data boundaries so BOTH sides are smooth:

    - ``center_mode='crop'``: symmetric Hann centred on the peak (half-width = the
      shorter of pre/post). Narrower, exactly symmetric — no centroid shift.
    - ``center_mode='pad'`` : Hann over the FULL region (tapers to zero at both
      region edges). Keeps all the region data; mildly asymmetric about the peak
      when the region is.

    Both reflections share one axis, so their inter-pulse phase is structural (no
    common-grid step). Called ONCE (both regions together).

    Decisions are printed per file/region. With ``show_graph=True`` two figures
    open: (1) the windowed pulses on the shared time axis, and (2) the pulses with
    their window functions on a sample-index axis — each as first/second subplots.
    """
    config = config or {}
    window_config = config.get('window', {'type': 'hann', 'alpha': 1.0})
    if center_mode not in ('crop', 'pad'):
        raise ValueError("center_mode must be 'crop' or 'pad'.")

    reflection_items = [
        (filename, obj) for filename, obj in dataset.data.items()
        if isinstance(obj, THzDataReflection)
    ]
    for filename, obj in dataset.data.items():
        if not isinstance(obj, THzDataReflection):
            print(f"[isolate_and_window] '{filename}': not a THzDataReflection, skipping.")
    if not reflection_items:
        return dataset

    # --- Pass 1: plan every reflection's centred window (per file) ---
    # The window placement is per-pulse, but the OUTPUT axis must be common to ALL
    # files (so the FFT frequency grids match for the transfer-function ratio). So
    # we first gather every window's extent, then build ONE shared axis below.
    window_plans = {}
    for filename, reflection_obj in reflection_items:
        if reflection_obj.first_region is None or reflection_obj.second_region is None:
            raise ValueError(
                f"'{filename}' has no regions defined; run define_reflection_regions first."
            )
        shared_time_seconds = reflection_obj.first_reflection.data[:, 0]
        first_region_seconds = (reflection_obj.first_region[0] / _S_TO_PS,
                                reflection_obj.first_region[1] / _S_TO_PS)
        second_region_seconds = (reflection_obj.second_region[0] / _S_TO_PS,
                                 reflection_obj.second_region[1] / _S_TO_PS)
        window_plans[filename] = (
            _plan_centered_window(shared_time_seconds, reflection_obj.first_reflection.data[:, 1],
                                  first_region_seconds, center_mode),
            _plan_centered_window(shared_time_seconds, reflection_obj.data[:, 1],
                                  second_region_seconds, center_mode),
        )

    # All files already share one axis (global_truncate trimmed them to a common
    # length), and every window now stays inside the region (inside that axis), so
    # we window IN PLACE — no axis extension, no padding zeros, no common-grid step.
    common_time_seconds = reflection_items[0][1].first_reflection.data[:, 0]

    graph_records = {}
    for filename, reflection_obj in reflection_items:
        first_plan, second_plan = window_plans[filename]
        first_holder = reflection_obj.first_reflection
        second_holder = reflection_obj.second_reflection

        if first_holder.data[:, 0].shape != common_time_seconds.shape:
            raise ValueError(
                f"'{filename}' time axis is not aligned with the others "
                f"({first_holder.data.shape[0]} vs {common_time_seconds.size} samples); "
                f"run global_truncate so all files share one axis before isolate_and_window."
            )

        def isolate_to_region(amplitude, plan):
            # Zero everything outside the region: the region is the only echo /
            # neighbour-pulse filter (the window also tapers to zero at the region
            # edges, so this is belt-and-braces and keeps the intent explicit).
            in_region = ((common_time_seconds >= plan['region_start_seconds'])
                         & (common_time_seconds <= plan['region_end_seconds']))
            return amplitude * in_region

        first_full_amplitude = first_holder.data[:, 1].copy()
        second_full_amplitude = second_holder.data[:, 1].copy()
        first_region_isolated = isolate_to_region(first_full_amplitude, first_plan)
        second_region_isolated = isolate_to_region(second_full_amplitude, second_plan)

        # plt.plot(common_time_seconds * _S_TO_PS, first_region_isolated, label='first isolated')
        # plt.plot(common_time_seconds * _S_TO_PS, second_region_isolated, label='second isolated')
        # plt.legend()
        # plt.show()

        first_windowed, first_window_function = _windowed_pulse_on_common_axis(
            common_time_seconds, first_region_isolated, first_plan, window_config)
        second_windowed, second_window_function = _windowed_pulse_on_common_axis(
            common_time_seconds, second_region_isolated, second_plan, window_config)

        first_holder.data = np.column_stack((common_time_seconds, first_windowed))
        second_holder.data = np.column_stack((common_time_seconds, second_windowed))
        first_holder.processing_dict['isolate_window_plan'] = first_plan
        second_holder.processing_dict['isolate_window_plan'] = second_plan

        _print_isolate_decision(filename, 'first_reflection', first_plan, window_config)
        _print_isolate_decision(filename, 'second_reflection', second_plan, window_config)

        is_reference = False
        try:
            is_reference = dataset.data.is_reference(filename)
        except Exception:
            pass
        graph_records[filename] = dict(
            common_time_seconds=common_time_seconds, is_reference=is_reference,
            first=dict(raw=first_full_amplitude, windowed=first_windowed,
                       window_function=first_window_function),
            second=dict(raw=second_full_amplitude, windowed=second_windowed,
                        window_function=second_window_function),
        )

    if show_graph and graph_records:
        _plot_isolate_centering(graph_records)
        _plot_isolate_windowing(graph_records)
    return dataset


def _print_isolate_decision(filename: str, segment_name: str, plan: dict, window_config: dict) -> None:
    """Report exactly how one reflection was centred and windowed."""
    window_kind = ('symmetric about peak' if plan['center_mode'] == 'crop'
                   else 'full region')
    print(
        f"[isolate_and_window] '{filename}' {segment_name}: "
        f"peak {plan['peak_time_seconds'] * _S_TO_PS:.2f} ps; "
        f"region [{plan['region_start_seconds'] * _S_TO_PS:.2f}, "
        f"{plan['region_end_seconds'] * _S_TO_PS:.2f}] ps; "
        f"pre/post half {plan['pre_half_seconds'] * _S_TO_PS:.2f}/"
        f"{plan['post_half_seconds'] * _S_TO_PS:.2f} ps -> "
        f"mode={plan['center_mode']} ({window_kind}) -> window "
        f"[{plan['window_start_seconds'] * _S_TO_PS:.2f}, "
        f"{plan['window_end_seconds'] * _S_TO_PS:.2f}] ps "
        f"(half-width {plan['half_width_seconds'] * _S_TO_PS:.2f} ps, tapers to 0 at both edges); "
        f"window={window_config.get('type', 'hann')} alpha={window_config.get('alpha', 1.0)}"
    )


def _plot_isolate_centering(graph_records: dict) -> None:
    """Figure 1: centred & tapered pulses (both reflections) on the shared TIME axis."""
    file_count = len(graph_records)
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('isolate_and_window — centred & tapered pulses (shared time axis)')
    for filename, record in graph_records.items():
        time_ps = record['common_time_seconds'] * _S_TO_PS
        axes[0].plot(time_ps, record['first']['windowed'], lw=1.3, label=f'{filename} first')
        axes[1].plot(time_ps, record['second']['windowed'], lw=1.3, label=f'{filename} second')
    plt.xlabel('time (ps)')
    plt.ylabel('amplitude')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.show()


def _plot_isolate_windowing(graph_records: dict) -> None:
    """Figure 2: windowed pulses + window functions on a sample-INDEX axis.

    Same layout as the centering figure — first reflection (top) and second
    reflection (bottom), every file overlaid — with each file's window function
    drawn dashed (scaled to the pulse amplitude) so the apodization is visible.
    """
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('isolate_and_window — pulses and window functions (sample index)')
    for filename, record in graph_records.items():
        sample_index = np.arange(record['common_time_seconds'].size)
        peak_amplitude = max(
            np.max(np.abs(record['first']['windowed'])),
            np.max(np.abs(record['second']['windowed'])),
            1e-30,
        )
        axes[0].plot(sample_index, record['first']['windowed'], lw=1.3, label=f'{filename} first')
        axes[0].plot(sample_index, record['first']['window_function'] * peak_amplitude,
                     '--', lw=1, alpha=0.6)
        axes[1].plot(sample_index, record['second']['windowed'], lw=1.3, label=f'{filename} second')
        axes[1].plot(sample_index, record['second']['window_function'] * peak_amplitude,
                     '--', lw=1, alpha=0.6)
    axes[1].set_xlabel('sample index')
    axes[0].set_ylabel('amplitude')
    axes[1].set_ylabel('amplitude')
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    plt.show()


# ---------------------------------------------------------------------------
# Fixed-width symmetric window (prototype / low-level path)
#
# Simpler, more transparent alternative to isolate_and_window. The window width
# is a single number (half_width_ps) shared by EVERY pulse, so the per-bin
# apodization weighting is identical for every file and every reflection — the
# region is used ONLY to locate the pulse (peak search), never to size the
# window. The data is never time-shifted: the window is centred on each pulse's
# peak on the existing shared axis and everything outside is zeroed, so the
# first<->second inter-pulse delay and the inter-file timing (set upstream by
# align_to_reference) are preserved exactly.
# ---------------------------------------------------------------------------


def _build_symmetric_window(length_samples: int, window_type: str, alpha: float) -> np.ndarray:
    """One symmetric apodization window of exactly ``length_samples`` samples.

    Mirrors the window shapes thz_core.window_time offers (hann / tukey / boxcar)
    so the prototype path stays consistent with the rest of the pipeline, but the
    LENGTH is fixed by the caller (not the gate). 'hann' is the default and the
    expression below is identical to ``scipy.signal.windows.hann(length_samples)``.
    """
    window_type = str(window_type).lower()
    if window_type == 'boxcar':
        return np.ones(length_samples)
    if window_type == 'tukey':
        from scipy.signal.windows import tukey
        return tukey(length_samples, alpha=alpha)
    if window_type != 'hann':
        raise ValueError("window type must be 'hann', 'tukey', or 'boxcar'.")
    sample_index = np.arange(length_samples)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * sample_index / (length_samples - 1)))


def window_pulses_fixed_width(
    dataset: DataSet,
    half_width_ps: float,
    config: dict | None = None,
    segments: tuple = ('first_reflection', 'second_reflection'),
    show_graph: bool = False,
) -> DataSet:
    """Apply ONE identical fixed-width symmetric window to every reflection pulse.

    For each segment of each ``THzDataReflection`` (already on a common shared axis
    from ``build_full_trace_reflection``), the pulse peak is located inside its
    region, then a symmetric Hann of a FIXED length — ``2*N+1`` samples where
    ``N = round(half_width_ps / dt)`` — is centred on that peak and everything
    outside is zeroed. The window length is the same for every pulse, so the
    apodization weighting is byte-identical across all files and both reflections;
    the only thing that differs under the envelope is the data.

    The data is NOT shifted: the window rides the existing time axis, so the
    inter-pulse delay and inter-file timing are preserved (no phase bookkeeping).
    The region bounds (``first_region`` / ``second_region``) are used only to
    constrain the peak search.

    Parameters
    ----------
    half_width_ps : float
        Half-width of the window in picoseconds (single value for all pulses).
    config : dict, optional
        ``config['window']`` keys ``type`` ('hann' default / 'tukey' / 'boxcar')
        and ``alpha`` (tukey only).
    segments : tuple
        Which segments to window. Defaults to both reflections.

    Notes
    -----
    Original data saved under ``processing_dict['pre_fixed_window']``; the window
    plan under ``processing_dict['fixed_window']``. If a pulse sits too close to a
    trace edge for the full half-width, the overlapping bins still use the
    identical window weights and a clip warning is printed (shrink ``half_width_ps``).
    """
    config = config or {}
    window_config = config.get('window', {'type': 'hann', 'alpha': 1.0})
    window_type = window_config.get('type', 'hann')
    alpha = float(window_config.get('alpha', 1.0))

    reflection_items = [
        (filename, obj) for filename, obj in dataset.data.items()
        if isinstance(obj, THzDataReflection)
    ]
    if not reflection_items:
        print("[window_pulses_fixed_width] No THzDataReflection objects found, skipping.")
        return dataset

    # One dt for the whole dataset -> one fixed sample count -> one window length.
    reference_time_seconds = reflection_items[0][1].first_reflection.data[:, 0]
    dt_seconds = float(np.median(np.diff(reference_time_seconds)))
    half_width_samples = int(round(half_width_ps * 1e-12 / dt_seconds))
    if half_width_samples < 2:
        raise ValueError(
            f"half_width_ps={half_width_ps} is < 2 samples at dt={dt_seconds * _S_TO_PS:.4f} ps."
        )
    window_length = 2 * half_width_samples + 1
    window_core = _build_symmetric_window(window_length, window_type, alpha)

    graph_records = {}
    applied_lengths = []
    for filename, reflection_obj in reflection_items:
        record = {}
        for segment in segments:
            holder = getattr(reflection_obj, segment)
            region_ps = getattr(reflection_obj, segment.replace('reflection', 'region'))
            if holder is None or region_ps is None:
                raise ValueError(
                    f"'{filename}' {segment}: holder/region missing; "
                    f"run build_full_trace_reflection + define_reflection_regions first."
                )

            time_seconds = holder.data[:, 0]
            amplitude = holder.data[:, 1]
            n_samples = amplitude.size
            if time_seconds.shape != reference_time_seconds.shape:
                raise ValueError(
                    f"'{filename}' {segment} axis ({n_samples}) differs from the first file "
                    f"({reference_time_seconds.size}); all files must share one axis."
                )

            region_seconds = (region_ps[0] / _S_TO_PS, region_ps[1] / _S_TO_PS)
            in_region = (time_seconds >= region_seconds[0]) & (time_seconds <= region_seconds[1])
            region_indices = np.where(in_region)[0]
            if region_indices.size < 4:
                raise ValueError(
                    f"'{filename}' {segment}: region [{region_ps[0]:.2f}, {region_ps[1]:.2f}] ps "
                    f"selects < 4 samples."
                )
            peak_index = region_indices[int(np.argmax(np.abs(amplitude[region_indices])))]

            # Drop the identical window_core at [peak-N, peak+N]; clip to the axis
            # but keep matching window weights on the bins that do exist.
            lo = peak_index - half_width_samples
            hi = peak_index + half_width_samples + 1
            data_lo, data_hi = max(lo, 0), min(hi, n_samples)
            core_lo = data_lo - lo
            core_hi = core_lo + (data_hi - data_lo)
            window_function = np.zeros(n_samples)
            window_function[data_lo:data_hi] = window_core[core_lo:core_hi]
            windowed = amplitude * window_function
            clipped = (lo < 0) or (hi > n_samples)

            holder.processing_dict['pre_fixed_window'] = holder.data.copy()
            holder.processing_dict['fixed_window'] = dict(
                peak_index=int(peak_index),
                peak_time_seconds=float(time_seconds[peak_index]),
                half_width_samples=half_width_samples,
                window_length=window_length,
                clipped=bool(clipped),
            )
            holder.data = np.column_stack((time_seconds, windowed))
            applied_lengths.append(window_length)

            clip_note = "  *** CLIPPED at trace edge — reduce half_width_ps ***" if clipped else ""
            print(
                f"[window_pulses_fixed_width] '{filename}' {segment}: "
                f"peak {time_seconds[peak_index] * _S_TO_PS:.2f} ps, "
                f"window {window_length} samples "
                f"(+/- {half_width_samples * dt_seconds * _S_TO_PS:.2f} ps).{clip_note}"
            )
            record[segment] = dict(time_seconds=time_seconds, windowed=windowed,
                                   window_function=window_function,
                                   peak_index=int(peak_index))
        graph_records[filename] = record

    identical = len(set(applied_lengths)) == 1
    print(
        f"[window_pulses_fixed_width] applied an IDENTICAL {window_type} window of "
        f"{window_length} samples to {len(applied_lengths)} pulses across "
        f"{len(reflection_items)} files (uniform length: {identical})."
    )

    if show_graph and graph_records:
        _plot_fixed_width_window(graph_records, segments)
    return dataset


def _plot_fixed_width_window(graph_records: dict, segments: tuple) -> None:
    """Overlay every windowed pulse + its (identical) window function per segment."""
    fig, axes = plt.subplots(len(segments), 1, sharex=True, squeeze=False)
    fig.suptitle('window_pulses_fixed_width — identical fixed-width Hann on every pulse')
    axes = axes[:, 0]
    for axis, segment in zip(axes, segments):
        peak_amplitude = 1e-30
        for record in graph_records.values():
            if segment in record:
                peak_amplitude = max(peak_amplitude, np.max(np.abs(record[segment]['windowed'])))
        for filename, record in graph_records.items():
            if segment not in record:
                continue
            time_ps = record[segment]['time_seconds'] * _S_TO_PS
            axis.plot(time_ps, record[segment]['windowed'], lw=1.3, label=filename)
            axis.plot(time_ps, record[segment]['window_function'] * peak_amplitude,
                      '--', lw=0.9, alpha=0.5)
        axis.set_title(segment)
        axis.set_ylabel('amplitude')
        axis.legend(fontsize=7)
    axes[-1].set_xlabel('time (ps)')
    plt.show()


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


def _averaged_as_single_scan(data_obj) -> np.ndarray:
    """Build a 1-column ``[time_s, mean]`` matrix from the averaged trace.

    The safe fallback when no per-scan matrix is available or the cached one has
    gone stale: the per-scan scatter is lost but the averaged pipeline result is
    preserved exactly.
    """
    averaged = np.asarray(data_obj.data, dtype=float)
    return np.column_stack((averaged[:, 0], averaged[:, 1]))


def _ensure_scan_matrix(data_obj) -> np.ndarray:
    """Return the per-scan working matrix ``[time_s, scan1, ..., scanN]``.

    Created lazily from ``raw_data`` (source ps time converted to SI seconds to
    match ``data_obj.data``) on first use and cached in ``processing_dict``.
    Subsequent calls return the same array so in-place transforms by the linear
    preprocessing steps (baseline, alignment, normalise) accumulate.

    **Staleness guard (single source of truth).** The matrix is only valid while
    it has the *same row count* as ``data_obj.data``. Steps that change the row
    count or resample the time axis (``global_truncate``, ``center_pulse``, and
    the windowing-hygiene crop in ``centering_manual``) mutate ``data`` but
    deliberately do **not** maintain this parallel matrix — keeping two full
    representations in lockstep through every step is the coupling we want to
    avoid. So instead of trusting the cache blindly, every call verifies the row
    count and, on a mismatch (or when ``raw_data`` is single-scan / absent),
    falls back to the averaged trace as a single "scan". This protects **all**
    consumers in one place rather than each re-checking.
    """
    target_rows = np.asarray(data_obj.data).shape[0]

    matrix = data_obj.processing_dict.get(_SCAN_MATRIX_KEY)
    if matrix is not None:
        if np.asarray(matrix).shape[0] == target_rows:
            return matrix
        # Cached matrix went stale: a row-count-changing step ran since it was
        # built. Drop the per-scan scatter and fall back to the averaged trace.
        if not data_obj.processing_dict.get('_scan_matrix_stale_warned'):
            print(
                f"Note: '{getattr(data_obj, 'filename', '<unknown>')}' per-scan "
                f"matrix is out of sync with the averaged trace "
                f"({np.asarray(matrix).shape[0]} vs {target_rows} rows); a "
                f"row-count-changing step ran. Using the averaged trace as a "
                f"single scan from here."
            )
            data_obj.processing_dict['_scan_matrix_stale_warned'] = True
        matrix = _averaged_as_single_scan(data_obj)
        data_obj.processing_dict[_SCAN_MATRIX_KEY] = matrix
        return matrix

    raw = np.asarray(data_obj.raw_data, dtype=float)
    if raw.ndim == 2 and raw.shape[1] >= 2 and raw.shape[0] == target_rows:
        time_seconds = raw[:, 0] / _S_TO_PS  # ps -> s, matching data_obj.data
        matrix = np.column_stack((time_seconds, raw[:, 1:]))
    else:
        # No multi-scan raw_data, or it no longer matches the (already-mutated)
        # averaged trace — fall back to the averaged trace as a single scan.
        matrix = _averaged_as_single_scan(data_obj)
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

def _subtract_baseline_reflection(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Specific pipeline step to subtract baseline from both reflections in a THzDataReflection."""
    config = config or {}
    n_points = int((config.get('baseline', {}) or {}).get('n_points', 10))

    show_graph = bool(config.get('show_graph', False))

    flattened_data_dict = {}
    for filename, data_obj in dataset.data.items():
        first_holder = _resolve_segment(data_obj, 'first_reflection')
        second_holder = _resolve_segment(data_obj, 'second_reflection')
        if first_holder is None or second_holder is None:
            continue
        flattened_data_dict[f"{filename}__first"] = first_holder.data
        flattened_data_dict[f"{filename}__second"] = second_holder.data

    corrected, metrics = core.subtract_baseline(flattened_data_dict, config)

    if show_graph:
        fig, ax = plt.subplots(2, 1, layout='constrained')
        for filename, data in corrected.items():
            basename, segment = filename.split('__')
            ax[1].plot(data[:, 1], label=f'{filename} (corrected)')
            ax[0].plot(flattened_data_dict[filename][:, 1], label=f'{filename} (original)',
                       linestyle='--', lw=1, alpha=0.5)
        ax[0].set_title(f'Original Traces — {segment} reflection')
        ax[1].set_title('Baseline-Corrected Traces')
        ax[1].set_xlabel('Time Point Index')
        plt.show()

    for filename, data_obj in dataset.data.items():
        first_name = filename + '__first'
        data_obj.first_reflection.data = corrected[first_name]
        second_name = filename + '__second'
        data_obj.second_reflection.data = corrected[second_name]

        data_obj.processing_dict['baseline_metrics_first'] = metrics
        # Mirror baseline subtraction onto the per-scan matrix so the
        # per-scan working data stays consistent with the averaged trace.
        # (first_reflection is already an averaged trace; no per-scan matrix.)
        scan_matrix = _ensure_scan_matrix(data_obj)
        scan_columns = scan_matrix[:, 1:]
        scan_columns -= scan_columns[:n_points, :].mean(axis=0, keepdims=True)

    return dataset

def subtract_baseline(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Subtract baseline from data, depending on the type of data object. For THzDataReflection, it processes both reflections."""

    config = dataset.config or config or {}
    n_points = int((config.get('baseline', {}) or {}).get('n_points', 10))

    show_graph = bool(config.get('show_graph', False))

    if all(isinstance(data_obj, THzDataReflection) for data_obj in dataset.data.values()):
        baselined_dataset = _subtract_baseline_reflection(dataset, config)
    elif all(isinstance(data_obj, THzData) for data_obj in dataset.data.values()):
        baselined_dataset = _subtract_baseline(dataset, segment='second_reflection', config=config, show_graph=show_graph)
    
    else:
        print("Warning: Dataset contains mixed or unrecognized data object types. No baseline subtraction applied.")
        baselined_dataset = dataset
    return baselined_dataset


def _subtract_baseline(
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
    config = dataset.config or config or {}
    n_points = int((config.get('baseline', {}) or {}).get('n_points', 10))

    show_graph = show_graph or bool(config.get('show_graph', False))

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
            # (first_reflection is already an averaged trace; no per-scan matrix.)
            scan_matrix = _ensure_scan_matrix(data_obj)
            scan_columns = scan_matrix[:, 1:]
            scan_columns -= scan_columns[:n_points, :].mean(axis=0, keepdims=True)

    return dataset


def pad_trace_start(
    dataset: DataSet,
    segment: str = 'second_reflection',
    extension_ps: float = 3.0,
    taper_ps: float = 1.0,
    enabled: bool = True,
    show_graph: bool = False,
) -> DataSet:
    """Extend the START of each trace backwards with zeros (smoothly tapered).

    Creates a baseline-zero region before the data so a later symmetric window can
    reach back without cutting real signal (e.g. the first reflection's limited
    pre-pulse). For each trace: ramp the leading ``taper_ps`` of real data up from
    zero (rising half-cosine, so the zero->data junction is smooth), then prepend
    ``extension_ps`` of zeros. Real samples keep their absolute times; only the axis
    is extended earlier. Run AFTER ``subtract_baseline``.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which segment to process. Call twice (same args) to keep both segments of a
        ``THzDataReflection`` on one shared axis.
    extension_ps : float
        Length (ps) of the prepended zero region.
    taper_ps : float
        Leading real data (ps) ramped 0->1 to smooth the junction.
    enabled : bool
        One-line on/off — returns the dataset unchanged when False.
    show_graph : bool
        Plot the extended traces (zeros + taper shaded) and a step-change metric to
        spot a poor taper / junction discontinuity at a glance.
    """
    if not enabled:
        return dataset

    pad_records = {}
    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None:
            continue

        time_seconds = holder.data[:, 0]
        other_columns = holder.data[:, 1:]  # amplitude (+ stderr if present)
        sample_interval_seconds = float(np.median(np.diff(time_seconds)))
        zero_count = int(round(extension_ps * 1e-12 / sample_interval_seconds))
        taper_count = min(int(round(taper_ps * 1e-12 / sample_interval_seconds)),
                          other_columns.shape[0])

        tapered_columns = other_columns.copy()
        if taper_count > 0:
            rising_ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_count, endpoint=False)))
            tapered_columns[:taper_count, 0] *= rising_ramp  # taper the amplitude only

        earlier_times = time_seconds[0] - sample_interval_seconds * np.arange(zero_count, 0, -1)
        new_time = np.concatenate([earlier_times, time_seconds])
        new_columns = np.concatenate(
            [np.zeros((zero_count, other_columns.shape[1])), tapered_columns], axis=0)

        holder.processing_dict['pre_pad_start'] = holder.data.copy()
        holder.data = np.column_stack([new_time, new_columns])

        print(f"[pad_trace_start/{segment}] '{filename}': prepended {zero_count} zero "
              f"samples ({zero_count * sample_interval_seconds * _S_TO_PS:.2f} ps); "
              f"tapered leading {taper_count} samples "
              f"({taper_count * sample_interval_seconds * _S_TO_PS:.2f} ps).")

        pad_records[filename] = dict(
            time_seconds=new_time, amplitude=holder.data[:, 1],
            zero_count=zero_count, taper_count=taper_count,
            sample_interval_seconds=sample_interval_seconds,
        )

    if show_graph and pad_records:
        _plot_pad_start(pad_records, segment)
    return dataset


def _plot_pad_start(pad_records: dict, segment: str) -> None:
    """Diagnostic for pad_trace_start: extended traces + a step-change metric.

    Top: padded traces (prepended-zeros and taper regions shaded). Bottom: the
    first difference normalised to the peak amplitude — ``(y[i]-y[i-1]) / max|y|`` —
    which reads directly as "the step as a fraction of the pulse height". A smooth
    taper stays small and flat through the junction; a poor taper / discontinuity
    spikes. A dashed reference line marks the typical steepest *in-pulse* slope, so
    anything in the junction poking above it is sharper than any real feature.

    (The literal "normalise to the previous step" idea is omitted on purpose: the
    prepended region is zeros, so the previous step is ~0 there and the ratio blows
    up across the whole flat region — drowning the one junction you want to see.)
    """
    figure, (trace_axis, step_axis) = plt.subplots(2, 1, sharex=True)
    figure.suptitle(f'pad_trace_start — extension & step-change check ({segment})')

    in_pulse_slope_fractions = []
    for filename, record in pad_records.items():
        time_ps = record['time_seconds'] * _S_TO_PS
        amplitude = record['amplitude']
        peak_amplitude = max(np.max(np.abs(amplitude)), 1e-30)
        junction_end = record['zero_count'] + record['taper_count']

        trace_axis.plot(time_ps, amplitude, lw=1.2, label=filename)

        step_fraction = np.diff(amplitude) / peak_amplitude
        step_axis.plot(time_ps[1:], step_fraction, lw=1.0)

        # steepest legitimate slope = max |step| well past the junction (real pulse)
        past_junction = np.abs(step_fraction[junction_end + 2:])
        if past_junction.size:
            in_pulse_slope_fractions.append(float(np.max(past_junction)))

        junction_step = float(np.max(np.abs(step_fraction[:junction_end + 2]))) \
            if junction_end + 2 <= step_fraction.size else float('nan')
        print(f"[pad_trace_start/{segment}] '{filename}': max junction step "
              f"{junction_step * 100:.2f}% of peak.")

    # shade the prepended-zeros and taper regions using a representative record
    representative = next(iter(pad_records.values()))
    rep_time_ps = representative['time_seconds'] * _S_TO_PS
    zero_count = representative['zero_count']
    taper_count = representative['taper_count']
    if zero_count > 0:
        trace_axis.axvspan(rep_time_ps[0], rep_time_ps[zero_count - 1],
                           alpha=0.15, color='tab:blue', label='prepended zeros')
    if taper_count > 0:
        trace_axis.axvspan(rep_time_ps[zero_count], rep_time_ps[zero_count + taper_count - 1],
                           alpha=0.25, color='tab:orange', label='taper')

    if in_pulse_slope_fractions:
        reference = float(np.median(in_pulse_slope_fractions))
        step_axis.axhline(reference, color='0.4', ls='--', lw=1,
                          label='typical steepest in-pulse slope')
        step_axis.axhline(-reference, color='0.4', ls='--', lw=1)

    trace_axis.set_ylabel('amplitude')
    trace_axis.legend(fontsize=7)
    step_axis.set_ylabel('(y[i]-y[i-1]) / max|y|')
    step_axis.set_xlabel('time (ps)')
    step_axis.legend(fontsize=7)
    plt.show()


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

    # Mask by value first, then trim ALL holders to the common minimum length. A
    # float boundary sample can land in some traces but not others (off-by-one),
    # which would leave files on slightly different-length axes; trimming to the
    # shortest common count guarantees one identical axis for every file.
    masked = []
    for fn, h in holders:
        kept_indices = np.where((h.data[:, 0] >= min_t) & (h.data[:, 0] <= max_t))[0]
        masked.append((fn, h, kept_indices))
    target_length = min(len(kept_indices) for _, _, kept_indices in masked)

    for fn, h, kept_indices in masked:
        h.data = h.data[kept_indices[:target_length], :]
        if show_graph:
            plt.plot(h.data[:, 0] * _S_TO_PS, h.data[:, 1], label=fn)

    print(f"[global_truncate] {segment}: truncated to common time range "
          f"({target_length} samples).")
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


def centering_manual(
    dataset: DataSet,
    segment: str = 'second_reflection',
    show_graph: bool = False,
    auto_range_ps: tuple | None = None,
    recalibrate: bool = False,
) -> DataSet:
    """Align all traces so that the center of the main pulse peak sits at the same time-scan index (with respect to the edges of the scan). The data is cropped to the common shared extent, so that windowing is performed exactly the same for all files.

    Used before windowing so the window function lands at the same T0 distance
    from the edge for every file.

    NOTE: This step does not affect the absolute time axis (the first column of each trace) — it only shifts the data in the second column to align the peaks. The time axis is preserved, so that later steps can still use the original time information.

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

    # if recalibrate:
    #     print("Recalibrating time axis for all files to share same T0, i.e. no instrumental timing offset remains.")
    #     print(" Select which file's time axis to use:")
    #     while True:
    #         for i, filename in enumerate(aligned.keys()):
    #             print(f"  {i}: {filename}")
    #         selection = input(f"Enter a number from 0 to {len(aligned)-1}: ")
    #         try:            
    #             idx = int(selection)
    #             if idx < 0 or idx >= len(aligned):
    #                 print("Must be a valid number from the list.")
    #                 continue
    #             selected_filename = list(aligned.keys())[idx]
    #             break
    #         except ValueError:
    #             print("Must be a valid number from the list.")
    #     print(f"Selected '{selected_filename}' as the T0 reference. Recalibrating all files to share its time axis.")

    #     ref_data = aligned[selected_filename]
    #     ref_axis = ref_data[:, 0]

    #     for filename, data in aligned.items():
    #         if filename == selected_filename:
    #             continue
    #         aligned[filename] = np.column_stack((ref_axis, data[:, 1]))

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

    Centres by padding the shorter side: prepend zeros if the peak is in the first
    half, append zeros if it is past the midpoint. If the peak is already at the
    midpoint, t_new and y_new are copies of the inputs and info['already_centred']
    is True.
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
    # Centre the peak at the array midpoint by padding the SHORTER side with zeros.
    # Peak in the first half (n_after > n_before) -> pad the FRONT; peak past the
    # midpoint (n_before > n_after) -> pad the BACK. Exactly one is non-zero (or both
    # zero when the peak is already at the midpoint), so we only ever pad one side.
    n_prepend = max(0, n_after - n_before)
    n_append = max(0, n_before - n_after)

    info = {
        'already_centred': (n_prepend == 0 and n_append == 0),
        'n_prepend': n_prepend,
        'n_append': n_append,
        'peak_idx_original': peak_idx,
        'peak_idx_new': n_prepend + peak_idx,
        'taper_samples': 0,
        'dt_s': dt,
    }

    if n_prepend == 0 and n_append == 0:
        return t.copy(), y.copy(), info

    taper_samples = int(round(taper_ps * 1e-12 / dt))

    t_prepend = t[0] - np.arange(n_prepend, 0, -1) * dt
    t_append = t[-1] + np.arange(1, n_append + 1) * dt
    t_new = np.concatenate([t_prepend, t, t_append])
    y_new = np.concatenate([np.zeros(n_prepend), y.copy(), np.zeros(n_append)])

    # Half-cosine taper at the single padded junction so the pad<->signal step is
    # smooth: a RISING ramp over the leading pre-pulse samples when padding the front,
    # a FALLING ramp over the trailing post-pulse samples when padding the back. The
    # ramp is clamped to the pre/post-peak sample count so it never crosses the peak.
    if n_prepend > 0:
        rise_count = min(taper_samples, n_before)
        if rise_count > 0:
            rising_ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, rise_count, endpoint=False)))
            y_new[n_prepend : n_prepend + rise_count] *= rising_ramp
        info['taper_samples'] = rise_count
    elif n_append > 0:
        fall_count = min(taper_samples, n_after)
        if fall_count > 0:
            signal_end_index = n_prepend + len(y)
            falling_ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fall_count, endpoint=False)))
            y_new[signal_end_index - fall_count : signal_end_index] *= falling_ramp
        info['taper_samples'] = fall_count

    return t_new, y_new, info


def _plot_centering_result(filename: str, pre_arr: np.ndarray, new_arr: np.ndarray, info: dict) -> None:
    """Two-panel centering diagnostic: full trace overlay + zoom on the padded junction."""
    t_orig_ps = pre_arr[:, 0] * _S_TO_PS
    y_orig = pre_arr[:, 1]
    t_new_ps = new_arr[:, 0] * _S_TO_PS
    y_new = new_arr[:, 1]

    n_prepend = info['n_prepend']
    n_append = info['n_append']
    dt_ps = info['dt_s'] * _S_TO_PS
    peak_ps = t_new_ps[info['peak_idx_new']]
    signal_end_index = len(y_new) - n_append  # first appended-zero sample

    # Zoom the second panel onto whichever junction was actually padded.
    junction_xlim = (t_new_ps[0], peak_ps + 1.0) if n_prepend > 0 else (peak_ps - 1.0, t_new_ps[-1])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    fig.suptitle(f"Pulse centering — '{filename}'")

    for ax, (panel_title, xlim) in zip(axes, [
        ('Full trace', (t_new_ps[0], t_new_ps[-1])),
        ('Junction zoom', junction_xlim),
    ]):
        ax.plot(t_orig_ps, y_orig, color='gray', lw=1, linestyle='--', alpha=0.7, label='Original')
        ax.plot(t_new_ps, y_new, color='tab:blue', lw=1.2, label='Centered')
        if n_prepend > 0:
            ax.axvspan(t_new_ps[0], t_new_ps[n_prepend - 1], alpha=0.15, color='tab:blue',
                       label=f'Front pad ({n_prepend} pts, {n_prepend * dt_ps:.2f} ps)')
        if n_append > 0:
            ax.axvspan(t_new_ps[signal_end_index], t_new_ps[-1], alpha=0.15, color='tab:green',
                       label=f'Back pad ({n_append} pts, {n_append * dt_ps:.2f} ps)')
        ax.axvline(peak_ps, color='tab:red', lw=1, linestyle=':', label=f'Peak @ {peak_ps:.2f} ps')
        ax.set_xlim(*xlim)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Amplitude')
        ax.set_title(panel_title)
        ax.legend(fontsize=8)



def center_pulse(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Centre the main pulse at the temporal midpoint by padding with zeros.

    Pads the shorter side: prepends zeros when the peak is in the first half,
    appends zeros when it is past the midpoint (so a pulse near either end of its
    gate is centred, not just an early one). Works for any measurement geometry
    (reflection, transmission) and either segment.  Call once per segment.

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
    ``processing_dict['pre_centering']``.  Call after ``centering_manual``
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
        n_append = info['n_append']
        taper_samples = info['taper_samples']
        dt_ps = info['dt_s'] * _S_TO_PS
        if n_prepend:
            pad_description = f"prepended {n_prepend} samples ({n_prepend * dt_ps:.2f} ps) to the front"
        else:
            pad_description = f"appended {n_append} samples ({n_append * dt_ps:.2f} ps) to the back"
        print(
            f"[center_pulse/{segment}] '{filename}': {pad_description}; "
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
        # exactly as applied to the averaged trace. _ensure_scan_matrix guarantees
        # the matrix is row-aligned with data_obj.data (or a clean averaged-trace
        # fallback if a row-count-changing step ran before segmentation).
        scan_matrix = np.asarray(_ensure_scan_matrix(data_obj), dtype=float)
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
    timing_segment: str = 'second_reflection',
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
    timing_segment : {'second_reflection', 'first_reflection'}
        Which segment to cross-correlate on to measure the shift.  Use
        ``'first_reflection'`` for reflection-mode data where the front-face
        pulse (window reflection, reference-invariant) is the stable T0 keeper.
        The measured integer shift is then applied rigidly to BOTH the first and
        second reflection time axes (same physical delay), keeping the inter-pulse
        delay intact. The subsample residual is stored on the outer
        ``data_obj.processing_dict`` where ``transfer_function`` reads it.
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

        # Resolve which segment drives the cross-correlation measurement.
        if timing_segment == 'first_reflection':
            samp_holder = _resolve_segment(data_obj, 'first_reflection')
            ref_holder = _resolve_segment(ref_obj, 'first_reflection')
            if samp_holder is None or ref_holder is None:
                print(
                    f"Warning: '{filename}' or its reference has no first_reflection "
                    f"segment; falling back to second_reflection for alignment."
                )
                samp_holder = data_obj
                ref_holder = ref_obj
        else:
            samp_holder = data_obj
            ref_holder = ref_obj

        ref_t = ref_holder.data[:, 0]
        ref_y = ref_holder.data[:, 1]
        samp_t = samp_holder.data[:, 0]
        samp_y = samp_holder.data[:, 1]

        align_cfg: dict = {}
        if roi is not None:
            align_cfg['roi'] = (roi[0] / _S_TO_PS, roi[1] / _S_TO_PS)
        if max_lag_ps is not None:
            dt_ref = float(np.median(np.diff(ref_t)))
            align_cfg['max_lag_samples'] = int(round((max_lag_ps / _S_TO_PS) / dt_ref))

        _, _, metrics = core.align_time(ref_t, ref_y, samp_t, samp_y, {'align': align_cfg})
        shift = float(metrics['values']['applied_shift_seconds'])

        # Split into a whole-sample part (slid on the time axis) and a sub-sample
        # residual (applied later as a spectral phase ramp in transfer_function).
        dt = float(np.median(np.diff(samp_t)))
        integer_samples = int(round(shift / dt))
        integer_shift = integer_samples * dt
        subsample_residual = shift - integer_shift
        applied_residual = subsample_residual if subsample_correction else 0.0

        # Always write the subsample residual onto the OUTER data_obj, regardless
        # of which segment was used for timing — transfer_function reads it there.
        data_obj.processing_dict['pre_align'] = data_obj.data.copy()
        data_obj.processing_dict['align_metrics'] = metrics
        data_obj.processing_dict['subsample_shift_seconds'] = applied_residual
        data_obj.processing_dict['subsample_shift_measured_seconds'] = subsample_residual
        data_obj.processing_dict['subsample_correction_enabled'] = bool(subsample_correction)

        # Apply the integer shift rigidly to the second reflection (always).
        shifted_data = data_obj.data.copy()
        shifted_data[:, 0] = data_obj.data[:, 0] + integer_shift
        data_obj.data = shifted_data
        scan_matrix = _ensure_scan_matrix(data_obj)
        scan_matrix[:, 0] = scan_matrix[:, 0] + integer_shift

        # When timing via the first reflection, also shift the first segment so
        # the inter-pulse delay is preserved exactly.
        if timing_segment == 'first_reflection' and isinstance(data_obj, THzDataReflection):
            first_seg = data_obj.first_reflection
            shifted_first = first_seg.data.copy()
            shifted_first[:, 0] = first_seg.data[:, 0] + integer_shift
            first_seg.data = shifted_first
            first_scan_matrix = _ensure_scan_matrix(first_seg)
            first_scan_matrix[:, 0] = first_scan_matrix[:, 0] + integer_shift

        corr = metrics['values'].get('corr_peak', float('nan'))
        state = 'ON' if subsample_correction else 'OFF'
        print(
            f"Aligned '{filename}' to '{ref_obj.filename}' "
            f"(via {timing_segment}): "
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


def align_to_common_time_axis(
    dataset: DataSet,
    segment: str = 'second_reflection',
    show_graph: bool = False,
) -> DataSet:
    """Place every trace of *segment* onto ONE shared absolute-time axis.

    Part 1 of the former combined ``zero_pad`` (the other part is the optional
    resolution padding, still in ``zero_pad``). This step alone is
    **physics-critical** and must run before ``fft_spectrum``.

    Computes the union of all per-file time ranges, builds one uniformly-sampled
    axis spanning it, and inserts each trace at its true absolute-time offset
    ``(t[0] − t_lo_global)/dt``, zero-filling the gaps (wraps
    ``thz_core.pad_to_common_grid``).

    Why it matters: ``centering_manual`` crops a different number of leading
    samples from each file, so afterwards every file **starts at a different
    absolute ``t[0]``** — and that ``t[0]`` difference *is* the sample↔reference
    group delay. ``np.fft.rfft`` references phase to array index 0 and ignores
    the absolute time column, so if you FFT'd now the group delay would be lost
    and ``n`` would collapse toward 1. Re-laying the traces on one shared axis
    converts those start-time differences into **index offsets**, which the FFT
    then sees as the ``exp(−iωΔt)`` ramp that ``invert_nk`` reads as ``n``. This
    is the structural equivalent of the legacy ``phioffset``
    (``2πf·(t0_sam − t0_ref)``). See ANALYSIS_NOTES §14 and §18.

    Parameters
    ----------
    segment : ``'second_reflection'`` (default) or ``'first_reflection'``
        Which data segment to align.  Call once per segment.
    """
    data_dict = _build_segment_data_dict(dataset, segment)
    if not data_dict:
        return dataset

    t_common, padded_dict, metrics = core.pad_to_common_grid(data_dict)

    if show_graph:
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        fig.suptitle(f'Common time axis — {segment}')
        for filename, data_obj in dataset.data.items():
            holder = _resolve_segment(data_obj, segment)
            if holder is None or filename not in padded_dict:
                continue
            pre = holder.data
            ax.plot(pre[:, 0] * _S_TO_PS, pre[:, 1], '--', lw=1, alpha=0.5,
                    label=f'{filename} (pre-align)')
            ax.plot(t_common * _S_TO_PS, padded_dict[filename], lw=1,
                    label=f'{filename} (shared axis)')
        ax.set_xlabel('Time (ps)')
        ax.legend(fontsize=8)
        plt.show()

    for filename, data_obj in dataset.data.items():
        holder = _resolve_segment(data_obj, segment)
        if holder is None or filename not in padded_dict:
            continue
        holder.data = np.column_stack((t_common, padded_dict[filename]))
        holder.processing_dict['common_grid_metrics'] = metrics

    return dataset


def zero_pad(
    dataset: DataSet,
    segment: str = 'second_reflection',
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Zero-pad traces of *segment* for finer FFT frequency resolution.

    Part 2 of the processing: appends trailing zeros to lengthen the FFT without
    adding spectral information (``Δf = 1/(N·dt)``). It first runs
    ``align_to_common_time_axis`` (Part 1) so the traces share one absolute-time
    axis — the group-delay-preserving step — then extends that shared grid.
    Splitting the two makes the physics-critical alignment a named, separately
    callable step while keeping this one backward-compatible (callers that only
    want resolution padding still get the alignment for free).

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

    # Part 1: shared absolute-time axis (group-delay-preserving). Idempotent if
    # the traces already share a grid.
    align_to_common_time_axis(dataset, segment=segment)

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


def minimum_fft_length(
    dataset: DataSet,
    segments: tuple = ('first_reflection', 'second_reflection'),
) -> int:
    """Smallest ``n_fft`` that transforms every segment without truncation.

    ``fft_spectrum`` calls ``np.fft.rfft(y, n=n_fft)``, which *truncates* ``y``
    when ``n_fft < len(y)``. On the shared-axis path the windowed trace keeps its
    full length — ``window_pulses_fixed_width`` zeros the samples outside the
    pulse but never crops — so the FFT input spans the whole acquisition. If
    ``n_fft`` is shorter than that, the tail of the trace (which is where the
    reflection pulses actually sit) is silently cut off.

    Returns the largest sample count across every file and segment (references
    included — they are transformed too). Use it to assert/derive ``n_fft``::

        required = thz.minimum_fft_length(dataset)
        assert n_fft >= required

    Returns 0 if no matching segment holders are present.
    """
    max_samples = 0
    for filename, data_obj in dataset.data.items():
        for segment in segments:
            holder = _resolve_segment(data_obj, segment)
            if holder is None:
                continue
            max_samples = max(max_samples, int(holder.data.shape[0]))
    return max_samples


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

        Prefers ``first_reflection.processing_dict['fft_spectrum']`` when the object
        is a ``THzDataReflection`` (already computed by the pipeline). Falls back
        to loading from disk via the legacy ``_first_reflection_spectrum`` path.
        """
        if isinstance(data_object, THzDataReflection):
            stored = data_object.first_reflection.processing_dict.get('fft_spectrum')
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


def remove_phase_offset(
    dataset: DataSet,
    config: dict | None = None,
    show_graph: bool = False,
) -> DataSet:
    """Force each sample's transfer-function phase through the origin (``phaseex``).

    Deterministic, non-interactive counterpart to ``phase_correction``. Runs
    AFTER ``transfer_function`` and operates on ``processing_dict['transfer_H']``
    — the array ``invert_nk`` actually reads — so the correction reaches the
    inversion (unlike ``phase_correction(source='fft')``, which edits the raw
    spectra after H is already built and therefore has no effect on n).

    A constant phase offset on H biases n by a term that diverges as 1/f toward
    DC (the low-frequency droop in n for an otherwise-flat sample). This fits a
    line to H's unwrapped phase over a trusted band and subtracts ONLY the
    intercept, keeping the slope (the group-delay / refractive-index signal).
    See ANALYSIS_NOTES §18 and ``thz_core.remove_phase_offset``.

    IMPORTANT — this corrects only the **sub-2π (fractional)** part of the offset.
    A whole-cycle (2π) error is invisible in the complex ``H`` and would be
    re-introduced by ``invert_nk``'s re-unwrap; that part is handled by
    ``invert_nk``'s origin anchor (``config['invert']['anchor_phase_origin']``,
    default True). For a thick sample (the common droop case) the anchor alone
    flattens n; this step then removes the small remaining residual.

    Parameters
    ----------
    config : dict, optional
        ``config['phase_offset']`` keys:

        ``band_thz`` : tuple[float, float], default ``(0.3, 2.0)``
            Trusted fit band in THz (converted to Hz for the core call).
        ``use_snr_mask`` : bool, default True
            Intersect the fit band with ``processing_dict['transfer_mask']`` when
            present, so only trusted bins drive the intercept fit.
    show_graph : bool
        Overlay the unwrapped phase before/after per sample.
    """
    config = config or {}
    phase_cfg = config.get('phase_offset', {})
    band_thz = phase_cfg.get('band_thz', (0.3, 2.0))
    use_snr_mask = phase_cfg.get('use_snr_mask', True)
    band_hz = (band_thz[0] / _HZ_TO_THZ, band_thz[1] / _HZ_TO_THZ)

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue
        processing = data_obj.processing_dict
        H = processing.get('transfer_H')
        freq = processing.get('fft_freq')
        if H is None or freq is None:
            print(f"[remove_phase_offset] '{filename}': no transfer function, skipping.")
            continue

        mask = processing.get('transfer_mask') if use_snr_mask else None
        H_corrected, metrics = core.remove_phase_offset(freq, H, fit_band_hz=band_hz, mask=mask)
        
        slope = metrics['values']['slope_rad_per_hz']
        intercept = metrics['values']['intercept_rad']
        fit_phase = np.polyval([slope, intercept], freq)

        if show_graph:
            # Unwrap only the finite bins — np.unwrap over the full NaN-containing
            # array propagates NaN forward and can blank the plot.
            finite = np.isfinite(H)
            f_thz = freq * _HZ_TO_THZ
            before_phase = np.full(freq.size, np.nan)
            after_phase = np.full(freq.size, np.nan)
            idx = np.where(finite)[0]
            if idx.size >= 2:
                before_phase[idx] = np.unwrap(np.angle(H[idx]))
                after_phase[idx] = np.unwrap(np.angle(H_corrected[idx]))
            fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
            ax.plot(f_thz, before_phase, color='steelblue', alpha=0.6, label='before')
            ax.plot(f_thz, after_phase, color='darkorange', label='after (intercept removed)')
            ax.axhline(0.0, color='gray', lw=0.5, linestyle='dashed')
            ax.axvline(0.0, color='gray', lw=0.5, linestyle='dashed')
            ax.scatter(0.0, 0.0, color='gray', s=20, zorder=3, marker='x')
            ax.plot(f_thz, fit_phase, color='steelblue', lw=1.0, alpha=0.7, label='phase fit', ls='dashed')
            ax.plot(f_thz, fit_phase - intercept, color='darkorange', lw=1.0, alpha=0.7, label='fit w/ intercept removed', ls='dashed')
            ax.axvspan(band_thz[0], band_thz[1], alpha=0.1, color='green', label='fit band')
            ax.set_xlabel('Frequency (THz)')
            ax.set_ylabel('Unwrapped phase of H (rad)')
            ax.set_title(
                f"Phase-offset removal — {filename} "
                f"(Δintercept = {metrics['values']['intercept_rad']:+.3f} rad)"
            )
            ax.legend(fontsize=8)
            plt.show()

        processing['transfer_H'] = H_corrected
        processing['phase_offset_metrics'] = metrics
        data_obj.data = np.column_stack((
            freq, np.abs(H_corrected), np.angle(H_corrected),
        ))
        if metrics['values']['applied']:
            print(
                f"[remove_phase_offset] '{filename}': removed intercept "
                f"{metrics['values']['intercept_rad']:+.4f} rad "
                f"({metrics['values']['n_fit_bins']} fit bins, "
                f"{band_thz[0]}-{band_thz[1]} THz)."
            )
        else:
            print(
                f"[remove_phase_offset] '{filename}': fit band held <2 trusted bins; "
                f"H left unchanged."
            )

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


def derive_eps_sigma(dataset: DataSet) -> DataSet:
    """Derive complex permittivity and optical conductivity from n, k."""
    # config = config or {}
    config = dataset.config or {}

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
