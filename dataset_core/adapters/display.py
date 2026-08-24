"""Generic, registry-driven plotting for THz-TDS quantities — one call, one figure.

This is the reusable display layer meant to be leveraged from ``open_session.py``, the
``run_me_*`` scripts, notebooks, and future analysis. It deliberately owns *no* knowledge
of what a quantity is or where it lives: every plottable series is defined once in
``quantity_registry`` (the single source of truth), and this module simply consumes that
registry. Registering a new quantity there makes it plottable here for free.

Two responsibilities, two clearly-separated config dicts:

``plot_config`` — how the FIGURE looks
    axis ranges/scales, title and label overrides, error bars on/off, legend, masking
    behaviour, normalisation, marker size. Nothing about individual data items.

``style`` — how each DATA ITEM looks
    per-filename appearance (colour / marker / linestyle / label / zorder / alpha),
    resolved from two composable sources:
      * ``per_file`` — a ``{filename-or-regex: {...}}`` override dict, and
      * ``style_fn`` — an optional ``callable(filename, data_obj) -> {...}`` for computed
        styling (e.g. a misalignment-angle colour gradient).
    Building the style dict is a *pre-plot* manipulation — you compute colours/labels once
    and hand them in, keeping the colouring logic decoupled from the plotting.

Frequency-domain quantities go through :func:`plot_quantity`. Time-domain traces have no
registry entry and a different x-axis, so they get their own small helper
:func:`plot_time_domain`. Use :func:`get_series` to pull the exact arrays that *would* be
plotted (x, y, error, mask) without drawing anything — for inspection or manipulation.

Rendering convention (Samuel's standing preference): every frequency-domain series is drawn
as SCATTER points with ERROR BARS on the stored grid. When the pipeline ran with
``limit_to_instrument_resolution`` on, that stored grid *is* the true instrument-resolution
grid (``apply_instrument_resolution`` decimated every freq-aligned array — values, masks and
``*_sigma`` errors alike — onto it), so this view is faithful, not sinc-interpolated.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np
import matplotlib.pyplot as plt

from . import quantity_registry

_HZ_TO_THZ = 1e-12
_S_TO_PS = 1e12


# ── figure defaults (override any key via the plot_config argument) ──────────────

DEFAULT_PLOT_CONFIG: dict = {
    "figsize": (8.0, 4.8),
    "title": None,          # None -> quantity.label (freq) / "Time-domain traces" (time)
    "x_label": None,        # None -> "Frequency (THz)" / "Time (ps)"
    "y_label": None,        # None -> quantity.y_label
    "x_range": None,        # (lo, hi) in plot units, or None
    "y_range": None,
    "xscale": None,         # None -> linear; else 'log'
    "yscale": None,         # None -> quantity.yscale
    "show_errorbars": True,
    "show_mask": True,      # honour the quantity's SNR/trusted-band mask
    "show_excluded": False,  # also draw the masked-out points, very faint
    "show_overlay": True,   # draw the registry overlay (e.g. a fit curve) if present
    "normalise": False,     # min-max normalise each series to [0, 1] (over trusted band)
    "guideline": False,     # shade a +/- band around each series (visual tracer)
    "guideline_depth": 0.2,
    "legend": True,
    "legend_fontsize": 8,
    "marker_size": 5,       # base marker size (points); style/style_fn can override per item
    "capsize": 2.0,
    "elinewidth": 0.8,
}


# ── item selection ───────────────────────────────────────────────────────────────


def _item_is_selected(dataset, filename: str, files, include_references: bool) -> bool:
    """Decide whether one file is included, given the ``files`` selector.

    ``files`` may be:
        None      -> all samples, plus references only if ``include_references``
        str       -> regex / substring match against the filename
        callable  -> ``files(filename) -> bool``
        iterable  -> membership in an explicit collection of filenames
    """
    if files is None:
        if dataset.data.is_reference(filename):
            return include_references
        return True
    if callable(files):
        return bool(files(filename))
    if isinstance(files, str):
        return re.search(files, filename) is not None
    return filename in set(files)


def _iter_selected_items(dataset, files, include_references: bool):
    """Yield ``(filename, data_obj)`` for every selected item (order = insertion order).

    Iterates ``dataset.data.data_dict`` (the full master dict, independent of the current
    grouping selection) so a series can be plotted regardless of the working set.
    """
    for filename, data_obj in dataset.data.data_dict.items():
        if _item_is_selected(dataset, filename, files, include_references):
            yield filename, data_obj


# ── per-item style resolution ────────────────────────────────────────────────────


def resolve_item_style(
    filename: str,
    data_obj,
    style: dict | None,
    index: int,
    color_cycle: list,
    base_defaults: dict,
) -> dict:
    """Resolve one item's appearance by composing the style sources, lowest priority first.

    Priority (each layer overrides the previous):
        1. ``base_defaults``           — the caller's rendering defaults (scatter vs line)
        2. ``style['default']``        — user base applied to every item
        3. ``style['style_fn'](...)``  — computed per-item styling (e.g. angle gradient)
        4. ``style['per_file']`` regex/substring matches (in dict order)
        5. ``style['per_file']`` exact-filename match (explicit override wins outright)

    An unset ``color`` falls back to the matplotlib colour cycle by ``index`` so distinct
    items are always visually separable. ``label`` defaults to the filename.
    """
    resolved = {"label": filename}
    resolved.update(base_defaults)

    if style:
        resolved.update(style.get("default", {}))

        style_fn = style.get("style_fn")
        if callable(style_fn):
            computed = style_fn(filename, data_obj)
            if computed:
                resolved.update(computed)

        per_file = style.get("per_file", {}) or {}
        use_regex = style.get("regex", True)
        if use_regex:
            for pattern, override in per_file.items():
                if pattern != filename and re.search(pattern, filename):
                    resolved.update(override)
        if filename in per_file:  # exact match has the final say
            resolved.update(per_file[filename])

    if resolved.get("color") is None:
        resolved["color"] = color_cycle[index % len(color_cycle)]
    return resolved


def _color_cycle() -> list:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color")
    return list(colors) if colors else ["C0", "C1", "C2", "C3", "C4"]


# ── frequency-domain plotting (registry-driven) ──────────────────────────────────


def _safe_extract(quantity, processing_dict):
    """Call ``quantity.extract`` defensively; return ``(freq_hz, y)`` or ``None``."""
    try:
        return quantity.extract(processing_dict)
    except Exception:
        return None


def _draw_scatter_series(ax, freq_thz, values, errors, mask, item_style, plot_config):
    """Draw one frequency-domain series as scatter + error bars, honouring the mask."""
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    errors = np.asarray(errors, dtype=float) if errors is not None else None

    if plot_config.get("normalise"):
        finite = np.isfinite(values) & mask
        if finite.any():
            low, high = np.nanmin(values[finite]), np.nanmax(values[finite])
            if high > low:
                values = (values - low) / (high - low)
                if errors is not None:
                    errors = errors / (high - low)

    x_in = freq_thz[mask]
    y_in = values[mask]
    err_in = errors[mask] if errors is not None else None

    ax.errorbar(
        x_in, y_in, yerr=err_in,
        color=item_style["color"],
        marker=item_style.get("marker", "o"),
        linestyle=item_style.get("linestyle", "none"),
        markersize=item_style.get("markersize", plot_config["marker_size"]),
        alpha=item_style.get("alpha", 0.85),
        label=item_style.get("label"),
        zorder=item_style.get("zorder", 2),
        capsize=plot_config["capsize"],
        elinewidth=plot_config["elinewidth"],
        ecolor=item_style.get("ecolor", item_style["color"]),
    )

    if plot_config.get("show_excluded") and (~mask).any():
        ax.plot(freq_thz[~mask], values[~mask], marker=".", linestyle="none",
                color=item_style["color"], alpha=0.12, markersize=3, zorder=1)

    if plot_config.get("guideline"):
        depth = plot_config.get("guideline_depth", 0.2)
        ax.fill_between(x_in, y_in - np.abs(y_in) * depth, y_in + np.abs(y_in) * depth,
                        color=item_style["color"], alpha=0.1, zorder=1)


def plot_quantity(
    dataset,
    quantity: str,
    *,
    files=None,
    style: dict | None = None,
    plot_config: dict | None = None,
    ax=None,
):
    """Plot one registered frequency-domain quantity as scatter + error bars → one figure.

    Parameters
    ----------
    dataset : DataSet
        A loaded/replayed dataset whose samples carry ``processing_dict`` results.
    quantity : str
        A name registered in ``quantity_registry`` (e.g. ``'n'``, ``'k'``, ``'fft_mag'``,
        ``'fft_phase'``, ``'transfer_phase'``, ``'sigma_real'``, ...).
    files : None | str | callable | iterable, optional
        Item selector — see :func:`_item_is_selected`. ``None`` selects samples (plus
        references when the quantity is reference-meaningful, e.g. the FFT spectra).
    style : dict, optional
        Per-item appearance: ``{'default': {...}, 'per_file': {name-or-regex: {...}},
        'style_fn': callable, 'regex': True}``. See :func:`resolve_item_style`.
    plot_config : dict, optional
        Figure-level options; overrides :data:`DEFAULT_PLOT_CONFIG`.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of creating a figure (for composition).

    Returns
    -------
    matplotlib Figure
    """
    quantity_spec = quantity_registry.QUANTITY_REGISTRY.get(quantity)
    if quantity_spec is None:
        available = ", ".join(quantity_registry.QUANTITY_REGISTRY)
        raise KeyError(f"Unknown quantity '{quantity}'. Registered: {available}")

    config = {**DEFAULT_PLOT_CONFIG, **(plot_config or {})}
    created_here = ax is None
    figure, ax = (plt.subplots(figsize=config["figsize"]) if created_here
                  else (ax.figure, ax))

    color_cycle = _color_cycle()
    scatter_defaults = {"marker": "o", "linestyle": "none",
                        "markersize": config["marker_size"], "alpha": 0.85}
    n_plotted = 0
    for index, (filename, data_obj) in enumerate(
            _iter_selected_items(dataset, files, quantity_spec.include_references)):
        processing_dict = data_obj.processing_dict
        extracted = _safe_extract(quantity_spec, processing_dict)
        if extracted is None:
            continue
        freq_hz, values = extracted
        freq_thz = np.asarray(freq_hz) * _HZ_TO_THZ

        errors = (processing_dict.get(quantity_spec.error_key)
                  if (config["show_errorbars"] and quantity_spec.error_key) else None)
        mask = (processing_dict.get(quantity_spec.mask_key)
                if (config["show_mask"] and quantity_spec.mask_key) else None)

        item_style = resolve_item_style(filename, data_obj, style, index,
                                        color_cycle, scatter_defaults)
        _draw_scatter_series(ax, freq_thz, values, errors, mask, item_style, config)

        if config["show_overlay"] and quantity_spec.overlay is not None:
            overlaid = quantity_spec.overlay(processing_dict)
            if overlaid is not None:
                ax.plot(np.asarray(overlaid[0]) * _HZ_TO_THZ, overlaid[1],
                        color=item_style["color"], linewidth=1.2, alpha=0.9, zorder=3)
        n_plotted += 1

    ax.set_xlabel(config["x_label"] or "Frequency (THz)")
    ax.set_ylabel(config["y_label"] or quantity_spec.y_label)
    ax.set_yscale(config["yscale"] or quantity_spec.yscale)
    if config["xscale"]:
        ax.set_xscale(config["xscale"])
    if config["x_range"]:
        ax.set_xlim(*config["x_range"])
    if config["y_range"]:
        ax.set_ylim(*config["y_range"])
    ax.set_title(config["title"] or quantity_spec.label)
    if config["legend"] and n_plotted:
        ax.legend(fontsize=config["legend_fontsize"])
    if created_here:
        figure.tight_layout()
    return figure


# ── time-domain plotting (no registry entry, different x-axis) ────────────────────


def plot_time_domain(
    dataset,
    *,
    files=None,
    style: dict | None = None,
    plot_config: dict | None = None,
    ax=None,
    time_key: str = "time_domain",
):
    """Plot the stored time-domain traces (amplitude vs time) → one figure.

    Time-domain traces are continuous field records, so they are drawn as LINES (not the
    scatter + error-bar convention used for frequency-domain quantities). Reads the ``Nx2``
    ``[time_s, amplitude]`` array stored under ``processing_dict[time_key]``. References are
    included by default (the aligned reference trace belongs in a misalignment series).
    """
    config = {**DEFAULT_PLOT_CONFIG, **(plot_config or {})}
    created_here = ax is None
    figure, ax = (plt.subplots(figsize=config["figsize"]) if created_here
                  else (ax.figure, ax))

    color_cycle = _color_cycle()
    line_defaults = {"linestyle": "solid", "linewidth": 1.4, "alpha": 0.9, "marker": None}
    n_plotted = 0
    for index, (filename, data_obj) in enumerate(
            _iter_selected_items(dataset, files, include_references=True)):
        trace = data_obj.processing_dict.get(time_key)
        if trace is None:
            continue
        trace = np.asarray(trace)
        time_ps = trace[:, 0] * _S_TO_PS
        amplitude = trace[:, 1]

        item_style = resolve_item_style(filename, data_obj, style, index,
                                        color_cycle, line_defaults)
        ax.plot(time_ps, amplitude,
                color=item_style["color"],
                linestyle=item_style.get("linestyle", "solid"),
                linewidth=item_style.get("linewidth", 1.4),
                marker=item_style.get("marker"),
                alpha=item_style.get("alpha", 0.9),
                label=item_style.get("label"),
                zorder=item_style.get("zorder", 2))
        n_plotted += 1

    ax.set_xlabel(config["x_label"] or "Time (ps)")
    ax.set_ylabel(config["y_label"] or "Field amplitude (a.u.)")
    if config["xscale"]:
        ax.set_xscale(config["xscale"])
    if config["yscale"]:
        ax.set_yscale(config["yscale"])
    if config["x_range"]:
        ax.set_xlim(*config["x_range"])
    if config["y_range"]:
        ax.set_ylim(*config["y_range"])
    ax.set_title(config["title"] or "Time-domain traces")
    if config["legend"] and n_plotted:
        ax.legend(fontsize=config["legend_fontsize"])
    if created_here:
        figure.tight_layout()
    return figure


# ── data access without plotting (inspect / manipulate before drawing) ───────────


def get_series(dataset, quantity: str, *, files=None) -> dict:
    """Return the exact arrays that :func:`plot_quantity` would draw, keyed by filename.

    Each value is ``{'freq_hz', 'freq_thz', 'y', 'error', 'mask'}``. Use it to inspect or
    manipulate the plotted data directly, or to build derived series before plotting.
    """
    quantity_spec = quantity_registry.QUANTITY_REGISTRY.get(quantity)
    if quantity_spec is None:
        available = ", ".join(quantity_registry.QUANTITY_REGISTRY)
        raise KeyError(f"Unknown quantity '{quantity}'. Registered: {available}")

    series = {}
    for filename, data_obj in _iter_selected_items(
            dataset, files, quantity_spec.include_references):
        processing_dict = data_obj.processing_dict
        extracted = _safe_extract(quantity_spec, processing_dict)
        if extracted is None:
            continue
        freq_hz, values = extracted
        freq_hz = np.asarray(freq_hz)
        series[filename] = {
            "freq_hz": freq_hz,
            "freq_thz": freq_hz * _HZ_TO_THZ,
            "y": np.asarray(values),
            "error": processing_dict.get(quantity_spec.error_key),
            "mask": processing_dict.get(quantity_spec.mask_key),
        }
    return series


def available_quantities(dataset, *, files=None) -> list[str]:
    """Names of registered quantities that actually have data in the selected items."""
    present: list[str] = []
    for _, data_obj in _iter_selected_items(dataset, files, include_references=True):
        for name in quantity_registry.available_names(data_obj.processing_dict):
            if name not in present:
                present.append(name)
    return present


# ── styling helpers (pre-plot manipulations that RETURN a style dict) ─────────────


def parse_signed_displacement(filename: str, displacement_key: str) -> float | None:
    """Signed magnitude from a ``...{key}{minus|plus}-{N}`` filename (0 for ``{key}-0``).

    Generic filename→value parse used to colour a misalignment/displacement series.
    Returns ``None`` when the key is absent, so unrelated files are left unstyled.
    """
    if displacement_key not in filename:
        return None
    signed = re.search(rf"{re.escape(displacement_key)}(minus|plus)-(\d+(?:\.\d+)?)", filename)
    if signed:
        magnitude = float(signed.group(2))
        return -magnitude if signed.group(1) == "minus" else magnitude
    if re.search(rf"{re.escape(displacement_key)}-0(?!\d)", filename):
        return 0.0
    return None


def _diverging_color(signed_value: float, max_negative: float, max_positive: float,
                     zero_color: str):
    """Zero -> ``zero_color``; deepening blue for negatives, deepening red for positives."""
    if signed_value == 0:
        return zero_color
    if signed_value < 0:
        fraction = abs(signed_value) / max_negative if max_negative else 1.0
        return plt.cm.Blues(0.2 + 0.6 * fraction)
    fraction = signed_value / max_positive if max_positive else 1.0
    return plt.cm.Reds(0.2 + 0.6 * fraction)


def diverging_series_style(
    dataset,
    displacement_key: str,
    *,
    parse: Callable[[str, str], float | None] = parse_signed_displacement,
    zero_color: str = "black",
    unit_label: str = "mrad",
) -> dict:
    """Build a ``style`` dict that colours a signed-displacement series blue→zero→red.

    This is the reusable, decoupled replacement for the old hard-coded misalignment
    colouring: it parses a signed value out of every filename once, then returns a
    ``style_fn`` closure mapping each file to a diverging colour, a ``±N unit`` label, and
    an emphasised zero trace. Pass the result straight into ``plot_quantity`` /
    ``plot_time_domain`` as ``style=``; layer explicit ``per_file`` overrides on top if
    needed (they win over this function).
    """
    signed_by_file = {
        filename: parse(filename, displacement_key)
        for filename in dataset.data.data_dict
    }
    signed_by_file = {f: v for f, v in signed_by_file.items() if v is not None}
    negatives = [v for v in signed_by_file.values() if v < 0]
    positives = [v for v in signed_by_file.values() if v > 0]
    max_negative = abs(min(negatives)) if negatives else 1.0
    max_positive = max(positives) if positives else 1.0

    def style_fn(filename, data_obj):
        value = signed_by_file.get(filename)
        if value is None:
            return {}
        is_zero = value == 0
        return {
            "color": _diverging_color(value, max_negative, max_positive, zero_color),
            "label": f"{value:+.1f} {unit_label}",
            "zorder": 6 if is_zero else 2,
            "markersize": 6 if is_zero else 4,
            "linewidth": 2.2 if is_zero else 1.4,
        }

    return {"style_fn": style_fn}
