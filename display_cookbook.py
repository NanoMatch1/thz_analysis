"""Worked examples of the ``display`` adapter — copy the block you need into your own script.

``open_session.py`` is the everyday viewer (pick a bundle -> interactive ResultsViewer).
``extract_session_data.py`` pulls the arrays out as dicts for your own processing.
This file is the reference for building *custom figures* with ``dataset_core.adapters.display``:
one ``plot_quantity`` call per figure, styling built once and shared, arrays via ``get_series``.

    python display_cookbook.py                 # run the cookbook on the default bundle
    python display_cookbook.py <id> --misalignment   # the gold-mirror misalignment figures
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from dataset_core.adapters import display
from open_session import open_from_catalogue


def show_misalignment_figures(
    dataset,
    displacement_key: str = "_gold_",
    band_thz: tuple[float, float] = (0.0, 4.0),
    nk_band_thz: tuple[float, float] = (0.2, 4.0),
    show: bool = True,
):
    """Group-meeting view: each misalignment-series panel as its OWN figure via ``display``.

    Independent figures built through the reusable registry-driven plotter. Add or drop a
    panel by adding or removing a single ``display.plot_quantity`` call — any registered quantity (``transfer_phase``,
    ``sigma_real``, ...) is available the same way.
    """
    # STEP 1 — build the per-item styling ONCE (a pre-plot manipulation that returns a
    # style dict). diverging_series_style parses the signed mrad out of each filename and
    # returns a style_fn that paints blue -> black -> red with "+/-N mrad" labels and an
    # emphasised zero trace. Sharing this one dict across every figure keeps colour/label
    # consistent from the time trace through to n and k.
    style = display.diverging_series_style(dataset, displacement_key=displacement_key)

    # STEP 2 — one call per figure. `style=` controls how each item looks; `plot_config=`
    # controls how the figure looks. Time-domain uses its own helper (line plot, time axis);
    # every frequency-domain quantity goes through plot_quantity (scatter + error bars on the
    # instrument-resolution grid, mask + overlay handled automatically from the registry).
    display.plot_time_domain(
        dataset, style=style,
        plot_config={"title": "Time-domain traces"})

    display.plot_quantity(
        dataset, "fft_mag", style=style,
        plot_config={"title": "Spectral magnitude", "x_range": band_thz})

    display.plot_quantity(
        dataset, "fft_phase", style=style,
        plot_config={"title": "FFT phase (unwrapped)", "x_range": band_thz})

    display.plot_quantity(
        dataset, "n", style=style,
        plot_config={"title": "Refractive index n", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "k", style=style,
        plot_config={"title": "Extinction coefficient k", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "sigma_real", style=style,
        plot_config={"title": "Conductivity Re σ", "x_range": nk_band_thz})

    display.plot_quantity(
        dataset, "sigma_imag", style=style,
        plot_config={"title": "Conductivity Im σ", "x_range": nk_band_thz})

    if show:
        plt.show()


def display_adapter_cookbook(dataset, displacement_key: str = "_gold_", show: bool = True):
    """Worked examples of the ``display`` adapter — the reference for future data views.

    Every pattern you are likely to want, each as a couple of lines. Read top to bottom;
    copy the block you need. Nothing here is misalignment-specific except the style helper.
    """
    # ── 1. The simplest possible call: one registered quantity -> one figure. ──────────
    # No style, no config: samples only, default colours, scatter + error bars, registry
    # axis label and y-scale. `quantity` is any name in quantity_registry.QUANTITY_REGISTRY.
    display.plot_quantity(dataset, "n")

    # ── 2. List what is actually available to plot for this dataset. ───────────────────
    print("Plottable quantities:", display.available_quantities(dataset))

    # ── 3. Shared computed style + explicit per-file override. ─────────────────────────
    # style resolution priority (low -> high): default < style_fn < per_file(regex) <
    # per_file(exact). So the gradient paints the series, and per_file tweaks individuals.
    # Keys in per_file are regex/substring by default; an EXACT filename wins outright.
    style = display.diverging_series_style(dataset, displacement_key=displacement_key)
    style["per_file"] = {
        "plus-4": {"marker": "s"},                 # every +4 mrad file -> square markers
        # "sample_gold_plus-2-mrad.acc": {"color": "magenta"},  # one exact file recoloured
    }
    display.plot_quantity(dataset, "k", style=style,
                          plot_config={"title": "k — with per-file overrides",
                                       "x_range": (0.2, 4.0)})

    # ── 4. plot_config knobs (how the FIGURE looks; overrides DEFAULT_PLOT_CONFIG). ────
    display.plot_quantity(
        dataset, "transfer_phase", style=style,
        plot_config={
            "title": "Transfer phase — config demo",
            "x_range": (0.2, 3.0),      # axis limits in plot units (THz)
            "show_errorbars": True,     # error bars from the MC *_sigma arrays
            "show_excluded": True,      # also draw the masked-out (low-SNR) points, faint
            "normalise": False,         # True -> min-max each series to [0, 1]
            "guideline": False,         # True -> shade a +/- band around each series
            "legend": True,
        })

    # ── 5. Select WHICH items to plot (files=): regex, list, or a predicate. ───────────
    # None (default) = samples (+ references when the quantity is reference-meaningful,
    # e.g. the FFT spectra). Override to focus a subset:
    display.plot_quantity(dataset, "fft_mag",
                          files="minus",                       # regex/substring on filename
                          plot_config={"title": "fft_mag — negative-mrad files only",
                                       "x_range": (0, 4.0)})
    # files=["a.acc", "b.acc"]              # explicit list
    # files=lambda name: "plus" in name      # arbitrary predicate

    # ── 6. Get the arrays WITHOUT plotting (inspect / manipulate / export yourself). ───
    # Returns {filename: {freq_hz, freq_thz, y, error, mask}} — exactly what would be drawn.
    series = display.get_series(dataset, "n")
    for filename, arrays in series.items():
        finite = arrays["mask"] if arrays["mask"] is not None else slice(None)
        print(f"  {filename}: n over {arrays['freq_thz'][finite].min():.2f}-"
              f"{arrays['freq_thz'][finite].max():.2f} THz")

    # ── 7. Compose into your own multi-panel figure by passing an existing axis. ───────
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(12, 4.5))
    display.plot_quantity(dataset, "n", style=style, ax=axis_left,
                          plot_config={"title": "n", "x_range": (0.2, 4.0)})
    display.plot_quantity(dataset, "k", style=style, ax=axis_right,
                          plot_config={"title": "k", "x_range": (0.2, 4.0), "legend": False})
    figure.tight_layout()

    if show:
        plt.show()


if __name__ == "__main__":
    # Default: the gold-mirror misalignment series (catalogue id 49ad3cc4).
    positionals = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    dataset = open_from_catalogue(identifier=positionals[0] if positionals else "49ad3cc4")
    if dataset is None:
        raise SystemExit("Nothing selected.")
    if "--misalignment" in sys.argv:
        # Each misalignment-series output (time / FFT / n / k / sigma) as its own figure,
        # coloured by displacement angle.
        show_misalignment_figures(dataset, displacement_key="_gold_", show=True)
    else:
        display_adapter_cookbook(dataset, displacement_key="_gold_", show=True)
