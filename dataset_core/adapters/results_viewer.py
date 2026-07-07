"""Registry-driven results viewer + CSV export (Phase 2 — replaces the legacy ResultViewer).

Both the interactive viewer and the CSV export iterate ``quantity_registry.QUANTITY_REGISTRY``,
so a new quantity appears in the GUI and the export from a single registration — display and
export can never drift apart.

Entry points:
    launch_results_viewer(dataset, ...)   -> ResultsViewer   (interactive; headless-safe)
    export_quantities(dataset, dir=None)  -> [written paths] (registry-driven CSVs)

Works on any dataset — a live pipeline dataset or one restored by ``session_bundle.load_session``.
"""

from __future__ import annotations

import os

import numpy as np

from dataset_core.adapters import quantity_registry as registry
from dataset_core.adapters.thz_adapter import (
    _sample_items,
    _plot_with_snr_mask,
    _resolution_marker_stride,
)

_HZ_TO_THZ = 1e-12
_S_TO_PS = 1e12


# ── registry-driven export ───────────────────────────────────────────────────────


def export_quantities(dataset, export_dir: str | None = None) -> list[str]:
    """Write one CSV per sample with freq_THz + every registered export column.

    Superset-safe: a quantity missing for a given sample is written as a NaN column, so every
    file has the same header. Also writes the raw and pre-FFT time traces (as the legacy
    ``export_results`` did). Returns the list of written paths.
    """
    if export_dir is None:
        export_dir = os.path.join(dataset.file_dir, "results")
    os.makedirs(export_dir, exist_ok=True)

    columns_spec = registry.export_quantities_list()
    written: list[str] = []

    for filename, data_obj in _sample_items(dataset):
        processing = data_obj.processing_dict
        stem = os.path.splitext(filename)[0]

        # time-domain traces (unchanged from the legacy exporter)
        raw = processing.get("time_domain")
        if raw is not None:
            raw_out = np.asarray(raw, dtype=float).copy()
            raw_out[:, 0] *= _S_TO_PS
            path = os.path.join(export_dir, f"{stem}_time_raw.csv")
            np.savetxt(path, raw_out, delimiter=",", header="time_ps,amplitude,stderr", comments="")
            written.append(path)
        prefft = processing.get("time_domain_prefft")
        if prefft is not None:
            prefft_out = np.asarray(prefft, dtype=float).copy()
            prefft_out[:, 0] *= _S_TO_PS
            path = os.path.join(export_dir, f"{stem}_time_prefft.csv")
            np.savetxt(path, prefft_out, delimiter=",", header="time_ps,amplitude", comments="")
            written.append(path)

        freq = processing.get("fft_freq")
        if freq is None:
            print(f"[export_quantities] '{filename}': no FFT data; skipping freq-domain.")
            continue
        freq = np.asarray(freq, dtype=float)
        nan_column = np.full(freq.shape, np.nan)

        headers = ["freq_THz"]
        data_columns = [freq * _HZ_TO_THZ]
        for quantity in columns_spec:
            headers.append(quantity.export_header)
            result = None
            try:
                result = quantity.extract(processing)
            except Exception:
                result = None
            data_columns.append(nan_column if result is None else np.asarray(result[1], dtype=float))

        path = os.path.join(export_dir, f"{stem}_results.csv")
        np.savetxt(path, np.column_stack(data_columns), delimiter=",",
                   header=",".join(headers), comments="")
        written.append(path)

    print(f"[export_quantities] wrote {len(written)} file(s) to {export_dir} "
          f"({len(columns_spec)} quantity columns).")
    return written


# ── interactive viewer ───────────────────────────────────────────────────────────


class ResultsViewer:
    """Registry-driven interactive results browser.

    A quantity selector (radio, grouped labels) + per-sample visibility checkboxes + a frequency
    range box. Each quantity is plotted with the resolution-aware faint-line + markers +
    untrusted-band style, with fit overlays and error bars drawn when the data provides them.

    Headless-safe: on a non-interactive backend it builds the figure without blocking (so smoke
    tests can drive it); widgets are pinned to the figure so callbacks survive the caller
    discarding the handle; the blocking show drops out of interactive mode so it truly blocks.
    """

    def __init__(self, dataset, *, show_snr_mask: bool = True,
                 initial: str | None = None, show: bool = True, block: bool = True):
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, CheckButtons, TextBox

        self._plt = plt
        self._dataset = dataset
        self._show_snr_mask = bool(show_snr_mask)
        self._quantities = registry.display_quantities()
        self._names = [q.name for q in self._quantities]
        self._labels = [f"{q.group}: {q.label}" for q in self._quantities]

        self._sample_names = [fn for fn, _ in _sample_items(dataset)]
        self._all_names = list(dataset.data.keys())
        self._visible = {fn: True for fn in self._sample_names}

        cmap = plt.get_cmap("tab10")
        self._colours = {fn: cmap(i % 10) for i, fn in enumerate(self._sample_names)}
        for fn in self._all_names:
            self._colours.setdefault(fn, (0.5, 0.5, 0.5, 0.6))

        self._current = initial if initial in self._names else self._names[0]
        self._freq_min = None
        self._freq_max = None
        self._marker_stride = _resolution_marker_stride(dataset)

        self._fig = plt.figure(figsize=(13, 7.5))
        self._ax = self._fig.add_axes([0.08, 0.10, 0.60, 0.83])

        radio_ax = self._fig.add_axes([0.70, 0.42, 0.28, 0.52])
        radio_ax.set_frame_on(False)
        active = self._names.index(self._current)
        self._radio = RadioButtons(radio_ax, self._labels, active=active)
        self._radio.on_clicked(self._on_quantity)

        sample_ax = self._fig.add_axes([0.70, 0.12, 0.28, 0.24])
        sample_ax.set_title("Samples", fontsize=9, loc="left")
        sample_ax.set_frame_on(False)
        short = [self._short(fn) for fn in self._sample_names]
        self._check = CheckButtons(sample_ax, short, [True] * len(self._sample_names))
        self._check.on_clicked(self._on_toggle)

        self._fig.text(0.70, 0.085, "Freq range (THz):", fontsize=8)
        fmin_ax = self._fig.add_axes([0.70, 0.04, 0.11, 0.035])
        fmax_ax = self._fig.add_axes([0.84, 0.04, 0.11, 0.035])
        self._tb_fmin = TextBox(fmin_ax, "", initial="", textalignment="center")
        self._tb_fmax = TextBox(fmax_ax, "", initial="", textalignment="center")
        self._tb_fmin.on_submit(self._on_fmin)
        self._tb_fmax.on_submit(self._on_fmax)

        # GC guard: matplotlib holds only weak refs to widget callbacks.
        self._fig._results_viewer_widgets = (
            self._radio, self._check, self._tb_fmin, self._tb_fmax)

        self._draw()

        interactive = "agg" not in matplotlib.get_backend().lower()
        if show and interactive:
            was_interactive = plt.isinteractive()
            try:
                if block:
                    plt.ioff()
                plt.show(block=block)
            finally:
                if was_interactive:
                    plt.ion()

    # -- helpers --
    @staticmethod
    def _short(filename: str) -> str:
        return ("..." + filename[-27:]) if len(filename) > 30 else filename

    def _quantity(self):
        return registry.QUANTITY_REGISTRY[self._current]

    # -- callbacks --
    def _on_quantity(self, label: str):
        self._current = self._names[self._labels.index(label)]
        self._draw()

    def _on_toggle(self, label: str):
        for fn in self._sample_names:
            if self._short(fn) == label:
                self._visible[fn] = not self._visible[fn]
                break
        self._draw()

    def _on_fmin(self, text: str):
        text = text.strip()
        self._freq_min = float(text) if text else None
        self._draw()

    def _on_fmax(self, text: str):
        text = text.strip()
        self._freq_max = float(text) if text else None
        self._draw()

    # -- drawing --
    def _iter_visible(self, quantity):
        for filename, data_obj in self._dataset.data.items():
            is_ref = self._dataset.data.is_reference(filename)
            if is_ref and not quantity.include_references:
                continue
            if (not is_ref) and not self._visible.get(filename, False):
                continue
            yield filename, data_obj

    def _draw(self):
        quantity = self._quantity()
        ax = self._ax
        ax.clear()
        ax.set_yscale(quantity.yscale)

        any_plotted = False
        for filename, data_obj in self._iter_visible(quantity):
            processing = data_obj.processing_dict
            result = None
            try:
                result = quantity.extract(processing)
            except Exception:
                result = None
            if result is None:
                continue
            freq_hz, values = result
            freq_thz = np.asarray(freq_hz) * _HZ_TO_THZ
            mask = processing.get(quantity.mask_key) if (self._show_snr_mask and quantity.mask_key) else None
            yerr = processing.get(quantity.error_key) if quantity.error_key else None
            colour = self._colours.get(filename)

            _plot_with_snr_mask(
                ax, freq_thz, np.asarray(values, dtype=float), mask,
                color=colour, label=self._short(filename), yerr=yerr,
                marker_every=self._marker_stride,
            )
            any_plotted = True

            if quantity.overlay is not None:
                try:
                    overlay = quantity.overlay(processing)
                except Exception:
                    overlay = None
                if overlay is not None:
                    overlay_freq_thz = np.asarray(overlay[0]) * _HZ_TO_THZ
                    overlay_y = np.asarray(overlay[1])
                    # Clip the model curve to the trusted band it was fit over, so it doesn't
                    # trail across the whole (mostly noise) FFT axis.
                    if mask is not None:
                        trusted = np.asarray(mask, dtype=bool)
                        if trusted.any():
                            lo, hi = freq_thz[trusted].min(), freq_thz[trusted].max()
                            in_band = (overlay_freq_thz >= lo) & (overlay_freq_thz <= hi)
                            overlay_freq_thz, overlay_y = overlay_freq_thz[in_band], overlay_y[in_band]
                    ax.plot(overlay_freq_thz, overlay_y, color=colour, lw=1.4, alpha=0.9,
                            zorder=5, ls="--")

        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(quantity.y_label)
        ax.set_title(quantity.label)
        ax.grid(alpha=0.3)
        if any_plotted:
            ax.legend(fontsize=8)
        if self._freq_min is not None or self._freq_max is not None:
            ax.set_xlim(left=self._freq_min, right=self._freq_max)
        self._fig.canvas.draw_idle()


def launch_results_viewer(dataset, *, show_snr_mask: bool = True, initial: str | None = None,
                          show: bool = True, block: bool = True) -> ResultsViewer:
    """Open the registry-driven results viewer on a dataset (live or loaded from a bundle)."""
    return ResultsViewer(dataset, show_snr_mask=show_snr_mask, initial=initial,
                         show=show, block=block)
