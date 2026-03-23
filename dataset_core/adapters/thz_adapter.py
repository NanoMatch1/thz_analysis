import numpy as np
import thz_core as core
from dataset_core.dataset import DataSet, DataService
"""Used to bridge the DataSet manager and the thz analysis library.

Minimal adapter layer: each function extracts arrays from THzData objects,
calls the corresponding thz_core routine, and writes results back onto
the dataset (in-place). All functions return the dataset for chaining.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _time_amplitude_array(data_obj) -> np.ndarray:
    """Extract (N, 2) [time, mean_amplitude] from a THzData object."""
    return data_obj.data[:, :2].copy()


def _build_data_dict(dataset: DataSet) -> dict:
    """Build {filename: (N,2) array} from all current data objects."""
    return {
        filename: _time_amplitude_array(data_obj)
        for filename, data_obj in dataset.data.items()
    }


# ---------------------------------------------------------------------------
# pipeline steps
# ---------------------------------------------------------------------------

def subtract_baseline(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Subtract DC baseline from each trace (removes detector/digitiser offset)."""
    config = config or {}

    data_dict = _build_data_dict(dataset)
    corrected, metrics = core.subtract_baseline(data_dict, config)

    for filename, data_obj in dataset.data.items():
        data_obj.data = corrected[filename]
        data_obj.processing_dict['baseline_metrics'] = metrics

    return dataset


def align_on_peak(dataset: DataSet, show_graph: bool = False, auto_range: tuple = None) -> DataSet:
    """Aligns all acquisitions in the dataset on their main peak."""

    data_dict = _build_data_dict(dataset)
    aligned = core.align_on_peak(data_dict, auto_range=auto_range)

    if show_graph:
        import matplotlib.pyplot as plt
        for filename, data in aligned.items():
            plt.plot(data[:, 1], label=filename)
        plt.legend()
        plt.show()

    for filename, data_obj in dataset.data.items():
        data_obj.data = aligned[filename]

    return dataset


def window_time(dataset: DataSet, config: dict | None = None, show_graph: bool = False) -> DataSet:
    """Apply a time-domain window to each trace in the dataset."""
    config = config or {}

    import matplotlib.pyplot as plt
    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        windowed_y, metrics = core.window_time(t, y, config)

        new_data = np.column_stack((t, windowed_y))
        if data_obj.data.shape[1] > 2:
            new_data = np.column_stack((new_data, data_obj.data[:, 2:]))

        data_obj.data = new_data
        data_obj.processing_dict['window_metrics'] = metrics

        if show_graph:
            plt.plot(y, label=f'{filename} original')
            plt.plot(windowed_y, label=f'{filename} windowed')
    plt.legend()
    plt.show()

    return dataset


def zero_pad(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Zero-pad all traces onto a common time grid."""
    config = config or {}

    data_dict = _build_data_dict(dataset)
    t_common, padded_dict, metrics = core.zero_pad(data_dict, config)

    for filename, data_obj in dataset.data.items():
        padded_y = padded_dict[filename]
        new_data = np.column_stack((t_common, padded_y))
        data_obj.data = new_data
        data_obj.processing_dict['pad_metrics'] = metrics

    return dataset


def fft_spectrum(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Compute the FFT for each trace and switch data to frequency domain."""
    config = config or {}

    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        freq, spectrum, metrics = core.fft_spectrum(t, y, config)

        data_obj.processing_dict['fft_freq'] = freq
        data_obj.processing_dict['fft_spectrum'] = spectrum
        data_obj.processing_dict['fft_metrics'] = metrics

        # Store magnitude + phase as the new "data" for plotting convenience
        data_obj.data = np.column_stack((
            freq,
            np.abs(spectrum),
            np.angle(spectrum),
        ))
        data_obj.current_state = 'frequency_domain'

    return dataset


def transfer_function(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Compute H(f) = Y_sample / Y_reference for each sample-reference pair."""
    config = config or {}

    for filename, data_obj in dataset.data.items():
        if dataset.data.is_reference(filename):
            continue

        ref_obj = dataset.get_reference(filename, ref_type='substrate')
        if ref_obj is None:
            print(f"Warning: no reference found for '{filename}', skipping transfer function.")
            continue

        freq = data_obj.processing_dict['fft_freq']
        Y_samp = data_obj.processing_dict['fft_spectrum']
        Y_ref = ref_obj.processing_dict['fft_spectrum']

        H, valid_mask, metrics = core.transfer_function(freq, Y_samp, Y_ref, config)

        data_obj.processing_dict['transfer_H'] = H
        data_obj.processing_dict['transfer_mask'] = valid_mask
        data_obj.processing_dict['transfer_metrics'] = metrics
        data_obj.reference_filename = ref_obj.filename

        data_obj.data = np.column_stack((
            freq,
            np.abs(H),
            np.angle(H),
        ))

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
# export
# ---------------------------------------------------------------------------

def export_results(dataset: DataSet, export_dir: str | None = None) -> list[str]:
    """Export per-sample CSV files containing all computed arrays.

    One file per sample. Each file has columns:
    freq_THz, fft_mag, n, k, eps_real, eps_imag, sigma_real, sigma_imag.

    Returns list of written file paths.
    """
    import os

    if export_dir is None:
        export_dir = os.path.join(dataset.file_dir, 'results')
    os.makedirs(export_dir, exist_ok=True)

    written = []
    for fn, data_obj in _sample_items(dataset):
        proc = data_obj.processing_dict
        freq = proc.get('fft_freq')
        if freq is None:
            print(f"Skipping '{fn}': no FFT data.")
            continue

        nan_col = np.full_like(freq, np.nan)

        def _safe(arr):
            return arr if arr is not None else nan_col

        spec = proc.get('fft_spectrum')
        eps = proc.get('eps')
        sigma = proc.get('sigma')

        columns = [
            freq,
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

        stem = os.path.splitext(fn)[0]
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


def plot_fft(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot FFT magnitude for every file (samples and references)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in dataset.data.items():
        freq = data_obj.processing_dict.get('fft_freq')
        spectrum = data_obj.processing_dict.get('fft_spectrum')
        if freq is None or spectrum is None:
            continue
        ax.semilogy(freq, np.abs(spectrum), label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('|FFT|')
    ax.set_title('FFT Magnitude')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_transfer_function(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot transfer function magnitude for each sample."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in _sample_items(dataset):
        H = data_obj.processing_dict.get('transfer_H')
        freq = data_obj.processing_dict.get('fft_freq')
        if H is None or freq is None:
            continue
        ax.plot(freq, np.abs(H), label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('|H(f)|')
    ax.set_title('Transfer Function Magnitude')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_transfer_phase(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot unwrapped phase of the transfer function for each sample."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in _sample_items(dataset):
        H = data_obj.processing_dict.get('transfer_H')
        freq = data_obj.processing_dict.get('fft_freq')
        if H is None or freq is None:
            continue
        phase = np.unwrap(np.angle(H))
        ax.plot(freq, phase, label=filename)

    if freq_range is not None:
        ax.set_xlim(freq_range)
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('Phase (rad, unwrapped)')
    ax.set_title('Transfer Function Phase (unwrapped)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_nk(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot refractive index n and extinction coefficient k for each sample."""
    import matplotlib.pyplot as plt

    fig, (ax_n, ax_k) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        n = data_obj.processing_dict.get('n')
        k = data_obj.processing_dict.get('k')
        if freq is None or n is None or k is None:
            continue
        ax_n.plot(freq, n, label=filename)
        ax_k.plot(freq, k, label=filename)

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


def plot_permittivity(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot real and imaginary parts of the complex permittivity."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        eps = data_obj.processing_dict.get('eps')
        if freq is None or eps is None:
            continue
        ax_r.plot(freq, eps.real, label=filename)
        ax_i.plot(freq, eps.imag, label=filename)

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


def plot_conductivity(dataset: DataSet, freq_range: tuple | None = None) -> None:
    """Plot real and imaginary parts of the optical conductivity."""
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_i) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for filename, data_obj in _sample_items(dataset):
        freq = data_obj.processing_dict.get('fft_freq')
        sigma = data_obj.processing_dict.get('sigma')
        if freq is None or sigma is None:
            continue
        ax_r.plot(freq, sigma.real, label=filename)
        ax_i.plot(freq, sigma.imag, label=filename)

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
        'Transfer |H|',
        'Transfer Phase',
        'n  (refractive index)',
        'k  (extinction)',
        'Permittivity',
        'Conductivity',
    ]

    def __init__(self, dataset: DataSet):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons, CheckButtons, TextBox

        self._plt = plt
        self._dataset = dataset

        # Collect sample filenames (ordered) and reference filenames
        self._sample_names = [
            fn for fn, _ in _sample_items(dataset)
        ]
        self._all_names = [fn for fn in dataset.data.keys()]
        self._visible = {fn: True for fn in self._sample_names}

        # Assign a stable colour per sample for consistency across views
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

    def _draw_fft(self):
        ax = self._ax_top
        # Plot references too (grey)
        for fn in self._all_names:
            freq = self._get(fn, 'fft_freq')
            spec = self._get(fn, 'fft_spectrum')
            if freq is None or spec is None:
                continue
            is_sample = fn in self._sample_names
            if is_sample and not self._visible.get(fn, False):
                continue
            ax.semilogy(freq, np.abs(spec), color=self._colours[fn],
                        label=self._short(fn), alpha=0.9 if is_sample else 0.4)
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('|FFT|')
        ax.set_title('FFT Magnitude')
        ax.legend(fontsize=7, loc='upper right')

    def _draw_transfer_mag(self):
        ax = self._ax_top
        for fn in self._visible_samples():
            H = self._get(fn, 'transfer_H')
            freq = self._get(fn, 'fft_freq')
            if H is None or freq is None:
                continue
            ax.plot(freq, np.abs(H), color=self._colours[fn], label=self._short(fn))
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
            ax.plot(freq, phase, color=self._colours[fn], label=self._short(fn))
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
            ax_n.plot(freq, n, color=self._colours[fn], label=self._short(fn))
            ax_k.plot(freq, k, color=self._colours[fn], label=self._short(fn))
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
            ax_r.plot(freq, eps.real, color=self._colours[fn], label=self._short(fn))
            ax_i.plot(freq, eps.imag, color=self._colours[fn], label=self._short(fn))
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
            ax_r.plot(freq, sigma.real, color=self._colours[fn], label=self._short(fn))
            ax_i.plot(freq, sigma.imag, color=self._colours[fn], label=self._short(fn))
        ax_r.set_ylabel(r'$\sigma_r$ (S/m)')
        ax_r.set_title('Optical Conductivity — Real')
        ax_r.legend(fontsize=7, loc='upper right')
        ax_i.set_xlabel('Frequency (THz)')
        ax_i.set_ylabel(r'$\sigma_i$ (S/m)')
        ax_i.set_title('Optical Conductivity — Imaginary')
        ax_i.legend(fontsize=7, loc='upper right')


def result_viewer(dataset: DataSet) -> ResultViewer:
    """Launch the interactive result viewer GUI."""
    return ResultViewer(dataset)