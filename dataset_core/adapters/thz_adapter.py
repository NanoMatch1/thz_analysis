import numpy as np
import thz_core as core
import matplotlib.pyplot as plt
from dataset_core.dataset import DataSet, DataService
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
# helpers
# ---------------------------------------------------------------------------

def _time_amplitude_array(data_obj) -> np.ndarray:
    """Extract (N, 2) [time_s, mean_amplitude] from a THzData object."""
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

def subtract_baseline(dataset: DataSet, config: dict | None = None, show_graph: bool = False) -> DataSet:
    """Subtract DC baseline from each trace (removes detector/digitiser offset)."""
    config = config or {}

    data_dict = _build_data_dict(dataset)
    corrected, metrics = core.subtract_baseline(data_dict, config)

    if show_graph:
        fig, ax = plt.subplots(2, 1, layout='constrained')
        for filename, data in corrected.items():
            ax[1].plot(data[:, 1], label='{} (corrected)'.format(filename))
            pre = data_dict[filename]
            ax[0].plot(pre[:, 1], label='{} (original)'.format(filename), linestyle='--', lw=1, alpha=0.5)
        # ax[0].legend()
        # ax[1].legend()
        ax[1].set_title('Baseline-Corrected Traces')
        ax[0].set_title('Original Traces')
        ax[1].set_xlabel('Time Point Index')
        plt.show()

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

    for filename, data_obj in dataset.data.items():       
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        windowed_y, metrics = core.window_time(t, y, config)

        new_data = np.column_stack((t, windowed_y))
        if data_obj.data.shape[1] > 2:
            new_data = np.column_stack((new_data, data_obj.data[:, 2:]))

        data_obj.data = new_data
        data_obj.processing_dict['window_metrics'] = metrics
        data_obj.processing_dict['pre-window'] = np.column_stack((t, y))

    if show_graph:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, layout='constrained')
        for filename, data_obj in dataset.data.items():
            data_pre = data_obj.processing_dict.get('pre-window')
            ax[0].plot(data_pre[:, 1], label='{} (original)'.format(filename), linestyle='--', lw=1, alpha=0.5)
            ax[1].plot(data_obj.data[:, 1], label='{} (windowed)'.format(filename))
        # ax[0].legend()
        # ax[1].legend()
        ax[0].set_title('Original Traces')
        ax[1].set_title('Windowed Traces')
        ax[1].set_xlabel('Time Point Index')    
        plt.show()

    return dataset

def plot_current(dataset: DataSet) -> None:
    """Plot the current time-domain traces for all files in the dataset."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]
        ax.plot(t * _S_TO_PS, y, label=filename)

    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Current Time-Domain Traces')
    ax.legend()
    plt.tight_layout()
    plt.show()


def zero_pad(dataset: DataSet, config: dict | None = None, show_graph: bool = False) -> DataSet:
    """Zero-pad all traces onto a common time grid."""
    config = config or {}

    data_dict = _build_data_dict(dataset)
    extend_factor = config.get('pad', {}).get('extend_factor', 1.0)

    t_common, padded_dict, metrics = core.pad_to_common_grid(data_dict)
    t_extended, extended_dict, metrics = core.extend_grid(t_common, padded_dict, extend_factor)

    for filename, data_obj in dataset.data.items():
        padded_y = extended_dict[filename]
        new_data = np.column_stack((t_extended, padded_y))
        data_obj.data = new_data
        data_obj.processing_dict['pad_metrics'] = metrics

    return dataset


def fft_spectrum(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Compute the FFT for each trace and switch data to frequency domain."""
    config = config or {}

    for filename, data_obj in dataset.data.items():
        t = data_obj.data[:, 0]
        y = data_obj.data[:, 1]

        # Snapshot the pre-FFT time trace so export_results can write it later
        data_obj.processing_dict['time_domain_prefft'] = np.column_stack((t, y))

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


def trusted_band_mask(dataset: DataSet, config: dict | None = None) -> DataSet:
    """Compute SNR-based trusted-band mask for each sample-reference pair.

    Must be called after transfer_function and before invert_nk.
    Replaces the finite-only 'transfer_mask' with a tighter SNR-filtered mask.
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
        mask, _mask_metrics = core.trusted_band_mask(freq, Y_air, Y_sub, H, config)

        data_obj.processing_dict['transfer_H'] = H
        data_obj.processing_dict['transfer_mask'] = mask

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


def plot_fft(dataset: DataSet, freq_range: tuple | None = None, normalise: bool = False) -> None:
    """Plot FFT magnitude for every file (samples and references)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for filename, data_obj in dataset.data.items():
        freq = data_obj.processing_dict.get('fft_freq')
        spectrum = data_obj.processing_dict.get('fft_spectrum')
        if freq is None or spectrum is None:
            continue
        # ax.semilogy(freq * _HZ_TO_THZ, np.abs(spectrum), label=filename)
        norm = np.abs(spectrum).max() if normalise else 1.0
        ax.plot(freq * _HZ_TO_THZ, np.abs(spectrum) / norm, label=filename)


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
        ax.plot(freq * _HZ_TO_THZ, np.abs(H), label=filename)

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
        ax.plot(freq * _HZ_TO_THZ, phase, label=filename)

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
        ax_n.plot(freq * _HZ_TO_THZ, n, label=filename)
        ax_k.plot(freq * _HZ_TO_THZ, k, label=filename)

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
        ax_r.plot(freq * _HZ_TO_THZ, eps.real, label=filename)
        ax_i.plot(freq * _HZ_TO_THZ, eps.imag, label=filename)

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
        ax_r.plot(freq * _HZ_TO_THZ, sigma.real, label=filename)
        ax_i.plot(freq * _HZ_TO_THZ, sigma.imag, label=filename)

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
        - ``'transfer'`` – unwrapped phase of H(f)  (default)
        - ``'fft'``      – unwrapped phase of the raw FFT spectrum

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
        corrected_unwrapped = unwrapped - correction

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
            ax.semilogy(freq * _HZ_TO_THZ, np.abs(spec), color=self._colours[fn],
                        label=self._short(fn), alpha=0.9 if is_sample else 0.4)
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
            ax.plot(freq * _HZ_TO_THZ, phase, color=self._colours[fn],
                    label=self._short(fn), alpha=0.9 if is_sample else 0.4)
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
            ax.plot(freq * _HZ_TO_THZ, np.abs(H), color=self._colours[fn], label=self._short(fn))
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
            ax.plot(freq * _HZ_TO_THZ, phase, color=self._colours[fn], label=self._short(fn))
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
            ax_n.plot(freq * _HZ_TO_THZ, n, color=self._colours[fn], label=self._short(fn))
            ax_k.plot(freq * _HZ_TO_THZ, k, color=self._colours[fn], label=self._short(fn))
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
            ax_r.plot(freq * _HZ_TO_THZ, eps.real, color=self._colours[fn], label=self._short(fn))
            ax_i.plot(freq * _HZ_TO_THZ, eps.imag, color=self._colours[fn], label=self._short(fn))
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
            ax_r.plot(freq * _HZ_TO_THZ, sigma.real, color=self._colours[fn], label=self._short(fn))
            ax_i.plot(freq * _HZ_TO_THZ, sigma.imag, color=self._colours[fn], label=self._short(fn))
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
