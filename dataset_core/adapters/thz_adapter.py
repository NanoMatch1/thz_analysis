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
            plt.plot(data[:, 0], data[:, 1], label=filename)
        plt.legend()
        plt.show()

    for filename, data_obj in dataset.data.items():
        data_obj.data = aligned[filename]

    return dataset


def window_time(dataset: DataSet, config: dict | None = None) -> DataSet:
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