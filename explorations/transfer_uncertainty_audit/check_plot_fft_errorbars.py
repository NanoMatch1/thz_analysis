"""Headless workflow check for the plot_fft error bars + resolution-grid markers.

Runs the run_me_reflection_single stage order on the live dataset, then exercises:
  * plot_fft BEFORE apply_instrument_resolution (oversampled arrays -> stride must be the
    decimation factor, not 1)
  * plot_fft AFTER  apply_instrument_resolution (already decimated -> stride 1)
  * the registry-driven display layer, which should now pick up the FFT error bars for free
Asserts the artefacts exist and that the markers/error bars actually land on the plot.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz
from dataset_core.adapters import display, quantity_registry

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = r'C:\Users\Samuel\Data\THz\CNTs\2026-08-20_CNT-paper-doped_2\export'

config: dict = {
    "general": {'show_graph': False},
    "resolution": {"limit_to_instrument_resolution": True, "broadening_factor": 1.0},
    "geometry": {"theta_external_deg": 45.0, "polarization": "s", "r_reference": -1.0},
    "regions": {"pulse": None},
    "centering": {"recalibrate": False, "target_t0_ps": None, "subsample": False},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 5},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": 2000},
    "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                 "regularization_eps": 1e-30, "unwrap_phase": True,
                 "correct_linear_phase": False},
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
    "invert": {"min_one_plus_r": 0.1},
    "derive": {"eps_background": 11.7},
}

captured = []
original_show = plt.show
plt.show = lambda *args, **kwargs: captured.append(plt.gcf())

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
dataset.group_files(keywords=['type'])
thz.subtract_baseline(dataset)
thz.window_single_pulse_fixed_width(dataset, half_width_ps=5, region_ps=None, show_graph=False)
thz.fft_spectrum(dataset, n_fft=2000)
thz.transfer_function(dataset, ref_type='reference')
thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)

print("\n--- artefacts written by compute_spectrum_uncertainty ---")
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    sigma = processing.get('fft_sigma')
    phase_sigma = processing.get('fft_phase_sigma')
    assert sigma is not None, f"{filename}: fft_sigma missing"
    assert phase_sigma is not None, f"{filename}: fft_phase_sigma missing"
    assert sigma.shape == np.asarray(processing['fft_spectrum']).shape
    print(f"  {filename}: fft_sigma {sigma.shape} const={np.ptp(sigma) == 0} "
          f"value={sigma[0]:.3e}; fft_phase_sigma range "
          f"{phase_sigma.min():.3e}..{phase_sigma.max():.3e} rad")

stride_before = thz._resolution_marker_stride(dataset)
print(f"\nmarker stride BEFORE apply_instrument_resolution: {stride_before} "
      f"(decimation_factor = {config['resolution']['decimation_factor']})")
assert stride_before == config['resolution']['decimation_factor'], \
    "plot markers would land on interpolated bins"

thz.plot_fft(dataset, normalise=False, scale='')
figure = captured[-1]
axis = figure.axes[0]
marker_lines = [line for line in axis.lines if line.get_linestyle() == 'None'
                and len(line.get_xdata()) > 0]
error_containers = axis.containers
print(f"  plot_fft: {len(marker_lines)} marker series, {len(error_containers)} error-bar "
      f"containers, {len(axis.patches)} shaded untrusted bands")
assert error_containers, "no error bars drawn on plot_fft"
figure.savefig(os.path.join(OUT_DIR, 'plot_fft_before_resolution.png'), dpi=120)

thz.apply_instrument_resolution(dataset, config)
stride_after = thz._resolution_marker_stride(dataset)
print(f"\nmarker stride AFTER apply_instrument_resolution: {stride_after}")
assert stride_after == 1, "already-decimated data should mark every bin"

thz.plot_fft(dataset, normalise=False, scale='log')
figure = captured[-1]
figure.savefig(os.path.join(OUT_DIR, 'plot_fft_after_resolution.png'), dpi=120)

print("\n--- registry-driven display layer picks up the same bars ---")
for name in ('fft_mag', 'fft_phase'):
    spec = quantity_registry.QUANTITY_REGISTRY[name]
    print(f"  {name}: error_key = {spec.error_key}")
    assert spec.error_key is not None
series = display.get_series(dataset, 'fft_mag')
for filename, entry in series.items():
    assert entry['error'] is not None, f"display.get_series gave no error for {filename}"
    print(f"  get_series('fft_mag')['{filename}']: {entry['y'].size} points, "
          f"error {entry['error'][0]:.3e}")
figure = display.plot_quantity(dataset, 'fft_mag')
figure.savefig(os.path.join(OUT_DIR, 'display_fft_mag.png'), dpi=120)

plt.show = original_show
print("\nAll plot_fft / display checks passed.")
