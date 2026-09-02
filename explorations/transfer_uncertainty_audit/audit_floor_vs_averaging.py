"""Audit part 4: the decisive test — does the estimated "noise floor" respond to averaging?

A genuine additive noise floor on an averaged spectrum must fall as 1/sqrt(N_scans). If the
tail-median estimator is instead measuring reproducible signal content, it will barely move
when N_scans changes — which is exactly the reported symptom ("I improved the noise and the
transfer-function error bars did not improve").

Sub-samples the scans of each file, re-averages, and recomputes:
  * the pipeline's tail-median floor        (thz_core.transfer._noise_floor)
  * the measured random scatter of the mean (std across scans / sqrt(N))
and reports how each scales with N.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_core import DataSet
from dataset_core.adapters import thz_adapter as thz

_HZ_TO_THZ = 1e-12
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_FFT = 2000
TAIL_FRACTION = 0.25

data_dir = r'C:\Users\Samuel\Data\THz\CNTs\2026-08-20_CNT-paper-doped_2\export'

config: dict = {
    "general": {'show_graph': False},
    "resolution": {"limit_to_instrument_resolution": False, "broadening_factor": 1.0},
    "geometry": {"theta_external_deg": 45.0, "polarization": "s", "r_reference": -1.0},
    "regions": {"pulse": None},
    "centering": {"recalibrate": False, "target_t0_ps": None, "subsample": False},
    "window": {"type": "hann", "alpha": 1.0, "half_width_ps": 5},
    "fft": {"norm": "backward", "amplitude_scale": 1.0, "n_fft": N_FFT},
    "transfer": {"self_reference": False, "apply_snr_mask": True, "min_ref_amp_rel": 1e-3,
                 "regularization_eps": 1e-30, "unwrap_phase": True,
                 "correct_linear_phase": False},
    "mask": {"snr_thresh_db": 10, "tail_fraction": TAIL_FRACTION, "min_contiguous_bins": 3},
    "invert": {"min_one_plus_r": 0.1},
    "derive": {"eps_background": 11.7},
}

dataset = DataSet(data_dir, config=config)
dataset.load_all_data()
thz.taper_and_pad_traces_universal(dataset, taper_ps=0.5)
dataset.group_files(keywords=['type'])
thz.subtract_baseline(dataset)
thz.window_single_pulse_fixed_width(dataset, half_width_ps=5, region_ps=None, show_graph=False)
thz.fft_spectrum(dataset, n_fft=N_FFT)
thz.transfer_function(dataset, ref_type='reference')

report_lines = []


def emit(text=""):
    print(text)
    report_lines.append(str(text))


def rebuild_pad_and_taper(amplitude, centering_info):
    n_prepend = int(centering_info['n_prepend'])
    n_append = int(centering_info['n_append'])
    taper_samples = int(centering_info['taper_samples'])
    if n_prepend == 0 and n_append == 0:
        return amplitude.copy()
    padded = np.concatenate([np.zeros(n_prepend), amplitude.copy(), np.zeros(n_append)])
    if n_prepend > 0 and taper_samples > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_samples, endpoint=False)))
        padded[n_prepend:n_prepend + taper_samples] *= ramp
    elif n_append > 0 and taper_samples > 0:
        end = n_prepend + amplitude.size
        ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper_samples, endpoint=False)))
        padded[end - taper_samples:end] *= ramp
    return padded


def clipped_window(n_samples, peak_index, half_width_samples):
    core = thz._build_symmetric_window(2 * half_width_samples + 1, 'hann', 1.0)
    lo, hi = peak_index - half_width_samples, peak_index + half_width_samples + 1
    data_lo, data_hi = max(lo, 0), min(hi, n_samples)
    core_lo = data_lo - lo
    window_function = np.zeros(n_samples)
    window_function[data_lo:data_hi] = core[core_lo:core_lo + (data_hi - data_lo)]
    return window_function


def noise_floor(magnitude, tail_fraction=TAIL_FRACTION):
    tail_points = max(8, int(np.ceil(magnitude.size * tail_fraction)))
    return max(float(np.median(np.abs(magnitude[-tail_points:]))), 1e-30)


emit("=" * 96)
emit("DOES THE ESTIMATED FLOOR FALL AS 1/sqrt(N_scans), THE WAY A REAL NOISE FLOOR MUST?")
emit("=" * 96)
emit("  tail floor  = median |Y_avg| over the top 25% of the frequency axis (what the pipeline uses)")
emit("  meas. noise = median over the same band of std_across_scans/sqrt(N) (the true random part)")
emit("  ideal       = the 1/sqrt(N) scaling a genuine additive floor would follow")

for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    windowed = processing['time_domain_prefft']
    n_samples = windowed.shape[0]
    dt_s = float(np.median(np.diff(windowed[:, 0])))
    fixed_window = processing['fixed_window']
    window_function = clipped_window(n_samples, int(fixed_window['peak_index']),
                                     int(fixed_window['half_width_samples']))
    raw = np.asarray(data_obj.raw_data, dtype=float)
    centering_info = processing['centering_info']
    stack = np.asarray([
        (lambda padded: padded - np.mean(padded[:10]))(
            rebuild_pad_and_taper(raw[:, column], centering_info))
        for column in range(1, raw.shape[1])
    ])
    spectra = np.fft.rfft(stack * window_function, n=N_FFT, axis=1)
    n_total = spectra.shape[0]
    tail_points = max(8, int(np.ceil(spectra.shape[1] * TAIL_FRACTION)))
    tail = slice(spectra.shape[1] - tail_points, spectra.shape[1])

    emit()
    emit(f"--- {filename}  ({n_total} scans total)")
    emit(f"    {'N used':>7} {'tail floor':>12} {'meas. noise':>12} {'floor/noise':>12} "
         f"{'floor vs N=all':>15} {'ideal 1/sqrtN':>14}")
    reference_floor = None
    for n_used in sorted({4, 8, 16, 32, min(64, n_total), n_total}):
        if n_used > n_total:
            continue
        subset = spectra[:n_used]
        mean_spectrum = subset.mean(axis=0)
        floor = noise_floor(np.abs(mean_spectrum))
        component_sigma = np.sqrt(0.5 * (subset.real.var(axis=0, ddof=1)
                                         + subset.imag.var(axis=0, ddof=1)))
        measured = np.median(component_sigma[tail]) / np.sqrt(n_used)
        if reference_floor is None:
            reference_floor = floor
            first_n = n_used
        emit(f"    {n_used:7d} {floor:12.4e} {measured:12.4e} {floor / measured:12.1f} "
             f"{floor / reference_floor:15.3f} {np.sqrt(first_n / n_used):14.3f}")

with open(os.path.join(OUT_DIR, 'audit_report_part4.txt'), 'w') as handle:
    handle.write("\n".join(report_lines))
print(f"\nWrote {os.path.join(OUT_DIR, 'audit_report_part4.txt')}")
