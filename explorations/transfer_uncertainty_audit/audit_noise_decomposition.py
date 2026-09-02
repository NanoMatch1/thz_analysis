"""Audit part 2: decompose the measured spectral noise and test the leakage hypothesis.

Part 1 showed the pipeline's flat noise floor sits ~4-5x above the scan-to-scan standard
error in the "tail" band it is measured from. This script asks *why*, and what the honest
error bar on H actually is:

  1. IS THE TAIL A NOISE PLATEAU? A flat additive floor should make |Y| level off at high
     frequency. Check whether |Y| is still falling through 7.5-10 THz (=> the "floor" is
     really residual signal / spectral leakage, and would NOT drop when the instrument
     noise drops).

  2. WHERE DOES THE HIGH-FREQUENCY CONTENT COME FROM? The fixed-width Hann window is
     CLIPPED by the trace edge (printed by the pipeline), so it does not return to zero at
     the end of the record. Re-transform with (a) the pipeline window, (b) the same window
     forced to zero at the trailing edge, (c) a fully-contained (unclipped) window, and
     compare the 7-10 THz level.

  3. WHITE vs DRIFT. Consecutive-scan differences see only the white part of the scan-to-
     scan scatter; block (batch-means) statistics see the slow drift as well. Split the
     measured variance into the two, because they scale differently with averaging.

  4. THE HONEST ERROR BAR ON H. Split the scans of BOTH sample and reference into blocks,
     form an independent H per block, and take the scatter of the mean directly. This is an
     end-to-end empirical uncertainty on H that assumes no noise model at all.
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
thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)

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


def clipped_window(n_samples, peak_index, half_width_samples, window_type='hann', alpha=1.0):
    window_length = 2 * half_width_samples + 1
    core = thz._build_symmetric_window(window_length, window_type, alpha)
    lo, hi = peak_index - half_width_samples, peak_index + half_width_samples + 1
    data_lo, data_hi = max(lo, 0), min(hi, n_samples)
    core_lo = data_lo - lo
    window_function = np.zeros(n_samples)
    window_function[data_lo:data_hi] = core[core_lo:core_lo + (data_hi - data_lo)]
    return window_function


def padded_scans(data_obj):
    raw = np.asarray(data_obj.raw_data, dtype=float)
    centering_info = data_obj.processing_dict['centering_info']
    stack = []
    for column in range(1, raw.shape[1]):
        padded = rebuild_pad_and_taper(raw[:, column], centering_info)
        stack.append(padded - np.mean(padded[:10]))
    return np.asarray(stack)


def block_means(stack, n_blocks):
    """Split rows into n_blocks contiguous blocks (acquisition order) and average each."""
    edges = np.linspace(0, stack.shape[0], n_blocks + 1).astype(int)
    return np.asarray([stack[edges[i]:edges[i + 1]].mean(axis=0) for i in range(n_blocks)])


DIAGNOSTIC_FREQS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

emit("=" * 96)
emit("1+2. IS THE 'NOISE FLOOR' BAND A NOISE PLATEAU, OR STILL SIGNAL / WINDOW LEAKAGE?")
emit("=" * 96)

spectra_cache = {}
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    windowed = processing['time_domain_prefft']
    n_samples = windowed.shape[0]
    dt_s = float(np.median(np.diff(windowed[:, 0])))
    fixed_window = processing['fixed_window']
    peak_index = int(fixed_window['peak_index'])
    half_width_samples = int(fixed_window['half_width_samples'])

    stack = padded_scans(data_obj)
    averaged_trace = stack.mean(axis=0)

    window_pipeline = clipped_window(n_samples, peak_index, half_width_samples)

    # (b) same window, but forced smoothly to zero over the last 10 samples of the record
    window_zero_terminated = window_pipeline.copy()
    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, 10, endpoint=False)))
    window_zero_terminated[-10:] *= fade

    # (c) largest fully-contained symmetric window (never clipped by the trace edge)
    contained_half_width = min(peak_index, n_samples - 1 - peak_index)
    window_contained = clipped_window(n_samples, peak_index, contained_half_width)

    freq = np.fft.rfftfreq(N_FFT, dt_s)
    freq_thz = freq * _HZ_TO_THZ
    variants = {
        'pipeline (clipped)': window_pipeline,
        'zero-terminated': window_zero_terminated,
        f'contained (+/-{contained_half_width * dt_s * 1e12:.2f} ps)': window_contained,
    }
    spectra_cache[filename] = dict(freq_thz=freq_thz, stack=stack, dt_s=dt_s,
                                   window_pipeline=window_pipeline, variants=variants)

    emit()
    emit(f"--- {filename}")
    emit(f"    windowed record: {n_samples} samples, peak at index {peak_index}, "
         f"window half-width {half_width_samples} samples "
         f"({half_width_samples * dt_s * 1e12:.2f} ps) -> "
         f"{'CLIPPED' if 2 * half_width_samples + 1 > n_samples else 'contained'}")
    emit(f"    window edge values: start {window_pipeline[0]:.4f}, end {window_pipeline[-1]:.4f}")
    emit(f"    trace value at last sample: {averaged_trace[-1]:.3e} "
         f"(peak {np.max(np.abs(averaged_trace)):.3e}) -> step into the pad = "
         f"{abs(averaged_trace[-1] * window_pipeline[-1]):.3e}")
    header = f"    {'f (THz)':>8}" + "".join(f"{name:>26}" for name in variants)
    emit(header)
    for target in DIAGNOSTIC_FREQS:
        index = int(np.argmin(np.abs(freq_thz - target)))
        row = f"    {freq_thz[index]:8.2f}"
        for window_function in variants.values():
            value = np.abs(np.fft.rfft(averaged_trace * window_function, n=N_FFT)[index])
            row += f"{value:26.4e}"
        emit(row)
    for name, window_function in variants.items():
        magnitude = np.abs(np.fft.rfft(averaged_trace * window_function, n=N_FFT))
        tail_points = max(8, int(np.ceil(magnitude.size * TAIL_FRACTION)))
        emit(f"    median |Y| over 7.5-10 THz, {name:>28}: "
             f"{np.median(magnitude[-tail_points:]):.4e}")

emit()
emit("=" * 96)
emit("3. WHITE (scan-to-scan) vs DRIFT (slow) COMPONENTS OF THE SPECTRAL NOISE")
emit("=" * 96)
emit("   sigma_white = std of consecutive-scan differences / sqrt(2) / sqrt(N)  (drift-blind)")
emit("   sigma_block = batch-means scatter over 8 contiguous blocks (sees drift)")

noise_estimates = {}
for filename, cached in spectra_cache.items():
    stack = cached['stack']
    window_function = cached['window_pipeline']
    freq_thz = cached['freq_thz']
    spectra = np.fft.rfft(stack * window_function, n=N_FFT, axis=1)
    n_scans = spectra.shape[0]
    mean_spectrum = spectra.mean(axis=0)

    differences = np.diff(spectra, axis=0)
    white_component = np.sqrt(0.5 * (differences.real.var(axis=0, ddof=1)
                                     + differences.imag.var(axis=0, ddof=1)) / 2.0)
    sigma_white_mean = white_component / np.sqrt(n_scans)

    n_blocks = 8
    blocks = block_means(spectra, n_blocks)
    block_component = np.sqrt(0.5 * (blocks.real.var(axis=0, ddof=1)
                                     + blocks.imag.var(axis=0, ddof=1)))
    sigma_block_mean = block_component / np.sqrt(n_blocks)

    noise_estimates[filename] = dict(mean_spectrum=mean_spectrum,
                                     sigma_white=sigma_white_mean,
                                     sigma_block=sigma_block_mean,
                                     freq_thz=freq_thz, n_scans=n_scans)
    emit()
    emit(f"--- {filename}  ({n_scans} scans)")
    emit(f"    {'f (THz)':>8} {'|Y|':>11} {'sig_white':>11} {'sig_block':>11} "
         f"{'rel_white':>10} {'rel_block':>10} {'block/white':>12}")
    for target in DIAGNOSTIC_FREQS:
        index = int(np.argmin(np.abs(freq_thz - target)))
        magnitude = np.abs(mean_spectrum[index])
        white = sigma_white_mean[index]
        block = sigma_block_mean[index]
        emit(f"    {freq_thz[index]:8.2f} {magnitude:11.3e} {white:11.3e} {block:11.3e} "
             f"{white / magnitude:10.4f} {block / magnitude:10.4f} {block / white:12.1f}")

emit()
emit("=" * 96)
emit("4. END-TO-END EMPIRICAL UNCERTAINTY ON H (no noise model at all)")
emit("=" * 96)
emit("   H_i = mean(Y_sample, block i) / mean(Y_reference, block i);  sigma_mean = std_i / sqrt(n_blocks)")

n_blocks = 6
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    transfer_H = processing.get('transfer_H')
    if transfer_H is None:
        continue
    reference_obj = dataset.get_reference(filename, ref_type='reference')
    reference_name = getattr(reference_obj, 'filename', None)

    sample_cached = spectra_cache[filename]
    reference_cached = spectra_cache[reference_name]
    freq_thz = sample_cached['freq_thz']

    sample_spectra = np.fft.rfft(sample_cached['stack'] * sample_cached['window_pipeline'],
                                 n=N_FFT, axis=1)
    reference_spectra = np.fft.rfft(reference_cached['stack'] * reference_cached['window_pipeline'],
                                    n=N_FFT, axis=1)
    sample_blocks = block_means(sample_spectra, n_blocks)
    reference_blocks = block_means(reference_spectra, n_blocks)
    transfer_blocks = sample_blocks / reference_blocks

    magnitude_blocks = np.abs(transfer_blocks)
    phase_blocks = np.angle(transfer_blocks)
    sigma_magnitude = magnitude_blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)
    sigma_phase = phase_blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

    pipeline_sigma = np.asarray(processing['transfer_H_sigma'])
    pipeline_phase_sigma = np.asarray(processing['transfer_phase_sigma'])
    magnitude_H = np.abs(transfer_H)

    emit()
    emit(f"--- {filename}   (reference: {reference_name}, {n_blocks} blocks)")
    emit(f"    {'f (THz)':>8} {'|H|':>8} {'sig|H| pipe':>12} {'sig|H| meas':>12} {'ratio':>7}"
         f" {'sigPhi pipe':>12} {'sigPhi meas':>12} {'ratio':>7}")
    for target in DIAGNOSTIC_FREQS:
        index = int(np.argmin(np.abs(freq_thz - target)))
        emit(f"    {freq_thz[index]:8.2f} {magnitude_H[index]:8.3f} "
             f"{pipeline_sigma[index]:12.4e} {sigma_magnitude[index]:12.4e} "
             f"{pipeline_sigma[index] / max(sigma_magnitude[index], 1e-30):7.1f} "
             f"{pipeline_phase_sigma[index]:12.4e} {sigma_phase[index]:12.4e} "
             f"{pipeline_phase_sigma[index] / max(sigma_phase[index], 1e-30):7.1f}")

with open(os.path.join(OUT_DIR, 'audit_report_part2.txt'), 'w') as handle:
    handle.write("\n".join(report_lines))
print(f"\nWrote {os.path.join(OUT_DIR, 'audit_report_part2.txt')}")
