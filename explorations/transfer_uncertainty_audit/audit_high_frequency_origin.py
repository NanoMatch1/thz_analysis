"""Audit part 3: what IS the reproducible 7-10 THz content, and is the block estimator stable?

Part 2 showed the "noise floor" band (7.5-10 THz) is dominated by content that repeats
scan-to-scan (~5x above the measured random scatter) and is unaffected by the window
treatment. Two remaining questions:

  1. WHERE IS IT IN TIME? Band-pass the averaged record above 7 THz and look at where the
     energy sits. Localised on the main pulse => it is the pulse's own (real or aliased)
     high-frequency shoulder. Spread over the record => a broadband reproducible artefact.
     Concentrated in one or two samples => an acquisition glitch.

  2. IS THE EMPIRICAL ERROR BAR STABLE? Repeat the block estimate of sigma_|H| for several
     block counts. If the answer barely moves, the "pipeline over-estimates by Nx" numbers
     from part 2 are trustworthy.
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
    "mask": {"snr_thresh_db": 10, "tail_fraction": 0.25, "min_contiguous_bins": 3},
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


def clipped_window(n_samples, peak_index, half_width_samples):
    window_length = 2 * half_width_samples + 1
    core = thz._build_symmetric_window(window_length, 'hann', 1.0)
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
    edges = np.linspace(0, stack.shape[0], n_blocks + 1).astype(int)
    return np.asarray([stack[edges[i]:edges[i + 1]].mean(axis=0) for i in range(n_blocks)])


emit("=" * 96)
emit("1. TIME LOCALISATION OF THE >7 THz CONTENT (band-pass of the averaged windowed record)")
emit("=" * 96)

cache = {}
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    windowed = processing['time_domain_prefft']
    time_ps = windowed[:, 0] * 1e12
    n_samples = windowed.shape[0]
    dt_s = float(np.median(np.diff(windowed[:, 0])))
    fixed_window = processing['fixed_window']
    window_function = clipped_window(n_samples, int(fixed_window['peak_index']),
                                     int(fixed_window['half_width_samples']))
    stack = padded_scans(data_obj)
    averaged_windowed = stack.mean(axis=0) * window_function
    cache[filename] = dict(stack=stack, window_function=window_function, dt_s=dt_s,
                           time_ps=time_ps, averaged_windowed=averaged_windowed)

    spectrum = np.fft.rfft(averaged_windowed, n=n_samples)
    freq = np.fft.rfftfreq(n_samples, dt_s) * _HZ_TO_THZ
    high_pass = spectrum.copy()
    high_pass[freq < 7.0] = 0.0
    high_band_trace = np.fft.irfft(high_pass, n=n_samples)

    # per-scan version of the same band, to separate reproducible from random
    scan_spectra = np.fft.rfft(stack * window_function, n=n_samples, axis=1)
    scan_high = scan_spectra.copy()
    scan_high[:, freq < 7.0] = 0.0
    scan_high_traces = np.fft.irfft(scan_high, n=n_samples, axis=1)
    per_scan_rms = scan_high_traces.std(axis=0, ddof=1)

    peak_index = int(np.argmax(np.abs(averaged_windowed)))
    order = np.argsort(np.abs(high_band_trace))[::-1][:6]
    emit()
    emit(f"--- {filename}")
    emit(f"    main pulse peak at index {peak_index} ({time_ps[peak_index]:.2f} ps), "
         f"amplitude {averaged_windowed[peak_index]:.3e}")
    emit(f"    >7 THz band: rms over record = {np.std(high_band_trace):.3e}, "
         f"max = {np.max(np.abs(high_band_trace)):.3e}")
    emit(f"    6 largest >7 THz samples (index, t/ps, value, dist. from peak in samples, "
         f"per-scan rms there):")
    for index in order:
        emit(f"        {index:4d}  {time_ps[index]:8.2f}  {high_band_trace[index]:+.3e}  "
             f"{index - peak_index:+4d}   {per_scan_rms[index]:.3e}")
    energy_near_peak = np.sum(high_band_trace[max(0, peak_index - 5):peak_index + 6] ** 2)
    emit(f"    fraction of >7 THz energy within +/-5 samples of the peak: "
         f"{energy_near_peak / np.sum(high_band_trace ** 2):.3f}")
    emit(f"    reproducible fraction (mean band amplitude / per-scan band rms) at the peak: "
         f"{abs(high_band_trace[peak_index]) / per_scan_rms[peak_index]:.2f}")

emit()
emit("=" * 96)
emit("2. STABILITY OF THE EMPIRICAL sigma_|H| vs NUMBER OF BLOCKS")
emit("=" * 96)

DIAGNOSTIC_FREQS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
for filename, data_obj in dataset.data.items():
    transfer_H = data_obj.processing_dict.get('transfer_H')
    if transfer_H is None:
        continue
    reference_name = getattr(dataset.get_reference(filename, ref_type='reference'), 'filename', None)
    sample_cache = cache[filename]
    reference_cache = cache[reference_name]
    freq_thz = np.asarray(data_obj.processing_dict['fft_freq']) * _HZ_TO_THZ
    sample_spectra = np.fft.rfft(sample_cache['stack'] * sample_cache['window_function'],
                                 n=N_FFT, axis=1)
    reference_spectra = np.fft.rfft(reference_cache['stack'] * reference_cache['window_function'],
                                    n=N_FFT, axis=1)
    pipeline_sigma = np.asarray(data_obj.processing_dict['transfer_H_sigma'])

    emit()
    emit(f"--- {filename}")
    header = f"    {'f (THz)':>8} {'pipeline':>11}" + "".join(
        f"{f'{n} blocks':>11}" for n in (4, 6, 8, 10))
    emit(header)
    for target in DIAGNOSTIC_FREQS:
        index = int(np.argmin(np.abs(freq_thz - target)))
        row = f"    {freq_thz[index]:8.2f} {pipeline_sigma[index]:11.4e}"
        for n_blocks in (4, 6, 8, 10):
            transfer_blocks = (block_means(sample_spectra, n_blocks)
                               / block_means(reference_spectra, n_blocks))
            sigma = np.abs(transfer_blocks).std(axis=0, ddof=1) / np.sqrt(n_blocks)
            row += f"{sigma[index]:11.4e}"
        emit(row)

with open(os.path.join(OUT_DIR, 'audit_report_part3.txt'), 'w') as handle:
    handle.write("\n".join(report_lines))

figure, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained')
for filename, cached in cache.items():
    axes[0].plot(cached['time_ps'], cached['averaged_windowed'], lw=1.1, label=filename)
axes[0].set_ylabel('windowed amplitude')
axes[0].set_title('Averaged windowed record')
axes[0].legend(fontsize=7)
for filename, cached in cache.items():
    windowed = cached['averaged_windowed']
    n_samples = windowed.size
    freq = np.fft.rfftfreq(n_samples, cached['dt_s']) * _HZ_TO_THZ
    spectrum = np.fft.rfft(windowed, n=n_samples)
    spectrum[freq < 7.0] = 0.0
    axes[1].plot(cached['time_ps'], np.fft.irfft(spectrum, n=n_samples), lw=1.1, label=filename)
axes[1].set_xlabel('time (ps)')
axes[1].set_ylabel('>7 THz component')
axes[1].set_title('Where the "noise floor" band actually lives in time')
axes[1].legend(fontsize=7)
figure.savefig(os.path.join(OUT_DIR, 'high_frequency_origin.png'), dpi=130)
print(f"\nWrote {os.path.join(OUT_DIR, 'audit_report_part3.txt')}")
