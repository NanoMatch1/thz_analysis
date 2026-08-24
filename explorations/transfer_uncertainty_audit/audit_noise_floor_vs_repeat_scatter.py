"""Audit: is the transfer-function error bar at HIGH frequency real, or an artifact?

Runs the run_me_reflection_single pipeline headlessly on the currently-loaded dataset, then
compares INDEPENDENT estimates of the spectral uncertainty of each averaged spectrum:

  A. the pipeline's estimate -- flat scalar noise floor = median|Y| over the top
     `tail_fraction` of the frequency axis (thz_core.transfer._noise_floor), turned into
     snr_db and then into transfer_H_sigma by compute_transfer_uncertainty.

  B. the EMPIRICAL repeat scatter -- push every individual .acc scan through the SAME
     linear chain (pad + taper + window + rfft) the pipeline used on the average, then take
     the standard error of the mean across scans, per frequency bin. This is the honest,
     measured uncertainty of the averaged spectrum, whatever its colour.

  C. the split-half test -- the random part of the averaged spectrum is
     (mean_A - mean_B)/2; any part of the spectral "floor" that is deterministic
     (window leakage, a fixed artefact) survives in (mean_A + mean_B)/2 but cancels in
     the difference. Distinguishes "the tail is noise" from "the tail is leakage".

The linear chain is verified against the pipeline's own stored windowed trace before any
conclusion is drawn.
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
_S_TO_PS = 1e12
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = r'C:\Users\Samuel\Data\THz\CNTs\2026-08-20_CNT-paper-doped_2\export'

TAIL_FRACTION = 0.25
N_FFT = 2000

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
thz.window_single_pulse_fixed_width(dataset, half_width_ps=config['window']['half_width_ps'],
                                    region_ps=None, show_graph=False)
thz.fft_spectrum(dataset, n_fft=N_FFT)
thz.transfer_function(dataset, ref_type='reference')
thz.compute_instrument_resolution(dataset, config)
thz.compute_transfer_uncertainty(dataset)


# ── reconstruct the pipeline's own (linear) pad + taper + window chain, per scan ──

def rebuild_pad_and_taper(amplitude: np.ndarray, centering_info: dict) -> np.ndarray:
    """Apply the SAME zero-pad + half-cosine junction taper _center_pulse_trace applied."""
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


def rebuild_window_function(n_samples: int, fixed_window: dict, window_type: str,
                            alpha: float) -> np.ndarray:
    """Rebuild the clipped fixed-width window the pipeline dropped on the peak."""
    peak_index = int(fixed_window['peak_index'])
    half_width_samples = int(fixed_window['half_width_samples'])
    window_length = int(fixed_window['window_length'])
    core = thz._build_symmetric_window(window_length, window_type, alpha)
    lo = peak_index - half_width_samples
    hi = peak_index + half_width_samples + 1
    data_lo, data_hi = max(lo, 0), min(hi, n_samples)
    core_lo = data_lo - lo
    core_hi = core_lo + (data_hi - data_lo)
    window_function = np.zeros(n_samples)
    window_function[data_lo:data_hi] = core[core_lo:core_hi]
    return window_function


def scan_spectra(data_obj) -> tuple:
    """(freq_hz, per-scan complex spectra [n_scans, n_bins], window_function)."""
    raw = np.asarray(data_obj.raw_data, dtype=float)          # col0 = time (ps)
    centering_info = data_obj.processing_dict['centering_info']
    fixed_window = data_obj.processing_dict['fixed_window']
    windowed_trace = data_obj.processing_dict['time_domain_prefft']
    n_samples = windowed_trace.shape[0]
    window_function = rebuild_window_function(
        n_samples, fixed_window, config['window']['type'], config['window']['alpha'])
    dt_s = float(np.median(np.diff(windowed_trace[:, 0])))
    baseline_n = 10

    spectra = []
    for column in range(1, raw.shape[1]):
        padded = rebuild_pad_and_taper(raw[:, column], centering_info)
        padded = padded - np.mean(padded[:baseline_n])       # pipeline's DC baseline step
        spectra.append(np.fft.rfft(padded * window_function, n=N_FFT, norm='backward'))
    freq = np.fft.rfftfreq(N_FFT, dt_s)
    return freq, np.asarray(spectra), window_function


def noise_floor(magnitude: np.ndarray, tail_fraction: float) -> float:
    """thz_core.transfer._noise_floor, reproduced so the audit is self-contained."""
    tail_points = max(8, int(np.ceil(magnitude.size * tail_fraction)))
    return max(float(np.median(np.abs(magnitude[-tail_points:]))), 1e-30)


report_lines = []


def emit(text=""):
    print(text)
    report_lines.append(str(text))


emit("=" * 90)
emit("PER-FILE: pipeline flat floor  vs  measured repeat scatter  vs  split-half random part")
emit("=" * 90)

results = {}
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    freq = np.asarray(processing['fft_freq'])
    pipeline_spectrum = np.asarray(processing['fft_spectrum'])
    freq_thz = freq * _HZ_TO_THZ

    freq_check, spectra, window_function = scan_spectra(data_obj)
    n_scans = spectra.shape[0]
    reconstructed_mean = spectra.mean(axis=0)

    # Validate the reconstruction against what the pipeline actually transformed.
    relative_mismatch = (np.max(np.abs(reconstructed_mean - pipeline_spectrum))
                         / np.max(np.abs(pipeline_spectrum)))

    # A. pipeline flat floor
    flat_floor = noise_floor(np.abs(pipeline_spectrum), TAIL_FRACTION)

    # B. empirical standard error of the mean, per bin (complex -> per-component)
    standard_error_complex = (np.std(spectra.real, axis=0, ddof=1)
                              + 1j * np.std(spectra.imag, axis=0, ddof=1)) / np.sqrt(n_scans)
    standard_error_radial = np.sqrt(0.5 * (standard_error_complex.real ** 2
                                           + standard_error_complex.imag ** 2)) * np.sqrt(2)
    # (radial 1-sigma for an isotropic complex error = rms of the two components)

    # C. split-half random part
    half = n_scans // 2
    mean_a = spectra[:half].mean(axis=0)
    mean_b = spectra[half:2 * half].mean(axis=0)
    split_half_random = np.abs(mean_a - mean_b) / 2.0
    split_half_total = np.abs(mean_a + mean_b) / 2.0

    tail_points = max(8, int(np.ceil(freq.size * TAIL_FRACTION)))
    tail = slice(freq.size - tail_points, freq.size)

    results[filename] = dict(freq_thz=freq_thz, pipeline_spectrum=pipeline_spectrum,
                             flat_floor=flat_floor, standard_error=standard_error_radial,
                             split_half_random=split_half_random, n_scans=n_scans,
                             snr_db=np.asarray(processing['snr_db']))

    emit()
    emit(f"--- {filename}   ({n_scans} scans)")
    emit(f"    reconstruction check (max |rebuilt - pipeline| / max|Y|): {relative_mismatch:.3e}")
    emit(f"    tail band used for the floor: {freq_thz[tail][0]:.2f} - {freq_thz[-1]:.2f} THz")
    emit(f"    A. pipeline flat floor                    = {flat_floor:.4e}")
    emit(f"    B. median measured SE(mean) in tail       = {np.median(standard_error_radial[tail]):.4e}")
    emit(f"    C. median split-half random part in tail  = {np.median(split_half_random[tail]):.4e}")
    emit(f"       -> flat floor / measured tail noise    = "
         f"{flat_floor / np.median(standard_error_radial[tail]):.2f} x")
    emit(f"    peak |Y| = {np.max(np.abs(pipeline_spectrum)):.4e}")
    emit()
    emit(f"    {'f (THz)':>9} {'|Y|':>11} {'flatfloor':>11} {'measSE':>11} {'splitrand':>11} "
         f"{'snr_db(pipe)':>13} {'snr_db(meas)':>13}")
    for target_thz in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0):
        index = int(np.argmin(np.abs(freq_thz - target_thz)))
        magnitude = np.abs(pipeline_spectrum[index])
        measured = standard_error_radial[index]
        emit(f"    {freq_thz[index]:9.2f} {magnitude:11.3e} {flat_floor:11.3e} "
             f"{measured:11.3e} {split_half_random[index]:11.3e} "
             f"{20 * np.log10(max(magnitude, flat_floor) / flat_floor):13.1f} "
             f"{20 * np.log10(magnitude / max(measured, 1e-30)):13.1f}")

# ── the transfer function: pipeline sigma vs an empirically-propagated sigma ─────
emit()
emit("=" * 90)
emit("TRANSFER FUNCTION uncertainty: pipeline vs measured")
emit("=" * 90)
for filename, data_obj in dataset.data.items():
    processing = data_obj.processing_dict
    transfer_H = processing.get('transfer_H')
    if transfer_H is None:
        continue
    reference_obj = dataset.get_reference(filename, ref_type='reference')
    reference_name = getattr(reference_obj, 'filename', None)
    sample_result = results[filename]
    reference_result = results[reference_name]
    freq_thz = sample_result['freq_thz']

    pipeline_sigma = np.asarray(processing['transfer_H_sigma'])
    pipeline_relative = pipeline_sigma / np.abs(transfer_H)

    measured_relative = np.sqrt(
        (sample_result['standard_error'] / np.abs(sample_result['pipeline_spectrum'])) ** 2
        + (reference_result['standard_error'] / np.abs(reference_result['pipeline_spectrum'])) ** 2
    )

    emit()
    emit(f"--- {filename}   (reference: {reference_name})")
    emit(f"    {'f (THz)':>9} {'|H|':>10} {'rel.sigma pipe':>15} {'rel.sigma meas':>15} {'ratio':>8}")
    for target_thz in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        index = int(np.argmin(np.abs(freq_thz - target_thz)))
        ratio = pipeline_relative[index] / max(measured_relative[index], 1e-30)
        emit(f"    {freq_thz[index]:9.2f} {np.abs(transfer_H[index]):10.3f} "
             f"{pipeline_relative[index]:15.4f} {measured_relative[index]:15.4f} {ratio:8.1f}")

with open(os.path.join(OUT_DIR, 'audit_report.txt'), 'w') as handle:
    handle.write("\n".join(report_lines))

# ── figure ──────────────────────────────────────────────────────────────────────
figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, layout='constrained')
for filename, result in results.items():
    line, = axes[0].semilogy(result['freq_thz'], np.abs(result['pipeline_spectrum']),
                             lw=1.2, label=f"|Y| {filename}")
    axes[0].axhline(result['flat_floor'], color=line.get_color(), ls='--', lw=0.9,
                    label=f"flat floor {filename}")
    axes[0].semilogy(result['freq_thz'], result['standard_error'], color=line.get_color(),
                     ls=':', lw=1.4, label=f"measured SE {filename}")
axes[0].set_ylabel('|Y|, noise estimates')
axes[0].set_title('Spectrum vs pipeline flat floor (dashed) vs measured repeat SE (dotted)')
axes[0].legend(fontsize=6, ncol=3)
axes[0].set_ylim(1e-8, None)

for filename, result in results.items():
    axes[1].semilogy(result['freq_thz'],
                     result['flat_floor'] / np.maximum(result['standard_error'], 1e-30),
                     lw=1.2, label=filename)
axes[1].axhline(1.0, color='k', lw=0.8)
axes[1].set_xlabel('Frequency (THz)')
axes[1].set_ylabel('flat floor / measured SE')
axes[1].set_title('Overestimation factor of the pipeline noise model')
axes[1].legend(fontsize=7)
figure.savefig(os.path.join(OUT_DIR, 'noise_floor_audit.png'), dpi=130)
print(f"\nWrote {os.path.join(OUT_DIR, 'noise_floor_audit.png')}")
