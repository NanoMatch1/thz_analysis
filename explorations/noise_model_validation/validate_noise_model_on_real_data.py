"""Validate thz_core.noise against real .acc acquisitions.

The synthetic known-answer tests in thz-core prove the estimators recover what they are
given. This script checks the two claims that only real data can test:

  1. The measured noise agrees with the empirical scatter established by the earlier
     audit (explorations/transfer_uncertainty_audit): a flat additive floor around
     1e-4 and a large drift component in the mid-band.

  2. THE DECISIVE ONE. The tail-median "noise floor" does not respond to a genuine
     change in instrument noise, because it is set by signal. The repeat-based
     estimate must. Tested two ways:
       (a) sub-sampling the scans of one acquisition — averaging fewer scans IS more
           noise, so a real estimator must rise as 1/sqrt(M);
       (b) comparing two acquisitions taken on different days.

Reads the raw per-scan matrix straight from the .acc files, so nothing here depends on
the analysis pipeline's own processing choices.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_core.io.loaders.acc_loader import ACCLoader
from thz_core.thz_core.noise import (
    block_spectral_scatter,
    drift_corrected_scatter,
    fit_noise_parameters,
    spectral_noise_moments,
)

_HZ_TO_THZ = 1e-12
DATA_ROOT = "/home/match/data/CNTs"

FILES = {
    "08-20 reference (bare)":
        f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/reference_CNT_undoped-19mm.acc",
    "08-20 sample (doped 12mm)":
        f"{DATA_ROOT}/2026-08-20_CNT-paper-doped_2/export/sample_CNT_doped-12mm.acc",
    "08-21 reference (bare)":
        f"{DATA_ROOT}/2026-08-21_CNT-paper-doped_3/comparison/reference_CNT-2_bare-5mm.acc",
    "08-21 sample (doped 15mm)":
        f"{DATA_ROOT}/2026-08-21_CNT-paper-doped_3/comparison/sample_CNT-2_doped-15mm.acc",
}


def load_scan_matrix(filepath: str) -> tuple:
    """(time_seconds, waveforms[n_scans, n_samples]) from an .acc file."""
    thz_object = ACCLoader(filepath).load()
    # THzData.raw_data is the compiled per-scan matrix: column 0 is time (in the
    # source's picoseconds), columns 1.. are the individual acquisitions.
    raw = np.asarray(thz_object.raw_data, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(
            f"{filepath}: raw_data has shape {raw.shape}; need (n_samples, 1 + >=2 scans)."
        )
    time_seconds = raw[:, 0] * 1e-12
    waveforms = raw[:, 1:].T                      # (n_scans, n_samples)
    return time_seconds, waveforms


def tail_median_floor(spectrum: np.ndarray, tail_fraction: float = 0.25) -> float:
    """The incumbent estimator: median |Y| over the top of the frequency axis."""
    magnitude = np.abs(spectrum)
    tail_points = max(8, int(np.ceil(magnitude.size * tail_fraction)))
    return float(np.median(magnitude[-tail_points:]))


print("=" * 92)
print("1. NOISE MODEL FIT ON REAL ACQUISITIONS")
print("=" * 92)

summaries = {}
for label, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f"\n--- {label}: FILE NOT FOUND ({filepath}); skipping")
        continue
    time_seconds, waveforms = load_scan_matrix(filepath)
    dt = float(np.median(np.diff(time_seconds)))
    n_scans, n_samples = waveforms.shape

    drift = drift_corrected_scatter(waveforms, dt)
    parameters, _ = fit_noise_parameters(waveforms, dt, drift=drift)
    fractions = parameters.term_contributions(drift.mean_waveform, dt)

    peak = float(np.max(np.abs(drift.mean_waveform)))
    delay_walk = float(np.ptp(drift.delays))
    amplitude_walk = float(np.ptp(drift.amplitudes))

    summaries[label] = dict(time=time_seconds, waveforms=waveforms, dt=dt,
                            drift=drift, parameters=parameters)

    print(f"\n--- {label}")
    print(f"    {n_scans} scans x {n_samples} samples, dt = {dt * 1e15:.1f} fs, "
          f"peak = {peak:.4e}")
    print(f"    sigma_alpha = {parameters.sigma_alpha:.4e}          "
          f"({parameters.sigma_alpha / peak * 100:.3f}% of peak)   [detection electronics]")
    print(f"    sigma_beta  = {parameters.sigma_beta * 100:.3f}%                  "
          f"                       [laser power]")
    print(f"    sigma_tau   = {parameters.sigma_tau * 1e15:.2f} fs                "
          f"                       [delay line jitter]")
    print(f"    dominant term: additive {fractions['additive']:.1%}, "
          f"multiplicative {fractions['multiplicative']:.1%}, "
          f"jitter {fractions['jitter']:.1%}")
    print(f"    fit: method={parameters.method}, profile_error="
          f"{parameters.profile_error:.3f}, trusted={parameters.converged} "
          f"(expected profile noise from M={n_scans}: "
          f"{1/np.sqrt(2*(n_scans-1)):.2f})")
    print(f"    drift over the acquisition: amplitude {amplitude_walk * 100:.2f}% "
          f"peak-to-peak, delay {delay_walk * 1e15:.1f} fs peak-to-peak")
    print(f"    drift inflation of raw repeat scatter: {drift.drift_inflation:.2f}x")

print()
print("=" * 92)
print("2. DOES THE ESTIMATE RESPOND TO AVERAGING?  (sub-sampling one acquisition)")
print("=" * 92)
print("   Fewer scans averaged = more noise. A real uncertainty estimate must rise as")
print("   1/sqrt(M); an estimator measuring signal will not move.")

for label in list(summaries)[:2]:
    entry = summaries[label]
    waveforms, dt = entry["waveforms"], entry["dt"]
    n_total = waveforms.shape[0]
    window = np.hanning(waveforms.shape[1])
    n_fft = waveforms.shape[1]

    print(f"\n--- {label}  ({n_total} scans)")
    print("    An estimator measuring NOISE gives sigma*sqrt(M) = constant (the")
    print("    per-scan noise level). One measuring SIGNAL keeps floor*sqrt(M) rising.")
    print(f"    {'M used':>7} {'tail floor':>13} {'repeat sigma':>14} "
          f"{'floor*sqrtM':>13} {'sigma*sqrtM':>13}")
    # M < 8 is below the useful floor: the variance estimate itself carries
    # ~1/sqrt(2(M-1)) ~ 40% error there, and the drift fit has almost nothing to
    # work with. Included for completeness, excluded from the spread.
    counts = sorted({4, 8, 16, 32, n_total} & set(range(2, n_total + 1)))
    floor_invariants, sigma_invariants = [], []
    for n_used in counts:
        subset = waveforms[:n_used]
        mean_spectrum = np.fft.rfft(subset.mean(axis=0) * window, n=n_fft)
        floor = tail_median_floor(mean_spectrum)
        estimate = drift_corrected_scatter(subset, dt)
        moments = spectral_noise_moments(estimate.sigma_t, window=window, n_fft=n_fft,
                                         spectrum=mean_spectrum, n_averaged=n_used)
        sigma = float(np.median(moments.sigma_magnitude))
        root = np.sqrt(n_used)
        floor_invariants.append(floor * root)
        sigma_invariants.append(sigma * root)
        flag = "  (M<8: unreliable)" if n_used < 8 else ""
        print(f"    {n_used:7d} {floor:13.4e} {sigma:14.4e} "
              f"{floor * root:13.4e} {sigma * root:13.4e}{flag}")

    def spread(values):
        values = np.asarray(values)
        return float(np.max(values) / np.min(values))

    usable = [index for index, count in enumerate(counts) if count >= 8]
    print(f"    -> spread of the invariant over M>=8 "
          f"(1.00 = perfect 1/sqrt(M) scaling):")
    print(f"         tail floor  : {spread([floor_invariants[i] for i in usable]):.2f}x")
    print(f"         repeat sigma: {spread([sigma_invariants[i] for i in usable]):.2f}x")

print()
print("=" * 92)
print("3. DOES THE ESTIMATE RESPOND ACROSS ACQUISITIONS?")
print("=" * 92)
print("   Normalised to each file's own spectral peak (so different signal levels")
print("   do not confound it) and to PER-SCAN noise, sigma*sqrt(M) (so different")
print("   scan counts do not either). Per-scan noise is the instrument property.")
print(f"\n    {'acquisition':<28} {'scans':>6} {'floor/peak':>12} "
      f"{'sigma/peak':>12} {'per-scan sigma/peak':>20} {'sigma_beta':>11}")
for label, entry in summaries.items():
    waveforms, dt, drift = entry["waveforms"], entry["dt"], entry["drift"]
    window = np.hanning(waveforms.shape[1])
    n_fft = waveforms.shape[1]
    mean_spectrum = np.fft.rfft(drift.mean_waveform * window, n=n_fft)
    peak = float(np.max(np.abs(drift.mean_waveform)))
    moments = spectral_noise_moments(drift.sigma_t, window=window, n_fft=n_fft,
                                     spectrum=mean_spectrum,
                                     n_averaged=waveforms.shape[0])
    floor = tail_median_floor(mean_spectrum)
    sigma = float(np.median(moments.sigma_magnitude))
    spectral_peak = float(np.max(np.abs(mean_spectrum)))
    per_scan = sigma * np.sqrt(waveforms.shape[0])
    print(f"    {label:<28} {waveforms.shape[0]:6d} {floor / spectral_peak:12.3e} "
          f"{sigma / spectral_peak:12.3e} {per_scan / spectral_peak:20.3e} "
          f"{entry['parameters'].sigma_beta * 100:10.3f}%")

print()
print("=" * 92)
print("4. WITHIN-SCAN NOISE vs DRIFT (which limits this measurement?)")
print("=" * 92)

for label, entry in summaries.items():
    waveforms, dt, drift = entry["waveforms"], entry["dt"], entry["drift"]
    window = np.hanning(waveforms.shape[1])
    n_fft = waveforms.shape[1]
    n_scans = waveforms.shape[0]
    spectra = np.fft.rfft(waveforms * window, n=n_fft, axis=1)
    mean_spectrum = spectra.mean(axis=0)
    frequency_thz = np.fft.rfftfreq(n_fft, dt) * _HZ_TO_THZ

    moments = spectral_noise_moments(drift.sigma_t, window=window, n_fft=n_fft,
                                     spectrum=mean_spectrum, n_averaged=n_scans)
    n_blocks = min(8, max(2, n_scans // 4))
    blocked = block_spectral_scatter(spectra, n_blocks=n_blocks)
    floor = tail_median_floor(mean_spectrum)

    print(f"\n--- {label}   ({n_scans} scans, {n_blocks} blocks)")
    print(f"    {'f (THz)':>8} {'|Y|':>11} {'within-scan':>12} {'incl. drift':>12} "
          f"{'tail floor':>12} {'floor/within':>13} {'drift/within':>13}")
    for target in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        index = int(np.argmin(np.abs(frequency_thz - target)))
        if index >= mean_spectrum.size:
            continue
        within = moments.sigma_magnitude[index]
        with_drift = blocked[index]
        print(f"    {frequency_thz[index]:8.2f} {np.abs(mean_spectrum[index]):11.3e} "
              f"{within:12.3e} {with_drift:12.3e} {floor:12.3e} "
              f"{floor / max(within, 1e-30):13.1f} {with_drift / max(within, 1e-30):13.1f}")
