"""Can a purge-equilibration check work without resolving the water lines?

Samuel's proposition: even at 0.15 THz resolution, where the rotational lines are
unresolved, the water effect is visible as a broadband intensity change — and
possibly as a frequency-dependent one, rising toward 3-7 THz.

Two things to separate, because they matter differently:

  * GAIN     — does the whole spectrum rise as the purge settles? Real, but degenerate:
               a laser power drift does exactly the same thing.
  * TILT     — does the rise depend on frequency? Water's continuum absorption grows
               with frequency, so removing water should lift the high band MORE than
               the low band. A laser drift scales everything uniformly and leaves the
               tilt flat. So tilt is the water-SPECIFIC observable, and the one that
               breaks the degeneracy.

Measured per scan against wall-clock time, on the acquisition with the strongest
known purge transient.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_ROOT = os.environ.get("THZ_DATA_ROOT", "/home/match/data")
FILES = {
    "doped 12mm (111 scans, 83 min)":
        f"{DATA_ROOT}/CNTs/2026-08-20_CNT-paper-doped_2/export/sample_CNT_doped-12mm.acc",
    "bare reference (44 scans)":
        f"{DATA_ROOT}/CNTs/2026-08-20_CNT-paper-doped_2/export/reference_CNT_undoped-19mm.acc",
}

LOW_BAND = (0.4, 1.0)      # strong signal, little water continuum
BANDS = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]


def load(filepath):
    with open(filepath) as handle:
        text = handle.read()
    stamps, waveforms, time_axis = [], [], None
    for chunk in text.split("%%"):
        stamp = re.search(r"Date and time,([0-9\-: .]+)", chunk)
        rows = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if len(rows) < 8:
            continue
        array = np.asarray(rows)
        if time_axis is None:
            time_axis = array[:, 0] * 1e-12
        if array.shape[0] != time_axis.size:
            continue
        waveforms.append(array[:, 1])
        stamps.append(datetime.fromisoformat(stamp.group(1).strip()) if stamp else None)
    waveforms = np.asarray(waveforms)
    minutes = (np.asarray([(s - stamps[0]).total_seconds() / 60.0 for s in stamps])
               if all(s is not None for s in stamps)
               else np.arange(waveforms.shape[0], dtype=float))
    return time_axis, waveforms, minutes


def band_amplitude(spectra, frequency_thz, band):
    inside = (frequency_thz >= band[0]) & (frequency_thz < band[1])
    return np.abs(spectra[:, inside]).mean(axis=1)


def trend(values, minutes):
    """Fractional change from first to last, and the straight-line slope per hour."""
    first = np.mean(values[:max(3, len(values) // 10)])
    last = np.mean(values[-max(3, len(values) // 10):])
    slope = np.polyfit(minutes, values / np.mean(values), 1)[0] * 60.0
    return (last / first - 1.0), slope


for label, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f"\n### {label}: not found")
        continue
    time_seconds, waveforms, minutes = load(filepath)
    dt = float(np.median(np.diff(time_seconds)))
    window = np.hanning(waveforms.shape[1])
    spectra = np.fft.rfft(waveforms * window, n=waveforms.shape[1], axis=1)
    frequency_thz = np.fft.rfftfreq(waveforms.shape[1], dt) * 1e-12

    low = band_amplitude(spectra, frequency_thz, LOW_BAND)
    low_change, low_slope = trend(low, minutes)

    # Per-scan SNR proxy: how far each band sits above the top-of-axis level.
    floor = np.abs(spectra[:, frequency_thz > 0.75 * frequency_thz.max()]).mean()

    print(f"\n{'=' * 88}")
    print(f"### {label}   ({waveforms.shape[0]} scans over {minutes[-1]:.0f} min)")
    print("=" * 88)
    print(f"  GAIN  low band {LOW_BAND[0]}-{LOW_BAND[1]} THz: "
          f"{low_change:+.2%} first-to-last, slope {low_slope:+.3f}/hour")
    print()
    print(f"  {'band (THz)':>12} {'gain':>9} {'band/low tilt':>15} "
          f"{'tilt slope/hr':>14} {'band SNR':>10}")
    for band in BANDS:
        values = band_amplitude(spectra, frequency_thz, band)
        gain, _ = trend(values, minutes)
        tilt = values / low
        tilt_change, tilt_slope = trend(tilt, minutes)
        snr = values.mean() / floor
        flag = ""
        if snr < 3:
            flag = "  (SNR too low to trust)"
        elif abs(tilt_slope) > 3 * abs(low_slope) and abs(tilt_change) > 0.01:
            flag = "  <-- water-like: rises FASTER than the low band"
        print(f"  {band[0]:5.1f}-{band[1]:<5.1f} {gain:+8.2%} {tilt_change:+14.2%} "
              f"{tilt_slope:+13.3f} {snr:10.1f}{flag}")

    print()
    print("  Interpretation: a uniform GAIN with a flat TILT is degenerate with laser")
    print("  power drift. A tilt that grows with frequency is water-specific.")
