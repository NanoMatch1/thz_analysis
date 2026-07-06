"""Which W = Y2/Y1 carries the spurious +17 ps: the Si-p sample or its reference?

Reruns the Si-p channel pipeline, then for sample and reference separately:
  - where the windowed pulses sit on each segment's array (time_domain_prefft);
  - each segment's t[0] and peak time;
  - the phase slope of W = Y2/Y1 (should be ~ -24.85 ps for both; their difference is arg(H)).

Run from repo root:
  PYTHONPATH=. ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/debug_si_p_segments.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from process_cnt21_channels import CHANNELS, process_channel

BAND_HZ = (0.4e12, 2.2e12)


def segment_report(holder, label):
    """Print timing bookkeeping for one segment holder; return its spectrum + freq."""
    processing = holder.processing_dict
    freq = np.asarray(processing["fft_freq"], float)
    spectrum = np.asarray(processing["fft_spectrum"], complex)
    prefft = np.asarray(processing["time_domain_prefft"], float)
    time_s = prefft[:, 0]
    y = prefft[:, 1]
    peak_index = int(np.argmax(np.abs(y)))
    print(f"    {label:18s}: t0 = {time_s[0]*1e12:9.2f} ps, n = {len(y):4d}, "
          f"peak at {time_s[peak_index]*1e12:9.2f} ps (index {peak_index}), "
          f"peak value {y[peak_index]:+.4f}")
    return freq, spectrum


def w_phase_slope(freq, spectrum_second, spectrum_first):
    W = spectrum_second / spectrum_first
    band = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1]) & np.isfinite(W)
    phase = np.unwrap(np.angle(W[band]))
    slope, intercept = np.polyfit(freq[band], phase, 1)
    return slope / (2 * np.pi) * 1e12, intercept, W, band   # slope in ps


def main():
    for channel in ("Si-p", "90-0"):
        print(f"\n===== {channel} =====")
        dataset, sample_name = process_channel(channel, CHANNELS[channel])
        for filename, data_obj in dataset.data.items():
            role = "REFERENCE" if dataset.data.is_reference(filename) else "SAMPLE"
            print(f"  {role}: {os.path.basename(filename)}")
            freq1, Y1 = segment_report(data_obj.first_reflection, "first_reflection")
            freq2, Y2 = segment_report(data_obj.second_reflection, "second_reflection")
            slope_ps, intercept, W, band = w_phase_slope(freq2, Y2, Y1)
            print(f"    W = Y2/Y1 phase slope = {slope_ps:+8.3f} ps, "
                  f"intercept {intercept:+7.2f} rad, |W| med "
                  f"{np.nanmedian(np.abs(W[band])):.3f}")


if __name__ == "__main__":
    main()
