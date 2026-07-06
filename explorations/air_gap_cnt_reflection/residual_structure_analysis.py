"""Identify the ~10% reproducible residual structure in the CNT-21 reflections.

Probe 1 — echo/delay analysis: subtract a smooth (low-order polynomial) fit from H(omega) in
band, FFT the complex residual onto a delay axis. Peaks at characteristic delays finger the
mechanism (GaP detection echo ~5.3 ps, window-edge truncation ~2 ps = the Hann half-width,
window round trip ~24.85 ps, etc.).

Probe 2 — reference cross-ratios: W = Y2/Y1 of each bare-window reference is a property of
the window + mount slot + detection-cone coupling. Ratios between slots at fixed pol
(W_SP/W_SS, W_PS/W_PP) and between pols quantify the slot-to-slot coupling structure that
per-channel referencing is meant to absorb — and that the sample mount can break again.
NOTE the realign repeat SHARED its reference, so reference-side systematics are invisible in
the 0.5% repeatability number.

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/residual_structure_analysis.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from process_cnt21_channels import CHANNELS, process_channel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
BAND_HZ = (0.4e12, 2.4e12)
POLY_DEGREE = 5

CHANNEL_NAMES = ["0-0", "0-90", "90-0", "90-90"]


def smooth_residual(freq, values, degree=POLY_DEGREE):
    """Complex residual after removing a low-order polynomial (fit on real+imag)."""
    x = (freq - freq.mean()) / (freq.max() - freq.min())
    real_fit = np.polyval(np.polyfit(x, values.real, degree), x)
    imag_fit = np.polyval(np.polyfit(x, values.imag, degree), x)
    return values - (real_fit + 1j * imag_fit)


def delay_spectrum(freq, residual):
    """|FFT| of the complex in-band residual onto a delay axis (ps)."""
    df = float(np.median(np.diff(freq)))
    window = np.hanning(len(residual))
    spectrum = np.fft.fft(residual * window, n=8 * len(residual))
    delay_ps = np.fft.fftfreq(spectrum.size, d=df) * 1e12
    order = np.argsort(delay_ps)
    return delay_ps[order], np.abs(spectrum[order]) / len(residual)


def main():
    # ---- run the four channels once, collect sample H and reference W ----
    sample_H = {}
    reference_W = {}
    for name in CHANNEL_NAMES:
        dataset, sample_name = process_channel(name, CHANNELS[name])
        for filename, data_obj in dataset.data.items():
            first = data_obj.first_reflection.processing_dict
            second = data_obj.second_reflection.processing_dict
            freq = np.asarray(second["fft_freq"], float)
            W = (np.asarray(second["fft_spectrum"], complex)
                 / np.asarray(first["fft_spectrum"], complex))
            if dataset.data.is_reference(filename):
                reference_W[name] = (freq, W)
            else:
                H = np.asarray(second["transfer_H"], complex)
                mask = np.asarray(second["transfer_mask"], bool)
                sample_H[name] = (freq, H, mask)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    figure.suptitle("Residual structure: echo delays + reference slot-to-slot coupling")

    # ---- probe 1: delay spectrum of the smooth-subtracted sample H ----
    ax = axes[0][0]
    print("=== probe 1: sample-H residual structure (in 0.4-2.4 THz) ===")
    for name, color in zip(CHANNEL_NAMES, ("C0", "C1", "C3", "C2")):
        freq, H, mask = sample_H[name]
        band = mask & (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1]) & np.isfinite(H)
        residual = smooth_residual(freq[band], H[band])
        rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
        rel = rms / float(np.median(np.abs(H[band])))
        print(f"  {name:6s}: residual rms {rms:.4f} ({rel*100:.1f}% of |H|)")
        delay_ps, amplitude = delay_spectrum(freq[band], residual)
        positive = delay_ps >= 0
        ax.plot(delay_ps[positive], amplitude[positive], color=color, lw=1.2, label=name)
    for marker, label in ((2.0, "window half-width 2 ps"), (5.3, "GaP echo 5.3 ps")):
        ax.axvline(marker, color="0.7", ls=":", lw=1.0)
        ax.text(marker, ax.get_ylim()[1] * 0.9, label, rotation=90, fontsize=7, va="top")
    ax.set_xlim(0, 12); ax.set_xlabel("delay [ps]"); ax.set_ylabel("|residual| amplitude")
    ax.set_title("sample H: residual delay spectrum", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- probe 2: reference W cross-ratios ----
    pairs = [("0-90", "0-0", "W_SP / W_SS (slot 90 vs 0, S-pol)"),
             ("90-0", "90-90", "W_PS / W_PP (slot 0 vs 90, P-pol)")]
    ax_mag, ax_phase = axes[0][1], axes[1][0]
    print("\n=== probe 2: reference slot-to-slot W ratios (same window!) ===")
    for (name_a, name_b, label), color in zip(pairs, ("C0", "C3")):
        freq_a, W_a = reference_W[name_a]
        freq_b, W_b = reference_W[name_b]
        band = (freq_a >= BAND_HZ[0]) & (freq_a <= BAND_HZ[1])
        ratio = W_a[band] / W_b[band]
        freq_thz = freq_a[band] * 1e-12
        magnitude_structure = float(np.nanstd(np.abs(ratio)))
        phase = np.unwrap(np.angle(ratio))
        slope, intercept = np.polyfit(freq_a[band], phase, 1)
        phase_structure = float(np.nanstd(phase - (slope * freq_a[band] + intercept)))
        print(f"  {label}: |ratio| med {np.nanmedian(np.abs(ratio)):.3f} "
              f"std {magnitude_structure:.3f}; phase slope {slope/(2*np.pi)*1e15:+.1f} fs, "
              f"structure {phase_structure:.3f} rad")
        ax_mag.plot(freq_thz, np.abs(ratio), color=color, lw=1.3, label=label)
        ax_phase.plot(freq_thz, phase - (slope * freq_a[band] + intercept), color=color,
                      lw=1.3, label=label)
    ax_mag.axhline(1.0, color="0.7", ls=":")
    ax_mag.set_xlabel("THz"); ax_mag.set_ylabel("|W ratio|")
    ax_mag.set_title("reference slot-to-slot |W| ratio (ideal = 1)", fontsize=10)
    ax_phase.set_xlabel("THz"); ax_phase.set_ylabel("phase residual [rad]")
    ax_phase.set_title("reference slot-to-slot W phase (detrended)", fontsize=10)
    for ax in (ax_mag, ax_phase):
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ---- |W| of each reference vs frequency (window loss + coupling shape) ----
    ax = axes[1][1]
    for name, color in zip(CHANNEL_NAMES, ("C0", "C1", "C3", "C2")):
        freq, W = reference_W[name]
        band = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
        ax.plot(freq[band] * 1e-12, np.abs(W[band]), color=color, lw=1.2,
                label=f"reference {name}")
    ax.set_xlabel("THz"); ax.set_ylabel("|W| = |Y2/Y1|")
    ax.set_title("bare-window |W| per reference (same window, 4 mounts)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "residual_structure_analysis.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
