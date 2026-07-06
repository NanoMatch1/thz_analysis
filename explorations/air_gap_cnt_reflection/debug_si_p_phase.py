"""Look at the actual Si-p phase structure: wrapped arg(H), Re/Im of H, and where the winding
comes from. Also compare against the Si-s-old control and one CNT p channel.

Run from repo root:
  PYTHONPATH=. ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/debug_si_p_phase.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
BAND_THZ = (0.3, 2.5)

CHANNELS = ["Si-p", "Si-s-old", "90-0", "0-0"]


def main():
    figure, axes = plt.subplots(len(CHANNELS), 3, figsize=(15, 3.6 * len(CHANNELS)),
                                layout="constrained")
    figure.suptitle("phase structure: wrapped arg(H), unwrapped arg(H), H trajectory")

    for row, channel in enumerate(CHANNELS):
        data = np.load(os.path.join(RESULTS_DIR, f"channel_{channel}.npz"))
        freq = data["freq"]; mask = data["mask"].astype(bool)
        H = data["transfer_H"]
        freq_thz = freq * 1e-12
        band = mask & (freq_thz >= BAND_THZ[0]) & (freq_thz <= BAND_THZ[1]) & np.isfinite(H)
        f_band = freq_thz[band]
        H_band = H[band]
        wrapped = np.angle(H_band)
        unwrapped = np.unwrap(wrapped)

        # per-channel numbers
        slope, intercept = np.polyfit(f_band * 1e12, unwrapped, 1)
        print(f"{channel:10s}: bins {band.sum():4d}, |H| med {np.nanmedian(np.abs(H_band)):.3f}, "
              f"unwrapped span {unwrapped[-1]-unwrapped[0]:+8.1f} rad, "
              f"slope {slope/(2*np.pi)*1e15:+9.1f} fs, intercept {intercept:+8.2f} rad")
        # contiguity check: are the band bins contiguous in the raw grid?
        indices = np.flatnonzero(band)
        n_gaps = int(np.sum(np.diff(indices) > 1))
        largest_gap = int(np.max(np.diff(indices))) if len(indices) > 1 else 0
        print(f"{'':10s}  mask gaps in band: {n_gaps} (largest {largest_gap} bins)")

        ax = axes[row][0]
        ax.plot(f_band, wrapped, ".", ms=2.5)
        ax.set_title(f"{channel}: wrapped arg(H)", fontsize=10)
        ax.set_xlabel("THz"); ax.set_ylabel("rad"); ax.grid(alpha=0.3)
        ax.set_ylim(-3.4, 3.4)

        ax = axes[row][1]
        ax.plot(f_band, unwrapped, ".", ms=2.5)
        ax.set_title(f"{channel}: unwrapped arg(H)", fontsize=10)
        ax.set_xlabel("THz"); ax.set_ylabel("rad"); ax.grid(alpha=0.3)

        ax = axes[row][2]
        scatter = ax.scatter(H_band.real, H_band.imag, c=f_band, s=4, cmap="viridis")
        ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
        ax.set_title(f"{channel}: H trajectory (colour = THz)", fontsize=10)
        ax.set_xlabel("Re H"); ax.set_ylabel("Im H"); ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        figure.colorbar(scatter, ax=ax, shrink=0.8)

    out = os.path.join(RESULTS_DIR, "debug_si_p_phase.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
