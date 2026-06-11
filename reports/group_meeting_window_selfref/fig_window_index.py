"""Figure E: Window refractive index n_SiO₂(ω) from a single bare-window acquisition.

Uses characterise_window to invert the intra-trace ratio W = Y2/Y1 of the
reference measurement for the complex window index. Also overlays the measured
|W| against the forward model built from the extracted n — this is a
consistency check that the inversion is well-conditioned.

Run:
    .venv/Scripts/python.exe reports/group_meeting_window_selfref/fig_window_index.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import thz_core.thz_core as core
from dataset_core.adapters import thz_adapter as thz

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Users\Samuel\Data\THz\Sam\Analysis\CNT-13\D\segmented"

REFERENCE_FILENAME = "reference_a-45.acc"
THICKNESS_M = 0.9e-3
THETA_DEG = 45.0
BAND_THZ = (0.25, 2.75)


def main() -> None:
    first_ref = os.path.join(DATA_DIR, "first_reflection", REFERENCE_FILENAME)
    second_ref = os.path.join(DATA_DIR, "second_reflection", REFERENCE_FILENAME)

    for path in (first_ref, second_ref):
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}")
            return

    result = thz.characterise_window(
        first_ref,
        second_ref,
        thickness_m=THICKNESS_M,
        theta_deg=THETA_DEG,
        band_thz=BAND_THZ,
        show_graph=False,
    )

    freq_hz = result["freq"]
    f_thz = freq_hz * 1e-12
    mask = result["mask"]
    n_vals = result["n"]
    k_vals = result["k"]

    n_mean = float(np.nanmedian(n_vals[mask]))
    k_mean = float(np.nanmedian(k_vals[mask]))

    # Forward model using the extracted mean index for a visual consistency check.
    w_model = core.window_transfer_model(
        freq_hz,
        complex(n_mean, -k_mean),
        THICKNESS_M,
        np.deg2rad(THETA_DEG),
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    ax_n, ax_k, ax_w = axes

    ax_n.plot(f_thz[mask], n_vals[mask], lw=1.6, color="tab:blue")
    ax_n.axhline(n_mean, color="tab:blue", lw=0.8, ls=":", alpha=0.6)
    ax_n.axhline(1.96, color="0.55", lw=1.0, ls="--", label="fused silica nominal 1.96")
    ax_n.set_ylabel("n  (SiO₂)")
    ax_n.legend(fontsize=8)
    ax_n.grid(True, alpha=0.2)
    ax_n.set_title(
        f"Window characterisation — CNT-13/D bare reference  "
        f"(d = {THICKNESS_M * 1e3:.1f} mm, θ = {THETA_DEG:.0f}°)"
    )
    ax_n.text(
        0.98, 0.05,
        f"median n = {n_mean:.4f}",
        transform=ax_n.transAxes, ha="right", va="bottom", fontsize=8,
    )

    ax_k.plot(f_thz[mask], k_vals[mask], lw=1.6, color="tab:orange")
    ax_k.axhline(0.0, color="0.55", lw=0.8, ls="--")
    ax_k.set_ylabel("k  (SiO₂)")
    ax_k.grid(True, alpha=0.2)
    ax_k.text(
        0.98, 0.95,
        f"median k = {k_mean:.5f}",
        transform=ax_k.transAxes, ha="right", va="top", fontsize=8,
    )

    # |W| measured vs model
    ax_w.plot(f_thz[mask], np.abs(result["W"][mask]), lw=1.6, label="|W| measured", color="tab:green")
    ax_w.plot(
        f_thz,
        np.abs(w_model),
        lw=1.2,
        ls="--",
        color="tab:red",
        label=f"|W| model  (n = {n_mean:.3f})",
    )
    ax_w.set_ylabel("|W|  (second/first pulse amplitude ratio)")
    ax_w.set_xlabel("Frequency (THz)")
    ax_w.legend(fontsize=8)
    ax_w.grid(True, alpha=0.2)

    print(f"\nWindow index summary ({BAND_THZ[0]}–{BAND_THZ[1]} THz, {int(mask.sum())} bins)")
    print(f"  n  = {n_mean:.4f} ± {float(np.nanstd(n_vals[mask])):.4f}")
    print(f"  k  = {k_mean:.5f} ± {float(np.nanstd(k_vals[mask])):.5f}")
    print(f"  delay measured: {result['delay_s'] * 1e12:.3f} ps")

    out_path = os.path.join(OUTPUT_DIR, "fig_window_index.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
