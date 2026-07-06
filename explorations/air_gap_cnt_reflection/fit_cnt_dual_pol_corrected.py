"""Dual-pol joint Drude+gap fits on the CORRECTLY PAIRED CNT-21 channels (task: does the
corrected pairing + wide bounds fix the F23 railing?), plus gap-pinned variants.

Fits per material axis (one n(omega) + one shared gap d to s AND p together):
    n_parallel : 0-0 (S) + 90-90 (P)
    n_perp     : 0-90 (S) + 90-0 (P)

Variants per axis:
  A. joint, gap FREE, default bounds        (the F23 configuration, corrected pairing)
  B. joint, gap FREE, wide bounds
  C. joint, gap FIXED at the phase-excess estimate (MEM, Si-systematic-corrected)
  D. per-pol single fits at the same fixed gap (consistency check: do s and p agree?)

sigma_1(0.5 THz) and the Drude sigma_dc are reported per fit (sign convention via
core.derive_eps_sigma).

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/fit_cnt_dual_pol_corrected.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from thz_core.thz_core.reflection_gap import (
    fit_reflection_gap_dual_pol, DRUDE_PARAM_BOUNDS,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
THETA_GAP = np.deg2rad(45.0)
CROP_BAND_HZ = (0.15e12, 3.0e12)
EPS0 = 8.8541878128e-12

WIDE_BOUNDS = ((1.0, 0.5, 0.2), (60.0, 300.0, 100.0))
WIDE_INITIAL = (5.0, 30.0, 10.0)

# phase-excess MEM estimates ~5.3 um; the Si-p control in the same slot showed a +2.4 um
# systematic (oracle 0), so the physical gap estimate is ~2.9 um. Test both.
GAP_CANDIDATES_UM = {"mem_raw": 5.3, "mem_si_corrected": 2.9}

AXES = {
    "n_parallel": ("0-0", "90-90"),
    "n_perp": ("0-90", "90-0"),
}


def load_channel(channel):
    data = np.load(os.path.join(RESULTS_DIR, f"channel_{channel}.npz"))
    freq = data["freq"]; crop = (freq >= CROP_BAND_HZ[0]) & (freq <= CROP_BAND_HZ[1])
    return dict(freq=freq[crop], mask=data["mask"].astype(bool)[crop],
                r=data["reflection_r"][crop], r_front=complex(data["r_reference"]),
                polarization=str(data["polarization"]))


def sigma_dc_s_per_cm(plasma_thz, damping_thz):
    """Drude sigma_dc = eps0 * omega_p^2 / gamma (angular), in S/cm."""
    omega_p = 2 * np.pi * plasma_thz * 1e12
    gamma = 2 * np.pi * damping_thz * 1e12
    return EPS0 * omega_p**2 / gamma * 1e-2


def report_fit(label, fit, freq, mask):
    sigma_dc = sigma_dc_s_per_cm(fit["plasma_thz"], fit["damping_thz"])
    eps, sigma, _ = core.derive_eps_sigma(freq, fit["n"], fit["k"], {"derive": {}})
    band = mask & (freq >= 0.4e12) & (freq <= 0.6e12)
    sigma_05 = float(np.nanmedian(sigma[band].real)) * 1e-2 if band.any() else np.nan
    print(f"    {label:34s}: eps_inf={fit['eps_inf']:6.2f} plasma={fit['plasma_thz']:7.2f} "
          f"damping={fit['damping_thz']:6.2f} gap={fit['gap_um']:5.2f}um "
          f"rms={fit['residual_rms']:.4f} sigma_dc={sigma_dc:8.1f} S/cm "
          f"sigma1(0.5THz)={sigma_05:8.1f} S/cm")
    return fit


def main():
    channels = {name: load_channel(name) for name in ("0-0", "0-90", "90-0", "90-90")}

    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    figure.suptitle("CNT-21 dual-pol joint Drude+gap fits (corrected pairing)")

    for row, (axis_label, (s_channel, p_channel)) in enumerate(AXES.items()):
        s_data = channels[s_channel]; p_data = channels[p_channel]
        freq = s_data["freq"]
        mask = s_data["mask"] & p_data["mask"]
        measured = {"s": s_data["r"], "p": p_data["r"]}
        front = {"s": s_data["r_front"], "p": p_data["r_front"]}

        print(f"\n===== axis {axis_label}  (s={s_channel}, p={p_channel}) =====")

        fits = {}
        fits["A joint free default"] = report_fit(
            "A joint, free gap, default bounds",
            fit_reflection_gap_dual_pol(freq, mask, measured, front, THETA_GAP,
                                        initial_gap_um=2.0, gap_bounds_um=(0.0, 60.0)),
            freq, mask)
        fits["B joint free wide"] = report_fit(
            "B joint, free gap, wide bounds",
            fit_reflection_gap_dual_pol(freq, mask, measured, front, THETA_GAP,
                                        initial_material_params=WIDE_INITIAL,
                                        material_param_bounds=WIDE_BOUNDS,
                                        initial_gap_um=2.0, gap_bounds_um=(0.0, 60.0)),
            freq, mask)
        for gap_label, gap_um in GAP_CANDIDATES_UM.items():
            fits[f"C joint fixed {gap_label}"] = report_fit(
                f"C joint, gap fixed {gap_um:.1f}um ({gap_label})",
                fit_reflection_gap_dual_pol(freq, mask, measured, front, THETA_GAP,
                                            initial_material_params=WIDE_INITIAL,
                                            material_param_bounds=WIDE_BOUNDS,
                                            fixed_gap_um=gap_um),
                freq, mask)
        # D. per-pol single fits at the Si-corrected gap — s/p consistency check
        for polarization, channel_data in (("s", s_data), ("p", p_data)):
            fits[f"D single {polarization}"] = report_fit(
                f"D single {polarization}, gap fixed 2.9um",
                fit_reflection_gap_dual_pol(freq, channel_data["mask"],
                                            {polarization: channel_data["r"]},
                                            {polarization: channel_data["r_front"]},
                                            THETA_GAP,
                                            initial_material_params=WIDE_INITIAL,
                                            material_param_bounds=WIDE_BOUNDS,
                                            fixed_gap_um=2.9),
                freq, channel_data["mask"])

        freq_thz = freq * 1e-12
        band = mask & (freq_thz >= 0.3) & (freq_thz <= 2.4)
        ax_n, ax_k, ax_s = axes_grid[row]
        colors = {"B joint free wide": "C0", "C joint fixed mem_si_corrected": "C2",
                  "D single s": "C1", "D single p": "C3"}
        for key, color in colors.items():
            fit = fits[key]
            eps, sigma, _ = core.derive_eps_sigma(freq, fit["n"], fit["k"], {"derive": {}})
            ax_n.plot(freq_thz[band], fit["n"][band], color=color, lw=1.5, label=key)
            ax_k.plot(freq_thz[band], fit["k"][band], color=color, lw=1.5, label=key)
            ax_s.plot(freq_thz[band], sigma[band].real * 1e-2, color=color, lw=1.5, label=key)
        ax_n.set_title(f"{axis_label}: n", fontsize=10)
        ax_k.set_title(f"{axis_label}: k", fontsize=10)
        ax_s.set_title(f"{axis_label}: sigma1 [S/cm]", fontsize=10)
        for ax in (ax_n, ax_k, ax_s):
            ax.set_xlabel("THz"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "fit_cnt_dual_pol_corrected.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
