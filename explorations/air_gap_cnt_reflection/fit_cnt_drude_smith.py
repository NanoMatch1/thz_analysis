"""Drude-Smith fits of the CNT-21 channels — the realignment repeat showed the data is
reproducible to ~0.5% in r-space while the plain-Drude+gap fit leaves 10-13% residual, i.e.
the MODEL is the limiter, not the measurement. CNT buckypaper is a canonical Drude-Smith
material (backscattering parameter c ~ -0.7..-0.9 suppresses low-frequency conductivity), so
try sigma_DS before blaming the geometry.

Model: eps(omega) = eps_inf + i sigma(omega) / (eps0 omega)   [thz_core e^{-iwt} convention
handled by matching drude_index's sign],   sigma_DS = (eps0 wp^2 tau / (1 - i w tau)) *
(1 + c / (1 - i w tau)).

Params: [eps_inf, plasma_thz, damping_thz, c_backscatter].

Variants per channel/axis: single-pol fits (s, p) and the dual-pol joint fit, gap free and
gap fixed at 2.9 um (phase-excess MEM, Si-systematic-corrected).

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/fit_cnt_drude_smith.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from thz_core.thz_core.reflection_gap import fit_reflection_gap_dual_pol

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
THETA_GAP = np.deg2rad(45.0)
CROP_BAND_HZ = (0.15e12, 3.0e12)
EPS0 = 8.8541878128e-12

DS_PARAM_NAMES = ("eps_inf", "plasma_thz", "damping_thz", "c_backscatter")
DS_INITIAL = (5.0, 30.0, 10.0, -0.7)
DS_BOUNDS = ((1.0, 0.5, 0.2, -1.0), (60.0, 300.0, 100.0, 0.0))

AXES = {
    "n_parallel": ("0-0", "90-90"),
    "n_perp": ("0-90", "90-0"),
}


def drude_smith_index(frequency_hz, eps_inf, plasma_thz, damping_thz, c_backscatter):
    """Complex index n - i k from the Drude-Smith permittivity (thz_core convention).

    eps(omega) = eps_inf - (wp^2 / (omega^2 + i gamma omega)) * (1 + c * gamma/(gamma - i omega))
    reduces to the plain Drude used in reflection_gap.drude_index at c = 0 (verified below).
    """
    omega = 2.0 * np.pi * np.asarray(frequency_hz, dtype=float)
    plasma = 2.0 * np.pi * plasma_thz * 1e12
    damping = 2.0 * np.pi * damping_thz * 1e12
    smith_factor = 1.0 + c_backscatter * damping / (damping - 1j * omega)
    eps = eps_inf - plasma**2 / (omega**2 + 1j * damping * omega) * smith_factor
    return np.conj(np.sqrt(eps))


def drude_smith_material_model(frequency_hz, params):
    eps_inf, plasma_thz, damping_thz, c_backscatter = params
    return drude_smith_index(frequency_hz, eps_inf, plasma_thz, damping_thz, c_backscatter)


def load_channel(channel):
    data = np.load(os.path.join(RESULTS_DIR, f"channel_{channel}.npz"))
    freq = data["freq"]; crop = (freq >= CROP_BAND_HZ[0]) & (freq <= CROP_BAND_HZ[1])
    return dict(freq=freq[crop], mask=data["mask"].astype(bool)[crop],
                r=data["reflection_r"][crop], r_front=complex(data["r_reference"]),
                polarization=str(data["polarization"]))


def run_fit(label, freq, mask, measured, front, fixed_gap_um=None):
    fit = fit_reflection_gap_dual_pol(
        freq, mask, measured, front, THETA_GAP,
        material_model=drude_smith_material_model,
        initial_material_params=DS_INITIAL,
        material_param_bounds=DS_BOUNDS,
        material_param_names=DS_PARAM_NAMES,
        fixed_gap_um=fixed_gap_um,
        initial_gap_um=2.0, gap_bounds_um=(0.0, 60.0))
    railed = [name for name, low, high in zip(DS_PARAM_NAMES, DS_BOUNDS[0], DS_BOUNDS[1])
              if fit[name] <= low + abs(low) * 1e-3 + 1e-9
              or fit[name] >= high - abs(high) * 1e-3]
    eps, sigma, _ = core.derive_eps_sigma(freq, fit["n"], fit["k"], {"derive": {}})
    band_05 = mask & (freq >= 0.4e12) & (freq <= 0.6e12)
    sigma_05 = float(np.nanmedian(sigma[band_05].real)) * 1e-2 if band_05.any() else np.nan
    print(f"    {label:30s}: eps_inf={fit['eps_inf']:6.2f} plasma={fit['plasma_thz']:7.2f} "
          f"damping={fit['damping_thz']:6.2f} c={fit['c_backscatter']:+.3f} "
          f"gap={fit['gap_um']:5.2f}um rms={fit['residual_rms']:.4f} "
          f"sigma1(0.5)={sigma_05:8.1f} S/cm railed={railed if railed else 'none'}")
    fit["sigma"] = sigma
    return fit


def main():
    # sanity: c=0 reproduces the plain Drude of reflection_gap
    from thz_core.thz_core.reflection_gap import drude_index
    freq_test = np.linspace(0.2e12, 3e12, 50)
    difference = np.max(np.abs(drude_smith_index(freq_test, 4.0, 30.0, 5.0, 0.0)
                               - drude_index(freq_test, 4.0, 30.0, 5.0)))
    print(f"[sanity] Drude-Smith(c=0) vs Drude max diff: {difference:.2e}")

    channels = {name: load_channel(name) for name in ("0-0", "0-90", "90-0", "90-90")}

    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    figure.suptitle("CNT-21 Drude-Smith fits (r-space, corrected pairing)")

    for row, (axis_label, (s_channel, p_channel)) in enumerate(AXES.items()):
        s_data = channels[s_channel]; p_data = channels[p_channel]
        freq = s_data["freq"]
        joint_mask = s_data["mask"] & p_data["mask"]

        print(f"\n===== axis {axis_label} (s={s_channel}, p={p_channel}) =====")
        fits = {}
        fits["p free"] = run_fit(f"single p ({p_channel}), free gap", freq, p_data["mask"],
                                 {"p": p_data["r"]}, {"p": p_data["r_front"]})
        fits["s free"] = run_fit(f"single s ({s_channel}), free gap", freq, s_data["mask"],
                                 {"s": s_data["r"]}, {"s": s_data["r_front"]})
        fits["joint free"] = run_fit("joint s+p, free gap", freq, joint_mask,
                                     {"s": s_data["r"], "p": p_data["r"]},
                                     {"s": s_data["r_front"], "p": p_data["r_front"]})
        fits["joint 2.9"] = run_fit("joint s+p, gap fixed 2.9um", freq, joint_mask,
                                    {"s": s_data["r"], "p": p_data["r"]},
                                    {"s": s_data["r_front"], "p": p_data["r_front"]},
                                    fixed_gap_um=2.9)

        freq_thz = freq * 1e-12
        band = joint_mask & (freq_thz >= 0.3) & (freq_thz <= 2.4)
        ax_n, ax_k, ax_s = axes_grid[row]
        for key, color in (("p free", "C3"), ("s free", "C1"),
                           ("joint free", "C0"), ("joint 2.9", "C2")):
            fit = fits[key]
            ax_n.plot(freq_thz[band], fit["n"][band], color=color, lw=1.5,
                      label=f"{key} (gap {fit['gap_um']:.1f}um, c {fit['c_backscatter']:+.2f})")
            ax_k.plot(freq_thz[band], fit["k"][band], color=color, lw=1.5, label=key)
            ax_s.plot(freq_thz[band], fit["sigma"][band].real * 1e-2, color=color, lw=1.5,
                      label=key)
        ax_n.set_title(f"{axis_label}: n", fontsize=10)
        ax_k.set_title(f"{axis_label}: k", fontsize=10)
        ax_s.set_title(f"{axis_label}: sigma1 [S/cm]", fontsize=10)
        for ax in (ax_n, ax_k, ax_s):
            ax.set_xlabel("THz"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "fit_cnt_drude_smith.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
