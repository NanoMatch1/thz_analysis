"""CNT-21 p-pol de-embed chase (Samuel: "p-pol alone looks really good, but affected by the
air gap at low frequency").

Per channel (p-pol heroes + s-pol contrast):
  1. algebraic window->air extraction  x = (r_meas - r_front)/(1 - r_front r_meas);
  2. phase-excess gap estimates (Hilbert + MEM engines);
  3. single-pol Drude+gap forward fit IN r-SPACE (fits the Fabry-Perot directly - no
     closed-form inversion, so no r=-1 pole and no p-pol root ambiguity) with both the
     default and widened parameter bounds;
  4. de-embedded closed-form n,k (air frame at the gap angle) at the MEM and fitted gaps;
  5. sigma_1, sigma_2 from the de-embedded n,k and from the fitted Drude model.

Run from repo root:
  PYTHONPATH=".;./explorations/air_gap_cnt_reflection" ./.venv/Scripts/python.exe \
      explorations/air_gap_cnt_reflection/chase_cnt_p_deembed.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from thz_core.thz_core.reflection_gap import (
    fit_reflection_gap_dual_pol, drude_material_model, DRUDE_PARAM_BOUNDS,
)
from explore_air_gap_deembedding import deembed_gap_layer, remove_round_trip_phase
from deembed_air_gap_iterative import estimate_gap_minimum_phase

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
THETA_GAP = np.deg2rad(45.0)
FIT_BAND_HZ = (0.4e12, 2.2e12)
CROP_BAND_HZ = (0.15e12, 3.0e12)
EPS0 = 8.8541878128e-12

CHANNELS = ["90-0", "90-0R", "90-90", "0-0", "0-90"]

WIDE_BOUNDS = ((1.0, 0.5, 0.2), (60.0, 300.0, 100.0))   # eps_inf, plasma_thz, damping_thz
WIDE_INITIAL = (5.0, 30.0, 10.0)


def conductivity_from_nk(freq, n_values, k_values, eps_background=1.0):
    """sigma = -i omega eps0 (eps - eps_background), eps = (n - i k)^2. Returns S/cm."""
    eps = (n_values - 1j * k_values) ** 2
    sigma = -1j * 2 * np.pi * freq * EPS0 * (eps - eps_background)
    return sigma * 1e-2   # S/m -> S/cm


def run_channel(channel):
    data = np.load(os.path.join(RESULTS_DIR, f"channel_{channel}.npz"))
    freq = data["freq"]; mask = data["mask"].astype(bool)
    r_meas = data["reflection_r"]; r_front = complex(data["r_reference"])
    polarization = str(data["polarization"])

    crop = (freq >= CROP_BAND_HZ[0]) & (freq <= CROP_BAND_HZ[1])
    freq_c = freq[crop]; mask_c = mask[crop]; r_c = r_meas[crop]
    x_c = deembed_gap_layer(r_c, r_front)

    print(f"\n===== {channel} (pol {polarization}, r_front {r_front:+.3f}) =====")

    # 1. phase-excess estimators
    gap_estimates = {}
    for engine in ("hilbert", "mem"):
        d_est, _ = estimate_gap_minimum_phase(
            freq_c, x_c, mask_c, THETA_GAP, FIT_BAND_HZ, phase_engine=engine)
        gap_estimates[engine] = d_est
        print(f"  phase-excess ({engine:7s}): d = {d_est*1e6:+7.2f} um")

    # 2. single-pol Drude+gap forward fits (r-space, window frame)
    fits = {}
    for label, bounds, initial in (("default", DRUDE_PARAM_BOUNDS, (2.0, 4.0, 2.0)),
                                   ("wide", WIDE_BOUNDS, WIDE_INITIAL)):
        fit = fit_reflection_gap_dual_pol(
            freq_c, mask_c, {polarization: r_c}, {polarization: r_front}, THETA_GAP,
            initial_material_params=initial, material_param_bounds=bounds,
            initial_gap_um=2.0, gap_bounds_um=(0.0, 60.0))
        fits[label] = fit
        railed = [name for name, (low, high) in
                  zip(("eps_inf", "plasma_thz", "damping_thz"),
                      zip(bounds[0], bounds[1]))
                  if fit[name] <= low * 1.001 or fit[name] >= high * 0.999]
        print(f"  Drude+gap fit ({label:7s}): eps_inf={fit['eps_inf']:6.2f} "
              f"plasma={fit['plasma_thz']:7.2f} THz damping={fit['damping_thz']:6.2f} THz "
              f"gap={fit['gap_um']:5.2f} um  rms={fit['residual_rms']:.4f} "
              f"railed={railed if railed else 'none'}")

    # 3. de-embedded closed-form n,k at candidate gaps
    curves = {}
    for label, d_value in (("mem", gap_estimates["mem"]),
                           ("fit-wide", fits["wide"]["gap_um"] * 1e-6)):
        x_shift = remove_round_trip_phase(x_c, freq_c, d_value, THETA_GAP)
        n_de, k_de, _ = core.invert_nk_reflection(
            freq_c, x_shift, mask_c, theta_rad=THETA_GAP, n_incident=1.0,
            polarization=polarization)
        curves[label] = dict(d=d_value, n=n_de, k=k_de,
                             sigma=conductivity_from_nk(freq_c, n_de, k_de))
    return dict(channel=channel, freq=freq, mask=mask, freq_c=freq_c, mask_c=mask_c,
                n_naive=data["n"], k_naive=data["k"], gap_estimates=gap_estimates,
                fits=fits, curves=curves, polarization=polarization)


def main():
    results = [run_channel(channel) for channel in CHANNELS]

    figure, axes = plt.subplots(len(results), 3, figsize=(16, 3.9 * len(results)),
                                layout="constrained", squeeze=False)
    figure.suptitle("CNT-21 de-embed chase: naive vs MEM-gap vs Drude-fit gap "
                    "(air-frame closed-form inversion + fitted Drude model)")
    for row, result in enumerate(results):
        freq_thz = result["freq"] * 1e-12
        freq_c_thz = result["freq_c"] * 1e-12
        band = result["mask"] & (freq_thz >= 0.3) & (freq_thz <= 2.4)
        band_c = result["mask_c"] & (freq_c_thz >= 0.3) & (freq_c_thz <= 2.4)
        fit = result["fits"]["wide"]

        ax_n, ax_k, ax_s = axes[row]
        ax_n.plot(freq_thz[band], result["n_naive"][band], "0.6", ls="--", lw=1.3,
                  label="naive (window)")
        ax_k.plot(freq_thz[band], result["k_naive"][band], "0.6", ls="--", lw=1.3,
                  label="naive (window)")
        for label, color in (("mem", "C0"), ("fit-wide", "C2")):
            curve = result["curves"][label]
            ax_n.plot(freq_c_thz[band_c], curve["n"][band_c], color=color, lw=1.4,
                      label=f"{label} d={curve['d']*1e6:+.1f}um")
            ax_k.plot(freq_c_thz[band_c], curve["k"][band_c], color=color, lw=1.4,
                      label=label)
            ax_s.plot(freq_c_thz[band_c], curve["sigma"][band_c].real, color=color, lw=1.4,
                      label=f"{label} sigma1")
        # fitted Drude model curves
        n_model = fit["n"]; k_model = fit["k"]
        sigma_model = conductivity_from_nk(result["freq_c"], n_model, k_model)
        ax_n.plot(freq_c_thz[band_c], n_model[band_c], "C3", ls=":", lw=1.8,
                  label=f"Drude fit (d={fit['gap_um']:.1f}um)")
        ax_k.plot(freq_c_thz[band_c], k_model[band_c], "C3", ls=":", lw=1.8, label="Drude fit")
        ax_s.plot(freq_c_thz[band_c], sigma_model[band_c].real, "C3", ls=":", lw=1.8,
                  label="Drude fit sigma1")

        channel = result["channel"]
        ax_n.set_title(f"{channel}: n", fontsize=10); ax_n.set_ylim(-1, 25)
        ax_k.set_title(f"{channel}: k", fontsize=10)
        ax_s.set_title(f"{channel}: sigma1 [S/cm]", fontsize=10)
        for ax in (ax_n, ax_k, ax_s):
            ax.set_xlabel("THz"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    out = os.path.join(RESULTS_DIR, "chase_cnt_p_deembed.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
