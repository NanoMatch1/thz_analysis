"""Debug the Si-p channel: why does the p-pol window inversion give n~0.7-0.9 not 3.418?

Checks, on the saved channel arrays (process_cnt21_channels.py output):
  1. sign/convention audit — what does thz_core.fresnel_reflection_p(1.96 -> 3.418, theta_int)
     predict for r_back and for the SiO2->air reference, and what H does that imply?
  2. measured vs predicted: |H|, arg(H), |r|, arg(r) across the trusted band;
  3. gap model: does a small SiO2|air|Si etalon explain the measured r better?
  4. root-picker behaviour: feed the PREDICTED r (no gap, n=3.418) into _invert_p_pol_index
     and confirm it round-trips; then feed the measured r and look at both quadratic roots.

Run from repo root:
  PYTHONPATH=. ./.venv/Scripts/python.exe explorations/air_gap_cnt_reflection/debug_si_p_inversion.py
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thz_core.thz_core as core
from thz_core.thz_core.invert import _invert_p_pol_index
from thz_core.thz_core.multilayer import fresnel_reflection_p, fresnel_reflection_s

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "cnt21_channel_results")
N_SIO2 = 1.96
N_SILICON = 3.418
THETA_EXTERNAL = np.deg2rad(45.0)
THETA_INTERNAL = np.arcsin(np.sin(THETA_EXTERNAL) / N_SIO2)
THETA_GAP = np.arcsin(N_SIO2 * np.sin(THETA_INTERNAL) / 1.0)   # = 45 deg (air gap)
BAND_THZ = (0.4, 2.2)


def gap_reflection(r_front_gap, r_back_gap, frequency_hz, gap_m, theta_gap):
    """SiO2 | air(d) | medium Fabry-Perot reflection seen from inside the SiO2."""
    beta = 2.0 * np.pi * frequency_hz / 299_792_458.0 * gap_m * np.cos(theta_gap)
    phase = np.exp(-2j * beta)
    return (r_front_gap + r_back_gap * phase) / (1.0 + r_front_gap * r_back_gap * phase)


def main():
    data = np.load(os.path.join(RESULTS_DIR, "channel_Si-p.npz"))
    freq = data["freq"]; mask = data["mask"].astype(bool)
    H_meas = data["transfer_H"]; r_meas = data["reflection_r"]
    r_reference_stored = complex(data["r_reference"])

    print(f"theta_internal = {np.rad2deg(THETA_INTERNAL):.2f} deg, "
          f"theta_gap = {np.rad2deg(THETA_GAP):.2f} deg")

    # --- 1. convention audit ------------------------------------------------
    r_back_expected = complex(fresnel_reflection_p(N_SIO2, N_SILICON, THETA_INTERNAL))
    r_ref_expected = complex(fresnel_reflection_p(N_SIO2, 1.0, THETA_INTERNAL))
    r_ref_expected_s = complex(fresnel_reflection_s(N_SIO2, 1.0, THETA_INTERNAL))
    print("\n--- thz_core Fresnel conventions (internal angle) ---")
    print(f"  r_p(SiO2->Si)   = {r_back_expected:+.4f}   (expected sample back reflection)")
    print(f"  r_p(SiO2->air)  = {r_ref_expected:+.4f}   (window reference back reflection)")
    print(f"  r_s(SiO2->air)  = {r_ref_expected_s:+.4f}   (s-pol, for comparison)")
    print(f"  stored pipeline r_reference = {r_reference_stored:+.4f}")
    H_expected = r_back_expected / r_ref_expected
    print(f"  => expected H (no gap) = {H_expected:+.4f}  (|H|={abs(H_expected):.3f}, "
          f"arg={np.angle(H_expected):+.3f} rad)")

    # --- 2. measured vs predicted -------------------------------------------
    freq_thz = freq * 1e-12
    band = mask & (freq_thz >= BAND_THZ[0]) & (freq_thz <= BAND_THZ[1]) & np.isfinite(H_meas)
    print("\n--- measured (median over band) ---")
    print(f"  |H|    = {np.nanmedian(np.abs(H_meas[band])):.3f}   "
          f"arg(H) = {np.nanmedian(np.angle(H_meas[band])):+.3f} rad")
    print(f"  |r|    = {np.nanmedian(np.abs(r_meas[band])):.3f}   "
          f"arg(r) = {np.nanmedian(np.angle(r_meas[band])):+.3f} rad "
          f"(expected r_back {r_back_expected:+.3f})")
    print(f"  |1+r|  = {np.nanmedian(np.abs(1 + r_meas[band])):.3f}")

    # phase slope of H across the band = timing/gap signature
    phase_unwrapped = np.unwrap(np.angle(H_meas[band]))
    slope, intercept = np.polyfit(freq[band], phase_unwrapped, 1)
    print(f"  arg(H) linear fit: slope {slope/(2*np.pi)*1e15:+.1f} fs, "
          f"intercept {intercept:+.3f} rad")

    # --- 3. gap model comparison ---------------------------------------------
    r_front_gap = complex(fresnel_reflection_p(N_SIO2, 1.0, THETA_INTERNAL))   # SiO2->air
    r_back_gap = complex(fresnel_reflection_p(1.0, N_SILICON, THETA_GAP))      # air->Si at 45
    print("\n--- gap etalon candidates: rms |r_model - r_meas| over band ---")
    print(f"  r_p(air->Si at 45deg) = {r_back_gap:+.4f}")
    for gap_um in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        r_model = gap_reflection(r_front_gap, r_back_gap, freq, gap_um * 1e-6, THETA_GAP)
        rms = float(np.sqrt(np.nanmean(np.abs(r_model[band] - r_meas[band]) ** 2)))
        print(f"  d = {gap_um:5.1f} um: rms = {rms:.4f}")

    # --- 4. root-picker round trip -------------------------------------------
    print("\n--- root-picker sanity ---")
    r_synthetic = np.full(8, r_back_expected, dtype=complex)
    n_round = _invert_p_pol_index(r_synthetic, N_SIO2, THETA_INTERNAL)
    print(f"  synthetic r (no gap, Si): inverted N2 = {n_round[0]:.4f} "
          f"(expect {N_SILICON:.3f}+0j)")

    n_from_meas = _invert_p_pol_index(r_meas[band], N_SIO2, THETA_INTERNAL)
    print(f"  measured r -> N2: median n = {np.nanmedian(n_from_meas.real):.3f}, "
          f"median k = {np.nanmedian(-n_from_meas.imag):.3f}")

    # both quadratic roots explicitly, on the measured r
    cos_t = np.cos(THETA_INTERNAL); m = (1 - r_meas[band]) / (1 + r_meas[band])
    denominator = 2 * (m * cos_t) ** 2
    disc = N_SIO2**4 * (1 - m**2 * np.sin(2 * THETA_INTERNAL) ** 2)
    sq = np.sqrt(disc)
    for label, u in (("root+", (N_SIO2**2 + sq) / denominator),
                     ("root-", (N_SIO2**2 - sq) / denominator)):
        N2 = np.sqrt(u)
        N2 = np.where(N2.real < 0, -N2, N2)
        print(f"  {label}: median n = {np.nanmedian(N2.real):.3f}, "
              f"median k(=-Im) = {np.nanmedian(-N2.imag):.3f}")

    # --- figure ---------------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    figure.suptitle("Si-p channel: measured vs modelled reflection (window, p-pol)")
    ax = axes[0]
    ax.plot(freq_thz[band], np.abs(H_meas[band]), "C0", label="|H| measured")
    ax.axhline(abs(H_expected), color="0.5", ls="--", label=f"|H| no-gap ({abs(H_expected):.2f})")
    ax.set_xlabel("THz"); ax.set_ylabel("|H|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(freq_thz[band], np.unwrap(np.angle(H_meas[band])), "C0", label="arg(H) measured")
    ax.axhline(np.angle(H_expected), color="0.5", ls="--", label="arg(H) no-gap")
    ax.set_xlabel("THz"); ax.set_ylabel("arg(H) [rad]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[2]
    for gap_um, color in ((0.0, "0.5"), (5.0, "C2"), (20.0, "C3")):
        r_model = gap_reflection(r_front_gap, r_back_gap, freq, gap_um * 1e-6, THETA_GAP)
        ax.plot(freq_thz[band], np.abs(r_model[band]), color=color, ls="--",
                label=f"|r| model d={gap_um:.0f}um")
    ax.plot(freq_thz[band], np.abs(r_meas[band]), "C0", label="|r| measured")
    ax.set_xlabel("THz"); ax.set_ylabel("|r|"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    out = os.path.join(RESULTS_DIR, "debug_si_p_inversion.png")
    figure.savefig(out, dpi=130)
    print(f"\nSaved figure: {out}")


if __name__ == "__main__":
    main()
